"""Render a verified V15 RL rollout, with a full episode and slow-motion replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher_rl.improved_rl import file_hash, read_json, rl_fingerprint
from teacher_rl.improved_teacher import ImprovedRecipe
from teacher_rl.model import load_checkpoint
from teacher_rl.residual_env import QualityConfig, ResidualEnv, RL_SCHEMA


def collect(env, model, seed, expected):
    observation = env.reset(seed)
    records = []
    hook = env.plant.hook_after_step
    next_time = env.plant.sim_now

    def record(t):
        nonlocal next_time
        hook(t)
        if t + 1e-9 >= next_time:
            records.append((t - 2., env.plant.data.qpos.copy(),
                env.plant.data.qvel.copy(), env.phase(), env.launched))
            next_time += 1. / 60.

    env.plant.hook_after_step = record
    while not env.done:
        with torch.inference_mode():
            action = model.distribution(torch.as_tensor(observation[None])).mean.tanh()[0].numpy()
        observation, _, _, result = env.step(action)
    env.plant.hook_after_step = hook
    if env.plant.sim_now-2. > records[-1][0]+1e-9:
        records.append((env.plant.sim_now-2., env.plant.data.qpos.copy(),
            env.plant.data.qvel.copy(), env.phase(), env.launched))
    if not all(result[key] for key in ("captured", "released", "hit15")):
        raise RuntimeError("Selected RL episode did not complete catch and target hit")
    for key in ("landing_error_m", "capture_time", "release_time", "weld_peak_n"):
        if not np.isclose(result[key], expected[key], rtol=1e-8, atol=1e-8):
            raise RuntimeError(f"Replay differs from evaluation: {key}")
    data = dict(time=np.array([r[0] for r in records]),
        qpos=np.stack([r[1] for r in records]), qvel=np.stack([r[2] for r in records]),
        phase=np.array([r[3] for r in records]), launched=np.array([r[4] for r in records]))
    print(json.dumps(dict(stage="rollout_verified", samples=len(records), result=result)), flush=True)
    return data, result


def camera(focus, distance, azimuth, elevation):
    result = mujoco.MjvCamera()
    result.type = mujoco.mjtCamera.mjCAMERA_FREE
    result.lookat[:] = focus
    result.distance, result.azimuth, result.elevation = distance, azimuth, elevation
    return result


def draw_text(frame, text, xy, size=.65, color=(242, 245, 247)):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("teacher_runs/v15_improved/rl_v1_formal"))
    parser.add_argument("--seed", type=int, default=3000001)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--azimuth", type=float, default=125.)
    parser.add_argument("--elevation", type=float, default=-25.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.run / "ppo_best.pt"
    model, payload = load_checkpoint(checkpoint, expected_schema=RL_SCHEMA)
    model.eval()
    contract = payload["run_contract"]
    if contract["rl_source_hash"] != rl_fingerprint():
        raise RuntimeError("Checkpoint and current simulation source differ")
    evaluation_contract = read_json(args.run / "evaluation_200/evaluation_contract.json")
    digest = file_hash(checkpoint)
    if digest != evaluation_contract["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from independently evaluated model")
    row = next(r for r in read_json(args.run / "evaluation_200/rl_episodes.json") if r["seed"] == args.seed)
    if not row["result"]["hit15"]:
        raise ValueError("Select a successful evaluation seed")
    output = args.run / "videos" / f"v15_rl_best_catch_throw_seed_{args.seed}"
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.with_suffix(".npz")
    metadata_path = output.with_suffix(".json")
    env = ResidualEnv(row["scenario"], ImprovedRecipe(**contract["recipe"]),
        QualityConfig(**contract["quality"]))
    cache_current = (cache.exists() and metadata_path.exists() and
        read_json(metadata_path).get("renderer_version") == 2)
    if cache_current:
        metadata = read_json(metadata_path)
        if metadata["checkpoint_sha256"] != digest or metadata["rl_source_hash"] != rl_fingerprint():
            raise RuntimeError("Cached trajectory belongs to a different checkpoint or source")
        if metadata["scenario"] != row["scenario"]:
            raise RuntimeError("Cached scenario differs from evaluation")
        with np.load(cache) as archive:
            states = {key: archive[key] for key in archive.files}
        result = metadata["result"]
        env.reset(args.seed)
    else:
        states, result = collect(env, model, args.seed, row["result"])
        np.savez_compressed(cache, **states)
        metadata = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
            rl_source_hash=rl_fingerprint(), update=payload["update"], seed=args.seed,
            scenario=row["scenario"], result=result, verified_against_evaluation=True,
            capture_sample_hz=60, simulation_changes="none", renderer_version=2,
            note="Selected successful simulated episode, not an aggregate performance claim")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plant = env.plant
    mj_model = plant.model
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720
    render_data = mujoco.MjData(mj_model)
    ball = states["qpos"][:, plant.ball_qadr:plant.ball_qadr+3]
    valid_ball = ball[states["launched"]]
    body = np.delete(plant.data.xpos, plant.ball_bid, axis=0)
    points = np.vstack((valid_ball, body, np.r_[env.target, 0.]))
    low, high = points.min(0), points.max(0)
    focus = (low + high) / 2
    distance = max(3.8, float(np.linalg.norm(high-low))*1.20)
    overview = camera(focus, distance, args.azimuth, args.elevation)
    captured_index = int(np.argmin(np.abs(states["time"]-result["capture_time"])))
    catch_focus = ball[captured_index].copy()
    close = camera(catch_focus + np.array([0., 0., .18]), 1.9, args.azimuth, -17.)
    times = states["time"]
    width, height, fps = 1280, 720, 60
    phases = ("APPROACH", "CAPTURE / BUFFER", "SETTLE", "SPIN-UP", "RELEASE", "BALL FLIGHT")
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    def render(index, replay=False, view=None):
        render_data.qpos[:] = states["qpos"][index]
        render_data.qvel[:] = states["qvel"][index]
        render_data.time = float(times[index]+2.)
        render_data.eq_active[:] = 0
        mujoco.mj_forward(mj_model, render_data)
        renderer.update_scene(render_data, camera=view or overview)
        frame = renderer.render().copy()
        frame[:100] = (20, 25, 28)
        frame[-62:] = (20, 25, 28)
        draw_text(frame, "V15 RESIDUAL RL | CATCH & THROW", (28, 34), .80)
        draw_text(frame, f"PPO best update {payload['update']}  |  seed {args.seed}  |  MuJoCo simulation", (28, 66), .56)
        draw_text(frame, "SLOW MOTION 0.25x" if replay else "FULL EPISODE 1.00x", (922, 38), .64,
            (100, 222, 175))
        draw_text(frame, f"t = {times[index]:5.2f} s   |   {phases[int(states['phase'][index])]}", (28, height-34), .64)
        label = "Target radius: 15 cm"
        if index == len(times)-1:
            label = f"TARGET HIT | error {result['landing_error_m']*100:.2f} cm"
        elif times[index] >= result["capture_time"]:
            label = "BALL CAUGHT" if times[index] < result["release_time"] else "BALL RELEASED"
        draw_text(frame, label, (825, height-34), .65, (100, 222, 175))
        return frame

    try:
        preview_indices = [max(0, captured_index-8), captured_index+5,
            int(np.argmin(np.abs(times-result["release_time"]))), len(times)-1]
        previews = [cv2.resize(render(i), (640, 360)) for i in preview_indices]
        montage = np.vstack((np.hstack(previews[:2]), np.hstack(previews[2:])))
        preview_path = output.with_name(output.name+"_preview.jpg")
        cv2.imwrite(str(preview_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        catch_preview = render(captured_index+5, True, close)
        cv2.imwrite(str(output.with_name(output.name+"_catch.jpg")), cv2.cvtColor(catch_preview, cv2.COLOR_RGB2BGR))
        print(json.dumps(dict(preview=str(preview_path), focus=focus.tolist(), distance=distance,
            catch_focus=catch_focus.tolist(), bounds=[low.tolist(), high.tolist()])), flush=True)
        if args.preview_only:
            return
        writer = imageio_ffmpeg.write_frames(str(output.with_suffix(".mp4")), (width, height),
            fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
            output_params=["-movflags", "+faststart", "-crf", "19", "-preset", "fast"])
        writer.send(None)
        count = 0

        def send(frame, repeat=1):
            nonlocal count
            for _ in range(repeat):
                writer.send(np.ascontiguousarray(frame).tobytes())
                count += 1

        try:
            for index in range(len(times)):
                send(render(index))
                if index % 300 == 0:
                    print(json.dumps(dict(stage="render_full_episode", frame=index, total=len(times))), flush=True)
            send(render(len(times)-1), 90)
            capture_range = np.flatnonzero((times >= result["capture_time"]-.5) &
                (times <= result["capture_time"]+1.0))
            for index in capture_range:
                send(render(index, True, close), 4)
            throw_range = np.flatnonzero(times >= result["release_time"]-.5)
            for index in throw_range:
                send(render(index, True), 4)
            send(render(len(times)-1), 120)
        finally:
            writer.close()
        metadata.update(video=str(output.with_suffix(".mp4").resolve()), width=width, height=height,
            fps=fps, frames=count, duration_s=count/fps,
            segments=["full episode 1x", "catch replay 0.25x", "throw replay 0.25x"],
            camera=dict(focus=focus.tolist(), distance=distance, azimuth=args.azimuth, elevation=args.elevation))
        capture = cv2.VideoCapture(str(output.with_suffix(".mp4")))
        decoded, checks = 0, []
        check_indices = {captured_index-8, captured_index+5, len(times)-1, count-1}
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise RuntimeError("Incorrect encoded video dimensions")
                if decoded in check_indices:
                    if frame[110:-70].std() < 5:
                        raise RuntimeError("Blank encoded scene")
                    checks.append(cv2.resize(frame, (640, 360)))
                decoded += 1
        finally:
            capture.release()
        if decoded != count:
            raise RuntimeError(f"Truncated video: decoded {decoded} of {count} frames")
        if np.abs(checks[0].astype(float)-checks[1].astype(float)).mean() < .1:
            raise RuntimeError("Encoded video does not show movement")
        cv2.imwrite(str(output.with_name(output.name+"_encoded_preview.jpg")),
            np.vstack((np.hstack(checks[:2]), np.hstack(checks[2:]))))
        metadata["verification"] = dict(decoded_frames=decoded, nonblank=True, movement=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(dict(stage="complete", video=metadata["video"], frames=count,
            duration_s=count/fps, hit15=result["hit15"], error_cm=result["landing_error_m"]*100)), flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
