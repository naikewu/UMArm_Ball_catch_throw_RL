"""Render a deterministic physical catch-and-throw checkpoint episode to MP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import torch

from .catch_throw_env import CatchThrowEnv, CatchThrowEnvConfig
from .networks import ActorCritic
from .train_catch_throw import COMPATIBLE_CHECKPOINT_SCHEMAS


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                rgba: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius), np.asarray(position, dtype=float),
        np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _label_frame(frame_rgb: np.ndarray, *, time_s: float, phase: str,
                  target_distance_m: float, target_radius_m: float,
                  launch_distance_m: float, intercept_offset_m: float,
                  curriculum_stage: int, success: bool) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 16), (570, 104), (20, 22, 24), -1)
    cv2.addWeighted(overlay, .72, frame, .28, 0.0, frame)
    cv2.putText(frame, "PPO + Koopman-MPPI catch-and-throw",
                (34, 43), cv2.FONT_HERSHEY_SIMPLEX, .60, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, f"t = {time_s:4.2f} s    phase: {phase}",
                (34, 70), cv2.FONT_HERSHEY_SIMPLEX, .56, (230, 230, 230), 1, cv2.LINE_AA)
    if curriculum_stage <= 2:
        detail = (
            f"launch: {launch_distance_m:.2f} m    "
            f"intercept offset: {intercept_offset_m * 1000.0:.0f} mm"
        )
    else:
        detail = (
            f"ball-target: {target_distance_m * 1000.0:5.1f} mm    "
            f"target radius: {target_radius_m * 1000.0:.0f} mm"
        )
    cv2.putText(frame, detail, (34, 95), cv2.FONT_HERSHEY_SIMPLEX,
                .50, (220, 220, 220), 1, cv2.LINE_AA)
    if success:
        cv2.rectangle(frame, (frame.shape[1] - 245, 22),
                      (frame.shape[1] - 22, 74), (45, 145, 70), -1)
        result_label = "BALL CAPTURED" if curriculum_stage <= 2 else "TARGET HIT"
        cv2.putText(frame, result_label, (frame.shape[1] - 220, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _render_frame(renderer: mujoco.Renderer, env: CatchThrowEnv,
                  camera: mujoco.MjvCamera, trail: list[np.ndarray],
                  info: dict, success: bool) -> np.ndarray:
    renderer.update_scene(env.plant.arm.data, camera=camera)
    stage = int(info["curriculum_stage"])
    if stage <= 2:
        _add_sphere(
            renderer.scene, np.asarray(info["task_intercept_world_m"], dtype=float),
            .025, (.95, .65, .10, .18),
        )
    else:
        target = np.asarray(info["target_center_world_m"], dtype=float)
        _add_sphere(
            renderer.scene, target, float(info["target_radius_m"]),
            (.15, .78, .30, .20),
        )
    for position in trail[::2]:
        _add_sphere(renderer.scene, position, .004, (.12, .45, .90, .40))
    frame = renderer.render()
    phase = (
        "FLIGHT" if info["released"]
        else "CAPTURED" if info["captured"] and stage <= 2
        else "FOLLOW-THROUGH" if info["captured"]
        else "APPROACH"
    )
    launch_distance = float(np.linalg.norm(
        env.task.intercept_world_m - env.task.ball_position0_world_m
    ))
    intercept_offset = float(np.linalg.norm(
        env.task.intercept_world_m - env.home_cup_world_m
    ))
    return _label_frame(
        frame, time_s=float(info["time_s"]), phase=phase,
        target_distance_m=float(info["target_distance_m"]),
        target_radius_m=float(info["target_radius_m"]),
        launch_distance_m=launch_distance, intercept_offset_m=intercept_offset,
        curriculum_stage=stage, success=success,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=404000)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--stage2-level", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--camera-distance", type=float,
        help="free-camera distance; defaults to 1.25 m for catch and 1.55 m for throw",
    )
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--grasp-mode", choices=("auto_volume", "physical_dwell"),
        help="override the grasp mode stored in the checkpoint",
    )
    parser.add_argument("--auto-grasp-radius", type=float)
    parser.add_argument("--require-target-hit", action="store_true",
                        help="require the selected episode to enter the target")
    args = parser.parse_args()
    if (args.episode < 0 or args.episode_count <= 0
            or args.width <= 0 or args.height <= 0 or args.fps <= 0
            or (args.camera_distance is not None and args.camera_distance <= 0)):
        parser.error("episode, image size, and fps must be valid")
    if not 0 < args.playback_speed <= 1:
        parser.error("playback speed must be in (0, 1]")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") not in COMPATIBLE_CHECKPOINT_SCHEMAS:
        raise ValueError("checkpoint does not match the collision-aware PPO renderer")
    model = ActorCritic(CatchThrowEnv.observation_size, CatchThrowEnv.action_size)
    model.load_state_dict(payload["model"])
    model.eval()
    checkpoint_config = payload.get("config", {})
    grasp_mode = args.grasp_mode or checkpoint_config.get(
        "grasp_mode", "physical_dwell"
    )
    auto_grasp_radius = (
        args.auto_grasp_radius
        if args.auto_grasp_radius is not None
        else checkpoint_config.get("auto_grasp_radius")
    )
    env = CatchThrowEnv(CatchThrowEnvConfig(
        seed=args.seed, curriculum_stage=args.curriculum_stage,
        stage2_level=args.stage2_level,
        grasp_mode=grasp_mode, auto_grasp_radius_m=auto_grasp_radius,
    ))
    frames: list[np.ndarray] = []
    final_info: dict | None = None
    episode_metadata: list[dict] = []
    renderer: mujoco.Renderer | None = None
    try:
        final_episode = args.episode + args.episode_count
        for episode in range(final_episode):
            observation, info = env.reset(args.seed + episode)
            selected = args.episode <= episode < final_episode
            trail = [info["ball_position_m"].copy()]
            initial_cup_position = info["cup_position_m"].copy()
            maximum_cup_displacement_m = 0.0
            maximum_cup_speed_mps = float(np.linalg.norm(info["cup_velocity_mps"]))
            if selected:
                if renderer is None:
                    renderer = mujoco.Renderer(
                        env.plant.arm.model, height=args.height, width=args.width
                    )
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                focus_end = (
                    env.task.intercept_world_m if args.curriculum_stage <= 2
                    else env.task.target_center_world_m
                )
                camera.lookat[:] = .5 * (
                    env.task.ball_position0_world_m + focus_end
                )
                camera.lookat[2] += .05
                camera.distance = (
                    args.camera_distance
                    if args.camera_distance is not None
                    else (1.25 if args.curriculum_stage <= 2 else 1.55)
                )
                camera.azimuth = args.camera_azimuth
                camera.elevation = args.camera_elevation
                opening_frame = _render_frame(renderer, env, camera, trail, info, False)
                frames.extend([opening_frame] * max(1, args.fps // 2))
            terminated = truncated = False
            while not (terminated or truncated):
                tensor = torch.as_tensor(observation[None], dtype=torch.float32)
                with torch.no_grad():
                    action, _, _, _ = model.act(
                        tensor, deterministic=True,
                        action_mask=torch.as_tensor(env.action_mask()),
                    )
                observation, _, terminated, truncated, info = env.step(action[0].numpy())
                if selected:
                    maximum_cup_displacement_m = max(
                        maximum_cup_displacement_m,
                        float(np.linalg.norm(
                            info["cup_position_m"] - initial_cup_position
                        )),
                    )
                    maximum_cup_speed_mps = max(
                        maximum_cup_speed_mps,
                        float(np.linalg.norm(info["cup_velocity_mps"])),
                    )
                    trail.append(info["ball_position_m"].copy())
                    success = bool(
                        info["captured"] if args.curriculum_stage <= 2 else (
                            terminated and info["released"]
                            and info["target_distance_m"] <= info["target_radius_m"]
                        )
                    )
                    frames.append(_render_frame(renderer, env, camera, trail, info, success))
            if selected:
                final_info = info
                completed = bool(
                    info["captured"]
                    and (args.curriculum_stage <= 2 or info["released"])
                )
                target_hit = bool(
                    info["released"]
                    and info["target_distance_m"] <= info["target_radius_m"]
                )
                if not completed:
                    raise RuntimeError(f"selected episode {episode} did not complete its stage")
                if args.require_target_hit and not target_hit:
                    raise RuntimeError(f"selected episode {episode} did not enter the target")
                frames.extend([frames[-1]] * max(1, 3 * args.fps // 4))
                episode_metadata.append({
                    "episode": episode,
                    "seed": args.seed + episode,
                    "launch_position_world_m": env.task.ball_position0_world_m.tolist(),
                    "intercept_position_world_m": env.task.intercept_world_m.tolist(),
                    "launch_distance_m": float(np.linalg.norm(
                        env.task.intercept_world_m - env.task.ball_position0_world_m
                    )),
                    "intercept_offset_m": float(np.linalg.norm(
                        env.task.intercept_world_m - env.home_cup_world_m
                    )),
                    "intercept_time_s": float(env.task.intercept_time_s),
                    "maximum_cup_displacement_m": maximum_cup_displacement_m,
                    "maximum_cup_speed_mps": maximum_cup_speed_mps,
                    "captured": bool(info["captured"]),
                    "capture_trigger_mode": info["capture_trigger_mode"],
                    "physical_contact_at_grasp": bool(info["physical_contact_at_grasp"]),
                    "physical_compatible_at_grasp": bool(
                        info["physical_compatible_at_grasp"]
                    ),
                    "released": bool(info["released"]),
                    "target_hit": target_hit,
                    "contact_relative_speed_mps": info[
                        "minimum_contact_relative_speed_mps"
                    ],
                    "contact_alignment": info["last_contact_velocity_alignment"],
                    "peak_force_n": float(env.peak_force_n),
                })
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    if final_info is None:
        raise RuntimeError("selected episode was not recorded")
    target_hit = bool(
        final_info["captured"] and final_info["released"]
        and final_info["target_distance_m"] <= final_info["target_radius_m"]
    )

    repeats = max(1, int(round(1.0 / args.playback_speed)))
    encoded_frames = [frame for frame in frames for _ in range(repeats)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.out), (args.width, args.height), fps=args.fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart", "-crf", "18"],
    )
    writer.send(None)
    try:
        for frame in encoded_frames:
            writer.send(np.ascontiguousarray(frame).tobytes())
    finally:
        writer.close()
    cv2.imwrite(
        str(args.out.with_suffix(".png")),
        cv2.cvtColor(encoded_frames[-1], cv2.COLOR_RGB2BGR),
    )
    metadata = {
        "checkpoint": str(args.checkpoint), "seed": args.seed,
        "episode": args.episode, "curriculum_stage": args.curriculum_stage,
        "episode_count": args.episode_count,
        "grasp_mode": grasp_mode,
        "auto_grasp_radius_m": env._auto_grasp_radial_limit(),
        "launch_position_world_m": env.task.ball_position0_world_m.tolist(),
        "intercept_position_world_m": env.task.intercept_world_m.tolist(),
        "target_position_world_m": env.task.target_center_world_m.tolist(),
        "captured": bool(final_info["captured"]),
        "capture_trigger_mode": final_info["capture_trigger_mode"],
        "physical_contact_at_grasp": bool(final_info["physical_contact_at_grasp"]),
        "physical_compatible_at_grasp": bool(
            final_info["physical_compatible_at_grasp"]
        ),
        "released": bool(final_info["released"]),
        "target_hit": target_hit,
        "target_distance_m": float(final_info["target_distance_m"]),
        "target_radius_m": float(final_info["target_radius_m"]),
        "contact_relative_speed_mps": final_info["minimum_contact_relative_speed_mps"],
        "contact_radial_offset_m": final_info["last_contact_radial_offset_m"],
        "contact_alignment": final_info["last_contact_velocity_alignment"],
        "peak_force_n": float(env.peak_force_n),
        "frames": len(encoded_frames), "fps": args.fps,
        "playback_speed": args.playback_speed,
        "camera_distance_m": (
            args.camera_distance
            if args.camera_distance is not None
            else (1.25 if args.curriculum_stage <= 2 else 1.55)
        ),
        "camera_azimuth_deg": args.camera_azimuth,
        "camera_elevation_deg": args.camera_elevation,
        "episodes": episode_metadata,
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"wrote video: {args.out}")


if __name__ == "__main__":
    main()
