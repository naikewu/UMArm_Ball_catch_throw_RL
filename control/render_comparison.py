"""Render saved controller traces on one clock, without stepping physics.

Examples::
    python -m control.render_comparison video --folder deliverable/dynamic_control/benchmark --speed slow --seed 202609211
    python -m control.render_comparison figures --folder deliverable/dynamic_control/benchmark

Every guide and measured stroke comes from the saved trace. Source hashes,
camera, encoding and decoded frame counts are recorded beside each video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

METHODS = ("pid", "ff_pid", "koopman_mppi")
LABELS = {"pid": "PID", "ff_pid": "Feedforward + PID", "koopman_mppi": "Koopman MPPI"}
# BGR, with companion plotting colours.
COLORS = {"pid": (196, 107, 40), "ff_pid": (51, 122, 221), "koopman_mppi": (122, 142, 31)}
HEX = {"pid": "#286bc4", "ff_pid": "#dd7a33", "koopman_mppi": "#1f8e7a"}
BACKGROUND = (246, 246, 242)
INK = (45, 47, 45)
MUTED = (109, 110, 106)
W, H, COL_W = 1920, 1080, 640
ARM_Y, ARM_H = 150, 490
PLOT_Y, PLOT_H = 678, 328
CAMERA = {"azimuth": 105.0, "elevation": -10.0, "distance": 1.65,
          "lookat": (0.0, 0.0, 0.80)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(image, text, xy, size=.7, color=INK, thickness=1):
    cv2.putText(image, str(text), tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def _load(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("t", "q_true", "tip_true", "tip_ref", "pen_down"):
        if key not in trace or not np.all(np.isfinite(trace[key])):
            raise ValueError(f"{path}: missing or nonfinite {key}")
    if len(trace["t"]) < 2 or np.any(np.diff(trace["t"]) <= 0):
        raise ValueError(f"{path}: clock is not strictly increasing")
    active = trace["pen_down"].astype(bool)
    if not active.any():
        raise ValueError(f"{path}: no writing interval")
    trace["error_mm"] = 1000*np.linalg.norm(trace["tip_true"]-trace["tip_ref"], axis=1)
    rms = float(np.sqrt(np.mean(trace["error_mm"][active]**2)))
    published = meta["metrics"]["writing_tip_rms_mm"]
    if not np.isclose(rms, published, rtol=1e-8):
        raise ValueError(f"{path}: saved RMS differs from trace ({published} vs {rms})")
    return {"trace": trace, "meta": meta, "rms_mm": rms, "path": str(path.resolve()),
            "sha256": _sha(path), "meta_sha256": _sha(meta_path)}


def load_comparison(folder, speed, seed, condition="nominal", test_trace=None):
    runs = {method: _load(test_trace if test_trace else
            Path(folder)/f"{speed}_{condition}_{seed}_{method}.npz") for method in METHODS}
    reference = runs[METHODS[0]]["trace"]
    reference_meta = runs[METHODS[0]]["meta"]
    for method, run in runs.items():
        if not test_trace and run["meta"]["method"] != method:
            raise ValueError(f"requested {method}, metadata names {run['meta']['method']}")
        if not test_trace and run["meta"]["seed"] != seed:
            raise ValueError(f"{method} metadata has a different seed")
        if run["meta"].get("source_sha256") != reference_meta.get("source_sha256"):
            raise ValueError(f"{method} was run with different controller or plant sources")
        for key in ("t", "tip_ref", "pen_down", "q_ref"):
            if not np.array_equal(reference[key], run["trace"][key]):
                raise ValueError(f"{method} has a different {key}; a shared-clock video is invalid")
    return runs


class ComparisonRenderer:
    def __init__(self, runs, *, test=False):
        from digital_twin.side_by_side import ArmRenderer
        from digital_twin.mjcf_generator import generate_xml
        from digital_twin.twin_params import load_twin_kwargs
        self.runs, self.test = runs, bool(test)
        self.trace = runs[METHODS[0]]["trace"]
        self.times = self.trace["t"]
        self.duration = float(self.times[-1] + np.median(np.diff(self.times)))
        kwargs = dict(load_twin_kwargs(log=lambda _: None))
        kwargs.pop("actuator", None)
        kwargs.update(base_pos=(0, 0, 1.25))
        self.xml = generate_xml(**kwargs)
        self.arm = ArmRenderer(width=COL_W-24, height=ARM_H, cam=CAMERA, xml=self.xml)
        # Identical x/y scales across all controllers; include every actual point.
        self.active = self.trace["pen_down"].astype(bool)
        cloud = np.concatenate([self.trace["tip_ref"][self.active, :2]] +
            [runs[m]["trace"]["tip_true"][self.active, :2] for m in METHODS])
        middle = .5*(cloud.max(0)+cloud.min(0))
        extent = np.maximum(cloud.max(0)-cloud.min(0), [0.65, 0.32])*1.08
        # Drawing area has 548 by 230 pixels; enforce equal physical scales.
        ratio = 548/230
        extent[0] = max(extent[0], extent[1]*ratio)
        extent[1] = extent[0]/ratio
        self.lower, self.upper = middle-extent/2, middle+extent/2
        self.guide = self._map(self.trace["tip_ref"][self.active, :2])
        self.plot_background = self._plot_background()
        self.baseline = self._baseline()

    def _map(self, points):
        unit = (np.asarray(points)-self.lower)/(self.upper-self.lower)
        return np.rint(np.stack([70+548*unit[..., 0], 276-230*unit[..., 1]], axis=-1)).astype(np.int32)

    def _plot_background(self):
        panel = np.full((PLOT_H, COL_W, 3), BACKGROUND, dtype=np.uint8)
        _text(panel, "Writing plane  |  x-y projection", (28, 25), .57, MUTED)
        for x in np.arange(-.9, .901, .15):
            if self.lower[0] <= x <= self.upper[0]:
                p = self._map([x, self.lower[1]])
                cv2.line(panel, (p[0], 46), (p[0], 276), (225, 226, 222), 1)
                _text(panel, f"{x*100:.0f}", (p[0]-13, 298), .42, MUTED)
        for y in np.arange(-1.05, 1.051, .15):
            if self.lower[1] <= y <= self.upper[1]:
                p = self._map([self.lower[0], y])
                cv2.line(panel, (70, p[1]), (618, p[1]), (225, 226, 222), 1)
                _text(panel, f"{y*100:.0f}", (29, p[1]+5), .42, MUTED)
        cv2.rectangle(panel, (70, 46), (618, 276), (211, 213, 209), 1)
        cv2.polylines(panel, [self.guide], False, (188, 191, 186), 2, cv2.LINE_AA)
        _text(panel, "x (cm)", (305, 322), .43, MUTED)
        _text(panel, "y (cm)", (9, 43), .4, MUTED)
        return panel

    def _baseline(self):
        image = np.full((H, W, 3), BACKGROUND, dtype=np.uint8)
        _text(image, 'Soft writing with the ProMax arm', (28, 46), 1.12, INK, 2)
        mode = "PLUMBING TEST: SAME PID TRACE IN ALL PANELS" if self.test else "SIMULATION"
        _text(image, mode, (28, 77), .55, (70, 70, 177) if self.test else MUTED, 1)
        meta = self.runs[METHODS[0]]["meta"]
        trajectory = meta["trajectory"]
        extent = trajectory["extent_m"]
        speed = trajectory["speed_m_s"]
        _text(image, f"{100*extent[0]:.1f} x {100*extent[1]:.1f} cm  |  cruise {1000*speed:.0f} mm/s",
              (1230, 78), .66, MUTED)
        for j, method in enumerate(METHODS):
            x = j*COL_W
            if j:
                cv2.line(image, (x, 103), (x, 1017), (213, 215, 210), 1)
            cv2.rectangle(image, (x+24, 108), (x+30, 138), COLORS[method], -1)
            _text(image, LABELS[method], (x+45, 132), .87, INK, 2)
            image[PLOT_Y:PLOT_Y+PLOT_H, x:x+COL_W] = self.plot_background
        cv2.line(image, (28, 1018), (1892, 1018), (211, 213, 209), 1)
        _text(image, "Gray: common target   |   Color: recorded tip   |   Rings: target now / filled dots: actual now",
              (28, 1046), .55, MUTED)
        _text(image, "150 Hz CAN pressure feedback  /  240 Hz noisy mocap  |  Saved poses only; no physics during rendering",
              (28, 1072), .5, MUTED)
        return image

    def nearest(self, t):
        k = min(int(np.searchsorted(self.times, t)), len(self.times)-1)
        return k-1 if k and abs(self.times[k-1]-t) < abs(self.times[k]-t) else k

    def frame(self, t):
        image = self.baseline.copy()
        k = self.nearest(t)
        _text(image, f"{t:05.2f} / {self.duration:05.2f} s     Playback 1x", (1280, 42), .79, INK, 1)
        phase = "WRITING" if self.active[k] else "MOVE TO START" if t < 3 else "HOLD"
        for j, method in enumerate(METHODS):
            x = j*COL_W
            run, color = self.runs[method], COLORS[method]
            tr = run["trace"]
            image[ARM_Y:ARM_Y+ARM_H, x+12:x+COL_W-12] = self.arm.render(tr["q_true"][k])
            _text(image, phase, (x+30, ARM_Y+30), .49, (230, 232, 228))
            _text(image, f"Now {tr['error_mm'][k]:5.1f} mm", (x+28, 665), .67, color, 1)
            _text(image, f"Word RMS {run['rms_mm']:5.1f} mm", (x+306, 665), .67, INK, 1)
            panel = image[PLOT_Y:PLOT_Y+PLOT_H, x:x+COL_W]
            indices = np.flatnonzero(self.active[:k+1])
            if len(indices) > 1:
                path = self._map(tr["tip_true"][indices, :2])
                cv2.polylines(panel, [path], False, color, 2, cv2.LINE_AA)
            if self.active[k] or len(indices):
                actual = self._map(tr["tip_true"][k, :2])
                desired = self._map(tr["tip_ref"][k, :2])
                cv2.circle(panel, tuple(desired), 6, (118, 120, 115), 2, cv2.LINE_AA)
                cv2.circle(panel, tuple(actual), 5, color, -1, cv2.LINE_AA)
        return image

    def close(self):
        self.arm.renderer.close()


def _ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install imageio-ffmpeg or provide ffmpeg on PATH for H.264 output") from exc


def render_video(folder, out, *, speed="slow", seed=202609211, condition="nominal",
                 fps=30, frames_only=False, test_trace=None):
    runs = load_comparison(folder, speed, seed, condition, test_trace)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{'TEST_' if test_trace else ''}soft_{speed}_{condition}_{seed}"
    renderer = ComparisonRenderer(runs, test=bool(test_trace))
    manifest = {"simulation": True, "test_duplicated_trace": bool(test_trace),
        "seed": seed, "speed": speed, "condition": condition, "panel_order": list(METHODS),
        "camera": CAMERA, "size": [W, H], "fps": fps, "playback_speed": 1,
        "pose_selection": "nearest recorded control timestamp; never integrates physics",
        "guide_source": "saved tip_ref and pen_down from the same episodes",
        "model_xml_sha256": hashlib.sha256(renderer.xml.encode()).hexdigest(),
        "runs": {m: {k: v for k, v in runs[m].items() if k not in ("trace", "meta")} for m in METHODS}}
    try:
        timing = runs[METHODS[0]]["meta"]["trajectory"]
        key_times = [0.0, .5*(timing["write_start_s"]+timing["write_end_s"]), renderer.duration-1/fps]
        shots = []
        for label, t in zip(("first", "mid", "final"), key_times):
            shot = renderer.frame(t)
            cv2.imwrite(str(out/f"{stem}_{label}.png"), shot)
            shots.append(cv2.resize(shot, (W//2, H//2)))
        contact = np.vstack(shots)
        cv2.imwrite(str(out/f"{stem}_contact_sheet.png"), contact)
        manifest["sample_times_s"] = key_times
        if not frames_only:
            path = out/f"{stem}.mp4"
            encoder = _ffmpeg()
            command = [encoder, "-y", "-loglevel", "error", "-f", "rawvideo",
                "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            count = int(np.ceil(renderer.duration*fps))
            try:
                for i in range(count):
                    process.stdin.write(renderer.frame(i/fps).tobytes())
                    if i % (fps*5) == 0:
                        print(f"{stem}: frame {i}/{count}", flush=True)
                process.stdin.close()
                process.stdin = None
                _, stderr = process.communicate(timeout=60)
            except BaseException:
                process.kill()
                process.communicate()
                raise
            if process.returncode:
                raise RuntimeError(stderr.decode(errors="replace"))
            reader = cv2.VideoCapture(str(path))
            decoded, reported_fps = 0, reader.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                if frame.shape[:2] != (H, W):
                    raise RuntimeError("decoded frame has wrong dimensions")
                decoded += 1
            reader.release()
            if decoded != count or abs(reported_fps-fps) > 1e-3:
                raise RuntimeError(f"video decode failed: {decoded}/{count}, fps={reported_fps}")
            manifest.update(video=str(path), video_sha256=_sha(path), frames=count,
                decoded_frames=decoded, duration_s=count/fps, codec="H.264 (libx264), yuv420p",
                encoder=encoder, simulated_duration_s=renderer.duration)
        (out/f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        renderer.close()


def performance_figure(folder, out, condition="nominal"):
    """Measured errors and compute cost across every matching saved seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = []
    for path in sorted(Path(folder).glob(f"*_{condition}_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("method") in METHODS:
            records.append((path.stem.split("_")[0], record))
    if not records:
        raise ValueError(f"No controller benchmark metadata in {folder}")
    source_hashes = records[0][1].get("source_sha256")
    for speed in {r[0] for r in records}:
        groups = {m: [r for s, r in records if s == speed and r["method"] == m]
                  for m in METHODS}
        seeds = sorted(r["seed"] for r in groups[METHODS[0]])
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{speed}: missing or duplicate PID seeds")
        for method, group in groups.items():
            if sorted(r["seed"] for r in group) != seeds:
                raise ValueError(f"{speed}: {method} does not have the same paired seeds")
            if any(r.get("source_sha256") != source_hashes for r in group):
                raise ValueError("performance figure mixes controller or plant source versions")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    speeds = [s for s in ("slow", "fast") if any(r[0] == s for r in records)]
    summary = {}
    for j, method in enumerate(METHODS):
        summary[method] = {}
        for i, speed in enumerate(speeds):
            cells = [r[1] for r in records if r[0] == speed and r[1]["method"] == method]
            if not cells:
                continue
            values = [np.array([r["metrics"]["writing_tip_rms_mm"] for r in cells]),
                      np.array([r["metrics"]["solver_ms"]["p95"] for r in cells])]
            summary[method][speed] = {"seeds": [r["seed"] for r in cells],
                "tip_rms_mm": values[0].tolist(), "solver_p95_ms": values[1].tolist()}
            for ax, samples in zip(axes, values):
                x = i+(j-1)*.24
                ax.bar(x, samples.mean(), .21, color=HEX[method], alpha=.85,
                       label=LABELS[method] if i == 0 else None)
                if len(samples) > 1:
                    ax.errorbar(x, samples.mean(), yerr=samples.std(ddof=1), fmt="none",
                                ecolor="#30332e", capsize=4, lw=1)
                offsets = np.linspace(-.045, .045, len(samples)) if len(samples) > 1 else np.zeros(1)
                ax.scatter(x+offsets, samples,
                           s=16, facecolor="white", edgecolor=HEX[method], zorder=4)
                ax.annotate(f"{samples.mean():.2f}", (x, samples.max()),
                            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(title="Whole-word tracking error", ylabel="3D tip RMSE (mm)")
    axes[1].set(title="Control solve cost", ylabel="95th-percentile solve time (ms)")
    axes[1].axhline(1000/150, color="#666666", linestyle="--", lw=1,
                   label="150 Hz cycle: 6.67 ms")
    for ax in axes:
        ax.set_xticks(np.arange(len(speeds)), [s.capitalize() for s in speeds])
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.20)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"ProMax Soft writing | {condition.capitalize()} simulation", fontsize=16)
    fig.supxlabel("Bars: paired-seed mean; whiskers: sample SD. Offline tracking excludes solver delay.", fontsize=10)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out/f"performance_{condition}.{suffix}", dpi=180)
    plt.close(fig)
    (out/f"performance_{condition}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("video", "figures"))
    parser.add_argument("--folder", type=Path, default=Path("deliverable/dynamic_control/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("deliverable/dynamic_control"))
    parser.add_argument("--speed", choices=("slow", "fast"), default="slow")
    parser.add_argument("--seed", type=int, default=202609211)
    parser.add_argument("--condition", default="nominal")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--test-trace", type=Path)
    args = parser.parse_args()
    if args.action == "figures":
        result = performance_figure(args.folder, args.out, args.condition)
    else:
        result = render_video(args.folder, args.out, speed=args.speed, seed=args.seed,
            condition=args.condition, fps=args.fps, frames_only=args.frames_only,
            test_trace=args.test_trace)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
