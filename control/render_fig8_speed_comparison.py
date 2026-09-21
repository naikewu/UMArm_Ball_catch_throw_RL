"""Render four simultaneous figure-eight tip-tracking speed experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from control.benchmark import run, save_run
from control.trajectory import FigureEightComplianceTrajectory
from digital_twin.tangent_compliance import TangentCompliance


DEFAULT_SPEED_SCALES = (0.5, 1.0, 1.5, 2.0)


def _speed_tag(speed_scale: float) -> str:
    return f"{speed_scale:g}x".replace(".", "p")


def make_trajectory(duration_s: float, speed_scale: float, *, entry_s: float = 2.0,
                    hold_s: float = 1.0, base_period_s: float = 9.0,
                    compliance_period_s: float | None = None,
                    compliance_soft: float = .098,
                    compliance_hard: float = .054) -> FigureEightComplianceTrajectory:
    """Keep the episode length fixed while changing only figure-eight rate."""
    drawing_s = duration_s - entry_s - hold_s
    if drawing_s <= 0:
        raise ValueError("duration must exceed entry plus hold time")
    period_s = base_period_s / speed_scale
    return FigureEightComplianceTrajectory(
        period_s=period_s,
        loops=drawing_s / period_s,
        entry_s=entry_s,
        hold_s=hold_s,
        grid_dt=.04,
        soft_m_per_n=compliance_soft,
        hard_m_per_n=compliance_hard,
        z_m_per_n=0,
        smooth_ramp_s=min(.75, drawing_s / 4),
        dome_m=.5,
        compliance_period_s=compliance_period_s,
    )


def _path_limits(traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xy = np.concatenate([
        np.concatenate((trace["tip_true"][:, :2], trace["tip_ref"][:, :2]), axis=0)
        for trace in traces
    ], axis=0)
    center = .5 * (xy.min(axis=0) + xy.max(axis=0))
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), [.08, .05]) * 1.12
    return center - span / 2, center + span / 2


def _nearest_indices(t_source: np.ndarray, t_frames: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_source, t_frames)
    idx = np.clip(idx, 1, len(t_source) - 1)
    prev = idx - 1
    return np.where(np.abs(t_source[prev] - t_frames) <= np.abs(t_source[idx] - t_frames), prev, idx)


def _ellipse_points(Cxy: np.ndarray, n: int = 160) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    points = (Cxy @ unit).T * 1000.0
    return np.concatenate((points, points[:1]))


def _centered_ellipse(Cxy: np.ndarray, tip_xy: np.ndarray, force_n: float) -> np.ndarray:
    return np.asarray(tip_xy) * 1000.0 + force_n * _ellipse_points(Cxy)


def _true_compliance(trace: dict, trace_path: Path, *, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and cache independent tangent compliance at video frame times."""
    frame_times = np.arange(0.0, trace["t"][-1] + np.median(np.diff(trace["t"])), 1.0 / fps)
    frame_idx = _nearest_indices(trace["t"], frame_times)
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    cache_path = trace_path.with_name(f"{trace_path.stem}_true_compliance_{fps}fps.npz")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if data["definition"].item() == "plate_tangent_v1" and data["trace_sha256"].item() == trace_sha:
                return data["C_true"], data["frame_idx"]

    probe = TangentCompliance()
    C_true = np.empty((len(frame_idx), 3, 3), dtype=np.float64)
    for i, k in enumerate(frame_idx):
        C_true[i] = probe.predict(trace["q_true"][k], trace["p_true_pa"][k])
        if (i + 1) % max(1, len(frame_idx) // 10) == 0:
            print(f"true compliance {trace_path.stem}: {i + 1}/{len(frame_idx)}", flush=True)
    np.savez_compressed(cache_path, C_true=C_true, frame_idx=frame_idx,
                        t=trace["t"][frame_idx], C_ref=trace["compliance_ref"][frame_idx],
                        definition="plate_tangent_v1", trace_sha256=trace_sha)
    return C_true, frame_idx


def render_mp4(traces: list[dict], metas: list[dict], speed_scales: list[float],
               out_path: Path, fps: int) -> None:
    """Show four independent rollouts against their common tip reference."""
    if len(traces) != 4:
        raise ValueError("the comparison renderer requires exactly four traces")
    duration_s = float(traces[0]["t"][-1] + np.median(np.diff(traces[0]["t"])))
    frame_times = np.arange(0.0, duration_s, 1.0 / fps)
    frame_indices = [np.minimum(np.searchsorted(trace["t"], frame_times), len(trace["t"]) - 1)
                     for trace in traces]
    lower, upper = _path_limits(traces)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.06, right=.98, bottom=.10, top=.89, wspace=.14, hspace=.26)
    actual_lines, target_marks, actual_marks, time_texts = [], [], [], []
    for ax, trace, meta, speed_scale in zip(axes.flat, traces, metas, speed_scales):
        active = trace["pen_down"].astype(bool)
        ax.set_facecolor("white")
        ax.grid(True, color="#e3e3e3", linewidth=.8)
        ax.plot(trace["tip_ref"][active, 0] * 1000, trace["tip_ref"][active, 1] * 1000,
                color="#9ca09a", linewidth=1.8, label="tip target")
        actual_line, = ax.plot([], [], color="#23835b", linewidth=1.8, label="tip actual")
        target_mark, = ax.plot([], [], marker="o", color="#333333", markersize=4, linestyle="")
        actual_mark, = ax.plot([], [], marker="o", color="#23835b", markersize=4, linestyle="")
        metrics = meta["metrics"]
        ax.set_title(
            f"{speed_scale:g}x path speed   |   tip RMS {metrics['writing_tip_rms_mm']:.2f} mm",
            fontsize=11,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lower[0] * 1000, upper[0] * 1000)
        ax.set_ylim(lower[1] * 1000, upper[1] * 1000)
        time_texts.append(ax.text(.02, .04, "", transform=ax.transAxes, fontsize=9, color="#333333"))
        actual_lines.append(actual_line)
        target_marks.append(target_mark)
        actual_marks.append(actual_mark)

    for ax in axes[:, 0]:
        ax.set_ylabel("y / mm")
    for ax in axes[-1, :]:
        ax.set_xlabel("x / mm")
    axes[0, 0].legend(loc="upper center", ncol=2, fontsize=8)
    fig.suptitle("Figure-eight tip tracking at four speed levels", fontsize=16)
    fig.text(.5, .025,
             "Same fitted twin, Koopman MPPI controller, 35 mm x 18 mm path, and 30 s simulated episode.",
             ha="center", fontsize=10, color="#333333")

    def update(frame: int):
        artists = []
        for trace, indices, actual_line, target_mark, actual_mark, time_text in zip(
                traces, frame_indices, actual_lines, target_marks, actual_marks, time_texts):
            k = int(indices[frame])
            actual_line.set_data(trace["tip_true"][:k + 1, 0] * 1000,
                                 trace["tip_true"][:k + 1, 1] * 1000)
            target_mark.set_data([trace["tip_ref"][k, 0] * 1000], [trace["tip_ref"][k, 1] * 1000])
            actual_mark.set_data([trace["tip_true"][k, 0] * 1000], [trace["tip_true"][k, 1] * 1000])
            time_text.set_text(f"t = {trace['t'][k]:.1f} / {duration_s:.1f} s")
            artists.extend((actual_line, target_mark, actual_mark, time_text))
        return artists

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2600,
                                    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    animation.FuncAnimation(fig, update, frames=len(frame_times), interval=1000 / fps, blit=False).save(
        out_path, writer=writer, dpi=120
    )
    update(len(frame_times) - 1)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def render_tip_and_compliance_mp4(traces: list[dict], metas: list[dict], speed_scales: list[float],
                                  C_true_all: list[np.ndarray], frame_indices: list[np.ndarray],
                                  out_path: Path, fps: int, ellipse_force_n: float = .2) -> None:
    """Render four path panels with moving target and true compliance ellipses."""
    if not (len(traces) == len(C_true_all) == len(frame_indices) == 4):
        raise ValueError("the compliance comparison requires four aligned traces")
    duration_s = float(traces[0]["t"][-1] + np.median(np.diff(traces[0]["t"])))
    n_frames = len(frame_indices[0])
    lower, upper = _path_limits(traces)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.06, right=.98, bottom=.10, top=.89, wspace=.14, hspace=.26)
    actual_lines, target_marks, actual_marks, target_ellipses, true_ellipses, time_texts = [], [], [], [], [], []
    for ax, trace, meta, speed_scale in zip(axes.flat, traces, metas, speed_scales):
        active = trace["pen_down"].astype(bool)
        ax.set_facecolor("white")
        ax.grid(True, color="#e3e3e3", linewidth=.8)
        ax.plot(trace["tip_ref"][active, 0] * 1000, trace["tip_ref"][active, 1] * 1000,
                color="#9ca09a", linewidth=1.8, label="tip target")
        actual_line, = ax.plot([], [], color="#23835b", linewidth=1.8, label="tip actual")
        target_mark, = ax.plot([], [], marker="o", color="#333333", markersize=4, linestyle="")
        actual_mark, = ax.plot([], [], marker="o", color="#23835b", markersize=4, linestyle="")
        target_ellipse, = ax.plot([], [], color="#d1495b", linestyle="--", linewidth=2.4,
                                  label="target compliance")
        true_ellipse, = ax.plot([], [], color="#1876b8", linewidth=1.4, label="MuJoCo tangent")
        metrics = meta["metrics"]
        compliance_error = metrics.get("true_compliance_xy_relF_median", float("nan")) * 100
        ax.set_title(f"{speed_scale:g}x   |   tip RMS {metrics['writing_tip_rms_mm']:.2f} mm   |   Cxy {compliance_error:.2f}%", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lower[0] * 1000, upper[0] * 1000)
        ax.set_ylim(lower[1] * 1000, upper[1] * 1000)
        actual_lines.append(actual_line)
        target_marks.append(target_mark)
        actual_marks.append(actual_mark)
        target_ellipses.append(target_ellipse)
        true_ellipses.append(true_ellipse)
        time_texts.append(ax.text(.02, .04, "", transform=ax.transAxes, fontsize=9, color="#333333"))

    for ax in axes[:, 0]:
        ax.set_ylabel("y / mm")
    for ax in axes[-1, :]:
        ax.set_xlabel("x / mm")
    axes[0, 0].legend(loc="upper center", ncol=2, fontsize=7.5)
    fig.suptitle("Figure-eight tip and compliance tracking at four matched speed levels", fontsize=16)
    fig.text(.5, .025,
             f"Tip and compliance references use the same speed scale. Ellipses: Cxy x {ellipse_force_n:g} N at actual tip. "
             "Red: target. Blue: independent MuJoCo tangent.",
             ha="center", fontsize=9, color="#333333")

    def update(frame: int):
        artists = []
        for trace, C_true, frame_idx, actual_line, target_mark, actual_mark, target_ellipse, true_ellipse, time_text in zip(
                traces, C_true_all, frame_indices, actual_lines, target_marks, actual_marks,
                target_ellipses, true_ellipses, time_texts):
            k = int(frame_idx[frame])
            actual_line.set_data(trace["tip_true"][:k + 1, 0] * 1000, trace["tip_true"][:k + 1, 1] * 1000)
            target_mark.set_data([trace["tip_ref"][k, 0] * 1000], [trace["tip_ref"][k, 1] * 1000])
            actual_mark.set_data([trace["tip_true"][k, 0] * 1000], [trace["tip_true"][k, 1] * 1000])
            C_ref = trace["compliance_ref"][frame_idx[frame], :2, :2]
            target_ellipse.set_data(*_centered_ellipse(C_ref, trace["tip_true"][k, :2], ellipse_force_n).T)
            true_ellipse.set_data(*_centered_ellipse(C_true[frame, :2, :2], trace["tip_true"][k, :2], ellipse_force_n).T)
            time_text.set_text(f"t = {trace['t'][k]:.1f} / {duration_s:.1f} s")
            artists.extend((actual_line, target_mark, actual_mark, target_ellipse, true_ellipse, time_text))
        return artists

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2800,
                                    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False).save(
        out_path, writer=writer, dpi=120
    )
    update(n_frames - 1)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def _write_report(out_dir: Path, speed_scales: list[float], metas: list[dict], args: argparse.Namespace,
                  compliance_metrics: list[dict] | None = None) -> None:
    headings = ["Speed", "Tip RMS", "Tip p95", "Tip max"]
    if compliance_metrics is not None:
        headings.append("True xy compliance median error")
    rows = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---:"] * len(headings)) + " |"]
    for index, (speed_scale, meta) in enumerate(zip(speed_scales, metas)):
        metrics = meta["metrics"]
        cells = [f"{speed_scale:g}x", f"{metrics['writing_tip_rms_mm']:.3f} mm",
                 f"{metrics['writing_tip_p95_mm']:.3f} mm", f"{metrics['writing_tip_max_mm']:.3f} mm"]
        if compliance_metrics is not None:
            cells.append(f"{compliance_metrics[index]['median_error'] * 100:.3f}%")
        rows.append("| " + " | ".join(cells) + " |")
    command = (
        ".\\.venv\\Scripts\\python.exe -m control.render_fig8_speed_comparison "
        f"--duration-s {args.duration_s:g} --fps {args.fps} --compliance-head {args.compliance_head}"
    )
    if args.with_compliance:
        command += " --with-compliance --reuse-traces"
    links = []
    tip_video = out_dir / f"fig8_tip_speed_comparison_{args.duration_s:g}s_seed{args.seed}.mp4"
    compliance_video = out_dir / f"fig8_tip_and_compliance_speed_comparison_{args.duration_s:g}s_seed{args.seed}.mp4"
    if tip_video.exists():
        links.append(f"[Tip-only four-panel MP4]({tip_video.name})")
    if compliance_video.exists():
        links.append(f"[Tip-and-compliance four-panel MP4]({compliance_video.name})")
    report = "\n".join([
        "# Figure-eight tip speed comparison",
        "",
        *links,
        "",
        "All panels retain the same fitted twin, Koopman MPPI settings, 35 mm by 18 mm tip path, seed, "
        "and compliance target range. The compliance reference uses the same phase and speed scale as the tip path.",
        "",
        *rows,
        "",
        "## Reproduce",
        "",
        "```powershell",
        command,
        "```",
        "",
        "The video shows nominal digital-twin tracking only. It is not a real-hardware speed qualification.",
        "",
    ])
    (out_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("deliverable/fig8_compliance_speed_matched_30s"))
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--speed-scales", nargs="+", type=float, default=list(DEFAULT_SPEED_SCALES))
    parser.add_argument("--mppi-horizon", type=int, default=16)
    parser.add_argument("--mppi-samples", type=int, default=32)
    parser.add_argument("--tip-weight", type=float, default=10.0)
    parser.add_argument("--compliance-weight", type=float, default=10.0)
    parser.add_argument("--compliance-soft", type=float, default=.098)
    parser.add_argument("--compliance-hard", type=float, default=.054)
    parser.add_argument("--compliance-period-s", type=float,
                        help="optional fixed compliance period; default follows each tip-path period")
    parser.add_argument("--compliance-head", type=Path,
                        default=Path("data/compliance_tangent_wide_5000/head.npz"))
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--reuse-traces", action="store_true")
    parser.add_argument("--with-compliance", action="store_true",
                        help="also render target and independent tangent-compliance tracking")
    parser.add_argument("--ellipse-force-n", type=float, default=.2)
    args = parser.parse_args()
    if len(args.speed_scales) != 4 or any(speed <= 0 for speed in args.speed_scales):
        parser.error("provide exactly four positive speed scales")
    if args.duration_s <= 3 or args.fps <= 0 or args.ellipse_force_n <= 0:
        parser.error("duration must exceed 3 seconds; fps and ellipse force must be positive")
    if not 0 < args.compliance_hard < args.compliance_soft:
        parser.error("require 0 < hard compliance < soft compliance")
    if not args.compliance_head.is_file():
        parser.error(f"compliance head not found: {args.compliance_head}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    duration_tag = f"{args.duration_s:g}s".replace(".", "p")
    traces, metas, trace_paths = [], [], []
    controller_kwargs = {
        "tip_weight": args.tip_weight,
        "compliance_weight": args.compliance_weight,
        "horizon": args.mppi_horizon,
        "samples": args.mppi_samples,
        "tangent_compliance": True,
        "noise_psi": .03,
        "common_noise_psi": .03,
        "action_regularization": 10.0,
        "compliance_head": str(args.compliance_head),
    }
    for speed_scale in args.speed_scales:
        run_name = f"fig8_speed_{_speed_tag(speed_scale)}_{duration_tag}_seed{args.seed}_koopman_mppi"
        trace_path = args.out_dir / f"{run_name}.npz"
        if args.reuse_traces and trace_path.exists():
            with np.load(trace_path, allow_pickle=False) as data:
                trace = {key: data[key] for key in data.files}
            meta = json.loads(trace_path.with_suffix(".json").read_text(encoding="utf-8"))
        else:
            trajectory = make_trajectory(
                args.duration_s, speed_scale,
                compliance_period_s=args.compliance_period_s,
                compliance_soft=args.compliance_soft,
                compliance_hard=args.compliance_hard,
            )
            trace, meta = run("koopman_mppi", trajectory, seed=args.seed,
                              duration_s=args.duration_s, controller_kwargs=controller_kwargs)
            save_run(args.out_dir, run_name, trace, meta)
        traces.append(trace)
        metas.append(meta)
        trace_paths.append(trace_path)
        print(json.dumps({"speed_scale": speed_scale, **meta["metrics"]}), flush=True)

    compliance_metrics = None
    if args.with_compliance:
        C_true_all, frame_indices, compliance_metrics = [], [], []
        for trace, meta, trace_path in zip(traces, metas, trace_paths):
            C_true, frame_idx = _true_compliance(trace, trace_path, fps=args.fps)
            active = trace["pen_down"][frame_idx].astype(bool)
            C_ref = trace["compliance_ref"][frame_idx, :2, :2]
            rel = np.linalg.norm(C_true[:, :2, :2] - C_ref, axis=(1, 2)) / np.maximum(
                np.linalg.norm(C_ref, axis=(1, 2)), 1e-12
            )
            metrics = {
                "median_error": float(np.median(rel[active])),
                "p90_error": float(np.percentile(rel[active], 90)),
                "correlation": [float(np.corrcoef(C_true[active, j, j], C_ref[active, j, j])[0, 1])
                                for j in range(2)],
            }
            meta["metrics"].update(true_compliance_xy_relF_median=metrics["median_error"],
                                   true_compliance_xy_relF_p90=metrics["p90_error"],
                                   true_compliance_xy_correlation=metrics["correlation"])
            trace_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            C_true_all.append(C_true)
            frame_indices.append(frame_idx)
            compliance_metrics.append(metrics)
        video_path = args.out_dir / f"fig8_tip_and_compliance_speed_comparison_{duration_tag}_seed{args.seed}.mp4"
        render_tip_and_compliance_mp4(traces, metas, args.speed_scales, C_true_all, frame_indices,
                                      video_path, args.fps, args.ellipse_force_n)
    else:
        video_path = args.out_dir / f"fig8_tip_speed_comparison_{duration_tag}_seed{args.seed}.mp4"
        render_mp4(traces, metas, args.speed_scales, video_path, args.fps)
    manifest = {
        "speed_scales": args.speed_scales,
        "duration_s": args.duration_s,
        "fps": args.fps,
        "video": video_path.name,
        "compliance_reference": ("same phase and speed scale as tip path"
                                 if args.compliance_period_s is None
                                 else f"fixed period {args.compliance_period_s:g} s"),
        "runs": [meta["metrics"] for meta in metas],
    }
    if compliance_metrics is not None:
        manifest["true_compliance"] = compliance_metrics
    manifest_name = "tip_and_compliance_comparison_manifest.json" if args.with_compliance else "comparison_manifest.json"
    (args.out_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(args.out_dir, args.speed_scales, metas, args, compliance_metrics)
    print(f"wrote video: {video_path}")


if __name__ == "__main__":
    main()
