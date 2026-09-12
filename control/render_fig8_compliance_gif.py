"""Run and render a figure-eight tip/compliance experiment as a GIF."""
from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from control.benchmark import run, save_run
from control.trajectory import FigureEightComplianceTrajectory
from digital_twin.tangent_compliance import TangentCompliance


def _nearest_indices(t_source: np.ndarray, t_frames: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_source, t_frames)
    idx = np.clip(idx, 1, len(t_source) - 1)
    prev = idx - 1
    return np.where(np.abs(t_source[prev] - t_frames) <= np.abs(t_source[idx] - t_frames), prev, idx)


def _finite_difference_compliance(q: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    probe = TangentCompliance()
    C = np.empty((len(q), 3, 3), dtype=np.float64)
    for i, (qi, pi) in enumerate(zip(q, p_pa)):
        C[i] = probe.predict(qi, pi)
        if (i + 1) % max(1, len(q) // 10) == 0:
            print(f"true compliance {i + 1}/{len(q)}", flush=True)
    return C


def _load_trace(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return trace, meta


def _ellipse_points(Cxy: np.ndarray, n: int = 160) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    unit = np.stack([np.cos(a), np.sin(a)], axis=0)
    points = (Cxy @ unit).T * 1000.0
    return np.concatenate((points, points[:1]))


def _centered_ellipse(Cxy, tip_xy, force_n):
    return np.asarray(tip_xy) * 1000 + force_n * _ellipse_points(Cxy)


def render_gif(trace: dict, meta: dict, frame_idx: np.ndarray,
               C_true: np.ndarray, out_gif: Path, fps: int,
               ellipse_force_n: float = .2) -> None:
    t = trace["t"]
    tip_true = trace["tip_true"]
    tip_ref = trace["tip_ref"]
    C_ref = trace["compliance_ref"][frame_idx]
    active = trace["pen_down"].astype(bool)
    duration = float(t[-1] + np.median(np.diff(t)))

    xy_cloud = np.concatenate([tip_true[:, :2], tip_ref[:, :2]], axis=0)
    mid = 0.5 * (xy_cloud.min(axis=0) + xy_cloud.max(axis=0))
    radius = max(np.linalg.norm(C_true[:, :2, :2], axis=(1, 2)).max(),
                 np.linalg.norm(C_ref[:, :2, :2], axis=(1, 2)).max()) * ellipse_force_n
    span = np.maximum(xy_cloud.max(axis=0) - xy_cloud.min(axis=0), [0.08, 0.05]) + 2 * radius

    fig = plt.figure(figsize=(12, 7), dpi=100)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], left=.07, right=.98,
                           bottom=.17, top=.85, wspace=.28, hspace=.50)
    ax_path = fig.add_subplot(grid[:, 0])
    axes_comp = [fig.add_subplot(grid[j, 1]) for j in range(2)]
    fig.patch.set_facecolor("white")
    for ax in (ax_path, *axes_comp):
        ax.set_facecolor("white")
        ax.grid(True, color="#e3e3e3", linewidth=0.8)

    ax_path.plot(tip_ref[active, 0] * 1000, tip_ref[active, 1] * 1000,
                 color="#9ca09a", linewidth=2.0, label="tip target")
    actual_line, = ax_path.plot([], [], color="#23835b", linewidth=1.8, label="tip actual")
    target_now, = ax_path.plot([], [], marker="o", color="#333333", markersize=6, linestyle="")
    actual_now, = ax_path.plot([], [], marker="o", color="#23835b", markersize=6, linestyle="")
    ax_path.set_title("Figure-eight + compliance at the actual tip", fontsize=12)
    ax_path.set_xlabel("x / mm")
    ax_path.set_ylabel("y / mm")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlim((mid[0] - span[0] / 2) * 1000, (mid[0] + span[0] / 2) * 1000)
    ax_path.set_ylim((mid[1] - span[1] / 2) * 1000, (mid[1] + span[1] / 2) * 1000)
    target_ellipse, = ax_path.plot([], [], color="#d1495b", linestyle="--", linewidth=3.4, label="target compliance")
    true_ellipse, = ax_path.plot([], [], color="#1876b8", linewidth=1.5, label="MuJoCo tangent compliance")
    ax_path.legend(loc="upper center", ncol=2, fontsize=8)
    comp_lines, cursors = [], []
    times = t[frame_idx]
    for j, ax in enumerate(axes_comp):
        ax.plot(times, C_ref[:, j, j] * 1000, "--", color="#d1495b", linewidth=1.6)
        line, = ax.plot([], [], color="#1876b8", linewidth=1.5)
        comp_lines.append(line)
        cursors.append(ax.axvline(0, color="#888888", linewidth=.8))
        ax.set(xlim=(0, duration), ylim=(45, max(90, C_true[:, j, j].max()*1050)),
               xlabel="Time / s", ylabel="Compliance / mm/N", title=["Cxx", "Cyy"][j])

    text = fig.text(0.07, 0.965, "", fontsize=12, color="#222222")
    metrics = meta.get("metrics", {})
    footer = (
        f"tip RMS {metrics.get('writing_tip_rms_mm', float('nan')):.2f} mm   "
        f"true xy compliance error {metrics.get('true_compliance_xy_relF_median', float('nan'))*100:.2f}%   "
        f"solver p99 {metrics.get('solver_ms', {}).get('p99', float('nan')):.2f} ms"
    )
    fig.text(0.07, 0.08, footer, fontsize=10, color="#333333")
    fig.text(0.07, 0.045, f"Both ellipses: actual tip centre; radius = Cxy x {ellipse_force_n:g} N. Raw samples, no smoothing.", fontsize=9)
    fig.text(0.07, 0.015, "Endpoint: final plate centre. Frozen-pressure local static tangent; not closed-loop dynamic admittance.", fontsize=9)

    def update(i: int):
        k = int(frame_idx[i])
        actual_line.set_data(tip_true[:k + 1, 0] * 1000, tip_true[:k + 1, 1] * 1000)
        target_now.set_data([tip_ref[k, 0] * 1000], [tip_ref[k, 1] * 1000])
        actual_now.set_data([tip_true[k, 0] * 1000], [tip_true[k, 1] * 1000])
        target_ellipse.set_data(*_centered_ellipse(C_ref[i, :2, :2], tip_true[k, :2], ellipse_force_n).T)
        true_ellipse.set_data(*_centered_ellipse(C_true[i, :2, :2], tip_true[k, :2], ellipse_force_n).T)
        for j in range(2):
            comp_lines[j].set_data(times[:i+1], C_true[:i+1, j, j] * 1000)
            cursors[j].set_xdata([t[k], t[k]])
        target_ratio = C_ref[i, 0, 0] / C_ref[i, 1, 1]
        actual_ratio = C_true[i, 0, 0] / C_true[i, 1, 1]
        text.set_text(f"t = {t[k]:05.2f} / {duration:05.2f} s     Cxx/Cyy: target {target_ratio:.2f}, actual {actual_ratio:.2f}")
        return actual_line, target_now, actual_now, target_ellipse, true_ellipse, text

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx), interval=1000 / fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=animation.PillowWriter(fps=fps))
    update(min(len(frame_idx)-1, int(17.5 * fps)))
    fig.savefig(out_gif.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("deliverable/fig8_compliance_gif"))
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mppi-horizon", type=int, default=16)
    parser.add_argument("--mppi-samples", type=int, default=32)
    parser.add_argument("--tip-weight", type=float, default=10.0)
    parser.add_argument("--compliance-weight", type=float, default=10.0)
    parser.add_argument("--ellipse-force-n", type=float, default=.2)
    parser.add_argument("--compliance-soft", type=float, default=.078)
    parser.add_argument("--compliance-hard", type=float, default=.060)
    parser.add_argument("--compliance-period-s", type=float, default=None)
    parser.add_argument("--compliance-head", type=Path)
    parser.add_argument("--mp4", action="store_true")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--reuse-trace", action="store_true")
    parser.add_argument("--reuse-true-compliance", action="store_true")
    args = parser.parse_args()
    if args.duration_s <= 0 or args.fps <= 0 or args.ellipse_force_n <= 0:
        parser.error("duration, fps and ellipse force must be positive")
    if not 0 < args.compliance_hard < args.compliance_soft:
        parser.error("require 0 < hard compliance < soft compliance")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    duration_tag = f"{args.duration_s:g}s".replace(".", "p")
    run_name = f"fig8_compliance_{duration_tag}_seed{args.seed}_koopman_mppi"
    trace_path = args.out_dir / f"{run_name}.npz"
    if args.reuse_trace and trace_path.exists():
        trace, meta = _load_trace(trace_path)
        if not meta.get("controller_options", {}).get("tangent_compliance"):
            raise ValueError("Legacy transient-compliance traces must be rerun with the new definition")
    else:
        period_s = 9.0
        entry_s = 2.0
        hold_s = 1.0
        loops = max(0.1, (args.duration_s - entry_s - hold_s) / period_s)
        trajectory = FigureEightComplianceTrajectory(
            period_s=period_s, loops=loops, entry_s=entry_s, hold_s=hold_s, grid_dt=0.04,
            soft_m_per_n=args.compliance_soft, hard_m_per_n=args.compliance_hard, z_m_per_n=0,
            smooth_ramp_s=min(.75, period_s*loops/4), dome_m=.5,
            compliance_period_s=args.compliance_period_s)
        trace, meta = run(
            "koopman_mppi",
            trajectory,
            seed=args.seed,
            duration_s=args.duration_s,
            controller_kwargs={
                "tip_weight": args.tip_weight,
                "compliance_weight": args.compliance_weight,
                "horizon": args.mppi_horizon,
                "samples": args.mppi_samples,
                "tangent_compliance": True,
                "noise_psi": .03,
                "common_noise_psi": .03,
                "action_regularization": 10.,
                "compliance_head": None if args.compliance_head is None else str(args.compliance_head),
            },
        )
        save_run(args.out_dir, run_name, trace, meta)

    frame_times = np.arange(0.0, args.duration_s, 1.0 / args.fps)
    frame_idx = _nearest_indices(trace["t"], frame_times)
    true_path = args.out_dir / f"{run_name}_true_compliance_{args.fps}fps.npz"
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if args.reuse_true_compliance and true_path.exists():
        with np.load(true_path, allow_pickle=False) as data:
            if data["definition"].item() != "plate_tangent_v1" or data["trace_sha256"].item() != trace_sha:
                raise ValueError("Compliance cache definition or source trace mismatch")
            C_true = data["C_true"]
            frame_idx = data["frame_idx"]
    else:
        C_true = _finite_difference_compliance(
            trace["q_true"][frame_idx],
            trace["p_true_pa"][frame_idx],
        )
        np.savez_compressed(true_path, C_true=C_true, frame_idx=frame_idx,
                            t=trace["t"][frame_idx], C_ref=trace["compliance_ref"][frame_idx],
                            definition="plate_tangent_v1", trace_sha256=trace_sha)

    active = trace["pen_down"][frame_idx].astype(bool)
    if not np.any(active):
        active = np.ones(len(frame_idx), dtype=bool)
    Cref = trace["compliance_ref"][frame_idx, :2, :2]
    rel = np.linalg.norm(C_true[:, :2, :2] - Cref, axis=(1, 2)) / np.linalg.norm(Cref, axis=(1, 2))
    metrics = meta["metrics"]
    metrics.update(true_compliance_xy_relF_median=float(np.median(rel[active])),
                   true_compliance_xy_relF_p90=float(np.percentile(rel[active], 90)),
                   true_compliance_xy_correlation=[float(np.corrcoef(C_true[active,j,j], Cref[active,j,j])[0,1]) for j in range(2)])
    eigenvalues = np.linalg.eigvalsh(C_true[active, :2, :2])
    metrics.update(true_compliance_max_aspect_ratio=float(np.max(eigenvalues[:,1]/eigenvalues[:,0])),
                   true_compliance_x_over_y_max=float(np.max(C_true[active,0,0]/C_true[active,1,1])),
                   true_compliance_y_over_x_max=float(np.max(C_true[active,1,1]/C_true[active,0,0])))
    meta["compliance_validation"] = dict(definition="local frozen-pressure static tangent at final plate centre",
        calculation="independent MuJoCo torque finite differences, C=J K^-1 J.T", smoothing=False,
        axes_controlled="xy; z is not independently targeted", ellipse_force_n=args.ellipse_force_n,
        old_target_m_per_n=[.035, .050], revised_target_m_per_n=[
            meta["trajectory"]["compliance_hard_m_per_n"], meta["trajectory"]["compliance_soft_m_per_n"]])
    trace_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    gif_path = args.out_dir / f"{run_name}.gif"
    render_gif(trace, meta, frame_idx, C_true, gif_path, args.fps, args.ellipse_force_n)
    if args.mp4:
        import imageio_ffmpeg
        mp4_path = gif_path.with_suffix(".mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                        "-y", "-ignore_loop", "1", "-i", str(gif_path), "-an", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(mp4_path)], check=True)
        print(f"wrote video: {mp4_path}")
    print(f"wrote trace: {trace_path}")
    print(f"wrote true compliance cache: {true_path}")
    print(f"wrote gif: {gif_path}")
    print(json.dumps(meta["metrics"], indent=2))


if __name__ == "__main__":
    main()
