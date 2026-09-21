"""Summarize paired validation without rerunning or retuning the policies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--variant", default="ball24_strict")
    args = parser.parse_args()

    def read(name):
        return json.loads((args.directory / name).read_text(encoding="utf-8"))

    baseline = {r["seed"]: r for r in read("v15_bc_episodes.json")}
    rows = read(args.variant + "_episodes.json")
    result = read("comparison.json")[args.variant]["report"]
    captured = [(r["result"], baseline[r["seed"]]["result"]) for r in rows
        if r["result"]["captured"] and baseline[r["seed"]]["result"]["captured"]]
    metrics = {}
    for key in ("impact_weld_peak_n", "impact_weld_impulse_ns", "relative_capture_speed_m_s",
                "pressure_integral_psi_s", "catch_to_orbit_s", "catch_to_release_s"):
        values = np.array([[a[key], b[key]] for a, b in captured
            if a.get(key) is not None and b.get(key) is not None], dtype=float)
        candidate, original = values.mean(axis=0)
        metrics[key] = dict(pairs=len(values), candidate=float(candidate), baseline=float(original),
            change_percent=float(100 * (candidate / original - 1)) if original else None)
    failures = []
    for row in rows:
        a, b = row["result"], baseline[row["seed"]]["result"]
        if a["hit15"]:
            continue
        failures.append(dict(seed=row["seed"], baseline_hit15=b["hit15"],
            cause="missed_catch" if not a["captured"] else "no_release" if not a["released"] else "landing_error",
            landing_error_m=a.get("landing_error_m"), baseline_landing_error_m=b.get("landing_error_m"),
            catch_to_release_s=a.get("catch_to_release_s"), scenario=row["scenario"]))
    analysis = dict(paired_captures=len(captured), paired_captured_metrics=metrics,
        failures=failures, checks=result["checks"], eligible=result["eligible"],
        task_nonregression=result["task_nonregression"],
        mean_handover_s=result["mean_catch_to_orbit_s"], max_handover_s=result["max_catch_to_orbit_s"],
        max_pause_s=result["max_longest_postcatch_pause_s"])
    (args.directory / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    grid = np.linspace(-.15, 2., 431)
    speeds = []
    for row in rows:
        capture = row["result"].get("capture_physics_time")
        if capture is not None:
            trace = np.asarray(row["motion_trace"])
            speeds.append(np.interp(grid, trace[:, 0] - capture, trace[:, 1]))
    low, median, high = np.percentile(speeds, [10, 50, 90], axis=0)
    axes[0].fill_between(grid, low, high, color="#52a8a0", alpha=.25, label="V16 10-90%")
    axes[0].plot(grid, median, color="#126b63", label="V16 median")
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].axhline(.15, color="#a44444", linestyle="--", label="Low-speed threshold")
    axes[0].set(xlabel="Time since capture (s)", ylabel="Tip speed (m/s)", title="Continuous catch-to-orbit motion")
    axes[0].legend(fontsize=8)
    points = [(baseline[r["seed"]]["result"].get("landing_error_m"), r["result"].get("landing_error_m")) for r in rows]
    points = np.array([(a, b) for a, b in points if a is not None and b is not None]) * 100
    limit = max(20., float(points.max()) * 1.1)
    axes[1].scatter(points[:, 0], points[:, 1], color="#126b63", s=24)
    axes[1].plot([0, limit], [0, limit], color="#888888", linestyle=":")
    axes[1].axhline(15, color="#a44444", linestyle="--")
    axes[1].axvline(15, color="#a44444", linestyle="--")
    axes[1].set(xlim=(0, limit), ylim=(0, limit), xlabel="V15 BC landing error (cm)",
        ylabel="V16 landing error (cm)", title="Paired landed episodes; failures counted separately")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(args.directory / "validation_motion_accuracy.png", dpi=150)
    plt.close(fig)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
