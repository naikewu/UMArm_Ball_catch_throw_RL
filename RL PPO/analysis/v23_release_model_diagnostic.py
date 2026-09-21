"""Offline-only nonlinear residual diagnostic; does not alter the online release gate."""
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from teacher_rl.improved_rl import read_json
from teacher_rl.release_calibration import fit_ridge
from teacher_rl.release_calibration_rl import (calibration_rows, stratified_fold,
    prediction_metrics, grouped_metrics, acceptance_checks)


def fit_predict(train, holdout, length, penalty, kind="rbf"):
    x = np.vstack([row["features"] for row in train])
    y = np.vstack([row["actual_landing"] - row["raw_landing"] for row in train])
    test = np.vstack([row["features"] for row in holdout])
    linear = fit_ridge(x, y)
    z, zt = (x - linear.mean) / linear.scale, (test - linear.mean) / linear.scale
    residual = y - np.vstack([linear.predict_delta(row) for row in x])
    def kernel_matrix(a, b):
        if kind == "quadratic":
            return (1. + a @ b.T / length ** 2) ** 2
        return np.exp(-np.sum((a[:, None] - b[None]) ** 2, axis=-1) / (2 * length ** 2))
    kernel = kernel_matrix(z, z)
    weights = solve(kernel + penalty * np.eye(len(x)), residual, assume_a="pos")
    cross = kernel_matrix(zt, z)
    correction = np.vstack([linear.predict_delta(row) for row in test]) + cross @ weights
    return [dict(trajectory=row["trajectory"], probe_age_s=row["probe_age_s"],
        raw_landing=row["raw_landing"], actual_landing=row["actual_landing"],
        calibrated_landing=row["raw_landing"] + delta) for row, delta in zip(holdout, correction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("rbf", "quadratic"), default="rbf")
    parser.add_argument("--data", type=Path, default=Path("teacher_runs/v22_trajectory_envelope/campaign_v1"))
    parser.add_argument("--out", type=Path, default=Path("teacher_runs/v23_contextual/release_diagnostic"))
    args = parser.parse_args()
    root = args.data
    data = calibration_rows(read_json(root / "controlled_release_episodes.json"))
    folds = {i: [row for row in data if stratified_fold(row, 36) == i] for i in range(5)}
    candidates = [(length, penalty) for length in (1., 2., 4., 8.) for penalty in (.01, .1, 1.)]
    predictions, outer_reports = [], []
    for outer in range(5):
        development = [i for i in range(5) if i != outer]
        scores = []
        for length, penalty in candidates:
            inner_predictions = []
            for inner in development:
                train = [row for i in development if i != inner for row in folds[i]]
                inner_predictions.extend(fit_predict(train, folds[inner], length, penalty, args.kind))
            scores.append(prediction_metrics(inner_predictions)["calibrated_mean_m"])
        length, penalty = candidates[int(np.argmin(scores))]
        train = [row for i in development for row in folds[i]]
        predicted = fit_predict(train, folds[outer], length, penalty, args.kind)
        metrics = prediction_metrics(predicted)
        outer_reports.append(dict(fold=outer, length=length, penalty=penalty, **metrics))
        predictions.extend(predicted)
        print(outer_reports[-1], flush=True)
    pooled = prediction_metrics(predictions)
    trajectories, phases = grouped_metrics(predictions, "trajectory"), grouped_metrics(predictions, "probe_age_s")
    result = dict(protocol="nested_5fold_outer_4fold_inner_by_trajectory_phase_repeat",
        use="diagnostic_only_no_online_deployment", pooled=pooled, folds=outer_reports,
        by_trajectory=trajectories, by_phase=phases,
        checks=acceptance_checks(pooled, outer_reports, trajectories, phases))
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.kind}_nested_cv.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(dict(pooled=pooled, checks=result["checks"]), flush=True)


if __name__ == "__main__":
    main()
