"""V21 collect and validate a sensor-only release calibration before another teacher screen."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe
from .release_calibration import SCHEMA, RidgeLandingCalibration, error_statistics, fit_ridge
from .release_calibration_env import CalibrationProbeConfig, CalibrationProbeEnv
from .trajectory_release_rl import previous_seeds as v20_previous_seeds, source_hash as v20_source_hash

DEFAULT_OUT = Path("teacher_runs/v21_release_calibration")
PHASES = (2.0, 2.5, 3.0, 3.5, 4.0)
CV_FOLDS = 5
LEGACY_COLLECTION_SOURCE_HASHES = frozenset({
    "954d5469d430a5b110ea46b10cf6423ad71c0b2c57e51461552d0b8e17b8966c",
})


def source_hash():
    digest = hashlib.sha256(v20_source_hash().encode())
    for name in ("release_calibration.py", "release_calibration_env.py", "release_calibration_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def previous_seeds(exclude=None):
    used = v20_previous_seeds()
    excluded = None if exclude is None else Path(exclude).resolve()
    for path in DEFAULT_OUT.rglob("collection_contract.json"):
        if excluded is None or path.parent.resolve() != excluded:
            used.update(row["seed"] for row in read_json(path)["manifest"]["scenarios"])
    return used


def trajectories():
    common = dict(release_ball_frames=24, align_release_clock=True, release_azimuth_bias_deg=0.,
        release_tolerance_m=.15, predict_ball_motion=False, release_speed_gain=1.)
    return {
        "q26_nominal": dict(**common, governor_start_deg=26., governor_band_deg=8.,
            force_start_scale=.90, force_ramp_s=.20, radius_ramp_s=.20, target_force_scale=1.,
            target_radius_scale=1., spinup_s=1.70, release_min_s=1.5, release_joint_deg=35.),
        "q26_fast": dict(**common, governor_start_deg=26., governor_band_deg=8.,
            force_start_scale=.95, force_ramp_s=.15, radius_ramp_s=.15, target_force_scale=1.05,
            target_radius_scale=.98, spinup_s=1.45, release_min_s=1.5, release_joint_deg=35.),
        "q28_safe": dict(**common, governor_start_deg=28., governor_band_deg=6.,
            force_start_scale=.92, force_ramp_s=.18, radius_ramp_s=.20, target_force_scale=1.02,
            target_radius_scale=1., spinup_s=1.60, release_min_s=1.5, release_joint_deg=36.),
    }


def cells():
    return [(name, age, CalibrationProbeConfig(**config, probe_name=name, probe_age_s=age))
        for name, config in trajectories().items() for age in PHASES]


def worker(job):
    scenario, anchor_path, config_data = job
    torch.set_num_threads(1)
    anchor, payload = load_anchor(anchor_path)
    env = CalibrationProbeEnv(scenario, ImprovedRecipe(**payload["recipe"]), anchor,
        CalibrationProbeConfig(**config_data))
    observation = env.reset(scenario["seed"])
    started = time.perf_counter()
    while not env.done:
        observation, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    return dict(seed=scenario["seed"], scenario=scenario, result=result, wall_s=time.perf_counter() - started)


def collect(args):
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    campaign_cells = cells()
    total = len(campaign_cells) * args.episodes_per_cell
    manifest = scenarios(total, args.seed, args.design_seed, recipe)
    assignments = []
    for index, scenario in enumerate(manifest["scenarios"]):
        name, age, config = campaign_cells[index % len(campaign_cells)]
        assignments.append(dict(scenario=scenario, trajectory=name, probe_age_s=age, config=asdict(config)))
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        manifest=manifest, episodes_per_cell=args.episodes_per_cell, phases_s=list(PHASES),
        trajectories=trajectories(), purpose="offline_release_departure_and_landing_calibration")
    args.out.mkdir(parents=True, exist_ok=True)
    contract = args.out / "collection_contract.json"
    if contract.exists():
        if read_json(contract) != identity:
            raise ValueError("Collection contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("V21 collection seeds overlap an earlier V15-V21 experiment")
    write_json(contract, identity)
    output = args.out / "controlled_release_episodes.json"
    rows = read_json(output) if output.exists() else []
    completed = {row["seed"] for row in rows}
    jobs = [(item["scenario"], str(args.init.resolve()), item["config"]) for item in assignments
        if item["scenario"]["seed"] not in completed]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output, sorted(rows, key=lambda item: item["seed"]))
            print(dict(completed=len(rows), total=total, seed=row["seed"],
                released=row["result"]["released"], hit15=row["result"]["hit15"]), flush=True)
    rows = sorted(rows, key=lambda item: item["seed"])
    write_json(output, rows)
    usable = sum(bool(row["result"].get("calibration_probe", {}).get("scheduled")) and
        row["result"].get("landing_xy") is not None for row in rows)
    write_json(args.out / "collection_summary.json", dict(schema=SCHEMA, source_hash=source_hash(),
        episodes=len(rows), usable_rows=usable, required_rows=100, output=str(output)))
    print(dict(episodes=len(rows), usable_rows=usable, next="fit"), flush=True)


def calibration_rows(rows):
    data = []
    for row in rows:
        result = row["result"]
        probe = result.get("calibration_probe", {})
        command = probe.get("command")
        actual = result.get("landing_xy")
        if not result.get("released") or command is None or actual is None:
            continue
        features = np.asarray(command.get("features", []), dtype=float)
        raw = np.asarray(command.get("raw_landing_xy", []), dtype=float)
        target = np.asarray(actual, dtype=float)
        if features.shape != (19,) or raw.shape != (2,) or target.shape != (2,) or not np.isfinite(np.r_[features, raw, target]).all():
            continue
        data.append(dict(seed=row["seed"], scenario_id=int(row["scenario"]["scenario_id"]),
            trajectory=probe["probe_name"], probe_age_s=float(probe["probe_age_s"]),
            features=features, raw_landing=raw, actual_landing=target))
    return data


def stratified_fold(row, cell_count, folds=CV_FOLDS):
    """Keep every trajectory/phase cell represented in every validation fold."""
    return (int(row["scenario_id"]) // int(cell_count)) % int(folds)


def evaluate_model(model, rows):
    features = np.vstack([row["features"] for row in rows])
    raw = np.vstack([row["raw_landing"] for row in rows])
    actual = np.vstack([row["actual_landing"] for row in rows])
    calibrated = np.vstack([model.predict_landing(landing, feature)
        for landing, feature in zip(raw, features)])
    metrics = error_statistics(raw, actual, calibrated)
    predictions = [dict(trajectory=row["trajectory"], probe_age_s=row["probe_age_s"],
        raw_landing=landing, actual_landing=truth, calibrated_landing=corrected)
        for row, landing, truth, corrected in zip(rows, raw, actual, calibrated)]
    return metrics, predictions


def prediction_metrics(predictions):
    raw = np.vstack([row["raw_landing"] for row in predictions])
    actual = np.vstack([row["actual_landing"] for row in predictions])
    calibrated = np.vstack([row["calibrated_landing"] for row in predictions])
    return error_statistics(raw, actual, calibrated)


def grouped_metrics(predictions, key):
    names = sorted({row[key] for row in predictions}, key=str)
    return {str(name): prediction_metrics([row for row in predictions if row[key] == name]) for name in names}


def acceptance_checks(pooled, folds, by_trajectory, by_phase):
    """Global requirements plus guards against one easy orbit or phase hiding a failure."""
    accurate_without_correction = lambda metrics: metrics["raw_mean_m"] <= .06
    return {
        "pooled_mean_le_8cm": pooled["calibrated_mean_m"] <= .08,
        "pooled_p90_le_16cm": pooled["calibrated_p90_m"] <= .16,
        "pooled_improvement_ge_25pct": pooled["relative_mean_improvement"] >= .25,
        "every_fold_mean_le_8cm": all(item["calibrated_mean_m"] <= .08 for item in folds),
        "every_fold_p90_le_16cm": all(item["calibrated_p90_m"] <= .16 for item in folds),
        "every_trajectory_rows_ge_30": all(item["rows"] >= 30 for item in by_trajectory.values()),
        "every_trajectory_mean_le_8cm": all(item["calibrated_mean_m"] <= .08 for item in by_trajectory.values()),
        "every_trajectory_p90_le_16cm": all(item["calibrated_p90_m"] <= .16 for item in by_trajectory.values()),
        "every_trajectory_improvement_ge_25pct": all(
            item["relative_mean_improvement"] >= .25 for item in by_trajectory.values()),
        "every_phase_rows_ge_20": all(item["rows"] >= 20 for item in by_phase.values()),
        "every_phase_mean_le_9cm": all(item["calibrated_mean_m"] <= .09 for item in by_phase.values()),
        "every_phase_p90_le_16cm": all(item["calibrated_p90_m"] <= .16 for item in by_phase.values()),
        "every_phase_improves_or_raw_le_6cm": all(
            item["relative_mean_improvement"] >= .20 or accurate_without_correction(item)
            for item in by_phase.values()),
    }


def fit(args):
    contract_path, episode_path = args.out / "collection_contract.json", args.out / "controlled_release_episodes.json"
    if not contract_path.exists() or not episode_path.exists():
        raise ValueError("Run Collect successfully before Fit")
    contract = read_json(contract_path)
    compatible_sources = LEGACY_COLLECTION_SOURCE_HASHES | {source_hash()}
    if (contract.get("schema") != SCHEMA or contract.get("source_hash") not in compatible_sources or
            contract.get("anchor_sha256") != file_hash(args.init)):
        raise ValueError("Collection does not match the current V21 source and BC anchor")
    data = calibration_rows(read_json(episode_path))
    if len(data) < 100:
        raise ValueError(f"Need at least 100 usable calibration rows; found {len(data)}")
    cell_count = len(contract["phases_s"]) * len(contract["trajectories"])
    fold_reports, predictions = [], []
    for fold in range(CV_FOLDS):
        train = [row for row in data if stratified_fold(row, cell_count) != fold]
        holdout = [row for row in data if stratified_fold(row, cell_count) == fold]
        if len(train) < 80 or len(holdout) < 20:
            raise ValueError(f"Fold {fold} has insufficient train/holdout rows: {len(train)}/{len(holdout)}")
        x_train = np.vstack([row["features"] for row in train])
        raw_train = np.vstack([row["raw_landing"] for row in train])
        actual_train = np.vstack([row["actual_landing"] for row in train])
        fold_model = fit_ridge(x_train, actual_train - raw_train, args.ridge_lambda)
        metrics, fold_predictions = evaluate_model(fold_model, holdout)
        fold_reports.append(dict(fold=fold, train_rows=len(train), holdout_rows=len(holdout), **metrics))
        predictions.extend(fold_predictions)
    pooled = prediction_metrics(predictions)
    by_trajectory = grouped_metrics(predictions, "trajectory")
    by_phase = grouped_metrics(predictions, "probe_age_s")
    checks = acceptance_checks(pooled, fold_reports, by_trajectory, by_phase)
    accepted = all(checks.values())
    all_features = np.vstack([row["features"] for row in data])
    all_raw = np.vstack([row["raw_landing"] for row in data])
    all_actual = np.vstack([row["actual_landing"] for row in data])
    model = fit_ridge(all_features, all_actual - all_raw, args.ridge_lambda)
    payload = dict(**model.to_dict(), fit_source_hash=source_hash(),
        collection_source_hash=contract["source_hash"], anchor_sha256=file_hash(args.init),
        collection_contract_sha256=file_hash(contract_path), deployment_fit_rows=len(data),
        validation_protocol="trajectory_phase_stratified_5fold_v2",
        cross_validation=dict(pooled=pooled, folds=fold_reports,
            by_trajectory=by_trajectory, by_phase=by_phase), checks=checks, accepted=accepted,
        acceptance=dict(max_pooled_mean_m=.08, max_pooled_p90_m=.16,
            min_pooled_improvement=.25, max_fold_mean_m=.08, max_fold_p90_m=.16,
            max_trajectory_mean_m=.08, max_trajectory_p90_m=.16,
            max_phase_mean_m=.09, max_phase_p90_m=.16))
    output = args.out / "release_calibration_stratified.json"
    write_json(output, payload)
    print(dict(accepted=accepted, deployment_fit_rows=len(data), pooled=pooled,
        checks=checks, output=str(output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    collect_parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "campaign_v1")
    collect_parser.add_argument("--episodes-per-cell", type=int, default=10)
    collect_parser.add_argument("--workers", type=int, default=8)
    collect_parser.add_argument("--seed", type=int, default=21100001)
    collect_parser.add_argument("--design-seed", type=int, default=20260925)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    fit_parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "campaign_v1")
    fit_parser.add_argument("--ridge-lambda", type=float, default=.25)
    args = parser.parse_args()
    if args.command == "collect" and (args.episodes_per_cell < 1 or args.workers < 1):
        parser.error("episodes-per-cell and workers must be positive")
    if args.command == "fit" and (not np.isfinite(args.ridge_lambda) or not 0. <= args.ridge_lambda <= 10.):
        parser.error("ridge-lambda must be finite and in [0, 10]")
    (collect if args.command == "collect" else fit)(args)


if __name__ == "__main__":
    main()
