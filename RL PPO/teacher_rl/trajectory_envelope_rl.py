"""V22 expand the throw-trajectory envelope before another online teacher screen."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe
from .release_calibration import SCHEMA as MODEL_SCHEMA, fit_ridge
from .release_calibration_env import ProbeRelease
from .release_calibration_rl import (CV_FOLDS, acceptance_checks, calibration_rows,
    evaluate_model, grouped_metrics, prediction_metrics, source_hash as v21_source_hash,
    stratified_fold)
from .trajectory_release_env import TrajectoryHandover, TrajectoryReleaseConfig, TrajectoryReleaseEnv

SCHEMA = "can_trajectory_envelope_v22"
DEFAULT_OUT = Path("teacher_runs/v22_trajectory_envelope")
PHASES = (2., 4., 6., 8., 12., 16.)


@dataclass(frozen=True)
class EnvelopeProbeConfig(TrajectoryReleaseConfig):
    probe_age_s: float = 4.
    probe_name: str = "unnamed"
    envelope_force_scale: float = 1.
    envelope_radius_scale: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.probe_age_s) or not 2. <= self.probe_age_s <= 18.:
            raise ValueError("probe_age_s must be in [2, 18]")
        if not self.probe_name or not self.probe_name.replace("_", "").isalnum():
            raise ValueError("probe_name must be a nonempty alphanumeric identifier")
        if (not np.isfinite(self.envelope_force_scale) or
                not 1. <= self.envelope_force_scale <= 1.4):
            raise ValueError("envelope_force_scale must be in [1, 1.4]")
        if (not np.isfinite(self.envelope_radius_scale) or
                not 1. <= self.envelope_radius_scale <= 1.18):
            raise ValueError("envelope_radius_scale must be in [1, 1.18]")


class EnvelopeHandover(TrajectoryHandover):
    """Apply V22's wider envelope outside the frozen V20 parameter bounds."""
    def command(self, observation):
        force, radius = self.drv.F_max, self.drv.r_final
        self.drv.F_max = force * self.config.envelope_force_scale
        self.drv.r_final = radius * self.config.envelope_radius_scale
        try:
            return super().command(observation)
        finally:
            self.drv.F_max, self.drv.r_final = force, radius


class EnvelopeProbeEnv(TrajectoryReleaseEnv):
    """V21 measured-state probe extended to V22's steady-orbit time envelope."""
    def __init__(self, scenario, recipe, anchor, config):
        if not isinstance(config, EnvelopeProbeConfig):
            raise TypeError("EnvelopeProbeEnv requires EnvelopeProbeConfig")
        super().__init__(scenario, recipe, anchor, config)

    def reset(self, seed):
        self.actual_departure = None
        self._last_held = False
        super().reset(seed)
        old = self.handover
        self.handover = EnvelopeHandover(self, old.catch, self.drive, self.model, self.continuous)
        self.controller.controller = self.handover
        self.releaser = ProbeRelease(self.releaser.original, self.handover, self.continuous)
        return self.observation()

    def _physics_metrics(self, t):
        held_before = self._last_held
        super()._physics_metrics(t)
        held_now = bool(self.plant.ball_held())
        if (self.actual_departure is None and held_before and not held_now and
                self.releaser.t_release is not None):
            self.actual_departure = dict(time_s=float(t - 2.), position_m=self.plant.ball_pos().tolist(),
                velocity_m_s=self.plant.ball_vel().tolist())
        self._last_held = held_now

    def summary(self):
        result = super().summary()
        result.update(calibration_probe=dict(self.releaser.audit),
            offline_actual_departure=self.actual_departure,
            envelope_probe_config=asdict(self.continuous))
        return result


def source_hash():
    digest = hashlib.sha256(v21_source_hash().encode())
    path = Path(__file__)
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def trajectories():
    common = dict(release_ball_frames=24, align_release_clock=True,
        release_azimuth_bias_deg=0., release_tolerance_m=.15,
        predict_ball_motion=False, release_speed_gain=1., release_min_s=1.5,
        release_joint_deg=35., governor_band_deg=8., target_force_scale=1.,
        target_radius_scale=1.)
    return {
        "baseline_mid": dict(**common, governor_start_deg=26., force_start_scale=.90,
            force_ramp_s=.20, radius_ramp_s=.20, envelope_force_scale=1.,
            envelope_radius_scale=1., spinup_s=1.70),
        "force_125": dict(**common, governor_start_deg=26., force_start_scale=.90,
            force_ramp_s=.20, radius_ramp_s=.20, envelope_force_scale=1.25,
            envelope_radius_scale=1., spinup_s=1.60),
        "force_140": dict(**common, governor_start_deg=25., force_start_scale=.85,
            force_ramp_s=.25, radius_ramp_s=.20, envelope_force_scale=1.40,
            envelope_radius_scale=1., spinup_s=1.55),
        "wide_115": dict(**common, governor_start_deg=25., force_start_scale=.90,
            force_ramp_s=.20, radius_ramp_s=.30, envelope_force_scale=1.15,
            envelope_radius_scale=1.12, spinup_s=1.70),
        "wide_130": dict(**common, governor_start_deg=25., force_start_scale=.85,
            force_ramp_s=.25, radius_ramp_s=.30, envelope_force_scale=1.30,
            envelope_radius_scale=1.12, spinup_s=1.60),
        "wide_140": dict(**common, governor_start_deg=24., force_start_scale=.80,
            force_ramp_s=.30, radius_ramp_s=.35, envelope_force_scale=1.40,
            envelope_radius_scale=1.18, spinup_s=1.55),
    }


def cells():
    return [(name, age, EnvelopeProbeConfig(**config, probe_name=name, probe_age_s=age))
        for name, config in trajectories().items() for age in PHASES]


def previous_seeds(exclude=None):
    from .release_calibration_rl import previous_seeds as v21_previous_seeds
    used = v21_previous_seeds()
    excluded = None if exclude is None else Path(exclude).resolve()
    for path in DEFAULT_OUT.rglob("collection_contract.json"):
        if excluded is None or path.parent.resolve() != excluded:
            used.update(row["seed"] for row in read_json(path)["manifest"]["scenarios"])
    return used


def worker(job):
    scenario, anchor_path, config_data = job
    torch.set_num_threads(1)
    anchor, payload = load_anchor(anchor_path)
    env = EnvelopeProbeEnv(scenario, ImprovedRecipe(**payload["recipe"]), anchor,
        EnvelopeProbeConfig(**config_data))
    observation = env.reset(scenario["seed"])
    started = time.perf_counter()
    while not env.done:
        observation, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    return dict(seed=scenario["seed"], scenario=scenario, result=result,
        wall_s=time.perf_counter() - started)


def collection_summary(rows):
    groups = {}
    for name in trajectories():
        selected = [row for row in rows
            if row["result"].get("calibration_probe", {}).get("probe_name") == name]
        usable = [row for row in selected
            if row["result"].get("calibration_probe", {}).get("scheduled") and
            row["result"].get("landing_xy") is not None]
        ranges = [float(np.linalg.norm(row["result"]["landing_xy"])) for row in usable]
        groups[name] = dict(episodes=len(selected), captured=sum(row["result"]["captured"] for row in selected),
            usable_rows=len(usable), max_joint_deg=max((row["result"]["max_joint_deg"] for row in selected), default=None),
            grip_broken=sum(row["result"]["grip_broken"] for row in selected),
            actual_range_min_m=min(ranges, default=None), actual_range_max_m=max(ranges, default=None))
    usable = sum(item["usable_rows"] for item in groups.values())
    actual_ranges = [float(np.linalg.norm(row["result"]["landing_xy"])) for row in rows
        if row["result"].get("landing_xy") is not None]
    target_ranges = [float(np.linalg.norm(row["result"]["target_xy"])) for row in rows]
    return dict(episodes=len(rows), usable_rows=usable, trajectories=groups,
        safety_pass=all(item["grip_broken"] == 0 and (item["max_joint_deg"] or 0.) <= 42.
            for item in groups.values()),
        actual_range_min_m=min(actual_ranges, default=None),
        actual_range_max_m=max(actual_ranges, default=None),
        target_range_min_m=min(target_ranges, default=None),
        target_range_max_m=max(target_ranges, default=None),
        range_overlap_pass=bool(actual_ranges and target_ranges and
            max(actual_ranges) >= min(target_ranges) - .10))


def collect(args):
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    campaign_cells = cells()
    total = len(campaign_cells) * args.episodes_per_cell
    manifest = scenarios(total, args.seed, args.design_seed, recipe)
    assignments = []
    for index, scenario in enumerate(manifest["scenarios"]):
        name, age, config = campaign_cells[index % len(campaign_cells)]
        assignments.append(dict(scenario=scenario, trajectory=name,
            probe_age_s=age, config=asdict(config)))
    identity = dict(schema=SCHEMA, source_hash=source_hash(),
        anchor_sha256=file_hash(args.init), manifest=manifest,
        episodes_per_cell=args.episodes_per_cell, phases_s=list(PHASES),
        trajectories=trajectories(), purpose="expanded_safe_trajectory_release_calibration")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "collection_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("V22 collection contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("V22 collection seeds overlap an earlier experiment")
    write_json(contract_path, identity)
    output = args.out / "controlled_release_episodes.json"
    rows = read_json(output) if output.exists() else []
    completed = {row["seed"] for row in rows}
    jobs = [(item["scenario"], str(args.init.resolve()), item["config"])
        for item in assignments if item["scenario"]["seed"] not in completed]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output, sorted(rows, key=lambda item: item["seed"]))
            print(dict(completed=len(rows), total=total, seed=row["seed"],
                released=row["result"]["released"]), flush=True)
    rows = sorted(rows, key=lambda item: item["seed"])
    write_json(output, rows)
    summary = dict(schema=SCHEMA, source_hash=source_hash(),
        output=str(output), **collection_summary(rows))
    write_json(args.out / "collection_summary.json", summary)
    print(summary, flush=True)


def fit(args):
    contract_path = args.out / "collection_contract.json"
    episode_path = args.out / "controlled_release_episodes.json"
    if not contract_path.exists() or not episode_path.exists():
        raise ValueError("Run V22 Collect successfully before Fit")
    contract = read_json(contract_path)
    if (contract.get("schema") != SCHEMA or contract.get("source_hash") != source_hash() or
            contract.get("anchor_sha256") != file_hash(args.init)):
        raise ValueError("V22 collection does not match current source and BC anchor")
    data = calibration_rows(read_json(episode_path))
    if len(data) < 150:
        raise ValueError(f"Need at least 150 usable V22 rows; found {len(data)}")
    cell_count = len(contract["phases_s"]) * len(contract["trajectories"])
    fold_reports, predictions = [], []
    for fold in range(CV_FOLDS):
        train = [row for row in data if stratified_fold(row, cell_count) != fold]
        holdout = [row for row in data if stratified_fold(row, cell_count) == fold]
        if len(train) < 120 or len(holdout) < 25:
            raise ValueError(f"Fold {fold} has insufficient train/holdout rows: {len(train)}/{len(holdout)}")
        x_train = np.vstack([row["features"] for row in train])
        raw_train = np.vstack([row["raw_landing"] for row in train])
        actual_train = np.vstack([row["actual_landing"] for row in train])
        model = fit_ridge(x_train, actual_train - raw_train, args.ridge_lambda)
        metrics, fold_predictions = evaluate_model(model, holdout)
        fold_reports.append(dict(fold=fold, train_rows=len(train), holdout_rows=len(holdout), **metrics))
        predictions.extend(fold_predictions)
    pooled = prediction_metrics(predictions)
    by_trajectory = grouped_metrics(predictions, "trajectory")
    by_phase = grouped_metrics(predictions, "probe_age_s")
    checks = acceptance_checks(pooled, fold_reports, by_trajectory, by_phase)
    cell_rows = {f"{name}|{age:g}": sum(row["trajectory"] == name and row["probe_age_s"] == age
        for row in data) for name in contract["trajectories"] for age in contract["phases_s"]}
    checks["every_cell_rows_ge_4"] = all(count >= 4 for count in cell_rows.values())
    accepted = all(checks.values())
    all_features = np.vstack([row["features"] for row in data])
    all_raw = np.vstack([row["raw_landing"] for row in data])
    all_actual = np.vstack([row["actual_landing"] for row in data])
    model = fit_ridge(all_features, all_actual - all_raw, args.ridge_lambda)
    payload = dict(**model.to_dict(), campaign_schema=SCHEMA, fit_source_hash=source_hash(),
        collection_source_hash=contract["source_hash"], anchor_sha256=file_hash(args.init),
        collection_contract_sha256=file_hash(contract_path), deployment_fit_rows=len(data),
        validation_protocol="trajectory_phase_stratified_5fold_v22",
        cross_validation=dict(pooled=pooled, folds=fold_reports,
            by_trajectory=by_trajectory, by_phase=by_phase, cell_rows=cell_rows),
        checks=checks, accepted=accepted)
    if payload["schema"] != MODEL_SCHEMA:
        raise ValueError("unexpected landing model schema")
    output = args.out / "release_calibration_v22.json"
    write_json(output, payload)
    print(dict(accepted=accepted, deployment_fit_rows=len(data), pooled=pooled,
        checks=checks, output=str(output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    collect_parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "campaign_v1")
    collect_parser.add_argument("--episodes-per-cell", type=int, default=8)
    collect_parser.add_argument("--workers", type=int, default=8)
    collect_parser.add_argument("--seed", type=int, default=24200001)
    collect_parser.add_argument("--design-seed", type=int, default=20261001)
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
