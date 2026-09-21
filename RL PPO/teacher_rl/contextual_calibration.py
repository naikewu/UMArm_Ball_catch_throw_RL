"""V23 measured-state release calibration with identical estimator warm-up.

The V21/V22 controlled probe originally began filling the ball fit and angular
rate history only at its requested release age. The online teacher instead
accumulated history while searching. V23 warms both paths from handover, leaving
the physical plant, grip, actuator bounds, and actual landing score unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import asdict
import hashlib

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .calibrated_teacher_env import CalibratedRelease
from .contextual_env import ContextualEnv
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe
from .release_calibration import RidgeLandingCalibration, fit_ridge
from .release_calibration_env import ProbeRelease
from .release_calibration_rl import (calibration_rows, stratified_fold, evaluate_model,
    prediction_metrics, grouped_metrics, acceptance_checks)
from .trajectory_envelope_rl import (EnvelopeProbeEnv, EnvelopeProbeConfig, PHASES,
    trajectories, source_hash as v22_source_hash, collection_summary)

SCHEMA = "can_warmed_release_calibration_v23"
DEFAULT_OUT = Path("teacher_runs/v23_contextual/warm_calibration_v1")


def source_hash():
    digest = hashlib.sha256(v22_source_hash().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def warm_before_release(original, controller, observation, release_age, last_time):
    """Advance sensor-only histories once/tick before the release search opens."""
    if (observation["t"] == last_time or not observation["held"] or
            original.t_release is not None or controller.t_hand is None):
        return last_time
    age = float(observation["t_win"] - controller.t_hand)
    if age >= release_age:
        return last_time
    position, velocity = original.ball_state(observation)
    if (np.isfinite(np.r_[position, velocity]).all() and
            getattr(original.ball_state, "lead", None) is not None):
        original._rates(float(observation["t_win"]), velocity)
    return observation["t"]


class WarmProbeRelease(ProbeRelease):
    def __init__(self, original, controller, config):
        super().__init__(original, controller, config)
        object.__setattr__(self, "warm_time", None)

    def __call__(self, plant, observation):
        stamp = warm_before_release(self.original, self.controller, observation,
            self.config.probe_age_s, self.warm_time)
        object.__setattr__(self, "warm_time", stamp)
        super().__call__(plant, observation)
        if self.audit["scheduled"]:
            self.audit["warm_history"] = dict(ball_frames=len(self.original.ball_state.fit.buf),
                rate_frames=len(self.original.r_hist), elevation_frames=len(self.original.e_hist),
                omega=float(self.original.omega), edot=float(self.original.edot))


class WarmCalibratedRelease(CalibratedRelease):
    ORIGIN = "v23_warmed_calibrated_release"

    def __init__(self, original, controller, config, calibration):
        super().__init__(original, controller, config, calibration)
        object.__setattr__(self, "warm_time", None)

    def refresh(self, observation):
        stamp = warm_before_release(self.original, self.controller, observation,
            self.config.release_min_s, self.warm_time)
        object.__setattr__(self, "warm_time", stamp)
        super().refresh(observation)


class WarmProbeEnv(EnvelopeProbeEnv):
    def reset(self, seed):
        super().reset(seed)
        self.releaser = WarmProbeRelease(self.releaser.original, self.handover, self.continuous)
        return self.observation()


class WarmContextualEnv(ContextualEnv):
    def reset(self, seed):
        super().reset(seed)
        self.releaser = WarmCalibratedRelease(self.releaser.original, self.handover,
                                            self.continuous, self.release_calibration)
        return self.observation()


def worker(job):
    scenario, anchor_path, config = job
    torch.set_num_threads(1)
    anchor, payload = load_anchor(Path(anchor_path))
    env = WarmProbeEnv(scenario, ImprovedRecipe(**payload["recipe"]), anchor, EnvelopeProbeConfig(**config))
    env.reset(scenario["seed"])
    while not env.done:
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    return dict(seed=scenario["seed"], scenario=scenario, result=result)


def collect(args):
    from .contextual_rl import previous_seeds
    _, payload = load_anchor(args.init)
    recipes = trajectories()
    cells = [(name, age, EnvelopeProbeConfig(**config, probe_name=name, probe_age_s=age))
             for name, config in recipes.items() for age in PHASES]
    manifest = scenarios(len(cells) * args.repeats, args.seed, args.design_seed,
                         ImprovedRecipe(**payload["recipe"]))
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        manifest=manifest, trajectories=recipes, phases_s=list(PHASES), repeats=args.repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "collection_contract.json"
    if path.exists():
        if read_json(path) != identity:
            raise ValueError("Warm calibration collection changed; use a new directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("Warm calibration seeds overlap existing experiments")
    write_json(path, identity)
    output = args.out / "episodes"
    output.mkdir(exist_ok=True)
    rows, jobs = {}, []
    for index, scenario in enumerate(manifest["scenarios"]):
        name, age, config = cells[index % len(cells)]
        path = output / f"episode_{scenario['seed']}.json"
        if path.exists():
            rows[scenario["seed"]] = read_json(path)
        else:
            jobs.append((scenario, str(args.init.resolve()), asdict(config)))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            write_json(output / f"episode_{row['seed']}.json", row)
            if len(rows) % 10 == 0 or len(rows) == len(manifest["scenarios"]):
                print(dict(warm_calibration_completed=len(rows), total=len(manifest["scenarios"])), flush=True)
    ordered = [rows[scenario["seed"]] for scenario in manifest["scenarios"]]
    write_json(args.out / "controlled_release_episodes.json", ordered)
    summary = collection_summary(ordered)
    summary["fully_warmed_labels"] = sum(row["result"].get("calibration_probe", {}).get(
        "warm_history", {}).get("rate_frames", 0) >= 20 for row in ordered)
    summary["joint_limit_36_pass"] = all(row["result"]["max_joint_deg"] <= 36. for row in ordered)
    write_json(args.out / "collection_summary.json", summary)
    print(summary, flush=True)


def fit(args):
    contract = read_json(args.out / "collection_contract.json")
    if contract["source_hash"] != source_hash() or contract["anchor_sha256"] != file_hash(args.init):
        raise ValueError("Warm calibration data source or V15 anchor changed")
    rows = read_json(args.out / "controlled_release_episodes.json")
    data = calibration_rows(rows)
    if len(data) < 150:
        raise ValueError("Need at least 150 fully warmed calibration rows")
    by_seed = {row["seed"]: row for row in rows}
    if any(by_seed[row["seed"]]["result"]["calibration_probe"]["warm_history"]["rate_frames"] < 20 for row in data):
        raise ValueError("Controlled release includes a cold estimator")
    predictions, folds = [], []
    cells = len(contract["trajectories"]) * len(contract["phases_s"])
    for fold in range(5):
        train = [row for row in data if stratified_fold(row, cells) != fold]
        holdout = [row for row in data if stratified_fold(row, cells) == fold]
        model = fit_ridge(np.vstack([row["features"] for row in train]),
                         np.vstack([row["actual_landing"] - row["raw_landing"] for row in train]))
        metrics, predicted = evaluate_model(model, holdout)
        folds.append(dict(fold=fold, **metrics))
        predictions.extend(predicted)
    pooled = prediction_metrics(predictions)
    by_trajectory, by_phase = grouped_metrics(predictions, "trajectory"), grouped_metrics(predictions, "probe_age_s")
    checks = acceptance_checks(pooled, folds, by_trajectory, by_phase)
    counts = {f"{name}|{age:g}": sum(row["trajectory"] == name and row["probe_age_s"] == age for row in data)
              for name in contract["trajectories"] for age in contract["phases_s"]}
    checks["every_cell_rows_ge_4"] = all(count >= 4 for count in counts.values())
    model = fit_ridge(np.vstack([row["features"] for row in data]),
                     np.vstack([row["actual_landing"] - row["raw_landing"] for row in data]))
    payload = dict(**model.to_dict(), campaign_schema=SCHEMA, fit_source_hash=source_hash(),
        anchor_sha256=file_hash(args.init), collection_contract_sha256=file_hash(args.out / "collection_contract.json"),
        deployment_fit_rows=len(data), validation_protocol="warmed_trajectory_phase_stratified_5fold_v23",
        cross_validation=dict(pooled=pooled, folds=folds, by_trajectory=by_trajectory, by_phase=by_phase, cell_rows=counts),
        checks=checks, accepted=all(checks.values()), calibration_seeds=[row["seed"] for row in rows])
    write_json(args.out / "release_calibration_v23.json", payload)
    print(dict(accepted=payload["accepted"], pooled=pooled, checks=checks), flush=True)


def load_calibration(path, anchor):
    payload = read_json(path)
    if (payload.get("campaign_schema") != SCHEMA or not payload.get("accepted") or
            payload.get("fit_source_hash") != source_hash() or payload.get("anchor_sha256") != file_hash(anchor) or
            not all(payload.get("checks", {}).values())):
        raise ValueError("V23 warmed calibration is missing, stale, or failed validation")
    return RidgeLandingCalibration.from_dict(payload), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "fit"))
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=27100001)
    parser.add_argument("--design-seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("repeats and workers must be positive")
    (collect if args.command == "collect" else fit)(args)


if __name__ == "__main__":
    main()
