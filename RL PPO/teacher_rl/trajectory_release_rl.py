"""V20 screen and independently validate a safe trajectory/release teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .trajectory_release_env import SCHEMA, TrajectoryReleaseConfig, TrajectoryReleaseEnv
from .safe_continuous_rl import (previous_seeds as v19_previous_seeds, report as v19_report,
    source_hash as v19_source_hash)
from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv
from .data import write_json

DEFAULT_OUT = Path("teacher_runs/v20_trajectory_release")


def source_hash():
    digest = hashlib.sha256(v19_source_hash().encode())
    for name in ("trajectory_release_env.py", "trajectory_release_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def previous_seeds(exclude=None):
    used = v19_previous_seeds()
    excluded = None if exclude is None else Path(exclude).resolve()
    for contract_name in ("screen_contract.json", "validation_contract.json"):
        for path in DEFAULT_OUT.rglob(contract_name):
            if excluded is None or not path.resolve().is_relative_to(excluded):
                used.update(row["seed"] for row in read_json(path)["manifest"]["scenarios"])
    return used


def worker(job):
    scenario, anchor_path, config_data = job
    torch.set_num_threads(1)
    anchor, payload = load_anchor(anchor_path)
    recipe = ImprovedRecipe(**payload["recipe"])
    env = (ImprovedTeacherEnv(scenario, recipe) if config_data is None else
        TrajectoryReleaseEnv(scenario, recipe, anchor, TrajectoryReleaseConfig(**config_data)))
    observation = env.reset(scenario["seed"])
    started = time.perf_counter()
    while not env.done:
        if config_data is None:
            with torch.no_grad():
                action = anchor.distribution(torch.tensor(observation[None])).mean.tanh()[0].numpy()
        else:
            action = np.zeros(7, dtype=np.float32)
        observation, _, _, result = env.step(action)
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or
        env.capture_physics_time is None else env.controller.t_hand - env.capture_physics_time)
    result.setdefault("catch_to_release_s", None if result["release_time"] is None or
        result["capture_time"] is None else result["release_time"] - result["capture_time"])
    return dict(seed=scenario["seed"], scenario=scenario, result=result,
        wall_s=time.perf_counter() - started)


def evaluate_rows(pool, manifest, anchor, config):
    data = None if config is None else asdict(config)
    futures = [pool.submit(worker, (scenario, str(anchor.resolve()), data)) for scenario in manifest]
    rows = []
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        print(dict(completed=len(rows), total=len(futures), seed=row["seed"],
            hit15=row["result"]["hit15"]), flush=True)
    return sorted(rows, key=lambda row: row["seed"])


def report(rows, baseline):
    result = v19_report(rows, baseline)
    scheduled = sum(bool(row["result"].get("predictive_release", {}).get("scheduled")) for row in rows)
    result["predictive_release_scheduled"] = scheduled
    result["checks"]["predictive_release_coverage"] = scheduled >= result["summary"]["released"]
    result["eligible"] = all(result["checks"].values())
    result["task_nonregression"] = (result["summary"]["hit15"] >= result["baseline"]["hit15"] and
        result["summary"]["captured"] >= result["baseline"]["captured"] and
        result["summary"]["released"] >= result["baseline"]["released"])
    bins = result["parameter_bins"]
    result["absolute_generalization_pass"] = bool(result["eligible"] and result["task_nonregression"] and
        len(rows) >= 80 and result["worst_parameter_bin_hit15_rate"] >= .8 and
        all(group["episodes"] >= 10 for groups in bins.values() for group in groups))
    return result


def selection_score(result):
    return (result["summary"]["hit15"], result["summary"]["captured"],
        -result["max_joint_deg"], -(result["summary"]["mean_landing_error_m"] or 1e9))


def variants():
    common = dict(release_ball_frames=24, align_release_clock=True, release_azimuth_bias_deg=0.,
        release_tolerance_m=.04, predict_ball_motion=False)
    return {
        "q26_speed82_fast": TrajectoryReleaseConfig(**common, governor_start_deg=26., governor_band_deg=8.,
            force_start_scale=.95, force_ramp_s=.15, radius_ramp_s=.15, target_force_scale=1.05,
            target_radius_scale=.98, spinup_s=1.45, release_min_s=2.05, release_joint_deg=35., release_speed_gain=.82),
        "q26_speed86_nominal": TrajectoryReleaseConfig(**common, governor_start_deg=26., governor_band_deg=8.,
            force_start_scale=.90, force_ramp_s=.20, radius_ramp_s=.20, target_force_scale=1.00,
            target_radius_scale=1.00, spinup_s=1.70, release_min_s=2.25, release_joint_deg=35., release_speed_gain=.86),
        "q28_speed82_fast": TrajectoryReleaseConfig(**common, governor_start_deg=28., governor_band_deg=7.,
            force_start_scale=.95, force_ramp_s=.15, radius_ramp_s=.15, target_force_scale=1.03,
            target_radius_scale=.98, spinup_s=1.50, release_min_s=2.05, release_joint_deg=36., release_speed_gain=.82),
        "q28_speed86_radius": TrajectoryReleaseConfig(**common, governor_start_deg=28., governor_band_deg=7.,
            force_start_scale=.90, force_ramp_s=.20, radius_ramp_s=.25, target_force_scale=1.00,
            target_radius_scale=1.04, spinup_s=1.70, release_min_s=2.25, release_joint_deg=36., release_speed_gain=.86),
        "q28_speed90_balanced": TrajectoryReleaseConfig(**common, governor_start_deg=28., governor_band_deg=6.,
            force_start_scale=.95, force_ramp_s=.18, radius_ramp_s=.20, target_force_scale=1.05,
            target_radius_scale=1.00, spinup_s=1.60, release_min_s=2.15, release_joint_deg=36., release_speed_gain=.90),
    }


def screen(args):
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    configurations = variants()
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        manifest=manifest, variants={name: asdict(config) for name, config in configurations.items()},
        purpose="development_only_safe_trajectory_and_predictive_release_teacher")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "screen_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("Screen contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("Screen seeds overlap an earlier V15-V20 experiment")
    write_json(contract_path, identity)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        bc_path = args.out / "v15_bc_episodes.json"
        bc = read_json(bc_path) if bc_path.exists() else evaluate_rows(pool, manifest["scenarios"], args.init, None)
        write_json(bc_path, bc)
        for name, config in configurations.items():
            path = args.out / (name + "_episodes.json")
            rows = read_json(path) if path.exists() else evaluate_rows(pool, manifest["scenarios"], args.init, config)
            write_json(path, rows)
            checked = report(rows, bc)
            results[name] = dict(config=asdict(config), report=checked)
            write_json(args.out / "comparison.json", results)
            print(dict(variant=name, eligible=checked["eligible"],
                task_nonregression=checked["task_nonregression"], summary=checked["summary"]), flush=True)
    eligible = [(name, data) for name, data in results.items()
        if data["report"]["eligible"] and data["report"]["task_nonregression"]]
    if eligible:
        name, selected = max(eligible, key=lambda item: selection_score(item[1]["report"]))
        write_json(args.out / "selected_config.json", dict(schema=SCHEMA, source_hash=source_hash(),
            anchor_sha256=file_hash(args.init), variant=name, config=selected["config"], eligible=True,
            development_manifest=manifest, development_episodes=args.episodes,
            primary_quality_improved=selected["report"]["primary_quality_improved"]))
        print(dict(selected=name, authorized_for_independent_validation=True), flush=True)
    else:
        print("No V20 teacher passed. Do not collect data or start RL training.", flush=True)


def validate(args):
    selected_path = args.screen / "selected_config.json"
    if not selected_path.exists():
        raise ValueError("Screen has not produced selected_config.json; no V20 teacher is authorized for validation")
    selected = read_json(selected_path)
    if (selected.get("schema") != SCHEMA or not selected.get("eligible") or
            selected.get("source_hash") != source_hash() or selected.get("anchor_sha256") != file_hash(args.init)):
        raise ValueError("Selected V20 configuration does not match the current source and BC anchor")
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        selected_config_sha256=file_hash(selected_path), manifest=manifest,
        purpose="independent_trajectory_and_predictive_release_teacher_validation")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "validation_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("Validation contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("Validation seeds overlap earlier V15-V20 development or evaluation")
    write_json(contract_path, identity)
    config = TrajectoryReleaseConfig(**selected["config"])
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        bc_path, candidate_path = args.out / "v15_bc_episodes.json", args.out / "v20_episodes.json"
        bc = read_json(bc_path) if bc_path.exists() else evaluate_rows(pool, manifest["scenarios"], args.init, None)
        write_json(bc_path, bc)
        candidate = read_json(candidate_path) if candidate_path.exists() else evaluate_rows(
            pool, manifest["scenarios"], args.init, config)
        write_json(candidate_path, candidate)
    checked = report(candidate, bc)
    qualified = checked["absolute_generalization_pass"]
    write_json(args.out / "comparison.json", dict(selected=selected, report=checked, qualified=qualified))
    if qualified:
        write_json(args.screen / "qualified_teacher.json", dict(**selected, validation_contract=identity,
            qualified=True, independent_report=checked))
    print(dict(qualified=qualified, summary=checked["summary"], checks=checked["checks"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command_name, default_out, default_episodes, default_seed in (
            ("screen", DEFAULT_OUT / "screen_v1", 20, 20100001),
            ("validate", DEFAULT_OUT / "screen_v1" / "validation_80", 80, 20200001)):
        command = sub.add_parser(command_name)
        command.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
        if command_name == "validate":
            command.add_argument("--screen", type=Path, default=DEFAULT_OUT / "screen_v1")
        command.add_argument("--out", type=Path, default=default_out)
        for name, default in (("episodes", default_episodes), ("workers", 8), ("seed", default_seed),
                ("design-seed", 20260923 if command_name == "screen" else 20260924)):
            command.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.episodes < 5 or args.workers < 1:
        parser.error("episodes must be >=5 and workers must be positive")
    if args.command == "validate" and args.episodes < 80:
        parser.error("Independent validation requires at least 80 episodes")
    if args.command == "validate" and not (args.screen / "selected_config.json").exists():
        parser.error("Screen has not produced selected_config.json; no V20 teacher is authorized for validation")
    (screen if args.command == "screen" else validate)(args)


if __name__ == "__main__":
    main()
