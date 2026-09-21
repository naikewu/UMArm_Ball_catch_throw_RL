"""V22 screen and independently validate the expanded-envelope online teacher."""
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
from .envelope_teacher_env import SCHEMA, EnvelopeTeacherConfig, EnvelopeTeacherEnv
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv
from .release_calibration import RidgeLandingCalibration
from .safe_continuous_rl import report as v19_report
from .trajectory_envelope_rl import (SCHEMA as CAMPAIGN_SCHEMA, previous_seeds as envelope_previous_seeds,
    source_hash as envelope_source_hash, trajectories as envelope_trajectories)

DEFAULT_OUT = Path("teacher_runs/v22_envelope_teacher")
DEFAULT_CALIBRATION = Path("teacher_runs/v22_trajectory_envelope/campaign_v1/release_calibration_v22.json")


def source_hash():
    digest = hashlib.sha256(envelope_source_hash().encode())
    for name in ("envelope_teacher_env.py", "envelope_teacher_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_calibration(path, anchor):
    payload = read_json(path)
    if (not payload.get("accepted") or payload.get("campaign_schema") != CAMPAIGN_SCHEMA or
            payload.get("validation_protocol") != "trajectory_phase_stratified_5fold_v22" or
            payload.get("fit_source_hash") != envelope_source_hash() or
            payload.get("anchor_sha256") != file_hash(anchor) or
            not all(payload.get("checks", {}).values())):
        raise ValueError("V22 trajectory calibration is absent, stale, or not fully accepted")
    return RidgeLandingCalibration.from_dict(payload), payload


def previous_seeds(exclude=None):
    used = envelope_previous_seeds()
    excluded = None if exclude is None else Path(exclude).resolve()
    for contract_name in ("screen_contract.json", "validation_contract.json"):
        for path in DEFAULT_OUT.rglob(contract_name):
            if excluded is None or not path.resolve().is_relative_to(excluded):
                used.update(row["seed"] for row in read_json(path)["manifest"]["scenarios"])
    return used


def worker(job):
    scenario, anchor_path, calibration_path, config_data = job
    torch.set_num_threads(1)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, payload = load_anchor(anchor_path)
    recipe = ImprovedRecipe(**payload["recipe"])
    if config_data is None:
        env = ImprovedTeacherEnv(scenario, recipe)
    else:
        calibration, _ = load_calibration(calibration_path, anchor_path)
        env = EnvelopeTeacherEnv(scenario, recipe, anchor, calibration,
            EnvelopeTeacherConfig(**config_data))
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


def evaluate_rows(pool, manifest, anchor, calibration, config):
    data = None if config is None else asdict(config)
    futures = [pool.submit(worker, (scenario, str(anchor.resolve()), str(calibration.resolve()), data))
        for scenario in manifest]
    rows = []
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        print(dict(completed=len(rows), total=len(futures), seed=row["seed"],
            hit15=row["result"]["hit15"]), flush=True)
    return sorted(rows, key=lambda row: row["seed"])


def report(rows, baseline):
    result = v19_report(rows, baseline)
    audits = [row["result"].get("calibrated_release", {}) for row in rows]
    scheduled = sum(bool(audit.get("scheduled")) for audit in audits)
    ages = [audit["release_age_s"] for audit in audits if audit.get("scheduled")]
    result["calibrated_release_scheduled"] = scheduled
    result["mean_release_age_s"] = float(np.mean(ages)) if ages else None
    result["max_release_age_s"] = float(np.max(ages)) if ages else None
    result["checks"]["calibrated_release_coverage"] = scheduled >= result["summary"]["released"]
    result["checks"]["release_window"] = all(not audit.get("scheduled") or
        2. <= audit["release_age_s"] <= 18. for audit in audits)
    result["checks"]["calibration_age_cap"] = all(not audit.get("scheduled") or
        audit["feature_age_s"] <= 16. for audit in audits)
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
    return (result["summary"]["hit15"], result["summary"]["released"],
        -result["max_joint_deg"], -(result["summary"]["mean_landing_error_m"] or 1e9))


def variants():
    bases = envelope_trajectories()
    tolerances = dict(baseline_mid=.06, force_125=.08, force_140=.10,
        wide_115=.08, wide_130=.10, wide_140=.10)
    result = {}
    for name, base in bases.items():
        values = dict(base, release_min_s=2., release_max_s=18.,
            calibration_age_cap_s=16., feature_z_limit=4.5,
            release_tolerance_m=tolerances[name], release_confirm_ticks=1,
            release_immediate_m=.04)
        result[name + "_dual"] = EnvelopeTeacherConfig(**values)
    return result


def screen(args):
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    _, calibration_payload = load_calibration(args.calibration, args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    configurations = variants()
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), calibration_fit_source=calibration_payload["fit_source_hash"],
        manifest=manifest, variants={name: asdict(config) for name, config in configurations.items()},
        purpose="development_only_expanded_envelope_teacher")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "screen_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("V22 Teacher screen contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("V22 Teacher screen seeds overlap an earlier experiment")
    write_json(contract_path, identity)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        bc_path = args.out / "v15_bc_episodes.json"
        bc = read_json(bc_path) if bc_path.exists() else evaluate_rows(
            pool, manifest["scenarios"], args.init, args.calibration, None)
        write_json(bc_path, bc)
        for name, configuration in configurations.items():
            path = args.out / (name + "_episodes.json")
            rows = read_json(path) if path.exists() else evaluate_rows(
                pool, manifest["scenarios"], args.init, args.calibration, configuration)
            write_json(path, rows)
            checked = report(rows, bc)
            results[name] = dict(config=asdict(configuration), report=checked)
            write_json(args.out / "comparison.json", results)
            print(dict(variant=name, eligible=checked["eligible"],
                task_nonregression=checked["task_nonregression"], summary=checked["summary"]), flush=True)
    eligible = [(name, data) for name, data in results.items()
        if data["report"]["eligible"] and data["report"]["task_nonregression"]]
    if eligible:
        name, selected = max(eligible, key=lambda item: selection_score(item[1]["report"]))
        write_json(args.out / "selected_config.json", dict(schema=SCHEMA, source_hash=source_hash(),
            anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
            variant=name, config=selected["config"], eligible=True,
            development_manifest=manifest, development_episodes=args.episodes,
            primary_quality_improved=selected["report"]["primary_quality_improved"]))
        print(dict(selected=name, authorized_for_independent_validation=True), flush=True)
    else:
        print("No V22 envelope teacher passed. Do not validate, collect BC data, or start PPO.", flush=True)


def validate(args):
    selected_path = args.screen / "selected_config.json"
    if not selected_path.exists():
        raise ValueError("Screen has not selected a V22 envelope teacher")
    selected = read_json(selected_path)
    load_calibration(args.calibration, args.init)
    if (selected.get("schema") != SCHEMA or not selected.get("eligible") or
            selected.get("source_hash") != source_hash() or
            selected.get("anchor_sha256") != file_hash(args.init) or
            selected.get("calibration_sha256") != file_hash(args.calibration)):
        raise ValueError("Selected V22 teacher does not match current source, BC, and calibration")
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), selected_config_sha256=file_hash(selected_path),
        manifest=manifest, purpose="independent_expanded_envelope_teacher_validation")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "validation_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("V22 validation contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("V22 validation seeds overlap earlier experiments")
    write_json(contract_path, identity)
    configuration = EnvelopeTeacherConfig(**selected["config"])
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        bc_path, candidate_path = args.out / "v15_bc_episodes.json", args.out / "v22_teacher_episodes.json"
        bc = read_json(bc_path) if bc_path.exists() else evaluate_rows(
            pool, manifest["scenarios"], args.init, args.calibration, None)
        write_json(bc_path, bc)
        candidate = read_json(candidate_path) if candidate_path.exists() else evaluate_rows(
            pool, manifest["scenarios"], args.init, args.calibration, configuration)
        write_json(candidate_path, candidate)
    checked = report(candidate, bc)
    qualified = checked["absolute_generalization_pass"]
    write_json(args.out / "comparison.json", dict(selected=selected, report=checked, qualified=qualified))
    if qualified:
        write_json(args.screen / "qualified_teacher.json", dict(**selected,
            validation_contract=identity, qualified=True, independent_report=checked))
    print(dict(qualified=qualified, summary=checked["summary"], checks=checked["checks"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, output, episodes, seed, design_seed in (
            ("screen", DEFAULT_OUT / "screen_v1", 20, 25100001, 20261002),
            ("validate", DEFAULT_OUT / "screen_v1" / "validation_80", 80, 25200001, 20261003)):
        command = sub.add_parser(name)
        command.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        if name == "validate":
            command.add_argument("--screen", type=Path, default=DEFAULT_OUT / "screen_v1")
        command.add_argument("--out", type=Path, default=output)
        command.add_argument("--episodes", type=int, default=episodes)
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--seed", type=int, default=seed)
        command.add_argument("--design-seed", type=int, default=design_seed)
    args = parser.parse_args()
    if args.episodes < 5 or args.workers < 1:
        parser.error("episodes must be >=5 and workers must be positive")
    if args.command == "validate" and args.episodes < 80:
        parser.error("Independent validation requires at least 80 episodes")
    (screen if args.command == "screen" else validate)(args)


if __name__ == "__main__":
    main()
