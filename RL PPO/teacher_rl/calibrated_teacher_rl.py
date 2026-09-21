"""V21 screen and independently validate the calibrated online teacher."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .calibrated_teacher_env import SCHEMA, CalibratedTeacherConfig, CalibratedTeacherEnv
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv
from .release_calibration import RidgeLandingCalibration
from .release_calibration_rl import (previous_seeds as v21_previous_seeds,
    source_hash as calibration_source_hash, trajectories as calibration_trajectories)
from .safe_continuous_rl import report as v19_report

DEFAULT_OUT = Path("teacher_runs/v21_calibrated_teacher")
DEFAULT_CALIBRATION = Path("teacher_runs/v21_release_calibration/campaign_v1/release_calibration_stratified.json")


def source_hash():
    digest = hashlib.sha256(calibration_source_hash().encode())
    for name in ("calibrated_teacher_env.py", "calibrated_teacher_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_calibration(path, anchor):
    payload = read_json(path)
    if (not payload.get("accepted") or payload.get("validation_protocol") != "trajectory_phase_stratified_5fold_v2" or
            payload.get("fit_source_hash") != calibration_source_hash() or
            payload.get("anchor_sha256") != file_hash(anchor) or not all(payload.get("checks", {}).values())):
        raise ValueError("V21 calibration is absent, stale, or not fully accepted")
    return RidgeLandingCalibration.from_dict(payload), payload


def previous_seeds(exclude=None):
    used = v21_previous_seeds()
    excluded = None if exclude is None else Path(exclude).resolve()
    for contract_name in ("screen_contract.json", "validation_contract.json"):
        for path in DEFAULT_OUT.rglob(contract_name):
            if excluded is None or not path.resolve().is_relative_to(excluded):
                used.update(row["seed"] for row in read_json(path)["manifest"]["scenarios"])
    return used


def worker(job):
    scenario, anchor_path, calibration_path, variant_data = job
    torch.set_num_threads(1)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, payload = load_anchor(anchor_path)
    recipe = ImprovedRecipe(**payload["recipe"])
    if variant_data is None:
        env = ImprovedTeacherEnv(scenario, recipe)
        selected_branch = None
    else:
        calibration, _ = load_calibration(calibration_path, anchor_path)
        config, selected_branch = resolve_variant(variant_data, scenario)
        env = CalibratedTeacherEnv(scenario, recipe, anchor, calibration,
            config)
    observation = env.reset(scenario["seed"])
    started = time.perf_counter()
    while not env.done:
        if variant_data is None:
            with torch.no_grad():
                action = anchor.distribution(torch.tensor(observation[None])).mean.tanh()[0].numpy()
        else:
            action = np.zeros(7, dtype=np.float32)
        observation, _, _, result = env.step(action)
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or
        env.capture_physics_time is None else env.controller.t_hand - env.capture_physics_time)
    result.setdefault("catch_to_release_s", None if result["release_time"] is None or
        result["capture_time"] is None else result["release_time"] - result["capture_time"])
    if selected_branch is not None:
        result["calibrated_teacher_branch"] = selected_branch
    return dict(seed=scenario["seed"], scenario=scenario, result=result,
        wall_s=time.perf_counter() - started)


def evaluate_rows(pool, manifest, anchor, calibration, config):
    futures = [pool.submit(worker, (scenario, str(anchor.resolve()), str(calibration.resolve()), config))
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
    release_ages = [audit["release_age_s"] for audit in audits if audit.get("scheduled")]
    result["calibrated_release_scheduled"] = scheduled
    result["mean_release_age_s"] = (float(np.mean(release_ages)) if release_ages else None)
    result["max_release_age_s"] = (float(np.max(release_ages)) if release_ages else None)
    result["checks"]["calibrated_release_coverage"] = scheduled >= result["summary"]["released"]
    result["checks"]["release_window"] = all(not audit.get("scheduled") or
        2. <= audit["release_age_s"] <= 18. for audit in audits)
    result["checks"]["calibration_age_cap"] = all(not audit.get("scheduled") or
        audit["feature_age_s"] <= 4. for audit in audits)
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


def resolve_variant(specification, scenario):
    selector = specification.get("selector")
    if selector == "constant":
        return CalibratedTeacherConfig(**specification["config"]), "constant"
    if selector != "target_distance":
        raise ValueError("unknown calibrated teacher selector")
    threshold = float(specification["threshold_m"])
    if not 1.2 <= threshold <= 1.8:
        raise ValueError("target-distance selector threshold is out of range")
    branch = "near" if float(scenario["target_distance"]) < threshold else "far"
    return CalibratedTeacherConfig(**specification[branch]), branch


def variants():
    bases = calibration_trajectories()
    def config(name, tolerance):
        values = dict(bases[name], release_min_s=2., release_tolerance_m=tolerance,
            release_max_s=18., calibration_age_cap_s=4., feature_z_limit=4.5,
            release_confirm_ticks=1, release_immediate_m=.04)
        return CalibratedTeacherConfig(**values)
    def constant(configuration):
        return dict(selector="constant", config=asdict(configuration))
    def adaptive(threshold, near_tolerance):
        return dict(selector="target_distance", threshold_m=threshold,
            near=asdict(config("q28_safe", near_tolerance)),
            far=asdict(config("q26_fast", .06)))
    return {
        "q26_fast_t06_dual": constant(config("q26_fast", .06)),
        "q28_safe_t10_dual": constant(config("q28_safe", .10)),
        "q28_safe_t12_dual": constant(config("q28_safe", .12)),
        "adaptive_d145_t10": adaptive(1.45, .10),
        "adaptive_d150_t12": adaptive(1.50, .12),
    }


def screen(args):
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    _, calibration_payload = load_calibration(args.calibration, args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    configurations = variants()
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), calibration_fit_source=calibration_payload["fit_source_hash"],
        manifest=manifest, variants=configurations,
        purpose="development_only_calibrated_online_teacher")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "screen_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("Screen contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("Screen seeds overlap an earlier V15-V21 experiment")
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
            results[name] = dict(config=configuration, report=checked)
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
        print("No calibrated V21 teacher passed. Do not validate, collect BC data, or start RL.", flush=True)


def validate(args):
    selected_path = args.screen / "selected_config.json"
    if not selected_path.exists():
        raise ValueError("Screen has not selected a V21 teacher")
    selected = read_json(selected_path)
    load_calibration(args.calibration, args.init)
    if (selected.get("schema") != SCHEMA or not selected.get("eligible") or
            selected.get("source_hash") != source_hash() or
            selected.get("anchor_sha256") != file_hash(args.init) or
            selected.get("calibration_sha256") != file_hash(args.calibration)):
        raise ValueError("Selected V21 teacher does not match current source, BC, and calibration")
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), selected_config_sha256=file_hash(selected_path),
        manifest=manifest, purpose="independent_calibrated_teacher_validation")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "validation_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("Validation contract changed; use a new output directory")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("Validation seeds overlap earlier V15-V21 experiments")
    write_json(contract_path, identity)
    configuration = selected["config"]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        bc_path, candidate_path = args.out / "v15_bc_episodes.json", args.out / "v21_teacher_episodes.json"
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
        write_json(args.screen / "qualified_teacher.json", dict(**selected, validation_contract=identity,
            qualified=True, independent_report=checked))
    print(dict(qualified=qualified, summary=checked["summary"], checks=checked["checks"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command_name, default_out, default_episodes, default_seed in (
            ("screen", DEFAULT_OUT / "screen_v1", 20, 22100001),
            ("validate", DEFAULT_OUT / "screen_v1" / "validation_80", 80, 22200001)):
        command = sub.add_parser(command_name)
        command.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        if command_name == "validate":
            command.add_argument("--screen", type=Path, default=DEFAULT_OUT / "screen_v1")
        command.add_argument("--out", type=Path, default=default_out)
        for name, default in (("episodes", default_episodes), ("workers", 8), ("seed", default_seed),
                ("design-seed", 20260926 if command_name == "screen" else 20260927)):
            command.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.episodes < 5 or args.workers < 1:
        parser.error("episodes must be >=5 and workers must be positive")
    if args.command == "validate" and args.episodes < 80:
        parser.error("Independent validation requires at least 80 episodes")
    if args.command == "validate" and not (args.screen / "selected_config.json").exists():
        parser.error("Screen has not selected a V21 teacher")
    (screen if args.command == "screen" else validate)(args)


if __name__ == "__main__":
    main()
