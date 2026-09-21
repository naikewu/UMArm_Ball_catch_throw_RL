"""V23 reproducible contextual trajectory experiments and outcome supervision."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_env import (SCHEMA, CONTEXT_NAMES, ContextualEnv, base_config,
                             normalized_action, episode_utility)
from .data import write_json
from .envelope_teacher_env import EnvelopeTeacherConfig
from .envelope_teacher_rl import (DEFAULT_CALIBRATION, load_calibration,
    source_hash as v22_source_hash, previous_seeds as v22_previous_seeds, report)
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv

DEFAULT_OUT = Path("teacher_runs/v23_contextual")


def source_hash():
    digest = hashlib.sha256(v22_source_hash().encode())
    for name in ("contextual_env.py", "contextual_rl.py", "contextual_model.py"):
        path = Path(__file__).with_name(name)
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def previous_seeds(exclude=None):
    used = v22_previous_seeds()
    excluded = None if exclude is None else Path(exclude).resolve()
    for path in DEFAULT_OUT.rglob("*_contract.json"):
        if excluded is None or not path.resolve().is_relative_to(excluded):
            used.update(row["seed"] for row in read_json(path).get("manifest", {}).get("scenarios", []))
    return used


def worker(job):
    torch.set_num_threads(1)
    scenario, anchor_path, calibration_path, config_data, action = job
    anchor, payload = load_anchor(Path(anchor_path))
    recipe = ImprovedRecipe(**payload["recipe"])
    if config_data is None:
        env = ImprovedTeacherEnv(scenario, recipe)
    else:
        calibration, _ = load_calibration(Path(calibration_path), Path(anchor_path))
        env = ContextualEnv(scenario, recipe, anchor, calibration,
            lambda context: np.asarray(action), EnvelopeTeacherConfig(**config_data))
    observation = env.reset(scenario["seed"])
    started = time.perf_counter()
    while not env.done:
        if config_data is None:
            with torch.no_grad():
                proposal = anchor.distribution(torch.tensor(observation[None])).mean.tanh()[0].numpy()
        else:
            proposal = np.zeros(7, dtype=np.float32)
        observation, _, _, result = env.step(proposal)
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or
        env.capture_physics_time is None else env.controller.t_hand - env.capture_physics_time)
    result.setdefault("catch_to_release_s", None if result["release_time"] is None or
        result["capture_time"] is None else result["release_time"] - result["capture_time"])
    return dict(seed=scenario["seed"], scenario=scenario, result=result,
                utility=episode_utility(result), wall_s=time.perf_counter() - started)


def run_cached(jobs, output, workers):
    rows = {}
    missing = []
    output.mkdir(parents=True, exist_ok=True)
    for key, job in jobs:
        path = output / (key + ".json")
        if path.exists():
            rows[key] = read_json(path)
        else:
            missing.append((key, job))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(worker, job): key for key, job in missing}
        for future in as_completed(futures):
            key = futures[future]
            row = future.result()
            write_json(output / (key + ".json"), row)
            rows[key] = row
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                print(dict(completed=len(rows), total=len(jobs), last=key), flush=True)
    return rows


def sweep(args):
    _, payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed, ImprovedRecipe(**payload["recipe"]))
    config = base_config(args.profile, args.tolerance)
    actions = [normalized_action(f, r).tolist() for f in np.linspace(1., 1.4, args.force_points)
               for r in np.linspace(1., 1.18, args.radius_points)]
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), manifest=manifest, config=asdict(config),
        actions=actions, purpose="training_and_development_only")
    contract = args.out / "sweep_contract.json"
    args.out.mkdir(parents=True, exist_ok=True)
    if contract.exists():
        if read_json(contract) != identity:
            raise ValueError("V23 sweep changed; use a new output directory")
    elif set(row["seed"] for row in manifest["scenarios"]) & previous_seeds(args.out):
        raise ValueError("V23 scenarios overlap previous experiments")
    write_json(contract, identity)
    jobs = []
    for scenario in manifest["scenarios"]:
        common = (scenario, str(args.init.resolve()), str(args.calibration.resolve()))
        jobs.append((f"{scenario['seed']}_bc", (*common, None, None)))
        for index, action in enumerate(actions):
            jobs.append((f"{scenario['seed']}_a{index:03d}", (*common, asdict(config), action)))
    rows = run_cached(jobs, args.out / "episodes", args.workers)
    baseline, oracle, choices = [], [], []
    for scenario in manifest["scenarios"]:
        seed = scenario["seed"]
        baseline.append(rows[f"{seed}_bc"])
        candidates = [rows[f"{seed}_a{i:03d}"] for i in range(len(actions))]
        best_index = max(range(len(actions)), key=lambda i: candidates[i]["utility"])
        oracle.append(candidates[best_index])
        choices.append(dict(seed=seed, scenario=scenario, action=actions[best_index],
            utility=candidates[best_index]["utility"], hits=sum(row["result"]["hit15"] for row in candidates),
            result=candidates[best_index]["result"]))
    checked = report(oracle, baseline)
    write_json(args.out / "oracle_diagnostic.json", dict(
        warning="Uses actual outcomes after each episode; NOT an executable policy or independent evaluation",
        choices=choices, report=checked))
    print(dict(oracle_only=True, summary=checked["summary"], baseline=checked["baseline"],
               checks=checked["checks"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("sweep")
    command.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    command.add_argument("--out", type=Path, default=DEFAULT_OUT / "sweep_v1")
    command.add_argument("--episodes", type=int, default=20)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=26100001)
    command.add_argument("--design-seed", type=int, default=20261010)
    command.add_argument("--profile", default="force_125")
    command.add_argument("--tolerance", type=float, default=.04)
    command.add_argument("--force-points", type=int, default=5)
    command.add_argument("--radius-points", type=int, default=4)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.force_points < 2 or args.radius_points < 2:
        parser.error("positive episodes/workers and at least two points per action are required")
    sweep(args)


if __name__ == "__main__":
    main()
