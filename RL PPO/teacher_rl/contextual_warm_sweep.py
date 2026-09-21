"""Paired two-parameter V23 training sweep with the accepted warmed calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_calibration import (WarmContextualEnv,
    source_hash as calibration_source_hash, DEFAULT_OUT as CALIBRATION_OUT)
from .contextual_kernel import load_calibration, source_hash as kernel_source_hash
from .contextual_env import SCHEMA, base_config, normalized_action, episode_utility
from .contextual_rl import previous_seeds, worker as baseline_worker
from .data import write_json
from .envelope_teacher_env import EnvelopeTeacherConfig
from .envelope_teacher_rl import report
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe


def source_hash():
    digest = hashlib.sha256((calibration_source_hash() + kernel_source_hash()).encode())
    for name in ("contextual_env.py", "contextual_warm_sweep.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(job):
    scenario, anchor_path, calibration_path, config_data, action = job
    if config_data is None:
        return baseline_worker(job)
    torch.set_num_threads(1)
    anchor, payload = load_anchor(Path(anchor_path))
    calibration, _ = load_calibration(Path(calibration_path), Path(anchor_path))
    env = WarmContextualEnv(scenario, ImprovedRecipe(**payload["recipe"]), anchor, calibration,
        lambda context: np.asarray(action), EnvelopeTeacherConfig(**config_data))
    env.reset(scenario["seed"])
    while not env.done:
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    return dict(seed=scenario["seed"], scenario=scenario, result=result, utility=episode_utility(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_OUT / "release_calibration_v23.json")
    parser.add_argument("--out", type=Path, default=Path("teacher_runs/v23_contextual/warm_sweep_v1"))
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=27200001)
    parser.add_argument("--design-seed", type=int, default=20261021)
    parser.add_argument("--force-points", type=int, default=3)
    parser.add_argument("--radius-points", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=.04)
    parser.add_argument("--profile", default="force_125")
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.force_points < 2 or args.radius_points < 2:
        parser.error("positive scene/worker counts and at least two action points required")
    _, payload = load_anchor(args.init)
    _, calibration_payload = load_calibration(args.calibration, args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed, ImprovedRecipe(**payload["recipe"]))
    actions = [normalized_action(f, r).tolist() for f in np.linspace(1., 1.4, args.force_points)
               for r in np.linspace(1., 1.18, args.radius_points)]
    config = base_config(args.profile, args.tolerance)
    identity = dict(schema=SCHEMA, calibration_mode="warmed", source_hash=source_hash(),
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        calibration_seeds=calibration_payload["calibration_seeds"], manifest=manifest,
        config=asdict(config), actions=actions, purpose="training_and_development_only")
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "sweep_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("Warm V23 sweep changed; use a new directory")
    elif {row["seed"] for row in manifest["scenarios"]} & (previous_seeds(args.out) | set(calibration_payload["calibration_seeds"])):
        raise ValueError("Warm trajectory sweep overlaps earlier scenes")
    write_json(contract_path, identity)
    output = args.out / "episodes"
    output.mkdir(exist_ok=True)
    rows, missing = {}, []
    for scenario in manifest["scenarios"]:
        common = (scenario, str(args.init.resolve()), str(args.calibration.resolve()))
        jobs = [(f"{scenario['seed']}_bc", (*common, None, None))]
        jobs += [(f"{scenario['seed']}_a{i:03d}", (*common, asdict(config), action)) for i, action in enumerate(actions)]
        for key, job in jobs:
            path = output / (key + ".json")
            if path.exists():
                rows[key] = read_json(path)
            else:
                missing.append((key, job))
    total = len(rows) + len(missing)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(worker, job): key for key, job in missing}
        for future in as_completed(futures):
            key, row = futures[future], future.result()
            rows[key] = row
            write_json(output / (key + ".json"), row)
            if len(rows) % 10 == 0 or len(rows) == total:
                print(dict(warm_sweep_completed=len(rows), total=total), flush=True)
    baseline, oracle, choices = [], [], []
    for scenario in manifest["scenarios"]:
        seed = scenario["seed"]
        baseline.append(rows[f"{seed}_bc"])
        candidates = [rows[f"{seed}_a{i:03d}"] for i in range(len(actions))]
        index = max(range(len(actions)), key=lambda i: candidates[i]["utility"])
        oracle.append(candidates[index])
        choices.append(dict(seed=seed, scenario=scenario, action=actions[index],
            utility=candidates[index]["utility"], hits=sum(row["result"]["hit15"] for row in candidates),
            result=candidates[index]["result"]))
    checked = report(oracle, baseline)
    write_json(args.out / "oracle_diagnostic.json", dict(warning="POST-HOC OUTCOME ORACLE, NOT A DEPLOYABLE POLICY",
        choices=choices, report=checked))
    print(dict(oracle_only=True, summary=checked["summary"], baseline=checked["baseline"], checks=checked["checks"]), flush=True)


if __name__ == "__main__":
    main()
