"""Collect >=1000 independent V26 post-catch contexts with terminal-twin labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_env import CONTEXT_NAMES
from .contextual_kernel import load_calibration
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe
from .twin_hybrid_gate import (HIGH_LEVEL_ACTIONS, SCHEMA as HYBRID_SCHEMA,
    source_hash as hybrid_source_hash, worker)
from .twin_mpc_ppo import DEFAULT_CALIBRATION, _extract_manifest_seeds, _v25_contract_seeds
from .twin_residual_env import TwinResidualConfig


SCHEMA = "can_v26_hybrid_terminal_dataset_v1"
DEFAULT_ROOT = Path("teacher_runs/v26_twin_mpc_ppo")


def source_hash():
    path = Path(__file__)
    digest = hashlib.sha256((SCHEMA + hybrid_source_hash()).encode())
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def require_validation(path):
    report = read_json(path)
    if (report.get("schema") != HYBRID_SCHEMA or
            report.get("source_hash") != hybrid_source_hash() or
            report.get("kind") != "hybrid_teacher_validation" or
            report.get("episodes_completed", 0) < 100 or
            not report.get("qualified_hybrid_teacher")):
        raise ValueError("dataset collection requires the qualified current V26 hybrid teacher")
    if not all(report.get("checks", {}).values()):
        raise ValueError("qualified hybrid report contains a failed acceptance check")
    return report


def previous_v26_seeds(excluded):
    result = set()
    excluded = Path(excluded).resolve()
    for root in (Path("teacher_runs/v26_twin_mpc_ppo"),
                 Path("teacher_runs/v26_twin_mpc_ppo_smoke")):
        if not root.exists():
            continue
        for pattern in ("*contract.json", "*report.json"):
            for path in root.rglob(pattern):
                if path.resolve().is_relative_to(excluded):
                    continue
                result.update(_extract_manifest_seeds(read_json(path)))
    return result


def dataset_summary(rows, requested_contexts):
    captured = [row for row in rows if row["result"].get("captured")]
    train = [row for row in captured if row["scenario"].get("split") == "train"]
    validation = [row for row in captured if row["scenario"].get("split") == "validation"]
    context_complete = all(isinstance(row.get("observed_context"), list) and
        len(row["observed_context"]) == len(CONTEXT_NAMES) for row in captured)
    branch_complete = all(isinstance(row.get("predicted_branches"), list) and
        len(row["predicted_branches"]) == 9 and
        all(branch is not None for branch in row["predicted_branches"]) for row in captured)
    exact = all(bool(row.get("prediction_exact")) for row in captured)
    fallback_exact = all(row["result"] == row["baseline"] for row in captured
                         if row.get("mode") == "v15_fallback")
    safety = all(not row["result"].get("grip_broken") and
                 row["result"].get("max_joint_deg", 1e9) <= 36. for row in rows)
    unique = len({int(row["seed"]) for row in rows}) == len(rows)
    actions = [int(row["high_level_action"]) for row in captured]
    import numpy as np
    checks = dict(captured_contexts=len(captured) >= requested_contexts,
        train_contexts=len(train) >= 800, validation_contexts=len(validation) >= 180,
        context_complete=context_complete, nine_terminal_labels=branch_complete,
        exact_twin_or_fallback_replay=exact, exact_v15_fallback=fallback_exact,
        unique_scenarios=unique, safety=safety)
    return dict(episodes_completed=len(rows), captured_contexts=len(captured),
        train_contexts=len(train), validation_contexts=len(validation),
        terminal_branch_labels=len(captured) * 9,
        v26_takeovers=sum(row.get("mode") == "v26_mpc" for row in captured),
        v15_fallbacks=sum(row.get("mode") == "v15_fallback" for row in captured),
        selected_high_level_action_counts=np.bincount(
            actions, minlength=len(HIGH_LEVEL_ACTIONS)).tolist(),
        checks=checks, dataset_ready=all(checks.values()))


def collect(args):
    qualification = require_validation(args.qualification)
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    used = (_v25_contract_seeds() | _extract_manifest_seeds(qualification) |
            previous_v26_seeds(args.out))
    current = {int(row["seed"]) for row in manifest["scenarios"]}
    if used & current:
        raise ValueError(f"V26 dataset collection overlaps prior scenarios: {sorted(used & current)[:5]}")
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, 2.)
    contract = dict(schema=SCHEMA, source_hash=source_hash(),
        kind="hybrid_terminal_dataset_collection",
        qualification_sha256=file_hash(args.qualification),
        hybrid_source_hash=qualification["source_hash"],
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        manifest=manifest, dynamic=asdict(dynamic),
        high_level_actions=list(HIGH_LEVEL_ACTIONS), requested_captured_contexts=args.contexts,
        per_context_terminal_branches=9, independent_scenarios=True,
        exact_v15_paired_baseline=True, smoke=bool(args.smoke))
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "collection_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("V26 dataset collection contract changed; use a new output directory")
    write_json(contract_path, contract)
    jobs = []
    for scenario in manifest["scenarios"]:
        path = args.out / "episodes" / f"{scenario['seed']}.json"
        if not path.exists():
            jobs.append((scenario, str(args.init.resolve()), str(args.calibration.resolve()),
                asdict(dynamic), str(path.resolve())))
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            completed = len(manifest["scenarios"]) - len(jobs)
            for future in as_completed(futures):
                row = future.result()
                completed += 1
                emit(dict(stage="v26_terminal_dataset", completed=completed,
                    total=len(manifest["scenarios"]), seed=row["seed"], mode=row["mode"],
                    captured=row["result"].get("captured"),
                    released=row["result"].get("released"), hit15=row["result"].get("hit15")))
    rows = [read_json(args.out / "episodes" / f"{scenario['seed']}.json")
            for scenario in manifest["scenarios"]]
    summary = dataset_summary(rows, args.contexts)
    report = dict(**contract, **summary)
    write_json(args.out / "collection_report.json", report)
    emit(dict(episodes=summary["episodes_completed"],
        captured_contexts=summary["captured_contexts"],
        train_contexts=summary["train_contexts"],
        validation_contexts=summary["validation_contexts"],
        terminal_branch_labels=summary["terminal_branch_labels"],
        dataset_ready=summary["dataset_ready"], checks=summary["checks"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qualification", type=Path,
        default=DEFAULT_ROOT / "stage3_hybrid_validation" / "validation_report.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "stage4_dataset")
    parser.add_argument("--episodes", type=int, default=1300)
    parser.add_argument("--contexts", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33300001)
    parser.add_argument("--design-seed", type=int, default=20261304)
    parser.add_argument("--decision-interval", type=float, default=.5)
    parser.add_argument("--force-step", type=float, default=.05)
    parser.add_argument("--radius-step", type=float, default=.0225)
    parser.add_argument("--release-tolerance", type=float, default=.08)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.episodes < 5 or args.contexts < 1 or args.workers < 1:
        parser.error("episodes must be >=5; contexts and workers must be positive")
    collect(args)


if __name__ == "__main__":
    main()
