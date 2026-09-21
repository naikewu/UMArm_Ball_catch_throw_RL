"""V26 terminal-twin verification and action-space feasibility audit.

Only the two pre-training stages are exposed here.  PPO stages must not be
added until the oracle audit proves that the bounded action space is capable of
reaching the release region on independent scenarios.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .anchored_dynamic_env import TRAJECTORY_ACTIONS, initial_action
from .anchored_dynamic_rl import _v15_rollout
from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_env import episode_utility
from .contextual_kernel import load_calibration
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json, summarize
from .improved_teacher import ImprovedRecipe
from .twin_residual_env import (HOLD_ACTION, TwinResidualConfig,
    TwinResidualEnv)
from .twin_snapshot import TwinSnapshot, numeric_fingerprint
from .twin_terminal_planner import terminal_branch, terminal_features, terminal_rank


SCHEMA = "can_v15_terminal_twin_mpc_v26"
DEFAULT_ROOT = Path("teacher_runs/v26_twin_mpc_ppo")
DEFAULT_CALIBRATION = Path(
    "teacher_runs/v23_contextual/warm_kernel_v1/release_calibration_v23_kernel.json")


def source_hash():
    digest = hashlib.sha256(SCHEMA.encode())
    for name in ("twin_snapshot.py", "twin_residual_env.py",
                 "twin_terminal_planner.py", "twin_mpc_ppo.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def identity(args, kind):
    return dict(schema=SCHEMA, source_hash=source_hash(), kind=kind,
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration))


def require_identity(payload, args, kind):
    expected = identity(args, kind)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{kind} artifact mismatch at {key}; rerun V26 verify")


def make_env(scenario, anchor, anchor_payload, calibration, selector, dynamic):
    return TwinResidualEnv(scenario, ImprovedRecipe(**anchor_payload["recipe"]),
        anchor, calibration, selector, dynamic)


def advance_to_decision(env, limit=6000):
    result = None
    for _ in range(limit):
        if env.dynamic_due() or env.done:
            return result
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    raise RuntimeError("V26 did not reach a post-catch decision")


def verify(args):
    anchor, anchor_payload = load_anchor(args.init)
    calibration, _ = load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    scenario = scenarios(5, args.seed, args.design_seed, recipe)["scenarios"][0]

    def selector(_context):
        return initial_action(4 + 1)

    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, 2.)
    env = make_env(scenario, anchor, anchor_payload, calibration, selector, dynamic)
    env.reset(scenario["seed"])
    advance_to_decision(env)
    if env.done or not env.dynamic_due():
        raise RuntimeError("V26 verification scenario never reached a decision")
    env.policy_observation()
    snapshot = TwinSnapshot(env)
    first, _, first_fingerprint = terminal_branch(env, snapshot, HOLD_ACTION)
    second, _, second_fingerprint = terminal_branch(env, snapshot, HOLD_ACTION)
    exact = bool(first == second and np.array_equal(first_fingerprint, second_fingerprint))
    snapshot.restore()
    restored = numeric_fingerprint(env)
    report = dict(**identity(args, "terminal_twin_verification"), passed=exact,
        scenario=scenario, exact_terminal_replay=exact,
        terminal_steps_elapsed_s=first["elapsed_s"], terminal_result=first,
        final_fingerprint_values=len(first_fingerprint),
        restored_fingerprint_values=len(restored), dynamic=asdict(dynamic))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "verification.json", report)
    emit(report)
    if not exact:
        raise RuntimeError("V26 terminal digital-twin replay was not deterministic")


def _run_fixed(scenario, anchor, anchor_payload, calibration, dynamic, action):
    """Run one fixed trajectory from reset for a trustworthy paired outcome."""
    holder = {}

    def selector(context):
        holder["context"] = np.asarray(context, dtype=np.float32).tolist()
        return initial_action(int(action) + 1)

    env = make_env(scenario, anchor, anchor_payload, calibration, selector, dynamic)
    env.reset(scenario["seed"])
    result = None
    while not env.done:
        if env.dynamic_due():
            env.policy_observation()
            env.apply_joint_action(HOLD_ACTION)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    return result or env.summary(), holder.get("context")


def oracle_worker(job):
    scenario, anchor_path, calibration_path, dynamic_data, output_path = job
    torch.set_num_threads(1)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, anchor_payload = load_anchor(anchor_path)
    calibration, _ = load_calibration(calibration_path, anchor_path)
    dynamic = TwinResidualConfig(**dynamic_data)
    started = time.perf_counter()
    baseline = _v15_rollout(scenario, anchor, anchor_payload)
    fixed = []
    context = None
    for action, parameters in enumerate(TRAJECTORY_ACTIONS):
        result, current_context = _run_fixed(
            scenario, anchor, anchor_payload, calibration, dynamic, action)
        if current_context is not None:
            context = current_context
        features = terminal_features(result)
        fixed.append(dict(action=action,
            parameters=dict(force=parameters[0], radius=parameters[1]),
            features=features, result=result))
    best = max(fixed, key=lambda row: terminal_rank(row["features"]))
    row = dict(seed=int(scenario["seed"]), scenario=scenario,
        context=context, baseline=baseline,
        fixed_actions=[dict(action=item["action"], parameters=item["parameters"],
                            features=item["features"]) for item in fixed],
        static_oracle=dict(action=best["action"], parameters=best["parameters"],
                           result=best["result"], features=best["features"]),
        wall_s=time.perf_counter()-started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, row)
    return row


def _extract_manifest_seeds(value):
    result = set()
    if isinstance(value, dict):
        scenarios_value = value.get("scenarios")
        if isinstance(scenarios_value, list):
            result.update(int(row["seed"]) for row in scenarios_value
                          if isinstance(row, dict) and "seed" in row)
        for row in value.values():
            result.update(_extract_manifest_seeds(row))
    elif isinstance(value, list):
        for row in value:
            result.update(_extract_manifest_seeds(row))
    return result


def _v25_contract_seeds():
    root = Path("teacher_runs/v25_twin_residual")
    result = set()
    if root.exists():
        for path in root.rglob("*contract.json"):
            result.update(_extract_manifest_seeds(read_json(path)))
    return result


def _summary_for(rows, key):
    return summarize([dict(seed=row["seed"], scenario=row["scenario"],
        result=(row[key] if key == "baseline" else row[key]["result"])) for row in rows])


def _conditional(result):
    captured = int(result["captured"])
    released = int(result["released"])
    hit = int(result["hit15"])
    return dict(release_given_capture=released/max(1, captured),
                hit15_given_release=hit/max(1, released))


def oracle_summary(rows):
    baseline = _summary_for(rows, "baseline")
    static = _summary_for(rows, "static_oracle")
    captured = [row for row in rows if row["static_oracle"]["result"].get("captured")]
    safe_positive = [row for row in captured
        if not row["static_oracle"]["result"].get("grip_broken")
        and row["static_oracle"]["result"].get("max_joint_deg", 1e9) <= 36.
        and episode_utility(row["static_oracle"]["result"]) > episode_utility(row["baseline"])]
    safety = all(not row["static_oracle"]["result"].get("grip_broken") and
        row["static_oracle"]["result"].get("max_joint_deg", 1e9) <= 36. for row in rows)
    rates = _conditional(static)
    positive_fraction = len(safe_positive)/max(1, len(captured))
    checks = dict(capture_nonregression=static["captured"] >= baseline["captured"],
        release_given_capture=rates["release_given_capture"] >= .9,
        hit15_given_release=rates["hit15_given_release"] >= .9,
        safe_positive_fraction=positive_fraction >= .25, safety=safety)
    return dict(baseline=baseline, static_oracle=static, static_rates=rates,
        safe_positive=len(safe_positive), captured_contexts=len(captured),
        safe_positive_fraction=positive_fraction, checks=checks,
        action_space_feasible=all(checks.values()))


def oracle(args):
    verification = read_json(args.verification)
    require_identity(verification, args, "terminal_twin_verification")
    if not verification.get("passed"):
        raise ValueError("V26 terminal-twin verification did not pass")
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    current_seeds = {int(row["seed"]) for row in manifest["scenarios"]}
    overlap = current_seeds & _v25_contract_seeds()
    if overlap:
        raise ValueError(f"V26 oracle overlaps V25 scenario seeds: {sorted(overlap)[:5]}")
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, 2.)
    contract = dict(**identity(args, "terminal_twin_oracle"),
        verification_sha256=file_hash(args.verification), manifest=manifest,
        dynamic=asdict(dynamic), fixed_actions=list(TRAJECTORY_ACTIONS),
        oracle_scope="privileged_best_of_nine_fixed_terminal_trajectories",
        purpose="prove_force_radius_action_space_reachability_before_dynamic_MPC",
        smoke=bool(args.smoke))
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "oracle_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("V26 oracle contract changed; use a new output directory")
    write_json(contract_path, contract)
    episode_dir = args.out / "episodes"
    jobs = []
    for scenario in manifest["scenarios"]:
        path = episode_dir / f"{scenario['seed']}.json"
        if not path.exists():
            jobs.append((scenario, str(args.init.resolve()), str(args.calibration.resolve()),
                asdict(dynamic), str(path.resolve())))
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            futures = [pool.submit(oracle_worker, job) for job in jobs]
            completed = 0
            for future in as_completed(futures):
                row = future.result()
                completed += 1
                emit(dict(stage="v26_terminal_oracle", completed=completed,
                    remaining=len(futures)-completed, seed=row["seed"],
                    static_released=row["static_oracle"]["result"].get("released"),
                    static_hit15=row["static_oracle"]["result"].get("hit15"),
                    wall_s=row["wall_s"]))
    rows = [read_json(episode_dir / f"{scenario['seed']}.json")
            for scenario in manifest["scenarios"]]
    report = dict(**contract, **oracle_summary(rows), episodes_completed=len(rows))
    write_json(args.out / "oracle_report.json", report)
    emit(dict(episodes=len(rows), action_space_feasible=report["action_space_feasible"],
              checks=report["checks"], static_rates=report["static_rates"],
              safe_positive_fraction=report["safe_positive_fraction"]))


def _common(parser):
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)


def _dynamic(parser):
    parser.add_argument("--decision-interval", type=float, default=.5)
    parser.add_argument("--force-step", type=float, default=.05)
    parser.add_argument("--radius-step", type=float, default=.0225)
    parser.add_argument("--release-tolerance", type=float, default=.08)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("verify")
    _common(command); _dynamic(command)
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT / "stage0_verify")
    command.add_argument("--seed", type=int, default=32700001)
    command.add_argument("--design-seed", type=int, default=20261127)
    command = sub.add_parser("oracle")
    _common(command); _dynamic(command)
    command.add_argument("--verification", type=Path,
                         default=DEFAULT_ROOT / "stage0_verify" / "verification.json")
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT / "stage1_oracle")
    command.add_argument("--episodes", type=int, default=200)
    command.add_argument("--workers", type=int, default=8)
    # 32100001--32900020 already occur in V25 contracts/pilots.  Keep V26's
    # formal oracle in a separate range and verify that separation at runtime.
    command.add_argument("--seed", type=int, default=33000001)
    command.add_argument("--design-seed", type=int, default=20261201)
    command.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if hasattr(args, "episodes") and args.episodes < 1:
        parser.error("episodes must be positive")
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("workers must be positive")
    (verify if args.command == "verify" else oracle)(args)


if __name__ == "__main__":
    main()
