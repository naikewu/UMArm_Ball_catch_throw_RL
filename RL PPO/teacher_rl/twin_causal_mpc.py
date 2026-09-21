"""Causal one-decision MPC using terminal rollouts of the verified CAN twin."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
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
from .twin_mpc_ppo import (DEFAULT_CALIBRATION, _extract_manifest_seeds,
    _v25_contract_seeds, make_env, source_hash as oracle_source_hash)
from .twin_residual_env import HOLD_ACTION, TwinResidualConfig
from .twin_snapshot import TwinSnapshot, numeric_fingerprint
from .twin_terminal_planner import terminal_features, terminal_rank


SCHEMA = "can_v15_causal_terminal_mpc_v26"
DEFAULT_ROOT = Path("teacher_runs/v26_twin_mpc_ppo")


def source_hash():
    path = Path(__file__)
    digest = hashlib.sha256((SCHEMA + oracle_source_hash()).encode())
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def apply_absolute_action(env, action):
    """Apply one of the audited 3x3 trajectories at the measured handover state."""
    if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(TRAJECTORY_ACTIONS):
        raise ValueError("terminal MPC action is outside the fixed 3x3 catalogue")
    force, radius = TRAJECTORY_ACTIONS[int(action)]
    parameters = dict(envelope_force_scale=float(force), envelope_radius_scale=float(radius))
    env.continuous = replace(env.continuous, **parameters)
    env.handover.config = env.continuous
    object.__setattr__(env.releaser, "config", env.continuous)
    env.releaser.set_permission(True)
    return parameters


def _finish(env, start_time, max_steps=6000):
    result = None
    for _ in range(max_steps):
        if env.done:
            break
        if env.dynamic_due():
            env.policy_observation()
            env.apply_joint_action(HOLD_ACTION)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    if not env.done:
        raise RuntimeError("causal terminal MPC branch exceeded its step limit")
    result = result or env.summary()
    features = terminal_features(result, elapsed_s=float(env.obs["t_win"])-start_time)
    return features, result, numeric_fingerprint(env)


def terminal_absolute_branch(env, snapshot, action):
    snapshot.restore()
    start = float(env.obs["t_win"])
    parameters = apply_absolute_action(env, action)
    features, result, fingerprint = _finish(env, start)
    features.update(action=int(action), parameters=parameters,
                    rank=list(terminal_rank(features)))
    return features, result, fingerprint


def choose_causal_action(env):
    """Plan only from the current state; no scenario parameters or future result enter."""
    if env.done or env.trajectory_decision is None or not env.obs["held"]:
        raise RuntimeError("causal terminal MPC requires a live post-catch state")
    snapshot = TwinSnapshot(env)
    branches = []
    for action in range(len(TRAJECTORY_ACTIONS)):
        features, _, _ = terminal_absolute_branch(env, snapshot, action)
        branches.append(features)
    snapshot.restore()
    chosen = max(range(len(branches)), key=lambda index: terminal_rank(branches[index]))
    return int(chosen), branches


def worker(job):
    scenario, anchor_path, calibration_path, dynamic_data, output_path = job
    torch.set_num_threads(1)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, anchor_payload = load_anchor(anchor_path)
    calibration, _ = load_calibration(calibration_path, anchor_path)
    dynamic = TwinResidualConfig(**dynamic_data)

    # The centre trajectory is only a placeholder during the handover tick.
    # Planning happens immediately after that tick, before another physics step.
    env = make_env(scenario, anchor, anchor_payload, calibration,
                   lambda _context: initial_action(4 + 1), dynamic)
    env.reset(scenario["seed"])
    result = None
    started = time.perf_counter()
    for _ in range(6000):
        if env.done or env.trajectory_decision is not None:
            break
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))

    action, branches, parameters, exact = None, None, None, None
    if not env.done and env.trajectory_decision is not None and env.obs["held"]:
        action, branches = choose_causal_action(env)
        prediction = branches[action]
        start = float(env.obs["t_win"])
        parameters = apply_absolute_action(env, action)
        actual_features, result, _ = _finish(env, start)
        exact = all(actual_features[key] == prediction[key] for key in (
            "captured", "released", "hit15", "unsafe", "grip_broken",
            "max_joint_deg", "terminal_error_m", "catch_to_release_s", "elapsed_s", "utility"))
    result = result or env.summary()
    baseline = _v15_rollout(scenario, anchor, anchor_payload)
    row = dict(seed=int(scenario["seed"]), scenario=scenario, result=result,
        baseline=baseline, observed_context=(None if env.trajectory_decision is None else
            env.trajectory_decision.get("context")), action=action, parameters=parameters,
        predicted_branches=branches, prediction_exact=exact,
        paired_utility_delta=float(episode_utility(result)-episode_utility(baseline)),
        wall_s=time.perf_counter()-started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, row)
    return row


def _summary(rows):
    candidate = summarize(rows)
    baseline = summarize([dict(seed=row["seed"], scenario=row["scenario"],
        result=row["baseline"]) for row in rows])
    captured = [row for row in rows if row["result"].get("captured")]
    released = [row for row in captured if row["result"].get("released")]
    hit = [row for row in released if row["result"].get("hit15")]
    safe_positive = [row for row in captured if row["result"].get("hit15") and
        not row["result"].get("grip_broken") and row["result"].get("max_joint_deg", 1e9) <= 36.
        and row["paired_utility_delta"] > 0.]
    release_rate = len(released)/max(1, len(captured))
    hit_rate = len(hit)/max(1, len(released))
    positive_rate = len(safe_positive)/max(1, len(captured))
    planned = [row for row in captured if row["action"] is not None]
    exact = bool(planned and all(row["prediction_exact"] for row in planned))
    safety = all(not row["result"].get("grip_broken") and
        row["result"].get("max_joint_deg", 1e9) <= 36. for row in rows)
    checks = dict(capture_nonregression=candidate["captured"] >= baseline["captured"],
        release_nonregression=candidate["released"] >= baseline["released"],
        hit15_nonregression=candidate["hit15"] >= baseline["hit15"],
        release_given_capture=release_rate >= .9, hit15_given_release=hit_rate >= .9,
        safe_positive_fraction=positive_rate >= .25, exact_twin_prediction=exact,
        safety=safety)
    actions = [row["action"] for row in planned]
    return dict(candidate=candidate, baseline=baseline,
        release_given_capture=release_rate, hit15_given_release=hit_rate,
        safe_positive=len(safe_positive), safe_positive_fraction=positive_rate,
        paired_mean_utility_delta=float(np.mean([row["paired_utility_delta"] for row in rows])),
        selected_action_counts=np.bincount(actions, minlength=len(TRAJECTORY_ACTIONS)).tolist(),
        prediction_exact_episodes=sum(bool(row.get("prediction_exact")) for row in planned),
        planned_episodes=len(planned), checks=checks,
        qualified_as_causal_mpc=all(checks.values()))


def pilot(args):
    oracle = read_json(args.oracle)
    if (oracle.get("source_hash") != oracle_source_hash() or
            oracle.get("kind") != "terminal_twin_oracle" or
            not oracle.get("action_space_feasible")):
        raise ValueError("causal MPC requires the completed feasible V26 oracle")
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    used = _v25_contract_seeds() | _extract_manifest_seeds(oracle)
    current = {int(row["seed"]) for row in manifest["scenarios"]}
    if used & current:
        raise ValueError(f"V26 causal MPC pilot overlaps earlier scenarios: {sorted(used & current)[:5]}")
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, 2.)
    contract = dict(schema=SCHEMA, source_hash=source_hash(), kind="causal_terminal_mpc_pilot",
        oracle_sha256=file_hash(args.oracle), oracle_source_hash=oracle["source_hash"],
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        manifest=manifest, dynamic=asdict(dynamic),
        candidates=[list(parameters) for parameters in TRAJECTORY_ACTIONS],
        decision_information="current_measured_postcatch_state_only",
        planner="restore_verified_twin_snapshot_and_roll_each_candidate_to_terminal",
        exact_v15_paired_baseline=True)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "pilot_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("V26 causal MPC pilot contract changed; use a new output directory")
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
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                emit(dict(stage="v26_causal_mpc_pilot", completed=completed,
                    remaining=len(futures)-completed, seed=row["seed"], action=row["action"],
                    captured=row["result"].get("captured"),
                    released=row["result"].get("released"), hit15=row["result"].get("hit15"),
                    prediction_exact=row["prediction_exact"]))
    rows = [read_json(args.out / "episodes" / f"{scenario['seed']}.json")
            for scenario in manifest["scenarios"]]
    report = dict(**contract, **_summary(rows), episodes_completed=len(rows))
    write_json(args.out / "pilot_report.json", report)
    emit(dict(episodes=len(rows), qualified=report["qualified_as_causal_mpc"],
        release_given_capture=report["release_given_capture"],
        hit15_given_release=report["hit15_given_release"],
        safe_positive_fraction=report["safe_positive_fraction"],
        prediction_exact_episodes=report["prediction_exact_episodes"],
        selected_action_counts=report["selected_action_counts"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path,
        default=DEFAULT_ROOT / "stage1_oracle" / "oracle_report.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "stage2_causal_mpc_pilot")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33100001)
    parser.add_argument("--design-seed", type=int, default=20261203)
    parser.add_argument("--decision-interval", type=float, default=.5)
    parser.add_argument("--force-step", type=float, default=.05)
    parser.add_argument("--radius-step", type=float, default=.0225)
    parser.add_argument("--release-tolerance", type=float, default=.08)
    args = parser.parse_args()
    if args.episodes < 5 or args.workers < 1:
        parser.error("episodes must be at least five and workers must be positive")
    pilot(args)


if __name__ == "__main__":
    main()
