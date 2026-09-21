"""V26 causal terminal-MPC teacher with an exact V15 safety fallback."""
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
from .twin_causal_mpc import (SCHEMA as MPC_SCHEMA, _finish, apply_absolute_action,
    choose_causal_action, source_hash as mpc_source_hash)
from .twin_mpc_ppo import (DEFAULT_CALIBRATION, _extract_manifest_seeds,
    _v25_contract_seeds, make_env, source_hash as oracle_source_hash)
from .twin_residual_env import TwinResidualConfig
from .twin_terminal_planner import terminal_features, terminal_rank


SCHEMA = "can_v15_exact_fallback_terminal_mpc_v26"
DEFAULT_ROOT = Path("teacher_runs/v26_twin_mpc_ppo")
HIGH_LEVEL_ACTIONS = ("v15_fallback",) + tuple(
    f"v26_force_{force:.2f}_radius_{radius:.2f}" for force, radius in TRAJECTORY_ACTIONS)


def source_hash():
    path = Path(__file__)
    digest = hashlib.sha256((SCHEMA + mpc_source_hash()).encode())
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def select_high_level_action(branches):
    """Return 0 for V15 unless the twin predicts a safe 15 cm V26 hit."""
    if len(branches) != len(TRAJECTORY_ACTIONS) or any(branch is None for branch in branches):
        raise ValueError("hybrid gate requires all nine terminal branch predictions")
    eligible = [index for index, branch in enumerate(branches)
                if branch.get("hit15") and not branch.get("unsafe")]
    if not eligible:
        return 0
    chosen = max(eligible, key=lambda index: terminal_rank(branches[index]))
    return int(chosen) + 1


def _prediction_matches(prediction, actual):
    actual_features = terminal_features(actual, elapsed_s=prediction["elapsed_s"])
    return all(actual_features[key] == prediction[key] for key in (
        "captured", "released", "hit15", "unsafe", "grip_broken",
        "max_joint_deg", "terminal_error_m", "catch_to_release_s", "elapsed_s", "utility"))


def worker(job):
    scenario, anchor_path, calibration_path, dynamic_data, output_path = job
    torch.set_num_threads(1)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, anchor_payload = load_anchor(anchor_path)
    calibration, _ = load_calibration(calibration_path, anchor_path)
    dynamic = TwinResidualConfig(**dynamic_data)
    baseline = None

    env = make_env(scenario, anchor, anchor_payload, calibration,
                   lambda _context: initial_action(4 + 1), dynamic)
    env.reset(scenario["seed"])
    result = None
    started = time.perf_counter()
    for _ in range(6000):
        if env.done or env.trajectory_decision is not None:
            break
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))

    action, high_level, branches, parameters, exact = None, None, None, None, None
    if not env.done and env.trajectory_decision is not None and env.obs["held"]:
        _, branches = choose_causal_action(env)
        high_level = select_high_level_action(branches)
        if high_level == 0:
            # This mirrors the established exact V15 passthrough contract: the
            # partial candidate simulation is discarded and V15 owns the whole
            # episode.  A physical controller would simply retain V15 at this
            # same handover decision instead of switching to the V26 handover.
            baseline = _v15_rollout(scenario, anchor, anchor_payload)
            result = baseline
            exact = True
        else:
            action = high_level - 1
            prediction = branches[action]
            start = float(env.obs["t_win"])
            parameters = apply_absolute_action(env, action)
            _, result, _ = _finish(env, start)
            exact = _prediction_matches(prediction, result)
    else:
        # No high-level post-catch decision exists.  Preserve exact V15 behavior.
        baseline = _v15_rollout(scenario, anchor, anchor_payload)
        result = baseline
        high_level = 0
        exact = True

    if baseline is None:
        baseline = _v15_rollout(scenario, anchor, anchor_payload)
    result = result or env.summary()
    mode = "v15_fallback" if high_level == 0 else "v26_mpc"
    row = dict(seed=int(scenario["seed"]), scenario=scenario, result=result,
        baseline=baseline, mode=mode, high_level_action=int(high_level),
        high_level_action_name=HIGH_LEVEL_ACTIONS[int(high_level)], action=action,
        parameters=parameters, observed_context=(None if env.trajectory_decision is None else
            env.trajectory_decision.get("context")), predicted_branches=branches,
        prediction_exact=exact,
        fallback_reason=("no_safe_predicted_hit15" if mode == "v15_fallback" and branches
                         else "no_postcatch_decision" if mode == "v15_fallback" else None),
        paired_utility_delta=float(episode_utility(result)-episode_utility(baseline)),
        wall_s=time.perf_counter()-started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, row)
    return row


def _rates(summary):
    return (summary["released"]/max(1, summary["captured"]),
            summary["hit15"]/max(1, summary["released"]))


def hybrid_summary(rows):
    candidate = summarize(rows)
    baseline = summarize([dict(seed=row["seed"], scenario=row["scenario"],
        result=row["baseline"]) for row in rows])
    captured = [row for row in rows if row["result"].get("captured")]
    takeovers = [row for row in captured if row["mode"] == "v26_mpc"]
    fallbacks = [row for row in captured if row["mode"] == "v15_fallback"]
    safe_positive = [row for row in takeovers if row["result"].get("hit15") and
        not row["result"].get("grip_broken") and row["result"].get("max_joint_deg", 1e9) <= 36.
        and row["paired_utility_delta"] > 0.]
    release_rate, hit_rate = _rates(candidate)
    takeover_fraction = len(takeovers)/max(1, len(captured))
    positive_fraction = len(safe_positive)/max(1, len(captured))
    exact = all(bool(row.get("prediction_exact")) for row in captured)
    fallback_exact = all(row["result"] == row["baseline"] for row in fallbacks)
    safety = all(not row["result"].get("grip_broken") and
        row["result"].get("max_joint_deg", 1e9) <= 36. for row in rows)
    candidate_error = candidate.get("mean_landing_error_m")
    baseline_error = baseline.get("mean_landing_error_m")
    landing_guard = (candidate_error is not None and baseline_error is not None and
                     candidate_error <= baseline_error + .005)
    checks = dict(capture_nonregression=candidate["captured"] >= baseline["captured"],
        release_nonregression=candidate["released"] >= baseline["released"],
        hit15_nonregression=candidate["hit15"] >= baseline["hit15"],
        release_given_capture=release_rate >= .9, hit15_given_release=hit_rate >= .9,
        v26_takeover_fraction=takeover_fraction >= .5,
        safe_positive_fraction=positive_fraction >= .25,
        landing_noninferiority_5mm=landing_guard,
        exact_twin_or_fallback_replay=exact, exact_v15_fallback=fallback_exact,
        safety=safety)
    actions = [row["high_level_action"] for row in captured]
    return dict(candidate=candidate, baseline=baseline,
        release_given_capture=release_rate, hit15_given_release=hit_rate,
        v26_takeovers=len(takeovers), v15_fallbacks=len(fallbacks),
        v26_takeover_fraction=takeover_fraction,
        safe_positive=len(safe_positive), safe_positive_fraction=positive_fraction,
        paired_mean_utility_delta=float(np.mean([row["paired_utility_delta"] for row in rows])),
        selected_high_level_action_counts=np.bincount(
            actions, minlength=len(HIGH_LEVEL_ACTIONS)).tolist(),
        exact_decision_episodes=sum(bool(row.get("prediction_exact")) for row in captured),
        checks=checks, qualified_hybrid_teacher=all(checks.values()))


def _require_development(report):
    if (report.get("schema") != MPC_SCHEMA or report.get("source_hash") != mpc_source_hash() or
            report.get("kind") != "causal_terminal_mpc_pilot" or
            report.get("episodes_completed", 0) < 100 or
            report.get("prediction_exact_episodes") != report.get("planned_episodes")):
        raise ValueError("hybrid validation requires the complete current V26 causal-MPC pilot")
    checks = report.get("checks", {})
    required = ("capture_nonregression", "release_given_capture", "hit15_given_release",
                "safe_positive_fraction", "exact_twin_prediction", "safety")
    if not all(checks.get(name) for name in required):
        raise ValueError("causal-MPC development report failed a prerequisite other than V15 fallback")


def validate(args):
    development = read_json(args.development)
    _require_development(development)
    oracle = read_json(args.oracle)
    if (oracle.get("source_hash") != oracle_source_hash() or
            oracle.get("kind") != "terminal_twin_oracle" or
            not oracle.get("action_space_feasible")):
        raise ValueError("hybrid validation requires the completed feasible V26 oracle")
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    used = (_v25_contract_seeds() | _extract_manifest_seeds(oracle) |
            _extract_manifest_seeds(development))
    current = {int(row["seed"]) for row in manifest["scenarios"]}
    if used & current:
        raise ValueError(f"V26 hybrid validation overlaps prior scenarios: {sorted(used & current)[:5]}")
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, 2.)
    contract = dict(schema=SCHEMA, source_hash=source_hash(), kind="hybrid_teacher_validation",
        development_sha256=file_hash(args.development),
        oracle_sha256=file_hash(args.oracle), oracle_source_hash=oracle["source_hash"],
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        manifest=manifest, dynamic=asdict(dynamic), high_level_actions=list(HIGH_LEVEL_ACTIONS),
        decision_rule="v26_best_safe_hit15_else_exact_v15_fallback",
        decision_information="current_measured_postcatch_state_and_terminal_twin_predictions",
        fallback_execution="exact_v15_episode_passthrough",
        exact_v15_paired_baseline=True)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "validation_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("V26 hybrid validation contract changed; use a new output directory")
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
                emit(dict(stage="v26_hybrid_validation", completed=completed,
                    remaining=len(futures)-completed, seed=row["seed"], mode=row["mode"],
                    action=row["action"], captured=row["result"].get("captured"),
                    released=row["result"].get("released"), hit15=row["result"].get("hit15"),
                    prediction_exact=row["prediction_exact"]))
    rows = [read_json(args.out / "episodes" / f"{scenario['seed']}.json")
            for scenario in manifest["scenarios"]]
    report = dict(**contract, **hybrid_summary(rows), episodes_completed=len(rows))
    write_json(args.out / "validation_report.json", report)
    emit(dict(episodes=len(rows), qualified=report["qualified_hybrid_teacher"],
        release_given_capture=report["release_given_capture"],
        hit15_given_release=report["hit15_given_release"],
        v26_takeover_fraction=report["v26_takeover_fraction"],
        safe_positive_fraction=report["safe_positive_fraction"],
        v15_fallbacks=report["v15_fallbacks"], checks=report["checks"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path,
        default=DEFAULT_ROOT / "stage1_oracle" / "oracle_report.json")
    parser.add_argument("--development", type=Path,
        default=DEFAULT_ROOT / "stage2_causal_mpc_pilot" / "pilot_report.json")
    parser.add_argument("--out", type=Path,
        default=DEFAULT_ROOT / "stage3_hybrid_validation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33200001)
    parser.add_argument("--design-seed", type=int, default=20261302)
    parser.add_argument("--decision-interval", type=float, default=.5)
    parser.add_argument("--force-step", type=float, default=.05)
    parser.add_argument("--radius-step", type=float, default=.0225)
    parser.add_argument("--release-tolerance", type=float, default=.08)
    args = parser.parse_args()
    if args.episodes < 5 or args.workers < 1:
        parser.error("episodes must be at least five and workers must be positive")
    validate(args)


if __name__ == "__main__":
    main()
