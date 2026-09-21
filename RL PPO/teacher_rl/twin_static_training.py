"""V26 static terminal-oracle pretraining and independent closed-loop pilot."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .anchored_dynamic_env import TRAJECTORY_ACTIONS, initial_action
from .anchored_dynamic_rl import load_pretraining_data, _v15_rollout
from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_env import episode_utility
from .contextual_kernel import load_calibration
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json, summarize
from .improved_teacher import ImprovedRecipe
from .twin_mpc_ppo import (DEFAULT_CALIBRATION, SCHEMA as ORACLE_SCHEMA,
    _extract_manifest_seeds, _v25_contract_seeds, make_env,
    source_hash as oracle_source_hash)
from .twin_residual_env import HOLD_ACTION, TwinResidualConfig
from .twin_static_model import (ACTION_COUNT, OUTCOME_SIZE, SCHEMA,
    StaticTerminalPolicy, load_policy, save_policy)


DEFAULT_ROOT = Path("teacher_runs/v26_twin_mpc_ppo")
DEFAULT_INVENTORY = Path("teacher_runs/v23_contextual/combined_dataset_inventory_v2.json")


def source_hash():
    digest = hashlib.sha256((SCHEMA + oracle_source_hash()).encode())
    for name in ("twin_static_model.py", "twin_static_training.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _softmax(values, temperature):
    values = np.asarray(values, dtype=np.float32)
    shifted = (values-values.max())/float(temperature)
    probability = np.exp(shifted)
    return (probability/probability.sum()).astype(np.float32)


def _oracle_action_score(features):
    return float(80.*features["released"] + 40.*features["hit15"]
        - 80.*min(float(features["terminal_error_m"]), 1.)
        - .15*float(features["catch_to_release_s"])
        - 50.*features["unsafe"])


def _oracle_rows(path):
    report = read_json(path)
    if (report.get("schema") != ORACLE_SCHEMA or
            report.get("source_hash") != oracle_source_hash() or
            report.get("kind") != "terminal_twin_oracle" or
            not report.get("action_space_feasible") or
            report.get("episodes_completed") != len(report.get("manifest", {}).get("scenarios", []))):
        raise ValueError("pretraining requires a complete feasible V26 oracle report")
    root = Path(path).parent / "episodes"
    rows = []
    for scenario in report["manifest"]["scenarios"]:
        payload = read_json(root / f"{scenario['seed']}.json")
        if payload.get("context") is None:
            continue
        features = [item["features"] for item in payload["fixed_actions"]]
        scores = np.asarray([_oracle_action_score(item) for item in features], np.float32)
        outcomes = np.asarray([[float(item["released"]), float(item["hit15"]),
            float(item["utility"])/40., min(float(item["terminal_error_m"]), 1.),
            min(float(item["catch_to_release_s"]), 18.)/18.] for item in features],
            dtype=np.float32)
        rows.append(dict(seed=int(payload["seed"]), split=scenario["split"],
            context=np.asarray(payload["context"], np.float32),
            actor_target=_softmax(scores, 3.), outcomes=outcomes,
            baseline_utility=float(episode_utility(payload["baseline"])),
            scores=scores, features=features))
    if len(rows) != report["captured_contexts"]:
        raise ValueError("V26 oracle captured-context count differs from episode files")
    return rows, report


def _old_rows(inventory, init, calibration):
    train, validation, identity = load_pretraining_data(inventory, init, calibration)
    for row in train + validation:
        row["actor_target"] = _softmax(row["utilities"][1:], 8.)
    return train, validation, identity


def _metrics(model, old_rows, oracle_rows):
    model.eval()
    with torch.no_grad():
        old_contexts = torch.tensor(np.vstack([row["context"] for row in old_rows]))
        old_targets = torch.tensor(np.vstack([row["actor_target"] for row in old_rows]))
        old_logits, _, _ = model(old_contexts)
        old_ce = -(old_targets*old_logits.log_softmax(-1)).sum(-1).mean()
        contexts = torch.tensor(np.vstack([row["context"] for row in oracle_rows]))
        targets = torch.tensor(np.vstack([row["actor_target"] for row in oracle_rows]))
        logits, _, predicted = model(contexts)
        oracle_ce = -(targets*logits.log_softmax(-1)).sum(-1).mean()
        selected = logits.argmax(-1).numpy()
        true = np.stack([row["outcomes"] for row in oracle_rows])
        selected_true = true[np.arange(len(true)), selected]
        outcome_target = torch.tensor(true)
        release_loss = F.binary_cross_entropy_with_logits(
            predicted[:, :, 0], outcome_target[:, :, 0])
        hit_loss = F.binary_cross_entropy_with_logits(
            predicted[:, :, 1], outcome_target[:, :, 1])
        regression = F.smooth_l1_loss(predicted[:, :, 2:], outcome_target[:, :, 2:])
    released = selected_true[:, 0] > .5
    hit = selected_true[:, 1] > .5
    deltas = selected_true[:, 2]*40.-np.asarray(
        [row["baseline_utility"] for row in oracle_rows])
    teacher = np.asarray([row["scores"].argmax() for row in oracle_rows])
    model.train()
    return dict(old_actor_cross_entropy=float(old_ce), oracle_actor_cross_entropy=float(oracle_ce),
        oracle_top1=float(np.mean(selected == teacher)), outcome_release_bce=float(release_loss),
        outcome_hit_bce=float(hit_loss), outcome_regression=float(regression),
        selected_release_rate=float(released.mean()),
        selected_hit_given_release=float(hit[released].mean()) if released.any() else 0.,
        selected_safe_positive_fraction=float(np.mean((hit) & (deltas > 0.))),
        selected_mean_utility_delta=float(deltas.mean()),
        selected_action_counts=np.bincount(selected, minlength=ACTION_COUNT).tolist())


def pretrain(args):
    oracle_rows, oracle_report = _oracle_rows(args.oracle)
    old_train, old_validation, old_identity = _old_rows(
        args.inventory, args.init, args.calibration)
    new_train = [row for row in oracle_rows if row["split"] == "train"]
    new_validation = [row for row in oracle_rows if row["split"] == "validation"]
    if not new_train or not new_validation:
        raise ValueError("V26 oracle needs captured train and validation contexts")
    normalization = np.vstack([row["context"] for row in old_train + new_train])
    mean = normalization.mean(0)
    scale = np.maximum(normalization.std(0), .05)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    model = StaticTerminalPolicy(mean, scale, args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    old_contexts = torch.tensor(np.vstack([row["context"] for row in old_train]))
    old_targets = torch.tensor(np.vstack([row["actor_target"] for row in old_train]))
    new_contexts = torch.tensor(np.vstack([row["context"] for row in new_train]))
    new_targets = torch.tensor(np.vstack([row["actor_target"] for row in new_train]))
    outcome_targets = torch.tensor(np.stack([row["outcomes"] for row in new_train]))
    history, best, best_key = [], None, None
    for epoch in range(1, args.epochs+1):
        old_logits, _, _ = model(old_contexts)
        new_logits, _, predicted = model(new_contexts)
        old_actor = -(old_targets*old_logits.log_softmax(-1)).sum(-1).mean()
        new_actor = -(new_targets*new_logits.log_softmax(-1)).sum(-1).mean()
        release = F.binary_cross_entropy_with_logits(predicted[:, :, 0], outcome_targets[:, :, 0])
        hit = F.binary_cross_entropy_with_logits(predicted[:, :, 1], outcome_targets[:, :, 1])
        regression = F.smooth_l1_loss(predicted[:, :, 2:], outcome_targets[:, :, 2:])
        loss = old_actor + new_actor + .5*(release+hit) + .25*regression
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            metrics = _metrics(model, old_validation, new_validation)
            metrics.update(epoch=epoch, train_loss=float(loss.detach()))
            history.append(metrics)
            key = (metrics["selected_release_rate"], metrics["selected_hit_given_release"],
                   metrics["selected_safe_positive_fraction"],
                   metrics["selected_mean_utility_delta"],
                   -metrics["oracle_actor_cross_entropy"])
            if best_key is None or key > best_key:
                best_key, best = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    metrics = _metrics(model, old_validation, new_validation)
    contract = dict(schema=SCHEMA, source_hash=source_hash(), kind="static_terminal_pretrain",
        oracle_sha256=file_hash(args.oracle), oracle_source_hash=oracle_report["source_hash"],
        inventory_sha256=file_hash(args.inventory), old_data=old_identity,
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        old_train_contexts=len(old_train), old_validation_contexts=len(old_validation),
        oracle_train_contexts=len(new_train), oracle_validation_contexts=len(new_validation),
        hidden=args.hidden, epochs=args.epochs,
        action_semantics="one_postcatch_choice_from_fixed_3x3_force_radius_grid")
    save_policy(args.out, model, **contract, metrics=metrics)
    write_json(args.out.parent / "pretrain_contract.json", contract)
    write_json(args.out.parent / "pretrain_history.json", history)
    emit(dict(checkpoint=str(args.out), metrics=metrics,
        contexts=dict(old_train=len(old_train), old_validation=len(old_validation),
                      oracle_train=len(new_train), oracle_validation=len(new_validation))))


def pilot_worker(job):
    scenario, anchor_path, calibration_path, checkpoint_path, dynamic_data, output_path = job
    torch.set_num_threads(1)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, anchor_payload = load_anchor(anchor_path)
    calibration, _ = load_calibration(calibration_path, anchor_path)
    policy, payload = load_policy(checkpoint_path)
    if payload.get("source_hash") != source_hash():
        raise ValueError("pilot checkpoint was produced by different static-selector source")
    holder = {}

    def selector(context):
        action, _, value, prediction = policy.act(context, deterministic=True)
        holder.update(action=action, context=np.asarray(context, np.float32).tolist(),
                      value=value, prediction=np.asarray(prediction).tolist())
        return initial_action(action+1)

    env = make_env(scenario, anchor, anchor_payload, calibration, selector,
                   TwinResidualConfig(**dynamic_data))
    env.reset(scenario["seed"])
    result = None
    started = time.perf_counter()
    while not env.done:
        if env.dynamic_due():
            env.policy_observation()
            env.apply_joint_action(HOLD_ACTION)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    result = result or env.summary()
    baseline = _v15_rollout(scenario, anchor, anchor_payload)
    row = dict(seed=int(scenario["seed"]), scenario=scenario, result=result,
        baseline=baseline, context=holder.get("context"), action=holder.get("action"),
        predicted_outcome=holder.get("prediction"), policy_value=holder.get("value"),
        paired_utility_delta=float(episode_utility(result)-episode_utility(baseline)),
        wall_s=time.perf_counter()-started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, row)
    return row


def pilot_summary(rows):
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
    safety = all(not row["result"].get("grip_broken") and
                 row["result"].get("max_joint_deg", 1e9) <= 36. for row in rows)
    checks = dict(capture_nonregression=candidate["captured"] >= baseline["captured"],
        release_given_capture=release_rate >= .8, hit15_given_release=hit_rate >= .9,
        safe_positive_fraction=positive_rate >= .25, safety=safety)
    actions = [row["action"] for row in captured if row.get("action") is not None]
    return dict(candidate=candidate, baseline=baseline, release_given_capture=release_rate,
        hit15_given_release=hit_rate, safe_positive=len(safe_positive),
        safe_positive_fraction=positive_rate,
        paired_mean_utility_delta=float(np.mean([row["paired_utility_delta"] for row in rows])),
        selected_action_counts=np.bincount(actions, minlength=ACTION_COUNT).tolist(),
        checks=checks, qualified_for_one_step_ppo=all(checks.values()))


def pilot(args):
    _, payload = load_policy(args.checkpoint)
    if (payload.get("source_hash") != source_hash() or
            payload.get("kind") != "static_terminal_pretrain"):
        raise ValueError("pilot requires the current V26 static pretrain checkpoint")
    anchor, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    if (payload.get("anchor_sha256") != file_hash(args.init) or
            payload.get("calibration_sha256") != file_hash(args.calibration)):
        raise ValueError("pilot checkpoint anchor/calibration mismatch")
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    used = _v25_contract_seeds()
    oracle_report = read_json(args.oracle)
    used.update(_extract_manifest_seeds(oracle_report))
    overlap = used & {int(row["seed"]) for row in manifest["scenarios"]}
    if overlap:
        raise ValueError(f"V26 pilot overlaps earlier scenarios: {sorted(overlap)[:5]}")
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, 2.)
    contract = dict(schema=SCHEMA, source_hash=source_hash(), kind="static_terminal_pilot",
        checkpoint_sha256=file_hash(args.checkpoint), oracle_sha256=file_hash(args.oracle),
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        manifest=manifest, dynamic=asdict(dynamic), deterministic=True,
        exact_v15_paired_baseline=True)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "pilot_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("V26 pilot contract changed; use a new output directory")
    write_json(contract_path, contract)
    jobs = []
    for scenario in manifest["scenarios"]:
        path = args.out / "episodes" / f"{scenario['seed']}.json"
        if not path.exists():
            jobs.append((scenario, str(args.init.resolve()), str(args.calibration.resolve()),
                str(args.checkpoint.resolve()), asdict(dynamic), str(path.resolve())))
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            futures = [pool.submit(pilot_worker, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                emit(dict(stage="v26_static_pilot", completed=completed,
                    remaining=len(futures)-completed, seed=row["seed"], action=row["action"],
                    captured=row["result"].get("captured"),
                    released=row["result"].get("released"), hit15=row["result"].get("hit15")))
    rows = [read_json(args.out / "episodes" / f"{scenario['seed']}.json")
            for scenario in manifest["scenarios"]]
    report = dict(**contract, **pilot_summary(rows), episodes_completed=len(rows))
    write_json(args.out / "pilot_report.json", report)
    emit(dict(episodes=len(rows), qualified=report["qualified_for_one_step_ppo"],
        release_given_capture=report["release_given_capture"],
        hit15_given_release=report["hit15_given_release"],
        safe_positive_fraction=report["safe_positive_fraction"],
        selected_action_counts=report["selected_action_counts"]))


def _common(parser):
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("pretrain")
    _common(command)
    command.add_argument("--oracle", type=Path,
        default=DEFAULT_ROOT / "stage1_oracle" / "oracle_report.json")
    command.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    command.add_argument("--out", type=Path,
        default=DEFAULT_ROOT / "stage2_static_pretrain" / "policy_init.pt")
    command.add_argument("--hidden", type=int, default=32)
    command.add_argument("--epochs", type=int, default=400)
    command.add_argument("--lr", type=float, default=1e-3)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--random-seed", type=int, default=20261202)
    command = sub.add_parser("pilot")
    _common(command)
    command.add_argument("--oracle", type=Path,
        default=DEFAULT_ROOT / "stage1_oracle" / "oracle_report.json")
    command.add_argument("--checkpoint", type=Path,
        default=DEFAULT_ROOT / "stage2_static_pretrain" / "policy_init.pt")
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT / "stage3_static_pilot")
    command.add_argument("--episodes", type=int, default=100)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=33100001)
    command.add_argument("--design-seed", type=int, default=20261203)
    command.add_argument("--decision-interval", type=float, default=.5)
    command.add_argument("--force-step", type=float, default=.05)
    command.add_argument("--radius-step", type=float, default=.0225)
    command.add_argument("--release-tolerance", type=float, default=.08)
    args = parser.parse_args()
    for name in ("hidden", "epochs", "episodes", "workers"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    (pretrain if args.command == "pretrain" else pilot)(args)


if __name__ == "__main__":
    main()
