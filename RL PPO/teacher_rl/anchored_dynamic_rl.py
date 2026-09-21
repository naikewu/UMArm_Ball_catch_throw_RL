"""Pretrain and optimize the V15-anchored low-frequency post-catch policy.

Workflow:
  1. ``pretrain`` distils all 1,200 V23 outcomes into a conservative selector
     over exact V15 passthrough plus the 3x3 force/radius grid.  The same paired
     outcomes teach the sequence actor which neighbouring motion improves a
     selected trajectory.
  2. ``train`` performs on-policy PPO after capture while V15 remains frozen.
  3. ``evaluate`` compares the selected checkpoint with V15 on fresh scenarios.

PPO can only choose V15 passthrough or adjust bounded force/radius settings.
The calibrated 150 Hz guard owns release timing.  PPO cannot alter approach/capture control or bypass
the release envelope, joint guard, or landing calibration.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .anchored_dynamic_env import (SCHEMA, DynamicControlConfig, INITIAL_ACTIONS,
    INITIAL_ACTION_NAMES, MOTION_NAMES, OBSERVATION_NAMES, RELEASE_NAMES,
    TRAJECTORY_ACTIONS, AnchoredDynamicEnv, initial_action)
from .anchored_dynamic_model import (CONTEXT_SIZE, OBS_SIZE, AnchoredDynamicPolicy,
    TAKEOVER_CONFIDENCE, load_policy, save_policy)
from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_kernel import load_calibration
from .contextual_env import CONTEXT_NAMES, episode_utility
from .contextual_rl import previous_seeds as contextual_previous_seeds
from .data import write_json
from .envelope_teacher_rl import report as envelope_report
from .improved_rl import (_init_worker, file_hash, load_anchor, read_json,
    summarize)
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv


DEFAULT_ROOT = Path("teacher_runs/v15_anchor_dynamic_ppo_v2")
DEFAULT_PRETRAIN = DEFAULT_ROOT / "pretrain" / "policy_init.pt"
DEFAULT_TRAIN = DEFAULT_ROOT / "formal"
DEFAULT_CALIBRATION = Path(
    "teacher_runs/v23_contextual/warm_kernel_v1/release_calibration_v23_kernel.json")
DEFAULT_INVENTORY = Path(
    "teacher_runs/v23_contextual/combined_dataset_inventory_v2.json")
TAKEOVER_MARGIN = 5.


def source_hash():
    digest = hashlib.sha256(SCHEMA.encode())
    for name in ("anchored_dynamic_env.py", "anchored_dynamic_model.py",
                 "anchored_dynamic_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    import json
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def report(rows, baseline):
    """Apply controller-specific checks only to rows where the new controller ran."""
    result = envelope_report(rows, baseline)
    takeover = [row for row in rows if row["result"].get("initial_policy_mode") ==
                "bounded_trajectory_takeover"]
    captured = [row for row in takeover if row["result"].get("captured")]
    audits = [row["result"].get("calibrated_release", {}) for row in takeover]
    result["checks"]["immediate_handover"] = all(
        row["result"].get("catch_to_orbit_s") is not None and
        row["result"]["catch_to_orbit_s"] <= .02 for row in captured)
    result["checks"]["no_sustained_pause"] = all(
        row["result"].get("longest_postcatch_pause_s") is not None and
        row["result"]["longest_postcatch_pause_s"] <= .1 for row in captured)
    result["checks"]["calibrated_release_coverage"] = sum(
        bool(audit.get("scheduled")) for audit in audits) >= sum(
            bool(row["result"].get("released")) for row in takeover)
    result["checks"]["release_window"] = all(not audit.get("scheduled") or
        2. <= audit["release_age_s"] <= 18. for audit in audits)
    result["checks"]["calibration_age_cap"] = all(not audit.get("scheduled") or
        audit["feature_age_s"] <= 16. for audit in audits)
    # V19 already applies the stricter paired +5 mm guard.  Requiring an
    # additional absolute 7 cm mean makes an exact V15 passthrough fail merely
    # because a small validation batch is difficult.
    result["checks"]["landing_guard"] = result["checks"]["landing_noninferiority_5mm"]
    result["eligible"] = all(result["checks"].values())
    result["task_nonregression"] = (
        result["summary"]["hit15"] >= result["baseline"]["hit15"] and
        result["summary"]["captured"] >= result["baseline"]["captured"] and
        result["summary"]["released"] >= result["baseline"]["released"])
    bins = result["parameter_bins"]
    result["absolute_generalization_pass"] = bool(result["eligible"] and
        result["task_nonregression"] and len(rows) >= 80 and
        result["worst_parameter_bin_hit15_rate"] >= .8 and
        all(group["episodes"] >= 10 for groups in bins.values() for group in groups))
    result["takeover_episodes"] = len(takeover)
    result["v15_passthrough_episodes"] = sum(
        row["result"].get("initial_policy_mode") == "v15_passthrough" for row in rows)
    return result


def _softmax(values, temperature=8.):
    values = np.asarray(values, dtype=np.float64)
    shifted = (values - values.max()) / temperature
    probability = np.exp(shifted)
    return (probability / probability.sum()).astype(np.float32)


def load_pretraining_data(inventory_path, anchor_path, calibration_path):
    inventory = read_json(inventory_path)
    if (not inventory.get("complete") or inventory.get("completed_rows", 0) < 1000 or
            inventory.get("action_outcomes", 0) < 9 * inventory.get("captured_contexts", -1) or
            inventory.get("action_outcomes", 0) + inventory.get("baseline_outcomes", 0)
            != inventory.get("completed_rows", -1)):
        raise ValueError("pretraining requires the complete >=1000-row V23 inventory")
    anchor_sha, calibration_sha = file_hash(anchor_path), file_hash(calibration_path)
    records, contracts = [], []
    observed_rows = observed_baselines = 0
    for source in inventory["sources"]:
        directory = Path(source["directory"])
        contract_path = directory / "sweep_contract.json"
        contract = read_json(contract_path)
        contracts.append(dict(path=str(contract_path.resolve()), sha256=file_hash(contract_path)))
        if (contract.get("schema") != "can_contextual_trajectory_v23" or
                contract.get("calibration_mode") != "warmed" or
                contract.get("anchor_sha256") != anchor_sha or
                contract.get("calibration_sha256") != calibration_sha or
                len(contract.get("actions", [])) != len(TRAJECTORY_ACTIONS)):
            raise ValueError(f"stale or incompatible V23 sweep: {directory}")
        for index, (force, radius) in enumerate(TRAJECTORY_ACTIONS):
            expected = initial_action(index + 1)
            if not np.allclose(contract["actions"][index], expected, atol=1e-6):
                raise ValueError("V23 action order does not match the anchored 3x3 catalogue")
        for scenario in contract["manifest"]["scenarios"]:
            candidates = []
            for index in range(len(TRAJECTORY_ACTIONS)):
                path = directory / "episodes" / f"{scenario['seed']}_a{index:03d}.json"
                if not path.is_file():
                    raise ValueError(f"missing V23 outcome {path}")
                candidates.append(read_json(path))
                observed_rows += 1
            baseline_path = directory / "episodes" / f"{scenario['seed']}_bc.json"
            if not baseline_path.is_file():
                raise ValueError(f"missing paired V15 baseline {baseline_path}")
            baseline = read_json(baseline_path)
            observed_baselines += 1
            decision = candidates[0]["result"].get("trajectory_decision")
            if decision is None:
                # V15 did not capture this scenario.  It never reaches the new
                # actor online, so it is not a valid post-catch training context.
                continue
            contexts = [row["result"]["trajectory_decision"]["context"]
                        for row in candidates]
            if any(not np.allclose(contexts[0], other, atol=1e-7) for other in contexts[1:]):
                raise ValueError("paired V23 actions do not share one measured handover context")
            utilities = np.asarray([episode_utility(baseline["result"])] +
                [row.get("utility", episode_utility(row["result"])) for row in candidates],
                dtype=np.float32)
            target_utilities = utilities.copy()
            target_utilities[0] += TAKEOVER_MARGIN
            context = np.asarray(contexts[0], dtype=np.float32)
            if context.shape != (CONTEXT_SIZE,) or not np.isfinite(context).all():
                raise ValueError("V23 handover context is invalid")
            records.append(dict(seed=scenario["seed"], split=scenario["split"],
                context=context, utilities=utilities, target=_softmax(target_utilities)))
    if (observed_rows != inventory["action_outcomes"] or
            observed_baselines != inventory["baseline_outcomes"] or
            len(records) != inventory["captured_contexts"]):
        raise ValueError("V23 inventory counts differ from the files on disk")
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("pretraining needs scenario-disjoint train and validation rows")
    identity = dict(inventory_sha256=file_hash(inventory_path),
        inventory=inventory, sweep_contracts=contracts, observed_rows=observed_rows,
        observed_baselines=observed_baselines,
        captured_contexts=len(records), train_contexts=len(train),
        validation_contexts=len(validation), anchor_sha256=anchor_sha,
        calibration_sha256=calibration_sha)
    return train, validation, identity


def _pretrain_metrics(model, rows):
    observations = np.zeros((len(rows), OBS_SIZE), dtype=np.float32)
    observations[:, :CONTEXT_SIZE] = np.vstack([row["context"] for row in rows])
    utilities = np.vstack([row["utilities"] for row in rows])
    targets = torch.tensor(np.vstack([row["target"] for row in rows]))
    with torch.no_grad():
        distribution = model.initial_distribution(torch.tensor(observations))
        candidate = distribution.probs[:, 1:].argmax(-1) + 1
        candidate_probability = distribution.probs.gather(-1, candidate[:, None]).squeeze(-1)
        selected = torch.where(candidate_probability >= TAKEOVER_CONFIDENCE,
                               candidate, torch.zeros_like(candidate)).numpy()
        loss = -(targets * distribution.logits.log_softmax(-1)).sum(-1).mean().item()
    chosen = utilities[np.arange(len(rows)), selected]
    oracle = utilities.max(-1)
    return dict(cross_entropy=float(loss), selected_mean_utility=float(chosen.mean()),
        oracle_mean_utility=float(oracle.mean()), oracle_action_rate=float(
            np.mean(selected == utilities.argmax(-1))),
        v15_passthrough_rate=float(np.mean(selected == 0)),
        nonregression_rate=float(np.mean(chosen + 1e-6 >= utilities[:, 0])))


def _dynamic_pretraining_data(rows):
    """Turn paired absolute outcomes into local, one-step motion supervision."""
    observations, labels = [], []
    for row in rows:
        candidate_utilities = row["utilities"][1:]
        for current in range(len(TRAJECTORY_ACTIONS)):
            force_index, radius_index = divmod(current, 3)
            neighbours = [(0, current)]
            if force_index > 0:
                neighbours.append((1, current - 3))
            if force_index < 2:
                neighbours.append((2, current + 3))
            if radius_index > 0:
                neighbours.append((3, current - 1))
            if radius_index < 2:
                neighbours.append((4, current + 1))
            motion, _ = max(neighbours, key=lambda item: candidate_utilities[item[1]])
            action = initial_action(current + 1)
            extra = np.r_[action, 0., 1., 0., 0.,
                          np.eye(len(MOTION_NAMES), dtype=np.float32)[0], 0.]
            observations.append(np.r_[row["context"], extra].astype(np.float32))
            labels.append(motion)
    return np.vstack(observations), np.asarray(labels, dtype=np.int64)


def _dynamic_accuracy(model, rows):
    observations, labels = _dynamic_pretraining_data(rows)
    with torch.no_grad():
        motion, _ = model.dynamic_distributions(torch.tensor(observations))
        predicted = motion.probs.argmax(-1).numpy()
    return float(np.mean(predicted == labels))


def pretrain(args):
    load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    train_rows, validation_rows, data_identity = load_pretraining_data(
        args.inventory, args.init, args.calibration)
    contexts = np.vstack([row["context"] for row in train_rows])
    mean, scale = contexts.mean(0), contexts.std(0)
    scale = np.maximum(scale, .05)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    model = AnchoredDynamicPolicy(mean, scale, hidden=args.hidden)
    optimizer = torch.optim.Adam([*model.initial_actor.parameters(),
        *model.dynamic_trunk.parameters(), *model.motion_head.parameters()], lr=args.lr,
                                 weight_decay=args.weight_decay)
    observations = np.zeros((len(train_rows), OBS_SIZE), dtype=np.float32)
    observations[:, :CONTEXT_SIZE] = contexts
    observations = torch.tensor(observations)
    targets = torch.tensor(np.vstack([row["target"] for row in train_rows]))
    dynamic_observations, dynamic_labels = _dynamic_pretraining_data(train_rows)
    dynamic_observations = torch.tensor(dynamic_observations)
    dynamic_labels = torch.tensor(dynamic_labels)
    best_state, best_key, history = None, None, []
    for epoch in range(1, args.epochs + 1):
        for indices in torch.randperm(len(train_rows)).split(args.batch_size):
            logits = model.initial_actor(model.normalized(observations[indices])[:, :CONTEXT_SIZE])
            initial_loss = -(targets[indices] * logits.log_softmax(-1)).sum(-1).mean()
            dynamic_indices = torch.randint(len(dynamic_labels),
                (max(args.batch_size, len(indices)),))
            motion, _ = model.dynamic_distributions(dynamic_observations[dynamic_indices])
            dynamic_loss = F.cross_entropy(motion.logits, dynamic_labels[dynamic_indices])
            loss = initial_loss + .5 * dynamic_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = _pretrain_metrics(model, train_rows)
            validation_metrics = _pretrain_metrics(model, validation_rows)
            train_metrics["dynamic_motion_accuracy"] = _dynamic_accuracy(model, train_rows)
            validation_metrics["dynamic_motion_accuracy"] = _dynamic_accuracy(model, validation_rows)
            history.append(dict(epoch=epoch, train=train_metrics,
                                validation=validation_metrics))
            key = (validation_metrics["nonregression_rate"],
                   validation_metrics["selected_mean_utility"],
                   -validation_metrics["cross_entropy"])
            if best_key is None or key > best_key:
                best_key, best_state = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    final_train_metrics = _pretrain_metrics(model, train_rows)
    final_validation_metrics = _pretrain_metrics(model, validation_rows)
    final_train_metrics["dynamic_motion_accuracy"] = _dynamic_accuracy(model, train_rows)
    final_validation_metrics["dynamic_motion_accuracy"] = _dynamic_accuracy(model, validation_rows)
    contract = dict(schema=SCHEMA, source_hash=source_hash(), kind="anchored_dynamic_pretrain",
        data=data_identity, context_names=list(CONTEXT_NAMES),
        observation_names=list(OBSERVATION_NAMES),
        initial_actions=[None if row is None else list(row) for row in INITIAL_ACTIONS],
        initial_action_names=list(INITIAL_ACTION_NAMES), motion_names=list(MOTION_NAMES),
        release_names=list(RELEASE_NAMES), hidden=args.hidden, epochs=args.epochs,
        takeover_confidence=TAKEOVER_CONFIDENCE, takeover_margin=TAKEOVER_MARGIN,
        dynamic_pretraining="best_observed_neighbour_from_paired_3x3_outcomes",
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, random_seed=args.random_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_policy(args.out, model, schema=SCHEMA, source_hash=source_hash(),
                kind="anchored_dynamic_pretrain",
                contract=contract, metrics=dict(train=final_train_metrics,
                validation=final_validation_metrics))
    write_json(args.out.parent / "pretrain_contract.json", contract)
    write_json(args.out.parent / "pretrain_history.json", history)
    emit(dict(checkpoint=str(args.out), samples=data_identity["observed_rows"],
        captured_contexts=data_identity["captured_contexts"],
        metrics=final_validation_metrics))


def _sample_record(observation, phase, initial_index, motion, release, logp, value,
                   time_s):
    return dict(observation=np.asarray(observation, dtype=np.float32).tolist(), phase=int(phase),
        initial_action=int(initial_index), motion_action=int(motion),
        release_action=int(release), old_logp=float(logp), value=float(value),
        time_s=float(time_s), reward=0.)


def _progress_potential(observation):
    """Observable progress signal; it never participates in acceptance checks."""
    value = np.asarray(observation, dtype=np.float32)
    error = float(np.clip(value[OBSERVATION_NAMES.index(
        "min_calibrated_error_fraction")], 0., 1.))
    ready = float(np.clip(value[OBSERVATION_NAMES.index("safe_candidate_ready")], 0., 1.))
    joint_margin = float(np.clip(value[OBSERVATION_NAMES.index("joint_margin_deg")] / 12.,
                                 -1., 1.))
    return .25 * (1.5 * (1. - error) + .25 * ready + .25 * joint_margin)


def _finish_episode(samples, result, gamma, gae_lambda):
    if not samples:
        return samples
    decision = result.get("trajectory_decision") or {}
    config = result.get("contextual_config") or {}
    if decision.get("time_s") is not None and config.get("release_max_s") is not None:
        horizon = float(decision["time_s"]) + float(config["release_max_s"])
        samples = [sample for sample in samples
                   if sample["phase"] == 0 or sample["time_s"] <= horizon + 1e-9]
    if not samples:
        return samples
    for current, following in zip(samples[:-1], samples[1:]):
        shaping = _progress_potential(following["observation"]) - _progress_potential(
            current["observation"])
        current["reward"] += float(np.clip(shaping, -.5, .5))
        current["progress_reward"] = float(np.clip(shaping, -.5, .5))
    terminal = episode_utility(result) / 40.
    samples[-1]["reward"] += terminal
    samples[-1]["terminal_reward"] = float(terminal)
    advantage, next_value = 0., 0.
    for sample in reversed(samples):
        delta = sample["reward"] + gamma * next_value - sample["value"]
        advantage = delta + gamma * gae_lambda * advantage
        sample["advantage"] = float(advantage)
        sample["return"] = float(advantage + sample["value"])
        next_value = sample["value"]
    return samples


def _v15_rollout(scenario, anchor, anchor_payload):
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    env = ImprovedTeacherEnv(scenario, recipe)
    observation = env.reset(scenario["seed"])
    while not env.done:
        with torch.no_grad():
            action = anchor.distribution(torch.tensor(observation[None])).mean.tanh()[0].numpy()
        observation, _, _, result = env.step(action)
    result.setdefault("catch_to_orbit_s", None)
    result.setdefault("catch_to_release_s", None if result["release_time"] is None or
        result["capture_time"] is None else result["release_time"] - result["capture_time"])
    return result


def episode_worker(job):
    (scenario, anchor_path, calibration_path, checkpoint_path, dynamic_data,
     stochastic, gamma, gae_lambda, policy_seed) = job
    torch.set_num_threads(1)
    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed % 2**32)
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    anchor, anchor_payload = load_anchor(anchor_path)
    calibration, _ = load_calibration(calibration_path, anchor_path)
    policy, payload = load_policy(checkpoint_path, SCHEMA)
    if payload.get("source_hash") != source_hash():
        raise ValueError("rollout policy source does not match the running code")
    policy.eval()
    samples, holder = [], {"v15_passthrough": False}

    def selector(context):
        observation = holder["env"].initial_observation(context)
        index, logp, value = policy.act_initial(observation, deterministic=not stochastic)
        holder["v15_passthrough"] = index == 0
        if stochastic:
            samples.append(_sample_record(observation, 0, index, 0, 1, logp, value,
                                          holder["env"].obs["t_win"]))
        return initial_action(index)

    env = AnchoredDynamicEnv(scenario, ImprovedRecipe(**anchor_payload["recipe"]),
        anchor, calibration, selector, DynamicControlConfig(**dynamic_data))
    holder["env"] = env
    env.reset(scenario["seed"])
    started = time.perf_counter()
    result = None
    while not env.done:
        if env.dynamic_due():
            observation = env.policy_observation()
            motion, release, logp, value = policy.act_dynamic(
                observation, deterministic=not stochastic)
            if stochastic:
                samples.append(_sample_record(observation, 1, 0, motion,
                                              release, logp, value, env.obs["t_win"]))
            env.apply_dynamic(motion, release)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
        if holder["v15_passthrough"]:
            # Restarting through the unmodified V15 environment gives an exact
            # fallback rather than an approximation from the new controller.
            result = _v15_rollout(scenario, anchor, anchor_payload)
            break
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or
        env.capture_physics_time is None else env.controller.t_hand - env.capture_physics_time)
    result.setdefault("catch_to_release_s", None if result["release_time"] is None or
        result["capture_time"] is None else result["release_time"] - result["capture_time"])
    if stochastic:
        _finish_episode(samples, result, gamma, gae_lambda)
    result["initial_policy_mode"] = ("no_handover" if env.trajectory_decision is None else
        "v15_passthrough" if holder["v15_passthrough"] else "bounded_trajectory_takeover")
    return dict(seed=scenario["seed"], scenario=scenario, result=result,
        utility=episode_utility(result), samples=samples,
        wall_s=time.perf_counter() - started)


def baseline_worker(job):
    scenario, anchor_path = job
    torch.set_num_threads(1)
    anchor, payload = load_anchor(Path(anchor_path))
    result = _v15_rollout(scenario, anchor, payload)
    return dict(seed=scenario["seed"], scenario=scenario, result=result)


def run_jobs(pool, jobs, worker, label):
    futures = [pool.submit(worker, job) for job in jobs]
    rows = []
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        if len(rows) % 5 == 0 or len(rows) == len(futures):
            emit(dict(stage=label, completed=len(rows), total=len(futures),
                      seed=row["seed"], hit15=row["result"]["hit15"]))
    return sorted(rows, key=lambda row: row["seed"])


def _policy_jobs(manifest, args, checkpoint, stochastic, update=0):
    dynamic = asdict(DynamicControlConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance))
    return [(scenario, str(args.init.resolve()), str(args.calibration.resolve()),
        str(Path(checkpoint).resolve()), dynamic, stochastic, args.gamma,
        args.gae_lambda, (args.random_seed + update * 100003 + scenario["seed"]) % 2**32)
        for scenario in manifest]


def _previous_seeds(output):
    used = contextual_previous_seeds()
    root = DEFAULT_ROOT
    excluded = Path(output).resolve()
    if root.exists():
        for path in root.rglob("*contract.json"):
            if path.resolve().is_relative_to(excluded):
                continue
            record = read_json(path)
            for key in ("train_manifest", "validation_manifest", "manifest"):
                used.update(row["seed"] for row in record.get(key, {}).get("scenarios", []))
    return used


def _ensure_disjoint(groups):
    seen = set()
    for name, values in groups.items():
        values = set(values)
        overlap = seen & values
        if overlap:
            raise ValueError(f"seed overlap in {name}: {sorted(overlap)[:5]}")
        seen.update(values)


def _phase_standardize(values, phases):
    result = torch.empty_like(values)
    for phase in phases.unique():
        mask = phases == phase
        selected = values[mask]
        result[mask] = ((selected - selected.mean()) /
                        selected.std(unbiased=False).clamp_min(1e-8))
    return result


def _phase_balanced_mean(values, phases):
    groups = [values[phases == phase].mean() for phase in phases.unique()
              if (phases == phase).any()]
    return torch.stack(groups).mean()


def ppo_update(model, reference, optimizer, samples, args):
    if not samples:
        raise RuntimeError("rollout contains no post-catch decisions")
    observations = torch.tensor([row["observation"] for row in samples], dtype=torch.float32)
    phases = torch.tensor([row["phase"] for row in samples], dtype=torch.long)
    initial_actions = torch.tensor([row["initial_action"] for row in samples], dtype=torch.long)
    motion_actions = torch.tensor([row["motion_action"] for row in samples], dtype=torch.long)
    release_actions = torch.tensor([row["release_action"] for row in samples], dtype=torch.long)
    old_logp = torch.tensor([row["old_logp"] for row in samples], dtype=torch.float32)
    returns = torch.tensor([row["return"] for row in samples], dtype=torch.float32)
    advantage = torch.tensor([row["advantage"] for row in samples], dtype=torch.float32)
    advantage = _phase_standardize(advantage, phases)
    records, stopped = [], False
    for epoch in range(args.update_epochs):
        for indices in torch.randperm(len(samples)).split(args.minibatch_size):
            logp, entropy, value = model.evaluate(observations[indices], phases[indices],
                initial_actions[indices], motion_actions[indices], release_actions[indices])
            log_ratio = logp - old_logp[indices]
            ratio = log_ratio.exp()
            approximate_kl = float(((ratio - 1.) - log_ratio).mean().detach())
            if approximate_kl > args.target_kl:
                stopped = True
                break
            surrogate = torch.minimum(ratio * advantage[indices],
                ratio.clamp(1. - args.clip_ratio, 1. + args.clip_ratio) * advantage[indices])
            policy_loss = -_phase_balanced_mean(surrogate, phases[indices])
            value_loss = .5 * (value - returns[indices]).square().mean()
            anchor_kl = _phase_balanced_mean(model.kl_from(
                reference, observations[indices], phases[indices]), phases[indices])
            entropy_loss = _phase_balanced_mean(entropy, phases[indices])
            loss = (policy_loss + args.value_coefficient * value_loss
                    + args.anchor_kl * anchor_kl - args.entropy * entropy_loss)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            records.append((float(policy_loss.detach()), float(value_loss.detach()),
                float(anchor_kl.detach()), float(entropy_loss.detach()), approximate_kl))
        if stopped:
            break
    if not records:
        raise RuntimeError("PPO target KL rejected every minibatch")
    average = np.mean(records, axis=0)
    return dict(policy_loss=float(average[0]), value_loss=float(average[1]),
        anchor_kl=float(average[2]), entropy=float(average[3]),
        approximate_kl=float(average[4]), epochs=epoch + 1,
        kl_early_stop=stopped, samples=len(samples),
        initial_samples=int((phases == 0).sum()), dynamic_samples=int((phases == 1).sum()))


def _selection_score(result):
    summary = result["summary"]
    error = summary["mean_landing_error_m"]
    return (summary["hit15"], summary["released"], summary["captured"],
            -(error if error is not None else 1e9), -result["max_joint_deg"])


def _publish_best(output, update):
    source = output / "checkpoints" / f"selected_{update:04d}.pt"
    temporary = output / "ppo_best.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(output / "ppo_best.pt")


def train(args):
    anchor, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    _, pretrain_payload = load_policy(args.pretrain, SCHEMA)
    if (pretrain_payload.get("kind") != "anchored_dynamic_pretrain" or
            pretrain_payload.get("source_hash") != source_hash() or
            pretrain_payload.get("contract", {}).get("data", {}).get("anchor_sha256") != file_hash(args.init) or
            pretrain_payload.get("contract", {}).get("data", {}).get("calibration_sha256") != file_hash(args.calibration)):
        raise ValueError("pretrained policy, source, V15 anchor, or calibration differ")
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    train_manifest = scenarios(args.updates * args.episodes_per_update, args.seed,
                               args.design_seed, recipe)
    validation_manifest = scenarios(args.eval_episodes, args.eval_seed,
                                    args.design_seed + 100000, recipe)
    historical = _previous_seeds(args.out)
    _ensure_disjoint(dict(history=historical,
        train=[row["seed"] for row in train_manifest["scenarios"]],
        validation=[row["seed"] for row in validation_manifest["scenarios"]]))
    dynamic = DynamicControlConfig(args.decision_interval, args.force_step,
                                   args.radius_step, args.release_tolerance)
    contract = dict(schema=SCHEMA, source_hash=source_hash(), kind="anchored_dynamic_ppo",
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        pretrain_sha256=file_hash(args.pretrain), dynamic=asdict(dynamic),
        train_manifest=train_manifest, validation_manifest=validation_manifest,
        excluded_seeds=sorted(historical), updates=args.updates,
        episodes_per_update=args.episodes_per_update, eval_every=args.eval_every,
        ppo=dict(gamma=args.gamma, gae_lambda=args.gae_lambda, actor_lr=args.lr,
            critic_lr=args.critic_lr, clip_ratio=args.clip_ratio, entropy=args.entropy,
            anchor_kl=args.anchor_kl, target_kl=args.target_kl,
            update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
            value_coefficient=args.value_coefficient, max_grad_norm=args.max_grad_norm,
            advantage_normalization="per_policy_phase",
            actor_loss_weighting="equal_initial_and_dynamic",
            release_control="always_allow_150hz_calibrated_guard",
            terminal_credit="last_decision_inside_release_window",
            observable_progress_shaping=True),
        random_seed=args.random_seed, smoke=args.smoke)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "training_contract.json"
    start, best_update, best_score = 0, None, None
    if args.resume:
        if Path(args.resume).resolve() != (args.out / "ppo_latest.pt").resolve():
            raise ValueError("resume must point to this output directory's ppo_latest.pt")
        model, payload = load_policy(args.resume, SCHEMA)
        if payload.get("kind") != "anchored_dynamic_ppo" or payload.get("contract") != contract:
            raise ValueError("resume checkpoint contract differs from this run")
        if read_json(contract_path) != contract:
            raise ValueError("training_contract.json differs from the checkpoint")
        start, best_update = payload["update"], payload["best_update"]
        best_score = None if payload["best_score"] is None else tuple(payload["best_score"])
    else:
        if any(args.out.iterdir()):
            raise ValueError("training output is not empty; use --resume or a new directory")
        model, _ = load_policy(args.pretrain, SCHEMA)
        write_json(contract_path, contract)
    reference, _ = load_policy(args.pretrain, SCHEMA)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([
        dict(params=[*model.initial_actor.parameters(), *model.dynamic_trunk.parameters(),
                     *model.motion_head.parameters()], lr=args.lr),
        dict(params=model.critic.parameters(), lr=args.critic_lr)])
    if args.resume:
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    else:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    def save(update, path):
        save_policy(path, model, schema=SCHEMA, source_hash=source_hash(),
            kind="anchored_dynamic_ppo", contract=contract, update=update,
            best_update=best_update, best_score=None if best_score is None else list(best_score),
            optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate())

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        baseline_path = args.out / "validation_v15.json"
        if baseline_path.exists():
            baseline = read_json(baseline_path)
        else:
            baseline = run_jobs(pool, [(row, str(args.init.resolve()))
                for row in validation_manifest["scenarios"]], baseline_worker, "validation_v15")
            write_json(baseline_path, baseline)
        if not args.resume:
            init_snapshot = args.out / "rollout_policy.pt"
            save_policy(init_snapshot, model, schema=SCHEMA, source_hash=source_hash(),
                        kind="rollout", update=0)
            initial = run_jobs(pool, _policy_jobs(validation_manifest["scenarios"], args,
                init_snapshot, False), episode_worker, "validation_0000")
            checked = report(initial, baseline)
            write_json(args.out / "validation_0000.json", dict(update=0,
                selected=checked["eligible"] and checked["task_nonregression"],
                report=checked, rows=initial))
            if checked["eligible"] and checked["task_nonregression"]:
                best_update, best_score = 0, _selection_score(checked)
                save(0, args.out / "checkpoints" / "selected_0000.pt")
                _publish_best(args.out, 0)
            save(0, args.out / "ppo_latest.pt")
            if not args.smoke and not (checked["eligible"] and checked["task_nonregression"]):
                write_json(args.out / "training_summary.json", dict(completed_updates=0,
                    training_episodes=0, best_update=None, selected_checkpoint_exists=False,
                    blocked_reason="initial_policy_failed_v15_paired_nonregression"))
                raise RuntimeError("initial policy failed paired V15 protection; PPO was not started")
        for update in range(start + 1, args.updates + 1):
            rollout = args.out / "rollout_policy.pt"
            save_policy(rollout, model, schema=SCHEMA, source_hash=source_hash(),
                        kind="rollout", update=update - 1)
            first = (update - 1) * args.episodes_per_update
            batch = train_manifest["scenarios"][first:first + args.episodes_per_update]
            rows = run_jobs(pool, _policy_jobs(batch, args, rollout, True, update),
                            episode_worker, f"train_{update:04d}")
            samples = [sample for row in rows for sample in row.pop("samples")]
            metrics = ppo_update(model, reference, optimizer, samples, args)
            write_json(args.out / f"update_{update:04d}.json", dict(update=update,
                metrics=metrics, training_summary=summarize(rows), rows=rows))
            if update % args.eval_every == 0 or update == args.updates:
                save_policy(rollout, model, schema=SCHEMA, source_hash=source_hash(),
                            kind="rollout", update=update)
                evaluated = run_jobs(pool, _policy_jobs(validation_manifest["scenarios"], args,
                    rollout, False, update), episode_worker, f"validation_{update:04d}")
                checked = report(evaluated, baseline)
                score = _selection_score(checked)
                selected = (checked["eligible"] and checked["task_nonregression"] and
                            (best_score is None or score > best_score))
                if selected:
                    best_update, best_score = update, score
                    save(update, args.out / "checkpoints" / f"selected_{update:04d}.pt")
                    _publish_best(args.out, update)
                write_json(args.out / f"validation_{update:04d}.json", dict(
                    update=update, selected=selected, report=checked, rows=evaluated))
                emit(dict(update=update, validation=checked["summary"],
                    eligible=checked["eligible"], task_nonregression=checked["task_nonregression"],
                    best_update=best_update))
            save(update, args.out / "ppo_latest.pt")
            emit(dict(update=update, metrics=metrics, best_update=best_update))
    write_json(args.out / "training_summary.json", dict(completed_updates=args.updates,
        training_episodes=args.updates * args.episodes_per_update,
        best_update=best_update, selected_checkpoint_exists=(args.out / "ppo_best.pt").is_file()))


def evaluate(args):
    load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    _, payload = load_policy(args.checkpoint, SCHEMA)
    contract = payload.get("contract", {})
    if (payload.get("kind") != "anchored_dynamic_ppo" or
            payload.get("source_hash") != source_hash() or
            contract.get("anchor_sha256") != file_hash(args.init) or
            contract.get("calibration_sha256") != file_hash(args.calibration)):
        raise ValueError("evaluation requires a selected checkpoint from matching source and assets")
    _, anchor_payload = load_anchor(args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed,
                         ImprovedRecipe(**anchor_payload["recipe"]))
    forbidden = set(contract["excluded_seeds"])
    forbidden.update(row["seed"] for row in contract["train_manifest"]["scenarios"])
    forbidden.update(row["seed"] for row in contract["validation_manifest"]["scenarios"])
    _ensure_disjoint(dict(previous=forbidden,
        evaluation=[row["seed"] for row in manifest["scenarios"]]))
    identity = dict(schema=SCHEMA, source_hash=source_hash(),
        checkpoint_sha256=file_hash(args.checkpoint), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), manifest=manifest,
        selected_update=payload["update"])
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "evaluation_contract.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("evaluation contract changed; use a new output directory")
    write_json(path, identity)
    # Reuse the exact dynamic bounds from training, independent of CLI defaults.
    dynamic = contract["dynamic"]
    for key, value in (("decision_interval", dynamic["decision_interval_s"]),
                       ("force_step", dynamic["force_step"]),
                       ("radius_step", dynamic["radius_step"]),
                       ("release_tolerance", dynamic["release_tolerance_m"])):
        setattr(args, key, value)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        baseline = run_jobs(pool, [(row, str(args.init.resolve()))
            for row in manifest["scenarios"]], baseline_worker, "test_v15")
        candidate = run_jobs(pool, _policy_jobs(manifest["scenarios"], args,
            args.checkpoint, False), episode_worker, "test_candidate")
    checked = report(candidate, baseline)
    output = dict(selected_update=payload["update"], episodes=args.episodes,
        v15=summarize(baseline), candidate=summarize(candidate), comparison=checked,
        absolute_generalization_pass=checked["absolute_generalization_pass"])
    write_json(args.out / "v15_episodes.json", baseline)
    write_json(args.out / "candidate_episodes.json", candidate)
    write_json(args.out / "comparison.json", output)
    emit(dict(v15=output["v15"], candidate=output["candidate"],
        eligible=checked["eligible"], task_nonregression=checked["task_nonregression"],
        absolute_generalization_pass=checked["absolute_generalization_pass"]))


def add_common(parser):
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)


def add_ppo_options(parser):
    parser.add_argument("--decision-interval", type=float, default=.5)
    parser.add_argument("--force-step", type=float, default=.1)
    parser.add_argument("--radius-step", type=float, default=.045)
    parser.add_argument("--release-tolerance", type=float, default=.08)
    parser.add_argument("--gamma", type=float, default=.98)
    parser.add_argument("--gae-lambda", type=float, default=.95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("pretrain")
    add_common(command)
    command.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    command.add_argument("--out", type=Path, default=DEFAULT_PRETRAIN)
    command.add_argument("--epochs", type=int, default=400)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--hidden", type=int, default=48)
    command.add_argument("--lr", type=float, default=1e-3)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--random-seed", type=int, default=20261030)

    command = sub.add_parser("train")
    add_common(command)
    add_ppo_options(command)
    command.add_argument("--pretrain", type=Path, default=DEFAULT_PRETRAIN)
    command.add_argument("--out", type=Path, default=DEFAULT_TRAIN)
    command.add_argument("--resume", type=Path)
    command.add_argument("--updates", type=int, default=100)
    command.add_argument("--episodes-per-update", type=int, default=32)
    command.add_argument("--eval-every", type=int, default=5)
    command.add_argument("--eval-episodes", type=int, default=40)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=31000001)
    command.add_argument("--eval-seed", type=int, default=31100001)
    command.add_argument("--design-seed", type=int, default=20261104)
    command.add_argument("--random-seed", type=int, default=20261104)
    command.add_argument("--lr", type=float, default=3e-5)
    command.add_argument("--critic-lr", type=float, default=3e-4)
    command.add_argument("--clip-ratio", type=float, default=.1)
    command.add_argument("--entropy", type=float, default=.002)
    command.add_argument("--anchor-kl", type=float, default=.03)
    command.add_argument("--target-kl", type=float, default=.02)
    command.add_argument("--update-epochs", type=int, default=4)
    command.add_argument("--minibatch-size", type=int, default=128)
    command.add_argument("--value-coefficient", type=float, default=.5)
    command.add_argument("--max-grad-norm", type=float, default=.5)
    command.add_argument("--smoke", action="store_true")

    command = sub.add_parser("evaluate")
    add_common(command)
    add_ppo_options(command)
    command.add_argument("--checkpoint", type=Path, default=DEFAULT_TRAIN / "ppo_best.pt")
    command.add_argument("--out", type=Path, default=DEFAULT_TRAIN / "evaluation_80")
    command.add_argument("--episodes", type=int, default=80)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=31400001)
    command.add_argument("--design-seed", type=int, default=20261105)
    command.add_argument("--random-seed", type=int, default=20261105)
    args = parser.parse_args()
    integer_names = ("epochs", "batch_size", "hidden", "updates", "episodes_per_update",
                     "eval_every", "eval_episodes", "workers", "update_epochs",
                     "minibatch_size", "episodes")
    for name in integer_names:
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.command == "train" and args.eval_episodes < (5 if args.smoke else 20):
        parser.error("formal validation needs >=20 episodes; smoke needs >=5")
    if args.command == "train" and args.updates * args.episodes_per_update < 5:
        parser.error("the training scenario manifest needs at least 5 episodes")
    if args.command == "evaluate" and args.episodes < 80:
        parser.error("independent evaluation needs at least 80 episodes")
    handler = dict(pretrain=pretrain, train=train, evaluate=evaluate)[args.command]
    handler(args)


if __name__ == "__main__":
    main()
