"""V25 five-stage digital-twin preview, forced-candidate PPO and safe gate.

The stages are deliberately contractual.  A later stage refuses to run unless
the earlier artifact exists, matches the current source/V15/calibration hashes,
and passed its declared acceptance checks.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .anchored_dynamic_env import TRAJECTORY_ACTIONS, initial_action
from .anchored_dynamic_rl import load_pretraining_data, _v15_rollout
from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_env import CONTEXT_NAMES, episode_utility, normalized_action
from .contextual_kernel import load_calibration
from .data import write_json
from .improved_rl import (_init_worker, file_hash, load_anchor, parameter_bins,
    read_json, summarize)
from .improved_teacher import ImprovedRecipe
from .twin_gate import ConservativeGate, GateMember, load_gate, save_gate
from .twin_residual_env import (HOLD_ACTION, JOINT_ACTIONS, JOINT_ACTION_NAMES,
    OBSERVATION_NAMES, SCHEMA, TwinResidualConfig, TwinResidualEnv)
from .twin_residual_model import (CONTEXT_SIZE, OBS_SIZE, TwinResidualPolicy,
    load_policy, save_policy)
from .twin_snapshot import TwinSnapshot, numeric_fingerprint


DEFAULT_ROOT = Path("teacher_runs/v25_twin_residual")
DEFAULT_CALIBRATION = Path(
    "teacher_runs/v23_contextual/warm_kernel_v1/release_calibration_v23_kernel.json")
DEFAULT_INVENTORY = Path(
    "teacher_runs/v23_contextual/combined_dataset_inventory_v2.json")


def source_hash():
    digest = hashlib.sha256(SCHEMA.encode())
    for name in ("twin_snapshot.py", "twin_residual_env.py",
                 "twin_residual_model.py", "twin_gate.py", "twin_residual_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _identity(args, kind):
    return dict(schema=SCHEMA, source_hash=source_hash(), kind=kind,
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration))


def _require_identity(payload, args, kind):
    expected = _identity(args, kind)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{kind} artifact mismatch at {key}; rerun the earlier V25 stage")


def _manifest_seeds(manifest):
    return {int(row["seed"]) for row in manifest.get("scenarios", [])}


def _assert_disjoint(**groups):
    seen = set()
    for name, seeds in groups.items():
        seeds = set(seeds)
        overlap = seen & seeds
        if overlap:
            raise ValueError(f"V25 scenario leak in {name}: {sorted(overlap)[:5]}")
        seen.update(seeds)


def _make_env(scenario, anchor, anchor_payload, calibration, selector, dynamic):
    return TwinResidualEnv(scenario, ImprovedRecipe(**anchor_payload["recipe"]),
        anchor, calibration, selector, dynamic)


def _advance_to_decision(env, limit=5000):
    result = None
    for _ in range(limit):
        if env.dynamic_due() or env.done:
            return result
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    raise RuntimeError("V25 did not reach a dynamic decision within the step limit")


def verify(args):
    anchor, anchor_payload = load_anchor(args.init)
    calibration, _ = load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    # The shared manifest builder intentionally requires a non-empty validation
    # partition, so construct its minimum five and consume only the first here.
    scenario = scenarios(5, args.seed, args.design_seed, recipe)["scenarios"][0]
    holder = {}

    def selector(context):
        return normalized_action(1.2, 1.09)

    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, args.preview_horizon)
    env = _make_env(scenario, anchor, anchor_payload, calibration, selector, dynamic)
    holder["env"] = env
    env.reset(scenario["seed"])
    _advance_to_decision(env)
    if env.done or not env.dynamic_due():
        raise RuntimeError("verification scenario never produced a post-catch decision")
    observation = env.policy_observation()
    mask = env.valid_action_mask()
    snapshot = TwinSnapshot(env)

    def trace():
        env.apply_joint_action(HOLD_ACTION)
        values = []
        for _ in range(args.verify_steps):
            env.step(np.zeros(7, dtype=np.float32))
            values.append(numeric_fingerprint(env))
        return np.vstack(values)

    first = trace()
    snapshot.restore()
    second = trace()
    exact = bool(np.array_equal(first, second))
    report = dict(**_identity(args, "snapshot_verification"), passed=exact,
        scenario=scenario, observation_size=len(observation), action_count=len(mask),
        state_values=int(first.size), max_absolute_error=float(np.max(np.abs(first-second))),
        exact_bitwise_replay=exact, snapshot_objects=len(snapshot.objects),
        dynamic=asdict(dynamic))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/"verification.json", report)
    emit(report)
    if not exact:
        raise RuntimeError("MuJoCo/CAN snapshot replay was not bitwise deterministic")


def _branch_score(env, snapshot, action, horizon_s):
    snapshot.restore()
    start = float(env.obs["t_win"])
    env.apply_joint_action(action)
    result, branch_minimum = None, float("inf")
    for _ in range(10000):
        if env.done:
            break
        if (env.releaser.original.t_release is None and
                float(env.obs["t_win"])-start >= horizon_s-1e-9):
            break
        if env.dynamic_due():
            env.apply_joint_action(HOLD_ACTION)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
        preview = env.releaser.current_preview
        if preview is not None:
            branch_minimum = min(branch_minimum, float(preview["calibrated_error_m"]))
    env.releaser.refresh(env.obs)
    preview = env.releaser.current_preview
    if preview is not None:
        branch_minimum = min(branch_minimum, float(preview["calibrated_error_m"]))
    minimum = .5 if not np.isfinite(branch_minimum) else branch_minimum
    released = bool((result or {}).get("released", env.releaser.original.t_release is not None))
    hit15 = bool((result or {}).get("hit15", False))
    landing = (result or {}).get("landing_error_m")
    joint = float((result or {}).get("max_joint_deg",
        np.degrees(np.abs(np.asarray(env.obs["q_meas"])).max())))
    unsafe = bool((result or {}).get("grip_broken", False) or joint > 36.)
    error = minimum if landing is None else float(landing)
    elapsed = float(env.obs["t_win"])-start
    move = sum(abs(value) for value in JOINT_ACTIONS[action])
    # Release and hit are lexicographically dominant.  For an unreleased branch
    # use only predictions observed *after this snapshot*.  Reusing the episode
    # audit's historical minimum makes late branches identical and lets the
    # movement penalty incorrectly teach the actor to hold forever.
    score = (80.*released + 40.*hit15 - 80.*min(error, 1.) - .15*elapsed
             - 50.*unsafe - .05*move)
    return dict(score=float(score), released=released, hit15=hit15,
        error_m=float(error), minimum_calibrated_error_m=float(minimum),
        elapsed_s=elapsed, max_joint_deg=joint, unsafe=unsafe)


def preview_episode(job):
    (scenario, episode_index, anchor_path, calibration_path, dynamic_data,
     output_path) = job
    torch.set_num_threads(1)
    anchor, anchor_payload = load_anchor(Path(anchor_path))
    calibration, _ = load_calibration(Path(calibration_path), Path(anchor_path))
    dynamic = TwinResidualConfig(**dynamic_data)
    nominal = episode_index % len(TRAJECTORY_ACTIONS)
    holder = {}

    def selector(context):
        return initial_action(nominal+1)

    env = _make_env(scenario, anchor, anchor_payload, calibration, selector, dynamic)
    holder["env"] = env
    env.reset(scenario["seed"])
    decisions, result, previous = [], None, None
    started = time.perf_counter()
    while not env.done:
        if env.dynamic_due():
            observation = env.policy_observation()
            mask = env.valid_action_mask()
            snapshot = TwinSnapshot(env)
            branches = [None]*len(JOINT_ACTIONS)
            for action in np.flatnonzero(mask):
                branches[int(action)] = _branch_score(
                    env, snapshot, int(action), dynamic.preview_horizon_s)
            snapshot.restore()
            valid = np.flatnonzero(mask)
            chosen = int(max(valid, key=lambda index: branches[int(index)]["score"]))
            row = dict(episode_id=int(scenario["seed"]), seed=int(scenario["seed"]),
                split=scenario.get("split", "train"), time_s=float(env.obs["t_win"]),
                observation=observation.tolist(), valid_mask=mask.astype(int).tolist(),
                branch_scores=[None if value is None else value for value in branches],
                chosen_action=chosen, chosen_name=JOINT_ACTION_NAMES[chosen],
                parameters_before=dict(force=env.continuous.envelope_force_scale,
                    radius=env.continuous.envelope_radius_scale), next_observation=None)
            if previous is not None:
                previous["next_observation"] = observation.tolist()
            decisions.append(row)
            previous = row
            env.apply_joint_action(chosen)
            row["parameters_after"] = dict(force=env.continuous.envelope_force_scale,
                radius=env.continuous.envelope_radius_scale)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    result = result or env.summary()
    row = dict(seed=scenario["seed"], scenario=scenario, nominal_action=nominal,
        decisions=decisions, result=result, utility=episode_utility(result),
        wall_s=time.perf_counter()-started)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, row)
    return row


def _run_jobs(pool, jobs, worker, label):
    futures = [pool.submit(worker, job) for job in jobs]
    rows = []
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        emit(dict(stage=label, completed=len(rows), total=len(futures), seed=row["seed"],
            captured=row["result"].get("captured"), released=row["result"].get("released"),
            hit15=row["result"].get("hit15")))
    return sorted(rows, key=lambda value: value["seed"])


def preview(args):
    verification = read_json(args.verification)
    _require_identity(verification, args, "snapshot_verification")
    if not verification.get("passed"):
        raise ValueError("snapshot verification did not pass")
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, args.preview_horizon)
    args.out.mkdir(parents=True, exist_ok=True)
    contract = dict(**_identity(args, "digital_twin_preview"),
        verification_sha256=file_hash(args.verification), manifest=manifest,
        dynamic=asdict(dynamic), actions=list(JOINT_ACTIONS),
        teacher="full_MuJoCo_CAN_snapshot_receding_horizon")
    contract_path = args.out/"preview_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("preview contract changed; use a new output directory")
    write_json(contract_path, contract)
    jobs = [(scenario, index, str(args.init.resolve()), str(args.calibration.resolve()),
        asdict(dynamic), str((args.out/"episodes"/f"{scenario['seed']}.json").resolve()))
        for index, scenario in enumerate(manifest["scenarios"])]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        rows = _run_jobs(pool, jobs, preview_episode, "digital_twin_preview")
    decisions = [decision for row in rows for decision in row["decisions"]]
    if not decisions:
        raise RuntimeError("preview produced zero post-catch decisions")
    dataset = dict(**contract, episodes_completed=len(rows), decisions=len(decisions),
        result_summary=summarize(rows), rows=rows)
    write_json(args.out/"preview_dataset.json", dataset)
    emit(dict(episodes=len(rows), decisions=len(decisions), summary=dataset["result_summary"]))


def _soft_targets(values, temperature):
    values = np.asarray(values, dtype=np.float32)
    shifted = (values-values.max())/temperature
    probability = np.exp(shifted)
    return probability/probability.sum()


def _initial_observation(context):
    action = normalized_action(1.2, 1.09)
    extra = np.r_[action, 0., 1., 1., 1., 0., 0., 0., 0., 1.,
        np.eye(len(JOINT_ACTIONS), dtype=np.float32)[HOLD_ACTION]]
    result = np.r_[context, extra].astype(np.float32)
    if result.shape != (OBS_SIZE,):
        raise RuntimeError("V25 initial observation construction is inconsistent")
    return result


def _preview_arrays(dataset):
    observations, masks, targets, splits = [], [], [], []
    for episode in dataset["rows"]:
        for row in episode["decisions"]:
            scores = np.asarray([(-1e9 if item is None else item["score"])
                                 for item in row["branch_scores"]], dtype=np.float32)
            mask = np.asarray(row["valid_mask"], dtype=bool)
            target = np.zeros(len(JOINT_ACTIONS), dtype=np.float32)
            target[mask] = _soft_targets(scores[mask], temperature=3.)
            observations.append(np.asarray(row["observation"], dtype=np.float32))
            masks.append(mask)
            targets.append(target)
            splits.append(row.get("split", episode["scenario"].get("split", "train")))
    return (np.vstack(observations), np.vstack(masks), np.vstack(targets),
            np.asarray(splits))


def _pretrain_metrics(model, nominal_observations, nominal_targets,
                      motion_observations, motion_masks, motion_targets):
    model.eval()
    with torch.no_grad():
        nominal = model.nominal_distribution(torch.tensor(nominal_observations))
        n_loss = -(torch.tensor(nominal_targets)*nominal.logits.log_softmax(-1)).sum(-1).mean()
        motion = model.motion_distribution(torch.tensor(motion_observations),
            torch.tensor(motion_masks))
        m_loss = -(torch.tensor(motion_targets)*motion.logits.log_softmax(-1)).sum(-1).mean()
        n_acc = (nominal.probs.argmax(-1).numpy() == nominal_targets.argmax(-1)).mean()
        m_acc = (motion.probs.argmax(-1).numpy() == motion_targets.argmax(-1)).mean()
    model.train()
    return dict(nominal_cross_entropy=float(n_loss), motion_cross_entropy=float(m_loss),
        nominal_top1=float(n_acc), motion_top1=float(m_acc))


def pretrain(args):
    preview_data = read_json(args.preview)
    _require_identity(preview_data, args, "digital_twin_preview")
    if preview_data.get("decisions", 0) < args.minimum_preview_decisions:
        raise ValueError("digital-twin preview dataset is too small for V25 pretraining")
    train_rows, validation_rows, old_identity = load_pretraining_data(
        args.inventory, args.init, args.calibration)
    motion_obs, motion_masks, motion_targets, motion_splits = _preview_arrays(preview_data)
    nominal_train = train_rows
    nominal_validation = validation_rows
    nominal_obs_train = np.vstack([_initial_observation(row["context"]) for row in nominal_train])
    nominal_obs_validation = np.vstack([_initial_observation(row["context"]) for row in nominal_validation])
    nominal_targets_train = np.vstack([_soft_targets(row["utilities"][1:], 8.)
                                       for row in nominal_train])
    nominal_targets_validation = np.vstack([_soft_targets(row["utilities"][1:], 8.)
                                            for row in nominal_validation])
    motion_train_mask = motion_splits != "validation"
    motion_validation_mask = ~motion_train_mask
    if not motion_train_mask.any() or not motion_validation_mask.any():
        # Whole preview episodes remain intact; deterministic seed grouping is
        # only a fallback for manifests whose legacy split field is all train.
        seeds = np.asarray([decision["seed"] for episode in preview_data["rows"]
            for decision in episode["decisions"]])
        unique = sorted(set(int(seed) for seed in seeds))
        if len(unique) < 2:
            raise ValueError("preview data needs decisions from at least two scenarios")
        validation_seeds = set(unique[4::5] or unique[-1:])
        motion_validation_mask = np.asarray([int(seed) in validation_seeds for seed in seeds])
        motion_train_mask = ~motion_validation_mask
    if not motion_train_mask.any() or not motion_validation_mask.any():
        raise ValueError("preview data needs scenario-disjoint train and validation decisions")
    normalization_rows = np.vstack([nominal_obs_train, motion_obs[motion_train_mask]])
    mean = normalization_rows.mean(0)
    scale = np.maximum(normalization_rows.std(0), .05)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    model = TwinResidualPolicy(mean, scale, args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_obs = torch.tensor(nominal_obs_train)
    n_targets = torch.tensor(nominal_targets_train)
    m_obs = torch.tensor(motion_obs[motion_train_mask])
    m_masks = torch.tensor(motion_masks[motion_train_mask])
    m_targets = torch.tensor(motion_targets[motion_train_mask])
    best, best_loss, history = None, float("inf"), []
    for epoch in range(1, args.epochs+1):
        nominal_indices = torch.randint(len(n_obs), (args.batch_size,))
        motion_indices = torch.randint(len(m_obs), (args.batch_size,))
        nominal = model.nominal_distribution(n_obs[nominal_indices])
        motion = model.motion_distribution(m_obs[motion_indices], m_masks[motion_indices])
        nominal_loss = -(n_targets[nominal_indices]*nominal.logits.log_softmax(-1)).sum(-1).mean()
        motion_loss = -(m_targets[motion_indices]*motion.logits.log_softmax(-1)).sum(-1).mean()
        loss = nominal_loss+motion_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            metrics = _pretrain_metrics(model, nominal_obs_validation,
                nominal_targets_validation, motion_obs[motion_validation_mask],
                motion_masks[motion_validation_mask], motion_targets[motion_validation_mask])
            history.append(dict(epoch=epoch, **metrics))
            key = metrics["nominal_cross_entropy"]+metrics["motion_cross_entropy"]
            if key < best_loss:
                best_loss, best = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    metrics = history[-1] if best is None else _pretrain_metrics(model,
        nominal_obs_validation, nominal_targets_validation,
        motion_obs[motion_validation_mask], motion_masks[motion_validation_mask],
        motion_targets[motion_validation_mask])
    contract = dict(**_identity(args, "twin_preview_pretrain"),
        preview_sha256=file_hash(args.preview), inventory_sha256=file_hash(args.inventory),
        old_data=old_identity, nominal_contexts=len(train_rows)+len(validation_rows),
        motion_decisions=len(motion_obs), hidden=args.hidden, epochs=args.epochs,
        action_semantics="nine_joint_force_radius_actions_no_release_head")
    save_policy(args.out, model, **contract, metrics=metrics)
    write_json(args.out.parent/"pretrain_contract.json", contract)
    write_json(args.out.parent/"pretrain_history.json", history)
    emit(dict(checkpoint=str(args.out), metrics=metrics,
        nominal_contexts=contract["nominal_contexts"], motion_decisions=len(motion_obs)))


def _sample(observation, phase, action, mask, logp, value, time_s, parameters):
    return dict(observation=np.asarray(observation, dtype=np.float32).tolist(),
        next_observation=None, phase=int(phase), action=int(action),
        valid_mask=np.asarray(mask, dtype=bool).astype(int).tolist(), old_logp=float(logp),
        value=float(value), time_s=float(time_s), reward=0.,
        parameters_before=dict(parameters), parameters_after=None, terminal=False)


def _potential(observation):
    value = np.asarray(observation, dtype=np.float32)
    current = float(value[OBSERVATION_NAMES.index("current_calibrated_error_fraction")])
    ready = float(value[OBSERVATION_NAMES.index("safe_candidate_ready")])
    ood = float(value[OBSERVATION_NAMES.index("ood_margin_fraction")])
    return float(np.clip(1.-current, -2., 1.)+.15*ready+.05*ood)


def _finish_paired(samples, result, baseline, gamma, gae_lambda):
    if not samples:
        return samples
    delta = float(np.clip((episode_utility(result)-episode_utility(baseline))/40., -2., 2.))
    nominal = [row for row in samples if row["phase"] == 0]
    dynamic = [row for row in samples if row["phase"] == 1]
    if len(nominal) != 1:
        raise RuntimeError("forced-candidate rollout must contain one nominal decision")
    nominal[0].update(reward=delta, terminal_reward=delta,
        advantage=delta-nominal[0]["value"], return_value=delta, sample_weight=1.)
    for current, following in zip(dynamic[:-1], dynamic[1:]):
        shaping = float(np.clip(_potential(following["observation"])-
                                _potential(current["observation"]), -.25, .25))
        current["reward"] = shaping
        current["progress_reward"] = shaping
        current["next_observation"] = following["observation"]
    for index, row in enumerate(dynamic):
        direction = JOINT_ACTIONS[row["action"]]
        movement_cost = .005*sum(abs(value) for value in direction)
        reversal_cost = 0.
        if index:
            previous = JOINT_ACTIONS[dynamic[index-1]["action"]]
            reversal_cost = .01*sum(a*b < 0 for a, b in zip(previous, direction))
        row["action_change_cost"] = float(movement_cost+reversal_cost)
        row["reward"] -= row["action_change_cost"]
    if dynamic:
        dynamic[-1]["reward"] += delta
        dynamic[-1]["terminal_reward"] = delta
        dynamic[-1]["terminal"] = True
        advantage, next_value = 0., 0.
        for row in reversed(dynamic):
            td = row["reward"]+gamma*next_value-row["value"]
            advantage = td+gamma*gae_lambda*advantage
            row["advantage"] = float(advantage)
            row["return_value"] = float(advantage+row["value"])
            row["sample_weight"] = 1./len(dynamic)
            next_value = row["value"]
        nominal[0]["next_observation"] = dynamic[0]["observation"]
    else:
        nominal[0]["terminal"] = True
    nominal[0]["paired_utility_delta"] = delta
    return samples


def candidate_worker(job):
    (scenario, anchor_path, calibration_path, checkpoint_path, dynamic_data,
     stochastic, gamma, gae_lambda, policy_seed) = job
    torch.set_num_threads(1)
    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed % 2**32)
    anchor, anchor_payload = load_anchor(Path(anchor_path))
    calibration, _ = load_calibration(Path(calibration_path), Path(anchor_path))
    policy, payload = load_policy(Path(checkpoint_path), SCHEMA)
    if payload.get("source_hash") != source_hash():
        raise ValueError("candidate checkpoint was produced by different V25 source")
    policy.eval()
    samples, holder = [], {}

    def selector(context):
        observation = holder["env"].initial_observation(context)
        action, logp, value = policy.act_nominal(observation, deterministic=not stochastic)
        if stochastic:
            sample = _sample(observation, 0, action,
                np.ones(len(JOINT_ACTIONS), dtype=bool), logp, value,
                holder["env"].obs["t_win"], dict(force=1.2, radius=1.09))
            force, radius = TRAJECTORY_ACTIONS[action]
            sample["parameters_after"] = dict(force=force, radius=radius)
            samples.append(sample)
        holder["nominal"] = action
        return initial_action(action+1)

    env = _make_env(scenario, anchor, anchor_payload, calibration, selector,
                    TwinResidualConfig(**dynamic_data))
    holder["env"] = env
    env.reset(scenario["seed"])
    result = None
    started = time.perf_counter()
    while not env.done:
        if env.dynamic_due():
            observation = env.policy_observation()
            mask = env.valid_action_mask()
            action, logp, value = policy.act_motion(
                observation, mask, deterministic=not stochastic)
            before = dict(force=env.continuous.envelope_force_scale,
                          radius=env.continuous.envelope_radius_scale)
            sample = _sample(observation, 1, action, mask, logp, value,
                             env.obs["t_win"], before)
            env.apply_joint_action(action)
            sample["parameters_after"] = dict(force=env.continuous.envelope_force_scale,
                radius=env.continuous.envelope_radius_scale)
            if stochastic:
                samples.append(sample)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    result = result or env.summary()
    baseline = _v15_rollout(scenario, anchor, anchor_payload)
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or
        env.capture_physics_time is None else env.controller.t_hand-env.capture_physics_time)
    result.setdefault("catch_to_release_s", None if result.get("release_time") is None or
        result.get("capture_time") is None else result["release_time"]-result["capture_time"])
    if stochastic:
        _finish_paired(samples, result, baseline, gamma, gae_lambda)
    for row in samples:
        row["episode_id"] = int(scenario["seed"])
    return dict(seed=scenario["seed"], scenario=scenario, result=result, baseline=baseline,
        context=(result.get("trajectory_decision") or {}).get("context"),
        nominal_action=holder.get("nominal"), utility=episode_utility(result),
        baseline_utility=episode_utility(baseline),
        paired_utility_delta=episode_utility(result)-episode_utility(baseline),
        samples=samples, wall_s=time.perf_counter()-started)


def _phase_standardize(values, phases):
    result = torch.empty_like(values)
    for phase in phases.unique():
        mask = phases == phase
        selected = values[mask]
        result[mask] = (selected-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8)
    return result


def _weighted_phase_mean(values, phases, weights):
    groups = []
    for phase in phases.unique():
        mask = phases == phase
        groups.append((values[mask]*weights[mask]).sum()/weights[mask].sum().clamp_min(1e-8))
    return torch.stack(groups).mean()


def ppo_update(model, reference, optimizer, samples, args):
    observations = torch.tensor(np.asarray([row["observation"] for row in samples]),
                                dtype=torch.float32)
    phases = torch.tensor([row["phase"] for row in samples], dtype=torch.long)
    actions = torch.tensor([row["action"] for row in samples], dtype=torch.long)
    masks = torch.tensor(np.asarray([row["valid_mask"] for row in samples]), dtype=torch.bool)
    old_logp = torch.tensor([row["old_logp"] for row in samples], dtype=torch.float32)
    returns = torch.tensor([row["return_value"] for row in samples], dtype=torch.float32)
    advantages = _phase_standardize(torch.tensor(
        [row["advantage"] for row in samples], dtype=torch.float32), phases)
    weights = torch.tensor([row["sample_weight"] for row in samples], dtype=torch.float32)
    records, stopped = [], False
    for epoch in range(args.update_epochs):
        for indices in torch.randperm(len(samples)).split(args.minibatch_size):
            logp, entropy, value = model.evaluate(observations[indices], phases[indices],
                                                   actions[indices], masks[indices])
            log_ratio = logp-old_logp[indices]
            ratio = log_ratio.exp()
            approximate_kl = float(((ratio-1.)-log_ratio).mean().detach())
            if approximate_kl > args.target_kl:
                stopped = True
                break
            surrogate = torch.minimum(ratio*advantages[indices],
                ratio.clamp(1.-args.clip_ratio, 1.+args.clip_ratio)*advantages[indices])
            policy_loss = -_weighted_phase_mean(surrogate, phases[indices], weights[indices])
            value_loss = .5*_weighted_phase_mean((value-returns[indices]).square(),
                                                  phases[indices], weights[indices])
            anchor_kl = _weighted_phase_mean(model.kl_from(reference,
                observations[indices], phases[indices], masks[indices]),
                phases[indices], weights[indices])
            entropy_mean = _weighted_phase_mean(entropy, phases[indices], weights[indices])
            loss = (policy_loss+args.value_coefficient*value_loss+
                    args.anchor_kl*anchor_kl-args.entropy*entropy_mean)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite V25 PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            records.append((float(policy_loss.detach()), float(value_loss.detach()),
                float(anchor_kl.detach()), float(entropy_mean.detach()), approximate_kl))
        if stopped:
            break
    if not records:
        raise RuntimeError("PPO target KL rejected every minibatch")
    average = np.mean(records, axis=0)
    return dict(policy_loss=float(average[0]), value_loss=float(average[1]),
        anchor_kl=float(average[2]), entropy=float(average[3]),
        approximate_kl=float(average[4]), epochs=epoch+1, kl_early_stop=stopped,
        samples=len(samples), nominal_samples=int((phases == 0).sum()),
        dynamic_samples=int((phases == 1).sum()),
        initial_credit="undiscounted_paired_episode_utility_delta")


def _candidate_summary(rows):
    candidate = summarize(rows)
    baseline_rows = [dict(seed=row["seed"], scenario=row["scenario"],
                          result=row["baseline"]) for row in rows]
    baseline = summarize(baseline_rows)
    captured = [row for row in rows if row["result"].get("captured")]
    released = [row for row in captured if row["result"].get("released")]
    safety = all(not row["result"].get("grip_broken") and
                 row["result"].get("max_joint_deg", 1e9) <= 36. for row in rows)
    release_rate = len(released)/max(1, len(captured))
    conditional_hit = sum(row["result"].get("hit15", False) for row in released)/max(1, len(released))
    paired_delta = float(np.mean([row["paired_utility_delta"] for row in rows]))
    checks = dict(nonzero_candidate=len(rows) > 0,
        captured_sample=len(captured) > 0, release_given_capture=release_rate >= .9,
        hit15_given_release=conditional_hit >= .9, capture_nonregression=
        candidate["captured"] >= baseline["captured"], safety=safety)
    return dict(candidate=candidate, baseline=baseline, release_given_capture=release_rate,
        hit15_given_release=conditional_hit, paired_mean_utility_delta=paired_delta,
        checks=checks, qualified=all(checks.values()))


def _save_transition_archive(path, rows):
    samples = [sample for row in rows for sample in row["samples"]]
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,
        observations=np.asarray([row["observation"] for row in samples], np.float32),
        next_observations=np.asarray([np.zeros(OBS_SIZE, dtype=np.float32)
            if row["next_observation"] is None else row["next_observation"] for row in samples],
            np.float32),
        next_observation_valid=np.asarray([row["next_observation"] is not None
                                           for row in samples], bool),
        terminals=np.asarray([row["terminal"] for row in samples], bool),
        phases=np.asarray([row["phase"] for row in samples], np.int8),
        actions=np.asarray([row["action"] for row in samples], np.int8),
        valid_masks=np.asarray([row["valid_mask"] for row in samples], bool),
        rewards=np.asarray([row["reward"] for row in samples], np.float32),
        advantages=np.asarray([row["advantage"] for row in samples], np.float32),
        returns=np.asarray([row["return_value"] for row in samples], np.float32),
        old_logp=np.asarray([row["old_logp"] for row in samples], np.float32),
        old_values=np.asarray([row["value"] for row in samples], np.float32),
        times_s=np.asarray([row["time_s"] for row in samples], np.float32),
        episode_ids=np.asarray([row["episode_id"] for row in samples], np.int64),
        parameters_before=np.asarray([[row["parameters_before"]["force"],
            row["parameters_before"]["radius"]] for row in samples], np.float32),
        parameters_after=np.asarray([[row["parameters_before"]["force"],
            row["parameters_before"]["radius"]] if row["parameters_after"] is None else
            [row["parameters_after"]["force"], row["parameters_after"]["radius"]]
            for row in samples], np.float32))
    temporary.replace(path)


def _candidate_jobs(manifest, args, checkpoint, stochastic, update):
    dynamic = asdict(TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, args.preview_horizon))
    return [(scenario, str(args.init.resolve()), str(args.calibration.resolve()),
        str(Path(checkpoint).resolve()), dynamic, stochastic, args.gamma, args.gae_lambda,
        (args.random_seed+update*100003+scenario["seed"]) % 2**32)
        for scenario in manifest]


def _checkpoint_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def train(args):
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    _, pretrain_payload = load_policy(args.pretrain, SCHEMA)
    _require_identity(pretrain_payload, args, "twin_preview_pretrain")
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    train_manifest = scenarios(args.updates*args.episodes_per_update,
        args.seed, args.design_seed, recipe)
    validation_manifest = scenarios(args.eval_episodes,
        args.eval_seed, args.design_seed+100000, recipe)
    _assert_disjoint(train=_manifest_seeds(train_manifest),
                     validation=_manifest_seeds(validation_manifest))
    dynamic = TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, args.preview_horizon)
    contract = dict(**_identity(args, "forced_candidate_ppo"),
        pretrain_sha256=file_hash(args.pretrain), dynamic=asdict(dynamic),
        train_manifest=train_manifest, validation_manifest=validation_manifest,
        updates=args.updates, episodes_per_update=args.episodes_per_update,
        eval_every=args.eval_every, forced_candidate=True, exact_v15_paired_baseline=True,
        v15_available_to_actor=False, release_head=False,
        ppo=dict(gamma=args.gamma, gae_lambda=args.gae_lambda, actor_lr=args.lr,
            critic_lr=args.critic_lr, clip_ratio=args.clip_ratio, entropy=args.entropy,
            anchor_kl=args.anchor_kl, target_kl=args.target_kl,
            update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
            initial_credit="undiscounted paired utility delta",
            dynamic_discount_interval_s=args.decision_interval,
            episode_normalized_dynamic_weight=True), smoke=args.smoke)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out/"training_contract.json"
    if args.resume:
        model, payload = load_policy(args.out/"ppo_latest.pt", SCHEMA)
        if payload.get("contract") != contract or read_json(contract_path) != contract:
            raise ValueError("resume contract differs from this V25 run")
        start = int(payload["update"])+1
    else:
        if any(args.out.iterdir()):
            raise ValueError("training output is not empty; use --resume or a new directory")
        model, _ = load_policy(args.pretrain, SCHEMA)
        payload, start = None, 1
        write_json(contract_path, contract)
    reference, _ = load_policy(args.pretrain, SCHEMA)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([
        dict(params=[*model.nominal_actor.parameters(), *model.motion_actor.parameters()],
             lr=args.lr), dict(params=model.critic.parameters(), lr=args.critic_lr)])
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    best_score, no_improvement = None, 0
    if payload is not None:
        best_score = None if payload.get("best_score") is None else tuple(payload["best_score"])
        no_improvement = int(payload.get("no_improvement", 0))

    def save(update, path):
        save_policy(path, model, schema=SCHEMA, source_hash=source_hash(),
            kind="forced_candidate_ppo", anchor_sha256=file_hash(args.init),
            calibration_sha256=file_hash(args.calibration), contract=contract, update=update,
            best_score=None if best_score is None else list(best_score),
            no_improvement=no_improvement, optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate())

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for update in range(start, args.updates+1):
            temporary = args.out/"rollout_policy.pt"
            save(update-1, temporary)
            lo = (update-1)*args.episodes_per_update
            selected = train_manifest["scenarios"][lo:lo+args.episodes_per_update]
            rows = _run_jobs(pool, _candidate_jobs(selected, args, temporary, True, update),
                             candidate_worker, f"ppo_{update:04d}")
            samples = [sample for row in rows for sample in row["samples"]]
            nominal_count = len([sample for sample in samples if sample["phase"] == 0])
            if nominal_count == 0:
                raise RuntimeError("candidate PPO batch contains no captured handover")
            _save_transition_archive(args.out/f"update_{update:04d}_transitions.npz", rows)
            stats = ppo_update(model, reference, optimizer, samples, args)
            serial_rows = [{key: value for key, value in row.items() if key != "samples"}
                           for row in rows]
            update_record = dict(update=update, rows=serial_rows, ppo=stats,
                on_policy=True, samples_consumed_once=True)
            validation = None
            checkpoint = args.out/"checkpoints"/f"update_{update:04d}.pt"
            save(update, checkpoint)
            if update % args.eval_every == 0 or update == args.updates:
                validation_rows = _run_jobs(pool, _candidate_jobs(
                    validation_manifest["scenarios"], args, checkpoint, False, update),
                    candidate_worker, f"forced_validation_{update:04d}")
                validation = _candidate_summary(validation_rows)
                validation["update"] = update
                write_json(args.out/f"validation_{update:04d}.json", dict(
                    **validation, rows=[{key: value for key, value in row.items()
                    if key != "samples"} for row in validation_rows]))
                score = (validation["candidate"]["hit15"],
                    validation["candidate"]["released"],
                    validation["paired_mean_utility_delta"])
                if best_score is None or score > best_score:
                    best_score, no_improvement = score, 0
                    save(update, checkpoint)
                    _checkpoint_copy(checkpoint, args.out/"candidate_best.pt")
                else:
                    no_improvement += 1
                if validation["qualified"] and update > 0:
                    _checkpoint_copy(checkpoint, args.out/"candidate_qualified.pt")
                    write_json(args.out/"candidate_qualification.json", validation)
            update_record["validation"] = validation
            write_json(args.out/f"update_{update:04d}.json", update_record)
            save(update, args.out/"ppo_latest.pt")
            temporary.unlink(missing_ok=True)
            emit(dict(update=update, ppo=stats, validation=validation,
                qualified=(args.out/"candidate_qualified.pt").exists()))
            if (args.early_stop_patience > 0 and update >= args.minimum_updates and
                    no_improvement >= args.early_stop_patience):
                write_json(args.out/"early_stop.json", dict(update=update,
                    reason="forced-candidate release/hit validation did not improve",
                    patience=no_improvement))
                break
    if not (args.out/"candidate_qualified.pt").exists() and not args.smoke:
        raise RuntimeError("forced candidate did not reach the declared 90% conditional thresholds")


def _gate_training_rows(training_dir):
    rows = []
    for path in sorted(Path(training_dir).glob("update_[0-9][0-9][0-9][0-9].json")):
        payload = read_json(path)
        for row in payload["rows"]:
            if row.get("context") is None:
                continue
            candidate, baseline = row["result"], row["baseline"]
            safe_success = bool(candidate.get("captured") and candidate.get("released") and
                candidate.get("hit15") and not candidate.get("grip_broken") and
                candidate.get("max_joint_deg", 1e9) <= 36.)
            baseline_success = bool(baseline.get("captured") and baseline.get("released") and
                                    baseline.get("hit15"))
            rows.append(dict(seed=row["seed"], context=row["context"],
                delta=float(row["paired_utility_delta"]), success=float(safe_success),
                baseline_success=float(baseline_success), candidate=candidate, baseline=baseline))
    if not rows:
        raise ValueError("candidate training directory contains no paired gate rows")
    return rows


def _gate_predictions(members, values):
    with torch.no_grad():
        output = np.stack([member(values).numpy() for member in members])
    deltas = output[:, :, 0]
    successes = 1./(1.+np.exp(-output[:, :, 1]))
    return (deltas.mean(0)-1.64*deltas.std(0),
            successes.mean(0)-1.64*successes.std(0))


def train_gate(args):
    _, candidate_payload = load_policy(args.candidate, SCHEMA)
    _require_identity(candidate_payload, args, "forced_candidate_ppo")
    qualification_path = Path(args.training_dir)/"candidate_qualification.json"
    if not qualification_path.is_file() and not args.smoke:
        raise ValueError("gate training requires a formally qualified forced candidate")
    _, anchor_payload = load_anchor(args.init)
    load_calibration(args.calibration, args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    candidate_contract = candidate_payload["contract"]
    _assert_disjoint(candidate_train=_manifest_seeds(candidate_contract["train_manifest"]),
        candidate_validation=_manifest_seeds(candidate_contract["validation_manifest"]),
        gate=_manifest_seeds(manifest))
    dynamic = asdict(TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, args.preview_horizon))
    jobs = [(scenario, str(args.init.resolve()), str(args.calibration.resolve()),
        str(args.candidate.resolve()), dynamic, False, .995, .95,
        (args.random_seed+scenario["seed"]) % 2**32) for scenario in manifest["scenarios"]]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        collected = _run_jobs(pool, jobs, candidate_worker, "gate_paired_collection")
    rows = []
    for row in collected:
        if row.get("context") is None:
            continue
        candidate, baseline = row["result"], row["baseline"]
        safe_success = bool(candidate.get("captured") and candidate.get("released") and
            candidate.get("hit15") and not candidate.get("grip_broken") and
            candidate.get("max_joint_deg", 1e9) <= 36.)
        baseline_success = bool(baseline.get("captured") and baseline.get("released") and
                                baseline.get("hit15"))
        rows.append(dict(seed=row["seed"], context=row["context"],
            delta=float(row["paired_utility_delta"]), success=float(safe_success),
            baseline_success=float(baseline_success), candidate=candidate, baseline=baseline))
    if len(rows) < (5 if args.smoke else 40):
        raise ValueError("dedicated gate collection produced too few captured contexts")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out.parent/"paired_dataset.json", dict(manifest=manifest, rows=rows,
        candidate_sha256=file_hash(args.candidate), dynamic=dynamic))
    contexts = np.asarray([row["context"] for row in rows], dtype=np.float32)
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float32)/40.
    success = np.asarray([row["success"] for row in rows], dtype=np.float32)
    unique_seeds = sorted(set(int(row["seed"]) for row in rows))
    validation_seeds = set(unique_seeds[4::5] or unique_seeds[-1:])
    validation = np.asarray([int(row["seed"]) in validation_seeds for row in rows])
    if not validation.any() or validation.all():
        raise ValueError("gate needs scenario-disjoint train and validation rows")
    mean = contexts[~validation].mean(0)
    scale = np.maximum(contexts[~validation].std(0), .05)
    values = torch.tensor((contexts-mean)/scale)
    target_delta = torch.tensor(deltas)
    target_success = torch.tensor(success)
    members = []
    for member_index in range(args.ensemble):
        torch.manual_seed(args.random_seed+member_index)
        rng = np.random.default_rng(args.random_seed+member_index)
        member = GateMember(args.hidden)
        optimizer = torch.optim.Adam(member.parameters(), lr=args.lr, weight_decay=1e-4)
        train_indices = np.flatnonzero(~validation)
        bootstrap = rng.choice(train_indices, len(train_indices), replace=True)
        bootstrap = torch.tensor(bootstrap, dtype=torch.long)
        for _ in range(args.epochs):
            indices = bootstrap[torch.randint(len(bootstrap), (min(args.batch_size,
                                                                    len(bootstrap)),))]
            output = member(values[indices])
            loss = F.smooth_l1_loss(output[:, 0], target_delta[indices])+F.binary_cross_entropy_with_logits(
                output[:, 1], target_success[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 1.)
            optimizer.step()
        member.eval()
        members.append(member)
    indices = np.flatnonzero(validation)
    delta_lcb, success_lcb = _gate_predictions(members, values[validation])
    validation_rows = [rows[index] for index in indices]
    candidates = sorted(set(float(value) for value in delta_lcb))
    select_all_threshold = float(np.min(delta_lcb)-1.)
    choices = []
    for success_threshold in (.9, .85, .8, .75, .7, .6, .5):
        for threshold in [select_all_threshold, *candidates]:
            selected = (delta_lcb > threshold) & (success_lcb >= success_threshold)
            count = int(selected.sum())
            minimum_selected = 1 if args.smoke else max(2, int(np.ceil(.1*len(indices))))
            if count < minimum_selected:
                continue
            selected_rows = [validation_rows[i] for i in np.flatnonzero(selected)]
            mean_delta = float(np.mean([row["delta"] for row in selected_rows]))
            candidate_success = sum(row["success"] for row in selected_rows)
            baseline_success = sum(row["baseline_success"] for row in selected_rows)
            if mean_delta >= 0. and candidate_success >= baseline_success:
                choices.append((count, success_threshold, threshold, mean_delta,
                                candidate_success, baseline_success))
    if choices:
        count, success_threshold, delta_threshold, mean_delta, cs, bs = max(choices)
    elif args.smoke:
        # Smoke mode proves the hybrid execution path, not gate quality.  Its
        # permissive threshold is stamped into a smoke artifact and formal
        # screen/evaluation can never consume it because their source contract
        # and candidate qualification requirements remain in force.
        count, delta_threshold, success_threshold = len(validation_rows), -1e6, -1e6
        mean_delta = float(np.mean([row["delta"] for row in validation_rows]))
        cs = sum(row["success"] for row in validation_rows)
        bs = sum(row["baseline_success"] for row in validation_rows)
    else:
        count, success_threshold, delta_threshold, mean_delta, cs, bs = (
            0, 1., float(np.max(delta_lcb)+1.), 0., 0., 0.)
    gate = ConservativeGate(members, mean, scale, delta_threshold, success_threshold, 1.64)
    contract = dict(**_identity(args, "conservative_gate"),
        candidate_sha256=file_hash(args.candidate), training_rows=len(rows),
        manifest=manifest, dedicated_final_candidate_rollouts=True, smoke=args.smoke,
        split="whole_scenario_seed_modulo_5", ensemble=args.ensemble,
        selection_rule="delta_LCB_above_calibrated_threshold_and_success_LCB",
        calibration=dict(validation_rows=len(indices), selected=count,
            selected_fraction=count/len(indices), mean_paired_delta=mean_delta,
            candidate_success=cs, baseline_success=bs))
    save_gate(args.out, gate, **contract)
    write_json(args.out.parent/"gate_contract.json", contract)
    emit(dict(gate=str(args.out), calibration=contract["calibration"],
        delta_threshold=delta_threshold, success_threshold=success_threshold))


def hybrid_worker(job):
    (scenario, anchor_path, calibration_path, candidate_path, gate_path,
     dynamic_data) = job
    torch.set_num_threads(1)
    anchor, anchor_payload = load_anchor(Path(anchor_path))
    calibration, _ = load_calibration(Path(calibration_path), Path(anchor_path))
    policy, _ = load_policy(Path(candidate_path), SCHEMA)
    gate, _ = load_gate(Path(gate_path), SCHEMA)
    policy.eval()
    holder = dict(takeover=False, gate=None, nominal=None)

    def selector(context):
        decision = gate.decision(context)
        holder["gate"] = decision
        holder["takeover"] = bool(decision["takeover"])
        if not holder["takeover"]:
            return initial_action(5)
        observation = holder["env"].initial_observation(context)
        action, _, _ = policy.act_nominal(observation, deterministic=True)
        holder["nominal"] = action
        return initial_action(action+1)

    env = _make_env(scenario, anchor, anchor_payload, calibration, selector,
                    TwinResidualConfig(**dynamic_data))
    holder["env"] = env
    env.reset(scenario["seed"])
    result = None
    while not env.done:
        if env.dynamic_due() and holder["takeover"]:
            observation = env.policy_observation()
            action, _, _ = policy.act_motion(observation, env.valid_action_mask(), True)
            env.apply_joint_action(action)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
        if holder["gate"] is not None and not holder["takeover"]:
            result = _v15_rollout(scenario, anchor, anchor_payload)
            break
    result = result or env.summary()
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or
        env.capture_physics_time is None else env.controller.t_hand-env.capture_physics_time)
    result.setdefault("catch_to_release_s", None if result.get("release_time") is None or
        result.get("capture_time") is None else result["release_time"]-result["capture_time"])
    baseline = _v15_rollout(scenario, anchor, anchor_payload)
    result["initial_policy_mode"] = ("no_handover" if holder["gate"] is None else
        "candidate_takeover" if holder["takeover"] else "v15_passthrough")
    return dict(seed=scenario["seed"], scenario=scenario, result=result, baseline=baseline,
        gate=holder["gate"], nominal_action=holder["nominal"],
        paired_utility_delta=episode_utility(result)-episode_utility(baseline))


def _hybrid_jobs(manifest, args):
    dynamic = asdict(TwinResidualConfig(args.decision_interval, args.force_step,
        args.radius_step, args.release_tolerance, args.preview_horizon))
    return [(scenario, str(args.init.resolve()), str(args.calibration.resolve()),
        str(args.candidate.resolve()), str(args.gate.resolve()), dynamic)
        for scenario in manifest]


def _hybrid_report(rows, minimum_coverage=.25, require_time=True,
                   require_generalization=False):
    baseline_rows = [dict(seed=row["seed"], scenario=row["scenario"],
                          result=row["baseline"]) for row in rows]
    candidate = summarize(rows)
    baseline = summarize(baseline_rows)
    decided = [row for row in rows if row["gate"] is not None]
    takeovers = [row for row in decided if row["result"].get("initial_policy_mode") ==
                 "candidate_takeover"]
    coverage = len(takeovers)/max(1, len(decided))
    paired_hits = [row for row in takeovers if row["result"].get("hit15") and
                   row["baseline"].get("hit15") and
                   row["result"].get("catch_to_release_s") is not None and
                   row["baseline"].get("catch_to_release_s") is not None]
    ratios = [1.-row["result"]["catch_to_release_s"]/
              max(row["baseline"]["catch_to_release_s"], 1e-6) for row in paired_hits]
    time_improvement = None if not ratios else float(np.mean(ratios))
    bins, worst_bin = parameter_bins(rows)
    candidate_error = candidate["mean_landing_error_m"]
    baseline_error = baseline["mean_landing_error_m"]
    checks = dict(nonzero_takeover=len(takeovers) > 0,
        takeover_coverage=coverage >= minimum_coverage,
        capture_nonregression=candidate["captured"] >= baseline["captured"],
        release_nonregression=candidate["released"] >= baseline["released"],
        hit15_nonregression=candidate["hit15"] >= baseline["hit15"],
        landing_noninferiority=(candidate_error is not None and baseline_error is not None and
                                candidate_error <= baseline_error+.005),
        safety=all(not row["result"].get("grip_broken") and
                   row["result"].get("max_joint_deg", 1e9) <= 36. for row in rows),
        immediate_handover=(len(takeovers) > 0 and all(
            row["result"].get("catch_to_orbit_s") is not None and
            row["result"]["catch_to_orbit_s"] <= .02 for row in takeovers)),
        no_sustained_pause=(len(takeovers) > 0 and all(
            row["result"].get("longest_postcatch_pause_s") is not None and
            row["result"]["longest_postcatch_pause_s"] <= .1 for row in takeovers)),
        release_time_improved_20pct=(not require_time or
            (time_improvement is not None and time_improvement >= .2)),
        worst_parameter_bin_hit15=(not require_generalization or worst_bin >= .8),
        parameter_bin_sample_size=(not require_generalization or all(
            group["episodes"] >= 10 for groups in bins.values() for group in groups)))
    return dict(candidate=candidate, baseline=baseline, decided=len(decided),
        takeover_episodes=len(takeovers), takeover_coverage=coverage,
        paired_hit_time_samples=len(paired_hits), mean_release_time_improvement=time_improvement,
        parameter_bins=bins, worst_parameter_bin_hit15_rate=worst_bin,
        checks=checks, passed=all(checks.values()))


def screen(args):
    _, candidate_payload = load_policy(args.candidate, SCHEMA)
    _require_identity(candidate_payload, args, "forced_candidate_ppo")
    _, gate_payload = load_gate(args.gate, SCHEMA)
    _require_identity(gate_payload, args, "conservative_gate")
    if gate_payload.get("candidate_sha256") != file_hash(args.candidate):
        raise ValueError("gate was trained for a different candidate checkpoint")
    if not args.smoke and (candidate_payload.get("contract", {}).get("smoke") or
                           gate_payload.get("smoke")):
        raise ValueError("formal screen cannot consume smoke candidate/gate artifacts")
    _, anchor_payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    candidate_contract = candidate_payload["contract"]
    _assert_disjoint(candidate_train=_manifest_seeds(candidate_contract["train_manifest"]),
        candidate_validation=_manifest_seeds(candidate_contract["validation_manifest"]),
        gate=_manifest_seeds(gate_payload["manifest"]), screen=_manifest_seeds(manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    contract = dict(**_identity(args, "hybrid_screen"), manifest=manifest,
        candidate_sha256=file_hash(args.candidate), gate_sha256=file_hash(args.gate),
        minimum_takeover_coverage=.25, required_release_time_improvement=.2,
        smoke=args.smoke)
    write_json(args.out/"screen_contract.json", contract)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        rows = _run_jobs(pool, _hybrid_jobs(manifest["scenarios"], args),
                         hybrid_worker, "hybrid_screen")
    report = _hybrid_report(rows, .25 if not args.smoke else 0., not args.smoke)
    if args.smoke:
        report["formal_pass"] = report["passed"]
        report["passed"] = all(report["checks"][key] for key in (
            "nonzero_takeover", "safety", "immediate_handover", "no_sustained_pause"))
        report["smoke_only"] = True
    output = dict(**contract, report=report, rows=rows)
    write_json(args.out/"comparison.json", output)
    if report["passed"]:
        write_json(args.out/"screen_passed.json", dict(**contract, report=report, passed=True))
    emit(report)
    if not report["passed"]:
        raise RuntimeError("hybrid V25 did not pass the fixed development screen")


def evaluate(args):
    screen_payload = read_json(args.screen_pass)
    _require_identity(screen_payload, args, "hybrid_screen")
    if not screen_payload.get("passed"):
        raise ValueError("independent evaluation requires a passed hybrid screen")
    if screen_payload.get("smoke"):
        raise ValueError("independent evaluation cannot consume a smoke screen")
    if args.episodes < 80:
        raise ValueError("independent V25 evaluation requires at least 80 scenarios")
    _, anchor_payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    _, candidate_payload = load_policy(args.candidate, SCHEMA)
    _, gate_payload = load_gate(args.gate, SCHEMA)
    _require_identity(candidate_payload, args, "forced_candidate_ppo")
    _require_identity(gate_payload, args, "conservative_gate")
    if (screen_payload.get("candidate_sha256") != file_hash(args.candidate) or
            screen_payload.get("gate_sha256") != file_hash(args.gate)):
        raise ValueError("screen pass belongs to different candidate/gate artifacts")
    candidate_contract = candidate_payload["contract"]
    _assert_disjoint(candidate_train=_manifest_seeds(candidate_contract["train_manifest"]),
        candidate_validation=_manifest_seeds(candidate_contract["validation_manifest"]),
        gate=_manifest_seeds(gate_payload["manifest"]),
        screen=_manifest_seeds(screen_payload["manifest"]), evaluation=_manifest_seeds(manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    contract = dict(**_identity(args, "independent_evaluation"), manifest=manifest,
        screen_sha256=file_hash(args.screen_pass), candidate_sha256=file_hash(args.candidate),
        gate_sha256=file_hash(args.gate))
    write_json(args.out/"evaluation_contract.json", contract)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        rows = _run_jobs(pool, _hybrid_jobs(manifest["scenarios"], args),
                         hybrid_worker, "independent_evaluation")
    report = _hybrid_report(rows, .25, True, True)
    output = dict(**contract, report=report, rows=rows)
    write_json(args.out/"comparison.json", output)
    if report["passed"]:
        write_json(args.out/"V25_ACCEPTED.json", dict(**contract, report=report, passed=True))
    emit(report)
    if not report["passed"]:
        raise RuntimeError("V25 failed the independent 80-scenario acceptance test")


def _common(parser):
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)


def _dynamic(parser):
    parser.add_argument("--decision-interval", type=float, default=.5)
    parser.add_argument("--force-step", type=float, default=.05)
    parser.add_argument("--radius-step", type=float, default=.0225)
    parser.add_argument("--release-tolerance", type=float, default=.08)
    parser.add_argument("--preview-horizon", type=float, default=2.)


def _hybrid_paths(parser):
    parser.add_argument("--candidate", type=Path,
        default=DEFAULT_ROOT/"stage3_candidate"/"candidate_qualified.pt")
    parser.add_argument("--gate", type=Path,
        default=DEFAULT_ROOT/"stage4_gate"/"gate.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("verify", help="stage 1a: prove exact twin snapshot replay")
    _common(command); _dynamic(command)
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT/"stage1_verify")
    command.add_argument("--seed", type=int, default=32000001)
    command.add_argument("--design-seed", type=int, default=20261120)
    command.add_argument("--verify-steps", type=int, default=20)

    command = sub.add_parser("preview", help="stage 1b: collect branched MPC supervision")
    _common(command); _dynamic(command)
    command.add_argument("--verification", type=Path,
        default=DEFAULT_ROOT/"stage1_verify"/"verification.json")
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT/"stage1_preview")
    command.add_argument("--episodes", type=int, default=48)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=32100001)
    command.add_argument("--design-seed", type=int, default=20261121)

    command = sub.add_parser("pretrain", help="stage 2: distil nominal and MPC decisions")
    _common(command)
    command.add_argument("--preview", type=Path,
        default=DEFAULT_ROOT/"stage1_preview"/"preview_dataset.json")
    command.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    command.add_argument("--out", type=Path,
        default=DEFAULT_ROOT/"stage2_pretrain"/"policy_init.pt")
    command.add_argument("--minimum-preview-decisions", type=int, default=100)
    command.add_argument("--epochs", type=int, default=300)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--hidden", type=int, default=32)
    command.add_argument("--lr", type=float, default=1e-3)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--random-seed", type=int, default=20261122)

    command = sub.add_parser("train", help="stage 3: forced-candidate paired on-policy PPO")
    _common(command); _dynamic(command)
    command.add_argument("--pretrain", type=Path,
        default=DEFAULT_ROOT/"stage2_pretrain"/"policy_init.pt")
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT/"stage3_candidate")
    command.add_argument("--resume", action="store_true")
    command.add_argument("--updates", type=int, default=60)
    command.add_argument("--episodes-per-update", type=int, default=64)
    command.add_argument("--eval-every", type=int, default=5)
    command.add_argument("--eval-episodes", type=int, default=40)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=32200001)
    command.add_argument("--eval-seed", type=int, default=32300001)
    command.add_argument("--design-seed", type=int, default=20261123)
    command.add_argument("--random-seed", type=int, default=20261123)
    command.add_argument("--gamma", type=float, default=.995)
    command.add_argument("--gae-lambda", type=float, default=.95)
    command.add_argument("--lr", type=float, default=1e-4)
    command.add_argument("--critic-lr", type=float, default=3e-4)
    command.add_argument("--clip-ratio", type=float, default=.1)
    command.add_argument("--entropy", type=float, default=.005)
    command.add_argument("--anchor-kl", type=float, default=.02)
    command.add_argument("--target-kl", type=float, default=.02)
    command.add_argument("--update-epochs", type=int, default=4)
    command.add_argument("--minibatch-size", type=int, default=256)
    command.add_argument("--value-coefficient", type=float, default=.5)
    command.add_argument("--max-grad-norm", type=float, default=.5)
    command.add_argument("--minimum-updates", type=int, default=10)
    command.add_argument("--early-stop-patience", type=int, default=2)
    command.add_argument("--smoke", action="store_true")

    command = sub.add_parser("gate", help="stage 4: fit the independent conservative gate")
    _common(command); _dynamic(command)
    command.add_argument("--candidate", type=Path,
        default=DEFAULT_ROOT/"stage3_candidate"/"candidate_qualified.pt")
    command.add_argument("--training-dir", type=Path,
        default=DEFAULT_ROOT/"stage3_candidate")
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT/"stage4_gate"/"gate.pt")
    command.add_argument("--episodes", type=int, default=200)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=32400001)
    command.add_argument("--design-seed", type=int, default=20261124)
    command.add_argument("--ensemble", type=int, default=5)
    command.add_argument("--hidden", type=int, default=16)
    command.add_argument("--epochs", type=int, default=400)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--lr", type=float, default=1e-3)
    command.add_argument("--random-seed", type=int, default=20261124)
    command.add_argument("--smoke", action="store_true")

    command = sub.add_parser("screen", help="stage 5a: fixed hybrid development screen")
    _common(command); _dynamic(command); _hybrid_paths(command)
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT/"stage5_screen")
    command.add_argument("--episodes", type=int, default=40)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=32500001)
    command.add_argument("--design-seed", type=int, default=20261125)
    command.add_argument("--smoke", action="store_true")

    command = sub.add_parser("evaluate", help="stage 5b: one independent 80-scenario test")
    _common(command); _dynamic(command); _hybrid_paths(command)
    command.add_argument("--screen-pass", type=Path,
        default=DEFAULT_ROOT/"stage5_screen"/"screen_passed.json")
    command.add_argument("--out", type=Path, default=DEFAULT_ROOT/"stage5_evaluation_80")
    command.add_argument("--episodes", type=int, default=80)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=32600001)
    command.add_argument("--design-seed", type=int, default=20261126)

    args = parser.parse_args()
    for name in ("episodes", "workers", "epochs", "batch_size", "hidden", "ensemble",
                 "updates", "episodes_per_update", "eval_every", "eval_episodes",
                 "update_epochs", "minibatch_size", "verify_steps"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.command == "train" and not args.smoke and args.eval_episodes < 40:
        parser.error("formal forced-candidate validation requires at least 40 episodes")
    handler = dict(verify=verify, preview=preview, pretrain=pretrain,
                   train=train, gate=train_gate, screen=screen, evaluate=evaluate)
    handler[args.command](args)


if __name__ == "__main__":
    main()
