"""V23 value-teacher fitting, independent gates, new BC and contextual PPO.

No post-hoc oracle is deployable.  All pilot/screen/validation reports use a
frozen observable-context selector before a single trajectory is executed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_env import SCHEMA, ContextualEnv, episode_utility
from .contextual_model import ContextPolicy, OutcomeTeacher, ValueTeacher, load_model, normalization, save_model
from .contextual_rl import DEFAULT_OUT, previous_seeds, worker as constant_worker, source_hash as sweep_source_hash
from .data import write_json
from .envelope_teacher_env import EnvelopeTeacherConfig
from .envelope_teacher_rl import DEFAULT_CALIBRATION, load_calibration, report, source_hash as v22_source_hash
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe


def runtime_hash():
    digest = hashlib.sha256(v22_source_hash().encode())
    for name in ("contextual_env.py", "contextual_model.py", "contextual_training.py", "contextual_calibration.py", "contextual_kernel.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def assert_model_contract(payload, anchor, calibration):
    expected = dict(runtime_hash=runtime_hash(), anchor_sha256=file_hash(anchor),
                    calibration_sha256=file_hash(calibration))
    if any(payload["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("V23 policy source, anchor, or calibration is stale")


def load_release(path, anchor):
    if read_json(path).get("campaign_schema") in ("can_warmed_release_calibration_v23", "can_warmed_kernel_calibration_v23"):
        from .contextual_calibration import WarmContextualEnv
        from .contextual_kernel import load_calibration as load_warm
        calibration, payload = load_warm(path, anchor)
        return calibration, payload, WarmContextualEnv
    calibration, payload = load_calibration(path, anchor)
    return calibration, payload, ContextualEnv


def lineage_seeds(value):
    """Track data/selection ancestry even when outputs live outside DEFAULT_OUT."""
    used = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "seed" and isinstance(child, int):
                used.add(child)
            elif key.endswith("_seeds") and isinstance(child, list):
                used.update(item for item in child if isinstance(item, int))
            else:
                used.update(lineage_seeds(child))
    elif isinstance(value, list):
        for child in value:
            used.update(lineage_seeds(child))
    return used


def assert_fresh(manifest, *ancestors):
    seeds = [row["seed"] for row in manifest["scenarios"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Scenario manifest contains duplicate seeds")
    forbidden = set().union(*(lineage_seeds(ancestor) for ancestor in ancestors))
    if set(seeds) & forbidden:
        raise ValueError("Independent scenarios overlap model data or earlier selection")


def immutable_contract(path, identity, *, reject_overlap=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != identity:
            raise ValueError(f"Experiment contract changed: use a new directory, {path}")
    else:
        seeds = {row["seed"] for row in identity.get("manifest", {}).get("scenarios", [])}
        if reject_overlap and seeds & previous_seeds(path.parent):
            raise ValueError("Independent V23 scenario seeds overlap an earlier experiment")
        write_json(path, identity)


def policy_worker(job):
    scenario, anchor_path, calibration_path, checkpoint, stochastic, policy_seed = job
    anchor_path, calibration_path = Path(anchor_path), Path(calibration_path)
    if checkpoint is None:
        return constant_worker((scenario, anchor_path, calibration_path, None, None))
    torch.set_num_threads(1)
    torch.manual_seed(policy_seed)
    anchor, anchor_payload = load_anchor(anchor_path)
    calibration, _, env_class = load_release(calibration_path, anchor_path)
    model, payload = load_model(checkpoint)
    assert_model_contract(payload, anchor_path, calibration_path)
    decision = {}
    def selector(context):
        action, info = model.decide(context, stochastic=stochastic)
        decision.update(context=context.tolist(), **info)
        return action
    env = env_class(scenario, ImprovedRecipe(**anchor_payload["recipe"]), anchor, calibration,
        selector, EnvelopeTeacherConfig(**payload["contract"]["config"]))
    env.reset(scenario["seed"])
    while not env.done:
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    return dict(seed=scenario["seed"], scenario=scenario, result=result,
                decision=decision or None, utility=episode_utility(result))


def evaluate(manifest, anchor, calibration, checkpoint, output, workers, *, stochastic=False, salt=0):
    output.mkdir(parents=True, exist_ok=True)
    model_sha = None if checkpoint is None else file_hash(checkpoint)
    identity = dict(schema=SCHEMA, runtime_hash=runtime_hash(), manifest=manifest,
        anchor_sha256=file_hash(anchor), calibration_sha256=file_hash(calibration),
        checkpoint_sha256=model_sha, stochastic=stochastic, salt=salt)
    immutable_contract(output / "rollout_contract.json", identity, reject_overlap=False)
    rows, missing = {}, []
    for scenario in manifest["scenarios"]:
        path = output / f"episode_{scenario['seed']}.json"
        if path.exists():
            rows[scenario["seed"]] = read_json(path)
        else:
            missing.append((scenario, str(anchor.resolve()), str(calibration.resolve()),
                None if checkpoint is None else str(checkpoint.resolve()), stochastic, scenario["seed"] + salt))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(policy_worker, job) for job in missing]
        for future in as_completed(futures):
            row = future.result()
            write_json(output / f"episode_{row['seed']}.json", row)
            rows[row["seed"]] = row
            if len(rows) % 10 == 0 or len(rows) == len(manifest["scenarios"]):
                print(dict(stage=output.name, completed=len(rows), total=len(manifest["scenarios"])), flush=True)
    return [rows[row["seed"]] for row in manifest["scenarios"]]


def check_sweep(directory):
    contract = read_json(directory / "sweep_contract.json")
    if contract["schema"] != SCHEMA:
        raise ValueError("Not a V23 trajectory sweep")
    if contract.get("calibration_mode") == "warmed":
        from .contextual_warm_sweep import source_hash as warmed_source
        if contract["source_hash"] != warmed_source():
            raise ValueError("Warm V23 sweep source changed")
        return contract
    if contract["source_hash"] != sweep_source_hash():
        # The first sweep predates the unused learning module. Reconstruct its
        # exact source identity and require identical rollout/environment code.
        snapshot = directory / "source_snapshot"
        digest = hashlib.sha256(v22_source_hash().encode())
        for name in ("contextual_env.py", "contextual_rl.py"):
            old = snapshot / name
            if not old.exists():
                raise ValueError("Missing original sweep source snapshot")
            original_text = old.read_text(encoding="utf-8")
            # Audited numerical-only migration: float32 policy endpoints map
            # 1.09 + .09 to 1.1800000000000002. Clamp the physical endpoints.
            # Original double-precision sweep actions already mapped in bounds.
            compatible_text = original_text.replace(
                "float(1.2 + .2 * value[0])", "float(np.clip(1.2 + .2 * value[0], 1., 1.4))").replace(
                "float(1.09 + .09 * value[1])", "float(np.clip(1.09 + .09 * value[1], 1., 1.18))")
            if compatible_text != Path(__file__).with_name(name).read_text(encoding="utf-8"):
                raise ValueError("V23 sweep runtime changed; old outcomes cannot silently be reused")
            digest.update(name.encode())
            digest.update(old.read_bytes())
        if digest.hexdigest() != contract["source_hash"]:
            raise ValueError("V23 sweep snapshot fingerprint mismatch")
    return contract


def fit(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.design_seed)
    data_directories = list(args.data)
    contracts = [check_sweep(directory) for directory in data_directories]
    contract = contracts[0]
    compatible = ("schema", "source_hash", "anchor_sha256", "calibration_sha256", "config", "actions")
    if any(any(other.get(key) != contract.get(key) for key in compatible) for other in contracts[1:]):
        raise ValueError("Combined V23 sweeps use different source, action, config, anchor, or calibration")
    if contract["anchor_sha256"] != file_hash(args.init) or contract["calibration_sha256"] != file_hash(args.calibration):
        raise ValueError("V23 training data uses a different BC/calibration")
    all_scenarios = [scenario for item in contracts for scenario in item["manifest"]["scenarios"]]
    all_seeds = [scenario["seed"] for scenario in all_scenarios]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Combined V23 sweeps contain repeated scenario seeds")
    actions = contract["actions"]
    contexts, utilities, seeds, paths = [], [], [], []
    for directory, item in zip(data_directories, contracts):
        for scenario in item["manifest"]["scenarios"]:
            group = []
            for index in range(len(actions)):
                path = directory / "episodes" / f"{scenario['seed']}_a{index:03d}.json"
                if not path.exists():
                    raise ValueError("Complete every paired action sweep before fitting")
                paths.append(path)
                group.append(read_json(path))
            decisions = [row["result"]["trajectory_decision"] for row in group]
            if not any(decisions):
                continue  # no post-capture action existed; these failures remain in reports
            if not all(decisions):
                raise ValueError("Trajectory action unexpectedly changed capture availability")
            context = np.asarray(decisions[0]["context"])
            if any(not np.allclose(context, row["context"], atol=1e-6, rtol=0.) for row in decisions):
                raise ValueError("Paired sweep changed the pre-action measured context")
            contexts.append(context)
            utilities.append([row["utility"] / 40. for row in group])
            seeds.append(scenario["seed"])
    if len(contexts) < 12:
        raise ValueError("At least 12 captured, fully swept scenarios are needed")
    rng = np.random.default_rng(args.design_seed)
    order = rng.permutation(len(contexts))
    cut = max(2, len(order) // 5)
    valid, train = order[:cut], order[cut:]
    x, y = torch.tensor(np.asarray(contexts), dtype=torch.float32), torch.tensor(utilities)
    model = ValueTeacher(actions)
    model.mean.copy_(normalization(np.asarray(contexts)[train])[0])
    model.scale.copy_(normalization(np.asarray(contexts)[train])[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    bootstrap = [torch.tensor(rng.choice(train, size=len(train), replace=True)) for _ in model.models]
    best_loss, best, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train()
        predictions = model(x)
        loss = sum(nn.functional.smooth_l1_loss(predictions[i, idx], y[idx])
                   for i, idx in enumerate(bootstrap)) / len(bootstrap)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                prediction = model(x[valid]).mean(0)
                validation_loss = float(nn.functional.mse_loss(prediction, y[valid]))
                utility = float(y[valid, prediction.argmax(-1)].mean() * 40.)
            history.append(dict(epoch=epoch + 1, loss=float(loss.detach()),
                                validation_loss=validation_loss, heldout_action_utility=utility))
            if validation_loss < best_loss:
                best_loss, best = validation_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    inventory = hashlib.sha256("".join(file_hash(path) for path in paths).encode()).hexdigest()
    model_contract = dict(runtime_hash=runtime_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), config=contract["config"],
        data_contract_sha256=[file_hash(directory / "sweep_contract.json") for directory in data_directories],
        data_directories=[str(directory.resolve()) for directory in data_directories],
        data_inventory_sha256=inventory,
        training_seeds=[seeds[i] for i in train], selection_seeds=[seeds[i] for i in valid],
        development_seeds=all_seeds)
    model_contract["calibration_seeds"] = sorted(set(
        seed for item in contracts for seed in item.get("calibration_seeds", [])))
    args.out.mkdir(parents=True, exist_ok=True)
    immutable_contract(args.out / "fit_contract.json", dict(schema=SCHEMA, model_contract=model_contract,
        epochs=args.epochs, design_seed=args.design_seed), reject_overlap=False)
    save_model(args.out / "teacher.pt", model, kind="value_teacher", contract=model_contract)
    write_json(args.out / "fit_history.json", history)
    print(dict(teacher=str(args.out / "teacher.pt"), train_scenarios=len(train),
               validation_scenarios=len(valid), independent_evaluation_pending=True), flush=True)


def fit_outcome(args):
    """Fit a compact, interpretable outcome selector on complete paired scenes."""
    torch.set_num_threads(1)
    torch.manual_seed(args.design_seed)
    data_directories = list(args.data)
    contracts = [check_sweep(directory) for directory in data_directories]
    contract = contracts[0]
    compatible = ("schema", "source_hash", "anchor_sha256", "calibration_sha256", "config", "actions")
    if any(any(other.get(key) != contract.get(key) for key in compatible) for other in contracts[1:]):
        raise ValueError("Combined V23 sweeps use different source, action, config, anchor, or calibration")
    if contract["anchor_sha256"] != file_hash(args.init) or contract["calibration_sha256"] != file_hash(args.calibration):
        raise ValueError("V23 outcome data uses a different BC/calibration")
    actions = contract["actions"]
    contexts, targets, utilities, seeds, paths, all_seeds = [], [], [], [], [], []
    for directory, item in zip(data_directories, contracts):
        for scenario in item["manifest"]["scenarios"]:
            seed = scenario["seed"]
            if seed in all_seeds:
                raise ValueError("Combined V23 sweeps contain repeated scenario seeds")
            all_seeds.append(seed)
            group = []
            for index in range(len(actions)):
                path = directory / "episodes" / f"{seed}_a{index:03d}.json"
                if not path.exists():
                    raise ValueError("Complete every paired action sweep before fitting")
                paths.append(path)
                group.append(read_json(path))
            decisions = [row["result"]["trajectory_decision"] for row in group]
            if not any(decisions):
                continue
            if not all(decisions):
                raise ValueError("Trajectory action unexpectedly changed capture availability")
            context = np.asarray(decisions[0]["context"])
            if any(not np.allclose(context, row["context"], atol=1e-6, rtol=0.) for row in decisions):
                raise ValueError("Paired sweep changed the pre-action measured context")
            outcome = []
            for row in group:
                result = row["result"]
                release_time = result.get("catch_to_release_s")
                outcome.append([float(result["released"]), float(result["hit15"]),
                    1. if release_time is None else float(np.clip(release_time / 18., 0., 1.)),
                    float(result["max_joint_deg"] / 36.),
                    float(result["pressure_integral_psi_s"] / 300.)])
            contexts.append(context)
            targets.append(outcome)
            utilities.append([row["utility"] for row in group])
            seeds.append(seed)
    if len(contexts) < 40:
        raise ValueError("Compact outcome fitting requires at least 40 captured paired contexts")
    rng = np.random.default_rng(args.design_seed)
    order = rng.permutation(len(contexts))
    cut = max(8, len(order) // 5)
    valid, train = order[:cut], order[cut:]
    x = torch.tensor(np.asarray(contexts), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)
    actual_utility = torch.tensor(np.asarray(utilities), dtype=torch.float32)
    model = OutcomeTeacher(actions)
    model.mean.copy_(normalization(np.asarray(contexts)[train])[0])
    model.scale.copy_(normalization(np.asarray(contexts)[train])[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=.02)

    def balanced_bce(logits, truth):
        rate = truth.mean().clamp(.05, .95)
        weights = torch.where(truth > .5, .5 / rate, .5 / (1. - rate))
        return (nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weights).mean()

    best_key, best, history = (-float("inf"), -float("inf")), None, []
    train_index = torch.tensor(train)
    valid_index = torch.tensor(valid)
    for epoch in range(args.epochs):
        model.train()
        raw = model(x[train_index])
        truth = y[train_index]
        release_loss = balanced_bce(raw[..., 0], truth[..., 0])
        hit_loss = balanced_bce(raw[..., 1], truth[..., 1])
        time_loss = nn.functional.smooth_l1_loss(raw[..., 2].sigmoid(), truth[..., 2])
        joint_loss = nn.functional.smooth_l1_loss(raw[..., 3], truth[..., 3])
        pressure_loss = nn.functional.smooth_l1_loss(raw[..., 4], truth[..., 4])
        loss = release_loss + 1.5 * hit_loss + .3 * time_loss + joint_loss + .2 * pressure_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                predicted = model(x[valid_index])
                selected = model.scores(predicted).argmax(-1)
                rows = torch.arange(len(valid_index))
                utility = float(actual_utility[valid_index][rows, selected].mean())
                release_rate = float(y[valid_index][rows, selected, 0].mean())
                hit_rate = float(y[valid_index][rows, selected, 1].mean())
                validation_loss = float(nn.functional.binary_cross_entropy_with_logits(
                    predicted[..., :2], y[valid_index][..., :2]))
            history.append(dict(epoch=epoch + 1, loss=float(loss.detach()),
                validation_loss=validation_loss, heldout_action_utility=utility,
                heldout_release_rate=release_rate, heldout_hit15_rate=hit_rate))
            key = (utility, -validation_loss)
            if key > best_key:
                best_key, best = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    inventory = hashlib.sha256("".join(file_hash(path) for path in paths).encode()).hexdigest()
    model_contract = dict(runtime_hash=runtime_hash(), anchor_sha256=file_hash(args.init),
        calibration_sha256=file_hash(args.calibration), config=contract["config"],
        data_contract_sha256=[file_hash(directory / "sweep_contract.json") for directory in data_directories],
        data_directories=[str(directory.resolve()) for directory in data_directories],
        data_inventory_sha256=inventory, outcome_names=list(OutcomeTeacher.OUTCOMES),
        training_seeds=[seeds[i] for i in train], selection_seeds=[seeds[i] for i in valid],
        development_seeds=all_seeds,
        calibration_seeds=sorted(set(seed for item in contracts for seed in item.get("calibration_seeds", []))))
    args.out.mkdir(parents=True, exist_ok=True)
    immutable_contract(args.out / "fit_contract.json", dict(schema=SCHEMA, kind="outcome_teacher",
        model_contract=model_contract, epochs=args.epochs, design_seed=args.design_seed), reject_overlap=False)
    save_model(args.out / "teacher.pt", model, kind="outcome_teacher", contract=model_contract)
    write_json(args.out / "fit_history.json", history)
    print(dict(teacher=str(args.out / "teacher.pt"), parameters=sum(p.numel() for p in model.parameters()),
        train_scenarios=len(train), validation_scenarios=len(valid), best_heldout_utility=best_key[0],
        independent_evaluation_pending=True), flush=True)


def eligible(checked, *, absolute=False):
    return bool(checked["absolute_generalization_pass"] if absolute else
                checked["eligible"] and checked["task_nonregression"])


def require_gate(directory, checkpoint, stage):
    path = directory / "gate_result.json"
    if not path.exists():
        raise ValueError(f"Missing {stage} closed-loop gate: {path}")
    gate = read_json(path)
    contract = gate["contract"]
    if (not gate["passed"] or contract["stage"] != stage or
            contract["runtime_hash"] != runtime_hash() or
            contract["checkpoint_sha256"] != file_hash(checkpoint) or
            not eligible(gate["report"], absolute=stage in ("validate", "bc-evaluate", "ppo-test"))):
        raise ValueError(f"{stage} did not qualify this exact V23 checkpoint")
    return gate


def gate(args):
    model, payload = load_model(args.checkpoint)
    assert_model_contract(payload, args.init, args.calibration)
    prerequisite = {}
    if args.stage == "screen":
        prerequisite = require_gate(args.prerequisite, args.checkpoint, "pilot")
    elif args.stage == "validate":
        prerequisite = require_gate(args.prerequisite, args.checkpoint, "screen")
    elif args.stage == "bc-evaluate":
        if payload["kind"] != "context_bc" or not payload["contract"].get("qualified_teacher_gate"):
            raise ValueError("BC must come from a qualified teacher's new demonstrations")
    elif args.stage == "ppo-test":
        if payload["kind"] != "context_ppo" or not payload.get("selection_report", {}).get("selection_pass"):
            raise ValueError("Final PPO testing requires a candidate that passed fixed selection")
    _, anchor_payload = load_anchor(args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed, ImprovedRecipe(**anchor_payload["recipe"]))
    assert_fresh(manifest, payload["contract"], prerequisite)
    identity = dict(schema=SCHEMA, stage=args.stage, runtime_hash=runtime_hash(),
        checkpoint_sha256=file_hash(args.checkpoint), checkpoint=str(args.checkpoint.resolve()),
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration), manifest=manifest)
    immutable_contract(args.out / "evaluation_contract.json", identity)
    baseline = evaluate(manifest, args.init, args.calibration, None, args.out / "v15_bc", args.workers)
    rows = evaluate(manifest, args.init, args.calibration, args.checkpoint, args.out / "candidate", args.workers)
    checked = report(rows, baseline)
    absolute = args.stage in ("validate", "bc-evaluate", "ppo-test")
    passed = eligible(checked, absolute=absolute)
    if args.stage == "ppo-test":
        bc_path = Path(payload["contract"]["ppo_bc_checkpoint"])
        if file_hash(bc_path) != payload["contract"]["ppo_bc_sha256"]:
            raise ValueError("PPO's frozen V23 BC baseline changed")
        bc_rows = evaluate(manifest, args.init, args.calibration, bc_path, args.out / "v23_bc", args.workers)
        vs_bc = report(rows, bc_rows)
        improvement = float(np.mean([row["utility"] for row in rows]) - np.mean([row["utility"] for row in bc_rows]))
        passed = passed and eligible(vs_bc, absolute=True) and improvement > 0.
        write_json(args.out / "ppo_vs_bc.json", dict(report=vs_bc, utility_improvement=improvement))
        if passed:
            save_model(args.out / "ppo_accepted.pt", model, kind="context_ppo", contract=payload["contract"],
                independent_report=checked, independent_vs_bc=vs_bc, accepted=True)
    write_json(args.out / "gate_result.json", dict(contract=identity, report=checked, passed=passed))
    if passed and args.stage in ("screen", "validate"):
        name = "selected_config.json" if args.stage == "screen" else "qualified_teacher.json"
        write_json(args.out / name, dict(contract=identity, config=payload["contract"]["config"],
            qualified=args.stage == "validate", checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=file_hash(args.checkpoint), report=checked))
    print(dict(stage=args.stage, passed=passed, summary=checked["summary"], checks=checked["checks"]), flush=True)
    return passed


def collect(args):
    qualified = require_gate(args.prerequisite, args.checkpoint, "validate")
    _, payload = load_model(args.checkpoint)
    assert_model_contract(payload, args.init, args.calibration)
    _, anchor_payload = load_anchor(args.init)
    manifest = scenarios(args.episodes, args.seed, args.design_seed, ImprovedRecipe(**anchor_payload["recipe"]))
    assert_fresh(manifest, payload["contract"], qualified)
    # Whole episodes, never individual frames, define the held-out BC split.
    order = np.random.default_rng(args.design_seed).permutation(len(manifest["scenarios"]))
    validation = set(order[:max(1, len(order) // 5)].tolist())
    for index, scenario in enumerate(manifest["scenarios"]):
        scenario["split"] = "validation" if index in validation else "train"
    identity = dict(schema=SCHEMA, runtime_hash=runtime_hash(), manifest=manifest,
        teacher_sha256=file_hash(args.checkpoint), model_contract=payload["contract"],
        actions=payload["actions"], qualified_teacher_gate=qualified["contract"])
    immutable_contract(args.out / "dataset_contract.json", identity)
    rows = evaluate(manifest, args.init, args.calibration, args.checkpoint,
        args.out / "episodes", args.workers)
    write_json(args.out / "dataset_summary.json", dict(episodes=len(rows),
        decisions=sum(row["decision"] is not None for row in rows),
        hit15=sum(row["result"]["hit15"] for row in rows)))


def bc(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.design_seed)
    data = read_json(args.data / "dataset_contract.json")
    if data["schema"] != SCHEMA or data["runtime_hash"] != runtime_hash() or not data.get("qualified_teacher_gate"):
        raise ValueError("BC needs current V23 data from a qualified continuous teacher")
    rows = [read_json(args.data / "episodes" / f"episode_{s['seed']}.json") for s in data["manifest"]["scenarios"]]
    samples = [row for row in rows if row["decision"] is not None]
    x = torch.tensor([row["decision"]["context"] for row in samples])
    y = torch.tensor([row["decision"]["index"] for row in samples], dtype=torch.long)
    train = np.array([i for i, row in enumerate(samples) if row["scenario"]["split"] == "train"])
    valid = np.array([i for i, row in enumerate(samples) if row["scenario"]["split"] == "validation"])
    if len(train) < 100 or len(valid) < 20:
        raise ValueError("Insufficient independent V23 BC episodes")
    model = ContextPolicy(data["actions"])
    model.mean.copy_(normalization(x[train].numpy())[0])
    model.scale.copy_(normalization(x[train].numpy())[1])
    optimizer = torch.optim.AdamW(model.actor.parameters(), lr=3e-4, weight_decay=.001)
    best_loss, best, history = float("inf"), None, []
    for epoch in range(args.epochs):
        order = torch.tensor(train)[torch.randperm(len(train))]
        for batch in order.split(128):
            loss = nn.functional.cross_entropy(model.actor(model.normalize(x[batch])), y[batch])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        with torch.no_grad():
            prediction = model.actor(model.normalize(x[valid]))
            val_loss = float(nn.functional.cross_entropy(prediction, y[valid]))
            accuracy = float((prediction.argmax(-1) == y[valid]).float().mean())
        history.append(dict(epoch=epoch + 1, validation_loss=val_loss, action_accuracy=accuracy))
        if val_loss < best_loss:
            best_loss, best = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    contract = dict(data["model_contract"], runtime_hash=runtime_hash(),
        qualified_teacher_gate=data["qualified_teacher_gate"], dataset_sha256=file_hash(args.data / "dataset_contract.json"),
        training_seeds=[row["seed"] for row in samples if row["scenario"]["split"] == "train"],
        selection_seeds=[row["seed"] for row in samples if row["scenario"]["split"] == "validation"])
    contract["ancestor_seeds"] = sorted(lineage_seeds(data))
    save_model(args.out / "bc_best.pt", model, kind="context_bc", contract=contract)
    write_json(args.out / "bc_history.json", history)
    print(dict(bc=str(args.out / "bc_best.pt"), independent_closed_loop_evaluation_pending=True), flush=True)


def ppo(args):
    """One-step episodic PPO: fresh post-capture contexts, categorical orbit action.

    GAE is unnecessary because this policy acts exactly once per episode. The
    terminal actual-outcome return is the contextual critic's regression target.
    """
    torch.set_num_threads(1)
    torch.manual_seed(args.design_seed)
    gate_record = require_gate(args.prerequisite, args.checkpoint, "bc-evaluate")
    model, payload = load_model(args.checkpoint)
    if payload["kind"] != "context_bc":
        raise ValueError("Formal V23 PPO starts only from independently accepted V23 BC")
    assert_model_contract(payload, args.init, args.calibration)
    anchor_policy = copy.deepcopy(model).eval()
    for parameter in anchor_policy.parameters():
        parameter.requires_grad_(False)
    _, anchor_payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    count = args.updates * args.episodes
    manifest = scenarios(count, args.seed, args.design_seed, recipe)
    validation = scenarios(20, args.validation_seed, args.design_seed + 1, recipe)
    assert_fresh(manifest, payload["contract"], gate_record)
    assert_fresh(validation, payload["contract"], gate_record)
    if {row["seed"] for row in manifest["scenarios"]} & {row["seed"] for row in validation["scenarios"]}:
        raise ValueError("PPO train and selection seeds overlap")
    identity = dict(schema=SCHEMA, runtime_hash=runtime_hash(), manifest=manifest, validation_manifest=validation,
        bc_sha256=file_hash(args.checkpoint), bc_gate=gate_record["contract"], updates=args.updates,
        episodes_per_update=args.episodes, learning_rate=1e-4, clip=.1, epochs=4,
        anchor_kl=.05, entropy=.01, model_contract=payload["contract"])
    if not (args.out / "ppo_contract.json").exists():
        if {row["seed"] for row in validation["scenarios"]} & previous_seeds(args.out):
            raise ValueError("PPO selection scenes overlap previous work")
    immutable_contract(args.out / "ppo_contract.json", identity)
    policy_contract = dict(payload["contract"], ppo_bc_checkpoint=str(args.checkpoint.resolve()),
        ppo_bc_sha256=file_hash(args.checkpoint),
        ancestor_seeds=sorted(lineage_seeds(identity)))
    optimizer = torch.optim.Adam([dict(params=model.actor.parameters(), lr=1e-4),
                                 dict(params=model.critic.parameters(), lr=1e-3)])
    start = 0
    latest = args.out / "ppo_latest.pt"
    if latest.exists():
        model, resumed = load_model(latest)
        if resumed["ppo_contract"] != identity:
            raise ValueError("PPO resume contract mismatch")
        optimizer = torch.optim.Adam([dict(params=model.actor.parameters(), lr=1e-4),
                                     dict(params=model.critic.parameters(), lr=1e-3)])
        optimizer.load_state_dict(resumed["optimizer"])
        torch.set_rng_state(resumed["torch_rng"])
        start = resumed["update"]
    bc_rows = evaluate(validation, args.init, args.calibration, args.checkpoint,
        args.out / "selection_bc", args.workers)
    baseline_rows = evaluate(validation, args.init, args.calibration, None,
        args.out / "selection_v15", args.workers)
    metrics_path = args.out / "ppo_history.json"
    history = read_json(metrics_path) if metrics_path.exists() else []
    history = [item for item in history if item["update"] <= start]
    best_score = max(float(np.mean([row["utility"] for row in bc_rows])),
                     max((item.get("selection_score", -1e9) for item in history), default=-1e9))
    for update in range(start, args.updates):
        rollout_model = args.out / "checkpoints" / f"rollout_{update:04d}.pt"
        save_model(rollout_model, model, kind="context_ppo", contract=policy_contract, update=update)
        training = dict(scenarios=manifest["scenarios"][update * args.episodes:(update + 1) * args.episodes])
        rows = evaluate(training, args.init, args.calibration, rollout_model,
            args.out / "rollouts" / f"update_{update + 1:04d}", args.workers, stochastic=True, salt=100000000)
        samples = [row for row in rows if row["decision"] is not None]
        if len(samples) < 2:
            raise ValueError("Too few captured episodes for a meaningful PPO update")
        x = torch.tensor([row["decision"]["context"] for row in samples])
        indices = torch.tensor([row["decision"]["index"] for row in samples])
        old_logp = torch.tensor([row["decision"]["logp"] for row in samples])
        targets = torch.tensor([row["utility"] / 40. for row in samples])
        old_value = torch.tensor([row["decision"]["value"] for row in samples])
        advantage = targets - old_value
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(.05)
        with torch.no_grad():
            reference_distribution = anchor_policy.distribution(x)
        approx_kl = 0.
        for epoch in range(4):
            distribution = model.distribution(x)
            logp = distribution.log_prob(indices)
            ratio = (logp - old_logp).exp()
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(.9, 1.1) * advantage).mean()
            value_loss = nn.functional.mse_loss(model.value(x), targets)
            anchor_kl = kl_divergence(reference_distribution, distribution).mean()
            loss = policy_loss + .5 * value_loss + .05 * anchor_kl - .01 * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .8)
            optimizer.step()
            with torch.no_grad():
                change = model.distribution(x).log_prob(indices) - old_logp
                approx_kl = float((change.exp() - 1. - change).mean())
            if approx_kl > .01:
                break
        record = dict(update=update + 1, captured_samples=len(samples), mean_utility=float(targets.mean() * 40.),
                      approx_kl=approx_kl, actual_hit15=sum(row["result"]["hit15"] for row in rows))
        checkpoint = args.out / "checkpoints" / f"ppo_{update + 1:04d}.pt"
        save_model(checkpoint, model, kind="context_ppo", contract=policy_contract, update=update + 1)
        if (update + 1) % 5 == 0 or update + 1 == args.updates:
            evaluated = evaluate(validation, args.init, args.calibration, checkpoint,
                args.out / "selection" / f"update_{update + 1:04d}", args.workers)
            vs_bc, vs_v15 = report(evaluated, bc_rows), report(evaluated, baseline_rows)
            accepted = eligible(vs_bc) and eligible(vs_v15)
            score = float(np.mean([row["utility"] for row in evaluated]))
            record.update(vs_bc=vs_bc, vs_v15=vs_v15, selection_pass=accepted)
            if accepted and score > best_score:
                best_score = score
                record["selection_score"] = score
                save_model(args.out / "ppo_candidate.pt", model, kind="context_ppo",
                    contract=policy_contract, update=update + 1, selection_report=record,
                    independent_test_pending=True)
        history.append(record)
        # Commit history and optimizer/RNG state after a complete update only.
        write_json(metrics_path, history)
        save_model(latest, model, kind="context_ppo", contract=policy_contract, update=update + 1,
            ppo_contract=identity, optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state())
        print(dict(update=update + 1, mean_utility=record["mean_utility"], approx_kl=approx_kl), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "fit-outcome", "gate", "collect", "bc", "ppo"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
        cmd.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--design-seed", type=int, default=20261011)
        if name in ("fit", "fit-outcome", "bc"):
            cmd.add_argument("--data", type=Path, nargs="+" if name in ("fit", "fit-outcome") else None, required=True)
            cmd.add_argument("--epochs", type=int, default=400 if name == "fit-outcome" else (600 if name == "fit" else 100))
        if name in ("gate", "collect", "ppo"):
            cmd.add_argument("--checkpoint", type=Path, required=True)
            cmd.add_argument("--prerequisite", type=Path, default=DEFAULT_OUT / "missing_gate")
            cmd.add_argument("--episodes", type=int, default=20 if name == "gate" else (1000 if name == "collect" else 16))
            cmd.add_argument("--workers", type=int, default=8)
            cmd.add_argument("--seed", type=int, required=True)
        if name == "gate":
            cmd.add_argument("--stage", choices=("pilot", "screen", "validate", "bc-evaluate", "ppo-test"), required=True)
        if name == "ppo":
            cmd.add_argument("--updates", type=int, default=100)
            cmd.add_argument("--validation-seed", type=int, required=True)
    args = parser.parse_args()
    if hasattr(args, "workers") and (args.workers < 1 or args.episodes < 1):
        parser.error("episodes and workers must be positive")
    if args.command == "gate":
        minimum = 80 if args.stage in ("validate", "bc-evaluate", "ppo-test") else 20
        if args.episodes < minimum:
            parser.error(f"{args.stage} requires at least {minimum} fresh paired scenes")
    if args.command == "collect" and args.episodes < 1000:
        parser.error("Formal V23 demonstration collection requires at least 1000 episodes")
    result = {"fit": fit, "fit-outcome": fit_outcome, "gate": gate,
              "collect": collect, "bc": bc, "ppo": ppo}[args.command](args)
    if args.command == "gate" and not result:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
