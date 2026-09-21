"""V15 bounded PPO with a frozen BC anchor, resumable updates and paired evaluation."""
from __future__ import annotations

import argparse
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

from rl_ppo.ppo import PPOConfig, RolloutBuffer
from .__main__ import emit
from .data import load_dataset, write_json
from .env import ACTION_SIZE, OBS_SIZE
from .generalized_teacher import PARAMETER_RANGES, _scenario_rows, make_scenario_manifest
from .improved_bc import DEFAULT_DATA, DEFAULT_OUT, balanced_quality_weights, masked_mse, tensor_data, validate_dataset
from .improved_teacher import IMPROVED_SCHEMA, ImprovedRecipe, ImprovedTeacherEnv, improved_fingerprint
from .model import load_checkpoint, save_checkpoint
from .reachable_collection import _grasp_height, constrain_manifest
from .residual_env import FIXED_ACTIONS, RL_SCHEMA, QualityConfig, ResidualEnv, policy_mask


DEFAULT_RL_OUT = Path("teacher_runs/v15_improved/rl_v1_formal")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_best(output, update):
    source = output / "checkpoints" / f"selected_{update:04d}.pt"
    temporary = output / "ppo_best.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(output / "ppo_best.pt")


def rl_fingerprint():
    digest = hashlib.sha256(improved_fingerprint().encode())
    paths = [Path(__file__), Path(__file__).with_name("residual_env.py"),
        Path(__file__).with_name("improved_bc.py"), Path(__file__).with_name("reachable_collection.py")]
    paths += [Path(__file__).parents[1] / "rl_ppo" / name for name in ("ppo.py", "networks.py")]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_anchor(path):
    model, payload = load_checkpoint(path, expected_schema=IMPROVED_SCHEMA)
    if payload.get("kind") != "v15_bc" or payload.get("source_hash") != improved_fingerprint():
        raise ValueError("requires a BC checkpoint matching the current V15 teacher")
    if payload.get("recipe") != payload.get("dataset_contract", {}).get("recipe"):
        raise ValueError("BC recipe and dataset contract differ")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def historical_seeds(anchor_path, payload):
    seeds = set(payload["demonstration_seeds"]) | set(payload["selection_seeds"])
    root = Path(anchor_path).resolve().parent
    for path in root.glob("**/evaluation_contract.json"):
        record = read_json(path)
        manifest = record.get("scenario_manifest", record.get("manifest", {}))
        seeds.update(row["seed"] for row in manifest.get("scenarios", []))
    audit = root.parent / "audit_formal" / "audit_summary.json"
    if audit.is_file():
        seeds.update(row["seed"] for row in read_json(audit)["manifest"]["scenarios"])
    return sorted(seeds)


def training_scenarios(count, first_seed, design_seed, focus_fraction, height):
    rows = _scenario_rows(count, "train", first_seed, design_seed, 0)
    rng = np.random.default_rng(design_seed + 17)
    focus = set(rng.choice(count, int(round(count * focus_fraction)), replace=False).tolist())
    for index, row in enumerate(rows):
        row["sampling_group"] = "mid_speed" if index in focus else "full_range"
        if index in focus:
            row["launch_speed"] = float(rng.uniform(5., 5.25))
    return constrain_manifest(dict(scenarios=rows, episodes=count), height)["scenarios"]


def ensure_disjoint(groups):
    seen = set()
    for name, values in groups.items():
        values = set(values)
        if values & seen:
            raise ValueError(f"seed overlap in {name}: {sorted(values & seen)[:5]}")
        seen.update(values)


def _init_worker():
    torch.set_num_threads(1)


def episode(job):
    scenario, recipe_data, checkpoint, schema, quality_data, stochastic, policy_seed = job
    torch.manual_seed(policy_seed)
    recipe = ImprovedRecipe(**recipe_data)
    policy = load_checkpoint(checkpoint, expected_schema=schema)[0] if checkpoint else None
    if policy is not None:
        policy.eval()
    env = (ResidualEnv(scenario, recipe, QualityConfig(**quality_data))
        if quality_data is not None else ImprovedTeacherEnv(scenario, recipe))
    started = time.perf_counter()
    observation, done = env.reset(scenario["seed"]), False
    samples = []
    while not done:
        mask = policy_mask(env.mask())
        if policy is None:
            action = env.teacher_action()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(observation[None])
                distribution = policy.distribution(obs_tensor)
                raw = distribution.sample() if stochastic else distribution.mean
                # V15 overrides these physical commands. Do not explore their
                # observation-history entries or include them in PPO likelihoods.
                raw[:, list(FIXED_ACTIONS)] = distribution.mean[:, list(FIXED_ACTIONS)]
                action = raw.tanh()[0].numpy()
                if stochastic:
                    log_prob = policy._squashed_log_probability(distribution, raw, torch.tensor(mask))
                    value = policy.critic(policy.normalize(obs_tensor)).squeeze(-1)
        following, reward, done, summary = env.step(action)
        if stochastic:
            samples.append((observation, raw[0].numpy(), log_prob.item(), reward,
                done, value.item(), mask))
        observation = following
    return dict(seed=scenario["seed"], scenario=scenario, result=summary,
        wall_s=time.perf_counter()-started, samples=samples)


def run_jobs(pool, jobs, label):
    futures = [pool.submit(episode, job) for job in jobs]
    rows = []
    try:
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            emit(dict(stage=label, completed=len(rows), total=len(jobs), seed=row["seed"],
                captured=row["result"]["captured"], hit15=row["result"]["hit15"]))
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return sorted(rows, key=lambda row: row["seed"])


def summarize(rows):
    results = [row["result"] for row in rows]
    summary = dict(episodes=len(rows))
    for key in ("captured", "released", "hit15", "hit30"):
        summary[key] = sum(bool(r[key]) for r in results)
        summary[key + "_rate"] = summary[key] / len(rows)
    for key in ("landing_error_m", "weld_peak_n", "impact_weld_impulse_ns",
            "relative_capture_speed_m_s", "pressure_integral_psi_s", "contact_impulse_ns"):
        values = [r[key] for r in results if r.get(key) is not None]
        summary["mean_"+key] = float(np.mean(values)) if values else None
    summary["max_contact_peak_n"] = max(r["contact_peak_n"] for r in results)
    return summary


def parameter_bins(rows):
    bins, worst = {}, 1.
    for key, (low, high) in PARAMETER_RANGES.items():
        edges = np.linspace(low, high, 5)
        bins[key] = []
        for index in range(4):
            selected = [row for row in rows if edges[index] <= row["scenario"][key] and
                (row["scenario"][key] < edges[index+1] or index == 3)]
            metric = summarize(selected) if selected else dict(episodes=0, hit15_rate=None)
            bins[key].append(dict(low=float(edges[index]), high=float(edges[index+1]), **metric))
            if selected:
                worst = min(worst, metric["hit15_rate"])
    return bins, worst


def compare(rows, reference):
    by_seed = {row["seed"]: row for row in reference}
    if len(by_seed) != len(rows) or {row["seed"] for row in rows} != set(by_seed):
        raise ValueError("comparison requires matching unique seeds")
    for row in rows:
        if row["scenario"] != by_seed[row["seed"]]["scenario"]:
            raise ValueError("comparison scenarios differ")
    current, baseline = summarize(rows), summarize(reference)
    pairs = [(row["result"], by_seed[row["seed"]]["result"]) for row in rows
        if row["result"]["hit15"] and by_seed[row["seed"]]["result"]["hit15"]]
    keys = ("weld_peak_n", "impact_weld_impulse_ns", "relative_capture_speed_m_s",
        "pressure_integral_psi_s", "contact_impulse_ns")
    ratios = {}
    for key in keys:
        values = [(a[key], b[key]) for a, b in pairs if a.get(key) is not None and b.get(key) is not None]
        if not values:
            ratios[key] = None
        else:
            a, b = np.mean(values, axis=0)
            ratios[key] = float((a+1e-6)/(b+1e-6))
    quality_score = (sum(weight * ratios[key] for key, weight in zip(keys, (.3, .25, .25, .15, .05)))
        if pairs and all(value is not None for value in ratios.values()) else None)
    error, base_error = current["mean_landing_error_m"], baseline["mean_landing_error_m"]
    checks = {key+"_drop_le_2pp": current[key+"_rate"]+1e-12 >= baseline[key+"_rate"]-.02
        for key in ("captured", "released", "hit15")}
    checks["landing_guard"] = error is not None and base_error is not None and error <= min(.07, base_error+.01)
    checks["paired_successes"] = bool(pairs)
    for key in ("weld_peak_n", "impact_weld_impulse_ns", "pressure_integral_psi_s"):
        checks[key+"_guard"] = ratios[key] is not None and ratios[key] <= 1.05+1e-12
    checks["p95_weld_guard"] = bool(pairs) and np.percentile(
        [a["weld_peak_n"] for a, _ in pairs], 95) <= 1.05*np.percentile(
            [b["weld_peak_n"] for _, b in pairs], 95)+1e-9
    checks["p95_weld_guard"] = bool(checks["p95_weld_guard"])
    checks["contact_guard"] = current["max_contact_peak_n"] <= baseline["max_contact_peak_n"]+10.
    bins, worst = parameter_bins(rows)
    eligible = all(checks.values())
    return dict(summary=current, baseline=baseline, checks=checks, eligible=eligible,
        paired_hit15=len(pairs), paired_quality_ratios=ratios, quality_score=quality_score,
        quality_improved=eligible and quality_score is not None and quality_score <= .97,
        parameter_bins=bins, worst_parameter_bin_hit15_rate=worst,
        absolute_generalization_pass=eligible and len(rows) >= 80 and worst >= .8 and
            all(group["episodes"] >= 10 for groups in bins.values() for group in groups))


def selection_score(report):
    summary = report["summary"]
    return (summary["hit15"], summary["captured"], summary["released"],
        -(report["quality_score"] if report["quality_score"] is not None else 1e9),
        -(summary["mean_landing_error_m"] if summary["mean_landing_error_m"] is not None else 1e9))


def update_policy(model, anchor, optimizer, buffer, config, bc_data, bc_weight, kl_weight):
    count = buffer.size
    obs = torch.tensor(buffer.observations[:count])
    raw = torch.tensor(buffer.raw_actions[:count])
    masks = torch.tensor(buffer.action_masks[:count])
    old = torch.tensor(buffer.log_probs[:count])
    returns = torch.tensor(buffer.returns[:count])
    advantage = torch.tensor(buffer.advantages[:count])
    active = masks.sum(-1) > 0
    if not active.any():
        raise ValueError("rollout has no effective actions")
    advantage = (advantage-advantage[active].mean()) / advantage[active].std(unbiased=False).clamp_min(1e-8)
    weights = balanced_quality_weights(bc_data)
    records, stopped = [], False
    for epoch in range(config.update_epochs):
        for indices in torch.randperm(count).split(config.minibatch_size):
            lp, entropy, value = model.evaluate(obs[indices], raw[indices], masks[indices])
            valid = active[indices]
            log_ratio = lp-old[indices]
            ratio = log_ratio.exp()
            approximate_kl = float(((ratio[valid]-1)-log_ratio[valid]).mean().detach()) if valid.any() else 0.
            if approximate_kl > config.target_kl:
                stopped = True
                break
            surrogate = torch.minimum(ratio*advantage[indices],
                ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio)*advantage[indices])
            policy_loss = -surrogate[valid].mean() if valid.any() else value.sum()*0.
            with torch.no_grad():
                reference = anchor.distribution(obs[indices])
            kl = (torch.distributions.kl_divergence(reference, model.distribution(obs[indices]))*
                masks[indices]).sum(-1)
            anchor_kl = kl[valid].mean() if valid.any() else value.sum()*0.
            demo_indices = torch.multinomial(weights, len(indices), replacement=True)
            bc_loss = masked_mse(model, bc_data, demo_indices)
            value_loss = .5*(value-returns[indices]).square().mean()
            entropy_loss = entropy[valid].mean() if valid.any() else value.sum()*0.
            loss = policy_loss+config.value_coefficient*value_loss+bc_weight*bc_loss+kl_weight*anchor_kl
            loss -= config.entropy_coefficient*entropy_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            # The BC critic is untrained; its large initial gradients must not
            # consume the actor's clipping budget.
            torch.nn.utils.clip_grad_norm_([*model.actor.parameters(), model.log_std], config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), config.max_grad_norm)
            optimizer.step()
            records.append([float(x.detach()) for x in (policy_loss, value_loss, bc_loss, anchor_kl)]+[approximate_kl])
        if stopped:
            break
    if not records:
        raise RuntimeError("PPO stopped before any update; rollout policy and current policy may differ")
    result = dict(zip(("policy_loss", "value_loss", "bc_loss", "anchor_kl", "approx_kl"),
        np.mean(records, axis=0).tolist()))
    return dict(**result, epochs=epoch+1, kl_early_stop=stopped, samples=count,
        active_samples=int(active.sum()))


def eval_policy(pool, scenarios, recipe, checkpoint, schema, quality, label):
    jobs = [(row, recipe, str(Path(checkpoint).resolve()) if checkpoint else None,
        schema, quality, False, 0) for row in scenarios]
    return run_jobs(pool, jobs, label)


def train(args):
    anchor, bc_payload = load_anchor(args.init)
    validated = validate_dataset(args.data)
    if (validated["inventory"] != bc_payload["dataset"] or
            validated["contract"] != bc_payload["dataset_contract"]):
        raise ValueError("BC anchor and demonstration dataset differ")
    recipe = ImprovedRecipe(**bc_payload["recipe"])
    quality = QualityConfig(weight=args.quality_weight)
    height = _grasp_height(recipe, validated["manifest"]["scenarios"][0])
    validation = constrain_manifest(make_scenario_manifest(args.eval_episodes,
        args.eval_seed, args.design_seed+100000), height)
    previous_seeds = historical_seeds(args.init, bc_payload)
    ensure_disjoint(dict(history=previous_seeds,
        train=range(args.seed, args.seed+args.updates*args.episodes_per_update),
        validation=[row["seed"] for row in validation["scenarios"]]))
    for path in args.out.glob("**/evaluation_contract.json"):
        tested = {row["seed"] for row in read_json(path)["manifest"]["scenarios"]}
        ensure_disjoint(dict(tested=tested,
            train=range(args.seed, args.seed+args.updates*args.episodes_per_update)))
    config = PPOConfig(gamma=.999, gae_lambda=.98, learning_rate=args.lr,
        clip_ratio=.1, entropy_coefficient=.0001, target_kl=.01,
        update_epochs=3, minibatch_size=256, max_grad_norm=1.)
    contract = dict(schema=RL_SCHEMA, source_hash=improved_fingerprint(),
        rl_source_hash=rl_fingerprint(), anchor_sha256=file_hash(args.init),
        dataset=validated["inventory"], dataset_contract=validated["contract"],
        recipe=asdict(recipe), ppo=asdict(config), quality=asdict(quality),
        train_seed=args.seed, design_seed=args.design_seed, random_seed=args.random_seed,
        episodes_per_update=args.episodes_per_update, focus_fraction=args.focus_fraction,
        eval_every=args.eval_every, validation_manifest=validation,
        excluded_seeds=previous_seeds, bc_weight=args.bc_weight, kl_weight=args.kl_weight,
        critic_lr=args.critic_lr, threads=args.threads, fixed_actions=list(FIXED_ACTIONS))
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "run_config.json"
    start, best_update = 0, 0
    if args.resume:
        if Path(args.resume).resolve() != (args.out / "ppo_latest.pt").resolve():
            raise ValueError("resume must use this output directory's ppo_latest.pt")
        model, payload = load_checkpoint(args.resume, expected_schema=RL_SCHEMA)
        if payload.get("kind") != "v15_ppo" or payload.get("run_contract") != contract:
            raise ValueError("resume source, dataset, anchor or training configuration changed")
        if not config_path.is_file() or read_json(config_path) != contract:
            raise ValueError("run_config.json differs from resume contract")
        start, best_update = payload["update"], payload["best_update"]
        if args.updates < start:
            raise ValueError("updates is the total target, and cannot be less than the resumed update")
        baseline, best_score = payload["baseline"], tuple(payload["best_score"])
        publish_best(args.out, best_update)
    else:
        if any(args.out.iterdir()):
            bootstrap_files = {"run_config.json", "validation_0000.json", "checkpoints"}
            if (not config_path.is_file() or read_json(config_path) != contract or
                    any(path.name not in bootstrap_files for path in args.out.iterdir()) or
                    any(path.name not in ("selected_0000.pt", "selected_0000.tmp")
                        for path in (args.out / "checkpoints").glob("*"))):
                raise ValueError("output is not empty; use Resume or a new output directory")
        model, _ = load_checkpoint(args.init, expected_schema=IMPROVED_SCHEMA)
        model.schema = RL_SCHEMA
        # Train only the privileged critic; preserve the BC actor and normalization.
        torch.nn.init.zeros_(model.critic[-1].weight)
        torch.nn.init.zeros_(model.critic[-1].bias)
        write_json(config_path, contract)
    optimizer = torch.optim.Adam([
        dict(params=[*model.actor.parameters(), model.log_std], lr=args.lr),
        dict(params=model.critic.parameters(), lr=args.critic_lr)])
    if args.resume:
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    dataset = load_dataset(args.data, expected_schema=IMPROVED_SCHEMA, source_hash=improved_fingerprint())
    bc_data = tensor_data(dataset["train"])
    bc_data = (*bc_data[:2], torch.tensor(policy_mask(bc_data[2].numpy())), *bc_data[3:])

    def save(update, path):
        save_checkpoint(path, model, kind="v15_ppo", update=update,
            run_contract=contract, optimizer=optimizer.state_dict(),
            best_update=best_update, best_score=list(best_score), baseline=baseline,
            training_seed_end=args.seed+update*args.episodes_per_update,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate())

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        if not args.resume:
            baseline_path = args.out / "validation_0000.json"
            baseline = (read_json(baseline_path)["rows"] if baseline_path.exists() else
                eval_policy(pool, validation["scenarios"], asdict(recipe),
                    args.init, IMPROVED_SCHEMA, None, "validation_bc"))
            report = compare(baseline, baseline)
            best_score = selection_score(report)
            write_json(baseline_path, dict(update=0, report=report, rows=baseline))
            save(0, args.out / "checkpoints" / "selected_0000.pt")
            save(0, args.out / "ppo_latest.pt")
            publish_best(args.out, 0)
        for update in range(start+1, args.updates+1):
            snapshot = args.out / "rollout_policy.pt"
            save_checkpoint(snapshot, model, kind="v15_rollout", update=update-1)
            scenarios = training_scenarios(args.episodes_per_update,
                args.seed+(update-1)*args.episodes_per_update,
                args.design_seed+update, args.focus_fraction, height)
            jobs = [(row, asdict(recipe), str(snapshot.resolve()), RL_SCHEMA, asdict(quality),
                True, (args.random_seed+row["seed"]) % (2**32)) for row in scenarios]
            rows = run_jobs(pool, jobs, f"train_{update}")
            buffer = RolloutBuffer(sum(len(row["samples"]) for row in rows), OBS_SIZE, ACTION_SIZE)
            for row in rows:
                for sample in row.pop("samples"):
                    buffer.add(*sample)
            buffer.finish(0., config)
            metrics = update_policy(model, anchor, optimizer, buffer, config, bc_data,
                args.bc_weight, args.kl_weight)
            # Per-update files are atomically replaced on replay after interruption.
            write_json(args.out / f"update_{update:04d}.json", dict(update=update,
                metrics=metrics, training_summary=summarize(rows), rows=rows))
            if update % args.eval_every == 0 or update == args.updates:
                save_checkpoint(snapshot, model, kind="v15_rollout", update=update)
                evaluation = eval_policy(pool, validation["scenarios"], asdict(recipe),
                    snapshot, RL_SCHEMA, asdict(quality), f"validation_{update}")
                report = compare(evaluation, baseline)
                improved = report["eligible"] and selection_score(report) > best_score
                if improved:
                    best_update, best_score = update, selection_score(report)
                write_json(args.out / f"validation_{update:04d}.json", dict(update=update,
                    selected=improved, report=report, rows=evaluation))
                if improved:
                    save(update, args.out / "checkpoints" / f"selected_{update:04d}.pt")
                emit(dict(update=update, validation=report["summary"], eligible=report["eligible"],
                    best_update=best_update, quality_score=report["quality_score"]))
            save(update, args.out / "ppo_latest.pt")
            publish_best(args.out, best_update)
            emit(dict(update=update, metrics=metrics, best_update=best_update))
    write_json(args.out / "training_summary.json", dict(completed_updates=args.updates,
        total_training_episodes=args.updates*args.episodes_per_update, best_update=best_update,
        best_is_bc_initialization=best_update == 0,
        elapsed_this_invocation_s=time.perf_counter()-started))


def evaluate(args):
    _, anchor_payload = load_anchor(args.init)
    _, payload = load_checkpoint(args.checkpoint, expected_schema=RL_SCHEMA)
    contract = payload.get("run_contract", {})
    if (payload.get("kind") != "v15_ppo" or contract.get("source_hash") != improved_fingerprint()
            or contract.get("rl_source_hash") != rl_fingerprint()
            or contract.get("anchor_sha256") != file_hash(args.init)):
        raise ValueError("evaluation requires matching V15 PPO source and BC anchor")
    recipe = ImprovedRecipe(**contract["recipe"])
    manifest = make_scenario_manifest(args.episodes, args.seed, args.design_seed)
    manifest = constrain_manifest(manifest, _grasp_height(recipe, manifest["scenarios"][0]))
    forbidden = set(contract["excluded_seeds"]) | set(historical_seeds(args.init, anchor_payload))
    forbidden.update(row["seed"] for row in contract["validation_manifest"]["scenarios"])
    # The best checkpoint may precede the latest training update.
    end = payload["training_seed_end"]
    latest_path = args.checkpoint.parent / "ppo_latest.pt"
    if latest_path.is_file():
        _, latest = load_checkpoint(latest_path, expected_schema=RL_SCHEMA)
        if latest.get("run_contract") != contract:
            raise ValueError("latest and selected checkpoints have different contracts")
        end = max(end, latest["training_seed_end"])
    forbidden.update(range(contract["train_seed"], end))
    ensure_disjoint(dict(previous=forbidden, test=[row["seed"] for row in manifest["scenarios"]]))
    identity = dict(schema=RL_SCHEMA, rl_source_hash=rl_fingerprint(),
        checkpoint_sha256=file_hash(args.checkpoint), anchor_sha256=file_hash(args.init),
        selected_update=payload["update"], manifest=manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "evaluation_contract.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("evaluation output contract changed; use a new directory")
    write_json(path, identity)
    all_rows = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for name, checkpoint, schema, quality in (
                ("teacher", None, IMPROVED_SCHEMA, None),
                ("bc", args.init, IMPROVED_SCHEMA, None),
                ("rl", args.checkpoint, RL_SCHEMA, contract["quality"])):
            result_path = args.out / (name+"_episodes.json")
            if result_path.exists():
                all_rows[name] = read_json(result_path)
            else:
                all_rows[name] = eval_policy(pool, manifest["scenarios"], asdict(recipe),
                    checkpoint, schema, quality, "test_"+name)
                write_json(result_path, all_rows[name])
    report = dict(selected_update=payload["update"], episodes=args.episodes,
        summaries={name: summarize(rows) for name, rows in all_rows.items()},
        rl_vs_bc=compare(all_rows["rl"], all_rows["bc"]),
        rl_vs_teacher=compare(all_rows["rl"], all_rows["teacher"]))
    write_json(args.out / "comparison.json", report)
    emit(dict(summaries=report["summaries"], selected_update=payload["update"],
        eligible=report["rl_vs_bc"]["eligible"],
        quality_improved=report["rl_vs_bc"]["quality_improved"],
        absolute_generalization_pass=report["rl_vs_bc"]["absolute_generalization_pass"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--data", type=Path, default=DEFAULT_DATA)
    command.add_argument("--init", type=Path, default=DEFAULT_OUT / "bc_best.pt")
    command.add_argument("--out", type=Path, default=DEFAULT_RL_OUT)
    command.add_argument("--resume", type=Path)
    for name, default in (("updates", 100), ("episodes-per-update", 8), ("workers", 8),
            ("eval-every", 10), ("eval-episodes", 80), ("seed", 100001),
            ("eval-seed", 2000001), ("design-seed", 20260930),
            ("random-seed", 20260930), ("threads", 4)):
        command.add_argument("--"+name, type=int, default=default)
    for name, default in (("lr", 3e-5), ("critic-lr", 3e-4), ("bc-weight", 1.),
            ("kl-weight", .05), ("quality-weight", 1.), ("focus-fraction", .25)):
        command.add_argument("--"+name, type=float, default=default)
    command = sub.add_parser("evaluate")
    command.add_argument("--init", type=Path, default=DEFAULT_OUT / "bc_best.pt")
    command.add_argument("--checkpoint", type=Path, default=DEFAULT_RL_OUT / "ppo_best.pt")
    command.add_argument("--out", type=Path, default=DEFAULT_RL_OUT / "evaluation_200")
    for name, default in (("episodes", 200), ("workers", 8), ("seed", 3000001), ("design-seed", 20261001)):
        command.add_argument("--"+name, type=int, default=default)
    args = parser.parse_args()
    for name in ("updates", "episodes_per_update", "workers", "eval_every", "threads"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("eval_episodes", "episodes"):
        if hasattr(args, name) and getattr(args, name) < 5:
            parser.error(f"{name} must be >= 5; use >= 80 for formal validation")
    for name in ("lr", "critic_lr", "bc_weight", "kl_weight", "quality_weight", "focus_fraction"):
        if hasattr(args, name) and (not np.isfinite(getattr(args, name)) or getattr(args, name) < 0):
            parser.error(f"{name} must be finite and nonnegative")
    if args.command == "train":
        if args.lr <= 0 or args.critic_lr <= 0 or not 0 <= args.focus_fraction <= .5:
            parser.error("learning rates must be positive and focus-fraction must be in [0, 0.5]")
        QualityConfig(weight=args.quality_weight)
        for name in ("seed", "eval_seed", "design_seed", "random_seed"):
            if not 0 <= getattr(args, name) < 2**32:
                parser.error(f"{name} must be in [0, 2**32)")
    torch.set_num_threads(getattr(args, "threads", 1))
    seed = getattr(args, "random_seed", 20260930)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    {"train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
