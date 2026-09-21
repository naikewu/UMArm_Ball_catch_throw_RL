"""CLI: collect, bc, dagger, evaluate, ppo. Run from RL PPO with python -m teacher_rl."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from rl_ppo.ppo import PPOConfig, RolloutBuffer
from .env import TeacherEnv, TaskConfig, OBS_SIZE, SCHEMA
from .data import collect_one, load_dataset, rollout, write_json, fingerprint, dataset_inventory, episode_manifests
from .model import IntentPolicy, load_checkpoint, save_checkpoint


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def log(path, value):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, allow_nan=False) + "\n")
    emit(value)


def task(args):
    fields = ("target_distance", "orbit_radius", "target_azimuth", "launch_distance", "launch_speed", "mass", "ball_radius")
    values = {key: getattr(args, key) for key in fields}
    return TaskConfig(**values, authority=args.authority, quality_weight=getattr(args, "quality_weight", 0.))


def collection(args):
    checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else None
    if args.command == "dagger" and checkpoint is None:
        raise ValueError("dagger requires --checkpoint")
    cfg = asdict(task(args))
    jobs = [(s, cfg, str(args.out.resolve()), checkpoint, args.beta, args.perturb)
            for s in range(args.seed, args.seed + args.episodes)]
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_one, job) for job in jobs]
        try:
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                emit(dict(completed=len(rows), total=len(jobs), **row["result"]))
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            emit(dict(interrupted=True, message="Completed episodes are retained; waiting for active workers."))
            return
    write_json(args.out / "collection_summary.json", dict(schema=SCHEMA, episodes=len(rows),
        captured=sum(r["result"]["captured"] for r in rows),
        hit15=sum(r["result"]["hit15"] for r in rows),
        samples=sum(r["samples"] for r in rows), source_hash=fingerprint()))


def tensor_data(data):
    return tuple(torch.as_tensor(a) for a in data)


def phase_weights(data):
    _, _, masks, phases, quality = data
    _, inv, counts = torch.unique(phases, return_inverse=True, return_counts=True)
    return quality / counts[inv].float() * (masks.sum(-1) > 0)


def imitation_loss(model, data, indices):
    obs, targets, masks, _, _ = data
    prediction = model.distribution(obs[indices]).mean.tanh()
    return (((prediction-targets[indices]).square()*masks[indices]).sum(-1)
            / masks[indices].sum(-1).clamp_min(1)).mean()


def clone(args, *, dataset=None, model=None, checkpoint_extra=None):
    dataset = load_dataset(args.data) if dataset is None else dataset
    checkpoint_extra = checkpoint_extra or {}
    inventory = dataset_inventory(args.data)
    train, validation = tensor_data(dataset["train"]), tensor_data(dataset["validation"])
    if model is None:
        model = load_checkpoint(args.init)[0] if args.init else IntentPolicy()
    if not args.init:
        model.obs_mean.copy_(train[0].mean(0))
        model.obs_std.copy_(train[0].std(0, unbiased=False).clamp_min(.05))
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    weights = phase_weights(train)
    valid_weights = phase_weights(validation)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "bc_config.json", dict(schema=getattr(model, "schema", SCHEMA), dataset=inventory,
        epochs=args.epochs, learning_rate=args.lr, batch=args.batch,
        train_samples=len(train[0]), validation_samples=len(validation[0]),
        init=str(args.init) if args.init else None, **checkpoint_extra))
    for epoch in range(1, args.epochs+1):
        indices = torch.multinomial(weights, len(train[0]), replacement=True)
        losses = []
        for batch in indices.split(args.batch):
            loss = imitation_loss(model, train, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            pred = model.distribution(validation[0]).mean.tanh()
            error = ((pred-validation[1]).square()*validation[2]).sum(-1)/validation[2].sum(-1).clamp_min(1)
            val_loss = float((error*valid_weights).sum()/valid_weights.sum())
        log(args.out / "bc_metrics.jsonl", dict(epoch=epoch, train_mse=float(np.mean(losses)), validation_mse=val_loss))
        if val_loss < best:
            best = val_loss
            save_checkpoint(args.out / "bc_best.pt", model, kind="bc", epoch=epoch,
                validation_mse=best, data=str(args.data.resolve()), dataset=inventory, source_hash=fingerprint(), **checkpoint_extra)
        save_checkpoint(args.out / "bc_latest.pt", model, kind="bc", epoch=epoch,
            validation_mse=val_loss, source_hash=fingerprint(), **checkpoint_extra)


def evaluate(model, cfg, seeds):
    env = TeacherEnv(cfg)
    rows = []
    for seed in seeds:
        _, summary = rollout(env, seed, model)
        rows.append(summary)
        emit(dict(evaluation_seed=seed, hit15=summary["hit15"], landing_error_m=summary["landing_error_m"]))
    return dict(episodes=len(rows), captured_rate=float(np.mean([r["captured"] for r in rows])),
        release_rate=float(np.mean([r["released"] for r in rows])),
        hit15_rate=float(np.mean([r["hit15"] for r in rows])),
        hit30_rate=float(np.mean([r["hit30"] for r in rows])),
        mean_weld_peak_n=float(np.mean([r["weld_peak_n"] for r in rows])),
        mean_contact_peak_n=float(np.mean([r["contact_peak_n"] for r in rows])), records=rows)


def score(summary):
    return (summary["hit15_rate"], summary["hit30_rate"], summary["release_rate"],
        summary["captured_rate"], -summary["mean_weld_peak_n"])


def ppo_update(model, anchor, optimizer, buffer, config, bc_data, bc_weight, kl_weight):
    n = buffer.size
    obs = torch.tensor(buffer.observations[:n])
    raw = torch.tensor(buffer.raw_actions[:n])
    masks = torch.tensor(buffer.action_masks[:n])
    old = torch.tensor(buffer.log_probs[:n])
    ret = torch.tensor(buffer.returns[:n])
    adv = torch.tensor(buffer.advantages[:n])
    adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    bc_weights = phase_weights(bc_data)
    metrics = []
    for epoch in range(config.update_epochs):
        for ix in torch.randperm(n).split(config.minibatch_size):
            lp, entropy, value = model.evaluate(obs[ix], raw[ix], masks[ix])
            lr = lp-old[ix]
            ratio = lr.exp()
            policy_loss = -torch.minimum(ratio*adv[ix], ratio.clamp(1-config.clip_ratio,1+config.clip_ratio)*adv[ix]).mean()
            value_loss = .5*(value-ret[ix]).square().mean()
            with torch.no_grad():
                reference = anchor.distribution(obs[ix])
            anchor_kl = (torch.distributions.kl_divergence(reference, model.distribution(obs[ix]))*masks[ix]).sum(-1).mean()
            bi = torch.multinomial(bc_weights, len(ix), replacement=True)
            bc = imitation_loss(model, bc_data, bi)
            loss = policy_loss + config.value_coefficient*value_loss + bc_weight*bc + kl_weight*anchor_kl
            loss -= config.entropy_coefficient*entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            approx_kl = float(((ratio-1)-lr).mean().detach())
            metrics.append([float(policy_loss.detach()), float(value_loss.detach()), float(bc.detach()),
                            float(anchor_kl.detach()), approx_kl])
            if approx_kl > config.target_kl:
                break
        if approx_kl > config.target_kl:
            break
    return dict(zip(("policy_loss", "value_loss", "bc_loss", "anchor_kl", "approx_kl"),
                    np.mean(metrics, axis=0).tolist()), epochs=epoch+1)


def train(args):
    cfg = task(args)
    dataset = load_dataset(args.data)
    inventory = dataset_inventory(args.data)
    bc_data = tensor_data(dataset["train"])
    model, payload = load_checkpoint(args.resume or args.init)
    anchor, _ = load_checkpoint(args.anchor or args.init)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    config = PPOConfig(gamma=.999, gae_lambda=.98, learning_rate=args.lr,
        clip_ratio=.10, entropy_coefficient=.0001, target_kl=.01, update_epochs=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start, next_seed, best = 0, args.seed, None
    if (args.out / "ppo_latest.pt").exists() and not args.resume:
        raise ValueError("output already contains PPO checkpoints; use --resume or a new --out")
    anchor_hash = __import__("hashlib").sha256((args.anchor or args.init).read_bytes()).hexdigest()
    if args.resume:
        if payload.get("kind") != "ppo" or payload.get("task_config") != asdict(cfg):
            raise ValueError("resume requires PPO checkpoint with identical task config; use --init for a new phase")
        if payload.get("source_hash") != fingerprint() or payload.get("anchor_hash") != anchor_hash:
            raise ValueError("resume source/anchor differs from checkpoint")
        if "dataset" in payload and payload["dataset"]["manifest_sha256"] != inventory["manifest_sha256"]:
            raise ValueError("resume dataset changed; use --init and a new --out after DAgger")
        if "eval_seeds" in payload and payload["eval_seeds"] != list(range(args.eval_seed, args.eval_seed+args.eval_episodes)):
            raise ValueError("resume evaluation seeds changed")
        if payload.get("ppo_config") != asdict(config):
            raise ValueError("resume PPO parameters changed; use --init and a new --out")
        for key in ("bc_weight", "kl_weight", "episodes_per_update"):
            if key in payload and payload[key] != getattr(args, key):
                raise ValueError(f"resume {key} changed; use --init and a new --out")
        optimizer.load_state_dict(payload["optimizer"])
        start, next_seed, best = payload["update"], payload["next_seed"], payload["best_score"]
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    env = TeacherEnv(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(range(args.eval_seed, args.eval_seed+args.eval_episodes))
    train_seeds = set(range(next_seed, next_seed+max(0,args.updates-start)*args.episodes_per_update))
    data_seeds = {json.loads(p.read_text(encoding="utf-8"))["seed"] for p in episode_manifests(args.data)}
    if set(eval_seeds) & (train_seeds | data_seeds):
        raise ValueError("evaluation seeds overlap training/demo seeds")
    def save(update, path):
        save_checkpoint(path, model, kind="ppo", optimizer=optimizer.state_dict(),
            update=update, next_seed=next_seed, best_score=best, task_config=asdict(cfg),
            ppo_config=asdict(config), source_hash=fingerprint(), anchor_hash=anchor_hash,
            dataset=inventory, eval_seeds=eval_seeds, bc_weight=args.bc_weight, kl_weight=args.kl_weight,
            episodes_per_update=args.episodes_per_update,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
    write_json(args.out / "run_config.json", dict(schema=SCHEMA, dataset=inventory,
        task=asdict(cfg), ppo=asdict(config), bc_weight=args.bc_weight, kl_weight=args.kl_weight,
        init=str(args.init), anchor=str(args.anchor or args.init), eval_seeds=eval_seeds,
        episodes_per_update=args.episodes_per_update, target_updates=args.updates))
    if best is None:
        baseline = evaluate(model, cfg, eval_seeds)
        best = score(baseline)
        write_json(args.out / "eval_0000.json", baseline)
        save(start, args.out / "ppo_best.pt")
    save(start, args.out / "ppo_latest.pt")
    try:
        for update in range(start+1, args.updates+1):
            # Complete episodes preserve long catch-to-throw credit and simplify safe resume.
            buffer = RolloutBuffer(args.episodes_per_update*1200, OBS_SIZE, 10)
            summaries = []
            for _ in range(args.episodes_per_update):
                obs = env.reset(next_seed)
                done = False
                while not done:
                    mask = env.mask()
                    with torch.no_grad():
                        a, raw, lp, val = model.act(torch.tensor(obs[None]), action_mask=torch.tensor(mask))
                    following, reward, done, summary = env.step(a[0].numpy())
                    buffer.add(obs, raw[0].numpy(), lp.item(), reward, done, val.item(), mask)
                    obs = following
                summaries.append(summary)
                emit(dict(training_seed=next_seed, hit15=summary["hit15"], captured=summary["captured"]))
                next_seed += 1
            buffer.finish(0., config)
            metrics = ppo_update(model, anchor, optimizer, buffer, config, bc_data, args.bc_weight, args.kl_weight)
            log(args.out / "ppo_metrics.jsonl", dict(update=update, samples=buffer.size,
                **metrics, episodes=summaries))
            if update % args.eval_every == 0 or update == args.updates:
                result = evaluate(model, cfg, eval_seeds)
                write_json(args.out / f"eval_{update:04d}.json", result)
                if score(result) > tuple(best):
                    best = score(result)
                    save(update, args.out / "ppo_best.pt")
            save(update, args.out / "ppo_latest.pt")
    except KeyboardInterrupt:
        emit(dict(interrupted=True, message="Use ppo_latest.pt; partial update is discarded on resume."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "dagger"):
        p = commands.add_parser(name)
        p.add_argument("--episodes", type=int, default=96)
        p.add_argument("--seed", type=int, default=401)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--out", type=Path, default=Path("teacher_runs/dataset_v1"))
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--beta", type=float, default=.5 if name == "dagger" else 0.)
        p.add_argument("--perturb", type=float, default=0.)
        p.add_argument("--authority", type=float, default=.25)
    p = commands.add_parser("bc")
    p.add_argument("--data", type=Path, default=Path("teacher_runs/dataset_v1"))
    p.add_argument("--out", type=Path, default=Path("teacher_runs/bc"))
    p.add_argument("--init", type=Path)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=90001)
    p.add_argument("--authority", type=float, default=.25)
    p.add_argument("--out", type=Path, default=Path("teacher_runs/evaluation.json"))
    p = commands.add_parser("ppo")
    p.add_argument("--data", type=Path, default=Path("teacher_runs/dataset_v1"))
    p.add_argument("--init", type=Path, default=Path("teacher_runs/bc/bc_best.pt"))
    p.add_argument("--anchor", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--out", type=Path, default=Path("teacher_runs/ppo"))
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=2)
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--eval-seed", type=int, default=90001)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--authority", type=float, default=.25)
    p.add_argument("--quality-weight", type=float, default=0.)
    p.add_argument("--bc-weight", type=float, default=1.)
    p.add_argument("--kl-weight", type=float, default=.05)
    p.add_argument("--lr", type=float, default=3e-5)
    for name in ("collect", "dagger", "evaluate", "ppo"):
        task_parser = commands.choices[name]
        defaults = TaskConfig()
        for key in ("target_distance", "orbit_radius", "target_azimuth", "launch_distance", "launch_speed", "mass", "ball_radius"):
            task_parser.add_argument("--"+key.replace("_", "-"), type=float, default=getattr(defaults, key))
    args = parser.parse_args()
    for key in ("episodes", "workers", "epochs", "batch", "updates", "episodes_per_update", "eval_episodes", "eval_every"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if hasattr(args,"beta") and not 0 <= args.beta <= 1:
        parser.error("beta must be in [0,1]")
    if hasattr(args,"perturb") and args.perturb < 0:
        parser.error("perturb must be nonnegative")
    for key in ("lr", "target_distance", "orbit_radius", "launch_distance", "launch_speed", "mass", "ball_radius"):
        if hasattr(args, key) and (not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            parser.error(f"{key} must be finite and positive")
    torch.set_num_threads(1)
    torch.manual_seed(20260917)
    np.random.seed(20260917)
    random.seed(20260917)
    if args.command in ("collect", "dagger"):
        collection(args)
    elif args.command == "bc":
        clone(args)
    elif args.command == "ppo":
        train(args)
    else:
        policy = load_checkpoint(args.checkpoint)[0] if args.checkpoint else None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, evaluate(policy, task(args), range(args.seed, args.seed+args.episodes)))


if __name__ == "__main__":
    main()
