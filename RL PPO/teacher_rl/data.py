"""Atomic per-episode datasets; validation is separated by launch seed."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .env import TeacherEnv, TaskConfig, SCHEMA, SOURCE, ROOT, PHASES


def fingerprint():
    paths = sorted((SOURCE / "umarm_can").glob("*.py"))
    paths += sorted((ROOT / "digital_twin").glob("*.py"))
    paths += sorted((ROOT / "digital_twin" / "checkpoints").glob("*"))
    paths += sorted((ROOT / "UMArm_KINEMATICS").glob("*.py"))
    paths += sorted((ROOT / "data").rglob("*.json"))
    paths += sorted((SOURCE / "assets" / "gripper_mk5").rglob("*"))
    paths += [SOURCE / "umarm_mk5" / "launches_throw.py", Path(__file__).with_name("env.py")]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def episode_manifests(directory):
    return sorted(p for p in Path(directory).glob("episode_*.json") if not p.name.endswith(".tmp.json"))


def dataset_inventory(directory):
    paths = episode_manifests(directory)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return dict(episodes=len(paths), manifest_sha256=digest.hexdigest(),
                manifests=[path.name for path in paths])


def rollout(env, seed, policy=None, beta=0., perturb=0.):
    import torch
    rng = np.random.default_rng(seed + 12345)
    observation = env.reset(seed)
    rows = {k: [] for k in ("observations", "teacher_actions", "actions", "masks", "rewards", "phases", "dones")}
    references, applied, estimates = [], [], []
    done = False
    while not done:
        teacher, mask, phase = env.teacher_action(), env.mask(), env.phase()
        t = env.obs["t_win"]
        plan = env.catch.plan
        references.append(np.r_[plan.q(t), plan.qd(t), plan.qdd(t)])
        bh = env.obs["ball_hat"]
        estimates.append(np.r_[bh["pos"], bh["vel"]] if bh["visible"] else np.zeros(6))
        if policy is None or rng.random() < beta:
            action = teacher.copy()
        else:
            with torch.no_grad():
                action = policy.act(torch.tensor(observation[None]), deterministic=True)[0][0].numpy()
        if perturb:
            action = np.clip(action + rng.normal(0., perturb, 10)*mask, -1., 1.)
        next_obs, reward, done, summary = env.step(action)
        applied.append(env.last_intent.copy())
        for key, value in zip(rows, (observation, teacher, action, mask, reward, phase, done)):
            rows[key].append(value)
        observation = next_obs
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in rows.items()}
    arrays["trace150"] = np.asarray(env.trace, dtype=np.float32)
    arrays["reference_q_qd_qdd"] = np.asarray(references, dtype=np.float32)
    arrays["ball_estimate"] = np.asarray(estimates, dtype=np.float32)
    arrays["applied_actions"] = np.asarray(applied, dtype=np.float32)
    return arrays, summary


def collect_one(job):
    import torch
    torch.set_num_threads(1)
    seed, config, out, checkpoint, beta, perturb = job
    config = TaskConfig(**config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = fingerprint()
    identity = dict(schema=SCHEMA, seed=seed, config=asdict(config), source_hash=source_hash,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        beta=beta, perturb=perturb)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    base = out / f"episode_{seed}_{key}"
    meta_path, data_path = base.with_suffix(".json"), base.with_suffix(".npz")
    if meta_path.exists() and data_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    policy = None
    if checkpoint:
        from .model import load_checkpoint
        policy, _ = load_checkpoint(checkpoint)
        policy.eval()
    started = time.perf_counter()
    env = TeacherEnv(config)
    arrays, summary = rollout(env, seed, policy, beta, perturb)
    # NPZ and its completion manifest are committed only after a complete episode.
    temp = base.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(data_path)
    metadata = dict(**identity, result=summary, samples=len(arrays["observations"]),
        split="validation" if seed % 5 == 0 else "train", data=data_path.name,
        wall_s=time.perf_counter()-started, events=env.plant.log_events,
        twin_provenance=env.kw.provenance)
    write_json(meta_path, metadata)
    return metadata


def load_dataset(directory, *, expected_schema=SCHEMA, source_hash=None):
    splits = {"train": [], "validation": []}
    source_hash = fingerprint() if source_hash is None else source_hash
    for path in episode_manifests(directory):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["schema"] != expected_schema or meta["source_hash"] != source_hash:
            raise ValueError(f"dataset source/schema mismatch: {path}; use a separate dataset directory")
        with np.load(path.with_name(meta["data"]), allow_pickle=False) as data:
            phase = data["phases"].astype(int)
            # Keep event-rich approach/release; downsample the long settling/orbit periods.
            keep = np.ones(len(phase), dtype=bool)
            for p in (2, 3, 5):
                indices = np.flatnonzero(phase == p)
                keep[indices] = False
                keep[indices[::5]] = True
            quality = 1. if meta["result"]["hit15"] else .5 if meta["result"]["captured"] else .25
            # Constraint force is a distinct metric, not finger contact force.
            quality /= 1 + max(0., meta["result"]["weld_peak_n"]-400)/800
            splits[meta["split"]].append((data["observations"][keep], data["teacher_actions"][keep],
                data["masks"][keep], phase[keep], np.full(keep.sum(), quality, dtype=np.float32)))
    result = {}
    for split, episodes in splits.items():
        if not episodes:
            raise ValueError(f"no {split} episodes; collect at least five consecutive seeds")
        result[split] = tuple(np.concatenate([ep[i] for ep in episodes]) for i in range(5))
    return result
