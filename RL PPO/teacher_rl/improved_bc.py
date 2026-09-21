"""Formal V15 behavior cloning and paired closed-loop evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .__main__ import emit
from .data import (dataset_inventory, episode_manifests, load_dataset, rollout,
    write_json)
from .env import PHASES
from .generalized_teacher import (_metrics, _parameter_bins,
    make_scenario_manifest)
from .improved_teacher import (IMPROVED_SCHEMA, ImprovedRecipe,
    ImprovedTeacherEnv, improved_fingerprint)
from .model import IntentPolicy, load_checkpoint, save_checkpoint
from .reachable_collection import (collector_fingerprint, constrain_manifest,
    _grasp_height)


TRAINING_SEED = 20260927
DEFAULT_DATA = Path("teacher_runs/v15_improved/dataset_1000_formal")
DEFAULT_OUT = Path("teacher_runs/v15_improved/bc_v1_formal")


def trainer_fingerprint():
    digest = hashlib.sha256(improved_fingerprint().encode())
    for path in (Path(__file__), Path(__file__).with_name("data.py"),
            Path(__file__).with_name("model.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_hash(manifest):
    payload = dict(manifest)
    claimed = payload.pop("manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return claimed, hashlib.sha256(encoded).hexdigest()


def _selection_seeds(contract, data_dir):
    selection_path = Path(contract.get("selection_file", ""))
    candidates = [selection_path, data_dir / selection_path,
        data_dir.parent.parent.parent / selection_path]
    selection = next((path for path in candidates if path.is_file()), None)
    if selection is None:
        return []
    payload = _read_json(selection)
    if (payload.get("schema") != IMPROVED_SCHEMA or
            payload.get("source_hash") != contract["selection_source_hash"] or
            payload.get("recipe") != contract["recipe"] or
            not payload.get("improved", False)):
        raise ValueError(f"teacher selection does not match dataset: {selection}")
    seeds = []
    for key in ("screen_manifest", "validation_manifest"):
        seeds.extend(row["seed"] for row in payload.get(key, {}).get("scenarios", []))
    return sorted(set(seeds))


def validate_dataset(data_dir):
    """Validate that every completed trajectory belongs to one V15 teacher contract."""
    data_dir = Path(data_dir)
    contract_path = data_dir / "dataset_contract.json"
    manifest_path = data_dir / "scenario_manifest.json"
    if not contract_path.is_file() or not manifest_path.is_file():
        raise ValueError("V15 dataset requires dataset_contract.json and scenario_manifest.json")
    contract, manifest = _read_json(contract_path), _read_json(manifest_path)
    source_hash = improved_fingerprint()
    if contract.get("schema") != IMPROVED_SCHEMA:
        raise ValueError("dataset is not can_generalized_teacher_v15")
    if contract.get("source_hash") != source_hash:
        raise ValueError("V15 teacher source changed; do not train against an unverified source")
    if (contract.get("collector_source_hash") is not None and
            contract["collector_source_hash"] != collector_fingerprint()):
        raise ValueError("reachable collector source changed since this dataset was collected")
    recipe = ImprovedRecipe(**contract["recipe"])
    claimed, actual = _manifest_hash(manifest)
    if claimed != actual or claimed != contract.get("manifest_sha256"):
        raise ValueError("scenario manifest hash does not match the dataset contract")
    scenarios = manifest.get("scenarios", [])
    if len(scenarios) != manifest.get("episodes") or not scenarios:
        raise ValueError("scenario manifest is incomplete")
    expected = {row["seed"]: row for row in scenarios}
    if len(expected) != len(scenarios):
        raise ValueError("scenario manifest contains duplicate seeds")

    paths = episode_manifests(data_dir)
    if len(paths) != len(scenarios):
        raise ValueError(f"dataset has {len(paths)} completed episodes; expected {len(scenarios)}")
    observed = set()
    for path in paths:
        row = _read_json(path)
        seed = row.get("seed")
        if seed in observed or seed not in expected:
            raise ValueError(f"duplicate or unexpected episode seed: {path}")
        observed.add(seed)
        if (row.get("schema") != IMPROVED_SCHEMA or
                row.get("source_hash") != source_hash or
                row.get("name") != "selected" or
                row.get("recipe") != asdict(recipe) or
                row.get("scenario") != expected[seed] or
                row.get("split") != expected[seed]["split"] or
                row.get("batched_actuator") is not True):
            raise ValueError(f"episode does not match the V15 teacher contract: {path}")
        if (row.get("checkpoint_sha256") is not None or
                row.get("beta") is not None or row.get("perturb") is not None):
            raise ValueError(f"student/DAgger trajectory found in teacher dataset: {path}")
        data_name = row.get("data", "")
        if Path(data_name).name != data_name or not (data_dir / data_name).is_file():
            raise ValueError(f"episode data file is missing or invalid: {path}")
        if not isinstance(row.get("samples"), int) or row["samples"] <= 0:
            raise ValueError(f"episode sample count is invalid: {path}")
    if observed != set(expected):
        raise ValueError("dataset does not cover the complete scenario manifest")

    return dict(contract=contract, manifest=manifest, recipe=recipe,
        inventory=dataset_inventory(data_dir), demonstration_seeds=sorted(observed),
        selection_seeds=_selection_seeds(contract, data_dir))


def tensor_data(data):
    return tuple(torch.as_tensor(value) for value in data)


def balanced_quality_weights(data):
    """Give each active phase equal mass, then prefer higher-quality trajectories."""
    _, _, masks, phases, quality = data
    weights = torch.zeros_like(quality, dtype=torch.float32)
    active = masks.sum(-1) > 0
    for phase in torch.unique(phases[active]):
        selected = active & (phases == phase)
        weights[selected] = quality[selected] / selected.sum()
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("training data has no finite active-action weights")
    return weights


def masked_mse(model, data, indices):
    observations, targets, masks, _, _ = data
    prediction = model.distribution(observations[indices]).mean.tanh()
    active = masks[indices].sum(-1).clamp_min(1.)
    return (((prediction - targets[indices]).square() * masks[indices]).sum(-1) /
        active).mean()


def validation_metrics(model, data):
    observations, targets, masks, phases, _ = data
    weights = balanced_quality_weights(data)
    prediction = model.distribution(observations).mean.tanh()
    active_count = masks.sum(-1)
    per_sample = ((prediction - targets).square() * masks).sum(-1) / active_count.clamp_min(1.)
    active = active_count > 0
    result = {
        "weighted_mse": float((per_sample * weights).sum() / weights.sum()),
        "unweighted_mse": float(per_sample[active].mean()),
        "active_action_rmse": float(torch.sqrt(
            ((prediction - targets).square() * masks).sum() / masks.sum().clamp_min(1.))),
    }
    per_phase = {}
    for index, name in enumerate(PHASES):
        selected = active & (phases == index)
        per_phase[name] = (float(per_sample[selected].mean())
            if selected.any() else None)
    result["phase_mse"] = per_phase
    return result


def _append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
    emit(value)


def train(args):
    validated = validate_dataset(args.data)
    output_files = ("bc_config.json", "bc_metrics.jsonl", "bc_best.pt",
        "bc_latest.pt", "bc_summary.json")
    if any((args.out / name).exists() for name in output_files):
        raise ValueError("BC output already exists; choose a new --out directory")

    dataset = load_dataset(args.data, expected_schema=IMPROVED_SCHEMA,
        source_hash=improved_fingerprint())
    train_data, validation_data = (tensor_data(dataset[split])
        for split in ("train", "validation"))
    model = IntentPolicy()
    model.schema = IMPROVED_SCHEMA
    with torch.no_grad():
        model.obs_mean.copy_(train_data[0].mean(0))
        model.obs_std.copy_(train_data[0].std(0, unbiased=False).clamp_min(.05))
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    weights = balanced_quality_weights(train_data)
    training_config = dict(epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.lr, optimizer="Adam", gradient_clip_norm=1.0,
        seed=args.seed, torch_threads=args.threads,
        sampling="active-phase-balanced, trajectory-quality-weighted",
        model_selection="minimum validation weighted masked MSE")
    checkpoint_common = dict(kind="v15_bc", source_hash=improved_fingerprint(),
        trainer_source_hash=trainer_fingerprint(), recipe=asdict(validated["recipe"]),
        dataset_contract=validated["contract"], dataset=validated["inventory"],
        demonstration_seeds=validated["demonstration_seeds"],
        selection_seeds=validated["selection_seeds"], training_config=training_config,
        data=str(args.data.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "bc_config.json", dict(schema=IMPROVED_SCHEMA,
        source_hash=improved_fingerprint(), trainer_source_hash=trainer_fingerprint(),
        data=str(args.data.resolve()), output=str(args.out.resolve()),
        dataset=validated["inventory"], dataset_contract=validated["contract"],
        raw_train_episodes=validated["manifest"]["train_episodes"],
        raw_validation_episodes=validated["manifest"]["validation_episodes"],
        retained_train_samples=len(train_data[0]),
        retained_validation_samples=len(validation_data[0]),
        actor_parameters=sum(parameter.numel() for parameter in model.actor.parameters()),
        actor_observations=82, actor_actions=10, training=training_config))

    best_mse, best_epoch = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        indices = torch.multinomial(weights, len(train_data[0]), replacement=True)
        loss_sum, sample_count = 0., 0
        model.train()
        for batch in indices.split(args.batch):
            loss = masked_mse(model, train_data, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            sample_count += len(batch)
        model.eval()
        with torch.no_grad():
            metrics = validation_metrics(model, validation_data)
        row = dict(epoch=epoch, train_mse=loss_sum / sample_count,
            validation_mse=metrics["weighted_mse"],
            validation_unweighted_mse=metrics["unweighted_mse"],
            validation_active_action_rmse=metrics["active_action_rmse"],
            validation_phase_mse=metrics["phase_mse"],
            elapsed_wall_s=time.perf_counter() - started)
        _append_jsonl(args.out / "bc_metrics.jsonl", row)
        if metrics["weighted_mse"] < best_mse:
            best_mse, best_epoch = metrics["weighted_mse"], epoch
            save_checkpoint(args.out / "bc_best.pt", model, epoch=epoch,
                validation_metrics=metrics, **checkpoint_common)
        save_checkpoint(args.out / "bc_latest.pt", model, epoch=epoch,
            validation_metrics=metrics, **checkpoint_common)

    summary = dict(schema=IMPROVED_SCHEMA, best_epoch=best_epoch,
        best_validation_mse=best_mse, epochs=args.epochs,
        elapsed_wall_s=time.perf_counter() - started,
        best_checkpoint=str((args.out / "bc_best.pt").resolve()),
        latest_checkpoint=str((args.out / "bc_latest.pt").resolve()))
    write_json(args.out / "bc_summary.json", summary)
    emit(summary)


_EVAL_MODEL = None
_EVAL_RECIPE = None


def _init_eval_worker(checkpoint, recipe_data):
    global _EVAL_MODEL, _EVAL_RECIPE
    torch.set_num_threads(1)
    _EVAL_MODEL, _ = load_checkpoint(checkpoint, expected_schema=IMPROVED_SCHEMA)
    _EVAL_MODEL.eval()
    _EVAL_RECIPE = ImprovedRecipe(**recipe_data)


def _paired_episode(job):
    scenario, output, checkpoint_hash, source_hash = job
    output = Path(output)
    identity = dict(schema=IMPROVED_SCHEMA, source_hash=source_hash,
        checkpoint_sha256=checkpoint_hash, recipe=asdict(_EVAL_RECIPE),
        scenario=scenario)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    path = output / f"episode_{scenario['seed']}_{key}.json"
    if path.exists():
        return _read_json(path)
    output.mkdir(parents=True, exist_ok=True)
    results, walls = {}, {}
    order = ("teacher", "bc") if scenario["seed"] % 2 else ("bc", "teacher")
    for name in order:
        started = time.perf_counter()
        policy = None if name == "teacher" else _EVAL_MODEL
        _, results[name] = rollout(ImprovedTeacherEnv(scenario, _EVAL_RECIPE),
            scenario["seed"], policy)
        walls[name] = time.perf_counter() - started
    row = dict(**identity, teacher=results["teacher"], bc=results["bc"],
        teacher_wall_s=walls["teacher"], bc_wall_s=walls["bc"])
    write_json(path, row)
    return row


def acceptance(teacher, bc, worst_bin):
    checks = {
        "capture_drop_le_3pp": bc["captured_rate"] >= teacher["captured_rate"] - .03,
        "hit15_drop_le_5pp": bc["hit15_rate"] >= teacher["hit15_rate"] - .05,
        "mean_landing_error_le_7cm": (bc["mean_landing_error_m"] is not None and
            bc["mean_landing_error_m"] <= .07),
        "worst_parameter_bin_hit15_ge_80pct": worst_bin >= .80,
    }
    return dict(passed=all(checks.values()), checks=checks,
        thresholds=dict(max_capture_drop_rate=.03, max_hit15_drop_rate=.05,
            max_mean_landing_error_m=.07, min_worst_parameter_bin_hit15_rate=.80))


def evaluate(args):
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_checkpoint(checkpoint_path,
        expected_schema=IMPROVED_SCHEMA)
    del model
    if (checkpoint.get("kind") != "v15_bc" or
            checkpoint.get("source_hash") != improved_fingerprint()):
        raise ValueError("checkpoint is not a BC model for the current V15 teacher")
    recipe = ImprovedRecipe(**checkpoint["recipe"])
    if checkpoint.get("dataset_contract", {}).get("recipe") != asdict(recipe):
        raise ValueError("checkpoint recipe and dataset contract disagree")

    base_manifest = make_scenario_manifest(args.episodes, args.seed, args.design_seed)
    manifest = constrain_manifest(base_manifest,
        _grasp_height(recipe, base_manifest["scenarios"][0]))
    evaluation_seeds = {row["seed"] for row in manifest["scenarios"]}
    used_seeds = set(checkpoint.get("demonstration_seeds", []))
    used_seeds.update(checkpoint.get("selection_seeds", []))
    if evaluation_seeds & used_seeds:
        raise ValueError("evaluation scenarios overlap teacher search or BC demonstrations")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    evaluation_contract = dict(schema=IMPROVED_SCHEMA,
        source_hash=improved_fingerprint(), checkpoint_sha256=checkpoint_hash,
        recipe=asdict(recipe), scenario_manifest=manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "evaluation_contract.json"
    if contract_path.exists() and _read_json(contract_path) != evaluation_contract:
        raise ValueError("evaluation output has a different contract; choose a new --out")
    write_json(contract_path, evaluation_contract)

    jobs = [(scenario, str(args.out / "episodes"), checkpoint_hash,
        improved_fingerprint()) for scenario in manifest["scenarios"]]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_eval_worker,
            initargs=(str(checkpoint_path.resolve()), asdict(recipe))) as pool:
        futures = [pool.submit(_paired_episode, job) for job in jobs]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                emit(dict(completed=completed, total=len(jobs),
                    seed=row["scenario"]["seed"],
                    teacher_hit15=row["teacher"]["hit15"], bc_hit15=row["bc"]["hit15"]))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda row: row["scenario"]["scenario_id"])
    teacher_rows = [dict(scenario=row["scenario"], result=row["teacher"],
        wall_s=row["teacher_wall_s"]) for row in rows]
    bc_rows = [dict(scenario=row["scenario"], result=row["bc"],
        wall_s=row["bc_wall_s"]) for row in rows]
    teacher_summary, bc_summary = _metrics(teacher_rows), _metrics(bc_rows)
    parameter_bins, worst_bin = _parameter_bins(bc_rows)
    report = dict(schema=IMPROVED_SCHEMA, source_hash=improved_fingerprint(),
        checkpoint_sha256=checkpoint_hash, episodes=len(rows), teacher=teacher_summary,
        bc=bc_summary, delta_bc_minus_teacher=dict(
            captured_rate=bc_summary["captured_rate"] - teacher_summary["captured_rate"],
            released_rate=bc_summary["released_rate"] - teacher_summary["released_rate"],
            hit15_rate=bc_summary["hit15_rate"] - teacher_summary["hit15_rate"],
            hit30_rate=bc_summary["hit30_rate"] - teacher_summary["hit30_rate"],
            mean_landing_error_m=(bc_summary["mean_landing_error_m"] -
                teacher_summary["mean_landing_error_m"]
                if bc_summary["mean_landing_error_m"] is not None and
                teacher_summary["mean_landing_error_m"] is not None else None)),
        paired_disagreements=dict(
            teacher_capture_bc_miss=sum(row["teacher"]["captured"] and
                not row["bc"]["captured"] for row in rows),
            teacher_hit15_bc_miss=sum(row["teacher"]["hit15"] and
                not row["bc"]["hit15"] for row in rows),
            teacher_miss_bc_hit15=sum(not row["teacher"]["hit15"] and
                row["bc"]["hit15"] for row in rows)),
        bc_parameter_bins=parameter_bins,
        bc_worst_parameter_bin_hit15_rate=worst_bin,
        acceptance=acceptance(teacher_summary, bc_summary, worst_bin))
    write_json(args.out / "comparison.json", report)
    emit({key: report[key] for key in ("teacher", "bc",
        "delta_bc_minus_teacher", "bc_worst_parameter_bin_hit15_rate", "acceptance")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--data", type=Path, default=DEFAULT_DATA)
    command.add_argument("--out", type=Path, default=DEFAULT_OUT)
    command.add_argument("--epochs", type=int, default=60)
    command.add_argument("--batch", type=int, default=256)
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--seed", type=int, default=TRAINING_SEED)
    command.add_argument("--threads", type=int, default=4)
    command = sub.add_parser("evaluate")
    command.add_argument("--checkpoint", type=Path,
        default=DEFAULT_OUT / "bc_best.pt")
    command.add_argument("--episodes", type=int, default=200)
    command.add_argument("--seed", type=int, default=92001)
    command.add_argument("--design-seed", type=int, default=20260928)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--out", type=Path, default=DEFAULT_OUT / "evaluation_200")
    args = parser.parse_args()
    for name in ("epochs", "batch", "threads", "episodes", "workers"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if hasattr(args, "lr") and (not np.isfinite(args.lr) or args.lr <= 0):
        parser.error("lr must be finite and positive")
    if args.command == "evaluate" and args.episodes < 20:
        parser.error("formal evaluation requires at least 20 episodes")
    torch.set_num_threads(args.threads if args.command == "train" else 1)
    torch.manual_seed(args.seed if args.command == "train" else TRAINING_SEED)
    np.random.seed(args.seed if args.command == "train" else TRAINING_SEED)
    random.seed(args.seed if args.command == "train" else TRAINING_SEED)
    {"train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
