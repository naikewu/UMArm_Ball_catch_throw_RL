"""Resumable V15 collection with a nominal ballistic-reachability constraint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .__main__ import emit
from .data import write_json
from .generalized_teacher import (_metrics, _parameter_bins, make_scenario_manifest)
from .improved_teacher import (IMPROVED_SCHEMA, collect_batch, improved_fingerprint,
    load_selected)


G = 9.81
LAUNCH_HEIGHT_M = .5


def _reachable(speed, distance, dz):
    speed2 = float(speed) ** 2
    k = G * float(distance) ** 2 / (2. * speed2)
    return float(distance) ** 2 - 4. * k * (float(dz) + k) >= 0.


def constrain_manifest(manifest, grasp_height_m):
    """Clamp only unreachable distances; preserve every valid LHS scenario verbatim."""
    result = json.loads(json.dumps(manifest))
    result.pop("manifest_sha256", None)
    dz = float(grasp_height_m) - LAUNCH_HEIGHT_M
    adjustments = []
    for scenario in result["scenarios"]:
        speed = float(scenario["launch_speed"])
        old_distance = float(scenario["launch_distance"])
        if _reachable(speed, old_distance, dz):
            continue
        low, high = 1.7, old_distance
        if not _reachable(speed, low, dz):
            raise ValueError(f"launch speed {speed} cannot reach the minimum distance")
        for _ in range(60):
            middle = .5 * (low + high)
            if _reachable(speed, middle, dz):
                low = middle
            else:
                high = middle
        new_distance = round(max(1.7, low - .03), 9)
        scenario["launch_distance"] = new_distance
        adjustments.append(dict(scenario_id=scenario["scenario_id"], seed=scenario["seed"],
            launch_speed=speed, original_launch_distance=old_distance,
            constrained_launch_distance=new_distance))
    result["design"] = "split_latin_hypercube_reachable_v1"
    result["reachability_constraint"] = dict(grasp_height_m=float(grasp_height_m),
        launch_height_m=LAUNCH_HEIGHT_M, safety_distance_margin_m=.03,
        adjusted_scenarios=adjustments)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def collector_fingerprint():
    digest = hashlib.sha256(improved_fingerprint().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _grasp_height(recipe, scenario):
    from .improved_teacher import ImprovedTeacherEnv
    probe = dict(scenario)
    probe["launch_speed"] = 5.5
    probe["launch_distance"] = 1.7
    env = ImprovedTeacherEnv(probe, recipe)
    return float(env.model.fk(np.zeros(12))[0][2])


def _check_existing_episodes(output, scenarios):
    expected = {row["scenario_id"]: row for row in scenarios}
    for path in output.glob("episode_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = row["scenario"]
        if scenario != expected.get(scenario["scenario_id"]):
            raise ValueError(f"existing episode scenario conflicts with constrained manifest: {path}")


def collect(args):
    recipe, selection = load_selected(args.recipe)
    base_manifest = json.loads(json.dumps(
        make_scenario_manifest(args.episodes, args.seed, args.design_seed)))
    grasp_height = _grasp_height(recipe, base_manifest["scenarios"][0])
    manifest = constrain_manifest(base_manifest, grasp_height)
    source_hash = improved_fingerprint()
    collector_hash = collector_fingerprint()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "scenario_manifest.json"
    contract_path = args.out / "dataset_contract.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing not in (base_manifest, manifest):
            raise ValueError(f"collection manifest changed: {manifest_path}")
    _check_existing_episodes(args.out, manifest["scenarios"])
    completed_before = len(list(args.out.glob("episode_*.json")))
    contract = dict(schema=IMPROVED_SCHEMA, source_hash=source_hash,
        collector_source_hash=collector_hash,
        manifest_sha256=manifest["manifest_sha256"], recipe=asdict(recipe),
        selection_file=str(args.recipe), selection_source_hash=selection["source_hash"],
        batched_actuator=True)
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable = (existing.get("schema") == IMPROVED_SCHEMA and
            existing.get("source_hash") == source_hash and
            existing.get("recipe") == asdict(recipe) and
            existing.get("selection_source_hash") == selection["source_hash"] and
            existing.get("manifest_sha256") in
                (base_manifest["manifest_sha256"], manifest["manifest_sha256"]))
        if not stable:
            raise ValueError(f"collection contract changed: {contract_path}")
    write_json(manifest_path, manifest)
    write_json(contract_path, contract)
    rows, elapsed = collect_batch(recipe, manifest["scenarios"], args.out,
        args.workers, source_hash)
    bins, worst_bin = _parameter_bins(rows)
    npz_bytes = sum(path.stat().st_size for path in args.out.glob("episode_*.npz"))
    simulated = len(rows) - completed_before
    report = dict(schema=IMPROVED_SCHEMA, source_hash=source_hash,
        collector_source_hash=collector_hash, manifest_sha256=manifest["manifest_sha256"],
        elapsed_resume_wall_s=elapsed, episodes_reused=completed_before,
        episodes_simulated_this_run=simulated,
        resumed_episodes_per_hour=simulated * 3600. / max(elapsed, 1e-9),
        dataset_size_mb=npz_bytes / (1024. ** 2), overall=_metrics(rows),
        train=_metrics([row for row in rows if row["split"] == "train"]),
        validation=_metrics([row for row in rows if row["split"] == "validation"]),
        parameter_bins=bins, worst_parameter_bin_hit15_rate=worst_bin,
        recipe=asdict(recipe), reachability_constraint=manifest["reachability_constraint"])
    write_json(args.out / "collection_summary.json", report)
    emit(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=90001)
    parser.add_argument("--design-seed", type=int, default=20260923)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path,
        default=Path("teacher_runs/v15_improved/dataset_1000"))
    args = parser.parse_args()
    if args.episodes < 5 or args.workers <= 0:
        parser.error("episodes must be at least five and workers must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(20260919)
    np.random.seed(20260919)
    collect(args)


if __name__ == "__main__":
    main()
