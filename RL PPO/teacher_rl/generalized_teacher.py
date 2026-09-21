"""V14 generalized teacher scenarios, batched-plant verification, and collection."""
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

from .__main__ import emit
from .data import rollout, write_json
from .env import TaskConfig, ThrowDistribution
from .soft_env import SoftRecipe, SoftTeacherEnv, soft_fingerprint


GENERALIZED_SCHEMA = "can_generalized_teacher_v14"
PARAMETER_RANGES = {
    "launch_speed": (4.5, 5.5),
    "launch_distance": (1.7, 2.3),
    "target_distance": (1.3, 1.7),
    "target_azimuth": (10.0, 30.0),
}
FIXED_SCENARIO = {
    "mass": 0.5,
    "ball_radius": 0.04,
    "orbit_radius": 0.62,
    "launch_angle_jitter_deg": 3.0,
    "launch_position_jitter_m": 0.04,
    "authority": 0.25,
}


def generalized_fingerprint():
    digest = hashlib.sha256(soft_fingerprint().encode())
    path = Path(__file__)
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _latin_hypercube(count, dimensions, rng):
    points = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        points[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return points


def _scenario_rows(count, split, first_seed, design_seed, first_index):
    rng = np.random.default_rng(design_seed)
    names = tuple(PARAMETER_RANGES)
    points = _latin_hypercube(count, len(names), rng)
    rows = []
    for row_index, point in enumerate(points):
        values = {}
        for name, coordinate in zip(names, point):
            low, high = PARAMETER_RANGES[name]
            values[name] = round(low + coordinate * (high - low), 9)
        rows.append(dict(scenario_id=first_index + row_index, seed=first_seed + row_index,
            split=split, **values, **FIXED_SCENARIO))
    return rows


def make_scenario_manifest(episodes=256, first_seed=60001, design_seed=20260918):
    if episodes < 5:
        raise ValueError("collect at least five episodes so validation is non-empty")
    validation_count = max(1, episodes // 5)
    train_count = episodes - validation_count
    train = _scenario_rows(train_count, "train", first_seed, design_seed, 0)
    validation = _scenario_rows(validation_count, "validation", first_seed + train_count,
        design_seed + 1, train_count)
    scenarios = train + validation
    payload = dict(schema=GENERALIZED_SCHEMA, design="split_latin_hypercube_v1",
        design_seed=design_seed, first_seed=first_seed, episodes=episodes,
        train_episodes=train_count, validation_episodes=validation_count,
        parameter_ranges=PARAMETER_RANGES, fixed=FIXED_SCENARIO, scenarios=scenarios)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _task_config(scenario):
    return TaskConfig(target_distance=scenario["target_distance"],
        orbit_radius=scenario["orbit_radius"], target_azimuth=scenario["target_azimuth"],
        launch_distance=scenario["launch_distance"], launch_speed=scenario["launch_speed"],
        mass=scenario["mass"], ball_radius=scenario["ball_radius"],
        authority=scenario["authority"], quality_weight=0.)


class GeneralizedTeacherEnv(SoftTeacherEnv):
    """Baseline V13 teacher under a V14 task scenario; actor dimensions stay unchanged."""

    def __init__(self, scenario, *, batched_actuator=True):
        self.scenario = dict(scenario)
        self.use_batched_actuator = bool(batched_actuator)
        config = _task_config(self.scenario)
        super().__init__(SoftRecipe(), config)
        centre, axis, _ = self.model.fk(np.zeros(12))
        self.distribution = ThrowDistribution(centre, axis,
            v0_nom=config.launch_speed, dist_m=config.launch_distance,
            v_jit=0., ang_jit_deg=self.scenario["launch_angle_jitter_deg"],
            pos_jit_m=self.scenario["launch_position_jitter_m"],
            ball_mass_kg=config.mass, ball_radius_m=config.ball_radius)

    def reset(self, seed):
        observation = super().reset(seed)
        # The batched path is algebraically equivalent and only changes how the
        # 24 pneumatic-node flow evaluations are scheduled.
        self.plant.batched_actuator = self.use_batched_actuator
        return observation


def _episode_identity(scenario, source_hash):
    return dict(schema=GENERALIZED_SCHEMA, source_hash=source_hash,
        seed=scenario["seed"], scenario=scenario, recipe=asdict(SoftRecipe()),
        batched_actuator=True, checkpoint_sha256=None)


def collect_one(job):
    scenario, output, source_hash = job
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = _episode_identity(scenario, source_hash)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    stem = output / f"episode_{scenario['seed']}_{key}"
    metadata_path, data_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if metadata_path.exists() and data_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    env = GeneralizedTeacherEnv(scenario, batched_actuator=True)
    arrays, result = rollout(env, scenario["seed"])
    arrays["pressure_schedule150"] = np.asarray(env.schedule_trace, dtype=np.float32)
    temporary = stem.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(data_path)
    metadata = dict(**identity, result=result, samples=len(arrays["observations"]),
        split=scenario["split"], data=data_path.name, wall_s=time.perf_counter() - started,
        launch=env.launch.as_dict(), events=env.plant.log_events,
        twin_provenance=env.kw.provenance)
    write_json(metadata_path, metadata)
    return metadata


def _run_collection(scenarios, output, workers, source_hash):
    rows = []
    jobs = [(scenario, str(output), source_hash) for scenario in scenarios]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect_one, job) for job in jobs]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                emit(dict(completed=completed, total=len(jobs), seed=row["seed"],
                    split=row["split"], captured=row["result"]["captured"],
                    hit15=row["result"]["hit15"], wall_s=round(row["wall_s"], 3)))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda row: row["scenario"]["scenario_id"])
    return rows, time.perf_counter() - started


def _metrics(rows):
    results = [row["result"] for row in rows]
    landed = [result for result in results if result["landing_error_m"] is not None]
    walls = np.asarray([row["wall_s"] for row in rows], dtype=float)
    count = len(rows)
    return dict(episodes=count,
        captured=sum(result["captured"] for result in results),
        released=sum(result["released"] for result in results),
        hit15=sum(result["hit15"] for result in results),
        hit30=sum(result["hit30"] for result in results),
        captured_rate=sum(result["captured"] for result in results) / count,
        released_rate=sum(result["released"] for result in results) / count,
        hit15_rate=sum(result["hit15"] for result in results) / count,
        hit30_rate=sum(result["hit30"] for result in results) / count,
        mean_landing_error_m=float(np.mean([result["landing_error_m"] for result in landed])) if landed else None,
        mean_relative_capture_speed_m_s=float(np.mean([result["relative_capture_speed_m_s"]
            for result in landed if result["relative_capture_speed_m_s"] is not None])) if landed else None,
        mean_wall_s=float(walls.mean()), median_wall_s=float(np.median(walls)),
        p95_wall_s=float(np.percentile(walls, 95)), worker_hours=float(walls.sum() / 3600.))


def _parameter_bins(rows):
    report = {}
    worst = 1.0
    for name, (low, high) in PARAMETER_RANGES.items():
        edges = np.linspace(low, high, 5)
        groups = []
        for index in range(4):
            selected = [row for row in rows if edges[index] <= row["scenario"][name]
                and (row["scenario"][name] < edges[index + 1] or index == 3)]
            metric = _metrics(selected)
            groups.append(dict(low=float(edges[index]), high=float(edges[index + 1]), **metric))
            worst = min(worst, metric["hit15_rate"])
        report[name] = groups
    return report, worst


def collect(args):
    manifest = make_scenario_manifest(args.episodes, args.seed, args.design_seed)
    source_hash = generalized_fingerprint()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "scenario_manifest.json"
    contract_path = args.out / "dataset_contract.json"
    contract = dict(schema=GENERALIZED_SCHEMA, source_hash=source_hash,
        manifest_sha256=manifest["manifest_sha256"], recipe=asdict(SoftRecipe()),
        batched_actuator=True)
    for path, expected in ((manifest_path, manifest), (contract_path, contract)):
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"collection contract changed; use a new output directory: {path}")
        write_json(path, expected)

    rows, elapsed = _run_collection(manifest["scenarios"], args.out, args.workers, source_hash)
    bins, worst_bin = _parameter_bins(rows)
    npz_bytes = sum(path.stat().st_size for path in args.out.glob("episode_*.npz"))
    report = dict(schema=GENERALIZED_SCHEMA, source_hash=source_hash,
        manifest_sha256=manifest["manifest_sha256"], elapsed_wall_s=elapsed,
        episodes_per_hour=len(rows) * 3600. / max(elapsed, 1e-9),
        dataset_size_mb=npz_bytes / (1024. ** 2), overall=_metrics(rows),
        train=_metrics([row for row in rows if row["split"] == "train"]),
        validation=_metrics([row for row in rows if row["split"] == "validation"]),
        parameter_bins=bins, worst_parameter_bin_hit15_rate=worst_bin)
    write_json(args.out / "collection_summary.json", report)
    emit(report)


def _compare_pair(job):
    scenario = job
    torch.set_num_threads(1)
    order = (False, True) if scenario["seed"] % 2 else (True, False)
    outputs = {}
    for batched in order:
        started = time.perf_counter()
        arrays, result = rollout(GeneralizedTeacherEnv(scenario,
            batched_actuator=batched), scenario["seed"])
        outputs[batched] = (arrays, result, time.perf_counter() - started)
    baseline, batched = outputs[False], outputs[True]
    shape_equal = all(baseline[0][key].shape == batched[0][key].shape for key in baseline[0])
    differences = {key: (float(np.max(np.abs(baseline[0][key] - batched[0][key])))
        if baseline[0][key].shape == batched[0][key].shape and baseline[0][key].size else None)
        for key in baseline[0]}
    boolean_keys = ("captured", "released", "hit15", "hit30", "grip_broken")
    booleans_equal = all(baseline[1][key] == batched[1][key] for key in boolean_keys)
    finite_differences = [value for value in differences.values() if value is not None]
    max_difference = max(finite_differences, default=0.)
    return dict(seed=scenario["seed"], scenario_id=scenario["scenario_id"],
        equivalent=shape_equal and booleans_equal and max_difference <= 1e-6,
        shape_equal=shape_equal, booleans_equal=booleans_equal,
        max_array_abs_difference=max_difference, array_differences=differences,
        baseline_wall_s=baseline[2], batched_wall_s=batched[2],
        speedup=baseline[2] / max(batched[2], 1e-9))


def verify_batch(args):
    manifest = make_scenario_manifest(args.episodes, args.seed, args.design_seed)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_compare_pair, scenario) for scenario in manifest["scenarios"]]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            emit(dict(completed=completed, total=len(futures), seed=row["seed"],
                equivalent=row["equivalent"], speedup=round(row["speedup"], 3)))
    rows.sort(key=lambda row: row["scenario_id"])
    report = dict(schema=GENERALIZED_SCHEMA, episodes=len(rows),
        all_equivalent=all(row["equivalent"] for row in rows),
        max_array_abs_difference=max(row["max_array_abs_difference"] for row in rows),
        mean_speedup=float(np.mean([row["speedup"] for row in rows])), records=rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "batch_verification.json", report)
    emit({key: value for key, value in report.items() if key != "records"})
    if not report["all_equivalent"]:
        raise RuntimeError("batched actuator changed one or more rollout trajectories")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-batch", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--episodes", type=int, default=24 if name == "verify-batch" else 256)
        command.add_argument("--seed", type=int, default=58001 if name == "verify-batch" else 60001)
        command.add_argument("--design-seed", type=int, default=20260918)
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--out", type=Path, default=Path("teacher_runs/v14_generalized") /
            ("verification" if name == "verify-batch" else "dataset_pilot"))
    args = parser.parse_args()
    if args.episodes < 5 or args.workers <= 0:
        parser.error("episodes must be at least five and workers must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(20260918)
    np.random.seed(20260918)
    {"verify-batch": verify_batch, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
