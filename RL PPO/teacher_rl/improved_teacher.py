"""V15 teacher: calibrated interception aim and target-conditioned throw energy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .__main__ import emit
from .data import rollout, write_json
from .env import ThrowDistribution
from .generalized_teacher import (_metrics, _parameter_bins, _task_config,
    make_scenario_manifest)
from .soft_env import SoftRecipe, SoftTeacherEnv, soft_fingerprint


IMPROVED_SCHEMA = "can_generalized_teacher_v15"


@dataclass(frozen=True)
class ImprovedRecipe:
    catch_aim_mode: str = "baseline"
    catch_match: float = .3
    catch_lead_s: float = .1
    catch_aim_max_m: float = .18
    catch_offset_scale: float = 1.
    catch_aim_alpha: float = 1.
    catch_ik_iterations: int = 6
    catch_ik_damping: float = 1e-4
    catch_q_limit_deg: float = 36.
    throw_reference_m: float = 1.45
    throw_force_base_n: float = 10.
    throw_force_gain_n_per_m: float = 10.
    throw_force_min_n: float = 7.
    throw_force_max_n: float = 14.
    throw_radius_base_m: float = .62
    throw_radius_gain_m_per_m: float = 0.
    throw_radius_min_m: float = .50
    throw_radius_max_m: float = .74

    def __post_init__(self):
        if self.catch_aim_mode not in ("baseline", "linear", "nonlinear"):
            raise ValueError("catch_aim_mode must be baseline, linear, or nonlinear")
        if not .2 <= self.catch_match <= .5:
            raise ValueError("catch_match must be in [0.2, 0.5]")
        if not 0. <= self.catch_lead_s <= .2:
            raise ValueError("catch_lead_s must be in [0, 0.2]")
        if not .10 <= self.catch_aim_max_m <= .25:
            raise ValueError("catch_aim_max_m must be in [0.10, 0.25]")
        if not .5 <= self.catch_offset_scale <= 1.5:
            raise ValueError("catch_offset_scale must be in [0.5, 1.5]")
        if not 0. < self.catch_aim_alpha <= 1.:
            raise ValueError("catch_aim_alpha must be in (0, 1]")
        if not 1 <= self.catch_ik_iterations <= 12:
            raise ValueError("catch_ik_iterations must be in [1, 12]")
        if not 1e-6 <= self.catch_ik_damping <= 1e-2:
            raise ValueError("catch_ik_damping must be in [1e-6, 1e-2]")
        if not 20. <= self.catch_q_limit_deg <= 45.:
            raise ValueError("catch_q_limit_deg must be in [20, 45]")
        if not -30. <= self.throw_force_gain_n_per_m <= 30.:
            raise ValueError("throw force gain must be in [-30, 30]")
        if not 0. < self.throw_force_min_n <= self.throw_force_max_n:
            raise ValueError("invalid throw force limits")
        if not -1. <= self.throw_radius_gain_m_per_m <= 1.:
            raise ValueError("throw radius gain must be in [-1, 1]")
        if not .4 <= self.throw_radius_min_m <= self.throw_radius_max_m <= .8:
            raise ValueError("invalid throw radius limits")

    def throw_force(self, target_distance):
        value = self.throw_force_base_n + self.throw_force_gain_n_per_m * (
            float(target_distance) - self.throw_reference_m)
        return float(np.clip(value, self.throw_force_min_n, self.throw_force_max_n))

    def throw_radius(self, target_distance):
        value = self.throw_radius_base_m + self.throw_radius_gain_m_per_m * (
            float(target_distance) - self.throw_reference_m)
        return float(np.clip(value, self.throw_radius_min_m, self.throw_radius_max_m))


def improved_fingerprint():
    digest = hashlib.sha256(soft_fingerprint().encode())
    for path in (Path(__file__), Path(__file__).with_name("generalized_teacher.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _AdaptiveController:
    def __init__(self, env, controller):
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "controller", controller)

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def __setattr__(self, name, value):
        if name in ("env", "controller"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.controller, name, value)

    def command(self, observation):
        env = self.env
        if env.phase() == 0 and env.improved_recipe.catch_aim_mode != "baseline":
            env.update_interception_target(observation)
        if env.phase() == 3:
            env.drive.F_max = env.throw_force_n
            env.drive.r_final = env.throw_radius_m
        return self.controller.command(observation)


class ImprovedTeacherEnv(SoftTeacherEnv):
    def __init__(self, scenario, recipe=ImprovedRecipe(), *, batched_actuator=True):
        self.scenario = dict(scenario)
        self.improved_recipe = recipe
        self.use_batched_actuator = bool(batched_actuator)
        config = _task_config(self.scenario)
        soft_recipe = SoftRecipe(match=recipe.catch_match)
        super().__init__(soft_recipe, config)
        centre, axis, _ = self.model.fk(np.zeros(12))
        self.distribution = ThrowDistribution(centre, axis,
            v0_nom=config.launch_speed, dist_m=config.launch_distance,
            v_jit=0., ang_jit_deg=self.scenario["launch_angle_jitter_deg"],
            pos_jit_m=self.scenario["launch_position_jitter_m"],
            ball_mass_kg=config.mass, ball_radius_m=config.ball_radius)
        self.throw_force_n = recipe.throw_force(config.target_distance)
        self.throw_radius_m = recipe.throw_radius(config.target_distance)

    def reset(self, seed):
        observation = super().reset(seed)
        self.plant.batched_actuator = self.use_batched_actuator
        self.catch.lead = self.improved_recipe.catch_lead_s
        if self.improved_recipe.catch_aim_mode != "baseline":
            self.catch.aim_correct = False
        self._interception_q = self.catch.q_hold0.copy()
        self._interception_offset = np.zeros(3)
        self.controller = _AdaptiveController(self, self.controller)
        return observation

    def update_interception_target(self, observation):
        catch, recipe = self.catch, self.improved_recipe
        ball = observation.get("ball_hat")
        if not ball or not ball.get("visible") or catch._vis < catch.min_vis - 1:
            return
        from can_catch import G_W, ball_eta
        eta = ball_eta(ball, catch.c0, catch.a0)
        if not np.isfinite(eta) or eta <= 0.:
            return
        position = np.asarray(ball["pos"], float)
        velocity = np.asarray(ball["vel"], float)
        offset = position + velocity * eta + .5 * G_W * eta**2 - catch.c0
        offset -= catch.a0 * float(offset @ catch.a0)
        length = float(np.linalg.norm(offset))
        if length > recipe.catch_aim_max_m:
            offset *= recipe.catch_aim_max_m / length
        scaled = recipe.catch_offset_scale * offset
        self._interception_offset = ((1. - recipe.catch_aim_alpha) *
            self._interception_offset + recipe.catch_aim_alpha * scaled)
        q = catch.q_hold0 + np.linalg.solve(catch._J0.T @ catch._J0 +
            1e-3 * np.eye(12), catch._J0.T @ self._interception_offset)
        if recipe.catch_aim_mode == "nonlinear":
            # Re-solve from the minimum-norm linear answer each tick. Carrying the
            # previous under-determined IK solution caused joint drift in V15 smoke1.
            target = catch.c0 + self._interception_offset
            q_limit = np.radians(recipe.catch_q_limit_deg)
            for _ in range(recipe.catch_ik_iterations):
                centre, jacobian, _ = self.model.jac(q)
                error = target - centre
                if np.linalg.norm(error) < 5e-4:
                    break
                step = np.linalg.solve(jacobian.T @ jacobian +
                    recipe.catch_ik_damping * np.eye(12), jacobian.T @ error)
                q = np.clip(q + step, -q_limit, q_limit)
        self._interception_q = q
        catch.aim_shift = self._interception_offset
        catch.plan.q0 = q

    def summary(self):
        landing_xy = (self.plant.ball_pos()[:2].tolist()
            if self.landing_error is not None else None)
        return dict(super().summary(), improved_recipe=asdict(self.improved_recipe),
            throw_force_n=self.throw_force_n,
            throw_radius_m=self.throw_radius_m,
            interception_q_max_deg=float(np.degrees(np.max(np.abs(self._interception_q)))),
            interception_offset_m=self._interception_offset.tolist(),
            landing_xy=landing_xy, target_xy=self.target.tolist(),
            release_prediction=self.releaser.state_at_cmd)


def candidates():
    base = ImprovedRecipe(catch_aim_mode="baseline", catch_match=.3,
        throw_force_gain_n_per_m=0.)
    return {
        "baseline": base,
        "match22": replace(base, catch_match=.22),
        "match20": replace(base, catch_match=.2),
        "lead05": replace(base, catch_lead_s=.05),
        "force_neg10": replace(base, throw_force_gain_n_per_m=-10.),
        "radius15": replace(base, throw_radius_gain_m_per_m=.15),
        "radius20": replace(base, throw_radius_gain_m_per_m=.2),
        "radius25": replace(base, throw_radius_gain_m_per_m=.25),
        "match20_radius15": replace(base, catch_match=.2,
            throw_radius_gain_m_per_m=.15),
        "match20_radius20": replace(base, catch_match=.2,
            throw_radius_gain_m_per_m=.2),
        "match20_radius25": replace(base, catch_match=.2,
            throw_radius_gain_m_per_m=.25),
        "match22_radius20": replace(base, catch_match=.22,
            throw_radius_gain_m_per_m=.2),
    }


def _identity(name, recipe, scenario, source_hash):
    return dict(schema=IMPROVED_SCHEMA, source_hash=source_hash, name=name,
        recipe=asdict(recipe), seed=scenario["seed"], scenario=scenario,
        batched_actuator=True)


def one_episode(job):
    name, recipe_data, scenario, output, source_hash = job
    torch.set_num_threads(1)
    recipe = ImprovedRecipe(**recipe_data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity(name, recipe, scenario, source_hash)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    path = output / f"episode_{scenario['seed']}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    env = ImprovedTeacherEnv(scenario, recipe)
    _, result = rollout(env, scenario["seed"])
    row = dict(**identity, result=result, launch=env.launch.as_dict(),
        wall_s=time.perf_counter() - started, events=env.plant.log_events)
    write_json(path, row)
    return row


def batch(recipe_map, scenarios, output, workers):
    source_hash = improved_fingerprint()
    jobs = [(name, asdict(recipe), scenario, str(output), source_hash)
        for name, recipe in recipe_map.items() for scenario in scenarios]
    rows = {name: [] for name in recipe_map}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_episode, job) for job in jobs]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows[row["name"]].append(row)
                emit(dict(completed=completed, total=len(jobs), candidate=row["name"],
                    seed=row["seed"], captured=row["result"]["captured"],
                    hit15=row["result"]["hit15"]))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    for candidate_rows in rows.values():
        candidate_rows.sort(key=lambda row: row["scenario"]["scenario_id"])
    return rows


def summarize(rows):
    summary = _metrics(rows)
    landed = [row["result"] for row in rows if row["result"]["landing_error_m"] is not None]
    summary.update(mean_weld_peak_n=float(np.mean([row["weld_peak_n"] for row in landed])) if landed else None,
        mean_pressure_integral_psi_s=float(np.mean([row["pressure_integral_psi_s"] for row in landed])) if landed else None,
        max_contact_peak_n=max(row["result"]["contact_peak_n"] for row in rows))
    return summary


def load_selected(path, *, require_improved=True):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != IMPROVED_SCHEMA:
        raise ValueError(f"recipe schema mismatch: {path}")
    if payload.get("source_hash") != improved_fingerprint():
        raise ValueError(f"recipe source fingerprint mismatch: {path}")
    if require_improved and not payload.get("improved", False):
        raise ValueError(f"recipe did not pass paired validation: {path}")
    return ImprovedRecipe(**payload["recipe"]), payload


def collect_episode(job):
    recipe_data, scenario, output, source_hash = job
    torch.set_num_threads(1)
    recipe = ImprovedRecipe(**recipe_data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity("selected", recipe, scenario, source_hash)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    stem = output / f"episode_{scenario['seed']}_{key}"
    metadata_path, data_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if metadata_path.exists() and data_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    env = ImprovedTeacherEnv(scenario, recipe)
    arrays, result = rollout(env, scenario["seed"])
    arrays["pressure_schedule150"] = np.asarray(env.schedule_trace, dtype=np.float32)
    temporary = stem.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(data_path)
    metadata = dict(**identity, result=result, samples=len(arrays["observations"]),
        split=scenario["split"], data=data_path.name,
        wall_s=time.perf_counter() - started, launch=env.launch.as_dict(),
        events=env.plant.log_events, twin_provenance=env.kw.provenance)
    write_json(metadata_path, metadata)
    return metadata


def collect_batch(recipe, scenarios, output, workers, source_hash):
    jobs = [(asdict(recipe), scenario, str(output), source_hash) for scenario in scenarios]
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect_episode, job) for job in jobs]
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


def collect(args):
    recipe, selection = load_selected(args.recipe)
    manifest = make_scenario_manifest(args.episodes, args.seed, args.design_seed)
    source_hash = improved_fingerprint()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=IMPROVED_SCHEMA, source_hash=source_hash,
        manifest_sha256=manifest["manifest_sha256"], recipe=asdict(recipe),
        selection_file=str(args.recipe), selection_source_hash=selection["source_hash"],
        batched_actuator=True)
    for path, expected in ((args.out / "scenario_manifest.json", manifest),
            (args.out / "dataset_contract.json", contract)):
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"collection contract changed; use a new output directory: {path}")
        write_json(path, expected)
    rows, elapsed = collect_batch(recipe, manifest["scenarios"], args.out,
        args.workers, source_hash)
    bins, worst_bin = _parameter_bins(rows)
    npz_bytes = sum(path.stat().st_size for path in args.out.glob("episode_*.npz"))
    report = dict(schema=IMPROVED_SCHEMA, source_hash=source_hash,
        manifest_sha256=manifest["manifest_sha256"], elapsed_wall_s=elapsed,
        episodes_per_hour=len(rows) * 3600. / max(elapsed, 1e-9),
        dataset_size_mb=npz_bytes / (1024. ** 2), overall=_metrics(rows),
        train=_metrics([row for row in rows if row["split"] == "train"]),
        validation=_metrics([row for row in rows if row["split"] == "validation"]),
        parameter_bins=bins, worst_parameter_bin_hit15_rate=worst_bin,
        recipe=asdict(recipe))
    write_json(args.out / "collection_summary.json", report)
    emit(report)


def audit(args):
    recipe, selection = load_selected(args.recipe)
    baseline = candidates()["baseline"]
    manifest = make_scenario_manifest(args.episodes, args.seed, args.design_seed)
    rows = batch({"baseline": baseline, "selected": recipe}, manifest["scenarios"],
        args.out / "episodes", args.workers)
    summary = {name: summarize(value) for name, value in rows.items()}
    base, chosen = summary["baseline"], summary["selected"]
    passed = (chosen["captured"] >= base["captured"] and
        chosen["released"] >= base["released"] and chosen["hit15"] > base["hit15"])
    report = dict(schema=IMPROVED_SCHEMA, source_hash=improved_fingerprint(),
        selection_source_hash=selection["source_hash"], passed=passed,
        manifest=manifest, summary=summary)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "audit_summary.json", report)
    emit(dict(passed=passed, summary=summary))
    if not passed:
        raise RuntimeError("selected teacher did not pass the independent audit")


def search(args):
    recipes = candidates()
    screen_manifest = make_scenario_manifest(args.screen_episodes, args.seed, args.design_seed)
    validation_manifest = make_scenario_manifest(args.validation_episodes,
        args.validation_seed, args.design_seed + 100)
    screen = batch(recipes, screen_manifest["scenarios"], args.out / "screen", args.workers)
    screen_summary = {name: summarize(rows) for name, rows in screen.items()}
    write_json(args.out / "screen_summary.json", screen_summary)
    order = sorted((name for name in recipes if name != "baseline"),
        key=lambda name: (-screen_summary[name]["hit15"], -screen_summary[name]["captured"],
            screen_summary[name]["mean_landing_error_m"] or float("inf")))
    finalists = {name: recipes[name] for name in ["baseline", *order[:args.top]]}
    validation = batch(finalists, validation_manifest["scenarios"],
        args.out / "validation", args.workers)
    validation_summary = {name: summarize(rows) for name, rows in validation.items()}
    selected = min((name for name in finalists if name != "baseline"),
        key=lambda name: (-validation_summary[name]["hit15"],
            -validation_summary[name]["captured"],
            validation_summary[name]["mean_landing_error_m"] or float("inf")))
    baseline = validation_summary["baseline"]
    chosen = validation_summary[selected]
    improved = (chosen["captured"] >= baseline["captured"] and
        chosen["released"] >= baseline["released"] and chosen["hit15"] > baseline["hit15"])
    payload = dict(schema=IMPROVED_SCHEMA, source_hash=improved_fingerprint(),
        selected=selected, recipe=asdict(recipes[selected]), improved=improved,
        screen_manifest=screen_manifest, validation_manifest=validation_manifest,
        screen=screen_summary, validation=validation_summary)
    write_json(args.out / "selected_recipe.json", payload)
    emit(dict(selected=selected, improved=improved, validation=validation_summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("search")
    command.add_argument("--screen-episodes", type=int, default=32)
    command.add_argument("--validation-episodes", type=int, default=64)
    command.add_argument("--seed", type=int, default=80001)
    command.add_argument("--validation-seed", type=int, default=81001)
    command.add_argument("--design-seed", type=int, default=20260919)
    command.add_argument("--top", type=int, default=3)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--out", type=Path,
        default=Path("teacher_runs/v15_improved/search"))
    audit_command = sub.add_parser("audit")
    audit_command.add_argument("--recipe", type=Path, required=True)
    audit_command.add_argument("--episodes", type=int, default=64)
    audit_command.add_argument("--seed", type=int, default=85001)
    audit_command.add_argument("--design-seed", type=int, default=20260922)
    audit_command.add_argument("--workers", type=int, default=8)
    audit_command.add_argument("--out", type=Path,
        default=Path("teacher_runs/v15_improved/audit"))
    collect_command = sub.add_parser("collect")
    collect_command.add_argument("--recipe", type=Path, required=True)
    collect_command.add_argument("--episodes", type=int, default=1000)
    collect_command.add_argument("--seed", type=int, default=90001)
    collect_command.add_argument("--design-seed", type=int, default=20260923)
    collect_command.add_argument("--workers", type=int, default=8)
    collect_command.add_argument("--out", type=Path,
        default=Path("teacher_runs/v15_improved/dataset_1000"))
    args = parser.parse_args()
    positive = ("screen_episodes", "validation_episodes", "top", "workers") \
        if args.command == "search" else ("episodes", "workers")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.command in ("audit", "collect") and args.episodes < 5:
        parser.error("episodes must be at least five")
    torch.set_num_threads(1)
    torch.manual_seed(20260919)
    np.random.seed(20260919)
    {"search": search, "audit": audit, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
