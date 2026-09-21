"""V17 buffered residual PPO with signed pressure sensitivity checks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import shutil
from pathlib import Path
import time

import numpy as np
import torch

from .buffered_env import BufferedConfig as ContinuousConfig, BufferedEnv as ContinuousEnv, SCHEMA, ACTION_NAMES, CONTROL_COLUMNS
from .buffered_env import OBS_DIM, RESIDUAL_SIZE
from .buffered_model import ResidualPolicy, load_policy, save_policy, update_policy
from .continuous_rl import source_hash as v16_source_hash, previous_seeds as v16_previous_seeds
from .data import write_json
from .generalized_teacher import make_scenario_manifest
from .improved_rl import (DEFAULT_RL_OUT, _init_worker, compare, file_hash,
    load_anchor, read_json, rl_fingerprint, run_jobs, summarize)
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv
from .reachable_collection import constrain_manifest, _grasp_height
from rl_ppo.ppo import PPOConfig, RolloutBuffer

DEFAULT_ANCHOR = Path("teacher_runs/v15_improved/bc_v1_formal/bc_best.pt")
DEFAULT_OUT = Path("teacher_runs/v17_buffered")


def source_hash():
    digest = hashlib.sha256(v16_source_hash().encode())
    for name in ("buffered_env.py", "buffered_rl.py", "buffered_model.py"):
        path = Path(__file__).with_name(name)
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(job):
    scenario, anchor_path, config, checkpoint, stochastic, policy_seed, constant_action = job
    torch.set_num_threads(1)
    torch.manual_seed(policy_seed)
    anchor, payload = load_anchor(anchor_path)
    recipe = ImprovedRecipe(**payload["recipe"])
    if config is None:
        env = ImprovedTeacherEnv(scenario, recipe)
    else:
        env = ContinuousEnv(scenario, recipe, anchor, ContinuousConfig(**config))
    model = None
    if checkpoint is not None:
        model, _ = load_policy(checkpoint, source_hash(), file_hash(anchor_path))
        model.eval()
    obs = env.reset(scenario["seed"])
    samples = []
    started = time.perf_counter()
    while not env.done:
        with torch.no_grad():
            if config is None:
                action = anchor.distribution(torch.tensor(obs[None])).mean.tanh()[0].numpy()
            elif model is None:
                action = np.zeros(RESIDUAL_SIZE, dtype=np.float32) if constant_action is None else np.asarray(constant_action, dtype=np.float32)
            else:
                mask = env.residual_mask()
                action, raw, logp, value = model.act(torch.tensor(obs[None]),
                    deterministic=not stochastic, action_mask=torch.tensor(mask))
                action = action[0].numpy()
        following, reward, done, result = env.step(action)
        if stochastic:
            samples.append((obs, raw[0].numpy(), logp.item(), reward, done, value.item(), mask))
        obs = following
    result.setdefault("catch_to_release_s", None if result["release_time"] is None or result["capture_time"] is None
        else result["release_time"]-result["capture_time"])
    result.setdefault("catch_to_orbit_s", None if env.controller.t_hand is None or env.capture_physics_time is None
        else env.controller.t_hand-env.capture_physics_time)
    row = dict(seed=scenario["seed"], scenario=scenario, result=result,
        wall_s=time.perf_counter()-started, samples=samples)
    if config is not None:
        trace = np.asarray(env.motion_trace)
        controls = np.asarray(env.control_trace)
        column = {name: controls[:, index] for index, name in enumerate(CONTROL_COLUMNS)}
        result["control_diagnostics"] = dict(
            max_catch_pressure_delta_psi=float(np.abs(column["catch_pressure_delta_psi"]).max()),
            min_catch_pressure_delta_psi=float(column["catch_pressure_delta_psi"].min()),
            max_buffer_pressure_delta_psi=float(np.abs(column["buffer_pressure_delta_psi"]).max()),
            min_buffer_kd_scale=float(column["buffer_kd_scale"].min()),
            max_buffer_kd_scale=float(column["buffer_kd_scale"].max()),
            clipped_fraction=float(column["clipped_fraction"].mean()),
            min_command_psi=float(column["command_pressure_min_psi"].min()),
            max_command_psi=float(column["command_pressure_max_psi"].max()))
        result["catch_diagnostics"] = dict(max_feedback_torque=float(trace[:,8].max()),
            max_feedforward_torque=float(trace[:,7].max()), max_saturation_fraction=float(trace[:,9].max()),
            terminal_residual=env.last_residual.tolist())
        if not stochastic:
            row["motion_trace_columns"] = ["time_s","tip_speed_m_s","weld_force_n","held","phase",
                "measured_pressure_psi","command_pressure_psi","tau_ff_max","tau_fb_max","saturation_fraction"]
            row["motion_trace"] = env.motion_trace
            row["control_trace_columns"] = CONTROL_COLUMNS
            row["control_trace"] = env.control_trace
    return row


def evaluate_rows(pool, scenarios, anchor, config, checkpoint=None, stochastic=False, constant_action=None):
    jobs = [(row, str(anchor.resolve()), None if config is None else asdict(config),
        None if checkpoint is None else str(checkpoint.resolve()), stochastic, row["seed"], constant_action) for row in scenarios]
    rows = []
    from concurrent.futures import as_completed
    futures = [pool.submit(worker, job) for job in jobs]
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        print(f"completed {len(rows)}/{len(jobs)} seed={row['seed']} hit15={row['result']['hit15']}", flush=True)
    return sorted(rows, key=lambda row: row["seed"])


def report(rows, baseline):
    result = compare(rows, baseline)
    common = [(r["result"], b["result"]) for r, b in zip(sorted(rows,key=lambda r:r["seed"]),
        sorted(baseline,key=lambda r:r["seed"])) if r["result"]["captured"] and b["result"]["captured"]]
    result["paired_captures"] = len(common)
    for key in ("catch_to_orbit_s", "catch_to_release_s", "postcatch_low_speed_s", "longest_postcatch_pause_s"):
        values = [row["result"][key] for row in rows if row["result"].get(key) is not None]
        result["mean_"+key] = float(np.mean(values)) if values else None
        result["max_"+key] = float(max(values)) if values else None
    if common:
        result["impact_peak_ratio"] = float(np.mean([r["impact_weld_peak_n"] for r,b in common]) /
            max(1e-6, np.mean([b["impact_weld_peak_n"] for r,b in common])))
        result["impact_impulse_ratio"] = float(np.mean([r["impact_weld_impulse_ns"] for r,b in common]) /
            max(1e-6, np.mean([b["impact_weld_impulse_ns"] for r,b in common])))
        result["impact_peak_p95_ratio"] = float(np.percentile([r["impact_weld_peak_n"] for r,b in common],95) /
            max(1e-6,np.percentile([b["impact_weld_peak_n"] for r,b in common],95)))
        for key in ("buffer_weld_peak_n", "buffer_weld_impulse_ns"):
            if all(key in r and key in b for r,b in common):
                ratio = float((np.mean([r[key] for r,b in common])+1e-6) /
                    (np.mean([b[key] for r,b in common])+1e-6))
                result[key+"_ratio"] = ratio
                result["checks"][key+"_guard"] = ratio <= 1.05
    result["task_nonregression"] = (result["summary"]["hit15"] >= result["baseline"]["hit15"] and
        result["summary"]["captured"] >= result["baseline"]["captured"] and
        result["summary"]["released"] >= result["baseline"]["released"])
    result["checks"]["joint_excursion_guard"] = max(r["result"]["max_joint_deg"] for r in rows) <= (
        max(r["result"]["max_joint_deg"] for r in baseline)+1.)
    if any("continuous_config" in row["result"] for row in rows):
        result["checks"]["immediate_handover"] = (result["max_catch_to_orbit_s"] is not None and
            result["max_catch_to_orbit_s"] <= .02)
        result["checks"]["no_sustained_pause"] = (result["max_longest_postcatch_pause_s"] is not None and
            result["max_longest_postcatch_pause_s"] <= .1)
        result["eligible"] = all(result["checks"].values())
    result["primary_quality_improved"] = (result["eligible"] and result["task_nonregression"] and bool(common)
        and result["impact_peak_ratio"] <= .97 and result["impact_impulse_ratio"] <= .97)
    return result


def selection_score(result):
    return (result["summary"]["hit15"], result["summary"]["captured"],
        -.5*(result.get("impact_peak_ratio",1e9)+result.get("impact_impulse_ratio",1e9)),
        -(result["summary"]["mean_landing_error_m"] or 1e9))


def candidate_eligible(result):
    return (result["eligible"] and result["task_nonregression"] and
        result.get("impact_peak_ratio",1e9) <= 1.+1e-9 and
        result.get("impact_peak_p95_ratio",1e9) <= 1.+1e-9 and
        result.get("impact_impulse_ratio",1e9) <= 1.+1e-9)


def pilot(args):
    _, payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    manifest = scenarios(args.episodes, args.seed, args.design_seed, recipe)
    config = ContinuousConfig(pressure_authority_psi=args.pressure_authority)
    variants = {"v15_bc": (None, None), "zero": (config, None)}
    if args.sensitivity:
        dimensions = list(ACTION_NAMES) if args.dimension == "all" else [args.dimension]
        for name in dimensions:
            levels = range(1, int(args.pressure_authority)+1) if name.endswith("pressure") else (None,)
            for level in levels:
                for sign in (-1., 1.):
                    action = np.zeros(RESIDUAL_SIZE).tolist()
                    action[ACTION_NAMES.index(name)] = sign if level is None else sign*level/args.pressure_authority
                    suffix = "" if level is None else f"_{level}psi"
                    variants[name + ("_minus" if sign < 0 else "_plus") + suffix] = (config, action)
    identity = dict(schema=SCHEMA, source_hash=source_hash(), anchor_sha256=file_hash(args.init),
        manifest=manifest, variants={name: dict(config=None if cfg is None else asdict(cfg), action=action)
            for name, (cfg, action) in variants.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out/"pilot_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != identity:
            raise ValueError("Pilot contract changed; use a new output directory")
    elif {r["seed"] for r in manifest["scenarios"]} & previous_seeds():
        raise ValueError("Pilot seeds overlap earlier development; choose fresh seeds")
    write_json(contract_path, identity)
    # An unapproved candidate is allowed only in bounded code-smoke tests.
    candidate = dict(variant="zero", config=asdict(config), pilot_source_hash=identity["source_hash"],
        anchor_sha256=identity["anchor_sha256"], development_seeds=[r["seed"] for r in manifest["scenarios"]],
        episodes=args.episodes, eligible=False, schema=SCHEMA)
    write_json(args.out/"candidate_config.json", candidate)
    comparisons, baseline, zero = {}, None, None
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for name, (cfg, action) in variants.items():
            path = args.out/(name+"_episodes.json")
            rows = read_json(path) if path.exists() else evaluate_rows(
                pool, manifest["scenarios"], args.init, cfg, constant_action=action)
            write_json(path, rows)
            if name == "v15_bc":
                baseline = rows
            if name == "zero":
                zero = rows
            comparison = report(rows, baseline)
            comparisons[name] = dict(vs_v15_bc=comparison,
                vs_zero=None if zero is None else report(rows, zero))
            write_json(args.out/"comparison.json", comparisons)
            print(dict(variant=name, summary=comparison["summary"], checks=comparison["checks"]), flush=True)
    result = comparisons["zero"]["vs_v15_bc"]
    if result["eligible"] and result["task_nonregression"]:
        write_json(args.out/"selected_config.json", dict(candidate, eligible=True,
            primary_quality_improved=result["primary_quality_improved"]))
        print("Zero-residual baseline passed development gates.", flush=True)
    else:
        print("Baseline failed: formal training remains blocked. Sensitivity is diagnostic only.", flush=True)

def previous_seeds(exclude_run=None):
    seeds = v16_previous_seeds()
    config_path = DEFAULT_RL_OUT/"run_config.json"
    if config_path.exists():
        contract = read_json(config_path)
        seeds.update(contract["excluded_seeds"])
        seeds.update(r["seed"] for r in contract["validation_manifest"]["scenarios"])
        for path in DEFAULT_RL_OUT.glob("update_*.json"):
            seeds.update(r["seed"] for r in read_json(path)["rows"])
        for path in DEFAULT_RL_OUT.glob("**/evaluation_contract.json"):
            seeds.update(r["seed"] for r in read_json(path)["manifest"]["scenarios"])
    excluded = None if exclude_run is None else Path(exclude_run).resolve()
    for path in DEFAULT_OUT.rglob("pilot_contract.json"):
        seeds.update(r["seed"] for r in read_json(path)["manifest"]["scenarios"])
    for path in DEFAULT_OUT.rglob("run_config.json"):
        if path.parent.resolve() == excluded:
            continue
        contract = read_json(path)
        seeds.update(r["seed"] for r in contract["validation_manifest"]["scenarios"])
        updates = [int(p.stem.split("_")[-1]) for p in path.parent.glob("update_*.json")]
        if updates:
            seeds.update(range(contract["seed"],contract["seed"]+max(updates)*contract["episodes_per_update"]))
    for path in DEFAULT_OUT.rglob("evaluation_contract.json"):
        if excluded is not None and path.resolve().is_relative_to(excluded):
            continue
        seeds.update(r["seed"] for r in read_json(path)["manifest"]["scenarios"])
    return seeds


def scenarios(count, seed, design_seed, recipe):
    manifest = make_scenario_manifest(count,seed,design_seed)
    return constrain_manifest(manifest,_grasp_height(recipe,manifest["scenarios"][0]))


def train(args):
    anchor, anchor_payload = load_anchor(args.init)
    selection = read_json(args.config)
    fingerprint, anchor_hash = source_hash(), file_hash(args.init)
    if ((not selection["eligible"] and not args.smoke) or selection["schema"] != SCHEMA or
            selection["pilot_source_hash"] != fingerprint or
            selection["anchor_sha256"] != anchor_hash):
        raise ValueError("Run the development pilot again with matching source and BC anchor")
    if selection["episodes"] < 40 and not args.smoke:
        raise ValueError("Formal training requires a >=40-scenario development pilot; use --smoke only for debugging")
    config = ContinuousConfig(**selection["config"])
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    validation = scenarios(args.eval_episodes,args.eval_seed,args.design_seed+10000,recipe)
    forbidden = previous_seeds(args.out) | set(selection["development_seeds"])
    validation_seeds = {r["seed"] for r in validation["scenarios"]}
    training_seeds = set(range(args.seed,args.seed+args.updates*args.episodes_per_update))
    if forbidden & (validation_seeds|training_seeds) or validation_seeds & training_seeds:
        raise ValueError("Development, previous, training and validation seeds must be disjoint")
    for path in args.out.glob("**/evaluation_contract.json"):
        tested = {r["seed"] for r in read_json(path)["manifest"]["scenarios"]}
        if tested & training_seeds:
            raise ValueError("Training would reuse an already evaluated test seed")
    ppo = PPOConfig(gamma=.997,gae_lambda=.995,clip_ratio=.1,learning_rate=args.lr,
        update_epochs=3,minibatch_size=256,max_grad_norm=1.,target_kl=.02,entropy_coefficient=.001)
    contract = dict(schema=SCHEMA,source_hash=fingerprint,anchor_sha256=anchor_hash,
        action_names=list(ACTION_NAMES),config=asdict(config),ppo=asdict(ppo),critic_lr=args.critic_lr,seed=args.seed,
        design_seed=args.design_seed,episodes_per_update=args.episodes_per_update,
        eval_every=args.eval_every,validation_manifest=validation,excluded_seeds=sorted(forbidden),
        smoke=args.smoke)
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.design_seed)
    np.random.seed(args.design_seed)
    start, best_update, best_score = 1, 0, None
    if args.resume:
        model,payload = load_policy(args.out/"ppo_latest.pt",fingerprint,anchor_hash)
        if payload["contract"] != contract:
            raise ValueError("Resume configuration differs from checkpoint")
        start,best_update,best_score = payload["update"]+1,payload["best_update"],tuple(payload["best_score"])
        if args.updates < start:
            raise ValueError("Requested updates already completed")
    else:
        if any(args.out.iterdir()):
            raise ValueError("Training output is not empty; use --resume or a new directory")
        model = ResidualPolicy(anchor)
    optimizer = torch.optim.Adam([
        dict(params=[*model.actor.parameters(),model.log_std],lr=args.lr),
        dict(params=model.critic.parameters(),lr=args.critic_lr)])
    if args.resume:
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
    write_json(args.out/"run_config.json",contract)

    def save(path, update, optimizer_state=True):
        save_policy(path,model,source_hash=fingerprint,anchor_sha256=anchor_hash,
            contract=contract,update=update,best_update=best_update,best_score=best_score,
            training_seed_end=args.seed+update*args.episodes_per_update,
            optimizer=optimizer.state_dict() if optimizer_state else None,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state())

    def publish():
        temporary = args.out/"ppo_best.tmp"
        shutil.copyfile(args.out/"checkpoints"/f"selected_{best_update:04d}.pt",temporary)
        temporary.replace(args.out/"ppo_best.pt")

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init_worker) as pool:
        baseline_path = args.out/"validation_0000.json"
        if args.resume:
            initial = read_json(baseline_path)
            baseline,v15_baseline = initial["rows"],initial["v15_bc_rows"]
        else:
            baseline = evaluate_rows(pool,validation["scenarios"],args.init,config)
            v15_baseline = evaluate_rows(pool,validation["scenarios"],args.init,None)
            initial_report = report(baseline,v15_baseline)
            write_json(baseline_path,dict(rows=baseline,v15_bc_rows=v15_baseline,
                report=initial_report,update=0))
            if not args.smoke and not (initial_report["eligible"] and initial_report["task_nonregression"]):
                raise RuntimeError("Continuous baseline failed fresh validation; do not begin formal RL")
            best_score = selection_score(report(baseline,baseline))
            save(args.out/"checkpoints/selected_0000.pt",0)
            save(args.out/"ppo_latest.pt",0)
            publish()
        for update in range(start,args.updates+1):
            snapshot = args.out/"rollout_policy.pt"
            save(snapshot,update-1,False)
            manifest = scenarios(args.episodes_per_update,args.seed+(update-1)*args.episodes_per_update,
                args.design_seed+update,recipe)
            rows = evaluate_rows(pool,manifest["scenarios"],args.init,config,snapshot,True)
            buffer = RolloutBuffer(sum(len(r["samples"]) for r in rows),OBS_DIM,RESIDUAL_SIZE)
            for row in rows:
                for sample in row.pop("samples"):
                    buffer.add(*sample)
            buffer.finish(0.,ppo)
            metrics = update_policy(model,optimizer,buffer,ppo)
            write_json(args.out/f"update_{update:04d}.json",dict(update=update,metrics=metrics,
                summary=summarize(rows),rows=rows))
            if update%args.eval_every == 0 or update == args.updates:
                save(snapshot,update,False)
                evaluated = evaluate_rows(pool,validation["scenarios"],args.init,config,snapshot)
                comparison = report(evaluated,baseline)
                vs_v15 = report(evaluated,v15_baseline)
                selected = (candidate_eligible(comparison) and vs_v15["eligible"] and
                    vs_v15["task_nonregression"] and selection_score(comparison)>tuple(best_score))
                if selected:
                    best_update,best_score = update,selection_score(comparison)
                    save(args.out/"checkpoints"/f"selected_{update:04d}.pt",update)
                write_json(args.out/f"validation_{update:04d}.json",dict(update=update,rows=evaluated,
                    report=comparison,vs_v15_bc=vs_v15,selected=selected))
            save(args.out/"ppo_latest.pt",update)
            publish()
            print(dict(update=update,best_update=best_update,metrics=metrics),flush=True)
    write_json(args.out/"training_summary.json",dict(completed_updates=args.updates,
        total_training_episodes=args.updates*args.episodes_per_update,best_update=best_update,
        best_is_zero_residual=best_update==0,smoke=args.smoke,
        elapsed_this_invocation_s=time.perf_counter()-started))


def evaluate(args):
    fingerprint,anchor_hash = source_hash(),file_hash(args.init)
    _,payload = load_policy(args.checkpoint,fingerprint,anchor_hash)
    _,anchor_payload = load_anchor(args.init)
    contract = payload["contract"]
    config = ContinuousConfig(**contract["config"])
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    manifest = scenarios(args.episodes,args.seed,args.design_seed,recipe)
    latest_path = args.checkpoint.parent/"ppo_latest.pt"
    latest = load_policy(latest_path,fingerprint,anchor_hash)[1] if latest_path.exists() else payload
    if latest["contract"] != contract:
        raise ValueError("Best/latest checkpoint contracts differ")
    forbidden = set(contract["excluded_seeds"]) | previous_seeds(args.checkpoint.parent)
    forbidden.update(r["seed"] for r in contract["validation_manifest"]["scenarios"])
    forbidden.update(range(contract["seed"],max(payload["training_seed_end"],latest["training_seed_end"])))
    if forbidden & {r["seed"] for r in manifest["scenarios"]}:
        raise ValueError("Final evaluation would reuse development/training/validation seeds")
    identity = dict(source_hash=fingerprint,anchor_sha256=anchor_hash,
        checkpoint_sha256=file_hash(args.checkpoint),selected_update=payload["update"],manifest=manifest)
    args.out.mkdir(parents=True,exist_ok=True)
    path = args.out/"evaluation_contract.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Evaluation contract differs; use a fresh output directory")
    write_json(path,identity)
    groups = {}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init_worker) as pool:
        for name,environment,checkpoint in (("v15_bc",None,None),("continuous_bc",config,None),
                ("rl",config,args.checkpoint)):
            path = args.out/(name+"_episodes.json")
            groups[name] = read_json(path) if path.exists() else evaluate_rows(pool,manifest["scenarios"],
                args.init,environment,checkpoint)
            write_json(path,groups[name])
    write_json(args.out/"comparison.json",dict(selected_update=payload["update"],
        summaries={name:summarize(rows) for name,rows in groups.items()},
        rl_vs_continuous_bc=report(groups["rl"],groups["continuous_bc"]),
        rl_vs_v15_bc=report(groups["rl"],groups["v15_bc"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("pilot")
    command.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    command.add_argument("--out", type=Path, default=DEFAULT_OUT/"pilot")
    command.add_argument("--episodes", type=int, default=10)
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--seed", type=int, default=6100001)
    command.add_argument("--design-seed", type=int, default=20261018)
    command.add_argument("--sensitivity", action="store_true")
    command.add_argument("--dimension", choices=[*ACTION_NAMES, "all"], default="catch_pressure")
    command.add_argument("--pressure-authority", type=float, choices=(1., 2., 3.), default=1.)
    command = sub.add_parser("train")
    command.add_argument("--init",type=Path,default=DEFAULT_ANCHOR)
    command.add_argument("--config",type=Path,required=True)
    command.add_argument("--out",type=Path,default=DEFAULT_OUT/"rl_formal")
    command.add_argument("--resume",action="store_true")
    command.add_argument("--smoke",action="store_true")
    for name,default in (("updates",100),("episodes-per-update",32),("workers",8),
            ("eval-episodes",80),("eval-every",10),("seed",7100001),("eval-seed",7200001),
            ("design-seed",20261020),("threads",4)):
        command.add_argument("--"+name,type=int,default=default)
    command.add_argument("--lr",type=float,default=1e-4)
    command.add_argument("--critic-lr",type=float,default=3e-4)
    command = sub.add_parser("evaluate")
    command.add_argument("--init",type=Path,default=DEFAULT_ANCHOR)
    command.add_argument("--checkpoint",type=Path,default=DEFAULT_OUT/"rl_formal/ppo_best.pt")
    command.add_argument("--out",type=Path,default=DEFAULT_OUT/"rl_formal/evaluation_200")
    for name,default in (("episodes",200),("workers",8),("seed",7300001),("design-seed",20261021)):
        command.add_argument("--"+name,type=int,default=default)
    args = parser.parse_args()
    for name in ("workers","updates","eval_every","threads"):
        if hasattr(args,name) and getattr(args,name)<1:
            parser.error(name+" must be positive")
    for name in ("episodes","episodes_per_update","eval_episodes"):
        if hasattr(args,name) and getattr(args,name)<5:
            parser.error(name+" must be >=5")
    if args.command == "train":
        if min(args.lr,args.critic_lr)<=0 or not np.isfinite([args.lr,args.critic_lr]).all():
            parser.error("Learning rates must be finite and positive")
        if args.smoke and (args.updates > 2 or args.episodes_per_update > 8 or args.eval_episodes > 10):
            parser.error("Smoke is bounded to <=2 updates, <=8 train episodes/update and <=10 validation episodes")
        if not args.smoke and args.eval_episodes < 80:
            parser.error("Formal validation requires >=80 scenarios")
    torch.set_num_threads(getattr(args,"threads",1))
    {"pilot":pilot,"train":train,"evaluate":evaluate}[args.command](args)


if __name__ == "__main__":
    main()
