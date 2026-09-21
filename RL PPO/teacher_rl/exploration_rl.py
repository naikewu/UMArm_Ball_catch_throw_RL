"""V18 on-policy simulator research: separate candidate and accepted checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .exploration_env import (ExplorationConfig, ExplorationEnv, SCHEMA, ACTION_NAMES,
    RESIDUAL_SIZE, OBS_DIM, CONTROL_COLUMNS, EXPLORATION_COLUMNS)
from .exploration_model import ResidualPolicy, load_policy, save_policy, update_policy
from .buffered_rl import (source_hash as v17_source_hash, previous_seeds as v17_previous_seeds,
    report as previous_report, candidate_eligible, scenarios, DEFAULT_ANCHOR)
from .improved_rl import _init_worker, load_anchor, file_hash, read_json, summarize
from .improved_teacher import ImprovedRecipe, ImprovedTeacherEnv
from .data import write_json
from rl_ppo.ppo import RolloutBuffer, PPOConfig

DEFAULT_OUT = Path("teacher_runs/v18_online")


def source_hash():
    digest = hashlib.sha256(v17_source_hash().encode())
    for name in ("exploration_env.py","exploration_model.py","exploration_rl.py"):
        path = Path(__file__).with_name(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def previous_seeds(exclude_run=None):
    result = v17_previous_seeds()
    excluded = None if exclude_run is None else Path(exclude_run).resolve()
    for path in DEFAULT_OUT.rglob("run_config.json"):
        if path.parent.resolve() == excluded:
            continue
        contract = read_json(path)
        result.update(r["seed"] for r in contract["validation_manifest"]["scenarios"])
        result.update(range(contract["seed"],contract["seed"]+contract["budget_updates"]*contract["episodes_per_update"]))
    for path in DEFAULT_OUT.rglob("evaluation_contract.json"):
        result.update(r["seed"] for r in read_json(path)["manifest"]["scenarios"])
    return result


def worker(job):
    scenario,anchor_path,config,checkpoint,stochastic,constant = job
    torch.set_num_threads(1)
    torch.manual_seed(scenario["seed"])
    anchor,payload = load_anchor(anchor_path)
    recipe = ImprovedRecipe(**payload["recipe"])
    env = (ImprovedTeacherEnv(scenario,recipe) if config is None else
        ExplorationEnv(scenario,recipe,anchor,ExplorationConfig(**config)))
    model = None if checkpoint is None else load_policy(checkpoint,source_hash(),file_hash(anchor_path))[0]
    if model is not None:
        model.eval()
    obs = env.reset(scenario["seed"])
    samples = []
    start = time.perf_counter()
    while not env.done:
        with torch.no_grad():
            if config is None:
                action = anchor.distribution(torch.tensor(obs[None])).mean.tanh()[0].numpy()
            elif model is None:
                action = np.zeros(RESIDUAL_SIZE,dtype=np.float32) if constant is None else np.asarray(constant,dtype=np.float32)
            else:
                mask = env.residual_mask()
                action,raw,lp,value = model.act(torch.tensor(obs[None]),deterministic=not stochastic,
                    action_mask=torch.tensor(mask))
                action = action[0].numpy()
        following,reward,done,result = env.step(action)
        if stochastic:
            samples.append((obs,raw[0].numpy(),lp.item(),reward,done,value.item(),mask))
        obs = following
    result.setdefault("catch_to_orbit_s",None if env.controller.t_hand is None or env.capture_physics_time is None
        else env.controller.t_hand-env.capture_physics_time)
    result.setdefault("catch_to_release_s",None if result["release_time"] is None or result["capture_time"] is None
        else result["release_time"]-result["capture_time"])
    row = dict(seed=scenario["seed"],scenario=scenario,result=result,wall_s=time.perf_counter()-start,samples=samples)
    if config is not None:
        controls = np.asarray(env.control_trace)
        if not np.isfinite(controls).all() or controls.shape[1] != len(CONTROL_COLUMNS):
            raise RuntimeError("Invalid control trace")
        result["control_bounds"] = dict(min_psi=float(controls[:,-2].min()),max_psi=float(controls[:,-1].max()))
        result["effective_action_counts"] = np.sum([s[-1] for s in samples],axis=0).tolist() if samples else None
        if not stochastic:
            row.update(control_trace_columns=CONTROL_COLUMNS,control_trace=env.control_trace,
                exploration_trace_columns=EXPLORATION_COLUMNS,exploration_trace=env.exploration_trace)
    return row


def evaluate_rows(pool,manifest,anchor,config,checkpoint=None,stochastic=False,constant=None):
    jobs = [(scenario,str(anchor.resolve()),None if config is None else asdict(config),
        None if checkpoint is None else str(checkpoint.resolve()),stochastic,constant) for scenario in manifest]
    futures = [pool.submit(worker,job) for job in jobs]
    rows = []
    for future in as_completed(futures):
        row = future.result()
        rows.append(row)
        print(dict(completed=len(rows),total=len(jobs),seed=row["seed"],hit15=row["result"]["hit15"],
            abort=row["result"].get("safety_abort",False)),flush=True)
    return sorted(rows,key=lambda r:r["seed"])


def report(rows,baseline):
    result = previous_report(rows,baseline)
    result["checks"]["no_safety_abort"] = not any(r["result"].get("safety_abort",False) for r in rows)
    result["checks"]["absolute_joint_limit"] = max(r["result"]["max_joint_deg"] for r in rows) <= 36.
    result["eligible"] = all(result["checks"].values())
    result["primary_quality_improved"] = bool(result["eligible"] and result["task_nonregression"] and
        result.get("impact_peak_ratio",1e9)<=.97 and result.get("impact_impulse_ratio",1e9)<=.97)
    return result


def research_score(rows):
    summary = summarize(rows)
    error = summary["mean_landing_error_m"]
    return (-sum(r["result"].get("safety_abort",False) for r in rows),summary["hit15"],summary["captured"],
        summary["released"],-(error if error is not None else 1e9),
        -max(r["result"]["max_joint_deg"] for r in rows))


def train(args):
    anchor,payload = load_anchor(args.init)
    recipe = ImprovedRecipe(**payload["recipe"])
    config = ExplorationConfig(stage=args.stage)
    validation = scenarios(args.eval_episodes,args.eval_seed,args.design_seed+10000,recipe)
    forbidden = previous_seeds(args.out)
    training = set(range(args.seed,args.seed+args.updates*args.episodes_per_update))
    validating = {r["seed"] for r in validation["scenarios"]}
    if (training|validating)&forbidden or training&validating:
        raise ValueError("Training/validation seeds overlap existing experiments or each other")
    ppo = PPOConfig(gamma=.997,gae_lambda=.995,clip_ratio=.1,learning_rate=args.lr,
        update_epochs=3,minibatch_size=256,max_grad_norm=1.,target_kl=.02,entropy_coefficient=.001)
    contract = dict(schema=SCHEMA,source_hash=source_hash(),anchor_sha256=file_hash(args.init),
        config=asdict(config),action_names=list(ACTION_NAMES),ppo=asdict(ppo),critic_lr=args.critic_lr,
        seed=args.seed,design_seed=args.design_seed,episodes_per_update=args.episodes_per_update,
        eval_every=args.eval_every,validation_manifest=validation,budget_updates=args.updates,
        research_only=True,initialization="frozen_bc_plus_zero_residual")
    torch.manual_seed(args.design_seed)
    np.random.seed(args.design_seed)
    args.out.mkdir(parents=True,exist_ok=True)
    start,best_update,accepted_update = 1,0,None
    if args.resume:
        model,saved = load_policy(args.out/"ppo_latest.pt",contract["source_hash"],contract["anchor_sha256"])
        if saved["contract"] != contract:
            raise ValueError("Resume requires the same stage, budget, seeds and hyperparameters")
        start = saved["update"]+1
        best_update,best_score,accepted_update = saved["best_update"],tuple(saved["best_score"]),saved["accepted_update"]
        if start>args.updates:
            print("All requested updates already completed.",flush=True)
            return
    else:
        if any(args.out.iterdir()):
            raise ValueError("Output is not empty; use Resume or a fresh output directory")
        model = ResidualPolicy(anchor)
    optimizer = torch.optim.Adam([dict(params=[*model.actor.parameters(),model.log_std],lr=args.lr),
        dict(params=model.critic.parameters(),lr=args.critic_lr)])
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
    write_json(args.out/"run_config.json",contract)

    def save(name,update):
        save_policy(args.out/name,model,source_hash=contract["source_hash"],anchor_sha256=contract["anchor_sha256"],
            contract=contract,update=update,best_update=best_update,best_score=best_score,
            accepted_update=accepted_update,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),training_seed_end=args.seed+update*args.episodes_per_update)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init_worker) as pool:
        if args.resume:
            initial = read_json(args.out/"validation_0000.json")
            baseline,bc = initial["rows"],initial["bc_rows"]
        else:
            baseline = evaluate_rows(pool,validation["scenarios"],args.init,config)
            bc = evaluate_rows(pool,validation["scenarios"],args.init,None)
            initial_report = report(baseline,bc)
            write_json(args.out/"validation_0000.json",dict(update=0,rows=baseline,bc_rows=bc,vs_bc=initial_report))
            best_score = research_score(baseline)
            save("ppo_candidate.pt",0)
            save("ppo_latest.pt",0)
            print(dict(research_only=True,baseline_eligible=initial_report["eligible"],
                message="Exploration may proceed; candidate is NOT an accepted model."),flush=True)
        for update in range(start,args.updates+1):
            save("rollout_policy.pt",update-1)
            manifest = scenarios(args.episodes_per_update,args.seed+(update-1)*args.episodes_per_update,
                args.design_seed+update,recipe)
            rows = evaluate_rows(pool,manifest["scenarios"],args.init,config,args.out/"rollout_policy.pt",True)
            buffer = RolloutBuffer(sum(len(row["samples"]) for row in rows),OBS_DIM,RESIDUAL_SIZE)
            for row in rows:
                for sample in row.pop("samples"):
                    buffer.add(*sample)
            buffer.finish(0.,ppo)
            if np.any(buffer.action_masks[:buffer.size]):
                metrics = update_policy(model,optimizer,buffer,ppo)
            else:
                metrics = dict(skipped=True,reason="No active actions; all episodes missed the controlled phase")
            write_json(args.out/f"update_{update:04d}.json",dict(update=update,metrics=metrics,summary=summarize(rows),rows=rows))
            if update%args.eval_every==0 or update==args.updates:
                save("rollout_policy.pt",update)
                evaluated = evaluate_rows(pool,validation["scenarios"],args.init,config,args.out/"rollout_policy.pt")
                vs_zero,vs_bc = report(evaluated,baseline),report(evaluated,bc)
                score = research_score(evaluated)
                selected = score>tuple(best_score)
                if selected:
                    best_update,best_score = update,score
                    save("ppo_candidate.pt",update)
                accepted = (args.eval_episodes>=80 and candidate_eligible(vs_zero) and
                    vs_bc["eligible"] and vs_bc["task_nonregression"] and vs_zero["primary_quality_improved"])
                if accepted:
                    accepted_update = update
                    save("ppo_accepted.pt",update)
                write_json(args.out/f"validation_{update:04d}.json",dict(update=update,rows=evaluated,
                    vs_zero=vs_zero,vs_bc=vs_bc,research_selected=selected,accepted=accepted))
            save("ppo_latest.pt",update)
            print(dict(update=update,best_update=best_update,accepted_update=accepted_update,metrics=metrics),flush=True)
    write_json(args.out/"training_summary.json",dict(completed_updates=args.updates,
        training_episodes=args.updates*args.episodes_per_update,best_update=best_update,
        accepted_update=accepted_update,research_only=True,elapsed_s=time.perf_counter()-started))


def evaluate(args):
    _,payload = load_policy(args.checkpoint,source_hash(),file_hash(args.init))
    _,anchor_payload = load_anchor(args.init)
    config = ExplorationConfig(**payload["contract"]["config"])
    manifest = scenarios(args.episodes,args.seed,args.design_seed,ImprovedRecipe(**anchor_payload["recipe"]))
    contract = payload["contract"]
    forbidden = previous_seeds()
    forbidden.update(range(contract["seed"],contract["seed"]+contract["budget_updates"]*contract["episodes_per_update"]))
    forbidden.update(r["seed"] for r in contract["validation_manifest"]["scenarios"])
    identity = dict(source_hash=source_hash(),anchor_sha256=file_hash(args.init),checkpoint_sha256=file_hash(args.checkpoint),
        selected_update=payload["update"],manifest=manifest,research_only=True)
    path = args.out/"evaluation_contract.json"
    if path.exists():
        if read_json(path)!=identity:
            raise ValueError("Evaluation contract changed")
    elif forbidden & {r["seed"] for r in manifest["scenarios"]}:
        raise ValueError("Evaluation reuses development/training/validation seeds")
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(path,identity)
    groups = {}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init_worker) as pool:
        for name,cfg,checkpoint in (("bc",None,None),("zero",config,None),("candidate",config,args.checkpoint)):
            path = args.out/(name+"_episodes.json")
            groups[name] = read_json(path) if path.exists() else evaluate_rows(pool,manifest["scenarios"],args.init,cfg,checkpoint)
            write_json(path,groups[name])
    write_json(args.out/"comparison.json",dict(research_only=True,selected_update=payload["update"],
        vs_bc=report(groups["candidate"],groups["bc"]),vs_zero=report(groups["candidate"],groups["zero"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    command = sub.add_parser("explore")
    command.add_argument("--init",type=Path,default=DEFAULT_ANCHOR)
    command.add_argument("--out",type=Path,default=DEFAULT_OUT/"explore_throw")
    command.add_argument("--stage",choices=("catch","throw","joint"),default="throw")
    command.add_argument("--resume",action="store_true")
    for name,default in (("updates",10),("episodes-per-update",16),("eval-episodes",20),("eval-every",5),
            ("workers",8),("seed",18100001),("eval-seed",18200001),("design-seed",20260919),("threads",4)):
        command.add_argument("--"+name,type=int,default=default)
    command.add_argument("--lr",type=float,default=1e-4)
    command.add_argument("--critic-lr",type=float,default=3e-4)
    command = sub.add_parser("evaluate")
    command.add_argument("--init",type=Path,default=DEFAULT_ANCHOR)
    command.add_argument("--checkpoint",type=Path,default=DEFAULT_OUT/"explore_throw/ppo_candidate.pt")
    command.add_argument("--out",type=Path,default=DEFAULT_OUT/"explore_throw/evaluation_40")
    for name,default in (("episodes",40),("workers",8),("seed",18300001),("design-seed",20260920)):
        command.add_argument("--"+name,type=int,default=default)
    args = parser.parse_args()
    for name in ("workers","threads","eval_every"):
        if hasattr(args,name) and getattr(args,name)<1:
            parser.error(name+" must be positive")
    if args.command=="explore":
        if not 1<=args.updates<=50 or not 5<=args.episodes_per_update<=32 or not 5<=args.eval_episodes<=200:
            parser.error("Research budgets: 1-50 updates, 5-32 episodes/update, 5-200 validation episodes")
        if not np.isfinite([args.lr,args.critic_lr]).all() or min(args.lr,args.critic_lr)<=0:
            parser.error("Learning rates must be finite and positive")
    elif args.episodes<5:
        parser.error("Evaluation needs at least 5 episodes")
    torch.set_num_threads(getattr(args,"threads",1))
    (train if args.command=="explore" else evaluate)(args)


if __name__ == "__main__":
    main()
