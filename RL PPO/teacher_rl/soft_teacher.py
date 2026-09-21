"""V13: paired teacher search, fresh demonstrations, BC, and closed-loop evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .__main__ import clone, emit
from .data import rollout, write_json, load_dataset, episode_manifests
from .env import TaskConfig
from .model import IntentPolicy, load_checkpoint
from .soft_env import SoftRecipe, SoftTeacherEnv, SOFT_SCHEMA, soft_fingerprint


def candidates():
    base = SoftRecipe()
    return {
        "baseline": base,
        "match35": replace(base,match=.35),
        "match40": replace(base,match=.40),
        "soft2": replace(base,pressure_drop_psi=2.),
        "soft4": replace(base,pressure_drop_psi=4.),
        "buffer50": replace(base,deceleration_s=.5,brake_kd=6.),
        "soft2_match35": replace(base,pressure_drop_psi=2.,match=.35),
        "soft2_buffer50": replace(base,pressure_drop_psi=2.,deceleration_s=.5,brake_kd=6.),
        "soft2_gain85": replace(base,pressure_drop_psi=2.,catch_gain_scale=.85),
    }


def one_episode(job):
    name, recipe, seed, out, save_data, checkpoint, authority, source = job
    torch.set_num_threads(1)
    config = TaskConfig(authority=authority)
    identity = dict(schema=SOFT_SCHEMA, source_hash=source, name=name, recipe=recipe,
        seed=seed, config=asdict(config),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None)
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(out)
    directory.mkdir(parents=True,exist_ok=True)
    stem = directory / f"episode_{seed}_{key}"
    manifest, data_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if manifest.exists() and (not save_data or data_path.exists()):
        return json.loads(manifest.read_text(encoding="utf-8"))
    model = load_checkpoint(Path(checkpoint),expected_schema=SOFT_SCHEMA)[0] if checkpoint else None
    env = SoftTeacherEnv(SoftRecipe(**recipe), config)
    arrays, result = rollout(env,seed,model)
    metadata = dict(**identity,result=result,samples=len(arrays["observations"]),
        split="validation" if seed%5 == 0 else "train",data=data_path.name,
        events=env.plant.log_events,twin_provenance=env.kw.provenance)
    if save_data:
        arrays["pressure_schedule150"] = np.asarray(env.schedule_trace,dtype=np.float32)
        temporary = stem.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,**arrays)
        temporary.replace(data_path)
    write_json(manifest,metadata)
    return metadata


def batch(recipes, seeds, out, workers, *, save_data=False, checkpoint=None, authority=0.):
    source = soft_fingerprint()
    jobs = [(name,asdict(recipe),seed,str(out),save_data,str(checkpoint) if checkpoint else None,authority,source)
        for name,recipe in recipes.items() for seed in seeds]
    result = {name: [] for name in recipes}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_episode,job) for job in jobs]
        try:
            for i,future in enumerate(as_completed(futures),1):
                row = future.result()
                result[row["name"]].append(row)
                emit(dict(completed=i,total=len(jobs),candidate=row["name"],seed=row["seed"],
                    hit15=row["result"]["hit15"],weld_peak_n=row["result"]["weld_peak_n"]))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    for rows in result.values():
        rows.sort(key=lambda row:row["seed"])
    return result


def summarize(rows):
    records = [r["result"] for r in rows]
    landed = [r for r in records if r["captured"] and r["released"] and r["landing_error_m"] is not None]
    def mean(key):
        values = [r[key] for r in landed if r[key] is not None]
        return float(np.mean(values)) if values else None
    peaks = [r["weld_peak_n"] for r in landed]
    return dict(episodes=len(records),captured=sum(r["captured"] for r in records),
        released=sum(r["released"] for r in records),hit15=sum(r["hit15"] for r in records),
        mean_error_m=mean("landing_error_m"),mean_weld_n=mean("weld_peak_n"),
        p95_weld_n=float(np.percentile(peaks,95)) if peaks else None,
        mean_contact_impulse_ns=mean("contact_impulse_ns"),
        mean_impact_impulse_ns=mean("impact_weld_impulse_ns"),
        mean_relative_speed_m_s=mean("relative_capture_speed_m_s"),
        mean_buffer_m=mean("buffer_displacement_m"),
        mean_pressure_psi_s=mean("pressure_integral_psi_s"),
        max_contact_peak_n=max(r["contact_peak_n"] for r in records))


def paired_comparison(rows, baseline):
    a, b = summarize(rows), summarize(baseline)
    baseline_by_seed = {r["seed"]:r["result"] for r in baseline}
    if set(baseline_by_seed) != {r["seed"] for r in rows}:
        raise ValueError("paired comparison requires identical seed sets")
    lost_hits = [r["seed"] for r in rows if baseline_by_seed[r["seed"]]["hit15"] and not r["result"]["hit15"]]
    pairs = [(r["result"],baseline_by_seed[r["seed"]]) for r in rows
        if r["result"]["released"] and baseline_by_seed[r["seed"]]["released"]
        and r["result"]["landing_error_m"] is not None and baseline_by_seed[r["seed"]]["landing_error_m"] is not None]
    ratios = {}
    for key in ("weld_peak_n","impact_weld_impulse_ns","pressure_integral_psi_s"):
        ratios[key] = float(np.mean([x[key] for x,y in pairs])/max(np.mean([y[key] for x,y in pairs]),1e-9)) if pairs else None
    reasons = []
    for key in ("captured","released","hit15"):
        if a[key] < b[key]:
            reasons.append(f"{key} decreased")
    if lost_hits:
        reasons.append("lost baseline hit seeds")
    if a["mean_error_m"] is None or b["mean_error_m"] is None:
        reasons.append("no landed comparison")
    elif a["mean_error_m"] > b["mean_error_m"]+.01:
        reasons.append("mean landing error increased by more than 1 cm")
    if a["p95_weld_n"] is None or b["p95_weld_n"] is None or a["p95_weld_n"] > b["p95_weld_n"]*1.05:
        reasons.append("p95 weld force guard")
    if a["max_contact_peak_n"] > b["max_contact_peak_n"]+10.:
        reasons.append("contact peak increased by more than 10 N")
    if a["mean_contact_impulse_ns"] is not None and b["mean_contact_impulse_ns"] is not None:
        if a["mean_contact_impulse_ns"] > b["mean_contact_impulse_ns"]+.05:
            reasons.append("contact impulse increased")
    if ratios["pressure_integral_psi_s"] is None or ratios["pressure_integral_psi_s"] > 1.05:
        reasons.append("pressure proxy increased by more than 5 percent")
    quality = sum(w*ratios[k] for k,w in (("weld_peak_n",.5),("impact_weld_impulse_ns",.3),("pressure_integral_psi_s",.2))) if pairs else None
    improved = not reasons and quality is not None and quality <= .97
    return dict(summary=a,ratios=ratios,paired_successes=len(pairs),lost_hit_seeds=lost_hits,
        eligible=not reasons,improved=improved,quality_score=quality,reasons=reasons)


def search(args):
    recipes = candidates()
    screen_seeds = list(range(args.seed,args.seed+args.screen_episodes))
    valid_seeds = list(range(args.validation_seed,args.validation_seed+args.validation_episodes))
    if set(screen_seeds)&set(valid_seeds):
        raise ValueError("screening and validation seeds overlap")
    screen = batch(recipes,screen_seeds,args.out/"screen",args.workers)
    comparisons = {name:paired_comparison(rows,screen["baseline"]) for name,rows in screen.items()}
    order = sorted((name for name in recipes if name != "baseline"),
        key=lambda n:(not comparisons[n]["eligible"],comparisons[n]["quality_score"] if comparisons[n]["quality_score"] is not None else float("inf")))
    finalists = {name:recipes[name] for name in ["baseline",*order[:args.top]]}
    write_json(args.out/"screen_summary.json",comparisons)
    valid = batch(finalists,valid_seeds,args.out/"validation",args.workers)
    reports = {name:paired_comparison(rows,valid["baseline"]) for name,rows in valid.items()}
    winners = [n for n in finalists if n != "baseline" and reports[n]["improved"]]
    selected = min(winners,key=lambda n:reports[n]["quality_score"]) if winners else "baseline"
    payload = dict(schema=SOFT_SCHEMA,source_hash=soft_fingerprint(),selected=selected,
        recipe=asdict(recipes[selected]),improved=bool(winners),screen_seeds=screen_seeds,
        validation_seeds=valid_seeds,validation_episodes=args.validation_episodes,
        comparisons=reports,selection_note="exploratory validation; final test must use new seeds")
    write_json(args.out/"selected_recipe.json",payload)
    emit(payload)


def read_recipe(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema"] != SOFT_SCHEMA or data["source_hash"] != soft_fingerprint():
        raise ValueError("recipe source/schema changed; rerun search in a new output directory")
    return SoftRecipe(**data["recipe"]), data


def collect(args):
    recipe, payload = read_recipe(args.recipe)
    seeds = list(range(args.seed,args.seed+args.episodes))
    if set(seeds)&set(payload["screen_seeds"]+payload["validation_seeds"]):
        raise ValueError("demonstration seeds overlap search/validation")
    args.out.mkdir(parents=True,exist_ok=True)
    contract = dict(schema=SOFT_SCHEMA,source_hash=soft_fingerprint(),recipe=asdict(recipe),
        recipe_search=payload,authority=.25)
    contract_path = args.out/"dataset_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise ValueError("dataset recipe changed; use a new output directory")
    write_json(contract_path,contract)
    rows = batch({payload["selected"]:recipe},seeds,args.out,args.workers,save_data=True,authority=.25)
    write_json(args.out/"collection_summary.json",summarize(rows[payload["selected"]]))
    if not payload["improved"]:
        emit(dict(note="No improved teacher was validated; this dataset reproduces the baseline."))


def train_bc(args):
    contract = json.loads((args.data/"dataset_contract.json").read_text(encoding="utf-8"))
    if contract["schema"] != SOFT_SCHEMA or contract["source_hash"] != soft_fingerprint():
        raise ValueError("dataset contract changed; recollect into a new directory")
    for path in episode_manifests(args.data):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["recipe"] != contract["recipe"] or meta["checkpoint_sha256"] is not None:
            raise ValueError(f"mixed recipes or student trajectories: {path}")
    dataset = load_dataset(args.data,expected_schema=SOFT_SCHEMA,source_hash=soft_fingerprint())
    model = IntentPolicy()
    model.schema = SOFT_SCHEMA
    if (args.out/"bc_latest.pt").exists():
        raise ValueError("BC output already exists; choose a new --out")
    clone(SimpleNamespace(data=args.data,out=args.out,init=None,epochs=args.epochs,batch=256,lr=3e-4),
        dataset=dataset,model=model,checkpoint_extra=dict(soft_source_hash=soft_fingerprint(),
            recipe=contract["recipe"],task_config=asdict(TaskConfig()),dataset_contract=contract))


def evaluate(args):
    recipe, payload = read_recipe(args.recipe)
    if args.checkpoint:
        _, checkpoint = load_checkpoint(args.checkpoint,expected_schema=SOFT_SCHEMA)
        if checkpoint.get("recipe") != asdict(recipe) or checkpoint.get("soft_source_hash") != soft_fingerprint():
            raise ValueError("checkpoint and recipe/source do not match")
    seeds = list(range(args.seed,args.seed+args.episodes))
    used = set(payload["screen_seeds"]+payload["validation_seeds"])
    if args.checkpoint:
        used.update(int(Path(name).stem.split('_')[1]) for name in checkpoint["dataset"]["manifests"])
    if set(seeds)&used:
        raise ValueError("evaluation seeds overlap search, validation, or demonstration seeds")
    reference = batch({"baseline":SoftRecipe()},seeds,args.out/"baseline",args.workers)
    teacher = batch({"optimized_teacher":recipe},seeds,args.out/"teacher",args.workers)
    report = dict(baseline=summarize(reference["baseline"]),
        optimized_teacher=paired_comparison(teacher["optimized_teacher"],reference["baseline"]))
    if args.checkpoint:
        student = batch({"bc":recipe},seeds,args.out/"bc",args.workers,checkpoint=args.checkpoint,authority=.25)
        report["bc_vs_baseline"] = paired_comparison(student["bc"],reference["baseline"])
        report["bc_vs_optimized_teacher"] = paired_comparison(student["bc"],teacher["optimized_teacher"])
    write_json(args.out/"comparison.json",report)
    emit(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    s = sub.add_parser("search")
    s.add_argument("--screen-episodes",type=int,default=12)
    s.add_argument("--validation-episodes",type=int,default=24)
    s.add_argument("--seed",type=int,default=40001)
    s.add_argument("--validation-seed",type=int,default=50001)
    s.add_argument("--top",type=int,default=2)
    s.add_argument("--out",type=Path,default=Path("teacher_runs/v13/search"))
    for name in ("collect","evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--recipe",type=Path,default=Path("teacher_runs/v13/search/selected_recipe.json"))
        s.add_argument("--episodes",type=int,default=256 if name=="collect" else 96)
        s.add_argument("--seed",type=int,default=60001 if name=="collect" else 70001)
        s.add_argument("--out",type=Path,default=Path("teacher_runs/v13/dataset" if name=="collect" else "teacher_runs/v13/evaluation"))
        if name=="evaluate":
            s.add_argument("--checkpoint",type=Path)
    s = sub.add_parser("bc")
    s.add_argument("--data",type=Path,default=Path("teacher_runs/v13/dataset"))
    s.add_argument("--out",type=Path,default=Path("teacher_runs/v13/bc"))
    s.add_argument("--epochs",type=int,default=60)
    for name in ("search","collect","evaluate"):
        sub.choices[name].add_argument("--workers",type=int,default=4)
    args = p.parse_args()
    for name in ("screen_episodes","validation_episodes","top","episodes","workers","epochs"):
        if hasattr(args,name) and getattr(args,name)<=0:
            p.error(f"{name} must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(20260917)
    np.random.seed(20260917)
    {"search":search,"collect":collect,"bc":train_bc,"evaluate":evaluate}[args.command](args)


if __name__ == "__main__":
    main()
