"""Screen on used development scenes, then validate one frozen continuous configuration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teacher_rl.continuous_env import ContinuousConfig
from teacher_rl.continuous_rl import (DEFAULT_ANCHOR, DEFAULT_OUT, evaluate_rows, previous_seeds,
    report, scenarios, selection_score, source_hash)
from teacher_rl.data import write_json
from teacher_rl.improved_rl import _init_worker, file_hash, load_anchor, read_json
from teacher_rl.improved_teacher import ImprovedRecipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("screen","validate"))
    parser.add_argument("--config",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=40)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seed",type=int,default=4300001)
    args = parser.parse_args()
    _,anchor_payload = load_anchor(DEFAULT_ANCHOR)
    recipe = ImprovedRecipe(**anchor_payload["recipe"])
    if args.mode == "screen":
        original = DEFAULT_OUT/"rl_smoke/validation_0000.json"
        baseline = read_json(original)["v15_bc_rows"]
        manifest = dict(scenarios=[r["scenario"] for r in baseline],episodes=len(baseline))
        variants = {
            "ball24_strict":ContinuousConfig(release_ball_frames=24,align_release_clock=True,
                release_azimuth_bias_deg=0.,release_tolerance_m=.04),
            "steady8_legacy":ContinuousConfig(spinup_s=8.,release_min_s=8.5),
            "steady8_ball24":ContinuousConfig(spinup_s=8.,release_min_s=8.5,
                release_ball_frames=24,align_release_clock=True,release_azimuth_bias_deg=0.,release_tolerance_m=.04)}
        origin = dict(baseline_path=str(original),baseline_sha256=file_hash(original))
    else:
        if args.config is None or args.episodes<5:
            parser.error("validation requires --config and episodes >=5")
        selection = read_json(args.config)
        if (selection["pilot_source_hash"] != source_hash() or not selection["eligible"] or
                selection["anchor_sha256"] != file_hash(DEFAULT_ANCHOR)):
            raise ValueError("Candidate has not passed current development checks")
        manifest = scenarios(args.episodes,args.seed,20261024,recipe)
        # Existing output must match the full identity below; its own seeds are not a new trial.
        if (not (args.out/"pilot_contract.json").exists() and
                {r["seed"] for r in manifest["scenarios"]}&previous_seeds()):
            raise ValueError("Fresh validation scenarios overlap past development")
        variants = {selection["variant"]:ContinuousConfig(**selection["config"])}
        baseline = None
        origin = dict(candidate_path=str(args.config),candidate_sha256=file_hash(args.config))
    identity = dict(source_hash=source_hash(),anchor_sha256=file_hash(DEFAULT_ANCHOR),manifest=manifest,
        variants={name:asdict(config) for name,config in variants.items()},**origin)
    args.out.mkdir(parents=True,exist_ok=True)
    contract = args.out/"pilot_contract.json"
    if contract.exists() and read_json(contract) != identity:
        raise ValueError("Output contract changed")
    write_json(contract,identity)
    comparisons = {}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init_worker) as pool:
        if baseline is None:
            path = args.out/"v15_bc_episodes.json"
            baseline = read_json(path) if path.exists() else evaluate_rows(pool,manifest["scenarios"],DEFAULT_ANCHOR,None)
        write_json(args.out/"v15_bc_episodes.json",baseline)
        for name,config in variants.items():
            print("variant "+name,flush=True)
            path = args.out/(name+"_episodes.json")
            rows = read_json(path) if path.exists() else evaluate_rows(pool,manifest["scenarios"],DEFAULT_ANCHOR,config)
            write_json(path,rows)
            result = report(rows,baseline)
            comparisons[name] = dict(config=asdict(config),report=result)
            write_json(args.out/"comparison.json",comparisons)
            print(dict(variant=name,summary=result["summary"],eligible=result["eligible"],
                nonregression=result["task_nonregression"],checks=result["checks"]),flush=True)
    candidates = [(name,data) for name,data in comparisons.items() if
        data["report"]["eligible"] and data["report"]["task_nonregression"]]
    if candidates:
        name,data = max(candidates,key=lambda item:selection_score(item[1]["report"]))
        write_json(args.out/"selected_config.json",dict(variant=name,config=data["config"],
            pilot_source_hash=source_hash(),anchor_sha256=file_hash(DEFAULT_ANCHOR),eligible=True,
            development_seeds=[r["seed"] for r in manifest["scenarios"]],episodes=len(baseline),
            primary_quality_improved=data["report"]["primary_quality_improved"]))
        print("Selected: "+name,flush=True)
    else:
        print("No candidate passed; no formal training authorized by the quality gates.",flush=True)


if __name__ == "__main__":
    main()
