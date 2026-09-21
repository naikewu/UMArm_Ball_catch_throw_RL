"""Probe V18 action wiring on one already-used development scenario."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher_rl.exploration_env import ExplorationConfig, CONTROL_COLUMNS, EXPLORATION_COLUMNS
from teacher_rl.exploration_rl import worker, source_hash, DEFAULT_ANCHOR
from teacher_rl.improved_rl import _init_worker, read_json, file_hash
from teacher_rl.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("teacher_runs/v18_online/smoke_throw"))
    args = parser.parse_args()
    contract = read_json(args.run/"run_config.json")
    scenario = contract["validation_manifest"]["scenarios"][0]
    config = ExplorationConfig()
    variants = {"request": [1.]*10, "veto": [0.]*9+[-1.]}
    jobs = [(scenario,str(DEFAULT_ANCHOR.resolve()),asdict(config),None,False,action)
        for action in variants.values()]
    with ProcessPoolExecutor(max_workers=2,initializer=_init_worker) as pool:
        rows = dict(zip(variants,pool.map(worker,jobs)))
    request, veto = rows["request"], rows["veto"]
    controls = np.asarray(request["control_trace"])
    trace = np.asarray(request["exploration_trace"])
    before = controls[:,0] < request["result"]["capture_time"]
    checks = dict(
        policy_released=request["result"]["release_decisions"]["policy_releases"]==1,
        veto_prevented_release=not veto["result"]["released"],
        catch_actions_masked=bool(np.all(controls[before,1:8]==0.)),
        force_changed=bool(np.ptp(trace[:,1])>.9),
        radius_changed=bool(np.ptp(trace[:,2])>.018),
        finite_controls=bool(np.isfinite(controls).all() and np.isfinite(trace).all()),
        pressure_bounded=bool(controls[:,-2].min()>=1. and controls[:,-1].max()<=30.),
        correct_columns=controls.shape[1]==len(CONTROL_COLUMNS) and trace.shape[1]==len(EXPLORATION_COLUMNS))
    summaries = {name:dict(result=row["result"],wall_s=row["wall_s"],
        force_range=[float(np.min(np.asarray(row["exploration_trace"])[:,1])),
            float(np.max(np.asarray(row["exploration_trace"])[:,1]))],
        radius_range=[float(np.min(np.asarray(row["exploration_trace"])[:,2])),
            float(np.max(np.asarray(row["exploration_trace"])[:,2]))]) for name,row in rows.items()}
    write_json(args.run/"control_probe.json",dict(source_hash=source_hash(),
        anchor_sha256=file_hash(DEFAULT_ANCHOR),seed=scenario["seed"],checks=checks,variants=summaries))
    print(dict(seed=scenario["seed"],checks=checks),flush=True)
    if not all(checks.values()):
        raise RuntimeError("V18 action wiring check failed")


if __name__ == "__main__":
    main()
