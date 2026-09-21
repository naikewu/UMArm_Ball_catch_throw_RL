"""Development-only paired probe of wider calibrated release tolerances.

The scenarios are deliberately taken from the outcome model's selection split.
They may select a tolerance, but can never be reported as independent evidence.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path

from teacher_rl.contextual_env import base_config
from teacher_rl.contextual_model import load_model
from teacher_rl.contextual_warm_sweep import worker, source_hash as warm_source_hash
from teacher_rl.data import write_json
from teacher_rl.envelope_teacher_rl import report
from teacher_rl.improved_rl import _init_worker, file_hash, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[.04, .06, .08])
    args = parser.parse_args()
    _, model_payload = load_model(args.checkpoint)
    selected_seeds = set(model_payload["contract"]["selection_seeds"])
    actions, scenarios, baselines, contracts = None, {}, {}, []
    for directory in args.data:
        contract = read_json(directory / "sweep_contract.json")
        contracts.append(contract)
        if actions is None:
            actions = contract["actions"]
        elif contract["actions"] != actions:
            raise ValueError("Tolerance probe needs one shared action catalogue")
        for scenario in contract["manifest"]["scenarios"]:
            if scenario["seed"] in selected_seeds:
                scenarios[scenario["seed"]] = scenario
                path = directory / "episodes" / f"{scenario['seed']}_bc.json"
                baselines[scenario["seed"]] = read_json(path)
    if set(scenarios) != selected_seeds:
        raise ValueError("Could not reconstruct every model-selection scenario")
    source = hashlib.sha256((warm_source_hash() + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()
    identity = dict(purpose="DEVELOPMENT TOLERANCE SELECTION ONLY; NOT INDEPENDENT",
        source_hash=source, checkpoint_sha256=file_hash(args.checkpoint),
        calibration_sha256=file_hash(args.calibration), selection_seeds=sorted(selected_seeds),
        data_contract_sha256=[file_hash(directory / "sweep_contract.json") for directory in args.data],
        tolerances=args.tolerances, actions=actions)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "tolerance_probe_contract.json"
    if contract_path.exists() and read_json(contract_path) != identity:
        raise ValueError("Tolerance probe changed; use a new output directory")
    write_json(contract_path, identity)
    jobs, rows = [], {}
    anchor = Path("teacher_runs/v15_improved/bc_v1_formal/bc_best.pt").resolve()
    for tolerance in args.tolerances:
        config = asdict(base_config(tolerance=tolerance))
        for seed, scenario in scenarios.items():
            for index, action in enumerate(actions):
                key = f"tol_{int(round(tolerance*100)):02d}_{seed}_a{index:03d}"
                path = args.out / "episodes" / f"{key}.json"
                if path.exists():
                    rows[key] = read_json(path)
                else:
                    job = (scenario, str(anchor), str(args.calibration.resolve()), config, action)
                    jobs.append((key, path, job))
    (args.out / "episodes").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(worker, job): (key, path) for key, path, job in jobs}
        total = len(rows) + len(jobs)
        for future in as_completed(futures):
            key, path = futures[future]
            row = future.result()
            write_json(path, row)
            rows[key] = row
            if len(rows) % 10 == 0 or len(rows) == total:
                print(dict(tolerance_probe_completed=len(rows), total=total), flush=True)
    baseline = [baselines[seed] for seed in sorted(selected_seeds)]
    reports = {}
    for tolerance in args.tolerances:
        oracle = []
        for seed in sorted(selected_seeds):
            candidates = [rows[f"tol_{int(round(tolerance*100)):02d}_{seed}_a{i:03d}"]
                          for i in range(len(actions))]
            oracle.append(max(candidates, key=lambda row: row["utility"]))
        reports[f"{tolerance:.3f}"] = report(oracle, baseline)
    write_json(args.out / "tolerance_comparison.json", dict(contract=identity, reports=reports))
    print({key: dict(summary=value["summary"], baseline=value["baseline"], checks=value["checks"])
           for key, value in reports.items()}, flush=True)


if __name__ == "__main__":
    main()
