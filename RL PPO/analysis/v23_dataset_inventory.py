"""Audit combined V23 paired sweeps without treating repeated actions as new contexts."""
import argparse
import json
from pathlib import Path

from teacher_rl.data import write_json
from teacher_rl.improved_rl import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds, completed, action_rows, baseline_rows = set(), 0, [], []
    expected = 0
    sources = []
    for directory in args.data:
        contract = read_json(directory / "sweep_contract.json")
        scenarios, actions = contract["manifest"]["scenarios"], contract["actions"]
        current = {row["seed"] for row in scenarios}
        if seeds & current:
            raise ValueError("Combined sweeps contain repeated scenario seeds")
        seeds.update(current)
        expected += len(scenarios) * (len(actions) + 1)
        for scenario in scenarios:
            seed = scenario["seed"]
            baseline = directory / "episodes" / f"{seed}_bc.json"
            if baseline.exists():
                baseline_rows.append(read_json(baseline))
                completed += 1
            for index in range(len(actions)):
                path = directory / "episodes" / f"{seed}_a{index:03d}.json"
                if path.exists():
                    action_rows.append(read_json(path))
                    completed += 1
        sources.append(dict(directory=str(directory.resolve()), scenarios=len(scenarios),
            actions=len(actions), expected_rows=len(scenarios) * (len(actions) + 1)))
    contexts = {}
    for row in action_rows:
        decision = row["result"].get("trajectory_decision")
        if decision is not None:
            contexts.setdefault(row["seed"], decision["context"])
    near_release = {"le_4cm": 0, "4_to_6cm": 0, "6_to_8cm": 0,
                    "8_to_15cm": 0, "over_15cm": 0}
    for row in action_rows:
        if row["result"]["released"]:
            continue
        value = row["result"].get("calibrated_release", {}).get("min_calibrated_error_m")
        if value is None:
            continue
        key = ("le_4cm" if value <= .04 else "4_to_6cm" if value <= .06 else
               "6_to_8cm" if value <= .08 else "8_to_15cm" if value <= .15 else "over_15cm")
        near_release[key] += 1
    result = dict(sources=sources, expected_rows=expected, completed_rows=completed,
        independent_scenarios=len(seeds), captured_contexts=len(contexts),
        action_outcomes=len(action_rows), baseline_outcomes=len(baseline_rows),
        action_released=sum(row["result"]["released"] for row in action_rows),
        action_hit15=sum(row["result"]["hit15"] for row in action_rows),
        unreleased_min_predicted_error_bins=near_release,
        baseline_captured=sum(row["result"]["captured"] for row in baseline_rows),
        complete=completed == expected)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
