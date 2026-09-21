"""Join saved strict and relaxed development outcomes on one scenario set."""
import argparse
from pathlib import Path

from teacher_rl.contextual_model import load_model
from teacher_rl.data import write_json
from teacher_rl.envelope_teacher_rl import report
from teacher_rl.improved_rl import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    model, payload = load_model(args.checkpoint)
    seeds = sorted(payload["contract"]["selection_seeds"])
    actions, strict, baseline = None, {}, {}
    for directory in args.data:
        contract = read_json(directory / "sweep_contract.json")
        actions = contract["actions"] if actions is None else actions
        for scenario in contract["manifest"]["scenarios"]:
            seed = scenario["seed"]
            if seed not in seeds:
                continue
            baseline[seed] = read_json(directory / "episodes" / f"{seed}_bc.json")
            strict[seed] = [read_json(directory / "episodes" / f"{seed}_a{i:03d}.json")
                            for i in range(len(actions))]
    strict_oracle = [max(strict[seed], key=lambda row: row["utility"]) for seed in seeds]
    comparison = read_json(args.probe / "tolerance_comparison.json")
    reports = {"0.040": report(strict_oracle, [baseline[seed] for seed in seeds]),
               **comparison["reports"]}
    selected_indices = {}
    for seed in seeds:
        context = strict[seed][0]["result"]["trajectory_decision"]["context"]
        selected_indices[seed] = model.decide(context)[1]["index"]
    policy_rows = {"0.040": [strict[seed][selected_indices[seed]] for seed in seeds]}
    for tolerance in (.06, .08):
        code = int(round(tolerance * 100))
        policy_rows[f"{tolerance:.3f}"] = [read_json(args.probe / "episodes" /
            f"tol_{code:02d}_{seed}_a{selected_indices[seed]:03d}.json") for seed in seeds]
    policy_reports = {key: report(rows, [baseline[seed] for seed in seeds])
                      for key, rows in policy_rows.items()}
    compact = {key: dict(captured=value["summary"]["captured"],
        released=value["summary"]["released"], hit15=value["summary"]["hit15"],
        mean_landing_error_m=value["summary"]["mean_landing_error_m"],
        max_joint_deg=value["max_joint_deg"], checks=value["checks"])
        for key, value in reports.items()}
    policy_compact = {key: dict(released=value["summary"]["released"],
        hit15=value["summary"]["hit15"],
        mean_landing_error_m=value["summary"]["mean_landing_error_m"], checks=value["checks"])
        for key, value in policy_reports.items()}
    output = dict(purpose="DEVELOPMENT MODEL-SELECTION DATA; NOT INDEPENDENT",
        selection_seeds=seeds, selected_action_indices=selected_indices,
        oracle_reports=reports, oracle_compact=compact,
        policy_reports=policy_reports, policy_compact=policy_compact)
    write_json(args.probe / "tolerance_summary.json", output)
    print(dict(oracle=compact, learned_policy=policy_compact))


if __name__ == "__main__":
    main()
