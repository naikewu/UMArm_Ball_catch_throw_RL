"""Summarize saved V23 evidence; never launch, fit, or qualify a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from teacher_rl.contextual_model import load_model
from teacher_rl.contextual_training import runtime_hash
from teacher_rl.data import write_json
from teacher_rl.improved_rl import read_json


def brief(report):
    return dict(candidate=report["summary"], baseline=report["baseline"],
        max_joint_deg=report["max_joint_deg"],
        failed_checks=[name for name, passed in report["checks"].items() if not passed],
        task_nonregression=report["task_nonregression"],
        absolute_generalization_pass=report["absolute_generalization_pass"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("teacher_runs/v23_contextual"))
    args = parser.parse_args()
    root = args.root
    calibration = read_json(root / "warm_kernel_v1/release_calibration_v23_kernel.json")
    oracle = read_json(root / "warm_sweep_v1/oracle_diagnostic.json")
    sweep = read_json(root / "warm_sweep_v1/sweep_contract.json")
    gate = read_json(root / "warm_pilot_v1/gate_result.json")
    model, payload = load_model(root / "warm_teacher_v1/teacher.pt")
    history = read_json(root / "warm_teacher_v1/fit_history.json")
    unreleased = []
    for choice in oracle["choices"]:
        if choice["result"]["captured"] and not choice["result"]["released"]:
            rows = [read_json(root / "warm_sweep_v1/episodes" / f"{choice['seed']}_a{i:03d}.json") for i in range(len(sweep["actions"]))]
            errors = [row["result"]["calibrated_release"]["min_calibrated_error_m"] for row in rows]
            unreleased.append(dict(seed=choice["seed"], scenario=choice["scenario"],
                actual_target_xy=choice["result"]["target_xy"],
                best_predicted_error_m=min(value for value in errors if value is not None),
                all_actions_released=sum(row["result"]["released"] for row in rows)))
    decisions = [read_json(path)["decision"] for path in sorted((root / "warm_pilot_v1/candidate").glob("episode_*.json"))]
    counts = np.bincount([row["index"] for row in decisions if row is not None], minlength=len(payload["actions"]))
    status = dict(schema="v23_evidence_status", calibration=dict(accepted=calibration["accepted"],
        labels=calibration["deployment_fit_rows"], metrics=calibration["cross_validation"]["pooled"],
        failed_checks=[name for name, passed in calibration["checks"].items() if not passed]),
        development_oracle=dict(warning=oracle["warning"], **brief(oracle["report"])),
        no_release_for_any_sampled_action=unreleased,
        teacher=dict(checkpoint="warm_teacher_v1/teacher.pt", current_runtime=payload["contract"]["runtime_hash"] == runtime_hash(),
            train_contexts=len(payload["contract"]["training_seeds"]), selection_contexts=len(payload["contract"]["selection_seeds"]),
            parameters=sum(p.numel() for p in model.parameters()), best_validation_record=min(history, key=lambda row: row["validation_loss"]),
            pilot_action_counts=counts.tolist()),
        independent_pilot=dict(passed=gate["passed"], **brief(gate["report"])),
        qualified_teacher_files=[str(path.relative_to(root)) for path in root.rglob("qualified_teacher.json")],
        new_bc_files=[str(path.relative_to(root)) for path in root.rglob("bc_best.pt")],
        formal_ppo_runs=[str(path.relative_to(root)) for path in root.rglob("ppo_contract.json")],
        accepted_ppo_files=[str(path.relative_to(root)) for path in root.rglob("ppo_accepted.pt")])
    write_json(root / "v23_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
