"""Read-only V18 run inspection, outside the frozen training fingerprint."""
import argparse
import json
from pathlib import Path
from pickle import UnpicklingError

from .exploration_model import load_policy
from .exploration_rl import DEFAULT_OUT, DEFAULT_ANCHOR, source_hash
from .improved_rl import read_json, file_hash


def checkpoint_info(path, fingerprint, anchor_hash):
    _, payload = load_policy(path, fingerprint, anchor_hash)
    return dict(update=payload["update"], best_update=payload["best_update"],
        accepted_update=payload["accepted_update"])


def inspect_run(directory, anchor=DEFAULT_ANCHOR):
    directory = Path(directory)
    result = dict(directory=str(directory.resolve()), state="not_started", issues=[],
        note="Disk state only; this does not detect running processes.")
    path = directory/"run_config.json"
    if not path.exists():
        return result
    contract = read_json(path)
    fingerprint, anchor_hash = source_hash(), file_hash(anchor)
    result.update(stage=contract["config"]["stage"], budget_updates=contract["budget_updates"],
        episodes_per_update=contract["episodes_per_update"],
        source_matches=contract["source_hash"]==fingerprint,
        anchor_matches=contract["anchor_sha256"]==anchor_hash, checkpoints={})
    if not result["source_matches"] or not result["anchor_matches"]:
        result["state"] = "source_or_anchor_mismatch"
        return result
    for name in ("latest", "candidate", "accepted"):
        path = directory/f"ppo_{name}.pt"
        if path.exists():
            try:
                result["checkpoints"][name] = checkpoint_info(path, fingerprint, anchor_hash)
            except (ValueError, RuntimeError, KeyError, EOFError, OSError, UnpicklingError) as exc:
                result["issues"].append(f"{path.name}: {exc}")
    latest = result["checkpoints"].get("latest")
    result["state"] = "initializing" if latest is None else "resumable"
    if latest is None and (directory/"training_summary.json").exists():
        result["issues"].append("Training summary exists without a readable latest checkpoint.")
    if latest is not None:
        completed = latest["update"]
        result.update(completed_updates=completed,
            training_episodes=completed*contract["episodes_per_update"])
        required = ["validation_0000.json", "ppo_candidate.pt"]
        required += [f"update_{i:04d}.json" for i in range(1,completed+1)]
        required += [f"validation_{i:04d}.json" for i in range(1,completed+1)
            if i%contract["eval_every"]==0 or i==contract["budget_updates"]]
        result["issues"] += ["Missing "+name for name in required if not (directory/name).exists()]
        candidate = result["checkpoints"].get("candidate")
        if candidate and candidate["update"] != latest["best_update"]:
            result["issues"].append("Candidate and latest selection disagree; update may have been interrupted.")
        accepted = result["checkpoints"].get("accepted")
        if latest["accepted_update"] is not None and (accepted is None or
                accepted["update"] != latest["accepted_update"]):
            result["issues"].append("Accepted checkpoint and latest selection disagree.")
        summary_path = directory/"training_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            result["summary"] = summary
            if (summary["completed_updates"]==completed==contract["budget_updates"] and
                    summary["best_update"]==latest["best_update"] and
                    summary["accepted_update"]==latest["accepted_update"]):
                result["state"] = "completed"
            else:
                result["issues"].append("Training summary and checkpoint disagree.")
        elif completed==contract["budget_updates"]:
            result["state"] = "updates_complete_summary_missing"
        if candidate and candidate["update"]==0:
            result["warning"] = "Candidate is the zero-residual baseline, NOT a learned improvement. Evaluate Latest to inspect the trained model."
    result["evaluations"] = []
    for path in sorted(directory.glob("evaluation*/evaluation_contract.json")):
        identity = read_json(path)
        result["evaluations"].append(dict(directory=path.parent.name,
            selected_update=identity["selected_update"],
            complete=(path.parent/"comparison.json").exists()))
    if result["issues"]:
        result["state"] = "inconsistent"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT/"explore_throw")
    parser.add_argument("--init",type=Path,default=DEFAULT_ANCHOR)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.out,args.init),indent=2,ensure_ascii=True))


if __name__ == "__main__":
    main()
