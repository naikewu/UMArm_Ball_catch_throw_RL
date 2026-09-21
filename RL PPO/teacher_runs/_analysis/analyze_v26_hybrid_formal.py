import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from teacher_rl import env as _bootstrap


root = Path("teacher_runs/v26_twin_mpc_ppo/stage3_hybrid_validation")
report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
contract = json.loads((root / "validation_contract.json").read_text(encoding="utf-8"))
paths = sorted((root / "episodes").glob("*.json"))
rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
manifest = report["manifest"]["scenarios"]
expected = [int(row["seed"]) for row in manifest]
actual = [int(row["seed"]) for row in rows]
print("integrity", dict(files=len(paths), rows=len(rows), expected=len(expected),
    unique=len(set(actual)), seed_set_exact=set(actual) == set(expected),
    report_contract_exact=all(report.get(key) == value for key, value in contract.items())))


def counts(part, key):
    return dict(episodes=len(part), captured=sum(bool(row[key].get("captured")) for row in part),
        released=sum(bool(row[key].get("released")) for row in part),
        hit15=sum(bool(row[key].get("hit15")) for row in part))


for split in ("train", "validation"):
    part = [row for row in rows if row["scenario"]["split"] == split]
    captured = [row for row in part if row["result"].get("captured")]
    print("split", split, "candidate", counts(part, "result"),
        "baseline", counts(part, "baseline"),
        "takeover", sum(row["mode"] == "v26_mpc" for row in captured),
        "fallback", sum(row["mode"] == "v15_fallback" for row in captured),
        "positive", sum(row["mode"] == "v26_mpc" and row["paired_utility_delta"] > 0
                        for row in captured))

captured = [row for row in rows if row["result"].get("captured")]
takeovers = [row for row in captured if row["mode"] == "v26_mpc"]
fallbacks = [row for row in captured if row["mode"] == "v15_fallback"]
print("fallbacks", [dict(seed=row["seed"], split=row["scenario"]["split"],
    reason=row["fallback_reason"], branch_hits=sum(branch.get("hit15", False)
        for branch in (row.get("predicted_branches") or [])),
    result=(row["result"].get("captured"), row["result"].get("released"),
            row["result"].get("hit15")), exact=row["result"] == row["baseline"])
    for row in fallbacks])
print("high_level_actions", Counter(row["high_level_action"] for row in captured))

both = [row for row in rows if row["result"].get("hit15") and row["baseline"].get("hit15")]
for key in ("landing_error_m", "catch_to_release_s", "pressure_integral_psi_s",
            "contact_impulse_ns", "impact_weld_impulse_ns", "impact_weld_peak_n",
            "relative_capture_speed_m_s", "max_joint_deg"):
    pairs = [(row["result"].get(key), row["baseline"].get(key)) for row in both]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if pairs:
        candidate, baseline = np.asarray(pairs, dtype=float).T
        print("paired", key, dict(n=len(candidate), candidate=float(candidate.mean()),
            baseline=float(baseline.mean()), delta=float((candidate-baseline).mean()),
            ratio=(None if baseline.mean() == 0 else float(candidate.mean()/baseline.mean())),
            improved=int(np.sum(candidate < baseline)), equal=int(np.sum(candidate == baseline))))

deltas = np.asarray([row["paired_utility_delta"] for row in captured], dtype=float)
print("utility", dict(mean=float(deltas.mean()), median=float(np.median(deltas)),
    positive=int(np.sum(deltas > 0)), zero=int(np.sum(deltas == 0)),
    negative=int(np.sum(deltas < 0)), minimum=float(deltas.min()), maximum=float(deltas.max())))

for parameter in ("launch_speed", "launch_distance", "target_distance", "target_azimuth"):
    ordered = sorted(rows, key=lambda row: row["scenario"][parameter])
    print("parameter", parameter)
    for index, part in enumerate(np.array_split(np.asarray(ordered, dtype=object), 4)):
        part = list(part)
        part_captured = [row for row in part if row["result"].get("captured")]
        print(index, dict(low=part[0]["scenario"][parameter],
            high=part[-1]["scenario"][parameter], captured=len(part_captured),
            hit15=sum(row["result"].get("hit15") for row in part),
            takeover=sum(row["mode"] == "v26_mpc" for row in part_captured),
            positive=sum(row["mode"] == "v26_mpc" and row["paired_utility_delta"] > 0
                         for row in part_captured)))

print("safety", dict(max_joint=max(row["result"].get("max_joint_deg", 0.) for row in rows),
    grip_broken=sum(bool(row["result"].get("grip_broken")) for row in rows),
    exact=sum(bool(row.get("prediction_exact")) for row in captured)))
