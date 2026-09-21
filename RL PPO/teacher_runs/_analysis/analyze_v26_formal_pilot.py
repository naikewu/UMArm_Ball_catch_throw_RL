import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from teacher_rl import env as _bootstrap
from teacher_rl.contextual_env import episode_utility


root = Path("teacher_runs/v26_twin_mpc_ppo/stage2_causal_mpc_pilot")
report = json.loads((root / "pilot_report.json").read_text(encoding="utf-8"))
rows = [json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "episodes").glob("*.json"))]


def state(row, key):
    value = row[key]
    return (bool(value.get("captured")), bool(value.get("released")),
            bool(value.get("hit15")))


print("paired_outcomes", Counter((state(row, "result"), state(row, "baseline")) for row in rows))
print("splits")
for split in ("train", "validation"):
    part = [row for row in rows if row["scenario"]["split"] == split]
    candidate = Counter(state(row, "result") for row in part)
    baseline = Counter(state(row, "baseline") for row in part)
    print(split, len(part), "candidate", candidate, "baseline", baseline)

losses = [row for row in rows if row["baseline"].get("hit15") and
          not row["result"].get("hit15")]
wins = [row for row in rows if row["result"].get("hit15") and
        not row["baseline"].get("hit15")]
print("loss_seeds", [row["seed"] for row in losses])
print("win_seeds", [row["seed"] for row in wins])

for row in losses:
    branches = row.get("predicted_branches") or []
    released = sum(bool(branch["released"]) for branch in branches)
    hit = sum(bool(branch["hit15"]) for branch in branches)
    errors = [branch["terminal_error_m"] for branch in branches]
    print("loss", dict(seed=row["seed"], split=row["scenario"]["split"],
        action=row["action"], candidate=state(row, "result"),
        baseline=state(row, "baseline"), branches_released=released,
        branches_hit15=hit, branch_min_error=min(errors),
        candidate_error=row["result"].get("landing_error_m"),
        baseline_error=row["baseline"].get("landing_error_m"),
        candidate_release_s=row["result"].get("catch_to_release_s"),
        baseline_release_s=row["baseline"].get("catch_to_release_s"),
        utility_delta=row["paired_utility_delta"]))

captured = [row for row in rows if row["result"].get("captured")]
both_hit = [row for row in captured if row["result"].get("hit15") and
            row["baseline"].get("hit15")]
print("both_hit", len(both_hit))
for key in ("landing_error_m", "catch_to_release_s", "pressure_integral_psi_s",
            "contact_impulse_ns", "impact_weld_impulse_ns", "relative_capture_speed_m_s",
            "max_joint_deg"):
    candidate = [row["result"].get(key) for row in both_hit]
    baseline = [row["baseline"].get(key) for row in both_hit]
    pairs = [(a, b) for a, b in zip(candidate, baseline) if a is not None and b is not None]
    if pairs:
        a, b = np.asarray(pairs).T
        print("paired", key, dict(candidate_mean=float(a.mean()), baseline_mean=float(b.mean()),
            ratio=float(a.mean()/b.mean()) if b.mean() else None,
            improved=int(np.sum(a < b)), equal=int(np.sum(a == b)), n=len(a)))

print("utility", dict(mean=float(np.mean([row["paired_utility_delta"] for row in captured])),
    median=float(np.median([row["paired_utility_delta"] for row in captured])),
    positive=sum(row["paired_utility_delta"] > 0 for row in captured),
    zero=sum(row["paired_utility_delta"] == 0 for row in captured),
    negative=sum(row["paired_utility_delta"] < 0 for row in captured)))

by_action = defaultdict(list)
for row in captured:
    by_action[row["action"]].append(row)
for action in sorted(by_action):
    part = by_action[action]
    print("action", action, dict(n=len(part), released=sum(r["result"].get("released") for r in part),
        hit15=sum(r["result"].get("hit15") for r in part),
        positive=sum(r["paired_utility_delta"] > 0 and r["result"].get("hit15") for r in part),
        delta=float(np.mean([r["paired_utility_delta"] for r in part]))))

print("safety", dict(max_candidate_joint=max(row["result"].get("max_joint_deg", 0) for row in rows),
    max_baseline_joint=max(row["baseline"].get("max_joint_deg", 0) for row in rows),
    candidate_grip_broken=sum(bool(row["result"].get("grip_broken")) for row in rows),
    exact=sum(bool(row.get("prediction_exact")) for row in captured)))

# Development-only replay of the causal rule suggested by the failure audit.
# This same 100-scene set cannot certify the rule; it only determines whether
# the rule is worth implementing and testing on a new, untouched manifest.
hybrid = []
fallback_captured = 0
for row in rows:
    has_predicted_hit = any(branch.get("hit15") for branch in
                            (row.get("predicted_branches") or []))
    if row["result"].get("captured") and not has_predicted_hit:
        chosen = row["baseline"]
        fallback_captured += 1
    else:
        chosen = row["result"]
    hybrid.append((row, chosen))
hybrid_captured = [(row, chosen) for row, chosen in hybrid if chosen.get("captured")]
hybrid_released = [(row, chosen) for row, chosen in hybrid_captured if chosen.get("released")]
hybrid_hit = [(row, chosen) for row, chosen in hybrid_released if chosen.get("hit15")]
hybrid_delta = [episode_utility(chosen)-episode_utility(row["baseline"])
                for row, chosen in hybrid]
print("development_hybrid", dict(fallback_captured=fallback_captured,
    mpc_captured=len(hybrid_captured)-fallback_captured,
    captured=len(hybrid_captured), released=len(hybrid_released), hit15=len(hybrid_hit),
    release_given_capture=len(hybrid_released)/len(hybrid_captured),
    hit_given_release=len(hybrid_hit)/len(hybrid_released),
    safe_positive=sum(chosen.get("hit15") and delta > 0 for (_, chosen), delta in
                      zip(hybrid, hybrid_delta)),
    mean_utility_delta=float(np.mean(hybrid_delta))))
