import json
from pathlib import Path


root = Path("teacher_runs/v26_twin_mpc_ppo/stage3_hybrid_validation")
rows = [json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "episodes").glob("*.json"))]
captured = [row for row in rows if row["result"].get("captured")]
released = [row for row in captured if row["result"].get("released")]
hit = [row for row in released if row["result"].get("hit15")]
base_captured = [row for row in rows if row["baseline"].get("captured")]
base_released = [row for row in base_captured if row["baseline"].get("released")]
base_hit = [row for row in base_released if row["baseline"].get("hit15")]
takeover = [row for row in captured if row["mode"] == "v26_mpc"]
fallback = [row for row in captured if row["mode"] == "v15_fallback"]
positive = [row for row in takeover if row["result"].get("hit15") and
            row["paired_utility_delta"] > 0]
candidate_errors = [row["result"].get("landing_error_m") for row in rows
                    if row["result"].get("landing_error_m") is not None]
baseline_errors = [row["baseline"].get("landing_error_m") for row in rows
                   if row["baseline"].get("landing_error_m") is not None]
print(json.dumps(dict(episodes=len(rows), candidate=dict(captured=len(captured),
    released=len(released), hit15=len(hit)), baseline=dict(captured=len(base_captured),
    released=len(base_released), hit15=len(base_hit)),
    release_given_capture=len(released)/max(1, len(captured)),
    hit_given_release=len(hit)/max(1, len(released)),
    takeovers=len(takeover), fallbacks=len(fallback),
    takeover_fraction=len(takeover)/max(1, len(captured)),
    safe_positive=len(positive), positive_fraction=len(positive)/max(1, len(captured)),
    candidate_mean_error=sum(candidate_errors)/max(1, len(candidate_errors)),
    baseline_mean_error=sum(baseline_errors)/max(1, len(baseline_errors)),
    exact=sum(bool(row.get("prediction_exact")) for row in captured),
    grip_broken=sum(bool(row["result"].get("grip_broken")) for row in rows),
    max_joint=max((row["result"].get("max_joint_deg", 0.) for row in rows), default=0.)),
    sort_keys=True))
