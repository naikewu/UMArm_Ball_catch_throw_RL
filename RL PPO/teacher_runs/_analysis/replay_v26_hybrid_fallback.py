from dataclasses import asdict
import json
from pathlib import Path

from teacher_rl import env as _bootstrap
from teacher_rl.twin_hybrid_gate import worker
from teacher_rl.twin_mpc_ppo import DEFAULT_CALIBRATION
from teacher_rl.buffered_rl import DEFAULT_ANCHOR
from teacher_rl.twin_residual_env import TwinResidualConfig


development = json.loads(Path(
    "teacher_runs/v26_twin_mpc_ppo/stage2_causal_mpc_pilot/pilot_report.json"
).read_text(encoding="utf-8"))
scenario = next(row for row in development["manifest"]["scenarios"]
                if row["seed"] == 33100047)
output = Path("teacher_runs/v26_twin_mpc_ppo_smoke/fallback_replay_33100047.json")
row = worker((scenario, str(DEFAULT_ANCHOR.resolve()),
    str(DEFAULT_CALIBRATION.resolve()), asdict(TwinResidualConfig()), str(output.resolve())))
print(json.dumps(dict(seed=row["seed"], mode=row["mode"],
    fallback_reason=row["fallback_reason"], high_level_action=row["high_level_action"],
    candidate_branch_hits=sum(branch["hit15"] for branch in row["predicted_branches"]),
    captured=row["result"]["captured"], released=row["result"]["released"],
    hit15=row["result"]["hit15"], exact_v15=row["result"] == row["baseline"]),
    sort_keys=True))
