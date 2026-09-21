"""Read-only progress summary for V23 experiment directories."""
import json
from pathlib import Path

import numpy as np

from teacher_rl.improved_rl import read_json
from teacher_rl.release_calibration import RidgeLandingCalibration

root = Path("teacher_runs/v23_contextual")
output = {}
for name in ("sweep_v1", "warm_sweep_v1", "warm_calibration_v1"):
    paths = sorted((root / name / "episodes").glob("*.json"))
    rows = [read_json(path) for path in paths if ".tmp." not in path.name]
    output[name] = dict(completed=len(rows), captured=sum(row["result"]["captured"] for row in rows),
        released=sum(row["result"]["released"] for row in rows),
        hit15=sum(row["result"]["hit15"] for row in rows))
    if name == "warm_calibration_v1":
        old = RidgeLandingCalibration.from_dict(read_json(Path(
            "teacher_runs/v22_trajectory_envelope/campaign_v1/release_calibration_v22.json")))
        raw_errors, old_errors = [], []
        for row in rows:
            result = row["result"]
            command = result.get("calibration_probe", {}).get("command")
            if command is not None and result.get("landing_xy") is not None:
                actual = np.asarray(result["landing_xy"])
                raw = np.asarray(command["raw_landing_xy"])
                raw_errors.append(np.linalg.norm(raw - actual))
                old_errors.append(np.linalg.norm(old.predict_landing(raw, command["features"]) - actual))
        output[name].update(raw_mean_m=float(np.mean(raw_errors)) if raw_errors else None,
            old_calibration_mean_m=float(np.mean(old_errors)) if old_errors else None)
for name in ("pilot_v1", "warm_pilot_v1"):
    record = root / name / "gate_result.json"
    if record.exists():
        data = read_json(record)
        output[name] = dict(passed=data["passed"], summary=data["report"]["summary"],
            failed_checks=[key for key, value in data["report"]["checks"].items() if not value])
    else:
        output[name] = {kind: len(list((root / name / kind).glob("episode_*.json"))) for kind in ("v15_bc", "candidate")}
print(json.dumps(output, indent=2))
