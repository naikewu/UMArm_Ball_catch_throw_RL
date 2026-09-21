"""Verify V17 zero-residual execution against archived V16 paired scenarios."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher_rl.buffered_env import BufferedConfig
from teacher_rl.buffered_rl import DEFAULT_ANCHOR, source_hash, worker
from teacher_rl.continuous_rl import source_hash as v16_source_hash
from teacher_rl.data import write_json
from teacher_rl.improved_rl import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4300002])
    parser.add_argument("--out", type=Path, default=Path("teacher_runs/v17_buffered/zero_equivalence.json"))
    args = parser.parse_args()
    directory = Path("teacher_runs/v16_continuous/continuous_holdout_v1")
    contract = read_json(directory/"pilot_contract.json")
    if contract["source_hash"] != v16_source_hash():
        raise ValueError("Archived V16 sources differ; cannot use archived golden results")
    rows = read_json(directory/"ball24_strict_episodes.json")
    selected = [r for r in rows if r["seed"] in args.seeds]
    if {r["seed"] for r in selected} != set(args.seeds):
        raise ValueError("Missing golden scenario")
    comparisons = []
    for golden in selected:
        cfg = BufferedConfig(**golden["result"]["continuous_config"])
        row = worker((golden["scenario"], str(DEFAULT_ANCHOR.resolve()), asdict(cfg), None,
            False, golden["seed"], None))
        checks = {}
        for key in ("captured", "released", "hit15", "landing_error_m", "capture_time", "release_time",
                    "impact_weld_peak_n", "impact_weld_impulse_ns", "catch_to_orbit_s"):
            a,b = row["result"][key],golden["result"][key]
            checks[key] = bool(a is b if a is None or b is None else np.isclose(a,b,rtol=1e-6,atol=1e-6))
        comparisons.append(dict(seed=golden["seed"], checks=checks, result=row["result"],
            golden=golden["result"], all_passed=all(checks.values())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out,dict(source_hash=source_hash(), v16_source_hash=v16_source_hash(), comparisons=comparisons))
    for row in comparisons:
        print(dict(seed=row["seed"],checks=row["checks"]),flush=True)
    if not all(row["all_passed"] for row in comparisons):
        raise RuntimeError("Zero-residual equivalence failed; inspect saved metrics")


if __name__ == "__main__":
    main()
