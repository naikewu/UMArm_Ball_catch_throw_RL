"""Read saved paired sensitivity results and verify delivered pressure bounds."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    def read(name):
        return json.loads((args.directory/name).read_text(encoding="utf-8"))

    comparison = read("comparison.json")
    result = {}
    for name, entry in comparison.items():
        rows = read(name+"_episodes.json")
        captured = [r["result"] for r in rows if r["result"]["captured"]]
        report = entry["vs_v15_bc"]
        item = dict(episodes=len(rows), captured=report["summary"]["captured"],
            hit15=report["summary"]["hit15"], landing_cm=100*report["summary"]["mean_landing_error_m"]
                if report["summary"]["mean_landing_error_m"] is not None else None,
            failed_guards=[k for k,v in report["checks"].items() if not v],
            max_contact_n=report["summary"]["max_contact_peak_n"],
            max_joint_deg=max(r["result"]["max_joint_deg"] for r in rows))
        for key in ("impact_weld_peak_n", "impact_weld_impulse_ns", "buffer_weld_peak_n", "buffer_weld_impulse_ns"):
            values = [r[key] for r in captured if key in r]
            item[key] = float(np.mean(values)) if values else None
        if all("control_trace" in r for r in rows):
            columns = rows[0]["control_trace_columns"]
            trace = np.concatenate([np.asarray(r["control_trace"]) for r in rows])
            if not np.isfinite(trace).all():
                raise RuntimeError("Nonfinite delivered-control trace")
            item["control_ranges"] = {key:[float(trace[:,i].min()),float(trace[:,i].max())]
                for i,key in enumerate(columns) if key in ("catch_pressure_delta_psi","catch_kp_scale",
                "catch_kd_scale","buffer_pressure_delta_psi","buffer_kd_scale",
                "command_pressure_min_psi","command_pressure_max_psi")}
            if (item["control_ranges"]["command_pressure_min_psi"][0] < 1.-1e-8 or
                    item["control_ranges"]["command_pressure_max_psi"][1] > 30.+1e-8):
                raise RuntimeError("Delivered pressure outside original limits")
        result[name] = item
    output = args.directory/"control_analysis.json"
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("variant  hit15  landing_cm  impact_N  impact_Ns  buffer_Ns  contact_max_N")
    for name,item in result.items():
        print(name, item["hit15"], item["landing_cm"], item["impact_weld_peak_n"],
            item["impact_weld_impulse_ns"],item["buffer_weld_impulse_ns"],item["max_contact_n"])


if __name__ == "__main__":
    main()
