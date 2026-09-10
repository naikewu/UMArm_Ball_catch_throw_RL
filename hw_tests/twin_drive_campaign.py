"""The 2026-08-21 drive campaign, replayed on the twin: one board at 12 psi, alone.

WHY.  The twin's responsiveness had been compared with the arm through two
metrics that do not measure the same thing.  One was a joint's deflection per psi
of *commanded differential* under co-contraction (0.53-0.88 deg/psi on the arm).
The other was a single muscle against gravity with every other joint held
(0.33-425 deg/psi on the twin, the diagonal of the gravity stiffness).  The
campaign ``hw_tests/canarm_drive_campaign.py`` measured the like-for-like
quantity on the metal: every board pressurised alone at 12 psi against an arm
whose other 23 boards are held enabled at 0.5 psi (because ``0x110`` leaks),
settled 6 s, and the plate markers averaged for 1.5 s.  The dominant joint's
deflection is in ``hw_tests/results/axis_analysis_2026-08-21.json`` (committed)
and in ``report_canarm_axis_2026-08-21.md`` section 3.  This script gives the
twin the same command, through ``replay.rollout`` open loop, and reads the same
joint the same way.

HOW, stated so the numbers can be checked: 150 Hz edges; 6 s at rest (every
board 0.5 psi, enabled), then board *k* at 12 psi for 7.5 s; each deflection is
the mean ``q`` over the last 1.5 s of the step less the mean over the last 1.5 s
of the rest.  The twin is ``twin_params.load_twin_kwargs(mech=...)``.

WHAT THIS DOES NOT SHOW.  The campaign's arm started from wherever the previous
board left it after a 4 s relax, and the twin starts every board from rest, so
hysteresis in the metal is not reproduced.  The dominant joint's sign and size
are compared; the runner-up joints, which the campaign also moved, are reported
but not scored.  The campaign's plates were solved from markers with the
2026-08-21 azimuth calibration; the twin's ``q`` is exact.

    .venv\\Scripts\\python.exe hw_tests\\twin_drive_campaign.py [--mech PATH] [--workers 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
for _p in (WS, WS / "TLE_PCB"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

PSI = 12.0
REST_PSI = 0.5
RATE_HZ = 150.0
REST_S = 6.0
SETTLE_S = 6.0
SAMPLE_S = 1.5
RESULTS = WS / "hw_tests" / "results"
MEASURED = WS / "hw_tests" / "results" / "axis_analysis_2026-08-21.json"

_WORKER: dict = {}


def _init(mech, flow):
    from digital_twin import twin_params as TP

    _WORKER["kwargs"] = TP.load_twin_kwargs(flow=flow, mech=mech, log=lambda *_: None)


def board_step(base: int) -> dict:
    from digital_twin import replay as R

    n_rest = int(round(REST_S * RATE_HZ))
    n_step = int(round((SETTLE_S + SAMPLE_S) * RATE_HZ))
    n_sample = int(round(SAMPLE_S * RATE_HZ))
    n = n_rest + n_step
    t = np.arange(n) / RATE_HZ
    board_type = [R.VARIANT_TLE_DVP] * 8 + [R.VARIANT_7MM] * 16
    rest = np.array([R.pa_to_counts(REST_PSI * R.PA_PER_PSI, v) for v in board_type])
    target = np.tile(rest, (n, 1))
    k = base - 0x101
    target[n_rest:, k] = R.pa_to_counts(PSI * R.PA_PER_PSI, board_type[k])
    rec = R.recording_from_arrays(t, np.zeros((n, 12)), target, target, board_type)
    roll = R.rollout(rec, batched_actuator=True, **_WORKER["kwargs"])
    q = np.asarray(roll.q_rad)[:n]
    dq = np.degrees(q[n - n_sample:n].mean(axis=0) - q[n_rest - n_sample:n_rest].mean(axis=0))
    return {"base": base, "dq_deg": dq.tolist(),
            "final_psi": float(np.asarray(roll.p_pa)[n - 1, k] / R.PA_PER_PSI),
            "ctrl_min_n": float(roll.ctrl_min_n)}


def main(argv=None) -> int:
    from digital_twin import twin_params as TP

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mech", default=str(TP.DEFAULT_MECH))
    ap.add_argument("--flow", default=str(TP.DEFAULT_FLOW))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    measured = {int(r["base"], 16): r for r in
                json.loads(MEASURED.read_text(encoding="utf-8"))["actuator_map"]["rows"]}
    import multiprocessing as mp

    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(args.mech, args.flow)) as pool:
        rows = pool.map(board_step, sorted(measured))
    wall = time.perf_counter() - t0

    out = []
    print(f"twin: {args.mech}\n{'board':>6} {'joint':>5} {'arm deg':>8} {'twin deg':>9} "
          f"{'ratio':>6} {'twin argmax':>11} {'psi':>5}")
    for row in rows:
        m = measured[row["base"]]
        j = int(m["joint"])
        twin = float(row["dq_deg"][j])
        ratio = twin / float(m["dq_deg"])
        arg = int(np.argmax(np.abs(row["dq_deg"])))
        out.append({"board": f"0x{row['base']:03X}", "joint": j, "arm_dq_deg": m["dq_deg"],
                    "twin_dq_deg": twin, "ratio": ratio, "twin_argmax_joint": arg,
                    "twin_dq_all_deg": row["dq_deg"], "twin_final_psi": row["final_psi"],
                    "ctrl_min_n": row["ctrl_min_n"]})
        print(f"0x{row['base']:03X} {j:>5d} {m['dq_deg']:>+8.2f} {twin:>+9.2f} {ratio:>6.2f} "
              f"{arg:>11d} {row['final_psi']:>5.2f}")
    ratios = np.array([r["ratio"] for r in out])
    same_joint = sum(1 for r in out if r["twin_argmax_joint"] == r["joint"])
    same_sign = sum(1 for r in out if np.sign(r["twin_dq_deg"]) == np.sign(r["arm_dq_deg"]))
    abs_err = np.array([abs(r["twin_dq_deg"] - r["arm_dq_deg"]) for r in out])
    summary = {"median_ratio": float(np.median(ratios)),
               "ratio_range": [float(ratios.min()), float(ratios.max())],
               "per_segment_median_ratio": [float(np.median(ratios[i:i + 8])) for i in (0, 8, 16)],
               "same_dominant_joint": same_joint, "same_sign": same_sign,
               "mean_abs_error_deg": float(abs_err.mean()),
               "boards": len(out), "wall_s": wall}
    print(json.dumps(summary, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = RESULTS / f"twin_drive_campaign_{time.strftime('%Y-%m-%d')}{tag}.json"
    path.write_text(json.dumps({"mech": os.path.abspath(args.mech), "flow": args.flow,
                                "method": {"psi": PSI, "rest_psi": REST_PSI, "rest_s": REST_S,
                                           "settle_s": SETTLE_S, "sample_s": SAMPLE_S},
                                "rows": out, "summary": summary}, indent=1),
                    encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
