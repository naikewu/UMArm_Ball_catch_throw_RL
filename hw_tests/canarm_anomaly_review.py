"""Decide whether the campaign's anomaly flags are faults or coincidences.

The live watcher is deliberately trigger-happy: it fires when a pair's
commanded differential moves by more than 6 psi, both boards report following
it, and the joint then moves less than 0.8 deg in 1.2 s.  That test is right for
*during* a session, when the point is to aim a camera at something while it is
still happening, and it is not enough to conclude anything, because a joint in a
serial chain can legitimately fail to move: gravity, the load of the segments
below it, and its own co-contraction can all hold it against a modest torque.

This script asks the question the watcher cannot.  For each joint it fits the
whole recording -- every commanded differential step and the joint motion that
followed -- and reports the **responsiveness**: how many degrees of joint angle
this arm gets per psi of commanded differential.  A joint that is genuinely
disconnected has a responsiveness near zero *everywhere*; a joint that merely
happened not to move at one instant has a normal responsiveness and one
outlier.  The two look identical in a single event and completely different
over twenty minutes.

The comparison is across joints rather than against a threshold, because
nothing here knows what this arm's responsiveness ought to be.  Twelve joints
measured the same way, and the question is whether one of them is unlike its
neighbours.

Usage::

    .venv\\Scripts\\python.exe hw_tests\\canarm_anomaly_review.py \\
        --session data\\session_20260910_013843
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from digital_twin import dataset as ds  # noqa: E402
from UMArm_KINEMATICS import canarm_actuators as ACT  # noqa: E402

#: Seconds after a commanded step over which the joint's response is measured.
#: Long enough for this arm's fill transient (the staircase's own step response
#: settles inside a second) and short enough that the next random target has
#: usually not arrived.
SETTLE_S = 0.8

#: Minimum commanded differential change, psi, for a step to count.  Below this
#: the joint motion is dominated by whatever the rest of the chain is doing.
MIN_STEP_PSI = 5.0


def responsiveness(rec, *, settle_s: float = SETTLE_S,
                   min_step_psi: float = MIN_STEP_PSI, kinds=None):
    """Degrees of joint angle per psi of commanded differential, per joint.

    Fitted through the origin: a commanded differential of zero should give no
    *change*, and letting the fit find an intercept would absorb the chain's
    own drift into a number meant to describe one joint's actuator.
    """
    pairs = ACT.joint_pairs()
    base_index = {b: i for i, b in enumerate(rec.ids)}
    t = rec.can_sync_time_s
    tgt_psi = rec.target_pa / 6894.757
    q_deg = np.degrees(rec.q)

    if kinds is not None:
        keep = np.array([str(p) in set(kinds) for p in rec.phase], dtype=bool)
    else:
        keep = np.ones(t.shape[0], dtype=bool)

    out = []
    for j, (pos, neg) in enumerate(pairs):
        a, b = base_index[pos], base_index[neg]
        diff = tgt_psi[:, a] - tgt_psi[:, b]
        # A step is where the commanded differential changes between cycles.
        d = np.diff(diff, prepend=diff[0])
        idx = np.nonzero((np.abs(d) >= min_step_psi) & keep)[0]
        dx, dy = [], []
        n_settle = int(round(settle_s * 150.0))
        for k in idx:
            k1 = k + n_settle
            if k1 >= t.shape[0] or not rec.q_valid[k:k1].all():
                continue
            if t[k1] - t[k] > settle_s * 1.5:      # a gap; not one settle
                continue
            dx.append(diff[k1] - diff[max(k - 1, 0)])
            dy.append(q_deg[k1, j] - q_deg[max(k - 1, 0), j])
        dx = np.asarray(dx)
        dy = np.asarray(dy)
        if dx.size < 20:
            out.append({"joint": j, "name": ACT.JOINT_NAMES[j], "n_steps": int(dx.size),
                        "deg_per_psi": float("nan"), "r": float("nan"),
                        "boards": [f"0x{pos:03X}", f"0x{neg:03X}"]})
            continue
        slope = float(np.sum(dx * dy) / np.sum(dx * dx))
        pred = slope * dx
        ss_res = float(np.sum((dy - pred) ** 2))
        ss_tot = float(np.sum(dy ** 2))
        out.append({
            "joint": j,
            "name": ACT.JOINT_NAMES[j],
            "boards": [f"0x{pos:03X}", f"0x{neg:03X}"],
            "n_steps": int(dx.size),
            "deg_per_psi": slope,
            "r": float(np.sqrt(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))),
            "median_abs_response_deg": float(np.median(np.abs(dy))),
        })
    return out


def verdicts(rows, *, low_frac: float = 0.35):
    """Flag a joint whose responsiveness is unlike its neighbours'.

    ``low_frac`` of the median across joints: a joint at a third of what the
    others manage is worth looking at, and the arm's three segments genuinely
    differ (a proximal joint carries more of the chain), so the bar cannot be
    tight without calling every distal joint a fault.
    """
    vals = np.array([r["deg_per_psi"] for r in rows], dtype=float)
    med = float(np.nanmedian(np.abs(vals)))
    for r in rows:
        v = abs(r["deg_per_psi"])
        r["fraction_of_median"] = float(v / med) if med > 0 else float("nan")
        r["suspect"] = bool(np.isfinite(v) and v < low_frac * med)
    return med


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True, action="append")
    ap.add_argument("--kinds", default="random_walk,validation,pair_sweep",
                    help="excitation families to measure over")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    all_rows = []
    flagged: dict = {}
    for sess in args.session:
        rec = ds.load_session(sess, episode_field="segment_index")
        rows = responsiveness(rec, kinds=kinds)
        all_rows.append(rows)
        report = os.path.join(sess, "session_report.json")
        if os.path.exists(report):
            with open(report, encoding="utf-8") as fh:
                for ev in json.load(fh).get("anomalies", []):
                    for j in ev["joints"]:
                        flagged.setdefault(int(j), 0)
                        flagged[int(j)] += 1

    # Average the per-session slopes weighted by how many steps each saw.
    merged = []
    for j in range(12):
        rs = [r[j] for r in all_rows if np.isfinite(r[j]["deg_per_psi"])]
        if not rs:
            merged.append(all_rows[0][j])
            continue
        w = np.array([r["n_steps"] for r in rs], dtype=float)
        merged.append({
            "joint": j, "name": rs[0]["name"], "boards": rs[0]["boards"],
            "n_steps": int(w.sum()),
            "deg_per_psi": float(np.sum(w * [r["deg_per_psi"] for r in rs]) / w.sum()),
            "r": float(np.sum(w * [r["r"] for r in rs]) / w.sum()),
            "median_abs_response_deg":
                float(np.sum(w * [r["median_abs_response_deg"] for r in rs]) / w.sum()),
        })
    med = verdicts(merged)

    print(f"responsiveness over {', '.join(kinds)}; "
          f"median across joints {med:.3f} deg/psi")
    print("joint  name       boards            steps  deg/psi   frac med    r   "
          "live flags  verdict")
    for r in merged:
        r["live_flags"] = int(flagged.get(r["joint"], 0))
        verdict = "SUSPECT" if r["suspect"] else "normal"
        if r["live_flags"] and not r["suspect"]:
            verdict = "normal (the live flags were coincidences)"
        print(f"  j{r['joint']:<2d}  {r['name']:<10s} "
              f"{r['boards'][0]}+{r['boards'][1]}  {r['n_steps']:5d}  "
              f"{r['deg_per_psi']:7.3f}  {r['fraction_of_median']:7.2f}  "
              f"{r['r']:5.2f}   {r['live_flags']:5d}      {verdict}")

    suspects = [r for r in merged if r["suspect"]]
    print()
    if suspects:
        print("SUSPECT joints, worth a physical look: " + ", ".join(
            f"j{r['joint']} ({r['name']}, {r['boards'][0]}+{r['boards'][1]})"
            for r in suspects))
    else:
        print("No joint's responsiveness is unlike its neighbours'. Every live "
              "flag was a joint that happened not to move at one instant, "
              "which is what a serial chain under gravity does.")
    print("This does NOT rule out a joint that moves the wrong way or by the "
          "wrong amount; that needs the fitted model to judge, and twin_compare "
          "is where it is judged.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"sessions": args.session, "kinds": kinds,
                       "median_deg_per_psi": med, "joints": merged}, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
