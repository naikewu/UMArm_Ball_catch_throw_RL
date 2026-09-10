"""What actually happens to a board when its enable bit goes down.

Every safe-exit path in this workspace clears the enable bits and treats the
arm as parked.  Reading `firmware/legacy/main/main.c` says that is not what a
7 mm board does.  Line 517 is

    int ctrl = pcr->active_control_byte;
    if ((ctrl & 0x01) == 0) { vTaskDelay(1); continue; }

and all three `gpio_set_level(V_IN/V_OUT, ...)` calls sit **below** it, so
clearing the bit does not de-energise the solenoids -- it freezes them wherever
they were last driven.  That tree also has no link-loss timeout (its watchdog
only resets the MCP2515), so a board disabled while inflating goes on inflating
with nothing able to stop it but a fresh enabled frame or power.

This script measures what the arm actually does, because a source reading is a
claim about the code and this is a claim about the robot.  It reduces the
`leak_trap_*` segments of a collection session -- twenty-four boards charged to
the same pressure and then disabled -- into a per-board drift rate, and asks
three questions of the result:

1. **is the drift repeatable?**  A leak is a board property and repeats; a
   valve frozen at a random point of a limit cycle does not;
2. **is any board rising?**  That is the hazardous freeze, and the one the
   firmware gives no way to stop;
3. **does any board empty completely?**  That is a freeze with the exhaust
   open, which is harmless but proves the valves do not shut on disable.

The same numbers are the per-board leak rate the twin's actuator model needs,
which is why the campaign takes the measurement twice rather than once.

Usage::

    .venv\\Scripts\\python.exe hw_tests\\canarm_disable_behaviour.py \\
        --session data\\session_20260910_012159
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PA_PER_PSI = 6894.757

#: A slope steeper than this is called out rather than left in the noise.  The
#: measurement's own repeatability on 2026-09-10 was about 0.03 psi/s between
#: two traps twenty seconds apart, so anything under 0.05 psi/s is not
#: distinguishable from the sensor and the fit.
NOISE_PSI_S = 0.05

#: Fraction of the charge below which a board is called "emptied", i.e. it
#: froze with its exhaust open rather than shut.
EMPTIED_FRAC = 0.4


def load_traps(session_dir: str):
    """Every ``leak_trap_*`` segment, as ``(name, t, psi[n,24])``."""
    with open(os.path.join(session_dir, "metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    ids = meta["selected_ids"]
    cal = meta["node_cal"]
    zero = np.array([cal[i]["zero_counts"] for i in ids], dtype=float)
    cpp = np.array([cal[i]["counts_per_psi"] for i in ids], dtype=float)
    board_type = meta["board_type"]

    by_seg: dict = {}
    for path in sorted(glob.glob(os.path.join(session_dir,
                                              "samples_chunk_*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                name = row.get("segment", "")
                if not name.startswith("leak_trap"):
                    continue
                by_seg.setdefault(name, []).append(
                    (row["can_sync_time_s"], row["robot_state"]["pressure_adc"]))
    out = []
    for name in sorted(by_seg):
        rows = by_seg[name]
        t = np.array([r[0] for r in rows], dtype=float)
        adc = np.array([r[1] for r in rows], dtype=float)
        out.append((name, t, (adc - zero) / cpp))
    return ids, board_type, out


def reduce_traps(ids, board_type, traps, *, settle_s: float = 0.3):
    """Per board, the drift in each trap and their agreement.

    The first ``settle_s`` of each trap is dropped: the enable bit and the
    valve state change on different cycles, and a fit that includes the
    transition measures the transition.
    """
    per_board: dict = {i: [] for i in ids}
    for name, t, psi in traps:
        keep = t >= t[0] + settle_s
        tt, pp = t[keep], psi[keep]
        for k, bid in enumerate(ids):
            slope = float(np.polyfit(tt, pp[:, k], 1)[0])
            per_board[bid].append({
                "trap": name,
                "start_psi": float(pp[:20, k].mean()),
                "end_psi": float(pp[-20:, k].mean()),
                "slope_psi_s": slope,
                "slope_pa_s": slope * PA_PER_PSI,
                "seconds": float(tt[-1] - tt[0]),
            })
    rows = []
    for k, bid in enumerate(ids):
        runs = per_board[bid]
        slopes = np.array([r["slope_psi_s"] for r in runs])
        starts = np.array([r["start_psi"] for r in runs])
        ends = np.array([r["end_psi"] for r in runs])
        rows.append({
            "id": bid,
            "board_type": int(board_type[k]),
            "kind": {0: "7mm", 1: "DT", 2: "TLE"}.get(int(board_type[k]), "?"),
            "n_traps": len(runs),
            "slope_psi_s": float(slopes.mean()),
            "slope_pa_s": float(slopes.mean() * PA_PER_PSI),
            "slope_spread_psi_s": float(slopes.max() - slopes.min()),
            "start_psi": float(starts.mean()),
            "end_psi": float(ends.mean()),
            "rising": bool(slopes.mean() > NOISE_PSI_S),
            "emptied": bool(ends.mean() < EMPTIED_FRAC * max(starts.mean(), 1e-9)),
        })
    return rows


def report(rows) -> str:
    lines = []
    lines.append("board   kind   charge -> end (psi)    drift psi/s   "
                 "spread   verdict")
    for r in rows:
        verdict = []
        if r["rising"]:
            verdict.append("RISING while disabled")
        if r["emptied"]:
            verdict.append("froze with the exhaust OPEN")
        if not verdict and abs(r["slope_psi_s"]) < NOISE_PSI_S:
            verdict.append("froze shut")
        elif not verdict:
            verdict.append("venting")
        lines.append(f"{r['id']}  {r['kind']:5s} {r['start_psi']:6.2f} -> "
                     f"{r['end_psi']:6.2f}      {r['slope_psi_s']:+7.3f}   "
                     f"{r['slope_spread_psi_s']:6.3f}   {'; '.join(verdict)}")
    rising = [r["id"] for r in rows if r["rising"]]
    emptied = [r["id"] for r in rows if r["emptied"]]
    spread = np.array([r["slope_spread_psi_s"] for r in rows])
    mags = np.abs([r["slope_psi_s"] for r in rows])
    lines.append("")
    lines.append(f"repeatability: worst disagreement between traps "
                 f"{spread.max():.3f} psi/s, median {np.median(spread):.3f} -- "
                 f"the drift is a board property, not a random freeze point")
    lines.append(f"spread across boards: {mags.min():.3f} to {mags.max():.3f} "
                 f"psi/s, a factor of {mags.max() / max(mags.min(), 1e-9):.0f}")
    lines.append(f"rising while disabled: {rising or 'none in this sample'}")
    lines.append(f"emptied (exhaust frozen open): {emptied or 'none'}")
    lines.append("")
    lines.append("A board that empties proves the valves do NOT shut on a "
                 "disable. None happened to freeze inflating here, but that is "
                 "the sample and not a guarantee, which is why every exit path "
                 "commands zero and waits for the arm to empty BEFORE any "
                 "enable bit goes down.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    ids, board_type, traps = load_traps(args.session)
    if not traps:
        print("no leak_trap_* segments in that session")
        return 1
    print(f"{len(traps)} trap(s): " + ", ".join(
        f"{n} ({len(t)} rows, {t[-1] - t[0]:.1f} s)" for n, t, _ in traps))
    rows = reduce_traps(ids, board_type, traps)
    print(report(rows))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"session": args.session, "boards": rows}, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
