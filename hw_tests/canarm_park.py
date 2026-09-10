"""Vent the arm and park it, verifying rather than assuming.

The safe park is not "clear the enable bits".  A 7 mm board freezes its
solenoids where a disable finds them (``firmware/legacy/main/main.c:517`` skips
the control body when the enable bit is clear, above every ``gpio_set_level``),
and that tree has no link-loss timeout, so a board disabled while inflating
goes on inflating.  The safe park is: command zero, **watch the pressures come
down**, and only then clear the bits.

Watching is the part that is easy to get wrong.  This script reads the
pressures out of the replies as they arrive, so what it reports is what the
boards said during the wait -- not a value cached before the wait began, which
is a check that always passes.

Run it after any session that ended abnormally, and before leaving the bench.

Usage::

    .venv\\Scripts\\python.exe hw_tests\\canarm_park.py
    .venv\\Scripts\\python.exe hw_tests\\canarm_park.py --seconds 20 --floor 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_env  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from tlelib.timing import fine_gil_handoff, hires_clock  # noqa: E402
from collection.safety import ALL_BASES  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--floor", type=float, default=1.0,
                    help="psi below which a board counts as vented")
    ap.add_argument("--port", default=None)
    args = ap.parse_args(argv)

    hires_clock()
    fine_gil_handoff()

    meas = np.full(24, np.nan)
    zero = np.zeros(24)
    scale = np.ones(24)

    def observe(t_sync, targets, replies):
        for k, base in enumerate(ALL_BASES):
            rep = replies.get(base)
            if rep is not None:
                meas[k] = scale[k] * (rep[1].counts - zero[k])

    be = Backend(port=bench_env.resolve_can_port(args.port),
                 bitrate=bench_env.BITRATE, log=lambda m: print("  " + m))
    be.open()
    try:
        found = be.scan()
        present = sorted(b for b, n in found.items() if n.present)
        print(f"park: {len(present)} of 24 boards present")
        for k, b in enumerate(ALL_BASES):
            if b in found:
                zero[k] = found[b].cal.zero_counts
                scale[k] = 1.0 / found[b].cal.counts_per_psi
        be.select(list(ALL_BASES))
        for b in ALL_BASES:
            be.set_target(b, 0.0)
            be.set_enabled(b, False)
        be.start_cycle()
        be.set_cycle_observer(observe)
        time.sleep(0.5)

        # Enabled and commanded to zero: the regulator is what actually opens
        # the exhaust, so the arm cannot vent while the enable bits are down.
        be.set_enabled_all(True)
        be.set_targets({b: 0.0 for b in ALL_BASES})
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            time.sleep(0.25)
            hot = np.nansum(meas > args.floor)
            print(f"\rpark: venting, {int(hot)} board(s) above {args.floor} psi, "
                  f"worst {np.nanmax(meas):6.2f} psi   ", end="", flush=True)
            if hot == 0:
                break
        print()

        held = {f"0x{b:03X}": round(float(meas[i]), 2)
                for i, b in enumerate(ALL_BASES) if meas[i] > args.floor}
        if held:
            print(f"park: STILL PRESSURISED after {args.seconds:.0f} s: {held}")
            print("park: 0x110 inflates from its supply side and will not "
                  "empty; anything else here is worth looking at")
        else:
            print(f"park: every board below {args.floor} psi")
        # Only now, with every board venting or already empty, is a disable a
        # safe thing to freeze.
        be.set_enabled_all(False)
        time.sleep(0.3)
        return 0 if not held else 2
    finally:
        be.set_cycle_observer(None)
        be.stop_cycle()
        be.close()
        print("park: cycle stopped, enable bits clear, port released")


if __name__ == "__main__":
    raise SystemExit(main())
