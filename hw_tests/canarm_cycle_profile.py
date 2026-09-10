"""Attribute the collection stack's cycle-rate loss to the thing that causes it.

``hw_tests/can_bringup.py`` measures 150.00 Hz with zero misses on a bare bus.
The first collection rehearsals measured 138 Hz with a perfect 6.667 ms median
and a 14 ms p99 — that shape is not a slow loop, it is a loop that *skips whole
cycles*, because ``Backend._cycle_loop`` resets its schedule after a stall
rather than bursting to catch up.  Something holds the cycle thread for longer
than one period, about eight times a second.

Guessing which addition does it is what this script exists to avoid.  Each
phase adds exactly one component and runs the same 20 s measurement, so the
loss is attributed by subtraction rather than by argument:

    bare      -- the cycle alone, every enable bit clear (the bring-up baseline)
    +mocap    -- a NatNet receiver running, nothing reading it
    +observer -- a per-cycle observer that only counts
    +record   -- the real Recorder, writing JSONL to disk
    +drive    -- the real excitation evaluated and pushed every cycle
    +camera   -- the bench camera grabbing and clipping
    +nogc     -- everything, with the cyclic collector frozen
    +fine_gil -- everything, with the interpreter's thread-switch interval cut
                 from CPython's 5 ms default to 0.5 ms

It commands nothing: every phase holds the arm at the idle setpoint the
campaign uses, and the excitation phase drives a small differential well inside
the envelope.

Usage::

    .venv\\Scripts\\python.exe hw_tests\\canarm_cycle_profile.py
    .venv\\Scripts\\python.exe hw_tests\\canarm_cycle_profile.py --seconds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS, os.path.join(_WS, "TLE_PCB"),
           os.path.join(_WS, "UMArm_MOCAP", "natnet_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_env  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from collection.excitation import validation  # noqa: E402
from collection.recorder import Recorder  # noqa: E402
from collection.safety import ALL_BASES, PairEnvelope  # noqa: E402

PHASES = ("bare", "mocap", "observer", "record", "drive", "camera", "nogc",
          "fine_gil")


def measure(be: Backend, seconds: float) -> dict:
    """Cycle statistics taken from the observer's own stamps.

    Measured from ``t_sync`` rather than from ``Backend.stats`` because the
    question is what the *cycle thread* experienced, and the backend's jitter
    deque reports the drift it corrected rather than the periods it skipped.
    """
    stamps: list = []
    prev = be._on_cycle  # noqa: SLF001 - this script is a probe on the backend

    def tap(t_sync, targets, replies):
        stamps.append((t_sync, len(replies)))
        if prev is not None:
            prev(t_sync, targets, replies)

    be.set_cycle_observer(tap)
    time.sleep(seconds)
    be.set_cycle_observer(prev)
    if len(stamps) < 10:
        return {"cycles": len(stamps)}
    t = np.array([s[0] for s in stamps])
    n = np.array([s[1] for s in stamps])
    dt = np.diff(t) * 1000.0
    return {
        "cycles": int(len(t)),
        "hz": float((len(t) - 1) / (t[-1] - t[0])),
        "dt_median_ms": float(np.median(dt)),
        "dt_p99_ms": float(np.percentile(dt, 99)),
        "dt_max_ms": float(dt.max()),
        "skipped_frac": float(np.mean(dt > 1.5 * np.median(dt))),
        "all24_frac": float(np.mean(n == 24)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--phases", default=",".join(PHASES))
    args = ap.parse_args()
    want = [p.strip() for p in args.phases.split(",") if p.strip()]

    env = PairEnvelope()
    scratch = os.path.join(_HERE, "results", "cycle_profile_tmp")
    os.makedirs(scratch, exist_ok=True)

    rx = cam = rec = None
    results = {}
    be = Backend(port=bench_env.resolve_can_port(args.port),
                 bitrate=bench_env.BITRATE)
    be.open()
    try:
        found = be.scan()
        if sum(1 for n in found.values() if n.present) != 24:
            print("profile: not all 24 boards present; stopping")
            return 1
        be.select(list(ALL_BASES))
        for b in ALL_BASES:
            be.set_target(b, env.idle_psi)
            be.set_enabled(b, False)
        be.start_cycle()
        time.sleep(0.5)
        be.set_enabled_all(True)
        time.sleep(1.0)

        for phase in want:
            if phase == "mocap":
                from UMArm_MOCAP.canarm_mocap import CanArmMocap
                rx = CanArmMocap(server_ip=bench_env.MOCAP_SERVER_IP,
                                 client_ip=bench_env.MOCAP_CLIENT_IP,
                                 ring_capacity=8000, marker_ring_capacity=4000)
                rx.start()
                time.sleep(2.0)
            elif phase == "observer":
                counter = {"n": 0}

                def count(t, tg, rp, c=counter):
                    c["n"] += 1
                be.set_cycle_observer(count)
            elif phase == "record":
                cals = {b: found[b].cal for b in ALL_BASES}
                bt = [P.VARIANT_TLE_DVP if found[b].is_tle else P.VARIANT_7MM
                      for b in ALL_BASES]
                rec = Recorder(os.path.join(scratch, "rec"), ids=ALL_BASES,
                               board_type=bt, cals=cals, mocap=rx)
                rec.start()
                be.set_cycle_observer(rec.on_cycle)
            elif phase == "drive":
                seg = validation(env, seed=1, seconds=600.0, block_s=600.0)[0]
                t0 = time.perf_counter()

                def drive(t_sync, tg, rp, seg=seg, t0=t0):
                    if rec is not None:
                        rec.on_cycle(t_sync, tg, rp)
                    psi = env.safe(seg.fn(t_sync - t0), where="profile")
                    be.set_targets({b: float(psi[i])
                                    for i, b in enumerate(ALL_BASES)})
                be.set_cycle_observer(drive)
            elif phase == "camera":
                from UMArm_CAMERA import BenchCamera
                cam = BenchCamera()
                cam.start()
                cam.clip(os.path.join(scratch, "profile.mp4"),
                         pre_s=2.0, post_s=args.seconds)
            elif phase == "fine_gil":
                from tlelib.timing import fine_gil_handoff
                fine_gil_handoff()
            elif phase == "nogc":
                # Freeze everything allocated so far into the permanent
                # generation and stop collecting cycles.  A gen-2 pass walks
                # every container in the process, and this stack allocates
                # twenty-four-element lists 150 times a second.
                gc.collect()
                gc.freeze()
                gc.disable()

            res = measure(be, args.seconds)
            results[phase] = res
            print(f"  {phase:<10} {res.get('hz', 0):7.2f} Hz  "
                  f"median {res.get('dt_median_ms', 0):.3f} ms  "
                  f"p99 {res.get('dt_p99_ms', 0):6.3f}  "
                  f"max {res.get('dt_max_ms', 0):7.3f}  "
                  f"skipped {res.get('skipped_frac', 0) * 100:5.2f}%  "
                  f"all24 {res.get('all24_frac', 0) * 100:6.2f}%")
    finally:
        gc.enable()
        try:
            be.set_cycle_observer(None)
            be.stop_cycle()
        finally:
            be.close()
        if rec is not None:
            rec.close()
        if rx is not None:
            rx.stop()
        if cam is not None:
            cam.drain(timeout_s=30.0)
            cam.stop()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"profile: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
