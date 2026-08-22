"""Drive the CAN arm one actuator at a time and record what mocap sees.

This is the data-collection half of the 2026-08-21 campaign.  It answers two
questions that nothing offline can:

1. **which joint does each of the 24 actuators move, and in which direction** —
   the actuator/axis map.  The legacy table
   (``UMARM_Variable_Stiffness_Oct2025/robot_constants.py:286``) is a claim
   about a top regulator platform that has since been replaced, so the eight
   TLE boards at ``0x101``-``0x108`` are the part most likely to be stale;
2. **where the marker brackets actually sit relative to the revolute axes** —
   the per-plate azimuth the marker-frame inference needs, measured from the
   mechanism instead of assumed from CAD or read off Motive's manually aligned
   rigid-body frames.

Both come from the same measurement: pressurise exactly one actuator, let the
arm settle, and record the marker positions.  Everything downstream
(``canarm_axis_analysis.py``) is arithmetic on the recorded file, so the
hardware session happens once and the analysis can be re-run.

SAFETY.  Only ever **one** actuator carries pressure at a time in the sweep
phase, so the sum over any antagonistic pair is that one pressure — the
operator's 30 psi per-pair ceiling cannot be approached even in principle.  The
pose phase pressurises at most one actuator *per joint* for the same reason.
``--psi`` is clamped to :data:`PSI_CEILING`.  Every exit path — normal, raised,
or ``Ctrl-C`` — runs ``Backend.stop_cycle()``, which sends a table with every
enable bit clear before the port closes.

USAGE::

    python hw_tests/canarm_drive_campaign.py --out results/drive_2026-08-21.json
    python hw_tests/canarm_drive_campaign.py --only 0x101,0x102 --psi 8
    python hw_tests/canarm_drive_campaign.py --phase rest        # markers only

The output JSON is one record per *step*, each holding the mean marker cloud of
every plate over the sample window, the streamed rigid-body poses over the same
window, and the measured board pressures.  Marker positions are the primary
record: every frame convention this workspace uses can be re-derived from them,
whereas a streamed orientation cannot be un-rotated after the fact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS, os.path.join(_WS, "TLE_PCB"), os.path.join(_WS, "UMArm_MOCAP", "natnet_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_env  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from UMArm_MOCAP.canarm_mocap import CANARM_N_BODIES, CanArmMocap  # noqa: E402

#: Hard ceiling on any commanded pressure, psi.  The bench supply is regulated
#: to ~20 psi and the operator's rule is 30 psi summed over one antagonistic
#: pair; with one actuator live at a time the pair sum *is* this number, so the
#: ceiling is set by what the supply can deliver, not by the pair rule.
PSI_CEILING = 14.0

#: The idle target every board that is not under test regulates to, psi.  Not
#: zero: a board commanded to exactly zero holds its exhaust valve open
#: continuously, which wears the valve and heats the driver for no benefit.
#: A small positive setpoint parks the loop just off the exhaust stop while
#: still venting the one actuator that leaks from its supply side, and 0.5 psi
#: is far below the pressure at which this arm's McKibbens develop useful
#: tension (single-actuator deflections at 12 psi are 13-22 deg; at 0.5 psi
#: the arm sits where gravity puts it).
IDLE_PSI = 0.5

#: Boards, proximal to distal.  0x101-0x108 are the TLE/DVP top platform
#: (segment 1), 0x109-0x118 the legacy 7 mm boards (segments 2 and 3).
ALL_BASES = tuple(P.ALL_IDS)


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def _quat_free_pose_mean(poses: np.ndarray) -> np.ndarray:
    """Mean of a stack of ``(n, 4, 4)`` poses: positions averaged, rotation
    taken from the middle frame.

    Averaging rotation matrices elementwise leaves a non-orthogonal matrix, and
    the campaign's windows are short and nearly static, so the median frame's
    rotation is both simpler and defensible: over a 1 s hold the plates move by
    less than the marker noise (0.02-0.07 mm sd, 2026-08-21 probe).
    """
    out = np.eye(4)
    out[0:3, 3] = poses[:, 0:3, 3].mean(axis=0)
    out[0:3, 0:3] = poses[poses.shape[0] // 2, 0:3, 0:3]
    return out


def sample_mocap(rx: CanArmMocap, t0: float, t1: float) -> dict:
    """Everything the analysis needs about one hold window.

    Per plate, the mean position of each of the four markers over the frames in
    which **all four** were tracked, plus the worst per-marker standard
    deviation (the stillness evidence) and the fraction of frames that
    qualified.  A plate that never had four markers is recorded as ``None``
    rather than as a partial mean: a mean over a changing subset is a number
    with no fixed meaning.
    """
    win = rx.snapshot_marker_window(t0, t1)
    qwin = rx.snapshot_window(t0, t1)
    n = len(win)
    plates: dict[str, dict | None] = {}
    for plate in range(CANARM_N_BODIES):
        rows = []
        for i in range(n):
            mk = win.markers[i]
            if mk is None or plate not in mk:
                continue
            arr = np.asarray(mk[plate], dtype=float)
            if arr.shape != (4, 3) or not np.isfinite(arr).all():
                continue
            fl = None if win.flags[i] is None else win.flags[i].get(plate)
            if fl is not None and int(np.sum(np.asarray(fl) != 0)) < 4:
                continue
            rows.append(arr)
        if not rows:
            plates[str(plate)] = None
            continue
        stack = np.stack(rows)
        sd = (stack.std(axis=0, ddof=1) if stack.shape[0] >= 2
              else np.zeros((4, 3)))
        plates[str(plate)] = {
            "mean_m": stack.mean(axis=0).tolist(),
            "worst_marker_sd_m": float(np.linalg.norm(sd, axis=1).max()),
            "frames": int(stack.shape[0]),
            "frac_of_window": float(stack.shape[0] / n) if n else 0.0,
        }
    streamed_all = ([_quat_free_pose_mean(win.streamed_poses[:, p]).tolist()
                     for p in range(CANARM_N_BODIES)] if n else None)
    return {
        "frames": n,
        "duration_s": float(win.duration),
        "plates": plates,
        "streamed_poses": streamed_all,
        "q_streamed_mean": (qwin.q.mean(axis=0).tolist() if len(qwin) else None),
        "q_streamed_sd": (qwin.q.std(axis=0, ddof=1).tolist()
                          if len(qwin) >= 2 else None),
    }


def sample_pressures(be: Backend) -> dict:
    """Measured and commanded pressure of every present board, psi."""
    nodes = be.snapshot_nodes()
    return {f"0x{b:03X}": {"psi": round(float(n.pressure_psi), 3),
                           "target_psi": round(float(n.target_psi), 3),
                           "enabled": bool(n.enabled),
                           "flags": int(n.flags)}
            for b, n in sorted(nodes.items()) if n.present}


# --------------------------------------------------------------------------
# The drive primitives
# --------------------------------------------------------------------------


def hold_and_sample(be: Backend, rx: CanArmMocap, *, settle_s: float,
                    sample_s: float, label: str, verbose: bool = True) -> dict:
    """Wait out the transient, then record ``sample_s`` of steady state."""
    time.sleep(settle_s)
    t0 = time.monotonic()
    time.sleep(sample_s)
    t1 = time.monotonic()
    rec = sample_mocap(rx, t0, t1)
    rec["label"] = label
    rec["pressures"] = sample_pressures(be)
    rec["t_wall"] = time.time()
    if verbose:
        worst = max((v["worst_marker_sd_m"] for v in rec["plates"].values()
                     if v is not None), default=float("nan"))
        missing = [k for k, v in rec["plates"].items() if v is None]
        print(f"  {label:<28} {rec['frames']:4d} frames, worst marker sd "
              f"{worst * 1000:.3f} mm"
              + (f", plates missing {missing}" if missing else ""))
    return rec


def hold_idle_all(be: Backend) -> None:
    """Every board enabled and regulating to :data:`IDLE_PSI`.

    Enabled-at-idle rather than disabled, which is the opposite of the obvious
    choice and the one the hardware forces.  A disabled board stops regulating,
    and an actuator that stops being regulated keeps whatever air is in it — and
    **one actuator on this arm leaks from its supply side** (``0x110``, observed
    drifting from 0.1 to 7.0 psi over the 2026-08-21 sweep whenever it was left
    disabled).  A leak that inflates a disabled actuator moves the arm between
    steps, which puts a slow ramp underneath every measured delta.  Holding the
    loop closed vents the leak continuously and makes the baseline repeatable.

    The setpoint is :data:`IDLE_PSI`, not zero, so the exhaust valve is not held
    open for the whole campaign — see that constant.
    """
    for base in ALL_BASES:
        be.set_target(base, IDLE_PSI)
        be.set_enabled(base, True)


def relax(be: Backend, seconds: float) -> None:
    """Return to the idle hold and let the actuators empty."""
    hold_idle_all(be)
    time.sleep(seconds)


def drive_one(be: Backend, base: int, psi: float) -> None:
    """Exactly one actuator pressurised; every other board regulating to idle."""
    hold_idle_all(be)
    be.set_target(base, psi)


# --------------------------------------------------------------------------
# The campaign
# --------------------------------------------------------------------------


def run(args) -> int:
    psi = min(float(args.psi), PSI_CEILING)
    if psi != float(args.psi):
        print(f"campaign: --psi clamped to the {PSI_CEILING} psi ceiling")

    bases = ALL_BASES
    if args.only:
        bases = tuple(int(tok, 0) for tok in args.only.split(","))

    port = bench_env.resolve_can_port(args.port)
    print(f"campaign: CAN {bench_env.describe_can_port(args.port)}")
    print(f"campaign: mocap {args.server_ip} <- {args.client_ip}, bodies "
          f"2000..{2000 + CANARM_N_BODIES - 1}")

    record = {
        "schema": "canarm_drive_campaign/1",
        "created_unix_s": time.time(),
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "psi": psi,
        "settle_s": args.settle,
        "sample_s": args.sample,
        "relax_s": args.relax,
        "can_port": port,
        "rb_id_base": 2000,
        "n_bodies": CANARM_N_BODIES,
        "steps": [],
    }

    rx = CanArmMocap(server_ip=args.server_ip, client_ip=args.client_ip,
                     use_multicast=not args.no_multicast,
                     ring_capacity=4000, marker_ring_capacity=4000)
    rx.start()
    try:
        deadline = time.monotonic() + 5.0
        while rx.get_state().frames < 10 and time.monotonic() < deadline:
            time.sleep(0.1)
        st = rx.get_state()
        if st.frames < 10:
            print("campaign: no mocap frames arrived; refusing to drive blind")
            return 1
        print(f"campaign: mocap up, {st.frames} frames, {st.fps:.1f} Hz")

        be = Backend(port=port, bitrate=bench_env.BITRATE)
        be.open()
        try:
            found = be.scan()
            present = sorted(b for b, n in found.items() if n.present)
            print(f"campaign: {len(present)} boards present")
            missing = [b for b in bases if b not in present]
            if missing:
                print("campaign: missing " + ",".join(f"0x{b:03X}" for b in missing))
                return 1
            be.select(list(ALL_BASES))
            for b in ALL_BASES:
                be.set_target(b, 0.0)
                be.set_enabled(b, False)
            be.start_cycle()
            time.sleep(0.5)
            hold_idle_all(be)
            time.sleep(args.relax)

            phases = args.phase.split(",")

            # -- rest ---------------------------------------------------- #
            if "rest" in phases or "all" in phases:
                print(f"campaign: rest capture, {args.rest} s, every board "
                      f"regulating to {IDLE_PSI} psi")
                t0 = time.monotonic()
                time.sleep(args.rest)
                rec = sample_mocap(rx, t0, time.monotonic())
                rec["label"] = "rest"
                rec["kind"] = "rest"
                rec["pressures"] = sample_pressures(be)
                rec["t_wall"] = time.time()
                worst = max((v["worst_marker_sd_m"] for v in rec["plates"].values()
                             if v is not None), default=float("nan"))
                print(f"  rest{'':<24} {rec['frames']:4d} frames, worst marker sd "
                      f"{worst * 1000:.3f} mm")
                record["steps"].append(rec)

            # -- single-actuator sweep ----------------------------------- #
            if "sweep" in phases or "all" in phases:
                print(f"campaign: sweep, {len(bases)} actuators at {psi:.1f} psi")
                for base in bases:
                    drive_one(be, base, psi)
                    rec = hold_and_sample(be, rx, settle_s=args.settle,
                                          sample_s=args.sample,
                                          label=f"drive 0x{base:03X}")
                    rec["kind"] = "single"
                    rec["base"] = base
                    record["steps"].append(rec)
                    relax(be, args.relax)
                    rec0 = hold_and_sample(be, rx, settle_s=0.4, sample_s=0.5,
                                           label=f"back 0x{base:03X}",
                                           verbose=False)
                    rec0["kind"] = "baseline"
                    rec0["base"] = base
                    record["steps"].append(rec0)

            # -- multi-joint poses --------------------------------------- #
            if args.poses and ("poses" in phases or "all" in phases):
                rng = np.random.default_rng(args.seed)
                print(f"campaign: {args.poses} random multi-joint poses")
                pairs = _pair_table(args.pair_table)
                for k in range(args.poses):
                    chosen: dict[int, float] = {}
                    for pair in pairs:
                        side = int(rng.integers(0, 2))
                        p = float(rng.uniform(0.35, 1.0)) * psi
                        chosen[pair[side]] = p
                    hold_idle_all(be)
                    for b, v in chosen.items():
                        be.set_target(b, v)
                    rec = hold_and_sample(be, rx, settle_s=args.settle,
                                          sample_s=args.sample,
                                          label=f"pose {k:02d}")
                    rec["kind"] = "pose"
                    rec["commanded"] = {f"0x{b:03X}": round(v, 3)
                                        for b, v in sorted(chosen.items())}
                    record["steps"].append(rec)
                    relax(be, args.relax)
                rec = hold_and_sample(be, rx, settle_s=0.5, sample_s=1.0,
                                      label="rest final")
                rec["kind"] = "rest"
                record["steps"].append(rec)
        finally:
            be.stop_cycle()
            be.close()
            print("campaign: bus released, every enable bit clear")
    finally:
        rx.stop()

    out = args.out or os.path.join(_HERE, "results",
                                   f"drive_{time.strftime('%Y-%m-%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    print(f"campaign: wrote {out} ({len(record['steps'])} steps)")
    return 0


def _pair_table(path: str | None):
    """The 12 antagonistic pairs, ``[(pos_base, neg_base), ...]``.

    Defaults to the legacy claim so the pose phase can run before the sweep has
    been analysed; ``--pair-table`` points at a JSON written by
    ``canarm_axis_analysis.py`` once the sweep has *measured* them.  Either way
    only one member of a pair is ever pressurised, so a wrong table costs
    coverage, never safety.
    """
    if path:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [(int(a, 0) if isinstance(a, str) else int(a),
                 int(b, 0) if isinstance(b, str) else int(b))
                for a, b in data["pairs"]]
    from UMArm_KINEMATICS.canarm_actuators import joint_pairs
    return list(joint_pairs(allow_legacy=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", default=None, help="CAN COM port override")
    ap.add_argument("--psi", type=float, default=12.0,
                    help=f"drive pressure, clamped to {PSI_CEILING}")
    ap.add_argument("--settle", type=float, default=2.5,
                    help="seconds between commanding and sampling")
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--relax", type=float, default=2.5)
    ap.add_argument("--rest", type=float, default=6.0)
    ap.add_argument("--poses", type=int, default=0)
    ap.add_argument("--pair-table", default=None)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--only", default=None,
                    help="comma-separated bases, e.g. 0x101,0x109")
    ap.add_argument("--phase", default="all",
                    help="comma-separated: rest, sweep, poses, or all")
    ap.add_argument("--server-ip", default=bench_env.MOCAP_SERVER_IP)
    ap.add_argument("--client-ip", default=bench_env.MOCAP_CLIENT_IP)
    ap.add_argument("--no-multicast", action="store_true")
    args = ap.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ncampaign: interrupted; the bus was released by the finally path")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
