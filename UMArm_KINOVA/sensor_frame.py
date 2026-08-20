"""Command the MOCAP frame, not the arm's frame.

The calibration exists to be used, and this is what using it looks like.  Once
``X = T_world_base`` and ``Y = T_tool_rb`` are known, a pose written in the
cameras' coordinates inverts straight back to a pose the arm accepts::

    T_base_tool  =  X^-1 . T_world_rb . Y^-1

so "put the sensor frame HERE, in the volume" becomes one Cartesian move.  That
is the whole point of the exercise: the bench cares where the pad is in the
mocap volume, the arm only knows about its own base, and this module is the one
place the two are reconciled.

    .venv_kinova/Scripts/python.exe UMArm_KINOVA/sensor_frame.py --where
    .venv_kinova/Scripts/python.exe UMArm_KINOVA/sensor_frame.py --move-world 0.03 0 0 --yes

WHAT IT VERIFIES.  ``--move-world`` does not merely command: it reads the rigid
body before and after and reports the achieved displacement against the asked
one.  A prediction that is never checked is a prediction nobody should trust,
and this is the cheapest possible check of the whole chain — a wrong ``X``
rotation shows up as a direction error, a wrong ``Y`` as an offset that grows
with rotation, a stale calibration as both.

WHAT IT DOES NOT DO.  It has no inverse kinematics of its own and no path
planning: it hands the target to the Gen3's own Cartesian planner, and inherits
that planner's opinion about singularities and joint limits.  The keep-out ball
of :class:`~UMArm_KINOVA.kinova_arm.SafeKinovaArm` still applies, and applies to
the TOOL origin, so a large rotation about a distant sensor frame can be refused
even though the sensor frame itself barely moved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from UMArm_KINOVA import mocap_calibration as MC              # noqa: E402
from UMArm_KINOVA.kinova_arm import DEFAULT_IP, SafeKinovaArm  # noqa: E402
from UMArm_KINOVA.vendor.kinova_driver import (               # noqa: E402
    SE3_to_pose, pose_to_SE3)


def load_calibration(path: str | None = None):
    """``(X, Y, path)`` from an ``analysis.json``; the newest one by default."""
    if path is None:
        # PORT NOTE (workspace port, 2026-08-20).  Upstream this read
        # ``check_pad_markers.latest_analysis``.  That module is about the
        # printed collision pad rather than the Gen3 controller and pulls
        # ``UMArm_COLLAB.mount_transforms``, so it was left behind; the walk
        # itself is byte-identical in ``bridge_mocap.latest_analysis_path``,
        # which imports nothing from ``kortex_api``.  See PORTING.md.
        from UMArm_KINOVA.bridge_mocap import latest_analysis_path

        path = latest_analysis_path()
    with open(path, encoding="utf-8") as fh:
        an = json.load(fh)
    return (np.array(an["X_world_base"], dtype=float),
            np.array(an["Y_tool_rb"], dtype=float), path)


def rb_from_tool(tool_pose, X, Y) -> np.ndarray:
    """Where the mocap rigid body is, given the arm's tool pose.  ``(4, 4)``."""
    return X @ pose_to_SE3(tool_pose) @ Y


def tool_pose_for_rb(T_world_rb, X, Y) -> list:
    """The arm pose that puts the rigid body at *T_world_rb*.  A 6-pose.

    The inverse of :func:`rb_from_tool`, and the function this module exists
    for.  Note there is no approximation anywhere: both transforms are exact
    rigid bodies, so the only error in the result is the error in the
    calibration itself.
    """
    T = MC.se3_inv(X) @ np.asarray(T_world_rb, dtype=float) @ MC.se3_inv(Y)
    return list(SE3_to_pose(T))


def rb_target_from_delta(T_world_rb_now, delta_world_m, delta_world_rot_deg=None):
    """A target pose: the current one, displaced in WORLD coordinates.

    The rotation, when given, is applied about the rigid body's own origin in
    world axes — ``T_new = Rot . T_now`` with the origin held — because that is
    what "turn the sensor 10 degrees about world z" means to somebody looking at
    the volume.  Turning it about the body's own axes instead is
    ``T_now . Rot``, which is a different move and is deliberately not what this
    does.
    """
    T = np.array(T_world_rb_now, dtype=float, copy=True)
    if delta_world_rot_deg is not None and any(
            abs(float(v)) > 1e-12 for v in delta_world_rot_deg):
        R = MC.rot_exp(np.deg2rad(np.asarray(delta_world_rot_deg, dtype=float)))
        T[0:3, 0:3] = R @ T[0:3, 0:3]
    T[0:3, 3] = T[0:3, 3] + np.asarray(delta_world_m, dtype=float)
    return T


def move_world(arm, rx, X, Y, delta_m, delta_rot_deg=None, speed_ms=0.02,
               settle_s=1.0, capture_s=1.0, log=print) -> dict:
    """Move the sensor frame by *delta_m* in the volume, and check it landed.

    *arm* is a connected :class:`SafeKinovaArm` with its envelope already armed;
    *rx* a started :class:`~UMArm_KINOVA.kinova_mocap.KinovaMocapRx`.
    """
    before = rx.capture(seconds=capture_s, settle=0.2)
    target = rb_target_from_delta(before["T"], delta_m, delta_rot_deg)
    pose = tool_pose_for_rb(target, X, Y)
    log("commanding tool to " + " ".join("%8.4f" % v for v in pose))
    reached = arm.move_to_pose(pose, speed_ms=speed_ms, name="sensor_frame")
    after = rx.capture(seconds=capture_s, settle=settle_s)

    asked = np.asarray(delta_m, dtype=float)
    got = after["T"][0:3, 3] - before["T"][0:3, 3]
    predicted = rb_from_tool(reached, X, Y)
    return {
        "asked_mm": (asked * 1e3).tolist(),
        "achieved_mm": (got * 1e3).tolist(),
        "error_mm": float(np.linalg.norm(got - asked) * 1e3),
        "direction_deg": (float(np.degrees(np.arccos(
            max(-1.0, min(1.0, float(np.dot(got, asked)
                                     / (np.linalg.norm(got)
                                        * np.linalg.norm(asked))))))))
            if np.linalg.norm(asked) > 1e-9 and np.linalg.norm(got) > 1e-9
            else float("nan")),
        "scale": (float(np.linalg.norm(got) / np.linalg.norm(asked))
                  if np.linalg.norm(asked) > 1e-9 else float("nan")),
        # Where the calibration SAYS the body is, given where the arm ended up.
        # Its gap from the measured pose is the calibration's own error at this
        # pose, separated from anything the planner did.
        "prediction_error_mm": float(np.linalg.norm(
            after["T"][0:3, 3] - predicted[0:3, 3]) * 1e3),
        "prediction_error_deg": MC.angle_between_deg(after["T"][0:3, 0:3],
                                                     predicted[0:3, 0:3]),
        "rotation_asked_deg": (list(delta_rot_deg) if delta_rot_deg
                               else [0.0, 0.0, 0.0]),
        "rotation_achieved_deg": MC.angle_between_deg(before["T"][0:3, 0:3],
                                                      after["T"][0:3, 0:3]),
        "tool_pose_commanded": list(pose),
        "tool_pose_reached": list(reached),
        "before_ptp_mm": before["pos_ptp_mm"],
        "after_ptp_mm": after["pos_ptp_mm"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--server-ip", default=None)
    ap.add_argument("--client-ip", default=None)
    ap.add_argument("--where", action="store_true",
                    help="print where the sensor frame is, and move nothing")
    ap.add_argument("--move-world", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                    help="displace the sensor frame by this, metres, WORLD axes")
    ap.add_argument("--rot-world", nargs=3, type=float, default=None,
                    metavar=("RX", "RY", "RZ"),
                    help="turn it by this, degrees, about world axes")
    ap.add_argument("--speed", type=float, default=0.02, help="m/s")
    ap.add_argument("--back", action="store_true",
                    help="return to the starting sensor pose afterwards")
    ap.add_argument("--yes", action="store_true", help="required to move")
    args = ap.parse_args(argv)

    from UMArm_KINOVA.kinova_mocap import KinovaMocapRx

    X, Y, path = load_calibration(args.analysis)
    print("calibration: " + path)

    kw = {}
    if args.server_ip:
        kw["server_ip"] = args.server_ip
    if args.client_ip:
        kw["client_ip"] = args.client_ip
    rx = KinovaMocapRx(**kw)
    rx.start()
    try:
        with SafeKinovaArm(ip=args.ip) as arm:
            arm.arm_envelope()
            snap = arm.averaged_snapshot(n=10)
            meas = rx.capture(seconds=1.0, settle=0.3)
            pred = rb_from_tool(snap["pose"], X, Y)
            print("sensor frame, MEASURED  : %s m"
                  % " ".join("%9.5f" % v for v in meas["T"][0:3, 3]))
            print("sensor frame, PREDICTED : %s m  (%.2f mm, %.3f deg apart)"
                  % (" ".join("%9.5f" % v for v in pred[0:3, 3]),
                     np.linalg.norm(meas["T"][0:3, 3] - pred[0:3, 3]) * 1e3,
                     MC.angle_between_deg(meas["T"][0:3, 0:3],
                                          pred[0:3, 0:3])))
            if args.where or args.move_world is None:
                return 0
            if not args.yes:
                print("this MOVES the arm.  Re-run with --yes.")
                return 2

            start = np.array(meas["T"], copy=True)
            out = move_world(arm, rx, X, Y, args.move_world, args.rot_world,
                             speed_ms=args.speed)
            print("")
            print("  asked    %s mm" % " ".join("%8.2f" % v
                                                for v in out["asked_mm"]))
            print("  achieved %s mm" % " ".join("%8.2f" % v
                                                for v in out["achieved_mm"]))
            print("  error %.2f mm, direction %.3f deg, scale %.5f"
                  % (out["error_mm"], out["direction_deg"], out["scale"]))
            print("  rotation asked %.2f deg, achieved %.2f deg"
                  % (float(np.linalg.norm(out["rotation_asked_deg"])),
                     out["rotation_achieved_deg"]))
            print("  the calibration predicted the landing pose to %.2f mm / "
                  "%.3f deg" % (out["prediction_error_mm"],
                                out["prediction_error_deg"]))
            if args.back:
                print("returning the sensor frame to where it started ...")
                arm.move_to_pose(tool_pose_for_rb(start, X, Y),
                                 speed_ms=args.speed, name="sensor_back")
    finally:
        rx.stop()
    return 0


__all__ = ["load_calibration", "rb_from_tool", "tool_pose_for_rb",
           "rb_target_from_delta", "move_world"]


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
