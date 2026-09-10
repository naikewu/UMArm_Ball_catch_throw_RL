"""Prove the three instruments agree: CAN arm, mocap and camera see one motion.

The campaign that follows this test assumes all three are pointed at the same
robot, and nothing offline can check that.  There are *two* ceiling-hung UMArms
in this room -- the RS485 sister arm on Motive bodies 500-505 and this one on
2000-2005 -- plus a Kinova Gen3, and the camera sees all three.  A campaign that
recorded the wrong arm's video would look perfect and be worthless.

So this test drives one actuator, alone, and asks each instrument what it saw:

  * the **bus** should report the commanded pressure arriving at that board;
  * **mocap** should report a joint angle moving, on the joint the measured
    actuator map names for that board, in the sign it names;
  * the **camera** should show motion, and the test writes a before/after pair
    plus a clip so the operator can confirm by eye which arm moved.

Safety: exactly one actuator carries pressure at a time, so the sum over any
antagonistic pair is that one pressure and the 30 psi pair ceiling cannot be
approached.  Every exit path stops the cycle, which sends a table with every
enable bit clear.

Usage::

    .venv\\Scripts\\python.exe hw_tests\\canarm_hello.py
    .venv\\Scripts\\python.exe hw_tests\\canarm_hello.py --base 0x115 --psi 12
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
for _p in (_WS, os.path.join(_WS, "TLE_PCB"),
           os.path.join(_WS, "UMArm_MOCAP", "natnet_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_env  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from UMArm_CAMERA import BenchCamera  # noqa: E402
from UMArm_KINEMATICS import canarm_actuators as ACT  # noqa: E402
from UMArm_MOCAP.canarm_mocap import CANARM_N_BODIES, CanArmMocap  # noqa: E402

#: Idle setpoint for every board not under test.  Held *enabled*, because
#: ``0x110`` leaks from its supply side and a disabled board stops venting it.
IDLE_PSI = 0.5

#: Ceiling on this test's drive pressure.  One actuator live at a time makes
#: the pair sum equal to this, well under the operator's 30 psi rule.
PSI_CEILING = 14.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="0x101",
                    help="board to drive, e.g. 0x101 (default: the most "
                         "proximal TLE board, so the whole arm below it swings)")
    ap.add_argument("--psi", type=float, default=12.0)
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--port", default=None)
    ap.add_argument("--server-ip", default=bench_env.MOCAP_SERVER_IP)
    ap.add_argument("--client-ip", default=bench_env.MOCAP_CLIENT_IP)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "media"))
    ap.add_argument("--no-camera", action="store_true")
    args = ap.parse_args()

    base = int(args.base, 0)
    psi = min(float(args.psi), PSI_CEILING)
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    b2j = ACT.base_to_joint()
    if base not in b2j:
        print(f"hello: 0x{base:03X} is not in the measured actuator map")
        return 1
    joint, sign = b2j[base]
    print(f"hello: driving 0x{base:03X} at {psi:.1f} psi -- the measured map "
          f"says joint {joint} ({ACT.JOINT_NAMES[joint]}) sign {sign:+d}")

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    cam = BenchCamera(enabled=not args.no_camera)
    cam_up = cam.start()
    check("camera delivers frames", cam_up or args.no_camera,
          f"{cam.state.frames} frames, {cam.state.fps:.1f} fps"
          if cam_up else (cam.state.error or "camera disabled by flag"))

    rx = CanArmMocap(server_ip=args.server_ip, client_ip=args.client_ip,
                     ring_capacity=6000, marker_ring_capacity=6000)
    rx.start()
    result = {"schema": "canarm_hello/1", "created_local": stamp,
              "base": f"0x{base:03X}", "psi": psi,
              "expected_joint": joint, "expected_sign": sign}
    try:
        deadline = time.monotonic() + 5.0
        while rx.get_state().frames < 20 and time.monotonic() < deadline:
            time.sleep(0.1)
        st = rx.get_state()
        check("mocap streaming", st.frames > 20,
              f"{st.frames} frames at {st.fps:.1f} Hz, bodies 2000-"
              f"{2000 + CANARM_N_BODIES - 1}")
        if st.frames <= 20:
            return 1

        port = bench_env.resolve_can_port(args.port)
        be = Backend(port=port, bitrate=bench_env.BITRATE)
        be.open()
        try:
            found = be.scan()
            present = sorted(b for b, n in found.items() if n.present)
            check("all 24 boards present", len(present) == 24,
                  f"{len(present)} of 24 answered the scan")
            be.select(list(P.ALL_IDS))
            for b in P.ALL_IDS:
                be.set_target(b, 0.0)
                be.set_enabled(b, False)
            be.start_cycle()
            time.sleep(0.5)
            for b in P.ALL_IDS:
                be.set_target(b, IDLE_PSI)
                be.set_enabled(b, True)
            time.sleep(args.settle)

            # -- baseline ------------------------------------------------- #
            t0 = time.monotonic()
            time.sleep(1.5)
            base_win = rx.snapshot_window(t0, time.monotonic())
            q_rest = base_win.q.mean(axis=0)
            frame_before = cam.latest_frame()

            # -- drive ---------------------------------------------------- #
            clip_path = os.path.join(args.out_dir, f"hello_{base:03X}_{stamp}.mp4")
            cam.clip(clip_path, pre_s=2.0, post_s=args.hold + 3.0,
                     label=f"0x{base:03X} @ {psi:.0f} psi")
            be.set_target(base, psi)
            time.sleep(args.hold)
            t1 = time.monotonic()
            time.sleep(1.5)
            drive_win = rx.snapshot_window(t1, time.monotonic())
            q_drive = drive_win.q.mean(axis=0)
            nodes = be.snapshot_nodes()
            reached = float(nodes[base].pressure_psi)
            frame_after = cam.latest_frame()

            # -- release -------------------------------------------------- #
            be.set_target(base, IDLE_PSI)
            time.sleep(args.settle)

            dq = np.degrees(np.asarray(q_drive) - np.asarray(q_rest))
            order = np.argsort(-np.abs(dq))
            print("  dq (deg), largest first: " + ", ".join(
                f"j{int(k)}={dq[k]:+.2f}" for k in order[:4]))
            result["q_rest_deg"] = np.degrees(q_rest).tolist()
            result["q_drive_deg"] = np.degrees(q_drive).tolist()
            result["dq_deg"] = dq.tolist()
            result["reached_psi"] = reached
            result["dominant_joint"] = int(order[0])

            check("commanded pressure arrived at the board",
                  reached >= 0.5 * psi,
                  f"0x{base:03X} read {reached:.2f} psi against a {psi:.1f} psi "
                  f"target")
            check("mocap saw the arm move",
                  float(np.abs(dq).max()) > 1.0,
                  f"largest joint change {np.abs(dq).max():.2f} deg on joint "
                  f"{int(order[0])} ({ACT.JOINT_NAMES[int(order[0])]})")
            check("the joint that moved is the one the map names",
                  int(order[0]) == joint,
                  f"moved joint {int(order[0])}, map says {joint}")
            check("the direction matches the map's sign",
                  np.sign(dq[joint]) == sign,
                  f"dq[{joint}] = {dq[joint]:+.2f} deg, map sign {sign:+d}")

            if frame_before is not None and frame_after is not None:
                import cv2
                diff = np.abs(frame_after.astype(np.int16)
                              - frame_before.astype(np.int16)).sum(axis=2)
                moved = float((diff > 40).mean())
                pair = np.hstack([frame_before, frame_after])
                pair_path = os.path.join(args.out_dir,
                                         f"hello_{base:03X}_{stamp}_pair.jpg")
                cv2.imwrite(pair_path, pair)
                # Where in the picture the motion is -- this is what tells the
                # operator which of the room's two arms actually swung.
                ys, xs = np.nonzero(diff > 40)
                centre = (float(xs.mean()), float(ys.mean())) if xs.size else None
                result["motion_fraction"] = moved
                result["motion_centroid_px"] = centre
                result["pair_image"] = pair_path
                check("camera saw motion", moved > 0.002,
                      f"{moved * 100:.2f}% of pixels changed"
                      + (f", centroid at x={centre[0]:.0f} y={centre[1]:.0f} "
                         f"of {frame_before.shape[1]}x{frame_before.shape[0]}"
                         if centre else ""))
        finally:
            be.stop_cycle()
            be.close()
    finally:
        rx.stop()
        cam.drain(timeout_s=30.0)
        cam.stop()

    result["clips"] = cam.state.clip_paths
    result["checks"] = [{"name": n, "ok": o, "detail": d} for n, o, d in checks]
    out = os.path.join(args.out_dir, f"hello_{base:03X}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"hello: wrote {out}")
    for p in cam.state.clip_paths:
        print(f"hello: clip {p} ({os.path.getsize(p) / 1e6:.2f} MB)")

    failed = [n for n, o, _ in checks if not o]
    print("=" * 70)
    print("  PASSED" if not failed else "  FAILED: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
