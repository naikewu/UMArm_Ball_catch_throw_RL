"""Live check of the control GUI's marker-frame path: Lock plates, then match.

Opens the real ``canarm_control_gui.py`` window against the live NatNet stream
and exercises the two things that were added to it on 2026-08-21:

* **Lock plates** — a 3 s rest capture that mints this Motive session's plate
  locks, writes them, and restarts the receiver as the marker-registered one.
  The check is not that it returns cleanly but that the *class* changes: before
  the press the window holds a ``CanArmMocap`` reading Motive's body frames,
  after it a ``CanArmMarkerMocap`` reading the markers.
* **the kinematics line** — the distance between each measured u-joint centre
  and the one fkine predicts from the same frame's ``q``.  This is the number
  the whole calibration exists to make small, and the check bounds it.

**Opens no serial port.**  The window's Connect button is never pressed, so the
CAN backend stays ``None`` and nothing can reach a board; this is a mocap and
kinematics test, and mixing it with the bus would make a failure ambiguous.
``hw_tests/integrated_gui_test.py`` is the one that owns the bus.

    python hw_tests/canarm_gui_kinematics.py [--seconds 8] [--no-lock]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MEDIA = os.path.join(_HERE, "media")

#: Bound on the live fkine-vs-mocap u-joint centre error, metres.  The
#: 2026-08-21 calibration reads 0.2-1.8 mm per plate at a static pose and
#: 2.0 mm RMS over driven multi-joint poses; 6 mm is loose enough that a still
#: arm cannot trip it and tight enough that a stale lock or a wrong azimuth
#: (which costs tens of millimetres) cannot pass.
FK_RESIDUAL_TOL_M = 0.006


def pump(root, seconds: float, tick: float = 0.02) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update_idletasks()
        root.update()
        time.sleep(tick)


def grab(root, name: str) -> str | None:
    try:
        from PIL import ImageGrab
    except Exception:
        return None
    root.update_idletasks()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    box = (x, y, x + root.winfo_width(), y + root.winfo_height())
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    os.makedirs(MEDIA, exist_ok=True)
    path = os.path.join(MEDIA, name)
    ImageGrab.grab(bbox=box, all_screens=True).save(path)
    return path


def run(args) -> int:
    import tkinter as tk
    from tkinter import ttk

    import numpy as np

    import canarm_control_gui as GUI
    from UMArm_MOCAP import canarm_frames as CF
    from UMArm_MOCAP.canarm_mocap import CanArmMarkerMocap

    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"   {detail}" if detail else ""))
        return bool(ok)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = GUI.CanArmControllerApp(root, mocap_source="live")
    pump(root, 0.5)
    check("window built with no serial port", app.backend is None,
          f"backend={app.backend}")

    app.mocap_source.set("live")
    app.toggle_mocap()
    pump(root, args.seconds)
    rx = app._mocap
    if rx is None:
        check("live mocap started", False, "receiver is None")
        root.destroy()
        return 1
    st = rx.get_state()
    check("live mocap streaming", st.frames > 100 and not st.stale,
          f"{st.frames} frames, {st.fps:.1f} fps, stale={st.stale}")
    print(f"  strip:     {app.mocap_status.get()}")
    print(f"  kinematics:{app.kin_status.get()}")
    before_line = app.kin_status.get()
    before_cls = type(rx).__name__
    grab(root, "canarm_gui_kinematics_before.png")

    if not args.no_lock:
        print("  pressing Lock plates (hold the arm still) ...")
        app.lock_plates()
        pump(root, args.seconds)
        rx = app._mocap
        check("Lock plates swapped in the marker receiver",
              isinstance(rx, CanArmMarkerMocap),
              f"{before_cls} -> {type(rx).__name__ if rx else None}")
        check("locks were written",
              os.path.isfile(CF.DEFAULT_LOCK_PATH), CF.DEFAULT_LOCK_PATH)

    line = app.kin_status.get()
    print(f"  kinematics:{line}")
    check("the kinematics line changed after locking",
          args.no_lock or line != before_line)

    poses = None
    if rx is not None and hasattr(rx, "get_marker_poses"):
        deadline = time.monotonic() + 5.0
        while poses is None and time.monotonic() < deadline:
            pump(root, 0.2)
            poses = rx.get_marker_poses()
    check("marker-registered plate frames are published", poses is not None)

    result = {"schema": "canarm_gui_kinematics/1",
              "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
              "strip": app.mocap_status.get(), "kinematics": line}
    if poses is not None:
        stats = rx.solve_stats()
        frac = stats.solved / max(stats.frames, 1)
        check("the marker solve succeeds on essentially every frame",
              frac > 0.98, f"{stats.solved}/{stats.frames} = {frac:.4f}")
        worst_rms = max(stats.last_rms_m.values(), default=float("nan"))
        check("template registration residual is inside its gate",
              worst_rms < 0.003, f"worst {worst_rms * 1000:.3f} mm")

        res = CF.fk_residual_m(poses)
        q = CF.q_from_plate_frames(poses)
        gaps = CF.chain_gaps_m(poses)
        corigid = CF.co_rigid_residual_deg(poses)
        print("  fkine vs mocap (mm): "
              + " ".join(f"u{p+1} {res[p]*1000:.2f}" for p in range(6)))
        print("  chain gaps (mm):     "
              + " ".join(f"{v*1000:.2f}" for v in gaps))
        print("  q (deg):             "
              + " ".join(f"{v:.1f}" for v in np.degrees(q)))
        check("fkine reproduces every measured u-joint centre",
              float(np.max(res)) < FK_RESIDUAL_TOL_M,
              f"worst {np.max(res)*1000:.2f} mm "
              f"(tol {FK_RESIDUAL_TOL_M*1000:.0f} mm)")
        from UMArm_KINEMATICS import canarm_params as cp
        gap_err = np.abs(np.asarray(gaps) - np.asarray(cp.CANARM_PLATE_CHAIN_M))
        check("the measured chain matches the committed table",
              float(gap_err.max()) < 0.002,
              f"worst {gap_err.max()*1000:.2f} mm")
        check("co-rigid plates still agree",
              max(abs(a) for a, _ in corigid) < 1.0
              and max(abs(b) for _, b in corigid) < 1.0,
              "; ".join(f"{a:+.2f} deg / {b:.2f} deg tilt" for a, b in corigid))
        result.update({
            "fk_residual_mm": (np.asarray(res) * 1000).tolist(),
            "chain_gaps_mm": (np.asarray(gaps) * 1000).tolist(),
            "q_deg": np.degrees(q).tolist(),
            "co_rigid_deg": [list(c) for c in corigid],
            "solve_fraction": frac,
            "worst_template_rms_mm": worst_rms * 1000,
        })
    shot = grab(root, "canarm_gui_kinematics.png")
    if shot:
        print(f"  screenshot -> {shot}")

    app.on_close()
    try:
        root.destroy()
    except Exception:
        pass

    passed = sum(1 for _n, ok, _d in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    result["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {args.json_out}")
    return 0 if passed == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--no-lock", action="store_true",
                    help="skip minting; use whatever locks already exist")
    ap.add_argument("--json-out", default=None)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
