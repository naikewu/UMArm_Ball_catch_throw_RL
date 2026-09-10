"""Drive the real operator GUI against the digital twin, with every port poisoned.

The operator's requirement was that ``canarm_control_gui.py`` can start a
simulated arm through the same interface as the real one, and that the
controller cannot tell which it is driving.  This script builds the real window,
connects its SIM adapter, and presses the same handlers an operator's clicks
reach -- the adapter list, Connect, the scan that follows, the board check
boxes, Start cycle, Enable, one channel's own bar, STOP ALL, Disconnect -- and
checks what comes back through the two interfaces a controller reads: the
backend's ``snapshot_nodes()`` and the mocap receiver's ``get_q()``.

OFFLINE, AND ENFORCED RATHER THAN INTENDED.  Before the window exists,
``serial.Serial``, both import paths' ``tlelib.canlink.CanLink.open`` and
``MocapRx.start`` (the NatNet path) are replaced by functions that record the
attempt and raise.  The CAN dongle may be plugged into this machine; nothing
here can open it.  One deliberate attempt is made on a fake ``COM999`` label to
prove the poison sits on the GUI's own connect path, and the tallies are checked
at the end.

Phases, each recorded as checks and into
``hw_tests/results/gui_sim_<date>.json``:

1. the factory: a COM label gives an unopened ``tlelib`` ``Backend``, the SIM
   label an unopened ``SimMaster``;
2. the adapter list: SIM present, preselected by ``prefer_sim``, kept across a
   Refresh, never chosen by a Refresh on its own;
3. the twin mocap source refuses, with a message, before SIM is connected;
4. the poisoned COM connect fails and leaves no backend;
5. SIM connect: 24 boards scanned before any cycle (TLE/DVP at 0x101-0x108,
   7mm at 0x109-0x118), the twin receiver chosen and started;
6. the cycle: only 0x101 selected and enabled, its bar dragged to 12 psi; within
   5 s its pressure approaches 12 psi and joint 2 (``s1.u2``, positive for
   0x101 in the measured map) moves positive as the receiver reports it;
7. the kinematics line for the twin;
8. the spawned viewer on the twin's shared-array feed for ~5 s, GUI and viewer
   screenshots to ``hw_tests/media/gui_sim_*.png``;
9. STOP ALL, disconnect, receiver reaped, window closed, poison tallies.

What it does not show: that the twin moves like the arm (``twin_compare``'s
question), or anything about the real bus -- ``integrated_gui_test.py`` owns
that, on the hardware.

    .venv\\Scripts\\python.exe hw_tests\\gui_sim_test.py [--no-viewer]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
for _p in (WS / "TLE_PCB", WS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MEDIA = WS / "hw_tests" / "media"
RESULTS = WS / "hw_tests" / "results"

#: The board this test drives: the first TLE/DVP board, the positive member of
#: joint 2's antagonistic pair in ``canarm_actuators.MEASURED_JOINT_PAIRS``.
DRIVE_BASE = 0x101
DRIVE_JOINT = 2

#: Well inside the operator's envelope: one line, pair sum 12 psi of 30.
DRIVE_PSI = 12.0

#: "Approaches 12 psi": within this of the target by the deadline.  The TLE
#: board model settles to ~11.9 psi in half a second headless (2026-09-10).
PSI_TOL = 1.0
DEADLINE_S = 5.0

#: Joint motion that counts as "moved positive".  The fitted twin swings this
#: joint about +20 deg at 12 psi, so 2 deg cannot be noise and cannot be missed.
MIN_DQ_DEG = 2.0

#: How long the viewer runs on the twin feed before the screenshots.
VIEWER_S = 5.0

#: Screenshots are downscaled to at most this width, so the media folder stays
#: small; the viewer at native size is several hundred kilobytes per frame.
SHOT_MAX_W = 960

ATTEMPTS: dict[str, list] = {"serial.Serial": [], "CanLink.open": [],
                             "MocapRx.start": []}


def poison() -> None:
    """Make every hardware and network open in this process raise, and count it."""
    import serial

    def refuse_serial(*args, **kwargs):
        ATTEMPTS["serial.Serial"].append(args[:1])
        raise RuntimeError("gui_sim_test: serial.Serial is poisoned")

    serial.Serial = refuse_serial

    def refuse_open(self, *args, **kwargs):
        ATTEMPTS["CanLink.open"].append(getattr(self, "port", None))
        raise RuntimeError("gui_sim_test: CanLink.open is poisoned")

    from tlelib import canlink as bare_canlink
    bare_canlink.CanLink.open = refuse_open
    from TLE_PCB.tlelib import canlink as pkg_canlink
    pkg_canlink.CanLink.open = refuse_open

    def refuse_natnet(self, *args, **kwargs):
        ATTEMPTS["MocapRx.start"].append(type(self).__name__)
        raise RuntimeError("gui_sim_test: MocapRx.start (NatNet) is poisoned")

    from UMArm_MOCAP import mocap_rx
    mocap_rx.MocapRx.start = refuse_natnet


def pump(root, seconds: float, tick: float = 0.02) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update_idletasks()
        root.update()
        time.sleep(tick)


def log_text(app) -> str:
    return app.log_text.get("1.0", "end")


def shrink_and_save(image, name: str) -> str:
    MEDIA.mkdir(parents=True, exist_ok=True)
    if image.width > SHOT_MAX_W:
        h = int(round(image.height * SHOT_MAX_W / image.width))
        image = image.resize((SHOT_MAX_W, h))
    path = MEDIA / name
    # An adaptive 256-colour palette: a window of flat widgets and a rendered
    # checkerboard lose nothing a reader looks for, and the file shrinks by
    # about two thirds against full RGB.
    image.convert("RGB").quantize(colors=256).save(path, optimize=True)
    return str(path)


def grab_rect(rect, name: str) -> str | None:
    try:
        import ctypes

        from PIL import ImageGrab
    except Exception:
        return None
    user32 = ctypes.windll.user32
    left, top, right, bottom = (int(v) for v in rect)
    left, top = max(left, 0), max(top, 0)
    right = min(right, user32.GetSystemMetrics(0))
    bottom = min(bottom, user32.GetSystemMetrics(1))
    if right <= left or bottom <= top:
        return None
    return shrink_and_save(ImageGrab.grab(bbox=(left, top, right, bottom),
                                          all_screens=True), name)


def grab_root(root, name: str) -> str | None:
    root.update_idletasks()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    return grab_rect((x, y, x + root.winfo_width(), y + root.winfo_height()), name)


def set_topmost(hwnd: int, on: bool) -> None:
    """Put a window above every other one, or return it to the normal band.

    ``PIL.ImageGrab`` grabs a screen rectangle, not a window, so whatever is on
    top of that rectangle is what gets saved.  The first run of this test
    raised the viewer with ``SetForegroundWindow``, which Windows refuses to a
    process that does not own the foreground, and both screenshots came back as
    the operator's editor.  ``HWND_TOPMOST`` through ``SetWindowPos`` needs no
    foreground rights.  A private ``WinDLL`` instance, so the argtypes set here
    cannot change how any other caller in the process reaches the same call.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_uint)
    after = ctypes.c_void_p(-1 if on else -2)       # HWND_TOPMOST / NOTOPMOST
    user32.SetWindowPos(hwnd, after, 0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0040)   # NOSIZE | NOMOVE | SHOWWINDOW


def grab_root_on_top(root, name: str) -> str | None:
    """Screenshot the Tk window with it held above everything for the grab."""
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    try:
        pump(root, 0.5)
        return grab_root(root, name)
    finally:
        root.attributes("-topmost", False)


def run(args) -> int:
    poison()

    import tkinter as tk
    from tkinter import ttk

    import numpy as np

    import canarm_control_gui as GUI
    from digital_twin.sim_master import SimMaster
    from digital_twin.sim_mocap import SimMocap
    from tlelib.backend import Backend as BareBackend

    checks: list[tuple[str, bool, str]] = []
    record: dict = {"schema": "gui_sim_test/1",
                    "created_local": time.strftime("%Y-%m-%d %H:%M:%S")}

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"   {detail}" if detail else ""))
        return bool(ok)

    # ---- 1. the factory --------------------------------------------------
    print("1. the factory")
    real = GUI.make_backend("COM58", 1_000_000, log=None)
    check("a COM label builds tlelib's Backend, unopened",
          type(real) is BareBackend and not real.link.is_open
          and real.link._ser is None,
          f"{type(real).__module__}.{type(real).__name__}, is_open={real.link.is_open}")
    sim_lines: list[str] = []
    twin = GUI.make_backend(GUI.SIM_PORT, 1_000_000, log=sim_lines.append)
    check("the SIM label builds SimMaster, unopened, from the fitted twin",
          isinstance(twin, SimMaster) and not twin.link.is_open
          and any("canarm_flow.npz" in line for line in sim_lines),
          sim_lines[-1] if sim_lines else "no log line")
    record["twin_describe"] = sim_lines[-1] if sim_lines else None
    del real, twin
    check("constructing either opened nothing",
          not ATTEMPTS["serial.Serial"] and not ATTEMPTS["CanLink.open"])

    # ---- 2. the window and the adapter list --------------------------------
    print("2. the adapter list")
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = GUI.CanArmControllerApp(root, mocap_source="off", prefer_sim=True)
    pump(root, 0.5)
    values = list(root.tk.splitlist(app.port.cget("values")))
    check("the SIM adapter is in the list", GUI.SIM_LABEL in values, values)
    check("prefer_sim preselected it", app.port.get() == GUI.SIM_LABEL,
          app.port.get())
    app.refresh_ports()
    check("a Refresh keeps a SIM selection", app.port.get() == GUI.SIM_LABEL,
          app.port.get())
    app.port.set("")
    app.refresh_ports()
    auto = app.port.get()
    check("a Refresh never chooses SIM on its own", auto != GUI.SIM_LABEL,
          repr(auto))
    check("the SIM entry survives that Refresh",
          GUI.SIM_LABEL in root.tk.splitlist(app.port.cget("values")))

    # ---- 3. the twin source needs SIM --------------------------------------
    print("3. twin mocap before SIM")
    app.mocap_source.set("twin")
    app.toggle_mocap()
    pump(root, 0.3)
    check("twin mocap refuses without the SIM backend, and says why",
          app._mocap is None and "connect the" in log_text(app),
          log_text(app).strip().splitlines()[-1] if log_text(app).strip() else "")
    app.mocap_source.set("off")

    # ---- 4. the poison is on the GUI's own path ----------------------------
    print("4. a real adapter label, poisoned")
    app.port.set("COM999 - gui_sim_test poison check")
    app.toggle_connect()
    pump(root, 0.3)
    check("connecting a COM label reaches CanLink.open, which refuses",
          app.backend is None and ATTEMPTS["CanLink.open"] == ["COM999"],
          ATTEMPTS["CanLink.open"])

    # ---- 5. SIM connect ----------------------------------------------------
    print("5. SIM connect and scan")
    app.port.set(GUI.SIM_LABEL)
    t0 = time.perf_counter()
    app.toggle_connect()
    connect_s = time.perf_counter() - t0
    pump(root, 0.6)
    backend = app.backend
    check("SIM connect built and opened a SimMaster",
          isinstance(backend, SimMaster) and backend.link.is_open,
          f"{type(backend).__name__}, connect+scan {connect_s:.2f} s")
    if not isinstance(backend, SimMaster):
        root.destroy()
        return 1
    nodes = backend.snapshot_nodes()
    present = {b: n for b, n in nodes.items() if n.present}
    tle = [b for b, n in present.items() if n.kind == "TLE/DVP"]
    seven = [b for b, n in present.items() if n.kind == "7mm"]
    check("the scan found 24 boards before any cycle ran",
          len(present) == 24 and not backend.running and backend.stats.cycles == 0,
          f"{len(present)} boards, running={backend.running}")
    check("TLE/DVP at 0x101-0x108 and 7mm at 0x109-0x118",
          tle == list(range(0x101, 0x109)) and seven == list(range(0x109, 0x119)),
          f"TLE {len(tle)}, 7mm {len(seven)}")
    check("the window rendered 24 board rows", len(app.node_widgets) == 24,
          len(app.node_widgets))
    rx = app._mocap
    check("connecting SIM chose and started the twin receiver",
          app.mocap_source.get() == "twin" and isinstance(rx, SimMocap),
          f"source={app.mocap_source.get()!r} rx={type(rx).__name__}")
    q0 = rx.wait_fresh(timeout=2.0) if isinstance(rx, SimMocap) else None
    check("the twin receiver publishes q", q0 is not None)
    record["q0_deg"] = None if q0 is None else np.degrees(q0).tolist()

    # ---- 6. the cycle, one board -------------------------------------------
    print("6. the cycle: 0x101 to 12 psi")
    app.select_all(False)
    app.selected[DRIVE_BASE].set(True)
    app._selection_changed()
    app.toggle_cycle()
    app.enable_selected(True)
    # The operator's path: the channel's own bar, which calls _channel_moved.
    app.node_bars[DRIVE_BASE]._commit(DRIVE_PSI)
    targets = backend._build_targets()
    enabled = sorted(b for b, (_c, en) in targets.items() if en)
    check("only 0x101 goes out with its enable bit set", enabled == [DRIVE_BASE],
          [hex(b) for b in enabled])
    trace = []
    t_start = time.monotonic()
    c_start = backend.stats.cycles
    reached_s = None
    while time.monotonic() - t_start < DEADLINE_S:
        pump(root, 0.1)
        snap = backend.snapshot_nodes()[DRIVE_BASE]
        q = rx.get_q()
        dt = time.monotonic() - t_start
        dq = float(np.degrees(q[DRIVE_JOINT] - q0[DRIVE_JOINT])) if q is not None else float("nan")
        trace.append((round(dt, 2), round(snap.pressure_psi, 3), round(dq, 3)))
        if reached_s is None and abs(snap.pressure_psi - DRIVE_PSI) <= PSI_TOL:
            reached_s = dt
    elapsed = time.monotonic() - t_start
    stats = backend.stats
    rate_hz = (stats.cycles - c_start) / elapsed
    snap = backend.snapshot_nodes()[DRIVE_BASE]
    q1 = rx.get_q()
    dq = np.degrees(q1 - q0)
    check("0x101's pressure approached 12 psi within 5 s",
          reached_s is not None and abs(snap.pressure_psi - DRIVE_PSI) <= PSI_TOL,
          f"{snap.pressure_psi:.2f} psi, within {PSI_TOL} psi after "
          f"{reached_s if reached_s is None else round(reached_s, 2)} s")
    check("joint 2 moved positive through the twin receiver",
          float(dq[DRIVE_JOINT]) > MIN_DQ_DEG,
          f"dq[{DRIVE_JOINT}] = {dq[DRIVE_JOINT]:+.2f} deg")
    check("the receiver stayed fresh while the cycle ran",
          not rx.get_state().q_stale, f"{rx.get_state().fps:.1f} fps")
    print(f"     cycle {rate_hz:.1f} Hz achieved (asked 150), "
          f"replies {stats.replies}, misses {stats.misses}, late {stats.late_cycles}")
    record.update({
        "drive": {"base": hex(DRIVE_BASE), "psi": DRIVE_PSI,
                  "final_psi": snap.pressure_psi, "reached_s": reached_s,
                  "dq_deg": dq.tolist(), "trace_t_psi_dq2": trace},
        "cycle": {"achieved_hz": rate_hz, "cycles": stats.cycles,
                  "replies": stats.replies, "misses": stats.misses,
                  "late_cycles": stats.late_cycles,
                  "jitter_ms_p95": stats.jitter_ms_p95,
                  "jitter_ms_max": stats.jitter_ms_max},
        "mocap_fps": rx.get_state().fps,
        "status_line": app.status_var.get(),
    })

    # ---- 7. the kinematics line ----------------------------------------------
    print("7. the kinematics line")
    pump(root, 0.4)
    kin = app.kin_status.get()
    strip = app.mocap_status.get()
    print(f"     strip: {strip}")
    print(f"     kin:   {kin}")
    res = rx.fk_residual_m()
    check("the kinematics line reports the twin against fkine",
          kin.startswith("fkine vs twin model") and res is not None
          and float(np.max(res)) < 1e-4,
          f"worst {float(np.max(res)) * 1000:.4f} mm" if res is not None else kin)
    record["kinematics_line"] = kin
    record["mocap_strip"] = strip

    # ---- 8. the viewer on the twin feed --------------------------------------
    shots = {}
    if not args.no_viewer:
        print("8. the viewer on the twin's shared-array feed")
        from viz import viz_layout as VZ

        app.toggle_viewer()
        pub = app._viz_publisher
        check("opening the viewer on the twin source started the publisher",
              pub is not None and app._viewer_alive())
        seq0 = float(pub.arr[VZ.SEQ]) if pub is not None else 0.0
        viewer_rect = None
        t_v = time.monotonic()
        while time.monotonic() - t_v < VIEWER_S:
            pump(root, 0.25)
            if viewer_rect is None and app._viewer_proc is not None:
                try:
                    import integrated_gui_test as IGT            # noqa: F401
                except Exception:
                    IGT = None
                if IGT is not None:
                    wins = [w for w in IGT.windows_of_pid(app._viewer_proc.pid)
                            if w[2][2] - w[2][0] > 200]
                    if wins:
                        viewer_rect = wins[0]
        alive = app._viewer_alive()
        seq1 = float(pub.arr[VZ.SEQ]) if pub is not None else 0.0
        blk = VZ.read_robot(pub.arr, "canarm") if pub is not None else None
        twin_q = rx.get_q()
        check("the viewer ran on the twin feed for 5 s",
              alive and seq1 - seq0 > 100,
              f"alive={alive}, {seq1 - seq0:.0f} publishes")
        if blk is not None and twin_q is not None:
            gap = float(np.max(np.abs(np.asarray(blk["q"]) - twin_q)))
            check("the shared array carries the twin receiver's q",
                  gap < math.radians(2.0), f"max gap {math.degrees(gap):.3f} deg")
        shots["gui"] = grab_root_on_top(root, "gui_sim_window.png")
        if viewer_rect is not None:
            hwnd = viewer_rect[0]
            set_topmost(hwnd, True)
            try:
                pump(root, 0.6)
                wins = (IGT.windows_of_pid(app._viewer_proc.pid)
                        if app._viewer_proc else [])
                rect = wins[0][2] if wins else viewer_rect[2]
                shots["viewer"] = grab_rect(rect, "gui_sim_viewer.png")
            finally:
                set_topmost(hwnd, False)
        check("screenshots written", bool(shots.get("gui")) and bool(shots.get("viewer")),
              shots)
        app.toggle_viewer()
        pump(root, 0.5)
        check("closing the viewer stopped the publisher and left the cycle running",
              app._viz_publisher is None and backend.running)
    record["screenshots"] = shots

    # ---- 9. STOP ALL, disconnect, close ------------------------------------
    print("9. STOP ALL and disconnect")
    app.stop_all()
    pump(root, 0.3)
    after = backend._build_targets()
    check("STOP ALL cleared every enable bit and every target",
          not any(en for _c, en in after.values())
          and all(n.target_psi == 0.0 for n in backend.snapshot_nodes().values()))
    link = backend.link
    app.toggle_connect()
    pump(root, 0.6)
    check("disconnect closed the twin's link and dropped the backend",
          app.backend is None and not link.is_open)
    check("the twin receiver was reaped with its backend", app._mocap is None,
          app.mocap_status.get())
    app.on_close()
    check("no serial port, CAN link or NatNet socket was opened",
          not ATTEMPTS["serial.Serial"] and ATTEMPTS["CanLink.open"] == ["COM999"]
          and not ATTEMPTS["MocapRx.start"],
          ATTEMPTS)

    passed = sum(1 for _n, ok, _d in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed")
    record["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
    record["attempts"] = {k: [str(v) for v in vals] for k, vals in ATTEMPTS.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"gui_sim_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if passed == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-viewer", action="store_true",
                    help="skip the spawned viewer and the screenshots")
    return run(ap.parse_args())


if __name__ == "__main__":
    # integrated_gui_test's Win32 helpers are imported by module name.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
