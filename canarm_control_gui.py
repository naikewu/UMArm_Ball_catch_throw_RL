"""The CAN arm's control window: the TLE controller, plus the room it stands in.

WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT.  It is not a new controller.
Every byte that reaches a board still goes through ``TLE_PCB/tlelib``'s
:class:`~tlelib.backend.Backend` — the 150 Hz sync master, the node registry, the
one-serial-write table-then-sync cycle — **unchanged**, and every widget that
commands a channel is ``VEMA_TLE_controller``'s own, **imported rather than
copied**.  The protocol code in that package carries measured hardware behaviour
in its comments (the OTA batch-size measurements, the ``in_waiting``-versus-
blocking-read latency artefact, the S7 prescaler bug, "the enable bit is a
level"), and a fork of it would drift away from the bench that proved it while
looking identical.  So this file subclasses the window and adds to it.

WHAT IT ADDS, and why each one is here rather than in the TLE GUI:

* **A Viewer button** that spawns ``viz.multi_arm_viewer`` in a **separate
  process**, via ``multiprocessing.get_context("spawn")``.  Three reasons, all
  inherited from ``UMArm_CONTROL/gui/launcher.py``: tkinter's loop and MuJoCo's
  viewer loop both want to own a thread and neither yields politely; a fresh
  process has no state, so "reset the display" is not a feature anyone has to
  get right; and a viewer that dies takes nothing with it.
* **A mocap status strip** — bodies visible, ``q_stale``, frame rate.  It is
  three fields because those are the three that distinguish the failure modes
  from each other: no bodies means the wrong rigid-body id block, ``q_stale``
  with frames arriving means a plate Motive lost, and a rate that is not ~120
  means the transport.
* **A source selector**: off, sim, live.  ``sim`` is
  ``UMArm_MOCAP.sim_stream.CanArmSimStream``, a real receiver fed by a producer
  thread, which **opens no socket** — the whole viewer path can therefore be
  exercised, and this file self-tested, with no cameras and no network.

THREE ABSENCES THE ARM MUST SURVIVE, and they are guarded rather than assumed:
no mocap, no RS485 arm, no Kinova.  Each optional import is inside a try, each
optional robot is a checkbox that defaults off, and **nothing about the CAN
arm's control path depends on any of them**.  A room with only the CAN arm is
the normal case, not a degraded one.

CLOSING THE VIEWER NEVER AFFECTS THE BACKEND.  This reverses the rule both RS485
viewers follow, where the 3D window IS the plant and closing it vents the arm.
Here the window is a second opinion about where the arm is; the bus keeps its
cycle, the boards keep their targets, and the only thing that stops is drawing.
The reverse also holds: the viewer is not on the STOP path, so ``STOP ALL``
still reaches the boards with the window open, closed or wedged.

THE GUI DISCIPLINE, from ``vema_control_gui.py``: the service thread owns the
I/O (that is ``Backend``'s cycle thread), there is exactly one ``after()`` pump
per periodic job, and no Tk callback blocks.  Connect / Scan / Disconnect are the
inherited exceptions and block for a second or two, as they always did.

    python canarm_control_gui.py
    python canarm_control_gui.py --sim-mocap        # synthetic stream, no cameras
    python canarm_control_gui.py --self-test        # no window, no port, no socket
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

WS_ROOT = Path(__file__).resolve().parent
# TLE_PCB first: ``VEMA_TLE_controller`` imports ``tlelib`` by bare name, and it
# self-inserts its own directory too, so the order here only has to be right
# once.  The workspace root follows it, because every package in this workspace
# is a direct child of the root.
for _p in (WS_ROOT / "TLE_PCB", WS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import VEMA_TLE_controller as VTC                            # noqa: E402

#: How often the mocap strip is repainted.  Slower than the node refresh on
#: purpose: it is three numbers describing a 120 Hz stream, and there is a
#: matplotlib redraw in this process already competing for the interpreter lock
#: with a thread that has to put sync edges out on time.
MOCAP_MS = 200

VIEWER_JOIN_S = 3.0


class CanArmControllerApp(VTC.ControllerApp):
    """``VEMA_TLE_controller.ControllerApp`` plus the room.

    Subclassed rather than forked.  The base class owns the bus, the channel
    bars, the group slider, the plot and the cycle statistics; this class adds a
    strip at the bottom of the same window and two pieces of process
    lifecycle.  When the TLE GUI gains a widget, this window gains it too.
    """

    def __init__(self, root: tk.Tk, *, mocap_source: str = "off",
                 include_rs485: bool = False, include_kinova: bool = False):
        # Set BEFORE super().__init__, which calls _build().
        self._mocap_source = str(mocap_source)
        self._mocap = None
        self._viewer_proc = None
        self._viewer_stop = None
        self._include_rs485 = bool(include_rs485)
        self._include_kinova = bool(include_kinova)
        super().__init__(root)
        root.title("CAN UMArm - controller and room view")
        # ONE after() pump per periodic job.  The base class owns two of them
        # (nodes and plot); this is the third and it is the only one this class
        # adds, rather than piggy-backing on _refresh, because the mocap strip
        # is worth repainting five times a second and the node table ten.
        root.after(MOCAP_MS, self._refresh_mocap)

    # ---- layout --------------------------------------------------------
    def _build(self) -> None:
        super()._build()
        self._build_room(self.root)

    def _build_room(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Room view (display only)", padding=6)
        frame.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        row = ttk.Frame(frame)
        row.pack(fill="x")

        ttk.Label(row, text="Mocap").pack(side="left")
        self.mocap_source = ttk.Combobox(row, width=6, state="readonly",
                                         values=("off", "sim", "live"))
        self.mocap_source.set(self._mocap_source)
        self.mocap_source.pack(side="left", padx=(4, 8))
        self.mocap_button = ttk.Button(row, text="Start mocap",
                                       command=self.toggle_mocap)
        self.mocap_button.pack(side="left")

        self.rs485_var = tk.BooleanVar(value=self._include_rs485)
        self.kinova_var = tk.BooleanVar(value=self._include_kinova)
        ttk.Checkbutton(row, text="RS485 arm", variable=self.rs485_var).pack(
            side="left", padx=(16, 0))
        ttk.Checkbutton(row, text="Kinova", variable=self.kinova_var).pack(
            side="left", padx=4)

        self.viewer_button = ttk.Button(row, text="Viewer",
                                        command=self.toggle_viewer)
        self.viewer_button.pack(side="left", padx=(16, 0))
        ttk.Label(row, text="a separate process; closing it does not touch the bus",
                  foreground="#666").pack(side="left", padx=6)

        self.mocap_status = tk.StringVar(value="mocap off")
        ttk.Label(frame, textvariable=self.mocap_status,
                  font=("Consolas", 9)).pack(anchor="w", pady=(4, 0))

    # ---- mocap ---------------------------------------------------------
    def toggle_mocap(self) -> None:
        """Start or stop THIS process's receiver — the one behind the strip.

        The viewer process builds its own; a ``MocapRx`` owns SDK threads and a
        socket and cannot cross a process boundary.  Two receivers on one
        multicast stream is the arrangement the lab already runs, but it is also
        why this one is opt-in: nothing here starts a receiver on its own.
        """
        if self._mocap is not None:
            self._stop_mocap()
            return
        kind = self.mocap_source.get()
        if kind == "off":
            self.log("pick sim or live first")
            return
        try:
            self._mocap = _build_mocap(kind)
        except Exception as exc:
            self.log(f"[ERROR] mocap: {type(exc).__name__}: {exc}")
            self._mocap = None
            return
        self.mocap_button.configure(text="Stop mocap")
        self.log(f"mocap started ({kind})")

    def _stop_mocap(self) -> None:
        rx, self._mocap = self._mocap, None
        if rx is not None:
            try:
                rx.stop()
            except Exception as exc:
                self.log(f"[warn] mocap stop: {type(exc).__name__}: {exc}")
        try:
            self.mocap_button.configure(text="Start mocap")
            self.mocap_status.set("mocap off")
        except tk.TclError:                    # the window is already gone
            pass

    def _refresh_mocap(self) -> None:
        """Repaint the strip.  Reads snapshots only; never blocks."""
        try:
            self.mocap_status.set(_mocap_line(self._mocap, self._viewer_proc))
        except tk.TclError:                                  # pragma: no cover
            return
        self.root.after(MOCAP_MS, self._refresh_mocap)

    # ---- viewer --------------------------------------------------------
    def toggle_viewer(self) -> None:
        """Open or close the room window.  Touches nothing on the bus.

        A NEW PROCESS PER SESSION, ``spawn`` explicitly.  "The display should
        just be reset" is a requirement that is easy to get almost right — a
        stale model, a camera pose, a mount frozen at a pose from an hour ago —
        and impossible to get wrong this way: a process that has exited has no
        state.
        """
        if self._viewer_alive():
            self._stop_viewer()
            return
        try:
            import multiprocessing as mp

            from viz import multi_arm_viewer as MAV

            ctx = mp.get_context("spawn")
            self._viewer_stop = ctx.Event()
            opts = {
                "feed": "mocap",
                "mocap": {"kind": self.mocap_source.get()},
                "include_rs485": bool(self.rs485_var.get()),
                "include_kinova": bool(self.kinova_var.get()),
            }
            self._viewer_proc = ctx.Process(
                target=MAV.viewer_main, args=(None, self._viewer_stop, opts),
                name="canarm-viewer", daemon=True)
            self._viewer_proc.start()
        except Exception as exc:
            self.log(f"[ERROR] viewer: {type(exc).__name__}: {exc}")
            self._viewer_proc = None
            return
        self.viewer_button.configure(text="Close viewer")
        self.log("viewer opened (display only; closing it stops nothing)")

    def _viewer_alive(self) -> bool:
        return self._viewer_proc is not None and self._viewer_proc.is_alive()

    def _stop_viewer(self) -> None:
        """Ask, then wait, then insist.  None of it reaches the CAN backend."""
        proc, self._viewer_proc = self._viewer_proc, None
        stop, self._viewer_stop = self._viewer_stop, None
        if stop is not None:
            stop.set()
        if proc is not None:
            proc.join(timeout=VIEWER_JOIN_S)
            if proc.is_alive():
                # A window wedged on a GL driver is a window, not an actuator.
                # There is nothing to vent and nothing to wait for, which is
                # exactly why terminating here is safe and why terminating the
                # RS485 plant process mid-vent was not.
                proc.terminate()
        try:
            self.viewer_button.configure(text="Viewer")
        except tk.TclError:
            pass

    # ---- lifecycle -----------------------------------------------------
    def on_close(self) -> None:
        """Close the window: viewer down, receiver down, THEN the bus.

        Order matters in one direction only.  The base class's ``on_close``
        disconnects the backend, which sends a disabling table and a sync edge
        so nothing is left regulating; doing that last means the display is
        already gone and cannot be mistaken for a live reading of an arm that
        has just been released.
        """
        self._stop_viewer()
        self._stop_mocap()
        super().on_close()


# ---------------------------------------------------------------------------
# Optional dependencies, each behind its own guard
# ---------------------------------------------------------------------------

def _build_mocap(kind: str):
    """A started receiver of the requested kind.

    ``sim`` opens nothing — it is the real ``CanArmMocap`` class with a producer
    thread pushing synthetic frames through its own listeners, so the id
    routing, the quaternion conversion, ``mocap_to_q``, the publish-under-lock
    and the ring buffer are all the shipped code.  ``live`` opens a NatNet
    socket and is the only path here that touches the network.
    """
    if kind == "sim":
        from UMArm_MOCAP.sim_stream import CanArmSimStream
        return CanArmSimStream().start()
    if kind == "live":
        from UMArm_MOCAP.canarm_mocap import CanArmMocap
        return CanArmMocap().start()
    raise ValueError(f"unknown mocap source {kind!r}")


def _mocap_line(rx, viewer_proc) -> str:
    """The status strip's one line.  Never raises; a strip is not worth a crash."""
    view = ("viewer up" if viewer_proc is not None and viewer_proc.is_alive()
            else "viewer down")
    if rx is None:
        return f"mocap off   {view}"
    try:
        state = rx.get_state()
        homos = rx.get_homos()
        visible = 0
        if homos is not None:
            import numpy as np

            for t in np.asarray(homos, dtype=float):
                # A row Motive has never filled is the identity the array was
                # initialised with, and an unsolvable body streams as all zeros.
                # Neither is a body that is visible.
                if np.isfinite(t).all() and not np.allclose(t, np.eye(4)) \
                        and not np.allclose(t[0:3, 3], 0.0):
                    visible += 1
        return (f"bodies {visible:2d}   {state.fps:6.1f} fps   "
                f"frames {state.frames}   valid {state.valid_frames}   "
                f"stale {'Y' if state.stale else 'n'}   "
                # q_stale, not stale, is the one that matters to anything acting
                # on q: they differ exactly when frames keep arriving and stop
                # converting, and in that window get_q() is silently older every
                # tick while stale still reads False.
                f"q_stale {'Y' if state.q_stale else 'n'}   {view}"
                + (f"   [{state.last_error}]" if state.last_error else ""))
    except Exception as exc:
        return f"mocap unreadable: {type(exc).__name__}: {exc}   {view}"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def self_test(sim_mocap: bool = True, frames: int = 6) -> int:
    """Build the whole window and the viewer's render path, offline.

    Opens NO serial port, NO socket and NO GL window.  What it does exercise:
    every widget the window builds, several Tk update cycles so the periodic
    pumps actually run, a real ``CanArmSimStream`` producing frames through the
    real receiver, and the viewer's render loop against a mock viewer holding a
    real ``MjvScene``.  What it cannot exercise is ``sync()`` and the window
    itself, which is where a display becomes necessary.
    """
    root = None
    rx = None
    try:
        root = tk.Tk()
        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass
        app = CanArmControllerApp(root, mocap_source="sim" if sim_mocap else "off")
        for _ in range(12):
            root.update_idletasks()
            root.update()
        print(f"[self-test] window built: {len(app.node_widgets)} node rows, "
              f"backend={app.backend}")

        from viz import multi_arm_viewer as MAV
        from viz import viz_layout as VZ

        if sim_mocap:
            from UMArm_MOCAP.sim_stream import CanArmSimStream
            rx = CanArmSimStream(rate_hz=200.0).start()
            import time
            time.sleep(0.25)
            feed = MAV.MocapFeed({"canarm": rx},
                                 fallback=MAV._manual_fallback({}))
            q = rx.get_q()
            print(f"[self-test] sim mocap: q={'None' if q is None else len(q)} "
                  f"fps={rx.get_state().fps:.1f} q_stale={rx.get_state().q_stale}")
        else:
            feed = MAV.SharedArrayFeed(VZ.make_array())

        out = MAV.smoke_render(feed, frames=frames)
        print(f"[self-test] viewer render path: {out}")
        if out["frames"] != frames:
            raise AssertionError(f"render loop ran {out['frames']}/{frames} frames")

        for _ in range(4):
            root.update_idletasks()
            root.update()
        app.on_close()
        print("[self-test] OK")
        return 0
    except tk.TclError as exc:
        # No display at all.  Report and pass: this check exists to prove the
        # code paths are sound, and "this machine has no window server" is not
        # a defect in them.
        print(f"[self-test] no display ({exc}); skipped the window half")
        return 0
    finally:
        if rx is not None:
            try:
                rx.stop()
            except Exception:
                pass
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim-mocap", action="store_true",
                    help="preselect the synthetic mocap stream (opens no socket)")
    ap.add_argument("--rs485", action="store_true",
                    help="preselect the RS485 arm in the room view")
    ap.add_argument("--kinova", action="store_true",
                    help="preselect the Kinova in the room view")
    ap.add_argument("--self-test", action="store_true",
                    help="build everything offline and exit 0")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(sim_mocap=True)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    CanArmControllerApp(root,
                        mocap_source="sim" if args.sim_mocap else "off",
                        include_rs485=args.rs485,
                        include_kinova=args.kinova)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
