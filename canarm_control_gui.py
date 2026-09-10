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
* **A source selector**: off, sim, live, twin.  ``sim`` is
  ``UMArm_MOCAP.sim_stream.CanArmSimStream``, a real receiver fed by a producer
  thread, which **opens no socket** — the whole viewer path can therefore be
  exercised, and this file self-tested, with no cameras and no network.
* **A Lock plates button and a kinematics line**, added 2026-08-21.  ``live``
  now prefers ``CanArmMarkerMocap`` — ``q`` registered onto the four markers of
  each plate rather than read off Motive's manually aligned body frames, which
  on this arm sit about 45 deg round from the mechanism's axes.  That receiver
  needs one rest-time lock per plate, locks are scoped to a Motive session and
  so cannot be checked in, and minting them is a 3 s capture of a still arm:
  hence a button.  The kinematics line then reports the number the whole
  calibration exists to make small — **the distance between each measured
  u-joint centre and the one fkine predicts from the same frame's ``q``** —
  because an operator watching the arm move is exactly who can tell a
  kinematics error from a stream problem.
* **A SIM adapter**, added 2026-09-10: ``SIM - digital twin (no hardware)`` is
  always in the adapter list.  The operator's requirement was that this window
  can start a simulated arm through the same interface, and that the controller
  cannot tell which one it is driving.  That property is made true by where the
  difference lives rather than by care: **exactly one call differs**, the
  construction in :func:`make_backend`, which returns
  ``digital_twin.sim_master.SimMaster`` for the SIM label and ``tlelib``'s
  ``Backend`` for anything else.  ``SimMaster`` subclasses ``Backend`` and
  replaces only its link, so scan, select, the 150 Hz cycle, set_target,
  set_enabled, apply_tuning, stop_all, snapshot_nodes, history, stats,
  missing_nodes, disconnect and on_close are the same inherited code in both
  cases, down to the method bodies.  The twin is the fitted one,
  ``digital_twin.twin_params.load_twin_kwargs()``, and the log line it prints on
  connect names both checkpoint files.  The SIM adapter is never selected on
  its own in place of the resolved CAN dongle; ``--sim`` preselects it.
* **A ``twin`` mocap source**, the twin's counterpart of ``live``.  It is
  ``digital_twin.sim_mocap.SimMocap``, a ``CanArmMocap`` fed from the live
  twin's joints, so ``q`` reaches anything reading the strip through the same
  receiver interface as on the real arm.  It is chosen and started on its own
  when the SIM adapter connects, and stopped when that adapter disconnects.
  The room viewer, a separate process that cannot hold the twin, is fed
  through ``viz.viz_layout``'s shared array by a publisher thread here.

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

WHAT THE SIM ADAPTER DOES NOT MATCH: THE CYCLE RATE.  The interface a
controller calls is the metal's, method for method; its timing is not.  The
twin's physics (about 0.4-0.5 s of CPU per simulated second, measured headless)
runs in this process, on the interpreter lock, largely on ``Backend``'s own
cycle thread inside ``SimCanLink.send_batch``, where it competes with the Tk
loop, the plot and the twin mocap producer.  Measured 2026-09-10 in this window
with one board enabled at 12 psi: 87 Hz achieved and 59 % of that board's
replies missed, against 148.6 Hz and essentially none missed on the real bus.
Shortening the interpreter's switch interval to 0.5 ms cut the misses to 1 %
and the rate to 62 Hz, and it would change thread scheduling for the real bus
too, so it is not applied.  A controller that watches ``stats`` or
``missing_nodes`` can tell the two apart until the physics leaves this
process's lock.

NO SIM-ONLY GUARD.  The operator's 30 psi envelope (``digital_twin/CONTRACT.md``
section 8: each line <= 30 psi, each antagonistic pair's sum <= 30 psi) is
enforced by ``collection/safety.py`` for campaigns, and **not** by this window
or by the inherited TLE window, whose bars and group slider reach
``VEMA_TLE_controller.TARGET_MAX_PSI`` = 40 psi with no pair check.  Adding the
check for the SIM adapter alone would make the twin behave differently from
the metal, which is the one thing the SIM adapter must not do, so the gap is
left identical on both and reported instead.

THE GUI DISCIPLINE, from ``vema_control_gui.py``: the service thread owns the
I/O (that is ``Backend``'s cycle thread), there is exactly one ``after()`` pump
per periodic job, and no Tk callback blocks.  Connect / Scan / Disconnect are the
inherited exceptions and block for a second or two, as they always did.

    python canarm_control_gui.py
    python canarm_control_gui.py --sim              # preselect the digital twin
    python canarm_control_gui.py --sim-mocap        # synthetic stream, no cameras
    python canarm_control_gui.py --self-test        # no window, no port, no socket
"""

from __future__ import annotations

import argparse
import sys
import threading
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

#: The real ``Backend`` class, captured once at import.  :meth:`CanArmControllerApp.
#: toggle_connect` rebinds ``VTC.Backend`` for the duration of one call, and
#: :func:`make_backend` must construct the class itself rather than whatever the
#: name points at during that call, or the SIM branch would recurse into itself.
REAL_BACKEND = VTC.Backend

#: How often the mocap strip is repainted.  Slower than the node refresh on
#: purpose: it is three numbers describing a 120 Hz stream, and there is a
#: matplotlib redraw in this process already competing for the interpreter lock
#: with a thread that has to put sync edges out on time.
MOCAP_MS = 200

VIEWER_JOIN_S = 3.0

#: Seconds of stillness the Lock plates button captures.  Same window
#: ``marker_mocap.mint_locks`` defaults to; short enough that an operator holds
#: still for it, long enough that the per-marker standard deviation it gates on
#: is a real sample rather than three frames of luck.
LOCK_CAPTURE_S = 3.0

#: The port name the SIM adapter hands the factory.  ``toggle_connect`` takes the
#: first space-separated token of the adapter label as the port, so the label
#: below must begin with exactly this.  No Windows serial device is named
#: ``SIM``; they are all ``COMn``.
SIM_PORT = "SIM"

#: The SIM adapter's entry in the adapter list.  32 characters, which is the
#: inherited combobox's width.
SIM_LABEL = f"{SIM_PORT} - digital twin (no hardware)"

#: The mocap source selector's choices.  ``twin`` needs the SIM adapter.
MOCAP_SOURCES = ("off", "sim", "live", "twin")

#: Rate the twin publisher writes the viewer's shared array at.  The viewer's
#: own redraw rate (``multi_arm_viewer.DEFAULT_FPS``); anything faster is
#: overwritten before it is drawn.
VIZ_PUBLISH_HZ = 60.0


def make_backend(port: str, bitrate: int, log=None):
    """The one construction seam: a SIM twin for :data:`SIM_PORT`, the bus otherwise.

    Returns an **unopened** backend, exactly as ``Backend(...)`` does — the
    caller's ``open()`` is what touches anything, and for the SIM branch that is
    a delivery thread rather than a port.  The SIM branch loads the fitted twin
    through ``digital_twin.twin_params`` and logs which files it used, or an
    ``UNFITTED`` line per missing part; a malformed fit raises, and the inherited
    connect handler prints it as an ``[ERROR]``.

    ``batched_actuator=True`` is a speed choice, not a physics choice:
    ``test_sim_core.test_batched_and_scalar_flow_paths_agree`` pins the two flow
    paths together, and over 2 s of the fitted twin they agreed to 3.6e-15 deg
    and 1.5e-11 Pa.  It is set here because the twin's physics runs on this
    process's interpreter lock, on the cycle thread.  Measured 2026-09-10 in
    this window, one board at 12 psi: 68 Hz achieved with 74 % of that board's
    replies missed on the scalar path, 87 Hz with 59 % missed batched.  See the
    module docstring for what that leaves unmatched.
    """
    if str(port) == SIM_PORT:
        from digital_twin import twin_params as TP
        from digital_twin.sim_master import SimMaster

        kwargs = TP.load_twin_kwargs(log=log if log is not None else print)
        return SimMaster(SIM_PORT, bitrate, log=log, batched_actuator=True,
                         **kwargs)
    return REAL_BACKEND(port, bitrate, log=log)


def twin_arm(backend):
    """The ``SimArm`` behind a SIM backend, or None for anything else.

    The only place this window asks which kind of backend it holds, and it asks
    for the mocap side's benefit alone: the ``twin`` receiver reads the arm's
    joints, and nothing on the bus path consults this.
    """
    if backend is None:
        return None
    try:
        from digital_twin.sim_master import SimMaster
    except Exception:                                        # pragma: no cover
        return None
    return backend.arm if isinstance(backend, SimMaster) else None


class CanArmControllerApp(VTC.ControllerApp):
    """``VEMA_TLE_controller.ControllerApp`` plus the room.

    Subclassed rather than forked.  The base class owns the bus, the channel
    bars, the group slider, the plot and the cycle statistics; this class adds a
    strip at the bottom of the same window, two pieces of process lifecycle, and
    the SIM adapter's construction.  When the TLE GUI gains a widget, this
    window gains it too.
    """

    def __init__(self, root: tk.Tk, *, mocap_source: str = "off",
                 include_rs485: bool = False, include_kinova: bool = False,
                 prefer_sim: bool = False):
        # Set BEFORE super().__init__, which calls _build() and refresh_ports().
        self._mocap_source = str(mocap_source)
        self._mocap = None
        self._viewer_proc = None
        self._viewer_stop = None
        self._viz_publisher = None
        self._include_rs485 = bool(include_rs485)
        self._include_kinova = bool(include_kinova)
        self._prefer_sim = bool(prefer_sim)
        super().__init__(root)
        root.title("CAN UMArm - controller and room view")
        fit_to_work_area(root)
        # ONE after() pump per periodic job.  The base class owns two of them
        # (nodes and plot); this is the third and it is the only one this class
        # adds, rather than piggy-backing on _refresh, because the mocap strip
        # is worth repainting five times a second and the node table ten.
        root.after(MOCAP_MS, self._refresh_mocap)

    # ---- layout --------------------------------------------------------
    def _build(self) -> None:
        super()._build()
        self._build_room(self.root)

    def _render_nodes(self, nodes) -> None:
        """Re-fit after a scan, which is when the window's height is decided.

        Twenty-four board rows are roughly 250 px more than the eight the base
        window was laid out against, so the size that has to fit the desktop is
        not known until the bus has been enumerated.
        """
        super()._render_nodes(nodes)
        self.root.after_idle(lambda: fit_to_work_area(self.root))

    def _build_room(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Room view (display only)", padding=6)
        # BEFORE the inherited body, not after it.  ``pack`` hands out parcels
        # in the order the widgets were packed, so when the window is shorter
        # than the sum of its parts the last one packed is the one that gets
        # nothing.  The inherited body is the expandable half and this strip is
        # a fixed 48 px, so packing the strip first costs the body 48 px it can
        # spare and guarantees the three fields that say whether mocap is live
        # are on screen rather than under the taskbar.
        slaves = parent.pack_slaves()
        placing = {"side": "bottom", "fill": "x", "padx": 8, "pady": (0, 8)}
        if slaves:
            placing["before"] = slaves[0]
        frame.pack(**placing)

        row = ttk.Frame(frame)
        row.pack(fill="x")

        ttk.Label(row, text="Mocap").pack(side="left")
        self.mocap_source = ttk.Combobox(row, width=6, state="readonly",
                                         values=MOCAP_SOURCES)
        self.mocap_source.set(self._mocap_source)
        self.mocap_source.pack(side="left", padx=(4, 8))
        self.mocap_button = ttk.Button(row, text="Start mocap",
                                       command=self.toggle_mocap)
        self.mocap_button.pack(side="left")
        self.lock_button = ttk.Button(row, text="Lock plates",
                                      command=self.lock_plates)
        self.lock_button.pack(side="left", padx=(4, 0))

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
        self.kin_status = tk.StringVar(value="kinematics: mocap off")
        ttk.Label(frame, textvariable=self.kin_status,
                  font=("Consolas", 9)).pack(anchor="w")

    # ---- the adapter list and the construction seam --------------------
    def refresh_ports(self) -> None:
        """The inherited enumeration, plus the SIM adapter at the end of the list.

        Two things in the inherited method would otherwise defeat the SIM entry.
        It rebuilds ``values`` from the serial ports alone, which drops the
        entry on every Refresh; and it re-selects the resolved CAN dongle on
        every Refresh, which would silently move an operator who had chosen SIM
        onto the real bus.  So a SIM selection survives a Refresh, and nothing
        else about the inherited choice changes: SIM is appended last, and is
        selected on its own only once, when ``prefer_sim`` (the ``--sim`` flag)
        asked for it.
        """
        keep_sim = self.port.get() == SIM_LABEL
        super().refresh_ports()
        values = [v for v in self.root.tk.splitlist(self.port.cget("values"))
                  if v != SIM_LABEL]
        values.append(SIM_LABEL)
        self.port["values"] = values
        if keep_sim or self._prefer_sim:
            self.port.set(SIM_LABEL)
            self._prefer_sim = False

    def toggle_connect(self) -> None:
        """The inherited connect handler, with :func:`make_backend` constructing.

        ``ControllerApp.toggle_connect`` builds its backend with a bare
        ``Backend(port, bitrate, log=...)`` looked up in ``VEMA_TLE_controller``'s
        module namespace.  That name is rebound to :func:`make_backend` for the
        duration of this one call and restored in ``finally``, so the handler's
        own body -- the label split, the open, the button states, the scan that
        follows -- runs unchanged for both the bus and the twin, and the TLE
        file needs no seam of its own.  Tk runs this on its one thread, and no
        other code in the process constructs through that name, so nothing else
        can observe the rebinding.
        """
        was_connected = self.backend is not None
        saved = VTC.Backend
        VTC.Backend = make_backend
        try:
            super().toggle_connect()
        finally:
            VTC.Backend = saved
        if not was_connected and self.backend is not None:
            self._on_connected()

    def _on_connected(self) -> None:
        """Choose and start the twin receiver when the adapter is the twin.

        Mocap-side only: the bus path is already running identical code.  A
        receiver the operator started by hand is not replaced, because swapping
        a live receiver out from under the strip without being asked is a
        surprise, and the log says how to switch.
        """
        if twin_arm(self.backend) is None:
            return
        if self._mocap is not None:
            self.log("SIM adapter connected; a mocap receiver is already running, "
                     "so choose 'twin' and restart mocap to read the simulated arm")
            return
        self.mocap_source.set("twin")
        self.toggle_mocap()

    # ---- mocap ---------------------------------------------------------
    def toggle_mocap(self) -> None:
        """Start or stop THIS process's receiver — the one behind the strip.

        The viewer process builds its own for ``sim`` and ``live``; a
        ``MocapRx`` owns SDK threads and a socket and cannot cross a process
        boundary.  Two receivers on one multicast stream is the arrangement the
        lab already runs, but it is also why ``live`` is opt-in: nothing here
        starts a network receiver on its own.  ``twin`` opens nothing and is
        started automatically when the SIM adapter connects.
        """
        if self._mocap is not None:
            self._stop_mocap()
            return
        kind = self.mocap_source.get()
        if kind == "off":
            self.log("pick sim, live or twin first")
            return
        if kind == "twin" and twin_arm(self.backend) is None:
            self.log(f"twin mocap reads the simulated arm's joints: connect the "
                     f"'{SIM_LABEL}' adapter first")
            return
        try:
            self._mocap = _build_mocap(kind, backend=self.backend)
        except Exception as exc:
            self.log(f"[ERROR] mocap: {type(exc).__name__}: {exc}")
            self._mocap = None
            return
        self.mocap_button.configure(text="Stop mocap")
        self.log(f"mocap started ({kind})")

    def lock_plates(self) -> None:
        """Mint this Motive session's plate locks from a 3 s rest capture.

        THE ARM MUST BE STILL for the window, and the button says so in the log
        rather than in a modal: a dialog that blocks the Tk loop also blocks the
        node pump, and this window is attached to a bus that is holding
        pressure.  ``mint_locks`` refuses on its own evidence — too few frames
        with all four markers tracked, or a per-marker standard deviation above
        its stillness bound — and the refusal is printed rather than swallowed,
        because a lock minted from a moving arm is a template of a shape the
        plate never has again and every later frame then fails the RMS gate.

        ``x_mode="diagonal45"`` and no streamed reference: the whole point on
        this arm is that Motive's alignment is not the mechanism's, so the lock
        must be reproducible from the marker file alone.  The azimuth that turns
        the bracket frame into the body frame is the measured
        ``canarm_frames.PLATE_AZIMUTH_DEG``, applied per frame, not baked into
        the lock.

        Blocks for :data:`LOCK_CAPTURE_S`, which is the one deliberate exception
        to "no Tk callback blocks" in this file besides the inherited
        connect/scan — it is a button the operator pressed, and the alternative
        is a state machine across three ``after()`` hops for a three-second
        capture.
        """
        rx = self._mocap
        if rx is None:
            self.log("start mocap first; locks are minted from a live stream")
            return
        if self.mocap_source.get() != "live":
            self.log("locks need labeled markers, which only the live stream "
                     "carries; the sim and twin streams inject rigid-body poses "
                     "only")
            return
        self.log(f"hold the arm still: capturing {LOCK_CAPTURE_S:.0f} s ...")
        self.root.update_idletasks()
        try:
            from UMArm_MOCAP import canarm_mocap as CM
            from UMArm_MOCAP.marker_mocap import mint_locks, save_locks

            locks, report = mint_locks(rx, LOCK_CAPTURE_S, x_mode="diagonal45",
                                       log=self.log)
            if not report.get("ok"):
                for line in report.get("refusals", []):
                    self.log(f"[ERROR] lock refused -- {line}")
                return
            import os
            os.makedirs(os.path.dirname(CM.DEFAULT_TEMPLATE_PATH), exist_ok=True)
            save_locks(CM.DEFAULT_TEMPLATE_PATH, locks,
                       meta={"source": "canarm_control_gui Lock plates",
                             "seconds": LOCK_CAPTURE_S, "x_mode": "diagonal45"})
            self.log(f"locks written to {CM.DEFAULT_TEMPLATE_PATH}")
        except Exception as exc:
            self.log(f"[ERROR] lock plates: {type(exc).__name__}: {exc}")
            return
        # Restart so the running receiver is the marker one.  A receiver cannot
        # grow locks in place: its solve method is chosen by its class.
        self._stop_mocap()
        self.toggle_mocap()

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

    def _reap_twin_mocap(self) -> None:
        """Stop a twin receiver whose SIM backend is no longer this window's.

        The receiver stops publishing the moment the SIM link closes and would
        read stale within 0.25 s on its own, but a stale receiver bound to a
        twin that no longer exists is not worth keeping: reconnecting builds a
        new twin, and the old receiver can never see it.
        """
        rx = self._mocap
        bound = getattr(rx, "bound_backend", None)
        if rx is None or bound is None or bound is self.backend:
            return
        self._stop_mocap()
        self.log("twin mocap stopped: the SIM backend it read was disconnected")

    def _twin_q(self):
        """The twin receiver's newest ``q``, or None.  Any thread."""
        rx = self._mocap
        if rx is None or getattr(rx, "bound_backend", None) is None:
            return None
        return rx.get_q()

    def _refresh_mocap(self) -> None:
        """Repaint the strip.  Reads snapshots only; never blocks."""
        self._reap_viewer()
        self._reap_twin_mocap()
        try:
            self.mocap_status.set(_mocap_line(self._mocap, self._viewer_proc))
            self.kin_status.set(_kin_line(self._mocap))
        except tk.TclError:                                  # pragma: no cover
            return
        self.root.after(MOCAP_MS, self._refresh_mocap)

    def _reap_viewer(self) -> None:
        """Notice a viewer the operator closed at its own window.

        Closing the room window is the ordinary way to dismiss it, and nothing
        told this process about it: the handle stayed non-None and the button
        went on reading "Close viewer", so the next press spawned a *second*
        viewer instead of closing the one that was already gone.  Joining the
        dead child here also reaps it, rather than leaving the handle until
        someone happens to press the button.
        """
        proc = self._viewer_proc
        if proc is None or proc.is_alive():
            return
        proc.join(timeout=0.1)
        self._viewer_proc = None
        self._viewer_stop = None
        self._stop_viz_publisher()
        try:
            self.viewer_button.configure(text="Viewer")
        except tk.TclError:                                  # pragma: no cover
            pass
        self.log(f"viewer closed at its own window (exit {proc.exitcode}); "
                 f"the bus is untouched")

    # ---- viewer --------------------------------------------------------
    def toggle_viewer(self) -> None:
        """Open or close the room window.  Touches nothing on the bus.

        A NEW PROCESS PER SESSION, ``spawn`` explicitly.  "The display should
        just be reset" is a requirement that is easy to get almost right — a
        stale model, a camera pose, a mount frozen at a pose from an hour ago —
        and impossible to get wrong this way: a process that has exited has no
        state.

        THE FEED IS CHOSEN WHEN THE WINDOW OPENS.  For ``sim`` and ``live`` the
        child builds its own receiver, as before.  For ``twin`` it cannot: the
        twin is a ``SimArm`` in this process, so the child reads
        ``viz.viz_layout``'s shared array and a :class:`_VizPublisher` thread
        here writes the twin receiver's ``q`` into it.  Changing the mocap
        source while the window is open does not change its feed; close and
        reopen it.
        """
        if self._viewer_alive():
            self._stop_viewer()
            return
        try:
            import multiprocessing as mp

            from viz import multi_arm_viewer as MAV

            ctx = mp.get_context("spawn")
            self._viewer_stop = ctx.Event()
            kind = self.mocap_source.get()
            opts = {
                "feed": "mocap",
                "mocap": {"kind": kind},
                "include_rs485": bool(self.rs485_var.get()),
                "include_kinova": bool(self.kinova_var.get()),
            }
            arr = None
            if kind == "twin":
                from viz import viz_layout as VZ

                arr = VZ.make_array()
                opts.update({"feed": "shared", "robots": ("canarm",),
                             "mocap": {"kind": "off"}})
                self._viz_publisher = _VizPublisher(arr, self._twin_q).start()
            self._viewer_proc = ctx.Process(
                target=MAV.viewer_main, args=(arr, self._viewer_stop, opts),
                name="canarm-viewer", daemon=True)
            self._viewer_proc.start()
        except Exception as exc:
            self.log(f"[ERROR] viewer: {type(exc).__name__}: {exc}")
            self._viewer_proc = None
            self._stop_viz_publisher()
            return
        self.viewer_button.configure(text="Close viewer")
        self.log("viewer opened (display only; closing it stops nothing)"
                 + ("; fed from the twin through the shared array"
                    if self._viz_publisher is not None else ""))

    def _viewer_alive(self) -> bool:
        return self._viewer_proc is not None and self._viewer_proc.is_alive()

    def _stop_viz_publisher(self) -> None:
        pub, self._viz_publisher = self._viz_publisher, None
        if pub is not None:
            pub.stop()

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
        self._stop_viz_publisher()
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


class _VizPublisher:
    """Writes a ``q`` source into ``viz.viz_layout``'s shared array, at 60 Hz.

    The viewer is a spawned process and the twin lives in this one, so this is
    the bridge.  It writes the CAN arm's block only: the receiver's ``q``, in
    the order ``MocapFeed`` would hand the same receiver's ``q`` to the viewer,
    so the two feeds draw one ``q`` identically; the default mount
    (``viz.mjcf_canarm.DEFAULT_CANARM_MOUNT``, where the twin's own MJCF puts its
    base); and ``fresh=False``, because that mount is a configured pose rather
    than a measured one.  ``plates_ok`` stays False, since the twin has no
    measurement to overlay.  ``SEQ`` is bumped per write, so a stalled
    publisher is distinguishable from a still arm by anything that reads it.

    No lock, per ``viz_layout.make_array``: a torn frame costs one frame drawn
    from two instants.
    """

    def __init__(self, arr, source, rate_hz: float = VIZ_PUBLISH_HZ):
        self.arr = arr
        self.source = source
        self.period = 1.0 / float(rate_hz)
        self.writes = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "_VizPublisher":
        from viz import mjcf_canarm as MJ
        from viz import transforms as TF
        from viz import viz_layout as VZ

        xyz, rpy = MJ.DEFAULT_CANARM_MOUNT
        VZ.write_robot(self.arr, "canarm", mount_pos=xyz,
                       mount_quat=TF.rpy_to_quat(rpy), fresh=False,
                       plates_ok=False)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="canarm-viz-publisher")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        from viz import viz_layout as VZ

        while not self._stop.is_set():
            try:
                q = self.source()
            except Exception:
                q = None
            if q is not None:
                VZ.write_robot(self.arr, "canarm", q=q)
                self.arr[VZ.SEQ] = float(self.arr[VZ.SEQ]) + 1.0
                self.writes += 1
            self._stop.wait(self.period)


# ---------------------------------------------------------------------------
# Optional dependencies, each behind its own guard
# ---------------------------------------------------------------------------

def work_area() -> tuple[int, int, int, int] | None:
    """The desktop rectangle a window may occupy, or None if it is unknown.

    Tk clamps a window against ``winfo_screenheight``, i.e. the whole screen,
    and has no notion of a taskbar.  On this bench that difference is 48 px on
    a 1080 px display, and 48 px is exactly the height of the mocap strip -- so
    the one widget that reports whether the room view is being fed live data
    was the one sitting underneath the taskbar.  Windows-only and best-effort:
    anywhere the call is unavailable this returns None and the caller leaves
    the geometry alone.
    """
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        # SPI_GETWORKAREA = 0x0030
        if not ctypes.WinDLL("user32").SystemParametersInfoW(
                0x0030, 0, ctypes.byref(rect), 0):
            return None
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return None


def fit_to_work_area(root) -> None:
    """Shrink and nudge *root* so the whole window is on the visible desktop.

    Only ever makes the window smaller than it asked to be, and only moves it
    when its far edge would fall outside the work area, so an operator who has
    placed the window somewhere keeps it there.  A display large enough for the
    natural size is left completely untouched.
    """
    area = work_area()
    if area is None:
        return
    left, top, right, bottom = area
    try:
        root.update_idletasks()
        # ``wm geometry`` sizes the CLIENT area while the work area bounds the
        # whole decorated window, and the title bar between them is 31 px here.
        # Ignoring it puts the bottom 31 px back under the taskbar, which is
        # the strip again.  The offset reads as zero until the window is
        # mapped, which is why this is called again after the scan.
        pad_y = max(0, root.winfo_rooty() - root.winfo_y())
        pad_x = max(0, root.winfo_rootx() - root.winfo_x())
        width = min(root.winfo_reqwidth(), right - left - 2 * pad_x)
        height = min(root.winfo_reqheight(), bottom - top - pad_y)
        x = min(max(root.winfo_x(), left), max(left, right - width - 2 * pad_x))
        y = min(max(root.winfo_y(), top), max(top, bottom - height - pad_y))
        # minsize otherwise vetoes the shrink on a small desktop.
        root.minsize(min(root.minsize()[0], width), min(root.minsize()[1], height))
        root.geometry(f"{width}x{height}+{x}+{y}")
    except tk.TclError:                                      # pragma: no cover
        pass


def _build_mocap(kind: str, backend=None):
    """A started receiver of the requested kind.

    ``sim`` opens nothing — it is the real ``CanArmMocap`` class with a producer
    thread pushing synthetic frames through its own listeners, so the id
    routing, the quaternion conversion, ``mocap_to_q``, the publish-under-lock
    and the ring buffer are all the shipped code.  ``twin`` is the same class
    fed from the SIM *backend*'s arm, publishing only while that backend's link
    is open, and it carries ``bound_backend`` so the window can tell when the
    backend it read has gone.  ``live`` opens a NatNet socket and is the only
    path here that touches the network.

    ALL BRANCHES RETURN THE RECEIVER, never whatever ``start()`` handed back.
    The two ``start()`` methods disagree about that: ``CanArmSimStream.start``
    returns ``self``, while ``MocapRx.start`` returns the ``NatNetClient`` it
    just built.  Returning the client cost this window both of the things it
    does with a receiver — ``get_state`` for the strip, which turned into
    "mocap unreadable: AttributeError", and ``stop`` for the shutdown, which
    turned into a warning and left the SDK's non-daemon threads running for the
    rest of the session with no way to reach them.  Measured on the bench,
    2026-08-20.
    """
    if kind == "sim":
        from UMArm_MOCAP.sim_stream import CanArmSimStream
        rx = CanArmSimStream()
    elif kind == "twin":
        arm = twin_arm(backend)
        if arm is None:
            raise RuntimeError(f"the twin source needs the '{SIM_LABEL}' "
                               f"adapter connected")
        from digital_twin.sim_mocap import SimMocap
        link = backend.link
        rx = SimMocap(arm, alive=lambda: link.is_open)
        rx.bound_backend = backend
    elif kind == "live":
        from UMArm_MOCAP.canarm_mocap import (CanArmMarkerMocap, CanArmMocap,
                                              load_canarm_locks)
        try:
            locks = load_canarm_locks()
        except (FileNotFoundError, ValueError) as exc:
            # Not an error: locks are per Motive session, so the first run
            # after a recalibration legitimately has none.  Degrade to the
            # streamed frames and SAY the q is on the wrong azimuth, because a
            # streamed q looks entirely healthy while being 45 deg round from
            # the mechanism on this arm.
            rx = CanArmMocap()
            rx.lock_note = f"no marker locks ({exc.__class__.__name__})"
        else:
            rx = CanArmMarkerMocap(locks)
            rx.lock_note = ""
    else:
        raise ValueError(f"unknown mocap source {kind!r}")
    rx.start()
    return rx


def _kin_line(rx) -> str:
    """The kinematics line: is the model where the cameras say the arm is?

    Reports the distance between each measured u-joint centre and the centre
    ``fkine`` predicts from the same frame's ``q``.  Plate 0 is omitted because
    it is identically zero — the chain is anchored there — so the five numbers
    printed are the whole signal.  On the 2026-08-21 calibration this reads
    roughly 0.2 / 0.3 / 0.5 / 1.0 / 1.8 mm over a static pose; a number an order
    of magnitude larger means the locks are stale, a plate has lost a marker, or
    the azimuth calibration does not belong to this Motive session.

    FOR THE TWIN there are no cameras, so the same five numbers mean something
    narrower and the line says so: the twin model's own u-joint centres against
    fkine from the receiver's ``q`` (``SimMocap.fk_residual_m``).  That checks
    the twin's geometry against the kinematics, not the kinematics against the
    arm.  Built from the same parameter table, it read 0.00 mm on 2026-09-10.

    Never raises: this is a status line, and a status line is not worth a crash.
    """
    if rx is None:
        return "kinematics: mocap off"
    twin_residual = getattr(rx, "fk_residual_m", None)
    if callable(twin_residual):
        try:
            import numpy as np

            res = twin_residual()
            if res is None:
                return "kinematics: twin -- no q published yet"
            q = rx.get_q()
            mm = np.asarray(res) * 1000.0
            return ("fkine vs twin model (mm) "
                    + " ".join(f"u{p+1}{mm[p]:6.2f}" for p in range(1, len(mm)))
                    + f"   rms {float(np.sqrt((mm[1:] ** 2).mean())):5.2f}"
                    + (f"   |q| max {float(np.degrees(np.abs(q)).max()):5.1f} deg"
                       if q is not None else "")
                    + "   [twin geometry, not a measurement]")
        except Exception as exc:
            return f"kinematics unavailable: {type(exc).__name__}: {exc}"
    getter = getattr(rx, "get_marker_poses", None)
    if not callable(getter):
        note = getattr(rx, "lock_note", "") or "streamed frames"
        return (f"kinematics: {note} -- press Lock plates for the marker-"
                f"registered q that fkine is calibrated against")
    try:
        import numpy as np

        from UMArm_MOCAP import canarm_frames as CF

        poses = getter()
        if poses is None:
            stats = rx.solve_stats()
            worst = max(stats.last_rms_m.values(), default=float("nan"))
            return (f"kinematics: markers not solving "
                    f"({stats.solved}/{stats.frames} frames, worst template "
                    f"residual {worst * 1000:.2f} mm)")
        res = CF.fk_residual_m(poses)
        if res is None:
            return "kinematics: frame set does not convert to q"
        q = CF.q_from_plate_frames(poses)
        mm = np.asarray(res) * 1000.0
        return ("fkine vs mocap (mm) "
                + " ".join(f"u{p+1}{mm[p]:6.2f}" for p in range(1, CF.N_PLATES))
                + f"   rms {float(np.sqrt((mm[1:] ** 2).mean())):5.2f}"
                + (f"   |q| max {float(np.degrees(np.abs(q)).max()):5.1f} deg"
                   if q is not None else ""))
    except Exception as exc:
        return f"kinematics unavailable: {type(exc).__name__}: {exc}"


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
                #
                # KNOWN UNDERCOUNT, and it is the sim's rather than this test's:
                # ``sim_stream.plate_poses_from_q`` puts the base plate at the
                # origin with an identity orientation by default, so its row is
                # bit-identical to the never-filled sentinel and the strip reads
                # "bodies 5" on a six-body synthetic stream.  A live base plate
                # has a real pose and is counted.  Distinguishing the two for
                # certain needs a per-body "seen this frame" counter on
                # ``MocapRx``, which is what ``hw_tests/canarm_mocap_live.py``
                # computes for itself; the strip settles for the heuristic.
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

def _pump(root, seconds: float) -> None:
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update_idletasks()
        root.update()
        time.sleep(0.02)


def self_test(sim_mocap: bool = True, frames: int = 6) -> int:
    """Build the whole window and the viewer's render path, offline.

    Opens NO serial port, NO socket and NO GL window.  What it does exercise:
    every widget the window builds, several Tk update cycles so the periodic
    pumps actually run, a real ``CanArmSimStream`` producing frames through the
    real receiver, the viewer's render loop against a mock viewer holding a
    real ``MjvScene``, and the SIM adapter: connect, the inherited scan of the
    twin's 24 boards, the twin receiver that starts on its own, and disconnect.
    No cycle is started.  What it cannot exercise is ``sync()`` and the window
    itself, which is where a display becomes necessary;
    ``hw_tests/gui_sim_test.py`` drives the cycle and the viewer against the
    twin.
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

        app.port.set(SIM_LABEL)
        app.toggle_connect()
        _pump(root, 0.6)
        nodes = app.backend.snapshot_nodes() if app.backend is not None else {}
        present = sum(1 for n in nodes.values() if n.present)
        twin_q = app._twin_q()
        print(f"[self-test] SIM adapter: {type(app.backend).__name__}, "
              f"{present} boards, mocap source {app.mocap_source.get()!r}, "
              f"twin q {'None' if twin_q is None else len(twin_q)}")
        print(f"[self-test] {app.kin_status.get()}")
        if present != 24 or twin_q is None:
            raise AssertionError("the SIM adapter did not scan 24 boards and "
                                 "publish a twin q")
        app.toggle_connect()                    # disconnect
        _pump(root, 0.4)
        if app.backend is not None or app._mocap is not None:
            raise AssertionError("disconnecting SIM left a backend or a twin "
                                 "receiver behind")

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
    ap.add_argument("--sim", action="store_true",
                    help="preselect the SIM adapter: the fitted digital twin, "
                         "no hardware")
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
                        include_kinova=args.kinova,
                        prefer_sim=args.sim)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
