"""The cameras, read from inside the bridge process — optional, non-blocking.

WHY IT LIVES IN THE BRIDGE AND NOT BESIDE THE PANEL.  A world-frame move needs
four things in the same instant: the operator's direction, the arm's own pose,
the marker body's pose, and the two calibration transforms.  Measured in this
session, ``.venv_kinova`` imports ``kortex_api``'s ``Base_pb2`` (0.028 s) and
``UMArm_KINOVA.kinova_mocap`` (0.069 s) into ONE process without complaint, and
``MocapRx.start()`` then returns in 0.0107 s with 120.0 Hz of rigid body 1008
arriving.  So the receiver goes where the arm session already is, and the world
frame is arithmetic on local variables rather than a third process and a second
pipe.  The alternative — a receiver beside the GUI — would have put the mocap
and the arm on opposite sides of the pipe and made every world-frame command a
round trip whose two halves were sampled at different instants.

**OPTIONAL IS THE DEFAULT READING OF EVERY FAILURE HERE.**  No Motive, no
``UMArm_MOCAP``, the wrong client IP, a body that is not being streamed: each of
those is a *state* this class publishes, not an exception it raises.  The arm
stays fully usable in its own base frame in every one of them, which is what the
operator asked for — the mocap is an option on the panel, not a dependency of it.

Two operational riders, both measured in this session on this machine:

1. **The SDK's threads are not daemons.**  A receiver started and never stopped
   holds the interpreter open: a script that started one, printed, and returned
   from ``main`` was still alive at a 15 s timeout (``EXIT=124``).  So
   :meth:`MocapHalf.stop` is not housekeeping, it is the difference between a
   bridge that exits and a bridge the panel has to kill; ``arm_bridge.serve``
   calls it from its ``finally``.
2. **The SDK prints to stdout, from its own threads, at times of its
   choosing.**  ``NAT_CONNECT to Motive with 4 1 0 0`` and ``resetting requested
   version ...`` on connect, ``shutdown called`` / ``shutting down`` on stop.
   Wrapping only ``start()`` in a redirect is not enough and was measured not to
   be: in one of two runs the ``resetting requested version`` line arrived after
   the redirect had already been unwound and landed on the real stdout, which in
   the bridge is the JSON channel.  So the redirect here covers the receiver's
   whole LIFE, and :data:`arm_bridge._JSON_OUT` holds the JSON channel's file
   object from before any of it, so no redirect can move it.

WHAT THIS MODULE DOES NOT IMPORT AT MODULE LEVEL.  ``sensor_frame`` and
``check_pad_markers`` both import ``kinova_arm`` at their top, which pulls
``kortex_api``, so importing either here would make this module unimportable
under the ordinary interpreter — and the protocol above (start, stop, publish,
"no frames is normal") is exactly the part that ought to be testable there.
Every such import is therefore made inside the function that needs it, and the
two conversions that genuinely need the vendored Euler convention are reached
through an injectable seam (:class:`MocapHalf`'s ``pose_to_se3``) rather than a
module-level name.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:      # runs by path, not just -m
    sys.path.insert(0, os.path.dirname(_HERE))

#: How much of the ring :meth:`MocapHalf.state` summarises, seconds.  Half a
#: second is 60 frames at the 120.0 Hz measured on this rig, which is enough for
#: the peak-to-peak spread to mean something and short enough that an arm which
#: has just stopped moving reads as still within half a second of stopping.
WINDOW_S = 0.5

#: Below this many frames in the window the body is not "live".  Ten frames is a
#: twelfth of a second, the same floor :data:`UMArm_KINOVA.kinova_mocap.
#: MIN_CAPTURE_FRAMES` uses, and it keeps a body that flickers in and out of the
#: cameras from being reported as a usable world reference.
MIN_LIVE_FRAMES = 10

#: How stale the single-frame base solve may be before the world frame is no
#: longer offered, seconds.  ``X = T_world_base`` dies the moment the cart is
#: dragged and there is no way to notice that from the cameras alone — the live
#: body-1008 origin sits 89.96 mm from the nearest of the 21 campaign poses,
#: which separates "the arm moved" from "the cart moved" not at all.  So the
#: session solve is given a lifetime rather than being trusted indefinitely.
#: Fifteen minutes is long enough to be a working session and short enough that
#: an X solved before lunch is not still on offer after it.
BASE_SOLVE_TTL_S = 900.0

#: How many of the SDK's stdout lines are kept for the record.
SDK_NOTE_LINES = 8


class _StdoutTee(io.TextIOBase):
    """Where the SDK's prints go while a receiver is up: stderr, and a list.

    NOT ``os.devnull``.  The lines are the only evidence in the record that the
    NatNet client got as far as talking to Motive, and on the failure this class
    exists to make survivable — Motive not running — they are the whole
    diagnosis.  They go to stderr because ``bridge_client`` already folds stderr
    into ``log`` messages, and to a capped list because the panel's log box is
    four lines tall and the same four SDK lines on every connect would own it.

    THE SINK IS RESOLVED AT WRITE TIME AND IS ``sys.__stderr__``, not a file
    object captured when this was built.  This object is installed as
    ``sys.stdout`` for as long as a receiver lives and is written to from the
    SDK's own threads, so a captured handle is a handle that can outlive whatever
    owned it — under pytest, ``sys.stderr`` is a per-test capture object that is
    closed when the test ends, and writing into one of those from an SDK thread
    during a later test's garbage collection took the whole interpreter down with
    a Windows ``STATUS_BREAKPOINT``.  ``sys.__stderr__`` is the process's own
    handle: in the bridge it is the pipe to the parent, and it does not go away.
    """

    def __init__(self, notes: list):
        self.notes = notes

    @property
    def sink(self):
        fh = sys.__stderr__
        return sys.stderr if fh is None else fh          # pythonw has no stderr

    def write(self, s):
        for line in str(s).splitlines():
            line = line.strip()
            if line:
                self.notes.append(line)
                del self.notes[:-SDK_NOTE_LINES]
        try:
            return self.sink.write(s)
        except Exception:
            return len(s)

    def flush(self):
        try:
            self.sink.flush()
        except Exception:
            pass


def world_dir_to_base(X, d):
    """A direction written in the CAMERAS' axes, in the arm's base axes.

    ``X = T_world_base``, so a vector carried in world coordinates arrives in
    base coordinates as ``R_X^T v``.  The translation of ``X`` plays no part: a
    direction is not a point, and adding the base's position to it is the single
    most obvious way to get a world-frame jog subtly and consistently wrong.

    The same expression serves an angular velocity, since an angular velocity is
    a free vector and rotates exactly as a direction does.
    """
    R = np.asarray(X, dtype=float)[0:3, 0:3]
    v = np.asarray(d, dtype=float).ravel()[0:3]
    return (R.T @ v).tolist()


def latest_analysis_path(root: str | None = None) -> str:
    """The newest ``results/*/analysis.json``, by folder name.

    The same walk as :func:`UMArm_KINOVA.check_pad_markers.latest_analysis`,
    repeated here for one reason: that module imports ``kinova_arm`` at its top
    and is therefore unreachable from the ordinary interpreter, and this one has
    to be importable there.  It is nine lines of ``listdir`` and the campaign
    folders are timestamp-named, so the duplication is cheap and cannot drift
    unnoticed: ``test_kinova_offline`` checks the two return the same path, on
    the one interpreter where both modules can be imported at once.
    """
    root = os.path.join(_HERE, "results") if root is None else root
    found = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, name, "analysis.json")
        if os.path.exists(path):
            found.append(path)
    if not found:
        raise FileNotFoundError(
            "no calibration under %s — run calibrate_mocap.py first" % root)
    return found[-1]


def load_calibration(path: str | None = None) -> dict:
    """``X``, ``Y`` and the path they came from.  Plain JSON, no ``kortex_api``.

    ``X`` is loaded and reported but is NOT to be used for a move: it is where
    the cart was parked during the campaign, and it dies the moment the cart is
    dragged.  It is here so :meth:`MocapHalf.solve_base` has something to
    disagree with, and that disagreement is the health indicator for the whole
    world-frame option.
    """
    path = latest_analysis_path() if path is None else path
    with open(path, encoding="utf-8") as fh:
        an = json.load(fh)
    return {"X_stored": np.array(an["X_world_base"], dtype=float),
            "Y": np.array(an["Y_tool_rb"], dtype=float),
            "path": path, "name": os.path.basename(os.path.dirname(path))}


def _pose_to_se3(pose):
    """The vendored driver's Euler convention, imported at the last moment.

    ``vendor/kinova_driver`` imports ``kortex_api`` at its top, so this name
    cannot be bound at import time in a module that has to load under the
    ordinary interpreter.  The convention itself — intrinsic-xyz, degrees — is
    the driver's and is not re-implemented here; re-implementing it is how two
    parts of one repo end up meaning different things by ``theta_y``.
    """
    from UMArm_KINOVA.vendor.kinova_driver import pose_to_SE3

    return np.asarray(pose_to_SE3(pose), dtype=float)


class MocapHalf:
    """One optional :class:`~UMArm_KINOVA.kinova_mocap.KinovaMocapRx`, off-thread.

    Every method returns; none of them raises at the caller.  The one thing this
    class promises the bridge is that :meth:`start` costs the calling thread a
    thread creation and nothing else, because it is called from
    ``_cmd_connect``, and a connect that waited on a socket to a machine where
    Motive is not running would be a connect that hung on the one thing the
    operator did not ask for.
    """

    def __init__(self, rx_factory=None, pose_to_se3=None, window_s=WINDOW_S,
                 analysis_path: str | None = None):
        #: How the receiver is built.  Defaults to the real
        #: :class:`KinovaMocapRx`; a test injects a stand-in, exactly as
        #: :class:`arm_bridge.Bridge` takes an ``arm_factory``.
        self.rx_factory = rx_factory
        self.pose_to_se3 = pose_to_se3 or _pose_to_se3
        self.window_s = float(window_s)
        self.analysis_path = analysis_path

        self.rx = None
        #: What the OPERATOR asked for, which is not what is running: a receiver
        #: takes a moment to construct and may never see a frame.  Published
        #: separately from ``live`` so "off" and "on but nothing is arriving" are
        #: two different lines in the panel rather than one ambiguous one.
        self.wanted = False
        self.error = ""
        self.sdk_notes: list[str] = []
        self.calib: dict | None = None
        self.calib_error = ""
        #: ``X`` solved from one frame THIS SESSION, and when.  Never the stored
        #: one — see :func:`load_calibration`.
        self.X_session = None
        self.base_solve: dict | None = None

        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        #: The real ``sys.stdout`` while the SDK's is swapped out from under it.
        self._saved_stdout = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Ask for a receiver.  Returns immediately; may never succeed.

        The construction and ``rx.start()`` go on a worker even though both were
        measured at 11 ms on this machine, because that measurement was taken
        with Motive UP.  What a socket does when the far end is absent is not a
        property this bridge should discover on the thread that is holding the
        arm session.
        """
        with self._lock:
            if self.wanted:
                return
            self.wanted = True
            self.error = ""
            worker = threading.Thread(target=self._open, name="kinova-mocap",
                                      daemon=True)
            self._worker = worker
        worker.start()

    def _open(self) -> None:
        rx = None
        try:
            factory = self.rx_factory
            if factory is None:
                from UMArm_KINOVA.kinova_mocap import KinovaMocapRx

                factory = KinovaMocapRx
            rx = factory()
            self._redirect_stdout()
            rx.start()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._restore_stdout()
            try:
                if rx is not None:
                    rx.stop()
            except Exception:
                pass
            return
        with self._lock:
            if not self.wanted:
                # A stop landed while this thread was inside ``rx.start()``.
                # NOTHING ELSE WILL EVER STOP THIS RECEIVER: ``stop`` has already
                # run and cleared the handle, and an SDK thread that is not a
                # daemon holds the process open for good.  So stop it here.
                self._restore_stdout()
                try:
                    rx.stop()
                except Exception:
                    pass
                return
            self.rx = rx

    def stop(self) -> None:
        """Shut the receiver down.  Idempotent, and safe from any thread.

        The starting worker is JOINED rather than abandoned.  A stop that raced a
        start would otherwise leave a receiver that no handle points at, and the
        SDK's non-daemon threads then keep the bridge alive after ``bye`` — the
        15 s hang measured in this module's docstring, except that it is now the
        panel's Disconnect that never completes.
        """
        with self._lock:
            self.wanted = False
            worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive() \
                and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        with self._lock:
            rx, self.rx = self.rx, None
        if rx is not None:
            try:
                rx.stop()
            except Exception as exc:
                self.error = f"stop: {type(exc).__name__}: {exc}"
        self._restore_stdout()

    # -- the stdout redirect ----------------------------------------------

    def _redirect_stdout(self) -> None:
        if getattr(self, "_saved_stdout", None) is not None:
            return
        self._saved_stdout = sys.stdout
        sys.stdout = _StdoutTee(self.sdk_notes)

    def _restore_stdout(self) -> None:
        saved = getattr(self, "_saved_stdout", None)
        if saved is None:
            return
        self._saved_stdout = None
        if isinstance(sys.stdout, _StdoutTee):
            sys.stdout = saved

    # -- what the panel sees ----------------------------------------------

    def window(self) -> list:
        """The last :data:`WINDOW_S` of body 1008.  Never sleeps.

        Deliberately NOT ``rx.capture()``, which sleeps for its whole averaging
        window: this runs on the bridge's publish thread ten times a second, and
        a publisher that sleeps for a second is a panel that stops repainting.
        """
        rx = self.rx
        if rx is None:
            return []
        try:
            return rx.kinova_window(t0=time.monotonic() - self.window_s)
        except Exception:
            return []

    def state(self) -> dict:
        """The mocap's whole published state.  Never raises."""
        out = {"on": bool(self.wanted), "live": False, "frames": 0,
               "rate_hz": 0.0, "error": self.error,
               "pos_m": None, "rotvec_deg": None,
               "pos_ptp_mm": None, "ang_ptp_deg": None,
               "calibration": None, "base_solve": self.base_solve,
               "world_ready": False}
        if self.sdk_notes:
            out["sdk_last"] = self.sdk_notes[-1]
        cal = self.calibration()
        if cal is not None:
            out["calibration"] = cal["name"]
        elif self.calib_error:
            out["calibration_error"] = self.calib_error
        rx = self.rx
        if rx is None:
            return out
        try:
            out["frames"] = int(rx.kinova_frames)
        except Exception:
            pass
        rows = self.window()
        out["rate_hz"] = len(rows) / self.window_s if self.window_s > 0 else 0.0
        if len(rows) < MIN_LIVE_FRAMES:
            return out
        try:
            from UMArm_KINOVA import kinova_mocap as KM
            from UMArm_KINOVA import mocap_calibration as MC

            s = KM.summarise(rows)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        out["live"] = True
        out["pos_m"] = [float(v) for v in s["pos_m"]]
        # AN AXIS-ANGLE VECTOR, NOT AN EULER TRIPLE.  The arm's own pose is
        # reported in the driver's intrinsic-xyz degrees and the panel prints it
        # verbatim; the marker body has no such native convention, and inventing
        # one here would be a second, unlike set of angles for the operator to
        # confuse with the first.  A rotation vector has no branch cut and no
        # ordering to get wrong.
        out["rotvec_deg"] = [float(v) for v in
                             np.degrees(MC.rot_log(s["T"][0:3, 0:3]))]
        out["pos_ptp_mm"] = float(s["pos_ptp_mm"])
        out["ang_ptp_deg"] = float(s["ang_ptp_deg"])
        out["world_ready"] = bool(self.world_ready())
        return out

    def calibration(self) -> dict | None:
        """``X_stored``, ``Y`` and the campaign's name, loaded once, lazily.

        A missing or unreadable ``analysis.json`` is recorded and returned as
        ``None``: the robot frame does not need it and must not be taken away by
        it.
        """
        if self.calib is None and not self.calib_error:
            try:
                self.calib = load_calibration(self.analysis_path)
            except Exception as exc:
                self.calib_error = f"{type(exc).__name__}: {exc}"
        return self.calib

    def world_ready(self) -> bool:
        """Whether a world-frame command may be accepted, and it takes THREE.

        Frames arriving now, ``Y`` loaded from a campaign, and an ``X`` solved
        in this session and not yet stale.  Any one of them missing leaves the
        panel in robot-base mode, which is the mode that needs nothing.
        """
        if self.X_session is None or self.base_solve is None:
            return False
        if (time.monotonic() - float(self.base_solve.get("t_mono", 0.0))
                > BASE_SOLVE_TTL_S):
            return False
        if self.calibration() is None:
            return False
        return len(self.window()) >= MIN_LIVE_FRAMES

    # -- the world frame ---------------------------------------------------

    def solve_base(self, tool_pose) -> dict:
        """``X = T_world_base`` from ONE still frame plus the arm's own pose.

        ``X = N Y^-1 M^-1`` (:func:`UMArm_KINOVA.mocap_calibration.
        base_from_single_sample`), with ``M`` the tool pose the arm reports and
        ``N`` the mean of the last :data:`WINDOW_S` of body 1008.

        WHY THIS IS RE-DERIVED RATHER THAN READ FROM THE CAMPAIGN.  ``Y`` is a
        property of how the markers are glued to the tool and survives being
        wheeled about; ``X`` is where the cart is parked and does not.  Nothing
        the cameras can see distinguishes a cart that has been dragged from an
        arm that has moved, so the stored ``X`` is treated as a number to
        disagree with rather than a number to move on.

        The disagreement is returned in millimetres and degrees and is the honest
        health indicator for the world frame.  Reference points, measured
        elsewhere and quoted rather than re-measured here: on a campaign sample
        the single-frame solve reproduced the jointly fitted ``X`` to 0.891 mm
        and 0.1297 deg, and the campaign's own ``base_recovery`` spread is
        2.07 mm RMS with a 6.97 mm maximum.  A disagreement of that order means
        the cart has not moved; a disagreement of centimetres means it has, and
        the newly solved ``X`` is the one to believe.

        Returns ``{"ok": False, "error": ...}`` rather than raising when there is
        nothing to solve from — no frames, no ``Y``, no pose.
        """
        from UMArm_KINOVA import mocap_calibration as MC

        cal = self.calibration()
        if cal is None:
            return {"ok": False,
                    "error": "no calibration to take Y from: "
                             + (self.calib_error or "none found")}
        rows = self.window()
        if len(rows) < MIN_LIVE_FRAMES:
            return {"ok": False,
                    "error": "only %d frames of rigid body 1008 in the last "
                             "%.2f s (need %d) — is Motive streaming and the "
                             "body visible?" % (len(rows), self.window_s,
                                                MIN_LIVE_FRAMES)}
        if tool_pose is None or len(tool_pose) < 6:
            return {"ok": False, "error": "no tool pose to solve from"}
        try:
            from UMArm_KINOVA import kinova_mocap as KM

            s = KM.summarise(rows)
            M_i = self.pose_to_se3(list(tool_pose))
            X = MC.base_from_single_sample(M_i, s["T"], cal["Y"])
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out = {
            "ok": True,
            "t_mono": time.monotonic(),
            "frames": len(rows),
            "pos_ptp_mm": float(s["pos_ptp_mm"]),
            "ang_ptp_deg": float(s["ang_ptp_deg"]),
            "pos_m": [float(v) for v in X[0:3, 3]],
            "vs_stored_mm": float(np.linalg.norm(
                X[0:3, 3] - cal["X_stored"][0:3, 3]) * 1e3),
            "vs_stored_deg": float(MC.angle_between_deg(
                cal["X_stored"][0:3, 0:3], X[0:3, 0:3])),
            "calibration": cal["name"],
        }
        self.X_session = X
        self.base_solve = out
        return out

    def to_base(self, d_world):
        """A world direction in base axes, or ``None`` if X is not solved."""
        if self.X_session is None:
            return None
        return world_dir_to_base(self.X_session, d_world)


__all__ = ["MocapHalf", "world_dir_to_base", "load_calibration",
           "latest_analysis_path", "WINDOW_S", "MIN_LIVE_FRAMES",
           "BASE_SOLVE_TTL_S"]
