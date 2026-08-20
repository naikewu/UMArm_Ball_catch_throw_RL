"""The Gen3 on the other side of a pipe — JSON lines in, JSON lines out.

WHY A BRIDGE AT ALL.  ``kortex_api`` pins protobuf 3.5.1 and lives only in
``.venv_kinova`` (see :mod:`UMArm_KINOVA.setup_env`); the collision bench's GUI
runs under the ordinary interpreter, with a MuJoCo and a NumPy that 2017's
protobuf would not survive sharing a process with.  So the real arm is reached
the way every other cross-runtime device is reached: a child process on the
interpreter that CAN import the driver, and one line of JSON per message.

Run it directly to talk to the arm by hand::

    .venv_kinova/Scripts/python.exe UMArm_KINOVA/arm_bridge.py
    {"cmd": "connect"}
    {"cmd": "arm", "on": true}
    {"cmd": "jog", "d_xyz": [0, 0, 0.01]}
    {"cmd": "jog_start", "d_xyz": [0, 0, 1], "speed": "slow"}
    {"cmd": "jog_stop"}
    {"cmd": "stop"}

THE SAFETY MODEL, and it is the whole reason this file is longer than a socket
wrapper:

* **Connect and arm are two steps.**  ``connect`` opens the session and starts
  reporting feedback; it moves nothing and cannot.  ``arm`` is a separate
  command, it centres :meth:`SafeKinovaArm.arm_envelope` on wherever the arm is
  standing at that instant, and every motion command is refused until it has
  been sent.  Disarming is instant and does not need the arm to be still.
* **One motion at a time, on a worker.**  ``move_joints`` and ``move_to_pose``
  block for as long as the move takes, and a loop that blocked in them could
  not read ``stop``.  Motions run on a single worker thread; the reader thread
  and the publisher keep running, so ``stop`` always lands.  A stop that lands
  in the window BEFORE the driver has issued its action -- the vendored driver
  spends three RPC round trips getting there -- would abort nothing, so the
  worker carries a stop epoch and refuses, or re-stops, on either side of it.
* **A completion flag is not an arrival.**  ``ACTION_ABORT`` sets the driver's
  event exactly as ``ACTION_END`` does, and a timed-out wait returns False with
  the action still running.  Every motion path here checks both.
* **A joint move is leashed too.**  ``SafeKinovaArm``'s envelope is Cartesian
  and ``move_joints`` leaves Cartesian space entirely, which is exactly why the
  wrapper refuses it by name.  Reaching it deliberately here means owning the
  limit here: :data:`MAX_JOINT_STEP_DEG` refuses a joint command that would move
  any single actuator further than that from where it is now, so a slider
  dragged across its whole travel arrives as a refusal rather than as a
  half-turn of the wrist.
* **One thing streams, and it is a dead-man.**  ``jog_start`` holds a Cartesian
  twist open for as long as the panel keeps sending heartbeats, which is what a
  press-and-hold button is; a twist has no end of its own and the vendored
  driver has no watchdog, so the ending is this file's job.  Three guards, and
  all three have to fail before the arm keeps moving: the jog expires
  :data:`JOG_DEADMAN_S` after the last heartbeat, the thread dies on the stop
  epoch and issues an ``arm.stop()`` on its way out, and every slice re-checks
  the live pose against the keep-out ball in
  :meth:`~UMArm_KINOVA.kinova_arm.SafeKinovaArm.twist_checked`.
* **The cameras are OPTIONAL and never block.**  ``connect`` asks for a mocap
  receiver on a worker thread and does not wait for it; a receiver that never
  sees a frame is a published state, not an error, and the arm stays fully
  usable in its own base frame without one.

Every reply is one JSON object on its own line with a ``kind``:

    ``state``   the arm's feedback, published at :data:`PUBLISH_HZ`
    ``ok``      a command was accepted (``cmd`` names which)
    ``error``   a command was refused or a move failed (``cmd``, ``error``)
    ``log``     a human-readable line
    ``bye``     the bridge is exiting
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:      # runs by path, not just -m
    sys.path.insert(0, os.path.dirname(_HERE))

#: How often the arm's feedback goes out, Hz.  One ``RefreshFeedback`` per
#: publish, which is the same call the calibration campaign samples at; 10 Hz is
#: fast enough for a panel and slow enough to leave the Kortex session alone.
PUBLISH_HZ = 10.0

#: How far one command may move a single actuator, degrees.  A GUI slider spans
#: the joint's whole range, so without this a mis-drag is a full-speed traverse.
#: 30 deg is about a second of travel at the Gen3's default joint speed and is
#: small enough to watch happen.
MAX_JOINT_STEP_DEG = 30.0

#: Cartesian jog step per press, metres and degrees, by the panel's three speed
#: names.  These are DISTANCES, not velocities: each ``{"cmd": "jog"}`` is one
#: bounded ``move_delta`` that finishes.
#:
#: THE PANEL NO LONGER SENDS THIS; ``jog_start`` does.  It is kept because it is
#: the only command here that moves a stated DISTANCE — a twist cannot, since it
#: is a velocity with no end — and a hand-typed session at the bridge's prompt
#: has no heartbeat to offer.  "Move it five millimetres and tell me when you
#: got there" is a useful sentence and this is the only one that says it.
JOG_STEP_M = {"slow": 0.005, "normal": 0.020, "fast": 0.050}
JOG_STEP_DEG = {"slow": 1.0, "normal": 5.0, "fast": 10.0}

#: Cartesian speeds a jog is executed at, m/s and deg/s.
JOG_SPEED_MS = {"slow": 0.02, "normal": 0.05, "fast": 0.10}
JOG_SPEED_DEG_S = {"slow": 5.0, "normal": 15.0, "fast": 30.0}

# --- the held jog: a twist, a ramp and a dead-man --------------------------
#
# WHY A TWIST AND NOT A FASTER SEQUENCE OF STEPS.  ``_cmd_jog`` runs one bounded
# ``move_delta`` on the single motion worker and ``_require_ready`` refuses a
# second while one is in flight, so a held button is mostly refusals.  MEASURED
# against a fake arm at the panel's 100 ms beat: a ``move_delta`` costing 0.25 s
# turned 20 beats into 7 finished moves and 13 refusals (65 %), 0.40 s into 5 and
# 15 (75 %), and 1.00 s into 2 and 18 (90 %) — every refusal reading "jog is
# still running".  At the real 0.25 m envelope with ``speed="normal"`` the same
# 20 beats gave 12 executed moves and 8 refusals of "0.2600 m leaves the
# envelope", i.e. a 1.3 s hold walked the tool out of the whole keep-out ball.
# Each refusal writes a line into a four-line log box and ``busy`` cycles between
# every beat, so the buttons grey out mid-hold.  A velocity has none of those
# properties: one command starts it, one stops it, and the ramp lives here.

#: How often the jog thread re-issues the twist, Hz.  NOT :data:`PUBLISH_HZ`,
#: which is 10 and would resolve a quarter-second ramp into two and a half steps.
#: 40 Hz is the rate the vendored ``smooth_move`` ramps at and the rate the
#: simulated task-space jog runs at, so the two tabs ramp the same way; it also
#: sets how often the envelope is re-polled, which at the fastest preset is one
#: check per 1.5 mm of travel.
JOG_TWIST_HZ = 40.0

#: How long after the last heartbeat the jog gives up, seconds.  THE SAME 0.35 s
#: THE SIMULATED JOG USES, deliberately: it is three missed beats at the panel's
#: ``JOG_BEAT_MS = 100``, and one number that means the same thing on both tabs
#: is worth more than a number tuned separately for each.  A release message is
#: best-effort — it can be dropped, and a button released outside its own
#: rectangle never sends one at all — so this, not the release, is what
#: guarantees the arm stops.
JOG_DEADMAN_S = 0.35

#: The ramp, at the ``normal`` preset: 30 mm/s reached in a quarter of a second,
#: 10 deg/s in the same quarter second.  Half the simulated jog's translation
#: (0.05 m/s) and half its rotation (20 deg/s), because this one is metal and
#: because the simulated pad can be driven through a printed part while this one
#: strikes it.  The acceleration is scaled with the velocity by
#: :data:`JOG_SPEEDS`, so the ramp stays 0.25 s at every preset and only the top
#: speed changes.
JOG_V_MAX = 0.030
JOG_A_MAX = 0.120
JOG_W_MAX = 10.0
JOG_ALPHA_MAX = 40.0

#: Preset multipliers, the same three names and the same three numbers as the
#: simulated jog.
#:
#: WHAT A LOST RELEASE COSTS, which is the number these were chosen against.  The
#: dead-man runs at full speed for :data:`JOG_DEADMAN_S` and then the ramp brings
#: it down over 0.25 s, so the coast is ``v * 0.35 + v^2 / (2 a)``.  MEASURED by
#: integrating the shipped :class:`JogProfile` at :data:`JOG_TWIST_HZ` with the
#: heartbeats simply stopping: **27.8 mm at fast, 13.9 mm at normal, 4.9 mm at
#: slow**, taking 0.625 s to come to rest, and 9.3 / 4.6 / 1.6 deg of rotation.
#: The simulated jog's own coasts are 46 / 22 / 8 mm, so every preset here is
#: under its twin, and the rehearsed pad standoffs are millimetres — which is why
#: ``slow`` exists and why it is the panel's default on this tab.
#:
#: WHAT THAT NUMBER IS NOT.  It is the distance the COMMAND travels, integrated
#: from the velocity this file sends.  What the arm's own controller does with a
#: decaying twist has not been measured, because no Gen3 was attached when this
#: was written; the first session on the metal should compare the two.
JOG_SPEEDS = {"slow": 0.35, "normal": 1.0, "fast": 2.0}

#: How close a joint move must land on what it asked for, degrees.  The Gen3's
#: own repeatability is quoted at 0.1 mm, so 1 deg catches an aborted or clamped
#: action rather than ordinary servo error -- the same number and the same
#: reasoning as ``kinova_arm.HOME_TOL_DEG``.
JOINT_ARRIVAL_TOL_DEG = 1.0

#: Radius of the keep-out ball armed on ``arm``, metres.  The bench's jog is a
#: poking-around motion around one pose, not a traverse; 0.25 m is enough to
#: cross the gap to the UMArm and small enough that a runaway is bounded.
ENVELOPE_R_M = 0.25


#: The JSON channel, captured at import.  ``_emit`` writes HERE and not to
#: ``sys.stdout``, because :mod:`UMArm_KINOVA.bridge_mocap` swaps ``sys.stdout``
#: out for as long as a mocap receiver is up — the vendored NatNet SDK prints to
#: it, from its own threads, at times of its choosing, and those lines would
#: otherwise land in the middle of the protocol.  Holding the file object from
#: before any of that means no redirect anybody installs can move the channel.
_JSON_OUT = sys.stdout


def _emit(obj: dict, out=None) -> None:
    """One JSON object, one line, flushed.

    Flushed on every message on purpose: the parent reads line by line and a
    buffered ``state`` is a panel that thinks the arm has stopped answering.
    """
    fh = _JSON_OUT if out is None else out
    fh.write(json.dumps(obj, default=float) + "\n")
    fh.flush()


def _ramp(current: float, target: float, rate: float, dt: float) -> float:
    """Move *current* toward *target* by at most ``rate * dt``.

    The whole of the jog's trapezoid, and the same four lines the simulated jog
    uses (``UMArm_COLLAB.gui.backend._ramp``) rather than an import of them: this
    file runs on ``.venv_kinova``, where MuJoCo and the bench are not installed
    and must not be.  The one property that matters is that the same limit
    accelerates and decelerates, so a release costs exactly as long as the press
    did.
    """
    step = abs(rate) * dt
    if target > current:
        return min(target, current + step)
    return max(target, current - step)


def _unit(v):
    """A direction, or ``None`` when there is none.  Length is not a speed here.

    The panel sends ``[0, 0, 1]`` and means "+z"; the magnitude of what it sends
    must not become a second, undocumented speed control on top of the preset.
    """
    if v is None:
        return None
    a = [float(x) for x in v]
    n = sum(x * x for x in a) ** 0.5
    return None if n < 1e-9 else [x / n for x in a]


class JogProfile:
    """The held button: a direction, a preset, a heartbeat and a ramp.

    Split out of the thread that drives it so the ramp is testable without a
    clock, a socket or an arm — which is also how the coast numbers quoted on
    :data:`JOG_SPEEDS` were obtained.  It holds no arm and sends nothing; it
    answers one question, "what velocity now", and :meth:`Bridge._jog_loop` is
    what turns that into a twist.
    """

    def __init__(self, deadman_s: float = JOG_DEADMAN_S):
        self.deadman_s = float(deadman_s)
        self.dir_xyz = None
        self.dir_rpy = None
        self.scale = 1.0
        self.frame = "robot"
        self.v = 0.0
        self.w = 0.0
        self.held = False
        self.beat_t = 0.0
        #: True once the ramp has been flat with the button released — the jog
        #: is over and the thread may go.
        self.done = False

    def beat(self, d_xyz=None, d_rpy_deg=None, speed="normal", frame="robot",
             now=None) -> None:
        """One heartbeat.  Cheap, and it is the ONLY thing that keeps it alive."""
        self.dir_xyz = _unit(d_xyz)
        self.dir_rpy = _unit(d_rpy_deg)
        self.scale = float(JOG_SPEEDS.get(str(speed), 1.0))
        self.frame = str(frame)
        self.held = True
        self.done = False
        self.beat_t = time.monotonic() if now is None else float(now)

    def release(self) -> None:
        """Let go.  The ramp bleeds down; it does NOT snap to zero.

        A twist commanded to zero from full speed is the arm stopping as hard as
        its controller will let it, with whatever is bolted to the wrist carrying
        on.  Ramping down at the same rate it ramped up means a release costs
        exactly as long as the press did, which is the property that makes a
        held button predictable.
        """
        self.held = False

    def step(self, dt: float, now=None) -> tuple:
        """Advance one slice.  Returns ``(linear, angular)`` in m/s and deg/s.

        The dead-man is applied HERE rather than by whoever calls this, so a
        driver that stops calling for a moment — a slow slice, a scheduler
        hiccup — cannot accidentally extend the hold: the expiry is measured
        against the last heartbeat's wall clock, not against a count of slices.
        """
        now = time.monotonic() if now is None else float(now)
        live = bool(self.held) and (now - self.beat_t) <= self.deadman_s
        s = self.scale
        self.v = _ramp(self.v, JOG_V_MAX * s if (live and self.dir_xyz) else 0.0,
                       JOG_A_MAX * s, dt)
        self.w = _ramp(self.w, JOG_W_MAX * s if (live and self.dir_rpy) else 0.0,
                       JOG_ALPHA_MAX * s, dt)
        if self.v <= 0.0 and self.w <= 0.0:
            self.done = True
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lin = [d * self.v for d in (self.dir_xyz or (0.0, 0.0, 0.0))]
        ang = [d * self.w for d in (self.dir_rpy or (0.0, 0.0, 0.0))]
        return lin, ang


class Bridge:
    """The arm, a motion worker, and the command table.  No I/O of its own."""

    def __init__(self, emit=_emit, envelope_r_m: float = ENVELOPE_R_M,
                 arm_factory=None, mocap=None):
        self.emit = emit
        self.envelope_r_m = float(envelope_r_m)
        #: How a session is opened.  Defaults to the real
        #: :class:`UMArm_KINOVA.kinova_arm.SafeKinovaArm`, which can only be
        #: imported on ``.venv_kinova``; a test injects a stand-in here so the
        #: PROTOCOL -- the arm/disarm gate, the joint-step limit, the
        #: one-motion-at-a-time worker -- is checkable on the ordinary
        #: interpreter, which is the interpreter everybody actually has.
        self.arm_factory = arm_factory
        self.arm = None
        self.armed = False
        self.error = ""
        #: What the worker is doing, or "" when it is idle.  Published, because
        #: a panel that cannot see a move in flight will queue another one.
        self.busy = ""
        self._worker: threading.Thread | None = None
        self._last_state: dict = {}
        self._lock = threading.Lock()
        #: Bumped by every ``stop``.  THE WORKER READS IT TWICE, and it has to.
        #: ``base.Stop()`` aborts whatever the arm is executing RIGHT NOW, and
        #: the vendored driver spends three RPC round trips -- servoing mode,
        #: ``ReadAllActions``, ``OnNotificationActionTopic`` -- getting to its
        #: ``ExecuteAction``.  A STOP inside that window used to abort nothing,
        #: flip ``armed`` to False, and let the move start anyway with the panel
        #: showing "safe (not armed)".  The worker now refuses to start after a
        #: stop it can see, and re-stops the arm if one landed while it was
        #: inside the driver.
        self._stop_epoch = 0

        #: The held jog, or ``None``.  Guarded by :attr:`_jog_lock` because two
        #: threads reach for it — the reader thread beats it, the jog thread
        #: retires it — and "is there a jog running" has to be one answer.
        self._jog: JogProfile | None = None
        self._jog_lock = threading.Lock()
        self._jog_thread: threading.Thread | None = None
        self._jog_note = ""
        #: What the last twist actually commanded, for the panel to paint.
        self._jog_last: dict = {}

        #: The cameras.  Constructed here rather than on ``connect`` so the
        #: command table can answer ``{"cmd": "mocap"}`` before an arm exists —
        #: an operator with Motive up and the Gen3 still powered down has every
        #: reason to check the volume, and nothing about the receiver needs the
        #: arm.  Constructing one costs no thread and touches no socket.
        if mocap is None:
            from UMArm_KINOVA.bridge_mocap import MocapHalf

            mocap = MocapHalf()
        self.mocap = mocap

    # -- lifecycle ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.arm is not None

    def close(self) -> None:
        """Drop the session, and everything that was hanging off it.

        THE MOCAP GOES TOO.  It is not part of the Kortex session and could in
        principle outlive it, but a receiver left running against a closed
        session is four SDK threads nobody is watching — and non-daemon ones,
        which hold the process open past ``bye``.  Bringing it back is one
        ``{"cmd": "mocap", "on": true}``, which is cheaper than a bridge that
        will not exit.
        """
        self.armed = False
        self._stop_epoch += 1               # the jog thread reads this and dies
        self._join_jog()
        try:
            self.mocap.stop()
        except Exception:
            pass
        arm, self.arm = self.arm, None
        if arm is not None:
            try:
                arm.stop()
            except Exception:
                pass
            try:
                arm.close()
            except Exception:
                pass

    # -- feedback ----------------------------------------------------------

    def state(self) -> dict:
        """The published snapshot.  Never raises: a dead session is a state.

        THE MOCAP BLOCK IS PUBLISHED WHETHER OR NOT AN ARM IS CONNECTED, and
        before the early return below.  An operator checking whether Motive is
        streaming has no reason to have powered the Gen3 on first, and a readout
        that appears only once the arm answers is a readout that cannot be used
        to diagnose the case it exists for.
        """
        out = {"kind": "state", "connected": self.connected,
               "armed": bool(self.armed), "busy": self.busy,
               "error": self.error, "t": time.time(),
               "jogging": self._jog is not None, "jog": dict(self._jog_last)}
        try:
            out["mocap"] = self.mocap.state()
        except Exception as exc:            # a readout is never worth a crash
            out["mocap"] = {"on": False, "live": False,
                            "error": f"{type(exc).__name__}: {exc}"}
        if self.arm is None:
            return out
        try:
            snap = self.arm.snapshot()
        except Exception as exc:
            # The session died under us.  Say so and drop the handle: a panel
            # showing a red ARMED border against a dead arm is worse than one
            # showing a disconnect.
            self.error = f"lost the arm: {type(exc).__name__}: {exc}"
            self.close()
            out.update(connected=False, armed=False, error=self.error)
            return out
        out["q_deg"] = [float(v) for v in snap["joints_deg"]]
        out["pose"] = [float(v) for v in snap["pose"]]
        out["envelope_centre"] = (None if self.arm.envelope_centre is None
                                  else [float(v) for v
                                        in self.arm.envelope_centre])
        return out

    # -- commands ----------------------------------------------------------

    def handle(self, msg: dict) -> None:
        cmd = str(msg.get("cmd", ""))
        fn = getattr(self, f"_cmd_{cmd}", None)
        if fn is None:
            self.emit({"kind": "error", "cmd": cmd,
                       "error": f"unknown command {cmd!r}"})
            return
        try:
            fn(msg)
        except Exception as exc:
            self.emit({"kind": "error", "cmd": cmd,
                       "error": f"{type(exc).__name__}: {exc}"})

    def _cmd_connect(self, msg) -> None:
        if self.connected:
            self.emit({"kind": "ok", "cmd": "connect",
                       "note": "already connected"})
            return
        if self.arm_factory is not None:
            arm = self.arm_factory(dict(msg), self.envelope_r_m)
        else:
            from UMArm_KINOVA import kinova_arm as KA

            arm = KA.SafeKinovaArm(
                ip=str(msg.get("ip", KA.DEFAULT_IP)),
                username=str(msg.get("username", KA.DEFAULT_USER)),
                password=str(msg.get("password", KA.DEFAULT_PASSWORD)),
                envelope_r_m=self.envelope_r_m)
        arm.connect()
        self.arm = arm
        self.armed = False
        self.error = ""
        snap = arm.snapshot()
        self.emit({"kind": "ok", "cmd": "connect", "ip": arm.ip,
                   "q_deg": [float(v) for v in snap["joints_deg"]],
                   "pose": [float(v) for v in snap["pose"]]})
        self.emit({"kind": "log",
                   "line": f"connected to {arm.ip}; NOT armed — nothing will "
                           f"move until you arm it"})
        # ...and the cameras, because the operator asked for them to start with
        # the arm.  AFTER the connect has been reported, and on a thread of its
        # own: a connect that waited on a NatNet socket would be a connect that
        # hung on the one part of this the operator called optional.
        try:
            self.mocap.start()
        except Exception as exc:            # a missing UMArm_MOCAP is a state
            self.emit({"kind": "log",
                       "line": f"mocap did not start: {type(exc).__name__}: "
                               f"{exc} — the arm's own base frame is unaffected"})

    def _cmd_disconnect(self, _msg) -> None:
        self.close()
        self.emit({"kind": "ok", "cmd": "disconnect"})

    def _cmd_arm(self, msg) -> None:
        on = bool(msg.get("on", True))
        if not self.connected:
            raise RuntimeError("not connected")
        if not on:
            self.armed = False
            self.emit({"kind": "ok", "cmd": "arm", "armed": False})
            return
        if self.busy:
            raise RuntimeError(f"{self.busy} is still running")
        centre = self.arm.arm_envelope()
        self.armed = True
        self.emit({"kind": "ok", "cmd": "arm", "armed": True,
                   "envelope_centre": [float(v) for v in centre],
                   "envelope_r_m": self.arm.envelope_r_m})
        self.emit({"kind": "log",
                   "line": "ARMED — the keep-out ball is centred on this pose, "
                           f"radius {self.arm.envelope_r_m * 1e3:.0f} mm"})

    def _cmd_stop(self, _msg) -> None:
        """Always accepted, even mid-move.  That is the point of it.

        DISARMING COMES FIRST AND CANNOT FAIL.  It used to come after
        ``arm.stop()``, and :meth:`handle` wraps a command in ``try/except`` --
        so a Stop RPC that raised (a faulted arm, a dropped session, an
        inactivity timeout: exactly when STOP is pressed) aborted the handler
        before the disarm, and the bridge went on accepting motion with the
        panel still showing a red border. ``armed`` is a local flag; it must
        never be conditional on the arm answering.

        THE JOG IS JOINED, NOT MERELY SIGNALLED.  A twist runs until something
        stops it, so a Stop that returned while the jog thread still had a slice
        to issue would be followed by another twist 25 ms later and the arm would
        carry on moving with the panel painted "safe".  The thread notices the
        stop epoch within one slice, issues its own ``arm.stop()`` on the way
        out, and only then does the Stop below run — so the LAST thing the arm
        hears is always a Stop.  MEASURED against a fake arm: this costs the stop
        path under 40 ms, against the 200 ms
        ``test_stop_lands_while_a_move_is_running_and_disarms`` allows.
        """
        self.armed = False
        self._stop_epoch += 1
        self._join_jog()
        err = ""
        if self.arm is not None:
            try:
                self.arm.stop()
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        self.emit({"kind": "ok", "cmd": "stop", "armed": False})
        self.emit({"kind": "log",
                   "line": "STOPPED and disarmed"
                           + (f" (the arm did not answer the Stop: {err})"
                              if err else "")})

    def _cmd_ping(self, _msg) -> None:
        self.emit({"kind": "ok", "cmd": "ping"})

    def _cmd_joints(self, msg) -> None:
        want = [float(v) for v in (msg.get("values_deg") or [])]
        self._require_ready("joints")
        got = self.arm.get_joint_angles()
        if len(want) != len(got):
            raise ValueError(f"this arm has {len(got)} actuators and the "
                             f"command carries {len(want)} angles")
        # Wrapped, because the odd joints are continuous: 359 and 1 are 2 deg
        # apart, not 358.
        delta = [(w - g + 180.0) % 360.0 - 180.0 for w, g in zip(want, got)]
        step = [abs(d) for d in delta]
        if max(step) > MAX_JOINT_STEP_DEG:
            j = int(max(range(len(step)), key=step.__getitem__))
            raise ValueError(
                f"joint {j + 1} would move {step[j]:.1f} deg in one command, "
                f"limit {MAX_JOINT_STEP_DEG:.0f} deg — REFUSED, nothing was "
                f"sent. Move it in stages, or use Home.")
        # AND THE ARM IS SENT THE MOTION THAT WAS CHECKED, not the number that
        # was typed.  The check wraps and the driver does not: it copies the
        # float straight into the action.  So ``values_deg = [360, ...]`` against
        # an arm at 0 measured a 0.0 deg step, passed the limit, and asked for a
        # full turn of the wrist.  Rebuilding the target from the wrapped delta
        # makes the command and its guard describe the same motion by
        # construction.
        want = [g + d for g, d in zip(got, delta)]
        self._start("joints", lambda: self._move_joints(want))

    def _move_joints(self, want) -> None:
        """``move_joints`` with the completion check the wrapper would have done.

        The vendored ``move_joints`` returns ``_execute_action_and_wait``'s
        flag: **False** when the wait timed out with the action STILL RUNNING,
        and **True** on ``ACTION_ABORT`` exactly as on ``ACTION_END``.  Both
        other motion paths check it -- ``SafeKinovaArm.move_to_pose`` and
        ``move_home`` stop the arm and raise ``ArrivalError`` -- and this is the
        one path that deliberately bypasses the wrapper, so bypassing the check
        with it left a stalled 20 s action reported as ``done``: ``busy`` went
        clear, the panel re-enabled every motion button, and the next click put
        a second action on an arm that was still executing the first.

        Arrival is verified against the JOINTS, wrapped, for the reason
        ``move_home``'s docstring gives: a completion flag cannot tell END from
        ABORT.
        """
        finished = self.arm.arm.move_joints(want)
        if finished is False:
            self.arm.stop()
            raise RuntimeError(
                "the joint move did not finish; the arm has been STOPPED")
        got = self.arm.get_joint_angles()
        err = [abs((g - w + 180.0) % 360.0 - 180.0) for g, w in zip(got, want)]
        if err and max(err) > JOINT_ARRIVAL_TOL_DEG:
            j = int(max(range(len(err)), key=err.__getitem__))
            raise RuntimeError(
                f"the joint move reported completion but joint {j + 1} is "
                f"{err[j]:.2f} deg from its target (tolerance "
                f"{JOINT_ARRIVAL_TOL_DEG:.2f} deg) \u2014 it was ABORTED, not "
                f"reached")

    def _cmd_home(self, _msg) -> None:
        self._require_ready("home")
        self._start("home", self.arm.move_home)

    def _cmd_jog(self, msg) -> None:
        self._require_ready("jog")
        speed = str(msg.get("speed", "normal"))
        if speed not in JOG_STEP_M:
            raise ValueError(f"unknown speed {speed!r}")
        d_xyz = [float(v) for v in (msg.get("d_xyz") or (0.0, 0.0, 0.0))]
        d_rpy = [float(v) for v in (msg.get("d_rpy_deg") or (0.0, 0.0, 0.0))]
        sm, sd = JOG_STEP_M[speed], JOG_STEP_DEG[speed]
        dx, dy, dz = (v * sm for v in d_xyz)
        rx, ry, rz = (v * sd for v in d_rpy)
        if max(abs(v) for v in (dx, dy, dz, rx, ry, rz)) <= 0.0:
            raise ValueError("a jog with no direction")
        self._start(
            "jog",
            lambda: self.arm.move_delta(
                dx=dx, dy=dy, dz=dz, dtheta_x=rx, dtheta_y=ry, dtheta_z=rz,
                speed_ms=JOG_SPEED_MS[speed],
                speed_deg_s=JOG_SPEED_DEG_S[speed]))

    # -- the held jog ------------------------------------------------------

    def _cmd_jog_start(self, msg) -> None:
        """A heartbeat from a held button.  The FIRST one starts the thread.

        Every later one is a beat and is answered with silence: the panel repeats
        at 10 Hz and an ``ok`` per beat would be ten protocol lines a second for
        as long as a finger is down, on the same channel ``stop`` has to arrive
        on.  What the panel watches instead is ``state``'s ``jogging`` flag.
        """
        speed = str(msg.get("speed", "normal"))
        if speed not in JOG_SPEEDS:
            raise ValueError(f"unknown speed {speed!r}")
        frame = str(msg.get("frame", "robot"))
        if frame not in ("robot", "world"):
            raise ValueError(f"unknown frame {frame!r}")
        d_xyz = msg.get("d_xyz")
        d_rpy = msg.get("d_rpy_deg")
        if _unit(d_xyz) is None and _unit(d_rpy) is None:
            raise ValueError("a jog with no direction")
        if frame == "world" and not self.mocap.world_ready():
            raise RuntimeError(
                "world-frame jog refused: the world frame needs mocap frames "
                "arriving, a Y from a calibration, and an X solved this session "
                "with mocap_solve_base — the robot base frame needs none of "
                "them and is always available")

        with self._jog_lock:
            jog = self._jog
            if jog is not None:
                jog.beat(d_xyz, d_rpy, speed, frame)
                return
            self._require_ready("jog")
            jog = JogProfile()
            jog.beat(d_xyz, d_rpy, speed, frame)
            self._jog = jog
            self._jog_note = ""
            self.busy = "jog"
            epoch = self._stop_epoch
            self._jog_thread = threading.Thread(
                target=self._jog_loop, args=(epoch,), name="kinova-jog",
                daemon=True)
            self._jog_thread.start()
        self.emit({"kind": "ok", "cmd": "jog_start", "started": True,
                   "speed": speed, "frame": frame})

    def _cmd_jog_stop(self, _msg) -> None:
        """Let go.  NOT an error when nothing is held.

        The panel releases on ``<ButtonRelease-1>`` *and* on ``<Leave>``, because
        a button released outside its own rectangle never delivers the first, so
        a stray release is the ordinary case rather than a fault.
        """
        with self._jog_lock:
            jog = self._jog
        if jog is not None:
            jog.release()
        self.emit({"kind": "ok", "cmd": "jog_stop"})

    def _jog_loop(self, epoch: int) -> None:
        """Issue the twist at :data:`JOG_TWIST_HZ` until something ends it.

        FOUR WAYS OUT, and every one of them leaves through the same ``finally``
        and the same ``arm.stop()``: the ramp reaching zero after a release or a
        dead-man expiry, a stop epoch, a disarm, and a twist the arm refused.
        The vendored driver has no watchdog and ``cmd.duration = 0`` — the
        docstring at ``vendor/kinova_driver.py:297`` and ``vendor/AGENT_GUIDE.md``
        both state the arm keeps moving until another twist or a Stop, and that
        is what the code shows rather than something measured here on metal — so
        the Stop on the way out is not tidying up, it is the end of the motion.
        """
        period = 1.0 / JOG_TWIST_HZ
        last = time.monotonic()
        why = "released"
        try:
            while True:
                jog = self._jog
                if jog is None:
                    why = "cancelled"
                    break
                if epoch != self._stop_epoch:
                    why = "STOPPED"
                    break
                if not self.armed:
                    why = "disarmed"
                    break
                now = time.monotonic()
                dt = min(max(now - last, 1e-4), 0.2)
                last = now
                lin, ang = jog.step(dt, now)
                if jog.done:
                    break
                if jog.frame == "world":
                    # THE GATE IS CHECKED PER HEARTBEAT, NOT PER SLICE.  A hold
                    # that started world-ready keeps its ``X`` for the rest of
                    # the hold even if the stream drops mid-way, because stopping
                    # the arm dead because one mocap frame went missing is worse
                    # than finishing a half-second of motion on the transform it
                    # started with.  ``_cmd_jog_start`` re-checks
                    # ``world_ready`` on every beat, so a gate that closes still
                    # ends the hold within a beat and a dead-man.
                    lin_b = self.mocap.to_base(lin)
                    ang_b = self.mocap.to_base(ang)
                    if lin_b is None or ang_b is None:
                        why = "the world frame went away mid-hold"
                        self.emit({"kind": "error", "cmd": "jog_start",
                                   "error": "the session's X was lost while a "
                                            "world-frame jog was held"})
                        break
                    lin, ang = lin_b, ang_b
                try:
                    # The twist is ALWAYS commanded in the arm's base frame; a
                    # world-frame hold is rotated into it above.  Handing the
                    # driver a different reference frame would move the envelope
                    # poll and the commanded direction into two different
                    # spaces, and the guard would be checking the wrong axis.
                    out = self.arm.twist_checked(lin, ang, frame="base")
                except Exception as exc:
                    self.armed = False
                    self.emit({"kind": "error", "cmd": "jog_start",
                               "error": f"{type(exc).__name__}: {exc}"})
                    why = "refused"
                    break
                self._jog_last = {"speed": jog.scale, "frame": jog.frame,
                                  "v_mps": jog.v, "w_dps": jog.w,
                                  "at_wall": bool(out.get("at_wall")),
                                  "r_mm": float(out.get("distance_m", 0.0)) * 1e3}
                # A WALL HOLDS, IT DOES NOT STOP.  ``twist_checked`` has already
                # dropped the outward component, so the tangential and inward
                # parts of the same button still move the arm and the operator
                # can drive back off the wall without disarming.  Said once per
                # arrival: at 40 Hz the alternative is forty identical lines a
                # second into a four-line log box.
                if out.get("at_wall") and self._jog_note != "wall":
                    self._jog_note = "wall"
                    self.emit({"kind": "log",
                               "line": "jog at the envelope wall (%.0f mm from "
                                       "centre): the outward part of this "
                                       "direction is being dropped, the rest "
                                       "still moves"
                                       % (out["distance_m"] * 1e3)})
                elif not out.get("at_wall"):
                    self._jog_note = ""
                rest = period - (time.monotonic() - now)
                if rest > 0:
                    time.sleep(rest)
        finally:
            try:
                if self.arm is not None:
                    self.arm.stop()
            except Exception:
                pass
            with self._jog_lock:
                self._jog = None
                self._jog_thread = None
                if self.busy == "jog":
                    self.busy = ""
            self._jog_last = {}
            self.emit({"kind": "ok", "cmd": "jog_start", "done": True,
                       "why": why})

    def _join_jog(self, timeout: float = 0.3) -> None:
        """Wait for the jog thread to notice and leave.  Safe to call anywhere."""
        with self._jog_lock:
            t = self._jog_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=timeout)

    # -- the cameras -------------------------------------------------------

    def _cmd_mocap(self, msg) -> None:
        """Turn the receiver on or off without touching the arm session.

        The operator asked for the mocap to start with the arm, and it does; this
        exists for the other half of that sentence — starting Motive AFTER
        connecting, which without this command would mean disconnecting the arm
        to retry something the arm has nothing to do with.
        """
        on = bool(msg.get("on", True))
        if on:
            self.mocap.start()
        else:
            self.mocap.stop()
        self.emit({"kind": "ok", "cmd": "mocap", "on": on})

    def _cmd_mocap_solve_base(self, _msg) -> None:
        """Re-derive ``X = T_world_base`` from one still frame plus the arm's FK.

        This is what unlocks the world frame, and it is deliberately a COMMAND
        rather than something the bridge does for itself on connect: it needs the
        arm to be standing still, and only the operator knows that.

        A solve with nothing to solve from is an ``error`` reply, not a
        traceback: no frames and no calibration are both ordinary states of a
        bench where the cameras have not been switched on.
        """
        if not self.connected:
            raise RuntimeError("not connected")
        if self.busy:
            raise RuntimeError(f"{self.busy} is still running — the arm has to "
                               f"be standing still to solve its base")
        snap = self.arm.snapshot()
        out = self.mocap.solve_base(snap["pose"])
        if not out.get("ok"):
            self.emit({"kind": "error", "cmd": "mocap_solve_base",
                       "error": str(out.get("error", "no solve"))})
            return
        self.emit({"kind": "ok", "cmd": "mocap_solve_base", **out})
        self.emit({"kind": "log",
                   "line": "base solved from one mocap frame: %.1f mm and "
                           "%.3f deg from the stored X (%s). The campaign's own "
                           "single-frame spread is 2.07 mm RMS / 6.97 mm max, so "
                           "centimetres here means the cart has been moved"
                           % (out["vs_stored_mm"], out["vs_stored_deg"],
                              out["calibration"])})

    def _cmd_quit(self, _msg) -> None:
        raise SystemExit(0)

    # -- the motion worker -------------------------------------------------

    def _require_ready(self, what: str) -> None:
        if not self.connected:
            raise RuntimeError("not connected")
        if not self.armed:
            raise RuntimeError(f"{what} refused: the arm is not armed")
        if self.busy:
            raise RuntimeError(f"{self.busy} is still running")

    def _start(self, what: str, fn) -> None:
        """Run *fn* on the single motion worker and report how it ended."""
        self.busy = what
        self.emit({"kind": "ok", "cmd": what, "started": True})

        epoch = self._stop_epoch

        def work():
            try:
                if epoch != self._stop_epoch or not self.armed:
                    # A stop landed between the command being accepted and this
                    # thread getting the CPU.  Nothing has been sent yet, so
                    # there is nothing to abort -- just do not send it.
                    self.emit({"kind": "error", "cmd": what,
                               "error": "stopped before it started"})
                    return
                fn()
                if epoch != self._stop_epoch:
                    # ...and a stop that landed WHILE the driver was setting the
                    # action up aborted nothing, because there was nothing
                    # running yet.  Stop again now that there is.
                    self.armed = False
                    try:
                        self.arm.stop()
                    except Exception:
                        pass
                    self.emit({"kind": "error", "cmd": what,
                               "error": "stopped mid-command; the arm has been "
                                        "STOPPED again"})
                    return
                self.emit({"kind": "ok", "cmd": what, "done": True})
            except Exception as exc:
                # A refused or failed move DISARMS.  Whatever the arm did, it is
                # no longer where the envelope was centred and the operator has
                # to look before it moves again.
                self.armed = False
                self.emit({"kind": "error", "cmd": what,
                           "error": f"{type(exc).__name__}: {exc}"})
            finally:
                self.busy = ""

        self._worker = threading.Thread(target=work, name=f"kinova-{what}",
                                        daemon=True)
        self._worker.start()


def serve(stdin=None, emit=_emit, publish_hz: float = PUBLISH_HZ,
          envelope_r_m: float = ENVELOPE_R_M, arm_factory=None,
          mocap=None) -> int:
    """Read commands until EOF or ``quit``, publishing state throughout.

    The reader is a thread and the publisher is this one, rather than the other
    way round, so a command that arrives while a state is being written cannot
    be lost to a blocking read.

    IT ALSO DECIDES WHICH THREAD OWNS THE MOCAP.  Every command runs on the
    reader thread, which is a daemon, so that is the thread that asks for a
    receiver and the thread that stops one — never this one, which is the
    process's main thread and the one whose return would otherwise sit waiting on
    the SDK's non-daemon threads.  MEASURED on this machine: a receiver started
    and never stopped kept the interpreter alive past a 15 s timeout
    (``EXIT=124``).  The ``finally`` below is what makes that impossible.
    """
    fh = sys.stdin if stdin is None else stdin
    bridge = Bridge(emit=emit, envelope_r_m=envelope_r_m,
                    arm_factory=arm_factory, mocap=mocap)
    stop = threading.Event()

    def read():
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception as exc:
                    emit({"kind": "error", "cmd": "", "error": f"bad JSON: {exc}"})
                    continue
                try:
                    bridge.handle(msg)
                except SystemExit:
                    break
        finally:
            stop.set()

    t = threading.Thread(target=read, name="kinova-bridge-reader", daemon=True)
    t.start()

    period = 1.0 / max(float(publish_hz), 0.1)
    try:
        # AT LEAST ONE STATE, ALWAYS.  A ``while not stop.is_set()`` can lose
        # the race against a reader that reaches EOF first -- a scripted session
        # piped in from a file does exactly that -- and a client that never sees
        # a ``state`` cannot tell "the arm answered" from "the bridge died".
        while True:
            emit(bridge.state())
            if stop.wait(period):
                break
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        # Belt as well as braces.  ``close`` already stops the receiver, but
        # ``close`` is also reachable from ``state`` on a dead session and could
        # in principle be left half-done; the one thing this process must not do
        # is exit ``serve`` with an SDK thread still running, because it would
        # then not exit at all.
        try:
            bridge.mocap.stop()
        except Exception:
            pass
        emit({"kind": "bye"})
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--publish-hz", type=float, default=PUBLISH_HZ)
    ap.add_argument("--envelope-mm", type=float, default=ENVELOPE_R_M * 1e3)
    args = ap.parse_args(argv)
    return serve(publish_hz=args.publish_hz,
                 envelope_r_m=args.envelope_mm * 1e-3)


__all__ = ["Bridge", "JogProfile", "serve", "main", "PUBLISH_HZ",
           "MAX_JOINT_STEP_DEG", "JOG_STEP_M", "JOG_STEP_DEG", "JOG_SPEED_MS",
           "JOG_SPEED_DEG_S", "ENVELOPE_R_M", "JOG_TWIST_HZ", "JOG_DEADMAN_S",
           "JOG_V_MAX", "JOG_A_MAX", "JOG_W_MAX", "JOG_ALPHA_MAX",
           "JOG_SPEEDS"]


if __name__ == "__main__":          # pragma: no cover
    raise SystemExit(main())
