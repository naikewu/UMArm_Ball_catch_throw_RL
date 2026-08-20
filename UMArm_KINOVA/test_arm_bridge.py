"""The real-arm bridge and its host-side client, WITHOUT a real arm.

Runs under the ORDINARY interpreter, which is the point: everything in
:mod:`UMArm_KINOVA.arm_bridge` that could hurt somebody -- the arm/disarm gate,
the per-command joint-step limit, the one-motion-at-a-time worker, the fact that
``stop`` lands while a move is running -- is protocol, and protocol is testable
without ``kortex_api``.  The driver itself is exercised by
``test_kinova_offline.py``, which needs ``.venv_kinova`` and skips here.

``UMArm_KINOVA.bridge_client`` is tested against a STUB bridge script rather
than the real one, so the process plumbing (spawn, JSON lines, the reader
thread, the shutdown path) is checked on a machine that may not have
``.venv_kinova`` at all.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from UMArm_KINOVA import arm_bridge as AB          # noqa: E402
from UMArm_KINOVA import bridge_client as BC       # noqa: E402
from UMArm_KINOVA import bridge_mocap as BM        # noqa: E402


class FakeArm:
    """A Gen3 that answers instantly and records what it was told.

    Deliberately NOT a mock: it enforces the one thing the bridge delegates --
    that a Cartesian move outside the armed ball raises rather than clips --
    because a stand-in that always succeeds would let a broken envelope pass.
    Its ``twist_checked`` is a second such enforcement rather than a pass-through
    for the same reason; what it is NOT is evidence that the real one is right,
    which is ``test_kinova_offline.py``'s job, since ``SafeKinovaArm`` cannot be
    imported here at all.
    """

    def __init__(self, envelope_r_m=0.25):
        self.ip = "fake"
        self.envelope_r_m = float(envelope_r_m)
        self.envelope_centre = None
        self.pose = [0.4, 0.0, 0.3, 90.0, 0.0, 90.0]
        self.joints = [0.0, 15.0, 180.0, 230.0, 0.0, 55.0, 90.0]
        self.calls: list[tuple] = []
        self.closed = False
        self.delay = 0.0
        self.arm = self                      # ``.arm.move_joints`` reaches here

    def connect(self):
        return self

    def close(self):
        self.closed = True

    def stop(self):
        self.calls.append(("stop",))

    def snapshot(self):
        return {"t_wall": time.time(), "pose": list(self.pose),
                "joints_deg": list(self.joints), "tool_twist": [0.0] * 6}

    def get_joint_angles(self):
        return list(self.joints)

    def get_pose(self):
        return list(self.pose)

    def arm_envelope(self, centre=None):
        self.envelope_centre = list(self.pose if centre is None else centre)
        return list(self.envelope_centre)

    def move_joints(self, angles, **kw):
        time.sleep(self.delay)
        self.calls.append(("move_joints", list(angles)))
        self.joints = [float(v) for v in angles]

    def move_home(self, **kw):
        time.sleep(self.delay)
        self.calls.append(("move_home",))

    def move_delta(self, dx=0.0, dy=0.0, dz=0.0, dtheta_x=0.0, dtheta_y=0.0,
                   dtheta_z=0.0, **kw):
        want = [self.pose[0] + dx, self.pose[1] + dy, self.pose[2] + dz,
                self.pose[3] + dtheta_x, self.pose[4] + dtheta_y,
                self.pose[5] + dtheta_z]
        if self.envelope_centre is None:
            raise RuntimeError("the envelope is not armed")
        d = sum((a - b) ** 2 for a, b in zip(want[:3],
                                             self.envelope_centre[:3])) ** 0.5
        if d > self.envelope_r_m + 1e-9:
            raise RuntimeError(f"{d:.4f} m leaves the envelope — REFUSED")
        self.calls.append(("move_delta", tuple(round(v, 6) for v in
                                               (dx, dy, dz, dtheta_x,
                                                dtheta_y, dtheta_z))))
        self.pose = want

    def send_twist(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0),
                   frame="base"):
        """The vendored driver's unguarded mover.  Records; moves no pose.

        ``self.arm = self`` above means ``bridge.arm.arm.send_twist`` finds this,
        which is the path a caller reaching past the wrapper would take.  It does
        NOT integrate the pose: a twist that moved this fake would make the
        envelope tests depend on how long the test's own scheduler took.
        """
        self.calls.append(("send_twist", tuple(float(v) for v in linear),
                           tuple(float(v) for v in angular), str(frame)))

    def twist_checked(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0),
                      frame="base"):
        """The guard the wrapper puts in front of it, re-stated here.

        Same three rules as ``SafeKinovaArm.twist_checked``: refuse when the
        envelope is unarmed, drop the OUTWARD component at the wall rather than
        refusing, and forward.  A fake that simply forwarded would let a bridge
        which never armed the envelope pass every test in this file.
        """
        if self.envelope_centre is None:
            raise RuntimeError("the envelope is not armed")
        lin = [float(v) for v in linear]
        r = [a - b for a, b in zip(self.pose[:3], self.envelope_centre[:3])]
        d = sum(v * v for v in r) ** 0.5
        at_wall = d >= self.envelope_r_m
        if at_wall and d > 1e-12:
            u = [v / d for v in r]
            out = sum(a * b for a, b in zip(lin, u))
            if out > 0.0:
                lin = [a - out * b for a, b in zip(lin, u)]
        self.send_twist(linear=lin, angular=angular, frame=frame)
        return {"linear": lin, "angular": [float(v) for v in angular],
                "distance_m": d, "at_wall": at_wall, "frame": frame}


class FakeRx:
    """A ``KinovaMocapRx`` that never opens a socket.

    It answers the two calls :class:`UMArm_KINOVA.bridge_mocap.MocapHalf` makes
    on the publish path -- ``kinova_frames`` and ``kinova_window`` -- and nothing
    else, because nothing else is on that path.  ``rows=0`` is the case the
    operator will meet most often and the one the panel has to survive: Motive
    not running looks exactly like this.
    """

    def __init__(self, rows: int = 0, start_delay: float = 0.0, pos=None):
        self.rows = int(rows)
        self.start_delay = float(start_delay)
        self.pos = [0.1, 0.2, 0.3] if pos is None else list(pos)
        self.started = 0
        self.stopped = 0

    def start(self):
        time.sleep(self.start_delay)
        self.started += 1
        print("NAT_CONNECT to Motive with 4 1 0 0")     # as the real SDK does

    def stop(self):
        self.stopped += 1

    @property
    def kinova_frames(self):
        return self.rows * self.started

    def kinova_window(self, t0=None, t1=None):
        now = time.monotonic()
        return [(now, time.time(), list(self.pos), [0.0, 0.0, 0.0, 1.0])
                for _ in range(self.rows if self.started else 0)]


def _fake_se3(pose):
    """A rigid transform from a 6-pose, WITHOUT the vendored Euler convention.

    ``base_from_single_sample`` needs a member of SE(3) and nothing more, so a
    stand-in that puts the position in and leaves the rotation at identity
    exercises every line of the solve.  What it deliberately does not claim is
    that intrinsic-xyz degrees are read correctly -- that is the vendored
    driver's convention, it needs ``kortex_api``, and it is checked in
    ``test_kinova_offline.py``.
    """
    import numpy as np

    T = np.eye(4)
    T[0:3, 3] = [float(v) for v in pose[0:3]]
    return T


#: Every mocap half built in this file, so the fixture below can shut them down.
_HALVES: list = []


@pytest.fixture(autouse=True)
def _no_half_outlives_its_test():
    """Stop every receiver a test started, and put ``sys.stdout`` back.

    NOT HOUSEKEEPING.  A running :class:`MocapHalf` holds ``sys.stdout`` as a tee
    for as long as it lives -- that is the point of it, since the vendored SDK
    prints from its own threads whenever it likes -- so a half left running by
    one test is a redirect installed for the rest of the session.  A real
    receiver would also leave non-daemon SDK threads behind, which is how a
    process stops being able to exit.
    """
    _HALVES.clear()
    yield
    for half in _HALVES:
        try:
            half.stop()
        except Exception:
            pass
    _HALVES.clear()


def _tracked_half(rx=None):
    """A :class:`MocapHalf` the fixture above will remember to shut down."""
    rx = FakeRx() if rx is None else rx
    half = BM.MocapHalf(rx_factory=lambda: rx, pose_to_se3=_fake_se3)
    _HALVES.append(half)
    return half


def _bridge(rows: int = 0, start_delay: float = 0.0):
    """A bridge with a fake arm and a fake receiver.  Opens no socket.

    The mocap half is the REAL :class:`MocapHalf` with a stand-in receiver rather
    than a stand-in half, because the part worth testing here is the half's own
    lifecycle -- the worker, the join, the "no frames is a state" rule -- and a
    fake half would have none of it.
    """
    out: list[dict] = []
    fake = FakeArm()
    half = _tracked_half(FakeRx(rows=rows, start_delay=start_delay))
    b = AB.Bridge(emit=out.append, arm_factory=lambda msg, r: fake, mocap=half)
    return b, out, fake


def _drain(bridge, out, what, timeout=2.0):
    """Wait for a reply whose ``cmd`` is *what* and which is finished."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for m in out:
            if m.get("cmd") == what and (m.get("done") or m.get("kind") == "error"):
                return m
        time.sleep(0.01)
    return None


# ---------------------------------------------------------------------------
# The bridge's safety model
# ---------------------------------------------------------------------------

def test_connect_does_not_arm_and_an_unarmed_bridge_refuses_every_move():
    """Two steps, and the first one cannot move the arm.

    This is the property the red border in the GUI is a picture of: connected
    means "reporting", armed means "can move", and they are never the same
    event.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    assert b.connected and not b.armed
    assert b.state()["q_deg"] == fake.joints

    for msg in ({"cmd": "joints", "values_deg": list(fake.joints)},
                {"cmd": "home"},
                {"cmd": "jog", "d_xyz": [0, 0, 1], "speed": "slow"}):
        out.clear()
        b.handle(msg)
        err = [m for m in out if m["kind"] == "error"]
        assert err and "not armed" in err[0]["error"], (msg, out)
    assert fake.calls == []                     # nothing reached the arm at all


def test_arming_centres_the_envelope_where_the_arm_is_standing():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    fake.pose = [0.5, 0.1, 0.2, 90.0, 0.0, 90.0]
    b.handle({"cmd": "arm", "on": True})
    assert b.armed
    assert fake.envelope_centre == fake.pose
    ok = [m for m in out if m.get("cmd") == "arm" and m["kind"] == "ok"]
    assert ok and ok[-1]["armed"] is True
    # ...and a jog that would leave the ball is refused by the arm, not clipped
    b.handle({"cmd": "jog", "d_xyz": [100.0, 0, 0], "speed": "fast"})
    err = _drain(b, out, "jog")
    assert err is not None and err["kind"] == "error"
    assert "envelope" in err["error"] or "REFUSED" in err["error"]


def test_a_joint_command_further_than_the_step_limit_is_refused_whole():
    """A slider dragged across its travel must not become a half-turn of wrist.

    ``SafeKinovaArm`` refuses ``move_joints`` by name because it leaves
    Cartesian space and the envelope has nothing to say about it; reaching it
    deliberately here means owning the limit here.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    want = list(fake.joints)
    want[3] += AB.MAX_JOINT_STEP_DEG + 5.0
    out.clear()
    b.handle({"cmd": "joints", "values_deg": want})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "REFUSED" in err[0]["error"]
    assert not any(c[0] == "move_joints" for c in fake.calls)

    # ...and one inside the limit goes through
    ok_want = list(fake.joints)
    ok_want[3] += AB.MAX_JOINT_STEP_DEG - 1.0
    out.clear()
    b.handle({"cmd": "joints", "values_deg": ok_want})
    assert _drain(b, out, "joints") is not None
    assert any(c[0] == "move_joints" for c in fake.calls)


def test_the_odd_joints_are_continuous_so_the_step_limit_wraps():
    """359 deg and 1 deg are two degrees apart, not 358.

    Joints 1/3/5/7 spin, and a limit that did not wrap would refuse the shortest
    move across zero while allowing the long way round.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    fake.joints = [359.0, 15.0, 180.0, 230.0, 0.0, 55.0, 90.0]
    want = list(fake.joints)
    want[0] = 1.0
    b.handle({"cmd": "joints", "values_deg": want})
    assert _drain(b, out, "joints") is not None
    assert any(c[0] == "move_joints" for c in fake.calls)


def test_stop_lands_while_a_move_is_running_and_disarms():
    """The command loop must never be inside a blocking move.

    A bridge that ran motions inline could not read ``stop`` until the motion
    it was meant to stop had finished, which is the one message that has to get
    through.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    fake.delay = 0.4
    want = list(fake.joints)
    want[1] += 5.0
    b.handle({"cmd": "joints", "values_deg": want})
    assert b.busy == "joints"
    t0 = time.monotonic()
    b.handle({"cmd": "stop"})
    assert time.monotonic() - t0 < 0.2, "stop waited for the move"
    assert not b.armed
    assert ("stop",) in fake.calls
    assert _drain(b, out, "joints") is not None


def test_a_second_move_is_refused_while_one_is_running():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    fake.delay = 0.3
    b.handle({"cmd": "home"})
    out.clear()
    b.handle({"cmd": "home"})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "still running" in err[0]["error"]
    assert _drain(b, out, "home") is not None


def test_the_arm_is_sent_the_motion_that_was_checked_not_the_number_typed():
    """The step limit wraps; the driver does not.

    ``values_deg = [360, ...]`` against an arm at 0 measures a 0.0 deg step and
    used to be forwarded verbatim — a full turn of the wrist through a guard
    that had just certified it as no motion at all.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    fake.joints = [0.0, 15.0, 180.0, 230.0, 0.0, 55.0, 90.0]
    want = list(fake.joints)
    want[0] = 360.0                       # the same place, one turn away
    b.handle({"cmd": "joints", "values_deg": want})
    assert _drain(b, out, "joints") is not None
    sent = [c for c in fake.calls if c[0] == "move_joints"]
    assert sent, "the command was refused instead of normalised"
    assert sent[-1][1][0] == pytest.approx(0.0), (
        f"the arm was asked for {sent[-1][1][0]} deg, not the 0 deg the step "
        f"limit checked")


def test_a_stop_whose_rpc_raises_still_disarms():
    """``armed`` is a local flag and must never depend on the arm answering.

    A Stop RPC raises in exactly the situations STOP is pressed in: a faulted
    arm, a dropped session, an inactivity timeout. It used to abort the handler
    before the disarm, leaving the bridge accepting motion with the panel still
    red.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})

    def boom():
        raise OSError("the arm is faulted")

    fake.stop = boom
    b.handle({"cmd": "stop"})
    assert not b.armed, "a raising Stop left the bridge ARMED"
    assert any("did not answer the Stop" in str(m.get("line", "")) for m in out)
    # ...and the next motion command is refused
    out.clear()
    b.handle({"cmd": "home"})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "not armed" in err[0]["error"]


def test_a_stop_before_the_driver_has_issued_the_action_is_not_lost():
    """The window the vendored driver spends getting to ``ExecuteAction``.

    ``base.Stop()`` aborts what the arm is executing NOW; three RPC round trips
    before that there is nothing to abort. A stop landing there used to flip the
    flag, publish "safe (not armed)", and let the move start anyway.
    """
    import threading

    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})

    gate = threading.Event()
    started = threading.Event()
    real_home = fake.move_home

    def slow_setup(**kw):
        started.set()
        gate.wait(2.0)              # stands in for the driver's RPC round trips
        real_home()

    fake.move_home = slow_setup
    b.handle({"cmd": "home"})
    assert started.wait(2.0)
    b.handle({"cmd": "stop"})       # lands INSIDE the setup window
    gate.set()
    msg = _drain(b, out, "home")
    assert msg is not None and msg["kind"] == "error", msg
    assert "stopped mid-command" in msg["error"]
    assert not b.armed
    # the arm was stopped a second time, now that there was something to stop
    assert sum(1 for c in fake.calls if c[0] == "stop") >= 2


def test_a_stop_that_lands_before_the_worker_runs_cancels_the_move_outright():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    b._stop_epoch += 1              # as ``stop`` would, before the worker ran
    b.armed = False
    b._start("home", fake.move_home)
    msg = _drain(b, out, "home")
    assert msg is not None and msg["kind"] == "error"
    assert "before it started" in msg["error"]
    assert not any(c[0] == "move_home" for c in fake.calls)


def test_an_unfinished_joint_move_is_an_error_not_a_done():
    """``_execute_action_and_wait`` returns False with the action STILL RUNNING.

    Reported as ``done`` it cleared ``busy``, the panel re-enabled every motion
    button, and the next click put a second action on a moving arm — the exact
    condition the single worker exists to prevent.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    fake.move_joints = lambda angles, **kw: False        # timed out, still running
    want = list(fake.joints)
    want[1] += 5.0
    b.handle({"cmd": "joints", "values_deg": want})
    msg = _drain(b, out, "joints")
    assert msg is not None and msg["kind"] == "error", msg
    assert "did not finish" in msg["error"]
    assert ("stop",) in fake.calls, "the arm was not stopped"
    assert not b.armed


def test_a_joint_move_that_aborts_is_caught_by_the_arrival_check():
    """ACTION_ABORT sets the driver's event exactly as ACTION_END does."""
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    # "completes" without moving, which is what an abort looks like
    fake.move_joints = lambda angles, **kw: True
    want = list(fake.joints)
    want[1] += 5.0
    b.handle({"cmd": "joints", "values_deg": want})
    msg = _drain(b, out, "joints")
    assert msg is not None and msg["kind"] == "error", msg
    assert "ABORTED" in msg["error"] and "joint 2" in msg["error"]
    assert not b.armed


def test_a_failed_move_disarms_so_nothing_else_runs_from_a_stale_envelope():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    b.handle({"cmd": "jog", "d_xyz": [50.0, 0, 0], "speed": "fast"})
    assert _drain(b, out, "jog") is not None
    assert not b.armed, "a refused move left the bridge armed"


def test_a_dead_session_shows_as_disconnected_rather_than_as_armed():
    """A red ARMED border against an arm that has gone is worse than no border."""
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})

    def boom():
        raise OSError("the network went away")

    fake.snapshot = boom
    st = b.state()
    assert st["connected"] is False and st["armed"] is False
    assert "lost the arm" in st["error"]
    assert not b.connected


def test_an_unknown_command_is_an_error_not_a_silent_drop():
    b, out, _ = _bridge()
    b.handle({"cmd": "wiggle"})
    assert any(m["kind"] == "error" and "unknown command" in m["error"]
               for m in out)


# ---------------------------------------------------------------------------
# The held jog: a twist, a ramp and a dead-man
# ---------------------------------------------------------------------------

def _twists(fake):
    return [c for c in fake.calls if c[0] == "send_twist"]


def _until(pred, timeout=3.0, dt=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


def _hold(bridge, seconds, **msg):
    """Beat ``jog_start`` the way the panel does, for *seconds*."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        bridge.handle({"cmd": "jog_start", **msg})
        time.sleep(0.05)


def test_the_ramp_is_symmetric_and_a_lost_release_costs_a_known_coast():
    """The profile alone, with no threads, no arm and no clock of its own.

    THIS IS WHERE THE COAST NUMBER IN ``JOG_SPEEDS``' DOCSTRING COMES FROM.  A
    lost release is the ordinary failure of a press-and-hold — the panel's
    release message is best-effort and a button released outside its own
    rectangle never sends one — so the distance it costs is a number the
    operator is entitled to see written down, and this is what pins it.
    """
    dt = 1.0 / AB.JOG_TWIST_HZ
    j = AB.JogProfile()
    t = 0.0
    up = []
    while t < 1.0:                       # held, with beats arriving
        j.beat(d_xyz=[0, 0, 1], speed="normal", now=t)
        lin, _ = j.step(dt, now=t)
        up.append(lin[2])
        t += dt
    assert up == sorted(up), "the ramp up was not monotonic"
    assert max(up) == pytest.approx(AB.JOG_V_MAX, rel=1e-9)
    # a quarter second to the top, which is what JOG_A_MAX was chosen for
    reached = next(i for i, v in enumerate(up) if v >= AB.JOG_V_MAX - 1e-12)
    assert reached * dt == pytest.approx(0.25, abs=dt)

    # ...and now the beats simply stop, which is the LOST release
    t_release, travel, down = t, 0.0, []
    while not j.done and t - t_release < 5.0:
        lin, _ = j.step(dt, now=t)
        down.append(lin[2])
        travel += abs(lin[2]) * dt
        t += dt
    assert down == sorted(down, reverse=True), "the ramp down was not monotonic"
    assert travel * 1e3 == pytest.approx(13.9, abs=0.5), (
        "the coast at 'normal' moved away from the 13.9 mm the panel's help "
        "text and the JOG_SPEEDS docstring both quote")
    # the dead-man holds full speed first, then the ramp: both are visible
    assert (t - t_release) == pytest.approx(AB.JOG_DEADMAN_S + 0.25, abs=0.05)
    for speed, want_mm in (("fast", 27.8), ("slow", 4.9)):
        j = AB.JogProfile()
        t, travel = 0.0, 0.0
        while t < 1.5:
            j.beat(d_xyz=[0, 0, 1], speed=speed, now=t)
            j.step(dt, now=t)
            t += dt
        while not j.done:
            lin, _ = j.step(dt, now=t)
            travel += abs(lin[2]) * dt
            t += dt
        assert travel * 1e3 == pytest.approx(want_mm, abs=0.5), speed


def test_a_held_jog_ramps_the_twist_and_a_lost_release_still_stops_it():
    """The whole thing on the bridge: heartbeat in, twist out, silence stops it.

    The property that matters is the last one.  A Kortex twist has no duration —
    ``cmd.duration = 0`` and the vendored driver has no watchdog — so an arm left
    holding one moves until somebody stops it, and "somebody" cannot be a release
    message that may never arrive.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1], "speed": "normal"})
    assert b.busy == "jog" and b.state()["jogging"] is True
    _hold(b, 0.4, d_xyz=[0, 0, 1], speed="normal")

    vz = [c[1][2] for c in _twists(fake)]
    assert vz, "no twist ever reached the arm"
    assert vz == sorted(vz), f"the twist did not ramp monotonically: {vz}"
    assert max(vz) == pytest.approx(AB.JOG_V_MAX, rel=0.05)
    assert all(abs(c[1][0]) < 1e-12 and abs(c[1][1]) < 1e-12
               for c in _twists(fake)), "a +z jog moved x or y"
    assert all(c[3] == "base" for c in _twists(fake)), (
        "the twist must be commanded in the base frame, since that is the frame "
        "the envelope poll measures in")

    # the release is LOST: no jog_stop, no further beats
    t0 = time.monotonic()
    assert _until(lambda: not b.state()["jogging"], timeout=3.0)
    took = time.monotonic() - t0
    assert AB.JOG_DEADMAN_S <= took <= 1.5, took
    assert fake.calls[-1] == ("stop",), (
        "the jog thread left without stopping the arm; a twist runs until "
        "something stops it")
    assert b.busy == "" and _drain(b, out, "jog_start") is not None


def test_a_stop_mid_hold_halts_the_twist_and_nothing_follows_it():
    """STOP is the one message that has to get through, and the LAST one sent.

    A stop that returned while the jog thread still had a slice queued would be
    followed by another twist 25 ms later, and the arm would carry on moving with
    the panel painted "safe (not armed)".  The assertion is therefore ordering,
    not merely that a Stop happened.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    _hold(b, 0.2, d_xyz=[1, 0, 0], speed="fast")
    assert _twists(fake), "nothing was moving to stop"

    t0 = time.monotonic()
    b.handle({"cmd": "stop"})
    assert time.monotonic() - t0 < 0.2, "stop waited for the jog"
    assert not b.armed
    assert not b.state()["jogging"]
    last_stop = max(i for i, c in enumerate(fake.calls) if c[0] == "stop")
    assert not any(c[0] == "send_twist" for c in fake.calls[last_stop:]), (
        "a twist was issued after the Stop")


def test_a_jog_is_refused_unarmed_and_nothing_reaches_the_arm():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    out.clear()
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1]})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "not armed" in err[0]["error"]
    assert not _twists(fake) and not b.state()["jogging"]


def test_a_jog_and_a_joint_move_cannot_both_run():
    """``busy`` is one field and a jog holds it, exactly as a joint move does."""
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1]})
    out.clear()
    b.handle({"cmd": "joints", "values_deg": list(fake.joints)})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "jog is still running" in err[0]["error"]
    assert not any(c[0] == "move_joints" for c in fake.calls)
    b.handle({"cmd": "jog_stop"})
    assert _until(lambda: not b.state()["jogging"])

    # ...and the other way round
    fake.delay = 0.3
    want = list(fake.joints)
    want[1] += 5.0
    b.handle({"cmd": "joints", "values_deg": want})
    out.clear()
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1]})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "joints is still running" in err[0]["error"]
    assert _drain(b, out, "joints") is not None


def test_at_the_wall_the_outward_half_is_dropped_and_the_way_back_is_not():
    """A refusal at the wall would leave Home as the only way out of the ball.

    Home is the ONE move that is not envelope-checked, so "you have driven to the
    edge, now use the unchecked move" is exactly the wrong instruction.  The
    outward component is dropped instead, which leaves the tangential and inward
    parts of the same button still driving the arm.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    centre = list(fake.envelope_centre)
    fake.pose = [centre[0] + fake.envelope_r_m, *centre[1:]]   # at the wall, +x

    _hold(b, 0.3, d_xyz=[1, 0, 0], speed="normal")
    outward = _twists(fake)
    assert outward, "the jog did not run at all"
    assert all(abs(c[1][0]) < 1e-9 for c in outward), (
        f"the outward component survived the wall: {outward[-1]}")
    assert any("envelope wall" in str(m.get("line", "")) for m in out)
    walls = [m for m in out if "envelope wall" in str(m.get("line", ""))]
    assert len(walls) == 1, (
        "the wall was reported once per slice rather than once per arrival")
    b.handle({"cmd": "jog_stop"})
    assert _until(lambda: not b.state()["jogging"])

    # ...and back the way it came still moves
    fake.calls.clear()
    _hold(b, 0.3, d_xyz=[-1, 0, 0], speed="normal")
    back = _twists(fake)
    assert back and min(c[1][0] for c in back) < -1e-4, (
        "the arm could not drive back off the wall")
    b.handle({"cmd": "jog_stop"})
    assert _until(lambda: not b.state()["jogging"])


def test_a_jog_that_the_arm_refuses_disarms_and_says_so():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})

    def boom(*a, **kw):
        raise RuntimeError("the arm is faulted")

    fake.twist_checked = boom
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1]})
    assert _until(lambda: not b.state()["jogging"])
    assert not b.armed
    assert any(m["kind"] == "error" and "faulted" in m["error"] for m in out)


def test_a_jog_with_no_direction_or_an_unknown_preset_is_refused():
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    for msg, want in (({"cmd": "jog_start"}, "no direction"),
                      ({"cmd": "jog_start", "d_xyz": [0, 0, 0]}, "no direction"),
                      ({"cmd": "jog_start", "d_xyz": [0, 0, 1],
                        "speed": "warp"}, "unknown speed"),
                      ({"cmd": "jog_start", "d_xyz": [0, 0, 1],
                        "frame": "galactic"}, "unknown frame")):
        out.clear()
        b.handle(msg)
        err = [m for m in out if m["kind"] == "error"]
        assert err and want in err[0]["error"], (msg, out)
    assert not _twists(fake)


def test_a_jog_stop_with_nothing_held_is_an_ok_not_an_error():
    """``<Leave>`` fires whenever the pointer crosses the button, held or not."""
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "jog_stop"})
    assert any(m["kind"] == "ok" and m.get("cmd") == "jog_stop" for m in out)
    assert not any(m["kind"] == "error" for m in out)


def test_the_bounded_jog_still_works_because_a_twist_cannot_move_a_distance():
    """``{"cmd": "jog"}`` is kept, and this is the reason it is kept.

    The panel no longer sends it.  It survives because it is the only command
    here that moves a stated DISTANCE and reports arrival, which a velocity
    cannot do, and a hand-typed session at the bridge's prompt has no heartbeat
    to offer.
    """
    b, out, fake = _bridge()
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    b.handle({"cmd": "jog", "d_xyz": [0, 0, 1], "speed": "slow"})
    assert _drain(b, out, "jog") is not None
    moves = [c for c in fake.calls if c[0] == "move_delta"]
    assert moves and moves[-1][1][2] == pytest.approx(AB.JOG_STEP_M["slow"])


# ---------------------------------------------------------------------------
# The cameras: optional, non-blocking, and never a reason to fail
# ---------------------------------------------------------------------------

def test_a_mocap_that_never_sees_a_frame_is_a_state_and_not_an_error():
    """Motive not running is the case this whole half has to survive.

    The arm must stay fully usable in its own base frame, the state must say so,
    and nothing must raise.
    """
    b, out, fake = _bridge(rows=0)
    st = b.state()["mocap"]
    assert st["on"] is False and st["live"] is False and not st["error"]

    b.handle({"cmd": "connect"})            # connect starts it, as asked for
    assert _until(lambda: b.mocap.rx is not None)
    st = b.state()["mocap"]
    assert st["on"] is True, "connect did not ask for the cameras"
    assert st["live"] is False and st["frames"] == 0
    assert not st["error"], st
    assert st["world_ready"] is False
    assert not any(m["kind"] == "error" for m in out)
    # ...and the arm still moves, in its own frame
    b.handle({"cmd": "arm", "on": True})
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1]})
    assert _until(lambda: any(c[0] == "send_twist" for c in fake.calls))
    b.handle({"cmd": "stop"})


def test_connect_does_not_wait_for_the_receiver():
    """A connect that hung on a NatNet socket would hang on the optional half.

    MEASURED against a receiver whose ``start()`` sleeps for half a second: the
    connect returns in a small fraction of that, because the receiver is built on
    a worker.
    """
    b, out, fake = _bridge(start_delay=0.5)
    t0 = time.monotonic()
    b.handle({"cmd": "connect"})
    took = time.monotonic() - t0
    assert took < 0.2, f"connect waited {took:.3f} s for the cameras"
    assert b.connected
    assert _until(lambda: b.mocap.rx is not None, timeout=3.0)
    b.mocap.stop()


def test_a_stop_that_races_a_start_still_stops_the_receiver():
    """The SDK's threads are not daemons; an unstopped receiver never exits.

    MEASURED on this machine: a script that started a receiver and returned from
    ``main`` without stopping it was still alive at a 15 s timeout.  So a stop
    that arrives while the start worker is still inside ``rx.start()`` must not
    simply drop the handle — the worker stops it on the worker's own way out.
    """
    b, out, fake = _bridge(start_delay=0.4)
    b.handle({"cmd": "mocap", "on": True})
    b.handle({"cmd": "mocap", "on": False})       # lands inside rx.start()
    rx = b.mocap.rx_factory()
    assert _until(lambda: rx.stopped >= 1, timeout=3.0), (
        "the receiver was started and then abandoned")
    assert b.mocap.rx is None and b.mocap.wanted is False


def test_the_mocap_can_be_turned_on_without_an_arm_at_all():
    """Checking whether Motive is streaming has nothing to do with the Gen3."""
    b, out, fake = _bridge(rows=60)
    b.handle({"cmd": "mocap", "on": True})
    assert _until(lambda: b.state()["mocap"]["live"])
    st = b.state()["mocap"]
    assert st["frames"] == 60 and st["rate_hz"] > 0
    assert st["pos_m"] == pytest.approx([0.1, 0.2, 0.3])
    assert not b.connected
    b.handle({"cmd": "mocap", "on": False})
    assert b.mocap.rx is None


def test_solving_the_base_with_no_frames_is_an_error_and_not_a_crash():
    b, out, fake = _bridge(rows=0)
    b.handle({"cmd": "connect"})
    out.clear()
    b.handle({"cmd": "mocap_solve_base"})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "frames" in err[0]["error"], out
    assert b.state()["mocap"]["world_ready"] is False


def test_the_world_frame_needs_an_X_solved_this_session():
    """Three things gate it, and the panel's radiobutton is a picture of them.

    ``Y`` survives being wheeled about; ``X`` does not, and nothing the cameras
    can see distinguishes a cart that has been dragged from an arm that has
    moved.  So the world frame is refused until the operator has solved ``X``
    here, now, and the refusal says which of the three is missing.
    """
    b, out, fake = _bridge(rows=60)
    b.handle({"cmd": "connect"})
    b.handle({"cmd": "arm", "on": True})
    assert _until(lambda: b.state()["mocap"]["live"])
    out.clear()
    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1], "frame": "world"})
    err = [m for m in out if m["kind"] == "error"]
    assert err and "world-frame jog refused" in err[0]["error"]
    assert not _twists(fake)

    out.clear()
    b.handle({"cmd": "mocap_solve_base"})
    ok = [m for m in out if m["kind"] == "ok"
          and m.get("cmd") == "mocap_solve_base"]
    assert ok, out
    assert ok[0]["vs_stored_mm"] > 0.0        # a real number, not a placeholder
    assert b.state()["mocap"]["world_ready"] is True

    b.handle({"cmd": "jog_start", "d_xyz": [0, 0, 1], "frame": "world"})
    assert _until(lambda: any(c[0] == "send_twist" for c in fake.calls))
    b.handle({"cmd": "stop"})


def test_a_world_frame_jog_is_rotated_into_the_base_and_not_merely_relabelled():
    """``R_X^T v``, and the translation of X plays no part.

    Adding the base's POSITION to a direction is the most obvious way to get this
    wrong, and it would be invisible on a bench where the base happens to sit
    near the origin of the volume.
    """
    import numpy as np

    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    X = np.eye(4)
    X[0:3, 0:3] = R
    X[0:3, 3] = [5.0, -3.0, 2.0]              # a base far from the origin
    assert BM.world_dir_to_base(X, [1.0, 0.0, 0.0]) == pytest.approx(
        [0.0, -1.0, 0.0])
    assert BM.world_dir_to_base(X, [0.0, 0.0, 1.0]) == pytest.approx(
        [0.0, 0.0, 1.0])
    assert BM.world_dir_to_base(np.eye(4), [0.3, 0.4, 0.0]) == pytest.approx(
        [0.3, 0.4, 0.0])


def test_the_json_channel_survives_the_sdk_stealing_stdout():
    """The vendored NatNet SDK prints to ``sys.stdout``, from its own threads.

    MEASURED in this session: wrapping only ``rx.start()`` in a redirect was not
    enough — in one of two runs the ``resetting requested version`` line arrived
    after the redirect had been unwound.  So the redirect covers the receiver's
    whole life, and the JSON channel is a file object captured before any of it,
    which is what this pins.
    """
    import io

    channel = io.StringIO()
    half = _tracked_half(FakeRx())
    saved_channel, saved_out = AB._JSON_OUT, sys.stdout
    try:
        AB._JSON_OUT = channel
        half._redirect_stdout()
        assert sys.stdout is not channel, "the redirect did not take"
        print("NAT_CONNECT to Motive with 4 1 0 0")     # the SDK's line
        AB._emit({"kind": "state", "connected": False})
    finally:
        half._restore_stdout()
        AB._JSON_OUT = saved_channel
        sys.stdout = saved_out
    assert '"kind": "state"' in channel.getvalue()
    assert "NAT_CONNECT" not in channel.getvalue(), (
        "an SDK print landed on the JSON channel")
    assert any("NAT_CONNECT" in line for line in half.sdk_notes)


def test_serve_stops_the_receiver_on_its_way_out():
    """A bridge that exits ``serve`` with an SDK thread running does not exit."""
    import io

    out: list[dict] = []
    fake = FakeArm()
    rx = FakeRx(rows=60)
    half = _tracked_half(rx)
    script = io.StringIO('{"cmd": "connect"}\n{"cmd": "quit"}\n')
    AB.serve(stdin=script, emit=out.append, publish_hz=50.0,
             arm_factory=lambda msg, r: fake, mocap=half)
    assert rx.started >= 1, "the receiver was never asked for"
    assert rx.stopped >= 1, "serve returned with the receiver still running"
    assert half.rx is None


def test_serve_reads_lines_publishes_state_and_closes_the_session():
    import io

    out: list[dict] = []
    fake = FakeArm()
    script = io.StringIO('{"cmd": "connect"}\n{"cmd": "arm", "on": true}\n'
                         '{"cmd": "quit"}\n')
    # A FAKE RECEIVER, because ``connect`` now asks for one and the real one
    # would open a NatNet socket on whatever machine is running the tests.
    AB.serve(stdin=script, emit=out.append, publish_hz=50.0,
             arm_factory=lambda msg, r: fake,
             mocap=_tracked_half(FakeRx()))
    kinds = [m["kind"] for m in out]
    assert "state" in kinds and kinds[-1] == "bye"
    assert fake.closed, "serve returned without closing the Kortex session"


# ---------------------------------------------------------------------------
# The host-side client
# ---------------------------------------------------------------------------

_STUB = '''
import json, sys, time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    cmd = msg.get("cmd")
    if cmd == "quit":
        break
    sys.stdout.write(json.dumps({"kind": "ok", "cmd": cmd}) + "\\n")
    sys.stdout.write(json.dumps({"kind": "state", "connected": cmd != "disconnect",
                                 "armed": cmd == "arm", "busy": "",
                                 "q_deg": [1, 2, 3, 4, 5, 6, 7],
                                 "pose": [0.4, 0.0, 0.3, 90.0, 0.0, 90.0]}) + "\\n")
    sys.stdout.flush()
sys.stdout.write(json.dumps({"kind": "bye"}) + "\\n")
sys.stdout.flush()
'''


def _wait(link, pred, timeout=6.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        link.poll()
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_the_link_spawns_talks_and_shuts_down(tmp_path):
    """The whole process plumbing, against a stub bridge.

    Deliberately NOT the real bridge: this machine may have no
    ``.venv_kinova``, and what is being tested is the pipe, not the driver.
    """
    stub = tmp_path / "stub_bridge.py"
    stub.write_text(_STUB, encoding="utf-8")
    link = BC.KinovaLink(python=sys.executable, script=str(stub))
    try:
        line = link.start()
        assert "bridge starting" in line or link.state == BC.STARTING, line
        link.send({"cmd": "connect"})
        assert _wait(link, lambda: link.connected), link.describe()
        link.send({"cmd": "arm", "on": True})
        assert _wait(link, lambda: link.armed), link.describe()
        assert link.arm_state["q_deg"] == [1, 2, 3, 4, 5, 6, 7]
        assert "ARMED" in link.describe()
        assert link.fresh
    finally:
        link.shutdown()
    assert link.state == BC.IDLE
    assert not link.connected and not link.armed


def test_the_link_says_so_when_the_venv_is_missing():
    """A missing interpreter is an explanation, not a traceback."""
    link = BC.KinovaLink(python=None, script="nowhere.py")
    line = link.start()
    if BC.venv_python() is None:
        assert ".venv_kinova is missing" in line
    else:
        assert "no bridge script" in line
    assert link.state == BC.IDLE


def test_a_bridge_that_dies_is_noticed_rather_than_left_looking_alive(tmp_path):
    stub = tmp_path / "dies.py"
    stub.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    link = BC.KinovaLink(python=sys.executable, script=str(stub))
    link.start()
    assert _wait(link, lambda: link.state == BC.IDLE), link.describe()
    assert "exited on its own" in link.error
    assert not link.connected


def test_send_is_dropped_rather_than_raising_when_nothing_is_listening():
    link = BC.KinovaLink(python=sys.executable, script="nowhere.py")
    link.send({"cmd": "arm", "on": True})       # must not raise
    link.stop_motion()
    assert link.poll() == []


def test_the_venv_path_is_the_one_setup_env_builds():
    """A bridge started on the wrong interpreter fails with an import error."""
    from UMArm_KINOVA import setup_env as SE

    assert os.path.normcase(os.path.dirname(os.path.dirname(BC.VENV_PYTHON))) \
        == os.path.normcase(SE.VENV_DIR)


if __name__ == "__main__":          # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
