"""Everything about the Kinova stack that can be checked with no arm.

The three modules under test here each have a part that only the hardware can
exercise and a part that is ordinary logic, and the logic is the part that
decides whether the hardware run is safe:
:class:`~UMArm_KINOVA.kinova_arm.SafeKinovaArm`'s envelope is the only thing
between a typo in a campaign script and the metal,
:func:`~UMArm_KINOVA.setup_env.patch_protobuf_abc` is the only reason
``kortex_api`` imports at all on CPython >= 3.10, and
:func:`~UMArm_KINOVA.kinova_mocap.summarise` is the arithmetic every recorded
pose passes through.  All three are checked here against answers written down
in advance, so that a disagreement on the rig is about the rig.

Nothing in this file opens a socket.  ``SafeKinovaArm`` is constructed but never
connected — the vendored ``KinovaArm.__init__`` only stores the connection
arguments and sets ``base``/``base_cyclic`` to ``None``, and every send goes
through those — and ``KinovaMocapRx`` is constructed but never started, since
``MocapRx.__init__`` likewise only builds rings and locks and leaves
``self._client = None`` for :meth:`~UMArm_MOCAP.mocap_rx.MocapRx.start` to fill.
Where a test needs to prove that a refusal came *before* a send, the send path
is monkeypatched to raise :class:`_Sent`, so a guard that stopped working would
surface as ``_Sent`` rather than as a silent pass.

This module imports ``kortex_api`` (through ``kinova_arm`` -> the vendored
driver), which exists only in ``.venv_kinova``.  Run it with that interpreter::

    .venv_kinova/Scripts/python.exe -m pytest UMArm_KINOVA/test_kinova_offline.py -q

Under the repo's ordinary interpreter the ``importorskip`` below skips the whole
module, so ``python -m pytest UMArm_KINOVA -q`` stays green.
"""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import pytest

#: The whole module needs the Kortex SDK.  Skipping is the honest outcome under
#: the repo's ordinary interpreter: these tests are not "passing" there, and a
#: collection error would be indistinguishable from a broken test file.
pytest.importorskip("kortex_api",
                    reason="kortex_api lives only in .venv_kinova — run this "
                           "file with .venv_kinova/Scripts/python.exe")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UMArm_KINOVA import arm_bridge as AB          # noqa: E402
from UMArm_KINOVA import kinova_arm as KA          # noqa: E402
from UMArm_KINOVA import kinova_mocap as KM        # noqa: E402
from UMArm_KINOVA import setup_env as SE           # noqa: E402
from UMArm_KINOVA.vendor.kinova_driver import pose_to_SE3   # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _Sent(AssertionError):
    """Raised by the stubbed driver: something reached the arm that should not.

    An ``AssertionError`` on purpose — if a guard under test stops working, the
    failure reads as a failed test rather than as an error in the fixture.
    """


#: A plausible parked pose for this rig: 46 cm out, 43 cm up, wrist pointing
#: down.  Nothing depends on the exact numbers; they are here so the poses in
#: this file look like poses the arm could actually hold.
HOME_POSE = [0.46, 0.02, 0.43, 180.0, 0.0, 90.0]


def _offline_arm(envelope_r_m: float = KA.DEFAULT_ENVELOPE_R_M) -> KA.SafeKinovaArm:
    """A ``SafeKinovaArm`` that has never connected, with its sends booby-trapped.

    ``get_pose``, ``_execute_action_and_wait`` and ``stop`` are replaced on the
    vendored driver instance so that any attempt to reach the arm raises
    :class:`_Sent`.  Without the stubs those calls would fail with
    ``AttributeError`` on a ``None`` client, which is also a failure but not one
    that distinguishes "the guard held" from "the guard let it through".
    """
    arm = KA.SafeKinovaArm(ip="0.0.0.0", envelope_r_m=envelope_r_m)

    def _boom(*_a, **_kw):
        raise _Sent("the driver was reached")

    arm.arm.get_pose = _boom
    arm.arm._execute_action_and_wait = _boom
    arm.arm.stop = _boom
    return arm


def _pose(x, y, z, tx=180.0, ty=0.0, tz=90.0):
    return [float(x), float(y), float(z), float(tx), float(ty), float(tz)]


# --------------------------------------------------------------------------- #
# SafeKinovaArm — the envelope, unarmed
# --------------------------------------------------------------------------- #
def test_construction_does_not_connect():
    """Constructing the wrapper leaves the driver's clients unmade.

    This assumption is what every other test in this file rests on, so it is
    checked rather than assumed.
    """
    arm = KA.SafeKinovaArm(ip="0.0.0.0")
    assert arm.arm.base is None
    assert arm.arm.base_cyclic is None
    assert arm.envelope_centre is None


@pytest.mark.parametrize("pose", [
    _pose(0.0, 0.0, 0.0),          # the origin — the tempting special case
    _pose(*HOME_POSE[0:3]),        # a pose the arm could plausibly be at
    _pose(0.0, 0.0, 1e-9),         # a hair off the origin
    _pose(-3.0, 12.0, -7.5),       # far outside anything reachable
])
def test_unarmed_envelope_refuses_every_pose(pose):
    """An unarmed envelope refuses, it does not pass.

    The origin is in the list deliberately: a guard written as "no centre means
    no offset" would admit ``[0, 0, 0]`` and refuse everything else, which is
    the one failure this design is meant to be immune to — forgetting to arm
    must stop the run rather than remove the limit.
    """
    arm = _offline_arm()
    with pytest.raises(KA.EnvelopeViolation) as exc:
        arm.check_envelope(pose)
    assert "not armed" in str(exc.value)
    assert "arm_envelope()" in str(exc.value)


def test_unarmed_envelope_distance_also_refuses():
    """The distance query refuses too, so no caller can compute its way around."""
    arm = _offline_arm()
    with pytest.raises(KA.EnvelopeViolation):
        arm.envelope_distance(_pose(0.0, 0.0, 0.0))


def test_unarmed_move_to_pose_refuses_before_sending_anything():
    arm = _offline_arm()
    with pytest.raises(KA.EnvelopeViolation):
        arm.move_to_pose(HOME_POSE)


# --------------------------------------------------------------------------- #
# SafeKinovaArm — the envelope, armed
# --------------------------------------------------------------------------- #
def test_armed_envelope_admits_inside_and_returns_the_distance():
    arm = _offline_arm(envelope_r_m=0.100)
    centre = arm.arm_envelope(HOME_POSE)
    assert centre == pytest.approx(HOME_POSE)

    target = _pose(HOME_POSE[0] + 0.03, HOME_POSE[1] - 0.04, HOME_POSE[2])
    d = arm.check_envelope(target)
    assert d == pytest.approx(0.05, abs=1e-12)          # 3-4-5, exactly
    assert arm.envelope_distance(target) == pytest.approx(d)


def test_armed_envelope_refuses_just_outside_and_names_both_numbers():
    """The refusal message carries the measured distance and the limit.

    A refusal that only says "refused" cannot be triaged from a log: the
    operator cannot tell a 1 mm overrun (a typo in a target) from a 400 mm one
    (the wrong frame entirely).
    """
    r = 0.070
    arm = _offline_arm(envelope_r_m=r)
    arm.arm_envelope(HOME_POSE)

    over = 0.0801                                       # 10.01 mm past the wall
    target = _pose(HOME_POSE[0] + over, HOME_POSE[1], HOME_POSE[2])
    with pytest.raises(KA.EnvelopeViolation) as exc:
        arm.check_envelope(target)
    msg = str(exc.value)
    assert "%.4f" % over in msg, msg                    # the distance reached
    assert "%.4f" % r in msg, msg                       # the limit it broke
    assert "REFUSED" in msg
    assert "nothing was sent" in msg


def test_envelope_wall_sits_at_the_radius():
    """Just inside passes, just outside raises — the limit is the radius itself."""
    r = 0.100
    arm = _offline_arm(envelope_r_m=r)
    arm.arm_envelope(_pose(0.0, 0.0, 0.0))
    assert arm.check_envelope(_pose(r * (1.0 - 1e-9), 0.0, 0.0)) < r
    with pytest.raises(KA.EnvelopeViolation):
        arm.check_envelope(_pose(r * (1.0 + 1e-6), 0.0, 0.0))


def test_envelope_ignores_orientation():
    """The ball is on the tool origin: a pure re-orientation never leaves it.

    Stated as a test because it is a real limitation, not an oversight — the
    envelope constrains where the tool point goes, and a wrist flip that sweeps
    the gripper through something is outside what it can see.
    """
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    spun = list(HOME_POSE[0:3]) + [-90.0, 45.0, -170.0]
    assert arm.check_envelope(spun) == pytest.approx(0.0)


def test_armed_centre_is_a_snapshot_not_a_reference():
    """Mutating the list handed in, or the list handed back, must not move the ball."""
    arm = _offline_arm()
    centre_in = list(HOME_POSE)
    centre_out = arm.arm_envelope(centre_in)
    centre_in[0] += 10.0
    centre_out[1] += 10.0
    assert arm.envelope_centre == pytest.approx(HOME_POSE)


def test_envelope_radius_is_the_one_passed_in():
    arm = _offline_arm(envelope_r_m=0.020)
    arm.arm_envelope(_pose(0.0, 0.0, 0.0))
    assert arm.check_envelope(_pose(0.019, 0.0, 0.0)) == pytest.approx(0.019)
    with pytest.raises(KA.EnvelopeViolation):
        arm.check_envelope(_pose(0.021, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# SafeKinovaArm — the refusal happens BEFORE anything is sent
# --------------------------------------------------------------------------- #
def test_the_fixture_would_have_caught_a_send():
    """Control for the two tests below: an ALLOWED move does reach the driver.

    Without this, a refusal test would pass just as well against a
    ``move_to_pose`` that had been accidentally turned into a no-op.
    """
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    with pytest.raises(_Sent):
        arm.move_to_pose(HOME_POSE)


def test_move_to_pose_refuses_before_reaching_the_driver():
    arm = _offline_arm(envelope_r_m=0.070)
    arm.arm_envelope(HOME_POSE)
    far = _pose(HOME_POSE[0] + 0.50, HOME_POSE[1], HOME_POSE[2])
    with pytest.raises(KA.EnvelopeViolation):
        arm.move_to_pose(far)          # _Sent here would mean it was sent


def test_move_delta_is_envelope_checked_too():
    """The relative move is guarded by the same wall, on the resolved target."""
    arm = _offline_arm(envelope_r_m=0.070)
    arm.arm_envelope(HOME_POSE)
    arm.arm.get_pose = lambda: list(HOME_POSE)      # reading is allowed
    with pytest.raises(KA.EnvelopeViolation):
        arm.move_delta(dx=0.50)
    # ... and the same call inside the ball does get through to the driver.
    with pytest.raises(_Sent):
        arm.move_delta(dx=0.01)


def test_move_to_pose_cannot_be_bypassed_by_delegation():
    """``__getattr__`` must not hand out the driver's unguarded ``move_to_pose``.

    The wrapper delegates unknown names to the vendored driver, so the envelope
    would be one attribute lookup away from bypass if either move method were
    ever renamed or removed here.
    """
    arm = _offline_arm()
    assert arm.move_to_pose.__func__ is KA.SafeKinovaArm.move_to_pose
    assert arm.move_delta.__func__ is KA.SafeKinovaArm.move_delta
    # Delegation itself still works, for the names the wrapper does not define.
    assert arm.get_pose_SE3.__self__ is arm.arm
    assert arm.get_pose_SE3.__func__ is type(arm.arm).get_pose_SE3


# --------------------------------------------------------------------------- #
# SafeKinovaArm — what an allowed move actually builds and checks
# --------------------------------------------------------------------------- #
def test_allowed_move_carries_the_speed_constraint_and_the_target():
    """The speed limit is a field in the message, not an intention.

    The action is captured instead of sent, so the check is on exactly the bytes
    the arm would have received.
    """
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    target = _pose(HOME_POSE[0] + 0.02, HOME_POSE[1], HOME_POSE[2] - 0.01)
    seen = {}

    def _capture(action, timeout=None):
        seen["action"] = action
        seen["timeout"] = timeout
        return True

    arm.arm._execute_action_and_wait = _capture
    arm.arm.get_pose = lambda: list(target)         # it arrived exactly

    reached = arm.move_to_pose(target, speed_ms=0.03, speed_deg_s=7.5,
                               timeout=12.0, name="probe")
    assert reached == pytest.approx(target)

    act = seen["action"]
    assert act.name == "probe"
    assert seen["timeout"] == pytest.approx(12.0)
    cp = act.reach_pose.target_pose
    assert [cp.x, cp.y, cp.z] == pytest.approx(target[0:3], abs=1e-6)
    assert [cp.theta_x, cp.theta_y, cp.theta_z] == pytest.approx(target[3:6],
                                                                 abs=1e-4)
    con = act.reach_pose.constraint
    assert con.speed.translation == pytest.approx(0.03, abs=1e-6)
    assert con.speed.orientation == pytest.approx(7.5, abs=1e-4)


def test_a_timeout_stops_the_arm_before_it_raises():
    """A timeout is not a failed move — it is a move still in progress.

    ``_execute_action_and_wait`` returning False means the WAIT expired, not
    that the action ended, so the ``reach_pose`` is still executing.  Raising
    without stopping would hand the caller an exception while the arm carried on
    across the room, and the error text would assert a stop that never happened.
    """
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    stops = []
    arm.arm._execute_action_and_wait = lambda *a, **k: False
    arm.arm.stop = lambda: stops.append(1)
    arm.arm.get_pose = lambda: list(HOME_POSE)
    with pytest.raises(KA.ArrivalError) as exc:
        arm.move_to_pose(_pose(HOME_POSE[0] + 0.02, HOME_POSE[1], HOME_POSE[2]),
                         timeout=5.0)
    assert "no END/ABORT" in str(exc.value)
    assert "STOPPED" in str(exc.value)
    assert stops == [1], "the arm was not stopped before the error was raised"


def test_a_timeout_raises_even_with_the_arrival_check_off():
    """``check_arrival=False`` waives the POSE check, not the running action.

    The two are different failures.  A caller may reasonably not care whether
    the arm landed within 2 mm; no caller may leave an action running, so the
    timeout branch is unconditional.
    """
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    stops = []
    arm.arm._execute_action_and_wait = lambda *a, **k: False
    arm.arm.stop = lambda: stops.append(1)
    arm.arm.get_pose = lambda: list(HOME_POSE)
    with pytest.raises(KA.ArrivalError):
        arm.move_to_pose(_pose(HOME_POSE[0] + 0.04, HOME_POSE[1], HOME_POSE[2]),
                         check_arrival=False)
    assert stops == [1]


def test_arrival_error_when_the_arm_stopped_short():
    """An action that ends 20 mm from its target is a fault, not a rounding error."""
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    target = _pose(HOME_POSE[0] + 0.04, HOME_POSE[1], HOME_POSE[2])
    stopped = _pose(HOME_POSE[0] + 0.02, HOME_POSE[1], HOME_POSE[2])
    arm.arm._execute_action_and_wait = lambda *a, **k: True
    arm.arm.get_pose = lambda: list(stopped)
    with pytest.raises(KA.ArrivalError) as exc:
        arm.move_to_pose(target)
    assert "20.0 mm" in str(exc.value)
    assert "%.1f" % (KA.ARRIVAL_TOL_M * 1e3) in str(exc.value)


def test_arrival_check_can_be_switched_off():
    """``check_arrival=False`` returns where it got to, without raising.

    The action must still have ENDED — see
    ``test_a_timeout_raises_even_with_the_arrival_check_off`` for the other half.
    """
    arm = _offline_arm()
    arm.arm_envelope(HOME_POSE)
    stopped = _pose(HOME_POSE[0] + 0.02, HOME_POSE[1], HOME_POSE[2])
    arm.arm._execute_action_and_wait = lambda *a, **k: True
    arm.arm.get_pose = lambda: list(stopped)
    got = arm.move_to_pose(_pose(HOME_POSE[0] + 0.04, HOME_POSE[1], HOME_POSE[2]),
                           check_arrival=False)
    assert got == pytest.approx(stopped)


# --------------------------------------------------------------------------- #
# SafeKinovaArm - the delegation hole
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", KA.UNGUARDED_MOVERS)
def test_the_unguarded_movers_are_not_reachable_by_delegation(name):
    """The envelope must not be one differently-spelled method away.

    ``__getattr__`` used to hand out every driver method this class did not
    define, which included ``move_tool_frame_delta`` (the same planner as
    ``move_to_pose``), ``move_joints`` (past Cartesian space entirely) and
    ``smooth_move`` (a ramped twist).  Each is now refused by name, and the
    message has to say what to use instead: a refusal with no alternative gets
    worked around rather than heeded.
    """
    arm = _offline_arm()
    guarded = {"move_to_pose", "move_delta", "move_home"}
    if name in guarded:
        assert callable(getattr(arm, name)), \
            f"{name} is defined on the wrapper and must stay reachable"
        return
    with pytest.raises(AttributeError) as exc:
        getattr(arm, name)
    assert "envelope" in str(exc.value)
    assert ".arm." + name in str(exc.value), "no escape hatch named"
    # ...and the raw driver still has it, so the escape hatch is real.
    assert hasattr(arm.arm, name)


def test_delegation_still_works_for_everything_that_does_not_move():
    arm = _offline_arm()
    for name in ("get_joint_angles", "get_pose_SE3", "base", "base_cyclic"):
        getattr(arm, name)


# --------------------------------------------------------------------------- #
# SafeKinovaArm.twist_checked - the one unguarded mover, taken on purpose
# --------------------------------------------------------------------------- #
def _twist_arm(pose=None, envelope_r_m=KA.DEFAULT_ENVELOPE_R_M):
    """An offline arm whose ``snapshot`` answers and whose twists are recorded.

    ``snapshot`` is stubbed on the WRAPPER rather than on the driver, because it
    reads ``base_cyclic.RefreshFeedback()`` and there is no client here; what is
    under test is the guard in front of the send, not the feedback read.
    """
    arm = _offline_arm(envelope_r_m=envelope_r_m)
    sent = []
    here = list(HOME_POSE if pose is None else pose)
    arm.snapshot = lambda: {"t_wall": 0.0, "pose": list(here),
                            "joints_deg": [0.0] * 7, "tool_twist": [0.0] * 6}
    arm.arm.send_twist = lambda linear, angular, frame="base": sent.append(
        (list(linear), list(angular), frame))
    return arm, sent


def test_a_twist_is_refused_outright_while_the_envelope_is_unarmed():
    """The same rule every other Cartesian command here obeys.

    An unarmed envelope refuses rather than allows, so forgetting to arm stops
    the session instead of removing the limit — and a velocity is the one command
    where "removing the limit" means an arm that never stops on its own.
    """
    arm, sent = _twist_arm()
    with pytest.raises(KA.EnvelopeViolation) as exc:
        arm.twist_checked(linear=[0.0, 0.0, 0.01])
    assert "not armed" in str(exc.value)
    assert sent == [], "a twist reached the driver with no envelope armed"


def test_a_twist_inside_the_ball_passes_through_untouched():
    arm, sent = _twist_arm()
    arm.arm_envelope(HOME_POSE)
    out = arm.twist_checked(linear=[0.01, -0.02, 0.03],
                            angular=[1.0, 0.0, -2.0])
    assert sent and sent[-1][0] == pytest.approx([0.01, -0.02, 0.03])
    assert sent[-1][1] == pytest.approx([1.0, 0.0, -2.0])
    assert sent[-1][2] == "base"
    assert out["at_wall"] is False
    assert out["distance_m"] == pytest.approx(0.0, abs=1e-12)


def test_at_the_wall_only_the_outward_component_is_dropped():
    """A hard refusal at the wall would leave Home as the only way out.

    Home is the ONE move that is not envelope-checked, so telling an operator who
    has driven to the edge to use it is exactly backwards.  What is dropped is
    the outward component; tangential and inward still drive the arm, so the tool
    can slide along the ball and come back in, and cannot leave.
    """
    r = KA.DEFAULT_ENVELOPE_R_M
    centre = list(HOME_POSE)
    at_wall = _pose(centre[0] + r, centre[1], centre[2])
    arm, sent = _twist_arm(pose=at_wall)
    arm.arm_envelope(centre)
    assert arm.envelope_distance(at_wall) == pytest.approx(r)

    out = arm.twist_checked(linear=[0.02, 0.0, 0.0])           # straight out
    assert out["at_wall"] is True
    assert sent[-1][0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)

    out = arm.twist_checked(linear=[-0.02, 0.0, 0.0])          # straight back
    assert sent[-1][0] == pytest.approx([-0.02, 0.0, 0.0])

    out = arm.twist_checked(linear=[0.0, 0.02, 0.0])           # tangential
    assert sent[-1][0] == pytest.approx([0.0, 0.02, 0.0])

    # ...and a diagonal keeps exactly its tangential half
    arm.twist_checked(linear=[0.02, 0.02, 0.0])
    assert sent[-1][0] == pytest.approx([0.0, 0.02, 0.0])

    # THE ROTATION IS NOT PROJECTED, and that is a decision rather than an
    # oversight: an angular velocity does not move the tool origin to first
    # order, and what the wrist's lever arm does produce is caught by the next
    # poll a slice later.
    arm.twist_checked(linear=[0.02, 0.0, 0.0], angular=[5.0, 0.0, 0.0])
    assert sent[-1][1] == pytest.approx([5.0, 0.0, 0.0])


def test_a_twist_is_clamped_to_the_methods_own_ceiling():
    """The ceiling belongs to the method, not to whoever is ramping into it.

    The bridge's fastest preset is 0.06 m/s, well under this, which is
    deliberate: a ceiling that fires in ordinary use is a clip nobody notices,
    and this one exists for the caller that has not been written yet.
    """
    arm, sent = _twist_arm()
    arm.arm_envelope(HOME_POSE)
    arm.twist_checked(linear=[10.0, 0.0, 0.0], angular=[0.0, 900.0, 0.0])
    assert np.linalg.norm(sent[-1][0]) == pytest.approx(KA.TWIST_LIN_CEIL_MS)
    assert np.linalg.norm(sent[-1][1]) == pytest.approx(KA.TWIST_ANG_CEIL_DEG_S)
    # the DIRECTION survives the clamp; only the magnitude is touched
    assert sent[-1][0][1] == pytest.approx(0.0) \
        and sent[-1][0][2] == pytest.approx(0.0)
    assert KA.TWIST_LIN_CEIL_MS >= max(
        AB.JOG_V_MAX * s for s in AB.JOG_SPEEDS.values()), (
        "the bridge's own jog can outrun the ceiling it is supposed to sit "
        "under")


def test_a_malformed_twist_is_a_refusal_rather_than_a_broadcast():
    arm, sent = _twist_arm()
    arm.arm_envelope(HOME_POSE)
    with pytest.raises(ValueError):
        arm.twist_checked(linear=[0.01, 0.0])
    assert sent == []


def test_the_bridges_analysis_walk_picks_the_newest_campaign():
    """The ``listdir`` walk, checked against the directory it walks.

    PORT NOTE (workspace port, 2026-08-20).  Upstream this test held
    ``bridge_mocap.latest_analysis_path`` against the second copy of the same
    walk in ``check_pad_markers.latest_analysis``.  That module is about the
    printed collision pad, pulls ``UMArm_COLLAB.mount_transforms``, and was
    left out of this workspace, so the second copy no longer exists.  The
    property the duplication was guarding is still worth asserting, and it is
    asserted directly here instead: campaign folders are timestamp-named, so
    the newest ``analysis.json`` is the last one in sorted order.
    """
    from UMArm_KINOVA import bridge_mocap as BM

    here = os.path.dirname(os.path.abspath(BM.__file__))
    root = os.path.join(here, "results")
    expected = [os.path.join(root, name, "analysis.json")
                for name in sorted(os.listdir(root))
                if os.path.exists(os.path.join(root, name, "analysis.json"))]
    assert expected, "no committed calibration campaign under results/"
    assert os.path.normcase(BM.latest_analysis_path()) \
        == os.path.normcase(expected[-1])


# --------------------------------------------------------------------------- #
# pose_delta — the geodesic angle
# --------------------------------------------------------------------------- #
def test_pose_delta_is_zero_for_identical_poses():
    d, ang = KA.pose_delta(HOME_POSE, list(HOME_POSE))
    assert d == pytest.approx(0.0, abs=1e-15)
    assert ang == pytest.approx(0.0, abs=1e-9)


def test_pose_delta_half_turn():
    """A 180 deg turn reads as 180 deg, the largest angle SO(3) has."""
    a = _pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for axis in range(3):
        b = _pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        b[3 + axis] = 180.0
        d, ang = KA.pose_delta(a, b)
        assert d == pytest.approx(0.0, abs=1e-15)
        assert ang == pytest.approx(180.0, abs=1e-6)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("deg", [0.25, 3.0, 12.5])
def test_pose_delta_small_turn_about_each_axis(axis, deg):
    a = _pose(0.1, -0.2, 0.3, 0.0, 0.0, 0.0)
    b = list(a)
    b[3 + axis] += deg
    d, ang = KA.pose_delta(a, b)
    assert d == pytest.approx(0.0, abs=1e-15)
    assert ang == pytest.approx(deg, abs=1e-6)


def test_pose_delta_position_only():
    """Poses that differ only in position: a distance, and no angle at all.

    The angle tolerance is 1e-4 deg rather than exact zero because the arccos of
    a trace is flat at its argument's maximum: the Euler-to-matrix round trip
    leaves the trace about 1e-16 short of 3, and the arccos turns that into
    ~2e-6 deg.  That is six orders of magnitude below the 1 deg arrival check it
    feeds, so it is a property of the formula rather than of the poses.
    """
    a = _pose(0.10, 0.20, 0.30, 12.0, -34.0, 56.0)
    b = _pose(0.10 + 0.003, 0.20 - 0.004, 0.30, 12.0, -34.0, 56.0)
    d, ang = KA.pose_delta(a, b)
    assert d == pytest.approx(0.005, abs=1e-12)
    assert ang == pytest.approx(0.0, abs=1e-4)


def test_pose_delta_is_symmetric():
    a = _pose(0.10, 0.20, 0.30, 175.0, -12.0, 88.0)
    b = _pose(-0.05, 0.44, 0.11, -20.0, 61.0, 5.0)
    da, anga = KA.pose_delta(a, b)
    db, angb = KA.pose_delta(b, a)
    assert da == pytest.approx(db, rel=1e-12)
    assert anga == pytest.approx(angb, rel=1e-9)
    assert 0.0 < anga <= 180.0


def test_pose_delta_beats_a_difference_of_euler_triples():
    """Two Euler triples 359.8 deg apart, 0.2 deg apart as rotations.

    This wrap-around is why the arrival check cannot subtract Euler angles: the
    naive difference would raise :class:`~UMArm_KINOVA.kinova_arm.ArrivalError`
    on an arm that arrived.
    """
    a = _pose(0.0, 0.0, 0.0, 0.0, 0.0, 179.9)
    b = _pose(0.0, 0.0, 0.0, 0.0, 0.0, -179.9)
    _, ang = KA.pose_delta(a, b)
    assert ang == pytest.approx(0.2, abs=1e-6)
    assert abs(a[5] - b[5]) == pytest.approx(359.8)      # the trap it avoids
    assert ang < KA.ARRIVAL_TOL_DEG                      # and would have failed


def test_pose_delta_angle_matches_the_rotation_matrices():
    """Cross-check against an independent trace computation on random poses."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        a = list(rng.uniform(-0.5, 0.5, 3)) + list(rng.uniform(-180, 180, 3))
        b = list(rng.uniform(-0.5, 0.5, 3)) + list(rng.uniform(-180, 180, 3))
        d, ang = KA.pose_delta(a, b)
        assert d == pytest.approx(
            float(np.linalg.norm(np.array(b[0:3]) - np.array(a[0:3]))))
        ra = pose_to_SE3(a)[0:3, 0:3]
        rb = pose_to_SE3(b)[0:3, 0:3]
        cos = np.clip((np.trace(ra.T @ rb) - 1.0) / 2.0, -1.0, 1.0)
        assert ang == pytest.approx(math.degrees(math.acos(cos)), abs=1e-9)


# --------------------------------------------------------------------------- #
# setup_env.patch_protobuf_abc
# --------------------------------------------------------------------------- #
_CONTAINERS_SRC = """\
import collections


class ScalarMap(collections.MutableMapping):
    pass


class RepeatedScalarFieldContainer(collections.MutableSequence):
    pass


def _is_mapping(x):
    return isinstance(x, collections.Mapping)


ALREADY_FIXED = collections.abc.MutableMapping
NOT_AN_ABC = collections.OrderedDict
NOT_OURS = mycollections.MutableMapping
"""

_WKT_SRC = """\
import collections

_SET = collections.Set
_SEQ = collections.Sequence
_MSET = collections.MutableSet
"""


def _protobuf_tree(root, containers=_CONTAINERS_SRC, wkt=_WKT_SRC,
                   drop=()) -> str:
    """Write a minimal fake ``site-packages`` and return its path."""
    internal = os.path.join(root, "google", "protobuf", "internal")
    os.makedirs(internal, exist_ok=True)
    for name, src in (("containers.py", containers),
                      ("well_known_types.py", wkt)):
        if name in drop:
            continue
        with open(os.path.join(internal, name), "w", encoding="utf-8",
                  newline="") as fh:
            fh.write(src)
    return root


def _read(root, name):
    with open(os.path.join(root, "google", "protobuf", "internal", name),
              encoding="utf-8") as fh:
        return fh.read()


def test_patch_rewrites_then_reports_nothing_left_to_do(tmp_path):
    """Correct on the first pass, a no-op on the second — byte for byte.

    Idempotence is not cosmetic here: :func:`~UMArm_KINOVA.setup_env.build`
    re-runs the patch on every repair of an existing environment, and a rewrite
    that also matched its own output would walk
    ``collections.abc.abc.MutableMapping`` one level deeper each time.
    """
    root = _protobuf_tree(str(tmp_path / "sp"))

    changed = SE.patch_protobuf_abc(root)
    assert len(changed) == 2
    assert {os.path.basename(p) for p in changed} == {"containers.py",
                                                      "well_known_types.py"}

    after = _read(root, "containers.py")
    assert "collections.abc.MutableMapping" in after
    assert "collections.abc.MutableSequence" in after
    assert "collections.abc.Mapping" in after
    # An unqualified reference must be gone.  Matched with the same word
    # boundary the patch uses, so ``mycollections.MutableMapping`` — which is
    # supposed to survive — is not mistaken for one that was missed.
    assert re.search(r"\bcollections\.MutableMapping\b", after) is None
    assert re.search(r"\bcollections\.MutableSequence\b", after) is None
    assert "collections.abc.abc" not in after

    wkt_after = _read(root, "well_known_types.py")
    for name in ("Set", "Sequence", "MutableSet"):
        assert "collections.abc." + name in wkt_after

    again = SE.patch_protobuf_abc(root)
    assert again == []                              # zero changes reported
    assert _read(root, "containers.py") == after    # and zero changes made
    assert _read(root, "well_known_types.py") == wkt_after


def test_patch_leaves_everything_that_is_not_a_moved_abc(tmp_path):
    """``collections.OrderedDict`` stays, and a name that merely ends in
    ``collections`` is not a match either."""
    root = _protobuf_tree(str(tmp_path / "sp"))
    SE.patch_protobuf_abc(root)
    after = _read(root, "containers.py")
    assert "NOT_AN_ABC = collections.OrderedDict" in after
    assert "NOT_OURS = mycollections.MutableMapping" in after
    assert "ALREADY_FIXED = collections.abc.MutableMapping" in after


@pytest.mark.parametrize("name", list(SE._ABC_NAMES))
def test_patch_covers_every_abc_it_claims_to(name, tmp_path):
    """Each name in ``_ABC_NAMES`` is actually rewritten.

    A name listed but not matched by the regex would be a patch that passes its
    own self-test and then fails at ``import kortex_api``.
    """
    root = _protobuf_tree(str(tmp_path / "sp"),
                          containers="x = collections.%s\n" % name,
                          wkt="y = collections.%s\n" % name)
    SE.patch_protobuf_abc(root)
    assert _read(root, "containers.py") == "x = collections.abc.%s\n" % name


def test_patch_raises_when_protobuf_is_absent(tmp_path):
    """A silently skipped patch is an environment that fails much later.

    The failure would then surface as ``AttributeError: module 'collections' has
    no attribute 'MutableMapping'`` at the first ``import kortex_api``, a long
    way from the install that caused it.
    """
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    with pytest.raises(FileNotFoundError) as exc:
        SE.patch_protobuf_abc(empty)
    assert "containers.py" in str(exc.value)
    assert "missing" in str(exc.value)


def test_patch_raises_on_a_half_installed_tree(tmp_path):
    """One of the two files present is still a refusal, not a partial success."""
    root = _protobuf_tree(str(tmp_path / "sp"), drop=("well_known_types.py",))
    with pytest.raises(FileNotFoundError) as exc:
        SE.patch_protobuf_abc(root)
    assert "well_known_types.py" in str(exc.value)


def test_patch_files_and_venv_paths_are_where_the_module_says():
    """The two constants the whole environment build depends on."""
    assert SE._PATCH_FILES == ("internal/containers.py",
                               "internal/well_known_types.py")
    assert SE.VENV_PYTHON.endswith("python.exe") or \
        SE.VENV_PYTHON.endswith("python")
    assert os.path.basename(SE.VENV_DIR) == ".venv_kinova"


# --------------------------------------------------------------------------- #
# kinova_mocap.quat_xyzw_to_matrix
# --------------------------------------------------------------------------- #
_S = math.sqrt(0.5)

#: ``(quaternion xyzw, the rotation it must produce)``.  The matrices are
#: written out rather than generated, so a sign convention that flipped would
#: have to flip these too.
_QUAT_CASES = [
    ((0.0, 0.0, 0.0, 1.0), np.eye(3)),
    ((_S, 0.0, 0.0, _S), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)),
    ((0.0, _S, 0.0, _S), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)),
    ((0.0, 0.0, _S, _S), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)),
    ((1.0, 0.0, 0.0, 0.0), np.diag([1.0, -1.0, -1.0])),      # 180 deg about x
    ((0.0, 0.0, 1.0, 0.0), np.diag([-1.0, -1.0, 1.0])),      # 180 deg about z
]


@pytest.mark.parametrize("q,R", _QUAT_CASES)
def test_quat_to_matrix_known_cases(q, R):
    assert KM.quat_xyzw_to_matrix(q) == pytest.approx(R, abs=1e-12)


@pytest.mark.parametrize("q,_R", _QUAT_CASES)
def test_quat_to_matrix_returns_a_proper_rotation(q, _R):
    """Orthonormal with determinant +1 — a reflection would fit points too."""
    R = KM.quat_xyzw_to_matrix(q)
    assert R.shape == (3, 3)
    assert R.T @ R == pytest.approx(np.eye(3), abs=1e-12)
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-12)


def test_quat_to_matrix_normalises_its_input():
    """Motive's quaternions arrive unit; a scaled one must not scale the frame."""
    R = KM.quat_xyzw_to_matrix((0.0, 0.0, 3.0 * _S, 3.0 * _S))
    assert R == pytest.approx(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                       float), abs=1e-12)


def test_quat_to_matrix_survives_the_degenerate_quaternion():
    """Motive sends ``(0, 0, 0, 0)`` for a body it cannot solve.

    Returning the identity keeps a NaN out of every downstream mean; the caller
    finds the frame through the spread it reports, not through a NaN that
    poisons the whole window.
    """
    R = KM.quat_xyzw_to_matrix((0.0, 0.0, 0.0, 0.0))
    assert np.all(np.isfinite(R))
    assert R == pytest.approx(np.eye(3), abs=1e-15)


def test_quat_to_matrix_agrees_with_the_axis_angle_it_encodes():
    """Random quaternions, checked against the angle they were built from."""
    rng = np.random.default_rng(11)
    for _ in range(30):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = float(rng.uniform(0.05, math.pi - 0.05))
        q = (*(axis * math.sin(ang / 2.0)), math.cos(ang / 2.0))
        R = KM.quat_xyzw_to_matrix(q)
        got = math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        assert got == pytest.approx(ang, abs=1e-9)
        assert R @ axis == pytest.approx(axis, abs=1e-9)   # axis is fixed


# --------------------------------------------------------------------------- #
# kinova_mocap.summarise
# --------------------------------------------------------------------------- #
def _rows(positions, quat=(0.0, 0.0, 0.0, 1.0), t0=100.0, dt=1.0 / 120.0):
    """Synthetic ``(t_mono, t_wall, pos, quat)`` rows, as the ring stores them."""
    positions = np.asarray(positions, dtype=float)
    return [(t0 + i * dt, 1.7e9 + t0 + i * dt, positions[i], np.asarray(quat,
                                                                        float))
            for i in range(len(positions))]


def test_summarise_mean_position_and_duration():
    centre = np.array([1.234, -0.567, 0.890])
    offsets = np.array([[+0.001, 0, 0], [-0.001, 0, 0],
                        [0, +0.002, 0], [0, -0.002, 0]])
    rows = _rows(centre + offsets, t0=50.0, dt=0.25)

    s = KM.summarise(rows)
    assert s["n"] == 4
    assert s["pos_m"] == pytest.approx(centre, abs=1e-12)
    assert s["duration_s"] == pytest.approx(0.75, abs=1e-12)   # 3 gaps of 0.25
    assert s["t_mono"] == pytest.approx(50.0 + 0.375, abs=1e-9)
    assert s["t_wall"] == pytest.approx(1.7e9 + 50.0 + 0.375, abs=1e-6)


@pytest.mark.parametrize("half_mm", [0.05, 0.5, 5.0])
def test_summarise_spread_tracks_the_spread_put_in(half_mm):
    """``pos_ptp_mm`` is the widest axis' peak-to-peak, in millimetres.

    It is the only evidence in a calibration record that the arm had settled, so
    it has to move with the input rather than merely be non-zero.
    """
    h = half_mm * 1e-3
    centre = np.array([0.4, 0.0, 0.5])
    rows = _rows(centre + np.array([[-h, 0, 0], [0, 0, 0], [+h, 0, 0],
                                    [0, h / 2, 0]]))
    s = KM.summarise(rows)
    assert s["pos_ptp_mm"] == pytest.approx(2.0 * half_mm, rel=1e-9)
    assert s["pos_m"] == pytest.approx(centre + [0.0, h / 8.0, 0.0], abs=1e-15)
    assert s["pos_sd_mm"][0] == pytest.approx(
        float(np.std([-h, 0.0, h, 0.0])) * 1e3, rel=1e-9)


def test_summarise_spread_is_zero_when_nothing_moved():
    rows = _rows(np.tile([0.3, 0.1, 0.2], (12, 1)))
    s = KM.summarise(rows)
    assert s["pos_ptp_mm"] == pytest.approx(0.0, abs=1e-12)
    assert s["ang_ptp_deg"] == pytest.approx(0.0, abs=1e-9)
    assert s["ang_sd_deg"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("q", [c[0] for c in _QUAT_CASES])
def test_summarise_rotation_mean_is_the_rotation_put_in(q):
    """When every row agrees, the projected mean must be exactly that rotation."""
    rows = _rows(np.tile([0.1, 0.2, 0.3], (30, 1)), quat=q)
    s = KM.summarise(rows)
    T = s["T"]
    assert T.shape == (4, 4)
    assert T[0:3, 0:3] == pytest.approx(KM.quat_xyzw_to_matrix(q), abs=1e-9)
    assert T[0:3, 3] == pytest.approx([0.1, 0.2, 0.3], abs=1e-12)
    assert T[3, :] == pytest.approx([0, 0, 0, 1], abs=1e-15)


def test_summarise_angular_spread_tracks_a_wobble():
    """Rotations at -a, 0, +a about z: the mean is 0 and the spread is a.

    The window is deliberately tight; over a fraction of a degree the chordal
    mean this function uses and the geodesic mean agree far below Motive's own
    jitter, which is the assumption the docstring makes and this test pins.
    """
    a = 0.4                                     # degrees
    pos = np.tile([0.0, 0.0, 0.0], (3, 1))
    rows = []
    for i, deg in enumerate((-a, 0.0, +a)):
        h = math.radians(deg) / 2.0
        rows.append((100.0 + i * 0.01, 1.7e9 + i * 0.01, pos[i],
                     np.array([0.0, 0.0, math.sin(h), math.cos(h)])))
    s = KM.summarise(rows)
    assert s["T"][0:3, 0:3] == pytest.approx(np.eye(3), abs=1e-9)
    assert s["ang_ptp_deg"] == pytest.approx(a, abs=1e-6)

    wider = []
    for i, deg in enumerate((-2 * a, 0.0, +2 * a)):
        h = math.radians(deg) / 2.0
        wider.append((100.0 + i * 0.01, 1.7e9 + i * 0.01, pos[i],
                      np.array([0.0, 0.0, math.sin(h), math.cos(h)])))
    assert KM.summarise(wider)["ang_ptp_deg"] == pytest.approx(2 * a, abs=1e-6)


def test_summarise_of_a_single_row_reports_no_spread_rather_than_failing():
    rows = _rows(np.array([[0.5, 0.5, 0.5]]))
    s = KM.summarise(rows)
    assert s["n"] == 1
    assert s["duration_s"] == pytest.approx(0.0)
    assert s["ang_ptp_deg"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# KinovaMocapRx — the added ring, driven by hand
# --------------------------------------------------------------------------- #
# ``MocapRx.__init__`` builds rings, locks and counters and ends with
# ``self._client = None``; the NatNet client is imported and started only in
# ``start()``.  Constructing the receiver therefore opens no socket, and these
# tests call the SDK listeners directly rather than letting anything stream.
class _FakeLabeledMarker:
    def __init__(self, model_id, slot, pos):
        self.id_num = (int(model_id) << 16) | int(slot)
        self.pos = tuple(float(v) for v in pos)
        self.param = 0


class _FakeLabeledData:
    def __init__(self, markers):
        self.labeled_marker_list = list(markers)


class _FakeMocapData:
    """Only what ``_on_mocap_data`` reads: no marker sets, some labeled markers.

    With no per-asset marker sets and no markers in the arm's id block, the
    parent's handler returns early, so the fake never has to imitate the rest of
    a ``MoCapData``.
    """

    def __init__(self, markers):
        self.labeled_marker_data = _FakeLabeledData(markers)


def _receiver():
    return KM.KinovaMocapRx(server_ip="0.0.0.0", client_ip="0.0.0.0")


def test_receiver_construction_starts_nothing():
    rx = _receiver()
    assert rx._client is None
    assert rx.kinova_frames == 0
    assert rx.kinova_window() == []
    assert rx.kinova_marker_window() == []


def test_receiver_records_only_the_kinova_body():
    """Body 1008 lands in the new ring; the arm's own plates do not."""
    rx = _receiver()
    rx._on_rigid_body(KM.mc.KINOVA_MOCAP_STREAM_ID, (1.0, 2.0, 3.0),
                      (0.0, 0.0, 0.0, 1.0))
    assert rx.kinova_frames == 1
    rx._on_rigid_body(KM.mc.RIGID_BODY_ID_MASK + 0, (0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0, 1.0))
    rx._on_rigid_body(7777, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert rx.kinova_frames == 1

    rows = rx.kinova_window()
    assert len(rows) == 1
    assert rows[0][2] == pytest.approx([1.0, 2.0, 3.0])


def test_receiver_window_filters_by_monotonic_time():
    rx = _receiver()
    for k in range(5):
        rx._on_rigid_body(KM.mc.KINOVA_MOCAP_STREAM_ID, (float(k), 0.0, 0.0),
                          (0.0, 0.0, 0.0, 1.0))
    all_rows = rx.kinova_window()
    assert len(all_rows) == 5
    cut = all_rows[2][0]
    later = rx.kinova_window(t0=cut)
    assert len(later) == 3
    assert later[0][2][0] == pytest.approx(2.0)
    assert rx.kinova_window(t0=all_rows[-1][0] + 1.0) == []


def test_receiver_ring_stores_copies():
    """A caller mutating what it got back must not corrupt the ring."""
    rx = _receiver()
    pos = np.array([1.0, 2.0, 3.0])
    rx._on_rigid_body(KM.mc.KINOVA_MOCAP_STREAM_ID, pos, (0.0, 0.0, 0.0, 1.0))
    pos[0] = 99.0                                   # the SDK reuses its buffers
    assert rx.kinova_window()[0][2][0] == pytest.approx(1.0)


def test_receiver_slots_labeled_markers_and_nans_the_missing_one():
    """Four rows always, so a consumer counting four never silently gets three."""
    rx = _receiver()
    pts = {1: (0.1, 0.0, 0.0), 2: (0.0, 0.1, 0.0),
           3: (0.0, 0.0, 0.1), 4: (0.1, 0.1, 0.1)}
    rx._on_mocap_data(_FakeMocapData(
        [_FakeLabeledMarker(KM.mc.KINOVA_MOCAP_STREAM_ID, s, p)
         for s, p in pts.items()]))
    rows = rx.kinova_marker_window()
    assert len(rows) == 1
    arr = rows[0][1]
    assert arr.shape == (4, 3)
    for s, p in pts.items():
        assert arr[s - 1] == pytest.approx(p)

    # Slot 3 occluded: the array keeps its width and the row goes NaN.
    rx._on_mocap_data(_FakeMocapData(
        [_FakeLabeledMarker(KM.mc.KINOVA_MOCAP_STREAM_ID, s, p)
         for s, p in pts.items() if s != 3]))
    arr2 = rx.kinova_marker_window()[-1][1]
    assert arr2.shape == (4, 3)
    assert np.all(np.isnan(arr2[2]))
    assert np.all(np.isfinite(arr2[[0, 1, 3]]))


def test_receiver_ignores_labeled_markers_of_other_bodies():
    rx = _receiver()
    rx._on_mocap_data(_FakeMocapData(
        [_FakeLabeledMarker(KM.mc.RIGID_BODY_ID_MASK + 1, 1, (0.0, 0.0, 0.0)),
         _FakeLabeledMarker(4242, 2, (0.0, 0.0, 0.0))]))
    assert rx.kinova_marker_window() == []


def test_capture_refuses_a_window_it_cannot_average():
    """Too few frames is a refusal, not a mean of three samples.

    ``seconds=0.0`` makes this instant: the window opens after every row above
    was recorded, so it is empty by construction — the same shape of failure a
    silent Motive produces.
    """
    rx = _receiver()
    for _ in range(3):
        rx._on_rigid_body(KM.mc.KINOVA_MOCAP_STREAM_ID, (0.0, 0.0, 0.0),
                          (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(RuntimeError) as exc:
        rx.capture(seconds=0.0)
    assert str(KM.MIN_CAPTURE_FRAMES) in str(exc.value)
    assert str(KM.mc.KINOVA_MOCAP_STREAM_ID) in str(exc.value)

    with pytest.raises(RuntimeError) as exc2:
        rx.capture_markers(seconds=0.0)
    assert "labeled markers" in str(exc2.value)


def test_ring_capacity_matches_the_arm_receiver_s_span():
    """20 s at Motive's nominal rate, the same window the q ring answers."""
    assert KM.RING_CAPACITY == int(KM.mc.RING_SECONDS * KM.mc.NOMINAL_RATE_HZ)
    assert KM.RING_CAPACITY == KM.mc.RING_CAPACITY

# --------------------------------------------------------------------------- #
# The envelope wall, after the 2026-08-19 review
# --------------------------------------------------------------------------- #
def test_a_plan_built_at_exactly_the_radius_is_not_refused_by_roundoff():
    """``norm((c + d) - c)`` is not bit-identical to ``norm(d)``.

    A campaign asked for exactly the envelope radius came back 2.8e-17 m over
    and was refused — on 63 % of plausible centres — for 28 attometres.  A limit
    that fires on that is not a limit, it is a coin toss.  Reproduced here on the
    real Home pose the dry run prints.
    """
    import itertools

    from UMArm_KINOVA import calibrate_mocap as CM

    centre = [0.4615, 0.0159, 0.4337, 90.16, 0.27, 90.65]
    arm = _offline_arm(envelope_r_m=0.100)
    arm.arm_envelope(centre)
    for _, pose in CM.plan_poses(centre, CM.build_plan(0.100)):
        arm.check_envelope(pose)          # must not raise

    # and the round-off really is there: at least one pose is over by a hair
    over = [d for d in
            (arm.envelope_distance(p) - 0.100
             for _, p in CM.plan_poses(centre, CM.build_plan(0.100)))
            if d > 0.0]
    assert over, "the round-off this tolerance exists for is gone; re-derive it"
    assert max(over) < KA.ENVELOPE_EPS_M

    # ...while a real over-reach is still refused
    far = list(centre)
    far[0] += 0.1005
    with pytest.raises(KA.EnvelopeViolation):
        arm.check_envelope(far)
    del itertools


def test_the_tolerance_is_physically_meaningless():
    """It absorbs float noise and nothing else: five orders under repeatability."""
    assert KA.ENVELOPE_EPS_M < 1e-6
    arm = _offline_arm(envelope_r_m=0.100)
    arm.arm_envelope(_pose(0.0, 0.0, 0.0))
    arm.check_envelope(_pose(0.100 + KA.ENVELOPE_EPS_M / 2.0, 0.0, 0.0))
    with pytest.raises(KA.EnvelopeViolation):
        arm.check_envelope(_pose(0.100 + 1e-6, 0.0, 0.0))


def test_envelope_violations_needs_no_hardware_and_finds_them_all():
    """A plan that cannot fit should be refused at the desk, not after Home."""
    centre = _pose(0.4, 0.0, 0.4)
    inside = [_pose(0.4 + 0.05, 0.0, 0.4), _pose(0.4, 0.09, 0.4)]
    outside = [_pose(0.4 + 0.2, 0.0, 0.4), _pose(0.4, 0.0, 0.4 - 0.15)]
    bad = KA.envelope_violations(centre, inside + outside, radius=0.100)
    assert [i for i, _ in bad] == [2, 3]
    assert all(d > 0.100 for _, d in bad)
    assert KA.envelope_violations(centre, inside, radius=0.100) == []


def test_the_dry_run_says_whether_the_plan_fits():
    """It used to print the plan and answer nothing.

    The dry run exists to be the check before the arm is touched, so a plan that
    would abort mid-campaign has to fail HERE, with a non-zero exit.
    """
    from UMArm_KINOVA import calibrate_mocap as CM

    assert CM.main(["--dry-run", "--radius", "0.100"]) == 0
    assert CM.main(["--dry-run", "--radius", "0.070"]) == 0
    # a radius the envelope cannot hold is refused before the plan is even built
    assert CM.main(["--dry-run", "--radius", "0.150"]) == 2
    assert CM.main(["--yes", "--radius", "0.150"]) == 2, \
        "a bad radius must be refused before --yes reaches the arm"
