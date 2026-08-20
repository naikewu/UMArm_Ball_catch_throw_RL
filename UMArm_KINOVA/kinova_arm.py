"""The repo's handle on the Kinova Gen3 — the vendored driver plus a leash.

:class:`SafeKinovaArm` wraps :class:`UMArm_KINOVA.vendor.kinova_driver.KinovaArm`
and adds the three things this lab needs and the portable driver deliberately
does not have:

1. **A keep-out envelope.**  Every Cartesian command is checked against a ball
   centred on the pose the arm held when the envelope was armed, and a command
   that leaves it is REFUSED — :class:`EnvelopeViolation` — rather than clipped.
   A clipped command is a command the arm silently did not obey, which is the
   one failure mode a calibration run must never have: the log would say one
   thing and the metal would have done another.
2. **Speed.**  ``move_to_pose`` on the portable driver sends a bare
   ``reach_pose`` action, which the Gen3 executes at its own default rate.
   Here every pose move carries a ``CartesianTrajectoryConstraint`` with an
   explicit translation and orientation speed, so "move it slowly" is a number
   in the log rather than an intention.
3. **Coherent feedback.**  :meth:`snapshot` reads the tool pose *and* the seven
   actuator angles out of ONE ``RefreshFeedback`` call.  The driver's
   ``get_pose`` and ``get_joint_angles`` each refresh separately, so a caller
   that wants both gets two instants; on a stationary arm that is harmless and
   on a moving one it is a silent inconsistency between the two halves of the
   calibration record.

Everything else is the vendored driver's, reached through the delegating
attribute lookup — **except anything that moves the arm**, which is refused
there and has to be taken deliberately off :attr:`SafeKinovaArm.arm`.  One of
those has been taken deliberately and is here:
:meth:`SafeKinovaArm.twist_checked` is a Cartesian velocity, guarded by polling
the live pose against the envelope rather than by pre-checking a target, because
a velocity has no target to pre-check.

Units, exactly as the vendored driver defines them and as the rest of this file
assumes: **positions in metres, orientations in degrees**, a pose is
``[x, y, z, theta_x, theta_y, theta_z]`` with intrinsic-xyz Euler angles, and
the reference frame is the **robot base** unless something says otherwise.

Run this module directly for a no-motion connectivity check::

    .venv_kinova/Scripts/python.exe UMArm_KINOVA/kinova_arm.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:      # runs by path, not just -m
    sys.path.insert(0, os.path.dirname(_HERE))

from UMArm_KINOVA.vendor.kinova_driver import (      # noqa: E402
    KinovaArm, SE3_to_pose, pose_to_SE3)

#: The arm's address on the lab's private network, and its factory credentials.
#: The network has no route off itself, which is why the password lives here.
DEFAULT_IP = "192.168.1.10"
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "admin"

#: Envelope radius, metres.  The operator's instruction for the calibration
#: campaign was "the end effector should stay in a 20 cm diameter ball", i.e. a
#: 100 mm radius; the campaign itself plans to 70 mm so a refusal means a real
#: mistake rather than a rounding error at the wall.
DEFAULT_ENVELOPE_R_M = 0.100

#: Default Cartesian speeds for a pose move.  0.05 m/s crosses the whole
#: envelope in four seconds, which is slow enough to watch and to stop.
DEFAULT_SPEED_MS = 0.05
DEFAULT_SPEED_DEG_S = 15.0

#: How closely an executed move must land on what was asked, before a caller is
#: told the arm did what it was told.  The Gen3's own repeatability is quoted at
#: 0.1 mm; 2 mm and 1 deg therefore catch an aborted or clamped action, not
#: ordinary servo error.
ARRIVAL_TOL_M = 0.002
ARRIVAL_TOL_DEG = 1.0

#: Slack at the envelope wall, metres.  A NANOMETRE, which is five orders of
#: magnitude below the arm's own 0.1 mm repeatability and therefore has no
#: physical meaning at all — it exists to absorb floating-point round-off.
#: A plan built as ``centre + delta`` recovers its distance as
#: ``norm((c + d) - c)``, which is not bit-identical to ``norm(d)``: a campaign
#: asked for exactly the envelope radius came back 2.8e-17 m over and was
#: refused, on 63 % of centres, for 28 attometres.  A limit that fires on that
#: is not a limit, it is a coin toss.
ENVELOPE_EPS_M = 1e-9

#: How closely ``move_home`` must land on the Home action's OWN joint targets,
#: degrees per joint.  Read off the arm rather than assumed — see
#: :meth:`SafeKinovaArm.home_joint_targets`.
HOME_TOL_DEG = 1.0

#: Ceilings on :meth:`SafeKinovaArm.twist_checked`, m/s and deg/s.  They are the
#: METHOD's limit, not any one caller's: whoever streams a twist owns a ramp of
#: their own, and this is the wall underneath every one of them.  Set at the
#: simulated task-space jog's own top speed (``UMArm_COLLAB.gui.backend``'s
#: ``JOG_V_MAX * JOG_SPEEDS["fast"]`` = 0.10 m/s, ``JOG_W_MAX * 2`` = 40 deg/s),
#: so no caller of this method can drive the metal faster than the fastest thing
#: the operator has already held a button on in the twin.  The real arm's own jog
#: (``arm_bridge.JOG_V_MAX``) sits at 0.06 m/s at its fastest preset, well under
#: these, which is deliberate: a ceiling that fires in ordinary use is a clip
#: nobody notices, and this one exists for the caller that has not been written
#: yet.
TWIST_LIN_CEIL_MS = 0.10
TWIST_ANG_CEIL_DEG_S = 40.0

#: Every vendored-driver method that MOVES THE ARM.  Three of them
#: (``move_to_pose``, ``move_delta``, ``move_home``) are OVERRIDDEN on the
#: wrapper and reach the guarded version; ``__getattr__`` never sees those names.
#: The rest are refused, by name, with the guarded alternative in the message.
#: Listing all seven keeps the set honest: a driver that grows an eighth mover
#: has to be added here, and the test parametrises over this tuple.  Delegation is
#: what makes this class thin, and it is also what would have made the envelope
#: one differently-spelled method away from a bypass: ``move_tool_frame_delta``
#: reaches the same planner ``move_to_pose`` does, and ``move_joints`` reaches
#: past Cartesian space entirely.  Reach through ``arm.arm`` deliberately if you
#: mean to, and own the limit when you do.  ``send_twist`` is the one that HAS
#: been reached that way, by :meth:`SafeKinovaArm.twist_checked`, and it stays in
#: this tuple: the guarded alternative is a differently-named method, so the
#: refusal by delegation still holds and still points at it.
UNGUARDED_MOVERS = ("move_to_pose", "move_delta", "move_tool_frame_delta",
                    "move_joints", "send_twist", "smooth_move", "move_home")


def envelope_violations(centre, poses, radius: float = DEFAULT_ENVELOPE_R_M):
    """Which of *poses* leave a ball of *radius* about *centre*.  No hardware.

    The wrapper's own :meth:`SafeKinovaArm.check_envelope` can only run once a
    session is open, because it needs the centre the envelope was armed on.  A
    plan, though, can be checked against a HYPOTHETICAL centre at the desk — and
    that is where a plan that cannot fit should be rejected, not after the arm
    has already been sent Home.

    Returns ``[(index, distance_m), ...]``, empty when the plan fits.
    """
    c = np.asarray(centre, dtype=float)[0:3]
    out = []
    for i, pose in enumerate(poses):
        d = float(np.linalg.norm(np.asarray(pose, dtype=float)[0:3] - c))
        if d > float(radius) + ENVELOPE_EPS_M:
            out.append((i, d))
    return out


class EnvelopeViolation(RuntimeError):
    """A commanded pose left the armed keep-out ball.  Nothing was sent."""


class ArrivalError(RuntimeError):
    """The action ended, but not at the pose it was given."""


def pose_delta(a, b) -> tuple:
    """``(distance_m, angle_deg)`` between two 6-poses.

    The angle is the true geodesic angle of ``Ra^T Rb``, not a difference of
    Euler triples: near a gimbal the Euler difference is meaningless and would
    fail an arrival check on an arm that arrived.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = float(np.linalg.norm(b[0:3] - a[0:3]))
    ra = pose_to_SE3(a)[0:3, 0:3]
    rb = pose_to_SE3(b)[0:3, 0:3]
    cos = (float(np.trace(ra.T @ rb)) - 1.0) / 2.0
    return d, float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


class SafeKinovaArm:
    """A connected Gen3 with an envelope, a speed limit and one-shot feedback.

    Use as a context manager; the vendored driver's session is opened on entry
    and closed (with a ``Stop``) on exit::

        with SafeKinovaArm() as arm:
            arm.arm_envelope()                 # centre on wherever it is now
            arm.move_to_pose(target, speed_ms=0.03)
    """

    def __init__(self, ip: str = DEFAULT_IP, username: str = DEFAULT_USER,
                 password: str = DEFAULT_PASSWORD,
                 envelope_r_m: float = DEFAULT_ENVELOPE_R_M):
        self.arm = KinovaArm(ip=ip, username=username, password=password)
        self.envelope_r_m = float(envelope_r_m)
        #: Centre of the keep-out ball, or ``None`` while it is not armed.  An
        #: unarmed envelope refuses every Cartesian move, so forgetting to arm
        #: it stops the run instead of removing the limit.
        self.envelope_centre = None
        self.ip = ip

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> "SafeKinovaArm":
        self.arm.connect()
        return self

    def close(self) -> None:
        self.arm.close()

    def __enter__(self) -> "SafeKinovaArm":
        return self.connect()

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        self.close()
        return False

    def __getattr__(self, name):
        """Delegate to the vendored driver — except for anything that MOVES.

        Delegation is what keeps this class thin, and left open it would also
        keep the envelope one differently-spelled method away from a bypass:
        ``move_tool_frame_delta`` reaches the same planner ``move_to_pose``
        does, ``smooth_move`` drives a twist, and ``move_joints`` leaves
        Cartesian space altogether.  Every one of those is refused here by name
        (:data:`UNGUARDED_MOVERS`), with the guarded alternative in the message.

        The refusal is not a claim that those methods are wrong — it is a claim
        that they must be reached DELIBERATELY.  ``arm.arm.move_joints(...)``
        still works, and whoever writes it owns the limit.
        """
        if name in ("arm", "envelope_centre", "envelope_r_m", "ip"):
            raise AttributeError(name)
        if name in UNGUARDED_MOVERS:
            raise AttributeError(
                "%s.%s does not exist because it would move the arm without an "
                "envelope check. Use move_to_pose / move_delta / move_home, "
                "which are checked; or reach the raw driver on purpose as "
                ".arm.%s and own the limit yourself."
                % (type(self).__name__, name, name))
        return getattr(self.arm, name)

    # -- feedback ------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """One ``RefreshFeedback`` -> tool pose, joint angles and the clock.

        ``t_wall`` is stamped in this process, so it is the host's idea of when
        the sample was taken and is directly comparable with a mocap timestamp
        read on the same host.  It is not the arm's own clock and the two are
        not synchronised; every consumer in this package samples a STATIONARY
        arm for exactly that reason.
        """
        fb = self.arm.base_cyclic.RefreshFeedback()
        return {
            "t_wall": time.time(),
            "pose": [fb.base.tool_pose_x, fb.base.tool_pose_y,
                     fb.base.tool_pose_z, fb.base.tool_pose_theta_x,
                     fb.base.tool_pose_theta_y, fb.base.tool_pose_theta_z],
            "joints_deg": [act.position for act in fb.actuators],
            "tool_twist": [fb.base.tool_twist_linear_x,
                           fb.base.tool_twist_linear_y,
                           fb.base.tool_twist_linear_z,
                           fb.base.tool_twist_angular_x,
                           fb.base.tool_twist_angular_y,
                           fb.base.tool_twist_angular_z],
        }

    def averaged_snapshot(self, n: int = 20, dt: float = 0.01) -> dict:
        """*n* snapshots, averaged, plus the spread — proof the arm was still.

        The spread matters more than the mean: it is the only evidence in the
        record that the arm had actually settled when the mocap was read, and a
        calibration fitted through a still-moving sample is wrong in a way no
        residual can distinguish from a bad transform.

        THE ORIENTATION IS AVERAGED AS A ROTATION, not as three numbers.  The
        Kortex feedback is an Euler triple, and an Euler triple has a branch cut:
        a wrist sitting near ``theta_z = 180`` reports +179.9 and -179.9 on
        alternate frames, whose element-wise mean is 0 — a rotation half a turn
        from anywhere the arm has been.  Averaging the matrices and projecting
        back onto SO(3) has no such point.  ``ang_ptp_deg`` is the geodesic
        spread, which is the number to gate stillness on; ``pose_sd``'s last
        three entries are kept for the record but are element-wise Euler
        scatter and mean nothing across the cut.
        """
        from UMArm_KINOVA.mocap_calibration import project_rotation

        rows = []
        for _ in range(int(n)):
            rows.append(self.snapshot())
            time.sleep(dt)
        pose = np.array([r["pose"] for r in rows], dtype=float)
        joints = np.array([r["joints_deg"] for r in rows], dtype=float)
        mats = np.array([pose_to_SE3(p)[0:3, 0:3] for p in pose])
        mean_R = project_rotation(mats.mean(axis=0))
        mean_pose = list(SE3_to_pose(
            np.vstack([np.hstack([mean_R, pose[:, 0:3].mean(axis=0)[:, None]]),
                       [0.0, 0.0, 0.0, 1.0]])))
        ang = [pose_delta(mean_pose, p)[1] for p in pose.tolist()]
        return {
            "n": len(rows),
            "t_wall": float(np.mean([r["t_wall"] for r in rows])),
            "pose": mean_pose,
            "pose_sd": pose.std(axis=0).tolist(),
            "pos_ptp_m": float(np.max(np.ptp(pose[:, 0:3], axis=0))),
            "ang_ptp_deg": float(np.ptp(ang)) if len(ang) > 1 else 0.0,
            "ang_max_deg": float(np.max(ang)),
            "joints_deg": joints.mean(axis=0).tolist(),
            "joints_sd_deg": joints.std(axis=0).tolist(),
        }

    # -- envelope ------------------------------------------------------------ #
    def arm_envelope(self, centre=None) -> list:
        """Arm the keep-out ball on *centre*, or on the current pose.

        Returns the centre pose, which the caller should log: every later
        refusal is only meaningful against the point it was measured from.
        """
        self.envelope_centre = list(self.arm.get_pose()) if centre is None \
            else [float(v) for v in centre]
        return list(self.envelope_centre)

    def envelope_distance(self, pose) -> float:
        """How far *pose*'s origin sits from the envelope centre, metres."""
        if self.envelope_centre is None:
            raise EnvelopeViolation(
                "the envelope is not armed — call arm_envelope() first; an "
                "unarmed envelope refuses everything rather than allowing it")
        return float(np.linalg.norm(np.asarray(pose[0:3], dtype=float)
                                    - np.asarray(self.envelope_centre[0:3],
                                                 dtype=float)))

    def check_envelope(self, pose) -> float:
        d = self.envelope_distance(pose)
        if d > self.envelope_r_m + ENVELOPE_EPS_M:
            raise EnvelopeViolation(
                "target %.4f m from the envelope centre, limit %.4f m — "
                "REFUSED, nothing was sent to the arm"
                % (d, self.envelope_r_m))
        return d

    # -- motion -------------------------------------------------------------- #
    def move_to_pose(self, pose, speed_ms: float = DEFAULT_SPEED_MS,
                     speed_deg_s: float = DEFAULT_SPEED_DEG_S,
                     timeout: float = 60.0, name: str = "move",
                     check_arrival: bool = True):
        """Envelope-checked, speed-limited, blocking Cartesian move.

        Returns the pose the arm actually reached.  Raises
        :class:`EnvelopeViolation` before sending anything, or
        :class:`ArrivalError` when the action ends somewhere else — which on
        this arm means a fault, a joint limit or a singularity the planner
        stopped at, and is exactly the case a campaign must not average over.
        """
        from kortex_api.autogen.messages import Base_pb2

        self.check_envelope(pose)
        action = Base_pb2.Action()
        action.name = name
        action.application_data = ""
        cp = action.reach_pose.target_pose
        cp.x, cp.y, cp.z = (float(pose[0]), float(pose[1]), float(pose[2]))
        cp.theta_x, cp.theta_y, cp.theta_z = (float(pose[3]), float(pose[4]),
                                              float(pose[5]))
        con = action.reach_pose.constraint
        con.speed.translation = float(speed_ms)
        con.speed.orientation = float(speed_deg_s)

        finished = self.arm._execute_action_and_wait(action, timeout=timeout)
        if not finished:
            # The action is STILL RUNNING: the wait timed out, it did not end.
            # Reading a pose here and calling it "where the arm stopped" would
            # be a reading taken mid-motion, and leaving the action alive would
            # let the arm keep going while the caller handles the exception.
            self.arm.stop()
            reached = self.arm.get_pose()
            d, ang = pose_delta(pose, reached)
            raise ArrivalError(
                "no END/ABORT within %.0f s; the arm has been STOPPED, %.1f mm "
                "and %.2f deg from the target" % (timeout, d * 1e3, ang))
        reached = self.arm.get_pose()
        if check_arrival:
            d, ang = pose_delta(pose, reached)
            if d > ARRIVAL_TOL_M or ang > ARRIVAL_TOL_DEG:
                # END and ABORT both set the driver's completion event, so
                # "finished" only means the action is no longer running.  This
                # is the check that tells them apart.
                raise ArrivalError(
                    "action ended %.1f mm / %.2f deg from the target "
                    "(tolerance %.1f mm / %.2f deg) — check the web app at "
                    "http://%s for a fault"
                    % (d * 1e3, ang, ARRIVAL_TOL_M * 1e3, ARRIVAL_TOL_DEG,
                       self.ip))
        return reached

    def move_delta(self, dx=0.0, dy=0.0, dz=0.0, dtheta_x=0.0, dtheta_y=0.0,
                   dtheta_z=0.0, **kw):
        """Base-frame relative move, envelope-checked like an absolute one."""
        p = self.arm.get_pose()
        return self.move_to_pose(
            [p[0] + dx, p[1] + dy, p[2] + dz, p[3] + dtheta_x,
             p[4] + dtheta_y, p[5] + dtheta_z], name="delta", **kw)

    def home_joint_targets(self) -> list:
        """The joint angles the arm's own Home action asks for, degrees.

        Read off the controller rather than written down here, because Home is
        the arm's setting and not this repo's: on this Gen3 it is
        ``[0, 15, 180, 230, 0, 55, 90]``, and a project that re-teaches it must
        not silently invalidate the check below.
        """
        from kortex_api.autogen.messages import Base_pb2

        want = Base_pb2.RequestedActionType()
        want.action_type = Base_pb2.REACH_JOINT_ANGLES
        for action in self.arm.base.ReadAllActions(want).action_list:
            if action.name == "Home":
                return [float(j.value) for j in
                        action.reach_joint_angles.joint_angles.joint_angles]
        raise RuntimeError("this arm has no action named 'Home'")

    def move_home(self, timeout: float = 40.0, check_arrival: bool = True):
        """The arm's own built-in Home.  NOT envelope-checked, on purpose.

        Home is the pose the envelope is normally armed *on*, and it is the one
        move that must still work when the arm has wandered — refusing to go
        home because the arm is not near home would be exactly backwards.

        ARRIVAL IS VERIFIED AGAINST THE JOINTS, not against the driver's
        completion flag.  That flag is set by ``ACTION_ABORT`` exactly as it is
        by ``ACTION_END`` (the vendored ``_check_for_end_or_abort`` subscribes to
        both), so an aborted Home returns ``True`` and looks like a Home that
        worked.  Every caller here uses Home as its recovery path, which is the
        worst possible place to believe a flag.
        """
        finished = self.arm.move_home(timeout=timeout)
        if not finished:
            self.arm.stop()
            raise ArrivalError("Home did not complete within %.0f s; the arm "
                               "has been STOPPED" % timeout)
        if check_arrival:
            want = self.home_joint_targets()
            got = self.arm.get_joint_angles()
            if len(got) != len(want):
                raise ArrivalError(
                    "Home wants %d joint angles and the arm reports %d"
                    % (len(want), len(got)))
            # Wrapped, because the Gen3's odd joints are continuous: 230 deg and
            # -130 deg are the same place and only one of them is what Home says.
            err = [abs((g - w + 180.0) % 360.0 - 180.0)
                   for g, w in zip(got, want)]
            if max(err) > HOME_TOL_DEG:
                raise ArrivalError(
                    "the Home action reported completion but joint %d is %.2f "
                    "deg away from its target (tolerance %.2f deg) — Home was "
                    "ABORTED, not reached; check http://%s for a fault"
                    % (int(max(range(len(err)), key=err.__getitem__)),
                       max(err), HOME_TOL_DEG, self.ip))
        return self.arm.get_pose()

    def twist_checked(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0),
                      frame: str = "base") -> dict:
        """A Cartesian velocity, leashed by a POLL of where the arm actually is.

        THE ONE UNGUARDED MOVER THIS CLASS REACHES ON PURPOSE.  ``send_twist``
        stays in :data:`UNGUARDED_MOVERS` and stays refused by
        :meth:`__getattr__`, because a velocity that anybody can pick up by
        spelling an attribute is a velocity nobody owns; this method is where the
        limit is owned, and it is the only place in this package that calls
        ``self.arm.send_twist``.

        WHY A POLL AND NOT A PRE-CHECK.  :meth:`check_envelope` needs a target
        pose, and a velocity has none — it has a direction and no end.  So the
        envelope is enforced the other way round: every call reads the live pose
        out of :meth:`snapshot` and measures :meth:`envelope_distance` from it,
        and a caller streaming at its own rate therefore gets the wall checked at
        that rate rather than once at the start.  The cost is stated: a twist is
        checked where the arm IS, not where it is going, so between two calls the
        arm travels ``v * dt`` unchecked.  At the bridge's 40 Hz and its 0.06 m/s
        fastest preset that is 1.5 mm, which is the resolution of this guard and
        is the number a caller should compare against its own clearances.

        AT THE WALL THE OUTWARD COMPONENT IS ZEROED, NOT REFUSED.  An operator
        who has driven to the edge of the keep-out ball must still be able to
        drive back out of it; a hard refusal at the wall leaves the arm stuck
        with Home as the only way out, and Home is the one move that is not
        envelope-checked at all.  So the commanded velocity is projected: the
        component along the outward radius is dropped and the tangential and
        inward parts are passed through.  The arm can then slide along the ball
        and come back in, and cannot leave.  Rotation is NOT projected — an
        angular velocity does not move the tool origin to first order, and the
        translation the wrist's lever arm does produce is caught by the next
        poll.

        Returns what was actually commanded, so a caller can see its own request
        being clipped rather than infer it from the arm not moving.
        """
        lin = np.array(linear, dtype=float).ravel()
        ang = np.array(angular, dtype=float).ravel()
        if lin.size != 3 or ang.size != 3:
            raise ValueError("a twist is two 3-vectors, got %d and %d components"
                             % (lin.size, ang.size))
        snap = self.snapshot()
        d = self.envelope_distance(snap["pose"])        # raises when unarmed

        at_wall = d >= self.envelope_r_m
        if at_wall and d > 1e-12:
            radial = (np.asarray(snap["pose"][0:3], dtype=float)
                      - np.asarray(self.envelope_centre[0:3], dtype=float)) / d
            outward = float(np.dot(lin, radial))
            if outward > 0.0:
                lin = lin - outward * radial

        for v, ceil in ((lin, TWIST_LIN_CEIL_MS), (ang, TWIST_ANG_CEIL_DEG_S)):
            n = float(np.linalg.norm(v))
            if n > ceil > 0.0:
                v *= ceil / n

        self.arm.send_twist(linear=lin.tolist(), angular=ang.tolist(),
                            frame=frame)
        return {"linear": lin.tolist(), "angular": ang.tolist(),
                "distance_m": float(d), "at_wall": bool(at_wall),
                "frame": str(frame)}

    def stop(self) -> None:
        """Stop whatever is running.

        Also the reason ``send_twist`` is refused by :meth:`__getattr__` rather
        than merely documented as unchecked: a velocity has no target pose to
        check, so the envelope has nothing to say about it, and an unchecked
        mover that is one attribute lookup away is an unchecked mover somebody
        will reach for.  Anything that genuinely wants a twist takes
        :meth:`twist_checked`, which owns the limit, and pairs it with this call
        — a twist runs until another twist or a Stop, so the pairing is not
        optional.
        """
        self.arm.stop()


def tool_configuration(arm: SafeKinovaArm) -> dict:
    """The tool transform and payload the arm has configured, if it will say.

    This matters more than it looks.  ``tool_pose_*`` is the pose of the arm's
    configured TOOL frame, not of the bare flange, so a tool offset left in the
    controller by an earlier project silently redefines every number this
    package fits.  Recording it makes the calibration reproducible; a later run
    that disagrees can at least see why.  Returns ``{"available": False, ...}``
    on an arm or an SDK build that does not expose the service, since that is a
    missing note rather than a failure.
    """
    try:
        from kortex_api.autogen.client_stubs.ControlConfigClientRpc import (
            ControlConfigClient)
        cc = ControlConfigClient(arm.arm._router)
        tool = cc.GetToolConfiguration()
        return {
            "available": True,
            "tool_transform_m_deg": [tool.tool_transform.x, tool.tool_transform.y,
                                     tool.tool_transform.z,
                                     tool.tool_transform.theta_x,
                                     tool.tool_transform.theta_y,
                                     tool.tool_transform.theta_z],
            "tool_mass_kg": float(tool.tool_mass),
            "tool_mass_centre_m": [tool.tool_mass_center.x,
                                   tool.tool_mass_center.y,
                                   tool.tool_mass_center.z],
        }
    except Exception as exc:                      # pragma: no cover - hardware
        return {"available": False, "why": "%s: %s" % (type(exc).__name__, exc)}


def main(argv=None) -> int:
    """Connect, read, print, disconnect.  Moves nothing."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--ip", default=DEFAULT_IP)
    args = ap.parse_args(argv)

    with SafeKinovaArm(ip=args.ip) as arm:
        snap = arm.averaged_snapshot(n=10)
        print("tool pose  (m, deg):",
              " ".join("%8.4f" % v for v in snap["pose"]))
        print("pose ptp   (mm)    : %.3f" % (snap["pos_ptp_m"] * 1e3))
        print("joints     (deg)   :",
              " ".join("%7.2f" % v for v in snap["joints_deg"]))
        print("tool config        :", json.dumps(tool_configuration(arm)))
        centre = arm.arm_envelope()
        print("envelope armed on  :",
              " ".join("%8.4f" % v for v in centre[0:3]),
              "radius %.0f mm" % (arm.envelope_r_m * 1e3))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
