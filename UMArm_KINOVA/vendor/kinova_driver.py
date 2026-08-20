#!/usr/bin/env python3
"""
kinova_driver.py — portable, standalone high-level wrapper for the Kinova Gen3.

This module distills the reusable parts of the lab's Kinova code into a single
class with **no dependency on the larger project** (no robot_constants, no
kinematics_mp, no mocap). It depends only on:

    * kortex_api        (install the bundled wheel — see README.md §3)
    * utilities.py      (shipped alongside this file: connection helpers)
    * numpy, scipy      (for the pose <-> SE(3) helpers)

Conventions
-----------
* Positions are in METERS, in the robot base frame.
* Orientations are in DEGREES, intrinsic XYZ Euler angles (theta_x/y/z),
  matching the Kortex `tool_pose_theta_*` feedback fields.
* A "pose" is a 6-list: ``[x, y, z, theta_x, theta_y, theta_z]``.

Typical use
-----------
    from kinova_driver import KinovaArm

    with KinovaArm(ip="192.168.1.10") as arm:
        print(arm.get_pose())
        arm.move_home()
        arm.move_delta(dx=0.05, dz=-0.03)
        arm.move_to_pose([0.45, 0.0, 0.35, 180.0, 0.0, 90.0])

The ``with`` block logs into a session on entry and closes it (and stops any
running twist) on exit.
"""

import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

# VENDORED DEVIATION (the only edit to this file — see PROVENANCE.md).  The
# original is a loose script and says ``import utilities``; inside a package
# that finds nothing.  Relative first, absolute second, so the same file works
# both as ``UMArm_KINOVA.vendor.kinova_driver`` and as a script run from this
# directory.
try:
    from . import utilities            # imported as part of the package
except ImportError:                    # pragma: no cover - loose-script path
    import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2


# Maximum time to wait for a blocking action to finish (seconds).
TIMEOUT_DURATION = 20


# --------------------------------------------------------------------------- #
# Small connection-args shim so you don't need argparse to pass ip/user/pass.
# --------------------------------------------------------------------------- #
class _ConnArgs:
    def __init__(self, ip="192.168.1.10", username="admin", password="admin"):
        self.ip = ip
        self.username = username
        self.password = password


def _check_for_end_or_abort(event):
    """Return a Kortex notification callback that sets *event* on END/ABORT."""
    def check(notification, e=event):
        # Uncomment for verbose action logging:
        # print("EVENT :", Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            e.set()
    return check


# --------------------------------------------------------------------------- #
# Pose <-> SE(3) helpers (pure math, no hardware needed)
# --------------------------------------------------------------------------- #
def pose_to_SE3(pose):
    """[x,y,z,tx,ty,tz] (m, deg) -> 4x4 homogeneous transform (SE(3))."""
    T = np.eye(4)
    T[0:3, 0:3] = R.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
    T[0:3, 3] = pose[0:3]
    return T


def SE3_to_pose(T):
    """4x4 homogeneous transform -> [x,y,z,tx,ty,tz] (m, deg)."""
    pos = T[0:3, 3]
    euler = R.from_matrix(T[0:3, 0:3]).as_euler("xyz", degrees=True)
    return [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]]


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #
class KinovaArm:
    """High-level wrapper around a Kinova Gen3 over the Kortex API.

    Parameters
    ----------
    ip : str
        Arm IP address (default ``192.168.1.10``).
    username, password : str
        Kortex session credentials (default ``admin`` / ``admin``).

    Use as a context manager so the session is opened and closed cleanly::

        with KinovaArm() as arm:
            arm.move_home()
    """

    def __init__(self, ip="192.168.1.10", username="admin", password="admin"):
        self._args = _ConnArgs(ip, username, password)
        self._conn = None
        self._router = None
        self.base = None          # BaseClient — actions, twists, notifications
        self.base_cyclic = None   # BaseCyclicClient — high-rate feedback

    # -- connection lifecycle ------------------------------------------------ #
    def connect(self):
        """Open a TCP session and create the Base / BaseCyclic services."""
        self._conn = utilities.DeviceConnection.createTcpConnection(self._args)
        self._router = self._conn.__enter__()
        self.base = BaseClient(self._router)
        self.base_cyclic = BaseCyclicClient(self._router)
        self._set_single_level_servoing()
        return self

    def close(self):
        """Stop any motion and close the session."""
        try:
            if self.base is not None:
                self.base.Stop()
        except Exception:
            pass
        if self._conn is not None:
            try:
                self._conn.__exit__(None, None, None)
            finally:
                self._conn = None
                self._router = None
                self.base = None
                self.base_cyclic = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _set_single_level_servoing(self):
        """Put the arm in high-level (single-level) servoing mode for actions."""
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)

    # -- feedback ------------------------------------------------------------ #
    def get_pose(self):
        """Current tool pose ``[x, y, z, theta_x, theta_y, theta_z]`` (m, deg)."""
        fb = self.base_cyclic.RefreshFeedback()
        return [
            fb.base.tool_pose_x,
            fb.base.tool_pose_y,
            fb.base.tool_pose_z,
            fb.base.tool_pose_theta_x,
            fb.base.tool_pose_theta_y,
            fb.base.tool_pose_theta_z,
        ]

    def get_joint_angles(self):
        """Current actuator angles in degrees (length = actuator count)."""
        fb = self.base_cyclic.RefreshFeedback()
        return [act.position for act in fb.actuators]

    def get_pose_SE3(self):
        """Current tool pose as a 4x4 SE(3) matrix in the base frame."""
        return pose_to_SE3(self.get_pose())

    # -- blocking action helper --------------------------------------------- #
    def _execute_action_and_wait(self, action, timeout=TIMEOUT_DURATION):
        """Execute a Base action and block until END/ABORT or timeout."""
        e = threading.Event()
        handle = self.base.OnNotificationActionTopic(
            _check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.base.ExecuteAction(action)
        finished = e.wait(timeout)
        self.base.Unsubscribe(handle)
        return finished

    # -- named / built-in positions ----------------------------------------- #
    def move_home(self, timeout=TIMEOUT_DURATION):
        """Move to the arm's built-in 'Home' position (blocking)."""
        self._set_single_level_servoing()
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        actions = self.base.ReadAllActions(action_type)
        handle = None
        for action in actions.action_list:
            if action.name == "Home":
                handle = action.handle
        if handle is None:
            raise RuntimeError("No 'Home' action found on the arm.")

        e = threading.Event()
        nh = self.base.OnNotificationActionTopic(
            _check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.base.ExecuteActionFromReference(handle)
        finished = e.wait(timeout)
        self.base.Unsubscribe(nh)
        return finished

    # -- Cartesian (pose) control -------------------------------------------- #
    def move_to_pose(self, pose, wait=True, timeout=TIMEOUT_DURATION, name="move"):
        """Move the tool to an absolute Cartesian *pose* in the base frame.

        Parameters
        ----------
        pose : sequence of 6 floats
            ``[x, y, z, theta_x, theta_y, theta_z]`` — meters and degrees.
        wait : bool
            If True (default), block until the motion finishes.
        """
        action = Base_pb2.Action()
        action.name = name
        action.application_data = ""
        cp = action.reach_pose.target_pose
        cp.x, cp.y, cp.z = float(pose[0]), float(pose[1]), float(pose[2])
        cp.theta_x = float(pose[3])
        cp.theta_y = float(pose[4])
        cp.theta_z = float(pose[5])

        if wait:
            return self._execute_action_and_wait(action, timeout)
        self.base.ExecuteAction(action)
        return None

    def move_delta(self, dx=0.0, dy=0.0, dz=0.0,
                   dtheta_x=0.0, dtheta_y=0.0, dtheta_z=0.0,
                   wait=True, timeout=TIMEOUT_DURATION):
        """Move RELATIVE to the current pose, in the BASE frame.

        Translation deltas in meters, rotation deltas in degrees. This adds the
        deltas to the current tool pose component-wise (base-frame offset).
        """
        p = self.get_pose()
        target = [
            p[0] + dx, p[1] + dy, p[2] + dz,
            p[3] + dtheta_x, p[4] + dtheta_y, p[5] + dtheta_z,
        ]
        return self.move_to_pose(target, wait=wait, timeout=timeout, name="delta")

    def move_tool_frame_delta(self, dx=0.0, dy=0.0, dz=0.0,
                              dtheta_x=0.0, dtheta_y=0.0, dtheta_z=0.0,
                              wait=True, timeout=TIMEOUT_DURATION):
        """Move RELATIVE to the current pose, expressed in the TOOL (body) frame.

        The offset is applied as ``T_target = T_current @ T_delta`` so that, e.g.,
        ``dz`` moves along the tool's own approach axis regardless of orientation.
        """
        T_cur = self.get_pose_SE3()
        T_delta = np.eye(4)
        T_delta[0:3, 3] = [dx, dy, dz]
        if abs(dtheta_x) + abs(dtheta_y) + abs(dtheta_z) > 1e-12:
            T_delta[0:3, 0:3] = R.from_euler(
                "xyz", [dtheta_x, dtheta_y, dtheta_z], degrees=True
            ).as_matrix()
        target = SE3_to_pose(T_cur @ T_delta)
        return self.move_to_pose(target, wait=wait, timeout=timeout, name="tool_delta")

    # -- joint control ------------------------------------------------------- #
    def move_joints(self, joint_angles_deg, wait=True, timeout=TIMEOUT_DURATION):
        """Move each actuator to an absolute angle (degrees).

        ``joint_angles_deg`` length must equal the arm's actuator count
        (7 for a Gen3).
        """
        action = Base_pb2.Action()
        action.name = "joint_move"
        action.application_data = ""
        for joint_id, angle in enumerate(joint_angles_deg):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = joint_id
            ja.value = float(angle)
        if wait:
            return self._execute_action_and_wait(action, timeout)
        self.base.ExecuteAction(action)
        return None

    # -- velocity (twist) control -------------------------------------------- #
    def send_twist(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0),
                   frame="base"):
        """Command a Cartesian velocity (twist). NON-blocking and CONTINUOUS.

        Parameters
        ----------
        linear : [vx, vy, vz]   in m/s
        angular : [wx, wy, wz]  in deg/s
        frame : 'base' or 'tool'
            Reference frame for the twist.

        The arm keeps moving at this velocity until you send another twist or
        call :meth:`stop`. Always pair a twist with a stop.
        """
        ref = (Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL if frame == "tool"
               else Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE)
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = ref
        cmd.twist.linear_x = float(linear[0])
        cmd.twist.linear_y = float(linear[1])
        cmd.twist.linear_z = float(linear[2])
        cmd.twist.angular_x = float(angular[0])
        cmd.twist.angular_y = float(angular[1])
        cmd.twist.angular_z = float(angular[2])
        cmd.duration = 0
        self.base.SendTwistCommand(cmd)

    def stop(self):
        """Stop all motion (zeroes any active twist / aborts the current action)."""
        self.base.Stop()

    def smooth_move(self, axis, speed, duration_s=5.0, ramp_s=1.0,
                    rate_hz=40.0, frame="tool"):
        """Move at constant velocity along ONE axis with a cosine ramp.

        Parameters
        ----------
        axis : 'x','y','z','rx','ry','rz'
        speed : m/s for translation, deg/s for rotation
        duration_s : total duration including ramp-up and ramp-down
        ramp_s : cosine ramp time at each end
        frame : 'base' or 'tool'

        Returns the pose the arm started from (useful for returning later).
        Blocks for ``duration_s``; sends a zero twist at the end.
        """
        import time
        start_pose = self.get_pose()
        dt = 1.0 / rate_hz
        t0 = time.time()
        try:
            while True:
                loop_start = time.time()
                t = loop_start - t0
                if t >= duration_s:
                    break
                if t < ramp_s:
                    env = 0.5 * (1.0 - math.cos(math.pi * t / ramp_s))
                elif t > duration_s - ramp_s:
                    t_end = t - (duration_s - ramp_s)
                    env = 0.5 * (1.0 + math.cos(math.pi * t_end / ramp_s))
                else:
                    env = 1.0
                v = speed * env
                lin = [0.0, 0.0, 0.0]
                ang = [0.0, 0.0, 0.0]
                idx = {"x": 0, "y": 1, "z": 2, "rx": 0, "ry": 1, "rz": 2}[axis]
                if axis.startswith("r"):
                    ang[idx] = v
                else:
                    lin[idx] = v
                self.send_twist(lin, ang, frame=frame)
                sleep_remaining = dt - (time.time() - loop_start)
                if sleep_remaining > 0:
                    time.sleep(sleep_remaining)
        finally:
            self.send_twist([0, 0, 0], [0, 0, 0], frame=frame)
        return start_pose


if __name__ == "__main__":
    # Minimal smoke test: connect and print the current pose.
    import argparse

    parser = argparse.ArgumentParser(description="Kinova driver smoke test")
    parser.add_argument("--ip", default="192.168.1.10")
    parser.add_argument("-u", "--username", default="admin")
    parser.add_argument("-p", "--password", default="admin")
    cli = parser.parse_args()

    with KinovaArm(ip=cli.ip, username=cli.username, password=cli.password) as arm:
        pose = arm.get_pose()
        print("Connected. Current tool pose [x,y,z (m), tx,ty,tz (deg)]:")
        print("  ", [round(v, 4) for v in pose])
        print("Joint angles (deg):", [round(v, 2) for v in arm.get_joint_angles()])
