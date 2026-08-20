"""A Kinova Gen3 as MJCF, on its own, at whatever base pose you name.

EXTRACTED, not written: every constant and every element below came out of
``C:/RUNZE_SRC/RS485_VEMA/UMArm_COLLAB/collab_scene.py`` — ``_add_kinova_assets``
(:660), ``_add_kinova_defaults`` (:667), ``_add_kinova`` (:695) and
``_add_kinova_actuators`` (:889), with ``KINOVA_CHAIN`` (:362),
``KINOVA_BASE_INERTIAL`` (:389), ``KINOVA_MESHES`` (:397), ``STAND_HALF`` (:82)
and the gain / armature / damping block (:147-166).  That module builds a
two-robot bench scene by merging into ``UMArm_SIM.mjcf_generator``'s XML; this
one carries the Gen3 half alone, so a workspace that has no UMArm twin yet can
still put the arm on screen.  The collision pad, joint covers, contact pairs,
force/torque sensors and strike targets were left behind: they are bench
furniture, not the robot.

Two entry points, and they are the same code:

* :func:`build_kinova_xml` returns a complete, standalone MJCF string — a
  skybox, a ground plane, and one Gen3 on a stand.
* :func:`attach_kinova` grafts the same robot into an ``ElementTree`` root that
  somebody else built, which is what a multi-arm room scene wants.  It creates
  ``<asset>``, ``<default>``, ``<contact>`` and ``<actuator>`` only if the host
  has none, so a merge never duplicates a section.

The base hangs off a MuJoCo **mocap body** named ``kinova_mount``.  That choice
is inherited and deliberate (``UMArm_COLLAB/base_poses.py:1-11``): a mocap body
is literally the thing MuJoCo provides for "a pose an external tracker writes",
it accepts a full six degrees of freedom, and writing it costs one array
assignment per frame.  Nothing but a base-pose source should ever move it.

    C:/Users/zuorunze/AppData/Local/Programs/Python/Python313/python.exe \\
        UMArm_KINOVA/kinova_scene.py --self-test

builds the model, runs ``mj_forward`` and prints the tool site's world position.
It opens no socket and contacts no arm.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# The same bootstrap the rest of the package uses: the workspace root is this
# file's grandparent, and putting it on the path is what makes
# ``UMArm_KINOVA.<module>`` importable when the file is run as a script.
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from UMArm_KINOVA import kinova_kinematics as KK      # noqa: E402

#: The eight Gen3 STLs plus their BSD-3 licence, copied from
#: ``UMArm_COLLAB/assets/kinova_gen3/`` (Kinova, via mujoco_menagerie).
ASSET_DIR = os.path.join(_HERE, "assets")
KINOVA_MESH_DIR = os.path.join(ASSET_DIR, "kinova_gen3")


# ---------------------------------------------------------------------------
# Inlined from UMArm_COLLAB/mount_transforms.py so this module has no
# UMArm_COLLAB dependency.  Four names, verbatim: MM (:156),
# KINOVA_TOOL_FLANGE_Z_M (:612), rpy_to_quat (:708), quat_to_mat (:727).
# ---------------------------------------------------------------------------

MM = 1.0e-3

#: Distance from the Gen3 ``bracelet_link`` origin to its tool flange, metres.
#: mujoco_menagerie puts ``pinch_site`` at ``pos="0 0 -0.061525"``; the bracelet
#: mesh bottoms out at z = -0.0644, so the flange face is the site, not the mesh.
KINOVA_TOOL_FLANGE_Z_M = -0.061525


def rpy_to_quat(rpy_deg) -> np.ndarray:
    """Extrinsic-xyz Euler degrees -> MuJoCo scalar-first quaternion.

    Written out rather than taken from scipy so the MJCF generator has no
    optional dependency: the scene has to build on a clone with nothing but
    numpy and mujoco installed.
    """
    r, p, y = (np.deg2rad(float(v)) for v in rpy_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)


def quat_to_mat(q) -> np.ndarray:
    """MuJoCo scalar-first quaternion -> 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Placement defaults
# ---------------------------------------------------------------------------

#: Where the Gen3's base sits in mocap-world coordinates before the cameras say
#: otherwise.  The bench's own number; on the rig a ``BasePoseSource`` overwrites
#: ``kinova_mount`` every tick and this is only the compile-time seed.
DEFAULT_KINOVA_BASE_POS = (0.220, 0.360, 0.300)
#: Level, facing the arm.  The example pitched its base 45 deg to reach up; ours
#: reaches across, so the tilt is zero by default and stays a knob.
DEFAULT_KINOVA_BASE_RPY_DEG = (0.0, 0.0, 0.0)

#: The stand the Gen3 bolts to, drawn under ``kinova_mount``.  Purely scenery
#: plus a floor stop; the mocap mount is what defines the base pose.
STAND_HALF = (0.20, 0.20, 0.15)

#: **Gravity compensation on the Gen3, 0..1.**  A real Gen3 compensates gravity
#: inside its own joint controller and holds a commanded pose; mujoco_menagerie's
#: ``position`` actuators are plain PD and do not, so the modelled arm SAGGED
#: under the 0.6 kg pad -- measured 0.0077 rad at ``joint_2``, which put the pad
#: 4 mm low, and every task-space nudge then re-measured the sagged pose and
#: dragged it 4 mm lower again.  Four clicks of "+X" lost 22 mm of height.
#: Setting this to 1 makes the modelled robot hold its pose the way the real one
#: does; drop it to 0 to see the uncompensated behaviour.
KINOVA_GRAVCOMP = 1.0

# ---------------------------------------------------------------------------
# How stiff the Gen3 is against being pushed
# ---------------------------------------------------------------------------
#
# MEASURED 2026-08-19, and it is why these constants exist: on a single
# ``cover2`` strike at 35 psi the pad was displaced **24.0 mm** while the arm
# leaned on it, joint_2 giving 1.82 deg and joint_6 1.51 deg.  The operator's
# report was "the collision force lets the Kinova arm move, and it should not".
#
# The cause is not the impact and it is not inertia.  A strike presses for
# 2.2 s, so the give is QUASI-STATIC: it is set by the position servo's
# proportional gain, and mujoco_menagerie's kp = 2000 / 500 N.m/rad is a
# CONTROLLER bandwidth chosen for simulation stability, not the mechanical
# stiffness of a harmonic-drive joint, which is 1e4-1e5 N.m/rad.  A real Gen3
# holding 63 N.m at joint_2 deflects a few tenths of a degree, not 1.8.
#
# So the servo gains are raised to :data:`KINOVA_KP_LARGE` / :data:`KINOVA_KP_SMALL`,
# which is 12.5x the menagerie's and is a CATALOGUE number rather than a knob:
# Harmonic Drive's CSD/CSF series lists a K3 torsional stiffness near
# 2.5e4 N.m/rad for a size-20 unit and ~6e3 for a size-14, which is what the
# Gen3's large and small joints are built around.  ``kv`` rises with it so the
# joints stay near the damping ratio the menagerie chose.
#
# **The force limits are NOT touched**: +-105 / +-52 N.m stay, and an over-large
# load still folds the arm, which is the honest failure.  Worth knowing that
# those numbers are the menagerie's rather than Kinova's: ``ros_kortex``'s own
# URDF declares effort limits of 39 and 9 N.m, 2.7x and 5.8x smaller.  They
# describe what the MOTOR can produce, while what resists an external push on a
# harmonic-drive joint is the gearbox; MuJoCo's ``position`` actuator has one
# number for both, and this bench models the reaction rather than the motor.
# With the joints this stiff that ceiling becomes load-bearing for the first
# time -- the Gen3 now reaches +-105 N.m resisting a ~190 N strike instead of
# quietly yielding.
#
# Measured, same trial, pad displacement during the strike and the peak contact
# force that goes with it (gain scale against the menagerie's):
#
#     scale    1     2     5    10    12.5   20    50
#     mm    24.04 14.02  6.35  3.32   2.7   1.70  0.69
#     N     121.8 148.8 173.1 183.6  ~186  189.3 192.9
#
# THE FORCE GOES UP, and that is the second half of the finding: the 24 mm of
# give was absorbing the impact, so the bench had been UNDER-reporting the
# collision by up to a third.
#
# Two levers that do NOT work were measured and rejected.  ARMATURE alone: a
# 250 N / 5 ms impulse moves 2.76 -> 2.65 mm at 0.3/0.1 and needs ~10 kg.m^2 --
# thirty times any defensible reflected inertia -- to reach 1 mm, and on the
# real strike it changes the tracking error by 1 %.  Inertia is invisible in a
# steady state, and a steady state is what a 2.2 s strike is.  DAMPING: 250
# N.m.s/rad does reach 1 mm on the impulse, and makes the real strike WORSE
# (peak tracking error 34 -> 83 mm) because it saturates the 105 N.m ceiling at
# 0.42 rad/s, well under the robot's own 1.4 rad/s velocity limit.
KINOVA_KP_LARGE = 25000.0
KINOVA_KP_SMALL = 6250.0
KINOVA_KV_LARGE = 360.0
KINOVA_KV_SMALL = 110.0

#: Reflected rotor inertia and viscous loss of the drive train, per joint.
#: ``J_reflected = N^2 (J_rotor + J_wavegen)``; for a Gen3-class harmonic drive
#: at N = 100-160 on a frameless BLDC sized for the joint's own effort limit
#: that is 0.15-0.77 kg.m^2 on the large joints and 0.03-0.15 on the wrist, and
#: these are mid-bracket.  The joints carried ZERO before -- mujoco_menagerie's
#: Gen3 declares none, where its own Panda and UR5e both ship ``armature=0.1``.
#:
#: They are here because they are real, NOT because they fixed the operator's
#: problem; see the block above for what they were measured to be worth.  What
#: they do buy is 19 % off the peak actuator torque during an impulsive contact
#: and a little numerical headroom at the higher gain.
KINOVA_ARMATURE_LARGE = 0.3
KINOVA_ARMATURE_SMALL = 0.1
KINOVA_JOINT_DAMPING = 5.0
KINOVA_JOINT_DAMPING_SMALL = 2.0

#: Kinova joint chain: ``(body, pos, quat, joint, mesh)``.  Verbatim from
#: mujoco_menagerie's Gen3 (7-DOF, no gripper) as the example repo carries it.
KINOVA_CHAIN = (
    ("shoulder_link", (0, 0, 0.15643), (0, 1, 0, 0), "joint_1", "shoulder_link",
     (-2.3e-05, -0.010364, -0.07336), (0.707051, 0.0451246, -0.0453544, 0.704263),
     1.3773, (0.00488868, 0.00457, 0.00135132)),
    ("half_arm_1_link", (0, 0.005375, -0.12838), (1, 1, 0, 0), "joint_2", "half_arm_1_link",
     (-4.4e-05, -0.09958, -0.013278), (0.482348, 0.516286, -0.516862, 0.483366),
     1.1636, (0.0113017, 0.011088, 0.00102532)),
    ("half_arm_2_link", (0, -0.21038, -0.006375), (1, -1, 0, 0), "joint_3", "half_arm_2_link",
     (-4.4e-05, -0.006641, -0.117892), (0.706144, 0.0213722, -0.0209128, 0.707437),
     1.1636, (0.0111633, 0.010932, 0.00100671)),
    ("forearm_link", (0, 0.006375, -0.21038), (1, 1, 0, 0), "joint_4", "forearm_link",
     (-1.8e-05, -0.075478, -0.015006), (0.483678, 0.515961, -0.515859, 0.483455),
     0.9302, (0.00834839, 0.008147, 0.000598606)),
    ("spherical_wrist_1_link", (0, -0.20843, -0.006375), (1, -1, 0, 0), "joint_5",
     "spherical_wrist_1_link",
     (1e-06, -0.009432, -0.063883), (0.703558, 0.0707492, -0.0707492, 0.703558),
     0.6781, (0.00165901, 0.001596, 0.000346988)),
    ("spherical_wrist_2_link", (0, 0.00017505, -0.10593), (1, 1, 0, 0), "joint_6",
     "spherical_wrist_2_link",
     (1e-06, -0.045483, -0.00965), (0.44426, 0.550121, -0.550121, 0.44426),
     0.6781, (0.00170087, 0.001641, 0.00035013)),
    ("bracelet_link", (0, -0.10593, -0.00017505), (1, -1, 0, 0), "joint_7",
     "bracelet_with_vision_link",
     (0.000281, 0.011402, -0.029798), (0.394358, 0.596779, -0.577293, 0.393789),
     0.5, (0.000657336, 0.000587019, 0.000320645)),
)

KINOVA_BASE_INERTIAL = ((-0.000648, -0.000166, 0.084487),
                        (0.999294, 0.00139618, -0.0118387, 0.035636),
                        1.697, (0.00462407, 0.00449437, 0.00207755))

#: The eight meshes the chain above uses.  mujoco_menagerie also ships
#: ``bracelet_no_vision_link.stl`` for the camera-less wrist; it is not carried
#: here because nothing references it — copy it in and swap the last entry if
#: the lab's Gen3 turns out to be that variant.
KINOVA_MESHES = ("base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link",
                 "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link",
                 "bracelet_with_vision_link")

#: The mocap body the base hangs off, and the site at the tool flange.  Named
#: constants because a base-pose writer and an IK caller both look them up.
MOUNT_BODY = "kinova_mount"
BASE_BODY = "kinova_base"
TOOL_SITE = "kinova_tool"


# ---------------------------------------------------------------------------
# Small XML helpers (collab_scene.py:406-440, verbatim)
# ---------------------------------------------------------------------------

def _fmt(vals) -> str:
    return " ".join(f"{float(v):.10g}" for v in np.atleast_1d(vals))


def _sub(parent, tag, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        if v is None:
            continue
        key = k.rstrip("_").replace("__", ":")
        el.set(key, v if isinstance(v, str) else _fmt(v))
    return el


def _mesh_file(directory: str, name: str) -> str:
    """Absolute, forward-slashed path.

    Absolute on purpose: the scene is normally compiled straight from a string,
    and MuJoCo resolves a relative ``meshdir`` against the process's working
    directory — so a GUI launched from anywhere but the repo root would fail to
    find its own meshes.
    """
    return os.path.join(directory, name).replace("\\", "/")


def _section(root, tag: str) -> ET.Element:
    """The host's ``<tag>``, created at the end of *root* if it has none."""
    found = root.find(tag)
    return found if found is not None else ET.SubElement(root, tag)


# ---------------------------------------------------------------------------
# The four builders, extracted from collab_scene.py
# ---------------------------------------------------------------------------

def _add_kinova_assets(asset) -> None:
    kdir = KINOVA_MESH_DIR
    for name in KINOVA_MESHES:
        _sub(asset, "mesh", name=f"kin_{name}",
             file=_mesh_file(kdir, f"{name}.stl"))


def _add_kinova_defaults(defaults) -> None:
    """The menagerie's Gen3 classes, verbatim in spirit.

    The FORCE RANGES are the menagerie's and the manufacturer's: the large joints
    carry 105 N.m, the wrist 52 N.m.  A pad held against a 400 N impact is
    exactly where those limits matter, so they are not softened.

    The GAINS are :data:`KINOVA_KP_LARGE` / :data:`KINOVA_KP_SMALL`, and the
    joints carry :data:`KINOVA_ARMATURE_LARGE` and :data:`KINOVA_JOINT_DAMPING`
    (and their small-joint siblings), which they previously did not carry at all
    -- see those constants for the measurement that put them there.
    """
    kin = _sub(defaults, "default", class_="kinova")
    _sub(kin, "site", size="0.001", rgba="0.5 0.5 0.5 0.3", group="4")
    vis = _sub(kin, "default", class_="kin_visual")
    _sub(vis, "geom", type="mesh", contype="0", conaffinity="0", group="2",
         rgba="0.75294 0.75294 0.75294 1")
    col = _sub(kin, "default", class_="kin_collision")
    _sub(col, "geom", type="mesh", group="3", contype="1", conaffinity="1",
         rgba="0.9 0.4 0.2 0.25")
    big = _sub(kin, "default", class_="large_actuator")
    _sub(big, "position", kp=(KINOVA_KP_LARGE,), kv=(KINOVA_KV_LARGE,),
         forcerange="-105 105")
    small = _sub(kin, "default", class_="small_actuator")
    _sub(small, "position", kp=(KINOVA_KP_SMALL,), kv=(KINOVA_KV_SMALL,),
         forcerange="-52 52")


def _add_kinova(worldbody, base_pos, base_rpy_deg, with_stand: bool = True) -> ET.Element:
    """The Gen3 on its stand, under a mocap mount.  Returns ``bracelet_link``."""
    mount = _sub(worldbody, "body", name=MOUNT_BODY, mocap="true",
                 pos=base_pos, quat=rpy_to_quat(base_rpy_deg))
    if with_stand:
        # the stand: scenery, and a floor stop so the base cannot be driven
        # under it
        _sub(mount, "geom", name="kinova_stand", type="box",
             pos=(0.0, 0.0, -STAND_HALF[2]), size=STAND_HALF,
             rgba="0.25 0.25 0.28 1", contype="1", conaffinity="1", mass="0")
        _sub(mount, "geom", name="kinova_stand_top", type="box",
             pos=(0.0, 0.0, -0.006), size=(STAND_HALF[0] + 0.02,
                                           STAND_HALF[1] + 0.02, 0.006),
             rgba="0.55 0.55 0.6 1", contype="0", conaffinity="0", mass="0")

    base = _sub(mount, "body", name=BASE_BODY, pos="0 0 0",
                childclass="kinova", gravcomp=(KINOVA_GRAVCOMP,))
    ip, iq, m, di = KINOVA_BASE_INERTIAL
    _sub(base, "inertial", pos=ip, quat=iq, mass=(m,), diaginertia=di)
    _sub(base, "geom", class_="kin_visual", mesh="kin_base_link")
    _sub(base, "geom", class_="kin_collision", mesh="kin_base_link")

    parent = base
    for (body, pos, quat, joint, mesh, ip, iq, m, di) in KINOVA_CHAIN:
        b = _sub(parent, "body", name=body, pos=pos, quat=quat,
                 gravcomp=(KINOVA_GRAVCOMP,))
        _sub(b, "inertial", pos=ip, quat=iq, mass=(m,), diaginertia=di)
        idx = KK.JOINT_NAMES.index(joint)
        rng = KK.JOINT_LIMITS.get(idx)
        # Armature and damping are written on the JOINT rather than into the
        # ``large_actuator``/``small_actuator`` classes, because those classes
        # are worn by the ``<position>`` elements and a joint that never names
        # one would inherit nothing from them.
        big = idx < 4
        _sub(b, "joint", name=joint,
             armature=(KINOVA_ARMATURE_LARGE if big
                       else KINOVA_ARMATURE_SMALL,),
             damping=(KINOVA_JOINT_DAMPING if big
                      else KINOVA_JOINT_DAMPING_SMALL,),
             frictionloss="0",
             range=(rng if rng is not None else None),
             limited=("true" if rng is not None else "false"))
        _sub(b, "geom", class_="kin_visual", mesh=f"kin_{mesh}")
        _sub(b, "geom", class_="kin_collision", mesh=f"kin_{mesh}")
        parent = b

    _sub(parent, "site", name=TOOL_SITE,
         pos=(0.0, 0.0, KINOVA_TOOL_FLANGE_Z_M), quat="0 1 0 0")
    return parent


def _add_kinova_contacts(root) -> None:
    """The one contact exclusion the Gen3 needs, and nothing else.

    ``collab_scene._add_contacts`` (:941) opens its ``<contact>`` block with
    ``<exclude body1="kinova_base" body2="shoulder_link"/>`` before it gets to
    any bench-specific ``<pair>``, so the exclusion belongs to the robot rather
    than to the bench and travels with it here.

    MEASURED after the extraction, because the reason was not written down
    upstream: the two menagerie collision meshes interpenetrate by **12.0 mm**
    at the ``joint_1`` origin, at every joint angle, since the shoulder's shell
    wraps the base's collar.  MuJoCo's parent-child filter does not remove the
    pair — ``kinova_base`` is welded to the mocap mount rather than jointed to
    it — so without this line ``mj_forward`` reports four spurious contacts in
    every pose, and :func:`kinova_kinematics.solve_pose_ik` reads them as a
    self-collision and refuses every target it is ever given.
    """
    _sub(_section(root, "contact"), "exclude",
         body1=BASE_BODY, body2="shoulder_link")


def _add_kinova_actuators(root) -> None:
    act = _section(root, "actuator")
    for i, name in enumerate(KK.JOINT_NAMES):
        cls = "large_actuator" if i < 4 else "small_actuator"
        rng = KK.CTRL_LIMITS.get(i)
        _sub(act, "position", class_=cls, name=name, joint=name,
             ctrlrange=(rng if rng is not None else None))


# ---------------------------------------------------------------------------
# The two public entry points
# ---------------------------------------------------------------------------

#: A bare host scene: radian angles, a light, a skybox and a ground plane, and
#: the empty sections :func:`attach_kinova` fills.  The compiler line is the one
#: piece that is NOT optional — :data:`KK.JOINT_LIMITS` is in radians, and the
#: MJCF default is degrees, so a host that forgets ``angle="radian"`` builds a
#: robot whose joint 2 stop is at 2.24 DEGREES.
STANDALONE_TEMPLATE = """<mujoco model="kinova_gen3_scene">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast" iterations="100"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <default/>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
      contype="1" conaffinity="1"/>
  </worldbody>

  <actuator/>
</mujoco>
"""


def attach_kinova(root,
                  base_pos=DEFAULT_KINOVA_BASE_POS,
                  base_rpy_deg=DEFAULT_KINOVA_BASE_RPY_DEG,
                  with_stand: bool = True) -> ET.Element:
    """Graft a Gen3 into an existing MJCF tree.  Returns ``bracelet_link``.

    *root* is the ``<mujoco>`` element of a scene somebody else authored.  The
    five sections the robot needs — ``asset``, ``default``, ``worldbody``,
    ``contact``, ``actuator`` — are reused when the host already has them and
    created when it does not, so merging into a one-arm scene adds a robot
    rather than a second copy of every section.

    The caller keeps one obligation the merge cannot check for it: the host's
    ``<compiler>`` must say ``angle="radian"``.  See
    :data:`STANDALONE_TEMPLATE`.
    """
    _add_kinova_assets(_section(root, "asset"))
    _add_kinova_defaults(_section(root, "default"))
    bracelet = _add_kinova(_section(root, "worldbody"), base_pos, base_rpy_deg,
                           with_stand=with_stand)
    _add_kinova_contacts(root)
    _add_kinova_actuators(root)
    return bracelet


def build_kinova_xml(base_pos=DEFAULT_KINOVA_BASE_POS,
                     base_rpy_deg=DEFAULT_KINOVA_BASE_RPY_DEG,
                     with_stand: bool = True) -> str:
    """A complete standalone MJCF string: one Gen3, a floor, and nothing else.

    Feed it straight to ``mujoco.MjModel.from_xml_string``.  Mesh paths are
    absolute, so the working directory does not matter.
    """
    if not os.path.isdir(KINOVA_MESH_DIR):
        raise FileNotFoundError(
            KINOVA_MESH_DIR + " is missing — the eight Gen3 STLs travel with "
            "this module and the scene cannot be compiled without them")
    root = ET.fromstring(STANDALONE_TEMPLATE)
    attach_kinova(root, base_pos, base_rpy_deg, with_stand=with_stand)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# Self-test — builds the model, runs mj_forward, touches no hardware
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    import mujoco

    xml = build_kinova_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # HOME_Q, not zeros: kinova_kinematics picked it because every joint sits
    # comfortably mid-range there, so the printed tool position is a pose the
    # arm can actually hold rather than one on a joint stop.
    data.qpos[:7] = KK.HOME_Q
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    # ncon is checked, not just printed: the base/shoulder exclusion is easy to
    # lose in a merge and its absence is silent — four contacts that penetrate
    # 12 mm and make every IK target look unreachable.
    ok = (model.nq == 7 and model.nu == 7 and sid >= 0
          and model.body_mocapid[mid] >= 0 and data.ncon == 0)
    if verbose:
        print(f"nq={model.nq} nv={model.nv} nu={model.nu} "
              f"nbody={model.nbody} nmesh={model.nmesh} ncon={data.ncon}")
        print(f"{TOOL_SITE} at {np.round(data.site_xpos[sid], 6).tolist()}")
        print(f"{MOUNT_BODY} mocapid={int(model.body_mocapid[mid])}")
        print("OK" if ok else "FAILED")
    return bool(ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="build the model, run mj_forward, print and exit")
    ap.add_argument("--dump", metavar="PATH",
                    help="write the generated MJCF to PATH")
    ap.add_argument("--pos", nargs=3, type=float,
                    default=list(DEFAULT_KINOVA_BASE_POS))
    ap.add_argument("--rpy-deg", nargs=3, type=float,
                    default=list(DEFAULT_KINOVA_BASE_RPY_DEG))
    args = ap.parse_args(argv)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            fh.write(build_kinova_xml(tuple(args.pos), tuple(args.rpy_deg)))
        print("wrote " + args.dump)
    if args.self_test or not args.dump:
        return 0 if self_test() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
