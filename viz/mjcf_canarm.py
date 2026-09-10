"""The CAN arm's **display** model, and the room scene it sits in.

WHAT THIS IS NOT.  It is not a twin.  ``digital_twin/mjcf_generator.py`` emits
24 muscle tendons, 24 motors, per-segment masses and the fitted dissipation
tunables, because a controller has to be unable to tell its output from the
metal's.  Everything in that list is absent here, on purpose:

* **no actuators and no tendons**, so nothing can be commanded through this
  model by accident;
* **no contacts** — every geom is ``contype="0" conaffinity="0"``, including the
  floor — so two arms drawn overlapping is a drawing, not a collision;
* **no fitted dynamics**.  The masses, damping and armature exist only so MuJoCo
  will compile the model and ``mj_forward`` will place it.

The consequence worth stating plainly: this model answers "where are the links,
given ``q``" and nothing else, and it stays the cheap thing a viewer compiles in
a spawned process.

THE CAN ARM IS DRAWN AS THE PROMAX IT IS, FROM THE TWIN'S OWN DRAWING.  Since
2026-09-10 each CAN-arm segment is the structure of Fig. 1C of the arm's paper
(``2606.29731v1.pdf``): a centre rod carrying two bearing hubs, two upright and
two upside-down Y supports 45 deg apart, eight McKibben sleeves hanging from the
Y tips, and flat u-joint disks on the bracket side of each joint.  The parts are
not re-authored here: :func:`build_arm_xml` asks
``digital_twin.mjcf_generator.promax_segment_elements`` for them, the same call
the physics model makes, so the room viewer and the twin cannot drift apart
about where a sleeve or a bearing is.  What the display leaves out is the part
that cannot be drawn with fixed geoms: a tendon's bearing-to-bracket span
crosses a joint, so only the sleeve-to-bearing stub is drawn.  If that import
fails the arm falls back to the old rod-and-disk drawing and
:data:`LAST_ARM_STYLE_NOTE` says why — a picture of the right joints is still
worth more than an exception.  The RS485 arm keeps the rod-and-disk drawing,
because nothing here describes its hardware.

THE CAN ARM'S PROXIMAL HINGES ARE DECLARED Y-THEN-X.  MuJoCo composes same-body
hinges in declaration order, so the old x-then-y declaration drew fkine's
legacy ``order="xy"``.  The CAN arm's measured assembly is ``order="yx"``
(``UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER``, 2026-08-21), and
``CanArmMarkerMocap`` — the receiver the room viewer builds for a live session —
publishes ``q`` under that order.  So until this change the viewer drew every
live pose of this arm with the wrong composition: checked with ``fkine`` on
2026-09-10, the two orders put the drawn u-joint centres up to 161.4 mm apart
(RMS 20.2 mm over all six centres) across 2000 random poses inside +-30 deg,
and up to 19.9 mm apart on the measured poses of the 90 s held-out validation
recording (95th percentile of each frame's worst centre 11.5 mm, RMS 3.0 mm).
That recording spans 19-28 deg per joint, so the second set is the error an
operator actually watched, and the first is what a larger pose would show.  The mapping from
``q`` to ``qpos`` is by joint NAME (``multi_arm_viewer._address_book``), and
:func:`joint_names` lists names in ``q`` order, so the reordered declaration
needed no index table anywhere.  The RS485 arm keeps ``"xy"``: nobody has
measured its assembly the same way.

CHAIN LENGTHS ARE ARGUMENTS, NOT MODULE CONSTANTS.  The RS485 generator pins
``FITTED_CHAIN_M`` as a module-level tuple and exposes only base pose and three
damping scalars, so a different-sized arm cannot be built without editing the
file.  :func:`build_arm_xml` takes ``chain_m`` instead, which is why one
function draws both arms — the CAN arm from :data:`DEFAULT_CANARM_CHAIN_M` and
the RS485 arm from :data:`DEFAULT_RS485_CHAIN_M`.

THE CAN ARM'S FIVE NUMBERS ARE MEASURED, since 2026-08-21.  They come from
``UMArm_KINEMATICS.canarm_params``, whose ``MEASURED`` flag is ``True``: the
five consecutive u-joint centre distances read off the marker-inferred plate
frames over 68 arm poses, standard deviation 0.03-0.47 mm.  The ProMax parts
take their CAD dimensions (hub heights, ring and bearing radii) from the same
table, with the actuator length following the chain.

EVERY ROBOT HANGS OFF A MOCAP BODY.  ``canarm_mount``, ``rs485_mount``,
``kinova_mount``.  A mocap body is what MuJoCo provides for "a pose an external
tracker writes": six degrees of freedom, written with one array assignment per
frame, and unlike a baked ``pos``/``quat`` it can be moved without recompiling.
See ``viz.base_poses`` for what writes them.

THE OPTIONAL ROBOTS ARE DROPPABLE, INDIVIDUALLY AND TOGETHER.
:func:`build_room_scene` composes the CAN arm first and merges the others in
afterwards; if the Kinova's scene module is missing, or present and unusable, the
room falls back to a small triad on a ``kinova_mount`` and records why in
:data:`LAST_KINOVA_NOTE`.  The CAN arm's XML is byte-identical across all four
flag combinations, which is the property that makes "drop the Gen3" a safe thing
to do mid-session.

WHAT THIS DOES NOT SHOW.  The Y tip radius is estimated from Fig. 1C, not
measured, and the drawing carries the twin's *component* masses, not its fitted
totals; neither changes a single site position ``q`` produces.

Run:  python -m viz.mjcf_canarm --rs485 --kinova --out room.xml
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:                                    # both import styles, as elsewhere here
    from . import transforms as TF
except ImportError:                     # pragma: no cover
    import transforms as TF             # type: ignore[no-redef]

__all__ = [
    "DEFAULT_CANARM_CHAIN_M", "DEFAULT_RS485_CHAIN_M",
    "DEFAULT_CANARM_MOUNT", "DEFAULT_RS485_MOUNT", "DEFAULT_KINOVA_MOUNT",
    "KINOVA_JOINT_NAMES", "LAST_KINOVA_NOTE", "LAST_ARM_STYLE_NOTE",
    "ARM_STYLES", "PROXIMAL_ORDERS", "CANARM_PROXIMAL_ORDER",
    "RS485_PROXIMAL_ORDER",
    "joint_names", "plate_site_names", "robot_joint_names",
    "build_arm_xml", "build_room_scene", "write_room_scene",
]

_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    # Every package in this workspace is a direct child of the root and six
    # modules compute that root as their own parent; keeping the same habit here
    # means `viz` can be imported from a spawned process with any cwd.
    sys.path.insert(0, str(_WS_ROOT))


def _default_chains() -> tuple:
    """``(canarm_chain, rs485_chain)``, from the kinematics package if present.

    Falls back to the RS485 nominal chain for both, rather than refusing: this
    module draws pictures, and a picture of an arm whose lengths are nominal is
    worth strictly more than an exception.  A caller whose *output is a length*
    should be going through ``canarm_params.require_measured()`` instead, which
    refuses on exactly that distinction.
    """
    nominal = (0.218868, 0.047871, 0.194540, 0.047382, 0.184111)
    try:
        from UMArm_KINEMATICS import canarm_params as cp
        from UMArm_KINEMATICS import robot_params as rp
        return tuple(cp.CANARM_PLATE_CHAIN_M), tuple(rp.PLATE_CHAIN_NOMINAL_M)
    except Exception:                                        # pragma: no cover
        return nominal, nominal


#: Consecutive u-joint-centre gaps, metres, proximal to distal:
#: ``(span1, JD2, span2, JD3, span3)``.  Measured on 2026-08-21 — see the
#: module docstring and ``UMArm_KINEMATICS.canarm_params.MEASURED``.
DEFAULT_CANARM_CHAIN_M, DEFAULT_RS485_CHAIN_M = _default_chains()

#: Where each robot is parked when nothing is writing its mount.  These are
#: **arbitrary but non-overlapping**: no one has measured where anything sits in
#: the room, and three robots stacked at the origin is a picture of nothing.  The
#: arms hang base-up, hence the 1.25 m mounts.
DEFAULT_CANARM_MOUNT = ((0.0, 0.0, 1.25), (0.0, 0.0, 0.0))
DEFAULT_RS485_MOUNT = ((0.9, 0.0, 1.25), (0.0, 0.0, 0.0))
DEFAULT_KINOVA_MOUNT = ((0.45, 0.75, 0.35), (0.0, 0.0, 0.0))

#: Plate disc radii, metres: ``(base/proximal, distal)``, for the rod-and-disk
#: drawing only.  Cosmetic — they size the drawn discs and nothing else.  The
#: ProMax drawing takes its ring radii from the parameter table instead.
DEFAULT_PLATE_RADII = (0.055, 0.055)
BASE_PLATE_RADIUS = 0.105

#: The two drawings :func:`build_arm_xml` knows.
ARM_STYLES = ("promax", "simple")

#: fkine's two proximal compositions (``UMArm_KINEMATICS.fkine.PROXIMAL_ORDERS``).
PROXIMAL_ORDERS = ("xy", "yx")

#: The CAN arm's measured proximal composition.  Restated rather than imported
#: from ``UMArm_MOCAP.canarm_frames`` so a spawned viewer does not pay for the
#: marker stack's import; ``viz/self_check.py`` asserts the two agree.
CANARM_PROXIMAL_ORDER = "yx"

#: The RS485 arm's: the legacy default, never re-measured.
RS485_PROXIMAL_ORDER = "xy"

#: Gen3 joint names, as ``mujoco_menagerie`` ships them and as
#: ``UMArm_COLLAB.kinova_kinematics.JOINT_NAMES:36`` re-declares them.  Used only
#: to map the Kinova's seven doubles onto qpos addresses by NAME; a merged scene
#: that names its joints something else is discovered, not assumed (see
#: :func:`robot_joint_names`).
KINOVA_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))

#: Why the last :func:`build_room_scene` call did or did not get a real Gen3.
#: Read it after building; it is the difference between "no Kinova was asked
#: for", "UMArm_KINOVA.kinova_scene is not importable yet" and "it is importable
#: and what it returned would not compile".
LAST_KINOVA_NOTE = "not attempted"

#: ``{prefix: note}`` — which drawing the last :func:`build_arm_xml` call per arm
#: actually produced, and why when it is not the one asked for.
LAST_ARM_STYLE_NOTE: dict = {}

_SQ2 = math.sqrt(2.0) / 2.0

#: Joint range and armature: the RS485 generator's inherited stability choices
#: (``mjcf_generator.py:188-189``), noted there as inherited and repeated here
#: for the same reason — they keep a display model from folding through itself
#: when handed a ``q`` from a bad frame.  Nothing is fitted to them.
JOINT_RANGE_RAD = 0.6981317007977318          # +-40 deg
JOINT_ARMATURE = 0.01

_SEG_COLOURS = {
    "canarm": ("0.20 0.75 0.30 1", "0.20 0.45 0.95 1", "0.70 0.30 0.85 1"),
    "rs485": ("0.55 0.62 0.35 1", "0.35 0.50 0.70 1", "0.60 0.45 0.65 1"),
}


# ---------------------------------------------------------------------------
# Names.  Everything a viewer needs to address a robot inside a merged scene.
# ---------------------------------------------------------------------------

def joint_names(prefix: str) -> tuple:
    """The twelve hinge names of one arm, in **q order**.

    ``uj1_x, uj1_y, uj2_x, uj2_y, ...``, which is exactly the pairing
    ``mocap_to_q`` and ``fkine`` use: ``q[2k]`` and ``q[2k+1]`` are joint
    ``k``'s two angles.  It is *not* necessarily the qpos order: the CAN arm
    declares its proximal ``y`` hinge first (``order="yx"``), so its ``uj1_x``
    lives at the second qpos address.  A consumer that writes ``q[i]`` to the
    address of ``joint_names(prefix)[i]`` — which is what
    ``multi_arm_viewer._address_book`` does — is right under either order.
    """
    return tuple(f"{prefix}uj{k}_{axis}"
                 for k in range(1, 7) for axis in ("x", "y"))


def plate_site_names(prefix: str) -> tuple:
    """The six plate sites of one arm, base to tip.

    Index ``p`` is the same ``p`` the mocap array uses: 0 is the base plate
    (``mocap_constants.IDX_BASE``), 5 is the distal plate of segment 3
    (``IDX_U3_DISTAL``).  Having them as named sites is what lets an offline
    check compare the drawn chain against the measured one without a viewer.
    """
    return tuple(f"{prefix}plate{p}" for p in range(6))


def robot_joint_names(model, robot: str) -> tuple:
    """The joint names *model* actually carries for *robot*, in q order, or ``()``.

    Name lookup rather than index arithmetic, and it is not fussiness: which
    qpos address a robot's block starts at depends on which optional robots were
    included, and — for the CAN arm — the order of the two proximal hinges within
    a joint is not the order of ``q``.  A viewer that assumed either would
    silently draw the wrong pose.
    """
    import mujoco

    wanted = (KINOVA_JOINT_NAMES if robot == "kinova"
              else joint_names(f"{robot}_"))
    out = []
    for name in wanted:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            out.append(name)
    return tuple(out)


# ---------------------------------------------------------------------------
# The arm itself
# ---------------------------------------------------------------------------

def _seg_geometry(chain_m) -> list:
    """``chain_m`` -> per-segment ``(span, jd)``.

    ``chain_m`` is ``(span1, JD2, span2, JD3, span3)``: the five consecutive
    u-joint-centre gaps, which is what ``robot_params.plate_chain_m`` returns and
    what the mocap chain-span gate measures directly.  ``JD1`` is not in it —
    it sits between the world and plate 0, and it is zero on this mechanism.
    """
    c = [float(v) for v in chain_m]
    if len(c) != 5:
        raise ValueError(
            "chain_m must be the five consecutive u-joint-centre gaps "
            "(span1, JD2, span2, JD3, span3); got %d values" % len(c))
    if min(c[0], c[2], c[4]) <= 0.0:
        raise ValueError("segment spans must be positive; got %r" % (c,))
    return [(c[0], 0.0), (c[2], c[1]), (c[4], c[3])]


def _promax_parts(chain_m, prefix: str, cols, rod_radius: float,
                  base_plate_radius: float):
    """``((base_elements, [segment_parts x3]), note)``, or ``(None, why)``.

    Lazy, and forgiving in the same way the Kinova merge is: the parts come from
    ``digital_twin.mjcf_generator``, and a checkout in which that package cannot
    be imported must still get a room with an arm in it.
    """
    try:
        from digital_twin import mjcf_generator as MG
        params = MG.params_from_chain(chain_m)
        geos = MG.segment_geometry(params)
        seats = MG.actuator_seats()
        masses = MG.segment_masses(geos)
        base = MG.promax_base_elements(
            prefix=prefix, ring_radius=geos[0].ja1,
            base_plate_radius=base_plate_radius, base_plate_mass=0.5)
        segs = [MG.promax_segment_elements(
                    geo, [s for s in seats if s.segment == n], n=n,
                    prefix=prefix, masses=masses[n - 1], rod_radius=rod_radius,
                    rod_rgba=cols[(n - 1) % len(cols)],
                    routing_sites=False, tendon_stubs=True)
                for n, geo in enumerate(geos, start=1)]
        return (base, segs), "promax (digital_twin.mjcf_generator)"
    except Exception as exc:
        return None, (f"simple: the ProMax drawing is unavailable "
                      f"({type(exc).__name__}: {exc})")


def _proximal_hinges(prefix: str, n: int, order: str) -> tuple:
    x = f'<joint name="{prefix}uj{2 * n - 1}_x" axis="1 0 0"/>'
    y = f'<joint name="{prefix}uj{2 * n - 1}_y" axis="0 1 0"/>'
    return (y, x) if order == "yx" else (x, y)


def build_arm_xml(*, prefix: str = "canarm_",
                  mount_body: str = "canarm_mount",
                  chain_m=None,
                  base_pos=None, base_rpy_deg=(0.0, 0.0, 0.0),
                  plate_radii=DEFAULT_PLATE_RADII,
                  base_plate_radius: float = BASE_PLATE_RADIUS,
                  rod_radius: float = 0.009,
                  colours=None,
                  style: str = "promax",
                  proximal_order: str = CANARM_PROXIMAL_ORDER,
                  indent: str = "    ") -> str:
    """One arm's ``<worldbody>`` subtree: a mocap mount with a chain under it.

    Topology, identical to ``UMArm_KINEMATICS.fkine``'s product-of-exponentials
    chain: three segments, each a **link** body carrying the proximal universal
    joint as two stacked hinges, and a **distal plate** body carrying the second
    universal joint as two hinges about the 45-degree bracket axes
    ``(+-1, 1, 0)/sqrt(2)``.  Segments 2 and 3 hang a rigid ``JD`` spacer below
    the previous distal plate.  Twelve hinges, six universal joints, six plates.

    *proximal_order* is the declaration order of each proximal pair and must be
    the arm's own: ``"yx"`` for the CAN arm (measured), ``"xy"`` for the RS485
    arm.  *style* ``"promax"`` draws Fig. 1C's segment through the twin's
    drawing helper; ``"simple"`` draws a rod and two disks.  Neither moves a
    site: ``{prefix}plate0..5`` are the six u-joint centres the cameras report,
    so the measured overlay and the rendered model are comparable point for
    point under either drawing.
    """
    if style not in ARM_STYLES:
        raise ValueError(f"style must be one of {ARM_STYLES}; got {style!r}")
    if proximal_order not in PROXIMAL_ORDERS:
        raise ValueError(f"proximal_order must be one of {PROXIMAL_ORDERS}; "
                         f"got {proximal_order!r}")
    chain_m = DEFAULT_CANARM_CHAIN_M if chain_m is None else chain_m
    segs = _seg_geometry(chain_m)
    bx, by, bz = (float(v) for v in
                  (DEFAULT_CANARM_MOUNT[0] if base_pos is None else base_pos))
    quat = _quat_attr(base_rpy_deg)
    cols = tuple(colours or _SEG_COLOURS.get(prefix.rstrip("_"),
                                             _SEG_COLOURS["canarm"]))
    ja1, ja2 = (float(v) for v in plate_radii)

    promax = None
    if style == "promax":
        promax, note = _promax_parts(chain_m, prefix, cols, rod_radius,
                                     base_plate_radius)
    else:
        note = "simple (asked for)"
    LAST_ARM_STYLE_NOTE[prefix] = note

    lines: list[str] = []
    depth = 0

    def emit(text: str) -> None:
        lines.append(indent + "  " * depth + text)

    emit(f'<body name="{mount_body}" mocap="true" '
         f'pos="{bx:.10g} {by:.10g} {bz:.10g}" quat="{quat}">')
    depth += 1
    emit(f'<body name="{prefix}base" pos="0 0 0" childclass="umarm_display">')
    depth += 1
    if promax is not None:
        for element in promax[0]:
            emit(element)
    else:
        emit(f'<geom name="{prefix}base_geom" type="cylinder" '
             f'fromto="0 0 0.008 0 0 -0.008" size="{base_plate_radius:.10g}" '
             f'rgba="0.30 0.35 0.45 1" mass="0.5"/>')
    # Plate 0 is the base plate, and it sits on the STATIC body rather than on
    # segment 1's link: it is the frame the arm's kinematics de-rotates by, and
    # it does not turn with the first universal joint.
    emit(f'<site name="{prefix}plate0" pos="0 0 0"/>')

    for n, (span, jd) in enumerate(segs, start=1):
        col = cols[(n - 1) % len(cols)]
        parts = None if promax is None else promax[1][n - 1]
        if parts is not None:
            # Spacer, proximal u-joint disk and its brackets, on the PARENT.
            for element in parts["parent"]:
                emit(element)
        elif jd:
            # The rigid JD spacer ahead of segments 2 and 3, and the proximal
            # plate at its foot.  Both belong to the PARENT body: the spacer is
            # rigid to the plate above it, so plate 2 and plate 4 turn with the
            # segment above them, not with the joint they carry.
            emit(f'<geom name="{prefix}jd{n}" type="cylinder" '
                 f'fromto="0 0 0 0 0 {-jd:.10g}" size="0.008" '
                 f'rgba="0.55 0.58 0.65 1" mass="0.06"/>')
            emit(f'<geom name="{prefix}seg{n}_plate1_geom" type="cylinder" '
                 f'fromto="0 0 {-jd - 0.004:.10g} 0 0 {-jd + 0.004:.10g}" '
                 f'size="{ja1:.10g}" rgba="0.55 0.58 0.65 1" mass="0.0001"/>')
        if jd:
            emit(f'<site name="{prefix}plate{2 * (n - 1)}" pos="0 0 {-jd:.10g}"/>')
        # Link body: the proximal universal joint, as two stacked hinges in the
        # arm's own composition order.  Hinge order IS the twist order.
        emit(f'<body name="{prefix}seg{n}_link" pos="0 0 {-jd:.10g}">')
        depth += 1
        for hinge in _proximal_hinges(prefix, n, proximal_order):
            emit(hinge)
        if parts is not None:
            for element in parts["link"]:
                emit(element)
        else:
            emit(f'<geom name="{prefix}seg{n}_rod" type="cylinder" '
                 f'fromto="0 0 0 0 0 {-span:.10g}" size="{rod_radius:.10g}" '
                 f'rgba="{col}" mass="0.02"/>')
        # Distal plate body: the second universal joint, on the 45-degree
        # bracket axes.
        emit(f'<body name="{prefix}seg{n}_plate2" pos="0 0 {-span:.10g}">')
        depth += 1
        emit(f'<joint name="{prefix}uj{2 * n}_x" '
             f'axis="{_SQ2:.10g} {_SQ2:.10g} 0"/>')
        emit(f'<joint name="{prefix}uj{2 * n}_y" '
             f'axis="{-_SQ2:.10g} {_SQ2:.10g} 0"/>')
        if parts is not None:
            for element in parts["distal"]:
                emit(element)
        else:
            emit(f'<geom name="{prefix}seg{n}_plate2_geom" type="cylinder" '
                 f'fromto="0 0 -0.004 0 0 0.004" size="{ja2:.10g}" '
                 f'rgba="0.55 0.58 0.65 1" '
                 f'mass="{0.12 if n == 3 else 0.0001}"/>')
        emit(f'<site name="{prefix}plate{2 * n - 1}" pos="0 0 0"/>')

    # A short stub past the last plate, so the tip is visible and has a name.
    # It is NOT an end effector: no such rod is installed on this arm, and the
    # RS485 workspace already paid for inheriting a tool geometry nobody had
    # measured (its EE_PARAM bent 51 deg out of the segment axis for a year).
    emit(f'<geom name="{prefix}tip_rod" type="cylinder" '
         f'fromto="0 0 0 0 0 -0.05" size="0.006" '
         f'rgba="0.95 0.55 0.20 1" mass="0.02"/>')
    emit(f'<site name="{prefix}tip" pos="0 0 -0.05"/>')

    while depth > 0:
        depth -= 1
        emit("</body>")
    return "\n".join(lines)


def _quat_attr(rpy_deg) -> str:
    q = TF.rpy_to_quat(rpy_deg)
    return " ".join(f"{float(v):.17g}" for v in q)


# ---------------------------------------------------------------------------
# The room
# ---------------------------------------------------------------------------

_ROOM_TEMPLATE = """<mujoco model="umarm_room">
  <!-- GENERATED by viz/mjcf_canarm.py - edit the generator, not this file.
       DISPLAY ONLY: no actuators, no tendons, no contacts, no fitted dynamics.
       {genparams} -->
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

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

  <default>
    <default class="umarm_display">
      <joint type="hinge" damping="0.05" frictionloss="0" armature="{armature}"
        limited="true" range="-{joint_range} {joint_range}"/>
      <site type="sphere" size="0.006" rgba="0.9 0.9 0.2 1"/>
      <geom contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0.4 3.2" dir="0 0 -1" directional="true"/>
    <light pos="1.5 -1.0 2.5" dir="-0.4 0.4 -0.7"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="groundplane" contype="0" conaffinity="0"/>

{bodies}
  </worldbody>
</mujoco>
"""


def _placeholder_kinova(mount_body: str, base_pos, base_rpy_deg,
                        indent: str = "    ") -> str:
    """A box and a triad on a mocap mount — the Gen3 when the real one is absent.

    Small, obviously schematic, and jointless.  It exists so that "is the Gen3
    roughly there?" has an answer during the weeks before the Gen3's own scene
    module lands, and so that the room's mocap-body roster does not change shape
    when it does: ``kinova_mount`` is written the same way either way.

    Jointless is the load-bearing part.  A placeholder with seven fake hinges
    would accept the seven doubles a Gen3 feed publishes and draw a shape that
    responds to them, which is a picture of a robot that is not there.
    """
    px, py, pz = (float(v) for v in base_pos)
    lines = [
        f'{indent}<body name="{mount_body}" mocap="true" '
        f'pos="{px:.10g} {py:.10g} {pz:.10g}" quat="{_quat_attr(base_rpy_deg)}">',
        f'{indent}  <geom name="kinova_placeholder" type="box" '
        f'size="0.09 0.09 0.05" pos="0 0 0.05" rgba="0.35 0.75 1.0 0.35" '
        f'contype="0" conaffinity="0" mass="0"/>',
    ]
    for axis, (dx, dy, dz), rgba in (
            ("x", (0.18, 0, 0), "0.95 0.2 0.2 0.9"),
            ("y", (0, 0.18, 0), "0.2 0.9 0.3 0.9"),
            ("z", (0, 0, 0.18), "0.3 0.5 1.0 0.9")):
        lines.append(
            f'{indent}  <geom name="kinova_axis_{axis}" type="capsule" '
            f'fromto="0 0 0 {dx:.10g} {dy:.10g} {dz:.10g}" size="0.006" '
            f'rgba="{rgba}" contype="0" conaffinity="0" mass="0"/>')
    lines.append(f'{indent}  <site name="kinova_placeholder_site" pos="0 0 0"/>')
    lines.append(f'{indent}</body>')
    return "\n".join(lines)


def _merge_kinova(xml: str, base_pos, base_rpy_deg):
    """``(xml, note)`` — the room with a real Gen3 grafted in, or ``(None, why)``.

    LAZY AND FORGIVING, because ``UMArm_KINOVA/kinova_scene.py`` was being
    written in parallel with this file: it may not exist, may not export what
    this expects, and may need mesh files a given checkout does not carry.  Each
    of those is a reason to draw a placeholder and say which one happened —
    never a reason for the CAN arm not to render.

    Two shapes are accepted, in this order:

    1. ``attach_kinova(root, base_pos, base_rpy_deg)`` — grafting into an
       existing tree, which is what that module was written for.  Preferred
       because the *robot's own module* then owns which sections it needs, and
       because a section-by-section graft reuses this scene's ``<asset>`` rather
       than appending a second one (two ``<asset>`` blocks declaring a texture
       called ``groundplane`` is a compile error, and was the first thing this
       merge hit).
    2. a builder returning a complete document, whose ``<worldbody>`` bodies and
       auxiliary sections are lifted out.  A fallback for a module that only
       knows how to build a standalone scene.

    The Gen3 arrives with whatever its own module considers part of it —
    position actuators, a contact exclusion.  That is deliberately not filtered:
    owning the Gen3's model is that module's job, ``mj_forward`` integrates
    nothing, and this viewer never writes ``data.ctrl``.  What stays true is
    that *this* file adds no actuator of its own.
    """
    try:
        from UMArm_KINOVA import kinova_scene as ks       # noqa: WPS433
    except Exception as exc:
        return None, f"UMArm_KINOVA.kinova_scene unavailable ({exc})"

    attach = getattr(ks, "attach_kinova", None)
    if callable(attach):
        try:
            root = ET.fromstring(xml)
            attach(root, base_pos, base_rpy_deg)
            return ET.tostring(root, encoding="unicode"), "grafted via attach_kinova"
        except TypeError:
            try:
                root = ET.fromstring(xml)
                attach(root)
                return (ET.tostring(root, encoding="unicode"),
                        "grafted via attach_kinova (its own default base pose)")
            except Exception as exc:
                return None, f"attach_kinova refused: {exc}"
        except Exception as exc:
            return None, f"attach_kinova raised: {exc}"

    fn = None
    for name in ("build_kinova_xml", "build_scene_xml", "kinova_xml",
                 "generate_xml"):
        fn = getattr(ks, name, None)
        if callable(fn):
            break
    if fn is None:
        return None, "kinova_scene exports neither attach_kinova nor a builder"
    text = None
    for kwargs in ({"base_pos": base_pos, "base_rpy_deg": base_rpy_deg},
                   {"base_pos": base_pos}, {}):
        try:
            text = fn(**kwargs)
            break
        except TypeError:
            continue
        except Exception as exc:
            return None, f"kinova_scene builder raised: {exc}"
    if not isinstance(text, str) or not text.strip():
        return None, "kinova_scene builder returned no XML"
    try:
        src = ET.fromstring(text)
        dst = ET.fromstring(xml)
    except ET.ParseError as exc:
        return None, f"kinova_scene XML did not parse: {exc}"
    _graft(src, dst)
    return ET.tostring(dst, encoding="unicode"), "merged a kinova_scene document"


def _graft(src, dst) -> None:
    """Copy *src*'s robot-bearing sections into *dst*, skipping name clashes.

    Crude on purpose: it exists for a Gen3 module that only knows how to emit a
    standalone scene, and the only clashes it can meet are the scenery both
    documents carry (a floor, a skybox, a groundplane material).  Anything
    subtler shows up as a failed compile, which :func:`build_room_scene`
    already answers by dropping the Gen3.
    """
    for tag in ("asset", "default", "worldbody", "contact", "actuator",
                "equality", "tendon"):
        source = src.find(tag)
        if source is None:
            continue
        target = dst.find(tag)
        if target is None:
            target = ET.SubElement(dst, tag)
        taken = {(c.tag, c.get("name")) for c in target if c.get("name")}
        for child in source:
            if child.tag == "geom" and tag == "worldbody":
                continue                      # the other scene's floor
            if (child.tag, child.get("name")) in taken:
                continue
            target.append(child)


def build_room_scene(include_rs485: bool = False, include_kinova: bool = False,
                     *,
                     canarm_chain_m=None, rs485_chain_m=None,
                     canarm_mount=DEFAULT_CANARM_MOUNT,
                     rs485_mount=DEFAULT_RS485_MOUNT,
                     kinova_mount=DEFAULT_KINOVA_MOUNT,
                     validate: bool = True) -> str:
    """The room's MJCF: the CAN arm, plus whoever else was asked for.

    The CAN arm is always present and is composed first, so its bodies, its
    joints and therefore its **qpos block** are the same in all four flag
    combinations.  A viewer addressing it by joint name is unaffected either
    way; a viewer addressing it by index is unaffected too, which is one fewer
    way for a dropped robot to move an arm.

    The CAN arm is the ProMax drawing under its measured ``"yx"`` composition;
    the RS485 arm is the rod-and-disk drawing under the legacy ``"xy"``.

    *validate* compiles the result once with ``mujoco`` before returning it, and
    on failure rebuilds with the Gen3 replaced by its placeholder.  It is on by
    default for the reason the whole composer exists: a scene module from
    another package can fail in ways this one cannot anticipate — a missing STL,
    a class name collision — and the correct response to every one of them is a
    room without a Gen3, not a room that will not open.
    """
    global LAST_KINOVA_NOTE

    bodies = [build_arm_xml(prefix="canarm_", mount_body="canarm_mount",
                            chain_m=canarm_chain_m or DEFAULT_CANARM_CHAIN_M,
                            base_pos=canarm_mount[0],
                            base_rpy_deg=canarm_mount[1],
                            style="promax",
                            proximal_order=CANARM_PROXIMAL_ORDER)]
    if include_rs485:
        bodies.append(build_arm_xml(prefix="rs485_", mount_body="rs485_mount",
                                    chain_m=rs485_chain_m or DEFAULT_RS485_CHAIN_M,
                                    base_pos=rs485_mount[0],
                                    base_rpy_deg=rs485_mount[1],
                                    style="simple",
                                    proximal_order=RS485_PROXIMAL_ORDER))
    arms_xml = _compose(bodies, include_rs485, include_kinova)

    if not include_kinova:
        LAST_KINOVA_NOTE = "not requested"
        return arms_xml

    merged, note = _merge_kinova(arms_xml, kinova_mount[0], kinova_mount[1])
    if merged is not None and validate:
        ok, why = _compiles(merged)
        if not ok:
            merged, note = None, f"{note}, but it did not compile ({why})"
    if merged is not None:
        LAST_KINOVA_NOTE = note
        return merged

    LAST_KINOVA_NOTE = f"placeholder: {note}"
    bodies.append(_placeholder_kinova("kinova_mount", kinova_mount[0],
                                      kinova_mount[1]))
    return _compose(bodies, include_rs485, include_kinova)


def _compose(bodies, include_rs485: bool, include_kinova: bool) -> str:
    return _ROOM_TEMPLATE.format(
        genparams=(f"include_rs485={bool(include_rs485)}, "
                   f"include_kinova={bool(include_kinova)}, "
                   f"canarm_proximal_order={CANARM_PROXIMAL_ORDER}"),
        armature=f"{JOINT_ARMATURE:.10g}",
        joint_range=f"{JOINT_RANGE_RAD:.17g}",
        bodies="\n".join(bodies))


def _compiles(xml: str) -> tuple:
    try:
        import mujoco
    except Exception as exc:                                 # pragma: no cover
        return True, f"mujoco not importable ({exc}); skipped"
    try:
        mujoco.MjModel.from_xml_string(xml)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def write_room_scene(out_path, **kwargs) -> Path:
    """Write the room XML and return the path written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_room_scene(**kwargs), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rs485", action="store_true", help="include the RS485 arm")
    ap.add_argument("--kinova", action="store_true", help="include the Gen3")
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    args = ap.parse_args(argv)
    xml = build_room_scene(include_rs485=args.rs485,
                           include_kinova=args.kinova)
    if args.out:
        print(f"wrote {write_room_scene(args.out, include_rs485=args.rs485, include_kinova=args.kinova)}")
    else:
        print(xml)
    if args.kinova:
        print(f"# kinova: {LAST_KINOVA_NOTE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
