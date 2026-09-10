r"""The CAN arm's simulation MJCF -- the single source of the twin's geometry.

The display model in ``viz/mjcf_canarm.py`` answers "where are the links, given
q" and deliberately carries no actuators, no tendons and no fitted dynamics.
This one is the other half: the same kinematic chain plus the twenty-four muscle
tendons, the masses, and the small number of MuJoCo-side tunables the twin is
allowed to fit.  They are separate files because they fail differently -- a
display that is slightly wrong is a picture, a twin that is slightly wrong is a
controller tuned against a lie -- and because the display must stay cheap enough
to compile in a spawned process.

WHAT CAME ACROSS FROM ``viz/mjcf_canarm.py`` UNTOUCHED: the topology (three link
bodies, three distal-plate bodies, twelve hinges, distal axes
``(+-1, 1, 0)/sqrt(2)``), the JD spacer belonging to the *parent* body, the six
``canarm_plate0..5`` sites, and the mocap mount.  That geometry is already
verified against ``UMArm_KINEMATICS.fkine`` to 4.4e-16 m over 200 random
configurations, so re-deriving it here would only be a chance to get it wrong.

FIVE THINGS CHANGED, and each one is a measurement or a contract clause:

1. **The proximal hinges are declared y-then-x.**  MuJoCo composes same-body
   hinges in declaration order, so declaring ``x`` first asserts fkine's
   ``order="xy"``.  The CAN arm's measured assembly is ``order="yx"``
   (``UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER``, 2026-08-21: it took held-out
   u-joint-centre error from 4.6 mm RMS / 20.4 mm worst to 2.0 / 7.0 mm).  Over
   200 random ``q`` in +-0.6 rad the two orders disagree by up to 0.189 m at a
   u-joint centre, so this is not a cosmetic swap.  The cost is that ``qpos`` is
   no longer ``q``: see :data:`QPOS_FROM_Q` and :func:`q_to_qpos`.
2. **The ring radii come from the parameter table, not from the display file.**
   ``viz.mjcf_canarm.DEFAULT_PLATE_RADII`` is ``(0.055, 0.055)`` and its own
   comment calls it cosmetic; in a twin the ring radius *is* the muscle moment
   arm.  ``CANARM_PARAMS`` gives ``JA1 = JA2 = 0.047`` m and
   ``AO1 = AO2 = 0.028`` m on all three segments.  Using the display numbers
   would inflate every plate-ring torque by 17 % and every link-ring torque by
   96 %.
3. **The ring azimuths are derived from the measured actuator map**, not
   transcribed.  See :func:`actuator_seats` and the section below.
4. **Masses exist.**  The display model carries 0.3204 kg of moving mass because
   MuJoCo needs *something* to compile; a dissipation fit run against it would
   absorb the mass error into ``damp_b1`` and report a good loss, since both
   ring frequency and zeta read mass.  What is here instead is a mass *model*
   with every term an argument and every default's provenance named -- which is
   still not a measurement.  See :data:`MASS_PROVENANCE`.
5. **Timestep 2 ms -> 1 ms, ``iterations=100``.**  ``digital_twin.TIMESTEP_S``
   pins 1 ms because ``sim_core`` counts node-logic passes in whole quanta;
   every consumer must read ``model.opt.timestep`` back rather than assume it.

------------------------------------------------------------------------------
HOW THE TENDONS ARE ROUTED, AND WHY THE TABLE IS COMPUTED RATHER THAN TYPED
------------------------------------------------------------------------------
Each segment carries eight muscles on four rings.  The four *lower* muscles run
from the proximal plate ring (radius ``JA1``, in the plane of the proximal
u-joint centre) down to the link's bottom ring (radius ``AO1``, ``AA1 + LL``
below that centre), so they span the proximal u-joint and nothing else.  The
four *upper* muscles run from the link's top ring (radius ``AO2``, ``AA1`` below
the proximal centre) down to the distal plate ring (radius ``JA2``, in the plane
of the distal centre), so they span the distal u-joint and nothing else.  Each
muscle therefore has exactly two non-zero moment arms, both about one universal
joint, and the azimuth alone decides which axis it drives and with which sign.

Working the moment arms out in closed form at ``q = 0`` (``UC1 = UC2 = 0`` on
this hardware, so the plate site sits exactly in the joint plane) gives, for a
lower muscle at azimuth ``phi`` and a tendon length ``len``::

    d(len)/d(t1) = -(L * JA1 / len) * sin(phi)        t1 about +x
    d(len)/d(t2) = +(L * JA1 / len) * cos(phi)        t2 about +y

and for an upper muscle at azimuth ``psi``, whose axes are the 45-degree
bracket vectors::

    d(len)/d(t3) = -(M * JA2 / (sqrt2 * len)) * (sin(psi) - cos(psi))
    d(len)/d(t4) = +(M * JA2 / (sqrt2 * len)) * (cos(psi) + sin(psi))

A muscle pulls, so it drives a joint in whichever direction shortens it.  Each
of the eight (axis, sign) roles therefore has exactly one azimuth at which it is
pure -- the cross-axis arm is *identically* zero, not merely small -- and those
eight azimuths are :data:`LOWER_SEAT_DEG` and :data:`UPPER_SEAT_DEG`.

Given that, the seat table is a *consequence* of
``UMArm_KINEMATICS.canarm_actuators.MEASURED_JOINT_PAIRS`` rather than an
independent claim, and :func:`actuator_seats` computes it.  This choice is what
makes the routing test meaningful: a hand-typed table and a hand-checked test
can agree with each other and both be wrong about the metal.  It also handles,
without a special case, the fact that **segment 1 does not split into locals
1-4 lower and 5-8 upper the way segments 2 and 3 do** -- its top regulator
platform was replaced and remounted a quarter turn, so its lower ring carries
boards ``0x108, 0x104, 0x102, 0x106`` and its upper ring ``0x101, 0x105,
0x103, 0x107``.  The legacy table describes the platform that was taken off and
must not be used here (``canarm_actuators`` says so at length).

``pam_k`` is board ``0x100 + k``, by construction and not by convention, so
``sim_core`` maps node ids to actuators by identity.

------------------------------------------------------------------------------
WHAT THIS MODEL DOES NOT SHOW
------------------------------------------------------------------------------
* **Nothing here is a fit.**  ``joint_damping``, ``joint_frictionloss`` and
  ``tendon_damping`` carry the RS485 twin's fitted values as *defaults*, and
  those were fitted to a shorter, lighter arm with different muscles.  They are
  starting points for ``fit_bounce`` on this arm, not answers.
* **No mass on this arm has been weighed** except the operator's ~30 g per
  actuator.  The rest of :data:`MASS_PROVENANCE` is RS485 numbers and geometry.
  A scale under one segment would replace all of it, and would change every
  frequency and every zeta the twin predicts.
* **No contacts, no keyframes, no end effector.**  Every geom is
  ``contype="0" conaffinity="0"``; two arms drawn overlapping is a drawing.  The
  tip stub is a stub -- no end effector is installed on this arm, and the RS485
  workspace already paid for inheriting a tool geometry nobody had measured.
* **The force clip is inherited, not derived.**  ``FORCE_RANGE_N = 4000`` was
  sized to the RS485 force law's envelope.  This arm's ``coeff``/``bf`` are not
  fitted yet, so the number cannot be re-derived today; ``replay.assert_no_clamp``
  is what will report it being wrong.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\mjcf_generator.py``
REFERENCE READ:  ``digital_twin/reference/mjcf_fit.md`` section 1.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    # Same habit as viz/mjcf_canarm.py and six other modules here: every package
    # is a direct child of the workspace root, so a spawned process with any cwd
    # can still import this one.
    sys.path.insert(0, str(_WS_ROOT))

from UMArm_KINEMATICS import canarm_actuators as ca     # noqa: E402
from UMArm_KINEMATICS import canarm_params as cp        # noqa: E402
from UMArm_KINEMATICS import robot_params as rp         # noqa: E402

__all__ = [
    "TIMESTEP_S", "INTEGRATOR", "SOLVER_ITERATIONS",
    "PROXIMAL_ORDER", "QPOS_FROM_Q", "N_ACTUATORS", "BASE_CAN_ID",
    "LOWER_SEAT_DEG", "UPPER_SEAT_DEG", "MASS_PROVENANCE",
    "DEFAULT_JOINT_DAMPING", "DEFAULT_JOINT_FRICTIONLOSS",
    "DEFAULT_TENDON_DAMPING", "DEFAULT_LINK_DENSITY_KG_M",
    "DEFAULT_PLATE_MASS_KG", "DEFAULT_BRACKET_MASS_KG",
    "DEFAULT_ACTUATOR_MASS_KG", "DEFAULT_BASE_PLATE_MASS_KG",
    "DEFAULT_TIP_MASS_KG", "DEFAULT_BASE_POS", "DEFAULT_BASE_RPY_DEG",
    "FORCE_RANGE_N", "JOINT_RANGE_RAD", "JOINT_ARMATURE",
    "Seat", "actuator_seats", "actuator_names", "tendon_names", "joint_names",
    "plate_site_names", "actuator_joint_map",
    "fitted_params", "q_to_qpos", "qpos_to_q",
    "build_arm_xml", "generate_xml", "generate_scene", "build_model",
    "tendon_rest_lengths",
]

# ---------------------------------------------------------------------------
# Contract constants.  digital_twin/CONTRACT.md section 3 fixes all of these.
# ---------------------------------------------------------------------------

#: Physics quantum, seconds.  Mirrors ``digital_twin.TIMESTEP_S`` rather than
#: importing it, so this module can be read as a flat file; the test asserts the
#: two agree and that a compiled model reads it back.
TIMESTEP_S = 0.001

#: ``implicitfast`` is load-bearing rather than cosmetic: ``sim_core`` rewrites
#: ``model.tendon_damping`` at every node pass (the McKibben's damping is affine
#: in pressure), and the implicit treatment is what makes that unconditionally
#: stable at any physical magnitude.
INTEGRATOR = "implicitfast"

#: Solver iterations, from the RS485 twin's option block.  Inherited, not fitted.
SOLVER_ITERATIONS = 100

#: The CAN arm's measured proximal-pair composition
#: (``UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER``).  Every ``fkine`` call that
#: is meant to describe *this* model must pass it.
PROXIMAL_ORDER = "yx"

#: ``qpos[i] = q[QPOS_FROM_Q[i]]``.  Under ``order="yx"`` the y hinge of each
#: proximal universal joint is declared first, so the model's first degree of
#: freedom is ``t2``, not ``t1``.  The permutation is its own inverse, which is
#: why :func:`q_to_qpos` and :func:`qpos_to_q` are the same map applied twice.
QPOS_FROM_Q = (1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11)

#: 24 muscles, and ``pam_k`` is CAN board ``BASE_CAN_ID + k``.
N_ACTUATORS = 24
BASE_CAN_ID = 0x100

# ---------------------------------------------------------------------------
# Seat azimuths: the eight roles, and the one azimuth that is pure for each
# ---------------------------------------------------------------------------

#: ``(axis_within_ujoint, sign) -> azimuth in degrees`` for a **lower** muscle,
#: which spans a proximal universal joint (axis 0 = ``t1`` about +x, axis 1 =
#: ``t2`` about +y).  Derived, not transcribed: see the module docstring's
#: closed form.  At each of these four azimuths the *other* axis's moment arm is
#: identically zero, so a muscle seated here drives one axis and only one.
LOWER_SEAT_DEG = {(0, +1): 90.0, (0, -1): 270.0, (1, +1): 180.0, (1, -1): 0.0}

#: The same for an **upper** muscle, which spans a distal universal joint whose
#: axes are the 45-degree bracket vectors ``(x+y)/sqrt2`` (axis 2 = ``t3``) and
#: ``(-x+y)/sqrt2`` (axis 3 = ``t4``).  The four azimuths sit 45 degrees round
#: from the lower ring's, which is the bracket rotation and nothing else.
UPPER_SEAT_DEG = {(2, +1): 135.0, (2, -1): 315.0, (3, +1): 225.0, (3, -1): 45.0}

# ---------------------------------------------------------------------------
# Tunable defaults.  Every one of these is a keyword argument of generate_xml;
# none of them is read at call time from module scope.  The constants exist so
# a caller can cite a default without calling the generator, and so this file
# can say where each number came from next to the number itself.
# ---------------------------------------------------------------------------

#: N*m*s/rad per hinge.  The RS485 twin's fitted value (run
#: ``20260816_222851_real_rec1``, artifacts ``bounce_fit_2026-08-17/``), carried
#: here as a **seed for this arm's own fit**, not as an answer.  The engineering
#: guess it superseded there was 2.25, under which every mode was
#: critically-to-over damped (poke-test zeta ~ 0.73) and the twin could not
#: sustain the 0.2-2 deg rings the metal shows at any parameterisation of the
#: rest of the model.  The CAN arm is longer and heavier, so its spine damping
#: is unlikely to be the same number; it is likely to be the same order.
DEFAULT_JOINT_DAMPING = 0.026

#: N*m per hinge, Coulomb.  Same fit, same caveat.  The superseded guess was
#: 0.70 N*m, which alone exceeded the peak elastic torque of a ~1 deg
#: oscillation (measured stick threshold 0.56-1.11 deg per joint at 15 psi).
#: Known divergence recorded rather than papered over: on the RS485 arm,
#: dropping to 0.025 regressed the free-pendulum single-muscle campaign to 8/12
#: because the model has no deflated-muscle passive elasticity to hold the chain.
DEFAULT_JOINT_FRICTIONLOSS = 0.025

#: N*s/m per tendon -- the ``p = 0`` base of the runtime schedule
#: ``tendon_damping = base + damp_b1 * p_pa`` that ``sim_core`` writes into
#: ``model.tendon_damping`` every node pass.  **Chosen, never fitted**, on either
#: arm: every measured ring episode is pressurised, so the deflated-braid loss
#: has no excitation of its own.  On the RS485 arm 1.0 N*s/m sat well below the
#: ~55 N*s/m the ``damp_b1`` term contributes at the ~8 psi it rings at.
DEFAULT_TENDON_DAMPING = 1.0

#: kg/m of segment rod.  Provenance: the RS485 twin's rod geoms, 0.02 kg over a
#: ~0.20 m span.  It is a thin tube's linear density and it is the term this
#: model is least sensitive to -- at 0.73 m of total span it contributes 0.073 kg
#: against ~1.8 kg of arm.
DEFAULT_LINK_DENSITY_KG_M = 0.10

#: kg per plate disc (five moving plates: the three distal plates and the two
#: proximal plates that ride on the JD spacers).  Provenance: the RS485 twin's
#: ``PLATE3_MASS = 0.12``, the only plate mass anyone in either workspace put a
#: number on.  The CAN arm's plate radius is 0.047 m against the RS485 arm's
#: ``JA2 = 0.05``, so a straight carry-over is defensible on size; nothing
#: defends it on thickness or material.
DEFAULT_PLATE_MASS_KG = 0.12

#: kg per JD spacer -- the universal-joint yoke assembly between one segment's
#: distal plate and the next segment's proximal plate.  Provenance: half the
#: RS485 twin's ``BRACKET_MASS = 0.48``, halved because the CAN arm's JD gap is a
#: 73 mm spacer rather than the RS485 arm's 48 mm cast bracket and the visible
#: hardware is lighter.  This is the largest single guess in the file.
DEFAULT_BRACKET_MASS_KG = 0.24

#: kg per McKibben actuator.  **The only measured mass on this arm**: the
#: operator's ~30 g each, 24 of them, 0.72 kg in total.  Each muscle is split in
#: half between the two bodies its ends are anchored to, as a capsule along its
#: own path, so the mass lands at its own radius and contributes the rotational
#: inertia it actually has instead of being lumped on an axis.
DEFAULT_ACTUATOR_MASS_KG = 0.030

#: kg of the base plate.  It hangs off a static mount, so it enters no equation
#: of motion; it is here so the drawn disc has the right heft and so the number
#: is not silently zero.  Provenance: ``viz/mjcf_canarm.py``.
DEFAULT_BASE_PLATE_MASS_KG = 0.5

#: kg of the 50 mm tip stub past the last plate.  Provenance:
#: ``viz/mjcf_canarm.py``.  It is a stub, not an end effector -- no tool is
#: installed on this arm.
DEFAULT_TIP_MASS_KG = 0.02

#: Where the arm hangs when nothing writes its mount.  Arbitrary but not zero:
#: the arm hangs base-up, and a twin at the floor would ring against gravity
#: pointing the wrong way through the chain.  Same value as
#: ``viz.mjcf_canarm.DEFAULT_CANARM_MOUNT``, so a twin and the display sit in
#: the same place in a merged room.
DEFAULT_BASE_POS = (0.0, 0.0, 1.25)
DEFAULT_BASE_RPY_DEG = (0.0, 0.0, 0.0)

#: N.  Pull-only by construction (``ctrlrange`` and ``forcerange`` are both
#: ``[-FORCE_RANGE_N, 0]``); a McKibben cannot push.  The magnitude is the RS485
#: twin's, sized to *its* force law, and cannot be re-derived until this arm's
#: ``coeff``/``bf`` are fitted.  ``replay.assert_no_clamp`` exists because a
#: rollout that reaches the clip is running different physics than the one
#: that was fitted, and must be a finding rather than a saturation.
FORCE_RANGE_N = 4000.0

#: Inherited stability choices, flagged as inherited in the RS485 generator and
#: repeated for the same reason: they keep the model from folding through itself
#: when handed a ``q`` from a bad frame.  Nothing is fitted to them.  +-40 deg is
#: markedly wider than the +-30 deg the 2026-08-21 campaign spanned.
JOINT_RANGE_RAD = 0.6981317007977318
JOINT_ARMATURE = 0.01

#: Gravity, m/s^2.  The arm hangs base-up from the ceiling; -z is down.
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

#: Where every mass default came from, as data, so a reader does not have to
#: trust the prose and a test can assert the roster is complete.  The single
#: entry that is a measurement of *this* arm is ``actuator_mass_kg``.
MASS_PROVENANCE = {
    "link_density_kg_m": "RS485 twin rod geoms, 0.02 kg over ~0.20 m span",
    "plate_mass_kg": "RS485 twin PLATE3_MASS = 0.12 kg, carried over on radius",
    "bracket_mass_kg": "half RS485 twin BRACKET_MASS = 0.48 kg (spacer, not cast yoke)",
    "actuator_mass_kg": "operator, 2026-08: ~30 g per McKibben -- MEASURED, this arm",
    "base_plate_mass_kg": "viz/mjcf_canarm.py base disc, static body",
    "tip_mass_kg": "viz/mjcf_canarm.py tip stub; no end effector is installed",
}

_SQ2 = math.sqrt(2.0) / 2.0


# ---------------------------------------------------------------------------
# The seat table, derived from the measured actuator map
# ---------------------------------------------------------------------------

class Seat(NamedTuple):
    """Where one muscle is anchored, and what it drives.

    ``ring`` is ``"lower"`` (spans the segment's proximal universal joint) or
    ``"upper"`` (spans its distal one); ``joint`` indexes ``q`` in
    ``UMArm_KINEMATICS`` order, not qpos order; ``sign`` is ``+1`` when
    pressurising this muscle drives that joint angle positive.
    """

    board: int
    index: int          # k in pam_k, 1..24
    segment: int        # 1..3
    ring: str           # "lower" | "upper"
    azimuth_deg: float
    joint: int          # 0..11, q order
    sign: int           # +1 | -1


def actuator_seats(joint_pairs=None, segment_blocks=None) -> tuple:
    """The 24 :class:`Seat` rows, ``pam_1`` first, derived from the measured map.

    *joint_pairs* defaults to ``canarm_actuators.joint_pairs()``, which refuses
    to hand back the legacy table -- an actuator map that is wrong by one pair
    does not fail loudly, it drives the arm somewhere unexpected.

    The derivation is the module docstring's closed form read backwards: a
    ``(joint axis, sign)`` role fixes the ring (proximal axes are on the lower
    ring, distal axes on the upper one) and one azimuth.  Everything else --
    which local position on the manifold a board occupies, whether the segment's
    lower ring happens to be locals 1-4 -- falls out, and segment 1's remounted
    platform needs no special case.

    Raises ``ValueError`` on any map that does not seat each of the 24 boards
    exactly once with four muscles on each of the six rings, because a silently
    half-populated ring would compile into a model that pulls the arm sideways.
    """
    pairs = ca.joint_pairs() if joint_pairs is None else tuple(joint_pairs)
    blocks = ca.SEGMENT_BLOCKS if segment_blocks is None else tuple(segment_blocks)
    if len(pairs) != 12:
        raise ValueError(f"expected 12 antagonistic pairs; got {len(pairs)}")

    by_board: dict[int, Seat] = {}
    for joint, (pos_board, neg_board) in enumerate(pairs):
        axis = joint % 4                     # 0,1 = proximal t1,t2; 2,3 = distal
        ring = "lower" if axis < 2 else "upper"
        table = LOWER_SEAT_DEG if axis < 2 else UPPER_SEAT_DEG
        for board, sign in ((int(pos_board), +1), (int(neg_board), -1)):
            segment = _segment_of(board, blocks)
            if joint // 4 != segment - 1:
                raise ValueError(
                    f"board 0x{board:03X} sits on segment {segment} but the map "
                    f"gives it joint {joint} on segment {joint // 4 + 1}")
            if board in by_board:
                raise ValueError(f"board 0x{board:03X} appears twice in the map")
            by_board[board] = Seat(
                board=board, index=board - BASE_CAN_ID, segment=segment,
                ring=ring, azimuth_deg=table[(axis, sign)],
                joint=joint, sign=sign)

    seats = tuple(by_board[BASE_CAN_ID + k] for k in range(1, N_ACTUATORS + 1)
                  if BASE_CAN_ID + k in by_board)
    if len(seats) != N_ACTUATORS:
        missing = sorted(set(range(1, N_ACTUATORS + 1))
                         - {s.index for s in by_board.values()})
        raise ValueError(f"the map seats {len(seats)}/24 boards; missing {missing}")
    for segment in (1, 2, 3):
        for ring in ("lower", "upper"):
            here = [s for s in seats if s.segment == segment and s.ring == ring]
            if len(here) != 4:
                raise ValueError(
                    f"segment {segment}'s {ring} ring carries {len(here)} "
                    f"muscles, not 4")
            azimuths = sorted(round(s.azimuth_deg, 6) for s in here)
            want = sorted(round(v, 6) for v in
                          (LOWER_SEAT_DEG if ring == "lower"
                           else UPPER_SEAT_DEG).values())
            if azimuths != want:
                raise ValueError(
                    f"segment {segment}'s {ring} ring azimuths {azimuths} are "
                    f"not a permutation of the four pure seats {want}")
    return seats


def _segment_of(board: int, blocks) -> int:
    for i, block in enumerate(blocks, start=1):
        if board in block:
            return i
    raise ValueError(f"board 0x{board:03X} is in none of the segment blocks")


def actuator_names(prefix: str = "") -> tuple:
    """``pam_1 .. pam_24``.  ``pam_k`` is CAN board ``0x100 + k``.

    *prefix* exists only for a room holding two twin arms; the contract pins the
    bare names, so it defaults to empty and a caller that sets it has opted out
    of ``sim_core``'s node-id-to-actuator identity.
    """
    return tuple(f"{prefix}pam_{k}" for k in range(1, N_ACTUATORS + 1))


def tendon_names(prefix: str = "") -> tuple:
    """``t_pam_1 .. t_pam_24``, index for index with :func:`actuator_names`."""
    return tuple(f"{prefix}t_pam_{k}" for k in range(1, N_ACTUATORS + 1))


def joint_names(prefix: str = "canarm_") -> tuple:
    """The twelve hinge names in **qpos order**, which is not ``q`` order.

    Under ``order="yx"`` the proximal pair is declared ``y`` then ``x``, so this
    reads ``uj1_y, uj1_x, uj2_x, uj2_y, ...``.  A caller wanting ``q`` order
    should index this with :data:`QPOS_FROM_Q`, or better, use
    :func:`q_to_qpos` and never hold the two orders in its head at once.
    """
    out = []
    for k in range(1, 7):
        out.extend((f"{prefix}uj{k}_y", f"{prefix}uj{k}_x") if k % 2
                   else (f"{prefix}uj{k}_x", f"{prefix}uj{k}_y"))
    return tuple(out)


def plate_site_names(prefix: str = "canarm_") -> tuple:
    """``plate0..plate5`` -- the six u-joint centres the cameras report."""
    return tuple(f"{prefix}plate{p}" for p in range(6))


def actuator_joint_map(seats=None) -> tuple:
    """``((joint, sign), ...)`` for ``pam_1 .. pam_24``, in ``q`` order.

    The compact form ``sim_core`` and the target generator want: given an
    actuator index, which degree of freedom it pulls and which way.
    """
    seats = actuator_seats() if seats is None else seats
    return tuple((s.joint, s.sign) for s in seats)


def fitted_params() -> np.ndarray:
    """A fresh, writable copy of ``CANARM_PARAMS`` -- this model's FK oracle.

    Handed to ``fkine(q, params, order="yx")`` it reproduces this model's body
    frames, because the generator reads its geometry from the same table.  A
    copy rather than the array itself: ``CANARM_PARAMS`` is read-only precisely
    because it is a default argument all over ``fkine``, and a caller that
    perturbs a length for a sensitivity sweep must not re-zero every other
    caller.
    """
    return np.array(cp.require_measured(), dtype=float, copy=True)


def q_to_qpos(q):
    """``q`` in ``UMArm_KINEMATICS`` order -> this model's ``qpos``.

    The permutation exists because the CAN arm's proximal universal joints
    compose y-then-x (fkine ``order="yx"``, measured 2026-08-21) and MuJoCo
    composes same-body hinges in declaration order.  It is its own inverse.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 12:
        raise ValueError(f"q must have 12 columns; got shape {q.shape}")
    return q[..., list(QPOS_FROM_Q)]


def qpos_to_q(qpos):
    """This model's ``qpos`` -> ``q`` in ``UMArm_KINEMATICS`` order."""
    return q_to_qpos(qpos)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class _SegGeom(NamedTuple):
    """Every z-coordinate one segment's bodies and rings hang off, in metres.

    Frames: ``jd`` and ``plate1`` are in the *parent* body (the previous
    segment's distal plate, or the base); everything else is in the segment's
    own link body, whose origin is the proximal u-joint centre.
    """

    jd: float               # spacer above the proximal centre, in the parent
    span: float             # proximal centre -> distal centre
    ja1: float              # proximal plate ring radius
    ja2: float              # distal plate ring radius
    ao1: float              # link bottom ring radius
    ao2: float              # link top ring radius
    z_top_ring: float       # link frame, = -(UC1 + AA1)
    z_bot_ring: float       # link frame, = -(UC1 + AA1 + LL)


def _seg_geometry(params) -> list:
    out = []
    for row in np.asarray(params, dtype=float):
        uc1 = float(row[rp.COL_UC1])
        aa1 = float(row[rp.COL_AA1])
        aa2 = float(row[rp.COL_AA2])
        ll = float(row[rp.COL_LL])
        span = uc1 + aa1 + ll + aa2 + float(row[rp.COL_UC2])
        if span <= 0.0:
            raise ValueError(f"segment span must be positive; got {span}")
        out.append(_SegGeom(
            jd=float(row[rp.COL_JD]), span=span,
            ja1=float(row[rp.COL_JA1]), ja2=float(row[rp.COL_JA2]),
            ao1=float(row[rp.COL_AO1]), ao2=float(row[rp.COL_AO2]),
            z_top_ring=-(uc1 + aa1), z_bot_ring=-(uc1 + aa1 + ll)))
    return out


def _ring_xy(radius: float, angle_deg: float) -> tuple:
    a = math.radians(angle_deg)
    return radius * math.cos(a), radius * math.sin(a)


def _g(v: float) -> str:
    return f"{float(v):.10g}"


def _quat_attr(rpy_deg) -> str:
    """Body quaternion from roll-pitch-yaw, via ``viz.transforms``.

    Routed through the same helper the display model uses so a twin and the
    display disagree about no angle convention, ever.
    """
    from viz import transforms as TF

    return " ".join(f"{float(v):.17g}" for v in TF.rpy_to_quat(rpy_deg))


# ---------------------------------------------------------------------------
# The arm subtree
# ---------------------------------------------------------------------------

def build_arm_xml(*,
                  prefix: str = "canarm_",
                  mount_body: str = "canarm_mount",
                  mocap_mount: bool = True,
                  actuator_prefix: str = "",
                  params=None,
                  seats=None,
                  base_pos=DEFAULT_BASE_POS,
                  base_rpy_deg=DEFAULT_BASE_RPY_DEG,
                  link_density=DEFAULT_LINK_DENSITY_KG_M,
                  plate_mass=DEFAULT_PLATE_MASS_KG,
                  bracket_mass=DEFAULT_BRACKET_MASS_KG,
                  actuator_mass=DEFAULT_ACTUATOR_MASS_KG,
                  base_plate_mass=DEFAULT_BASE_PLATE_MASS_KG,
                  tip_mass=DEFAULT_TIP_MASS_KG,
                  rod_radius: float = 0.009,
                  muscle_radius: float = 0.006,
                  plate_half_thickness: float = 0.004,
                  spacer_radius: float = 0.008,
                  tip_length: float = 0.05,
                  force_range_n: float = FORCE_RANGE_N,
                  class_name: str = "umarm",
                  indent: str = "    ") -> tuple:
    """``(worldbody_subtree, tendon_block, actuator_block)`` for one twin arm.

    Split into three strings rather than one because MJCF puts tendons and
    actuators in their own top-level sections, and because that is the shape a
    room composer needs: ``build_room_scene``-style merging appends each section
    to the section of the same name in the destination document.

    The subtree is the display model's, with the five changes the module
    docstring lists.  Every mass, radius and dissipation scalar is an argument;
    nothing in the body of this function reads a module constant at call time.
    """
    params = cp.require_measured() if params is None else params
    segs = _seg_geometry(params)
    seats = actuator_seats() if seats is None else tuple(seats)
    by_seg_ring: dict = {}
    for s in seats:
        by_seg_ring.setdefault((s.segment, s.ring), []).append(s)

    bx, by, bz = (float(v) for v in base_pos)
    lines: list[str] = []
    depth = 0

    def emit(text: str) -> None:
        lines.append(indent + "  " * depth + text)

    mocap_attr = ' mocap="true"' if mocap_mount else ""
    emit(f'<body name="{mount_body}"{mocap_attr} '
         f'pos="{_g(bx)} {_g(by)} {_g(bz)}" quat="{_quat_attr(base_rpy_deg)}">')
    depth += 1
    emit(f'<body name="{prefix}base" pos="0 0 0" childclass="{class_name}">')
    depth += 1
    emit(f'<geom name="{prefix}base_geom" type="cylinder" '
         f'fromto="0 0 0.008 0 0 -0.008" size="{_g(segs[0].ja1 + 0.02)}" '
         f'rgba="0.30 0.35 0.45 1" mass="{_g(base_plate_mass)}"/>')
    # Plate 0 is the base plate and it sits on the STATIC body, not on segment
    # 1's link: it is the frame the arm's kinematics de-rotates by, and it does
    # not turn with the first universal joint.
    emit(f'<site name="{prefix}plate0" pos="0 0 0"/>')

    for n, geo in enumerate(segs, start=1):
        lower = by_seg_ring[(n, "lower")]
        upper = by_seg_ring[(n, "upper")]

        if geo.jd:
            # The rigid JD spacer ahead of segments 2 and 3, and the proximal
            # plate at its foot.  Both belong to the PARENT body: the spacer is
            # rigid to the plate above it, so plates 2 and 4 turn with the
            # segment above them, not with the joint they carry.
            emit(f'<geom name="{prefix}jd{n}" type="cylinder" '
                 f'fromto="0 0 0 0 0 {_g(-geo.jd)}" size="{_g(spacer_radius)}" '
                 f'rgba="0.55 0.58 0.65 1" mass="{_g(bracket_mass)}"/>')
            emit(f'<geom name="{prefix}seg{n}_plate1_geom" type="cylinder" '
                 f'fromto="0 0 {_g(-geo.jd - plate_half_thickness)} '
                 f'0 0 {_g(-geo.jd + plate_half_thickness)}" '
                 f'size="{_g(geo.ja1)}" rgba="0.55 0.58 0.65 1" '
                 f'mass="{_g(plate_mass)}"/>')
            emit(f'<site name="{prefix}plate{2 * (n - 1)}" '
                 f'pos="0 0 {_g(-geo.jd)}"/>')

        # The proximal ring, and the upper half of each lower muscle.  Both live
        # in the parent body: the muscle's top end is anchored to the plate above
        # the joint, so half its mass never sees the joint move.
        for s in lower:
            x, y = _ring_xy(geo.ja1, s.azimuth_deg)
            emit(f'<site name="{prefix}{_ring_site(s)}" '
                 f'pos="{_g(x)} {_g(y)} {_g(-geo.jd)}"/>')
            mx, my = _ring_xy(0.5 * (geo.ja1 + geo.ao1), s.azimuth_deg)
            emit(f'<geom name="{prefix}m{s.index}_hi" type="capsule" '
                 f'fromto="{_g(x)} {_g(y)} {_g(-geo.jd)} '
                 f'{_g(mx)} {_g(my)} {_g(-geo.jd + 0.5 * geo.z_bot_ring)}" '
                 f'size="{_g(muscle_radius)}" rgba="0.85 0.30 0.25 0.7" '
                 f'mass="{_g(0.5 * actuator_mass)}"/>')

        # Link body: the proximal universal joint as two stacked hinges at one
        # origin, **y declared first**.  Declaration order is the composition
        # order, so this is what makes the model agree with fkine order="yx" --
        # the CAN arm's measured assembly -- and it is why qpos is a permutation
        # of q rather than q itself.
        emit(f'<body name="{prefix}seg{n}_link" pos="0 0 {_g(-geo.jd)}">')
        depth += 1
        emit(f'<joint name="{prefix}uj{2 * n - 1}_y" axis="0 1 0"/>')
        emit(f'<joint name="{prefix}uj{2 * n - 1}_x" axis="1 0 0"/>')
        emit(f'<geom name="{prefix}seg{n}_rod" type="cylinder" '
             f'fromto="0 0 0 0 0 {_g(-geo.span)}" size="{_g(rod_radius)}" '
             f'rgba="0.20 0.55 0.75 1" '
             f'mass="{_g(link_density * geo.span)}"/>')

        for s in lower:
            # Lower muscle, link end: the bottom ring, AA2 above the distal
            # centre, and the lower half of the muscle's mass along its own path.
            x, y = _ring_xy(geo.ao1, s.azimuth_deg)
            emit(f'<site name="{prefix}{_ring_site(s, end="lo")}" '
                 f'pos="{_g(x)} {_g(y)} {_g(geo.z_bot_ring)}"/>')
            mx, my = _ring_xy(0.5 * (geo.ja1 + geo.ao1), s.azimuth_deg)
            emit(f'<geom name="{prefix}m{s.index}_lo" type="capsule" '
                 f'fromto="{_g(mx)} {_g(my)} {_g(0.5 * geo.z_bot_ring)} '
                 f'{_g(x)} {_g(y)} {_g(geo.z_bot_ring)}" '
                 f'size="{_g(muscle_radius)}" rgba="0.85 0.30 0.25 0.7" '
                 f'mass="{_g(0.5 * actuator_mass)}"/>')
        for s in upper:
            # Upper muscle, link end: the top ring, AA1 below the proximal
            # centre.  This half of the muscle's mass stays with the link when
            # the distal joint moves; the other half rides the distal plate,
            # which is why the mass is split at the midpoint rather than lumped.
            x, y = _ring_xy(geo.ao2, s.azimuth_deg)
            emit(f'<site name="{prefix}{_ring_site(s, end="hi")}" '
                 f'pos="{_g(x)} {_g(y)} {_g(geo.z_top_ring)}"/>')
            mx, my = _ring_xy(0.5 * (geo.ao2 + geo.ja2), s.azimuth_deg)
            mz = 0.5 * (geo.z_top_ring - geo.span)
            emit(f'<geom name="{prefix}m{s.index}_hi" type="capsule" '
                 f'fromto="{_g(x)} {_g(y)} {_g(geo.z_top_ring)} '
                 f'{_g(mx)} {_g(my)} {_g(mz)}" '
                 f'size="{_g(muscle_radius)}" rgba="0.85 0.30 0.25 0.7" '
                 f'mass="{_g(0.5 * actuator_mass)}"/>')

        # Distal plate body: the second universal joint, on the 45-degree
        # bracket axes.  Declared x then y, which IS fkine's composition for the
        # distal pair -- the 2026-08-21 experiment tested swapping this one too
        # and it was markedly worse, so xi3 is genuinely the link-fixed axis.
        emit(f'<body name="{prefix}seg{n}_plate2" pos="0 0 {_g(-geo.span)}">')
        depth += 1
        emit(f'<joint name="{prefix}uj{2 * n}_x" axis="{_SQ2:.10g} {_SQ2:.10g} 0"/>')
        emit(f'<joint name="{prefix}uj{2 * n}_y" axis="{-_SQ2:.10g} {_SQ2:.10g} 0"/>')
        emit(f'<geom name="{prefix}seg{n}_plate2_geom" type="cylinder" '
             f'fromto="0 0 {_g(-plate_half_thickness)} '
             f'0 0 {_g(plate_half_thickness)}" size="{_g(geo.ja2)}" '
             f'rgba="0.55 0.58 0.65 1" mass="{_g(plate_mass)}"/>')
        emit(f'<site name="{prefix}plate{2 * n - 1}" pos="0 0 0"/>')
        for s in upper:
            x, y = _ring_xy(geo.ja2, s.azimuth_deg)
            emit(f'<site name="{prefix}{_ring_site(s, end="lo")}" '
                 f'pos="{_g(x)} {_g(y)} 0"/>')
            mx, my = _ring_xy(0.5 * (geo.ao2 + geo.ja2), s.azimuth_deg)
            mz = 0.5 * (geo.z_top_ring - geo.span) + geo.span
            emit(f'<geom name="{prefix}m{s.index}_lo" type="capsule" '
                 f'fromto="{_g(mx)} {_g(my)} {_g(mz)} {_g(x)} {_g(y)} 0" '
                 f'size="{_g(muscle_radius)}" rgba="0.85 0.30 0.25 0.7" '
                 f'mass="{_g(0.5 * actuator_mass)}"/>')

    # A short stub past the last plate so the tip is visible and has a name.  It
    # is NOT an end effector: no such rod is installed on this arm, and the RS485
    # workspace already paid for inheriting a tool geometry nobody had measured
    # (its EE_PARAM bent 51 deg out of the segment axis for a year).
    emit(f'<geom name="{prefix}tip_rod" type="cylinder" '
         f'fromto="0 0 0 0 0 {_g(-tip_length)}" size="0.006" '
         f'rgba="0.95 0.55 0.20 1" mass="{_g(tip_mass)}"/>')
    emit(f'<site name="{prefix}tip" pos="0 0 {_g(-tip_length)}"/>')

    while depth > 0:
        depth -= 1
        emit("</body>")
    bodies = "\n".join(lines)

    ten_lines, act_lines = [], []
    for s in seats:
        ten = f"{actuator_prefix}t_pam_{s.index}"
        act = f"{actuator_prefix}pam_{s.index}"
        hi = f"{prefix}{_ring_site(s, end='hi')}"
        lo = f"{prefix}{_ring_site(s, end='lo')}"
        ten_lines.append(
            f'{indent}<spatial name="{ten}" class="{class_name}">\n'
            f'{indent}  <site site="{hi}"/>\n'
            f'{indent}  <site site="{lo}"/>\n'
            f'{indent}</spatial>')
        # Pull-only by construction, on both the command and the force: the
        # clip is what replay.assert_no_clamp watches, and a clip that can be
        # reached from the positive side would let a McKibben push.
        act_lines.append(
            f'{indent}<motor name="{act}" tendon="{ten}" gear="1" '
            f'ctrllimited="true" ctrlrange="{_g(-force_range_n)} 0" '
            f'forcelimited="true" forcerange="{_g(-force_range_n)} 0"/>')
    return bodies, "\n".join(ten_lines), "\n".join(act_lines)


def _ring_site(seat: Seat, end: str = "hi") -> str:
    """Site name for one end of one muscle.

    Named by board index rather than by ring position, so a site name answers
    "which board is this" without a lookup, and a routing mistake shows up as a
    name that does not exist rather than as a plausible wrong tendon.
    """
    return f"s{seat.segment}_a{seat.index}_{end}"


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

_SCENE_TEMPLATE = """<mujoco model="umarm_canarm_twin">
  <!-- GENERATED by digital_twin/mjcf_generator.py - edit the generator, not
       this file.  PHYSICS model: 24 tendon actuators, masses, fitted-seed
       dissipation.  No contacts, no keyframes, no end effector.
       {genparams} -->
  <compiler angle="radian"/>
  <option gravity="{gravity}" timestep="{timestep}" integrator="{integrator}"
    iterations="{iterations}"/>

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
    <default class="{class_name}">
      <joint type="hinge" damping="{joint_damping}" frictionloss="{joint_frictionloss}"
        armature="{armature}" limited="true" range="-{joint_range} {joint_range}"/>
      <site type="sphere" size="0.0035" rgba="0.9 0.9 0.2 1"/>
      <tendon width="0.0035" damping="{tendon_damping}" rgba="0.9 0.4 0.3 1"/>
      <geom contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0.4 3.2" dir="0 0 -1" directional="true"/>
    <light pos="1.5 -1.0 2.5" dir="-0.4 0.4 -0.7"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="groundplane" contype="0" conaffinity="0"/>

{bodies}
  </worldbody>

  <tendon>
{tendons}
  </tendon>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def generate_xml(*,
                 base_pos=DEFAULT_BASE_POS,
                 base_rpy_deg=DEFAULT_BASE_RPY_DEG,
                 joint_damping: float = DEFAULT_JOINT_DAMPING,
                 joint_frictionloss: float = DEFAULT_JOINT_FRICTIONLOSS,
                 tendon_damping: float = DEFAULT_TENDON_DAMPING,
                 link_density: float = DEFAULT_LINK_DENSITY_KG_M,
                 plate_mass: float = DEFAULT_PLATE_MASS_KG,
                 bracket_mass: float = DEFAULT_BRACKET_MASS_KG,
                 actuator_mass: float = DEFAULT_ACTUATOR_MASS_KG,
                 base_plate_mass: float = DEFAULT_BASE_PLATE_MASS_KG,
                 tip_mass: float = DEFAULT_TIP_MASS_KG,
                 timestep: float = TIMESTEP_S,
                 integrator: str = INTEGRATOR,
                 iterations: int = SOLVER_ITERATIONS,
                 gravity=DEFAULT_GRAVITY,
                 joint_range_rad: float = JOINT_RANGE_RAD,
                 joint_armature: float = JOINT_ARMATURE,
                 force_range_n: float = FORCE_RANGE_N,
                 params=None,
                 prefix: str = "canarm_",
                 mount_body: str = "canarm_mount",
                 mocap_mount: bool = True,
                 actuator_prefix: str = "",
                 class_name: str = "umarm",
                 **arm_kwargs) -> str:
    """The twin's complete MJCF document.

    Every tunable is a keyword with a named default, per ``CONTRACT.md`` section
    3, and the reason is worth repeating: the RS485 generator kept its masses and
    lengths at module scope so a re-fit would be an explicit constant change,
    which works when one arm is being fitted and fails as soon as a second plant
    exists.  Here the same generator has to draw a 265/234/230 mm CAN arm today
    and whatever the next re-plumbing produces, so the numbers are arguments and
    the *defaults* carry the provenance.

    The argument list is echoed into the document's header comment, so a shipped
    XML always names the parameters it was built with -- the property that makes
    an anomalous rollout traceable to a scene rather than to a guess.
    """
    bodies, tendons, actuators = build_arm_xml(
        prefix=prefix, mount_body=mount_body, mocap_mount=mocap_mount,
        actuator_prefix=actuator_prefix, params=params,
        base_pos=base_pos, base_rpy_deg=base_rpy_deg,
        link_density=link_density, plate_mass=plate_mass,
        bracket_mass=bracket_mass, actuator_mass=actuator_mass,
        base_plate_mass=base_plate_mass, tip_mass=tip_mass,
        force_range_n=force_range_n, class_name=class_name, **arm_kwargs)

    genparams = ", ".join((
        f"base_pos={tuple(float(v) for v in base_pos)}",
        f"base_rpy_deg={tuple(float(v) for v in base_rpy_deg)}",
        f"joint_damping={joint_damping:g}",
        f"joint_frictionloss={joint_frictionloss:g}",
        f"tendon_damping={tendon_damping:g}",
        f"link_density={link_density:g}",
        f"plate_mass={plate_mass:g}",
        f"bracket_mass={bracket_mass:g}",
        f"actuator_mass={actuator_mass:g}",
        f"timestep={timestep:g}",
        f"force_range_n={force_range_n:g}",
        f"proximal_order={PROXIMAL_ORDER}",
        f"params={'measured 2026-08-21' if params is None else 'caller-supplied'}",
    ))
    return _SCENE_TEMPLATE.format(
        genparams=genparams,
        gravity=" ".join(_g(v) for v in gravity),
        timestep=f"{float(timestep):.17g}",
        integrator=integrator,
        iterations=int(iterations),
        class_name=class_name,
        joint_damping=_g(joint_damping),
        joint_frictionloss=_g(joint_frictionloss),
        tendon_damping=_g(tendon_damping),
        armature=_g(joint_armature),
        joint_range=f"{float(joint_range_rad):.17g}",
        bodies=bodies, tendons=tendons, actuators=actuators)


def generate_scene(out_path=None, **kwargs):
    """Write :func:`generate_xml` to *out_path*, or return it as a string."""
    xml = generate_xml(**kwargs)
    if out_path is None:
        return xml
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return path


def build_model(xml: str = None, **kwargs):
    """A compiled ``mujoco.MjModel``, from *xml* if given else from the tunables.

    ``xml=`` is the ``build_room_scene``-compatible seam ``CONTRACT.md`` section
    3 asks for: a merged multi-robot room is composed elsewhere and handed in,
    so the arm is authored in exactly one place and the merge cannot drift from
    it.  ``sim_core`` takes ``xml=`` for the same reason and should pass it
    straight through.
    """
    import mujoco

    return mujoco.MjModel.from_xml_string(generate_xml(**kwargs)
                                          if xml is None else xml)


def tendon_rest_lengths(model, prefix: str = "", actuator_prefix: str = ""):
    """``(24,)`` tendon lengths at ``qpos = 0``, in tendon-name order.

    The anchor the force law needs: ``CONTRACT.md`` section 1 defines muscle
    length as ``l = l0_seg + (ten_length - tendon_length0)``, so the twin's
    ``l`` is only as good as the configuration this was taken at.  Taken at
    ``qpos = 0`` -- the arm hanging straight -- because that is the pose the
    2026-08-21 campaign's baselines were recorded in, not because it is where
    the muscles are unstretched.
    """
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
           for name in tendon_names(actuator_prefix)]
    if min(ids) < 0:
        raise ValueError("this model does not carry the 24 pam_* tendons")
    return np.array([data.ten_length[i] for i in ids], dtype=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    ap.add_argument("--joint-damping", type=float, default=DEFAULT_JOINT_DAMPING)
    ap.add_argument("--joint-frictionloss", type=float,
                    default=DEFAULT_JOINT_FRICTIONLOSS)
    ap.add_argument("--tendon-damping", type=float, default=DEFAULT_TENDON_DAMPING)
    ap.add_argument("--link-density", type=float, default=DEFAULT_LINK_DENSITY_KG_M)
    ap.add_argument("--plate-mass", type=float, default=DEFAULT_PLATE_MASS_KG)
    ap.add_argument("--actuator-mass", type=float, default=DEFAULT_ACTUATOR_MASS_KG)
    args = ap.parse_args(argv)
    xml = generate_xml(joint_damping=args.joint_damping,
                       joint_frictionloss=args.joint_frictionloss,
                       tendon_damping=args.tendon_damping,
                       link_density=args.link_density,
                       plate_mass=args.plate_mass,
                       actuator_mass=args.actuator_mass)
    if args.out:
        print(f"wrote {generate_scene(args.out, joint_damping=args.joint_damping, joint_frictionloss=args.joint_frictionloss, tendon_damping=args.tendon_damping, link_density=args.link_density, plate_mass=args.plate_mass, actuator_mass=args.actuator_mass)}")
    else:
        print(xml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
