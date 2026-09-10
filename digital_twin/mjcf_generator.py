r"""The CAN arm's simulation MJCF -- the single source of the twin's geometry.

The display model in ``viz/mjcf_canarm.py`` answers "where are the links, given
q" and deliberately carries no actuators, no tendons and no fitted dynamics.
This one is the other half: the same kinematic chain plus the twenty-four muscle
tendons, the masses, and the small number of MuJoCo-side tunables the twin is
allowed to fit.  They are separate files because they fail differently -- a
display that is slightly wrong is a picture, a twin that is slightly wrong is a
controller tuned against a lie -- and because the display must stay cheap enough
to compile in a spawned process.  **The drawing, however, exists once**: the
display calls :func:`promax_segment_elements` and :func:`promax_base_elements`
from here, so the room viewer and the twin cannot disagree about where a Y arm,
a bearing or an actuator sits.

WHAT CAME ACROSS FROM ``viz/mjcf_canarm.py`` UNTOUCHED: the topology (three link
bodies, three distal-plate bodies, twelve hinges, distal axes
``(+-1, 1, 0)/sqrt(2)``), the JD spacer belonging to the *parent* body, the six
``canarm_plate0..5`` sites, and the mocap mount.  That geometry is verified
against ``UMArm_KINEMATICS.fkine`` to float noise over 200 random configurations,
so re-deriving it here would only be a chance to get it wrong.

FIVE THINGS DIFFER FROM THAT FILE'S ORIGINAL SIMPLE ARM, and each one is a
measurement or a contract clause:

1. **The proximal hinges are declared y-then-x.**  MuJoCo composes same-body
   hinges in declaration order, so declaring ``x`` first asserts fkine's
   ``order="xy"``.  The CAN arm's measured assembly is ``order="yx"``
   (``UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER``, 2026-08-21: it took held-out
   u-joint-centre error from 4.6 mm RMS / 20.4 mm worst to 2.0 / 7.0 mm).  Over
   200 random ``q`` in +-0.6 rad the two orders disagree by up to 0.189 m at a
   u-joint centre, so this is not a cosmetic swap.  The cost is that ``qpos`` is
   no longer ``q``: see :data:`QPOS_FROM_Q` and :func:`q_to_qpos`.
2. **The ring radii come from the parameter table.**  ``CANARM_PARAMS`` gives
   ``JA1 = JA2 = 0.047`` m (the u-joint outer ring, where the tendon brackets
   sit) and ``AO1 = AO2 = 0.028`` m (the routing-bearing radius) on all three
   segments.  In a twin these radii *are* the muscle moment arm.
3. **The ring azimuths are derived from the measured actuator map**, not
   transcribed.  See :func:`actuator_seats` and the section below.
4. **Masses exist**, as a per-segment model the fit can drive
   (``link_mass_kg``, ``bracket_mass_kg``) with every default's provenance named
   in :data:`MASS_PROVENANCE` -- which is still not a measurement.
5. **Timestep 1 ms, ``iterations=100``.**  ``digital_twin.TIMESTEP_S`` pins 1 ms
   because ``sim_core`` counts node-logic passes in whole quanta; every consumer
   must read ``model.opt.timestep`` back rather than assume it.

------------------------------------------------------------------------------
THE PROMAX SEGMENT, AS FIG. 1C OF THE PAPER DRAWS IT
------------------------------------------------------------------------------
Zuo et al., *Real-Time Compliance and Position Control of a Hyper-redundant Soft
Robotic Arm* (``2606.29731v1.pdf``), Fig. 1C shows one segment in section and
from outside; Sec. II.A: "the center link connects to the inner ring, the
bearings sit in the middle ring, and the tendon attachments load the outer ring
... The V-shaped actuator supports route longer McKibben actuators around the
joint while offsetting neighboring muscles to reduce interference; braided
carbon-fiber rods reinforce these supports".  Read against that figure and the
operator's description, **everything below except the u-joint outer rings is
rigid with the segment's centre rod** -- the link body, from its proximal u-joint
centre (``z = 0``) down to its distal u-joint centre (``z = -S``):

* **Two upright Ys** in perpendicular planes, at the :data:`UPPER_SEAT_DEG`
  azimuths (45/135/225/315 deg).  Their V arms root on the rod at the *top*
  bearing hub and spread up and out to tips near the proximal joint plane, so
  the proximal u-joint sits inside that cone; the stem is the rod.  The four
  actuators hanging from those tips drive the **distal** u-joint: each runs
  toward the *bottom* hub, and its tendon continues around a routing bearing
  there (radius ``AO2``, ``AA2`` above the distal centre) to the distal
  u-joint's outer-ring bracket (radius ``JA2``, in the distal joint plane, on
  the distal plate body).
* **Two upside-down Ys**, 45 deg round from the upright pair, at the
  :data:`LOWER_SEAT_DEG` azimuths (0/90/180/270 deg): V arms root at the bottom
  hub and spread down and out around the distal u-joint.  Their four actuators
  drive the **proximal** u-joint: tendon from the actuator's free end, around a
  bearing on the top hub (``AO1``, ``AA1`` below the proximal centre), to the
  proximal u-joint's outer ring (``JA1``, in the proximal joint plane, on the
  **parent** body).
* The u-joints are drawn as flat disks of the outer-ring radius, on the bracket
  side of the joint.

The actuator is ``LL`` long (the measured rest length the force law's ``l0``
seeds from), hung from its Y tip along the line to its bearing.  With the tip at
:data:`DEFAULT_Y_TIP_RADIUS_M` in the joint plane that leaves 51-52 mm of free
tendon between the sleeve and the bearing on all three segments, which is what
Fig. 1C shows; the figure was used to estimate the tip radius and nothing in the
routing that sets a moment arm depends on it.

------------------------------------------------------------------------------
HOW THE TENDONS ARE ROUTED, AND WHY THE CONSTANT PART DOES NOT MATTER
------------------------------------------------------------------------------
Each tendon is a three-site spatial path: actuator free end (link) -> routing
bearing (link) -> outer-ring bracket (the body across the joint).  The first
span joins two sites on one rigid body, so its length is a constant; every
consumer of tendon length uses the excursion ``ten_length - tendon_length0``
(``sim_core.SimArm._step_quantum`` and ``_node_pass``, ``dataset.TendonKinematics
.dlen``, ``actuator_model.ActuatorModel.force_n``), so that constant cancels
exactly and the physics sees only the bearing-to-bracket span.  The first span
is there so the drawn tendon runs from the sleeve, not from mid-air.

For a proximal muscle at azimuth ``phi`` the bearing sits at
``(AO cos phi, AO sin phi, -AA)`` in the link frame and the bracket at
``(JA cos phi, JA sin phi, 0)`` in the parent; rotating the link by ``t`` about a
unit axis ``u`` through the joint centre gives, at ``q = 0``::

    d(len)/dt = -u . (p x b) / ell,   p x b = (AA JA sin phi, -AA JA cos phi, 0)
    ell = sqrt((JA - AO)^2 + AA^2)

and the distal muscle gives the same expression with the bracket rotating and
the bearing ``AA2`` above the distal centre.  So the moment arm about the one
axis a pure seat drives is ``AA*JA/ell`` = **43.105 mm** on this arm, and about
the other axis of that u-joint it is identically zero.

WHAT THIS REPLACED, AND WHY THE SEAT TABLE SURVIVES IT.  Until 2026-09-10 each
proximal muscle ran from the parent's plate ring to the link's *far* ring,
``AA1 + LL`` (222 mm on segment 1) below the proximal centre, and each distal
muscle from the link's top ring to the distal plate; the muscles were drawn as
capsules hanging off the u-joint plates, which is what made the Y arms look
connected to the u-joint in the 2026-09-10 deliverable video.  That routing gave
``(AA+LL)*JA/sqrt((JA-AO)^2+(AA+LL)^2)`` = 46.8 mm at ``q = 0``, *falling* both
ways (42.1 mm at +30 deg, 39.4 mm at -30 deg on segment 1).  The bearing
routing gives 43.1 mm at rest and an asymmetric arm: it *rises* to 47.0 mm as the
muscle shortens through +30 deg and falls to 35.6 mm as it lengthens through
-30 deg, identically on all three segments.  Over the measured poses of the 90 s
held-out validation recording the two excursions differ by little more than a
scale -- new = 0.930 x old, residual 0.22 mm RMS against 9.61 mm -- so a force
law fitted on the old routing is mostly mis-scaled on the new one, not
mis-shaped.  Both routings put the link-side point on the link
side of the joint plane at the seat azimuth, so ``p x b`` has the same sign in
each: **every muscle pulls its joint the same way under both**, and
:data:`LOWER_SEAT_DEG`/:data:`UPPER_SEAT_DEG` carry over unchanged.  (The
proximal table's own 2026-09-10 history -- swapped by ``a35f94a`` to hide a
``qpos``-as-``q`` read, and swapped back once the read was fixed -- is written
out under :data:`LOWER_SEAT_DEG`.)

------------------------------------------------------------------------------
THE SEAT TABLE IS COMPUTED RATHER THAN TYPED
------------------------------------------------------------------------------
A muscle pulls, so it drives a joint in whichever direction shortens it.  Each of
the eight (axis, sign) roles therefore has exactly one azimuth at which it is
pure, and those eight azimuths are :data:`LOWER_SEAT_DEG` and
:data:`UPPER_SEAT_DEG`.  Given that, the seat table is a *consequence* of
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
  ``tendon_damping`` carry the RS485 twin's fitted values as *defaults*; the
  fitted values for this arm live in checkpoint files and are handed in.
* **No mass on this arm has been weighed** except the operator's ~30 g per
  actuator.  The per-segment defaults are the Koopman ProMax MuJoCo model's
  prior (:data:`MASS_PROVENANCE`), which is a model, not a scale reading.
* **No passive joint stiffness has been measured.**  ``joint_stiffness``
  defaults to zero; a value a fit hands in stands for tubing and wiring
  elasticity nobody has isolated on the metal.
* **The ring radii and the bearing hub heights are CAD**, and the bearing is a
  point: a real tendon leaves a bearing wheel tangentially, so its effective
  routing point sits up to one wheel radius from the axle.  If ``AO``, ``AA`` or
  ``JA`` is wrong on the metal, every routing test still passes and every
  predicted torque is wrong by the same factor.
* **The Y tip radius and height are estimated from Fig. 1C**, not measured.
  They move the actuator's drawn position and its share of the link inertia;
  they do not enter any moment arm.
* **No contacts, no keyframes, no end effector.**  Every geom is
  ``contype="0" conaffinity="0"``; two arms drawn overlapping is a drawing.
* **The force clip is inherited, not derived.**  ``FORCE_RANGE_N = 4000`` was
  sized to the RS485 force law's envelope; ``replay.assert_no_clamp`` is what
  will report it being wrong.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\mjcf_generator.py``
REFERENCE READ:  ``digital_twin/reference/mjcf_fit.md`` section 1.
PROMAX PRIOR:    ``C:\RUNZE_SRC\UMArm_dynamic_koopman_compliance\runze_trying_MPC\
                 real_system_fitting _experiment\mujoco_fit\generator.py`` with
                 ``robot_config.py`` configuration ``"original"`` (read-only).
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
    "DEFAULT_TENDON_DAMPING", "DEFAULT_JOINT_STIFFNESS",
    "DEFAULT_LINK_DENSITY_KG_M",
    "DEFAULT_PLATE_MASS_KG", "DEFAULT_SPACER_MASS_KG",
    "DEFAULT_ACTUATOR_MASS_KG", "DEFAULT_Y_MASS_KG", "DEFAULT_HUB_MASS_KG",
    "DEFAULT_LINK_MASS_KG", "DEFAULT_BRACKET_MASS_KG",
    "KOOPMAN_PROMAX_LINK_MASS_KG", "KOOPMAN_PROMAX_UJOINT_MASS_KG",
    "DEFAULT_BASE_PLATE_MASS_KG", "DEFAULT_TIP_MASS_KG",
    "DEFAULT_Y_TIP_RADIUS_M", "DEFAULT_Y_TIP_Z_OFFSET_M",
    "DEFAULT_ACTUATOR_RADIUS_M",
    "DEFAULT_BASE_POS", "DEFAULT_BASE_RPY_DEG",
    "FORCE_RANGE_N", "JOINT_RANGE_RAD", "JOINT_ARMATURE",
    "Seat", "SegGeom", "SegmentMasses", "actuator_seats", "actuator_names",
    "tendon_names", "joint_names", "plate_site_names", "actuator_joint_map",
    "routing_site_names", "segment_geometry", "params_from_chain",
    "bearing_moment_arm_m", "resolve_per_segment",
    "resolve_per_segment_nonnegative", "segment_masses",
    "fitted_params", "q_to_qpos", "qpos_to_q",
    "promax_base_elements", "promax_segment_elements",
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
#: which spans a proximal universal joint.  At each of these four azimuths the
#: *other* axis's moment arm is identically zero, so a muscle seated here drives
#: one axis and only one.  These are also the azimuths of the two **upside-down
#: Ys** whose tips carry these four actuators.
#:
#: **This is the closed form, and it is the physical assignment.**  A muscle at
#: ring azimuth ``th`` pulling along -z has moment ``(-F r sin th, +F r cos th,
#: 0)``, so 90 deg is pure about +x and 180 deg pure about +y: axis 0 (``t1``,
#: about +x, hinge ``uj*_x`` at ``qpos[QPOS_FROM_Q[0]] = qpos[1]``) takes the
#: 90/270 pair and axis 1 (``t2``, about +y) the 180/0 pair.  On the compiled
#: model all 12 proximal muscles drive the hinge their measured joint lives on,
#: with the measured sign, at the 43.104 mm closed-form arm
#: (``test_mjcf.test_each_muscle_drives_the_hinge_its_measured_joint_lives_on``).
#:
#: HISTORY, BECAUSE THIS TABLE WAS ONCE "CORRECTED" THE WRONG WAY.  Commit
#: ``a35f94a`` swapped the two pairs to ``{(0,+1): 180, (0,-1): 0, (1,+1): 90,
#: (1,-1): 270}`` after rolling the twin against the arm gave a cross-correlation
#: matrix that was a permutation with three transpositions, all of them proximal
#: (the twin's ``j0`` tracked the arm's ``j1`` at +0.93, and so on).  The
#: permutation was real, but its cause was the read, not the seats:
#: ``sim_core.SimArm.q()`` returned ``data.qpos`` raw and ``replay`` recorded it
#: as ``q``, so the twin's column 0 was the *y* hinge, which the 90/270 muscles
#: do not move.  Rotating the proximal muscles 90 deg made the columns line up
#: while every proximal muscle tilted its link about the wrong world axis; under
#: that table 0 of the 12 drove their measured hinge.  The cost is not cosmetic.
#: The distal u-joint's axes sit at +-45 deg, and a parent tilt about x loads
#: them with gravity as ``(-, +)`` where a tilt about y loads them as ``(-, -)``,
#: so one distal axis per segment was pushed by gravity the wrong way whenever
#: its proximal joint moved.  Measured on the 90 s held-out validation sequence
#: with the same flow net and outer fit on the bearing routing (fit agent,
#: 2026-09-10):
#:
#: ===============================  =========  =====  ==============================
#: seats / read                     joint RMS  nrmse  mean corr (seg 1 / 2 / 3)
#: ===============================  =========  =====  ==============================
#: a35f94a table, raw ``qpos``      11.746 deg 1.045  +0.737 (+0.836 / +0.888 / +0.485)
#: this table, ``q`` by hinge name   9.691 deg 0.869  +0.838 (+0.960 / +0.945 / +0.610)
#: this table, raw ``qpos``         14.314 deg 1.312  +0.392 (the original permutation)
#: ===============================  =========  =====  ==============================
#:
#: The third row reproduces the transpositions exactly, which is what shows the
#: read was the cause.  ``SimArm.q()`` and ``dataset.TendonKinematics.dlen`` now
#: resolve the hinges by name in ``q`` order, so a seat table and a read can no
#: longer cancel each other's error without a test noticing.
LOWER_SEAT_DEG = {(0, +1): 90.0, (0, -1): 270.0, (1, +1): 180.0, (1, -1): 0.0}

#: The same for an **upper** muscle, which spans a distal universal joint whose
#: axes are the 45-degree bracket vectors ``(x+y)/sqrt2`` (axis 2 = ``t3``) and
#: ``(-x+y)/sqrt2`` (axis 3 = ``t4``).  The four azimuths sit 45 degrees round
#: from the lower ring's, which is the bracket rotation and nothing else; they
#: are also the azimuths of the two **upright Ys**.
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
#: rest of the model.  The Koopman ProMax model also carries 2.25.
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
#: has no excitation of its own.
DEFAULT_TENDON_DAMPING = 1.0

#: N*m/rad per hinge, per segment ``(seg1, seg2, seg3)``, applied to all four
#: hinges of that segment (its proximal and its distal u-joint), with the spring
#: at rest at ``q = 0``, the arm hanging straight.  **Zero, and not measured**:
#: it stands for the passive elastic restoring torque the pneumatic tubing and
#: the busline wiring put across every u-joint, which no experiment on this arm
#: has isolated.  Added 2026-09-10 because the first mechanical fit, which had no
#: such term, leaned on its bounds exactly where the model is short of restoring
#: torque (``bf/l0`` at its upper bound on all three segments, a 0.60 kg segment-2
#: bracket, a 1.13 kg segment-3 link), and because joints 10 and 11 carry only
#: 0.005 N*m/rad of gravity stiffness, so without it only antagonist
#: co-contraction holds the tip.  Zero reproduces every model built before that
#: date exactly; a fitted value arrives through the checkpoint.
DEFAULT_JOINT_STIFFNESS = (0.0, 0.0, 0.0)

# -- the per-segment mass model ---------------------------------------------

#: kg per segment, ``(seg1, seg2, seg3)``: **the Koopman ProMax MuJoCo model's
#: ``rod_mass_kg``**, which is the whole link body of that model *including its
#: eight actuators* (``robot_config.py`` configuration ``"original"``).  It is
#: the only mass model anyone has written down for the ProMax, and the operator
#: pointed at that checkpoint as "the MuJoCo that reflects the UMArm ProMax"; it
#: is a prior, not a weighing.
KOOPMAN_PROMAX_LINK_MASS_KG = (0.70, 0.50, 0.50)

#: kg per segment: the same model's ``ujoint_mass_kg``, the inter-segment
#: linkage body.  Its third entry, 0.001 kg, is a placeholder for a linkage that
#: does not exist after segment 3 -- that model gives its u-joint disks 0.1 g --
#: so it does not describe this model's segment-3 plate body, which carries the
#: last u-joint ring and plate 5.
KOOPMAN_PROMAX_UJOINT_MASS_KG = (0.20, 0.20, 0.001)

#: kg, total **link-body** mass per segment (rod + two bearing hubs + four Ys +
#: eight actuators), the ``link_mass_kg`` default.  The Koopman ProMax prior,
#: as is.  When a total is given, the eight sleeves keep ``actuator_mass`` each
#: -- the one mass measured on this arm -- and only the remainder, the
#: *structure*, is spread over the rod, hubs and Ys in proportion to their
#: component defaults below.  Until the 2026-09-10 refit the whole total was
#: spread, so a 0.70 kg prior link carried 43 g sleeves against the measured 30 g.
DEFAULT_LINK_MASS_KG = KOOPMAN_PROMAX_LINK_MASS_KG

#: kg, total **bracket-body** mass per segment -- everything rigid with that
#: segment's distal u-joint outer ring: the ring/plate disk, and for segments 1
#: and 2 the JD spacer and the next segment's proximal ring.  Segments 1 and 2
#: are the Koopman prior's 0.20 kg connectors.  Segment 3's 0.10 kg is an
#: **estimate** (half a connector: the ring and plate 5 without a spacer), made
#: because the prior's 0.001 kg is a no-linkage placeholder.  The tip stub's
#: :data:`DEFAULT_TIP_MASS_KG` rides on that body and is **not** included.
DEFAULT_BRACKET_MASS_KG = (0.20, 0.20, 0.10)

#: kg/m of centre rod.  Provenance: the RS485 twin's rod geoms, 0.02 kg over a
#: ~0.20 m span.  With ``link_mass_kg`` given this is a *proportion* (it decides
#: the rod's share of the link), and it is the smallest share: 0.027 kg of the
#: 0.49 kg component link on segment 1.
DEFAULT_LINK_DENSITY_KG_M = 0.10

#: kg per bearing hub, with its four steel-shaft routing bearings.  **Estimated
#: from Fig. 1C, not measured** -- an aluminium hub about 50 mm across; a
#: proportion when ``link_mass_kg`` is given.
DEFAULT_HUB_MASS_KG = 0.030

#: kg per Y (V-struct) support: two aluminium sandwich arms, the tip blocks and
#: the braided carbon braces.  **Estimated from Fig. 1C, not measured**; a
#: proportion when ``link_mass_kg`` is given.
DEFAULT_Y_MASS_KG = 0.040

#: kg per McKibben actuator.  **The only measured mass on this arm**: the
#: operator's ~30 g each, valves and regulator PCB inside the sleeve, 24 of
#: them.  All of it sits on the link body, because on the ProMax the sleeve hangs
#: from a Y arm that is rigid with the centre rod; the tendon is what crosses the
#: joint.  Honoured exactly whether or not ``link_mass_kg`` is given; a link
#: total below ``8 * actuator_mass`` is refused.
DEFAULT_ACTUATOR_MASS_KG = 0.030

#: kg per u-joint ring/plate disk.  Provenance: the RS485 twin's
#: ``PLATE3_MASS = 0.12``, the only plate mass anyone in either workspace put a
#: number on; a proportion when ``bracket_mass_kg`` is given.
DEFAULT_PLATE_MASS_KG = 0.12

#: kg per JD spacer (the connector between one segment's distal ring and the
#: next segment's proximal ring).  Provenance: half the RS485 twin's
#: ``BRACKET_MASS = 0.48``; a proportion when ``bracket_mass_kg`` is given.
DEFAULT_SPACER_MASS_KG = 0.24

#: kg of the base plate.  It hangs off a static mount, so it enters no equation
#: of motion; it is here so the drawn disc has the right heft and so the number
#: is not silently zero.  Provenance: ``viz/mjcf_canarm.py``.
DEFAULT_BASE_PLATE_MASS_KG = 0.5

#: kg of the 50 mm tip stub past the last plate.  Provenance:
#: ``viz/mjcf_canarm.py``.  It is a stub, not an end effector -- no tool is
#: installed on this arm.
DEFAULT_TIP_MASS_KG = 0.02

# -- drawing geometry that is not in the parameter table ---------------------

#: m, radius of a Y arm's tip, where the actuator hangs.  **Estimated from
#: Fig. 1C, not measured**: the tip blocks sit about 250 px from the spine in
#: the section view, whose segment spans about 800 px.  Supporting check: with
#: the tip in the joint plane at this radius, an actuator ``LL`` long leaves
#: 51-52 mm of tendon to its bearing on all three segments, the proportion the
#: figure shows.  Enters no moment arm.
DEFAULT_Y_TIP_RADIUS_M = 0.085

#: m, how far past its u-joint plane a Y tip sits (positive = away from the
#: segment: above the proximal plane for an upright Y, below the distal plane for
#: an upside-down one).  Fig. 1C puts the tip blocks within about 15 mm of the
#: plane; 0 is chosen and **not measured**.  Enters no moment arm.
DEFAULT_Y_TIP_Z_OFFSET_M = 0.0

#: m, drawn radius of a McKibben sleeve.  Fig. 1C's sleeves are about 20 mm
#: across; the radius also shapes the capsule's inertia about its own axis,
#: which is negligible beside its offset from the rod.
DEFAULT_ACTUATOR_RADIUS_M = 0.0095

#: Where the arm hangs when nothing writes its mount.  Arbitrary but not zero:
#: the arm hangs base-up, and a twin at the floor would ring against gravity
#: pointing the wrong way through the chain.  Same value as
#: ``viz.mjcf_canarm.DEFAULT_CANARM_MOUNT``, so a twin and the display sit in
#: the same place in a merged room.
DEFAULT_BASE_POS = (0.0, 0.0, 1.25)
DEFAULT_BASE_RPY_DEG = (0.0, 0.0, 0.0)

#: N.  Pull-only by construction (``ctrlrange`` and ``forcerange`` are both
#: ``[-FORCE_RANGE_N, 0]``); a McKibben cannot push.  The magnitude is the RS485
#: twin's, sized to *its* force law.  ``replay.assert_no_clamp`` exists because a
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
    "link_mass_kg": (
        "Koopman ProMax MuJoCo prior, robot_config 'original' rod_mass_kg "
        "0.70/0.50/0.50 kg = whole link incl. 8 actuators; spread over rod, hubs, "
        "Ys and actuators in proportion to the component defaults -- NOT measured"),
    "bracket_mass_kg": (
        "Koopman ProMax prior ujoint_mass_kg 0.20/0.20 kg for the two inter-segment "
        "connectors; segment 3's 0.001 kg there is a no-linkage placeholder, so 0.10 kg "
        "(ring + plate 5, no spacer) is an estimate -- NOT measured; excludes tip stub"),
    "link_density_kg_m": "RS485 twin rod geoms, 0.02 kg over ~0.20 m span (a proportion)",
    "hub_mass_kg": "estimate from Fig. 1C: one bearing hub + 4 bearings, NOT measured (a proportion)",
    "y_mass_kg": "estimate from Fig. 1C: one V-struct support + carbon braces, NOT measured (a proportion)",
    "actuator_mass_kg": (
        "operator, 2026-08: ~30 g per McKibben -- MEASURED, this arm; full mass on the "
        "link body (sleeve rigid with the rod); honoured exactly, a link_mass_kg total "
        "only sets the structure (rod + hubs + Ys) beside it"),
    "plate_mass_kg": "RS485 twin PLATE3_MASS = 0.12 kg, carried over on radius (a proportion)",
    "spacer_mass_kg": "half RS485 twin BRACKET_MASS = 0.48 kg (a proportion)",
    "base_plate_mass_kg": "viz/mjcf_canarm.py base disc, static body",
    "tip_mass_kg": "viz/mjcf_canarm.py tip stub; no end effector is installed",
}

_SQ2 = math.sqrt(2.0) / 2.0

# Colours.  One hue per kind of part, chosen so a render separates them at a
# glance: carbon rod, aluminium Ys, grey hubs with bright bearings, translucent
# blue u-joint disks with dark-blue brackets, two sleeve tints (the four that
# drive the proximal joint against the four that drive the distal one), and
# orange tendons.
RGBA_ROD = "0.13 0.13 0.15 1"
RGBA_Y = "0.86 0.84 0.78 1"
RGBA_Y_TIP = "0.93 0.90 0.80 1"
RGBA_BRACE = "0.42 0.42 0.46 1"
RGBA_HUB = "0.50 0.53 0.58 1"
RGBA_BEARING = "0.88 0.88 0.92 1"
RGBA_DISK = "0.50 0.70 0.95 0.60"
RGBA_BRACKET = "0.18 0.36 0.70 1"
RGBA_SPACER = "0.40 0.44 0.52 1"
RGBA_SLEEVE_PROXIMAL = "0.46 0.26 0.24 1"
RGBA_SLEEVE_DISTAL = "0.22 0.32 0.44 1"
RGBA_TENDON = "1.0 0.50 0.05 1"
RGBA_BASE = "0.30 0.35 0.45 1"


# ---------------------------------------------------------------------------
# The seat table, derived from the measured actuator map
# ---------------------------------------------------------------------------

class Seat(NamedTuple):
    """Where one muscle is anchored, and what it drives.

    ``ring`` is ``"lower"`` (spans the segment's proximal universal joint; its
    actuator hangs from an upside-down Y) or ``"upper"`` (spans its distal one;
    its actuator hangs from an upright Y); ``joint`` indexes ``q`` in
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


def routing_site_names(seat: Seat, prefix: str = "canarm_") -> dict:
    """The four named sites of one muscle, by board index.

    ``end`` is the actuator's free end (link body), ``brg`` the routing bearing
    (link body, ``AA`` from the joint the muscle drives), ``brk`` the outer-ring
    bracket (the body across the joint), ``tip`` the Y tip the actuator hangs
    from (link body).  Named by board rather than by ring position, so a site
    name answers "which board is this" without a lookup, and a routing mistake
    shows up as a name that does not exist rather than as a plausible wrong
    tendon.
    """
    base = f"{prefix}s{seat.segment}_a{seat.index}"
    return {"end": f"{base}_end", "brg": f"{base}_brg",
            "brk": f"{base}_brk", "tip": f"{base}_tip"}


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

class SegGeom(NamedTuple):
    """Every length one segment's bodies, hubs and rings hang off, in metres.

    Frames: ``jd`` is in the *parent* body (the previous segment's distal plate,
    or the base); everything else is in the segment's own link body, whose
    origin is the proximal u-joint centre.  ``UC1 = UC2 = 0`` on this hardware,
    which is what lets the hinges sit at the body origins.
    """

    jd: float               # spacer above the proximal centre, in the parent
    span: float             # proximal centre -> distal centre
    ja1: float              # proximal outer-ring (bracket) radius
    ja2: float              # distal outer-ring (bracket) radius
    ao1: float              # top-hub bearing radius (proximal muscles)
    ao2: float              # bottom-hub bearing radius (distal muscles)
    aa1: float              # proximal centre -> top hub
    aa2: float              # bottom hub -> distal centre
    ll: float               # measured actuator rest length
    z_top_hub: float        # link frame, = -(UC1 + AA1)
    z_bot_hub: float        # link frame, = -(span - UC2 - AA2)


def segment_geometry(params=None) -> list:
    """``[SegGeom, SegGeom, SegGeom]`` from a ``(3, 10)`` parameter table."""
    params = cp.require_measured() if params is None else params
    out = []
    for row in np.asarray(params, dtype=float):
        uc1 = float(row[rp.COL_UC1])
        uc2 = float(row[rp.COL_UC2])
        aa1 = float(row[rp.COL_AA1])
        aa2 = float(row[rp.COL_AA2])
        ll = float(row[rp.COL_LL])
        span = uc1 + aa1 + ll + aa2 + uc2
        if span <= 0.0:
            raise ValueError(f"segment span must be positive; got {span}")
        out.append(SegGeom(
            jd=float(row[rp.COL_JD]), span=span,
            ja1=float(row[rp.COL_JA1]), ja2=float(row[rp.COL_JA2]),
            ao1=float(row[rp.COL_AO1]), ao2=float(row[rp.COL_AO2]),
            aa1=aa1, aa2=aa2, ll=ll,
            z_top_hub=-(uc1 + aa1), z_bot_hub=-(span - uc2 - aa2)))
    return out


def params_from_chain(chain_m, template=None) -> np.ndarray:
    """A ``(3, 10)`` table whose u-joint-centre chain is *chain_m*.

    ``chain_m`` is ``(span1, JD2, span2, JD3, span3)``.  The CAD columns
    (``JA``, ``AO``, ``AA``, ``UC``) come from *template* (default
    ``CANARM_PARAMS``) and ``LL`` absorbs the difference, the same convention
    ``canarm_params`` uses to turn the measured chain into its table.  Exists so
    the display model, which is argument-driven by the chain, can draw the same
    ProMax segment the twin simulates.  For the measured chain it reproduces
    ``CANARM_PARAMS`` to float rounding.
    """
    c = [float(v) for v in chain_m]
    if len(c) != 5:
        raise ValueError(f"chain_m must have 5 entries; got {len(c)}")
    out = np.array(cp.CANARM_PARAMS if template is None else template,
                   dtype=float, copy=True)
    spans = (c[0], c[2], c[4])
    jds = (0.0, c[1], c[3])
    for i in range(3):
        out[i, rp.COL_LL] = (spans[i] - out[i, rp.COL_UC1] - out[i, rp.COL_AA1]
                             - out[i, rp.COL_AA2] - out[i, rp.COL_UC2])
        out[i, rp.COL_JD] = jds[i]
    return out


def bearing_moment_arm_m(aa: float, ja: float, ao: float) -> float:
    """``AA*JA / sqrt((JA-AO)^2 + AA^2)`` -- a pure seat's moment arm at ``q = 0``.

    The closed form of the module docstring, exposed so a test and an analysis
    script compare against one expression rather than two transcriptions.
    43.105 mm on this arm.
    """
    return float(aa) * float(ja) / math.hypot(float(ja) - float(ao), float(aa))


def resolve_per_segment(value, name: str, n: int = 3):
    """``None`` -> ``None``; a number -> ``(v,)*n``; a sequence -> ``n`` floats.

    Every entry must be finite and positive, because a zero-mass body with a
    hinge is a model MuJoCo will compile and then integrate into nonsense.
    """
    if value is None:
        return None
    if np.ndim(value) == 0:
        out = (float(value),) * n
    else:
        out = tuple(float(v) for v in value)
    if len(out) != n:
        raise ValueError(f"{name} must be a number or {n} numbers; got {value!r}")
    if not all(math.isfinite(v) and v > 0.0 for v in out):
        raise ValueError(f"{name} entries must be finite and positive; got {out}")
    return out


def resolve_per_segment_nonnegative(value, name: str, n: int = 3) -> tuple:
    """A number or ``n`` numbers -> ``n`` floats, each finite and ``>= 0``.

    The per-segment form of a tunable whose zero is meaningful -- a joint
    stiffness of zero is the model without the term -- so, unlike
    :func:`resolve_per_segment`, it accepts zero and has no ``None`` form.
    """
    if value is None:
        raise ValueError(f"{name} takes a number or {n} numbers, not None")
    out = ((float(value),) * n if np.ndim(value) == 0
           else tuple(float(v) for v in value))
    if len(out) != n:
        raise ValueError(f"{name} must be a number or {n} numbers; got {value!r}")
    if not all(math.isfinite(v) and v >= 0.0 for v in out):
        raise ValueError(f"{name} entries must be finite and non-negative; got {out}")
    return out


def _ring_xy(radius: float, angle_deg: float) -> tuple:
    a = math.radians(angle_deg)
    return radius * math.cos(a), radius * math.sin(a)


def _g(v: float) -> str:
    return f"{float(v):.10g}"


def _v(p) -> str:
    return " ".join(_g(c) for c in p)


def _sub(a, b) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


def _lerp(a, b, s: float) -> tuple:
    return tuple(x + s * (y - x) for x, y in zip(a, b))


def _norm(a) -> float:
    return math.sqrt(sum(c * c for c in a))


def _unit(a) -> tuple:
    n = _norm(a)
    return tuple(c / n for c in a)


def _cross(a, b) -> tuple:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _quat_attr(rpy_deg) -> str:
    """Body quaternion from roll-pitch-yaw, via ``viz.transforms``.

    Routed through the same helper the display model uses so a twin and the
    display disagree about no angle convention, ever.
    """
    from viz import transforms as TF

    return " ".join(f"{float(v):.17g}" for v in TF.rpy_to_quat(rpy_deg))


# ---------------------------------------------------------------------------
# The ProMax drawing, shared with viz/mjcf_canarm.py
# ---------------------------------------------------------------------------

class SegmentMasses(NamedTuple):
    """kg per geom for one segment, after any per-segment total was applied."""

    rod: float
    hub: float              # each of two
    y_arm: float            # each of eight (a Y is two arms)
    actuator: float         # each of eight
    prox_disk: float        # on the parent body
    spacer: float           # on the parent body; unused when jd == 0
    dist_disk: float        # on this segment's distal plate body


def promax_base_elements(*, prefix: str, ring_radius: float,
                         base_plate_radius: float, base_plate_mass: float,
                         post_radius: float = 0.008,
                         mount_gap: float = 0.044,
                         plate_half_thickness: float = 0.008) -> list:
    """The static base's geoms: a mount plate lifted clear of the first u-joint.

    Lifted, not at ``z = 0``, because on the ProMax segment 1's upright Ys reach
    ``DEFAULT_Y_TIP_RADIUS_M`` in the first u-joint's plane, and a mount disk
    there would swallow their tips.  The first u-joint's own disk is drawn by
    :func:`promax_segment_elements` as segment 1's ``parent`` part.
    """
    top = mount_gap + 2.0 * plate_half_thickness
    return [
        f'<geom name="{prefix}base_geom" type="cylinder" '
        f'fromto="0 0 {_g(mount_gap)} 0 0 {_g(top)}" '
        f'size="{_g(base_plate_radius)}" rgba="{RGBA_BASE}" '
        f'mass="{_g(base_plate_mass)}"/>',
        f'<geom name="{prefix}base_post" type="cylinder" '
        f'fromto="0 0 0 0 0 {_g(mount_gap)}" size="{_g(post_radius)}" '
        f'rgba="{RGBA_SPACER}" mass="0"/>',
    ]


def promax_segment_elements(geo: SegGeom, seats, *, n: int, prefix: str,
                            masses: SegmentMasses,
                            y_tip_radius: float = DEFAULT_Y_TIP_RADIUS_M,
                            y_tip_z_offset: float = DEFAULT_Y_TIP_Z_OFFSET_M,
                            actuator_radius: float = DEFAULT_ACTUATOR_RADIUS_M,
                            rod_radius: float = 0.009,
                            hub_radius: float = 0.022,
                            hub_half_thickness: float = 0.005,
                            bearing_radius: float = 0.0065,
                            plate_half_thickness: float = 0.004,
                            spacer_radius: float = 0.008,
                            rod_rgba: str = RGBA_ROD,
                            routing_sites: bool = True,
                            tendon_stubs: bool = False) -> dict:
    """One ProMax segment as MJCF element strings, split by the body they go in.

    Returns ``{"parent": [...], "link": [...], "distal": [...]}``: ``parent``
    is emitted into the body above the proximal u-joint (the base, or the
    previous segment's distal plate body) *before* the link body opens,
    ``link`` into the link body after its hinges, and ``distal`` into the
    distal plate body after its hinges.  The caller owns the bodies, the hinges
    and the plate sites, which is what lets the twin and the display declare
    their hinges differently and still share every drawn part.

    *seats* are this segment's eight :class:`Seat` rows.  *routing_sites* emits
    the ``end``/``brg``/``brk``/``tip`` sites of :func:`routing_site_names` (in
    site group 3, hidden by default) -- the twin needs them for its tendons.
    *tendon_stubs* draws the actuator-end-to-bearing span as a thin capsule on
    the link body, for a display that carries no tendons; the bearing-to-bracket
    span crosses a joint and cannot be a fixed geom, so no model here draws it
    as one.

    Every geom is non-colliding by the caller's default class; masses are given
    explicitly so a per-segment total can be spread by the caller.
    """
    seats = tuple(seats)
    lower = [s for s in seats if s.ring == "lower"]
    upper = [s for s in seats if s.ring == "upper"]
    if len(lower) != 4 or len(upper) != 4:
        raise ValueError(f"segment {n} needs 4 lower and 4 upper seats; got "
                         f"{len(lower)} and {len(upper)}")
    parent, link, distal = [], [], []
    site_grp = ' group="3"'

    # -- parent side: the JD spacer and the proximal u-joint's outer ring ----
    if geo.jd:
        parent.append(
            f'<geom name="{prefix}jd{n}" type="cylinder" '
            f'fromto="0 0 0 0 0 {_g(-geo.jd)}" size="{_g(spacer_radius)}" '
            f'rgba="{RGBA_SPACER}" mass="{_g(masses.spacer)}"/>')
    parent.append(
        f'<geom name="{prefix}seg{n}_plate1_geom" type="cylinder" '
        f'fromto="0 0 {_g(-geo.jd - plate_half_thickness)} '
        f'0 0 {_g(-geo.jd + plate_half_thickness)}" size="{_g(geo.ja1)}" '
        f'rgba="{RGBA_DISK}" mass="{_g(masses.prox_disk)}"/>')

    # -- link body: rod, hubs, bearings -------------------------------------
    link.append(
        f'<geom name="{prefix}seg{n}_rod" type="cylinder" '
        f'fromto="0 0 0 0 0 {_g(-geo.span)}" size="{_g(rod_radius)}" '
        f'rgba="{rod_rgba}" mass="{_g(masses.rod)}"/>')
    for tag, z in (("top", geo.z_top_hub), ("bot", geo.z_bot_hub)):
        link.append(
            f'<geom name="{prefix}seg{n}_hub_{tag}" type="cylinder" '
            f'fromto="0 0 {_g(z - hub_half_thickness)} '
            f'0 0 {_g(z + hub_half_thickness)}" size="{_g(hub_radius)}" '
            f'rgba="{RGBA_HUB}" mass="{_g(masses.hub)}"/>')

    for s in seats:
        names = routing_site_names(s, prefix)
        proximal = s.ring == "lower"
        az = math.radians(s.azimuth_deg)
        radial = (math.cos(az), math.sin(az), 0.0)
        tangent = (-math.sin(az), math.cos(az), 0.0)
        ao = geo.ao1 if proximal else geo.ao2
        ja = geo.ja1 if proximal else geo.ja2
        # The bearing this muscle routes around: the top hub for a proximal
        # muscle (AA1 below the joint it drives), the bottom hub for a distal
        # one (AA2 above the joint it drives).  Opposite end of the segment from
        # the Y tip its actuator hangs on.
        z_hub = geo.z_top_hub if proximal else geo.z_bot_hub
        brg = (ao * radial[0], ao * radial[1], z_hub)
        # The Y tip: near the distal plane for an upside-down Y (proximal
        # muscle), near the proximal plane for an upright one (distal muscle).
        z_tip = (-geo.span - y_tip_z_offset) if proximal else y_tip_z_offset
        tip = (y_tip_radius * radial[0], y_tip_radius * radial[1], z_tip)
        # The Y arm roots on the rod at the hub on the tip's own side of the
        # segment, so the u-joint that side sits inside the V.
        z_root = geo.z_bot_hub if proximal else geo.z_top_hub
        root = (rod_radius * radial[0], rod_radius * radial[1], z_root)

        # Bearing wheel: axle tangential, so the tendon's plane (radial and
        # vertical at this azimuth) is the wheel's plane.
        a0 = tuple(b - 0.0025 * t for b, t in zip(brg, tangent))
        a1 = tuple(b + 0.0025 * t for b, t in zip(brg, tangent))
        link.append(
            f'<geom name="{prefix}s{n}_a{s.index}_bearing" type="cylinder" '
            f'fromto="{_v(a0)} {_v(a1)}" size="{_g(bearing_radius)}" '
            f'rgba="{RGBA_BEARING}" mass="0"/>')

        # Y arm: a flat plate in the Y's own vertical plane.  xyaxes puts the
        # thin dimension along the tangent and the long one along the arm.
        d = _unit(_sub(tip, root))
        wide = _cross(d, tangent)
        half_len = 0.5 * _norm(_sub(tip, root))
        link.append(
            f'<geom name="{prefix}s{n}_a{s.index}_yarm" type="box" '
            f'pos="{_v(_lerp(root, tip, 0.5))}" '
            f'xyaxes="{_v(tangent)} {_v(wide)}" '
            f'size="0.003 0.009 {_g(half_len)}" rgba="{RGBA_Y}" '
            f'mass="{_g(masses.y_arm)}"/>')
        link.append(
            f'<geom name="{prefix}s{n}_a{s.index}_ytip" type="box" '
            f'pos="{_v(tip)}" xyaxes="{_v(tangent)} {_v(radial)}" '
            f'size="0.006 0.011 0.008" rgba="{RGBA_Y_TIP}" mass="0"/>')
        # Braided carbon brace from 70 % along the arm back to the rod, about
        # 0.42 of the span from the tip's own joint plane (Fig. 1C).
        brace_from = _lerp(root, tip, 0.7)
        brace_to = (0.0, 0.0, (-geo.span + 0.42 * geo.span) if proximal
                    else -0.42 * geo.span)
        link.append(
            f'<geom name="{prefix}s{n}_a{s.index}_ybrace" type="capsule" '
            f'fromto="{_v(brace_from)} {_v(brace_to)}" size="0.0035" '
            f'rgba="{RGBA_BRACE}" mass="0"/>')

        # The actuator: LL long, hung from the tip along the line to its
        # bearing.  Never longer than 95 % of that line, so a caller-supplied
        # table with a short segment still draws a tendon rather than a sleeve
        # through its own bearing.
        run = _norm(_sub(brg, tip))
        length = min(geo.ll, 0.95 * run)
        end = _lerp(tip, brg, length / run)
        link.append(
            f'<geom name="{prefix}s{n}_a{s.index}_sleeve" type="capsule" '
            f'fromto="{_v(tip)} {_v(end)}" size="{_g(actuator_radius)}" '
            f'rgba="{RGBA_SLEEVE_PROXIMAL if proximal else RGBA_SLEEVE_DISTAL}" '
            f'mass="{_g(masses.actuator)}"/>')
        if tendon_stubs:
            link.append(
                f'<geom name="{prefix}s{n}_a{s.index}_tendon" type="capsule" '
                f'fromto="{_v(end)} {_v(brg)}" size="0.0015" '
                f'rgba="{RGBA_TENDON}" mass="0"/>')
        if routing_sites:
            link.append(f'<site name="{names["tip"]}" pos="{_v(tip)}"{site_grp}/>')
            link.append(f'<site name="{names["end"]}" pos="{_v(end)}"{site_grp}/>')
            link.append(f'<site name="{names["brg"]}" pos="{_v(brg)}"{site_grp}/>')

        # The bracket on the outer ring, in the driven joint's plane, on the
        # body across that joint.
        bx, by = ja * radial[0], ja * radial[1]
        bz = -geo.jd if proximal else 0.0
        bracket = [
            f'<geom name="{prefix}s{n}_a{s.index}_bracket" type="box" '
            f'pos="{_g(bx)} {_g(by)} {_g(bz)}" '
            f'xyaxes="{_v(radial)} {_v(tangent)}" size="0.005 0.007 0.006" '
            f'rgba="{RGBA_BRACKET}" mass="0"/>']
        if routing_sites:
            bracket.append(f'<site name="{names["brk"]}" '
                           f'pos="{_g(bx)} {_g(by)} {_g(bz)}"{site_grp}/>')
        (parent if proximal else distal).extend(bracket)

    # -- distal side: the distal u-joint's outer ring ------------------------
    distal.insert(0,
        f'<geom name="{prefix}seg{n}_plate2_geom" type="cylinder" '
        f'fromto="0 0 {_g(-plate_half_thickness)} 0 0 {_g(plate_half_thickness)}" '
        f'size="{_g(geo.ja2)}" rgba="{RGBA_DISK}" mass="{_g(masses.dist_disk)}"/>')
    return {"parent": parent, "link": link, "distal": distal}


def segment_masses(segs, *, link_mass_kg=None, bracket_mass_kg=None,
                   link_density=DEFAULT_LINK_DENSITY_KG_M,
                   plate_mass=DEFAULT_PLATE_MASS_KG,
                   spacer_mass=DEFAULT_SPACER_MASS_KG,
                   actuator_mass=DEFAULT_ACTUATOR_MASS_KG,
                   y_mass=DEFAULT_Y_MASS_KG,
                   hub_mass=DEFAULT_HUB_MASS_KG) -> list:
    """Per-segment :class:`SegmentMasses`, with any totals spread proportionally.

    A link total keeps the eight sleeves at ``actuator_mass`` each, the one mass
    measured on this arm, and scales the *structure* -- rod, hubs and Y arms --
    by one factor, so the component defaults decide where the structure's mass
    sits and the total decides how much there is.  A total below the sleeves'
    ``8 * actuator_mass`` is refused rather than given a negative structure.
    A bracket total for segment ``i`` scales the geoms rigid with that segment's
    distal ring -- its distal disk, and the next segment's spacer and proximal
    disk, which are emitted into the same body -- and nothing else.  Segment 1's
    proximal disk rides the static base and is left at ``plate_mass``.

    CHANGED 2026-09-10 (the physical refit): a link total used to scale the
    sleeves with the structure, which put 43 g sleeves on the 0.70 kg prior link
    and let a fitted link total change the one number that was measured.
    """
    links = resolve_per_segment(link_mass_kg, "link_mass_kg", len(segs))
    brackets = resolve_per_segment(bracket_mass_kg, "bracket_mass_kg", len(segs))
    sleeves = 8.0 * actuator_mass
    link_scale, bracket_scale = [], []
    for i, geo in enumerate(segs):
        structure = link_density * geo.span + 2.0 * hub_mass + 4.0 * y_mass
        if structure <= 0.0:
            raise ValueError("the component link structure masses sum to zero")
        if links is not None and links[i] <= sleeves:
            raise ValueError(
                f"link_mass_kg[{i}] = {links[i]:g} kg does not exceed the eight "
                f"sleeves it carries ({sleeves:g} kg at actuator_mass "
                f"{actuator_mass:g} kg, the measured mass)")
        link_scale.append(1.0 if links is None else (links[i] - sleeves) / structure)
        body = plate_mass
        if i + 1 < len(segs):
            body += plate_mass + (spacer_mass if segs[i + 1].jd else 0.0)
        if body <= 0.0:
            raise ValueError("the component bracket masses sum to zero")
        bracket_scale.append(1.0 if brackets is None else brackets[i] / body)

    out = []
    for i, geo in enumerate(segs):
        ls = link_scale[i]
        up = bracket_scale[i - 1] if i > 0 else 1.0
        out.append(SegmentMasses(
            rod=link_density * geo.span * ls,
            hub=hub_mass * ls,
            y_arm=0.5 * y_mass * ls,
            actuator=actuator_mass,
            prox_disk=plate_mass * up,
            spacer=spacer_mass * up,
            dist_disk=plate_mass * bracket_scale[i]))
    return out


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
                  link_mass_kg=DEFAULT_LINK_MASS_KG,
                  bracket_mass_kg=DEFAULT_BRACKET_MASS_KG,
                  link_density=DEFAULT_LINK_DENSITY_KG_M,
                  plate_mass=DEFAULT_PLATE_MASS_KG,
                  spacer_mass=DEFAULT_SPACER_MASS_KG,
                  actuator_mass=DEFAULT_ACTUATOR_MASS_KG,
                  y_mass=DEFAULT_Y_MASS_KG,
                  hub_mass=DEFAULT_HUB_MASS_KG,
                  base_plate_mass=DEFAULT_BASE_PLATE_MASS_KG,
                  tip_mass=DEFAULT_TIP_MASS_KG,
                  joint_stiffness=DEFAULT_JOINT_STIFFNESS,
                  y_tip_radius: float = DEFAULT_Y_TIP_RADIUS_M,
                  y_tip_z_offset: float = DEFAULT_Y_TIP_Z_OFFSET_M,
                  actuator_radius: float = DEFAULT_ACTUATOR_RADIUS_M,
                  rod_radius: float = 0.009,
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

    The bodies and hinges are the display model's, with the proximal pair
    declared y-then-x; the drawn parts are :func:`promax_segment_elements`; the
    tendons are the three-site bearing routing of the module docstring.  Every
    mass, radius and dissipation scalar is an argument; nothing in the body of
    this function reads a module constant at call time.
    """
    params = cp.require_measured() if params is None else params
    segs = segment_geometry(params)
    seats = actuator_seats() if seats is None else tuple(seats)
    stiffness = resolve_per_segment_nonnegative(joint_stiffness, "joint_stiffness",
                                                len(segs))
    masses = segment_masses(
        segs, link_mass_kg=link_mass_kg, bracket_mass_kg=bracket_mass_kg,
        link_density=link_density, plate_mass=plate_mass,
        spacer_mass=spacer_mass, actuator_mass=actuator_mass, y_mass=y_mass,
        hub_mass=hub_mass)

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
    for element in promax_base_elements(
            prefix=prefix, ring_radius=segs[0].ja1,
            base_plate_radius=segs[0].ja1 + 0.02,
            base_plate_mass=base_plate_mass, post_radius=spacer_radius):
        emit(element)
    # Plate 0 is the base plate and it sits on the STATIC body, not on segment
    # 1's link: it is the frame the arm's kinematics de-rotates by, and it does
    # not turn with the first universal joint.
    emit(f'<site name="{prefix}plate0" pos="0 0 0"/>')

    for n, geo in enumerate(segs, start=1):
        # Passive stiffness on this segment's four hinges, emitted only when it
        # is non-zero so the default document is byte-identical to the models
        # every earlier fit ran on.  springref stays MuJoCo's 0: at rest the
        # arm hangs straight.
        k_attr = (f' stiffness="{_g(stiffness[n - 1])}"'
                  if stiffness[n - 1] > 0.0 else "")
        parts = promax_segment_elements(
            geo, [s for s in seats if s.segment == n], n=n, prefix=prefix,
            masses=masses[n - 1], y_tip_radius=y_tip_radius,
            y_tip_z_offset=y_tip_z_offset, actuator_radius=actuator_radius,
            rod_radius=rod_radius, plate_half_thickness=plate_half_thickness,
            spacer_radius=spacer_radius, routing_sites=True,
            tendon_stubs=False)

        # Parent side: the JD spacer, the proximal u-joint ring and its four
        # brackets.  All rigid with the plate above, so plates 2 and 4 turn
        # with the segment above them, not with the joint they carry.
        for element in parts["parent"]:
            emit(element)
        if geo.jd:
            emit(f'<site name="{prefix}plate{2 * (n - 1)}" '
                 f'pos="0 0 {_g(-geo.jd)}"/>')

        # Link body: the proximal universal joint as two stacked hinges at one
        # origin, **y declared first**.  Declaration order is the composition
        # order, so this is what makes the model agree with fkine order="yx" --
        # the CAN arm's measured assembly -- and it is why qpos is a permutation
        # of q rather than q itself.
        emit(f'<body name="{prefix}seg{n}_link" pos="0 0 {_g(-geo.jd)}">')
        depth += 1
        emit(f'<joint name="{prefix}uj{2 * n - 1}_y" axis="0 1 0"{k_attr}/>')
        emit(f'<joint name="{prefix}uj{2 * n - 1}_x" axis="1 0 0"{k_attr}/>')
        for element in parts["link"]:
            emit(element)

        # Distal plate body: the second universal joint, on the 45-degree
        # bracket axes.  Declared x then y, which IS fkine's composition for the
        # distal pair -- the 2026-08-21 experiment tested swapping this one too
        # and it was markedly worse, so xi3 is genuinely the link-fixed axis.
        emit(f'<body name="{prefix}seg{n}_plate2" pos="0 0 {_g(-geo.span)}">')
        depth += 1
        emit(f'<joint name="{prefix}uj{2 * n}_x" axis="{_SQ2:.10g} {_SQ2:.10g} 0"{k_attr}/>')
        emit(f'<joint name="{prefix}uj{2 * n}_y" axis="{-_SQ2:.10g} {_SQ2:.10g} 0"{k_attr}/>')
        emit(f'<site name="{prefix}plate{2 * n - 1}" pos="0 0 0"/>')
        for element in parts["distal"]:
            emit(element)

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
        names = routing_site_names(s, prefix)
        # Actuator end -> bearing -> bracket.  The first span is rigid (both
        # sites on the link) and cancels in ten_length - tendon_length0; it is
        # there so the drawn tendon starts at the sleeve.
        ten_lines.append(
            f'{indent}<spatial name="{ten}" class="{class_name}">\n'
            f'{indent}  <site site="{names["end"]}"/>\n'
            f'{indent}  <site site="{names["brg"]}"/>\n'
            f'{indent}  <site site="{names["brk"]}"/>\n'
            f'{indent}</spatial>')
        # Pull-only by construction, on both the command and the force: the
        # clip is what replay.assert_no_clamp watches, and a clip that can be
        # reached from the positive side would let a McKibben push.
        act_lines.append(
            f'{indent}<motor name="{act}" tendon="{ten}" gear="1" '
            f'ctrllimited="true" ctrlrange="{_g(-force_range_n)} 0" '
            f'forcelimited="true" forcerange="{_g(-force_range_n)} 0"/>')
    return bodies, "\n".join(ten_lines), "\n".join(act_lines)


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

_SCENE_TEMPLATE = """<mujoco model="umarm_canarm_twin">
  <!-- GENERATED by digital_twin/mjcf_generator.py - edit the generator, not
       this file.  PHYSICS model: 24 tendon actuators routed around the ProMax
       hub bearings, masses, fitted-seed dissipation.  No contacts, no
       keyframes, no end effector.
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
      <tendon width="0.003" damping="{tendon_damping}" rgba="{tendon_rgba}"/>
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


def _fmt_seg(value) -> str:
    if value is None:
        return "None"
    if np.ndim(value) == 0:
        return f"{float(value):g}"
    return "(" + ", ".join(f"{float(v):g}" for v in value) + ")"


def generate_xml(*,
                 base_pos=DEFAULT_BASE_POS,
                 base_rpy_deg=DEFAULT_BASE_RPY_DEG,
                 joint_damping: float = DEFAULT_JOINT_DAMPING,
                 joint_frictionloss: float = DEFAULT_JOINT_FRICTIONLOSS,
                 tendon_damping: float = DEFAULT_TENDON_DAMPING,
                 link_mass_kg=DEFAULT_LINK_MASS_KG,
                 bracket_mass_kg=DEFAULT_BRACKET_MASS_KG,
                 link_density: float = DEFAULT_LINK_DENSITY_KG_M,
                 plate_mass: float = DEFAULT_PLATE_MASS_KG,
                 spacer_mass: float = DEFAULT_SPACER_MASS_KG,
                 actuator_mass: float = DEFAULT_ACTUATOR_MASS_KG,
                 y_mass: float = DEFAULT_Y_MASS_KG,
                 hub_mass: float = DEFAULT_HUB_MASS_KG,
                 base_plate_mass: float = DEFAULT_BASE_PLATE_MASS_KG,
                 tip_mass: float = DEFAULT_TIP_MASS_KG,
                 joint_stiffness=DEFAULT_JOINT_STIFFNESS,
                 y_tip_radius: float = DEFAULT_Y_TIP_RADIUS_M,
                 y_tip_z_offset: float = DEFAULT_Y_TIP_Z_OFFSET_M,
                 actuator_radius: float = DEFAULT_ACTUATOR_RADIUS_M,
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
    exists.  Here the numbers are arguments and the *defaults* carry the
    provenance.

    THE MASS MODEL A FIT DRIVES is two per-segment totals: ``link_mass_kg``
    (rod + hubs + Ys + eight actuators, i.e. body ``seg{n}_link``) and
    ``bracket_mass_kg`` (body ``seg{n}_plate2`` minus the tip stub).  Each takes
    ``None`` (the component model), one number for all three segments, or three
    numbers.  The component keywords (``link_density``, ``hub_mass``,
    ``y_mass``, ``plate_mass``, ``spacer_mass``) set the proportions within a
    body when its total is given, and the totals when it is ``None``;
    ``actuator_mass`` is honoured exactly either way.

    ``joint_stiffness`` (N*m/rad, one number or three per segment) is the
    passive restoring stiffness on each segment's four hinges, zero by default
    (:data:`DEFAULT_JOINT_STIFFNESS`).

    The argument list is echoed into the document's header comment, so a shipped
    XML always names the parameters it was built with -- the property that makes
    an anomalous rollout traceable to a scene rather than to a guess.
    """
    bodies, tendons, actuators = build_arm_xml(
        prefix=prefix, mount_body=mount_body, mocap_mount=mocap_mount,
        actuator_prefix=actuator_prefix, params=params,
        base_pos=base_pos, base_rpy_deg=base_rpy_deg,
        link_mass_kg=link_mass_kg, bracket_mass_kg=bracket_mass_kg,
        link_density=link_density, plate_mass=plate_mass,
        spacer_mass=spacer_mass, actuator_mass=actuator_mass,
        y_mass=y_mass, hub_mass=hub_mass,
        base_plate_mass=base_plate_mass, tip_mass=tip_mass,
        joint_stiffness=joint_stiffness,
        y_tip_radius=y_tip_radius, y_tip_z_offset=y_tip_z_offset,
        actuator_radius=actuator_radius,
        force_range_n=force_range_n, class_name=class_name, **arm_kwargs)

    genparams = ", ".join((
        f"base_pos={tuple(float(v) for v in base_pos)}",
        f"base_rpy_deg={tuple(float(v) for v in base_rpy_deg)}",
        f"joint_damping={joint_damping:g}",
        f"joint_frictionloss={joint_frictionloss:g}",
        f"tendon_damping={tendon_damping:g}",
        f"link_mass_kg={_fmt_seg(link_mass_kg)}",
        f"bracket_mass_kg={_fmt_seg(bracket_mass_kg)}",
        f"link_density={link_density:g}",
        f"plate_mass={plate_mass:g}",
        f"spacer_mass={spacer_mass:g}",
        f"actuator_mass={actuator_mass:g}",
        f"y_mass={y_mass:g}",
        f"hub_mass={hub_mass:g}",
        f"joint_stiffness={_fmt_seg(joint_stiffness)}",
        f"y_tip_radius={y_tip_radius:g}",
        f"y_tip_z_offset={y_tip_z_offset:g}",
        f"actuator_radius={actuator_radius:g}",
        f"timestep={timestep:g}",
        f"force_range_n={force_range_n:g}",
        f"proximal_order={PROXIMAL_ORDER}",
        "routing=promax_bearing",
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
        tendon_rgba=RGBA_TENDON,
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
    the muscles are unstretched.  The absolute value includes the rigid
    actuator-end-to-bearing span, which no consumer reads.
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
    ap.add_argument("--link-mass-kg", type=float, nargs=3,
                    default=list(DEFAULT_LINK_MASS_KG))
    ap.add_argument("--bracket-mass-kg", type=float, nargs=3,
                    default=list(DEFAULT_BRACKET_MASS_KG))
    args = ap.parse_args(argv)
    kwargs = dict(joint_damping=args.joint_damping,
                  joint_frictionloss=args.joint_frictionloss,
                  tendon_damping=args.tendon_damping,
                  link_mass_kg=tuple(args.link_mass_kg),
                  bracket_mass_kg=tuple(args.bracket_mass_kg))
    if args.out:
        print(f"wrote {generate_scene(args.out, **kwargs)}")
    else:
        print(generate_xml(**kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
