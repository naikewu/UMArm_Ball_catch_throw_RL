r"""Fit the twin's segment masses, force law and dissipation together, with CMA-ES.

WHY THIS EXISTS.  Before this module the twin's mechanical half had been fitted
once, by :mod:`digital_twin.outer_fit`: a coordinate search of 2 rounds x 5
points over three per-segment force-gain multipliers and two damping scalars, on
the first 15 s of ``random_walk``.  **No mass was ever fitted.**  The body masses
were RS485 carry-overs (1.833 kg moving) until the 2026-09-10 geometry rework
replaced them with the Koopman ProMax prior (2.220 kg), and neither is a
weighing; the only mass measured on this arm is the operator's ~30 g per
actuator.  The force law's length stiffness ``bf`` and the Coulomb friction were
seeds as well.  The operator cannot weigh a segment yet, so this module asks the
recordings instead.

WHAT IS SEARCHED.  Fifteen numbers, each in log space between bounds whose
evidence is written beside them in :data:`PARAMS`:

=========================  ===============================  ================================
parameter                  bounds                           start
=========================  ===============================  ================================
``link_mass_kg`` x3        0.25 - 1.50 kg                   Koopman prior 0.70 / 0.50 / 0.50
``bracket_mass_kg`` x3     0.05 - 0.60 kg (seg 3 0.01-0.40)  0.20 / 0.20 / 0.10
``rest_gain_n_per_pa`` x3  1e-5 - 1e-3 N/Pa                 outer fit 1.62e-4/1.34e-4/3.15e-5
``bf_over_l0`` x3          0.50 - 1.45                      RS485 shape 1.216 / 1.364 / 1.269
``tendon_damping``         0.01 - 100 N s/m                 outer fit 0.316
``joint_damping``          0.001 - 3.0 N m s/rad            outer fit 0.00822
``joint_frictionloss``     0.001 - 0.5 N m                  RS485 fit 0.025
=========================  ===============================  ================================

The force law is searched as a rest gain and a shape rather than as ``coeff`` and
``bf``.  ``ActuatorModel.force_n`` pulls with ``coeff * p * (3 l^2 - bf^2)``, so
at the rest length the gain is ``K = coeff (3 l0^2 - bf^2)`` N/Pa and the shape
``r = bf / l0`` sets the normalised length stiffness ``(dF/dl) l0 / F = 6 / (3 -
r^2)``.  Moving ``bf`` alone moves both, which puts a ridge between ``coeff`` and
``bf`` that an optimiser must learn before it can learn anything else, and the
``(K, r)`` coordinates remove it.  The checkpoint still stores ``coeff`` and
``bf``, since those are what ``twin_params`` loads.

WHY CMA-ES AND NOT DIFFERENTIAL EVOLUTION.  Three properties of this problem
decide it.  First, the statics pin only the ratio of force gain to mass, so the
objective has long diagonal valleys along ``(mass x k, gain x k)``.  CMA-ES
learns the covariance of such a valley and then steps along it, whereas
differential evolution's per-coordinate crossover proposes moves across it and
spends most of its evaluations there.  Second, one evaluation is 178.5 s of
open-loop arm, about 5.5 s of wall clock even at 24 workers, so the whole budget
is a few thousand evaluations, which is the regime CMA-ES is built for at 15
parameters.  Third, ``ask``/``tell`` makes one generation exactly one
pool-filling batch of :data:`DEFAULT_POPSIZE` candidates, so the parallelism
costs the algorithm nothing.  The ``cma`` package (4.4.4, installed into
``.venv`` for this) is used rather than a local implementation, because its
boundary handling is tested and a hand-written one would not be.

THE OBJECTIVE.  Joint deflection RMS in degrees, open loop, through
``twin_compare.twin_rollout`` -- the alignment and re-referencing the held-out
score uses -- averaged over the twelve joints exactly as ``compare_metrics``
averages ``rms_deg_mean``, then over the windows of a family, then over the
five families with equal weight.  Equal family weight is a choice: the families
differ in length by a factor of four, and weighting by seconds would let the
chirps and staircases outvote the six short ringdowns that carry the dynamics.
The windows (:func:`window_specs`) are:

* ``random_walk`` -- the first 20 s of ``walk_000``, all twelve joints at once;
* ``chirp`` -- joints 0 (segment 1 proximal) and 6 (segment 2 distal), 0.15 to
  4 Hz over 18 s plus 1.5 s of the settle, where the ring frequency is read;
* ``ringdown`` -- six contiguous ``ringdown_charge`` + ``ringdown`` pairs, one
  proximal and one distal released joint per segment, 6.5 s each: the ring's
  frequency reads gain over inertia and its decay reads dissipation over inertia;
* ``staircase`` -- the positive board of joints 4, 8 and 11 stepped 4 to 24 psi
  against gravity, 17.5 s each: statics from the middle of the chain to its tip,
  where only the segment-3 bracket hangs below joint 11;
* ``pair_sweep`` -- joint 2's differential swept at co-contraction sums of 10 and
  18 psi, 25 s: the stiffening between the two sums is what reads ``bf``.

Every window is **contiguous**, with no sync gap over :data:`MAX_GAP_S`, and none
contains a row of a held-out family: :func:`resolve_rows` raises on either.
Windows start with a 1 s lead-in taken from the idle segment before them, so
``twin_compare``'s reference rule finds its pre-drive hold (every target at or
below 1 psi for 0.5 s) at the start, where the twin also starts at rest.  The
ringdown pairs have no idle segment before them, because each charge starts from
the previous release's background pressures.  They are therefore referenced over
the last 80 rows of the release, where the ring has decayed; the 2.5 s charge
brings the twin, which starts empty, to the same loaded state first.

A window whose rollout touches the force clip, raises, has fewer than
``twin_compare.MIN_SAMPLES`` usable rows, returns a non-finite error, or makes
MuJoCo reset its state (``mjWARN_BADQACC``) is invalid, and a candidate with any
invalid window scores :data:`PENALTY_DEG` plus its count of invalid windows
(:func:`aggregate`).  Averaging the penalty in with the valid windows instead
would dilute it: one bad ringdown of six in one family of five would add only
1/30 of the penalty.  The score is finite, so CMA-ES can still rank the
candidate and step away, and it exceeds the 160 deg that is the largest RMS two
deflection traces inside a +-40 deg joint range can differ by.

WHAT THE DATA CAN PIN, STATED BEFORE THE FIT RATHER THAN AFTER IT.  Scaling
every mass and every muscle gain by one factor ``k`` scales the gravity torque,
the inertia and the muscle torque by ``k``.  The equation of motion is then
unchanged except that every dissipation term and the joint armature are ``1/k``
as large relative to it.  Therefore:

* the statics (staircase, pair sweep) pin ``gain / mass`` and nothing more;
* the dynamics separate the two only through terms that do not scale with ``k``:
  ``damp_b1`` (the pressure-scheduled tendon damping, 9.9e-4 N s/m per Pa, an
  RS485 fit that outweighs a 0.3 N s/m base by two orders of magnitude at
  10 psi) and ``mjcf_generator.JOINT_ARMATURE`` = 0.01 kg m^2.  Both are carried
  over, and neither has been measured on this arm;
* scaling mass, gain, the three fitted dissipation scalars **and** ``damp_b1`` by
  ``k`` together leaves only the armature to break the symmetry.

:func:`profile_candidates` measures the loss along all three directions and
along each mass alone, so the checkpoint says which masses the data pinned
rather than implying that it pinned all of them.

WHAT THIS DOES NOT SHOW.  A mass reported here is a mass that makes *this model*
reproduce *these recordings*, which is not a weighing.  The model has no
passive joint stiffness, so any elastic restoring torque the arm gets from its
pneumatic tubing and wiring can only be expressed as gravity (mass) or as the
force law's shape.  The ring radii ``AO``/``AA``/``JA`` are CAD, so a moment arm
that is wrong on the metal is absorbed into the gain.  The flow net is held
fixed, and a pressure error it makes is fitted as a mechanical one.

Usage::

    .venv\Scripts\python.exe -m digital_twin.mech_fit --workers 24 --generations 70
    .venv\Scripts\python.exe -m digital_twin.mech_fit --profile-only digital_twin/checkpoints/canarm_mech.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from digital_twin import actuator_model as AM  # noqa: E402
from digital_twin import mjcf_generator as MG  # noqa: E402
from digital_twin import replay as R  # noqa: E402
from digital_twin import twin_compare as TC  # noqa: E402

# ---------------------------------------------------------------------------
# Constants, each with its reason
# ---------------------------------------------------------------------------

#: The fit date every checkpoint this module writes carries.
FIT_DATE = "2026-09-10"

#: The two campaigns of 2026-09-10.  The first holds the statics (staircases,
#: pair sweeps; cut at 14.2 min by a USB-CDC fault), the second the dynamics
#: (chirps, random walk, ringdowns) and the held-out ``validation`` sequence.
SESSION_STATICS = "session_20260910_012159"
SESSION_DYNAMICS = "session_20260910_013843"

#: Families no window may contain a single row of.  ``validation`` is the
#: sequence the twin is scored on; fitting on it would be the mistake
#: ``outer_fit``'s docstring names, made easier here by fifteen parameters.
HELDOUT_KINDS = ("validation",)

#: The families, in the order they are logged.  Each gets equal weight.
FAMILIES = ("random_walk", "chirp", "ringdown", "staircase", "pair_sweep")

#: s.  A sync gap above this inside a window is a break in the recording rather
#: than a skipped cycle.  Tied to the firmware, not to jitter: below half the TLE
#: boards' 500 ms sync-loss failsafe a gap changes nothing on the metal, since
#: every board holds its last target, and the replay reproduces exactly that
#: because it advances on the recorded clock.  The staircases carry 60 ms
#: hiccups (eight skipped cycles; the collector skipped 0.90 % of periods), and
#: the session's one real break, before ``ring_014_charge``, is 6.75 s.
MAX_GAP_S = 0.25

#: deg.  The per-window loss of an invalid rollout, and the floor of the score of
#: any candidate with one (:func:`aggregate`).  Finite so CMA-ES can rank the
#: candidate; above the 160 deg that is the largest RMS two deflection traces
#: inside a +-40 deg joint range can differ by, so an invalid candidate always
#: ranks below every valid one.
PENALTY_DEG = 1000.0

#: A window must have at least this fraction of rows with a usable pose, or the
#: window is refused while planning rather than silently scored on fewer rows.
MIN_POSED_FRACTION = 0.98

#: CMA-ES population: one candidate per worker, so a generation is one
#: pool-filling batch.  The ``cma`` default for 15 parameters is 12; 24 doubles the
#: rank-mu update's sample per generation for the same per-evaluation cost.
DEFAULT_POPSIZE = 24

#: Workers.  Measured on this machine (Ryzen 9 9950X3D, 16 cores / 32 threads),
#: rolling a 7 s window: 27.5 s of arm per wall second at 16 workers, 32.3 at 24
#: and 33.4 at 32, i.e. 24 workers take 97 % of what the machine gives.
DEFAULT_WORKERS = 24

#: Measured throughput at :data:`DEFAULT_WORKERS`, seconds of arm per wall second,
#: used only to print a wall-clock estimate before the run.
MEASURED_ARM_S_PER_WALL_S = 32.3

#: Generations and a wall-clock cap.  70 x 24 x 178.5 s = 300 k s of arm, about
#: 2.6 h at the measured throughput; the cap stops a slower run cleanly.
DEFAULT_GENERATIONS = 70
DEFAULT_MAX_HOURS = 2.75

#: Initial step, in the unit cube the bounds define (each parameter mapped
#: log-linearly onto [0, 1]).  0.2 is a factor of 1.4 on a link mass and 2.5 on a
#: rest gain, i.e. proportional to how wide each bound had to be.
DEFAULT_SIGMA0 = 0.2

#: Threads per worker for every BLAS/OpenMP pool.  The flow net's forward pass is a
#: (24, 5) x (5, 64) product: threading it costs more than it saves, and 24
#: processes times N threads would oversubscribe the machine.
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")

#: Factors each profile line is evaluated at, around the fitted optimum.
PROFILE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)

#: A profile line "pins" its parameter on a side when moving it by the outer
#: factor (0.5 or 2) raises the loss by more than this fraction of the optimum's.
#: The objective is deterministic, so this is a judgement about what counts as a
#: difference worth stating, not a noise floor: 1 % of a ~5 deg loss is 0.05 deg.
PIN_FRACTION = 0.01

#: The earlier coordinate search's per-segment ``coeff``
#: (``checkpoints/canarm_outer.json``), the start point's force gain.  Carried
#: as numbers so a start point exists without that file.
OUTER_FIT_COEFF = (0.003362352603854604, 0.005423333560277954,
                   0.0011178183929440736)
OUTER_FIT_TENDON_DAMPING = 0.31622776601683805
OUTER_FIT_JOINT_DAMPING = 0.008221921916437789


def _rest_gain(coeff, bf, l0):
    c, b, l = (np.asarray(v, dtype=float) for v in (coeff, bf, l0))
    return c * (3.0 * l * l - b * b)


_K_START = _rest_gain(OUTER_FIT_COEFF, AM.BF_SEED_M, AM.L0_SEED_M)


# ---------------------------------------------------------------------------
# The parameter vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    """One searched number: its bounds, its start and the evidence for both."""

    name: str
    lo: float
    hi: float
    start: float
    unit: str
    evidence: str


_LINK_WHY = (
    "lower 0.25 kg: the link carries its eight actuators, 8 x 30 g = 0.24 kg, the "
    "only mass measured on this arm; upper 1.50 kg: 2.1x the heaviest Koopman "
    "ProMax prior link (0.70 kg) and 3x the other two (0.50 kg)")
_BRACKET_WHY = (
    "lower 0.05 kg: the body carries two u-joint rings and a spacer, and one "
    "47 mm-radius aluminium ring 4 mm thick with an estimated 12 mm annulus is "
    "33 g; upper 0.60 kg: 3x the Koopman prior connector (0.20 kg)")
_BRACKET3_WHY = (
    "segment 3's distal body carries one ring and plate 5 and no spacer, so it may "
    "be small: lower 0.01 kg is a third of one estimated 33 g ring, below the "
    "0.10 kg default (itself an estimate, the Koopman 0.001 kg being a "
    "no-linkage placeholder); upper 0.40 kg: 4x that default")
_GAIN_WHY = (
    "rest force per pascal K = coeff (3 l0^2 - bf^2); upper 1e-3 N/Pa: the "
    "Chou-Hannaford zero-contraction limit pi D0^2/4 (3 cos^2 20deg - 1) of a 25 mm "
    "braid is 8.1e-4 N/Pa, wider than the ~20 mm sleeves of Fig. 1C; lower "
    "1e-5 N/Pa: 3.2x below the weakest earlier fit (segment 3, 3.15e-5).  The "
    "earlier multipliers 0.009-0.045 of the RS485 seed span 3.15e-5..1.62e-4 N/Pa, "
    "bracketed 3.2x below and 6.2x above")
_SHAPE_WHY = (
    "r = bf / l0 sets the normalised length stiffness 6/(3 - r^2); lower 0.50 "
    "(stiffness 2.18, within 9 % of its r -> 0 limit of 2, below which the shape "
    "is indistinguishable); upper 1.45 (stiffness 6.69; the muscle goes slack at "
    "l = r l0/sqrt3 = 0.837 l0, i.e. 29.0/23.9/23.2 mm of contraction on segments "
    "1/2/3, beyond the 21.7 mm largest excursion over the validation poses)")

PARAMS = (
    Param("link_mass_kg_1", 0.25, 1.50, MG.DEFAULT_LINK_MASS_KG[0], "kg", _LINK_WHY),
    Param("link_mass_kg_2", 0.25, 1.50, MG.DEFAULT_LINK_MASS_KG[1], "kg", _LINK_WHY),
    Param("link_mass_kg_3", 0.25, 1.50, MG.DEFAULT_LINK_MASS_KG[2], "kg", _LINK_WHY),
    Param("bracket_mass_kg_1", 0.05, 0.60, MG.DEFAULT_BRACKET_MASS_KG[0], "kg", _BRACKET_WHY),
    Param("bracket_mass_kg_2", 0.05, 0.60, MG.DEFAULT_BRACKET_MASS_KG[1], "kg", _BRACKET_WHY),
    Param("bracket_mass_kg_3", 0.01, 0.40, MG.DEFAULT_BRACKET_MASS_KG[2], "kg", _BRACKET3_WHY),
    Param("rest_gain_n_per_pa_1", 1e-5, 1e-3, float(_K_START[0]), "N/Pa", _GAIN_WHY),
    Param("rest_gain_n_per_pa_2", 1e-5, 1e-3, float(_K_START[1]), "N/Pa", _GAIN_WHY),
    Param("rest_gain_n_per_pa_3", 1e-5, 1e-3, float(_K_START[2]), "N/Pa", _GAIN_WHY),
    Param("bf_over_l0_1", 0.50, 1.45, float(AM.BF_TO_L0_REF[0]), "-", _SHAPE_WHY),
    Param("bf_over_l0_2", 0.50, 1.45, float(AM.BF_TO_L0_REF[1]), "-", _SHAPE_WHY),
    Param("bf_over_l0_3", 0.50, 1.45, float(AM.BF_TO_L0_REF[2]), "-", _SHAPE_WHY),
    Param("tendon_damping", 0.01, 100.0, OUTER_FIT_TENDON_DAMPING, "N s/m",
          "the p = 0 base of the tendon damping; damp_b1 x p adds 68 N s/m at 10 psi, "
          "so upper 100 N s/m lets the base rival the scheduled term at 15 psi; "
          "lower 0.01 is 1/32 of the earlier fit's 0.316"),
    Param("joint_damping", 1e-3, 3.0, OUTER_FIT_JOINT_DAMPING, "N m s/rad",
          "upper 3.0 brackets the Koopman ProMax model's 2.25 (the RS485 engineering "
          "guess under which every mode was over-damped); lower 0.001 is 1/8 of the "
          "earlier fit's 0.0082"),
    Param("joint_frictionloss", 1e-3, 0.5, MG.DEFAULT_JOINT_FRICTIONLOSS, "N m",
          "upper 0.5 sits below the RS485 arm's superseded 0.70 N m, which alone "
          "exceeded the peak elastic torque of a 1 deg oscillation; lower 0.001 is "
          "1/25 of the RS485 fit's 0.025 default"),
)

NAMES = tuple(p.name for p in PARAMS)
N_PARAMS = len(PARAMS)
MASS_NAMES = NAMES[0:6]
GAIN_NAMES = NAMES[6:9]
SHAPE_NAMES = NAMES[9:12]
DISSIPATION_NAMES = NAMES[12:15]
_LOG_LO = np.log(np.array([p.lo for p in PARAMS]))
_LOG_HI = np.log(np.array([p.hi for p in PARAMS]))


def start_values() -> dict:
    """The start point: the earlier outer fit's gains and damping, the prior masses."""
    return {p.name: float(p.start) for p in PARAMS}


def as_vector(values) -> np.ndarray:
    """A ``{name: value}`` dict or a ``(15,)`` sequence as a ``(15,)`` float array."""
    if isinstance(values, dict):
        missing = [n for n in NAMES if n not in values]
        if missing:
            raise KeyError(f"parameter values are missing {missing}")
        return np.array([float(values[n]) for n in NAMES], dtype=float)
    vec = np.asarray(values, dtype=float)
    if vec.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters; got shape {vec.shape}")
    return vec


def to_unit(values) -> np.ndarray:
    """Physical values -> the unit cube, log-linearly between each bound pair."""
    vec = as_vector(values)
    if np.any(vec <= 0.0) or not np.all(np.isfinite(vec)):
        raise ValueError(f"every parameter is positive and finite; got {vec.tolist()}")
    return (np.log(vec) - _LOG_LO) / (_LOG_HI - _LOG_LO)


def from_unit(z) -> dict:
    """The unit cube -> physical values, clipped onto the bounds first."""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    if z.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} coordinates; got shape {z.shape}")
    return dict(zip(NAMES, np.exp(_LOG_LO + z * (_LOG_HI - _LOG_LO)).tolist()))


def in_bounds(values) -> bool:
    vec = as_vector(values)
    return bool(np.all(vec >= np.exp(_LOG_LO) * (1 - 1e-12))
                and np.all(vec <= np.exp(_LOG_HI) * (1 + 1e-12)))


def force_law(values, l0=AM.L0_SEED_M):
    """``(coeff, bf)`` per segment from the rest gains and shapes in *values*."""
    l0 = np.asarray(l0, dtype=float)
    gain = np.array([float(values[n]) for n in GAIN_NAMES])
    shape = np.array([float(values[n]) for n in SHAPE_NAMES])
    bf = shape * l0
    denom = 3.0 * l0 * l0 - bf * bf
    if np.any(denom <= 0.0):
        raise ValueError(f"bf/l0 must stay below sqrt(3); got {shape.tolist()}")
    return gain / denom, bf


def gain_and_shape(coeff, bf, l0=AM.L0_SEED_M):
    """The inverse of :func:`force_law`: ``(rest gain N/Pa, bf/l0)`` per segment."""
    return _rest_gain(coeff, bf, l0), np.asarray(bf, float) / np.asarray(l0, float)


def values_from_mech(doc: dict, l0=AM.L0_SEED_M) -> dict:
    """Parameter values from a mechanical JSON in the ``twin_params`` schema.

    Missing entries (an ``outer_fit`` file has no masses, no ``bf`` and no
    friction) take the start point's, so ``--start canarm_outer.json`` works.
    """
    values = start_values()
    bf = np.asarray(doc.get("bf", AM.BF_SEED_M), dtype=float)
    gain, shape = gain_and_shape(doc["coeff"], bf, l0)
    for i in range(3):
        values[GAIN_NAMES[i]] = float(gain[i])
        values[SHAPE_NAMES[i]] = float(shape[i])
    for key in DISSIPATION_NAMES:
        if key in doc:
            values[key] = float(doc[key])
    mjcf = doc.get("mjcf") or {}
    for key, names in (("link_mass_kg", MASS_NAMES[0:3]),
                       ("bracket_mass_kg", MASS_NAMES[3:6])):
        if key in mjcf:
            for n, v in zip(names, MG.resolve_per_segment(mjcf[key], key)):
                values[n] = float(v)
    return values


def moving_mass_kg(values) -> float:
    """Links + brackets + the tip stub: everything below the static base."""
    return float(sum(float(values[n]) for n in MASS_NAMES) + MG.DEFAULT_TIP_MASS_KG)


# ---------------------------------------------------------------------------
# From values to a twin
# ---------------------------------------------------------------------------

def twin_kwargs(values, base, *, damp_b1_scale: float = 1.0,
                joint_armature=None) -> dict:
    """``SimArm`` keywords for one candidate, built the way ``twin_params`` builds them.

    *base* is the flow checkpoint's ``ActuatorModel``; only ``coeff``, ``bf`` and,
    for a profile line, ``damp_b1`` are replaced.  ``test_mech_fit`` pins that
    a written checkpoint loads through ``twin_params.load_twin_kwargs`` to the same
    numbers, so the twin a candidate was scored as is the twin the GUI will run.
    """
    coeff, bf = force_law(values, base.l0)
    actuator = AM.ActuatorModel(
        net=base.net, is_tle=base.is_tle,
        fill_gain=base.fill_gain, vent_gain=base.vent_gain,
        blend_width_pa=base.blend_width_pa, leak_pa_s=base.leak_pa_s,
        coeff=coeff, bf=bf, l0=base.l0,
        damp_b1=np.asarray(base.damp_b1, dtype=float) * float(damp_b1_scale))
    actuator.meta = getattr(base, "meta", None)
    kw = {
        "actuator": actuator,
        "tendon_damping": float(values["tendon_damping"]),
        "joint_damping": float(values["joint_damping"]),
        "joint_frictionloss": float(values["joint_frictionloss"]),
        "link_mass_kg": tuple(float(values[n]) for n in MASS_NAMES[0:3]),
        "bracket_mass_kg": tuple(float(values[n]) for n in MASS_NAMES[3:6]),
    }
    if joint_armature is not None:
        kw["joint_armature"] = float(joint_armature)
    return kw


def mech_document(values, *, l0=AM.L0_SEED_M, status: str = "complete",
                  provenance: dict | None = None) -> dict:
    """The checkpoint, in the ``twin_params`` schema plus provenance keys."""
    vec = as_vector(values)
    values = dict(zip(NAMES, vec.tolist()))
    coeff, bf = force_law(values, l0)
    gain, shape = gain_and_shape(coeff, bf, l0)
    doc = {
        "coeff": coeff.tolist(),
        "bf": bf.tolist(),
        "tendon_damping": float(values["tendon_damping"]),
        "joint_damping": float(values["joint_damping"]),
        "joint_frictionloss": float(values["joint_frictionloss"]),
        "mjcf": {
            "link_mass_kg": [float(values[n]) for n in MASS_NAMES[0:3]],
            "bracket_mass_kg": [float(values[n]) for n in MASS_NAMES[3:6]],
        },
        "kind": "canarm mechanical fit: segment masses, force law, dissipation "
                "(digital_twin.mech_fit, CMA-ES)",
        "date": FIT_DATE,
        "status": status,
        "parameters": values,
        "rest_gain_n_per_psi": (gain * AM.PA_PER_PSI).tolist(),
        "bf_over_l0": shape.tolist(),
        "moving_mass_kg": moving_mass_kg(values),
        "bounds": {p.name: {"lo": p.lo, "hi": p.hi, "start": p.start,
                            "unit": p.unit, "evidence": p.evidence}
                   for p in PARAMS},
    }
    doc.update(provenance or {})
    return doc


def write_json(path: str, doc: dict) -> str:
    """Write atomically: a GUI loading the checkpoint mid-run sees old or new, never half."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowSpec:
    """A contiguous window, named by the planned segments it starts and ends in."""

    family: str
    session: str
    first: str
    last: str
    lead_in_s: float
    keep_last_s: float | None
    why: str


@dataclass(frozen=True)
class Window:
    family: str
    name: str
    session: str
    rows: tuple
    kinds: tuple
    seconds: float
    rec: R.Recording


#: Joints whose chirp is fitted: segment 1's proximal x axis, which carries the
#: whole arm's inertia, and segment 2's distal axis.
CHIRP_JOINTS = (0, 6)

#: Joints whose positive board's staircase is fitted, from the middle of the
#: chain to its tip: joint 4 carries segments 2-3, joint 8 segment 3, and joint 11
#: only the segment-3 bracket and the tip stub.
STAIR_JOINTS = (4, 8, 11)

#: Joints whose pair sweep is fitted (co-contraction 10 then 18 psi).  The sweep
#: phase recorded joints 0-6 in full before the session was cut; joint 2 is
#: segment 1's distal axis, which the staircases and chirps leave uncovered.
SWEEP_JOINTS = (2,)


def fixed_specs() -> list:
    """The windows that do not depend on which boards or episodes a session holds."""
    specs = [WindowSpec(
        "random_walk", SESSION_DYNAMICS, "walk_000", "walk_000", 1.0, 20.0,
        "all twelve joints retargeted at random; the lead-in is chirp_j11_rest")]
    for j in CHIRP_JOINTS:
        specs.append(WindowSpec(
            "chirp", SESSION_DYNAMICS, f"chirp_j{j}", f"chirp_j{j}_rest", 1.0, 1.5,
            f"joint {j} swept 0.15-4 Hz at 18 psi co-contraction, then 1.5 s settle"))
    for j in SWEEP_JOINTS:
        specs.append(WindowSpec(
            "pair_sweep", SESSION_STATICS, f"sweep_j{j}_co10", f"sweep_j{j}_co18",
            1.0, None, f"joint {j} differential swept at pair sums 10 and 18 psi"))
    return specs


def staircase_specs(planned: list, pairs) -> list:
    """One staircase window per :data:`STAIR_JOINTS`, found by the board's metadata."""
    specs = []
    for j in STAIR_JOINTS:
        board = int(pairs[j][0])
        steps = [s for s in planned if s.get("kind") == "staircase"
                 and int(str((s.get("meta") or {}).get("base", "0")), 16) == board]
        first = [s for s in steps if float(s["meta"].get("psi", -1)) == 4.0]
        vent = [s for s in steps if float(s["meta"].get("psi", -1)) == 0.0]
        if not first or not vent:
            raise ValueError(f"no complete staircase for board 0x{board:03X} (joint {j})")
        specs.append(WindowSpec(
            "staircase", SESSION_STATICS, first[0]["name"], vent[0]["name"], 1.0, 1.5,
            f"board 0x{board:03X} (joint {j} positive) stepped 4-24 psi against gravity"))
    return specs


def ringdown_candidates(planned: list) -> dict:
    """``{(segment, proximal): [(charge, release), ...]}`` in episode order."""
    out = {(s, p): [] for s in range(3) for p in (True, False)}
    for seg in planned:
        if seg.get("kind") != "ringdown_charge":
            continue
        joint = int((seg.get("meta") or {}).get("joint", -1))
        if joint < 0:
            continue
        release = seg["name"].replace("_charge", "_release")
        out[(joint // 4, joint % 4 < 2)].append((seg["name"], release))
    return out


def resolve_rows(planned: list, episode: np.ndarray, t_sync: np.ndarray,
                 spec: WindowSpec):
    """``(i0, i1, kinds)``: the rows of one window, refused if unusable.

    ``episode`` labels rows by planned segment index (``dataset.load_session``
    with ``episode_field="segment_index"``; :func:`check_labels` verifies the two
    agree).  Raises when the window names a held-out family, when it is not
    contiguous, when a segment it spans was never recorded, or when the lead-in
    would reach past the segment immediately before ``first``.
    """
    names = [s["name"] for s in planned]
    try:
        k0, k1 = names.index(spec.first), names.index(spec.last)
    except ValueError as exc:
        raise ValueError(f"{spec.family}: planned segment missing ({exc})") from exc
    if k1 < k0:
        raise ValueError(f"{spec.first} comes after {spec.last}")
    k_lead = k0 - 1 if spec.lead_in_s > 0.0 else k0
    if k_lead < 0:
        raise ValueError(f"{spec.first} has no segment before it for a lead-in")
    kinds = tuple(dict.fromkeys(str(planned[k]["kind"]) for k in range(k_lead, k1 + 1)))
    held = sorted(set(kinds) & set(HELDOUT_KINDS))
    if held or spec.family in HELDOUT_KINDS:
        raise ValueError(f"window {spec.first}..{spec.last} spans held-out "
                         f"family {held or [spec.family]}; it may never be fitted")
    episode = np.asarray(episode)
    for k in range(k_lead, k1 + 1):
        if not np.any(episode == k):
            raise ValueError(f"planned segment {names[k]} (index {k}) was not recorded")
    rows0 = np.flatnonzero(episode == k0)
    rows1 = np.flatnonzero(episode == k1)
    i0, i1 = int(rows0[0]), int(rows1[-1]) + 1
    if spec.keep_last_s is not None:
        stop = t_sync[rows1[0]] + float(spec.keep_last_s)
        i1 = int(rows1[0] + np.searchsorted(t_sync[rows1], stop, side="right"))
    if spec.lead_in_s > 0.0:
        i_lead = int(np.searchsorted(t_sync, t_sync[i0] - float(spec.lead_in_s), side="left"))
        if episode[i_lead] != k_lead:
            raise ValueError(f"a {spec.lead_in_s} s lead-in before {spec.first} reaches "
                             f"past {names[k_lead]}")
        i0 = i_lead
    span = episode[i0:i1]
    if np.any(np.diff(span) < 0) or span.min() != k_lead or span.max() != k1:
        raise ValueError(f"rows of {spec.first}..{spec.last} are not in segment order")
    gaps = np.diff(t_sync[i0:i1])
    if gaps.size and float(gaps.max()) > MAX_GAP_S:
        raise ValueError(f"window {spec.first}..{spec.last} is not contiguous: a "
                         f"{float(gaps.max()):.3f} s sync gap")
    return i0, i1, kinds


def check_labels(planned: list, episode: np.ndarray, phase: np.ndarray) -> None:
    """Every recorded episode's kind is the planned kind at that index, or raise."""
    episode = np.asarray(episode)
    phase = np.asarray(phase, dtype=object)
    for k in np.unique(episode):
        kinds = set(str(v) for v in np.unique(phase[episode == k]))
        want = str(planned[int(k)]["kind"]) if int(k) < len(planned) else None
        if kinds != {want}:
            raise ValueError(f"episode {int(k)} carries kinds {sorted(kinds)} but the "
                             f"plan's segment {int(k)} is {want!r}; episode labels are "
                             f"not segment indices here")


def slice_rows(rec: R.Recording, i0: int, i1: int) -> R.Recording:
    """Rows ``[i0, i1)`` of a recording, with the relative clock re-zeroed."""
    sl = slice(int(i0), int(i1))
    t = rec.t_sync_s[sl]
    return R.Recording(
        t_sync_s=t, t_rel_s=t - t[0], cycle=rec.cycle[sl], ids=rec.ids,
        board_type=rec.board_type, is_tle=rec.is_tle, q_rad=rec.q_rad[sl],
        qdot_rad_s=rec.qdot_rad_s[sl], p_adc=rec.p_adc[sl],
        target_adc=rec.target_adc[sl], p_pa=rec.p_pa[sl],
        target_pa=rec.target_pa[sl], q_valid=rec.q_valid[sl],
        meta={"rows": [int(i0), int(i1)]}, path=rec.path)


def build_windows(data_dir: str, *, log=print) -> list:
    """Load both sessions and cut every training window, refusing any unusable one."""
    from digital_twin import dataset as ds
    from UMArm_KINEMATICS import canarm_actuators as CA

    pairs = CA.joint_pairs()
    loaded = {}
    for name in (SESSION_STATICS, SESSION_DYNAMICS):
        path = os.path.join(data_dir, name)
        with open(os.path.join(path, "metadata.json"), encoding="utf-8") as fh:
            planned = json.load(fh)["segments_planned"]
        labels = ds.load_session(path, episode_field="segment_index")
        rec = R.recording_from_session(path)
        if rec.n != labels.n_cycles or not np.array_equal(rec.t_sync_s, labels.can_sync_time_s):
            raise ValueError(f"{name}: the two readers disagree on the rows")
        check_labels(planned, labels.episode, labels.phase)
        loaded[name] = (planned, labels.episode, rec)

    specs = fixed_specs() + staircase_specs(loaded[SESSION_STATICS][0], pairs)
    windows = []

    def cut(spec):
        planned, episode, rec = loaded[spec.session]
        i0, i1, kinds = resolve_rows(planned, episode, rec.t_sync_s, spec)
        sub = slice_rows(rec, i0, i1)
        posed = float(np.mean(sub.q_valid))
        if posed < MIN_POSED_FRACTION:
            raise ValueError(f"{spec.first}..{spec.last}: only {posed:.3f} of rows posed")
        return Window(family=spec.family, name=f"{spec.first}..{spec.last}",
                      session=spec.session, rows=(i0, i1), kinds=kinds,
                      seconds=float(sub.duration_s), rec=sub)

    for spec in specs:
        windows.append(cut(spec))

    planned2 = loaded[SESSION_DYNAMICS][0]
    by_name = {s["name"]: s for s in planned2}
    for (segment, proximal), options in ringdown_candidates(planned2).items():
        chosen = None
        for charge, release in options:
            spec = WindowSpec("ringdown", SESSION_DYNAMICS, charge, release, 0.0, None,
                              f"segment {segment + 1} {'proximal' if proximal else 'distal'} "
                              f"joint {by_name[charge]['meta']['joint']} charged then released")
            try:
                w = cut(spec)
            except ValueError as exc:
                log(f"  skip {charge}: {exc}")
                continue
            planned_s = float(by_name[charge]["duration_s"]) + float(by_name[release]["duration_s"])
            if w.seconds < planned_s - 0.1:
                log(f"  skip {charge}: {w.seconds:.2f} s recorded of {planned_s:.2f} s planned")
                continue
            chosen = w
            break
        if chosen is None:
            raise ValueError(f"no usable ringdown for segment {segment + 1} "
                             f"{'proximal' if proximal else 'distal'}")
        windows.append(chosen)
    return windows


def describe_windows(windows) -> list:
    return [{"family": w.family, "name": w.name, "session": w.session,
             "rows": list(w.rows), "kinds": list(w.kinds),
             "seconds": round(w.seconds, 3)} for w in windows]


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------

def loss_from_result(tw) -> tuple:
    """``(loss_deg, valid, reason)`` for one :class:`twin_compare.TwinResult`."""
    if not tw.valid:
        return PENALTY_DEG, False, str(tw.reason)[:300]
    real = np.asarray(tw.real_defl_rad, dtype=float)
    sim = np.asarray(tw.twin_defl_rad, dtype=float)
    use = np.all(np.isfinite(real), axis=1) & np.all(np.isfinite(sim), axis=1)
    if int(use.sum()) < TC.MIN_SAMPLES:
        return PENALTY_DEG, False, (f"{int(use.sum())} usable rows, fewer than "
                                    f"{TC.MIN_SAMPLES}")
    err = sim[use] - real[use]
    loss = float(np.mean(np.degrees(np.sqrt(np.mean(err * err, axis=0)))))
    if not math.isfinite(loss):
        return PENALTY_DEG, False, "non-finite joint error"
    return loss, True, ""


def window_loss(rec: R.Recording, kwargs: dict) -> tuple:
    """Roll one window open loop and score it, penalising every invalid outcome."""
    import mujoco

    from digital_twin import sim_core as SC

    variants = {int(b): int(v) for b, v in zip(rec.ids, rec.board_type)}
    try:
        # batched_actuator: one (24, 5) forward per node pass instead of 24;
        # test_sim_core pins it to the scalar path, and it is 0.40 against 0.51
        # s of wall per simulated second on the SimMaster benchmark.
        arm = SC.SimArm(variants=variants, batched_actuator=True, **kwargs)
        tw = TC.twin_rollout(rec, arm=arm)
    except Exception as exc:                                  # noqa: BLE001
        return PENALTY_DEG, False, f"{type(exc).__name__}: {exc}"[:300]
    resets = int(arm.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    if resets:
        return PENALTY_DEG, False, f"MuJoCo reset the state {resets} time(s) (bad qacc)"
    return loss_from_result(tw)


def aggregate(losses, families, valid=None) -> tuple:
    """``(total, per_family)``: mean of windows within a family, then of families.

    *losses* is ``(n_candidates, n_windows)``; *families* names each column;
    *valid*, the same shape, marks the windows that rolled cleanly.  A candidate
    with any invalid window scores ``PENALTY_DEG + (number invalid)``, so it
    ranks below every valid candidate and a candidate with fewer invalid windows
    still ranks above one with more.  ``per_family`` keeps the plain means.
    """
    losses = np.atleast_2d(np.asarray(losses, dtype=float))
    fam = np.asarray(list(families), dtype=object)
    if losses.shape[1] != fam.size:
        raise ValueError(f"{losses.shape[1]} window columns but {fam.size} family labels")
    order = [f for f in FAMILIES if np.any(fam == f)] + sorted(
        {str(f) for f in fam} - set(FAMILIES))
    per = {f: losses[:, fam == f].mean(axis=1) for f in order}
    total = np.mean(np.stack([per[f] for f in order]), axis=0)
    if valid is not None:
        n_bad = np.sum(~np.atleast_2d(np.asarray(valid, dtype=bool)), axis=1)
        total = np.where(n_bad > 0, PENALTY_DEG + n_bad, total)
    return total, per


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(windows, flow_path) -> None:
    if not flow_path or not os.path.isfile(flow_path):
        raise RuntimeError(f"mech_fit needs the flow checkpoint; none at {flow_path}")
    _WORKER["windows"] = windows
    _WORKER["base"] = AM.ActuatorModel.load(flow_path)


def _evaluate_task(task) -> tuple:
    ci, wi, vec, extra = task
    t0 = time.perf_counter()
    values = dict(zip(NAMES, [float(v) for v in vec]))
    kw = twin_kwargs(values, _WORKER["base"], **(extra or {}))
    loss, valid, reason = window_loss(_WORKER["windows"][wi].rec, kw)
    return ci, wi, float(loss), bool(valid), reason, time.perf_counter() - t0


def evaluate_candidates(pool, candidates, windows) -> tuple:
    """``(total, per_family, losses, valid, reasons)`` for ``[(values, extra), ...]``.

    Every (candidate, window) pair is one task, longest windows first, so the
    pool stays full while the short ringdowns fill the gaps.
    """
    from digital_twin import mech_fit as _self

    n_c, n_w = len(candidates), len(windows)
    order = sorted(range(n_w), key=lambda w: -windows[w].seconds)
    tasks = [(ci, wi, as_vector(vals).tolist(), extra)
             for ci, (vals, extra) in enumerate(candidates) for wi in order]
    losses = np.full((n_c, n_w), PENALTY_DEG)
    valid = np.zeros((n_c, n_w), dtype=bool)
    reasons = {}
    for ci, wi, loss, ok, reason, _wall in pool.imap_unordered(
            _self._evaluate_task, tasks, chunksize=1):
        losses[ci, wi] = loss
        valid[ci, wi] = ok
        if not ok:
            reasons[(ci, wi)] = reason
    total, per = aggregate(losses, [w.family for w in windows], valid)
    return total, per, losses, valid, reasons


def _make_pool(workers, windows, flow_path):
    import multiprocessing as mp

    from digital_twin import mech_fit as _self

    for key in THREAD_ENV:
        os.environ[key] = "1"
    ctx = mp.get_context("spawn")
    return ctx.Pool(int(workers), initializer=_self._init_worker,
                    initargs=(windows, str(flow_path)))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def profile_candidates(best, *, factors=PROFILE_FACTORS) -> list:
    """``[(line, factor, values, extra), ...]`` around *best*.

    Lines: each of the six masses alone; every mass and every rest gain together
    (the direction the statics cannot see); those plus the three fitted
    dissipation scalars and ``damp_b1`` (the direction only the joint armature
    breaks); and the armature alone at 0.5x and 2x, which is not fitted and is
    here to say how much of the answer it holds.
    """
    best = dict(zip(NAMES, as_vector(best).tolist()))
    out = [("best", 1.0, dict(best), {})]
    outer = [k for k in factors if k != 1.0]
    for name in MASS_NAMES:
        for k in outer:
            v = dict(best)
            v[name] = best[name] * k
            out.append((name, k, v, {}))
    for k in outer:
        v = dict(best)
        for name in MASS_NAMES + GAIN_NAMES:
            v[name] = best[name] * k
        out.append(("mass_and_gain", k, v, {}))
    for k in outer:
        v = dict(best)
        for name in MASS_NAMES + GAIN_NAMES + DISSIPATION_NAMES:
            v[name] = best[name] * k
        out.append(("mass_gain_and_all_dissipation", k, v, {"damp_b1_scale": k}))
    for k in (0.5, 2.0):
        out.append(("joint_armature", k, dict(best),
                    {"joint_armature": MG.JOINT_ARMATURE * k}))
    return out


def summarise_profiles(cands, totals, per_family) -> dict:
    """Per line: the losses, their rise over the optimum, and whether it pins."""
    best_loss = float(totals[[i for i, c in enumerate(cands) if c[0] == "best"][0]])
    lines = {}
    for i, (line, k, _v, _e) in enumerate(cands):
        if line == "best":
            continue
        entry = lines.setdefault(line, {"factor": [1.0], "loss_deg": [best_loss],
                                        "per_family_deg": [None]})
        entry["factor"].append(float(k))
        entry["loss_deg"].append(float(totals[i]))
        entry["per_family_deg"].append({f: float(v[i]) for f, v in per_family.items()})
    thr = PIN_FRACTION * best_loss
    for line, entry in lines.items():
        order = np.argsort(entry["factor"])
        for key in ("factor", "loss_deg", "per_family_deg"):
            entry[key] = [entry[key][j] for j in order]
        f = np.asarray(entry["factor"])
        loss = np.asarray(entry["loss_deg"])
        entry["rise_deg"] = (loss - best_loss).tolist()
        low_side = loss[f < 1.0]
        high_side = loss[f > 1.0]
        low = bool(low_side.size and (low_side[0] - best_loss) > thr)
        high = bool(high_side.size and (high_side[-1] - best_loss) > thr)
        entry["pinned"] = ("both sides" if low and high else
                           "below only" if low else "above only" if high else "not pinned")
        entry["within_1pct_factor_range"] = _flat_range(f, loss, best_loss + thr)
    return {"best_loss_deg": best_loss, "pin_threshold_deg": thr,
            "factors": list(PROFILE_FACTORS), "lines": lines}


def _flat_range(factors, loss, ceiling) -> list:
    """The factor interval (log-interpolated) over which the loss stays under *ceiling*."""
    lf = np.log(np.asarray(factors, dtype=float))
    loss = np.asarray(loss, dtype=float)
    i1 = int(np.argmin(np.abs(lf)))
    lo = lf[0]
    for j in range(i1, 0, -1):
        if loss[j - 1] > ceiling:
            a, b = loss[j - 1], loss[j]
            lo = lf[j] if a == b else lf[j] + (lf[j - 1] - lf[j]) * (ceiling - b) / (a - b)
            break
    hi = lf[-1]
    for j in range(i1, lf.size - 1):
        if loss[j + 1] > ceiling:
            a, b = loss[j], loss[j + 1]
            hi = lf[j] if a == b else lf[j] + (lf[j + 1] - lf[j]) * (ceiling - a) / (b - a)
            break
    return [float(np.exp(lo)), float(np.exp(hi))]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class _Log:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        line = str(text)
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def _fmt_values(values) -> str:
    v = dict(zip(NAMES, as_vector(values).tolist()))
    return ("mass link " + "/".join(f"{v[n]:.3f}" for n in MASS_NAMES[0:3])
            + " bracket " + "/".join(f"{v[n]:.3f}" for n in MASS_NAMES[3:6])
            + f" kg (moving {moving_mass_kg(v):.3f}) | K "
            + "/".join(f"{v[n] * AM.PA_PER_PSI:.3f}" for n in GAIN_NAMES)
            + " N/psi | bf/l0 " + "/".join(f"{v[n]:.3f}" for n in SHAPE_NAMES)
            + f" | tendon {v['tendon_damping']:.4g} joint {v['joint_damping']:.4g}"
              f" fric {v['joint_frictionloss']:.4g}")


def run_fit(windows, *, flow, out, log, evals_path, workers=DEFAULT_WORKERS,
            popsize=DEFAULT_POPSIZE, generations=DEFAULT_GENERATIONS,
            max_hours=DEFAULT_MAX_HOURS, sigma0=DEFAULT_SIGMA0, seed=20260910,
            start=None, profile=True) -> dict:
    """CMA-ES over :data:`PARAMS`; writes the best-so-far checkpoint every generation."""
    import cma

    t_start = time.perf_counter()
    seconds = float(sum(w.seconds for w in windows))
    budget = {
        "seconds_of_arm_per_candidate": seconds,
        "popsize": int(popsize), "generations_planned": int(generations),
        "workers": int(workers), "max_hours": float(max_hours),
        "seconds_of_arm_planned": seconds * popsize * generations,
        "estimated_wall_h": seconds * popsize * generations / MEASURED_ARM_S_PER_WALL_S / 3600.0,
        "throughput_note": (f"{MEASURED_ARM_S_PER_WALL_S} s of arm per wall s measured at "
                            f"{DEFAULT_WORKERS} workers on 2026-09-10"),
    }
    log(f"windows: {len(windows)}, {seconds:.1f} s of arm per candidate")
    for w in windows:
        log(f"  {w.family:<11s} {w.name:<40s} {w.seconds:6.2f} s  rows {w.rows}")
    log(f"budget: {seconds:.1f} s x {popsize} candidates x {generations} generations = "
        f"{budget['seconds_of_arm_planned'] / 3600:.1f} h of arm, about "
        f"{budget['estimated_wall_h']:.2f} h of wall clock at the measured throughput; "
        f"cap {max_hours} h")

    start = dict(start or start_values())
    z0 = to_unit(start)
    es = cma.CMAEvolutionStrategy(z0.tolist(), float(sigma0), {
        "bounds": [0.0, 1.0], "popsize": int(popsize), "seed": int(seed),
        "maxiter": int(generations), "verbose": -9})
    fams = [w.family for w in windows]
    provenance_base = {
        "flow_checkpoint": os.path.abspath(str(flow)),
        "families": {f: [d for d in describe_windows(windows) if d["family"] == f]
                     for f in FAMILIES},
        "heldout_families_never_rolled": list(HELDOUT_KINDS),
        "objective": ("mean over five families (equal weight) of the mean over "
                      "windows of the twelve-joint mean joint deflection RMS, deg, "
                      "open loop through twin_compare.twin_rollout"),
        "optimizer": {"name": "CMA-ES", "package": f"cma {cma.__version__}",
                      "space": "log, mapped onto [0, 1] between each bound pair",
                      "sigma0_unit": float(sigma0), "seed": int(seed),
                      "penalty_deg": PENALTY_DEG},
        "budget": budget,
        "start": {"values": start, "why": "outer_fit's gains and damping, the "
                  "Koopman ProMax prior masses, the RS485 shape and friction"},
    }

    evals = open(evals_path, "a", encoding="utf-8")
    best = {"loss": math.inf}
    history = []

    def record(tag, cand_values, totals, per, losses, valid, reasons):
        for i, vals in enumerate(cand_values):
            evals.write(json.dumps({
                "tag": tag, "loss_deg": float(totals[i]),
                "per_family_deg": {f: float(v[i]) for f, v in per.items()},
                "window_loss_deg": losses[i].tolist(),
                "invalid": {windows[wi].name: reasons[(ci, wi)]
                            for (ci, wi) in reasons if ci == i},
                "values": dict(zip(NAMES, as_vector(vals).tolist()))}) + "\n")
        evals.flush()
        k = int(np.argmin(totals))
        if totals[k] < best["loss"]:
            best.update(loss=float(totals[k]), values=dict(zip(NAMES, as_vector(cand_values[k]).tolist())),
                        per_family={f: float(v[k]) for f, v in per.items()},
                        windows={windows[w].name: float(losses[k, w]) for w in range(len(windows))},
                        tag=tag)

    with _make_pool(workers, windows, flow) as pool:
        t0 = time.perf_counter()
        totals, per, losses, valid, reasons = evaluate_candidates(pool, [(start, {})], windows)
        record("start", [start], totals, per, losses, valid, reasons)
        log(f"start: {totals[0]:.3f} deg  " + "  ".join(f"{f} {v[0]:.3f}" for f, v in per.items())
            + f"  ({time.perf_counter() - t0:.0f} s)")
        log(f"       {_fmt_values(start)}")
        start_loss = float(totals[0])

        gen = 0
        while not es.stop() and gen < generations:
            if time.perf_counter() - t_start > max_hours * 3600.0:
                log(f"wall-clock cap of {max_hours} h reached after {gen} generations")
                break
            tg = time.perf_counter()
            Z = es.ask()
            cands = [from_unit(z) for z in Z]
            totals, per, losses, valid, reasons = evaluate_candidates(
                pool, [(c, {}) for c in cands], windows)
            es.tell(Z, totals.tolist())
            gen += 1
            record(f"gen{gen}", cands, totals, per, losses, valid, reasons)
            n_bad = int((~valid).any(axis=1).sum())
            history.append({"generation": gen, "best_deg": float(totals.min()),
                            "median_deg": float(np.median(totals)),
                            "best_so_far_deg": best["loss"], "sigma": float(es.sigma),
                            "invalid_candidates": n_bad,
                            "wall_s": time.perf_counter() - tg})
            elapsed = time.perf_counter() - t_start
            log(f"gen {gen:3d}/{generations}  best {totals.min():7.3f}  median "
                f"{np.median(totals):7.3f}  so far {best['loss']:7.3f} deg  sigma "
                f"{es.sigma:.4f}  invalid {n_bad:2d}  {time.perf_counter() - tg:5.0f} s  "
                f"elapsed {elapsed / 3600:.2f} h")
            log(f"         best so far: {_fmt_values(best['values'])}")
            write_json(out, mech_document(best["values"], status=(
                f"running: generation {gen} of {generations}"), provenance=dict(
                    provenance_base, fit={
                        "loss_deg": best["loss"], "start_loss_deg": start_loss,
                        "per_family_deg": best["per_family"], "window_loss_deg": best["windows"],
                        "generations_done": gen, "evaluations": 1 + gen * popsize,
                        "elapsed_h": elapsed / 3600.0, "history": history})))

        stop = {k: str(v) for k, v in es.stop().items()} if es.stop() else {}
        fit_block = {
            "loss_deg": best["loss"], "start_loss_deg": start_loss,
            "per_family_deg": best["per_family"], "window_loss_deg": best["windows"],
            "generations_done": gen, "evaluations": 1 + gen * popsize,
            "seconds_of_arm_rolled": seconds * (1 + gen * popsize),
            "elapsed_h": (time.perf_counter() - t_start) / 3600.0,
            "cma_stop": stop, "history": history}
        log(f"fit done: {best['loss']:.3f} deg against {start_loss:.3f} at the start, "
            f"{gen} generations, {1 + gen * popsize} evaluations")
        doc = mech_document(best["values"], status="complete (profiles pending)"
                            if profile else "complete",
                            provenance=dict(provenance_base, fit=fit_block))
        write_json(out, doc)

        if profile:
            doc["profiles"] = run_profiles(pool, windows, best["values"], log=log)
            doc["status"] = "complete"
            write_json(out, doc)
    evals.close()
    log(f"wrote {out}")
    return doc


def run_profiles(pool, windows, best_values, *, log=print) -> dict:
    """Evaluate :func:`profile_candidates` and summarise which masses the data pins."""
    cands = profile_candidates(best_values)
    t0 = time.perf_counter()
    totals, per, _losses, _valid, reasons = evaluate_candidates(
        pool, [(v, e) for _line, _k, v, e in cands], windows)
    summary = summarise_profiles(cands, totals, per)
    summary["evaluations"] = len(cands)
    summary["invalid"] = {f"{cands[ci][0]} x{cands[ci][1]}": r for (ci, _wi), r in reasons.items()}
    log(f"profiles: {len(cands)} evaluations in {time.perf_counter() - t0:.0f} s; "
        f"optimum {summary['best_loss_deg']:.3f} deg, pin threshold "
        f"{summary['pin_threshold_deg']:.3f} deg")
    for line, entry in summary["lines"].items():
        log(f"  {line:<32s} " + "  ".join(f"x{f:<4g} {l:7.3f}" for f, l in
                                           zip(entry["factor"], entry["loss_deg"]))
            + f"  -> {entry['pinned']}, flat over x{entry['within_1pct_factor_range'][0]:.2f}"
              f"..x{entry['within_1pct_factor_range'][1]:.2f}")
    return summary


def main(argv=None) -> int:
    from digital_twin import twin_params as TP

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(_WS, "data"))
    ap.add_argument("--flow", default=str(TP.DEFAULT_FLOW))
    ap.add_argument("--out", default=str(TP.DEFAULT_MECH))
    ap.add_argument("--log-dir", default=os.path.join(_WS, "data", "fit"))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    ap.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    ap.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    ap.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--start", default=None,
                    help="a mechanical JSON (canarm_mech.json or canarm_outer.json) to "
                         "start from instead of the built-in start point")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--profile-only", default=None, metavar="MECH_JSON",
                    help="skip the fit; profile the parameters in this file and write "
                         "the result into it")
    ap.add_argument("--plan-only", action="store_true",
                    help="cut and print the windows and the budget, roll nothing")
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = _Log(os.path.join(args.log_dir, f"mech_fit_{stamp}.log"))
    log(f"digital_twin.mech_fit {stamp}: {' '.join(sys.argv[1:] if argv is None else argv)}")
    windows = build_windows(args.data_dir, log=log)
    seconds = sum(w.seconds for w in windows)
    if args.plan_only:
        for w in windows:
            log(f"  {w.family:<11s} {w.name:<40s} {w.seconds:6.2f} s  kinds {w.kinds}")
        log(f"{len(windows)} windows, {seconds:.1f} s of arm per candidate")
        return 0

    if args.profile_only:
        with open(args.profile_only, encoding="utf-8") as fh:
            doc = json.load(fh)
        values = values_from_mech(doc)
        with _make_pool(args.workers, windows, args.flow) as pool:
            doc["profiles"] = run_profiles(pool, windows, values, log=log)
        write_json(args.profile_only, doc)
        log(f"wrote profiles into {args.profile_only}")
        return 0

    start = None
    if args.start:
        with open(args.start, encoding="utf-8") as fh:
            start = values_from_mech(json.load(fh))
        start = {k: float(np.clip(v, p.lo, p.hi)) for (k, v), p in zip(start.items(), PARAMS)}
    run_fit(windows, flow=args.flow, out=args.out, log=log,
            evals_path=os.path.join(args.log_dir, f"mech_fit_{stamp}_evals.jsonl"),
            workers=args.workers, popsize=args.popsize, generations=args.generations,
            max_hours=args.max_hours, sigma0=args.sigma0, seed=args.seed,
            start=start, profile=not args.no_profile)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
