r"""The valve-and-muscle model: pressure error in, ``dp/dt`` out, force out.

Pure numpy, and it must stay that way.  ``sim_core`` calls
:meth:`ActuatorModel.net_flow_pa_s_batch` and :meth:`ActuatorModel.force_n` once
per 1 ms quantum, and an eight-second replay is therefore eight thousand calls;
a rollout has to run on the bench laptop, which has no CUDA.  ``import torch``
anywhere in this module's import graph is a defect, and
``test_actuator_model.py`` asserts against it.

WHAT IS MODELLED HERE, and what deliberately is not.  Three pieces, decoupled:

* a small **flow net** (5 -> 64 -> 64 -> 1, tanh hidden, linear output) shared by
  all 24 boards, predicting ``dp/dt`` in units of :data:`DP_SCALE_PA_S` and
  nothing else;
* per-node **fill/vent gains** blended by a logistic in the commanded error, and
  a per-node **leak**, which is fitted by least squares outside the net and
  subtracted at the plant seam;
* the closed-form **anchored McKibben force law** and its pressure-scheduled
  tendon damping, which share the net's ``l`` and share no parameters with it.

The leak is kept out of the net for a reason the RS485 fit paid for: per-tick
leak is of order 36 Pa per 50 ms, below one ADC count and below the sensor
noise, so a net asked to learn it learns noise instead and idle nodes
self-inflate.  The measured failure there was ``net_flow(p=0, closed)`` coming
out at +3.4 kPa/s, i.e. every resting board gaining about 0.4 psi/s, which
refused every joint at the 0.5 psi rest interlock.

THE INPUT ROW IS THE CONTRACT (``CONTRACT.md`` section 2), five features::

    x = [p/P_SCALE_PA, (target-p)/E_SCALE_PA, is_tle, l/L_SCALE_M, ldot/LDOT_SCALE_M_S]

The departures from the RS485 reference are stated in ``CONTRACT.md`` section 0
and repeated here only where the code would otherwise look arbitrary:

1. **No valve one-hot.**  ``tlelib.proto.CompactStatus`` carries ``ENABLED``,
   ``OTA_ACTIVE``, ``COMMAND_SEEN`` and ``ERROR`` and no valve state, so the
   reference's ``ST_INFLATING``/``ST_VENTING`` one-hot is not recordable on this
   bus.  The commanded error ``e = target - p`` replaces it: it is recorded, it
   is what both firmwares act on, and it is continuous, so a proportional
   TLE92464 valve and a bang-bang 7 mm solenoid pair are described in the same
   coordinate instead of the latter's three-state alphabet.
2. **A population bit.**  ``is_tle`` is an input, and the per-node scalars stay
   per node, because a single number spanning both valve populations describes
   neither.  It comes from the variant byte the board answered
   (:func:`is_tle_from_variants`), never from the id range -- a TLE board sat at
   ``0x114`` for the whole 2026-08 bench session.
3. **The gain switch is a logistic blend, not a branch.**  The shooting loss
   back-propagates through the fill/vent selection, so a step at ``e = 0`` puts
   a kink in the middle of the operating region.  The blend reduces to the
   reference's hard switch as its width goes to zero, which
   ``test_gain_blend_reduces_to_hard_switch`` checks.

BECAUSE ``p`` ENTERS THE FEATURE ROW TWICE -- once as column 0 and once, with
the opposite sign, inside the error of column 1 -- the state Jacobian the
shooting BPTT needs is *not* the reference's ``dy/dx0``.  It is
``dy/dx0 / P_SCALE_PA - dy/dx1 / E_SCALE_PA``.  :meth:`FlowNet.forward_full`
returns that total derivative, and a finite-difference test compares it against
the whole feature construction rather than against column 0 alone, because
getting this wrong costs a term of the same order and opposite sign and still
trains.

NOTHING IN THIS FILE IS FITTED TO THIS ARM YET.  :meth:`ActuatorModel.fresh`
seeds the segment constants from two sources and labels which is which; see
:data:`L0_SEED_M`, :data:`BF_SEED_M` and :data:`COEFF_SEED`.  A checkpoint is
what makes the model a measurement, and until one exists every number below is a
starting point for a fit.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\actuator_model.py``, read in
detail at ``digital_twin/reference/actuator_net.md`` section 2.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterable, Sequence

import numpy as np

# ``UMArm_KINEMATICS`` is a sibling package of ``digital_twin`` under the
# workspace root, not a subpackage, so there is no relative import that reaches
# it; the root goes on ``sys.path`` the same way every hw_test in this workspace
# does it.  The measured rest lengths are worth this: the alternative is a
# second copy of the chain table, and a second copy is what drifts.
try:
    from UMArm_KINEMATICS import canarm_params as _cp
except ImportError:  # pragma: no cover - depends on how the caller was started
    _WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _WS_ROOT not in sys.path:
        sys.path.insert(0, _WS_ROOT)
    from UMArm_KINEMATICS import canarm_params as _cp  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Units and normalisation.  Every one of these is written into a checkpoint's
# meta and re-checked on load; see ActuatorModel.load.
# ---------------------------------------------------------------------------

#: Pa per psi.  psi appears only at a human boundary -- the operator GUI, the
#: campaign's ``--psi`` argument -- and never inside this module's arithmetic.
PA_PER_PSI = 6894.757

#: Boards, and the two partitions over them.  ``pam_k`` is board ``0x100 + k``
#: for ``k`` in 1..24, so array index ``k - 1`` is board ``0x100 + k``.
N_ACT = 24
N_SEG = 3

#: Actuator -> segment, 0-based.  The segment blocks are the regulator
#: platforms, ``0x101``-``0x108`` / ``0x109``-``0x110`` / ``0x111``-``0x118``,
#: verified 24 for 24 by the 2026-08-21 drive campaign
#: (``UMArm_KINEMATICS.canarm_actuators.SEGMENT_BLOCKS``: every board's dominant
#: ``dq`` fell inside its own block's four joints).  This is a *different*
#: partition from the valve population even though segment 0 happens to hold the
#: eight TLE boards today, and the two are given separate index vectors here for
#: exactly that reason.
SEG_INDEX = np.repeat(np.arange(N_SEG), N_ACT // N_SEG)

#: Population index into :attr:`ActuatorModel.blend_width_pa`.
POP_7MM = 0
POP_TLE = 1

#: Firmware variant byte that identifies a TLE92464/DVP board
#: (``tlelib.proto.VARIANT_TLE_DVP``).  Mirrored rather than imported so this
#: module stays importable with no bus tooling on the path.
VARIANT_TLE_DVP = 0x02

N_IN = 5                                   # p, e, is_tle, l, ldot
HIDDEN = (64, 64)                          # two tanh layers, linear output

#: The operator's per-line cap is 30 psi (``CONTRACT.md`` section 8), so a
#: normalised pressure of 1.0 is the largest pressure it is ever legal to
#: command on this arm.  ``e`` takes the same scale so the two columns are
#: commensurate: a full-scale error and a full-scale pressure both read 1.
P_SCALE_PA = 30.0 * PA_PER_PSI             # 206842.71 Pa
E_SCALE_PA = 30.0 * PA_PER_PSI             # 206842.71 Pa

#: 100 kPa/s.  The RS485 rig's measured fast fill was about 275 kPa/s, putting
#: the net's output in roughly [-3, 3]; this arm's TLE boards are slew-limited
#: in firmware and are expected slower, so the same scale keeps the output
#: inside the tanh stack's useful range rather than pinning it near zero.
DP_SCALE_PA_S = 1.0e5

#: Muscle lengths on this arm run 0.142 to 0.178 m at rest (:data:`L0_SEED_M`)
#: plus a bent-arm tendon excursion of about +/-25 mm, so ``l/L_SCALE_M`` lands
#: in roughly [1.2, 2.0].
L_SCALE_M = 0.1
LDOT_SCALE_M_S = 0.5

#: Newtons.  A guard rail on the magnitude of a pull rather than a fitted
#: quantity: ``replay.assert_no_clamp`` raises if a rollout ever touches it,
#: because a rollout that clips is reporting the clip and not the model.
FORCE_CLIP_N = 4000.0

#: Fill/vent gain clip applied by the trainer after each gains step.  A negative
#: or exploding gain is a numerical accident rather than a pneumatic circuit.
GAIN_CLIP = (0.05, 20.0)

#: Leak clip, Pa/s.  The seam *subtracts* the leak, so a negative fitted slope
#: (sensor noise read as self-inflation) would ship as idle-board inflation.
LEAK_CLIP_PA_S = (0.0, 2000.0)

#: Logistic blend width per population, Pa, as ``(7mm, tle)``.  **Both are
#: assumptions to be re-measured, not fits.**  2000 Pa is the 7 mm firmware's
#: own +/-2000 Pa bang-bang hysteresis, so the blend is soft across exactly the
#: band in which that firmware does nothing.  6000 Pa is the proportional band
#: the TLE firmware's dead zone and slew limit imply, and it is the number in
#: this file most likely to be wrong.
DEFAULT_BLEND_WIDTH_PA = (2000.0, 6000.0)


# ---------------------------------------------------------------------------
# Segment constants.  READ THE PROVENANCE LINE ON EACH ONE.
# ---------------------------------------------------------------------------

#: **MEASURED ON THIS ARM.**  Anchored rest length per segment, metres, taken
#: straight from ``UMArm_KINEMATICS.canarm_params.CANARM_PARAMS`` column ``LL``:
#: ``(0.177932, 0.146946, 0.142433)`` m.  ``LL`` is the free span between the two
#: actuator attachment points at zero joint angle -- that table is built so
#: ``AA1 + LL + AA2`` reproduces the measured u-joint centre distance -- so it is
#: the muscle's rest length by construction rather than by analogy.  Behind it
#: are the five plate-gap distances measured over 68 poses on 2026-08-21 with
#: standard deviations of 0.03 to 0.47 mm.
L0_SEED_M = tuple(float(v) for v in _cp.CANARM_PARAMS[:, 8])

#: **FROM THE RS485 REFERENCE, FOR A DIFFERENT ACTUATOR.**  That arm's ``MUSCLE``
#: table (``reference/actuator_net.md`` section 2.5), already x0.8-prescaled by
#: the legacy code that produced it -- never rescale it again.  ``Nf`` is
#: nominally a braid turn count and ``Bf`` a braid thread length, but the
#: reference's own force audit found the fitted law implies a McKibben of 40-52
#: mm rest diameter on an arm whose muscles are 12-25 mm, so both are
#: curve-shape parameters and neither is measured geometry.
MUSCLE_REFERENCE = {
    "Bf": (0.1304, 0.132, 0.1208),          # m, braid thread length
    "Nf": (0.688, 0.808, 0.8),              # dimensionless
    "LBase": (0.1736, 0.1544, 0.1472),      # m
    "LActOffset": (0.0664, 0.0576, 0.052),  # m
}

#: The reference arm's own rest lengths, ``LBase - LActOffset`` =
#: ``(0.1072, 0.0968, 0.0952)`` m.  Kept only as the denominator of
#: :data:`BF_TO_L0_REF`; this arm's rest lengths are 50 to 66 % longer, which is
#: the whole reason its force constants cannot be inherited unscaled.
_L0_REFERENCE_M = tuple(
    b - o for b, o in zip(MUSCLE_REFERENCE["LBase"], MUSCLE_REFERENCE["LActOffset"]))

#: Reference ratio ``Bf / l0`` per segment, ``(1.2164, 1.3636, 1.2689)``.  This
#: ratio is what sets the contraction-to-slack margin, since the anchored law
#: goes slack at ``l = Bf/sqrt(3)``.
BF_TO_L0_REF = tuple(b / l for b, l in zip(MUSCLE_REFERENCE["Bf"], _L0_REFERENCE_M))

#: **DERIVED: this arm's measured geometry carrying the reference's shape
#: ratio.**  ``(0.21644, 0.20038, 0.18073)`` m, and not a measurement of this
#: arm's braid.  Carrying ``Bf`` over verbatim would put this arm's rest length
#: at 2.36x the slack boundary ``Bf/sqrt(3)`` against the reference arm's 1.42x,
#: moving the operating point onto a different part of the ``Bf^2 - 3 l^2``
#: parabola; preserving the ratio instead keeps the seed on the same part of the
#: curve the reference's fit ended on.  A starting point for the outer fit, and
#: it must not be reported as geometry.
BF_SEED_M = tuple(r * l for r, l in zip(BF_TO_L0_REF, L0_SEED_M))

#: **FROM THE RS485 REFERENCE.**  ``1/(4 pi Nf^2)``, the Chou-Hannaford
#: denominator, ``(0.16812, 0.12189, 0.12434)``.  Dimensionless, so unlike
#: :data:`BF_SEED_M` it is not scaled by this arm's lengths.  With the two seeds
#: above it predicts 1674, 621 and 725 N of pull at 30 psi and rest length on
#: segments 1 to 3, against the reference arm's 608, 269 and 324 N at the same
#: pressure -- 2.75, 2.30 and 2.24x, a difference that follows from the longer
#: muscle rather than from any measurement of this one's braid.  It leaves a
#: factor of 2.4 to :data:`FORCE_CLIP_N` on the worst segment, and it makes
#: ``coeff`` the number the outer fit is most likely to move.
COEFF_SEED = tuple(1.0 / (4.0 * np.pi * n * n) for n in MUSCLE_REFERENCE["Nf"])

#: **FROM THE RS485 REFERENCE.**  Pressure slope of the tendon damping,
#: N.s/m per Pa, fitted by that arm's ``fit_bounce.py`` on flight recording
#: ``20260816_222851_real_rec1``: 24 ring episodes, real f = 1.83 Hz,
#: zeta = 0.054 median, twin zeta = 0.063 against a real IQR of 0.047-0.073.
#: A checkpoint that omits ``damp_b1`` inherits this and **not zero**: defaulting
#: it to zero silently revives the ring-killing behaviour that work removed.
DAMP_B1_N_S_M_PER_PA = (9.9e-4, 9.9e-4, 9.9e-4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(z) -> np.ndarray:
    """Logistic, branch-split on the sign so a saturated argument cannot overflow.

    The blend argument is ``e / blend_width``, and a 30 psi error against the
    1e-9 Pa width ``test_gain_blend_reduces_to_hard_switch`` uses is 2e14 --
    ``np.exp`` of that overflows and warns.  Splitting on the sign keeps the
    exponent negative on both branches, so the limit is reached quietly and
    exactly.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def is_tle_from_variants(variants: Iterable[int]) -> np.ndarray:
    """``(24,)`` bool population mask from the variant bytes the boards answered.

    Read the population from ``MSG_FW_VERSION``'s variant byte and never from the
    id range.  A TLE board ships defaulting into ``0x101``-``0x108`` but the
    operator assigns the id, and the bench board spent the whole 2026-08 session
    at ``0x114``; choosing the population from the block would have handed it the
    7 mm pressure calibration -- a 10 % psi error at the top of the range -- and
    the wrong blend width on top of that.

    ``variants`` is indexed by ``node_id - 1``, i.e. entry ``k`` describes board
    ``0x101 + k``.
    """
    v = np.asarray(list(variants), dtype=np.int64)
    if v.shape != (N_ACT,):
        raise ValueError(
            f"variant bytes must be one per board, shape ({N_ACT},); got {v.shape}")
    return v == VARIANT_TLE_DVP


def _as_vec(value, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or check an ``(n,)`` sequence, always returning a copy.

    The copy is what keeps two models built from the same default tuple from
    sharing a mutable array: the trainer clips gains in place and the outer fit
    writes ``coeff``/``bf``/``l0`` in place, so an aliased default would leak one
    fit into every other model in the process.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or shape ({n},); got {arr.shape}")
    return arr.astype(np.float64, copy=True)


# ---------------------------------------------------------------------------
# The flow net
# ---------------------------------------------------------------------------

class FlowNet:
    """5 -> 64 -> 64 -> 1 MLP, tanh hidden, linear output, plain numpy.

    Wider than the reference's 32-wide net because this input row carries a
    population bit: one net now has to represent two valve characteristics at
    once where the reference's represented one.  4609 parameters against the
    reference's 1313, still small enough that a whole rack's forward pass is one
    ``(24,5) @ (5,64)`` GEMM.

    The parameter arrays are live -- a trainer that mutates ``net.W1`` in place
    mutates the model.  That is deliberate, and is how the reference's optimiser
    bound itself to its model, but it means :meth:`ActuatorModel.save` records
    whatever the arrays hold at the moment it is called.
    """

    PARAM_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")

    def __init__(self, W1, b1, W2, b2, W3, b3):
        h1, h2 = HIDDEN
        shapes = {"W1": (N_IN, h1), "b1": (h1,), "W2": (h1, h2),
                  "b2": (h2,), "W3": (h2, 1), "b3": (1,)}
        for key, arr in zip(self.PARAM_KEYS, (W1, b1, W2, b2, W3, b3)):
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != shapes[key]:
                raise ValueError(
                    f"FlowNet.{key} must have shape {shapes[key]}; got {a.shape}")
            setattr(self, key, a.astype(np.float64, copy=True))

    # -- construction -------------------------------------------------------

    @classmethod
    def fresh(cls, *, seed: int = 20260910, w3_sd: float = 0.05) -> "FlowNet":
        """Xavier-ish hidden layers, a deliberately small output layer.

        ``w3_sd`` defaults small so an untrained net produces near-zero flow.
        The alternative -- a full-variance output layer -- gives a fresh model
        that moves every board's pressure on the first quantum of a rollout,
        which is indistinguishable at a glance from a plumbing fault in
        ``sim_core``.
        """
        rng = np.random.default_rng(seed)
        h1, h2 = HIDDEN
        return cls(
            W1=rng.normal(0.0, np.sqrt(2.0 / (N_IN + h1)), size=(N_IN, h1)),
            b1=np.zeros(h1),
            W2=rng.normal(0.0, np.sqrt(2.0 / (h1 + h2)), size=(h1, h2)),
            b2=np.zeros(h2),
            W3=rng.normal(0.0, w3_sd, size=(h2, 1)),
            b3=np.zeros(1),
        )

    def params(self) -> dict:
        """The live arrays, keyed as :data:`PARAM_KEYS`, for a trainer to mutate."""
        return {k: getattr(self, k) for k in self.PARAM_KEYS}

    # -- evaluation ---------------------------------------------------------

    def forward(self, x) -> np.ndarray:
        """``(B, 5)`` normalised features -> ``(B,)`` dimensionless flow."""
        x = np.asarray(x, dtype=np.float64)
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return (h2 @ self.W3)[:, 0] + self.b3[0]

    def forward_full(self, x, dx_dp: Sequence[float] | None = None):
        """``(y, h1, h2, dy_dp)`` -- the hidden activations and the state Jacobian.

        ``dy_dp`` is ``dy/d(p_pa)`` in 1/Pa, by a forward-mode chain along the
        input direction ``dx_dp``.  It defaults to :data:`ActuatorModel.DX_DP`,
        i.e. ``[1/P_SCALE_PA, -1/E_SCALE_PA, 0, 0, 0]``: pressure enters the
        feature row twice, positively as column 0 and negatively inside the
        commanded error of column 1, and the shooting BPTT's state Jacobian
        needs the total derivative.  Taking column 0 alone -- which is what the
        RS485 reference did, correctly, because its row had no error column --
        drops a term of the same magnitude and the opposite sign, and the
        resulting model still trains and still reports a falling loss.

        ``h1``/``h2`` are returned so a hand-written backward pass can reuse the
        forward's activations instead of recomputing them.  Nothing in this
        module consumes them; ``train_actuator_net.py`` uses autograd, and this
        return exists so a numpy trainer remains possible without touching the
        forward.
        """
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(ActuatorModel.DX_DP if dx_dp is None else dx_dp,
                       dtype=np.float64)
        if v.shape != (N_IN,):
            raise ValueError(f"dx_dp must have shape ({N_IN},); got {v.shape}")
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y = (h2 @ self.W3)[:, 0] + self.b3[0]
        t1 = (1.0 - h1 * h1) * (v @ self.W1)          # (B, H1)
        t2 = (1.0 - h2 * h2) * (t1 @ self.W2)         # (B, H2)
        dy_dp = t2 @ self.W3[:, 0]                    # (B,)
        return y, h1, h2, dy_dp


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class ActuatorModel:
    """The 24 boards' plant: flow net, per-node scalars, per-segment force law.

    Indexing throughout: ``node_id`` is 1..24 and names board ``0x100 + node_id``
    and MJCF actuator ``pam_<node_id>``; ``node_idx`` is ``node_id - 1`` and is
    the array index.  The two are never interchangeable, and every method says
    which of them it takes.
    """

    #: ``d(feature)/d(p_pa)``.  See :meth:`FlowNet.forward_full`.
    DX_DP = (1.0 / P_SCALE_PA, -1.0 / E_SCALE_PA, 0.0, 0.0, 0.0)

    #: Keys :meth:`load` refuses to guess at.  Each is a coordinate of the net's
    #: input or output space, so a mismatch on any of them means the weights
    #: describe a different function of the same physical quantities.
    GUARDED_META = ("n_in", "hidden", "p_scale_pa", "e_scale_pa",
                    "dp_scale_pa_s", "l_scale_m", "ldot_scale_m_s")

    def __init__(self, *, net, is_tle, fill_gain, vent_gain, blend_width_pa,
                 leak_pa_s, coeff, bf, l0, damp_b1):
        self.net = net
        mask = np.asarray(is_tle)
        if mask.shape != (N_ACT,):
            raise ValueError(f"is_tle must have shape ({N_ACT},); got {mask.shape}")
        self.is_tle = mask.astype(bool, copy=True)

        self.fill_gain = _as_vec(fill_gain, N_ACT, "fill_gain")
        self.vent_gain = _as_vec(vent_gain, N_ACT, "vent_gain")
        self.leak_pa_s = _as_vec(leak_pa_s, N_ACT, "leak_pa_s")
        self.blend_width_pa = _as_vec(blend_width_pa, 2, "blend_width_pa")
        if np.any(self.blend_width_pa <= 0.0):
            raise ValueError(
                "blend_width_pa must be strictly positive; the hard switch is "
                "the width -> 0 limit of the blend rather than a legal width, "
                "and a zero width divides the commanded error by zero")
        self.coeff = _as_vec(coeff, N_SEG, "coeff")
        self.bf = _as_vec(bf, N_SEG, "bf")
        self.l0 = _as_vec(l0, N_SEG, "l0")
        self.damp_b1 = _as_vec(damp_b1, N_SEG, "damp_b1")

        #: Population index into :attr:`blend_width_pa`.  Held as its own vector
        #: rather than recomputed, because :meth:`gain` runs once per board per
        #: millisecond inside a rollout.
        self.pop_index = np.where(self.is_tle, POP_TLE, POP_7MM).astype(np.int64)
        self.seg_index = SEG_INDEX.copy()
        #: What :meth:`load` read, or ``None`` for a model built in memory.
        self.meta = None

    # -- per-actuator expansions of the per-segment constants ---------------
    #
    # Properties rather than cached arrays, because the outer fit writes
    # ``coeff``/``bf``/``l0`` in place and a cache would go on serving the
    # pre-fit force law for the rest of the rollout.

    @property
    def coeff_per_act(self) -> np.ndarray:
        return self.coeff[self.seg_index]

    @property
    def bf2_per_act(self) -> np.ndarray:
        b = self.bf[self.seg_index]
        return b * b

    @property
    def l0_per_act(self) -> np.ndarray:
        return self.l0[self.seg_index]

    @property
    def damp_b1_per_act(self) -> np.ndarray:
        return self.damp_b1[self.seg_index]

    # -- construction -------------------------------------------------------

    @classmethod
    def fresh(cls, *, is_tle, seed: int = 20260910,
              fill_gain=1.0, vent_gain=1.0, leak_pa_s=0.0,
              blend_width_pa=DEFAULT_BLEND_WIDTH_PA,
              coeff=COEFF_SEED, bf=BF_SEED_M, l0=L0_SEED_M,
              damp_b1=DAMP_B1_N_S_M_PER_PA, w3_sd: float = 0.05) -> "ActuatorModel":
        """An unfitted model: small random net, unit gains, zero leak.

        Every seed is an argument with a named default rather than a constant
        read at call time, so a fit or a test can vary one without editing the
        module.  Their provenance is split and is stated on each constant:
        ``l0`` is **measured on this arm**, ``bf`` is this arm's geometry
        carrying the reference's shape ratio, and ``coeff``, ``damp_b1`` and both
        blend widths come from the RS485 arm or from its firmware and describe a
        different actuator.

        ``leak_pa_s`` defaults to zero rather than to the reference's fitted
        721.6 Pa/s because that number is a property of the RS485 arm's fittings,
        and on this arm the leak that is known is one board's: ``0x110`` leaks
        from its supply side and drifted 0.1 to 7.0 psi across the 2026-08-21
        sweep.  A uniform seed would spread that one fault over twenty-four
        boards.  Run :func:`fit_leak_segments` on guaranteed-closed holds before
        a rollout is expected to hold pressure.
        """
        return cls(
            net=FlowNet.fresh(seed=seed, w3_sd=w3_sd),
            is_tle=is_tle,
            fill_gain=fill_gain, vent_gain=vent_gain, leak_pa_s=leak_pa_s,
            blend_width_pa=blend_width_pa,
            coeff=coeff, bf=bf, l0=l0, damp_b1=damp_b1,
        )

    def clip_parameters(self, *, gain_clip=GAIN_CLIP,
                        leak_clip=LEAK_CLIP_PA_S) -> None:
        """The trainer's post-step hook, in place.

        Both clips exist because the quantity they bound is degenerate or
        sign-critical rather than merely large: the absolute gain scale trades
        off against the shared net's output scale (net x c, gains / c) so nothing
        in the shooting loss pins it, and the plant seam *subtracts* the leak so
        a negative one inflates an idle board.
        """
        np.clip(self.fill_gain, gain_clip[0], gain_clip[1], out=self.fill_gain)
        np.clip(self.vent_gain, gain_clip[0], gain_clip[1], out=self.vent_gain)
        np.clip(self.leak_pa_s, leak_clip[0], leak_clip[1], out=self.leak_pa_s)

    # -- features and gains -------------------------------------------------

    def _norm_x(self, p_pa, target_pa, is_tle, l_m, ldot_m_s) -> np.ndarray:
        """The five-column feature row of ``CONTRACT.md`` section 2, in order."""
        cols = np.broadcast_arrays(
            np.atleast_1d(np.asarray(p_pa, dtype=np.float64)),
            np.atleast_1d(np.asarray(target_pa, dtype=np.float64)),
            np.atleast_1d(np.asarray(is_tle, dtype=np.float64)),
            np.atleast_1d(np.asarray(l_m, dtype=np.float64)),
            np.atleast_1d(np.asarray(ldot_m_s, dtype=np.float64)),
        )
        p, tgt, tle, l, ld = cols
        return np.column_stack((
            p / P_SCALE_PA,
            (tgt - p) / E_SCALE_PA,
            tle,
            l / L_SCALE_M,
            ld / LDOT_SCALE_M_S,
        ))

    def gain(self, node_idx, e_pa) -> np.ndarray:
        """Logistic blend of the fill and vent gains on the commanded error.

        ``node_idx`` is 0-based.  ``s`` is 1 when the board is being asked to
        fill and 0 when it is being asked to vent::

            s = 1 / (1 + exp(-e_pa / blend_width_pa[pop]))
            g = s * fill_gain[node_idx] + (1 - s) * vent_gain[node_idx]

        At ``e = 0`` this returns the midpoint of the two gains rather than the
        reference's fixed closed-state gain of 1.0.  That is the price of
        differentiability: with no observable valve state there is no "closed" to
        assign a gain to, so what has to go to zero on a satisfied setpoint is
        the net's own output and not the gain.
        """
        idx = np.asarray(node_idx, dtype=np.int64)
        e = np.asarray(e_pa, dtype=np.float64)
        width = self.blend_width_pa[self.pop_index[idx]]
        s = _sigmoid(e / width)
        return np.asarray(s * self.fill_gain[idx] + (1.0 - s) * self.vent_gain[idx])

    # -- flow ---------------------------------------------------------------

    def net_flow_pa_s(self, node_id: int, p_pa: float, target_pa: float,
                      l_m: float, ldot_m_s: float) -> float:
        """``dp/dt`` for one board, Pa/s, **before** the leak is subtracted."""
        if not 1 <= int(node_id) <= N_ACT:
            raise ValueError(f"node_id must be 1..{N_ACT}; got {node_id}")
        idx = int(node_id) - 1
        x = self._norm_x(p_pa, target_pa, float(self.is_tle[idx]), l_m, ldot_m_s)
        y = self.net.forward(x)[0]
        g = float(self.gain(idx, float(target_pa) - float(p_pa)))
        return float(g * y * DP_SCALE_PA_S)

    def net_flow_pa_s_batch(self, p_pa, target_pa, l_m, ldot_m_s) -> np.ndarray:
        """The whole rack in one GEMM, ``(24,)`` Pa/s, before the leak.

        This is the path ``sim_core`` steps at 1 ms; the scalar
        :meth:`net_flow_pa_s` exists for a single-board probe and for tests.  The
        two agree to a few ulp and **not** bitwise, because BLAS picks a
        different summation order for a ``(24,5)`` product than for a ``(1,5)``
        one.  Write acceptance tests with ``rtol``, never with ``==``.
        """
        x = self._norm_x(p_pa, target_pa, self.is_tle.astype(np.float64),
                         l_m, ldot_m_s)
        if x.shape[0] != N_ACT:
            raise ValueError(
                f"batch flow wants one entry per board, {N_ACT}; got {x.shape[0]}")
        y = self.net.forward(x)
        e = np.asarray(target_pa, dtype=np.float64) \
            - np.asarray(p_pa, dtype=np.float64)
        g = self.gain(np.arange(N_ACT), np.broadcast_to(e, (N_ACT,)))
        return g * y * DP_SCALE_PA_S

    def flow_pa_s(self, node_id: int, p_pa: float, target_pa: float,
                  l_m: float, ldot_m_s: float) -> float:
        """``dp/dt`` for one board with its leak subtracted, Pa/s.

        Keeping the leak outside the net is what lets a fault override it:
        ``0x110`` leaks from its supply side and reads +5.1 psi at rest, which is
        a per-board number a net shared by twenty-four boards cannot carry.  The
        subtraction happens either here or at the plant seam in
        ``sim_core.SimNode._integrate`` -- a caller uses one, never both.
        """
        idx = int(node_id) - 1
        return self.net_flow_pa_s(node_id, p_pa, target_pa, l_m, ldot_m_s) \
            - float(self.leak_pa_s[idx])

    def flow_pa_s_batch(self, p_pa, target_pa, l_m, ldot_m_s) -> np.ndarray:
        """``(24,)`` Pa/s with each board's own leak subtracted."""
        return self.net_flow_pa_s_batch(p_pa, target_pa, l_m, ldot_m_s) \
            - self.leak_pa_s

    # -- force and damping --------------------------------------------------

    def force_n(self, p_pa, dlen_m) -> np.ndarray:
        """Anchored McKibben pull, ``(24,)`` N, ``<= 0``.

        ``dlen_m`` is the tendon excursion from the MJCF's rest configuration,
        ``ten_length - ten_length0``; the anchored length is ``l = l0_seg +
        dlen``, and it is the *same* number the net takes as its ``l`` feature.
        That sharing is the only coupling between the net and the force law, and
        it has a consequence worth stating: an outer fit that moves ``l0`` moves
        the net's input distribution too, so tick traces must be rebuilt before
        the net is re-evaluated against them.

        Negative is a pull.  The clip is two-sided for two different reasons: the
        bound at 0 is physics -- a braided bladder cannot push, and a muscle
        contracted past ``l = bf/sqrt(3)`` has gone slack -- while the bound at
        :data:`FORCE_CLIP_N` is a guard rail, and a rollout that reaches it is
        reporting the guard rail and not the model.
        """
        l = self.l0_per_act + np.asarray(dlen_m, dtype=np.float64)
        f = self.coeff_per_act * np.asarray(p_pa, dtype=np.float64) \
            * (self.bf2_per_act - 3.0 * l * l)
        return np.clip(f, -FORCE_CLIP_N, 0.0)

    def tendon_damping_n_s_m(self, p_pa, base, l_m) -> np.ndarray:
        """Pressure-scheduled tendon damping, ``(24,)`` N.s/m, ``>= 0``.

        ``base + damp_b1_seg * p``, with the pressure term zeroed wherever the
        muscle is slack by the same ``3 l^2 <= bf^2`` boundary the force law
        uses.  The gate is not cosmetic: MuJoCo tendon damping is bilateral, so a
        slack muscle carrying a pressure-scheduled damping term resists being
        lengthened -- it pushes -- which is exactly what the pull-only clip
        exists to forbid.

        ``l_m`` is the anchored length and not the excursion, so a caller holding
        ``dlen`` adds :attr:`l0_per_act` first.  ``base`` is the MJCF's own tendon
        damping and is passed in rather than read from a constant because
        ``mjcf_generator`` takes it as a tunable and the outer fit moves it.
        """
        l = np.asarray(l_m, dtype=np.float64)
        p = np.asarray(p_pa, dtype=np.float64)
        slack = 3.0 * l * l <= self.bf2_per_act
        d = np.asarray(base, dtype=np.float64) \
            + np.where(slack, 0.0, self.damp_b1_per_act * p)
        return np.maximum(d, 0.0)

    # -- persistence --------------------------------------------------------

    @staticmethod
    def _reference_meta() -> dict:
        """The normalisation this module evaluates, as :meth:`load` expects it."""
        return {"n_in": N_IN, "hidden": list(HIDDEN), "p_scale_pa": P_SCALE_PA,
                "e_scale_pa": E_SCALE_PA, "dp_scale_pa_s": DP_SCALE_PA_S,
                "l_scale_m": L_SCALE_M, "ldot_scale_m_s": LDOT_SCALE_M_S}

    def _meta(self, extra=None) -> dict:
        """Everything :meth:`load` checks, plus whatever the caller adds.

        The scale constants are written into the checkpoint rather than assumed
        on load because a checkpoint from a different normalisation reproduces
        its own training set and diverges everywhere else, which is the failure
        mode that looks most like success.
        """
        meta = dict(self._reference_meta())
        # Recorded but not guarded: changing either is a change of guard rail or
        # of a human-boundary conversion rather than of the net's coordinates.
        meta["force_clip_n"] = FORCE_CLIP_N
        meta["pa_per_psi"] = PA_PER_PSI
        if extra:
            meta.update(extra)
        return meta

    def save(self, path, meta=None) -> None:
        """npz plus a JSON ``meta`` string carrying every scale constant.

        ``np.savez`` appends ``.npz`` to a path that lacks it, which makes a
        save/load round trip on a bare name fail with a confusing
        ``FileNotFoundError``; the extension is normalised here so the path the
        caller passed is the path that exists afterwards.
        """
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(
            path,
            meta=json.dumps(self._meta(meta)),
            is_tle=self.is_tle,
            fill_gain=self.fill_gain, vent_gain=self.vent_gain,
            blend_width_pa=self.blend_width_pa, leak_pa_s=self.leak_pa_s,
            coeff=self.coeff, bf=self.bf, l0=self.l0, damp_b1=self.damp_b1,
            **self.net.params(),
        )

    @classmethod
    def load(cls, path) -> "ActuatorModel":
        """Load a checkpoint, **raising** on any normalisation mismatch.

        ``allow_pickle=False``: a checkpoint is data and must not be able to
        execute anything as it is read.

        The unit guard raises :class:`ValueError` naming the offending key rather
        than warning or coercing, because there is no safe fallback.  A net
        trained against the reference's 25 psi pressure scale and stepped against
        this arm's 30 psi one is wrong by 20 % on its dominant input at every
        quantum, reports a plausible residual on the trajectories it saw, and
        diverges on the rest.

        A checkpoint that omits ``damp_b1`` inherits
        :data:`DAMP_B1_N_S_M_PER_PA` and not zero, which is the reference's
        measured lesson: the RS485 arm's shipped checkpoint pre-dates its damping
        work, and a port that defaults the missing array to zero silently
        restores ring-killing behaviour.
        """
        path = str(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            want = cls._reference_meta()
            for key in cls.GUARDED_META:
                if key not in meta:
                    raise ValueError(
                        f"{path}: checkpoint meta has no '{key}'.  It was written "
                        "by a version of digital_twin.actuator_model that did not "
                        "record its normalisation, so its weights cannot be shown "
                        "to describe the same function this module evaluates.")
                got, exp = meta[key], want[key]
                if key == "hidden":
                    got, exp = tuple(got), tuple(exp)
                if got != exp:
                    raise ValueError(
                        f"{path}: checkpoint meta['{key}'] is {got!r}, this module "
                        f"uses {exp!r}.  Re-train against the current "
                        "normalisation; a checkpoint from a different one "
                        "reproduces its training set and diverges everywhere else.")
            damp = z["damp_b1"] if "damp_b1" in z.files \
                else np.asarray(DAMP_B1_N_S_M_PER_PA)
            model = cls(
                net=FlowNet(**{k: z[k] for k in FlowNet.PARAM_KEYS}),
                is_tle=z["is_tle"],
                fill_gain=z["fill_gain"], vent_gain=z["vent_gain"],
                blend_width_pa=z["blend_width_pa"], leak_pa_s=z["leak_pa_s"],
                coeff=z["coeff"], bf=z["bf"], l0=z["l0"], damp_b1=damp,
            )
            model.meta = meta
        return model


# ---------------------------------------------------------------------------
# Leak fitting
# ---------------------------------------------------------------------------
#
# The leak is fitted here and nowhere else, by least squares on segments that
# are *guaranteed* closed -- constant target, pressure well clear of the noise
# floor, no gap in the record -- rather than learned by the net.  Choosing those
# segments is the caller's job, because it needs the recording schema that
# ``dataset.py`` owns; this module turns a time/pressure pair into a slope and
# says when it refuses to.

def fit_leak(t_s, p_pa, *, min_duration_s: float = 2.0) -> float:
    """Least-squares decay rate of one closed segment, Pa/s, positive = losing.

    The sign is flipped from the fitted slope so the returned number is the one
    the plant seam subtracts.  ``min_duration_s`` defaults to 2.0 s because the
    quantity being estimated is small: a leak of a few hundred Pa/s against a
    consecutive-sample noise scale of order 500 Pa needs seconds of lever arm
    before the slope separates from the noise, and a shorter segment returns a
    number whose sign is close to random.

    Raises :class:`ValueError` naming the shortfall rather than returning a weak
    estimate, since a weak estimate is indistinguishable downstream from a good
    one.
    """
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(p_pa, dtype=np.float64)
    if t.ndim != 1 or t.shape != p.shape:
        raise ValueError(
            f"fit_leak wants two matching 1-D arrays; got {t.shape} and {p.shape}")
    if t.size < 3:
        raise ValueError(
            f"leak segment too short: {t.size} samples, need at least 3")
    span = float(t[-1] - t[0])
    if span < min_duration_s:
        raise ValueError(
            f"leak segment too short: {span:.3f} s, need {min_duration_s:.3f} s")
    return -float(np.polyfit(t, p, 1)[0])


def fit_leak_segments(segments, *, min_duration_s: float = 2.0,
                      leak_clip_pa_s=LEAK_CLIP_PA_S,
                      require_segments: int = 1) -> float:
    """Duration-weighted mean leak over several closed segments of one board.

    ``segments`` is an iterable of ``(t_s, p_pa)`` pairs.  Weighting by duration
    rather than by sample count is what makes the estimate insensitive to a
    dropped reply: the bus answers every sync edge at 150 Hz, but a segment with
    a gap in it still spans the same wall time and should still count for it.

    Segments shorter than ``min_duration_s`` are skipped rather than raising, so
    a caller can hand over every hold it found and let the duration rule decide;
    :func:`fit_leak` raises on the same condition because there the caller has
    already asserted the segment is usable.  ``require_segments`` is the floor
    below which the result is refused outright -- one usable segment by default,
    so a board with no usable hold keeps whatever leak it already had instead of
    silently taking a zero.

    The result is clipped to ``leak_clip_pa_s``.  The lower bound at 0 matters:
    the plant seam subtracts this number, so a negative estimate -- sensor noise
    read as self-inflation -- would ship as an idle board that inflates on its
    own.
    """
    total_w = 0.0
    total_wx = 0.0
    n_used = 0
    for t_s, p_pa in segments:
        t = np.asarray(t_s, dtype=np.float64)
        if t.size < 3 or float(t[-1] - t[0]) < min_duration_s:
            continue
        w = float(t[-1] - t[0])
        total_wx += w * fit_leak(t, p_pa, min_duration_s=min_duration_s)
        total_w += w
        n_used += 1
    if n_used < require_segments:
        raise ValueError(
            f"leak fit needs {require_segments} segment(s) of at least "
            f"{min_duration_s:.3f} s; {n_used} qualified")
    return float(np.clip(total_wx / total_w, leak_clip_pa_s[0], leak_clip_pa_s[1]))


__all__ = [
    "PA_PER_PSI", "N_ACT", "N_SEG", "N_IN", "HIDDEN",
    "P_SCALE_PA", "E_SCALE_PA", "DP_SCALE_PA_S", "L_SCALE_M", "LDOT_SCALE_M_S",
    "FORCE_CLIP_N", "GAIN_CLIP", "LEAK_CLIP_PA_S", "DEFAULT_BLEND_WIDTH_PA",
    "SEG_INDEX", "POP_7MM", "POP_TLE", "VARIANT_TLE_DVP",
    "MUSCLE_REFERENCE", "BF_TO_L0_REF", "L0_SEED_M", "BF_SEED_M", "COEFF_SEED",
    "DAMP_B1_N_S_M_PER_PA",
    "FlowNet", "ActuatorModel",
    "is_tle_from_variants", "fit_leak", "fit_leak_segments",
]
