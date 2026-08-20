"""The CAN UMArm's geometric parameter table — **placeholder lengths**.

The CAN arm has the same topology as the RS485 mk5: three segments, two
universal joints per segment, two degrees of freedom per joint, twelve in all,
with the plate centre coincident with the joint centre (``UC1 = UC2 = 0``).
Nothing in :mod:`UMArm_KINEMATICS.fkine` hard-codes a length — every entry
point takes ``params=None`` and routes through
:func:`robot_params.as_params` — so porting the kinematics to a bigger arm of
the same topology is a new ``(3, 10)`` table and nothing else.  This is that
table.

WHAT IS REAL HERE AND WHAT IS NOT.  The **topology** is real: it is the same
mechanism, and the 24 boards at 8 per segment match the RS485 arm's actuator
ring exactly.  The **lengths are not**.  Every metre in :data:`CANARM_PARAMS`
is the RS485 arm's fitted value multiplied by :data:`PLACEHOLDER_SCALE`, and
:data:`PLACEHOLDER_SCALE` is 1.0 because no one has measured this arm.  A
scale factor invented to look plausible would be worse than the identity: the
identity is obviously the RS485 arm's number wearing the CAN arm's name, while
1.3 would read as a measurement.

THE PROVENANCE DISCIPLINE, which is why :data:`MEASURED` exists.  The RS485
table's own numbers came from ``robot_constants.py``'s three ``param_builder``
calls and were then *re-fitted* against mocap by ``fkine_benchmark.py``, whose
one knob is ``LL`` per segment (``p[i, COL_LL] = gap - (AA1 + AA2)``: UC and AA
stay at their hardware values because CAD is surest about them and least sure
about the actuated length).  The CAN arm owes the same treatment.  Until a live
mocap session measures the plate-to-plate distances and writes them here,
anything derived from this table is a shape, not a length, and
:func:`require_measured` refuses on behalf of any caller that needs the
difference to matter.

TO UPDATE THIS FILE.  Run the mocap probe against the CAN arm
(``mocap_probe.py --rb-id-base 2000 --n-bodies 6``), take the five consecutive
u-joint-centre gaps it reports, and set ``LL`` per segment so that
:func:`robot_params.plate_chain_m` of this table reproduces them.  Then set
:data:`MEASURED` to ``True``, replace :data:`PLACEHOLDER_SCALE` with the
campaign name the numbers came from, and cite it the way ``robot_params``
cites its own sources.
"""

from __future__ import annotations

import numpy as np

try:  # both import styles must work, as elsewhere in this workspace
    from . import robot_params as rp
except ImportError:  # pragma: no cover
    import robot_params as rp  # type: ignore[no-redef]

#: ``True`` only when every length below came from a measurement of the CAN
#: arm.  Read it before trusting a number out of this module; the alternative
#: is a fitted result that is confidently wrong by whatever this arm's real
#: proportions turn out to be.
MEASURED = False

#: Uniform multiplier applied to the RS485 arm's length columns.  **1.0, i.e.
#: no scaling**, because nothing has measured the CAN arm.  See the module
#: docstring for why the identity is preferred to a plausible guess.  TODO:
#: replace with per-segment measured lengths, not a scale factor — a bigger arm
#: is unlikely to be uniformly bigger.
PLACEHOLDER_SCALE = 1.0

#: Columns carrying a length, and therefore the ones a scale factor touches.
#: ``JA*`` (joint travel allowances) and ``AO*`` (actuator offsets) ride along
#: unchanged: the forward kinematics consumes neither, and they exist only so a
#: row can be handed to the legacy oracle intact.
LENGTH_COLUMNS = (rp.COL_UC1, rp.COL_UC2, rp.COL_AA1, rp.COL_AA2,
                  rp.COL_LL, rp.COL_JD)


def _scaled_placeholder(scale: float) -> np.ndarray:
    """The RS485 table with its length columns multiplied by ``scale``."""
    out = np.array(rp.DEFAULT_PARAMS, dtype=float)   # a writable copy
    for col in LENGTH_COLUMNS:
        out[:, col] *= float(scale)
    out.flags.writeable = False
    return out


#: ``(3, 10)`` — one row per segment, proximal to distal, columns as
#: ``robot_params.PARAM_COLUMNS``.  **PLACEHOLDER**: see :data:`MEASURED`.
#: Read-only for the same reason ``DEFAULT_PARAMS`` is — it is a default
#: argument all over :mod:`fkine`, and a fit that edited it in place would
#: silently re-zero every other caller.
CANARM_PARAMS = _scaled_placeholder(PLACEHOLDER_SCALE)

#: The five consecutive u-joint-centre gaps :data:`CANARM_PARAMS` implies,
#: ``(span1, JD2, span2, JD3, span3)``, metres.  **PLACEHOLDER.**  This is the
#: tuple a live session will contradict first, because it is exactly what the
#: mocap chain-span gate measures directly.
CANARM_PLATE_CHAIN_M = rp.plate_chain_m(CANARM_PARAMS)


def require_measured() -> np.ndarray:
    """:data:`CANARM_PARAMS`, or a refusal naming what has not been done.

    For callers whose output is a length rather than a shape — a fitted
    stiffness, a workspace volume, a reach claim.  Callers that only need the
    topology (structure tests, a visualiser's proportions, an offline plumbing
    check) should use :data:`CANARM_PARAMS` directly and say so.
    """
    if not MEASURED:
        raise RuntimeError(
            "UMArm_KINEMATICS.canarm_params carries PLACEHOLDER lengths: the "
            "RS485 arm's table scaled by %g, with no measurement of the CAN "
            "arm behind any of it.  Measure the plate-to-plate distances in a "
            "live mocap session, write them here, and set MEASURED = True."
            % PLACEHOLDER_SCALE)
    return CANARM_PARAMS


__all__ = [
    "MEASURED",
    "PLACEHOLDER_SCALE",
    "LENGTH_COLUMNS",
    "CANARM_PARAMS",
    "CANARM_PLATE_CHAIN_M",
    "require_measured",
]
