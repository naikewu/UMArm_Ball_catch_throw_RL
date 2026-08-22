"""The CAN UMArm's geometric parameter table — **measured 2026-08-21**.

The CAN arm has the same topology as the RS485 mk5: three segments, two
universal joints per segment, two degrees of freedom per joint, twelve in all,
with the plate centre coincident with the joint centre (``UC1 = UC2 = 0``).
Nothing in :mod:`UMArm_KINEMATICS.fkine` hard-codes a length — every entry
point takes ``params=None`` and routes through
:func:`robot_params.as_params` — so porting the kinematics to a bigger arm of
the same topology is a new ``(3, 10)`` table and nothing else.  This is that
table, and as of 2026-08-21 it is a measurement rather than the placeholder it
used to be.

WHERE THE NUMBERS COME FROM.  Two sources, and the split is deliberate.

* ``UC``, ``AA``, ``JA`` and ``AO`` are transcribed from the research tree's
  ``robot_constants.py`` (``param_link0/1/2``,
  ``UMARM_Variable_Stiffness_Oct2025``), which describes **this** arm rather
  than the RS485 one.  They are CAD dimensions of parts a mocap campaign cannot
  see: ``UC`` is zero by construction, ``AA`` is where the actuator attaches,
  and ``JA``/``AO`` are consumed by nothing in the forward kinematics and ride
  along only so a row can be handed to the legacy oracle intact.
* ``LL`` and ``JD`` are **measured**: the five consecutive u-joint centre
  distances read straight off the marker-inferred plate frames over 68 arm
  poses (``hw_tests/canarm_axis_analysis.py`` on
  ``hw_tests/results/drive_2026-08-21.json``), then refined against the centres
  fkine predicts.  These five distances are rigid — plate centre *is* joint
  centre here — so they do not depend on the arm's pose, and they did not: the
  standard deviation across all 68 poses was 0.03 to 0.47 mm.

The measured chain sits within **0.04 to 1.8 mm** of what the research tree's
table implies, which is the substantive finding of the exercise: the legacy CAN
table was right, and the placeholder this file used to carry — the RS485 arm's
table wearing the CAN arm's name — was wrong by 46 to 182 mm at the u-joint
centres.

WHAT THIS TABLE IS NOT.  It is not a claim that the *model* is right, only that
its lengths are.  Two model corrections were needed alongside it and live
elsewhere, because they are properties of the mechanism rather than of its
dimensions: the per-plate marker azimuth
(``UMArm_MOCAP.canarm_frames.PLATE_AZIMUTH_DEG``) and the proximal-pair
composition order (``fkine``'s ``order="yx"``,
``UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER``).  With all three, fkine
reproduces the measured u-joint centres to 0.74 mm RMS on the poses it was
fitted to and **1.99 mm RMS (7.0 mm worst) on eighteen held-out multi-joint
poses** spanning up to 30 deg of joint travel.
"""

from __future__ import annotations

import numpy as np

try:  # both import styles must work, as elsewhere in this workspace
    from . import robot_params as rp
except ImportError:  # pragma: no cover
    import robot_params as rp  # type: ignore[no-redef]

#: ``True``: every length below came from this arm.  Read it before trusting a
#: number out of this module.
MEASURED = True

#: What measured them.
MEASURED_SOURCE = (
    "hw_tests/results/drive_2026-08-21.json (68 poses, marker-inferred plate "
    "frames) reduced by hw_tests/canarm_axis_analysis.py; CAD columns from "
    "UMARM_Variable_Stiffness_Oct2025/robot_constants.py param_link0/1/2")

#: Columns carrying a length.  Kept because callers that scale or perturb a
#: table need to know which columns are metres and which are not.
LENGTH_COLUMNS = (rp.COL_UC1, rp.COL_UC2, rp.COL_AA1, rp.COL_AA2,
                  rp.COL_LL, rp.COL_JD)

#: The five consecutive u-joint centre distances measured on the arm,
#: ``(span1, JD2, span2, JD3, span3)``, metres.  This is the tuple the table
#: below is built to reproduce, and the one a live session re-measures first.
#: Measured means over 68 poses were
#: ``(0.265357, 0.072886, 0.234371, 0.072987, 0.229918)`` with standard
#: deviations ``(0.47, 0.03, 0.10, 0.05, 0.12)`` mm; span3 is the one the
#: position refinement moved, by 0.06 mm.
CANARM_PLATE_CHAIN_M = (0.265357, 0.072886, 0.234371, 0.072987, 0.229858)

#: ``(3, 10)`` — one row per segment, proximal to distal, columns as
#: ``robot_params.PARAM_COLUMNS``.  Read-only for the same reason
#: ``DEFAULT_PARAMS`` is: it is a default argument all over :mod:`fkine`, and a
#: fit that edited it in place would silently re-zero every other caller.
#:
#: ``LL`` is ``span - (AA1 + AA2)`` for each segment, which is how the measured
#: chain enters the table: ``AA`` is the CAD number and ``LL`` absorbs the
#: difference, the same convention ``fkine_benchmark`` uses on the RS485 arm
#: (CAD is surest about the attachment points and least sure about the actuated
#: length).
CANARM_PARAMS = np.array([
    # JA1    JA2    UC1  UC2  AA1        AA2        AO1    AO2    LL         JD
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.177932, 0.0],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.146946, 0.072886],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.142433, 0.072987],
], dtype=float)
CANARM_PARAMS.flags.writeable = False

#: The research tree's table for the same arm, kept for the diff.  Its chain is
#: ``(265.05, 73.26, 234.14, 72.49, 231.66)`` mm against the measured
#: ``(265.36, 72.89, 234.37, 72.99, 229.86)``.
LEGACY_CANARM_PARAMS = np.array([
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.177625, 0.0],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.146715, 0.07326],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.144235, 0.07249],
], dtype=float)
LEGACY_CANARM_PARAMS.flags.writeable = False


def require_measured() -> np.ndarray:
    """:data:`CANARM_PARAMS`, or a refusal naming what has not been done.

    For callers whose output is a length rather than a shape — a fitted
    stiffness, a workspace volume, a reach claim.  Kept now that
    :data:`MEASURED` is ``True`` because the next arm, or the next
    re-plumbing of this one, will set it back to ``False``, and the call sites
    that need the distinction should already be routed through here.
    """
    if not MEASURED:
        raise RuntimeError(
            "UMArm_KINEMATICS.canarm_params carries lengths that have not been "
            "measured on this arm.  Run hw_tests/canarm_drive_campaign.py, "
            "reduce it with hw_tests/canarm_axis_analysis.py, write the chain "
            "here and set MEASURED = True.")
    return CANARM_PARAMS


__all__ = [
    "MEASURED",
    "MEASURED_SOURCE",
    "LENGTH_COLUMNS",
    "CANARM_PARAMS",
    "CANARM_PLATE_CHAIN_M",
    "LEGACY_CANARM_PARAMS",
    "require_measured",
]
