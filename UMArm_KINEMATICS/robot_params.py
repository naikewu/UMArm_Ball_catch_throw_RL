"""The UMArm mk5 geometric parameter table, and the chain nominals it implies.

Every number here is *lifted*, not invented: the three ``param_builder`` calls
in the legacy tree (``UMArm_compliance_TRO/robot_constants.py:21-46``) are
transcribed as literals, one row per segment, in the legacy column order.  This
module is the **authoritative home** of ``PLATE_CHAIN_NOMINAL_M`` (design
``docs/fkine_design.md`` D5/D8): ``UMArm_ROBOT_CONTROL.arm_constants`` keeps its
own literal copy for its plate-chain validity gate, and a cross-package equality
test in ``test_fkine.py`` guards both — deliberately **no runtime import in
either direction**, so the mocap probe can depend on this package without
dragging in the bus stack, and the bus stack keeps working on a machine that
has never heard of kinematics.

Column layout (legacy ``param_builder`` argument order, consumed positionally
by ``exponentials_mk5.get_e1234`` and ``kinematics_mp``)::

    [JA1, JA2, UC1, UC2, AA1, AA2, AO1, AO2, LL, JD]

Of these, the forward kinematics consumes **UC/AA/LL/JD only** (verified by
recon over ``kinematics_mp.py:241-297``): ``JD`` is the rigid spacer between the
previous segment's distal plate and this segment's proximal plate, ``UC1``/
``UC2`` place the u-joint centres relative to the plates (**zero on this
hardware** — plate centre *is* joint centre, ``robot_constants.py:23,32,41``),
``AA1 + LL + AA2`` is the actuated span between the two centres.  ``JA*`` (joint
travel allowances) and ``AO*`` (actuator offsets) ride along so a params row
can be handed unchanged to the legacy oracle, which unpacks all ten.

Derived facts, pinned by tests:

* segment spans ``UC1+AA1+LL+AA2+UC2`` = 0.218868 / 0.194540 / 0.184111 m —
  **bit-exact** to the ``PLATE_CHAIN_NOMINAL_M`` literals (the sums land on the
  same doubles);
* ``fkine(0)`` translation = (0, 0, -0.692772) = -(sum of the 3 spans + 2 JD
  gaps), matching the u-joint centre heights in ``UMArm_structure.html`` §2.

Pure numpy, **no intra-repo imports** (design D5).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Column indices  (legacy: robot_constants.param_builder argument order)
# --------------------------------------------------------------------------

#: Names of the ten columns, in table order.  Kept as data so tests and debug
#: printers can label a row without hard-coding positions twice.
PARAM_COLUMNS = ("JA1", "JA2", "UC1", "UC2", "AA1", "AA2", "AO1", "AO2", "LL", "JD")

COL_JA1 = 0
COL_JA2 = 1
COL_UC1 = 2
COL_UC2 = 3
COL_AA1 = 4
COL_AA2 = 5
COL_AO1 = 6
COL_AO2 = 7
COL_LL = 8
COL_JD = 9

# --------------------------------------------------------------------------
# The table  (legacy: robot_constants.py:21-46, param_link0/1/2)
# --------------------------------------------------------------------------

#: ``(3, 10)`` — one row per segment, proximal to distal, columns as
#: :data:`PARAM_COLUMNS`.  Metres and radians throughout.  Marked read-only
#: because it is used as a default argument all over :mod:`fkine`; anyone
#: fitting parameters must pass their own copy, never edit this one.
DEFAULT_PARAMS = np.array([
    # JA1    JA2   UC1  UC2  AA1     AA2     AO1   AO2   LL        JD
    [0.10, 0.05, 0.0, 0.0, 0.0285, 0.0285, 0.03, 0.03, 0.161868, 0.0],       # segment 1
    [0.05, 0.05, 0.0, 0.0, 0.0285, 0.0285, 0.03, 0.03, 0.137540, 0.047871],  # segment 2
    [0.05, 0.05, 0.0, 0.0, 0.0285, 0.0285, 0.03, 0.03, 0.127111, 0.047382],  # segment 3
], dtype=float)
DEFAULT_PARAMS.flags.writeable = False


def as_params(params) -> np.ndarray:
    """Validate and return ``params`` as a float ``(3, 10)`` array.

    ``None`` means :data:`DEFAULT_PARAMS`.  Anything mis-shaped raises
    ``ValueError`` — the legacy habit of printing and returning ``None`` is not
    ported (design §2), because a silent ``None`` from deep inside a control
    loop is exactly how bad frames used to propagate.
    """
    if params is None:
        return DEFAULT_PARAMS
    out = np.asarray(params, dtype=float)
    if out.shape != DEFAULT_PARAMS.shape:
        raise ValueError(
            f"params must have shape {DEFAULT_PARAMS.shape} "
            f"(rows = segments, columns = {PARAM_COLUMNS}); got {out.shape}")
    return out


# --------------------------------------------------------------------------
# Named accessors
# --------------------------------------------------------------------------


def segment_spans(params=None) -> np.ndarray:
    """``(3,)`` centre-to-centre span of each segment: ``UC1+AA1+LL+AA2+UC2``.

    This is the distance from a segment's proximal u-joint centre to its distal
    one, and (because ``UC1 = UC2 = 0`` on this hardware) also the plate-to-
    plate distance the mocap chain gate measures.  The additions are written
    out left-to-right, matching the legacy ``link_length``
    (``kinematics_mp.py:389-391``) term order, so the sums are bit-identical to
    the legacy values *and* to the :data:`PLATE_CHAIN_NOMINAL_M` literals.
    """
    p = as_params(params)
    return (((p[:, COL_UC1] + p[:, COL_AA1]) + p[:, COL_LL])
            + p[:, COL_AA2]) + p[:, COL_UC2]


def segment_jds(params=None) -> np.ndarray:
    """``(3,)`` rigid spacer ``JD`` ahead of each segment (JD1 = 0: no spacer
    between the base plate and segment 1's proximal centre)."""
    return as_params(params)[:, COL_JD].copy()


def plate_chain_m(params=None) -> tuple[float, float, float, float, float]:
    """The five consecutive u-joint-centre gaps a params table implies.

    Order: ``(span1, JD2, span2, JD3, span3)`` — exactly the layout of
    ``PLATE_CHAIN_NOMINAL_M`` (proximal to distal: plates 0-1, 1-2, 2-3, 3-4,
    4-5).  JD1 is *not* in the chain: it sits between the world origin and
    plate 0, and it is zero on this arm anyway.
    """
    spans = segment_spans(params)
    jds = segment_jds(params)
    return (float(spans[0]), float(jds[1]), float(spans[1]),
            float(jds[2]), float(spans[2]))


# --------------------------------------------------------------------------
# The chain nominals  (authoritative home — design D8)
# --------------------------------------------------------------------------

#: Nominal consecutive u-joint centre distances, metres, proximal to distal:
#: ``(span1, JD2, span2, JD3, span3)``.  Same five literals as
#: ``UMArm_ROBOT_CONTROL.arm_constants.PLATE_CHAIN_NOMINAL_M``
#: (``arm_constants.py:540``), which keeps its copy so the bus stack stays
#: import-free of this package; ``test_fkine.py`` asserts the two tuples are
#: equal element for element, and that :func:`plate_chain_m` of the default
#: table reproduces them **bit-exactly**.
PLATE_CHAIN_NOMINAL_M = (0.218868, 0.047871, 0.194540, 0.047382, 0.184111)
