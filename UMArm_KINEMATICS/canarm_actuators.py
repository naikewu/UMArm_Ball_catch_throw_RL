"""Which CAN board drives which joint, and in which direction.

The arm carries 24 McKibben actuators on 12 antagonistic pairs, one pair per
degree of freedom of the six universal joints.  Pressurising one member of a
pair drives its joint angle positive; pressurising the other drives it
negative.  This module is the single place that correspondence is written down.

TWO TABLES LIVE HERE, and the difference between them is the point.

:data:`LEGACY_JOINT_PAIRS` is the **claim**, transcribed from the research
tree's ``robot_constants.py`` (``actuator_address_list`` at line 286 composed
with ``q_to_actuator_list_index`` at line 292, whose first column is the
actuator that drives positive theta).  That file describes the arm as it was
wired before the top regulator platform was replaced, so its lower 16 entries
(segments 2 and 3, the legacy 7 mm boards at ``0x109``-``0x118``) describe
hardware that has not changed, while its top eight (``0x101``-``0x108``, now
TLE/DVP boards on a new manifold) describe a platform that no longer exists.

:data:`MEASURED_JOINT_PAIRS` is the **measurement**: every board pressurised
alone against a resting arm on 2026-08-21, with the joint identified as the one
whose marker-derived angle moved most and the sign read off that motion
(``hw_tests/canarm_drive_campaign.py`` collects, ``canarm_axis_analysis.py``
reduces).  :data:`MEASURED` says whether that campaign has run.

Use :func:`joint_pairs` rather than either table directly: it returns the
measured map when there is one and refuses otherwise, which is the behaviour a
controller wants — an actuator map that is wrong by one pair does not fail
loudly, it drives the arm somewhere unexpected.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The legacy claim (research tree, pre-2026-08 top platform)
# --------------------------------------------------------------------------

#: ``actuator_address_list`` — legacy list index -> CAN base id.  Kept whole so
#: the composition below can be checked against the source line by line.
LEGACY_ADDRESS_LIST = (
    0x104, 0x106, 0x108, 0x102, 0x105, 0x107, 0x101, 0x103,
    0x109, 0x10C, 0x10B, 0x10A, 0x10D, 0x10E, 0x10F, 0x110,
    0x111, 0x112, 0x113, 0x114, 0x115, 0x118, 0x117, 0x116,
)

#: ``q_to_actuator_list_index`` rows 0..11 — joint -> (positive, negative) list
#: index.  The legacy file carries four rows more, for a fourth segment this
#: arm does not have; they are dropped rather than carried as dead weight.
LEGACY_JOINT_LIST_INDEX = (
    (3, 1), (0, 2), (7, 5), (4, 6),
    (11, 9), (8, 10), (15, 13), (12, 14),
    (19, 17), (16, 18), (23, 21), (20, 22),
)

#: Joint ``j`` -> ``(base_positive, base_negative)``, the legacy claim.
LEGACY_JOINT_PAIRS = tuple(
    (LEGACY_ADDRESS_LIST[pos], LEGACY_ADDRESS_LIST[neg])
    for pos, neg in LEGACY_JOINT_LIST_INDEX
)

# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

#: ``True`` once :data:`MEASURED_JOINT_PAIRS` came from a live drive campaign
#: on this arm rather than from the legacy file.
MEASURED = True

#: Provenance of :data:`MEASURED_JOINT_PAIRS`: the campaign record it was
#: reduced from.
MEASURED_SOURCE = (
    "hw_tests/results/drive_2026-08-21.json -- 24/24 boards pressurised alone "
    "at 12 psi against a resting arm, joint identified as the argmax of |dq| "
    "against the bracketing baselines, reduced by "
    "hw_tests/canarm_axis_analysis.py.  Reproduced independently on the "
    "earlier drive_sweep_2026-08-21.json run, which differed in its idle "
    "setpoint (0 psi rather than 0.5) and gave the same 24 rows.")

#: Joint ``j`` -> ``(base_positive, base_negative)``, measured 2026-08-21.
#:
#: **Segments 2 and 3 (``0x109``-``0x118``) reproduce the legacy table exactly**,
#: pairing and sign — which is also what fixes the sign convention, since a
#: robot frame rotated 180 deg about z would flip every sign and disagree with
#: sixteen boards at once.
#:
#: **Segment 1 (``0x101``-``0x108``) is rotated by +90 deg** relative to the
#: legacy table: the four antagonistic *pairs* are the same sets of boards, but
#: each pair drives the other axis of its universal joint, with the sign that a
#: +90 deg rotation implies (a pair formerly on +x now drives +y; a pair
#: formerly on +y now drives -x).  That is exactly the signature of the top
#: regulator platform having been remounted a quarter turn round, and it is why
#: the legacy table must not be used on this arm.
MEASURED_JOINT_PAIRS = (
    (0x108, 0x104), (0x102, 0x106), (0x101, 0x105), (0x103, 0x107),
    (0x10A, 0x10C), (0x109, 0x10B), (0x110, 0x10E), (0x10D, 0x10F),
    (0x114, 0x112), (0x111, 0x113), (0x116, 0x118), (0x115, 0x117),
)

#: Boards whose CAN id block places them on each segment's regulator platform.
#: Verified by the same campaign: every board's dominant ``dq`` fell inside its
#: own block's four joints, 24 for 24.
SEGMENT_BLOCKS = (range(0x101, 0x109), range(0x109, 0x111), range(0x111, 0x119))

#: Boards observed misbehaving on 2026-08-21, so a later session does not spend
#: an hour rediscovering them.  ``0x110`` leaks from its supply side: left
#: disabled it drifted from 0.1 to 7.0 psi across the sweep, which is why every
#: campaign here holds unused boards *enabled* at a small idle setpoint rather
#: than disabling them.  ``0x104`` and ``0x110`` also read a non-zero pressure
#: at rest (+1.1 and +5.1 psi), so their commanded psi is offset by that much.
KNOWN_BOARD_FAULTS = {
    0x110: "leaks from the supply side; reads +5.1 psi at rest",
    0x104: "reads +1.1 psi at rest",
}

#: Joints, in ``q`` order, named the way the mechanism is named: segment,
#: which universal joint of that segment, and which of its two axes.
JOINT_NAMES = (
    "s1.u1.t1", "s1.u1.t2", "s1.u2.t3", "s1.u2.t4",
    "s2.u3.t1", "s2.u3.t2", "s2.u4.t3", "s2.u4.t4",
    "s3.u5.t1", "s3.u5.t2", "s3.u6.t3", "s3.u6.t4",
)


def joint_pairs(allow_legacy: bool = False):
    """The 12 antagonistic pairs, measured if a campaign has run.

    ``allow_legacy`` lets an offline caller (a plumbing test, a plot of the
    nominal wiring) opt into the unverified table explicitly.  A controller
    must not.
    """
    if MEASURED:
        return MEASURED_JOINT_PAIRS
    if allow_legacy:
        return LEGACY_JOINT_PAIRS
    raise RuntimeError(
        "the CAN arm's actuator/axis map has not been measured on this "
        "hardware; the legacy table describes the pre-2026-08 top regulator "
        "platform. Run hw_tests/canarm_drive_campaign.py, reduce it with "
        "canarm_axis_analysis.py, and write MEASURED_JOINT_PAIRS here.")


def base_to_joint(allow_legacy: bool = False) -> dict:
    """``{base: (joint, sign)}`` — sign +1 drives the joint angle positive."""
    out: dict[int, tuple[int, int]] = {}
    for j, (pos, neg) in enumerate(joint_pairs(allow_legacy)):
        out[pos] = (j, +1)
        out[neg] = (j, -1)
    return out


__all__ = [
    "LEGACY_ADDRESS_LIST",
    "LEGACY_JOINT_LIST_INDEX",
    "LEGACY_JOINT_PAIRS",
    "MEASURED",
    "MEASURED_SOURCE",
    "MEASURED_JOINT_PAIRS",
    "SEGMENT_BLOCKS",
    "KNOWN_BOARD_FAULTS",
    "JOINT_NAMES",
    "joint_pairs",
    "base_to_joint",
]
