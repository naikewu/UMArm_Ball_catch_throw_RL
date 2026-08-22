"""Receivers bound to the CAN UMArm's block of Motive rigid bodies.

The CAN UMArm streams **six** rigid bodies, ids **2000 to 2005**, base to tip.
**Verified live on 2026-08-21**: a wide-open census
(``hw_tests/mocap_census.py``) enumerated the whole volume at 115 Hz and found
exactly three contiguous blocks — ``500``-``505`` (the RS485 sister arm),
``1008`` (the Kinova Gen3) and ``2000``-``2005`` (this arm) — each carrying
four labeled markers in slots 1..4.  Height settles which end is which: body
2000 sits at z = 0.92 m and 2005 at z = 0.04 m, and this arm hangs base-up, so
2000 is the base plate and the array index is ``id - 2000``.  The three-way id
dispute the 2026-08-20 notes recorded is therefore closed, and the 500-block
claim in ``mocap_constants`` is right about the *other* arm rather than wrong.

(The 2026-08-20 session saw no stream at all — Motive's host answered ICMP but
its streaming engine was not serving, so NAT_PING went unanswered and the
census came back empty.  Nothing in this package was at fault, and nothing here
had to change when the stream came back.)

TWO RECEIVERS, AND THE DIFFERENCE THAT MATTERS.  :class:`CanArmMocap` builds
``q`` from the poses Motive streams.  :class:`CanArmMarkerMocap` builds it from
the four markers of each plate, registered against a rest-time template.  On
this arm the difference is not a refinement, it is a correction: Motive's
manually aligned body frames sit about 45 deg round from the mechanism's axes
(measured 2026-08-21 — the streamed body x lands within ~2 deg of a marker
diagonal, and the revolute axes lie along the diagonals), and a ``q`` read off
the streamed frames is smooth, repeatable and wrong by that rotation.  Prefer
the marker receiver for anything that acts on ``q``; keep the streamed one for
transport diagnostics, where the pose is only being used to say "the body is
there".

WHAT IS NOT PORTABLE FROM THE RS485 ARM.  Marker locks are minted per Motive
session against particular plates, so the RS485 arm's committed ``locks.json``
describes neither these plates nor this asset roster.
:data:`DEFAULT_TEMPLATE_PATH` names this arm's file; ``load_canarm_locks``
raises rather than falling back when it is absent.  Mint with
``mocap_probe.py --rb-id-base 2000 --n-bodies 6 --x-mode diagonal45
--lock-out <path>``, with :func:`UMArm_MOCAP.canarm_frames.mint_locks` against
a rest capture, or from the control GUI's **Lock plates** button.  The RS485
file is carried at ``templates/rs485_locks_example.json`` purely as a worked
example of the format.
"""

from __future__ import annotations

import os

# Package-relative only, unlike mocap_rx/mocap_to_q which support both styles:
# this module reaches ``marker_mocap``, which is itself package-relative only
# (``marker_mocap.py:53``).  Import it as ``UMArm_MOCAP.canarm_mocap``.
from .marker_mocap import MarkerMocap
from .mocap_rx import MocapRx

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Motive streaming id of the CAN arm's base plate.  Array row ``i`` is id
#: ``CANARM_RB_ID_BASE + i``.  Verified live 2026-08-21; see the module
#: docstring.
CANARM_RB_ID_BASE = 2000

#: How many rigid bodies that block carries: 2000 (base) .. 2005 (tip).  Six,
#: not the RS485 arm's seven, because the CAN arm's Motive project carries no
#: end-effector plate.  Verified live 2026-08-21.
CANARM_N_BODIES = 6

#: The CAN arm's marker locks.  Session-scoped and therefore not checked in;
#: loading must refuse loudly rather than reach for the RS485 arm's templates,
#: because a lock minted against different plates registers markers onto
#: geometry that is not there and reports a confident ``q`` for it.
DEFAULT_TEMPLATE_PATH = os.path.join(_HERE, "templates", "canarm_locks.json")


class CanArmMocap(MocapRx):
    """:class:`MocapRx` bound to rigid bodies 2000-2005 rather than 500-506.

    Nothing else differs.  Transport, rings, staleness, marker health,
    ``capture_rest`` and the streamed-pose ``q`` solve are all inherited, so a
    CAN-arm receiver and an RS485-arm receiver are the same code reading two
    blocks of one NatNet stream, and can run side by side in one process.

    Its ``q`` is read off **Motive's** body frames, whose azimuth this arm's
    2026-08-21 session measured 45 deg away from the mechanism — see the module
    docstring, and prefer :class:`CanArmMarkerMocap` for control.
    """

    def __init__(self, *,
                 rb_id_base: int = CANARM_RB_ID_BASE,
                 n_bodies: int = CANARM_N_BODIES,
                 **kwargs) -> None:
        super().__init__(rb_id_base=rb_id_base, n_bodies=n_bodies, **kwargs)


class CanArmMarkerMocap(MarkerMocap):
    """:class:`MarkerMocap` bound to the same block, with this arm's azimuths.

    Three things are supplied here that the base class cannot know: the id
    block, **the per-plate bracket azimuth**, and **which hinge of the proximal
    universal joint is bolted to the upper bracket**
    (``canarm_frames.PROXIMAL_ORDER``, measured rather than assumed).  Those
    two defaults are the whole reason this subclass is worth having — ``MarkerMocap`` falls back to the
    RS485 arm's ``FAMILY_PHI_RAD``, this arm's brackets are 45 deg round from
    those, and the failure mode of getting it wrong is a plausible ``q`` rather
    than an error.  Pass ``phis=None`` explicitly to opt out and take the
    family defaults, which is what the before/after comparison in
    ``hw_tests/canarm_axis_analysis.py`` does.

    *locks* must be the **CAN arm's** locks; see :data:`DEFAULT_TEMPLATE_PATH`.
    ``MarkerMocap.REQUIRED_PLATES`` is already ``range(6)``, which is exactly
    this arm's plate count, so nothing about the required set needs relaxing.
    """

    #: Sentinel distinguishing "the caller said nothing" from "the caller asked
    #: for the family defaults", which are different requests and would
    #: otherwise both arrive as ``None``.
    _AZIMUTH_DEFAULT = object()

    def __init__(self, locks: dict, *,
                 rb_id_base: int = CANARM_RB_ID_BASE,
                 n_bodies: int = CANARM_N_BODIES,
                 phis=_AZIMUTH_DEFAULT,
                 order=_AZIMUTH_DEFAULT,
                 **kwargs) -> None:
        from .canarm_frames import PROXIMAL_ORDER, plate_phis_rad
        if phis is CanArmMarkerMocap._AZIMUTH_DEFAULT:
            phis = plate_phis_rad()
        if order is CanArmMarkerMocap._AZIMUTH_DEFAULT:
            order = PROXIMAL_ORDER
        super().__init__(locks, rb_id_base=rb_id_base, n_bodies=n_bodies,
                         phis=phis, order=order, **kwargs)


def load_canarm_locks(path: str | None = None) -> dict:
    """The CAN arm's locks, or a refusal that names what is missing.

    Wraps :func:`marker_mocap.load_locks` only to make the absent-file case
    say *which* file and *how to make one*, since "no such file" on a path
    nobody has minted yet is a question rather than an error.
    """
    from .marker_mocap import load_locks

    path = path or DEFAULT_TEMPLATE_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "%s does not exist.  The CAN arm's marker locks have not been "
            "minted for this Motive session; the RS485 arm's "
            "templates/rs485_locks_example.json describes different plates and "
            "must not be substituted.  Mint with: python "
            "UMArm_MOCAP/mocap_probe.py --rb-id-base %d --n-bodies %d "
            "--x-mode diagonal45 --lock-out %s, or press Lock plates in "
            "canarm_control_gui.py."
            % (path, CANARM_RB_ID_BASE, CANARM_N_BODIES, path))
    return load_locks(path)


__all__ = [
    "CANARM_RB_ID_BASE",
    "CANARM_N_BODIES",
    "DEFAULT_TEMPLATE_PATH",
    "CanArmMocap",
    "CanArmMarkerMocap",
    "load_canarm_locks",
]
