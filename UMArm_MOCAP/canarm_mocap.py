"""Receivers bound to the CAN UMArm's block of Motive rigid bodies.

The CAN UMArm streams **six** rigid bodies, ids **2000 to 2005**, base to tip,
per the 2026-08-20 brief.  That number is **TO BE VERIFIED LIVE AGAINST MOTIVE**
and nothing in this workspace has yet seen a frame from those cameras.  It is
written here as a named constant rather than spread through the receiver so
that settling it later is one edit in one file.

WHY THE BASE ID IS IN DOUBT.  Three sources disagree, and only Motive can
arbitrate:

* the committed RS485 code says that arm is **500-505**
  (``mocap_constants.RIGID_BODY_ID_MASK = 500``, and the index map beside it
  describes 500..507 plus the Kinova at 1008);
* the 2026-08-20 brief says the RS485 arm is **1000-1005** and the CAN arm
  **2000-2005**;
* the legacy VNEMA client agrees with the brief about the RS485 arm
  (``portable_sync_comm_layer/mocap/mocap.py``: ``DEFAULT_ARM_IDS =
  range(1000, 1006)``).

Either the Motive project was renumbered since the 2026-08-11 recalibration and
the RS485 repo is stale, or the brief's 1000-1005 is approximate.  The live
check settles it.  Only the Kinova's **1008** is consistent across all three,
and it keeps its own routing in :class:`~UMArm_MOCAP.mocap_rx.MocapRx`
regardless of which arm a receiver is bound to.

WHAT IS SAFE ABOUT GUESSING WRONG.  ``MocapRx._on_rigid_body`` drops ids
outside its own block rather than trusting them, so a wrong base yields an arm
that never converts — loudly stale — not an arm whose joints quietly encode a
stray body.  What it does **not** protect against is a base that is wrong by an
amount that still lands inside another asset's block; that is why the check is
against Motive's asset list and not against whether frames arrive.

WHAT IS NOT PORTABLE FROM THE RS485 ARM.  Marker locks are minted per Motive
session against particular plates, so the RS485 arm's committed
``locks.json`` describes neither these plates nor this asset roster.
:data:`DEFAULT_TEMPLATE_PATH` names a file that **does not exist yet**, and
``marker_mocap.load_locks`` raises rather than falling back when it is absent.
Mint the CAN arm's own with ``mocap_probe.py --rb-id-base 2000 --n-bodies 6
--lock-out <path>`` or :func:`marker_mocap.mint_locks` against a live rest
capture.  The RS485 file is carried at ``templates/rs485_locks_example.json``
purely as a worked example of the format.
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
#: ``CANARM_RB_ID_BASE + i``.  **Unverified**; see the module docstring.
CANARM_RB_ID_BASE = 2000

#: How many rigid bodies that block carries: 2000 (base) .. 2005 (tip).  Six,
#: not the RS485 arm's seven, because the CAN arm's Motive project carries no
#: end-effector plate — which matches the RS485 volume in practice, where plate
#: 6 is reported ABSENT every window.  **Unverified**; see the module docstring.
CANARM_N_BODIES = 6

#: The CAN arm's marker locks.  **This file does not exist yet** and is not
#: created by this module; loading must refuse loudly rather than reach for the
#: RS485 arm's templates, because a lock minted against different plates
#: registers markers onto geometry that is not there and reports a confident
#: ``q`` for it.
DEFAULT_TEMPLATE_PATH = os.path.join(_HERE, "templates", "canarm_locks.json")


class CanArmMocap(MocapRx):
    """:class:`MocapRx` bound to rigid bodies 2000-2005 rather than 500-506.

    Nothing else differs.  Transport, rings, staleness, marker health,
    ``capture_rest`` and the streamed-pose ``q`` solve are all inherited, so a
    CAN-arm receiver and an RS485-arm receiver are the same code reading two
    blocks of one NatNet stream, and can run side by side in one process.

    ``rb_id_base`` and ``n_bodies`` remain overridable, since the brief's
    2000-2005 is not yet confirmed and an operator who finds the real numbers
    should not have to edit a module to use them.
    """

    def __init__(self, *,
                 rb_id_base: int = CANARM_RB_ID_BASE,
                 n_bodies: int = CANARM_N_BODIES,
                 **kwargs) -> None:
        super().__init__(rb_id_base=rb_id_base, n_bodies=n_bodies, **kwargs)


class CanArmMarkerMocap(MarkerMocap):
    """:class:`MarkerMocap` bound to the same block.

    ``MarkerMocap`` subclasses ``MocapRx`` and forwards ``**kwargs`` to it
    untouched, so the same two arguments reach the same five read sites and no
    further parameterization is needed.  What it adds over
    :class:`CanArmMocap` is the marker-registered ``q`` — the reason that path
    exists is that Motive is free to move a rigid body's pivot when it
    re-solves an asset, without telling any client, whereas the marker
    positions stay true.

    *locks* must be the **CAN arm's** locks; see :data:`DEFAULT_TEMPLATE_PATH`.
    ``MarkerMocap.REQUIRED_PLATES`` is already ``range(6)``, which is exactly
    this arm's plate count, so nothing about the required set needs relaxing.
    """

    def __init__(self, locks: dict, *,
                 rb_id_base: int = CANARM_RB_ID_BASE,
                 n_bodies: int = CANARM_N_BODIES,
                 **kwargs) -> None:
        super().__init__(locks, rb_id_base=rb_id_base, n_bodies=n_bodies,
                         **kwargs)


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
            "minted; the RS485 arm's templates in templates/"
            "rs485_locks_example.json describe different plates and must not "
            "be substituted.  Mint with: python UMArm_MOCAP/mocap_probe.py "
            "--rb-id-base %d --n-bodies %d --lock-out %s"
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
