"""The CAN arm's plate frames, inferred from markers alone.

WHY THIS EXISTS.  Motive streams a rigid body's orientation as well as its
position, but that orientation is whatever a human aligned by hand in Motive's
asset editor.  The alignment cannot be checked from the stream, it moves
whenever the asset is re-solved, and on this arm it is **45 deg away from the
mechanism's axes** — the 2026-08-21 session measured the streamed body x within
about 2 deg of a marker diagonal, while the revolute axes lie along the
diagonals too, so a pipeline that trusts the streamed frame inherits whatever
the last person to touch Motive did.  Everything here therefore reads the four
markers and nothing else.

WHAT THE MARKERS SAY, MEASURED RATHER THAN ASSUMED.  Each plate carries four
markers on radial arms about the u-joint centre.  The 2026-08-21 probe measured
them: the arms are 67-80 mm long and unequal, the two diagonals differ by
2.3-9.1 mm, their midpoints sit 2.7-5.4 mm apart, the four markers are coplanar
to 0.05-0.24 mm RMS, and the diagonal *lines* cross within 0.3 deg of
perpendicular.  So the centre is the intersection of the diagonal **lines** —
exact for radial arms of any length — and never the midpoint mean, which the
unequal arms would bias by half of that 2.7-5.4 mm.

THE ONE THING THE HARDWARE CONTRADICTED.  The brief says the marker arms poke
out *between* the actuator attachment points, i.e. 45 deg from the revolute
axes.  On this arm they do not: driving each of the twelve antagonistic pairs
alone and measuring the axis it actually rotates about puts the revolute axes
**along the diagonals**, to within 1 deg on segment 1 and 3 deg on segment 2
(``hw_tests/canarm_axis_analysis.py``).  That is a 45 deg difference, which is
exactly the size that a frame convention gets silently wrong and then hides
inside a plausible-looking ``q``.  :data:`PLATE_AZIMUTH_DEG` carries the
measured answer, and :data:`AZIMUTH_MEASURED` says it is a measurement.

THE FRAME, IN ONE PARAGRAPH.  Fit a plane to the four markers; the plate z is
its normal, signed so it points along the arm's own base-to-tip axis rather
than along any world axis (the volume is re-calibrated periodically; world axes
are precisely what may have moved).  The origin is the least-squares
intersection of the two diagonal lines.  The azimuth zero is one of the four
half-diagonals rotated by 45 deg, disambiguated against world +x — which fixes
the *branch*, not the angle — and then corrected by the per-plate
:data:`PLATE_AZIMUTH_DEG`, which is what makes the frame the **body** frame the
kinematics is written in.  At rest the whole recipe is captured once as a rigid
marker **template**; every later frame is a Kabsch registration of that template
onto whatever markers are visible, so **three of the four suffice** and nothing
directional is ever re-derived from a single frame.

WHAT IS INHERITED AND WHAT IS NEW.  The plane fit, the diagonal-line origin, the
template and the per-frame registration are :mod:`marker_frame`'s, unchanged —
they were written for the RS485 arm and they are arm-independent.  What is new
here is (a) the CAN arm's measured azimuths, (b) minting locks with **no**
reference to the streamed orientation at all, and (c) the co-rigidity check,
which is the cheapest honest test of the whole construction: plates 1/2 and 3/4
are bolted to the same connector, so their inferred body frames must agree, and
on 2026-08-21 they agreed to 0.03 deg of standard deviation over 49 arm poses.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

try:  # both import styles, as everywhere in this workspace
    from . import mocap_constants as mc
    from .marker_frame import (FAMILY_PHI_RAD, PlateLock, compute_plate_lock,
                               infer_plate_frame, q_from_frames)
except ImportError:  # pragma: no cover - flat style
    import mocap_constants as mc  # type: ignore[no-redef]
    from marker_frame import (FAMILY_PHI_RAD, PlateLock,  # type: ignore[no-redef]
                              compute_plate_lock, infer_plate_frame,
                              q_from_frames)

_HERE = os.path.dirname(os.path.abspath(__file__))

#: How many plates the CAN arm streams: rigid bodies 2000..2005.
N_PLATES = 6

#: Where :func:`load_locks` and :func:`save_locks` look by default.
DEFAULT_LOCK_PATH = os.path.join(_HERE, "templates", "canarm_locks.json")

#: Where :func:`load_calibration` looks for a session's azimuths and lengths.
DEFAULT_CALIB_PATH = os.path.join(_HERE, "templates", "canarm_frame_calib.json")

# --------------------------------------------------------------------------
# The measured azimuths
# --------------------------------------------------------------------------

#: ``True`` because every number in :data:`PLATE_AZIMUTH_DEG` came off this
#: arm.  Read it before trusting the frames for anything whose sign matters.
AZIMUTH_MEASURED = True

#: Provenance of :data:`PLATE_AZIMUTH_DEG`.
AZIMUTH_SOURCE = (
    "hw_tests/results/drive_2026-08-21.json, reduced by "
    "hw_tests/canarm_axis_analysis.py: 24 single-actuator drives at 12 psi "
    "plus multi-joint poses; proximal plates from the rotation axis of the "
    "intra-segment link tilt, distal plates from co-rigidity with the next "
    "proximal plate, plate 5 from the swing-free distal rotation axis, and "
    "the two intra-segment relative azimuths refined against the u-joint "
    "centre positions fkine predicts.")

#: Per-plate azimuth ``phi_p`` in **degrees**: the body frame is
#: ``R_body = R_inferred @ Rz(-phi_p)``, the same convention
#: :func:`marker_frame.q_from_frames` takes in its ``phis`` argument.
#:
#: Two facts are visible in these six numbers.  The even plates sit near
#: -45 deg and the odd plates near 0 deg, a 45 deg alternation, which is the
#: bracket family difference the mechanism was designed with — the lower
#: bracket of each segment is rotated 45 deg from the upper one.  And the whole
#: set is shifted 45 deg from :data:`marker_frame.FAMILY_PHI_RAD`'s
#: ``(0, 45, 0, 45, 0, 45)``, which is the "marker arms lie along the axes, not
#: between them" finding stated in the module docstring.
#:
#: Two independent measurements stand behind them and are worth keeping apart.
#: The *mechanism* one — the rotation axis each antagonistic pair actually
#: excites, for the proximal plates; co-rigidity with the next proximal plate,
#: for plates 1 and 3; the swing-free distal axis, for plate 5 — gives
#: ``(-45.190, +0.060, -45.018, -0.522, -45.490, +0.200)``.  The *position* one
#: refines the two intra-segment relative azimuths against the u-joint centres
#: fkine predicts and gives the values below.  **They agree to 0.82 deg**, and
#: the values shipped are the refined ones because reproducing the measured
#: centres is what this frame is for.
PLATE_AZIMUTH_DEG = (-44.557, -0.757, -45.834, -0.339, -45.306, 0.383)

#: The mechanism-only answer, kept beside the shipped one so the agreement
#: above is auditable rather than asserted.
PLATE_AZIMUTH_FROM_DRIVE_AXES_DEG = (-45.190, 0.060, -45.018, -0.522,
                                     -45.490, 0.200)

#: Which hinge of each **proximal** universal joint is bolted to the upper
#: bracket, expressed as the order the two twists compose in
#: (``UMArm_KINEMATICS.fkine.PROXIMAL_ORDERS``).  ``"yx"`` means the y hinge is
#: the fixed one and the x hinge is carried by it.
#:
#: Measured, not assumed, and the measurement is unusually clean: with the
#: legacy ``"xy"`` order the residual rotation the two-angle distal reader
#: drops is ``-0.91`` to ``-0.95`` times ``t1 * t2`` across all three segments
#: (residual standard deviation 0.09-0.25 deg against a raw spread of
#: 1.0-1.4 deg), which is the algebraic signature of exactly this swap.  Making
#: it collapses that residual to 0.12-0.31 deg of spread and takes the u-joint
#: centre error on held-out multi-joint poses from 4.6 mm RMS (worst 20.4 mm)
#: to 2.0 mm (worst 7.0 mm).  See ``UMArm_KINEMATICS.fkine``'s docstring.
PROXIMAL_ORDER = "yx"

#: The gauge.  A rotation applied to every plate at once turns the whole robot
#: frame about z and changes no predicted position, so one convention has to be
#: named: the body x of every plate is the branch nearest the mocap volume's
#: +x at rest.  That choice is not arbitrary in its consequences — it is what
#: makes the measured actuator/axis map reproduce the legacy table's signs on
#: all sixteen boards of segments 2 and 3.
AZIMUTH_GAUGE = "body x on the branch nearest mocap world +x at rest"


def plate_phis_rad(azimuth_deg=None) -> np.ndarray:
    """``(6,)`` radians, ready for :func:`marker_frame.q_from_frames`."""
    src = PLATE_AZIMUTH_DEG if azimuth_deg is None else azimuth_deg
    out = np.radians(np.asarray(src, dtype=float))
    if out.shape != (N_PLATES,):
        raise ValueError(f"azimuths must have shape ({N_PLATES},); got {out.shape}")
    return out


#: The family angles the RS485 arm uses, for the before/after comparison the
#: report makes.  Not this arm's answer; see :data:`PLATE_AZIMUTH_DEG`.
FAMILY_AZIMUTH_DEG = tuple(math.degrees(FAMILY_PHI_RAD[p]) for p in range(N_PLATES))


# --------------------------------------------------------------------------
# Minting locks
# --------------------------------------------------------------------------


def arm_up(rest_markers) -> np.ndarray:
    """The arm's own base-to-tip axis at rest, unit.

    ``normalize(centroid(plate 0) - centroid(plate 5))``.  Deliberately not a
    world axis: the plane normal's *sign* is the one thing in the construction
    that needs an external reference, and the reference must be something that
    moves with the arm rather than something a volume recalibration can rotate
    out from under it.  This arm hangs base-up, so the vector points up.
    """
    p0 = np.asarray(rest_markers[0], dtype=float).mean(axis=0)
    p5 = np.asarray(rest_markers[N_PLATES - 1], dtype=float).mean(axis=0)
    v = p0 - p5
    n = float(np.linalg.norm(v))
    if not (n > 1e-6):
        raise ValueError("plates 0 and 5 coincide; cannot derive the arm axis")
    return v / n


def mint_locks(rest_markers, plates=None) -> dict:
    """Rest marker clouds -> one :class:`~marker_frame.PlateLock` per plate.

    ``rest_markers`` maps plate index to either a ``(4, 3)`` mean cloud or an
    ``(n, 4, 3)`` stack of frames in which all four markers were tracked; the
    stack is preferred, because it is what puts a real jitter figure in the
    lock instead of ``nan``.

    ``streamed_rot`` is **never** passed on, which is the point of this
    function: the streamed orientation is not consulted even to break the
    45 deg branch ambiguity, so a lock minted here is reproducible from the
    marker file alone.  The branch is broken against world +x instead, and the
    angle that matters is then supplied by :data:`PLATE_AZIMUTH_DEG`.
    """
    plates = range(N_PLATES) if plates is None else plates
    u_up = arm_up(rest_markers)
    out = {}
    for p in plates:
        stack = np.asarray(rest_markers[p], dtype=float)
        if stack.ndim == 2:
            stack = stack[None]
        out[int(p)] = compute_plate_lock(stack, int(p), u_up,
                                         streamed_rot=None, x_mode="diagonal45")
    return out


def save_locks(locks: dict, path: str | None = None) -> str:
    path = path or DEFAULT_LOCK_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"schema": "canarm_locks/1",
               "azimuth_deg": list(PLATE_AZIMUTH_DEG),
               "azimuth_source": AZIMUTH_SOURCE,
               "plates": {str(p): lock.to_dict() for p, lock in sorted(locks.items())}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path


def load_locks(path: str | None = None) -> dict:
    """Locks from disk, or a refusal that says how to make them.

    Locks are minted per Motive session against particular plates; a lock from
    another session registers this session's markers onto geometry that has
    moved, and the template RMS gate in :func:`marker_frame.infer_plate_frame`
    is what catches that.  Refusing loudly when the file is missing is the same
    discipline one step earlier.
    """
    path = path or DEFAULT_LOCK_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist: the CAN arm's marker locks have not been "
            f"minted for this Motive session. Capture a rest window and run "
            f"hw_tests/canarm_axis_analysis.py --mint-locks, or call "
            f"canarm_frames.mint_locks() on a rest capture.")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {int(p): PlateLock.from_dict(d)
            for p, d in payload["plates"].items()}


# --------------------------------------------------------------------------
# Per-frame solve
# --------------------------------------------------------------------------


def infer_frames(frame_markers, frame_flags, locks: dict):
    """One frame -> ``(poses (6, 4, 4), valid (6,), quality {plate: FrameQuality})``.

    ``poses`` are the **inferred bracket** frames, identity where nothing was
    solved; :func:`body_frames` turns them into body frames.  Registration is
    :func:`marker_frame.infer_plate_frame`, so a plate with three of its four
    markers still solves exactly and a plate whose labels swapped fails the
    template-RMS gate rather than returning a confident wrong frame.
    """
    poses = np.tile(np.eye(4), (N_PLATES, 1, 1))
    valid = np.zeros(N_PLATES, dtype=bool)
    quality = {}
    for plate, lock in locks.items():
        plate = int(plate)
        if not 0 <= plate < N_PLATES:
            raise ValueError(f"lock for plate {plate} is outside 0..{N_PLATES - 1}")
        if frame_markers is None or plate not in frame_markers:
            continue
        flags = None if frame_flags is None else frame_flags.get(plate)
        t, qual = infer_plate_frame(frame_markers[plate], flags, lock)
        quality[plate] = qual
        if t is not None:
            poses[plate] = t
            valid[plate] = True
    return poses, valid, quality


def _rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_frames(poses, azimuth_deg=None) -> np.ndarray:
    """Bracket frames -> body frames: ``R_body = R_inferred @ Rz(-phi_p)``."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < N_PLATES:
        raise ValueError(f"poses must have shape (>= {N_PLATES}, 4, 4); "
                         f"got {poses.shape}")
    phis = plate_phis_rad(azimuth_deg)
    out = np.array(poses[0:N_PLATES], dtype=float)
    for p in range(N_PLATES):
        out[p, 0:3, 0:3] = poses[p, 0:3, 0:3] @ _rz(-float(phis[p]))
    return out


def q_from_plate_frames(poses, azimuth_deg=None, order: str | None = None):
    """Inferred bracket frames -> the 12-DOF ``q``, radians, or ``None``.

    A thin wrapper over :func:`marker_frame.q_from_frames` whose only job is to
    supply this arm's two measured conventions instead of the RS485 defaults:
    the per-plate azimuth and the proximal composition order.  Kept as a named
    function rather than a call-site convention because both are silent when
    wrong — the azimuth is 45 deg per plate and the order is a second-order
    twist, and each produces a ``q`` that looks entirely reasonable.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.shape[0] < mc.N_USED_RIGID_BODIES:
        pad = np.tile(np.eye(4), (mc.N_USED_RIGID_BODIES - poses.shape[0], 1, 1))
        poses = np.concatenate([poses, pad])
    return q_from_frames(poses, plate_phis_rad(azimuth_deg),
                         PROXIMAL_ORDER if order is None else order)


# --------------------------------------------------------------------------
# The checks that make the construction falsifiable
# --------------------------------------------------------------------------

#: Plate pairs bolted to the same connector body.  Their inferred **body**
#: frames must be identical, which is a rigid-body fact rather than a model
#: assumption — hence the sharpest available test of the whole frame
#: construction.  Measured spread on 2026-08-21: 0.03 deg of standard deviation
#: over 49 poses, 0.4 deg of worst-case out-of-plane tilt.
CO_RIGID_PAIRS = ((1, 2), (3, 4))


def co_rigid_residual_deg(poses, azimuth_deg=None) -> list:
    """Per co-rigid pair, ``(azimuth_error_deg, out_of_plane_deg)``.

    Both should be zero.  A non-zero azimuth error is a mis-set
    :data:`PLATE_AZIMUTH_DEG` for one of the two plates; a non-zero
    out-of-plane term is a marker set that has moved on its bracket, and no
    azimuth can absorb it.
    """
    b = body_frames(poses, azimuth_deg)
    out = []
    for a, c in CO_RIGID_PAIRS:
        r = b[a, 0:3, 0:3].T @ b[c, 0:3, 0:3]
        out.append((math.degrees(math.atan2(r[1, 0], r[0, 0])),
                    math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))))
    return out


def chain_gaps_m(poses) -> np.ndarray:
    """``(5,)`` consecutive u-joint centre distances, metres.

    Rigid: the plate centre *is* the joint centre on this arm (``UC1 = UC2 =
    0``), so these five numbers do not depend on the arm's pose and any drift
    in them across a campaign is measurement error, not motion.  Measured
    2026-08-21 over 49 poses: 265.37 / 72.85 / 234.47 / 72.97 / 230.12 mm with
    0.04-0.17 mm of standard deviation.
    """
    poses = np.asarray(poses, dtype=float)
    o = poses[0:N_PLATES, 0:3, 3]
    return np.linalg.norm(o[:-1] - o[1:], axis=1)


def fk_centres_world(q, base_body_se3, params=None,
                     order: str | None = None) -> np.ndarray:
    """``(6, 3)`` u-joint centres fkine predicts, in the mocap spatial frame.

    ``base_body_se3`` is plate 0's **body** pose — row 0 of :func:`body_frames`
    with the measured origin — which is the frame the kinematics is written in.
    ``params`` defaults to the CAN arm's measured table.
    """
    from UMArm_KINEMATICS.fkine import ujoint_centres

    if params is None:
        from UMArm_KINEMATICS.canarm_params import CANARM_PARAMS
        params = CANARM_PARAMS
    base = np.asarray(base_body_se3, dtype=float)
    if base.shape != (4, 4):
        raise ValueError(f"base_body_se3 must be (4, 4); got {base.shape}")
    return (ujoint_centres(q, params, PROXIMAL_ORDER if order is None else order)
            @ base[0:3, 0:3].T + base[0:3, 3])


def fk_residual_m(poses, azimuth_deg=None, params=None, order=None):
    """``(6,)`` distance between each measured plate centre and fkine's, metres.

    The one number the whole campaign exists to make small, and the honest way
    to read it: plate 0 is identically zero because the chain is anchored
    there, so the signal is plates 1 through 5.  ``None`` when the frame set
    does not convert.
    """
    poses = np.asarray(poses, dtype=float)
    q = q_from_plate_frames(poses, azimuth_deg, order)
    if q is None:
        return None
    b = body_frames(poses, azimuth_deg)
    pred = fk_centres_world(q, b[0], params, order)
    return np.linalg.norm(pred - poses[0:N_PLATES, 0:3, 3], axis=1)


__all__ = [
    "N_PLATES",
    "DEFAULT_LOCK_PATH",
    "DEFAULT_CALIB_PATH",
    "AZIMUTH_MEASURED",
    "AZIMUTH_SOURCE",
    "PLATE_AZIMUTH_DEG",
    "PLATE_AZIMUTH_FROM_DRIVE_AXES_DEG",
    "AZIMUTH_GAUGE",
    "PROXIMAL_ORDER",
    "FAMILY_AZIMUTH_DEG",
    "CO_RIGID_PAIRS",
    "plate_phis_rad",
    "arm_up",
    "mint_locks",
    "save_locks",
    "load_locks",
    "infer_frames",
    "body_frames",
    "q_from_plate_frames",
    "co_rigid_residual_deg",
    "chain_gaps_m",
    "fk_centres_world",
    "fk_residual_m",
]
