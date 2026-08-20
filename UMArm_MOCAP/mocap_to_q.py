"""Rigid-body poses -> the UMArm 12-DOF joint vector ``q``, in pure numpy.

A clean re-implementation of the legacy ``mocap_to_config.mocap_to_config``
(itself ``kinematics_mp.get_configuration_from_mocap_mk8``).  Same geometry, same
conventions, same numbers — verified against the legacy code as an oracle in
``test_mocap_to_q.py`` (max |dq| < 1e-9 over 500 random pose sets plus the
structured cases).

Why re-implement instead of importing the old folder: this file must run in the
real-time control loop of a different repo, with no dependency on a research
tree that also happens to contain the Kinova stack, and with **no scipy**.  The
only scipy call in the legacy path was ``Rotation.align_vectors``, five times per
frame inside the per-joint loop; it is replaced here by a direct Rodrigues
construction (see :func:`rotation_aligning`).  Dependency removal was the point,
but it is also faster — measured on the bench PC (numpy 2.5, scipy 1.18): 17 us
per align call against scipy's 33 us, i.e. 122 us against 206 us for a whole
frame, against an 8.3 ms budget at 120 Hz.

------------------------------------------------------------------------------
THE GEOMETRY, IN ONE PARAGRAPH
------------------------------------------------------------------------------
The arm is a serial chain of six universal joints, each with 2 DOF, hence 12
angles.  Motive tracks a plate at every U-joint centre, so each consecutive pair
of centres gives a *link vector*.  A serial joint angle only means something
relative to the link before it, so link *k* is read in the frame where link
*k-1* points along +z (that is the ``align`` step), the "even" joints get the
45 deg mounting offset removed (``RZn45``), and the two angles then fall out of
one ``atan2`` pair.  Link vectors are taken **proximal minus distal**
(``pos[k] - pos[k+1]``), which points back up toward the base — the arm hangs
base-up, so at rest every link vector is +z in the base frame.

------------------------------------------------------------------------------
CONTRACT
------------------------------------------------------------------------------
Input  ``homos``: array-like ``(>=7, 4, 4)`` — SE(3) pose of each tracked plate
        in the mocap spatial frame, ordered by the index map in
        :mod:`mocap_constants`.  Rotations of index 0 and 5 are used; positions
        of 0..5 are used; index 6 is required to be present but its position is
        not part of mk8's math.
Output ``q``: ``ndarray (12,)`` in **radians**, ordered
        ``[u1_t1, u1_t2, u2_t1, u2_t2, ... u6_t1, u6_t2]`` (joints 0-3 =
        segment 1, 4-7 = segment 2, 8-11 = segment 3) — or ``None`` when the
        frame carries no usable data.

``q`` is repeatable, not absolute: the zero depends on how the plates are
mounted, and calibration owns that offset.
"""

from __future__ import annotations

import numpy as np

# Dual-mode import: this module is used both as a package submodule
# (``from UMArm_MOCAP.mocap_to_q import mocap_to_q``, e.g. from
# UMArm_ROBOT_CONTROL) and as a flat module by lab scripts and the test, which
# put this directory on sys.path.  Neither style should be the only one that
# works.
try:  # pragma: no cover - exercised by whichever import style the caller uses
    from . import mocap_constants as mc
except ImportError:  # pragma: no cover
    import mocap_constants as mc  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# Reading the two angles off one link vector
# --------------------------------------------------------------------------


def ujoint_angles(v) -> tuple[float, float]:
    """The two universal-joint angles (radians) encoded by a direction ``v``.

    Verbatim the convention used everywhere in the legacy stack::

        theta1 = atan2(v_z, v_y) - pi/2          # rotation read in the y-z plane
        theta2 = -(atan2(hypot(v_z, v_y), v_x) - pi/2)   # tilt off the x axis

    Both are zero for ``v = +z``, i.e. for a joint whose link continues straight
    along the previous one.  The offsets of ``pi/2`` and the minus sign in front
    of ``theta2`` are what puts the zero there and fixes the handedness; they are
    part of the hardware convention, not a simplification to be tidied away.

    ``v`` need not be unit length — both formulas are scale invariant.
    """
    v = np.asarray(v, dtype=float)
    theta1 = np.arctan2(v[2], v[1]) - np.pi * 0.5
    theta2 = -(np.arctan2(np.hypot(v[2], v[1]), v[0]) - np.pi * 0.5)
    return float(theta1), float(theta2)


def unit_from_ujoint_angles(theta1: float, theta2: float) -> np.ndarray:
    """Analytic inverse of :func:`ujoint_angles`: angles -> unit direction.

    Inverting the two ``atan2`` calls above, with ``phi = theta1 + pi/2`` the
    in-plane angle and ``psi = pi/2 - theta2`` the polar angle off +x::

        v = [cos(psi), sin(psi) * cos(phi), sin(psi) * sin(phi)]

    Exact for ``theta2`` in ``[-pi/2, pi/2]`` and ``theta1`` in ``(-3pi/2,
    pi/2]``, which covers the whole physical range with room to spare (the
    ``atan2`` branch cuts sit far outside it).

    This is what makes the round-trip test possible: it lets a *known* ``q`` be
    turned into plate positions, so the reader can be checked against ground
    truth rather than only against the legacy implementation.  It is also the
    honest place to look up the sign convention — ``+theta1`` swings the link
    vector toward -y, hence swings the distal end (which lies along ``-v``)
    toward +y.
    """
    phi = theta1 + np.pi * 0.5
    psi = np.pi * 0.5 - theta2
    s = np.sin(psi)
    return np.array([np.cos(psi), s * np.cos(phi), s * np.sin(phi)], dtype=float)


# --------------------------------------------------------------------------
# The scipy replacement: minimal rotation mapping one vector onto another
# --------------------------------------------------------------------------

#: Below this ``|sin(angle)|`` a rotation of ~180 deg is treated as exactly
#: antiparallel.  Deliberately tiny: inside this window the rotation axis is
#: mathematically undetermined (any axis perpendicular to the target works), so
#: the tolerance only decides *which* arbitrary answer is returned, and it must
#: be small enough never to pre-empt a well-conditioned case.  1e-12 means the
#: link would have to be within a picoradian of pointing exactly backwards.
ANTIPARALLEL_SIN_TOL = 1e-12


def _skew(c: np.ndarray) -> np.ndarray:
    """Matrix form of the cross product: ``_skew(c) @ x == cross(c, x)``."""
    return np.array([
        [0.0, -c[2], c[1]],
        [c[2], 0.0, -c[0]],
        [-c[1], c[0], 0.0],
    ], dtype=float)


def _orthogonal_axis(a: np.ndarray) -> np.ndarray:
    """Some unit vector perpendicular to unit ``a``, chosen deterministically.

    Zeroing ``a``'s smallest component and rotating the other two by 90 deg
    gives the numerically safest perpendicular (the two surviving components are
    the largest ones, so the result is never near zero).  This mirrors the rule
    scipy uses in the same corner of ``align_vectors``, so even the exactly
    antiparallel case agrees with the oracle: for ``a = +z`` both pick the y
    axis.
    """
    i = int(np.argmin(np.abs(a)))
    r = np.zeros(3, dtype=float)
    r[i - 1] = a[i - 2]
    r[i - 2] = -a[i - 1]
    return r / np.linalg.norm(r)


def rotation_aligning(v_from, v_to) -> np.ndarray:
    """Minimal rotation ``R`` with ``R @ v_from`` parallel to ``v_to``.

    Drop-in replacement for the legacy
    ``Rotation.align_vectors(v_to, v_from)[0].as_matrix()`` in the single-vector
    case, where "best fit" degenerates to "shortest rotation": the axis is
    ``v_from x v_to`` and the angle is the angle between them.  numpy only, no
    quaternion round trip, no object allocation.

    Three branches, each covering the regime where it is the best conditioned:

    * ``dot > 0`` (< 90 deg apart, i.e. every pose the arm can physically reach)
      uses the trig-free closed form ``I + K + K@K/(1+dot)``.  With ``dot > 0``
      the ``1+dot`` divisor is at least 1, so there is no cancellation, and this
      is exact at ``dot == 1`` (identity) where an axis-angle form would divide
      by a vanishing ``sin``.
    * ``>= 90 deg`` but not antiparallel: explicit axis-angle Rodrigues, which
      stays accurate as ``dot -> -1`` where ``1+dot`` would cancel catastrophically.
    * antiparallel (``|sin| <= ANTIPARALLEL_SIN_TOL``): the axis is undetermined —
      the cross product is zero, or just inside the tolerance is nothing but
      rounding noise — so pick a deterministic perpendicular (see
      :func:`_orthogonal_axis`) and turn 180 deg about it, ``R = 2 k k^T - I``.
      scipy applies the same perpendicular-picking rule when the cross product is
      *exactly* zero, so the oracle agrees there; in the picoradian-wide band
      just short of exactly backwards scipy keeps using the noisy axis and we do
      not.  Physically unreachable either way — it means a link folded back onto
      the one before it.

    Both inputs are normalised first (as scipy does), so callers may pass raw
    link vectors; a zero-length input raises, since "align nothing" has no
    answer.
    """
    b = np.asarray(v_from, dtype=float)
    a = np.asarray(v_to, dtype=float)
    nb = np.linalg.norm(b)
    na = np.linalg.norm(a)
    if nb == 0.0 or na == 0.0:
        raise ValueError("cannot align a zero-length vector")
    b = b / nb
    a = a / na

    cross = np.cross(b, a)
    dot = float(b @ a)
    sin = float(np.linalg.norm(cross))

    if dot > 0.0:
        k = _skew(cross)
        return np.eye(3) + k + (k @ k) / (1.0 + dot)

    if sin > ANTIPARALLEL_SIN_TOL:
        k = _skew(cross / sin)
        angle = np.arctan2(sin, dot)
        return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)

    axis = _orthogonal_axis(a)
    return 2.0 * np.outer(axis, axis) - np.eye(3)


def quat_xyzw_to_matrix(quat) -> np.ndarray:
    """Rotation matrix from a NatNet ``[x, y, z, w]`` quaternion, numpy only.

    Replaces ``scipy Rotation.from_quat(q).as_matrix()`` in the receiver.  The
    component order is scipy's and NatNet's (**scalar last**) — Motive streams
    ``qx qy qz qw``, so this is the one place where getting the order wrong
    yields a plausible-looking but silently wrong base frame.  The quaternion is
    normalised first, matching scipy, because Motive's values are only unit to
    float32 precision.

    A zero-norm or non-finite quaternion **raises** ``ValueError``, exactly as
    ``Rotation.from_quat`` does ("Found zero norm quaternions").  This is not
    defensive boilerplate: Motive streams ``pos=(0,0,0), quat=(0,0,0,0)`` for a
    rigid body it cannot solve, and ``NatNetClient.__unpack_rigid_body`` hands
    that to the listener *before* it parses the tracking-valid bit.  Normalising
    it silently yields an all-NaN matrix (a bare RuntimeWarning, no exception),
    which de-rotates every link into NaN and publishes a frame that looks
    perfectly healthy — see the "known limitation" note in :mod:`mocap_rx`.
    Raising puts the failure where the receiver's listener can record it.
    """
    q = np.asarray(quat, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError(f"quaternion must be finite and non-zero; got {q!r}")
    x, y, z, w = q / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


# --------------------------------------------------------------------------
# The conversion
# --------------------------------------------------------------------------


def link_vectors(homos) -> np.ndarray | None:
    """The six link directions, in the base frame, unit length — or ``None``.

    Five come from differences of consecutive U-joint centres; the sixth has no
    centre beyond it to difference against, so it is plate 5's own body z-axis.
    Every vector is de-rotated by the base plate's orientation
    (``R_base^T @ v``) before use, which is what makes ``q`` independent of how
    the mocap volume happens to be oriented — move the whole rig and ``q`` does
    not change.

    ``None`` means the frame is unusable, in exactly two cases:

    * **A collapsed link.** Two *adjacent* plates at the same point — which is
      what two plates Motive has not seen yet look like, both still at the
      identity — makes the "direction" ``0/0``.
    * **A non-finite result.** Any NaN or infinity reaching the output, whether
      from a plate position or from the base/plate-5 rotation.

    Note the first case needs two *adjacent* plates to collapse.  A *single*
    untracked plate leaves both of its neighbouring differences non-degenerate,
    so the frame converts and ``q`` looks plausible while the affected joints
    actually encode the direction to the mocap volume's origin.  That hole is
    inherited from the legacy implementation and deliberately not closed here: a
    plausibility bound on link length would be the fix, but it would also make
    this function disagree with the oracle on the random pose sets the
    equivalence test is built from.  Detecting one bad plate belongs upstream,
    with Motive's tracking-valid bit — see the "known limitation" note in
    :mod:`mocap_rx`, and the raw U-joint centres in its ring buffer.
    """
    homos = np.asarray(homos, dtype=float)
    if homos.ndim != 3 or homos.shape[1:] != (4, 4) or homos.shape[0] < mc.N_USED_RIGID_BODIES:
        raise ValueError(
            f"homos must have shape (>= {mc.N_USED_RIGID_BODIES}, 4, 4); got {homos.shape}")

    pos = homos[:, 0:3, 3]
    rot_base = homos[mc.IDX_BASE, 0:3, 0:3]
    last_z = homos[mc.IDX_U3_DISTAL, 0:3, 2]

    # Reject non-finite input before doing arithmetic on it.  Checking the inputs
    # rather than only the result keeps NaN/inf out of the norms and matmuls
    # below, which would otherwise raise RuntimeWarnings on the SDK thread on the
    # way to the same answer.  Only the slices the math actually reads are
    # checked: rows 7-8 (spare, Kinova) are none of this function's business.
    if not (np.isfinite(pos[0:6]).all() and np.isfinite(rot_base).all()
            and np.isfinite(last_z).all()):
        return None

    rot_base_t = np.transpose(rot_base)

    raw = [pos[i] - pos[i + 1] for i in range(5)]        # segment links + couplings
    raw.append(last_z.copy())                            # last link = plate 5 body z

    out = np.empty((6, 3), dtype=float)
    for i, v in enumerate(raw):
        n = np.linalg.norm(v)
        # Written as "not >=" rather than "<" so a NaN norm rejects too: every
        # comparison against NaN is False, so `n < MIN_LINK_NORM` would wave one
        # straight through.  Finite input can still norm to NaN — differences of
        # ~1e308 positions overflow — so this is not redundant with the gate above.
        if not (n >= mc.MIN_LINK_NORM):
            return None
        # Rotate first, scale after, exactly as the legacy line does — a
        # rotation preserves length, so this is the same number to the last bit
        # either way, but keeping the order makes the two implementations
        # trivially diffable.
        out[i] = (rot_base_t @ v) / n

    # Final invariant: this function never returns a non-finite direction.  The
    # norm gate sees only the *raw spatial* vector, so on its own it cannot catch
    # a bad rotation — a NaN base frame leaves every raw length perfectly finite
    # and turns every de-rotated link into NaN, which is how an all-NaN q used to
    # be published as a healthy frame.
    if not np.isfinite(out).all():
        return None
    return out


def mocap_to_q(homos) -> np.ndarray | None:
    """Convert rigid-body SE(3) poses to the 12-DOF joint vector ``q`` (radians).

    Returns ``None`` for a degenerate frame (see :func:`link_vectors`), which is
    what a caller sees before Motive has produced a first frame with all plates
    visible.  Stateless: every frame stands alone, so dropped frames are
    harmless and this is safe to call from any thread.
    """
    rv = link_vectors(homos)
    if rv is None:
        return None

    q = np.empty(mc.NUM_JOINTS, dtype=float)
    for k in range(6):
        if k == 0:
            # The first joint has no previous link: its vector is already
            # expressed in the base frame, which *is* its measuring frame.
            v = rv[0]
        else:
            v = rotation_aligning(rv[k - 1], mc.V_BASE) @ rv[k]
            if k % 2 == 1:
                # u2, u4, u6 — the joints mounted 45 deg round from their
                # neighbour.
                v = mc.RZn45 @ v
        q[2 * k], q[2 * k + 1] = ujoint_angles(v)
    return q
