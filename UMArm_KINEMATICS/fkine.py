"""Forward kinematics of the UMArm mk5: ``q`` -> plate poses, pure numpy.

A clean re-implementation of the legacy ``kinematics_mp.fkine_mk5`` (only the
``use_optimization=False`` branch of the original is live — the ``True`` branch
dereferences ``rc.opvar = None``), verified oracle-equivalent in
``test_fkine.py`` (max |dT| = 7.2e-16 over 500 random q).  Why re-implement
instead of importing: the legacy module needs scipy at import time and lives in
a OneDrive research tree; this package must run on the bench PC's control side
with numpy alone (design ``docs/fkine_design.md`` D2/D5).

------------------------------------------------------------------------------
THE MODEL, IN ONE PARAGRAPH
------------------------------------------------------------------------------
Three identical-in-form segments, each a product of exponentials
(``kinematics_mp.py:241-297``)::

    T_i = T_{i-1} @ Tz(-JD_i) @ e1234_i(q[4i:4i+4]) @ Tz(-span_i)

where ``span = UC1+AA1+LL+AA2+UC2`` and ``e1234`` is the ordered product
``exp(xi1 t1) exp(xi2 t2) exp(xi3 t3) exp(xi4 t4)`` of four revolute twists,
per segment frame (``kinematics_mp.py:393-426`` at ``Distance = 0``):

* xi1: about **+x** through ``(0, 0, -UC1)`` — the proximal u-joint centre;
* xi2: about **+y** through the same point;
* xi3: about **(x+y)/sqrt(2)** through ``(0, 0, -(UC1+AA1+LL+AA2))`` — the
  distal centre;
* xi4: about **(-x+y)/sqrt(2)** through the same point.

**The 45 deg bracket lives entirely in the xi3/xi4 axis directions.**  ``gst0``
has identity rotation, so no Rz(45) ever accumulates between segments — the
mocap->q extraction applies ``RZn45`` to *vectors* it reads, and fkine must not
apply one to frames it builds (design §1).  The legacy ``get_e1234`` is a
Mathematica-generated closed form; here the same matrix is built as the
explicit four-factor product (design D1), readable and pinned equal to the
closed form by the oracle test (3.3e-16 in recon, 4.4e-16 in this port's
suite).

------------------------------------------------------------------------------
BODIES, PLATES AND CENTRES
------------------------------------------------------------------------------
With ``B_i = T_{i-1} @ Tz(-JD_i)`` (the segment's *proximal body* frame) and
``E_i = e1234_i``:

* proximal centre  ``u_{2i-1} = B_i @ [0, 0, -UC1, 1]`` (= the ``B_i`` origin,
  since UC1 = 0 on this hardware);
* distal centre    ``u_{2i}   = B_i @ E_i @ [0, 0, -(UC1+AA1+LL+AA2), 1]`` —
  invariant to t3/t4, both of whose axes pass through it (1.1e-16 in recon).

Plates are mounted on specific bodies of the chain ``base(plate0) -u1- link
-u2- connector(plates1+2) -u3- link -u4- connector(plates3+4) -u5- link -u6-
ee(plates5+6)``, so per-plate orientation follows the mounting model (design
D6, review finding geo-1):

* plates 0, 2, 4 (proximal side): rotation of ``B_i`` — invariant to their own
  u-joint's angles;
* plates 1, 3, 5 (distal side): rotation of ``B_i @ E_i`` — plate 5's rotation
  is the full ``T_3`` rotation **including q[10:12]**.  This is the orientation
  mk8 reads u6 from (``kinematics_mp.py:889-899``); q[10:12] moves *no centre*,
  only plate 5's orientation (and the absent EE body 506).

Every synthetic-pose consumer (tests, fake_mocap markers) goes through
:func:`plate_transforms`, so the mounting model lives in exactly one function.

------------------------------------------------------------------------------
CONTRACT
------------------------------------------------------------------------------
``q``: length 12, **radians**, ``[seg1: t1..t4, seg2: t1..t4, seg3: t1..t4]``
(recon verified mk8/``mocap_to_q`` matches this order, sign and units exactly
for single-joint excitation).  Frames are in the **robot (body-500) frame**:
plate 0 at the origin, arm hanging down -z.  Mocap correspondence
(``kinematics_mp.py:760-774``): predicted u1..u6 <-> bodies 500..505, and a
prediction in the mocap spatial frame is ``SE4(body 500) @ p_robot`` — that is
:func:`predict_spatial`.  Wrong-shaped input raises ``ValueError``; the legacy
print-and-return-``None`` is not ported (design §2).
"""

from __future__ import annotations

import numpy as np

# Dual-mode import: this module is used both as a package submodule
# (``from UMArm_KINEMATICS.fkine import fkine``) and as a flat module by lab
# scripts that put this directory on sys.path.  Same pattern as
# ``UMArm_MOCAP.mocap_to_q``; neither style should be the only one that works.
try:  # pragma: no cover - exercised by whichever import style the caller uses
    from . import robot_params as rp
except ImportError:  # pragma: no cover
    import robot_params as rp  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# Twist exponential (Rodrigues)
# --------------------------------------------------------------------------


def _skew(w: np.ndarray) -> np.ndarray:
    """Matrix form of the cross product: ``_skew(w) @ x == cross(w, x)``."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ], dtype=float)


def twist_exp(xi, theta: float) -> np.ndarray:
    """``exp(xi^ * theta)`` for a revolute twist, as a ``(4, 4)`` SE(3) matrix.

    ``xi = [v, w]`` in the legacy component order — **velocity first, axis
    last** (``kinematics_mp.get_xi_and_twist_exp`` builds e.g.
    ``xi1 = [0, -UC1, 0, 1, 0, 0]``).  For a revolute axis ``w`` through a
    point ``p``, ``v = -cross(w, p)``.

    Rodrigues, the standard unit-``w`` closed form (MLS eq. 2.36)::

        R = I + sin(theta) w^ + (1 - cos(theta)) w^ w^
        p = (I - R) @ cross(w, v) + w w^T v theta

    The ``w w^T v theta`` pitch term is kept for exactness of the formula, but
    it is identically zero for every twist this arm has (``v`` is perpendicular
    to ``w`` whenever ``v = -cross(w, p)``).  ``w`` must be unit length — the
    formula silently scales the *angle* otherwise, which is a bug factory, so a
    non-unit axis raises instead.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (6,):
        raise ValueError(f"twist must have shape (6,) = [v, w]; got {xi.shape}")
    v = xi[0:3]
    w = xi[3:6]
    if abs(float(np.linalg.norm(w)) - 1.0) > 1e-9:
        raise ValueError(
            f"revolute twist needs a unit rotation axis; got |w| = {np.linalg.norm(w)!r}")
    k = _skew(w)
    r = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    out = np.eye(4)
    out[0:3, 0:3] = r
    out[0:3, 3] = (np.eye(3) - r) @ np.cross(w, v) + np.outer(w, w) @ v * theta
    return out


# --------------------------------------------------------------------------
# One segment: four twists, one transform
# --------------------------------------------------------------------------

#: sqrt(1/2), spelled the way the Mathematica export spells ``2**(-1/2)`` so
#: the twist components are bit-identical to the legacy ``get_xi_and_twist_exp``
#: values at ``Distance = 0``.
_ISQ2 = 2.0 ** (-0.5)


def segment_twists(params_row) -> np.ndarray:
    """The four revolute twists of one segment, ``(4, 6)``, legacy ``[v, w]`` order.

    Transcribed from ``kinematics_mp.get_xi_and_twist_exp``
    (``kinematics_mp.py:397-402``) at ``Distance = 0`` — the Jacobian code
    passes cumulative distances to express every twist in the base frame; here
    each segment works in its own frame and the chaining is done by matrix
    product in :func:`fkine`, so ``Distance`` never appears.

    The proximal pair (xi1 about +x, xi2 about +y) passes through
    ``(0, 0, -UC1)``; the distal pair (xi3 about ``(x+y)/sqrt(2)``, xi4 about
    ``(-x+y)/sqrt(2)``) passes through ``(0, 0, -(UC1+AA1+LL+AA2))``.  Note the
    distal point does **not** include UC2 — the distal u-joint centre sits UC2
    short of the segment's end plate (both are the same point on this arm,
    where UC2 = 0, but the distinction is the legacy formula's).
    """
    row = np.asarray(params_row, dtype=float)
    if row.shape != (10,):
        raise ValueError(f"params row must have shape (10,); got {row.shape}")
    uc1 = row[rp.COL_UC1]
    # Distance from the segment frame origin down to the distal u-joint centre.
    # Written in the legacy term order (AA1+AA2+LL+UC1, kinematics_mp.py:399)
    # so the float sum is bit-identical to the oracle's.
    l_dist = row[rp.COL_AA1] + row[rp.COL_AA2] + row[rp.COL_LL] + uc1
    return np.array([
        [0.0, -uc1, 0.0, 1.0, 0.0, 0.0],                                # xi1: +x
        [uc1, 0.0, 0.0, 0.0, 1.0, 0.0],                                 # xi2: +y
        [l_dist * _ISQ2, -l_dist * _ISQ2, 0.0, _ISQ2, _ISQ2, 0.0],      # xi3: (x+y)/sqrt2
        [l_dist * _ISQ2, l_dist * _ISQ2, 0.0, -_ISQ2, _ISQ2, 0.0],      # xi4: (-x+y)/sqrt2
    ], dtype=float)


def segment_transform(params_row, q4) -> np.ndarray:
    """``e1234`` for one segment: the explicit ordered four-factor product.

    Equals the legacy Mathematica closed form ``exponentials_mk5.get_e1234``
    to 4.4e-16 (pinned by the oracle test over 500 random q), but readable:
    the factor order *is* the joint order, and each factor is one Rodrigues
    exponential (design D1)::

        exp(xi1 t1) @ exp(xi2 t2) @ exp(xi3 t3) @ exp(xi4 t4)
    """
    q4 = np.asarray(q4, dtype=float)
    if q4.shape != (4,):
        raise ValueError(f"segment angles must have shape (4,); got {q4.shape}")
    xi1, xi2, xi3, xi4 = segment_twists(params_row)
    return (twist_exp(xi1, q4[0]) @ twist_exp(xi2, q4[1])
            @ twist_exp(xi3, q4[2]) @ twist_exp(xi4, q4[3]))


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


def _tz(d: float) -> np.ndarray:
    """Pure z-translation SE(3): the ``JD_shift`` / ``gst0`` building block."""
    out = np.eye(4)
    out[2, 3] = d
    return out


def _as_q(q) -> np.ndarray:
    """Validate and return ``q`` as a float ``(12,)`` vector, or raise."""
    q = np.asarray(q, dtype=float)
    if q.shape != (12,):
        raise ValueError(
            f"q must have shape (12,) = [seg1 t1..t4, seg2 t1..t4, seg3 t1..t4]; "
            f"got {q.shape}")
    return q


def _chain_frames(q: np.ndarray, params: np.ndarray):
    """Yield ``(row, B_i, B_i @ E_i)`` for each segment, proximal to distal.

    ``B_i = T_{i-1} @ Tz(-JD_i)`` is the segment's proximal body frame — the
    frame the proximal-side plate is rigid to; ``B_i @ E_i`` is the distal body
    frame — what the distal-side plate is rigid to.  Everything public
    (:func:`fkine`, :func:`ujoint_centres`, :func:`plate_transforms`) is a thin
    projection of this walk, so the chain recursion exists exactly once.
    """
    t = np.eye(4)
    for i in range(3):
        row = params[i]
        span = ((((row[rp.COL_UC1] + row[rp.COL_AA1]) + row[rp.COL_LL])
                 + row[rp.COL_AA2]) + row[rp.COL_UC2])
        b = t @ _tz(-row[rp.COL_JD])
        be = b @ segment_transform(row, q[4 * i:4 * i + 4])
        yield row, b, be
        t = be @ _tz(-span)


def fkine(q, params=None) -> np.ndarray:
    """``(4, 4)`` pose of the last plate, in the robot (body-500) frame.

    Oracle-equivalent to the legacy ``fkine_mk5(params, q)``
    (``kinematics_mp.py:241-297``, live branch).  The returned frame is
    exactly the plate-5 body pose: its origin is the sixth u-joint centre
    (``fkine(0)`` translation = ``(0, 0, -0.692772)``) and its rotation is the
    full chain rotation including q[10:12] — ``gst0`` is a pure translation, so
    the rotation of ``T_3`` and of ``B_3 @ E_3`` are the same matrix.
    """
    q = _as_q(q)
    params = rp.as_params(params)
    t = np.eye(4)
    for row, _b, be in _chain_frames(q, params):
        span = ((((row[rp.COL_UC1] + row[rp.COL_AA1]) + row[rp.COL_LL])
                 + row[rp.COL_AA2]) + row[rp.COL_UC2])
        t = be @ _tz(-span)
    return t


def ujoint_centres(q, params=None) -> np.ndarray:
    """``(6, 3)`` u-joint centres u1..u6, robot frame, proximal to distal.

    Row ``2i`` is segment ``i+1``'s proximal centre, row ``2i+1`` its distal
    centre (the mocap plate with the same index sits at each centre —
    ``mocap_constants`` index map).  The distal centre is invariant to that
    segment's t3/t4: both axes pass through it, which the structure tests pin
    at 1e-14.  **q[10:12] therefore moves no centre at all** — its signal lives
    in plate 5's orientation only (see :func:`plate_transforms`).
    """
    q = _as_q(q)
    params = rp.as_params(params)
    out = np.empty((6, 3), dtype=float)
    for i, (row, b, be) in enumerate(_chain_frames(q, params)):
        uc1 = row[rp.COL_UC1]
        l_dist = row[rp.COL_AA1] + row[rp.COL_AA2] + row[rp.COL_LL] + uc1
        out[2 * i] = (b @ np.array([0.0, 0.0, -uc1, 1.0]))[0:3]
        out[2 * i + 1] = (be @ np.array([0.0, 0.0, -l_dist, 1.0]))[0:3]
    return out


def plate_transforms(q, params=None) -> np.ndarray:
    """``(6, 4, 4)`` body pose of plates 0..5, robot frame — the mounting model.

    Origins are the u-joint centres (plate i sits at centre i+1, exactly
    :func:`ujoint_centres` row for row); orientations follow the mounting
    sides (design D6, review finding geo-1):

    * plates 0, 2, 4 — rotation of the **proximal** body ``B_i``: invariant to
      their own u-joint's angles (the joint rotates *away* from them);
    * plates 1, 3, 5 — rotation of the **distal** body ``B_i @ E_i``: plate 5
      carries the full chain rotation **including q[10:12]**, which is how mk8
      reads u6 (``kinematics_mp.py:889-899``).  The ``mocap_constants`` phrase
      "z = last link" is rest-pose shorthand, not a spec: at q != 0 the plate z
      and the link z differ by exactly the u6 rotation being measured.

    Consequence worth pinning (and pinned, in the co-rigid tests): plates
    {1, 2} and {3, 4} live on the same rigid connector, and this model gives
    them **identity** relative rotation — ``B_{i+1} = T_i @ Tz(-JD)`` and
    ``T_i`` share their rotation with ``B_i @ E_i``.  The +45 deg bracket
    family difference is in the *marker bracket*, not in these body frames
    (marker-frame design §2).
    """
    q = _as_q(q)
    params = rp.as_params(params)
    # Origins come from the same chain walk as ujoint_centres, same expressions,
    # so the two functions can never disagree about where a plate sits.
    out = np.empty((6, 4, 4), dtype=float)
    for i, (row, b, be) in enumerate(_chain_frames(q, params)):
        uc1 = row[rp.COL_UC1]
        l_dist = row[rp.COL_AA1] + row[rp.COL_AA2] + row[rp.COL_LL] + uc1
        out[2 * i] = np.eye(4)
        out[2 * i, 0:3, 0:3] = b[0:3, 0:3]
        out[2 * i, 0:3, 3] = (b @ np.array([0.0, 0.0, -uc1, 1.0]))[0:3]
        out[2 * i + 1] = np.eye(4)
        out[2 * i + 1, 0:3, 0:3] = be[0:3, 0:3]
        out[2 * i + 1, 0:3, 3] = (be @ np.array([0.0, 0.0, -l_dist, 1.0]))[0:3]
    return out


def predict_spatial(base_se4, q, params=None, ee_lever_m=None) -> np.ndarray:
    """Predicted u-joint centres in the **mocap spatial frame**: ``(6, 3)``.

    ``p_spatial = SE4(body 500) @ p_robot`` — the correspondence the legacy
    stack uses (``kinematics_mp.py:760-774``): predicted u1..u6 line up with
    streamed bodies 500..505.  Body 500 is a degenerate test point (JD1 = UC1_1
    = 0 makes the prediction identically the base origin); the signal is bodies
    501-505.

    ``ee_lever_m``, if given, is a 3-vector **in the last-plate (plate 5) body
    frame** — the lever arm to an end-effector point (fkine benchmark §4 fits
    it from rest frames).  It appends a 7th row: ``base @ fkine(q) @ [lever,
    1]``, lining up with body 506 when that plate returns to the volume.
    """
    base_se4 = np.asarray(base_se4, dtype=float)
    if base_se4.shape != (4, 4):
        raise ValueError(f"base_se4 must have shape (4, 4); got {base_se4.shape}")
    centres = ujoint_centres(q, params)                       # validates q/params
    rot = base_se4[0:3, 0:3]
    pos = base_se4[0:3, 3]
    out_rows = 6 if ee_lever_m is None else 7
    out = np.empty((out_rows, 3), dtype=float)
    out[0:6] = centres @ rot.T + pos
    if ee_lever_m is not None:
        lever = np.asarray(ee_lever_m, dtype=float)
        if lever.shape != (3,):
            raise ValueError(f"ee_lever_m must have shape (3,); got {lever.shape}")
        ee = base_se4 @ fkine(q, params) @ np.array([*lever, 1.0])
        out[6] = ee[0:3]
    return out
