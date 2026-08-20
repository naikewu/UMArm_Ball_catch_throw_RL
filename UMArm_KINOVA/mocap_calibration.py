"""Solving the two transforms that tie the Gen3 to the mocap volume.

THE PROBLEM, in one line.  The cameras report a rigid body glued to the arm's
tool; the arm reports where it thinks its tool is.  Neither knows the other's
frame.  Two constant transforms close the loop::

    T_world_rb(i)  =  X . T_base_tool(i) . Y

* ``X = T_world_base`` — where the Gen3's base sits in the mocap volume.  It is
  a property of **where the cart is parked** and dies the moment the cart moves.
* ``Y = T_tool_rb`` — where Motive's rigid-body frame sits on the arm's tool.
  It is a property of **how the markers are glued on** and survives cart moves,
  power cycles and everything short of knocking a marker.

That split is the whole point of fitting them together.  ``Y`` is the reusable
half: once it is known, ONE mocap frame plus the arm's own forward kinematics
puts the base back in the volume, which is what
:class:`UMArm_COLLAB.base_poses.KinovaBaseFromEE` already does structurally and
what this module supplies the numbers for.

HOW IT IS SOLVED.  This is the robot-world / hand-eye problem in its ``AX = ZB``
form, and it is solved in three stages so that each stage's failure is legible:

1. **Y's rotation, from relative motions** (Park and Martin, 1994).  Between two
   samples the unknown ``X`` cancels::

       Rn_i' Rn_j  =  Ry' (Rm_i' Rm_j) Ry

   so the axis of every relative *sensor* rotation is ``Ry'`` times the axis of
   the matching relative *robot* rotation.  Stacking the rotation-vector pairs
   and taking ``R = (M'M)^-1/2 M'`` recovers ``Ry`` in closed form.
   **This stage needs real rotation.**  A campaign of pure translations leaves
   every relative rotation at the identity, ``M`` collapses to zero and ``Ry``
   is unobservable; :func:`observability` says so in the report rather than
   letting a random rotation through.
2. **X's rotation**, then, is ``Rn_i (Rm_i Ry)'`` averaged over the samples —
   averaged in the SVD sense, not element-wise, so the result is a rotation.
3. **Both translations, linearly.**  With the rotations fixed, the model is
   linear in ``(py, px)``::

       pn_i - Rx.pm_i  =  (Rx.Rm_i) py  +  px

   three rows per sample, six unknowns.  Separating ``py`` (a lever arm that
   only shows itself when the tool turns) from ``px`` (a constant offset) again
   needs orientation variety, and the design matrix's condition number is
   reported for exactly that reason.

A final :func:`scipy.optimize.least_squares` polish over all twelve parameters
removes the bias every closed form carries from splitting rotation off
translation.  The polish is a refinement, never the whole solve: started from
the closed form it converges in a few iterations, and both answers are reported
so a large gap between them is visible instead of averaged away.

WHAT THE RESIDUAL MEANS, and what it does not.  After the fit, the leftover
distance between the measured and predicted rigid-body origin lumps together
the arm's forward-kinematics error, the mocap's own solve error, any flex in
the printed bracket the markers stand on, and any non-rigidity in the glue.  It
is an UPPER BOUND on the arm's forward-kinematics error, never a measurement of
it alone.  Splitting the terms apart needs a second, independent measurement of
the tool pose, which this rig does not have.

Nothing in this module touches hardware; ``test_mocap_calibration.py`` exercises
all of it on synthesised poses with known answers.
"""

from __future__ import annotations

import numpy as np

#: Below this, a relative rotation is treated as pure noise and dropped from the
#: hand-eye stage.  Motive's own orientation jitter on a four-marker body is a
#: few tenths of a degree, and a rotation vector of that size carries no usable
#: axis; 2 deg is an order of magnitude clear of it while still admitting every
#: deliberate wrist move a 20 cm campaign can make.
MIN_RELATIVE_ROT_DEG = 2.0

#: Below this axis spread (degrees, see :func:`axis_spread_deg`) the relative
#: rotations all lie along one line and ``Ry`` is determined only up to a turn
#: about it.  20 deg is well clear of the noise on a set of deliberately
#: different wrist axes, and well under the 90 deg two orthogonal ones give.
MIN_AXIS_SPREAD_DEG = 20.0

#: A per-axis scale difference smaller than this is not called significant
#: however small the statistical uncertainty happens to be.  It is a floor on
#: MEANING rather than on confidence: 100 ppm over the 140 mm this campaign
#: spans is 14 micrometres, an order of magnitude below the fit's own 0.19 mm
#: residual, so a difference that size changes nothing anybody would do.  It
#: also keeps the test honest on noiseless synthetic data, where the
#: uncertainty is zero and any spread at all would otherwise clear 2 sigma.
SCALE_SIGNIFICANCE_FLOOR_PPM = 100.0

#: Rotation residuals are folded into the least-squares polish through a length,
#: so that a radian of orientation error costs the same as this many metres of
#: position error.  0.1 m is the scale of the lever arm between the tool and the
#: markers: at that distance the two error kinds have comparable physical
#: meaning, which is what makes the weighting a choice rather than an arbitrary
#: number.
ROT_WEIGHT_M_PER_RAD = 0.10


# --------------------------------------------------------------------------- #
# Small SE(3) helpers.  Kept local rather than imported from the driver so this
# module can be tested with no kortex_api installed at all.
# --------------------------------------------------------------------------- #
def rot_log(R) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle, radians).

    Routed through a quaternion (Shepperd's branch on the largest of the four
    candidate divisors) rather than through ``arccos(trace)``.  The direct
    formula loses the axis entirely as the angle approaches pi — the skew part
    of ``R`` vanishes there — and hand-eye residuals are read in millidegrees,
    so an axis that degrades near a half turn is not good enough.  The
    quaternion branch has no such point.
    """
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        d = np.array([R[0, 0], R[1, 1], R[2, 2]])
        k = int(np.argmax(d))
        i, j, l = k, (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[l, l]) * 2.0
        q = np.zeros(4)
        q[0] = (R[l, j] - R[j, l]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + l] = (R[l, i] + R[i, l]) / s
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:                       # shortest-arc branch: |theta| <= pi
        q = -q
    v = q[1:4]
    nv = float(np.linalg.norm(v))
    if nv < 1e-15:
        return np.zeros(3)
    return v * (2.0 * float(np.arctan2(nv, q[0])) / nv)


def rot_exp(w) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def project_rotation(M) -> np.ndarray:
    """Nearest rotation matrix to *M* (SVD, ``det = +1`` enforced)."""
    u, _, vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vt
    return R


def se3(R, p) -> np.ndarray:
    T = np.eye(4)
    T[0:3, 0:3] = np.asarray(R, dtype=float)
    T[0:3, 3] = np.asarray(p, dtype=float).ravel()
    return T


def se3_inv(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[0:3, 0:3]
    out = np.eye(4)
    out[0:3, 0:3] = R.T
    out[0:3, 3] = -R.T @ T[0:3, 3]
    return out


def angle_between_deg(Ra, Rb) -> float:
    """Geodesic angle between two rotations, degrees."""
    return float(np.degrees(np.linalg.norm(
        rot_log(np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)))))


# --------------------------------------------------------------------------- #
# Stage 1-3: the closed form
# --------------------------------------------------------------------------- #
def axis_spread_deg(vectors) -> float:
    """How far a set of rotation vectors is from sharing ONE axis, degrees.

    Rotation axes are LINES, not arrows: turning +12 deg and -12 deg about the
    same wrist joint gives opposed rotation vectors and identical information.
    Comparing signed directions would call that pair 180 deg apart and pronounce
    a single-axis campaign richly determined, which is the exact opposite of the
    truth.  So the comparison runs on ``|cos|``, and the result saturates at
    90 deg for a genuinely two-axis set.
    """
    V = np.asarray(vectors, dtype=float).reshape(-1, 3)
    if V.shape[0] < 2:
        return 0.0
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip(np.abs(U @ U.T).min(), 0.0, 1.0))))


def hand_eye_rotation(M, N, min_rot_deg: float = MIN_RELATIVE_ROT_DEG):
    """``Ry`` from every usable pair of samples (Park and Martin).

    *M* and *N* are ``(n, 4, 4)`` stacks of ``T_base_tool`` and ``T_world_rb``.
    Returns ``(Ry, info)``; ``info`` carries the pairs used and the axis spread,
    which is what tells a reader whether the answer was determined or merely
    produced.
    """
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    n = M.shape[0]
    alphas, betas = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a = rot_log(M[i, 0:3, 0:3].T @ M[j, 0:3, 0:3])
            b = rot_log(N[i, 0:3, 0:3].T @ N[j, 0:3, 0:3])
            if np.degrees(np.linalg.norm(a)) < min_rot_deg:
                continue
            alphas.append(a)
            betas.append(b)
    info = {"pairs_total": n * (n - 1) // 2, "pairs_used": len(alphas),
            "min_rot_deg": float(min_rot_deg),
            "min_axis_spread_deg": float(MIN_AXIS_SPREAD_DEG),
            "axis_spread_deg": 0.0}
    if len(alphas) < 2:
        info["refusal"] = (
            "fewer than two relative rotations above %.1f deg — Ry's rotation "
            "is not observable from this campaign; it needs poses that turn "
            "the tool about at least two different axes" % min_rot_deg)
        return np.eye(3), info

    A = np.array(alphas)           # (k, 3) robot-side rotation vectors
    B = np.array(betas)            # (k, 3) sensor-side rotation vectors
    info["axis_spread_deg"] = axis_spread_deg(A)
    # COUNTING the rotations is not the test.  Two turns of the same wrist joint
    # are two rotations and ONE axis, and one axis leaves Ry free to turn about
    # it: the cross-covariance is rank-deficient and the closed form returns
    # whichever member of that one-parameter family the decomposition lands on,
    # with no hint that it did.  The refusal below used to promise "at least two
    # different axes" while testing only the count, which is the kind of message
    # that gets believed.
    if info["axis_spread_deg"] < MIN_AXIS_SPREAD_DEG:
        info["refusal"] = (
            "every relative rotation lies within %.1f deg of one axis (need "
            "%.1f) - Ry is determined only up to a turn about that axis, so any "
            "value returned is one arbitrary member of a one-parameter family"
            % (info["axis_spread_deg"], MIN_AXIS_SPREAD_DEG))
        return np.eye(3), info

    Mx = np.zeros((3, 3))
    for a, b in zip(A, B):
        Mx += np.outer(b, a)
    # R = (M'M)^-1/2 M'  — the closed form; the eigendecomposition is of a
    # 3x3 SPD matrix, so the inverse square root is exact rather than iterative.
    w, V = np.linalg.eigh(Mx.T @ Mx)
    w = np.clip(w, 1e-18, None)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    Ry = project_rotation(inv_sqrt @ Mx.T)
    info["axis_residual_deg"] = float(np.degrees(np.mean(
        [np.linalg.norm(a - Ry @ b) / max(np.linalg.norm(a), 1e-12)
         for a, b in zip(A, B)])))
    return Ry, info


def _average_rotation(rots) -> np.ndarray:
    """Chordal mean of a list of rotations: project their arithmetic mean."""
    return project_rotation(np.mean(np.asarray(rots, dtype=float), axis=0))


def solve_translations(M, N, Rx, Ry):
    """``(px, py, cond)`` by linear least squares with the rotations fixed."""
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    rows, rhs = [], []
    for i in range(M.shape[0]):
        Rm, pm = M[i, 0:3, 0:3], M[i, 0:3, 3]
        pn = N[i, 0:3, 3]
        rows.append(np.hstack([Rx @ Rm, np.eye(3)]))
        rhs.append(pn - Rx @ pm)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[3:6], sol[0:3], float(np.linalg.cond(A))


def closed_form_fit(M, N, min_rot_deg: float = MIN_RELATIVE_ROT_DEG):
    """``(X, Y, info)`` from the three closed-form stages."""
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    Ry, rot_info = hand_eye_rotation(M, N, min_rot_deg)
    Rx = _average_rotation([N[i, 0:3, 0:3] @ (M[i, 0:3, 0:3] @ Ry).T
                            for i in range(M.shape[0])])
    # How far the per-sample Rx estimates disagree.  On a rigid rig this is the
    # combined orientation noise of the two systems.  An error E in Ry maps each
    # estimate to Rx . Rm_i (Ry E Ry') Rm_i', so it appears here as a SPREAD only
    # because Rm_i varies across the campaign; samples that all shared one tool
    # orientation would turn the same error into a pure bias this statistic
    # cannot see.  Read it beside axis_spread_deg, which says whether Rm_i
    # varied at all.
    spread = [angle_between_deg(Rx, N[i, 0:3, 0:3] @ (M[i, 0:3, 0:3] @ Ry).T)
              for i in range(M.shape[0])]
    px, py, cond = solve_translations(M, N, Rx, Ry)
    info = dict(rot_info)
    info["rx_spread_deg_max"] = float(np.max(spread))
    info["rx_spread_deg_rms"] = float(np.sqrt(np.mean(np.square(spread))))
    info["translation_cond"] = cond
    return se3(Rx, px), se3(Ry, py), info


# --------------------------------------------------------------------------- #
# The polish, and what to make of the answer
# --------------------------------------------------------------------------- #
def _pack(X, Y) -> np.ndarray:
    return np.concatenate([rot_log(X[0:3, 0:3]), X[0:3, 3],
                           rot_log(Y[0:3, 0:3]), Y[0:3, 3]])


def _unpack(v):
    v = np.asarray(v, dtype=float)
    return se3(rot_exp(v[0:3]), v[3:6]), se3(rot_exp(v[6:9]), v[9:12])


def residuals(X, Y, M, N) -> dict:
    """Per-sample position and orientation error of ``N ~= X M Y``.

    Positions come back in **millimetres** and angles in **degrees**, because
    that is the unit every consumer of this report thinks in and a metre-scale
    residual printed as ``0.0007`` reads as zero when it is 0.7 mm.
    """
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    pos, ang = [], []
    for i in range(M.shape[0]):
        P = X @ M[i] @ Y
        pos.append(np.linalg.norm(N[i, 0:3, 3] - P[0:3, 3]) * 1e3)
        ang.append(angle_between_deg(N[i, 0:3, 0:3], P[0:3, 0:3]))
    pos = np.array(pos)
    ang = np.array(ang)
    return {
        "pos_mm": pos.tolist(), "ang_deg": ang.tolist(),
        "pos_rms_mm": float(np.sqrt(np.mean(pos ** 2))),
        "pos_max_mm": float(pos.max()),
        "ang_rms_deg": float(np.sqrt(np.mean(ang ** 2))),
        "ang_max_deg": float(ang.max()),
    }


def refine(X, Y, M, N, rot_weight_m: float = ROT_WEIGHT_M_PER_RAD):
    """Least-squares polish of all twelve parameters.  Returns ``(X, Y, info)``."""
    from scipy.optimize import least_squares

    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)

    def resid(v):
        Xv, Yv = _unpack(v)
        out = []
        for i in range(M.shape[0]):
            P = Xv @ M[i] @ Yv
            out.append(N[i, 0:3, 3] - P[0:3, 3])
            out.append(rot_log(P[0:3, 0:3].T @ N[i, 0:3, 0:3]) * rot_weight_m)
        return np.concatenate(out)

    sol = least_squares(resid, _pack(X, Y), method="lm", xtol=1e-14,
                        ftol=1e-14, gtol=1e-14)
    Xr, Yr = _unpack(sol.x)
    return Xr, Yr, {"nfev": int(sol.nfev), "cost": float(sol.cost),
                    "success": bool(sol.success),
                    "rot_weight_m_per_rad": float(rot_weight_m)}


def observability(M, N, min_rot_deg: float = MIN_RELATIVE_ROT_DEG) -> dict:
    """What this set of samples can and cannot determine.

    Read this BEFORE the residual.  A campaign of pure translations fits with a
    beautiful residual and an arbitrary ``Ry`` rotation, because nothing in the
    data ever asked about it; the residual cannot tell you that and this can.
    """
    M = np.asarray(M, dtype=float)
    rots = [rot_log(M[i, 0:3, 0:3].T @ M[j, 0:3, 0:3])
            for i in range(M.shape[0]) for j in range(i + 1, M.shape[0])]
    mags = np.degrees([np.linalg.norm(r) for r in rots]) if rots else np.zeros(1)
    pos = M[:, 0:3, 3]
    # The RADIUS of the tool-origin cloud about its own centroid, which is half
    # the span of a set that straddles its centre.  Named for what it is: the
    # note below reads it as leverage, and leverage is a radius.
    radius = float(np.max(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))
    used = [r for r in rots if np.degrees(np.linalg.norm(r)) >= min_rot_deg]
    spread = axis_spread_deg(used) if len(used) >= 2 else 0.0
    notes = []
    if mags.max() < min_rot_deg:
        notes.append("no relative rotation above %.1f deg: Ry's ROTATION is "
                     "unobservable and the lever arm py is unobservable too — "
                     "only X's rotation and the sum px + Rx.Rm.py are fitted"
                     % min_rot_deg)
    elif spread < 20.0:
        notes.append("every rotation shares one axis to within %.1f deg: Ry is "
                     "determined only up to a turn about that axis" % spread)
    if radius < 0.02:
        notes.append("the tool origin stayed within 20 mm of its own centroid: "
                     "X's translation rests on very little leverage")
    return {"n": int(M.shape[0]),
            "max_relative_rot_deg": float(mags.max()),
            "rotation_axis_spread_deg": spread,
            "tool_radius_m": radius,
            "notes": notes,
            "ok": not notes}


def fit(M, N, min_rot_deg: float = MIN_RELATIVE_ROT_DEG) -> dict:
    """The whole solve: closed form, polish, residuals and observability.

    Returns a dict with ``X`` (``T_world_base``), ``Y`` (``T_tool_rb``) as
    ``(4, 4)`` arrays, both stages' residuals, and everything needed to judge
    the answer without re-running it.
    """
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    if M.shape != N.shape or M.ndim != 3 or M.shape[1:] != (4, 4):
        raise ValueError("M and N must be matching (n, 4, 4) pose stacks, got "
                         "%s and %s" % (M.shape, N.shape))
    if M.shape[0] < 3:
        raise ValueError("at least three samples are needed; got %d"
                         % M.shape[0])
    X0, Y0, info = closed_form_fit(M, N, min_rot_deg)
    r0 = residuals(X0, Y0, M, N)
    X, Y, pinfo = refine(X0, Y0, M, N)
    r1 = residuals(X, Y, M, N)
    return {
        "X_world_base": X, "Y_tool_rb": Y,
        "X_closed_form": X0, "Y_closed_form": Y0,
        "closed_form_info": info, "refine_info": pinfo,
        "residual_closed_form": r0, "residual": r1,
        "observability": observability(M, N, min_rot_deg),
        "improvement_mm": r0["pos_rms_mm"] - r1["pos_rms_mm"],
    }


# --------------------------------------------------------------------------- #
# The operator's question, answered directly
# --------------------------------------------------------------------------- #
def axis_fidelity(M, N, Rx, pairs=None, min_move_m: float = 0.005) -> dict:
    """"Command +x — does the mocap move +x, by the same amount?"

    For every pair of samples whose tool ORIENTATION is the same, the model
    collapses to ``pn_j - pn_i = Rx (pm_j - pm_i)`` exactly: the lever arm and
    both origins cancel, leaving nothing but the base rotation.  So each such
    pair is a direct, assumption-free comparison of a commanded displacement
    with a measured one — a **scale** (measured length over commanded length,
    1.000 if the two systems agree on distance) and an **angle** (how far the
    measured direction sits from the predicted one).

    Pairs whose orientation differs are skipped rather than corrected: including
    them would fold the fitted lever arm back into a test whose whole value is
    that it does not depend on it.
    """
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    n = M.shape[0]
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    out = []
    for i, j in pairs:
        if angle_between_deg(M[i, 0:3, 0:3], M[j, 0:3, 0:3]) > 0.5:
            continue
        d_cmd = M[j, 0:3, 3] - M[i, 0:3, 3]
        if np.linalg.norm(d_cmd) < min_move_m:
            continue
        d_meas = N[j, 0:3, 3] - N[i, 0:3, 3]
        if np.linalg.norm(d_meas) < 1e-9:
            # The arm moved and the cameras did not: a frozen or dropped rigid
            # body.  Skipping is the honest answer - the cosine would be 0/0,
            # and ``max(-1, min(1, nan))`` returns +1.0 in Python, so a dead
            # marker would have been recorded as a PERFECT direction match.
            continue
        pred = Rx @ d_cmd
        cos = float(np.dot(pred, d_meas)
                    / (np.linalg.norm(pred) * np.linalg.norm(d_meas)))
        out.append({
            "i": int(i), "j": int(j),
            "commanded_mm": (np.linalg.norm(d_cmd) * 1e3),
            "measured_mm": (np.linalg.norm(d_meas) * 1e3),
            "scale": float(np.linalg.norm(d_meas) / np.linalg.norm(d_cmd)),
            "angle_deg": float(np.degrees(np.arccos(max(-1.0, min(1.0, cos))))),
            "error_mm": float(np.linalg.norm(d_meas - pred) * 1e3),
        })
    if not out:
        return {"pairs": [], "note": "no same-orientation pair moved far enough"}
    scale = np.array([r["scale"] for r in out])
    ang = np.array([r["angle_deg"] for r in out])
    err = np.array([r["error_mm"] for r in out])
    return {
        "pairs": out,
        "n_pairs": len(out),
        "scale_mean": float(scale.mean()),
        "scale_sd": float(scale.std()),
        "scale_ppm_from_unity": float((scale.mean() - 1.0) * 1e6),
        "angle_mean_deg": float(ang.mean()),
        "angle_max_deg": float(ang.max()),
        "error_rms_mm": float(np.sqrt(np.mean(err ** 2))),
        "error_max_mm": float(err.max()),
    }


def axis_fidelity_by_axis(M, N, Rx, min_move_m: float = 0.005) -> dict:
    """:func:`axis_fidelity`, split by which base axis the move was along.

    The split is what turns "the two systems disagree about distance by 0.16 %"
    into a diagnosis.  A single number common to x, y and z is a SCALE — one
    ruler is longer than the other, which on this rig means the mocap volume's
    wand calibration or a uniform error in the arm's link lengths.  Three
    different numbers are not a scale at all; they are a per-axis distortion,
    which points at the cameras' geometry rather than at either ruler.

    Only pairs whose commanded displacement lies within 15 deg of a base axis
    are counted, so a diagonal move is not attributed to whichever axis happens
    to dominate it.
    """
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    n = M.shape[0]
    out = {}
    for k, name in enumerate("xyz"):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if angle_between_deg(M[i, 0:3, 0:3], M[j, 0:3, 0:3]) > 0.5:
                    continue
                d = M[j, 0:3, 3] - M[i, 0:3, 3]
                mag = float(np.linalg.norm(d))
                if mag < min_move_m or abs(d[k]) / mag < np.cos(np.deg2rad(15)):
                    continue
                pairs.append((i, j))
        if not pairs:
            out[name] = {"n_pairs": 0}
            continue
        res = axis_fidelity(M, N, Rx, pairs=pairs, min_move_m=min_move_m)
        out[name] = {"n_pairs": res["n_pairs"], "scale_mean": res["scale_mean"],
                     "scale_sd": res["scale_sd"],
                     "scale_ppm_from_unity": res["scale_ppm_from_unity"],
                     "angle_mean_deg": res["angle_mean_deg"],
                     "error_rms_mm": res["error_rms_mm"]}
    have = [v["scale_ppm_from_unity"] for v in out.values() if v["n_pairs"]]
    out["spread_ppm"] = float(max(have) - min(have)) if len(have) > 1 else 0.0
    out["mean_ppm"] = float(np.mean(have)) if have else float("nan")
    # The per-axis means come from a handful of OVERLAPPING pairs - five pairs
    # along one axis are built from at most four distinct poses - so they are
    # correlated and their spread is not a clean statistic.  This is the scale
    # the spread has to beat before "the axes differ" means anything: each
    # axis's own pair-to-pair sd over the square root of the number of DISTINCT
    # POSES behind it, not of the number of pairs.
    noise = []
    for name in "xyz":
        v = out[name]
        if not v["n_pairs"]:
            continue
        poses = 0.5 * (1.0 + np.sqrt(1.0 + 8.0 * v["n_pairs"]))   # pairs -> n
        noise.append(v["scale_sd"] * 1e6 / max(np.sqrt(poses), 1.0))
    out["per_axis_uncertainty_ppm"] = float(np.mean(noise)) if noise else 0.0
    out["significance_threshold_ppm"] = float(max(
        2.0 * out["per_axis_uncertainty_ppm"], SCALE_SIGNIFICANCE_FLOOR_PPM))
    out["spread_is_significant"] = bool(
        noise and out["spread_ppm"] > out["significance_threshold_ppm"])
    return out


def fit_with_scale(M, N, min_rot_deg: float = MIN_RELATIVE_ROT_DEG) -> dict:
    """The same solve with ONE extra parameter: a scale on the arm's ruler.

    The model becomes ``pn = Rx (s . pm + Rm . py) + px``.  ``s`` multiplies the
    arm's reported tool POSITION and not the lever arm, because that is what a
    uniform link-length error or a mocap wand-length error looks like from here:
    the two systems agree about direction and disagree about how long a metre is.

    Reported next to the twelve-parameter answer rather than instead of it.  A
    thirteenth parameter can only lower a residual, so the number that matters
    is how MUCH it lowers it: a scale that buys nothing is a scale that was not
    there, whatever value the optimiser settles on.
    """
    from scipy.optimize import least_squares

    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    base = fit(M, N, min_rot_deg)
    v0 = np.concatenate([_pack(base["X_world_base"], base["Y_tool_rb"]), [1.0]])

    def resid(v):
        Xv, Yv = _unpack(v[0:12])
        s = float(v[12])
        out = []
        for i in range(M.shape[0]):
            Rm, pm = M[i, 0:3, 0:3], M[i, 0:3, 3]
            pred = Xv[0:3, 0:3] @ (s * pm + Rm @ Yv[0:3, 3]) + Xv[0:3, 3]
            out.append(N[i, 0:3, 3] - pred)
            Rp = Xv[0:3, 0:3] @ Rm @ Yv[0:3, 0:3]
            out.append(rot_log(Rp.T @ N[i, 0:3, 0:3]) * ROT_WEIGHT_M_PER_RAD)
        return np.concatenate(out)

    sol = least_squares(resid, v0, method="lm", xtol=1e-14, ftol=1e-14)
    X, Y = _unpack(sol.x[0:12])
    s = float(sol.x[12])
    pos = []
    for i in range(M.shape[0]):
        pred = X[0:3, 0:3] @ (s * M[i, 0:3, 3] + M[i, 0:3, 0:3] @ Y[0:3, 3]) \
            + X[0:3, 3]
        pos.append(np.linalg.norm(N[i, 0:3, 3] - pred) * 1e3)
    pos = np.array(pos)
    return {
        "scale": s,
        "scale_ppm_from_unity": (s - 1.0) * 1e6,
        "X_world_base": X, "Y_tool_rb": Y,
        "pos_rms_mm": float(np.sqrt(np.mean(pos ** 2))),
        "pos_max_mm": float(pos.max()),
        "pos_rms_mm_without_scale": base["residual"]["pos_rms_mm"],
        "improvement_mm": base["residual"]["pos_rms_mm"]
        - float(np.sqrt(np.mean(pos ** 2))),
    }


def base_from_single_sample(M_i, N_i, Y) -> np.ndarray:
    """``T_world_base`` back-solved from ONE tool pose and ONE mocap frame.

    ``X = N Y^-1 M^-1``.  This is the operation that decides whether the cart
    needs its own marker body: if the ``X`` this returns from any single frame
    agrees with the campaign's fitted ``X``, then re-finding the base after the
    cart has been dragged is one still frame of the arm, not a re-calibration.
    """
    return np.asarray(N_i, dtype=float) @ se3_inv(Y) @ se3_inv(M_i)


def base_recovery_spread(M, N, Y, X_ref) -> dict:
    """How much the single-frame base solve wanders across the campaign.

    Reported as the position and orientation spread of :func:`base_from_single_sample`
    over every sample, against the jointly fitted ``X_ref``.

    **IN-SAMPLE, and that is a real caveat.**  The ``Y`` used to back-solve each
    frame was fitted from those same frames, so this understates what the
    calibration would do on a pose it has never seen.  The out-of-sample twin is
    in ``calibrate_mocap._leave_one_out``, which refits WITHOUT each sample
    before back-solving it; on this rig the two agree closely, because the
    scatter is dominated by the orientation residual levered over the reach
    rather than by how well ``Y`` fits.  Quote the out-of-sample number when
    deciding whether to glue a marker set onto the base.
    """
    M = np.asarray(M, dtype=float)
    N = np.asarray(N, dtype=float)
    pos, ang, mats = [], [], []
    for i in range(M.shape[0]):
        Xi = base_from_single_sample(M[i], N[i], Y)
        mats.append(Xi)
        pos.append(np.linalg.norm(Xi[0:3, 3] - X_ref[0:3, 3]) * 1e3)
        ang.append(angle_between_deg(X_ref[0:3, 0:3], Xi[0:3, 0:3]))
    pos = np.array(pos)
    ang = np.array(ang)
    centres = np.array([m[0:3, 3] for m in mats])
    return {
        "pos_rms_mm": float(np.sqrt(np.mean(pos ** 2))),
        "pos_max_mm": float(pos.max()),
        "ang_rms_deg": float(np.sqrt(np.mean(ang ** 2))),
        "ang_max_deg": float(ang.max()),
        "centre_sd_mm": (centres.std(axis=0) * 1e3).tolist(),
        "centre_ptp_mm": (np.ptp(centres, axis=0) * 1e3).tolist(),
    }


__all__ = [
    "MIN_RELATIVE_ROT_DEG", "ROT_WEIGHT_M_PER_RAD",
    "rot_log", "rot_exp", "project_rotation", "se3", "se3_inv",
    "MIN_AXIS_SPREAD_DEG", "SCALE_SIGNIFICANCE_FLOOR_PPM",
    "angle_between_deg", "axis_spread_deg", "hand_eye_rotation",
    "solve_translations",
    "closed_form_fit", "residuals", "refine", "observability", "fit",
    "axis_fidelity", "axis_fidelity_by_axis", "fit_with_scale",
    "base_from_single_sample", "base_recovery_spread",
]
