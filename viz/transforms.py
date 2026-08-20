"""Rotation conversions, written out rather than imported from scipy.

PROVENANCE: ported verbatim in behaviour from
``C:\\RUNZE_SRC\\RS485_VEMA\\UMArm_COLLAB\\mount_transforms.py`` lines 708-762
(``rpy_to_quat``, ``quat_to_mat``, ``mat_to_quat``) and from the Shepperd branch
in ``C:\\RUNZE_SRC\\RS485_VEMA\\UMArm_CONTROL\\gui\\viewer.py:54`` (``_mat_to_wxyz``),
which are the same algorithm written twice in that repo.  They live in their own
module here for the reason the original states: the MJCF generator must have no
optional dependency, because the scene has to build on a clone with nothing but
numpy and mujoco installed.

Quaternions are MuJoCo's **scalar-first** ``[w, x, y, z]`` throughout.  That is
not the mocap convention — NatNet streams scalar-*last* ``[x, y, z, w]``, and
``UMArm_MOCAP.mocap_to_q.quat_xyzw_to_matrix`` is where that one is read — so any
value crossing between the two worlds goes through a matrix, never through a
reordering by hand.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rpy_to_quat", "quat_to_mat", "mat_to_quat", "pose_matrix",
           "IDENTITY_QUAT"]

#: The rotation that is no rotation, in MuJoCo's ordering.
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def rpy_to_quat(rpy_deg) -> np.ndarray:
    """Extrinsic-xyz Euler degrees -> MuJoCo scalar-first quaternion."""
    r, p, y = (np.deg2rad(float(v)) for v in rpy_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)


def quat_to_mat(q) -> np.ndarray:
    """MuJoCo scalar-first quaternion -> ``(3, 3)`` rotation matrix.

    A quaternion whose norm has collapsed returns the identity rather than a
    division by something vanishing: Motive publishes ``(0, 0, 0, 0)`` for a
    body it cannot solve, and that value reaches listeners before the
    tracking-valid bit is parsed.
    """
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=float)


def mat_to_quat(m) -> np.ndarray:
    """``(3, 3)`` rotation matrix -> MuJoCo scalar-first quaternion (Shepperd).

    Shepperd's branch method: each branch divides by the largest of the four
    candidate magnitudes, so none of them divides by something vanishing.
    """
    m = np.asarray(m, dtype=float)
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array(IDENTITY_QUAT, dtype=float)


def pose_matrix(pos, quat) -> np.ndarray:
    """``(pos, scalar-first quat)`` -> a ``(4, 4)`` homogeneous transform."""
    out = np.eye(4)
    out[0:3, 0:3] = quat_to_mat(quat)
    out[0:3, 3] = np.asarray(pos, dtype=float)
    return out
