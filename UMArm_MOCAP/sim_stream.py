"""A synthetic Motive stream, fed straight into a real receiver in-process.

The idea is lifted from the legacy VNEMA client
(``VNEMA_MK8_PIDPWM/portable_sync_comm_layer/mocap/mocap.py``:
``simulated_body_poses`` / ``run_simulator``), which could exercise its whole
mocap path with no cameras.  What is different here is the injection point:
that client printed JSON records, whereas this one calls the **real** listeners
of a real :class:`~UMArm_MOCAP.mocap_rx.MocapRx`, so ``_on_rigid_body``,
``_on_new_frame``, the id routing, ``quat_xyzw_to_matrix``, ``mocap_to_q``, the
publish-under-lock and the ring buffer are all the shipped code.  No socket is
opened and no UDP is sent; ``start()`` is never called on the receiver.

WHAT IT PROVES AND WHAT IT DOES NOT.  It proves that a pose array shaped like
the CAN arm's converts, publishes and lands in the ring at the rate asked for,
which is enough to develop ``mocap_to_q`` consumers and the visualiser offline.
It cannot prove anything about the transport, the rigid-body ids Motive really
uses, or the marker labelling — all three need cameras.  In particular
:func:`plate_poses_from_q` inverts the reader's own steps with the reader's own
constants, so an error shared with the reader cancels exactly; it is a
consistency and plumbing fixture, not a convention oracle.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

# Package-relative only, for the reason given in ``canarm_mocap``: the marker
# stack underneath is.  Import this as ``UMArm_MOCAP.sim_stream``.
from . import mocap_constants as mc
from .canarm_mocap import CANARM_N_BODIES, CANARM_RB_ID_BASE, CanArmMocap
from .mocap_probe import matrix_to_quat_xyzw
from .mocap_to_q import rotation_aligning, unit_from_ujoint_angles

#: Placeholder link lengths, metres, proximal to distal — see
#: ``UMArm_KINEMATICS.canarm_params`` for the provenance discipline.  Only the
#: *shape* of the synthetic arm depends on these; ``q`` does not, which is the
#: point of the invariance the reader is built on.
DEFAULT_LINKS_M = (0.28, 0.06, 0.24, 0.06, 0.23, 0.06)

#: Frame rate the fake producer aims for, matching this lab's volume.
DEFAULT_RATE_HZ = 120.0


def _frame_with_z(z: np.ndarray) -> np.ndarray:
    """Any right-handed rotation whose third column is ``z``."""
    z = np.asarray(z, dtype=float) / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def plate_poses_from_q(q, links_m=DEFAULT_LINKS_M, rot_base=None,
                       base_pos=None) -> np.ndarray:
    """``(6, 4, 4)`` plate poses a perfect reader turns back into ``q``.

    Inverts :func:`mocap_to_q.mocap_to_q` step by step.  For joint ``k`` the
    reader forms ``v_k = [RZn45] @ align(rv_{k-1} -> +z) @ rv_k`` and reads the
    pair off ``v_k``; so given the wanted angles we take ``v_k =
    unit_from_ujoint_angles(...)`` and undo both rotations by transpose.  The
    centres then walk the chain as ``pos[k+1] = pos[k] - R_base @ (L_k rv_k)``,
    the minus sign being the reader's proximal-minus-distal convention (the arm
    hangs base-up).  Plate 5's *orientation* carries the sixth link direction,
    which has no centre beyond it to difference against.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (mc.NUM_JOINTS,):
        raise ValueError("q must be (%d,); got %s" % (mc.NUM_JOINTS, q.shape))
    links = np.asarray(links_m, dtype=float)
    rot_base = np.eye(3) if rot_base is None else np.asarray(rot_base, dtype=float)
    base_pos = np.zeros(3) if base_pos is None else np.asarray(base_pos, dtype=float)

    rv = np.empty((6, 3))
    for k in range(6):
        v = unit_from_ujoint_angles(q[2 * k], q[2 * k + 1])
        if k == 0:
            rv[0] = v
            continue
        if k % 2 == 1:                      # u2, u4, u6 sit on the 45 deg bracket
            v = mc.RZn45.T @ v
        rv[k] = rotation_aligning(rv[k - 1], mc.V_BASE).T @ v

    poses = np.tile(np.eye(4), (6, 1, 1)).astype(float)
    poses[mc.IDX_BASE, 0:3, 0:3] = rot_base
    pos = np.asarray(base_pos, dtype=float).copy()
    poses[0, 0:3, 3] = pos
    for k in range(5):                      # centres 1..5
        pos = pos - rot_base @ (links[k] * rv[k])
        poses[k + 1, 0:3, 3] = pos
    poses[mc.IDX_U3_DISTAL, 0:3, 0:3] = _frame_with_z(rot_base @ rv[5])
    return poses


def inject_frame(rx, poses, frame_number: int = -1) -> None:
    """Push one frame of plate poses through a receiver's real listeners.

    The fake-injection hook: works on any :class:`MocapRx` (or subclass) and
    respects its ``rb_id_base``, so a receiver bound to the wrong block sees
    nothing, exactly as it would live.
    """
    poses = np.asarray(poses, dtype=float)
    for i in range(min(poses.shape[0], rx.n_bodies)):
        rx._on_rigid_body(rx.rb_id_base + i, poses[i, 0:3, 3].copy(),
                          matrix_to_quat_xyzw(poses[i, 0:3, 0:3]))
    rx._on_new_frame({"frame_number": frame_number})


def sweep_q(t_s: float, amplitude_rad: float = 0.25) -> np.ndarray:
    """A slow, smooth, all-twelve-joints trajectory to feed the stream.

    Each joint gets its own frequency so no two ever sit at the same angle;
    a q that is accidentally symmetric hides an index swap.
    """
    return np.array([amplitude_rad * math.sin(0.30 * (j + 1) * t_s + 0.4 * j)
                     for j in range(mc.NUM_JOINTS)], dtype=float)


class CanArmSimStream(CanArmMocap):
    """A :class:`CanArmMocap` fed by a producer thread instead of by Motive.

    Overrides :meth:`start` / :meth:`stop` only.  Everything a consumer reads —
    ``get_q``, ``get_state``, ``wait_fresh``, ``get_homos``, the rings,
    ``capture_rest`` — is the inherited implementation running on genuinely
    published frames, so code developed against this and code run against the
    cameras differ in what produced the poses and in nothing else.
    """

    def __init__(self, *, q_of_t=sweep_q, rate_hz: float = DEFAULT_RATE_HZ,
                 links_m=DEFAULT_LINKS_M, rot_base=None, base_pos=None,
                 rb_id_base: int = CANARM_RB_ID_BASE,
                 n_bodies: int = CANARM_N_BODIES, **kwargs) -> None:
        super().__init__(rb_id_base=rb_id_base, n_bodies=n_bodies, **kwargs)
        self.q_of_t = q_of_t
        self.rate_hz = float(rate_hz)
        self.links_m = links_m
        self.rot_base = rot_base
        self.base_pos = base_pos
        self._sim_stop = threading.Event()
        self._sim_thread: threading.Thread | None = None

    def start(self):
        """Begin producing frames.  Opens nothing; returns self, not a client."""
        if self._sim_thread is not None:
            return self
        self._sim_stop.clear()
        # Daemon on purpose, unlike the SDK's own threads: a fixture that
        # outlives its test must not hold the interpreter open the way a
        # forgotten real receiver was measured to (EXIT=124, 15 s timeout).
        self._sim_thread = threading.Thread(target=self._produce, daemon=True,
                                            name="canarm-sim-stream")
        self._sim_thread.start()
        return self

    def stop(self) -> None:
        """Stop producing.  Safe to call twice, and safe if start never ran."""
        self._sim_stop.set()
        thread, self._sim_thread = self._sim_thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _produce(self) -> None:
        period = 1.0 / self.rate_hz
        t0 = time.monotonic()
        next_tick = t0
        frame = 0
        while not self._sim_stop.is_set():
            poses = plate_poses_from_q(self.q_of_t(time.monotonic() - t0),
                                       self.links_m, self.rot_base,
                                       self.base_pos)
            inject_frame(self, poses, frame_number=frame)
            frame += 1
            next_tick += period
            self._sim_stop.wait(max(0.0, next_tick - time.monotonic()))
