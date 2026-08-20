"""Tests for the CAN-arm receivers, the per-instance id block, and sim_stream.

No sockets, no cameras, no arm: every frame here is synthesised and pushed
through the shipped listeners in-process.  Four claims are checked, and they
fail for different reasons:

1. **The id block is per-instance.**  Two receivers on different bases, fed one
   interleaved stream, each see their own arm and nothing of the other's.  This
   is the one behaviour the RS485 code could not express, and the reason for
   the refactor.
2. **The defaults did not move.**  A bare :class:`MocapRx` still claims
   ``500..507`` and still routes the Kinova's 1008 to row 8, so nothing that
   worked before the refactor reads differently after it.
3. **The synthetic round trip closes.**  A known ``q`` becomes six plate poses
   and ``mocap_to_q`` recovers it.  Note what this does *not* prove: the
   generator inverts the reader's own steps with the reader's own constants, so
   an error shared with both cancels exactly.  It pins plumbing and invariance,
   not convention; ``test_mocap_to_q.TestConventionsPinned`` owns the latter.
4. **The stream feeds a receiver end to end.**  ``CanArmSimStream`` publishes
   frames a consumer can read through ``get_q``/``get_state``/``wait_fresh``
   without any of them knowing the poses were invented.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Package-relative, matching test_marker_mocap.py: the modules under test reach
# the marker stack, which is package-relative only.
from . import mocap_constants as mc
from .canarm_mocap import (CANARM_N_BODIES, CANARM_RB_ID_BASE,
                           DEFAULT_TEMPLATE_PATH, CanArmMocap,
                           load_canarm_locks)
from .mocap_probe import matrix_to_quat_xyzw
from .mocap_rx import MocapRx
from .mocap_to_q import mocap_to_q
from .sim_stream import (CanArmSimStream, inject_frame, plate_poses_from_q,
                         sweep_q)

Q_A = np.array([0.10, -0.05, 0.07, 0.12, -0.09, 0.04,
                0.06, -0.11, 0.03, 0.08, -0.02, 0.05])
Q_B = -0.6 * Q_A


# --------------------------------------------------------------------------
# 1. The id block is per-instance
# --------------------------------------------------------------------------


class TestPerInstanceIdBlock:
    def test_two_receivers_on_one_stream_see_different_arms(self):
        """The refactor's whole purpose, stated as a test.

        One interleaved frame carries both arms.  Each receiver must convert
        its own six bodies and must not have been perturbed by the other's,
        which is the failure the module constant made unavoidable.
        """
        rs485 = MocapRx(rb_id_base=mc.RIGID_BODY_ID_MASK, n_bodies=6)
        canarm = CanArmMocap()
        poses_a = plate_poses_from_q(Q_A)
        poses_b = plate_poses_from_q(Q_B, base_pos=np.array([1.5, 0.0, 0.0]))

        for i in range(6):     # interleaved, as one Motive frame would arrive
            rs485._on_rigid_body(mc.RIGID_BODY_ID_MASK + i,
                                 poses_a[i, 0:3, 3], _quat(poses_a[i]))
            canarm._on_rigid_body(mc.RIGID_BODY_ID_MASK + i,
                                  poses_a[i, 0:3, 3], _quat(poses_a[i]))
            rs485._on_rigid_body(CANARM_RB_ID_BASE + i,
                                 poses_b[i, 0:3, 3], _quat(poses_b[i]))
            canarm._on_rigid_body(CANARM_RB_ID_BASE + i,
                                  poses_b[i, 0:3, 3], _quat(poses_b[i]))
        rs485._on_new_frame({"frame_number": 1})
        canarm._on_new_frame({"frame_number": 1})

        assert np.allclose(rs485.get_q(), Q_A, atol=1e-9)
        assert np.allclose(canarm.get_q(), Q_B, atol=1e-9)

    def test_an_id_outside_the_block_is_dropped_not_trusted(self):
        """A stray body must not land on a joint's row."""
        rx = CanArmMocap()
        before = rx._incoming.copy()
        rx._on_rigid_body(777, np.array([9.0, 9.0, 9.0]),
                          np.array([0.0, 0.0, 0.0, 1.0]))
        assert np.array_equal(rx._incoming, before)

    def test_n_bodies_is_validated_rather_than_silently_clamped(self):
        with pytest.raises(ValueError):
            MocapRx(n_bodies=0)
        with pytest.raises(ValueError):
            MocapRx(n_bodies=mc.KINOVA_RIGID_BODY_INDEX + 1)


# --------------------------------------------------------------------------
# 2. The defaults did not move
# --------------------------------------------------------------------------


class TestDefaultsUnchanged:
    def test_bare_receiver_claims_the_rs485_block(self):
        rx = MocapRx()
        assert rx.rb_id_base == mc.RIGID_BODY_ID_MASK
        assert rx.n_bodies == mc.N_USED_RIGID_BODIES
        # Pre-refactor window: [MASK, MASK + KINOVA_RIGID_BODY_INDEX).
        assert rx._n_rb_rows == mc.KINOVA_RIGID_BODY_INDEX

    def test_the_kinova_still_routes_to_its_own_row(self):
        """1008 is a single id, not a block, and is unaffected by the base."""
        for rx in (MocapRx(), CanArmMocap()):
            rx._on_rigid_body(mc.KINOVA_MOCAP_STREAM_ID,
                              np.array([0.1, 0.2, 0.3]),
                              np.array([0.0, 0.0, 0.0, 1.0]))
            row = rx._incoming[mc.KINOVA_RIGID_BODY_INDEX]
            assert np.allclose(row[0:3, 3], [0.1, 0.2, 0.3])

    def test_canarm_defaults_are_the_briefed_block(self):
        rx = CanArmMocap()
        assert (rx.rb_id_base, rx.n_bodies) == (2000, 6)
        assert (CANARM_RB_ID_BASE, CANARM_N_BODIES) == (2000, 6)


# --------------------------------------------------------------------------
# 3. The synthetic round trip
# --------------------------------------------------------------------------


class TestSyntheticRoundTrip:
    @pytest.mark.parametrize("q", [Q_A, Q_B, np.zeros(mc.NUM_JOINTS)])
    def test_poses_from_q_recover_q(self, q):
        homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
        homos[0:6] = plate_poses_from_q(q)
        assert np.allclose(mocap_to_q(homos), q, atol=1e-9)

    def test_q_is_invariant_to_where_the_volume_origin_is(self):
        """``q`` describes the arm, not the room.

        A rigid motion of the whole volume must change no joint angle; if it
        did, every campaign would be tied to where the cart happened to be.
        """
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
        homos[0:6] = plate_poses_from_q(Q_A, rot_base=rot,
                                        base_pos=np.array([-2.0, 0.7, 1.1]))
        assert np.allclose(mocap_to_q(homos), Q_A, atol=1e-9)

    def test_link_lengths_do_not_enter_q(self):
        """Which is why placeholder lengths are survivable for now."""
        homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
        homos[0:6] = plate_poses_from_q(
            Q_A, links_m=(0.40, 0.09, 0.35, 0.09, 0.33, 0.09))
        assert np.allclose(mocap_to_q(homos), Q_A, atol=1e-9)


# --------------------------------------------------------------------------
# 4. The stream, end to end
# --------------------------------------------------------------------------


class TestSimStream:
    def test_injected_frames_publish_a_readable_q(self):
        rx = CanArmMocap()
        inject_frame(rx, plate_poses_from_q(Q_A), frame_number=7)
        st = rx.get_state()
        assert st.frame_number == 7
        # q_stale, not stale, is the flag a control loop checks; assert both,
        # since an injected frame should clear each for the same reason.
        assert not st.q_stale and not st.stale
        assert np.allclose(rx.get_q(), Q_A, atol=1e-9)

    def test_injection_respects_the_receivers_own_base(self):
        """Feeding a CAN-shaped frame to an RS485-bound receiver yields
        nothing, which is the same silence a wrong base gives live."""
        rx = MocapRx(rb_id_base=mc.RIGID_BODY_ID_MASK, n_bodies=6)
        rx.rb_id_base = CANARM_RB_ID_BASE     # inject at 2000...
        poses = plate_poses_from_q(Q_A)
        for i in range(6):
            rx._on_rigid_body(CANARM_RB_ID_BASE + i, poses[i, 0:3, 3],
                              _quat(poses[i]))
        rx.rb_id_base = mc.RIGID_BODY_ID_MASK  # ...read as if bound to 500
        assert rx.get_q() is None

    def test_the_producer_thread_reaches_a_consumer(self):
        """Opens no socket: ``start`` here spawns a thread, not a client."""
        stream = CanArmSimStream(q_of_t=sweep_q, rate_hz=400.0)
        try:
            stream.start()
            q = stream.wait_fresh(timeout=2.0)
        finally:
            stream.stop()
        assert q is not None and q.shape == (mc.NUM_JOINTS,)
        assert np.all(np.abs(q) <= 0.30)         # sweep_q's own amplitude
        st = stream.get_state()
        assert st.frames >= 1 and st.valid_frames >= 1
        assert stream._client is None            # nothing was ever connected

    def test_stop_is_safe_before_start_and_twice(self):
        stream = CanArmSimStream()
        stream.stop()
        stream.start()
        stream.stop()
        stream.stop()


# --------------------------------------------------------------------------
# 5. The CAN arm's locks do not exist, and saying so is the point
# --------------------------------------------------------------------------


class TestMarkerVariant:
    """``CanArmMarkerMocap`` parameterizes by pass-through, and that is enough.

    ``MarkerMocap`` forwards ``**kwargs`` to ``MocapRx`` untouched, so the two
    new arguments reach the same five read sites with no further change.  The
    fixtures come from ``test_marker_mocap`` rather than being restated here:
    a second copy of the bracket geometry is a second thing to keep true.
    """

    def test_the_marker_path_routes_at_the_can_arms_base(self):
        from .canarm_mocap import CanArmMarkerMocap
        from .test_marker_mocap import (REST_Q, all_tracked, arm_locks,
                                        arm_markers, arm_plates, volume_base)

        base = volume_base(np.random.default_rng(11))
        rx = CanArmMarkerMocap(arm_locks(base))
        assert (rx.rb_id_base, rx.n_bodies) == (CANARM_RB_ID_BASE,
                                                CANARM_N_BODIES)

        plates = arm_plates(REST_Q, base)
        for i in range(6):
            rx._on_rigid_body(CANARM_RB_ID_BASE + i, plates[i, 0:3, 3],
                              _quat(plates[i]))
        rx._marker_scratch = arm_markers(REST_Q, base)
        rx._flags_scratch = all_tracked()
        rx._on_new_frame({"frame_number": 3})

        assert rx.stats.solved == 1
        assert rx.get_marker_poses() is not None
        assert np.allclose(rx.get_q(), REST_Q, atol=1e-6)


class TestLocksRefusal:
    def test_the_default_template_is_absent_and_refused_loudly(self):
        """A lock minted against other plates registers markers onto geometry
        that is not there and reports a confident ``q`` for it, so falling back
        to the RS485 file would be worse than failing."""
        assert not os.path.isfile(DEFAULT_TEMPLATE_PATH)
        with pytest.raises(FileNotFoundError) as exc:
            load_canarm_locks()
        assert "mocap_probe" in str(exc.value)
        assert "rs485_locks_example.json" in str(exc.value)


# --------------------------------------------------------------------------


def _quat(pose: np.ndarray) -> np.ndarray:
    return matrix_to_quat_xyzw(pose[0:3, 0:3])
