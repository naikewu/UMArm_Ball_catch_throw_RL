"""``SimMocap``: the twin's joints reach a controller through the real receiver.

Offline, no Tk, no socket, no serial port.  What is pinned:

* it IS the CAN arm's receiver class, bound to rigid bodies 2000-2005;
* a posed twin's ``q`` comes back out of ``get_q`` to float precision, having
  been through plate poses, the listeners and ``mocap_to_q``;
* the producer thread never steps physics;
* a source that has gone away publishes nothing and reads stale;
* the twin model's u-joint centres agree with ``fkine`` from the same ``q``;
* a board driven through ``SimMaster``'s bus moves the joint the measured
  actuator map says it moves, and the receiver sees it.

What it does not show: anything about Motive, markers or the 45 deg frame
offset.  This receiver carries no markers.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from digital_twin import mjcf_generator as MG  # noqa: E402
from digital_twin import sim_core as SC  # noqa: E402
from digital_twin import sim_mocap as SMC  # noqa: E402
from UMArm_MOCAP import mocap_constants as mc  # noqa: E402
from UMArm_MOCAP.canarm_mocap import (CANARM_N_BODIES, CANARM_RB_ID_BASE,  # noqa: E402
                                      CanArmMocap)


def _posed_arm(q) -> SC.SimArm:
    arm = SC.SimArm()
    with arm.lock:
        arm.data.qpos[:] = MG.q_to_qpos(q)
        mujoco.mj_forward(arm.model, arm.data)
    return arm


def test_it_is_the_can_arm_receiver_on_the_can_arm_block():
    rx = SMC.SimMocap(SC.SimArm())
    assert isinstance(rx, CanArmMocap)
    assert rx.rb_id_base == CANARM_RB_ID_BASE and rx.n_bodies == CANARM_N_BODIES


def test_q_of_t_is_not_an_argument():
    with pytest.raises(TypeError):
        SMC.SimMocap(SC.SimArm(), q_of_t=lambda t: np.zeros(12))


def test_joint_noise_is_seeded_and_has_the_requested_scale():
    q = np.full(12, 0.05)
    arm = _posed_arm(q)
    sigma = np.deg2rad(0.10)
    a = SMC.SimMocap(arm, joint_noise_std_rad=sigma, seed=41)
    b = SMC.SimMocap(arm, joint_noise_std_rad=sigma, seed=41)
    samples = np.array([a._sample_q(i / 240) for i in range(1000)])
    repeat = np.array([b._sample_q(i / 240) for i in range(1000)])
    np.testing.assert_array_equal(samples, repeat)
    assert abs(np.std(samples - q) / sigma - 1) < 0.03
    assert abs(np.mean(samples - q)) < sigma * 0.03


def test_latency_uses_only_samples_old_enough_and_keeps_history_bounded():
    arm = _posed_arm(np.zeros(12))
    rx = SMC.SimMocap(arm, latency_s=0.010)
    assert rx._sample_q(0.0) is None
    with arm.lock:
        arm.data.qpos[:] = MG.q_to_qpos(np.full(12, 0.1))
    np.testing.assert_array_equal(rx._sample_q(0.01), np.zeros(12))
    np.testing.assert_allclose(rx._sample_q(0.02), 0.1)
    for i in range(3, 200):
        rx._sample_q(i * 0.01)
    assert len(rx._measurement_history) <= 3


@pytest.mark.parametrize("kwargs", [dict(joint_noise_std_rad=-1),
                                    dict(joint_noise_std_rad=float("nan")),
                                    dict(latency_s=-1)])
def test_invalid_measurement_options_refused(kwargs):
    with pytest.raises(ValueError):
        SMC.SimMocap(SC.SimArm(), **kwargs)


def test_a_posed_twin_round_trips_through_the_receiver_exactly():
    q = np.random.default_rng(20260910).uniform(-0.35, 0.35, 12)
    rx = SMC.SimMocap(_posed_arm(q), rate_hz=200.0).start()
    try:
        got = rx.wait_fresh(timeout=2.0)
    finally:
        rx.stop()
    assert got is not None
    np.testing.assert_allclose(got, q, rtol=0.0, atol=1e-9)
    assert not rx.get_state().last_error


def test_the_stream_never_steps_physics():
    arm = SC.SimArm()
    before = (arm.quanta_done, arm._t_target, float(arm.data.time))
    rx = SMC.SimMocap(arm, rate_hz=200.0).start()
    try:
        time.sleep(0.3)
    finally:
        rx.stop()
    assert rx.get_state().frames >= 20
    assert (arm.quanta_done, arm._t_target, float(arm.data.time)) == before


def test_a_source_that_has_gone_away_publishes_nothing_and_reads_stale():
    alive = [True]
    rx = SMC.SimMocap(SC.SimArm(), rate_hz=200.0, alive=lambda: alive[0]).start()
    try:
        assert rx.wait_fresh(timeout=2.0) is not None
        alive[0] = False
        time.sleep(0.05)
        frames = rx.get_state().frames
        time.sleep(mc.STALE_AFTER_S + 0.15)
        state = rx.get_state()
    finally:
        rx.stop()
    assert state.frames == frames
    assert state.stale and state.q_stale
    assert rx.frames_skipped > 0


def test_the_twin_centres_agree_with_fkine_from_the_same_q():
    """Measured 0.0000 mm at build time; 0.1 mm is the tolerance."""
    q = np.random.default_rng(7).uniform(-0.3, 0.3, 12)
    arm = _posed_arm(q)
    rx = SMC.SimMocap(arm)
    res = rx.fk_residual_m(q)
    assert res is not None and res.shape == (6,)
    assert res[0] == pytest.approx(0.0, abs=1e-12)
    assert float(np.max(res)) < 1e-4
    assert rx.fk_residual_m() is None          # no frame published yet


def test_a_board_driven_through_the_bus_moves_its_joint_and_the_receiver_sees_it():
    """0x101 is the positive member of joint 2's pair in the measured map."""
    from digital_twin import sim_master as SM
    from digital_twin import twin_params as TP
    from UMArm_KINEMATICS import canarm_actuators as ACT

    joint, sign = ACT.base_to_joint()[0x101]
    assert (joint, sign) == (2, +1)

    master = SM.SimMaster(**TP.load_twin_kwargs(log=None))
    rx = SMC.SimMocap(master.arm, rate_hz=120.0,
                      alive=lambda: master.link.is_open)
    master.open()
    try:
        assert len(master.scan()) == 24
        rx.start()
        q0 = rx.wait_fresh(timeout=2.0)
        assert q0 is not None
        master.select([0x101])
        master.start_cycle()
        master.set_enabled(0x101, True)
        master.set_target(0x101, 12.0)
        time.sleep(1.5)
        q1 = rx.wait_fresh(timeout=1.0)
        psi = master.snapshot_nodes()[0x101].pressure_psi
    finally:
        master.stop_all()
        master.close()
        rx.stop()
    assert q1 is not None
    assert psi > 9.0
    assert q1[joint] - q0[joint] > math.radians(1.0)
