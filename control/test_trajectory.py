"""Kinematic, derivative and task-equivalence checks for the controller path."""
import numpy as np
import pytest

from UMArm_KINEMATICS import fkine
from UMArm_KINEMATICS.canarm_params import CANARM_PARAMS
from control.trajectory import (SoftTrajectory, tip_position, tip_jacobian,
                                 inverse_kinematics, LENGTH_SCALE, Q_LIMIT_RAD)


@pytest.fixture(scope="module")
def word():
    return SoftTrajectory()


def test_forward_and_jacobian_use_measured_joint_order():
    qs = np.random.default_rng(19).uniform(-0.3, 0.3, (20, 12))
    expected = np.array([fkine(q, CANARM_PARAMS, order="yx")[:3, 3] for q in qs])
    np.testing.assert_allclose(tip_position(qs), expected, atol=8e-16)
    q = qs[0]
    h = 1e-6
    numerical = np.column_stack([(tip_position(q+row*h)-tip_position(q-row*h))/(2*h)
                                  for row in np.eye(12)])
    np.testing.assert_allclose(tip_jacobian(q), numerical, atol=3e-10)
    # The final universal joint controls orientation, not this plate centre.
    np.testing.assert_allclose(tip_jacobian(q)[:, 10:], 0, atol=5e-16)


def test_ik_reports_unreachable_without_violating_angle_bound():
    q, info = inverse_kinematics([1.5, 0, -0.5], return_info=True)
    assert np.max(np.abs(q)) <= Q_LIMIT_RAD
    assert info["residual_m"] > 0.5
    q = inverse_kinematics(tip_position(np.full(12, 0.05)))
    assert np.linalg.norm(tip_position(q)-tip_position(np.full(12, 0.05))) < 2e-5


def test_scaled_whole_word_is_reachable(word):
    ts = np.linspace(word.write_start_s, word.write_end_s, 600)
    qs = np.array([word.sample(t)[0] for t in ts])
    target = np.array([word.tip_target(t) for t in ts])
    assert np.max(np.linalg.norm(tip_position(qs)-target, axis=1)) < 1e-4
    assert np.max(np.abs(qs)) < Q_LIMIT_RAD
    assert word.metadata["extent_m"][0] == pytest.approx(0.60083, abs=1e-5)
    assert np.all(word.pen_down(ts))
    assert not word.pen_down(word.write_start_s-0.01)
    assert not word.pen_down(word.write_end_s+0.01)


def test_path_and_speed_scale_together(word):
    fast = SoftTrajectory("fast")
    assert word.speed_m_s == pytest.approx(0.15*LENGTH_SCALE)
    assert fast.speed_m_s == pytest.approx(2*word.speed_m_s)
    assert fast.length_m == word.length_m
    assert fast.metadata["extent_m"] == word.metadata["extent_m"]
    assert ((word.write_end_s-word.write_start_s-word.ramp_s)
            == pytest.approx(2*(fast.write_end_s-fast.write_start_s-fast.ramp_s)))


def test_q_velocity_acceleration_share_one_curve(word):
    for t in (1.4, word.write_start_s+0.4, word.write_start_s+2.345,
              word.write_end_s-0.43):
        h = 1e-5
        q, qd, qdd = word.sample(t)
        qm, vm, _ = word.sample(t-h)
        qp, vp, _ = word.sample(t+h)
        np.testing.assert_allclose(qd, (qp-qm)/(2*h), atol=1e-7, rtol=2e-5)
        np.testing.assert_allclose(qdd, (vp-vm)/(2*h), atol=1e-5, rtol=3e-4)
    np.testing.assert_allclose(word.future(3, 4, 1/150),
        np.array([word.sample(3+k/150)[0] for k in range(1, 5)]))


def test_all_start_stop_joins_have_continuous_acceleration(word):
    joins = (0, word.entry_s, word.write_start_s,
             word.write_start_s+word.ramp_s,
             word.write_end_s-word.ramp_s, word.write_end_s)
    for t in joins:
        left, right = word.sample(t-1e-8), word.sample(t+1e-8)
        for a, b in zip(left, right):
            np.testing.assert_allclose(a, b, atol=1e-5)
    _, qd, qdd = word.sample(word.duration_s+1)
    np.testing.assert_array_equal(qd, np.zeros(12))
    np.testing.assert_array_equal(qdd, np.zeros(12))


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        inverse_kinematics([np.nan, 0, 0])
    with pytest.raises(ValueError):
        SoftTrajectory(size_scale=0)
    with pytest.raises(ValueError):
        SoftTrajectory(speed="unknown")
