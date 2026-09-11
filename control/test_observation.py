import numpy as np
import pytest

from control.observation import CausalJointObserver


def test_camera_observer_matches_collection_recurrence_exactly():
    rng = np.random.default_rng(31)
    times = np.arange(140)/240
    q = .15*np.sin(2*np.pi*times[:, None]*np.linspace(.2, 2, 12))
    q += rng.normal(0, np.deg2rad(.1), q.shape)
    observer = CausalJointObserver()
    velocity = np.zeros(12)
    np.testing.assert_array_equal(observer.update(times[0], q[0]), velocity)
    for i in range(1, len(times)):
        delta = times[i]-times[i-1]
        alpha = -np.expm1(-delta/.04)
        velocity += alpha*((q[i]-q[i-1])/delta-velocity)
        np.testing.assert_array_equal(observer.update(times[i], q[i]), velocity)


def test_delay_transport_preserves_acquisition_based_velocity():
    """Filtering before fixed delay equals replaying all delayed frames."""
    samples = np.random.default_rng(10).normal(0, .001, (100, 12))
    acquisition, delivery = CausalJointObserver(), CausalJointObserver()
    recorded = []
    for i, q in enumerate(samples):
        recorded.append(acquisition.update(i/240, q))
    for i, q in enumerate(samples):
        np.testing.assert_array_equal(delivery.update(i/240, q), recorded[i])


def test_repeated_frames_cannot_create_velocity_or_alias_returned_state():
    observer = CausalJointObserver()
    observer.update(0, np.zeros(12))
    expected = observer.update(1/240, np.full(12, .001))
    assert np.linalg.norm(expected) > 0
    output = observer.update(1/240, np.ones(12))
    np.testing.assert_array_equal(output, expected)
    output[:] = 10
    np.testing.assert_array_equal(observer.qdot, expected)
    observer.reset()
    np.testing.assert_array_equal(observer.update(5, np.ones(12)), np.zeros(12))


def test_invalid_and_backward_observations_raise():
    observer = CausalJointObserver()
    observer.update(1, np.zeros(12))
    with pytest.raises(ValueError):
        observer.update(.5, np.zeros(12))
    with pytest.raises(ValueError):
        observer.update(2, np.full(12, np.nan))
    with pytest.raises(ValueError):
        CausalJointObserver(tau_s=0)
