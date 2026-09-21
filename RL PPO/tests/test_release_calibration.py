from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.release_calibration import (FEATURE_NAMES, RidgeLandingCalibration, feature_vector,
    fit_ridge, release_candidate)
from teacher_rl.release_calibration_env import CalibrationProbeConfig, ProbeRelease
from teacher_rl import release_calibration_rl as v21


@pytest.mark.parametrize("kwargs", [dict(probe_age_s=1.), dict(probe_age_s=7.), dict(probe_name="bad name")])
def test_probe_configuration_rejects_invalid_campaign_contract(kwargs):
    with pytest.raises(ValueError):
        CalibrationProbeConfig(**kwargs)


def test_ridge_calibration_recovers_holdout_landing_correction():
    rng = np.random.default_rng(21)
    features = rng.normal(size=(100, len(FEATURE_NAMES)))
    correction = np.c_[.04 + .03 * features[:, 0] - .01 * features[:, 4],
        -.02 + .02 * features[:, 2] + .04 * features[:, 5]]
    model = fit_ridge(features[:80], correction[:80], ridge_lambda=.01)
    predicted = np.vstack([model.predict_delta(row) for row in features[80:]])
    assert np.mean(np.linalg.norm(predicted - correction[80:], axis=1)) < 1e-3
    restored = RidgeLandingCalibration.from_dict(model.to_dict())
    assert np.allclose(restored.predict_delta(features[99]), model.predict_delta(features[99]))


def release_fixture(*, fresh=True, angle=30.):
    class Original:
        t_release = None
        target = np.array([1., 0.])
        lat, tick, sub, horizon = .02, 1 / 150, 2, .01
        depart_fn = None
        m_ball, v_min, min_elev = .5, 1.5, -.1
        omega = 0.
        def _rates(self, t, velocity):
            pass
        def predict_from(self, position, velocity, delay):
            return np.array([1.02, 0., .04]), np.array([2., 0., .5]), position
    class Estimator:
        lead = .01 if fresh else None
        def __call__(self, observation):
            return np.array([.1, 0., .5]), np.array([2., 0., .5])
    class Plant:
        sim_now = 5.
        def gripper_open(self, t=None, **kwargs):
            self.commands = getattr(self, "commands", []) + [kwargs]
    config = CalibrationProbeConfig(probe_age_s=2., release_joint_deg=35.)
    controller = SimpleNamespace(t_hand=0., drv=SimpleNamespace(st={"r": .4}, F_max=10.))
    observation = dict(t=5.006, t_win=3.006, held=True, q_meas=np.full(12, np.radians(angle)))
    original = Original(); original.ball_state = Estimator()
    return ProbeRelease(original, controller, config), Plant(), observation


def test_probe_release_uses_fresh_state_and_joint_guard_without_target_error_gate():
    wrapper, plant, observation = release_fixture(fresh=False)
    wrapper(plant, observation)
    assert not hasattr(plant, "commands")
    wrapper, plant, observation = release_fixture(angle=36.)
    wrapper(plant, observation)
    assert not hasattr(plant, "commands") and wrapper.audit["blocked_joint_ticks"] == 1
    wrapper, plant, observation = release_fixture()
    wrapper(plant, observation)
    assert len(plant.commands) == 1 and wrapper.audit["scheduled"]
    assert len(wrapper.audit["command"]["features"]) == len(FEATURE_NAMES)


def test_feature_vector_and_candidate_are_sensor_state_only():
    original = SimpleNamespace(lat=.02, depart_fn=None, omega=0., target=np.array([1., 0.]),
        predict_from=lambda p, v, delay: (np.array([1.1, .2, .04]), np.array([2., .1, .5]), p))
    candidate = release_candidate(original, np.array([.2, .1, .5]), np.array([2., .1, .5]))
    controller = SimpleNamespace(t_hand=1., drv=SimpleNamespace(st={"r": .4}, F_max=10.))
    features = feature_vector(np.array([.2, .1, .5]), np.array([2., .1, .5]), candidate, controller,
        dict(t_win=3., q_meas=np.zeros(12)), original.target)
    assert features.shape == (len(FEATURE_NAMES),) and np.isfinite(features).all()


def test_v21_fit_requires_collected_campaign(tmp_path):
    with pytest.raises(ValueError, match="Run Collect"):
        v21.fit(SimpleNamespace(out=tmp_path, init=tmp_path))


def test_stratified_fold_covers_every_cell_in_every_fold():
    cell_count = 15
    coverage = {fold: set() for fold in range(5)}
    for scenario_id in range(150):
        row = dict(scenario_id=scenario_id)
        coverage[v21.stratified_fold(row, cell_count)].add(scenario_id % cell_count)
    assert all(cells == set(range(cell_count)) for cells in coverage.values())


def test_acceptance_checks_allow_an_already_accurate_phase_without_fake_improvement():
    good = dict(rows=30, raw_mean_m=.12, calibrated_mean_m=.06, calibrated_p90_m=.12,
        relative_mean_improvement=.5)
    accurate = dict(rows=25, raw_mean_m=.04, calibrated_mean_m=.045, calibrated_p90_m=.08,
        relative_mean_improvement=-.125)
    checks = v21.acceptance_checks(good, [good] * 5, {"q26": good}, {"2.0": accurate, "3.0": good})
    assert all(checks.values())


def test_acceptance_checks_reject_a_hidden_trajectory_or_phase_failure():
    good = dict(rows=30, raw_mean_m=.12, calibrated_mean_m=.06, calibrated_p90_m=.12,
        relative_mean_improvement=.5)
    bad = dict(rows=30, raw_mean_m=.20, calibrated_mean_m=.10, calibrated_p90_m=.18,
        relative_mean_improvement=.5)
    checks = v21.acceptance_checks(good, [good] * 5, {"safe": good, "bad": bad}, {"2.0": good})
    assert not checks["every_trajectory_mean_le_8cm"]
    assert not checks["every_trajectory_p90_le_16cm"]
