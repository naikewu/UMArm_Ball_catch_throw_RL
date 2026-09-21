from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.safe_continuous_env import SafeHandover
from teacher_rl import trajectory_release_rl as v20
from teacher_rl.trajectory_release_env import (PredictiveRelease, TrajectoryHandover,
    TrajectoryReleaseConfig)


@pytest.mark.parametrize("kwargs", [dict(target_force_scale=.7), dict(target_radius_scale=1.2),
    dict(release_joint_deg=40.), dict(release_speed_gain=.6), dict(release_ball_frames=12),
    dict(align_release_clock=False)])
def test_v20_configuration_rejects_invalid_trajectory_or_sensor_contract(kwargs):
    with pytest.raises(ValueError):
        TrajectoryReleaseConfig(**kwargs)


def release_fixture(*, fresh=True, angle=30., error=.02, speed_gain=.88):
    class Original:
        t_release = None
        target = np.array([1., 0.])
        lat, tick, sub, horizon = .02, 1 / 150, 2, .01
        depart_fn = None
        m_ball, v_min, min_elev = .5, 1.5, -.1
        best = (np.inf, None)
        omega = 0.
        def _rates(self, t, velocity):
            pass
        def predict_from(self, position, velocity, delay):
            return np.array([1. + error, 0., .04]), np.array([2., 0., .5]), position
    class Estimator:
        lead = .01 if fresh else None
        def __call__(self, obs):
            return np.array([0., 0., .5]), np.array([2., 0., .5])
    class Plant:
        sim_now = 5.
        def gripper_open(self, t=None, **kwargs):
            self.commands = getattr(self, "commands", []) + [kwargs]
    original = Original()
    original.ball_state = Estimator()
    controller = SimpleNamespace(t_hand=0.)
    config = TrajectoryReleaseConfig(release_min_s=.5, release_tolerance_m=.04, release_speed_gain=speed_gain)
    obs = dict(t=5.006, t_win=3.006, held=True, q_meas=np.full(12, np.radians(angle)))
    return PredictiveRelease(original, controller, config), Plant(), obs


def test_predictive_release_requires_fresh_fit_and_joint_guard_then_schedules_once():
    wrapper, plant, obs = release_fixture(fresh=False)
    wrapper(plant, obs)
    assert not hasattr(plant, "commands")
    wrapper, plant, obs = release_fixture(angle=37.)
    wrapper(plant, obs)
    assert not hasattr(plant, "commands") and wrapper.audit["blocked_joint_ticks"] == 1
    wrapper, plant, obs = release_fixture()
    wrapper(plant, obs)
    wrapper(plant, dict(obs, t=5.012, t_win=3.012))
    assert len(plant.commands) == 1
    assert wrapper.state_at_cmd["origin"] == "predictive_teacher"
    assert wrapper.audit["scheduled"] and wrapper.audit["predicted_error_m"] == pytest.approx(.02)
    assert wrapper.original.speed_gain == pytest.approx(.88)


def test_predictive_release_rejects_prediction_outside_tolerance():
    wrapper, plant, obs = release_fixture(error=.08)
    wrapper(plant, obs)
    assert not hasattr(plant, "commands") and wrapper.audit["blocked_error_ticks"] == 1


def test_trajectory_handover_scales_nominal_orbit_then_restores(monkeypatch):
    monkeypatch.setattr(SafeHandover, "command", lambda controller, obs: (controller.drv.F_max, controller.drv.r_final))
    controller = object.__new__(TrajectoryHandover)
    controller.config = TrajectoryReleaseConfig(target_force_scale=1.1, target_radius_scale=.9)
    controller.drv = SimpleNamespace(F_max=10., r_final=.62)
    assert controller.command({}) == pytest.approx((11., .558))
    assert controller.drv.F_max == 10. and controller.drv.r_final == .62


def test_v20_report_requires_predictive_coverage(monkeypatch):
    source = dict(summary=dict(hit15=2, captured=2, released=2), baseline=dict(hit15=2, captured=2, released=2),
        checks=dict(existing=True), eligible=True, task_nonregression=True, parameter_bins={},
        worst_parameter_bin_hit15_rate=1., max_joint_deg=35., primary_quality_improved=False)
    rows = [dict(result=dict(predictive_release=dict(scheduled=True))),
        dict(result=dict(predictive_release=dict(scheduled=False)))]
    monkeypatch.setattr(v20, "v19_report", lambda rows, baseline: source.copy())
    result = v20.report(rows, rows)
    assert not result["eligible"] and not result["checks"]["predictive_release_coverage"]


def test_v20_validation_requires_eighty_episodes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["trajectory_release_rl", "validate", "--episodes", "20"])
    with pytest.raises(SystemExit) as error:
        v20.main()
    assert error.value.code == 2
