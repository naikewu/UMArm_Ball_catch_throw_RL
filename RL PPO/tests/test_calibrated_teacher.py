from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.calibrated_teacher_env import CalibratedRelease, CalibratedTeacherConfig
from teacher_rl.release_calibration import FEATURE_NAMES, RidgeLandingCalibration
from teacher_rl import calibrated_teacher_rl as v21_teacher


@pytest.mark.parametrize("kwargs", [dict(release_max_s=3.), dict(release_max_s=25.),
    dict(feature_z_limit=1.), dict(feature_z_limit=9.),
    dict(release_speed_gain=.9), dict(calibration_age_cap_s=3.5),
    dict(release_confirm_ticks=-1), dict(release_hysteresis_m=.03),
    dict(release_tolerance_m=.04, release_immediate_m=.05)])
def test_calibrated_teacher_configuration_rejects_out_of_contract_values(kwargs):
    with pytest.raises(ValueError):
        CalibratedTeacherConfig(**kwargs)


def release_fixture(*, correction_x=-.2, scale=100., age=3., angle=30.):
    class Original:
        t_release = None
        target = np.array([1., 0.])
        lat, tick, sub, horizon = .02, 1 / 150, 2, .01
        depart_fn = None
        m_ball, v_min, min_elev = .5, 1.5, -.1
        omega = 0.
        best = (np.inf, None)
        def _rates(self, t, velocity):
            pass
        def predict_from(self, position, velocity, delay):
            return np.array([1.2, 0., .04]), np.array([2., 0., .5]), position
    class Estimator:
        lead = .01
        def __call__(self, observation):
            return np.array([.1, 0., .5]), np.array([2., 0., .5])
    class Plant:
        sim_now = 5.
        def gripper_open(self, t=None, **kwargs):
            self.commands = getattr(self, "commands", []) + [kwargs]
    original = Original(); original.ball_state = Estimator()
    coefficients = np.zeros((len(FEATURE_NAMES) + 1, 2)); coefficients[0, 0] = correction_x
    calibration = RidgeLandingCalibration(np.zeros(len(FEATURE_NAMES)),
        np.full(len(FEATURE_NAMES), scale), coefficients, .25)
    config = CalibratedTeacherConfig(release_min_s=2., release_tolerance_m=.04,
        release_joint_deg=35.)
    controller = SimpleNamespace(t_hand=0., drv=SimpleNamespace(st={"r": .4}, F_max=10.))
    observation = dict(t=5.006, t_win=age, held=True, q_meas=np.full(12, np.radians(angle)))
    return CalibratedRelease(original, controller, config, calibration), Plant(), observation


def test_calibrated_release_schedules_sensor_only_corrected_candidate_once():
    wrapper, plant, observation = release_fixture()
    wrapper(plant, observation)
    wrapper(plant, dict(observation, t=5.012, t_win=3.006))
    wrapper(plant, dict(observation, t=5.018, t_win=3.012))
    wrapper(plant, dict(observation, t=5.024, t_win=3.018))
    assert len(plant.commands) == 1
    assert wrapper.audit["scheduled"] and wrapper.audit["calibrated_error_m"] == pytest.approx(0.)
    assert wrapper.audit["trigger_mode"] == "immediate_strict"
    assert wrapper.state_at_cmd["origin"] == "v21_calibrated_teacher"


def test_calibrated_release_blocks_out_of_distribution_joint_and_time():
    wrapper, plant, observation = release_fixture(scale=.01)
    wrapper(plant, observation)
    assert not hasattr(plant, "commands") and wrapper.audit["blocked_ood_ticks"] == 1
    wrapper, plant, observation = release_fixture(angle=36.)
    wrapper(plant, observation)
    assert not hasattr(plant, "commands") and wrapper.audit["blocked_joint_ticks"] == 1
    wrapper, plant, observation = release_fixture(age=24.1)
    wrapper(plant, observation)
    assert not hasattr(plant, "commands") and wrapper.audit["after_release_window_ticks"] == 1


def test_calibrated_release_caps_only_age_feature_after_calibration_window():
    wrapper, plant, observation = release_fixture(age=5.)
    wrapper(plant, observation)
    wrapper(plant, dict(observation, t=5.012, t_win=5.006))
    assert len(plant.commands) == 1
    assert wrapper.audit["age_clipped_ticks"] == 1
    assert wrapper.audit["feature_age_s"] == pytest.approx(4.)
    assert wrapper.audit["release_age_s"] == pytest.approx(5.)


def test_calibrated_release_audits_best_rejected_candidate():
    wrapper, plant, observation = release_fixture(correction_x=0.)
    wrapper(plant, observation)
    assert not hasattr(plant, "commands")
    assert wrapper.audit["blocked_error_ticks"] == 1
    assert wrapper.audit["min_calibrated_error_m"] == pytest.approx(.2)
    assert wrapper.audit["raw_error_at_min_m"] == pytest.approx(.2)
    assert wrapper.audit["min_error_age_s"] == pytest.approx(3.)
    assert wrapper.audit["calibrated_landing_at_min"] == pytest.approx([1.2, 0.])


def test_v21_validation_requires_selected_screen(tmp_path):
    with pytest.raises(ValueError, match="not selected"):
        v21_teacher.validate(SimpleNamespace(screen=tmp_path))


def test_target_distance_variant_selects_near_and_far_trajectories():
    specification = v21_teacher.variants()["adaptive_d145_t10"]
    near, near_branch = v21_teacher.resolve_variant(specification, {"target_distance": 1.3})
    far, far_branch = v21_teacher.resolve_variant(specification, {"target_distance": 1.6})
    assert near_branch == "near" and far_branch == "far"
    assert near.governor_start_deg == pytest.approx(28.)
    assert far.governor_start_deg == pytest.approx(26.)
    assert near.release_tolerance_m == pytest.approx(.10)
    assert far.release_tolerance_m == pytest.approx(.06)
    assert near.release_confirm_ticks == 1
    assert far.release_confirm_ticks == 1
    assert near.release_immediate_m == pytest.approx(.04)
