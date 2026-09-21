from types import SimpleNamespace

import pytest

from teacher_rl.envelope_teacher_env import EnvelopeTeacherConfig
from teacher_rl import envelope_teacher_rl as teacher


@pytest.mark.parametrize("kwargs", [dict(release_speed_gain=.9), dict(release_max_s=15.),
    dict(calibration_age_cap_s=4.), dict(feature_z_limit=9.),
    dict(envelope_force_scale=1.5), dict(envelope_radius_scale=1.2),
    dict(release_tolerance_m=.04, release_immediate_m=.05)])
def test_v22_teacher_rejects_values_outside_calibrated_envelope(kwargs):
    with pytest.raises(ValueError):
        EnvelopeTeacherConfig(**kwargs)


def test_v22_teacher_variants_match_collected_trajectory_envelope():
    variants = teacher.variants()
    assert len(variants) == 6
    assert {name.removesuffix("_dual") for name in variants} == set(teacher.envelope_trajectories())
    assert max(config.envelope_force_scale for config in variants.values()) == pytest.approx(1.4)
    assert max(config.envelope_radius_scale for config in variants.values()) == pytest.approx(1.18)


def test_v22_validation_requires_selected_screen(tmp_path):
    with pytest.raises(ValueError, match="not selected"):
        teacher.validate(SimpleNamespace(screen=tmp_path))
