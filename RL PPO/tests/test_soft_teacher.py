from dataclasses import replace

import numpy as np
import pytest

from teacher_rl.data import rollout, load_dataset, write_json
from teacher_rl.env import TeacherEnv, TaskConfig
from teacher_rl.soft_env import SoftRecipe, SoftTeacherEnv, pressure_window, SOFT_SCHEMA
from teacher_rl.soft_teacher import paired_comparison


def test_pressure_window_smooth_edges_and_bounds():
    recipe = SoftRecipe(pressure_drop_psi=4.)
    times = np.linspace(.5,1.5,1001)
    window = np.array([pressure_window(t,1.,recipe) for t in times])
    assert window.min() == 0 and window.max() == 1
    assert pressure_window(.8,1.,recipe) == 0
    assert pressure_window(1.25,1.,recipe) == 0
    assert pressure_window(.9,1.,recipe) == pytest.approx(1.)
    assert np.max(np.abs(np.diff(window))) < .016
    assert pressure_window(1.,None,recipe) == 0
    with pytest.raises(ValueError):
        replace(recipe,pressure_drop_psi=float("nan"))


def row(seed=1, **updates):
    result = dict(captured=True,released=True,hit15=True,landing_error_m=.08,
        weld_peak_n=400.,contact_peak_n=0.,contact_impulse_ns=0.,impact_weld_impulse_ns=10.,
        relative_capture_speed_m_s=2.,buffer_displacement_m=.1,pressure_integral_psi_s=260.)
    result.update(updates)
    return dict(seed=seed,result=result)


def test_selection_requires_paired_success_and_real_quality_improvement():
    baseline = [row()]
    improved = paired_comparison([row(weld_peak_n=340.,impact_weld_impulse_ns=8.)],baseline)
    assert improved["improved"]
    assert not paired_comparison(baseline,baseline)["improved"]
    assert not paired_comparison([row(hit15=False,landing_error_m=.16,weld_peak_n=100.)],baseline)["eligible"]
    assert not paired_comparison([row(weld_peak_n=100.,contact_peak_n=80.)],baseline)["eligible"]
    assert not paired_comparison([row(weld_peak_n=100.,pressure_integral_psi_s=290.)],baseline)["eligible"]
    with pytest.raises(ValueError):
        paired_comparison([row(seed=2)],baseline)


def test_v13_data_cannot_silently_enter_v12_training(tmp_path):
    write_json(tmp_path/"episode_1.json",dict(schema=SOFT_SCHEMA,source_hash="soft"))
    with pytest.raises(ValueError,match="mismatch"):
        load_dataset(tmp_path)


def test_zero_recipe_reproduces_original_full_episode():
    config = TaskConfig(authority=0.)
    original, expected = rollout(TeacherEnv(config),401)
    soft, actual = rollout(SoftTeacherEnv(SoftRecipe(),config),401)
    for key in ("capture_time","release_time","landing_error_m","weld_peak_n","pressure_integral_psi_s"):
        assert actual[key] == pytest.approx(expected[key],abs=1e-8)
    np.testing.assert_allclose(soft["trace150"],original["trace150"],atol=1e-7)
    assert actual["relative_capture_speed_m_s"] > 0
    assert actual["capture_physics_time"] <= actual["capture_time"]+.01
    assert actual["impact_weld_peak_n"] <= actual["weld_peak_n"]+1e-6
    assert actual["impact_weld_impulse_ns"] > 0
