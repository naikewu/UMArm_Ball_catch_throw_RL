from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.safe_continuous_env import SafeContinuousConfig, SafeHandover
from teacher_rl.buffered_env import BufferedHandover
from teacher_rl import safe_continuous_rl as v19


@pytest.mark.parametrize("kwargs", [dict(governor_start_deg=20.), dict(governor_band_deg=20.),
    dict(force_start_scale=.1), dict(force_ramp_s=2.), dict(radius_ramp_s=float("nan")),
    dict(governor_start_deg=36., governor_band_deg=8.)])
def test_safe_configuration_rejects_invalid_governor_or_ramp(kwargs):
    with pytest.raises(ValueError):
        SafeContinuousConfig(**kwargs)


def test_safe_handover_ramps_force_radius_and_restores_nominal_drive(monkeypatch):
    def parent(controller, obs):
        return controller.drv.F_max, controller.drv.r_final
    monkeypatch.setattr(BufferedHandover, "command", parent)
    config = SafeContinuousConfig(force_start_scale=.5, force_ramp_s=1., radius_ramp_s=1.)
    controller = object.__new__(SafeHandover)
    controller.config = config
    controller.t_hand = 3.
    controller.q_cap, controller.q_band, controller.gov_q = .5, .1, .8
    controller.drv = SimpleNamespace(F_max=10., r_final=.62, r0=.20)
    controller.env = SimpleNamespace(applied_force_scale=1., applied_radius_scale=1.,
        min_joint_governor=1., min_force_scale=1., min_radius_scale=1.)
    force, radius = controller.command(dict(t_win=3.5, held=True))
    assert force == pytest.approx(7.5)
    assert radius == pytest.approx(.41)
    assert controller.drv.F_max == 10.
    assert controller.drv.r_final == .62
    assert controller.env.min_joint_governor == .8
    assert controller.env.min_force_scale == pytest.approx(.75)
    assert controller.env.min_radius_scale == pytest.approx(.5)


def test_first_held_tick_keeps_measured_radius_but_applies_force_ramp(monkeypatch):
    monkeypatch.setattr(BufferedHandover, "command", lambda controller, obs:(controller.drv.F_max,controller.drv.r_final))
    config = SafeContinuousConfig(force_start_scale=.6, force_ramp_s=.6, radius_ramp_s=.6)
    controller = object.__new__(SafeHandover)
    controller.config, controller.t_hand, controller.gov_q = config, None, 1.
    controller.drv = SimpleNamespace(F_max=10., r_final=.62, r0=.20)
    controller.env = SimpleNamespace(applied_force_scale=1., applied_radius_scale=1.,
        min_joint_governor=1., min_force_scale=1., min_radius_scale=1.)
    force, radius = controller.command(dict(t_win=3., held=True))
    assert force == pytest.approx(6.) and radius == pytest.approx(.62)
    assert controller.drv.F_max == 10. and controller.drv.r_final == .62


def report_fixture(max_joint=35., error=.045, base_error=.042):
    return dict(summary=dict(hit15=10,captured=10,released=10,mean_landing_error_m=error),
        baseline=dict(hit15=10,captured=10,released=10,mean_landing_error_m=base_error),
        checks=dict(joint_excursion_guard=True, other=True), eligible=True,
        task_nonregression=True, impact_peak_ratio=1., impact_impulse_ratio=1.,
        parameter_bins=dict(axis=[dict(episodes=10)]), worst_parameter_bin_hit15_rate=.8), [
            dict(result=dict(max_joint_deg=max_joint,safety_abort=False))]


def test_v19_report_uses_absolute_joint_limit_and_5mm_landing_margin(monkeypatch):
    source, rows = report_fixture(max_joint=35.9, error=.047, base_error=.042)
    monkeypatch.setattr(v19, "v17_report", lambda rows, baseline:source.copy())
    result = v19.report(rows, rows)
    assert result["eligible"] and result["checks"]["joint_excursion_guard"]
    assert result["checks"]["landing_noninferiority_5mm"]
    source, rows = report_fixture(max_joint=36.01)
    monkeypatch.setattr(v19, "v17_report", lambda rows, baseline:source.copy())
    assert not v19.report(rows, rows)["eligible"]


def test_v19_validation_requires_eighty_episodes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["safe_continuous_rl", "validate", "--episodes", "20"])
    with pytest.raises(SystemExit) as error:
        v19.main()
    assert error.value.code == 2


def test_v19_validation_requires_a_passing_screen(tmp_path):
    with pytest.raises(ValueError, match="not produced"):
        v19.validate(SimpleNamespace(screen=tmp_path, init=tmp_path))


def test_v19_selection_prefers_lower_joint_excursion_after_task_ties():
    def candidate(q):
        return dict(report=dict(summary=dict(hit15=10,captured=10,mean_landing_error_m=.04),max_joint_deg=q))
    assert v19.selection_score(candidate(34)["report"]) > v19.selection_score(candidate(35)["report"])
