from types import SimpleNamespace

import pytest

from teacher_rl import trajectory_envelope_rl as v22


def test_v22_envelope_has_six_trajectories_and_six_phases():
    trajectories = v22.trajectories()
    assert len(trajectories) == 6
    assert len(v22.PHASES) == 6
    assert len(v22.cells()) == 36
    assert min(config["envelope_radius_scale"] for config in trajectories.values()) == pytest.approx(1.)
    assert max(config["envelope_radius_scale"] for config in trajectories.values()) == pytest.approx(1.18)
    assert min(config["envelope_force_scale"] for config in trajectories.values()) == pytest.approx(1.)
    assert max(config["envelope_force_scale"] for config in trajectories.values()) == pytest.approx(1.4)


@pytest.mark.parametrize("age", [1.9, 18.1])
def test_v22_probe_rejects_age_outside_campaign(age):
    with pytest.raises(ValueError):
        v22.EnvelopeProbeConfig(probe_age_s=age)


def test_v22_fit_requires_collection(tmp_path):
    with pytest.raises(ValueError, match="V22 Collect"):
        v22.fit(SimpleNamespace(out=tmp_path, init=tmp_path))


def test_v22_stratification_covers_all_cells():
    cell_count = len(v22.cells())
    coverage = {fold: set() for fold in range(5)}
    for scenario_id in range(cell_count * 5):
        fold = v22.stratified_fold(dict(scenario_id=scenario_id), cell_count)
        coverage[fold].add(scenario_id % cell_count)
    assert all(cells == set(range(cell_count)) for cells in coverage.values())
