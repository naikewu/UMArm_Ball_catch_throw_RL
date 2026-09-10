"""Offline checks of :mod:`digital_twin.mech_fit`: no recording, no hardware.

What these pin is where a wrong number hides in a fit rather than the
optimisation itself, since a CMA-ES run on this objective costs hours.  They
cover the parameter packing and its bounds, the rest-gain/shape coordinates of
the force law, the penalty an invalid rollout scores and how it ranks, the
window planner's refusals (held-out rows, sync gaps, an over-reaching lead-in),
the profile lines, and that the checkpoint the fit writes loads through
``twin_params`` as exactly the twin that was scored.  One 0.6 s synthetic window
rolls through the worker path, so that path runs end to end at least once.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from digital_twin import actuator_model as AM
from digital_twin import mech_fit as MF
from digital_twin import mjcf_generator as MG
from digital_twin import replay as R
from digital_twin import twin_compare as TC
from digital_twin import twin_params as TP


# ---------------------------------------------------------------------------
# Parameters and bounds
# ---------------------------------------------------------------------------

def test_every_parameter_has_bounds_around_its_start_and_names_its_evidence():
    assert MF.N_PARAMS == 15, "3 link + 3 bracket masses, 3 gains, 3 shapes, 3 dissipation"
    assert len(set(MF.NAMES)) == MF.N_PARAMS
    for p in MF.PARAMS:
        assert 0.0 < p.lo < p.start < p.hi, p.name
        assert len(p.evidence) > 40, f"{p.name} states no evidence for its bounds"


def test_the_mass_bounds_bracket_the_prior_and_respect_the_actuator_floor():
    """A link cannot weigh less than its eight 30 g sleeves; the total brackets ~2.1 kg."""
    links = [p for p in MF.PARAMS if p.name.startswith("link_mass")]
    assert all(p.lo > 8 * MG.DEFAULT_ACTUATOR_MASS_KG for p in links)
    masses = MF.PARAMS[:6]
    lo_total = sum(p.lo for p in masses) + MG.DEFAULT_TIP_MASS_KG
    hi_total = sum(p.hi for p in masses) + MG.DEFAULT_TIP_MASS_KG
    prior = sum(MG.KOOPMAN_PROMAX_LINK_MASS_KG) + sum(MG.KOOPMAN_PROMAX_UJOINT_MASS_KG)
    assert 24 * MG.DEFAULT_ACTUATOR_MASS_KG < lo_total < prior < hi_total
    assert MF.PARAMS[5].name == "bracket_mass_kg_3"
    assert MF.PARAMS[5].lo < MF.PARAMS[3].lo, "segment 3's ring-only body may be small"


def test_the_gain_bounds_bracket_the_earlier_outer_fit_by_a_factor_of_three():
    gains = [p for p in MF.PARAMS if p.name.startswith("rest_gain")]
    earlier = MF._rest_gain(MF.OUTER_FIT_COEFF, AM.BF_SEED_M, AM.L0_SEED_M)
    for p, k in zip(gains, earlier):
        assert p.lo * 3.0 <= k <= p.hi / 3.0, (p.name, k)


def test_unit_packing_round_trips_and_clips_onto_the_bounds():
    rng = np.random.default_rng(1)
    for _ in range(50):
        z = rng.uniform(0.0, 1.0, MF.N_PARAMS)
        values = MF.from_unit(z)
        assert MF.in_bounds(values)
        np.testing.assert_allclose(MF.to_unit(values), z, atol=1e-12)
    np.testing.assert_allclose(MF.to_unit({p.name: p.lo for p in MF.PARAMS}), 0.0, atol=1e-12)
    np.testing.assert_allclose(MF.to_unit({p.name: p.hi for p in MF.PARAMS}), 1.0, atol=1e-12)
    clipped = MF.from_unit(np.full(MF.N_PARAMS, 1.7))
    assert all(math.isclose(clipped[p.name], p.hi, rel_tol=1e-12) for p in MF.PARAMS)
    with pytest.raises(KeyError):
        MF.as_vector({"link_mass_kg_1": 1.0})
    with pytest.raises(ValueError):
        MF.to_unit(np.zeros(MF.N_PARAMS))


def test_the_force_law_coordinates_round_trip_and_mean_rest_force_per_pascal():
    values = MF.start_values()
    coeff, bf = MF.force_law(values, AM.L0_SEED_M)
    gain, shape = MF.gain_and_shape(coeff, bf, AM.L0_SEED_M)
    np.testing.assert_allclose(gain, [values[n] for n in MF.GAIN_NAMES], rtol=1e-12)
    np.testing.assert_allclose(shape, [values[n] for n in MF.SHAPE_NAMES], rtol=1e-12)
    np.testing.assert_allclose(coeff, MF.OUTER_FIT_COEFF, rtol=1e-12)

    model = AM.ActuatorModel.fresh(is_tle=np.zeros(AM.N_ACT, dtype=bool), coeff=coeff, bf=bf)
    p = 50_000.0
    pull = -model.force_n(np.full(AM.N_ACT, p), np.zeros(AM.N_ACT))
    np.testing.assert_allclose(pull, np.repeat(gain, 8) * p, rtol=1e-12)

    bad = dict(values)
    bad["bf_over_l0_1"] = 1.8
    with pytest.raises(ValueError, match="sqrt"):
        MF.force_law(bad)


def test_values_come_back_from_an_outer_fit_file_with_the_rest_at_the_start():
    outer = {"coeff": list(MF.OUTER_FIT_COEFF), "tendon_damping": 0.5, "joint_damping": 0.01}
    values = MF.values_from_mech(outer)
    np.testing.assert_allclose(MF.force_law(values)[0], MF.OUTER_FIT_COEFF, rtol=1e-12)
    assert values["tendon_damping"] == 0.5 and values["joint_damping"] == 0.01
    assert values["link_mass_kg_1"] == MG.DEFAULT_LINK_MASS_KG[0]


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------

def _result(real, sim, *, valid=True):
    n = real.shape[0]
    return TC.TwinResult(
        rec_rows=np.arange(n), t_rel_s=np.arange(n) / 150.0, q_sim_rad=sim,
        twin_defl_rad=sim, real_defl_rad=real, p_sim_pa=np.zeros((n, 24)),
        p_real_pa=np.zeros((n, 24)), ref_rows=np.arange(min(n, 5)),
        sim_ref_rad=np.zeros(12), real_ref_rad=np.zeros(12), alignment=None,
        ctrl_min_n=0.0, valid=valid, reason="" if valid else "test")


def test_the_loss_is_the_twelve_joint_mean_rms_in_degrees():
    rng = np.random.default_rng(3)
    real = rng.normal(0.0, 0.1, (300, 12))
    sim = real + np.deg2rad(np.arange(12, dtype=float))[None, :]
    loss, ok, _ = MF.loss_from_result(_result(real, sim))
    assert ok
    assert loss == pytest.approx(5.5, rel=1e-12)


def test_an_invalid_rollout_scores_a_finite_penalty_and_ranks_below_every_valid_one():
    loss, ok, reason = MF.loss_from_result(TC._invalid("clamp violation: test"))
    assert (loss, ok) == (MF.PENALTY_DEG, False) and "clamp" in reason
    assert math.isfinite(loss)
    for sim in (np.full((300, 12), np.nan), np.full((300, 12), np.inf)):
        assert MF.loss_from_result(_result(np.zeros((300, 12)), sim))[1] is False
    assert MF.loss_from_result(_result(np.zeros((5, 12)), np.zeros((5, 12))))[1] is False

    fams = ["random_walk"] + ["chirp"] * 2 + ["ringdown"] * 6 + ["staircase"] * 3 + ["pair_sweep"]
    worst_valid = np.full(13, 160.0)
    one_bad = np.zeros(13)
    one_bad[4] = MF.PENALTY_DEG
    two_bad = one_bad.copy()
    two_bad[5] = MF.PENALTY_DEG
    valid = np.ones((3, 13), dtype=bool)
    valid[1, 4] = False
    valid[2, [4, 5]] = False
    total, per = MF.aggregate(np.stack([worst_valid, one_bad, two_bad]), fams, valid)
    assert total[0] < total[1] < total[2]
    assert per["ringdown"][1] == pytest.approx(MF.PENALTY_DEG / 6)


def test_the_families_weigh_equally_whatever_their_window_count():
    fams = ["ringdown"] * 6 + ["chirp"] * 2 + ["random_walk"]
    losses = np.array([[1.0] * 6 + [4.0] * 2 + [10.0]])
    total, per = MF.aggregate(losses, fams)
    assert per["ringdown"][0] == 1.0 and per["chirp"][0] == 4.0
    assert total[0] == pytest.approx(5.0)
    assert list(per) == ["random_walk", "chirp", "ringdown"]


def test_one_window_rolls_through_the_worker_path_and_scores_finite():
    """0.6 s of synthetic arm on a fresh net: the SimArm, rollout and scoring path."""
    n = 90
    t = np.arange(n) / 150.0
    board_type = [R.VARIANT_TLE_DVP] * 8 + [R.VARIANT_7MM] * 16
    idle = np.array([R.pa_to_counts(0.5 * R.PA_PER_PSI, v) for v in board_type])
    target = np.tile(idle, (n, 1))
    target[30:, 0] = R.pa_to_counts(8.0 * R.PA_PER_PSI, R.VARIANT_TLE_DVP)
    rec = R.recording_from_arrays(t, np.zeros((n, 12)), target, target, board_type)
    base = AM.ActuatorModel.fresh(is_tle=np.asarray(board_type) == R.VARIANT_TLE_DVP)
    loss, ok, reason = MF.window_loss(rec, MF.twin_kwargs(MF.start_values(), base))
    assert ok, reason
    assert 0.0 <= loss < MF.PENALTY_DEG


# ---------------------------------------------------------------------------
# The window planner
# ---------------------------------------------------------------------------

def _plan():
    return [
        {"name": "rest", "kind": "rest", "duration_s": 2.0},
        {"name": "stair_101_4", "kind": "staircase", "duration_s": 1.0,
         "meta": {"base": "0x101", "psi": 4.0}},
        {"name": "stair_101_vent", "kind": "staircase", "duration_s": 1.0,
         "meta": {"base": "0x101", "psi": 0.0}},
        {"name": "validation_0", "kind": "validation", "duration_s": 1.0},
        {"name": "ring_000_charge", "kind": "ringdown_charge", "duration_s": 2.5,
         "meta": {"joint": 10}},
        {"name": "ring_000_release", "kind": "ringdown", "duration_s": 4.0,
         "meta": {"joint": 10}},
    ]


def _labels(durations, *, rate=150.0, gap_from=None):
    episode = np.concatenate([np.full(int(round(d * rate)), k) for k, d in enumerate(durations)])
    t = np.arange(episode.size) / rate
    if gap_from is not None:
        t = t + np.where(episode >= gap_from, 5.0, 0.0)
    return episode, t


def test_a_window_is_cut_from_its_named_segments_with_its_lead_in():
    ep, t = _labels([2, 1, 1, 1])
    spec = MF.WindowSpec("staircase", "s", "stair_101_4", "stair_101_vent", 1.0, 0.5, "")
    i0, i1, kinds = MF.resolve_rows(_plan(), ep, t, spec)
    assert i0 == 150, "a 1 s lead-in starts 150 rows before the 4 psi step"
    assert i1 == 450 + 76, "0.5 s kept of the vent, both ends inclusive"
    assert kinds == ("rest", "staircase")


def test_the_planner_refuses_held_out_rows_gaps_and_an_overreaching_lead_in():
    plan = _plan()
    ep, t = _labels([2, 1, 1, 1])
    with pytest.raises(ValueError, match="held-out"):
        MF.resolve_rows(plan, ep, t, MF.WindowSpec(
            "staircase", "s", "stair_101_vent", "validation_0", 0.0, None, ""))
    with pytest.raises(ValueError, match="held-out"):
        MF.resolve_rows(plan, ep, t, MF.WindowSpec(
            "validation", "s", "validation_0", "validation_0", 0.0, None, ""))
    ep_gap, t_gap = _labels([2, 1, 1, 1], gap_from=2)
    with pytest.raises(ValueError, match="contiguous"):
        MF.resolve_rows(plan, ep_gap, t_gap, MF.WindowSpec(
            "staircase", "s", "stair_101_4", "stair_101_vent", 0.0, None, ""))
    with pytest.raises(ValueError, match="reaches past"):
        MF.resolve_rows(plan, ep, t, MF.WindowSpec(
            "staircase", "s", "stair_101_vent", "stair_101_vent", 1.5, None, ""))
    with pytest.raises(ValueError, match="not recorded"):
        MF.resolve_rows(plan, ep, t, MF.WindowSpec(
            "ringdown", "s", "ring_000_charge", "ring_000_release", 0.0, None, ""))


def test_episode_labels_must_be_the_plans_segment_indices():
    plan = _plan()
    ep, _ = _labels([2, 1, 1, 1])
    phase = np.array([plan[k]["kind"] for k in ep], dtype=object)
    MF.check_labels(plan, ep, phase)
    shifted = phase.copy()
    shifted[ep == 3] = "staircase"
    with pytest.raises(ValueError, match="not segment indices"):
        MF.check_labels(plan, ep, shifted)


def test_staircases_and_ringdowns_are_found_by_their_metadata():
    plan = _plan()
    specs = MF.staircase_specs(plan, [(0x101, 0x105)] * 12)
    assert [(s.first, s.last) for s in specs] == [("stair_101_4", "stair_101_vent")] * 3
    rings = MF.ringdown_candidates(plan)
    assert rings[(2, False)] == [("ring_000_charge", "ring_000_release")]
    assert all(not v for k, v in rings.items() if k != (2, False))


def test_no_fitted_window_is_a_held_out_family():
    assert "validation" in MF.HELDOUT_KINDS
    assert not set(MF.FAMILIES) & set(MF.HELDOUT_KINDS)
    for spec in MF.fixed_specs():
        assert spec.family in MF.FAMILIES
        assert "validation" not in spec.first + spec.last


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def test_profile_lines_scale_exactly_what_they_name():
    best = MF.start_values()
    cands = MF.profile_candidates(best)
    assert [c[0] for c in cands].count("best") == 1
    assert len(cands) == 1 + 6 * 4 + 4 + 4 + 2
    for line, k, v, extra in cands:
        if line in MF.MASS_NAMES:
            assert v[line] == pytest.approx(best[line] * k)
            assert all(v[n] == best[n] for n in MF.NAMES if n != line)
        elif line == "mass_and_gain":
            assert all(v[n] == pytest.approx(best[n] * k) for n in MF.MASS_NAMES + MF.GAIN_NAMES)
            assert all(v[n] == best[n] for n in MF.SHAPE_NAMES + MF.DISSIPATION_NAMES)
            assert extra == {}
        elif line == "mass_gain_and_all_dissipation":
            assert extra == {"damp_b1_scale": k}
            assert all(v[n] == pytest.approx(best[n] * k) for n in MF.DISSIPATION_NAMES)
        elif line == "joint_armature":
            assert extra == {"joint_armature": pytest.approx(MG.JOINT_ARMATURE * k)}


def test_a_flat_profile_is_not_pinned_and_a_valley_is():
    cands = ([("best", 1.0, {}, {})]
             + [("flat", k, {}, {}) for k in (0.5, 0.75, 1.5, 2.0)]
             + [("valley", k, {}, {}) for k in (0.5, 0.75, 1.5, 2.0)])
    totals = np.array([5.0, 5.0, 5.0, 5.0, 5.001, 7.0, 5.5, 5.4, 6.0])
    summary = MF.summarise_profiles(cands, totals, {"chirp": totals})
    assert summary["lines"]["flat"]["pinned"] == "not pinned"
    assert summary["lines"]["flat"]["within_1pct_factor_range"] == [0.5, 2.0]
    assert summary["lines"]["valley"]["pinned"] == "both sides"
    lo, hi = summary["lines"]["valley"]["within_1pct_factor_range"]
    assert 0.75 < lo < 1.0 < hi < 1.5


# ---------------------------------------------------------------------------
# The checkpoint
# ---------------------------------------------------------------------------

def test_the_checkpoint_loads_through_twin_params_as_the_twin_that_was_scored(tmp_path):
    rng = np.random.default_rng(7)
    values = MF.from_unit(rng.uniform(0.1, 0.9, MF.N_PARAMS))
    path = tmp_path / "canarm_mech.json"
    MF.write_json(str(path), MF.mech_document(values, status="test",
                                              provenance={"families": {"chirp": []}}))
    assert not (tmp_path / "canarm_mech.json.tmp").exists()

    kw = TP.load_twin_kwargs(flow=None, mech=path, log=lambda *_: None)
    mine = MF.twin_kwargs(values, kw["actuator"])
    assert set(kw) == set(mine)
    np.testing.assert_allclose(kw["actuator"].coeff, mine["actuator"].coeff, rtol=1e-15)
    np.testing.assert_allclose(kw["actuator"].bf, mine["actuator"].bf, rtol=1e-15)
    for key in ("tendon_damping", "joint_damping", "joint_frictionloss"):
        assert kw[key] == pytest.approx(mine[key], rel=1e-15)
    for key in ("link_mass_kg", "bracket_mass_kg"):
        np.testing.assert_allclose(kw[key], mine[key], rtol=1e-15)

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["date"] == MF.FIT_DATE == "2026-09-10"
    assert set(doc["bounds"]) == set(MF.NAMES)
    back = MF.values_from_mech(doc, l0=kw["actuator"].l0)
    np.testing.assert_allclose(MF.as_vector(back), MF.as_vector(values), rtol=1e-12)
