"""The audit must be arithmetic anybody can check, and must change nothing.

Two properties carry the module: the geometry part is closed form and is pinned
here against hand arithmetic, and a full run leaves the audited law bit-identical
so the audit stays evidence rather than another thing that adjusts the model.
The three parts that need a stepped model are driven through their seams by
analytic stand-ins, so what is NOT covered here is whether MuJoCo agrees with
those stand-ins -- stated in the report.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from digital_twin import force_audit as FA
from digital_twin import replay as R
from digital_twin.test_replay import SpringArm, synthetic_recording

#: A braid shaped like this arm's: about 0.19 m of muscle on the 265/234/230 mm
#: segments, with ``l0`` ABOVE the slack length ``Bf/sqrt(3) = 0.162 m`` so the
#: muscle pulls at rest.  Not a fit -- a fixture, chosen so every number below is
#: checkable by hand.
LAW = FA.ForceLaw(coeff=np.array([2.5e-3, 2.5e-3, 2.5e-3]),
                  bf=np.array([0.28, 0.28, 0.28]),
                  l0=np.array([0.19, 0.19, 0.19]),
                  source="test fixture, not a fit")


def test_the_force_law_is_pull_only_and_matches_hand_arithmetic():
    psi = 8.0
    p_pa = psi * R.PA_PER_PSI
    expected = 2.5e-3 * p_pa * (0.28 ** 2 - 3.0 * 0.19 ** 2)
    assert expected < 0.0            # 3 l0^2 exceeds Bf^2, so the muscle pulls
    got = FA.force_n(LAW, psi)
    assert got[0] == pytest.approx(expected)
    assert np.all(got <= 0.0)


def test_a_muscle_shorter_than_its_slack_length_carries_no_force():
    """The law crosses zero at Bf/sqrt(3); BELOW it the pull-only clip zeroes it.

    Above the slack length the muscle still pulls, which is why the test checks
    both sides -- a clip that zeroed the wrong half would leave the arm with
    twenty-four ropes that push.
    """
    slack = LAW.slack_len_m[0]
    assert slack == pytest.approx(0.28 / math.sqrt(3.0))
    assert np.all(FA.force_n(LAW, 8.0, l_m=np.full(3, slack * 0.95)) == 0.0)
    assert np.all(FA.force_n(LAW, 8.0, l_m=np.full(3, slack * 1.05)) < 0.0)


def test_geometry_reads_the_braid_back_out_of_the_coefficient():
    g = FA.audit_geometry(LAW)
    nf = math.sqrt(1.0 / (4.0 * math.pi * 2.5e-3))
    assert g["turns_nf"][0] == pytest.approx(nf)
    assert g["braid_diameter_zero_angle_m"][0] == pytest.approx(0.28 / (math.pi * nf))
    assert g["braid_angle_rest_deg"][0] == pytest.approx(
        math.degrees(math.acos(0.19 / 0.28)))
    assert g["slack_length_m"][0] == pytest.approx(0.28 / math.sqrt(3.0))


def test_geometry_checks_the_muscles_against_the_ring_they_are_seated_on():
    """Four muscles share each anchor ring; above one they do not fit inside it."""
    g = FA.audit_geometry(LAW)
    d_rest = g["rest_diameter_m"][0]
    expected = (4.0 * math.pi * (d_rest / 2.0) ** 2) / (math.pi * 0.028 ** 2)
    assert g["overfill_on_AO_ring"][0] == pytest.approx(expected)
    assert g["anchor_radii_m"]["AO1"] == 0.028      # not the RS485 arm's 0.030
    assert g["anchor_radii_m"]["JA1"] == 0.047      # not its 0.100/0.050


def test_force_and_stiffness_cannot_be_scaled_apart():
    """The lever is geometry alone, so no coefficient fixes one without moving the other."""
    g = FA.audit_geometry(LAW)
    lever = g["force_over_stiffness_lever_m"][0]
    doubled = FA.audit_geometry(FA.scaled(LAW, 2.0))
    assert doubled["force_over_stiffness_lever_m"][0] == pytest.approx(lever)
    assert doubled["force_at_8psi_n"][0] == pytest.approx(2.0 * g["force_at_8psi_n"][0])


def test_scaling_returns_a_new_law_and_leaves_the_original_alone():
    before = LAW.coeff.copy()
    FA.scaled(LAW, 4.0)
    assert np.array_equal(LAW.coeff, before)


def test_a_full_run_changes_nothing_and_repeats_exactly():
    before = (LAW.coeff.copy(), LAW.bf.copy(), LAW.l0.copy())
    a = FA.audit(LAW)
    b = FA.audit(LAW)
    assert a["model_unmodified"] is True
    assert a == b
    for was, now in zip(before, (LAW.coeff, LAW.bf, LAW.l0)):
        assert np.array_equal(was, now)


def test_a_part_without_its_seam_says_so_rather_than_vanishing():
    out = FA.audit(LAW)
    for part in ("static", "ring", "replay"):
        assert out[part]["available"] is False
        assert out[part]["reason"]
    assert out["geometry"]["available"] is True
    assert out["clamp"]["available"] is True


def test_the_clamp_headroom_is_reported_closed_form_and_measured():
    rec = synthetic_recording(n=200, drive_psi=8.0)
    roll = R.rollout(rec, arm_factory=SpringArm)
    out = FA.audit_clamp(LAW, roll=roll)
    at_x1 = [row for row in out["closed_form"] if row["scale"] == 1.0][0]
    assert at_x1["peak_force_n"] == pytest.approx(
        float(FA.force_n(LAW, FA.ENVELOPE_PSI)[0]))
    assert at_x1["headroom_n"] == pytest.approx(R.FORCE_CLIP_N + at_x1["peak_force_n"])
    assert at_x1["touches_clip"] is False
    assert out["measured"]["clamped"] is False
    assert out["measured"]["ctrl_min_n"] == roll.ctrl_min_n


def test_a_law_that_would_saturate_the_clip_is_flagged():
    strong = FA.scaled(LAW, 2000.0)
    out = FA.audit_clamp(strong)
    assert all(row["touches_clip"] for row in out["closed_form"])
    assert out["measured"] is None and out["reason_no_measurement"]


def test_savgol_recovers_the_derivatives_of_a_known_signal():
    """Raw differencing is unusable on mocap angles; the filter has to be right."""
    fs, f = 150.0, 1.0
    t = np.arange(0.0, 6.0, 1.0 / fs)
    y = np.sin(2.0 * np.pi * f * t)[:, None]
    tt, d1, d2 = FA.savgol_derivatives(t, y, skip_s=0.5)
    w = 2.0 * np.pi * f
    # Tolerances stated against the signal's own peaks: 0.02 rad/s on a
    # 6.28 rad/s peak rate (0.3 %) and 1.0 rad/s^2 on a 39.5 rad/s^2 peak
    # acceleration (2.5 %).  The second derivative is the looser one because a
    # cubic over a 113 ms window curves less than a 1 Hz sine does.
    assert np.allclose(d1[:, 0], w * np.cos(w * tt), atol=0.02)
    assert np.allclose(d2[:, 0], -w * w * np.sin(w * tt), atol=1.0)


def test_savgol_drops_unposed_rows_and_the_start_transient():
    fs = 150.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    y = np.zeros((t.size, 2))
    y[10:20] = np.nan
    tt, d1, _ = FA.savgol_derivatives(t, y, skip_s=1.0)
    # The ten unposed rows fell inside the first second, so the skip already
    # covers them; what must hold is that neither a NaN nor the transient
    # survives.
    assert tt.size == t.size - int(round(1.0 * fs))
    assert tt[0] >= 1.0
    assert np.all(np.isfinite(d1))


def test_savgol_refuses_a_trace_shorter_than_its_own_window():
    t = np.arange(0.0, 0.05, 1.0 / 150.0)
    with pytest.raises(ValueError, match="Savitzky-Golay window"):
        FA.savgol_derivatives(t, np.zeros((t.size, 1)), skip_s=0.0)


def test_ring_separates_the_gravity_share_from_the_elastic_share():
    """f^2 is affine in the coefficient scale; the intercept is the pendulum."""
    fs = 150.0
    t = np.arange(0.0, 8.0, 1.0 / fs)
    slope_hz2, intercept_hz2 = 3.0, 1.0

    def poke_fn(law, poke_n):
        scale = float(law.coeff[0]) / float(LAW.coeff[0])
        f = math.sqrt(intercept_hz2 + slope_hz2 * scale)
        return t, (2.0 * np.exp(-0.5 * t) * np.cos(2.0 * np.pi * f * t))[:, None]

    out = FA.audit_ring(LAW, poke_fn=poke_fn, real_ring_hz=1.83)
    assert out["available"]
    assert out["f2_slope_hz2_per_scale"] == pytest.approx(slope_hz2, rel=0.05)
    assert out["f2_intercept_hz2"] == pytest.approx(intercept_hz2, rel=0.10)
    assert out["gravity_share_at_x1"] == pytest.approx(
        intercept_hz2 / (slope_hz2 + intercept_hz2), rel=0.10)
    assert out["scale_for_real_hz"] == pytest.approx(
        (1.83 ** 2 - intercept_hz2) / slope_hz2, rel=0.15)


def test_ring_refuses_to_invent_a_measured_frequency_for_this_arm():
    """This arm's ring frequency has not been measured; a default would be the RS485 arm's."""
    fs = 150.0
    t = np.arange(0.0, 6.0, 1.0 / fs)

    def poke_fn(law, poke_n):
        return t, (2.0 * np.exp(-0.5 * t) * np.cos(2.0 * np.pi * 1.9 * t))[:, None]

    out = FA.audit_ring(LAW, poke_fn=poke_fn)
    assert out["scale_for_real_hz"] is None
    assert "has not been measured" in out["reason_no_scale"]


def test_static_reports_how_blind_a_pressure_versus_angle_campaign_is():
    """A sixteen-fold coefficient sweep that barely moves the angle is the finding."""
    def settle_fn(law, psi, node):
        # Geometry-limited: the settled angle saturates, so the coefficient
        # barely shows.  A force-limited plant would be linear in it.
        s = float(law.coeff[0]) / float(LAW.coeff[0])
        return 12.0 * s / (s + 0.05)

    def force_limited(law, psi, node):
        return 12.0 * float(law.coeff[0]) / float(LAW.coeff[0])

    blind = FA.audit_static(LAW, settle_fn=settle_fn, scales=(0.25, 4.0),
                            psis=(8.0,), nodes=(1,))
    seeing = FA.audit_static(LAW, settle_fn=force_limited, scales=(0.25, 4.0),
                             psis=(8.0,), nodes=(1,))
    assert blind["available"]
    # A sixteen-fold sweep moves the geometry-limited settle by 17 % of its own
    # mean, against 176 % for a plant whose settle is set by the force.  That
    # contrast IS the finding, and it is what the verdict states.
    assert blind["rows"][0]["spread_frac"] < 0.25
    assert seeing["rows"][0]["spread_frac"] > 1.5
    assert "geometry-limited" in blind["verdict"]


def test_replay_compares_peak_acceleration_at_two_coefficient_scales():
    rec = synthetic_recording(n=1200, drive_psi=8.0)

    def rollout_fn(law, r):
        s = float(law.coeff[0]) / float(LAW.coeff[0])
        return r.t_rel_s, s * np.degrees(r.q_rad)

    out = FA.audit_replay(LAW, rec, rollout_fn=rollout_fn)
    assert out["available"]
    at_half = [row for row in out["rows"] if row["scale"] == 0.5][0]
    at_one = [row for row in out["rows"] if row["scale"] == 1.0][0]
    assert at_one["qddot_ratio_median"] == pytest.approx(1.0, rel=1e-6)
    assert at_half["qddot_ratio_median"] == pytest.approx(0.5, rel=1e-6)
    assert out["window_ms"] == pytest.approx(FA.SAVGOL_WINDOW / 150.0 * 1000.0, rel=0.02)


def test_replay_refuses_a_cached_twin():
    """An algorithm-2 cache scored the RS485 twin four times too slow -- the opposite answer."""
    rec = synthetic_recording(n=100)
    out = FA.audit_replay(LAW, rec, rollout_fn=None)
    assert out["available"] is False
    assert "cached twin" in out["reason"]


def test_force_law_from_reads_a_model_by_attribute():
    class Fake:
        coeff = np.array([1e-3, 2e-3, 3e-3])
        bf = np.array([0.3, 0.3, 0.3])
        l0 = np.array([0.2, 0.2, 0.2])

    law = FA.force_law_from(Fake())
    assert law.coeff.shape == (3,) and law.bf.shape == (3,) and law.l0.shape == (3,)
    assert law.source == "Fake"
