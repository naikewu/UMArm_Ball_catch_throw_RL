"""Alignment first, then the metrics, then the population split.

The alignment tests use a synthetic recording whose answer is known by
construction -- the rollout is driven on the recording's own sync stamps, so the
correspondence is the identity and any deviation from it is a defect rather than
a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin import replay as R
from digital_twin import twin_compare as TC
from digital_twin.test_replay import PlaybackArm, SpringArm, synthetic_recording


def idle_then_drive(*, n=900, idle_rows=200):
    """A recording that holds at the idle setpoint before it drives.

    The idle hold is what :func:`pick_reference_rows` is supposed to find, so a
    recording without one would test the fallback instead of the intended path.
    """
    rec = synthetic_recording(n=n, step_row=idle_rows)
    return rec


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_alignment_is_exact_on_a_recording_whose_answer_is_known():
    rec = idle_then_drive()
    roll = R.rollout(rec, arm_factory=SpringArm)
    al = TC.align_on_sync(rec, roll)
    assert al.exact is True
    assert al.max_dt_s == 0.0
    assert al.n_unmatched == 0
    assert np.array_equal(al.rec_rows, np.arange(rec.n))
    assert np.array_equal(al.roll_rows, np.arange(rec.n))


def test_the_tail_is_not_matched_to_a_recording_row():
    rec = idle_then_drive(n=300)
    roll = R.rollout(rec, arm_factory=SpringArm, tail_s=1.0)
    al = TC.align_on_sync(rec, roll)
    assert np.array_equal(al.rec_rows, np.arange(rec.n))
    assert int(al.roll_rows.max()) == rec.n - 1


def test_a_truncated_rollout_drops_the_rows_it_never_reached():
    rec = idle_then_drive(n=600)
    roll = R.rollout(rec, arm_factory=SpringArm, t_max_s=1.0)
    al = TC.align_on_sync(rec, roll)
    reached = int(roll.row_index.max()) + 1
    assert al.rec_rows.size == reached
    assert al.n_unmatched == rec.n - reached
    assert al.exact is True


def test_a_rollout_on_a_nominal_grid_fails_the_exactness_test():
    """The check that would catch a replay quietly resampled onto 6.667 ms."""
    rec = idle_then_drive(n=200)
    roll = R.rollout(rec, arm_factory=SpringArm)
    nominal = R.Rollout(
        t_s=rec.t_sync_s[0] + np.arange(rec.n) / 150.0, row_index=roll.row_index,
        target_pa=roll.target_pa, p_pa=roll.p_pa, q_rad=roll.q_rad,
        ctrl_n=roll.ctrl_n, ctrl_min_n=roll.ctrl_min_n, clamped=False, meta={})
    al = TC.align_on_sync(rec, nominal)
    assert al.rec_rows.size < rec.n      # nothing matches at zero tolerance
    al_loose = TC.align_on_sync(rec, nominal, tol_s=2.0e-3)
    assert al_loose.exact is False and al_loose.max_dt_s > 0.0


# ---------------------------------------------------------------------------
# Reference selection
# ---------------------------------------------------------------------------


def test_the_reference_is_the_pre_drive_idle_hold():
    rec = idle_then_drive(n=900, idle_rows=200)
    rows, how = TC.pick_reference_rows(rec)
    assert "idle hold" in how
    assert rows.size >= 20
    assert int(rows.max()) < 200


def test_a_session_that_drives_from_row_zero_falls_back_to_the_end():
    rec = synthetic_recording(n=400, step_row=0)
    rows, how = TC.pick_reference_rows(rec)
    assert "last" in how
    assert int(rows.min()) > 200        # the stiller end, not the pressurising start


def test_both_references_average_over_the_same_rows():
    rec = idle_then_drive()
    tw = TC.twin_rollout(rec, arm_factory=SpringArm)
    assert tw.valid
    # The rows are the intersection, so both means are over exactly these.
    assert tw.ref_rows.size == tw.meta["n_ref_rows"]
    sim_ref = np.mean([tw.q_sim_rad[np.searchsorted(tw.rec_rows, r)]
                       for r in tw.ref_rows], axis=0)
    assert np.allclose(sim_ref, tw.sim_ref_rad)
    real_ref = rec.q_rad[tw.ref_rows].mean(axis=0)
    assert np.allclose(real_ref, tw.real_ref_rad)


def test_a_reference_window_with_no_rollout_samples_is_invalid_not_re_referenced():
    rec = idle_then_drive(n=900, idle_rows=200)
    # Truncate to half a second: the idle reference window is inside it, so
    # force the fallback by asking for a window past the truncation instead.
    tw = TC.twin_rollout(rec, arm_factory=SpringArm, t_max_s=0.5,
                         explicit_ref_window_s=(4.0, 4.5))
    assert tw.valid is False
    assert "reference window has no rollout samples" in tw.reason


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_a_perfect_twin_scores_zero_and_a_null_twin_scores_one():
    rec = idle_then_drive(n=600)
    perfect = TC.twin_rollout(rec, arm=PlaybackArm(rec.q_rad, rec.p_pa))
    m = TC.compare_metrics(rec, perfect)
    assert m["valid"] and m["overall"]["scored"]
    assert m["overall"]["joints"]["rms_deg_mean"] == pytest.approx(0.0, abs=1e-12)
    assert m["overall"]["joints"]["nrmse_mean"] == pytest.approx(0.0, abs=1e-9)

    # A twin frozen at the recording's mean is the null NRMSE is defined against:
    # it predicts the arm no better than "it does not move".  It lands NEAR 1.0
    # rather than exactly on it because both traces are referenced to the
    # settled window's mean, not to the scored window's mean, so the numerator
    # carries that offset while the denominator does not -- 1 % here.
    frozen = np.repeat(rec.q_rad.mean(axis=0)[None, :], rec.n, axis=0)
    null = TC.twin_rollout(rec, arm=PlaybackArm(frozen, rec.p_pa))
    mn = TC.compare_metrics(rec, null)
    assert mn["overall"]["joints"]["nrmse_mean"] == pytest.approx(1.0, rel=1e-2)


def test_metrics_are_reported_per_joint_and_per_board():
    rec = idle_then_drive(n=600)
    tw = TC.twin_rollout(rec, arm_factory=SpringArm)
    m = TC.compare_metrics(rec, tw)
    j = m["overall"]["joints"]
    b = m["overall"]["boards"]
    assert len(j["rms_deg"]) == R.N_JOINTS and len(j["nrmse"]) == R.N_JOINTS
    assert len(b["rms_pa"]) == R.N_NODES and len(b["ids"]) == R.N_NODES
    assert b["ids"][0] == 0x101 and b["ids"][-1] == 0x118


def test_the_population_split_comes_from_the_variant_byte_not_the_id_range():
    """A TLE board at 0x114 must be scored as a TLE board."""
    rec = idle_then_drive(n=400)
    labels = np.array([("tle" if v else "7mm") for v in rec.is_tle], dtype=object)
    col_114 = int(np.flatnonzero(rec.ids == 0x114)[0])
    assert labels[col_114] == "tle"

    tw = TC.twin_rollout(rec, arm_factory=SpringArm)
    m = TC.compare_metrics(rec, tw)
    by_pop = m["overall"]["boards"]["by_population"]
    assert set(by_pop) == {"tle", "7mm"}
    assert by_pop["tle"]["n_columns"] == int(np.count_nonzero(rec.is_tle)) == 9
    assert by_pop["7mm"]["n_columns"] == R.N_NODES - 9


def test_a_joint_whose_antagonists_straddle_the_populations_is_marked_mixed():
    """Assigning it to one population would put a TLE board's error in the 7 mm column."""
    rec = idle_then_drive(n=200)
    labels = TC.joint_population(rec)
    assert set(labels) <= {"tle", "7mm", "mixed", "unknown"}
    # 0x114 is TLE while its partner 0x112 is not, so that joint is mixed.
    assert "mixed" in set(labels)


def test_a_window_with_too_few_samples_returns_no_number():
    rec = idle_then_drive(n=600)
    tw = TC.twin_rollout(rec, arm_factory=SpringArm)
    m = TC.compare_metrics(rec, tw, segments=[("sliver", 0.0, 0.05)])
    seg = m["segments"][0]
    assert seg["scored"] is False and "fewer than" in seg["reason"]


def test_a_clamp_violation_returns_an_invalid_result_rather_than_a_score():
    rec = synthetic_recording(n=400, drive_psi=30.0)
    tw = TC.twin_rollout(rec, arm_factory=lambda: SpringArm(ctrl_scale_n_per_pa=0.03))
    assert tw.valid is False and "clamp violation" in tw.reason
    m = TC.compare_metrics(rec, tw)
    assert m["valid"] is False


# ---------------------------------------------------------------------------
# The ring instrument runs by the same code path on both traces
# ---------------------------------------------------------------------------


def test_the_same_signal_scores_identically_as_a_recording_and_as_a_rollout():
    """The rule ring_analysis exists to enforce, asserted end to end.

    The twin's trace here IS the recording's trace, bit for bit, so any
    difference the ring comparison reported would be the instrument disagreeing
    with itself rather than two plants disagreeing.
    """
    n, fs = 1500, 150.0
    t = 100.0 + np.arange(n) / fs
    q = np.zeros((n, R.N_JOINTS))
    tt = t - t[0]
    q[:, 3] = np.radians(3.0 * np.exp(-0.62 * tt) * np.cos(2.0 * np.pi * 1.8 * tt))
    rec = R.recording_from_arrays(t, q, np.zeros((n, R.N_NODES)),
                                  np.zeros((n, R.N_NODES)),
                                  [R.VARIANT_TLE_DVP] * 8 + [R.VARIANT_7MM] * 16)
    tw = TC.twin_rollout(rec, arm=PlaybackArm(rec.q_rad))
    assert tw.valid
    assert np.array_equal(tw.twin_defl_rad, tw.real_defl_rad)
    ring = TC.ring_compare(tw)
    assert ring["available"]
    assert ring["real"] == ring["twin"]
    assert ring["real"]["n_trusted"] >= 1
    assert ring["real"]["median_freq_hz"] == pytest.approx(1.8, rel=0.05)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_a_custom_rollout_never_gets_the_canonical_cache_key():
    rec = idle_then_drive(n=100)
    assert TC.cache_key(rec, custom=True) is None
    assert TC.cache_key(rec, custom=False).startswith(f"algo{TC.ALGO_VERSION}|")


# ---------------------------------------------------------------------------
# End to end, against the real twin
# ---------------------------------------------------------------------------


def _real_arm_factory(**kwargs):
    sim_core = pytest.importorskip("digital_twin.sim_core",
                                   reason="sim_core is a sibling module under construction")
    try:
        return sim_core.SimArm(**kwargs)
    except Exception as exc:               # a mid-edit sibling is not this test's failure
        pytest.skip(f"SimArm did not build: {exc!r}")


def test_a_real_rollout_scores_and_reports_both_populations():
    """The whole chain: recording -> SimArm -> alignment -> per-joint, per-board, per-population.

    The NUMBERS are not asserted, deliberately -- nothing on this arm has been
    fitted yet, so an assertion on a residual would pin an unfitted model.  What
    is asserted is that the chain produces a complete, finite report card with
    the alignment exact.
    """
    rec = idle_then_drive(n=400, idle_rows=100)
    tw = TC.twin_rollout(rec, arm_factory=_real_arm_factory)
    assert tw.valid, tw.reason
    assert tw.meta["alignment_exact"] is True
    m = TC.compare_metrics(rec, tw)
    assert m["overall"]["scored"] is True
    assert np.all(np.isfinite(m["overall"]["joints"]["rms_deg"]))
    assert np.all(np.isfinite(m["overall"]["boards"]["rms_pa"]))
    assert set(m["overall"]["boards"]["by_population"]) == {"tle", "7mm"}
