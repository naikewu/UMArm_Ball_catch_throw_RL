"""The instrument has to be pinned before anything measured with it means much.

Every test here builds a signal whose frequency and damping ratio are known by
construction, so a failure says the estimator moved rather than that the arm did.
No hardware, no MuJoCo, no sibling module.
"""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin import ring_analysis as RA

FS_HZ = 150.0          # this arm's CAN sync rate
TRUE_F_HZ = 1.8        # near the RS485 arm's measured 1.83 Hz median
TRUE_ZETA = 0.055      # near its 0.054 median


def damped_ring(*, f_hz=TRUE_F_HZ, zeta=TRUE_ZETA, amp_deg=3.0, fs_hz=FS_HZ,
                dur_s=8.0, offset_deg=0.0, trend_deg_s=0.0, t0_s=0.0):
    """A ringdown that starts at ``t0_s`` on top of an offset and a linear trend.

    ``t0_s`` defaults to zero so the trace IS the episode.  A non-zero onset is
    used only where the test is about detection, because a hard onset is the one
    case the zero-phase bandpass handles badly -- see
    ``test_a_hard_onset_biases_the_detected_window_low``.
    """
    t = np.arange(0.0, dur_s, 1.0 / fs_hz)
    omega = 2.0 * np.pi * f_hz
    sigma = zeta / np.sqrt(1.0 - zeta ** 2) * omega
    y = offset_deg + trend_deg_s * t
    live = t >= t0_s
    tt = t[live] - t0_s
    y[live] += amp_deg * np.exp(-sigma * tt) * np.cos(omega * tt)
    return t, y, sigma


def test_fit_recovers_a_known_frequency_and_damping_ratio():
    # Tolerances stated rather than tuned: 1 % on frequency and 15 % on the
    # damping ratio.  Damping is the looser of the two because it is read from
    # the decay envelope, which a linear trend and a finite window both bias,
    # whereas frequency is read from the zero crossings.
    t, y, _ = damped_ring()
    fit = RA.fit_damped_sine(t, y, f0=TRUE_F_HZ)
    assert fit.freq_hz == pytest.approx(TRUE_F_HZ, rel=0.01)
    assert fit.zeta == pytest.approx(TRUE_ZETA, rel=0.15)
    assert fit.r2 > 0.9


def test_blind_fit_finds_the_same_ring_without_a_seed():
    """The twin is always fitted blind, so the blind path must stand on its own."""
    t, y, _ = damped_ring()
    seeded = RA.fit_damped_sine(t, y, f0=TRUE_F_HZ)
    blind = RA.fit_damped_sine(t, y, f0=None)
    assert blind.seeded is False and seeded.seeded is True
    assert blind.freq_hz == pytest.approx(seeded.freq_hz, rel=0.02)
    assert blind.zeta == pytest.approx(seeded.zeta, rel=0.25)


def test_r2_refuses_to_certify_a_ring_that_is_not_there():
    """A pure drift must not score well.

    This is the whole reason ss_tot is the DETRENDED residual's variance: against
    the raw variance a straight line fits its own trend and reads r2 about 1.
    """
    t = np.arange(0.0, 6.0, 1.0 / FS_HZ)
    fit = RA.fit_damped_sine(t, 2.0 + 0.5 * t)
    assert fit.r2 < 0.5


def test_a_growing_oscillation_is_not_reported_as_negative_damping():
    t = np.arange(0.0, 6.0, 1.0 / FS_HZ)
    y = np.exp(0.4 * t) * np.cos(2.0 * np.pi * 2.0 * t)
    fit = RA.fit_damped_sine(t, y, f0=2.0)
    assert fit.sigma >= 0.0 and fit.zeta >= 0.0


def test_episode_detection_finds_the_ring_and_pads_the_window():
    t, y, _ = damped_ring(t0_s=1.5, dur_s=10.0)
    eps = RA.episodes_from_trace(t, y, thresh=0.30)
    assert len(eps) >= 1
    ep = eps[0]
    assert ep.joint == 0
    assert ep.period_s == pytest.approx(1.0 / TRUE_F_HZ, rel=0.05)
    # The padded window is at least as wide as the detected one on both sides.
    assert ep.a0 <= ep.i0 and ep.a1 >= ep.i1
    assert ep.fit is not None
    assert ep.fit.freq_hz == pytest.approx(TRUE_F_HZ, rel=0.05)


def test_a_hard_onset_biases_the_detected_window_low():
    """A measured limit of the instrument, asserted so it cannot drift unnoticed.

    ``filtfilt`` is zero-phase, so a step onset smears the envelope BACKWARDS by
    the filter's tail and the detected window opens before the ring did.  The
    flat lead-in then pulls the fitted decay down: on this synthetic the damping
    ratio reads about 0.022 against a true 0.055.  The fit is exact on the true
    window, so the bias is the detector's and not the fitter's -- and it is
    common-mode in ``fit_bounce``, where both traces are windowed on the same
    ``[t0, t1]``.  What this does NOT show is that an absolute damping ratio
    quoted from a detected episode is trustworthy; it is trustworthy only as a
    comparison against another trace measured the same way.
    """
    t, y, _ = damped_ring(t0_s=1.0, dur_s=8.0)
    exact = (t >= 1.0) & (t <= 5.0)
    on_true_window = RA.fit_damped_sine(t[exact], y[exact], f0=TRUE_F_HZ)
    assert on_true_window.zeta == pytest.approx(TRUE_ZETA, rel=0.02)

    ep = RA.episodes_from_trace(t, y, thresh=0.30)[0]
    assert ep.t0 < 1.0                    # the window opened before the ring did
    assert ep.fit.zeta < 0.6 * TRUE_ZETA  # and the decay reads low because of it


def test_episodes_are_sorted_loudest_first():
    """A caller keeping the first few must keep the loudest, not the earliest."""
    t = np.arange(0.0, 10.0, 1.0 / FS_HZ)
    q = np.zeros((t.size, 3))
    for j, amp in enumerate((0.8, 4.0, 2.0)):
        _, y, _ = damped_ring(amp_deg=amp, dur_s=10.0)
        q[:, j] = y
    eps = RA.episodes_from_trace(t, q, thresh=0.30)
    firsts = []
    for e in eps:
        if e.joint not in firsts:
            firsts.append(e.joint)
    assert firsts[0] == 1  # the 4 deg ring, not joint 0 which comes first in time


def test_the_same_array_gives_bit_identical_answers():
    """No hidden state, no RNG: the instrument must be a pure function of its input."""
    t, y, _ = damped_ring()
    a = RA.episodes_from_trace(t, y, thresh=0.30)
    b = RA.episodes_from_trace(t, y, thresh=0.30)
    assert [e.as_dict() for e in a] == [e.as_dict() for e in b]


def test_sample_rate_uses_the_median_gap_not_the_mean():
    """One 25 ms stall in a 6.667 ms recording must not move the derived rate."""
    t = np.arange(0.0, 2.0, 1.0 / FS_HZ)
    t[100:] += 0.025          # a single dropped-cycle stall, as the timing summary reports
    assert RA.sample_rate_hz(t) == pytest.approx(FS_HZ, rel=1e-9)


def test_nan_gaps_are_reported_not_hidden():
    t = np.arange(0.0, 2.0, 1.0 / FS_HZ)
    y = np.sin(2.0 * np.pi * 2.0 * t)
    y[10:20] = np.nan
    filled, frac = RA.interp_nans(t, y)
    assert np.all(np.isfinite(filled))
    assert frac == pytest.approx(10.0 / t.size)


def test_a_mostly_missing_joint_is_blanked_rather_than_scored():
    t, y, _ = damped_ring()
    q = np.column_stack([y, y.copy()])
    q[: int(0.5 * t.size), 1] = np.nan       # half the second joint is gone
    eps = RA.episodes_from_trace(t, q, thresh=0.30, max_nan_frac=0.2)
    assert {e.joint for e in eps} == {0}


def test_a_trace_shorter_than_the_filter_pad_raises_naming_its_length():
    t = np.arange(0.0, 0.1, 1.0 / FS_HZ)
    with pytest.raises(ValueError, match="samples"):
        RA.bandpass(np.zeros(t.size), FS_HZ)


def test_a_band_above_nyquist_is_pulled_in_rather_than_crashing():
    """A 20 Hz recording cannot carry a 12 Hz edge; that is the recording's limit, not an error."""
    t = np.arange(0.0, 20.0, 1.0 / 20.0)
    out = RA.bandpass(np.sin(2.0 * np.pi * 1.8 * t), 20.0)
    assert np.all(np.isfinite(out))


def test_ring_metrics_separates_found_from_trusted():
    t, y, _ = damped_ring()
    eps = RA.episodes_from_trace(t, y, thresh=0.30)
    m = RA.ring_metrics(eps)
    assert m["n_episodes"] >= 1
    assert m["n_trusted"] >= 1
    assert m["median_freq_hz"] == pytest.approx(TRUE_F_HZ, rel=0.05)
    assert m["median_zeta"] == pytest.approx(TRUE_ZETA, rel=0.25)


def test_band_rms_is_defined_for_a_dead_trace():
    """A dead twin must still produce a number, since the loss takes its logarithm."""
    t = np.arange(0.0, 4.0, 1.0 / FS_HZ)
    assert RA.band_rms(t, np.zeros(t.size), 0.5, 3.5) == pytest.approx(0.0, abs=1e-12)
