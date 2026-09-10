r"""The ringdown instrument: bandpass, episode finder, damped-sine fit.

Three functions and one rule.  The rule is that the same code runs on the
recording and on the rollout, so that "the twin rings at the wrong frequency"
cannot be an artefact of two different analyses.  Every caller in this package
enters through :func:`episodes_from_trace`, which is the single entry point that
derives the sample rate the same way for both traces; a caller that computes its
own ``fs`` and calls :func:`find_episodes` directly has already broken the rule.

WHY THE MODULE HAS NO REPO IMPORTS AND NO I/O.  It is handed arrays.  That is
what lets it run unchanged on a JSONL recording, on a :mod:`digital_twin.replay`
rollout and on a synthetic trace in a test, and it is why the comparison in
:mod:`digital_twin.fit_bounce` is a statement about two plants rather than about
two estimators.

WHAT THE RS485 ARM'S NUMBERS BOUGHT, and why they are defaults rather than
constants here.  That arm rings after every step -- 53 measured episodes, median
1.83 Hz, median damping ratio 0.054 -- and finding that out is what showed its
inherited damping constants left every mode critically-to-over damped (poke test
zeta 0.73), i.e. that the twin could not sustain the oscillation the metal shows
at any parameterisation of the rest of the model.  The CAN arm is longer and
heavier (u-joint-centre spans 265.36 / 234.37 / 229.92 mm against the RS485
arm's 225 / 194 / 185 mm) and its ring frequency has not been measured here, so
a lower ring is expected.  Every gate below is therefore an argument with a
named default, not a module constant read at call time: copying a 1.5 Hz low
edge onto a 1.2 Hz arm reports an arm that barely rings, and the filter that did
it leaves no trace in the result.

THE SAMPLE RATE IS THE OTHER OPEN QUESTION.  The RS485 bounce fit needed 160 Hz
recordings to resolve a 1.83 Hz / zeta 0.054 ring and its 20 Hz campaign was
blind to it; the CAN bus syncs at 150 Hz, close enough to that edge that
:func:`episodes_from_trace` records the rate it derived on every episode rather
than leaving it implicit.

Ported from ``C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\
UMArm_SIM\ring_analysis.py`` via ``reference/mjcf_fit.md`` section 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, hilbert

# ---------------------------------------------------------------------------
# Gate defaults.  Every one of these is a default, never read at call time.
# ---------------------------------------------------------------------------

#: Passband, Hz.  The low edge sits just under the RS485 arm's measured 1.7-2.0
#: Hz structural mode on purpose, to reject the ~1 Hz step-settle content that
#: otherwise reads as a very slow, very damped ring.  The in-band attenuation
#: the filter costs is tolerated because the damped-sine fit runs on the RAW
#: segment; the bandpass only decides *where* to look.
BAND_HZ = (1.5, 12.0)

#: Butterworth order.  Three is what the RS485 instrument used; raising it
#: sharpens the low edge and lengthens ``filtfilt``'s edge transient, which
#: costs episodes at the start of a recording where the arm is still settling.
BAND_ORDER = 3

#: An episode must hold this many cycles of its OWN dominant period.  Below
#: about two and a half cycles the damped-sine fit cannot separate the decay
#: from the linear trend, and the resulting damping ratio is a fit of the settle.
MIN_CYCLES = 2.5

#: Envelope threshold, in the caller's own units.  0.25 deg is the RS485 arm's
#: mocap plate jitter floor; anything below it is the measurement, not the arm.
#: The CAN arm's plate jitter has not been measured on this rig, so a caller
#: working in degrees should re-derive this rather than inherit it.
THRESH_DEG = 0.25

#: An episode shorter than this many samples is refused outright, before any
#: period arithmetic, because the zero-crossing count below is meaningless on a
#: handful of points.
MIN_SAMPLES = 8

#: r2 below which a fit is not trusted to carry a frequency or a damping ratio.
#: See :class:`RingFit` for why this reads far lower than an ordinary r2.
R2_TRUST = 0.30


@dataclass(frozen=True)
class RingFit:
    """One damped-sine fit: ``offset + trend*t + amp*exp(-sigma*t)*cos(2 pi f t + phase)``.

    The linear term is not cosmetic.  Nearly every real episode sits inside a
    post-step settle, and without a trend the fit spends its decay budget
    tracking the settle and reports a damping ratio that belongs to the
    controller rather than to the structure.

    ``r2`` IS RING-BEYOND-TREND, NOT THE USUAL MEAN-REFERENCED r2.  ``ss_tot``
    is the variance of the *detrended* residual, so a pure drift with no
    oscillation at all scores near zero instead of near one.  Scoring against
    the raw variance would let a drift fit its own trend and read r2 about 1 --
    an instrument that certifies rings that are not there.  The consequence is
    that these numbers are harsh: a visually clean ring inside a strong
    post-step settle scores about 0.3, which is why :data:`R2_TRUST` is 0.30 and
    why it must not be raised back toward mean-referenced values to widen a thin
    target set.
    """

    freq_hz: float
    zeta: float
    sigma: float
    amp: float
    phase: float
    offset: float
    trend: float
    r2: float
    n: int
    seeded: bool

    def as_dict(self) -> dict:
        return {
            "freq_hz": float(self.freq_hz),
            "zeta": float(self.zeta),
            "sigma": float(self.sigma),
            "amp": float(self.amp),
            "phase": float(self.phase),
            "offset": float(self.offset),
            "trend": float(self.trend),
            "r2": float(self.r2),
            "n": int(self.n),
            "seeded": bool(self.seeded),
        }


@dataclass(frozen=True)
class Episode:
    """One detected ringdown on one joint.

    ``i0``/``i1`` bracket where the envelope was above threshold; ``a0``/``a1``
    bracket the padded window the fit actually ran on.  Both are kept because
    they answer different questions -- detection quality is ``i`` -- and because
    any downstream comparison of two traces must window both on the same
    ``t0``/``t1``, which are the padded edges.
    """

    joint: int
    i0: int
    i1: int
    a0: int
    a1: int
    t0: float
    t1: float
    peak_band_amp: float
    period_s: float
    fs_hz: float
    fit: "RingFit | None" = None

    @property
    def duration_s(self) -> float:
        return float(self.t1 - self.t0)

    def as_dict(self) -> dict:
        return {
            "joint": int(self.joint),
            "i0": int(self.i0),
            "i1": int(self.i1),
            "a0": int(self.a0),
            "a1": int(self.a1),
            "t0": float(self.t0),
            "t1": float(self.t1),
            "peak_band_amp": float(self.peak_band_amp),
            "period_s": float(self.period_s),
            "fs_hz": float(self.fs_hz),
            "fit": self.fit.as_dict() if self.fit is not None else None,
        }


# ---------------------------------------------------------------------------
# Filtering and gap filling
# ---------------------------------------------------------------------------


def bandpass(q, fs_hz: float, *, band=BAND_HZ, order: int = BAND_ORDER):
    """Zero-phase Butterworth bandpass along axis 0.

    ``filtfilt`` rather than ``lfilter`` because a causal filter shifts the
    envelope in time, and the episode bounds this function feeds are then used
    to window the RAW signal on two different traces -- a phase shift would move
    the two windows by different amounts wherever the traces differ in content,
    which is exactly where the comparison matters.
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[:, None]
    n = q.shape[0]
    nyq = 0.5 * float(fs_hz)
    lo, hi = float(band[0]), float(band[1])
    if not (0.0 < lo < nyq):
        raise ValueError(
            f"band low edge {lo} Hz must lie in (0, Nyquist={nyq} Hz); "
            f"fs={fs_hz} Hz")
    # A high edge at or above Nyquist is not something the caller can act on --
    # it only means the recording cannot carry that content -- so it is pulled
    # in rather than raised, and pulled to 0.99 of Nyquist rather than to
    # Nyquist because a Butterworth designed exactly at the rail is numerically
    # unstable.
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:
        raise ValueError(
            f"band ({band[0]}, {band[1]}) Hz collapsed to ({lo}, {hi}) after "
            f"clamping to Nyquist {nyq} Hz: this trace cannot resolve the band")
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    # filtfilt's default padlen is 3*max(len(a), len(b)); a segment at or below
    # it raises inside scipy with a message that does not name the caller's
    # array, so the length is checked here where n can be reported.
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        raise ValueError(
            f"trace has {n} samples, not more than filtfilt's padlen {padlen} "
            f"for an order-{order} bandpass; at least {padlen + 1} are needed")
    out = filtfilt(b, a, q, axis=0)
    return out[:, 0] if single else out


def interp_nans(t, y):
    """Linearly fill NaN gaps in ``y``; return ``(filled, nan_fraction)``.

    The fraction is handed back rather than swallowed so a caller can refuse to
    score a mostly-missing signal.  A joint whose mocap plate dropped out for
    half the window interpolates to a smooth line that the envelope test reads
    as "no ring", which is indistinguishable in the result from an arm that
    genuinely did not ring.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).astype(np.float64, copy=True)
    bad = ~np.isfinite(y)
    frac = float(bad.mean()) if y.size else 1.0
    if frac == 0.0:
        return y, 0.0
    if bad.all():
        return y, 1.0
    good = ~bad
    y[bad] = np.interp(t[bad], t[good], y[good])
    return y, frac


# ---------------------------------------------------------------------------
# The damped-sine fit
# ---------------------------------------------------------------------------


def _model(p, tt):
    c0, c1, amp, sigma, f, phase = p
    return c0 + c1 * tt + amp * np.exp(-sigma * tt) * np.cos(2.0 * np.pi * f * tt + phase)


def _fft_peak_hz(resid, fs_hz: float) -> float:
    """Dominant frequency of the detrended residual, DC skipped.

    Hanning-windowed because an episode that starts and ends mid-cycle leaks
    across the whole spectrum otherwise, and the leak from a large
    low-frequency settle can outweigh the ring sitting on it.
    """
    n = int(resid.size)
    spec = np.abs(np.fft.rfft(resid * np.hanning(n)))
    if spec.size < 2:
        return float(fs_hz) / max(n, 1)
    k = int(np.argmax(spec[1:])) + 1
    return float(np.fft.rfftfreq(n, d=1.0 / float(fs_hz))[k])


def _sigma_seed(resid, tt, f0: float) -> float:
    """Decay-rate seed from the log envelope over the MIDDLE HALF of the window.

    The edges are excluded because the Hilbert transform's envelope is an
    artefact there -- it treats the segment as circular, so the first and last
    fraction of a period carry the other end's amplitude, and a regression that
    includes them reports a decay rate that is a property of the transform.
    """
    n = int(tt.size)
    lo, hi = n // 4, max(n // 4 + 2, (3 * n) // 4)
    seg = np.abs(hilbert(resid))[lo:hi]
    tseg = tt[lo:hi]
    if seg.size < 3 or not np.all(np.isfinite(seg)) or np.any(seg <= 0.0):
        return float(f0)
    slope = float(np.polyfit(tseg, np.log(np.maximum(seg, 1e-18)), 1)[0])
    # Clipped at twenty times the seed frequency: a mode that loses that much
    # amplitude per cycle has no cycles left, so a larger seed only drags the
    # optimiser into the trend.
    return float(np.clip(-slope, 0.0, 20.0 * max(f0, 1e-6)))


def fit_damped_sine(t, y, f0: float | None = None) -> RingFit:
    """Fit a trended damped sine to a RAW segment.

    ``f0`` is the seeded/blind switch and the distinction is load-bearing for
    :mod:`digital_twin.fit_bounce`: with a seed the frequency is confined to
    +-60 % of it, without one the search opens to half-to-double the FFT peak.
    Seeding the twin's fit with the real arm's frequency would clamp the very
    number a twin-vs-real comparison exists to measure.

    ``sigma`` is bounded at or above zero, so a growing oscillation -- which a
    numerically unstable rollout produces -- is reported as an undamped fit with
    a poor r2 rather than as a negative damping ratio, which reads like a very
    lightly damped ring.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    if n < 6:
        raise ValueError(f"a six-parameter fit needs at least 6 samples, got {n}")
    tt = t - t[0]
    span_s = float(tt[-1] - tt[0])
    fs_hz = (n - 1) / span_s if span_s > 0.0 else 1.0

    c = np.polyfit(tt, y, 1)
    resid = y - np.polyval(c, tt)

    seeded = f0 is not None
    f_seed = max(float(f0) if seeded else _fft_peak_hz(resid, fs_hz), 1e-6)
    sigma0 = _sigma_seed(resid, tt, f_seed)
    env = np.abs(hilbert(resid))
    amp0 = float(np.percentile(env, 90.0))
    if not np.isfinite(amp0) or amp0 <= 0.0:
        amp0 = float(np.std(resid)) or 1e-9

    if seeded:
        f_lo, f_hi = 0.6 * f_seed, 1.6 * f_seed
    else:
        f_lo, f_hi = max(0.1, 0.5 * f_seed), 2.0 * f_seed + 1.0
    # The decay cap is tied to the frequency cap rather than fixed: a mode
    # decaying faster than one and a half radians of amplitude per radian of
    # phase has no cycles left to fit, so any larger sigma fits the trend.
    sig_cap = 1.5 * 2.0 * np.pi * f_hi
    lo = np.array([-np.inf, -np.inf, 0.0, 0.0, f_lo, -2.0 * np.pi])
    hi = np.array([np.inf, np.inf, np.inf, sig_cap, f_hi, 2.0 * np.pi])

    best = None
    best_cost = np.inf
    # Phase is the one genuinely multi-modal start: a half-cycle error puts the
    # optimiser in a basin that fits the ring inverted and pays for it with the
    # trend.  Four seeds a quarter period apart is what the RS485 instrument
    # needed to stop reporting sign-flipped amplitudes.
    for phase0 in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
        p0 = np.array([c[1], c[0], amp0, sigma0, f_seed, phase0])
        p0 = np.minimum(np.maximum(p0, lo + 1e-12), hi - 1e-12)
        try:
            sol = least_squares(lambda p: _model(p, tt) - y, p0,
                                bounds=(lo, hi), max_nfev=2000)
        except Exception:  # a singular Jacobian on a flat segment is not fatal
            continue
        if float(sol.cost) < best_cost:
            best_cost = float(sol.cost)
            best = sol
    if best is None:
        return RingFit(float("nan"), float("nan"), float("nan"), 0.0, 0.0,
                       float(c[1]), float(c[0]), 0.0, n, seeded)

    c0, c1, amp, sigma, f, phase = (float(v) for v in best.x)
    omega = 2.0 * np.pi * f
    zeta = sigma / float(np.hypot(sigma, omega)) if omega > 0.0 else 1.0

    ss_res = float(np.sum((y - _model(best.x, tt)) ** 2))
    # See the RingFit docstring: ss_tot is the DETRENDED residual's variance.
    ss_tot = float(np.sum((resid - resid.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return RingFit(f, float(zeta), sigma, abs(amp), phase, c0, c1, float(r2), n,
                   seeded)


# ---------------------------------------------------------------------------
# Episode detection
# ---------------------------------------------------------------------------


def _contiguous_regions(mask):
    """``[(i0, i1), ...]`` half-open runs of True.

    Written with ``diff`` on the int cast plus explicit end handling rather than
    with a library grouper, because a run touching either edge of the array is
    the common case here -- an episode still ringing when the recording stops --
    and a grouper that drops those loses the loudest ones.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = [int(i) + 1 for i in edges if mask[i + 1]]
    ends = [int(i) + 1 for i in edges if not mask[i + 1]]
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(int(mask.size))
    return list(zip(starts, ends))


def find_episodes(t, q, fs_hz: float, *, band=BAND_HZ, order: int = BAND_ORDER,
                  thresh: float = THRESH_DEG, min_cycles: float = MIN_CYCLES,
                  min_samples: int = MIN_SAMPLES,
                  joints: "Sequence[int] | None" = None,
                  fit: bool = True) -> "list[Episode]":
    """Find ringdown episodes in ``q`` ``(n, nj)``, one joint at a time.

    ``q`` and ``thresh`` are in the caller's own units and this function never
    converts: the RS485 instrument works in degrees and so does every caller in
    this package, but the arithmetic is unit-free and a caller working in
    radians only has to pass a radian threshold.  Returned sorted by descending
    in-band amplitude, so a caller keeping the first few keeps the loudest.

    WHAT A DETECTED WINDOW DOES NOT GIVE YOU.  ``filtfilt`` is zero-phase, so a
    sharp onset smears the envelope BACKWARDS by the filter's tail and the
    window opens before the ring did.  Measured on a synthetic 1.8 Hz / 0.055
    ringdown starting from a flat trace, the window opened about 0.39 s early
    and the fitted damping ratio read 0.022 against a true 0.055, while the same
    fitter on the true window returned 0.055 to five figures
    (``test_a_hard_onset_biases_the_detected_window_low``).  The bias is the
    detector's, not the fitter's, and it is common-mode in
    :mod:`digital_twin.fit_bounce`, which windows both traces on the same
    ``[t0, t1]`` -- so an episode's damping ratio is trustworthy as a comparison
    against another trace measured the same way, and not as an absolute.
    """
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    if q.shape[0] != t.shape[0]:
        raise ValueError(f"t has {t.shape[0]} samples, q has {q.shape[0]}")
    qb = bandpass(q, fs_hz, band=band, order=order)
    idx = range(q.shape[1]) if joints is None else [int(j) for j in joints]

    out: "list[Episode]" = []
    for j in idx:
        col = qb[:, j]
        if not np.all(np.isfinite(col)):
            continue
        env = np.abs(hilbert(col))
        for i0, i1 in _contiguous_regions(env > float(thresh)):
            seg = col[i0:i1]
            if seg.size < int(min_samples):
                continue
            # Dominant period from zero crossings of the BAND-PASSED segment:
            # the raw segment's crossings are set by wherever the settle put the
            # signal relative to zero, which is not the mode's period.
            zc = np.flatnonzero(np.diff(np.signbit(seg)))
            if zc.size < 2.0 * float(min_cycles):
                continue
            period_s = 2.0 * float(np.mean(np.diff(zc))) / float(fs_hz)
            if not (period_s > 0.0):
                continue
            if (t[i1 - 1] - t[i0]) < float(min_cycles) * period_s:
                continue
            # A quarter period of padding each side: the envelope threshold cuts
            # the episode where the amplitude has already decayed past it, and a
            # fit starting there has thrown away the largest, best-conditioned
            # part of the decay it is trying to measure.
            pad = int(round(period_s * float(fs_hz) / 4.0))
            a0 = max(0, i0 - pad)
            a1 = min(q.shape[0], i1 + pad)
            rf = None
            if fit and (a1 - a0) >= 6:
                rf = fit_damped_sine(t[a0:a1], q[a0:a1, j], f0=1.0 / period_s)
            out.append(Episode(joint=int(j), i0=int(i0), i1=int(i1), a0=int(a0),
                               a1=int(a1), t0=float(t[a0]), t1=float(t[a1 - 1]),
                               peak_band_amp=float(np.max(env[i0:i1])),
                               period_s=period_s, fs_hz=float(fs_hz), fit=rf))
    out.sort(key=lambda e: -e.peak_band_amp)
    return out


def sample_rate_hz(t) -> float:
    """Sample rate from the MEDIAN inter-sample gap, not the mean.

    The median is what survives a recording with dropped cycles: the collector's
    own timing summary reports a 6.666 ms median against a 25.069 ms maximum on
    the flagship legacy hour run, so a mean would be pulled by the stalls and
    every derived period biased with it.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        raise ValueError(f"need at least 2 samples to derive a rate, got {t.size}")
    dt = float(np.median(np.diff(t)))
    if not (dt > 0.0):
        raise ValueError(f"median sample gap is {dt} s; timestamps must increase")
    return 1.0 / dt


def episodes_from_trace(t, q, *, band=BAND_HZ, order: int = BAND_ORDER,
                        thresh: float = THRESH_DEG, min_cycles: float = MIN_CYCLES,
                        min_samples: int = MIN_SAMPLES,
                        joints: "Sequence[int] | None" = None,
                        max_nan_frac: float = 0.2,
                        fit: bool = True) -> "list[Episode]":
    """THE entry point.  Derive ``fs``, fill gaps, detect -- identically for every trace.

    This exists so that a recording and a rollout of that recording cannot be
    analysed by two slightly different call sequences.  Both go through here;
    the only thing that differs between them is the array handed in.

    A joint with more than ``max_nan_frac`` of its samples missing is blanked
    rather than scored: interpolating across a long mocap dropout manufactures a
    smooth trace, and a smooth trace is indistinguishable from an arm that did
    not ring.  The RS485 fit used one fifth, which is also the point past which
    a 150 Hz recording no longer resolves a 1.8 Hz period between gaps.
    """
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    fs = sample_rate_hz(t)
    filled = np.empty_like(q)
    keep: "list[int]" = []
    for j in range(q.shape[1]):
        col, frac = interp_nans(t, q[:, j])
        filled[:, j] = col
        if frac <= float(max_nan_frac):
            keep.append(j)
    if joints is not None:
        want = {int(x) for x in joints}
        keep = [j for j in keep if j in want]
    if not keep:
        return []
    return find_episodes(t, filled, fs, band=band, order=order, thresh=thresh,
                         min_cycles=min_cycles, min_samples=min_samples,
                         joints=keep, fit=fit)


def ring_metrics(episodes, *, r2_trust: float = R2_TRUST) -> dict:
    """Summary of a set of episodes, reporting how many were TRUSTED, not only how many were found.

    A bare count hides the failure this instrument was built to catch: an
    over-damped twin still produces episodes, but every fit falls below the r2
    gate, and "18 episodes" then reads the same as eighteen good ones.
    """
    fits = [e.fit for e in episodes if e.fit is not None]
    good = [f for f in fits if np.isfinite(f.r2) and f.r2 >= float(r2_trust)]
    out = {
        "n_episodes": len(episodes),
        "n_fits": len(fits),
        "n_trusted": len(good),
        "r2_trust": float(r2_trust),
    }
    for key, vals in (("median_freq_hz", [f.freq_hz for f in good]),
                      ("median_zeta", [f.zeta for f in good]),
                      ("median_amp", [f.amp for f in good]),
                      ("median_r2", [f.r2 for f in good])):
        out[key] = float(np.median(vals)) if good else float("nan")
    return out


def band_rms(t, y, t0: float, t1: float, *, band=BAND_HZ,
             order: int = BAND_ORDER) -> float:
    """In-band RMS of ``y`` over ``[t0, t1]``.

    Always defined, which is why :mod:`digital_twin.fit_bounce` leans on it: a
    twin that is dead in the water still has a well-defined, near-zero band RMS,
    whereas its damped-sine fit is garbage that would steer an optimiser with
    noise.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fs = sample_rate_hz(t)
    filled, _ = interp_nans(t, y)
    yb = bandpass(filled, fs, band=band, order=order)
    m = (t >= float(t0)) & (t <= float(t1))
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean(yb[m] ** 2)))


__all__ = [
    "BAND_HZ", "BAND_ORDER", "MIN_CYCLES", "THRESH_DEG", "MIN_SAMPLES", "R2_TRUST",
    "RingFit", "Episode",
    "bandpass", "interp_nans", "fit_damped_sine", "find_episodes",
    "sample_rate_hz", "episodes_from_trace", "ring_metrics", "band_rms",
]
