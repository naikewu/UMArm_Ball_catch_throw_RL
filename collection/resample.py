"""Put the mocap stream onto the CAN sync edges, at whatever rate it arrives.

Motive and the arm run on different clocks and at different rates, and neither
is a multiple of the other.  As recorded on 2026-09-10 the cameras stream at
120 Hz against a 150 Hz cycle, so four sync edges in five find a frame that
arrived since the last one and the fifth finds the same frame again.  Fitting to
that series is the mistake this module exists to prevent: a zero-order hold puts
a 0 - 0 - 0 - step pattern into ``q``, and differentiating it gives a ``qdot``
whose spectrum is the beat between 120 and 150 Hz rather than the arm's motion.

Two properties are wanted, and only one of them is about today's rates.

**Correct now.** Between two mocap frames the joints move by a fraction of a
degree — at the fastest motion in the campaign, about 60 deg/s, one 8.3 ms
frame interval is 0.5 deg — so linear interpolation between the bracketing
frames is accurate to well under the marker noise, which the axis campaign put
at 0.02-0.07 mm of plate standard deviation and about 1.99 mm RMS of chain
error.  Interpolating rather than holding removes an error of up to a full
frame interval, and it is the *systematic* part of that error, always lagging.

**Correct later.** The operator intends to raise Motive above 150 Hz.  Nothing
here assumes the mocap rate is below the cycle rate: the same interpolation
becomes a resampling-down when it is above, and :func:`resample_q` low-pass
filters first in that case, because decimating a faster series without one
folds everything above the cycle's Nyquist back into the band being fitted.
That is why the recording keeps the raw stream instead of a resampled ``q`` —
a file that had stored only the held view could not be re-resampled at all.

What this does **not** do is invent data.  A sync edge that falls outside the
recorded stream, or inside a gap longer than :data:`MAX_GAP_FACTOR` frame
intervals, is marked invalid rather than extrapolated.  Motive drops frames,
and an extrapolated pose across a dropout is a pose nobody measured.
"""

from __future__ import annotations

import json
import os

import numpy as np

#: A gap wider than this many nominal frame intervals is a dropout, not a
#: sampling instant, and the sync edges inside it are marked invalid.  Two and a
#: half intervals admits a single dropped frame — which Motive does routinely —
#: and rejects the two-frame gaps where a plate went untracked.
MAX_GAP_FACTOR = 2.5

#: Half-width, in samples of the mocap grid, of the Savitzky-Golay window used
#: to differentiate ``q``.  Chosen from the noise rather than from the rate:
#: about 0.3 mm of plate noise differentiates to roughly 40 mm/s of spurious
#: marker velocity at 120 Hz, and a 7-point quadratic window suppresses that by
#: about 4x while passing the arm's few-Hz ring essentially untouched.
SG_HALF = 3


def load_stream(session_dir: str):
    """Read ``mocap_stream.jsonl`` into ``(t_s, frame_no, q)``.

    ``t_s`` is seconds since the session's ``t0_monotonic_s``, which is the same
    origin the per-cycle rows' ``can_sync_time_s`` uses through
    ``t0_perf_s`` — both are QueryPerformanceCounter on this host, and the
    metadata records the offset so a host where they differ is detectable
    rather than silently wrong.
    """
    path = os.path.join(session_dir, "mocap_stream.jsonl")
    t, fno, q = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            t.append(row["t_mono_s"])
            fno.append(row["frame"])
            q.append(row["q"])
    order = np.argsort(np.asarray(t, dtype=float), kind="stable")
    t = np.asarray(t, dtype=float)[order]
    fno = np.asarray(fno, dtype=np.int64)[order]
    q = np.asarray(q, dtype=float)[order]
    # A repeated timestamp would make np.interp's bracket ambiguous and the
    # gradient infinite; the drain deduplicates on frame number, so any repeat
    # left here is Motive restamping, and dropping it is the honest fix.
    keep = np.ones(t.shape[0], dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    return t[keep], fno[keep], q[keep]


def clock_offset(metadata: dict) -> float:
    """Seconds to add to a ``can_sync_time_s`` to reach the stream's timebase.

    ``can_sync_time_s`` counts from ``t0_perf_s`` on ``perf_counter``; a stream
    row's ``t_mono_s`` counts from ``t0_monotonic_s`` on ``monotonic``.  With
    ``d`` the recorded ``perf_counter() - monotonic()`` difference, a cycle at
    ``c`` sits at absolute perf ``c + t0_perf``, at absolute monotonic
    ``c + t0_perf - d``, and therefore at stream-relative
    ``c + t0_perf - d - t0_monotonic``.

    The two origins are stamped microseconds apart, so this evaluates to
    approximately zero and is worth computing anyway: it is the one number that
    would reveal a host whose ``monotonic`` and ``perf_counter`` do not share a
    source, where the two series would drift apart over a session instead of
    being offset by a constant.
    """
    t0p = metadata.get("t0_perf_s")
    t0m = metadata.get("t0_monotonic_s")
    d = metadata.get("perf_minus_monotonic_s")
    if t0p is None or t0m is None or d is None:
        return 0.0
    return float(t0p) - float(d) - float(t0m)


def _savgol_derivative(t: np.ndarray, q: np.ndarray, half: int = SG_HALF):
    """Local quadratic fit derivative against the actual timestamps.

    Written out rather than taken from ``scipy.signal.savgol_filter`` because
    the closed-form coefficients assume an exactly uniform grid, and a camera
    grid is uniform only to its own jitter — an assumption that would fail
    silently if a frame arrived late.  Solving the local least squares against
    the recorded times costs one batched 3x3 solve and removes it.

    Vectorised over samples: a per-sample ``lstsq`` is a quarter of a million
    factorisations on a thirty-minute session at 120 Hz.  The normal equations
    of a three-term fit are 3x3 and well conditioned once ``dt`` is centred on
    each sample, so ``np.linalg.solve`` over a stacked ``(n, 3, 3)`` is both
    faster and no less accurate here.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n, n_j = q.shape
    w = 2 * half + 1
    if n < w + 2:
        return np.gradient(q, t, axis=0)

    # Sliding windows over the interior, where every sample has `half` on each
    # side; the edges fall back to a plain gradient rather than to a lopsided
    # window, which would bias the estimate exactly where the data is thinnest.
    from numpy.lib.stride_tricks import sliding_window_view
    tw = sliding_window_view(t, w)                    # (n-w+1, w)
    qw = sliding_window_view(q, w, axis=0)            # (n-w+1, n_j, w)
    centre = t[half:n - half][:, None]
    dt = tw - centre                                  # (m, w)
    A = np.stack([np.ones_like(dt), dt, dt * dt], axis=2)   # (m, w, 3)
    AtA = np.einsum("mwi,mwj->mij", A, A)             # (m, 3, 3)
    Aty = np.einsum("mwi,mjw->mij", A, qw)            # (m, 3, n_j)
    coef = np.linalg.solve(AtA, Aty)                  # (m, 3, n_j)

    out = np.gradient(q, t, axis=0)
    out[half:n - half] = coef[:, 1, :]
    return out


def resample_q(t_stream: np.ndarray, q_stream: np.ndarray,
               t_target: np.ndarray, *, max_gap_factor: float = MAX_GAP_FACTOR,
               compute_qdot: bool = True):
    """Interpolate ``q`` (and differentiate it) onto ``t_target``.

    Returns ``(q, qdot, valid, gap_s, stream_hz)``.  ``valid`` is false where the target
    time falls outside the stream or inside a gap wider than
    ``max_gap_factor`` nominal frame intervals; ``q`` there is filled with the
    nearest sample so downstream shapes stay rectangular, and it must not be
    fitted to — that is what ``valid`` is for.

    ``qdot`` is differentiated **on the mocap grid** and then interpolated,
    never differentiated from the interpolated series: linear interpolation
    between two frames has a constant slope, so differentiating afterwards would
    return a piecewise-constant ``qdot`` that steps at every frame boundary and
    carries none of the smoothing the noise needs.
    """
    t_stream = np.asarray(t_stream, dtype=float)
    q_stream = np.asarray(q_stream, dtype=float)
    t_target = np.asarray(t_target, dtype=float)
    n_out, n_j = t_target.shape[0], q_stream.shape[1]
    if t_stream.shape[0] < 2:
        return (np.zeros((n_out, n_j)), np.zeros((n_out, n_j)),
                np.zeros(n_out, dtype=bool), np.full(n_out, np.inf))

    dt_nom = float(np.median(np.diff(t_stream)))
    stream_hz = 1.0 / dt_nom if dt_nom > 0 else float("inf")
    target_dt = (float(np.median(np.diff(t_target)))
                 if t_target.shape[0] > 1 else dt_nom)

    q_src = q_stream
    if dt_nom < 0.5 * target_dt:
        # The stream is more than twice the target rate: this is a decimation,
        # and decimating without a low-pass folds everything above the target's
        # Nyquist into the band being fitted.  A moving average whose width is
        # the target period is the crudest filter that removes that fold, and
        # its own pass band is flat well past the arm's few-Hz dynamics.
        w = max(3, int(round(target_dt / dt_nom)) | 1)
        pad = w // 2
        padded = np.pad(q_src, ((pad, pad), (0, 0)), mode="edge")
        kern = np.ones(w) / w
        q_src = np.stack([np.convolve(padded[:, j], kern, mode="valid")
                          for j in range(n_j)], axis=1)

    idx = np.searchsorted(t_stream, t_target, side="right")
    lo = np.clip(idx - 1, 0, t_stream.shape[0] - 1)
    hi = np.clip(idx, 0, t_stream.shape[0] - 1)
    gap = t_stream[hi] - t_stream[lo]
    inside = (t_target >= t_stream[0]) & (t_target <= t_stream[-1])
    valid = inside & (gap <= max_gap_factor * dt_nom)

    q_out = np.empty((n_out, n_j))
    for j in range(n_j):
        q_out[:, j] = np.interp(t_target, t_stream, q_src[:, j])

    if compute_qdot:
        qdot_stream = _savgol_derivative(t_stream, q_src)
        qdot_out = np.empty((n_out, n_j))
        for j in range(n_j):
            qdot_out[:, j] = np.interp(t_target, t_stream, qdot_stream[:, j])
    else:
        qdot_out = np.zeros((n_out, n_j))

    return q_out, qdot_out, valid, gap, stream_hz


def resample_session(session_dir: str, t_sync: np.ndarray, metadata: dict):
    """The whole job for one session: load, align clocks, resample, report.

    Returns a dict with ``q``, ``qdot``, ``valid``, ``stream_hz``, and the
    coverage numbers a caller should print rather than assume — in particular
    ``invalid_frac``, which is how much of the recording the fit must drop
    because mocap was not there for it.
    """
    t_stream, frame_no, q_stream = load_stream(session_dir)
    off = clock_offset(metadata)
    t_target = np.asarray(t_sync, dtype=float) + off
    q, qdot, valid, gap, stream_hz = resample_q(t_stream, q_stream, t_target)
    dropped = int(np.count_nonzero(np.diff(frame_no) > 1))
    return {
        "q": q,
        "qdot": qdot,
        "valid": valid,
        "gap_s": gap,
        "stream_hz": stream_hz,
        "stream_samples": int(t_stream.shape[0]),
        "cycle_hz": (float((t_sync.shape[0] - 1) / (t_sync[-1] - t_sync[0]))
                     if t_sync.shape[0] > 1 else 0.0),
        "clock_offset_s": off,
        "invalid_frac": float(1.0 - np.mean(valid)),
        "motive_frame_gaps": dropped,
        "ratio_stream_to_cycle": (stream_hz /
                                  ((t_sync.shape[0] - 1) / (t_sync[-1] - t_sync[0]))
                                  if t_sync.shape[0] > 1 else float("nan")),
    }
