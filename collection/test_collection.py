"""Offline tests for the collection stack.  No port, no socket, no camera.

The safety tests are the ones that matter: they are what stands between a
generator bug and twenty-four McKibben actuators, and they are written to fail
on the *rule* rather than on the working limit, so tightening or loosening the
campaign's own headroom cannot quietly weaken them.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collection import excitation as EX  # noqa: E402
from collection.resample import (  # noqa: E402
    _savgol_derivative, clock_offset, resample_q,
)
from collection.safety import (  # noqa: E402
    ALL_BASES, IDLE_PSI, PAIR_SUM_MAX_PSI, SINGLE_MAX_PSI, PairEnvelope,
)


# --------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------


def test_pairs_cover_every_board_exactly_once():
    env = PairEnvelope()
    seen = sorted(int(i) for row in env.pair_idx for i in row)
    assert seen == list(range(24))
    assert len(env.pairs) == 12


def test_clamp_respects_the_rule_on_random_vectors():
    """Ten thousand adversarial vectors, including ones far over the rule."""
    env = PairEnvelope()
    rng = np.random.default_rng(7)
    for _ in range(10000):
        raw = rng.uniform(-5.0, 60.0, size=24)
        env.assert_safe(env.clamp(raw), where="fuzz")


def test_clamp_scales_a_pair_rather_than_truncating_one_member():
    """The pair's difference is the joint torque; scaling preserves it.

    Truncating the larger member would silently rotate the command, which is
    the failure a clamp is supposed to prevent rather than cause.
    """
    env = PairEnvelope(pair_sum_max_psi=20.0)
    j = 0
    a, b = env.pair_idx[j]
    raw = env.idle_vector()
    raw[a], raw[b] = 24.0, 12.0
    out = env.clamp(raw)
    assert out[a] + out[b] == pytest.approx(20.0, abs=1e-9)
    # Both members moved, and in proportion above the idle floor.
    head_in = (24.0 - IDLE_PSI) / (12.0 - IDLE_PSI)
    head_out = (out[a] - IDLE_PSI) / (out[b] - IDLE_PSI)
    assert head_out == pytest.approx(head_in, rel=1e-9)


def test_clamp_never_drops_a_board_below_the_idle_floor():
    """The floor is what keeps 0x110 venting; a clamp must not undo it."""
    env = PairEnvelope(pair_sum_max_psi=8.0)
    out = env.clamp(np.full(24, 40.0))
    assert out.min() >= IDLE_PSI - 1e-12


def test_assert_safe_rejects_what_it_should():
    env = PairEnvelope()
    ok = env.idle_vector()
    env.assert_safe(ok, where="t")
    for bad in (np.full(24, SINGLE_MAX_PSI + 0.5),
                np.full(24, np.nan),
                np.full(23, 1.0)):
        with pytest.raises(AssertionError):
            env.assert_safe(bad, where="t")
    over_pair = env.idle_vector()
    a, b = env.pair_idx[3]
    over_pair[a] = PAIR_SUM_MAX_PSI
    over_pair[b] = 1.0
    with pytest.raises(AssertionError):
        env.assert_safe(over_pair, where="t")


def test_envelope_refuses_to_be_constructed_above_the_operators_rule():
    with pytest.raises(ValueError):
        PairEnvelope(pair_sum_max_psi=PAIR_SUM_MAX_PSI + 1.0)
    with pytest.raises(ValueError):
        PairEnvelope(single_max_psi=SINGLE_MAX_PSI + 1.0)


def test_from_pairs_round_trips_co_and_differential():
    env = PairEnvelope()
    co = np.full(12, 18.0)
    diff = np.linspace(-15.0, 15.0, 12)
    psi = env.from_pairs(co, diff)
    assert np.allclose(env.pair_sums(psi), co)
    got = psi[env.pair_idx[:, 0]] - psi[env.pair_idx[:, 1]]
    assert np.allclose(got, diff)


# --------------------------------------------------------------------------
# The campaign plan
# --------------------------------------------------------------------------


def test_full_campaign_never_needs_the_clamp():
    """The generator must satisfy the rule by construction.

    A clamp that fires during a campaign is a finding about the generator, so
    this asserts on the clamp counters rather than only on the safety of the
    result: a plan that is safe *because* it was clamped is a plan whose
    commanded torque was silently rewritten.
    """
    env = PairEnvelope()
    segs = EX.full_campaign(env, seed=1, scale=1.0)
    assert len(segs) > 300
    for sg in segs:
        for t in np.linspace(0.0, sg.duration_s, 41):
            psi = np.asarray(sg.fn(float(t)), dtype=float)
            env.assert_safe(psi, where=f"{sg.name}@{t:.2f}")
            env.clamp(psi)
    assert env.clamped_single == 0
    assert env.clamped_pair == 0


def test_every_segment_returns_a_24_vector_at_its_own_floor():
    env = PairEnvelope()
    for sg in EX.full_campaign(env, seed=2, scale=0.1):
        psi = np.asarray(sg.fn(0.0), dtype=float)
        assert psi.shape == (24,)
        assert psi.min() >= IDLE_PSI - 1e-12


def test_scale_shortens_durations_without_dropping_phases():
    """A rehearsal must exercise the same code path as the session.

    The phase that breaks is never the one that got dropped, so ``scale``
    shortens every segment and removes none.
    """
    env = PairEnvelope()
    full = EX.full_campaign(env, seed=3, scale=1.0)
    short = EX.full_campaign(env, seed=3, scale=0.1)
    assert len(full) == len(short)
    assert [s.kind for s in full] == [s.kind for s in short]
    assert EX.total_seconds(short) < EX.total_seconds(full)


def test_the_campaign_covers_both_populations_in_every_per_board_phase():
    """Two plants on one bus: a phase that visits one describes neither."""
    env = PairEnvelope()
    segs = EX.full_campaign(env, seed=4, scale=0.1)
    tle = set(range(0x101, 0x109))
    seven = set(range(0x109, 0x119))
    for kind in ("staircase", "supply"):
        driven = set()
        for sg in segs:
            if sg.kind == kind:
                driven |= set(sg.driven)
        assert driven & tle, f"{kind} never drives a TLE board"
        assert driven & seven, f"{kind} never drives a 7 mm board"


def test_random_walk_holds_have_no_single_period():
    """A fixed hold gives every trace one spectrum a net can match instead of
    the dynamics.  The drawn holds must therefore actually vary."""
    env = PairEnvelope()
    segs = EX.random_walk(env, seed=5, seconds=120.0, block_s=30.0)
    n = [sg.meta["n_targets"] for sg in segs]
    assert len(set(n)) > 1


def test_ringdown_release_is_a_step_not_a_ramp():
    """A release slower than the mode being measured reports the release."""
    env = PairEnvelope()
    segs = EX.ringdowns(env, seed=6, episodes=4)
    rel = [s for s in segs if s.kind == "ringdown"]
    assert rel
    for sg in rel:
        a = np.asarray(sg.fn(0.0))
        b = np.asarray(sg.fn(sg.duration_s * 0.5))
        assert np.allclose(a, b), "the released target must be constant"


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def test_savgol_derivative_recovers_a_known_slope():
    t = np.arange(0.0, 5.0, 1.0 / 120.0)
    q = np.stack([3.0 * t + 1.0, -0.5 * t], axis=1)
    d = _savgol_derivative(t, q)
    assert np.allclose(d[:, 0], 3.0, atol=1e-8)
    assert np.allclose(d[:, 1], -0.5, atol=1e-8)


def test_savgol_derivative_survives_a_non_uniform_grid():
    """The closed-form coefficients assume uniformity; a camera is not.

    Jitter the grid by a tenth of a frame and the local fit must still recover
    the slope, which is the whole reason it is a fit rather than a stencil.
    """
    rng = np.random.default_rng(11)
    t = np.arange(0.0, 5.0, 1.0 / 120.0)
    t = t + rng.uniform(-1.0, 1.0, size=t.shape) / 1200.0
    t = np.sort(t)
    q = (2.5 * t)[:, None]
    d = _savgol_derivative(t, q)
    assert np.allclose(d[5:-5, 0], 2.5, atol=1e-6)


@pytest.mark.parametrize("stream_hz", [60.0, 120.0, 150.0, 240.0, 360.0])
def test_resample_is_rate_agnostic(stream_hz):
    """The same call must work below, at, and above the cycle rate.

    Motive runs at 120 Hz today and the operator intends to raise it past 150.
    Above the cycle rate this becomes a decimation, and the tolerance is looser
    there on purpose: the anti-alias filter that makes decimation correct also
    attenuates the signal slightly, and attenuating is the right trade against
    folding out-of-band energy into the band being fitted.
    """
    f = 1.5
    t_s = np.arange(0.0, 8.0, 1.0 / stream_hz)
    q_s = np.stack([np.sin(2 * np.pi * f * t_s),
                    np.cos(2 * np.pi * f * t_s)], axis=1)
    t_t = np.arange(0.05, 7.95, 1.0 / 150.0)
    q, qdot, valid, gap, hz = resample_q(t_s, q_s, t_t)
    assert hz == pytest.approx(stream_hz, rel=0.02)
    assert valid.all()
    truth = np.sin(2 * np.pi * f * t_t)
    tol = 0.02 if stream_hz >= 300.0 else 0.01
    assert np.abs(q[:, 0] - truth).max() < tol
    dtruth = 2 * np.pi * f * np.cos(2 * np.pi * f * t_t)
    assert np.abs(qdot[:, 0] - dtruth).max() < 0.12 * np.abs(dtruth).max()


def test_resample_marks_a_dropout_invalid_rather_than_extrapolating():
    t_s = np.arange(0.0, 4.0, 1.0 / 120.0)
    keep = ~((t_s > 1.5) & (t_s < 1.7))          # a 200 ms hole
    q_s = np.sin(2 * np.pi * t_s)[:, None]
    t_t = np.arange(0.1, 3.9, 1.0 / 150.0)
    q, qdot, valid, gap, hz = resample_q(t_s[keep], q_s[keep], t_t)
    inside = (t_t > 1.55) & (t_t < 1.65)
    assert not valid[inside].any()
    assert valid[t_t < 1.4].all()
    assert q.shape == (t_t.shape[0], 1)          # shape stays rectangular


def test_resample_marks_outside_the_stream_invalid():
    t_s = np.arange(1.0, 3.0, 1.0 / 120.0)
    q_s = t_s[:, None]
    t_t = np.arange(0.0, 4.0, 1.0 / 150.0)
    _, _, valid, _, _ = resample_q(t_s, q_s, t_t)
    assert not valid[t_t < 0.99].any()
    assert not valid[t_t > 3.01].any()


def test_clock_offset_is_zero_for_simultaneous_origins():
    """Both origins are stamped microseconds apart, so the offset is nil.

    The value is worth computing anyway: a host whose monotonic and
    perf_counter do not share a source would show up here rather than as a slow
    drift between the two series over a session.
    """
    meta = {"t0_perf_s": 1000.0, "t0_monotonic_s": 400.0,
            "perf_minus_monotonic_s": 600.0}
    assert clock_offset(meta) == pytest.approx(0.0, abs=1e-12)
    meta_drift = dict(meta, perf_minus_monotonic_s=600.25)
    assert clock_offset(meta_drift) == pytest.approx(-0.25, abs=1e-12)
    assert clock_offset({}) == 0.0


def test_held_series_differentiates_worse_than_the_resampled_one():
    """The measurement that motivated recording the raw stream at all.

    A 120 Hz stream held onto a 150 Hz grid repeats one sample in five, and the
    repeats differentiate into a staircase.  On the live rest recording of
    2026-09-10 the held series' qdot ran 1.8x the interpolated one on an arm
    that was not moving; this reproduces the mechanism synthetically.
    """
    rng = np.random.default_rng(19)
    t_s = np.arange(0.0, 20.0, 1.0 / 120.0)
    # A nearly-static arm with realistic plate noise: 0.3 mm of marker noise
    # over a 0.25 m link is about 1.2 mrad, which is the regime the live rest
    # recording sat in and the regime where the staircase dominates.
    q_s = (0.004 * np.sin(2 * np.pi * 0.2 * t_s)
           + rng.normal(0.0, 1.2e-3, size=t_s.shape))[:, None]
    t_t = np.arange(0.1, 19.9, 1.0 / 150.0)
    q, qdot, valid, _, _ = resample_q(t_s, q_s, t_t)
    idx = np.clip(np.searchsorted(t_s, t_t, side="right") - 1, 0, len(t_s) - 1)
    held = q_s[idx]
    d_held = np.gradient(held[:, 0], t_t)
    ratio = (np.percentile(np.abs(d_held), 99)
             / np.percentile(np.abs(qdot[:, 0]), 99))
    assert ratio > 1.5, f"held/resampled qdot p99 ratio was only {ratio:.2f}"


# --------------------------------------------------------------------------
# The recorder's schema
# --------------------------------------------------------------------------


def test_recorder_writes_the_documented_schema(tmp_path):
    """Exercised without hardware by feeding it synthetic cycles."""
    from tlelib import proto as P
    from collection.recorder import Recorder

    cals = {b: P.NodeCal.for_base(b) for b in ALL_BASES}
    bt = [P.VARIANT_TLE_DVP if b < 0x109 else P.VARIANT_7MM for b in ALL_BASES]
    rec = Recorder(str(tmp_path / "session"), ids=ALL_BASES, board_type=bt,
                   cals=cals, mocap=None, chunk_cycles=50)
    rec.start()
    try:
        for k in range(120):
            t = 100.0 + k / 150.0
            targets = {b: (cals[b].psi_to_counts(2.0), True) for b in ALL_BASES}
            replies = {b: (t + 0.002, P.CompactStatus(b, 900 + k, 1))
                       for b in ALL_BASES}
            rec.segment, rec.segment_kind, rec.segment_index = "seg", "kind", 3
            rec.on_cycle(t, targets, replies)
        import time as _t
        _t.sleep(0.5)
    finally:
        rec.close()

    meta = json.loads((tmp_path / "session" / "metadata.json").read_text())
    assert meta["board_type"] == bt
    assert len(meta["selected_ids"]) == 24
    assert "sync_semantics" in meta and "t0_perf_s" in meta
    man = json.loads((tmp_path / "session" / "manifest.json").read_text())
    assert man["active"] is False
    assert man["total_samples"] == 120
    assert man["rows_dropped"] == 0

    rows = []
    for chunk in man["chunks"]:
        with open(tmp_path / "session" / chunk["path"], encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    assert len(rows) == 120
    r = rows[0]
    assert len(r["robot_state"]["pressure_adc"]) == 24
    assert len(r["input"]["target_adc"]) == 24
    assert r["segment"] == "seg" and r["segment_index"] == 3
    assert r["cycle_responded"] == 24
    # can_sync_time_s must be the sync edge relative to t0, monotone increasing.
    t = np.array([x["can_sync_time_s"] for x in rows])
    assert np.all(np.diff(t) > 0)


def test_recorder_counts_drops_rather_than_blocking(tmp_path):
    """A stalled disk must cost samples and say so, not stall the cycle."""
    from tlelib import proto as P
    from collection.recorder import Recorder, QUEUE_LIMIT

    cals = {b: P.NodeCal.for_base(b) for b in ALL_BASES}
    rec = Recorder(str(tmp_path / "s"), ids=ALL_BASES,
                   board_type=[0] * 24, cals=cals, mocap=None)
    rec.t0_perf = 0.0
    rec.t0_mono = 0.0
    # No writer thread started, so nothing drains the queue.
    for k in range(QUEUE_LIMIT + 25):
        rec.on_cycle(float(k), {}, {})
    assert rec.dropped == 25
    assert rec.cycles == QUEUE_LIMIT + 25
