"""What ``dataset.py`` must not get wrong, and how each failure would look.

Every test here is offline: the synthetic session generator writes the real
schema, so the reader under test is the same reader a bench recording will meet.
No test opens a port, a socket or a camera.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin import actuator_model as am  # noqa: E402
from digital_twin import dataset as ds  # noqa: E402


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Six 8 s episodes at 150 Hz -- 7200 cycles, 24 boards, two populations."""
    d = tmp_path_factory.mktemp("sessions")
    return ds.write_synthetic_session(str(d / "session_20260910_120000"),
                                      n_episodes=6, episode_s=8.0, seed=7)


@pytest.fixture(scope="module")
def rec(session):
    return ds.load_session(session)


# ---------------------------------------------------------------------------
# ADC -> Pa
# ---------------------------------------------------------------------------
def test_calibration_constants_match_the_bus_tooling():
    """A second copy of the four calibration numbers is a copy that drifts.

    If ``tlelib.proto`` is importable it is authoritative and the literals in
    ``dataset`` are only a fallback; this asserts the fallback has not gone stale
    against it.
    """
    proto = pytest.importorskip("tlelib.proto")
    assert ds.TLE_ZERO_COUNTS == proto.TLE_ZERO_COUNTS
    assert ds.TLE_COUNTS_PER_PSI == proto.TLE_COUNTS_PER_PSI
    assert ds.LEGACY_ZERO_COUNTS == proto.LEGACY_ZERO_COUNTS
    assert ds.LEGACY_COUNTS_PER_PSI == proto.LEGACY_COUNTS_PER_PSI
    assert ds.ADC_MAX == proto.PRESSURE_MASK


def test_adc_uses_the_boards_own_cal_not_its_id_range(tmp_path):
    """The 2026-08 bench case: a TLE board sitting at 0x114.

    Reading it on the 7 mm scale is a 10 % psi error at the top of the range.
    Here the same count is converted both ways and the two answers must differ by
    more than the 7 mm firmware's own +-2000 Pa hysteresis, which is what makes
    the error look like valve behaviour rather than like a unit mistake.
    """
    board_type = [ds.VARIANT_7MM] * 24
    board_type[0x114 - 0x101] = ds.VARIANT_TLE_DVP
    sd = ds.write_synthetic_session(str(tmp_path / "session_mixed"),
                                    n_episodes=2, episode_s=4.0,
                                    board_type=board_type, seed=3)
    r = ds.load_session(sd)

    j = 0x114 - 0x101
    assert r.is_tle[j], "the population must come from the recorded variant byte"
    assert r.cals[j].zero_counts == ds.TLE_ZERO_COUNTS
    assert r.cals[j].counts_per_psi == ds.TLE_COUNTS_PER_PSI
    # every other board on this session is legacy, including its id-block neighbours
    assert not r.is_tle[j - 1] and not r.is_tle[j + 1]

    counts = r.pressure_adc[:, j].max()
    right = r.cals[j].adc_to_pa(counts)
    wrong = ds.cal_for_variant(ds.VARIANT_7MM).adc_to_pa(counts)
    assert abs(right - wrong) > 2000.0, (
        "reading a TLE board on the 7 mm scale must be a bigger error than the "
        "7 mm firmware's own hysteresis band, or the test is not on the range "
        "where it matters")
    np.testing.assert_allclose(r.pressure_pa[:, j],
                               r.cals[j].adc_to_pa(r.pressure_adc[:, j]))


def test_adc_round_trip_is_exact_on_the_grid():
    """``pa_to_adc`` and ``adc_to_pa`` must be inverse on representable counts.

    The shooting loop puts the sensor back inside the rollout, so a round trip
    that lost a count would inject a systematic offset of 113-123 Pa into every
    regulated cycle.
    """
    for variant in (ds.VARIANT_7MM, ds.VARIANT_TLE_DVP):
        cal = ds.cal_for_variant(variant)
        counts = np.arange(0, ds.ADC_MAX + 1, 37, dtype=np.float64)
        np.testing.assert_allclose(cal.pa_to_adc(cal.adc_to_pa(counts)), counts)


def test_missing_board_type_is_refused(tmp_path, session):
    """A session without ``board_type`` cannot be split by population afterwards."""
    import shutil

    d = tmp_path / "session_nobt"
    shutil.copytree(session, d)
    meta = json.loads((d / "metadata.json").read_text())
    del meta["board_type"]
    (d / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="board_type"):
        ds.load_session(str(d))


# ---------------------------------------------------------------------------
# Episodes and windowing
# ---------------------------------------------------------------------------
def test_episodes_are_found_from_the_sync_gap(rec):
    """The generator inserts a >5 s wall-clock jump between episodes and writes
    no episode label, so the reader has to find the boundaries the way it will
    have to on a real multi-run session."""
    slices = rec.episode_slices()
    assert len(slices) == 6
    assert sum(s.stop - s.start for s in slices) == rec.n_cycles
    for s in slices:
        dt = np.diff(rec.can_sync_time_s[s])
        assert dt.min() > 0.0
        assert dt.max() < 0.05, "no within-episode gap may exceed the split threshold"


def test_windows_never_straddle_an_episode_boundary(rec):
    """The leak this whole module exists to prevent.

    A window that spans a boundary is anchored on a pressure from a different
    recording and carries a dt of several seconds in the middle of a 6.67 ms
    grid; neither is visible in the loss, both corrupt the fit.
    """
    l, ldot = ds.muscle_traces(rec, tendon=ds.MomentArmTendonGeometry())
    win = ds.make_windows(rec, l, ldot, window_cycles=200)
    assert len(win) > 0

    bounds = np.array([s.stop for s in rec.episode_slices()])
    dt = np.diff(win.t, axis=1)
    assert dt.min() > 0.0
    assert dt.max() < 0.05, "a straddling window shows up as a multi-second dt"

    # and directly: every window's time span must fit inside one episode's span
    ep_span = {int(rec.episode[s.start]):
               (rec.can_sync_time_s[s.start], rec.can_sync_time_s[s.stop - 1])
               for s in rec.episode_slices()}
    for i in range(len(win)):
        lo, hi = ep_span[int(win.episode[i])]
        assert lo - 1e-9 <= win.t[i, 0] and win.t[i, -1] <= hi + 1e-9
    assert bounds.size == 6


def test_split_is_by_whole_episode_and_seeded(rec):
    """Adjacent samples at 150 Hz are 6.67 ms apart and are not independent.

    A holdout that shares an episode with the training set reports the training
    error, which would make any guarded selection accept anything.
    """
    l, ldot = ds.muscle_traces(rec, tendon=ds.MomentArmTendonGeometry())
    win = ds.make_windows(rec, l, ldot, window_cycles=200)
    tr, ho = ds.split_episodes(win, holdout_frac=0.34, seed=11)

    te, he = set(tr.episode.tolist()), set(ho.episode.tolist())
    assert te and he
    assert te.isdisjoint(he), "an episode may not appear on both sides"
    assert te | he == set(win.episode.tolist())
    assert len(tr) + len(ho) == len(win)

    tr2, ho2 = ds.split_episodes(win, holdout_frac=0.34, seed=11)
    assert set(ho2.episode.tolist()) == he, "same seed must give the same split"

    with pytest.raises(ValueError):
        ds.split_episodes(win, holdout_frac=1.5)


def test_single_episode_refuses_to_invent_a_holdout(tmp_path):
    sd = ds.write_synthetic_session(str(tmp_path / "session_one"),
                                    n_episodes=1, episode_s=4.0, seed=5)
    r = ds.load_session(sd)
    l, ldot = ds.muscle_traces(r, tendon=ds.MomentArmTendonGeometry())
    win = ds.make_windows(r, l, ldot, window_cycles=150)
    with pytest.raises(ValueError, match="at least two"):
        ds.split_episodes(win)


def test_window_fields_are_per_node_and_anchored(rec):
    """Each row must carry its own board's population and calibration.

    A window set that spread one calibration over both populations would convert
    the TLE columns on the 7 mm scale, which is the failure this arm is most
    exposed to.
    """
    l, ldot = ds.muscle_traces(rec, tendon=ds.MomentArmTendonGeometry())
    win = ds.make_windows(rec, l, ldot, window_cycles=200)
    np.testing.assert_array_equal(win.is_tle, rec.is_tle[win.node_idx])
    np.testing.assert_allclose(win.zero_counts, rec.zero_counts[win.node_idx])
    np.testing.assert_allclose(win.p0, win.p_rec[:, 0])

    tle = win.population(True)
    seven = win.population(False)
    assert len(tle) + len(seven) == len(win)
    assert tle.is_tle.all() and not seven.is_tle.any()
    assert set(np.unique(tle.zero_counts)) == {ds.TLE_ZERO_COUNTS}
    assert set(np.unique(seven.zero_counts)) == {ds.LEGACY_ZERO_COUNTS}


def test_windows_reproduce_the_recording_they_were_cut_from(rec):
    """Indexing arithmetic, checked against the source arrays rather than trusted."""
    l, ldot = ds.muscle_traces(rec, tendon=ds.MomentArmTendonGeometry())
    kw = 120
    win = ds.make_windows(rec, l, ldot, window_cycles=kw)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(win), size=25, replace=False):
        j = int(win.node_idx[i])
        s = int(np.searchsorted(rec.can_sync_time_s, win.t[i, 0]))
        assert abs(rec.can_sync_time_s[s] - win.t[i, 0]) < 1e-12
        np.testing.assert_allclose(win.p_rec[i], rec.pressure_pa[s:s + kw, j])
        np.testing.assert_allclose(win.target_pa[i], rec.target_pa[s:s + kw, j])
        np.testing.assert_allclose(win.l[i], l[s:s + kw, j])


# ---------------------------------------------------------------------------
# Muscle geometry
# ---------------------------------------------------------------------------
def test_smooth3_leaves_the_edges_alone():
    a = np.array([[0.0], [3.0], [0.0], [0.0], [6.0]])
    s = ds.smooth3(a, axis=0)
    assert s[0, 0] == 0.0 and s[-1, 0] == 6.0
    assert s[1, 0] == pytest.approx(1.0)
    assert s[2, 0] == pytest.approx(1.0)


def test_muscle_traces_are_anchored_and_do_not_differentiate_across_episodes(rec):
    """``ldot`` computed across a boundary would be a single enormous sample.

    The generator's inter-episode jump is >5 s of wall clock against a 6.67 ms
    grid, so a gradient taken across it would be nearly zero rather than large --
    which is worse, because it is not visibly wrong.  The check is therefore on
    the *inputs* to the gradient: no ``ldot`` sample may have been formed from two
    different episodes.
    """
    l, ldot = ds.muscle_traces(rec, tendon=ds.MomentArmTendonGeometry())
    l0 = ds.l0_per_actuator()
    assert l.shape == (rec.n_cycles, 24) and ldot.shape == l.shape
    # anchored around this arm's measured rest lengths
    assert np.all(l > 0.5 * l0.min()) and np.all(l < 2.0 * l0.max())
    np.testing.assert_allclose(np.unique(l0), np.unique(np.asarray(am.L0_SEED_M)))

    # rebuild episode by episode and require an exact match
    for s in rec.episode_slices():
        seg = ds.smooth3(l0[None, :] + ds.MomentArmTendonGeometry().dlen(rec.q[s]),
                         axis=0)
        want = np.gradient(seg, rec.can_sync_time_s[s], axis=0)
        np.testing.assert_allclose(ldot[s], want)


def test_moment_arm_geometry_follows_the_measured_actuator_map():
    """``pam_k`` is board ``0x100+k``, and the pair drives its joint in opposite senses."""
    from UMArm_KINEMATICS import canarm_actuators as CA

    geo = ds.MomentArmTendonGeometry(ring_radius_m=0.03)
    q = np.zeros((1, 12))
    q[0, 3] = 0.1
    d = geo.dlen(q)[0]
    pos, neg = CA.joint_pairs()[3]
    assert d[pos - 0x101] < 0.0 < d[neg - 0x101]
    assert d[pos - 0x101] == pytest.approx(-0.003)
    others = [k for k in range(24) if k not in (pos - 0x101, neg - 0x101)]
    assert np.allclose(d[others], 0.0)


def test_tendon_auto_reports_whether_it_is_the_placeholder():
    """A fit made without the twin's real tendons must say so on its own face."""
    geo = ds.TendonKinematics.auto()
    assert hasattr(geo, "is_placeholder")
    assert geo.dlen(np.zeros((3, 12))).shape == (3, 24)


def test_the_twins_own_mjcf_gives_a_physical_l_envelope(rec):
    """The real geometry path, exercised whenever ``mjcf_generator`` emits tendons.

    Skipped while that module is a skeleton, because then ``auto`` is honestly
    returning the placeholder and there is no twin geometry to check.  What is
    asserted is the envelope rather than the values: the anchored length must sit
    on this arm's measured rest lengths at ``q = 0`` and its excursion must stay
    inside the +-25 mm of bent-arm tendon travel the reference measured, because
    an ``l`` outside that band is extrapolation for the net whatever the geometry
    behind it was.
    """
    geo = ds.TendonKinematics.auto()
    if getattr(geo, "is_placeholder", True):
        pytest.skip("digital_twin.mjcf_generator emits no tendons yet")

    assert np.allclose(geo.dlen(np.zeros((1, 12))), 0.0, atol=1e-12), \
        "the excursion is anchored at tendon_length0, so q=0 must give exactly 0"

    l, ldot = ds.muscle_traces(rec, tendon=geo)
    l0 = ds.l0_per_actuator()
    excursion = l - l0[None, :]
    assert np.abs(excursion).max() < 0.025, (
        f"tendon excursion reached {np.abs(excursion).max() * 1e3:.1f} mm; the "
        "recording is outside the envelope the fit can speak for")
    assert l.min() > 0.05, "an anchored muscle length below 50 mm is not this arm"
    assert np.abs(ldot).max() < 2.0


# ---------------------------------------------------------------------------
# Leak fit
# ---------------------------------------------------------------------------
def test_leak_fit_recovers_the_injected_leak(rec):
    """The synthetic plant's per-node leak, back out of the least squares.

    A tolerance rather than an equality, because the estimate is a slope over a
    few seconds of quantised pressure: one legacy count is 122.8 Pa against a
    leak of about 2 Pa per cycle, so the fit is noise-limited by construction and
    its accuracy is set by how much closed time the board happened to get.

    The tolerance is therefore split on that quantity rather than set to one
    loose number.  On this session, boards with at least 4 s of closed hold come
    back within 7 % and boards with a single 2-3 s segment reach 37 % -- so a
    single generous bound would pass a fit that had stopped working on the
    well-observed boards.  What this does **not** show is that 4 s is enough on
    the bench: the synthetic pressure carries only quantisation and 1.2 counts of
    noise, and a real board carries supply ripple and the arm's own motion too.
    """
    truth = np.asarray(rec.meta["synthetic_truth"]["leak_pa_s"])
    leak, info = ds.fit_leaks(rec)
    fitted = [j for j in range(24) if info[j]["fitted_pa_s"] is not None]
    assert len(fitted) >= 18, "most boards must yield a usable closed hold"

    rel = np.abs(leak[fitted] - truth[fitted]) / truth[fitted]
    assert np.median(rel) < 0.10, f"median relative error {np.median(rel):.3f}"
    assert rel.max() < 0.50, f"worst relative error {rel.max():.3f}"

    well_observed = [j for j in fitted if info[j]["closed_s"] >= 4.0]
    assert len(well_observed) >= 10
    for j in well_observed:
        assert abs(leak[j] - truth[j]) / truth[j] < 0.15, (
            f"board {j} had {info[j]['closed_s']:.1f} s of closed hold and still "
            f"missed its leak by {abs(leak[j] - truth[j]) / truth[j]:.2f}")


def test_unfitted_boards_keep_their_incoming_leak(rec):
    """Zeroing an unfitted board would let it inflate: the seam subtracts this."""
    init = np.full(24, 123.0)
    leak, info = ds.fit_leaks(rec, initial_pa_s=init,
                              min_duration_s=1e6)   # nothing can qualify
    assert all(info[j]["fitted_pa_s"] is None for j in range(24))
    np.testing.assert_allclose(leak, init)
    np.testing.assert_allclose(init, 123.0), "fit_leaks must not mutate its input"


def test_closed_segments_need_a_constant_target():
    """On a ramp, "inside the band" instants alias rising pressure."""
    t = np.arange(0, 6.0, 1 / 150.0)
    p = 10.0 * ds.PA_PER_PSI - 300.0 * t
    const = np.full_like(t, 10.0 * ds.PA_PER_PSI)
    ramp = 10.0 * ds.PA_PER_PSI - 300.0 * t

    assert len(ds.closed_segments(t, p, const, band_pa=2000.0)) == 1
    assert ds.closed_segments(t, p, ramp, band_pa=2000.0) == []

    (a, b), = ds.closed_segments(t, p, const, band_pa=2000.0)
    assert ds.fit_leak(t[a:b], p[a:b]) == pytest.approx(300.0, rel=1e-6)


def test_closed_segments_reject_a_blip_and_a_gap():
    t = np.arange(0, 6.0, 1 / 150.0)
    base = 10.0 * ds.PA_PER_PSI - 300.0 * t
    const = np.full_like(t, 10.0 * ds.PA_PER_PSI)

    blipped = base.copy()
    blipped[450:] += 8000.0        # an 8 kPa step mid-hold
    segs = ds.closed_segments(t, blipped, const, band_pa=20000.0)
    # a segment (a, b) covers samples a..b-1, so it spans the blip at 450 only
    # if it starts before it and ends after it
    assert all(not (a < 450 < b) for a, b in segs), "no segment may span the blip"
    assert segs, "the clean stretches on either side must still qualify"

    gapped_t = t.copy()
    gapped_t[450:] += 0.5          # half a second of missing record
    segs = ds.closed_segments(gapped_t, base, const, band_pa=2000.0)
    assert all(not (a < 450 < b) for a, b in segs)


def test_fit_leak_refuses_a_short_segment():
    t = np.linspace(0.0, 0.45, 60)
    with pytest.raises(ValueError, match="too short"):
        ds.fit_leak(t, np.zeros_like(t))


# ---------------------------------------------------------------------------
# The synthetic session is the real schema
# ---------------------------------------------------------------------------
def test_synthetic_session_writes_the_documented_schema(session):
    """One reader must serve both a generated and a bench session.

    A bespoke fixture would let the reader and the format drift, and the drift
    would surface for the first time on the day the first real recording arrives.
    """
    meta = json.loads(open(os.path.join(session, "metadata.json")).read())
    for key in ("schema_version", "sample_rate_hz", "state_fields",
                "input_fields", "pressure_units", "selected_ids", "board_type",
                "adc_ranges", "sync_semantics"):
        assert key in meta, key
    assert meta["pressure_units"] == "adc_counts"
    assert len(meta["selected_ids"]) == 24 and len(meta["board_type"]) == 24

    man = json.loads(open(os.path.join(session, "manifest.json")).read())
    assert man["total_samples"] == sum(c["samples"] for c in man["chunks"])
    assert man["active"] is False

    with open(os.path.join(session, man["chunks"][0]["path"])) as fh:
        row = json.loads(fh.readline())
    assert set(row["robot_state"]) == {"q", "qdot", "pressure_adc"}
    assert len(row["robot_state"]["pressure_adc"]) == 24
    assert len(row["robot_state"]["q"]) == 12
    assert len(row["input"]["target_adc"]) == 24
    assert "can_sync_time_s" in row and "board_type" in row
    assert all(isinstance(v, int) for v in row["robot_state"]["pressure_adc"]), \
        "pressures are recorded RAW; a session that stored psi would be "\
        "invalidated by every recalibration"


def test_synthetic_targets_respect_the_operator_limit(rec):
    """CONTRACT section 8: no commanded line above 30 psi, no pair above 30 psi.

    Checked on the generated session so the excitation it produces could be
    replayed on the bench without breaking the rule it is meant to respect.
    """
    psi = rec.target_pa / ds.PA_PER_PSI
    assert psi.max() <= 30.0 + 1e-9
    from UMArm_KINEMATICS import canarm_actuators as CA
    for pos, neg in CA.joint_pairs():
        s = psi[:, pos - 0x101] + psi[:, neg - 0x101]
        assert s.max() <= 30.0 + 1e-9


def test_this_reader_agrees_with_replays_reader(session):
    """Two modules now read this schema, and they must not drift apart.

    ``replay.load_recording`` reads a ``session_*/`` for the scoring side and
    ``dataset.load_session`` for the fitting side.  If they ever disagree on the
    calibration, the twin would be scored in one coordinate against a net fitted
    in another, and the gap would read as model error.

    The one difference is deliberate and is asserted here rather than left
    implicit: ``replay`` subtracts the two known rest offsets by default (``0x104``
    +1.1 psi, ``0x110`` +5.1 psi) because it scores against physical pressure,
    while this reader keeps them because both firmwares compute their error from
    the board's own reported pressure, offset included.  Turn that correction off
    and the two must agree exactly.
    """
    replay = pytest.importorskip("digital_twin.replay")

    a = ds.load_session(session)
    b = replay.load_recording(session, rest_offset_psi={})

    np.testing.assert_array_equal(a.can_sync_time_s, b.t_sync_s)
    np.testing.assert_array_equal(a.board_type, b.board_type)
    np.testing.assert_array_equal(a.is_tle, b.is_tle)
    np.testing.assert_array_equal(a.pressure_adc, b.p_adc)
    np.testing.assert_array_equal(a.target_adc, b.target_adc)
    np.testing.assert_allclose(a.pressure_pa, b.p_pa, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(a.target_pa, b.target_pa, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(a.q, b.q_rad)

    # and the offset path itself, on the two boards it applies to
    c = ds.load_session(session, rest_offset_psi=ds.KNOWN_REST_OFFSET_PSI)
    for base, psi in ds.KNOWN_REST_OFFSET_PSI.items():
        j = base - 0x101
        np.testing.assert_allclose(a.pressure_pa[:, j] - c.pressure_pa[:, j],
                                   psi * ds.PA_PER_PSI)
        np.testing.assert_allclose(a.target_pa[:, j], c.target_pa[:, j]), \
            "a setpoint is a number the host chose, not one the sensor reported"
    untouched = [j for j in range(24)
                 if (j + 0x101) not in ds.KNOWN_REST_OFFSET_PSI]
    np.testing.assert_array_equal(a.pressure_pa[:, untouched],
                                  c.pressure_pa[:, untouched])


def test_dataset_does_not_import_torch():
    """CONTRACT section 7: ``train_actuator_net`` is the only module here allowed
    to import torch, because ``sim_core`` steps this side of the package at 1 ms
    on a bench laptop with no CUDA.  Checked on the import graph rather than on
    the source text, so a transitive import through ``actuator_model`` would fail
    it too."""
    import importlib
    import subprocess

    code = ("import sys; import digital_twin.dataset; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code],
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(ds.__file__))),
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "importing digital_twin.dataset pulled torch into sys.modules; "
        f"stderr: {r.stderr[-400:]}")
    assert importlib is not None


def test_build_windows_is_the_one_path_from_file_to_windows(session):
    tr, ho, r, l, ldot = ds.build_windows(
        session, window_cycles=200, holdout_frac=0.34, seed=11,
        tendon=ds.MomentArmTendonGeometry())
    assert len(tr) > 0 and len(ho) > 0
    assert set(tr.episode.tolist()).isdisjoint(set(ho.episode.tolist()))
    assert l.shape == (r.n_cycles, 24)
