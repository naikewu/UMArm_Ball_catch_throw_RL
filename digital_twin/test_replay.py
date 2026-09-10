"""Replay's contract: the sync clock, determinism, and the force clip.

Most rollouts here run against :class:`SpringArm`, a deterministic stand-in that
reproduces the clock semantics ``CONTRACT.md`` section 4 pins -- first call pins
the origin, a stale target is a no-op, the sub-quantum remainder accumulates --
and nothing else.  It is used because a stand-in isolates the failure: any
non-determinism these tests find is replay's, not MuJoCo's, and the arithmetic
is checkable by hand.

The last three tests then run the same loop against the real
``sim_core.SimArm``, skipping if that sibling does not build.  What the
stand-in tests do NOT show is whether the twin's physics is right; they show
that replay drives whatever arm it is given on the recorded sync clock, samples
before it commands, and refuses a saturated rollout.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from digital_twin import replay as R


# ---------------------------------------------------------------------------
# A deterministic stand-in for SimArm
# ---------------------------------------------------------------------------


class SpringArm:
    """Twenty-four first-order pressure lags driving twelve damped oscillators.

    Not a model of anything -- it exists so replay's loop, clock and clamp check
    can be exercised without MuJoCo.  Its arithmetic is float, in a fixed order,
    with no RNG and no clock read, which is what lets the determinism test mean
    something: any non-determinism a test finds is replay's.
    """

    def __init__(self, *, dt=0.001, node_logic_every=5, fill_rate_hz=8.0,
                 ctrl_scale_n_per_pa=0.005, omega_rad_s=11.3, zeta=0.05,
                 gain_rad_per_n=2.0e-4):
        self.dt = float(dt)
        self.node_logic_every = int(node_logic_every)
        self.fill_rate_hz = float(fill_rate_hz)
        self.ctrl_scale = float(ctrl_scale_n_per_pa)
        self.omega = float(omega_rad_s)
        self.zeta = float(zeta)
        self.gain = float(gain_rad_per_n)

        self.p = np.zeros(R.N_NODES)
        self.target = np.zeros(R.N_NODES)
        self.q_rad = np.zeros(R.N_JOINTS)
        self.qd = np.zeros(R.N_JOINTS)

        self.quanta_done = 0
        self._t_origin = None
        self._t_target = None
        self._accum = 0.0
        self._last_node_quantum = 0
        self.advance_marks = []      # every t advance_to was asked for
        self.tick_marks = []         # every t a target was promoted at

    # -- the clock, per CONTRACT.md section 4 --------------------------------
    def advance_to(self, t):
        t = float(t)
        self.advance_marks.append(t)
        if self._t_target is None:
            self._t_origin = t
            self._t_target = t
            return
        if t <= self._t_target:
            return
        self._accum += t - self._t_target
        self._t_target = t
        n = int(self._accum / self.dt + 1e-9)
        if n > 0:
            self._accum -= n * self.dt
            for _ in range(n):
                self._step_quantum()

    def _step_quantum(self):
        if self.quanta_done % self.node_logic_every == 0:
            node_dt = (self.quanta_done - self._last_node_quantum) * self.dt
            self._last_node_quantum = self.quanta_done
            self.p += (self.target - self.p) * self.fill_rate_hz * node_dt
            np.maximum(self.p, 0.0, out=self.p)
        ctrl = self.ctrl_n()
        tau = self.gain * (ctrl[0:24:2] - ctrl[1:24:2])
        qdd = tau - 2.0 * self.zeta * self.omega * self.qd - self.omega ** 2 * self.q_rad
        self.qd = self.qd + qdd * self.dt
        self.q_rad = self.q_rad + self.qd * self.dt
        self.quanta_done += 1

    # -- the seams replay uses ----------------------------------------------
    def set_targets_pa(self, mapping, now):
        self.tick_marks.append(float(now))
        for base, pa in mapping.items():
            self.target[int(base) - 0x101] = float(pa)

    def q(self):
        return self.q_rad

    def pressures_pa(self):
        return self.p

    def ctrl_n(self):
        return np.clip(-self.ctrl_scale * self.p, -R.FORCE_CLIP_N, 0.0)


class PlaybackArm:
    """Replays a fixed ``q`` trace, one row per sample call, bit for bit.

    Used where a test needs the twin's trace to be EXACTLY the recording's, so
    that any difference a metric reports is the metric's own doing.
    """

    def __init__(self, q_rad, p_pa=None):
        self.trace = np.asarray(q_rad, dtype=np.float64)
        self.p_trace = None if p_pa is None else np.asarray(p_pa, dtype=np.float64)
        self.k = 0
        self.p = np.zeros(R.N_NODES)

    def advance_to(self, t):
        pass

    def set_targets_pa(self, mapping, now):
        pass

    def q(self):
        row = self.trace[min(self.k, self.trace.shape[0] - 1)]
        if self.p_trace is not None:
            self.p = self.p_trace[min(self.k, self.p_trace.shape[0] - 1)]
        self.k += 1
        return row

    def pressures_pa(self):
        return self.p

    def ctrl_n(self):
        return np.zeros(R.N_NODES)


# ---------------------------------------------------------------------------
# Synthetic recordings
# ---------------------------------------------------------------------------

#: One TLE board deliberately parked outside its id block, reproducing the
#: 2026-08 bench session where a TLE board sat at 0x114.  Every population test
#: in this package has to survive it, since the split is read from the variant
#: byte and not from the id range.
BOARD_TYPE = ([R.VARIANT_TLE_DVP] * 8 + [R.VARIANT_7MM] * 11
              + [R.VARIANT_TLE_DVP] + [R.VARIANT_7MM] * 4)


def synthetic_recording(*, n=900, fs_hz=150.0, jitter_s=8.0e-4, drive_psi=8.0,
                        step_row=300, board_type=None):
    """A recording whose sync clock JITTERS, so a nominal grid cannot pass for it."""
    k = np.arange(n, dtype=np.float64)
    # A deterministic wander of up to 0.8 ms about the nominal 6.667 ms period:
    # smaller than the collector's measured 25.069 ms worst case, large enough
    # that a rollout driven on a nominal grid lands somewhere else entirely.
    t = 100.0 + k / fs_hz + jitter_s * np.sin(0.7 * k)
    bt = np.asarray(board_type if board_type is not None else BOARD_TYPE, dtype=int)

    target_adc = np.zeros((n, R.N_NODES))
    p_adc = np.zeros((n, R.N_NODES))
    for c in range(R.N_NODES):
        zero, per_psi = R.VARIANT_CAL[int(bt[c])]
        psi = np.where(k >= step_row, drive_psi if c % 2 == 0 else 0.5, 0.5)
        target_adc[:, c] = zero + psi * per_psi
        # The measured column lags the commanded one by a first-order fill, so
        # the recording is not simply its own input played back.
        lag = np.zeros(n)
        for i in range(1, n):
            lag[i] = lag[i - 1] + (psi[i] - lag[i - 1]) * 0.05
        p_adc[:, c] = zero + lag * per_psi

    q = np.zeros((n, R.N_JOINTS))
    for j in range(R.N_JOINTS):
        q[:, j] = np.radians(0.5 * (j + 1)) * np.sin(2.0 * np.pi * 1.8 * (t - t[0]) + j)
    return R.recording_from_arrays(t, q, p_adc, target_adc, bt)


# ---------------------------------------------------------------------------
# Units and loading
# ---------------------------------------------------------------------------


def test_the_calibration_copy_matches_tlelib():
    """The duplicated transfer functions must not drift from the host bus stack."""
    import sys
    ws = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    sys.path.insert(0, os.path.join(ws, "TLE_PCB"))
    try:
        from tlelib import proto as P
    except Exception:  # pragma: no cover - tlelib absent is not this test's business
        pytest.skip("tlelib not importable in this environment")
    finally:
        sys.path.pop(0)
    assert R.VARIANT_CAL[R.VARIANT_TLE_DVP] == (P.TLE_ZERO_COUNTS, P.TLE_COUNTS_PER_PSI)
    assert R.VARIANT_CAL[R.VARIANT_7MM] == (P.LEGACY_ZERO_COUNTS, P.LEGACY_COUNTS_PER_PSI)
    assert R.VARIANT_TLE_DVP == P.VARIANT_TLE_DVP


def test_pressure_is_converted_on_the_board_s_own_variant():
    """Reading a TLE board on the 7 mm scale is a 10 % psi error at the top of the range."""
    counts = 943.75 + 60.78125 * 20.0
    tle = R.counts_to_pa(counts, R.VARIANT_TLE_DVP)
    legacy = R.counts_to_pa(counts, R.VARIANT_7MM)
    assert tle == pytest.approx(20.0 * R.PA_PER_PSI)
    assert abs(legacy - tle) / tle > 0.05


def test_session_round_trip_through_the_schema_layout(tmp_path):
    rec = synthetic_recording(n=200)
    d = R.save_recording(str(tmp_path / "session_test"), rec)
    back = R.load_recording(d, rest_offset_psi={})
    assert np.allclose(back.t_sync_s, rec.t_sync_s)
    assert np.allclose(back.q_rad, rec.q_rad)
    assert np.allclose(back.p_pa, rec.p_pa)
    assert np.array_equal(back.board_type, rec.board_type)
    assert np.array_equal(back.is_tle, rec.is_tle)
    # 1 % rather than tighter: this recording's clock jitters by up to 0.8 ms on
    # purpose, which moves the median gap off the nominal 6.667 ms.
    assert back.rate_hz == pytest.approx(150.0, rel=1e-2)


def test_a_recording_without_board_type_is_refused(tmp_path):
    rec = synthetic_recording(n=50)
    d = R.save_recording(str(tmp_path / "session_nobt"), rec)
    for name in os.listdir(d):
        if name.startswith("samples_chunk_"):
            path = os.path.join(d, name)
            rows = [json.loads(line) for line in open(path, encoding="utf-8")]
            for row in rows:
                row.pop("board_type")
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
    os.remove(os.path.join(d, "metadata.json"))
    with pytest.raises(ValueError, match="board_type"):
        R.load_recording(d)


def test_a_non_increasing_sync_clock_is_refused(tmp_path):
    rec = synthetic_recording(n=60)
    t = rec.t_sync_s.copy()
    t[30] = t[29]                     # a duplicated cycle
    bad = R.recording_from_arrays(t, rec.q_rad, rec.p_adc, rec.target_adc,
                                  rec.board_type)
    d = R.save_recording(str(tmp_path / "session_dup"), bad)
    with pytest.raises(ValueError, match="increasing"):
        R.load_recording(d)


def test_the_rest_offset_is_removed_from_the_measurement_and_not_from_the_command():
    """0x110 reads +5.1 psi at rest; the board still regulates against that reading."""
    rec_off = synthetic_recording(n=20)
    col = int(np.flatnonzero(rec_off.ids == 0x110)[0])
    with_offset = R.recording_from_arrays(
        rec_off.t_sync_s, rec_off.q_rad, rec_off.p_adc, rec_off.target_adc,
        rec_off.board_type, rest_offset_psi=R.KNOWN_REST_OFFSET_PSI)
    dp = rec_off.p_pa[:, col] - with_offset.p_pa[:, col]
    assert np.allclose(dp, 5.1 * R.PA_PER_PSI)
    assert np.allclose(rec_off.target_pa[:, col], with_offset.target_pa[:, col])


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------


def test_a_rollout_repeated_twice_is_bit_identical():
    rec = synthetic_recording()
    a = R.rollout(rec, arm_factory=SpringArm)
    b = R.rollout(rec, arm_factory=SpringArm)
    for name in ("t_s", "q_rad", "p_pa", "ctrl_n", "target_pa", "row_index"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.ctrl_min_n == b.ctrl_min_n


def test_the_rollout_is_driven_on_the_recording_s_own_sync_stamps():
    """Not a nominal 6.667 ms grid.  This is the reason the schema records the edge."""
    rec = synthetic_recording(n=300)
    arm = SpringArm()
    R.rollout(rec, arm=arm)
    assert np.array_equal(np.asarray(arm.advance_marks), rec.t_sync_s)
    assert np.array_equal(np.asarray(arm.tick_marks), rec.t_sync_s)
    nominal = rec.t_sync_s[0] + np.arange(rec.n) / 150.0
    assert np.max(np.abs(rec.t_sync_s - nominal)) > 5.0e-4


def test_the_sample_precedes_the_target_it_is_recorded_beside():
    """Row k is the arm BEFORE edge k promotes its targets, as the firmware replies."""
    rec = synthetic_recording(n=40, step_row=10)
    arm = SpringArm()
    roll = R.rollout(rec, arm=arm)
    # Row 0 is sampled before anything was ever commanded, so it must be at rest
    # no matter what row 0 commands.
    assert np.all(roll.p_pa[0] == 0.0)
    assert np.all(np.isfinite(roll.target_pa[0]))


def test_the_tail_samples_with_no_further_edges():
    rec = synthetic_recording(n=100)
    roll = R.rollout(rec, arm_factory=SpringArm, tail_s=1.0)
    n_tail = int(round(1.0 / R.TAIL_DT_S))
    assert roll.n == rec.n + n_tail
    assert np.all(roll.row_index[rec.n:] == -1)
    assert np.all(np.isnan(roll.target_pa[rec.n:]))
    assert np.all(np.diff(roll.t_s[rec.n:]) == pytest.approx(R.TAIL_DT_S))


def test_the_force_clip_raises_rather_than_warning():
    rec = synthetic_recording(n=400, drive_psi=30.0)
    # 0.03 N per Pa puts 30 psi (206 843 Pa) at 6205 N, well past the 4000 N clip.
    hot = lambda: SpringArm(ctrl_scale_n_per_pa=0.03)
    with pytest.raises(R.ClampViolation, match="pull-only clip"):
        R.rollout(rec, arm_factory=hot)
    roll = R.rollout(rec, arm_factory=hot, assert_no_clamp_on_touch=False)
    assert roll.clamped is True
    assert roll.ctrl_min_n == pytest.approx(-R.FORCE_CLIP_N)
    with pytest.raises(R.ClampViolation):
        R.assert_no_clamp(roll)


def test_a_rollout_inside_the_envelope_keeps_its_headroom():
    rec = synthetic_recording(n=400, drive_psi=8.0)
    roll = R.rollout(rec, arm_factory=SpringArm)
    assert roll.clamped is False
    assert roll.ctrl_min_n > -R.FORCE_CLIP_N
    R.assert_no_clamp(roll)          # must not raise


def test_the_accessor_path_is_recorded_rather_than_assumed():
    """A rollout that silently read zeros because an accessor was missing is the hazard."""
    rec = synthetic_recording(n=50)
    roll = R.rollout(rec, arm_factory=SpringArm)
    assert roll.meta["accessors"]["q_via"] == "arm.q()"
    assert roll.meta["accessors"]["p_via"] == "arm.pressures_pa()"
    assert roll.meta["accessors"]["ctrl_via"] == "arm.ctrl_n()"


def test_an_arm_with_no_target_seam_names_the_seam():
    class Mute:
        def advance_to(self, t):
            pass

        def q(self):
            return np.zeros(R.N_JOINTS)

    rec = synthetic_recording(n=10)
    with pytest.raises(TypeError, match="set_targets_pa"):
        R.rollout(rec, arm=Mute())


def test_truncation_keeps_the_sync_origin():
    rec = synthetic_recording(n=600)
    short = rec.slice_time(1.0)
    assert short.t_rel_s[0] == 0.0
    assert short.t_rel_s[-1] <= 1.0
    assert short.n < rec.n
    assert np.array_equal(short.t_sync_s, rec.t_sync_s[:short.n])


def test_a_truncation_that_removes_everything_is_refused():
    rec = synthetic_recording(n=100)
    with pytest.raises(ValueError, match="cuts away"):
        rec.slice_time(-1.0)


def test_a_rollout_saves_and_loads_without_pickle(tmp_path):
    rec = synthetic_recording(n=80)
    roll = R.rollout(rec, arm_factory=SpringArm)
    p = str(tmp_path / "roll.npz")
    roll.save(p)
    back = R.Rollout.load(p)
    assert np.array_equal(back.q_rad, roll.q_rad)
    assert back.meta["n_rows"] == roll.meta["n_rows"]


# ---------------------------------------------------------------------------
# Against the real SimArm, when it is importable and buildable
# ---------------------------------------------------------------------------


def _real_arm_factory(**kwargs):
    sim_core = pytest.importorskip("digital_twin.sim_core",
                                   reason="sim_core is a sibling module under construction")
    try:
        return sim_core.SimArm(**kwargs)
    except Exception as exc:               # a mid-edit sibling is not this test's failure
        pytest.skip(f"SimArm did not build: {exc!r}")


def test_a_short_rollout_through_the_real_simarm_repeats_bit_for_bit():
    """The determinism claim, against the arm it is actually claimed for.

    Short on purpose -- a few hundred cycles is enough to catch a clock read or a
    thread, which are the only ways determinism is lost, and it keeps this test
    inside a second of MuJoCo stepping.
    """
    rec = synthetic_recording(n=150, drive_psi=8.0)
    a = R.rollout(rec, arm_factory=_real_arm_factory)
    b = R.rollout(rec, arm_factory=_real_arm_factory)
    for name in ("t_s", "q_rad", "p_pa", "ctrl_n"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.meta["accessors"]["p_via"] == "arm.pressures_pa()"
    assert a.meta["accessors"]["ctrl_via"] == "data.ctrl"


def test_the_real_arm_is_built_on_the_recorded_variant_bytes():
    """A TLE board at 0x114 must become a TLE node, valve, failsafe and all."""
    rec = synthetic_recording(n=20)
    arm = _real_arm_factory(
        variants={int(b): int(v) for b, v in zip(rec.ids, rec.board_type)})
    R.rollout(rec, arm=arm)
    assert arm.variants[0x114] == R.VARIANT_TLE_DVP
    assert type(arm.nodes[0x114]).__name__ == type(arm.nodes[0x101]).__name__
    assert type(arm.nodes[0x114]).__name__ != type(arm.nodes[0x109]).__name__


def test_the_real_arm_keeps_its_clamp_headroom_inside_the_envelope():
    rec = synthetic_recording(n=300, drive_psi=8.0)
    roll = R.rollout(rec, arm_factory=_real_arm_factory)
    assert roll.clamped is False
    assert roll.ctrl_min_n > -R.FORCE_CLIP_N


def test_the_force_clip_matches_the_generator_s_own_force_range():
    """Two constants naming the same clip must not drift apart.

    ``replay.FORCE_CLIP_N`` arms the ClampViolation; ``mjcf_generator``'s
    ``FORCE_RANGE_N`` is what MuJoCo actually saturates at.  If they disagree,
    either the check fires on rollouts that are fine or -- worse -- it stays
    silent through real saturation.
    """
    mg = pytest.importorskip("digital_twin.mjcf_generator")
    am = pytest.importorskip("digital_twin.actuator_model")
    assert R.FORCE_CLIP_N == mg.FORCE_RANGE_N == am.FORCE_CLIP_N
