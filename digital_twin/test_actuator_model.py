r"""Offline acceptance for :mod:`digital_twin.actuator_model`.

No hardware, no MuJoCo, no torch.  Everything here runs against arrays built in
the test, because the module it covers is stepped 1000 times a second inside a
rollout and the properties that matter are arithmetic ones.

WHAT THESE TESTS DO NOT SHOW.  Nothing below says the model is *right* about
this arm.  There is no fitted checkpoint yet, so every constant under test is a
seed: the tests check that the seeds are the ones the module claims they are
(measured geometry where it says measured), that the arithmetic matches
``CONTRACT.md`` section 2, and that the guards fire.  Accuracy against the real
boards is a question for the campaign and for ``train_actuator_net.py``, and it
cannot be asked here.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
for _p in (_WS,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from digital_twin import actuator_model as am  # noqa: E402


# The bench arrangement of 2026-08: eight TLE/DVP boards at 0x101-0x108 and
# sixteen 7 mm boards below.  Built from variant bytes rather than from the id
# range on purpose -- that is the code path a recording feeds.
_VARIANTS_BENCH = [am.VARIANT_TLE_DVP] * 8 + [0x00] * 16


def _model(**kw):
    return am.ActuatorModel.fresh(is_tle=am.is_tle_from_variants(_VARIANTS_BENCH), **kw)


def _rack(rng, *, p_hi=25.0):
    """A plausible whole-rack state: pressures, targets, lengths, rates."""
    p = rng.uniform(0.0, p_hi * am.PA_PER_PSI, am.N_ACT)
    tgt = rng.uniform(0.0, 30.0 * am.PA_PER_PSI, am.N_ACT)
    l = np.asarray(am.L0_SEED_M)[am.SEG_INDEX] + rng.uniform(-0.025, 0.025, am.N_ACT)
    ldot = rng.uniform(-0.2, 0.2, am.N_ACT)
    return p, tgt, l, ldot


# ---------------------------------------------------------------------------
# The module's own ground rules
# ---------------------------------------------------------------------------

def test_the_module_does_not_drag_torch_in():
    """``sim_core`` steps this at 1 ms on a machine with no CUDA.

    Asked in a **subprocess** that imports this module and nothing else.  The
    obvious version -- asserting on ``sys.modules`` in-process -- passes when
    this file runs alone and fails the moment it shares a pytest session with
    ``test_train_actuator_net.py``, which imports torch legitimately.  That
    version tests the session's import history rather than this module's import
    graph, and the second is the property worth pinning.
    """
    import subprocess
    code = ("import sys; sys.path.insert(0, r'" + _WS + "'); "
            "import digital_twin.actuator_model; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False", (
        "torch was imported by digital_twin.actuator_model or by something it "
        "imports; the rollout must run without it")


def test_l0_is_this_arm_s_measured_geometry_and_bf_is_not():
    """The seeds' provenance, asserted so a re-copy of the RS485 table is loud.

    ``l0`` must be the measured ``LL`` column and nothing else.  ``bf`` must
    *not* be the reference's verbatim ``Bf``, which is the mistake that looks
    most reasonable: it would move this arm's rest length from 1.42 to 2.36
    times the slack boundary ``bf/sqrt(3)``.
    """
    from UMArm_KINEMATICS import canarm_params as cp

    assert cp.MEASURED, "the chain table is a placeholder; l0 is not measured"
    np.testing.assert_allclose(am.L0_SEED_M, cp.CANARM_PARAMS[:, 8], rtol=0, atol=0)
    # AA1 + LL + AA2 is the measured plate gap, which is what makes LL the
    # actuator's free span rather than an arbitrary column.
    span = cp.CANARM_PARAMS[:, 4] + cp.CANARM_PARAMS[:, 8] + cp.CANARM_PARAMS[:, 5]
    np.testing.assert_allclose(span[0], cp.CANARM_PLATE_CHAIN_M[0], atol=1e-9)

    assert not np.allclose(am.BF_SEED_M, am.MUSCLE_REFERENCE["Bf"]), (
        "bf was seeded with the reference arm's verbatim Bf")
    margin = np.asarray(am.L0_SEED_M) / (np.asarray(am.BF_SEED_M) / np.sqrt(3.0))
    ref_margin = np.asarray(am.MUSCLE_REFERENCE["Bf"]) / np.sqrt(3.0)
    ref_margin = np.asarray([0.1072, 0.0968, 0.0952]) / ref_margin
    np.testing.assert_allclose(margin, ref_margin, rtol=1e-12)


def test_population_comes_from_the_variant_byte_not_the_id_range():
    """A TLE board at 0x114 must be read as TLE; it sat there all of 2026-08."""
    variants = [0x00] * am.N_ACT
    variants[0x114 - 0x101] = am.VARIANT_TLE_DVP
    mask = am.is_tle_from_variants(variants)
    assert mask[0x114 - 0x101]
    assert mask.sum() == 1
    assert not mask[0]                       # 0x101 is a 7 mm board in this frame

    m = am.ActuatorModel.fresh(is_tle=mask)
    assert m.pop_index[0x114 - 0x101] == am.POP_TLE
    assert m.pop_index[0] == am.POP_7MM

    with pytest.raises(ValueError, match="one per board"):
        am.is_tle_from_variants([0x00] * 23)


def test_segment_and_population_are_separate_partitions():
    """Segment 0 coincides with the TLE block today; the code must not rely on it."""
    m = _model()
    np.testing.assert_array_equal(m.seg_index, np.repeat(np.arange(3), 8))
    moved = list(_VARIANTS_BENCH)
    moved[0], moved[20] = 0x00, am.VARIANT_TLE_DVP
    m2 = am.ActuatorModel.fresh(is_tle=am.is_tle_from_variants(moved))
    np.testing.assert_array_equal(m2.seg_index, m.seg_index)
    assert m2.pop_index[20] == am.POP_TLE and m2.seg_index[20] == 2


# ---------------------------------------------------------------------------
# The net
# ---------------------------------------------------------------------------

def test_forward_full_state_jacobian_matches_finite_differences():
    """``dy/dp`` against central differences on the whole feature construction.

    The difference is taken on ``p_pa`` and not on a feature column, because
    pressure enters the row twice and the point of the test is the total
    derivative.  Step is 1.0 Pa -- about 1/60 of an ADC count on a TLE board, so
    small against the physics and large against float64 round-off at a pressure
    of order 1e5 Pa.
    """
    rng = np.random.default_rng(7)
    m = _model(seed=11)
    p, tgt, l, ldot = _rack(rng)
    tle = m.is_tle.astype(float)

    _, _, _, dy_dp = m.net.forward_full(m._norm_x(p, tgt, tle, l, ldot))

    h = 1.0  # Pa
    y_plus = m.net.forward(m._norm_x(p + h, tgt, tle, l, ldot))
    y_minus = m.net.forward(m._norm_x(p - h, tgt, tle, l, ldot))
    fd = (y_plus - y_minus) / (2.0 * h)

    rel = np.abs(dy_dp - fd) / np.maximum(np.abs(fd), 1e-12)
    assert rel.max() < 1e-6, f"worst relative error {rel.max():.3e}"


def test_ignoring_the_error_column_would_have_been_a_different_function():
    """The trap the state Jacobian exists to avoid, made explicit.

    Chaining along column 0 alone is what the RS485 reference did, correctly,
    because its feature row had no error column.  Here it drops a term of the
    same magnitude and the opposite sign, so the two answers must differ
    substantially -- if they ever agree, ``E_SCALE_PA`` has been decoupled from
    ``P_SCALE_PA`` and this test is the place to notice.
    """
    rng = np.random.default_rng(3)
    m = _model(seed=5)
    p, tgt, l, ldot = _rack(rng)
    x = m._norm_x(p, tgt, m.is_tle.astype(float), l, ldot)

    _, _, _, full = m.net.forward_full(x)
    _, _, _, p_only = m.net.forward_full(x, dx_dp=(1.0 / am.P_SCALE_PA, 0, 0, 0, 0))
    rel = np.abs(full - p_only) / np.maximum(np.abs(full), 1e-12)
    assert rel.min() > 0.1, (
        "the p-only chain agrees with the total derivative; the error column is "
        "no longer carrying pressure")

    with pytest.raises(ValueError, match="dx_dp"):
        m.net.forward_full(x, dx_dp=(1.0, 0.0))


def test_batch_and_scalar_flow_agree():
    """One GEMM for the rack against 24 single-row calls, to BLAS tolerance."""
    rng = np.random.default_rng(101)
    m = _model(seed=2)
    m.fill_gain[:] = rng.uniform(0.3, 3.0, am.N_ACT)
    m.vent_gain[:] = rng.uniform(0.3, 3.0, am.N_ACT)
    p, tgt, l, ldot = _rack(rng)

    batch = m.net_flow_pa_s_batch(p, tgt, l, ldot)
    one_at_a_time = np.array([
        m.net_flow_pa_s(k + 1, p[k], tgt[k], l[k], ldot[k]) for k in range(am.N_ACT)])
    np.testing.assert_allclose(batch, one_at_a_time, rtol=1e-12, atol=1e-9)

    m.leak_pa_s[:] = rng.uniform(0.0, 900.0, am.N_ACT)
    np.testing.assert_allclose(
        m.flow_pa_s_batch(p, tgt, l, ldot),
        [m.flow_pa_s(k + 1, p[k], tgt[k], l[k], ldot[k]) for k in range(am.N_ACT)],
        rtol=1e-12, atol=1e-9)

    with pytest.raises(ValueError, match="node_id"):
        m.net_flow_pa_s(0, 0.0, 0.0, 0.15, 0.0)
    with pytest.raises(ValueError, match="node_id"):
        m.net_flow_pa_s(25, 0.0, 0.0, 0.15, 0.0)


def test_the_population_bit_actually_reaches_the_net():
    """Two boards with identical state but different variants must differ.

    Otherwise ``is_tle`` is a column of zeros in disguise and the net is
    describing one plant, which is the failure ``CONTRACT.md`` section 0
    departure 2 exists to prevent.
    """
    m = _model(seed=4)
    p, tgt, l, ldot = 8.0 * am.PA_PER_PSI, 20.0 * am.PA_PER_PSI, 0.16, 0.0
    x_tle = m._norm_x(p, tgt, 1.0, l, ldot)
    x_7mm = m._norm_x(p, tgt, 0.0, l, ldot)
    assert abs(float(m.net.forward(x_tle)[0] - m.net.forward(x_7mm)[0])) > 1e-6


# ---------------------------------------------------------------------------
# The gain blend
# ---------------------------------------------------------------------------

def test_gain_blend_reduces_to_the_hard_switch_as_width_goes_to_zero():
    """``CONTRACT.md`` section 0 departure 3: the blend's limit is the reference.

    Widths are swept down by decades and the blend is compared against the
    reference's three-way branch away from ``e = 0``.  At ``e = 0`` exactly the
    blend returns the midpoint of the two gains, which is stated in
    :meth:`ActuatorModel.gain` and asserted here so the departure is recorded
    rather than discovered.
    """
    m = _model()
    m.fill_gain[:] = 2.5
    m.vent_gain[:] = 0.4
    e = np.array([-30e3, -2e3, -1.0, 1.0, 2e3, 30e3])
    hard = np.where(e > 0, 2.5, 0.4)

    worst = []
    for width in (2000.0, 20.0, 0.2, 1e-9):
        m.blend_width_pa[:] = width
        g = m.gain(np.zeros(e.size, dtype=int), e)
        worst.append(float(np.abs(g - hard).max()))
    assert worst == sorted(worst, reverse=True), (
        f"the blend did not tighten monotonically onto the switch: {worst}")
    assert worst[-1] < 1e-12

    m.blend_width_pa[:] = 1e-9
    assert float(m.gain(0, 0.0)) == pytest.approx(0.5 * (2.5 + 0.4))

    with pytest.raises(ValueError, match="blend_width_pa"):
        am.ActuatorModel.fresh(is_tle=am.is_tle_from_variants(_VARIANTS_BENCH),
                               blend_width_pa=(0.0, 6000.0))


def test_each_population_blends_across_its_own_band():
    """A 3 kPa error is inside the TLE band and outside the 7 mm one."""
    m = _model()
    m.fill_gain[:] = 3.0
    m.vent_gain[:] = 1.0
    e = 3000.0
    g_7mm = float(m.gain(8, e))           # 0x109, a 7 mm board, width 2000 Pa
    g_tle = float(m.gain(0, e))           # 0x101, a TLE board,  width 6000 Pa
    assert g_7mm > g_tle, (
        "the narrower 7 mm band must commit to fill sooner than the TLE band")
    assert 1.0 < g_tle < g_7mm < 3.0

    # Monotone in the error for every board, which is what makes the blend a
    # usable surrogate for the switch rather than merely a smooth function.
    sweep = np.linspace(-30e3, 30e3, 61)
    for idx in (0, 8, 23):
        g = m.gain(np.full(sweep.size, idx), sweep)
        assert np.all(np.diff(g) > 0.0)


# ---------------------------------------------------------------------------
# Force and damping
# ---------------------------------------------------------------------------

def test_force_is_pull_only_and_clipped():
    m = _model()
    l0 = m.l0_per_act

    # Formula match, at rest length and 12 psi -- the campaign's drive pressure.
    p = np.full(am.N_ACT, 12.0 * am.PA_PER_PSI)
    want = m.coeff_per_act * p * (m.bf2_per_act - 3.0 * l0 * l0)
    np.testing.assert_allclose(m.force_n(p, np.zeros(am.N_ACT)), want, rtol=1e-12)
    assert np.all(want < 0.0), "a pressurised muscle at rest length must pull"

    # Contracted past bf/sqrt(3): slack, exactly zero, not a small push.
    dlen_slack = (m.bf2_per_act ** 0.5) / np.sqrt(3.0) - l0 - 1e-3
    np.testing.assert_array_equal(
        m.force_n(p, dlen_slack), np.zeros(am.N_ACT))

    # Zero pressure is zero force in both directions.
    np.testing.assert_array_equal(
        m.force_n(np.zeros(am.N_ACT), np.zeros(am.N_ACT)), np.zeros(am.N_ACT))

    # Far past the guard rail: exactly -FORCE_CLIP_N, never beyond.
    hard = m.force_n(np.full(am.N_ACT, 500.0 * am.PA_PER_PSI),
                     np.full(am.N_ACT, 0.05))
    np.testing.assert_array_equal(hard, np.full(am.N_ACT, -am.FORCE_CLIP_N))

    # The whole legal envelope stays inside the clip on both sides.
    rng = np.random.default_rng(19)
    for _ in range(200):
        p = rng.uniform(0.0, 30.0 * am.PA_PER_PSI, am.N_ACT)
        f = m.force_n(p, rng.uniform(-0.03, 0.03, am.N_ACT))
        assert np.all(f <= 0.0) and np.all(f >= -am.FORCE_CLIP_N)


def test_the_seeded_force_law_stays_clear_of_the_clip_at_the_operator_ceiling():
    """A seed that clips at 30 psi would report the guard rail, not the model.

    30 psi is the operator's per-line cap (``CONTRACT.md`` section 8), and the
    excursion is +/-25 mm, the bent-arm tendon travel the reference measured.
    This does not show the seeded ``coeff`` is right -- it is the reference
    arm's, and the fit is expected to move it -- only that it leaves the outer
    fit somewhere to start from.
    """
    m = _model()
    p = np.full(am.N_ACT, 30.0 * am.PA_PER_PSI)
    worst = min(float(m.force_n(p, np.full(am.N_ACT, d)).min())
                for d in (-0.025, 0.0, 0.025))
    assert worst > -am.FORCE_CLIP_N, f"the seed clips at 30 psi: {worst:.1f} N"
    assert worst < -100.0, f"the seed barely pulls at 30 psi: {worst:.1f} N"


def test_tendon_damping_schedule_and_slack_gate():
    m = _model()
    base = 1.0
    l = m.l0_per_act
    p = np.full(am.N_ACT, 10.0 * am.PA_PER_PSI)
    np.testing.assert_allclose(
        m.tendon_damping_n_s_m(p, base, l), base + m.damp_b1_per_act * p, rtol=1e-12)

    # Slack: the pressure term is gated off, the MJCF's own base damping remains.
    l_slack = (m.bf2_per_act ** 0.5) / np.sqrt(3.0) - 1e-3
    np.testing.assert_allclose(
        m.tendon_damping_n_s_m(p, base, l_slack), np.full(am.N_ACT, base), rtol=1e-12)

    # The gate is exactly the force law's boundary, so a muscle can never be
    # simultaneously slack for force and pressurised for damping.
    assert np.all(m.force_n(p, l_slack - m.l0_per_act) == 0.0)

    # Never negative, even if a caller hands over a negative base.
    assert np.all(m.tendon_damping_n_s_m(np.zeros(am.N_ACT), -3.0, l) == 0.0)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def _fitted_looking(seed=31):
    """A model whose every array differs from the defaults, so a round trip bites."""
    rng = np.random.default_rng(seed)
    m = _model(seed=seed)
    for k in am.FlowNet.PARAM_KEYS:
        arr = getattr(m.net, k)
        arr += rng.normal(0.0, 0.1, arr.shape)
    m.fill_gain[:] = rng.uniform(0.1, 4.0, am.N_ACT)
    m.vent_gain[:] = rng.uniform(0.1, 4.0, am.N_ACT)
    m.leak_pa_s[:] = rng.uniform(0.0, 800.0, am.N_ACT)
    m.coeff[:] = rng.uniform(0.05, 0.3, am.N_SEG)
    m.bf[:] = rng.uniform(0.15, 0.25, am.N_SEG)
    m.l0[:] = rng.uniform(0.13, 0.19, am.N_SEG)
    m.damp_b1[:] = rng.uniform(5e-4, 2e-3, am.N_SEG)
    m.blend_width_pa[:] = (1500.0, 7000.0)
    return m


def test_checkpoint_round_trip_is_bit_exact(tmp_path):
    m = _fitted_looking()
    path = tmp_path / "ckpt.npz"
    m.save(path, meta={"kind": "unit-test", "campaign": "none"})
    back = am.ActuatorModel.load(path)

    for k in am.FlowNet.PARAM_KEYS:
        np.testing.assert_array_equal(getattr(back.net, k), getattr(m.net, k))
    for k in ("fill_gain", "vent_gain", "leak_pa_s", "blend_width_pa",
              "coeff", "bf", "l0", "damp_b1"):
        np.testing.assert_array_equal(getattr(back, k), getattr(m, k))
    np.testing.assert_array_equal(back.is_tle, m.is_tle)
    assert back.meta["kind"] == "unit-test"
    assert back.meta["p_scale_pa"] == am.P_SCALE_PA

    rng = np.random.default_rng(0)
    p, tgt, l, ldot = _rack(rng)
    np.testing.assert_array_equal(
        back.net_flow_pa_s_batch(p, tgt, l, ldot),
        m.net_flow_pa_s_batch(p, tgt, l, ldot))


def test_save_normalises_the_extension_so_the_round_trip_finds_the_file(tmp_path):
    m = _model()
    m.save(tmp_path / "bare")
    assert (tmp_path / "bare.npz").exists()
    am.ActuatorModel.load(tmp_path / "bare.npz")


@pytest.mark.parametrize("key,bad", [
    ("n_in", 6),
    ("hidden", [32, 32]),
    ("p_scale_pa", 25.0 * am.PA_PER_PSI),      # the reference's 25 psi sysid cap
    ("e_scale_pa", 1.0e5),
    ("dp_scale_pa_s", 2.75e5),
    ("l_scale_m", 0.2),
    ("ldot_scale_m_s", 1.0),
])
def test_the_unit_guard_raises_on_a_doctored_meta(tmp_path, key, bad):
    """Every guarded scale, one at a time, doctored in the npz and reloaded.

    The 25 psi entry is the realistic case: it is the RS485 checkpoint's own
    pressure scale, so a checkpoint copied across from that arm fails here
    rather than eight thousand quanta into a rollout.
    """
    m = _model()
    src = tmp_path / "good.npz"
    m.save(src)

    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta")))
    meta[key] = bad
    dst = tmp_path / "doctored.npz"
    np.savez(dst, meta=json.dumps(meta), **arrays)

    with pytest.raises(ValueError, match=key):
        am.ActuatorModel.load(dst)


def test_a_checkpoint_with_no_normalisation_recorded_is_refused(tmp_path):
    m = _model()
    src = tmp_path / "good.npz"
    m.save(src)
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.pop("meta")
    dst = tmp_path / "nometa.npz"
    np.savez(dst, meta=json.dumps({"kind": "written by something older"}), **arrays)
    with pytest.raises(ValueError, match="n_in"):
        am.ActuatorModel.load(dst)


def test_a_checkpoint_without_damp_b1_inherits_the_reference_and_not_zero(tmp_path):
    """The RS485 shipped checkpoint pre-dates its own damping work.

    Zero here would silently restore ring-killing behaviour, and a twin that
    cannot oscillate reports a clean residual on a step and the wrong dynamics
    on everything else.
    """
    m = _model()
    src = tmp_path / "good.npz"
    m.save(src)
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "damp_b1"}
    dst = tmp_path / "predamping.npz"
    np.savez(dst, **arrays)

    back = am.ActuatorModel.load(dst)
    np.testing.assert_allclose(back.damp_b1, am.DAMP_B1_N_S_M_PER_PA, rtol=0, atol=0)
    assert np.all(back.damp_b1 > 0.0)


def test_defaults_are_not_shared_between_models():
    """Two fresh models must not alias one default tuple's array."""
    a, b = _model(), _model()
    a.coeff[0] += 1.0
    a.fill_gain[3] = 7.0
    assert b.coeff[0] != a.coeff[0]
    assert b.fill_gain[3] == 1.0


def test_clip_parameters_bounds_gains_and_refuses_a_negative_leak():
    m = _model()
    m.fill_gain[:3] = (-2.0, 0.0, 1e3)
    m.vent_gain[:2] = (-1.0, 500.0)
    m.leak_pa_s[:2] = (-400.0, 9e3)
    m.clip_parameters()
    np.testing.assert_allclose(m.fill_gain[:3], (0.05, 0.05, 20.0))
    np.testing.assert_allclose(m.vent_gain[:2], (0.05, 20.0))
    np.testing.assert_allclose(m.leak_pa_s[:2], (0.0, 2000.0))


# ---------------------------------------------------------------------------
# Leak fitting
# ---------------------------------------------------------------------------

def _closed_decay(rng, *, leak_pa_s, duration_s, p0_psi, rate_hz=150.0,
                  noise_pa=520.0):
    """One guaranteed-closed hold: a straight decay plus reply noise.

    520 Pa is the reference's consecutive-delta noise scale carried over -- 1.7
    to 2.7 ADC counts per sample times the nominal 7 mm scale, about 370 Pa, and
    the wire LSB on top.  It is an assumption about this arm's sensors until the
    campaign measures one.
    """
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    p = p0_psi * am.PA_PER_PSI - leak_pa_s * t
    return t, p + rng.normal(0.0, noise_pa, t.size)


def test_leak_fit_recovers_a_known_slope():
    rng = np.random.default_rng(2026)
    truth = 300.0                                   # Pa/s, 0.0435 psi/s
    t, p = _closed_decay(rng, leak_pa_s=truth, duration_s=6.0, p0_psi=20.0)
    got = am.fit_leak(t, p)
    assert abs(got - truth) / truth < 0.05, f"fitted {got:.1f} Pa/s against {truth}"

    # Noise-free is exact to the least-squares solve, which separates the
    # estimator from the noise model above.
    t = np.arange(0.0, 6.0, 1 / 150)
    np.testing.assert_allclose(
        am.fit_leak(t, 20.0 * am.PA_PER_PSI - truth * t), truth, rtol=1e-9)

    # Sign convention: positive means losing pressure.
    assert am.fit_leak(t, 20.0 * am.PA_PER_PSI + truth * t) == pytest.approx(-truth)


def test_leak_segments_are_duration_weighted():
    rng = np.random.default_rng(5)
    segs = [_closed_decay(rng, leak_pa_s=300.0, duration_s=d, p0_psi=psi)
            for d, psi in ((6.0, 20.0), (4.0, 12.0), (3.0, 6.0))]
    got = am.fit_leak_segments(segs)
    assert abs(got - 300.0) / 300.0 < 0.10, f"fitted {got:.1f} Pa/s against 300"

    # Two exact segments with different slopes, weighted 6 s against 2 s: the
    # duration-weighted mean is 0.75*100 + 0.25*500 = 200 Pa/s.  The two are
    # sampled at 150 Hz and 20 Hz respectively, so a sample-count weighting
    # would give 121 Pa/s instead and this test separates the two rules.
    t_long = np.linspace(0.0, 6.0, 901)
    t_short = np.linspace(0.0, 2.0, 41)
    exact = [(t_long, 1e5 - 100.0 * t_long), (t_short, 1e5 - 500.0 * t_short)]
    assert am.fit_leak_segments(exact) == pytest.approx(200.0, rel=1e-6)


def test_leak_fit_refuses_what_it_cannot_estimate():
    t = np.arange(0.0, 0.45, 1 / 150)
    with pytest.raises(ValueError, match="too short"):
        am.fit_leak(t, 1e5 - 300.0 * t)
    with pytest.raises(ValueError, match="too short"):
        am.fit_leak([0.0, 1.0], [1e5, 1e5])
    with pytest.raises(ValueError, match="matching 1-D"):
        am.fit_leak(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError, match="segment"):
        am.fit_leak_segments([(t, 1e5 - 300.0 * t)])


def test_a_negative_fitted_leak_is_clipped_to_zero():
    """The seam subtracts this number; a negative one inflates an idle board."""
    t = np.arange(0.0, 6.0, 1 / 150)
    assert am.fit_leak_segments([(t, 1e5 + 200.0 * t)]) == 0.0
    assert am.fit_leak_segments([(t, 1e5 - 5000.0 * t)]) == am.LEAK_CLIP_PA_S[1]
