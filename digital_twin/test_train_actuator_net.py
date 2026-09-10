"""What the trainer must not get wrong.

The first test in this file is the one the whole arrangement rests on: the numpy
forward in ``actuator_model`` and the torch forward here must be the same
function on the same weights.  If they are not, the trained model and the stepped
model are different models, every rollout disagrees with the checkpoint it
loaded, and nothing anywhere else in the twin would notice.

Everything here runs on the CPU by default and touches no hardware.  The one GPU
test is skipped when there is no CUDA device.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin import actuator_model as am  # noqa: E402
from digital_twin import dataset as ds  # noqa: E402
from digital_twin import train_actuator_net as tr  # noqa: E402

IS_TLE = np.array([True] * 8 + [False] * 16)


def _model(seed=1234, *, randomise=True):
    """A model with a non-degenerate net and unequal per-node gains.

    ``fresh`` deliberately ships a small output layer so an untrained net moves
    nothing; that is right for a rollout and useless for an equality test, which
    would then be comparing two ways of computing approximately zero.  So the
    weights are re-drawn at full scale here, and the gains are made unequal so a
    path that dropped the per-node indexing would show up.
    """
    m = am.ActuatorModel.fresh(is_tle=IS_TLE, seed=seed)
    if randomise:
        rng = np.random.default_rng(seed)
        for k, a in m.net.params().items():
            a[...] = rng.normal(0.0, 0.7, size=a.shape)
        m.fill_gain[...] = rng.uniform(0.3, 3.0, am.N_ACT)
        m.vent_gain[...] = rng.uniform(0.3, 3.0, am.N_ACT)
        m.leak_pa_s[...] = rng.uniform(0.0, 800.0, am.N_ACT)
    return m


# ---------------------------------------------------------------------------
# THE critical test
# ---------------------------------------------------------------------------
def test_numpy_and_torch_forwards_are_the_same_function():
    """``ActuatorModel.net_flow_pa_s_batch`` == :func:`train.net_flow_pa_s_batch`.

    Over the whole feature construction -- normalisation, the population bit, the
    logistic gain blend and the MLP -- not merely over the MLP, because three of
    the four are places a port silently disagrees: a swapped feature column, the
    population read from the id range, or the blend width taken from the wrong
    population all produce a net that trains fine and steps wrong.

    The tolerance is 1e-13 relative rather than exact equality because BLAS
    chooses a different summation order for a ``(24,5)`` GEMM than torch does, so
    the two agree to a few ulp and not bitwise -- which is the same caveat
    ``net_flow_pa_s_batch``'s own docstring gives for its scalar twin.
    """
    m = _model()
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(99)

    for _ in range(20):
        # spread the operating point across the whole legal envelope, including
        # zero pressure, the 30 psi cap, and errors of both signs at both
        # populations' blend widths
        p = rng.uniform(0.0, 30.0, am.N_ACT) * am.PA_PER_PSI
        tgt = rng.uniform(0.0, 30.0, am.N_ACT) * am.PA_PER_PSI
        l = rng.uniform(0.10, 0.22, am.N_ACT)
        ldot = rng.uniform(-0.4, 0.4, am.N_ACT)

        want = m.net_flow_pa_s_batch(p, tgt, l, ldot)
        got = tr.net_flow_pa_s_batch(
            tm, torch.arange(am.N_ACT),
            torch.tensor(p), torch.tensor(tgt),
            torch.tensor(l), torch.tensor(ldot)).detach().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-9)

    # and the scalar path, board by board, so the per-node gain indexing is
    # checked against an implementation that takes one node at a time
    p = rng.uniform(0.0, 30.0, am.N_ACT) * am.PA_PER_PSI
    tgt = rng.uniform(0.0, 30.0, am.N_ACT) * am.PA_PER_PSI
    l = rng.uniform(0.10, 0.22, am.N_ACT)
    ldot = rng.uniform(-0.4, 0.4, am.N_ACT)
    got = tr.net_flow_pa_s_batch(tm, torch.arange(am.N_ACT), torch.tensor(p),
                                 torch.tensor(tgt), torch.tensor(l),
                                 torch.tensor(ldot)).detach().numpy()
    for j in range(am.N_ACT):
        want = m.net_flow_pa_s(j + 1, p[j], tgt[j], l[j], ldot[j])
        assert got[j] == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_the_two_forwards_agree_in_the_saturated_tails():
    """A 30 psi error against a 2000 Pa blend width is a logistic argument of 100.

    Both implementations split the logistic on the sign of its argument so that
    ``exp`` only ever sees a non-positive number.  If one of them did not, this is
    where it would overflow, warn, and return a NaN that the mean-squared loss
    would carry silently into every parameter.
    """
    m = _model()
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    p = np.concatenate([np.zeros(12), np.full(12, 30.0 * am.PA_PER_PSI)])
    tgt = np.concatenate([np.full(12, 30.0 * am.PA_PER_PSI), np.zeros(12)])
    l = np.full(am.N_ACT, 0.15)
    ldot = np.zeros(am.N_ACT)

    want = m.net_flow_pa_s_batch(p, tgt, l, ldot)
    got = tr.net_flow_pa_s_batch(tm, torch.arange(am.N_ACT), torch.tensor(p),
                                 torch.tensor(tgt), torch.tensor(l),
                                 torch.tensor(ldot)).detach().numpy()
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-9)


def test_autograd_reproduces_the_modules_own_state_jacobian():
    """``FlowNet.forward_full``'s ``dy_dp`` against autograd on the same point.

    Worth its own test because that derivative is the one the reference got to
    take for free and this arm does not: ``p`` enters the feature row as column 0
    and again, negated, inside the commanded error of column 1, so the total
    derivative is ``dy/dx0 / P_SCALE_PA - dy/dx1 / E_SCALE_PA``.  A hand-written
    adjoint carrying only the first term drops a comparable, opposite-signed
    contribution and still trains, so autograd checking it is the only thing that
    would catch the mistake.
    """
    m = _model()
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(5)
    p = rng.uniform(2000.0, 1.8e5, am.N_ACT)
    tgt = rng.uniform(0.0, 2.0e5, am.N_ACT)
    l = rng.uniform(0.10, 0.22, am.N_ACT)
    ldot = rng.uniform(-0.4, 0.4, am.N_ACT)

    x = m._norm_x(p, tgt, m.is_tle.astype(float), l, ldot)
    _, _, _, want = m.net.forward_full(x)

    pt = torch.tensor(p, requires_grad=True)
    e = torch.tensor(tgt) - pt
    y = tr.net_forward(tm.net, tr.norm_x(pt, e, tm.is_tle, torch.tensor(l),
                                         torch.tensor(ldot)))
    got = torch.autograd.grad(y.sum(), pt)[0].numpy()
    np.testing.assert_allclose(got, want, rtol=1e-11)


# ---------------------------------------------------------------------------
# The shooting objective
# ---------------------------------------------------------------------------
def _small_batch(seed=3, b=6, k=9, device="cpu"):
    """A hand-built window set small enough to finite-difference."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.full((b, k), 1 / 150.0), axis=1)
    win = ds.WindowSet(
        node_idx=rng.integers(0, am.N_ACT, b).astype(np.int32),
        episode=np.zeros(b, dtype=np.int32),
        is_tle=IS_TLE[rng.integers(0, am.N_ACT, b)],
        t=t,
        p_rec=rng.uniform(1.0e4, 1.2e5, (b, k)),
        target_pa=rng.uniform(0.0, 1.5e5, (b, k)),
        l=rng.uniform(0.12, 0.20, (b, k)),
        ldot=rng.uniform(-0.2, 0.2, (b, k)),
        zero_counts=np.full(b, ds.LEGACY_ZERO_COUNTS),
        counts_per_psi=np.full(b, ds.LEGACY_COUNTS_PER_PSI),
    )
    win.is_tle = IS_TLE[win.node_idx]
    return win


def test_shooting_loss_is_the_normalised_pressure_residual():
    """The objective, checked against its own definition rather than trusted."""
    m = _model()
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    batch = tr.make_batch(_small_batch(), device="cpu")
    loss, traj = tr.shoot(tm, batch, return_traj=True)

    traj = traj.detach()
    resid = (traj[:, 1:] - batch.p_rec[:, 1:]) / am.P_SCALE_PA
    assert float(loss.detach()) == pytest.approx(float((resid ** 2).mean()), rel=1e-14)
    # tick 0 is the anchor: every window starts exactly on its own recorded p0
    np.testing.assert_allclose(traj[:, 0].numpy(), batch.p_rec[:, 0].numpy())
    assert float(loss.detach()) > 0.0


def test_windows_are_re_anchored_at_their_own_p0():
    """Multiple shooting, not one long rollout.

    Changing the anchor of one window must change that window's trajectory and no
    other's; a loop that carried state between rows would fail this.
    """
    m = _model()
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    win = _small_batch()
    _, traj_a = tr.shoot(tm, tr.make_batch(win, device="cpu"), return_traj=True)

    win2 = ds.WindowSet(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in win.__dict__.items()})
    win2.p_rec[2, 0] += 1.0e4
    _, traj_b = tr.shoot(tm, tr.make_batch(win2, device="cpu"), return_traj=True)

    d = (traj_b - traj_a).detach().abs().numpy()
    assert d[2].max() > 1.0, "the perturbed window must move"
    assert np.allclose(np.delete(d, 2, axis=0), 0.0), \
        "no other window may move: each is its own shooting problem"


def test_shooting_gradient_matches_finite_differences():
    """Autograd's adjoint against central differences on the full rollout.

    Run with ``adc_in_loop=False``.  With the simulated sensor in the loop the
    commanded error is piecewise constant in the pressure, so a difference
    quotient that happens to step across a count boundary measures the 113 Pa
    count and not the derivative; that treatment is deliberate and matches the
    reference's held-fixed quantisation, but it is not finite-differenceable.
    The recursion, the clamp-free interior, the gain blend and the net are all
    still under test here.
    """
    m = _model(seed=77)
    m.leak_pa_s[...] = 0.0            # keep p clear of the p >= 0 clamp's kink
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    batch = tr.make_batch(_small_batch(seed=8, b=4, k=6), device="cpu")

    loss, _ = tr.shoot(tm, batch, adc_in_loop=False)
    loss.backward()

    checked = 0
    for name, t in list(tm.net.items()) + [("fill_gain", tm.fill_gain),
                                           ("vent_gain", tm.vent_gain)]:
        flat = t.detach().reshape(-1)
        grad = t.grad.detach().reshape(-1)
        # The four entries carrying the most gradient, not four random ones.  A
        # central difference on a loss of order 1e-1 with a 1e-6 step has an
        # absolute floor near 2e-11 from cancellation alone, so an entry whose
        # true gradient is 1e-8 is being compared against noise: a random draw
        # would then fail on a correct adjoint about as often as on a wrong one.
        top = torch.topk(grad.abs(), k=min(4, grad.numel())).indices
        for i in top.tolist():
            orig = float(flat[i])
            h = 1e-6 * max(1.0, abs(orig))
            with torch.no_grad():
                flat[i] = orig + h
            lo_p, _ = tr.shoot(tm, batch, adc_in_loop=False)
            with torch.no_grad():
                flat[i] = orig - h
            lo_m, _ = tr.shoot(tm, batch, adc_in_loop=False)
            with torch.no_grad():
                flat[i] = orig
            fd = (float(lo_p.detach()) - float(lo_m.detach())) / (2 * h)
            ana = float(grad[i])
            denom = max(abs(fd), abs(ana), 1e-30)
            assert abs(fd - ana) / denom < 1e-5, (
                f"{name}[{int(i)}]: autograd {ana:.6e} vs finite difference "
                f"{fd:.6e}")
            checked += 1
    assert checked >= 20


def test_gradient_accumulation_equals_one_full_batch():
    """``chunk`` buys VRAM and must not change the optimisation problem."""
    m = _model(seed=21)
    win = _small_batch(seed=4, b=10, k=7)

    grads = []
    for chunk in (10, 3):
        tm = tr.to_torch(_model(seed=21), device="cpu", dtype=torch.float64)
        batch = tr.make_batch(win, device="cpu")
        for t in tm.net_params() + tm.gain_params():
            t.grad = None
        loss = tr._accumulate(tm, batch, chunk)
        grads.append((loss, [t.grad.clone() for t in tm.net_params()]))

    assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-12)
    for a, b in zip(grads[0][1], grads[1][1]):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-11, atol=1e-18)
    assert m is not None


def test_pressure_never_goes_negative_in_a_rollout():
    """The ``p >= 0`` clamp after every substep, one of the four physical
    constraints the model actually enforces."""
    m = _model(seed=13)
    m.leak_pa_s[...] = 2000.0          # the clip's maximum, to drive p down hard
    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    win = _small_batch(seed=6, b=8, k=40)
    win.target_pa[...] = 0.0
    _, traj = tr.shoot(tm, tr.make_batch(win, device="cpu"), return_traj=True)
    assert float(traj.detach().min()) >= 0.0


# ---------------------------------------------------------------------------
# End to end on a synthetic session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    return ds.write_synthetic_session(str(d / "session_synth"), n_episodes=5,
                                      episode_s=8.0, seed=31)


def test_training_recovers_the_synthetic_plant(synth, tmp_path_factory):
    """The end-to-end claim: fit a plant we generated and score on unseen episodes.

    Three assertions, each about a different failure:

    * the shooting loss must fall -- otherwise the adjoint or the optimiser is
      wrong;
    * the held-out rollout RMS must beat the unfitted baseline by a wide margin
      and land inside a stated absolute bound, on **episodes the fit never
      saw** -- otherwise the improvement is memorisation;
    * the fitted leak must be close to the injected one, because the leak is
      fitted outside the net and a net that had absorbed it would still show a
      falling loss.

    WHAT IT DOES NOT SHOW.  It does not show that the twin matches the arm.  The
    synthetic plant is a dead-banded orifice model with 1.2 counts of sensor
    noise and no supply dynamics, hysteresis, temperature or cross-talk, and the
    tendon geometry underneath it is the moment-arm placeholder.  It also says
    nothing about the absolute gain scale, which is degenerate with the net's
    output scale by construction.
    """
    out = tmp_path_factory.mktemp("ckpt") / "flow.npz"
    metrics = tr.run(synth, str(out), epochs=60, warm_epochs=120,
                     device="cpu", holdout_frac=0.4, window_cycles=200,
                     chunk=400, seed=31,
                     tendon=ds.MomentArmTendonGeometry(), log=lambda *a: None)

    first, last = metrics["shoot_loss_first_last"]
    assert last < first, f"shooting loss did not fall: {first:.3e} -> {last:.3e}"

    base = metrics["baseline_holdout_rms_pa"]
    final = metrics["holdout_rms_pa"]
    assert final < 0.30 * base, (
        f"held-out RMS {final:.0f} Pa against an unfitted baseline of "
        f"{base:.0f} Pa is not a fit")
    # 6 kPa is a regression guard at a deliberately short epoch budget, not the
    # accuracy this objective reaches.  Measured on this generator, seed 31, five
    # 8 s episodes: 60 CPU epochs land near 4.3 kPa and 250 epochs on the 5090
    # near 2.10 kPa (0.304 psi) against a 21.3 kPa unfitted baseline, with the
    # training RMS at 1.94 kPa -- so the gap is the budget and not overfitting.
    assert final < 6000.0, f"held-out RMS {final:.0f} Pa = {final / 6894.757:.2f} psi"
    assert final < 3.0 * metrics["train_rms_pa"], (
        f"held-out {final:.0f} Pa against training {metrics['train_rms_pa']:.0f} Pa "
        "is a memorisation gap, not a fit")

    assert set(metrics["train_episodes"]).isdisjoint(metrics["holdout_episodes"])
    assert metrics["holdout_episodes"], "there must be a holdout to report"

    rec = ds.load_session(synth)
    truth = np.asarray(rec.meta["synthetic_truth"]["leak_pa_s"])
    model = am.ActuatorModel.load(str(out))
    fitted = metrics["leak_fitted_boards"]
    rel = np.abs(model.leak_pa_s[fitted] - truth[fitted]) / truth[fitted]
    assert np.median(rel) < 0.15


def test_the_checkpoint_the_trainer_writes_is_one_the_model_loads(synth, tmp_path):
    """The trainer's only output is a checkpoint ``ActuatorModel.load`` accepts.

    Loaded and then *stepped*: a file that parses but whose weights do not
    reproduce the trainer's own forward would pass a round-trip check and fail
    the only thing the checkpoint is for.
    """
    out = tmp_path / "ck.npz"
    tr.run(synth, str(out), epochs=3, warm_epochs=3, device="cpu",
           holdout_frac=0.4, window_cycles=150, seed=31,
           tendon=ds.MomentArmTendonGeometry(), log=lambda *a: None)

    m = am.ActuatorModel.load(str(out))
    assert m.meta["p_scale_pa"] == am.P_SCALE_PA
    assert m.meta["n_in"] == am.N_IN
    assert m.meta["tendon_geometry"] == "moment-arm placeholder"
    assert m.meta["holdout_episodes"] and m.meta["train_episodes"]
    assert np.all(m.leak_pa_s >= 0.0) and np.all(m.leak_pa_s <= 2000.0)
    assert np.all(m.fill_gain >= am.GAIN_CLIP[0]) and np.all(m.fill_gain <= am.GAIN_CLIP[1])

    # the population mask must have survived the round trip from the variant byte
    rec = ds.load_session(synth)
    np.testing.assert_array_equal(m.is_tle,
                                  rec.board_type == ds.VARIANT_TLE_DVP)

    tm = tr.to_torch(m, device="cpu", dtype=torch.float64)
    p = np.full(am.N_ACT, 5.0 * am.PA_PER_PSI)
    tgt = np.full(am.N_ACT, 10.0 * am.PA_PER_PSI)
    l = np.full(am.N_ACT, 0.15)
    ldot = np.zeros(am.N_ACT)
    np.testing.assert_allclose(
        tr.net_flow_pa_s_batch(tm, torch.arange(am.N_ACT), torch.tensor(p),
                               torch.tensor(tgt), torch.tensor(l),
                               torch.tensor(ldot)).detach().numpy(),
        m.net_flow_pa_s_batch(p, tgt, l, ldot), rtol=1e-13, atol=1e-9)


def test_warm_start_filter_rejects_physically_impossible_intervals():
    """A board being asked to fill cannot lose pressure; venting cannot gain it."""
    b, k = 3, 4
    t = np.cumsum(np.full((b, k), 1 / 150.0), axis=1)
    p = np.zeros((b, k))
    p[0] = [5.0e4, 5.2e4, 5.4e4, 5.6e4]          # filling and rising: keep
    p[1] = [5.0e4, 3.0e4, 1.0e4, 0.5e4]          # asked to fill, falling: drop
    p[2] = [5.0e4, 5.0e4, 5.0e4, 5.0e4]          # holding, flat: keep
    tgt = np.zeros((b, k))
    tgt[0] = 1.5e5
    tgt[1] = 1.5e5
    tgt[2] = 5.0e4
    win = ds.WindowSet(node_idx=np.array([0, 1, 2], dtype=np.int32),
                       episode=np.zeros(b, dtype=np.int32),
                       is_tle=IS_TLE[:3], t=t, p_rec=p, target_pa=tgt,
                       l=np.full((b, k), 0.15), ldot=np.zeros((b, k)),
                       zero_counts=np.full(b, ds.LEGACY_ZERO_COUNTS),
                       counts_per_psi=np.full(b, ds.LEGACY_COUNTS_PER_PSI))
    node, *_ = tr.build_warmstart_dataset(win, np.zeros(am.N_ACT))
    kept = set(int(n) for n in node)
    assert 0 in kept and 2 in kept
    assert 1 not in kept, "a muscle asked to fill cannot lose 4.5 kPa"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_and_cpu_agree_on_the_loss(synth):
    """The 5090 is where a real fit runs; the CPU is where the tests run.

    Both must compute the same objective.  The tolerance is loose because CUDA
    and MKL reduce in different orders, but the two must agree far better than
    the loss changes over one epoch, or a GPU fit is optimising a different
    number from the one the tests check.
    """
    rec = ds.load_session(synth)
    l, ldot = ds.muscle_traces(rec, tendon=ds.MomentArmTendonGeometry())
    win = ds.make_windows(rec, l, ldot, window_cycles=150).select(slice(0, 256))
    m = _model(seed=44)
    a, _ = tr.shoot(tr.to_torch(m, device="cpu"), tr.make_batch(win, device="cpu"))
    b, _ = tr.shoot(tr.to_torch(m, device="cuda"), tr.make_batch(win, device="cuda"))
    assert float(a.detach()) == pytest.approx(float(b.detach()), rel=1e-10)
