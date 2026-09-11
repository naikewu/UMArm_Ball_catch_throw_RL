"""Torque allocation at unilateral and capped pressure limits."""
import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from control.controller import (InverseDynamics, FeedforwardPIDController,
                                PA_PER_PSI, pair_indices,
                                differential_to_pressure, assert_pressures)
from control.trajectory import SoftTrajectory


def _legacy_from_matrix(B, tau):
    pairs = pair_indices()
    D = np.zeros((24, 12))
    D[pairs[:, 0], np.arange(12)] = .5
    D[pairs[:, 1], np.arange(12)] = -.5
    diff = np.linalg.lstsq(B @ D, tau - B @ np.full(24, 6.), rcond=1e-6)[0]
    p = differential_to_pressure(diff * PA_PER_PSI)
    return p, diff


def _matrix(dynamics):
    m, d = dynamics.m, dynamics.d
    moment = np.zeros((24, m.nv))
    for i, a in enumerate(dynamics.acts):
        adr, n = d.moment_rowadr[a], d.moment_rownnz[a]
        moment[i, d.moment_colind[adr:adr+n]] = d.actuator_moment[adr:adr+n]
    dl = d.ten_length[dynamics.tens] - dynamics.arm._ten_len0
    B = moment[:, dynamics.dofs].T * dynamics.arm.actuator.force_n(np.full(24, PA_PER_PSI), dl)
    return B, d.qfrc_inverse[dynamics.dofs].copy()


def test_unilateral_limit_reallocates_missing_torque():
    B = np.zeros((12, 24))
    pairs = pair_indices()
    B[np.arange(12), pairs[:, 0]] = .25
    B[np.arange(12), pairs[:, 1]] = -.25
    tau = np.linspace(-6., 6., 12)
    old, diff = _legacy_from_matrix(B, tau)
    new, residual = InverseDynamics._bounded_pressures(B, tau, diff, old)
    assert_pressures(new)
    assert np.linalg.norm(B @ (old / PA_PER_PSI) - tau) > 2.
    assert residual < 1e-12
    assert np.allclose(B @ (new / PA_PER_PSI), tau, atol=1e-12)
    inside = abs(diff) <= 12
    assert np.allclose(new[pairs[inside]], old[pairs[inside]], rtol=0, atol=1e-9)


def test_capped_coupled_allocation_is_safe_and_never_worse():
    rng = np.random.default_rng(99)
    pairs = pair_indices()
    for _ in range(40):
        B = rng.normal(0., .002, (12, 24))
        B[np.arange(12), pairs[:, 0]] += rng.uniform(.1, .3, 12)
        B[np.arange(12), pairs[:, 1]] -= rng.uniform(.1, .3, 12)
        tau = rng.normal(0., 7., 12)
        old, diff = _legacy_from_matrix(B, tau)
        new, residual = InverseDynamics._bounded_pressures(B, tau, diff, old)
        assert_pressures(new)
        assert residual <= np.linalg.norm(B @ (old / PA_PER_PSI) - tau) + 1e-12
        assert residual == pytest.approx(np.linalg.norm(B @ (new / PA_PER_PSI) - tau))


@pytest.mark.parametrize("speed,t", [("slow", 6.08), ("slow", 18.46), ("fast", 11.48)])
def test_full_reference_preview_residual_and_bounded_optimum(speed, t):
    ref = SoftTrajectory(speed)
    q, qd, qdd = ref.sample(t)
    q = q + .070 * qd + .5 * .070**2 * qdd
    qd = qd + .070 * qdd
    legacy = InverseDynamics(allocation_mode="legacy_clip")
    bounded = InverseDynamics()
    pressures = np.full(24, 6 * PA_PER_PSI)
    old, old_res = legacy.allocate(q, qd, qdd, pressures)
    new, new_res = bounded.allocate(q, qd, qdd, pressures)
    B, tau = _matrix(legacy)
    reconstructed, _ = _legacy_from_matrix(B, tau)
    assert np.array_equal(old, reconstructed)
    assert_pressures(new)
    assert old_res > 1.
    assert new_res < old_res * .95
    pair_sum = np.zeros((12, 24))
    for j, (a, b) in enumerate(pair_indices()):
        pair_sum[j, a] = pair_sum[j, b] = 1.
    optimum = minimize(lambda p: .5 * np.sum((B @ p-tau)**2), new / PA_PER_PSI,
                       jac=lambda p: B.T @ (B @ p-tau), method="SLSQP",
                       bounds=Bounds(np.zeros(24), np.full(24, 30.)),
                       constraints=[LinearConstraint(pair_sum, np.zeros(12), np.full(12, 30.))],
                       options={"ftol": 1e-12, "maxiter": 200})
    assert optimum.success
    # The production solver retains a low common mode, so it need not find
    # the full 24-variable optimum. These actual coupled cases do agree.
    assert new_res <= np.linalg.norm(B @ optimum.x-tau) + 1e-3


def test_allocator_mode_is_explicit_and_validated():
    assert FeedforwardPIDController().dynamics.allocation_mode == "bounded"
    assert FeedforwardPIDController(allocation_mode="legacy_clip").dynamics.allocation_mode == "legacy_clip"
    with pytest.raises(ValueError, match="allocation_mode"):
        InverseDynamics(allocation_mode="misspelled")
