"""Pressure controllers for the fitted CAN arm; inputs are observed SI state.

The RS485 feedforward/Koopman implementations supplied the architecture, while
the model, measured antagonist map, 150 Hz sampling and pressure caps are CAN
specific.  No controller imports a transport or opens hardware.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np

from UMArm_KINEMATICS.canarm_actuators import joint_pairs

PA_PER_PSI = 6894.757
CAP_PA = 30 * PA_PER_PSI
DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "canarm_koopman.pt"


def pair_indices():
    return np.asarray(joint_pairs(), dtype=int) - 0x101


def project_pressures(p):
    """Euclidean projection onto nonnegative line and antagonist-sum caps."""
    p = np.asarray(p, dtype=float).copy()
    if p.shape[-1] != 24 or not np.isfinite(p).all():
        raise ValueError("pressure must have finite final dimension 24")
    p = np.maximum(p, 0)
    for a, b in pair_indices():
        excess = np.maximum(p[..., a] + p[..., b] - CAP_PA, 0)
        a0, b0 = p[..., a].copy(), p[..., b].copy()
        p[..., a] = np.clip(a0 - excess / 2, 0, CAP_PA)
        p[..., b] = np.clip(b0 - excess / 2, 0, CAP_PA)
    return p


def assert_pressures(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (24,) or not np.isfinite(p).all():
        raise ValueError("expected finite pressure Pa[24]")
    pairs = pair_indices()
    if np.min(p) < -1e-8 or np.max(p) > CAP_PA + 1e-8 or np.any(p[pairs].sum(axis=1) > CAP_PA + 1e-8):
        raise ValueError("30 psi individual / measured antagonist pair cap exceeded")


def differential_to_pressure(diff_pa, base_pa=6 * PA_PER_PSI):
    diff = np.asarray(diff_pa, dtype=float)
    p = np.zeros(diff.shape[:-1] + (24,))
    pairs = pair_indices()
    p[..., pairs[:, 0]] = np.asarray(base_pa) + diff / 2
    p[..., pairs[:, 1]] = np.asarray(base_pa) - diff / 2
    return project_pressures(p)


class PIDController:
    """Plain pressure PID: angle error only, with anti-windup and derivative state."""
    def __init__(self, dt=1/150, kp=85., ki=8., kd=12., **_):
        self.dt = float(dt)
        self.kp, self.ki, self.kd = (np.broadcast_to(v, (12,)).copy() * PA_PER_PSI for v in (kp, ki, kd))
        self.last_diagnostics = {}
        self.reset(np.zeros(12), np.zeros(24))

    def reset(self, q, p_pa):
        self.integral = np.zeros(12)
        self.last_p = project_pressures(p_pa)

    def _feedback(self, q, qdot, q_ref, qd_ref):
        e = np.asarray(q_ref) - q
        self.integral = np.clip(self.integral + self.dt * e, -.3, .3)
        return self.kp * e + self.ki * self.integral + self.kd * (np.asarray(qd_ref) - qdot)

    def command(self, q, qdot, p_pa, q_ref, qd_ref, qdd_ref, future_q=None,
                future_tip=None, future_compliance=None, external_torque_nm=None,
                future_qd=None, future_tip_velocity=None):
        del external_torque_nm
        started = time.perf_counter()
        d = self._feedback(q, qdot, q_ref, qd_ref)
        p = differential_to_pressure(d)
        pairs = pair_indices()
        actual = p[pairs[:,0]] - p[pairs[:,1]]
        # Back calculation only where saturation would wind the integrator up.
        self.integral += .1 * (actual - d) / np.maximum(self.ki, PA_PER_PSI)
        self.integral = np.clip(self.integral, -.3, .3)
        assert_pressures(p)
        self.last_p = p
        self.last_diagnostics = {"solve_ms": 1000*(time.perf_counter()-started), "saturated_fraction":float(np.mean(abs(actual-d)>1))}
        return p


class InverseDynamics:
    """Independent fitted-model inverse dynamics; never reads the stepped plant."""
    def __init__(self, twin_kwargs=None, allocation_mode="bounded"):
        import mujoco
        from digital_twin.sim_core import SimArm
        from digital_twin.twin_params import load_twin_kwargs
        if allocation_mode not in ("bounded", "legacy_clip"):
            raise ValueError("allocation_mode must be bounded or legacy_clip")
        self.allocation_mode = allocation_mode
        self.mj = mujoco
        self.arm = SimArm(**(load_twin_kwargs(log=lambda _: None) if twin_kwargs is None else twin_kwargs))
        self.m, self.d = self.arm.model, self.arm.data
        self.order = self.arm._q_qposadr
        self.dofs = np.array([self.m.jnt_dofadr[self.m.jnt_qposadr.tolist().index(int(i))] for i in self.order])
        self.acts = self.arm._act_ids
        self.tens = self.arm._ten_ids
        self.damping = self.m.tendon_damping[self.tens].copy()

    @staticmethod
    def _bounded_pressures(B, tau, diff, legacy_p):
        """Reallocate torque when a fixed-mean antagonist would be negative.

        The 6 psi pair mean is retained for differences within +/-12 psi.
        Beyond that, the opposite muscle vents and the active pressure is
        solved again. Each linear branch uses at most 12 variables, with a
        bounded least-squares solve only when the 30 psi cap is active.
        A legacy feasible candidate guarantees that allocation residual cannot
        increase, including for poorly conditioned or unreachable demands.
        """
        from scipy.optimize import lsq_linear
        pairs = pair_indices()
        best = np.asarray(legacy_p, dtype=float) / PA_PER_PSI
        best_error = float(np.linalg.norm(B @ best - tau))
        state = np.where(diff > 12., 1, np.where(diff < -12., -1, 0))
        if not np.any(state):
            return best * PA_PER_PSI, best_error
        seen = set()
        for _ in range(12):
            key = tuple(state)
            if key in seen:
                break
            seen.add(key)
            base = np.zeros(24)
            D = np.zeros((24, 12))
            for j, (a, b) in enumerate(pairs):
                if state[j] == 0:
                    base[a] = base[b] = 6.
                    D[a, j], D[b, j] = .5, -.5
                elif state[j] == 1:
                    D[a, j] = 1.
                else:
                    D[b, j] = -1.
            matrix, rhs = B @ D, tau - B @ base
            diff = np.linalg.lstsq(matrix, rhs, rcond=1e-6)[0]
            if np.any(np.abs(diff) > 30.):
                diff = lsq_linear(matrix, rhs, bounds=(-30., 30.),
                                  method="bvls", tol=1e-10, max_iter=50).x
            diff = np.clip(diff, -30., 30.)
            common = np.maximum(6., np.abs(diff) / 2)
            candidate = np.zeros(24)
            candidate[pairs[:, 0]] = common + diff / 2
            candidate[pairs[:, 1]] = common - diff / 2
            error = float(np.linalg.norm(B @ candidate - tau))
            if error < best_error:
                best, best_error = candidate, error
            new_state = np.where(diff > 12., 1, np.where(diff < -12., -1, 0))
            if np.array_equal(new_state, state):
                break
            state = new_state
        return best * PA_PER_PSI, best_error

    def allocate(self, q, qd, qdd, p_pa, external_torque_nm=None):
        m,d,mj = self.m,self.d,self.mj
        d.qpos[self.order] = q
        d.qvel[self.dofs] = qd
        d.ctrl[:] = 0
        mj.mj_fwdPosition(m,d)
        dl = d.ten_length[self.tens] - self.arm._ten_len0
        law = self.arm.actuator
        length = law.l0_per_act + dl
        m.tendon_damping[self.tens] = law.tendon_damping_n_s_m(p_pa, self.damping, length)
        mj.mj_forward(m,d)
        d.qacc[self.dofs] = qdd
        mj.mj_inverse(m,d)
        tau = d.qfrc_inverse[self.dofs].copy()
        if external_torque_nm is not None:
            external = np.asarray(external_torque_nm, dtype=float)
            if external.shape != (12,) or not np.isfinite(external).all():
                raise ValueError("external_torque_nm must be a finite 12-vector")
            tau -= external
        moments = np.zeros((24,m.nv))
        for i,a in enumerate(self.acts):
            adr,n = d.moment_rowadr[a],d.moment_rownnz[a]
            moments[i,d.moment_colind[adr:adr+n]] = d.actuator_moment[adr:adr+n]
        B = moments[:,self.dofs].T * law.force_n(np.full(24, PA_PER_PSI), dl)
        self.last_B = B
        self.last_tau = tau
        pairs = pair_indices()
        D = np.zeros((24,12))
        D[pairs[:,0],np.arange(12)] = .5
        D[pairs[:,1],np.arange(12)] = -.5
        base = np.full(24, 6.)
        diff = np.linalg.lstsq(B@D, tau-B@base, rcond=1e-6)[0]
        p = differential_to_pressure(diff*PA_PER_PSI)
        if self.allocation_mode == "bounded":
            return self._bounded_pressures(B, tau, diff, p)
        return p, float(np.linalg.norm(B@(p/PA_PER_PSI)-tau))


class FeedforwardPIDController(PIDController):
    """Fitted inverse-dynamics pressure preview plus independent pressure PID."""
    def __init__(self, dt=1/150, twin_kwargs=None, kp=25., ki=8., kd=5., preview=.070,
                 allocation_mode="bounded", **kwargs):
        super().__init__(dt=dt, kp=kp, ki=ki, kd=kd, **kwargs)
        self.dynamics = InverseDynamics(twin_kwargs, allocation_mode=allocation_mode)
        self.preview = float(preview)

    def command(self, q, qdot, p_pa, q_ref, qd_ref, qdd_ref, future_q=None,
                future_tip=None, future_compliance=None, external_torque_nm=None,
                future_qd=None, future_tip_velocity=None):
        started = time.perf_counter()
        # Pressure dynamics introduce lag; preview was selected on a separate
        # joint multisine, before scoring the Soft references.
        preview = self.preview
        qr = np.asarray(q_ref)+preview*np.asarray(qd_ref)+.5*preview**2*np.asarray(qdd_ref)
        qdr = np.asarray(qd_ref)+preview*np.asarray(qdd_ref)
        ff,residual = self.dynamics.allocate(
            qr, qdr, qdd_ref, p_pa, external_torque_nm=external_torque_nm
        )
        fb = self._feedback(q,qdot,q_ref,qd_ref)
        pairs = pair_indices()
        desired = ff.copy()
        desired[pairs[:,0]] += fb/2
        desired[pairs[:,1]] -= fb/2
        p = project_pressures(desired)
        actual = (p-desired)[pairs[:,0]]-(p-desired)[pairs[:,1]]
        self.integral = np.clip(self.integral+.1*actual/np.maximum(self.ki,PA_PER_PSI),-.3,.3)
        self.last_p=p
        assert_pressures(p)
        external_norm = 0.0 if external_torque_nm is None else float(np.linalg.norm(external_torque_nm))
        self.last_diagnostics={"solve_ms":1000*(time.perf_counter()-started),
                               "allocation_residual_nm":residual,
                               "external_torque_norm_nm":external_norm}
        return p


def make_controller(name, dt=1/150, checkpoint=None, twin_kwargs=None, seed=20260911, **kwargs):
    if name == "pid":
        return PIDController(dt=dt, **kwargs)
    if name == "ff_pid":
        return FeedforwardPIDController(dt=dt, twin_kwargs=twin_kwargs, **kwargs)
    if name == "koopman_mppi":
        from .koopman import KoopmanMPPIController
        return KoopmanMPPIController(dt=dt, checkpoint=checkpoint or DEFAULT_CHECKPOINT, twin_kwargs=twin_kwargs,seed=seed, **kwargs)
    raise ValueError(f"unknown controller {name!r}; choose pid, ff_pid, koopman_mppi")
