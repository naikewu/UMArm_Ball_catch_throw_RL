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
    def __init__(self, dt=1/150, kp=150., ki=30., kd=12., **_):
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

    def command(self, q, qdot, p_pa, q_ref, qd_ref, qdd_ref, future_q=None):
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
    def __init__(self, twin_kwargs=None):
        import mujoco
        from digital_twin.sim_core import SimArm
        from digital_twin.twin_params import load_twin_kwargs
        self.mj = mujoco
        self.arm = SimArm(**(load_twin_kwargs(log=lambda _: None) if twin_kwargs is None else twin_kwargs))
        self.m, self.d = self.arm.model, self.arm.data
        self.order = self.arm._q_qposadr
        self.dofs = np.array([self.m.jnt_dofadr[self.m.jnt_qposadr.tolist().index(int(i))] for i in self.order])
        self.acts = self.arm._act_ids
        self.tens = self.arm._ten_ids
        self.damping = self.m.tendon_damping[self.tens].copy()

    def allocate(self, q, qd, qdd, p_pa):
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
        moments = np.zeros((24,m.nv))
        for i,a in enumerate(self.acts):
            adr,n = d.moment_rowadr[a],d.moment_rownnz[a]
            moments[i,d.moment_colind[adr:adr+n]] = d.actuator_moment[adr:adr+n]
        B = moments[:,self.dofs].T * law.force_n(np.full(24, PA_PER_PSI), dl)
        pairs = pair_indices()
        D = np.zeros((24,12))
        D[pairs[:,0],np.arange(12)] = .5
        D[pairs[:,1],np.arange(12)] = -.5
        base = np.full(24, 6.)
        diff = np.linalg.lstsq(B@D, tau-B@base, rcond=1e-6)[0]
        p = differential_to_pressure(diff*PA_PER_PSI)
        return p, float(np.linalg.norm(B@(p/PA_PER_PSI)-tau))


class FeedforwardPIDController(PIDController):
    """Fitted inverse-dynamics pressure preview plus independent pressure PID."""
    def __init__(self, dt=1/150, twin_kwargs=None, kp=25., ki=8., kd=5., preview=.070, **kwargs):
        super().__init__(dt=dt, kp=kp, ki=ki, kd=kd, **kwargs)
        self.dynamics = InverseDynamics(twin_kwargs)
        self.preview = float(preview)

    def command(self, q, qdot, p_pa, q_ref, qd_ref, qdd_ref, future_q=None):
        started = time.perf_counter()
        # Pressure dynamics introduce lag; preview was selected on a separate
        # joint multisine, before scoring the Soft references.
        preview = self.preview
        qr = np.asarray(q_ref)+preview*np.asarray(qd_ref)+.5*preview**2*np.asarray(qdd_ref)
        qdr = np.asarray(qd_ref)+preview*np.asarray(qdd_ref)
        ff,residual = self.dynamics.allocate(qr,qdr,qdd_ref,p_pa)
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
        self.last_diagnostics={"solve_ms":1000*(time.perf_counter()-started),"allocation_residual_nm":residual}
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
