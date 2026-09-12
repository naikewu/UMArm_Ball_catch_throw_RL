"""Local, frozen-pressure compliance of the fitted MuJoCo mechanics.

K = -d(tau_passive + tau_muscle - tau_gravity)/dq, C = J K^-1 J.T.
This is a tangent at the supplied configuration, with a constant holding
torque balancing any residual. It is not closed-loop dynamic admittance.
"""
from __future__ import annotations

import numpy as np

from digital_twin.sim_core import SimArm
from digital_twin.twin_params import load_twin_kwargs


class TangentCompliance:
    def __init__(self, twin_kwargs=None):
        self.arm = SimArm(**(load_twin_kwargs(log=lambda _: None)
                             if twin_kwargs is None else twin_kwargs))
        self.m, self.d, self.mj = self.arm.model, self.arm.data, self.arm._mujoco
        self.order = self.arm._q_qposadr
        self.dofs = np.array([self.m.jnt_dofadr[
            np.flatnonzero(self.m.jnt_qposadr == adr)[0]] for adr in self.order])
        self.site = self.m.site("canarm_tip").id
        self.body = int(self.m.site_bodyid[self.site])

    def torque(self, q, p_pa):
        self.mj.mj_resetData(self.m, self.d)
        self.d.qpos[self.order] = q
        self.mj.mj_forward(self.m, self.d)
        dl = self.d.ten_length[self.arm._ten_ids] - self.arm._ten_len0
        self.d.ctrl[self.arm._act_ids] = self.arm.actuator.force_n(p_pa, dl)
        self.mj.mj_forward(self.m, self.d)
        return (self.d.qfrc_passive + self.d.qfrc_actuator - self.d.qfrc_bias)[self.dofs].copy()

    def stiffness(self, q, p_pa, eps=1e-5):
        q = np.asarray(q, dtype=float)
        K = np.empty((12, 12))
        for j in range(12):
            dq = np.eye(12)[j] * eps
            K[:, j] = -(self.torque(q + dq, p_pa) - self.torque(q - dq, p_pa)) / (2 * eps)
        return (K + K.T) / 2

    def predict(self, q, p_pa):
        K = self.stiffness(q, p_pa)
        self.torque(q, p_pa)
        J = np.zeros((3, self.m.nv))
        # The controller tracks the final plate centre, not the 50 mm stub.
        self.mj.mj_jac(self.m, self.d, J, None, self.d.xpos[self.body], self.body)
        J = J[:, self.dofs]
        if np.linalg.eigvalsh(K).min() <= 0:
            raise ValueError("Nonpositive tangent stiffness: no stable compliance ellipse")
        return J @ np.linalg.solve(K, J.T)
