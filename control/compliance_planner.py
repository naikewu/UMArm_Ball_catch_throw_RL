"""Torque-preserving common-pressure prior for compliance-aware MPPI."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from control.controller import PA_PER_PSI, pair_indices, project_pressures


class CompliancePressurePlanner:
    def __init__(self, head, dt):
        self.head, self.dt = head, dt
        self.mean_bounds = tuple(head.meta.get("pair_mean_psi", [2., 14.]))
        pairs = pair_indices()
        self.M, self.D = np.zeros((24, 12)), np.zeros((24, 12))
        self.M[pairs[:, 0], np.arange(12)] = 1
        self.M[pairs[:, 1], np.arange(12)] = 1
        self.D[pairs[:, 0], np.arange(12)] = .5
        self.D[pairs[:, 1], np.arange(12)] = -.5
        self.reset()

    def reset(self):
        self.mean = np.full(12, 6.)
        self.goal = self.mean.copy()
        self.tick = 0

    def command(self, prior, B, q_ref, C_ref):
        # Changing common pressure can produce torque on an asymmetric arm.
        # Solve the differences again so feedback/feedforward torque is retained.
        inverse = np.linalg.pinv(B @ self.D, rcond=1e-6)
        offset = self.D @ inverse @ (B @ (prior / PA_PER_PSI))
        gain = self.M - self.D @ inverse @ B @ self.M
        if self.tick % 15 == 0:
            def residual(mean):
                p = offset + gain @ mean
                C = self.head.predict(q_ref, p * PA_PER_PSI)
                return np.r_[(C[:2, :2] - C_ref[:2, :2]).ravel() / .01,
                             .001 * (mean - 8.),
                             np.minimum(p, 0.), np.maximum(p - 30., 0.)]
            result = least_squares(residual, self.goal, bounds=self.mean_bounds,
                                   max_nfev=15, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            self.goal = result.x
        self.tick += 1
        self.mean += np.clip(self.goal - self.mean, -5 * self.dt, 5 * self.dt)
        return project_pressures((offset + gain @ self.mean) * PA_PER_PSI)
