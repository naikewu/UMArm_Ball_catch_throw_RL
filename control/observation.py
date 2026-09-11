"""One causal camera-frame derivative for collection and deployed control."""
from __future__ import annotations

import numpy as np


class CausalJointObserver:
    """Low-pass finite differences, updated once per acquired camera frame.

    Each interval uses alpha=1-exp(-delta/tau). Repeated timestamps carry no
    new observation and leave the estimate unchanged. Updating at the camera
    source preserves intermediate samples when a slower control loop reads
    only the latest state.
    """
    def __init__(self, tau_s=0.04):
        if not np.isfinite(tau_s) or tau_s <= 0:
            raise ValueError("tau_s must be finite and positive")
        self.tau_s = float(tau_s)
        self.reset()

    def reset(self):
        self.stamp = None
        self.q = None
        self.qdot = np.zeros(12)

    def update(self, stamp, q):
        stamp = float(stamp)
        q = np.asarray(q, dtype=float)
        if not np.isfinite(stamp) or q.shape != (12,) or not np.all(np.isfinite(q)):
            raise ValueError("camera observation must be a finite timestamp and 12 finite radians")
        if self.stamp is None:
            self.stamp, self.q = stamp, q.copy()
            return self.qdot.copy()
        delta = stamp - self.stamp
        if delta < -1e-12:
            raise ValueError("camera acquisition timestamps must be increasing")
        if delta <= 1e-12:
            return self.qdot.copy()
        alpha = -np.expm1(-delta / self.tau_s)
        self.qdot += alpha * ((q - self.q) / delta - self.qdot)
        self.q, self.stamp = q.copy(), stamp
        return self.qdot.copy()
