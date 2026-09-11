"""Measured ProMax kinematics and the versioned RS485 cursive word.

The controlled point is the last plate centre, in the arm's base frame.
Scaling both path length and speed by the reach ratio preserves the original
traversal time. This geometric scaling does not assert pressure feasibility.
"""
from __future__ import annotations

from functools import lru_cache
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from UMArm_KINEMATICS.canarm_params import CANARM_PARAMS

_fk = importlib.import_module("UMArm_KINEMATICS.fkine")
_rp = importlib.import_module("UMArm_KINEMATICS.robot_params")
Q_LIMIT_RAD = np.deg2rad(20.0)
_SOURCE = json.loads((Path(__file__).parent / "assets/soft_reference.json").read_text())
REACH_M = float(-_fk.fkine(np.zeros(12), CANARM_PARAMS, order="yx")[2, 3])
ORIGINAL_REACH_M = float(_SOURCE["original_reach_m"])
LENGTH_SCALE = REACH_M / ORIGINAL_REACH_M


def _axes():
    axes, z = [], 0.0
    for row in CANARM_PARAMS:
        z -= row[_rp.COL_JD]
        for xi in _fk.segment_twists(row):
            v, w = xi[:3], xi[3:]
            axes.append(np.r_[v + np.cross([0, 0, z], w), w])
        z -= sum(row[i] for i in (_rp.COL_UC1, _rp.COL_AA1, _rp.COL_LL,
                                   _rp.COL_AA2, _rp.COL_UC2))
    return np.asarray(axes)


_AXES = _axes()
_ORDER = (1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11)
_W = _AXES[:, 3:]
_V = _AXES[:, :3]
_K = np.array([[[0, -w[2], w[1]], [w[2], 0, -w[0]],
                [-w[1], w[0], 0]] for w in _W])
_K2 = _K @ _K
_WXV = np.cross(_W, _V)
_HOME = np.array([0.0, 0.0, -REACH_M])


def _position(q, with_jacobian=False):
    q = np.asarray(q, dtype=float)
    if q.shape[-1:] != (12,) or not np.all(np.isfinite(q)):
        raise ValueError("q must be finite with final dimension 12")
    if with_jacobian and q.shape != (12,):
        raise ValueError("tip_jacobian accepts one joint vector")
    rot = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3)).copy()
    trans = np.zeros(q.shape[:-1] + (3,))
    velocities, axes = np.zeros((12, 3)), np.zeros((12, 3))
    for j in _ORDER:
        if with_jacobian:
            axes[j] = rot @ _W[j]
            velocities[j] = rot @ _V[j] + np.cross(trans, axes[j])
        a = q[..., j]
        r = np.eye(3) + np.sin(a)[..., None, None] * _K[j] + (
            1 - np.cos(a))[..., None, None] * _K2[j]
        p = np.einsum("...ij,j->...i", np.eye(3) - r, _WXV[j])
        trans += np.einsum("...ij,...j->...i", rot, p)
        rot = rot @ r
    pos = trans + np.einsum("...ij,j->...i", rot, _HOME)
    return (pos, (velocities + np.cross(axes, pos)).T) if with_jacobian else pos


def tip_position(q):
    """Plate-centre xyz in metres; accepts (12,) or (..., 12) radians."""
    return _position(q)


def tip_jacobian(q):
    """Analytic (3, 12) position derivative in metres per radian."""
    return _position(q, True)[1]


def inverse_kinematics(target_m, q0=None, *, max_angle_rad=Q_LIMIT_RAD,
                       return_info=False):
    """Bounded, warm-started position IK with a weak posture-continuity cost.

    Unreachable requests return the closest bounded solution and an explicit
    residual when return_info=True; callers can display it without promising
    that an arbitrary Cartesian target is attainable.
    """
    target = np.asarray(target_m, dtype=float)
    previous = np.zeros(12) if q0 is None else np.asarray(q0, dtype=float)
    limit = float(max_angle_rad)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_m must contain three finite coordinates")
    if previous.shape != (12,) or not np.all(np.isfinite(previous)) or limit <= 0:
        raise ValueError("q0 must contain 12 finite angles and limit must be positive")
    previous = np.clip(previous, -limit + 1e-9, limit - 1e-9)
    regularizer = 2e-4

    def residual(q):
        return np.r_[tip_position(q) - target, regularizer * (q - previous)]

    def jacobian(q):
        return np.vstack([tip_jacobian(q), regularizer * np.eye(12)])

    result = least_squares(residual, previous, jac=jacobian, bounds=(-limit, limit),
                           ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=60)
    info = {"residual_m": float(np.linalg.norm(tip_position(result.x) - target)),
            "converged": bool(result.success),
            "max_angle_deg": float(np.rad2deg(np.max(np.abs(result.x))))}
    return (result.x, info) if return_info else result.x


@lru_cache(maxsize=8)
def _word_plan(size_scale):
    xy = np.asarray(_SOURCE["xy_m"]) * LENGTH_SCALE * size_scale
    dome = _SOURCE["original_dome_m"] * LENGTH_SCALE
    r2 = (xy * xy).sum(axis=1)
    if np.any(r2 >= dome * dome):
        raise ValueError("word extends beyond the reference dome")
    points = np.column_stack([xy, -REACH_M + dome - np.sqrt(dome*dome - r2)])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-10]
    shape = CubicSpline(arc[keep], points[keep], axis=0)
    length = float(arc[-1])
    grid = np.linspace(0, length, int(np.ceil(length / 0.004)) + 1)
    qs, residuals = [], []
    q = np.zeros(12)
    for s in grid:
        q, info = inverse_kinematics(shape(s), q, return_info=True)
        qs.append(q.copy())
        residuals.append(info["residual_m"])
    joint = CubicSpline(grid, np.asarray(qs), axis=0)
    check_s = np.linspace(0, length, 2001)
    max_residual = float(np.max(np.linalg.norm(tip_position(joint(check_s)) - shape(check_s), axis=1)))
    max_angle = float(np.max(np.abs(joint(check_s))))
    if max_residual > 0.002 or max_angle > Q_LIMIT_RAD + 1e-5:
        raise ValueError(f"word IK is infeasible: {max_residual*1000:.2f} mm, "
                         f"{np.rad2deg(max_angle):.2f} deg; reduce size_scale")
    return shape, joint, length, max_residual, np.ptp(points[:, :2], axis=0)


def _blend(u):
    u = np.clip(u, 0.0, 1.0)
    return (10*u**3 - 15*u**4 + 6*u**5,
            30*u**2 - 60*u**3 + 30*u**4,
            60*u - 180*u**2 + 120*u**3)


class SoftTrajectory:
    """One identical space-time reference for every controller.

    Slow/fast retain the source 150/300 mm/s timing after reach scaling.
    Entry and speed ramps have zero acceleration at their endpoints. Joint
    position, velocity and acceleration come from the same C2 spline.
    """
    def __init__(self, speed="slow", *, size_scale=1.0, speed_scale=1.0,
                 entry_s=3.0, settle_s=1.0, hold_s=2.0, ramp_s=1.0):
        if size_scale <= 0 or speed_scale <= 0 or min(entry_s, ramp_s) <= 0 or min(settle_s, hold_s) < 0:
            raise ValueError("scales and entry/ramp durations must be positive")
        original_speed = {"slow": 0.15, "fast": 0.30}.get(speed, speed)
        if not isinstance(original_speed, (int, float)) or original_speed <= 0:
            raise ValueError("speed must be slow, fast, or a positive source speed in m/s")
        self.speed_m_s = float(original_speed) * LENGTH_SCALE * speed_scale
        self._shape, self._joint, self.length_m, residual, extent = _word_plan(float(size_scale))
        self.entry_s, self.ramp_s = float(entry_s), float(ramp_s)
        self.write_start_s = self.entry_s + float(settle_s)
        self.cruise_s = self.length_m / self.speed_m_s - self.ramp_s
        if self.cruise_s < 0:
            raise ValueError("speed is too high for the chosen ramp duration")
        self.write_end_s = self.write_start_s + 2*self.ramp_s + self.cruise_s
        self.duration_s = self.write_end_s + float(hold_s)
        self.seconds = self.duration_s
        self.metadata = {"source": {k: v for k, v in _SOURCE.items() if k != "xy_m"},
            "reach_m": REACH_M, "length_scale": LENGTH_SCALE,
            "size_scale": size_scale, "speed_scale": speed_scale,
            "extent_m": extent.tolist(), "path_length_m": self.length_m,
            "speed_m_s": self.speed_m_s, "duration_s": self.duration_s,
            "write_start_s": self.write_start_s, "write_end_s": self.write_end_s,
            "ik_max_residual_mm": 1000*residual, "joint_limit_deg": 20.0,
            "endpoint": "last plate centre in arm base frame; measured yx kinematics"}

    def _arc(self, t):
        x = float(t) - self.write_start_s
        r, v = self.ramp_s, self.speed_m_s
        if x <= 0:
            return 0.0, 0.0, 0.0
        if x < r:
            return (v*(x/2-r*np.sin(np.pi*x/r)/(2*np.pi)),
                    v*(1-np.cos(np.pi*x/r))/2, v*np.pi*np.sin(np.pi*x/r)/(2*r))
        if x < r+self.cruise_s:
            return v*(r/2+x-r), v, 0.0
        if x < 2*r+self.cruise_s:
            u = x-r-self.cruise_s
            return (v*(r/2+self.cruise_s+u/2+r*np.sin(np.pi*u/r)/(2*np.pi)),
                    v*(1+np.cos(np.pi*u/r))/2, -v*np.pi*np.sin(np.pi*u/r)/(2*r))
        return self.length_m, 0.0, 0.0

    def sample(self, t):
        if t < self.entry_s:
            q0 = self._joint(0)
            if t <= 0:
                return np.zeros(12), np.zeros(12), np.zeros(12)
            b, bd, bdd = _blend(float(t)/self.entry_s)
            return q0*b, q0*bd/self.entry_s, q0*bdd/self.entry_s**2
        s, sd, sdd = self._arc(t)
        q, qs, qss = self._joint(s), self._joint(s, 1), self._joint(s, 2)
        return q, qs*sd, qss*sd*sd + qs*sdd

    def future(self, t, H, dt):
        return np.stack([self.sample(t+(k+1)*dt)[0] for k in range(H)])

    def tip_target(self, t):
        return tip_position(self.sample(t)[0]) if t < self.entry_s else self._shape(self._arc(t)[0])

    def pen_down(self, t):
        return (np.asarray(t) >= self.write_start_s) & (np.asarray(t) <= self.write_end_s)

    def q(self, t):
        return self.sample(t)[0]

    def qd(self, t):
        return self.sample(t)[1]

    def qdd(self, t):
        return self.sample(t)[2]

    pos = tip_target
    active = pen_down
