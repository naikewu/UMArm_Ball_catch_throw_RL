"""V23: one observable-state trajectory decision at continuous handover.

V15--V22 sources are intentionally unchanged so their checkpoint and calibration
fingerprints remain meaningful.  Labels may use simulation outcomes; decisions
here never read future states, ground-truth ball motion, or scenario seeds.
"""
from dataclasses import asdict, replace

import numpy as np

from .envelope_teacher_env import EnvelopeTeacherConfig, EnvelopeTeacherEnv
from .trajectory_envelope_rl import EnvelopeHandover, trajectories

SCHEMA = "can_contextual_trajectory_v23"
ACTION_NAMES = ("force_scale", "radius_scale")
CONTEXT_NAMES = (
    ("target_x", "target_y")
    + tuple(f"q_{i}" for i in range(12))
    + tuple(f"qd_{i}" for i in range(12))
    + ("ball_rel_x", "ball_rel_y", "ball_rel_z")
    + ("ball_vx", "ball_vy", "ball_vz")
    + ("tip_vx", "tip_vy", "tip_vz")
    + ("pressure_mean_psi", "pressure_min_psi", "pressure_max_psi", "joint_margin_deg")
)


def bounded_parameters(action):
    value = np.asarray(action, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1.000001):
        raise ValueError("V23 needs two finite actions in [-1, 1]")
    value = np.clip(value, -1., 1.)
    return dict(envelope_force_scale=float(1.2 + .2 * value[0]),
                envelope_radius_scale=float(1.09 + .09 * value[1]))


def normalized_action(force, radius):
    action = np.array([(force - 1.2) / .2, (radius - 1.09) / .09])
    bounded_parameters(action)
    return np.clip(action, -1., 1.)


def base_config(profile="force_125", tolerance=.04):
    values = dict(trajectories()[profile], release_min_s=2., release_max_s=18.,
                  calibration_age_cap_s=16., feature_z_limit=4.5,
                  release_tolerance_m=tolerance, release_confirm_ticks=1,
                  release_immediate_m=.04)
    return EnvelopeTeacherConfig(**values)


def measured_context(model, observation, target):
    q = np.asarray(observation["q_meas"], dtype=float)
    qd = np.asarray(observation["qd_hat"], dtype=float)
    tip, jacobian, _ = model.jac(q)
    ball = observation["ball_hat"]
    pressure = np.asarray(observation["p_view_pa"], dtype=float) / 6894.757293168
    vector = np.r_[target, q, qd, np.asarray(ball["pos"]) - tip, ball["vel"],
                   jacobian @ qd, pressure.mean(), pressure.min(), pressure.max(),
                   36. - np.degrees(np.abs(q).max())]
    if vector.shape != (len(CONTEXT_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("V23 handover context must contain finite measured features")
    return vector.astype(np.float32)


class ContextualHandover(EnvelopeHandover):
    def command(self, observation):
        if observation["held"] and self.env.trajectory_decision is None:
            self.env.choose_trajectory(observation)
        return super().command(observation)


class ContextualEnv(EnvelopeTeacherEnv):
    def __init__(self, scenario, recipe, anchor, calibration, selector, config=None):
        self.selector = selector
        self.initial_config = base_config() if config is None else config
        self.trajectory_decision = None
        super().__init__(scenario, recipe, anchor, calibration, self.initial_config)

    def reset(self, seed):
        self.continuous = self.initial_config
        self.trajectory_decision = None
        super().reset(seed)
        old = self.handover
        self.handover = ContextualHandover(self, old.catch, self.drive, self.model, self.continuous)
        self.controller.controller = self.handover
        object.__setattr__(self.releaser, "controller", self.handover)
        return self.observation()

    def choose_trajectory(self, observation):
        context = measured_context(self.model, observation, self.target)
        action = np.asarray(self.selector(context.copy()), dtype=float)
        parameters = bounded_parameters(action)
        self.continuous = replace(self.continuous, **parameters)
        self.handover.config = self.continuous
        object.__setattr__(self.releaser, "config", self.continuous)
        self.trajectory_decision = dict(context=context.tolist(), action=action.tolist(),
            parameters=parameters, time_s=float(observation["t_win"]))

    def summary(self):
        result = super().summary()
        result.update(contextual_schema=SCHEMA, trajectory_decision=self.trajectory_decision,
                      contextual_config=asdict(self.continuous))
        return result


def episode_utility(result):
    """Bounded, actual-outcome objective for the episode-level contextual policy.

    This is a learning signal, never an acceptance certificate.  Formal checks
    retain all individual task, contact, continuity, and joint constraints.
    """
    if not result["captured"]:
        return -30.
    if result.get("grip_broken") or result["max_joint_deg"] > 36.:
        return -40. - min(20., max(0., result["max_joint_deg"] - 36.))
    if not result["released"] or result.get("landing_error_m") is None:
        # A small diagnostic shaping term cannot outweigh actually releasing.
        error = result.get("calibrated_release", {}).get("min_calibrated_error_m")
        return -20. - min(5., error if error is not None else 5.)
    error = result["landing_error_m"]
    age = result.get("catch_to_release_s") or 0.
    return float(10. + 20. * bool(result["hit15"]) + 10. * np.exp(-error / .05)
                 - 20. * min(error, 1.) - .1 * age)
