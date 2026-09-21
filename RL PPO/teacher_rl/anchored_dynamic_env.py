"""V15-anchored, low-frequency post-catch control for constrained PPO.

The frozen V15 policy still owns approach and capture.  Once the ball is held,
an initial selector chooses one of the measured V23 force/radius settings.  A
second policy then acts every 0.5 s: hold, adjust force, adjust radius, and
independently wait or permit a calibrated release.  A release permission never
opens the gripper by itself; the accepted sensor-only release guard must still
produce an in-envelope candidate.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np

from .contextual_calibration import WarmCalibratedRelease, WarmContextualEnv
from .contextual_env import (CONTEXT_NAMES, base_config, measured_context,
    normalized_action)


SCHEMA = "can_v15_anchor_dynamic_ppo_v2"
TRAJECTORY_ACTIONS = tuple(
    (force, radius) for force in (1., 1.2, 1.4) for radius in (1., 1.09, 1.18)
)
INITIAL_ACTIONS = (None,) + TRAJECTORY_ACTIONS
INITIAL_ACTION_NAMES = ("v15_passthrough",) + tuple(
    f"force_{force:.2f}_radius_{radius:.2f}" for force, radius in TRAJECTORY_ACTIONS)
MOTION_NAMES = ("hold", "force_down", "force_up", "radius_down", "radius_up")
RELEASE_NAMES = ("wait", "allow_safe_release")
DYNAMIC_NAMES = (
    "force_normalized", "radius_normalized", "handover_age_fraction",
    "min_calibrated_error_fraction", "error_improvement_fraction",
    "safe_candidate_ready",
) + tuple(f"previous_motion_{name}" for name in MOTION_NAMES) + (
    "previous_release_allow",)
OBSERVATION_NAMES = CONTEXT_NAMES + DYNAMIC_NAMES


@dataclass(frozen=True)
class DynamicControlConfig:
    decision_interval_s: float = .5
    force_step: float = .1
    radius_step: float = .045
    release_tolerance_m: float = .08

    def __post_init__(self):
        values = (self.decision_interval_s, self.force_step,
                  self.radius_step, self.release_tolerance_m)
        if not np.isfinite(values).all():
            raise ValueError("dynamic control values must be finite")
        if not .25 <= self.decision_interval_s <= 1.:
            raise ValueError("decision_interval_s must be in [0.25, 1]")
        if not .025 <= self.force_step <= .2:
            raise ValueError("force_step must be in [0.025, 0.2]")
        if not .015 <= self.radius_step <= .09:
            raise ValueError("radius_step must be in [0.015, 0.09]")
        if not .04 <= self.release_tolerance_m <= .08:
            raise ValueError("release_tolerance_m must be in [0.04, 0.08]")


def initial_action(index):
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(INITIAL_ACTIONS):
        raise ValueError("initial action index is outside V15 plus the 3x3 action grid")
    # Index zero is executed by restarting the episode through the exact V15
    # baseline in episode_worker.  This neutral value is only a placeholder for
    # the handover tick that detects the selection; that partial simulation is
    # discarded and cannot contribute an outcome or PPO sample.
    parameters = (1.2, 1.09) if int(index) == 0 else INITIAL_ACTIONS[int(index)]
    return normalized_action(*parameters).astype(np.float32)


class PolicyGatedRelease(WarmCalibratedRelease):
    """Refresh the validated release estimator continuously, but obey its gate."""
    ORIGIN = "v15_anchor_dynamic_safe_release"

    def __init__(self, original, controller, config, calibration):
        super().__init__(original, controller, config, calibration)
        object.__setattr__(self, "allow_release", False)
        self.audit.update(policy_allow_ticks=0, policy_wait_ticks=0,
                          policy_blocked_candidates=0)

    def set_permission(self, allow):
        object.__setattr__(self, "allow_release", bool(allow))

    def __call__(self, plant, observation):
        # Refresh even while waiting so that waiting cannot hide stale history or
        # calibration-envelope violations from the audit.
        self.refresh(observation)
        key = "policy_allow_ticks" if self.allow_release else "policy_wait_ticks"
        self.audit[key] += 1
        if not self.allow_release:
            if self.choice is not None:
                self.audit["policy_blocked_candidates"] += 1
            return
        # CalibratedRelease.__call__ refreshes again, but its timestamp cache
        # makes that a no-op before it schedules the already-vetted candidate.
        super().__call__(plant, observation)


class AnchoredDynamicEnv(WarmContextualEnv):
    """Continuous V23 handover with bounded low-frequency policy decisions."""

    def __init__(self, scenario, recipe, anchor, calibration, selector,
                 dynamic=DynamicControlConfig()):
        if not isinstance(dynamic, DynamicControlConfig):
            raise TypeError("dynamic must be DynamicControlConfig")
        self.dynamic = dynamic
        self.dynamic_decisions = []
        self.next_dynamic_time = None
        self.previous_motion = 0
        self.previous_release = 0
        self.previous_min_error = None
        # At 150 Hz a valid <=8 cm candidate can last only one control tick.
        # Treat the selected tolerance as the immediate safe window; retaining
        # the old 4 cm immediate threshold would silently discard those valid
        # candidates while waiting for a second valley-confirmation tick.
        config = replace(base_config(tolerance=dynamic.release_tolerance_m),
            release_immediate_m=dynamic.release_tolerance_m,
            release_confirm_ticks=0)
        super().__init__(scenario, recipe, anchor, calibration, selector, config)

    def reset(self, seed):
        observation = super().reset(seed)
        self.releaser = PolicyGatedRelease(self.releaser.original, self.handover,
            self.continuous, self.release_calibration)
        self.dynamic_decisions = []
        self.next_dynamic_time = None
        self.previous_motion = 0
        self.previous_release = 0
        self.previous_min_error = None
        return observation

    def choose_trajectory(self, observation):
        super().choose_trajectory(observation)
        self.next_dynamic_time = (float(observation["t_win"])
                                  + self.dynamic.decision_interval_s)

    def initial_observation(self, context):
        context = np.asarray(context, dtype=np.float32)
        if context.shape != (len(CONTEXT_NAMES),) or not np.isfinite(context).all():
            raise ValueError("initial context has the wrong shape or nonfinite values")
        action = normalized_action(self.continuous.envelope_force_scale,
                                   self.continuous.envelope_radius_scale)
        extra = np.r_[action, 0., 1., 0., 0., np.eye(len(MOTION_NAMES))[0], 0.]
        result = np.r_[context, extra].astype(np.float32)
        if result.shape != (len(OBSERVATION_NAMES),):
            raise RuntimeError("anchored dynamic observation schema is inconsistent")
        return result

    def dynamic_due(self):
        age = (None if self.handover.t_hand is None else
               float(self.obs["t_win"] - self.handover.t_hand))
        return bool(self.trajectory_decision is not None and self.next_dynamic_time is not None
                    and not self.done and self.releaser.original.t_release is None
                    and self.obs["held"] and age is not None
                    and age <= self.continuous.release_max_s + 1e-9
                    and self.obs["t_win"] + 1e-9 >= self.next_dynamic_time)

    def policy_observation(self):
        if self.trajectory_decision is None or not self.obs["held"]:
            raise RuntimeError("dynamic policy observation requested before handover")
        context = measured_context(self.model, self.obs, self.target)
        action = normalized_action(self.continuous.envelope_force_scale,
                                   self.continuous.envelope_radius_scale)
        audit = self.releaser.audit
        error = audit.get("min_calibrated_error_m")
        error_fraction = 1. if error is None else float(np.clip(error / .30, 0., 1.))
        improvement = 0. if error is None or self.previous_min_error is None else float(
            np.clip((self.previous_min_error - error) / .30, -1., 1.))
        if error is not None:
            self.previous_min_error = float(error)
        age = float(np.clip((self.obs["t_win"] - self.handover.t_hand) / 18., 0., 1.))
        extra = np.r_[action, age, error_fraction, improvement,
                      float(self.releaser.choice is not None),
                      np.eye(len(MOTION_NAMES))[self.previous_motion],
                      float(self.previous_release)]
        result = np.r_[context, extra].astype(np.float32)
        if result.shape != (len(OBSERVATION_NAMES),) or not np.isfinite(result).all():
            raise RuntimeError("dynamic policy produced an invalid measured observation")
        return result

    def apply_dynamic(self, motion, release_allow):
        if not self.dynamic_due():
            raise RuntimeError("dynamic action applied outside its decision time")
        if not isinstance(motion, (int, np.integer)) or not 0 <= int(motion) < len(MOTION_NAMES):
            raise ValueError("motion action is outside the bounded catalogue")
        if int(release_allow) not in (0, 1):
            raise ValueError("release action must be wait or allow_safe_release")
        force = self.continuous.envelope_force_scale
        radius = self.continuous.envelope_radius_scale
        motion = int(motion)
        if motion == 1:
            force -= self.dynamic.force_step
        elif motion == 2:
            force += self.dynamic.force_step
        elif motion == 3:
            radius -= self.dynamic.radius_step
        elif motion == 4:
            radius += self.dynamic.radius_step
        force = float(np.clip(force, 1., 1.4))
        radius = float(np.clip(radius, 1., 1.18))
        parameters = dict(envelope_force_scale=force, envelope_radius_scale=radius)
        self.continuous = replace(self.continuous, **parameters)
        self.handover.config = self.continuous
        object.__setattr__(self.releaser, "config", self.continuous)
        self.releaser.set_permission(bool(release_allow))
        self.previous_motion, self.previous_release = motion, int(release_allow)
        self.dynamic_decisions.append(dict(time_s=float(self.obs["t_win"]),
            motion=motion, motion_name=MOTION_NAMES[motion],
            release=int(release_allow), release_name=RELEASE_NAMES[int(release_allow)],
            parameters=parameters))
        # Preserve cadence after a delayed process step without accumulating drift.
        while self.next_dynamic_time <= self.obs["t_win"] + 1e-9:
            self.next_dynamic_time += self.dynamic.decision_interval_s

    def summary(self):
        result = super().summary()
        result.update(anchored_dynamic_schema=SCHEMA,
            anchored_dynamic_config=asdict(self.dynamic),
            dynamic_decisions=list(self.dynamic_decisions),
            calibrated_release=dict(self.releaser.audit))
        return result
