"""V25 candidate controller with joint bounded actions and preview observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .anchored_dynamic_env import AnchoredDynamicEnv, PolicyGatedRelease
from .contextual_env import CONTEXT_NAMES, measured_context, normalized_action
from .release_calibration import FEATURE_NAMES, feature_vector, release_candidate


SCHEMA = "can_v15_twin_residual_v25"
JOINT_ACTIONS = tuple((force, radius) for force in (-1, 0, 1)
                      for radius in (-1, 0, 1))
JOINT_ACTION_NAMES = tuple(f"force_{force:+d}_radius_{radius:+d}"
                           for force, radius in JOINT_ACTIONS)
HOLD_ACTION = JOINT_ACTIONS.index((0, 0))
PREVIEW_NAMES = (
    "force_normalized", "radius_normalized", "handover_age_fraction",
    "time_remaining_fraction", "current_calibrated_error_fraction",
    "min_calibrated_error_fraction", "error_delta_fraction",
    "landing_error_x_fraction", "landing_error_y_fraction",
    "safe_candidate_ready", "ood_margin_fraction",
) + tuple(f"previous_joint_{name}" for name in JOINT_ACTION_NAMES)
OBSERVATION_NAMES = CONTEXT_NAMES + PREVIEW_NAMES


@dataclass(frozen=True)
class TwinResidualConfig:
    decision_interval_s: float = .5
    force_step: float = .05
    radius_step: float = .0225
    release_tolerance_m: float = .08
    preview_horizon_s: float = 2.

    def __post_init__(self):
        values = tuple(getattr(self, name) for name in (
            "decision_interval_s", "force_step", "radius_step",
            "release_tolerance_m", "preview_horizon_s"))
        if not np.isfinite(values).all():
            raise ValueError("V25 control values must be finite")
        if not .25 <= self.decision_interval_s <= 1.:
            raise ValueError("decision interval must be in [0.25, 1]")
        if not .02 <= self.force_step <= .1:
            raise ValueError("force step must be in [0.02, 0.1]")
        if not .01 <= self.radius_step <= .045:
            raise ValueError("radius step must be in [0.01, 0.045]")
        if not .04 <= self.release_tolerance_m <= .08:
            raise ValueError("release tolerance must be in [0.04, 0.08]")
        if not .5 <= self.preview_horizon_s <= 4.:
            raise ValueError("preview horizon must be in [0.5, 4]")


class TwinPreviewRelease(PolicyGatedRelease):
    """The V24 release guard plus the current sensor-only landing prediction."""
    ORIGIN = "v25_twin_residual_safe_release"

    def __init__(self, original, controller, config, calibration):
        super().__init__(original, controller, config, calibration)
        object.__setattr__(self, "current_preview", None)

    def __setattr__(self, name, value):
        if name == "current_preview":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def refresh(self, observation):
        if self.cache_time == observation["t"]:
            return
        self.cache_time, self.choice = observation["t"], None
        self.current_preview = None
        if not observation["held"] or self.original.t_release is not None:
            return
        handover = self.controller.t_hand
        if handover is None:
            return
        age = float(observation["t_win"] - handover)
        if age > self.config.release_max_s:
            self.audit["after_release_window_ticks"] += 1
            return
        original = self.original
        original.speed_gain = 1.
        original.az_bias = np.radians(self.config.release_azimuth_bias_deg)
        position, velocity = original.ball_state(observation)
        if (not np.isfinite(np.r_[position, velocity]).all() or
                getattr(original.ball_state, "lead", None) is None):
            return
        self.audit["fresh_ticks"] += 1
        original._rates(float(observation["t_win"]), velocity)
        candidate = release_candidate(original, position, velocity, vent_s=0.)
        self.audit["candidate_ticks"] += 1
        if candidate["speed_m_s"] < original.v_min or candidate["elevation_rad"] < original.min_elev:
            return
        feature_age = min(age, self.config.calibration_age_cap_s)
        feature_observation = observation
        if feature_age != age:
            self.audit["age_clipped_ticks"] += 1
            feature_observation = dict(observation,
                t_win=float(handover) + self.config.calibration_age_cap_s)
        features = feature_vector(position, velocity, candidate, self.controller,
                                  feature_observation, original.target)
        normalized = np.abs((features - self.calibration.mean) / self.calibration.scale)
        max_feature_index = int(normalized.argmax())
        max_feature_z = float(normalized[max_feature_index])
        if max_feature_z > (self.audit["max_feature_z"] or 0.):
            self.audit["max_feature_z"] = max_feature_z
            self.audit["max_feature_z_name"] = FEATURE_NAMES[max_feature_index]
        if max_feature_z > self.config.feature_z_limit:
            self.audit["blocked_ood_ticks"] += 1
            return
        joint_deg = float(np.degrees(np.abs(np.asarray(observation["q_meas"], dtype=float)).max()))
        if joint_deg > self.config.release_joint_deg:
            self.audit["blocked_joint_ticks"] += 1
            return
        calibrated_landing = self.calibration.predict_landing(candidate["landing"], features)
        calibrated_error = float(np.linalg.norm(calibrated_landing - original.target))
        raw_error = float(np.linalg.norm(candidate["landing"][:2] - original.target))
        current = dict(**candidate, features=features,
            calibrated_landing=calibrated_landing, calibrated_error_m=calibrated_error,
            raw_error_m=raw_error, age_s=age, feature_age_s=feature_age,
            max_feature_z=max_feature_z)
        self.current_preview = current
        if (self.audit["min_calibrated_error_m"] is None or
                calibrated_error < self.audit["min_calibrated_error_m"]):
            self.audit.update(min_calibrated_error_m=calibrated_error,
                raw_error_at_min_m=raw_error, min_error_age_s=age,
                calibrated_landing_at_min=calibrated_landing.tolist(),
                raw_landing_at_min=candidate["landing"][:2].tolist(),
                min_error_max_feature_z=max_feature_z)
        if calibrated_error < original.best[0]:
            original.best = (calibrated_error, float(observation["t_win"]))
        # The actor may observe the calibrated counterfactual landing estimate
        # before release is permitted.  The guard still owns the hard minimum
        # release age, so preview availability cannot open the gripper early.
        if age < self.config.release_min_s:
            return
        if calibrated_error <= self.config.release_immediate_m:
            self.valley_best, self.ticks_since_best = current, 0
            self.audit["allowed_ticks"] += 1
            self.audit["trigger_best_error_m"] = calibrated_error
            self.audit["trigger_mode"] = "immediate_strict"
            self.choice = current
            return
        if calibrated_error > self.config.release_tolerance_m + self.config.release_hysteresis_m:
            self.valley_best, self.ticks_since_best = None, 0
            self.audit["blocked_error_ticks"] += 1
            return
        if self.valley_best is None or calibrated_error < self.valley_best["calibrated_error_m"]:
            self.valley_best, self.ticks_since_best = current, 0
        else:
            self.ticks_since_best += 1
        if (self.valley_best["calibrated_error_m"] > self.config.release_tolerance_m or
                self.ticks_since_best < self.config.release_confirm_ticks):
            self.audit["valley_wait_ticks"] += 1
            return
        self.audit["allowed_ticks"] += 1
        self.audit["trigger_best_error_m"] = self.valley_best["calibrated_error_m"]
        self.audit["trigger_mode"] = "confirmed_valley"
        self.choice = current


class TwinResidualEnv(AnchoredDynamicEnv):
    """Forced candidate path used by preview supervision and candidate PPO."""

    def __init__(self, scenario, recipe, anchor, calibration, selector,
                 dynamic=TwinResidualConfig()):
        if not isinstance(dynamic, TwinResidualConfig):
            raise TypeError("TwinResidualEnv requires TwinResidualConfig")
        self.twin_dynamic = dynamic
        # The parent validates equivalent physical bounds; V25 overrides actions.
        from .anchored_dynamic_env import DynamicControlConfig
        parent = DynamicControlConfig(dynamic.decision_interval_s, dynamic.force_step,
            dynamic.radius_step, dynamic.release_tolerance_m)
        super().__init__(scenario, recipe, anchor, calibration, selector, parent)
        self.previous_joint_action = HOLD_ACTION
        self.previous_current_error = None

    def reset(self, seed):
        observation = super().reset(seed)
        self.releaser = TwinPreviewRelease(self.releaser.original, self.handover,
            self.continuous, self.release_calibration)
        self.previous_joint_action = HOLD_ACTION
        self.previous_current_error = None
        return observation

    def initial_observation(self, context):
        context = np.asarray(context, dtype=np.float32)
        action = normalized_action(self.continuous.envelope_force_scale,
                                   self.continuous.envelope_radius_scale)
        extra = np.r_[action, 0., 1., 1., 1., 0., 0., 0., 0., 1.,
                      np.eye(len(JOINT_ACTIONS), dtype=np.float32)[HOLD_ACTION]]
        value = np.r_[context, extra].astype(np.float32)
        if value.shape != (len(OBSERVATION_NAMES),):
            raise RuntimeError("V25 initial observation schema is inconsistent")
        return value

    def policy_observation(self):
        if self.trajectory_decision is None or not self.obs["held"]:
            raise RuntimeError("V25 policy observation requested before handover")
        self.releaser.refresh(self.obs)
        context = measured_context(self.model, self.obs, self.target)
        action = normalized_action(self.continuous.envelope_force_scale,
                                   self.continuous.envelope_radius_scale)
        age_s = float(self.obs["t_win"] - self.handover.t_hand)
        age = float(np.clip(age_s / self.continuous.release_max_s, 0., 1.))
        remaining = float(np.clip((self.continuous.release_max_s - age_s) /
                                  self.continuous.release_max_s, 0., 1.))
        preview = self.releaser.current_preview
        current_error = .30 if preview is None else float(preview["calibrated_error_m"])
        minimum = self.releaser.audit.get("min_calibrated_error_m")
        minimum = .30 if minimum is None else float(minimum)
        delta = 0. if self.previous_current_error is None else self.previous_current_error-current_error
        self.previous_current_error = current_error
        landing_delta = (np.zeros(2) if preview is None else
                         np.asarray(preview["calibrated_landing"]) - np.asarray(self.target))
        max_z = self.continuous.feature_z_limit if preview is None else float(preview["max_feature_z"])
        ood_margin = (self.continuous.feature_z_limit-max_z)/self.continuous.feature_z_limit
        extra = np.r_[action, age, remaining,
            np.clip(current_error/.30, 0., 3.), np.clip(minimum/.30, 0., 3.),
            np.clip(delta/.30, -1., 1.), np.clip(landing_delta/.30, -2., 2.),
            float(self.releaser.choice is not None), np.clip(ood_margin, -1., 1.),
            np.eye(len(JOINT_ACTIONS), dtype=np.float32)[self.previous_joint_action]]
        value = np.r_[context, extra].astype(np.float32)
        if value.shape != (len(OBSERVATION_NAMES),) or not np.isfinite(value).all():
            raise RuntimeError("V25 produced an invalid policy observation")
        return value

    def valid_action_mask(self):
        force = self.continuous.envelope_force_scale
        radius = self.continuous.envelope_radius_scale
        mask = []
        for force_direction, radius_direction in JOINT_ACTIONS:
            next_force = float(np.clip(force + force_direction*self.twin_dynamic.force_step, 1., 1.4))
            next_radius = float(np.clip(radius + radius_direction*self.twin_dynamic.radius_step, 1., 1.18))
            changed = abs(next_force-force) > 1e-10 or abs(next_radius-radius) > 1e-10
            mask.append(changed or (force_direction == 0 and radius_direction == 0))
        return np.asarray(mask, dtype=bool)

    def apply_joint_action(self, action):
        if not self.dynamic_due():
            raise RuntimeError("V25 action applied outside its decision time")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(JOINT_ACTIONS):
            raise ValueError("V25 joint action is outside the catalogue")
        action = int(action)
        if not self.valid_action_mask()[action]:
            raise ValueError("V25 joint action has no physical effect at this boundary")
        force_direction, radius_direction = JOINT_ACTIONS[action]
        force = float(np.clip(self.continuous.envelope_force_scale +
                              force_direction*self.twin_dynamic.force_step, 1., 1.4))
        radius = float(np.clip(self.continuous.envelope_radius_scale +
                               radius_direction*self.twin_dynamic.radius_step, 1., 1.18))
        parameters = dict(envelope_force_scale=force, envelope_radius_scale=radius)
        self.continuous = replace(self.continuous, **parameters)
        self.handover.config = self.continuous
        object.__setattr__(self.releaser, "config", self.continuous)
        self.releaser.set_permission(True)
        self.previous_joint_action = action
        self.dynamic_decisions.append(dict(time_s=float(self.obs["t_win"]), action=action,
            action_name=JOINT_ACTION_NAMES[action], parameters=parameters))
        while self.next_dynamic_time <= self.obs["t_win"] + 1e-9:
            self.next_dynamic_time += self.twin_dynamic.decision_interval_s

    def summary(self):
        result = super().summary()
        result.update(twin_residual_schema=SCHEMA,
                      twin_residual_config=asdict(self.twin_dynamic))
        return result
