"""V20 bounded orbit shaping and sensor-fitted predictive release."""
from dataclasses import asdict, dataclass

import numpy as np

from .continuous_release import TimestampedValve
from .safe_continuous_env import SafeContinuousConfig, SafeContinuousEnv, SafeHandover

SCHEMA = "can_trajectory_release_teacher_v20"


@dataclass(frozen=True)
class TrajectoryReleaseConfig(SafeContinuousConfig):
    """A safe continuous orbit with a measured-state, release-time advisor."""
    target_force_scale: float = 1.
    target_radius_scale: float = 1.
    release_joint_deg: float = 36.
    release_speed_gain: float = .88

    def __post_init__(self):
        super().__post_init__()
        for name, low, high in (
                ("target_force_scale", .80, 1.15), ("target_radius_scale", .85, 1.08),
                ("release_joint_deg", 30., 36.), ("release_speed_gain", .75, 1.00)):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.release_ball_frames != 24 or not self.align_release_clock:
            raise ValueError("V20 requires the 24-frame timestamp-aligned ball fit")
        if self.predict_ball_motion:
            raise ValueError("V20 release uses its own measured-state prediction")


class TrajectoryHandover(SafeHandover):
    """Apply the candidate orbit envelope before V19's per-tick safety shaping."""
    def command(self, obs):
        nominal_force, nominal_radius = self.drv.F_max, self.drv.r_final
        self.drv.F_max = nominal_force * self.config.target_force_scale
        self.drv.r_final = nominal_radius * self.config.target_radius_scale
        try:
            return super().command(obs)
        finally:
            self.drv.F_max, self.drv.r_final = nominal_force, nominal_radius


class PredictiveRelease:
    """Release only when a fresh fitted ball state predicts a safe current departure.

    This is intentionally a deterministic teacher advisor, not an RL release action.
    It examines the valve-delay candidates every control tick and commits only the
    best candidate that can be commanded in the current valve window.
    """
    def __init__(self, original, controller, config):
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "cache_time", None)
        object.__setattr__(self, "choice", None)
        object.__setattr__(self, "audit", dict(candidate_ticks=0, fresh_ticks=0,
            allowed_ticks=0, blocked_joint_ticks=0, blocked_error_ticks=0,
            scheduled=False, predicted_error_m=None, origin=None))

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __setattr__(self, name, value):
        if name in ("cache_time", "choice", "audit"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.original, name, value)

    def refresh(self, obs):
        if self.cache_time == obs["t"]:
            return
        self.cache_time, self.choice = obs["t"], None
        if not obs["held"] or self.original.t_release is not None:
            return
        handover = self.controller.t_hand
        if handover is None:
            return
        age = float(obs["t_win"] - handover)
        if age < self.config.release_min_s:
            return
        original = self.original
        # The measured attached-ball velocity is consistently larger than the
        # post-vent departure velocity.  Screen a bounded calibration instead
        # of pretending that a measured hand-held velocity is a free-flight one.
        original.speed_gain = self.config.release_speed_gain
        original.az_bias = np.radians(self.config.release_azimuth_bias_deg)
        position, velocity = original.ball_state(obs)
        if not np.isfinite(np.r_[position, velocity]).all():
            return
        fresh = getattr(original.ball_state, "lead", None) is not None
        if not fresh:
            return
        self.audit["fresh_ticks"] += 1
        original._rates(float(obs["t_win"]), velocity)
        ahead = int(round(original.horizon / (original.tick / original.sub)))
        candidates = []
        for index in range(original.sub + 1 + ahead):
            vent = index * original.tick / original.sub
            departure_delay = original.lat
            if original.depart_fn is not None:
                for _ in range(2):
                    _, predicted_velocity, _ = original.predict_from(position, velocity, vent + departure_delay)
                    acceleration = abs(original.omega) * float(np.hypot(predicted_velocity[0], predicted_velocity[1]))
                    departure_delay = float(original.depart_fn(original.m_ball * np.hypot(acceleration, 9.81)))
            landing, predicted_velocity, _ = original.predict_from(position, velocity, vent + departure_delay)
            error = float(np.linalg.norm(landing[:2] - original.target))
            speed = float(np.linalg.norm(predicted_velocity))
            elevation = float(np.arctan2(predicted_velocity[2], np.hypot(predicted_velocity[0], predicted_velocity[1])))
            if np.isfinite(np.r_[landing, predicted_velocity, error, departure_delay]).all():
                candidates.append(dict(index=index, vent_s=vent, delay_s=vent + departure_delay,
                    departure_s=departure_delay, landing=landing, velocity=predicted_velocity,
                    error_m=error, speed_m_s=speed, elevation_rad=elevation))
        if not candidates:
            return
        self.audit["candidate_ticks"] += 1
        commandable = [candidate for candidate in candidates if candidate["index"] <= original.sub and
            candidate["speed_m_s"] >= original.v_min and candidate["elevation_rad"] >= original.min_elev]
        if not commandable:
            return
        candidate = min(commandable, key=lambda row: row["error_m"])
        joint_deg = float(np.degrees(np.abs(obs["q_meas"]).max()))
        if joint_deg > self.config.release_joint_deg:
            self.audit["blocked_joint_ticks"] += 1
            return
        if candidate["error_m"] > self.config.release_tolerance_m:
            self.audit["blocked_error_ticks"] += 1
            return
        self.audit["allowed_ticks"] += 1
        self.choice = candidate
        if candidate["error_m"] < original.best[0]:
            original.best = (candidate["error_m"], float(obs["t_win"]))

    def __call__(self, plant, obs):
        self.refresh(obs)
        if self.choice is None or self.original.t_release is not None or not obs["held"]:
            return
        choice = self.choice
        valve = TimestampedValve(plant, obs["t"])
        if self.original.depart_fn is None:
            valve.gripper_open(due_in_s=choice["delay_s"])
        else:
            valve.gripper_open(vent_in_s=choice["vent_s"])
        original = self.original
        original.t_release = float(obs["t_win"])
        original.vent_in, original.depart_pred = choice["vent_s"], choice["departure_s"]
        original.state_at_cmd = dict(landing_pred=choice["landing"].tolist(),
            landing_err_pred=choice["error_m"], v=choice["speed_m_s"],
            elev_pred=choice["elevation_rad"], due_in_s=choice["delay_s"],
            vent_in_s=choice["vent_s"], depart_pred_s=choice["departure_s"], origin="predictive_teacher")
        self.audit.update(scheduled=True, predicted_error_m=choice["error_m"], origin="predictive_teacher")


class TrajectoryReleaseEnv(SafeContinuousEnv):
    """Freeze BC capture; vary only bounded orbit generation and release selection."""
    def __init__(self, scenario, recipe, anchor, config=TrajectoryReleaseConfig()):
        if not isinstance(config, TrajectoryReleaseConfig):
            raise TypeError("TrajectoryReleaseEnv requires TrajectoryReleaseConfig")
        super().__init__(scenario, recipe, anchor, config)

    def reset(self, seed):
        super().reset(seed)
        old = self.handover
        self.handover = TrajectoryHandover(self, old.catch, self.drive, self.model, self.continuous)
        self.controller.controller = self.handover
        self.releaser = PredictiveRelease(self.releaser.original, self.handover, self.continuous)
        return self.observation()

    def summary(self):
        result = super().summary()
        result.update(trajectory_release_config=asdict(self.continuous),
            predictive_release=dict(self.releaser.audit) if isinstance(self.releaser, PredictiveRelease) else {})
        return result
