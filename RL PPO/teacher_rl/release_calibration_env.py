"""V21 controlled release probes with offline-only departure labels."""
from dataclasses import asdict, dataclass

import numpy as np

from .continuous_release import TimestampedValve
from .release_calibration import feature_vector, release_candidate
from .trajectory_release_env import TrajectoryReleaseConfig, TrajectoryReleaseEnv


@dataclass(frozen=True)
class CalibrationProbeConfig(TrajectoryReleaseConfig):
    """One bounded orbit recipe and a deterministic post-handover probe phase."""
    probe_age_s: float = 3.
    probe_name: str = "unnamed"

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.probe_age_s) or not 1.5 <= self.probe_age_s <= 6.:
            raise ValueError("probe_age_s must be in [1.5, 6.0]")
        if not self.probe_name or not self.probe_name.replace("_", "").isalnum():
            raise ValueError("probe_name must be a nonempty alphanumeric identifier")


class ProbeRelease:
    """Schedule a fresh measured-state release at a planned phase without target gating."""
    def __init__(self, original, controller, config):
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "cache_time", None)
        object.__setattr__(self, "audit", dict(probe_name=config.probe_name, probe_age_s=config.probe_age_s,
            fresh_ticks=0, blocked_joint_ticks=0, scheduled=False, command=None))

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __setattr__(self, name, value):
        if name in ("cache_time", "audit"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.original, name, value)

    def __call__(self, plant, observation):
        if self.cache_time == observation["t"] or self.original.t_release is not None or not observation["held"]:
            return
        self.cache_time = observation["t"]
        handover = self.controller.t_hand
        if handover is None or observation["t_win"] - handover < self.config.probe_age_s:
            return
        original = self.original
        original.speed_gain = 1.
        original.az_bias = np.radians(self.config.release_azimuth_bias_deg)
        position, velocity = original.ball_state(observation)
        if not np.isfinite(np.r_[position, velocity]).all() or getattr(original.ball_state, "lead", None) is None:
            return
        self.audit["fresh_ticks"] += 1
        original._rates(float(observation["t_win"]), velocity)
        candidate = release_candidate(original, position, velocity)
        q_max = float(np.degrees(np.abs(np.asarray(observation["q_meas"], dtype=float)).max()))
        if q_max > self.config.release_joint_deg:
            self.audit["blocked_joint_ticks"] += 1
            return
        if candidate["speed_m_s"] < original.v_min or candidate["elevation_rad"] < original.min_elev:
            return
        features = feature_vector(position, velocity, candidate, self.controller, observation, original.target)
        valve = TimestampedValve(plant, observation["t"])
        if original.depart_fn is None:
            valve.gripper_open(due_in_s=candidate["delay_s"])
        else:
            valve.gripper_open(vent_in_s=candidate["vent_s"])
        original.t_release = float(observation["t_win"])
        original.vent_in, original.depart_pred = candidate["vent_s"], candidate["departure_s"]
        original.state_at_cmd = dict(landing_pred=candidate["landing"].tolist(),
            landing_err_pred=float(np.linalg.norm(candidate["landing"][:2] - original.target)),
            v=candidate["speed_m_s"], elev_pred=candidate["elevation_rad"], due_in_s=candidate["delay_s"],
            vent_in_s=candidate["vent_s"], depart_pred_s=candidate["departure_s"], origin="v21_probe")
        self.audit.update(scheduled=True, command=dict(features=features.tolist(),
            measured_position_m=np.asarray(position, dtype=float).tolist(),
            measured_velocity_m_s=np.asarray(velocity, dtype=float).tolist(),
            raw_landing_xy=candidate["landing"][:2].tolist(), raw_velocity_m_s=candidate["velocity"].tolist(),
            raw_position_m=candidate["position"].tolist(), delay_s=candidate["delay_s"],
            vent_s=candidate["vent_s"], departure_s=candidate["departure_s"], q_max_deg=q_max))


class CalibrationProbeEnv(TrajectoryReleaseEnv):
    """Adds offline true-departure labels while leaving online control sensor-only."""
    def __init__(self, scenario, recipe, anchor, config=CalibrationProbeConfig()):
        if not isinstance(config, CalibrationProbeConfig):
            raise TypeError("CalibrationProbeEnv requires CalibrationProbeConfig")
        super().__init__(scenario, recipe, anchor, config)

    def reset(self, seed):
        self.actual_departure = None
        self._last_held = False
        super().reset(seed)
        self.releaser = ProbeRelease(self.releaser.original, self.handover, self.continuous)
        return self.observation()

    def _physics_metrics(self, t):
        held_before = self._last_held
        super()._physics_metrics(t)
        held_now = bool(self.plant.ball_held())
        if (self.actual_departure is None and held_before and not held_now and
                self.releaser.t_release is not None):
            self.actual_departure = dict(time_s=float(t - 2.), position_m=self.plant.ball_pos().tolist(),
                velocity_m_s=self.plant.ball_vel().tolist())
        self._last_held = held_now

    def summary(self):
        result = super().summary()
        result.update(calibration_probe=dict(self.releaser.audit) if isinstance(self.releaser, ProbeRelease) else {},
            offline_actual_departure=self.actual_departure, calibration_probe_config=asdict(self.continuous))
        return result
