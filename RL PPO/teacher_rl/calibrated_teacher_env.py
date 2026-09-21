"""V21 online teacher using the accepted sensor-only landing calibration."""
from dataclasses import asdict, dataclass

import numpy as np

from .continuous_release import TimestampedValve
from .release_calibration import FEATURE_NAMES, RidgeLandingCalibration, feature_vector, release_candidate
from .trajectory_release_env import TrajectoryReleaseConfig, TrajectoryReleaseEnv

SCHEMA = "can_calibrated_teacher_v21"


@dataclass(frozen=True)
class CalibratedTeacherConfig(TrajectoryReleaseConfig):
    """Restrict online release decisions to the V21 calibration envelope."""
    release_speed_gain: float = 1.
    release_min_s: float = 2.
    release_max_s: float = 18.
    calibration_age_cap_s: float = 4.
    feature_z_limit: float = 4.5
    release_confirm_ticks: int = 1
    release_hysteresis_m: float = .01
    release_immediate_m: float = .04

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.release_max_s) or not 4. <= self.release_max_s <= 24.:
            raise ValueError("release_max_s must be in [4, 24]")
        if self.release_max_s <= self.release_min_s:
            raise ValueError("release_max_s must exceed release_min_s")
        if self.calibration_age_cap_s != 4.:
            raise ValueError("V21 calibration age must be capped at its validated 4 s boundary")
        if self.release_confirm_ticks not in (0, 1, 2, 3):
            raise ValueError("release_confirm_ticks must be 0, 1, 2, or 3")
        if not np.isfinite(self.release_hysteresis_m) or not 0. <= self.release_hysteresis_m <= .02:
            raise ValueError("release_hysteresis_m must be in [0, 0.02]")
        if (not np.isfinite(self.release_immediate_m) or self.release_immediate_m < 0. or
                self.release_immediate_m > self.release_tolerance_m):
            raise ValueError("release_immediate_m must be in [0, release_tolerance_m]")
        if not np.isfinite(self.feature_z_limit) or not 2. <= self.feature_z_limit <= 8.:
            raise ValueError("feature_z_limit must be in [2, 8]")
        if self.release_speed_gain != 1.:
            raise ValueError("V21 calibration requires the raw release model speed gain of 1")


class CalibratedRelease:
    """Apply the offline calibration to online-observable release features only."""
    ORIGIN = "v21_calibrated_teacher"

    def __init__(self, original, controller, config, calibration):
        if not isinstance(calibration, RidgeLandingCalibration):
            raise TypeError("calibration must be RidgeLandingCalibration")
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "cache_time", None)
        object.__setattr__(self, "choice", None)
        object.__setattr__(self, "valley_best", None)
        object.__setattr__(self, "ticks_since_best", 0)
        object.__setattr__(self, "audit", dict(candidate_ticks=0, fresh_ticks=0, allowed_ticks=0,
            blocked_joint_ticks=0, blocked_error_ticks=0, blocked_ood_ticks=0,
            after_release_window_ticks=0, age_clipped_ticks=0,
            valley_wait_ticks=0, trigger_best_error_m=None, trigger_mode=None,
            scheduled=False, calibrated_error_m=None,
            raw_error_m=None, max_feature_z=None, max_feature_z_name=None,
            min_calibrated_error_m=None, raw_error_at_min_m=None, min_error_age_s=None,
            calibrated_landing_at_min=None, raw_landing_at_min=None,
            min_error_max_feature_z=None, release_age_s=None, feature_age_s=None, origin=None))

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __setattr__(self, name, value):
        if name in ("cache_time", "choice", "valley_best", "ticks_since_best", "audit"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.original, name, value)

    def refresh(self, observation):
        if self.cache_time == observation["t"]:
            return
        self.cache_time, self.choice = observation["t"], None
        if not observation["held"] or self.original.t_release is not None:
            return
        handover = self.controller.t_hand
        if handover is None:
            return
        age = float(observation["t_win"] - handover)
        if age < self.config.release_min_s:
            return
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
        # The calibration campaign used immediate valve commands.  Re-evaluate at
        # 150 Hz instead of extrapolating to uncalibrated future vent offsets.
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
        if (self.audit["min_calibrated_error_m"] is None or
                calibrated_error < self.audit["min_calibrated_error_m"]):
            self.audit.update(min_calibrated_error_m=calibrated_error,
                raw_error_at_min_m=raw_error, min_error_age_s=age,
                calibrated_landing_at_min=calibrated_landing.tolist(),
                raw_landing_at_min=candidate["landing"][:2].tolist(),
                min_error_max_feature_z=max_feature_z)
        if calibrated_error < original.best[0]:
            original.best = (calibrated_error, float(observation["t_win"]))
        current = dict(**candidate, features=features, calibrated_landing=calibrated_landing,
            calibrated_error_m=calibrated_error, raw_error_m=raw_error, age_s=age,
            feature_age_s=feature_age, max_feature_z=max_feature_z)
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

    def __call__(self, plant, observation):
        self.refresh(observation)
        if self.choice is None or self.original.t_release is not None or not observation["held"]:
            return
        choice = self.choice
        valve = TimestampedValve(plant, observation["t"])
        if self.original.depart_fn is None:
            valve.gripper_open(due_in_s=choice["delay_s"])
        else:
            valve.gripper_open(vent_in_s=choice["vent_s"])
        original = self.original
        original.t_release = float(observation["t_win"])
        original.vent_in, original.depart_pred = choice["vent_s"], choice["departure_s"]
        original.state_at_cmd = dict(landing_pred=choice["calibrated_landing"].tolist(),
            landing_err_pred=choice["calibrated_error_m"], raw_landing_pred=choice["landing"].tolist(),
            raw_landing_err_pred=choice["raw_error_m"], v=choice["speed_m_s"],
            elev_pred=choice["elevation_rad"], due_in_s=choice["delay_s"],
            vent_in_s=choice["vent_s"], depart_pred_s=choice["departure_s"],
            max_feature_z=choice["max_feature_z"], feature_age_s=choice["feature_age_s"],
            origin=self.ORIGIN)
        self.audit.update(scheduled=True, calibrated_error_m=choice["calibrated_error_m"],
            raw_error_m=choice["raw_error_m"], release_age_s=choice["age_s"],
            feature_age_s=choice["feature_age_s"], origin=self.ORIGIN)


class CalibratedTeacherEnv(TrajectoryReleaseEnv):
    """V15 capture plus bounded continuous orbit and calibrated sensor-only release."""
    def __init__(self, scenario, recipe, anchor, calibration, config=CalibratedTeacherConfig()):
        if not isinstance(config, CalibratedTeacherConfig):
            raise TypeError("CalibratedTeacherEnv requires CalibratedTeacherConfig")
        self.release_calibration = calibration
        super().__init__(scenario, recipe, anchor, config)

    def reset(self, seed):
        super().reset(seed)
        self.releaser = CalibratedRelease(self.releaser.original, self.handover,
            self.continuous, self.release_calibration)
        return self.observation()

    def summary(self):
        result = super().summary()
        result.update(calibrated_teacher_config=asdict(self.continuous),
            calibrated_release=dict(self.releaser.audit) if isinstance(self.releaser, CalibratedRelease) else {})
        return result
