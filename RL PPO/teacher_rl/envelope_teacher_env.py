"""V22 online teacher over the accepted expanded trajectory envelope."""
from dataclasses import asdict, dataclass

import numpy as np

from .calibrated_teacher_env import CalibratedRelease
from .release_calibration import RidgeLandingCalibration
from .trajectory_envelope_rl import EnvelopeHandover
from .trajectory_release_env import TrajectoryReleaseConfig, TrajectoryReleaseEnv

SCHEMA = "can_envelope_teacher_v22"


@dataclass(frozen=True)
class EnvelopeTeacherConfig(TrajectoryReleaseConfig):
    release_speed_gain: float = 1.
    release_min_s: float = 2.
    release_max_s: float = 18.
    calibration_age_cap_s: float = 16.
    feature_z_limit: float = 4.5
    release_confirm_ticks: int = 1
    release_hysteresis_m: float = .01
    release_immediate_m: float = .04
    envelope_force_scale: float = 1.
    envelope_radius_scale: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if self.release_speed_gain != 1.:
            raise ValueError("V22 calibration requires release_speed_gain=1")
        if not 16. <= self.release_max_s <= 18. or self.release_max_s <= self.release_min_s:
            raise ValueError("V22 release_max_s must be in [16, 18]")
        if self.calibration_age_cap_s != 16.:
            raise ValueError("V22 calibration age must be capped at 16 s")
        if not np.isfinite(self.feature_z_limit) or not 2. <= self.feature_z_limit <= 8.:
            raise ValueError("feature_z_limit must be in [2, 8]")
        if self.release_confirm_ticks not in (0, 1, 2, 3):
            raise ValueError("release_confirm_ticks must be in [0, 3]")
        if not np.isfinite(self.release_hysteresis_m) or not 0. <= self.release_hysteresis_m <= .02:
            raise ValueError("release_hysteresis_m must be in [0, 0.02]")
        if (not np.isfinite(self.release_immediate_m) or self.release_immediate_m < 0. or
                self.release_immediate_m > self.release_tolerance_m):
            raise ValueError("release_immediate_m must not exceed release_tolerance_m")
        if not 1. <= self.envelope_force_scale <= 1.4:
            raise ValueError("envelope_force_scale must be in [1, 1.4]")
        if not 1. <= self.envelope_radius_scale <= 1.18:
            raise ValueError("envelope_radius_scale must be in [1, 1.18]")


class EnvelopeCalibratedRelease(CalibratedRelease):
    ORIGIN = "v22_envelope_teacher"


class EnvelopeTeacherEnv(TrajectoryReleaseEnv):
    def __init__(self, scenario, recipe, anchor, calibration, config=EnvelopeTeacherConfig()):
        if not isinstance(config, EnvelopeTeacherConfig):
            raise TypeError("EnvelopeTeacherEnv requires EnvelopeTeacherConfig")
        if not isinstance(calibration, RidgeLandingCalibration):
            raise TypeError("calibration must be RidgeLandingCalibration")
        self.release_calibration = calibration
        super().__init__(scenario, recipe, anchor, config)

    def reset(self, seed):
        super().reset(seed)
        old = self.handover
        self.handover = EnvelopeHandover(self, old.catch, self.drive, self.model, self.continuous)
        self.controller.controller = self.handover
        self.releaser = EnvelopeCalibratedRelease(self.releaser.original, self.handover,
            self.continuous, self.release_calibration)
        return self.observation()

    def summary(self):
        result = super().summary()
        result.update(envelope_teacher_config=asdict(self.continuous),
            calibrated_release=dict(self.releaser.audit))
        return result
