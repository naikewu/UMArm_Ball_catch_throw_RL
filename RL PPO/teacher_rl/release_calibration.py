"""Shared measured-state release features and a small offline calibration model."""
from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "measured_x_rel_target", "measured_y_rel_target", "measured_z",
    "measured_vx", "measured_vy", "measured_vz",
    "raw_land_x_rel_target", "raw_land_y_rel_target",
    "raw_vx", "raw_vy", "raw_vz", "handover_age_s", "command_delay_s",
    "vent_in_s", "orbit_radius_m", "force_limit_n", "target_sin", "target_cos", "q_max_deg",
)
SCHEMA = "can_release_calibration_v21"


def departure_delay(original, position, velocity, vent_s):
    """Use the existing spring-delay model for a prospective valve instant."""
    delay = float(original.lat)
    if original.depart_fn is not None:
        for _ in range(2):
            _, predicted_velocity, _ = original.predict_from(position, velocity, vent_s + delay)
            acceleration = abs(original.omega) * float(np.hypot(predicted_velocity[0], predicted_velocity[1]))
            delay = float(original.depart_fn(original.m_ball * np.hypot(acceleration, 9.81)))
    return delay


def release_candidate(original, position, velocity, vent_s=0.):
    """Predict one valve/departure candidate using only the fitted state."""
    vent_s = float(vent_s)
    depart_s = departure_delay(original, position, velocity, vent_s)
    landing, predicted_velocity, predicted_position = original.predict_from(position, velocity, vent_s + depart_s)
    return dict(vent_s=vent_s, delay_s=vent_s + depart_s, departure_s=depart_s,
        landing=np.asarray(landing, dtype=float), velocity=np.asarray(predicted_velocity, dtype=float),
        position=np.asarray(predicted_position, dtype=float),
        speed_m_s=float(np.linalg.norm(predicted_velocity)), elevation_rad=float(np.arctan2(
            predicted_velocity[2], np.hypot(predicted_velocity[0], predicted_velocity[1]))))


def feature_vector(position, velocity, candidate, controller, observation, target):
    """Fixed online-observable feature order for fitting and later deployment."""
    target = np.asarray(target, dtype=float)
    q_max = float(np.degrees(np.abs(np.asarray(observation["q_meas"], dtype=float)).max()))
    age = max(0., float(observation["t_win"]) - float(controller.t_hand))
    drive = controller.drv
    target_angle = float(np.arctan2(target[1], target[0]))
    return np.asarray((
        position[0] - target[0], position[1] - target[1], position[2],
        velocity[0], velocity[1], velocity[2],
        candidate["landing"][0] - target[0], candidate["landing"][1] - target[1],
        candidate["velocity"][0], candidate["velocity"][1], candidate["velocity"][2],
        age, candidate["delay_s"], candidate["vent_s"], float(drive.st.get("r", 0.)),
        float(drive.F_max), float(np.sin(target_angle)), float(np.cos(target_angle)), q_max,
    ), dtype=float)


@dataclass(frozen=True)
class RidgeLandingCalibration:
    """Predict actual landing correction from online-observable release features."""
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    ridge_lambda: float

    def predict_delta(self, features):
        normalized = (np.asarray(features, dtype=float) - self.mean) / self.scale
        return np.r_[1., normalized] @ self.coefficients

    def predict_landing(self, raw_landing, features):
        return np.asarray(raw_landing, dtype=float)[:2] + self.predict_delta(features)

    def to_dict(self):
        return dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES), mean=self.mean.tolist(),
            scale=self.scale.tolist(), coefficients=self.coefficients.tolist(), ridge_lambda=self.ridge_lambda)

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema") != SCHEMA or tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("release calibration feature schema does not match V21")
        mean = np.asarray(payload["mean"], dtype=float)
        scale = np.asarray(payload["scale"], dtype=float)
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        if (mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape or
                coefficients.shape != (len(FEATURE_NAMES) + 1, 2) or
                not np.isfinite(np.r_[mean, scale, coefficients.ravel()]).all() or np.any(scale <= 0.)):
            raise ValueError("invalid release calibration parameters")
        return cls(mean, scale, coefficients, float(payload["ridge_lambda"]))


def fit_ridge(features, targets, ridge_lambda=.25):
    features, targets = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or targets.shape != (len(features), 2):
        raise ValueError("invalid calibration fit shapes")
    if len(features) < len(FEATURE_NAMES) + 5 or not np.isfinite(np.r_[features.ravel(), targets.ravel()]).all():
        raise ValueError("insufficient finite calibration rows")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-5)
    design = np.c_[np.ones(len(features)), (features - mean) / scale]
    penalty = np.eye(design.shape[1]) * float(ridge_lambda)
    penalty[0, 0] = 0.
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return RidgeLandingCalibration(mean, scale, coefficients, float(ridge_lambda))


def error_statistics(raw_landing, actual_landing, calibrated_landing):
    raw = np.linalg.norm(np.asarray(raw_landing) - np.asarray(actual_landing), axis=1)
    calibrated = np.linalg.norm(np.asarray(calibrated_landing) - np.asarray(actual_landing), axis=1)
    return dict(rows=int(len(raw)), raw_mean_m=float(raw.mean()), calibrated_mean_m=float(calibrated.mean()),
        raw_median_m=float(np.median(raw)), calibrated_median_m=float(np.median(calibrated)),
        raw_p90_m=float(np.quantile(raw, .9)), calibrated_p90_m=float(np.quantile(calibrated, .9)),
        relative_mean_improvement=float(1. - calibrated.mean() / max(raw.mean(), 1e-9)))
