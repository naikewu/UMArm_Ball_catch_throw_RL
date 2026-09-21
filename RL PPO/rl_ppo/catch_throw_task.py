"""Physical ball launch and fixed-target contract for catch-and-throw PPO."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRAVITY_WORLD_MPS2 = np.array([0.0, 0.0, -9.81])
DEFAULT_TARGET_OFFSET_WORLD_M = (0.0, 0.80, 0.0)
FINAL_BALL_MASS_RANGE_KG = (0.020, 0.035)


@dataclass(frozen=True)
class CatchThrowTask:
    ball_mass_kg: float
    ball_radius_m: float
    intercept_world_m: np.ndarray
    intercept_time_s: float
    ball_position0_world_m: np.ndarray
    ball_velocity0_world_mps: np.ndarray
    target_center_world_m: np.ndarray
    target_radius_m: float
    curriculum_stage: int = 5
    stage2_level: int = 3


def ballistic_state(task: CatchThrowTask, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    time_s = max(0.0, float(time_s))
    position = task.ball_position0_world_m + task.ball_velocity0_world_mps * time_s + .5 * GRAVITY_WORLD_MPS2 * time_s**2
    return position, task.ball_velocity0_world_mps + GRAVITY_WORLD_MPS2 * time_s


def desired_release_velocity(release_position_world_m: np.ndarray, target_world_m: np.ndarray,
                             flight_time_s: float) -> np.ndarray:
    """Ballistic velocity that reaches the fixed target point after ``flight_time_s``."""
    if flight_time_s <= 0:
        raise ValueError("flight time must be positive")
    return (np.asarray(target_world_m) - np.asarray(release_position_world_m)
            - .5 * GRAVITY_WORLD_MPS2 * flight_time_s**2) / flight_time_s


def _annular_intercept_offset(rng: np.random.Generator, *,
                              min_radius_m: float, max_radius_m: float,
                              vertical_offset_m: float) -> np.ndarray:
    """Sample an offset that forces the arm to move instead of waiting at home."""
    angle = float(rng.uniform(-np.pi, np.pi))
    radius = float(rng.uniform(min_radius_m, max_radius_m))
    return np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        rng.uniform(-vertical_offset_m, vertical_offset_m),
    ])


def sample_task(rng: np.random.Generator, home_cup_world_m: np.ndarray, *,
                ball_mass_kg: float = .025, ball_radius_m: float = .018,
                final_ball_mass_range_kg: tuple[float, float] = FINAL_BALL_MASS_RANGE_KG,
                target_offset_world_m: np.ndarray | None = None,
                target_radius_m: float = .080,
                curriculum_stage: int = 5,
                stage2_level: int = 3) -> CatchThrowTask:
    """Sample a side-to-side physical catch-and-throw task.

    The launcher is on the negative world-y side and remains well outside the
    arm workspace in every curriculum stage.  Flight time is solved from a
    sampled impact slope so the ball enters along the tilted catcher aperture,
    rather than acquiring an untrackable near-vertical velocity on a long arc.
    The caller still validates the sampled intercept with IK.
    """
    if curriculum_stage not in range(1, 6):
        raise ValueError("curriculum_stage must be in [1, 5]")
    if stage2_level not in range(1, 4):
        raise ValueError("stage2_level must be in [1, 3]")
    if (len(final_ball_mass_range_kg) != 2
            or final_ball_mass_range_kg[0] <= 0
            or final_ball_mass_range_kg[0] > final_ball_mass_range_kg[1]):
        raise ValueError("invalid final-stage ball mass range")
    home = np.asarray(home_cup_world_m, dtype=float)
    if curriculum_stage == 2:
        intercept_radius = ((.025, .050), (.040, .075), (.055, .100))[stage2_level - 1]
        vertical_offset = (.006, .010, .014)[stage2_level - 1]
        lateral_speed = (.045, .070, .100)[stage2_level - 1]
        launch_distance = ((.78, .90), (.84, 1.00), (.90, 1.10))[stage2_level - 1]
        launch_height = ((-.015, .015), (-.020, .020), (-.025, .025))[stage2_level - 1]
        incoming_slope = ((.70, .80), (.65, .85), (.60, .90))[stage2_level - 1]
    else:
        intercept_radius = (
            (.010, .025), (.025, .055), (.060, .115), (.080, .150), (.100, .180)
        )[curriculum_stage - 1]
        vertical_offset = (.005, .008, .016, .022, .028)[curriculum_stage - 1]
        lateral_speed = (.020, .055, .110, .155, .220)[curriculum_stage - 1]
        launch_distance = (
            (.85, 1.05), (.82, .95), (.90, 1.10), (1.00, 1.25), (1.10, 1.40)
        )[curriculum_stage - 1]
        launch_height = (
            (-.010, .010), (-.015, .015), (-.025, .025), (-.030, .030), (-.035, .035)
        )[curriculum_stage - 1]
        incoming_slope = (
            (.72, .78), (.65, .85), (.60, .90), (.55, .95), (.50, 1.00)
        )[curriculum_stage - 1]
    intercept = home + _annular_intercept_offset(
        rng,
        min_radius_m=intercept_radius[0],
        max_radius_m=intercept_radius[1],
        vertical_offset_m=vertical_offset,
    )
    distance = float(rng.uniform(*launch_distance))
    launch_z = float(home[2] + rng.uniform(*launch_height))
    vertical_displacement = float(intercept[2] - launch_z)
    impact_slope = float(rng.uniform(*incoming_slope))
    # For a ballistic path, v_z(T) / v_y(T) = -impact_slope when
    # T^2 = 2 * (impact_slope * distance + dz) / g.
    flight_term = impact_slope * distance + vertical_displacement
    if flight_term <= 0.0:
        raise RuntimeError("sampled far launch has no positive ballistic flight time")
    intercept_time = float(np.sqrt(2.0 * flight_term / -GRAVITY_WORLD_MPS2[2]))
    forward = distance / intercept_time
    lateral_velocity = float(rng.uniform(-lateral_speed, lateral_speed))
    position0 = np.array([
        intercept[0] - lateral_velocity * intercept_time,
        intercept[1] - distance,
        launch_z,
    ])
    displacement = intercept - position0
    velocity0 = (
        displacement - .5 * GRAVITY_WORLD_MPS2 * intercept_time**2
    ) / intercept_time
    # The default is a broad fixed target region.  It is deliberately a task
    # input, not a direction invented by the policy.
    offset = np.array(DEFAULT_TARGET_OFFSET_WORLD_M) if target_offset_world_m is None else np.asarray(target_offset_world_m, dtype=float)
    curriculum_radius = (.18, .14, .11, .08, target_radius_m)[curriculum_stage - 1]
    sampled_mass = (
        rng.uniform(*final_ball_mass_range_kg)
        if curriculum_stage == 5 else ball_mass_kg
    )
    return CatchThrowTask(
        ball_mass_kg=float(sampled_mass), ball_radius_m=float(ball_radius_m),
        intercept_world_m=intercept, intercept_time_s=intercept_time,
        ball_position0_world_m=position0, ball_velocity0_world_mps=velocity0,
        target_center_world_m=home + offset,
        target_radius_m=float(max(target_radius_m, curriculum_radius)),
        curriculum_stage=curriculum_stage,
        stage2_level=stage2_level,
    )
