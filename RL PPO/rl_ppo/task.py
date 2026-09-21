"""Ballistic catch benchmark definitions for the current fitted UMArm frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.trajectory import inverse_kinematics


GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class CatchBenchmark:
    """Reference-derived ball/capture constants and local training envelope."""

    ball_mass_kg: float = .025
    ball_radius_m: float = .018
    capture_radius_m: float = .055
    maximum_relative_speed_mps: float = 2.5
    max_episode_s: float = .80
    # These are offsets from the fitted twin's home plate centre, not the
    # reference glove-world coordinates. Each candidate is IK checked.
    intercept_offset_x_m: tuple[float, float] = (.015, .080)
    intercept_offset_y_m: tuple[float, float] = (-.040, .040)
    intercept_offset_z_m: tuple[float, float] = (.010, .050)
    modes: tuple[str, ...] = ("lob_arc", "flat_fast", "mixed")
    mode_probabilities: tuple[float, ...] = (.40, .30, .30)
    max_resample_attempts: int = 128


@dataclass(frozen=True)
class BallisticCase:
    position0_m: np.ndarray
    velocity0_mps: np.ndarray
    intercept_m: np.ndarray
    incoming_velocity_mps: np.ndarray
    intercept_time_s: float
    mode: str


def ball_state(case: BallisticCase, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic free-flight state under gravity in fitted-arm base axes."""
    t = max(0.0, float(time_s))
    gravity = np.array([0.0, 0.0, -GRAVITY_MPS2])
    position = case.position0_m + case.velocity0_mps * t + .5 * gravity * t * t
    velocity = case.velocity0_mps + gravity * t
    return position, velocity


def _incoming_velocity(rng: np.random.Generator, mode: str) -> tuple[np.ndarray, float]:
    if mode == "lob_arc":
        return np.array([rng.uniform(-.12, .12), rng.uniform(.45, .85), rng.uniform(-1.30, -.65)]), rng.uniform(.42, .58)
    if mode == "flat_fast":
        return np.array([rng.uniform(-.25, .25), rng.uniform(.95, 1.45), rng.uniform(-.35, .05)]), rng.uniform(.30, .46)
    if mode == "mixed":
        return np.array([rng.uniform(-.18, .18), rng.uniform(.65, 1.15), rng.uniform(-.85, -.20)]), rng.uniform(.36, .52)
    raise ValueError(f"unknown ball mode {mode!r}")


def sample_case(rng: np.random.Generator, home_tip_m: np.ndarray) -> BallisticCase:
    """Sample a reference-style incoming ball with an IK-reachable intercept."""
    benchmark = CatchBenchmark()
    home_tip_m = np.asarray(home_tip_m, dtype=float)
    probabilities = np.asarray(benchmark.mode_probabilities, dtype=float)
    for _ in range(benchmark.max_resample_attempts):
        mode = str(rng.choice(benchmark.modes, p=probabilities))
        offset = np.array([
            rng.uniform(*benchmark.intercept_offset_x_m),
            rng.uniform(*benchmark.intercept_offset_y_m),
            rng.uniform(*benchmark.intercept_offset_z_m),
        ])
        intercept = home_tip_m + offset
        _, info = inverse_kinematics(intercept, return_info=True)
        if info["residual_m"] > 1e-4:
            continue
        incoming, intercept_time_s = _incoming_velocity(rng, mode)
        gravity = np.array([0.0, 0.0, -GRAVITY_MPS2])
        velocity0 = incoming - gravity * intercept_time_s
        position0 = intercept - incoming * intercept_time_s + .5 * gravity * intercept_time_s**2
        return BallisticCase(position0, velocity0, intercept, incoming, float(intercept_time_s), mode)
    raise RuntimeError("could not sample an IK-reachable catch benchmark case")
