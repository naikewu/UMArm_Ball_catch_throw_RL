"""Current-twin ballistic catch environment with a safe pressure-rate action."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.controller import PA_PER_PSI, pair_indices, project_pressures
from control.sim_env import ControlEnv
from control.trajectory import tip_position

from .task import BallisticCase, CatchBenchmark, ball_state, sample_case


@dataclass(frozen=True)
class CatchEnvConfig:
    policy_hz: float = 30.0
    plant_hz: float = 150.0
    randomized_twin: bool = True
    max_mean_rate_psi_s: float = 8.0
    max_diff_rate_psi_s: float = 24.0
    initial_pair_mean_psi: float = 6.0


class UMArmBallCatchEnv:
    """A fresh RL interface over the existing pressure-safe fitted twin.

    A physical ball/cup is not yet part of the fitted XML. The benchmark uses
    a virtual capture region around the final plate centre; policy observation
    still receives only noisy arm feedback plus the ball state.
    """

    observation_size = 55
    action_size = 24

    def __init__(self, config: CatchEnvConfig | None = None):
        self.config = config or CatchEnvConfig()
        ratio = self.config.plant_hz / self.config.policy_hz
        self.inner_steps = int(round(ratio))
        if self.inner_steps <= 0 or abs(ratio - self.inner_steps) > 1e-9:
            raise ValueError("policy_hz must divide the 150 Hz plant rate")
        self.policy_dt = 1.0 / self.config.policy_hz
        self.pairs = pair_indices()
        self.benchmark = CatchBenchmark()
        self.plant: ControlEnv | None = None
        self.rng = np.random.default_rng()

    def close(self) -> None:
        if self.plant is not None:
            self.plant.close()
            self.plant = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.close()
        self.rng = np.random.default_rng(seed)
        plant_seed = int(self.rng.integers(0, 2**31 - 1))
        self.plant = ControlEnv(seed=plant_seed, randomize=self.config.randomized_twin)
        self.observation = self.plant.reset()
        self.home_tip_m = tip_position(np.zeros(12))
        self.case = sample_case(self.rng, self.home_tip_m)
        self.elapsed_s = 0.0
        self.steps = 0
        self.done = False
        self.mean_psi = np.full(12, self.config.initial_pair_mean_psi)
        self.diff_psi = np.zeros(12)
        self.last_targets_pa = self._pressures_from_pair_state()
        self.previous_distance_m = self._distance_to_ball()
        return self._observation_vector(), self._info(caught=False)

    def _require_ready(self) -> None:
        if self.plant is None or not hasattr(self, "case"):
            raise RuntimeError("call reset before step")

    def _pressures_from_pair_state(self) -> np.ndarray:
        self.mean_psi = np.clip(self.mean_psi, 0.0, 15.0)
        self.diff_psi = np.clip(self.diff_psi, -2.0 * self.mean_psi, 2.0 * self.mean_psi)
        pressure_psi = np.zeros(24)
        pressure_psi[self.pairs[:, 0]] = self.mean_psi + .5 * self.diff_psi
        pressure_psi[self.pairs[:, 1]] = self.mean_psi - .5 * self.diff_psi
        return project_pressures(pressure_psi * PA_PER_PSI)

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError("action must be 24 finite values")
        action = np.clip(action, -1.0, 1.0)
        self.diff_psi += action[:12] * self.config.max_diff_rate_psi_s * self.policy_dt
        self.mean_psi += action[12:] * self.config.max_mean_rate_psi_s * self.policy_dt
        return self._pressures_from_pair_state()

    def _ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        return ball_state(self.case, self.elapsed_s)

    def _tip_truth_m(self) -> np.ndarray:
        return tip_position(self.observation["q_true"])

    def _distance_to_ball(self) -> float:
        ball_position, _ = self._ball_state()
        return float(np.linalg.norm(ball_position - self._tip_truth_m()))

    def _observation_vector(self) -> np.ndarray:
        self._require_ready()
        ball_position, ball_velocity = self._ball_state()
        relative_position = ball_position - tip_position(self.observation["q"])
        remaining = max(0.0, self.benchmark.max_episode_s - self.elapsed_s)
        vector = np.concatenate((
            self.observation["q"] / .35,
            self.observation["qdot"] / 2.0,
            self.observation["p_pa"] / (30.0 * PA_PER_PSI),
            relative_position / .50,
            ball_velocity / 3.0,
            np.array([remaining / self.benchmark.max_episode_s]),
        ))
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise RuntimeError("nonfinite catch observation")
        return vector.astype(np.float32)

    def _info(self, *, caught: bool) -> dict:
        ball_position, ball_velocity = self._ball_state()
        tip_position_m = self._tip_truth_m()
        return {
            "time_s": self.elapsed_s,
            "mode": self.case.mode,
            "ball_position_m": ball_position.copy(),
            "ball_velocity_mps": ball_velocity.copy(),
            "tip_position_m": tip_position_m.copy(),
            "distance_m": float(np.linalg.norm(ball_position - tip_position_m)),
            "intercept_time_s": self.case.intercept_time_s,
            "intercept_m": self.case.intercept_m.copy(),
            "caught": bool(caught),
            "virtual_catcher": True,
            "max_force_n": 0.0 if self.plant is None else self.plant.max_force_n,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._require_ready()
        if self.done:
            raise RuntimeError("episode ended; call reset")
        targets = self._apply_action(action)
        for _ in range(self.inner_steps):
            self.observation = self.plant.step(targets)
            self.elapsed_s += self.plant.dt
        self.steps += 1
        ball_position, ball_velocity = self._ball_state()
        tip_truth = self._tip_truth_m()
        distance_m = float(np.linalg.norm(ball_position - tip_truth))
        tip_velocity = tip_position(self.observation["q_true"] + self.observation["qdot_true"] * 1e-3)
        tip_velocity = (tip_velocity - tip_truth) / 1e-3
        relative_speed = float(np.linalg.norm(ball_velocity - tip_velocity))
        descending = bool(ball_velocity[2] < 0.0)
        caught = descending and distance_m <= self.benchmark.capture_radius_m and relative_speed <= self.benchmark.maximum_relative_speed_mps
        progress = self.previous_distance_m - distance_m
        shaping = .08 * np.exp(-distance_m / .06) + 15.0 * progress
        effort = .002 * float(np.mean(np.square(np.asarray(action, dtype=float))))
        reward = shaping - effort
        terminated = bool(caught)
        # In the fitted-base coordinates a valid upward launch can begin below
        # the plate's numeric z coordinate. Only treat that as a missed ball
        # after the prescribed intercept time has passed.
        missed_below_plate = (
            self.elapsed_s >= self.case.intercept_time_s
            and ball_position[2] < self.home_tip_m[2] - .15
        )
        truncated = bool(self.elapsed_s >= self.benchmark.max_episode_s or missed_below_plate)
        if caught:
            reward += 20.0
        elif truncated:
            reward -= 2.0
        self.previous_distance_m = distance_m
        self.done = terminated or truncated
        info = self._info(caught=caught)
        info["relative_speed_mps"] = relative_speed
        return self._observation_vector(), float(reward), terminated, truncated, info
