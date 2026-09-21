"""Hierarchical PPO environment above the physical gripper and Koopman-MPPI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.controller import make_controller
from control.trajectory import inverse_kinematics
from digital_twin.train_tangent_compliance import TangentComplianceHead

from .catch_throw_plant import CatchThrowPlant
from .catch_throw_task import (
    DEFAULT_TARGET_OFFSET_WORLD_M,
    GRAVITY_WORLD_MPS2,
    CatchThrowTask,
    ballistic_state,
    desired_release_velocity,
    sample_task,
)
from .physical_scene import CatcherSpec
from .reference import HierarchicalReference, ReferenceConfig


def _moving_point_sphere_hit_time(position: np.ndarray, velocity: np.ndarray,
                                  center: np.ndarray, radius: float,
                                  horizon_s: float) -> float | None:
    offset = position - center
    speed_squared = float(velocity @ velocity)
    if float(offset @ offset) <= radius**2:
        return 0.0
    if speed_squared <= 1e-12:
        return None
    linear = 2.0 * float(offset @ velocity)
    constant = float(offset @ offset) - radius**2
    discriminant = linear**2 - 4.0 * speed_squared * constant
    if discriminant < 0.0:
        return None
    root = np.sqrt(discriminant)
    candidates = (
        (-linear - root) / (2.0 * speed_squared),
        (-linear + root) / (2.0 * speed_squared),
    )
    return next((float(time_s) for time_s in candidates
                 if 0.0 <= time_s <= horizon_s), None)


@dataclass(frozen=True)
class CatchThrowEnvConfig:
    policy_hz: float = 30.0
    plant_hz: float = 150.0
    max_episode_s: float = 2.8
    controller: str = "koopman_mppi"
    mppi_horizon: int = 8
    mppi_samples: int = 16
    mppi_tip_weight: float = 1.0
    mppi_tip_velocity_weight: float = .25
    mppi_joint_velocity_weight: float = .05
    mppi_compliance_weight: float = 1.0
    mppi_correction_blend: float = .20
    target_offset_world_m: tuple[float, float, float] = DEFAULT_TARGET_OFFSET_WORLD_M
    target_radius_m: float = .060
    force_limit_n: float = 350.0
    action_filter: float = .25
    capture_hold_s: float = .10
    grasp_mode: str = "auto_volume"
    auto_grasp_radius_m: float | None = None
    curriculum_stage: int = 1
    stage2_level: int = 1
    precharge_s: float = .40
    precharge_pressure_psi: float = 6.0
    seed: int = 20260914


class CatchThrowEnv:
    """PPO selects motion/compliance intent; it never writes q or pressure."""

    observation_size = 81
    action_size = HierarchicalReference.action_size

    def __init__(self, config: CatchThrowEnvConfig | None = None,
                 catcher: CatcherSpec | None = None):
        self.config = config or CatchThrowEnvConfig()
        ratio = self.config.plant_hz / self.config.policy_hz
        self.inner_steps = int(round(ratio))
        if abs(ratio - self.inner_steps) > 1e-9 or self.inner_steps <= 0:
            raise ValueError("policy_hz must divide plant_hz")
        if self.config.curriculum_stage not in range(1, 6):
            raise ValueError("curriculum_stage must be in [1, 5]")
        if self.config.stage2_level not in range(1, 4):
            raise ValueError("stage2_level must be in [1, 3]")
        if self.config.grasp_mode not in ("physical_dwell", "auto_volume"):
            raise ValueError("grasp_mode must be physical_dwell or auto_volume")
        if (self.config.auto_grasp_radius_m is not None
                and self.config.auto_grasp_radius_m <= 0.0):
            raise ValueError("auto_grasp_radius_m must be positive")
        self.policy_dt = 1 / self.config.policy_hz
        self.curriculum_stage = self.config.curriculum_stage
        self.stage2_level = self.config.stage2_level
        self.catcher = catcher or CatcherSpec()
        self.rng = np.random.default_rng(self.config.seed)
        self.plant = CatchThrowPlant(seed=self.config.seed, catcher=self.catcher)
        self.compliance_head = TangentComplianceHead()
        controller_kwargs = dict(dt=1 / self.config.plant_hz, seed=self.config.seed)
        if self.config.controller == "koopman_mppi":
            controller_kwargs.update(
                horizon=self.config.mppi_horizon, samples=self.config.mppi_samples,
                tip_weight=self.config.mppi_tip_weight,
                tip_velocity_weight=self.config.mppi_tip_velocity_weight,
                joint_velocity_weight=self.config.mppi_joint_velocity_weight,
                compliance_weight=self.config.mppi_compliance_weight,
                correction_blend=self.config.mppi_correction_blend,
                tangent_compliance=True, compliance_axes="xyz", common_noise_psi=.03,
            )
        self.controller = make_controller(self.config.controller, **controller_kwargs)
        self.done = True

    def close(self) -> None:
        self.plant.close()

    def set_curriculum(self, stage: int, stage2_level: int | None = None) -> None:
        if stage not in range(1, 6):
            raise ValueError("curriculum stage must be in [1, 5]")
        if stage2_level is not None and stage2_level not in range(1, 4):
            raise ValueError("stage2 level must be in [1, 3]")
        self.curriculum_stage = int(stage)
        if stage2_level is not None:
            self.stage2_level = int(stage2_level)

    def action_mask(self) -> np.ndarray:
        """Expose only the action dimensions relevant to the current curriculum."""
        active = 6 if self.curriculum_stage == 1 else 8 if self.curriculum_stage == 2 else 10
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[:active] = 1.0
        return mask

    def _sample_reachable_task(self) -> CatchThrowTask:
        home = self.plant.observe()["cup_position_m"]
        for _ in range(128):
            task = sample_task(
                self.rng, home, ball_mass_kg=self.catcher.ball_mass_kg,
                ball_radius_m=self.catcher.ball_radius_m,
                target_offset_world_m=np.asarray(self.config.target_offset_world_m, dtype=float),
                target_radius_m=self.config.target_radius_m,
                curriculum_stage=self.curriculum_stage,
                stage2_level=self.stage2_level,
            )
            _, info = inverse_kinematics(self.plant.world_to_base(task.intercept_world_m), return_info=True)
            if info["residual_m"] <= .003:
                return task
        raise RuntimeError("could not sample an IK-reachable physical interception")

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.plant.seed = int(seed)
        self.plant.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3))
        precharge = np.full(24, self.config.precharge_pressure_psi * 6894.757)
        for _ in range(int(round(self.config.precharge_s * self.config.plant_hz))):
            self.plant.step(precharge)
        self.home_cup_world_m = self.plant.observe()["cup_position_m"].copy()
        self.task = self._sample_reachable_task()
        self.plant.set_ball_mass(self.task.ball_mass_kg)
        self.state = self.plant.begin_episode(
            self.task.ball_position0_world_m, self.task.ball_velocity0_world_mps
        )
        self.reference = HierarchicalReference(
            self.task, start_world_m=self.home_cup_world_m,
            world_to_base=self.plant.world_to_base,
            vector_world_to_base=self.plant.vector_world_to_base,
            config=ReferenceConfig(
                policy_dt_s=self.policy_dt,
                capture_hold_s=self.config.capture_hold_s,
            ),
        )
        self.reference.reset()
        self.controller.reset(self.state["q"], self.state["p_pa"])
        self.captured = False
        self.released = False
        self.ever_contacted = False
        self.grasp_time_s: float | None = None
        self.release_time_s: float | None = None
        self.release_event_metrics: dict[str, float | list[float]] | None = None
        self.grasp_jump_m = 0.0
        self.maximum_cup_displacement_m = 0.0
        self.capture_trigger_mode: str | None = None
        self.physical_contact_at_grasp = False
        self.physical_compatible_at_grasp = False
        self.physical_compatible_contact_occurred = False
        self.peak_force_n = 0.0
        self.minimum_relative_speed_mps = np.inf
        self.last_contact_relative_speed_mps: float | None = None
        self.last_contact_velocity_alignment: float | None = None
        self.last_contact_radial_offset_m: float | None = None
        self.last_contact_axial_offset_m: float | None = None
        self.first_physical_contact_metrics: dict[str, float | bool | list[float]] | None = None
        self.best_pregrasp_contact_metrics: dict[str, float | bool | list[float]] | None = None
        self.best_pregrasp_contact_score = np.inf
        self.grasp_event_contact_metrics: dict[str, float | bool | list[float]] | None = None
        self.rendezvous_event_metrics: dict[str, float | bool | str | list[float]] | None = None
        self.minimum_contact_radial_offset_m = np.inf
        self.filtered_action = np.zeros(self.action_size)
        self.previous_action = np.zeros(self.action_size)
        self.frozen_phase_action: np.ndarray | None = None
        collision_geometry = self._collision_geometry(self.state)
        capture_volume = self._capture_volume_metrics(self.state)
        self.previous_capture_volume_error_m = capture_volume["error_m"]
        self.minimum_capture_volume_error_m = capture_volume["error_m"]
        self.previous_maximum_grasp_dwell_fraction = 0.0
        self.previous_entry_radial_m = collision_geometry["entry_radial_m"]
        self.minimum_entry_radial_m = collision_geometry["entry_radial_m"]
        self.minimum_predicted_contact_radial_m = collision_geometry[
            "predicted_contact_radial_m"
        ]
        self.precontact_entry_radial_m = collision_geometry["entry_radial_m"]
        self.precontact_predicted_contact_radial_m = collision_geometry[
            "predicted_contact_radial_m"
        ]
        self.contact_failure_radial = False
        self.contact_failure_axial = False
        self.contact_failure_speed = False
        self.contact_failure_alignment = False
        self.previous_landing_error_m = self._predicted_landing_error(self.state)
        self.previous_release_velocity_error_mps = self._release_velocity_error(self.state)
        rendezvous = self._rendezvous_metrics(self.state)
        self.previous_rendezvous_position_error_m = float(
            rendezvous["position_error_m"]
        )
        self.previous_rendezvous_velocity_error_mps = float(
            rendezvous["velocity_error_mps"]
        )
        self.previous_work_j = self.state["mechanical_work_abs_j"]
        self.previous_gripper_energy_j = self.state["gripper_energy_j"]
        incoming = ballistic_state(self.task, self.task.intercept_time_s)[1]
        self.incoming_speed_mps = float(np.linalg.norm(incoming))
        self.incoming_direction_world = incoming / max(self.incoming_speed_mps, 1e-8)
        relative_velocity = self.state["ball_velocity_mps"] - self.state["cup_velocity_mps"]
        self.previous_relative_speed_mps = float(np.linalg.norm(relative_velocity))
        self.previous_velocity_alignment = self._velocity_alignment(self.state)
        self.reference_velocity_saturation_sum = 0.0
        self.reference_velocity_saturation_count = 0
        self.reference_velocity_saturation_peak = 0.0
        self.done = False
        return self._observation(), self._info()

    def _ball_gripper_distance(self, state: dict) -> float:
        return float(np.linalg.norm(state["ball_position_m"] - state["cup_position_m"]))

    def _rendezvous_metrics(self, state: dict) -> dict[str, np.ndarray | float]:
        desired_position, desired_velocity = self.reference.nominal_catch_state(
            state["time_s"]
        )
        return {
            "desired_position_world_m": desired_position,
            "desired_velocity_world_mps": desired_velocity,
            "position_error_m": float(np.linalg.norm(
                state["cup_position_m"] - desired_position
            )),
            "velocity_error_mps": float(np.linalg.norm(
                state["cup_velocity_mps"] - desired_velocity
            )),
        }

    def _rendezvous_record(self, state: dict, *, event_metrics: dict | None = None,
                           ) -> dict[str, float | bool | str | list[float]]:
        metrics = self._rendezvous_metrics(state)
        event_metrics = event_metrics or {}
        cup_position = np.asarray(
            event_metrics.get("cup_position_world_m", state["cup_position_m"]),
            dtype=float,
        )
        cup_velocity = np.asarray(
            event_metrics.get("cup_velocity_world_mps", state["cup_velocity_mps"]),
            dtype=float,
        )
        ball_position = np.asarray(
            event_metrics.get("ball_position_world_m", state["ball_position_m"]),
            dtype=float,
        )
        ball_velocity = np.asarray(
            event_metrics.get("ball_velocity_world_mps", state["ball_velocity_mps"]),
            dtype=float,
        )
        desired_position = np.asarray(metrics["desired_position_world_m"])
        desired_velocity = np.asarray(metrics["desired_velocity_world_mps"])
        return {
            "time_s": float(state["time_s"]),
            "scheduled_time_s": float(self.task.intercept_time_s),
            "timing_error_s": float(state["time_s"] - self.task.intercept_time_s),
            "position_error_m": float(np.linalg.norm(cup_position - desired_position)),
            "velocity_error_mps": float(np.linalg.norm(cup_velocity - desired_velocity)),
            "desired_position_world_m": desired_position.tolist(),
            "desired_velocity_world_mps": desired_velocity.tolist(),
            "cup_position_world_m": cup_position.tolist(),
            "cup_velocity_world_mps": cup_velocity.tolist(),
            "ball_position_world_m": ball_position.tolist(),
            "ball_velocity_world_mps": ball_velocity.tolist(),
            "ball_cup_relative_speed_mps": float(np.linalg.norm(
                ball_velocity - cup_velocity
            )),
            "velocity_alignment": float(event_metrics.get(
                "velocity_alignment", self._velocity_alignment(state)
            )),
            "grasp_event": bool(state["grasp_event"]),
            "trigger": (
                "physical_contact" if event_metrics else "scheduled_intercept"
            ),
        }

    def _predicted_finger_contact(self, position_local: np.ndarray,
                                  velocity_local: np.ndarray,
                                  horizon_s: float,
                                  acceleration_local: np.ndarray | None = None,
                                  ) -> tuple[float | None, np.ndarray | None, str | None]:
        if acceleration_local is not None and np.linalg.norm(acceleration_local) > 1e-10:
            acceleration_local = np.asarray(acceleration_local, dtype=float)
            segment_s = .01
            for start_s in np.arange(0.0, horizon_s, segment_s):
                duration_s = min(segment_s, horizon_s - start_s)
                segment_position = (
                    position_local + velocity_local * start_s
                    + .5 * acceleration_local * start_s**2
                )
                midpoint_velocity = velocity_local + acceleration_local * (
                    start_s + .5 * duration_s
                )
                hit_time, hit_point, hit_kind = self._predicted_finger_contact(
                    segment_position, midpoint_velocity, duration_s
                )
                if hit_time is not None:
                    return float(start_s + hit_time), hit_point, hit_kind
            return None, None, None

        z_front = self.catcher.palm_z_m - self.catcher.finger_length_m
        combined_radius = self.catcher.ball_radius_m + self.catcher.finger_radius_m
        candidates: list[tuple[float, np.ndarray, str]] = []
        for index in range(3):
            angle = 2.0 * np.pi * index / 3.0
            center_xy = self.catcher.grip_radius_m * np.array([
                np.cos(angle), np.sin(angle)
            ])
            offset_xy = position_local[:2] - center_xy
            speed_xy_squared = float(velocity_local[:2] @ velocity_local[:2])
            if speed_xy_squared > 1e-12:
                linear = 2.0 * float(offset_xy @ velocity_local[:2])
                constant = float(offset_xy @ offset_xy) - combined_radius**2
                discriminant = linear**2 - 4.0 * speed_xy_squared * constant
                if discriminant >= 0.0:
                    root = np.sqrt(discriminant)
                    for time_s in (
                        (-linear - root) / (2.0 * speed_xy_squared),
                        (-linear + root) / (2.0 * speed_xy_squared),
                    ):
                        z_at_contact = position_local[2] + velocity_local[2] * time_s
                        if (0.0 <= time_s <= horizon_s
                                and z_front <= z_at_contact <= self.catcher.palm_z_m):
                            point = position_local + velocity_local * time_s
                            candidates.append((float(time_s), point, f"finger_{index}_side"))
            for endpoint_z, endpoint_name in (
                (z_front, "tip"), (self.catcher.palm_z_m, "back")
            ):
                center = np.array([center_xy[0], center_xy[1], endpoint_z])
                time_s = _moving_point_sphere_hit_time(
                    position_local, velocity_local, center, combined_radius, horizon_s
                )
                if time_s is not None:
                    point = position_local + velocity_local * time_s
                    candidates.append((time_s, point, f"finger_{index}_{endpoint_name}"))

        palm_front_z = self.catcher.palm_z_m - .004 - self.catcher.ball_radius_m
        if velocity_local[2] > 1e-8:
            palm_time = (palm_front_z - position_local[2]) / velocity_local[2]
            if 0.0 <= palm_time <= horizon_s:
                point = position_local + velocity_local * palm_time
                if np.linalg.norm(point[:2]) <= self.catcher.grip_radius_m + .008:
                    candidates.append((float(palm_time), point, "palm_front"))
        if not candidates:
            return None, None, None
        return min(candidates, key=lambda candidate: candidate[0])

    def _collision_geometry(self, state: dict) -> dict[str, np.ndarray | float | bool | str | None]:
        relative_position = state["ball_position_m"] - state["cup_position_m"]
        relative_velocity = state["ball_velocity_mps"] - state["cup_velocity_mps"]
        rotation_world_from_cup = state["cup_rotation"]
        position_local = rotation_world_from_cup.T @ relative_position
        velocity_local = rotation_world_from_cup.T @ relative_velocity
        acceleration_local = rotation_world_from_cup.T @ GRAVITY_WORLD_MPS2
        horizon_s = .55
        z_front = self.catcher.palm_z_m - self.catcher.finger_length_m
        entry_z = z_front - self.catcher.ball_radius_m - self.catcher.finger_radius_m
        if position_local[2] >= entry_z:
            raw_entry_time = 0.0
        else:
            constant = position_local[2] - entry_z
            linear = velocity_local[2]
            quadratic = .5 * acceleration_local[2]
            roots: list[float] = []
            if abs(quadratic) <= 1e-10:
                if linear > 1e-8:
                    roots.append(-constant / linear)
            else:
                discriminant = linear**2 - 4.0 * quadratic * constant
                if discriminant >= 0.0:
                    root = np.sqrt(discriminant)
                    roots.extend((
                        (-linear - root) / (2.0 * quadratic),
                        (-linear + root) / (2.0 * quadratic),
                    ))
            valid_roots = [
                value for value in roots
                if 0.0 <= value <= horizon_s
                and velocity_local[2] + acceleration_local[2] * value > 0.0
            ]
            raw_entry_time = min(valid_roots) if valid_roots else horizon_s
        time_to_entry = float(np.clip(raw_entry_time, 0.0, horizon_s))
        entry_local = (
            position_local + velocity_local * time_to_entry
            + .5 * acceleration_local * time_to_entry**2
        )
        if time_to_entry <= .20:
            contact_horizon = min(horizon_s, max(.08, time_to_entry + .06))
            contact_time, contact_local, contact_kind = self._predicted_finger_contact(
                position_local, velocity_local, contact_horizon, acceleration_local
            )
        else:
            contact_time, contact_local, contact_kind = None, None, None
        predicted_contact = contact_local is not None
        if contact_local is None:
            contact_local = entry_local.copy()
            contact_time = horizon_s
        return {
            "entry_local_m": entry_local,
            "time_to_entry_s": time_to_entry,
            "entry_radial_m": float(np.linalg.norm(entry_local[:2])),
            "predicted_contact": predicted_contact,
            "predicted_contact_local_m": contact_local,
            "time_to_predicted_contact_s": float(contact_time),
            "predicted_contact_radial_m": float(np.linalg.norm(contact_local[:2])),
            "predicted_contact_kind": contact_kind,
            "current_local_m": position_local,
        }

    def _auto_grasp_radial_limit(self) -> float:
        if self.config.auto_grasp_radius_m is not None:
            return float(self.config.auto_grasp_radius_m)
        if self.curriculum_stage == 2:
            return (.025, .022, .020)[self.stage2_level - 1]
        return (.028, .022, .020, .020, .018)[self.curriculum_stage - 1]

    def _capture_volume_metrics(self, state: dict) -> dict[str, float | bool]:
        local = state["cup_rotation"].T @ (
            state["ball_position_m"] - state["cup_position_m"]
        )
        radial = float(np.linalg.norm(local[:2]))
        axial = float(local[2])
        radial_limit = self._auto_grasp_radial_limit()
        axial_min, axial_max = self.catcher.grasp_axial_bounds_m
        radial_excess = max(0.0, radial - radial_limit)
        axial_excess = max(axial_min - axial, axial - axial_max, 0.0)
        return {
            "radial_offset_m": radial,
            "axial_offset_m": axial,
            "radial_limit_m": radial_limit,
            "radial_excess_m": radial_excess,
            "axial_excess_m": axial_excess,
            "error_m": float(np.hypot(radial_excess, axial_excess)),
            "inside": radial_excess == 0.0 and axial_excess == 0.0,
        }

    def _grasp_metrics(self, state: dict) -> dict[str, float | bool]:
        local = state["cup_rotation"].T @ (state["ball_position_m"] - state["cup_position_m"])
        relative_velocity = state["ball_velocity_mps"] - state["cup_velocity_mps"]
        relative_speed = float(np.linalg.norm(relative_velocity))
        alignment = self._velocity_alignment(state)
        radial_limit, speed_limit, alignment_threshold = self._grasp_thresholds()
        axial_min, axial_max = self.catcher.grasp_axial_bounds_m
        radial = float(np.linalg.norm(local[:2]))
        axial = float(local[2])
        inside = bool(radial <= radial_limit
                      and axial_min <= local[2] <= axial_max)
        physical_contact = state["contact"].contact_quanta > 0
        compatible = bool(
            physical_contact and inside
            and relative_speed <= speed_limit
            and alignment >= alignment_threshold
        )
        return {
            "inside": inside, "physical_contact": physical_contact,
            "relative_speed_mps": relative_speed, "velocity_alignment": alignment,
            "radial_offset_m": radial, "axial_offset_m": axial,
            "auto_volume_inside": bool(
                radial <= self._auto_grasp_radial_limit()
                and axial_min <= axial <= axial_max
            ),
            "compatible": compatible,
        }

    def _velocity_alignment(self, state: dict) -> float:
        relative_speed = float(np.linalg.norm(state["ball_velocity_mps"] - state["cup_velocity_mps"]))
        ball_speed = float(np.linalg.norm(state["ball_velocity_mps"]))
        cup_speed = float(np.linalg.norm(state["cup_velocity_mps"]))
        if ball_speed < .05 or cup_speed < .05:
            return 1.0 if relative_speed < .05 else 0.0
        return float(np.dot(state["ball_velocity_mps"], state["cup_velocity_mps"])
                     / (ball_speed * cup_speed))

    def _contact_gate_diagnostics(self, metrics: dict[str, float]) -> dict[str, float | bool]:
        radial_limit, speed_limit, alignment_limit = self._grasp_thresholds()
        quality_radial, quality_speed, quality_alignment = self._soft_grasp_quality_targets()
        axial_min, axial_max = self.catcher.grasp_axial_bounds_m
        radial = float(metrics["radial_offset_m"])
        axial = float(metrics["axial_offset_m"])
        relative_speed = float(metrics["relative_speed_mps"])
        alignment = float(metrics["velocity_alignment"])
        radial_excess = max(0.0, radial - radial_limit)
        axial_excess = max(axial_min - axial, axial - axial_max, 0.0)
        speed_excess = max(0.0, relative_speed - speed_limit)
        alignment_shortfall = max(0.0, alignment_limit - alignment)
        radial_failure = radial_excess > 0.0
        axial_failure = axial_excess > 0.0
        speed_failure = speed_excess > 0.0
        alignment_failure = alignment_shortfall > 0.0
        score = (
            (radial_excess / .010) ** 2
            + (axial_excess / .010) ** 2
            + (speed_excess / .10) ** 2
            + (alignment_shortfall / .20) ** 2
        )
        quality_score = (
            (max(0.0, radial - quality_radial) / .010) ** 2
            + (max(0.0, relative_speed - quality_speed) / .10) ** 2
            + (max(0.0, quality_alignment - alignment) / .20) ** 2
        )
        return {
            "radial_limit_m": radial_limit,
            "speed_limit_mps": speed_limit,
            "alignment_limit": alignment_limit,
            "quality_radial_target_m": quality_radial,
            "quality_speed_target_mps": quality_speed,
            "quality_alignment_target": quality_alignment,
            "axial_min_m": axial_min,
            "axial_max_m": axial_max,
            "radial_failure": radial_failure,
            "axial_failure": axial_failure,
            "speed_failure": speed_failure,
            "alignment_failure": alignment_failure,
            "compatible": not (
                radial_failure or axial_failure or speed_failure or alignment_failure
            ),
            "gate_score": float(score),
            "quality_score": float(quality_score),
        }

    def _contact_record(self, metrics: dict, *,
                        desired_velocity_world_mps: np.ndarray | None = None,
                        actual_velocity_world_mps: np.ndarray | None = None,
                        ball_velocity_world_mps: np.ndarray | None = None,
                        ) -> dict[str, float | bool | str | list[float]]:
        diagnostics = self._contact_gate_diagnostics(metrics)
        physical_contact = bool(metrics.get("physical_contact", True))
        record: dict[str, float | bool | str | list[float]] = {
            "relative_speed_mps": float(metrics["relative_speed_mps"]),
            "velocity_alignment": float(metrics["velocity_alignment"]),
            "radial_offset_m": float(metrics["radial_offset_m"]),
            "axial_offset_m": float(metrics["axial_offset_m"]),
            **diagnostics,
            "physical_contact": physical_contact,
            "physical_compatible": bool(metrics.get(
                "physical_compatible",
                physical_contact and diagnostics["compatible"],
            )),
            "auto_volume_inside": bool(metrics.get("auto_volume_inside", False)),
        }
        if metrics.get("trigger_mode") is not None:
            record["trigger_mode"] = str(metrics["trigger_mode"])
        if metrics.get("physical_dwell_fraction") is not None:
            record["physical_dwell_fraction"] = float(
                metrics["physical_dwell_fraction"]
            )
        if desired_velocity_world_mps is not None and actual_velocity_world_mps is not None:
            desired = np.asarray(desired_velocity_world_mps, dtype=float)
            actual = np.asarray(actual_velocity_world_mps, dtype=float)
            tracking_error = actual - desired
            record.update({
                "desired_cup_velocity_world_mps": desired.tolist(),
                "actual_cup_velocity_world_mps": actual.tolist(),
                "mppi_tracking_velocity_error_mps": float(np.linalg.norm(tracking_error)),
                "cup_velocity_error_mps": float(np.linalg.norm(tracking_error)),
            })
            if ball_velocity_world_mps is not None:
                ball = np.asarray(ball_velocity_world_mps, dtype=float)
                planning_error = desired - ball
                record.update({
                    "ball_velocity_world_mps": ball.tolist(),
                    "reference_planning_velocity_error_mps": float(
                        np.linalg.norm(planning_error)
                    ),
                })
        return record

    def _grasp_thresholds(self) -> tuple[float, float, float]:
        speed_scale = (1.80, 1.40, 1.35, 1.35, 1.00)[self.curriculum_stage - 1]
        alignment = (.35, .45, .50, .50, self.catcher.grasp_alignment_cos)[self.curriculum_stage - 1]
        radial_margin = (.040, .020, .018, .018, 0.0)[self.curriculum_stage - 1]
        return (
            self.catcher.grasp_radial_limit_m + radial_margin,
            self.catcher.grasp_relative_speed_mps * speed_scale,
            alignment,
        )

    def _soft_grasp_quality_targets(self) -> tuple[float, float, float]:
        """Reward targets for soft-catch quality, stricter than early grasp gates."""
        if self.curriculum_stage == 1:
            return .035, .70, .45
        if self.curriculum_stage == 2:
            alignment = (.60, .65, .68)[self.stage2_level - 1]
            return .025, .55, alignment
        if self.curriculum_stage == 3:
            return .024, .50, .68
        if self.curriculum_stage == 4:
            return .022, .48, .72
        return .017, self.catcher.grasp_relative_speed_mps, self.catcher.grasp_alignment_cos

    def _first_contact_speed_target(self) -> float:
        """Achievable impact target before the compliant cup dissipates momentum."""
        if self.curriculum_stage == 1:
            return 1.70
        if self.curriculum_stage == 2:
            return (1.65, 1.55, 1.45)[self.stage2_level - 1]
        return (1.40, 1.35, 1.30)[min(self.curriculum_stage - 3, 2)]

    def _shaping_radial_scale(self) -> float:
        if self.curriculum_stage == 1:
            return .026
        if self.curriculum_stage == 2:
            return (.019, .020, .022)[self.stage2_level - 1]
        return .024 if self.curriculum_stage < 5 else .018

    def _action_authority(self) -> np.ndarray:
        """Scale residual-policy groups without suppressing all controls equally."""
        if self.curriculum_stage == 2:
            position, velocity, compliance = (
                (.50, .70, .55),
                (.70, .85, .75),
                (.90, 1.00, .95),
            )[self.stage2_level - 1]
            return np.array(
                [position] * 3 + [velocity] * 3 + [compliance] * 2 + [0.0] * 2
            )
        groups = (
            (.30, .50, 0.0, 0.0),
            (.50, .70, .55, 0.0),
            (.80, .90, .85, .65),
            (.90, 1.00, 1.00, .85),
            (1.00, 1.00, 1.00, 1.00),
        )[self.curriculum_stage - 1]
        position, velocity, compliance, throw = groups
        return np.array(
            [position] * 3 + [velocity] * 3 + [compliance] * 2 + [throw] * 2
        )

    def _stable_capture(self) -> bool:
        return bool(
            self.captured
            and self.grasp_time_s is not None
            and self.state["time_s"] - self.grasp_time_s >= self.config.capture_hold_s
        )

    def _phase(self, state: dict) -> int:
        if self.released:
            return 3
        if self.captured:
            return 2
        if state["contact"].contact_quanta:
            return 1
        return 0

    def _compliance_features(self, state: dict) -> np.ndarray:
        incoming_world = ballistic_state(self.task, self.task.intercept_time_s)[1]
        direction = self.plant.vector_world_to_base(incoming_world)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < .8 else np.array([0.0, 1.0, 0.0])
        transverse_1 = np.cross(direction, helper)
        transverse_1 /= max(float(np.linalg.norm(transverse_1)), 1e-8)
        transverse_2 = np.cross(direction, transverse_1)
        try:
            compliance = self.compliance_head.predict(state["q"], state["p_pa"])
            projected = np.array([
                direction @ compliance @ direction,
                transverse_1 @ compliance @ transverse_1,
                transverse_2 @ compliance @ transverse_2,
            ])
            if np.isfinite(projected).all():
                return np.clip(projected, 0.0, .25)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass
        return np.full(3, .06)

    def _predicted_landing_error(self, state: dict) -> float:
        flight_time = self.reference.release_flight_s if hasattr(self, "reference") else .45
        landing = (state["ball_position_m"] + state["ball_velocity_mps"] * flight_time
                   + .5 * GRAVITY_WORLD_MPS2 * flight_time**2)
        distance = float(np.linalg.norm(landing - self.task.target_center_world_m))
        return max(0.0, distance - self.task.target_radius_m)

    def _release_velocity_target(self, state: dict) -> np.ndarray:
        flight_time = self.reference.release_flight_s if hasattr(self, "reference") else .55
        return desired_release_velocity(
            state["ball_position_m"], self.task.target_center_world_m, flight_time
        )

    def _release_velocity_error(self, state: dict) -> float:
        target_velocity = self._release_velocity_target(state)
        return float(np.linalg.norm(state["ball_velocity_mps"] - target_velocity))

    def _predicted_landing_point(self, state: dict) -> np.ndarray:
        flight_time = self.reference.release_flight_s if hasattr(self, "reference") else .55
        return (
            state["ball_position_m"]
            + state["ball_velocity_mps"] * flight_time
            + .5 * GRAVITY_WORLD_MPS2 * flight_time**2
        )

    def _release_record(self, state: dict) -> dict[str, float | list[float]]:
        target_velocity = self._release_velocity_target(state)
        landing = self._predicted_landing_point(state)
        landing_distance = float(np.linalg.norm(landing - self.task.target_center_world_m))
        return {
            "time_s": float(state["time_s"]),
            "position_world_m": state["ball_position_m"].tolist(),
            "actual_velocity_world_mps": state["ball_velocity_mps"].tolist(),
            "desired_velocity_world_mps": target_velocity.tolist(),
            "velocity_error_mps": float(np.linalg.norm(
                state["ball_velocity_mps"] - target_velocity
            )),
            "predicted_landing_world_m": landing.tolist(),
            "landing_distance_m": landing_distance,
            "landing_error_m": max(0.0, landing_distance - self.task.target_radius_m),
            "flight_time_s": float(
                self.reference.release_flight_s if hasattr(self, "reference") else .55
            ),
        }

    def _observation(self) -> np.ndarray:
        state = self.state
        geometry = self._collision_geometry(state)
        phase = np.zeros(4, dtype=float)
        phase[self._phase(state)] = 1.0
        vector = np.concatenate((
            state["q"] / .35, state["qdot"] / 2.0, state["p_pa"] / (30 * 6894.757),
            (state["cup_position_m"] - self.home_cup_world_m) / .5,
            state["cup_velocity_mps"] / 2.0,
            (state["ball_position_m"] - state["cup_position_m"]) / .8,
            state["ball_velocity_mps"] / 3.0,
            np.asarray(geometry["entry_local_m"]) / .20,
            np.array([geometry["time_to_entry_s"] / .55]),
            np.array([geometry["entry_radial_m"] / .08]),
            np.array([geometry["predicted_contact_radial_m"] / .08]),
            (self.task.target_center_world_m - state["cup_position_m"]) / .8,
            np.array([self.task.ball_mass_kg / .025]), self._compliance_features(state) / .16,
            np.clip(state["contact"].force_world_n / self.config.force_limit_n, -2.0, 2.0),
            phase, np.array([(self.config.max_episode_s - state["time_s"]) / self.config.max_episode_s]),
        ))
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise RuntimeError("nonfinite or malformed hierarchical PPO observation")
        return vector.astype(np.float32)

    def _info(self) -> dict:
        state = self.state
        metrics = self._grasp_metrics(state)
        geometry = self._collision_geometry(state)
        capture_volume = self._capture_volume_metrics(state)
        rendezvous = self._rendezvous_metrics(state)
        grasp_dwell_fraction = float(np.clip(
            state["grasp_candidate_s"] / max(state["grasp_dwell_s"], 1e-9),
            0.0, 1.0,
        ))
        maximum_grasp_dwell_fraction = float(np.clip(
            state["maximum_grasp_candidate_s"]
            / max(state["grasp_dwell_s"], 1e-9),
            0.0, 1.0,
        ))
        return {
            "time_s": state["time_s"], "curriculum_stage": self.curriculum_stage,
            "stage2_level": self.stage2_level,
            "grasp_mode": self.config.grasp_mode,
            "auto_grasp_radius_m": self._auto_grasp_radial_limit(),
            "task_intercept_world_m": self.task.intercept_world_m.copy(),
            "task_intercept_time_s": self.task.intercept_time_s,
            "task_incoming_velocity_world_mps": ballistic_state(
                self.task, self.task.intercept_time_s
            )[1],
            "target_center_world_m": self.task.target_center_world_m.copy(),
            "target_radius_m": self.task.target_radius_m,
            "ball_position_m": state["ball_position_m"].copy(),
            "ball_velocity_mps": state["ball_velocity_mps"].copy(),
            "ball_momentum_kg_mps": state["ball_momentum_kg_mps"].copy(),
            "ball_kinetic_energy_j": state["ball_kinetic_energy_j"],
            "cup_position_m": state["cup_position_m"].copy(),
            "cup_velocity_mps": state["cup_velocity_mps"].copy(),
            "contact_force_n": state["contact"].force_norm_n,
            "contact_impulse_ns": state["contact"].impulse_ns,
            "window_contact_impulse_ns": state["contact"].window_impulse_ns,
            "external_torque_nm": state["contact"].generalized_torque_nm.copy(),
            "constraint_torque_nm": state["constraint_torque_nm"].copy(),
            "captured": self.captured, "grasp_constraint_active": state["grasped"],
            "capture_trigger_mode": self.capture_trigger_mode,
            "auto_capture_occurred": self.capture_trigger_mode == "auto_volume",
            "physical_capture_occurred": self.capture_trigger_mode == "physical_dwell",
            "physical_contact_at_grasp": self.physical_contact_at_grasp,
            "physical_compatible_at_grasp": self.physical_compatible_at_grasp,
            "stable_capture": self._stable_capture(),
            "capture_hold_elapsed_s": (
                0.0 if self.grasp_time_s is None
                else max(0.0, state["time_s"] - self.grasp_time_s)
            ),
            "released": self.released, "ever_contacted": self.ever_contacted,
            "relative_speed_mps": metrics["relative_speed_mps"],
            "velocity_alignment": metrics["velocity_alignment"],
            "last_contact_relative_speed_mps": self.last_contact_relative_speed_mps,
            "last_contact_velocity_alignment": self.last_contact_velocity_alignment,
            "last_contact_radial_offset_m": self.last_contact_radial_offset_m,
            "last_contact_axial_offset_m": self.last_contact_axial_offset_m,
            "first_physical_contact_metrics": self.first_physical_contact_metrics,
            "best_compatible_pregrasp_contact_metrics": self.best_pregrasp_contact_metrics,
            "grasp_event_metrics": self.grasp_event_contact_metrics,
            "rendezvous_event_metrics": self.rendezvous_event_metrics,
            "release_event_metrics": self.release_event_metrics,
            "rendezvous_position_error_m": rendezvous["position_error_m"],
            "rendezvous_velocity_error_mps": rendezvous["velocity_error_mps"],
            "minimum_contact_radial_offset_m": (
                None if not np.isfinite(self.minimum_contact_radial_offset_m)
                else self.minimum_contact_radial_offset_m
            ),
            "minimum_contact_relative_speed_mps": (
                None if not np.isfinite(self.minimum_relative_speed_mps)
                else self.minimum_relative_speed_mps
            ),
            "grasp_jump_m": self.grasp_jump_m,
            "maximum_cup_displacement_m": self.maximum_cup_displacement_m,
            "capture_volume_inside": capture_volume["inside"],
            "capture_volume_radial_offset_m": capture_volume["radial_offset_m"],
            "capture_volume_axial_offset_m": capture_volume["axial_offset_m"],
            "capture_volume_error_m": capture_volume["error_m"],
            "minimum_capture_volume_error_m": self.minimum_capture_volume_error_m,
            "grasp_dwell_fraction": grasp_dwell_fraction,
            "maximum_grasp_dwell_fraction": maximum_grasp_dwell_fraction,
            "predicted_entry_local_m": np.asarray(geometry["entry_local_m"]).copy(),
            "predicted_entry_radial_m": geometry["entry_radial_m"],
            "time_to_entry_s": geometry["time_to_entry_s"],
            "predicted_contact": geometry["predicted_contact"],
            "predicted_contact_local_m": np.asarray(
                geometry["predicted_contact_local_m"]
            ).copy(),
            "predicted_contact_radial_m": geometry["predicted_contact_radial_m"],
            "time_to_predicted_contact_s": geometry["time_to_predicted_contact_s"],
            "predicted_contact_kind": geometry["predicted_contact_kind"],
            "minimum_predicted_entry_radial_m": self.minimum_entry_radial_m,
            "minimum_predicted_contact_radial_m": self.minimum_predicted_contact_radial_m,
            "precontact_entry_radial_m": self.precontact_entry_radial_m,
            "precontact_predicted_contact_radial_m": (
                self.precontact_predicted_contact_radial_m
            ),
            "contact_failure_radial": self.contact_failure_radial,
            "contact_failure_axial": self.contact_failure_axial,
            "contact_failure_speed": self.contact_failure_speed,
            "contact_failure_alignment": self.contact_failure_alignment,
            "predicted_landing_error_m": self._predicted_landing_error(state),
            "release_velocity_error_mps": self._release_velocity_error(state),
            "target_distance_m": float(np.linalg.norm(state["ball_position_m"] - self.task.target_center_world_m)),
            "mechanical_work_abs_j": state["mechanical_work_abs_j"],
            "gripper_energy_j": state["gripper_energy_j"],
            "energy_proxy_j": state["mechanical_work_abs_j"] + state["gripper_energy_j"],
            "reference_velocity_saturation_fraction": (
                self.reference_velocity_saturation_sum
                / max(self.reference_velocity_saturation_count, 1)
            ),
            "reference_velocity_saturation_peak": self.reference_velocity_saturation_peak,
            "physical_contact_occurred": self.ever_contacted,
            "compatible_contact_occurred": self.physical_compatible_contact_occurred,
            "unsafe_force": self.peak_force_n > self.config.force_limit_n,
            "physical_contact": True,
        }

    def _curriculum_success(self, target_distance: float) -> bool:
        if self.curriculum_stage <= 2:
            return self._stable_capture()
        if self.curriculum_stage == 3:
            return self.released
        return self.released and target_distance <= self.task.target_radius_m

    def _success_terminal_bonus(self) -> float:
        if self.curriculum_stage != 2:
            return (12.0, 12.0, 20.0, 40.0, 40.0)[self.curriculum_stage - 1]
        if self.grasp_event_contact_metrics is None:
            return 20.0
        _, _, alignment_gate = self._grasp_thresholds()
        quality_radial, quality_speed, quality_alignment = self._soft_grasp_quality_targets()
        relative_speed = float(self.grasp_event_contact_metrics["relative_speed_mps"])
        radial = float(self.grasp_event_contact_metrics["radial_offset_m"])
        alignment = float(self.grasp_event_contact_metrics["velocity_alignment"])
        alignment_quality = float(np.clip(
            (alignment - alignment_gate) / max(quality_alignment - alignment_gate, 1e-6),
            0.0,
            1.0,
        ))
        speed_quality = float(np.exp(-(relative_speed / quality_speed) ** 2))
        radial_quality = float(np.exp(-(radial / quality_radial) ** 2))
        quality = .35 * alignment_quality + .35 * speed_quality + .30 * radial_quality
        # Capture coverage is the primary Stage-2 objective. Kinematic quality
        # remains visible in the bounded bonus but can no longer erase success.
        return float(16.0 + 4.0 * quality)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("episode ended; call reset")
        raw_action = np.asarray(action, dtype=float)
        if raw_action.shape != (self.action_size,) or not np.isfinite(raw_action).all():
            raise ValueError("action must contain 10 finite values")
        raw_action = np.clip(raw_action, -1.0, 1.0) * self.action_mask()
        alpha = self.config.action_filter
        self.filtered_action = (1.0 - alpha) * self.filtered_action + alpha * raw_action
        # Early curriculum stages deliberately expose only a fraction of the
        # final action authority, so initial Gaussian exploration cannot erase
        # the known feasible velocity-matching reference before PPO sees a
        # successful physical grasp.
        effective_action = self._action_authority() * self.filtered_action
        if self.frozen_phase_action is not None:
            effective_action[8:10] = self.frozen_phase_action[8:10]
        self.reference.set_action(
            effective_action, time_s=self.state["time_s"], q_now=self.state["q"],
            ball_velocity_world_mps=self.state["ball_velocity_mps"],
        )

        reward = -.001 * float(np.sum(np.square(raw_action - self.previous_action)))
        self.previous_action = raw_action
        grasp_event = False
        release_event = False
        for _ in range(self.inner_steps):
            reference = self.reference.low_level_reference(
                q_now=self.state["q"], qdot_now=self.state["qdot"],
                force_world_n=self.state["contact"].force_world_n, dt=self.plant.dt,
                horizon=getattr(self.controller, "horizon", 1),
                time_s=self.state["time_s"],
            )
            velocity_limit = self.reference.config.joint_velocity_limit_rad_s
            saturation = float(np.mean(
                np.abs(reference["future_qd"]) >= velocity_limit - 1e-6
            ))
            self.reference_velocity_saturation_sum += saturation
            self.reference_velocity_saturation_count += 1
            self.reference_velocity_saturation_peak = max(
                self.reference_velocity_saturation_peak, saturation
            )
            reference["external_torque_nm"] = (
                self.state["constraint_torque_nm"] if self.state["grasped"]
                else self.state["contact"].generalized_torque_nm
            )
            radial_limit, speed_limit, alignment_limit = self._grasp_thresholds()
            if not self.captured:
                self.plant.configure_grasp_gate(
                    radial_limit_m=radial_limit,
                    axial_bounds_m=self.catcher.grasp_axial_bounds_m,
                    relative_speed_limit_mps=speed_limit,
                    alignment_cosine=alignment_limit,
                    mode=self.config.grasp_mode,
                    auto_radial_limit_m=self._auto_grasp_radial_limit(),
                )
            else:
                self.plant.disable_grasp_gate()
            geometry_before_step = (
                self._collision_geometry(self.state) if not self.captured else None
            )
            pressure = self.controller.command(
                self.state["q"], self.state["qdot"], self.state["p_pa"], **reference
            )
            self.state = self.plant.step(pressure)
            self.maximum_cup_displacement_m = max(
                self.maximum_cup_displacement_m,
                float(np.linalg.norm(
                    self.state["cup_position_m"] - self.home_cup_world_m
                )),
            )
            contact = self.state["contact"]
            self.peak_force_n = max(self.peak_force_n, contact.force_norm_n)
            metrics = self._grasp_metrics(self.state)
            relative_speed = float(metrics["relative_speed_mps"])
            geometry = self._collision_geometry(self.state)
            entry_radial = float(geometry["entry_radial_m"])
            predicted_contact_radial = float(geometry["predicted_contact_radial_m"])
            capture_volume = self._capture_volume_metrics(self.state)
            capture_volume_error = float(capture_volume["error_m"])
            self.minimum_capture_volume_error_m = min(
                self.minimum_capture_volume_error_m, capture_volume_error
            )
            maximum_dwell_fraction = float(np.clip(
                self.state["maximum_grasp_candidate_s"]
                / max(self.state["grasp_dwell_s"], 1e-9),
                0.0, 1.0,
            ))
            dwell_progress = max(
                0.0,
                maximum_dwell_fraction
                - self.previous_maximum_grasp_dwell_fraction,
            )
            self.previous_maximum_grasp_dwell_fraction = max(
                self.previous_maximum_grasp_dwell_fraction,
                maximum_dwell_fraction,
            )
            if not self.captured:
                volume_progress = float(np.clip(
                    self.previous_capture_volume_error_m - capture_volume_error,
                    -.02, .02,
                ))
                reward += 8.0 * volume_progress
                reward += .030 * np.exp(-(capture_volume_error / .020) ** 2)
                reward += .75 * dwell_progress
                self.previous_capture_volume_error_m = capture_volume_error
            self.minimum_entry_radial_m = min(
                self.minimum_entry_radial_m, entry_radial
            )
            self.minimum_predicted_contact_radial_m = min(
                self.minimum_predicted_contact_radial_m, predicted_contact_radial
            )
            contact_metrics = self.state["contact_event_metrics"]
            if self.state["contact_event"] and contact_metrics is not None:
                relative_speed = float(contact_metrics["relative_speed_mps"])
                contact_radial = float(contact_metrics["radial_offset_m"])
                _, desired_velocity_world = self.reference.desired_catch_state(
                    self.state["time_s"]
                )
                actual_velocity_world = np.asarray(
                    contact_metrics["cup_velocity_world_mps"], dtype=float
                )
                ball_velocity_world = np.asarray(
                    contact_metrics["ball_velocity_world_mps"], dtype=float
                )
                contact_record = self._contact_record(
                    contact_metrics,
                    desired_velocity_world_mps=desired_velocity_world,
                    actual_velocity_world_mps=actual_velocity_world,
                    ball_velocity_world_mps=ball_velocity_world,
                )
                if self.first_physical_contact_metrics is None:
                    self.first_physical_contact_metrics = dict(contact_record)
                self.physical_compatible_contact_occurred |= bool(
                    contact_record["physical_compatible"]
                )
                gate_score = float(contact_record["gate_score"])
                quality_score = float(contact_record["quality_score"])
                compatible_rank = 0 if bool(contact_record["compatible"]) else 1
                best_key = (
                    compatible_rank,
                    quality_score if compatible_rank == 0 else gate_score,
                    relative_speed,
                    contact_radial,
                    -float(contact_record["velocity_alignment"]),
                )
                current_best_key = (
                    1,
                    self.best_pregrasp_contact_score,
                    np.inf,
                    np.inf,
                    np.inf,
                )
                if self.best_pregrasp_contact_metrics is not None:
                    current_rank = (
                        0 if bool(self.best_pregrasp_contact_metrics["compatible"]) else 1
                    )
                    current_best_key = (
                        current_rank,
                        self.best_pregrasp_contact_score,
                        float(self.best_pregrasp_contact_metrics["relative_speed_mps"]),
                        float(self.best_pregrasp_contact_metrics["radial_offset_m"]),
                        -float(self.best_pregrasp_contact_metrics["velocity_alignment"]),
                    )
                if best_key < current_best_key:
                    self.best_pregrasp_contact_score = (
                        quality_score if compatible_rank == 0 else gate_score
                    )
                    self.best_pregrasp_contact_metrics = dict(contact_record)
                self.minimum_relative_speed_mps = min(
                    self.minimum_relative_speed_mps, relative_speed
                )
                self.minimum_contact_radial_offset_m = min(
                    self.minimum_contact_radial_offset_m, contact_radial
                )
                self.last_contact_relative_speed_mps = relative_speed
                self.last_contact_velocity_alignment = float(
                    contact_metrics["velocity_alignment"]
                )
                self.last_contact_radial_offset_m = contact_radial
                contact_axial = float(contact_metrics["axial_offset_m"])
                contact_alignment = float(contact_metrics["velocity_alignment"])
                self.last_contact_axial_offset_m = contact_axial
                axial_min, axial_max = self.catcher.grasp_axial_bounds_m
                radial_failure = bool(contact_record["radial_failure"])
                axial_failure = bool(contact_record["axial_failure"])
                speed_failure = bool(contact_record["speed_failure"])
                alignment_failure = bool(contact_record["alignment_failure"])
                self.contact_failure_radial |= radial_failure
                self.contact_failure_axial |= axial_failure
                self.contact_failure_speed |= speed_failure
                self.contact_failure_alignment |= alignment_failure
                radial_excess = max(0.0, contact_radial - radial_limit)
                axial_excess = max(axial_min - contact_axial, contact_axial - axial_max, 0.0)
                speed_excess = max(0.0, relative_speed - speed_limit)
                alignment_shortfall = max(0.0, alignment_limit - contact_alignment)
                quality_radial, quality_speed, quality_alignment = self._soft_grasp_quality_targets()
                quality_radial_excess = max(0.0, contact_radial - quality_radial)
                quality_speed_excess = max(0.0, relative_speed - quality_speed)
                quality_alignment_shortfall = max(0.0, quality_alignment - contact_alignment)
                contact_gate_penalty = (
                    1.20 * min((radial_excess / .010) ** 2, 4.0)
                    + .60 * min((axial_excess / .010) ** 2, 4.0)
                    + .50 * min((speed_excess / .10) ** 2, 4.0)
                    + .50 * min((alignment_shortfall / .20) ** 2, 4.0)
                )
                repeat_contact_scale = 1.0 if not self.ever_contacted else .15
                reward -= repeat_contact_scale * min(contact_gate_penalty, 4.0)
                contact_quality_penalty = (
                    .25 * min((quality_radial_excess / .010) ** 2, 4.0)
                    + .25 * min((quality_speed_excess / .10) ** 2, 4.0)
                    + .45 * min((quality_alignment_shortfall / .20) ** 2, 4.0)
                )
                reward -= repeat_contact_scale * min(contact_quality_penalty, 1.2)
                if not self.ever_contacted:
                    self.reference.on_contact(
                        self.state["time_s"],
                        np.asarray(contact_metrics["cup_position_world_m"], dtype=float),
                        np.asarray(contact_metrics["cup_velocity_world_mps"], dtype=float),
                    )
                    if self.rendezvous_event_metrics is None:
                        self.rendezvous_event_metrics = self._rendezvous_record(
                            self.state, event_metrics=contact_metrics
                        )
                    if geometry_before_step is not None:
                        self.precontact_entry_radial_m = float(
                            geometry_before_step["entry_radial_m"]
                        )
                        self.precontact_predicted_contact_radial_m = float(
                            geometry_before_step["predicted_contact_radial_m"]
                        )
                    radial_scale = self._shaping_radial_scale()
                    radial_quality = float(np.exp(-(contact_radial / radial_scale) ** 2))
                    following_quality = float(np.clip(
                        (self.incoming_speed_mps - relative_speed)
                        / max(self.incoming_speed_mps, 1e-6),
                        0.0,
                        1.0,
                    ))
                    first_contact_speed_excess = max(
                        0.0, relative_speed - self._first_contact_speed_target()
                    )
                    reward += (
                        .50
                        + 1.50 * radial_quality
                        + 1.50 * following_quality
                        + .75 * np.clip(contact_alignment, 0.0, 1.0)
                    )
                    reward -= .80 * min(
                        (first_contact_speed_excess / .25) ** 2, 4.0
                    )
                    self.ever_contacted = True
            elif geometry_before_step is not None and not self.ever_contacted:
                self.precontact_entry_radial_m = float(
                    geometry_before_step["entry_radial_m"]
                )
                self.precontact_predicted_contact_radial_m = float(
                    geometry_before_step["predicted_contact_radial_m"]
                )

            if not self.captured:
                radial_scale = self._shaping_radial_scale()
                entry_ratio = entry_radial / radial_scale
                entry_progress = float(np.clip(
                    self.previous_entry_radial_m - entry_radial, -.02, .02
                ))
                reward += 10.0 * entry_progress
                reward += .040 * np.exp(-entry_ratio**2)
                reward -= .006 * min(entry_ratio**2, 9.0)
                if not self.ever_contacted:
                    rendezvous = self._rendezvous_metrics(self.state)
                    rendezvous_position_error = float(rendezvous["position_error_m"])
                    rendezvous_velocity_error = float(rendezvous["velocity_error_mps"])
                    position_progress = float(np.clip(
                        self.previous_rendezvous_position_error_m
                        - rendezvous_position_error,
                        -.015, .015,
                    ))
                    velocity_progress = float(np.clip(
                        self.previous_rendezvous_velocity_error_mps
                        - rendezvous_velocity_error,
                        -.10, .10,
                    ))
                    phase = float(np.clip(
                        self.state["time_s"] / max(self.task.intercept_time_s, 1e-6),
                        0.0, 1.0,
                    ))
                    time_gate = .25 + .75 * phase * phase * (3.0 - 2.0 * phase)
                    stage1_gain = 1.0 if self.curriculum_stage == 1 else .35
                    reward += stage1_gain * (
                        3.0 * position_progress
                        + .30 * velocity_progress
                        + .015 * np.exp(-(rendezvous_position_error / .035) ** 2)
                        + .025 * time_gate * np.exp(-(rendezvous_velocity_error / .55) ** 2)
                        - .006 * min((rendezvous_position_error / .060) ** 2, 6.0)
                        - .010 * time_gate
                        * min((rendezvous_velocity_error / .80) ** 2, 6.0)
                    )
                    self.previous_rendezvous_position_error_m = rendezvous_position_error
                    self.previous_rendezvous_velocity_error_mps = rendezvous_velocity_error
                predicted_excess = max(0.0, predicted_contact_radial - radial_limit)
                if geometry["predicted_contact"]:
                    reward += .015 * np.exp(-(predicted_contact_radial / radial_limit) ** 4)
                    reward -= .020 * min((predicted_excess / .010) ** 2, 9.0)
                    event_time = float(geometry["time_to_predicted_contact_s"])
                else:
                    event_time = float(geometry["time_to_entry_s"])
                geometry_gate = np.exp(-(predicted_excess / .010) ** 2)
                near_contact = float(np.exp(-event_time / .120) * geometry_gate)
                relative_velocity = self.state["ball_velocity_mps"] - self.state["cup_velocity_mps"]
                parallel_relative_velocity = float(
                    np.dot(relative_velocity, self.incoming_direction_world)
                )
                transverse_relative_velocity = float(np.linalg.norm(
                    relative_velocity
                    - parallel_relative_velocity * self.incoming_direction_world
                ))
                alignment = float(metrics["velocity_alignment"])
                relative_speed_improvement = float(np.clip(
                    self.previous_relative_speed_mps - relative_speed, -.03, .03
                ))
                alignment_improvement = float(np.clip(
                    alignment - self.previous_velocity_alignment, -.03, .03
                ))
                quality_radial, quality_speed, quality_alignment = self._soft_grasp_quality_targets()
                alignment_range = max(quality_alignment - alignment_limit, 1e-6)
                alignment_quality = float(np.clip(
                    (alignment - alignment_limit) / alignment_range, 0.0, 1.0
                ))
                quality_alignment_shortfall = max(0.0, quality_alignment - alignment)
                reward += .025 * near_contact * np.exp(-relative_speed / .35)
                reward += .25 * near_contact * relative_speed_improvement
                reward += .08 * near_contact * alignment_improvement
                reward += .060 * near_contact * alignment_quality
                reward += .040 * near_contact * np.clip(
                    1.0 - relative_speed / max(quality_speed, 1e-6), 0.0, 1.0
                )
                reward -= .015 * near_contact * min((relative_speed / speed_limit) ** 2, 4.0)
                reward -= .025 * near_contact * min(
                    (quality_alignment_shortfall / .20) ** 2, 4.0
                )
                reward -= .006 * near_contact * min(
                    (transverse_relative_velocity / speed_limit) ** 2, 4.0
                )
                reward -= .004 * near_contact * min(
                    (abs(parallel_relative_velocity) / speed_limit) ** 2, 4.0
                )
                self.previous_entry_radial_m = entry_radial
                self.previous_relative_speed_mps = relative_speed
                self.previous_velocity_alignment = alignment
                if (self.rendezvous_event_metrics is None
                        and (self.state["time_s"] >= self.task.intercept_time_s
                             or self.state["grasp_event"])):
                    self.rendezvous_event_metrics = self._rendezvous_record(
                        self.state,
                        event_metrics=(
                            self.state["grasp_event_metrics"]
                            if self.state["grasp_event"] else None
                        ),
                    )
                if self.state["grasp_event"]:
                    event_metrics = self.state["grasp_event_metrics"]
                    if event_metrics is None:
                        raise RuntimeError("grasp event is missing 1 kHz kinematic metrics")
                    relative_speed = float(event_metrics["relative_speed_mps"])
                    self.grasp_event_contact_metrics = self._contact_record(
                        event_metrics,
                        desired_velocity_world_mps=self.reference.desired_catch_state(
                            self.state["time_s"]
                        )[1],
                        actual_velocity_world_mps=np.asarray(
                            event_metrics["cup_velocity_world_mps"], dtype=float
                        ),
                        ball_velocity_world_mps=np.asarray(
                            event_metrics["ball_velocity_world_mps"], dtype=float
                        ),
                    )
                    self.grasp_event_contact_metrics[
                        "cup_displacement_from_home_m"
                    ] = float(np.linalg.norm(
                        np.asarray(event_metrics["cup_position_world_m"], dtype=float)
                        - self.home_cup_world_m
                    ))
                    self.capture_trigger_mode = str(event_metrics.get(
                        "trigger_mode", self.config.grasp_mode
                    ))
                    self.physical_contact_at_grasp = bool(
                        event_metrics.get("physical_contact", False)
                    )
                    self.physical_compatible_at_grasp = bool(
                        event_metrics.get("physical_compatible", False)
                    )
                    if self.physical_contact_at_grasp:
                        self.last_contact_relative_speed_mps = relative_speed
                        self.last_contact_velocity_alignment = float(
                            event_metrics["velocity_alignment"]
                        )
                        self.last_contact_radial_offset_m = float(
                            event_metrics["radial_offset_m"]
                        )
                        self.last_contact_axial_offset_m = float(
                            event_metrics["axial_offset_m"]
                        )
                        self.minimum_relative_speed_mps = min(
                            self.minimum_relative_speed_mps, relative_speed
                        )
                        self.minimum_contact_radial_offset_m = min(
                            self.minimum_contact_radial_offset_m,
                            float(event_metrics["radial_offset_m"]),
                        )
                    self.grasp_jump_m = float(self.state["grasp_jump_m"])
                    self.captured = True
                    grasp_event = True
                    self.grasp_time_s = self.state["time_s"]
                    self.frozen_phase_action = effective_action.copy()
                    self.reference.on_grasp(
                        self.state["time_s"], self.state["ball_position_m"],
                        self.state["ball_velocity_mps"],
                        throw_enabled=self.curriculum_stage >= 3,
                    )
                    self.reference.set_action(
                        effective_action, time_s=self.state["time_s"], q_now=self.state["q"],
                        ball_velocity_world_mps=self.state["ball_velocity_mps"],
                    )
                    contact_radial = float(event_metrics["radial_offset_m"])
                    contact_alignment = float(event_metrics["velocity_alignment"])
                    quality_radial, quality_speed, quality_alignment = self._soft_grasp_quality_targets()
                    alignment_range = max(quality_alignment - alignment_limit, 1e-6)
                    alignment_quality = float(np.clip(
                        (contact_alignment - alignment_limit) / alignment_range,
                        0.0, 1.0,
                    ))
                    speed_quality = float(np.exp(-(relative_speed / quality_speed) ** 2))
                    radial_quality = float(np.exp(-(contact_radial / quality_radial) ** 2))
                    quality_alignment_shortfall = max(0.0, quality_alignment - contact_alignment)
                    reward += (
                        6.0
                        + 2.0 * np.exp(-(contact_radial / radial_scale) ** 2)
                        + 3.0 * alignment_quality
                        + .75 * speed_quality
                        + .75 * radial_quality
                        - .5 * min((relative_speed / speed_limit) ** 2, 4.0)
                        - 1.25 * min((quality_alignment_shortfall / .20) ** 2, 4.0)
                    )
            else:
                if self.curriculum_stage <= 2:
                    reward += .05
                else:
                    landing_error = self._predicted_landing_error(self.state)
                    release_velocity_error = self._release_velocity_error(self.state)
                    landing_progress = float(np.clip(
                        self.previous_landing_error_m - landing_error, -.05, .05
                    ))
                    velocity_progress = float(np.clip(
                        self.previous_release_velocity_error_mps - release_velocity_error,
                        -.05, .05,
                    ))
                    reward += .05 + 3.0 * landing_progress + .80 * velocity_progress
                    reward += .15 * np.exp(-(landing_error / .35) ** 2)
                    reward += .08 * np.exp(-(release_velocity_error / .75) ** 2)
                    reward -= .050 * min(landing_error**2, 4.0)
                    reward -= .020 * min(release_velocity_error**2, 4.0)
                    forward_speed = float(np.dot(
                        self.state["cup_velocity_mps"], self.incoming_direction_world
                    ))
                    minimum_follow_speed = .15 * self.incoming_speed_mps
                    reward += .04 * np.clip(
                        forward_speed / max(self.incoming_speed_mps, 1e-6), 0.0, 1.0
                    )
                    reward -= 5.0 * max(0.0, minimum_follow_speed - forward_speed) ** 2
                    self.previous_landing_error_m = landing_error
                    self.previous_release_velocity_error_mps = release_velocity_error
                    if (not self.released and self.grasp_time_s is not None
                            and self.state["time_s"] - self.grasp_time_s
                            >= self.reference.follow_through_s):
                        self.plant.release_grasp()
                        self.state = self.plant.observe()
                        self.released = True
                        release_event = True
                        self.release_time_s = self.state["time_s"]
                        self.reference.on_release(self.state["time_s"])
                        self.release_event_metrics = self._release_record(self.state)
                        release_landing_error = float(
                            self.release_event_metrics["landing_error_m"]
                        )
                        release_velocity_error = float(
                            self.release_event_metrics["velocity_error_mps"]
                        )
                        reward += 7.0
                        reward += 5.0 * np.exp(-(release_landing_error / .25) ** 2)
                        reward += 2.0 * np.exp(-(release_velocity_error / .55) ** 2)
                        reward -= .75 * min(release_landing_error, 2.0)
                        reward -= .25 * min(release_velocity_error, 3.0)

            reward -= 2.0 * contact.window_impulse_ns
            excess_force = max(0.0, contact.force_norm_n - 45.0)
            reward -= 2e-5 * excess_force**2
            work_delta = self.state["mechanical_work_abs_j"] - self.previous_work_j
            gripper_delta = self.state["gripper_energy_j"] - self.previous_gripper_energy_j
            self.previous_work_j = self.state["mechanical_work_abs_j"]
            self.previous_gripper_energy_j = self.state["gripper_energy_j"]
            energy_weight = (0.0, 0.0, 0.0, .0015, .006)[self.curriculum_stage - 1]
            reward -= energy_weight * max(0.0, work_delta + gripper_delta)

        target_distance = float(np.linalg.norm(
            self.state["ball_position_m"] - self.task.target_center_world_m
        ))
        if self.released:
            reward -= .10 * max(0.0, target_distance - self.task.target_radius_m)
        success = self._curriculum_success(target_distance)
        unsafe = self.peak_force_n > self.config.force_limit_n
        missed_intercept = (
            self.state["time_s"] > self.task.intercept_time_s + .20
            and not self.captured
        )
        release_timed_out = (
            self.released and self.release_time_s is not None
            and self.state["time_s"] > self.release_time_s + self.reference.release_flight_s + .25
        )
        truncated = bool(
            self.state["time_s"] >= self.config.max_episode_s
            or missed_intercept or release_timed_out or unsafe
        )
        terminated = bool(success)
        if terminated:
            reward += self._success_terminal_bonus()
        elif truncated:
            reward -= 8.0 if unsafe else 4.0
        self.done = terminated or truncated
        info = self._info()
        info["grasp_event"] = grasp_event
        info["release_event"] = release_event
        return self._observation(), float(reward), terminated, truncated, info
