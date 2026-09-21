"""Convert compact PPO intent into continuous tip/compliance references."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.trajectory import tip_jacobian, tip_position

from .catch_throw_task import CatchThrowTask, ballistic_state, desired_release_velocity
from .catch_throw_task import GRAVITY_WORLD_MPS2


@dataclass(frozen=True)
class ReferenceConfig:
    policy_dt_s: float = 1 / 30
    velocity_match_window_s: float = .75
    contact_offset_limit_m: float = .025
    contact_velocity_adjustment_mps: float = .45
    joint_position_limit_rad: float = .34
    joint_velocity_limit_rad_s: float = 2.0
    differential_ik_damping: float = 2e-3
    differential_ik_step_limit_rad: float = .10
    actuation_lead_s: float = .10
    contact_follow_through_s: float = .14
    capture_hold_s: float = .10
    compliance_ramp_s: float = .12
    minimum_compliance_m_n: float = .025
    maximum_compliance_m_n: float = .16
    maximum_transverse_compliance_m_n: float = .075
    minimum_directional_softness_delta_m_n: float = .015
    follow_through_s: tuple[float, float] = (.16, .70)
    release_flight_s: tuple[float, float] = (.35, .75)
    admittance_mass_kg: float = .16
    admittance_damping_n_s_m: float = 6.0
    admittance_stiffness_n_m: float = 45.0
    admittance_limit_m: float = .025


class AdmittanceShaper:
    """High-rate impact correction, separate from static tangent compliance."""

    def __init__(self, config: ReferenceConfig):
        self.config = config
        self.displacement = np.zeros(3)
        self.velocity = np.zeros(3)

    def reset(self) -> None:
        self.displacement[:] = 0.0
        self.velocity[:] = 0.0

    def step(self, force_base_n: np.ndarray, dt: float) -> np.ndarray:
        acceleration = (
            np.asarray(force_base_n) - self.config.admittance_damping_n_s_m * self.velocity
            - self.config.admittance_stiffness_n_m * self.displacement
        ) / self.config.admittance_mass_kg
        self.velocity += float(dt) * acceleration
        self.displacement += float(dt) * self.velocity
        norm = float(np.linalg.norm(self.displacement))
        if norm > self.config.admittance_limit_m:
            self.displacement *= self.config.admittance_limit_m / norm
            self.velocity[:] = 0.0
        return self.displacement.copy()


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class HierarchicalReference:
    """Produces safe q/tip/C previews while PPO selects trajectory intent."""

    action_size = 10

    def __init__(self, task: CatchThrowTask, *, start_world_m: np.ndarray,
                 world_to_base, vector_world_to_base,
                 config: ReferenceConfig | None = None):
        self.task = task
        self.world_to_base = world_to_base
        self.vector_world_to_base = vector_world_to_base
        self.config = config or ReferenceConfig()
        self.approach_start_world_m = np.asarray(start_world_m, dtype=float).copy()
        if self.approach_start_world_m.shape != (3,) or not np.isfinite(
                self.approach_start_world_m).all():
            raise ValueError("start_world_m must contain three finite values")
        self.admittance = AdmittanceShaper(self.config)
        self.q_goal = np.zeros(12)
        self.tip_goal_base_m = tip_position(self.q_goal)
        self.tip_velocity_base_mps = np.zeros(3)
        self.compliance_ref = np.eye(3) * .06
        self.follow_through_s = float(np.mean(self.config.follow_through_s))
        self.release_flight_s = float(np.mean(self.config.release_flight_s))
        self.contact_offset_world_m = np.zeros(3)
        self.velocity_adjustment_world_mps = np.zeros(3)
        self.contact_velocity_world_mps = ballistic_state(task, task.intercept_time_s)[1]
        self.contact_time_s: float | None = None
        self.contact_cup_position_world_m: np.ndarray | None = None
        self.contact_cup_velocity_world_mps: np.ndarray | None = None
        self.grasp_time_s: float | None = None
        self.grasp_position_world_m: np.ndarray | None = None
        self.grasp_velocity_world_mps: np.ndarray | None = None
        self.release_time_s: float | None = None
        self.throw_enabled = True

    def reset(self) -> None:
        self.admittance.reset()
        self.contact_time_s = None
        self.contact_cup_position_world_m = None
        self.contact_cup_velocity_world_mps = None
        self.grasp_time_s = None
        self.grasp_position_world_m = None
        self.grasp_velocity_world_mps = None
        self.release_time_s = None
        self.throw_enabled = True

    def on_contact(self, time_s: float, cup_position_world_m: np.ndarray,
                   cup_velocity_world_mps: np.ndarray) -> None:
        if self.contact_time_s is not None:
            return
        self.contact_time_s = float(time_s)
        self.contact_cup_position_world_m = np.asarray(
            cup_position_world_m, dtype=float
        ).copy()
        self.contact_cup_velocity_world_mps = np.asarray(
            cup_velocity_world_mps, dtype=float
        ).copy()

    def on_grasp(self, time_s: float, position_world_m: np.ndarray,
                 velocity_world_mps: np.ndarray, *, throw_enabled: bool = True) -> None:
        self.grasp_time_s = float(time_s)
        self.grasp_position_world_m = np.asarray(position_world_m, dtype=float).copy()
        self.grasp_velocity_world_mps = np.asarray(velocity_world_mps, dtype=float).copy()
        self.throw_enabled = bool(throw_enabled)

    def on_release(self, time_s: float) -> None:
        self.release_time_s = float(time_s)

    def _catch_rendezvous_state(self, time_s: float, *, contact_world_m: np.ndarray,
                                contact_velocity_world_mps: np.ndarray,
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Cubic approach with mutually consistent position and velocity."""
        intercept_time = float(self.task.intercept_time_s)
        duration = min(intercept_time, self.config.velocity_match_window_s)
        start_time = intercept_time - duration
        time_s = float(time_s)
        if time_s <= start_time:
            return self.approach_start_world_m.copy(), np.zeros(3)
        if time_s >= intercept_time:
            elapsed = time_s - intercept_time
            return (
                contact_world_m
                + elapsed * contact_velocity_world_mps
                + .5 * GRAVITY_WORLD_MPS2 * elapsed**2,
                contact_velocity_world_mps + GRAVITY_WORLD_MPS2 * elapsed,
            )

        phase = (time_s - start_time) / max(duration, 1e-6)
        phase2 = phase * phase
        phase3 = phase2 * phase
        h00 = 2.0 * phase3 - 3.0 * phase2 + 1.0
        h01 = -2.0 * phase3 + 3.0 * phase2
        h11 = phase3 - phase2
        position = (
            h00 * self.approach_start_world_m
            + h01 * contact_world_m
            + h11 * duration * contact_velocity_world_mps
        )
        dh00 = 6.0 * phase2 - 6.0 * phase
        dh01 = -6.0 * phase2 + 6.0 * phase
        dh11 = 3.0 * phase2 - 2.0 * phase
        velocity = (
            (dh00 * self.approach_start_world_m + dh01 * contact_world_m)
            / max(duration, 1e-6)
            + dh11 * contact_velocity_world_mps
        )
        return position, velocity

    def nominal_catch_state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Action-independent rendezvous target used by reward diagnostics."""
        incoming = ballistic_state(self.task, self.task.intercept_time_s)[1]
        return self._catch_rendezvous_state(
            time_s,
            contact_world_m=self.task.intercept_world_m,
            contact_velocity_world_mps=incoming,
        )

    def desired_catch_state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Current action-conditioned catch state at an explicit time."""
        return self._catch_rendezvous_state(
            time_s,
            contact_world_m=self.task.intercept_world_m + self.contact_offset_world_m,
            contact_velocity_world_mps=self.contact_velocity_world_mps,
        )

    def _desired_world_state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.grasp_time_s is None:
            if self.contact_time_s is not None:
                anchor = self.contact_cup_position_world_m
                initial_velocity = self.contact_cup_velocity_world_mps
                assert anchor is not None and initial_velocity is not None
                elapsed = max(0.0, float(time_s) - self.contact_time_s)
                duration = max(self.config.contact_follow_through_s, 1e-3)
                blend = _smoothstep(elapsed / duration)
                velocity = (
                    (1.0 - blend) * initial_velocity
                    + blend * self.contact_velocity_world_mps
                )
                position = anchor + elapsed * .5 * (initial_velocity + velocity)
                return position, velocity
            return self.desired_catch_state(time_s)

        anchor = self.grasp_position_world_m
        initial_velocity = self.grasp_velocity_world_mps
        assert anchor is not None and initial_velocity is not None
        if not self.throw_enabled:
            duration = max(self.config.capture_hold_s, 1e-3)
            elapsed = max(0.0, float(time_s) - self.grasp_time_s)
            phase = float(np.clip(elapsed / duration, 0.0, 1.0))
            blend = _smoothstep(phase)
            velocity = (1.0 - blend) * initial_velocity
            integrated_blend = phase - phase**3 + .5 * phase**4
            position = anchor + initial_velocity * duration * integrated_blend
            return position, velocity
        release_velocity = (
            desired_release_velocity(anchor, self.task.target_center_world_m, self.release_flight_s)
            + self.velocity_adjustment_world_mps
        )
        elapsed = max(0.0, time_s - self.grasp_time_s)
        duration = max(self.follow_through_s, 1e-3)
        blend = _smoothstep(elapsed / duration)
        velocity = (1.0 - blend) * initial_velocity + blend * release_velocity
        # Integrating the smooth velocity exactly is unnecessary at this short
        # horizon; the trapezoid preserves momentum continuity at the grasp.
        position = anchor + elapsed * .5 * (initial_velocity + velocity)
        return position, velocity

    def set_action(self, action: np.ndarray, *, time_s: float, q_now: np.ndarray,
                   ball_velocity_world_mps: np.ndarray) -> None:
        del q_now
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError("high-level catch action must be 10 finite values")
        action = np.clip(action, -1.0, 1.0)
        self.contact_offset_world_m = action[:3] * self.config.contact_offset_limit_m
        lookahead_s = self.task.intercept_time_s - float(time_s)
        incoming_velocity = (
            np.asarray(ball_velocity_world_mps, dtype=float)
            + GRAVITY_WORLD_MPS2 * lookahead_s
        )
        self.velocity_adjustment_world_mps = action[3:6] * self.config.contact_velocity_adjustment_mps
        self.contact_velocity_world_mps = incoming_velocity + self.velocity_adjustment_world_mps
        self.follow_through_s = float(np.interp(action[8], (-1, 1), self.config.follow_through_s))
        self.release_flight_s = float(np.interp(action[9], (-1, 1), self.config.release_flight_s))

        incoming_base = self.vector_world_to_base(incoming_velocity)
        norm = np.linalg.norm(incoming_base)
        direction = np.array([0.0, 0.0, 1.0]) if norm < 1e-8 else incoming_base / norm
        c_perpendicular = float(np.interp(
            action[7], (-1, 1),
            (self.config.minimum_compliance_m_n, self.config.maximum_transverse_compliance_m_n),
        ))
        softness_delta = float(np.interp(
            action[6], (-1, 1),
            (self.config.minimum_directional_softness_delta_m_n,
             self.config.maximum_compliance_m_n - c_perpendicular),
        ))
        c_parallel = min(self.config.maximum_compliance_m_n, c_perpendicular + softness_delta)
        self.compliance_ref = (
            c_perpendicular * np.eye(3)
            + (c_parallel - c_perpendicular) * np.outer(direction, direction)
        )

    def _damped_joint_step(self, q: np.ndarray, target_base_m: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).copy()
        damping2 = self.config.differential_ik_damping**2
        for _ in range(4):
            error = np.asarray(target_base_m, dtype=float) - tip_position(q)
            if float(np.linalg.norm(error)) < 2e-5:
                break
            jacobian = tip_jacobian(q)
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping2 * np.eye(3), error
            )
            norm = float(np.linalg.norm(delta))
            limit = self.config.differential_ik_step_limit_rad
            if norm > limit:
                delta *= limit / norm
            q = np.clip(
                q + delta,
                -self.config.joint_position_limit_rad,
                self.config.joint_position_limit_rad,
            )
        return q

    def _joint_velocity(self, q: np.ndarray, velocity_base_mps: np.ndarray) -> np.ndarray:
        jacobian = tip_jacobian(q)
        damping2 = self.config.differential_ik_damping**2
        qd = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping2 * np.eye(3),
            np.asarray(velocity_base_mps, dtype=float),
        )
        return np.clip(
            qd,
            -self.config.joint_velocity_limit_rad_s,
            self.config.joint_velocity_limit_rad_s,
        )

    def _compliance_at(self, time_s: float) -> np.ndarray:
        if self.contact_time_s is not None or self.grasp_time_s is not None:
            blend = 1.0
        else:
            start = self.task.intercept_time_s - self.config.compliance_ramp_s
            blend = _smoothstep(
                (float(time_s) - start) / max(self.config.compliance_ramp_s, 1e-6)
            )
        tracking = np.eye(3) * self.config.minimum_compliance_m_n
        return (1.0 - blend) * tracking + blend * self.compliance_ref

    def low_level_reference(self, *, q_now: np.ndarray, qdot_now: np.ndarray,
                            force_world_n: np.ndarray, dt: float, horizon: int,
                            time_s: float = 0.0) -> dict[str, np.ndarray]:
        del qdot_now
        if horizon < 1 or dt <= 0:
            raise ValueError("horizon and dt must be positive")
        force_base = self.vector_world_to_base(force_world_n)
        displacement = self.admittance.step(force_base, dt)
        sample_times = (
            float(time_s)
            + self.config.actuation_lead_s
            + np.arange(1, horizon + 1, dtype=float) * float(dt)
        )
        future_q = np.empty((horizon, 12), dtype=float)
        future_qd = np.empty((horizon, 12), dtype=float)
        future_tip = np.empty((horizon, 3), dtype=float)
        future_tip_velocity = np.empty((horizon, 3), dtype=float)
        future_compliance = np.empty((horizon, 3, 3), dtype=float)
        q_seed = np.asarray(q_now, dtype=float).copy()
        for index, sample_time in enumerate(sample_times):
            desired_world, desired_velocity_world = self._desired_world_state(sample_time)
            desired_base = self.world_to_base(desired_world) + displacement
            desired_velocity_base = self.vector_world_to_base(desired_velocity_world)
            q_seed = self._damped_joint_step(q_seed, desired_base)
            future_q[index] = q_seed
            future_qd[index] = self._joint_velocity(q_seed, desired_velocity_base)
            future_tip[index] = desired_base
            future_tip_velocity[index] = desired_velocity_base
            future_compliance[index] = self._compliance_at(sample_time)

        q_ref = future_q[0]
        correction_horizon = max(self.config.actuation_lead_s + dt, dt)
        qd_correction = .45 * (q_ref - np.asarray(q_now, dtype=float)) / correction_horizon
        qd_ref = np.clip(
            future_qd[0] + qd_correction,
            -self.config.joint_velocity_limit_rad_s,
            self.config.joint_velocity_limit_rad_s,
        )
        # The pneumatic inverse-dynamics feedforward was identified with a
        # smooth acceleration reference. Finite-differencing IK velocities at
        # 150 Hz creates impulsive qdd commands, so MPPI receives the velocity
        # preview while feedforward remains acceleration-neutral.
        qdd_ref = np.zeros(12)
        self.q_goal = future_q[-1].copy()
        self.tip_goal_base_m = future_tip[-1].copy()
        self.tip_velocity_base_mps = future_tip_velocity[-1].copy()
        return {
            "q_ref": q_ref, "qd_ref": qd_ref, "qdd_ref": qdd_ref,
            "future_q": future_q, "future_qd": future_qd,
            "future_tip": future_tip, "future_tip_velocity": future_tip_velocity,
            "future_compliance": future_compliance,
        }
