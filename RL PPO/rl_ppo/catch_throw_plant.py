"""150 Hz pneumatic plant with 1 kHz gripper contact and rigid grasp events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from control.controller import project_pressures
from digital_twin.replay import pa_to_counts
from digital_twin.sim_core import SimArm
from digital_twin.twin_params import load_twin_kwargs

from .physical_scene import (
    BALL_BODY, BALL_GEOM, BALL_JOINT, CUP_BODY, CUP_SITE, GRASP_WELD,
    CatcherSpec, build_catch_xml, catcher_geom_names,
)


@dataclass(frozen=True)
class ContactTelemetry:
    force_world_n: np.ndarray
    force_norm_n: float
    average_force_norm_n: float
    impulse_ns: float
    window_impulse_ns: float
    contact_count: int
    contact_quanta: int
    generalized_torque_nm: np.ndarray

    @classmethod
    def zeros(cls) -> "ContactTelemetry":
        return cls(np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0, 0, np.zeros(12))


@dataclass(frozen=True)
class _GraspGate:
    mode: str
    radial_limit_m: float
    auto_radial_limit_m: float
    axial_bounds_m: tuple[float, float]
    relative_speed_limit_mps: float
    alignment_cosine: float


class _ContactArm(SimArm):
    """SimArm with mechanical-work and every-quantum contact integration."""

    def __init__(self, *, ball_geom_name: str, contact_geom_names: tuple[str, ...], **kwargs: Any):
        self.mechanical_work_abs_j = 0.0
        self._contact_ready = False
        super().__init__(**kwargs)
        self._ball_contact_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, ball_geom_name
        )
        self._gripper_contact_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in contact_geom_names
        }
        if self._ball_contact_geom < 0 or min(self._gripper_contact_geoms) < 0:
            raise RuntimeError("could not bind gripper contact geoms")
        self._total_impulse_ns = 0.0
        self._reset_contact_window()
        self.contact_callback = None
        self._contact_ready = True

    def _reset_contact_window(self) -> None:
        self._window_force_vector_impulse = np.zeros(3)
        self._window_force_norm_impulse = 0.0
        self._window_peak_force = 0.0
        self._window_contact_count = 0
        self._window_contact_quanta = 0

    def _contact_force_on_gripper(self) -> tuple[np.ndarray, int]:
        force_world = np.zeros(3)
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._ball_contact_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self._ball_contact_geom else contact.geom1
            if other not in self._gripper_contact_geoms:
                continue
            force_local = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force_local)
            mapped = contact.frame.reshape(3, 3).T @ force_local[:3]
            # mj_contactForce is the force on geom2. Return the force applied to
            # the gripper regardless of MuJoCo's geom ordering.
            force_world += mapped if contact.geom2 in self._gripper_contact_geoms else -mapped
            count += 1
        return force_world, count

    def _step_quantum(self) -> None:
        length_before = self.data.ten_length[self._ten_ids].copy()
        super()._step_quantum()
        delta_length = self.data.ten_length[self._ten_ids] - length_before
        # Absolute muscle boundary work, not compressor electrical energy.
        self.mechanical_work_abs_j += float(np.sum(np.abs(self.data.ctrl[self._act_ids] * delta_length)))
        if self._contact_ready:
            force, count = self._contact_force_on_gripper()
            norm = float(np.linalg.norm(force))
            self._window_force_vector_impulse += force * self.dt
            self._window_force_norm_impulse += norm * self.dt
            self._window_peak_force = max(self._window_peak_force, norm)
            self._window_contact_count += count
            self._window_contact_quanta += int(count > 0)
            self._total_impulse_ns += norm * self.dt
            if self.contact_callback is not None:
                self.contact_callback(count > 0)

    def consume_contact_window(self, elapsed_s: float, cup_site: int) -> ContactTelemetry:
        elapsed_s = max(float(elapsed_s), self.dt)
        mean_force = self._window_force_vector_impulse / elapsed_s
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, cup_site)
        torque = jacobian[:, self._q_qposadr].T @ mean_force
        telemetry = ContactTelemetry(
            force_world_n=mean_force.copy(),
            force_norm_n=self._window_peak_force,
            average_force_norm_n=self._window_force_norm_impulse / elapsed_s,
            impulse_ns=self._total_impulse_ns,
            window_impulse_ns=self._window_force_norm_impulse,
            contact_count=self._window_contact_count,
            contact_quanta=self._window_contact_quanta,
            generalized_torque_nm=torque,
        )
        self._reset_contact_window()
        return telemetry

    def reset_diagnostics(self) -> None:
        self.mechanical_work_abs_j = 0.0
        self._total_impulse_ns = 0.0
        self._reset_contact_window()


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float).copy()
    result[1:] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class CatchThrowPlant:
    """Physical ball/gripper scene; MPPI remains the sole pressure commander."""

    dt: float = 1 / 150

    def __init__(self, *, seed: int = 20260914, catcher: CatcherSpec | None = None,
                 twin_kwargs: dict[str, Any] | None = None):
        self.seed = int(seed)
        self.catcher = catcher or CatcherSpec()
        self._twin_kwargs = load_twin_kwargs(log=lambda _: None) if twin_kwargs is None else twin_kwargs
        xml_kwargs = {key: value for key, value in self._twin_kwargs.items() if key != "actuator"}
        self.xml = build_catch_xml(catcher=self.catcher, **xml_kwargs)
        self.arm: _ContactArm | None = None
        self.grasped = False
        self.ball_mass_kg = self.catcher.ball_mass_kg
        self.gripper_energy_j = 0.0
        self._grasp_gate: _GraspGate | None = None
        self._grasp_event_this_step = False
        self._contact_event_this_step = False
        self._last_contact_metrics: dict[str, float] | None = None
        self._last_grasp_metrics: dict[str, float] | None = None
        self._last_grasp_jump_m = 0.0
        self._grasp_candidate_s = 0.0
        self._maximum_grasp_candidate_s = 0.0

    def reset(self, ball_position_m: np.ndarray, ball_velocity_mps: np.ndarray) -> dict[str, Any]:
        self.close()
        geom_names = catcher_geom_names(self.catcher)
        self.arm = _ContactArm(
            actuator=self._twin_kwargs["actuator"], xml=self.xml, seed=self.seed,
            batched_actuator=True, ball_geom_name=BALL_GEOM, contact_geom_names=geom_names,
        )
        self.arm.advance_to(0.0)
        self.t = 0.0
        self.episode_origin_s = 0.0
        self._ids = tuple(sorted(self.arm.nodes))
        self._variants = np.array([self.arm.nodes[base].variant for base in self._ids], dtype=int)
        self._ball_joint = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT)
        self._ball_qadr = int(self.arm.model.jnt_qposadr[self._ball_joint])
        self._ball_dadr = int(self.arm.model.jnt_dofadr[self._ball_joint])
        self._ball_geom = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM)
        self._ball_body = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
        self._cup_site = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_SITE, CUP_SITE)
        self._cup_body = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_BODY, CUP_BODY)
        self._base_body = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_BODY, "canarm_base")
        self._grasp_weld = mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD)
        self._cup_geoms = {
            mujoco.mj_name2id(self.arm.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names
        }
        required = (self._ball_joint, self._ball_geom, self._ball_body, self._cup_site,
                    self._cup_body, self._base_body, self._grasp_weld, *self._cup_geoms)
        if min(required) < 0:
            raise RuntimeError("catch scene is missing a required named object")
        self.arm.data.eq_active[self._grasp_weld] = 0
        self._ball_contact_bits = (
            int(self.arm.model.geom_contype[self._ball_geom]),
            int(self.arm.model.geom_conaffinity[self._ball_geom]),
        )
        self.base_position_world_m = self.arm.data.xpos[self._base_body].copy()
        self.base_rotation_world = self.arm.data.xmat[self._base_body].reshape(3, 3).copy()
        self.grasped = False
        self.ball_mass_kg = self.catcher.ball_mass_kg
        self.gripper_energy_j = 0.0
        self._latest_contact = ContactTelemetry.zeros()
        self._contact_event_this_step = False
        self._last_contact_metrics = None
        self._grasp_event_this_step = False
        self._last_grasp_metrics = None
        self._last_grasp_jump_m = 0.0
        self._grasp_candidate_s = 0.0
        self._maximum_grasp_candidate_s = 0.0
        self.arm.contact_callback = self._try_grasp_at_quantum
        self.set_ball_state(ball_position_m, ball_velocity_mps)
        return self.observe()

    def begin_episode(self, ball_position_m: np.ndarray,
                      ball_velocity_mps: np.ndarray) -> dict[str, Any]:
        """Launch a ball after arm precharge and zero episode diagnostics/time."""
        if self.arm is None:
            raise RuntimeError("reset plant before beginning an episode")
        if self.grasped:
            self.release_grasp()
        self.set_ball_state(ball_position_m, ball_velocity_mps)
        self.episode_origin_s = self.t
        self.arm.reset_diagnostics()
        self._latest_contact = ContactTelemetry.zeros()
        self.gripper_energy_j = 0.0
        self._contact_event_this_step = False
        self._last_contact_metrics = None
        self._grasp_event_this_step = False
        self._last_grasp_metrics = None
        self._last_grasp_jump_m = 0.0
        self._grasp_candidate_s = 0.0
        self._maximum_grasp_candidate_s = 0.0
        return self.observe()

    def configure_grasp_gate(self, *, radial_limit_m: float,
                             axial_bounds_m: tuple[float, float],
                             relative_speed_limit_mps: float,
                             alignment_cosine: float,
                             mode: str = "physical_dwell",
                             auto_radial_limit_m: float | None = None) -> None:
        """Arm a physical-dwell or geometry-only closure test at 1 kHz."""
        if mode not in ("physical_dwell", "auto_volume"):
            raise ValueError("grasp mode must be physical_dwell or auto_volume")
        radial_limit_m = float(radial_limit_m)
        auto_radial_limit_m = (
            radial_limit_m
            if auto_radial_limit_m is None else float(auto_radial_limit_m)
        )
        axial_bounds = tuple(float(value) for value in axial_bounds_m)
        if (radial_limit_m <= 0.0 or auto_radial_limit_m <= 0.0
                or len(axial_bounds) != 2 or axial_bounds[0] >= axial_bounds[1]):
            raise ValueError("invalid grasp volume")
        self._grasp_gate = _GraspGate(
            mode=mode,
            radial_limit_m=radial_limit_m,
            auto_radial_limit_m=auto_radial_limit_m,
            axial_bounds_m=axial_bounds,
            relative_speed_limit_mps=float(relative_speed_limit_mps),
            alignment_cosine=float(alignment_cosine),
        )

    def disable_grasp_gate(self) -> None:
        self._grasp_gate = None

    def _try_grasp_at_quantum(self, contact_active: bool) -> None:
        if self.arm is None or self.grasped:
            return
        cup_position = self.arm.data.site_xpos[self._cup_site]
        cup_rotation = self.arm.data.site_xmat[self._cup_site].reshape(3, 3)
        ball_position = self.arm.data.qpos[self._ball_qadr:self._ball_qadr + 3]
        jacobian = np.zeros((3, self.arm.model.nv))
        mujoco.mj_jacSite(self.arm.model, self.arm.data, jacobian, None, self._cup_site)
        cup_velocity = jacobian @ self.arm.data.qvel
        ball_velocity = self.arm.data.qvel[self._ball_dadr:self._ball_dadr + 3]
        relative_speed = float(np.linalg.norm(ball_velocity - cup_velocity))
        ball_speed = float(np.linalg.norm(ball_velocity))
        cup_speed = float(np.linalg.norm(cup_velocity))
        if ball_speed < .05 or cup_speed < .05:
            alignment = 1.0 if relative_speed < .05 else 0.0
        else:
            alignment = float(np.dot(ball_velocity, cup_velocity) / (ball_speed * cup_speed))
        local = cup_rotation.T @ (ball_position - cup_position)
        radial = float(np.linalg.norm(local[:2]))
        gate = self._grasp_gate
        physical_compatible = False
        auto_volume_inside = False
        if gate is not None:
            axial_min, axial_max = gate.axial_bounds_m
            physical_compatible = bool(
                contact_active
                and radial <= gate.radial_limit_m
                and axial_min <= local[2] <= axial_max
                and relative_speed <= gate.relative_speed_limit_mps
                and alignment >= gate.alignment_cosine
            )
            auto_volume_inside = bool(
                radial <= gate.auto_radial_limit_m
                and axial_min <= local[2] <= axial_max
            )
        metrics = {
            "relative_speed_mps": relative_speed,
            "velocity_alignment": alignment,
            "radial_offset_m": radial,
            "axial_offset_m": float(local[2]),
            "physical_contact": bool(contact_active),
            "physical_compatible": physical_compatible,
            "auto_volume_inside": auto_volume_inside,
            "ball_position_world_m": ball_position.copy(),
            "cup_position_world_m": cup_position.copy(),
            "ball_velocity_world_mps": ball_velocity.copy(),
            "cup_velocity_world_mps": cup_velocity.copy(),
        }
        if contact_active and not self._contact_event_this_step:
            self._last_contact_metrics = metrics
        self._contact_event_this_step |= bool(contact_active)
        if gate is None:
            self._grasp_candidate_s = 0.0
            return
        if physical_compatible:
            self._grasp_candidate_s += self.arm.dt
            self._maximum_grasp_candidate_s = max(
                self._maximum_grasp_candidate_s, self._grasp_candidate_s
            )
        else:
            self._grasp_candidate_s = 0.0

        if gate.mode == "auto_volume":
            should_grasp = auto_volume_inside
        else:
            should_grasp = (
                self._grasp_candidate_s + 1e-12 >= self.catcher.grasp_dwell_s
            )
        if not should_grasp:
            return
        metrics["trigger_mode"] = gate.mode
        metrics["physical_dwell_fraction"] = float(np.clip(
            self._grasp_candidate_s / self.catcher.grasp_dwell_s, 0.0, 1.0
        ))
        self._last_grasp_metrics = metrics
        self._grasp_event_this_step = True
        self._last_grasp_jump_m = self.activate_grasp()

    def world_to_base(self, position_world_m: np.ndarray) -> np.ndarray:
        return self.base_rotation_world.T @ (np.asarray(position_world_m, dtype=float) - self.base_position_world_m)

    def base_to_world(self, position_base_m: np.ndarray) -> np.ndarray:
        return self.base_position_world_m + self.base_rotation_world @ np.asarray(position_base_m, dtype=float)

    def vector_world_to_base(self, vector_world: np.ndarray) -> np.ndarray:
        return self.base_rotation_world.T @ np.asarray(vector_world, dtype=float)

    def vector_base_to_world(self, vector_base: np.ndarray) -> np.ndarray:
        return self.base_rotation_world @ np.asarray(vector_base, dtype=float)

    def close(self) -> None:
        if self.arm is not None:
            self.arm.all_off()
            self.arm = None

    def set_ball_state(self, position_m: np.ndarray, velocity_mps: np.ndarray) -> None:
        if self.arm is None:
            raise RuntimeError("reset plant before setting ball state")
        if self.grasped:
            raise RuntimeError("cannot overwrite ball state while grasped")
        position, velocity = np.asarray(position_m, dtype=float), np.asarray(velocity_mps, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,) or not np.isfinite(np.r_[position, velocity]).all():
            raise ValueError("ball position and velocity must be finite xyz vectors")
        self.arm.data.qpos[self._ball_qadr:self._ball_qadr + 3] = position
        self.arm.data.qpos[self._ball_qadr + 3:self._ball_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.arm.data.qvel[self._ball_dadr:self._ball_dadr + 3] = velocity
        self.arm.data.qvel[self._ball_dadr + 3:self._ball_dadr + 6] = 0.0
        mujoco.mj_forward(self.arm.model, self.arm.data)

    def set_ball_mass(self, mass_kg: float) -> None:
        """Set the free ball mass and matching solid-sphere inertia for an episode."""
        if self.arm is None:
            raise RuntimeError("reset plant before setting ball mass")
        mass_kg = float(mass_kg)
        if not np.isfinite(mass_kg) or mass_kg <= 0.0:
            raise ValueError("ball mass must be finite and positive")
        inertia = .4 * mass_kg * self.catcher.ball_radius_m**2
        self.arm.model.body_mass[self._ball_body] = mass_kg
        self.arm.model.body_inertia[self._ball_body] = inertia
        mujoco.mj_setConst(self.arm.model, self.arm.data)
        mujoco.mj_forward(self.arm.model, self.arm.data)
        self.ball_mass_kg = mass_kg

    def activate_grasp(self) -> float:
        """Close the fingers at the current relative pose without teleporting the ball."""
        if self.arm is None:
            raise RuntimeError("reset plant before grasping")
        if self.grasped:
            return 0.0
        before_qpos = self.arm.data.qpos[self._ball_qadr:self._ball_qadr + 7].copy()
        ball_world_position = self.arm.data.xpos[self._ball_body].copy()
        gripper_pos = self.arm.data.xpos[self._cup_body]
        gripper_rotation = self.arm.data.xmat[self._cup_body].reshape(3, 3)
        relative_position = gripper_rotation.T @ (ball_world_position - gripper_pos)
        relative_quaternion = _quat_multiply(
            _quat_conjugate(self.arm.data.xquat[self._cup_body]), self.arm.data.xquat[self._ball_body]
        )
        relative_quaternion /= np.linalg.norm(relative_quaternion)
        equality_data = self.arm.model.eq_data[self._grasp_weld]
        equality_data[3:6] = relative_position
        equality_data[6:10] = relative_quaternion
        self.arm.data.eq_active[self._grasp_weld] = 1
        # Once the fingers have closed, the weld represents their internal
        # holding force. Leaving the same ball/finger contacts active would
        # overconstrain one rigid assembly and count artificial internal force.
        self.arm.model.geom_contype[self._ball_geom] = 0
        self.arm.model.geom_conaffinity[self._ball_geom] = 0
        mujoco.mj_forward(self.arm.model, self.arm.data)
        self.grasped = True
        self.gripper_energy_j += self.catcher.close_energy_j
        after_qpos = self.arm.data.qpos[self._ball_qadr:self._ball_qadr + 7]
        return float(np.linalg.norm(after_qpos - before_qpos))

    def release_grasp(self) -> np.ndarray:
        """Open the fingers while preserving the physical free-joint velocity."""
        if self.arm is None:
            raise RuntimeError("reset plant before releasing")
        velocity = self.arm.data.qvel[self._ball_dadr:self._ball_dadr + 3].copy()
        if self.grasped:
            self.arm.data.eq_active[self._grasp_weld] = 0
            self.arm.model.geom_contype[self._ball_geom] = self._ball_contact_bits[0]
            self.arm.model.geom_conaffinity[self._ball_geom] = self._ball_contact_bits[1]
            mujoco.mj_forward(self.arm.model, self.arm.data)
            self.grasped = False
            self.gripper_energy_j += self.catcher.open_energy_j
            self._grasp_candidate_s = 0.0
        return velocity

    def observe(self) -> dict[str, Any]:
        if self.arm is None:
            raise RuntimeError("plant is not reset")
        cup_position = self.arm.data.site_xpos[self._cup_site].copy()
        cup_rotation = self.arm.data.site_xmat[self._cup_site].reshape(3, 3).copy()
        jacobian = np.zeros((3, self.arm.model.nv))
        mujoco.mj_jacSite(self.arm.model, self.arm.data, jacobian, None, self._cup_site)
        cup_velocity = jacobian @ self.arm.data.qvel
        ball_velocity = self.arm.data.qvel[self._ball_dadr:self._ball_dadr + 3].copy()
        qd = self.arm.data.qvel[self.arm._q_qposadr].copy()
        return {
            "time_s": self.t - self.episode_origin_s,
            "q": self.arm.q(), "qdot": qd, "p_pa": self.arm.pressures_pa(),
            "ball_position_m": self.arm.data.qpos[self._ball_qadr:self._ball_qadr + 3].copy(),
            "ball_velocity_mps": ball_velocity,
            "ball_angular_velocity_rad_s": self.arm.data.qvel[self._ball_dadr + 3:self._ball_dadr + 6].copy(),
            "ball_momentum_kg_mps": self.ball_mass_kg * ball_velocity,
            "ball_kinetic_energy_j": .5 * self.ball_mass_kg * float(ball_velocity @ ball_velocity),
            "cup_position_m": cup_position, "cup_rotation": cup_rotation,
            "cup_velocity_mps": cup_velocity, "contact": self._latest_contact,
            "grasped": self.grasped,
            "contact_event": self._contact_event_this_step,
            "contact_event_metrics": (
                None if self._last_contact_metrics is None else dict(self._last_contact_metrics)
            ),
            "grasp_event": self._grasp_event_this_step,
            "grasp_event_metrics": (
                None if self._last_grasp_metrics is None else dict(self._last_grasp_metrics)
            ),
            "grasp_jump_m": self._last_grasp_jump_m,
            "grasp_candidate_s": self._grasp_candidate_s,
            "maximum_grasp_candidate_s": self._maximum_grasp_candidate_s,
            "grasp_dwell_s": self.catcher.grasp_dwell_s,
            "mechanical_work_abs_j": self.arm.mechanical_work_abs_j,
            "gripper_energy_j": self.gripper_energy_j,
            "constraint_torque_nm": self.arm.data.qfrc_constraint[self.arm._q_qposadr].copy(),
        }

    def step(self, target_pressure_pa: np.ndarray) -> dict[str, Any]:
        if self.arm is None:
            raise RuntimeError("reset plant before stepping")
        pressure = project_pressures(np.asarray(target_pressure_pa, dtype=float))
        if pressure.shape != (24,):
            raise ValueError("target pressure must have 24 values")
        self._contact_event_this_step = False
        self._grasp_event_this_step = False
        counts = np.array([
            int(round(float(pa_to_counts(value, variant))))
            for value, variant in zip(pressure, self._variants)
        ])
        self.arm.stage_targets({base: (int(count), True) for base, count in zip(self._ids, counts)})
        self.arm.sync_edge(self.t)
        quanta_before = self.arm.quanta_done
        self.arm.advance_to(self.t + self.dt)
        elapsed = (self.arm.quanta_done - quanta_before) * self.arm.dt
        self.t += self.dt
        self._latest_contact = self.arm.consume_contact_window(elapsed, self._cup_site)
        if not np.isfinite(self.arm.data.qpos).all() or not np.isfinite(self.arm.data.qvel).all():
            raise RuntimeError("nonfinite physical catch simulation state")
        return self.observe()
