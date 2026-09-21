import mujoco
import numpy as np

from rl_ppo.catch_throw_plant import CatchThrowPlant
from rl_ppo.catch_throw_task import ballistic_state, desired_release_velocity, sample_task
from rl_ppo.physical_scene import (
    BALL_GEOM, CUP_SITE, CUP_WALL_PREFIX, FINGER_COUNT, GRASP_WELD, build_catch_xml,
)


def test_catch_scene_has_free_ball_three_fingers_and_disabled_grasp() -> None:
    model = mujoco.MjModel.from_xml_string(build_catch_xml())
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CUP_SITE) >= 0
    assert model.nu == 24
    assert model.nq == 19  # 12 arm hinges plus the ball's 7 free-joint coordinates.
    assert sum(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"{CUP_WALL_PREFIX}{i}") >= 0
               for i in range(FINGER_COUNT)) == 3
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD)
    assert weld >= 0
    assert not model.eq_active0[weld]


def test_physical_plant_reports_ball_state_and_contact_wrench_shape() -> None:
    plant = CatchThrowPlant(seed=3)
    try:
        state = plant.reset(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
        stepped = plant.step(np.full(24, 6 * 6894.757))
    finally:
        plant.close()
    assert state["ball_position_m"].shape == (3,)
    assert stepped["contact"].force_world_n.shape == (3,)
    assert stepped["contact"].generalized_torque_nm.shape == (12,)


def test_ball_launch_reaches_offset_intercept_instead_of_static_cup() -> None:
    plant = CatchThrowPlant(seed=5)
    try:
        home = plant.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3))["cup_position_m"]
        task = sample_task(np.random.default_rng(5), home)
        position, velocity = ballistic_state(task, task.intercept_time_s)
    finally:
        plant.close()
    np.testing.assert_allclose(position, task.intercept_world_m, atol=1e-7)
    np.testing.assert_allclose(velocity, ballistic_state(task, task.intercept_time_s)[1])
    assert np.linalg.norm(task.intercept_world_m[:2] - home[:2]) >= .015


def test_final_task_crosses_the_arm_horizontally_toward_the_far_target() -> None:
    plant = CatchThrowPlant(seed=12)
    try:
        state = plant.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3))
        home = state["cup_position_m"]
        task = sample_task(np.random.default_rng(12), home, curriculum_stage=5)
        incoming = ballistic_state(task, task.intercept_time_s)[1]
        aperture_axis = state["cup_rotation"][:, 2]
    finally:
        plant.close()
    assert home[1] - 1.45 < task.ball_position0_world_m[1] < home[1] - .90
    assert task.target_center_world_m[1] > home[1] + .75
    assert abs(task.ball_position0_world_m[2] - home[2]) <= .035
    assert 2.0 <= incoming[1] <= 3.8
    assert np.dot(aperture_axis, incoming) / np.linalg.norm(incoming) > .97
    assert .10 <= np.linalg.norm(task.intercept_world_m[:2] - home[:2]) <= .18
    assert .020 <= task.ball_mass_kg <= .035


def test_every_curriculum_stage_uses_a_far_side_launch() -> None:
    home = np.array([0.0, 0.0, .375])
    rng = np.random.default_rng(19)
    for stage in range(1, 6):
        for _ in range(100):
            task = sample_task(rng, home, curriculum_stage=stage, stage2_level=2)
            distance = task.intercept_world_m[1] - task.ball_position0_world_m[1]
            incoming = ballistic_state(task, task.intercept_time_s)[1]
            assert .75 <= distance <= 1.42
            assert abs(task.ball_position0_world_m[2] - home[2]) <= .036
            assert np.linalg.norm(task.intercept_world_m[:2] - home[:2]) >= .010
            assert .45 <= -incoming[2] / incoming[1] <= 1.01


def test_grasp_weld_does_not_teleport_and_release_preserves_velocity() -> None:
    plant = CatchThrowPlant(seed=8)
    pressure = np.full(24, 6 * 6894.757)
    try:
        home = plant.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3))["cup_position_m"]
        plant.reset(home, np.zeros(3))
        qpos_before = plant.arm.data.qpos.copy()
        jump = plant.activate_grasp()
        qpos_after = plant.arm.data.qpos.copy()
        local_positions = []
        for _ in range(12):
            state = plant.step(pressure)
            local_positions.append(
                state["cup_rotation"].T @ (state["ball_position_m"] - state["cup_position_m"])
            )
        velocity_before_release = state["ball_velocity_mps"].copy()
        returned_velocity = plant.release_grasp()
        velocity_after_release = plant.observe()["ball_velocity_mps"]
    finally:
        plant.close()
    np.testing.assert_allclose(qpos_after, qpos_before, atol=0.0)
    assert jump < 5e-4
    assert np.max(np.ptp(np.asarray(local_positions), axis=0)) < 2e-3
    np.testing.assert_allclose(returned_velocity, velocity_before_release, atol=0.0)
    np.testing.assert_allclose(velocity_after_release, velocity_before_release, atol=0.0)


def test_auto_volume_grasps_without_contact_and_physical_mode_does_not() -> None:
    pressure = np.full(24, 6 * 6894.757)
    auto_plant = CatchThrowPlant(seed=18)
    try:
        state = auto_plant.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3))
        inside_position = (
            state["cup_position_m"] + state["cup_rotation"] @ np.array([0.0, 0.0, -.005])
        )
        auto_plant.set_ball_state(inside_position, np.zeros(3))
        auto_plant.configure_grasp_gate(
            radial_limit_m=.037,
            auto_radial_limit_m=.025,
            axial_bounds_m=(-.028, .046),
            relative_speed_limit_mps=.63,
            alignment_cosine=.45,
            mode="auto_volume",
        )
        auto_state = auto_plant.step(pressure)
        assert auto_state["grasped"]
        assert auto_state["grasp_event"]
        assert auto_state["grasp_event_metrics"]["trigger_mode"] == "auto_volume"
        assert auto_state["grasp_event_metrics"]["auto_volume_inside"]
        assert not auto_state["grasp_event_metrics"]["physical_contact"]
        assert not auto_state["grasp_event_metrics"]["physical_compatible"]
    finally:
        auto_plant.close()

    physical_plant = CatchThrowPlant(seed=18)
    try:
        state = physical_plant.reset(np.array([0.0, 0.0, -3.0]), np.zeros(3))
        inside_position = (
            state["cup_position_m"] + state["cup_rotation"] @ np.array([0.0, 0.0, -.005])
        )
        physical_plant.set_ball_state(inside_position, np.zeros(3))
        physical_plant.configure_grasp_gate(
            radial_limit_m=.037,
            axial_bounds_m=(-.028, .046),
            relative_speed_limit_mps=.63,
            alignment_cosine=.45,
            mode="physical_dwell",
        )
        physical_state = physical_plant.step(pressure)
        assert not physical_state["grasped"]
        assert not physical_state["grasp_event"]
    finally:
        physical_plant.close()


def test_release_velocity_reaches_fixed_target_under_ballistics() -> None:
    release = np.array([0.1, -0.2, 0.4])
    target = np.array([0.4, 0.1, 0.2])
    flight_time = .45
    velocity = desired_release_velocity(release, target, flight_time)
    landed = release + velocity * flight_time + .5 * np.array([0.0, 0.0, -9.81]) * flight_time**2
    np.testing.assert_allclose(landed, target, atol=1e-10)
