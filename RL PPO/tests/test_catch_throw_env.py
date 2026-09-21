import numpy as np

from rl_ppo.catch_throw_env import CatchThrowEnv, CatchThrowEnvConfig
from rl_ppo.catch_throw_task import ballistic_state


def test_stage_one_has_a_physically_reachable_grasp() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(
        curriculum_stage=1, mppi_samples=8, mppi_horizon=6,
        grasp_mode="physical_dwell",
    ))
    try:
        observation, reset_info = env.reset(4)
        initial_cup = env.state["cup_position_m"].copy()
        horizontal_offset = np.linalg.norm(
            reset_info["task_intercept_world_m"][:2] - env.home_cup_world_m[:2]
        )
        incoming = reset_info["task_incoming_velocity_world_mps"]
        aperture_axis = env.state["cup_rotation"][:, 2]
        maximum_cup_displacement = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(12):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_size))
            maximum_cup_displacement = max(
                maximum_cup_displacement,
                float(np.linalg.norm(env.state["cup_position_m"] - initial_cup)),
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    assert .010 <= horizontal_offset <= .025
    assert np.dot(aperture_axis, incoming) / np.linalg.norm(incoming) > .98
    assert maximum_cup_displacement > .01
    assert observation.shape == (81,)
    assert np.isfinite(reset_info["predicted_entry_local_m"]).all()
    np.testing.assert_array_equal(env.action_mask(), np.r_[np.ones(6), np.zeros(4)])


def test_curriculum_action_masks_expand_by_stage() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        assert int(env.action_mask().sum()) == 6
        np.testing.assert_allclose(
            env._action_authority(), [.30] * 3 + [.50] * 3 + [0.0] * 4
        )
        env.set_curriculum(2, 1)
        assert int(env.action_mask().sum()) == 8
        np.testing.assert_allclose(
            env._action_authority(), [.50] * 3 + [.70] * 3 + [.55] * 2 + [0.0] * 2
        )
        env.set_curriculum(2, 2)
        np.testing.assert_allclose(
            env._action_authority(), [.70] * 3 + [.85] * 3 + [.75] * 2 + [0.0] * 2
        )
        env.set_curriculum(2, 3)
        np.testing.assert_allclose(
            env._action_authority(), [.90] * 3 + [1.0] * 3 + [.95] * 2 + [0.0] * 2
        )
        env.set_curriculum(3)
        assert int(env.action_mask().sum()) == 10
        np.testing.assert_allclose(
            env._action_authority(), [.80] * 3 + [.90] * 3 + [.85] * 2 + [.65] * 2
        )
        env.set_curriculum(4)
        np.testing.assert_allclose(
            env._action_authority(), [.90] * 3 + [1.0] * 3 + [1.0] * 2 + [.85] * 2
        )
    finally:
        env.close()


def test_v11_reference_authority_defaults() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        env.reset(4)
        assert env.reference.config.velocity_match_window_s == .75
        assert env.reference.config.contact_velocity_adjustment_mps == .45
    finally:
        env.close()


def test_auto_grasp_volume_tightens_across_stage_two() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(
        controller="ff_pid", curriculum_stage=2, stage2_level=1,
        grasp_mode="auto_volume",
    ))
    try:
        assert env._auto_grasp_radial_limit() == .025
        env.set_curriculum(2, 2)
        assert env._auto_grasp_radial_limit() == .022
        env.set_curriculum(2, 3)
        assert env._auto_grasp_radial_limit() == .020
    finally:
        env.close()


def test_stage_two_b_soft_quality_target_matches_promotion_gate() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(
        controller="ff_pid", curriculum_stage=2, stage2_level=2
    ))
    try:
        _, _, alignment_gate = env._grasp_thresholds()
        radial_target, speed_target, alignment_target = env._soft_grasp_quality_targets()
    finally:
        env.close()
    assert alignment_gate == .45
    assert radial_target == .025
    assert speed_target == .55
    assert alignment_target == .65


def test_finger_capsule_predictor_finds_outer_tip_collision() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=2))
    try:
        time_s, point, kind = env._predicted_finger_contact(
            np.array([.047, 0.0, -.10]), np.array([0.0, 0.0, 1.0]), .20
        )
    finally:
        env.close()
    assert time_s is not None and 0.0 < time_s < .10
    assert point is not None and np.isclose(point[0], .047)
    assert kind == "finger_0_tip"


def test_ballistic_entry_predictor_accounts_for_gravity_on_far_launch() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        _, info = env.reset(4)
        intercept_horizontal = np.linalg.norm(
            info["task_intercept_world_m"][:2] - env.home_cup_world_m[:2]
        )
        intercept_time = env.task.intercept_time_s
    finally:
        env.close()
    assert .010 <= intercept_horizontal <= .025
    assert .30 < intercept_time < .47


def test_rendezvous_reference_position_derivative_matches_velocity() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        env.reset(4)
        time_s = .55 * env.task.intercept_time_s
        epsilon = 1e-5
        before, _ = env.reference.nominal_catch_state(time_s - epsilon)
        after, _ = env.reference.nominal_catch_state(time_s + epsilon)
        _, velocity = env.reference.nominal_catch_state(time_s)
        position_at_intercept, velocity_at_intercept = (
            env.reference.nominal_catch_state(env.task.intercept_time_s)
        )
        incoming = ballistic_state(env.task, env.task.intercept_time_s)[1]
    finally:
        env.close()
    np.testing.assert_allclose((after - before) / (2 * epsilon), velocity, atol=1e-5)
    np.testing.assert_allclose(position_at_intercept, env.task.intercept_world_m, atol=1e-9)
    np.testing.assert_allclose(velocity_at_intercept, incoming, atol=1e-9)


def test_v10_low_level_preview_uses_horizon_times_plus_actuation_lead() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        env.reset(4)
        env.reference.set_action(
            np.zeros(env.action_size), time_s=0.0, q_now=env.state["q"],
            ball_velocity_world_mps=env.state["ball_velocity_mps"],
        )
        dt = 1 / env.config.plant_hz
        preview = env.reference.low_level_reference(
            q_now=env.state["q"], qdot_now=env.state["qdot"],
            force_world_n=np.zeros(3), dt=dt, horizon=8, time_s=0.0,
        )
        expected_tip = np.asarray([
            env.plant.world_to_base(env.reference.desired_catch_state(
                env.reference.config.actuation_lead_s + (index + 1) * dt
            )[0])
            for index in range(8)
        ])
    finally:
        env.close()
    assert preview["future_q"].shape == (8, 12)
    assert preview["future_qd"].shape == (8, 12)
    assert preview["future_tip_velocity"].shape == (8, 3)
    np.testing.assert_allclose(preview["future_tip"], expected_tip, atol=1e-9)
    assert np.linalg.norm(preview["future_tip"][-1] - preview["future_tip"][0]) > 1e-4


def test_early_curriculum_requires_a_stable_capture_hold() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        env.reset(4)
        env.captured = True
        env.grasp_time_s = env.state["time_s"]
        assert not env._stable_capture()
        env.state["time_s"] += env.config.capture_hold_s
        assert env._stable_capture()
    finally:
        env.close()


def test_parallel_compliance_is_never_stiffer_than_transverse() -> None:
    env = CatchThrowEnv(CatchThrowEnvConfig(controller="ff_pid", curriculum_stage=1))
    try:
        env.reset(9)
        for action_6 in (-1.0, 0.0, 1.0):
            action = np.zeros(env.action_size)
            action[6] = action_6
            action[7] = 1.0
            env.reference.set_action(
                action, time_s=0.0, q_now=env.state["q"],
                ball_velocity_world_mps=env.state["ball_velocity_mps"],
            )
            incoming = env.reference.contact_velocity_world_mps
            direction = env.plant.vector_world_to_base(incoming)
            direction /= np.linalg.norm(direction)
            compliance = env.reference.compliance_ref
            parallel = float(direction @ compliance @ direction)
            transverse = float((np.trace(compliance) - parallel) / 2)
            assert parallel >= transverse - 1e-12
            assert parallel <= env.reference.config.maximum_compliance_m_n + 1e-12
    finally:
        env.close()
