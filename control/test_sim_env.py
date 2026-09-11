import numpy as np
import pytest

from control.sim_env import ControlEnv
from digital_twin.replay import PA_PER_PSI


def test_observations_are_causal_and_sync_has_one_edge_per_command():
    env = ControlEnv(seed=91)
    obs = env.reset()
    assert not np.array_equal(obs["q"], obs["q_true"])
    for _ in range(25):
        obs = env.step(np.full(24, 4 * PA_PER_PSI))
        assert obs["mocap_time_s"] <= max(0, obs["t"] - env.latency_s) + 1e-12
    assert all(n.sync_count == 25 for n in env.arm.nodes.values())
    assert obs["mocap_age_s"] < env.latency_s + env.camera_dt + 1e-10
    assert not np.array_equal(obs["p_pa"], obs["p_true_pa"])


def test_repeatability_and_randomization_are_separate():
    outputs = []
    for randomized in (False, False, True):
        env = ControlEnv(seed=41, randomize=randomized)
        env.reset()
        p = np.full(24, 2 * PA_PER_PSI)
        p[0] = 10 * PA_PER_PSI
        for _ in range(40):
            obs = env.step(p)
        outputs.append(obs)
    for key in ("q", "qdot", "p_pa", "q_true"):
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
    assert np.linalg.norm(outputs[0]["q_true"] - outputs[2]["q_true"]) > 1e-7


def test_pressure_caps_are_enforced_before_and_after_adc_quantization():
    env = ControlEnv()
    env.reset()
    for invalid in (np.full(24, 16 * PA_PER_PSI), np.full(24, np.nan)):
        with pytest.raises(AssertionError):
            env.step(invalid)
    env.step(np.full(24, 15 * PA_PER_PSI))
    env.envelope.assert_safe(env.last_targets_pa / PA_PER_PSI)
    assert all(n.sync_count == 1 for n in env.arm.nodes.values())
    p = np.zeros(24)
    p[8] = 30 * PA_PER_PSI
    env.step(p)
    env.envelope.assert_safe(np.maximum(env.last_targets_pa, 0) / PA_PER_PSI)


def test_batched_fitted_flow_preserves_the_plant():
    batched = ControlEnv(seed=72, noise_std_rad=0)
    scalar = ControlEnv(seed=72, noise_std_rad=0)
    batched.reset()
    scalar.reset()
    scalar.arm.batched_actuator = False
    p = np.full(24, 2 * PA_PER_PSI)
    p[0::3] = 12 * PA_PER_PSI
    for _ in range(45):
        a, b = batched.step(p), scalar.step(p)
    np.testing.assert_allclose(a["q_true"], b["q_true"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(a["p_true_pa"], b["p_true_pa"], atol=1e-7, rtol=1e-10)


def test_force_clip_at_intermediate_physics_step_invalidates_episode(monkeypatch):
    env = ControlEnv(seed=7)
    env.reset()
    original = env.arm.actuator.force_n
    calls = [0]

    def transient_clip(*args, **kwargs):
        force = original(*args, **kwargs).copy()
        calls[0] += 1
        if calls[0] == 2:
            force[0] = -4000.0
        return force

    monkeypatch.setattr(env.arm.actuator, "force_n", transient_clip)
    with pytest.raises(RuntimeError, match="force clip"):
        env.step(np.full(24, 2*PA_PER_PSI))
    assert calls[0] > 2
    assert np.max(np.abs(env.arm.data.ctrl)) < 4000
    assert env.max_force_n == 4000


def test_common_observer_does_not_change_existing_sensor_recurrence():
    env = ControlEnv(seed=62)
    env.reset()
    previous_q, previous_stamp = env._q.copy(), env._frame_time
    velocity = np.zeros(12)
    original_update = env.observer.update

    def checked_update(stamp, q):
        nonlocal previous_q, previous_stamp, velocity
        delta = stamp-previous_stamp
        alpha = -np.expm1(-delta/env.velocity_tau_s)
        velocity += alpha*((q-previous_q)/delta-velocity)
        result = original_update(stamp, q)
        np.testing.assert_array_equal(result, velocity)
        previous_q, previous_stamp = q.copy(), stamp
        return result

    env.observer.update = checked_update
    for _ in range(20):
        obs = env.step(np.full(24, 3*PA_PER_PSI))
    np.testing.assert_array_equal(obs["qdot"], velocity)
