import numpy as np
import torch

from teacher_rl.anchored_dynamic_env import (INITIAL_ACTIONS, MOTION_NAMES,
    OBSERVATION_NAMES, RELEASE_NAMES, TRAJECTORY_ACTIONS, DynamicControlConfig,
    initial_action)
from teacher_rl.anchored_dynamic_model import (CONTEXT_SIZE, OBS_SIZE,
    AnchoredDynamicPolicy, load_policy, save_policy)
from teacher_rl.anchored_dynamic_rl import (_finish_episode, _phase_standardize,
    _sample_record)
from teacher_rl.contextual_env import bounded_parameters


def test_initial_catalogue_exactly_covers_v23_grid():
    assert len(INITIAL_ACTIONS) == 10
    assert INITIAL_ACTIONS[0] is None
    actual = []
    for index in range(1, 10):
        parameters = bounded_parameters(initial_action(index))
        actual.append((parameters["envelope_force_scale"],
                       parameters["envelope_radius_scale"]))
    assert np.allclose(actual, TRAJECTORY_ACTIONS)
    assert len(OBSERVATION_NAMES) == 51


def test_dynamic_policy_defaults_to_hold_and_guarded_release(tmp_path):
    model = AnchoredDynamicPolicy(np.zeros(CONTEXT_SIZE), np.ones(CONTEXT_SIZE))
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    motion, release, _, value = model.act_dynamic(observation, deterministic=True)
    assert MOTION_NAMES[motion] == "hold"
    assert RELEASE_NAMES[release] == "allow_safe_release"
    assert value == 0.
    path = tmp_path / "policy.pt"
    save_policy(path, model, schema="test", kind="test")
    restored, payload = load_policy(path, "test")
    assert payload["kind"] == "test"
    assert restored.act_dynamic(observation, deterministic=True)[:2] == (motion, release)


def test_hybrid_evaluation_backpropagates_through_both_policy_phases():
    model = AnchoredDynamicPolicy(np.zeros(CONTEXT_SIZE), np.ones(CONTEXT_SIZE))
    observations = torch.randn(6, OBS_SIZE)
    phases = torch.tensor([0, 1, 1, 0, 1, 0])
    initial = torch.tensor([0, 0, 0, 1, 0, 2])
    motion = torch.tensor([0, 1, 2, 0, 3, 0])
    release = torch.tensor([1, 0, 1, 1, 1, 1])
    logp, entropy, value = model.evaluate(
        observations, phases, initial, motion, release)
    loss = -(logp + .01 * entropy).mean() + value.square().mean()
    loss.backward()
    assert model.initial_actor[-1].weight.grad.abs().sum() > 0
    assert model.motion_head.weight.grad.abs().sum() > 0
    assert model.release_head.weight.grad is None


def test_deterministic_policy_starts_with_exact_v15_fallback():
    model = AnchoredDynamicPolicy(np.zeros(CONTEXT_SIZE), np.ones(CONTEXT_SIZE))
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    action, _, _ = model.act_initial(observation, deterministic=True)
    assert action == 0


def test_low_frequency_policy_cannot_block_calibrated_release_guard():
    model = AnchoredDynamicPolicy(np.zeros(CONTEXT_SIZE), np.ones(CONTEXT_SIZE))
    model.release_head.bias.data[:] = torch.tensor([100., -100.])
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    assert model.act_dynamic(observation, deterministic=True)[1] == 1


def test_terminal_credit_discards_decisions_after_release_window():
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    samples = [_sample_record(observation, phase, 0, 0, 1, 0., 0., time_s)
               for phase, time_s in ((0, 0.), (1, 1.), (1, 19.))]
    result = dict(captured=True, grip_broken=False, max_joint_deg=20., released=False,
        landing_error_m=None, calibrated_release=dict(min_calibrated_error_m=.2),
        trajectory_decision=dict(time_s=0.), contextual_config=dict(release_max_s=18.))
    finished = _finish_episode(samples, result, gamma=.98, gae_lambda=.95)
    assert len(finished) == 2
    assert "terminal_reward" in finished[-1]
    assert all("advantage" in sample and "return" in sample for sample in finished)


def test_advantages_are_normalized_separately_for_initial_and_dynamic_phases():
    values = torch.tensor([1., 3., 100., 104.])
    phases = torch.tensor([0, 0, 1, 1])
    normalized = _phase_standardize(values, phases)
    torch.testing.assert_close(normalized[:2], torch.tensor([-1., 1.]))
    torch.testing.assert_close(normalized[2:], torch.tensor([-1., 1.]))


def test_dynamic_control_bounds_reject_unsafe_tolerance():
    DynamicControlConfig(release_tolerance_m=.08)
    try:
        DynamicControlConfig(release_tolerance_m=.081)
    except ValueError:
        pass
    else:
        raise AssertionError("release tolerance above the audited 8 cm bound was accepted")
