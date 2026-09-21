import numpy as np
import torch

from rl_ppo.networks import ActorCritic
from rl_ppo.ppo import PPOConfig, RolloutBuffer, update


def test_residual_actor_starts_from_nominal_reference() -> None:
    model = ActorCritic(5, 3)
    with torch.no_grad():
        action, _, _, _ = model.act(torch.randn(4, 5), deterministic=True)
    torch.testing.assert_close(action, torch.zeros_like(action))


def test_ppo_update_accepts_squashed_actions() -> None:
    torch.manual_seed(3)
    model = ActorCritic(5, 2)
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = RolloutBuffer(4, 5, 2)
    rng = np.random.default_rng(5)
    for _ in range(4):
        observation = rng.normal(size=5).astype(np.float32)
        with torch.no_grad():
            _, raw_action, log_prob, value = model.act(torch.as_tensor(observation[None]))
        buffer.add(observation, raw_action.squeeze(0).numpy(), float(log_prob.item()), 0.1, False, float(value.item()))
    buffer.finish(last_value=0.0, config=config)
    metrics = update(model, optimizer, buffer, config, torch.device("cpu"))
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert 1 <= metrics["update_epochs"] <= config.update_epochs


def test_masked_actions_are_zero_and_do_not_affect_policy_probability() -> None:
    torch.manual_seed(11)
    model = ActorCritic(4, 3)
    observation = torch.randn(1, 4)
    mask = torch.tensor([1.0, 1.0, 0.0])
    with torch.no_grad():
        action, raw_action, _, _ = model.act(
            observation, deterministic=True, action_mask=mask
        )
        log_prob_before, entropy_before, _ = model.evaluate(
            observation, raw_action, mask
        )
        model.actor[-1].bias[2] += 100.0
        log_prob_after, entropy_after, _ = model.evaluate(
            observation, raw_action, mask
        )
    assert action[0, 2] == 0.0
    assert raw_action[0, 2] == 0.0
    torch.testing.assert_close(log_prob_after, log_prob_before)
    torch.testing.assert_close(entropy_after, entropy_before)
