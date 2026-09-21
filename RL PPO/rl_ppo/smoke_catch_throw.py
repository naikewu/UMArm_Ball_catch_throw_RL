"""Verify physical collision, MPPI handoff, and one PPO update in a short run."""
from __future__ import annotations

import json

import torch

from .catch_throw_env import CatchThrowEnv, CatchThrowEnvConfig
from .networks import ActorCritic
from .ppo import PPOConfig, RolloutBuffer, update


def main() -> None:
    torch.manual_seed(17)
    env = CatchThrowEnv(CatchThrowEnvConfig(mppi_samples=8, mppi_horizon=6, seed=17))
    observation, reset_info = env.reset(17)
    model = ActorCritic(env.observation_size, env.action_size)
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rollout_steps = 12
    buffer = RolloutBuffer(rollout_steps, env.observation_size, env.action_size)
    physical_contact_observed = False
    capture_observed = False
    peak_contact_force_n = 0.0
    try:
        for _ in range(rollout_steps):
            action_mask = env.action_mask()
            with torch.no_grad():
                action, raw_action, log_prob, value = model.act(
                    torch.as_tensor(observation[None], dtype=torch.float32),
                    action_mask=torch.as_tensor(action_mask),
                )
            next_observation, reward, terminated, truncated, info = env.step(action[0].numpy())
            physical_contact_observed |= bool(info["physical_contact_occurred"])
            capture_observed |= bool(info["captured"])
            peak_contact_force_n = max(peak_contact_force_n, float(info["contact_force_n"]))
            buffer.add(
                observation, raw_action[0].numpy(), float(log_prob.item()), reward,
                terminated or truncated, float(value.item()), action_mask,
            )
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset(18)
        with torch.no_grad():
            last_value = float(model.critic(torch.as_tensor(observation[None], dtype=torch.float32)).item())
        buffer.finish(last_value, config)
        metrics = update(model, optimizer, buffer, config, torch.device("cpu"))
        result = {"observation_size": env.observation_size, "action_size": env.action_size,
                  "physical_contact_scene": physical_contact_observed,
                  "capture_observed": capture_observed,
                  "intercept_world_m": reset_info["task_intercept_world_m"].tolist(),
                  "peak_contact_force_n": peak_contact_force_n,
                  "ppo_policy_loss": metrics["policy_loss"]}
    finally:
        env.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
