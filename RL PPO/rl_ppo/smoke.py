"""Small deterministic end-to-end check for the environment and PPO update."""

from __future__ import annotations

import json

import torch

from .env import CatchEnvConfig, UMArmBallCatchEnv
from .networks import ActorCritic
from .ppo import PPOConfig, RolloutBuffer, update


def main() -> None:
    torch.manual_seed(7)
    environment = UMArmBallCatchEnv(CatchEnvConfig(randomized_twin=False))
    observation, _ = environment.reset(seed=7)
    home_tip_m = environment.home_tip_m.copy()
    model = ActorCritic(environment.observation_size, environment.action_size)
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    buffer = RolloutBuffer(4, environment.observation_size, environment.action_size)
    try:
        for _ in range(4):
            observation_tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            with torch.no_grad():
                action, raw_action, log_prob, value = model.act(observation_tensor)
            next_observation, reward, terminated, truncated, info = environment.step(action.squeeze(0).numpy())
            buffer.add(observation, raw_action.squeeze(0).numpy(), float(log_prob.item()), reward, terminated or truncated, float(value.item()))
            observation = next_observation
            if terminated or truncated:
                observation, _ = environment.reset(seed=8)
        with torch.no_grad():
            last_value = float(model.critic(torch.as_tensor(observation[None], dtype=torch.float32)).item())
        buffer.finish(last_value, config)
        metrics = update(model, optimizer, buffer, config, torch.device("cpu"))
    finally:
        environment.close()
    print(json.dumps({
        "reset_home_tip_m": home_tip_m.tolist(), "observation_size": int(observation.shape[0]),
        "action_size": environment.action_size, "last_distance_m": float(info["distance_m"]),
        "policy_loss": metrics["policy_loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
