"""Train a fresh PPO policy for the current-twin virtual ball-catch task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .env import CatchEnvConfig, UMArmBallCatchEnv
from .networks import ActorCritic
from .ppo import PPOConfig, RolloutBuffer, update


def save_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer,
                    *, observation_size: int, action_size: int, update_index: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "umarm_ball_catch_ppo_v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "observation_size": observation_size,
                "action_size": action_size, "update": update_index, "config": config}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic-twin", action="store_true")
    args = parser.parse_args()
    if args.total_steps <= 0 or args.rollout_steps <= 1:
        parser.error("total steps must be positive and rollout steps must exceed one")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    run_config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    env = UMArmBallCatchEnv(CatchEnvConfig(randomized_twin=not args.deterministic_twin))
    observation, _ = env.reset(args.seed)
    model = ActorCritic(env.observation_size, env.action_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPOConfig().learning_rate)
    config = PPOConfig()
    history, episode_returns, episode_lengths = [], [], []
    episode_return, episode_length, episodes, catches = 0.0, 0, 0, 0
    started = time.perf_counter()
    try:
        for update_index, start in enumerate(range(0, args.total_steps, args.rollout_steps), start=1):
            steps = min(args.rollout_steps, args.total_steps - start)
            buffer = RolloutBuffer(steps, env.observation_size, env.action_size)
            for _ in range(steps):
                with torch.no_grad():
                    tensor = torch.as_tensor(observation[None], device=device)
                    action, raw_action, log_prob, value = model.act(tensor)
                next_observation, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                done = terminated or truncated
                buffer.add(observation, raw_action[0].cpu().numpy(), float(log_prob.item()), reward, float(done), float(value.item()))
                observation = next_observation
                episode_return += reward
                episode_length += 1
                if done:
                    episodes += 1
                    catches += int(info["caught"])
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    observation, _ = env.reset(args.seed + episodes)
                    episode_return, episode_length = 0.0, 0
            with torch.no_grad():
                last_value = float(model.critic(torch.as_tensor(observation[None], device=device)).item())
            buffer.finish(last_value, config)
            metrics = update(model, optimizer, buffer, config, device)
            row = {"update": update_index, "steps": start + steps, "episodes": episodes,
                   "catch_rate": catches / max(episodes, 1),
                   "mean_completed_return": float(np.mean(episode_returns[-20:])) if episode_returns else float("nan"),
                   "mean_completed_length": float(np.mean(episode_lengths[-20:])) if episode_lengths else float("nan"),
                   **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            save_checkpoint(args.out_dir / "ppo_latest.pt", model, optimizer,
                            observation_size=env.observation_size, action_size=env.action_size,
                            update_index=update_index, config=run_config)
    finally:
        env.close()
    report = {"config": run_config, "elapsed_s": time.perf_counter() - started, "history": history,
              "episodes": episodes, "catches": catches}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "training_history.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
