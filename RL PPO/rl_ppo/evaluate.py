"""Evaluate a trained PPO policy on the held-out catch benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import CatchEnvConfig, UMArmBallCatchEnv
from .networks import ActorCritic


def load_policy(checkpoint_path: Path, device: torch.device) -> ActorCritic:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("observation_size") != UMArmBallCatchEnv.observation_size:
        raise ValueError("Checkpoint observation dimension does not match this environment.")
    if payload.get("action_size") != UMArmBallCatchEnv.action_size:
        raise ValueError("Checkpoint action dimension does not match this environment.")
    model = ActorCritic(UMArmBallCatchEnv.observation_size, UMArmBallCatchEnv.action_size).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("eval"))
    parser.add_argument("--deterministic-twin", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    device = torch.device("cpu")
    policy = load_policy(args.checkpoint, device)
    environment = UMArmBallCatchEnv(CatchEnvConfig(randomized_twin=not args.deterministic_twin))
    returns: list[float] = []
    caught: list[bool] = []
    closest_distances: list[float] = []
    try:
        for episode in range(args.episodes):
            observation, _ = environment.reset(seed=args.seed + episode)
            total_reward, minimum_distance = 0.0, float("inf")
            terminated = truncated = False
            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, _, _, _ = policy.act(observation_tensor, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(action.squeeze(0).cpu().numpy())
                total_reward += reward
                minimum_distance = min(minimum_distance, float(info["distance_m"]))
            returns.append(total_reward)
            caught.append(bool(info["caught"]))
            closest_distances.append(minimum_distance)
    finally:
        environment.close()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(args.checkpoint), "episodes": args.episodes,
        "success_rate": float(np.mean(caught)), "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)), "mean_closest_distance_m": float(np.mean(closest_distances)),
        "virtual_catcher": True, "physical_ball_contact_model": False,
        "randomized_twin": not args.deterministic_twin,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
