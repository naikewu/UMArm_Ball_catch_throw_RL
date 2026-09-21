"""Fresh clipped-PPO and GAE update code; no external RL implementation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .networks import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = .995
    gae_lambda: float = .95
    clip_ratio: float = .16
    value_coefficient: float = .5
    entropy_coefficient: float = .0015
    learning_rate: float = 1.2e-4
    update_epochs: int = 5
    minibatch_size: int = 128
    max_grad_norm: float = .8
    target_kl: float = .018


class RolloutBuffer:
    def __init__(self, steps: int, observation_size: int, action_size: int):
        self.observations = np.zeros((steps, observation_size), dtype=np.float32)
        self.raw_actions = np.zeros((steps, action_size), dtype=np.float32)
        self.action_masks = np.ones((steps, action_size), dtype=np.float32)
        self.log_probs = np.zeros(steps, dtype=np.float32)
        self.rewards = np.zeros(steps, dtype=np.float32)
        self.dones = np.zeros(steps, dtype=np.float32)
        self.values = np.zeros(steps, dtype=np.float32)
        self.advantages = np.zeros(steps, dtype=np.float32)
        self.returns = np.zeros(steps, dtype=np.float32)
        self.size = 0

    def add(self, observation, raw_action, log_prob, reward, done, value,
            action_mask=None) -> None:
        i = self.size
        if i >= len(self.rewards):
            raise RuntimeError("rollout buffer is full")
        self.observations[i] = observation
        self.raw_actions[i] = raw_action
        if action_mask is not None:
            self.action_masks[i] = action_mask
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.size += 1

    def finish(self, last_value: float, config: PPOConfig) -> None:
        advantage = 0.0
        for i in range(self.size - 1, -1, -1):
            next_value = last_value if i == self.size - 1 else self.values[i + 1]
            nonterminal = 1.0 - self.dones[i]
            delta = self.rewards[i] + config.gamma * next_value * nonterminal - self.values[i]
            advantage = delta + config.gamma * config.gae_lambda * nonterminal * advantage
            self.advantages[i] = advantage
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]


def update(model: ActorCritic, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer,
           config: PPOConfig, device: torch.device) -> dict[str, float]:
    count = buffer.size
    if count == 0:
        raise ValueError("cannot update from an empty rollout")
    observations = torch.as_tensor(buffer.observations[:count], device=device)
    raw_actions = torch.as_tensor(buffer.raw_actions[:count], device=device)
    action_masks = torch.as_tensor(buffer.action_masks[:count], device=device)
    old_log_probs = torch.as_tensor(buffer.log_probs[:count], device=device)
    returns = torch.as_tensor(buffer.returns[:count], device=device)
    advantages = torch.as_tensor(buffer.advantages[:count], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    metrics = {
        "policy_loss": [], "value_loss": [], "entropy": [],
        "clip_fraction": [], "approx_kl": [],
    }
    epochs_completed = 0
    for _ in range(config.update_epochs):
        permutation = torch.randperm(count, device=device)
        epoch_kls: list[float] = []
        for indices in permutation.split(config.minibatch_size):
            log_probs, entropy, values = model.evaluate(
                observations[indices], raw_actions[indices], action_masks[indices]
            )
            log_ratio = log_probs - old_log_probs[indices]
            ratio = log_ratio.exp()
            surrogate_a = ratio * advantages[indices]
            surrogate_b = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[indices]
            policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
            value_loss = .5 * (returns[indices] - values).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics["policy_loss"].append(float(policy_loss.detach()))
            metrics["value_loss"].append(float(value_loss.detach()))
            metrics["entropy"].append(float(entropy_mean.detach()))
            metrics["clip_fraction"].append(float((torch.abs(ratio - 1.0) > config.clip_ratio).float().mean()))
            approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach())
            metrics["approx_kl"].append(approximate_kl)
            epoch_kls.append(approximate_kl)
        epochs_completed += 1
        if epoch_kls and float(np.mean(epoch_kls)) > config.target_kl:
            break
    result = {name: float(np.mean(values)) for name, values in metrics.items()}
    result["update_epochs"] = float(epochs_completed)
    return result
