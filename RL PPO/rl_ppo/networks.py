"""New Torch actor-critic architecture for the ball-catch PPO task."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(input_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 384), nn.SiLU(),
        nn.Linear(384, 384), nn.SiLU(),
        nn.Linear(384, 384), nn.SiLU(),
        nn.Linear(384, output_size),
    )


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.actor = _mlp(observation_size, action_size)
        self.critic = _mlp(observation_size, 1)
        # Residual actions start at the validated nominal reference instead of
        # perturbing it with arbitrary output-layer initialization.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_size,), -1.2))

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.actor(observation)
        std = self.log_std.clamp(-2.5, -.5).exp()
        return Normal(mean, std.expand_as(mean))

    @staticmethod
    def _action_mask(action_mask: torch.Tensor | None,
                     reference: torch.Tensor) -> torch.Tensor:
        if action_mask is None:
            return torch.ones_like(reference)
        mask = action_mask.to(device=reference.device, dtype=reference.dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        try:
            return mask.expand_as(reference)
        except RuntimeError as error:
            raise ValueError("action mask is not broadcastable to the action tensor") from error

    @staticmethod
    def _squashed_log_probability(distribution: Normal, raw_action: torch.Tensor,
                                  action_mask: torch.Tensor | None = None) -> torch.Tensor:
        action = raw_action.tanh()
        correction = torch.log(1.0 - action.square() + 1e-6)
        terms = distribution.log_prob(raw_action) - correction
        if action_mask is not None:
            terms = terms * action_mask
        return terms.sum(dim=-1)

    def act(self, observation: torch.Tensor, deterministic: bool = False,
            action_mask: torch.Tensor | None = None):
        distribution = self.distribution(observation)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        mask = self._action_mask(action_mask, raw_action)
        raw_action = raw_action * mask
        return (
            raw_action.tanh(), raw_action,
            self._squashed_log_probability(distribution, raw_action, mask),
            self.critic(observation).squeeze(-1),
        )

    def evaluate(self, observation: torch.Tensor, raw_action: torch.Tensor,
                 action_mask: torch.Tensor | None = None):
        distribution = self.distribution(observation)
        mask = self._action_mask(action_mask, raw_action)
        return (
            self._squashed_log_probability(distribution, raw_action, mask),
            (distribution.entropy() * mask).sum(dim=-1),
            self.critic(observation).squeeze(-1),
        )
