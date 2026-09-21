"""Compact one-decision trajectory selector for V26."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .anchored_dynamic_env import TRAJECTORY_ACTIONS
from .contextual_env import CONTEXT_NAMES, normalized_action


SCHEMA = "can_v15_static_terminal_selector_v26"
CONTEXT_SIZE = len(CONTEXT_NAMES)
ACTION_COUNT = len(TRAJECTORY_ACTIONS)
OUTCOME_SIZE = 5  # release logit, hit logit, utility, error, release time


class StaticTerminalPolicy(nn.Module):
    def __init__(self, observation_mean=None, observation_scale=None, hidden=32):
        super().__init__()
        mean = np.zeros(CONTEXT_SIZE, dtype=np.float32) if observation_mean is None else np.asarray(
            observation_mean, dtype=np.float32)
        scale = np.ones(CONTEXT_SIZE, dtype=np.float32) if observation_scale is None else np.asarray(
            observation_scale, dtype=np.float32)
        if mean.shape != (CONTEXT_SIZE,) or scale.shape != (CONTEXT_SIZE,) or np.any(scale <= 0):
            raise ValueError("invalid V26 static-selector normalization")
        self.register_buffer("observation_mean", torch.tensor(mean))
        self.register_buffer("observation_scale", torch.tensor(scale))
        actions = np.vstack([normalized_action(force, radius)
                             for force, radius in TRAJECTORY_ACTIONS]).astype(np.float32)
        self.register_buffer("action_parameters", torch.tensor(actions))
        self.encoder = nn.Sequential(nn.Linear(CONTEXT_SIZE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, ACTION_COUNT)
        self.critic = nn.Linear(hidden, 1)
        self.outcome = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Tanh(),
                                     nn.Linear(hidden, OUTCOME_SIZE))

    def encode(self, observations):
        return self.encoder((observations-self.observation_mean)/self.observation_scale)

    def forward(self, observations):
        encoded = self.encode(observations)
        batch = encoded.shape[0]
        expanded = encoded[:, None, :].expand(batch, ACTION_COUNT, encoded.shape[-1])
        actions = self.action_parameters[None, :, :].expand(batch, ACTION_COUNT, 2)
        return self.actor(encoded), self.critic(encoded).squeeze(-1), self.outcome(
            torch.cat((expanded, actions), dim=-1))

    def distribution(self, observations):
        logits, _, _ = self(observations)
        return torch.distributions.Categorical(logits=logits)

    def act(self, context, deterministic=True):
        value = np.asarray(context, dtype=np.float32)
        if value.shape != (CONTEXT_SIZE,) or not np.isfinite(value).all():
            raise ValueError("V26 static-selector context is invalid")
        with torch.no_grad():
            logits, critic, outcomes = self(torch.tensor(value[None]))
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
        index = int(action.item())
        return index, float(distribution.log_prob(action).item()), float(critic.item()), \
            outcomes[0, index].numpy()


def save_policy(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = dict(metadata)
    payload.update(schema=SCHEMA, state_dict=model.state_dict(),
        hidden=int(model.encoder[0].weight.shape[0]))
    torch.save(payload, temporary)
    temporary.replace(path)


def load_policy(path, expected_schema=SCHEMA):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != expected_schema:
        raise ValueError("V26 static-selector checkpoint schema mismatch")
    state = payload["state_dict"]
    model = StaticTerminalPolicy(state["observation_mean"].numpy(),
        state["observation_scale"].numpy(), int(payload["hidden"]))
    model.load_state_dict(state)
    model.eval()
    return model, payload
