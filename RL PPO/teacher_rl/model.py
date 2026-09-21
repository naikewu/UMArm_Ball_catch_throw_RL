"""Sensor-only actor, privileged critic, and fixed training-set normalization."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rl_ppo.networks import ActorCritic
from .env import ACTOR_SIZE, OBS_SIZE, ACTION_SIZE, SCHEMA


class IntentPolicy(ActorCritic):
    def __init__(self):
        nn.Module.__init__(self)
        self.actor = nn.Sequential(nn.Linear(ACTOR_SIZE, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, ACTION_SIZE))
        self.critic = nn.Sequential(nn.Linear(OBS_SIZE, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -3.))
        self.register_buffer("obs_mean", torch.zeros(OBS_SIZE))
        self.register_buffer("obs_std", torch.ones(OBS_SIZE))

    def normalize(self, obs):
        return ((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-10,10)

    def distribution(self, observation):
        mean = self.actor(self.normalize(observation)[..., :ACTOR_SIZE])
        return Normal(mean, self.log_std.clamp(-4., -1.5).exp().expand_as(mean))

    def act(self, observation, deterministic=False, action_mask=None):
        dist = self.distribution(observation)
        raw = dist.mean if deterministic else dist.sample()
        logp = self._squashed_log_probability(dist, raw, action_mask)
        value = self.critic(self.normalize(observation)).squeeze(-1)
        return raw.tanh(), raw, logp, value

    def evaluate(self, observation, raw_action, action_mask=None):
        dist = self.distribution(observation)
        mask = self._action_mask(action_mask, raw_action)
        return (self._squashed_log_probability(dist, raw_action, mask),
            (dist.entropy()*mask).sum(-1), self.critic(self.normalize(observation)).squeeze(-1))


def save_checkpoint(path, model, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(schema=getattr(model, "schema", SCHEMA), model=model.state_dict(), **extra), temporary)
    temporary.replace(path)


def load_checkpoint(path, *, expected_schema=SCHEMA):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != expected_schema:
        raise ValueError("checkpoint belongs to another environment or action schema")
    model = IntentPolicy()
    model.schema = expected_schema
    model.load_state_dict(payload["model"])
    return model, payload
