"""V17 seven-action residual PPO; checkpoint-incompatible with V16."""
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from rl_ppo.networks import ActorCritic
from .buffered_env import ACTOR_DIM, OBS_DIM, RESIDUAL_SIZE, SCHEMA
from .env import ACTOR_SIZE
from .continuous_model import update_policy


class ResidualPolicy(ActorCritic):
    def __init__(self, anchor=None):
        nn.Module.__init__(self)
        self.actor = nn.Sequential(nn.Linear(ACTOR_DIM,256), nn.SiLU(), nn.Linear(256,256),
            nn.SiLU(), nn.Linear(256,RESIDUAL_SIZE))
        self.critic = nn.Sequential(nn.Linear(OBS_DIM,256), nn.SiLU(), nn.Linear(256,256),
            nn.SiLU(), nn.Linear(256,1))
        self.log_std = nn.Parameter(torch.full((RESIDUAL_SIZE,), -1.2))
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))
        if anchor is not None:
            with torch.no_grad():
                self.actor[0].weight.zero_()
                self.actor[0].weight[:,:ACTOR_SIZE].copy_(anchor.actor[0].weight)
                self.actor[0].bias.copy_(anchor.actor[0].bias)
                self.actor[2].load_state_dict(anchor.actor[2].state_dict())
                self.obs_mean[:ACTOR_SIZE].copy_(anchor.obs_mean[:ACTOR_SIZE])
                self.obs_mean[ACTOR_DIM:].copy_(anchor.obs_mean[ACTOR_SIZE:])
                self.obs_std[:ACTOR_SIZE].copy_(anchor.obs_std[:ACTOR_SIZE])
                self.obs_std[ACTOR_DIM:].copy_(anchor.obs_std[ACTOR_SIZE:])
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def normalize(self, obs):
        return ((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-10,10)

    def distribution(self, obs):
        mean = self.actor(self.normalize(obs)[...,:ACTOR_DIM])
        return Normal(mean, self.log_std.clamp(-3.,-.5).exp().expand_as(mean))

    def act(self, observation, deterministic=False, action_mask=None):
        distribution = self.distribution(observation)
        raw = distribution.mean if deterministic else distribution.sample()
        mask = self._action_mask(action_mask, raw)
        raw = raw*mask
        return (raw.tanh(), raw, self._squashed_log_probability(distribution,raw,mask),
            self.critic(self.normalize(observation)).squeeze(-1))

    def evaluate(self, observations, raw, masks=None):
        distribution = self.distribution(observations)
        mask = self._action_mask(masks,raw)
        return (self._squashed_log_probability(distribution,raw,mask),
            (distribution.entropy()*mask).sum(-1), self.critic(self.normalize(observations)).squeeze(-1))


def save_policy(path, model, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(schema=SCHEMA, model=model.state_dict(), **extra), temporary)
    temporary.replace(path)


def load_policy(path, source_hash, anchor_sha256):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != SCHEMA or payload.get("source_hash") != source_hash or
            payload.get("anchor_sha256") != anchor_sha256):
        raise ValueError("V17 checkpoint source/schema/BC anchor mismatch")
    model = ResidualPolicy()
    model.load_state_dict(payload["model"])
    return model, payload
