"""Four-action residual PPO: frozen BC execution, separate trainable correction head."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rl_ppo.networks import ActorCritic
from .continuous_env import ACTOR_DIM, OBS_DIM, RESIDUAL_SIZE, SCHEMA
from .env import ACTOR_SIZE


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
        raise ValueError("V16 checkpoint source/schema/BC anchor mismatch")
    model = ResidualPolicy()
    model.load_state_dict(payload["model"])
    return model, payload


def update_policy(model, optimizer, buffer, config, mean_weight=.005):
    size = buffer.size
    obs = torch.tensor(buffer.observations[:size])
    raw = torch.tensor(buffer.raw_actions[:size])
    masks = torch.tensor(buffer.action_masks[:size])
    old = torch.tensor(buffer.log_probs[:size])
    returns = torch.tensor(buffer.returns[:size])
    advantage = torch.tensor(buffer.advantages[:size])
    active = masks.sum(-1)>0
    if not active.any():
        raise ValueError("No effective residual actions")
    advantage = (advantage-advantage[active].mean())/advantage[active].std(unbiased=False).clamp_min(1e-8)
    records = []
    stopped = False
    kl_trigger = 0.
    for epoch in range(config.update_epochs):
        for idx in torch.randperm(size).split(config.minibatch_size):
            logp, entropy, value = model.evaluate(obs[idx],raw[idx],masks[idx])
            valid = active[idx]
            ratio = (logp-old[idx]).exp()
            kl = (((ratio-1)-(logp-old[idx]))[valid].mean() if valid.any() else value.sum()*0.)
            if kl.item()>config.target_kl:
                stopped, kl_trigger = True, kl.item()
                break
            surrogate = torch.minimum(ratio*advantage[idx],
                ratio.clamp(1-config.clip_ratio,1+config.clip_ratio)*advantage[idx])
            policy_loss = -surrogate[valid].mean() if valid.any() else value.sum()*0.
            means = model.distribution(obs[idx]).mean.tanh()
            regularizer = (means.square()*masks[idx]).sum()/masks[idx].sum().clamp_min(1.)
            value_loss = .5*(value-returns[idx]).square().mean()
            actor_loss = policy_loss+mean_weight*regularizer
            if valid.any():
                actor_loss -= config.entropy_coefficient*entropy[valid].mean()
            loss = actor_loss+config.value_coefficient*value_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            actor_norm = nn.utils.clip_grad_norm_([*model.actor.parameters(),model.log_std],config.max_grad_norm)
            critic_norm = nn.utils.clip_grad_norm_(model.critic.parameters(),config.max_grad_norm)
            optimizer.step()
            clip_fraction = ((ratio[valid]-1).abs()>config.clip_ratio).float().mean() if valid.any() else value.sum()*0.
            records.append([x.item() for x in (policy_loss,value_loss,regularizer,kl,actor_norm,critic_norm,clip_fraction)])
        if stopped:
            break
    if not records:
        raise RuntimeError("Stopped PPO before first gradient update")
    with torch.no_grad():
        predicted = model.critic(model.normalize(obs)).squeeze(-1)
        explained = 1.-(returns-predicted).var(unbiased=False)/returns.var(unbiased=False).clamp_min(1e-8)
    result = dict(zip(("policy_loss","value_loss","residual_mean_square","approx_kl",
        "actor_grad_norm","critic_grad_norm","clip_fraction"),np.mean(records,axis=0).tolist()))
    return dict(**result, kl_early_stop=stopped, kl_trigger=kl_trigger, epochs=epoch+1,
        samples=size, active_samples=int(active.sum()), explained_variance=explained.item(),
        action_std=model.log_std.clamp(-3.,-.5).exp().detach().tolist())
