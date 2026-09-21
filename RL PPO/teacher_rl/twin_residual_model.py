"""Compact V25 candidate actor/critic with no learned release gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .anchored_dynamic_env import TRAJECTORY_ACTIONS
from .contextual_env import CONTEXT_NAMES
from .twin_residual_env import JOINT_ACTIONS, OBSERVATION_NAMES


OBS_SIZE = len(OBSERVATION_NAMES)
CONTEXT_SIZE = len(CONTEXT_NAMES)


def _network(input_size, hidden, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, output_size))


def _masked_logits(logits, masks):
    masks = masks.to(dtype=torch.bool, device=logits.device)
    if masks.shape != logits.shape or not masks.any(-1).all():
        raise ValueError("each V25 action mask must contain at least one valid action")
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)


class TwinResidualPolicy(nn.Module):
    """Nine nominal trajectories and nine bounded online joint adjustments."""

    def __init__(self, observation_mean=None, observation_scale=None, hidden=32):
        super().__init__()
        mean = np.zeros(OBS_SIZE, dtype=np.float32)
        scale = np.ones(OBS_SIZE, dtype=np.float32)
        if observation_mean is not None:
            mean = np.asarray(observation_mean, dtype=np.float32)
            scale = np.asarray(observation_scale, dtype=np.float32)
            if (mean.shape != (OBS_SIZE,) or scale.shape != (OBS_SIZE,) or
                    not np.isfinite(np.r_[mean, scale]).all() or np.any(scale <= 0)):
                raise ValueError("V25 observation normalization is invalid")
        self.register_buffer("obs_mean", torch.tensor(mean))
        self.register_buffer("obs_scale", torch.tensor(np.maximum(scale, .05)))
        self.nominal_actor = _network(CONTEXT_SIZE, hidden, len(TRAJECTORY_ACTIONS))
        self.motion_actor = _network(OBS_SIZE, hidden, len(JOINT_ACTIONS))
        self.critic = _network(OBS_SIZE, hidden, 1)
        self._initialize_motion_hold()

    def _initialize_motion_hold(self):
        nn.init.zeros_(self.motion_actor[-1].weight)
        nn.init.zeros_(self.motion_actor[-1].bias)
        self.motion_actor[-1].bias.data[JOINT_ACTIONS.index((0, 0))] = 1.
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def normalized(self, observations):
        return ((observations-self.obs_mean)/self.obs_scale.clamp_min(.05)).clamp(-10., 10.)

    def nominal_distribution(self, observations):
        value = self.normalized(observations)[..., :CONTEXT_SIZE]
        return Categorical(logits=self.nominal_actor(value))

    def motion_distribution(self, observations, masks):
        logits = self.motion_actor(self.normalized(observations))
        return Categorical(logits=_masked_logits(logits, masks))

    def value(self, observations):
        return self.critic(self.normalized(observations)).squeeze(-1)

    @torch.no_grad()
    def act_nominal(self, observation, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(1, OBS_SIZE)
        distribution = self.nominal_distribution(observation)
        action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(
            self.value(observation).item())

    @torch.no_grad()
    def act_motion(self, observation, mask, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(1, OBS_SIZE)
        mask = torch.as_tensor(mask, dtype=torch.bool).reshape(1, len(JOINT_ACTIONS))
        distribution = self.motion_distribution(observation, mask)
        action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(
            self.value(observation).item())

    def evaluate(self, observations, phases, actions, masks):
        logp = torch.zeros(len(observations), device=observations.device)
        entropy = torch.zeros_like(logp)
        nominal = phases == 0
        motion = ~nominal
        if nominal.any():
            distribution = self.nominal_distribution(observations[nominal])
            logp[nominal] = distribution.log_prob(actions[nominal])
            entropy[nominal] = distribution.entropy()
        if motion.any():
            distribution = self.motion_distribution(observations[motion], masks[motion])
            logp[motion] = distribution.log_prob(actions[motion])
            entropy[motion] = distribution.entropy()
        return logp, entropy, self.value(observations)

    def kl_from(self, reference, observations, phases, masks):
        result = torch.zeros(len(observations), device=observations.device)
        nominal = phases == 0
        motion = ~nominal
        if nominal.any():
            old = reference.nominal_distribution(observations[nominal])
            new = self.nominal_distribution(observations[nominal])
            result[nominal] = torch.distributions.kl_divergence(old, new)
        if motion.any():
            old = reference.motion_distribution(observations[motion], masks[motion])
            new = self.motion_distribution(observations[motion], masks[motion])
            result[motion] = torch.distributions.kl_divergence(old, new)
        return result


def save_policy(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(dict(state_dict=model.state_dict(), **metadata), temporary)
    temporary.replace(path)


def load_policy(path, expected_schema=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ValueError(f"checkpoint schema {payload.get('schema')!r} does not match {expected_schema!r}")
    state = payload["state_dict"]
    hidden = int(state["nominal_actor.0.weight"].shape[0])
    model = TwinResidualPolicy(state["obs_mean"].numpy(), state["obs_scale"].numpy(), hidden)
    model.load_state_dict(state)
    return model, payload
