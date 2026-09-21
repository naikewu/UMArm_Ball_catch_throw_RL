"""Small hybrid actor/critic for V15-anchored post-catch control."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .anchored_dynamic_env import (INITIAL_ACTIONS, MOTION_NAMES,
    OBSERVATION_NAMES, RELEASE_NAMES)
from .contextual_env import CONTEXT_NAMES


OBS_SIZE = len(OBSERVATION_NAMES)
CONTEXT_SIZE = len(CONTEXT_NAMES)
TAKEOVER_CONFIDENCE = .55


def _network(input_size, hidden, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, output_size))


class AnchoredDynamicPolicy(nn.Module):
    """An absolute initialization head plus low-frequency motion/release heads."""

    def __init__(self, context_mean=None, context_scale=None, hidden=48):
        super().__init__()
        mean = np.zeros(OBS_SIZE, dtype=np.float32)
        scale = np.ones(OBS_SIZE, dtype=np.float32)
        if context_mean is not None:
            context_mean = np.asarray(context_mean, dtype=np.float32)
            context_scale = np.asarray(context_scale, dtype=np.float32)
            if (context_mean.shape != (CONTEXT_SIZE,) or
                    context_scale.shape != (CONTEXT_SIZE,) or
                    not np.isfinite(np.r_[context_mean, context_scale]).all() or
                    np.any(context_scale <= 0)):
                raise ValueError("context normalization is invalid")
            mean[:CONTEXT_SIZE] = context_mean
            scale[:CONTEXT_SIZE] = np.maximum(context_scale, .05)
        self.register_buffer("obs_mean", torch.tensor(mean))
        self.register_buffer("obs_scale", torch.tensor(scale))
        self.initial_actor = _network(CONTEXT_SIZE, hidden, len(INITIAL_ACTIONS))
        self.dynamic_trunk = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.Tanh(),
                                           nn.Linear(hidden, hidden), nn.Tanh())
        self.motion_head = nn.Linear(hidden, len(MOTION_NAMES))
        self.release_head = nn.Linear(hidden, len(RELEASE_NAMES))
        self.critic = _network(OBS_SIZE, hidden, 1)
        self._initialize_safe_dynamic_default()

    def _initialize_safe_dynamic_default(self):
        # Pretraining data contains absolute actions rather than action sequences.
        # Begin the dynamic policy at the closest equivalent: keep parameters and
        # allow only the separately calibrated safe release mechanism.
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)
        self.motion_head.bias.data[0] = 1.
        nn.init.zeros_(self.release_head.weight)
        nn.init.zeros_(self.release_head.bias)
        self.release_head.bias.data[1] = 2.5
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def normalized(self, observations):
        return ((observations - self.obs_mean) / self.obs_scale.clamp_min(.05)).clamp(-10., 10.)

    def initial_distribution(self, observations):
        value = self.normalized(observations)[..., :CONTEXT_SIZE]
        return Categorical(logits=self.initial_actor(value))

    def dynamic_distributions(self, observations):
        hidden = self.dynamic_trunk(self.normalized(observations))
        return Categorical(logits=self.motion_head(hidden)), Categorical(
            logits=self.release_head(hidden))

    def value(self, observations):
        return self.critic(self.normalized(observations)).squeeze(-1)

    @torch.no_grad()
    def act_initial(self, observation, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(1, OBS_SIZE)
        distribution = self.initial_distribution(observation)
        if deterministic:
            candidate = distribution.probs[..., 1:].argmax(-1) + 1
            candidate_probability = distribution.probs.gather(-1, candidate.unsqueeze(-1)).squeeze(-1)
            action = torch.where(candidate_probability >= TAKEOVER_CONFIDENCE,
                                 candidate, torch.zeros_like(candidate))
        else:
            action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(
            self.value(observation).item())

    @torch.no_grad()
    def act_dynamic(self, observation, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(1, OBS_SIZE)
        motion, _ = self.dynamic_distributions(observation)
        if deterministic:
            motion_action = motion.probs.argmax(-1)
        else:
            motion_action = motion.sample()
        # The calibrated 150 Hz guard owns release safety.  A low-frequency
        # wait action can miss a one-tick candidate, so PPO no longer gates it.
        release_action = torch.ones_like(motion_action)
        log_probability = motion.log_prob(motion_action)
        return (int(motion_action.item()), int(release_action.item()),
                float(log_probability.item()), float(self.value(observation).item()))

    def evaluate(self, observations, phases, initial_actions, motion_actions,
                 release_actions):
        count = len(observations)
        log_probability = torch.zeros(count, device=observations.device)
        entropy = torch.zeros(count, device=observations.device)
        initial_mask = phases == 0
        dynamic_mask = ~initial_mask
        if initial_mask.any():
            distribution = self.initial_distribution(observations[initial_mask])
            log_probability[initial_mask] = distribution.log_prob(initial_actions[initial_mask])
            entropy[initial_mask] = distribution.entropy()
        if dynamic_mask.any():
            motion, _ = self.dynamic_distributions(observations[dynamic_mask])
            log_probability[dynamic_mask] = motion.log_prob(motion_actions[dynamic_mask])
            entropy[dynamic_mask] = motion.entropy()
        return log_probability, entropy, self.value(observations)

    def kl_from(self, reference, observations, phases):
        result = torch.zeros(len(observations), device=observations.device)
        initial_mask = phases == 0
        dynamic_mask = ~initial_mask
        if initial_mask.any():
            old = reference.initial_distribution(observations[initial_mask])
            new = self.initial_distribution(observations[initial_mask])
            result[initial_mask] = torch.distributions.kl_divergence(old, new)
        if dynamic_mask.any():
            old_motion, _ = reference.dynamic_distributions(observations[dynamic_mask])
            new_motion, _ = self.dynamic_distributions(observations[dynamic_mask])
            result[dynamic_mask] = torch.distributions.kl_divergence(old_motion, new_motion)
        return result


def save_policy(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state_dict=model.state_dict(), **metadata), temporary)
    temporary.replace(path)


def load_policy(path, expected_schema=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    schema = payload.get("schema")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(f"checkpoint schema {schema!r} does not match {expected_schema!r}")
    state = payload["state_dict"]
    model = AnchoredDynamicPolicy(
        state["obs_mean"][:CONTEXT_SIZE].numpy(),
        state["obs_scale"][:CONTEXT_SIZE].numpy(),
        hidden=int(state["initial_actor.0.weight"].shape[0]))
    model.load_state_dict(state)
    return model, payload
