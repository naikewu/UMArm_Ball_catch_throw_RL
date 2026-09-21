"""V23 observable-context value teacher and categorical BC/PPO policy.

Each category denotes an explicitly bounded force/radius pair, applied once per
captured episode.  This avoids unverified interpolation between orbit recipes.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .contextual_env import SCHEMA, CONTEXT_NAMES, bounded_parameters


class ContextPolicy(nn.Module):
    def __init__(self, actions):
        super().__init__()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) < 2:
            raise ValueError("V23 action catalogue must contain at least two force/radius pairs")
        for action in actions:
            bounded_parameters(action)
        n = len(CONTEXT_NAMES)
        self.register_buffer("actions", torch.tensor(actions))
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        self.actor = nn.Sequential(nn.Linear(n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, len(actions)))
        self.critic = nn.Sequential(nn.Linear(n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

    def normalize(self, contexts):
        return ((contexts - self.mean) / self.scale).clamp(-8., 8.)

    def distribution(self, contexts):
        return Categorical(logits=self.actor(self.normalize(contexts)))

    def value(self, contexts):
        return self.critic(self.normalize(contexts)).squeeze(-1)

    def decide(self, context, stochastic=False):
        tensor = torch.tensor(np.asarray(context), dtype=torch.float32)
        with torch.no_grad():
            distribution = self.distribution(tensor)
            index = distribution.sample() if stochastic else distribution.logits.argmax()
            return self.actions[index].cpu().numpy(), dict(index=int(index),
                logp=float(distribution.log_prob(index)), value=float(self.value(tensor)))


class ValueTeacher(nn.Module):
    """Bootstrap ensemble predicting actual-outcome utility for each action."""
    def __init__(self, actions, ensemble_size=5):
        super().__init__()
        self.register_buffer("actions", torch.tensor(np.asarray(actions), dtype=torch.float32))
        for action in actions:
            bounded_parameters(action)
        n = len(CONTEXT_NAMES)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        self.models = nn.ModuleList([nn.Sequential(nn.Linear(n, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, len(actions))) for _ in range(ensemble_size)])

    def forward(self, contexts):
        normalized = ((contexts - self.mean) / self.scale).clamp(-8., 8.)
        return torch.stack([model(normalized) for model in self.models])

    def decide(self, context, stochastic=False):
        if stochastic:
            raise ValueError("Value teacher is deterministic; exploration uses ContextPolicy")
        with torch.no_grad():
            values = self(torch.tensor(np.asarray(context), dtype=torch.float32))
            scores = values.mean(0) - .25 * values.std(0, unbiased=False)
            index = int(scores.argmax())
            return self.actions[index].cpu().numpy(), dict(index=index,
                predicted_utility=float(values.mean(0)[index] * 40.),
                uncertainty=float(values.std(0, unbiased=False)[index] * 40.))


class OutcomeTeacher(nn.Module):
    """Compact selector that predicts interpretable outcomes for every action."""
    OUTCOMES = ("release_logit", "hit15_logit", "release_time", "joint_ratio", "pressure_ratio")

    def __init__(self, actions):
        super().__init__()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) < 2:
            raise ValueError("V23 action catalogue must contain at least two force/radius pairs")
        for action in actions:
            bounded_parameters(action)
        n = len(CONTEXT_NAMES)
        self.register_buffer("actions", torch.tensor(actions))
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        # 39->48->32->(9*5) is 4,973 parameters for the current nine actions.
        self.network = nn.Sequential(nn.Linear(n, 48), nn.Tanh(), nn.Linear(48, 32), nn.Tanh(),
                                     nn.Linear(32, len(actions) * len(self.OUTCOMES)))

    def forward(self, contexts):
        normalized = ((contexts - self.mean) / self.scale).clamp(-8., 8.)
        shape = (*normalized.shape[:-1], len(self.actions), len(self.OUTCOMES))
        return self.network(normalized).reshape(shape)

    @staticmethod
    def scores(raw):
        release = raw[..., 0].sigmoid()
        hit15 = raw[..., 1].sigmoid()
        release_time = raw[..., 2].sigmoid()
        joint = raw[..., 3]
        pressure = raw[..., 4]
        # Success dominates; time and physical costs break ties. Ratios over one
        # receive explicit penalties instead of being hidden in a scalar label.
        score = 2. * hit15 + release - .25 * release_time
        score -= 2. * torch.relu(joint - 1.) + .2 * torch.relu(pressure - 1.)
        return score

    def decide(self, context, stochastic=False):
        if stochastic:
            raise ValueError("Outcome teacher is deterministic; exploration uses ContextPolicy")
        with torch.no_grad():
            raw = self(torch.tensor(np.asarray(context), dtype=torch.float32))
            index = int(self.scores(raw).argmax())
            return self.actions[index].cpu().numpy(), dict(index=index,
                predicted_release_probability=float(raw[index, 0].sigmoid()),
                predicted_hit15_probability=float(raw[index, 1].sigmoid()),
                predicted_release_time_ratio=float(raw[index, 2].sigmoid()),
                predicted_joint_ratio=float(raw[index, 3]),
                predicted_pressure_ratio=float(raw[index, 4]))


def normalization(contexts):
    # Floors express sensor/physical scales; near-constant training features
    # must not become huge noise amplifiers at deployment.
    floors = np.r_[[.1, .1], np.full(12, .03), np.full(12, .2),
                   np.full(3, .02), np.full(3, .2), np.full(3, .2), [1., 1., 1., 2.]]
    array = np.asarray(contexts, dtype=np.float32)
    return torch.tensor(array.mean(0)), torch.tensor(np.maximum(array.std(0), floors), dtype=torch.float32)


def save_model(path, model, *, kind, contract, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = dict(schema=SCHEMA, context_names=CONTEXT_NAMES, kind=kind,
        actions=model.actions.cpu().numpy().tolist(), model=model.state_dict(), contract=contract, **extra)
    if isinstance(model, ValueTeacher):
        payload["ensemble_size"] = len(model.models)
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or tuple(payload.get("context_names", ())) != CONTEXT_NAMES:
        raise ValueError("V23 model schema or measured context does not match")
    if payload["kind"] == "value_teacher":
        model = ValueTeacher(payload["actions"], payload["ensemble_size"])
    elif payload["kind"] == "outcome_teacher":
        model = OutcomeTeacher(payload["actions"])
    elif payload["kind"] in ("context_bc", "context_ppo", "context_exploration"):
        model = ContextPolicy(payload["actions"])
    else:
        raise ValueError("unknown V23 policy kind")
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
