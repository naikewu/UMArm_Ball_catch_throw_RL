"""Independent conservative V25 candidate/V15 deployment gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contextual_env import CONTEXT_NAMES


CONTEXT_SIZE = len(CONTEXT_NAMES)


class GateMember(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CONTEXT_SIZE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, value):
        return self.net(value)


class ConservativeGate:
    def __init__(self, members, mean, scale, delta_threshold, success_threshold,
                 uncertainty_multiplier=1.64):
        self.members = list(members)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.delta_threshold = float(delta_threshold)
        self.success_threshold = float(success_threshold)
        self.uncertainty_multiplier = float(uncertainty_multiplier)

    def predict(self, context):
        context = np.asarray(context, dtype=np.float32)
        if context.shape != (CONTEXT_SIZE,) or not np.isfinite(context).all():
            raise ValueError("gate context is invalid")
        value = torch.tensor(((context-self.mean)/self.scale).reshape(1, -1))
        with torch.no_grad():
            output = np.vstack([member(value).numpy()[0] for member in self.members])
        delta = output[:, 0]
        success = 1./(1.+np.exp(-output[:, 1]))
        k = self.uncertainty_multiplier
        return dict(delta_mean=float(delta.mean()), delta_lcb=float(delta.mean()-k*delta.std()),
            success_mean=float(success.mean()),
            success_lcb=float(success.mean()-k*success.std()))

    def decision(self, context):
        prediction = self.predict(context)
        prediction["takeover"] = bool(
            prediction["delta_lcb"] > self.delta_threshold and
            prediction["success_lcb"] >= self.success_threshold)
        return prediction


def save_gate(path, gate, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(dict(member_states=[member.state_dict() for member in gate.members],
        mean=gate.mean, scale=gate.scale, delta_threshold=gate.delta_threshold,
        success_threshold=gate.success_threshold,
        uncertainty_multiplier=gate.uncertainty_multiplier,
        hidden=int(gate.members[0].net[0].weight.shape[0]), **metadata), temporary)
    temporary.replace(path)


def load_gate(path, expected_schema=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ValueError("gate checkpoint schema mismatch")
    members = []
    for state in payload["member_states"]:
        member = GateMember(payload["hidden"])
        member.load_state_dict(state)
        member.eval()
        members.append(member)
    return ConservativeGate(members, payload["mean"], payload["scale"],
        payload["delta_threshold"], payload["success_threshold"],
        payload["uncertainty_multiplier"]), payload
