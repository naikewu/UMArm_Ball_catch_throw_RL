"""In-process snapshots for counterfactual branches of the CAN MuJoCo twin.

The snapshot is intentionally tied to one live environment instance.  It copies
MuJoCo integration state, mutable equality-model data, CAN node state, controller
history, estimators, valve events, and Python-side episode bookkeeping.  Static
models, neural networks, calibration objects, locks, and object references stay
shared.  This is enough for sequential MPC branches without constructing a new
MuJoCo model for every candidate.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import is_dataclass
import inspect
import random
import types

import mujoco
import numpy as np
import torch


_ALLOWED_MODULE_PREFIXES = (
    "teacher_rl.", "can_", "launches_", "digital_twin.sim_core",
)
_STATIC_CLASS_NAMES = {
    "CanModel", "IntentPolicy", "KernelCalibration", "RidgeLandingCalibration",
    "ActuatorModel", "FlowNet", "TwinKwargs", "ThrowDistribution",
}


def _is_graph_object(value):
    if value is None or not hasattr(value, "__dict__"):
        return False
    if isinstance(value, (torch.nn.Module, types.ModuleType)):
        return False
    if is_dataclass(value):
        return False
    cls = type(value)
    if cls.__name__ in _STATIC_CLASS_NAMES:
        return False
    return cls.__module__.startswith(_ALLOWED_MODULE_PREFIXES)


def _children(value):
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, (list, tuple, deque, set)):
        return value
    return ()


def _state_objects(root):
    result, seen, pending = [], set(), [root]
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if _is_graph_object(value):
            result.append(value)
            pending.extend(value.__dict__.values())
        else:
            pending.extend(_children(value))
    return result


def _copy_value(value, graph_ids):
    if id(value) in graph_ids:
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.random.Generator):
        return ("__numpy_generator__", copy.deepcopy(value.bit_generator.state))
    if isinstance(value, random.Random):
        return ("__python_random__", value.getstate())
    if isinstance(value, deque):
        return deque((_copy_value(row, graph_ids) for row in value), maxlen=value.maxlen)
    if isinstance(value, list):
        return [_copy_value(row, graph_ids) for row in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(row, graph_ids) for row in value)
    if isinstance(value, dict):
        return {_copy_value(key, graph_ids): _copy_value(row, graph_ids)
                for key, row in value.items()}
    if isinstance(value, set):
        return {_copy_value(row, graph_ids) for row in value}
    if (value is None or isinstance(value, (bool, int, float, str, bytes, np.generic)) or
            is_dataclass(value) or inspect.ismethod(value) or inspect.isfunction(value) or
            isinstance(value, (torch.nn.Module, types.ModuleType))):
        return value
    # Unknown values here are immutable/static references (models, locks, maps).
    return value


def _restore_value(current, saved):
    if isinstance(saved, tuple) and saved and saved[0] == "__numpy_generator__":
        if not isinstance(current, np.random.Generator):
            raise TypeError("snapshot numpy RNG target changed type")
        current.bit_generator.state = copy.deepcopy(saved[1])
        return current
    if isinstance(saved, tuple) and saved and saved[0] == "__python_random__":
        if not isinstance(current, random.Random):
            raise TypeError("snapshot Python RNG target changed type")
        current.setstate(saved[1])
        return current
    if isinstance(saved, np.ndarray):
        return saved.copy()
    if _is_graph_object(saved):
        return saved
    if isinstance(saved, deque):
        return deque((_restore_value(None, row) for row in saved), maxlen=saved.maxlen)
    if isinstance(saved, list):
        return [_restore_value(None, row) for row in saved]
    if isinstance(saved, tuple):
        return tuple(_restore_value(None, row) for row in saved)
    if isinstance(saved, dict):
        return {_restore_value(None, key): _restore_value(None, row)
                for key, row in saved.items()}
    if isinstance(saved, set):
        return {_restore_value(None, row) for row in saved}
    return saved


class TwinSnapshot:
    """A restorable snapshot of one live V25 environment."""

    STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, env):
        self.env = env
        self.plant = env.plant
        model, data = self.plant.model, self.plant.data
        size = mujoco.mj_stateSize(model, self.STATE_SPEC)
        self.mujoco_state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(model, data, self.mujoco_state, self.STATE_SPEC)
        # mj_getState contains the integration variables.  An in-process MPC
        # branch also needs derived caches (ten_length, qacc, constraints) to
        # restart bit-for-bit, so retain a full MjData copy as well.
        self.mujoco_data = mujoco.MjData(model)
        mujoco.mj_copyData(self.mujoco_data, model, data)
        # CanPlant mutates equality anchor data when attaching the ball.
        self.eq_data = model.eq_data.copy()
        self.eq_solref = model.eq_solref.copy()
        self.eq_solimp = model.eq_solimp.copy()
        # SimArm updates tendon damping at every firmware-grid pass, while the
        # spring gripper changes pair margins on latch/release.  Both live on
        # MjModel rather than MjData and therefore are absent from mj_copyData.
        self.tendon_damping = model.tendon_damping.copy()
        self.pair_margin = model.pair_margin.copy()
        self.objects = _state_objects(env)
        graph_ids = {id(value) for value in self.objects}
        self.states = []
        for value in self.objects:
            state = {name: _copy_value(row, graph_ids)
                     for name, row in value.__dict__.items()}
            self.states.append((value, state))

    def restore(self):
        if self.env.plant is not self.plant:
            raise RuntimeError("snapshot can only restore its original environment")
        # Restore object attributes first so hooks invoked by mj_forward see the
        # matching controller/plant flags.
        for value, state in self.states:
            for name in tuple(value.__dict__):
                if name not in state:
                    del value.__dict__[name]
            for name, saved in state.items():
                current = value.__dict__.get(name)
                value.__dict__[name] = _restore_value(current, saved)
        model, data = self.plant.model, self.plant.data
        model.eq_data[:] = self.eq_data
        model.eq_solref[:] = self.eq_solref
        model.eq_solimp[:] = self.eq_solimp
        model.tendon_damping[:] = self.tendon_damping
        model.pair_margin[:] = self.pair_margin
        mujoco.mj_copyData(data, model, self.mujoco_data)
        return self.env


def numeric_fingerprint(env):
    """Compact state vector used by snapshot determinism tests and audits."""
    plant = env.plant
    node_values = []
    for key in sorted(plant.nodes):
        node = plant.nodes[key]
        node_values.extend(float(getattr(node, name, 0.)) for name in
            ("p_pa", "filtered_raw", "output_permille", "integral_raw", "deriv_filt"))
    ball_state = np.r_[plant.ball_pos(), plant.ball_vel(), float(plant.ball_held()),
                       float(plant.grip_broken)]
    return np.r_[plant.data.time, plant.data.qpos, plant.data.qvel,
                 plant.data.ctrl, node_values, ball_state].astype(np.float64)
