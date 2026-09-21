"""Terminal-horizon digital-twin branches for V26 feasibility audits.

V25 labelled one action followed by at most two seconds of simulation.  These
helpers deliberately run each branch until release/episode termination so the
planner can distinguish short-term error reduction from actual reachability.
"""
from __future__ import annotations

import numpy as np

from .contextual_env import episode_utility
from .twin_residual_env import HOLD_ACTION, JOINT_ACTION_NAMES
from .twin_snapshot import TwinSnapshot, numeric_fingerprint


def terminal_features(result, *, elapsed_s=0.):
    """Return finite, JSON-safe features used to rank a terminal branch."""
    captured = bool(result.get("captured", False))
    released = bool(result.get("released", False))
    hit15 = bool(result.get("hit15", False))
    max_joint = float(result.get("max_joint_deg", 1e9))
    unsafe = bool(result.get("grip_broken", False) or max_joint > 36.)
    landing = result.get("landing_error_m")
    minimum = (result.get("calibrated_release") or {}).get("min_calibrated_error_m")
    error = landing if landing is not None else minimum
    error = 1. if error is None or not np.isfinite(error) else float(error)
    release_s = result.get("catch_to_release_s")
    release_s = 18. if release_s is None or not np.isfinite(release_s) else float(release_s)
    utility = float(episode_utility(result))
    return dict(captured=captured, released=released, hit15=hit15, unsafe=unsafe,
        grip_broken=bool(result.get("grip_broken", False)), max_joint_deg=max_joint,
        terminal_error_m=error, catch_to_release_s=release_s,
        elapsed_s=float(elapsed_s), utility=utility)


def terminal_rank(features):
    """Lexicographic safety/reachability rank; any safe release beats no release."""
    return (int(not features["unsafe"]), int(features["captured"]),
        int(features["released"]), int(features["hit15"]),
        -float(features["terminal_error_m"]), -float(features["catch_to_release_s"]),
        float(features["utility"]))


def terminal_branch(env, snapshot, action, *, max_steps=6000):
    """Restore ``snapshot``, apply one action, then hold through termination."""
    snapshot.restore()
    if env.done or not env.dynamic_due():
        raise RuntimeError("terminal branch requires a live V26 decision state")
    start = float(env.obs["t_win"])
    env.apply_joint_action(int(action))
    result = None
    for _ in range(max_steps):
        if env.done:
            break
        if env.dynamic_due():
            env.apply_joint_action(HOLD_ACTION)
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    if not env.done:
        raise RuntimeError("terminal digital-twin branch exceeded its step limit")
    result = result or env.summary()
    features = terminal_features(result, elapsed_s=float(env.obs["t_win"])-start)
    features.update(action=int(action), action_name=JOINT_ACTION_NAMES[int(action)],
                    rank=list(terminal_rank(features)))
    return features, result, numeric_fingerprint(env)


def choose_terminal_action(env):
    """Evaluate every valid action to terminal and restore the decision state."""
    mask = env.valid_action_mask()
    snapshot = TwinSnapshot(env)
    branches = [None] * len(mask)
    for action in np.flatnonzero(mask):
        features, _, _ = terminal_branch(env, snapshot, int(action))
        branches[int(action)] = features
    snapshot.restore()
    valid = [int(index) for index in np.flatnonzero(mask)]
    chosen = max(valid, key=lambda index: terminal_rank(branches[index]))
    return chosen, branches
