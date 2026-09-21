from dataclasses import replace

from teacher_rl.anchored_dynamic_env import TRAJECTORY_ACTIONS
from teacher_rl.twin_causal_mpc import apply_absolute_action


class _Release:
    def __init__(self):
        self.config = None
        self.allowed = False

    def set_permission(self, allowed):
        self.allowed = bool(allowed)


class _Env:
    def __init__(self):
        from teacher_rl.contextual_env import base_config
        self.continuous = base_config(tolerance=.08)
        self.handover = type("Handover", (), {})()
        self.releaser = _Release()


def test_absolute_action_covers_audited_grid_and_updates_all_consumers():
    env = _Env()
    for action, expected in enumerate(TRAJECTORY_ACTIONS):
        parameters = apply_absolute_action(env, action)
        assert (parameters["envelope_force_scale"], parameters["envelope_radius_scale"]) == expected
        assert env.handover.config is env.continuous
        assert env.releaser.config is env.continuous
        assert env.releaser.allowed


def test_absolute_action_rejects_out_of_catalogue_index():
    env = _Env()
    try:
        apply_absolute_action(env, len(TRAJECTORY_ACTIONS))
    except ValueError as error:
        assert "catalogue" in str(error)
    else:
        raise AssertionError("invalid terminal action was accepted")
