import numpy as np
import torch

from teacher_rl import env as _bootstrap  # installs sibling CAN source paths
from teacher_rl.contextual_env import CONTEXT_NAMES
from teacher_rl.twin_static_model import ACTION_COUNT, OUTCOME_SIZE, StaticTerminalPolicy
from teacher_rl.twin_static_training import _oracle_action_score


def test_static_policy_scores_nine_actions_and_outcomes():
    model = StaticTerminalPolicy(hidden=16)
    context = torch.zeros((3, len(CONTEXT_NAMES)))
    logits, values, outcomes = model(context)
    assert logits.shape == (3, ACTION_COUNT)
    assert values.shape == (3,)
    assert outcomes.shape == (3, ACTION_COUNT, OUTCOME_SIZE)


def test_terminal_teacher_prioritizes_release_over_small_unreleased_error():
    unreleased = dict(released=False, hit15=False, terminal_error_m=.001,
                      catch_to_release_s=18., unsafe=False)
    released = dict(released=True, hit15=False, terminal_error_m=.14,
                    catch_to_release_s=10., unsafe=False)
    assert _oracle_action_score(released) > _oracle_action_score(unreleased)


def test_static_policy_rejects_invalid_context():
    model = StaticTerminalPolicy(hidden=16)
    try:
        model.act(np.zeros(len(CONTEXT_NAMES)-1, dtype=np.float32))
    except ValueError as error:
        assert "context" in str(error)
    else:
        raise AssertionError("invalid context was accepted")
