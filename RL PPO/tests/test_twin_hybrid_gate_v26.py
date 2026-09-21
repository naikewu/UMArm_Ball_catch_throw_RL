from teacher_rl import env as _bootstrap
from teacher_rl.anchored_dynamic_env import TRAJECTORY_ACTIONS
from teacher_rl.twin_hybrid_gate import HIGH_LEVEL_ACTIONS, select_high_level_action


def branch(action, *, hit=False, unsafe=False, error=.2, release=18.):
    return dict(action=action, captured=True, released=hit, hit15=hit, unsafe=unsafe,
        grip_broken=False, max_joint_deg=30., terminal_error_m=error,
        catch_to_release_s=release, elapsed_s=release, utility=0.)


def test_hybrid_catalogue_is_v15_plus_nine_v26_trajectories():
    assert len(HIGH_LEVEL_ACTIONS) == len(TRAJECTORY_ACTIONS) + 1
    assert HIGH_LEVEL_ACTIONS[0] == "v15_fallback"


def test_hybrid_gate_falls_back_when_no_branch_predicts_safe_hit():
    branches = [branch(index) for index in range(len(TRAJECTORY_ACTIONS))]
    assert select_high_level_action(branches) == 0


def test_hybrid_gate_selects_best_safe_hit_and_uses_one_based_v26_action():
    branches = [branch(index) for index in range(len(TRAJECTORY_ACTIONS))]
    branches[2] = branch(2, hit=True, error=.08, release=9.)
    branches[7] = branch(7, hit=True, error=.04, release=7.)
    branches[8] = branch(8, hit=True, unsafe=True, error=.01, release=5.)
    assert select_high_level_action(branches) == 8


def test_hybrid_gate_requires_all_nine_predictions():
    try:
        select_high_level_action([branch(0)])
    except ValueError as error:
        assert "nine" in str(error)
    else:
        raise AssertionError("incomplete twin branches were accepted")
