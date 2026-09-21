from teacher_rl import env as _bootstrap
from teacher_rl.contextual_env import CONTEXT_NAMES
from teacher_rl.twin_dataset_collection import dataset_summary


def row(seed, split="train", branches=9, captured=True):
    result = dict(captured=captured, released=captured, hit15=captured,
        grip_broken=False, max_joint_deg=30.)
    return dict(seed=seed, scenario=dict(split=split), result=result,
        baseline=dict(result), mode="v26_mpc", high_level_action=1,
        observed_context=([0.] * len(CONTEXT_NAMES) if captured else None),
        predicted_branches=([{} for _ in range(branches)] if captured else None),
        prediction_exact=True)


def test_dataset_ready_requires_one_thousand_complete_independent_contexts():
    rows = [row(index, "train" if index < 800 else "validation")
            for index in range(1000)]
    summary = dataset_summary(rows, 1000)
    assert summary["dataset_ready"]
    assert summary["terminal_branch_labels"] == 9000


def test_dataset_rejects_missing_terminal_branch_label():
    rows = [row(index, "train" if index < 800 else "validation")
            for index in range(1000)]
    rows[-1] = row(999, "validation", branches=8)
    summary = dataset_summary(rows, 1000)
    assert not summary["dataset_ready"]
    assert not summary["checks"]["nine_terminal_labels"]
