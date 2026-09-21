from types import SimpleNamespace

import numpy as np

from teacher_rl import env as _bootstrap  # installs sibling CAN source paths
from teacher_rl.twin_terminal_planner import terminal_features, terminal_rank


def _result(*, released, hit=False, unsafe=False, error=None, release_s=None):
    return dict(captured=True, released=released, hit15=hit, hit30=hit,
        grip_broken=unsafe, max_joint_deg=40. if unsafe else 25.,
        landing_error_m=error if released else None,
        catch_to_release_s=release_s, calibrated_release=dict(
            min_calibrated_error_m=.2 if error is None else error))


def test_terminal_rank_makes_safe_release_dominate_unreleased_error():
    unreleased = terminal_features(_result(released=False, error=.001))
    released = terminal_features(_result(released=True, hit=False, error=.14, release_s=9.))
    assert terminal_rank(released) > terminal_rank(unreleased)


def test_terminal_rank_makes_safety_dominate_hit():
    unsafe_hit = terminal_features(
        _result(released=True, hit=True, unsafe=True, error=.01, release_s=3.))
    safe_miss = terminal_features(
        _result(released=True, hit=False, unsafe=False, error=.16, release_s=10.))
    assert terminal_rank(safe_miss) > terminal_rank(unsafe_hit)


def test_terminal_features_are_finite_when_no_release_prediction_exists():
    result = _result(released=False)
    result["calibrated_release"] = {}
    features = terminal_features(result)
    values = [features[key] for key in (
        "max_joint_deg", "terminal_error_m", "catch_to_release_s", "utility")]
    assert np.isfinite(values).all()
