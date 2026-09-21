import numpy as np
import pytest

from control.render_fig8_speed_comparison import DEFAULT_SPEED_SCALES, make_trajectory


def test_speed_comparison_keeps_episode_and_path_fixed():
    trajectories = [make_trajectory(30, speed) for speed in DEFAULT_SPEED_SCALES]
    np.testing.assert_allclose([trajectory.duration_s for trajectory in trajectories], 30)
    np.testing.assert_allclose([trajectory.path_s for trajectory in trajectories], 27)
    np.testing.assert_allclose(
        [trajectory.tip_target(2) for trajectory in trajectories],
        np.repeat(trajectories[0].tip_target(2)[None, :], len(trajectories), axis=0),
    )
    np.testing.assert_allclose(
        [trajectory.tip_target(29) for trajectory in trajectories],
        np.repeat(trajectories[0].tip_target(29)[None, :], len(trajectories), axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose([trajectory.metadata["period_s"] for trajectory in trajectories],
                               [18, 9, 6, 4.5])
    np.testing.assert_allclose([trajectory.metadata["compliance_period_s"] for trajectory in trajectories],
                               [18, 9, 6, 4.5])


def test_speed_comparison_rejects_too_short_episode():
    with pytest.raises(ValueError, match="duration"):
        make_trajectory(3, 1)
