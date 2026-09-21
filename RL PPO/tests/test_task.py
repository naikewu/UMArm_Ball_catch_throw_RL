import numpy as np

from rl_ppo.task import CatchBenchmark, ball_state, sample_case


def test_sampled_ballistic_case_reaches_its_intercept() -> None:
    task = CatchBenchmark()
    home_tip = np.array([0.0, 0.0, -0.875459])
    case = sample_case(np.random.default_rng(11), home_tip)
    position, velocity = ball_state(case, case.intercept_time_s)
    np.testing.assert_allclose(position, case.intercept_m, atol=1e-7)
    np.testing.assert_allclose(velocity, case.incoming_velocity_mps, atol=1e-7)
    assert case.mode in set(task.modes)
