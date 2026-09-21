import numpy as np

from rl_ppo.env import UMArmBallCatchEnv


def test_pair_pressure_mapping_respects_hardware_limits() -> None:
    environment = UMArmBallCatchEnv()
    environment.mean_psi = np.full(12, 15.0)
    environment.diff_psi = np.full(12, 30.0)
    pressure_psi = environment._pressures_from_pair_state() / 6894.757293168
    assert pressure_psi.shape == (24,)
    assert np.all(pressure_psi >= 0.0)
    assert np.all(pressure_psi <= 30.0)
    for index_a, index_b in environment.pairs:
        assert pressure_psi[index_a] + pressure_psi[index_b] <= 30.0 + 1e-6
