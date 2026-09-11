"""The independent stress reference supplies consistent PID derivative targets."""
import numpy as np
from .tune_pid_stress import StressTrajectory


def test_stress_derivatives_and_smooth_boundaries():
    reference=StressTrajectory()
    for t in (1.,2.5,6.,10.5,15.,21.):
        h=1e-4
        q,qd,qdd=reference.sample(t)
        qm=reference.sample(t-h)[0];qp=reference.sample(t+h)[0]
        np.testing.assert_allclose((qp-qm)/(2*h),qd,atol=2e-8)
        np.testing.assert_allclose((qp-2*q+qm)/(h*h),qdd,atol=2e-6)
    for t in (3.,5.,8.,10.,13.,20.,23.):
        before=reference.sample(t-1e-7);after=reference.sample(t+1e-7)
        for a,b in zip(before,after):np.testing.assert_allclose(a,b,atol=3e-6)


def test_stress_bounds_and_initial_pose():
    reference=StressTrajectory(initial_deg=4.)
    q=np.array([reference.sample(t)[0] for t in np.linspace(0,28,4201)])
    assert np.rad2deg(abs(q).max())<=14+1e-10
    np.testing.assert_array_equal(reference.sample(0)[0],reference.q0)
    assert not q[-1].any()
