"""Independent mechanics and allocation checks for the figure-eight task."""
import numpy as np
from scipy.optimize import root

from control.controller import InverseDynamics, PA_PER_PSI, pair_indices
from control.compliance_planner import CompliancePressurePlanner
from control.trajectory import FigureEightComplianceTrajectory, tip_jacobian
from digital_twin.tangent_compliance import TangentCompliance
from digital_twin.train_tangent_compliance import TangentComplianceHead


def test_plate_jacobian_matches_mujoco_and_batched_kinematics():
    exact = TangentCompliance()
    qs = np.random.default_rng(45).uniform(-.06, .06, (5, 12))
    batch = tip_jacobian(qs)
    for q, J in zip(qs, batch):
        exact.torque(q, np.full(24, 6 * PA_PER_PSI))
        mj_J = np.zeros((3, exact.m.nv))
        exact.mj.mj_jac(exact.m, exact.d, mj_J, None, exact.d.xpos[exact.body], exact.body)
        np.testing.assert_allclose(J, mj_J[:, exact.dofs], atol=1e-12)


def test_tangent_matches_nonlinear_force_equilibrium_and_is_order_independent():
    exact = TangentCompliance()
    q = np.linspace(-.025, .035, 12)
    p = np.linspace(5, 10, 24) * PA_PER_PSI
    baseline = exact.torque(q, p)
    C = exact.predict(q, p)
    columns = []
    force = 1e-4
    for axis in np.eye(3):
        tips = []
        for sign in (1, -1):
            def balance(qi):
                tau = exact.torque(qi, p)
                J = np.zeros((3, exact.m.nv))
                exact.mj.mj_jac(exact.m, exact.d, J, None, exact.d.xpos[exact.body], exact.body)
                return tau - baseline + J[:, exact.dofs].T @ (sign * force * axis)
            sol = root(balance, q, tol=1e-10)
            assert np.linalg.norm(balance(sol.x)) < 1e-8
            tips.append(exact.d.xpos[exact.body].copy())
        columns.append((tips[0] - tips[1]) / (2 * force))
    np.testing.assert_allclose(C, np.array(columns).T, rtol=2e-5, atol=1e-8)
    exact.predict(-q, p * .8)
    np.testing.assert_array_equal(C, exact.predict(q, p))
    assert np.linalg.eigvalsh(C).min() >= -1e-12


def test_new_head_matches_independent_twin_at_trajectory_states():
    exact, head = TangentCompliance(), TangentComplianceHead()
    for mean in (3., 6., 10., 13.):
        q = np.linspace(-.04, .04, 12)
        p = np.full(24, mean * PA_PER_PSI)
        C = exact.predict(q, p)
        assert np.linalg.norm(head.predict(q, p) - C) / np.linalg.norm(C) < .003


def test_common_pressure_changes_without_changing_commanded_torque():
    dynamics = InverseDynamics()
    q = np.linspace(-.02, .02, 12)
    prior, _ = dynamics.allocate(q, np.zeros(12), np.zeros(12), np.full(24, 6*PA_PER_PSI))
    planner = CompliancePressurePlanner(TangentComplianceHead(), 1/150)
    target = np.diag([.06, .077, 0])
    for _ in range(150):
        pressure = planner.command(prior, dynamics.last_B, q, target)
    np.testing.assert_allclose(dynamics.last_B @ (pressure / PA_PER_PSI),
                               dynamics.last_B @ (prior / PA_PER_PSI), atol=1e-10)
    assert np.ptp(pressure[pair_indices()].mean(axis=1)/PA_PER_PSI) > 1.
    assert np.max(pressure[pair_indices()].sum(axis=1)) <= 30*PA_PER_PSI


def test_revised_figure_eight_is_reachable_and_starts_smoothly():
    traj = FigureEightComplianceTrajectory(entry_s=2, loops=3, smooth_ramp_s=.75, dome_m=.5,
                                          hard_m_per_n=.060, soft_m_per_n=.078, z_m_per_n=0)
    assert traj.metadata["ik_max_residual_mm"] < .1
    for t in (traj.write_start_s, traj.write_end_s):
        _, velocity, acceleration = traj.sample(t)
        assert np.max(np.abs(velocity)) < .001
        assert np.max(np.abs(acceleration)) < .1


def test_ellipses_share_tip_center_and_physical_scale():
    from control.render_fig8_compliance_gif import _centered_ellipse
    for center in (np.array([.025, -.018]), np.array([-.02, .01])):
        for C in (np.diag([.06, .078]), np.array([[.07, .004], [.004, .063]])):
            points = _centered_ellipse(C, center, .2)
            np.testing.assert_allclose(points[:-1].mean(axis=0), center*1000, atol=1e-12)
            np.testing.assert_allclose(points[0]-center*1000, .2*1000*C[:,0], atol=1e-12)
            np.testing.assert_array_equal(points[0], points[-1])


def test_compliance_period_changes_orientation_without_changing_tip_path():
    options = dict(entry_s=2, loops=3, smooth_ramp_s=.75, dome_m=.5,
                   hard_m_per_n=.054, soft_m_per_n=.098, z_m_per_n=0)
    normal = FigureEightComplianceTrajectory(**options)
    slower = FigureEightComplianceTrajectory(**options, compliance_period_s=18)
    times = np.linspace(2, 29, 200)
    np.testing.assert_array_equal(normal.tip_target(times), slower.tip_target(times))
    C = slower.compliance_target(times)
    assert np.max(C[:,0,0] / C[:,1,1]) > 1.81
    assert np.max(C[:,1,1] / C[:,0,0]) > 1.81
    np.testing.assert_allclose(C[:,0,0] + C[:,1,1], .152)
