# Figure-eight tip and compliance experiments

This branch contains the local experiment work based on commit `1e3f77b`.
It preserves the fitted digital twin and Koopman dynamics used for the runs;
later upstream model changes have not been merged into the experimental baseline.

## Videos and results

- [Large horizontal/vertical variation, 30 s MP4](../deliverable/fig8_compliance_wide_slow_30s/fig8_compliance_30s_seed20260912_koopman_mppi.mp4)
- [Large-variation report and reproduction commands](../deliverable/fig8_compliance_wide_slow_30s/README.md)
- [Small-variation report and reproduction commands](../deliverable/fig8_compliance_revised_30s/README.md)

The small-variation run achieves 1.071 mm drawing tip RMS and 0.925% median
true xy compliance error. Increasing the target aspect ratio from 1.30 to
1.815 yields an actual maximum ratio of about 1.681, 1.527 mm tip RMS and
3.143% median compliance error. The wider compliance schedule is slower;
the tip trajectory and its timing are unchanged. These are offline simulations,
not demonstrated 150 Hz wall-clock control or hardware validation.

## Models and definition

The new readout fits joint stiffness with polynomial q/pressure features and
computes `C = J inv(K) J.T`. The original 3000-point head and expanded 5000-point
head, their datasets, and holdout metrics are committed under `data/`.
The older finite-time force-response fits are retained for provenance but are
not the static compliance definition used in the revised experiments.

Control combines feedforward PID, a torque-preserving common-pressure planner,
and Koopman MPPI. It uses the trained compliance head for optimization; blue
ellipses are calculated separately from MuJoCo torque derivatives at recorded
actual q and pressure. No temporal smoothing or anisotropic display stretching
is used. Both ellipses are centred at the actual tip with the same 0.2 N scale.
Compliance means local frozen-pressure static tangent at the final plate centre;
it does not mean the dynamic admittance of the complete feedback controller.

## Reference repository

`UMArm_dynamic_koopman_compliance` is an unmodified reference repository recorded
as a Git submodule at commit `3f4bf512befec4c1985429de1e104e44e94e46ca`.
After cloning this branch, retrieve its files with:

```powershell
git submodule update --init --recursive
```

The new outer compliance pipeline runs independently of this reference submodule.
Python virtual environments, caches and nested Git object databases are not
versioned. Training artifacts and final MP4s are explicitly included despite
the original repository's general NPZ/video ignore rules.

## Verification

```powershell
python -m pytest control/test_tangent_compliance.py control/test_trajectory.py control/test_controller.py control/test_benchmark.py -q -p no:cacheprovider
```

These checks cover the MuJoCo/analytic endpoint Jacobian, nonlinear force
equilibrium versus tangent compliance, fitted-head accuracy, torque-preserving
pressure allocation, reachable references, ellipse centring/scaling, and
independent compliance-period changes without altering the tip path.
