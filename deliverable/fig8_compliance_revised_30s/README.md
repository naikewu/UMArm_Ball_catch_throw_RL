# Revised figure-eight and compliance experiment

Open `fig8_compliance_30s_seed20260912_koopman_mppi.gif` for the 30 s animation
(300 frames, 10 fps). Red dashed and blue solid compliance ellipses share the
actual tip position on the trajectory plot. Their radius is displacement for
a 0.2 N in-plane force. The curves on the right show unsmoothed Cxx and Cyy.

## Results

- Tip 3D RMS during drawing: 1.071 mm; p95: 2.032 mm; maximum: 3.327 mm.
- Independently calculated true xy compliance error: median 0.925%, p90 1.417%.
- True Cxx/Cyy versus target correlation: 0.9944 / 0.9942.
- No temporal smoothing of true compliance was applied.
- 4500 control cycles over 30 simulated seconds; three figure-eight loops.
- Maximum requested antagonist pair sum: 24.20 psi (limit 30 psi).
- Solver median: 16.25 ms, p99: 72.77 ms. This is an offline validation,
  not a controller meeting a 150 Hz wall-clock deadline. An independent
  ablation also ran during part of this experiment; timings are not an
  isolated CPU performance benchmark.

The old tip RMS was 8.325 mm. This comparison includes a corrected reachable
reference and controller; it is not a same-reference controller-only ablation.
The 12 s same-reference ablation in `../fig8_compliance_revised/` gives:

| Experiment | Tip RMS | True xy compliance median error | Cxx/Cyy correlation |
| --- | ---: | ---: | --- |
| Common-pressure control disabled | 1.078 mm | 9.151% | -0.026 / 0.112 |
| Common-pressure control enabled | 1.054 mm | 0.951% | 0.9936 / 0.9935 |

## What changed

The old sampler changed only antagonist pressure differences, holding sums
at approximately 12 psi. The new compliance pressure planner optimizes pair
means and recalculates differences to preserve commanded joint torque. It
provides a prior to the Koopman MPPI rollout, which also samples common-mode
corrections. Pressure-difference exploration was reduced from 0.5 to 0.03 psi.
The rollout uses a polynomial compliance head alongside learned Koopman
dynamics; compliance is not newly trained into the Koopman state transition.

The old labels were finite-time force responses after releasing the robot for
0.25 s. They mixed transient motion with compliance, did not establish static
convergence, and used a point 50 mm beyond the controlled plate centre.
The new definition is **local static, frozen-pressure compliance** of the
final plate centre, in arm-base axes:

`K = -d(tau_passive + tau_muscle - tau_gravity)/dq`, `C = J inv(K) J.T`.

At a moving/non-equilibrium configuration this is the tangent with a constant
balancing torque. It is not measured hardware compliance, finite-time step
response, or the dynamic admittance of the feedback controller. True values
come independently from MuJoCo torque derivatives at the actual simulated
q and pressure. No training-head predictions are used to draw the blue ellipse.
Tests also compare this tangent with nonlinear equilibrium solutions under
small positive/negative Cartesian forces at the same plate centre.

The revised head fits joint stiffness with polynomial features of q and p,
then maps it through the tip Jacobian. It uses 3000 new labels with a held-out
600-point split: median relative compliance error 0.0488%, p90 0.1024%, maximum
0.2856%. Training covers joint angles +/-0.1 rad, pair means 2--14 psi, and
differences +/-3 psi. This accuracy is local to that sampled domain.

## Reference changes and limits

The xy compliance target varies between **0.060 and 0.078 m/N**, with x and y
in opposite phases. The previous 0.035--0.050 m/N request is not claimed solved.
At the straight pose, a uniform mean-pressure sweep from 2 to 14 psi produced
approximately 0.089--0.053 m/N. The new targets are supported by the pressure
planner and independently validated in the actual closed-loop trace.

Only the xy block is targeted. z compliance and xz/yz coupling are recorded
but are not independent tracking objectives. Legacy full-matrix predicted
error keys remain in the JSON for compatibility; use the explicitly named
`true_compliance_xy_*` keys to assess this experiment.

The xy figure eight retains amplitudes 35 mm and 18 mm. z rises on a 0.5 m
dome instead of staying at the fully extended minimum height. Entry is 2 s,
drawing is 27 s, and final hold is 1 s; 0.75 s speed ramps remove the abrupt
start/stop. The IK residual is below 0.001 mm.

## Reproduce

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m digital_twin.train_tangent_compliance
.\.venv\Scripts\python.exe -m control.render_fig8_compliance_gif --duration-s 30 --fps 10 --out-dir deliverable/fig8_compliance_revised_30s
```

To only re-render verified existing data, append `--reuse-trace --reuse-true-compliance`.
The trace, JSON metrics, independent-compliance NPZ, and static PNG accompany
the GIF. Original experiment files remain in `../fig8_compliance_gif_30s/`.
