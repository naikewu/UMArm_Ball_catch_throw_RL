# Work Completed: Figure-Eight Tip and Compliance Control

For the detailed mathematical architecture, training losses, and MPPI settings,
see [Koopman, compliance and MPPI technical details](KOOPMAN_COMPLIANCE_MPPI_TECHNICAL_DETAILS.md).

Updated: 2026-09-13.

This document records the changes made to the downloaded digital-twin project,
the models trained, the experiments performed, and the current limitations.
It supplements the original [README.md](README.md).

## 1. Objective and Current Outcome

The objective was to make the latest fitted MuJoCo digital twin follow a
figure-eight tip trajectory while also tracking a changing compliance target.
Target and actual compliance are displayed as ellipses centred on the actual
tip, and the experiments are exported as 30-second GIF and MP4 files.

The current implementation combines:

1. The existing fitted digital twin and learned Koopman dynamics.
2. The existing inverse-dynamics feedforward PID for position control.
3. A newly trained polynomial stiffness/compliance readout.
4. A new common-pressure planner that preserves commanded joint torque.
5. Compliance-aware Koopman MPPI and independent MuJoCo evaluation.

The compliance readout was retrained. The existing Koopman dynamics checkpoint
and the digital twin's actuator/mechanical fits were not retrained in this work.

### Main Deliverables

- [Latest video: large horizontal-to-vertical variation](deliverable/fig8_compliance_wide_slow_30s/fig8_compliance_30s_seed20260912_koopman_mppi.mp4)
- [Latest experiment report](deliverable/fig8_compliance_wide_slow_30s/README.md)
- [Small-variation video with lower tracking error](deliverable/fig8_compliance_revised_30s/fig8_compliance_30s_seed20260912_koopman_mppi.mp4)
- [Small-variation experiment report](deliverable/fig8_compliance_revised_30s/README.md)

## 2. Baseline and Reference Repositories

The experimental baseline is upstream commit `1e3f77b`. The fitted model is
loaded through `digital_twin/twin_params.py` and simulated by
`digital_twin/sim_core.py`. Existing checkpoints include:

- `digital_twin/checkpoints/canarm_flow.npz`
- `digital_twin/checkpoints/canarm_mech.json`
- `control/checkpoints/canarm_koopman.pt`

The older `UMArm_dynamic_koopman_compliance` repository was inspected as a
reference for compliance fitting and MPC integration. It is now recorded as
an unmodified Git submodule at `3f4bf512befec4c1985429de1e104e44e94e46ca`.
The new outer-repository control pipeline runs independently of that submodule.

Later upstream changes have not been merged into the tested model baseline.
Changing the twin requires rechecking the trained head and control results.

## 3. Problems Found and Corrected

### 3.1 Pressure Differences Alone Did Not Provide Compliance Control

The original MPPI sampled antagonist pressure differences. Pair sums stayed
approximately fixed at 12 psi, limiting its ability to change stiffness.

For one antagonist pair, pressure can be parameterized as:

```text
p_a = mean + difference / 2
p_b = mean - difference / 2
```

We added optimization of the common pressure (`mean`). Because the real fitted
geometry is asymmetric, changing common pressure can also change torque. The
new allocator recalculates differences to preserve the position controller's
requested joint torque, subject to pressure projection and model accuracy.

### 3.2 The First Compliance Labels Were Not Static Compliance

The initial `train_compliance_head.py` collector applied positive/negative
forces and integrated for a fixed duration, including a 0.25-second experiment.
This did not establish static convergence. It mixed force response with
transient motion and used the visible tip stub, 50 mm beyond the plate centre
tracked by the controller.

Those scripts and artifacts remain for provenance, but the collector is now
explicitly labelled legacy. The revised experiments use local static tangent
compliance at the final plate centre instead.

The smoother revised ellipse does not prove that the old finite-time dynamic
response has stopped oscillating: the evaluated physical quantity changed.

### 3.3 The Original MPPI Settings Degraded Tip Tracking

The first 30-second experiment had 8.325 mm drawing tip RMS. A separate
feedforward-PID check already achieved approximately 1.16 mm, indicating that
the original MPPI settings were disrupting otherwise useful position control.

The revised settings reduce pressure exploration, favour position tracking,
and penalize large corrections:

| Setting | First GIF run | Revised runs |
| --- | ---: | ---: |
| Pressure-difference sampling noise | 0.5 psi | 0.03 psi |
| Tip weight | 1 | 10 |
| Compliance weight | 80 | 10 |
| Action regularization | 0.1 | 10 |
| MPPI horizon | 8 steps | 16 steps |
| MPPI samples | 16 | 32 |

Common-mode MPPI sampling uses 0.03 psi noise. The larger common-pressure
changes mainly come from the learned-head pressure planner.

### 3.4 Reference Reachability and Start/Stop Motion

The original figure eight kept z at the fully extended minimum height, which
was not exactly reachable during lateral motion. The revised reference uses a
0.5 m dome for z, while retaining xy amplitudes of 35 mm and 18 mm.

The revised 30-second run has 2 seconds of entry, 27 seconds of drawing, and
1 second of final hold. A 0.75-second phase ramp smooths the start and stop.
Its maximum IK residual is below 0.001 mm.

The reduction from 8.325 mm to approximately 1 mm includes both controller
and reference changes. It is not a same-reference, single-factor comparison.

## 4. New Compliance Training

### Definition

At a supplied joint configuration and fixed actual pressure:

```text
K(q, p) = -d(tau_passive + tau_muscle - tau_gravity) / dq
C(q, p) = J(q) inverse(K(q, p)) J(q)^T
```

`J` is the final plate centre's position Jacobian in arm-base axes. `C` has
units of m/N. At a moving or non-equilibrium state, this is a local tangent
with a constant balancing torque. It is not closed-loop dynamic admittance.

### Fitted Model

`digital_twin/train_tangent_compliance.py` samples q and pressure, calculates
joint-stiffness labels from the fitted MuJoCo mechanics, and fits the 78 unique
entries of the symmetric 12-by-12 stiffness matrix using ridge regression.

Polynomial features include a constant, q, quadratic q terms, pressure, and
q-pressure cross terms. Runtime prediction reconstructs the fitted stiffness
matrix and maps it to tip compliance using the analytic Jacobian.

This is a polynomial stiffness fit with a physics-based compliance mapping.
Compliance is not newly encoded into the Koopman state-transition model.

| Training configuration | Initial static head | Expanded static head |
| --- | ---: | ---: |
| Total samples | 3000 | 5000 |
| Training / holdout split | 2400 / 600 | 4000 / 1000 |
| Joint-angle sampling range | +/-0.1 rad | +/-0.1 rad |
| Pair-mean pressure range | 2--14 psi | 0.3--14.5 psi |
| Holdout median relative compliance error | 0.0488% | 0.0701% |
| Holdout p90 relative compliance error | 0.1024% | 0.1858% |
| Holdout maximum relative compliance error | 0.2856% | 0.8875% |

Pair differences are sampled within +/-3 psi; the expanded collector clips
differences as needed to prevent negative line pressures. Pair sums remain
within the controller's 30 psi limit.

Model, dataset and metrics locations:

- [Initial static model directory](data/compliance_tangent_3000/)
- [Expanded static model directory](data/compliance_tangent_wide_5000/)
- Legacy finite-time fits: `data/compliance_fit_*`

## 5. How the Trained Model Is Used in Control

The online sequence is:

1. Read observed q, estimated velocity, and measured pressure.
2. Compute a position-control prior using inverse-dynamics feedforward PID.
3. Every 15 control ticks, optimize pair means using the trained compliance
   head, and recalculate pressure differences to retain commanded torque.
4. Apply a common-pressure slew limit of 5 psi/s. Mean-pressure bounds are
   taken from the selected head's training metadata.
5. Generate pressure candidates and roll them forward with the Koopman model.
6. Evaluate predicted tip error and predicted xy compliance error, then apply
   the selected pressure command through the simulated valve/ADC interface.

The MPPI compliance cost uses:

```python
Cx = self.compliance_head.predict(xp[:, :12], xp[:, 24:48])
```

The second argument is predicted actual pressure, not simply the candidate
pressure command. The planner also calls the trained head to select its prior.
Exact `C_true` is not fed back into online control. The method is therefore a
hybrid of learned prediction and fitted-physics inverse dynamics, not pure
Koopman-only control.

## 6. Trajectory and Ellipse Changes

Both ellipses now move with the actual tip on the same trajectory plot:

- Red dashed: target compliance.
- Blue solid: independently calculated MuJoCo tangent compliance.
- Green: actual tip trajectory.

The ellipse is the xy displacement locus under a constant-magnitude 0.2 N
in-plane force. Both directions use the same display scale; there is no
anisotropic display stretching. Raw compliance frames are not smoothed.
Separate time plots show Cxx and Cyy, and the title reports their ratio.

| Target configuration | Small variation | Large variation |
| --- | ---: | ---: |
| Cxx and Cyy range | 0.060--0.078 m/N | 0.054--0.098 m/N |
| Peak-to-peak variation per axis | 0.018 m/N | 0.044 m/N |
| Maximum target aspect ratio | 1.30:1 | 1.815:1 |
| Nominal compliance phase period | 9 s | 18 s in final run |

Cxx and Cyy vary in opposite phases. The target amplitude genuinely increased
by approximately 2.44 times. The large-variation run first used a 9-second
period, but pressure slew caused lag. Doubling only the compliance period
reduced this lag without changing the tip reference or its timing.

The final actual ellipse reaches approximately 1.681:1 in both orientations:
horizontal around 6.7 seconds, vertical around 15.4 seconds. It is not forced
to equal the target's 1.815:1 ratio.

## 7. Recorded Results

Errors below are measured during drawing. Compliance errors use the Frobenius
norm of the xy block, normalized by the target xy norm.

| Experiment | Tip RMS | True xy compliance median error | True xy compliance p90 error |
| --- | ---: | ---: | ---: |
| First legacy GIF | 8.325 mm | Not directly comparable | Not directly comparable |
| Revised, small variation | 1.071 mm | 0.925% | 1.417% |
| Large variation, 9 s compliance period | 1.359 mm | 8.651% | 15.110% |
| Large variation, 18 s compliance period | 1.527 mm | 3.143% | 4.326% |

For the final large-variation run, Cxx/Cyy correlations with the target are
0.9903 and 0.9893. Tip p95 error is 2.902 mm. The maximum requested antagonist
pair sum is 29.224 psi, below the 30 psi cap.

A separate 12-second same-reference ablation compared common-pressure control:

| Ablation | Tip RMS | True xy compliance median error |
| --- | ---: | ---: |
| Common-pressure control disabled | 1.078 mm | 9.151% |
| Common-pressure control enabled | 1.054 mm | 0.951% |

The bigger visual change has a real accuracy tradeoff: the final wide-range
run is less accurate than the smaller-range run. No result was relabelled to
hide that difference.

## 8. Checks Against Circular Evaluation

The controller uses the trained head. The blue ellipse is evaluated afterwards
from recorded actual q and pressure using MuJoCo torque finite differences,
without using the target or trained head as its output.

Checks performed on the small-variation run included:

- Recomputing all 300 compliance frames exactly reproduced the saved truth cache.
- Recomputed head predictions matched the saved predictions numerically.
- Blocking exact-compliance calls during online control did not prevent control;
  the trained head was called 1101 times in a 40-command check.
- Multiplying head outputs by 1.2 in memory, with identical recorded inputs and
  references, changed commanded pressures by up to about 1.67 psi. This was a
  sensitivity check, not a new closed-loop performance benchmark.
- A separate nonlinear equilibrium test under small positive/negative Cartesian
  forces agrees with the tangent calculation at the tested configuration.

The logged `compliance_pred` field uses actual simulation states for offline
readout evaluation; those logged values are not passed back to the controller.

Training and truth evaluation still share the same nominal digital twin.
These checks establish data-flow separation and simulator consistency, not
independence from the twin's physical assumptions or accuracy on real hardware.

## 9. Main Source Changes

| File | Change |
| --- | --- |
| [control/trajectory.py](control/trajectory.py) | Figure-eight reference, compliance schedule, reachable dome, smooth ramps, independent compliance period, batched tip Jacobian |
| [control/controller.py](control/controller.py) | Optional future tip/compliance arguments and inverse-dynamics torque-map access for allocation |
| [control/koopman.py](control/koopman.py) | Tip/compliance costs, trained-head selection, common-pressure planner and common-mode sampling |
| [control/compliance_planner.py](control/compliance_planner.py) | Learned-head mean-pressure optimization with torque-preserving reallocation |
| [control/benchmark.py](control/benchmark.py) | Compliance references/predictions, xy metrics, figure-eight controller options and trace logging |
| [control/render_fig8_compliance_gif.py](control/render_fig8_compliance_gif.py) | Independent truth calculation, tip-centred ellipses, cache checks, GIF/PNG/MP4 export and target configuration |
| [control/test_tangent_compliance.py](control/test_tangent_compliance.py) | Mechanics, head, allocation, reachability, ellipse and reference-period regression tests |
| [digital_twin/tangent_compliance.py](digital_twin/tangent_compliance.py) | Exact local stiffness and plate-centre compliance evaluator |
| [digital_twin/train_tangent_compliance.py](digital_twin/train_tangent_compliance.py) | Static label collection, polynomial fit, metadata and configurable pressure domain |
| [digital_twin/compliance_head.py](digital_twin/compliance_head.py) | Initial legacy direct-compliance readout retained for provenance |
| [digital_twin/train_compliance_head.py](digital_twin/train_compliance_head.py) | Initial finite-time collector, now explicitly marked legacy |
| [.gitignore](.gitignore) | Explicitly include compliance NPZ artifacts and figure-eight MP4s |
| [.gitmodules](.gitmodules) | Pin the unmodified older reference repository as a submodule |

## 10. Reproduction

Run from the project root. The following commands use the existing Windows
virtual environment; a new checkout must install `requirements.txt` in its
own environment. Exact numeric results can vary with library/platform versions.

### Retrieve the Reference Repository

```powershell
git submodule update --init --recursive
```

### Train the Initial Static Head

```powershell
.\.venv\Scripts\python.exe -m digital_twin.train_tangent_compliance
```

### Train the Expanded Head

```powershell
.\.venv\Scripts\python.exe -m digital_twin.train_tangent_compliance --samples 5000 --mean-min-psi 0.3 --mean-max-psi 14.5 --out data/compliance_tangent_wide_5000
```

### Run the Small-Variation Experiment

```powershell
.\.venv\Scripts\python.exe -m control.render_fig8_compliance_gif --duration-s 30 --fps 10 --out-dir deliverable/fig8_compliance_revised_30s --mp4
```

### Run the Final Large-Variation Experiment

```powershell
.\.venv\Scripts\python.exe -m control.render_fig8_compliance_gif --duration-s 30 --fps 10 --compliance-soft 0.098 --compliance-hard 0.054 --compliance-period-s 18 --compliance-head data/compliance_tangent_wide_5000/head.npz --out-dir deliverable/fig8_compliance_wide_slow_30s --mp4
```

These output-directory examples reuse the recorded experiment locations;
choose a different `--out-dir` to preserve existing results during a new run.
Append `--reuse-trace --reuse-true-compliance` to render the saved run instead
of rerunning control. This renders the saved targets, not newly supplied ones.

### Run the Focused Tests

```powershell
.\.venv\Scripts\python.exe -m pytest control/test_tangent_compliance.py control/test_trajectory.py control/test_controller.py control/test_benchmark.py -q -p no:cacheprovider
```

The latest executed test suite passed 29 tests. Final MP4 exports were checked
by full decoding and have 300 frames at 10 fps, 30 seconds total, 1200 x 700.

## 11. GitHub Publication

- Original repository: <https://github.com/zuorunze/UMArm_koopman_compliance_control_espproject>
- User-owned fork: <https://github.com/naikewu/UMArm_koopman_compliance_control_espproject>
- Experimental/default fork branch: `feature/fig8-tip-compliance-20260913`
- Published experimental implementation commit: `f542dc22398279de3ad1b81f1859a29639796b85`
- Local `origin` points to the user-owned fork; `upstream` points to the original repository.

The original publication created a branch in the original repository. A true
fork was subsequently created in the user's account, and the experimental
branch was made its default. The original repository's experimental branch
still exists; its `main` was not changed by this publication.

The implementation publication contains 65 changed paths, approximately 59 MB
of staged file contents, including datasets, models, experiment traces and
videos. Virtual environments, caches and Git object databases are excluded.
The nested reference repository's files are retrieved through its submodule.

## 12. Remaining Limits

- These are nominal digital-twin experiments, not new real-hardware tests.
- The controlled endpoint is the final plate centre, not the visible 50 mm stub.
- Only xy compliance is targeted; z and xz/yz coupling are not independent targets.
- The previous 0.035--0.050 m/N target is not claimed solved by changing targets.
- A 0.2 N display scale describes a linear tangent ellipse, not a measured
  finite-amplitude displacement under a continuously applied 0.2 N force.
- High fit accuracy applies to the sampled domain and shared nominal model;
  it does not establish robustness to mechanical/actuator mismatch.
- Solver p99 is approximately 89.35 ms in the final wide-range run, above the
  6.67 ms budget for 150 Hz. Simulation timestamps do not demonstrate realtime control.
- Dynamic admittance, physical collision response and force-disturbed trajectory
  tracking have not been established by these figure-eight compliance plots.
