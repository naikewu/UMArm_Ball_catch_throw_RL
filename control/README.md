# ProMax dynamic control in simulation

This package runs three selectable pressure controllers against the fitted CAN
digital twin: `pid`, `ff_pid`, and `koopman_mppi`. The measured actuator map and
the ProMax's `yx` joint convention are shared with the existing operator GUI.
The controller GUI is SIM-only for this implementation.

Start the operator window from the repository root:

```powershell
.venv/Scripts/python.exe canarm_control_gui.py --sim
```

Connect the SIM adapter, scan all 24 nodes, select a controller, and start it.
The target window contains 12 joint sliders, a Cartesian tip target in metres
relative to the robot base, and positive/negative axis nudges. The controlled
tip is the centre of the final plate. Inverse kinematics reports unreachable
requests rather than promising an arbitrary tip position. Closing the target
window stops the controller. The worker process, applied command rate, and
computation time are visible in the target window.

Startup includes a discarded solver warmup while the simulated actuators are
disabled, followed by a fresh-state reset. This resolves the optimizer's lazy
imports before any command reaches the plant; the 150 ms stale-state check
still applies during control. The [final acceptance record](results/gui_acceptance.json)
reports active timing separately from warmup and retains each controller's
tracking bias.

While the sliders permit requests within +/-25 degrees, this interval is a
kinematic input bound. It does not establish that every requested pose can be
held under the 30 psi individual and antagonistic-pair caps. The training
campaign's largest commanded pressure was 17.73 psi, compared with the 30 psi
runtime cap ([campaign summary](checkpoints/campaign_summary.json)). Predictions
above that observed pressure range extrapolate from the collected data. The
window therefore displays measured joint error and tip-to-request error
separately from the inverse-kinematics residual, so an operator can distinguish
geometric reachability from convergence of the pressure controller.

An earlier [mixed-posture trial](results/gui_acceptance_mixed_postures.json)
increased MPPI tip error from 9.75 to 17.55 mm after a 10 mm jog at a larger warm
pose. That trial used different starting poses for the three controllers and
does not provide an equal-condition ranking. The fresh-plant GUI acceptance
uses a common joint target and reports motion, remaining bias, and measured
timing; its success does not establish convergence over the full slider range.

The mechanism has two antagonistic muscles per joint. Their pressure difference
produces torque, while their combined pressure changes joint stiffness. The
vanilla PID commands this difference from angle error, accumulated error, and
velocity error. Its integral is limited when the pressure command saturates.
The feedforward PID first estimates the pressure required by the desired motion
using the fitted mass, gravity, passive stiffness, damping, and tendon geometry;
feedback then corrects the remaining error. A bounded torque allocation retains
the 6 psi pair mean at small demands and vents the opposing muscle when larger
torques require it. It resolves the torque equation after that change, since
simply clipping a negative pressure would discard part of the requested torque.
Preview compensates for pressure response delay. These nominal model parameters
are available to both model-based
controllers, which makes this a comparison within the fitted twin.

The Koopman model lifts the 48 observed state variables (12 angles, 12 filtered
velocities, and 24 pressures) into 96 coordinates, retaining the physical state
and adding 48 learned observables. A linear update plus a low-rank bilinear
state/input term predicts how those coordinates change under all 24 pressure
commands. The state/input term allows pressure authority to depend on the
current configuration. This finite representation approximates the dynamics;
it does not establish an invariant Koopman subspace.

MPPI evaluates sampled pressure sequences over a short horizon and weights
them by their predicted tracking and command costs. Feedforward PID supplies
the nominal sequence. Only the first optimized command is applied, and the
controller replans from the next pressure and camera observation. A bounded
innovation correction, the difference between the predicted and subsequently
observed state, compensates for small persistent model errors. Pressure limits
are checked again after converting targets to each board's ADC calibration.

The benchmark samples camera frames at 240 Hz with provisional 0.10 degree
independent joint noise and 8.33 ms latency. A causal 40 ms filter estimates
velocity. Simulation and GUI share this observer. Feedback uses these observed
states; the uncorrupted plant state is stored separately to score the result.
No measurement of this arm's mocap noise is implied by these settings.

Reproduce the simulation dataset and training with the workspace GPU:

```powershell
.venv/Scripts/python.exe -m control.collect --episodes 84 --seconds 20 --workers 24
.venv/Scripts/python.exe -m control.collect --start 84 --episodes 12 --seconds 20 --workers 12 --family disturbance
.venv/Scripts/python.exe -m control.train --epochs 80 --steps 40 --batch 512 --horizon 30 --device cuda --out control/checkpoints/canarm_koopman_retrained.pt
```

The 96 episodes cover independent and coupled pressure steps, chirps,
multisines, co-contraction, ringdowns, joint tracking, and external force pulses.
Parameter perturbations vary mass, stiffness, damping, force gain, valve gains,
and leakage. These ranges are assumed robustness scenarios. They do not amount
to exhaustive modal coverage or fitted confidence intervals. Whole episodes
are assigned to training, validation, and test sets. Validation chooses the
checkpoint; held-out test episodes are scored after that choice.

The reproduction command writes a separate checkpoint. To evaluate those new
weights, pass `--checkpoint control/checkpoints/canarm_koopman_retrained.pt`
to `control.benchmark`; the default comparison uses the shipped checkpoint.

Feedforward and MPPI settings use a separate 6-degree joint multisine. The
initial PID gains passed that small-motion test but became unstable on the
full word. The final PID gains, 85/8/12 in psi-based radian units, were selected
using independent 10–14 degree postures, reversals, multisines, and settling
tests on nominal and perturbed plants. The final six cases include a faster
boundary test; passing these cases is empirical evidence, not a stability proof:

```powershell
.venv/Scripts/python.exe -m control.tune --method pid --out data/control_pid_tuning.json
.venv/Scripts/python.exe -m control.tune --method pid --refine --out data/control_pid_refine.json
.venv/Scripts/python.exe -m control.tune --method ff_pid --out data/control_ff_tuning.json
.venv/Scripts/python.exe -m control.tune_pid_stress --configs control/checkpoints/pid_stress_confirmation_configs.json --confirm --workers 3 --out data/control_pid_final_confirmation.json
```

Reproduce the common-time writing benchmark and comparison videos:

```powershell
.venv/Scripts/python.exe -m control.benchmark
.venv/Scripts/python.exe -m control.benchmark --randomized --out deliverable/dynamic_control/benchmark_perturbed
.venv/Scripts/python.exe -m control.render_comparison video --speed slow
.venv/Scripts/python.exe -m control.render_comparison video --speed fast
.venv/Scripts/python.exe -m control.render_comparison figures
.venv/Scripts/python.exe -m control.render_comparison figures --folder deliverable/dynamic_control/benchmark_perturbed --condition perturbed
.venv/Scripts/python.exe -m control.summarize
.venv/Scripts/python.exe -m control.summarize --folder deliverable/dynamic_control/benchmark_perturbed
.venv/Scripts/python.exe -m control.valve_probe
```

The benchmark clock advances the existing 1 ms MuJoCo plant with 150 Hz command
edges. Computation time is measured separately: offline tracking does not
include operating-system delays or CAN reply delivery time. The GUI acceptance
test measures the actual controller process, bus cycle, and mocap rates:

```powershell
.venv/Scripts/python.exe hw_tests/gui_controller_sim_test.py --screenshots
.venv/Scripts/python.exe -m pytest control digital_twin/test_sim_mocap.py -q
```

The warm-start training interface supports a later adaptation stage:

```powershell
.venv/Scripts/python.exe -m control.train --data path/to/real_episodes --warm-start control/checkpoints/canarm_koopman.pt --freeze-encoder --anchor-strength 0.01 --out control/checkpoints/canarm_koopman_adapted.pt
```

Each input `episode_*.npz` must contain `state` with shape `(N+1,48)`, `action`
with shape `(N,24)`, and a JSON string `meta`. State order is
`q12,qdot12,p24_Pa_gauge`; actions are Pa gauge in ascending CAN ID order.
The [metadata example](assets/episode_metadata_example.json) lists the required
fields: episode/family, state order, measured antagonist indices, ascending
board IDs and variant bytes, sample rates, pressure-observation definition,
and camera-noise/latency/filter settings. Replace its assumed sensor values
with the recording's documented estimates. One model requires a fixed valve
layout, at least seven episodes total, and at least three episodes per family.
Preserve each physical episode as a split unit;
do not divide one recording into adjacent training and test fragments. Use
causal camera/velocity processing matching the controller and each recorded
board's pressure calibration. The current batch trainer requires equal episode
lengths exceeding both the training and 75-step evaluation horizons, and the
checkpoint's fixed sample period, so irregular or dropped
samples need explicit preprocessing before this command can be used.

Warm-start adaptation preserves the original normalization, optionally freezes
the encoder, and penalizes movement away from the previous parameters. Keeping
some simulation episodes in an adaptation dataset can limit forgetting; reserve
separate real episodes to assess whether transfer improves. The current release
demonstrates simulation control and exposes this training path. It has not
validated an adapted controller on hardware.
