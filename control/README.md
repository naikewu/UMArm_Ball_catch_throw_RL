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

The mechanism has two antagonistic muscles per joint. Their pressure difference
produces torque, while their combined pressure changes joint stiffness. The
vanilla PID commands this difference from angle error, accumulated error, and
velocity error. Its integral is limited when the pressure command saturates.
The feedforward PID first estimates the pressure required by the desired motion
using the fitted mass, gravity, passive stiffness, damping, and tendon geometry;
feedback then corrects the remaining error. Preview compensates for pressure
response delay. These nominal model parameters are available to both model-based
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
.venv/Scripts/python.exe -m control.train --epochs 80 --steps 40 --batch 512 --horizon 30 --device cuda
```

The 96 episodes cover independent and coupled pressure steps, chirps,
multisines, co-contraction, ringdowns, joint tracking, and external force pulses.
Parameter perturbations vary mass, stiffness, damping, force gain, valve gains,
and leakage. These ranges are assumed robustness scenarios. They do not amount
to exhaustive modal coverage or fitted confidence intervals. Whole episodes
are assigned to training, validation, and test sets. Validation chooses the
checkpoint; held-out test episodes are scored after that choice.

Gain selection uses a separate joint multisine instead of the writing path:

```powershell
.venv/Scripts/python.exe -m control.tune --method pid --out data/control_pid_tuning.json
.venv/Scripts/python.exe -m control.tune --method pid --refine --out data/control_pid_refine.json
.venv/Scripts/python.exe -m control.tune --method ff_pid --out data/control_ff_tuning.json
```

Reproduce the common-time writing benchmark and comparison videos:

```powershell
.venv/Scripts/python.exe -m control.benchmark
.venv/Scripts/python.exe -m control.benchmark --randomized --out deliverable/dynamic_control/benchmark_perturbed
.venv/Scripts/python.exe -m control.render_comparison video --speed slow
.venv/Scripts/python.exe -m control.render_comparison video --speed fast
.venv/Scripts/python.exe -m control.render_comparison figures
.venv/Scripts/python.exe -m control.valve_probe
```

The benchmark clock advances the existing 1 ms MuJoCo plant with 150 Hz command
edges. Computation time is measured separately: offline tracking does not
include operating-system delays or CAN reply delivery time. The GUI acceptance
test measures the actual controller process, bus cycle, and mocap rates:

```powershell
.venv/Scripts/python.exe hw_tests/gui_controller_sim_test.py --help
.venv/Scripts/python.exe -m pytest control digital_twin/test_sim_mocap.py -q
```

The warm-start training interface supports a later adaptation stage:

```powershell
.venv/Scripts/python.exe -m control.train --data path/to/real_episodes --warm-start control/checkpoints/canarm_koopman.pt --freeze-encoder --anchor-strength 0.01 --out control/checkpoints/canarm_koopman_adapted.pt
```

Each input `episode_*.npz` must contain `state` with shape `(N+1,48)`, `action`
with shape `(N,24)`, and a JSON string `meta`. State order is
`q12,qdot12,p24_Pa_gauge`; actions are Pa gauge in ascending CAN ID order.
Metadata requires `episode`, `family`, `dt_s=1/150`, and
`environment.board_variants`. Preserve each physical episode as a split unit;
do not divide one recording into adjacent training and test fragments. Use
causal camera/velocity processing matching the controller and each recorded
board's pressure calibration. The current batch trainer requires equal episode
lengths and the checkpoint's fixed sample period, so irregular or dropped
samples need explicit preprocessing before this command can be used.

Warm-start adaptation preserves the original normalization, optionally freezes
the encoder, and penalizes movement away from the previous parameters. Keeping
some simulation episodes in an adaptation dataset can limit forgetting; reserve
separate real episodes to assess whether transfer improves. The current release
demonstrates simulation control and exposes this training path. It has not
validated an adapted controller on hardware.
