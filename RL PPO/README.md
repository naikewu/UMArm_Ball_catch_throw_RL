# UMArm Three-Finger Catch-and-Throw PPO

This folder contains the new hierarchical reinforcement-learning benchmark for
the fitted UMArm digital twin. PPO plans the end-effector motion, directional
compliance, follow-through, and release. The existing Koopman-MPPI controller
alone converts those references into 24 pressure commands. PPO never writes
joint positions or pressures directly.

The physical task uses a side-to-side layout. In every curriculum stage, the
ball starts 0.50-0.70 m away on the negative world-y side and travels mainly in
positive y; it is never launched from underneath or placed artificially close
to the gripper. Launch height stays within 30 mm of the home gripper in the
full distribution. The fixed target is on the opposite side at the same
nominal height, using the home-relative offset `[0.00, +0.55, 0.00] m`.
MuJoCo gravity remains enabled and launch velocity is solved ballistically.

The old direct-pressure virtual-catch code remains only as a software baseline.
The primary path is `rl_ppo.train_catch_throw`.

## Physical Grasp Model

`rl_ppo.physical_scene` adds a free 25 g ball and a rigidly mounted open
three-finger gripper to a separate copy of the fitted MJCF. Existing arm
self-collision remains disabled. Ball/gripper collision is solved by MuJoCo and
sampled at the 1 kHz physics rate, including contacts that begin and end between
two 150 Hz controller observations.

Finger closure is a discrete physical event, not a distance-only catch. It
requires all of the following:

- a real MuJoCo ball/gripper contact;
- the ball centre inside the current curriculum's three-finger capture region;
- gripper and ball velocity directions sufficiently aligned; and
- sufficiently low relative speed.

These gates run in the same 1 kHz physics quantum as the contact, rather than
waiting for the next 150 Hz controller observation.

When those conditions hold, the environment activates a previously disabled
MuJoCo weld at the ball's current relative pose. No ball position or velocity
is assigned during closure. The ball then acts as a 25 g end-effector payload.
Release only disables the weld, preserving the physical linear and angular
velocity at that instant.

While the weld is active, redundant ball/finger collision pairs are disabled
so internal holding force is not counted again as impact impulse. They are
restored at release.

This is the intended abstraction of a real three-finger hand closing firmly
around the ball. Finger motor dynamics are not yet modelled; open/close energy
is therefore an explicit assumption in `physical_task_contract.json`.

## Continuous Motion

The reference generator first asks the gripper to match the predicted velocity
of the incoming ball. After grasp, it continuously blends that velocity toward
the ballistic release velocity required by the fixed target:

```text
v_release = (target - release_position - 0.5 * gravity * T^2) / T
```

There is no settle-and-rethrow phase and no reward for stopping. A penalty is
applied when the gripper loses too much speed in the incoming momentum
direction during follow-through. Follow-through duration and release flight
time are sampled by PPO and locked at grasp, so they cannot jitter every 30 Hz
policy step.

Directional compliance is always constrained to
`c_parallel >= c_transverse`: the arm may be soft along the incoming ball path
while remaining tighter laterally. A force-driven admittance correction runs
at 150 Hz above the static tangent-compliance target.

## Curriculum And Reward

Every PPO rollout uses one fixed task distribution. Every 5-10 completed PPO
updates, the current policy is evaluated deterministically on 20 held-out,
fixed-seed episodes. A stage cannot advance until it has received at least 20
updates and reaches 80 percent evaluation success:

1. 0.85-1.05 m launch with a 10-25 mm offset grasp;
2. 0.78-1.10 m launch through 2A/2B/2C with increasing disturbances;
3. 0.90-1.10 m launch, broader approach geometry, follow-through, and release;
4. 1.00-1.25 m launch and release into a broad far-side target;
5. 1.10-1.40 m position/speed/mass distribution, 60 mm target, force and energy optimization.

V9 samples ballistic flight time so impact direction remains close to the
catcher aperture axis. Its cubic rendezvous reference has analytically
consistent position and velocity, and pre-contact reward tracks both. A missed
catch ends 0.20 s after the scheduled intercept so rollouts do not spend most
of their samples following an escaped ball.

Before grasp, the reward uses radial error where the ball reaches the gripper
entrance plane. A swept sphere-versus-capsule calculation also predicts the
earliest collision with each of the three fingers, with a palm-face fallback.
This prevents a trajectory from receiving a good score for passing near the
centre only after it would already have hit an outer finger. Velocity matching
is rewarded only near a geometrically valid predicted contact. A first physical
contact gives `+3`; radial, axial, speed, and alignment gate violations receive
direct squared penalties; and a valid grasp receives a further `+10`. During
follow-through, predicted ballistic landing error provides a dense target
signal. Contact impulse, excessive peak force, abrupt policy changes, stopping
after grasp, and later-stage energy use are penalized.

The observation remains at 81 values. In addition to arm, pressure, ball,
target, contact, compliance, phase, and time state, it includes predicted entry
position xyz in the gripper frame, time to entry, entry radial error, and
predicted first-contact radial error.

The actor and critic are separate `384-384-384 SiLU` networks. Stage 1 exposes
only the six trajectory actions, Stage 2 exposes those plus two compliance
actions, and Stages 3-5 expose all ten. Masked dimensions are zero and are
excluded from PPO log probability and entropy. PPO uses initial
`log_std=-1.2`, learning rate `1.2e-4`, five epochs, clip ratio `0.16`, entropy
coefficient `0.0015`, and a `0.018` target-KL early stop.

Action authority opens gradually at `18%` in Stage 1, `20/35/50%` in Stage
2A/2B/2C, `65%` in Stage 3, `85%` in Stage 4, and `100%` in Stage 5. This keeps
the initial far-launch reference reachable while requiring PPO to improve the
disturbed stages.

The online energy proxy is:

```text
E_proxy = sum_i integral(abs(F_muscle_i * dl_i)) + E_finger_close/open
```

It is mechanical boundary work plus assumed gripper actuation energy. It is
not compressor electrical energy. A supply-energy claim requires calibrated
valve mass flow, thermodynamics, and compressor efficiency.

## Commands

Run from this folder:

```powershell
# First time only; this reuses the repository .venv.
.\setup.ps1

# Physics, PPO, and Koopman-MPPI integration check.
..\.venv\Scripts\python.exe -m rl_ppo.smoke_catch_throw

# Diagnose catch learning before a long run.
.\run_v9_rendezvous_stage1_probe.ps1 -TotalSteps 20000

# Automatic five-stage V9 training run.
.\run_v9_rendezvous_full_curriculum.ps1 -TotalSteps 120000

# Equivalent direct command. New outputs use runs/catch_throw_v9_rendezvous.
..\.venv\Scripts\python.exe -m rl_ppo.train_catch_throw --total-steps 100000

# Hold one stage for debugging.
.\run_physical_training.ps1 -TotalSteps 20000 -StartStage 1 -FixedStage

# Evaluate the complete final-stage task.
..\.venv\Scripts\python.exe -m rl_ppo.evaluate_catch_throw `
  --checkpoint runs\catch_throw_v9_rendezvous_full_curriculum\ppo_latest.pt --episodes 20 --curriculum-stage 5
```

Per-update progress is written immediately to `metrics.jsonl`; the latest model
is written to `ppo_latest.pt`, and the best deterministic-evaluation checkpoint
is written to `ppo_best.pt`. The untouched reference-led initialization is saved
as `ppo_initial.pt`. Rendezvous, first-contact, best pre-grasp, and
grasp-event diagnostics are logged separately. Earlier checkpoints remain
untouched. V9 writes to a separate directory and does not resume an old policy
because task ballistics, reward credit, and reference timing changed.
Stage 2 promotion is based on deterministic capture rate for full-curriculum
probing; soft-catch speed/alignment/radial metrics remain logged and shaped but
do not block the automatic run.

See `CATCH_AND_THROW_PHYSICAL_PLAN.md` for the modelling rationale and
`physical_task_contract.json` for the exact current assumptions.
