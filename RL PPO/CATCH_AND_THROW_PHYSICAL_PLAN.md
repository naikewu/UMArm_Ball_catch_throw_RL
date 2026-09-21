# Three-Finger Catch-and-Throw Plan

## Objective

Train UMArm to receive a moving ball softly, continue in the incoming momentum
direction, redirect that motion, and release the ball into a fixed target
region. The maneuver must be one continuous motion. Catching, stopping, and
starting a separate throw is not the desired solution.

The hierarchy is fixed:

```text
ball/target/arm/contact state
        -> PPO trajectory and compliance intent at 30 Hz
        -> continuous reference and IK
        -> Koopman-MPPI pressure control at 150 Hz
        -> fitted pneumatic MuJoCo arm at 1 kHz physics
```

PPO never sets `q`, `qdot`, or chamber pressure. It can only influence the
reference tracked by the existing low-level controller.

## Task Geometry

The launcher and target are on opposite horizontal sides of the robot. In all
five curriculum stages, the ball begins 0.50-0.70 m from the interception point
on the negative world-y side. Even the easiest task is a real far-field catch,
not a ball placed next to or underneath the gripper. Launch height remains
within 30 mm of the home gripper in the full distribution. The gripper's local
entry axis is mounted along the nominal incoming direction `[0, +1, -0.75]`.
The default target is 0.55 m farther in positive y at the same nominal height
as the home gripper, on the side opposite the launcher.

Gravity is active throughout. "Horizontal" describes the side-to-side task
layout and dominant travel direction, not an artificial straight line with
gravity disabled.

## Gripper Model

The catch-specific MJCF adds a palm and three equally spaced collision-enabled
fingers to the final plate. There are no passive cup slides and no artificial
three-dimensional spring mount. Before closure, the ball is a normal MuJoCo
free body and all impacts are solved by the contact solver.

Closure models a real three-finger hand completing a firm grasp. It occurs only
when:

1. MuJoCo reports ball/gripper contact;
2. the ball centre is inside the allowed three-finger region;
3. ball and gripper velocity directions are aligned; and
4. their relative speed is below the stage threshold.

The environment then activates a disabled weld equality at the measured
relative pose. It does not write the ball pose or velocity. This makes the ball
and gripper one rigid assembly after closure while keeping the impact and
pre-grasp momentum physical. A low relative velocity is important because a
real finger closure also has to remove any remaining relative momentum.

The four closure tests execute at the 1 kHz MuJoCo step where contact exists;
the 150 Hz policy loop receives a latched event afterward. A short collision is
therefore never converted into a delayed proximity-only grasp.

Release deactivates the equality constraint. The free ball retains its current
linear and angular velocity and continues under gravity. Finger motor dynamics
are outside version 3; fixed open/close energy assumptions are logged
separately and must later be replaced by measurements.

Ball/finger collision is disabled only while the weld is active, because the
weld already represents the closed fingers and keeping both constraints would
double-count internal holding force. Collision is restored before free flight.

## Momentum And Compliance

At interception, the nominal gripper velocity is the predicted ball velocity,
plus a bounded PPO correction. This directly optimizes the desired soft catch:

```text
relative_velocity = ball_velocity - gripper_velocity
impact_impulse = integral(contact_force dt)
```

Reducing relative velocity reduces the momentum change that the fingers and arm
must absorb. The desired static Cartesian compliance is constrained to:

```text
C_ref = c_parallel n n^T + c_transverse (I - n n^T)
c_parallel >= c_transverse
```

Here `n` is the incoming direction. The arm can therefore be softer along the
impact path without becoming equally loose laterally. The learned compliance
head and MPPI track this static target. A separate force-driven admittance
shaper adjusts the tip reference during impact.

After grasp, the reference velocity begins at the captured ball velocity. It
is smoothly redirected toward the ballistic velocity for target point `g`:

```text
v_release = (g - x_release - 0.5 * gravity * T^2) / T
```

Follow-through duration and flight time are frozen when grasp occurs. This
prevents policy noise from changing the release plan every 33 ms. There is no
intermediate zero-velocity target.

## Contact And Force Accounting

MuJoCo advances at 1 kHz while MPPI observes at 150 Hz. Contact force is
integrated every physics quantum so a short impact cannot disappear between
controller samples. Each 150 Hz state includes:

- average world-frame contact force and window peak force;
- window and cumulative contact impulse;
- contact sample/quanta counts;
- `J_position^T F_contact` generalized torque;
- MuJoCo equality/contact constraint torque;
- ball momentum, kinetic energy, pose, linear velocity, and angular velocity.

The Jacobian mapping is an analysis and control signal. MuJoCo's constraint
solver remains the source of physical contact and grasp forces.

## Action And Observation

The 10-dimensional PPO action is:

```text
intercept offset xyz                         3
phase velocity adjustment xyz               3
incoming-direction softness                 1
transverse compliance                       1
follow-through duration                      1
release flight time                          1
```

Actions are low-pass filtered. The two trajectory-level timing values are
frozen at grasp; fast motion and compliance components remain available to the
reference generator. The xyz velocity residual adjusts contact matching before
grasp and the target-directed release velocity after grasp. Observations contain the 12 joint angles, 12 joint
velocities, 24 pressures, gripper and ball state, target displacement,
directional compliance, contact force, phase, ball mass, and remaining time.
Six engineered values describe collision-aware interception: gripper-frame
entry position xyz, time to entry, entry radial error, and predicted first
physical-contact radial error. The complete observation remains at 81 values.

Only actions needed by the active curriculum participate in the policy
probability: Stage 1 uses offset and velocity actions (6), Stage 2 also uses the
two compliance actions (8), and Stages 3-5 use all 10. Inactive actions are
fixed at zero and excluded from log probability and entropy.

The active action amplitude also opens with the curriculum: 5 percent in Stage
1, 10/15/20 percent through Stage 2A/2B/2C, 40 percent in Stage 3, 65 percent in
Stage 4, and 100 percent in Stage 5. This preserves a successful nominal
far-launch trajectory under initial exploration without making the disturbed
stages automatically successful.

## Phase-Gated Reward

The reward intentionally exposes each prerequisite before precision throwing:

```text
relative_position = ball_position - gripper_position
relative_velocity = ball_velocity - gripper_velocity
relative_acceleration = rotate_world_to_gripper(gravity)
z_entry = z_finger_tip - ball_radius - finger_radius
t_entry = positive_root(
    relative_position_z + relative_velocity_z * t
    + 0.5 * relative_acceleration_z * t^2 - z_entry)
radial_entry = norm((relative_position + relative_velocity * t_entry
                     + 0.5 * relative_acceleration * t_entry^2)_xy)
t_contact = earliest piecewise-ballistic swept-sphere hit against the fingers

R = R_entry_radial_progress
  + R_predicted_first_contact
  + R_velocity_match
  + 3 * first_physical_contact
  + 10 * valid_grasp_event
  + R_follow_through
  + R_release_event
  + R_ballistic_target
  - lambda_impulse * integral(norm(F_contact) dt)
  - lambda_peak * max(0, peak_force - threshold)^2
  - lambda_gate * grasp_gate_violation^2
  - lambda_stop * lost_forward_momentum^2
  - lambda_smooth * norm(action_t - action_t-1)^2
  - lambda_energy * E_proxy
```

Near the gripper, the finger predictor subdivides the gravity-aware arc and
solves moving-point intersections against each capsule expanded by the ball
radius. A palm-face candidate covers a centred trajectory.
Velocity matching is gated by time and predicted radial validity, so it cannot
dominate before geometry is correct. At real 1 kHz contact, radial, axial,
relative-speed, and alignment violations are penalized and logged separately.

The target term uses predicted ballistic landing error while the ball is held,
so throw-related actions receive feedback before a rare successful release.
Terminal target success still depends on the actual released MuJoCo ball
entering the region. Energy is disabled in the first three curriculum stages so
the policy cannot improve reward by refusing to move.

## Energy Scope

The implemented proxy is:

```text
W_muscle_abs = sum_i integral(abs(F_muscle_i dl_i))
E_proxy = W_muscle_abs + E_finger_close + E_finger_open
```

`W_muscle_abs` measures absolute mechanical boundary work in the simulated
pneumatic muscles. The finger terms are explicit uncalibrated assumptions.
Neither quantity equals electrical supply energy. A validated supply metric
requires inlet/vent mass-flow curves, supply pressure, gas-state modelling,
valve losses, and compressor efficiency. Policies must be compared at equal
catch and target success before comparing this proxy.

## Success-Gated Curriculum

The training script never changes difficulty inside a rollout. Every 10 PPO
updates it runs 20 deterministic held-out episodes with fixed seeds. Each
difficulty needs at least 20 PPO updates and 80 percent held-out success:

1. Fixed 0.55 m launch, fixed speed, centered velocity-matched physical grasp.
2. 0.50-0.60 m launch through 2A/2B/2C with increasing small position and
   velocity disturbances.
3. 0.50-0.62 m launch with broader speed variation, momentum-following motion,
   and physical release.
4. 0.50-0.65 m launch with broader disturbances and a broad far-side target.
5. 0.50-0.70 m launch with the full position, speed, and 20-35 g mass
   distribution, final 60 mm target, strict grasp thresholds, force penalties,
   and the full energy-proxy weight.

Launch distance remains realistic in every stage. Difficulty increases through
position spread, speed range, intercept error, release, target accuracy, mass,
force, and energy instead of an artificially short launch. Stage 5 uses the
full 1.4-2.7 m/s and 20-35 g distribution with strict grasp geometry. A fixed
stage can be selected for debugging. Stage 1 also requires mean predicted entry
radial error no larger than 25 mm before promotion. First-contact ball-centre
radius is logged separately because it is set partly by finger geometry.

The actor and critic remain separate two-layer, 256-unit Tanh networks. PPO v6
starts at `log_std=-1.2`, uses learning rate `1e-4`, four update epochs, clip
ratio `0.15`, entropy coefficient `0.0005`, and target KL `0.015`. This focuses
the change on observability, credit assignment, and exploration instead of
adding network capacity before those causes have been tested.

## Validation Gates

Completed software gates:

- the model has exactly three named fingers, one free ball, and an initially
  disabled grasp equality;
- pre-grasp contact is physical and sampled at 1 kHz;
- enabling grasp does not assign ball `qpos` or `qvel`;
- a grasped ball maintains its relative pose while the arm moves;
- release preserves the free-joint velocity;
- the easiest stage has a deterministic physically reachable grasp;
- PPO, reference generation, Koopman-MPPI, pressure projection, and MuJoCo run
  end to end; and
- tests cover the 3D compliance interface and pressure safety.

Required before hardware transfer:

1. Measure the real finger geometry, closure time, and open/close energy.
2. Calibrate ball/finger restitution, friction, and contact damping.
3. Validate the static compliance prediction in the incoming direction.
4. Add grasp state and payload mass explicitly to a contact-aware Koopman model
   or switch contact-phase prediction to short MuJoCo rollouts.
5. Evaluate held-out launch speed, direction, mass, target, and model parameter
   distributions.
6. Enforce real-system force, workspace, pressure, and emergency-stop limits.

Simulation success is not evidence that an unguarded real impact is safe.
