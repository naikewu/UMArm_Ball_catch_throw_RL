# Mocap State Observer

This document describes how the backend turns ~360 Hz OptiTrack/NatNet rigid-body
poses into a 12-DOF joint state `(q, qdot)` sampled at the 150 Hz CAN sync mark.

> This is the piece that was missing in earlier copies of this bundle. A previous
> portable backend shipped a *placeholder* observer that always returned zero
> `q`/`qdot`, and a `mocap.py` that streamed only a single centroid. The current
> backend implements the full multi-rigid-body observer described here. If you are
> auditing a port, the acceptance check is simple: in a `--mocap-sim` run the
> `robot_state` lines must report `joint_current_valid:true` with non-zero,
> time-varying `joint_current_theta`.

Reference implementation:

- `host/pc_backend/src/main.cpp` — `mocap_poses_to_q_mk8`, `joint_state_from_kinematic_samples`, `MocapClockCalibrator`, `MocapBridge`, `MocapSimulator`.
- `host/pc_backend/src/umarm_fk.{h,cpp}` — forward kinematics from `q`.

## Rigid-Body Layout

The reference robot ("MK8") is a 3-segment continuum/U-joint arm. Six rigid bodies
are tracked, IDs `1000..1005`:

| Rigid body | Role |
| --- | --- |
| `1000` | Base frame. Its full pose (position + orientation) defines the frame all link vectors are expressed in. |
| `1001..1004` | Intermediate link frames along the arm. |
| `1005` | Tip link frame. Its local +Z axis closes the last joint. |

All six must be visible for a valid estimate. `mocap_poses_to_q_mk8` returns false
if the body mask is incomplete or any pose is non-finite, and the cycle is then
reported with `joint_current_valid:false`.

## Pipeline

```
mocap.py (NatNet)  --JSON lines, ~360 Hz-->  MocapBridge
   body_poses = "id:x:y:z:qx:qy:qz:qw|..."        |
                                                  v
                              MocapClockCalibrator (raw NatNet time -> backend clock)
                                                  |
                                                  v
                              mocap_poses_to_q_mk8(poses) -> theta[12]   (per frame)
                                                  |
                                 deque<{timestamp_s, theta[12]}>  (ring buffer, 2048)
                                                  |
       150 Hz cycle asks: joint_state_at(can_sync_time_s)
                                                  v
              joint_state_from_kinematic_samples -> q[12], qdot[12]
```

### 1. Clock alignment (`MocapClockCalibrator`)

NatNet timestamps may use a different epoch and drift slightly versus the host
`steady_clock` that times the 150 Hz loop. The calibrator maintains a rolling
offset (mean over a 5 s window, recomputed when ~1700 samples are present and the
measured frame rate is 320–400 Hz) and produces a corrected timestamp in backend
time. **Never** combine raw NatNet timestamps directly with CAN sync timestamps;
use the calibrated value. It also reports frame rate, frame-drop count, sample
count, and update count for monitoring.

### 2. Pose → joint angles (`mocap_poses_to_q_mk8`)

For each received mocap frame the six poses are reduced to 12 joint angles:

1. Take the base body (`1000`) rotation `R_base`; express all vectors in that frame
   (`normalize_in_base` = normalize of `R_baseᵀ · v`).
2. Build five normalized link vectors from consecutive body centers
   `rv_k = R_baseᵀ · (pos_{k-1} − pos_k)` for `k = 1..5`, and a sixth vector
   `rv_6 = R_baseᵀ · zaxis(R_{1005})` from the tip body orientation.
3. The first U-joint angles come from `angle_pair(rv_1)`.
4. Each subsequent joint is resolved in the frame of the previous link by rotating
   the previous link vector onto +Z (`rotation_from_to(rv_{k}, ẑ)`), applying a
   −45° Z twist on alternating joints (matching the segment construction), then
   `angle_pair` of the transformed next vector.

`angle_pair(v)` returns the two orthogonal U-joint angles:
`θ₁ = atan2(v.z, v.y) − π/2`, `θ₂ = −(atan2(√(v.y²+v.z²), v.x) − π/2)`.

The result is 12 angles in radians: 6 universal joints × 2 DOF, ordered to match
the forward-kinematics construction (segment `s` owns `q[4s .. 4s+3]`).

### 3. Cycle-aligned sampling (`joint_state_from_kinematic_samples`)

The 150 Hz loop requests the joint state at the **CAN sync timestamp** of the
current cycle. The observer:

- finds the two kinematic samples bracketing the target time and **linearly
  interpolates** `theta` (with `wrap_angle` so wrap-around near ±π is handled);
- estimates `theta_dot` by finite difference over the bracketing samples;
- if the target is just past the newest sample, **extrapolates** up to
  `kMaxMocapExtrapolationS = 12 ms`; beyond that the estimate is marked invalid.

Each estimate carries `valid`, `extrapolated`, `source_time_error_ms`, and
`extrapolation_ms` so consumers can gate on freshness.

### 4. Fixed-delay state

Alongside the current-time estimate, the loop also computes a state at
`can_sync_time_s − kFixedDelayStateS` (`kFixedDelayStateS = 8 ms`). This
delay-compensated snapshot (`joint_fixed_delay_*` in `robot_state`) is useful for
delay-aware control/MPC that must reason about actuation/transport latency.

## Forward Kinematics (`umarm_fk`)

`umarm_forward_kinematics(q)` maps the 12 joint angles to U-joint centers and a tip
position. Config signature `original_umarm_3seg_fk_20260522`:

- Base at `(0, 0, 1.2) m`, identity orientation.
- 3 segments, rod lengths `{0.26505, 0.23414, 0.23166} m`, per-segment +45° Z twist.
- Inter-segment offsets `{0,0,0.07249}` and `{0,0,0.07326} m` with −45° Z.
- End-effector rod `0.12987985 m`.

Per segment: `Rx(q[4s]) · Ry(q[4s+1])` at the first U-joint, then rod + twist, then
`Rx(q[4s+2]) · Ry(q[4s+3])` at the second U-joint. The backend runs FK every cycle
and publishes `fk_tip` (and validity/timing) in `robot_state`. If your robot
differs, replace `umarm_fk.cpp` and the observer's pose→angle mapping together;
they share the joint ordering and twist convention.

## Timing budgets

The observer and FK are budgeted inside the 150 Hz cycle and report over-budget
counts in `robot_state` and the run report:

| Stage | Budget |
| --- | ---: |
| Observer (`joint_state_at` ×2) | 0.5 ms |
| Forward kinematics | 0.25 ms |

## Tunable constants (in `main.cpp`)

| Constant | Value | Meaning |
| --- | ---: | --- |
| `kRigidBodyCount` | 6 | Tracked rigid bodies (`1000..1005`). |
| `kJointCount` | 12 | Estimated joint DOF. |
| `kFixedDelayStateS` | 0.008 s | Fixed-delay snapshot offset. |
| `kMaxMocapExtrapolationS` | 0.012 s | Max forward extrapolation past newest mocap sample. |
| Clock window / min samples / rate band | 5 s / 1700 / 320–400 Hz | Mocap→backend clock offset recalibration gate. |

## Porting to a different arm

1. Keep the clock-alignment and cycle-sampling stages (`MocapClockCalibrator`,
   `joint_state_from_kinematic_samples`) — they are robot-independent.
2. Replace `mocap_poses_to_q_mk8` with your kinematic mapping and update
   `kRigidBodyCount` / `kJointCount` (and the matching `umarm_fk`).
3. Re-run the simulated acceptance check and confirm non-zero, valid `q`/`qdot`.
