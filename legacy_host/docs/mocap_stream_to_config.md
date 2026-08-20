# Motion Capture Stream to Robot Configuration

This note documents how the live OptiTrack/NatNet stream is converted into the 12-DOF robot configuration vector used by the controller. It is written as a reconstruction guide: another agent should be able to recreate the same data flow and joint-angle math in another folder without needing to infer hidden conventions from the original code.

## Source Citations

Each source entry gives the original absolute path first, then the workspace-relative citation and function or constant name.

- Full path: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025\mocap_natnet_receiver_routine.py`
  - Relative citations: [mocap_natnet_receiver_routine.py](mocap_natnet_receiver_routine.py#L35) `receive_new_frame`, [mocap_natnet_receiver_routine.py](mocap_natnet_receiver_routine.py#L70) `receive_rigid_body_frame`, [mocap_natnet_receiver_routine.py](mocap_natnet_receiver_routine.py#L193) `mocap_routine`, [mocap_natnet_receiver_routine.py](mocap_natnet_receiver_routine.py#L248) `mocap_routine_main`.
- Full path: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025\kinematics_mp.py`
  - Relative citations: [kinematics_mp.py](kinematics_mp.py#L25) `RZn45`, [kinematics_mp.py](kinematics_mp.py#L671) `mocap_to_config_main`, [kinematics_mp.py](kinematics_mp.py#L981) `get_configuration_from_mocap_mk8`.
- Full path: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025\robot_constants.py`
  - Relative citations: [robot_constants.py](robot_constants.py#L8) `num_rigid_bodies`, [robot_constants.py](robot_constants.py#L9) `num_joints`, [robot_constants.py](robot_constants.py#L405) `ADP_new_frame_data`, [robot_constants.py](robot_constants.py#L406) `ADP_new_frame_compliance`.
- Full path: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025\unified_robot_adapter.py`
  - Relative citation: [unified_robot_adapter.py](unified_robot_adapter.py#L127) `mp.Process(target=mocap_natnet_receiver_routine.mocap_routine_main, ...)`.
- Full path: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025\data_processing_mp.py`
  - Relative citation: [data_processing_mp.py](data_processing_mp.py#L24) `data_processing_mp_main`.
- Full path: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025\mp_compliance.py`
  - Relative citation: [mp_compliance.py](mp_compliance.py#L50) `mp_compliance_main`.

## Runtime Data Flow

The real-hardware path starts a dedicated mocap process from the unified robot adapter. The process target is `mocap_natnet_receiver_routine.mocap_routine_main`, and it receives three shared multiprocessing objects:

- `ADP_RigidBody_list_from_robot`: shared list of rigid-body transforms.
- `ADP_q_from_robot`: shared list of robot joint angles.
- `ADP_new_frame`: shared list of frame flags.

The constants are:

- `rc.num_rigid_bodies = 9`.
- `rc.nlinks = 3` and `rc.num_joints = 4 * rc.nlinks = 12`.
- `rc.ADP_new_frame_data = 0`.
- `rc.ADP_new_frame_compliance = 1`.

`mocap_routine` creates process-global state:

```python
homogeneous_of_rigid_bodies = np.zeros([rc.num_rigid_bodies, 4, 4], dtype=float)
q_current = mp_q
new_frame_flag = ADP_new_frame
```

It configures the NatNet client with default addresses `clientAddress = "192.168.1.120"`, `serverAddress = "192.168.1.100"`, and `use_multicast = True`. Command-line arguments can override those values through `my_parse_args`. The NatNet callbacks are then assigned:

```python
streaming_client.new_frame_listener = receive_new_frame
streaming_client.rigid_body_listener = receive_rigid_body_frame
```

After `streaming_client.run()` succeeds, `mocap_routine` continuously copies the internal latest transforms into the shared list:

```python
for i in range(rc.num_rigid_bodies):
    mp_rigid_body_homo_list[i] = homogeneous_of_rigid_bodies[i]
```

## NatNet Rigid-Body Callback

`receive_rigid_body_frame(new_id, position, rotation)` is called once per rigid body per mocap frame. The code assumes Motive rigid-body IDs are contiguous starting at `1000`:

```python
id_mask = 1000
if new_id in range(id_mask, id_mask + rc.num_rigid_bodies):
    index = new_id - id_mask
```

For accepted IDs, the callback converts the incoming quaternion into a rotation matrix using SciPy:

```python
rotation_matrix = Rotation.from_quat(rotation).as_matrix()
```

Then it writes an SE(3)-style homogeneous matrix into `homogeneous_of_rigid_bodies[index]`:

```python
homogeneous_of_rigid_bodies[index][:3, :3] = rotation_matrix
homogeneous_of_rigid_bodies[index][:3, 3] = position
homogeneous_of_rigid_bodies[index][3, 3] = 1
```

The code only sets the bottom-right element explicitly. Because the array was zero-initialized, the bottom row becomes `[0, 0, 0, 1]` after the first update for that rigid body.

## Per-Frame Callback

`receive_new_frame(data_dict)` is called once per mocap frame. The incoming `data_dict` is not used by the current implementation. Instead, the function consumes the latest `homogeneous_of_rigid_bodies` values accumulated by rigid-body callbacks:

```python
kinematics_mp.mocap_to_config_main(homogeneous_of_rigid_bodies, q_current)
```

After conversion, it marks both downstream consumers as having fresh data:

```python
new_frame_flag[rc.ADP_new_frame_compliance] = 1
new_frame_flag[rc.ADP_new_frame_data] = 1
```

`data_processing_mp_main` watches `ADP_new_frame_data` and clears it after recording a sample. `mp_compliance_main` watches `ADP_new_frame_compliance` and clears it before computing compliance outputs. Those consumers are not part of the conversion math, but the flags explain why the mocap handler writes both indexes each frame.

## Top-Level Conversion Function

`kinematics_mp.mocap_to_config_main(mp_rigid_body_homo_list, mp_q)` is the active conversion entry point. It calls only the mk8 converter:

```python
q_out = get_configuration_from_mocap_mk8(mp_rigid_body_homo_list)
for i in range(rc.num_joints):
    mp_q[i] = float(q_out[i])
```

The output `mp_q` is a length-12 vector in radians, ordered as two angular coordinates for each of six universal-joint pairs:

```text
[u1_theta1, u1_theta2,
 u2_theta1, u2_theta2,
 u3_theta1, u3_theta2,
 u4_theta1, u4_theta2,
 u5_theta1, u5_theta2,
 u6_theta1, u6_theta2]
```

The older `get_configuration_from_mocap_mk7` and duplicated `get_configuration_from_mocap` functions are not called by `mocap_to_config_main` in the current active path.

## Rigid-Body Index Assumptions

`get_configuration_from_mocap_mk8` copies positions for the first eight rigid bodies, but the active math uses indices `0` through `6`, plus the orientation of index `5`. The ninth allocated rigid body, index `8`, is streamed and copied by the handler but is not used by the mk8 conversion.

The code-level assumptions are:

| Index | NatNet ID | Used as |
| --- | --- | --- |
| `0` | `1000` | Robot base pose. Position is `u_1_1`; rotation `rot_base` defines the robot base frame. |
| `1` | `1001` | Position `u_1_2`. |
| `2` | `1002` | Position `u_2_1`. |
| `3` | `1003` | Position `u_2_2`. |
| `4` | `1004` | Position `u_3_1`. |
| `5` | `1005` | Position `u_3_2`; rotation z-axis is used for the sixth joint pair when `rod_mocap_installed = False`. |
| `6` | `1006` | End-effector or rod rigid-body center, used only if `rod_mocap_installed = True`. |
| `7` | `1007` | Copied into `pos_current` but unused by active mk8 math. |
| `8` | `1008` | Allocated and streamed but unused by active mk8 math. |

## Coordinate Frames

The incoming positions and rotations are in the mocap/spatial frame. The conversion uses rigid body `0` as the robot base frame:

```python
rot_base = mp_rigid_body_homo_list[0][0:3, 0:3]
```

Any spatial vector `v_spatial` is converted into the robot frame with:

```python
v_robot = rot_base.T @ v_spatial
v_robot = v_robot / np.linalg.norm(v_robot)
```

The base-frame positive z-axis is:

```python
v_base = np.array([0, 0, 1])
```

For joints after the first pair, SciPy `Rotation.align_vectors(v_base, previous_RV)` is used exactly in that argument order. In this code path, the returned rotation matrix is applied to the next vector so that the next vector is expressed in a frame where the previous link vector has been aligned with +z.

## Hardware 45-Degree Frame Rotation

The hardware alternates universal-joint frame orientation. For joint pairs 2, 4, and 6, the transformed vector is additionally multiplied by `RZn45`, a -45 degree rotation about z:

```python
RZn45 = np.array([
    [np.cos(-np.pi / 4.0), -np.sin(-np.pi / 4.0), 0.0],
    [np.sin(-np.pi / 4.0),  np.cos(-np.pi / 4.0), 0.0],
    [0.0,                  0.0,                  1.0],
], dtype=float)
```

Do not replace this with `RZ45`; the active code applies the negative 45 degree rotation.

## Angle Formula

Every universal-joint pair uses the same vector-to-two-angles formula. Given a unit vector `v = [x, y, z]` in the appropriate local frame:

```python
theta1 = np.atan2(z, y) - np.pi * 0.5
theta2 = -1.0 * (np.atan2((z**2 + y**2)**0.5, x) - np.pi / 2)
```

The current implementation first converts these angles to degrees for intermediate variables, then packs them into `q`, then converts the whole vector back to radians with `np.radians(q)`. A reconstruction can compute in radians directly if it preserves the same formulas and final units.

## mk8 Conversion Algorithm

Use this procedure to recreate `get_configuration_from_mocap_mk8` exactly.

1. Build `pos_current = np.zeros((rc.num_rigid_bodies, 3))`.
2. For `i in range(8)`, set `pos_current[i, :] = mp_rigid_body_homo_list[i][0:3, 3]`.
3. Set `rot_base = mp_rigid_body_homo_list[0][0:3, 0:3]` and `v_base = np.array([0, 0, 1])`.
4. Assign the rigid-body positions directly as universal-joint centers:

```python
u_1_1 = pos_current[0, :]
u_1_2 = pos_current[1, :]
u_2_1 = pos_current[2, :]
u_2_2 = pos_current[3, :]
u_3_1 = pos_current[4, :]
u_3_2 = pos_current[5, :]
ee_rigid_body_center = pos_current[6, :]
```

5. Compute the first relative vector. If its norm is below `1e-10`, return without producing a configuration:

```python
RV1 = u_1_1 - u_1_2
if np.linalg.norm(RV1) < 1e-10:
    return
RV1 = rot_base.T @ RV1 / np.linalg.norm(RV1)
u1_theta1, u1_theta2 = angle_pair(RV1)
```

6. Compute pair 2 from the vector from `u_1_2` to `u_2_1`, expressed relative to `RV1`, then apply `RZn45`:

```python
RV2 = u_1_2 - u_2_1
RV2 = rot_base.T @ RV2 / np.linalg.norm(RV2)
R_trans_to_z = Rotation.align_vectors(v_base, RV1)
RV2_transformed = R_trans_to_z[0].as_matrix() @ RV2
RV2_transformed = RZn45 @ RV2_transformed
u2_theta1, u2_theta2 = angle_pair(RV2_transformed)
```

7. Compute pair 3 from the vector from `u_2_1` to `u_2_2`, expressed relative to the unshifted `RV2`:

```python
RV3 = u_2_1 - u_2_2
RV3 = rot_base.T @ RV3 / np.linalg.norm(RV3)
R_trans_to_z = Rotation.align_vectors(v_base, RV2)
RV3_transformed = R_trans_to_z[0].as_matrix() @ RV3
u3_theta1, u3_theta2 = angle_pair(RV3_transformed)
```

8. Compute pair 4 from the vector from `u_2_2` to `u_3_1`, expressed relative to `RV3`, then apply `RZn45`:

```python
RV4 = u_2_2 - u_3_1
RV4 = rot_base.T @ RV4 / np.linalg.norm(RV4)
R_trans_to_z = Rotation.align_vectors(v_base, RV3)
RV4_transformed = R_trans_to_z[0].as_matrix() @ RV4
RV4_transformed = RZn45 @ RV4_transformed
u4_theta1, u4_theta2 = angle_pair(RV4_transformed)
```

9. Compute pair 5 from the vector from `u_3_1` to `u_3_2`, expressed relative to the unshifted `RV4`:

```python
RV5 = u_3_1 - u_3_2
RV5 = rot_base.T @ RV5 / np.linalg.norm(RV5)
R_trans_to_z = Rotation.align_vectors(v_base, RV4)
RV5_transformed = R_trans_to_z[0].as_matrix() @ RV5
u5_theta1, u5_theta2 = angle_pair(RV5_transformed)
```

10. Compute pair 6. The active code has `rod_mocap_installed = False`, so it uses the z-axis of rigid body `5` rather than a position vector to the rod body:

```python
rod_mocap_installed = False

if rod_mocap_installed:
    RV6 = u_3_2 - ee_rigid_body_center
else:
    RV6 = mp_rigid_body_homo_list[5][0:3, 2]

RV6 = rot_base.T @ RV6 / np.linalg.norm(RV6)
R_trans_to_z = Rotation.align_vectors(v_base, RV5)
RV6_transformed = R_trans_to_z[0].as_matrix() @ RV6
RV6_transformed = RZn45 @ RV6_transformed
u6_theta1, u6_theta2 = angle_pair(RV6_transformed)
```

11. Pack the 12 angles in order and return radians:

```python
q_degrees = np.array([
    u1_theta1, u1_theta2,
    u2_theta1, u2_theta2,
    u3_theta1, u3_theta2,
    u4_theta1, u4_theta2,
    u5_theta1, u5_theta2,
    u6_theta1, u6_theta2,
])
q = np.radians(q_degrees)
return q
```

## Minimal Reconstruction Skeleton

```python
import numpy as np
from scipy.spatial.transform import Rotation

NUM_RIGID_BODIES = 9
NUM_JOINTS = 12
ID_MASK = 1000

RZn45 = np.array([
    [np.cos(-np.pi / 4.0), -np.sin(-np.pi / 4.0), 0.0],
    [np.sin(-np.pi / 4.0),  np.cos(-np.pi / 4.0), 0.0],
    [0.0,                  0.0,                  1.0],
], dtype=float)


def angle_pair_degrees(vector):
    x, y, z = vector
    theta1 = np.atan2(z, y) - np.pi * 0.5
    theta2 = -1.0 * (np.atan2((z**2 + y**2)**0.5, x) - np.pi / 2)
    return theta1 * 180.0 / np.pi, theta2 * 180.0 / np.pi


def normalize_in_base(rot_base, vector):
    return rot_base.T @ vector / np.linalg.norm(vector)


def mocap_se3_to_q_mk8(rigid_body_homo_list):
    pos_current = np.zeros((NUM_RIGID_BODIES, 3), dtype=float)
    for index in range(8):
        pos_current[index, :] = rigid_body_homo_list[index][0:3, 3]

    v_base = np.array([0, 0, 1])
    rot_base = rigid_body_homo_list[0][0:3, 0:3]

    u_1_1 = pos_current[0, :]
    u_1_2 = pos_current[1, :]
    u_2_1 = pos_current[2, :]
    u_2_2 = pos_current[3, :]
    u_3_1 = pos_current[4, :]
    u_3_2 = pos_current[5, :]
    ee_rigid_body_center = pos_current[6, :]

    RV1_raw = u_1_1 - u_1_2
    if np.linalg.norm(RV1_raw) < 1e-10:
        return None

    RV1 = normalize_in_base(rot_base, RV1_raw)
    u1 = angle_pair_degrees(RV1)

    RV2 = normalize_in_base(rot_base, u_1_2 - u_2_1)
    RV2_transformed = Rotation.align_vectors(v_base, RV1)[0].as_matrix() @ RV2
    RV2_transformed = RZn45 @ RV2_transformed
    u2 = angle_pair_degrees(RV2_transformed)

    RV3 = normalize_in_base(rot_base, u_2_1 - u_2_2)
    RV3_transformed = Rotation.align_vectors(v_base, RV2)[0].as_matrix() @ RV3
    u3 = angle_pair_degrees(RV3_transformed)

    RV4 = normalize_in_base(rot_base, u_2_2 - u_3_1)
    RV4_transformed = Rotation.align_vectors(v_base, RV3)[0].as_matrix() @ RV4
    RV4_transformed = RZn45 @ RV4_transformed
    u4 = angle_pair_degrees(RV4_transformed)

    RV5 = normalize_in_base(rot_base, u_3_1 - u_3_2)
    RV5_transformed = Rotation.align_vectors(v_base, RV4)[0].as_matrix() @ RV5
    u5 = angle_pair_degrees(RV5_transformed)

    rod_mocap_installed = False
    if rod_mocap_installed:
        RV6_source = u_3_2 - ee_rigid_body_center
    else:
        RV6_source = rigid_body_homo_list[5][0:3, 2]
    RV6 = normalize_in_base(rot_base, RV6_source)
    RV6_transformed = Rotation.align_vectors(v_base, RV5)[0].as_matrix() @ RV6
    RV6_transformed = RZn45 @ RV6_transformed
    u6 = angle_pair_degrees(RV6_transformed)

    q_degrees = np.array([*u1, *u2, *u3, *u4, *u5, *u6], dtype=float)
    return np.radians(q_degrees)
```

## Behavior and Failure Notes

- The only explicit zero-length guard in mk8 is for `RV1`. If any later vector has near-zero norm, NumPy can produce invalid values and the function may fall into the broad exception handler.
- The mk8 exception handler prints `get_config_mk7:` even though the function is `get_configuration_from_mocap_mk8`. Preserve this only if matching console text matters.
- If `get_configuration_from_mocap_mk8` returns `None`, `mocap_to_config_main` will fail when indexing `q_out`; `receive_new_frame` catches that exception and prints `receive_new_frame:`.
- The sixth joint pair currently depends on rigid body `5` orientation, not rigid body `6` position, because `rod_mocap_installed` is hard-coded to `False`.
- The active output angles are radians. Any controller or logger that reads `ADP_q_from_robot` should treat the values as radians.