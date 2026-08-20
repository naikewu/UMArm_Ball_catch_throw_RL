# Forward Kinematics Implementation Details

System path of current folder: `c:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMArm_mujoco_reconfiguration_2026`

This note documents exactly how the UMArm forward kinematics are computed in this repository. It is written for another agent that needs to reimplement both the Python and C++ versions in another folder. The workflow-oriented SOP remains in `implementation_notes/forward_kinematics_sop.md`; this file is the implementation reference.

Citation convention: code citations use workspace-relative paths and the line where the cited class, function, generated function, or data block starts. Example: `UMARM_kinematics/forward_kinematics/symbolic.py:76` cites the start of `_chain_expressions`.

## Source Of Truth

All kinematic geometry is sourced from `robot_config.py`; generated files are downstream artifacts. The active robot is selected by `ROBOT = robot_for_configuration(active_configuration_name())` at `robot_config.py:775`, so a reimplementation must load the same active configuration before deriving FK. The FK code should not hard-code only the original design unless the caller explicitly wants the original configuration.

The important configuration objects are:

- `Pose` at `robot_config.py:25`: stores a world position in meters and XYZ Euler angles in degrees.
- `EndEffectorConfig` at `robot_config.py:65`: stores whether the end effector exists and its rod length.
- `SegmentConfig` at `robot_config.py:76`: stores each segment rod length, U-joint twist, and other physical fields.
- `TransformConfig` at `robot_config.py:91`: stores inter-segment translation and XYZ Euler angles.
- `RobotConfig` at `robot_config.py:97`: groups base pose, end effector, segments, inter-segment offsets, nominal joint state, and joint limits.

The current original robot geometry used by the checked-in generated C++ contains these values:

- Base pose: position `(0.0, 0.0, 1.2)` m and Euler `(0.0, 0.0, 0.0)` deg in `_ORIGINAL_ROBOT` at `robot_config.py:171`.
- Segment rod lengths: `0.26505`, `0.23414`, and `0.23166` m in the `SEGMENTS` tuple that starts at `robot_config.py:128`.
- Segment U-joint twist: `45.0` deg for all three original segments at `robot_config.py:137`, `robot_config.py:150`, and `robot_config.py:163`.
- Inter-segment offsets: `(0.0, 0.0, 0.07249)` m with Euler `(0.0, 0.0, -45.0)` deg, then `(0.0, 0.0, 0.07326)` m with Euler `(0.0, 0.0, -45.0)` deg, at `robot_config.py:199` through `robot_config.py:201`.
- End-effector rod length: `0.12987985` m at `robot_config.py:189` through `robot_config.py:191`.

The robot has `segment_count() = len(ROBOT.segments)` at `robot_config.py:778`. It has `joint_count() = 4 * segment_count()` at `robot_config.py:782`, which is 12 for the current three-segment robot. The public joint order comes from `joint_names()` at `robot_config.py:862`: for each segment, the order is top U-joint X, top U-joint Y, bottom U-joint X, bottom U-joint Y. For segment `N`, the names are `uj{2N-1}_x`, `uj{2N-1}_y`, `uj{2N}_x`, and `uj{2N}_y`.

Generated C++ embeds `robot_config_signature()` from `robot_config.py:918`. The Python wrapper compares that signature against the active `robot_config.py` at runtime. A reimplementation should include the same stale-library guard if it loads generated artifacts.

## Coordinate And Matrix Conventions

All FK transforms are 4 by 4 homogeneous matrices. The code composes transforms by left-to-right matrix multiplication and applies them to homogeneous column vectors. In Python NumPy, the POE implementation uses `@`; in SymPy, the symbolic implementation uses `*`. The translation component is always the first three entries of the fourth column, `T[:3, 3]`.

The helper transforms are the same in the symbolic generator and the POE model:

- `translate(x_m, y_m, z_m)` builds an identity 4 by 4 transform with `(x_m, y_m, z_m)` in the translation column. See `UMARM_kinematics/forward_kinematics/symbolic.py:25` and `UMARM_kinematics/forward_kinematics/poe.py:22`.
- `rotate_x(theta_rad)`, `rotate_y(theta_rad)`, and `rotate_z(theta_rad)` build standard right-handed homogeneous rotations about local X, Y, and Z. See `UMARM_kinematics/forward_kinematics/symbolic.py:36`, `UMARM_kinematics/forward_kinematics/symbolic.py:43`, `UMARM_kinematics/forward_kinematics/symbolic.py:50`, and the NumPy versions at `UMARM_kinematics/forward_kinematics/poe.py:28`, `UMARM_kinematics/forward_kinematics/poe.py:34`, and `UMARM_kinematics/forward_kinematics/poe.py:40`.
- `euler_xyz(euler_deg)` converts degrees to radians and multiplies `Rx * Ry * Rz`, not ZYX. See `UMARM_kinematics/forward_kinematics/symbolic.py:57` and `UMARM_kinematics/forward_kinematics/poe.py:46`.

The model uses negative local Z translations for rods and the end effector. Inter-segment offset translations from `robot_config.py` are applied as `(tx, ty, -tz)`, which mirrors the MuJoCo body convention used by this project.

## Direct Homogeneous Transform Chain

The direct chain is implemented in `_chain_expressions` at `UMARM_kinematics/forward_kinematics/symbolic.py:76`. This is the mathematical source used to generate the closed-form C++ functions.

Use this exact algorithm to reimplement the direct chain:

```text
q must contain rc.joint_count() values in radians.
The active robot must have end_effector.enabled == True.

base = rc.ROBOT.base_pose
T = translate(*base.position_m) * euler_xyz(base.euler_deg)
ujoint_centers = []

for segment_index, segment in enumerate(rc.ROBOT.segments, start=1):
    if segment_index > 1:
        offset = rc.inter_segment_offset(segment_index - 1)
        tx, ty, tz = offset.translation_m
        T = T * translate(tx, ty, -tz) * euler_xyz(offset.euler_deg)

    # Top universal-joint center for this segment.
    ujoint_centers.append(T[:3, 3])

    q_index = 4 * (segment_index - 1)

    # Top universal joint, segment rod, and fixed twist to the bottom joint.
    T = (
        T
        * rotate_x(q[q_index])
        * rotate_y(q[q_index + 1])
        * translate(0.0, 0.0, -segment.rod_length_m)
        * rotate_z(radians(segment.ujoint_twist_deg))
    )

    # Bottom universal-joint center for this segment.
    ujoint_centers.append(T[:3, 3])

    # Bottom universal joint.
    T = T * rotate_x(q[q_index + 2]) * rotate_y(q[q_index + 3])

tip_transform = T * translate(0.0, 0.0, -rc.ROBOT.end_effector.rod_length_m)
```

There are two important details that are easy to miss:

- The inter-segment translation uses `translate(tx, ty, -tz)`, not `translate(tx, ty, tz)`.
- Segment rods and the end-effector rod extend along negative local Z.

The direct symbolic API splits the returned values into `tip_transform_expression()` at `UMARM_kinematics/forward_kinematics/symbolic.py:107` and `ujoint_center_expressions()` at `UMARM_kinematics/forward_kinematics/symbolic.py:112`.

## Python POE Reimplementation

The readable Python implementation is the Product of Exponentials model in `UMARM_kinematics/forward_kinematics/poe.py`. It should produce the same tip transform and U-joint centers as the generated direct chain.

The screw-axis convention is `S = [v, w]`, where `w` is the unit angular axis and `v = p x w` for a revolute joint passing through world point `p`. This is implemented by `revolute_twist()` at `UMARM_kinematics/forward_kinematics/poe.py:51`. The `twist_hat()` helper at `UMARM_kinematics/forward_kinematics/poe.py:58` converts the 6-vector into a 4 by 4 matrix representation if needed.

The local U-joint axes are fixed constants:

```text
LOCAL_X = [1, 0, 0]
LOCAL_Y = [0, 1, 0]
```

They are defined near `UMARM_kinematics/forward_kinematics/poe.py:12`. At each U-joint frame, the world-space axes are `rotation @ LOCAL_X` and `rotation @ LOCAL_Y`, where `rotation = T[:3, :3]`. Each universal joint contributes two consecutive twists in X-then-Y order.

The skew-symmetric matrix helper is `skew()` at `UMARM_kinematics/forward_kinematics/poe.py:17`. The twist exponential is implemented in `twist_exp()` at `UMARM_kinematics/forward_kinematics/poe.py:66`. For a revolute twist, the rotation is Rodrigues' formula:

```text
R(theta) = I + sin(theta) * [w]x + (1 - cos(theta)) * [w]x * [w]x
```

The translation part is:

```text
p(theta) = (I - R(theta)) * ([w]x * v) + w * dot(w, v) * theta
```

If `norm(w) < 1e-12`, the code treats the twist as prismatic and returns identity rotation with translation `v * theta`. The current UMArm joints are revolute, but the branch is part of the implementation.

The home model is built by `_home_model()` at `UMARM_kinematics/forward_kinematics/poe.py:83`. It walks the same zero-angle geometry as the direct chain, but instead of applying joint rotations it records screw axes and home points:

1. Initialize `T = translate(*base.position_m) @ euler_xyz(base.euler_deg)`.
2. For each segment after the first, apply `T = T @ translate(tx, ty, -tz) @ euler_xyz(offset.euler_deg)`.
3. At the current top U-joint frame, record the origin as a home center and store a copy of the transform. Append two revolute twists through that origin about `rotation @ LOCAL_X` and `rotation @ LOCAL_Y`.
4. Advance to the bottom U-joint home frame with `T = T @ translate(0, 0, -segment.rod_length_m) @ rotate_z(radians(segment.ujoint_twist_deg))`.
5. Record the bottom center and transform. Append two more twists through the bottom origin about its local X and Y axes.
6. After all segments, compute the home tip transform as `T @ translate(0, 0, -rc.ROBOT.end_effector.rod_length_m)`.

The generic POE evaluator is `forward_kinematics()` at `UMARM_kinematics/forward_kinematics/poe.py:137`:

```text
T(q) = exp(S1 * q1) * exp(S2 * q2) * ... * exp(Sn * qn) * M
```

It validates `space_twists.shape == (len(q), 6)` and `home_transform.shape == (4, 4)`, accumulates from identity, multiplies each twist exponential in order, then post-multiplies the home transform.

The public POE helpers are:

- `tip_transform(q_rad)` at `UMARM_kinematics/forward_kinematics/poe.py:152`: validates the joint vector and evaluates the full POE tip transform.
- `tip_position(q_rad)` at `UMARM_kinematics/forward_kinematics/poe.py:159`: returns `tip_transform(q_rad)[:3, 3]`.
- `ujoint_centers(q_rad)` at `UMARM_kinematics/forward_kinematics/poe.py:163`: transforms each home U-joint center by the prefix product of all earlier joint exponentials. Center index `c` uses twists with indices `< 2 * c`.
- `ujoint_transforms(q_rad)` at `UMARM_kinematics/forward_kinematics/poe.py:181`: transforms each home U-joint frame by the same prefix logic used for centers.

## Python Public API

The package API in `UMARM_kinematics/forward_kinematics/__init__.py` exposes both compiled direct FK and POE helpers.

The loader `_load_library()` at `UMARM_kinematics/forward_kinematics/__init__.py:65` searches platform-specific build locations, loads the generated shared library with `ctypes`, declares C function signatures, checks the compiled joint count, checks the U-joint count, and verifies `umarm_fk_robot_config_signature()` against `rc.robot_config_signature()`. If a library was generated for a different active robot configuration, loading fails instead of silently returning stale geometry.

All public functions call `_joint_array()` at `UMARM_kinematics/forward_kinematics/__init__.py:124`, which converts the input to contiguous `float64` and requires shape `(rc.joint_count(),)`.

The compiled direct FK wrappers are:

- `tip_transform(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:131`: allocates 16 doubles, calls `umarm_fk_tip_transform`, and reshapes the row-major C array into `(4, 4)`.
- `tip_position(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:141`: allocates 3 doubles and calls `umarm_fk_tip_position`.
- `ujoint_centers(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:151`: allocates `3 * ujoint_count()` doubles, calls `umarm_fk_ujoint_centers`, and reshapes into `(ujoint_count(), 3)`.

The Python POE and compiled POE helpers are:

- `tip_transform_poe(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:173`: calls the readable Python POE model.
- `tip_position_poe(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:177`: returns the Python POE tip position.
- `tip_position_poe_cpp(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:181`: calls the generated C++ POE position helper.
- `ujoint_centers_poe(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:191`: calls the Python POE center helper.
- `ujoint_transforms(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:195`: calls the Python POE frame helper.
- `ujoint_centers_poe_cpp(q_rad)` at `UMARM_kinematics/forward_kinematics/__init__.py:199`: calls the generated C++ POE center helper.

When reimplementing the Python API, keep this separation: `tip_transform`, `tip_position`, and `ujoint_centers` are the compiled direct symbolic path; names ending in `_poe` are the readable Python POE path; names ending in `_poe_cpp` are generated C++ POE point-only helpers.

## C++ Generation

The generated C++ library is written by `UMARM_kinematics.forward_kinematics.symbolic`. The generator writes the header, source, and CMake file in `write_cpp_library()` at `UMARM_kinematics/forward_kinematics/symbolic.py:432`; `main()` at `UMARM_kinematics/forward_kinematics/symbolic.py:440` calls it.

The generator creates three files:

- `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h`
- `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc`
- `UMARM_kinematics/forward_kinematics/cpp/CMakeLists.txt`

The generated C ABI is declared in `umarm_fk.h`. The joint-count and U-joint-count functions start at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:14`. The direct and POE center functions are declared at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:17`. The tip transform and point functions are declared at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:19` through `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:21`.

The source generator is `_source_text()` at `UMARM_kinematics/forward_kinematics/symbolic.py:326`. Its direct symbolic path does this:

1. Create symbolic joint variables using `joint_symbols()`.
2. Compute `transform = tip_transform_expression(q_rad)` from `_chain_expressions`.
3. Flatten the 4 by 4 transform row-major with `[transform[row, col] for row in range(4) for col in range(4)]`.
4. Compute `ujoint_centers = ujoint_center_expressions(q_rad)` and flatten centers as `x, y, z` triples.
5. Emit C++ `const double q0 = q_rad[0]`, etc., via `_append_q_inputs()`.
6. Emit common-subexpression-eliminated C++ assignments with `_append_cse_outputs()` at `UMARM_kinematics/forward_kinematics/symbolic.py:167`.

The direct generated C++ functions are not hand-written matrix multiplication. They are closed-form scalar expressions from the symbolic chain:

- `umarm_fk_ujoint_centers` starts at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:165` and writes 18 doubles for the current six U-joint centers.
- `umarm_fk_tip_transform` starts at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:340` and writes 16 doubles in row-major order.
- `umarm_fk_tip_position` starts at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:535`, calls `umarm_fk_tip_transform`, and copies `transform16[3]`, `transform16[7]`, and `transform16[11]`.

Regeneration is part of the system update script. `scripts/update_system.py:250` invokes `python -m UMARM_kinematics.forward_kinematics.symbolic`, and the surrounding logic configures and builds the FK CMake project when FK regeneration is not skipped.

## Generated C++ POE Helpers

The same generated source also embeds a faster point-only POE implementation. `_append_poe_helpers()` at `UMARM_kinematics/forward_kinematics/symbolic.py:175` asks the Python POE model for space twists, home tip transform, and home U-joint centers, then writes them as C++ constants.

The generated constants in the current source are:

- `kJointCount` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:8`.
- `kPoeSpaceTwists` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:12`, with shape `[kJointCount][6]` and twist order matching `q_rad`.
- `kPoeHomeTipPosition` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:27`, which stores only the home tip point.
- `kPoeHomeUJointCenters` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:31`, which stores all home U-joint centers.

The C++ POE helper functions are direct translations of the Python POE math using row-major arrays:

- `identity_transform()` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:40`: zeroes a 16-double array and sets diagonal entries `0`, `5`, `10`, and `15` to `1.0`.
- `multiply_transform()` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:50`: multiplies two row-major 4 by 4 transforms into scratch storage, then copies scratch to the result. Scratch allows in-place calls such as `multiply_transform(prefix, exponent, prefix)`.
- `transform_point()` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:66`: applies the row-major homogeneous transform to a 3D point with implicit homogeneous coordinate `1`.
- `twist_exp()` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:72`: implements the same Rodrigues and translation formulas as Python `twist_exp()`.
- `apply_next_poe_twist()` at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:145`: builds `exp(S_i * q_i)` and right-multiplies it into the prefix transform.

The exported C++ POE functions are point-only helpers:

- `umarm_fk_ujoint_centers_poe` starts at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:327`. It initializes `prefix = I`, then for each center index applies twist exponentials while `next_twist < 2 * center_index`. It transforms `kPoeHomeUJointCenters[center_index]` by the current prefix and writes the result into `centers18 + 3 * center_index`.
- `umarm_fk_tip_position_poe` starts at `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:543`. It applies all `kJointCount` twist exponentials to the prefix and transforms `kPoeHomeTipPosition`.

The current C++ API does not export a full POE tip transform. Full tip orientation through POE is available in Python via `tip_transform_poe`; the generated C++ POE path exports tip position and U-joint centers because those are the fast point queries used by visualization and control code.

## Reimplementation Checklist

Use this checklist when rebuilding the FK system elsewhere:

1. Implement or port the active robot configuration loader from `robot_config.py`, including the same segment fields, base pose, inter-segment offsets, end-effector fields, and `joint_count = 4 * segment_count`.
2. Preserve joint order from `joint_names()`: top X, top Y, bottom X, bottom Y for each segment.
3. Implement homogeneous `translate`, `rotate_x`, `rotate_y`, `rotate_z`, and `euler_xyz = Rx * Ry * Rz`.
4. Implement the direct chain exactly as described above, including `translate(tx, ty, -tz)`, negative-Z segment rods, negative-Z end-effector rod, and `Rz(segment.ujoint_twist_deg)` between each segment's top and bottom U-joint frames.
5. Implement the POE home-model walk using the zero-angle geometry. At every U-joint frame, derive two screw axes from world-space local X and local Y.
6. Implement twist exponentials with `S = [v, w]`, `v = p x w`, Rodrigues rotation, and the matching POE translation formula.
7. Implement public Python wrappers that validate the joint vector length and, if using generated compiled code, validate a robot configuration signature.
8. For C++, either generate closed-form scalar functions from symbolic expressions or hand-write a matrix-chain evaluator. If compatibility with this repository matters, keep the exported C ABI names and row-major array layout from `umarm_fk.h`.
9. If generating C++ POE helpers, embed space twists and home points derived from the same active robot configuration, then use row-major `identity_transform`, `multiply_transform`, `transform_point`, and `twist_exp` helpers.
10. Validate against MuJoCo sites and cross-check direct compiled FK, Python POE FK, and compiled POE FK.

## Validation Surface

The focused unit test suite is `tests/test_forward_kinematics.py`. It validates direct compiled FK, Python POE, and compiled POE against MuJoCo for representative direct joint states in `assert_fk_matches_mujoco()` at `tests/test_forward_kinematics.py:31`. It validates U-joint centers against MuJoCo in `assert_fk_centers_match_mujoco()` at `tests/test_forward_kinematics.py:53`. It cross-checks Python POE against compiled direct FK in `test_poe_matches_compiled_fk()` at `tests/test_forward_kinematics.py:84` and compiled POE against Python POE in `test_compiled_poe_matches_python_poe()` at `tests/test_forward_kinematics.py:109`.

The pressure-actuated validation script is `experiments/validate_kinematics.py`. It computes a compiled FK tip transform inside `tip_rotation_error()` at `experiments/validate_kinematics.py:40`, and its main comparison loop starts in `validate()` at `experiments/validate_kinematics.py:48`.

Run the focused validation from the workspace root with the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_forward_kinematics
```

If geometry or generated code changes, regenerate and rebuild first:

```powershell
.\.venv\Scripts\python.exe scripts\reconfiguration.py update --config <name> -- --skip-tests
.\.venv\Scripts\python.exe -m unittest tests.test_forward_kinematics
```

For heavier MuJoCo-plus-plugin validation, run:

```powershell
.\.venv\Scripts\python.exe -m experiments.validate_kinematics
```

## Citation Table

| Reference | Meaning |
| --- | --- |
| `robot_config.py:25` | `Pose` geometry pose type. |
| `robot_config.py:65` | `EndEffectorConfig` end-effector fields. |
| `robot_config.py:76` | `SegmentConfig` segment geometry fields. |
| `robot_config.py:91` | `TransformConfig` inter-segment transform fields. |
| `robot_config.py:97` | `RobotConfig` top-level robot geometry container. |
| `robot_config.py:128` | Original `SEGMENTS` tuple. |
| `robot_config.py:171` | Original robot base pose block. |
| `robot_config.py:189` | Original robot end-effector block. |
| `robot_config.py:775` | Active `ROBOT` selection. |
| `robot_config.py:778` | `segment_count()`. |
| `robot_config.py:782` | `joint_count()`. |
| `robot_config.py:796` | `inter_segment_offset()`. |
| `robot_config.py:862` | `joint_names()`. |
| `robot_config.py:918` | `robot_config_signature()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:25` | Symbolic `translate()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:36` | Symbolic `rotate_x()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:43` | Symbolic `rotate_y()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:50` | Symbolic `rotate_z()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:57` | Symbolic `euler_xyz()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:76` | Direct transform chain `_chain_expressions()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:107` | `tip_transform_expression()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:112` | `ujoint_center_expressions()`. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:167` | C++ common-subexpression output helper. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:175` | C++ POE helper generation. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:326` | Generated source text assembly. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:432` | C++ library file writer. |
| `UMARM_kinematics/forward_kinematics/symbolic.py:440` | Generator entry point. |
| `UMARM_kinematics/forward_kinematics/poe.py:22` | NumPy `translate()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:28` | NumPy `rotate_x()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:34` | NumPy `rotate_y()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:40` | NumPy `rotate_z()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:46` | NumPy `euler_xyz()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:51` | `revolute_twist()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:58` | `twist_hat()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:17` | Skew-symmetric matrix helper. |
| `UMARM_kinematics/forward_kinematics/poe.py:66` | `twist_exp()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:83` | POE home-model derivation. |
| `UMARM_kinematics/forward_kinematics/poe.py:137` | POE product evaluator. |
| `UMARM_kinematics/forward_kinematics/poe.py:152` | Python POE `tip_transform()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:159` | Python POE `tip_position()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:163` | Python POE `ujoint_centers()`. |
| `UMARM_kinematics/forward_kinematics/poe.py:181` | Python POE `ujoint_transforms()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:65` | Shared-library loader and signature checks. |
| `UMARM_kinematics/forward_kinematics/__init__.py:124` | Joint-vector validation helper. |
| `UMARM_kinematics/forward_kinematics/__init__.py:131` | Compiled direct `tip_transform()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:141` | Compiled direct `tip_position()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:151` | Compiled direct `ujoint_centers()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:173` | Public Python POE `tip_transform_poe()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:177` | Public Python POE `tip_position_poe()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:181` | Public compiled POE `tip_position_poe_cpp()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:191` | Public Python POE `ujoint_centers_poe()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:195` | Public POE `ujoint_transforms()`. |
| `UMARM_kinematics/forward_kinematics/__init__.py:199` | Public compiled POE `ujoint_centers_poe_cpp()`. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:14` | C ABI count and signature declarations. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:17` | C ABI U-joint center declarations. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.h:19` | C ABI tip transform and position declarations. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:8` | Generated C++ joint-count constant. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:12` | Generated C++ POE space twists. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:27` | Generated C++ POE home tip point. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:31` | Generated C++ POE home U-joint centers. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:40` | C++ row-major identity helper. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:50` | C++ row-major transform multiply helper. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:66` | C++ row-major point transform helper. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:72` | C++ POE twist exponential helper. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:145` | C++ prefix twist application helper. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:165` | Generated direct symbolic U-joint centers. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:327` | Generated C++ POE U-joint centers. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:340` | Generated direct symbolic tip transform. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:535` | Generated direct symbolic tip position wrapper. |
| `UMARM_kinematics/forward_kinematics/cpp/umarm_fk.cc:543` | Generated C++ POE tip position. |
| `scripts/update_system.py:250` | FK generator invocation in the update workflow. |
| `tests/test_forward_kinematics.py:31` | Tip FK validation against MuJoCo. |
| `tests/test_forward_kinematics.py:53` | U-joint center validation against MuJoCo. |
| `tests/test_forward_kinematics.py:84` | Python POE versus compiled direct FK cross-check. |
| `tests/test_forward_kinematics.py:109` | Compiled POE versus Python POE cross-check. |
| `experiments/validate_kinematics.py:40` | Rotation comparison using compiled FK transform. |
| `experiments/validate_kinematics.py:48` | Pressure-actuated validation loop. |
