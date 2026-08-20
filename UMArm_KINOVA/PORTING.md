# Porting `UMArm_KINOVA` out of `RS485_VEMA`

Source: `C:\RUNZE_SRC\RS485_VEMA\UMArm_KINOVA\` — the **working tree**, not the
git remote. `bridge_mocap.py` is untracked there and `arm_bridge.py` /
`kinova_arm.py` are modified, so a `git archive` or a fresh clone would have
dropped exactly the newest work. Ported 2026-08-20.

The Gen3 is the same physical robot, on the same flat `192.168.1.0/24`, with the
same credentials and the same fitted `Y = T_tool_rb`. Nothing about it changes
because a second arm arrived, so the port is deliberately close to a copy: the
list below is the whole of what differs.

---

## 1. What changed

### The venv path, which needed no edit

`bridge_client.py:39-47` and `setup_env.py:40-45` both compute `REPO` as
`dirname(dirname(__file__))` and hang `.venv_kinova` off it. Dropping the package
in beside `UMArm_MOCAP/` therefore landed the environment at the workspace root
with no source change, and `bridge_client.VENV_PYTHON` now reads

```
C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\.venv_kinova\Scripts\python.exe
```

which is what `venv_python()` resolves and what `setup_env` built. That
derivation is a property worth not losing: a hard-coded interpreter path is the
one thing that would have made this package non-relocatable.

The layout the derivation depends on is `UMArm_KINOVA/` and `UMArm_MOCAP/` as
siblings under one root — `setup_env.check()`'s probe (`setup_env.py:147`)
imports `UMArm_MOCAP.mocap_rx`, so the venv self-test fails if the mocap package
moves.

### `pad_frames.py` and `check_pad_markers.py` did not travel

Both are about the printed collision pad rather than the Gen3 controller, and
both reach into `UMArm_COLLAB` (`pad_frames.py:49` wants `MT.PAD_FIX_XYZ_MM`,
`MT.marker_points_pad_mm` and three more). `test_pad_frames.py` went with them.

Two places referred to them, and both are fixed rather than stubbed:

* `sensor_frame.load_calibration` (`sensor_frame.py:51-66`) took its default
  `analysis.json` from `check_pad_markers.latest_analysis`. It now takes it from
  `bridge_mocap.latest_analysis_path`, which is the **same nine-line `listdir`**
  — upstream kept two copies precisely because the bridge's mocap half may not
  import `kortex_api` — and imports nothing from the Kortex SDK.
* `test_kinova_offline.test_the_bridges_analysis_walk_agrees_with_check_pad_markers`
  existed only to hold those two copies together. With one copy left it is
  renamed `..._picks_the_newest_campaign` and asserts the property directly
  against `results/`, which is what the duplication was guarding anyway.

Nothing else in the ported set imports either module; the remaining mentions are
prose in `README.md` and `__init__.py`, both annotated.

### `kinova_scene.py` is new, and is an extraction

`UMArm_COLLAB/collab_scene.py` builds one bench scene holding a UMArm *and* a
Gen3 by merging into `UMArm_SIM.mjcf_generator`'s XML. This workspace has no
UMArm twin yet and wants the Gen3 on its own, so the Gen3-only half was lifted
into `kinova_scene.py`: `_add_kinova_assets` (`:660`), `_add_kinova_defaults`
(`:667`), `_add_kinova` (`:695`), `_add_kinova_actuators` (`:889`), plus
`KINOVA_CHAIN` (`:362`), `KINOVA_BASE_INERTIAL` (`:389`), `KINOVA_MESHES`
(`:397`), `STAND_HALF` (`:82`), `KINOVA_GRAVCOMP` and the gain / armature /
damping block (`:147-166`) with its measurement narrative intact. The four
helpers it needed from `mount_transforms` — `MM` (`:156`),
`KINOVA_TOOL_FLANGE_Z_M` (`:612`), `rpy_to_quat` (`:708`), `quat_to_mat`
(`:727`) — are inlined verbatim under a provenance comment, so the module's only
intra-workspace import is `kinova_kinematics`.

Left behind, all bench furniture: the collision pad, the joint covers, the
strike targets, the `<pair>` list, the force/torque sensors and the payload
geom.

Two entry points, and they are the same code: `build_kinova_xml()` returns a
complete standalone MJCF string, and `attach_kinova(root, ...)` grafts the robot
into an `ElementTree` a multi-arm room scene already owns, creating the
`asset` / `default` / `worldbody` / `contact` / `actuator` sections only when the
host has none.

**One line was recovered rather than copied.** `collab_scene:941` opens its
`<contact>` block with `<exclude body1="kinova_base" body2="shoulder_link"/>`
before any bench-specific `<pair>`, with no comment saying why. Measured here:
the two menagerie collision meshes interpenetrate by **12.0 mm** at the
`joint_1` origin in every pose, MuJoCo's parent-child filter does not remove the
pair because `kinova_base` is welded to the mocap mount rather than jointed to
it, and without the exclusion `mj_forward` reports four contacts at rest and
`kinova_kinematics.solve_pose_ik` reads them as self-collision and refuses every
target it is given. The exclusion belongs to the robot, so it travelled;
`self_test` now asserts `ncon == 0` so losing it again is loud.

`kinova_kinematics.py` was copied whole from `UMArm_COLLAB` and needed no edit —
it has no `sys.path` bootstrap and imports only `mujoco` and `numpy`. Its module
default `TOOL_SITE = "pad_center"` is the bench's pad, which does not exist here;
every function takes `site=`, so pass `kinova_scene.TOOL_SITE` (`"kinova_tool"`).
`KINOVA_ROOT_BODY = "kinova_base"` already matches.

The eight Gen3 STLs and their BSD-3 `LICENSE` were copied from
`UMArm_COLLAB/assets/kinova_gen3/` (4.2 MB, Kinova, via mujoco_menagerie) to
`assets/kinova_gen3/`, and `_mesh_file` still emits absolute forward-slashed
paths so the working directory never matters.

### Everything else is byte-identical

`kinova_arm.py`, `arm_bridge.py`, `bridge_client.py`, `bridge_mocap.py`,
`kinova_mocap.py`, `mocap_calibration.py`, `calibrate_mocap.py`, `setup_env.py`,
`vendor/` (4 files), `wheels/kortex_api-2.6.0.post3-py3-none-any.whl`,
`results/mocap_calib_20260819_{213247,215949,222953}/`, `test_arm_bridge.py`,
`test_mocap_calibration.py`. All three campaigns came across, not just the
newest: `bridge_mocap.latest_analysis_path` only ever reads the last one, but the
0.025 mm / 0.015 deg agreement between the two 2026-08-19 campaigns is the
evidence that `Y` is real, and that evidence lives in having both.

---

## 2. The protobuf patch, in three lines

`kortex_api 2.6.0.post3` is not on PyPI and hard-pins `protobuf==3.5.1`, a 2017
release that reads `collections.MutableMapping` — a name Python 3.10 moved to
`collections.abc` — so the import dies before any arm is contacted.
`setup_env.patch_protobuf_abc` (`setup_env.py:75-97`) rewrites
`collections.<ABC>` to `collections.abc.<ABC>` in exactly two files,
`google/protobuf/internal/{containers,well_known_types}.py`, idempotently, and
**no wheel supplies it** — a `.venv_kinova` rebuilt with plain `pip` will not
import.

The `--no-deps` on the wheel install (`setup_env.py:128`) is load-bearing for the
same reason: pip has been seen to re-resolve the protobuf pin upward, and an
environment that quietly holds protobuf 4 imports fine and then fails inside the
generated stubs.

That is why the environment is a quarantine and not a convenience. Nothing else
in this workspace may see protobuf 3.5.1, which is exactly why `arm_bridge.py`
speaks one line of JSON per message over stdio and `bridge_client.py` imports
nothing from `kortex_api`.

---

## 3. The standing caveat

**None of the streamed-twist / held-jog path has ever run against a real arm.**
A Kortex twist has no end of its own; the only thing that stops it is
`arm_bridge`'s dead-man (`JOG_DEADMAN_S = 0.35`, `arm_bridge.py:137`), and a lost
release coasts 27.8 / 13.9 / 4.9 mm at fast / normal / slow. `twist_checked`
(`kinova_arm.py:454`) is guarded by *polling* the live pose rather than by
pre-checking a target, because a velocity has no target, which leaves `v·dt`
unchecked between calls — 1.5 mm at 40 Hz and 0.06 m/s.

The eight things the first hardware session must check are listed at

```
C:\RUNZE_SRC\RS485_VEMA\docs\reports\real_arm_jog_and_mocap_2026-08-20.md
```

That report did not travel into this workspace and should be read there. The
code being green under `test_arm_bridge.py` and `test_kinova_offline.py` says the
protocol and the refusals behave; it says nothing about the metal.

Two smaller riders that did travel, both measured and both still true: the
NatNet SDK's threads are **not daemons**, so a receiver started and never stopped
holds the process past a timeout (`EXIT=124` was seen); and the SDK prints to
stdout from its own threads at times of its choosing, which is why `_StdoutTee`
(`bridge_mocap.py:91`) covers the receiver's whole life rather than only
`start()` — wrapping only `start()` leaked a line onto the JSON channel in one
run of two.

---

## 4. How this port was verified, offline

No COM port was opened, no NatNet socket was created, and nothing was sent to
`192.168.1.10`.

| check | interpreter | result |
|---|---|---|
| `setup_env.py` build + `check()` | base 3.13.13 | venv built, protobuf ABC patch rewrote 2 files, probe imported `kortex_api` / `KinovaArm` / `MocapRx` / the speed constraint |
| `setup_env.py --check` | base 3.13.13 | `environment OK` |
| `pytest test_arm_bridge.py test_mocap_calibration.py -q` | base 3.13.13 | 71 passed |
| `pytest test_kinova_offline.py -q` | `.venv_kinova` 3.13.13, run under a guard that makes `socket.connect/bind/send*` raise | 102 passed |
| `pytest test_kinova_offline.py -q` | base 3.13.13 | 1 skipped, as the module's `importorskip` intends |
| `kinova_scene.py --self-test` | base 3.13.13 | `nq=7 nv=7 nu=7 nbody=10 nmesh=8 ncon=0`, `mj_forward` clean |
| `attach_kinova` into a foreign host root | base 3.13.13 | host body survives, mount is a mocap body, writing `mocap_pos` moves the tool |
| extracted Gen3 vs `collab_scene.build_scene_xml` | base 3.13.13 | 9 bodies, `max|dpos| = 0`, `max|dquat| = 0`, `max|dmass| = 0`; all 7 joints and actuators identical in range, armature, damping, gain, bias, forcerange, ctrlrange; `fk_in_base` agrees to 0 |

The last row is the one that matters for the extraction: the Gen3 this workspace
builds is the same robot the bench builds, bit for bit, and the difference
between the two scenes is entirely what was left out.

`kinova_kinematics.solve_pose_ik` was additionally exercised against the new
scene at `HOME_Q` with a 37 mm target offset, and converges to 0.12 mm with
collision checking on.

One result that looks like a failure and is not: `import UMArm_KINOVA.kinova_scene`
raises `ModuleNotFoundError: mujoco` under `.venv_kinova`. That is the quarantine
holding. `kinova_scene` and `kinova_kinematics` are ordinary-interpreter modules
and MuJoCo must never be installed beside protobuf 3.5.1; the seam between the
two halves is `arm_bridge`'s JSON-over-stdio, not a shared import.

Reproduce the whole sweep with, from the workspace root:

```
set PY=C:\Users\zuorunze\AppData\Local\Programs\Python\Python313\python.exe
%PY% UMArm_KINOVA\setup_env.py --check
%PY% -m pytest UMArm_KINOVA\test_arm_bridge.py UMArm_KINOVA\test_mocap_calibration.py -q
.venv_kinova\Scripts\python.exe -m pytest UMArm_KINOVA\test_kinova_offline.py -q
%PY% UMArm_KINOVA\kinova_scene.py --self-test
```
