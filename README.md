# UMArm Koopman compliance control — CAN-arm workspace

The working folder for the large ceiling-hung UMArm: a 24-actuator pneumatic arm on a
1 Mbit/s CAN bus, tracked by an OptiTrack/Motive volume it shares with the smaller
RS485 arm and a Kinova Gen3. This workspace holds everything needed to run the arm —
both board firmwares, the controller and flash GUIs, the mocap-to-joint-angle
pipeline, a display-only multi-arm MuJoCo room viewer, the Kinova stack, and the
skeleton of the arm's future digital twin. Firmware *development* stays in the source
repos; this folder is for operating the robot.

## The hardware, in one table

| thing | identity |
|---|---|
| CAN bus | CANable 2.0 slcan dongle, VID:PID 16D0:117E (COM58 as of 2026-08-20), 1 Mbit/s |
| top 8 actuators | TLE92464/DVP proportional boards, CAN IDs `0x101–0x108`, variant byte 2 |
| lower 16 actuators | legacy 7 mm boards (ESP32-S3 + MCP25625), IDs `0x109–0x118`, variant 0, fw 0.2.1 |
| mocap | Motive at `192.168.1.100`, bind the client to `192.168.1.120` (the I226-V NIC), multicast. CAN arm = rigid bodies 2000–2005; Kinova end effector = 1008 |
| Kinova Gen3 | `192.168.1.10` (web app there clears faults; a pose move silently no-ops in a fault state) |

`bench_env.py` is the single source of these facts — every entry point resolves ports
and addresses through it, so a re-enumerated COM port is a one-place fix.

## Quick start

```powershell
# the workspace interpreter (3.13 + mujoco + python-can); never a bare `python`
$PY = ".\.venv\Scripts\python.exe"

& $PY canarm_control_gui.py           # controller GUI + mocap strip + room viewer
& $PY TLE_PCB\VEMA_TLE_flash.py       # USB flash / CAN broadcast OTA / set board ID
& $PY canarm_control_gui.py --self-test   # opens no port; CI-style check
& $PY viz\self_check.py               # offline visualizer verification
& $PY -m pytest UMArm_MOCAP UMArm_KINEMATICS UMArm_KINOVA -q   # offline suites
```

The Kinova half runs on its own quarantined interpreter (`.venv_kinova`, protobuf
3.5.1 — rebuild any time with `& $PY UMArm_KINOVA\setup_env.py`); talk to the arm
through `UMArm_KINOVA\bridge_client.py`, never by importing `kortex_api` directly.

## Layout

| path | what |
|---|---|
| `firmware/tle`, `firmware/legacy` | the two ESP-IDF projects (both compile here; `MINIMAL_BUILD ON` is mandatory on this machine) |
| `firmware/images/` | prebuilt flashable images with provenance — `legacy_7mm` is byte-identical to what runs on the lower 16 boards |
| `TLE_PCB/` | tlelib (protocol, slcan link, 150 Hz backend, broadcast OTA) + the two GUIs + `tools/tle_bench.py` + the protocol report. Read `PORTING.md` and `README.md` there |
| `legacy_host/` | the legacy portable bundle: OTA/diag/calibration tools, the C++ 150 Hz `pc_backend` (builds and self-tests here), the canonical `docs/can_protocol.md`, `calibration.json` |
| `UMArm_MOCAP/` | NatNet receiver (per-instance rigid-body base; `CanArmMocap` = 2000–2005), marker→q math, marker locks, `sim_stream.py` for camera-free development |
| `UMArm_KINEMATICS/` | product-of-exponentials FK; `canarm_params.py` is this arm's parameter table |
| `UMArm_KINOVA/` | the Gen3 stack: leashed driver, stdio-JSON bridge, hand-eye calibration results, `kinova_scene.py` (MJCF) |
| `viz/` | the display-only room viewer (`mj_forward` only, never physics; closing the window stops nothing) |
| `digital_twin/` | documented skeleton — collect real data first (`README.md` there has the plan), then fit; RS485 actuator fits do NOT transfer |
| `hw_tests/` | hardware-in-the-loop scripts and dated acceptance reports |
| `repo_doc/` | directory map + change log — keep both current |

## The four rules that keep the arm alive

1. **OTA cross-flash guard.** A TLE image broadcast to the 16 legacy boards would be
   accepted and brick the lower arm (USB recovery only). The only guard is the
   image-project-name check in the flash GUI / `tlelib/ota.py` — never bypass it,
   never "optimise" `BATCH_FRAMES=1`, and never drop the `finally` that closes OTA
   sessions (the protocol has no abort).
2. **The enable bit is a level.** The backend addresses every known board every
   cycle; de-selecting a board must clear its enable, not drop it from the table.
   On host stall the top 8 boards release after 500 ms; the lower 16 hold forever.
3. **`CMD_SET_ID` is the only big-endian field in the protocol** (its native 0x400
   twin is little-endian). A wrong byte order sends a board to an ID no scan finds.
4. **Bind the mocap client IP explicitly** (`192.168.1.120`) — three interfaces
   claim the multicast route and a wrong bind receives nothing, silently. In a
   control loop check `q_stale`, not `stale`.

## Hardware validation

Dated acceptance reports live in `hw_tests/` (CAN bring-up, legacy-board OTA, live
mocap, Kinova, integrated GUI). See `repo_doc/repo_log.md` for the change history.
