# Where everything is — UMArm CAN-arm workspace

One line per place. Check here first; fix here in the same change that makes it wrong.

The robot: a ceiling-hung 24-actuator pneumatic UMArm on a 1 Mbit/s CAN bus reached
through a CANable 2.0 slcan dongle (VID:PID 16D0:117E, currently COM58). Top 8
actuators = TLE92464/DVP boards at `0x101–0x108`; lower 16 = legacy 7 mm boards at
`0x109–0x118`. Mocap rigid bodies 2000–2005; RS485 sister arm and Kinova Gen3
(body 1008) share the Motive volume at 192.168.1.100.

## Root

| file | what it does |
|---|---|
| `bench_env.py` | the single source of ports/IPs/interpreters: CAN dongle resolved by USB identity, mocap/Kinova IPs, IDF env incantation |
| `canarm_control_gui.py` | **the operator GUI**: TLE controller (all 24 boards) + mocap status strip + spawned MuJoCo room viewer. `--self-test` opens no port |
| `requirements.txt` | workspace deps; base Python 3.13 already has all but `python-can` |
| `.venv/` | workspace venv (3.13 + system site-packages + python-can). `.venv_kinova/` is the protobuf-3.5.1 quarantine — rebuild via `UMArm_KINOVA/setup_env.py` |

## Firmware

| path | what is there |
|---|---|
| `firmware/tle/` | TLE board firmware (ESP-IDF project `VEMA_MAX22200`, esp32s3). MINIMAL_BUILD is mandatory on this machine |
| `firmware/legacy/` | legacy 16-board firmware (`Valve_not_embedded_XL` v0.2.1) incl. the load-bearing `sdkconfig` that upstream gitignores |
| `firmware/images/` | prebuilt flashable images with provenance README: `legacy_7mm` (byte-identical to what runs on 0x109–0x118), `legacy_dt`, `tle` |

## Host tooling

| path | what is there |
|---|---|
| `TLE_PCB/` | the ported TLE toolkit: `tlelib/` (proto, slcan canlink, 150 Hz backend, broadcast OTA, usbflash), `VEMA_TLE_controller.py`, `VEMA_TLE_flash.py`, `tools/tle_bench.py` (hardware acceptance), `docs/` (protocol report). `PORTING.md` lists what changed and the five deadliest protocol traps |
| `legacy_host/` | the legacy portable bundle: OTA/diag/probe/calibration tools, C++ 150 Hz `pc_backend` (builds here, self-test OK), `mocap/mocap.py` bridge, the canonical `docs/can_protocol.md`, `calibration.json` |

## Mocap, kinematics, Kinova

| path | what is there |
|---|---|
| `UMArm_MOCAP/` | NatNet receiver with per-instance rigid-body base (`CanArmMocap` = 2000–2005), marker→q math, marker-frame locks, `sim_stream.py` (camera-free synthetic stream), vendored patched NatNet SDK |
| `UMArm_KINEMATICS/` | product-of-exponentials FK + `canarm_params.py` (the CAN arm's (3,10) table — placeholder until measured live) |
| `UMArm_KINOVA/` | the Gen3 stack: leashed driver, stdio-JSON bridge (`arm_bridge.py` on `.venv_kinova` / `bridge_client.py` anywhere), mocap body 1008 + hand-eye calibration results, `kinova_scene.py` (MJCF, bit-identical to the bench's Gen3), `setup_env.py`. See its `PORTING.md` |

## Visualizer and twin

| path | what is there |
|---|---|
| `viz/` | display-only MuJoCo room: `mjcf_canarm.py` (argument-driven arm MJCF + `build_room_scene`), `multi_arm_viewer.py` (passive viewer, `mj_forward` only, closing it stops nothing), `base_poses.py` (N-robot mocap mounts), `viz_layout.py` (shared array), `self_check.py` |
| `digital_twin/` | documented skeleton mirroring RS485 `UMArm_SIM` — stubs with the paid-for design decisions; `README.md` = collect-then-fit plan, `data_schema.md` = JSONL collection schema + `board_type` addition |

## Testing

| path | what is there |
|---|---|
| `hw_tests/` | hardware-in-the-loop test scripts and dated result reports (CAN bring-up, OTA, mocap, Kinova) |

## Source repos (read-only references)

- `C:\ESP\ESP_Projects\VEMA_MAX22200` (TLE firmware + TLE_PCB origin, branch TLE_PCB)
- `C:\ESP\ESP_Projects\VNEMA_MK8_PIDPWM` (legacy firmware + portable bundle origin; NEVER recursive-copy — 191 GB CSV inside)
- `C:\RUNZE_SRC\RS485_VEMA` (mocap/Kinova/twin origin; other agents work there — read-only)
