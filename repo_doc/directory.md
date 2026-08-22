# Where everything is — UMArm CAN-arm workspace

One line per place. Check here first; fix here in the same change that makes it wrong.

The robot: a ceiling-hung 24-actuator pneumatic UMArm on a 1 Mbit/s CAN bus reached
through a CANable 2.0 slcan dongle (VID:PID 16D0:117E, currently COM58). Top 8
actuators = TLE92464/DVP boards at `0x101–0x108`; lower 16 = legacy 7 mm boards at
`0x109-0x118`. Mocap rigid bodies **2000-2005, verified live 2026-08-21**: the
census sees 500-505 (RS485 arm), 1008 (Kinova) and 2000-2005 (this arm), four
labeled markers each. RS485 sister arm and Kinova Gen3 share the Motive volume
at 192.168.1.100.

Calibrated 2026-08-21, and all three answers live in code: the actuator/axis map
(`UMArm_KINEMATICS/canarm_actuators.py`), the per-plate marker azimuth and the
proximal-joint composition order (`UMArm_MOCAP/canarm_frames.py`), and the link
lengths (`UMArm_KINEMATICS/canarm_params.py`, `MEASURED = True`). fkine now
reproduces the measured u-joint centres to **1.99 mm RMS on held-out multi-joint
poses**; see `hw_tests/report_canarm_axis_2026-08-21.md`. Two hardware faults to
know about: `0x110` leaks from its supply side and reads +5.1 psi at rest, and
`0x104` reads +1.1 psi.

## Root

| file | what it does |
|---|---|
| `bench_env.py` | the single source of ports/IPs/interpreters: CAN dongle resolved by USB identity, mocap/Kinova IPs, IDF env incantation |
| `canarm_control_gui.py` | **the operator GUI**: TLE controller (all 24 boards) + mocap status strip + **Lock plates** (mints this Motive session's marker locks and swaps in the marker receiver) + a **kinematics line** showing the live fkine-vs-mocap u-joint centre error + spawned MuJoCo room viewer. `--self-test` opens no port. KNOWN COST: its 24-board pressure plot holds the cycle to ~145 Hz instead of 150 (measured 2026-08-20; the boards still answer every edge) |
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
| `UMArm_MOCAP/canarm_frames.py` | **the CAN arm's frame convention**: plate frames from the four markers alone, per-plate azimuth `PLATE_AZIMUTH_DEG` and `PROXIMAL_ORDER` both measured; lock minting, the co-rigidity check, `fk_residual_m`. Motive's streamed body frames sit 45 deg round from the mechanism, so this is what a controller reads. Its `templates/canarm_locks.json` is minted per Motive session and **gitignored** |
| `UMArm_KINEMATICS/` | product-of-exponentials FK, `fkine(q, params, order)`, where `order="yx"` is the CAN arm's measured proximal-joint assembly and `"xy"` the legacy default; plus `canarm_params.py` (the measured (3,10) table) and `canarm_actuators.py` (the measured actuator/axis map, with the legacy claim beside it) |
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
| `hw_tests/can_bringup.py` | **CAN acceptance, read-only**: discovery + variant check, CAN diagnostics before/after, a 60 s 150 Hz soak with every enable bit clear, port-release check. Emits no OTA/set-ID/enable frame under any argument and asserts the enable bits are clear in the table before starting. 15/15 on 2026-08-20 |
| `hw_tests/integrated_gui_test.py` | **operator-GUI acceptance, read-only**: drives the real `canarm_control_gui.py` window against the live arm — connect/scan/cycle, mocap strip sim+live, spawned viewer for 60 s, target staged with the enable bit clear, port release. `--phase profile` gates the window's two periodic jobs on and off to attribute the cycle-rate loss. 28/29 on 2026-08-20; the one failure is the plot's cost, not the bus |
| `hw_tests/media/` | screenshots from the GUI tests (window, viewer, staged target) |
| `hw_tests/results/` | dated machine-readable records from the above. **Gitignored except** the records the committed calibration constants cite: `drive_2026-08-21.json` (the campaign), `axis_analysis_2026-08-21.{json,md}` (its reduction) and `gui_kinematics_2026-08-21.json` |
| `hw_tests/canarm_drive_campaign.py` | **the calibration session**: drives each of the 24 boards alone at 12 psi against a resting arm, then random multi-joint poses, recording every plate's mean marker cloud. One actuator live at a time; unused boards held enabled at 0.5 psi because `0x110` leaks; every exit path clears the enable bits |
| `hw_tests/canarm_axis_analysis.py` | **the reduction**: frame-inference health, chain lengths, the actuator/axis map, the azimuth calibration, and fkine-vs-mocap with a held-out split. Prints the constants to paste into the modules |
| `hw_tests/canarm_gui_kinematics.py` | live GUI check of Lock plates and the kinematics line; opens no serial port. 11/11 on 2026-08-21 |
| `hw_tests/report_canarm_axis_2026-08-21.md` | **the write-up**: the map, the 45 deg marker-azimuth finding, the proximal-order defect in fkine, and the error budget |
| `hw_tests/mocap_census.py` | **run this first** — wide-open NatNet receiver, enumerates every rigid-body id with per-id rates; settles which block belongs to which arm |
| `hw_tests/canarm_mocap_live.py` | CAN-arm q health + per-body dropouts + the five plate-gap chain measurement + room roster; `--sim` rehearses it without cameras |
| `hw_tests/mocap_wire_probe.py`, `mocap_sniff.py`, `mocap_discover.py` | below-the-SDK diagnostics for "run() returned true but no frames": raw multicast/unicast sockets, NAT_PING, all-interface port sweep |

## Source repos (read-only references)

- `C:\ESP\ESP_Projects\VEMA_MAX22200` (TLE firmware + TLE_PCB origin, branch TLE_PCB)
- `C:\ESP\ESP_Projects\VNEMA_MK8_PIDPWM` (legacy firmware + portable bundle origin; NEVER recursive-copy — 191 GB CSV inside)
- `C:\RUNZE_SRC\RS485_VEMA` (mocap/Kinova/twin origin; other agents work there — read-only)
