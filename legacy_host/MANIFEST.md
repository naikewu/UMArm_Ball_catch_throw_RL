# Manifest

This file lists the intentionally included portable-bundle files. Generated artifacts, firmware binaries, local reports, virtual environments, and vendor SDK folders are intentionally excluded.

## Firmware Reference Source

| File | Purpose |
| --- | --- |
| `firmware/esp_idf/main/can_routine.c` | Reference firmware CAN routing, sync application, compact status, diagnostics, OTA, and MCP2515 filter setup. |
| `firmware/esp_idf/include/can_routine.h` | Firmware CAN task and recovery declarations. |
| `firmware/esp_idf/include/main.h` | Current reference protocol constants, board pins, valve-type constants, and shared state declarations. |
| `firmware/esp_idf/include/can.h` | Standard CAN frame and ID definitions. |
| `firmware/esp_idf/include/mcp2515.c` | MCP2515 SPI driver implementation. |
| `firmware/esp_idf/include/mcp2515.h` | MCP2515 register, bitrate, filter, and frame APIs. |
| `firmware/port/README.md` | Hardware porting scaffold for future PCBs. |
| `firmware/port/vnema_board_port.h` | Target board-port interface for future cleanup/refactors. |
| `firmware/example_project/README.md` | ESP-IDF integration notes for copied firmware source. |
| `firmware/example_project/main/CMakeLists.txt` | Example component registration snippet. |

## Host Runtime And Tools

| File | Purpose |
| --- | --- |
| `host/pc_backend/src/main.cpp` | C++ host scheduler, SLCAN/Fake transports, runtime broadcast table generator, mocap clock alignment + pose→joint observer, JSON IPC, CSV/report logging. |
| `host/pc_backend/src/umarm_fk.{h,cpp}` | Forward kinematics from the 12-DOF joint state (U-joint centers + tip). |
| `host/pc_backend/src/viz_shared_memory.{h,cpp}` | Optional Windows shared-memory visualization feed (off unless `--viz-enable`). |
| `host/pc_backend/CMakeLists.txt` | CMake build file for `vnema_backend` (builds the three source units above). |
| `examples/mpc_client_example.py` | Reference controller/MPC client over the JSON stdin/stdout protocol. |
| `tools/can_ota.py` | OTA GUI/CLI, firmware-version scan, broadcast data upload, unicast repair. |
| `tools/can_diag.py` | CAN diagnostic query/clear helper. |
| `tools/can_probe.py` | Compact command/reply probe helper. |
| `tools/can_dropout_tester.py` | 150 Hz dropout and runtime-broadcast validation helper. |
| `tools/calibrate_pressure.py` | Per-actuator ADC↔psi calibration helper that writes `calibration.json`. |
| `mocap/mocap.py` | Mocap simulator and NatNet JSON bridge (emits full rigid-body poses). |
| `requirements.txt` | Python package dependencies for portable tools. |

## Documentation

| File | Purpose |
| --- | --- |
| `README.md` | Bundle entry point and quick start. |
| `docs/architecture.md` | Top-level communication architecture. |
| `docs/mpc_integration.md` | Controller/MPC JSON command + `robot_state` contract. |
| `docs/state_observer.md` | Mocap pose → 12-DOF `(q, qdot)` observer at the 150 Hz sync mark. |
| `docs/can_protocol.md` | Canonical CAN protocol specification. |
| `docs/sync_rules.md` | CAN and mocap synchronization rules. |
| `docs/hardware_porting.md` | Future-PCB migration guide. |
| `docs/install.md` | Build and setup guide. |
| `docs/validation.md` | Smoke, simulation, and hardware validation checklist. |
