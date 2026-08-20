# Portable Synchronized Communication Layer

This folder packages the VNEMA low-latency communication layer so a future project or AI agent can copy one directory and recreate the same CAN + mocap runtime behavior.

The reference implementation is the current ESP32-S3 + MCP2515/MCP25625 actuator network, a Windows SLCAN host backend, and an OptiTrack/NatNet mocap bridge. Future hardware can change the PCB, CAN controller, sensors, or valve driver as long as it preserves the protocol and sync semantics documented here.

The backend does two jobs every 150 Hz cycle: it **commands the robot** (sends each
actuator's target pressure, then a broadcast sync, then collects pressure replies)
and it **reads the robot's state** (aligns ~360 Hz motion-capture poses to the sync
mark and converts them into a 12-DOF joint state `q`, `qdot`). A controller (e.g.
MPC) consumes that state and publishes target pressures back — over a simple JSON
stdin/stdout protocol — without touching real-time code.

## Quick Start (no hardware)

```powershell
cmake -S host\pc_backend -B host\pc_backend\build -G Ninja -DCMAKE_CXX_COMPILER=C:\Strawberry\c\bin\g++.exe
cmake --build host\pc_backend\build
host\pc_backend\build\vnema_backend.exe --self-test
python examples\mpc_client_example.py --seconds 3
```

The example client launches the backend in CAN+mocap simulation, streams
synchronized `q`/`qdot`/pressure at 150 Hz, and pushes a target vector each cycle.

## Start Here

1. Read [docs/architecture.md](docs/architecture.md) for the system boundary and data flow.
2. Read [docs/mpc_integration.md](docs/mpc_integration.md) for the JSON command / `robot_state` contract a controller uses.
3. Read [docs/state_observer.md](docs/state_observer.md) for how mocap poses become `q`/`qdot` at the 150 Hz sync mark.
4. Read [docs/can_protocol.md](docs/can_protocol.md) before changing IDs, payloads, filters, or OTA behavior.
5. Read [docs/sync_rules.md](docs/sync_rules.md) before changing timing or scheduler code.
6. Read [docs/hardware_porting.md](docs/hardware_porting.md) when moving to a new PCB or CAN controller.
7. Follow [docs/install.md](docs/install.md) to build the backend and install Python dependencies.
8. Follow [docs/validation.md](docs/validation.md) before trusting a new port on hardware.

## Contents

- `firmware/esp_idf`: reference firmware communication source copied from the validated ESP-IDF project.
- `firmware/port`: hardware-porting scaffold and checklist for future boards.
- `host/pc_backend`: C++ backend for 150 Hz CAN scheduling, SLCAN transport, the mocap clock alignment + pose→joint observer, forward kinematics, JSON IPC, CSV logging, and reports.
- `examples`: `mpc_client_example.py` — runnable reference controller showing how to command the robot and read `q`/`qdot` at 150 Hz.
- `tools`: Python OTA, diagnostic, probe, dropout-test, and pressure-calibration tools (CLI).
- `mocap`: Python NatNet bridge and mocap simulator (emits full rigid-body poses).
- `docs`: architecture, controller/MPC integration, state observer, protocol, synchronization, porting, install, and validation guides.

## Copy Policy

Copy this whole folder into a future project. Do not copy generated `build`, `reports`, virtual environment, or binary files into source control. The official NatNet SDK is an external dependency unless its license explicitly permits redistribution in the target project.

## Current Reference Assumptions

- Standard 11-bit CAN at 1 Mbps.
- SLCAN serial link at 2 Mbps by default.
- Actuator base IDs `0x101..0x118`.
- Runtime broadcast sync ID `0x090`.
- Runtime target table IDs `0x091..0x098`.
- Host backend cycle rate 150 Hz.
- Mocap sample rate about 360 Hz, corrected into backend clock time.
- 6 tracked rigid bodies (`1000..1005`) reduced to a 12-DOF joint state `(q, qdot)` at each 150 Hz sync mark.
- Controller link is line-delimited JSON on the backend's stdin/stdout.

The current firmware source is a reference port, not a hardware-neutral HAL. When adapting to a new PCB, preserve the protocol and sync rules first, then replace the board-specific pieces listed in [docs/hardware_porting.md](docs/hardware_porting.md).
