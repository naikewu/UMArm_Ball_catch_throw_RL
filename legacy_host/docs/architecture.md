# Architecture

## System Roles

The portable layer has three runtime domains:

| Domain | Responsibility | Reference files |
| --- | --- | --- |
| Firmware actuator node | Receive compact CAN commands, stage targets, latch pressure at sync, promote staged commands to active control, report compact status, support diagnostics and OTA. | `firmware/esp_idf/main/can_routine.c` |
| PC backend | Own the 150 Hz runtime clock, send CAN command frames, send sync, collect replies, align mocap samples to the sync mark, run the mocap→joint observer + forward kinematics, emit synchronized `robot_state` plus JSON/CSV/report data. | `host/pc_backend/src/main.cpp`, `umarm_fk.cpp` |
| Controller / consumer | Drive the backend over JSON stdin/stdout: send target-pressure vectors, read `q`/`qdot`/pressure at 150 Hz. | `examples/mpc_client_example.py` |
| Support tools | Mocap bridge, firmware OTA, diagnostics, probe, dropout/calibration helpers. | `tools/*.py`, `mocap/mocap.py` |

The PC backend is the clock authority for combined robot state. Firmware nodes do not negotiate time with each other. They make staged commands active only when they observe the shared CAN sync frame.

## Runtime Data Flow

1. The host computes the next target/control payload for each selected actuator.
2. In broadcast mode, the host packs those payloads into runtime table frames on IDs `0x091..0x098`.
3. The host sends all selected runtime table frames.
4. The host sends one zero-length sync frame on ID `0x090`.
5. Each firmware node processes its own table slot, then on sync latches the latest filtered pressure and promotes pending target/control to active target/control.
6. Each active node replies with a 2-byte compact status frame on its base ID.
7. The backend collects replies until the configured receive window closes.
8. The backend samples mocap at the same backend-cycle timestamp and emits a synchronized robot-state row.

## Firmware Boundaries

The reference firmware source currently depends on board/application state from `main.h` and the larger ESP-IDF app. When moving to a new PCB, keep the CAN protocol and sync rules stable and replace these board-specific pieces:

- CAN controller initialization and filtering.
- GPIO/SPI/interrupt pin mapping.
- CAN ID persistence.
- Pressure sampling and filtered pressure storage.
- Valve output disable behavior.
- OTA partition and reboot functions.
- FreeRTOS task placement and priorities.

Use `firmware/port/vnema_board_port.h` as the target shape for a cleaner board port.

## Host Boundaries

The backend separates transport from runtime logic through `ICanTransport`.

- `FakeTransport` supports simulation and self-test.
- `SlcanTransport` is the current Windows SLCAN implementation.
- Future host ports should keep `CanFrame`, compact packing, broadcast-table generation, receive-window logic, and mocap clock correction unchanged unless the protocol version intentionally changes.

## Mocap Boundary

The mocap bridge emits JSON lines with raw Motive/NatNet timestamps and full
rigid-body poses (`body_poses` = `id:x:y:z:qx:qy:qz:qw`). The backend calibrates
those raw timestamps into the backend clock domain, then converts the per-frame
poses into 12 joint angles and interpolates them to the CAN sync mark to produce
`(q, qdot)`. The full pose→joint observer, clock alignment, and 150 Hz sampling are
documented in [state_observer.md](state_observer.md). Future native NatNet
implementations must preserve both the corrected-timestamp fields and the
`body_poses` payload the observer depends on.

## Controller Boundary

Consumers (MPC, RL, teleop, system-ID) do not link against the backend. They run as
a separate process and exchange line-delimited JSON over stdin/stdout: commands in
(`set_targets`, `enable_outputs`, `start`/`stop`), `robot_state` out. The protocol
and `robot_state` schema are in [mpc_integration.md](mpc_integration.md); the
reference consumer is `examples/mpc_client_example.py`.
