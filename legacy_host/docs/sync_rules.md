# Synchronization Rules

## CAN Sync Semantics

The CAN sync frame is a zero-length standard CAN frame on ID `0x090`.

Firmware treats the sync frame as a shared state transition:

```text
pending target/control -> active target/control
latest filtered pressure -> latched pressure for reply
```

Commands received before sync are staged. They do not become active control commands until sync is processed.

## Broadcast Runtime Cycle

The reference 150 Hz broadcast cycle is:

1. Host starts a backend-cycle timestamp.
2. Host sends runtime table frames for selected actuators.
3. Host sends DLC-0 sync on `0x090`.
4. Boards process sync, latch pressure, promote commands, and send compact replies.
5. Host receives replies until `period * rx_window_frac` expires.
6. Host samples mocap in backend time for the same cycle.

The compact reply means the board processed the sync and promoted the staged command. It does not prove the physical valve output has already changed. The pressure-control task consumes active command state on its own loop cadence.

## Timing Limitations

The broadcast sync is firmware-observed, not a dedicated hardware latch edge. On the reference board, the MCP2515 receives sync, asserts the interrupt line, the ESP32-S3 CAN task wakes, and the task reads the frame over SPI before calling the sync handler.

The observed design is valid for the tested 150 Hz path, but future agents must not document it as exact simultaneous physical execution at the electrical CAN edge.

## Pressure Latch Rule

The pressure value in compact status is the filtered pressure latched at sync. It is intentionally decoupled from per-board reply transmit time, so host rows from one cycle represent the same logical sync instant even though replies arrive at different times.

## Safe Disable Rule

When outputs are disabled, host commands clear `CAN_CONTROL_ENABLE`. Firmware also forces outputs off when OTA starts and when active control is disabled. A future board port must preserve a deterministic all-outputs-off path that does not depend on normal control-loop progress.

## Mocap Clock Rule

The backend clock is the authority for synchronized robot state. Raw Motive/NatNet timestamps may use a different epoch or drift slightly relative to backend time.

The backend therefore:

1. Receives raw mocap timestamp and backend receive timestamp.
2. Maintains a rolling offset between mocap time and backend time.
3. Recomputes offset over about 5 seconds when about 360 Hz samples are present.
4. Produces corrected mocap timestamp in backend time.
5. Computes latency and cycle-aligned robot state from corrected timestamps.

Do not combine raw NatNet timestamps directly with CAN sync timestamps.

## Mocap Sampling Rule

Mocap runs faster than CAN in the reference setup, about 360 Hz versus 150 Hz. The backend selects or interpolates a mocap state at the CAN cycle timestamp. Velocity estimates should use neighboring mocap samples when available.

The full pipeline — clock alignment, the rigid-body pose → 12-DOF joint mapping, the 150 Hz interpolation/extrapolation, and the fixed-delay snapshot — is documented in [state_observer.md](state_observer.md).

## Validation Rule

Any hardware port must demonstrate:

- No persistent compact status error flags during the target cycle rate.
- No RX overflow under the selected actuator set.
- Command application is sync-gated.
- Pressure replies represent sync-latched values.
- Mocap latency and corrected timestamp fields remain bounded during a soak run.
