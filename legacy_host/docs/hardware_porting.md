# Hardware Porting Guide

Use this checklist when moving the communication layer to a new PCB or actuator board.

## Preserve First

Do not change these while bringing up new hardware:

- Compact command/status packing.
- Base/control/data/status ID derivation.
- Broadcast sync ID and DLC-0 sync rule.
- Runtime table frame layout.
- Staged-command promotion on sync.
- Pressure latch at sync.
- Detailed diagnostics for RX overflow, TX failures, bad frames, MERR/ERRIF, and starvation.

## Board Items To Replace

| Area | Current reference | New-board action |
| --- | --- | --- |
| CAN controller | MCP2515 over SPI | Provide equivalent send, receive, filter, interrupt, error, and recovery behavior. |
| Transceiver | MCP25625-class CAN transceiver | Confirm 1 Mbps physical layer, termination, and bus wiring. |
| Oscillator/bit timing | MCP2515 16 MHz, `CAN_1000KBPS` | Recalculate bit timing if oscillator or controller changes. |
| Filters | RXB0 device/runtime group, RXB1 sync/control/data | Preserve logical separation if hardware has receive FIFOs/mailboxes. |
| Interrupt | Falling-edge GPIO from CAN controller | Wake a high-priority CAN task and drain all pending frames. |
| SPI pins | Board-specific ESP32-S3 pins in `main.h` | Move to board config or Kconfig. |
| CAN ID storage | NVS-backed `device_can_id` | Provide persistent base ID with a safe default and `CMD_SET_ID` support if needed. |
| Pressure input | ADC filtered pressure in ADC counts | Provide latest filtered pressure before sync; keep compact range `0..4095`. |
| Outputs | Valve GPIO/LEDC outputs | Provide forced-off and enable-gated output writes. |
| OTA | ESP-IDF OTA partitions | Keep protocol if using ESP-IDF; replace with target bootloader protocol otherwise. |
| Tasking | FreeRTOS CAN/control/watchdog tasks | Keep CAN receive task priority high enough to drain frames before sync pressure. |

## CAN Controller Migration

If a future board uses native ESP32 TWAI or another CAN controller, map the reference behavior rather than the MCP2515 register API:

1. Configure standard 11-bit filters for the actuator base ID, runtime group ID, broadcast sync ID, host control ID, and host data ID.
2. Ensure runtime group traffic cannot block or overwrite sync traffic under peak load. Use separate FIFOs/mailboxes if available.
3. Drain all pending RX frames on each interrupt or task wake.
4. Record overflow and bus errors in detailed diagnostics.
5. Never leave the compact error bit sticky forever after a recoverable condition; expose details through diagnostic frames.

## Pressure And Control Migration

The communication layer expects these shared state concepts:

- `pending_target_pressure` and `pending_control_byte`: written by command handling.
- `active_target_pressure` and `active_control_byte`: promoted only by sync.
- `latched_pressure`: copied from the latest filtered pressure during sync.
- `status_flags`: compact status bits and error reporting.

Future control algorithms can differ. The protocol only requires that active target/control are sync-gated and that disabled outputs become safe quickly.

## Hardware Bring-Up Order

1. Build firmware with outputs forced disabled.
2. Verify CAN ID scan and firmware version query.
3. Clear diagnostics and run compact probe with target 0, flags 0.
4. Run 150 Hz status-only broadcast with outputs disabled.
5. Query diagnostics and confirm no RX overflow or status errors.
6. Enable actuator outputs only after communication passes the status-only tests.
7. Run a short controlled step test and confirm target changes appear only after sync.
