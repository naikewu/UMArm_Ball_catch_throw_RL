# CAN Protocol

This is the canonical protocol for the portable synchronized communication layer.

## Physical And Frame Format

- CAN frame format: standard 11-bit identifiers only.
- CAN bus bitrate: 1 Mbps in the reference system.
- Host adapter: SLCAN at 2 Mbps serial link by default.
- Payload size: classic CAN, maximum DLC 8.

## ID Map

| ID or range | Meaning |
| --- | --- |
| `0x090` | Broadcast sync when DLC is 0. Broadcast OTA data when OTA is active and DLC is nonzero. |
| `0x091..0x098` | Runtime broadcast target-table groups. |
| `0x101..0x118` | Reference actuator base IDs. |
| `0x101..0x108` | Reference DT/big-valve actuator IDs. |
| `0x109..0x118` | Reference 7mm actuator IDs. |
| `base + 0x100` | Host-control command ID for one actuator. |
| `base + 0x200` | Host-data ID for unicast OTA data/repair. |
| `base + 0x300` | ESP status ID for ACK/NACK, diagnostics, OTA status, firmware version. |

## Compact Command

A direct pressure command is a standard frame sent to the actuator base ID with DLC 2.

```text
bits  0..11: target pressure in ADC counts
bits 12..15: control flags
```

Control flags:

| Bit | Value | Meaning |
| --- | ---: | --- |
| 0 | `0x01` | Enable actuator output control. |

Any direct compact command frame with a DLC other than 2 is invalid. The reference firmware records a bad-frame diagnostic and returns compact status.

## Compact Status

Each normal compact status reply is a standard frame on the actuator base ID with DLC 2.

```text
bits  0..11: pressure latched at the most recent sync edge
bits 12..15: status flags
```

Status flags:

| Bit | Value | Meaning |
| --- | ---: | --- |
| 0 | `0x01` | Active control enable is set. |
| 1 | `0x02` | OTA receive mode is active. |
| 2 | `0x04` | At least one valid command has been seen. |
| 3 | `0x08` | A compact CAN/protocol error was observed. |

Status value `12` (`0x0C`) means `command_seen` plus `error`; query detailed diagnostics before guessing the root cause.

## Runtime Broadcast Table

The runtime broadcast path is the validated 24-actuator, 150 Hz path. It reduces host traffic by packing three actuator commands into one 8-byte frame.

Frame IDs are `0x091..0x098`. Each group covers three slots. Slot 0 maps to base ID `0x101`; slot 23 maps to base ID `0x118`.

| Byte | Meaning |
| ---: | --- |
| 0 | `0x80 \| start_slot` |
| 1 | 3-bit active mask for slots 0, 1, and 2 in this frame |
| 2..3 | Compact command payload for `start_slot + 0` |
| 4..5 | Compact command payload for `start_slot + 1` |
| 6..7 | Compact command payload for `start_slot + 2` |

Runtime sequence:

1. Host sends all selected runtime table frames.
2. Host sends DLC-0 sync on `0x090`.
3. Firmware promotes staged commands on sync.
4. Firmware replies with compact status only if its table slot was active before that sync.

The marker bit `0x80` prevents runtime table frames from colliding with broadcast OTA data, whose sequence byte is `0..127`.

## Host-Control Commands

Host-control commands are sent to `base + 0x100` with the command in byte 0.

| Command | Value | Reply/behavior |
| --- | ---: | --- |
| `CMD_START` | `0x01` | Start OTA, force outputs off, reset OTA state, reply `MSG_ACK` on `base + 0x300`. |
| `CMD_END` | `0x02` | Flush/finalize OTA image, select boot partition, ACK, reboot. |
| `CMD_SET_ID` | `0x05` | Persist a new base CAN ID and reboot. |
| `CMD_GET_CAN_DIAG` | `0x06` | Send diagnostic frames `MSG_CAN_DIAG0..3`. |
| `CMD_CLEAR_CAN_DIAG` | `0x07` | Clear diagnostics and send fresh diagnostic frames. |
| `CMD_GET_OTA_STATUS` | `0x08` | Send `MSG_OTA_STATUS`. |
| `CMD_GET_FW_VERSION` | `0x09` | Send chunked `MSG_FW_VERSION` frames. |

## OTA Status

`CMD_GET_OTA_STATUS` replies on `base + 0x300` with DLC 8.

| Byte | Meaning |
| ---: | --- |
| 0 | `MSG_OTA_STATUS` (`0xD4`) |
| 1 | OTA status flags |
| 2 | Expected sequence in the current block |
| 3..4 | Buffered byte count, little-endian |
| 5..6 | Blocks written, little-endian |
| 7 | Low byte of last OTA error code |

OTA status flags are `0x01` active, `0x02` sequence error, `0x04` write error, and `0x08` bad frame.

## Firmware Version

`CMD_GET_FW_VERSION` replies on `base + 0x300` with one or more DLC-8 frames.

| Byte | Meaning |
| ---: | --- |
| 0 | `MSG_FW_VERSION` (`0xD5`) |
| 1 | Zero-based chunk index |
| 2 | Total chunk count |
| 3 | Variant: `0` = 7mm, `1` = DT/big valve |
| 4..7 | Four ASCII version bytes, zero-padded in the last chunk |

## MCP2515 Filter Rule

The validated MCP2515 filter split is part of the protocol implementation contract for this reference hardware:

- RXB0: actuator base ID and that actuator's runtime-table group ID.
- RXB1: broadcast sync ID and host control/data IDs.
- RXB0 rollover into RXB1 disabled.

This prevents the runtime table frame from occupying RXB1 immediately before sync, which previously caused RX1 overflow on high-ID boards during 24-board broadcast operation.
