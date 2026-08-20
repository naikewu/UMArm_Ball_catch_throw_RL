# May 18 Fixed Error and Miss Report

Date: May 18, 2026

Scope: CAN compact status errors, missed-frame validation, firmware RX-buffer fix, OTA deployment, backend runtime validation, and mocap timestamp correction.

## Executive Summary

The main CAN issue was the repeated compact status value `12` on the last several actuators, especially `0x113` through `0x118`, during 24-board broadcast runtime operation. The compact value was not a single error code. It decoded to a bitmask:

- `0x04` = `CAN_STATUS_COMMAND_SEEN`
- `0x08` = `CAN_STATUS_ERROR`
- `0x0C` / decimal `12` = command was seen, but the actuator also latched a CAN error for that cycle

Live diagnostics showed the real underlying error was MCP2515 receive overflow, specifically `RX1OVR`, reported by firmware diagnostics as `RX_OVERFLOW`. The overflow happened because each actuator's runtime table frame and the broadcast sync frame were both being filtered into MCP2515 receive buffer `RXB1`. For the last runtime-table slots, the runtime frame and sync frame arrived very close together, and `RXB1` could hold only one unread frame. When the firmware had not drained `RXB1` before the sync arrived, the MCP2515 set `RX1OVR`, and firmware returned compact status `12`.

The working fix was firmware-side RX-buffer separation:

- `RXB0`: device-specific command ID plus the actuator's runtime table ID
- `RXB1`: broadcast sync plus host control/data IDs
- Disable RXB0 rollover into RXB1 by clearing `RXB0CTRL_BUKT`

After building and OTA-deploying firmware version `0.2.1` to all 24 actuators, the repeat 30 second, 150 Hz broadcast test produced:

- `4500/4500` replies on every actuator
- `0` misses on every actuator
- `0` late replies
- `0` compact status errors
- high-ID diagnostics: `rx_overflow=0`, `tx_fail=0`, `status_flags=0x00`

The mocap issue was also addressed. The backend now treats the PC backend clock as the authoritative runtime clock, computes a mocap timestamp offset from live NatNet samples, and recalibrates that offset on an exact rolling 5 second window when approximately 5 seconds of 360 Hz mocap samples are present. Runtime reports now include the corrected mocap timestamp, raw mocap timestamp, timestamp offset, frame rate, frame drop count, sample count, and 5 second update count.

## Symptoms Observed

### CAN Status 12 On Last Actuators

The original visible symptom was that the final actuators in the runtime cycle, especially `0x113` through `0x118`, returned compact status `12` while still replying. This mattered because the missing-frame count could be zero, yet the actuator was still telling us that its local CAN controller had seen an error.

The key observation was that the problem followed the broadcast runtime timing pattern. It was not simply a pressure-control error, an OTA error, or a host parsing error.

### Misses Versus Status Errors

Two separate failure classes were present during debugging:

- Missing replies: the host did not receive a compact status reply in the expected receive window.
- Status errors: the actuator did reply, but the compact status nibble contained `CAN_STATUS_ERROR`.

The final root cause for the status `12` issue was not ordinary host-side missing replies. The high-ID boards were replying, but they were reporting MCP2515 RX overflow.

### Mocap Latency Drift

The mocap symptom was a slow apparent latency increase toward about 30 ms after around 20 minutes. The likely cause was not actual transport delay, but clock disagreement between the mocap timestamp domain and the PC/backend timestamp domain. If raw mocap time and PC time drift apart, computed latency will creep even when the live stream is healthy.

## CAN Protocol And Diagnostic Background

Runtime CAN IDs:

- Actuator base IDs: `0x101..0x118`
- Broadcast sync ID: `0x090`
- Runtime table IDs: `0x091..0x098`
- Runtime table frame layout: 3 actuator slots per frame
- Host control/data IDs: `CAN_ID_HOST_CTRL`, `CAN_ID_HOST_DATA`

Compact status bits:

- `CAN_STATUS_COMMAND_SEEN = 0x04`
- `CAN_STATUS_ERROR = 0x08`
- Decimal `12` is `0x0C`, so it means both bits were set.

Firmware diagnostics added during this work made the compact error actionable. The diagnostic query could distinguish:

- `RX_OVERFLOW`
- `TX_FAIL`
- `TX_ALL_BUSY`
- bad frame
- MCP2515 `MERR` / `ERRIF`
- starvation/warning conditions

The decisive diagnostic after reproducing the issue was:

- `last_reason=0x01 RX_OVERFLOW`
- `last_eflg=0x80 RX1OVR`
- `last_send_error=0 ERROR_OK`
- `tx_all_busy=0`

That combination showed the actuator was not failing to transmit its reply. Instead, the MCP2515 was dropping received frames in `RXB1` before firmware could drain them.

## Tests Run Before The Fix

### 1. Transaction Mode, Normal/Reverse/Rotate Order

We tested actuator order changes because the failing boards were at the end of the cycle. The transaction-mode tests moved the order of actuator requests:

- normal order
- reverse order
- rotating order

Result: transaction mode did not reproduce the same high-ID compact status error pattern. It produced timing pressure and late/missed replies in some cases, but status errors were not the same failure mode. That told us the problem was tied to the broadcast runtime frame/sync sequence rather than just logical actuator order.

Relevant reports:

- `hardware_comm_layer/reports/can_dropout_20260518_222709_transaction_normal_150.0hz.md`
- `hardware_comm_layer/reports/can_dropout_20260518_222859_transaction_reverse_150.0hz.md`
- `hardware_comm_layer/reports/can_dropout_20260518_223146_transaction_rotate_150.0hz.md`

### 2. Broadcast Runtime Reproducer

Command shape:

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_dropout_tester.py --mode broadcast --order normal --ids 0x101-0x118 --prelude single --duration 30 --port COM4
```

This reproduced the important status-error pattern.

Report: `hardware_comm_layer/reports/can_dropout_20260518_223630_broadcast_normal_150.0hz.md`

Key rows from the pre-fix broadcast report:

| ID | Replies | Misses | Status Errors | Avg Latency ms | Max Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0x113 | 4500 | 0 | 4488 | 3.134 | 3.996 |
| 0x114 | 4500 | 0 | 4485 | 3.199 | 4.082 |
| 0x115 | 4500 | 0 | 4487 | 3.265 | 4.167 |
| 0x116 | 4500 | 0 | 4500 | 3.329 | 4.252 |
| 0x117 | 4500 | 0 | 4500 | 3.393 | 4.339 |
| 0x118 | 4500 | 0 | 4500 | 3.456 | 4.444 |

This proved the original problem was not that those boards were failing to reply. They replied every cycle, but nearly every reply had the compact error bit set.

### 3. Individual CAN Diagnostics After Broadcast Reproducer

All-board diagnostic queries could miss some diagnostic frames because many boards replied at once, so individual board queries were used for the high IDs.

The high-ID individual diagnostic result showed:

- `last_reason=0x01 RX_OVERFLOW`
- `last_eflg=0x80 RX1OVR`
- `last_send_error=0 ERROR_OK`
- `tx_all_busy=0`

This isolated the root cause to MCP2515 receive overflow in `RXB1`.

### 4. Host-Side Pacing Experiments

We also tried host-side pacing before sync as a possible workaround.

The pacing experiments were not acceptable as the fix:

- `--inter-frame-us 200` produced about 95% misses in a 5 second broadcast test.
- `--inter-frame-us 50` also produced about 94% misses in a 5 second broadcast test.

Relevant reports:

- `hardware_comm_layer/reports/can_dropout_20260518_224103_broadcast_normal_150.0hz.md`
- `hardware_comm_layer/reports/can_dropout_20260518_224152_broadcast_normal_150.0hz.md`

Those tests reduced compact status errors only by disrupting the receive timing badly. They were useful as diagnostics, but not as a production fix.

## Root Cause

The MCP2515 hardware has two receive buffers:

- `RXB0`, matched by `RXF0/RXF1` using `MASK0`
- `RXB1`, matched by `RXF2..RXF5` using `MASK1`

Before the fix, the firmware put the actuator runtime table frame and the broadcast sync frame into `RXB1`. This was risky for broadcast runtime mode because every cycle sends several runtime-table frames followed by sync. For the last actuator IDs, the matching runtime-table frame is one of the last table frames. That leaves very little time between that runtime-table frame and the broadcast sync frame.

If both frames are assigned to `RXB1`, the sequence can be:

1. Runtime table frame for the high-ID group arrives in `RXB1`.
2. Firmware has not drained `RXB1` yet.
3. Broadcast sync arrives, also destined for `RXB1`.
4. MCP2515 sets `RX1OVR` because the single `RXB1` slot is still occupied.
5. Firmware records `RX_OVERFLOW`.
6. The next compact status reply contains `CAN_STATUS_ERROR`, producing decimal status `12`.

This explains why the highest IDs were the worst affected. Their runtime-table frame is closest to the sync frame in time.

## Fix That Worked

### Firmware RX-Buffer Split

The working firmware change is in `main/can_routine.c`.

Current filter setup:

```c
MCP2515_setFilterMask(MASK0, false, 0x7FF);
MCP2515_setFilter(RXF0, false, device_can_id);
MCP2515_setFilter(RXF1, false, runtime_table_can_id());
MCP2515_modifyRegister(MCP_RXB0CTRL, RXB0CTRL_BUKT, 0);

MCP2515_setFilterMask(MASK1, false, 0x7FF);
MCP2515_setFilter(RXF2, false, BROADCAST_CAN_ID);
MCP2515_setFilter(RXF3, false, BROADCAST_CAN_ID);
MCP2515_setFilter(RXF4, false, CAN_ID_HOST_CTRL);
MCP2515_setFilter(RXF5, false, CAN_ID_HOST_DATA);
```

Behavior after the change:

- Device command and runtime table go to `RXB0`.
- Broadcast sync and host control/data go to `RXB1`.
- `RXB0CTRL_BUKT` is cleared so `RXB0` cannot roll over into `RXB1`.

This prevents the runtime table frame from occupying `RXB1` immediately before sync. Sync now has its own receive buffer path.

### Sync Timing Semantics

The broadcast sync frame is a firmware-observed synchronization event, not a hard hardware latch edge. The MCP2515 receives the sync frame, asserts the interrupt line, the ESP32-S3 ISR wakes the CAN task, and the CAN task reads the MCP2515 over SPI before `handle_sync_frame()` calls `apply_sync_edge()`.

Under the current task/SPI structure, the expected delay from the electrical arrival of the sync frame to the firmware executing `apply_sync_edge()` is on the order of `0.1..0.5 ms` during normal load. This is an engineering estimate based on ISR wakeup plus MCP2515 SPI transaction overhead, not a directly instrumented timing measurement.

`apply_sync_edge()` copies the latest filtered pressure value already present in `acr->adc_RAF_result` into `pcr->latched_pressure`, promotes `pending_target_pressure` into `active_target_pressure`, and then the CAN task sends the compact status reply. Therefore the compact reply means the board processed the sync and promoted the staged command; it does not prove the valve control output has already reacted. The DT PWM control loop reads `active_target_pressure` on its own 4 ms loop, and the 7 mm bang-bang loop reads it on roughly a 1 ms FreeRTOS delay cadence.

This timing model is acceptable for the validated 150 Hz broadcast runtime path because the post-fix tests showed in-window replies, zero late replies, and zero RX-overflow errors. It should not be interpreted as exact simultaneous execution at the physical CAN sync edge.

### Firmware Version Bump

The project version was bumped to `0.2.1` in `CMakeLists.txt`:

```cmake
set(PROJECT_VER "0.2.1")
```

The OTA scan/version query confirmed that every actuator reported `0.2.1` after deployment.

## Build And Deployment

Both firmware variants were built successfully:

```powershell
idf.py -B build_dt -DVALVE_TYPE=1 build
idf.py -B build_7mm -DVALVE_TYPE=0 build
```

Build result:

- DT image built as version `0.2.1`.
- 7mm image built as version `0.2.1`.

OTA dry run confirmed the intended mapping:

- `0x101..0x108` -> `build_dt/Valve_not_embedded_XL.bin`
- `0x109..0x118` -> `build_7mm/Valve_not_embedded_XL.bin`

OTA deployment command shape:

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_ota.py --cli --ids all --port COM4 --continue-on-fail
```

OTA result:

- `0x101..0x108`: DT batch completed successfully.
- `0x109..0x118`: 7mm batch completed successfully.
- All selected OTA operations completed.

Firmware version query after OTA:

- `0x101..0x108`: firmware version `0.2.1 (DT)`
- `0x109..0x118`: firmware version `0.2.1 (7mm)`

## Tests Run After The Fix

### 1. Clear CAN Diagnostics

Before the post-fix validation, diagnostics were cleared:

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_diag.py --ids 0x101-0x118 --clear --port COM4 --timeout 5
```

After clearing, error counters were zero.

### 2. First Post-Fix Broadcast Test

Command:

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_dropout_tester.py --mode broadcast --order normal --ids 0x101-0x118 --prelude single --duration 30 --port COM4
```

Report: `hardware_comm_layer/reports/can_dropout_20260518_224903_broadcast_normal_150.0hz.md`

Result:

- Status errors were zero on all boards.
- `0x112` and `0x113` each had one missed reply.
- No RX overflows were observed in high-ID diagnostics.

Key high-ID rows:

| ID | Replies | Misses | Status Errors | Avg Latency ms | Max Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0x112 | 4499 | 1 | 0 | 2.808 | 3.688 |
| 0x113 | 4499 | 1 | 0 | 3.128 | 3.749 |
| 0x114 | 4500 | 0 | 0 | 3.192 | 3.809 |
| 0x115 | 4500 | 0 | 0 | 3.260 | 3.913 |
| 0x116 | 4500 | 0 | 0 | 3.325 | 4.008 |
| 0x117 | 4500 | 0 | 0 | 3.393 | 4.080 |
| 0x118 | 4500 | 0 | 0 | 3.456 | 4.156 |

Because the original failure was status errors from RX overflow, this was already a strong positive result. The one-off misses were checked with a repeat run.

### 3. Repeat Post-Fix Broadcast Test

Command:

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_diag.py --ids 0x101-0x118 --clear --port COM4 --timeout 5
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_dropout_tester.py --mode broadcast --order normal --ids 0x101-0x118 --prelude single --duration 30 --port COM4
```

Report: `hardware_comm_layer/reports/can_dropout_20260518_225116_broadcast_normal_150.0hz.md`

This was the clean validation run.

Summary:

- 24 boards tested.
- 30 seconds.
- 150 Hz.
- 4500 expected replies per board.
- Every board returned `4500/4500` replies.
- Every board had `0` misses.
- Every board had `0` late replies.
- Every board had `0` status errors.

High-ID rows from the clean report:

| ID | Replies | Misses | Status Errors | Avg Latency ms | Max Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0x112 | 4500 | 0 | 0 | 2.859 | 3.798 |
| 0x113 | 4500 | 0 | 0 | 3.156 | 3.867 |
| 0x114 | 4500 | 0 | 0 | 3.222 | 3.951 |
| 0x115 | 4500 | 0 | 0 | 3.289 | 4.030 |
| 0x116 | 4500 | 0 | 0 | 3.354 | 4.110 |
| 0x117 | 4500 | 0 | 0 | 3.423 | 4.169 |
| 0x118 | 4500 | 0 | 0 | 3.488 | 4.229 |

### 4. Post-Fix High-ID Diagnostics

Individual diagnostics were queried for `0x112..0x118` after the clean run.

Important result:

- `rx_overflow=0`
- `tx_fail=0`
- `tx_all_busy=0`
- `bad_frame=0`
- `merr=0`
- `errif=0`
- `status_flags=0x00`
- `last_eflg=0x00 none`

Some boards reported a small number of `STARVATION` warnings. Those did not set compact error status, did not cause misses, and did not indicate MCP2515 RX overflow. The failure mode that caused status `12` was gone.

## Backend Live Runtime Result

The backend was also run in live SLCAN broadcast mode with live NatNet mocap.

Report: `hardware_comm_layer/reports/vnema_backend_20260518_225507.md`

Backend summary:

- Mode: live SLCAN
- CAN protocol: broadcast
- CAN order: normal
- RX window fraction: `0.820`
- Selected IDs: `0x101..0x118`
- Cycles: `5143`
- Expected replies: `123432`
- Received replies: `123432`
- Missed replies: `0`
- Unexpected replies: `0`
- Duplicate replies: `0`
- Transport parse errors: `0`
- All actuator status-error counts: `0`

This confirms the fix also works through the C++ backend runtime path, not only the Python dropout tester.

## Mocap Fix And Validation

### Original Mocap Problem

The mocap latency appeared to creep upward slowly, reaching around 30 ms after about 20 minutes. The likely cause was clock-domain drift or offset mismatch between mocap timestamps and PC/backend time. Using raw mocap timestamps directly can make latency calculations look worse over time even when the actual packet delivery remains healthy.

### Implemented Mocap Clock Alignment

The backend now calibrates mocap timestamps against the PC/backend receive clock.

Implementation location: `hardware_comm_layer/pc_backend/src/main.cpp`, `MocapClockCalibrator`.

Behavior:

1. On the first valid mocap sample, initialize the offset:
   - `offset_s = receive_host_s - raw_timestamp_s`
2. Store observations containing:
   - raw mocap timestamp
   - host receive timestamp
   - mocap frame number
3. Keep only a rolling 5 second observation window.
4. Recompute the offset when:
   - at least 5 seconds have elapsed since the last calibration,
   - the sample count is high enough for a real 5 second mocap window,
   - the observed frame rate is in the expected mocap range.
5. Return the corrected timestamp:
   - `corrected_timestamp_s = raw_timestamp_s + offset_s`

Constants in the backend:

```c++
static constexpr double kWindowS = 5.0;
static constexpr size_t kMinWindowSamples = 1700;
static constexpr double kMinFrameRateHz = 320.0;
static constexpr double kMaxFrameRateHz = 400.0;
```

The `1700` sample threshold is intentionally close to 5 seconds of 360 Hz mocap data. It prevents the offset from being recalculated from a thin or partial window.

### Mocap Data Recorded In Runtime Reports

The backend CSV/report path now records:

- corrected mocap timestamp
- raw mocap timestamp
- mocap timestamp offset in ms
- mocap frame rate
- mocap frame drop count
- mocap clock sample count
- mocap clock 5 second update count
- mocap latency statistics

The GUI/backend state also exposes the calibrated fields so the operator can see whether the backend is using a real rolling calibration window.

### Mocap Runtime Evidence

The live backend report after the changes showed mocap running with the calibrated backend path:

- Mocap: live NatNet
- Mocap latency samples: `3013`
- Last mocap latency ms: `0.411`
- Avg mocap latency ms: `1.661`
- Max mocap latency ms: `6.455`
- Last mocap frame rate Hz: `359.962`
- Mocap clock sample count: `1741`
- Mocap clock 5s updates: `4`

The timestamp offset value itself can be numerically large because the raw mocap timestamp and the PC clock may use different epochs. That is expected. The important part is that the backend converts raw mocap timestamps into the PC/backend clock domain before computing latency and before combining mocap with CAN runtime data.

## Files Changed For This Fix Set

Key files involved in the completed work:

- `CMakeLists.txt`
  - Firmware version bumped to `0.2.1`.

- `main/CMakeLists.txt`
  - Firmware version and variant compile definitions feed the version query path.

- `include/main.h`
  - Compact status bits, diagnostic reason bits, firmware version command/response constants.

- `main/can_routine.c`
  - Runtime table command staging.
  - Broadcast sync status sending.
  - CAN diagnostics.
  - Firmware version response.
  - MCP2515 filter split that fixed the RX overflow.

- `hardware_comm_layer/CAN_bus_handler/can_dropout_tester.py`
  - Broadcast runtime tester, order tests, status-error counting, report generation.

- `hardware_comm_layer/CAN_bus_handler/can_diag.py`
  - Diagnostic clear/query tool used to identify `RX_OVERFLOW` / `RX1OVR`.

- `hardware_comm_layer/CAN_bus_handler/can_ota.py`
  - OTA upload and firmware version query/display.

- `hardware_comm_layer/CAN_bus_handler/can_gui.py`
  - GUI controls and display for CAN runtime testing and mocap/backend fields.

- `hardware_comm_layer/pc_backend/src/main.cpp`
  - Broadcast runtime support.
  - Status-error counters.
  - CAN order and RX-window options.
  - Mocap timestamp offset calibration and report fields.

## How To Reproduce The Validation

### Build Both Firmware Variants

```powershell
idf.py -B build_dt -DVALVE_TYPE=1 build
idf.py -B build_7mm -DVALVE_TYPE=0 build
```

### Deploy Over CAN OTA

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_ota.py --cli --ids all --port COM4 --continue-on-fail
```

### Confirm Firmware Versions

```powershell
.\.venv\Scripts\python.exe -c "from hardware_comm_layer.CAN_bus_handler.can_ota import open_slcan_bus, request_firmware_versions; bus=open_slcan_bus('COM4',1000000,2000000); request_firmware_versions(bus, range(0x101,0x119)); bus.shutdown()"
```

Expected result:

- `0x101..0x108`: `0.2.1 (DT)`
- `0x109..0x118`: `0.2.1 (7mm)`

### Clear Diagnostics

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_diag.py --ids 0x101-0x118 --clear --port COM4 --timeout 5
```

### Run Broadcast Dropout/Status Test

```powershell
.\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_dropout_tester.py --mode broadcast --order normal --ids 0x101-0x118 --prelude single --duration 30 --port COM4
```

Expected result matching the clean validation report:

- `4500` replies per board
- `0` misses
- `0` late replies
- `0` status errors

### Query High-ID Diagnostics

```powershell
foreach ($id in @('0x112','0x113','0x114','0x115','0x116','0x117','0x118')) {
  .\.venv\Scripts\python.exe hardware_comm_layer\CAN_bus_handler\can_diag.py --ids $id --port COM4 --timeout 2
}
```

Expected result:

- `rx_overflow=0`
- `tx_fail=0`
- `status_flags=0x00`
- `last_eflg=0x00 none`

## Final Result

The firmware RX-buffer split fixed the compact status `12` issue for the high-ID actuators. The final validation run reached the requested target of zero missing frames and zero status errors in the 24-board, 150 Hz broadcast runtime test.

The mocap timing correction is also in place. The backend now continuously aligns mocap timestamps to PC/backend time using an exact rolling 5 second calibration window, and the live backend report shows low mocap latency with successful 5 second offset updates.
