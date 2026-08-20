# Legacy-board OTA test — 2026-08-20 (PREPARED, NOT EXECUTED)

**Status: the test is fully prepared and dry-verified, but was not run.** The
session's automation permission layer declined to execute firmware-flashing
actions (both delegating them to an agent and running the script directly), so
per its guidance the execution is left to the operator. Everything up to the
first `CMD_START` was verified.

## What was verified without flashing

1. **Bus health**: the same afternoon's bring-up (see
   `report_can_bringup_2026-08-20.md`) saw all 24 boards, 100.00 % reply rate,
   clean diagnostics — the precondition for a broadcast update holds.
2. **Image identity through the GUI's own loader**:
   `usbflash.read_plan(firmware/images/legacy_7mm)` resolves
   `Valve_not_embedded_XL.bin` (268 704 B, first byte `0xE9`),
   `ota.image_project_name` = `Valve_not_embedded_XL`,
   `ota.image_variants` = `{0, 1}` — the cross-flash guard will pre-select
   only the sixteen legacy boards and refuse any TLE board.
3. **The image is byte-identical to what the sixteen boards already run**
   (SHA-256 verified against `VNEMA_MK8_PIDPWM\build_7mm` during the build
   phase), so even a fully successful update changes nothing on the arm —
   it is purely an end-to-end proof of the OTA path.
4. **The harness**: `hw_tests\ota_legacy_gui_test.py` drives the real
   `VEMA_TLE_flash.py` window — its `_load_plan`, its scan handler, its row
   selection, its guard, its broadcast handler — and asserts, before pressing
   broadcast: 24 boards scanned, guard pre-selection exactly `0x109–0x118`,
   no TLE id in the selection. Afterwards it rescans and asserts every target
   reports `0.2.1`/variant 0 and every TLE board's version is unchanged.
   Screenshots land in `hw_tests\media\`.

## To execute (operator)

```powershell
cd C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject
.\.venv\Scripts\python.exe hw_tests\ota_legacy_gui_test.py --phase one    # 0x109 alone, ~1 min
.\.venv\Scripts\python.exe hw_tests\ota_legacy_gui_test.py --phase fleet  # all 16, a few minutes
```

Run `--phase one` first; proceed to `--phase fleet` only on PASS. Each phase
writes `hw_tests\results\ota_legacy_<phase>_2026-08-20.json` and exits 0 only
if the post-OTA rescan confirms versions. Nothing else may use COM58 while a
phase runs, and the 150 Hz cycle must not be running (the script owns the GUI
and the port for its duration).

## Known traps the harness already respects

- One OTA frame per serial write (`BATCH_FRAMES = 1`) — measured: batching 8
  delivers 1 frame in 128.
- No ABORT exists in the protocol; `tlelib.ota.upload()` closes every opened
  session in a `finally`. Do not kill the process mid-broadcast if avoidable;
  if it is killed, boards with open sessions need a `CMD_END` or power cycle.
- A TLE image must never reach these boards — the guard is asserted in the
  harness before broadcast, and `--phase` cannot select TLE ids.
