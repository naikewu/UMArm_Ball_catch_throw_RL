# CAN bring-up verification — 24-board arm, 2026-08-20

**Verdict: PASSED.** All fifteen acceptance checks. Twenty-four boards at `0x101`–`0x118` answered discovery with the expected variant bytes, every one of them answered every one of 9003 sync edges over a 60 s cycle soak, and no board reported an RX overflow, a transmit failure or an error flag. The whole run was read-and-status only: no OTA command, no board-ID command, and no enable bit was ever set.

| | |
| --- | --- |
| Adapter | CANable 2.0, `16e7497-dirty github.com/normaldotcom/canable2.git`, slcan on **COM58** |
| Port resolution | `bench_env.resolve_can_port()` — VID:PID+serial match (`16D0:117E` / `3370376F3435`) |
| Bit rate | 1 Mbit/s (`S8`), normal mode (`O`), never listen-only, never `S7` |
| Host | `WS/.venv/Scripts/python.exe`, `TLE_PCB/tlelib` |
| Script | `hw_tests/can_bringup.py --soak-seconds 60` |
| Run | 2026-08-20 17:54:02 → 17:55:06 |
| Machine-readable record | `hw_tests/results/can_bringup_2026-08-20.json` (+ `.md` table) |
| Comparison run | `hw_tests/results/can_bringup_2026-08-20_drained_rx.json`, 30 s, `--drain-rx` |

No pneumatic supply was connected for any part of this test, so nothing could move even had a command escaped. That is a second line of defence and not the first: the first is that the script decodes the runtime table it is about to send, slot by slot, and refuses to start the cycle if any slot carries the enable bit.

## 1. What was run, and why in this order

Bring-up here means establishing that every board is on the bus, that it is the board the addressing plan says it is, and that it answers the synchronisation edge inside the cycle budget. Those are three separate claims and they fail in different ways, so the run is staged and each stage's evidence is kept even when the next stage passes.

1. **Discovery.** `CMD_GET_FW_VERSION` broadcast across `0x101`–`0x118`, twice. `CanLink.scan()` already runs two internal rounds — a board that has been sitting on a dead bus parks its transmit path, and the frame that revives the bus is the first request of round one, whose reply is the casualty. Running the whole sweep twice on top of that distinguishes a board that answers one sweep but not the other (a timing problem) from a board that answers neither (off the bus).
2. **Diagnostics, before.** `CMD_GET_CAN_DIAG` on every board, all four reply frames kept as raw bytes, then `CMD_CLEAR_CAN_DIAG` so the soak's counters start from a known baseline.
3. **Soak.** The `Backend` sync master at 150 Hz for 60 s, every discovered board registered and tabled, **none selected and none enabled**.
4. **Diagnostics, after.** The same four frames again — where an RX overflow or a transmit failure would surface, since neither is visible in the reply stream itself.
5. **Port release.** The adapter reopened and closed by a fresh `CanLink`.

## 2. Discovery

Both sweeps returned the identical set of twenty-four boards in 0.58 s each, and every version string arrived complete.

| Range | Count | Variant byte | Firmware |
| --- | --- | --- | --- |
| `0x101`–`0x108` | 8 | `2` (TLE/DVP) — as expected | `tle-fw1.43-13-g1a45afc-dirty` |
| `0x109`–`0x118` | 16 | `0` (7 mm) — as expected | `0.2.1` |

No variant deviations. The check matters beyond bookkeeping: the variant byte selects a board's pressure calibration and is the key the OTA cross-flash guard uses. The TLE version string comes from `git describe`, hence `-13-g1a45afc-dirty`: a build thirteen commits past the `tle-fw1.43` tag from a dirty tree. The strings being byte-identical across the eight boards is the evidence they carry one image (not proof — two dirty trees at one commit describe identically).

## 3. Per-board results

Reply counts are from this script's own tap on the receive thread (the backend credits replies only to *selected* boards, and none were selected). Latency is measured from the cycle thread's pre-write timestamp, so it includes the host serial write and adapter transmission. Idle pressure is quoted on the scale the board's **reported** variant selects.

| Board | Variant | Replies / cycles | Rate | Latency med / p95 / max (ms) | Idle psi | Sync Δ | Cmd Δ | RX ovf | TX fail | Bus err | Invalid | Starv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0x101` | 2 | 9003 / 9003 | 100.00 % | 1.87 / 2.15 / 3.03 | -0.13 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x102` | 2 | 9003 / 9003 | 100.00 % | 1.93 / 2.22 / 3.16 | +0.14 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x103` | 2 | 9003 / 9003 | 100.00 % | 2.00 / 2.28 / 6.50 | +0.07 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x104` | 2 | 9003 / 9003 | 100.00 % | 2.06 / 2.34 / 3.20 | **+4.27** | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x105` | 2 | 9003 / 9003 | 100.00 % | 2.11 / 2.39 / 3.20 | +0.09 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x106` | 2 | 9003 / 9003 | 100.00 % | 2.17 / 2.44 / 5.95 | +0.14 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x107` | 2 | 9003 / 9003 | 100.00 % | 2.21 / 2.50 / 3.30 | +0.07 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x108` | 2 | 9003 / 9003 | 100.00 % | 2.26 / 2.55 / 3.43 | +0.17 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x109` | 0 | 9003 / 9003 | 100.00 % | 2.34 / 2.62 / 3.45 | -0.29 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x10A` | 0 | 9003 / 9003 | 100.00 % | 2.40 / 2.69 / 3.58 | -0.43 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x10B` | 0 | 9003 / 9003 | 100.00 % | 2.47 / 2.76 / 3.58 | -0.01 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x10C` | 0 | 9003 / 9003 | 100.00 % | 2.55 / 2.83 / 3.66 | -0.95 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x10D` | 0 | 9003 / 9003 | 100.00 % | 2.62 / 2.89 / 3.68 | -0.13 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x10E` | 0 | 9003 / 9003 | 100.00 % | 2.71 / 2.97 / 4.15 | -0.54 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x10F` | 0 | 9003 / 9003 | 100.00 % | 2.79 / 3.08 / 4.16 | -0.26 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x110` | 0 | 9003 / 9003 | 100.00 % | 2.85 / 3.20 / 4.17 | -0.15 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x111` | 0 | 9003 / 9003 | 100.00 % | 2.92 / 3.32 / 4.17 | -0.54 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x112` | 0 | 9003 / 9003 | 100.00 % | 2.99 / 3.39 / 4.17 | -0.72 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x113` | 0 | 9003 / 9003 | 100.00 % | 3.06 / 3.47 / 4.18 | -0.51 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x114` | 0 | 9003 / 9003 | 100.00 % | 3.13 / 3.51 / 4.18 | -0.54 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x115` | 0 | 9003 / 9003 | 100.00 % | 3.18 / 3.54 / 4.58 | -0.47 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x116` | 0 | 9003 / 9003 | 100.00 % | 3.24 / 3.56 / 4.60 | -0.52 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x117` | 0 | 9003 / 9003 | 100.00 % | 3.29 / 3.58 / 4.60 | -0.56 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |
| `0x118` | 0 | 9003 / 9003 | 100.00 % | 3.35 / 3.62 / 4.61 | -0.11 | +9005 | +9005 | 0 | 0 | 0 | 0 | 0 |

Zero misses and zero duplicate replies out of 216 072 expected replies. The status flag byte was `0x04` (`COMMAND_SEEN`) on every reply and never `ENABLED` or `ERROR`.

## 4. Cycle and timing

| Quantity | Result | Reference / condition |
| --- | --- | --- |
| Cycles | 9003 in 60.027 s = **149.98 Hz** | requested 150 Hz |
| Unanswered sync edges | **0** | every edge drew ≥1 reply |
| Host cycle jitter p95 / max | **0.0016 ms / 0.108 ms** | report's headless 0.001 / 0.016 ms |
| Reply latency median | **1.87 ms at `0x101` → 3.35 ms at `0x118`** | reference 2.04 → 3.49 ms |
| Latency slope | **64.3 µs per ID step** | reference ≈65 µs |
| Extremes | 1.55 ms min, 6.50 ms max (one sample, `0x103`) | did not become a miss |
| Predicted bus load | 36.7–44.3 % at 1 Mbit/s | `proto.bus_load(8, 16)` |

The latency ordering by node ID is CAN arbitration, not a per-board property. The single 6.50 ms outlier at `0x103` (1 of 9003, absent from the drained comparison run) can likely be attributed to a host-side scheduling hiccup on the receive thread rather than to anything on the wire.

## 5. What went on the wire

Eight runtime-table frames per cycle (`0x091`–`0x098`, `0x80` marker, all-slots mask `0x07`, every command word's enable nibble zero; targets = the zero-psi counts, 944 TLE / 754 legacy, the boundary falling inside frame `0x093` exactly where the variants change) plus the DLC-0 sync on `0x090`. The only other traffic: sixteen frames of TLE native telemetry (`0x21`–`0x24`) in the first fraction of a second, before the 200 ms sync-active suppression engaged — and they show every TLE board in Mission Mode, FAULT pin healthy, zero output, no error.

## 6. Diagnostics, before and after

After the soak: **0 RX overflows, 0 TX failures, 0 bus errors, 0 invalid frames, 0 starvation events on every board**, and every board's sync and command counters advanced by an identical **+9005** (9003 cycles + the in-flight cycle + `stop_cycle()`'s disabling table).

Two findings from the before/after comparison:

- **`CMD_CLEAR_CAN_DIAG` does not reset the sync/command counters** — they run free from boot. Acceptance checks must use the *difference*, never "non-zero".
- **The sixteen legacy boards carried a starvation history** (`last_error_reason=32`, `warning_count=13`, `starvation_count=13`) accumulated *between* runs of this afternoon (no sync master present), zeroed by the clear and staying zero through the soak. The TLE boards report zero because their firmware does not keep that field.

## 7. A host-side artefact, measured and dismissed

199 722 of 216 106 received frames were "evicted from the host RX queue" — not a bus loss: nothing consumes `CanLink.frames` during a cycle (the backend reads its tap), so the 16 384-slot queue saturates and evicts oldest-first. A 30 s `--drain-rx` comparison run put per-board median latencies within 0.063 ms of the 60 s run at all 24 IDs, so the eviction path is not measurably on the latency budget at 3600 frames/s.

## 8. Acceptance — 15/15 PASS

Discovery 24/24 twice · versions complete · variants correct · diag 4×24 before and after · 9003 cycles at 150.0 Hz · 0 unanswered edges · worst reply rate 100.00 % · no error flag · no enable bit on any of 216 072 replies · 0 RX overflows · 0 TX failures · sync counters +9005 identical · port free after the run (verified from a separate process).

## 9. Open items

1. **`0x104` reads +4.27 psi at rest** (vs +0.07…+0.17 on the other TLE boards); confirmed by both the compact status (1202 counts) and its own `0x21` telemetry (`pressure_raw` 19 261). Sensor/front-end offset or genuinely trapped pressure — venting the actuator and re-reading separates them. Settle before closed-loop work.
2. **Legacy boards read −0.95…−0.01 psi at rest** on the nominal transfer function; load `legacy_host/calibration.json` for quantitative legacy pressures.
3. **The TLE boards run a dirty build** (`tle-fw1.43-13-g1a45afc-dirty`) — not reproducible from the tag alone. This workspace's repo is tagged `tle-fw1.46` so future builds report a meaningful string.
4. **Latency under host load is unmeasured** — this run was headless; the GUI-plotting case is a separate measurement (reference: jitter rises to 0.009/5.0 ms with plotting).

## 10. Reproducing this

```
WS/.venv/Scripts/python.exe hw_tests/can_bringup.py --soak-seconds 60
WS/.venv/Scripts/python.exe hw_tests/can_bringup.py --soak-seconds 30 --drain-rx
```

Exits 0 only when every check passes; emits no OTA command, no set-ID command, and asserts the absence of the enable bit in the frames it is about to send before starting the cycle.
