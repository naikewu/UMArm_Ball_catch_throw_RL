# Collection schema — what a CAN-arm recording must contain

Summarised from the legacy `real_system_data_collection/` format, which is what
`legacy_host/host/pc_backend/src/main.cpp` writes when it is sent
`start_data_collection` (`legacy_host/docs/mpc_integration.md:51`). The flagship
legacy session is 540 301 samples over 3599.998 s at 150.00008 Hz in 61 chunks —
so the format is known to survive an hour at rate, which is the property that
matters.

**Adopt it as-is, with one addition** (`board_type`, below).

Three file kinds per session, under `session_<YYYYmmdd_HHMMSS>/`.

---

## `metadata.json` — written once, at start

| field | note |
| --- | --- |
| `schema_version` | `1` in the legacy sessions; bump it for the `board_type` addition |
| `session_tag`, `created_at_unix_s`, `duration_s` | |
| `sample_rate_hz` | 150 |
| `checkpoint_s`, `deflate_s` | chunk cadence, and the deflate tail |
| `sync_semantics` | prose: what a cycle's timestamps mean relative to the sync edge |
| `state_fields` | `["q", "qdot", "pressure_adc"]` |
| `state_dimension` | 48 = 12 + 12 + 24 |
| `input_fields`, `input_dimension` | `["target_adc"]`, 24 |
| `pressure_units` | `"adc_counts"` — **raw**, always |
| `selected_ids` | 24 hex strings, `0x101`..`0x118` |
| `adc_ranges` | per-id `[min, max]`, a snapshot of the calibration **in force at record time** |
| `actuator_pairs` | the 12 antagonistic pairs — **copy the table, do not derive it** |
| `single_actuator_limit_psi`, `pair_sum_limit_psi`, `slew_psi_per_s_reference`, `adc_range_reference_psi` | the target-generation envelope |
| `can_protocol`, `can_order`, `rx_window_frac` | |
| `mocap_live`, `mocap_sim`, `mocap_server`, `mocap_local`, `mocap_rigid_ids` | |
| **`board_type`** | **NEW — see below** |

### The addition: `board_type`

Twenty-four entries, aligned with `selected_ids`, carrying the **variant byte
each board reported** during the scan:

```
0x00  7 mm          (legacy)
0x01  DT/big-valve  (legacy)
0x02  TLE/DVP       (the top eight)
```

Why it is not optional. This arm is two plants on one bus: the top eight are
TLE92464 proportional valves with a 500 ms sync-loss failsafe, the bottom sixteen
are 7 mm solenoid pairs with **no** link-loss timeout at all. They have different
valve physics, different calibrations (TLE: 943.75 zero-counts, 60.78125
counts/psi; legacy: 754.4 and 56.14) and different behaviour when the host
stalls. A recording without `board_type` cannot be split by population after the
fact, and **the split is not recoverable from the id range** — a TLE board
legitimately sat at 0x114 during the 2026-08 bench session, and reading it on the
7 mm scale is a 10 % error at the top of the range.

Take the value from what the board **answered** to `CMD_GET_FW_VERSION` (byte 3
of `MSG_FW_VERSION`), the same source `tlelib.proto.NodeCal.for_variant` uses,
and record it once per session rather than per cycle: it cannot change without a
reflash, and a reflash mid-session is a different session.

### The antagonistic pairs, verbatim

Not a naive `i, i+4` mapping. From `main.cpp:929-936`, mirrored into the
metadata:

```
(0x102,0x106) (0x104,0x108) (0x103,0x107) (0x105,0x101)
(0x10A,0x10C) (0x109,0x10B) (0x110,0x10E) (0x10D,0x10F)
(0x114,0x112) (0x111,0x113) (0x116,0x118) (0x115,0x117)
```

---

## `manifest.json` — the chunk index

`total_samples`, `checkpoint_count`, `active`, and `chunks[]` of
`{path, reason, samples, start_cycle, end_cycle, start_time_s, end_time_s}` with
`reason ∈ {checkpoint, complete, stop}`. A session whose `active` is still true
was interrupted; its last chunk is the short one.

---

## `samples_chunk_NNNN_<reason>_<ts>.jsonl` — one object per 150 Hz cycle

About 4.1 kB per line, 9000 lines per 60 s chunk.

```jsonc
{
  "schema_version": 1,
  "cycle": 12345,
  "phase": "collecting",            // or "deflating"
  "timestamp_unix_s": 1747953612.34,
  "timestamp_s": 82.3,
  "cycle_start_time_s": 82.2967,
  "can_sync_time_s": 82.2981,       // the sync edge itself: align rollouts to THIS
  "jitter_ms": 0.012,

  "ids": ["0x101", "...24 entries..."],
  "board_type": [2, 2, 2, 2, 2, 2, 2, 2, 0, "...24 entries..."],   // NEW

  "robot_state": {
    "q":            [ /* 12, radians */ ],
    "qdot":         [ /* 12, rad/s   */ ],
    "pressure_adc": [ /* 24, RAW ADC counts */ ]
  },
  "input": { "target_adc": [ /* 24, RAW ADC counts */ ] },

  "actuator_status":            [ /* 24 */ ],
  "actuator_stale":             [ /* 24 */ ],
  "control_next_sync":          [ /* 24 */ ],
  "actuator_missed_total":      [ /* 24 */ ],
  "actuator_reply_latency_ms":  [ /* 24 */ ],

  "cycle_responded": 24, "cycle_expected": 24, "total_missed": 0,
  "unexpected_replies": 0, "duplicate_replies": 0,

  "joint_current_valid": true,
  "joint_current_extrapolated": false,
  "joint_current_source_error_ms": 0.4,
  "joint_current_extrapolation_ms": 0.0,

  "mocap": {
    "valid": true, "stale": false, "frame": 998877,
    "timestamp_s": 82.2975, "raw_timestamp_s": 82.2971, "received_s": 82.2979,
    "latency_ms": 0.8, "age_ms": 0.6, "timestamp_offset_ms": -0.4,
    "frame_rate_hz": 120.0, "frame_drop_count": 0,
    "clock_sample_count": 512, "clock_update_count": 33,
    "body_count": 6, "body_ids": [ /* per body */ ],
    "body_centers":     [ /* 18 = 6 x 3 */ ],
    "body_rotations":   [ /* 54 = 6 x 9 */ ],
    "body_quaternions": [ /* 24 = 6 x 4 */ ],
    "body_points": null, "body_poses": null
  }
}
```

### Reading it

* **Everything is raw.** ADC counts, radians. Convert to psi with the session's
  own `adc_ranges` and the board's own variant calibration, at read time. A
  recording that stored psi would be invalidated by every recalibration.
* **`can_sync_time_s` is the alignment anchor**, not `timestamp_s`. The sync edge
  is the instant every board latches its filtered pressure and promotes its
  staged target, which makes it the one moment the whole arm agrees on.
* **`actuator_reply_latency_ms` is per board and it is supposed to spread.** On
  the real 24-board bus, replies run 2.04 ms at 0x101 to 3.49 ms at 0x118, about
  65 µs per id step, from CAN arbitration. A flat column means a one-board bench,
  not a healthy bus.
* **`mocap.stale` and joint staleness are different questions.** The receiver
  distinguishes `stale` (no frame at all) from `q_stale` (frames arriving, none
  converting), and it is the second that invalidates a sample's `q` while
  `mocap.valid` can still read true. Whichever field the collector writes, record
  both, and prefer `q_stale` when deciding whether a sample's joints are usable.
* **A single untracked plate is invisible.** Only two *adjacent* plates
  collapsing is caught; one bad plate leaves both neighbouring differences
  non-degenerate, the frame converts, and the affected joints quietly encode the
  direction to the mocap volume's origin. Nothing in the file marks it. If the
  fit later shows one joint behaving unlike its neighbours, this is the first
  thing to rule out.

### `timing_summary_<phase>.json`

Written per phase alongside the chunks: `dt_avg_ms`, `dt_median`, `dt_min`,
`dt_max`, `max_interval_delay_ms`, `missing_cycles_from_rows`. The legacy hour
run reports 6.66666 / 6.666 / 4.306 / 25.069 ms and **0** missing cycles. Treat
`missing_cycles_from_rows > 0` as a reason to re-run rather than to interpolate.

---

## Predecessor format, for continuity

`.npz` with `(n_windows, window_samples, dim)` arrays and pressures in **Pa**,
fully specified at
`legacy_implementation_notes/koopman_data_collection_handoff_20260522.md:304-357`
in the legacy repo. Older Koopman data is in that shape. The JSONL schema is a
deliberate, better-synchronised replacement; converting forward is preferable to
collecting in the old shape.
