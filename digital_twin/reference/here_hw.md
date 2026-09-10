# Hardware interfaces of `UMArm_koopman_compliance_control_espproject` — the exact Python API to drive 24 boards and read mocap

All paths absolute. Line numbers are from the files as they stand on disk (2026-09-10). Nothing was opened: no serial port, no socket, no file written.

---

## 0. The one-screen answer (minimal working sketch)

```python
# --- path setup, copied from hw_tests/canarm_drive_campaign.py:52-61 ---------
import os, sys, time
_WS = r"C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject"
for _p in (_WS, os.path.join(_WS, "TLE_PCB"),
           os.path.join(_WS, "UMArm_MOCAP", "natnet_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import bench_env
from tlelib import proto as P                     # wire formats, no I/O
from tlelib.backend import Backend                # the 150 Hz sync master
from UMArm_MOCAP.canarm_mocap import (CANARM_N_BODIES, CANARM_RB_ID_BASE,
                                      CanArmMarkerMocap, load_canarm_locks)
from UMArm_MOCAP import canarm_frames as CF
from UMArm_KINEMATICS.fkine import fkine, ujoint_centres
from UMArm_KINEMATICS.canarm_params import CANARM_PARAMS
from UMArm_KINEMATICS.canarm_actuators import joint_pairs, base_to_joint

# --- mocap first: never drive blind (campaign:263-269) -----------------------
locks = load_canarm_locks()                       # raises if not minted this session
rx = CanArmMarkerMocap(locks,                     # phis + order default to MEASURED
                       server_ip=bench_env.MOCAP_SERVER_IP,
                       client_ip=bench_env.MOCAP_CLIENT_IP,
                       use_multicast=True,
                       ring_capacity=4000, marker_ring_capacity=4000)
rx.start()                                        # opens the NatNet socket
deadline = time.monotonic() + 5.0
while rx.get_state().frames < 10 and time.monotonic() < deadline:
    time.sleep(0.1)
assert rx.get_state().frames >= 10, "no mocap frames; refuse to drive"

# --- CAN: connect -> scan -> select -> cycle ---------------------------------
port = bench_env.resolve_can_port(None)           # enumerates only; opens nothing
be = Backend(port=port, bitrate=bench_env.BITRATE)   # bitrate 1_000_000
be.open()                                         # opens COM, starts rx thread
try:
    nodes = be.scan()                             # 2 rounds of CMD_GET_FW_VERSION
    present = sorted(b for b, n in nodes.items() if n.present)
    assert len(present) == 24, present

    be.select(list(P.ALL_IDS))                    # REQUIRED: enable is gated on this
    for b in P.ALL_IDS:                           # stage a safe table before the edge
        be.set_target(b, 0.0)
        be.set_enabled(b, False)
    be.start_cycle()                              # spawns thread "sync-master"
    time.sleep(0.5)

    # --- idle hold: every board ENABLED at 0.5 psi (0x110 leaks) -------------
    for b in P.ALL_IDS:
        be.set_target(b, 0.5)
        be.set_enabled(b, True)
    time.sleep(2.5)

    # --- command one actuator ------------------------------------------------
    be.set_target(0x101, 12.0)                    # psi, float; clamped to 12 bits
    time.sleep(2.5)                               # settle (campaign default)

    # --- read back -----------------------------------------------------------
    snap = be.snapshot_nodes()                    # {base: NodeState}, deep copies
    print(snap[0x101].pressure_psi,               # measured, psi
          snap[0x101].pressure_counts,            # measured, 12-bit counts
          snap[0x101].target_psi,                 # commanded, psi
          snap[0x101].flags,                      # STATUS_* bits
          snap[0x101].reply_latency_ms,
          snap[0x101].consecutive_misses)

    q = rx.wait_fresh(timeout=0.5)                # (12,) rad, or None on timeout
    poses = rx.get_marker_poses()                 # (7,4,4) inferred BRACKET frames
    body = CF.body_frames(poses)                  # (6,4,4) BODY frames
    centres = CF.fk_centres_world(q, body[0], CANARM_PARAMS)   # (6,3) m, mocap frame
    resid_m = CF.fk_residual_m(poses)             # (6,) m; row 0 is identically 0

finally:
    be.stop_cycle()      # sends one final table, every enable bit CLEAR
    be.close()           # removes tap, sends slcan "C", closes the COM port
    rx.stop()            # shuts down the NatNet SDK threads
```

---

## 1. `TLE_PCB/tlelib/` — the bus stack

Layering, stated at `C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\tlelib\__init__.py:5-14`:

```
proto     wire formats and addressing, no I/O
canlink   the slcan adapter, a receive thread, node discovery
usblink   one board over its USB-Serial/JTAG port, same frames as CAN
native    the TLE board's own command set (re-exported from vema_proto)
ota       broadcast firmware update
usbflash  esptool over USB
backend   the 150 Hz sync master and the multi-board registry
wsenv     optional workspace defaults from <workspace>/bench_env.py
```

Public names re-exported at `__init__.py:18-23`: `Backend, BroadcastOta, CanLink, NodeState, OtaCancelled, SlcanError, UsbFlasher, UsbLink, find_build, load_image, native, proto`.

### 1.1 Addressing and the wire (`TLE_PCB/tlelib/proto.py`)

| constant | value | file:line |
|---|---|---|
| `ID_BROADCAST` | `0x090` — DLC 0 = sync edge; DLC > 0 = OTA data | proto.py:21 |
| `ID_RUNTIME_TABLE_BASE` | `0x091` (`0x091..0x098`, 3 actuators per frame) | proto.py:22 |
| `ACTUATOR_FIRST` / `ACTUATOR_LAST` / `ACTUATOR_COUNT` | `0x101` / `0x118` / `24` | proto.py:24-26 |
| `TLE_IDS` | `range(0x101, 0x109)` (8 DVP boards) | proto.py:30 |
| `SEVEN_MM_IDS` | `range(0x109, 0x119)` (16 legacy) | proto.py:31 |
| `ALL_IDS` | `range(0x101, 0x119)` | proto.py:32 |
| `OFFSET_HOST_CTRL / HOST_DATA / STATUS / EXTENDED` | `0x100 / 0x200 / 0x300 / 0x400` | proto.py:34-37 |
| `RUNTIME_TABLE_SLOTS / MARKER / SLOTMASK` | `3 / 0x80 / 0x1F` | proto.py:39-41 |
| `CYCLE_HZ` | `150.0` | proto.py:91 |

`P.ids(base) -> NodeIds` (proto.py:126) gives `.ctrl = base+0x100`, `.data = base+0x200`, `.status = base+0x300`, `.extended = base+0x400`, `.slot = base-0x101`, `.table_id = 0x091 + slot//3`.

**Trap:** compact status replies come back on the **bare base id** (`0x101..0x118`), *not* on `base+0x300`. `parse_compact_status` (proto.py:241-245) accepts only `len(data) == 2 and ACTUATOR_FIRST <= can_id <= ACTUATOR_LAST`. `base+0x300` carries only the host-control replies (`MSG_ACK/NACK/CAN_DIAG0..3/OTA_STATUS/FW_VERSION`).

Host-control commands (sent to `base+0x100`, one byte, `build_simple_command`): `CMD_START 0x01`, `CMD_END 0x02`, `CMD_SET_ID 0x05`, `CMD_GET_CAN_DIAG 0x06`, `CMD_CLEAR_CAN_DIAG 0x07`, `CMD_GET_OTA_STATUS 0x08`, `CMD_GET_FW_VERSION 0x09` (proto.py:46-52). Replies: `MSG_ACK 0xAA`, `MSG_NACK 0xFF`, `MSG_CAN_DIAG0..3 0xD0..0xD3`, `MSG_OTA_STATUS 0xD4`, `MSG_FW_VERSION 0xD5` (proto.py:54-61).

Compact command/status bit layout (proto.py:70-76):
```python
PRESSURE_MASK = 0x0FFF     # bits 0..11 = 12-bit counts
FLAGS_MASK    = 0x000F     # bits 12..15
CONTROL_ENABLE     = 0x01  # host -> board
STATUS_ENABLED     = 0x01  # board -> host
STATUS_OTA_ACTIVE  = 0x02
STATUS_COMMAND_SEEN= 0x04
STATUS_ERROR       = 0x08
```

Variant byte inside `MSG_FW_VERSION` (proto.py:81-84):
```python
VARIANT_7MM = 0x00
VARIANT_DT  = 0x01
VARIANT_TLE_DVP = 0x02
VARIANT_NAMES = {0: "7mm", 1: "DT", 2: "TLE/DVP"}
```

`build_runtime_table(targets: dict[int, tuple[int, bool]]) -> list[tuple[int, bytes]]` (proto.py:251-283). Packs base→(counts, enable) into 8-byte group frames; byte 0 = `0x80 | start_slot`, byte 1 = presence mask over the 3 slots, bytes 2/4/6 = little-endian `counts | (flags << 12)`. `build_sync() -> (0x090, b"")` (proto.py:286-288).

`build_set_id(new_base)` is **big-endian** — the lone exception in the protocol (proto.py:294-301).

### 1.2 The units, layer by layer — the single most important table

| layer | unit | conversion | file:line |
|---|---|---|---|
| CAN wire (compact command & status) | 12-bit **counts**, 0..4095 | — | proto.py:70 |
| TLE/DVP board internal sensor | 16-bit **raw**, `raw = counts << 4` | `TLE_RAW_SHIFT = 4` | proto.py:150, 200-206 |
| TLE psi ↔ counts | `zero = 943.75`, `60.78125 counts/psi` (≈ **0.016452 psi/count**) | `TLE_RAW_0_PSI=15100.0`, `TLE_COUNTS_PER_PSI_RAW=972.5` | proto.py:148-152 |
| legacy 7 mm psi ↔ counts | `zero = 754.4`, `56.14 counts/psi` (≈ **0.017813 psi/count**) | `LEGACY_ZERO_COUNTS`, `LEGACY_COUNTS_PER_PSI` | proto.py:154-155 |
| `Backend` API surface | **psi, float** | `NodeCal.psi_to_counts` / `counts_to_psi` | proto.py:193-197 |
| extended (`base+0x400`, TLE only) telemetry | 16-bit **raw** | `native.raw_to_psi(raw) = (raw-15100)/972.5` | vema_proto.py:38-41, 165-171 |
| valve coil current | **code** 0..116, `1.5748 mA/code` | `HOST_MAX_CODE = 120-4 = 116`, `MA_PER_CODE = 200.0/127.0` | vema_proto.py:93-96, 106 |

```python
# proto.py:193-197
def psi_to_counts(self, psi: float) -> int:
    return max(0, min(PRESSURE_MASK, int(round(self.zero_counts + psi * self.counts_per_psi))))

def counts_to_psi(self, counts: float) -> float:
    return (counts - self.zero_counts) / self.counts_per_psi
```

**Calibration is chosen from the reported variant byte, not from the id range** — `NodeCal.for_variant(variant, base)` (proto.py:177-191). The docstring notes the bench board once sat at `0x114` while being a TLE board, and getting this wrong is "a 10 % error in psi at the top of the range". `for_base` (proto.py:170-175) is the id-range guess and is the last resort only.

### 1.3 `CanLink` — the slcan transport (`TLE_PCB/tlelib/canlink.py`)

```python
class CanLink:                                                  # canlink.py:44
    def __init__(self, port: str = DEFAULT_PORT, bitrate: int = 1_000_000,
                 rx_queue_depth: int = 16384) -> None: ...       # :47
```
- `BITRATES = {1_000_000: "S8", 500_000: "S6"}` (canlink.py:32). **S7 is never emitted**: the CANable 2.0 firmware mis-prescales it to 1.43 Mbit/s (canlink.py:11-13).
- `DEFAULT_PORT = wsenv.can_port("COM58")` (canlink.py:37) → resolves through `bench_env.resolve_can_port()`.
- `open()` (canlink.py:66-83): `serial.Serial(port, 115200, timeout=0, write_timeout=1.0)`, 0.3 s settle, `"C"`, reset input, `"V"` query, bitrate letter, `"O"` (**normal mode, ACKs enabled — never listen-only**), then starts the daemon thread `"canlink-rx"`.
- `send(can_id, data)` / `send_batch(frames)` (canlink.py:131-152). `send_batch` concatenates into one `write()`; pacing is the caller's job.
- `add_tap(fn)` / `remove_tap(fn)` — `fn(timestamp, can_id, data)` called **on the reader thread** (canlink.py:155-167).
- `scan(bases=None, settle=0.5, rounds=2) -> dict[int, FirmwareVersion]` (canlink.py:293-332). Sends `CMD_GET_FW_VERSION` to every candidate's `.ctrl` with `time.sleep(0.003)` between them, then listens for `settle`. Two rounds by default because "the first request of the first round … whose reply can be the casualty".
- `can_diag(base, timeout=0.5) -> CanDiag | None` (canlink.py:338-348), `ota_status`, `set_node_id`, `request`, `wait_ack`, `collect`, `drain`.
- `list_serial_ports() -> list[tuple[str, str]]` (canlink.py:369-371).

Threading: reads and writes run on separate threads **without a shared lock** (writes take `_write_lock`, reads do not) — deliberate, because a lock held across a blocking read would put the read timeout into the 150 Hz jitter budget (canlink.py:15-19). The reader polls `in_waiting` and sleeps 200 µs when idle (canlink.py:189-191).

### 1.4 `Backend` — the 150 Hz sync master (`TLE_PCB/tlelib/backend.py`)

```python
class Backend:                                                   # backend.py:100
    def __init__(self, port: str = DEFAULT_CAN_PORT, bitrate: int = 1_000_000,
                 cycle_hz: float = P.CYCLE_HZ, log=None) -> None  # :103-104
```

Module constants (backend.py:37-47):
```python
HISTORY_SECONDS = 60.0
RX_WINDOW_FRAC  = 0.82     # 0.82 * 6.667 ms = 5.47 ms of listening per cycle
EXTENDED_POLL_HZ = 4.0     # per selected TLE board
NODE_TIMEOUT_S  = 0.5      # no reply in this long => "missing"
```

**Lifecycle**
| method | line | effect |
|---|---|---|
| `open()` | 131-137 | `hires_clock()` (1 ms Windows timer), `link.open()`, `link.add_tap(self._on_frame)` |
| `close()` | 138-141 | `stop_cycle()`, `remove_tap`, `link.close()` |
| `__enter__` / `__exit__` | 143-148 | open / close |
| `scan(bases=None)` | 151-169 | `link.scan`, fills registry, sets `node.cal = P.NodeCal.for_variant(...)`, marks non-responders `present=False`. **Returns `snapshot_nodes()` — every known node, present or not.** |
| `select(bases)` | 171-173 | `self.selected = [b for b in bases if b in self.nodes]` |
| `start_cycle()` | 233-239 | daemon thread named `"sync-master"` |
| `stop_cycle()` | 241-260 | clears run flag, joins 1 s, then **`_send_cycle(disable_everything=True)`** |
| `stop_all()` | 201-214 | clears `enabled` on **every** node; if the cycle is *not* running, sends the disabling table + sync once |

**Commands** (all take psi):
```python
be.set_target(base, psi)          # backend.py:176-180
be.set_target_all(psi)            # :182-185  (selected only)
be.set_enabled(base, on)          # :187-192
be.set_enabled_all(on)            # :194-199  (selected only)
be.send_native(base, payload)     # :216-222  -> base+0x400, flushed after the RX window
be.apply_tuning(base, params=None)# :224-227  -> N.tuning_frames(params), 7 frames
be.request_extended(base)         # :229-230
```

**The enable bit has two gates.** `_build_targets` (backend.py:285-303):
```python
enable = node.enabled and not disable_everything and base in self.selected
```
so `set_enabled(base, True)` on a board that is not in `selected` does nothing on the wire. The docstring explains why every *known* board is tabled every cycle: a board keeps its last control byte, so de-selecting a running board would leave it regulating with nothing able to reach it, and the sync-loss failsafe cannot help because the master is still sending syncs.

**Read-back**
```python
be.snapshot_nodes() -> dict[int, NodeState]           # :411-415, deep copies, sorted
be.history(base, max_points=None) -> [(t, meas_psi, target_psi)]  # :417-432
be.missing_nodes() -> list[int]                        # :434-438
be.stats -> CycleStats(cycles, period_ms, jitter_ms_p95, jitter_ms_max,
                       late_cycles, replies, misses)   # :89-97
be.log_lines -> deque[str] (maxlen 500)                # :110
```
`history(max_points=...)` decimates *under the lock*; copying the full 60 s × 9000-sample deque per board cost a measured 2.8 Hz on the 24-board bus (backend.py:423-425). A display should ask for ~900.

`NodeState` (backend.py:50-86) fields: `base, kind, version, cal, present, enabled, target_psi, pressure_counts, pressure_psi, flags, replies, misses, consecutive_misses, last_reply_t, reply_latency_ms, extended (dict), extended_t`; properties `is_tle` (`kind == "TLE/DVP"`), `error` (`flags & STATUS_ERROR`), `target_counts`.

**The cycle**, `_cycle_loop` (backend.py:338-375), one iteration:
1. `sleep_until(next_tick)`; record drift into `_jitter`; a drift > half a period counts a `late_cycle`.
2. Clear `_cycle_replies`, stamp `_cycle_t0 = t_sync`.
3. `_send_cycle()` → `P.build_runtime_table(...)` + `P.build_sync()` appended, all in **one `link.send_batch`** so the adapter transmits the sync immediately after its targets (backend.py:305-311; rationale at backend.py:13-17).
4. `sleep_until(t_sync + 0.82*period)`; snapshot the replies the tap collected.
5. `_publish` — credits replies **only to boards in `self.selected`** (backend.py:377-402), converts counts→psi with that node's `cal`, appends to the 60 s history.
6. `_flush_native()` then `_poll_extended()` — extended traffic never touches the real-time path.
7. On a long stall, `next_tick` is reset to `now + period` rather than bursting catch-up cycles.

A transmit exception inside the loop logs, **clears `_running` and breaks** (backend.py:357-360) — which is exactly why `stop_cycle` is gated on `self._thread is not None`, not on the run flag (backend.py:242-248).

**Threads in play:** `canlink-rx` (daemon, parses slcan, fires taps), `sync-master` (daemon, the cycle), plus the caller's thread. `Backend._lock` is an `RLock` guarding the node registry and history; `Backend._cycle_lock` is a plain `Lock` guarding `_cycle_replies` / `_pending_native`.

**Clean shutdown, in order:** `stop_cycle()` → `close()`. `stop_cycle` sends one final all-clear table even if the transmit thread already died; if that final send throws it logs `"[ERROR] sync master stopped but the safe-disable did not reach the bus … Boards may still be driving"` (backend.py:258-260).

### 1.5 `native` / `vema_proto` — the TLE board's own command set

`TLE_PCB/tlelib/native.py` is a thin re-export of `TLE_PCB/tlelib/vema_proto.py` (native.py:25). Over CAN these payloads go to `base + 0x400`, which the legacy boards ignore (native.py:13-14).

Telemetry types decoded into `NodeState.extended` by `Backend._on_frame` (backend.py:277-283): `TLM_PRESSURE_CONTROL 0x21`, `TLM_PRESSURE_DVP 0x22`, `TLM_DVP_TIMING 0x23`, `TLM_DVP_DEBUG 0x24`.

`decode_into(data, state)` writes these keys (vema_proto.py:318-345):
- from `0x21`: `state`, `action`, `target_raw`, `pressure_raw` (all 16-bit raw / enum)
- from `0x22`: `permille`, `code` (valve current code), `channel`, `valve_mode`, `armed`, `fault_pin`, `error_code`
- from `0x23`: `late_updates`, `max_service_us`
- from `0x24`: `i_pm`, `d_pm` (integral / derivative, per-mille)

Enums: `PRESSURE_STATES = {0:"idle",1:"running",2:"settling",3:"fault",4:"config error",5:"stale ADC"}`, `PRESSURE_ACTIONS = {0:"-",1:"inlet",2:"outlet",3:"coast/deadband",4:"settling"}`, `VALVE_MODES = {0:"bang-bang",1:"proportional"}` (vema_proto.py:113-115).

`native.tuning_frames(params=None) -> list[bytes]` (native.py:80-101) emits, **in this order** (order matters — 0x24 carries legacy 8-bit gains that 0x27 then overrides): `build_dvp_config`, `build_gains16`, `build_shaping`, `build_transient`, `build_flowshape`, `build_dither`, `build_adc_filter`. The tuned defaults live in `vema_proto.DEFAULTS` (vema_proto.py:141-160): `kp=2048, ki=2500, kd=2500, open=51, max=110, slew=4, outdb=0, dither_n=4, code_jump=2, dlp=3, deadtime=10, dfade_psi=0.5, vent_pm=20, dzone_up=80, dzone_down=30, outlet_open=57, outlet_max=96, ff_scale=30, adc_shift=3, fs=1, inlet_far_max=95, fs_seed=17, fs_rate=2, dither_hz=150.0, dither_ma=0.0, dither_ch=1`.

---

## 2. The operator GUI on top

### 2.1 `TLE_PCB/VEMA_TLE_controller.py` — `ControllerApp`

Constants (lines 58-96): `REFRESH_MS = 100`, `PLOT_MS = 500`, `PLOT_WINDOW_S = 30.0`, `BITRATES = {"1 Mbit/s": 1_000_000, "500 kbit/s": 500_000}`, `TARGET_MAX_PSI = 40.0`, `TARGET_STEP_PSI = 0.05`, `WHEEL_STEP_PSI = 0.25`, `SETTLED_PSI = 0.5`.

- **Connect** (`toggle_connect`, :407-426): parses `port = label.split(" ")[0]` from the combobox, builds `Backend(port, BITRATES[...], log=self.log)`, `open()`, then **calls `self.scan()` immediately**.
- **Scan** (`scan`, :454-462): refuses while the cycle runs (`"stop the cycle before scanning"`), calls `self.backend.scan()`, `_render_nodes(nodes)`, then `backend.select([...checked...])`.
- **`_render_nodes`** (:492-545): iterates `nodes.items()`, **skips `not node.present`**, assigns a stable per-board colour, and renders `node.kind` verbatim into a `width=8` label — that is where the **variant/board_type byte becomes visible** to the operator. There is no separate `board_type` field: the type is `NodeState.kind`, which is `FirmwareVersion.variant_name` from the `MSG_FW_VERSION` frame's byte 3 (`proto.py:363`, `FirmwareVersion.feed`), mapped through `VARIANT_NAMES`.
- **Cycle** (`toggle_cycle`, :464-477): refuses with no boards checked, else `backend.select(chosen)` then `backend.start_cycle()`; toggling off calls `backend.stop_cycle()`.
- **STOP ALL** (`stop_all`, :479-490): `backend.stop_all()`, then zeroes the group slider and **every** bar including unselected boards.
- **Enable** (`enable_selected(on)`, :556-561): loops the checked boards calling `backend.set_enabled`.
- **Tuning** (`apply_tuning`, :586-598): `backend.apply_tuning(base)` for checked boards where `node.is_tle`; logs "7 mm boards have no tuning surface here".
- `_measured_hz` (:659) reports the rate the cycle is *achieving* from `(monotonic, cycles)` marks, not the requested rate.

### 2.2 `canarm_control_gui.py` — `CanArmControllerApp(VTC.ControllerApp)`

Path bootstrap at lines 74-78 inserts `TLE_PCB` **then** the workspace root. Constants: `MOCAP_MS = 200` (line 87), `VIEWER_JOIN_S = 3.0`, `LOCK_CAPTURE_S = 3.0` (line 99).

What it adds on top of the base class:
- **A mocap strip** with a source selector `off | sim | live`, built by `_build_mocap(kind)` (:459-500). `sim` → `UMArm_MOCAP.sim_stream.CanArmSimStream()` (opens nothing). `live` → tries `load_canarm_locks()`; on `FileNotFoundError/ValueError` it degrades to `CanArmMocap()` and sets `rx.lock_note`, otherwise `CanArmMarkerMocap(locks)`. **Both branches return the receiver, never what `start()` returned** — `MocapRx.start` returns a `NatNetClient`, `CanArmSimStream.start` returns `self`, and returning the client previously broke both `get_state` and `stop` (documented failure, :468-478).
- **Lock plates** (`lock_plates`, :217-274): requires `live` (the sim stream carries no labeled markers), blocks for `LOCK_CAPTURE_S`, calls `marker_mocap.mint_locks(rx, 3.0, x_mode="diagonal45", log=self.log)`, refuses loudly on `report["refusals"]`, writes to `canarm_mocap.DEFAULT_TEMPLATE_PATH`, then **stops and restarts the receiver** so the running class becomes `CanArmMarkerMocap` (a receiver cannot grow locks in place — its solve method is chosen by its class).
- **A kinematics line** (`_kin_line`, :503-547): `CF.fk_residual_m(rx.get_marker_poses())` printed per plate in mm plus RMS over plates 1..5. On the 2026-08-21 calibration it reads ≈ 0.2 / 0.3 / 0.5 / 1.0 / 1.8 mm.
- A spawned MuJoCo room viewer in a separate process. **Closing the viewer never affects the bus** (:49-54) — the reverse of the RS485 GUIs' rule.
- Known cost, recorded in `repo_doc/directory.md:27`: the 24-board pressure plot holds the cycle to ~145 Hz instead of 150.

---

## 3. `hw_tests/canarm_drive_campaign.py` — the template for a collection campaign

Full path: `C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\hw_tests\canarm_drive_campaign.py`.

**Safety constants**
```python
PSI_CEILING = 14.0   # campaign.py:67  — --psi is clamped to this
IDLE_PSI    = 0.5    # campaign.py:77  — the hold for every board not under test
ALL_BASES   = tuple(P.ALL_IDS)   # :81
```
`--psi` default is 12.0 (`:393-394`); `min(float(args.psi), PSI_CEILING)` at `:230`.

**The 0.5 psi idle hold, and why it exists** (`hold_idle_all`, :192-210) — quoted verbatim from the docstring:

> Enabled-at-idle rather than disabled, which is the opposite of the obvious choice and the one the hardware forces. A disabled board stops regulating, and an actuator that stops being regulated keeps whatever air is in it — and **one actuator on this arm leaks from its supply side** (``0x110``, observed drifting from 0.1 to 7.0 psi over the 2026-08-21 sweep whenever it was left disabled). … The setpoint is :data:`IDLE_PSI`, not zero, so the exhaust valve is not held open for the whole campaign

```python
def hold_idle_all(be: Backend) -> None:          # :192
    for base in ALL_BASES:
        be.set_target(base, IDLE_PSI)
        be.set_enabled(base, True)

def relax(be: Backend, seconds: float) -> None:  # :212
    hold_idle_all(be); time.sleep(seconds)

def drive_one(be: Backend, base: int, psi: float) -> None:   # :218
    hold_idle_all(be)
    be.set_target(base, psi)
```
`drive_one` re-asserts the whole idle table before raising one target, so exactly one actuator is above 0.5 psi at any instant. The pair-sum ceiling (operator rule: 30 psi over an antagonistic pair) therefore cannot be approached even in principle (:21-27).

**Bring-up order** (`run`, :229-367):
1. `psi` clamped (:230-232); `--only` parsed as hex bases (:234-236).
2. `port = bench_env.resolve_can_port(args.port)` (:238).
3. **Mocap first** (`:257-269`): `CanArmMocap(server_ip, client_ip, use_multicast=not args.no_multicast, ring_capacity=4000, marker_ring_capacity=4000)`, `rx.start()`, then a 5 s wait for `rx.get_state().frames >= 10`; otherwise `print("campaign: no mocap frames arrived; refusing to drive blind"); return 1`.
4. `Backend(port=port, bitrate=bench_env.BITRATE)`, `be.open()` (:271-272).
5. `found = be.scan()`; `present = [b for b,n in found.items() if n.present]`; **abort with `return 1` if any requested base is missing** (:274-280).
6. `be.select(list(ALL_BASES))`, then `set_target(b, 0.0)` + `set_enabled(b, False)` for all 24, **then** `be.start_cycle()`, `sleep(0.5)`, `hold_idle_all(be)`, `sleep(args.relax)` (:281-288). Note the enable bits are staged *clear* before the first sync edge goes out.

**Phases** (`--phase` = comma list of `rest, sweep, poses, all`, default `all`):
- `rest` (:293-307): `args.rest` s (default 6.0) with every board at 0.5 psi; one `sample_mocap` record `kind="rest"`.
- `sweep` (:310-326): for each base — `drive_one(be, base, psi)`, `hold_and_sample(settle_s=2.5, sample_s=1.0, label=f"drive 0x{base:03X}")` → `kind="single"`, then `relax(be, 2.5)` and a bracketing `hold_and_sample(settle_s=0.4, sample_s=0.5)` → `kind="baseline"`. Every drive is bracketed by a baseline, which is what makes the argmax-|dq| reduction robust to slow drift.
- `poses` (:329-353): `rng = np.random.default_rng(args.seed)` (seed 20260821); for each of `args.poses`, for each antagonistic pair pick **one side** and `p = uniform(0.35, 1.0) * psi`; `hold_idle_all` then set those targets. "only one member of a pair is ever pressurised, so a wrong table costs coverage, never safety" (:377-378). Pair table from `--pair-table` JSON, else `UMArm_KINEMATICS.canarm_actuators.joint_pairs(allow_legacy=True)` (:370-386).

**Every exit path clears the enable bits.** There are exactly two `finally` blocks (:354-359):
```python
        finally:
            be.stop_cycle()
            be.close()
            print("campaign: bus released, every enable bit clear")
    finally:
        rx.stop()
```
plus the `KeyboardInterrupt` handler in `main` (:411-415) which returns 130 and prints `"campaign: interrupted; the bus was released by the finally path"`. The inner `finally` wraps everything from the `Backend` construction onward, so a raise anywhere in scan/sweep/poses still runs `stop_cycle()`, and `stop_cycle` itself sends the all-clear table (`backend.py:256`). `be.close()` then calls `stop_cycle()` a second time — harmless, gated on `_thread is None` at `backend.py:249`.

**How mocap is recorded** — `sample_mocap(rx, t0, t1)` (:104-153) is the shape the new collection campaign should copy:
```python
win  = rx.snapshot_marker_window(t0, t1)   # MarkerWindow
qwin = rx.snapshot_window(t0, t1)          # MocapWindow
```
Per plate 0..5 it keeps only frames where **all four markers were tracked** (`arr.shape == (4,3)`, finite, and `sum(flags != 0) >= 4`), and emits
```python
{"mean_m": (4,3) list, "worst_marker_sd_m": float, "frames": int, "frac_of_window": float}
```
or `None` for a plate that never had four markers — "a mean over a changing subset is a number with no fixed meaning" (:111-113). Plus `streamed_poses` (per-plate `_quat_free_pose_mean`: positions averaged, rotation taken from the **middle** frame, because averaging rotation matrices elementwise leaves a non-orthogonal matrix, :89-101), `q_streamed_mean`, `q_streamed_sd`, `frames`, `duration_s`.

`sample_pressures(be)` (:156-163):
```python
{f"0x{b:03X}": {"psi": round(n.pressure_psi,3), "target_psi": round(n.target_psi,3),
                "enabled": bool(n.enabled), "flags": int(n.flags)}
 for b, n in sorted(be.snapshot_nodes().items()) if n.present}
```

**Output**: `schema = "canarm_drive_campaign/1"`, one record per step, fields `created_unix_s, created_local, psi, settle_s, sample_s, relax_s, can_port, rb_id_base=2000, n_bodies=6, steps[]` (:243-255); default path `hw_tests/results/drive_%Y-%m-%d_%H%M%S.json` (:361-366). Marker positions are the primary record: "every frame convention this workspace uses can be re-derived from them, whereas a streamed orientation cannot be un-rotated after the fact" (:37-39).

**Note for the new campaign:** it uses `CanArmMocap` (streamed poses), *not* `CanArmMarkerMocap`. Its `q_streamed_*` fields are therefore on the wrong azimuth by 45 deg — deliberately, since the point of the campaign was to *measure* the azimuth from raw markers. A Koopman data-collection campaign that wants a correct `q` online should use `CanArmMarkerMocap` instead, or re-derive `q` offline via `canarm_frames`.

---

## 4. `hw_tests/can_bringup.py` — the read-only acceptance test

Five stages, one process (`can_bringup.py:11-25`): `scan` → `diag` (read + clear) → `soak` (60 s at 150 Hz) → `diag` again → `port` (reopen/close).

**What it never emits** (:1-9): no OTA command, no set-ID command, and no enable bit — under *any* argument. It proves this rather than asserting it:

```python
def decode_table(backend: Backend) -> dict:            # :264
    frames = P.build_runtime_table(backend._build_targets())
    ...
            if flags & P.CONTROL_ENABLE:
                raise AssertionError(
                    f"refusing to start: table frame {hexid(can_id)} carries the enable "
                    f"bit for {hexid(base)} (word 0x{word:04X})")
```
and in `stage_soak` (:305-307):
```python
    assert not backend.selected, "selection must be empty for a read-only soak"
    assert not any(n.enabled for n in backend.nodes.values()), "no node may be enabled"
```

Expected wiring, checked per board (:66-68):
```python
EXPECTED_VARIANT = {base: P.VARIANT_TLE_DVP for base in P.TLE_IDS}
EXPECTED_VARIANT.update({base: P.VARIANT_7MM for base in P.SEVEN_MM_IDS})
```

Reply accounting comes from the test's **own tap** (`SoakRecorder`, :209), not from `Backend.stats`, because `_publish` credits replies only to `backend.selected` and this test selects none (:34-40). Latency is measured against `backend._cycle_t0`; reference figures are 2.04 ms at `0x101` rising to 3.49 ms at `0x118`, the spread being CAN arbitration by node id.

**The 15 checks it proves** (`evaluate`, :479-565): all 24 boards answered discovery; both sweeps agreed; every FW version string arrived complete; variant 2 on `0x101-0x108` and variant 0 on `0x109-0x118`; four diag frames from every board (before and after); the 150 Hz cycle ran the whole soak; every sync edge answered by at least one board; per-board reply rate ≥ floor; no board raised `STATUS_ERROR`; **no board reported itself enabled**; 0 RX overflows; 0 TX failures; the sync counter advanced by exactly the soak's cycle count (`cycles <= delta <= cycles+5`) and identically across the bus; the adapter port is free afterwards. 15/15 on 2026-08-20 (`repo_doc/directory.md:67`).

`stage_port_free(port, bitrate)` (:430-443) reopens and closes a fresh `CanLink` — "a leaked handle is otherwise invisible until [the next] phase fails".

---

## 5. `UMArm_MOCAP/` — reading the arm

### 5.1 The id block and the receivers (`UMArm_MOCAP/canarm_mocap.py`)

```python
CANARM_RB_ID_BASE = 2000     # canarm_mocap.py:58  — array row i is streaming id 2000+i
CANARM_N_BODIES   = 6        # :63  — 2000 (base) .. 2005 (tip); six, not the RS485 arm's seven
DEFAULT_TEMPLATE_PATH = <UMArm_MOCAP>/templates/canarm_locks.json   # :69
```
Verified live 2026-08-21: the census found exactly three contiguous blocks — `500-505` (RS485 arm), `1008` (Kinova), `2000-2005` (this arm) — each with four labeled markers. Body 2000 sits at z = 0.92 m and 2005 at z = 0.04 m; the arm hangs base-up, so **2000 is the base plate and the array index is `id - 2000`** (:3-12).

Two receivers, and the difference is a **correction, not a refinement** (:19-29):
```python
class CanArmMocap(MocapRx):                    # :72
    def __init__(self, *, rb_id_base=2000, n_bodies=6, **kwargs)      # :85-89

class CanArmMarkerMocap(MarkerMocap):          # :92
    def __init__(self, locks: dict, *, rb_id_base=2000, n_bodies=6,
                 phis=_AZIMUTH_DEFAULT, order=_AZIMUTH_DEFAULT, **kwargs)  # :116-128
```
`CanArmMarkerMocap` supplies three things the base class cannot know: the id block, `canarm_frames.plate_phis_rad()` for `phis`, and `canarm_frames.PROXIMAL_ORDER` (`"yx"`) for `order` (:122-126). A sentinel `_AZIMUTH_DEFAULT` distinguishes "caller said nothing" from "caller asked for the family defaults" — passing `phis=None` explicitly opts into the RS485 `FAMILY_PHI_RAD` (:111-120).

`load_canarm_locks(path=None) -> dict` (:131-151) raises `FileNotFoundError` naming the file *and* the three ways to mint it. It never falls back to the RS485 arm's `templates/rs485_locks_example.json`, "because a lock minted against different plates registers markers onto geometry that is not there and reports a confident `q` for it" (:66-68).

### 5.2 `MocapRx` — the API a controller uses (`UMArm_MOCAP/mocap_rx.py`)

```python
def __init__(self, server_ip=mc.DEFAULT_SERVER_IP, client_ip=mc.DEFAULT_CLIENT_IP,
             use_multicast=mc.DEFAULT_USE_MULTICAST, on_q=None,
             ring_capacity=mc.RING_CAPACITY, marker_ring_capacity=None,
             rb_id_base=mc.RIGID_BODY_ID_MASK, n_bodies=mc.N_USED_RIGID_BODIES)  # :342-349
```
Defaults from `UMArm_MOCAP/mocap_constants.py`: `DEFAULT_SERVER_IP = "192.168.1.100"` (:117), `DEFAULT_CLIENT_IP = "192.168.1.120"` (:121), `DEFAULT_USE_MULTICAST = True` (:123), `NOMINAL_RATE_HZ = 120.0` (:131), `STALE_AFTER_S = 0.25` (:136), `RING_CAPACITY = 2400` (:142, 20 s × 120 Hz), `NUM_JOINTS = 12` (:33), `NUM_RIGID_BODIES = 9` (:38), `N_USED_RIGID_BODIES = 7` (:42), `KINOVA_MOCAP_STREAM_ID = 1008` / `KINOVA_RIGID_BODY_INDEX = 8` (:65-66).

| method | line | returns |
|---|---|---|
| `start()` | 1044-1069 | the `NatNetClient`; **raises `RuntimeError` if `client.run()` is false** |
| `stop()` | 1071-1078 | None; safe twice and safe if `start` failed |
| `get_q()` | 783-792 | `(12,)` float **copy**, radians, or `None` |
| `get_homos()` | 793-797 | `(9,4,4)` streamed poses, spatial frame, or `None` |
| `get_state()` | 798-821 | `MocapState` |
| `wait_fresh(timeout=1.0)` | 823-846 | `q` from a frame arriving **after** the call, or `None` |
| `snapshot_window(t0, t1)` | 848-869 | `MocapWindow(t, frame_no, q (n,12), u (n,6,3))` |
| `snapshot_marker_window(t0, t1)` | 871-899 | `MarkerWindow(t, frame_no, mapping_epoch, markers, flags, streamed_poses (n,7,4,4))` |
| `marker_health()` | 902-923 | `MarkerHealth` |
| `resize_rings(...)`, `clear_history()` | 925-947 | — |
| `capture_rest(seconds=3.0, timeout=2.0)` | 949-1026 | `RestCapture(n, duration, mean, sd, ptp, window)` or `None` |

Both `t0`/`t1` are `time.monotonic()`.

`MocapState` fields (:138-178): `running, frames, valid_frames, frame_number, last_frame_wall, last_frame_mono, fps, stale, last_q_mono, q_stale, ring_len, last_error`.

> **`q_stale`, not `stale`, is the field a control loop must check** (mocap_rx.py:165-173). They differ exactly when frames keep arriving and stop converting; in that state `get_q()` still returns a pose, silently older every tick, while `stale` reads False.

`MarkerWindow.markers[i]` is `{plate: (m,3) float}` **or `None`** — `None` means *not streamed*, an empty dict means *streamed but no arm asset mapped* (:190-193 of the dataclass block, mocap_rx.py:229-234). `flags[i]` is `{plate: (m,) uint8}` with bit0 = tracked-this-frame, or `None` when labeled markers were absent, i.e. flags are *unknown*, never assumed all-present.

`capture_rest` returns `None` — never a reassuring number — when no fresh frame within `timeout`, when the stream went stale partway, when fewer than two samples landed, or when the samples span less than `MIN_REST_COVERAGE = 0.5` of `seconds` (mocap_rx.py:129, 953-966).

### 5.3 `MarkerMocap` — the marker-registered path (`UMArm_MOCAP/marker_mocap.py`)

```python
class MarkerMocap(MocapRx):                                       # :114
    def __init__(self, locks: dict, *, fallback_to_streamed: bool = False,
                 phis=None, order: str = "xy", **kwargs)          # :126-127
```
Constructor **raises `ValueError`** if any of `REQUIRED_PLATES = tuple(range(6))` (:60) is missing a lock (:143-149). `_solve_q` (:164-202) runs on the SDK thread — six Kabsch fits of four points — and produces `q` only when all six plates solved. Extra reads: `get_marker_poses() -> (7,4,4)` (:204-211), `get_streamed_poses()` (:213-217), `solve_stats() -> SolveStats(frames, solved, fallback, unsolved, reasons, last_rms_m)` (:219-225), `reset_stats()` (:227).

`mint_locks(rx, seconds=3.0, *, x_mode="streamed", require_tracked=True, log=None) -> (locks, report)` (:259-344). Gates: `MIN_REST_FRAMES = 10` (:70), `REST_STILLNESS_SD_M = 0.001` (1 mm per-marker sd, :76). `report["ok"]` is set on **every** path. **The GUI passes `x_mode="diagonal45"`, not the `"streamed"` default**, because on this arm Motive's alignment is not the mechanism's.

`save_locks(path, locks, meta=None)` (:346) and `load_locks(path)` (:354-378) — the latter tolerates the probe's `"plates"` key, `save_locks`' `"locks"` key, and a bare mapping.

### 5.4 `UMArm_MOCAP/canarm_frames.py` — the frame convention

```python
N_PLATES = 6                                                     # :76
DEFAULT_LOCK_PATH  = <UMArm_MOCAP>/templates/canarm_locks.json   # :79
DEFAULT_CALIB_PATH = <UMArm_MOCAP>/templates/canarm_frame_calib.json  # :82
AZIMUTH_MEASURED = True                                          # :90
PLATE_AZIMUTH_DEG = (-44.557, -0.757, -45.834, -0.339, -45.306, 0.383)   # :123
PLATE_AZIMUTH_FROM_DRIVE_AXES_DEG = (-45.190, 0.060, -45.018, -0.522, -45.490, 0.200)  # :127
PROXIMAL_ORDER = "yx"                                            # :143
AZIMUTH_GAUGE = "body x on the branch nearest mocap world +x at rest"    # :151
CO_RIGID_PAIRS = ((1, 2), (3, 4))                                # :328
```

Convention: `R_body = R_inferred @ Rz(-phi_p)` (`:104`, implemented at `body_frames`, :288-298). Even plates sit near −45 deg, odd near 0 deg — a 45 deg alternation from the bracket family — and **the whole set is shifted 45 deg from the RS485 `FAMILY_PHI_RAD` of `(0,45,0,45,0,45)`**: on this arm the marker arms lie *along* the revolute axes, not between them (:105-123). The two independent measurements (mechanism vs position refinement) agree to **0.82 deg**.

API:
| function | line | signature / returns |
|---|---|---|
| `plate_phis_rad(azimuth_deg=None)` | 154-160 | `(6,)` radians; raises unless shape `(6,)` |
| `arm_up(rest_markers)` | 173-188 | unit base→tip axis = `normalize(centroid(plate0) − centroid(plate5))`; **never a world axis** |
| `mint_locks(rest_markers, plates=None)` | 191-214 | `{plate: PlateLock}`; `streamed_rot=None`, `x_mode="diagonal45"` always |
| `save_locks(locks, path=None)` | 217-226 | writes `{"schema":"canarm_locks/1","azimuth_deg":[...],"azimuth_source":...,"plates":{...}}` |
| `load_locks(path=None)` | 229-248 | raises `FileNotFoundError` naming how to mint |
| `infer_frames(frame_markers, frame_flags, locks)` | 256-280 | `(poses (6,4,4), valid (6,), quality {plate: FrameQuality})` — bracket frames, identity where unsolved |
| `body_frames(poses, azimuth_deg=None)` | 288-298 | `(6,4,4)` body frames |
| `q_from_plate_frames(poses, azimuth_deg=None, order=None)` | 301-316 | `(12,)` radians or `None`; defaults to the measured azimuths **and** `"yx"` |
| `co_rigid_residual_deg(poses, azimuth_deg=None)` | 331-345 | per pair `(azimuth_error_deg, out_of_plane_deg)`; both should be zero |
| `chain_gaps_m(poses)` | 348-359 | `(5,)` consecutive u-joint centre distances, metres |
| `fk_centres_world(q, base_body_se3, params=None, order=None)` | 362-379 | `(6,3)` fkine centres in the mocap spatial frame; `base_body_se3` is **plate 0's body pose**, i.e. `body_frames(poses)[0]` |
| `fk_residual_m(poses, azimuth_deg=None, params=None, order=None)` | 382-396 | `(6,)` metres; **row 0 is identically zero** (the chain is anchored there) |

Measured chain, 2026-08-21 over 49 poses (`chain_gaps_m` docstring, :353-356): 265.37 / 72.85 / 234.47 / 72.97 / 230.12 mm with 0.04–0.17 mm sd. Co-rigidity residual: 0.03 deg sd over 49 poses, 0.4 deg worst out-of-plane (:326-327).

### 5.5 `UMArm_MOCAP/sim_stream.py`

```python
DEFAULT_LINKS_M = (0.28, 0.06, 0.24, 0.06, 0.23, 0.06)    # :42  placeholder shape only
DEFAULT_RATE_HZ = 120.0                                    # :45

def plate_poses_from_q(q, links_m=DEFAULT_LINKS_M, rot_base=None, base_pos=None) -> (6,4,4)  # :57
def inject_frame(rx, poses, frame_number=-1) -> None                                          # :98
def sweep_q(t_s, amplitude_rad=0.25) -> (12,)                                                 # :112

class CanArmSimStream(CanArmMocap):                                                           # :122
    def __init__(self, *, q_of_t=sweep_q, rate_hz=120.0, links_m=DEFAULT_LINKS_M,
                 rot_base=None, base_pos=None, rb_id_base=2000, n_bodies=6, **kwargs)         # :132-135
```
It overrides **only** `start()`/`stop()`; `start()` opens nothing and returns `self` (:145-156). `inject_frame` calls the receiver's real private listeners `rx._on_rigid_body(...)` and `rx._on_new_frame({"frame_number": ...})` (:106-109), so id routing, quaternion conversion, `mocap_to_q`, the publish-under-lock and the ring are all shipped code. It injects **rigid-body poses only — no markers**, so a `MarkerMocap`/`CanArmMarkerMocap` cannot be driven by it (the GUI enforces this at `canarm_control_gui.py:246-249`). `plate_poses_from_q` inverts the reader's own steps with the reader's own constants, so it is "a consistency and plumbing fixture, not a convention oracle" (:16-20).

---

## 6. `UMArm_KINEMATICS/` — the model

### 6.1 `fkine` and friends (`UMArm_KINEMATICS/fkine.py`)

```python
PROXIMAL_ORDERS = ("xy", "yx")                                       # :209
def segment_transform(params_row, q4, order="xy") -> (4,4)           # :212
def fkine(q, params=None, order="xy") -> (4,4)                       # :284
def ujoint_centres(q, params=None, order="xy") -> (6,3)              # :304
def plate_transforms(q, params=None, order="xy") -> (6,4,4)          # :325
def predict_spatial(base_se4, q, params=None, ee_lever_m=None, order="xy") -> (6,3) | (7,3)  # :364
def twist_exp(xi, theta) -> (4,4)                                    # :127
def segment_twists(params_row) -> (4,6)                              # :172
```
`q` is `(12,)` radians, ordered `[seg1 t1..t4, seg2 t1..t4, seg3 t1..t4]` (`_as_q`, :254-262). `params` is `(3,10)`, `None` routes through `robot_params.as_params`.

**`order` is the load-bearing argument.** `"xy"` composes `exp(ξ1 t1) exp(ξ2 t2)`; `"yx"` composes `exp(ξ2 t2) exp(ξ1 t1)` (:220-224). Both proximal twists pass through the segment origin, so the swap moves **no** u-joint centre by itself — it changes the orientation the rest of the chain inherits, and that is what propagates (:229-232). The **CAN arm measured `"yx"`**; the default is `"xy"` for bit-identity with the legacy Mathematica oracle. Using the wrong one does not fail: it leaves a `−t1·t2` twist the two-angle distal reader silently drops, showing up downstream as millimetres of u-joint-centre error (`marker_frame.q_from_frames` docstring).

`ujoint_centres` note: the distal centre is invariant to that segment's t3/t4, so **`q[10:12]` moves no centre at all** — its signal lives in plate 5's orientation only (:311-317).

### 6.2 `UMArm_KINEMATICS/canarm_params.py` — the MEASURED table

```python
MEASURED = True                                      # :59
MEASURED_SOURCE = ("hw_tests/results/drive_2026-08-21.json (68 poses, marker-inferred plate "
                   "frames) reduced by hw_tests/canarm_axis_analysis.py; CAD columns from "
                   "UMARM_Variable_Stiffness_Oct2025/robot_constants.py param_link0/1/2")   # :62-65

CANARM_PLATE_CHAIN_M = (0.265357, 0.072886, 0.234371, 0.072987, 0.229858)   # :79, metres

CANARM_PARAMS = np.array([
    # JA1    JA2    UC1  UC2  AA1        AA2        AO1    AO2    LL         JD
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.177932, 0.0],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.146946, 0.072886],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.142433, 0.072987],
], dtype=float)                                      # :91-96
CANARM_PARAMS.flags.writeable = False                # :97  — read-only on purpose
```
Column order is `robot_params.PARAM_COLUMNS = ("JA1","JA2","UC1","UC2","AA1","AA2","AO1","AO2","LL","JD")` (`UMArm_KINEMATICS/robot_params.py:49`, indices `COL_JA1=0 … COL_JD=9` at :51-60). All lengths are **metres**.

`UC1 = UC2 = 0`: **the plate centre *is* the joint centre on this arm** (:5-6), which is what makes `chain_gaps_m` pose-independent. `LL = span − (AA1 + AA2)`, so the measured chain enters through `LL` while `AA` keeps the CAD number (:86-90). `JA`/`AO` are consumed by nothing in the forward kinematics and ride along only so a row can be handed to the legacy oracle intact (:20-21).

Accuracy achieved with all three corrections (measured lengths + measured azimuth + `order="yx"`): **0.74 mm RMS on the fitted poses, 1.99 mm RMS / 7.0 mm worst on eighteen held-out multi-joint poses** spanning up to 30 deg of joint travel (:42-45). With the legacy `"xy"` order it was 4.6 mm RMS / 20.4 mm worst (`canarm_frames.py:140-142`). The prior placeholder — the RS485 table wearing the CAN arm's name — was wrong by 46 to 182 mm (:32-34).

`require_measured() -> np.ndarray` (:110-125) returns the table or raises; use it for any caller whose output is a length.

### 6.3 `UMArm_KINEMATICS/canarm_actuators.py` — THE FULL ACTUATOR MAP

```python
MEASURED = True                                                  # :66
MEASURED_SOURCE = ("hw_tests/results/drive_2026-08-21.json -- 24/24 boards pressurised alone "
                   "at 12 psi against a resting arm, joint identified as the argmax of |dq| "
                   "against the bracketing baselines, reduced by "
                   "hw_tests/canarm_axis_analysis.py.  Reproduced independently on the "
                   "earlier drive_sweep_2026-08-21.json run, which differed in its idle "
                   "setpoint (0 psi rather than 0.5) and gave the same 24 rows.")   # :70-76

MEASURED_JOINT_PAIRS = (                                         # :92-96
    (0x108, 0x104), (0x102, 0x106), (0x101, 0x105), (0x103, 0x107),
    (0x10A, 0x10C), (0x109, 0x10B), (0x110, 0x10E), (0x10D, 0x10F),
    (0x114, 0x112), (0x111, 0x113), (0x116, 0x118), (0x115, 0x117),
)

JOINT_NAMES = (                                                  # :116-120
    "s1.u1.t1", "s1.u1.t2", "s1.u2.t3", "s1.u2.t4",
    "s2.u3.t1", "s2.u3.t2", "s2.u4.t3", "s2.u4.t4",
    "s3.u5.t1", "s3.u5.t2", "s3.u6.t3", "s3.u6.t4",
)

SEGMENT_BLOCKS = (range(0x101, 0x109), range(0x109, 0x111), range(0x111, 0x119))   # :101

KNOWN_BOARD_FAULTS = {                                           # :109-112
    0x110: "leaks from the supply side; reads +5.1 psi at rest",
    0x104: "reads +1.1 psi at rest",
}
```

**Antagonistic pairs, joint by joint** (`(base_positive, base_negative)` — pressurising the first drives that joint angle positive):

| j | joint name | segment | + base | − base | board type |
|---|---|---|---|---|---|
| 0 | `s1.u1.t1` | 1 | `0x108` | `0x104` | TLE/DVP |
| 1 | `s1.u1.t2` | 1 | `0x102` | `0x106` | TLE/DVP |
| 2 | `s1.u2.t3` | 1 | `0x101` | `0x105` | TLE/DVP |
| 3 | `s1.u2.t4` | 1 | `0x103` | `0x107` | TLE/DVP |
| 4 | `s2.u3.t1` | 2 | `0x10A` | `0x10C` | 7 mm |
| 5 | `s2.u3.t2` | 2 | `0x109` | `0x10B` | 7 mm |
| 6 | `s2.u4.t3` | 2 | `0x110` ⚠ | `0x10E` | 7 mm |
| 7 | `s2.u4.t4` | 2 | `0x10D` | `0x10F` | 7 mm |
| 8 | `s3.u5.t1` | 3 | `0x114` | `0x112` | 7 mm |
| 9 | `s3.u5.t2` | 3 | `0x111` | `0x113` | 7 mm |
| 10 | `s3.u6.t3` | 3 | `0x116` | `0x118` | 7 mm |
| 11 | `s3.u6.t4` | 3 | `0x115` | `0x117` | 7 mm |

**The same table inverted — board → (joint index, joint name, sign)**, which is what `base_to_joint()` (:141-147) returns as `{base: (joint, sign)}`:

| base | joint | name | sign | | base | joint | name | sign |
|---|---|---|---|---|---|---|---|---|
| `0x101` | 2 | `s1.u2.t3` | **+1** | | `0x10D` | 7 | `s2.u4.t4` | **+1** |
| `0x102` | 1 | `s1.u1.t2` | **+1** | | `0x10E` | 6 | `s2.u4.t3` | **−1** |
| `0x103` | 3 | `s1.u2.t4` | **+1** | | `0x10F` | 7 | `s2.u4.t4` | **−1** |
| `0x104` ⚠ | 0 | `s1.u1.t1` | **−1** | | `0x110` ⚠ | 6 | `s2.u4.t3` | **+1** |
| `0x105` | 2 | `s1.u2.t3` | **−1** | | `0x111` | 9 | `s3.u5.t2` | **+1** |
| `0x106` | 1 | `s1.u1.t2` | **−1** | | `0x112` | 8 | `s3.u5.t1` | **−1** |
| `0x107` | 3 | `s1.u2.t4` | **−1** | | `0x113` | 9 | `s3.u5.t2` | **−1** |
| `0x108` | 0 | `s1.u1.t1` | **+1** | | `0x114` | 8 | `s3.u5.t1` | **+1** |
| `0x109` | 5 | `s2.u3.t2` | **+1** | | `0x115` | 11 | `s3.u6.t4` | **+1** |
| `0x10A` | 4 | `s2.u3.t1` | **+1** | | `0x116` | 10 | `s3.u6.t3` | **+1** |
| `0x10B` | 5 | `s2.u3.t2` | **−1** | | `0x117` | 11 | `s3.u6.t4` | **−1** |
| `0x10C` | 4 | `s2.u3.t1` | **−1** | | `0x118` | 10 | `s3.u6.t3` | **−1** |

⚠ = in `KNOWN_BOARD_FAULTS`.

**The legacy claim, kept for the diff** (:39-58). `LEGACY_ADDRESS_LIST` and `LEGACY_JOINT_LIST_INDEX` compose to:

| j | legacy (+, −) | measured (+, −) | same? |
|---|---|---|---|
| 0 | `0x102, 0x106` | `0x108, 0x104` | **no** |
| 1 | `0x104, 0x108` | `0x102, 0x106` | **no** |
| 2 | `0x103, 0x107` | `0x101, 0x105` | **no** |
| 3 | `0x105, 0x101` | `0x103, 0x107` | **no** |
| 4–11 | identical | identical | **yes** |

Segments 2 and 3 (all sixteen 7 mm boards) reproduce the legacy table exactly, pairing *and* sign — which is what fixes the sign convention, "since a robot frame rotated 180 deg about z would flip every sign and disagree with sixteen boards at once" (:80-83). Segment 1 is the same four *pairs* of boards rotated **+90 deg**: each pair drives the other axis of its universal joint, with the sign a +90 deg rotation implies (a pair formerly on +x now drives +y; a pair formerly on +y now drives −x). "That is exactly the signature of the top regulator platform having been remounted a quarter turn round, and it is why the legacy table must not be used on this arm" (:85-91).

**Use `joint_pairs()`, never the tables directly** (:123-138):
```python
def joint_pairs(allow_legacy: bool = False):
    if MEASURED:  return MEASURED_JOINT_PAIRS
    if allow_legacy:  return LEGACY_JOINT_PAIRS
    raise RuntimeError("the CAN arm's actuator/axis map has not been measured …")
```

---

## 7. `bench_env.py` — the single source of ports and IPs

`C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\bench_env.py`:

| name | value | line |
|---|---|---|
| `CAN_ADAPTER_VID / PID / SERIAL` | `0x16D0` / `0x117E` / `"3370376F3435"` | 89-91 |
| `CAN_PORT_ENV_VAR` | `"VEMA_CAN_PORT"` | 95 |
| `CAN_PORT_FALLBACK` | `"COM58"` | 99 |
| `CAN_INTERFACE` | `"slcan"` | 102 |
| `BITRATE` | `1_000_000` | 106 |
| `TTY_BAUDRATE` | `2_000_000` | 111 |
| `USB_PORT` | `"COM8"` | 75 |
| `MOCAP_SERVER_IP` | `"192.168.1.100"` | 204 |
| `MOCAP_CLIENT_IP` | `"192.168.1.120"` | 210 |
| `MULTICAST` / group | `True` / `"239.255.42.99"` | 213, 216 |
| `KINOVA_IP` | `"192.168.1.10"` | 222 |
| `PY_WS_VENV` | `<WS>/.venv/Scripts/python.exe` | 239 |

`resolve_can_port(override=None) -> str` (:147) — env var, then a `list_ports` scan ranked by VID/PID/serial (`_score`, :128), then the fallback. **Enumeration opens nothing.** `describe_can_port(override=None) -> str` (:177) for logging. `tlelib.wsenv.can_port()` (`TLE_PCB/tlelib/wsenv.py:237-254`) routes `tlelib`'s default through this, degrading silently to `"COM58"` if `bench_env.py` is missing or broken.

---

## 8. What I did NOT read

Read in full or in the parts that matter: `tlelib/{__init__,proto,canlink,backend,native,timing,wsenv}.py`, the relevant parts of `tlelib/vema_proto.py`, `VEMA_TLE_controller.py` (the `ControllerApp` control path; not the `ChannelBar` canvas drawing or `_replot`), `canarm_control_gui.py` (mocap strip, lock button, kinematics line, `_build_mocap`; not the viewer subprocess plumbing or `self_test`), `hw_tests/canarm_drive_campaign.py` (entire), `hw_tests/can_bringup.py` (docstring, `sweep`, `stage_scan`, `decode_table`, `stage_soak`, `stage_port_free`, `evaluate`; **not** `SoakRecorder`'s internals, `diag_raw`, `stage_diag`, `sync_deltas`, `markdown_table`, `main`), `UMArm_MOCAP/{canarm_mocap,canarm_frames,sim_stream}.py` (entire), `mocap_rx.py` (dataclasses, `__init__`, and every public read/lifecycle method; **not** `_on_mocap_data`, `_derive_mapping`, `_plate_flags`, `_solve_q`, `_on_new_frame`), `marker_mocap.py` (class + lock functions), `UMArm_KINEMATICS/{canarm_params,canarm_actuators}.py` (entire), `fkine.py` (lines 200-400).

**Not opened at all:** `tlelib/{ota,usblink,usbflash}.py`; `TLE_PCB/VEMA_TLE_flash.py`; `TLE_PCB/tools/`, `TLE_PCB/docs/`, `TLE_PCB/PORTING.md`, `TLE_PCB/README.md`; `UMArm_MOCAP/marker_frame.py` (the Kabsch registration, `compute_plate_lock`, `infer_plate_frame`, `PlateLock`, `FrameQuality`, the RMS gate — I read only `q_from_frames`' docstring), `mocap_to_q.py`, `mocap_probe.py`, `plate_markers.py`, `marker_frame_benchmark.py`, all `test_*.py`, `natnet_sdk/`; `UMArm_KINEMATICS/{robot_params (except the column constants), fkine_benchmark}.py`; `hw_tests/{canarm_axis_analysis, canarm_mocap_live, canarm_gui_kinematics, integrated_gui_test, mocap_census, mocap_*probe/sniff/discover, ota_legacy_gui_test, kinova_bridge_connectivity}.py` and every `report_*.md` / `results/*.json`; `viz/`, `digital_twin/`, `legacy_host/`, `firmware/`, `UMArm_KINOVA/`; the RS485 reference repo at `C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision` (outside my scope).

---

## 9. Flagged choices

- **I did not update `repo_doc/`**, despite the standing rule in `CLAUDE.md`, because this task was explicitly scoped read-only. Nothing I read contradicted `repo_doc/directory.md`, so it is accurate as it stands; the one thing worth adding on a later writable pass is a line naming `bench_env.resolve_can_port` as the sanctioned way to get the port.
- I read `hw_tests/can_bringup.py` selectively (the docstring plus the stages that carry the guarantees) rather than all 33 kB, since the port work needs its *guarantees* and its *safety idioms*, not its report formatting.
- I quoted the actuator map as two tables (pairs and the inverse) rather than only the source tuple, because the port needs the board→(joint, sign) direction and the source only carries the pair direction.

## KEY FILES
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\tlelib\backend.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\tlelib\proto.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\tlelib\canlink.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\hw_tests\canarm_drive_campaign.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\canarm_actuators.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\canarm_frames.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\canarm_mocap.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\canarm_params.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\mocap_rx.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\marker_mocap.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\fkine.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\bench_env.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\hw_tests\can_bringup.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\canarm_control_gui.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\VEMA_TLE_controller.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\tlelib\vema_proto.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\sim_stream.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\mocap_constants.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\robot_params.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\TLE_PCB\tlelib\native.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\repo_doc\directory.md

## GOTCHAS
- The enable bit is gated TWICE: `Backend._build_targets` (backend.py:301) computes `enable = node.enabled and not disable_everything and base in self.selected`. Calling `set_enabled(base, True)` on a board that is not in `select(...)` puts nothing on the wire. The campaign calls `be.select(list(P.ALL_IDS))` at canarm_drive_campaign.py:281 for exactly this reason.
- Compact status replies arrive on the BARE base id 0x101-0x118, not on base+0x300. `proto.parse_compact_status` (proto.py:241) only accepts `len(data)==2` with `0x101 <= can_id <= 0x118`. base+0x300 carries only MSG_ACK/NACK/CAN_DIAG/OTA_STATUS/FW_VERSION.
- `Backend._publish` credits replies ONLY to boards in `self.selected` (backend.py:380). Boards that are tabled but not selected show `pressure_psi = 0.0`, `replies = 0` forever, and their history stays empty - the reply is on the wire and the tap saw it, but nothing records it. can_bringup.py works around this with its own tap (can_bringup.py:34-40).
- psi<->counts calibration is chosen from the REPORTED variant byte, not the id range (`NodeCal.for_variant`, proto.py:177). A TLE board that has been assigned an id in 0x109-0x118 (the bench board once sat at 0x114) would be commanded on the 7mm scale if you keyed off the id - a ~10% psi error at the top of the range. Always scan before commanding, and never build a NodeCal from `for_base`.
- Board 0x110 leaks from its supply side: left DISABLED it drifted 0.1 -> 7.0 psi across a sweep. Every campaign must hold unused boards ENABLED at IDLE_PSI = 0.5 psi (canarm_drive_campaign.py:192-210), not disabled. Exactly zero psi is also wrong - it holds the exhaust valve open continuously. 0x104 and 0x110 additionally read +1.1 and +5.1 psi at rest, so their commanded psi carries that offset (canarm_actuators.py:109-112).
- `fkine`/`q_from_frames`/`ujoint_centres` default to `order="xy"` (the legacy Mathematica-oracle order). The CAN arm measured `order="yx"`. Getting it wrong does not raise - it leaves a -t1*t2 twist that the two-angle distal reader silently drops, and surfaces only as 4.6 mm RMS / 20.4 mm worst u-joint-centre error instead of 2.0 / 7.0. Use `UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER`, or the `CanArmMarkerMocap` / `canarm_frames.q_from_plate_frames` wrappers, which supply it.
- Plate azimuth: this arm's marker arms lie ALONG the revolute axes, not between them - `PLATE_AZIMUTH_DEG` is shifted 45 deg from `marker_frame.FAMILY_PHI_RAD`. A wrong azimuth yields a smooth, repeatable, entirely plausible-looking q that is 45 deg round from the mechanism. `MarkerMocap` defaults `phis=None` (= RS485 family angles); only `CanArmMarkerMocap` substitutes this arm's measured ones, and `phis=None` passed explicitly opts back OUT.
- `load_canarm_locks()` / `canarm_frames.load_locks()` raise FileNotFoundError by design: `templates/canarm_locks.json` is minted per Motive session and is gitignored. It must be re-minted after any Motive recalibration or asset re-solve. There is no fallback - the RS485 arm's `templates/rs485_locks_example.json` describes different plates and must never be substituted.
- `MocapState.stale` is NOT the field a control loop checks - `q_stale` is (mocap_rx.py:165-173). They differ exactly when frames keep arriving and stop converting; in that state `get_q()` keeps returning a pose, silently older every tick, while `stale` reads False. Use `wait_fresh(timeout)` after commanding, since `get_q()` will happily hand back a pre-command value.
- `MarkerWindow.markers[i] is None` means 'marker data did not arrive that frame'; an EMPTY DICT means 'streamed but no arm asset mapped'. Same for `flags[i]`: None means the tracked bits are UNKNOWN, never 'all present'. Conflating the two silently admits untracked markers into a mean.
- `Backend.scan()` returns `snapshot_nodes()` - every node the registry has ever seen, including ones now absent. Filter on `node.present`; do not take `len(scan())` as the board count (canarm_drive_campaign.py:275).
- `Backend.history(base)` without `max_points` copies the full 60 s / 9000-sample deque per board WHILE HOLDING THE LOCK the cycle thread needs to publish - a measured 2.8 Hz loss on the 24-board bus. Always pass `max_points` (~900) from any display or logger (backend.py:423-425).
- `CanArmSimStream` injects rigid-body poses ONLY, no labeled markers. A `MarkerMocap`/`CanArmMarkerMocap` cannot be driven by it, and locks cannot be minted from it (canarm_control_gui.py:246-249). Its `plate_poses_from_q` inverts the reader's own steps with the reader's own constants, so a shared error cancels exactly - it is a plumbing fixture, never a convention oracle.
- `MocapRx.start()` returns a `NatNetClient`; `CanArmSimStream.start()` returns `self`. Never assign `rx = SomeReceiver(...).start()` for the live path - doing so previously cost the GUI both `get_state` and `stop`, leaving the SDK's non-daemon threads running with no handle to reach them (canarm_control_gui.py:468-478).
- The 24-board pressure plot in the GUI holds the cycle to ~145 Hz instead of 150 (measured 2026-08-20). Any per-cycle logging or plotting added to a collection campaign must be budgeted against the 6.667 ms period, and the achieved rate read from `Backend.stats.cycles` over wall time rather than assumed.
- `slcan` bitrate S7 (750 kbit/s) is never emitted: the CANable 2.0 firmware mis-prescales it to 1.43 Mbit/s. Only S6 (500 k) and S8 (1 M) are offered (canlink.py:11-13). The adapter is also always opened in NORMAL mode, never listen-only, or a lone board's controller never sees an ACK and retries forever.
- `ujoint_centres` is invariant to q[10:12] - both axes of the last u-joint pass through its centre. Any fit or residual computed on centres alone carries zero information about the last two joints; their signal lives only in plate 5's orientation (fkine.py:311-317).
- `CANARM_PARAMS` is `writeable = False` on purpose (canarm_params.py:97): it is a default argument all over `fkine`, and an in-place edit would silently re-zero every other caller. Copy before perturbing.
