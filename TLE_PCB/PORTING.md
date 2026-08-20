# Porting notes — `TLE_PCB/` from `VEMA_MAX22200` into this workspace

Source: `C:\ESP\ESP_Projects\VEMA_MAX22200\TLE_PCB\` @ branch `TLE_PCB`, HEAD `947375a`.
Everything not listed below is byte-for-byte identical to the source repository, comments included —
those comments record measured hardware behaviour, and paraphrasing them loses the evidence.

## What changed

1. **`tlelib/vema_proto.py` is new — vendored verbatim** from `VEMA_MAX22200/vema_gui/vema_proto.py`
   (SHA-256 verified identical). It carries the fw 1.45 bench tuning in `DEFAULTS` and its rationale.
2. **`tlelib/native.py`** no longer path-loads `../../vema_gui/vema_proto.py` through `importlib`;
   it now does `from . import vema_proto as _vp`. The re-export list, `build_set_device_id()` and
   `tuning_frames()` are unchanged, and the docstring still explains why the split exists.
   This was the single dangling dependency on the old repository root.
3. **`tlelib/wsenv.py` is new** — a guarded reader for `<workspace>/bench_env.py`. A missing,
   older, or broken `bench_env.py` degrades to the values measured on this bench rather than
   raising, so `TLE_PCB/` still runs standalone.
4. **`tlelib/usbflash.py`**: `REPO_ROOT` is now the workspace root and
   `DEFAULT_BUILD = <workspace>/firmware/tle/build`, overridable by a `bench_env.TLE_BUILD_DIR`
   if one is ever added. `IDF_PYTHON_CANDIDATES` and the otadata-rewrite logic are untouched.
5. **Port defaults** resolve through `wsenv`, falling back to `COM58` (CANable) / `COM8` (board).
   `canlink.DEFAULT_PORT` calls `bench_env.resolve_can_port()`, which ranks enumerated ports by the
   dongle's USB descriptor triple `VID 16D0 / PID 117E / serial 3370376F3435` and so survives the
   replug that already invalidated the literals `COM31` and `COM4` in two older repositories;
   enumeration opens nothing. `usblink.DEFAULT_PORT` still resolves to `COM8`, since `bench_env`
   names no board port yet. `backend.Backend(port=...)`, `tools/tle_bench.py` `USB_PORT`/`CAN_PORT`
   and the two combobox pre-selections in the GUIs now read `canlink.DEFAULT_PORT` rather than the
   literal `"COM58"`; both GUIs gained one `from tlelib import canlink` line for that.
6. **`README.md`** bench commands repointed at `firmware/tle/`. Layout of the GUIs is unchanged.

## Traps that survive the move — do not "optimise" these

- **Cross-flash guard.** A `VEMA_MAX22200` image broadcast to the sixteen legacy 7 mm boards *will*
  be accepted by all of them and take the lower arm off the bus until each is hand-reflashed over
  USB. The only guard is the `esp_app_desc_t.project_name` check, at three call sites:
  `ota.PROJECT_VARIANTS` (`VEMA_MAX22200` → `{TLE_DVP}`, `Valve_not_embedded_XL` → `{7MM, DT}`),
  `VEMA_TLE_flash.py` before broadcast, and `tools/tle_bench.py` in the `ota` stage. The wire path
  itself is variant-agnostic by design, so dropping any of the three removes the guard silently.
- **`BATCH_FRAMES = 1` in `ota.py` is a measurement, not a placeholder.** One 896 B block at a
  300 µs gap: batch 8 delivered 1 frame of 128, batch 4 delivered 2, batch 2 delivered 3, batch 1
  delivered all 128 — and was faster. Frames written together land in the board's single RX buffer
  for `0x090` and overwrite each other.
- **`CMD_SET_ID` (0x05 on `base+0x100` over CAN) takes a BIG-endian ID** in bytes 1–2, the only
  big-endian field in the protocol. The native set-device-ID, same opcode over USB or `base+0x400`,
  is little-endian. Reversing either sends a board to an ID no scan will find.
- **This protocol has no ABORT.** A board that received `CMD_START` sits with its control loop
  stopped and outputs off until an `END` arrives or it is power-cycled. `ota.upload()` closes every
  session it opened in a `finally` block; that block must survive any re-drive of the OTA path.
- **Never open the adapter listen-only, never emit `S7`.** Listen-only withholds ACK bits and a lone
  board's controller then retries forever; CANable 2.0's prescaler bug puts nominal 750 k at
  1.43 Mbit/s. `canlink.BITRATES` offers only `S6` (500 k) and `S8` (1 M), and the open sequence
  ends in `O\r` (normal mode). Related: the receive thread polls `in_waiting` instead of blocking —
  a blocking read moved median reply latency from 1.07 ms to 3.31 ms.

## Verified offline (no port opened, no board touched)

`import tlelib` and every submodule under base Python 3.13; `ID_BROADCAST == 0x090`; runtime-table
IDs `0x101→0x091 … 0x118→0x098` with the `0x80` marker present; `PROJECT_VARIANTS` intact;
`build_set_id(0x105) == 05 01 05` against `native.build_set_device_id(0x105) == 05 05 01`;
`bus_load` 37–44 % at 1 Mbit/s for 24 boards; `docs/make_figures.py` regenerated all six SVGs
**byte-identical** to the committed ones; `compileall` clean; both GUI classes constructed against a
withdrawn Tk root without connecting.
