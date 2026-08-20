# Prebuilt firmware images — provenance

Copied 2026-08-20 into this workspace. Every file here is a byte-for-byte copy of a build
artifact that was produced elsewhere; nothing in this directory was rebuilt. The point of
keeping them is that flashing the arm must not depend on a working toolchain, and that the
image which is *actually running* on a board is a stronger reference than any tree that could
be rebuilt to something slightly different.

Fields below were read out of each image's `esp_app_desc_t` (offset `0x20`), so they describe
the binary itself rather than the directory it came from.

## `legacy_7mm/` — the image running on `0x109`–`0x118`

| | |
| --- | --- |
| Source | `C:\ESP\ESP_Projects\VNEMA_MK8_PIDPWM\build_7mm\` |
| Project name | `Valve_not_embedded_XL` |
| Version | `0.2.1` |
| Built | May 18 2026, 22:45:50, ESP-IDF v5.5.1 |
| App size | 268 704 B |
| SHA-256 (app) | `60ce5a6a135ad76e…` |
| Build variant | `VALVE_TYPE=0` (7 mm Clippard, GPIO bang-bang) |

**This image is byte-identical to what is flashed on the sixteen lower boards, CAN IDs `0x109`
through `0x118`.** The identity was confirmed by SHA-256 against the source directory at copy
time, and the running version was independently confirmed over CAN in
`may18_fixed_error_and_miss.md`, which records `0.2.1 (7mm)` returned by all sixteen boards
after the 2026-05-18 rollout. Version `0.2.1` is the build that carries the MCP2515 RXB0/RXB1
filter split; the older `0.2.0` in the upstream `build/` directory reintroduces the `RX1OVR`
bug and was deliberately not copied.

## `legacy_dt/` — the image running on `0x101`–`0x108`

| | |
| --- | --- |
| Source | `C:\ESP\ESP_Projects\VNEMA_MK8_PIDPWM\build_dt\` |
| Project name | `Valve_not_embedded_XL` |
| Version | `0.2.1` |
| Built | May 18 2026, 22:45:39, ESP-IDF v5.5.1 |
| App size | 281 344 B |
| SHA-256 (app) | `a534e04250501794…` |
| Build variant | `VALVE_TYPE=1` (DT big valve, LEDC PWM) |

These are the eight boards being replaced by the TLE92464 hardware. The image is kept so a DT
board can be restored, and so `can_ota.py`'s DT/7 mm firmware split still resolves to a real
file.

## `tle/` — the TLE92464 image

| | |
| --- | --- |
| Source | `C:\ESP\ESP_Projects\VEMA_MAX22200\build\` |
| Project name | `VEMA_MAX22200` |
| Version | `tle-fw1.43-13-g1a45afc-dirty` |
| Built | Aug 14 2026, 14:21:28, ESP-IDF v5.5.1 |
| App size | 324 528 B |
| SHA-256 (app) | `71b6f6b2e9035458…` |

Two caveats attach to this one. First, the version string is produced by `git describe` on the
source repository at build time, and the recorded value is `…-13-g1a45afc-dirty`, i.e. the build
predates the current `VEMA_MAX22200` HEAD (`947375a`, which `git describe` renders as
`tle-fw1.43-15-g947375a`) by two commits and was made from a dirty working tree. The image is
therefore the last artifact on disk, not necessarily the last source state. Second, unlike the
legacy pair there is no independent over-CAN confirmation recorded that this exact image is what
sits on the eight TLE boards today; treat it as "the newest image that exists" rather than as a
verified running version until a `CMD_GET_FW_VERSION` query says otherwise.

## Flash geometry

Offsets differ between the two firmwares and are carried verbatim in each directory's
`flasher_args.json`. Do not cross them.

| | legacy (`Valve_not_embedded_XL`) | TLE (`VEMA_MAX22200`) |
| --- | --- | --- |
| flash size | 4 MB | 8 MB |
| bootloader | `0x0` | `0x0` |
| partition table | `0x8000` | `0x8000` |
| otadata | `0xd000` | `0xd000` |
| app | `0x10000` | `0x10000` |

The `flasher_args.json` in each directory still names its files by the relative paths the
original build tree used (`bootloader/bootloader.bin`, `partition_table/partition-table.bin`).
The bins have been flattened into one directory here, so a tool that consumes `flasher_args.json`
literally must either be pointed at the flattened names or given the original build tree.

## What is not here

No `.elf`, `.map`, or `compile_commands.json`. Those are 4–5 MB apiece and are reproducible from
`firmware/tle` and `firmware/legacy`; the images are not, once the source repositories move on.
