# Firmware trees

Two ESP-IDF projects and one directory of prebuilt images.

| Path | Project | Target boards |
| --- | --- | --- |
| `tle/` | `VEMA_MAX22200` | the eight TLE92464 boards, CAN IDs `0x101`–`0x108` |
| `legacy/` | `Valve_not_embedded_XL` | the sixteen Clippard 7 mm boards, `0x109`–`0x118` |
| `images/` | — | the exact binaries flashed on the arm; provenance in `images/README.md` |

Both were copied from their source repositories on 2026-08-20 and both were rebuilt here to prove
the trees are complete. Neither `build/` directory is tracked.

## Building

Each PowerShell invocation is a fresh process, so the environment setup, `export.ps1` and the
`idf.py` call must be one command. The same prologue is available programmatically as
`bench_env.IDF_ENV_INCANTATION`.

```powershell
$env:IDF_TOOLS_PATH = "C:\ESP\ESP_tools"
$env:IDF_PYTHON_ENV_PATH = "C:\ESP\ESP_tools\python_env\idf5.5_py3.11_env"
$env:PATH = "C:\ESP\ESP_tools\python_env\idf5.5_py3.11_env\Scripts;" + $env:PATH
. C:\ESP\ESP_container\v5.5.1\esp-idf\export.ps1

# TLE
idf.py -C firmware\tle -B firmware\tle\build build

# Legacy, 7 mm variant (VALVE_TYPE=0). VALVE_TYPE=1 builds the DT image instead.
idf.py -C firmware\legacy -B firmware\legacy\build_7mm -DVALVE_TYPE=0 build
```

Both root `CMakeLists.txt` files set `idf_build_set_property(MINIMAL_BUILD ON)`, and that line is
not optional on this machine: the xtensa gcc 14.2.0 toolchain deterministically segfaults
compiling `esp_lcd/rgb/esp_lcd_panel_rgb.c`, so a project that builds every component fails in a
way that reads as a corrupt toolchain rather than a missing property.

## Two things that will not survive a naive re-clone

**`legacy/sdkconfig` is gitignored upstream.** It carries `CONFIG_PARTITION_TABLE_TWO_OTA=y`,
`CONFIG_ESPTOOLPY_FLASHSIZE="4MB"` and `CONFIG_FREERTOS_HZ=1000`; regenerating it from defaults
produces a single-app partition table and OTA then fails with "Could not find OTA partition".
The working file was copied here by hand and is tracked in this workspace. The TLE tree carries
both `sdkconfig` and `sdkconfig.defaults` for the same reason.

**The TLE project takes its version string from `git describe`.** Upstream that yields
`tle-fw1.43-15-g947375a`, which is what the board reports over CAN in response to
`CMD_GET_FW_VERSION`. This workspace's repository had no commits when the tree was first built,
so `git describe` failed and ESP-IDF fell back to `PROJECT_VER = "1"`:

```
-- git describe returned 'fatal: bad revision 'HEAD''
-- Could not use 'git describe' to determine PROJECT_VER.
-- App "VEMA_MAX22200" version: 1
```

The condition clears itself once the workspace has a commit — `git describe --always` then
returns a short hash — but a build made from this tree will never reproduce the upstream
`tle-fw1.43-*` string unless the corresponding tag is created here or `PROJECT_VER` is set
explicitly. Anything that keys off the reported version, rather than off the `esp_app_desc`
project name, must account for that. The legacy tree is unaffected: it pins
`set(PROJECT_VER "0.2.1")` in its `CMakeLists.txt` and rebuilds to `0.2.1` unchanged.

## Verified

| Build | Result |
| --- | --- |
| `firmware/tle` | `VEMA_MAX22200.bin`, 324 528 B, `esp_app_desc` project `VEMA_MAX22200`, IDF v5.5.1 |
| `firmware/legacy` (`VALVE_TYPE=0`) | `Valve_not_embedded_XL.bin`, 268 704 B, project `Valve_not_embedded_XL`, version `0.2.1` |

The legacy image matches the flashed `images/legacy_7mm/` binary in size, project name and
version; the bytes differ only in the embedded build timestamp and SHA. The `VALVE_TYPE=1` (DT)
variant was not rebuilt here — only the 7 mm variant was exercised.
