"""USB flashing through esptool.

Offsets are read from the build's own ``flasher_args.json`` rather than
hard-coded. The two firmware projects in this repository do not agree on them
-- the TLE project puts otadata at 0xd000 and the app at 0x10000, the V4PCB one
at 0xf000 and 0x20000 -- so a hard-coded table is a way to brick the wrong
board, and the file the build already emits is authoritative.

The NVS partition sits below the app in every layout here, and nothing written
by a flash reaches it, so a board keeps its ID across a USB flash exactly as it
does across an OTA.

esptool is run as a subprocess because the interpreter running the GUI usually
is not the one that has esptool: the project venv has pyserial, numpy and
tkinter but no esptool, while the ESP-IDF environment has esptool but no
tkinter. `resolve_esptool_python()` picks whichever interpreter can actually
import it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import wsenv

# In the VEMA_MAX22200 repository the TLE firmware was the repository-root
# ESP-IDF project, so its build tree sat two levels above this file. In this
# workspace the firmware lives at <workspace>/firmware/tle/, and that build
# directory is where the flash tools look first. `find_build()` below and the
# flash GUI's Browse button both take an explicit directory, so this stays a
# default rather than a requirement; bench_env.py overrides it if present.
REPO_ROOT = wsenv.WORKSPACE_ROOT
DEFAULT_BUILD = Path(wsenv.get("TLE_BUILD_DIR",
                               REPO_ROOT / "firmware" / "tle" / "build"))

# Machine-specific but correct for this bench: the ESP-IDF 5.5 environment is
# the interpreter that has esptool, and the GUI's interpreter is the one that
# has tkinter. Left as found.
IDF_PYTHON_CANDIDATES = [
    Path(r"C:\ESP\ESP_tools\python_env\idf5.5_py3.11_env\Scripts\python.exe"),
]

ESP32S3_USB_VID = 0x303A
ESP32S3_USB_PID = 0x1001


def resolve_esptool_python() -> str:
    """An interpreter that can `import esptool`, preferring the current one."""
    candidates = [Path(sys.executable)]
    env_python = os.environ.get("IDF_PYTHON_ENV_PATH")
    if env_python:
        candidates.append(Path(env_python) / "Scripts" / "python.exe")
    candidates.extend(IDF_PYTHON_CANDIDATES)
    for python in candidates:
        if not python.exists():
            continue
        try:
            probe = subprocess.run([str(python), "-c", "import esptool"],
                                   capture_output=True, timeout=30)
        except Exception:
            continue
        if probe.returncode == 0:
            return str(python)
    raise FileNotFoundError(
        "no python interpreter with esptool available; run the ESP-IDF export "
        "script, or pip install esptool into this environment")


@dataclass
class FlashPlan:
    """What to write where, taken from a build's flasher_args.json."""
    build_dir: Path
    chip: str = "esp32s3"
    write_flash_args: list[str] = field(default_factory=list)
    files: dict[int, Path] = field(default_factory=dict)   # offset -> file
    app_offset: int = 0x10000
    app_file: Path | None = None
    otadata_offset: int | None = None
    otadata_file: Path | None = None

    @property
    def app_size(self) -> int:
        return self.app_file.stat().st_size if self.app_file and self.app_file.is_file() else 0

    def missing(self) -> list[Path]:
        return [p for p in self.files.values() if not p.is_file()]


def find_build(start: Path | None = None) -> Path | None:
    """Locate the TLE firmware build directory."""
    if start is not None and (start / "flasher_args.json").is_file():
        return start
    if (DEFAULT_BUILD / "flasher_args.json").is_file():
        return DEFAULT_BUILD
    return None


def read_plan(build_dir: Path) -> FlashPlan:
    args_path = Path(build_dir) / "flasher_args.json"
    if not args_path.is_file():
        raise FileNotFoundError(
            f"{args_path} not found; build the firmware first (idf.py build)")
    blob = json.loads(args_path.read_text())
    plan = FlashPlan(build_dir=Path(build_dir))
    plan.chip = blob.get("extra_esptool_args", {}).get("chip", "esp32s3")
    plan.write_flash_args = list(blob.get("write_flash_args", []))
    for offset, name in blob.get("flash_files", {}).items():
        plan.files[int(offset, 16)] = Path(build_dir) / name
    app = blob.get("app", {})
    if app:
        plan.app_offset = int(app["offset"], 16)
        plan.app_file = Path(build_dir) / app["file"]
    otadata = blob.get("otadata", {})
    if otadata:
        plan.otadata_offset = int(otadata["offset"], 16)
        plan.otadata_file = Path(build_dir) / otadata["file"]
    return plan


def build_command(port: str, plan: FlashPlan, mode: str = "app",
                  baud: int = 460800, python: str | None = None) -> list[str]:
    """`mode` is 'app' (application only) or 'full' (everything).

    'app' writes the application image, and with it the initial otadata, so the
    bootloader is pointed back at the partition just written. That second part
    is not optional: the app offset in flasher_args.json is the *factory*
    partition, while an update writes ota_0 or ota_1 and leaves otadata
    selecting it. Without restoring otadata, flashing a board over USB after it
    has ever been updated over CAN writes a partition the bootloader is no
    longer choosing, and the board silently comes back running the old image.

    'full' additionally rewrites the bootloader and the partition table, which
    is required when the layout changes and is the way back for a board whose
    OTA slots are both bad.

    Neither touches NVS, so neither changes the board ID.
    """
    if mode not in ("app", "full"):
        raise ValueError(f"unknown flash mode {mode!r}")
    if plan.app_file is None:
        raise FileNotFoundError("build has no application image")

    interpreter = python or resolve_esptool_python()
    cmd = [interpreter, "-m", "esptool", "--chip", plan.chip, "-p", port, "-b", str(baud),
           "--before", "default_reset", "--after", "hard_reset", "write_flash"]
    cmd += plan.write_flash_args

    if mode == "full":
        missing = plan.missing()
        if missing:
            raise FileNotFoundError("full flash needs " + ", ".join(str(p) for p in missing))
        for offset in sorted(plan.files):
            cmd += [hex(offset), str(plan.files[offset])]
    else:
        if not plan.app_file.is_file():
            raise FileNotFoundError(f"{plan.app_file} not found")
        cmd += [hex(plan.app_offset), str(plan.app_file)]
        if plan.otadata_offset is not None and plan.otadata_file is not None:
            if not plan.otadata_file.is_file():
                raise FileNotFoundError(f"{plan.otadata_file} not found")
            cmd += [hex(plan.otadata_offset), str(plan.otadata_file)]
    return cmd


_PROGRESS = re.compile(r"\((\d+)\s*%\)")


class UsbFlasher:
    def __init__(self, on_line=None, on_progress=None):
        self.on_line = on_line
        self.on_progress = on_progress
        self._proc: subprocess.Popen | None = None

    def flash(self, port: str, plan: FlashPlan, mode: str = "app", baud: int = 460800) -> int:
        cmd = build_command(port, plan, mode, baud)
        if self.on_line:
            self.on_line("$ " + " ".join(cmd))

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1, env=env)
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip()
            if self.on_line:
                self.on_line(line)
            if self.on_progress:
                found = _PROGRESS.search(line)
                if found:
                    self.on_progress(int(found.group(1)))
        self._proc.wait()
        code = self._proc.returncode
        self._proc = None
        return code

    def cancel(self) -> None:
        if self._proc is not None:
            self._proc.terminate()


def list_ports() -> list[tuple[str, str]]:
    """Every serial port, ESP32-S3 ones tagged."""
    from serial.tools import list_ports as lp
    out = []
    for port in lp.comports():
        tag = " (ESP32-S3)" if (port.vid, port.pid) == (ESP32S3_USB_VID, ESP32S3_USB_PID) else ""
        out.append((port.device, f"{port.device}{tag} - {port.description}"))
    return out
