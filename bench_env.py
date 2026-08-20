"""Bench-wide constants and device resolution for the UMArm workspace.

Every entry point in this workspace imports this module instead of hard-coding a
COM number, an IP address, or an interpreter path. The motivation is a failure
mode that has already cost time twice: the USB-CAN dongle enumerates as
whichever COM number Windows assigns to the hub port it happens to be plugged
into, so the literals ``COM31`` (VEMA_MAX22200) and ``COM4`` (VNEMA_MK8_PIDPWM)
that both source repos still carry name a port that no longer exists. The
dongle's *stable* identity is its USB descriptor triple, and pyserial can read
that triple by enumeration alone, without opening the port.

The module is deliberately dependency-light: only ``pyserial`` is used, and it is
imported lazily inside :func:`resolve_can_port` so that importing ``bench_env``
never fails on an interpreter that lacks it.

Nothing here opens a port, a socket, or a file. Import is free of side effects.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "WS_ROOT",
    "CAN_ADAPTER_VID",
    "CAN_ADAPTER_PID",
    "CAN_ADAPTER_SERIAL",
    "CAN_PORT_ENV_VAR",
    "CAN_PORT_FALLBACK",
    "CAN_INTERFACE",
    "BITRATE",
    "TTY_BAUDRATE",
    "MOCAP_SERVER_IP",
    "MOCAP_CLIENT_IP",
    "MULTICAST",
    "MOCAP_MULTICAST_GROUP",
    "NATNET_SDK_DIR",
    "KINOVA_IP",
    "KINOVA_USER",
    "KINOVA_PASSWORD",
    "PY_BASE",
    "PY_WS_VENV",
    "PY_KINOVA_VENV",
    "IDF_PATH",
    "IDF_TOOLS_PATH",
    "IDF_PYTHON_ENV_PATH",
    "IDF_EXPORT_PS1",
    "IDF_ENV_INCANTATION",
    "FIRMWARE_IMAGES_DIR",
    "USB_PORT",
    "TLE_BUILD_DIR",
    "resolve_can_port",
    "describe_can_port",
    "list_serial_ports",
]

# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------

#: Absolute path to the workspace root, derived from this file's location so the
#: whole tree can be relocated without editing anything.
WS_ROOT = Path(__file__).resolve().parent

#: Prebuilt, provenance-tracked firmware images (see ``firmware/images/README.md``).
FIRMWARE_IMAGES_DIR = WS_ROOT / "firmware" / "images"

#: Vendored NatNet SDK. Written by the mocap agent into ``UMArm_MOCAP/natnet_sdk``.
NATNET_SDK_DIR = WS_ROOT / "UMArm_MOCAP" / "natnet_sdk"

#: Default board-side USB-Serial/JTAG port for a single cabled ESP32-S3 (bench
#: flashing / framed text link). Unlike the CAN dongle this has no stable USB
#: identity worth ranking on — boards come and go; override at the call site.
USB_PORT = "COM8"

#: Where the TLE firmware build lands; TLE_PCB/tlelib/usbflash.py reads this
#: (via wsenv) to find flasher_args.json for USB flashing.
TLE_BUILD_DIR = WS_ROOT / "firmware" / "tle" / "build"

# ---------------------------------------------------------------------------
# (a) USB-CAN adapter
# ---------------------------------------------------------------------------

#: USB descriptor triple of the CANable-class SLCAN dongle on this bench, read
#: from ``USB\VID_16D0&PID_117E\3370376F3435``. This triple, not the COM number,
#: is the adapter's identity: the COM number tracks the hub port
#: (``LOCATION=1-6.4.4.4.3.1`` at the time of writing) and moves on a replug.
CAN_ADAPTER_VID = 0x16D0
CAN_ADAPTER_PID = 0x117E
CAN_ADAPTER_SERIAL = "3370376F3435"

#: Environment variable that overrides port resolution outright. Set it when the
#: adapter is swapped for a different unit, or when a second dongle is present.
CAN_PORT_ENV_VAR = "VEMA_CAN_PORT"

#: Last-resort port, used when neither the environment variable nor the USB scan
#: yields anything. This is where the adapter enumerated on 2026-08-20.
CAN_PORT_FALLBACK = "COM58"

#: python-can interface name. The dongle speaks Lawicel/SLCAN ASCII.
CAN_INTERFACE = "slcan"

#: CAN bus bit rate, 1 Mbit/s. Both firmwares configure the MCP2515 for this
#: rate with a 16 MHz crystal; it is not adjustable without reflashing.
BITRATE = 1_000_000

#: Host-side USB CDC line rate. The 24-board 150 Hz runtime moves roughly
#: 24 replies plus 8 table frames per 6.67 ms cycle, and the legacy install
#: notes require the serial link to stay at or above 2 Mbaud to keep up.
TTY_BAUDRATE = 2_000_000


def list_serial_ports():
    """Return the pyserial ``ListPortInfo`` records for every enumerated port.

    Enumeration only; no port is opened. Returns an empty list if pyserial is
    not importable, so a caller on a minimal interpreter degrades to the
    fallback port rather than raising.
    """
    try:
        from serial.tools import list_ports  # lazy: pyserial is optional here
    except Exception:
        return []
    return list(list_ports.comports())


def _score(port) -> int:
    """Rank one enumerated port against the known adapter identity.

    A full VID/PID/serial match scores 3, VID+PID scores 2, VID alone scores 1,
    and anything else scores 0. Ranking rather than filtering means a dongle
    whose serial number was reprogrammed still wins over an unrelated CDC device.
    """
    vid = getattr(port, "vid", None)
    pid = getattr(port, "pid", None)
    ser = (getattr(port, "serial_number", None) or "").upper()
    if vid != CAN_ADAPTER_VID:
        return 0
    if pid != CAN_ADAPTER_PID:
        return 1
    if ser == CAN_ADAPTER_SERIAL.upper():
        return 3
    return 2


def resolve_can_port(override: str | None = None) -> str:
    """Resolve the USB-CAN adapter to a COM device name.

    Resolution order, most explicit first:

    1. ``override`` — whatever a ``--port`` command-line flag supplied.
    2. The ``VEMA_CAN_PORT`` environment variable.
    3. A ``serial.tools.list_ports`` scan ranked by :func:`_score`, i.e. the
       adapter's own USB identity.
    4. :data:`CAN_PORT_FALLBACK` (``COM58``).

    The explicit sources are consulted before the scan because a human naming a
    port is always more authoritative than an inference, and because the scan
    must not silently win when two dongles are attached.
    """
    if override:
        return str(override)
    env = os.environ.get(CAN_PORT_ENV_VAR)
    if env:
        return env
    best, best_score = None, 0
    for port in list_serial_ports():
        score = _score(port)
        if score > best_score:
            best, best_score = port, score
    if best is not None:
        return best.device
    return CAN_PORT_FALLBACK


def describe_can_port(override: str | None = None) -> str:
    """One human-readable line naming the resolved port and how it was chosen.

    Intended for a banner line at tool start-up, since a wrong-port failure is
    otherwise indistinguishable from a dead bus.
    """
    if override:
        return f"{override} (explicit --port)"
    env = os.environ.get(CAN_PORT_ENV_VAR)
    if env:
        return f"{env} (from ${CAN_PORT_ENV_VAR})"
    best, best_score = None, 0
    for port in list_serial_ports():
        score = _score(port)
        if score > best_score:
            best, best_score = port, score
    if best is not None:
        how = {3: "VID:PID+serial match", 2: "VID:PID match", 1: "VID match"}[best_score]
        return f"{best.device} ({how}: {best.description})"
    return f"{CAN_PORT_FALLBACK} (fallback; no matching USB device enumerated)"


# ---------------------------------------------------------------------------
# (c) Motion capture
# ---------------------------------------------------------------------------

#: Motive host PC on the camera network.
MOCAP_SERVER_IP = "192.168.1.100"

#: This machine's Intel I226-V NIC on that same /24. The bind address must be
#: named explicitly: three interfaces on this machine claim the 224.0.0.0/4
#: route, so a client that lets the OS choose can join the multicast group on
#: Wi-Fi and then receive nothing, silently and without an error.
MOCAP_CLIENT_IP = "192.168.1.120"

#: NatNet transport. Motive's stock configuration here is multicast.
MULTICAST = True

#: Motive's default multicast group, for reference when debugging with a sniffer.
MOCAP_MULTICAST_GROUP = "239.255.42.99"

# ---------------------------------------------------------------------------
# (d) Kinova Gen3
# ---------------------------------------------------------------------------

KINOVA_IP = "192.168.1.10"
#: Factory-default credentials. The 192.168.1.0/24 segment is flat and has no
#: gateway, so anything on it can command the arm; do not publish these outward.
KINOVA_USER = "admin"
KINOVA_PASSWORD = "admin"

# ---------------------------------------------------------------------------
# (e) Interpreters
# ---------------------------------------------------------------------------

#: Base CPython 3.13.13. Carries mujoco 3.11, numpy, scipy, matplotlib,
#: pyserial, tkinter and pytest. Use this when the workspace venv is absent.
PY_BASE = Path(r"C:\Users\zuorunze\AppData\Local\Programs\Python\Python313\python.exe")

#: Workspace venv: base 3.13 with --system-site-packages plus python-can.
#: A bare ``python`` on this machine resolves to an unrelated 3.12 venv, so every
#: subprocess launch must name an absolute interpreter.
PY_WS_VENV = WS_ROOT / ".venv" / "Scripts" / "python.exe"

#: The Kortex quarantine. ``kortex_api`` pins protobuf 3.5.1, which cannot
#: coexist with MuJoCo or a modern protobuf, so the Kinova half runs in its own
#: interpreter behind a stdio JSON bridge.
PY_KINOVA_VENV = Path(r"C:\RUNZE_SRC\RS485_VEMA\.venv_kinova\Scripts\python.exe")

# ---------------------------------------------------------------------------
# ESP-IDF
# ---------------------------------------------------------------------------

IDF_PATH = Path(r"C:\ESP\ESP_container\v5.5.1\esp-idf")
IDF_TOOLS_PATH = Path(r"C:\ESP\ESP_tools")
IDF_PYTHON_ENV_PATH = IDF_TOOLS_PATH / "python_env" / "idf5.5_py3.11_env"
IDF_EXPORT_PS1 = IDF_PATH / "export.ps1"

#: The exact PowerShell prologue that makes ``idf.py`` work from a plain shell on
#: this machine. Each PowerShell invocation is a fresh process, so the env-var
#: assignments, ``export.ps1`` and the ``idf.py`` call must all be one command.
#: A project built without ``idf_build_set_property(MINIMAL_BUILD ON)`` in its
#: root ``CMakeLists.txt`` will fail here: the xtensa gcc 14.2.0 toolchain
#: deterministically segfaults compiling ``esp_lcd/rgb/esp_lcd_panel_rgb.c``, and
#: the failure reads as a corrupt toolchain rather than a missing config line.
IDF_ENV_INCANTATION = r"""$env:IDF_TOOLS_PATH = "C:\ESP\ESP_tools"
$env:IDF_PYTHON_ENV_PATH = "C:\ESP\ESP_tools\python_env\idf5.5_py3.11_env"
$env:PATH = "C:\ESP\ESP_tools\python_env\idf5.5_py3.11_env\Scripts;" + $env:PATH
. C:\ESP\ESP_container\v5.5.1\esp-idf\export.ps1
idf.py -C <project-dir> -B <build-dir> build"""


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(f"workspace root : {WS_ROOT}")
    print(f"CAN port       : {describe_can_port()}")
    print(f"CAN wire       : {CAN_INTERFACE} @ {BITRATE} bit/s, tty {TTY_BAUDRATE} baud")
    print(f"mocap          : server {MOCAP_SERVER_IP}, client {MOCAP_CLIENT_IP}, "
          f"multicast {MULTICAST}")
    print(f"kinova         : {KINOVA_IP}")
    print(f"python (base)  : {PY_BASE}  exists={PY_BASE.exists()}")
    print(f"python (ws)    : {PY_WS_VENV}  exists={PY_WS_VENV.exists()}")
    print(f"python (kinova): {PY_KINOVA_VENV}  exists={PY_KINOVA_VENV.exists()}")
    print(f"natnet sdk     : {NATNET_SDK_DIR}  exists={NATNET_SDK_DIR.exists()}")
