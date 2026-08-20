"""Host library for the TLE boards on the shared 24-actuator CAN bus.

Layering, which the two GUIs rely on:

    proto     wire formats and addressing, no I/O
    canlink   the slcan adapter, a receive thread, node discovery
    usblink   one board over its USB-Serial/JTAG port, same frames as CAN
    native    the TLE board's own command set (re-exported from vema_proto)
    ota       broadcast firmware update
    usbflash  esptool over USB
    backend   the 150 Hz sync master and the multi-board registry
    wsenv     optional workspace defaults from <workspace>/bench_env.py

Nothing above `backend` touches a serial port, and nothing in the library
knows what a widget is. `vema_proto` is vendored here rather than imported
from a sibling project, so this package depends on nothing outside TLE_PCB/.
"""
from . import native, proto  # noqa: F401
from .backend import Backend, NodeState  # noqa: F401
from .canlink import CanLink, SlcanError  # noqa: F401
from .ota import BroadcastOta, OtaCancelled, load_image  # noqa: F401
from .usbflash import UsbFlasher, find_build  # noqa: F401
from .usblink import UsbLink  # noqa: F401

__all__ = [
    "Backend", "BroadcastOta", "CanLink", "NodeState", "OtaCancelled",
    "SlcanError", "UsbFlasher", "UsbLink", "find_build", "load_image",
    "native", "proto",
]
