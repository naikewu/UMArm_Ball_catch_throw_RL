"""The TLE board's own command set, as opposed to the shared-bus protocol.

The compact frames on the shared bus carry a target and a pressure and nothing
else, because that is all twenty-four actuators have in common. A proportional
board has a great deal more state -- gains, current codes, shaping, dither, the
integral and derivative terms -- and the existing bench tooling already speaks
that command set over USB. It is defined once, in ``vema_proto.py``, together
with the tuned defaults that came off this bench, so this module re-exports it
rather than growing a second copy that would drift. The split is worth keeping:
``vema_proto`` is the bench GUIs' file and carries the measured tuning, while
this module holds only the names ``tlelib`` promises its callers.

Over CAN these frames go to base + 0x400, which the old boards ignore. Over
USB they go to base + 0x100 as they always have.
"""
from __future__ import annotations

# Vendored into this package when the tooling moved into the UMArm workspace.
# In the VEMA_MAX22200 repository this module path-loaded
# ../../vema_gui/vema_proto.py through importlib, because that repository root
# held both a `vema_gui.py` entry script and a `vema_gui/` directory and the
# script won the module name, so `import vema_gui.vema_proto` could not reach
# the file. Here the file sits beside this one, a plain relative import finds
# it, and `tlelib` depends on nothing outside TLE_PCB/.
from . import vema_proto as _vp

# Scaling and enumerations
COUNTS_PER_PSI = _vp.COUNTS_PER_PSI
RAW_0_PSI = _vp.RAW_0_PSI
HOST_MAX_CODE = _vp.HOST_MAX_CODE
DEFAULTS = _vp.DEFAULTS
CHANNEL_LABELS = _vp.CHANNEL_LABELS
PRESSURE_STATES = _vp.PRESSURE_STATES
PRESSURE_ACTIONS = _vp.PRESSURE_ACTIONS
VALVE_MODES = _vp.VALVE_MODES
psi_to_raw = _vp.psi_to_raw
raw_to_psi = _vp.raw_to_psi
psi_span_to_raw = _vp.psi_span_to_raw
code_to_ma = _vp.code_to_ma

# Command codes and payload builders
CMD_REQUEST_TELEMETRY = _vp.CMD_REQUEST_TELEMETRY
CMD_SET_PERIOD_MS = _vp.CMD_SET_PERIOD_MS
CMD_PRESSURE_CONTROL = _vp.CMD_PRESSURE_CONTROL
MODE_STOP = _vp.MODE_STOP
MODE_START = _vp.MODE_START
MODE_SET_TARGET = _vp.MODE_SET_TARGET
build_keepalive = _vp.build_keepalive
build_set_period = _vp.build_set_period
build_control_start = _vp.build_control_start
build_control_set_target = _vp.build_control_set_target
build_control_stop = _vp.build_control_stop
build_dvp_config = _vp.build_dvp_config
build_gains16 = _vp.build_gains16
build_shaping = _vp.build_shaping
build_transient = _vp.build_transient
build_flowshape = _vp.build_flowshape
build_dither = _vp.build_dither
build_adc_filter = _vp.build_adc_filter
build_dvp_misc = _vp.build_dvp_misc

# Telemetry types and the decoder
TLM_PRESSURE_CONTROL = _vp.TLM_PRESSURE_CONTROL
TLM_PRESSURE_DVP = _vp.TLM_PRESSURE_DVP
TLM_DVP_TIMING = _vp.TLM_DVP_TIMING
TLM_DVP_DEBUG = _vp.TLM_DVP_DEBUG
decode_into = _vp.decode_into

# The firmware's own device-ID command. Only reachable over USB or on the
# extended CAN ID; on the shared bus, SET_ID is the old protocol's 0x05 on
# base + 0x100 (proto.build_set_id), which is what the flash tool uses.
CMD_SET_DEVICE_ID = 0x05


def build_set_device_id(base_id: int) -> bytes:
    """Native set-ID: little-endian, unlike the shared-bus command."""
    return bytes([CMD_SET_DEVICE_ID, base_id & 0xFF, (base_id >> 8) & 0xFF])


def tuning_frames(params: dict | None = None) -> list[bytes]:
    """Every payload needed to put a board into a known tuning.

    The order matters: the 0x24 frame carries the legacy 8-bit gains and the
    current-code mapping, and the 16-bit gains in 0x27 then override the gain
    part of it.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    return [
        build_dvp_config(p["open"], p["max"], p["slew"], p["outdb"]),
        build_gains16(p["kp"], p["ki"], p["kd"]),
        build_shaping(p["dither_n"], p["code_jump"], p["dlp"], p["deadtime"],
                      p["dfade_psi"], p["vent_pm"]),
        build_transient(p["vmax_up"], p["vmax_down"], p["dzone_up"], p["dzone_down"],
                        p["outlet_open"], p["outlet_max"], p["ff_scale"]),
        build_flowshape(p["fs"], p["inlet_far_max"], p["decel_up"], p["decel_down"],
                        p["fs_look"], p["fs_seed"], p["fs_rate"]),
        build_dither(p["dither_hz"], p["dither_ma"], p["dither_ch"], p["dither_deep"]),
        build_adc_filter(p["adc_shift"]),
    ]


__all__ = [name for name in dir() if not name.startswith("_")]
