"""Shared protocol + scaling for the split-process VEMA pressure-control GUI.

Both the backend (serial owner) and the control UI import this so the frame
encoding lives in exactly one place. The board speaks the text frame protocol
on its native USB-Serial/JTAG port: ">III:HEX\\n" out, "<III:HEX\\n" in, where
III is the 11-bit CAN id in hex and HEX is the payload. The same IDs/payloads
are used whether the transport is CAN or USB.

Tuning defaults below are the fw 1.40 crack-region values bench-tuned on the
real TLE92464 hardware (2026-06-19): inlet-only hold matching the leak, holds
at the sensor noise floor (<=0.02 % of SP at 15/25 psi), no limit cycle.
"""

from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# Link / addressing
# ---------------------------------------------------------------------------
USB_PORT = "COM8"               # board's native USB-Serial/JTAG port (TLE_ALL_IN_ONE unit #2)
DEVICE_BASE_ID = 0x114
COMMAND_ID = DEVICE_BASE_ID + 0x100
TELEMETRY_ID = DEVICE_BASE_ID + 0x300

# Local backend socket the two UI processes connect to.
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8765

TELEMETRY_PERIOD_MS = 20         # 50 Hz telemetry -> smooth plot
KEEPALIVE_S = 1.0                # board mirrors USB telemetry only while it hears us
TARGET_SEND_HZ = 50              # max SET_TARGET rate while dragging the slider
PLOT_WINDOW_S = 30.0

# ---------------------------------------------------------------------------
# Sensor scaling (LTC1864 raw counts <-> psi)
# ---------------------------------------------------------------------------
RAW_0_PSI = 15100
RAW_40_PSI = 54000
PSI_FULL_SCALE = 40.0
COUNTS_PER_PSI = (RAW_40_PSI - RAW_0_PSI) / PSI_FULL_SCALE  # 972.5

# ---------------------------------------------------------------------------
# Commands / telemetry types (must match pressure_controller.c)
# ---------------------------------------------------------------------------
CMD_ADC_RAW_CAPTURE = 0x10         # fw 1.44: raw 3 kHz ADC capture + ASCII dump
CMD_REQUEST_TELEMETRY = 0x01
CMD_SET_PERIOD_MS = 0x04
CMD_PRESSURE_DVP_CONFIG = 0x24
CMD_PRESSURE_CONTROL = 0x22
CMD_MANUAL_CURRENT = 0x23          # fw 1.44: open-loop constant valve current
CMD_PRESSURE_DVP_GAINS16 = 0x27
CMD_PRESSURE_DVP_SHAPING = 0x28
CMD_PRESSURE_DVP_TRANSIENT = 0x29  # fw 1.41: reference generator + outlet range + FF
CMD_PRESSURE_DVP_FLOWSHAPE = 0x2A  # fw 1.42: flow shaping + make-up seed scale
CMD_PRESSURE_DVP_DITHER = 0x2B     # fw 1.42: TLE92464 hardware dither
CMD_ADC_FILTER = 0x2C              # fw 1.43: runtime ADC EMA depth (lag vs noise)
CMD_DVP_MISC = 0x2D                # fw 1.44: integral floor mode (light apply)
MODE_STOP, MODE_START, MODE_SET_TARGET = 0, 1, 2
VALVE_MODE_REQUEST_DVP = 1

# Manual-current roles (command 0x23).
MANUAL_ROLE_OFF, MANUAL_ROLE_INLET, MANUAL_ROLE_OUTLET = 0, 1, 2
MANUAL_TIMEOUT_MS_DEFAULT = 500     # firmware failsafe window; host re-sends to sustain
MANUAL_PRESSURE_LIMIT_RAW_DEFAULT = 54000  # firmware default when the field is 0

# Integral floor modes (command 0x2D).
INT_FLOOR_NO_CHANGE, INT_FLOOR_ZERO, INT_FLOOR_SYMMETRIC = 0, 1, 2

TLM_PRESSURE_CONTROL = 0x21
TLM_PRESSURE_DVP = 0x22
TLM_DVP_TIMING = 0x23
TLM_DVP_DEBUG = 0x24

# TLE92464 output channels as reported in telemetry frame 0x22 byte[4].
# TLE_ALL_IN_ONE plumbing: inlet = LOAD2 (40 psi supply -> bladder), outlet =
# LOAD3 (bladder -> atmosphere, larger orifice). The legacy V2+breakout jig used
# LOAD0 as the inlet and the MAX22200 era used channel 1 as the outlet -- those
# values only ever show up when re-analysing old logs.
CHANNEL_INLET = 2
CHANNEL_OUTLET = 3
CHANNEL_OFF = 255
CHANNEL_LABELS = {
    CHANNEL_INLET: "inlet", CHANNEL_OUTLET: "outlet", CHANNEL_OFF: "off",
    0: "inlet (legacy)", 1: "outlet (legacy)",
}

# Valve current: I_mA = code * 200/127. The Clippard DVP datasheet hard max is
# 190 mA continuous = code 120. That is the DC + hardware-dither BUDGET: the
# TLE overlays I_peak = steps*step_size*61 uA on top of the DC setpoint, so the
# firmware reserves DITHER_PEAK_MAX_CODE of it for the overlay and clamps every
# host-supplied DC code to HOST_MAX_CODE. Mirror the firmware exactly.
SAFE_MAX_CODE = 120          # DC + dither peak budget (PRESSURE_DVP_SAFE_MAX_CODE)
DITHER_PEAK_MAX_CODE = 4     # reserved for the hardware-dither overlay
HOST_MAX_CODE = SAFE_MAX_CODE - DITHER_PEAK_MAX_CODE
MA_PER_CODE = 200.0 / 127.0
# Hardware-dither scaling (command 0x2B), mirrored from tuning/vema_bench.py:
# peak_lsb = steps*step_size*TLE92464_CODE_MAX/TLE92464_SETPOINT_SAT.
TLE_FSYS_HZ = 28.0e6                  # TLE92464 system clock (datasheet typ)
TLE_DITHER_LSB_MA = 2000.0 / 32767.0  # 0.06104 mA per (steps*step_size)
DITHER_PEAK_MAX_LSB = (DITHER_PEAK_MAX_CODE * 0x0CCD) // 0x7F
# Firmware clamps a manual-drive pressure ceiling into 1..PRESSURE_DVP_FAULT_MAX_RAW.
MANUAL_LIMIT_MAX_RAW = 54000 + 2048


def code_to_ma(code: float) -> float:
    return code * MA_PER_CODE


def ma_to_code(ma: float) -> int:
    return _clamp(ma / MA_PER_CODE, 0, HOST_MAX_CODE)

PRESSURE_STATES = {0: "idle", 1: "running", 2: "settling", 3: "fault", 4: "config error", 5: "stale ADC"}
PRESSURE_ACTIONS = {0: "-", 1: "inlet", 2: "outlet", 3: "coast/deadband", 4: "settling"}
VALVE_MODES = {0: "bang-bang", 1: "proportional"}

# Firmware shaping command units (must match the #defines).
D_FADE_STEP_RAW = 128.0          # PRESSURE_DVP_D_FADE_BAND_STEP_RAW
VENT_STEP_PM = 4.0               # PRESSURE_DVP_VENT_THRESHOLD_STEP_PM

# fw 1.43 bench-tuned defaults (2026-06-20, the current "best tuning"): a HIGH-gain
# derivative-damped PID (like the MAX22200 era), NOT the fw1.40 low-gain crack-band
# + feedforward shaping. High kp keeps the valve fully open until close to target
# (fast smooth slew both directions); kd damps the approach (no overshoot); the
# soft-zone + D-fade + asymmetric outlet keep the hold inlet-only and quiet; the
# low ADC EMA lag (adc_shift=1) lets the loop arrest the slew. The reference ramp
# (vmax_up/down) and flow-shaping (cmd 0x2A) are OFF -- the PID does the shaping.
# 5->30 rise 0.22 s, 30->5 fall 0.35 s, ~1.5 % overshoot, smooth. See
# TLE_tuning_results.md §9.  (Previous fw1.40/1.41 values were kp=160/kd=0/max=70/
# vmax=26,60 -- those gave the slow/kinky transients; do not use them.)
#
# fw 1.45 TLE_ALL_IN_ONE unit #2 bench tuning (2026-08-10) -- SUPERSEDES the
# fw1.43 note above. Same values as tuning/vema_bench.py DEFAULTS; see the long
# rationale there. The short version: this plant leaks 2-3x more than the one the
# firmware's leak make-up model was fitted to, so the integral SEED applied at
# every setpoint jump over-drives by ~2.7x. Only the flow-shaping path (cmd 0x2A)
# can scale that seed, so fs=1 with fs_seed=17, and ff_scale=30 carries the rest
# of the hold burden continuously (fs_seed + ff_scale ~= 47/128 = the measured
# steady drive). Hardware dither is OFF: in-loop it injected a 150 Hz pressure
# line 27x the noise floor and cost 1.6x in hold RMSE for no stiction benefit.
DEFAULTS = dict(
    kp=2048, ki=2500, kd=2500,
    open=51, max=110, slew=4,
    soft_psi=0.05, outdb=0,
    dither_n=4, code_jump=2, dlp=3, deadtime=10,
    dfade_psi=0.5, vent_pm=20,
    vmax_up=0, vmax_down=0, dzone_up=80, dzone_down=30,
    outlet_open=57, outlet_max=96, ff_scale=30,
    adc_shift=3,
    # fw 1.42 flow shaping (cmd 0x2A). fs=1 is required: it is the only path that
    # scales the integral make-up seed (fs_seed/128). decel_up/down = 0 disables
    # the distance-scaled current cap, which was measured to cost more than it
    # saved (it releases discontinuously at the 0.4 psi capture band).
    fs=1, inlet_far_max=95, decel_up=0, decel_down=0, fs_look=0, fs_seed=17, fs_rate=2,
    # fw 1.42 hardware dither (cmd 0x2B). ch bit0=inlet, bit1=outlet;
    # dither_ma=0 disables (the tuned default).
    dither_hz=150.0, dither_ma=0.0, dither_ch=1, dither_deep=0,
    target_psi=0.0,
)


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------
def psi_to_raw(psi: float) -> int:
    return max(0, min(0xFFFF, int(round(RAW_0_PSI + psi * COUNTS_PER_PSI))))


def raw_to_psi(raw: float) -> float:
    return (raw - RAW_0_PSI) / COUNTS_PER_PSI


def psi_span_to_raw(psi: float) -> int:
    return max(1, min(0xFFFF, int(round(psi * COUNTS_PER_PSI))))


def _clamp(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


# ---------------------------------------------------------------------------
# Command payload builders (return raw bytes; the backend prepends the frame id)
# ---------------------------------------------------------------------------
def build_set_period(ms: int) -> bytes:
    return bytes([CMD_SET_PERIOD_MS]) + struct.pack("<H", ms)


def build_dvp_config(open_code: int, max_code: int, slew: int, outdb: int) -> bytes:
    # 255,0,0 = neutral legacy gains; the 16-bit gains in the 0x27 frame win.
    return bytes([CMD_PRESSURE_DVP_CONFIG, 255, 0, 0,
                  _clamp(open_code, 1, HOST_MAX_CODE), _clamp(max_code, 1, HOST_MAX_CODE),
                  _clamp(slew, 1, 127), _clamp(outdb, 0, 255)])


def build_gains16(kp: int, ki: int, kd: int) -> bytes:
    return bytes([CMD_PRESSURE_DVP_GAINS16]) + struct.pack(
        "<HHH", _clamp(kp, 1, 65535), _clamp(ki, 0, 65535), _clamp(kd, 0, 65535))


def build_shaping(dither_n: int, code_jump: int, dlp: int, deadtime: int,
                  dfade_psi: float, vent_pm: float) -> bytes:
    d_fade_lsb = _clamp(dfade_psi * COUNTS_PER_PSI / D_FADE_STEP_RAW, 0, 255)
    vent_lsb = _clamp(vent_pm / VENT_STEP_PM, 0, 255)
    return bytes([CMD_PRESSURE_DVP_SHAPING,
                  _clamp(dither_n, 0, 8), _clamp(code_jump, 0, 127),
                  _clamp(dlp, 0, 64), _clamp(deadtime, 0, 64),
                  d_fade_lsb, vent_lsb])


def build_transient(vmax_up: int, vmax_down: int, dzone_up: int, dzone_down: int,
                    outlet_open: int, outlet_max: int, ff_scale: int) -> bytes:
    return bytes([CMD_PRESSURE_DVP_TRANSIENT,
                  _clamp(vmax_up, 0, 255), _clamp(vmax_down, 0, 255),
                  _clamp(dzone_up, 0, 255), _clamp(dzone_down, 0, 255),
                  _clamp(outlet_open, 1, HOST_MAX_CODE), _clamp(outlet_max, 1, HOST_MAX_CODE),
                  _clamp(ff_scale, 0, 255)])


def build_adc_filter(shift: int) -> bytes:
    return bytes([CMD_ADC_FILTER, _clamp(shift, 0, 6)])


# fw 1.42 flow shaping (0x2A) and hardware dither (0x2B). The GUI previously did
# not send either, so it silently ran the firmware boot defaults (fs OFF, dither
# ON) no matter what DEFAULTS said. Both are load-bearing for the 2026-08-10
# tuning -- fs=1 is what scales the integral make-up seed, and dither must be off
# -- so the apply path now sends them. Packing mirrors tuning/vema_bench.py.
def build_flowshape(enable: int, inlet_far_max: int, decel_up: int, decel_down: int,
                    lookahead: int, seed_scale: int, rate_thresh: int) -> bytes:
    return bytes([CMD_PRESSURE_DVP_FLOWSHAPE,
                  1 if enable else 0,
                  _clamp(inlet_far_max, 0, HOST_MAX_CODE),
                  _clamp(decel_up, 0, 255), _clamp(decel_down, 0, 255),
                  _clamp(lookahead, 0, 255), _clamp(seed_scale, 0, 255),
                  _clamp(rate_thresh, 0, 255)])


def dither_params(hz: float, ma: float, steps: int = 8, flat: int = 0) -> dict:
    """Friendly (Hz, mA-peak) -> raw TLE dither fields.
    I = steps*step_size*0.06104 mA ; T = (4*steps+2*flat)*mant*2^exp/fSYS."""
    if ma <= 0.0 or hz <= 0.0:
        return dict(steps=steps, flat=flat, step_size=0, mant=1, exp=0)
    step_size = int(round(ma / TLE_DITHER_LSB_MA / max(1, steps)))
    step_size = max(1, min(4095, DITHER_PEAK_MAX_LSB // max(1, steps), step_size))
    mant_raw = TLE_FSYS_HZ / (hz * (4 * steps + 2 * flat))
    exp = 0
    while mant_raw / (2 ** exp) > 1023 and exp < 15:
        exp += 1
    mant = max(1, min(1023, int(round(mant_raw / (2 ** exp)))))
    return dict(steps=steps, flat=flat, step_size=step_size, mant=mant, exp=exp)


def build_dither(hz: float, ma: float, ch: int = 1, deep: int = 0) -> bytes:
    d = dither_params(hz, ma)
    return bytes([CMD_PRESSURE_DVP_DITHER,
                  (int(ch) & 0x03) | (0x80 if deep else 0x00),
                  d["steps"] & 0xFF, d["flat"] & 0xFF,
                  d["step_size"] & 0xFF, (d["step_size"] >> 8) & 0x0F,
                  d["mant"] & 0xFF, ((d["mant"] >> 8) & 0x03) | ((d["exp"] & 0x0F) << 2)])


def build_manual_current(role: int, code: int,
                         timeout_ms: int = MANUAL_TIMEOUT_MS_DEFAULT,
                         pressure_limit_raw: int = 0) -> bytes:
    """fw 1.44 open-loop valve drive (command 0x23).

    role 0 = both valves off, 1 = inlet, 2 = outlet. `code` is the constant
    current code, clamped to HOST_MAX_CODE (the DC share of the 189 mA budget).
    The firmware drops the valves again timeout_ms after the last 0x23, so the
    host must re-send (~every 300 ms for the 500 ms default) to sustain a drive
    -- that is the dead-host failsafe. pressure_limit_raw (0 -> firmware default
    54000 = 40 psi) trips the valves off while role=1 (inlet) using the filtered
    pressure; a non-zero value is clamped into 1..MANUAL_LIMIT_MAX_RAW on both
    sides so it can never name a ceiling above the ADC range (= a guard that
    never trips). Only accepted while the controller is not RUNNING/SETTLING.
    """
    limit = int(pressure_limit_raw)
    limit = 0 if limit <= 0 else _clamp(limit, 1, MANUAL_LIMIT_MAX_RAW)
    return (bytes([CMD_MANUAL_CURRENT, _clamp(role, 0, 2), _clamp(code, 0, HOST_MAX_CODE)]) +
            struct.pack("<HH", _clamp(timeout_ms, 0, 65535), limit))


def build_adc_capture(n_samples: int) -> bytes:
    """fw 1.44 raw ADC capture (command 0x10): n consecutive unfiltered 16-bit
    LTC1864 samples at 3 kHz, dumped as '#RAW,<idx>,v0,..' console lines then
    '#RAWEND,<n>,<sample_hz>'. 0 -> firmware default 3000; clamped 100..12000."""
    return bytes([CMD_ADC_RAW_CAPTURE]) + struct.pack("<H", _clamp(n_samples, 0, 65535))


def build_dvp_misc(integral_floor_mode: int) -> bytes:
    """fw 1.44 misc controller tweak (command 0x2D). mode 0 = no change,
    1 = floor the integral at 0 (legacy leaky-plant behaviour), 2 = symmetric
    floor at -integral_limit. Light apply: does not reset the controller."""
    return bytes([CMD_DVP_MISC, _clamp(integral_floor_mode, 0, 2)])


def build_control_start(target_raw: int, deadband_raw: int) -> bytes:
    return (bytes([CMD_PRESSURE_CONTROL, MODE_START]) +
            struct.pack("<HH", target_raw, deadband_raw) +
            bytes([VALVE_MODE_REQUEST_DVP, 0]))


def build_control_set_target(target_raw: int, deadband_raw: int) -> bytes:
    return bytes([CMD_PRESSURE_CONTROL, MODE_SET_TARGET]) + struct.pack("<HH", target_raw, deadband_raw)


def build_control_stop() -> bytes:
    return bytes([CMD_PRESSURE_CONTROL, MODE_STOP])


def build_keepalive() -> bytes:
    return bytes([CMD_REQUEST_TELEMETRY])


# ---------------------------------------------------------------------------
# Telemetry decode -> merge into a latest-state dict
# ---------------------------------------------------------------------------
def decode_into(data: bytes, state: dict) -> bool:
    """Merge one telemetry payload into `state`. Returns True if it was the
    main pressure frame (0x21), i.e. a new plottable sample."""
    if not data:
        return False
    kind = data[0]
    if kind == TLM_PRESSURE_CONTROL and len(data) >= 8:
        state["state"] = data[1]
        state["action"] = data[2]
        state["target_raw"] = struct.unpack_from("<H", data, 3)[0]
        state["pressure_raw"] = struct.unpack_from("<H", data, 5)[0]
        return True
    if kind == TLM_PRESSURE_DVP and len(data) >= 8:
        state["permille"] = struct.unpack_from("<h", data, 1)[0]
        state["code"] = data[3]
        state["channel"] = data[4]
        state["valve_mode"] = data[5] & 0x0F
        state["armed"] = (data[5] & 0x80) != 0
        state["fault_pin"] = (data[6] & 0x80) != 0
        state["error_code"] = data[7]
    elif kind == TLM_DVP_TIMING and len(data) >= 8:
        state["late_updates"] = struct.unpack_from("<H", data, 3)[0]
        state["max_service_us"] = struct.unpack_from("<H", data, 5)[0]
    elif kind == TLM_DVP_DEBUG and len(data) >= 8:
        state["i_pm"] = struct.unpack_from("<h", data, 1)[0]
        state["d_pm"] = struct.unpack_from("<h", data, 3)[0]
    return False
