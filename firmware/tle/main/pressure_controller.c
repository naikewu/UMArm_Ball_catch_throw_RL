/*
 * VEMA V2 PCB pressure controller.
 *
 * Two independent SPI buses:
 *   SPI2: MCP25625 CAN controller (pins unchanged from V1).
 *   SPI3: MAX22200 valve driver + LTC1864 ADC (shared SCK/MISO, see
 *         shared_spi_bus.c; the LTC1864 SDO is gated onto the shared MISO by
 *         a 74LVC1G125 buffer whose OE# is the CONV line).
 *
 * Core split:
 *   Core 0: can_task (interrupt-driven, low latency) + pressure_pid_task.
 *   Core 1: adc_sample_task + valve_task -- dedicated to the SPI3
 *           transactions (LTC1864 sampling and MAX22200 current updates).
 *
 * The MAX22200 is operated write-only: the V2 board has no pull-up on its
 * SDO line, so register readback does not work and is never used. Chip
 * health is monitored through the active-low FAULT pin.
 */

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "board_pins.h"
#include "can_ota.h"
#include "driver/gpio.h"
#include "driver/gptimer.h"
#include "driver/spi_master.h"
#include "esp_app_desc.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_ota_ops.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ltc1864.h"
#include "mcp2515_min.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "shared_spi_bus.h"
#include "solenoid.h"
#include "tle92464.h" /* TLE92464_SETPOINT_SAT/CODE_MAX: hardware-dither headroom */
#include "tle_can_legacy.h"
#include "tle_verify.h"
#include "usb_link.h"

/* =====================================================================
 * User-adjustable configuration (no CMake flags -- edit and rebuild)
 * ===================================================================== */

/* PCB revision this file targets. The V1 PCB (MAX22200 sharing the CAN SPI
 * bus, bit-banged LTC1864) is served by the firmware on git branch `main`
 * (commit 0e001ca); this file only supports the V2 board. */
#define VEMA_PCB_VERSION 2

/* Firmware version string (printed at boot, quoted in the tuning logs).
 * 1.44: TLE_ALL_IN_ONE bring-up -- board-conditional inlet channel, Clippard
 * DVP current ceiling (code 120 = 189 mA), manual-current command 0x23, raw ADC
 * capture 0x10 and the runtime integral-floor switch 0x2D. */
#define VEMA_FIRMWARE_VERSION "1.46"

/* Control loop frequencies. The base tick drives the other three; each must
 * divide CONTROL_BASE_TICK_HZ exactly (checked below). */
#define CONTROL_BASE_TICK_HZ 6000U      /* GPTimer base tick */
#define ADC_SAMPLE_HZ 3000U             /* LTC1864 sample rate (core 1) */
#define PRESSURE_PID_HZ 750U            /* PID update rate (core 0) */
#define MAX_CURRENT_UPDATE_HZ 1500U     /* MAX22200 current update rate (core 1) */

/* Boot-default control mode; the GUIs can switch it at runtime through the
 * start frame (PRESSURE_CONTROL byte 6). */
#define PRESSURE_VALVE_MODE_BANGBANG 0U /* on/off valves, pulsed (V1 "7mm") */
#define PRESSURE_VALVE_MODE_DVP 1U      /* proportional valves, PI current control */
#define PRESSURE_VALVE_MODE_DEFAULT PRESSURE_VALVE_MODE_DVP

/* Solenoid driver chip selection. The V2 PCB has the original MAX22200; the
 * extension breakout adds a TLE92464 on the same SPI3 bus (CS=GPIO11). Default
 * to the TLE92464 (15-bit current control, full-resolution code, no software
 * dither). Set to VEMA_SOLENOID_DRIVER_MAX22200 to fall back to the MAX22200.
 * (See solenoid.h / tle92464.h.) */
#define VEMA_SOLENOID_DRIVER VEMA_SOLENOID_DRIVER_TLE92464

/* Verification build: when 1, app_main runs the TLE92464 / ADC / CAN
 * communication self-test (tle_verify.c) and never starts the pressure
 * controller. Use this for the unpowered comms bring-up on COM4; set back to 0
 * for normal operation. */
#define VEMA_TLE_VERIFY 0

/* ADC filtering: median outlier rejection over the per-PID window, then a
 * fixed-gain (steady-state Kalman) exponential average with gain 1/2^shift.
 * fw 1.38: shift 2 -> 3 (TC ~10.7 ms) and reject 64 -> 32 raw. The plant's
 * flow noise is ~20-100 mpsi broadband; the deeper filter halves the noise
 * fed to the PID so the valve does not chew on it. Worst legitimate slew
 * (~15 psi/s step) moves ~19 raw per 1.33 ms window, well under the reject
 * threshold. */
#define ADC_AVG_SHIFT 3U
#define ADC_AVG_SHIFT_MAX 6U   /* runtime cap (command 0x2C); EMA TC ~= 2^shift / PID_HZ */
#define ADC_FILTER_REJECT_DELTA_RAW 32U
#define ADC_FILTER_MIN_ACCEPTED_SAMPLES 3U
#define ADC_FILTER_TRIM_SAMPLES 2U

/* CAN node addressing (base persisted in NVS, set via command 0x05).
 *
 * The default is the first of the eight actuator slots this board type takes
 * over on the shared 24-actuator bus (0x101..0x108; the sixteen 7 mm boards
 * keep 0x109..0x118). Every board still has to be given its own ID before it
 * joins the bus -- over USB with the flash GUI, or over CAN with the old
 * protocol's SET_ID -- but defaulting into the right block means a board that
 * has never been configured cannot masquerade as one of the old sixteen.
 *
 * On CAN, base + 0x100 belongs to the old protocol (see tle_can_legacy.h), so
 * this project's own command set gets an ID of its own at base + 0x400. Over
 * USB, where nothing is shared, base + 0x100 keeps its original meaning and
 * every existing bench tool still works unchanged. */
#define CAN_ID_DEVICE_BASE_DEFAULT 0x101U
#define CAN_ID_HOST_COMMAND_OFFSET 0x100U
#define CAN_ID_HOST_OTA_DATA_OFFSET 0x200U
#define CAN_ID_DEVICE_STATUS_OFFSET 0x300U
#define CAN_ID_EXTENDED_COMMAND_OFFSET 0x400U
#define CAN_ID_BROADCAST 0x090U

/* Wire speed. The production bus runs at 1 Mbit/s, which is what the sixteen
 * old boards are configured for, so this is not independently choosable per
 * board -- it is a property of the bus. 500 kbit/s is offered for a longer or
 * noisier harness, and every board on the bus then has to be rebuilt for it.
 * Bus load at 150 Hz for 24 actuators is about 12 % at 1 Mbit/s. */
#ifndef CAN_BUS_BITRATE
#define CAN_BUS_BITRATE MCP2515_BITRATE_1MBPS
#endif

/* Telemetry pacing. */
#define TELEMETRY_DEFAULT_PERIOD_MS 20U
#define TELEMETRY_PERIOD_MIN_MS 10U
#define TELEMETRY_PERIOD_MAX_MS 5000U

/* SPI clocks. The MAX/LTC bus runs slow for signal-integrity margin on the
 * shared MISO net; CAN has its own bus and stays fast. */
#define MAX22200_SPI_CLOCK_HZ (1U * 1000U * 1000U)
#define LTC1864_SPI_CLOCK_HZ (1U * 1000U * 1000U)
#define MCP25625_SPI_CLOCK_HZ (10U * 1000U * 1000U)
/* TLE92464: <=8 MHz. Brought up at 1 MHz, but the shared multi-drop MISO net
 * has no pull-up and corrupts TLE readbacks (impossible/garbled GLOBAL_DIAG0,
 * intermittent FB_VOLTAGE) -- see TLE_verification_report. Dropped to 500 kHz
 * for read signal-integrity margin; control tolerates it (writes are CRC-armed
 * and the driver never read-modify-writes). The mission-mode/diag readbacks
 * need to be trustworthy, so this matters. */
#define TLE92464_SPI_CLOCK_HZ (500U * 1000U)

/* =====================================================================
 * Fixed protocol / control constants (port of the V1 controller)
 * ===================================================================== */

#if VEMA_PCB_VERSION != 2
#error "This firmware targets the V2 PCB only. V1 PCB firmware lives on git branch main (commit 0e001ca)."
#endif

#define CAN_SPI_HOST SPI2_HOST
#define MAXLTC_SPI_HOST SPI3_HOST
#define CAN_SERVICE_FALLBACK_PERIOD_MS 1U
/* Both MCP receive buffers, plus headroom for frames that arrive while the
 * pass is running. Bounded so a busy bus cannot hold the CAN task forever. */
#define CAN_RX_SERVICE_MAX_FRAMES 6U
#define CAN_TX_QUEUE_DEPTH 16U
/* Frames a single CAN service pass will try to push out. */
#define CAN_TX_BURST_MAX_FRAMES 4U
#define VALVE_SERVICE_PERIOD_MS 1U

/* CAN link breaker. With no bus attached nothing ever acknowledges the
 * MCP25625: the loaded frame is retried forever, TXB0 never frees, the TX
 * queue fills, and can_task (priority 7, core 0) then spends its life on retry
 * SPI plus one console warning per dropped frame -- which starves the USB
 * parser (priority 3) and IDLE0 until the task watchdog fires. So: after this
 * many consecutive busy TX polls spanning at least this long, declare the link
 * down, abort the stuck transmission and drop the CAN copies until a re-probe
 * proves the bus is back. A live bus frees TXB0 within a frame time (~130 us at
 * 1 Mbit/s), so the two thresholds together cannot trip on bus load. */
#define CAN_LINK_DOWN_BUSY_POLLS 25U
#define CAN_LINK_DOWN_BUSY_US (250LL * 1000)
#define CAN_LINK_REPROBE_PERIOD_US (10LL * 1000 * 1000)
#define CAN_LINK_PROBE_TIMEOUT_US (20LL * 1000)
/* Rate limit for the TX-queue-full warning: it used to print once per dropped
 * frame, i.e. 4 lines per telemetry burst (~200/s at the 20 ms default). */
#define CAN_QUEUE_FULL_LOG_PERIOD_US (5LL * 1000 * 1000)
/* Unconditional EFLG sweep. 100 ms is 15 cycles of the 150 Hz bus: short
 * enough that a latched receive-buffer overflow costs a fraction of a second
 * of feedback rather than the rest of the session, and slow enough that the
 * extra SPI read is nothing next to the per-frame traffic. */
#define CAN_EFLG_SWEEP_PERIOD_US (100LL * 1000)
/* Rate limit for the RX-overflow warning. The console is a blocking writer
 * and this runs at priority 7 on core 0, so logging every event would hold
 * the CAN task open exactly while the overflow it reports keeps recurring --
 * the same positive feedback the TX-queue-full warning already avoids. */
#define CAN_RX_OVERFLOW_LOG_PERIOD_US (5LL * 1000 * 1000)
/* Defence in depth: after this many consecutive wakes where the notify wait
 * returned a notification instead of timing out, can_task yields the core
 * outright so lower-priority tasks and IDLE0 always run. Healthy operation
 * never comes close -- notifies are sparse and the 1 ms wait times out. */
#define CAN_TASK_BREATHER_ITERATIONS 200U

#define CONTROL_TIMER_RESOLUTION_HZ 1200000U
#define CONTROL_BASE_TICK_PERIOD_TICKS (CONTROL_TIMER_RESOLUTION_HZ / CONTROL_BASE_TICK_HZ)
#define ADC_TICK_DIVIDER (CONTROL_BASE_TICK_HZ / ADC_SAMPLE_HZ)
#define PRESSURE_PID_TICK_DIVIDER (CONTROL_BASE_TICK_HZ / PRESSURE_PID_HZ)
#define MAX_UPDATE_TICK_DIVIDER (CONTROL_BASE_TICK_HZ / MAX_CURRENT_UPDATE_HZ)
#define ADC_SAMPLES_PER_PID (ADC_SAMPLE_HZ / PRESSURE_PID_HZ)
#define ADC_PID_WINDOW_SAMPLES ADC_SAMPLES_PER_PID

#if (CONTROL_TIMER_RESOLUTION_HZ % CONTROL_BASE_TICK_HZ) != 0U
#error "CONTROL_BASE_TICK_HZ must divide CONTROL_TIMER_RESOLUTION_HZ"
#endif
#if (CONTROL_BASE_TICK_HZ % ADC_SAMPLE_HZ) != 0U
#error "ADC_SAMPLE_HZ must divide CONTROL_BASE_TICK_HZ"
#endif
#if (CONTROL_BASE_TICK_HZ % PRESSURE_PID_HZ) != 0U
#error "PRESSURE_PID_HZ must divide CONTROL_BASE_TICK_HZ"
#endif
#if (CONTROL_BASE_TICK_HZ % MAX_CURRENT_UPDATE_HZ) != 0U
#error "MAX_CURRENT_UPDATE_HZ must divide CONTROL_BASE_TICK_HZ"
#endif
#if (ADC_SAMPLE_HZ % PRESSURE_PID_HZ) != 0U
#error "PRESSURE_PID_HZ must divide ADC_SAMPLE_HZ"
#endif

#define CAN_ID_NVS_NAMESPACE "config"
#define CAN_ID_NVS_KEY "can_id"

/* Device -> host telemetry frame types (payload byte 0). */
#define TELEMETRY_TYPE_COMMAND_STATUS 0x06U
#define TELEMETRY_TYPE_PRESSURE_CONTROL 0x21U
#define TELEMETRY_TYPE_PRESSURE_DVP 0x22U
#define TELEMETRY_TYPE_PRESSURE_DVP_TIMING 0x23U
#define TELEMETRY_TYPE_PRESSURE_DVP_DEBUG 0x24U /* fw 1.38: I/D terms + gain schedule */
#define TELEMETRY_TYPE_OTA_STATUS 0x30U

/* Host -> device commands (payload byte 0). */
#define COMMAND_REQUEST_TELEMETRY 0x01U
#define COMMAND_SET_PERIOD_MS 0x04U
#define COMMAND_SET_DEVICE_ID 0x05U
#define COMMAND_PRESSURE_CONFIG 0x21U
#define COMMAND_PRESSURE_CONTROL 0x22U
#define COMMAND_PRESSURE_DVP_CONFIG 0x24U
#define COMMAND_PRESSURE_PENDING_TARGET 0x25U
#define COMMAND_SYNC 0x26U
#define COMMAND_PRESSURE_DVP_GAINS16 0x27U
#define COMMAND_PRESSURE_DVP_SHAPING 0x28U
#define COMMAND_PRESSURE_DVP_TRANSIENT 0x29U  /* fw 1.41: reference vmax/dzone + outlet range */
#define COMMAND_PRESSURE_DVP_FLOWSHAPE 0x2AU  /* fw 1.42: distance-scaled current (bang-bang->PID) */
#define COMMAND_PRESSURE_DVP_DITHER 0x2BU     /* fw 1.42: TLE92464 hardware dither config */
#define COMMAND_ADC_FILTER 0x2CU              /* fw 1.42: runtime ADC EMA depth (lag vs noise) */
#define COMMAND_PRESSURE_DVP_MISC 0x2DU       /* fw 1.44: integral-floor mode (light apply) */
#define COMMAND_ADC_RAW_CAPTURE 0x10U         /* fw 1.44: raw 3 kHz LTC1864 capture + console dump */
#define COMMAND_MANUAL_CURRENT 0x23U          /* fw 1.44: open-loop constant valve current (bench) */
#define COMMAND_OTA_START 0x30U
#define COMMAND_OTA_END 0x31U
#define COMMAND_OTA_ABORT 0x32U

#define COMMAND_STATUS_OK 0U
#define COMMAND_STATUS_DISABLED 1U
#define COMMAND_STATUS_INVALID 2U
#define COMMAND_STATUS_BUSY 3U

#define PRESSURE_CONTROL_MODE_STOP 0U
#define PRESSURE_CONTROL_MODE_START 1U
#define PRESSURE_CONTROL_MODE_SET_TARGET 2U

/* PRESSURE_CONTROL start-frame byte 6: requested valve mode. */
#define VALVE_MODE_REQUEST_KEEP 0U
#define VALVE_MODE_REQUEST_DVP 1U
#define VALVE_MODE_REQUEST_BANGBANG 2U

/* Valve channels, per board variant. All four TLE channels are put in ICC mode
 * at init, so a non-contiguous pair is fine.
 *   TLE_ALL_IN_ONE (this bench, 2026-08-09): inlet = channel 2 (40 PSI supply
 *     -> bladder), outlet = channel 3 (bladder -> atmosphere, larger orifice).
 *   Legacy V2 PCB + TLE92464 breakout jig (2026-06-18): inlet on LOAD0 (ch0),
 *     outlet on LOAD3 = load header H11 (ch3). (The older MAX22200 jig used
 *     SL0/SL1.) */
#if VEMA_BOARD_TLE_ALL_IN_ONE
#define VALVE_CHANNEL_INLET 2U
#define VALVE_CHANNEL_OUTLET 3U
#else
#define VALVE_CHANNEL_INLET 0U
#define VALVE_CHANNEL_OUTLET 3U
#endif
#define VALVE_CHANNEL_NONE 0xFFU

/* Bang-bang (on/off valve) defaults. */
#define PRESSURE_CONFIG_FLAG_SENSOR_INCREASES 0x01U
#define PRESSURE_CONFIG_FLAG_FAST_TRACK 0x02U
#define PRESSURE_DEFAULT_FLAGS (PRESSURE_CONFIG_FLAG_SENSOR_INCREASES | PRESSURE_CONFIG_FLAG_FAST_TRACK)
#define PRESSURE_DEFAULT_TARGET_RAW 15100U
#define PRESSURE_DEFAULT_DEADBAND_RAW 49U   /* ~0.05 psi; also the DVP soft-zone width (P/I fade near target -> high-kp hold stays quiet) */
#define PRESSURE_DEFAULT_HIT_CURRENT 80U
#define PRESSURE_DEFAULT_HOLD_CURRENT 25U
#define PRESSURE_DEFAULT_HIT_TIME 0U
#define PRESSURE_DEFAULT_MIN_PULSE_MS 1U
#define PRESSURE_DEFAULT_MAX_PULSE_MS 10U
#define PRESSURE_DEFAULT_SETTLE_MS 2U
#define PRESSURE_MIN_PULSE_MS 1U
#define PRESSURE_MAX_PULSE_MS 50U
#define PRESSURE_MIN_SETTLE_MS 1U
#define PRESSURE_MAX_SETTLE_MS 100U
#define PRESSURE_ADC_STALE_TIMEOUT_MS 120U
#define PRESSURE_FAULT_DEBOUNCE_COUNT 5U

/* Proportional (DVP) defaults. The legacy 8-bit values are kept for the 0x24
 * fallback mapping; the 16-bit values are the fw 1.38 bench-tuned gains
 * (2026-06-11, "gains at 5 psi" -- the pressure schedule scales them). */
#define PRESSURE_DVP_DEFAULT_KP_Q8 255U
#define PRESSURE_DVP_DEFAULT_KI_Q8 12U
#define PRESSURE_DVP_DEFAULT_KD_Q8 0U
/* fw 1.43 TLE92464 high-gain damped-PID tuning (2026-06-20, bench-tuned on real
 * hardware). SUPERSEDES the fw1.40 "crack-region low-gain + feedforward shaping"
 * approach. Root cause found in round 3: the fw1.40 13x gain drop + kd=0 made the
 * output DE-SATURATE far from target (sluggish, kinky transients; the outlet
 * cut early -> the "8 psi fall crawl"), and the feedforward shaping (ref-gen /
 * flow-shaping) added visible deceleration kinks. The MAX22200 era got smooth,
 * fast (0.19 s), low-overshoot steps with a HIGH-gain (kp 2048) DERIVATIVE-DAMPED
 * PID -- so we restore that on the TLE. The crack-band MAPPING (open=46) is kept,
 * but max is raised (110 = ~173 mA) and the gains/D restored: high kp keeps the
 * valve fully open until close to target (fast slew, both directions), kd damps
 * the approach (no overshoot), and the soft-zone + D-fade + asymmetric outlet
 * keep the hold inlet-only and quiet (verified: 0 switching, <=0.004 psi RMSE).
 * The lower ADC EMA lag (shift 1, ~2.7 ms) lets the loop arrest the fast slew.
 * Result: 5->30 rise 0.22 s, 30->5 fall 0.35 s (both ~5x the fw1.41/old speed),
 * ~1.5 % overshoot, smooth. Flow-shaping (cmd 0x2A) is kept but default OFF. */
#define PRESSURE_DVP_DEFAULT_KP16 2048U
#define PRESSURE_DVP_DEFAULT_KI16 2500U
#define PRESSURE_DVP_DEFAULT_KD16 2500U /* fw1.46: de-tuned for 10x noisier ADC; kd>=2500 needed for small-step damping */
/* fw 1.44 Clippard DVP mapping (I_mA = code * 200/127): the valve cracks at
 * ~60 mA (code 38) and saturates at ~180 mA (code 114). */
#define PRESSURE_DVP_DEFAULT_OPEN_CODE 51U /* measured inlet crack 80.3 mA (2026-08-10 sweep); codes 38..50 are dead */
/* HARD current ceiling for every host-supplied code. The Clippard DVP datasheet
 * limit is 190 mA continuous, so the firmware never emits more than code 120
 * (= 189 mA) no matter what the host asks for -- the full-scale code 127
 * (200 mA) would exceed the coil rating. The inlet/outlet are driven up to
 * their configured max for a fast slew; the hold sits inlet-only around the
 * leak make-up well below max via the integral + soft-zone. */
#define PRESSURE_DVP_SAFE_MAX_CODE 120U
/* ...but the DC setpoint is not the whole coil current: the TLE92464's hardware
 * dither generator overlays a triangular waveform ON TOP of the SETPOINT
 * register (I_peak = steps * step_size * 61.04 uA, see tle92464.h), and the
 * inlet carries that overlay by default (chmask 0x01, 8 x 12 = 5.86 mA peak).
 * A DC code of 120 with the default overlay would peak at ~195 mA, i.e. above
 * the very 190 mA limit SAFE_MAX_CODE exists to enforce. So SAFE_MAX_CODE is
 * the DC+dither BUDGET: a fixed slice of it is reserved for the overlay
 * (command 0x2B is bounded to that slice) and every host-supplied DC code is
 * clamped to the remainder. 116 = 182.7 mA DC is still comfortably above the
 * ~180 mA valve saturation, so no control authority is lost. */
#define PRESSURE_DVP_DITHER_PEAK_MAX_CODE 4U /* ~6.3 mA reserved for the overlay */
#define PRESSURE_DVP_HOST_MAX_CODE (PRESSURE_DVP_SAFE_MAX_CODE - PRESSURE_DVP_DITHER_PEAK_MAX_CODE)
/* Dither amplitude budget in TLE setpoint LSBs:
 *   peak_code = steps * step_size * TLE92464_CODE_MAX / TLE92464_SETPOINT_SAT */
#define PRESSURE_DVP_DITHER_PEAK_MAX_LSB \
    (((uint32_t)PRESSURE_DVP_DITHER_PEAK_MAX_CODE * TLE92464_SETPOINT_SAT) / TLE92464_CODE_MAX)
#define PRESSURE_DVP_DEFAULT_MAX_CODE 110U
#define PRESSURE_DVP_DEFAULT_SLEW_CODE 4U
#define PRESSURE_DVP_DEFAULT_OUTPUT_DEADBAND 0U
#define PRESSURE_DVP_SOFT_ZONE_MIN_GAIN_Q8 64U
#define PRESSURE_DVP_CONTROL_RAW_SCALE 4096L
/* fw 1.41 leak make-up feedforward, fitted to the measured steady hold integral
 * (514 raw @5 psi, 2261 @25 psi): ff_raw = 77 + 23*(target-0psi)>>8. Used two
 * ways: (a) optional CONTINUOUS feedforward (dvp_ff_scale, default OFF -- it
 * fights sine descents), and (b) the one-shot integral SEED at step capture
 * (always on), which makes the slew->hold handoff bumpless: no post-step sag,
 * undershoot, or ring. */
#define PRESSURE_DVP_FF_BASE_RAW 77L
#define PRESSURE_DVP_FF_GAIN_Q8 23L
#define PRESSURE_DVP_FF_SCALE_UNITY 128U
#define PRESSURE_DVP_ZERO_PSI_RAW 15100L   /* sensor raw at 0 psi (gauge) */
/* fw 1.38: the integral is held in a Q16 accumulator (no per-step truncation)
 * and clamped to +/-2x full output instead of the former 1000x, so a leak
 * feed survives but a saturation phase cannot wind up minutes of integral. */
#define PRESSURE_DVP_INTEGRAL_LIMIT_RAW (PRESSURE_DVP_CONTROL_RAW_SCALE * 2L)
#define PRESSURE_DVP_INTEGRAL_FRAC_BITS 16
#define PRESSURE_DVP_INTEGRAL_LIMIT_Q16 ((int64_t)PRESSURE_DVP_INTEGRAL_LIMIT_RAW << PRESSURE_DVP_INTEGRAL_FRAC_BITS)
/* fw 1.40: integral floor. On this leak-down plant the steady control output is
 * ALWAYS a positive inlet feed (the inlet meters the leak); active venting is
 * the P-term driving the outlet, not the integral. So the integral must never
 * wind negative -- otherwise a long down-step vent winds it deeply negative and
 * it then has to swing all the way back to the (positive) leak make-up,
 * undershooting and ringing on arrival. Clamping the floor at 0 makes the
 * vent->hold handoff immediate. (Set <0 to restore a symmetric integrator.) */
#define PRESSURE_DVP_INTEGRAL_FLOOR_Q16 ((int64_t)0)
/* fw 1.44: the floor is switchable at runtime (command 0x2D) so the assumption
 * above can be A/B'd on a plumbing setup whose outlet is not a pure leak (this
 * bench's outlet valve has a larger orifice than the inlet). Mode 1 = the legacy
 * floor at 0, mode 2 = a symmetric integrator floored at -integral_limit. */
#define PRESSURE_DVP_INT_FLOOR_MODE_KEEP 0U
#define PRESSURE_DVP_INT_FLOOR_MODE_ZERO 1U
#define PRESSURE_DVP_INT_FLOOR_MODE_SYMMETRIC 2U
#define PRESSURE_DVP_INT_FLOOR_MODE_DEFAULT PRESSURE_DVP_INT_FLOOR_MODE_ZERO

/* The steady make-up integral for a target raw (the FF model), clamped to the
 * integral range. Used to seed the integrator at step capture for a bumpless
 * slew->hold handoff. */
static inline int64_t dvp_makeup_seed_q16(int32_t target_pressure_raw)
{
    int32_t p_above = target_pressure_raw - PRESSURE_DVP_ZERO_PSI_RAW;
    if (p_above < 0) {
        p_above = 0;
    }
    const int64_t seed_raw = PRESSURE_DVP_FF_BASE_RAW + (PRESSURE_DVP_FF_GAIN_Q8 * (int64_t)p_above) / 256;
    int64_t seed_q16 = seed_raw << PRESSURE_DVP_INTEGRAL_FRAC_BITS;
    if (seed_q16 > PRESSURE_DVP_INTEGRAL_LIMIT_Q16) {
        seed_q16 = PRESSURE_DVP_INTEGRAL_LIMIT_Q16;
    } else if (seed_q16 < PRESSURE_DVP_INTEGRAL_FLOOR_Q16) {
        seed_q16 = PRESSURE_DVP_INTEGRAL_FLOOR_Q16;
    }
    return seed_q16;
}
/* The stale-integral reset must never fire near the operating point: the
 * standing integral IS the leak make-up feed there. Reset only as a backstop
 * when the error opposes the integral by at least ~2 psi (or 4x soft zone,
 * whichever is larger). */
#define PRESSURE_DVP_INTEGRAL_RESET_MIN_RAW 2048U
/* Pressure-adaptive gain schedule. The plant's small-signal gain (flow per
 * current code) grows roughly proportionally with absolute sensor reading:
 * hand-tuned stability optima were kp16~2048 at 5 psi (raw 19963), ~1280 at
 * 15 psi (raw 29688), ~1024 at 25 psi (raw 39413) -- a 1/raw law predicts
 * 1377 and 1038 for the latter two. P, I and D products are multiplied by
 * REF/raw so commanded gains are "gains at 5 psi". */
#define PRESSURE_DVP_GAIN_REF_RAW 19963L
#define PRESSURE_DVP_GAIN_SCHED_MIN_RAW 4096L
#define PRESSURE_DVP_GAIN_SCHED_MIN_Q8 64L  /* 0.25x floor */
#define PRESSURE_DVP_GAIN_SCHED_MAX_Q8 512L /* 2x ceiling */
/* Derivative low-pass: filt += (delta - filt) / div per PID step (~10.7 ms at
 * div 8). The divisor is runtime-tunable (command 0x28, PRESSURE_DVP_DERIV_LP_
 * DIV_*). The filter state is Q12 so the truncation stick-zone of the division
 * is ~0.002 raw/step (a Q8 state parks visibly off its fixed point and turns
 * into a standing one-sided D drive). */
/* 16-bit gain scaling (command 0x27):
 *   P_raw = error * kp16 / 256            (kp16: Q8, legacy kp_q8 maps 1:1)
 *   I_q16 += error * ki16 * soft / 256    (ki16: Q16 per step; legacy ki_q8*256)
 *   D_raw = -(kd16 * deriv_q12) >> 16     (deriv_q12: filtered raw/step, Q12)
 */
#define PRESSURE_DVP_KD_SHIFT 16
#define PRESSURE_DVP_TARGET_MAX_RAW 54000U
#define PRESSURE_DVP_FAULT_MARGIN_RAW 2048U
#define PRESSURE_DVP_FAULT_MAX_RAW (PRESSURE_DVP_TARGET_MAX_RAW + PRESSURE_DVP_FAULT_MARGIN_RAW)
#define PRESSURE_DVP_TARGET_RAMP_UP_RAW_PER_STEP 96U   /* legacy; superseded by the fw1.41 reference generator */
#define PRESSURE_DVP_TARGET_RAMP_DOWN_RAW_PER_STEP 96U
/* fw 1.41 transient reference generator (command 0x29). vmax_* are raw per PID
 * step (750 Hz): 1 raw/step = 0.77 psi/s. dzone_* (deceleration zone) are in
 * PRESSURE_DVP_DZONE_STEP_RAW raw units. Defaults are seeded from the measured
 * plant: fill ~35 psi/s (45 raw/step), full-authority vent ~5 psi/s; the up
 * decel zone is wide because up-braking (cut inlet + leak) is weak, the down
 * zone is narrow because arresting a fall (add inlet) is fast. Re-measured and
 * finalised on hardware. vmax=0 disables shaping (instant target). */
#define PRESSURE_DVP_DZONE_STEP_RAW 64U
#define PRESSURE_DVP_DEFAULT_VMAX_UP 0U        /* fw1.43: direct target (no reference ramp); the PID does the shaping */
#define PRESSURE_DVP_DEFAULT_VMAX_DOWN 0U      /* fw1.43: direct target */
#define PRESSURE_DVP_DEFAULT_DZONE_UP 80U      /* ~5.3 psi decel zone */
#define PRESSURE_DVP_DEFAULT_DZONE_DOWN 30U    /* ~2.0 psi decel zone */
#define PRESSURE_DVP_DEFAULT_OUTLET_OPEN_CODE 57U /* outlet cracks at 46 but has no authority vs the leak until ~55 (measured) */
#define PRESSURE_DVP_DEFAULT_OUTLET_MAX_CODE 96U /* outlet saturates by code 90 (measured); 96 keeps ~97% authority, drops dead codes */
#define PRESSURE_DVP_DEFAULT_FF_SCALE 30U /* continuous leak make-up: carries down-steps where the stale-integral backstop wipes the seed; sine impact disproven (deadtime was the cause) */
/* fw 1.42 round-2 "flow shaping" (command 0x2A): a distance-scheduled current
 * ceiling that realises the user's bang-bang->PID idea. FAR from target the
 * inlet/outlet are handed FULL current (max orifice = max flow = fastest
 * pressure slew); as the pressure nears the target the ceiling tapers linearly
 * back to the steady hold range so the flow rate (hence dP/dt) eases toward zero
 * on arrival = sharp rise, no overshoot. The integrator is seeded to the leak
 * make-up the instant a setpoint JUMP is seen, so when the bulk slew desaturates
 * near the target the valve is already at the right hold current (bumpless). The
 * taper also restores the make-up permille->code calibration near target (the
 * seed is calibrated for the [open,max] hold range, so the ceiling must return
 * to dvp_max_code there or the seeded integral lands too high and overshoots).
 * When disabled the fw1.41 reference generator runs instead (runtime A/B). The
 * decel zones are in PRESSURE_DVP_DZONE_STEP_RAW (64 raw) units. */
#define PRESSURE_DVP_DEFAULT_FS_ENABLE 1U /* fw1.46: ON solely because fs is the only path that scales the integral seed (fs_seed); decel caps disabled below */
/* fw1.46 (2026-08-10, TLE_ALL_IN_ONE two-valve bench): decel_up/down = 0 — the
 * distance cap releases discontinuously at the 0.4 psi capture band and measurably
 * degraded 10-psi steps, so fs is enabled ONLY for its scaled integral seed.
 * far_max=95 is inert while decel_up==0 (old fw1.42 rationale: ~150 mA bulk-fill
 * ceiling, see TLE_tuning_results.md / second_round_tuning). */
#define PRESSURE_DVP_DEFAULT_FS_INLET_FAR_MAX 95U  /* ~150 mA bulk-fill ceiling */
#define PRESSURE_DVP_DEFAULT_FS_DECEL_UP 0U /* distance cap off: releases discontinuously at the capture band, degraded 10-psi steps */
#define PRESSURE_DVP_DEFAULT_FS_DECEL_DOWN 0U
/* Lag compensation: the deceleration (clamp + ceiling taper) is computed against
 * the pressure projected this many PID steps (1.33 ms each) ahead using the
 * filtered rate, so the current reaches the make-up ~one sensor/actuator lag
 * before the real pressure reaches target -> the real pressure arrives with low
 * velocity and does not overshoot, without slowing the bulk slew. 0 = off. */
#define PRESSURE_DVP_DEFAULT_FS_LOOKAHEAD 0U
/* Make-up seed scale (/128). The FF make-up model can be a slight over-estimate
 * as the supply/leak drift; landing a touch BELOW make-up lets the pressure
 * approach the target from below (the released integral trims up) instead of
 * overshooting. 128 = full FF estimate. */
#define PRESSURE_DVP_DEFAULT_FS_SEED_SCALE 17U /* old-plant leak model over-seeds this bench 2.7x; 17/128 + ff 30/128 = measured hold burden */
/* During a flow-shaping slew the integrator is FROZEN at the make-up seed while
 * the pressure is farther than this from target. Without the freeze the integral
 * winds up past the make-up during the fast approach (the output desaturates
 * while the error is still positive), commanding too much current at arrival ->
 * overshoot. Released inside the band so it still trims out any steady offset
 * (the FF seed is a deliberate slight under-estimate). ~0.4 psi. */
#define PRESSURE_DVP_FS_CAPTURE_RAW 389
/* Above this distance the flow-shaping deceleration (seed freeze + drive clamp)
 * is ALWAYS active (bulk slew), even at the step instant when the velocity is
 * still zero -- this prevents integral windup at step start. */
#define PRESSURE_DVP_FS_BULK_RAW 972
/* Between the capture band and the bulk band the deceleration is active only
 * while the pressure is still MOVING toward target faster than this (raw/step;
 * ~9 psi/s). Once it stalls near the target (a stale make-up seed can land a
 * touch low) the clamp/freeze release so the integral trims out the residual
 * with no steady-state error -- while a fast genuine approach stays clamped. */
#define PRESSURE_DVP_DEFAULT_FS_RATE_THRESH 2
/* fw 1.42 round-2 hardware dither (command 0x2B). Applied to the inlet (and
 * optionally outlet) ICC channel(s) to break valve stiction/hysteresis so the
 * small-signal flow is linear/repeatable (widens the usable control range). The
 * TLE has a hardware dither generator (datasheet 4.7); the MAX22200 backend
 * ignores it. chmask bit0=inlet, bit1=outlet. Defaults ~150 Hz, ~5.9 mA peak. */
#define PRESSURE_DVP_DEFAULT_DITHER_CHMASK 0x00U /* OFF: 150 Hz line 27x noise floor in-loop, 1.6x worse hold RMSE, zero stiction benefit (measured) */
#define PRESSURE_DVP_DITHER_DEFAULT_STEPS 8U
#define PRESSURE_DVP_DITHER_DEFAULT_FLAT 0U
#define PRESSURE_DVP_DITHER_DEFAULT_STEP_SIZE 12U  /* 8*12*61uA = 5.86 mA peak */
#define PRESSURE_DVP_DITHER_DEFAULT_MANT 365U      /* t_ref = 365*16/28MHz = 208 us */
#define PRESSURE_DVP_DITHER_DEFAULT_EXP 4U         /* T = 32*208us = 6.68 ms -> 150 Hz */
/* Step-detect latch: the reference generator (anti-overshoot ramp) engages only
 * for a setpoint JUMP larger than this; smaller/continuous setpoint motion (a
 * sine or a slider drag) is tracked DIRECTLY so it is not lagged by the ramp.
 * It disengages (back to direct) once the shaped reference has captured. */
#define PRESSURE_DVP_STEP_DETECT_RAW 972L          /* ~1.0 psi: jump that engages the ramp */
#define PRESSURE_DVP_STEP_PCAPTURE_RAW 117L        /* ~0.12 psi: pressure-captured -> exit to direct */
/* Reference governor: the shaped reference may lead the actual pressure by at
 * most this (raw), so it paces the plant instead of running ahead. This keeps
 * the deceleration zone engaged on the REAL approach (no overshoot on a
 * down-step to a high target, where the vent is strong) while still allowing a
 * fast bulk slew (the lead-band error floods the valve). ~5 psi. */
#define PRESSURE_DVP_REF_LEAD_RAW 4863.0f
#define PRESSURE_DVP_CODE_FRAC_BITS 9U
#define PRESSURE_DVP_CODE_Q_SCALE (1U << PRESSURE_DVP_CODE_FRAC_BITS)
/* Moving-window dither. N is runtime-configurable (command 0x28) up to MAX:
 * larger N = finer effective current resolution but slower dither, smaller N =
 * faster but coarser. The MAX22200 has only 7 effective bits and the valve
 * opens over a fraction of that, so the dither recovers sub-LSB resolution. */
#define PRESSURE_DVP_DITHER_HISTORY_MAX 8U
#define PRESSURE_DVP_DITHER_HISTORY_N 4U /* default window length */
/* fw 1.39 output-shaping defaults (command 0x28). Headline: quiet the valve
 * current near the setpoint without losing the no-overshoot step response. */
#define PRESSURE_DVP_DEFAULT_CODE_JUMP_MAX 2U   /* max code change / MAX update; 0 = off */
#define PRESSURE_DVP_DERIV_LP_DIV_DEFAULT 3U    /* derivative low-pass divisor (light: fast D for transient damping; D-fade zeroes it at hold) */
#define PRESSURE_DVP_DERIV_LP_DIV_MAX 64U
#define PRESSURE_DVP_DEFAULT_DEADTIME_STEPS 10U /* 13.3 ms: 27x small-step overshoot cut for ~9 ms sine lag; >=45 unstable */
#define PRESSURE_DVP_DEADTIME_STEPS_MAX 64U
/* Derivative-fade band: D scales linearly from 0 at the setpoint to full at
 * this error. Damps transients (large error) while staying quiet in the
 * noise-only zone near target. Stored in raw counts; 0 = D always full. */
#define PRESSURE_DVP_DEFAULT_D_FADE_BAND_RAW 486U /* ~0.5 psi */
#define PRESSURE_DVP_D_FADE_BAND_STEP_RAW 128U     /* command unit: counts per LSB */
/* fw 1.40 asymmetric split-range vent threshold (command 0x28 byte 6, permille).
 * The outlet engages only for drive below -this; smaller overshoots coast on
 * the leak. Command unit: 4 permille per LSB. 0 = legacy (outlet on any negative
 * drive). Default ~248 permille keeps the outlet shut through normal hold ripple
 * at the lowest setpoints while still venting promptly on real down-steps. */
#define PRESSURE_DVP_DEFAULT_VENT_THRESHOLD_PM 20U /* early outlet engagement wins on this leaky plant (measured monotonic 250..20) */
#define PRESSURE_DVP_VENT_THRESHOLD_STEP_PM 4U     /* command unit: permille per LSB */
#define PRESSURE_DVP_VENT_THRESHOLD_MAX_PM 1000U
#define PRESSURE_DVP_TIMING_PERIOD_TICK_US 16U
#define PRESSURE_DVP_CURRENT_UPDATE_PERIOD_US (1000000U / MAX_CURRENT_UPDATE_HZ)
#define PRESSURE_DVP_CURRENT_UPDATE_PERIOD_TICKS ((PRESSURE_DVP_CURRENT_UPDATE_PERIOD_US + (PRESSURE_DVP_TIMING_PERIOD_TICK_US / 2U)) / PRESSURE_DVP_TIMING_PERIOD_TICK_US)
/* FAULT reads low at idle on the bench; keep it telemetry-only in DVP mode. */
#define PRESSURE_DVP_STOP_ON_FAULT_PIN 0U

/* fw 1.44 manual constant-current drive (command 0x23). A bench/plumbing tool:
 * open-loop, one valve at a time, only accepted while the closed loop is NOT
 * running. Every activation carries its own dead-host timeout (the host must
 * re-send to sustain the drive) and, for the inlet, a filtered-pressure ceiling
 * so a stuck-open inlet cannot over-pressurise the bladder. */
#define MANUAL_ROLE_OFF 0U
#define MANUAL_ROLE_INLET 1U
#define MANUAL_ROLE_OUTLET 2U
#define MANUAL_TIMEOUT_DEFAULT_MS 500U
#define MANUAL_TIMEOUT_MIN_MS 10U
#define MANUAL_TIMEOUT_MAX_MS 5000U
#define MANUAL_PRESSURE_LIMIT_DEFAULT_RAW PRESSURE_DVP_TARGET_MAX_RAW

/* fw 1.44 raw ADC capture (command 0x10). n consecutive UNFILTERED LTC1864
 * samples at the native ADC_SAMPLE_HZ, buffered in RAM and then dumped over the
 * console as ASCII by a low-priority task. The sampling tap is a bounds check
 * plus a store, so it costs the 3 kHz ADC task nothing measurable. */
#define ADC_CAPTURE_DEFAULT_SAMPLES 3000U
#define ADC_CAPTURE_MIN_SAMPLES 100U
#define ADC_CAPTURE_MAX_SAMPLES 12000U /* 24 KB buffer, allocated once on first use */
#define ADC_CAPTURE_VALUES_PER_LINE 16U
#define ADC_CAPTURE_LINES_PER_YIELD 8U
#define ADC_CAPTURE_POLL_MS 20U

/* valve_task notification bits. */
#define MAX_CURRENT_UPDATE_NOTIFY_BIT (1UL << 0)
#define ACTUATOR_REQUEST_NOTIFY_BIT (1UL << 1)
/* can_task notification bit (MCP INT ISR and TX-queue pushes). */
#define CAN_SERVICE_NOTIFY_BIT (1UL << 2)

static const char *TAG = "vema";

/* =====================================================================
 * Types and shared state
 * ===================================================================== */

typedef enum {
    PRESSURE_STATE_IDLE = 0,
    PRESSURE_STATE_RUNNING = 1,
    PRESSURE_STATE_SETTLING = 2,
    PRESSURE_STATE_FAULT = 3,
    PRESSURE_STATE_CONFIG_ERROR = 4,
    PRESSURE_STATE_STALE_ADC = 5,
} pressure_state_t;

typedef enum {
    PRESSURE_ACTION_NONE = 0,
    PRESSURE_ACTION_INLET = 1,
    PRESSURE_ACTION_OUTLET = 2,
    PRESSURE_ACTION_DEADBAND = 3,
    PRESSURE_ACTION_SETTLING = 4,
} pressure_action_t;

typedef struct {
    pressure_state_t state;
    pressure_action_t action;
    uint8_t valve_mode; /* PRESSURE_VALVE_MODE_BANGBANG / _DVP */
    uint8_t flags;
    uint16_t target_raw;
    uint16_t previous_target_raw;
    uint16_t deadband_raw;
    uint8_t pending_valid;
    uint16_t pending_target_raw;
    uint16_t pending_deadband_raw;
    /* bang-bang parameters */
    uint8_t hit_current;
    uint8_t hold_current;
    uint8_t hit_time;
    uint8_t min_pulse_ms;
    uint8_t max_pulse_ms;
    uint8_t settle_ms;
    /* DVP parameters (fw 1.38: 16-bit gains, see COMMAND_PRESSURE_DVP_GAINS16;
     * the legacy 0x24 config maps its 8-bit gains onto these). */
    uint16_t dvp_kp16;
    uint16_t dvp_ki16;
    uint16_t dvp_kd16;
    uint8_t dvp_open_code;
    uint8_t dvp_max_code;
    uint8_t dvp_slew_code;
    uint8_t dvp_output_deadband;
    /* fw 1.39 output shaping (command 0x28) */
    uint8_t dvp_dither_n;
    uint8_t dvp_code_jump_max;
    uint8_t dvp_deriv_lp_div;
    uint8_t dvp_deadtime_steps;
    uint16_t dvp_d_fade_band_raw;
    /* fw 1.40 asymmetric split-range (command 0x28 byte 6): the outlet only
     * engages when the controller drive falls below -dvp_vent_threshold_pm
     * (permille). Between -vent_threshold and the inlet deadband the loop
     * coasts and the plant leak does the venting -- this removes the
     * inlet<->outlet relay limit cycle that otherwise appears near low
     * setpoints where the steady (leak-make-up) output sits close to 0. */
    uint16_t dvp_vent_threshold_pm;
    /* fw 1.41 transient reference generator + per-channel outlet range
     * (command 0x29). The reference trajectory cruises at dvp_vmax_* (raw per
     * PID step) and decelerates over dvp_dzone_* (in PRESSURE_DVP_DZONE_STEP_RAW
     * units) into the target; the outlet vents over its own open/max range. */
    uint8_t dvp_vmax_up;
    uint8_t dvp_vmax_down;
    uint8_t dvp_dzone_up;
    uint8_t dvp_dzone_down;
    uint8_t dvp_outlet_open_code;
    uint8_t dvp_outlet_max_code;
    uint8_t dvp_ff_scale;   /* leak make-up feedforward scale /128 (0 = off) */
    uint8_t dvp_slewing;    /* 1 while the step reference generator is active */
    float dvp_ref_pos;   /* shaped reference, raw (target space); seeded at START */
    /* fw 1.42 flow shaping (command 0x2A): distance-scaled current ceiling +
     * step-detect integral seed. When enabled the reference ramp is bypassed
     * (ref = target) and the current ceiling does the trajectory shaping. */
    uint8_t dvp_fs_enable;
    uint8_t dvp_fs_inlet_far_max;  /* inlet current ceiling when far from target */
    uint8_t dvp_fs_decel_up;       /* rise taper zone (DZONE_STEP_RAW units) */
    uint8_t dvp_fs_decel_down;     /* fall taper zone (DZONE_STEP_RAW units) */
    uint8_t dvp_fs_lookahead;      /* lag-comp: decel against pressure this many PID steps ahead */
    uint8_t dvp_fs_seed_scale;     /* make-up seed scale /128 (128 = full FF; <128 lands below -> approach from below) */
    uint8_t dvp_fs_rate_thresh;    /* velocity gate: decel releases below this |rate| (raw/step) near target */
    uint8_t dvp_fs_rising;         /* 1 if the active flow-shaping slew is upward (set at seed) */
    uint16_t dvp_fs_seed_target;   /* last target the integrator was seeded for */
    /* fw 1.42 hardware dither (command 0x2B); applied to the TLE via the
     * solenoid shim at START and on command. */
    uint8_t dvp_dither_chmask;     /* bit0 = inlet, bit1 = outlet */
    uint8_t dvp_dither_steps;
    uint8_t dvp_dither_flat;
    uint8_t dvp_dither_deep;
    uint16_t dvp_dither_step_size;
    uint16_t dvp_dither_mant;
    uint8_t dvp_dither_exp;
    /* live output state */
    uint8_t output_channel;
    uint8_t output_on;
    uint8_t last_pulse_ms;
    uint8_t error_code;
    uint8_t last_direction;
    uint8_t dvp_current_code;
    uint8_t dvp_output_channel;
    uint16_t dvp_target_code_q;
    uint16_t dvp_applied_code_q;
    int64_t dvp_integral_q16;
    int32_t dvp_deriv_filt_q12;
    int32_t dvp_output_permille;
    int16_t dvp_integral_permille;   /* telemetry only */
    int16_t dvp_derivative_permille; /* telemetry only */
    uint8_t dvp_sched_gain_q8_div2;  /* telemetry only: schedule gain, Q8/2 */
    uint32_t dvp_update_count;
    uint32_t dvp_late_updates;
    uint16_t dvp_max_service_us;
    uint8_t fault_low_count;
    uint16_t adc_average;
    uint16_t previous_average;
    uint32_t adc_samples;
    uint32_t pulse_count;
    int64_t pulse_end_us;
    int64_t settle_until_us;
} pressure_control_t;

typedef struct {
    uint16_t raw;
    uint16_t average;
    uint32_t samples;
    int64_t last_update_us;
} adc_status_t;

typedef struct {
    uint16_t samples[ADC_PID_WINDOW_SAMPLES];
    uint8_t count;
    uint16_t latest_raw;
    uint32_t raw_samples;
    uint32_t read_errors;
} adc_pid_buffer_t;

typedef struct {
    uint16_t can_id;
    uint8_t length;
    uint8_t payload[8];
} can_tx_queue_entry_t;

typedef enum {
    ACTUATOR_REQUEST_NONE = 0,
    ACTUATOR_REQUEST_ALL_OFF = 1,
    ACTUATOR_REQUEST_PULSE_ON = 2,
    ACTUATOR_REQUEST_CHANNEL_OFF = 3,
} actuator_request_type_t;

typedef struct {
    uint32_t sequence;
    actuator_request_type_t type;
    uint8_t channel;
} actuator_request_t;

/* PID -> valve_task command for the proportional mode (generation-numbered). */
typedef struct {
    uint32_t generation;
    uint8_t active;
    uint8_t channel;
    uint16_t code_q;
    uint8_t min_code;
    uint8_t max_code;
    uint8_t dither_n;       /* shaping: dither window length */
    uint8_t code_jump_max;  /* shaping: max code change per MAX update (0=off) */
} dvp_drive_t;

/* valve_task private dither state for the proportional mode. */
typedef struct {
    uint32_t generation;
    uint8_t active;
    uint8_t channel;
    uint16_t target_code_q;
    uint16_t target_history_q[PRESSURE_DVP_DITHER_HISTORY_MAX];
    uint8_t actual_history_code[PRESSURE_DVP_DITHER_HISTORY_MAX];
    uint8_t history_count;
    uint8_t history_head;
    uint8_t window_n;       /* active dither window length (1..MAX) */
    uint8_t code_jump_max;  /* max code change per update (0 = unlimited) */
    uint8_t min_code;
    uint8_t max_code;
    uint8_t last_code;
} dvp_actuator_state_t;

/* Manual (open-loop) constant-current drive, command 0x23. Supervised by the
 * PID task at PRESSURE_PID_HZ even while the controller is IDLE: it republishes
 * the drive, expires it and enforces the inlet pressure ceiling. */
typedef struct {
    uint8_t active;
    uint8_t role;    /* MANUAL_ROLE_* */
    uint8_t channel; /* VALVE_CHANNEL_* resolved from role */
    uint8_t code;    /* constant current code, already clamped to HOST_MAX_CODE */
    uint16_t pressure_limit_raw;
    int64_t expires_us;
} manual_drive_t;

/* Raw ADC capture (command 0x10) lifecycle. ARMED..FULL are owned by the 3 kHz
 * ADC task, DUMPING by the low-priority dump task. */
typedef enum {
    ADC_CAPTURE_STATE_IDLE = 0,
    ADC_CAPTURE_STATE_RUNNING = 1,
    ADC_CAPTURE_STATE_FULL = 2,
    ADC_CAPTURE_STATE_DUMPING = 3,
} adc_capture_state_t;

static portMUX_TYPE s_adc_status_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_adc_pid_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_can_tx_queue_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_pressure_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_actuator_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_dvp_drive_lock = portMUX_INITIALIZER_UNLOCKED;
static portMUX_TYPE s_manual_lock = portMUX_INITIALIZER_UNLOCKED;

static adc_status_t s_adc_status;
static adc_pid_buffer_t s_adc_pid_buffer;

static pressure_control_t s_pressure = {
    .state = PRESSURE_STATE_IDLE,
    .action = PRESSURE_ACTION_NONE,
    .valve_mode = PRESSURE_VALVE_MODE_DEFAULT,
    .flags = PRESSURE_DEFAULT_FLAGS,
    .target_raw = PRESSURE_DEFAULT_TARGET_RAW,
    .previous_target_raw = PRESSURE_DEFAULT_TARGET_RAW,
    .deadband_raw = PRESSURE_DEFAULT_DEADBAND_RAW,
    .hit_current = PRESSURE_DEFAULT_HIT_CURRENT,
    .hold_current = PRESSURE_DEFAULT_HOLD_CURRENT,
    .hit_time = PRESSURE_DEFAULT_HIT_TIME,
    .min_pulse_ms = PRESSURE_DEFAULT_MIN_PULSE_MS,
    .max_pulse_ms = PRESSURE_DEFAULT_MAX_PULSE_MS,
    .settle_ms = PRESSURE_DEFAULT_SETTLE_MS,
    .dvp_kp16 = PRESSURE_DVP_DEFAULT_KP16,
    .dvp_ki16 = PRESSURE_DVP_DEFAULT_KI16,
    .dvp_kd16 = PRESSURE_DVP_DEFAULT_KD16,
    .dvp_open_code = PRESSURE_DVP_DEFAULT_OPEN_CODE,
    .dvp_max_code = PRESSURE_DVP_DEFAULT_MAX_CODE,
    .dvp_slew_code = PRESSURE_DVP_DEFAULT_SLEW_CODE,
    .dvp_output_deadband = PRESSURE_DVP_DEFAULT_OUTPUT_DEADBAND,
    .dvp_dither_n = PRESSURE_DVP_DITHER_HISTORY_N,
    .dvp_code_jump_max = PRESSURE_DVP_DEFAULT_CODE_JUMP_MAX,
    .dvp_deriv_lp_div = PRESSURE_DVP_DERIV_LP_DIV_DEFAULT,
    .dvp_deadtime_steps = PRESSURE_DVP_DEFAULT_DEADTIME_STEPS,
    .dvp_d_fade_band_raw = PRESSURE_DVP_DEFAULT_D_FADE_BAND_RAW,
    .dvp_vent_threshold_pm = PRESSURE_DVP_DEFAULT_VENT_THRESHOLD_PM,
    .dvp_vmax_up = PRESSURE_DVP_DEFAULT_VMAX_UP,
    .dvp_vmax_down = PRESSURE_DVP_DEFAULT_VMAX_DOWN,
    .dvp_dzone_up = PRESSURE_DVP_DEFAULT_DZONE_UP,
    .dvp_dzone_down = PRESSURE_DVP_DEFAULT_DZONE_DOWN,
    .dvp_outlet_open_code = PRESSURE_DVP_DEFAULT_OUTLET_OPEN_CODE,
    .dvp_outlet_max_code = PRESSURE_DVP_DEFAULT_OUTLET_MAX_CODE,
    .dvp_ff_scale = PRESSURE_DVP_DEFAULT_FF_SCALE,
    .dvp_fs_enable = PRESSURE_DVP_DEFAULT_FS_ENABLE,
    .dvp_fs_inlet_far_max = PRESSURE_DVP_DEFAULT_FS_INLET_FAR_MAX,
    .dvp_fs_decel_up = PRESSURE_DVP_DEFAULT_FS_DECEL_UP,
    .dvp_fs_decel_down = PRESSURE_DVP_DEFAULT_FS_DECEL_DOWN,
    .dvp_fs_lookahead = PRESSURE_DVP_DEFAULT_FS_LOOKAHEAD,
    .dvp_fs_seed_scale = PRESSURE_DVP_DEFAULT_FS_SEED_SCALE,
    .dvp_fs_rate_thresh = PRESSURE_DVP_DEFAULT_FS_RATE_THRESH,
    .dvp_dither_chmask = PRESSURE_DVP_DEFAULT_DITHER_CHMASK,
    .dvp_dither_steps = PRESSURE_DVP_DITHER_DEFAULT_STEPS,
    .dvp_dither_flat = PRESSURE_DVP_DITHER_DEFAULT_FLAT,
    .dvp_dither_deep = 0U,
    .dvp_dither_step_size = PRESSURE_DVP_DITHER_DEFAULT_STEP_SIZE,
    .dvp_dither_mant = PRESSURE_DVP_DITHER_DEFAULT_MANT,
    .dvp_dither_exp = PRESSURE_DVP_DITHER_DEFAULT_EXP,
    .output_channel = VALVE_CHANNEL_NONE,
    .dvp_output_channel = VALVE_CHANNEL_NONE,
};

static actuator_request_t s_actuator_request = {
    .type = ACTUATOR_REQUEST_NONE,
    .channel = VALVE_CHANNEL_NONE,
};
static dvp_drive_t s_dvp_drive = {
    .channel = VALVE_CHANNEL_NONE,
};
static manual_drive_t s_manual = {
    .role = MANUAL_ROLE_OFF,
    .channel = VALVE_CHANNEL_NONE,
};

static can_tx_queue_entry_t s_can_tx_queue[CAN_TX_QUEUE_DEPTH];
static uint8_t s_can_tx_queue_head;
static uint8_t s_can_tx_queue_tail;
static uint8_t s_can_tx_queue_count;

static volatile uint32_t s_max_update_ticks;
static volatile uint16_t s_telemetry_period_ms = TELEMETRY_DEFAULT_PERIOD_MS;
/* Runtime ADC EMA depth (command 0x2C). Lower = less lag (faster transient, less
 * overshoot) but more pressure noise into the PID; higher = smoother hold. The
 * per-window median still rejects valve spikes regardless. Default = compiled. */
static volatile uint8_t s_adc_avg_shift = 3U; /* fw1.46: white sigma-9.5ct ADC needs TC 10.7 ms; sigma 1.1 ct filtered (measured) */
/* Runtime integral floor (command 0x2D); see PRESSURE_DVP_INT_FLOOR_MODE_*. */
static volatile uint8_t s_dvp_int_floor_mode = PRESSURE_DVP_INT_FLOOR_MODE_DEFAULT;

/* Raw ADC capture (command 0x10). The buffer is allocated on the first capture
 * and kept for the life of the run; `want`/`count` are only advanced by the ADC
 * task, and `state` is the handshake between it and the dump task. */
static uint16_t *s_adc_capture_buffer;
static volatile uint32_t s_adc_capture_want;
static volatile uint32_t s_adc_capture_count;
static volatile uint8_t s_adc_capture_state = ADC_CAPTURE_STATE_IDLE;

static TaskHandle_t s_can_task_handle;
static TaskHandle_t s_pid_task_handle;
static TaskHandle_t s_adc_task_handle;
static TaskHandle_t s_valve_task_handle;
static TaskHandle_t s_adc_dump_task_handle;
static gptimer_handle_t s_control_timer;

static mcp2515_t s_mcp;
static volatile bool s_mcp_ready;

/* CAN link breaker state, owned by can_task (see service_can_tx_slot). */
static bool s_can_link_down;
static bool s_can_link_probe_pending;
static bool s_can_link_probe_force;
static uint32_t s_can_tx_busy_polls;
static int64_t s_can_tx_busy_since_us;
static int64_t s_can_link_probe_us;
/* Rate limiter for the TX-queue-full warning (see send_can_frame). */
static int64_t s_can_queue_full_log_us;
static uint32_t s_can_queue_full_suppressed;
/* Last unconditional EFLG sweep (see service_can_rx). */
static int64_t s_can_eflg_sweep_us;
static int64_t s_can_rx_overflow_log_us;
static uint32_t s_can_rx_overflow_suppressed;

static uint16_t s_can_device_base = CAN_ID_DEVICE_BASE_DEFAULT;
static uint16_t s_can_id_host_command = CAN_ID_DEVICE_BASE_DEFAULT + CAN_ID_HOST_COMMAND_OFFSET;
static uint16_t s_can_id_host_ota_data = CAN_ID_DEVICE_BASE_DEFAULT + CAN_ID_HOST_OTA_DATA_OFFSET;
static uint16_t s_can_id_device_status = CAN_ID_DEVICE_BASE_DEFAULT + CAN_ID_DEVICE_STATUS_OFFSET;
static uint16_t s_can_id_extended_command = CAN_ID_DEVICE_BASE_DEFAULT + CAN_ID_EXTENDED_COMMAND_OFFSET;

/* =====================================================================
 * Small helpers
 * ===================================================================== */

static uint16_t read_u16_le(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static void write_u16_le(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8) & 0xFFU);
}

static void write_i16_le(uint8_t *data, int16_t value)
{
    write_u16_le(data, (uint16_t)value);
}

static uint8_t clamp_u8(uint8_t value, uint8_t min_value, uint8_t max_value)
{
    if (value < min_value) {
        return min_value;
    }
    return value > max_value ? max_value : value;
}

static uint16_t clamp_u16(uint16_t value, uint16_t min_value, uint16_t max_value)
{
    if (value < min_value) {
        return min_value;
    }
    return value > max_value ? max_value : value;
}

static int32_t clamp_i32(int32_t value, int32_t min_value, int32_t max_value)
{
    if (value < min_value) {
        return min_value;
    }
    return value > max_value ? max_value : value;
}

/* =====================================================================
 * CAN device ID persistence (NVS)
 * ===================================================================== */

static bool can_device_base_is_valid(uint32_t base_id)
{
    /* The extended-command ID is the highest derived one, so base + 0x400 must
     * still fit in 11 bits. */
    if (base_id == 0U || base_id > (0x7FFU - CAN_ID_EXTENDED_COMMAND_OFFSET)) {
        return false;
    }

    const uint32_t derived[] = {
        base_id,
        base_id + CAN_ID_HOST_COMMAND_OFFSET,
        base_id + CAN_ID_HOST_OTA_DATA_OFFSET,
        base_id + CAN_ID_DEVICE_STATUS_OFFSET,
        base_id + CAN_ID_EXTENDED_COMMAND_OFFSET,
    };
    for (size_t i = 0; i < sizeof(derived) / sizeof(derived[0]); ++i) {
        if (derived[i] == CAN_ID_BROADCAST) {
            return false;
        }
        /* Only the base itself may sit in the actuator band. A derived ID
         * landing there would give this board a hardware filter on another
         * actuator's 150 Hz status replies, and their payload bytes would
         * then be read as commands -- 0x01 is OTA start. */
        if (i != 0U && derived[i] >= TLE_LEGACY_RUNTIME_FIRST_ID &&
            derived[i] <= TLE_LEGACY_RUNTIME_LAST_ID) {
            return false;
        }
        /* The runtime target table occupies 0x091..0x098; a board answering on
         * one of those would corrupt three actuators' commands. */
        if (derived[i] >= TLE_LEGACY_RUNTIME_TABLE_BASE_ID &&
            derived[i] < TLE_LEGACY_RUNTIME_TABLE_BASE_ID + 8U) {
            return false;
        }
    }
    return true;
}

static void can_apply_device_base(uint16_t base_id)
{
    s_can_device_base = base_id;
    s_can_id_host_command = base_id + CAN_ID_HOST_COMMAND_OFFSET;
    s_can_id_host_ota_data = base_id + CAN_ID_HOST_OTA_DATA_OFFSET;
    s_can_id_device_status = base_id + CAN_ID_DEVICE_STATUS_OFFSET;
    s_can_id_extended_command = base_id + CAN_ID_EXTENDED_COMMAND_OFFSET;
    can_ota_set_status_can_id(s_can_id_device_status);
    tle_can_legacy_set_base_id(base_id);
}

static esp_err_t can_id_nvs_init(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_RETURN_ON_ERROR(nvs_flash_erase(), TAG, "erase NVS");
        err = nvs_flash_init();
    }
    return err;
}

static bool save_device_id_to_nvs(uint16_t base_id)
{
    if (!can_device_base_is_valid(base_id)) {
        return false;
    }

    nvs_handle_t handle;
    if (nvs_open(CAN_ID_NVS_NAMESPACE, NVS_READWRITE, &handle) != ESP_OK) {
        return false;
    }
    bool saved = nvs_set_u32(handle, CAN_ID_NVS_KEY, base_id) == ESP_OK &&
                 nvs_commit(handle) == ESP_OK;
    nvs_close(handle);
    if (saved) {
        /* Apply immediately so the OK acknowledgment goes out on the NEW
         * status ID and reports the new addressing (as host tools expect). */
        can_apply_device_base(base_id);
    }
    return saved;
}

static void load_device_id_from_nvs(void)
{
    nvs_handle_t handle;
    if (nvs_open(CAN_ID_NVS_NAMESPACE, NVS_READWRITE, &handle) != ESP_OK) {
        return;
    }

    uint32_t stored = 0;
    esp_err_t err = nvs_get_u32(handle, CAN_ID_NVS_KEY, &stored);
    if (err == ESP_OK && can_device_base_is_valid(stored)) {
        can_apply_device_base((uint16_t)stored);
    } else {
        (void)nvs_set_u32(handle, CAN_ID_NVS_KEY, CAN_ID_DEVICE_BASE_DEFAULT);
        (void)nvs_commit(handle);
    }
    nvs_close(handle);
}

/* =====================================================================
 * ADC status (filtered feedback shared between tasks)
 * ===================================================================== */

static adc_status_t adc_status_snapshot(void)
{
    adc_status_t snapshot;
    portENTER_CRITICAL(&s_adc_status_lock);
    snapshot = s_adc_status;
    portEXIT_CRITICAL(&s_adc_status_lock);
    return snapshot;
}

static void adc_status_update(uint16_t raw, uint16_t average, uint32_t samples)
{
    portENTER_CRITICAL(&s_adc_status_lock);
    s_adc_status.raw = raw;
    s_adc_status.average = average;
    s_adc_status.samples = samples;
    s_adc_status.last_update_us = esp_timer_get_time();
    portEXIT_CRITICAL(&s_adc_status_lock);
}

static bool adc_feedback_is_stale(void)
{
    adc_status_t status = adc_status_snapshot();
    return status.last_update_us == 0 ||
           (esp_timer_get_time() - status.last_update_us) > ((int64_t)PRESSURE_ADC_STALE_TIMEOUT_MS * 1000);
}

/* =====================================================================
 * Pressure control state
 * ===================================================================== */

static pressure_control_t pressure_control_snapshot(void)
{
    pressure_control_t snapshot;
    portENTER_CRITICAL(&s_pressure_lock);
    snapshot = s_pressure;
    portEXIT_CRITICAL(&s_pressure_lock);
    return snapshot;
}

static void pressure_control_store_state(const pressure_control_t *state)
{
    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure = *state;
    portEXIT_CRITICAL(&s_pressure_lock);
}

/* Store from the PID task: refuses the write-back if the CAN side changed
 * configuration mid-step (the caller then forces all-off), and merges in any
 * target/deadband/pending update that arrived while the step ran. */
static bool pressure_control_store_task_state(const pressure_control_t *state)
{
    bool stored = false;
    portENTER_CRITICAL(&s_pressure_lock);
    if ((s_pressure.state == PRESSURE_STATE_RUNNING || s_pressure.state == PRESSURE_STATE_SETTLING) &&
        s_pressure.valve_mode == state->valve_mode &&
        s_pressure.flags == state->flags &&
        s_pressure.hit_current == state->hit_current &&
        s_pressure.hold_current == state->hold_current &&
        s_pressure.hit_time == state->hit_time &&
        s_pressure.min_pulse_ms == state->min_pulse_ms &&
        s_pressure.max_pulse_ms == state->max_pulse_ms &&
        s_pressure.settle_ms == state->settle_ms &&
        s_pressure.dvp_kp16 == state->dvp_kp16 &&
        s_pressure.dvp_ki16 == state->dvp_ki16 &&
        s_pressure.dvp_kd16 == state->dvp_kd16 &&
        s_pressure.dvp_open_code == state->dvp_open_code &&
        s_pressure.dvp_max_code == state->dvp_max_code &&
        s_pressure.dvp_slew_code == state->dvp_slew_code &&
        s_pressure.dvp_output_deadband == state->dvp_output_deadband &&
        s_pressure.dvp_dither_n == state->dvp_dither_n &&
        s_pressure.dvp_code_jump_max == state->dvp_code_jump_max &&
        s_pressure.dvp_deriv_lp_div == state->dvp_deriv_lp_div &&
        s_pressure.dvp_deadtime_steps == state->dvp_deadtime_steps &&
        s_pressure.dvp_d_fade_band_raw == state->dvp_d_fade_band_raw &&
        s_pressure.dvp_vent_threshold_pm == state->dvp_vent_threshold_pm &&
        s_pressure.dvp_vmax_up == state->dvp_vmax_up &&
        s_pressure.dvp_vmax_down == state->dvp_vmax_down &&
        s_pressure.dvp_dzone_up == state->dvp_dzone_up &&
        s_pressure.dvp_dzone_down == state->dvp_dzone_down &&
        s_pressure.dvp_outlet_open_code == state->dvp_outlet_open_code &&
        s_pressure.dvp_outlet_max_code == state->dvp_outlet_max_code &&
        s_pressure.dvp_ff_scale == state->dvp_ff_scale &&
        s_pressure.dvp_fs_enable == state->dvp_fs_enable &&
        s_pressure.dvp_fs_inlet_far_max == state->dvp_fs_inlet_far_max &&
        s_pressure.dvp_fs_decel_up == state->dvp_fs_decel_up &&
        s_pressure.dvp_fs_decel_down == state->dvp_fs_decel_down &&
        s_pressure.dvp_fs_lookahead == state->dvp_fs_lookahead &&
        s_pressure.dvp_fs_seed_scale == state->dvp_fs_seed_scale &&
        s_pressure.dvp_fs_rate_thresh == state->dvp_fs_rate_thresh &&
        s_pressure.dvp_dither_chmask == state->dvp_dither_chmask &&
        s_pressure.dvp_dither_steps == state->dvp_dither_steps &&
        s_pressure.dvp_dither_flat == state->dvp_dither_flat &&
        s_pressure.dvp_dither_deep == state->dvp_dither_deep &&
        s_pressure.dvp_dither_step_size == state->dvp_dither_step_size &&
        s_pressure.dvp_dither_mant == state->dvp_dither_mant &&
        s_pressure.dvp_dither_exp == state->dvp_dither_exp) {
        pressure_control_t merged_state = *state;
        merged_state.target_raw = s_pressure.target_raw;
        merged_state.deadband_raw = s_pressure.deadband_raw;
        merged_state.pending_valid = s_pressure.pending_valid;
        merged_state.pending_target_raw = s_pressure.pending_target_raw;
        merged_state.pending_deadband_raw = s_pressure.pending_deadband_raw;
        s_pressure = merged_state;
        stored = true;
    }
    portEXIT_CRITICAL(&s_pressure_lock);
    return stored;
}

static void dvp_drive_clear(void)
{
    portENTER_CRITICAL(&s_dvp_drive_lock);
    ++s_dvp_drive.generation;
    if (s_dvp_drive.generation == 0U) {
        s_dvp_drive.generation = 1U;
    }
    s_dvp_drive.active = 0;
    s_dvp_drive.channel = VALVE_CHANNEL_NONE;
    s_dvp_drive.code_q = 0;
    s_dvp_drive.min_code = PRESSURE_DVP_DEFAULT_OPEN_CODE;
    s_dvp_drive.max_code = PRESSURE_DVP_DEFAULT_MAX_CODE;
    s_dvp_drive.dither_n = PRESSURE_DVP_DITHER_HISTORY_N;
    s_dvp_drive.code_jump_max = PRESSURE_DVP_DEFAULT_CODE_JUMP_MAX;
    portEXIT_CRITICAL(&s_dvp_drive_lock);
}

static void dvp_drive_publish(uint8_t channel, uint16_t code_q, uint8_t min_code, uint8_t max_code,
                              uint8_t dither_n, uint8_t code_jump_max)
{
    portENTER_CRITICAL(&s_dvp_drive_lock);
    ++s_dvp_drive.generation;
    if (s_dvp_drive.generation == 0U) {
        s_dvp_drive.generation = 1U;
    }
    s_dvp_drive.active = 1;
    s_dvp_drive.channel = channel;
    s_dvp_drive.code_q = code_q;
    s_dvp_drive.min_code = min_code;
    s_dvp_drive.max_code = max_code;
    s_dvp_drive.dither_n = dither_n;
    s_dvp_drive.code_jump_max = code_jump_max;
    portEXIT_CRITICAL(&s_dvp_drive_lock);
}

static dvp_drive_t dvp_drive_snapshot(void)
{
    dvp_drive_t snapshot;
    portENTER_CRITICAL(&s_dvp_drive_lock);
    snapshot = s_dvp_drive;
    portEXIT_CRITICAL(&s_dvp_drive_lock);
    return snapshot;
}

static void pressure_control_note_dvp_update(uint8_t channel, uint8_t current_code, uint32_t service_us, uint32_t late_count)
{
    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.dvp_current_code = current_code;
    s_pressure.dvp_output_channel = channel;
    s_pressure.dvp_applied_code_q = (uint16_t)current_code << PRESSURE_DVP_CODE_FRAC_BITS;
    ++s_pressure.dvp_update_count;
    s_pressure.dvp_late_updates += late_count;
    if (service_us > s_pressure.dvp_max_service_us) {
        s_pressure.dvp_max_service_us = service_us > UINT16_MAX ? UINT16_MAX : (uint16_t)service_us;
    }
    portEXIT_CRITICAL(&s_pressure_lock);
}

static void pressure_control_note_dvp_off(void)
{
    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.dvp_current_code = 0;
    s_pressure.dvp_output_channel = VALVE_CHANNEL_NONE;
    s_pressure.dvp_applied_code_q = 0;
    portEXIT_CRITICAL(&s_pressure_lock);
}

static void actuator_publish(actuator_request_type_t type, uint8_t channel)
{
    portENTER_CRITICAL(&s_actuator_lock);
    ++s_actuator_request.sequence;
    if (s_actuator_request.sequence == 0U) {
        s_actuator_request.sequence = 1U;
    }
    s_actuator_request.type = type;
    s_actuator_request.channel = channel;
    portEXIT_CRITICAL(&s_actuator_lock);

    if (s_valve_task_handle != NULL) {
        (void)xTaskNotify(s_valve_task_handle, ACTUATOR_REQUEST_NOTIFY_BIT, eSetBits);
    }
}

static actuator_request_t actuator_snapshot(void)
{
    actuator_request_t snapshot;
    portENTER_CRITICAL(&s_actuator_lock);
    snapshot = s_actuator_request;
    portEXIT_CRITICAL(&s_actuator_lock);
    return snapshot;
}

static bool pressure_control_is_active(void)
{
    pressure_control_t state = pressure_control_snapshot();
    return state.state == PRESSURE_STATE_RUNNING || state.state == PRESSURE_STATE_SETTLING || state.output_on != 0U;
}

/* =====================================================================
 * Manual constant-current drive (command 0x23)
 * ===================================================================== */

static manual_drive_t manual_drive_snapshot(void)
{
    manual_drive_t snapshot;
    portENTER_CRITICAL(&s_manual_lock);
    snapshot = s_manual;
    portEXIT_CRITICAL(&s_manual_lock);
    return snapshot;
}

static bool manual_drive_is_active(void)
{
    portENTER_CRITICAL(&s_manual_lock);
    const bool active = s_manual.active != 0U;
    portEXIT_CRITICAL(&s_manual_lock);
    return active;
}

/* Latch the manual drive off and de-energise the valves. Safe from either core
 * (the PID task on expiry/over-pressure, the CAN task on START/STOP/0x23).
 *
 * UNCONDITIONAL by design: "turn the valves off" must be true even when the arm
 * flag is already clear. The published DVP drive can outlive the arm flag (a
 * supervisor tick that was preempted mid-publish, or a stale drive from any
 * other path), and this is the host's only recovery command -- an early-out on
 * `!was_active` would answer OK while leaving a coil energised. The cost of
 * always running it is one redundant ALL_OFF notification on an idle board. */
static void manual_drive_stop(void)
{
    portENTER_CRITICAL(&s_manual_lock);
    s_manual.active = 0;
    s_manual.role = MANUAL_ROLE_OFF;
    s_manual.channel = VALVE_CHANNEL_NONE;
    s_manual.code = 0;
    s_manual.expires_us = 0;
    portEXIT_CRITICAL(&s_manual_lock);

    dvp_drive_clear();
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.action = PRESSURE_ACTION_NONE;
    s_pressure.dvp_current_code = 0;
    s_pressure.dvp_output_channel = VALVE_CHANNEL_NONE;
    s_pressure.dvp_target_code_q = 0;
    s_pressure.dvp_applied_code_q = 0;
    portEXIT_CRITICAL(&s_pressure_lock);
}

/* Apply the configured hardware dither to the valve channels (TLE92464 only;
 * a no-op on the MAX22200). step_size 0 (or a channel not in the mask) disables
 * the overlay on that channel. Called at START and on command 0x2B. */
static void pressure_control_apply_dither(const pressure_control_t *state)
{
    if (!solenoid_supports_hw_dither()) {
        return;
    }
    (void)solenoid_set_channel_dither(VALVE_CHANNEL_INLET,
                                      state->dvp_dither_steps, state->dvp_dither_flat,
                                      (state->dvp_dither_chmask & 0x01U) ? state->dvp_dither_step_size : 0U,
                                      state->dvp_dither_mant, state->dvp_dither_exp,
                                      state->dvp_dither_deep != 0U);
    (void)solenoid_set_channel_dither(VALVE_CHANNEL_OUTLET,
                                      state->dvp_dither_steps, state->dvp_dither_flat,
                                      (state->dvp_dither_chmask & 0x02U) ? state->dvp_dither_step_size : 0U,
                                      state->dvp_dither_mant, state->dvp_dither_exp,
                                      state->dvp_dither_deep != 0U);
}

static esp_err_t pressure_control_configure_channels(const pressure_control_t *state)
{
    ESP_RETURN_ON_ERROR(solenoid_set_all_channels_off(), TAG, "pressure all off before config");
    if (state->valve_mode == PRESSURE_VALVE_MODE_DVP) {
        ESP_RETURN_ON_ERROR(solenoid_configure_valve_channel(VALVE_CHANNEL_INLET, 0U, 0U, 0U,
                                                             SOLENOID_DRIVE_PROPORTIONAL_CURRENT),
                            TAG, "configure pressure inlet");
        ESP_RETURN_ON_ERROR(solenoid_configure_valve_channel(VALVE_CHANNEL_OUTLET, 0U, 0U, 0U,
                                                             SOLENOID_DRIVE_PROPORTIONAL_CURRENT),
                            TAG, "configure pressure outlet");
        pressure_control_apply_dither(state);
        return ESP_OK;
    }
    ESP_RETURN_ON_ERROR(solenoid_configure_valve_channel(VALVE_CHANNEL_INLET,
                                                         state->hit_current, state->hold_current, state->hit_time,
                                                         SOLENOID_DRIVE_ONOFF_VOLTAGE),
                        TAG, "configure pressure inlet");
    return solenoid_configure_valve_channel(VALVE_CHANNEL_OUTLET,
                                            state->hit_current, state->hold_current, state->hit_time,
                                            SOLENOID_DRIVE_ONOFF_VOLTAGE);
}

static void pressure_control_request_stop(pressure_state_t final_state, uint8_t error_code)
{
    /* A STOP (or any fault/stale-feedback abort) also cancels a manual drive --
     * the closed loop and the bench tool must never both own a valve. */
    manual_drive_stop();
    dvp_drive_clear();
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    pressure_control_t state = pressure_control_snapshot();
    state.state = final_state;
    state.action = PRESSURE_ACTION_NONE;
    state.output_channel = VALVE_CHANNEL_NONE;
    state.output_on = 0;
    state.last_pulse_ms = 0;
    state.error_code = error_code;
    state.last_direction = PRESSURE_ACTION_NONE;
    state.previous_target_raw = state.target_raw;
    state.pending_valid = 0;
    state.dvp_current_code = 0;
    state.dvp_output_channel = VALVE_CHANNEL_NONE;
    state.dvp_target_code_q = 0;
    state.dvp_applied_code_q = 0;
    state.dvp_integral_q16 = 0;
    state.dvp_deriv_filt_q12 = 0;
    state.dvp_output_permille = 0;
    state.fault_low_count = 0;
    state.pulse_end_us = 0;
    state.settle_until_us = 0;
    pressure_control_store_state(&state);
}

static uint16_t pressure_control_clamp_target_raw(const pressure_control_t *state, uint16_t target_raw)
{
    if (state->valve_mode == PRESSURE_VALVE_MODE_DVP && target_raw > PRESSURE_DVP_TARGET_MAX_RAW) {
        return PRESSURE_DVP_TARGET_MAX_RAW;
    }
    return target_raw;
}

static void pressure_control_reset_output_fields(pressure_control_t *state)
{
    state->action = PRESSURE_ACTION_NONE;
    state->output_channel = VALVE_CHANNEL_NONE;
    state->output_on = 0;
    state->last_pulse_ms = 0;
    state->error_code = 0;
    state->last_direction = PRESSURE_ACTION_NONE;
    state->dvp_current_code = 0;
    state->dvp_output_channel = VALVE_CHANNEL_NONE;
    state->dvp_target_code_q = 0;
    state->dvp_applied_code_q = 0;
    state->dvp_integral_q16 = 0;
    state->dvp_deriv_filt_q12 = 0;
    state->dvp_output_permille = 0;
    state->fault_low_count = 0;
    state->pulse_end_us = 0;
    state->settle_until_us = 0;
    /* Force the flow-shaping integral seed to fire on the first step after a
     * (re)start so the inlet engages at the leak make-up immediately. */
    state->dvp_fs_seed_target = 0;
}

static bool pressure_control_apply_config(uint8_t flags,
                                          uint8_t hit_current,
                                          uint8_t hold_current,
                                          uint8_t hit_time,
                                          uint8_t min_pulse_ms,
                                          uint8_t max_pulse_ms,
                                          uint8_t settle_ms)
{
    dvp_drive_clear();
    pressure_control_t state = pressure_control_snapshot();
    state.flags = flags & (PRESSURE_CONFIG_FLAG_SENSOR_INCREASES | PRESSURE_CONFIG_FLAG_FAST_TRACK);
    /* On this board BOTH bang-bang channels are the Clippard DVP coils, and the
     * TLE maps the on/off "hit" code straight onto a current setpoint
     * (solenoid_configure_valve_channel -> setpoint_code_q9 = hit << 9), so 0x21
     * is a host->current path like any other and has to respect the same
     * ceiling. The legacy on/off jig (VEMA_BOARD_TLE_ALL_IN_ONE == 0) keeps the
     * full 7-bit range. */
#if VEMA_BOARD_TLE_ALL_IN_ONE
    const uint8_t hit_code_max = PRESSURE_DVP_HOST_MAX_CODE;
#else
    const uint8_t hit_code_max = SOLENOID_CURRENT_CODE_MAX;
#endif
    state.hit_current = clamp_u8(hit_current == 0U ? PRESSURE_DEFAULT_HIT_CURRENT : hit_current, 1U, hit_code_max);
    state.hold_current = clamp_u8(hold_current == 0U ? PRESSURE_DEFAULT_HOLD_CURRENT : hold_current, 1U, state.hit_current);
    state.hit_time = hit_time;
    state.min_pulse_ms = clamp_u8(min_pulse_ms == 0U ? PRESSURE_DEFAULT_MIN_PULSE_MS : min_pulse_ms, PRESSURE_MIN_PULSE_MS, PRESSURE_MAX_PULSE_MS);
    state.max_pulse_ms = clamp_u8(max_pulse_ms == 0U ? PRESSURE_DEFAULT_MAX_PULSE_MS : max_pulse_ms, state.min_pulse_ms, PRESSURE_MAX_PULSE_MS);
    state.settle_ms = clamp_u8(settle_ms == 0U ? PRESSURE_DEFAULT_SETTLE_MS : settle_ms, PRESSURE_MIN_SETTLE_MS, PRESSURE_MAX_SETTLE_MS);
    state.previous_target_raw = state.target_raw;
    /* Keep the direction memory across a bang-bang retune so the first
     * reversing pulse after a config change still gets the halving damping. */
    const uint8_t last_direction = state.last_direction;
    pressure_control_reset_output_fields(&state);
    state.last_direction = last_direction;

    esp_err_t err = pressure_control_configure_channels(&state);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "MAX22200 pressure configure failed: %s", esp_err_to_name(err));
        state.state = PRESSURE_STATE_CONFIG_ERROR;
        state.error_code = 1;
        pressure_control_store_state(&state);
        return false;
    }
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
        state.state = PRESSURE_STATE_IDLE;
    }
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "Pressure config: flags=0x%02x hit=%u hold=%u hit_time=%u pulse=%u-%ums settle=%ums",
             state.flags, state.hit_current, state.hold_current, state.hit_time, state.min_pulse_ms, state.max_pulse_ms, state.settle_ms);
    return true;
}

static bool pressure_control_apply_dvp_config(uint8_t kp_q8,
                                              uint8_t ki_q8,
                                              uint8_t kd_q8,
                                              uint8_t open_code,
                                              uint8_t max_code,
                                              uint8_t slew_code,
                                              uint8_t output_deadband)
{
    dvp_drive_clear();
    pressure_control_t state = pressure_control_snapshot();
    /* Legacy 8-bit gains map onto the fw 1.38 16-bit fields. */
    state.dvp_kp16 = (kp_q8 == 0U) ? PRESSURE_DVP_DEFAULT_KP_Q8 : kp_q8;
    state.dvp_ki16 = (uint16_t)ki_q8 << 8;
    state.dvp_kd16 = (uint16_t)kd_q8 << 4;
    state.dvp_open_code = clamp_u8(open_code == 0U ? PRESSURE_DVP_DEFAULT_OPEN_CODE : open_code, 1U, PRESSURE_DVP_HOST_MAX_CODE);
    state.dvp_max_code = clamp_u8(max_code == 0U ? PRESSURE_DVP_DEFAULT_MAX_CODE : max_code, state.dvp_open_code, PRESSURE_DVP_HOST_MAX_CODE);
    state.dvp_slew_code = clamp_u8(slew_code == 0U ? PRESSURE_DVP_DEFAULT_SLEW_CODE : slew_code, 1U, SOLENOID_CURRENT_CODE_MAX);
    state.dvp_output_deadband = output_deadband;
    state.previous_target_raw = state.target_raw;
    pressure_control_reset_output_fields(&state);

    esp_err_t err = pressure_control_configure_channels(&state);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "MAX22200 DVP configure failed: %s", esp_err_to_name(err));
        state.state = PRESSURE_STATE_CONFIG_ERROR;
        state.error_code = 8;
        pressure_control_store_state(&state);
        return false;
    }
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
        state.state = PRESSURE_STATE_IDLE;
    }
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "DVP config: kp16=%u ki16=%u kd16=%u open=%u max=%u slew=%u deadband=%u",
             state.dvp_kp16, state.dvp_ki16, state.dvp_kd16, state.dvp_open_code, state.dvp_max_code,
             state.dvp_slew_code, state.dvp_output_deadband);
    return true;
}

/* fw 1.38: 16-bit gain set (command 0x27). Only swaps the PID gains; the
 * valve mapping (open/max/slew/deadband) keeps its current values. The
 * controller state is reset like any other reconfiguration. */
static bool pressure_control_apply_dvp_gains16(uint16_t kp16, uint16_t ki16, uint16_t kd16)
{
    dvp_drive_clear();
    pressure_control_t state = pressure_control_snapshot();
    state.dvp_kp16 = (kp16 == 0U) ? PRESSURE_DVP_DEFAULT_KP16 : kp16;
    state.dvp_ki16 = ki16;
    state.dvp_kd16 = kd16;
    state.previous_target_raw = state.target_raw;
    pressure_control_reset_output_fields(&state);

    esp_err_t err = pressure_control_configure_channels(&state);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "MAX22200 DVP gains16 configure failed: %s", esp_err_to_name(err));
        state.state = PRESSURE_STATE_CONFIG_ERROR;
        state.error_code = 8;
        pressure_control_store_state(&state);
        return false;
    }
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
        state.state = PRESSURE_STATE_IDLE;
    }
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "DVP gains16: kp16=%u ki16=%u kd16=%u", state.dvp_kp16, state.dvp_ki16, state.dvp_kd16);
    return true;
}

/* fw 1.39/1.40: output shaping (command 0x28) -- dither window length, per-update
 * code-jump limit, derivative low-pass depth, dead-time prediction horizon,
 * the derivative-fade band, and (fw 1.40) the asymmetric-outlet vent threshold.
 * Resets the controller like any reconfigure. */
static bool pressure_control_apply_dvp_shaping(uint8_t dither_n,
                                               uint8_t code_jump_max,
                                               uint8_t deriv_lp_div,
                                               uint8_t deadtime_steps,
                                               uint8_t d_fade_band_lsb,
                                               uint8_t vent_threshold_lsb)
{
    dvp_drive_clear();
    pressure_control_t state = pressure_control_snapshot();
    state.dvp_dither_n = clamp_u8(dither_n == 0U ? PRESSURE_DVP_DITHER_HISTORY_N : dither_n,
                                  1U, PRESSURE_DVP_DITHER_HISTORY_MAX);
    state.dvp_code_jump_max = code_jump_max; /* 0 = unlimited */
    state.dvp_deriv_lp_div = clamp_u8(deriv_lp_div == 0U ? PRESSURE_DVP_DERIV_LP_DIV_DEFAULT : deriv_lp_div,
                                      1U, PRESSURE_DVP_DERIV_LP_DIV_MAX);
    state.dvp_deadtime_steps = deadtime_steps > PRESSURE_DVP_DEADTIME_STEPS_MAX ? PRESSURE_DVP_DEADTIME_STEPS_MAX : deadtime_steps;
    state.dvp_d_fade_band_raw = (uint16_t)d_fade_band_lsb * PRESSURE_DVP_D_FADE_BAND_STEP_RAW;
    state.dvp_vent_threshold_pm = (uint16_t)vent_threshold_lsb * PRESSURE_DVP_VENT_THRESHOLD_STEP_PM;
    if (state.dvp_vent_threshold_pm > PRESSURE_DVP_VENT_THRESHOLD_MAX_PM) {
        state.dvp_vent_threshold_pm = PRESSURE_DVP_VENT_THRESHOLD_MAX_PM;
    }
    state.previous_target_raw = state.target_raw;
    pressure_control_reset_output_fields(&state);

    esp_err_t err = pressure_control_configure_channels(&state);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "MAX22200 DVP shaping configure failed: %s", esp_err_to_name(err));
        state.state = PRESSURE_STATE_CONFIG_ERROR;
        state.error_code = 8;
        pressure_control_store_state(&state);
        return false;
    }
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
        state.state = PRESSURE_STATE_IDLE;
    }
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "DVP shaping: dither_n=%u code_jump=%u deriv_lp_div=%u deadtime=%u d_fade_raw=%u vent_pm=%u",
             state.dvp_dither_n, state.dvp_code_jump_max, state.dvp_deriv_lp_div,
             state.dvp_deadtime_steps, state.dvp_d_fade_band_raw, state.dvp_vent_threshold_pm);
    return true;
}

/* fw 1.41: transient reference + outlet-range config (command 0x29). Light
 * apply -- only updates the trajectory/outlet params (no channel reconfigure,
 * no integral reset), so it can be tuned live during a hold. The next PID step
 * picks up the new values (store_task_state compares them). */
static bool pressure_control_apply_dvp_transient(uint8_t vmax_up, uint8_t vmax_down,
                                                 uint8_t dzone_up, uint8_t dzone_down,
                                                 uint8_t outlet_open, uint8_t outlet_max,
                                                 uint8_t ff_scale)
{
    pressure_control_t state = pressure_control_snapshot();
    state.dvp_vmax_up = vmax_up;
    state.dvp_vmax_down = vmax_down;
    state.dvp_dzone_up = dzone_up;
    state.dvp_dzone_down = dzone_down;
    state.dvp_outlet_max_code = clamp_u8(outlet_max == 0U ? PRESSURE_DVP_DEFAULT_OUTLET_MAX_CODE : outlet_max,
                                         1U, PRESSURE_DVP_HOST_MAX_CODE);
    state.dvp_outlet_open_code = clamp_u8(outlet_open == 0U ? PRESSURE_DVP_DEFAULT_OUTLET_OPEN_CODE : outlet_open,
                                          1U, state.dvp_outlet_max_code);
    state.dvp_ff_scale = ff_scale;
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "DVP transient: vmax_up=%u vmax_down=%u dzone_up=%u dzone_down=%u outlet_open=%u outlet_max=%u ff=%u",
             state.dvp_vmax_up, state.dvp_vmax_down, state.dvp_dzone_up, state.dvp_dzone_down,
             state.dvp_outlet_open_code, state.dvp_outlet_max_code, state.dvp_ff_scale);
    return true;
}

/* fw 1.42: flow-shaping config (command 0x2A). Light apply -- tunable live during
 * a hold (the next PID step picks it up). decel_up/down = 0 disables the taper
 * (instant full->hold ceiling). */
static bool pressure_control_apply_dvp_flowshape(uint8_t enable, uint8_t inlet_far_max,
                                                 uint8_t decel_up, uint8_t decel_down,
                                                 uint8_t lookahead, uint8_t seed_scale,
                                                 uint8_t rate_thresh)
{
    pressure_control_t state = pressure_control_snapshot();
    const bool was_enabled = state.dvp_fs_enable != 0U;
    state.dvp_fs_enable = enable ? 1U : 0U;
    state.dvp_fs_inlet_far_max = clamp_u8(inlet_far_max == 0U ? PRESSURE_DVP_DEFAULT_FS_INLET_FAR_MAX : inlet_far_max,
                                          1U, PRESSURE_DVP_HOST_MAX_CODE);
    state.dvp_fs_decel_up = decel_up;
    state.dvp_fs_decel_down = decel_down;
    state.dvp_fs_lookahead = lookahead;
    state.dvp_fs_seed_scale = seed_scale == 0U ? PRESSURE_DVP_DEFAULT_FS_SEED_SCALE : seed_scale;
    state.dvp_fs_rate_thresh = rate_thresh == 0U ? PRESSURE_DVP_DEFAULT_FS_RATE_THRESH : rate_thresh;
    /* On a disabled->enabled transition, force the integral seed to fire on the
     * next setpoint jump (the stored seed_target may be stale from a prior
     * flow-shaping session, which would skip the bumpless handoff). */
    if (state.dvp_fs_enable != 0U && !was_enabled) {
        state.dvp_fs_seed_target = 0U;
    }
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "DVP flowshape: en=%u inlet_far=%u decel_up=%u decel_down=%u look=%u seed=%u rate=%u",
             state.dvp_fs_enable, state.dvp_fs_inlet_far_max, state.dvp_fs_decel_up,
             state.dvp_fs_decel_down, state.dvp_fs_lookahead, state.dvp_fs_seed_scale,
             state.dvp_fs_rate_thresh);
    return true;
}

/* fw 1.42: hardware-dither config (command 0x2B). Updates the stored params,
 * applies them to the TLE channels immediately (so it can be tuned live), and
 * leaves them to be re-applied at each START. */
static bool pressure_control_apply_dvp_dither(uint8_t chmask, bool deep, uint8_t steps,
                                              uint8_t flat, uint16_t step_size,
                                              uint16_t mant, uint8_t exp)
{
    pressure_control_t state = pressure_control_snapshot();
    state.dvp_dither_chmask = chmask & 0x03U;
    state.dvp_dither_deep = deep ? 1U : 0U;
    state.dvp_dither_steps = steps;
    state.dvp_dither_flat = flat;
    state.dvp_dither_step_size = step_size & 0x0FFFU;
    /* The overlay rides ON TOP of the DC setpoint, so an unbounded amplitude
     * would defeat the current ceiling entirely (steps 255 x step_size 4095 is
     * orders of magnitude past full scale). Hold steps*step_size inside the
     * peak budget that PRESSURE_DVP_HOST_MAX_CODE was derated for; the default
     * 8 x 12 = 96 LSB (5.86 mA) already fits. */
    if (state.dvp_dither_steps != 0U &&
        ((uint32_t)state.dvp_dither_steps * state.dvp_dither_step_size) > PRESSURE_DVP_DITHER_PEAK_MAX_LSB) {
        state.dvp_dither_step_size =
            (uint16_t)(PRESSURE_DVP_DITHER_PEAK_MAX_LSB / state.dvp_dither_steps);
        ESP_LOGW(TAG, "DVP dither amplitude clamped: step_size -> %u (peak budget %u LSB)",
                 state.dvp_dither_step_size, (unsigned)PRESSURE_DVP_DITHER_PEAK_MAX_LSB);
    }
    state.dvp_dither_mant = mant & 0x03FFU;
    state.dvp_dither_exp = exp & 0x0FU;
    pressure_control_store_state(&state);
    pressure_control_apply_dither(&state);
    ESP_LOGI(TAG, "DVP dither: chmask=%u deep=%u steps=%u flat=%u step_size=%u mant=%u exp=%u",
             state.dvp_dither_chmask, state.dvp_dither_deep, state.dvp_dither_steps,
             state.dvp_dither_flat, state.dvp_dither_step_size, state.dvp_dither_mant, state.dvp_dither_exp);
    return true;
}

/* fw 1.44: DVP misc runtime switches (command 0x2D); currently just the integral
 * floor mode. LIGHT apply -- no controller reset, no integral clear, no channel
 * reconfigure and no forced valve-off: the next PID step simply uses the new
 * floor. Switching back to the legacy floor with a wound-negative integral
 * clamps it up to 0 so the loop does not have to swing back through zero. */
static bool pressure_control_apply_dvp_misc(uint8_t integral_floor_mode)
{
    if (integral_floor_mode == PRESSURE_DVP_INT_FLOOR_MODE_ZERO ||
        integral_floor_mode == PRESSURE_DVP_INT_FLOOR_MODE_SYMMETRIC) {
        s_dvp_int_floor_mode = integral_floor_mode;
        if (integral_floor_mode == PRESSURE_DVP_INT_FLOOR_MODE_ZERO) {
            portENTER_CRITICAL(&s_pressure_lock);
            if (s_pressure.dvp_integral_q16 < 0) {
                s_pressure.dvp_integral_q16 = 0;
            }
            portEXIT_CRITICAL(&s_pressure_lock);
        }
    }
    ESP_LOGI(TAG, "DVP misc: integral_floor_mode=%u", s_dvp_int_floor_mode);
    return true;
}

/* fw 1.44: manual constant-current drive (command 0x23). Only accepted while the
 * closed loop is not RUNNING/SETTLING. Arms the drive; the PID task's manual
 * supervisor (manual_drive_service) publishes it every step and owns expiry and
 * the inlet pressure ceiling, so a dead host always ends with the valves off. */
static uint8_t manual_drive_apply(uint8_t role, uint8_t code, uint16_t timeout_ms,
                                  uint16_t pressure_limit_raw)
{
    if (role > MANUAL_ROLE_OUTLET) {
        return COMMAND_STATUS_INVALID;
    }

    pressure_control_t state = pressure_control_snapshot();
    if (state.state == PRESSURE_STATE_RUNNING || state.state == PRESSURE_STATE_SETTLING) {
        return COMMAND_STATUS_BUSY;
    }

    /* Datasheet ceiling: never emit more than the DC budget whatever the host
     * asked for (this is the only host path that sets a code directly). */
    if (code > PRESSURE_DVP_HOST_MAX_CODE) {
        code = PRESSURE_DVP_HOST_MAX_CODE;
    }
    if (role == MANUAL_ROLE_OFF || code == 0U) {
        manual_drive_stop();
        ESP_LOGI(TAG, "Manual current: off");
        return COMMAND_STATUS_OK;
    }

    const uint16_t timeout = clamp_u16(timeout_ms == 0U ? MANUAL_TIMEOUT_DEFAULT_MS : timeout_ms,
                                       MANUAL_TIMEOUT_MIN_MS, MANUAL_TIMEOUT_MAX_MS);
    /* The inlet ceiling is the ONLY over-pressure protection while manual mode
     * owns the valves (the closed loop's PRESSURE_DVP_FAULT_MAX_RAW trip lives
     * inside the control step, which returns early while the controller is
     * IDLE). So a host value is never taken verbatim: it is clamped into
     * 1..FAULT_MAX_RAW, which makes a too-large limit (or a host that clamped a
     * psi->raw conversion to 0xFFFF, i.e. a guard that can never trip)
     * degenerate to the same absolute backstop the closed loop uses. */
    const uint16_t limit_raw = clamp_u16(pressure_limit_raw == 0U ? MANUAL_PRESSURE_LIMIT_DEFAULT_RAW
                                                                  : pressure_limit_raw,
                                         1U, PRESSURE_DVP_FAULT_MAX_RAW);
    const uint8_t channel = (role == MANUAL_ROLE_INLET) ? VALVE_CHANNEL_INLET : VALVE_CHANNEL_OUTLET;

    /* The TLE outputs may still be disarmed (VBAT ramped after boot); arming
     * here makes a manual drive work without a reflash, like START does. */
    if (!solenoid_is_armed()) {
        (void)solenoid_arm();
    }
    /* Manual current rides the proportional drive path, so both valve channels
     * must be in current regulation even if the stored valve mode is bang-bang.
     * The stored mode is left alone -- a later START reconfigures them. */
    if (state.valve_mode != PRESSURE_VALVE_MODE_DVP) {
        pressure_control_t dvp_view = state;
        dvp_view.valve_mode = PRESSURE_VALVE_MODE_DVP;
        const esp_err_t err = pressure_control_configure_channels(&dvp_view);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Manual current configure failed: %s", esp_err_to_name(err));
            manual_drive_stop();
            return COMMAND_STATUS_INVALID;
        }
    }

    portENTER_CRITICAL(&s_manual_lock);
    s_manual.active = 1;
    s_manual.role = role;
    s_manual.channel = channel;
    s_manual.code = code;
    s_manual.pressure_limit_raw = limit_raw;
    s_manual.expires_us = esp_timer_get_time() + ((int64_t)timeout * 1000);
    portEXIT_CRITICAL(&s_manual_lock);

    ESP_LOGI(TAG, "Manual current: role=%u ch%u code=%u (~%u mA) timeout=%ums limit=%u",
             role, channel, code, (unsigned)((((uint32_t)code * 200U) + 63U) / 127U), timeout, limit_raw);
    return COMMAND_STATUS_OK;
}

static bool pressure_control_request_start(uint16_t target_raw, uint16_t deadband_raw, uint8_t mode_request)
{
    manual_drive_stop(); /* the closed loop takes the valves back */
    dvp_drive_clear();

    pressure_control_t state = pressure_control_snapshot();
    if (mode_request == VALVE_MODE_REQUEST_DVP) {
        state.valve_mode = PRESSURE_VALVE_MODE_DVP;
    } else if (mode_request == VALVE_MODE_REQUEST_BANGBANG) {
        state.valve_mode = PRESSURE_VALVE_MODE_BANGBANG;
    }
    state.target_raw = pressure_control_clamp_target_raw(&state, target_raw);
    state.previous_target_raw = state.target_raw;
    state.deadband_raw = clamp_u16(deadband_raw == 0U ? PRESSURE_DEFAULT_DEADBAND_RAW : deadband_raw, 1U, UINT16_MAX);
    state.hold_current = clamp_u8(state.hold_current, 1U, state.hit_current);
    state.pending_valid = 0;

    /* Arm the driver before configuring channels: on the TLE92464 the boot-time
     * Mission-Mode attempt fails if VBAT had not ramped up yet, so the outputs
     * stay disabled until re-armed. Doing it here makes an explicit START
     * recover the valves without a reflash (the valve_task also re-arms ~1 Hz
     * while idle). No-op once armed / on the MAX22200. */
    if (!solenoid_is_armed()) {
        (void)solenoid_arm();
    }

    esp_err_t err = pressure_control_configure_channels(&state);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "MAX22200 pressure start configure failed: %s", esp_err_to_name(err));
        pressure_control_request_stop(PRESSURE_STATE_CONFIG_ERROR, 2);
        return false;
    }
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);

    adc_status_t adc = adc_status_snapshot();
    state.state = PRESSURE_STATE_RUNNING;
    pressure_control_reset_output_fields(&state);
    state.dvp_update_count = 0;
    state.dvp_late_updates = 0;
    state.dvp_max_service_us = 0;
    state.adc_average = adc.average;
    state.previous_average = adc.average;
    state.adc_samples = adc.samples;
    /* Seed the shaped reference to the live pressure so the trajectory starts
     * from where we are (not from 0 or a stale value); the step latch then
     * ramps it to the commanded target. */
    state.dvp_ref_pos = (float)adc.average;
    state.dvp_slewing = 0U;
    state.pulse_count = 0;
    pressure_control_store_state(&state);
    ESP_LOGI(TAG, "Pressure controller started (%s): target=%u deadband=%u",
             state.valve_mode == PRESSURE_VALVE_MODE_DVP ? "proportional" : "bang-bang",
             state.target_raw, state.deadband_raw);
    return true;
}

static bool pressure_control_update_target(uint16_t target_raw, uint16_t deadband_raw)
{
    pressure_control_t state = pressure_control_snapshot();
    if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
        return false;
    }
    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.target_raw = pressure_control_clamp_target_raw(&state, target_raw);
    s_pressure.deadband_raw = clamp_u16(deadband_raw == 0U ? PRESSURE_DEFAULT_DEADBAND_RAW : deadband_raw, 1U, UINT16_MAX);
    portEXIT_CRITICAL(&s_pressure_lock);
    return true;
}

/* Store a target without activating it; a SYNC frame activates it later. */
static void pressure_control_set_pending_target(uint16_t target_raw, uint16_t deadband_raw)
{
    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.pending_target_raw = target_raw;
    s_pressure.pending_deadband_raw = deadband_raw;
    s_pressure.pending_valid = 1;
    portEXIT_CRITICAL(&s_pressure_lock);
}

/* SYNC: activate the pending target (if any). Returns true if one applied. */
static bool pressure_control_activate_pending_target(void)
{
    pressure_control_t state = pressure_control_snapshot();
    if (state.pending_valid == 0U) {
        return false;
    }
    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.pending_valid = 0;
    portEXIT_CRITICAL(&s_pressure_lock);
    return pressure_control_update_target(state.pending_target_raw,
                                          state.pending_deadband_raw == 0U ? state.deadband_raw : state.pending_deadband_raw);
}

static uint16_t pressure_normalized_raw(const pressure_control_t *state, uint16_t raw)
{
    if ((state->flags & PRESSURE_CONFIG_FLAG_SENSOR_INCREASES) != 0U) {
        return raw;
    }
    return (uint16_t)(UINT16_MAX - raw);
}

/* Bang-bang pulse width: proportional to error within a deadband-derived
 * span, scaled for the asymmetric inlet/outlet authority, halved after a
 * direction reversal. */
static uint8_t pressure_compute_pulse_ms(const pressure_control_t *state, uint32_t abs_error, pressure_action_t action)
{
    uint32_t span = (uint32_t)state->deadband_raw * ((state->flags & PRESSURE_CONFIG_FLAG_FAST_TRACK) != 0U ? 4U : 8U);
    if (span < 2048U) {
        span = 2048U;
    }
    uint32_t limited_error = abs_error > span ? span : abs_error;
    uint32_t range = (uint32_t)state->max_pulse_ms - (uint32_t)state->min_pulse_ms;
    uint32_t pulse_ms = (uint32_t)state->min_pulse_ms + ((range * limited_error) / span);

    uint32_t pressure_proxy = pressure_normalized_raw(state, state->adc_average);
    uint32_t scale_percent = 100U;
    if (action == PRESSURE_ACTION_INLET) {
        scale_percent = 75U + ((pressure_proxy * 25U) / UINT16_MAX);
    } else if (action == PRESSURE_ACTION_OUTLET) {
        scale_percent = 75U + (((UINT16_MAX - pressure_proxy) * 25U) / UINT16_MAX);
    }
    pulse_ms = (pulse_ms * scale_percent) / 100U;

    if (state->last_direction != PRESSURE_ACTION_NONE && state->last_direction != (uint8_t)action) {
        pulse_ms = (pulse_ms + 1U) / 2U;
    }
    pulse_ms = pulse_ms < state->min_pulse_ms ? state->min_pulse_ms : pulse_ms;
    pulse_ms = pulse_ms > state->max_pulse_ms ? state->max_pulse_ms : pulse_ms;
    return (uint8_t)pulse_ms;
}

/* =====================================================================
 * CAN TX queue and telemetry
 * ===================================================================== */

static bool can_tx_queue_push(uint16_t can_id, const uint8_t *payload, uint8_t length)
{
    if (payload == NULL || length > 8U) {
        return false;
    }

    bool queued = false;
    portENTER_CRITICAL(&s_can_tx_queue_lock);
    if (s_can_tx_queue_count < CAN_TX_QUEUE_DEPTH) {
        can_tx_queue_entry_t *entry = &s_can_tx_queue[s_can_tx_queue_tail];
        entry->can_id = can_id;
        entry->length = length;
        memset(entry->payload, 0, sizeof(entry->payload));
        memcpy(entry->payload, payload, length);
        s_can_tx_queue_tail = (uint8_t)((s_can_tx_queue_tail + 1U) % CAN_TX_QUEUE_DEPTH);
        ++s_can_tx_queue_count;
        queued = true;
    }
    portEXIT_CRITICAL(&s_can_tx_queue_lock);
    return queued;
}

static bool can_tx_queue_peek(can_tx_queue_entry_t *entry)
{
    bool has_entry = false;
    portENTER_CRITICAL(&s_can_tx_queue_lock);
    if (s_can_tx_queue_count > 0U) {
        *entry = s_can_tx_queue[s_can_tx_queue_head];
        has_entry = true;
    }
    portEXIT_CRITICAL(&s_can_tx_queue_lock);
    return has_entry;
}

static void can_tx_queue_pop(void)
{
    portENTER_CRITICAL(&s_can_tx_queue_lock);
    if (s_can_tx_queue_count > 0U) {
        s_can_tx_queue_head = (uint8_t)((s_can_tx_queue_head + 1U) % CAN_TX_QUEUE_DEPTH);
        --s_can_tx_queue_count;
    }
    portEXIT_CRITICAL(&s_can_tx_queue_lock);
}

/* Set while a telemetry burst is being composed for USB only: the CAN copy
 * is dropped because a sync master owns the cycle or an update is running,
 * but the USB mirror costs the bus nothing and always goes out. */
static bool s_telemetry_usb_only;

static esp_err_t send_can_frame(const uint8_t *payload, uint8_t length)
{
    /* Mirror every device->host frame onto the USB link (no-op when no USB
     * host is talking), independent of the CAN queue state. */
    usb_link_send_frame(s_can_id_device_status, payload, length);

    if (s_telemetry_usb_only) {
        return ESP_OK;
    }

    if (can_tx_queue_push(s_can_id_device_status, payload, length)) {
        if (s_can_task_handle != NULL) {
            (void)xTaskNotify(s_can_task_handle, CAN_SERVICE_NOTIFY_BIT, eSetBits);
        }
        return ESP_OK;
    }
    /* One line per CAN_QUEUE_FULL_LOG_PERIOD_US, with the suppressed count.
     * This runs at can_task's priority (7) and a console write blocks until the
     * host drains it, so warning per dropped frame was itself part of the
     * starvation that killed USB. */
    const int64_t now_us = esp_timer_get_time();
    if ((now_us - s_can_queue_full_log_us) >= CAN_QUEUE_FULL_LOG_PERIOD_US) {
        ESP_LOGW(TAG, "CAN telemetry type 0x%02x queue full (%" PRIu32 " more suppressed)",
                 payload[0], s_can_queue_full_suppressed);
        s_can_queue_full_log_us = now_us;
        s_can_queue_full_suppressed = 0;
    } else {
        ++s_can_queue_full_suppressed;
    }
    return ESP_ERR_NO_MEM;
}

static esp_err_t can_ota_send_frame(uint16_t can_id, const uint8_t *payload, uint8_t length)
{
    usb_link_send_frame(can_id, payload, length);

    if (can_tx_queue_push(can_id, payload, length)) {
        if (s_can_task_handle != NULL) {
            (void)xTaskNotify(s_can_task_handle, CAN_SERVICE_NOTIFY_BIT, eSetBits);
        }
        return ESP_OK;
    }
    ESP_LOGW(TAG, "CAN OTA status queue full");
    return ESP_ERR_NO_MEM;
}

static void legacy_flush_tx(void);

/* Stop the loop and drop every output on a path that does not wait for the
 * control loop to make progress. */
static void control_stop_and_outputs_off(void)
{
    pressure_control_request_stop(PRESSURE_STATE_IDLE, 0);
    actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);
    ESP_ERROR_CHECK_WITHOUT_ABORT(solenoid_set_all_channels_off());
}

static void can_ota_safe_shutdown(void)
{
    control_stop_and_outputs_off();
}

/* =====================================================================
 * Hooks for the old-PCB-compatible protocol layer (tle_can_legacy.c)
 * ===================================================================== */

static uint16_t legacy_get_filtered_raw(void)
{
    const adc_status_t adc = adc_status_snapshot();
    return adc.samples != 0U ? adc.average : pressure_control_snapshot().adc_average;
}

static void legacy_set_target_raw(uint16_t raw)
{
    (void)pressure_control_update_target(raw, 0U);
}

static void legacy_set_enabled(bool enable, uint16_t target_raw)
{
    if (enable) {
        (void)pressure_control_request_start(target_raw, 0U, VALVE_MODE_REQUEST_KEEP);
    } else {
        control_stop_and_outputs_off();
    }
}

static bool legacy_is_running(void)
{
    return pressure_control_is_active();
}

static bool legacy_save_base_id(uint16_t base_id)
{
    return save_device_id_to_nvs(base_id);
}

/* Push the TX queue onto the wire from the calling task instead of leaving it
 * for the next pass of the CAN task.
 *
 * Frames are normally queued and drained by service_can_tx_slot, which is the
 * last thing can_task does each pass. That is fine until the frame is the
 * acknowledgement for a command whose next act is to reboot: the reboot runs
 * inside the receive handler, so can_task never reaches its transmit slot and
 * the acknowledgement dies in the queue. The host then sees a board that took
 * the command and never answered. Both reboot paths -- OTA end and set-ID --
 * flush before restarting. */
static void legacy_flush_tx(void)
{
    if (!s_mcp_ready || s_can_link_down) {
        return;
    }
    can_tx_queue_entry_t entry;
    for (uint8_t sent = 0; sent < CAN_TX_QUEUE_DEPTH; ++sent) {
        if (!can_tx_queue_peek(&entry)) {
            return;
        }
        (void)mcp2515_send_standard(&s_mcp, entry.can_id, entry.payload, entry.length);
        can_tx_queue_pop();
    }
}

static void send_command_status(uint8_t command, uint8_t status)
{
    uint8_t payload[8] = {TELEMETRY_TYPE_COMMAND_STATUS, command, status, 0, 0, 0, 0, 0};
    write_u16_le(&payload[3], s_can_device_base);
    write_u16_le(&payload[5], s_can_id_host_command);
    payload[7] = (uint8_t)(s_can_id_device_status & 0xFFU);
    (void)send_can_frame(payload, sizeof(payload));
}

/* The pressure feedback frame. */
static void send_pressure_control_status(void)
{
    pressure_control_t state = pressure_control_snapshot();
    adc_status_t adc = adc_status_snapshot();
    uint16_t average = adc.samples != 0U ? adc.average : state.adc_average;
    uint8_t payload[8] = {
        TELEMETRY_TYPE_PRESSURE_CONTROL,
        (uint8_t)state.state,
        (uint8_t)state.action,
        0,
        0,
        0,
        0,
        state.last_pulse_ms,
    };
    write_u16_le(&payload[3], state.target_raw);
    write_u16_le(&payload[5], average);
    (void)send_can_frame(payload, sizeof(payload));
}

static void send_pressure_dvp_status(void)
{
    pressure_control_t state = pressure_control_snapshot();
    const uint8_t fault_low_count = state.fault_low_count > 7U ? 7U : state.fault_low_count;
    const uint8_t fault_level = (uint8_t)solenoid_get_fault();
    const uint8_t packed_deadband_fault = (uint8_t)((state.dvp_output_deadband & 0x0FU)
                                                    | ((fault_low_count & 0x07U) << 4)
                                                    | (fault_level != 0U ? 0x80U : 0U));
    uint8_t payload[8] = {
        TELEMETRY_TYPE_PRESSURE_DVP,
        0,
        0,
        state.dvp_current_code,
        state.dvp_output_channel,
        /* byte 5: valve_mode in bits[1:0]; bit7 = driver armed (TLE Mission
         * Mode reached / MAX22200 always). Lets the host see the arm state
         * without the boot log (the USJ port re-enumerates on reset). */
        (uint8_t)(state.valve_mode | (solenoid_is_armed() ? 0x80U : 0U)),
        packed_deadband_fault,
        state.error_code,
    };
    write_i16_le(&payload[1], (int16_t)clamp_i32(state.dvp_output_permille, -1000, 1000));
    (void)send_can_frame(payload, sizeof(payload));
}

static void send_pressure_dvp_timing_status(void)
{
    pressure_control_t state = pressure_control_snapshot();
    const uint8_t period_ticks = PRESSURE_DVP_CURRENT_UPDATE_PERIOD_TICKS > UINT8_MAX ? UINT8_MAX : (uint8_t)PRESSURE_DVP_CURRENT_UPDATE_PERIOD_TICKS;
    uint8_t payload[8] = {TELEMETRY_TYPE_PRESSURE_DVP_TIMING, 0, 0, 0, 0, 0, 0, period_ticks};
    write_u16_le(&payload[1], (uint16_t)(state.dvp_update_count & 0xFFFFU));
    write_u16_le(&payload[3], (uint16_t)(state.dvp_late_updates & 0xFFFFU));
    write_u16_le(&payload[5], state.dvp_max_service_us);
    (void)send_can_frame(payload, sizeof(payload));
}

static void send_pressure_dvp_debug_status(void)
{
    pressure_control_t state = pressure_control_snapshot();
    uint8_t payload[8] = {TELEMETRY_TYPE_PRESSURE_DVP_DEBUG, 0, 0, 0, 0, state.dvp_sched_gain_q8_div2, 0, 0};
    write_i16_le(&payload[1], state.dvp_integral_permille);
    write_i16_le(&payload[3], state.dvp_derivative_permille);
    (void)send_can_frame(payload, sizeof(payload));
}

static void send_telemetry_burst(bool to_can)
{
    s_telemetry_usb_only = !to_can;
    send_pressure_control_status();
    send_pressure_dvp_status();
    send_pressure_dvp_timing_status();
    send_pressure_dvp_debug_status();
    s_telemetry_usb_only = false;
}

/* =====================================================================
 * CAN command handling (core 0, interrupt driven)
 * ===================================================================== */

/* Defined with the rest of the ADC plumbing further down. */
static uint8_t adc_capture_start(uint16_t n_samples);

static void handle_command(const mcp2515_frame_t *frame, bool *request_immediate_telemetry, bool is_broadcast)
{
    if (frame->dlc == 0) {
        *request_immediate_telemetry = true;
        return;
    }

    if (can_ota_in_progress() &&
        frame->data[0] != COMMAND_OTA_START &&
        frame->data[0] != COMMAND_OTA_END &&
        frame->data[0] != COMMAND_OTA_ABORT) {
        send_command_status(frame->data[0], COMMAND_STATUS_BUSY);
        *request_immediate_telemetry = true;
        return;
    }

    switch (frame->data[0]) {
    case COMMAND_REQUEST_TELEMETRY:
        *request_immediate_telemetry = true;
        break;

    case COMMAND_SET_PERIOD_MS:
        if (frame->dlc >= 3) {
            s_telemetry_period_ms = clamp_u16(read_u16_le(&frame->data[1]), TELEMETRY_PERIOD_MIN_MS, TELEMETRY_PERIOD_MAX_MS);
            *request_immediate_telemetry = true;
        }
        break;

    case COMMAND_SET_DEVICE_ID:
        if (is_broadcast || frame->dlc < 3) {
            send_command_status(COMMAND_SET_DEVICE_ID, COMMAND_STATUS_INVALID);
            *request_immediate_telemetry = true;
            break;
        }
        if (save_device_id_to_nvs(read_u16_le(&frame->data[1]))) {
            send_command_status(COMMAND_SET_DEVICE_ID, COMMAND_STATUS_OK);
            vTaskDelay(pdMS_TO_TICKS(100));
            esp_restart();
        } else {
            send_command_status(COMMAND_SET_DEVICE_ID, COMMAND_STATUS_INVALID);
            *request_immediate_telemetry = true;
        }
        break;

    case COMMAND_OTA_START:
    case COMMAND_OTA_END:
    case COMMAND_OTA_ABORT:
        if (is_broadcast) {
            send_command_status(frame->data[0], COMMAND_STATUS_INVALID);
            *request_immediate_telemetry = true;
        } else {
            (void)can_ota_handle_control_frame(frame->data, frame->dlc);
        }
        break;

    case COMMAND_PRESSURE_CONFIG: {
        uint8_t flags = frame->dlc >= 2 ? frame->data[1] : PRESSURE_DEFAULT_FLAGS;
        uint8_t hit_current = frame->dlc >= 3 ? frame->data[2] : PRESSURE_DEFAULT_HIT_CURRENT;
        uint8_t hold_current = frame->dlc >= 4 ? frame->data[3] : PRESSURE_DEFAULT_HOLD_CURRENT;
        uint8_t hit_time = frame->dlc >= 5 ? frame->data[4] : PRESSURE_DEFAULT_HIT_TIME;
        uint8_t min_pulse_ms = frame->dlc >= 6 ? frame->data[5] : PRESSURE_DEFAULT_MIN_PULSE_MS;
        uint8_t max_pulse_ms = frame->dlc >= 7 ? frame->data[6] : PRESSURE_DEFAULT_MAX_PULSE_MS;
        uint8_t settle_ms = frame->dlc >= 8 ? frame->data[7] : PRESSURE_DEFAULT_SETTLE_MS;
        (void)pressure_control_apply_config(flags, hit_current, hold_current, hit_time, min_pulse_ms, max_pulse_ms, settle_ms);
        send_pressure_control_status();
        *request_immediate_telemetry = true;
        break;
    }

    case COMMAND_PRESSURE_DVP_CONFIG: {
        uint8_t kp_q8 = frame->dlc >= 2 ? frame->data[1] : PRESSURE_DVP_DEFAULT_KP_Q8;
        uint8_t ki_q8 = frame->dlc >= 3 ? frame->data[2] : PRESSURE_DVP_DEFAULT_KI_Q8;
        uint8_t kd_q8 = frame->dlc >= 4 ? frame->data[3] : PRESSURE_DVP_DEFAULT_KD_Q8;
        uint8_t open_code = frame->dlc >= 5 ? frame->data[4] : PRESSURE_DVP_DEFAULT_OPEN_CODE;
        uint8_t max_code = frame->dlc >= 6 ? frame->data[5] : PRESSURE_DVP_DEFAULT_MAX_CODE;
        uint8_t slew_code = frame->dlc >= 7 ? frame->data[6] : PRESSURE_DVP_DEFAULT_SLEW_CODE;
        uint8_t output_deadband = frame->dlc >= 8 ? frame->data[7] : PRESSURE_DVP_DEFAULT_OUTPUT_DEADBAND;
        (void)pressure_control_apply_dvp_config(kp_q8, ki_q8, kd_q8, open_code, max_code, slew_code, output_deadband);
        send_pressure_dvp_status();
        *request_immediate_telemetry = true;
        break;
    }

    case COMMAND_PRESSURE_CONTROL: {
        uint8_t mode = frame->dlc >= 2 ? frame->data[1] : PRESSURE_CONTROL_MODE_STOP;
        pressure_control_t current = pressure_control_snapshot();
        uint16_t target_raw = frame->dlc >= 4 ? read_u16_le(&frame->data[2]) : current.target_raw;
        uint16_t deadband_raw = frame->dlc >= 6 ? read_u16_le(&frame->data[4]) : current.deadband_raw;
        uint8_t mode_request = frame->dlc >= 7 ? frame->data[6] : VALVE_MODE_REQUEST_KEEP;

        bool send_immediate_burst = true;
        if (mode == PRESSURE_CONTROL_MODE_STOP) {
            pressure_control_request_stop(PRESSURE_STATE_IDLE, 0);
        } else if (mode == PRESSURE_CONTROL_MODE_START) {
            (void)pressure_control_request_start(target_raw, deadband_raw, mode_request);
        } else if (mode == PRESSURE_CONTROL_MODE_SET_TARGET) {
            (void)pressure_control_update_target(target_raw, deadband_raw);
            send_immediate_burst = false;
        } else {
            pressure_control_request_stop(PRESSURE_STATE_CONFIG_ERROR, 3);
        }
        send_pressure_control_status();
        *request_immediate_telemetry = send_immediate_burst;
        break;
    }

    case COMMAND_PRESSURE_DVP_GAINS16:
        if (frame->dlc >= 7) {
            (void)pressure_control_apply_dvp_gains16(read_u16_le(&frame->data[1]),
                                                     read_u16_le(&frame->data[3]),
                                                     read_u16_le(&frame->data[5]));
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_PRESSURE_DVP_GAINS16, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_PRESSURE_DVP_SHAPING:
        if (frame->dlc >= 6) {
            /* byte 6 (vent threshold) is fw 1.40; absent on older hosts -> keep
             * the compiled default rather than forcing 0 (legacy outlet). */
            const uint8_t vent_lsb = frame->dlc >= 7
                ? frame->data[6]
                : (uint8_t)(PRESSURE_DVP_DEFAULT_VENT_THRESHOLD_PM / PRESSURE_DVP_VENT_THRESHOLD_STEP_PM);
            (void)pressure_control_apply_dvp_shaping(frame->data[1], frame->data[2], frame->data[3],
                                                     frame->data[4], frame->data[5], vent_lsb);
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_PRESSURE_DVP_SHAPING, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_PRESSURE_DVP_TRANSIENT:
        if (frame->dlc >= 7) {
            const uint8_t ff_scale = frame->dlc >= 8 ? frame->data[7] : PRESSURE_DVP_DEFAULT_FF_SCALE;
            (void)pressure_control_apply_dvp_transient(frame->data[1], frame->data[2], frame->data[3],
                                                       frame->data[4], frame->data[5], frame->data[6], ff_scale);
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_PRESSURE_DVP_TRANSIENT, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_PRESSURE_DVP_FLOWSHAPE:
        if (frame->dlc >= 5) {
            const uint8_t look = frame->dlc >= 6 ? frame->data[5] : PRESSURE_DVP_DEFAULT_FS_LOOKAHEAD;
            const uint8_t seed_scale = frame->dlc >= 7 ? frame->data[6] : PRESSURE_DVP_DEFAULT_FS_SEED_SCALE;
            const uint8_t rate_thresh = frame->dlc >= 8 ? frame->data[7] : PRESSURE_DVP_DEFAULT_FS_RATE_THRESH;
            (void)pressure_control_apply_dvp_flowshape(frame->data[1], frame->data[2],
                                                       frame->data[3], frame->data[4], look, seed_scale, rate_thresh);
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_PRESSURE_DVP_FLOWSHAPE, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_PRESSURE_DVP_DITHER:
        /* data[1]=chmask(bits1:0)|deep(bit7), [2]=steps, [3]=flat,
         * [4]=step_size lo8, [5]=step_size hi4, [6]=mant lo8,
         * [7]=mant hi2(bits1:0)|exp(bits5:2). */
        if (frame->dlc >= 8) {
            const uint8_t chmask = frame->data[1] & 0x03U;
            const bool deep = (frame->data[1] & 0x80U) != 0U;
            const uint16_t step_size = (uint16_t)frame->data[4] | ((uint16_t)(frame->data[5] & 0x0FU) << 8);
            const uint16_t mant = (uint16_t)frame->data[6] | ((uint16_t)(frame->data[7] & 0x03U) << 8);
            const uint8_t exp = (uint8_t)((frame->data[7] >> 2) & 0x0FU);
            (void)pressure_control_apply_dvp_dither(chmask, deep, frame->data[2], frame->data[3],
                                                    step_size, mant, exp);
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_PRESSURE_DVP_DITHER, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_ADC_FILTER:
        if (frame->dlc >= 2) {
            uint8_t shift = frame->data[1];
            if (shift > ADC_AVG_SHIFT_MAX) {
                shift = ADC_AVG_SHIFT_MAX;
            }
            s_adc_avg_shift = shift;
            ESP_LOGI(TAG, "ADC EMA shift = %u (TC ~%.1f ms)", shift, (1U << shift) * 1000.0 / PRESSURE_PID_HZ);
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_ADC_FILTER, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_PRESSURE_PENDING_TARGET:
        if (frame->dlc >= 3) {
            uint16_t target_raw = read_u16_le(&frame->data[1]);
            uint16_t deadband_raw = frame->dlc >= 5 ? read_u16_le(&frame->data[3]) : 0U;
            pressure_control_set_pending_target(target_raw, deadband_raw);
        } else {
            send_command_status(COMMAND_PRESSURE_PENDING_TARGET, COMMAND_STATUS_INVALID);
            *request_immediate_telemetry = true;
        }
        break;

    case COMMAND_SYNC:
        /* Activate the pending target (usually broadcast so every node on
         * the bus switches in the same instant) and answer with the
         * pressure feedback burst. */
        (void)pressure_control_activate_pending_target();
        *request_immediate_telemetry = true;
        break;

    case COMMAND_PRESSURE_DVP_MISC:
        /* [1] = integral floor mode (0 keep, 1 floor at 0, 2 symmetric). */
        if (frame->dlc >= 2) {
            (void)pressure_control_apply_dvp_misc(frame->data[1]);
            send_command_status(COMMAND_PRESSURE_DVP_MISC, COMMAND_STATUS_OK);
            send_pressure_dvp_status();
        } else {
            send_command_status(COMMAND_PRESSURE_DVP_MISC, COMMAND_STATUS_INVALID);
        }
        *request_immediate_telemetry = true;
        break;

    case COMMAND_ADC_RAW_CAPTURE: {
        /* [1..2] = u16le sample count (0 -> default). The samples are dumped to
         * the console by adc_dump_task once the capture completes. */
        const uint16_t n_samples = (uint16_t)((frame->dlc >= 2 ? (uint16_t)frame->data[1] : 0U) |
                                              (frame->dlc >= 3 ? ((uint16_t)frame->data[2] << 8) : 0U));
        send_command_status(COMMAND_ADC_RAW_CAPTURE, adc_capture_start(n_samples));
        *request_immediate_telemetry = true;
        break;
    }

    case COMMAND_MANUAL_CURRENT: {
        /* [1] = role (0 off, 1 inlet, 2 outlet), [2] = code, [3..4] = u16le
         * timeout_ms, [5..6] = u16le inlet pressure ceiling (raw). */
        if (frame->dlc < 3) {
            send_command_status(COMMAND_MANUAL_CURRENT, COMMAND_STATUS_INVALID);
            *request_immediate_telemetry = true;
            break;
        }
        const uint16_t timeout_ms = (uint16_t)((frame->dlc >= 4 ? (uint16_t)frame->data[3] : 0U) |
                                               (frame->dlc >= 5 ? ((uint16_t)frame->data[4] << 8) : 0U));
        const uint16_t limit_raw = (uint16_t)((frame->dlc >= 6 ? (uint16_t)frame->data[5] : 0U) |
                                              (frame->dlc >= 7 ? ((uint16_t)frame->data[6] << 8) : 0U));
        send_command_status(COMMAND_MANUAL_CURRENT,
                            manual_drive_apply(frame->data[1], frame->data[2], timeout_ms, limit_raw));
        *request_immediate_telemetry = true;
        break;
    }

    /* V1 test/diagnostic commands, removed in V2; answer DISABLED so old
     * host tools get a deterministic response. */
    case 0x02: /* set MAX CMD pin */
    case 0x03: /* MAX probe (needs SDO, dead on V2) */
    case 0x11: /* capture ping */
    case 0x20: /* manual valve sweep */
        send_command_status(frame->data[0], COMMAND_STATUS_DISABLED);
        *request_immediate_telemetry = true;
        break;

    default:
        break;
    }
}

/* Defined with the rest of the link-breaker state below. */
static void can_link_note_rx_activity(void);

static void service_can_rx(bool *request_immediate_telemetry)
{
    uint8_t interrupts = mcp2515_read_interrupts(&s_mcp);
    /* While the link is down the error flags are deliberately left standing:
     * they are masked out of CANINTE so they cannot raise INT, and re-reading
     * plus clearing them on every pass is exactly the SPI busywork the breaker
     * exists to stop. RX servicing below is untouched. */
    /* EFLG is examined whenever an error interrupt is flagged AND on a slow
     * unconditional beat. The beat is what makes overflow recovery reliable:
     * RXnOVR latches, a buffer that has overrun keeps rejecting frames until
     * the flag is cleared, and the controller goes on acknowledging those
     * frames on the wire in hardware -- so a transmitter sees success while
     * this board hears nothing. Recovery must not depend on any condition the
     * overflow itself removes. */
    const int64_t now_us = esp_timer_get_time();
    const bool error_flagged = (interrupts & (MCP2515_CANINTF_ERRIF | MCP2515_CANINTF_MERRF)) != 0;
    const bool sweep_due = (now_us - s_can_eflg_sweep_us) >= CAN_EFLG_SWEEP_PERIOD_US;
    if (!s_can_link_down && (error_flagged || sweep_due)) {
        s_can_eflg_sweep_us = now_us;
        const uint8_t eflg = mcp2515_read_error_flags(&s_mcp);
        if ((eflg & MCP2515_EFLG_RX_OVERFLOW) != 0) {
            (void)mcp2515_clear_rx_overflow(&s_mcp);
            tle_can_legacy_note_rx_overflow();
            if ((now_us - s_can_rx_overflow_log_us) >= CAN_RX_OVERFLOW_LOG_PERIOD_US) {
                ESP_LOGW(TAG, "MCP RX overflow cleared (EFLG=0x%02x, %" PRIu32 " more suppressed)",
                         eflg, s_can_rx_overflow_suppressed);
                s_can_rx_overflow_log_us = now_us;
                s_can_rx_overflow_suppressed = 0;
            } else {
                ++s_can_rx_overflow_suppressed;
            }
        } else if ((eflg & MCP2515_EFLG_ERRORMASK) != 0) {
            tle_can_legacy_note_bus_error(eflg);
        }
        /* EWARN/RXWAR/TXWAR alone mean the controller is counting errors but
         * is still fully on the bus. The old boards record those as warnings
         * and deliberately do not raise the protocol error flag, so neither
         * does this. */
        if (error_flagged) {
            (void)mcp2515_clear_interrupts(&s_mcp, MCP2515_CANINTF_ERRIF | MCP2515_CANINTF_MERRF);
        }
    }

    /* Drain both receive buffers before returning, re-reading CANINTF as we go.
     * The INT line is edge-triggered, so a frame left sitting in the second
     * buffer would not raise a new edge and would wait for the 1 ms fallback
     * wake. In a 150 Hz cycle this board receives its runtime-table frame and
     * then the sync frame back to back, and a millisecond of latency on the
     * sync edge is a millisecond of jitter on when the whole robot changes
     * target -- so they have to come out in one pass. */
    for (uint8_t serviced = 0; serviced < CAN_RX_SERVICE_MAX_FRAMES; ++serviced) {
        uint8_t rx_flag = 0;
        uint8_t buffer_index = 0;
        if ((interrupts & MCP2515_CANINTF_RX0IF) != 0) {
            rx_flag = MCP2515_CANINTF_RX0IF;
            buffer_index = 0;
        } else if ((interrupts & MCP2515_CANINTF_RX1IF) != 0) {
            rx_flag = MCP2515_CANINTF_RX1IF;
            buffer_index = 1;
        } else {
            break;
        }

        mcp2515_frame_t frame;
        esp_err_t err = mcp2515_read_rx_buffer(&s_mcp, buffer_index, &frame);
        (void)mcp2515_clear_interrupts(&s_mcp, rx_flag);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "MCP RXB%u read failed: %s", buffer_index, esp_err_to_name(err));
            interrupts = mcp2515_read_interrupts(&s_mcp);
            continue;
        }

        can_link_note_rx_activity();

        if (frame.extended || frame.rtr) {
            /* Nothing in either protocol uses extended IDs or remote frames. */
        } else if (tle_can_legacy_handle_frame(&frame)) {
            /* Shared-bus protocol: sync, runtime table, compact command,
             * legacy host control, OTA data. */
        } else if (frame.id == s_can_id_extended_command) {
            handle_command(&frame, request_immediate_telemetry, false);
        } else if (frame.id == s_can_id_host_ota_data) {
            /* Only reached when no legacy OTA session owns this ID, i.e. a
             * single-board bench flash driven by the native command set. */
            (void)can_ota_handle_data_frame(frame.data, frame.dlc);
        }

        interrupts = mcp2515_read_interrupts(&s_mcp);
    }
}

/* USB frames carry the same IDs and payloads as CAN frames and go through
 * the identical dispatch, from the same task. */
static void service_usb_rx(bool *request_immediate_telemetry)
{
    usb_link_frame_t usb_frame;
    while (usb_link_take_frame(&usb_frame)) {
        mcp2515_frame_t frame = {
            .id = usb_frame.id,
            .dlc = usb_frame.dlc,
            .extended = false,
            .rtr = false,
        };
        memcpy(frame.data, usb_frame.data, sizeof(frame.data));
        /* USB is point to point, so nothing is shared and base + 0x100 keeps
         * its original meaning here: every existing bench tool works unchanged.
         * base + 0x400 is accepted as well so one host library can address a
         * board over either transport with the same frames. */
        if (frame.id == s_can_id_host_command || frame.id == s_can_id_extended_command ||
            frame.id == CAN_ID_BROADCAST) {
            handle_command(&frame, request_immediate_telemetry, frame.id == CAN_ID_BROADCAST);
        } else if (frame.id == s_can_id_host_ota_data) {
            (void)can_ota_handle_data_frame(frame.data, frame.dlc);
        }
    }
}

/* ---------------------------------------------------------------------
 * CAN link breaker: see CAN_LINK_DOWN_BUSY_POLLS. Only can_task touches this
 * state, so no locking is needed.
 * --------------------------------------------------------------------- */

static void can_link_declare_down(int64_t now_us)
{
    s_can_link_down = true;
    s_can_link_probe_pending = false;
    s_can_link_probe_force = false;
    s_can_link_probe_us = now_us;
    s_can_tx_busy_polls = 0;
    /* Stop the controller retrying the frame nobody acknowledges, and make
     * sure the error flags cannot drive INT (and with it this task) while the
     * link is down. The RX enables are left alone. */
    (void)mcp2515_abort_pending_tx(&s_mcp);
    (void)mcp2515_mask_error_interrupts(&s_mcp);
    (void)mcp2515_clear_interrupts(&s_mcp, MCP2515_CANINTF_ERRIF | MCP2515_CANINTF_MERRF);
    ESP_LOGW(TAG, "CAN link down (no bus?): dropping CAN telemetry");
}

static void can_link_declare_up(void)
{
    s_can_link_down = false;
    s_can_link_probe_pending = false;
    s_can_tx_busy_polls = 0;
    /* Nothing to re-enable: ERRIE/MERRE are off in the driver's normal
     * configuration too, only the flag servicing in service_can_rx resumes. */
    (void)mcp2515_clear_interrupts(&s_mcp, MCP2515_CANINTF_ERRIF | MCP2515_CANINTF_MERRF);
    ESP_LOGI(TAG, "CAN link restored");
}

/* Hand exactly one real frame to the controller as a live-bus test; the
 * verdict is read out of TXB0CTRL on a later pass. A dead bus therefore costs
 * one stuck transmission per CAN_LINK_REPROBE_PERIOD_US and nothing else. */
static void can_link_probe_start(const can_tx_queue_entry_t *entry, int64_t now_us)
{
    s_can_link_probe_us = now_us;
    (void)mcp2515_abort_pending_tx(&s_mcp);
    if (mcp2515_try_send_standard(&s_mcp, entry->can_id, entry->payload, entry->length) == ESP_OK) {
        s_can_link_probe_pending = true;
    }
}

/* A frame arrived, so something out there is transmitting and the transceiver
 * is connected to a live bus. That is far better evidence than the re-probe
 * timer, and without it a host tool that opens its adapter has to wait out the
 * rest of a ten-second period before this board will answer anything. */
static void can_link_note_rx_activity(void)
{
    if (s_can_link_down && !s_can_link_probe_pending) {
        s_can_link_probe_force = true;
    }
}

static void can_link_service_down(int64_t now_us)
{
    if (s_can_link_probe_pending) {
        const esp_err_t result = mcp2515_tx_result(&s_mcp);
        if (result == ESP_OK) {
            can_link_declare_up();
            return;
        }
        if (result == ESP_ERR_TIMEOUT && (now_us - s_can_link_probe_us) < CAN_LINK_PROBE_TIMEOUT_US) {
            /* Verdict still pending. Hold the queue: if the bus turns out to
             * be alive, everything behind the probe is still worth sending. */
            return;
        }
        s_can_link_probe_pending = false;
        (void)mcp2515_abort_pending_tx(&s_mcp);
        (void)mcp2515_clear_interrupts(&s_mcp, MCP2515_CANINTF_ERRIF | MCP2515_CANINTF_MERRF);
    }

    can_tx_queue_entry_t entry;
    if (!can_tx_queue_peek(&entry)) {
        return;
    }

    if (s_can_link_probe_force || (now_us - s_can_link_probe_us) >= CAN_LINK_REPROBE_PERIOD_US) {
        /* Hand the real frame to the controller as the probe and pop it: it is
         * being transmitted either way, and if the bus is alive it arrives. */
        s_can_link_probe_force = false;
        can_link_probe_start(&entry, now_us);
        can_tx_queue_pop();
        if (s_can_link_probe_pending) {
            return;
        }
    }

    /* Still no bus. Drop the CAN copies -- send_can_frame mirrored every one of
     * these to the USB link when it queued them, so a USB host loses nothing,
     * and an empty queue keeps the queue-full path (console writes at priority
     * 7) out of the picture entirely. */
    for (uint8_t dropped = 0; dropped < CAN_TX_QUEUE_DEPTH; ++dropped) {
        if (!can_tx_queue_peek(&entry)) {
            break;
        }
        can_tx_queue_pop();
    }
}

static void service_can_tx_slot(bool *request_immediate_telemetry, int64_t *last_telemetry_us)
{
    const int64_t now_us = esp_timer_get_time();
    const uint16_t period_ms = s_telemetry_period_ms;
    /* The free-running burst is four frames per period per board. With eight
     * boards at the 20 ms default that is 1600 frames/s of diagnostics laid on
     * top of a bus doing real-time control, so it stops as soon as a sync
     * master is driving the cycle; the compact status reply is the per-cycle
     * feedback then. An explicit request still answers, which is how the
     * controller GUI reads the richer state of one board on demand. */
    const bool periodic_due = (now_us - *last_telemetry_us) >= ((int64_t)period_ms * 1000);
    if (*request_immediate_telemetry || periodic_due) {
        /* The CAN copy is dropped while a sync master owns the cycle, and
         * always during an update -- the broadcast stream needs the boards
         * silent. The USB mirror costs the bus nothing and always goes, so a
         * board on the production bus stays visible to the bench tools. */
        const bool to_can = !can_ota_in_progress() &&
                            !(periodic_due && !*request_immediate_telemetry &&
                              tle_can_legacy_bus_synced());
        send_telemetry_burst(to_can);
        *last_telemetry_us = now_us;
        *request_immediate_telemetry = false;
    }

    if (s_mcp_ready && s_can_link_down) {
        can_link_service_down(now_us);
        return;
    }

    can_tx_queue_entry_t entry;
    if (!can_tx_queue_peek(&entry)) {
        return;
    }

    if (!s_mcp_ready) {
        /* No CAN controller: drop the CAN copy (USB mirroring already
         * happened when the frame was queued). */
        can_tx_queue_pop();
        return;
    }

    const esp_err_t err = mcp2515_try_send_standard(&s_mcp, entry.can_id, entry.payload, entry.length);
    if (err == ESP_OK) {
        can_tx_queue_pop();
        s_can_tx_busy_polls = 0;
        /* Keep going while the controller keeps accepting. One frame per
         * wake meant a queued burst drained at the 1 ms fallback rate, and
         * a sync reply queued behind it would miss its cycle. A frame is
         * ~130 us on the wire, so the loop self-limits on ESP_ERR_TIMEOUT
         * long before it can hold this task open. */
        for (uint8_t more = 1; more < CAN_TX_BURST_MAX_FRAMES; ++more) {
            if (!can_tx_queue_peek(&entry) ||
                mcp2515_try_send_standard(&s_mcp, entry.can_id, entry.payload, entry.length) != ESP_OK) {
                break;
            }
            can_tx_queue_pop();
        }
    } else if (err == ESP_ERR_TIMEOUT) {
        /* TX buffer still busy, retry next wake -- unless it stays busy long
         * enough that nothing on the bus can be acknowledging us. */
        if (s_can_tx_busy_polls == 0U) {
            s_can_tx_busy_since_us = now_us;
        }
        ++s_can_tx_busy_polls;
        if (s_can_tx_busy_polls >= CAN_LINK_DOWN_BUSY_POLLS &&
            (now_us - s_can_tx_busy_since_us) >= CAN_LINK_DOWN_BUSY_US) {
            can_link_declare_down(now_us);
        }
    } else {
        can_tx_queue_pop();
        s_can_tx_busy_polls = 0;
        tle_can_legacy_note_tx_failure();
        ESP_LOGW(TAG, "CAN queued type 0x%02x send failed: %s", entry.payload[0], esp_err_to_name(err));
    }
}

static void IRAM_ATTR mcp_int_isr(void *arg)
{
    (void)arg;
    BaseType_t high_task_wakeup = pdFALSE;
    if (s_can_task_handle != NULL) {
        (void)xTaskNotifyFromISR(s_can_task_handle, CAN_SERVICE_NOTIFY_BIT, eSetBits, &high_task_wakeup);
    }
    if (high_task_wakeup == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

static void usb_link_wake_can_task(void)
{
    if (s_can_task_handle != NULL) {
        (void)xTaskNotify(s_can_task_handle, CAN_SERVICE_NOTIFY_BIT, eSetBits);
    }
}

/*
 * Core 0: CAN service. Owns the MCP25625 on its dedicated SPI2 bus. Driven
 * by the MCP INT ISR and TX-queue notifications (with a short fallback wait
 * bounding latency if a notify is ever missed), so command latency is
 * decoupled from valve/ADC timing.
 */
static void can_task(void *arg)
{
    (void)arg;
    int64_t last_telemetry_us = 0;
    bool request_immediate_telemetry = true;
    uint32_t notified_wakes = 0;

    while (true) {
        uint32_t notify_value = 0;
        if (xTaskNotifyWait(0, UINT32_MAX, &notify_value, pdMS_TO_TICKS(CAN_SERVICE_FALLBACK_PERIOD_MS)) == pdTRUE) {
            ++notified_wakes;
        } else {
            notified_wakes = 0; /* the wait timed out: the task slept */
        }
        /* Under an unforeseen notify storm the wait never blocks, so nothing
         * below priority 7 -- the USB parser at 3, IDLE0 -- would ever run on
         * this core. One forced tick of sleep per CAN_TASK_BREATHER_ITERATIONS
         * costs nothing when notifies are sparse (the counter is reset by the
         * first wait that times out, i.e. every ~1 ms in normal operation). */
        if (notified_wakes >= CAN_TASK_BREATHER_ITERATIONS) {
            notified_wakes = 0;
            vTaskDelay(1);
        }

        if (s_mcp_ready) {
            service_can_rx(&request_immediate_telemetry);
        }
        service_usb_rx(&request_immediate_telemetry);
        tle_can_legacy_service();
        service_can_tx_slot(&request_immediate_telemetry, &last_telemetry_us);
    }
}

/* =====================================================================
 * ADC sampling and filtering (core 1 sampling, core 0 filtering + PID)
 * ===================================================================== */

static void adc_pid_buffer_push_sample(uint16_t raw)
{
    portENTER_CRITICAL(&s_adc_pid_lock);
    s_adc_pid_buffer.latest_raw = raw;
    ++s_adc_pid_buffer.raw_samples;
    if (s_adc_pid_buffer.count < ADC_PID_WINDOW_SAMPLES) {
        s_adc_pid_buffer.samples[s_adc_pid_buffer.count] = raw;
        ++s_adc_pid_buffer.count;
    } else {
        for (uint8_t index = 1U; index < ADC_PID_WINDOW_SAMPLES; ++index) {
            s_adc_pid_buffer.samples[index - 1U] = s_adc_pid_buffer.samples[index];
        }
        s_adc_pid_buffer.samples[ADC_PID_WINDOW_SAMPLES - 1U] = raw;
    }
    portEXIT_CRITICAL(&s_adc_pid_lock);
}

static uint8_t adc_pid_buffer_take_samples(uint16_t *samples, uint16_t *latest_raw, uint32_t *raw_samples)
{
    uint8_t count = 0;
    portENTER_CRITICAL(&s_adc_pid_lock);
    count = s_adc_pid_buffer.count;
    for (uint8_t index = 0; index < count; ++index) {
        samples[index] = s_adc_pid_buffer.samples[index];
    }
    *latest_raw = s_adc_pid_buffer.latest_raw;
    *raw_samples = s_adc_pid_buffer.raw_samples;
    s_adc_pid_buffer.count = 0;
    portEXIT_CRITICAL(&s_adc_pid_lock);
    return count;
}

static uint16_t adc_raw_delta(uint16_t left, uint16_t right)
{
    return (left > right) ? (uint16_t)(left - right) : (uint16_t)(right - left);
}

static void adc_sort_samples(uint16_t *values, uint8_t count)
{
    for (uint8_t index = 1; index < count; ++index) {
        const uint16_t value = values[index];
        uint8_t insert_index = index;
        while (insert_index > 0U && values[insert_index - 1U] > value) {
            values[insert_index] = values[insert_index - 1U];
            --insert_index;
        }
        values[insert_index] = value;
    }
}

static uint16_t adc_average_sum(uint32_t sum, uint8_t count)
{
    return (uint16_t)((sum + ((uint32_t)count / 2U)) / count);
}

/* Median outlier rejection: average the samples within
 * ADC_FILTER_REJECT_DELTA_RAW of the window median (valve switching injects
 * spikes on the sensor); fall back to a trimmed mean if too few survive. */
static uint16_t adc_filter_window_average(const uint16_t *samples, uint8_t count)
{
    if (count == 0U) {
        return 0;
    }
    if (count == 1U) {
        return samples[0];
    }

    uint16_t sorted[ADC_PID_WINDOW_SAMPLES];
    for (uint8_t index = 0; index < count; ++index) {
        sorted[index] = samples[index];
    }
    adc_sort_samples(sorted, count);

    const uint16_t median = sorted[count / 2U];
    uint32_t accepted_sum = 0;
    uint8_t accepted_count = 0;
    for (uint8_t index = 0; index < count; ++index) {
        if (adc_raw_delta(sorted[index], median) <= ADC_FILTER_REJECT_DELTA_RAW) {
            accepted_sum += sorted[index];
            ++accepted_count;
        }
    }
    if (accepted_count >= ADC_FILTER_MIN_ACCEPTED_SAMPLES) {
        return adc_average_sum(accepted_sum, accepted_count);
    }

    uint8_t trim = ADC_FILTER_TRIM_SAMPLES;
    if (count <= (uint8_t)(trim * 2U)) {
        trim = 0;
    }

    uint32_t trimmed_sum = 0;
    uint8_t trimmed_count = 0;
    for (uint8_t index = trim; index < (uint8_t)(count - trim); ++index) {
        trimmed_sum += sorted[index];
        ++trimmed_count;
    }
    return adc_average_sum(trimmed_sum, trimmed_count);
}

/* =====================================================================
 * Raw ADC capture (command 0x10)
 * ===================================================================== */

/* The 3 kHz tap. Called from adc_sample_task only: one volatile load, a bounds
 * check and a store, so the sampling path keeps its timing. */
static inline void adc_capture_push(uint16_t raw)
{
    if (s_adc_capture_state != ADC_CAPTURE_STATE_RUNNING) {
        return;
    }
    const uint32_t index = s_adc_capture_count;
    if (index < s_adc_capture_want) {
        s_adc_capture_buffer[index] = raw;
        s_adc_capture_count = index + 1U;
    }
    if (s_adc_capture_count >= s_adc_capture_want) {
        /* Hand the buffer to the dump task; the sampling path is free again. */
        s_adc_capture_state = ADC_CAPTURE_STATE_FULL;
    }
}

/* Arm a capture (CAN/USB command context). The buffer is allocated once, at the
 * maximum size, and kept for the life of the run. */
static uint8_t adc_capture_start(uint16_t n_samples)
{
    if (s_adc_capture_state != ADC_CAPTURE_STATE_IDLE) {
        return COMMAND_STATUS_BUSY;
    }

    uint32_t want = (n_samples == 0U) ? ADC_CAPTURE_DEFAULT_SAMPLES : (uint32_t)n_samples;
    if (want < ADC_CAPTURE_MIN_SAMPLES) {
        want = ADC_CAPTURE_MIN_SAMPLES;
    } else if (want > ADC_CAPTURE_MAX_SAMPLES) {
        want = ADC_CAPTURE_MAX_SAMPLES;
    }

    if (s_adc_capture_buffer == NULL) {
        s_adc_capture_buffer = heap_caps_malloc(ADC_CAPTURE_MAX_SAMPLES * sizeof(uint16_t),
                                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (s_adc_capture_buffer == NULL) {
            ESP_LOGW(TAG, "ADC raw capture: cannot allocate %u bytes",
                     (unsigned)(ADC_CAPTURE_MAX_SAMPLES * sizeof(uint16_t)));
            return COMMAND_STATUS_INVALID;
        }
    }

    s_adc_capture_count = 0;
    s_adc_capture_want = want;
    /* Publish the buffer and the counters before the ADC task (other core) can
     * observe the RUNNING state. */
    __sync_synchronize();
    s_adc_capture_state = ADC_CAPTURE_STATE_RUNNING;
    ESP_LOGI(TAG, "ADC raw capture armed: %" PRIu32 " samples @ %uHz", want, ADC_SAMPLE_HZ);
    return COMMAND_STATUS_OK;
}

/*
 * Low-priority console dump. Never runs in the ADC task or an ISR: it waits for
 * the capture to fill, then writes the samples out in short ASCII lines,
 * yielding every few lines so USB RX servicing, the keepalive, the PID loop and
 * the idle/watchdog tasks all keep running through a multi-second dump.
 */
static void adc_dump_task(void *arg)
{
    (void)arg;
    char line[192];

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(ADC_CAPTURE_POLL_MS));
        if (s_adc_capture_state != ADC_CAPTURE_STATE_FULL) {
            continue;
        }
        s_adc_capture_state = ADC_CAPTURE_STATE_DUMPING;

        const uint32_t count = s_adc_capture_count;
        uint32_t emitted_lines = 0;
        for (uint32_t start = 0; start < count; start += ADC_CAPTURE_VALUES_PER_LINE) {
            int written = snprintf(line, sizeof(line), "#RAW,%" PRIu32, start);
            size_t used = (written > 0) ? (size_t)written : 0U;
            for (uint32_t index = start;
                 index < count && index < (start + ADC_CAPTURE_VALUES_PER_LINE);
                 ++index) {
                if (used >= sizeof(line) - 1U) {
                    break;
                }
                written = snprintf(line + used, sizeof(line) - used, ",%u",
                                   (unsigned)s_adc_capture_buffer[index]);
                if (written <= 0) {
                    break;
                }
                used += (size_t)written;
                if (used > sizeof(line) - 1U) {
                    used = sizeof(line) - 1U;
                    break;
                }
            }
            /* Terminate inside the buffer: newlib's FILE lock only serialises
             * one stdio call, so a separate fputc('\n') lets another task's
             * ESP_LOG output splice into the middle of a dump line and corrupt
             * the sample it lands on. One locked write per line instead.
             * (Worst case is ~107 of the 192 bytes, so there is always room.) */
            if (used < sizeof(line) - 1U) {
                line[used++] = '\n';
            } else {
                line[sizeof(line) - 1U] = '\n';
                used = sizeof(line);
            }
            (void)fwrite(line, 1, used, stdout);
            if ((++emitted_lines % ADC_CAPTURE_LINES_PER_YIELD) == 0U) {
                (void)fflush(stdout);
                vTaskDelay(1);
            }
        }
        (void)fprintf(stdout, "#RAWEND,%" PRIu32 ",%u\n", count, ADC_SAMPLE_HZ);
        (void)fflush(stdout);
        ESP_LOGI(TAG, "ADC raw capture dumped: %" PRIu32 " samples", count);
        s_adc_capture_state = ADC_CAPTURE_STATE_IDLE;
    }
}

/*
 * Core 1: LTC1864 sampling at ADC_SAMPLE_HZ, notified by the GPTimer ISR.
 * Shares its core (and the SPI3 bus arbitration) only with valve_task so the
 * bus is serviced with tight timing.
 */
static void adc_sample_task(void *arg)
{
    (void)arg;

    while (true) {
        (void)ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        uint16_t raw = 0;
        esp_err_t err = ltc1864_read_raw(&raw);
        if (err == ESP_OK) {
            adc_capture_push(raw);
            adc_pid_buffer_push_sample(raw);
        } else {
            portENTER_CRITICAL(&s_adc_pid_lock);
            ++s_adc_pid_buffer.read_errors;
            portEXIT_CRITICAL(&s_adc_pid_lock);
            ESP_LOGW(TAG, "LTC1864 read failed: %s", esp_err_to_name(err));
            if (pressure_control_is_active() && adc_feedback_is_stale()) {
                ESP_LOGW(TAG, "Pressure controller stopped: stale ADC feedback");
                pressure_control_request_stop(PRESSURE_STATE_STALE_ADC, 5);
            }
        }
    }
}

static void pressure_control_step_from_adc(uint16_t adc_average, uint32_t adc_samples);
static void manual_drive_service(uint16_t filtered_raw);

/*
 * Core 0: PID. Consumes the sample window at PRESSURE_PID_HZ, filters it
 * (median reject + fixed-gain exponential average), then runs the control
 * step. Lower priority than can_task, so CAN handling preempts it.
 */
static void pressure_pid_task(void *arg)
{
    (void)arg;
    bool ema_init = false;
    int32_t ema_q16 = 0;
    uint16_t last_average = 0;

    while (true) {
        (void)ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        uint16_t samples[ADC_PID_WINDOW_SAMPLES];
        uint16_t raw = 0;
        uint32_t raw_samples = 0;
        const uint8_t sample_count = adc_pid_buffer_take_samples(samples, &raw, &raw_samples);
        if (sample_count > 0U) {
            const uint16_t window_average = adc_filter_window_average(samples, sample_count);
            /* Value-form EMA (state = average << 16): avg += (x - avg) >> shift.
             * This lets the smoothing depth change at runtime (command 0x2C)
             * without corrupting the accumulator. shift 0 = no smoothing (lowest
             * lag). The per-window median above already rejects valve spikes, so
             * this only trades residual noise vs lag. */
            uint8_t shift = s_adc_avg_shift;
            if (shift > ADC_AVG_SHIFT_MAX) {
                shift = ADC_AVG_SHIFT_MAX;
            }
            if (!ema_init) {
                ema_q16 = (int32_t)((uint32_t)window_average << 16);
                ema_init = true;
            } else {
                ema_q16 += ((int32_t)((uint32_t)window_average << 16) - ema_q16) >> shift;
            }
            uint16_t average = (uint16_t)(((uint32_t)ema_q16 + 0x8000U) >> 16);
            last_average = average;
            adc_status_update(raw, average, raw_samples);
            pressure_control_step_from_adc(average, raw_samples);
        } else if (pressure_control_is_active() && adc_feedback_is_stale()) {
            ESP_LOGW(TAG, "Pressure controller stopped: stale PID feedback");
            pressure_control_request_stop(PRESSURE_STATE_STALE_ADC, 5);
        }
        /* fw 1.44: the manual bench drive (command 0x23) is supervised on EVERY
         * PID tick, including ticks with no fresh ADC window -- its timeout is
         * the dead-host failsafe and has to fire even if the sensor goes quiet.
         * It is a no-op unless a manual drive is armed, and a manual drive can
         * only be armed while the closed loop is not running. */
        manual_drive_service(last_average);
    }
}

/* =====================================================================
 * Proportional (DVP) output shaping
 * ===================================================================== */

static uint16_t dvp_slew_code_q(uint16_t current_code_q, uint16_t desired_code_q, uint8_t slew_code)
{
    const uint16_t slew_q = (uint16_t)slew_code << PRESSURE_DVP_CODE_FRAC_BITS;
    if (desired_code_q > current_code_q) {
        uint32_t next_code_q = (uint32_t)current_code_q + slew_q;
        return next_code_q > desired_code_q ? desired_code_q : (uint16_t)next_code_q;
    }
    if (desired_code_q < current_code_q) {
        if ((current_code_q - desired_code_q) <= slew_q) {
            return desired_code_q;
        }
        return current_code_q - slew_q;
    }
    return current_code_q;
}

/* Map |output| in permille onto the open_code..max_code current span (Q9).
 * open/max are passed explicitly so the inlet and outlet can use different
 * current ranges (the outlet vents at full authority for a fast fall). */
static uint16_t dvp_active_code_q_for_output(uint8_t open_code, uint8_t max_code, uint32_t active_permille)
{
    if (active_permille > 1000U) {
        active_permille = 1000U;
    }

    if (open_code > max_code) {
        open_code = max_code;
    }

    const uint32_t range = (uint32_t)max_code - (uint32_t)open_code;
    const uint32_t open_code_q = (uint32_t)open_code << PRESSURE_DVP_CODE_FRAC_BITS;
    const uint32_t max_code_q = (uint32_t)max_code << PRESSURE_DVP_CODE_FRAC_BITS;
    uint64_t code_q = (uint64_t)open_code_q +
                      ((((uint64_t)range * active_permille * PRESSURE_DVP_CODE_Q_SCALE) + 500U) / 1000U);
    if (code_q > max_code_q) {
        code_q = max_code_q;
    }
    return (uint16_t)code_q;
}

/* Inside the soft zone (deadband-sized) the P and I contributions fade
 * linearly down to PRESSURE_DVP_SOFT_ZONE_MIN_GAIN_Q8/256.
 *
 * fw 1.38 note: the former "direction guard" (drop the integral whenever the
 * combined PI drive disagreed with the error sign) is gone. On this plant the
 * load leaks continuously, so the equilibrium output is a standing positive
 * integral (inlet metering the leak make-up flow); cutting that feed every
 * time the pressure peeked above target turned the loop into a relay
 * oscillator (~0.16 psi p2p limit cycle). Reversal safety is provided by the
 * stale-integral reset (error >= 4x soft zone with opposite-signed integral)
 * plus the much tighter +/-2x-full-scale integral clamp. */
static uint32_t dvp_soft_zone_gain_q8(uint32_t abs_error, uint16_t soft_zone_raw)
{
    if (soft_zone_raw == 0U || abs_error >= soft_zone_raw) {
        return 256U;
    }
    const uint32_t gain_range_q8 = 256U - PRESSURE_DVP_SOFT_ZONE_MIN_GAIN_Q8;
    return PRESSURE_DVP_SOFT_ZONE_MIN_GAIN_Q8 + ((gain_range_q8 * abs_error) / soft_zone_raw);
}

static void dvp_prepare_output(pressure_control_t *state, int32_t desired_permille,
                               uint8_t inlet_max_eff, uint8_t outlet_max_eff)
{
    desired_permille = clamp_i32(desired_permille, -1000, 1000);

    /* Asymmetric split-range. Inlet for positive drive beyond the output
     * deadband. The outlet only engages once the drive falls below
     * -(vent_threshold + deadband); the magnitude mapped onto [open,max] is
     * measured from that threshold so venting ramps in smoothly. Between the
     * two thresholds the loop coasts (all valves off) and the fast plant leak
     * provides the gentle downward correction. This removes the inlet<->outlet
     * relay limit cycle near low setpoints, where the steady leak-make-up
     * output sits close to zero permille and any ripple would otherwise keep
     * crossing into the outlet. (vent_threshold = 0 restores legacy behaviour:
     * the outlet engages on any negative drive.) */
    const int32_t vent_threshold = (int32_t)state->dvp_vent_threshold_pm;
    pressure_action_t action;
    uint8_t channel;
    uint32_t abs_permille;
    if (desired_permille > (int32_t)state->dvp_output_deadband) {
        action = PRESSURE_ACTION_INLET;
        channel = VALVE_CHANNEL_INLET;
        abs_permille = (uint32_t)desired_permille;
    } else if (desired_permille < -(vent_threshold + (int32_t)state->dvp_output_deadband)) {
        action = PRESSURE_ACTION_OUTLET;
        channel = VALVE_CHANNEL_OUTLET;
        abs_permille = (uint32_t)(-desired_permille - vent_threshold);
        if (abs_permille > 1000U) {
            abs_permille = 1000U;
        }
    } else {
        /* Coast band: neither valve drives; the leak bleeds pressure down. */
        dvp_drive_clear();
        if (state->output_on != 0U) {
            actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);
        }
        state->action = PRESSURE_ACTION_DEADBAND;
        state->output_channel = VALVE_CHANNEL_NONE;
        state->output_on = 0;
        state->last_pulse_ms = 0;
        state->dvp_current_code = 0;
        state->dvp_output_channel = VALVE_CHANNEL_NONE;
        state->dvp_target_code_q = 0;
        state->dvp_applied_code_q = 0;
        state->dvp_output_permille = desired_permille;
        return;
    }

    /* Per-channel current range: the outlet gets its own (full-authority)
     * open/max so a down-step vents as fast as the valve allows, while the
     * inlet keeps its narrow crack-region range for fine hold resolution. */
    const uint8_t chan_open = (channel == VALVE_CHANNEL_OUTLET) ? state->dvp_outlet_open_code : state->dvp_open_code;
    /* Effective per-channel current ceiling. With flow shaping the caller passes
     * a distance-scaled ceiling (full when far -> hold range near target); with
     * it off these are just the configured maxima. */
    uint8_t chan_max = (channel == VALVE_CHANNEL_OUTLET) ? outlet_max_eff : inlet_max_eff;
    if (chan_max < chan_open) {
        chan_max = chan_open;
    }

    const bool activating_output = state->output_on == 0U || state->output_channel != channel;
    if (state->output_on != 0U && state->output_channel != channel) {
        state->dvp_current_code = 0;
        state->dvp_target_code_q = 0;
    }

    const uint16_t desired_code_q = dvp_active_code_q_for_output(chan_open, chan_max, abs_permille);
    const uint16_t next_code_q = dvp_slew_code_q(state->dvp_target_code_q, desired_code_q, state->dvp_slew_code);
    dvp_drive_publish(channel, next_code_q, chan_open, chan_max,
                      state->dvp_dither_n, state->dvp_code_jump_max);

    state->state = PRESSURE_STATE_RUNNING;
    state->action = action;
    state->last_direction = (uint8_t)action;
    state->output_channel = channel;
    state->output_on = 1;
    state->last_pulse_ms = 0;
    if (activating_output && state->dvp_current_code == 0U) {
        uint8_t telemetry_code = chan_open;
        if (telemetry_code > chan_max) {
            telemetry_code = chan_max;
        }
        state->dvp_current_code = telemetry_code;
    }
    state->dvp_output_channel = channel;
    state->dvp_target_code_q = next_code_q;
    state->dvp_output_permille = desired_permille;
}

/* =====================================================================
 * The control step (PID / bang-bang), at PRESSURE_PID_HZ on core 0
 * ===================================================================== */

/* fw 1.44: PRESSURE_PID_HZ supervision of the manual (open-loop) drive, command
 * 0x23. Called from the PID task on every tick (a no-op unless a drive is
 * armed, and one can only be armed while the closed loop is not running): it
 * republishes the drive through dvp_drive_publish -- keeping valve_task's
 * one-valve-at-a-time interlock the single authority -- expires it on the host
 * timeout, and cuts it on the inlet pressure ceiling. `filtered_raw` is the EMA
 * output, in the same raw space as target_raw. */
static void manual_drive_service(uint16_t filtered_raw)
{
    const manual_drive_t manual = manual_drive_snapshot();
    if (manual.active == 0U) {
        return;
    }

    /* Every failsafe branch de-energises FIRST and logs second: ESP_LOG goes to
     * the shared USB-Serial/JTAG console, which blocks for up to the TX flush
     * timeout when the host has stopped draining -- which is precisely the
     * dead-host case the timeout exists for. Cutting the coil must not wait on
     * the stalled link it is protecting against. */
    if (esp_timer_get_time() >= manual.expires_us) {
        manual_drive_stop();
        ESP_LOGI(TAG, "Manual current expired (no host refresh)");
        return;
    }
    /* A manual inlet drive is guarded by `filtered_raw`, which the PID task only
     * refreshes when an ADC window arrives -- if the sensor dies the value
     * FREEZES at its last good reading and the guard silently stops guarding.
     * The closed loop's stale-feedback abort cannot cover this (it is gated on
     * pressure_control_is_active(), false by construction in manual mode), so
     * failing feedback must drop the manual drive here, exactly as it drops the
     * closed loop. Only the inlet is affected: an outlet drive vents to
     * atmosphere and stays usable (and useful) with a dead sensor. */
    if (manual.role == MANUAL_ROLE_INLET && adc_feedback_is_stale()) {
        manual_drive_stop();
        ESP_LOGW(TAG, "Manual inlet cut: stale ADC feedback");
        return;
    }
    if (manual.role == MANUAL_ROLE_INLET && filtered_raw >= manual.pressure_limit_raw) {
        manual_drive_stop();
        ESP_LOGW(TAG, "Manual inlet cut: pressure %u >= limit %u", filtered_raw, manual.pressure_limit_raw);
        return;
    }

    /* Constant code: pin min == max so valve_task applies exactly this code
     * (the window dither and the per-update jump limit are both neutralised). */
    const uint16_t code_q = (uint16_t)((uint16_t)manual.code << PRESSURE_DVP_CODE_FRAC_BITS);
    dvp_drive_publish(manual.channel, code_q, manual.code, manual.code,
                      PRESSURE_DVP_DITHER_HISTORY_N, 0U);

    /* TOCTOU guard, and the reason this whole function is safe. can_task runs at
     * a HIGHER priority on this same core, so a host STOP / 0x22 START / 0x23
     * role=0 can land between the snapshot above and the publish just executed:
     * manual_drive_stop() clears the drive first and this publish then re-arms
     * it -- leaving the valve energised with the timeout, the pressure ceiling
     * AND the closed-loop overpressure trip all disarmed, and no code path left
     * that would ever clear it again.
     *
     * So re-read the armed state and undo the publish unless it is still exactly
     * what the armed drive wants. Comparing the published parameters rather than
     * a change counter means an ordinary sustain re-send (identical role+code,
     * new expiry) is NOT undone -- no 1.33 ms dropout on the ~0.1 % of re-sends
     * that land in this window -- while every genuine cancellation is. The
     * invariant on return is: the published drive is either what s_manual asks
     * for, or nothing. */
    portENTER_CRITICAL(&s_manual_lock);
    const bool still_armed = (s_manual.active != 0U) &&
                             (s_manual.channel == manual.channel) &&
                             (s_manual.code == manual.code);
    portEXIT_CRITICAL(&s_manual_lock);
    if (!still_armed) {
        dvp_drive_clear();
        actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);
        return;
    }

    portENTER_CRITICAL(&s_pressure_lock);
    s_pressure.action = (manual.role == MANUAL_ROLE_INLET) ? PRESSURE_ACTION_INLET : PRESSURE_ACTION_OUTLET;
    s_pressure.dvp_target_code_q = code_q;
    portEXIT_CRITICAL(&s_pressure_lock);
}

static void pressure_control_step_from_adc(uint16_t adc_average, uint32_t adc_samples)
{
    pressure_control_t state = pressure_control_snapshot();
    if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
        /* The closed loop is idle; the manual bench drive (command 0x23) owns
         * the valves here and is serviced by the PID task itself. */
        return;
    }

    const int64_t now_us = esp_timer_get_time();
    const int fault_pin = solenoid_get_fault();
    const bool output_armed = (state.output_on != 0U) ||
                              (state.dvp_target_code_q != 0U) ||
                              (state.dvp_current_code != 0U) ||
                              (state.dvp_output_channel != VALVE_CHANNEL_NONE);
    const bool fault_active = (fault_pin == 0) && output_armed &&
                              (state.valve_mode != PRESSURE_VALVE_MODE_DVP || PRESSURE_DVP_STOP_ON_FAULT_PIN != 0U);
    if (fault_active) {
        if (state.fault_low_count < UINT8_MAX) {
            ++state.fault_low_count;
        }
        if (state.fault_low_count >= PRESSURE_FAULT_DEBOUNCE_COUNT) {
            ESP_LOGW(TAG, "Solenoid driver FAULT asserted during pressure control");
            dvp_drive_clear();
            actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);
            pressure_control_request_stop(PRESSURE_STATE_FAULT, 4);
            return;
        }
    } else {
        state.fault_low_count = 0;
    }

    if (adc_samples != state.adc_samples) {
        state.previous_average = state.adc_average;
        state.adc_average = adc_average;
        state.adc_samples = adc_samples;
    }

    if (state.valve_mode == PRESSURE_VALVE_MODE_DVP) {
        state.state = PRESSURE_STATE_RUNNING;
        const int32_t current_pressure = (int32_t)pressure_normalized_raw(&state, state.adc_average);

        /* fw 1.42 flow-shaping vs. fw 1.41 reference generator (runtime A/B,
         * command 0x2A). Both produce control_target_raw, the setpoint the PID
         * tracks this step. */
        uint16_t control_target_raw;
        if (state.dvp_fs_enable != 0U) {
            /* Flow-shaping mode: aim straight at the target (no reference ramp)
             * and let the distance-scaled current ceiling (computed below) brake
             * the approach. Seed the integrator to the steady leak make-up the
             * instant a setpoint JUMP is seen, so when the bulk slew desaturates
             * near the target the valve is already at the right hold current
             * (bumpless: no post-step sag / undershoot / ring). */
            control_target_raw = state.target_raw;
            state.dvp_ref_pos = (float)state.target_raw;
            state.dvp_slewing = 0U;
            int32_t seed_jump = (int32_t)state.target_raw - (int32_t)state.dvp_fs_seed_target;
            if (seed_jump < 0) {
                seed_jump = -seed_jump;
            }
            if (seed_jump > (int32_t)PRESSURE_DVP_STEP_DETECT_RAW) {
                if ((state.flags & PRESSURE_CONFIG_FLAG_SENSOR_INCREASES) != 0U) {
                    int64_t seed_q16 =
                        dvp_makeup_seed_q16((int32_t)pressure_normalized_raw(&state, state.target_raw));
                    seed_q16 = (seed_q16 * (int64_t)state.dvp_fs_seed_scale) / 128;
                    state.dvp_integral_q16 = seed_q16;
                }
                state.dvp_deriv_filt_q12 = 0;
                state.dvp_fs_seed_target = state.target_raw;
                /* Record the slew direction so the integral freeze (below) only
                 * holds the seed while still APPROACHING; once the pressure
                 * crosses the target the integral is released to correct an
                 * over/under-shoot (the make-up seed/FF model can be stale as the
                 * supply pressure / leak drift). */
                state.dvp_fs_rising =
                    ((int32_t)pressure_normalized_raw(&state, state.target_raw) >= current_pressure) ? 1U : 0U;
            }
        } else {
        /* --- Smart reference generator (fw 1.41) ---------------------------
         * Drive a feasible trajectory toward the commanded target instead of a
         * linear ramp: cruise at vmax, then decelerate (v ~ sqrt(remaining)) so
         * the reference eases into the target with ~zero velocity -> no
         * overshoot. Rates are asymmetric (the plant fills far faster than it
         * vents) and runtime-tunable (command 0x29). The PID below TRACKS this
         * reference, so transients are fast yet overshoot-free while the
         * steady-state hold (ref == target) is unchanged. dvp_ref_pos is in raw
         * (target space) and was seeded to the live pressure at START. */
        float ref_pos = state.dvp_ref_pos;
        const float ref_target = (float)state.target_raw;
        const float p_now = (float)current_pressure;
        /* Engage the ramp only on a real setpoint JUMP; track small/continuous
         * setpoint motion directly (so sines / slider drags are not lagged). */
        if (state.dvp_slewing == 0U && fabsf(ref_target - ref_pos) > (float)PRESSURE_DVP_STEP_DETECT_RAW) {
            state.dvp_slewing = 1U;
        }
        if (state.dvp_slewing != 0U) {
            /* Decelerate and exit on the PRESSURE's real remaining distance, so
             * the trajectory adapts to the (pressure-dependent, asymmetric)
             * plant rate and never overshoots regardless of target. */
            const float arem = fabsf(ref_target - p_now);
            const bool rising = ref_target >= p_now;
            const float vmax = (float)(rising ? state.dvp_vmax_up : state.dvp_vmax_down);
            if (vmax <= 0.0f || arem <= (float)PRESSURE_DVP_STEP_PCAPTURE_RAW) {
                /* shaping off, or pressure has captured -> hand back to direct */
                ref_pos = ref_target;
                state.dvp_slewing = 0U;
                /* Bumpless handoff: seed the integrator with the steady make-up
                 * for the new target so the inlet engages at the right level
                 * immediately -- no post-step sag (up) or undershoot/ring (down).
                 * Assumes the FF model's sensor-increases convention. */
                if ((state.flags & PRESSURE_CONFIG_FLAG_SENSOR_INCREASES) != 0U) {
                    state.dvp_integral_q16 =
                        dvp_makeup_seed_q16((int32_t)pressure_normalized_raw(&state, state.target_raw));
                }
            } else {
                /* desired slew velocity: cruise at vmax, decelerate ~sqrt of the
                 * remaining distance so the approach eases to zero velocity. */
                const float dzone = (float)(rising ? state.dvp_dzone_up : state.dvp_dzone_down)
                                    * (float)PRESSURE_DVP_DZONE_STEP_RAW;
                const float v = (dzone > 1.0f && arem < dzone) ? vmax * sqrtf(arem / dzone) : vmax;
                ref_pos += rising ? v : -v;
                /* Reference governor: keep the shaped ref within the lead band of
                 * the actual pressure so it paces the plant (a large bulk error
                 * still floods the valve, but the ref can't run away). */
                if (ref_pos < p_now - PRESSURE_DVP_REF_LEAD_RAW) {
                    ref_pos = p_now - PRESSURE_DVP_REF_LEAD_RAW;
                } else if (ref_pos > p_now + PRESSURE_DVP_REF_LEAD_RAW) {
                    ref_pos = p_now + PRESSURE_DVP_REF_LEAD_RAW;
                }
                /* never command past the target */
                if ((rising && ref_pos > ref_target) || (!rising && ref_pos < ref_target)) {
                    ref_pos = ref_target;
                }
            }
        } else {
            ref_pos = ref_target;                /* direct tracking */
        }
        if (ref_pos < 0.0f) {
            ref_pos = 0.0f;
        } else if (ref_pos > 65535.0f) {
            ref_pos = 65535.0f;
        }
        state.dvp_ref_pos = ref_pos;
        control_target_raw = (uint16_t)(ref_pos + 0.5f);
        }
        const int32_t target_pressure = (int32_t)pressure_normalized_raw(&state, control_target_raw);

        /* Leak make-up feedforward: the steady inlet drive for this target
         * pressure, so the integral need not wind to it on a step (no windup
         * overshoot) and the setpoint can be tracked directly (low sine lag).
         * Only when the sensor increases with pressure (the FF model assumes it);
         * deliberately a slight under-estimate so the floored integral trims up. */
        int32_t feedforward_raw = 0;
        if (state.dvp_ff_scale != 0U && (state.flags & PRESSURE_CONFIG_FLAG_SENSOR_INCREASES) != 0U) {
            int32_t p_above = target_pressure - PRESSURE_DVP_ZERO_PSI_RAW;
            if (p_above < 0) {
                p_above = 0;
            }
            const int64_t ff = PRESSURE_DVP_FF_BASE_RAW + (PRESSURE_DVP_FF_GAIN_Q8 * (int64_t)p_above) / 256;
            feedforward_raw = (int32_t)((ff * (int64_t)state.dvp_ff_scale) / PRESSURE_DVP_FF_SCALE_UNITY);
        }

        /* Derivative on the filtered measurement, low-pass filtered again so
         * ADC noise does not chew the valve. The divisor is runtime-tunable
         * (command 0x28): deeper = quieter D but more lag. */
        const int32_t previous_pressure = (int32_t)pressure_normalized_raw(&state, state.previous_average);
        const int32_t delta_q12 = (current_pressure - previous_pressure) * 4096;
        uint8_t deriv_lp_div = state.dvp_deriv_lp_div;
        if (deriv_lp_div < 1U) {
            deriv_lp_div = 1U;
        } else if (deriv_lp_div > PRESSURE_DVP_DERIV_LP_DIV_MAX) {
            deriv_lp_div = PRESSURE_DVP_DERIV_LP_DIV_MAX;
        }
        state.dvp_deriv_filt_q12 += (delta_q12 - state.dvp_deriv_filt_q12) / (int32_t)deriv_lp_div;
        /* Filtered pressure rate in raw counts per PID step. */
        const int32_t rate_raw_per_step = state.dvp_deriv_filt_q12 / 4096;

        /* Dead-time compensation: drive the error off the pressure projected
         * dvp_deadtime_steps ahead using the filtered rate, so the loop reacts
         * to where the (delayed) pressure is heading, not where it was. The
         * prediction only affects P and I; D stays on the raw measured rate. */
        int32_t predicted_pressure = current_pressure;
        if (state.dvp_deadtime_steps != 0U) {
            predicted_pressure += rate_raw_per_step * (int32_t)state.dvp_deadtime_steps;
        }
        const int32_t error = target_pressure - predicted_pressure;
        const uint32_t abs_error = (error < 0) ? (uint32_t)(-error) : (uint32_t)error;
        /* True (un-predicted) error, for anti-windup direction and the
         * derivative-fade band. */
        const int32_t meas_error = target_pressure - current_pressure;
        const uint32_t abs_meas_error = (meas_error < 0) ? (uint32_t)(-meas_error) : (uint32_t)meas_error;
        /* Flow-shaping deceleration distance, lag-compensated: project the
         * pressure dvp_fs_lookahead steps ahead (current + rate*lookahead) so the
         * deceleration anticipates the sensor/actuator lag and the real pressure
         * arrives at target with low velocity (no overshoot). Falls back to the
         * measured distance when lookahead is 0. Used by the rise clamp and the
         * current-ceiling taper below; works in both directions. */
        uint32_t fs_abs_dist = abs_meas_error;
        if (state.dvp_fs_enable != 0U && state.dvp_fs_lookahead != 0U) {
            const int32_t fs_pred_err = meas_error - rate_raw_per_step * (int32_t)state.dvp_fs_lookahead;
            fs_abs_dist = (fs_pred_err < 0) ? (uint32_t)(-fs_pred_err) : (uint32_t)fs_pred_err;
        }
        /* Flow-shaping deceleration active: hold the make-up seed (freeze the
         * integral) and cap the drive while bulk-slewing toward the target. It
         * stays active far from target (bulk) or while still moving fast toward
         * it, but RELEASES once the pressure stalls near the target (low
         * velocity) -- so a stale/low make-up seed is trimmed out by the integral
         * (no steady-state error) -- and once the target is crossed (overshoot)
         * so the integral can wind back. Only on the approaching side. */
        const bool fs_decel_active = (state.dvp_fs_enable != 0U) &&
            (fs_abs_dist > (uint32_t)PRESSURE_DVP_FS_CAPTURE_RAW) &&
            ((state.dvp_fs_rising != 0U && meas_error > 0) ||
             (state.dvp_fs_rising == 0U && meas_error < 0)) &&
            (fs_abs_dist > (uint32_t)PRESSURE_DVP_FS_BULK_RAW ||
             rate_raw_per_step > (int32_t)state.dvp_fs_rate_thresh ||
             rate_raw_per_step < -(int32_t)state.dvp_fs_rate_thresh);

        if (current_pressure >= (int32_t)PRESSURE_DVP_FAULT_MAX_RAW) {
            ESP_LOGW(TAG, "DVP pressure overrange stop: pressure_raw=%" PRId32, current_pressure);
            pressure_control_request_stop(PRESSURE_STATE_FAULT, 13);
            return;
        }

        /* Anti-windup: stale-integral reset backstop. (No hard reset at the
         * pressure ceiling: 54000 raw is a legal setpoint and the standing
         * integral there is the leak make-up feed; the positive-output clamp
         * below plus natural negative integration protect the ceiling.)
         * Gated on the MEASURED error, not the dead-time prediction: this
         * backstop must fire on a real, observed pressure reversal -- a
         * predicted error can flip sign on a transient rate and either
         * suppress a genuine reset or spuriously dump the leak-make-up
         * integral. (Identical to the predicted path when deadtime is 0.) */
        uint32_t integral_reset_error_raw = (uint32_t)state.deadband_raw * 4U;
        if (integral_reset_error_raw < PRESSURE_DVP_INTEGRAL_RESET_MIN_RAW) {
            integral_reset_error_raw = PRESSURE_DVP_INTEGRAL_RESET_MIN_RAW;
        }
        if (abs_meas_error >= integral_reset_error_raw &&
            ((meas_error > 0 && state.dvp_integral_q16 < 0) || (meas_error < 0 && state.dvp_integral_q16 > 0))) {
            state.dvp_integral_q16 = 0;
        }

        const uint32_t soft_gain_q8 = dvp_soft_zone_gain_q8(abs_error, state.deadband_raw);

        /* Pressure-adaptive gain schedule (see PRESSURE_DVP_GAIN_REF_RAW). */
        int32_t sched_pressure = current_pressure;
        if (sched_pressure < PRESSURE_DVP_GAIN_SCHED_MIN_RAW) {
            sched_pressure = PRESSURE_DVP_GAIN_SCHED_MIN_RAW;
        }
        int32_t sched_gain_q8 = (int32_t)((PRESSURE_DVP_GAIN_REF_RAW * 256L) / sched_pressure);
        sched_gain_q8 = clamp_i32(sched_gain_q8, PRESSURE_DVP_GAIN_SCHED_MIN_Q8, PRESSURE_DVP_GAIN_SCHED_MAX_Q8);

        /* Derivative (computed from the filtered rate above). Faded toward the
         * setpoint: full during transients for damping, ~0 in the noise-only
         * zone near target so it stops chewing the valve current. */
        int32_t derivative_raw = 0;
        if (state.dvp_kd16 != 0U) {
            derivative_raw = (int32_t)(-((((int64_t)state.dvp_kd16 * (int64_t)state.dvp_deriv_filt_q12) >> PRESSURE_DVP_KD_SHIFT) *
                                        (int64_t)sched_gain_q8 / 256));
            if (state.dvp_d_fade_band_raw != 0U && abs_meas_error < state.dvp_d_fade_band_raw) {
                derivative_raw = (int32_t)(((int64_t)derivative_raw * (int64_t)abs_meas_error) /
                                           (int64_t)state.dvp_d_fade_band_raw);
            }
        }

        const int32_t proportional_raw =
            (int32_t)(((((int64_t)error * (int64_t)state.dvp_kp16) / 256) * (int64_t)soft_gain_q8 / 256) *
                      (int64_t)sched_gain_q8 / 256);

        /* Conditional integration: freeze the integral while the pre-update
         * P+I+D drive already saturates in the error direction, or while the
         * pressure ceiling blocks positive drive. The Q16 accumulator keeps
         * sub-raw integral steps, so small errors still integrate out instead
         * of dying in integer truncation. */
        if (state.dvp_ki16 != 0U) {
            /* P+I only: with D in the test, a braking derivative desaturates
             * the sum and the integrator instantly recharges it back to the
             * rail, cancelling the brake (observed as 1 psi overshoot on a
             * 5->25 step with the output pinned at +1000 past the target). */
            const int64_t presat_raw = (int64_t)proportional_raw + (int64_t)feedforward_raw +
                                       (state.dvp_integral_q16 >> PRESSURE_DVP_INTEGRAL_FRAC_BITS);
            const bool saturated_same_direction =
                (error > 0 && presat_raw >= PRESSURE_DVP_CONTROL_RAW_SCALE) ||
                (error < 0 && presat_raw <= -PRESSURE_DVP_CONTROL_RAW_SCALE);
            const bool ceiling_blocks = current_pressure >= (int32_t)PRESSURE_DVP_TARGET_MAX_RAW && error > 0;
            /* Asymmetric-outlet anti-windup: in the coast band the negative
             * drive cannot actuate (the leak does the work), so freeze a
             * further-negative integral there -- otherwise it winds down
             * against the vent deadzone and causes a slow undershoot on the
             * recovery. Positive integration in the band is still allowed so
             * the inlet re-engages once the leak overshoots target. */
            const int64_t vent_threshold_raw =
                ((int64_t)state.dvp_vent_threshold_pm * PRESSURE_DVP_CONTROL_RAW_SCALE) / 1000;
            const int64_t out_deadband_raw =
                ((int64_t)state.dvp_output_deadband * PRESSURE_DVP_CONTROL_RAW_SCALE) / 1000;
            const bool coast_windup = error < 0 &&
                                      presat_raw <= out_deadband_raw &&
                                      presat_raw >= -vent_threshold_raw;
            /* Flow-shaping slew freeze: hold the integral at the make-up seed
             * while the deceleration is active (bulk-slewing toward target), so
             * it does not wind past the make-up during the fast approach
             * (-> overshoot). Releases when stalled near target or after crossing
             * it (see fs_decel_active) so a stale seed self-corrects with no
             * steady-state error. */
            if (!saturated_same_direction && !ceiling_blocks && !coast_windup && !fs_decel_active) {
                const int64_t integral_step_q16 =
                    (((int64_t)error * (int64_t)state.dvp_ki16 * (int64_t)soft_gain_q8) / 256) *
                    (int64_t)sched_gain_q8 / 256;
                int64_t integral_q16 = state.dvp_integral_q16 + integral_step_q16;
                /* fw 1.44: the floor is runtime-selectable (command 0x2D) --
                 * legacy 0 for the pure leak-down plant, symmetric for plumbing
                 * whose outlet valve does the venting. */
                const int64_t integral_floor_q16 =
                    (s_dvp_int_floor_mode == PRESSURE_DVP_INT_FLOOR_MODE_SYMMETRIC)
                        ? -PRESSURE_DVP_INTEGRAL_LIMIT_Q16
                        : PRESSURE_DVP_INTEGRAL_FLOOR_Q16;
                if (integral_q16 > PRESSURE_DVP_INTEGRAL_LIMIT_Q16) {
                    integral_q16 = PRESSURE_DVP_INTEGRAL_LIMIT_Q16;
                } else if (integral_q16 < integral_floor_q16) {
                    integral_q16 = integral_floor_q16;
                }
                state.dvp_integral_q16 = integral_q16;
            }
        }

        const int64_t drive_raw = (int64_t)proportional_raw + (int64_t)feedforward_raw +
                                  (state.dvp_integral_q16 >> PRESSURE_DVP_INTEGRAL_FRAC_BITS) +
                                  derivative_raw;
        int32_t desired_permille = (int32_t)((drive_raw * 1000L) / PRESSURE_DVP_CONTROL_RAW_SCALE);

        /* Debug telemetry mirrors of the controller internals. */
        state.dvp_integral_permille = (int16_t)clamp_i32(
            (int32_t)(((state.dvp_integral_q16 >> PRESSURE_DVP_INTEGRAL_FRAC_BITS) * 1000L) / PRESSURE_DVP_CONTROL_RAW_SCALE),
            -32000, 32000);
        state.dvp_derivative_permille = (int16_t)clamp_i32(
            (int32_t)(((int64_t)derivative_raw * 1000L) / PRESSURE_DVP_CONTROL_RAW_SCALE), -32000, 32000);
        state.dvp_sched_gain_q8_div2 = (uint8_t)(sched_gain_q8 / 2);
        if (current_pressure >= (int32_t)PRESSURE_DVP_TARGET_MAX_RAW && desired_permille > 0) {
            desired_permille = 0;
        }

        /* fw 1.42 flow-shaping rise deceleration: as the pressure nears the
         * target, fade the commanded drive from full down to the make-up seed,
         * so the inlet lands at the steady hold current with ~zero velocity =
         * no overshoot. This is the user's "reduce the current near the target
         * to control the rate of change". Only on the RISE (the up-brake is
         * weak -- only the plant leak removes excess pressure) and only while
         * slewing; the make-up seed is the frozen integral. Combined with the
         * inlet ceiling taper below, the make-up permille maps to the make-up
         * code at arrival. */
        if (fs_decel_active && error > 0 &&
            (state.flags & PRESSURE_CONFIG_FLAG_SENSOR_INCREASES) != 0U) {
            const int32_t seed_permille =
                (int32_t)(((state.dvp_integral_q16 >> PRESSURE_DVP_INTEGRAL_FRAC_BITS) * 1000L) /
                          PRESSURE_DVP_CONTROL_RAW_SCALE);
            const uint32_t zone = (uint32_t)state.dvp_fs_decel_up * PRESSURE_DVP_DZONE_STEP_RAW;
            if (zone > 0U && seed_permille >= 0 && seed_permille < 1000) {
                const uint32_t d = fs_abs_dist - (uint32_t)PRESSURE_DVP_FS_CAPTURE_RAW;
                const uint32_t num = (d >= zone) ? zone : d;
                const int32_t cap = seed_permille +
                                    (int32_t)(((uint32_t)(1000 - seed_permille) * num) / zone);
                if (desired_permille > cap) {
                    desired_permille = cap;
                }
            }
        }

        /* fw 1.42 flow shaping: distance-scaled current ceiling. Far from the
         * target the channel gets full current (fast slew); within the decel
         * zone the ceiling tapers linearly back to the steady hold range so the
         * flow rate (dP/dt) eases to ~0 on arrival. abs_meas_error is the real
         * remaining distance to target in raw counts (972.5 raw = 1 psi). When
         * flow shaping is off these stay at the configured maxima. */
        uint8_t eff_inlet_max = state.dvp_max_code;
        uint8_t eff_outlet_max = state.dvp_outlet_max_code;
        if (state.dvp_fs_enable != 0U) {
            if (state.dvp_fs_inlet_far_max > state.dvp_max_code) {
                const uint32_t zone = (uint32_t)state.dvp_fs_decel_up * PRESSURE_DVP_DZONE_STEP_RAW;
                const uint32_t span = (uint32_t)(state.dvp_fs_inlet_far_max - state.dvp_max_code);
                if (zone == 0U || fs_abs_dist >= zone) {
                    eff_inlet_max = state.dvp_fs_inlet_far_max;
                } else {
                    eff_inlet_max = (uint8_t)(state.dvp_max_code + (span * fs_abs_dist) / zone);
                }
            }
            if (state.dvp_outlet_max_code > state.dvp_outlet_open_code) {
                const uint32_t zone = (uint32_t)state.dvp_fs_decel_down * PRESSURE_DVP_DZONE_STEP_RAW;
                const uint32_t span = (uint32_t)(state.dvp_outlet_max_code - state.dvp_outlet_open_code);
                if (zone == 0U || fs_abs_dist >= zone) {
                    eff_outlet_max = state.dvp_outlet_max_code;
                } else {
                    eff_outlet_max = (uint8_t)(state.dvp_outlet_open_code + (span * fs_abs_dist) / zone);
                }
            }
        }

        dvp_prepare_output(&state, desired_permille, eff_inlet_max, eff_outlet_max);
        state.error_code = 0;
        state.previous_target_raw = control_target_raw;
        if (state.output_on != 0U) {
            ++state.pulse_count;
        }
        if (!pressure_control_store_task_state(&state)) {
            /* A stop/reconfig raced this step: kill the drive we may have
             * just republished, or the valve stays energized while IDLE. */
            dvp_drive_clear();
            actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);
        }
        return;
    }

    /* ---- Bang-bang (on/off valve) mode ---- */

    if (state.output_on != 0U && now_us >= state.pulse_end_us) {
        actuator_publish(ACTUATOR_REQUEST_CHANNEL_OFF, state.output_channel);
        state.output_on = 0;
        state.output_channel = VALVE_CHANNEL_NONE;
        state.state = PRESSURE_STATE_SETTLING;
        state.action = PRESSURE_ACTION_SETTLING;
        state.settle_until_us = now_us + ((int64_t)state.settle_ms * 1000);
        (void)pressure_control_store_task_state(&state);
        return;
    }

    if (state.output_on != 0U) {
        (void)pressure_control_store_task_state(&state);
        return;
    }

    if (state.state == PRESSURE_STATE_SETTLING && now_us < state.settle_until_us) {
        (void)pressure_control_store_task_state(&state);
        return;
    }

    state.state = PRESSURE_STATE_RUNNING;
    const int32_t current_pressure = (int32_t)pressure_normalized_raw(&state, state.adc_average);
    const int32_t previous_pressure = (int32_t)pressure_normalized_raw(&state, state.previous_average);
    const int32_t target_pressure = (int32_t)pressure_normalized_raw(&state, state.target_raw);
    const int32_t error = target_pressure - current_pressure;
    const uint32_t abs_error = (error < 0) ? (uint32_t)(-error) : (uint32_t)error;

    if (abs_error <= state.deadband_raw) {
        state.action = PRESSURE_ACTION_DEADBAND;
        state.last_pulse_ms = 0;
        state.error_code = 0;
        (void)pressure_control_store_task_state(&state);
        return;
    }

    /* Momentum check: if pressure is already moving toward the target and
     * will arrive within a step, settle instead of pulsing (no overshoot). */
    const int32_t pressure_delta = current_pressure - previous_pressure;
    if ((error > 0 && pressure_delta > 0 && error <= (pressure_delta + (int32_t)state.deadband_raw)) ||
        (error < 0 && pressure_delta < 0 && (-error) <= ((-pressure_delta) + (int32_t)state.deadband_raw))) {
        state.action = PRESSURE_ACTION_SETTLING;
        state.last_pulse_ms = 0;
        state.settle_until_us = now_us + ((int64_t)state.settle_ms * 1000);
        state.state = PRESSURE_STATE_SETTLING;
        (void)pressure_control_store_task_state(&state);
        return;
    }

    const pressure_action_t next_action = (error > 0) ? PRESSURE_ACTION_INLET : PRESSURE_ACTION_OUTLET;
    const uint8_t channel = (next_action == PRESSURE_ACTION_INLET) ? VALVE_CHANNEL_INLET : VALVE_CHANNEL_OUTLET;
    const uint8_t pulse_ms = pressure_compute_pulse_ms(&state, abs_error, next_action);

    actuator_publish(ACTUATOR_REQUEST_PULSE_ON, channel);

    state.action = next_action;
    state.last_direction = (uint8_t)next_action;
    state.output_channel = channel;
    state.output_on = 1;
    state.last_pulse_ms = pulse_ms;
    state.pulse_end_us = now_us + ((int64_t)pulse_ms * 1000);
    state.settle_until_us = 0;
    state.error_code = 0;
    ++state.pulse_count;
    if (!pressure_control_store_task_state(&state)) {
        dvp_drive_clear();
        actuator_publish(ACTUATOR_REQUEST_ALL_OFF, VALVE_CHANNEL_NONE);
    }
}

/* =====================================================================
 * Valve actuation (core 1): requests + DVP current dithering
 * ===================================================================== */

static void pressure_actuator_mark_fault(uint8_t error_code, const char *operation, esp_err_t err)
{
    ESP_LOGW(TAG, "%s failed: %s", operation, esp_err_to_name(err));
    dvp_drive_clear();
    (void)solenoid_set_all_channels_off();

    pressure_control_t state = pressure_control_snapshot();
    state.state = PRESSURE_STATE_FAULT;
    state.action = PRESSURE_ACTION_NONE;
    state.output_channel = VALVE_CHANNEL_NONE;
    state.output_on = 0;
    state.last_pulse_ms = 0;
    state.error_code = error_code;
    state.dvp_current_code = 0;
    state.dvp_output_channel = VALVE_CHANNEL_NONE;
    state.dvp_target_code_q = 0;
    state.dvp_applied_code_q = 0;
    state.dvp_output_permille = 0;
    state.pulse_end_us = 0;
    state.settle_until_us = 0;
    pressure_control_store_state(&state);
}

static uint64_t dvp_abs_i64(int64_t value)
{
    return value < 0 ? (uint64_t)(-value) : (uint64_t)value;
}

static uint8_t dvp_clamp_code_candidate(int64_t candidate, uint8_t min_code, uint8_t max_code)
{
    if (candidate < (int64_t)min_code) {
        return min_code;
    }
    if (candidate > (int64_t)max_code) {
        return max_code;
    }
    return (uint8_t)candidate;
}

static void dvp_window_clear(dvp_actuator_state_t *dvp_state)
{
    memset(dvp_state->target_history_q, 0, sizeof(dvp_state->target_history_q));
    memset(dvp_state->actual_history_code, 0, sizeof(dvp_state->actual_history_code));
    dvp_state->history_count = 0;
    dvp_state->history_head = 0;
}

static void dvp_actuator_state_clear(dvp_actuator_state_t *dvp_state)
{
    dvp_state->active = 0;
    dvp_state->channel = VALVE_CHANNEL_NONE;
    dvp_state->target_code_q = 0;
    dvp_state->min_code = PRESSURE_DVP_DEFAULT_OPEN_CODE;
    dvp_state->max_code = PRESSURE_DVP_DEFAULT_MAX_CODE;
    dvp_state->last_code = 0xFFU;
    if (dvp_state->window_n < 1U || dvp_state->window_n > PRESSURE_DVP_DITHER_HISTORY_MAX) {
        dvp_state->window_n = PRESSURE_DVP_DITHER_HISTORY_N;
    }
    dvp_window_clear(dvp_state);
}

static void dvp_drive_bounds(const dvp_drive_t *drive, uint8_t *min_code, uint8_t *max_code)
{
    *min_code = drive->min_code;
    *max_code = drive->max_code;
    if (*max_code > PRESSURE_DVP_HOST_MAX_CODE) {
        *max_code = PRESSURE_DVP_HOST_MAX_CODE;
    }
    if (*min_code > *max_code) {
        *min_code = *max_code;
    }
}

static uint16_t dvp_clamp_target_code_q(uint16_t target_code_q, uint8_t min_code, uint8_t max_code)
{
    const uint16_t min_code_q = (uint16_t)min_code << PRESSURE_DVP_CODE_FRAC_BITS;
    const uint16_t max_code_q = (uint16_t)max_code << PRESSURE_DVP_CODE_FRAC_BITS;
    if (target_code_q < min_code_q) {
        return min_code_q;
    }
    if (target_code_q > max_code_q) {
        return max_code_q;
    }
    return target_code_q;
}

static void dvp_history_push(dvp_actuator_state_t *dvp_state, uint16_t target_code_q, uint8_t actual_code)
{
    const uint8_t window_n = dvp_state->window_n;
    dvp_state->target_history_q[dvp_state->history_head] = target_code_q;
    dvp_state->actual_history_code[dvp_state->history_head] = actual_code;
    if (dvp_state->history_count < window_n) {
        ++dvp_state->history_count;
    }
    dvp_state->history_head = (uint8_t)((dvp_state->history_head + 1U) % window_n);
}

static uint8_t dvp_step_distance(uint8_t left, uint8_t right)
{
    return left > right ? (uint8_t)(left - right) : (uint8_t)(right - left);
}

static bool dvp_window_candidate_better(const dvp_actuator_state_t *dvp_state,
                                        uint8_t candidate,
                                        uint64_t candidate_abs_error,
                                        uint8_t best_code,
                                        uint64_t best_abs_error)
{
    if (candidate_abs_error < best_abs_error) {
        return true;
    }
    if (candidate_abs_error > best_abs_error) {
        return false;
    }
    if (dvp_state->last_code == 0xFFU) {
        return candidate < best_code;
    }
    return dvp_step_distance(candidate, dvp_state->last_code) < dvp_step_distance(best_code, dvp_state->last_code);
}

/* Moving-window dither: pick the integer code whose window sum best matches
 * the Q9 target sum, so the average current approximates the fractional
 * target between updates. */
static uint8_t dvp_select_window_code(const dvp_actuator_state_t *dvp_state)
{
    uint32_t target_sum_q = dvp_state->target_code_q;
    uint32_t actual_sum_code = 0;
    for (uint8_t index = 0; index < dvp_state->history_count; ++index) {
        target_sum_q += dvp_state->target_history_q[index];
        actual_sum_code += dvp_state->actual_history_code[index];
    }

    const int64_t ideal_numerator_q = (int64_t)target_sum_q - ((int64_t)actual_sum_code << PRESSURE_DVP_CODE_FRAC_BITS);
    int64_t lower_candidate = ideal_numerator_q / PRESSURE_DVP_CODE_Q_SCALE;
    if (ideal_numerator_q < 0 && (ideal_numerator_q % PRESSURE_DVP_CODE_Q_SCALE) != 0) {
        --lower_candidate;
    }
    const int64_t upper_candidate = lower_candidate + 1;

    uint8_t best_code = dvp_clamp_code_candidate(lower_candidate, dvp_state->min_code, dvp_state->max_code);
    int64_t best_error_q = (((int64_t)actual_sum_code + best_code) << PRESSURE_DVP_CODE_FRAC_BITS) - (int64_t)target_sum_q;
    uint64_t best_abs_error = dvp_abs_i64(best_error_q);

    const uint8_t upper_code = dvp_clamp_code_candidate(upper_candidate, dvp_state->min_code, dvp_state->max_code);
    const int64_t upper_error_q = (((int64_t)actual_sum_code + upper_code) << PRESSURE_DVP_CODE_FRAC_BITS) - (int64_t)target_sum_q;
    const uint64_t upper_abs_error = dvp_abs_i64(upper_error_q);
    if (dvp_window_candidate_better(dvp_state, upper_code, upper_abs_error, best_code, best_abs_error)) {
        best_code = upper_code;
    }

    return best_code;
}

static void dvp_account_skipped_updates(dvp_actuator_state_t *dvp_state, uint32_t skipped_updates)
{
    if (dvp_state->last_code == 0xFFU) {
        return;
    }
    if (skipped_updates > dvp_state->window_n) {
        skipped_updates = dvp_state->window_n;
    }
    for (uint32_t index = 0; index < skipped_updates; ++index) {
        dvp_history_push(dvp_state, dvp_state->target_code_q, dvp_state->last_code);
    }
}

static esp_err_t dvp_apply_current_update(dvp_actuator_state_t *dvp_state,
                                          uint8_t *applied_channel,
                                          uint32_t elapsed_updates,
                                          uint32_t late_updates)
{
    const int64_t start_us = esp_timer_get_time();
    esp_err_t err = ESP_OK;
    dvp_drive_t drive = dvp_drive_snapshot();

    if (drive.active == 0U || (drive.channel != VALVE_CHANNEL_INLET && drive.channel != VALVE_CHANNEL_OUTLET)) {
        if (dvp_state->active != 0U || *applied_channel != VALVE_CHANNEL_NONE) {
            err = solenoid_set_all_channels_off();
            if (err == ESP_OK) {
                *applied_channel = VALVE_CHANNEL_NONE;
                pressure_control_note_dvp_off();
            }
        }
        dvp_actuator_state_clear(dvp_state);
        return err;
    }

    uint8_t min_code = 0;
    uint8_t max_code = 0;
    dvp_drive_bounds(&drive, &min_code, &max_code);
    const uint16_t target_code_q = dvp_clamp_target_code_q(drive.code_q, min_code, max_code);
    const uint8_t window_n = clamp_u8(drive.dither_n == 0U ? PRESSURE_DVP_DITHER_HISTORY_N : drive.dither_n,
                                      1U, PRESSURE_DVP_DITHER_HISTORY_MAX);
    dvp_state->code_jump_max = drive.code_jump_max;

    if (drive.generation != dvp_state->generation) {
        const bool same_window = dvp_state->active != 0U &&
                                 dvp_state->channel == drive.channel &&
                                 dvp_state->min_code == min_code &&
                                 dvp_state->max_code == max_code &&
                                 dvp_state->window_n == window_n;
        dvp_state->generation = drive.generation;
        dvp_state->active = 1;
        dvp_state->channel = drive.channel;
        dvp_state->target_code_q = target_code_q;
        dvp_state->min_code = min_code;
        dvp_state->max_code = max_code;
        dvp_state->window_n = window_n;
        if (!same_window) {
            dvp_window_clear(dvp_state);
            dvp_state->last_code = 0xFFU;
        }
    }

    const bool fine = solenoid_supports_fine_current();

    uint8_t code;     /* integer code: telemetry / last_code / dither history */
    uint16_t code_q9; /* fractional code actually applied to the driver */
    if (fine) {
        /* Full-resolution backend (TLE92464): apply the fractional Q9 target
         * straight to the chip's 15-bit setpoint -- the moving-window dither
         * is unnecessary, so it is bypassed entirely. The integer `code` is
         * telemetry-only here (the chip gets the full Q9), so round it to the
         * nearest code rather than flooring, for an honest GUI readout. */
        code_q9 = target_code_q;
        code = (uint8_t)((target_code_q + (1U << (PRESSURE_DVP_CODE_FRAC_BITS - 1U))) >> PRESSURE_DVP_CODE_FRAC_BITS);
        if (code > PRESSURE_DVP_HOST_MAX_CODE) {
            code = PRESSURE_DVP_HOST_MAX_CODE;
            code_q9 = (uint16_t)((uint16_t)PRESSURE_DVP_HOST_MAX_CODE << PRESSURE_DVP_CODE_FRAC_BITS);
        }
    } else {
        /* Coarse backend (MAX22200, 7-bit code): moving-window dither recovers
         * sub-LSB resolution, with the optional per-update code-jump limit. */
        const uint32_t skipped_updates = elapsed_updates > 1U ? elapsed_updates - 1U : 0U;
        dvp_account_skipped_updates(dvp_state, skipped_updates);

        code = dvp_select_window_code(dvp_state);
        if (code > PRESSURE_DVP_HOST_MAX_CODE) {
            code = PRESSURE_DVP_HOST_MAX_CODE;
        }
        /* Dither range limit (command 0x28): cap the per-update integer code
         * jump so the commanded valve current cannot leap, even if the
         * dither/target asks for it. The applied (limited) code is what gets
         * pushed into the window, so the dither catches up over later updates. */
        if (dvp_state->code_jump_max != 0U && dvp_state->last_code != 0xFFU) {
            const int32_t delta = (int32_t)code - (int32_t)dvp_state->last_code;
            if (delta > (int32_t)dvp_state->code_jump_max) {
                code = (uint8_t)((int32_t)dvp_state->last_code + (int32_t)dvp_state->code_jump_max);
            } else if (delta < -(int32_t)dvp_state->code_jump_max) {
                code = (uint8_t)((int32_t)dvp_state->last_code - (int32_t)dvp_state->code_jump_max);
            }
            code = dvp_clamp_code_candidate((int64_t)code, dvp_state->min_code, dvp_state->max_code);
        }
        code_q9 = (uint16_t)((uint16_t)code << PRESSURE_DVP_CODE_FRAC_BITS);
    }

    const uint8_t other_channel = (dvp_state->channel == VALVE_CHANNEL_INLET) ? VALVE_CHANNEL_OUTLET : VALVE_CHANNEL_INLET;
    if (*applied_channel != dvp_state->channel) {
        if (*applied_channel != VALVE_CHANNEL_NONE) {
            err = solenoid_set_all_channels_off();
        }
        if (err == ESP_OK) {
            err = solenoid_set_channel_state(other_channel, false);
        }
        if (err == ESP_OK) {
            err = solenoid_set_channel_current_code_q9(dvp_state->channel, code_q9);
        }
        if (err == ESP_OK) {
            err = solenoid_set_channel_state(dvp_state->channel, true);
        }
        if (err == ESP_OK) {
            *applied_channel = dvp_state->channel;
            dvp_state->last_code = code;
        }
    } else {
        err = solenoid_set_channel_current_code_q9(dvp_state->channel, code_q9);
        if (err == ESP_OK) {
            dvp_state->last_code = code;
        }
    }

    if (err == ESP_OK) {
        if (!fine) {
            dvp_history_push(dvp_state, dvp_state->target_code_q, code);
        }
        pressure_control_note_dvp_update(dvp_state->channel, code, (uint32_t)(esp_timer_get_time() - start_us), late_updates);
    }
    return err;
}

/*
 * Core 1: valve controller. Owns the MAX22200 on the shared MAX/LTC SPI3
 * bus. Applies the actuator requests published by the PID (core 0) and, in
 * proportional mode, runs the moving-window current update at
 * MAX_CURRENT_UPDATE_HZ.
 */
static void valve_task(void *arg)
{
    (void)arg;
    uint32_t observed_update_ticks = s_max_update_ticks;
    dvp_actuator_state_t dvp_state = {
        .channel = VALVE_CHANNEL_NONE,
        .last_code = 0xFFU,
    };
    uint32_t applied_sequence = 0;
    uint8_t applied_channel = VALVE_CHANNEL_NONE;
    int64_t last_rearm_us = 0;

    while (true) {
        uint32_t notify_value = 0;
        (void)xTaskNotifyWait(0, UINT32_MAX, &notify_value, pdMS_TO_TICKS(VALVE_SERVICE_PERIOD_MS));

        /* Self-heal: if the valve supply (VBAT) came up after boot, the one-shot
         * arm in solenoid_init() failed and the TLE outputs are still disabled.
         * Retry ~1 Hz while idle so the board becomes drivable on its own
         * (no reflash) the moment VBAT is in range. */
        if (applied_channel == VALVE_CHANNEL_NONE && !solenoid_is_armed()) {
            const int64_t now = esp_timer_get_time();
            if (now - last_rearm_us > 1000000) {
                last_rearm_us = now;
                (void)solenoid_arm();
            }
        }

        actuator_request_t request = actuator_snapshot();
        bool handled_request = false;
        if (request.sequence != applied_sequence && request.type != ACTUATOR_REQUEST_NONE) {
            applied_sequence = request.sequence;
            handled_request = true;
        }

        uint32_t current_update_ticks = s_max_update_ticks;
        uint32_t pending_updates = current_update_ticks - observed_update_ticks;

        if (!handled_request && pending_updates == 0U) {
            continue;
        }

        esp_err_t err = ESP_OK;
        if (handled_request) {
            switch (request.type) {
            case ACTUATOR_REQUEST_ALL_OFF:
                dvp_actuator_state_clear(&dvp_state);
                err = solenoid_set_all_channels_off();
                if (err == ESP_OK) {
                    applied_channel = VALVE_CHANNEL_NONE;
                    pressure_control_note_dvp_off();
                }
                break;

            case ACTUATOR_REQUEST_PULSE_ON: {
                pressure_control_t state = pressure_control_snapshot();
                if (state.state != PRESSURE_STATE_RUNNING && state.state != PRESSURE_STATE_SETTLING) {
                    break;
                }
                if (request.channel != VALVE_CHANNEL_INLET && request.channel != VALVE_CHANNEL_OUTLET) {
                    err = ESP_ERR_INVALID_ARG;
                    break;
                }
                const uint8_t other_channel = (request.channel == VALVE_CHANNEL_INLET) ? VALVE_CHANNEL_OUTLET : VALVE_CHANNEL_INLET;
                err = solenoid_set_channel_state(other_channel, false);
                if (err == ESP_OK) {
                    err = solenoid_set_channel_state(request.channel, true);
                }
                if (err == ESP_OK) {
                    applied_channel = request.channel;
                }
                break;
            }

            case ACTUATOR_REQUEST_CHANNEL_OFF:
                if (request.channel == VALVE_CHANNEL_INLET || request.channel == VALVE_CHANNEL_OUTLET) {
                    err = solenoid_set_channel_state(request.channel, false);
                    if (err == ESP_OK && applied_channel == request.channel) {
                        applied_channel = VALVE_CHANNEL_NONE;
                    }
                }
                break;

            case ACTUATOR_REQUEST_NONE:
            default:
                break;
            }
        }

        if (err != ESP_OK) {
            uint8_t error_code = 9;
            const char *operation = "Pressure actuator update";
            if (request.type == ACTUATOR_REQUEST_CHANNEL_OFF) {
                error_code = 6;
                operation = "Pressure valve off";
            } else if (request.type == ACTUATOR_REQUEST_PULSE_ON) {
                error_code = 7;
                operation = "Pressure valve pulse";
            } else if (request.type == ACTUATOR_REQUEST_ALL_OFF) {
                error_code = 4;
                operation = "Pressure all off";
            }
            pressure_actuator_mark_fault(error_code, operation, err);
            applied_channel = VALVE_CHANNEL_NONE;
            continue;
        }

        current_update_ticks = s_max_update_ticks;
        pending_updates = current_update_ticks - observed_update_ticks;
        if (pending_updates != 0U) {
            /* Current updates belong to the proportional mode only. In
             * bang-bang mode the channel is owned by the pulse requests
             * above and must not be touched here (the inactive-drive path
             * inside dvp_apply_current_update would cut a pulse short) --
             * unless a manual constant-current drive (command 0x23) is armed,
             * which also runs through the proportional path. */
            if (pressure_control_snapshot().valve_mode == PRESSURE_VALVE_MODE_DVP ||
                manual_drive_is_active()) {
                const uint32_t late_updates = pending_updates > 1U ? pending_updates - 1U : 0U;
                err = dvp_apply_current_update(&dvp_state, &applied_channel, pending_updates, late_updates);
                if (err != ESP_OK) {
                    pressure_actuator_mark_fault(9, "Pressure DVP current update", err);
                    applied_channel = VALVE_CHANNEL_NONE;
                }
            }
            observed_update_ticks = current_update_ticks;
        }
    }
}

/* =====================================================================
 * Timer, buses, startup
 * ===================================================================== */

static bool control_timer_callback(gptimer_handle_t timer,
                                   const gptimer_alarm_event_data_t *edata,
                                   void *user_ctx)
{
    (void)timer;
    (void)edata;
    (void)user_ctx;

    static uint32_t tick;
    ++tick;
    BaseType_t high_task_wakeup = pdFALSE;

    if ((tick % ADC_TICK_DIVIDER) == 0U && s_adc_task_handle != NULL) {
        vTaskNotifyGiveFromISR(s_adc_task_handle, &high_task_wakeup);
    }

    if ((tick % PRESSURE_PID_TICK_DIVIDER) == 0U && s_pid_task_handle != NULL) {
        vTaskNotifyGiveFromISR(s_pid_task_handle, &high_task_wakeup);
    }

    if ((tick % MAX_UPDATE_TICK_DIVIDER) == 0U) {
        ++s_max_update_ticks;
        if (s_valve_task_handle != NULL) {
            (void)xTaskNotifyFromISR(s_valve_task_handle, MAX_CURRENT_UPDATE_NOTIFY_BIT, eSetBits, &high_task_wakeup);
        }
    }

    return high_task_wakeup == pdTRUE;
}

static esp_err_t control_timer_start(void)
{
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = CONTROL_TIMER_RESOLUTION_HZ,
    };
    ESP_RETURN_ON_ERROR(gptimer_new_timer(&timer_config, &s_control_timer), TAG, "create control timer");

    gptimer_event_callbacks_t callbacks = {
        .on_alarm = control_timer_callback,
    };
    ESP_RETURN_ON_ERROR(gptimer_register_event_callbacks(s_control_timer, &callbacks, NULL), TAG, "register control timer");

    gptimer_alarm_config_t alarm_config = {
        .alarm_count = CONTROL_BASE_TICK_PERIOD_TICKS,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    ESP_RETURN_ON_ERROR(gptimer_set_alarm_action(s_control_timer, &alarm_config), TAG, "configure control timer");
    ESP_RETURN_ON_ERROR(gptimer_enable(s_control_timer), TAG, "enable control timer");
    return gptimer_start(s_control_timer);
}

static esp_err_t configure_spi(spi_device_handle_t *can_spi, spi_device_handle_t *max_spi,
                               spi_device_handle_t *ltc_spi, spi_device_handle_t *tle_spi)
{
    /* Bus 1: MCP25625 CAN controller, alone on its host. */
    spi_bus_config_t can_bus_config = {
        .mosi_io_num = PIN_SPI_MOSI,
        .miso_io_num = PIN_SPI_MISO,
        .sclk_io_num = PIN_SPI_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 16,
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(CAN_SPI_HOST, &can_bus_config, SPI_DMA_DISABLED), TAG, "init CAN SPI bus");

    spi_device_interface_config_t can_config = {
        .clock_speed_hz = MCP25625_SPI_CLOCK_HZ,
        .mode = 0,
        .spics_io_num = PIN_SPI_CS_CAN,
        .queue_size = 1,
    };
    ESP_RETURN_ON_ERROR(spi_bus_add_device(CAN_SPI_HOST, &can_config, can_spi), TAG, "add MCP25625 SPI device");

    /* Bus 2: MAX22200 + LTC1864 share this host. The MAX22200 uses a hardware
     * chip-select; the LTC1864 has no CS (its CONV strobe is a plain GPIO). */
    spi_bus_config_t maxltc_bus_config = {
        .mosi_io_num = PIN_MAXLTC_MOSI,
        .miso_io_num = PIN_MAXLTC_MISO,
        .sclk_io_num = PIN_MAXLTC_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 16,
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(MAXLTC_SPI_HOST, &maxltc_bus_config, SPI_DMA_DISABLED), TAG, "init MAX/LTC SPI bus");

#if !VEMA_BOARD_TLE_ALL_IN_ONE
    spi_device_interface_config_t max_config = {
        .clock_speed_hz = MAX22200_SPI_CLOCK_HZ,
        .mode = 0,
        .spics_io_num = PIN_SPI_CS_MOS,
        .queue_size = 1,
    };
    ESP_RETURN_ON_ERROR(spi_bus_add_device(MAXLTC_SPI_HOST, &max_config, max_spi), TAG, "add MAX22200 SPI device");
#else
    /* No MAX22200 on the TLE_ALL_IN_ONE board -- and its old CS pin (GPIO9)
     * IS the TLE's chip-select, so registering a second device on the same
     * pad would double-claim the CS GPIO. */
    *max_spi = NULL;
#endif

    spi_device_interface_config_t ltc_config = {
        .clock_speed_hz = LTC1864_SPI_CLOCK_HZ,
        .mode = 0,
        .spics_io_num = -1,
        .queue_size = 1,
    };
    ESP_RETURN_ON_ERROR(spi_bus_add_device(MAXLTC_SPI_HOST, &ltc_config, ltc_spi), TAG, "add LTC1864 SPI device");

    /* TLE92464 extension breakout: third device on the shared SPI3 bus, its
     * own hardware chip-select (GPIO11) and SPI mode 1 (CPOL=0, CPHA=1) -- the
     * MAX22200/LTC1864 use mode 0, but each device keeps its own mode and the
     * shared_spi_bus lock serializes the three so they never overlap. */
    spi_device_interface_config_t tle_config = {
        .clock_speed_hz = TLE92464_SPI_CLOCK_HZ,
        .mode = 1,
        .spics_io_num = PIN_SPI_CS_TLE,
        .queue_size = 1,
        /* The on-board TLE (ALL_IN_ONE) rejects frames whose CS hugs the clock
         * burst: with the IDF default of 0 the chip answers every frame with
         * its framing-error word (0xE581xxxx), while generous CS margins (as
         * in the bit-banged probe) work. 2 SPI cycles = 4 us @ 500 kHz. */
        .cs_ena_pretrans = 2,
        .cs_ena_posttrans = 2,
    };
    ESP_RETURN_ON_ERROR(spi_bus_add_device(MAXLTC_SPI_HOST, &tle_config, tle_spi), TAG, "add TLE92464 SPI device");
    return ESP_OK;
}

static esp_err_t configure_mcp_interrupt(void)
{
    gpio_config_t int_config = {
        .pin_bit_mask = 1ULL << PIN_MCP25625_INT,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&int_config), TAG, "configure MCP interrupt GPIO");

    esp_err_t err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        return err;
    }
    return gpio_isr_handler_add(PIN_MCP25625_INT, mcp_int_isr, NULL);
}

static void log_boot_info(void)
{
    ESP_LOGI(TAG, "VEMA V2 pressure controller fw %s (reset=%d)", VEMA_FIRMWARE_VERSION, esp_reset_reason());
    /* Which slot is running is the only way to tell a successful OTA from one
     * that silently fell back, since a rebuild of the same source reports the
     * same version string. */
    const esp_partition_t *running = esp_ota_get_running_partition();
    const esp_app_desc_t *desc = esp_app_get_description();
    ESP_LOGI(TAG, "Running from partition '%s' @0x%06" PRIx32 ", app version '%s' built %s %s",
             running != NULL ? running->label : "?",
             running != NULL ? running->address : 0,
             desc != NULL ? desc->version : "?",
             desc != NULL ? desc->date : "?", desc != NULL ? desc->time : "?");
    ESP_LOGI(TAG, "Board variant: %s; valve channels: inlet=ch%u outlet=ch%u; DVP code ceiling=%u DC + %u dither = %u (~%u mA)",
             VEMA_BOARD_TLE_ALL_IN_ONE ? "TLE_ALL_IN_ONE" : "V2 PCB + TLE breakout",
             VALVE_CHANNEL_INLET, VALVE_CHANNEL_OUTLET, PRESSURE_DVP_HOST_MAX_CODE,
             PRESSURE_DVP_DITHER_PEAK_MAX_CODE, PRESSURE_DVP_SAFE_MAX_CODE,
             (unsigned)(((PRESSURE_DVP_SAFE_MAX_CODE * 200U) + 63U) / 127U));
    ESP_LOGI(TAG, "V2 PCB: two-SPI split (CAN alone on SPI2; MAX22200+LTC1864 on SPI3)");
    ESP_LOGI(TAG, "CAN bus (SPI2): MOSI=%d SCK=%d MISO=%d CS_CAN=%d INT=%d", PIN_SPI_MOSI, PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_CS_CAN, PIN_MCP25625_INT);
    ESP_LOGI(TAG, "MAX+LTC bus (SPI3): MOSI=%d SCK=%d MISO=%d CS_MOS=%d LTC_CONV=%d", PIN_MAXLTC_MOSI, PIN_MAXLTC_SCK, PIN_MAXLTC_MISO, PIN_SPI_CS_MOS, PIN_ADC_CONV);
    ESP_LOGI(TAG, "MAX22200 control: CMD=%d FAULT=%d (write-only, no SDO readback)", PIN_MAX22200_CMD, PIN_MAX22200_FAULT);
    ESP_LOGI(TAG, "TLE92464 extension: CS=%d EN=%d FAULT=%d RESET=%d (NC=-1); active solenoid driver: %s",
             PIN_SPI_CS_TLE, PIN_TLE92464_EN, PIN_TLE92464_FAULT, PIN_TLE92464_RESET,
             VEMA_SOLENOID_DRIVER == VEMA_SOLENOID_DRIVER_TLE92464 ? "TLE92464" : "MAX22200");
#if VEMA_TLE_VERIFY
    ESP_LOGW(TAG, "VEMA_TLE_VERIFY build: running TLE/ADC/CAN comm self-test only (no controller)");
#endif
    ESP_LOGI(TAG, "CAN IDs: base=0x%03x legacy_ctrl=0x%03x ota_data=0x%03x status=0x%03x extended_cmd=0x%03x broadcast=0x%03x",
             s_can_device_base, s_can_id_host_command, s_can_id_host_ota_data, s_can_id_device_status,
             s_can_id_extended_command, CAN_ID_BROADCAST);
    const uint16_t table_id = tle_can_legacy_runtime_table_id(s_can_device_base);
    if (table_id != 0U) {
        ESP_LOGI(TAG, "Shared-bus slot %u of 24, runtime target table 0x%03x, sync 0x%03x @150 Hz",
                 (unsigned)(s_can_device_base - TLE_LEGACY_RUNTIME_FIRST_ID), table_id, CAN_ID_BROADCAST);
    } else {
        ESP_LOGW(TAG, "base 0x%03x is outside the actuator range 0x%03x..0x%03x: no runtime table slot, "
                      "broadcast target updates will not reach this board",
                 s_can_device_base, TLE_LEGACY_RUNTIME_FIRST_ID, TLE_LEGACY_RUNTIME_LAST_ID);
    }
    ESP_LOGI(TAG, "Rates: base=%uHz adc=%uHz pid=%uHz max_update=%uHz; default mode: %s",
             CONTROL_BASE_TICK_HZ, ADC_SAMPLE_HZ, PRESSURE_PID_HZ, MAX_CURRENT_UPDATE_HZ,
             PRESSURE_VALVE_MODE_DEFAULT == PRESSURE_VALVE_MODE_DVP ? "proportional (DVP)" : "bang-bang");
}

void app_main(void)
{
    ESP_ERROR_CHECK(can_id_nvs_init());
    load_device_id_from_nvs();
    can_ota_config_t ota_config = {
        .status_can_id = s_can_id_device_status,
        .send_frame = can_ota_send_frame,
        .safe_shutdown = can_ota_safe_shutdown,
        .flush_tx = legacy_flush_tx,
    };
    can_ota_init(&ota_config);

    const tle_can_legacy_hooks_t legacy_hooks = {
        .get_filtered_raw = legacy_get_filtered_raw,
        .set_target_raw = legacy_set_target_raw,
        .set_enabled = legacy_set_enabled,
        .is_running = legacy_is_running,
        .send_frame = can_ota_send_frame,
        .flush_tx = legacy_flush_tx,
        .save_base_id = legacy_save_base_id,
    };
    tle_can_legacy_init(&legacy_hooks, s_can_device_base);

    log_boot_info();

    /* Create the MAX/LTC bus lock before any task that uses the bus starts
     * (the MAX22200 is reached from both cores: core 1 for control, core 0
     * for command-driven reconfiguration). */
    shared_spi_bus_init();

    spi_device_handle_t can_spi = NULL;
    spi_device_handle_t max_spi = NULL;
    spi_device_handle_t ltc_spi = NULL;
    spi_device_handle_t tle_spi = NULL;
    ESP_ERROR_CHECK(configure_spi(&can_spi, &max_spi, &ltc_spi, &tle_spi));

    /* LTC1864 first: its init parks CONV high, which gates the LTC SDO buffer
     * off the shared MISO before any MAX22200/TLE92464 traffic. */
    ltc1864_config_t adc_config = {
        .spi = ltc_spi,
        .conv_io = PIN_ADC_CONV,
    };
    ESP_ERROR_CHECK(ltc1864_init(&adc_config));

#if VEMA_TLE_VERIFY
    /* Unpowered communication self-test: exercise the TLE92464, re-check the
     * on-board ADC and CAN controller, print a report and loop. The pressure
     * controller is not started in this build. */
    tle_verify_run(tle_spi, can_spi);
    return; /* tle_verify_run never returns */
#endif

    /* Solenoid driver bring-up (selected backend, default TLE92464). Non-fatal
     * on a comms failure: control still starts so the fault is observable over
     * telemetry rather than wedging boot. */
    const spi_device_handle_t solenoid_spi =
        (VEMA_SOLENOID_DRIVER == VEMA_SOLENOID_DRIVER_MAX22200) ? max_spi : tle_spi;
    ESP_ERROR_CHECK_WITHOUT_ABORT(solenoid_init(VEMA_SOLENOID_DRIVER, solenoid_spi));
    ESP_ERROR_CHECK_WITHOUT_ABORT(pressure_control_configure_channels(&s_pressure));
    ESP_LOGI(TAG, "%s configured: inlet=ch%u outlet=ch%u fault_pin=%d",
             solenoid_name(), VALVE_CHANNEL_INLET, VALVE_CHANNEL_OUTLET, solenoid_get_fault());

    esp_err_t mcp_err = mcp2515_init(&s_mcp, can_spi);
    if (mcp_err == ESP_OK) {
        mcp_err = mcp2515_configure_bitrate_16mhz(&s_mcp, CAN_BUS_BITRATE);
    }
    if (mcp_err == ESP_OK) {
        /* Six exact filters for the six IDs this board answers to. The two
         * that arrive every cycle go to RXB0 with rollover disabled, so a
         * runtime-table frame can never be sitting in RXB1 when the sync frame
         * lands -- the failure that used to overflow RX1 on high-ID boards
         * during 24-board broadcast operation. */
        const uint16_t table_id = tle_can_legacy_runtime_table_id(s_can_device_base);
        const mcp2515_filter_set_t filters = {
            .rxb0 = {s_can_device_base, table_id != 0U ? table_id : MCP2515_FILTER_UNUSED},
            .rxb1 = {CAN_ID_BROADCAST, s_can_id_extended_command, s_can_id_host_command, s_can_id_host_ota_data},
        };
        mcp_err = mcp2515_configure_filter_set(&s_mcp, &filters);
    }
    if (mcp_err == ESP_OK) {
        mcp_err = mcp2515_set_normal_mode(&s_mcp);
    }
    if (mcp_err == ESP_OK) {
        ESP_ERROR_CHECK(configure_mcp_interrupt());
        s_mcp_ready = true;
        ESP_LOGI(TAG, "MCP25625 ready: %s, standard CAN frames",
                 CAN_BUS_BITRATE == MCP2515_BITRATE_1MBPS ? "1 Mbit/s" : "500 kbit/s");
    } else {
        s_mcp_ready = false;
        ESP_LOGE(TAG, "MCP25625 init failed: %s. Control continues, CAN unavailable.", esp_err_to_name(mcp_err));
    }

    /* Core 0: CAN (interrupt-driven, highest) + PID (lower priority).
     * Core 1: ADC sampling + valve actuation -- the SPI3 transaction core. */
    xTaskCreatePinnedToCore(can_task, "can", 4096, NULL, 7, &s_can_task_handle, 0);
    xTaskCreatePinnedToCore(pressure_pid_task, "pid", 4096, NULL, 6, &s_pid_task_handle, 0);
    xTaskCreatePinnedToCore(adc_sample_task, "adc", 4096, NULL, 6, &s_adc_task_handle, 1);
    xTaskCreatePinnedToCore(valve_task, "valve", 4096, NULL, 5, &s_valve_task_handle, 1);
    /* Raw-capture console dump (command 0x10): the lowest-priority control task
     * (below even the USB parser), so a multi-second ASCII dump can never
     * starve CAN/USB servicing, the PID loop, the ADC or the valve updates. */
    xTaskCreatePinnedToCore(adc_dump_task, "rawdump", 4096, NULL, 2, &s_adc_dump_task_handle, 0);

    /* USB control link (one-to-one alternative to the CAN bus; same frames,
     * same handler). Failure is non-fatal: CAN operation is independent. */
    esp_err_t usb_err = usb_link_init(usb_link_wake_can_task);
    if (usb_err == ESP_OK) {
        ESP_LOGI(TAG, "USB control link ready: '>%03x:<hex>' in, '<%03x:<hex>' out",
                 s_can_id_host_command, s_can_id_device_status);
    } else {
        ESP_LOGE(TAG, "USB control link init failed: %s. CAN control unaffected.", esp_err_to_name(usb_err));
    }

    ESP_ERROR_CHECK(control_timer_start());

    /* One-shot self-check for bench bring-up over USB serial. */
    vTaskDelay(pdMS_TO_TICKS(1000));
    adc_status_t adc = adc_status_snapshot();
    portENTER_CRITICAL(&s_adc_pid_lock);
    const uint32_t adc_errors = s_adc_pid_buffer.read_errors;
    portEXIT_CRITICAL(&s_adc_pid_lock);
    ESP_LOGI(TAG, "Self-check @1s: adc_samples=%" PRIu32 " raw=%u filtered=%u read_errors=%" PRIu32 " mcp_ready=%d solenoid=%s fault_pin=%d",
             adc.samples, adc.raw, adc.average, adc_errors, (int)s_mcp_ready, solenoid_name(), solenoid_get_fault());
}
