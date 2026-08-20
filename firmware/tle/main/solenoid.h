#ifndef SOLENOID_H_
#define SOLENOID_H_

#include <stdbool.h>
#include <stdint.h>

#include "driver/spi_master.h"
#include "esp_err.h"

/*
 * Solenoid-driver abstraction for the V2 PCB.
 *
 * The pressure controller talks to the valve hardware exclusively through this
 * thin layer so the physical driver chip can be swapped without touching the
 * control architecture. Two backends are provided and selected at init time:
 *
 *   - MAX22200 : the original write-only driver (7-bit current code, dithered
 *                in software to recover sub-LSB resolution).
 *   - TLE92464 : the extension-board driver (15-bit current setpoint; the
 *                abstract 0..127 code is mapped onto it at full resolution, so
 *                no software dither is needed -- solenoid_supports_fine_current
 *                returns true and the controller applies the Q9 code directly).
 *
 * The abstract "current code" (0..127, optionally Q9-fractional) and the
 * channel numbering are common to both backends, so the PID gains, open/max
 * codes and telemetry are unchanged across the swap.
 */

#define VEMA_SOLENOID_DRIVER_MAX22200 0
#define VEMA_SOLENOID_DRIVER_TLE92464 1

/* Abstract current code range (kept from the MAX22200 7-bit scale). */
#define SOLENOID_CURRENT_CODE_MAX 0x7FU
/* Fractional bits carried in the Q9 code (matches PRESSURE_DVP_CODE_FRAC_BITS). */
#define SOLENOID_CODE_FRAC_BITS 9U

typedef enum {
    SOLENOID_DRIVE_PROPORTIONAL_CURRENT = 0, /* proportional valve, current regulation */
    SOLENOID_DRIVE_ONOFF_VOLTAGE = 1,        /* on/off valve, full drive */
} solenoid_drive_t;

/* Initialise the selected backend. `spi` must be the SPI device handle for
 * that backend (max_spi for MAX22200, tle_spi for TLE92464). Returns ESP_OK
 * when the driver is usable for control (for the TLE this includes the
 * comms-alive-but-no-VBAT case; check the boot log for Mission Mode status). */
esp_err_t solenoid_init(uint8_t driver, spi_device_handle_t spi);

/* Human-readable name of the active backend (for logs/telemetry). */
const char *solenoid_name(void);

/* True when the backend honours the fractional Q9 code directly (TLE92464),
 * so the controller can skip the moving-window dither. */
bool solenoid_supports_fine_current(void);

/* 1 = no fault (FAULT pin high or not wired), 0 = fault asserted, -1 = error. */
int solenoid_get_fault(void);

/* True when the active backend's outputs are armed/ready to energise. For the
 * TLE92464 this means Mission Mode has been reached (requires VBAT in range);
 * the MAX22200 has no such gate and always reports armed. */
bool solenoid_is_armed(void);

/* (Re)attempt to arm the outputs. For the TLE92464 this re-runs the
 * Config->Mission transition (clearing latched supply diagnostics first), which
 * is how the controller recovers when VBAT came up after boot -- the one-shot
 * arm in solenoid_init() runs too early if the valve supply ramps slowly. A
 * no-op returning ESP_OK when already armed or on the MAX22200. */
esp_err_t solenoid_arm(void);

/* Configure a valve channel's drive mode (and, for bang-bang, its currents). */
esp_err_t solenoid_configure_valve_channel(uint8_t channel, uint8_t hit_current,
                                           uint8_t hold_current, uint8_t hit_time,
                                           solenoid_drive_t drive);

/* Set the channel current from the abstract Q9 code (full resolution on TLE;
 * truncated to the 7-bit code on MAX22200). */
esp_err_t solenoid_set_channel_current_code_q9(uint8_t channel, uint16_t code_q9);

/* Program hardware current dither on a channel (TLE92464 only -- a no-op
 * returning ESP_OK on the MAX22200, which has no hardware dither). step_size = 0
 * disables the overlay. See tle92464.h for the field semantics. */
esp_err_t solenoid_set_channel_dither(uint8_t channel, uint8_t steps, uint8_t flat,
                                      uint16_t step_size, uint16_t mant, uint8_t exp,
                                      bool deep);

/* True when the active backend has a hardware current-dither generator. */
bool solenoid_supports_hw_dither(void);

/* Enable/disable a channel's output. */
esp_err_t solenoid_set_channel_state(uint8_t channel, bool enabled);

/* Disable every channel. */
esp_err_t solenoid_set_all_channels_off(void);

#endif
