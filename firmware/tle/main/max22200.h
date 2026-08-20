#ifndef MAX22200_H_
#define MAX22200_H_

#include <stdbool.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

/*
 * Write-only MAX22200 driver for the V2 PCB.
 *
 * The V2 board has no pull-up on the MAX22200 SDO line, so register readback
 * does not work. The chip does not need SDO to operate: every register the
 * firmware touches (STATUS 0x00, CFG_CHn 0x02+2n) is kept in a driver-side
 * shadow and written in full, never read-modify-written. Chip health is
 * monitored through the active-low FAULT pin only. One read TRANSACTION is
 * still issued (result discarded) after activation: the STATUS fault flags
 * are clear-on-read, and that read releases the power-on UVM fault latch so
 * the FAULT pin can deassert.
 */

#define MAX22200_CHANNEL_COUNT 8U
#define MAX22200_CURRENT_CODE_MAX 0x7FU

typedef enum {
    MAX22200_VALVE_DRIVE_LOW_SIDE_CURRENT = 0,  /* proportional valve, current regulation, half full-scale */
    MAX22200_VALVE_DRIVE_HIGH_SIDE_VOLTAGE = 1, /* on/off valve, voltage drive, full scale */
} max22200_valve_drive_t;

typedef struct {
    gpio_num_t cmd_io;
    gpio_num_t fault_io;
    spi_device_handle_t spi;
    SemaphoreHandle_t lock;
    uint32_t status_shadow;
    uint32_t channel_config_shadow[MAX22200_CHANNEL_COUNT];
    bool channel_config_shadow_valid[MAX22200_CHANNEL_COUNT];
    bool active;
} max22200_t;

esp_err_t max22200_init(max22200_t *device, spi_device_handle_t spi, gpio_num_t cmd_io, gpio_num_t fault_io);
/* 1 = FAULT pin high (no fault), 0 = fault asserted, -1 = bad device. */
int max22200_get_fault(const max22200_t *device);
esp_err_t max22200_configure_valve_channel(max22200_t *device, uint8_t channel, uint8_t hit_current, uint8_t hold_current, uint8_t hit_time, max22200_valve_drive_t drive);
/* Constant-current update: HOLD = HIT = code, HIT time 0. Channel must have
 * been configured first (the full-register shadow must be valid). */
esp_err_t max22200_set_channel_current_code(max22200_t *device, uint8_t channel, uint8_t current_code);
esp_err_t max22200_set_channel_state(max22200_t *device, uint8_t channel, bool enabled);
esp_err_t max22200_set_all_channels_off(max22200_t *device);

#endif
