#include "shared_spi_bus.h"

#include <stdbool.h>
#include <stdint.h>

#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#define SHARED_SPI_DEVICE_SWITCH_GUARD_US 5U
#define SHARED_SPI_MAX_COMMAND_DATA_GUARD_US 20U
#define SHARED_SPI_DEVICE_NONE 0xFFU

static StaticSemaphore_t s_shared_spi_lock_storage;
static SemaphoreHandle_t s_shared_spi_lock;
static uint8_t s_shared_spi_active_device = SHARED_SPI_DEVICE_NONE;
static uint8_t s_shared_spi_last_device = SHARED_SPI_DEVICE_NONE;
static int64_t s_shared_spi_last_release_us;

static bool shared_spi_device_valid(shared_spi_device_t device)
{
    return device == SHARED_SPI_DEVICE_MAX22200 || device == SHARED_SPI_DEVICE_LTC1864 ||
           device == SHARED_SPI_DEVICE_TLE92464;
}

static SemaphoreHandle_t shared_spi_lock_handle(void)
{
    if (s_shared_spi_lock == NULL) {
        s_shared_spi_lock = xSemaphoreCreateMutexStatic(&s_shared_spi_lock_storage);
    }
    return s_shared_spi_lock;
}

void shared_spi_bus_init(void)
{
    (void)shared_spi_lock_handle();
}

static void shared_spi_wait_for_device_switch_gap(shared_spi_device_t device)
{
    if (s_shared_spi_last_release_us == 0 || s_shared_spi_last_device == SHARED_SPI_DEVICE_NONE || s_shared_spi_last_device == (uint8_t)device) {
        return;
    }

    const int64_t elapsed_us = esp_timer_get_time() - s_shared_spi_last_release_us;
    if (elapsed_us >= 0 && elapsed_us < (int64_t)SHARED_SPI_DEVICE_SWITCH_GUARD_US) {
        esp_rom_delay_us((uint32_t)((int64_t)SHARED_SPI_DEVICE_SWITCH_GUARD_US - elapsed_us));
    }
}

esp_err_t shared_spi_bus_acquire(shared_spi_device_t device)
{
    if (!shared_spi_device_valid(device)) {
        return ESP_ERR_INVALID_ARG;
    }

    SemaphoreHandle_t lock = shared_spi_lock_handle();
    if (lock == NULL) {
        return ESP_ERR_NO_MEM;
    }
    if (xSemaphoreTake(lock, portMAX_DELAY) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    if (s_shared_spi_active_device != SHARED_SPI_DEVICE_NONE) {
        xSemaphoreGive(lock);
        return ESP_ERR_INVALID_STATE;
    }

    shared_spi_wait_for_device_switch_gap(device);
    s_shared_spi_active_device = (uint8_t)device;
    return ESP_OK;
}

esp_err_t shared_spi_bus_release(shared_spi_device_t device)
{
    if (!shared_spi_device_valid(device)) {
        return ESP_ERR_INVALID_ARG;
    }

    SemaphoreHandle_t lock = shared_spi_lock_handle();
    if (lock == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    if (s_shared_spi_active_device != (uint8_t)device) {
        s_shared_spi_active_device = SHARED_SPI_DEVICE_NONE;
        xSemaphoreGive(lock);
        return ESP_ERR_INVALID_STATE;
    }

    s_shared_spi_last_device = (uint8_t)device;
    s_shared_spi_last_release_us = esp_timer_get_time();
    s_shared_spi_active_device = SHARED_SPI_DEVICE_NONE;
    xSemaphoreGive(lock);
    return ESP_OK;
}

esp_err_t shared_spi_bus_transmit(shared_spi_device_t device, spi_device_handle_t spi, spi_transaction_t *transaction)
{
    if (spi == NULL || transaction == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t err = shared_spi_bus_acquire(device);
    if (err != ESP_OK) {
        return err;
    }

    err = spi_device_transmit(spi, transaction);
    const esp_err_t release_err = shared_spi_bus_release(device);
    return err == ESP_OK ? release_err : err;
}

void shared_spi_bus_delay_between_max_command_and_data(void)
{
    esp_rom_delay_us(SHARED_SPI_MAX_COMMAND_DATA_GUARD_US);
}
