#include "ltc1864.h"

#include <stdbool.h>

#include "esp_check.h"
#include "esp_rom_sys.h"
#include "shared_spi_bus.h"

#define LTC1864_TAG "ltc1864"

/* CONV high time to start and complete a conversion. The LTC1864 conversion
 * time is ~3.2 us max; 5 us matches the original bit-banged timing and leaves
 * margin. */
#define LTC1864_CONV_PULSE_US 5U
/* Settle delay after CONV falls before clocking the result out. */
#define LTC1864_CONV_SETTLE_US 1U

static ltc1864_config_t s_config;
static bool s_initialized;

esp_err_t ltc1864_init(const ltc1864_config_t *config)
{
    if (config == NULL || config->spi == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    s_config = *config;

    gpio_config_t conv_config = {
        .pin_bit_mask = 1ULL << s_config.conv_io,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&conv_config), LTC1864_TAG, "configure CONV pin");

    /* Idle CONV HIGH: the LTC1864 three-states SDO only while CONV is high, so
     * idling high keeps SDO off the shared MISO whenever the MAX22200 is being
     * accessed. SDO is only driven during this driver's own read window. */
    gpio_set_level(s_config.conv_io, 1);
    s_initialized = true;
    return ESP_OK;
}

esp_err_t ltc1864_read_raw(uint16_t *raw_value)
{
    if (!s_initialized || raw_value == NULL || s_config.spi == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    /* Serialize against the MAX22200 on the shared SPI3 bus so a MAX22200
     * command/data sequence is never split by an LTC1864 conversion. */
    esp_err_t err = shared_spi_bus_acquire(SHARED_SPI_DEVICE_LTC1864);
    if (err != ESP_OK) {
        return err;
    }

    /* CONV idles HIGH (SDO high-Z). Drop CONV (SDO active), pulse CONV high to
     * run a fresh conversion (SDO high-Z during the convert), drop CONV to
     * present the result, clock out 16 bits, then return CONV high to start the
     * next conversion and release SDO from the shared MISO. */
    gpio_set_level(s_config.conv_io, 0);
    esp_rom_delay_us(LTC1864_CONV_SETTLE_US);
    gpio_set_level(s_config.conv_io, 1);
    esp_rom_delay_us(LTC1864_CONV_PULSE_US);
    gpio_set_level(s_config.conv_io, 0);
    esp_rom_delay_us(LTC1864_CONV_SETTLE_US);

    uint8_t tx_data[2] = {0, 0};
    uint8_t rx_data[2] = {0, 0};
    spi_transaction_t transaction = {
        .length = 16,
        .tx_buffer = tx_data,
        .rx_buffer = rx_data,
    };
    err = spi_device_transmit(s_config.spi, &transaction);

    /* Always return CONV high (SDO high-Z) before releasing the bus, even on
     * error, so the LTC1864 never holds the shared MISO during MAX22200 work. */
    gpio_set_level(s_config.conv_io, 1);

    const esp_err_t release_err = shared_spi_bus_release(SHARED_SPI_DEVICE_LTC1864);
    if (err != ESP_OK) {
        return err;
    }
    if (release_err != ESP_OK) {
        return release_err;
    }

    *raw_value = (uint16_t)(((uint16_t)rx_data[0] << 8) | rx_data[1]);
    return ESP_OK;
}
