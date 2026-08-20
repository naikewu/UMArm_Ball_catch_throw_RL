#include "max22200.h"

#include "esp_check.h"
#include "esp_rom_sys.h"
#include "shared_spi_bus.h"

#define MAX_TAG "max22200"

#define MAX22200_STATUS_REG 0x00U
#define MAX22200_CFG_CH(channel) (0x02U + (2U * (uint8_t)(channel)))

#define MAX22200_WRITE_BIT 0x80U
#define MAX22200_ADDR_MASK 0x7FU
#define MAX22200_ACTIVE_MASK (1UL << 0)
#define MAX22200_STATUS_ONCH_MASK 0xFF000000UL
#define MAX22200_ONCH_MASK(channel) (1UL << (24U + (uint8_t)(channel)))
#define MAX22200_STATUS_FREQ_MASK (1UL << 16)
#define MAX22200_CH_MODE_MASK(pair) (0x3UL << (8U + (2U * (uint8_t)(pair))))

#define MAX22200_HFS_MASK (1UL << 31)
#define MAX22200_HOLD_MASK (0x7FUL << 24)
#define MAX22200_TRGNSP_IO_MASK (1UL << 23)
#define MAX22200_HIT_MASK (0x7FUL << 16)
#define MAX22200_HIT_T_MASK (0xFFUL << 8)
#define MAX22200_VDRNCDR_MASK (1UL << 7)
#define MAX22200_HSNLS_MASK (1UL << 6)
#define MAX22200_FREQ_CFG_MASK (0x3UL << 4)

#define MAX22200_CHOP_FREQ_80KHZ 1U
#define MAX22200_VOLTAGE_DRIVE 1U
#define MAX22200_CURRENT_DRIVE 0U
#define MAX22200_HIGH_SIDE 1U
#define MAX22200_LOW_SIDE 0U
#define MAX22200_FREQMAIN_DIV_2 2U
#define MAX22200_FULL_SCALE 0U
#define MAX22200_HALF_FULL_SCALE 1U

/* Power-up settle time after writing the ACTIVE bit (datasheet: >= 1 ms). */
#define MAX22200_ACTIVATE_DELAY_US 2500U

static uint8_t clamp_current_code(uint8_t value)
{
    return value > MAX22200_CURRENT_CODE_MAX ? MAX22200_CURRENT_CODE_MAX : value;
}

static uint8_t field_shift(uint32_t mask)
{
    uint8_t shift = 0;
    while (((mask >> shift) & 1UL) == 0UL && shift < 31U) {
        ++shift;
    }
    return shift;
}

static uint32_t field_prep(uint32_t mask, uint32_t value)
{
    return (value << field_shift(mask)) & mask;
}

static esp_err_t max22200_lock(max22200_t *device)
{
    if (device == NULL || device->spi == NULL || device->lock == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (xSemaphoreTake(device->lock, portMAX_DELAY) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

static void max22200_unlock(max22200_t *device)
{
    xSemaphoreGive(device->lock);
}

static esp_err_t max22200_transfer(max22200_t *device, const uint8_t *tx_data, uint8_t length)
{
    spi_transaction_t transaction = {
        .length = (uint16_t)length * 8U,
        .tx_buffer = tx_data,
    };
    return spi_device_transmit(device->spi, &transaction);
}

/*
 * One register access = CMD-high 8-bit command frame, CMD-low 32-bit data
 * frame, each under its own hardware-CS assertion, with a guard delay in
 * between. Byte order on the wire is kept exactly as validated on V1
 * hardware (data sent LSByte first).
 */
static esp_err_t max22200_reg_frames_unlocked(max22200_t *device, uint8_t command, const uint8_t data[4])
{
    esp_err_t err = shared_spi_bus_acquire(SHARED_SPI_DEVICE_MAX22200);
    if (err != ESP_OK) {
        return err;
    }

    err = gpio_set_level(device->cmd_io, 1);
    if (err == ESP_OK) {
        err = max22200_transfer(device, &command, 1);
    }
    const esp_err_t cmd_low_err = gpio_set_level(device->cmd_io, 0);
    if (err == ESP_OK) {
        err = cmd_low_err;
    }
    if (err == ESP_OK) {
        shared_spi_bus_delay_between_max_command_and_data();
        err = max22200_transfer(device, data, 4);
    }

    const esp_err_t release_err = shared_spi_bus_release(SHARED_SPI_DEVICE_MAX22200);
    return err == ESP_OK ? release_err : err;
}

static esp_err_t max22200_reg_write_unlocked(max22200_t *device, uint8_t reg, uint32_t value)
{
    const uint8_t tx_data[4] = {
        (uint8_t)(value & 0xFFU),
        (uint8_t)((value >> 8) & 0xFFU),
        (uint8_t)((value >> 16) & 0xFFU),
        (uint8_t)((value >> 24) & 0xFFU),
    };
    return max22200_reg_frames_unlocked(device, MAX22200_WRITE_BIT | (reg & MAX22200_ADDR_MASK), tx_data);
}

/* Issue a read transaction and discard the result. The fault flags in STATUS
 * (UVM is latched at power-on, asserting FAULT low) are clear-on-READ; the
 * chip-side clear happens even though the returned SDO data is unusable on
 * the V2 board, so this is how the FAULT pin gets released after boot. */
static esp_err_t max22200_reg_read_discard_unlocked(max22200_t *device, uint8_t reg)
{
    const uint8_t dummy[4] = {0, 0, 0, 0};
    return max22200_reg_frames_unlocked(device, reg & MAX22200_ADDR_MASK, dummy);
}

static esp_err_t max22200_status_update_unlocked(max22200_t *device, uint32_t mask, uint32_t value)
{
    uint32_t next_value = device->status_shadow;
    next_value &= ~mask;
    next_value |= value & mask;
    ESP_RETURN_ON_ERROR(max22200_reg_write_unlocked(device, MAX22200_STATUS_REG, next_value), MAX_TAG, "write status shadow");
    device->status_shadow = next_value;
    return ESP_OK;
}

static esp_err_t max22200_activate_unlocked(max22200_t *device)
{
    if (device->active) {
        return ESP_OK;
    }

    /* All channel pairs independent, all channels off, ACTIVE set. */
    const uint32_t status_value = MAX22200_ACTIVE_MASK;
    ESP_RETURN_ON_ERROR(max22200_reg_write_unlocked(device, MAX22200_STATUS_REG, status_value), MAX_TAG, "activate MAX22200");
    device->status_shadow = status_value;
    esp_rom_delay_us(MAX22200_ACTIVATE_DELAY_US);
    /* Dummy STATUS read: clears the power-on UVM fault latch so the FAULT
     * pin can deassert (datasheet init sequence). Result is discarded. */
    (void)max22200_reg_read_discard_unlocked(device, MAX22200_STATUS_REG);
    device->active = true;
    return ESP_OK;
}

esp_err_t max22200_init(max22200_t *device, spi_device_handle_t spi, gpio_num_t cmd_io, gpio_num_t fault_io)
{
    if (device == NULL || spi == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    device->spi = spi;
    device->cmd_io = cmd_io;
    device->fault_io = fault_io;
    device->status_shadow = 0;
    device->active = false;
    for (uint8_t channel = 0; channel < MAX22200_CHANNEL_COUNT; ++channel) {
        device->channel_config_shadow[channel] = 0;
        device->channel_config_shadow_valid[channel] = false;
    }
    device->lock = xSemaphoreCreateMutex();
    if (device->lock == NULL) {
        return ESP_ERR_NO_MEM;
    }

    gpio_config_t cmd_config = {
        .pin_bit_mask = 1ULL << cmd_io,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&cmd_config), MAX_TAG, "configure CMD");
    gpio_set_level(cmd_io, 0);

    gpio_config_t fault_config = {
        .pin_bit_mask = 1ULL << fault_io,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    return gpio_config(&fault_config);
}

int max22200_get_fault(const max22200_t *device)
{
    if (device == NULL) {
        return -1;
    }
    return gpio_get_level(device->fault_io);
}

esp_err_t max22200_configure_valve_channel(max22200_t *device,
                                           uint8_t channel,
                                           uint8_t hit_current,
                                           uint8_t hold_current,
                                           uint8_t hit_time,
                                           max22200_valve_drive_t drive)
{
    if (channel >= MAX22200_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }
    if (drive != MAX22200_VALVE_DRIVE_LOW_SIDE_CURRENT && drive != MAX22200_VALVE_DRIVE_HIGH_SIDE_VOLTAGE) {
        return ESP_ERR_INVALID_ARG;
    }

    hit_current = clamp_current_code(hit_current);
    hold_current = clamp_current_code(hold_current);

    const bool low_side_current = drive == MAX22200_VALVE_DRIVE_LOW_SIDE_CURRENT;

    ESP_RETURN_ON_ERROR(max22200_lock(device), MAX_TAG, "lock configure valve");
    esp_err_t err = max22200_activate_unlocked(device);
    if (err == ESP_OK) {
        const uint8_t pair = channel / 2U;
        err = max22200_status_update_unlocked(device,
                                              MAX22200_CH_MODE_MASK(pair) | MAX22200_STATUS_FREQ_MASK,
                                              field_prep(MAX22200_STATUS_FREQ_MASK, MAX22200_CHOP_FREQ_80KHZ));
    }

    if (err == ESP_OK) {
        /* Full register image; the fault-detection enables (low nibble) stay 0,
         * matching the chip reset state. */
        const uint32_t value = field_prep(MAX22200_HFS_MASK, low_side_current ? MAX22200_HALF_FULL_SCALE : MAX22200_FULL_SCALE) |
                               field_prep(MAX22200_HOLD_MASK, hold_current) |
                               field_prep(MAX22200_TRGNSP_IO_MASK, 0U) |
                               field_prep(MAX22200_HIT_MASK, hit_current) |
                               field_prep(MAX22200_HIT_T_MASK, hit_time) |
                               field_prep(MAX22200_VDRNCDR_MASK, low_side_current ? MAX22200_CURRENT_DRIVE : MAX22200_VOLTAGE_DRIVE) |
                               field_prep(MAX22200_HSNLS_MASK, low_side_current ? MAX22200_LOW_SIDE : MAX22200_HIGH_SIDE) |
                               field_prep(MAX22200_FREQ_CFG_MASK, MAX22200_FREQMAIN_DIV_2);
        err = max22200_reg_write_unlocked(device, MAX22200_CFG_CH(channel), value);
        if (err == ESP_OK) {
            device->channel_config_shadow[channel] = value;
            device->channel_config_shadow_valid[channel] = true;
        }
    }
    max22200_unlock(device);
    return err;
}

esp_err_t max22200_set_channel_current_code(max22200_t *device, uint8_t channel, uint8_t current_code)
{
    if (channel >= MAX22200_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }

    current_code = clamp_current_code(current_code);

    ESP_RETURN_ON_ERROR(max22200_lock(device), MAX_TAG, "lock current update");
    esp_err_t err = ESP_OK;
    if (!device->channel_config_shadow_valid[channel]) {
        err = ESP_ERR_INVALID_STATE; /* channel must be configured first */
    }
    if (err == ESP_OK) {
        const uint32_t mask = MAX22200_HOLD_MASK | MAX22200_HIT_MASK | MAX22200_HIT_T_MASK;
        uint32_t next_value = device->channel_config_shadow[channel];
        next_value &= ~mask;
        next_value |= field_prep(MAX22200_HOLD_MASK, current_code) |
                      field_prep(MAX22200_HIT_MASK, current_code) |
                      field_prep(MAX22200_HIT_T_MASK, 0U);
        err = max22200_reg_write_unlocked(device, MAX22200_CFG_CH(channel), next_value);
        if (err == ESP_OK) {
            device->channel_config_shadow[channel] = next_value;
        }
    }
    max22200_unlock(device);
    return err;
}

esp_err_t max22200_set_channel_state(max22200_t *device, uint8_t channel, bool enabled)
{
    if (channel >= MAX22200_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }

    ESP_RETURN_ON_ERROR(max22200_lock(device), MAX_TAG, "lock channel state");
    esp_err_t err = max22200_activate_unlocked(device);
    if (err == ESP_OK) {
        err = max22200_status_update_unlocked(device,
                                              MAX22200_ONCH_MASK(channel),
                                              enabled ? MAX22200_ONCH_MASK(channel) : 0U);
    }
    max22200_unlock(device);
    return err;
}

esp_err_t max22200_set_all_channels_off(max22200_t *device)
{
    ESP_RETURN_ON_ERROR(max22200_lock(device), MAX_TAG, "lock all channels off");
    esp_err_t err = max22200_activate_unlocked(device);
    if (err == ESP_OK) {
        err = max22200_status_update_unlocked(device, MAX22200_STATUS_ONCH_MASK, 0U);
    }
    max22200_unlock(device);
    return err;
}
