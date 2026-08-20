#include "solenoid.h"

#include "board_pins.h"
#include "esp_check.h"
#include "esp_log.h"
#include "max22200.h"
#include "tle92464.h"

#define SOLENOID_TAG "solenoid"

/* The TLE breakout's VIO is fed from the board's 3.3 V LDO. */
#define SOLENOID_TLE_VIO_IS_3V3 true

static uint8_t s_driver = VEMA_SOLENOID_DRIVER_TLE92464;
static max22200_t s_max22200;
static tle92464_t s_tle92464;

esp_err_t solenoid_init(uint8_t driver, spi_device_handle_t spi)
{
    s_driver = driver;

    if (driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        ESP_RETURN_ON_ERROR(max22200_init(&s_max22200, spi, PIN_MAX22200_CMD, PIN_MAX22200_FAULT),
                            SOLENOID_TAG, "MAX22200 init");
        ESP_LOGI(SOLENOID_TAG, "MAX22200 backend (write-only, 7-bit code, software dither)");
        return max22200_set_all_channels_off(&s_max22200);
    }

    if (!tle92464_crc_selftest()) {
        ESP_LOGE(SOLENOID_TAG, "TLE92464 CRC self-test FAILED -- frame builder is broken");
    }
    esp_err_t err = tle92464_init(&s_tle92464, spi, PIN_TLE92464_EN, PIN_TLE92464_FAULT,
                                  PIN_TLE92464_RESET, SOLENOID_TLE_VIO_IS_3V3);
    if (err != ESP_OK) {
        ESP_LOGE(SOLENOID_TAG, "TLE92464 SPI did not respond: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(SOLENOID_TAG,
             "TLE92464 backend: spi_alive=%d mission_mode=%d (EN pin %s). %s",
             (int)s_tle92464.spi_alive, (int)s_tle92464.mission_mode,
             PIN_TLE92464_EN >= 0 ? "wired" : "NOT wired",
             s_tle92464.mission_mode ? "outputs armed"
                                     : "Mission Mode not reached -- connect VBAT (6-18 V) to drive valves");
    return tle92464_set_all_channels_off(&s_tle92464);
}

const char *solenoid_name(void)
{
    return s_driver == VEMA_SOLENOID_DRIVER_MAX22200 ? "MAX22200" : "TLE92464";
}

bool solenoid_supports_fine_current(void)
{
    return s_driver == VEMA_SOLENOID_DRIVER_TLE92464;
}

int solenoid_get_fault(void)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        return max22200_get_fault(&s_max22200);
    }
    return tle92464_get_fault(&s_tle92464);
}

bool solenoid_is_armed(void)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        return true;
    }
    return s_tle92464.mission_mode;
}

esp_err_t solenoid_arm(void)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200 || s_tle92464.mission_mode) {
        return ESP_OK;
    }
    esp_err_t err = tle92464_enter_mission_mode(&s_tle92464);
    if (s_tle92464.mission_mode) {
        ESP_LOGI(SOLENOID_TAG, "TLE92464 armed: Mission Mode reached (VBAT present)");
    }
    return err;
}

esp_err_t solenoid_configure_valve_channel(uint8_t channel, uint8_t hit_current,
                                           uint8_t hold_current, uint8_t hit_time,
                                           solenoid_drive_t drive)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        const max22200_valve_drive_t max_drive = (drive == SOLENOID_DRIVE_ONOFF_VOLTAGE)
                                                      ? MAX22200_VALVE_DRIVE_HIGH_SIDE_VOLTAGE
                                                      : MAX22200_VALVE_DRIVE_LOW_SIDE_CURRENT;
        return max22200_configure_valve_channel(&s_max22200, channel, hit_current, hold_current,
                                                hit_time, max_drive);
    }

    if (channel >= TLE92464_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }
    /* The TLE runs every channel in ICC (current control) mode -- set at init.
     * For on/off (bang-bang) drive the enable simply switches a fixed current
     * setpoint derived from the hit current code; for proportional drive the
     * setpoint starts at 0 and is driven by the controller. */
    const uint16_t setpoint_code_q9 = (drive == SOLENOID_DRIVE_ONOFF_VOLTAGE)
                                          ? (uint16_t)((uint16_t)hit_current << SOLENOID_CODE_FRAC_BITS)
                                          : 0U;
    return tle92464_set_channel_code_q9(&s_tle92464, channel, setpoint_code_q9);
}

esp_err_t solenoid_set_channel_current_code_q9(uint8_t channel, uint16_t code_q9)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        return max22200_set_channel_current_code(&s_max22200, channel,
                                                 (uint8_t)(code_q9 >> SOLENOID_CODE_FRAC_BITS));
    }
    return tle92464_set_channel_code_q9(&s_tle92464, channel, code_q9);
}

esp_err_t solenoid_set_channel_dither(uint8_t channel, uint8_t steps, uint8_t flat,
                                      uint16_t step_size, uint16_t mant, uint8_t exp,
                                      bool deep)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        return ESP_OK; /* MAX22200 has no hardware dither */
    }
    const tle92464_dither_t dither = {
        .steps = steps,
        .flat = flat,
        .step_size = step_size,
        .mant = mant,
        .exp = exp,
        .deep = deep,
    };
    return tle92464_set_channel_dither(&s_tle92464, channel, &dither);
}

bool solenoid_supports_hw_dither(void)
{
    return s_driver == VEMA_SOLENOID_DRIVER_TLE92464;
}

esp_err_t solenoid_set_channel_state(uint8_t channel, bool enabled)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        return max22200_set_channel_state(&s_max22200, channel, enabled);
    }
    return tle92464_set_channel_state(&s_tle92464, channel, enabled);
}

esp_err_t solenoid_set_all_channels_off(void)
{
    if (s_driver == VEMA_SOLENOID_DRIVER_MAX22200) {
        return max22200_set_all_channels_off(&s_max22200);
    }
    return tle92464_set_all_channels_off(&s_tle92464);
}
