#include "tle_verify.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "board_pins.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ltc1864.h"
#include "mcp2515_min.h"
#include "tle92464.h"

#define V_TAG "tle-verify"

#define VERIFY_ICVID_READS 5
#define VERIFY_ADC_SAMPLES 32
#define VERIFY_REPORT_PERIOD_MS 3000U

/* FB_VOLTAGE LSB scalings (datasheet 5.3.2.24/25), kept in integer microvolt
 * math so the report does not depend on printf float formatting. */
static uint32_t fb_voltage_code_to_mv(uint32_t code)
{
    /* V = 0.0034534 V * code  ->  mV = code * 3.4534. */
    return (code * 34534U) / 10000U;
}

static uint32_t fb_vbat_code_to_mv(uint32_t code)
{
    /* V = 41.47 V * code / (2^11 - 1). */
    return (code * 41470U) / 2047U;
}

static bool read_reg(tle92464_t *tle, uint16_t addr, tle92464_reply_t *reply)
{
    esp_err_t err = tle92464_read_reg(tle, addr, reply);
    if (err != ESP_OK) {
        ESP_LOGW(V_TAG, "  read 0x%04X failed: %s", addr, esp_err_to_name(err));
        return false;
    }
    return true;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/* Read a register N times and return the median data value -- robust to the
 * occasional MISO-corrupted read on the pull-up-less shared net. */
static uint32_t read_reg_median(tle92464_t *tle, uint16_t addr, int n, int *valid_out)
{
    uint32_t vals[33];
    if (n > 33) n = 33;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        tle92464_reply_t r;
        if (tle92464_read_reg(tle, addr, &r) == ESP_OK) {
            vals[valid++] = r.data;
        }
    }
    if (valid_out) *valid_out = valid;
    if (valid == 0) return 0;
    qsort(vals, valid, sizeof(vals[0]), cmp_u32);
    return vals[valid / 2];
}

/* Robust supply diagnosis: vote/median the registers that gate Mission Mode so
 * MISO glitches do not mislead. This is the decisive check for why Config->
 * Mission is refused. */
static void verify_robust_supply(tle92464_t *tle)
{
    /* Clear-effectiveness test. Some TLE diagnosis latches need TWO clear
     * commands (datasheet). Write 0xFFFF to GLOBAL_DIAG0 and re-read after one
     * and after two clears. If VIO_UV/VDD_OV survive an immediate clear they are
     * either truly live (re-asserting instantly) or need the double clear; if a
     * second clear removes them, the live condition is gone and the firmware
     * clear was just insufficient. */
    int cv1 = 0, vio1 = 0, ov1 = 0;
    (void)tle92464_write_reg(tle, 0x03U, 0xFFFFU);
    for (int i = 0; i < 11; ++i) {
        tle92464_reply_t r;
        if (tle92464_read_reg(tle, TLE92464_GLOBAL_DIAG0_RD, &r) == ESP_OK && r.mode == TLE92464_REPLY_16BIT) {
            ++cv1;
            if (r.data & TLE92464_DIAG0_VIO_UV) ++vio1;
            if (r.data & TLE92464_DIAG0_VDD_OV) ++ov1;
        }
    }
    int cv2 = 0, vio2 = 0, ov2 = 0;
    (void)tle92464_write_reg(tle, 0x03U, 0xFFFFU);
    (void)tle92464_write_reg(tle, 0x03U, 0xFFFFU);
    for (int i = 0; i < 11; ++i) {
        tle92464_reply_t r;
        if (tle92464_read_reg(tle, TLE92464_GLOBAL_DIAG0_RD, &r) == ESP_OK && r.mode == TLE92464_REPLY_16BIT) {
            ++cv2;
            if (r.data & TLE92464_DIAG0_VIO_UV) ++vio2;
            if (r.data & TLE92464_DIAG0_VDD_OV) ++ov2;
        }
    }
    ESP_LOGW(V_TAG, "  CLEAR-TEST after 1 clear: VIO_UV=%d/%d VDD_OV=%d/%d | after 2 clears: VIO_UV=%d/%d VDD_OV=%d/%d",
             vio1, cv1, ov1, cv1, vio2, cv2, ov2, cv2);

    /* VIO_SEL: count how many GLOBAL_CONFIG reads show 5 V mode (bit14). */
    int gc_valid = 0, vio_sel_5v = 0;
    uint32_t gc_median = read_reg_median(tle, 0x0002U, 31, &gc_valid);
    for (int i = 0; i < 31; ++i) {
        tle92464_reply_t r;
        if (tle92464_read_reg(tle, 0x0002U, &r) == ESP_OK && ((r.data >> 14) & 1U)) ++vio_sel_5v;
    }
    /* FB_VOLTAGE1 medians: VIO bits[10:0], VDD bits[21:11]. */
    int v1_valid = 0;
    uint32_t v1 = read_reg_median(tle, TLE92464_FB_VOLTAGE1, 31, &v1_valid);
    const uint32_t vio_mv = fb_voltage_code_to_mv(v1 & 0x7FFU);
    const uint32_t vdd_mv = fb_voltage_code_to_mv((v1 >> 11) & 0x7FFU);
    /* FB_STAT SUP_NOK_EXT (bit12) vote. */
    int fs_valid = 0, sup_nok_ext = 0;
    for (int i = 0; i < 31; ++i) {
        tle92464_reply_t r;
        if (tle92464_read_reg(tle, TLE92464_FB_STAT, &r) == ESP_OK && r.mode == TLE92464_REPLY_22BIT) {
            ++fs_valid;
            if (r.data & (1UL << 12)) ++sup_nok_ext;
        }
    }
    ESP_LOGW(V_TAG, "  ROBUST: GLOBAL_CONFIG~0x%04" PRIX32 " VIO_SEL=5V in %d/31 reads => %s",
             gc_median & 0xFFFFU, vio_sel_5v, vio_sel_5v > 15 ? "5V MODE (BUG: VIO_UV trips at 3.3V!)" : "3.3V mode ok");
    ESP_LOGW(V_TAG, "  ROBUST: VIO(median)=%" PRIu32 " mV (UV<3000 in 3V3 mode), VDD(median)=%" PRIu32 " mV (UV<4500)",
             vio_mv, vdd_mv);
    ESP_LOGW(V_TAG, "  ROBUST: SUP_NOK_EXT set in %d/%d FB_STAT reads", sup_nok_ext, fs_valid);

    /* Per-bit majority vote on GLOBAL_DIAG0: a genuinely-set fault bit reads 1
     * in most reads; MISO-float artifacts stay a minority. This pins down WHICH
     * external supply the chip thinks is faulting. */
    int d_valid = 0;
    int bitcount[6] = {0, 0, 0, 0, 0, 0}; /* VBAT_UV,VBAT_OV,VIO_UV,VIO_OV,VDD_UV,VDD_OV */
    for (int i = 0; i < 41; ++i) {
        tle92464_reply_t r;
        if (tle92464_read_reg(tle, TLE92464_GLOBAL_DIAG0_RD, &r) == ESP_OK && r.mode == TLE92464_REPLY_16BIT) {
            ++d_valid;
            for (int b = 0; b < 6; ++b) {
                if (r.data & (1U << b)) ++bitcount[b];
            }
        }
    }
    ESP_LOGW(V_TAG, "  ROBUST DIAG0 vote /%d: VBAT_UV=%d VBAT_OV=%d VIO_UV=%d VIO_OV=%d VDD_UV=%d VDD_OV=%d",
             d_valid, bitcount[0], bitcount[1], bitcount[2], bitcount[3], bitcount[4], bitcount[5]);
}

/* Re-readable TLE status block (no re-init); safe to call repeatedly. */
static void verify_tle_report(tle92464_t *tle)
{
    ESP_LOGI(V_TAG, "-- TLE92464 --");
    verify_robust_supply(tle);

    /* ICVID repeatability -- the classic "is SPI alive" check. */
    uint32_t first = 0;
    bool stable = true;
    bool trivial = true;
    for (int i = 0; i < VERIFY_ICVID_READS; ++i) {
        tle92464_reply_t reply;
        if (!read_reg(tle, TLE92464_FB_ICVID, &reply)) {
            stable = false;
            continue;
        }
        if (i == 0) {
            first = reply.raw;
        } else if (reply.raw != first) {
            stable = false;
        }
        if (reply.raw != 0x00000000UL && reply.raw != 0xFFFFFFFFUL) {
            trivial = false;
        }
    }
    const bool spi_ok = stable && !trivial;
    ESP_LOGI(V_TAG, "  ICVID=0x%08" PRIX32 " (x%d)  SPI verdict: %s (stable=%d non-trivial=%d)",
             first, VERIFY_ICVID_READS, spi_ok ? "COMMUNICATING" : "NOT communicating",
             (int)stable, (int)!trivial);

    /* FB_STAT: INIT_DONE + supply-fault summary. */
    tle92464_reply_t fb_stat;
    if (read_reg(tle, TLE92464_FB_STAT, &fb_stat)) {
        ESP_LOGI(V_TAG, "  FB_STAT=0x%06" PRIX32 " INIT_DONE=%d SUP_NOK_EXT=%d",
                 fb_stat.data,
                 (int)((fb_stat.data & TLE92464_FB_STAT_INIT_DONE_BIT) != 0U),
                 (int)((fb_stat.data & TLE92464_FB_STAT_SUP_NOK_EXT_BIT) != 0U));
    }

    /* GLOBAL_CONFIG: confirm VIO_SEL actually took (3.3 V = bit14 clear). A
     * stuck 5 V selection keeps VIO_UV live at 3.3 V and blocks Mission Mode. */
    tle92464_reply_t gconf;
    if (read_reg(tle, 0x0002U, &gconf)) {
        const uint32_t vio_sel = (gconf.data >> 14) & 0x1U;
        ESP_LOGI(V_TAG, "  GLOBAL_CONFIG=0x%04" PRIX32 " VIO_SEL=%" PRIu32 " (%s)",
                 gconf.data & 0xFFFFU, vio_sel, vio_sel ? "5.0V" : "3.3V");
    }

    /* GLOBAL_DIAG0: which supply rails are flagged. Read it many times and
     * bitwise-AND/OR the results to separate genuine persistent faults (set in
     * EVERY read) from MISO-float artifacts (the shared net has no pull-up, so
     * idle bit windows read as spurious 1s and flicker between reads). AND =
     * bits real in all reads; OR = bits seen in any read. */
    uint16_t diag_and = 0xFFFFU;
    uint16_t diag_or = 0U;
    int diag_reads = 0;
    for (int i = 0; i < 24; ++i) {
        tle92464_reply_t d;
        if (read_reg(tle, TLE92464_GLOBAL_DIAG0_RD, &d) && d.mode == TLE92464_REPLY_16BIT) {
            diag_and &= (uint16_t)d.data;
            diag_or |= (uint16_t)d.data;
            ++diag_reads;
        }
    }
    ESP_LOGI(V_TAG, "  GLOBAL_DIAG0 x%d  AND=0x%04X OR=0x%04X", diag_reads, diag_and, diag_or);
    ESP_LOGI(V_TAG,
             "  DIAG0(AND/persistent): VBAT_UV=%d VBAT_OV=%d VIO_UV=%d VIO_OV=%d VDD_UV=%d VDD_OV=%d POR=%d",
             (int)((diag_and & TLE92464_DIAG0_VBAT_UV) != 0U),
             (int)((diag_and & TLE92464_DIAG0_VBAT_OV) != 0U),
             (int)((diag_and & TLE92464_DIAG0_VIO_UV) != 0U),
             (int)((diag_and & TLE92464_DIAG0_VIO_OV) != 0U),
             (int)((diag_and & TLE92464_DIAG0_VDD_UV) != 0U),
             (int)((diag_and & TLE92464_DIAG0_VDD_OV) != 0U),
             (int)((diag_and & TLE92464_DIAG0_POR_EVENT) != 0U));
    tle92464_reply_t vth;
    if (read_reg(tle, 0x0006U, &vth)) {
        const uint32_t uv = vth.data & 0xFFU;
        const uint32_t ov = (vth.data >> 8) & 0xFFU;
        ESP_LOGI(V_TAG, "  VBAT_TH=0x%04" PRIX32 " UV_TH=%" PRIu32 "(~%" PRIu32 "mV) OV_TH=%" PRIu32 "(~%" PRIu32 "mV)",
                 vth.data & 0xFFFFU, uv, (uv * 16208U) / 100U, ov, (ov * 16208U) / 100U);
    }

    /* Measured supplies. VDD ~5000 mV, VIO ~3300 mV, VBAT ~0 (not connected). */
    tle92464_reply_t v1;
    if (read_reg(tle, TLE92464_FB_VOLTAGE1, &v1)) {
        const uint32_t vio_code = v1.data & 0x7FFU;
        const uint32_t vdd_code = (v1.data >> 11) & 0x7FFU;
        ESP_LOGI(V_TAG, "  FB_VOLTAGE1: VDD=%" PRIu32 " mV VIO=%" PRIu32 " mV",
                 fb_voltage_code_to_mv(vdd_code), fb_voltage_code_to_mv(vio_code));
    }
    tle92464_reply_t v2;
    if (read_reg(tle, TLE92464_FB_VOLTAGE2, &v2)) {
        const uint32_t vbat_code = (v2.data >> 11) & 0x7FFU;
        ESP_LOGI(V_TAG, "  FB_VOLTAGE2: VBAT=%" PRIu32 " mV (expect ~0 -- VBAT not connected)",
                 fb_vbat_code_to_mv(vbat_code));
    }

    if (tle->mission_mode) {
        ESP_LOGI(V_TAG, "  Mission Mode: REACHED (outputs can be enabled -- VBAT present)");
    } else {
        ESP_LOGW(V_TAG, "  Mission Mode: NOT reached -- EXPECTED without VBAT (VBAT_UV blocks the");
        ESP_LOGW(V_TAG, "    Config->Mission transition). SPI/config healthy; connect VBAT 6-18V + EN to drive valves.");
    }
}

static void verify_adc(void)
{
    uint16_t min_raw = 0xFFFFU;
    uint16_t max_raw = 0;
    uint32_t sum = 0;
    uint32_t errors = 0;
    for (int i = 0; i < VERIFY_ADC_SAMPLES; ++i) {
        uint16_t raw = 0;
        if (ltc1864_read_raw(&raw) == ESP_OK) {
            if (raw < min_raw) {
                min_raw = raw;
            }
            if (raw > max_raw) {
                max_raw = raw;
            }
            sum += raw;
        } else {
            ++errors;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }
    const uint32_t got = VERIFY_ADC_SAMPLES - errors;
    const uint32_t avg = got != 0U ? sum / got : 0U;
    const bool stuck = (min_raw == max_raw) || (min_raw == 0U && max_raw == 0U) ||
                       (min_raw == 0xFFFFU && max_raw == 0xFFFFU);
    ESP_LOGI(V_TAG, "-- LTC1864 ADC -- samples=%" PRIu32 " errors=%" PRIu32 " raw min=%u max=%u avg=%" PRIu32 " => %s",
             got, errors, min_raw, max_raw, avg,
             (errors == 0U && !stuck) ? "COMMUNICATING (live)"
                                      : (got == 0U ? "NOT communicating" : "reads OK but value stuck"));
}

/* MAX22200 is write-only on V2 (no SDO pull-up), so the FAULT pin is the only
 * observable: probe it against the internal pull-up/pull-down. Driven high in
 * both cases = powered MAX (or external pull-up) holding FAULT inactive;
 * following the pulls = floating net (chip absent/unpowered); low in both =
 * FAULT asserted. */
static void verify_max_fault_pin(void)
{
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << PIN_MAX22200_FAULT,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    (void)gpio_config(&io);
    vTaskDelay(pdMS_TO_TICKS(2));
    const int with_pu = gpio_get_level(PIN_MAX22200_FAULT);
    (void)gpio_set_pull_mode(PIN_MAX22200_FAULT, GPIO_PULLDOWN_ONLY);
    vTaskDelay(pdMS_TO_TICKS(2));
    const int with_pd = gpio_get_level(PIN_MAX22200_FAULT);
    (void)gpio_set_pull_mode(PIN_MAX22200_FAULT, GPIO_FLOATING);
    ESP_LOGI(V_TAG, "-- MAX22200 FAULT pin (GPIO%d, write-only chip) -- pu=%d pd=%d => %s",
             PIN_MAX22200_FAULT, with_pu, with_pd,
             (with_pu == 1 && with_pd == 1) ? "driven HIGH (FAULT inactive; chip/pull-up present)"
             : (with_pu == 1 && with_pd == 0) ? "FLOATING (chip absent or unpowered, no pull-up)"
                                              : "held LOW (FAULT asserted or short)");
}

/* Dump individual FB_VOLTAGE1 reads with NO averaging -- print the full 32-bit
 * MISO word + decoded VIO/VDD for each, so the corruption pattern (shift /
 * bit-flip / pipeline misalign) is visible read-by-read. */
static void verify_voltage_dump(tle92464_t *tle)
{
    ESP_LOGW(V_TAG, "  VOLT-DUMP FB_VOLTAGE1 (no avg) -- raw / mode / VIO / VDD (expect VIO~3300 VDD~5050):");
    for (int i = 0; i < 16; ++i) {
        tle92464_reply_t r;
        esp_err_t e = tle92464_read_reg(tle, TLE92464_FB_VOLTAGE1, &r);
        if (e != ESP_OK) {
            ESP_LOGW(V_TAG, "    [%2d] read err %s", i, esp_err_to_name(e));
            continue;
        }
        const uint32_t vio_code = r.data & 0x7FFU;
        const uint32_t vdd_code = (r.data >> 11) & 0x7FFU;
        ESP_LOGW(V_TAG, "    [%2d] raw=0x%08" PRIX32 " mode=%d VIO=%" PRIu32 "mV(0x%03" PRIX32 ") VDD=%" PRIu32 "mV(0x%03" PRIX32 ")",
                 i, r.raw, (int)r.mode,
                 fb_voltage_code_to_mv(vio_code), vio_code,
                 fb_voltage_code_to_mv(vdd_code), vdd_code);
    }
}

/* =====================================================================
 * Bit-banged frame-reject probe (TLE_ALL_IN_ONE bring-up)
 *
 * Used when every SPI reply is one constant diagnosis frame (e.g. 0xE5810000):
 * that means the chip drives SO and frames our clocks correctly but accepts
 * none of our requests. Two candidate causes with opposite fixes:
 *   - SI never reaches the chip (open joint / wrong net) -> every frame is
 *     garbage to it, all variants below reply identically;
 *   - the chip validates frames with a different CRC convention than ours ->
 *     exactly one variant below suddenly gets a real (mode 0/1) reply.
 * Bit-bangs CS/SCK/SI as plain GPIOs at ~50 kHz so the SPI peripheral and its
 * timing are out of the loop; logs both pipeline frames of each read. This
 * HIJACKS the SPI pads via the GPIO matrix -- run it last, no SPI after it.
 * ===================================================================== */

static uint8_t probe_crc8(const uint8_t *d, int n, uint8_t init, bool xorout)
{
    uint8_t crc = init;
    for (int i = 0; i < n; ++i) {
        crc ^= d[i];
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x1DU) : (uint8_t)(crc << 1);
        }
    }
    return xorout ? (uint8_t)(crc ^ 0xFFU) : crc;
}

static uint8_t probe_reflect8(uint8_t v)
{
    v = (uint8_t)(((v & 0xF0U) >> 4) | ((v & 0x0FU) << 4));
    v = (uint8_t)(((v & 0xCCU) >> 2) | ((v & 0x33U) << 2));
    v = (uint8_t)(((v & 0xAAU) >> 1) | ((v & 0x55U) << 1));
    return v;
}

/* One 32-clock SPI mode-1 frame, bit-banged: drive SI on the rising edge,
 * sample SO on the falling edge, MSB first. */
static uint32_t probe_xfer32(uint32_t tx)
{
    const int half_us = 10; /* ~50 kHz */
    uint32_t rx = 0;
    gpio_set_level(PIN_SPI_CS_TLE, 0);
    esp_rom_delay_us(half_us);
    for (int bit = 31; bit >= 0; --bit) {
        gpio_set_level(PIN_MAXLTC_MOSI, (int)((tx >> bit) & 1U));
        gpio_set_level(PIN_MAXLTC_SCK, 1);
        esp_rom_delay_us(half_us);
        gpio_set_level(PIN_MAXLTC_SCK, 0);
        rx = (rx << 1) | (uint32_t)gpio_get_level(PIN_MAXLTC_MISO);
        esp_rom_delay_us(half_us);
    }
    gpio_set_level(PIN_SPI_CS_TLE, 1);
    esp_rom_delay_us(5);
    return rx;
}

static void probe_read_twice(const char *label, uint32_t tx)
{
    const uint32_t rx1 = probe_xfer32(tx);
    const uint32_t rx2 = probe_xfer32(tx);
    ESP_LOGW(V_TAG, "  %-22s tx=0x%08" PRIX32 " rx1=0x%08" PRIX32 " rx2=0x%08" PRIX32,
             label, tx, rx1, rx2);
}

/* Measure the SO/MISO net (GPIO4 = ADC1_CH3) with the ESP32's own ADC to
 * pin down the TLE's VIO domain. The chip's monitor claims VIO ~1 V while its
 * SPI I/O behaves like 3.3 V; the SO drive levels arbitrate:
 *   - VIO pad really at 3.3 V  -> idle ~3.3 V (R20), SO-driving-1 ~3.3 V
 *   - VIO pad floating (~1 V)  -> idle dragged to ~1.6 V by the pad's clamp
 *                                 diode, SO-driving-1 ~1.0-1.7 V
 *   - VIO fine, monitor broken -> everything reads ~3.3 V (chip defect). */
static uint32_t probe_adc_mv(adc_oneshot_unit_handle_t adc)
{
    int sum = 0;
    for (int i = 0; i < 8; ++i) {
        int r = 0;
        (void)adc_oneshot_read(adc, ADC_CHANNEL_3, &r);
        sum += r;
    }
    /* 12 dB attenuation: ~3100 mV full scale at 4095 (uncalibrated; the three
     * hypotheses are ~1.0 / ~1.6 / ~3.1+ V, far apart). */
    return ((uint32_t)(sum / 8) * 3100U) / 4095U;
}

static void verify_miso_analog_levels(void)
{
    ESP_LOGW(V_TAG, "==== SO/MISO ANALOG LEVELS (ESP ADC1_CH3 on GPIO4) ====");
    adc_oneshot_unit_handle_t adc = NULL;
    adc_oneshot_unit_init_cfg_t unit_cfg = {.unit_id = ADC_UNIT_1};
    if (adc_oneshot_new_unit(&unit_cfg, &adc) != ESP_OK) {
        ESP_LOGE(V_TAG, "  ADC init failed");
        return;
    }
    adc_oneshot_chan_cfg_t ch_cfg = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_12};
    (void)adc_oneshot_config_channel(adc, ADC_CHANNEL_3, &ch_cfg);

    /* Idle: CS high, nothing drives, only R20. */
    gpio_set_level(PIN_SPI_CS_TLE, 1);
    esp_rom_delay_us(100);
    ESP_LOGW(V_TAG, "  idle (R20 pull-up only): ~%" PRIu32 " mV", probe_adc_mv(adc));

    /* Prime the pipeline with a valid ICVID request, then clock ONE bit of the
     * reply frame (reply 0x9000C1FF: bit31 = 1) and pause with CS low while SO
     * actively drives that 1. */
    const uint8_t lsb[3] = {0x00, 0x02, 0x00};
    const uint32_t req = ((uint32_t)probe_crc8(lsb, 3, 0xFF, true) << 24) | 0x0200U;
    (void)probe_xfer32(req);
    gpio_set_level(PIN_SPI_CS_TLE, 0);
    esp_rom_delay_us(10);
    gpio_set_level(PIN_MAXLTC_MOSI, (int)((req >> 31) & 1U));
    gpio_set_level(PIN_MAXLTC_SCK, 1); /* chip launches reply bit31 = 1 */
    esp_rom_delay_us(10);
    ESP_LOGW(V_TAG, "  SO driving '1' (reply MSB of 0x9000C1FF): ~%" PRIu32 " mV", probe_adc_mv(adc));
    gpio_set_level(PIN_MAXLTC_SCK, 0);
    esp_rom_delay_us(10);
    /* Clock two more bits: reply bits 30,29 = 0,0 -- SO now drives a 0. */
    for (int i = 0; i < 2; ++i) {
        gpio_set_level(PIN_MAXLTC_SCK, 1);
        esp_rom_delay_us(10);
        gpio_set_level(PIN_MAXLTC_SCK, 0);
        esp_rom_delay_us(10);
    }
    ESP_LOGW(V_TAG, "  SO driving '0' (reply bit29): ~%" PRIu32 " mV", probe_adc_mv(adc));
    gpio_set_level(PIN_SPI_CS_TLE, 1);
    (void)adc_oneshot_del_unit(adc);
    ESP_LOGW(V_TAG, "==== ~3100mV everywhere = VIO pin fine, chip monitor broken;");
    ESP_LOGW(V_TAG, "     idle ~1.6V + '1' ~1.0-1.7V = VIO pin NOT connected (joint) ====");
}

static void verify_bitbang_probe(void)
{
    ESP_LOGW(V_TAG, "==== BIT-BANG FRAME-REJECT PROBE (pads hijacked; no SPI after this) ====");

    /* Pin levels before the hijack: FAULTN/RESN context. */
    ESP_LOGW(V_TAG, "  control pins: EN(GPIO%d)=out FAULTN(GPIO%d)=%d RESN(GPIO%d)=out",
             PIN_TLE92464_EN, PIN_TLE92464_FAULT, gpio_get_level(PIN_TLE92464_FAULT),
             PIN_TLE92464_RESET);

    gpio_config_t out = {
        .pin_bit_mask = (1ULL << PIN_SPI_CS_TLE) | (1ULL << PIN_MAXLTC_SCK) |
                        (1ULL << PIN_MAXLTC_MOSI),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    (void)gpio_config(&out);
    gpio_config_t in = {
        .pin_bit_mask = 1ULL << PIN_MAXLTC_MISO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    (void)gpio_config(&in);
    gpio_set_level(PIN_SPI_CS_TLE, 1);
    gpio_set_level(PIN_MAXLTC_SCK, 0);
    gpio_set_level(PIN_MAXLTC_MOSI, 0);
    esp_rom_delay_us(50);

    /* SO idle while deselected: with the on-board pull-up (R20) this must read
     * 1; a hard 0 here is a short or a chip driving SO with CS high. */
    ESP_LOGW(V_TAG, "  SO idle (CS high): %d (expect 1: R20 pull-up)", gpio_get_level(PIN_MAXLTC_MISO));

    /* MOSI echo: reading our own pad back catches a short of the SI net. */
    gpio_set_direction(PIN_MAXLTC_MOSI, GPIO_MODE_INPUT_OUTPUT);
    gpio_set_level(PIN_MAXLTC_MOSI, 1);
    esp_rom_delay_us(5);
    const int echo_hi = gpio_get_level(PIN_MAXLTC_MOSI);
    gpio_set_level(PIN_MAXLTC_MOSI, 0);
    esp_rom_delay_us(5);
    const int echo_lo = gpio_get_level(PIN_MAXLTC_MOSI);
    ESP_LOGW(V_TAG, "  SI pad echo: drive1->%d drive0->%d (1/0 = net not shorted)", echo_hi, echo_lo);

    /* ICVID read request, CRC/layout variants. Payload bits: addr16=0x0200. */
    const uint32_t payload = 0x0200U; /* read ICVID, bits 23:0 */
    const uint8_t lsb_first[3] = {(uint8_t)payload, (uint8_t)(payload >> 8), (uint8_t)(payload >> 16)};
    const uint8_t msb_first[3] = {(uint8_t)(payload >> 16), (uint8_t)(payload >> 8), (uint8_t)payload};

    const uint8_t crc_std = probe_crc8(lsb_first, 3, 0xFF, true);   /* = driver's 0x69 */
    const uint8_t crc_noxor = probe_crc8(lsb_first, 3, 0xFF, false);
    const uint8_t crc_i0 = probe_crc8(lsb_first, 3, 0x00, true);
    const uint8_t crc_i0nx = probe_crc8(lsb_first, 3, 0x00, false);
    const uint8_t crc_msb = probe_crc8(msb_first, 3, 0xFF, true);
    uint8_t refl[3] = {probe_reflect8(lsb_first[0]), probe_reflect8(lsb_first[1]),
                       probe_reflect8(lsb_first[2])};
    const uint8_t crc_refl = probe_reflect8(probe_crc8(refl, 3, 0xFF, true));

    probe_read_twice("baseline(std CRC)", ((uint32_t)crc_std << 24) | payload);
    probe_read_twice("crc no-xorout", ((uint32_t)crc_noxor << 24) | payload);
    probe_read_twice("crc init00", ((uint32_t)crc_i0 << 24) | payload);
    probe_read_twice("crc init00 no-xor", ((uint32_t)crc_i0nx << 24) | payload);
    probe_read_twice("crc msb-byte-order", ((uint32_t)crc_msb << 24) | payload);
    probe_read_twice("crc reflected", ((uint32_t)crc_refl << 24) | payload);
    probe_read_twice("crc-last layout", (payload << 8) | crc_std);
    probe_read_twice("all zeros", 0x00000000U);
    probe_read_twice("all ones", 0xFFFFFFFFU);
    /* Known-good breakout frames (datasheet worked examples, driver CRC). */
    probe_read_twice("FB_STAT read", 0x6A000202U);
    probe_read_twice("GLOBAL_CONFIG wr 3V3", 0x11050005U);
    probe_read_twice("baseline again", ((uint32_t)crc_std << 24) | payload);

    ESP_LOGW(V_TAG, "==== PROBE DONE. If ALL rows reply identically the chip never accepts a");
    ESP_LOGW(V_TAG, "     frame (suspect SI joint at U18.7 / SCK at U18.6). If one CRC variant");
    ESP_LOGW(V_TAG, "     got a mode-0 reply (e.g. data 0xC1FF), it's a CRC-convention part. ====");

    verify_miso_analog_levels();
}

/* Scope aid: alternate a dense TLE-read BURST (the TLE drives its SO output
 * from VIO every frame, so this is the heaviest VIO load the chip imposes) with
 * total TLE SPI silence. Watch VIO (and GND) on a scope: if VIO sags during the
 * 1 s BURST windows and recovers during the 1 s IDLE windows, VIO_UV is being
 * caused by the SPI/SO switching load (fix = stiffer VIO / decoupling at the
 * pin / pull-UP instead of pull-down). If VIO is flat in both, it's not SPI. */
static void verify_scope_stimulus(tle92464_t *tle)
{
    for (int rep = 0; rep < 4; ++rep) {
        ESP_LOGW(V_TAG, "  SCOPE: >>> BURST 1.0s (dense TLE reads, SO active) <<<");
        const int64_t t0 = esp_timer_get_time();
        while ((esp_timer_get_time() - t0) < 1000000) {
            tle92464_reply_t r;
            (void)tle92464_read_reg(tle, TLE92464_FB_STAT, &r);
        }
        ESP_LOGW(V_TAG, "  SCOPE: --- IDLE 1.0s (no TLE SPI) ---");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void tle_verify_run(spi_device_handle_t tle_spi, spi_device_handle_t can_spi)
{
    ESP_LOGI(V_TAG, "================ TLE92464 EXTENSION COMM SELF-TEST ================");
    ESP_LOGI(V_TAG, "Hookup: CS=GPIO%d shared SPI3 (MOSI=%d SCK=%d MISO=%d) EN=%d FAULT=%d RESET=%d. VDD=5V VIO=3V3.",
             PIN_SPI_CS_TLE, PIN_MAXLTC_MOSI, PIN_MAXLTC_SCK, PIN_MAXLTC_MISO,
             PIN_TLE92464_EN, PIN_TLE92464_FAULT, PIN_TLE92464_RESET);

    /* CRC / frame builder self-test (no hardware needed). */
    ESP_LOGI(V_TAG, "CRC/frame self-test: %s",
             tle92464_crc_selftest() ? "PASS (matches datasheet worked frames)"
                                     : "FAIL -- protocol implementation is broken");

    /* TLE92464 bring-up (once). */
    tle92464_t tle;
    esp_err_t tle_err = tle92464_init(&tle, tle_spi, PIN_TLE92464_EN, PIN_TLE92464_FAULT,
                                      PIN_TLE92464_RESET, true /* VIO = 3.3 V */);
    ESP_LOGI(V_TAG, "tle92464_init: %s spi_alive=%d mission_mode=%d (EN pin %s, FAULT pin %s)",
             esp_err_to_name(tle_err), (int)tle.spi_alive, (int)tle.mission_mode,
             PIN_TLE92464_EN >= 0 ? "wired" : "NC", PIN_TLE92464_FAULT >= 0 ? "wired" : "NC");

    /* MCP25625 CAN controller bring-up (once; bus traffic not testable -- CAN
     * to PC not wired). */
    mcp2515_t mcp;
    esp_err_t can_err = mcp2515_init(&mcp, can_spi);
    if (can_err == ESP_OK) {
        can_err = mcp2515_configure_1mbps_16mhz(&mcp);
    }
    ESP_LOGI(V_TAG, "MCP25625 CAN controller: %s => %s",
             esp_err_to_name(can_err),
             can_err == ESP_OK ? "CONTROLLER OK (SPI + CONFIG mode); bus to PC NOT tested"
                               : "NOT responding on SPI");

    const bool crc_ok = tle92464_crc_selftest();
    ESP_LOGI(V_TAG, "================ live report every %u s ================",
             (unsigned)(VERIFY_REPORT_PERIOD_MS / 1000U));

    (void)crc_ok;
    /* One context report up front, then the clear-UV + re-arm @1Hz test below. */
    verify_tle_report(&tle);
    verify_voltage_dump(&tle);
    verify_adc();
    verify_max_fault_pin();

    ESP_LOGW(V_TAG, "==== CLEAR-UV + RE-ARM @1Hz (slow-ramp latch test) ====");
    ESP_LOGW(V_TAG, "  Reads DIAG0 (before), writes the 0xFFFF clear, reads DIAG0 (after_clear),");
    ESP_LOGW(V_TAG, "  then re-arms. If after_clear UV=0 the latch clears (ramp artifact); if it");
    ESP_LOGW(V_TAG, "  stays 1 the fault is live (real or bad monitor reference).");
    uint32_t cycle = 0;
    while (cycle < 3) {
        tle92464_reply_t before;
        esp_err_t eb = tle92464_read_reg(&tle, TLE92464_GLOBAL_DIAG0_RD, &before);
        /* Explicit clear of the latched supply-fault flags (DIAG0 + DIAG1).
         * These are WRITE-0-TO-CLEAR (datasheet POR_EVENT: "set the bit to 0");
         * writing 0xFFFF was a no-op -- write 0x0000 to actually clear. */
        (void)tle92464_write_reg(&tle, 0x03U, 0x0000U);
        (void)tle92464_write_reg(&tle, 0x04U, 0x0000U);
        tle92464_reply_t after;
        esp_err_t ea = tle92464_read_reg(&tle, TLE92464_GLOBAL_DIAG0_RD, &after);
        /* Re-arm: re-assert VIO config, clear again, request Mission, read OP_MODE. */
        (void)tle92464_enter_mission_mode(&tle);
        ESP_LOGW(V_TAG,
                 "  [%" PRIu32 "] before=0x%04" PRIX32 "(VIO_UV=%d VDD_UV=%d VDD_OV=%d)%s -> after_clear=0x%04" PRIX32 "(VIO_UV=%d VDD_UV=%d)%s -> mission=%d",
                 cycle++,
                 before.data & 0xFFFFU,
                 (int)((before.data & TLE92464_DIAG0_VIO_UV) != 0U),
                 (int)((before.data & TLE92464_DIAG0_VDD_UV) != 0U),
                 (int)((before.data & TLE92464_DIAG0_VDD_OV) != 0U),
                 eb == ESP_OK ? "" : "(readerr)",
                 after.data & 0xFFFFU,
                 (int)((after.data & TLE92464_DIAG0_VIO_UV) != 0U),
                 (int)((after.data & TLE92464_DIAG0_VDD_UV) != 0U),
                 ea == ESP_OK ? "" : "(readerr)",
                 (int)tle.mission_mode);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    verify_bitbang_probe();
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
