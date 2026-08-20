#include "tle92464.h"

#include <inttypes.h>

#include "esp_check.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "freertos/task.h"
#include "shared_spi_bus.h"

#define TLE_TAG "tle92464"

/* ---- Register addresses (write addr = 7-bit; read addr = 16-bit) ---------- */
#define TLE_CH_CTRL_WR 0x00U      /* OP_MODE bit15, EN_CHx bits3:0 */
#define TLE_CH_CTRL_RD 0x0000U
#define TLE_GLOBAL_CONFIG_WR 0x02U
#define TLE_GLOBAL_DIAG0_WR 0x03U
#define TLE_GLOBAL_DIAG1_WR 0x04U
#define TLE_CH_BASE(ch) (0x40U + (0x10U * (uint8_t)(ch)))
#define TLE_REG_SETPOINT_WR(ch) (TLE_CH_BASE(ch) + 0x00U)
#define TLE_REG_DITHER_CLK_WR(ch) (TLE_CH_BASE(ch) + 0x04U)
#define TLE_REG_DITHER_STEP_WR(ch) (TLE_CH_BASE(ch) + 0x05U)
#define TLE_REG_DITHER_CTRL_WR(ch) (TLE_CH_BASE(ch) + 0x06U)
#define TLE_REG_MODE_WR(ch) (TLE_CH_BASE(ch) + 0x0CU)
#define TLE_DITHER_CTRL_DEEP (1U << 13)

#define TLE_MODE_ICC 0x0001U          /* autonomous current control */
#define TLE_CH_CTRL_MISSION 0x8000U   /* OP_MODE = Mission, all channels off */
#define TLE_GLOBAL_CONFIG_3V3 0x0005U /* VIO 3.3V, CRC on, SPI-WD off, clk-WD on */
#define TLE_GLOBAL_CONFIG_5V 0x4005U  /* VIO 5.0V, CRC on, SPI-WD off, clk-WD on */

/* RESETN low pulse and post-reset settle (datasheet tPOR ~ 0.1 ms). */
#define TLE_RESET_PULSE_US 20U
#define TLE_POST_RESET_MS 2U
/* INIT_DONE poll budget (~10 ms). */
#define TLE_INIT_DONE_POLL_COUNT 100
#define TLE_INIT_DONE_POLL_US 100U
/* CS-high gap between the two frames of a read (tCSN_TD >= 600 ns). */
#define TLE_INTERFRAME_US 2U

/* =====================================================================
 * CRC-8 SAE J1850 (poly 0x1D, init 0xFF, final XOR 0xFF) + frame builders
 * ===================================================================== */

static uint8_t crc8_j1850(const uint8_t *d, int n)
{
    uint8_t crc = 0xFF;
    for (int i = 0; i < n; ++i) {
        crc ^= d[i];
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x1DU) : (uint8_t)(crc << 1);
        }
    }
    return (uint8_t)(crc ^ 0xFFU);
}

/* CRC covers frame bytes [7:0], [15:8], [23:16] in that order. */
static uint32_t add_crc(uint32_t frame)
{
    const uint8_t d[3] = {(uint8_t)frame, (uint8_t)(frame >> 8), (uint8_t)(frame >> 16)};
    return ((uint32_t)crc8_j1850(d, 3) << 24) | (frame & 0x00FFFFFFUL);
}

static uint32_t frame_write(uint8_t addr7, uint16_t data)
{
    return add_crc(((uint32_t)(addr7 & 0x7FU) << 17) | (1UL << 16) | (uint32_t)data);
}

static uint32_t frame_read(uint16_t addr16)
{
    return add_crc((uint32_t)addr16);
}

static tle92464_reply_t parse_reply(uint32_t rx)
{
    tle92464_reply_t reply;
    reply.raw = rx;
    const uint8_t mode = (uint8_t)((rx >> 22) & 0x3U);
    if (mode == 0U) {
        reply.mode = TLE92464_REPLY_16BIT;
        reply.status = (uint8_t)((rx >> 17) & 0x1FU);
        reply.data = rx & 0xFFFFUL;
    } else if (mode == 1U) {
        reply.mode = TLE92464_REPLY_22BIT;
        reply.status = 0;
        reply.data = rx & 0x3FFFFFUL;
    } else {
        reply.mode = TLE92464_REPLY_CRITICAL;
        reply.status = 0xFFU;
        reply.data = rx & 0xFFFFUL;
    }
    return reply;
}

uint16_t tle92464_code_q9_to_setpoint(uint16_t code_q9)
{
    uint32_t setpoint = ((uint32_t)code_q9 * (uint32_t)TLE92464_SETPOINT_SAT) /
                        ((uint32_t)TLE92464_CODE_MAX << TLE92464_CODE_FRAC_BITS);
    if (setpoint > TLE92464_SETPOINT_SAT) {
        setpoint = TLE92464_SETPOINT_SAT;
    }
    return (uint16_t)setpoint;
}

bool tle92464_crc_selftest(void)
{
    /* Datasheet worked example frames (CRC pre-computed, verified). */
    struct {
        uint32_t got;
        uint32_t expect;
    } cases[] = {
        {frame_read(0x0202U), 0x6A000202UL},              /* READ FB_STAT */
        {frame_write(0x02U, 0x0005U), 0x11050005UL},      /* WRITE GLOBAL_CONFIG = 0x0005 */
        {frame_write(0x4CU, 0x0001U), 0x60990001UL},      /* WRITE CH0 MODE = 1 */
        {frame_write(0x00U, 0x8000U), 0x25018000UL},      /* WRITE CH_CTRL = 0x8000 */
        {frame_write(0x00U, 0x8001U), 0xAA018001UL},      /* WRITE CH_CTRL = 0x8001 */
        {frame_write(0x40U, 0x2000U), 0xBF812000UL},      /* WRITE CH0 SETPOINT = 0x2000 */
        {frame_read(0x0242U), 0x0A000242UL},              /* READ CH0 FB_I_AVG */
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (cases[i].got != cases[i].expect) {
            ESP_LOGE(TLE_TAG, "CRC self-test FAIL case %u: got 0x%08" PRIX32 " expect 0x%08" PRIX32,
                     (unsigned)i, cases[i].got, cases[i].expect);
            return false;
        }
    }
    return true;
}

/* =====================================================================
 * Bus transfers (device lock held by caller; shared bus acquired here)
 * ===================================================================== */

static esp_err_t tle_one_frame(tle92464_t *device, uint32_t mosi, uint32_t *miso_out)
{
    uint8_t tx[4] = {(uint8_t)(mosi >> 24), (uint8_t)(mosi >> 16), (uint8_t)(mosi >> 8), (uint8_t)mosi};
    uint8_t rx[4] = {0, 0, 0, 0};
    spi_transaction_t transaction = {
        .length = 32, /* exactly 32 clocks per CS-low window */
        .tx_buffer = tx,
        .rx_buffer = rx,
    };
    esp_err_t err = spi_device_transmit(device->spi, &transaction);
    if (err == ESP_OK && miso_out != NULL) {
        *miso_out = ((uint32_t)rx[0] << 24) | ((uint32_t)rx[1] << 16) | ((uint32_t)rx[2] << 8) | rx[3];
    }
    return err;
}

static esp_err_t tle_write_unlocked(tle92464_t *device, uint8_t addr7, uint16_t data)
{
    esp_err_t err = shared_spi_bus_acquire(SHARED_SPI_DEVICE_TLE92464);
    if (err != ESP_OK) {
        return err;
    }
    err = tle_one_frame(device, frame_write(addr7, data), NULL);
    const esp_err_t release_err = shared_spi_bus_release(SHARED_SPI_DEVICE_TLE92464);
    return err == ESP_OK ? release_err : err;
}

/* Read: clock the read request twice under one bus lock; the reply to the
 * first frame is shifted out during the second (pipelined). */
static esp_err_t tle_read_unlocked(tle92464_t *device, uint16_t addr16, tle92464_reply_t *reply)
{
    esp_err_t err = shared_spi_bus_acquire(SHARED_SPI_DEVICE_TLE92464);
    if (err != ESP_OK) {
        return err;
    }
    const uint32_t request = frame_read(addr16);
    err = tle_one_frame(device, request, NULL);
    if (err == ESP_OK) {
        esp_rom_delay_us(TLE_INTERFRAME_US);
    }
    uint32_t raw = 0;
    if (err == ESP_OK) {
        err = tle_one_frame(device, request, &raw);
    }
    const esp_err_t release_err = shared_spi_bus_release(SHARED_SPI_DEVICE_TLE92464);
    if (err == ESP_OK) {
        err = release_err;
    }
    if (err == ESP_OK && reply != NULL) {
        *reply = parse_reply(raw);
    }
    return err;
}

static esp_err_t tle_enter_mission_unlocked(tle92464_t *device)
{
    /* Re-assert the VIO range on every arm attempt. The chip's reset default is
     * 5 V mode (GLOBAL_CONFIG 0x4005); if the one-shot 3.3 V config write in
     * tle92464_init() was lost or raced the supply ramp, VIO_SEL stays at 5 V,
     * VIO_UV is permanently live at 3.3 V, and it blocks the Config->Mission
     * transition no matter how many times the latches are cleared (datasheet
     * Fig 4 gate: <VIO_UV>=0). Writing it here makes re-arm self-correcting. */
    (void)tle_write_unlocked(device, TLE_GLOBAL_CONFIG_WR,
                             device->vio_is_3v3 ? TLE_GLOBAL_CONFIG_3V3 : TLE_GLOBAL_CONFIG_5V);
    /* A latched UV/OV blocks the Config -> Mission transition; clear them. The
     * GLOBAL_DIAG fault latches are WRITE-0-TO-CLEAR (datasheet: POR_EVENT is
     * cleared by setting the bit to 0). Writing 0xFFFF is a no-op and leaves a
     * boot-ramp UV latch stuck, blocking Mission forever -- write 0x0000. */
    (void)tle_write_unlocked(device, TLE_GLOBAL_DIAG0_WR, 0x0000U);
    (void)tle_write_unlocked(device, TLE_GLOBAL_DIAG1_WR, 0x0000U);

    device->ch_ctrl_shadow = TLE_CH_CTRL_MISSION; /* mission, all channels off */
    esp_err_t err = tle_write_unlocked(device, TLE_CH_CTRL_WR, device->ch_ctrl_shadow);
    if (err != ESP_OK) {
        return err;
    }

    /* Confirm OP_MODE by majority vote over several readbacks. The shared MISO
     * net has no pull-up, so the CS-turnaround window occasionally corrupts the
     * reply's top (mode/CRC) byte and makes a single read reject a frame whose
     * OP_MODE data bit is actually correct (the TLE drives SO push-pull, so the
     * data bits themselves are solid). A single read therefore gives spurious
     * mission_mode=0 false-negatives; voting filters the turnaround glitches. */
    int op_mode_set = 0;
    int valid_reads = 0;
    for (int i = 0; i < 9; ++i) {
        tle92464_reply_t reply;
        if (tle_read_unlocked(device, TLE_CH_CTRL_RD, &reply) == ESP_OK &&
            reply.mode == TLE92464_REPLY_16BIT) {
            ++valid_reads;
            if ((reply.data & 0x8000U) != 0U) {
                ++op_mode_set;
            }
        }
    }
    device->mission_mode = (valid_reads > 0 && (op_mode_set * 2 > valid_reads));
    err = ESP_OK;

    if (device->en_io >= 0) {
        /* Keep EN high even when Mission is not (yet) confirmed: dropping EN
         * puts the chip in Off Mode where SPI dies, so the ~1 Hz re-arm path
         * could never recover (and outputs are already held off by EN_CHx=0
         * + SETPOINT=0, which is the real safety gate). */
        gpio_set_level(device->en_io, 1);
    }
    return err;
}

static esp_err_t tle_set_all_off_unlocked(tle92464_t *device)
{
    device->ch_ctrl_shadow &= ~0x000FU; /* clear EN_CHx, keep OP_MODE */
    esp_err_t err = tle_write_unlocked(device, TLE_CH_CTRL_WR, device->ch_ctrl_shadow);
    for (uint8_t ch = 0; ch < TLE92464_CHANNEL_COUNT; ++ch) {
        if (err == ESP_OK) {
            err = tle_write_unlocked(device, TLE_REG_SETPOINT_WR(ch), 0U);
        }
    }
    return err;
}

/* =====================================================================
 * Locking + public API
 * ===================================================================== */

static esp_err_t tle_lock(tle92464_t *device)
{
    if (device == NULL || device->spi == NULL || device->lock == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (xSemaphoreTake(device->lock, portMAX_DELAY) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

static void tle_unlock(tle92464_t *device)
{
    xSemaphoreGive(device->lock);
}

esp_err_t tle92464_write_reg(tle92464_t *device, uint8_t addr7, uint16_t data)
{
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock write");
    esp_err_t err = tle_write_unlocked(device, addr7, data);
    tle_unlock(device);
    return err;
}

esp_err_t tle92464_read_reg(tle92464_t *device, uint16_t addr16, tle92464_reply_t *reply)
{
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock read");
    esp_err_t err = tle_read_unlocked(device, addr16, reply);
    tle_unlock(device);
    return err;
}

esp_err_t tle92464_enter_mission_mode(tle92464_t *device)
{
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock mission");
    esp_err_t err = tle_enter_mission_unlocked(device);
    tle_unlock(device);
    return err;
}

esp_err_t tle92464_set_channel_code_q9(tle92464_t *device, uint8_t channel, uint16_t code_q9)
{
    if (channel >= TLE92464_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }
    const uint16_t setpoint = tle92464_code_q9_to_setpoint(code_q9);
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock setpoint");
    esp_err_t err = tle_write_unlocked(device, TLE_REG_SETPOINT_WR(channel), setpoint);
    tle_unlock(device);
    return err;
}

esp_err_t tle92464_set_channel_dither(tle92464_t *device, uint8_t channel,
                                      const tle92464_dither_t *dither)
{
    if (channel >= TLE92464_CHANNEL_COUNT || dither == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    const uint16_t clk = (uint16_t)(((uint16_t)(dither->exp & 0x0FU) << 10) |
                                    (dither->mant & 0x03FFU));
    const uint16_t step = (uint16_t)(((uint16_t)dither->steps << 8) | dither->flat);
    const uint16_t ctrl = (uint16_t)((dither->deep ? TLE_DITHER_CTRL_DEEP : 0U) |
                                     (dither->step_size & 0x0FFFU));
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock dither");
    /* CLK_DIV + STEP first; CTRL last -- writing CTRL latches the update at the
     * start of the next dither period (datasheet 4.7.2). */
    esp_err_t err = tle_write_unlocked(device, TLE_REG_DITHER_CLK_WR(channel), clk);
    if (err == ESP_OK) {
        err = tle_write_unlocked(device, TLE_REG_DITHER_STEP_WR(channel), step);
    }
    if (err == ESP_OK) {
        err = tle_write_unlocked(device, TLE_REG_DITHER_CTRL_WR(channel), ctrl);
    }
    tle_unlock(device);
    return err;
}

esp_err_t tle92464_set_channel_state(tle92464_t *device, uint8_t channel, bool enabled)
{
    if (channel >= TLE92464_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock channel state");
    if (enabled) {
        device->ch_ctrl_shadow |= (uint16_t)(1U << channel);
    } else {
        device->ch_ctrl_shadow &= (uint16_t)~(1U << channel);
    }
    esp_err_t err = tle_write_unlocked(device, TLE_CH_CTRL_WR, device->ch_ctrl_shadow);
    tle_unlock(device);
    return err;
}

esp_err_t tle92464_set_all_channels_off(tle92464_t *device)
{
    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock all off");
    esp_err_t err = tle_set_all_off_unlocked(device);
    tle_unlock(device);
    return err;
}

int tle92464_get_fault(const tle92464_t *device)
{
    if (device == NULL) {
        return -1;
    }
    if (device->fault_io < 0) {
        return 1; /* FAULTN not wired: treat as no fault */
    }
    return gpio_get_level(device->fault_io);
}

static void tle_config_gpio(gpio_num_t pin, gpio_mode_t mode, bool pull_up, int initial_level)
{
    if (pin < 0) {
        return;
    }
    if (mode == GPIO_MODE_OUTPUT) {
        gpio_set_level(pin, initial_level);
    }
    gpio_config_t config = {
        .pin_bit_mask = 1ULL << (uint32_t)pin,
        .mode = mode,
        .pull_up_en = pull_up ? GPIO_PULLUP_ENABLE : GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    (void)gpio_config(&config);
    if (mode == GPIO_MODE_OUTPUT) {
        gpio_set_level(pin, initial_level);
    }
}

esp_err_t tle92464_init(tle92464_t *device, spi_device_handle_t spi,
                        gpio_num_t en_io, gpio_num_t fault_io, gpio_num_t reset_io,
                        bool vio_is_3v3)
{
    if (device == NULL || spi == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    device->spi = spi;
    device->en_io = en_io;
    device->fault_io = fault_io;
    device->reset_io = reset_io;
    device->vio_is_3v3 = vio_is_3v3;
    device->ch_ctrl_shadow = 0;
    device->spi_alive = false;
    device->mission_mode = false;
    device->lock = xSemaphoreCreateMutex();
    if (device->lock == NULL) {
        return ESP_ERR_NO_MEM;
    }

    /* EN HIGH from the start: on the TLE92464 EN is the device enable, not an
     * output enable -- EN low is Off Mode, where SPI is dead and registers
     * reset, so nothing below would work with EN held low. (The breakout bench
     * strapped EN high on the breakout itself with en_io=NC; the ALL_IN_ONE
     * board wires EN to a GPIO with no pull, so the driver must drive it.)
     * Outputs stay safe regardless: after POR/RESN the chip is in Config Mode
     * with every EN_CHx=0 and SETPOINT=0. RESETN released high (board also
     * pulls it up), FAULTN input. NC pins are skipped. */
    tle_config_gpio(en_io, GPIO_MODE_OUTPUT, false, 1);
    tle_config_gpio(reset_io, GPIO_MODE_OUTPUT, false, 1);
    tle_config_gpio(fault_io, GPIO_MODE_INPUT, true, 0);

    if (reset_io >= 0) {
        gpio_set_level(reset_io, 0);
        esp_rom_delay_us(TLE_RESET_PULSE_US);
        gpio_set_level(reset_io, 1);
    }
    vTaskDelay(pdMS_TO_TICKS(TLE_POST_RESET_MS));

    ESP_RETURN_ON_ERROR(tle_lock(device), TLE_TAG, "lock init");
    esp_err_t err = ESP_OK;

    /* Prove SPI: read ICVID. A sane, non-trivial reply means mode-1/32-bit
     * framing + CRC are correct. */
    tle92464_reply_t reply;
    err = tle_read_unlocked(device, TLE92464_FB_ICVID, &reply);
    if (err == ESP_OK) {
        device->spi_alive = (reply.raw != 0x00000000UL && reply.raw != 0xFFFFFFFFUL &&
                             reply.mode != TLE92464_REPLY_CRITICAL);
    }

    if (err == ESP_OK) {
        /* Wait for chip initialization (INIT_DONE in FB_STAT). */
        for (int i = 0; i < TLE_INIT_DONE_POLL_COUNT; ++i) {
            tle92464_reply_t stat;
            if (tle_read_unlocked(device, TLE92464_FB_STAT, &stat) == ESP_OK &&
                stat.mode == TLE92464_REPLY_22BIT && (stat.data & TLE92464_FB_STAT_INIT_DONE_BIT) != 0U) {
                break;
            }
            esp_rom_delay_us(TLE_INIT_DONE_POLL_US);
        }

        /* VIO range (trap: chip boots assuming VIO=5V), CRC on, clock-WD on. */
        (void)tle_write_unlocked(device, TLE_GLOBAL_CONFIG_WR,
                                 vio_is_3v3 ? TLE_GLOBAL_CONFIG_3V3 : TLE_GLOBAL_CONFIG_5V);
        /* Write-0-to-clear the boot diagnostic latches (see enter_mission). */
        (void)tle_write_unlocked(device, TLE_GLOBAL_DIAG0_WR, 0x0000U);
        (void)tle_write_unlocked(device, TLE_GLOBAL_DIAG1_WR, 0x0000U);

        /* All channels to autonomous current control before Mission Mode. */
        for (uint8_t ch = 0; ch < TLE92464_CHANNEL_COUNT; ++ch) {
            (void)tle_write_unlocked(device, TLE_REG_MODE_WR(ch), TLE_MODE_ICC);
        }

        (void)tle_enter_mission_unlocked(device);
    }

    tle_unlock(device);
    return err;
}
