#include "mcp2515_min.h"

#include <string.h>

#include "esp_check.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MCP_INSTRUCTION_WRITE 0x02
#define MCP_INSTRUCTION_READ 0x03
#define MCP_INSTRUCTION_BITMOD 0x05
#define MCP_INSTRUCTION_READ_STATUS 0xA0
#define MCP_INSTRUCTION_RESET 0xC0

#define MCP_RXF0SIDH 0x00
#define MCP_RXF1SIDH 0x04
#define MCP_RXF2SIDH 0x08
#define MCP_CANSTAT 0x0E
#define MCP_CANCTRL 0x0F
#define MCP_RXF3SIDH 0x10
#define MCP_RXF4SIDH 0x14
#define MCP_RXF5SIDH 0x18
#define MCP_RXM0SIDH 0x20
#define MCP_RXM1SIDH 0x24
#define MCP_CNF3 0x28
#define MCP_CNF2 0x29
#define MCP_CNF1 0x2A
#define MCP_CANINTE 0x2B
#define MCP_CANINTF 0x2C
#define MCP_EFLG 0x2D
#define MCP_TXB0CTRL 0x30
#define MCP_TXB0SIDH 0x31
#define MCP_TXB1CTRL 0x40
#define MCP_TXB2CTRL 0x50
#define MCP_RXB0CTRL 0x60
#define MCP_RXB0SIDH 0x61
#define MCP_RXB1CTRL 0x70
#define MCP_RXB1SIDH 0x71

#define CANCTRL_REQOP 0xE0
#define CANCTRL_ABAT 0x10
#define CANCTRL_OSM 0x08
#define CANSTAT_OPMOD 0xE0
#define CANCTRL_REQOP_NORMAL 0x00
#define CANCTRL_REQOP_CONFIG 0x80
#define RXB0CTRL_BUKT 0x04
#define TXB_ABTF 0x40
#define TXB_MLOA 0x20
#define TXB_TXERR 0x10
#define TXB_TXREQ 0x08
#define TXB_EXIDE_MASK 0x08
#define DLC_MASK 0x0F
#define RTR_MASK 0x40

#define MCP_TAG "mcp2515"

static esp_err_t mcp_transmit(mcp2515_t *device, spi_transaction_t *transaction)
{
    if (device == NULL || device->spi == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    /* V2: MCP25625 owns SPI2_HOST by itself, so no cross-device bus guard is
     * needed. The ESP-IDF SPI driver serializes transactions on the device. */
    return spi_device_transmit(device->spi, transaction);
}

static esp_err_t mcp_reset(mcp2515_t *device)
{
    spi_transaction_t transaction = {
        .length = 8,
        .flags = SPI_TRANS_USE_TXDATA,
        .tx_data = {MCP_INSTRUCTION_RESET},
    };
    ESP_RETURN_ON_ERROR(mcp_transmit(device, &transaction), MCP_TAG, "reset");
    vTaskDelay(pdMS_TO_TICKS(10));
    return ESP_OK;
}

static esp_err_t mcp_read_register(mcp2515_t *device, uint8_t reg, uint8_t *value)
{
    spi_transaction_t transaction = {
        .length = 24,
        .flags = SPI_TRANS_USE_TXDATA | SPI_TRANS_USE_RXDATA,
        .tx_data = {MCP_INSTRUCTION_READ, reg, 0x00},
    };
    ESP_RETURN_ON_ERROR(mcp_transmit(device, &transaction), MCP_TAG, "read reg 0x%02x", reg);
    *value = transaction.rx_data[2];
    return ESP_OK;
}

static esp_err_t mcp_read_registers(mcp2515_t *device, uint8_t reg, uint8_t *values, uint8_t count)
{
    uint8_t tx_data[16] = {0};
    uint8_t rx_data[16] = {0};

    if (count > 14) {
        return ESP_ERR_INVALID_SIZE;
    }

    tx_data[0] = MCP_INSTRUCTION_READ;
    tx_data[1] = reg;
    spi_transaction_t transaction = {
        .length = (uint32_t)(count + 2U) * 8U,
        .tx_buffer = tx_data,
        .rx_buffer = rx_data,
    };
    ESP_RETURN_ON_ERROR(mcp_transmit(device, &transaction), MCP_TAG, "read regs 0x%02x", reg);
    memcpy(values, &rx_data[2], count);
    return ESP_OK;
}

static esp_err_t mcp_write_register(mcp2515_t *device, uint8_t reg, uint8_t value)
{
    spi_transaction_t transaction = {
        .length = 24,
        .flags = SPI_TRANS_USE_TXDATA,
        .tx_data = {MCP_INSTRUCTION_WRITE, reg, value},
    };
    return mcp_transmit(device, &transaction);
}

static esp_err_t mcp_write_registers(mcp2515_t *device, uint8_t reg, const uint8_t *values, uint8_t count)
{
    uint8_t tx_data[16] = {0};

    if (count > 14) {
        return ESP_ERR_INVALID_SIZE;
    }

    tx_data[0] = MCP_INSTRUCTION_WRITE;
    tx_data[1] = reg;
    memcpy(&tx_data[2], values, count);

    spi_transaction_t transaction = {
        .length = (uint32_t)(count + 2U) * 8U,
        .tx_buffer = tx_data,
    };
    return mcp_transmit(device, &transaction);
}

static esp_err_t mcp_modify_register(mcp2515_t *device, uint8_t reg, uint8_t mask, uint8_t data)
{
    spi_transaction_t transaction = {
        .length = 32,
        .flags = SPI_TRANS_USE_TXDATA,
        .tx_data = {MCP_INSTRUCTION_BITMOD, reg, mask, data},
    };
    return mcp_transmit(device, &transaction);
}

static esp_err_t mcp_set_mode(mcp2515_t *device, uint8_t mode)
{
    ESP_RETURN_ON_ERROR(mcp_modify_register(device, MCP_CANCTRL, CANCTRL_REQOP, mode), MCP_TAG, "set mode");

    for (int attempt = 0; attempt < 10; ++attempt) {
        uint8_t canstat = 0;
        ESP_RETURN_ON_ERROR(mcp_read_register(device, MCP_CANSTAT, &canstat), MCP_TAG, "read mode");
        if ((canstat & CANSTAT_OPMOD) == mode) {
            return ESP_OK;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    return ESP_ERR_TIMEOUT;
}

static void prepare_standard_id(uint16_t can_id, uint8_t *buffer)
{
    can_id &= 0x07FF;
    buffer[0] = (uint8_t)(can_id >> 3);
    buffer[1] = (uint8_t)((can_id & 0x07U) << 5);
    buffer[2] = 0;
    buffer[3] = 0;
}

static esp_err_t set_filter_or_mask(mcp2515_t *device, uint8_t base_reg, uint16_t can_id)
{
    uint8_t id_buffer[4];
    prepare_standard_id(can_id, id_buffer);
    return mcp_write_registers(device, base_reg, id_buffer, sizeof(id_buffer));
}

esp_err_t mcp2515_init(mcp2515_t *device, spi_device_handle_t spi)
{
    if (device == NULL || spi == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    device->spi = spi;
    ESP_RETURN_ON_ERROR(mcp_reset(device), MCP_TAG, "reset");
    return mcp2515_configure_1mbps_16mhz(device);
}

esp_err_t mcp2515_configure_1mbps_16mhz(mcp2515_t *device)
{
    return mcp2515_configure_bitrate_16mhz(device, MCP2515_BITRATE_1MBPS);
}

esp_err_t mcp2515_configure_bitrate_16mhz(mcp2515_t *device, mcp2515_bitrate_t bitrate)
{
    /* 16 MHz crystal. Same register values as the old-PCB driver, so both
     * board types sample the bus identically. */
    uint8_t cnf1 = 0x00;
    uint8_t cnf2 = 0xD0;
    uint8_t cnf3 = 0x82;
    if (bitrate == MCP2515_BITRATE_500KBPS) {
        cnf1 = 0x00;
        cnf2 = 0xF0;
        cnf3 = 0x86;
    }

    ESP_RETURN_ON_ERROR(mcp_set_mode(device, CANCTRL_REQOP_CONFIG), MCP_TAG, "config mode");

    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_CNF1, cnf1), MCP_TAG, "CNF1");
    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_CNF2, cnf2), MCP_TAG, "CNF2");
    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_CNF3, cnf3), MCP_TAG, "CNF3");

    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_RXB0CTRL, RXB0CTRL_BUKT), MCP_TAG, "RXB0CTRL");
    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_RXB1CTRL, 0x00), MCP_TAG, "RXB1CTRL");

    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXM0SIDH, 0), MCP_TAG, "mask0");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXM1SIDH, 0), MCP_TAG, "mask1");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF0SIDH, 0), MCP_TAG, "filter0");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF1SIDH, 0), MCP_TAG, "filter1");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF2SIDH, 0), MCP_TAG, "filter2");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF3SIDH, 0), MCP_TAG, "filter3");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF4SIDH, 0), MCP_TAG, "filter4");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF5SIDH, 0), MCP_TAG, "filter5");

    /* INT pin on RX only. Error flags (ERRIF/MERRF) are still set in CANINTF
     * and read + cleared during normal servicing, but they must not drive the
     * INT pin: with no CAN bus connected every one-shot TX aborts with MERRF,
     * and an error-driven INT then re-wakes the CAN task in a tight loop that
     * starves the rest of its core. */
    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_CANINTE, MCP2515_CANINTF_RX0IF | MCP2515_CANINTF_RX1IF), MCP_TAG, "CANINTE");
    /* One-shot mode OFF. It was a second line of defence against the bus-less
     * bench trap (a frame nobody ACKs retried at wire rate), but on a populated
     * bus it is wrong: every node answers the 150 Hz sync at the same instant,
     * so losing arbitration is routine, and a one-shot transmitter silently
     * drops the frame instead of retrying it. Only the lowest-ID board would
     * report reliably. The bus-less case is handled by the CAN link breaker in
     * pressure_controller.c (see CAN_LINK_DOWN_BUSY_POLLS), which is written
     * around exactly the retry-forever semantics restored here. */
    ESP_RETURN_ON_ERROR(mcp_modify_register(device, MCP_CANCTRL, CANCTRL_OSM, 0), MCP_TAG, "one-shot mode off");
    ESP_RETURN_ON_ERROR(mcp2515_clear_interrupts(device, 0xFF), MCP_TAG, "clear interrupts");
    return ESP_OK;
}

esp_err_t mcp2515_clear_rx_overflow(mcp2515_t *device)
{
    if (device == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    return mcp_modify_register(device, MCP_EFLG, MCP2515_EFLG_RX_OVERFLOW, 0);
}

esp_err_t mcp2515_configure_filter_set(mcp2515_t *device, const mcp2515_filter_set_t *filters)
{
    if (device == NULL || filters == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    static const uint8_t rxb0_regs[2] = {MCP_RXF0SIDH, MCP_RXF1SIDH};
    static const uint8_t rxb1_regs[4] = {MCP_RXF2SIDH, MCP_RXF3SIDH, MCP_RXF4SIDH, MCP_RXF5SIDH};

    ESP_RETURN_ON_ERROR(mcp_set_mode(device, CANCTRL_REQOP_CONFIG), MCP_TAG, "config mode for filter set");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXM0SIDH, 0x7FF), MCP_TAG, "mask0 exact");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXM1SIDH, 0x7FF), MCP_TAG, "mask1 exact");

    for (uint8_t i = 0; i < 2; ++i) {
        const uint16_t id = filters->rxb0[i] == MCP2515_FILTER_UNUSED ? filters->rxb0[0] : filters->rxb0[i];
        ESP_RETURN_ON_ERROR(set_filter_or_mask(device, rxb0_regs[i], id), MCP_TAG, "RXB0 filter");
    }
    for (uint8_t i = 0; i < 4; ++i) {
        const uint16_t id = filters->rxb1[i] == MCP2515_FILTER_UNUSED ? filters->rxb1[0] : filters->rxb1[i];
        ESP_RETURN_ON_ERROR(set_filter_or_mask(device, rxb1_regs[i], id), MCP_TAG, "RXB1 filter");
    }

    /* BUKT off: RXB0 must not roll over into RXB1. */
    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_RXB0CTRL, 0x00), MCP_TAG, "RXB0CTRL no rollover");
    ESP_RETURN_ON_ERROR(mcp_write_register(device, MCP_RXB1CTRL, 0x00), MCP_TAG, "RXB1CTRL");
    ESP_RETURN_ON_ERROR(mcp2515_clear_interrupts(device, 0xFF), MCP_TAG, "clear interrupts after filter set");
    return ESP_OK;
}

esp_err_t mcp2515_configure_standard_filters(mcp2515_t *device, uint16_t host_command_id, uint16_t host_data_id, uint16_t broadcast_id)
{
    if (device == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    ESP_RETURN_ON_ERROR(mcp_set_mode(device, CANCTRL_REQOP_CONFIG), MCP_TAG, "config mode for filters");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXM0SIDH, 0x7FF), MCP_TAG, "mask0 exact");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXM1SIDH, 0x7FF), MCP_TAG, "mask1 exact");

    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF0SIDH, host_command_id), MCP_TAG, "filter host command 0");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF1SIDH, broadcast_id), MCP_TAG, "filter broadcast 1");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF2SIDH, host_data_id), MCP_TAG, "filter OTA data 2");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF3SIDH, host_command_id), MCP_TAG, "filter host command 3");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF4SIDH, broadcast_id), MCP_TAG, "filter broadcast 4");
    ESP_RETURN_ON_ERROR(set_filter_or_mask(device, MCP_RXF5SIDH, host_data_id), MCP_TAG, "filter OTA data 5");
    ESP_RETURN_ON_ERROR(mcp2515_clear_interrupts(device, 0xFF), MCP_TAG, "clear interrupts after filters");
    return ESP_OK;
}

esp_err_t mcp2515_set_normal_mode(mcp2515_t *device)
{
    return mcp_set_mode(device, CANCTRL_REQOP_NORMAL);
}

esp_err_t mcp2515_try_send_standard(mcp2515_t *device, uint16_t can_id, const uint8_t *data, uint8_t dlc)
{
    if (device == NULL || data == NULL || dlc > 8) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t ctrl = 0;
    ESP_RETURN_ON_ERROR(mcp_read_register(device, MCP_TXB0CTRL, &ctrl), MCP_TAG, "TXB0CTRL");
    if ((ctrl & TXB_TXREQ) != 0) {
        return ESP_ERR_TIMEOUT;
    }

    uint8_t tx_buffer[13] = {0};
    prepare_standard_id(can_id, tx_buffer);
    tx_buffer[4] = dlc & DLC_MASK;
    memcpy(&tx_buffer[5], data, dlc);
    ESP_RETURN_ON_ERROR(mcp2515_clear_interrupts(device, MCP2515_CANINTF_TX0IF), MCP_TAG, "clear TX0IF");
    ESP_RETURN_ON_ERROR(mcp_write_registers(device, MCP_TXB0SIDH, tx_buffer, (uint8_t)(5U + dlc)), MCP_TAG, "load tx");
    return mcp_modify_register(device, MCP_TXB0CTRL, TXB_TXREQ, TXB_TXREQ);
}

esp_err_t mcp2515_send_standard(mcp2515_t *device, uint16_t can_id, const uint8_t *data, uint8_t dlc)
{
    if (device == NULL || data == NULL || dlc > 8) {
        return ESP_ERR_INVALID_ARG;
    }

    for (int attempt = 0; attempt < 4; ++attempt) {
        const esp_err_t err = mcp2515_try_send_standard(device, can_id, data, dlc);
        if (err != ESP_ERR_TIMEOUT) {
            return err;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    return ESP_ERR_TIMEOUT;
}

esp_err_t mcp2515_abort_pending_tx(mcp2515_t *device)
{
    if (device == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    static const uint8_t tx_ctrl_regs[] = {MCP_TXB0CTRL, MCP_TXB1CTRL, MCP_TXB2CTRL};

    /* ABAT requests the abort, the TXREQ bits release the buffers. ABAT is
     * always dropped again on the way out -- left set it would block every
     * later transmission -- so failures are collected instead of returned
     * early. */
    esp_err_t err = mcp_modify_register(device, MCP_CANCTRL, CANCTRL_ABAT, CANCTRL_ABAT);
    for (size_t index = 0; index < sizeof(tx_ctrl_regs); ++index) {
        const esp_err_t clear_err = mcp_modify_register(device, tx_ctrl_regs[index], TXB_TXREQ, 0x00);
        if (err == ESP_OK) {
            err = clear_err;
        }
    }
    const esp_err_t resume_err = mcp_modify_register(device, MCP_CANCTRL, CANCTRL_ABAT, 0x00);
    if (err == ESP_OK) {
        err = resume_err;
    }
    const esp_err_t flag_err = mcp2515_clear_interrupts(device, MCP2515_CANINTF_TX0IF);
    if (err == ESP_OK) {
        err = flag_err;
    }
    return err;
}

esp_err_t mcp2515_mask_error_interrupts(mcp2515_t *device)
{
    if (device == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    /* Only the error enables; RX0IE/RX1IE stay on so a bus that comes back
     * still raises INT. This is also the state configure_1mbps_16mhz() leaves
     * behind, so there is nothing to restore afterwards. */
    return mcp_modify_register(device, MCP_CANINTE, MCP2515_CANINTF_ERRIF | MCP2515_CANINTF_MERRF, 0x00);
}

esp_err_t mcp2515_tx_result(mcp2515_t *device)
{
    if (device == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t ctrl = 0;
    ESP_RETURN_ON_ERROR(mcp_read_register(device, MCP_TXB0CTRL, &ctrl), MCP_TAG, "TXB0CTRL result");
    if ((ctrl & TXB_TXREQ) != 0) {
        return ESP_ERR_TIMEOUT;
    }
    /* ABTF/MLOA/TXERR are cleared by the hardware when TXREQ is set, so they
     * describe the attempt that just finished. */
    if ((ctrl & (TXB_ABTF | TXB_MLOA | TXB_TXERR)) != 0) {
        return ESP_FAIL;
    }
    return ESP_OK;
}

esp_err_t mcp2515_read_rx_buffer(mcp2515_t *device, uint8_t buffer_index, mcp2515_frame_t *frame)
{
    if (device == NULL || frame == NULL || buffer_index > 1) {
        return ESP_ERR_INVALID_ARG;
    }

    const uint8_t base_reg = (buffer_index == 0) ? MCP_RXB0SIDH : MCP_RXB1SIDH;
    uint8_t rx_buffer[13] = {0};
    ESP_RETURN_ON_ERROR(mcp_read_registers(device, base_reg, rx_buffer, sizeof(rx_buffer)), MCP_TAG, "read rx");

    memset(frame, 0, sizeof(*frame));
    frame->extended = (rx_buffer[1] & TXB_EXIDE_MASK) != 0;
    frame->rtr = (rx_buffer[4] & RTR_MASK) != 0;
    frame->dlc = rx_buffer[4] & DLC_MASK;
    if (frame->dlc > 8) {
        frame->dlc = 8;
    }

    if (frame->extended) {
        frame->id = ((uint32_t)rx_buffer[0] << 21) |
                    ((uint32_t)(rx_buffer[1] & 0xE0) << 13) |
                    ((uint32_t)(rx_buffer[1] & 0x03) << 16) |
                    ((uint32_t)rx_buffer[2] << 8) |
                    rx_buffer[3];
    } else {
        frame->id = ((uint32_t)rx_buffer[0] << 3) | ((uint32_t)rx_buffer[1] >> 5);
    }

    memcpy(frame->data, &rx_buffer[5], frame->dlc);
    return ESP_OK;
}

esp_err_t mcp2515_clear_interrupts(mcp2515_t *device, uint8_t flags)
{
    return mcp_modify_register(device, MCP_CANINTF, flags, 0x00);
}

uint8_t mcp2515_read_status(mcp2515_t *device)
{
    spi_transaction_t transaction = {
        .length = 16,
        .flags = SPI_TRANS_USE_TXDATA | SPI_TRANS_USE_RXDATA,
        .tx_data = {MCP_INSTRUCTION_READ_STATUS, 0x00},
    };
    if (mcp_transmit(device, &transaction) != ESP_OK) {
        return 0xFF;
    }
    return transaction.rx_data[1];
}

uint8_t mcp2515_read_interrupts(mcp2515_t *device)
{
    uint8_t value = 0xFF;
    (void)mcp_read_register(device, MCP_CANINTF, &value);
    return value;
}

uint8_t mcp2515_read_error_flags(mcp2515_t *device)
{
    uint8_t value = 0xFF;
    (void)mcp_read_register(device, MCP_EFLG, &value);
    return value;
}