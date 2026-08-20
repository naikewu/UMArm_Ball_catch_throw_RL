#ifndef MCP2515_MIN_H_
#define MCP2515_MIN_H_

#include <stdbool.h>
#include <stdint.h>

#include "driver/spi_master.h"
#include "esp_err.h"

#define MCP2515_CANINTF_RX0IF 0x01
#define MCP2515_CANINTF_RX1IF 0x02
#define MCP2515_CANINTF_TX0IF 0x04
#define MCP2515_CANINTF_ERRIF 0x20
#define MCP2515_CANINTF_MERRF 0x80

/* EFLG bits. The split between the error mask and the warning bits matters:
 * warnings mean the controller is counting errors but still fully on the
 * bus, which is normal on a loaded bus and must not be reported as a
 * protocol error. */
#define MCP2515_EFLG_EWARN 0x01
#define MCP2515_EFLG_RXWAR 0x02
#define MCP2515_EFLG_TXWAR 0x04
#define MCP2515_EFLG_RXEP 0x08
#define MCP2515_EFLG_TXEP 0x10
#define MCP2515_EFLG_TXBO 0x20
#define MCP2515_EFLG_RX0OVR 0x40
#define MCP2515_EFLG_RX1OVR 0x80
#define MCP2515_EFLG_RX_OVERFLOW (MCP2515_EFLG_RX0OVR | MCP2515_EFLG_RX1OVR)
#define MCP2515_EFLG_WARNING (MCP2515_EFLG_EWARN | MCP2515_EFLG_RXWAR | MCP2515_EFLG_TXWAR)
#define MCP2515_EFLG_ERRORMASK (MCP2515_EFLG_RX_OVERFLOW | MCP2515_EFLG_TXBO | MCP2515_EFLG_TXEP | MCP2515_EFLG_RXEP)

typedef struct {
    uint32_t id;
    uint8_t dlc;
    uint8_t data[8];
    bool extended;
    bool rtr;
} mcp2515_frame_t;

typedef struct {
    spi_device_handle_t spi;
} mcp2515_t;

/* Bit timings for the 16 MHz crystal on this board. The register values are
 * the same ones the VNEMA_MK8_PIDPWM boards use, so a TLE board and an old
 * board sample the wire identically and can share a bus. */
typedef enum {
    MCP2515_BITRATE_1MBPS = 0,
    MCP2515_BITRATE_500KBPS = 1,
} mcp2515_bitrate_t;

/* Which IDs land in which receive buffer. RXF0/RXF1 feed RXB0, RXF2..RXF5 feed
 * RXB1, and rollover from RXB0 into RXB1 is disabled. Keeping the two
 * high-rate per-cycle IDs (own base, own runtime-table group) in RXB0 and the
 * broadcast/control IDs in RXB1 is the split validated on the 24-board bus: a
 * runtime-table frame must never occupy RXB1 immediately before the sync frame
 * arrives, which is how RX1 used to overflow on high-ID boards. An entry of
 * MCP2515_FILTER_UNUSED is aliased onto the first entry of its buffer. */
#define MCP2515_FILTER_UNUSED 0xFFFFU

typedef struct {
    uint16_t rxb0[2];
    uint16_t rxb1[4];
} mcp2515_filter_set_t;

esp_err_t mcp2515_init(mcp2515_t *device, spi_device_handle_t spi);
esp_err_t mcp2515_configure_1mbps_16mhz(mcp2515_t *device);
esp_err_t mcp2515_configure_bitrate_16mhz(mcp2515_t *device, mcp2515_bitrate_t bitrate);
esp_err_t mcp2515_configure_standard_filters(mcp2515_t *device, uint16_t host_command_id, uint16_t host_data_id, uint16_t broadcast_id);
esp_err_t mcp2515_configure_filter_set(mcp2515_t *device, const mcp2515_filter_set_t *filters);
esp_err_t mcp2515_set_normal_mode(mcp2515_t *device);
esp_err_t mcp2515_send_standard(mcp2515_t *device, uint16_t can_id, const uint8_t *data, uint8_t dlc);
esp_err_t mcp2515_try_send_standard(mcp2515_t *device, uint16_t can_id, const uint8_t *data, uint8_t dlc);
esp_err_t mcp2515_read_rx_buffer(mcp2515_t *device, uint8_t buffer_index, mcp2515_frame_t *frame);
esp_err_t mcp2515_clear_interrupts(mcp2515_t *device, uint8_t flags);
/* Drop every pending transmission (ABAT + TXREQ), so a frame nobody ever
 * acknowledges stops being retried and the TX buffers are free again. */
esp_err_t mcp2515_abort_pending_tx(mcp2515_t *device);
/* Clear ERRIE/MERRE in CANINTE, leaving the RX enables alone: error flags are
 * still set in CANINTF but can no longer drive the INT pin. */
esp_err_t mcp2515_mask_error_interrupts(mcp2515_t *device);
/* Verdict on the frame last handed to mcp2515_try_send_standard():
 * ESP_OK          - transmitted and acknowledged (buffer empty, no error bits),
 * ESP_ERR_TIMEOUT - still pending (TXREQ set),
 * ESP_FAIL        - aborted, arbitration lost or bus error. */
esp_err_t mcp2515_tx_result(mcp2515_t *device);
uint8_t mcp2515_read_status(mcp2515_t *device);
uint8_t mcp2515_read_interrupts(mcp2515_t *device);
uint8_t mcp2515_read_error_flags(mcp2515_t *device);
/* Clear EFLG.RX0OVR / RX1OVR.
 *
 * These latch and do not clear themselves. A receive buffer that has overrun
 * keeps rejecting frames until software clears its flag, and it does so
 * silently -- the controller still acknowledges frames on the wire in
 * hardware, so the transmitter sees success while the firmware sees nothing at
 * all. Recovery must never be gated on a condition the overflow itself
 * prevents. */
esp_err_t mcp2515_clear_rx_overflow(mcp2515_t *device);

#endif