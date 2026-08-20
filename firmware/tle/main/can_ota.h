#ifndef CAN_OTA_H_
#define CAN_OTA_H_

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

#define CAN_OTA_COMMAND_START 0x30U
#define CAN_OTA_COMMAND_END 0x31U
#define CAN_OTA_COMMAND_ABORT 0x32U

#define CAN_OTA_TELEMETRY_TYPE_STATUS 0x30U
#define CAN_OTA_STATUS_ACK 0xAAU
#define CAN_OTA_STATUS_NACK 0xFFU

typedef esp_err_t (*can_ota_send_frame_fn_t)(uint16_t can_id, const uint8_t *data, uint8_t dlc);
typedef void (*can_ota_safe_shutdown_fn_t)(void);
/* Put whatever send_frame queued on the wire before returning. The native
 * end-of-update path reboots from the same task that drains the queue, so
 * without this its acknowledgement never leaves. */
typedef void (*can_ota_flush_tx_fn_t)(void);

typedef struct {
    uint16_t status_can_id;
    can_ota_send_frame_fn_t send_frame;
    can_ota_safe_shutdown_fn_t safe_shutdown;
    can_ota_flush_tx_fn_t flush_tx;
} can_ota_config_t;

void can_ota_init(const can_ota_config_t *config);
void can_ota_set_status_can_id(uint16_t status_can_id);
bool can_ota_in_progress(void);
bool can_ota_handle_control_frame(const uint8_t *data, uint8_t dlc);
bool can_ota_handle_data_frame(const uint8_t *data, uint8_t dlc);

/* ---------------------------------------------------------------------
 * Silent core, used by the legacy (old-PCB compatible) protocol layer.
 *
 * The functions above answer in this project's native telemetry format. The
 * shared 24-board bus speaks the old protocol instead, where the same session
 * is driven with CMD_START/CMD_END on base+0x100, data broadcast on 0x090 with
 * no per-frame reply at all, and progress read back with an explicit
 * CMD_GET_OTA_STATUS poll. So the primitives below do the work and report by
 * return value; the caller owns every frame that goes on the wire.
 * --------------------------------------------------------------------- */

typedef enum {
    CAN_OTA_FEED_ACCEPTED = 0,   /* frame stored, block still filling */
    CAN_OTA_FEED_BLOCK_WRITTEN,  /* frame completed a block and it flashed OK */
    CAN_OTA_FEED_NOT_ACTIVE,
    CAN_OTA_FEED_BAD_DLC,
    CAN_OTA_FEED_BAD_SEQUENCE,
    CAN_OTA_FEED_WRITE_FAILED,
} can_ota_feed_result_t;

/* Mirrors the old firmware's OTA status flag byte. */
#define CAN_OTA_FLAG_ACTIVE 0x01U
#define CAN_OTA_FLAG_SEQ_ERROR 0x02U
#define CAN_OTA_FLAG_WRITE_ERROR 0x04U
#define CAN_OTA_FLAG_BAD_FRAME 0x08U

typedef struct {
    bool active;
    uint8_t flags;
    uint8_t expected_seq;
    uint16_t buffer_index;
    uint16_t blocks_written;
    uint32_t bytes_written;
    uint8_t last_error;
} can_ota_stats_t;

/* Begin a session (runs safe_shutdown first). False if no partition / begin failed. */
bool can_ota_begin(void);
/* Flush the partial block, validate, select the new boot slot. On success this
 * does NOT return: the caller's ACK must be sent first, then call
 * can_ota_reboot(). Returns false and leaves the session closed on failure. */
bool can_ota_finish(void);
void can_ota_reboot(void);
void can_ota_cancel(void);
can_ota_feed_result_t can_ota_feed(const uint8_t *data, uint8_t dlc);
can_ota_stats_t can_ota_get_stats(void);

#endif
