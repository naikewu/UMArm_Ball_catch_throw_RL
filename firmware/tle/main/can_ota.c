#include "can_ota.h"

#include <inttypes.h>
#include <string.h>

#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* 128 frames x 7 payload bytes = 896 B per flash write. Identical to the old
 * PCB firmware, so the same host can stream one image to both board types. */
#define CAN_OTA_FRAMES_PER_BLOCK 128U
#define CAN_OTA_DATA_PER_FRAME 7U
#define CAN_OTA_BLOCK_BYTES (CAN_OTA_FRAMES_PER_BLOCK * CAN_OTA_DATA_PER_FRAME)

#define CAN_OTA_ERROR_NONE 0U
#define CAN_OTA_ERROR_NOT_ACTIVE 1U
#define CAN_OTA_ERROR_BUSY 2U
#define CAN_OTA_ERROR_BAD_DLC 3U
#define CAN_OTA_ERROR_BAD_SEQUENCE 4U
#define CAN_OTA_ERROR_NO_PARTITION 5U
#define CAN_OTA_ERROR_BEGIN_FAILED 6U
#define CAN_OTA_ERROR_WRITE_FAILED 7U
#define CAN_OTA_ERROR_END_FAILED 8U
#define CAN_OTA_ERROR_BOOT_PARTITION_FAILED 9U
#define CAN_OTA_ERROR_ABORTED 10U

static const char *TAG = "can_ota";

static can_ota_config_t s_config;
static uint8_t s_ota_buffer[CAN_OTA_BLOCK_BYTES];
static uint16_t s_buffer_index;
static uint8_t s_expected_seq;
static bool s_ota_in_progress;
static esp_ota_handle_t s_ota_handle;
static const esp_partition_t *s_update_partition;
static uint32_t s_bytes_written;
static uint16_t s_blocks_written;
static uint8_t s_status_flags;
static uint8_t s_last_error;

static void write_u32_le(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8) & 0xFFU);
    data[2] = (uint8_t)((value >> 16) & 0xFFU);
    data[3] = (uint8_t)((value >> 24) & 0xFFU);
}

static void can_ota_send_status(uint8_t status, uint8_t expected_seq, uint8_t error_code)
{
    if (s_config.send_frame == NULL) {
        return;
    }

    uint8_t payload[8] = {
        CAN_OTA_TELEMETRY_TYPE_STATUS,
        status,
        expected_seq,
        error_code,
        0,
        0,
        0,
        0,
    };
    write_u32_le(&payload[4], s_bytes_written);
    (void)s_config.send_frame(s_config.status_can_id, payload, sizeof(payload));
}

/* =====================================================================
 * Silent core
 * ===================================================================== */

static bool can_ota_write_pending_block_silent(void)
{
    if (s_buffer_index == 0U) {
        return true;
    }

    esp_err_t err = esp_ota_write(s_ota_handle, s_ota_buffer, s_buffer_index);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_write failed: %s", esp_err_to_name(err));
        s_buffer_index = 0;
        s_expected_seq = 0;
        s_status_flags |= CAN_OTA_FLAG_WRITE_ERROR;
        s_last_error = CAN_OTA_ERROR_WRITE_FAILED;
        return false;
    }

    s_bytes_written += s_buffer_index;
    s_buffer_index = 0;
    s_expected_seq = 0;
    ++s_blocks_written;
    s_status_flags &= (uint8_t)~CAN_OTA_FLAG_WRITE_ERROR;
    return true;
}

bool can_ota_begin(void)
{
    if (s_ota_in_progress) {
        /* Restarting a session is how the host recovers a board that fell out
         * of step, so tear the old one down rather than refusing. */
        (void)esp_ota_abort(s_ota_handle);
        s_ota_in_progress = false;
    }

    if (s_config.safe_shutdown != NULL) {
        s_config.safe_shutdown();
    }

    s_update_partition = esp_ota_get_next_update_partition(NULL);
    if (s_update_partition == NULL) {
        ESP_LOGE(TAG, "No OTA update partition available");
        s_last_error = CAN_OTA_ERROR_NO_PARTITION;
        return false;
    }

    esp_err_t err = esp_ota_begin(s_update_partition, OTA_WITH_SEQUENTIAL_WRITES, &s_ota_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_begin failed: %s", esp_err_to_name(err));
        s_last_error = CAN_OTA_ERROR_BEGIN_FAILED;
        return false;
    }

    s_ota_in_progress = true;
    s_buffer_index = 0;
    s_expected_seq = 0;
    s_bytes_written = 0;
    s_blocks_written = 0;
    s_status_flags = 0;
    s_last_error = CAN_OTA_ERROR_NONE;
    ESP_LOGI(TAG, "CAN OTA started: partition=%s offset=0x%" PRIx32,
             s_update_partition->label, s_update_partition->address);
    return true;
}

bool can_ota_finish(void)
{
    if (!s_ota_in_progress) {
        s_last_error = CAN_OTA_ERROR_NOT_ACTIVE;
        return false;
    }

    if (!can_ota_write_pending_block_silent()) {
        s_ota_in_progress = false;
        (void)esp_ota_abort(s_ota_handle);
        return false;
    }

    esp_err_t err = esp_ota_end(s_ota_handle);
    s_ota_in_progress = false;
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_end failed: %s", esp_err_to_name(err));
        s_status_flags |= CAN_OTA_FLAG_WRITE_ERROR;
        s_last_error = CAN_OTA_ERROR_END_FAILED;
        return false;
    }

    err = esp_ota_set_boot_partition(s_update_partition);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_set_boot_partition failed: %s", esp_err_to_name(err));
        s_status_flags |= CAN_OTA_FLAG_WRITE_ERROR;
        s_last_error = CAN_OTA_ERROR_BOOT_PARTITION_FAILED;
        return false;
    }

    ESP_LOGI(TAG, "CAN OTA complete: bytes=%" PRIu32 " blocks=%u", s_bytes_written, s_blocks_written);
    return true;
}

void can_ota_reboot(void)
{
    /* Long enough for the ACK to clear the TX buffer and reach the host. */
    vTaskDelay(pdMS_TO_TICKS(250));
    esp_restart();
}

void can_ota_cancel(void)
{
    if (s_ota_in_progress) {
        (void)esp_ota_abort(s_ota_handle);
    }
    s_ota_in_progress = false;
    s_buffer_index = 0;
    s_expected_seq = 0;
    s_last_error = CAN_OTA_ERROR_ABORTED;
}

can_ota_feed_result_t can_ota_feed(const uint8_t *data, uint8_t dlc)
{
    if (data == NULL || dlc == 0U) {
        s_status_flags |= CAN_OTA_FLAG_BAD_FRAME;
        s_last_error = CAN_OTA_ERROR_BAD_DLC;
        return CAN_OTA_FEED_BAD_DLC;
    }

    if (!s_ota_in_progress) {
        s_last_error = CAN_OTA_ERROR_NOT_ACTIVE;
        return CAN_OTA_FEED_NOT_ACTIVE;
    }

    const uint8_t incoming_seq = data[0];
    if (incoming_seq != s_expected_seq) {
        /* Not logged: on a broadcast stream a board that missed one frame would
         * otherwise log every one of the ~127 that follow, at the priority of
         * the CAN task. The host sees the miss in the status poll instead. */
        s_status_flags |= CAN_OTA_FLAG_SEQ_ERROR;
        s_last_error = CAN_OTA_ERROR_BAD_SEQUENCE;
        return CAN_OTA_FEED_BAD_SEQUENCE;
    }

    const uint8_t payload_len = (uint8_t)(dlc - 1U);
    if ((uint32_t)s_buffer_index + payload_len > sizeof(s_ota_buffer)) {
        s_buffer_index = 0;
        s_expected_seq = 0;
        s_status_flags |= CAN_OTA_FLAG_BAD_FRAME;
        s_last_error = CAN_OTA_ERROR_WRITE_FAILED;
        return CAN_OTA_FEED_WRITE_FAILED;
    }

    memcpy(&s_ota_buffer[s_buffer_index], &data[1], payload_len);
    s_buffer_index = (uint16_t)(s_buffer_index + payload_len);
    ++s_expected_seq;
    s_status_flags &= (uint8_t)~(CAN_OTA_FLAG_SEQ_ERROR | CAN_OTA_FLAG_BAD_FRAME);

    if (s_expected_seq >= CAN_OTA_FRAMES_PER_BLOCK) {
        if (!can_ota_write_pending_block_silent()) {
            return CAN_OTA_FEED_WRITE_FAILED;
        }
        s_last_error = CAN_OTA_ERROR_NONE;
        return CAN_OTA_FEED_BLOCK_WRITTEN;
    }

    return CAN_OTA_FEED_ACCEPTED;
}

can_ota_stats_t can_ota_get_stats(void)
{
    can_ota_stats_t stats = {
        .active = s_ota_in_progress,
        .flags = (uint8_t)(s_status_flags | (s_ota_in_progress ? CAN_OTA_FLAG_ACTIVE : 0U)),
        .expected_seq = s_expected_seq,
        .buffer_index = s_buffer_index,
        .blocks_written = s_blocks_written,
        .bytes_written = s_bytes_written,
        .last_error = s_last_error,
    };
    return stats;
}

/* =====================================================================
 * Native protocol wrappers (USB bench path and the single-board CAN tools)
 * ===================================================================== */

static void can_ota_start_native(void)
{
    if (can_ota_begin()) {
        can_ota_send_status(CAN_OTA_STATUS_ACK, s_expected_seq, CAN_OTA_ERROR_NONE);
    } else {
        can_ota_send_status(CAN_OTA_STATUS_NACK, 0, s_last_error);
    }
}

static void can_ota_end_native(void)
{
    if (!s_ota_in_progress) {
        can_ota_send_status(CAN_OTA_STATUS_NACK, 0, CAN_OTA_ERROR_NOT_ACTIVE);
        return;
    }
    if (!can_ota_finish()) {
        can_ota_send_status(CAN_OTA_STATUS_NACK, 0, s_last_error);
        return;
    }
    ESP_LOGI(TAG, "rebooting into the new image");
    can_ota_send_status(CAN_OTA_STATUS_ACK, 0, CAN_OTA_ERROR_NONE);
    if (s_config.flush_tx != NULL) {
        s_config.flush_tx(); /* this call does not return from the reboot below */
    }
    can_ota_reboot();
}

static void can_ota_abort_native(void)
{
    can_ota_cancel();
    can_ota_send_status(CAN_OTA_STATUS_NACK, 0, CAN_OTA_ERROR_ABORTED);
}

void can_ota_init(const can_ota_config_t *config)
{
    if (config == NULL) {
        memset(&s_config, 0, sizeof(s_config));
    } else {
        s_config = *config;
    }
}

void can_ota_set_status_can_id(uint16_t status_can_id)
{
    s_config.status_can_id = status_can_id;
}

bool can_ota_in_progress(void)
{
    return s_ota_in_progress;
}

bool can_ota_handle_control_frame(const uint8_t *data, uint8_t dlc)
{
    if (data == NULL || dlc == 0U) {
        return false;
    }

    switch (data[0]) {
    case CAN_OTA_COMMAND_START:
        can_ota_start_native();
        return true;
    case CAN_OTA_COMMAND_END:
        can_ota_end_native();
        return true;
    case CAN_OTA_COMMAND_ABORT:
        can_ota_abort_native();
        return true;
    default:
        return false;
    }
}

bool can_ota_handle_data_frame(const uint8_t *data, uint8_t dlc)
{
    const can_ota_feed_result_t result = can_ota_feed(data, dlc);
    switch (result) {
    case CAN_OTA_FEED_ACCEPTED:
        break;
    case CAN_OTA_FEED_BLOCK_WRITTEN:
        can_ota_send_status(CAN_OTA_STATUS_ACK, s_expected_seq, CAN_OTA_ERROR_NONE);
        break;
    case CAN_OTA_FEED_NOT_ACTIVE:
        can_ota_send_status(CAN_OTA_STATUS_NACK, 0, CAN_OTA_ERROR_NOT_ACTIVE);
        break;
    case CAN_OTA_FEED_BAD_DLC:
        can_ota_send_status(CAN_OTA_STATUS_NACK, s_expected_seq, CAN_OTA_ERROR_BAD_DLC);
        break;
    case CAN_OTA_FEED_BAD_SEQUENCE:
        can_ota_send_status(CAN_OTA_STATUS_NACK, s_expected_seq, CAN_OTA_ERROR_BAD_SEQUENCE);
        break;
    case CAN_OTA_FEED_WRITE_FAILED:
        can_ota_send_status(CAN_OTA_STATUS_NACK, 0, CAN_OTA_ERROR_WRITE_FAILED);
        break;
    }
    return true;
}
