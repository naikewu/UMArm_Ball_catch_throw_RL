#include "tle_can_legacy.h"

#include <string.h>

#include "can_ota.h"
#include "esp_app_desc.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "can_legacy";

static tle_can_legacy_hooks_t s_hooks;
static uint16_t s_base_id;
static uint16_t s_runtime_table_id;
static uint16_t s_ctrl_id;
static uint16_t s_data_id;
static uint16_t s_status_id;

/* Staged by a command frame, promoted by the next sync edge. */
static uint16_t s_pending_target_raw;
static uint8_t s_pending_control;
static uint16_t s_active_target_raw;
static uint8_t s_active_control;
static bool s_enabled;
/* Filtered pressure as it stood at the last sync edge. Replies carry this, not
 * the value at transmit time, so one host cycle is one logical instant. */
static uint16_t s_latched_raw;

static bool s_broadcast_command_pending;
static bool s_ota_session;
static bool s_command_seen;
static bool s_error_sticky;

static int64_t s_last_sync_us;
static bool s_sync_seen_ever;
static int64_t s_last_rearm_us;
static uint16_t s_rearm_count;

static uint16_t s_sync_counter;
static uint16_t s_command_counter;
static uint16_t s_rx_overflow_count;
static uint16_t s_tx_fail_count;
static uint16_t s_invalid_frame_count;
static uint16_t s_bus_error_count;
static uint16_t s_sync_loss_count;
static uint8_t s_last_eflg;

static void put_u16_le(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8) & 0xFFU);
}

static uint16_t get_u16_le(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static esp_err_t send(uint16_t can_id, const uint8_t *data, uint8_t dlc)
{
    if (s_hooks.send_frame == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    const esp_err_t err = s_hooks.send_frame(can_id, data, dlc);
    if (err != ESP_OK) {
        /* The frame never reached the wire, so whatever build_status_flags
         * consumed while composing it is lost. Re-arm the sticky error so
         * the host still learns something went wrong. */
        ++s_tx_fail_count;
        s_error_sticky = true;
    }
    return err;
}

uint16_t tle_can_legacy_runtime_table_id(uint16_t base_id)
{
    if (base_id < TLE_LEGACY_RUNTIME_FIRST_ID || base_id > TLE_LEGACY_RUNTIME_LAST_ID) {
        return 0;
    }
    const uint16_t slot = (uint16_t)(base_id - TLE_LEGACY_RUNTIME_FIRST_ID);
    return (uint16_t)(TLE_LEGACY_RUNTIME_TABLE_BASE_ID + (slot / TLE_LEGACY_RUNTIME_TABLE_SLOTS));
}

void tle_can_legacy_set_base_id(uint16_t base_id)
{
    s_base_id = base_id;
    s_runtime_table_id = tle_can_legacy_runtime_table_id(base_id);
    s_ctrl_id = (uint16_t)(base_id + TLE_LEGACY_HOST_CTRL_OFFSET);
    s_data_id = (uint16_t)(base_id + TLE_LEGACY_HOST_DATA_OFFSET);
    s_status_id = (uint16_t)(base_id + TLE_LEGACY_STATUS_OFFSET);
}

void tle_can_legacy_init(const tle_can_legacy_hooks_t *hooks, uint16_t base_id)
{
    if (hooks != NULL) {
        s_hooks = *hooks;
    } else {
        memset(&s_hooks, 0, sizeof(s_hooks));
    }
    tle_can_legacy_set_base_id(base_id);
    s_pending_target_raw = 0;
    s_pending_control = 0;
    s_active_target_raw = 0;
    s_active_control = 0;
    s_enabled = false;
    s_latched_raw = 0;
    s_broadcast_command_pending = false;
    s_ota_session = false;
    s_command_seen = false;
    s_error_sticky = false;
    s_last_sync_us = 0;
    s_sync_seen_ever = false;
    s_last_rearm_us = 0;
    s_rearm_count = 0;
}

void tle_can_legacy_note_rx_overflow(void)
{
    ++s_rx_overflow_count;
    s_error_sticky = true;
}

void tle_can_legacy_note_tx_failure(void)
{
    ++s_tx_fail_count;
    s_error_sticky = true;
}

void tle_can_legacy_note_bus_error(uint8_t eflg)
{
    ++s_bus_error_count;
    s_last_eflg = eflg;
    s_error_sticky = true;
}

bool tle_can_legacy_bus_synced(void)
{
    if (!s_sync_seen_ever) {
        return false;
    }
    return (esp_timer_get_time() - s_last_sync_us) < TLE_LEGACY_SYNC_ACTIVE_US;
}

/* ---------------------------------------------------------------------
 * Compact command / status
 * --------------------------------------------------------------------- */

static uint16_t compact_from_raw(uint16_t raw)
{
    const uint16_t counts = (uint16_t)(raw >> TLE_LEGACY_RAW_SHIFT);
    return counts > TLE_LEGACY_PRESSURE_MASK ? TLE_LEGACY_PRESSURE_MASK : counts;
}

static uint16_t raw_from_compact(uint16_t counts)
{
    return (uint16_t)((counts & TLE_LEGACY_PRESSURE_MASK) << TLE_LEGACY_RAW_SHIFT);
}

static uint8_t build_status_flags(void)
{
    uint8_t flags = 0;
    if (s_error_sticky) {
        flags |= TLE_LEGACY_STATUS_ERROR;
        s_error_sticky = false; /* sticky until read, as on the old boards */
    }
    if ((s_active_control & TLE_LEGACY_CONTROL_ENABLE) != 0U) {
        flags |= TLE_LEGACY_STATUS_ENABLED;
    }
    if (can_ota_in_progress()) {
        flags |= TLE_LEGACY_STATUS_OTA_ACTIVE;
    }
    if (s_command_seen) {
        flags |= TLE_LEGACY_STATUS_COMMAND_SEEN;
    }
    /* The host has this board enabled but its loop is not running: the ADC
     * went stale, the driver faulted, or channel setup failed. The old boards
     * have no state that can do this, so nothing in the compact reply is
     * designed to say it -- but leaving the host to believe an actuator is
     * being regulated when it is not is the worse outcome, so it is reported
     * on the one channel available. Detailed cause is on base + 0x400. */
    if (s_enabled && s_hooks.is_running != NULL && !s_hooks.is_running()) {
        flags |= TLE_LEGACY_STATUS_ERROR;
    }
    return flags & TLE_LEGACY_FLAGS_MASK;
}

static void send_compact_status(void)
{
    const uint16_t payload = (uint16_t)(compact_from_raw(s_latched_raw) |
                                        ((uint16_t)build_status_flags() << 12));
    uint8_t data[2];
    put_u16_le(data, payload);
    (void)send(s_base_id, data, 2);
}

static void stage_compact_command(uint16_t payload)
{
    s_pending_target_raw = raw_from_compact(payload & TLE_LEGACY_PRESSURE_MASK);
    s_pending_control = (uint8_t)((payload >> 12) & TLE_LEGACY_FLAGS_MASK);
    s_command_seen = true;
    ++s_command_counter;
}

/* The synchronised state transition. Everything that must look simultaneous
 * across boards happens here and nowhere else. */
static void handle_sync(void)
{
    s_last_sync_us = esp_timer_get_time();
    s_sync_seen_ever = true;
    ++s_sync_counter;

    if (can_ota_in_progress()) {
        return; /* outputs are already off and the loop is stopped */
    }

    s_latched_raw = s_hooks.get_filtered_raw != NULL ? s_hooks.get_filtered_raw() : 0U;
    s_active_target_raw = s_pending_target_raw;
    s_active_control = s_pending_control;

    const bool want_enabled = (s_active_control & TLE_LEGACY_CONTROL_ENABLE) != 0U;
    const bool running = (s_hooks.is_running != NULL) ? s_hooks.is_running() : true;

    if (want_enabled != s_enabled) {
        s_enabled = want_enabled;
        if (s_hooks.set_enabled != NULL) {
            s_hooks.set_enabled(want_enabled, s_active_target_raw);
        }
    } else if (want_enabled && !running) {
        /* The enable bit is a level, not an edge. The control loop can stop
         * itself from the PID task -- stale ADC feedback, a debounced driver
         * fault, an over-pressure trip -- and none of those clear the host's
         * command. Treating enable as an edge would leave the actuator dead
         * for the rest of the session while this board went on reporting
         * ENABLED, because set_target_raw is rejected by a stopped controller.
         *
         * Re-arming is rate limited: request_start reconfigures the driver
         * over SPI3 and arms the TLE92464, which is not something to attempt
         * at 150 Hz from the CAN task while a fault persists. */
        const int64_t now_us = esp_timer_get_time();
        if ((now_us - s_last_rearm_us) >= TLE_LEGACY_REARM_PERIOD_US) {
            s_last_rearm_us = now_us;
            ++s_rearm_count;
            s_error_sticky = true;
            if (s_hooks.set_enabled != NULL) {
                s_hooks.set_enabled(true, s_active_target_raw);
            }
        }
    } else if (want_enabled && s_hooks.set_target_raw != NULL) {
        s_hooks.set_target_raw(s_active_target_raw);
    }

    if (s_broadcast_command_pending) {
        s_broadcast_command_pending = false;
        send_compact_status();
    }
}

static void handle_direct_command(const mcp2515_frame_t *frame)
{
    if (can_ota_in_progress()) {
        send_compact_status();
        return;
    }

    if (frame->dlc != 2U) {
        ++s_invalid_frame_count;
        s_error_sticky = true;
        send_compact_status();
        return;
    }

    stage_compact_command(get_u16_le(frame->data));
    send_compact_status();
}

static void handle_runtime_table(const mcp2515_frame_t *frame)
{
    if (frame->dlc != TLE_LEGACY_RUNTIME_TABLE_DLC ||
        (frame->data[0] & TLE_LEGACY_RUNTIME_TABLE_MARKER) == 0U) {
        /* The marker bit is what separates a table frame from broadcast OTA
         * data, whose first byte is a sequence number of 0..127. */
        return;
    }
    if (s_runtime_table_id == 0U) {
        return;
    }

    const uint8_t start_slot = frame->data[0] & TLE_LEGACY_RUNTIME_TABLE_SLOTMASK;
    const uint8_t device_slot = (uint8_t)(s_base_id - TLE_LEGACY_RUNTIME_FIRST_ID);
    if (device_slot < start_slot) {
        return;
    }

    const uint8_t slot_offset = (uint8_t)(device_slot - start_slot);
    if (slot_offset >= TLE_LEGACY_RUNTIME_TABLE_SLOTS ||
        ((frame->data[1] >> slot_offset) & 0x01U) == 0U) {
        return;
    }

    stage_compact_command(get_u16_le(&frame->data[2U + (slot_offset * 2U)]));
    s_broadcast_command_pending = true;
}

/* ---------------------------------------------------------------------
 * Host control (base + 0x100)
 * --------------------------------------------------------------------- */

/* Drain the transmit queue now. Only needed on the paths that reboot. */
static void flush_tx(void)
{
    if (s_hooks.flush_tx != NULL) {
        s_hooks.flush_tx();
    }
}

static void send_ack(void)
{
    const uint8_t data[1] = {TLE_LEGACY_MSG_ACK};
    (void)send(s_status_id, data, 1);
}

static void send_nack(uint8_t requested_seq)
{
    const uint8_t data[2] = {TLE_LEGACY_MSG_NACK, requested_seq};
    (void)send(s_status_id, data, 2);
}

static void send_ota_status(void)
{
    const can_ota_stats_t stats = can_ota_get_stats();
    uint8_t data[8] = {TLE_LEGACY_MSG_OTA_STATUS, stats.flags, stats.expected_seq, 0, 0, 0, 0, stats.last_error};
    put_u16_le(&data[3], stats.buffer_index);
    put_u16_le(&data[5], stats.blocks_written);
    (void)send(s_status_id, data, 8);
}

static void send_firmware_version(void)
{
    const esp_app_desc_t *desc = esp_app_get_description();
    const char *version = (desc != NULL && desc->version[0] != '\0') ? desc->version : "0.0.0";
    const size_t length = strlen(version);

    uint8_t total_chunks = (uint8_t)((length + TLE_LEGACY_FW_VERSION_CHUNK_BYTES - 1U) /
                                     TLE_LEGACY_FW_VERSION_CHUNK_BYTES);
    if (total_chunks == 0U) {
        total_chunks = 1U;
    }

    for (uint8_t chunk = 0; chunk < total_chunks; ++chunk) {
        uint8_t data[8] = {TLE_LEGACY_MSG_FW_VERSION, chunk, total_chunks, TLE_LEGACY_FW_VARIANT_TLE_DVP, 0, 0, 0, 0};
        for (uint8_t i = 0; i < TLE_LEGACY_FW_VERSION_CHUNK_BYTES; ++i) {
            const size_t index = ((size_t)chunk * TLE_LEGACY_FW_VERSION_CHUNK_BYTES) + i;
            data[4 + i] = index < length ? (uint8_t)version[index] : 0U;
        }
        (void)send(s_status_id, data, 8);
    }
}

static void send_can_diag(void)
{
    uint8_t data[8];

    /* Same four-frame shape as the old boards. Counters this firmware does not
     * keep are sent as zero rather than as something invented. */
    memset(data, 0, sizeof(data));
    data[0] = TLE_LEGACY_MSG_CAN_DIAG0;
    data[1] = s_bus_error_count != 0U ? 0x04U : 0x00U; /* last reason: bus error */
    data[2] = 0;
    data[3] = s_last_eflg;
    data[4] = 0;
    data[5] = (uint8_t)(s_error_sticky ? TLE_LEGACY_STATUS_ERROR : 0U);
    put_u16_le(&data[6], s_bus_error_count);
    (void)send(s_status_id, data, 8);

    memset(data, 0, sizeof(data));
    data[0] = TLE_LEGACY_MSG_CAN_DIAG1;
    /* Bytes 1..2 are the reference's warning counter; this firmware keeps no
     * equivalent, so they stay zero rather than carrying an unrelated tally. */
    put_u16_le(&data[3], s_rx_overflow_count);
    put_u16_le(&data[5], s_tx_fail_count);
    (void)send(s_status_id, data, 8);

    memset(data, 0, sizeof(data));
    data[0] = TLE_LEGACY_MSG_CAN_DIAG2;
    put_u16_le(&data[2], s_invalid_frame_count);
    (void)send(s_status_id, data, 8);

    memset(data, 0, sizeof(data));
    data[0] = TLE_LEGACY_MSG_CAN_DIAG3;
    /* The reference reports starvation recoveries here. The analogue on this
     * board is the sync-timeout failsafe plus the loop re-arms it forced. */
    put_u16_le(&data[1], (uint16_t)(s_sync_loss_count + s_rearm_count));
    put_u16_le(&data[3], s_sync_counter);
    put_u16_le(&data[5], s_command_counter);
    data[7] = s_active_control;
    (void)send(s_status_id, data, 8);
}

static void clear_can_diag(void)
{
    s_rx_overflow_count = 0;
    s_tx_fail_count = 0;
    s_invalid_frame_count = 0;
    s_bus_error_count = 0;
    s_sync_loss_count = 0;
    s_rearm_count = 0;
    s_last_eflg = 0;
    s_error_sticky = false;
}

static void ota_start(void)
{
    if (can_ota_begin()) {
        s_ota_session = true;
        s_enabled = false;
        s_active_control = 0;
        s_pending_control = 0;
        send_ack();
    } else {
        /* can_ota_begin ran the safe shutdown before it failed, so the loop
         * is stopped and the outputs are off. Claiming otherwise would make
         * the next sync edge see no transition and never restart it. */
        s_ota_session = false;
        s_enabled = false;
        s_active_control = 0;
        s_pending_control = 0;
        send_nack(0);
    }
}

static void ota_end(void)
{
    if (!can_ota_in_progress()) {
        send_nack(0);
        return;
    }
    if (!can_ota_finish()) {
        s_ota_session = false;
        send_nack(can_ota_get_stats().expected_seq);
        return;
    }
    s_ota_session = false;
    send_ack();
    flush_tx();
    ESP_LOGI(TAG, "OTA image accepted, rebooting");
    can_ota_reboot();
}

static void handle_host_ctrl(const mcp2515_frame_t *frame)
{
    if (frame->dlc < 1U) {
        ++s_invalid_frame_count;
        s_error_sticky = true;
        return;
    }

    switch (frame->data[0]) {
    case TLE_LEGACY_CMD_START:
        ota_start();
        break;

    case TLE_LEGACY_CMD_END:
        ota_end();
        break;

    case TLE_LEGACY_CMD_SET_ID:
        /* DLC 3, big-endian ID in bytes 1..2. Note the endianness: this one
         * field is big-endian on the old protocol while every other multi-byte
         * value is little-endian. */
        if (frame->dlc != 3U) {
            ++s_invalid_frame_count;
            s_error_sticky = true;
            break;
        } else {
            const uint16_t new_id = (uint16_t)(((uint16_t)frame->data[1] << 8) | frame->data[2]);
            if (s_hooks.save_base_id != NULL && s_hooks.save_base_id(new_id)) {
                tle_can_legacy_set_base_id(new_id);
                send_ack();
                flush_tx();
                ESP_LOGI(TAG, "base ID set to 0x%03X, rebooting", new_id);
                vTaskDelay(pdMS_TO_TICKS(250));
                esp_restart();
            } else {
                send_nack(0);
            }
        }
        break;

    case TLE_LEGACY_CMD_GET_CAN_DIAG:
        send_can_diag();
        break;

    case TLE_LEGACY_CMD_CLEAR_CAN_DIAG:
        clear_can_diag();
        send_can_diag();
        break;

    case TLE_LEGACY_CMD_GET_OTA_STATUS:
        send_ota_status();
        break;

    case TLE_LEGACY_CMD_GET_FW_VERSION:
        send_firmware_version();
        break;

    default:
        ++s_invalid_frame_count;
        s_error_sticky = true;
        ESP_LOGW(TAG, "unknown host control command 0x%02X", frame->data[0]);
        break;
    }
}

/* Unicast repair data on base + 0x200: the host resends the frames one board
 * missed from the broadcast stream, and this one does answer per block. */
static void handle_unicast_ota_data(const mcp2515_frame_t *frame)
{
    switch (can_ota_feed(frame->data, frame->dlc)) {
    case CAN_OTA_FEED_ACCEPTED:
        break;
    case CAN_OTA_FEED_BLOCK_WRITTEN:
        send_ack();
        break;
    case CAN_OTA_FEED_NOT_ACTIVE:
        send_nack(0);
        break;
    case CAN_OTA_FEED_BAD_SEQUENCE:
    case CAN_OTA_FEED_BAD_DLC:
        send_nack(can_ota_get_stats().expected_seq);
        break;
    case CAN_OTA_FEED_WRITE_FAILED:
        send_nack(0);
        break;
    }
}

bool tle_can_legacy_handle_frame(const mcp2515_frame_t *frame)
{
    if (frame == NULL || frame->extended || frame->rtr) {
        return false;
    }

    /* The same OTA session can also be started or cancelled through the
     * native command set, which does not know about this flag. Follow the
     * real state so the broadcast-data and repair routes below cannot stay
     * claimed after a session has ended elsewhere. */
    if (s_ota_session && !can_ota_in_progress()) {
        s_ota_session = false;
    }

    const uint16_t id = (uint16_t)(frame->id & 0x7FFU);

    if (id == s_base_id) {
        handle_direct_command(frame);
        return true;
    }

    if (id == TLE_LEGACY_BROADCAST_ID) {
        if (frame->dlc == 0U) {
            handle_sync();
        } else if (s_ota_session) {
            /* Broadcast image data. Deliberately silent: with eight boards
             * receiving the same stream, a per-frame reply would put eight
             * frames on the bus for every one the host sends. The host reads
             * progress back with an explicit status poll instead. */
            (void)can_ota_feed(frame->data, frame->dlc);
        }
        return true;
    }

    if (s_runtime_table_id != 0U && id == s_runtime_table_id) {
        if (!can_ota_in_progress()) {
            handle_runtime_table(frame);
        }
        return true;
    }

    if (id == s_ctrl_id) {
        handle_host_ctrl(frame);
        return true;
    }

    if (id == s_data_id) {
        if (s_ota_session) {
            handle_unicast_ota_data(frame);
            return true;
        }
        if (!can_ota_in_progress()) {
            /* No session at all: answer as the old firmware does rather than
             * letting this fall through to a native-format reply the host
             * does not recognise, which would cost it a full timeout. */
            send_nack(0);
            return true;
        }
        /* A native bench session owns the transfer; let it reply. */
    }

    return false;
}

void tle_can_legacy_service(void)
{
    if (!s_sync_seen_ever || !s_enabled || can_ota_in_progress()) {
        return;
    }
    if ((esp_timer_get_time() - s_last_sync_us) < TLE_LEGACY_SYNC_TIMEOUT_US) {
        return;
    }

    /* The host stopped driving the cycle. Holding the last commanded pressure
     * would leave an actuator loaded with nobody watching it, so drop out. */
    ++s_sync_loss_count;
    s_error_sticky = true;
    s_enabled = false;
    s_active_control = 0;
    s_pending_control = 0;
    if (s_hooks.set_enabled != NULL) {
        s_hooks.set_enabled(false, s_active_target_raw);
    }
    ESP_LOGW(TAG, "sync lost for %lld ms, outputs off",
             (long long)(TLE_LEGACY_SYNC_TIMEOUT_US / 1000));
}
