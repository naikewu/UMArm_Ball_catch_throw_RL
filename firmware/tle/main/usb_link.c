#include "usb_link.h"

#include <stdio.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_check.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define USB_TAG "usb_link"

#define USB_LINK_RX_DRIVER_BUF 1024
#define USB_LINK_TX_DRIVER_BUF 2048
#define USB_LINK_LINE_MAX 48U
#define USB_LINK_QUEUE_DEPTH 8U
/* Host is considered gone after this much RX silence; the GUIs send a 1 Hz
 * keepalive, so 5 s tolerates plenty of jitter. */
#define USB_LINK_ACTIVE_TIMEOUT_US (5LL * 1000 * 1000)

static portMUX_TYPE s_usb_lock = portMUX_INITIALIZER_UNLOCKED;
static usb_link_frame_t s_rx_queue[USB_LINK_QUEUE_DEPTH];
static uint8_t s_rx_head;
static uint8_t s_rx_tail;
static uint8_t s_rx_count;
static int64_t s_last_rx_us;
static usb_link_notify_fn_t s_notify;

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    return -1;
}

static void usb_link_queue_frame(const usb_link_frame_t *frame)
{
    portENTER_CRITICAL(&s_usb_lock);
    if (s_rx_count < USB_LINK_QUEUE_DEPTH) {
        s_rx_queue[s_rx_tail] = *frame;
        s_rx_tail = (uint8_t)((s_rx_tail + 1U) % USB_LINK_QUEUE_DEPTH);
        ++s_rx_count;
    }
    s_last_rx_us = esp_timer_get_time();
    portEXIT_CRITICAL(&s_usb_lock);

    if (s_notify != NULL) {
        s_notify();
    }
}

/* ">III:HHHH..." -> frame. Anything malformed (including log echo) is
 * silently dropped. */
static void usb_link_parse_line(const char *line, size_t length)
{
    if (length < 5U || line[0] != '>' || line[4] != ':') {
        return;
    }

    int id = 0;
    for (size_t index = 1; index <= 3U; ++index) {
        const int nibble = hex_nibble(line[index]);
        if (nibble < 0) {
            return;
        }
        id = (id << 4) | nibble;
    }

    const size_t hex_chars = length - 5U;
    if ((hex_chars % 2U) != 0U || hex_chars > 16U) {
        return;
    }

    usb_link_frame_t frame = {
        .id = (uint16_t)id,
        .dlc = (uint8_t)(hex_chars / 2U),
    };
    for (size_t index = 0; index < frame.dlc; ++index) {
        const int high = hex_nibble(line[5U + (2U * index)]);
        const int low = hex_nibble(line[6U + (2U * index)]);
        if (high < 0 || low < 0) {
            return;
        }
        frame.data[index] = (uint8_t)((high << 4) | low);
    }

    usb_link_queue_frame(&frame);
}

static void usb_link_task(void *arg)
{
    (void)arg;
    char line[USB_LINK_LINE_MAX];
    size_t line_length = 0;
    bool line_overflow = false;
    uint8_t buffer[64];

    while (true) {
        const int read = usb_serial_jtag_read_bytes(buffer, sizeof(buffer), portMAX_DELAY);
        for (int index = 0; index < read; ++index) {
            const char ch = (char)buffer[index];
            if (ch == '\n' || ch == '\r') {
                if (line_length > 0U && !line_overflow) {
                    usb_link_parse_line(line, line_length);
                }
                line_length = 0;
                line_overflow = false;
            } else if (line_length < USB_LINK_LINE_MAX) {
                line[line_length++] = ch;
            } else {
                line_overflow = true;
            }
        }
    }
}

esp_err_t usb_link_init(usb_link_notify_fn_t notify)
{
    s_notify = notify;

    /* The driver is used for interrupt-driven RX only; TX goes through
     * stdout (see usb_link_send_frame). */
    usb_serial_jtag_driver_config_t config = {
        .rx_buffer_size = USB_LINK_RX_DRIVER_BUF,
        .tx_buffer_size = USB_LINK_TX_DRIVER_BUF,
    };
    ESP_RETURN_ON_ERROR(usb_serial_jtag_driver_install(&config), USB_TAG, "install USB-Serial/JTAG driver");

    /* Parser only; the frames are processed by the CAN task at its priority. */
    if (xTaskCreatePinnedToCore(usb_link_task, "usb", 4096, NULL, 3, NULL, 0) != pdPASS) {
        (void)usb_serial_jtag_driver_uninstall();
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

bool usb_link_take_frame(usb_link_frame_t *frame)
{
    bool has_frame = false;
    portENTER_CRITICAL(&s_usb_lock);
    if (s_rx_count > 0U) {
        *frame = s_rx_queue[s_rx_head];
        s_rx_head = (uint8_t)((s_rx_head + 1U) % USB_LINK_QUEUE_DEPTH);
        --s_rx_count;
        has_frame = true;
    }
    portEXIT_CRITICAL(&s_usb_lock);
    return has_frame;
}

bool usb_link_active(void)
{
    portENTER_CRITICAL(&s_usb_lock);
    const int64_t last_rx_us = s_last_rx_us;
    portEXIT_CRITICAL(&s_usb_lock);
    return last_rx_us != 0 &&
           (esp_timer_get_time() - last_rx_us) < USB_LINK_ACTIVE_TIMEOUT_US &&
           usb_serial_jtag_is_connected();
}

void usb_link_send_frame(uint16_t can_id, const uint8_t *data, uint8_t length)
{
    if (data == NULL || length > 8U || !usb_link_active()) {
        return;
    }

    static const char hex_digits[] = "0123456789ABCDEF";
    char out[USB_LINK_LINE_MAX];
    size_t position = 0;
    /* Leading newline: even if some raw output left the console mid-line,
     * the frame still starts at a host line boundary. */
    out[position++] = '\n';
    out[position++] = '<';
    out[position++] = hex_digits[(can_id >> 8) & 0xFU];
    out[position++] = hex_digits[(can_id >> 4) & 0xFU];
    out[position++] = hex_digits[can_id & 0xFU];
    out[position++] = ':';
    for (uint8_t index = 0; index < length; ++index) {
        out[position++] = hex_digits[(data[index] >> 4) & 0xFU];
        out[position++] = hex_digits[data[index] & 0xFU];
    }
    out[position++] = '\n';

    /* Frames share the USB-Serial/JTAG console (the primary console) with the
     * logs and go out through stdout: the newlib FILE lock serializes whole
     * lines against ESP_LOG writes, so a frame can never be spliced into a
     * log line or vice versa. The console TX path is bounded (it fails fast
     * when the host stops draining), so this cannot stall the CAN task for
     * more than one flush timeout. */
    (void)fwrite(out, 1, position, stdout);
    (void)fflush(stdout);
}
