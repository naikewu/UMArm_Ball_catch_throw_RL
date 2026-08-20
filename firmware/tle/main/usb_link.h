#ifndef USB_LINK_H_
#define USB_LINK_H_

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

/*
 * USB control link over the native USB-Serial/JTAG port.
 *
 * Carries CAN-equivalent frames as text lines so the one-to-one USB
 * connection can drive the controller exactly like the CAN bus does:
 *
 *   host -> board:  ">III:HHHH...\n"  (III = 11-bit CAN ID in hex, then
 *                                      0..8 data bytes as hex pairs)
 *   board -> host:  "<III:HHHH...\n"
 *
 * Lines not starting with '>' are ignored by the board; the host ignores
 * lines not starting with '<' (boot/warning logs share the same port as the
 * secondary console). Received frames are queued and the registered notify
 * hook wakes the CAN task, which dispatches them through the same command
 * handler as CAN frames. Outbound frames are only written while a USB host
 * is actively talking (recent RX + port connected), so an absent host never
 * blocks or affects CAN operation.
 */

typedef struct {
    uint16_t id;
    uint8_t dlc;
    uint8_t data[8];
} usb_link_frame_t;

typedef void (*usb_link_notify_fn_t)(void);

esp_err_t usb_link_init(usb_link_notify_fn_t notify);
/* Pop one received frame; false if the queue is empty. */
bool usb_link_take_frame(usb_link_frame_t *frame);
/* True while a USB host is connected and has sent a frame recently. */
bool usb_link_active(void);
/* Mirror a device->host frame onto the USB link (no-op when inactive). */
void usb_link_send_frame(uint16_t can_id, const uint8_t *data, uint8_t length);

#endif
