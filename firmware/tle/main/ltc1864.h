#ifndef LTC1864_H_
#define LTC1864_H_

#include <stdint.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"

/*
 * V2: the LTC1864 is a hardware-SPI device sharing SPI3_HOST with the
 * MAX22200 (see shared_spi_bus). SCK and SDO are the shared bus clock and
 * MISO lines; CONV is its own GPIO strobe.
 *
 * IMPORTANT: the LTC1864 has no chip-select; it three-states SDO ONLY while
 * CONV is HIGH. To keep SDO off the shared MISO whenever the MAX22200 is
 * accessed, the driver idles CONV HIGH and only drives SDO during its own,
 * bus-locked read window. This assumes the V2 board wires both SDOs to one
 * MISO net and relies on CONV-gated three-stating (the default LTC1864 way to
 * share a bus). If the board instead buffers/muxes the LTC1864 SDO or routes
 * it to a separate pin, this idle-high behavior is still harmless. Confirm the
 * topology against the V2 schematic (see board_pins.h TODO).
 */
typedef struct {
    spi_device_handle_t spi; /* LTC1864 device on the shared MAX/LTC bus */
    gpio_num_t conv_io;      /* convert/strobe line, driven as a plain GPIO */
} ltc1864_config_t;

esp_err_t ltc1864_init(const ltc1864_config_t *config);
esp_err_t ltc1864_read_raw(uint16_t *raw_value);

#endif
