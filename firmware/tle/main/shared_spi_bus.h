#ifndef SHARED_SPI_BUS_H_
#define SHARED_SPI_BUS_H_

#include "driver/spi_master.h"
#include "esp_err.h"

/*
 * V2: the shared hardware SPI bus now carries the MAX22200 solenoid driver
 * and the LTC1864 ADC (SPI3_HOST). The MCP25625 CAN controller has its own
 * dedicated bus (SPI2_HOST) and no longer uses this guard.
 *
 * This guard keeps each MAX22200 command/data register sequence atomic
 * relative to LTC1864 conversions, and enforces a small guard time when the
 * active device switches. Both devices are normally serviced by the core-B
 * tasks, but the MAX22200 is also reconfigured from the core-A command
 * handler, so the lock must be cross-core safe.
 */
typedef enum {
    SHARED_SPI_DEVICE_MAX22200 = 0,
    SHARED_SPI_DEVICE_LTC1864 = 1,
    SHARED_SPI_DEVICE_TLE92464 = 2,
} shared_spi_device_t;

/* Eagerly create the bus lock from a single context before any task that
 * uses the bus is started. Safe to call more than once. */
void shared_spi_bus_init(void);
esp_err_t shared_spi_bus_acquire(shared_spi_device_t device);
esp_err_t shared_spi_bus_release(shared_spi_device_t device);
esp_err_t shared_spi_bus_transmit(shared_spi_device_t device, spi_device_handle_t spi, spi_transaction_t *transaction);
void shared_spi_bus_delay_between_max_command_and_data(void);

#endif
