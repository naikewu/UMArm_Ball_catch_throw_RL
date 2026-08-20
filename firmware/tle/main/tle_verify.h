#ifndef TLE_VERIFY_H_
#define TLE_VERIFY_H_

#include "driver/spi_master.h"

/*
 * Unpowered communication self-test for the TLE92464 extension board.
 *
 * Exercises the TLE92464 over SPI (CRC self-test, ICVID, INIT_DONE, supply
 * diagnostics, Mission-Mode attempt), re-checks the on-board LTC1864 ADC and
 * MCP25625 CAN controller after adding the TLE to the shared bus, prints a
 * structured report over the USB-Serial/JTAG console, then loops with a
 * periodic heartbeat. Never returns -- the normal pressure controller is not
 * started in verification builds. The caller must have created the SPI bus
 * and devices and initialised the LTC1864 first.
 */
void tle_verify_run(spi_device_handle_t tle_spi, spi_device_handle_t can_spi);

#endif
