#ifndef BOARD_PINS_H_
#define BOARD_PINS_H_

#include "driver/gpio.h"

/*
 * =====================================================================
 * V2 PCB pin map (two independent SPI buses)
 * =====================================================================
 *
 * V1 had CAN (MCP25625) + MAX22200 sharing one hardware SPI bus, and the
 * LTC1864 ADC bit-banged on its own GPIOs. The V2 PCB moves the MAX22200
 * OFF the CAN bus and onto the LTC1864 bus, so:
 *
 *   - CAN bus  : MCP25625 alone on SPI2_HOST. UNCHANGED from V1 (confirmed
 *                by hardware: same MOSI/SCK/MISO/CS/INT GPIOs).
 *   - MAX/LTC  : MAX22200 + LTC1864 share SPI3_HOST. LTC1864 is now a real
 *                hardware-SPI device (no longer bit-banged). CS pins are
 *                unchanged (MAX CS = GPIO9).
 *
 * This decouples CAN traffic from valve/ADC control: core 0 owns CAN + PID,
 * core 1 owns the ADC + valve controller on the second bus. See
 * implementation_notes/v2_architecture.md. The SPI3 GPIOs below were
 * confirmed against Netlist_Schematic5_1_2026-05-29.tel and verified on
 * hardware during V2 bring-up.
 */

/* ---- CAN bus: MCP25625 on SPI2_HOST (UNCHANGED from V1) ---- */
#define PIN_SPI_CS_CAN GPIO_NUM_8
#define PIN_SPI_MOSI GPIO_NUM_35
#define PIN_SPI_SCK GPIO_NUM_36
#define PIN_SPI_MISO GPIO_NUM_37
#define PIN_MCP25625_INT GPIO_NUM_18

/* ---- MAX22200 + LTC1864 bus: SPI3_HOST ---- */
/* MAX22200 chip-select and control (unchanged from V1) */
#define PIN_SPI_CS_MOS GPIO_NUM_9
#define PIN_MAX22200_CMD GPIO_NUM_1
#define PIN_MAX22200_FAULT GPIO_NUM_2

/* SPI3 bus signal lines shared by MAX22200 + LTC1864 */
#define PIN_MAXLTC_MOSI GPIO_NUM_7  /* MAX22200 SDI; LTC1864 has no data-in */
#define PIN_MAXLTC_SCK GPIO_NUM_6   /* was V1 LTC1864 SCK */
/* Shared MISO: MAX22200 SDO + LTC1864 SDO. The LTC1864 SDO reaches this net
 * through a 74LVC1G125 buffer (U17) whose OE# is tied to CONV, so the driver
 * idles CONV high (buffer Hi-Z). Note: the V2 board has no pull-up on the
 * MAX22200 SDO, so MAX register readback does not work -- the MAX22200 is
 * operated write-only (see max22200.h). */
#define PIN_MAXLTC_MISO GPIO_NUM_4

/* LTC1864 convert/strobe line (own GPIO, not driven by the SPI peripheral) */
#define PIN_ADC_CONV GPIO_NUM_5     /* was V1 LTC1864 CONV */

/* ---- TLE92464ED proportional driver ----
 * Two hardware variants share this firmware:
 *
 * TLE_ALL_IN_ONE PCB (ProDoc_ver_TLE_2026-08-09.epro2) -- the TLE92464 is ON
 * BOARD and fully replaces the MAX22200 (which is not in the design at all;
 * the MAX22200_CMD/FAULT nets dead-end at the ESP32). Netlist:
 *   CSN    = CS_MOS net = GPIO9 (the former MAX22200 chip-select)
 *   SCK/SI/SO = shared SPI3 GPIO6/7/4; SO net has a 10k pull-up (R20), so
 *               unlike the old V2 board the shared MISO idles high
 *   RESN   = GPIO13, 3.3k pull-up (R3) -- released even if the GPIO floats
 *   FAULTN = GPIO14, 3.3k pull-up (R21) -- open-drain, idle high
 *   EN     = GPIO17, NO pull -- firmware must drive it high; EN low is chip
 *            Off Mode (SPI dead, registers reset), see tle92464.c
 *   VDD = own 5V LDO (U19 TPS7A2050), VIO = 3.3V rail, VBAT = VVL (13V)
 * GPIO11/GPIO12 route only to test pads (H19/H20) on this board.
 *
 * V2 PCB + extension breakout (the earlier bench) -- TLE on a breakout with
 * CS on GPIO11, EN/RESN strapped on the breakout itself, so EN/FAULT/RESET
 * are NC from the ESP32. Set VEMA_BOARD_TLE_ALL_IN_ONE=0 to build for it. */
#ifndef VEMA_BOARD_TLE_ALL_IN_ONE
#define VEMA_BOARD_TLE_ALL_IN_ONE 1
#endif

#if VEMA_BOARD_TLE_ALL_IN_ONE
#define PIN_SPI_CS_TLE GPIO_NUM_9 /* = CS_MOS; no MAX22200 on this board */
#define PIN_TLE92464_EN GPIO_NUM_17
#define PIN_TLE92464_FAULT GPIO_NUM_14
#define PIN_TLE92464_RESET GPIO_NUM_13
#else
#define PIN_SPI_CS_TLE GPIO_NUM_11
#ifndef PIN_TLE92464_EN
#define PIN_TLE92464_EN GPIO_NUM_NC
#endif
#ifndef PIN_TLE92464_FAULT
#define PIN_TLE92464_FAULT GPIO_NUM_NC
#endif
#ifndef PIN_TLE92464_RESET
#define PIN_TLE92464_RESET GPIO_NUM_NC
#endif
#endif

#endif
