#ifndef TLE92464_H_
#define TLE92464_H_

#include <stdbool.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

/*
 * Driver for the Infineon TLE92464ED quad-channel low-side proportional
 * solenoid driver, on the V2 PCB's TLE92464 extension breakout.
 *
 * SPI: mode 1 (CPOL=0, CPHA=1), MSB first, exactly 32 clocks per CS-low
 *      window, <= 8 MHz (this driver runs at 1 MHz to match the shared bus).
 *      Every frame carries a CRC-8 SAE-J1850 in bits [31:24]. Replies are
 *      pipelined: the answer to frame N is shifted out during frame N+1, so a
 *      register read clocks the read request twice and parses the second
 *      frame's MISO. (Protocol per the TLE92464_Research MCU guide and
 *      datasheet Rev 1.2 chapter 5; verified bit-for-bit against the guide's
 *      worked example frames -- see tle92464_crc_selftest().)
 *
 * Unlike the write-only MAX22200 on this board, the TLE drives SO push-pull,
 * so register readback works and is used for bring-up and diagnostics.
 *
 * Bring-up requires VBAT in 6-18 V: the Config -> Mission transition is gated
 * on <VBAT_UV>=0 (datasheet Figure 4). With VBAT unconnected the chip still
 * powers its digital core/SPI from VDD and reaches Config Mode, but cannot
 * enter Mission Mode -- tle92464_init() reports this via mission_mode.
 *
 * Outputs additionally require the EN pin high. If PIN_TLE92464_EN is
 * GPIO_NUM_NC (not wired) the driver never asserts EN and the outputs stay in
 * their safe disabled state regardless of register state.
 */

#define TLE92464_CHANNEL_COUNT 4U

/* Abstract current "code" space shared with the controller and the MAX22200
 * driver: 0..127, optionally carrying TLE92464_CODE_FRAC_BITS of fraction.
 * The TLE maps a code onto its 15-bit SETPOINT at full resolution, so the
 * fractional part is honoured directly (no dither needed).
 *
 * Current scale (2026-06-19): code 127 maps to ~200 mA, the Clippard valve's
 * mechanical full-open / max rating (it first cracks ~60 mA). The chip's own
 * range is fixed (Iset = 2 A * SETPOINT / 0x7FFF, ~61 uA/LSB, saturating at
 * 0x6000 = 1.5 A -- there is NO programmable current-range register), so the
 * code space is scaled in software to span the valve's 0..200 mA instead of
 * 0..1.5 A. This (a) caps current at the valve rating regardless of code, and
 * (b) spreads the controllable "crack" band (~60..110 mA) across ~30 integer
 * codes instead of 3, so the controller works in a robust regime. Note: this
 * does NOT change the underlying 61 uA setpoint LSB (~0.078 psi/LSB on the
 * steep crack curve, beaten to ~0.002 psi by natural setpoint dithering); a
 * genuine resolution gain needs an inlet restrictor / lower supply pressure to
 * flatten dP/dI. SETPOINT_SAT = round(0.200 A * 0x7FFF / 2 A) = 0x0CCD. */
#define TLE92464_CODE_MAX 0x7FU
#define TLE92464_CODE_FRAC_BITS 9U
#define TLE92464_SETPOINT_SAT 0x0CCDU /* code 127 -> ~200 mA (valve max); chip LSB ~61 uA */

typedef enum {
    TLE92464_REPLY_16BIT = 0,   /* status in [21:17], data in [15:0] */
    TLE92464_REPLY_22BIT = 1,   /* wide feedback data in [21:0] */
    TLE92464_REPLY_CRITICAL = 2, /* chip in Critical Fault state */
} tle92464_reply_mode_t;

typedef struct {
    tle92464_reply_mode_t mode;
    uint8_t status; /* 16-bit replies: 0 ok, 1 frame err, 2 CRC err, 3 RO, >=4 bus fault */
    uint32_t data;
    uint32_t raw; /* the full 32-bit MISO word, for logging */
} tle92464_reply_t;

typedef struct {
    spi_device_handle_t spi;
    gpio_num_t en_io;    /* GPIO_NUM_NC = not wired */
    gpio_num_t fault_io; /* GPIO_NUM_NC = not wired */
    gpio_num_t reset_io; /* GPIO_NUM_NC = not wired */
    SemaphoreHandle_t lock;
    uint16_t ch_ctrl_shadow; /* OP_MODE (bit15) + EN_CHx (bits3:0) */
    bool vio_is_3v3;
    bool spi_alive;    /* a sane, non-trivial reply was seen during init */
    bool mission_mode; /* OP_MODE read back as 1 after the Mission request */
} tle92464_t;

/* Bring up the chip: optional RESETN pulse, prove SPI (ICVID), wait
 * INIT_DONE, set VIO range, clear boot diagnostics, set MODE=ICC on the valve
 * channels, request Mission Mode and (if reached and EN is wired) raise EN.
 * Returns ESP_OK whenever SPI is alive, even if Mission Mode is not reached
 * (e.g. VBAT absent); inspect device->mission_mode for the latter. Returns an
 * error only when SPI itself does not respond. */
esp_err_t tle92464_init(tle92464_t *device, spi_device_handle_t spi,
                        gpio_num_t en_io, gpio_num_t fault_io, gpio_num_t reset_io,
                        bool vio_is_3v3);

/* Low-level register access (also used by the bring-up verification). */
esp_err_t tle92464_write_reg(tle92464_t *device, uint8_t addr7, uint16_t data);
esp_err_t tle92464_read_reg(tle92464_t *device, uint16_t addr16, tle92464_reply_t *reply);

/* Re-run the Config -> Mission transition (clears diagnostics first). Useful
 * after VBAT is connected without a full reboot. Updates device->mission_mode. */
esp_err_t tle92464_enter_mission_mode(tle92464_t *device);

/* Set the current setpoint for a channel from the abstract Q9 code. The valve
 * channel must already be in ICC mode (done by tle92464_init). */
esp_err_t tle92464_set_channel_code_q9(tle92464_t *device, uint8_t channel, uint16_t code_q9);

/*
 * Hardware dither (per ICC channel; datasheet 4.7 + register map 5.3.3.7-9).
 * The TLE overlays a triangular/trapezoidal current waveform on the DC setpoint
 * to break the solenoid's static friction / magnetic hysteresis, so the average
 * flow responds more linearly and repeatably to a small DC change -- i.e. it
 * widens the usable, monotonic control range of a proportional valve. This is a
 * different mechanism from the (bypassed) MAX22200 software dither, which only
 * recovered current-code quantisation; hardware dither is mechanical.
 *
 *   I_dither(peak) = <STEPS> * <STEP_SIZE> * 2 A / (2^15 - 1)   (61.04 uA / unit)
 *   T_dither       = (4*<STEPS> + 2*<FLAT>) * t_ref
 *   t_ref          = <MANT> * 2^<EXP> / fSYS         (fSYS = 28 MHz typ)
 *
 * Set step_size = 0 (or steps = 0) to disable the overlay. The overlay is also
 * auto-disabled by hardware whenever the setpoint is 0 (channel coasting/off).
 */
typedef struct {
    uint8_t steps;       /* DITHER_STEP[15:8]: steps per quarter dither period */
    uint8_t flat;        /* DITHER_STEP[7:0]:  flat t_ref clocks at each plateau */
    uint16_t step_size;  /* DITHER_CTRL[11:0]: amplitude per step (setpoint LSBs) */
    uint16_t mant;       /* DITHER_CLK_DIV[9:0]:  reference-clock mantissa */
    uint8_t exp;         /* DITHER_CLK_DIV[13:10]: reference-clock exponent */
    bool deep;           /* DITHER_CTRL[13]: deep-dither (autolimit for steep overlays) */
} tle92464_dither_t;

/* Program the per-channel dither registers (CLK_DIV, STEP, then CTRL last, which
 * latches the update at the next dither period). Channel must be in ICC mode. */
esp_err_t tle92464_set_channel_dither(tle92464_t *device, uint8_t channel,
                                      const tle92464_dither_t *dither);

/* Default dither for the Clippard inlet valve: ~150 Hz, ~5.9 mA peak.
 *   I = 8*12 * 61.04 uA = 5.86 mA ; T = 32 * (365*2^4 / 28 MHz) = 6.68 ms. */
#define TLE92464_DITHER_DEFAULT_STEPS 8U
#define TLE92464_DITHER_DEFAULT_FLAT 0U
#define TLE92464_DITHER_DEFAULT_STEP_SIZE 12U
#define TLE92464_DITHER_DEFAULT_MANT 365U
#define TLE92464_DITHER_DEFAULT_EXP 4U

/* Enable/disable a channel (sets/clears its EN_CH bit; requires Mission Mode
 * and EN high to actually energise). */
esp_err_t tle92464_set_channel_state(tle92464_t *device, uint8_t channel, bool enabled);

/* Clear all EN_CH bits (keeps OP_MODE) and zero the setpoints. */
esp_err_t tle92464_set_all_channels_off(tle92464_t *device);

/* FAULTN pin: 1 = no fault (high or pin not wired), 0 = fault asserted. */
int tle92464_get_fault(const tle92464_t *device);

/* Convert an abstract Q9 code to the 15-bit SETPOINT value. */
uint16_t tle92464_code_q9_to_setpoint(uint16_t code_q9);

/* Self-test the CRC + frame builders against the datasheet worked examples.
 * Returns true if every generated frame matches; no hardware needed. */
bool tle92464_crc_selftest(void);

/* Read addresses (16-bit read frames) for bring-up diagnostics. */
#define TLE92464_FB_ICVID 0x0200U
#define TLE92464_FB_STAT 0x0202U
#define TLE92464_FB_VOLTAGE1 0x0203U /* VDD bits21:11, VIO bits10:0; V = 0.0034534 * code */
#define TLE92464_FB_VOLTAGE2 0x0204U /* VBAT bits21:11; V = 41.47 * code / 2047 */
#define TLE92464_GLOBAL_DIAG0_RD 0x0003U
#define TLE92464_FB_STAT_INIT_DONE_BIT (1UL << 21)
#define TLE92464_FB_STAT_SUP_NOK_EXT_BIT (1UL << 12)
/* GLOBAL_DIAG0 bit fields (16-bit register). */
#define TLE92464_DIAG0_VBAT_UV (1U << 0)
#define TLE92464_DIAG0_VBAT_OV (1U << 1)
#define TLE92464_DIAG0_VIO_UV (1U << 2)
#define TLE92464_DIAG0_VIO_OV (1U << 3)
#define TLE92464_DIAG0_VDD_UV (1U << 4)
#define TLE92464_DIAG0_VDD_OV (1U << 5)
#define TLE92464_DIAG0_POR_EVENT (1U << 10)

#endif
