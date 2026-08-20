#ifndef MAINH
#define MAINH

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_system.h"

#include "can.h"
#include "mcp2515.h"

#include <inttypes.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_intr_alloc.h"
#include "esp_log.h"

#include "driver/ledc.h"
#include "esp_adc/adc_oneshot.h"


#include "sdkconfig.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"

#include "can_routine.h"


#define TAG "CAN_MODULE"

#define BROADCAST_CAN_ID 0x90


// ======================== CAN IDs and Commands ========================
extern uint32_t device_can_id;


#define CAN_ID_HOST_CTRL (device_can_id + 0x100)
#define CAN_ID_HOST_DATA (device_can_id + 0x200)
#define CAN_ID_ESP_STAT (device_can_id + 0x300)
#define CMD_START 0x01
#define CMD_END   0x02
#define CMD_SET_ID 0x05
#define CMD_GET_CAN_DIAG 0x06
#define CMD_CLEAR_CAN_DIAG 0x07
#define CMD_GET_OTA_STATUS 0x08
#define CMD_GET_FW_VERSION 0x09
#define MSG_ACK   0xAA
#define MSG_NACK  0xFF
#define MSG_CAN_DIAG0 0xD0
#define MSG_CAN_DIAG1 0xD1
#define MSG_CAN_DIAG2 0xD2
#define MSG_CAN_DIAG3 0xD3
#define MSG_OTA_STATUS 0xD4
#define MSG_FW_VERSION 0xD5

#define OTA_STATUS_ACTIVE      0x01u
#define OTA_STATUS_SEQ_ERROR   0x02u
#define OTA_STATUS_WRITE_ERROR 0x04u
#define OTA_STATUS_BAD_FRAME   0x08u

#define FW_VERSION_VARIANT_7MM 0x00u
#define FW_VERSION_VARIANT_DT  0x01u
#define FW_VERSION_CHUNK_BYTES 4u

#define CAN_COMPACT_PRESSURE_MASK 0x0FFFu
#define CAN_COMPACT_FLAGS_MASK    0x000Fu

#define CAN_RUNTIME_FIRST_ID       0x101u
#define CAN_RUNTIME_LAST_ID        0x118u
#define CAN_RUNTIME_TABLE_BASE_ID  0x091u
#define CAN_RUNTIME_TABLE_MARKER   0x80u
#define CAN_RUNTIME_TABLE_SLOTMASK 0x1Fu
#define CAN_RUNTIME_TABLE_DLC      8u
#define CAN_RUNTIME_TABLE_SLOTS    3u

#define CAN_CONTROL_ENABLE        0x01u

#define CAN_STATUS_ENABLED        0x01u
#define CAN_STATUS_OTA_ACTIVE     0x02u
#define CAN_STATUS_COMMAND_SEEN   0x04u
#define CAN_STATUS_ERROR          0x08u

#define CAN_ERR_REASON_RX_OVERFLOW 0x01u
#define CAN_ERR_REASON_TX_ERROR    0x02u
#define CAN_ERR_REASON_BUS_ERROR   0x04u
#define CAN_ERR_REASON_BAD_FRAME   0x08u
#define CAN_ERR_REASON_SEND_FAIL   0x10u
#define CAN_ERR_REASON_STARVATION  0x20u
#define CAN_ERR_REASON_WARNING     0x40u
#define CAN_ERR_REASON_MERR        0x80u
// ========================                      ========================

#define PIN_NUM_MISO 37
#define PIN_NUM_MOSI 35
#define PIN_NUM_CLK  36
#define PIN_NUM_CS   8
#define PIN_NUM_INTERRUPT 18




#define V_IN 6
#define V_OUT 7
#define LOW 0
#define HIGH 1

#define ESP_INTR_FLAG_DEFAULT 0


#define CAN_INTR_GPIO     18
#define CAN_INTR_BITMASK  (1ULL<<CAN_INTR_GPIO)

#define ADC_GPIO_PIN 	  9

// valve type 1 if device_can_id in 0x101-0x108, 0 if device_can_id in 0x109-0x118
#ifndef VALVE_TYPE
#define VALVE_TYPE 1 // 0 for clippard 7mm valves; 1 for big D series valves
#endif

#if VALVE_TYPE == 0 // for 7mm valve
	#define VALVE_PWM_FREQ_HZ 800
	#define PCTRL_FREQ 800 // PID
	#define PSLACK 5
	#define PWM_CAP 1023
	#define PWM_BOT 300  // minimum duty floor — prevents decay to zero
	#define INLET_OVERSHOOT_THRESHOLD  1
	#define OUTLET_OVERSHOOT_THRESHOLD 1
	#define FAST_APPROACH_ENABLE 0       // 1 = enable full-open on large target change, 0 = disable
	#define APPROACH_ENTRY_THRESHOLD 120
	#define APPROACH_FAR_THRESHOLD   300
	#define APPROACH_MAX_DUTY        1023
	#define MAINTAIN_BAND            15
	#define BRAKE_SETTLE_FACTOR      2
#else // for DT valve
	#define VALVE_PWM_FREQ_HZ 500
	#define PCTRL_FREQ 300 // PID
	#define PSLACK 2
	#define PWM_CAP 700
	#define PWM_BOT 300  // minimum duty floor for maintain corrections (lowered for fine control)
	#define INLET_OVERSHOOT_THRESHOLD  0
	#define OUTLET_OVERSHOOT_THRESHOLD 0

	// 2-layer controller: APPROACH → BRAKE → MAINTAIN
	#define APPROACH_ENTRY_THRESHOLD 120  // |perr| to enter approach (~2 PSI)
	#define APPROACH_FAR_THRESHOLD   300  // |perr| for max duty  (~5 PSI)
	#define APPROACH_MAX_DUTY        1023 // max hardware duty during approach
	#define MAINTAIN_BAND            15   // ±tolerance in maintain state (~0.25 PSI)
	#define BRAKE_SETTLE_FACTOR      2    // brake duration = system_delay * this
#endif

#define LEDC_1_TIMER              LEDC_TIMER_0
#define LEDC_1_MODE               LEDC_LOW_SPEED_MODE
#define LEDC_1_OUTPUT_IO          (V_IN) // Define the output GPIO
#define LEDC_1_CHANNEL            LEDC_CHANNEL_0
#define LEDC_1_DUTY_RES           LEDC_TIMER_10_BIT // Set duty resolution to 10 bits
#define LEDC_1_DUTY               (512) // Set duty to 50%.
#define LEDC_1_FREQUENCY          (VALVE_PWM_FREQ_HZ) // Frequency in Hertz. Set frequency at 400

#define LEDC_2_TIMER              LEDC_TIMER_0
#define LEDC_2_MODE               LEDC_LOW_SPEED_MODE
#define LEDC_2_OUTPUT_IO          (V_OUT) // Define the output GPIO
#define LEDC_2_CHANNEL            LEDC_CHANNEL_1
#define LEDC_2_DUTY_RES           LEDC_TIMER_10_BIT // Set duty resolution to 10 bits
#define LEDC_2_DUTY               (512) // Set duty to 50%.
#define LEDC_2_FREQUENCY          (VALVE_PWM_FREQ_HZ) // Frequency in Hertz. Set frequency at 4 kHz


#define PCAP 3000 // that's about 40   psi
#define PBOT 760  // that's about 0.1  psi


// ======================== Last-Known-Good LUT ========================
// 20-bin lookup table mapping pressure to adaptive duty cycles.
#define LUT_SIZE 30
#define LUT_PRESSURE_MIN 600
#define LUT_PRESSURE_MAX 3200
#define LUT_BIN_WIDTH ((LUT_PRESSURE_MAX - LUT_PRESSURE_MIN) / (LUT_SIZE - 1))



extern struct pressure_control_registers  * pcr;
extern struct ADC_read_registers 		  * acr;

struct pressure_control_registers {
	volatile uint16_t current_pressure;
	volatile uint16_t target_pressure;
	volatile uint16_t pending_target_pressure;
	volatile uint16_t active_target_pressure;
	volatile uint16_t latched_pressure;
	volatile uint16_t last_sync_counter;
	volatile uint16_t last_command_counter;
	volatile uint16_t pressure_slack;
	volatile uint16_t inlet_duty_cycle;
	volatile uint16_t outlet_duty_cycle;
	volatile uint8_t control_byte;
	volatile uint8_t pending_control_byte;
	volatile uint8_t active_control_byte;
	volatile uint8_t status_flags;
	volatile uint8_t last_can_error_reason;
	volatile uint8_t last_can_irq;
	volatile uint8_t last_can_eflg;
	volatile uint8_t last_can_send_error;
	volatile uint16_t can_error_count;
	volatile uint16_t can_warning_count;
	volatile uint16_t can_rx_overflow_count;
	volatile uint16_t can_tx_fail_count;
	volatile uint16_t can_tx_all_busy_count;
	volatile uint16_t can_invalid_frame_count;
	volatile uint16_t can_merr_count;
	volatile uint16_t can_errif_count;
	volatile uint16_t can_starvation_recover_count;
};

struct ADC_read_registers {
	volatile uint16_t adc_out;
	volatile uint16_t adc_RAF_result;
};


void app_main(void);
void CAN_Subroutine( void * pvParameters);
void Serial_Subroutine(void * pvParameters);
void WATCHDOG_subroutine(void * pvParameters);
bool SPI_Init(void);
void save_device_id_to_nvs(uint32_t new_id);

#endif