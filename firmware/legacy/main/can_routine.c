#include "can_routine.h"
#include "esp_app_desc.h"

#ifndef VNEMA_FIRMWARE_VERSION
#define VNEMA_FIRMWARE_VERSION "0.2.0"
#endif

#ifndef VNEMA_FIRMWARE_VARIANT
#define VNEMA_FIRMWARE_VARIANT "DT"
#endif

volatile int  CAN_frame_count = 0;
volatile int  CAN_starv_count = 0;

struct can_frame can_frame_rx;

TaskHandle_t handle_CAN_Subroutine = NULL;
static TaskHandle_t CAN_recv_handle;

#define FRAMES_PER_BLOCK 128
#define DATA_PER_FRAME   7
// OTA State Variables
static uint8_t ota_buffer[FRAMES_PER_BLOCK * DATA_PER_FRAME];
static uint16_t buffer_index = 0;
static uint8_t expected_seq = 0;
static bool ota_in_progress = false;
static uint16_t ota_blocks_written = 0;
static uint8_t ota_error_flags = 0;
static esp_err_t ota_last_error = ESP_OK;

static esp_ota_handle_t ota_handle = 0;
static const esp_partition_t *update_partition = NULL;
static bool broadcast_command_pending = false;

void send_ack(void);
void send_nack(uint8_t req_seq);
void send_can_diag(void);
void start_ota(void);
void end_ota(void);
void process_ota_data_frame(struct can_frame *frame, bool reply_immediately);
void send_ota_status(void);
void send_firmware_version(void);
static void send_compact_status(void);

static void record_can_warning(uint8_t reason, uint8_t irq, uint8_t eflg) {
	if (pcr->can_error_count == 0) {
		pcr->last_can_error_reason = reason;
		pcr->last_can_irq = irq;
		pcr->last_can_eflg = eflg;
		pcr->last_can_send_error = ERROR_OK;
	}
	pcr->can_warning_count++;
}

static void mark_can_error(uint8_t reason, uint8_t irq, uint8_t eflg, ERROR_t send_error) {
	pcr->status_flags |= CAN_STATUS_ERROR;
	pcr->last_can_error_reason = reason;
	pcr->last_can_irq = irq;
	pcr->last_can_eflg = eflg;
	pcr->last_can_send_error = (uint8_t)send_error;
	pcr->can_error_count++;
}

static void clear_can_diagnostics(void) {
	pcr->status_flags &= ~CAN_STATUS_ERROR;
	pcr->last_can_error_reason = 0;
	pcr->last_can_irq = 0;
	pcr->last_can_eflg = 0;
	pcr->last_can_send_error = ERROR_OK;
	pcr->can_error_count = 0;
	pcr->can_warning_count = 0;
	pcr->can_rx_overflow_count = 0;
	pcr->can_tx_fail_count = 0;
	pcr->can_tx_all_busy_count = 0;
	pcr->can_invalid_frame_count = 0;
	pcr->can_merr_count = 0;
	pcr->can_errif_count = 0;
	pcr->can_starvation_recover_count = 0;
}

static inline uint16_t compact_pressure_clamp(uint16_t pressure) {
	return (pressure > CAN_COMPACT_PRESSURE_MASK) ? CAN_COMPACT_PRESSURE_MASK : pressure;
}

static inline uint16_t unpack_le16(const struct can_frame *frame) {
	return (uint16_t)frame->data[0] | ((uint16_t)frame->data[1] << 8);
}

static inline uint16_t unpack_le16_at(const struct can_frame *frame, uint8_t index) {
	return (uint16_t)frame->data[index] | ((uint16_t)frame->data[index + 1] << 8);
}

static uint32_t runtime_table_can_id(void) {
	if (device_can_id < CAN_RUNTIME_FIRST_ID || device_can_id > CAN_RUNTIME_LAST_ID) {
		return 0;
	}
	uint32_t slot = device_can_id - CAN_RUNTIME_FIRST_ID;
	return CAN_RUNTIME_TABLE_BASE_ID + (slot / CAN_RUNTIME_TABLE_SLOTS);
}

static uint8_t build_status_flags(void) {
	uint8_t flags = pcr->status_flags & CAN_STATUS_ERROR;
	pcr->status_flags &= ~CAN_STATUS_ERROR;
	if ((pcr->active_control_byte & CAN_CONTROL_ENABLE) != 0) flags |= CAN_STATUS_ENABLED;
	if (ota_in_progress) flags |= CAN_STATUS_OTA_ACTIVE;
	if (pcr->last_command_counter != 0) flags |= CAN_STATUS_COMMAND_SEEN;
	return flags & CAN_COMPACT_FLAGS_MASK;
}

static void recover_tx_buffers(void) {
	MCP2515_modifyRegister(MCP_TXB0CTRL, TXB_TXREQ, 0);
	MCP2515_modifyRegister(MCP_TXB1CTRL, TXB_TXREQ, 0);
	MCP2515_modifyRegister(MCP_TXB2CTRL, TXB_TXREQ, 0);
	MCP2515_clearTXInterrupts();
}

static void recover_mcp2515_errors(uint8_t irq) {
	uint8_t eflg = MCP2515_getErrorFlags();
	uint8_t reason = 0;
	uint8_t severe_eflg = eflg & EFLG_ERRORMASK;
	uint8_t warning_eflg = eflg & (EFLG_TXWAR | EFLG_RXWAR | EFLG_EWARN);

	if ((eflg & (EFLG_RX0OVR | EFLG_RX1OVR)) != 0) {
		reason |= CAN_ERR_REASON_RX_OVERFLOW;
		pcr->can_rx_overflow_count++;
		MCP2515_clearRXnOVRFlags();
	}
	if ((eflg & (EFLG_TXBO | EFLG_TXEP)) != 0) {
		reason |= CAN_ERR_REASON_TX_ERROR;
		recover_tx_buffers();
		MCP2515_setNormalMode();
	}
	if ((eflg & EFLG_RXEP) != 0) {
		reason |= CAN_ERR_REASON_BUS_ERROR;
	}
	if ((irq & CANINTF_MERRF) != 0) {
		reason |= CAN_ERR_REASON_MERR;
		pcr->can_merr_count++;
		MCP2515_clearMERR();
	}
	if ((irq & CANINTF_ERRIF) != 0) {
		pcr->can_errif_count++;
		MCP2515_clearERRIF();
	}
	if (reason != 0 || severe_eflg != 0) {
		mark_can_error(reason == 0 ? CAN_ERR_REASON_BUS_ERROR : reason, irq, eflg, ERROR_OK);
	} else if (warning_eflg != 0 || ((irq & CANINTF_ERRIF) != 0 && eflg != 0)) {
		record_can_warning(CAN_ERR_REASON_WARNING, irq, eflg);
	}
}

static void stage_compact_command(uint16_t target_pressure, uint8_t control_flags) {
	pcr->pending_target_pressure = compact_pressure_clamp(target_pressure);
	pcr->pending_control_byte = control_flags & CAN_COMPACT_FLAGS_MASK;
	pcr->last_command_counter++;
}

static void apply_sync_edge(void) {
	pcr->latched_pressure = compact_pressure_clamp(acr->adc_RAF_result);
	pcr->active_target_pressure = pcr->pending_target_pressure;
	pcr->active_control_byte = pcr->pending_control_byte & CAN_COMPACT_FLAGS_MASK;
	pcr->target_pressure = pcr->active_target_pressure;
	pcr->control_byte = pcr->active_control_byte;
	pcr->last_sync_counter++;
}

static void handle_sync_frame(void) {
	apply_sync_edge();
	if (broadcast_command_pending) {
		broadcast_command_pending = false;
		send_compact_status();
	}
}

static void send_compact_status(void) {
	uint16_t payload = compact_pressure_clamp(pcr->latched_pressure);
	payload |= ((uint16_t)build_status_flags() << 12);

	struct can_frame frame = {.can_id = device_can_id, .can_dlc = 2};
	frame.data[0] = (payload & 0x00FF) >> 0;
	frame.data[1] = (payload & 0xFF00) >> 8;
	ERROR_t send_result = MCP2515_sendMessageAfterCtrlCheck(&frame);
	if (send_result != ERROR_OK) {
		if (send_result == ERROR_ALLTXBUSY) {
			pcr->can_tx_all_busy_count++;
		} else {
			pcr->can_tx_fail_count++;
		}
		mark_can_error(CAN_ERR_REASON_SEND_FAIL, 0, MCP2515_getErrorFlags(), send_result);
		recover_tx_buffers();
		ERROR_t retry_result = MCP2515_sendMessageAfterCtrlCheck(&frame);
		if (retry_result != ERROR_OK) {
			if (retry_result == ERROR_ALLTXBUSY) {
				pcr->can_tx_all_busy_count++;
			} else {
				pcr->can_tx_fail_count++;
			}
			mark_can_error(CAN_ERR_REASON_SEND_FAIL, 0, MCP2515_getErrorFlags(), retry_result);
		}
	}
}

static void force_outputs_off(void) {
	if (VALVE_TYPE == 0) {
		gpio_set_level(V_IN, 0);
		gpio_set_level(V_OUT, 0);
	} else {
		ledc_set_duty(LEDC_1_MODE, LEDC_1_CHANNEL, 0);
		ledc_set_duty(LEDC_2_MODE, LEDC_2_CHANNEL, 0);
		ledc_update_duty(LEDC_1_MODE, LEDC_1_CHANNEL);
		ledc_update_duty(LEDC_2_MODE, LEDC_2_CHANNEL);
	}
}

static void handle_device_command_frame(const struct can_frame *frame) {
	if (ota_in_progress) {
		send_compact_status();
		return;
	}

	if (frame->can_dlc == 2) {
		uint16_t payload = unpack_le16(frame);
		stage_compact_command(payload & CAN_COMPACT_PRESSURE_MASK,
							  (uint8_t)((payload >> 12) & CAN_COMPACT_FLAGS_MASK));
		send_compact_status();
		return;
	}

	pcr->can_invalid_frame_count++;
	mark_can_error(CAN_ERR_REASON_BAD_FRAME, 0, MCP2515_getErrorFlags(), ERROR_OK);
	ESP_LOGW(TAG, "Ignoring compact device command with invalid DLC: %d", frame->can_dlc);
	send_compact_status();
}

static void handle_broadcast_runtime_frame(const struct can_frame *frame) {
	if (frame->can_dlc != CAN_RUNTIME_TABLE_DLC || (frame->data[0] & CAN_RUNTIME_TABLE_MARKER) == 0) {
		return;
	}
	if (device_can_id < CAN_RUNTIME_FIRST_ID || device_can_id > CAN_RUNTIME_LAST_ID) {
		return;
	}

	uint8_t start_slot = frame->data[0] & CAN_RUNTIME_TABLE_SLOTMASK;
	uint8_t device_slot = (uint8_t)(device_can_id - CAN_RUNTIME_FIRST_ID);
	if (device_slot < start_slot) {
		return;
	}

	uint8_t slot_offset = device_slot - start_slot;
	if (slot_offset >= CAN_RUNTIME_TABLE_SLOTS || ((frame->data[1] >> slot_offset) & 0x01u) == 0) {
		return;
	}

	uint8_t payload_index = 2u + (slot_offset * 2u);
	uint16_t payload = unpack_le16_at(frame, payload_index);
	stage_compact_command(payload & CAN_COMPACT_PRESSURE_MASK,
	                      (uint8_t)((payload >> 12) & CAN_COMPACT_FLAGS_MASK));
	broadcast_command_pending = true;
}

static void handle_host_ctrl_frame(const struct can_frame *frame) {
	if (frame->can_dlc < 1) {
		pcr->can_invalid_frame_count++;
		mark_can_error(CAN_ERR_REASON_BAD_FRAME, 0, MCP2515_getErrorFlags(), ERROR_OK);
		ESP_LOGW(TAG, "HOST_CTRL frame received with no command byte");
		return;
	}

	uint8_t host_cmd = frame->data[0];
	ESP_LOGI(TAG, "HOST_CTRL command received: 0x%02X", host_cmd);
	if (host_cmd == CMD_START) {
		start_ota();
	} else if (host_cmd == CMD_END && ota_in_progress) {
		end_ota();
	} else if (host_cmd == CMD_END) {
		ESP_LOGW(TAG, "Received CMD_END while OTA is not active");
	} else if (host_cmd == CMD_SET_ID) {
		if (frame->can_dlc != 3) {
			pcr->can_invalid_frame_count++;
			mark_can_error(CAN_ERR_REASON_BAD_FRAME, 0, MCP2515_getErrorFlags(), ERROR_OK);
			ESP_LOGW(TAG, "SET_ID frame received with incorrect data length (!=3: CMD_SET_ID + 2 bytes ID)");
		} else {
			uint16_t new_id = ((uint16_t)frame->data[1] << 8) | frame->data[2];
			printf("CAN_MODULE: Received CMD_SET_ID. Changing ID to: 0x%X\n", new_id);
			save_device_id_to_nvs((uint32_t)new_id);
			send_ack();
			printf("Rebooting in 1 second to apply new CAN ID...\n");
			vTaskDelay(1000 / portTICK_PERIOD_MS);
			esp_restart();
		}
	} else if (host_cmd == CMD_GET_CAN_DIAG) {
		send_can_diag();
	} else if (host_cmd == CMD_CLEAR_CAN_DIAG) {
		clear_can_diagnostics();
		send_can_diag();
	} else if (host_cmd == CMD_GET_OTA_STATUS) {
		send_ota_status();
	} else if (host_cmd == CMD_GET_FW_VERSION) {
		send_firmware_version();
	} else {
		pcr->can_invalid_frame_count++;
		mark_can_error(CAN_ERR_REASON_BAD_FRAME, 0, MCP2515_getErrorFlags(), ERROR_OK);
		ESP_LOGW(TAG, "Unknown HOST_CTRL command: 0x%02X", host_cmd);
	}
}

static void route_can_frame(struct can_frame *frame) {
	uint32_t rx_id = frame->can_id & CAN_SFF_MASK;
	if (rx_id == device_can_id) {
		handle_device_command_frame(frame);
	} else if (rx_id == BROADCAST_CAN_ID) {
		if (frame->can_dlc == 0) {
			handle_sync_frame();
		} else if (ota_in_progress) {
			process_ota_data_frame(frame, false);
		}
	} else if (!ota_in_progress && rx_id == runtime_table_can_id()) {
		handle_broadcast_runtime_frame(frame);
	} else if (rx_id == CAN_ID_HOST_CTRL) {
		handle_host_ctrl_frame(frame);
	} else if (rx_id == CAN_ID_HOST_DATA) {
		process_ota_data_frame(frame, true);
	}
}

void send_ack(void) {
    struct can_frame frame = {.can_id = CAN_ID_ESP_STAT, .can_dlc = 1};
    frame.data[0] = MSG_ACK;
    MCP2515_sendMessageAfterCtrlCheck(&frame);
    ESP_LOGI(TAG, "Sent ACK");
}

void send_nack(uint8_t req_seq) {
    struct can_frame frame = {.can_id = CAN_ID_ESP_STAT, .can_dlc = 2};
    frame.data[0] = MSG_NACK;
    frame.data[1] = req_seq;
    MCP2515_sendMessageAfterCtrlCheck(&frame);
    ESP_LOGW(TAG, "Sent NACK, requesting seq: %d", req_seq);
}

static void put_u16(uint8_t *data, int index, uint16_t value) {
	data[index] = (uint8_t)(value & 0x00FFu);
	data[index + 1] = (uint8_t)((value >> 8) & 0x00FFu);
}

void send_can_diag(void) {
	struct can_frame frame = {.can_id = CAN_ID_ESP_STAT, .can_dlc = 8};

	frame.data[0] = MSG_CAN_DIAG0;
	frame.data[1] = pcr->last_can_error_reason;
	frame.data[2] = pcr->last_can_irq;
	frame.data[3] = pcr->last_can_eflg;
	frame.data[4] = pcr->last_can_send_error;
	frame.data[5] = pcr->status_flags;
	put_u16(frame.data, 6, pcr->can_error_count);
	MCP2515_sendMessageAfterCtrlCheck(&frame);

	frame.data[0] = MSG_CAN_DIAG1;
	put_u16(frame.data, 1, pcr->can_warning_count);
	put_u16(frame.data, 3, pcr->can_rx_overflow_count);
	put_u16(frame.data, 5, pcr->can_tx_fail_count);
	frame.data[7] = (uint8_t)(pcr->can_tx_all_busy_count & 0x00FFu);
	MCP2515_sendMessageAfterCtrlCheck(&frame);

	frame.data[0] = MSG_CAN_DIAG2;
	frame.data[1] = (uint8_t)((pcr->can_tx_all_busy_count >> 8) & 0x00FFu);
	put_u16(frame.data, 2, pcr->can_invalid_frame_count);
	put_u16(frame.data, 4, pcr->can_merr_count);
	put_u16(frame.data, 6, pcr->can_errif_count);
	MCP2515_sendMessageAfterCtrlCheck(&frame);

	frame.data[0] = MSG_CAN_DIAG3;
	put_u16(frame.data, 1, pcr->can_starvation_recover_count);
	put_u16(frame.data, 3, pcr->last_sync_counter);
	put_u16(frame.data, 5, pcr->last_command_counter);
	frame.data[7] = pcr->active_control_byte;
	MCP2515_sendMessageAfterCtrlCheck(&frame);
}

void send_ota_status(void) {
	struct can_frame frame = {.can_id = CAN_ID_ESP_STAT, .can_dlc = 8};
	uint8_t flags = ota_error_flags;
	if (ota_in_progress) flags |= OTA_STATUS_ACTIVE;

	frame.data[0] = MSG_OTA_STATUS;
	frame.data[1] = flags;
	frame.data[2] = expected_seq;
	put_u16(frame.data, 3, buffer_index);
	put_u16(frame.data, 5, ota_blocks_written);
	frame.data[7] = (uint8_t)ota_last_error;
	MCP2515_sendMessageAfterCtrlCheck(&frame);
}

void send_firmware_version(void) {
	const esp_app_desc_t *app_desc = esp_app_get_description();
	const char *version = VNEMA_FIRMWARE_VERSION;
	if (app_desc != NULL && app_desc->version[0] != '\0') {
		version = app_desc->version;
	}

	size_t version_len = strlen(version);
	uint8_t total_chunks = (uint8_t)((version_len + FW_VERSION_CHUNK_BYTES - 1u) / FW_VERSION_CHUNK_BYTES);
	if (total_chunks == 0u) {
		total_chunks = 1u;
	}

	uint8_t variant = FW_VERSION_VARIANT_DT;
#if VALVE_TYPE == 0
	variant = FW_VERSION_VARIANT_7MM;
#endif

	for (uint8_t chunk = 0; chunk < total_chunks; ++chunk) {
		struct can_frame frame = {.can_id = CAN_ID_ESP_STAT, .can_dlc = 8};
		frame.data[0] = MSG_FW_VERSION;
		frame.data[1] = chunk;
		frame.data[2] = total_chunks;
		frame.data[3] = variant;
		for (uint8_t index = 0; index < FW_VERSION_CHUNK_BYTES; ++index) {
			size_t source_index = ((size_t)chunk * FW_VERSION_CHUNK_BYTES) + index;
			frame.data[4 + index] = source_index < version_len ? (uint8_t)version[source_index] : 0u;
		}
		MCP2515_sendMessageAfterCtrlCheck(&frame);
	}
}

void start_ota(void) {
    update_partition = esp_ota_get_next_update_partition(NULL);
    if (update_partition == NULL) {
        ESP_LOGE(TAG, "Could not find OTA partition");
        return;
    }
    
    ESP_LOGI(TAG, "Starting OTA. Writing to partition: %s", update_partition->label);
    esp_err_t err = esp_ota_begin(update_partition, OTA_WITH_SEQUENTIAL_WRITES, &ota_handle);
    if (err == ESP_OK) {
        ota_in_progress = true;
        buffer_index = 0;
        expected_seq = 0;
		ota_blocks_written = 0;
		ota_error_flags = 0;
		ota_last_error = ESP_OK;
		
		// SAFETY SHUTDOWN: Turn off valves while flashing
		pcr->pending_control_byte = 0;
		pcr->active_control_byte = 0;
		pcr->control_byte = 0;
		force_outputs_off();

        send_ack();
    } else {
        ESP_LOGE(TAG, "esp_ota_begin failed: %s", esp_err_to_name(err));
    }

	
}

void end_ota(void) {
    if (buffer_index > 0) {
		esp_err_t write_err = esp_ota_write(ota_handle, ota_buffer, buffer_index); // Flush remaining
		if (write_err != ESP_OK) {
			ESP_LOGE(TAG, "Final flash write failed: %s", esp_err_to_name(write_err));
			ota_error_flags |= OTA_STATUS_WRITE_ERROR;
			ota_last_error = write_err;
			send_nack(expected_seq);
			return;
		}
		buffer_index = 0;
		expected_seq = 0;
		ota_blocks_written++;
    }
    
    esp_err_t err = esp_ota_end(ota_handle);
    if (err == ESP_OK) {
        err = esp_ota_set_boot_partition(update_partition);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "OTA Success! Rebooting...");
            send_ack();
            vTaskDelay(pdMS_TO_TICKS(1000));
            esp_restart();
			return;
        }
		ESP_LOGE(TAG, "esp_ota_set_boot_partition failed: %s", esp_err_to_name(err));
		ota_error_flags |= OTA_STATUS_WRITE_ERROR;
		ota_last_error = err;
	} else {
		ESP_LOGE(TAG, "esp_ota_end failed: %s", esp_err_to_name(err));
		ota_error_flags |= OTA_STATUS_WRITE_ERROR;
		ota_last_error = err;
    }
	send_nack(expected_seq);
}

void process_ota_data_frame(struct can_frame *frame, bool reply_immediately) {
	if (!ota_in_progress) {
		if (reply_immediately) {
			ESP_LOGW(TAG, "Ignoring OTA data frame while OTA is not active");
			send_nack(0);
		}
		return;
	}

	if (frame->can_dlc < 1) {
		ESP_LOGW(TAG, "Ignoring OTA data frame with invalid DLC: %d", frame->can_dlc);
		ota_error_flags |= OTA_STATUS_BAD_FRAME;
		if (reply_immediately) send_nack(expected_seq);
		return;
	}

    uint8_t incoming_seq = frame->data[0];

    if (incoming_seq != expected_seq) {
        ESP_LOGW(TAG, "Seq mismatch. Expected %d, got %d", expected_seq, incoming_seq);
		ota_error_flags |= OTA_STATUS_SEQ_ERROR;
		if (reply_immediately) send_nack(expected_seq);
        return;
    }

	uint16_t payload_len = frame->can_dlc - 1;
	if (buffer_index + payload_len > sizeof(ota_buffer)) {
		ESP_LOGE(TAG, "OTA block overflow (idx=%u payload=%u)", buffer_index, payload_len);
		buffer_index = 0;
		expected_seq = 0;
		ota_error_flags |= OTA_STATUS_BAD_FRAME;
		if (reply_immediately) send_nack(0);
		return;
	}

	memcpy(&ota_buffer[buffer_index], &frame->data[1], payload_len);
	buffer_index += payload_len;
    expected_seq++;
	ota_error_flags &= ~(OTA_STATUS_SEQ_ERROR | OTA_STATUS_BAD_FRAME);

    if (expected_seq >= FRAMES_PER_BLOCK) {
        esp_err_t err = esp_ota_write(ota_handle, ota_buffer, buffer_index);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Flash write failed");
			ota_error_flags |= OTA_STATUS_WRITE_ERROR;
			ota_last_error = err;
			buffer_index = 0;
			expected_seq = 0;
			if (reply_immediately) send_nack(0); // Tell host to resend entire block
            return;
        }
        buffer_index = 0;
        expected_seq = 0;
        ota_blocks_written++;
        ota_error_flags &= ~OTA_STATUS_WRITE_ERROR;
        ota_last_error = ESP_OK;

		
        if (reply_immediately) send_ack();
    }
}


static void IRAM_ATTR CAN_isr_handler(void* arg) {
	BaseType_t mustYield = pdFALSE;
	vTaskNotifyGiveFromISR(CAN_recv_handle, &mustYield);
	portYIELD_FROM_ISR(mustYield);
}

void CAN_Subroutine( void * pvParameters) {
	CAN_recv_handle = xTaskGetCurrentTaskHandle();

	MCP2515_init();
	SPI_Init();
	MCP2515_reset();
	MCP2515_setBitrate(CAN_1000KBPS, MCP_16MHZ);
	

	MCP2515_setFilterMask(MASK0, false, 0x7FF);
	MCP2515_setFilter(RXF0, false, device_can_id);
	MCP2515_setFilter(RXF1, false, runtime_table_can_id());
	MCP2515_modifyRegister(MCP_RXB0CTRL, RXB0CTRL_BUKT, 0);

	MCP2515_setFilterMask(MASK1, false, 0x7FF);
	MCP2515_setFilter(RXF2, false, BROADCAST_CAN_ID);
	MCP2515_setFilter(RXF3, false, BROADCAST_CAN_ID);
	MCP2515_setFilter(RXF4, false, CAN_ID_HOST_CTRL);
	MCP2515_setFilter(RXF5, false, CAN_ID_HOST_DATA);

	MCP2515_setNormalMode();


	//install gpio isr service
    gpio_install_isr_service(0);
	//hook isr handler for specific gpio pin
	gpio_set_intr_type(CAN_INTR_GPIO, GPIO_INTR_NEGEDGE);
    gpio_isr_handler_add(CAN_INTR_GPIO, CAN_isr_handler, (void*) CAN_INTR_GPIO);

	MCP2515_clearInterrupts();

	while(1){
		// block while not receive notification
		ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

		// reset the CAN starvation counter
		CAN_starv_count = 0;

		while (1) {
			uint8_t irq = MCP2515_getInterrupts();

			recover_mcp2515_errors(irq);

			if ((irq & (CANINTF_RX0IF | CANINTF_RX1IF)) == 0) {
				break;
			}

			if (irq & CANINTF_RX0IF) {
				if (MCP2515_readMessage(RXB0, &can_frame_rx) == ERROR_OK) {
					CAN_frame_count++;
					route_can_frame(&can_frame_rx);
				}
			}

			if (irq & CANINTF_RX1IF) {
				if (MCP2515_readMessage(RXB1, &can_frame_rx) == ERROR_OK) {
					CAN_frame_count++;
					route_can_frame(&can_frame_rx);
				}
			}
		}

		if (!ota_in_progress) {
			if ((pcr->active_control_byte & CAN_CONTROL_ENABLE) == 0) {
				force_outputs_off();
			}
		}
	}
}

void CAN_recover_from_starvation(void) {
	pcr->can_starvation_recover_count++;
	record_can_warning(CAN_ERR_REASON_STARVATION, MCP2515_getInterrupts(), MCP2515_getErrorFlags());
	MCP2515_clearRXnOVR();
	recover_tx_buffers();
	MCP2515_clearMERR();
	MCP2515_clearERRIF();
	MCP2515_clearInterrupts();
	MCP2515_setNormalMode();
	if (CAN_recv_handle != NULL) {
		xTaskNotifyGive(CAN_recv_handle);
	}
}