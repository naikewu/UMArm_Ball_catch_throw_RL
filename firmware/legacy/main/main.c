#include "main.h"

#define ACT_NONE   0
#define ACT_INLET  1
#define ACT_OUTLET 2
#define CONTROL_HISTORY_LEN 16
#define BLAME_WINDOW_BACKWARD 2
#define BLAME_WINDOW_FORWARD  2

typedef struct {
	int pressure;
	int raw;
	int target;
	int action;
	int duty;
	int prev_same_action;
	int active_ticks;
} control_history_entry_t;

// DT control bands

uint32_t device_can_id = 0x114; // Default fallback if NVS is empty

TaskHandle_t handle_Serial_Subroutine = NULL;
TaskHandle_t handle_WATCHDOG_subroutine = NULL;


struct pressure_control_registers * pcr = NULL;
struct ADC_read_registers 		  * acr = NULL;

// Adaptive LUTs: on->on and off->on cases for inlet/outlet.
int16_t lut_inlet_on_on[LUT_SIZE];
int16_t lut_inlet_off_on[LUT_SIZE];
int16_t lut_outlet_on_on[LUT_SIZE];
int16_t lut_outlet_off_on[LUT_SIZE];

#define GOOD_STATE_BAND_P_LOW    720   // ADC value at 0 psi
#define GOOD_STATE_BAND_P_HIGH   2900  // ADC value at 40 psi

static inline int IRAM_ATTR lerp_clamped_int(int pressure, int p_low, int p_high, int y_low, int y_high) {
	if (pressure <= p_low) return y_low;
	if (pressure >= p_high) return y_high;
	return y_low + (pressure - p_low) * (y_high - y_low) / (p_high - p_low);
}

static inline int IRAM_ATTR pressure_deadband_counts(int pressure) {
	// Low pressure -> 10 counts, high pressure -> 30 counts.
	return lerp_clamped_int(pressure, GOOD_STATE_BAND_P_LOW, GOOD_STATE_BAND_P_HIGH, 10, 25);
}

static inline int IRAM_ATTR pressure_inlet_min_undershoot_threshold(int pressure) {
	// Inlet pdiff is highest at low chamber pressure and lowest at high chamber pressure.
	// High pdiff -> 8, low pdiff -> 2.
	return lerp_clamped_int(pressure, GOOD_STATE_BAND_P_LOW, GOOD_STATE_BAND_P_HIGH, 6, 2);
}

static inline int IRAM_ATTR pressure_outlet_min_undershoot_threshold(int pressure) {
	// Outlet pdiff is lowest at low chamber pressure and highest at high chamber pressure.
	// High pdiff -> 8, low pdiff -> 2.
	return lerp_clamped_int(pressure, GOOD_STATE_BAND_P_LOW, GOOD_STATE_BAND_P_HIGH, 2, 6);
}

static inline int IRAM_ATTR clamp_duty(int duty) {
    if (duty > PWM_CAP) return PWM_CAP;
    if (duty < PWM_BOT) return PWM_BOT;
    return duty;
}

static inline int16_t *IRAM_ATTR lut_for_case(int action, int prev_same_action) {
    if (action == ACT_INLET) {
        return prev_same_action ? lut_inlet_on_on : lut_inlet_off_on;
    }
    return prev_same_action ? lut_outlet_on_on : lut_outlet_off_on;
}

// Integer linear interpolation from LUT. Returns -1 if no data available.
static int IRAM_ATTR lut_interpolate(int16_t *lut, int pressure) {
	if (pressure <= LUT_PRESSURE_MIN) {
		return (lut[0] >= 0) ? lut[0] : -1;
	}
	if (pressure >= LUT_PRESSURE_MAX) {
		return (lut[LUT_SIZE - 1] >= 0) ? lut[LUT_SIZE - 1] : -1;
	}

	int offset = pressure - LUT_PRESSURE_MIN;
	int bin_low = offset / LUT_BIN_WIDTH;
	if (bin_low >= LUT_SIZE - 1) bin_low = LUT_SIZE - 2;
	int bin_high = bin_low + 1;

	int frac_num = offset - bin_low * LUT_BIN_WIDTH; // 0..LUT_BIN_WIDTH

	// If both bins empty, no data
	if (lut[bin_low] < 0 && lut[bin_high] < 0) return -1;
	// If only one has data, use it
	if (lut[bin_low] < 0) return lut[bin_high];
	if (lut[bin_high] < 0) return lut[bin_low];

	// Linear interpolation: low + (high - low) * frac_num / BIN_WIDTH
	return lut[bin_low] + (lut[bin_high] - lut[bin_low]) * frac_num / LUT_BIN_WIDTH;
}

static int IRAM_ATTR lut_seed_duty(int action, int pressure, int prev_same_action) {
    int16_t *primary = lut_for_case(action, prev_same_action);
    int16_t *secondary = lut_for_case(action, !prev_same_action);
    int val = lut_interpolate(primary, pressure);
    if (val < 0) val = lut_interpolate(secondary, pressure);
    return val;
}

static void lut_store(int16_t *lut, int pressure, int duty);

static void IRAM_ATTR lut_adjust_from_history(control_history_entry_t *history, int center_index, int delta) {
	for (int offset = -BLAME_WINDOW_BACKWARD; offset <= BLAME_WINDOW_FORWARD; offset++) {
		int idx = center_index + offset;
		while (idx < 0) idx += CONTROL_HISTORY_LEN;
		while (idx >= CONTROL_HISTORY_LEN) idx -= CONTROL_HISTORY_LEN;

		const control_history_entry_t *entry = &history[idx];
		if (entry->action != ACT_INLET && entry->action != ACT_OUTLET) continue;
		if (entry->duty <= 0) continue;

		lut_store(lut_for_case(entry->action, entry->prev_same_action),
		          entry->pressure,
		          clamp_duty(entry->duty + delta));
	}
}

static bool IRAM_ATTR history_has_recent_actuation(control_history_entry_t *history, int center_index, int min_active_ticks) {
	for (int offset = -BLAME_WINDOW_BACKWARD; offset <= BLAME_WINDOW_FORWARD; offset++) {
		int idx = center_index + offset;
		while (idx < 0) idx += CONTROL_HISTORY_LEN;
		while (idx >= CONTROL_HISTORY_LEN) idx -= CONTROL_HISTORY_LEN;

		const control_history_entry_t *entry = &history[idx];
		if ((entry->action == ACT_INLET || entry->action == ACT_OUTLET) && entry->active_ticks > min_active_ticks) {
			return true;
		}
	}
	return false;
}

// Load the ID from NVS. If it doesn't exist, it keeps the default 0x102.
void load_device_id_from_nvs(void) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("config", NVS_READONLY, &nvs);
    if (err == ESP_OK) {
        uint32_t saved_id = 0;
        if (nvs_get_u32(nvs, "can_id", &saved_id) == ESP_OK) {
            device_can_id = saved_id;
            printf("Loaded CAN ID from NVS: 0x%lX\n", device_can_id);
        } else {
            printf("CAN ID not found in NVS. writing default: 0x%lX\n", device_can_id);
            save_device_id_to_nvs(device_can_id);
        }
        nvs_close(nvs);
    } else {
        printf("Config NVS namespace empty. Using default CAN ID: 0x%lX\n", device_can_id);
		save_device_id_to_nvs(device_can_id);
    }
}

// Call this function whenever you want to permanently change this actuator's address
void save_device_id_to_nvs(uint32_t new_id) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("config", NVS_READWRITE, &nvs);
    if (err == ESP_OK) {
        nvs_set_u32(nvs, "can_id", new_id);
        nvs_commit(nvs);
        nvs_close(nvs);
        device_can_id = new_id; // Update the active RAM variable
        printf("Successfully permanently changed CAN ID to: 0x%lX\n", device_can_id);
    } else {
        printf("Failed to open NVS to save new CAN ID!\n");
    }
}


static inline int16_t IRAM_ATTR inlet_default_from_index(int idx) {
    // 450 -> 650 from low pressure to high pressure
    return (int16_t)(450 + (200 * idx) / (LUT_SIZE - 1));
}

static inline int16_t IRAM_ATTR outlet_default_from_index(int idx) {
    // 650 -> 450 from low pressure to high pressure
    return (int16_t)(650 - (200 * idx) / (LUT_SIZE - 1));
}

static void lut_fill_missing_with_defaults(void) {
    for (int i = 0; i < LUT_SIZE; i++) {
        if (lut_inlet_on_on[i] < 0) lut_inlet_on_on[i] = inlet_default_from_index(i);
        if (lut_inlet_off_on[i] < 0) lut_inlet_off_on[i] = inlet_default_from_index(i);
        if (lut_outlet_on_on[i] < 0) lut_outlet_on_on[i] = outlet_default_from_index(i);
        if (lut_outlet_off_on[i] < 0) lut_outlet_off_on[i] = outlet_default_from_index(i);
    }
}

// Initialize LUT with default pressure-shaped seed values.
static void lut_init(void) {
	for (int i = 0; i < LUT_SIZE; i++) {
        lut_inlet_on_on[i] = inlet_default_from_index(i);
        lut_inlet_off_on[i] = inlet_default_from_index(i);
        lut_outlet_on_on[i] = outlet_default_from_index(i);
        lut_outlet_off_on[i] = outlet_default_from_index(i);
	}
}

// Load LUT from NVS. Returns true if data was found.
static bool lut_load_from_nvs(void) {
	nvs_handle_t nvs;
	esp_err_t err = nvs_open("lut", NVS_READONLY, &nvs);
	if (err != ESP_OK) return false;

    size_t len = sizeof(int16_t) * LUT_SIZE;
    bool loaded = false;

    // New format: per-direction and per-transition LUTs.
    if (nvs_get_blob(nvs, "in_on_on", lut_inlet_on_on, &len) == ESP_OK &&
        len == sizeof(int16_t) * LUT_SIZE) {
        len = sizeof(int16_t) * LUT_SIZE;
        if (nvs_get_blob(nvs, "in_off_on", lut_inlet_off_on, &len) == ESP_OK &&
            len == sizeof(int16_t) * LUT_SIZE) {
            len = sizeof(int16_t) * LUT_SIZE;
            if (nvs_get_blob(nvs, "out_on_on", lut_outlet_on_on, &len) == ESP_OK &&
                len == sizeof(int16_t) * LUT_SIZE) {
                len = sizeof(int16_t) * LUT_SIZE;
                if (nvs_get_blob(nvs, "out_off_on", lut_outlet_off_on, &len) == ESP_OK &&
                    len == sizeof(int16_t) * LUT_SIZE) {
                    loaded = true;
                }
            }
        }
    }

    // Backward-compatible migration from legacy 2-table format.
    if (!loaded) {
        int16_t old_inlet[LUT_SIZE];
        int16_t old_outlet[LUT_SIZE];
        len = sizeof(int16_t) * LUT_SIZE;
        if (nvs_get_blob(nvs, "inlet", old_inlet, &len) == ESP_OK &&
            len == sizeof(int16_t) * LUT_SIZE) {
            len = sizeof(int16_t) * LUT_SIZE;
            if (nvs_get_blob(nvs, "outlet", old_outlet, &len) == ESP_OK &&
                len == sizeof(int16_t) * LUT_SIZE) {
                for (int i = 0; i < LUT_SIZE; i++) {
                    lut_inlet_on_on[i] = old_inlet[i];
                    lut_inlet_off_on[i] = old_inlet[i];
                    lut_outlet_on_on[i] = old_outlet[i];
                    lut_outlet_off_on[i] = old_outlet[i];
                }
                loaded = true;
            }
        }
    }

	nvs_close(nvs);
	if (!loaded) {
		// If partial/corrupted or old LUT shape, reinitialize with defaults.
		lut_init();
	} else {
		// Safety: if any cells are missing, backfill with default ramp values.
		lut_fill_missing_with_defaults();
	}
	return loaded;
}

// Save LUT to NVS
static void lut_save_to_nvs(void) {
	nvs_handle_t nvs;
	esp_err_t err = nvs_open("lut", NVS_READWRITE, &nvs);
	if (err != ESP_OK) return;

    nvs_set_blob(nvs, "in_on_on", lut_inlet_on_on, sizeof(int16_t) * LUT_SIZE);
    nvs_set_blob(nvs, "in_off_on", lut_inlet_off_on, sizeof(int16_t) * LUT_SIZE);
    nvs_set_blob(nvs, "out_on_on", lut_outlet_on_on, sizeof(int16_t) * LUT_SIZE);
    nvs_set_blob(nvs, "out_off_on", lut_outlet_off_on, sizeof(int16_t) * LUT_SIZE);
	nvs_commit(nvs);
	nvs_close(nvs);
}



// Store a known-good duty cycle into the LUT, distributing across the two
// neighboring bins so that lut_interpolate(pressure) reproduces 'duty'.
//
// Interpolation reads:  duty = low*(w-f)/w + high*f/w
//   where f = fractional offset into the bin pair, w = LUT_BIN_WIDTH
//
// Store strategy:
//   - If exactly on a bin edge: write directly.
//   - If between bins: keep the farther bin fixed (or assume 'duty' if empty)
//     and solve for the nearer bin. When both exist, update the nearer one.
static void IRAM_ATTR lut_store(int16_t *lut, int pressure, int duty) {
	if (pressure < LUT_PRESSURE_MIN || pressure > LUT_PRESSURE_MAX) return;

	int offset = pressure - LUT_PRESSURE_MIN;
	int bin_low = offset / LUT_BIN_WIDTH;
	if (bin_low >= LUT_SIZE - 1) bin_low = LUT_SIZE - 2;
	if (bin_low < 0) bin_low = 0;
	int bin_high = bin_low + 1;

	int f = offset - bin_low * LUT_BIN_WIDTH; // 0..LUT_BIN_WIDTH
	int w = LUT_BIN_WIDTH;

	// Exactly on bin_low edge
	if (f == 0) {
		lut[bin_low] = (int16_t)duty;
		return;
	}

	// Both bins empty: store duty to both (flat assumption)
	if (lut[bin_low] < 0 && lut[bin_high] < 0) {
		lut[bin_low]  = (int16_t)duty;
		lut[bin_high] = (int16_t)duty;
		return;
	}

	// Solve for the unknown/nearer bin, keeping the other fixed.
	// From: duty = low_val * (w - f) / w + high_val * f / w
	//   => low_val  = (duty * w - high_val * f) / (w - f)
	//   => high_val = (duty * w - low_val * (w - f)) / f

	if (lut[bin_low] < 0 || (lut[bin_high] >= 0 && f <= w / 2)) {
		// Update bin_low, keep bin_high fixed
		int hv = lut[bin_high] >= 0 ? lut[bin_high] : duty;
		int val = (duty * w - hv * f) / (w - f);
		if (val < 0) val = 0;
		if (val > 1023) val = 1023;
		lut[bin_low] = (int16_t)val;
	} else {
		// Update bin_high, keep bin_low fixed
		int lv = lut[bin_low] >= 0 ? lut[bin_low] : duty;
		int val = (duty * w - lv * (w - f)) / f;
		if (val < 0) val = 0;
		if (val > 1023) val = 1023;
		lut[bin_high] = (int16_t)val;
	}
}
// =====================================================================


static void initialize_io(void) {
	gpio_config_t io_conf = {};
	//disable interrupt
    io_conf.intr_type = GPIO_INTR_DISABLE;
	io_conf.mode = GPIO_MODE_OUTPUT;
	io_conf.pin_bit_mask = ((1ULL<<V_IN) | (1ULL<<V_OUT));
	io_conf.pull_down_en = 0;
	io_conf.pull_up_en   = 0;
	gpio_config(&io_conf);

	pcr = malloc(sizeof(struct pressure_control_registers));
	acr = malloc(sizeof(struct ADC_read_registers));
	memset(pcr, 0, sizeof(struct pressure_control_registers));
	memset(acr, 0, sizeof(struct ADC_read_registers));
	// initialize the pressure control registers
	pcr->target_pressure = 1300;
	pcr->pending_target_pressure = 1300;
	pcr->active_target_pressure = 1300;
	pcr->latched_pressure = 0;
	pcr->last_sync_counter = 0;
	pcr->last_command_counter = 0;
	pcr->inlet_duty_cycle = 450;
	pcr->outlet_duty_cycle = 450;
	pcr->control_byte = 0;
	pcr->pending_control_byte = 0;
	pcr->active_control_byte = 0;
	pcr->status_flags = 0;
}


static void ledc_init(void)
{
    // Prepare and then apply the LEDC PWM timer configuration
    ledc_timer_config_t ledc_1_timer = {
        .speed_mode       = LEDC_1_MODE,
        .duty_resolution  = LEDC_1_DUTY_RES,
        .timer_num        = LEDC_1_TIMER,
        .freq_hz          = LEDC_1_FREQUENCY,  // Set output frequency at 4 kHz
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&ledc_1_timer));

    // Prepare and then apply the LEDC PWM channel configuration
    ledc_channel_config_t ledc_1_channel = {
        .speed_mode     = LEDC_1_MODE,
        .channel        = LEDC_1_CHANNEL,
        .timer_sel      = LEDC_1_TIMER,
        .intr_type      = LEDC_INTR_DISABLE,
        .gpio_num       = LEDC_1_OUTPUT_IO,
        .duty           = 0, // Set duty to 0%
        .hpoint         = 0
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ledc_1_channel));


	// Both channels share LEDC_TIMER_0 for synchronized PWM updates
    // Prepare and then apply the LEDC PWM channel configuration
    ledc_channel_config_t ledc_2_channel = {
        .speed_mode     = LEDC_2_MODE,
        .channel        = LEDC_2_CHANNEL,
        .timer_sel      = LEDC_2_TIMER,
        .intr_type      = LEDC_INTR_DISABLE,
        .gpio_num       = LEDC_2_OUTPUT_IO,
        .duty           = 0, // Set duty to 0%
        .hpoint         = 0
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ledc_2_channel));
}

static void initialize_lut(void) {
	// initialize LUT: try NVS first, fall back to blank
	if (!lut_load_from_nvs()) {
		lut_init();
		printf("LUT: no NVS data, starting fresh\n");
	} else {
		printf("LUT: loaded from NVS\n");
	}
    printf("LUT IN_ON_ON[");
	for (int i = 0; i < LUT_SIZE; i++) {
        printf("%d", lut_inlet_on_on[i]);
		if (i < LUT_SIZE - 1) printf(",");
	}
    printf("] IN_OFF_ON[");
	for (int i = 0; i < LUT_SIZE; i++) {
        printf("%d", lut_inlet_off_on[i]);
        if (i < LUT_SIZE - 1) printf(",");
    }
    printf("] OUT_ON_ON[");
    for (int i = 0; i < LUT_SIZE; i++) {
        printf("%d", lut_outlet_on_on[i]);
        if (i < LUT_SIZE - 1) printf(",");
    }
    printf("] OUT_OFF_ON[");
    for (int i = 0; i < LUT_SIZE; i++) {
        printf("%d", lut_outlet_off_on[i]);
		if (i < LUT_SIZE - 1) printf(",");
	}
	printf("]\n");
}

// --- Simple 1D Kalman Filter Variables ---
float k_est_pressure = 0.0; // The filtered output
float k_err_estimate = 1.0; // Error in estimate
const float k_err_measure = 2.0; // Measurement noise (Variance of your ADC)
const float k_q = 0.01;     // Process noise (How fast the actual pressure changes)

// 1D Kalman Filter Function
float update_kalman(float raw_measurement) {
    // Prediction update
    k_err_estimate = k_err_estimate + k_q;

    // Measurement update
    float kalman_gain = k_err_estimate / (k_err_estimate + k_err_measure);
    k_est_pressure = k_est_pressure + kalman_gain * (raw_measurement - k_est_pressure);
    k_err_estimate = (1.0 - kalman_gain) * k_err_estimate;

    return k_est_pressure;
}



void clippard_7mm_valve_bangbang_ctrl(void* args) {
	// Initialize oneshot ADC
	adc_oneshot_unit_handle_t adc_handle;
	adc_oneshot_unit_init_cfg_t init_cfg = {
		.unit_id = ADC_UNIT_1,
		.ulp_mode = ADC_ULP_MODE_DISABLE,
	};
	ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_cfg, &adc_handle));

	adc_oneshot_chan_cfg_t chan_cfg = {
		.atten = ADC_ATTEN_DB_12,
		.bitwidth = ADC_BITWIDTH_DEFAULT,
	};
	ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ADC_CHANNEL_8, &chan_cfg));


	#define DEADBAND 5
	while (1) {
		int tp = pcr->active_target_pressure;
		

		int deadband;
		if (tp < 1000) {deadband = DEADBAND;}
		else if (tp < 1500) {deadband = DEADBAND + 1;}
		else if (tp < 2000) {deadband = DEADBAND + 2;}
		else {deadband = DEADBAND + 3;}
        // 1. Take a burst of ADC readings (Oversampling)
        // Taking 8 readings at ~30us each takes ~240us total. 
        // This leaves plenty of time in our 2000us (2ms) window.
		int raw_sum = 0;
		int raw_samples = 0;
		for (int i = 0; i < 8; i++) {
            int raw;
			if (adc_oneshot_read(adc_handle, ADC_CHANNEL_8, &raw) != ESP_OK) continue;
			raw_sum += raw;
			raw_samples++;
            update_kalman(raw); // Feed the filter
        }

        // 2. Get the newest data from the Kalman filter
		float current_clean_pressure = k_est_pressure;
		int raw_avg = (raw_samples > 0) ? (raw_sum / raw_samples) : (int)current_clean_pressure;
		acr->adc_out = (uint16_t)raw_avg;
		acr->adc_RAF_result = (int) current_clean_pressure;
		pcr->current_pressure = (uint16_t)current_clean_pressure;
		
		// check ctrl byte
		int ctrl = pcr->active_control_byte;
		
		if ((ctrl & 0x01) == 0) {
			// if bit 0 is 0, skip control
			vTaskDelay(1);
			continue;
		}

        // 3. Bang-Bang Control Logic (using the deadband!)
        if (current_clean_pressure < (tp - deadband)) {
            gpio_set_level(V_IN, 1);
            gpio_set_level(V_OUT, 0);
        } 
        else if (current_clean_pressure > (tp + deadband)) {
            gpio_set_level(V_IN, 0);
            gpio_set_level(V_OUT, 1);
        } 
        else {
            gpio_set_level(V_IN, 0);
            gpio_set_level(V_OUT, 0);
        }
		vTaskDelay(1);
    }
}


void big_valve_pwm_ctrl(void* args) {
	// ========= 1. ADC Initialization ==========
	adc_oneshot_unit_handle_t adc_handle;
	adc_oneshot_unit_init_cfg_t init_cfg = {
		.unit_id = ADC_UNIT_1,
		.ulp_mode = ADC_ULP_MODE_DISABLE,
	};
	ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_cfg, &adc_handle));

	adc_oneshot_chan_cfg_t chan_cfg = {
		.atten = ADC_ATTEN_DB_12,
		.bitwidth = ADC_BITWIDTH_DEFAULT,
	};
	ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ADC_CHANNEL_8, &chan_cfg));

	// ========= 2. State Variables ==========
	int prev_cp = -1;
	int cmd_prev = ACT_NONE;
	int cmd_prev_ticks = 0;
	int loop_tick = 0;
	int last_lut_adjust_tick = -1000000;
	int inlet_overshoot_count = 0;
	int outlet_overshoot_count = 0;
	int inlet_undershoot_count = 0;
	int outlet_undershoot_count = 0;
	control_history_entry_t control_history[CONTROL_HISTORY_LEN] = {0};
	int history_head = 0;
	int history_count = 0;

	// 2-layer state machine: APPROACH → BRAKE → MAINTAIN
	#define STATE_MAINTAIN  0
	#define STATE_APPROACH  1
	#define STATE_BRAKE     2
	int ctrl_state = STATE_MAINTAIN;
	int brake_tick_counter = 0;
	int brake_settle_count = 0;   // consecutive low-dp ticks in brake
	int dp_ema_x10 = 0;          // smoothed dp, scaled ×10 for integer precision

	// ========= 3. Tunable Parameters ==========
	int system_delay = 5;
	int undershoot_confirm_ticks = 15;
	int overshoot_confirm_ticks = 2;
	int band_scale = 2;

	// ========= 4. Task Timing ==========
	const TickType_t xFrequency = pdMS_TO_TICKS(4);
	TickType_t xLastWakeTime = xTaskGetTickCount();

	while (1) {
		// ========= 5. ADC Sampling ==========
		int raw_sum = 0;
		int raw_samples = 0;
		for (int i = 0; i < 16; i++) {
			int raw;
			if (adc_oneshot_read(adc_handle, ADC_CHANNEL_8, &raw) != ESP_OK) continue;
			raw_sum += raw;
			raw_samples++;
			update_kalman(raw);
		}

		int cp = (int)k_est_pressure;
		int raw_avg = (raw_samples > 0) ? (raw_sum / raw_samples) : cp;
		acr->adc_out = raw_avg;
		acr->adc_RAF_result = cp;
		pcr->current_pressure = (uint16_t)cp;

		// ========= 6. Control Disable Check ==========
		int ctrl = pcr->active_control_byte;
		if ((ctrl & 0x01) == 0) {
			ledc_set_duty(LEDC_1_MODE, LEDC_1_CHANNEL, 0);
			ledc_set_duty(LEDC_2_MODE, LEDC_2_CHANNEL, 0);
			ledc_update_duty(LEDC_1_MODE, LEDC_1_CHANNEL);
			ledc_update_duty(LEDC_2_MODE, LEDC_2_CHANNEL);

			cmd_prev = ACT_NONE;
			cmd_prev_ticks = 0;
			loop_tick = 0;
			last_lut_adjust_tick = -1000000;
			inlet_overshoot_count = 0;
			outlet_overshoot_count = 0;
			inlet_undershoot_count = 0;
			outlet_undershoot_count = 0;
			ctrl_state = STATE_MAINTAIN;
			brake_tick_counter = 0;
			brake_settle_count = 0;
			dp_ema_x10 = 0;
			history_head = 0;
			history_count = 0;
			prev_cp = cp;
			vTaskDelayUntil(&xLastWakeTime, xFrequency);
			continue;
		}

		// ========= 7. Band & Error Calculations ==========
		int tp = pcr->active_target_pressure;
		if (prev_cp < 0) prev_cp = cp;

		if (system_delay < 1) system_delay = 1;
		if (system_delay > CONTROL_HISTORY_LEN) system_delay = CONTROL_HISTORY_LEN;
		if (undershoot_confirm_ticks < 1) undershoot_confirm_ticks = 1;
		if (overshoot_confirm_ticks < 1) overshoot_confirm_ticks = 1;
		if (band_scale < 1) band_scale = 1;

		int effective_system_delay = system_delay;
		int deadband_counts = pressure_deadband_counts(cp);
		deadband_counts *= band_scale;
		int inlet_min_undershoot_threshold = pressure_inlet_min_undershoot_threshold(cp);
		int outlet_min_undershoot_threshold = pressure_outlet_min_undershoot_threshold(cp);

		int perr = tp - cp;
		int dp = cp - prev_cp;
		int projected_cp = cp + (dp * effective_system_delay);

		// Smoothed dp for braking prediction (scaled ×10 for integer precision)
		dp_ema_x10 = (dp_ema_x10 * 7 + dp * 10 * 3) / 10;

		bool delayed_history_ready = (history_count >= effective_system_delay);
		int delayed_index = -1;
		if (delayed_history_ready) {
			delayed_index = history_head - effective_system_delay;
			if (delayed_index < 0) delayed_index += CONTROL_HISTORY_LEN;
		}

		// ========= 8–11. 2-Layer State Machine (APPROACH → BRAKE → MAINTAIN) ==========
		int desired_action = ACT_NONE;
		int duty_in = 0;
		int duty_out = 0;
		int abs_perr = (perr >= 0) ? perr : -perr;
		int dp_ema_abs = (dp_ema_x10 >= 0) ? dp_ema_x10 : -dp_ema_x10;

		if (ctrl_state == STATE_APPROACH) {
			// --- APPROACH: graduated duty + predictive braking ---
			// Determine approach direction
			int approach_dir = (perr > 0) ? ACT_INLET : ACT_OUTLET;

			// Predictive braking: stopping_distance from smoothed dp
			// brake_delay_ticks = system_delay * 3/2 (safety factor for delay variability)
			int brake_delay_ticks = effective_system_delay * 3 / 2;
			if (brake_delay_ticks < 2) brake_delay_ticks = 2;
			// stopping_distance in ADC counts (dp_ema_x10 is ×10, so divide by 10)
			int stopping_dist = (dp_ema_abs * brake_delay_ticks) / 10;

			// Brake condition: projected arrival crosses target ± maintain_band/2
			int brake_target_margin = MAINTAIN_BAND / 2;
			if (brake_target_margin < 1) brake_target_margin = 1;
			bool should_brake = false;

			if (approach_dir == ACT_INLET) {
				// Pressure rising toward target: brake if cp + stopping_dist >= tp - margin
				if (cp + stopping_dist >= tp - brake_target_margin) should_brake = true;
				// Already overshot
				if (perr < 0) should_brake = true;
			} else {
				// Pressure falling toward target: brake if cp - stopping_dist <= tp + margin
				if (cp - stopping_dist <= tp + brake_target_margin) should_brake = true;
				// Already overshot
				if (perr > 0) should_brake = true;
			}

			if (should_brake) {
				// Transition to BRAKE
				ctrl_state = STATE_BRAKE;
				brake_tick_counter = 0;
				brake_settle_count = 0;
				desired_action = ACT_NONE;
				duty_in = 0;
				duty_out = 0;
				printf("STATE APPROACH->BRAKE cp=%d tp=%d perr=%d dp_ema=%d stop_dist=%d\n",
				       cp, tp, perr, dp_ema_x10, stopping_dist);
			} else {
				// Graduated duty based on |perr|
				int approach_duty;
				if (abs_perr >= APPROACH_FAR_THRESHOLD) {
					approach_duty = APPROACH_MAX_DUTY;
				} else {
					// lerp from PWM_CAP to APPROACH_MAX_DUTY over [APPROACH_ENTRY_THRESHOLD, APPROACH_FAR_THRESHOLD]
					int range = APPROACH_FAR_THRESHOLD - APPROACH_ENTRY_THRESHOLD;
					int excess = abs_perr - APPROACH_ENTRY_THRESHOLD;
					if (excess < 0) excess = 0;
					approach_duty = PWM_CAP + (APPROACH_MAX_DUTY - PWM_CAP) * excess / range;
				}
				if (approach_duty > APPROACH_MAX_DUTY) approach_duty = APPROACH_MAX_DUTY;
				if (approach_duty < PWM_CAP) approach_duty = PWM_CAP;

				desired_action = approach_dir;
				if (approach_dir == ACT_INLET) {
					duty_in = approach_duty;
					pcr->inlet_duty_cycle = duty_in;
				} else {
					duty_out = approach_duty;
					pcr->outlet_duty_cycle = duty_out;
				}
			}

		} else if (ctrl_state == STATE_BRAKE) {
			// --- BRAKE: mandatory coast, both valves off ---
			desired_action = ACT_NONE;
			duty_in = 0;
			duty_out = 0;
			brake_tick_counter++;

			// Check settle: |dp_ema| < 10 (i.e. <1.0 in real units) for 3 consecutive ticks
			if (dp_ema_abs < 10) {
				brake_settle_count++;
			} else {
				brake_settle_count = 0;
			}

			// Exit to MAINTAIN when settled or max brake time elapsed
			int brake_max_ticks = effective_system_delay * BRAKE_SETTLE_FACTOR;
			if (brake_max_ticks < 4) brake_max_ticks = 4;
			if (brake_settle_count >= 3 || brake_tick_counter >= brake_max_ticks) {
				ctrl_state = STATE_MAINTAIN;
				// Reset OSUS counters on entering maintain
				inlet_overshoot_count = 0;
				outlet_overshoot_count = 0;
				inlet_undershoot_count = 0;
				outlet_undershoot_count = 0;
				printf("STATE BRAKE->MAINTAIN cp=%d tp=%d perr=%d dp_ema=%d brk_ticks=%d\n",
				       cp, tp, perr, dp_ema_x10, brake_tick_counter);
			}

		} else {
			// --- STATE_MAINTAIN: LUT + proportional correction ---

			// Check if error is large enough to enter APPROACH
			if (abs_perr > APPROACH_ENTRY_THRESHOLD) {
				ctrl_state = STATE_APPROACH;
				printf("STATE MAINTAIN->APPROACH cp=%d tp=%d perr=%d\n", cp, tp, perr);
				// On first approach tick, start with graduated duty immediately
				int approach_dir = (perr > 0) ? ACT_INLET : ACT_OUTLET;
				int approach_duty;
				if (abs_perr >= APPROACH_FAR_THRESHOLD) {
					approach_duty = APPROACH_MAX_DUTY;
				} else {
					int range = APPROACH_FAR_THRESHOLD - APPROACH_ENTRY_THRESHOLD;
					int excess = abs_perr - APPROACH_ENTRY_THRESHOLD;
					if (excess < 0) excess = 0;
					approach_duty = PWM_CAP + (APPROACH_MAX_DUTY - PWM_CAP) * excess / range;
				}
				if (approach_duty > APPROACH_MAX_DUTY) approach_duty = APPROACH_MAX_DUTY;
				if (approach_duty < PWM_CAP) approach_duty = PWM_CAP;

				desired_action = approach_dir;
				if (approach_dir == ACT_INLET) {
					duty_in = approach_duty;
					pcr->inlet_duty_cycle = duty_in;
				} else {
					duty_out = approach_duty;
					pcr->outlet_duty_cycle = duty_out;
				}

			} else if (abs_perr <= MAINTAIN_BAND) {
				// Within tolerance: both valves off
				desired_action = ACT_NONE;
				duty_in = 0;
				duty_out = 0;

			} else {
				// Proportional correction zone: MAINTAIN_BAND < |perr| <= APPROACH_ENTRY_THRESHOLD
				// Use projected_cp for action decision (delay compensation)
				int projected_err = tp - projected_cp;
				int abs_projected_err = (projected_err >= 0) ? projected_err : -projected_err;

				if (abs_projected_err <= MAINTAIN_BAND) {
					// Projected pressure is within band — coast, the pipeline is already handling it
					desired_action = ACT_NONE;
					duty_in = 0;
					duty_out = 0;
				} else {
					// Proportional correction: scale duty between LUT and halfway to PWM_CAP
					int correct_dir = (projected_err > 0) ? ACT_INLET : ACT_OUTLET;
					int prev_same_case = (cmd_prev == correct_dir);
					int lut_duty = clamp_duty(lut_seed_duty(correct_dir, cp, prev_same_case));

					int excess = abs_perr - MAINTAIN_BAND;
					int max_excess = APPROACH_ENTRY_THRESHOLD - MAINTAIN_BAND;
					if (max_excess < 1) max_excess = 1;
					// Scale from LUT_duty up to LUT_duty + (PWM_CAP - LUT_duty)/2
					int duty_boost_range = (PWM_CAP - lut_duty) / 2;
					int scaled_duty = lut_duty + duty_boost_range * excess / max_excess;
					if (scaled_duty > PWM_CAP) scaled_duty = PWM_CAP;
					if (scaled_duty < PWM_BOT) scaled_duty = PWM_BOT;

					desired_action = correct_dir;
					if (correct_dir == ACT_INLET) {
						duty_in = scaled_duty;
						pcr->inlet_duty_cycle = duty_in;
					} else {
						duty_out = scaled_duty;
						pcr->outlet_duty_cycle = duty_out;
					}
				}
			}

			// --- OSUS / LUT learning (only in MAINTAIN) ---
			if (delayed_history_ready) {
				bool learning_unlocked = history_has_recent_actuation(control_history, delayed_index, effective_system_delay);
				int lut_adjust_cooldown_ticks = effective_system_delay * 2;
				bool can_adjust_lut = ((loop_tick - last_lut_adjust_tick) >= lut_adjust_cooldown_ticks);
				if (learning_unlocked) {
					int delta_up = dp;
					int delta_down = -dp;

					if (delta_up > deadband_counts) {
						inlet_overshoot_count++;
						if (inlet_overshoot_count >= overshoot_confirm_ticks) {
							if (can_adjust_lut) {
								printf("OSUS evt=OVERSHOOT src=RATE blame=INLET cp=%d raw=%d prev_cp=%d tp=%d dp=%d db=%d cnt=%d cd=%d\n",
								       cp, raw_avg, prev_cp, tp, dp, deadband_counts, inlet_overshoot_count, lut_adjust_cooldown_ticks);
								lut_adjust_from_history(control_history, delayed_index, -1);
								last_lut_adjust_tick = loop_tick;
							}
							inlet_overshoot_count = 0;
						}
						outlet_overshoot_count = 0;
						inlet_undershoot_count = 0;
						outlet_undershoot_count = 0;
					} else if (delta_up > 0 && delta_up < inlet_min_undershoot_threshold) {
						inlet_overshoot_count = 0;
						outlet_overshoot_count = 0;
						inlet_undershoot_count++;
						outlet_undershoot_count = 0;
						if (inlet_undershoot_count >= undershoot_confirm_ticks) {
							if (can_adjust_lut) {
								printf("OSUS evt=UNDERSHOOT src=RATE blame=INLET cp=%d raw=%d prev_cp=%d tp=%d dp=%d in_us_thr=%d cnt=%d cd=%d\n",
								       cp, raw_avg, prev_cp, tp, dp, inlet_min_undershoot_threshold, inlet_undershoot_count, lut_adjust_cooldown_ticks);
								lut_adjust_from_history(control_history, delayed_index, +1);
								last_lut_adjust_tick = loop_tick;
							}
							inlet_undershoot_count = 0;
						}
					} else if (delta_down > deadband_counts) {
						outlet_overshoot_count++;
						if (outlet_overshoot_count >= overshoot_confirm_ticks) {
							if (can_adjust_lut) {
								printf("OSUS evt=OVERSHOOT src=RATE blame=OUTLET cp=%d raw=%d prev_cp=%d tp=%d dp=%d db=%d cnt=%d cd=%d\n",
								       cp, raw_avg, prev_cp, tp, dp, deadband_counts, outlet_overshoot_count, lut_adjust_cooldown_ticks);
								lut_adjust_from_history(control_history, delayed_index, -1);
								last_lut_adjust_tick = loop_tick;
							}
							outlet_overshoot_count = 0;
						}
						inlet_overshoot_count = 0;
						inlet_undershoot_count = 0;
						outlet_undershoot_count = 0;
					} else if (delta_down > 0 && delta_down < outlet_min_undershoot_threshold) {
						inlet_overshoot_count = 0;
						outlet_overshoot_count = 0;
						inlet_undershoot_count = 0;
						outlet_undershoot_count++;
						if (outlet_undershoot_count >= undershoot_confirm_ticks) {
							if (can_adjust_lut) {
								printf("OSUS evt=UNDERSHOOT src=RATE blame=OUTLET cp=%d raw=%d prev_cp=%d tp=%d dp=%d out_us_thr=%d cnt=%d cd=%d\n",
								       cp, raw_avg, prev_cp, tp, dp, outlet_min_undershoot_threshold, outlet_undershoot_count, lut_adjust_cooldown_ticks);
								lut_adjust_from_history(control_history, delayed_index, +1);
								last_lut_adjust_tick = loop_tick;
							}
							outlet_undershoot_count = 0;
						}
					} else {
						inlet_overshoot_count = 0;
						outlet_overshoot_count = 0;
						inlet_undershoot_count = 0;
						outlet_undershoot_count = 0;
					}
				} else {
					inlet_overshoot_count = 0;
					outlet_overshoot_count = 0;
					inlet_undershoot_count = 0;
					outlet_undershoot_count = 0;
				}
			} else {
				inlet_undershoot_count = 0;
				outlet_undershoot_count = 0;
				inlet_overshoot_count = 0;
				outlet_overshoot_count = 0;
			}
		} // end state machine

		// ========= 12. Zero-Pressure Safety Guard ==========
		if (cp == 0) {
			duty_in = 0;
			duty_out = 0;
			desired_action = ACT_NONE;
		}

		// ========= 13. PWM Output ==========
		ledc_set_duty(LEDC_1_MODE, LEDC_1_CHANNEL, duty_in);
		ledc_set_duty(LEDC_2_MODE, LEDC_2_CHANNEL, duty_out);
		ledc_update_duty(LEDC_1_MODE, LEDC_1_CHANNEL);
		ledc_update_duty(LEDC_2_MODE, LEDC_2_CHANNEL);

		// ========= 14. State Update ==========
		int next_cmd_prev_ticks = 0;
		if (desired_action != ACT_NONE) {
			next_cmd_prev_ticks = (desired_action == cmd_prev) ? (cmd_prev_ticks + 1) : 1;
		}
		control_history[history_head].pressure = cp;
		control_history[history_head].raw = raw_avg;
		control_history[history_head].target = tp;
		control_history[history_head].action = desired_action;
		control_history[history_head].duty = (desired_action == ACT_INLET) ? duty_in : ((desired_action == ACT_OUTLET) ? duty_out : 0);
		control_history[history_head].prev_same_action = (desired_action != ACT_NONE && desired_action == cmd_prev);
		control_history[history_head].active_ticks = next_cmd_prev_ticks;
		history_head = (history_head + 1) % CONTROL_HISTORY_LEN;
		if (history_count < CONTROL_HISTORY_LEN) history_count++;
		cmd_prev = desired_action;
		cmd_prev_ticks = next_cmd_prev_ticks;
		prev_cp = cp;
		loop_tick++;
		vTaskDelayUntil(&xLastWakeTime, xFrequency);
	}
}

void app_main(void)
{	
	// let the PWM run on core 0.
	// Initialize NVS
	esp_err_t nvs_err = nvs_flash_init();
	if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
		nvs_flash_erase();
		nvs_flash_init();
	}

    load_device_id_from_nvs();

	if (VALVE_TYPE==0) {
		// bang bang control only needs IO, no PWM
		initialize_io();
	} else {
		initialize_lut();
		initialize_io();
		ledc_init();
	}



	// Control tasks run on core 0; CAN and maintenance tasks run on core 1.
	if (VALVE_TYPE==0) {
		xTaskCreatePinnedToCore(clippard_7mm_valve_bangbang_ctrl, "BangBangCtrl", 10000, NULL, 10, NULL, 0);
	} else {
		xTaskCreatePinnedToCore(big_valve_pwm_ctrl, "PWMCtrl", 10000, NULL, 10, NULL, 0);
	}

	xTaskCreatePinnedToCore(CAN_Subroutine, "CAN_routine", 10000, NULL, 10, &handle_CAN_Subroutine, 1);
	xTaskCreatePinnedToCore(Serial_Subroutine, "SER_routine", 10000, NULL, 3, &handle_Serial_Subroutine, 1);
	xTaskCreatePinnedToCore(WATCHDOG_subroutine, "WD_routine", 5000, NULL, 5, &handle_WATCHDOG_subroutine, 1);

	while (1) {
		vTaskDelay(1000 / portTICK_PERIOD_MS);
	}
}


void WATCHDOG_subroutine(void * pvParameters) {
	while(1){
		// for every 0.5 seconds, reset the watchdog
		CAN_starv_count++;
		if (CAN_starv_count > 20) {
			// if more than 10 seconds no CAN frames received, reset the ESP32
			CAN_starv_count = 0;
			CAN_recover_from_starvation();
			// printf("No CAN frames received for 10 seconds, resetting MCP2515 Interrupts...\n");
		}
		vTaskDelay(500 / portTICK_PERIOD_MS);
	}
}

void Serial_Subroutine(void * pvParameters) {
	while(1){
		// Periodic telemetry prints are intentionally disabled.
		// Only OSUS event logs should appear in monitor.

		// Save LUT to NVS every 30 seconds (silent)
		{
			static int nvs_save_counter = 0;
			nvs_save_counter++;
			if (nvs_save_counter >= 300) { // 300 * 100ms = 30s
				nvs_save_counter = 0;
				lut_save_to_nvs();
			}
		}

		vTaskDelay(1000 / portTICK_PERIOD_MS);
	}
}


bool SPI_Init(void)
{
	printf("Hello from SPI_Init!\n\r");
	esp_err_t ret;
	//Configuration for the SPI bus
	spi_bus_config_t bus_cfg={
		.miso_io_num=PIN_NUM_MISO,
		.mosi_io_num=PIN_NUM_MOSI,
		.sclk_io_num=PIN_NUM_CLK,
		.quadwp_io_num=-1,
		.quadhd_io_num=-1,
		.max_transfer_sz = 0 // no limit
	};

	// Define MCP2515 SPI device configuration
	spi_device_interface_config_t dev_cfg = {
		.mode = 0, // (0,0)
		.clock_speed_hz = 16000000, // 16 mhz
		.spics_io_num = PIN_NUM_CS,
		.queue_size = 1024
	};

	// Initialize SPI bus (no DMA)
	ret = spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_DISABLED);
	ESP_ERROR_CHECK(ret);

    // Add MCP2515 SPI device to the bus
    ret = spi_bus_add_device(SPI2_HOST, &dev_cfg, &MCP2515_Object->spi);
    ESP_ERROR_CHECK(ret);

    return true;
}
