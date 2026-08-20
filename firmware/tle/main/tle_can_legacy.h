#ifndef TLE_CAN_LEGACY_H_
#define TLE_CAN_LEGACY_H_

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "mcp2515_min.h"

/*
 * Old-PCB-compatible CAN protocol for the TLE board.
 * =================================================
 *
 * The production robot runs 24 actuators on one bus at 150 Hz. Sixteen of them
 * are the original VNEMA_MK8_PIDPWM boards driving 7 mm valves; the top eight
 * are being replaced by TLE boards driving Clippard DVP proportional valves.
 * Both must answer the same host, on the same wire, to the same sync edge, so
 * this layer implements the old protocol byte for byte. The canonical
 * description lives in
 * VNEMA_MK8_PIDPWM/portable_sync_comm_layer/docs/can_protocol.md; what follows
 * is the part a reader of this file needs.
 *
 * Identifiers (11-bit standard, 1 Mbit/s):
 *
 *   0x090            broadcast. DLC 0 is the sync edge; DLC > 0 is OTA image
 *                    data, but only while an OTA session is open.
 *   0x091..0x098     runtime target table. One frame carries three actuators,
 *                    so the host places 24 targets in 8 frames.
 *   0x101..0x118     actuator base IDs -- also the ID a board answers on.
 *                    0x101..0x108 are the eight this board type replaces.
 *   base + 0x100     host control (OTA start/end, set ID, diagnostics).
 *   base + 0x200     unicast OTA data, used to repair one board that fell
 *                    behind the broadcast stream.
 *   base + 0x300     board status.
 *   base + 0x400     this project's own richer command set. Not part of the
 *                    old protocol; it exists because the tuning and telemetry
 *                    surface of the proportional controller has no equivalent
 *                    in the compact frames, and the old boards never look at
 *                    this ID. Handled by pressure_controller.c, not here.
 *
 * The 150 Hz cycle:
 *
 *   1. Host sends the runtime table frames. Boards stage what they receive;
 *      nothing changes yet.
 *   2. Host sends the DLC-0 sync on 0x090.
 *   3. Every board promotes its staged target to the live one and, in the same
 *      handler, latches the filtered pressure it will report. The reported
 *      value therefore belongs to the sync instant even though the replies
 *      themselves trickle out over the following milliseconds.
 *   4. Boards whose slot was active reply with compact status on their base ID.
 *
 * Pressure on the wire is a 12-bit field. This board's sensor path is 16-bit,
 * so the compact value is the raw reading shifted right by
 * TLE_LEGACY_RAW_SHIFT: one count is 16 raw, about 0.0165 psi, which is finer
 * than the old boards' own ~0.018 psi/count and a third of this controller's
 * default deadband. Absolute psi still comes from the host's per-actuator
 * calibration, exactly as it does for the old boards.
 */

/* ---- wire constants (shared with the old firmware) ---- */
#define TLE_LEGACY_BROADCAST_ID 0x090U
#define TLE_LEGACY_RUNTIME_TABLE_BASE_ID 0x091U
#define TLE_LEGACY_RUNTIME_FIRST_ID 0x101U
#define TLE_LEGACY_RUNTIME_LAST_ID 0x118U
#define TLE_LEGACY_RUNTIME_TABLE_SLOTS 3U
#define TLE_LEGACY_RUNTIME_TABLE_DLC 8U
#define TLE_LEGACY_RUNTIME_TABLE_MARKER 0x80U
#define TLE_LEGACY_RUNTIME_TABLE_SLOTMASK 0x1FU

#define TLE_LEGACY_HOST_CTRL_OFFSET 0x100U
#define TLE_LEGACY_HOST_DATA_OFFSET 0x200U
#define TLE_LEGACY_STATUS_OFFSET 0x300U

#define TLE_LEGACY_PRESSURE_MASK 0x0FFFU
#define TLE_LEGACY_FLAGS_MASK 0x000FU

#define TLE_LEGACY_CONTROL_ENABLE 0x01U

#define TLE_LEGACY_STATUS_ENABLED 0x01U
#define TLE_LEGACY_STATUS_OTA_ACTIVE 0x02U
#define TLE_LEGACY_STATUS_COMMAND_SEEN 0x04U
#define TLE_LEGACY_STATUS_ERROR 0x08U

#define TLE_LEGACY_CMD_START 0x01U
#define TLE_LEGACY_CMD_END 0x02U
#define TLE_LEGACY_CMD_SET_ID 0x05U
#define TLE_LEGACY_CMD_GET_CAN_DIAG 0x06U
#define TLE_LEGACY_CMD_CLEAR_CAN_DIAG 0x07U
#define TLE_LEGACY_CMD_GET_OTA_STATUS 0x08U
#define TLE_LEGACY_CMD_GET_FW_VERSION 0x09U

#define TLE_LEGACY_MSG_ACK 0xAAU
#define TLE_LEGACY_MSG_NACK 0xFFU
#define TLE_LEGACY_MSG_CAN_DIAG0 0xD0U
#define TLE_LEGACY_MSG_CAN_DIAG1 0xD1U
#define TLE_LEGACY_MSG_CAN_DIAG2 0xD2U
#define TLE_LEGACY_MSG_CAN_DIAG3 0xD3U
#define TLE_LEGACY_MSG_OTA_STATUS 0xD4U
#define TLE_LEGACY_MSG_FW_VERSION 0xD5U

/* Old firmware reports 0 = 7 mm valve, 1 = DT big valve. This board is a third
 * kind, so it reports 2 and a host that only knows the first two prints it as
 * unknown rather than mistaking it for either. */
#define TLE_LEGACY_FW_VARIANT_TLE_DVP 0x02U
#define TLE_LEGACY_FW_VERSION_CHUNK_BYTES 4U

/* 16-bit sensor raw <-> 12-bit wire field. */
#define TLE_LEGACY_RAW_SHIFT 4U

/* A board that has been driven by a sync master and then stops hearing one has
 * lost its host: hold the outputs off rather than the last commanded target. */
#define TLE_LEGACY_SYNC_TIMEOUT_US (500LL * 1000)
/* Periodic native telemetry is suppressed while a sync master is present, so
 * eight boards do not add ~1600 frames/s to a bus doing real-time control. */
#define TLE_LEGACY_SYNC_ACTIVE_US (200LL * 1000)
/* The enable bit is a level: while the host still commands it, a loop that
 * stopped itself is re-armed at this interval rather than on every edge.
 * Re-arming reconfigures the driver over SPI3, so it must not be attempted
 * at cycle rate while a fault persists. */
#define TLE_LEGACY_REARM_PERIOD_US (500LL * 1000)

typedef struct {
    /* Latest filtered sensor reading in 16-bit raw counts. */
    uint16_t (*get_filtered_raw)(void);
    /* Apply a control target immediately, in 16-bit raw counts. Called from
     * the sync handler, which is what makes it the synchronised edge. */
    void (*set_target_raw)(uint16_t raw);
    /* Start or stop closed-loop control. Stopping must force every output off
     * on a path that does not depend on the control loop making progress. */
    void (*set_enabled)(bool enable, uint16_t target_raw);
    bool (*is_running)(void);
    esp_err_t (*send_frame)(uint16_t can_id, const uint8_t *data, uint8_t dlc);
    /* Put whatever send_frame has queued on the wire before returning. Needed
     * before a reboot, which would otherwise take the acknowledgement with it. */
    void (*flush_tx)(void);
    /* Persist a new base ID. False if the ID is rejected. */
    bool (*save_base_id)(uint16_t base_id);
} tle_can_legacy_hooks_t;

void tle_can_legacy_init(const tle_can_legacy_hooks_t *hooks, uint16_t base_id);
void tle_can_legacy_set_base_id(uint16_t base_id);

/* Runtime-table group ID for this board, or 0 when the base ID sits outside
 * the 0x101..0x118 actuator range (a bench board, say). */
uint16_t tle_can_legacy_runtime_table_id(uint16_t base_id);

/* Route one received standard frame. True when the frame belonged to this
 * protocol and has been dealt with. */
bool tle_can_legacy_handle_frame(const mcp2515_frame_t *frame);

/* True while a sync master is driving the bus (see TLE_LEGACY_SYNC_ACTIVE_US). */
bool tle_can_legacy_bus_synced(void);

/* Run from the CAN task: enforces the sync-loss failsafe. */
void tle_can_legacy_service(void);

/* Counters the CAN task owns but the diagnostics frames report. */
void tle_can_legacy_note_rx_overflow(void);
void tle_can_legacy_note_tx_failure(void);
void tle_can_legacy_note_bus_error(uint8_t eflg);

#endif
