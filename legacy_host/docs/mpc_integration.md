# Controller / MPC Integration

This bundle is meant to be copied into a project that runs a controller (MPC, RL
policy, teleoperation, system-ID) on top of the robot. The C++ backend
(`vnema_backend.exe`) owns the hard-real-time 150 Hz loop — CAN command + sync,
reply collection, mocap clock alignment, and the mocap→joint observer
([state_observer.md](state_observer.md)). Your controller runs as a separate
process and talks to the backend over a tiny **line-delimited JSON** protocol:

- **stdin** ← your controller writes one JSON command object per line.
- **stdout** → the backend writes one JSON message object per line.

This keeps the controller language-agnostic. A complete, runnable reference client
is in [`examples/mpc_client_example.py`](../examples/mpc_client_example.py).

## Launch modes

**Interactive (controller-driven) — use this for MPC:**

```powershell
host\pc_backend\build\vnema_backend.exe --simulate-can --mocap-sim --ids 0x101-0x118 --stream-state
```

The backend prints `{"type":"backend","state":"ready",...}` and then waits for
commands on stdin. The 150 Hz loop does **not** start until you send
`{"cmd":"start"}`. Swap `--simulate-can --mocap-sim` for `--port COM4 --mocap-live ...`
on real hardware.

**Auto-run (one-shot, no stdin):** add `--status-only` or `--duration <s>` and the
loop runs immediately for that duration. Useful for soak/diagnostic runs.

### Why `--stream-state`

By default `robot_state` is emitted only at the ~1 Hz telemetry heartbeat. A
closed-loop controller needs fresh state every cycle, so pass **`--stream-state`**
to emit one `robot_state` line per 150 Hz cycle. The verbose `board`/`cycle`
telemetry stays at 1 Hz regardless.

## Commands (controller → backend, one JSON object per line)

| Command | Fields | Effect |
| --- | --- | --- |
| `start` | — | Start the 150 Hz loop. |
| `stop` | — | Stop the loop and send an all-outputs-off frame. |
| `shutdown` | — | Stop and exit the process. |
| `enable_outputs` | `enable` (bool) | Set the control-enable flag for all actuators. Outputs are **off** until enabled. |
| `set_targets` | `targets` (int[]), `enable` (bool, optional) | **MPC fast path.** One target-pressure (ADC counts) per actuator, aligned to `robot_state.ids` order. Length must equal the selected actuator count. Optional `enable` toggles control-enable atomically in the same frame. |
| `set_target` | `id` (int), `target` (int) | Set one actuator's target by CAN id. |
| `set_all_targets` | `target` (int) | Set the same target for every actuator. |
| `run_status_check` | `duration_s` (int) | Run a fixed-duration status-only loop (outputs forced off). |
| `start_data_collection` | `duration_min` (number) | Begin the random-target data-collection workflow (writes JSONL under `real_system_data_collection/`). Optional; for system-ID, not control. |
| `stop_data_collection` | — | End data collection (deflate + finalize). |

Targets are **12-bit ADC counts** in `[0, 4095]` (0 = empty, 4095 = full scale).
Out-of-range values are clamped. Use [`tools/calibrate_pressure.py`](../tools/calibrate_pressure.py)
to map ADC↔psi per actuator.

Example command lines:

```json
{"cmd":"start"}
{"cmd":"set_targets","targets":[1200,1200,800,800,0,0,1200,1200,800,800,0,0,1200,1200,800,800,0,0,1200,1200,800,800,0,0],"enable":true}
{"cmd":"stop"}
{"cmd":"shutdown"}
```

## `robot_state` message (backend → controller)

One per cycle when `--stream-state` is set. The control-relevant fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `cycle` | int | Monotonic 150 Hz cycle counter. |
| `can_sync_time_s` | float | Backend-clock timestamp of this cycle's DLC-0 sync edge (the instant the state is aligned to). |
| `ids` | int[] | Selected actuator CAN ids, **decimal**, sorted ascending (`257` = `0x101`). Defines the column order for all per-actuator arrays and for `set_targets`. |
| `joint_current_valid` | bool | True if the observer produced a usable estimate this cycle. Gate your controller on this. |
| `joint_current_theta` | float[12] | `q` — joint angles [rad] at `can_sync_time_s`. |
| `joint_current_theta_dot` | float[12] | `qdot` — joint velocities [rad/s]. |
| `joint_current_extrapolated` | bool | True if `q` was extrapolated past the newest mocap sample. |
| `joint_fixed_delay_valid` | bool | Validity of the delay-compensated snapshot. |
| `joint_fixed_delay_s` | float | Delay offset (default `0.008` s). |
| `joint_fixed_delay_theta` / `..._theta_dot` | float[12] | `q`,`qdot` at `can_sync_time_s − joint_fixed_delay_s` (for delay-aware control). |
| `pressure_adc_filtered` | int[] | Per-actuator measured pressure, latched at sync [ADC counts]. |
| `pressure_calibrated` | float[] | Per-actuator pressure [psi] if `calibration.json` is loaded. |
| `target_next_sync` | int[] | Per-actuator target that will be applied at the next sync [ADC counts]. |
| `control_next_sync` | int[] | Per-actuator control flags (bit0 = enable). |
| `actuator_status` | int[] | Compact status byte per actuator (see `can_protocol.md`). |
| `actuator_stale` | bool[] | True if an actuator missed its reply this cycle. |
| `fk_valid` | bool | Forward-kinematics validity. |
| `fk_tip` | float[3] | Tip position `[x,y,z]` [m] from FK(`q`). |
| `mocap_stale` | bool | True if the mocap sample is older than the freshness threshold. |
| `mocap_frame_rate_hz`, `mocap_timestamp_offset_ms`, `mocap_frame_drop_count` | num | Mocap clock-alignment health. |
| `cycle_responded` / `cycle_expected` | int | Actuators that replied this cycle vs. selected count. |
| `observer_time_ms`, `fk_time_ms`, `*_over_budget_count` | num | Per-cycle compute timing vs. budget. |

## Other message types

- `backend` — lifecycle (`ready`, `running`, `outputs`, `stopped`).
- `board` — per-actuator status (1 Hz heartbeat).
- `cycle` — per-cycle aggregate stats (1 Hz heartbeat).
- `mocap_status` / `mocap_log` — mocap bridge state.
- `collection` — data-collection lifecycle events.
- `error` — backend error (string `message`). Always drain these.

## Recommended controller loop

1. Launch the backend (interactive, `--stream-state`).
2. Wait for `{"type":"backend","state":"ready"}`, send `{"cmd":"start"}`.
3. Continuously read stdout; keep the latest `robot_state` (run the reader on its
   own thread so I/O never blocks your solve — see the example client).
4. Once per new `cycle`: if `joint_current_valid`, run your MPC solve on
   `(q, qdot, pressure_adc_filtered, ...)` and publish a full target vector with
   `set_targets`.
5. On exit, send `{"cmd":"stop"}` then `{"cmd":"shutdown"}`.

## Safety

- Outputs are **disabled by default**. The firmware will not actuate until control
  enable is set (via `enable_outputs` or the `enable` flag on `set_targets`).
- `stop` and `shutdown` send an all-outputs-off frame.
- Always bring up a new system with outputs disabled and follow
  [validation.md](validation.md) before enabling actuation.
