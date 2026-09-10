# RS485 twin — simulation core and the "behave like the real system" protocol

Worktree root: `C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/`
All paths below are absolute under that root unless noted.

**Read completely:** `UMArm_SIM/sim_core.py` (953), `UMArm_SIM/sim_master.py` (166), `UMArm_SIM/replay.py` (341), `UMArm_SIM/__init__.py` (49), `UMArm_SIM/test_sim_core.py` (486), `UMArm_SIM/test_sim_master.py` (415), `UMArm_ROBOT_CONTROL/fake_arm.py` (739).
**Read in part (targeted):** `UMArm_SIM/actuator_model.py` (constants + `ActuatorModel` class, lines 96–555), `UMArm_SIM/mjcf_generator.py` (constants block + `<option>` line), `UMArm_SIM/sim_mocap.py` (params, `PLATE_BODY_TABLE`, `step`, `_run`), `UMArm_ROBOT_CONTROL/arm_constants.py` (constants + unit helpers), `UMArm_COLLAB/protocol.py` (all 710 lines, but the schedule/rehearsal geometry is summarized, not transcribed).
**Did NOT read:** `sim_core`'s siblings `tune.py`, `twin_compare.py`, `fit_bounce.py`, `ring_analysis.py`, `bounce_demo.py`, `force_audit.py`; the trainers in `actuator_model.py` (lines 556–1160); the MJCF body/tendon construction in `mjcf_generator.py`; `vema_proto.py`; `arm_bus.py`; `UMArm_KoopmanMPPI/arch_plants.py` (referenced by sim_core's opt-in seams but not opened).

---

## 0. The one-paragraph version

The twin is a **four-layer stack with exactly one mutex**. `FakeNode` (in `UMArm_ROBOT_CONTROL/fake_arm.py`) is a line-for-line model of the node firmware — bang-bang, cal gating, 0xFFFF hold, 1000 ms failsafe, ADC sampler, status byte. `SimNode` subclasses it and replaces **only the plant** (three methods: `_integrate`, `true_psi`, `leak_pa_s`), so every firmware semantic is literally the same code. `SimArm` owns MuJoCo `model`/`data` + 24 `SimNode`s + one `threading.RLock`, and exposes a deterministic grid-quantized `advance_to(t)` callable from any thread. `SimMaster` subclasses `FakeMaster` (the transport shell — TDMA reply heap, MGMT collision window, `collect` blocking) and changes exactly three things. `replay.py` drives a bare `SimArm` on recorded timestamps with no clock and no threads. **The protocol is not a document — it is the set of assertions in `test_sim_core.py` / `test_sim_master.py`, listed verbatim in §7.**

---

## 1. `UMArm_SIM/sim_core.py` — `SimArm` and `SimNode`

### 1.1 Module-level constants

```python
# sim_core.py:117
NODE_LOGIC_EVERY = 5
# sim_core.py:121
_DWELL_EPS_S = 1e-12
# sim_core.py:153
HIRES_ADC = AdcSpec(counts_per_psi=1094.1,
                    noise_sd_counts=0.004 * 1094.1,
                    adc_max=65535, ambient_counts=3000.0)
# sim_core.py:163
LEAK_FAULT_PA_PER_S = 3000.0
```

`AdcSpec` (`sim_core.py:124-148`) is a frozen dataclass:

```python
@dataclass(frozen=True)
class AdcSpec:
    counts_per_psi: float
    noise_sd_counts: float
    adc_max: int
    ambient_counts: float = 3000.0

    @property
    def noise_psi(self) -> float:
        return self.noise_sd_counts / self.counts_per_psi
```

Semantics per its docstring (`sim_core.py:126-139`): the shipped 12-bit rack is ~34–37 counts/psi, `ADC_MAX = 4095`, measured noise sd 1.7–2.7 counts = 0.05–0.08 psi. `HIRES_ADC` is the ESP32S3/V4 LTC1864 front end: **16-bit, 1094.1 counts/psi, 4.3-count = 0.004 psi noise floor**, a 15–20× lower floor. `LEAK_FAULT_PA_PER_S = 3000.0` is chosen against the rig's ~720 Pa/s natural leak and a 1 %/s-of-15-psi watch threshold (~1030 Pa/s).

### 1.2 `SimNode.__init__` — exact signature

```python
# sim_core.py:176-184
def __init__(self, node_id: int, ambient: float, span_counts: float,
             sim: SimParams, fault: NodeFault, rng: random.Random, *,
             actuator: ActuatorModel,
             leak_fault_pa_per_s: float = LEAK_FAULT_PA_PER_S,
             ideal_pressure_tau_s: float | None = None,
             adc: AdcSpec | None = None,
             min_dwell_s: float = 0.0,
             foh_ramp: bool = False,
             ramp_span_s: float | None = None) -> None:
```

State it adds on top of `FakeNode`:

| attribute | type / units | meaning | line |
|---|---|---|---|
| `p_pa` | float, Pa gauge | **THE plant state**, one scalar per node | 188 |
| `_l_m` | float, m | anchored muscle length, init `actuator.l0_per_act[node_id-1]` | 199 |
| `_ldot_m_s` | float, m/s | anchored muscle rate | 200 |
| `air_in_psi` | float, psi | Σ max(dp,0) over inlet substeps | 208 |
| `air_out_psi` | float, psi | Σ \|dp\| over exhaust substeps | 211 |
| `valve_switches` | int | valve state CHANGES since construction | 214 |
| `adc` | `AdcSpec\|None` | 16-bit front end, `None` = parent's 12-bit | 218 |
| `min_dwell_s` | float, s | minimum valve dwell; 0.0 = today | 225 |
| `_last_switch_s` | float, s | init `-math.inf` | 226 |
| `_now_s` | float, s | node's own monotone clock | 230 |
| `ramp_ref_pa` | `float\|None` | FOH internal setpoint; `None` = disabled | 234 |
| `ramp_rate_pa_s` | float, Pa/s | current slope | 235 |
| `ramp_span_s` | float, s | default `1.0 / K.CONTROL_TICK_HZ` = **1/160 s** | 238 |
| `pending_ramp_rate_pa_s` | `float\|None` | staged slower slope, consumed by `bus_setpoint` | 243 |
| `_net_flow_cache` | `float\|None` | batched-forward result, consumed by `_integrate` | 247 |
| `switch_log` | `list[(t,state)]\|None` | off by default | 252 |

Inherited state (mirrors `control.cpp` by name, `fake_arm.py:181-216`): `act_state` (0 off / 1 inlet / 2 outlet), `controller_sel`, `target_pa` (Pa), `pressure_pa` (Pa, *measured*), `failsafe_active`, `ever_ticked`, `last_tick`, `raw` (ADC counts), `ambient`, `span_counts`, `counts_per_psi = span_counts/sim.ceiling_psi`, `zero_off`, `scale`, `tcs`, `tref`, `inlet_open_s`, `outlet_open_s`, `cal_writes`.

### 1.3 The plant seam — exactly three overrides

**(a) ADC coupling pinned** (`sim_core.py:256-262`):

```python
@property
def true_psi(self) -> float:
    return self.p_pa / K.PA_PER_PSI
```

This is the *whole* coupling. The parent's `_sample_adc` (`fake_arm.py:236-251`) then computes, untouched:

```python
limit = 3.0 * self.sim.noise_sd_counts
self._noise = max(-limit, min(limit, self._rng.gauss(0.0, self.sim.noise_sd_counts)))
raw = self.ambient + self.true_psi * self.counts_per_psi + self._noise
self.raw = int(max(0, min(K.ADC_MAX, round(raw))))
if self.cal_valid():
    self.pressure_pa = (float(self.raw) - float(self.zero_off)) * self.scale
else:
    self.pressure_pa = 0.0
```

`SimNode._sample_adc` (`sim_core.py:264-286`) is a pass-through to `super()` unless `self.adc` is set, in which case it is the identical model at that spec's resolution/noise/full-scale.

**(b) leak** (`sim_core.py:299-305`):

```python
@property
def leak_pa_s(self) -> float:
    if self.fault.leak:
        return self.leak_fault_pa_per_s
    return float(self.actuator.leak_pa_s[self.node_id - 1])
```

**(c) `_integrate`** (`sim_core.py:307-346`) — replaces the parent's double-exponential:

```python
cached, self._net_flow_cache = self._net_flow_cache, None
if dt <= 0.0:
    return
if self.act_state == 1:
    self.inlet_open_s += dt
elif self.act_state == 2:
    self.outlet_open_s += dt

p0 = self.p_pa
if self.ideal_pressure_tau_s is not None:
    self._integrate_ideal(dt)
    self._accrue_air(p0)
    return

valve = self.act_state
if valve == VALVE_EXHAUST and self.fault.stuck_vent:
    valve = VALVE_CLOSED
net = (self.actuator.net_flow_pa_s(self.node_id, self.p_pa, valve,
                                   self._l_m, self._ldot_m_s)
       if cached is None else cached)
dp = net - self.leak_pa_s
self.p_pa = max(0.0, self.p_pa + dp * dt)      # clamp >= 0 EVERY substep
self._accrue_air(p0)
```

`VALVE_CLOSED/INLET/EXHAUST = 0/1/2` (`actuator_model.py:138-140`), numerically equal to firmware `act_state`.

`stuck_vent` flips **only the flow model's valve input**; `act_state` — and therefore `status_bits()` and `outlet_open_s` — stays honest. That asymmetry is asserted in `test_sim_core.py:204-220`.

### 1.4 `SimArm.__init__` — exact signature

```python
# sim_core.py:548-572
def __init__(self, *,
             actuator: ActuatorModel | None = None,
             xml: str | None = None,
             base_pos=mjcf_generator.DEFAULT_BASE_POS,          # (0.220, -0.443, 1.042)
             base_yaw_deg: float = mjcf_generator.DEFAULT_BASE_YAW_DEG,      # 0.0
             joint_damping: float = mjcf_generator.DEFAULT_JOINT_DAMPING,    # 0.026
             joint_frictionloss: float = mjcf_generator.DEFAULT_JOINT_FRICTIONLOSS,  # 0.025
             tendon_damping: float = mjcf_generator.DEFAULT_TENDON_DAMPING,  # 1.0
             sim: SimParams | None = None,
             faults: dict[int, NodeFault] | None = None,
             precalibrated: tuple[int, ...] | None = None,      # None => ALL present nodes
             ids: tuple[int, ...] = K.NODE_IDS,                 # (1..24)
             absent: tuple[int, ...] = (),
             seed: int = 20260811,
             leak_fault_pa_per_s: float = LEAK_FAULT_PA_PER_S,
             tendon_damping_const: float | None = None,
             ideal_pressure_tau_s: float | None = None,
             timestep_s: float | None = None,
             node_logic_every: int | None = None,
             batched_actuator: bool = False,
             adc: AdcSpec | None = None,
             min_dwell_s: float = 0.0,
             foh_ramp: bool = False,
             ramp_span_s: float | None = None,
             node_hook=None) -> None:
```

Construction order (`sim_core.py:573-693`):

1. `self.actuator = actuator or ActuatorModel.default()` — the shipped `synthetic_pretrained.npz`.
2. If `xml is None`, run `mjcf_generator.generate_xml(...)`; `MjModel.from_xml_string`; **then** override `model.opt.timestep` if `timestep_s` given (raises `ValueError` if ≤ 0); `MjData`; `mj_forward`.
3. `self._dt = float(self.model.opt.timestep)` — **read back, never assumed** (`sim_core.py:593`). The generator pins `TIMESTEP_S = 0.001` and `integrator="implicitfast" iterations="100"` (`mjcf_generator.py:181-183, 396`).
4. `self.node_logic_every = int(NODE_LOGIC_EVERY if node_logic_every is None else node_logic_every)`; raises if < 1.
5. `self.lock = threading.RLock()` — **THE mutex**, reentrant (`sim_core.py:611`).
6. Actuator/tendon bookkeeping by NAME (`sim_core.py:614-633`): `_act_ids` from `mj_name2id(..., mjOBJ_ACTUATOR, f"pam_{a+1}")` for `a in range(24)`; `_ten_ids = model.actuator_trnid[a, 0]`; `_ten_len0 = model.tendon_length0[t]` (the compiler's qpos0 anchor); `_tendon_damping_base = model.tendon_damping[_ten_ids]` captured **before** any overwrite.
7. Node population (`sim_core.py:649-675`), **RNG stream identical to `FakeMaster`**: `self._rng = random.Random(seed)`, then per node in `ids` order, skipping `absent`: `ambient = rng.uniform(540.0, 640.0)`, `span = rng.uniform(1350.0, 1470.0)`. Per-node sensor RNG is `random.Random(seed + nid)`. With an `AdcSpec` the draws are remapped: `ambient = adc.ambient_counts + (ambient - 590.0)`, `span = adc.counts_per_psi * sim.ceiling_psi`.
8. Precalibration (`sim_core.py:676-682`): `node.zero_off = int(round(node.ambient))`, `node.scale = K.f32(K.scale_from_span(node.span_counts))` where `scale_from_span(s) = REF_PSI * PA_PER_PSI / s` with `REF_PSI = 40.0`, `PA_PER_PSI = 6894.757`.
9. Clock state: `_t_origin = None`, `_t_target = None`, `_accum = 0.0`, `quanta_done = 0`, `node_passes = 0`, `_last_node_quantum = 0`.

### 1.5 The clock — `advance_to` verbatim

```python
# sim_core.py:699-705
@property
def sim_now(self) -> float:
    if self._t_origin is None:
        return 0.0
    return self._t_origin + self.quanta_done * self._dt

# sim_core.py:707-735
def advance_to(self, t: float) -> float:
    t = float(t)
    with self.lock:
        if self._t_target is None:
            self._t_origin = t
            self._t_target = t
            return self.sim_now
        if t <= self._t_target:
            return self.sim_now
        self._accum += t - self._t_target
        self._t_target = t
        n = int(self._accum / self._dt + 1e-9)
        if n > 0:
            self._accum -= n * self._dt
            for _ in range(n):
                self._step_quantum()
        return self.sim_now
```

Four properties that together **are** the determinism contract:

- **First call pins the origin and consumes nothing.**
- **A target at or behind `_t_target` is a no-op** — this is what makes racing advancers order-independent.
- **The sub-quantum remainder accumulates across calls.** 1000 hops of 1.5 ms consume 1500 quanta, not 1000 (per-hop flooring would lose a third of wall time).
- **`+ 1e-9` quanta of forgiveness** so a sum that lands 1 ulp low does not consume `n-1`.

### 1.6 THE EXACT ORDER OF OPERATIONS INSIDE ONE 1 ms QUANTUM

```python
# sim_core.py:737-746
def _step_quantum(self) -> None:
    if self.quanta_done % self.node_logic_every == 0:
        self._node_pass()
    dlen = self.data.ten_length[self._ten_ids] - self._ten_len0
    self.data.ctrl[self._act_ids] = self.actuator.force_n(self._pressures_pa(), dlen)
    mujoco.mj_step(self.model, self.data)
    self.quanta_done += 1
```

Fully expanded, in order, lock held:

1. **If `quanta_done % node_logic_every == 0`** → `_node_pass()` (§1.7 below). Passes therefore fall at quanta 0, 5, 10, … — the *first* pass integrates `dt = 0`.
2. `dlen = data.ten_length[_ten_ids] - _ten_len0` — shape `(24,)`, metres.
3. `p = self._pressures_pa()` — `np.zeros(24)`, `p[nid-1] = node.p_pa`; **absent nodes read 0** (`sim_core.py:856-861`).
4. `data.ctrl[_act_ids] = actuator.force_n(p, dlen)` — the anchored McKibben law, **every quantum**, not every node pass:
   ```python
   # actuator_model.py:484-493
   l = self.l0_per_act + dlen_m
   f = self.coeff_per_act * p_pa * (self.bf2_per_act - 3.0 * l * l)
   return np.clip(f, -FORCE_CLIP_N, 0.0)      # FORCE_CLIP_N = 4000.0 N, pull-only
   ```
   `l0_per_act = np.repeat(l0, 8)`, `coeff_per_act = np.repeat(coeff, 8)`, `bf2_per_act = np.repeat(bf,8)**2` — per-segment params expanded 8× (`actuator_model.py:361-378`).
5. `mujoco.mj_step(model, data)` — one 1 ms physics step.
6. `quanta_done += 1`.

### 1.7 `_node_pass()` — the firmware grid, verbatim

```python
# sim_core.py:748-782
def _node_pass(self) -> None:
    node_dt = (self.quanta_done - self._last_node_quantum) * self._dt
    self._last_node_quantum = self.quanta_done
    now = self.sim_now
    dlen = self.data.ten_length[self._ten_ids] - self._ten_len0
    l = self.actuator.l0_per_act + dlen
    ldot = self.data.ten_velocity[self._ten_ids]
    if self.batched_actuator and node_dt > 0.0:
        self._batch_flows(l, ldot)
    for nid, node in self.nodes.items():
        i = nid - 1
        node.set_muscle_state(float(l[i]), float(ldot[i]))
        node.step(node_dt, now)
    if self.tendon_damping_const is None:
        self.model.tendon_damping[self._ten_ids] = \
            self.actuator.tendon_damping_n_s_m(
                self._pressures_pa(), base=self._tendon_damping_base, l_m=l)
    self.node_passes += 1
    if self.node_hook is not None:
        self.node_hook(now)
```

And inside each `node.step(dt, now)` the order is the parent's (`SimNode.step` at `sim_core.py:458-461` stamps the clock then calls `super().step`; `FakeNode.step` at `fake_arm.py:313-327`):

```python
def step(self, dt, now):
    self._integrate(dt)          # 1. plant
    self._sample_adc()           # 2. sensor: noise -> round -> clamp -> CalRecord
    if (self.controller_sel == 1 and self.ever_ticked and self.last_tick is not None
            and (now - self.last_tick) * 1000.0 > self.sim.failsafe_ms):
        self.failsafe_active = True     # 3. failsafe trap
        self.controller_sel = 0
        self._switch_state(0)
    if self.controller_sel == 1:
        self._bangbang()         # 4. regulate
```

So the **complete 1 ms tape** is:

```
quantum k:
  if k % 5 == 0:
     node_dt = (k - last_node_quantum) * dt          # 0 on the first pass, else 5 ms
     now     = t_origin + k*dt
     read l, ldot from mjData tendons (24-vectors)
     [optional] one 24-row flow-net forward -> per-node _net_flow_cache
     for each node:
        set_muscle_state(l[i], ldot[i])
        _stamp_now(now)                              # monotone max
        _integrate(node_dt):  bookkeep open time -> net = gain*f_theta(p,valve,l,ldot)
                              dp = net - leak;  p = max(0, p + dp*node_dt);  accrue air
        _sample_adc():        truncated-Gaussian noise, round to int count, clamp,
                              then pressure_pa = (raw - zero_off) * scale  (0 if uncal)
        failsafe check:       (now - last_tick)*1000 > failsafe_ms  -> trap
        _bangbang():          three-way compare of pressure_pa vs target_pa +- margin
     model.tendon_damping[ten_ids] = base + damp_b1 * p     (slack-gated)
     node_passes += 1;  node_hook(now)
  dlen = ten_length - ten_len0
  data.ctrl[act_ids] = clip(coeff * p * (bf^2 - 3 l^2), -4000, 0)
  mj_step(model, data)
  quanta_done += 1
```

### 1.8 Where noise / quantization / latency are injected — exhaustive list

| # | injection | where | magnitude / units |
|---|---|---|---|
| 1 | **Per-node zero offset** (fixed) | `sim_core.py:655` `rng.uniform(540.0, 640.0)` counts | ±50 counts ≈ ±1.4 psi at 12-bit |
| 2 | **Per-node span** (fixed) | `sim_core.py:656` `rng.uniform(1350.0, 1470.0)` counts over 40 psi | → `counts_per_psi` ≈ 33.75–36.75 |
| 3 | **ADC sample noise** | `fake_arm.py:243-244` (or `sim_core.py:278-280` with `AdcSpec`) | Gaussian sd `SimParams.noise_sd_counts` **default 1.9 counts**, **truncated at 3σ** |
| 4 | **ADC quantization + rail clamp** | `fake_arm.py:246` `int(max(0, min(K.ADC_MAX, round(raw))))` | `ADC_MAX = 4095` (12-bit); `AdcSpec.adc_max = 65535` for the V4 |
| 5 | **Cal gate** | `fake_arm.py:248-251` | `pressure_pa = 0.0` until a CalRecord exists |
| 6 | **CalRecord float32 rounding** | `K.f32()` on write, `struct.unpack("<hffh4x")` on read (`fake_arm.py:440-446`, `CAL_FMT = "<hffh4x"`) | scale is float32, not float64 |
| 7 | **Wire quantization of the reported pressure** | `fake_arm.py:407-417` `pressure_10pa()` → `int(pressure_pa/10)`, clamped `[0, 65534]` | **10 Pa LSB**, truncation not rounding |
| 8 | **Wire quantization of the setpoint** | `K.psi_to_10pa` → `int(round(psi*6894.757/10))`, clamp `[0, 65534]` | 10 Pa LSB; `0xFFFF` reserved = HOLD |
| 9 | **Bang-bang deadband** | `fake_arm.py:304-311`, `K.MARGIN_PA = 2000.0` | ±2000 Pa = ±0.29 psi hysteresis |
| 10 | **Node-logic latency** | `NODE_LOGIC_EVERY * dt` = **5 ms** | a setpoint delivered off-grid waits ≤ 5 ms for `_bangbang` on the bare-`SimArm` path |
| 11 | **Physics quantization** | `advance_to` consumes whole quanta; `sim_now` trails `_t_target` by < 1 quantum + remainder | ≤ 1 ms |
| 12 | **Sensor-fault noise (MGMT reads only)** | `fake_arm.py:383-405` `mgmt_raw()` — 12-periodic zero-mean sinusoid, amp `sensor_noise_sd*sqrt(2)*min(1, psi/20)` | models boards 9/17: burst sd 36–43 counts, 12-sample means within ±2 |
| 13 | **TDMA reply latency** | `fake_arm.py:582` `due = now + TICK_GAP_US/1e6 + slot * slot_us(baud)/1e6` | `TICK_GAP_US = 500`, `slot_us = 10*10*1e6/baud + 160 µs` = **360 µs at 500 kbaud**; 24-slot window ≈ **10.2 ms** |
| 14 | **MGMT RTT** | `fake_arm.py:85` `MGMT_RTT_S = 0.0015` — a real `time.sleep` | 1.5 ms, and `SimMaster.mgmt` holds the arm lock across it |
| 15 | **Reply / MGMT dropout** | `NodeFault.reply_dropout`, `.mgmt_dropout` ∈ [0,1] | Bernoulli per event, in the **shell**, not the plant |
| 16 | **Mocap position noise** | `sim_mocap.py:269-271` | 0.3 mm 3-D rms per plate, per-axis sd = `0.3mm/√3` |
| 17 | **Mocap base wobble** | `sim_mocap.py:281-284` | rotation-vector sd 0.0004 rad, common-mode on the base plate only |
| 18 | **Mocap fixed mount offsets** | `SimMocapParams.mount_offset_m = 0.0015` m, `mount_rot_rad = 0.02` | drawn once, constant for the run |
| 19 | **Mocap stream rate** | `mocap_constants.NOMINAL_RATE_HZ = 120.0` | producer thread, `wait()` not `sleep()` |

Note: `replay.Rollout.plates` is **truth with no measurement noise** — "replay feeds fits, not gates" (`replay.py:187-189`).

### 1.9 The mutex discipline (design D7)

Stated at `sim_core.py:9-21`. The rule that was **deliberately replaced**: fake_mocap's "never advance from the mocap thread". The new rule:

- `SimArm.lock` is a **reentrant** `RLock` so a consumer can hold it across advance-then-read.
- `advance_to` is callable **from any thread** and takes the lock itself.
- Determinism, not exclusion, is what makes interleaving safe: a stale target is a no-op and the state after reaching T is a pure function of T.
- Everything that reads mjData takes the lock: `q()` (`sim_core.py:937-940`), `body_pose(name)` (`942-949`), `tick()` (`888`).
- `psi_of(node_id)` at `sim_core.py:951-953` is the **one unlocked read** (a single float).
- `SimMaster` wraps `tick` and `mgmt` in `self.arm.lock`; `collect` deliberately is **not** wrapped (`sim_master.py:26-28`).
- `SimMocap.step` takes the lock for `advance_to` + pose copy, then does the noise/fault/publish work **outside** it (`sim_mocap.py:257-267`).

### 1.10 How targets go in and sensor values come out

**In** — `SimArm.tick` (`sim_core.py:867-897`):

```python
def tick(self, setpoints: dict[int, int] | None = None,
         now: float | None = None, flags: int = 0) -> None:
    with self.lock:
        t = self.sim_now if now is None else float(now)
        all_off = bool(int(flags) & K.TICK_FLAG_ALL_OFF)
        for nid, node in self.nodes.items():
            node.tick_seen(t)
            if all_off:
                node.all_off()
            else:
                sp = (setpoints or {}).get(nid, K.SETPOINT_HOLD)
                node.bus_setpoint(int(sp) & 0xFFFF)
```

`bus_setpoint` (`fake_arm.py:263-270`) is "gate for gate":

```python
if sp_10pa == K.SETPOINT_HOLD:   # 0xFFFF
    return
if not self.cal_valid():
    return
self.target_pa = float(sp_10pa) * 10.0
self.controller_sel = 1
```

`all_off` (`fake_arm.py:272-277`): `controller_sel = 0; target_pa = 0.0; _switch_state(0)`. `SimArm.all_off(now)` is `self.tick(now=now, flags=K.TICK_FLAG_ALL_OFF)` (`sim_core.py:935`), and `TICK_FLAG_ALL_OFF = 0x01`.

**Out**: `node.pressure_10pa()` (wire, 10 Pa, raw counts if uncalibrated), `node.status_bits()`, `node.raw`, `node.p_pa` (truth), `arm.psi_of(n)` (truth psi), `arm.q()` (12 rad), `arm.body_pose(name)`.

`status_bits()` (`fake_arm.py:419-434`), bit values from `arm_constants.py:70-79`:

```
ST_INFLATING   = 1<<0   act_state == 1
ST_VENTING     = 1<<1   act_state == 2
ST_FAILSAFE    = 1<<2   failsafe_active
ST_STAGED_IMAGE= 1<<3
ST_NO_CAL      = 1<<4   not cal_valid()   -> pressure field carries RAW COUNTS
ST_SENSOR_RAIL = 1<<5   raw <= 5 or raw >= 4090
ST_TICK_CRC    = 1<<6
ST_ENGAGED     = 1<<7   controller_sel == 1
```

### 1.11 Air accounting, dwell log, opt-in seams

`_accrue_air` (`sim_core.py:348-366`) charges `air_in_psi` only on `act_state == VALVE_INLET` with `dp>0`, `air_out_psi` on `VALVE_EXHAUST` as `|dp|`, and **charges nothing on the closed branch** (the net's non-zero closed flow is an artifact, not air the operator paid for). These are REALIZED changes, so a stuck vent costs nothing here while `outlet_open_s` still counts the coil.

- `air_row()` → `(air_in_psi, air_out_psi, valve_switches, inlet_open_s, outlet_open_s)` (`sim_core.py:487-491`).
- `SimArm.air_totals()` → `dict` of five `np.ndarray` of length 24, absent nodes 0 (`sim_core.py:838-854`).
- `enable_switch_log()` / `dwell_stats()` → `{"n", "min_s", "mean_s"}` over inter-switch gaps; the still-open final dwell is excluded (`sim_core.py:814-836`).
- `_switch_state` (`sim_core.py:370-384`) is **unconditional** — the dwell rule lives in the *decision* layer (`_bangbang`, `servo_switch`), never on the safety paths (`all_off`, failsafe trap).
- `servo_switch(want) -> bool` (`sim_core.py:393-403`): a change inside the dwell window is **refused, not queued**.
- FOH ramp: `bus_setpoint` treats the wire word as the ARRIVAL target and sets `ramp_rate_pa_s = |target - ref| / ramp_span_s`, capped so a staged slower rate can only slow it (`sim_core.py:416-437`).
- `_stamp_now` (`sim_core.py:439-451`) takes the **max** of the tick mark and `sim_now` — this is the exact mechanism that reconciles the asynchronous control-tick grid with the physics grid (see §6).
- `_integrate_ideal` (`sim_core.py:493-517`) uses the **exact ZOH solution** `p += (1-exp(-dt/τ))·(target-p)`, not explicit Euler: on a 5 ms grid with τ = 51 ms Euler's effective τ is 48.5 ms, a 5 % bias.

---

## 2. `UMArm_SIM/sim_master.py` — the transport shell

### 2.1 How the illusion is built

**By subclassing, not copying** (`sim_master.py:1-11`): `class SimMaster(FakeMaster)`. The TDMA reply grid, MGMT collision window, `collect` blocking semantics, fault plumbing, and every observation hook (`ticks`, `hold_seen`, `mgmt_collisions`, `max_concurrent_inlets`) are *the same code* the calibration suites already proved. The chain the design depends on: `ArmBus`/`SysidBus` cannot tell `SimMaster` from `FakeMaster`, and `FakeMaster` cannot be told from the real `vema_proto.Master`.

**Exactly three things change** (`sim_master.py:13-28`):

1. `self.nodes = self.arm.nodes` — the nodes are `SimNode`s.
2. `_advance()` is `arm.advance_to(time.monotonic())`.
3. `tick` and `mgmt` are wrapped in `self.arm.lock`.

### 2.2 Exact code

```python
# sim_master.py:103-128
def __init__(self, port: str = "SIM", baud: int = 115200, timeout: float = 0.05,
             *, config: SimConfig | None = None,
             faults: dict[int, NodeFault] | None = None,
             precalibrated: tuple[int, ...] = (1, 2),
             ids: tuple[int, ...] = K.NODE_IDS,
             absent: tuple[int, ...] = (),
             seed: int = 20260811) -> None:
    cfg = config if config is not None else SimConfig()
    super().__init__(port, baud, timeout, sim=cfg.sim, ids=(),
                     precalibrated=(), absent=(), seed=seed)
    self.arm = SimArm(actuator=cfg.actuator, xml=cfg.xml, ...,
                      sim=self.sim, faults=faults, precalibrated=precalibrated,
                      ids=ids, absent=absent, seed=seed,
                      leak_fault_pa_per_s=cfg.leak_fault_pa_per_s)
    self.nodes = self.arm.nodes
```

`ids=()` in the `super().__init__` is the trick: the parent builds the whole wire model (reply heap, slot window, seq/collision bookkeeping, observation lists) and resolves `self.sim`, but builds **no** `FakeNode`s. The arm builds the population instead — with the identical RNG stream, so a node here and its `FakeMaster` twin share sensor character for the same seed.

```python
# sim_master.py:134-163
def _advance(self) -> float:
    now = time.monotonic()
    self.arm.advance_to(now)
    self._record_concurrency()
    return now

def tick(self, setpoints=None, seq: int = 0, flags: int = 0, responders=None):
    with self.arm.lock:
        return super().tick(setpoints, seq, flags, responders)

def mgmt(self, target: int, op: int, payload: bytes = b"", timeout: float = 0.6):
    with self.arm.lock:
        return super().mgmt(target, op, payload, timeout)
```

`SimConfig` (`sim_master.py:65-90`) is a dataclass: `actuator`, `xml`, `base_pos`, `base_yaw_deg`, `joint_damping`, `joint_frictionloss`, `tendon_damping`, `sim: SimParams|None`, `leak_fault_pa_per_s`.

### 2.3 What the inherited `FakeMaster.tick` does (`fake_arm.py:532-583`) — CRITICAL

```python
sp = [K.SETPOINT_HOLD] * K.MAX_NODES          # 24 slots, all 0xFFFF
for node, val in (setpoints or {}).items():
    sp[node - 1] = val
now = self._advance()
self.ticks.append(TickRecord(now, seq & 0xFF, tuple(sp)))
self._last_tick_mono = now
all_off = bool(flags & K.TICK_FLAG_ALL_OFF)
answering = sorted(responders) if responders is not None else None

for nid, node in self.nodes.items():
    node.tick_seen(now)
    node.sample_now()                          # <-- ADC sample AT THE MARK
    if all_off:
        node.all_off()
    else:
        node.bus_setpoint(sp[nid - 1] & 0xFFFF)
        node.apply_now()                       # <-- regulate AT THE MARK
    if answering is not None and nid not in responders:
        continue
    slot = answering.index(nid) if answering is not None else nid - 1
    payload = bytes([nid, T_DATA,
                     node.pressure_10pa() & 0xFF, (node.pressure_10pa() >> 8) & 0xFF,
                     node.status_bits(), seq & 0xFF])
    if node.fault.reply_dropout and self._rng.random() < node.fault.reply_dropout:
        self.dropped_replies += 1
        continue
    due = now + K.TICK_GAP_US / 1e6 + slot * K.slot_us(self.baud) / 1e6
    heapq.heappush(self._pending, (due, nid, payload))
```

The comment in the source is the protocol statement: *"bus.cpp:handle_tick's order, and the order matters: sample at the mark, then act on the setpoint at the mark, then build the reply from what was just sampled."* Docstrings of `sample_now`/`apply_now` (`fake_arm.py:279-299`) explain why: at 160 Hz on a 5 ms node grid, a reading taken at the previous node pass is most of a cycle old, and a target left to the next node pass sits idle for 80 % of a cycle.

Responder-bitmap semantics also modelled: **every node applies its own setpoint on every tick, in or out of the responder set**, and a responder's slot is the *rank* of its bit among those set, not its device ID (so the window shrinks rather than thins).

`collect` (`fake_arm.py:588-616`) occupies the whole duration, wakes on the next scheduled reply capped at `MAX_SUBSTEP_S = 0.005`. `mgmt` (`617-679`) charges the collision if `now - _last_tick_mono < slot_window_s`, purges stale pending replies, sleeps `MGMT_RTT_S`, then dispatches ops (`OP_PING`, `OP_GET_VERSION` → `FW_TDMA_V2 = 0x02020000`, `OP_GET_SERIAL`, `OP_CAL_READ/WRITE`, `OP_GET_RAW`, `OP_REBOOT` → silence).

### 2.4 Deliberate omission

`--sim-speed` has **no equivalent** (`sim_master.py:41-44`): time-scaling a learned flow model but not gravity is not a faster version of the same physics. `joint_verification --sim-backend mujoco` refuses `--sim-speed != 1`. Fast offline runs are `replay.py`'s job.

---

## 3. `UMArm_SIM/replay.py` — deterministic offline rollouts

**No threads, no wall clock. Nothing sleeps, nothing reads the clock.** Two replays of the same trial are bit-identical.

### 3.1 CSV contract (pinned)

```python
# replay.py:77-83
CMD_HEADER = ["t_mono"] + [f"sp_{n:02d}" for n in range(1, N_NODES + 1)]   # 25 cols
REPLIES_HEADER = ["t_mono", "node", "pressure_10pa", "status", "seq"]
TAIL_DT_S = 0.05
_CLAMP_EPS_N = 1e-6                                                        # line 88
```

`load_cmd_csv(path) -> (t (K,), setpoints (K,24) int)` — rejects a wrong header (`ValueError`, message contains "header") and **non-strictly-increasing timestamps** (message contains "increasing"). `save_cmd_csv(path, t, setpoints) -> Path` writes `f"{t:.6f}"`. `load_replies_csv(path) -> {node: {"t","pressure_10pa","status","seq"}}`.

Recorded tick timestamps **are** the sysid time base — the 8–9 ms/tick Windows slip is in the data on purpose (`replay.py:5-7`).

### 3.2 The sampling order — advance → sample → apply

```python
# replay.py:257-267
for k in range(t.shape[0]):
    arm.advance_to(t[k])                 # first call pins the origin
    sample(k, t[k])
    arm.tick({nid: int(sp[k, nid - 1]) for nid in arm.nodes}, now=t[k])
    out_sp[k] = sp[k]
for j in range(n_tail):
    now = t[-1] + (j + 1) * TAIL_DT_S
    arm.advance_to(now)
    sample(t.shape[0] + j, now)
```

Row *k* of every trace is what the arm looked like **at the instant tick k hit the wire** — mirroring `bus.cpp` building the reply payload from the pre-tick sample. The tail keeps sampling with **no ticks**, so the failsafe trips `FAILSAFE_MS` in and traps whatever pressure is left; tail rows carry `setpoints = -1`.

### 3.3 `Rollout` shapes (`replay.py:170-202`)

| field | shape | units |
|---|---|---|
| `t` | `(K,)` | s |
| `setpoints` | `(K, 24)` int | 10 Pa, `-1` = no tick |
| `p_pa` | `(K, 24)` | Pa, plant truth |
| `p_rep_10pa` | `(K, 24)` int | wire 10 Pa |
| `status` | `(K, 24)` int | status byte |
| `q` | `(K, 12)` | rad |
| `plates` | `(K, 7, 3)` | m, **truth, no noise** |
| `ctrl_min_n` | scalar | N |

`save(path)` writes one `.npz` under those field names.

### 3.4 The −4000 N clamp assertion

```python
# replay.py:249-255
if assert_no_clamp and low <= -(FORCE_CLIP_N - _CLAMP_EPS_N):
    raise ClampViolation(...)
```

Rationale (`replay.py:45-50`): the generator's `ctrlrange`/`forcerange` is `"-4000 0"`, chosen so the clamp *never* engages inside the 25 psi envelope. A rollout that touches it is **silently different physics**, which is fatal for tuning. `assert_no_clamp=False` records `ctrl_min_n` without raising.

`replay_trial(trial_dir, *, arm=None, tail_s=0.0, assert_no_clamp=True, **simarm_kwargs)` — a **fresh arm per trial is the deterministic default**; replaying two trials on one arm would chain their state.

CLI: `python -m UMArm_SIM.replay <trial_dir> [--out rollout.npz] [--tail-s 2] [--no-clamp-check]`; `_report_rms` skips `ST_NO_CAL` replies (they carry raw counts, not pressure).

---

## 4. `UMArm_SIM/__init__.py` — what it actually exports

The task brief expected `TIMESTEP_S` here. **It is not.** `__init__.py` exports only:

```python
# __init__.py:49
__all__ = ["mjcf_generator", "actuator_model", "sim_core", "REPO_ROOT"]
```

and it does one runtime thing (`__init__.py:44-47`): prepends the repo root to `sys.path`, because the simulator imports sibling packages (`UMArm_ROBOT_CONTROL`, `UMArm_KINEMATICS`).

The real numeric constants live in the modules:

| constant | file:line | value | meaning |
|---|---|---|---|
| `TIMESTEP_S` | `mjcf_generator.py:183` | `0.001` | physics quantum, s. **Read back from `model.opt.timestep`, never assumed** |
| `FORCE_RANGE` | `mjcf_generator.py:185` | `"-4000 0"` | tendon motor clamp, N |
| `DEFAULT_JOINT_DAMPING` | `mjcf_generator.py:156` | `0.026` | N·m·s/rad |
| `DEFAULT_JOINT_FRICTIONLOSS` | `mjcf_generator.py:168` | `0.025` | N·m |
| `DEFAULT_TENDON_DAMPING` | `mjcf_generator.py:174` | `1.0` | N·s/m, the p = 0 base |
| `DEFAULT_BASE_POS` | `mjcf_generator.py:177` | `(0.220, -0.443, 1.042)` | m |
| `NODE_LOGIC_EVERY` | `sim_core.py:117` | `5` | quanta per node pass → 5 ms |
| `LEAK_FAULT_PA_PER_S` | `sim_core.py:163` | `3000.0` | Pa/s |
| `FORCE_CLIP_N` | `actuator_model.py:159` | `4000.0` | N |
| `N_NODES` | `actuator_model.py:133` | `24` | |
| `HIDDEN`, `N_IN` | `actuator_model.py:146-147` | `32`, `6` | net arch: `p, onehot(3), l, ldot` |
| `P_SCALE_PA` | `actuator_model.py:152` | `25.0 * 6894.757` | net input normalization |
| `DP_SCALE_PA_S` | `actuator_model.py:153` | `1.0e5` | net output scale, Pa/s |
| `L_SCALE_M`, `LDOT_SCALE_M_S` | `actuator_model.py:154-155` | `0.1`, `0.5` | m, m/s |
| `DAMP_B1_N_S_M_PER_PA` | `actuator_model.py:183` | `(9.9e-4,)*3` | N·s/m per Pa, per segment |
| `MAX_SUBSTEP_S` | `fake_arm.py:89` | `0.005` | FakeMaster's substep = the node grid |
| `MGMT_RTT_S` | `fake_arm.py:85` | `0.0015` | s |
| `PA_PER_PSI` | `arm_constants.py:219` | `6894.757` | |
| `REF_PSI` | `arm_constants.py:223` | `40.0` | two-point cal reference |
| `MARGIN_PA` | `arm_constants.py:67` | `2000.0` | bang-bang hysteresis |
| `FAILSAFE_MS` | `arm_constants.py:58` | `1000` | ms of silence before the trap |
| `FAILSAFE_SILENCE_S` | `arm_constants.py:63` | `1.3` | test-side silence |
| `ADC_MAX` | `arm_constants.py:104` | `4095` | |
| `SETPOINT_HOLD` | `arm_constants.py:39` | `0xFFFF` | |
| `MAX_SETPOINT_10PA` | `arm_constants.py:40` | `65534` | |
| `TICK_FLAG_ALL_OFF` | `arm_constants.py:46` | `0x01` | |
| `MAX_NODES` / `NODE_IDS` | `arm_constants.py:30-32` | `24` / `(1..24)` | |
| `CONTROL_TICK_HZ` | `arm_constants.py:182` | `160.0` | **the RS485 sync rate** |
| `RESPONDER_GROUPS` | `arm_constants.py:~191` | `4` | telemetry sweep takes 4 cycles → 40 Hz/node |
| `TICK_GAP_US` / `SLOT_GUARD_US` | `arm_constants.py:137-138` | `500` / `160` | |
| `BUS_BAUD` | `arm_constants.py:~127` | `500000` | |
| `CAL_FMT` | `arm_constants.py:100` | `"<hffh4x"` | 16-byte CalRecord |
| `NOMINAL_RATE_HZ` | `UMArm_MOCAP/mocap_constants.py:131` | `120.0` | mocap |

---

## 5. `UMArm_COLLAB/protocol.py` — what it actually is

**It is NOT the sim/real behavioural protocol.** It is the *collision-trial* protocol: windup → strike → push → retract, expressed as pressures. The sim/real behavioural protocol lives in `sim_core`/`sim_master`/`fake_arm` and in the tests.

That said, it is the best worked example of **driving a bare `SimArm` on a synthetic clock**, and two things in it matter for the port:

**(a) The transferable core is a pure function.**

```python
# protocol.py:284
def pressure_schedule(spec: StrikeSpec, t: float) -> dict[int, float]:
# protocol.py:326
def schedule_10pa(spec: StrikeSpec, t: float) -> dict[int, int]:
    return {nid: int(K.psi_to_10pa(psi))
            for nid, psi in pressure_schedule(spec, t).items()}
```

Its docstring states the wire rule the port must keep: *"Returns a value for EVERY node (0.0 where the schedule wants a muscle empty), never a partial dict — an omitted node on the wire means 'hold your current target', which during a retract is the difference between an empty arm and an arm still pushing."*

**(b) The synthetic-clock drive loop** (`protocol.py:632-641`, `log_dt = 0.005`):

```python
t = 0.0
arm.advance_to(0.0)
while t < total:
    sched = pressure_schedule(spec, t)
    arm.tick({nid: int(K.psi_to_10pa(v)) for nid, v in sched.items()})
    t += log_dt
    with arm.lock:
        reach.update(data)
        arm.advance_to(t)
        ... read contacts, gauge, q ...
```

Note `arm.tick(...)` with **no `now=`** — it defaults to `self.sim_now` (`sim_core.py:889`). `rehearse` (`protocol.py:432-497`) uses the same pattern at `dt = 0.005`.

Also relevant: `default_actuator()` (`protocol.py:155-160`) loads `UMArm_COLLAB/assets/bench_actuator.npz`, and its docstring warns that `SimArm`'s own default is the **synthetic placeholder** — "a trial run on it is not a trial run on this arm". The bench checkpoint had `leak_pa_s` re-derived from measured hold decay and `fill_gain` from the two places a real valve is seen wide open, because the campaign's ramp-only data could not see either and the twin filled **2.7–13× slower than the metal** as a result.

Other constants: `T_REACH=2.5, T_SETTLE=0.6, T_BIAS=0.5, T_WINDUP=1.5, T_STRIKE=2.2, T_RETRACT=1.5` s; `BAR_TO_PSI = {2.0: 29.0, 2.5: 36.3, 3.0: 43.5}`; per-actuator envelope `psi_max = 35.0`, pair sum `pair_max = 35.0`; `FACE_AZIMUTH_WARN_DEG = 30.0`. `condition_from_bar` **refuses** an over-envelope level rather than clipping, because clipping made 2.5 and 3.0 bar produce byte-identical setpoints under different labels.

---

## 6. The 150 Hz sync edge vs the 1 ms grid — the part you must get right

### 6.1 How the RS485 twin does it

There are **two independent clocks and they are deliberately not aligned**:

- **The physics grid**: `sim_now = t_origin + quanta_done * dt`, advanced only in whole quanta, trailing `_t_target` by < 1 quantum plus the accumulator remainder.
- **The control-tick mark**: the wall/replay time `t` passed to `tick(..., now=t)`, which is *whatever the controller's own timebase says* — `time.monotonic()` on the shell path, the recorded CSV timestamp on the replay path.

At 160 Hz the period is **6.25 ms = 6.25 quanta of 1 ms**, which is not an integer. `SimNode._stamp_now`'s docstring (`sim_core.py:439-451`) names this explicitly:

> *"Two grids write it: the bus tick, at the mark `t`, and the node pass, at the arm's grid time `sim_now`, which trails the last mark by up to one quantum (6.25 ms is 12.5 quanta of 0.5 ms, so the mark is routinely ahead). Taking the max keeps the dwell window and the ramp on a clock that never runs backwards, which would otherwise make a just-elapsed dwell look negative for one pass."*

So the reconciliation rule is exactly three lines:

```python
def _stamp_now(self, now: float) -> None:
    now = float(now)
    if now > self._now_s:
        self._now_s = now
```

**The sync edge is applied instantaneously and off-grid.** A tick at time `t`:
- writes `last_tick = t`, `ever_ticked = True`, `failsafe_active = False` (`tick_seen`);
- writes `target_pa`, `controller_sel` (`bus_setpoint`);
- on the **shell** path only, also runs `sample_now()` + `apply_now()` — an ADC sample and a bang-bang decision **at the mark**;
- does **not** advance physics. Physics moves only when someone calls `advance_to`.

The failsafe comparison therefore mixes the two clocks: `(now - self.last_tick) * 1000.0 > self.sim.failsafe_ms` where `now` is `sim_now` (grid) inside `_node_pass`, and `last_tick` is the off-grid mark. The ≤ 1 quantum skew is the whole error budget, and it is why the design accepts it.

### 6.2 What that means for the CAN arm at 150 Hz

150 Hz → **6.6667 ms period**. Options, in order of how much you should prefer them:

| choice | period in quanta | node grid | verdict |
|---|---|---|---|
| `dt = 1 ms`, `node_logic_every = 5` (today's) | 6.6667 — **not integer** | 5 ms | works, but the sync edge floats across the node grid; `_stamp_now`'s max() is mandatory |
| `dt = 1/1500 s = 0.6667 ms`, `node_logic_every = 5` | **10 exactly** | 3.333 ms | sync edge lands exactly on a quantum boundary every period; node passes at 0, 5 → 2 per control period, phase-locked |
| `dt = 1/3000 s = 0.3333 ms`, `node_logic_every = 10` | **20 exactly** | 3.333 ms | same node grid, finer physics; 2× the `mj_step` cost |
| `dt = 1 ms`, `node_logic_every = 1` | 6.6667 | 1 ms | matches the "direct valve" architecture pattern; costs 5× the node work unless you use `batched_actuator=True` |

`SimArm` already supports all of these without code changes: `timestep_s=` overrides the compiled quantum and `node_logic_every=` overrides the module constant per instance (`sim_core.py:565-566, 581-599`). The **warning attached to that seam** (`sim_core.py:110-116`) transfers directly: changing `node_logic_every` for a plant identified around a bang-bang "would silently desynchronize the sim from the trained actuator net", because `actuator_model.train_shooting` trains on the same 5 ms grid `fake_arm.MAX_SUBSTEP_S` uses. If your CAN flow model is identified at 150 Hz / 6.667 ms, **identify it on whatever node grid you will simulate on**, and say so in the checkpoint metadata (the `load()` unit-drift guard at `actuator_model.py:~530` already refuses a checkpoint whose scaling constants differ; add the node grid to that guard).

My recommendation for the port: **`timestep_s = 1/1500` (0.6667 ms), `node_logic_every = 5` → a 3.333 ms node grid, exactly 2 node passes per 150 Hz period, sync edge on a quantum boundary.** Then `_stamp_now`'s max() becomes a no-op safety net rather than load-bearing, and dwell/ramp arithmetic is exact.

### 6.3 Two populations, one arm — where each maps onto this design

The RS485 twin already contains, as opt-in seams, exactly the machinery the two CAN populations need:

**Top 8 (0x101–0x108), TLE92464 proportional DVP, coil-current codes.** These are not bang-bang. The design's seams for this are `foh_ramp` + `ramp_span_s` + `pending_ramp_rate_pa_s` (continuous setpoint slewing, `sim_core.py:405-437`), `servo_switch` (an externally commanded valve state, `393-403`), and `node_hook(now)` (`sim_core.py:606-607, 781-782`) — "the seam the direct-valve architecture's host servo runs in". For a genuinely proportional valve you replace the three-way `_bangbang` with a continuous map `current_code -> effective orifice area`, and `_integrate`'s `valve` input becomes a float rather than a member of `{0,1,2}`. That means the one-hot block of the flow net's input (`(valve==CLOSED, valve==INLET, valve==EXHAUST)`, `actuator_model.py:~440`) must become a scalar (or signed) command channel. **`N_IN = 6` changes, so every checkpoint's unit-drift guard fires — good.**

**Bottom 16 (0x109–0x118), legacy 7 mm solenoid pairs, PWM duty.** These stay bang-bang-shaped but with a duty ratio. `min_dwell_s` (`sim_core.py:219-226`) is exactly the seam: "Real switching solenoids give ~1 ms", enforced in the decision layer and **never on the safety paths**. A PWM duty is a dwell-constrained switching sequence; model it either as a fast `node_logic_every=1` grid with `min_dwell_s` set to the minimum on-time, or as an effective-area average if your CAN sync period is long compared to the PWM period.

**Two populations means two `AdcSpec`s and two flow-net input encodings.** `SimArm` today passes one `adc=` to every node (`sim_core.py:664-675`); for the CAN arm make it per-node (a dict keyed by id, or a per-population tuple). Same for `min_dwell_s`, `foh_ramp`, and `leak_fault_pa_per_s`.

---

## 7. THE PROTOCOL — every invariant the tests assert

These are the deliverable. Each is a line the CAN port must reproduce.

### 7.1 `test_sim_core.py` — clock and quantization

| # | invariant | test | line |
|---|---|---|---|
| C1 | 1000 hops of 1.5 ms consume **exactly 1500 quanta**, `sim_now == 1.5` to 1e-9. Per-hop flooring would consume 1000 and lose a third of wall time. | `test_accumulator_conserves_mgmt_hops` | 45-53 |
| C2 | A target **behind** the furthest one seen is a no-op; a target **equal** to it is a no-op. | `test_stale_target_is_noop` | 56-64 |
| C3 | The **first** `advance_to` pins the origin and consumes **zero** quanta; `sim_now` equals the pinned value exactly. | `test_first_advance_pins_origin_consumes_nothing` | 67-73 |
| C4 | Node logic runs at quanta 0, 5, 10, 15, 20 — 25 quanta ⇒ **5** passes; 30 ⇒ 6. And `NODE_LOGIC_EVERY * model.opt.timestep == 0.005`. | `test_node_logic_every_fifth_quantum` | 76-85 |
| C5 | **Two racing advancer threads with incommensurate hops (3.1 ms, 7.3 ms) plus a reader thread doing `q()` and `body_pose()` produce bit-identical `data.qpos` to one serial coarse advancer** — `np.array_equal`, not a tolerance. Also `p_pa`, `raw`, `act_state` per node must be exactly equal, and both must have consumed 800 quanta. | `test_two_thread_advance_is_deterministic` | 102-143 |

### 7.2 `test_sim_core.py` — the plant seam

| # | invariant | test | line |
|---|---|---|---|
| P1 | `p_pa` is clamped ≥ 0 after **every** substep and **stays** at 0 (a monster leak cannot drive it negative on a later step either). | `test_p_pa_clamped_nonnegative` | 151-163 |
| P2 | With the net zeroed, closed-valve flow is **exactly `-leak`, to the pascal** (`abs=1e-6`). Over 2.0 s the integrated time is **1.995 s**, not 2.0 — node passes at quanta 0, 5, …, 1995, and the first integrates dt = 0. | `test_leak_seam_arithmetic_exact` | 166-183 |
| P3 | `NodeFault(leak=True)` overrides the per-node scalar with `leak_fault_pa_per_s` (default `LEAK_FAULT_PA_PER_S = 3000.0`), not the model's. | `test_leak_fault_default_severity` | 186-188 |
| P4 | `raw == round(ambient + (p_pa/PA_PER_PSI) * counts_per_psi)` exactly, with noise disabled, through the **untouched** parent `_sample_adc`; and `pressure_pa == (raw - zero_off) * scale`. | `test_adc_coupling_pinned` | 191-201 |
| P5 | `stuck_vent` blocks flow but **not** honesty: `act_state == 2`, `status_bits() & ST_VENTING`, `p_pa` unchanged, and `outlet_open_s` keeps accumulating (> 0.4 s over a 0.5 s vent). | `test_stuck_vent_flow_blocked_status_honest` | 204-220 |

### 7.3 `test_sim_core.py` — firmware semantics survive subclassing

| # | invariant | test | line |
|---|---|---|---|
| F1 | **Cal gating**: an uncalibrated node ignores the bus setpoint entirely — `controller_sel == 0`, and `status_bits() & ST_NO_CAL`. A calibrated one gets `controller_sel == 1`, `target_pa == sp * 10.0`. | `test_cal_gating_hold_and_failsafe_transfer` | 228-238 |
| F2 | **0xFFFF hold**: `tick(None)` must not retarget a regulating node — `target_pa` and `controller_sel` unchanged. | same | 240-242 |
| F3 | **Failsafe**: ≥ `FAILSAFE_MS` (1000 ms) without a tick sets `failsafe_active`, `act_state == 0`, `status_bits() & ST_FAILSAFE`. **It is a trap, not a vent** — the air stays. | same | 243-248 |
| F4 | An **absent** node is not in `arm.nodes` and its actuator's `data.ctrl` is exactly `0.0`; a present pressurized neighbour's is `< 0.0`. | `test_absent_nodes_produce_no_force` | 251-258 |
| F5 | `SimArm`'s `precalibrated` default is **all nodes** (the shell's is `(1, 2)`). | `test_default_precalibrated_is_all_nodes` | 261-263 |
| F6 | One quantum in, `ctrl` of a pressurized node's motor equals `coeff[0] * p * (bf[0]**2 - 3*l0**2)` to `rel=1e-12`, and every other actuator is **exactly** 0.0. | `test_muscle_force_written_each_quantum` | 271-284 |
| F7 | `model.tendon_damping = base + damp_b1 * p` is refreshed at every node pass; an unpressurized muscle holds `base`. | `test_tendon_damping_follows_pressure` | 324-341 |

### 7.4 `test_sim_core.py` — closed loop and ALL_OFF

| # | invariant | test | line |
|---|---|---|---|
| L1 | Driving nodes 4 and 12 to 10 psi through the real tick/bang-bang path lands them in **9.0–11.0 psi**, and deflects the *verified agonist joints* q0 and q4 positive by > 0.02 rad. A healthy node vents to < 30 % of hold; a stuck node keeps > 80 %. | `test_bangbang_fill_vent_moves_the_right_joint` | 292-321 |
| A1 | `TICK_FLAG_ALL_OFF` is acted on **INSTEAD of the setpoint, in the same tick, on every node in the same instant**: `act_state == 0`, `controller_sel == 0`, `target_pa == 0.0` for all. | `test_all_off_shuts_both_valves_and_ignores_the_setpoint_in_the_same_tick` | 348-378 |
| A2 | Nothing re-opens a valve on its own afterwards (20 further all-off ticks, `act_state` stays 0), **and an ordinary tick with a real setpoint re-engages it** — ALL_OFF is a valve state, not a latch. | same | 380-389 |
| A3 | **Both valves shut means TRAPPED, not vented.** After ALL_OFF from 25 psi held 10 s, the drift's **sign** must match the model's own closed-branch prediction node-by-node, and its magnitude must be < 10× the prediction + 0.1 psi. The test asks the checkpoint what it predicts rather than pinning a sign, because the bench checkpoint **inflates** (−1.088 psi/s at 0 psi, zero-crossing near 17.5 psi, +0.228 psi/s at 25 psi ⇒ 25.15 → 27.36 psi over 10 s) while the shipped synthetic one drifts down. | `test_all_off_traps_the_air_it_finds_and_the_closed_branch_is_not_zero` | 392-452 |
| A4 | **Vent past the closed branch's fixed point BEFORE shutting the valves.** After a 3.0 s vent (`experiments.T_VENT`), the trapped pressure is < 1.5 psi and < 6 % of the start, and what is left **bleeds away rather than growing** (`after.max() <= trapped.max() + 1e-9`). | same | 454-486 |

### 7.5 `test_sim_master.py` — interface identity

| # | invariant | test | line |
|---|---|---|---|
| M1 | For `tick`, `collect`, `mgmt`, `parse_data`, `close`, `send_inner`: **parameter names in order, defaults, and kinds all equal `vema_proto.Master`'s.** `SimMaster.__init__` has `port="SIM"`, `baud=115200`, `timeout=0.05`. `parse_data` **is** the bench parser by inheritance and returns an equal result on a real frame. | `test_sim_master_interface_matches_vema_proto_master` | 57-76 |
| M2 | FakeMaster's keyword-only surface kept verbatim — `faults`, `precalibrated`, `ids`, `absent`, `seed` all KEYWORD_ONLY with FakeMaster's defaults; `config` present, `sim` **absent**; **`precalibrated` defaults to `(1, 2)`, NOT SimArm's all-nodes.** | `test_constructor_keyword_surface_pinned_to_fakemaster` | 79-93 |
| M3 | Same seed ⇒ **same `ambient` and `span_counts` as a `FakeMaster`, for all 24 nodes**, exact equality. | `test_same_seed_same_sensor_character_as_fakemaster` | 96-104 |
| M4 | `master.sim is master.arm.sim` — shell and arm share **one** `SimParams` object; `master.nodes is master.arm.nodes`. | `test_config_reaches_the_arm` | 107-112 |

### 7.6 `test_sim_master.py` — transport semantics

| # | invariant | test | line |
|---|---|---|---|
| T1 | An uncalibrated node ignores every setpoint: `act_state == 0`, `inlet_open_s == 0.0`, `p_pa == 0.0`, `controller_sel == 0`, `ST_NO_CAL` set. | `test_uncalibrated_node_ignores_every_setpoint` | 120-131 |
| T2 | Failsafe **traps**: after `FAILSAFE_SILENCE_S` (1.3 s) of no ticks, `failsafe_active`, `act_state == 0`, `ST_FAILSAFE`, and `true_psi > hot - 1.0` psi. | `test_failsafe_traps_pressure_rather_than_venting` | 134-147 |
| T3 | S9 hold: a raw all-hold frame leaves `target_pa` unchanged and is **visible** as such via `hold_seen()`. | `test_hold_semantics_resume_a_previous_target` | 150-159 |
| T4 | S5 made physical: MGMT inside the slot window **collides** — returns `None` and increments `mgmt_collisions`. On a quiet bus `ping` succeeds and `mgmt_collisions == 0`. | `test_mgmt_needs_a_quiet_bus` | 162-171 |
| T5 | **TDMA grid**: 24 pending replies, node `i+1` at index `i`, due at `t_tick + TICK_GAP_US/1e6 + i*slot_us/1e6` to `abs=1e-9`; a full-window `collect` returns **all 24, exactly once, in slot order**. | `test_tdma_reply_grid` | 173-191 |
| T6 | Transport faults live in the **shell**: an absent node answers nothing anywhere; `reply_dropout=1.0` loses every slot reply (and `dropped_replies >= res.ticks`); `mgmt_dropout=1.0` loses MGMT responses. | `test_transport_faults_absent_dropout` | 194-209 |
| T7 | The CalRecord round-trips through `struct` as **float32**: `rec.scale == K.f32(scale) != scale`. | `test_cal_write_read_back_is_float32` | 212-221 |
| T8 | E-stop sends a **zero tick then goes silent**: `ticks[-1].setpoints == (0,)*24`, then failsafe, `act_state == 0`, **and `true_psi > 1.0` — e-stop TRAPS pressure; venting is a separate act.** | `test_estop_sends_a_zero_tick_then_goes_silent` | 224-236 |
| T9 | S1 observability: `max_concurrent_inlets` and `concurrent_inlet_ids` are measurable on this shell (sampled once per `_advance`). Single-node drive ⇒ 1; two-node ⇒ 2 with ids `{1,2}`. | `test_max_concurrent_inlets_is_observable` | 239-249 |
| T10 | An `ArmBus` refusal (`SetpointRefused`) leaves the MuJoCo arm **untouched**: `master.ticks == []` and every `p_pa == 0.0`. | `test_armbus_refusals_never_reach_the_wire` | 252-261 |
| T11 | Closed loop through `ArmBus`: 8 psi on node 4 lands 7.0–9.0 psi and makes q0 the **largest-magnitude** joint, > 0.05 rad. | `test_bus_drive_moves_the_intended_joint` | 264-275 |

### 7.7 `test_sim_master.py` — replay

| # | invariant | test | line |
|---|---|---|---|
| R1 | **Two replays of the same trial are bit-identical** — `np.array_equal` on `p_pa`, `q`, `plates`, `status`. | `test_replay_is_deterministic_and_faster_than_wall` | 295-309 |
| R2 | **Replay must not pace**: two replays of an N-second trial together take **less** than N seconds of wall time. | same | 302-305 |
| R3 | The tail reproduces the failsafe trap: `len == n + round(tail_s/TAIL_DT_S)`, tail rows carry `setpoints == -1`, final status has `ST_FAILSAFE` and **neither** `ST_INFLATING` nor `ST_VENTING`, and pressure loses only leak-sized air (> `p_hot - 3000 Pa` over 2 s at ~700 Pa/s). | `test_replay_tail_reproduces_the_failsafe_trap` | 317-333 |
| R4 | `cmd.csv` round-trips; a **wrong header raises** (`match="header"`); **non-monotonic timestamps raise** (`match="increasing"`). | `test_replay_cmd_csv_round_trip` | 336-352 |
| R5 | `replies.csv` groups by node; `.npz` shapes are `p_pa (K,24)` and `plates (K,7,3)`. | `test_replay_trial_dir_and_replies_loader` | 355-369 |
| R6 | **A rollout sample touching the −4000 N clamp is an error, not a warning**: `ClampViolation` raised; with `assert_no_clamp=False` the contact is still recorded as `ctrl_min_n == -FORCE_CLIP_N`; and the shipped model never goes near it on a normal trial. | `test_replay_asserts_the_minus_4000_clamp` | 372-390 |
| R7 | (slow) `joint_verification --sim --sim-backend mujoco --joints 0` exits `EXIT_OK` with verdict `MATCH` on the **unmodified harness**. | `test_joint_verification_mujoco_smoke` | 398-415 |

---

## 8. Re-implementation checklist for the CAN arm

1. **One mutex, reentrant, on the arm object.** Everything that touches mjData takes it. `advance_to` takes it itself.
2. **`advance_to(t)` is the only way time moves**, it consumes whole quanta, it carries a float accumulator, it no-ops on a stale target, and its first call pins the origin. Give yourself `quanta_done` and `node_passes` as public counters so C1/C4 are first-class assertions.
3. **Read `model.opt.timestep` back; never assume it.**
4. **Node logic on an integer multiple of the quantum.** Compute `node_dt = (quanta_done - last_node_quantum) * dt` so the first pass is dt = 0 and no plant time is lost or double-counted (this is what makes P2's 1.995 s exact).
5. **Per-node step order is immovable**: integrate → sample ADC → failsafe check → regulate. Write it once and inherit it.
6. **Force to `data.ctrl` every physics quantum**, from `tendon_length - tendon_length0`; pressure-scheduled tendon damping only at the node grid (`implicitfast` makes any magnitude stable).
7. **The plant seam is three methods**: `true_psi` (the pinned ADC coupling), `leak_pa_s` (the fault override), `_integrate` (flow − leak, clamped ≥ 0 every substep). Nothing else about the node changes.
8. **Faults split by layer**: `stuck_vent` and `leak` at the plant seam; `reply_dropout`, `mgmt_dropout`, `absent` in the transport shell. Never mix them.
9. **The transport shell subclasses the real fake, it does not copy it.** Pin the constructor signature against the real driver's with `inspect.signature`, in a test.
10. **Same seed ⇒ same sensor character** as the non-MuJoCo fake, node by node. Draw ambient/span from the same RNG in the same order.
11. **Air accounting is not optional**: `air_in_psi`, `air_out_psi`, `valve_switches` beside `inlet_open_s`/`outlet_open_s`. It changes no dynamics and every plant carries it.
12. **Replay: advance → sample → apply, no clock, no threads, fresh arm per trial, and a hard clamp assertion.**
13. **Write the closed-branch drift test.** Ask the model what it predicts and check the plant against its own answer; do not pin a checkpoint's sign.

## 9. Design docs referenced but not in this worktree's scope

`docs/simulator_design.md` §2.1 (physics quantum, clamp), §2.2 (SimArm, plant seam), §2.3 (SimMaster shell, mocap table), §2.4 (replay), §3 (actuator/scene split), §4 (actuator model), §4.7 (air accounting), §4a (ring analysis, divergences), §5 (outer fit), §6 (test matrix), §9; decisions D1–D8, especially **D2** (shell copied verbatim), **D3** (offline tuning is replay's job), **D7** (one mutex, any-thread advance). Also `docs/arch_recon_design.md` §2/§3 (the seven opt-in seams) and `docs/collab_bench.md`. None of these were opened.

## KEY FILES
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\sim_core.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_ROBOT_CONTROL\fake_arm.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\test_sim_core.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\test_sim_master.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\sim_master.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\replay.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\actuator_model.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_ROBOT_CONTROL\arm_constants.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\mjcf_generator.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\sim_mocap.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_COLLAB\protocol.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\__init__.py

## GOTCHAS
- THE BARE-SimArm PATH AND THE SHELL PATH DO NOT SAMPLE THE SAME WAY. `FakeMaster.tick` (fake_arm.py:565-568) calls `node.tick_seen(now); node.sample_now(); node.bus_setpoint(...); node.apply_now()` — an ADC sample AND a bang-bang decision AT THE TICK MARK, mirroring bus.cpp:handle_tick. `SimArm.tick` (sim_core.py:888-897) calls ONLY `tick_seen` + `bus_setpoint`/`all_off` — no `sample_now`, no `apply_now`. So replay.py, UMArm_COLLAB.protocol.run_trial and experiments.run_condition (all of which drive a bare SimArm) see up to one full node period (5 ms) of extra latency between a setpoint arriving and the valve reacting, and their reply payloads come from the last node pass rather than from the mark. If you port only SimArm.tick you silently get the low-fidelity variant.
- `_node_pass` runs BEFORE `mj_step` inside the same quantum, and the first pass integrates `node_dt = 0`. Consequence asserted in test_sim_core.py:180-183: advancing 2.0 s integrates exactly 1.995 s of plant, because passes fall at quanta 0, 5, ..., 1995 and the pass at quantum 0 moves nothing. Any leak/drift arithmetic you validate against must use (N-1)*node_grid, not N*node_grid.
- `advance_to` uses `int(self._accum / self._dt + 1e-9)` — a deliberate 1e-9-quantum forgiveness (sim_core.py:726-730). Drop it and a target meaning 'exactly n quanta' consumes n-1 whenever the float sum lands 1 ulp low, which silently changes every trajectory. Symmetrically, `_DWELL_EPS_S = 1e-12` (sim_core.py:121) exists so a just-elapsed dwell window is not refused.
- `p_pa` MUST be clamped to >= 0 after EVERY substep, not once per node pass. A learned dp/dt integrated through a vent goes negative otherwise, and a McKibben cannot go below ambient. The clamp is inside `_integrate` (sim_core.py:345).
- THE LEARNED FLOW NET'S CLOSED-VALVE BRANCH IS NOT ZERO. A rack with both valves shut walks toward that branch's own fixed point, and WHICH WAY depends on the checkpoint: the bench fit INFLATES a settled 25.15 psi rack to 27.36 psi over 10 s of ALL_OFF; the shipped synthetic one drifts down. ALL_OFF/e-stop TRAPS pressure, never vents it. Vent for 3.0 s past the fixed point BEFORE shutting valves (test_sim_core.py:392-486). Never write a test that pins the drift's sign — ask the model what it predicts and compare.
- `stuck_vent` must flip ONLY the flow model's valve input, never `act_state`. The firmware does not know its exhaust is blocked, so `status_bits()` must still report ST_VENTING and `outlet_open_s` must keep accumulating energized time (sim_core.py:337-339, asserted at test_sim_core.py:204-220). Applied in TWO places — the scalar seam and `_batch_flows` (sim_core.py:805-806) — so a batched port must not forget the second.
- `_switch_state` is UNCONDITIONAL; the min-dwell rule lives in the decision layer (`_bangbang`, `servo_switch`). Put the dwell check inside `_switch_state` and the safety paths — `all_off` and the failsafe trap — become refusable, which is a safety defect (sim_core.py:370-384).
- The 150 Hz sync edge does NOT land on the 1 ms grid: 6.6667 ms is 6.6667 quanta. The control mark and `sim_now` are two clocks, and `SimNode._stamp_now` reconciles them by taking the MAX, monotonically (sim_core.py:439-451) — without it a just-elapsed dwell reads negative for one pass. Prefer `timestep_s = 1/1500` so a 150 Hz period is exactly 10 quanta.
- `node_logic_every` is a CONSTANT for the shipped rack on purpose (sim_core.py:110-116): the actuator net is trained by `train_shooting` on the same 5 ms grid `fake_arm.MAX_SUBSTEP_S` uses, so changing it silently desynchronizes sim numerics from train-time numerics. Identify your CAN flow model on whatever node grid you will simulate on, and put the grid in the checkpoint's unit-drift guard alongside p_scale/dp_scale/l_scale.
- `SimArm`'s `precalibrated` defaults to ALL present nodes; `SimMaster`'s defaults to `(1, 2)` — deliberately NOT inherited, because it pins FakeMaster's signature (sim_master.py:33-38, test_sim_master.py:79-93). Getting this backwards makes a rehearsal silently actuate nodes that should be cal-gated off.
- The wire's reported pressure is TRUNCATED, not rounded: `pressure_10pa()` does `int(self.pressure_pa / 10.0)` after clamping to [0, 65534] (fake_arm.py:407-417). And when uncalibrated it returns RAW ADC COUNTS with ST_NO_CAL set — every consumer must gate on the status bits (the JointMonitor/vent_arm lesson, cited at replay.py:31-33 and 296-300).
- `SimMaster.mgmt` holds the arm lock across a real `time.sleep(MGMT_RTT_S)` of 1.5 ms (sim_master.py:155-163). That is intentional — it delays the 120 Hz mocap thread by a fraction of its period, which is what a bus transaction costs everyone on the real rig — but it means a lock-contention profile will show 1.5 ms stalls that are physics, not a bug.
- `SimArm.psi_of` is the ONE read that does not take the lock (sim_core.py:951-953). Everything else that touches mjData or node state does. Do not add unlocked reads casually.
- `replay.Rollout.plates` is TRUTH with zero measurement noise; the mocap noise (0.3 mm/plate, 0.0004 rad base wobble, fixed mount offsets) exists only in `SimMocap.step`. Replay feeds fits; it is not a gate.
