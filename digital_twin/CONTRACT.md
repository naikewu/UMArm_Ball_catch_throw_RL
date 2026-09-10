# The CAN arm's twin — module contract

Every signature in this file is binding. It exists because the twin is being
built by several hands at once against a plant that differs from the one the
reference implementation was fitted to, and an interface agreed after the fact
is an interface nobody wrote to.

Read `reference/sim_core.md` §7 first: that list of invariants is the RS485
twin's sim/real protocol, and this port reproduces it wherever the transport
allows. Read `reference/actuator_net.md` §2 for the flow net it re-implements.
Where this file departs from those, the departure is stated with its cause.

---

## 0. What is different about this arm, and why the reference cannot be copied

The RS485 twin models **one** plant: 24 identical 7 mm solenoid pairs, bang-bang
regulated, on a TDMA serial bus. This arm is **two plants on one CAN bus**, and
three of the reference's load-bearing assumptions do not survive the change.

| | RS485 arm | CAN arm, `0x101`–`0x108` | CAN arm, `0x109`–`0x118` |
| --- | --- | --- | --- |
| valve | 7 mm solenoid pair | TLE92464 proportional (DVP) | 7 mm solenoid pair |
| firmware drive | PWM duty, fixed orifice | coil current, 0–182.7 mA | PWM duty |
| regulation | bang-bang, ±2000 Pa | onboard proportional loop | bang-bang |
| link loss | — | outputs drop 500 ms after the last sync edge | **no timeout at all** |
| pressure cal | — | 943.75 zero-counts, 60.78125 counts/psi | 754.4 and 56.14 |

**Departure 1 — the valve one-hot is replaced by the pressure error.** The
reference feeds the net a three-way one-hot of the firmware's `act_state`,
recovered from `ST_INFLATING`/`ST_VENTING` status bits. This bus has no such
bits: `tlelib.proto.CompactStatus` carries four flags — `ENABLED`, `OTA_ACTIVE`,
`COMMAND_SEEN`, `ERROR` — and none of them is a valve state. The valve state is
therefore **not observable on this arm**, and a feature that cannot be recorded
cannot be a training input. In its place the net takes the **commanded error**
`e = target − measured`, in Pa, which *is* recorded, is what both firmwares
actually act on, and is continuous — so it describes a proportional valve and a
bang-bang valve in the same coordinate instead of forcing the proportional one
into three discrete states it does not have.

**Departure 2 — a population bit, and per-population everything.** Anything that
reports one number across all twenty-four boards reports a number that describes
neither population. The net takes `is_tle` as an input, and the fitted per-node
scalars stay per-node. **The population is read from the variant byte the board
answered, never from its id range** — a TLE board legitimately sat at `0x114`
during the 2026-08 bench session.

**Departure 3 — the gain switch is smoothed.** The reference selects
`fill_gain`/`vent_gain` by a hard three-way branch on the valve state. With a
proportional valve there is no switch to branch on, and the shooting loss
back-propagates through this selection, so a step discontinuity at `e = 0` puts
a kink in the middle of the operating region. Here the selection is a logistic
blend in `e` whose width is the population's own regulation band. It reduces to
the reference's hard switch as that width goes to zero.

What is **kept** verbatim, because it was paid for there and is structural
rather than numerical: the net predicts `dp/dt` and nothing else; the leak is
fitted separately by least squares and subtracted *outside* the net at the plant
seam; the anchored McKibben force law is closed form and fully decoupled from
the net; the shooting/BPTT objective with the firmware regulator inside the
loop; the `p >= 0` clamp after every substep; the pull-only force clip; the
1 ms timestep with the node logic on a whole-quantum count.

---

## 1. Units, everywhere

| quantity | unit | note |
| --- | --- | --- |
| pressure, target, error | **Pa gauge** | psi only at a human boundary |
| ADC | raw counts | what the wire carries; convert with the board's own `NodeCal` |
| muscle length `l`, `l0` | m | anchored: `l = l0_seg + (ten_length − tendon_length0)` |
| `ldot` | m/s | MuJoCo `ten_velocity` |
| force | N | **negative is pull**; a McKibben cannot push |
| `q` | rad | 12 joints, `UMArm_KINEMATICS` order |
| time | s | `time.perf_counter()` on the bench, sim seconds in a rollout |

`PA_PER_PSI = 6894.757`.

---

## 2. `digital_twin/actuator_model.py`

Pure numpy. **Must not import torch** — `sim_core` steps this at 1 ms and a
rollout must run on a machine that has no CUDA.

```python
N_IN = 5                  # p, e, is_tle, l, ldot
HIDDEN = (64, 64)         # two tanh layers, linear output
P_SCALE_PA      = 30.0 * PA_PER_PSI   # 206842.71 -- the operator's per-line cap
E_SCALE_PA      = 30.0 * PA_PER_PSI   # same scale, so p and e are commensurate
DP_SCALE_PA_S   = 1.0e5
L_SCALE_M       = 0.1
LDOT_SCALE_M_S  = 0.5
FORCE_CLIP_N    = 4000.0
```

The feature row, in this order, is the contract:

```python
x = [ p_pa / P_SCALE_PA,
      (target_pa - p_pa) / E_SCALE_PA,
      is_tle,                             # 1.0 on variant 0x02, else 0.0
      l_m / L_SCALE_M,
      ldot_m_s / LDOT_SCALE_M_S ]
```

Parameters:

| name | shape | fitted by |
| --- | --- | --- |
| `W1,b1,W2,b2,W3,b3` | (5,64),(64,),(64,64),(64,),(64,1),(1,) | the trainer |
| `fill_gain`, `vent_gain` | (24,) each | the trainer, clipped to `[0.05, 20.0]` |
| `blend_width_pa` | (2,) — `[7mm, tle]` | fixed per population, not fitted |
| `leak_pa_s` | (24,) | separate least squares, clipped `[0, 2000]` |
| `coeff`, `bf`, `l0` | (3,) each, per segment | the outer fit |
| `damp_b1` | (3,) | `fit_bounce` |
| `is_tle` | (24,) bool | **from the recorded variant byte** |

Required methods:

```python
@classmethod
def fresh(cls, *, is_tle, seed=20260910) -> "ActuatorModel"
@classmethod
def load(cls, path) -> "ActuatorModel"          # allow_pickle=False; unit guard, see below
def save(self, path, meta=None) -> None
def gain(self, node_idx, e_pa) -> np.ndarray    # the logistic blend, below
def net_flow_pa_s(self, node_id, p_pa, target_pa, l_m, ldot_m_s) -> float
def net_flow_pa_s_batch(self, p_pa, target_pa, l_m, ldot_m_s) -> np.ndarray   # (24,)
def flow_pa_s(self, ...) -> float               # net_flow_pa_s minus leak
def force_n(self, p_pa, dlen_m) -> np.ndarray   # (24,), <= 0
def tendon_damping_n_s_m(self, p_pa, base, l_m) -> np.ndarray
```

The gain blend, verbatim:

```python
s = 1.0 / (1.0 + np.exp(-e_pa / blend_width_pa[pop]))     # 1 filling, 0 venting
g = s * fill_gain[node_idx] + (1.0 - s) * vent_gain[node_idx]
```

`blend_width_pa` defaults to `(2000.0, 6000.0)` — the 7 mm firmware's own
±2000 Pa hysteresis, and for the TLE boards the proportional band implied by
their dead zone and slew limit. Both are documented as assumptions to be
re-measured, not as fits.

`save` writes an npz plus a `meta` JSON string carrying every scale constant.
`load` **raises** unless `n_in`, `hidden`, `p_scale_pa`, `e_scale_pa`,
`dp_scale_pa_s`, `l_scale_m`, `ldot_scale_m_s` match the module exactly. A
checkpoint from a different normalisation reproduces its training set and
diverges everywhere else, which is the failure mode that looks most like success.

Force law, unchanged from the reference except that its constants are refitted:

```python
l = l0_per_act + dlen_m
f = coeff_per_act * p_pa * (bf2_per_act - 3.0 * l * l)
return np.clip(f, -FORCE_CLIP_N, 0.0)
```

---

## 3. `digital_twin/mjcf_generator.py`

`generate_xml(**tunables) -> str` and `build_model(...)`. The chain is
`UMArm_KINEMATICS.canarm_params` (`MEASURED is True`; the five u-joint-centre
gaps are 265.36 / 72.89 / 234.37 / 72.99 / 229.92 mm), the joint order and the
proximal composition are `fkine`'s with `order="yx"`, and `viz/mjcf_canarm.py`
is the display-side model already verified against `fkine` to 4.4e-16 m — start
from its geometry rather than re-deriving it.

Requirements:

* `<option timestep="0.001" integrator="implicitfast" iterations="100">`.
  `digital_twin.TIMESTEP_S` is `0.001` and `sim_core` reads
  `model.opt.timestep` back rather than assuming it.
* 24 tendon-driven actuators named **`pam_1` … `pam_24`**, where `pam_k` is
  board `0x100 + k`. Routing must reproduce the measured actuator map:
  `UMArm_KINEMATICS.canarm_actuators.MEASURED_JOINT_PAIRS` gives, per joint, the
  `(positive, negative)` board. Segment 1 is rotated +90° from the legacy table
  and the legacy table must not be used.
* Every tunable an argument with a named default, never a module constant read
  at call time: `joint_damping`, `joint_frictionloss`, `tendon_damping`,
  `link_density`, `plate_mass`, and the base pose.
* A `build_room_scene`-compatible seam: `sim_core` takes `xml=`, so the merged
  multi-robot scene is composed elsewhere and handed in.

---

## 4. `digital_twin/sim_core.py`

`CanNode` models the **firmware**, `SimNode` replaces **only the plant**, and
`SimArm` owns MuJoCo plus 24 nodes plus one `threading.RLock`. Two node
flavours, chosen by variant byte:

* `SevenMmNode` — bang-bang on `pressure_pa` against `target_pa ± MARGIN_PA`,
  **no link-loss timeout**: it holds its last target indefinitely, which is a
  safety property of the real boards and must be reproduced, not fixed.
* `TleNode` — the onboard proportional loop, and a **500 ms sync-loss failsafe**
  that drops the outputs. Read `firmware/tle/` for the actual control law
  (dead zone, slew limit, inlet/outlet open and max codes, the 120-code budget
  with 4 reserved for dither) rather than assuming a plain P controller.

The sync edge is the CAN arm's tick: at each edge every board latches its
filtered pressure and promotes its staged target. `NODE_LOGIC_EVERY` must make
the firmware's control period an integer count of 1 ms quanta.

`advance_to(t)` reproduces the reference's determinism contract exactly — the
first call pins the origin and consumes nothing; a target at or behind the
furthest one seen is a no-op; the sub-quantum remainder accumulates across
calls; `+ 1e-9` quanta of forgiveness. Two racing advancer threads with
incommensurate hops must produce bit-identical `data.qpos` against one serial
advancer. These are `reference/sim_core.md` §7.1 C1–C5 and they port unchanged.

The order inside one quantum is fixed:

```
if quanta_done % NODE_LOGIC_EVERY == 0: node_pass()
dlen = data.ten_length[ten_ids] - ten_len0
data.ctrl[act_ids] = actuator.force_n(pressures_pa(), dlen)
mj_step(model, data)
quanta_done += 1
```

and inside one node pass: integrate the plant, sample the ADC, check the
failsafe, then regulate — in that order, because the regulator must act on the
pressure the sensor reported, not on the true one.

---

## 5. `digital_twin/sim_master.py`

Wraps `TLE_PCB/tlelib/backend.py::Backend` rather than reimplementing it. The
property to preserve: **a controller must not be able to discriminate between
the real robot and the twin.** Same method names, same signatures, same units,
same `snapshot_nodes()` shape, same `set_cycle_observer` semantics, and the same
per-id reply-latency spread — 1.87 ms at `0x101` rising about 64 µs per id step
to 3.35 ms at `0x118`, which is CAN arbitration and is what a flat column would
give away.

---

## 6. `digital_twin/replay.py`

Offline rollouts: no threads, no wall clock, bit-identical repeats. Drives a
bare `SimArm` on a recording's own `can_sync_time_s` stamps. `assert_no_clamp`
raises `ClampViolation` if a rollout ever touches the force clip.

---

## 7. `digital_twin/dataset.py` and `train_actuator_net.py`

`dataset.py` reads a `session_*/` recording (see `data_schema.md`) into arrays,
splitting by population, and is pure numpy. `train_actuator_net.py` is the only
module allowed to import torch: it runs the shooting/BPTT objective on the GPU
with autograd, and its **only** output is a checkpoint `actuator_model.load`
accepts. A test must assert the numpy forward and the torch forward agree to
float64 tolerance on the same weights, because the two paths are what make the
trained model and the stepped model the same model.

---

## 8. The safety rule, which is not negotiable

For every antagonistic pair `(a, b)` and at every instant, commanded
`psi[a] + psi[b] <= 30.0`, and every individual `psi <= 30.0`. This is the
operator's rule for this arm. It is enforced in the target generator, asserted
again immediately before transmission, and any code path that can put a target
on the wire without passing that assertion is a defect.
