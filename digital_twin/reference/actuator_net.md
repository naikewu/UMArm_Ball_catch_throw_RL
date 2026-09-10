# ACTUATOR NET RECON — `UMArm_SIM/actuator_model.py` + `tune.py` + their tests

Worktree read: `C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision` (branch `feat/kmppi-collision`).

**Read completely:** `UMArm_SIM/actuator_model.py` (1160 lines), `UMArm_SIM/tune.py` (1578 lines), `UMArm_SIM/test_actuator_model.py` (465), `UMArm_SIM/test_tune.py` (548). Also dumped the shipped checkpoint with numpy, and read the relevant slices of `UMArm_ROBOT_CONTROL/arm_constants.py`, `UMArm_ROBOT_CONTROL/fake_arm.py` (`SimParams`, `FakeNode.step/_integrate/_bangbang/_sample_adc`), `UMArm_SIM/sim_core.py` (`SimNode._integrate`, `SimArm._node_pass`, `_batch_flows`, `_step_quantum`), `UMArm_SIM/replay.py` (`Rollout`, `rollout` signature), `UMArm_ROBOT_CONTROL/data_collection.py` (`TrialSpec` fields + `validate_spec`).

**NOT read** (say so explicitly): the bulk of `sim_core.py` (49.5 kB), `replay.py`, `mjcf_generator.py`, `force_audit.py`, `fit_bounce.py`, `twin_compare.py`, `sim_mocap.py`, `data_collection.py` beyond the spec/validation section, `docs/simulator_design.md` and `docs/data_collection_design.md` (referenced constantly by the code but not present in what I opened), and every test file other than the two named.

---

## 1. Architecture at a glance

There is **one** learned object: a 6→32→32→1 MLP called the **flow net**, shared by all 24 nodes, predicting **dp/dt** (pressure derivative), in plain numpy with hand-written forward/backward. Everything else is closed form:

```
dp/dt (node i) = gain_i(valve) * DP_SCALE_PA_S * f_theta(normalized[p, onehot(valve), l, ldot])
                 - leak_i                                   # leak OUTSIDE the net
p <- max(0, p + dp/dt * h)                                  # clamp, every substep
F  (node i)    = clip( coeff_seg * p * (Bf_seg^2 - 3 l^2), -4000, 0 )   # anchored McKibben, pull-only
tendon_damping = max(0, base + damp_b1_seg * p)  gated to 0 where 3l^2 <= Bf^2
```

Parameter inventory:

| group | shape | trained by | notes |
|---|---|---|---|
| `W1,b1,W2,b2,W3,b3` | (6,32),(32,),(32,32),(32,),(32,1),(1,) | warm start + shooting | **1313 scalars total**, shared by all 24 nodes |
| `fill_gain`, `vent_gain` | (24,) each | shooting only (`train_gains=True`) | per-actuator, index = `node_id - 1` |
| `leak_pa_s` | (24,) | **never** by the net — separate LSQ fit | per-actuator |
| `coeff`, `bf`, `l0` | (3,) each | outer scipy fit (`tune.outer_fit`) | **per-segment**, expanded `np.repeat(x, 8)` |
| `damp_b1` | (3,) | not fitted here — `fit_bounce.py` | per-segment |

---

## 2. `actuator_model.py` — exhaustive

### 2.1 The net (`class FlowNet`, line 191)

```python
HIDDEN = 32
N_IN = 6                                   # p, onehot(3), l, ldot      (line 146-147)

PARAM_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")
shapes = {"W1": (N_IN, HIDDEN), "b1": (HIDDEN,),
          "W2": (HIDDEN, HIDDEN), "b2": (HIDDEN,),
          "W3": (HIDDEN, 1), "b3": (1,)}          # actuator_model.py:203-205
```

Forward (`FlowNet.forward`, line 234) — two `tanh` hidden layers, **linear** output, no output activation, no dropout, no batchnorm:

```python
h1 = np.tanh(x @ self.W1 + self.b1)
h2 = np.tanh(h1 @ self.W2 + self.b2)
return (h2 @ self.W3)[:, 0] + self.b3[0]
```

Init (`FlowNet.fresh`, line 213), seeded `np.random.default_rng(seed)`:
- `W1 ~ N(0, sqrt(2/(6+32)))`, `W2 ~ N(0, sqrt(2/64))` (Xavier/He-ish),
- `W3 ~ N(0, 0.05)` deliberately small so an untrained net produces near-zero flow,
- all biases zero.

`forward_full` (line 240) also returns `dy/dx0` (sensitivity to the **p** input) by a forward-mode chain, because the shooting BPTT needs it at every substep for the state Jacobian:

```python
t1 = (1.0 - h1 * h1) * self.W1[0, :]           # (B, H)
t2 = (1.0 - h2 * h2) * (t1 @ self.W2)          # (B, H)
dy_dx0 = t2 @ self.W3[:, 0]                    # (B,)
```

Backward (line 258) accumulates `d(sum(u*y))/dtheta` into a caller-owned grads dict:

```python
grads["W3"] += h2.T @ u[:, None]
grads["b3"] += u.sum(keepdims=True)
dh2 = (u[:, None] @ self.W3.T) * (1.0 - h2 * h2)
grads["W2"] += h1.T @ dh2
grads["b2"] += dh2.sum(axis=0)
dh1 = (dh2 @ self.W2.T) * (1.0 - h1 * h1)
grads["W1"] += x.T @ dh1
grads["b1"] += dh1.sum(axis=0)
```

`input_grad` (line 271) returns `dh1 @ self.W1.T` — used **only** by the gradient-check test.

### 2.2 Optimizer (`class Adam`, line 280)

Textbook Adam over a dict of arrays, Kingma–Ba defaults `lr=1e-3, betas=(0.9,0.999), eps=1e-8`, bias-corrected, in-place `p -= ...`. No weight decay, no gradient clipping, no LR schedule anywhere in the file. Because it mutates the arrays it was handed (`model.net.params()` returns the live arrays via `getattr`), constructing the optimizer is what binds it to the model.

### 2.3 Input feature vector — exact (`ActuatorModel._norm_x`, line 410)

**6 features, single time step, NO history stacking at all.** The temporal information enters only through `ldot` and through the shooting rollout's recurrence.

```python
return np.column_stack((
    p_pa / P_SCALE_PA,                                   # col 0: gauge pressure, Pa -> ~[0,1]
    (valve == VALVE_CLOSED).astype(np.float64),          # col 1
    (valve == VALVE_INLET).astype(np.float64),           # col 2
    (valve == VALVE_EXHAUST).astype(np.float64),         # col 3
    l_m / L_SCALE_M,                                     # col 4: anchored muscle length, m
    ldot_m_s / LDOT_SCALE_M_S,                           # col 5: muscle rate, m/s
))
```

Normalization constants, lines 152-155 (these are **hashed into the checkpoint** and re-checked on load):

```python
P_SCALE_PA     = 25.0 * K.PA_PER_PSI    # = 172368.925 Pa  (the 25 psi sysid cap)
DP_SCALE_PA_S  = 1.0e5                  # 100 kPa/s; measured fast-fill ~275 kPa/s -> y in ~[-3,3]
L_SCALE_M      = 0.1                    # muscle lengths ~0.08-0.17 m
LDOT_SCALE_M_S = 0.5
```

Valve codes (lines 138-140), numerically identical to firmware `act_state`: `VALVE_CLOSED=0`, `VALVE_INLET=1` (V_IN), `VALVE_EXHAUST=2` (V_OUT). `l` uses the anchored convention `l = l0_seg + (ten_length - tendon_length0)` — the SAME number fed to the force law.

**No per-node identity is given to the net.** Node character enters only through the three per-node scalars.

### 2.4 Output and post-processing

Output: **one scalar `y`, dimensionless**, interpreted as `dp/dt` in units of `DP_SCALE_PA_S`:

```python
def net_flow_pa_s(self, node_id, p_pa, valve, l_m, ldot_m_s) -> float:   # line 432
    x = self._norm_x(p_pa, valve, l_m, ldot_m_s)
    y = self.net.forward(x)[0]
    g = self._gain(np.array([node_id - 1]), np.array([valve]))[0]
    return float(g * y * DP_SCALE_PA_S)

def flow_pa_s(self, ...):                                                # line 476
    return self.net_flow_pa_s(...) - float(self.leak_pa_s[node_id - 1])
```

Per-node gain selection (`_gain`, line 427): **multiplicative, valve-conditional**:

```python
np.where(valve == VALVE_INLET, self.fill_gain[node_index],
         np.where(valve == VALVE_EXHAUST, self.vent_gain[node_index], 1.0))
```
i.e. the closed state has gain fixed at **1.0** — it is not a fitted degree of freedom.

Leak is subtracted **outside** the model at the plant seam so a fault can override it (`sim_core.py:300-305, 340-345`):

```python
net = self.actuator.net_flow_pa_s(self.node_id, self.p_pa, valve, self._l_m, self._ldot_m_s)
dp = net - self.leak_pa_s
self.p_pa = max(0.0, self.p_pa + dp * dt)
```

`net_flow_pa_s_batch` (line 443) is the identical math for a whole rack in one `(24,6)` GEMM — measured **381 us → 21.7 us** per tick. Docstring warns the two paths agree to a few ulp, not bitwise, because BLAS picks a different summation order.

### 2.5 Anchored McKibben force law and how the net composes with it

The net and the force law are **completely decoupled**: no residual, no multiplicative gate, no shared parameters. The net produces pressure; the force law consumes pressure. Their only coupling is that both use the *same* `l` (and the outer fit's `l0` therefore shifts the net's input distribution — see gotchas).

```python
def force_n(self, p_pa, dlen_m):                             # line 484
    l = self.l0_per_act + np.asarray(dlen_m, dtype=np.float64)
    f = self.coeff_per_act * np.asarray(p_pa) * (self.bf2_per_act - 3.0 * l * l)
    return np.clip(f, -FORCE_CLIP_N, 0.0)                    # FORCE_CLIP_N = 4000.0 N
```

Constants (`MUSCLE`, lines 99-104) — copied verbatim from `C:/RUNZE_SRC/UMArm_maxi_collab/sim/mk5_generator.py`, **already ×0.8 prescaled by the legacy code, never rescale**:

```python
MUSCLE = {
    "Bf":         (0.1304, 0.132,  0.1208),     # m
    "Nf":         (0.688,  0.808,  0.8),        # turns (dimensionless)
    "LBase":      (0.1736, 0.1544, 0.1472),     # m
    "LActOffset": (0.0664, 0.0576, 0.052),      # m
}
```
Inits (lines 338-343): `coeff = 1/(4*pi*Nf^2)` (Chou–Hannaford denominator) → `(0.16811763, 0.12188981, 0.1243398)`; `bf = MUSCLE["Bf"]`; `l0 = LBase - LActOffset` → `(0.1072, 0.0968, 0.0952)` m.

`BF_PRETUNE_SCALE = 1.16` (line 131) is applied **only in `pretrain_synthetic`**, not in `fresh()`. Rationale in a 25-line docstring: with verbatim `Bf` the anchored law goes slack at `l = Bf/sqrt(3)` and the equilibrium is geometry-limited, driving distal joints to 36-43° at 8 psi against a 30° mechanical trip; ×1.16 shrinks the contraction-to-slack margin to ~55 % giving 5-21°. `Bf` (not `l0`) carries the pre-tune because scaling `l0` moves the net's `l` input outside its synthetic training grid and the extrapolation inflated idle bystanders ~0.35 psi/trial. **`Bf` never enters the net.**

Pressure-scheduled damping (`tendon_damping_n_s_m`, line 382): `max(0, base + damp_b1_seg * p)`, with the pressure term zeroed where `3l² <= Bf²` (same slack boundary as the force clip) because MuJoCo tendon damping is bilateral and a slack muscle must not push. `DAMP_B1_N_S_M_PER_PA = (9.9e-4, 9.9e-4, 9.9e-4)` N·s/m per Pa (line 183), fitted on flight recording `20260816_222851_real_rec1` by `fit_bounce.py`: 24 clean-ring episodes, real f = 1.83 Hz, ζ = 0.054 median; twin gives ζ = 0.063 against a real IQR 0.047–0.073.

### 2.6 Hand-written backward pass through the shooting rollout (`_shoot`, line 784)

This is the piece that most needs faithful porting. Forward: integrate the batch at ~5 ms with the **real bang-bang and a simulated ADC in the loop**, matching pressures at reply instants.

```python
for k in range(kn - 1):
    tgt = batch.target_pa[:, k]; lk = batch.l[:, k]; ldk = batch.ldot[:, k]; h = batch.h[:, k]
    for _ in range(batch.nsub[k]):
        valve = act
        x = model._norm_x(p, valve, lk, ldk)
        y, h1, h2, dydx0 = net.forward_full(x)
        f = y * DP_SCALE_PA_S
        g = np.where(valve == VALVE_INLET, fill_g,
                     np.where(valve == VALVE_EXHAUST, vent_g, 1.0))
        p_new = p + (g * f - leak) * h
        clamped = p_new < 0.0
        p_new = np.where(clamped, 0.0, p_new)
        if want_grads: tape.append((x, h1, h2, dydx0, f, g, h, clamped, valve))
        p = p_new
        # FakeNode.step order: integrate with the old state, THEN sample ADC, THEN bang-bang
        raw   = np.clip(np.rint(batch.ambient + p / K.PA_PER_PSI * batch.cpp), 0, K.ADC_MAX)
        p_rep = (raw - batch.zero_off) * batch.scale
        act = np.where(p_rep > tgt + margin_pa, VALVE_EXHAUST,
                       np.where(p_rep < tgt - margin_pa, VALVE_INLET, VALVE_CLOSED))
    p_sim[:, k + 1] = p

resid = (p_sim[:, 1:] - batch.p_rec[:, 1:]) / P_SCALE_PA
loss  = float(np.mean(resid ** 2))
```

Backward (adjoint / BPTT), lines 839-861:

```python
w = 2.0 / (b * (kn - 1) * P_SCALE_PA ** 2)
lam = np.zeros(b); idx = len(tape) - 1
for k in reversed(range(kn - 1)):
    lam = lam + w * (p_sim[:, k + 1] - batch.p_rec[:, k + 1])
    for _ in range(batch.nsub[k]):
        x, h1, h2, dydx0, f, g, h, clamped, valve = tape[idx]; idx -= 1
        lam_eff = np.where(clamped, 0.0, lam)                 # clamp subgradient = 0
        net.backward(x, h1, h2, lam_eff * g * h * DP_SCALE_PA_S, grads)
        gh = lam_eff * f * h
        m = valve == VALVE_INLET
        if m.any(): np.add.at(gfill, batch.node_idx[m], gh[m])
        m = valve == VALVE_EXHAUST
        if m.any(): np.add.at(gvent, batch.node_idx[m], gh[m])
        dfdp = dydx0 * (DP_SCALE_PA_S / P_SCALE_PA)
        lam = lam_eff * (1.0 + g * h * dfdp)                  # state Jacobian
```

Key modelling decisions in the gradient:
- The **bang-bang and ADC quantization decisions are held fixed from the forward pass** — they are piecewise constant in θ, so this is the exact gradient almost everywhere (module docstring lines 44-47).
- The **`p >= 0` clamp uses subgradient zero** (`lam_eff`).
- **Leak, `coeff`, `bf`, `l0`, `damp_b1` receive no gradient here.**
- Returns `(net_grads, d_fill_gain, d_vent_gain)`.

### 2.7 Gradient check

Two tests, both in `test_actuator_model.py`:

1. `test_gradient_check_mlp` (line 36) — central differences `h = 1e-6`, random projection `L = sum(r*y)` over a `(5,6)` batch; 8 random entries per parameter array; **`rel_err < 1e-6`**. Also checks `input_grad` at four `(batch, feature)` positions, and asserts `input_grad(x, ones)[:, 0] == dy_dx0` to `rtol=1e-12`.
2. `test_gradient_check_shooting_bptt` (line 76) — same central-difference scheme on the full `_shoot` loss, 4 entries per array, **`rel_err < 1e-4`**; plus one finite-difference check of `gfill[0]`. It sets `leak_pa_s = 0` to keep `p` away from the 0-clamp kink and sets the target to 35 psi (unreachable) so the bang-bang never switches inside the horizon — away from switching boundaries the held-fixed-valve treatment IS exact.

### 2.8 Batching (`_Batch`, line 730; `_prepare_batch`, line 756)

Traces are column-stacked; **every trace in a batch must share the same tick count** (`ValueError("all traces in one shooting batch must share a tick count")`). Timestamps may differ per trace (the recorded Windows slip is intentionally kept). Substeps:

```python
nsub = np.maximum(1, np.rint(dt / substep_s).astype(int)).max(axis=0)   # (K-1,) batch max
h    = dt / nsub[None, :]                                               # (B, K-1) per-trace
```
so each trace lands exactly on its own reply instants with ≈5 ms steps. Timestamps must be strictly increasing.

Batch fields (all `(B,K)` unless noted): `node_idx (B,)`, `t`, `p_rec`, `target_pa`, `l`, `ldot`, `ambient (B,)`, `cpp (B,)` counts/psi, `zero_off (B,)`, `scale (B,)`, `p0 (B,)`, `nsub (K-1,)`, `h (B,K-1)`.

**Note: `TickTrace.valve` is NOT copied into `_Batch`** — the shooting loss never uses the recorded valve labels; the in-loop bang-bang regenerates them. The labels are used only by the warm start and the interval filter.

### 2.9 The two training objectives

**Warm start** — `build_warmstart_dataset` (line 669) + `train_warm_start` (line 703). Single-interval Δp regression:
- interval filter (§2.10) first;
- input `X` = `_norm_x(p_mid_smoothed, agreed valve, l_mid, ldot_mid)` where `p_mid = 0.5*(smooth3(p)[k] + smooth3(p)[k+1])`;
- target `y = ((dp/dt) + leak_node) / gain / DP_SCALE_PA_S` — i.e. exactly what `f_theta` must return so `gain*f - leak` reproduces the slope;
- **full-batch Adam, `epochs=300` default (400 in pretrain), `lr=3e-3`**, MSE `mean(resid²)`, upstream `2*resid/n`;
- **gains held fixed** (the warm start cannot separate node gain from shared net scale).

**Shooting (primary)** — `train_shooting` (line 864):
```python
def train_shooting(model, traces, *, epochs=120, lr=1e-3, gains_lr=1e-2,
                   substep_s=0.005, margin_pa=None, train_gains=True) -> list[float]
```
- `margin_pa` defaults to `K.MARGIN_PA = 2000.0` Pa;
- **full batch, one `_shoot` per epoch**, two separate Adams (net `lr=1e-3`, gains `lr=1e-2`);
- leak stays fixed;
- after each gains step: `np.clip(model.fill_gain, 0.05, 20.0, out=...)` and the same for `vent_gain`;
- returns the per-epoch loss list.

**No early stopping, no validation split, no LR schedule, no regularization anywhere.** The only "stop if worse" logic lives one level up in `tune.py`'s guard.

`rollout(model, trace, substep_s=0.005, margin_pa=None)` (line 896) is the evaluation primitive: `_shoot` with `want_grads=False`, returns `p_sim[0]`, shape `(K,)`.

### 2.10 Leak fitting and interval filtering

```python
def filter_single_intervals(t, p, valve, *, sigma_pa, leak_max_pa_s) -> np.ndarray:   # line 608
    same = valve[:-1] == valve[1:]
    lim = 3.0 * sigma_pa
    ok_closed = (state == VALVE_CLOSED) & (np.abs(dp) <= lim + leak_max_pa_s * dt)
    ok_fill   = (state == VALVE_INLET)  & (dp >= -lim)     # a filling muscle cannot lose pressure
    ok_vent   = (state == VALVE_EXHAUST)& (dp <=  lim)     # a venting muscle cannot gain it
    return same & (ok_closed | ok_fill | ok_vent)
```

```python
def fit_leak(t, p, *, min_duration_s=2.0) -> float:        # line 635
    if len(t) < 3 or t[-1] - t[0] < min_duration_s: raise ValueError("...too short...")
    slope = np.polyfit(t, p, 1)[0]
    return float(-slope)                                    # positive = decaying
```
`fit_leak_segments` (line 652) = **duration-weighted mean** over several segments of one node.

Why the leak is not learned (module docstring lines 16-21): per-interval leak ≈ **36 Pa / 50 ms**, below one ADC count and below sensor noise; holds are limit cycles whose closed-conditioned mean Δp is ≈0.

`smooth3` (line 599): centered 3-sample mean, edges untouched.

### 2.11 `TickTrace` (line 555) — the training-data record

```python
node_id: int            # 1..24, validated
t: np.ndarray           # (K,) recorded tick/reply timestamps, seconds
p_rec: np.ndarray       # (K,) reported pressure, Pa (wire 10 Pa units x 10)
target_pa: np.ndarray   # (K,) setpoint applied from tick k on, Pa
valve: np.ndarray       # (K,) 0/1/2 from ST_INFLATING/ST_VENTING at reply instants
l: np.ndarray           # (K,) anchored muscle length, m
ldot: np.ndarray        # (K,) muscle rate, m/s
ambient: float          # counts at 0 psi gauge
counts_per_psi: float
zero_off: int           # CalRecord
scale: float            # CalRecord, Pa/count (float32-rounded)
p0: float               # initial true pressure, Pa
```
`__post_init__` enforces every array is exactly `(K,)` and `1 <= node_id <= 24`.

### 2.12 Checkpoint format — `synthetic_pretrained.npz`, verified by loading it

`np.savez` (line 505), **`allow_pickle=False` on load**. Actual contents of the shipped file:

| key | shape | dtype | shipped value |
|---|---|---|---|
| `W1` | (6, 32) | float64 | learned |
| `b1` | (32,) | float64 | learned |
| `W2` | (32, 32) | float64 | learned |
| `b2` | (32,) | float64 | learned |
| `W3` | (32, 1) | float64 | learned |
| `b3` | (1,) | float64 | learned |
| `fill_gain` | (24,) | float64 | **all 1.0** |
| `vent_gain` | (24,) | float64 | **all 1.0** |
| `leak_pa_s` | (24,) | float64 | **all 721.6158605428371** (= 0.1047 psi/s) |
| `coeff` | (3,) | float64 | `[0.16811763, 0.12188981, 0.1243398]` |
| `bf` | (3,) | float64 | `[0.151264, 0.15312, 0.140128]` (= MUSCLE Bf × 1.16) |
| `l0` | (3,) | float64 | `[0.1072, 0.0968, 0.0952]` m |
| `damp_b1` | (3,) | float64 | **ABSENT in the shipped file** (written 2026-08-12, pre-dates the 2026-08-17 damping work) |
| `meta` | () | `<U791` | JSON string |

`meta` JSON of the shipped checkpoint:
```json
{"kind": "synthetic-pretrained placeholder (design D1) ...",
 "seed": 20260811, "quick": false, "created": "2026-08-12 02:39:44",
 "metrics": {"leak_fit_pa_s": 721.6158605428371, "leak_truth_pa_s": 723.7049900709219,
             "leak_rel_err": 0.0028867142782586356,
             "warm_loss_first_last": [0.3083488413517322, 0.0010281174811475192],
             "shoot_loss_first_last": [1.124338837150617e-05, 5.837642796941256e-06],
             "heldout_rms_pa": 1334.7722686459747, "heldout_rms_psi": 0.19359235846107045,
             "n_traces": 36, "train_seconds": 27.715528200031258},
 "hidden": 32, "n_in": 6, "p_scale_pa": 172368.925, "dp_scale_pa_s": 100000.0,
 "l_scale_m": 0.1, "ldot_scale_m_s": 0.5}
```

**Unit-drift guard on load** (lines 522-531): the loader raises `ValueError` unless `meta["hidden"|"n_in"|"p_scale_pa"|"dp_scale_pa_s"|"l_scale_m"|"ldot_scale_m_s"]` match the module constants exactly. A missing `damp_b1` array falls back to `DAMP_B1_N_S_M_PER_PA`, **not zero** (line 541 + a dedicated test).

`ActuatorModel.default()` loads `HERE/"synthetic_pretrained.npz"`; a missing file raises `FileNotFoundError` whose message tells you to run `python -m UMArm_SIM.actuator_model --pretrain`.

### 2.13 Physical constraints baked in

There is **no monotonicity, no passivity, no saturation constraint on the net itself**. What exists:
1. `p >= 0` clamp after every substep (in `_shoot`, and in `sim_core.SimNode._integrate`).
2. Force `clip(f, -4000, 0)` — pull-only + hard clamp; `replay.rollout(assert_no_clamp=True)` raises `ClampViolation` if a rollout ever touches it.
3. Slack boundary `3l² <= Bf²` zeroes both force and the pressure-scheduled damping.
4. Gain clip `[0.05, 20.0]` after each gains step (an exploding/negative gain is "a numerical accident, not a pneumatic circuit").
5. Leak clip `[0, LEAK_MAX_PA_S=2000]` in `tune.fit_leaks` — a negative fitted leak would inflate idle nodes because the seam *subtracts* it.
6. The valve one-hot and the interval filter's sign rules (fill cannot lose pressure, vent cannot gain) are the only things pushing the net toward physically-signed flow — and the filter only shapes the warm-start dataset, not the shooting loss.

### 2.14 Synthetic pretraining (`pretrain_synthetic`, line 1065)

Uses genuine `fake_arm.FakeNode` instances (not a re-implementation) whose plant is the measured bench rig:
- double-exponential fill, `tau_fast_s=0.9 s` (90 % of ceiling), `tau_slow_s=18.0 s` (10 %), `vent_tau_s=1.0 s`, `decay_counts_per_s=3.7` present in **every** valve state, `noise_sd_counts=1.9` (truncated at 3σ), ceiling `K.REF_PSI = 40 psi`;
- `make_synth_node` (line 913) scales the rates by *dividing* the time constants by `fill_gain`/`vent_gain`; `ambient=590.0` counts, `span_counts=1410.0`, `zero_off=round(ambient)`, `scale=f32(scale_from_span(1410))`.

Corpus (`default_synthetic_traces`, line 1012), full (non-`--quick`) setting:
- amplitudes `(8, 15, 20, 25, 32)` psi × lengths `(0.062, 0.077, 0.092, 0.107, 0.122, 0.137)` m = **30 step traces**, each 9 s, step on at 0.1 s, off at 5.5 s;
- **plus 6 "rest" traces** (target 0, valves closed, p ≈ 0) at every grid `l` — 36 traces total, matching `n_traces: 36` in the checkpoint.
- Trace grid: `tick_s = 0.05` (20 Hz), `substep_s = 0.005`, `p_rec` quantized to the wire's 10 Pa units.

The rest traces exist because of a measured failure: without them `net_flow(p=0, closed)` came out at **+3.4 kPa/s**, so every idle node self-inflated ~0.4 psi/s and the joint-verification campaign's 0.5 psi rest interlock refused every joint. The `l` grid was widened from 0.087-0.127 m to 0.062-0.137 m for the same reason (bent-arm tendon excursion ±25 mm).

Pipeline: (1) `fit_leak_segments` on three 6 s guaranteed-closed decays from 20/12/6 psi; (2) `bf *= 1.16`, `leak_pa_s[:] = leak`; (3) `train_warm_start(epochs=400, sigma_pa = noise_sd*scale + 10 Pa, leak_max = 2*leak)`; (4) `train_shooting(epochs=200, train_gains=False)` — gains are *not* trained because one rig character means they are 1 by construction and any drift is net-scale degeneracy; (5) held-out verification at an unseen amplitude (18 psi) and length (0.102 m); (6) save with metrics in meta.

CLI (line 1142): `python -m UMArm_SIM.actuator_model --pretrain [--out PATH] [--seed 20260811] [--warm-epochs 400] [--shoot-epochs 200] [--quick]`.

---

## 3. `tune.py` — the fitting driver, full pipeline

CLI (line 1525):
```
python UMArm_SIM/tune.py CAMPAIGN_DIR [--out-dir DIR] [--holdout 0.2] [--seed 0]
       [--skip-outer] [--checkpoint PATH] [--epochs 60] [--window-ticks 160]
       [--outer-budget 40] [--outer-method nm|de] [--max-plot-trials 3]
```

### 3.1 Input contract (campaign directory)

Pinned headers (lines 100-101):
- `cmd.csv`: `t_s,node,sp_10pa` — **LONG** format, one row per driven node per tick; loader pivots to wide `(K, 24)` int, zero-filling undriven slots; refuses a bad header or non-increasing timestamps.
- `replies.csv`: `t_s,node,psi,status,flags` — psi field is **EMPTY** on `ST_NO_CAL` rows → NaN; `flags` ignored.
- `mocap.csv`: `t_s,frame,q0_rad..q11_rad,u{0..5}_{xyz},chain_ok` — rows with `chain_ok != 1` dropped; missing/empty file returns `None` and the pipeline degrades to constant rest-length muscle inputs.
- `trial.json`: `index, name, kind, status, driven[], spec{}`.
- `manifest.json`: `{"schema":1, "trials":[{index,name,kind,status,reason,seconds,dir}]}`.

Only statuses `("ok", "ok-data-incomplete")` are loaded (`USABLE_STATUSES`, line 105). Trials are sorted by `index`.

### 3.2 Stage 1 — split (`split_trials`, line 316)

Seeded (`np.random.default_rng(seed)`), **whole-trial**, **stratified by `kind`**. Per kind: `n < 2` → all to train; else `n_hold = min(n-1, max(1, round(holdout_frac*n)))`, a random permutation picks the holdout. Guarantees at least one training trial per kind and at least one holdout trial for any kind with ≥2 trials. Raises unless `0 < holdout_frac < 1`.

### 3.3 Muscle lengths from mocap (`TendonKinematics`, line 355; `trial_muscle_lengths`, line 442)

A private scratch MuJoCo model from `mjcf_generator.generate_xml()`; per-actuator tendon id via `model.actuator_trnid[a,0]` for actuators named `pam_1..pam_24`; `len0 = model.tendon_length0[t]`. `dlen(q)` runs `mj_kinematics → mj_comPos → mj_tendon` per sample and returns `(M, 24)` excursions.

Per trial: mocap q is `np.interp`'d onto the tick timestamps, pushed through the geometry (cached on the trial as `_dlen`, model-independent), then `l = model.l0_per_act + dlen`, `smooth3` per column, `ldot = np.gradient(l, t_cmd, axis=0)`. The 3-sample smoothing is there because ~0.3 mm mocap plate noise differentiates to ~20 mm/s spikes at 20 Hz.

### 3.4 Trace construction (`build_tick_traces`, line 475)

- `nominal_cal()` (line 392): **campaign data carries no per-node CalRecords**, so a fleet-nominal self-consistent calibration is used — `ambient=590.0` counts, `counts_per_psi = 1410/40 = 35.25`, `zero_off = 590`, `scale = f32(scale_from_span(1410)) ≈ 195.6 Pa/count`. Justified as within the ±4 % span spread, far inside the ±2000 Pa hysteresis the loop switches on.
- Reply→tick matching (`_match_replies_to_ticks`, line 414): nearest of the two bracketing replies, accepted if `|dt| <= REPLY_MATCH_TOL_S = 0.06 s`.
- `MAX_MISSING_FRAC = 0.25` — a node missing more than 25 % of tick replies is dropped with a warning; isolated holes are linearly interpolated (`_fill_nan_1d`).
- Only calibrated replies count: `np.isfinite(psi) & ((status & K.ST_NO_CAL) == 0)`.
- Valve labels: `valve_from_status` (line 400) maps `ST_INFLATING(0x01) → 1`, `ST_VENTING(0x02) → 2`, else 0.
- `target_pa = trial.sp[:, node-1] * 10.0`; `p0 = p_rec[0]`.
- Default node set = `trial.driven` (the idle-bystander regime is pinned by the checkpoint's rest traces and the leak fit, not by campaign traces).

### 3.5 Windowing (`window_traces`, line 523)

`kw = min(window_ticks (default 160), shortest trace)`. Starts at `range(0, n-kw+1, kw)` plus one extra window ending exactly at the trace end (overlap accepted as harmless reweighting). **Each window re-anchors `p0 = p_rec[start]`** — classic multiple shooting. 160 ticks at 20 Hz = 8 s horizon; 10 substeps per tick ⇒ 1590 tape entries per epoch.

### 3.6 Stage 2 — leak fit (`closed_segments_for_node` line 578, `fit_leaks` line 642)

A guaranteed-closed segment is a maximal run of consecutive calibrated replies that:
1. carry neither valve bit (`(status & K.ST_VALVES) == 0`),
2. sit inside a **constant commanded target** (holds, rest brackets, the zero tail — never a ramp/chirp, where "closed" reply instants alias rising pressure),
3. have no sampling gap `> LEAK_MAX_GAP_S = 0.2 s`,
4. never jump more than `3*sigma + leak_max*dt` between samples,

and that span `>= 2.0 s` with `mean(p) >= LEAK_P_FLOOR_PSI = 1.0 psi`.

Constants (lines 568-575): `LEAK_SIGMA_PA = 550.0` (the *consecutive-delta* noise scale: 1.7-2.7 counts/sample × nominal scale ≈ 370 Pa, ×√2 between samples, + 10 Pa wire LSB), `LEAK_MAX_PA_S = 2000.0`, `LEAK_MAX_GAP_S = 0.2`, `LEAK_P_FLOOR_PSI = 1.0`.

`fit_leaks` returns a new 24-vector (never mutates the model) plus per-node info `{initial_pa_s, n_segments, closed_s, fitted_raw_pa_s?, fitted_pa_s|None}`. Fitted values clipped to `[0, 2000]`; a node with no usable segment **keeps its checkpoint leak**.

### 3.7 Stage 3 — net training

```python
windows = window_traces(train_traces, window_ticks=window_ticks)     # tune.py:1106
model_net = copy.deepcopy(model_leak)
shoot_losses = am.train_shooting(model_net, windows, epochs=epochs)  # default epochs=60, train_gains=True
```
Everything else (lr, gains lr, substep, margin) is `train_shooting`'s default. Warm start is **NOT** run by `tune.py` — it only exists in `pretrain_synthetic` and the tests.

### 3.8 Evaluation

- **Open-loop** (`eval_open_loop`, line 833): `am.rollout` per held-out trace, residual on ticks `1..K-1`, per-node RMS in Pa.
- **Closed-loop** (`eval_closed_loop`, line 792 / `rollout_trial`, line 729): full `replay.rollout` on a fresh `SimArm` per trial, `tail_s=0.0`, `assert_no_clamp=True`. Starting pose seeded from the mean chain-valid mocap q over the 2 s leading rest bracket (`_POSE_SEED_S = 2.0`) — measured to cut mean q RMS from 1.6° to 0.27°. Metrics: per-node pressure RMS (Pa) against interpolated `p_rep_10pa*10`, per-joint q RMS (rad) against mocap, `ctrl_min_n`.
- Aggregates steered by: `pressure_rms_driven_pa` (mean over driven nodes) and `q_rms_mean_rad` (mean over all 12 joints).
- **Step features** (`step_features`, line 853): baseline = 0.5 s before ramp; settled = mean over the last half of the hold; rise = 10→90 % crossing span; overshoot = peak past settled as a fraction of the step; settle = last excursion outside a 5 % band. Returns `None` if `|Δ| < 0.005 rad` (~0.3°).

### 3.9 Guarded selection

`GUARD_TOL_FRAC = 0.01` (line 1041). `_not_worse(cand, ref)` (line 1044): a candidate ships only if neither held-out metric degrades by more than 1 %. Documented against the first real campaign (2026-08-12) where net training cut held-out pressure RMS 43 % (2673 → 1521 Pa) while q RMS moved 1.4434 → 1.4455° and a strict inequality discarded the whole improvement.

Candidates `["baseline", "leak-fit", "leak-fit+net"]` are filtered by `_not_worse(agg, closed_base)`, then `min` by `(pressure_rms_driven_pa, q_rms_mean_rad)`. A hard `assert` at line 1174 enforces the self-consistency invariant.

### 3.10 Stage 4 — outer fit (`outer_fit`, line 951)

12 **multiplicative** parameters in `OUTER_NAMES` order (line 931):
```
coeff_seg1..3, bf_seg1..3, l0_seg1..3, joint_damping, joint_frictionloss, tendon_damping
```
MuJoCo defaults (from `mjcf_generator`): `DEFAULT_JOINT_DAMPING = 0.026`, `DEFAULT_JOINT_FRICTIONLOSS = 0.025`, `DEFAULT_TENDON_DAMPING = 1.0`. `damp_b1` is deliberately **not** an outer degree of freedom — 20 Hz sysid data cannot see the ~1.8 Hz ring (Nyquist sits at the mode); fit it with `fit_bounce.py` on a flight recording instead.

Search in **log-multiplier space**, bounds `[log(0.7), log(1.3)]` (`span=0.3`), `x=0` evaluated first, best-seen point returned so the fit can only match or improve. Nelder-Mead with an **explicit initial simplex of 8 % steps** (scipy's default 2.5e-4 steps are invisible against replay noise), `maxfev=budget`, `xatol=1e-3`, `fatol=1e-7`; or `differential_evolution` with `popsize=4, init="sobol", polish=False`. A candidate raising `replay.ClampViolation` returns loss `1e6` and increments `n_clamp`. Objective = `q_rms_mean_rad` on the held-out trials with mocap.

### 3.11 Outputs written

Under `--out-dir` (default `CAMPAIGN_DIR/tuning`):
- `tuned_checkpoint.npz` — same format as §2.12, `meta` replaced with `{kind, campaign, input_checkpoint, created, seed, selection[], mujoco_tunables{}, holdout_pressure_rms_driven_pa, holdout_q_rms_mean_rad}`;
- `tuned_params.json` — `{force_law:{coeff,bf,l0:{initial,fitted}}, mujoco:{...}, per_node:{leak_pa_s:{n:{initial,fitted,n_segments,closed_s}}, fill_gain, vent_gain}}`;
- `tuning_report.json` — full dict (split, leak, net_training, open/closed loop, selection, outer, step_features, self_consistency, artifacts);
- `tuning_report.md` — human summary;
- `plots/pressure_trial_NNN_kind.png`, `plots/q_trial_NNN_kind.png` (matplotlib Agg, up to 3 held-out trials, recorded `#2a78d6` / tuned `#eb6834` / baseline `#898781` dashed).

---

## 4. Every invariant asserted by the two test files

### `test_actuator_model.py`
1. `test_gradient_check_mlp` — hand backward vs central FD, `rel_err < 1e-6` on 8 random entries per param array; input gradient at 4 positions `< 1e-6`; `input_grad(x, ones)[:,0] == dy_dx0` to `rtol=1e-12`.
2. `test_gradient_check_shooting_bptt` — `_shoot` BPTT vs FD, `rel_err < 1e-4` on 4 entries per array; `gfill[0]` FD `< 1e-4`; `loss > 0`.
3. `test_interval_filter_rejects_blip_contaminated` — an 8 kPa jump inside a closed-closed interval is rejected; all clean neighbours pass.
4. `test_interval_filter_endpoint_states_must_agree` — inlet→closed rejected; closed-closed with dp ≫ budget rejected; inlet-inlet rising accepted.
5. `test_interval_filter_state_consistency` — a filling interval that loses pressure and a venting one that gains it are both rejected.
6. `test_smooth3` — edges untouched, interior = 3-point mean.
7. `test_leak_fit_recovers_known_leak` — three 6 s closed decays from 20/12/6 psi at `noise_sd_counts=1.0` recover the 3.7 counts/s truth within **10 %**.
8. `test_leak_fit_refuses_short_segments` — a 0.45 s segment raises `ValueError(match="too short")`.
9. `test_synthetic_recovery_of_fill_vent_rates` — **the headline recovery test**. Node 2 fills 1.4× faster and vents 0.7× slower than node 1; after leak fit + warm start (250 epochs) + shooting (80 epochs, `train_gains=True`): `warm[-1] < 0.1*warm[0]`; `shoot[-1] < shoot[0]`; `fill_ratio ∈ [1.15, 1.70]` (truth 1.4); `vent_ratio ∈ [0.50, 0.92]` (truth 0.7); held rollout RMS `< 1.5 psi`. **Only the between-node RATIO is identifiable** — the absolute gain scale is degenerate (net × c, gains ÷ c).
10. `test_force_law_anchored_verbatim` — exact formula match to `rel=1e-12` for actuator 1; actuator index 8 (node 9) uses segment index 1's constants; a muscle at rest length under pressure pulls (`want < 0`); every other actuator reads exactly 0.
11. `test_force_law_pull_only_and_clip` — contracted past `Bf/sqrt(3)` → exactly 0.0; 500 psi on a stretched muscle → exactly `-4000 N`.
12. `test_checkpoint_roundtrip` — all 6 net arrays plus `fill_gain, vent_gain, leak_pa_s, coeff, bf, l0, damp_b1` bit-equal after save/load; `meta` preserved; forward output equal.
13. `test_load_missing_checkpoint_says_how_to_build` — `FileNotFoundError` mentioning `--pretrain`.
14. `test_pre_elasticity_checkpoint_inherits_damping_defaults` — a checkpoint with `damp_b1` stripped loads with `DAMP_B1_N_S_M_PER_PA`, not zero.
15. `test_tendon_damping_schedule` — `base + damp_b1_seg * p` per actuator with the ×8 segment expansion.
16. `test_shipped_checkpoint_is_sane` — `heldout_rms_psi < 0.5`; `leak_rel_err < 0.10`; all `leak_pa_s > 0`; `net_flow(p=0, INLET) > 5e4 Pa/s`; `net_flow(15 psi, EXHAUST) < -2e4 Pa/s`; `|flow(10 psi, CLOSED)| < 5e3 Pa/s`; a fresh synthetic 12 psi step rolls out at RMS `< 1.5 psi`.
17. `test_tick_trace_validation` — mismatched array length raises `ValueError(match="p_rec")`; `node_id=99` raises.
18. `test_shooting_batch_requires_shared_tick_count` — `ValueError(match="tick count")`.
19. `test_the_force_audit_reports_the_muscle_the_law_describes` (skipped without a bench checkpoint) — **the law implies a McKibben of 40-52 mm rest diameter on an arm whose muscles are 12-25 mm and share a 30 mm anchor ring four to a ring; `overfill_on_AO_ring > 2.0`; `coeff_scale_for_diameter["25mm"] < 0.5`.** The force scale is an unmeasured curve-shape parameter, not a measured physical constant.
20. `test_scaling_the_audit_s_model_touches_nothing_on_disk` — `force_audit.scaled` shares the net object but copies `l0`/`coeff`.

### `test_tune.py`
1. `test_load_cmd_csv_pivots_long_to_wide` — `(3,)`/`(3,24)` shapes, correct slot indexing, unnamed slots zero, strictly increasing t; bad header → `ValueError(match="header")`; reversed time → `ValueError(match="increasing")`.
2. `test_load_replies_groups_and_nan` — per-node grouping; empty psi → NaN with status preserved.
3. `test_load_mocap_filters_chain_invalid` — `chain_ok=0` rows dropped; missing file → `None`.
4. `test_load_campaign_filters_by_status` — only `ok` / `ok-data-incomplete` loaded; missing manifest → `FileNotFoundError(match="manifest")`.
5. `test_split_is_seeded_whole_trial_and_stratified` — same seed ⇒ identical split; disjoint and complete partition; from `{step:10, chirp:5, multi:1}` the holdout is exactly `{step:2, chirp:1}` (the lone multi trains); a different seed gives a different holdout; `holdout=1.5` raises.
6. `test_closed_segments_constant_target_and_floor` — exactly **1** segment from the constant-target hold; `>= 2 s`; mean `> 7 psi`; `fit_leak` recovers the injected 300 Pa/s to `rel=1e-3`; the 0 psi bystander yields `[]`.
7. `test_fit_leaks_fitted_vs_kept` — driven node fitted to 300 Pa/s; all 23 other nodes keep 123.0 (not zeroed) with `fitted_pa_s is None`; **`fit_leaks` never mutates the input model**.
8. `test_build_tick_traces_alignment` — `t == trial.t_cmd`; `target_pa == sp[:,node-1]*10`; `p_rec == psi*PA_PER_PSI` (`atol=1e-6`); valve labels INLET while ramping, CLOSED at hold tick 25, EXHAUST at tick 72; `p0 == p_rec[0]`; `0.05 < l < 0.2` and `l.std() > 0`; a never-heard node yields `[]`.
9. `test_window_traces_equal_length_and_coverage` — one shared tick count; first window starts at the trace start; last window ends exactly at the trace end; every window re-anchors `p0 = p_rec[0]`; mixing lengths still yields one shared count.
10. `test_step_features_on_analytic_first_order` — on `1-exp(-t/tau)`: `rise ≈ tau*ln(9)` (`rel=0.05`), overshoot ≈ 0 (`abs=0.02`), `settle ≈ tau*ln(20)` (`rel=0.10`); a flat response returns `None`.
11. `test_apply_outer_multipliers` — index mapping `0..2 coeff / 3..5 bf / 6..8 l0 / 9 joint_damping / 10 joint_frictionloss / 11 tendon_damping`; the input model is deep-copied, never mutated; `OUTER_NAMES[9] == "joint_damping"`.
12. `test_report_writers_smoke` / `test_write_tuned_params_pairs_initial_and_fitted` — required markdown sections and initial/fitted pairing.
13. `test_dry_run_end_to_end` (`@pytest.mark.slow`, gated on `UMARM_RUN_SLOW=1`, ~10 min) — a MuJoCo-backend campaign through `tune.main` exits 0, writes all four artifacts and PNGs, `self_consistency == {pressure_ok: True, q_ok: True}`, and `final <= baseline` on **both** aggregate metrics.
14. `test_the_stage_guard_tolerates_noise_but_not_a_real_regression` — the real-campaign case passes; a 5 % q regression and a 5 % pressure regression are both refused; exactly `1 + GUARD_TOL_FRAC` is accepted.

---

## 5. What data the training actually needs

### Fields (per trial, one shared time origin `t0`)
| stream | rate | fields | why |
|---|---|---|---|
| `cmd.csv` | 20 Hz (tick) | `t_s, node, sp_10pa` | `target_pa` for the in-loop bang-bang; also the constant-target rule for leak segments |
| `replies.csv` | 20 Hz per node, all 24 answer every tick | `t_s, node, psi, status, flags` | `p_rec` (the loss target), `valve` labels via `ST_INFLATING`/`ST_VENTING`, `ST_NO_CAL` gating, closed runs for the leak fit |
| `mocap.csv` | 120 Hz | `t_s, frame, q0..q11_rad, u0..u5_xyz, chain_ok` | `l`, `ldot` net inputs; q RMS for the outer fit |
| `trial.json` | — | `index,name,kind,status,driven[],spec{q_index,side,amp_psi,ramp_s,hold_s,...}` | split stratification, driven-node set, step features |

Sensor calibration is **not** required per node — `nominal_cal()` substitutes ambient 590 counts / span 1410 counts. If the new arm can log real CalRecords, feeding them per node removes that approximation.

### Rate and timing
- Tick grid 20 Hz (`K.TICK_HZ = 20.0`), integration substep 5 ms ⇒ 10 substeps/tick. **The recorded tick timestamps are used as Δt** — the 8-9 ms/tick Windows slip is in the data on purpose and must be preserved in the CSV, not resampled.
- Replies must land within 60 ms of their tick, and a node may miss at most 25 % of ticks.
- Bang-bang hysteresis `K.MARGIN_PA = 2000.0` Pa; ADC `K.ADC_MAX = 4095`; `K.PA_PER_PSI = 6894.757`; `K.REF_PSI = 40.0`.

### Duration and excitation
- **Shooting windows are 160 ticks = 8 s**, so any usable trial must exceed 8 s (or the window shrinks to the shortest trace, degrading the BPTT horizon for everything).
- **Leak fit needs `>= 2 s` of constant-target, valve-closed, gap-free reply run at `>= 1 psi` mean.** In the campaign these are the bang-bang's long closed stretches inside step/co-contract holds (at 8 psi the ±2 kPa band takes ~7 s to leak across, so a 4 s hold is one closed run), plus the trailing rests, the 2 s rest brackets, the 20 s cooldowns and failsafe traps. **Without such segments the leak is unfitted and the node keeps its checkpoint value.**
- Excitation kinds the pipeline expects (`data_collection.TrialSpec`): `step` (`ramp_s=2.0`, `hold_s=4.0` defaults), `staircase` (`treads=4`, `tread_s=2.0`), `ramp` (`rates_psi_s`), `chirp` (`f0_hz=0.05 → f1_hz=1.5`, `seconds <= 60`), `prbs` (`>=2 levels_psi`, `dwell_s >= 0.5`, `seconds <= 75`), `cocontract` (`sum_psi`, `split_psi`, probes of `SYSID_PROBE_STEP_PSI=2.0`), `multi` (two `q_indices` on *different* segments, per-pair `sum_psi <= 24`).
- Amplitude envelope: `SYSID_CAP_PSI = 25.0` per node (this is what `P_SCALE_PA` normalizes to), `SYSID_PAIR_SUM_PSI = 30.0` per pair, escalation ladder `(8, 10, 12, 15, 18, 21, 25)` psi.
- **The `l` envelope matters at its edges**, not just the middle: coverage must include the bent-arm tendon excursion (±25 mm around rest), or closed-valve flow at bent-arm lengths is pure extrapolation and idle bystanders self-inflate (measured 0.05-0.1 psi per quiet window, past the 0.5 psi rest interlock).
- **Rest/idle evidence is mandatory** — the (closed, p≈0, dp=0) operating point is the most common one in service and the step grid alone leaves it unconstrained (measured +3.4 kPa/s spurious self-inflation).
- Stratification needs **at least 2 trials of each kind** for that kind to appear in the holdout at all; the pipeline raises if the holdout ends up empty.

---

## 6. Porting notes for the 24-actuator CAN arm (two valve populations)

These are my inferences from the code, flagged as such — they are not statements the source makes.

1. **The 3-way valve one-hot is the single biggest structural change.** Columns 1-3 of the input are `onehot(closed, inlet, exhaust)` — a binary solenoid alphabet. For the top 8 TLE92464 proportional DVP valves the command is a **coil current code**, a continuous quantity; for the bottom 16 legacy 7 mm solenoid pairs it is **PWM duty**. Two clean options: (a) one net per population (simplest, and the segment machinery already partitions 24 into groups of 8); (b) one net with input `[p/P_SCALE, u_in, u_out, population_flag, l/L_SCALE, ldot/LDOT_SCALE]` where `u_in`/`u_out` are normalized drive levels that collapse to the current one-hot when the drive is binary.
2. **`np.repeat(x, 8)` (lines 365-378) hard-codes 3 segments × 8 actuators.** By coincidence segment index 0 = actuators 1-8 = exactly the TLE population (`0x101-0x108`). Do not rely on that coincidence: the population split and the segment split are different concerns and each needs its own index vector.
3. **150 Hz sync = 6.67 ms/tick.** `substep_s = 0.005` would give `nsub = round(0.00667/0.005) = 1` — one substep per tick, no sub-tick bang-bang resolution at all. Either drop `substep_s` to ~1 ms (≈7 substeps/tick, 7× the tape and 7× the BPTT cost) or accept one substep and re-derive the shooting horizon. `window_ticks=160` at 150 Hz is 1.07 s, not 8 s — retune it to keep an ~8 s horizon (≈1200 ticks) and watch the tape memory: `sum(nsub)` entries each holding `(B,6) + 2×(B,32)` float64.
4. `MARGIN_PA`, `ADC_MAX`, `zero_off`/`scale`/`counts_per_psi` are all firmware-mirror constants in `arm_constants.py`; the new CAN firmware's equivalents must replace them inside `_shoot`'s ADC/bang-bang block, and if the TLE population is not bang-bang at all that whole block has to be replaced by the proportional control law for those 8 nodes.
5. `fill_gain`/`vent_gain` are the natural per-population per-node handle and already exist; you likely want a third scalar for the proportional valves (a current→flow slope) since "inlet gain" and "exhaust gain" no longer partition the drive space.

## 7. Cost and scale reference (for sizing the re-train)

- Net: 1313 parameters. Pretraining 36 traces × 181 ticks, warm 400 epochs + shooting 200 epochs: **27.7 s wall** (from the checkpoint metadata).
- Shooting cost per epoch ≈ `epochs × sum(nsub) × 2` net passes on a `(B,6)` batch, where `B` = number of windows.
- Achieved quality on synthetic data: held-out rollout RMS **1334.8 Pa = 0.194 psi**; leak recovered to 0.29 % relative error.


## KEY FILES
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/actuator_model.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/tune.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/synthetic_pretrained.npz
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/test_actuator_model.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/test_tune.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_ROBOT_CONTROL/arm_constants.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_ROBOT_CONTROL/fake_arm.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/sim_core.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/replay.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_ROBOT_CONTROL/data_collection.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/mjcf_generator.py

## GOTCHAS
- The net predicts dp/dt ONLY. Leak is never learned by the net: it is fitted separately by least-squares slope on >=2 s guaranteed-closed segments and SUBTRACTED outside the net at the plant seam (dp/dt = gain(valve)*f_theta - leak). Reason given: per-interval leak is ~36 Pa/50 ms, below one ADC count and below sensor noise. If you let the net absorb the leak you will get self-inflating idle nodes.
- There is NO input history and NO history stacking. The input is exactly 6 numbers at one instant: [p/172368.925, onehot_closed, onehot_inlet, onehot_exhaust, l/0.1, ldot/0.5]. All temporal structure enters through the shooting rollout's recurrence and through ldot.
- TickTrace.valve is NOT used by the shooting loss. _prepare_batch never copies it into _Batch; the in-loop bang-bang regenerates valve states from a simulated ADC. The recorded labels feed ONLY the warm start and filter_single_intervals. Also, _shoot starts every window with act = 0 (valves closed) regardless of the recorded label at the window start.
- The shipped synthetic_pretrained.npz has NO damp_b1 array (written 2026-08-12, before the 2026-08-17 damping work). ActuatorModel.load falls back to DAMP_B1_N_S_M_PER_PA = (9.9e-4,)*3, NOT zero. A port that defaults missing damping to zero silently revives ring-killing behaviour.
- ActuatorModel.load raises ValueError unless meta['hidden','n_in','p_scale_pa','dp_scale_pa_s','l_scale_m','ldot_scale_m_s'] match the module constants exactly. Change any normalization constant and every existing checkpoint becomes unloadable by design - this is a feature, keep it.
- The absolute scale of fill_gain/vent_gain is DEGENERATE with the shared net (net x c, gains / c). Only BETWEEN-NODE ratios are identifiable, which is exactly what test_synthetic_recovery_of_fill_vent_rates asserts. Do not write acceptance tests on absolute gain values. Closed-state gain is hard-fixed at 1.0 and is not a fitted DOF.
- BF_PRETUNE_SCALE = 1.16 is applied only in pretrain_synthetic, NOT in ActuatorModel.fresh(). fresh() keeps the verbatim MUSCLE constants. The shipped checkpoint's bf is (0.151264, 0.15312, 0.140128) = MUSCLE Bf x 1.16. Confusing the two gives distal joints at 36-43 deg instead of 5-21 deg at 8 psi.
- MUSCLE constants are ALREADY x0.8-prescaled by the legacy code. Never rescale them again (module docstring lines 31-33).
- force_audit's test finds the anchored law implies a McKibben of 40-52 mm rest diameter on an arm whose muscles are 12-25 mm and share a 30 mm anchor ring four to a ring. coeff is therefore an unmeasured curve-shape parameter, not a measured physical constant. Do not treat coeff/Bf as ground truth when porting.
- l0 is BOTH a force-law parameter AND the offset of the net's l input. If the outer fit changes l0, the tick traces MUST be rebuilt before open-loop evaluation (tune.py does this at lines 1165-1167, but only when outer_accepted is True). Miss this and the net is evaluated on inputs from a different convention than it was trained on.
- np.repeat(x, 8) at actuator_model.py:365-378 hard-codes 3 segments x 8 actuators. For the new arm the population split (8 TLE + 16 solenoid) is a DIFFERENT partition than the segment split, even though segment 0 happens to coincide with the TLE population. Give each its own index vector.
- substep_s = 0.005 s against a 150 Hz (6.67 ms) sync gives nsub = 1: a single substep per tick, i.e. no sub-tick valve resolution. And window_ticks = 160 is 8 s at 20 Hz but only 1.07 s at 150 Hz. Both defaults must be re-derived or the BPTT horizon collapses.
- The simulated ADC inside _shoot has NO sensor noise, while the recorded p_rec carries noise plus 10 Pa wire quantization, and p0 = p_rec[0] treats a quantized noisy reading as true pressure at every window anchor. This is a deliberate errors-in-variables choice, not a bug - but it is a real bias source at short windows.
- Stage 4 (outer_fit) optimizes q RMS on the HOLDOUT trials (hold_q) and then the same holdout judges acceptance. It is a fit on held-out data, guarded only by the 1 % _not_worse tolerance. Flag this if the new pipeline needs an honest generalization number.
- fit_leaks clips fitted leaks to [0, 2000] Pa/s because the seam SUBTRACTS the leak - a negative slope estimate (sensor noise read as self-inflation) would shipped as idle-node inflation. Nodes without a usable closed segment KEEP the checkpoint value; they are not zeroed.
- There is no early stopping, no validation split, no LR schedule and no regularization inside the trainers. Epoch counts are fixed (warm 300/400, shooting 60/120/200 depending on caller). The only 'stop if worse' mechanism is tune.py's guarded selection with GUARD_TOL_FRAC = 0.01.
- Training data must contain explicit REST evidence (closed valves, p ~ 0) across the whole l envelope. Without it the measured failure was net_flow(p=0, closed) = +3.4 kPa/s, i.e. every idle node self-inflating ~0.4 psi/s, which refused every joint at the 0.5 psi rest interlock.
- The l training grid must span bent-arm tendon excursion (+-25 mm around rest), not just the rest lengths. A grid stopping at 0.087-0.127 m left bent-arm closed-valve flow as pure extrapolation and inflated idle bystanders 0.05-0.1 psi per quiet window.
- Recorded tick timestamps are used verbatim as dt (the 8-9 ms/tick Windows slip is intentionally kept in the data). Do not resample or regularize the time base when logging the new campaign.
- damp_b1 is deliberately NOT an outer-fit degree of freedom: 20 Hz sysid data cannot see the ~1.8 Hz structural ring it shapes (Nyquist sits at the mode). Dissipation is fitted separately by fit_bounce.py against a 160 Hz flight recording.
