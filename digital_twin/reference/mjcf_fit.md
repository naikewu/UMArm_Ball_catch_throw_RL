# Recon: twin GEOMETRY, DISSIPATION FIT, and SCORING instrument

Read completely: `mjcf_generator.py` (497 l), `ring_analysis.py` (250 l), `fit_bounce.py` (419 l), `twin_compare.py` (555 l), `force_audit.py` (508 l), `umarm_sim.xml` (193 l), `viz/mjcf_canarm.py` (613 l), `UMArm_KINEMATICS/canarm_params.py`, `UMArm_KINEMATICS/canarm_actuators.py`.
Read partially (targeted, for the seams the six files depend on): `UMArm_SIM/actuator_model.py` lines 1–200 and 360–520 of 1160; `UMArm_SIM/sim_core.py` lines 520–660 and 700–800 of 953; `UMArm_SIM/replay.py`, `bounce_demo.py`, `test_mjcf.py` by grep only.
**NOT read at all**: `tune.py` (1578 l), `sim_master.py`, `sim_mocap.py`, all `test_*.py` bodies, `UMArm_CONTROL/flight_recorder.py` (defines `RunData`, `FLAG_ENGAGED`, `FLAG_HAVE_POSE`, `wire_timeline`, `event_t` — the whole recording contract that `twin_compare` consumes), `docs/simulator_design.md`, and the CAN workspace's `hw_tests/`, `firmware/`, `legacy_host/`.

---

## 1. `mjcf_generator.py` — the geometry

`C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\mjcf_generator.py`

### 1.1 The argument-not-constant pattern, exactly as implemented

The public surface is **five keywords and nothing else** (`generate_xml`, l. 440–464):

```python
def generate_xml(*, base_pos=DEFAULT_BASE_POS,
                 base_yaw_deg: float = DEFAULT_BASE_YAW_DEG,
                 joint_damping: float = DEFAULT_JOINT_DAMPING,
                 joint_frictionloss: float = DEFAULT_JOINT_FRICTIONLOSS,
                 tendon_damping: float = DEFAULT_TENDON_DAMPING) -> str:
```

`generate_scene(out_path=None, **kwargs)` (l. 467) writes it; `main()` (l. 475) exposes the same five as CLI flags. Everything else — lengths, masses, ring azimuths, timestep, force clamp, armature, joint range — is a **module constant**, and the docstring at l. 21–26 states the reason explicitly: "the generator must work on a clone without campaign artifacts, and a re-fit becomes an explicit constant change, not a silent geometry change." Note this is the pattern the CAN workspace deliberately *broke*: `viz/mjcf_canarm.py:22-28` says the single structural change it made is `build_arm_xml(chain_m=…)`, and `digital_twin/mjcf_generator.py` (the skeleton, l. 13–19) records that the twin generator must do the same. So the CAN twin's surface is `chain_m` + base pose + the ring table + the dissipation tunables.

The genparams string is echoed into the XML comment (l. 451–455 → `umarm_sim.xml:5`), so a shipped model always names the arguments it was built with. Keep that.

### 1.2 Fitted geometry constants (metres)

```python
FITTED_CHAIN_M = (0.22467284, 0.04806482, 0.19411817, 0.04676957, 0.18537953)   # l.88
AA1_M = 0.0285 ; AA2_M = 0.0285                                                  # l.92-93
```
`FITTED_CHAIN_M` is `(span1, JD2, span2, JD3, span3)` — the five consecutive u-joint-centre gaps, proximal to distal. Provenance in the docstring: `campaign_2026-08-11_recal/fkine_benchmark/fkine_benchmark.json → lengths.fitted_streamed_m`, residuals ~1.5e-5 m.

`PARAMS` (l. 100–105) is the legacy `(3, 10)` table in column order `[JA1, JA2, UC1, UC2, AA1, AA2, AO1, AO2, LL, JD]`, built as `LL = span − AA1 − AA2`, `UC1 = UC2 = 0`, `JA1 = 0.10` on segment 1 and `0.05` on segments 2–3, `JA2 = 0.05`, `AO1 = AO2 = 0.03`. It is frozen: `PARAMS.flags.writeable = False`. `fitted_params()` (l. 204) returns a **fresh copy** for handing to `fkine` as the FK oracle of this exact model.

`seg_geometry(i)` (l. 219–228) turns a row into the z-coordinates all the sites hang off:
- `z_top_ring = −(UC1 + AA1)` = −0.0285
- `z_bot_ring = −(UC1 + AA1 + LL)`
- `z_distal = z_plate2 = −(UC1 + AA1 + LL + AA2)` = −span (the distal u-joint centre)

### 1.3 Bodies and joints (`build_umarm`, l. 270–358)

Chain, one open kinematic branch, no free joint:

```
base_plate  (pos = base_pos, quat = yaw about world z, childclass="umarm")
 └ seg1_link          joints uj1_x (axis 1 0 0), uj1_y (axis 0 1 0)
    └ seg1_plate2     joints uj2_x (axis  √2/2  √2/2 0), uj2_y (axis −√2/2 √2/2 0)
       ├ geom jd2_bracket, geom seg2_plate1_geom, sites s2_plate1_a1..4
       └ seg2_link    joints uj3_x, uj3_y
          └ seg2_plate2  joints uj4_x, uj4_y
             ├ geom jd3_bracket, geom seg3_plate1_geom, sites s3_plate1_a1..4
             └ seg3_link  joints uj5_x, uj5_y
                └ seg3_plate2  joints uj6_x, uj6_y
                   └ ee_assembly (rigid, plate 6)
```

`nq = nv = 12`, `nu = ntendon = 24` (pinned by `test_mjcf.py:123-124`).

The load-bearing property, stated at l. 5–8 of the docstring: **"MuJoCo composes same-body hinges in declaration order, provably matching fkine's `exp(xi1 t1)…exp(xi4 t4)`"**. Declaration order `x` then `y` on the proximal pair == fkine `order="xy"`. `test_mjcf.py:161-173` pins body `xpos`/`xmat` against `fkine` at 1e-9 and `test_mjcf.py:227-238` pins the 24 tendon lengths against an analytic two-site formula at 1e-9.

The distal pair's axes are the 45-degree bracket vectors `(±1, 1, 0)/√2`, written with `SQ2 = math.sqrt(2.0)/2.0` (l. 201).

### 1.4 Sites — the tendon routing

Four rings per segment, each of four sites, radii from the params row:

| ring | radius | z | site names | which tendon end |
|---|---|---|---|---|
| `s{n}_plate1_a1..4` | `JA1` | 0 (seg 1) / `−JD` (seg 2,3) | proximal plate | upper end of lower muscles |
| `s{n}_link_bot_a1..4` | `AO1` | `z_bot_ring` | link bottom disk | lower end of lower muscles |
| `s{n}_link_top_a5..8` | `AO2` | `z_top_ring` | link top disk | upper end of upper muscles |
| `s{n}_plate2_a5..8` | `JA2` | 0 in `seg{n}_plate2` | distal plate | lower end of upper muscles |

Azimuths (l. 196–197) — **the operator-verified hardware seat table, not a nominal compass**:

```python
LOWER_RING_DEG = (180.0, 270.0, 0.0, 90.0)    # local acts 1-4 (proximal ring)
UPPER_RING_DEG = (225.0, 315.0, 45.0, 135.0)  # local acts 5-8 (distal ring)
```
Literal copy of `UMArm_ROBOT_CONTROL.joint_actuator_map.SEAT_DEG_BY_LOCAL_ID`; `test_mjcf.py:325` asserts the two agree. The docstring (l. 46–57) explains that a 180-deg ring relabel changes no tendon length and no dynamics, only *which name sits where*, and that with this table **`pam_i` IS node `i`'s muscle**, so `sim_core` maps node ids to actuators by identity.

`ring_xy(radius, angle_deg)` (l. 214) is just `(r·cos, r·sin)`.

### 1.5 Tendons and actuators (`build_tendons_actuators`, l. 361–387)

24 **two-site straight-line spatial tendons**. Global index `gi = (seg−1)*8 + local`, so `pam_1..8` = segment 1, `pam_9..16` = segment 2, `pam_17..24` = segment 3.

- Lower group, local `k = 1..4`: `t_s{n}_a{k}` = `s{n}_plate1_a{k}` → `s{n}_link_bot_a{k}` — **spans the proximal u-joint plus the link**.
- Upper group, local `k = 5..8`: `t_s{n}_a{k}` = `s{n}_link_top_a{k}` → `s{n}_plate2_a{k}` — **spans the link plus the distal u-joint**.

Every actuator is a plain `<motor>` on its tendon:
```
<motor name="pam_{gi}" tendon="t_s{n}_a{k}" gear="1" ctrllimited="true"
       ctrlrange="-4000 0" forcelimited="true" forcerange="-4000 0"/>
```
Pull-only by construction. `FORCE_RANGE = "-4000 0"` (l. 185) newtons; the example's −1000 N clamp bit inside the 25 psi envelope for stretched antagonists (docstring l. 29–31). `replay.rollout(..., assert_no_clamp=True)` raises `ClampViolation` when a rollout comes within `_CLAMP_EPS_N` of −4000 (`replay.py:249`).

### 1.6 Masses (l. 127–131) — no measured inertias anywhere

```python
LINK_MASS = [0.02, 0.02, 0.02]   # the seg{n}_rod geoms
BRACKET_MASS = 0.48              # each JD bracket geom (jd2_bracket, jd3_bracket)
BASE_BRACKET_MASS = 0.24         # added to the base plate's 0.5
PLATE3_MASS = 0.12               # seg3_plate2 only
EE_MASS = 0.12                   # split 0.06 + 0.06 over ee_rod1/ee_rod2
```
Every other disc is `mass="0.0001"` — deliberately massless (`_plate1_geom` docstring, l. 246–256, measures the consequence: adding both discs moves total mass 2e-4 kg and a 2 s hard asymmetric drive ends 0.0135 deg from the same rollout). MuJoCo derives all inertias from the primitives; the module docstring (l. 59–61) says plainly "no measured inertias exist". `test_mjcf.py:262-282` pins every body mass. Total moving mass is quoted in `force_audit.audit_replay`'s docstring as **1.261 kg** against the operator's ~1.2 kg for the metal.

### 1.7 Fittable tunables — exactly three, with units, defaults, and provenance

| constant | default | unit | status |
|---|---|---|---|
| `DEFAULT_JOINT_DAMPING` (l. 156) | **0.026** | N·m·s/rad, per hinge | **fitted** 2026-08-17 |
| `DEFAULT_JOINT_FRICTIONLOSS` (l. 168) | **0.025** | N·m, per hinge | **fitted** 2026-08-17 |
| `DEFAULT_TENDON_DAMPING` (l. 174) | **1.0** | N·s/m, per tendon | **chosen, not fitted** |

They are emitted on the `umarm` default class, so they propagate to every joint / every tendon (`test_mjcf.py:300-309` proves both the defaults and the propagation of overrides).

The docstring at l. 137–174 is the single most important paragraph in the file and must survive the port:
- fit run: `20260816_222851_real_rec1`, artifacts `UMArm_ROBOT_CONTROL/results/bounce_fit_2026-08-17/`;
- the superseded engineering guesses were joint damping **2.25**, frictionloss **0.70**, tendon damping **50**;
- under those, "every mode was critically-to-over damped (poke-test ζ ≈ 0.73)" and "the 0.7 N·m Coulomb floor alone exceeded the peak elastic torque of a ~1° oscillation (measured stick threshold 0.56–1.11° per joint at 15 psi, up to 3.3° at 5 psi)", so the twin **could not sustain the observed 0.2–2° rings at any parameterization of the rest of the model**;
- `DEFAULT_TENDON_DAMPING = 1.0` is only the **p = 0 base** of the runtime schedule `tendon_damping = base + damp_b1·p`; it is unfittable because every measured ring episode is pressurized, and 1.0 N·s/m is "well below the ≈55 N·s/m the b1 term contributes at the ~8 psi the arm rings at";
- known divergence, recorded rather than papered over (l. 161–167): with frictionloss 0.025 the free-pendulum single-muscle campaign regressed to **8/12** with frozen-plate aborts on q2–q5, because the model has no deflated-muscle passive elasticity to hold the chain.

### 1.8 Solver / option settings (`SCENE_TEMPLATE`, l. 390–437)

```
<option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast" iterations="100"/>
```
- `TIMESTEP_S = 0.001` (l. 182) — the example ran 0.5 ms. **Anything discretizing lag or dissipation must read `model.opt.timestep`, never assume it.** `sim_core.SimArm.__init__` does exactly that (`self._dt = float(self.model.opt.timestep)`) and only *then* allows a `timestep_s` override.
- `integrator="implicitfast"` is load-bearing, not cosmetic: `sim_core` rewrites `model.tendon_damping` at every node pass and the implicit treatment makes that "unconditionally stable at any physical magnitude" (`sim_core.py:625-631`).
- Inherited stability choices, flagged as inherited (l. 187–189): `JOINT_RANGE_RAD = 0.6981317007977318` (±40 deg) and `JOINT_ARMATURE = 0.01`.
- No contacts anywhere: `<geom contype="0" conaffinity="0"/>` on the `umarm` class **and** on the floor. `test_mjcf.py:134-138` asserts `np.all(geom_contype == 0)`; `:130-131` asserts `nkey == 0` and `npair == 0`.
- Site default: `type="sphere" size="0.0035"`; tendon default `width="0.0035"`.

---

## 2. `ring_analysis.py` — the measuring stick

`…\UMArm_SIM\ring_analysis.py`. Pure numpy/scipy, **no repo imports, no I/O** (docstring l. 24–25) — the property that lets it run identically on a flight recording, a twin rollout and a poke-demo trace. Port it verbatim; nothing in it is RS485-specific.

Constants (l. 39–48):
```python
BAND_HZ = (1.5, 12.0)   # Hz
MIN_CYCLES = 2.5        # cycles of the episode's own dominant period
THRESH_DEG = 0.25       # envelope threshold, degrees (below = mocap plate jitter)
```
The low edge sits just under the measured 1.7–2.0 Hz structural mode on purpose, to reject ~1 Hz step-settle content; the in-band attenuation is tolerated because **fitting happens on the RAW segment**.

### Episode detection — `find_episodes(t, q, fs, *, band, thresh, min_cycles, fit=True) -> list[Episode]` (l. 188–246)

`q` is `(n, nj)` in the caller's units (degrees for recordings), `thresh` in the same units. Algorithm, per joint `j`:
1. `qb = bandpass(q, fs, 1.5, 12.0)` — order-3 Butterworth, `filtfilt` (zero-phase) along axis 0 (l. 50–55).
2. `env = np.abs(hilbert(qb[:, j]))`; `above = env > thresh`.
3. Contiguous `above` regions become candidate `(i0, i1)` via `np.flatnonzero(np.diff(above.astype(int)))` with explicit handling of `above[0]` / `above[-1]`.
4. Reject `seg.size < 8`.
5. Dominant period from zero crossings of the **band-passed** segment: `zc = np.flatnonzero(np.diff(np.signbit(seg)))`; reject if `zc.size < 2*min_cycles`; `period_s = 2.0*mean(diff(zc))/fs`.
6. Reject if `(t[i1-1] − t[i0]) < min_cycles * period_s`.
7. Pad a **quarter period** each side: `pad = round(period_s*fs/4)`, window `[a0, a1)`.
8. Fit the damped sine on the **RAW** signal over the padded window, seeded `f0 = 1/period_s`.

Returned list is sorted by descending `peak_band_amp`.

### The fit — `fit_damped_sine(t, y, f0=None) -> RingFit` (l. 90–162)

Model: `y = c0 + c1·t + A·exp(−σ·t)·cos(2π f t + φ)` — the linear trend absorbs post-step settling.

Seeds: `c = polyfit(tt, y, 1)`, `resid = y − polyval(c, tt)`; if `f0 is None` it is the FFT peak of the Hanning-windowed residual with DC skipped; `σ0` is the negated slope of `log(|hilbert(resid)|)` over the **middle half** (edges are Hilbert artifacts), clipped to `[0, 20·f0]`; `A0 = percentile(env, 90)`.

Bounds — the seeded/blind distinction matters and `fit_bounce` depends on it (l. 132–141):
```python
f_lo, f_hi = ((0.6*f0, 1.6*f0) if seeded else (max(0.1, 0.5*f0), 2.0*f0 + 1.0))
sig_cap = 1.5 * 2.0 * np.pi * f_hi
lo = [-inf, -inf, 0.0, 0.0, f_lo, -2π] ; hi = [inf, inf, inf, sig_cap, f_hi, 2π]
```
σ is bounded ≥ 0 so a "growing ringdown" cannot be reported as negative damping. `least_squares` is run **four times** with phase seeds `(0, π/2, π, −π/2)` and the lowest-cost solution kept — the phase is the one genuinely multi-modal start.

Outputs (`RingFit`, l. 70–87): `freq_hz`, `zeta = σ/√(σ²+ω²)`, `sigma` (1/s), `amp`, `phase`, `offset`, `trend` (units/s), `r2`, `n`.

**The r² definition is a deliberate instrument choice** (l. 151–155): `ss_tot` is the variance of the **detrended** residual, not of the raw signal, "so scoring against the raw variance would let a pure drift (no oscillation at all) fit its own trend and read r² ≈ 1 — an instrument that certifies rings that are not there." Consequence: r² here is harsh — a visually clean ring inside a strong post-step settle scores ~0.3.

`interp_nans(t, y) -> (filled, nan_fraction)` (l. 58–67) linearly interpolates NaN gaps and hands back the fraction so callers can refuse to score a mostly-missing signal.

---

## 3. `fit_bounce.py` — the four dissipation scalars

`…\UMArm_SIM\fit_bounce.py`

### 3.1 The four scalars, named, with physics and units

| scalar | lives in | unit | what it physically is | search bounds (`FitSpace`, l. 93–111) |
|---|---|---|---|---|
| `joint_damping` | MJCF hinge class | N·m·s/rad | spine viscous dissipation: bearings + rod flex | `(0.005, 1.5)`, searched in **log** space |
| `joint_frictionloss` | MJCF hinge class | N·m | spine Coulomb friction | `(0.0, 0.35)`, searched **linearly** |
| `tendon_damping` | MJCF tendon class | N·s/m | p = 0 base of muscle damping — deflated braid/bladder loss | `(0.05, 30.0)`, **log** |
| `damp_b1` | `ActuatorModel.damp_b1` (a per-segment 3-vector, here written to all three) | **N·s/m per Pa** | pressure slope of the McKibben's viscous damping (Reynolds/Repperger et al. 2003: B(P) affine in P; Tondu & Lopez 2000 attribute it to inter-fiber kinetic friction) | `(1e-5, 3e-3)`, **log** |

The runtime law is `tendon_damping_per_muscle = tendon_damping + damp_b1 · p_pa`, written into `model.tendon_damping` at every node pass by `SimArm._node_pass` (`sim_core.py:775-779`), **gated on the force law's slack boundary** — `actuator_model.tendon_damping_n_s_m` zeroes the pressure term where `3ℓ² ≤ Bf²`, because "MuJoCo tendon damping is bilateral, so an ungated slack muscle at pressure would push on the joint through a rope."

Fitted result shipped in `actuator_model.py:183`: `DAMP_B1_N_S_M_PER_PA = (9.9e-4, 9.9e-4, 9.9e-4)` — one value for all three segments, because "the recordings ring at one shared ~1.8 Hz structural mode and cannot separate segments". At 8 psi (≈55 kPa) that is **≈55 N·s/m per muscle**.

**Stiffness is explicitly NOT fitted here** (l. 22–26): the anchored force law already carries `K ∝ p` (`∂F/∂ℓ = −6·coeff·p·ℓ`), the sysid campaign pinned `coeff`, and with dissipation removed the model rings within 2 % of its own tangent-stiffness mode 0 and ~10 % of the measured frequency.

### 3.2 Target selection — `ring_targets(run)` (l. 114–142)

Signal is `run.rows["q_raw_deg"]`, NaN-interpolated per joint, and a joint with `frac > 0.2` missing is **blanked to NaN rather than scored**. `fs = 1/median(diff(t))`. Detection runs at `thresh=0.30` deg (note: not the module default 0.25). Gates:
```python
RING_ZETA_MAX = 0.25 ; RING_R2_MIN = 0.30
RING_FREQ_HZ = (0.8, 6.0) ; RING_AMP_DEG = (0.4, 6.0)
```
The frequency band is "wide open on purpose — the gate that separates ring from step-settle is ζ" (ζ ≈ 0.4+ transients are closed-loop shapes). `RING_R2_MIN`'s docstring (l. 74–79) warns not to raise it back toward mean-referenced values; 0.30 keeps 15 of the 24 episodes passing the other gates, including every visually verified ring.

### 3.3 Loss — `score_candidate` (l. 168–200)

Per kept episode, over the same window `[ep.t0, ep.t1]` on both traces:
```python
term_amp  = log(s_amp / r_amp)**2                      # band-RMS ratio, always defined
term_freq = log(sf["freq_hz"] / rf["freq_hz"])**2      # only if sf["r2"] >= 0.3
term_zeta = (sf["zeta"] - rf["zeta"])**2               # only if sf["r2"] >= 0.3
loss += W_AMP*term_amp + W_FREQ*term_freq + W_ZETA*term_zeta
```
with `W_AMP, W_FREQ, W_ZETA = 1.0, 1.0, 4.0` (l. 90), and the total divided by `max(len(episodes), 1)`.

The asymmetry in `episode_features` (l. 145–165) is critical and easy to get wrong on a port: the **real** trace is re-fitted with `seed_f0=True` (its own detected frequency), the **twin** is fitted **blind** (FFT-seeded) — "a seeded fit confines the frequency to ±60 % of the seed, so seeding the twin with the real frequency would clamp the very number the loss compares and understate any frequency mismatch."

When the twin's own fit is untrusted (`r2 < 0.3` — overdamped or barely moving) only the amplitude term contributes, on purpose: amplitude already punishes a dead twin hard via `log²` of a near-zero ratio, and "charging fabricated frequencies from garbage fits would steer the optimizer with noise."

### 3.4 Method — `fit(...)` (l. 220–390)

1. Load `RunData`, load base `ActuatorModel` from `DEFAULT_CKPT`, detect targets. Refuse if none.
2. Refuse if `t_max` leaves no episode inside the fit window ("an empty set would make the loss identically 0 for every candidate — refuse instead of 'fitting'").
3. Seed `x0 = space.to_vector(0.03, 0.02, 1.0, 2.0e-4)`, i.e. `[log(jd), jf, log(tb), log(b1)]`. **Documented trap** (l. 250–254): "Nelder-Mead's initial simplex steps a coordinate whose seed is exactly 0 by only 2.5e-4, and `log(tendon_damping = 1.0)` IS 0 — seed tb away from 1.0 if you actually want it explored."
4. Evaluate the **baseline** with the pre-elasticity shipped constants `{2.25, 0.70, 50.0, 0.0}`, deliberately bypassing the fit-space clipping "because the guard must be measured against the constants that actually shipped."
5. `scipy.optimize.minimize(..., method="Nelder-Mead", options={"maxfev": budget, "xatol": 0.05, "fatol": 5e-3})`, default budget 60 evaluations, default `t_max = 60.0` s.
6. Take the **best evaluated point, not scipy's simplex centroid** ("with a 50-eval budget those can differ").
7. **The guard**: `guard_ok = isfinite(baseline_q_rms) and best.q_rms_deg <= baseline_q_rms * (1 + GUARD_TOL)`, `GUARD_TOL = 0.10`. A failed baseline must fail the guard, not pass it vacuously. On violation `fit_bounce.json` is written "for forensics" and `bounce_checkpoint.npz` is **not** — "a fit that bought its ring by wrecking the trajectory match must not become a checkpoint anyone can load by accident."
8. Validation pass: best params over the **full** run (`t_max=None`), producing `compare_metrics` plus the per-episode real-vs-twin `episode_rows` table.

Each candidate rollout (`rollout_candidate`, l. 203–217) rebuilds the `ActuatorModel` with `damp_b1=np.full(3, params["damp_b1"])` and forwards the other three as `sim_kwargs` to `twin_rollout`. Rollout exceptions and `not tw.valid` both return loss `1e3` and are recorded with their reason.

---

## 4. `twin_compare.py` — open-loop scoring

`…\UMArm_SIM\twin_compare.py`

### 4.1 What a rollout is

The recording carries the **exact wire setpoints per tick on a strictly increasing clock**; feeding that timeline through `replay.rollout` drives MuJoCo **open loop** with no controller, "so what differs between the recorded q and the rolled-out q is the PLANT MODEL and nothing else."

`twin_rollout(run, *, ckpt=None, actuator=None, sim_kwargs=None, t_max=None, progress=None) -> TwinResult` (l. 275). Builds a **fresh** `SimArm` every call — "replaying two runs on one arm would chain their state". Runs in `CHUNK_TICKS = 2000` chunks so a progress callback fires every couple of seconds of compute.

### 4.2 `TwinResult` — the exact data shapes (l. 102–158)

```
row_index      (n,)      indices into run.rows
t              (n,)      seconds since the recording's first row
q_sim_rad      (n, 12)
twin_defl_rad  (n, 12)   twin deflection from the common reference window
real_defl_rad  (n, 12)   recorded deflection, NaN where the row had no pose
p_sim_pa       (n, 24)   twin true gauge pressure
ref_rows       (k,)  sim_ref_rad (12,)  real_ref_rad (12,)   [RAW mocap coords]
gaps: list of {"t0","t1","kind": "vent"|"unknown"}
ctrl_min_n: float        most negative tendon command seen (clamp is -4000)
valid: bool ; reason: str ; meta: dict
```
Persisted with `np.savez_compressed`, `allow_pickle=False`, `gaps`/`meta` as JSON strings.

### 4.3 Timeline reconstruction — `build_timeline` (l. 166–238)

Three kinds of synthesised tick, each with a stated reason:
- **Warmup pre-roll**: `BusRuntime.warmup` ticks the baseline before the loop exists. The rollout is prefixed with `n_warm = round(warm_s/dt)` ticks at their true instants, setpoints `= sp[:1]` (the first recorded row's command — both are the controller's idle setpoints), `sample_row = −1`. Any quiet stretch after warmup "plays out exactly as the metal did: the node failsafe trips and traps the pressure."
- **Watchdog vents**: any inter-row gap `> GAP_S = 0.5` s that brackets a `watchdog` event (window `g0 − 1.0 ≤ t_event ≤ g1`) gets explicit **zero-setpoint** frames synthesised on a `VENT_TICK_S = 0.05` s grid for at most `VENT_S = 6.0` s, then silence. Without this "a rollout that held the pre-trip setpoints across that hole would show a fully pressurised twin against a deliberately emptied arm and call it model error."
- **Other long gaps** get nothing — the failsafe trap is the closest reconstruction — and are recorded as `kind: "unknown"`.

Finally everything is stably sorted by time and strictly-increasing-deduplicated (`keep[1:] = np.diff(tt) > 0.0`).

The clamp stays armed: `rollout(..., assert_no_clamp=True)`. Measured headroom at the 35 psi envelope is ~500 N against the 4000 N clamp, so contact "would mean the force law was driven somewhere new — a finding, not a nuisance." A `ClampViolation` returns an **explicitly invalid** empty `TwinResult` rather than quietly different physics.

### 4.4 Alignment — the part most likely to be got wrong on a port

Both traces are re-referenced to a **common settled window** (`REF_WINDOW_S = 0.5` s). `pick_reference_rows` (l. 241–267) preference order:
1. the window `[t_ref − 0.6, t_ref − 0.1]` before the first `benchmark_start` event;
2. else the window before the first `FLAG_ENGAGED` row;
3. else the **last** posed window (up to 80 rows) — "a run engaged from row 0 has no settled pre-engage hold; the end of the recording is a stiller reference than the start, where the arm is still pressurising."
Only rows with `FLAG_HAVE_POSE` participate.

Two rules that carry the correctness:
- **Both references average over the SAME rows** — the reference rows that actually have a rollout sample, found with `searchsorted` + an equality check. "Averaging the two traces over different sets would bake their difference into every later number." If that intersection is empty the result is **invalid** with reason `"reference window has no rollout samples (fell in a gap)"`.
- The real reference is taken in **RAW mocap coordinates** (`np.radians(rows["q_raw_deg"])`), deliberately not `q_raw − q_zero`: "`q_zero` is a per-row column that STEPS when the operator re-zeros mid-session, and a constant reference cannot cancel a step in the origin — a perfect twin would read a bogus error for everything after the press."

`t_max` truncation refuses loudly if it would cut away the reference window (l. 306–312) rather than burning a fit budget on invalid rollouts.

### 4.5 Caching

`twin_for(run_path, ...)` caches to `twin.npz` beside the recording. Key = `f"algo{_ALGO_VERSION}|{path:size:mtime_ns of the replayed bytes}|{same for ckpt}"`. `_ALGO_VERSION = 3` and the comment names what bumped it: "the plant gained the McKibben elasticity/damping model … a v2 twin rolled under the overdamped constants must not answer for it." A rollout with a custom actuator, custom `sim_kwargs` or a `t_max` gets `cache_key = None` so it "must never carry the canonical key — `twin_for` would serve it as THE twin."

### 4.6 The metrics — `compare_metrics(run, tw)` (l. 456–513)

Rows after the **first `unknown` gap are excluded from everything** ("the drive that produced them is not fully known"), and only rows where `real_defl_rad` is finite on **all 12** joints are scored. A window with `< 20` usable samples returns `None` rather than a number.

Per window (`score`), all per-joint arrays of length 12:
```
rms_deg      = degrees(sqrt(mean(err**2, axis=0)))         err = twin_defl - real_defl
rms_deg_mean, rms_deg_max
nrmse        = rms / max(spread, 1e-9)   spread = RMS(real - mean(real)) per joint
nrmse_mean
peak_err_deg = degrees(max|err|)
n
```
"NRMSE is RMS(twin − real) over RMS(real − mean(real)): **1.0 means the twin predicts the arm no better than 'it does not move'**, which is the null every RMS number must be read against."

### 4.7 How results are split by population — **they are not**

There is **no population split anywhere in `twin_compare.py`**. The only splits are:
- **temporal**: `out["segments"]`, one entry per `benchmark_start` event, matched to the `benchmark_end` of the same `name` (falling back to `st + seconds`), plus one `out["overall"]`;
- **per-joint**: every metric is a 12-vector plus its mean/max.

The whole repo's only actuator-heterogeneity axis is **segment**, expressed as 3-vectors expanded by `np.repeat(x, 8)` (`actuator_model.py:368-378`: `coeff_per_act`, `bf2_per_act`, `l0_per_act`, `damp_b1_per_act`). Per-node character is limited to three scalars — `fill_gain`, `vent_gain`, `leak_pa_s` — with one shared flow net. **Adding a two-population split is new work, not a port.** See §8.

---

## 5. `force_audit.py` — the five re-runnable measurements

`…\UMArm_SIM\force_audit.py`. **Nothing here changes the model** — it reads the shipped checkpoint, scales `coeff`/`bf` **in memory** (`scaled()`, l. 172–180) and reports. `PARTS = ("geometry", "static", "ring", "replay", "collision")`.

Reference constants: `BENCH_CKPT = REPO/UMArm_COLLAB/assets/bench_actuator.npz`; `REAL_RUN = …/data_collection_2026-08-16/flight_records/20260816_222851_real_rec1`; `REAL_RING_HZ = 1.83`, `REAL_RING_ZETA = 0.054` (53 episodes); `REAL_STATIC_8PSI_DEG = (10.86, 27.79)`; `ANCHOR_RADII_M = {"JA1_seg1": 0.100, "JA1_seg23": 0.050, "JA2": 0.050, "AO": 0.030}`; `OPERATOR_ARM_KG = 1.2`, `OPERATOR_ACTUATOR_KG = 0.030`.

1. **`geometry`** (`audit_geometry`, l. 100–150) — pure arithmetic on the law's own constants. Reads Chou–Hannaford backwards: `Nf = √(1/(4π·coeff))`, `D0 = Bf/(π·Nf)`, braid angle `cos θ = l0/Bf`, rest diameter `D0·sin θ`, bladder volume `π(D/2)²·l0`. Then checks that against the arm: four muscles share each anchor ring, so it reports `four_muscle_area_mm2 / AO_ring_area_mm2` as `overfill_on_AO_ring`, and `coeff_scale_for_diameter` for 12/15/20/25 mm muscles. Also `force_at_{8,25,35}psi_n` (`_force_at`: `coeff·p·(Bf² − 3l0²)`, `p = psi·6894.757`) and `stiffness_at_8psi_n_per_m` (`_stiffness_at`: `|coeff·p·6·l0|`). *Can show the modelled muscle cannot physically exist; cannot say what the right force is.*
2. **`static`** (`audit_static`, l. 187–231) — one muscle, fixed pressure, settle, re-running the 2026-08-12 campaign's own experiment at coeff scales `(0.25, 1.0, 4.0)`, psis `(8, 25)`, nodes `(1, 9, 17)`, 3 s at 5 ms ticks, reporting `max|q|` in degrees and `spread_frac`. Verdict, hard-coded because it is the finding: the anchored law goes slack at `ℓ = Bf/√3`, so the equilibrium is **geometry-limited, not force-limited** — a 16× coeff change moves the settled angle a few per cent, hence "a pressure-vs-angle campaign is blind to the force scale by construction."
3. **`ring`** (`audit_ring`, l. 238–285) — free ringdown against the measured 1.83 Hz per coeff scale, via `bounce_demo.run_poke` + `ring_metrics`. Fits `f²` affine in the coeff scale and reports `slope`, `intercept`, `gravity_share_at_x1 = b/(a+b)` and `scale_for_real_hz = (1.83² − b)/a`. Also reports the fixed lever `F/(dF/dℓ) = (3l0² − Bf²)/(6 l0)` — "force and stiffness cannot be scaled apart". *The one experiment that constrains the scale, and it pulls the opposite way from everything else.* `poke_n = 2.0` N rather than `bounce_demo`'s 12 N default, because the twin's ring damping is amplitude-dependent (ζ 0.099 at 3.9 deg, 0.059 at 23.9 deg) while the real episodes are 0.46–2.9 deg.
4. **`replay`** (`audit_replay`, l. 315–364) — "**acceleration is the closest thing to a force measurement this repo has**". Rolls the real recording through the twin at coeff scales `(0.5, 1.0)` and compares peak `|q̇|` and peak `|q̈|` per joint. Derivatives via `_savgol_derivatives` (l. 292–312): Savitzky-Golay, **window 17 samples (107 ms at 159 Hz), cubic**, `deriv=1` and `deriv=2` with `delta=dt`, `mode="interp"`, non-finite rows dropped, **first second discarded** (start glitch) — because raw differentiation is unusable: "one frame of the 2026-08-16 recording carries 11 rad/s and a quarter of the rows repeat the previous mocap frame." Reports `qdot_ratio_median`, `qddot_ratio_median`, `qddot_ratio_per_joint`. Explicitly **does not use the cached `twin.npz`** (algo-2 caches score the twin as four times too slow — the exact opposite answer).
5. **`collision`** (`audit_collision`, l. 371–392) — one `PR.StrikeSpec(target="cover2", strike_dir_deg=90.0, psi=35.0)` trial per coeff scale, reporting `peak_contact_n`, `gauge_peak_n`, `impact_speed_mps`, `on_target`. Separates a momentum impact (scales with speed) from a quasi-static push-through (scales with the law) — "the number the operator's question is really about."

CLI: `--parts`, `--quick` (skips `replay` and `collision`, the two slow ones), `--scales` (default `0.25 0.5 1 2`), `--json`.

---

## 6. `umarm_sim.xml` — the shipped model

193 lines, generated, header comment records `base_pos=(0.22, -0.443, 1.042), base_yaw_deg=0, joint_damping=0.026, joint_frictionloss=0.025, tendon_damping=1`.

- `<option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast" iterations="100"/>` (l. 7).
- `umarm` default class (l. 23–29): joint `damping="0.026" frictionloss="0.025" armature="0.01" limited="true" range="-0.69813170079773179 0.69813170079773179"`; `<tendon width="0.0035" damping="1"/>`; `<geom contype="0" conaffinity="0"/>`.
- `base_plate` at `pos="0.22 -0.443 1.042" quat="1 0 0 0"`, geom mass `0.74` (= 0.5 + 0.24).
- Fitted lengths visible in the body positions: `seg1_plate2` at `z = −0.22467284`; `jd2_bracket` `0 → −0.04806482` (mass 0.48); `seg2_link` at `−0.04806482`; `seg2_plate2` at `−0.19411817`; `jd3_bracket` `→ −0.04676957` (mass 0.48); `seg3_plate2` at `−0.18537953`. Rods run `−0.0285 → −(span − 0.0285)` with `size="0.009"`, mass 0.02 each. Disks at ±0.004 about the ring z, `size="0.036"` (= AO + 0.006), mass 1e-4.
- Ring radii as generated: `s1_plate1_*` at 0.10; `s2/s3_plate1_*` at 0.05; `s{n}_plate2_*` at 0.05 (site coords `±0.03535533906`); `s{n}_link_*` at 0.03 (link_bot) and 0.03 → but written as `±0.02121320344` on the 45-deg upper ring, i.e. `0.03·cos45`. Azimuths match `LOWER_RING_DEG`/`UPPER_RING_DEG` exactly: `s1_plate1_a1` at `(−0.1, ~0, 0)` = 180 deg, `a2` at 270, `a3` at 0, `a4` at 90.
- `seg3_plate2_geom` mass `0.12`; `ee_assembly` = `ee_rod1` (0 → −0.078, mass 0.06) + `ee_rod2` (−0.078 → −0.155, **straight, on the z axis**, mass 0.06) + `ee_tip_sphere` at `−0.155` (mass 1e-4); sites `ee_wrist1` at `−0.06`, `ee_tip` at `−0.155`.
- 24 `<spatial>` tendons, each exactly two `<site>` children (l. 141–164); 24 `<motor>`s, all `gear="1" ctrlrange="-4000 0" forcerange="-4000 0"` (l. 168–191).
- No `<contact>`, no `<sensor>`, no `<keyframe>`, no mesh assets.

---

## 7. `viz/mjcf_canarm.py` → physics twin: precisely what must be added

`C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\viz\mjcf_canarm.py`. Measured today, compiling the room with `canarm_mount=((0,0,0),(0,0,0))`:

```
nq 12   nu 0   ntendon 0   timestep 0.002
canarm_seg1_link 0.02   canarm_seg1_plate2 0.0602
canarm_seg2_link 0.02   canarm_seg2_plate2 0.0602
canarm_seg3_link 0.02   canarm_seg3_plate2 0.14
canarm_base 0.5 (static)      MOVING TOTAL 0.3204 kg
```

What it already has and the twin keeps: the topology (three link bodies + three distal-plate bodies, twelve hinges, distal axes `(±1,1,0)/√2`), the JD spacers belonging to the **parent** body, the six named `canarm_plate0..5` sites, the mocap-body mount (`canarm_mount`, `mocap="true"`) with `build_arm_xml(chain_m=…)`, and the `_compiles` validation pass.

### 7.1 Nine additions, in dependency order

1. **The proximal hinge order must become `y` then `x`** — see gotcha 1. This is a geometry change, not a dynamics one, and it invalidates the existing `self_check` assertion if done naively.
2. **Twenty-four attachment-site rings** — four rings per segment at radii `JA1`, `AO1`, `AO2`, `JA2` from `canarm_params.CANARM_PARAMS`: `JA1 = JA2 = 0.047 m`, `AO1 = AO2 = 0.028 m` on **all three** segments (vs RS485's 0.10/0.05/0.03). `AA1 = AA2 = 0.0437125 m` sets the ring z-heights: `z_top_ring = −AA1`, `z_bot_ring = −(AA1 + LL)`. Note the display file's `DEFAULT_PLATE_RADII = (0.055, 0.055)` and `BASE_PLATE_RADIUS = 0.105` (l. 119–120) are **cosmetic RS485 leftovers** — in a twin the ring radius *is* the moment arm and must come from the params table.
3. **Twenty-four two-site spatial tendons + twenty-four `<motor>`s**, `gear="1"`, `ctrlrange`/`forcerange` set to the CAN arm's own clamp (the RS485 −4000 N is sized to *its* force law and envelope; keep the pattern, re-derive the number, and keep `assert_no_clamp`).
4. **A ring azimuth table that is the CAN arm's own.** `UMArm_KINEMATICS.canarm_actuators.MEASURED_JOINT_PAIRS` (measured 2026-08-21, `MEASURED = True`) shows **segment 1 is rotated +90 deg relative to the legacy table** while segments 2 and 3 reproduce it exactly, pairing and sign — "the signature of the top regulator platform having been remounted a quarter turn round." So segment 1's azimuths differ from segments 2–3's by 90 deg. `SEGMENT_BLOCKS = (range(0x101,0x109), range(0x109,0x111), range(0x111,0x119))`, verified 24/24 by dominant `dq`.
5. **Masses and the `mass="0.0001"` convention.** The display model carries 0.32 kg moving against the RS485 twin's 1.261 kg on a *shorter* arm — the CAN arm's spans are 265/234/230 mm vs 225/194/185 mm. Nothing in this workspace has measured the CAN arm's mass distribution; the only published numbers are the operator's ~30 g per actuator × 24. This is the single largest unmeasured input to a dissipation fit, since ζ and f both read mass.
6. **Timestep 2 ms → 1 ms**, and every downstream consumer must read `model.opt.timestep`. `digital_twin/__init__.py` already pins `TIMESTEP_S = 0.001` for exactly this purpose.
7. **Integrator stays `implicitfast`** (already correct) — required because the twin rewrites `model.tendon_damping` at every node pass.
8. **The three MJCF dissipation tunables as generator arguments.** The display class currently hard-codes `damping="0.05" frictionloss="0" armature="{armature}"` (l. 355). A twin needs `joint_damping`, `joint_frictionloss`, `tendon_damping` templated, defaults carrying an inline provenance comment, and the tendon default class emitted at all (there is no `<tendon>` block today).
9. **Keep everything non-colliding and keyframe-free**, and keep the "edit the generator, not this file" header plus the `genparams` echo.

### 7.2 What must NOT be inherited

`digital_twin/__init__.py` states it and it is right: the RS485 `actuator_model` checkpoint (flow net + `coeff`/`bf`/`l0`/`damp_b1`) is fitted to a 7 mm solenoid pair on a PWM node. Loading it into the CAN plant "produces a twin that is confidently wrong and reports a good residual on the trajectory it was fitted on." `BF_PRETUNE_SCALE = 1.16` and `MUSCLE = {"Bf": (0.1304, 0.132, 0.1208), "Nf": (0.688, 0.808, 0.8), "LBase": (0.1736, 0.1544, 0.1472), "LActOffset": (0.0664, 0.0576, 0.052)}` are RS485 legacy constants, already 0.8-prescaled, for a shorter arm.

---

## 8. The two populations — where they fit in this architecture

The single most useful structural fact I found: **on the CAN arm, the population split coincides exactly with the segment split.** `SEGMENT_BLOCKS` puts `0x101–0x108` (the eight TLE92464 proportional DVP boards) on **segment 1** and `0x109–0x118` (the sixteen legacy 7 mm solenoid boards) on **segments 2 and 3**. Under the RS485 naming convention `gi = (seg−1)*8 + local`, that is `pam_1..8` = DVP and `pam_9..24` = solenoid.

The consequence for the port: the RS485 model's *only* actuator-heterogeneity axis — per-segment 3-vectors expanded by `np.repeat(x, 8)` (`coeff`, `bf`, `l0`, `damp_b1`) — **already has exactly the granularity a population split needs**, provided the CAN twin keeps the `gi = (seg−1)*8 + local` identity. `damp_b1` can go from "one value for all three segments" (the RS485 fit's honest admission that its recordings could not separate segments) to two values: one for segment 1, one shared by 2–3.

Where a genuine second population must be built rather than reused:
- **the drive input.** The RS485 flow net takes `valve ∈ {CLOSED, INLET, EXHAUST}` as a **three-way one-hot** (`_norm_x`, `actuator_model.py:409-425`). A DVP board driven by a coil current code has a continuous input; a one-hot cannot represent it. The top eight need their own `f_theta` input layout (`N_IN = 6` becomes population-dependent), which means two nets or one net with a population feature — a design decision, not a port.
- **the scoring split.** `compare_metrics` splits temporally and per-joint only. Since joints 0–3 are segment 1 (`canarm_actuators.JOINT_NAMES` = `s1.u1.t1, s1.u1.t2, s1.u2.t3, s1.u2.t4, …`), a population split of the per-joint 12-vectors is `[:4]` vs `[4:]` and is cheap to add — and worth adding, because a single `nrmse_mean` over two plants averages away exactly the difference the twin exists to represent.
- **the four dissipation scalars.** `joint_damping` and `joint_frictionloss` are *spine* terms (bearings, rod flex) and should stay global — they are not a valve property. `tendon_damping` and `damp_b1` are *muscle* properties and are the ones that plausibly split. That takes `FitSpace` from 4 to 6 dimensions, on a Nelder-Mead budget of 60 rollouts; expect to raise the budget or fit the populations on separate excitations.

---

## 9. Known board faults that will contaminate a fit

`canarm_actuators.KNOWN_BOARD_FAULTS`: `0x110` "leaks from the supply side; reads +5.1 psi at rest" (it drifted 0.1 → 7.0 psi across a sweep when left disabled — which is why every campaign here holds unused boards *enabled* at a small idle setpoint), and `0x104` "reads +1.1 psi at rest". Both are segment-2 and segment-1 boards respectively, i.e. one in each population. A `leak_pa_s` per-node scalar (the RS485 model has one, fitted separately by `fit_leak` on ≥2 s guaranteed-closed segments) is the right home for `0x110`; the rest-offset is a calibration matter that must be applied before any `p_pa` reaches the force law.

## 10. Suggested port order

1. `ring_analysis.py` — verbatim, zero dependencies, and it is the instrument every later number is measured with. Port it first and pin it with its own test.
2. `mjcf_generator.py` — with `chain_m`/`params` as arguments, the `yx` hinge order, the CAN ring table, and an FK test against `fkine(..., order="yx", params=CANARM_PARAMS)` at 1e-9 plus a tendon-length test at 1e-9.
3. `twin_compare.py` — but its whole timeline reconstruction is written against `UMArm_CONTROL.flight_recorder.RunData` (`wire_timeline`, `events`, `flag`, `rows["q_raw_deg"]`, `rows["t_mono"]`). **Decide the CAN recording schema before porting this file**; `digital_twin/data_schema.md` exists and I did not read it.
4. `force_audit.py` — as soon as there is a checkpoint to audit; it is the cheapest guard against an overtuned force law.
5. `fit_bounce.py` — last, because it needs 3 and a ≥150 Hz recording of a real ring. The 150 Hz CAN sync is right at the edge: the RS485 fit needed 160 Hz to resolve a 1.83 Hz / ζ ≈ 0.054 ring, and the 20 Hz campaign was blind to it. Confirm the CAN arm's ring frequency before assuming 150 Hz suffices — a longer, heavier arm will ring lower, which helps.

## KEY FILES
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\mjcf_generator.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\umarm_sim.xml
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\ring_analysis.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\fit_bounce.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\twin_compare.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\force_audit.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\viz\mjcf_canarm.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\canarm_params.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\canarm_actuators.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_MOCAP\canarm_frames.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\UMArm_KINEMATICS\fkine.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\digital_twin\mjcf_generator.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\digital_twin\__init__.py
- C:\ESP\ESP_Projects\UMArm_koopman_compliance_control_espproject\viz\self_check.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\actuator_model.py
- C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\UMArm_SIM\sim_core.py

## GOTCHAS
- PROXIMAL HINGE ORDER: viz/mjcf_canarm.py:291-292 declares uj{2n-1}_x before uj{2n-1}_y, i.e. fkine order='xy', and viz/self_check.py:81 verifies it against K.ujoint_centres(q, params=cp.CANARM_PARAMS) with fkine's DEFAULT order='xy'. But the CAN arm's measured composition is UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER = 'yx' (measured 2026-08-21; it took held-out u-joint-centre error from 4.6 mm RMS / 20.4 mm worst to 2.0 / 7.0 mm), and canarm_frames.q_from_plate_frames reads q under 'yx'. I measured the disagreement: over 200 random q in +-0.6 rad the two orders differ by up to 0.189 m at a u-joint centre (0.0116 m at a single segment with t1=t2=0.3 rad). A physics twin must declare the y hinge FIRST on every proximal pair, and then qpos[0] is uj1_y, so the fkine q vector maps as qpos = [t2, t1, t3, t4, ...] per segment - a remapping table the RS485 generator deliberately does not need. I did not trace the live viewer's q source end to end, so I cannot say whether today's display is drawing mocap q under the wrong order or is fed an xy-consistent q.
- PLATE RADII IN THE DISPLAY FILE ARE COSMETIC RS485 NUMBERS. viz/mjcf_canarm.py:119-120 sets DEFAULT_PLATE_RADII = (0.055, 0.055) and BASE_PLATE_RADIUS = 0.105, and its own comment says they are 'cosmetic - they size the drawn discs and nothing else'. The CAN arm's real ring radii are JA1 = JA2 = 0.047 and AO1 = AO2 = 0.028 on all three segments (canarm_params.CANARM_PARAMS). In a twin the ring radius IS the muscle moment arm; using the display values would inflate every joint torque by 17 % on the plate rings and by 96 % on the link rings.
- RING AZIMUTHS DO NOT TRANSFER FROM RS485. mjcf_generator LOWER_RING_DEG=(180,270,0,90)/UPPER_RING_DEG=(225,315,45,135) are a verified RS485 seat table. canarm_actuators.MEASURED_JOINT_PAIRS (measured, 24/24 boards) shows segment 1 (0x101-0x108) is rotated +90 deg relative to the legacy table while segments 2 and 3 reproduce it exactly - the top regulator platform was remounted a quarter turn. So the CAN arm needs a per-segment azimuth table, not one shared pair of tuples, or pam_i stops being node i's muscle and every agonist sign flips on segment 1.
- MASS IS UNMEASURED AND CURRENTLY ~4x TOO LIGHT. Compiled today, viz/mjcf_canarm.py's CAN arm carries 0.3204 kg of moving mass (seg links 0.02 each, seg1/2_plate2 0.0602, seg3_plate2 0.14) against the RS485 twin's 1.261 kg on a SHORTER arm (CAN spans 265/234/230 mm vs 225/194/185 mm). Frequency and zeta both read mass, so a dissipation fit run against these masses will absorb the mass error into damp_b1 and report a good loss.
- TIMESTEP: the display model is 2 ms (mjcf_canarm.py:338). The twin must be 1 ms - digital_twin/__init__.py pins TIMESTEP_S = 0.001 and explains why (sim_core counts node-logic passes in whole quanta via NODE_LOGIC_EVERY, so halving the timestep silently halves the firmware control period). Every consumer must read model.opt.timestep rather than assume it.
- fit_bounce SEED TRAP, documented at fit_bounce.py:250-254: Nelder-Mead's initial simplex steps a coordinate whose seed is exactly 0 by only 2.5e-4, and log(tendon_damping = 1.0) IS 0, so seeding tb at 1.0 leaves it effectively unexplored while the run still reports a converged fit.
- fit_bounce ASYMMETRIC SEEDING (episode_features, seed_f0): the REAL trace is re-fitted seeded with its own detected frequency (+-60 % band); the TWIN is fitted blind (FFT-seeded). Seeding the twin with the real frequency clamps the very number the loss compares and silently understates any frequency mismatch.
- ring_analysis r2 IS RING-BEYOND-TREND, not the usual mean-referenced r2 (ring_analysis.py:151-155): ss_tot is the variance of the DETRENDED residual. It reads far lower than an ordinary r2 - a visually clean ring inside a post-step settle scores ~0.3 - which is why RING_R2_MIN is 0.30 and why fit_bounce's docstring explicitly forbids raising it back toward old values to widen a thin target set.
- twin_compare ALIGNMENT: both traces must be averaged over the SAME reference rows (the intersection of the settled window with rows that actually have a rollout sample) or their difference is baked into every later number, and the real reference must be taken in RAW mocap coordinates, never q_raw - q_zero, because q_zero STEPS on a mid-run re-zero and a constant reference cannot cancel a step in the origin. If the intersection is empty the result must be marked INVALID, not re-referenced elsewhere.
- twin_compare CACHE POISONING: any rollout with a custom actuator, custom sim_kwargs, or a t_max must get cache_key = None, or twin_for will serve a fit candidate as THE twin. _ALGO_VERSION (currently 3) must be bumped whenever the rollout or alignment changes meaning; force_audit.audit_replay refuses the cached twin.npz outright because algo-2 caches score the twin four times too SLOW - the exact opposite answer.
- NO POPULATION SPLIT EXISTS TO PORT. twin_compare.compare_metrics splits only temporally (benchmark_start/end events) and per-joint (12-vectors). The RS485 model's whole heterogeneity axis is segment, as 3-vectors expanded by np.repeat(x, 8). On the CAN arm that axis happens to coincide with the population split (SEGMENT_BLOCKS puts 0x101-0x108 = DVP on segment 1, 0x109-0x118 = solenoid on segments 2-3), so the per-segment machinery can carry it - but the flow net cannot: its valve input is a three-way one-hot (CLOSED/INLET/EXHAUST) that cannot represent a proportional coil-current code.
- KNOWN BAD BOARDS that will contaminate any fit: canarm_actuators.KNOWN_BOARD_FAULTS - 0x110 leaks from the supply side and reads +5.1 psi at rest (it drifted 0.1 to 7.0 psi across a sweep when left disabled, which is why campaigns hold unused boards ENABLED at a small idle setpoint), 0x104 reads +1.1 psi at rest.
- SAMPLE RATE: the RS485 bounce fit needed 160 Hz recordings to resolve a 1.83 Hz / zeta 0.054 ring; the 20 Hz sysid campaign was blind to it and shipped engineering-guess dissipation that made the twin unable to ring at any parameterization. The CAN bus syncs at 150 Hz. Confirm the CAN arm's actual ring frequency before assuming 150 Hz resolves it.
- twin_compare and fit_bounce are written entirely against UMArm_CONTROL.flight_recorder.RunData (wire_timeline(with_rows=True), events with kinds 'warmup'/'watchdog'/'benchmark_start'/'benchmark_end', flags FLAG_ENGAGED/FLAG_HAVE_POSE, rows['q_raw_deg'], rows['t_mono']). I did NOT read that module. The CAN recording schema must be settled (digital_twin/data_schema.md exists and I did not read it) before either file can be ported.
