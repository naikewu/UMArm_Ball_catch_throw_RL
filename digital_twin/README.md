# `digital_twin/` — the plan, before the code

This package is a **documented skeleton**. Every module raises
`NotImplementedError`, carries the role its RS485 counterpart plays rewritten for
this arm, and names the absolute path of that counterpart so the port is a read
rather than a search. Nothing here has been fitted, and nothing here should be
trusted to produce a number until it has been.

The layout mirrors `C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\` deliberately:

| file | role, for the CAN arm |
| --- | --- |
| `mjcf_generator.py` | the twin's geometry: the display chain plus tendons, masses and the fittable MuJoCo tunables |
| `sim_core.py` | `SimArm` — model, data, one firmware model per board, one mutex, `advance_to` on a 1 ms grid |
| `actuator_model.py` | flow in, force out — the piece that is **not** inherited |
| `replay.py` | offline rollouts: no threads, no wall clock, bit-identical repeats |
| `sim_master.py` | the transport shell, wrapping `TLE_PCB/tlelib.backend.Backend` rather than reimplementing it |
| `sim_mocap.py` | simulated plate poses pushed through the **real** receiver (mostly done already, in `UMArm_MOCAP/sim_stream.py`) |
| `twin_compare.py` | roll a real recording through the twin, open loop, and score it |
| `ring_analysis.py` | the ringdown instrument, applied identically to recording and rollout |
| `fit_bounce.py` | fit the four dissipation scalars to the ring episodes |
| `force_audit.py` | five re-runnable measurements that change nothing |

---

## The one rule everything else follows

*A controller must not be able to discriminate between the real robot and the
simulator.* Same interface, same timing, same node behaviour. Anything that makes
the twin easier to build at the cost of that property is not a shortcut, it is a
different project.

## Two design decisions carried in from the RS485 twin

Both were paid for there, and neither is re-derivable from the code that will
replace them.

1. **The 1 ms timestep is locked.** `sim_core` counts node-logic passes in whole
   quanta (`NODE_LOGIC_EVERY`), so the firmware's control period is an integer
   count of timesteps rather than a duration. Halving the timestep silently
   halves that period, with nothing raising and every rollout still looking
   plausible. Read `model.opt.timestep`; never assume it. The constant lives at
   `digital_twin.TIMESTEP_S` so a test can assert on it without building a scene.
2. **`sim_core` must take `xml=`.** A merged multi-robot scene — the CAN arm, the
   RS485 arm and the Gen3 in one room — is composed elsewhere and handed in.
   `viz/mjcf_canarm.py::build_room_scene` is the display-side version of that
   merge today. The RS485 workspace states the dependency the other way round at
   `UMArm_COLLAB/collab_scene.py:1-7`: the merge exists *because* `SimArm` accepts
   a scene, rather than the arm being re-authored inside the merge.

---

## The plan: collect first, fit second

### Why not inherit the RS485 fits

`UMArm_SIM/actuator_model.py` ships a fitted checkpoint
(`synthetic_pretrained.npz`) and `mjcf_generator.py` ships fitted dissipation
constants. **Neither transfers, and loading either would be worse than starting
empty**, because a checkpoint from the wrong plant reproduces the trajectory it
was fitted on and diverges everywhere else — which is the failure mode that looks
most like success.

The plants differ at the actuator, which is where the twin's error budget lives:

| | RS485 arm | CAN arm, top 8 | CAN arm, bottom 16 |
| --- | --- | --- | --- |
| valve | 7 mm solenoid pair | **TLE92464 proportional (DVP)** | 7 mm solenoid pair |
| drive | PWM duty against a fixed orifice | **coil current, 0–182.7 mA** | PWM duty |
| host codes | duty | **0–116 of a 120-code budget** (`I_mA = code × 200/127`; 4 codes reserved for the hardware dither overlay) | duty |
| shaping | — | inlet/outlet open and max codes, slew limit, dead zone, optional dither | — |
| firmware | mk8 node | `VEMA_MAX22200` | `Valve_not_embedded_XL` |
| link-loss | — | **outputs drop 500 ms after the last sync edge** | **no timeout; holds the last target indefinitely** |

What *does* transfer is structural, not numerical: the anchored McKibben force
law's shape (same braid, same bladder), the practice of keeping per-node scalars
rather than one global gain, the gradient check on the hand-written backward
pass, and the whole `ring_analysis` → `fit_bounce` → `twin_compare` instrument
chain.

Note also that a twin of *this* arm is a twin of **two populations**. Anything
that reports one number across all twenty-four boards reports a number that
describes neither, and the population a board belongs to is read from the
**variant byte the board reported**, never from its id range — a TLE board
legitimately sat at 0x114 during the 2026-08 bench session, and reading it on the
7 mm calibration is a 10 % error at the top of the range.

### Step 1 — collect

Record real CAN-arm data in the **`legacy_host` collection schema**: one JSON
object per 150 Hz cycle, written as `samples_chunk_NNNN_<reason>_<ts>.jsonl`
beside a `metadata.json` and a `manifest.json`, under a
`session_<YYYYmmdd_HHMMSS>/` directory. The schema is summarised in
[`data_schema.md`](./data_schema.md); the producer of record is the C++ backend
at `legacy_host/host/pc_backend/src/main.cpp`, driven by its
`start_data_collection` / `stop_data_collection` JSON commands
(`legacy_host/docs/mpc_integration.md:51`).

Reasons to use that schema rather than invent one:

* it already stores everything **raw** — ADC counts and radians — and leaves the
  psi conversion to the reader via the per-id `adc_ranges` snapshot, so a
  recalibration does not invalidate a recording;
* it stamps `can_sync_time_s` and `jitter_ms` per cycle, which is what lets a
  rollout be aligned to the metal rather than to a nominal 6.667 ms grid;
* it carries the mocap block *in the same object* as the joint state, with
  `latency_ms`, `age_ms` and `timestamp_offset_ms`, so the two halves of a sample
  are known to be the same instant rather than assumed to be;
* the predecessor `.npz` schema (windows of `(n_windows, window_samples, dim)`
  with pressures in Pa) exists and is documented, so continuity with older
  Koopman data is a conversion rather than a re-collection.

**One addition this arm needs: a `board_type` array**, twenty-four entries, one
per id, carrying the variant byte each board reported during the scan
(`0x00` = 7 mm, `0x01` = DT/big-valve, `0x02` = TLE/DVP). See `data_schema.md`.
Without it a recording cannot be split by population after the fact, and the
split is not recoverable from the id range.

Collection hygiene, from the RS485 campaigns:

* a rest capture per session, because **the joint zero is a mounting property,
  not a constant** — a straight arm does not read all zeros, and `q` is
  repeatable rather than absolute;
* the base correction re-fitted per session, per arm;
* the campaign must contain **rotations, not only translations**, wherever a
  transform is being solved: a campaign of pure translations returns a beautiful
  residual with an arbitrary answer;
* excitation that actually rings the arm. The RS485 20 Hz system-identification
  campaign could not see past its own damping constants, and it took 53 ringdown
  episodes to find out why.

### Step 2 — fit

In this order, because each step's instrument is the previous step's output:

1. `mjcf_generator` with **measured** chain lengths. Today
   `UMArm_KINEMATICS.canarm_params.MEASURED` is `False` and its numbers are the
   RS485 arm's, unscaled. Measure the five consecutive u-joint-centre gaps in a
   live mocap session first; everything downstream is a shape until then.
2. `actuator_model` per population, against the collected JSONL.
3. `ring_analysis` on the recording, to find out what this arm's ring actually is
   — frequency, damping ratio, episode count — rather than assuming the RS485
   arm's 1.83 Hz / ζ ≈ 0.054.
4. `fit_bounce` for the four dissipation scalars, against those episodes.
5. `twin_compare`, open loop, split by population.
6. `force_audit`, which changes nothing and is therefore worth running after
   every one of the above.

### Step 3 — the room

Only then is a merged multi-robot scene worth stepping rather than drawing. Until
then, `viz/` is the honest version of "show me the arm": forward kinematics of a
measured `q`, with the measured plate poses drawn on top, and no physics at all.

---

## What exists already, and is not in this package

* **`viz/mjcf_canarm.py`** — the display model, verified against
  `UMArm_KINEMATICS.fkine` to 4.4e-16 m over 200 random configurations. Its
  `build_arm_xml(chain_m=...)` is the argument-not-constant pattern this
  package's `mjcf_generator` must follow.
* **`UMArm_MOCAP/sim_stream.py`** — a real `CanArmMocap` fed by a producer
  thread, opening no socket. `sim_mocap.py` here is mostly the job of handing it
  `sim_core`'s `q` instead of its analytic sweep.
* **`TLE_PCB/tlelib/backend.py`** — the 150 Hz sync master `sim_master` must wrap
  rather than reimplement.
