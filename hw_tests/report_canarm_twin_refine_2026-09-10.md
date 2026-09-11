# The twin refined: its geometry, its axes, its masses, and a SIM adapter — CAN arm, 2026-09-10

The digital twin, the MuJoCo model of this arm driven by the same firmware logic
and fitted pneumatics as the metal, was scored against the real arm earlier the
same day and rendered side by side with it. That video showed three problems,
and the operator named them in `prompts/20260910_1_refine_digital_twin.md`: the
drawn arm did not look like the ProMax, the twin and the arm did not move alike,
and the operator window could drive only the metal. This report records what was
changed for each, what was measured, and what the result does not show.

Section 1 answers the operator's questions. Section 2 is the bulleted list of
every step taken to make the twin match the recordings. Sections 3–5 cover the
geometry, a defect in how the twin read its own joints, and the pneumatic model.
Sections 6–9 cover the mass fit and what the recordings pin. Section 10 is the
held-out score, section 11 the SIM adapter, and sections 12–15 the rest.

---

## 1. The short answer

* **The drawn arm was wrong, and so was its tendon routing.** The twin hung its
  muscles off the u-joint plates and ran every tendon to the far end of the
  link. It now carries the ProMax segment of Fig. 1C of 2606.29731v1.pdf: Ys on
  the centre rod, actuators on the Y tips, tendons around hub bearings to the
  u-joint brackets, and u-joints drawn as flat disks. The rest moment arm moved
  from 46.8 mm to **43.104 mm** (section 3).
* **The segment weights had never been tuned.** Until this session the twin
  carried masses carried over from the RS485 arm (1.833 kg moving), then the
  Koopman ProMax prior (2.220 kg). Neither is a weighing; the only mass measured
  on this arm is the operator's ~30 g per actuator.
* **They are now fitted** by CMA-ES together with the force law, a passive joint
  stiffness and the dissipation, with the masses modelled as the hardware is
  built. Held out, open loop, the twin went from **10.850 deg joint RMS / nrmse
  0.967 / mean correlation +0.756** at the start of the session to **2.124 deg /
  0.187 / +0.993** (section 10). A first fit with free per-body masses reached
  5.639 deg but put implausible weights on the bodies (sections 6–8).
* **The weight answer is 2.21 kg moving**: link bodies of about 0.56 kg each
  (0.32 kg of structure plus 0.24 kg of sleeves) and inter-segment connectors of
  0.22 kg. The recordings pin the link structure to x0.56–1.74; the connector
  masses are not pinned and sit at the Koopman prior. None of these is a
  weighing (section 9).
* **Segment 3 needs a passive joint stiffness** (3.95 N m/rad, pinned), which
  segments 1 and 2 do not. The model had no such term, and the first fit's
  implausible masses were compensating for its absence.
* **The twin had been reading `qpos` as `q`**, and a 90 deg rotation of the
  proximal seats was hiding it. Fixing the read alone took the held-out error
  from 11.746 to 9.691 deg (section 4).
* **The operator window can now start a simulated arm.** A `SIM - digital twin`
  adapter builds the twin behind the same `Backend` interface, at 143.6 Hz with
  99.67 % of replies heard against the real window's 145.43 Hz and 98.81 %
  (section 11).

## 2. Every step taken to make the twin match the arm

### Before this session (2026-09-10, 01:58–04:10)

* **Flow net by multiple shooting.** A 5-64-64-1 net predicts `dp/dt` from
  `[p, target - p, is_tle, l, ldot]`. It was trained on 25 056 windows of 200
  cycles from both sessions (400 warm-start epochs, then 150 shooting epochs on
  the GPU). Held-out pressure RMS fell from 20 308 Pa to **3682 Pa (0.534 psi)**
  over 10 824 windows from whole held-out episodes.
* **Per-board fill and vent gains** inside the same shooting loop: TLE fill
  0.94–1.07 and vent 0.81–1.08; 7 mm fill 0.68–1.15 and vent 0.73–1.36.
* **Per-board leak by least squares** on closed holds, outside the net: 24 of 24
  boards, median 33.4 Pa/s, largest 253 Pa/s on `0x102`.
* **The proximal seat azimuths were rotated** (`a35f94a`) after the twin's
  joints cross-correlated with the arm's as a permutation. The symptom was real;
  the diagnosis was wrong (section 4).
* **An outer coordinate search** (`outer_fit.py`, 2 rounds x 5 points on the
  first 15 s of `random_walk`) fitted three per-segment force-gain multipliers,
  `tendon_damping` and `joint_damping`: multipliers 0.020 / 0.045 / 0.009,
  `tendon_damping` 0.316 N s/m, `joint_damping` 0.0082 N m s/rad.
* **Held-out score of that twin:** 10.850 deg, nrmse 0.967, correlation +0.756.
* **A responsiveness check, not a fit:** all twelve joints move 0.529–0.875 deg
  per psi of commanded differential.

### This session

* **Geometry** (`65d38c3`, `90975c7`): the ProMax Y/hub/bearing routing, the
  Koopman prior masses, and the room viewer's proximal hinge order (section 3).
  Unrefitted, but still carrying the axis defect: 11.746 deg / 1.045 / +0.737.
* **The axis read** (`ac794e7`): hinges read by name, closed-form seats restored.
  9.691 deg / 0.869 / +0.838, no refit (section 4).
* **The flow net re-checked** under the corrected kinematics: 3676.2 Pa against
  3681.9 Pa as trained, so it was kept (section 5).
* **A CMA-ES mechanical fit** (`110e3cc`, `5dcea2a`): 15 parameters, 13 training
  windows, 1681 rollouts. Training loss 2.034 → 1.444 deg; held out 5.639 deg /
  0.507 / +0.952 (sections 6–7).
* **Profiles** of each mass and of the scaling directions (section 8).
* **An as-built refit** (`8cd4c94`; section 9): masses as built, a per-segment
  joint stiffness, 1441 rollouts on the same windows; data loss 1.444 → 1.024 deg;
  held out 2.124 deg / 0.187 / +0.993.
* **A like-for-like responsiveness replay** of the 2026-08-21 single-board
  campaign on the twin (`1179241`; section 12).
* **The deliverable** now builds the twin through `twin_params`, the loader the
  GUI's SIM adapter also uses.

### What was not fitted, and why each matters

* **The flow net during the mechanical fit.** Its 0.534 psi held-out pressure
  error is therefore fitted as a mechanical error.
* **`l0`**, the measured `LL`. It is also the offset of the net's `l` input, so
  moving it would require retraining the net.
* **The blend widths** (7 mm 2000 Pa, TLE 6000 Pa), firmware assumptions that
  also define the closed holds the leak fit keys off.
* **`damp_b1`**, the pressure-scheduled tendon damping (9.9e-4 N s/m per Pa, an
  RS485 fit; 68 N s/m at 10 psi). With the joint armature it is the only term
  that separates mass from force gain (section 8), so the fitted total mass
  rests on it.
* **The joint armature** (0.01 kg m^2 per hinge, an inherited stability choice).
  At the first fit's masses it is 99.7 % of the inertia of joints 10 and 11.
* **The ring radii** `AO` = 28 mm, `AA` = 43.7 mm and `JA` = 47 mm are CAD, and
  they set the moment arm, so an error in them is absorbed into the fitted gain.
  The routing bearing is modelled as a point.
* **The Y-tip radius (85 mm) and height**, estimated from Fig. 1C. They move the
  drawing and the sleeves' share of link inertia, not a moment arm.
* **Masses, `bf` and `joint_frictionloss`** had never been fitted before this
  session; they are fitted now.

## 3. The ProMax geometry

Fig. 1C of the paper shows one segment in section and from outside. Every part
below except the u-joint outer rings is rigid with the segment's centre rod:

* **Two upright Ys**, in perpendicular planes, spread their arms up and out so
  the proximal u-joint sits inside their cone. Their four tips carry the four
  actuators that drive the **distal** u-joint, at 45 / 135 / 225 / 315 deg.
  Each tendon runs from the sleeve's free end, around a bearing on the bottom
  hub (`AO` = 28 mm, `AA` = 43.7 mm above the distal centre), to the distal
  u-joint's outer-ring bracket (`JA` = 47 mm, in the distal joint plane).
* **Two upside-down Ys**, 45 deg round from the upright pair, spread their arms
  down around the distal u-joint. Their tips carry the actuators that drive the
  **proximal** u-joint, at 0 / 90 / 180 / 270 deg, through the top hub
  (`AA` = 43.7 mm below the proximal centre) to the proximal outer ring on the
  parent body.

The previous model ran each proximal tendon from the parent plate to a ring
`AA1 + LL` below the joint (222 mm on segment 1), and drew each muscle as two
half-capsules hanging off the u-joint plates. That drawing is what made the Y
arms look attached to the u-joint in the video. The Koopman ProMax generator
(`UMArm_dynamic_koopman_compliance/runze_trying_MPC`, config `original`) ends its
proximal tendons at z = −0.0437 m, i.e. at the top hub, which independently
supports the bearing routing. The parameter table agrees too: with the tip in the
joint plane, an actuator `LL` long leaves about 43 mm of tendon to its bearing on
all three segments.

| quantity | old (far ring) | new (hub bearing) |
| --- | --- | --- |
| rest moment arm, every muscle | 46.8 mm | **43.104 mm** = `AA JA / sqrt((JA − AO)^2 + AA^2)` |
| moment arm as the muscle shortens to +30 deg | falls | rises to 47.0 mm |
| tendon excursion over the validation poses | 1 | 0.930 |
| moving mass (default model) | 1.833 kg (RS485 numbers) | 2.220 kg (Koopman ProMax prior) |

Each muscle drives one axis only, with the measured sign. The sleeve-to-bearing
span is constant, and `sim_core`, `dataset` and `actuator_model` all use
`ten_length − tendon_length0`, so it cancels. The plate sites still reproduce
`fkine(order="yx")` to 4.4e-16 m.

`hw_tests/media/promax_geometry_{segment,hub,arm_rest,arm_bent}_{new,old}.png`
show the two models from the same cameras. The hub close-up shows each tendon
leaving its sleeve, turning on a hub bearing and landing on the disk rim.

**The room viewer drew the arm with the wrong proximal composition.**
`viz/mjcf_canarm.py` declared the CAN arm's proximal hinges x first, while the
live receiver publishes `q` in the measured y-then-x order. Over the measured
poses of the 90 s validation recording the drawn u-joint centres were up to
19.9 mm off (95th percentile 11.5 mm), and up to 161 mm within ±30 deg. The
display now reuses the twin's drawing and declares y first; `viz/self_check.py`
checks it against `fkine`.

What this section does not show: the Y-tip radius and height are read off a CAD
render, not measured, and `AO`, `AA` and `JA` are CAD numbers. The routing is
now the right shape; its dimensions are still the drawing's.

## 4. The twin was reading its hinges in declaration order

The proximal universal joint composes y-then-x (measured 2026-08-21). MuJoCo
composes same-body hinges in declaration order, so the twin declares the y hinge
first, and `qpos` swaps joints 0/1, 4/5 and 8/9 relative to `q`. `SimArm.q()`
returned `data.qpos` unconverted, and `replay` recorded it as `q`.

That produced the permutation seen that morning: the twin's `j0` tracked the
arm's `j1`, and likewise `j4/j5` and `j8/j9`. Commit `a35f94a` read it as a
seat-table error and rotated every proximal muscle 90 deg until the columns
lined up. Counting on each tree which of the 24 muscles drive their measured
hinge with the measured sign gives **12/24 (0/12 proximal) at `8472ce4` and
`65d38c3`, and 24/24 at `ac794e7`**. The rotation was not cosmetic: the distal
axes sit at ±45 deg, so a parent tilt about the wrong axis loads one distal axis
per segment with gravity of the wrong sign.

Held-out validation, 90 s, open loop, same flow net and outer fit, bearing
routing:

| seats / read | joint RMS | nrmse | TLE RMS | 7 mm RMS | mean corr | seg 1 / 2 / 3 corr |
| --- | --- | --- | --- | --- | --- | --- |
| a35f94a table, raw qpos | 11.746 deg | 1.045 | 9.877 | 12.680 | +0.737 | +0.836 / +0.888 / +0.485 |
| closed form, q by hinge name | **9.691 deg** | **0.869** | **6.204** | 11.435 | **+0.838** | **+0.960 / +0.945 / +0.610** |
| closed form, raw qpos (control) | 14.314 deg | 1.312 | 11.015 | 15.964 | +0.392 | +0.630 / +0.361 / +0.186 |

The control row reproduces the original transpositions exactly (best-matching
real joint for twin joints 0..11: 1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11). This
result can therefore be attributed to the read rather than to the seats. The fix
(`ac794e7`) restores the closed-form `LOWER_SEAT_DEG`, and `SimArm.q()` and
`dataset.TendonKinematics.dlen` now resolve the twelve hinges by name. A test pins
the axis of every muscle, and a second test confirms the a35f94a table fails on
all twelve proximal muscles.

## 5. The flow net survives the new kinematics

The net was trained with the proximal muscles' `l` and `ldot` inputs taken from
the other axis of their u-joint. Re-evaluated over the same 10 824 held-out
windows:

| kinematics | held-out pressure RMS | TLE | 7 mm |
| --- | --- | --- | --- |
| as trained (far ring, raw write) | 3681.9 Pa | 2150.0 Pa | 4245.4 Pa |
| hub bearing, closed-form seats, write by name | **3676.2 Pa** | 2147.2 Pa | 4238.7 Pa |

The 0.15 % change sits well inside the 10 % retraining threshold, so the net was
kept. The small effect can likely be attributed to the net drawing little on `l`
compared with `p` and `target − p`. It does not show that the net would ignore a
larger geometry error.

## 6. The mechanical fit

### What is searched

The first fit searched fifteen numbers in log space (`digital_twin/mech_fit.py`,
`110e3cc`):

| parameter | bounds | start | why these bounds |
| --- | --- | --- | --- |
| link mass x3 | 0.25–1.50 kg | 0.70 / 0.50 / 0.50 (Koopman prior) | floor = the link's 8 x 30 g sleeves; top = 2.1x the heaviest prior link |
| bracket mass, seg 1–2 | 0.05–0.60 kg | 0.20 / 0.20 | two rings + spacer; top = 3x prior |
| bracket mass, seg 3 | 0.01–0.40 kg | 0.10 | one ring only, so allowed small |
| rest gain K x3 | 1e-5–1e-3 N/Pa | outer fit | top = Chou–Hannaford pull of a 25 mm braid |
| shape bf/l0 x3 | 0.50–1.45 | RS485 | see below |
| tendon_damping | 0.01–100 N s/m | 0.316 | top rivals `damp_b1 p` at 15 psi |
| joint_damping | 0.001–3.0 N m s/rad | 0.0082 | brackets the Koopman model's 2.25 |
| joint_frictionloss | 0.001–0.5 N m | 0.025 | below the RS485 arm's superseded 0.70 |

The force law is searched as a rest gain K = coeff (3 l0^2 − bf^2) and a shape
bf/l0 rather than as `coeff` and `bf`, because moving `bf` alone changes gain and
stiffness together and puts a ridge between the two coordinates.

**One bound leaked held-out information.** The first fit's bf/l0 ceiling of 1.45
was sized from the largest muscle contraction in the *validation* recording
(21.7 mm), and the fit ended on that ceiling on all three segments. The refit
(section 9) sizes it from the training windows only: 18.64 / 18.42 / 21.45 mm of
contraction, bounds 1.55 / 1.51 / 1.47, recomputed on every run.

### Why CMA-ES

The optimizer had to follow the valleys the scaling argument predicts (section 8)
on a budget of a few thousand rollouts. CMA-ES, a covariance-matrix-adaptation
evolution strategy, is a natural choice for that. It learns the covariance of a
diagonal valley such as (mass x k, gain x k) and steps along it, whereas
differential evolution's per-coordinate crossover keeps proposing moves across
it. One generation is one batch of 24 candidates across 24 worker processes. The
`cma` package (4.4.4) is used rather than a local implementation, because its
boundary handling is tested.

### The objective and the windows

The objective is open-loop joint deflection RMS in degrees through
`twin_compare.twin_rollout`. It is averaged over the twelve joints, then over the
windows of a family, then over five families with equal weight. Equal family
weight keeps the long families from outvoting the short ringdowns that carry the
dynamics.

| family | windows | s of arm | what it reads |
| --- | --- | --- | --- |
| random_walk | 1 s lead-in + 20 s | 21.0 | all twelve joints loaded together |
| chirp | joints 0 and 6, 0.15–4 Hz | 41.0 | ring frequency and phase |
| ringdown | 6 charge+release pairs | 39.1 | frequency (gain / inertia) and decay (dissipation / inertia) |
| staircase | boards 0x10A, 0x114, 0x115, 4–24 psi | 52.8 | statics against gravity |
| pair_sweep | joint 2 at pair sums 10 and 18 psi | 25.0 | stiffening with co-contraction, i.e. bf |

No window contains a validation row or spans a sync gap over 0.25 s. A candidate
with any invalid window (force clip, exception, non-finite error) scores
1000 deg plus its count, so it ranks below every valid candidate instead of being
diluted into an average.

### Budget and result

Throughput on this 16-core / 32-thread machine is 32.3 s of arm per wall second
at 24 workers, 97 % of what 32 workers yield. The first fit ran the start point
and 70 generations of 24: **1681 rollouts, 300 735 s of arm, 2.67 h**. It stopped
on its generation limit with the step size down from 0.20 to 0.050, and the last
20 generations lowered the loss by 1.4 %.

| family | start (outer fit + prior masses) | fitted |
| --- | --- | --- |
| random_walk | 4.844 deg | 3.370 deg |
| chirp | 1.070 | 0.868 |
| ringdown | 1.553 | 1.142 |
| staircase | 1.237 | 0.891 |
| pair_sweep | 1.466 | 0.948 |
| **objective** | **2.034** | **1.444** |

## 7. The first fit's parameters

| parameter | start | fitted | note |
| --- | --- | --- | --- |
| link mass | 0.70 / 0.50 / 0.50 kg | 0.400 / 0.712 / 1.131 kg | |
| bracket mass | 0.20 / 0.20 / 0.10 kg | 0.118 / 0.592 / 0.025 kg | segment 2 at its bound |
| moving mass | 2.220 kg | 2.998 kg | sleeve floor 0.72 kg |
| rest gain | 1.116 / 0.921 / 0.217 N/psi | 2.221 / 2.057 / 0.845 N/psi | |
| bf/l0 | 1.216 / 1.364 / 1.269 | 1.450 / 1.450 / 1.450 | all at the (validation-sized) bound |
| tendon_damping | 0.316 N s/m | 45.4 N s/m | |
| joint_damping | 0.0082 N m s/rad | 0.0014 N m s/rad | |
| joint_frictionloss | 0.025 N m | 0.079 N m | |

These masses are not plausible as parts. Link 1 at 0.400 kg leaves 0.16 kg for a
rod, two hubs and four Ys beyond its 0.24 kg of sleeves, while link 3 at 1.131 kg
is 2.26x its prior. Bracket 3 at 0.025 kg is below the 33 g estimated for its
ring alone, and the segment-2 connector sits on its 0.60 kg bound although it is
built identically to segment 1's. Section 8 explains why the recordings allow
this, and section 9 removes the freedom that produced it.

## 8. What the recordings pin

### Stated before the fit

Scaling every mass and every muscle gain by one factor k scales gravity torque,
inertia and muscle torque by k. The equation of motion is unchanged except that
every dissipation term and the joint armature become 1/k as large relative to it.
Hence:

* the statics (staircases, pair sweep) pin gain / mass and nothing more;
* the dynamics separate the two only through terms that do not scale with k,
  namely `damp_b1` and the joint armature, neither measured on this arm;
* scaling mass, gain, all fitted dissipation **and** `damp_b1` by k leaves only
  the armature to break the symmetry.

### Measured after the first fit

A line pins its parameter on a side when the outer factor raises the loss by more
than 1 % of the optimum (0.014 deg).

| line | x0.5 | x0.75 | x1 | x1.5 | x2 | verdict | 1 % band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| link mass 1 | 1.441 | 1.442 | 1.444 | 1.447 | 1.450 | not pinned | x0.50–2.00 |
| link mass 2 | 1.454 | 1.448 | 1.444 | 1.441 | 1.445 | not pinned | x0.50–2.00 |
| link mass 3 | 1.462 | 1.449 | 1.444 | 1.453 | 1.471 | **pinned** | x0.57–1.63 |
| bracket mass 1 | 1.443 | 1.444 | 1.444 | 1.444 | 1.445 | not pinned | x0.50–2.00 |
| bracket mass 2 | 1.456 | 1.449 | 1.444 | 1.437 | 1.436 | not pinned; still falling at x2 | x0.50–2.00 |
| bracket mass 3 | 1.443 | 1.443 | 1.444 | 1.444 | 1.445 | not pinned | x0.50–2.00 |
| all masses and gains | 1.636 | 1.484 | 1.444 | 1.483 | 1.548 | **pinned** | x0.90–1.16 |
| … plus all dissipation and damp_b1 | 1.467 | 1.448 | 1.444 | 1.448 | 1.457 | pinned below only | x0.61–2.00 |
| joint armature (not fitted) | 1.457 | – | 1.444 | – | 1.467 | pinned above only | x0.50–1.53 |

* **The scaling argument holds numerically.** The family losses along "masses,
  gains and all dissipation x k" match "armature x 1/k" to within 0.004 deg in
  every family (at x2: random walk 3.309 against 3.307; at x0.5: 3.509 against
  3.513).
* **The recordings pin a combination, not the weights.** The total of masses and
  gains is pinned to about −10 / +16 %, but only against `damp_b1` and the
  armature. Individually only link 3 is pinned. The data constrains the gravity
  torque each joint sees, not how that torque is split among the bodies.
* **The fit leans on its bounds where the model is short of restoring torque.**
  bf/l0 and the segment-2 connector sit on their bounds, and link 3 came out
  2.26x its prior. All three can likely be attributed to one missing term:
  without passive joint stiffness (tubing, wiring), the only ways the model can
  add restoring torque are a stiffer muscle shape and more hanging mass.

Joints 10 and 11 carry only the segment-3 ring, so gravity offers them almost no
restoring torque and the armature, not any mass, sets their inertia. They remain
the worst held-out joints (section 10).

## 9. The refit, with the masses modelled as the hardware is built

The adversarial review rejected the first fit's per-body masses as a weight
answer, and section 7 agrees. Commit `8cd4c94` removes the freedom that produced
them rather than tightening bounds around it:

* **each link** = eight sleeves fixed at the measured 30 g + one link-structure
  mass shared by all three segments (rod, hubs and Ys are the same parts);
* **each inter-segment connector** = ring + spacer + ring, identical on segments
  1 and 2; **segment 3's end body** = one ring;
* **a per-segment passive joint stiffness** (`joint_stiffness`, default 0), the
  term section 8 points to;
* **a prior pull** toward the Koopman masses (link structure 0.327 kg, ring
  0.05 kg, spacer 0.10 kg): 0.0144 deg of objective per factor of two, sized to
  the profiles' 1 % pin threshold, and reported separately from the data loss;
* **bf/l0 bounds from the training windows only** (section 6);
* **a progress file**: a running fit writes `canarm_mech.running.json`, and only
  a complete one replaces `canarm_mech.json`, so an aborted run can no longer
  change the twin the GUI and the deliverable load.

The first attempt at this refit reached generation 9 of 70 before its host
process exited with the fix agent. Its best point then was objective 1.378 deg
(data 1.310 + prior 0.068), against 1.444 for the first fit on a different
objective: links 0.812 / 0.805 / 0.803 kg, connectors 0.330 kg, end ring
0.020 kg, segment-3 stiffness 2.64 N m/rad. It was restarted from that point.

### The result

The restarted run evaluated its start point and 60 generations of 24: **1441
rollouts, 257 798 s of arm, 2.39 h**, stopping on its generation limit with the
step size at 0.055. It rolled the same 13 windows with the same loss as the first
fit, so the data losses compare directly:

| family | first fit | as-built refit |
| --- | --- | --- |
| random_walk | 3.370 deg | 2.125 deg |
| chirp | 0.868 | 0.716 |
| ringdown | 1.142 | 0.794 |
| staircase | 0.891 | 0.637 |
| pair_sweep | 0.948 | 0.847 |
| **data loss** | **1.444** | **1.024** (−29 %) |
| prior penalty | – | 0.0013 |

| parameter | refit | Koopman prior or bound | note |
| --- | --- | --- | --- |
| link structure (shared) | 0.321 kg | prior 0.327 kg | **pinned by the data**, x0.56–1.74 |
| connector ring | 0.061 kg | prior 0.05 kg | not pinned: the prior sets it |
| connector spacer | 0.100 kg | prior 0.10 kg | not pinned: the prior sets it |
| → link body incl. 8 x 30 g sleeves | **0.564 / 0.560 / 0.559 kg** | Koopman 0.70 / 0.50 / 0.50 | |
| → connector; segment-3 end ring | **0.223 / 0.223 / 0.061 kg** | Koopman 0.20 / 0.20 | |
| moving mass | **2.210 kg** | 2.220 kg | first fit 2.998 kg |
| rest gain | 5.54 / 5.09 / 3.80 N/psi | top 6.89 N/psi | first fit 2.22 / 2.06 / 0.84 |
| bf/l0 | 1.549 / 1.509 / 1.470 | bounds 1.55 / 1.51 / 1.47 | **at the slack-limit bound on all three** |
| joint stiffness | 0.025 / 0.017 / **3.95** N m/rad | 0.001–10 | segment 3 **pinned**, x0.87–1.55 |
| tendon_damping | 94.7 N s/m | bound 100 | 0.95 of its bound |
| joint_damping | 0.0010 N m s/rad | bound 0.001 | at its floor |
| joint_frictionloss | 0.108 N m | | |

Profiles around the optimum, data loss only (pin threshold 1 % = 0.010 deg):

| line | x0.5 | x0.75 | x1 | x1.5 | x2 | verdict | 1 % band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| link structure | 1.037 | 1.027 | 1.024 | 1.028 | 1.040 | **pinned** | x0.56–1.74 |
| connector ring | 1.025 | 1.024 | 1.024 | 1.025 | 1.028 | not pinned | x0.50–2.00 |
| connector spacer | 1.026 | 1.025 | 1.024 | 1.023 | 1.022 | not pinned | x0.50–2.00 |
| joint stiffness 1 | 1.024 | 1.024 | 1.024 | 1.024 | 1.023 | not pinned | x0.50–2.00 |
| joint stiffness 2 | 1.025 | 1.024 | 1.024 | 1.023 | 1.022 | not pinned | x0.50–2.00 |
| joint stiffness 3 | 1.098 | 1.046 | 1.024 | 1.030 | 1.067 | **pinned** | x0.87–1.55 |
| masses, gains and stiffness | 1.111 | 1.030 | 1.024 | 1.061 | 1.106 | **pinned** | x0.73–1.12 |
| … plus all dissipation | 1.053 | 1.031 | 1.024 | 1.024 | 1.028 | pinned below only | x0.70–2.00 |
| joint armature (not fitted) | 1.022 | – | 1.024 | – | 1.046 | pinned above only | x0.50–1.38 |

Four readings follow.

* **The weight answer.** Modelled as built, the arm moves 2.21 kg: link bodies of
  0.56 kg each (0.32 kg of structure plus 0.24 kg of sleeves) and connectors of
  0.22 kg. The data pins the shared link structure from both sides and nothing
  finer. The connector ring and spacer are not pinned and sit at the Koopman
  prior, because the prior is the only term that cares about them; those two
  numbers are therefore the prior's, not the recordings'.
* **The missing stiffness was real, and it sits in segment 3.** A passive joint
  stiffness of 3.95 N m/rad on segment 3 is pinned from both sides, while
  segments 1 and 2 take almost none. Joints 10 and 11 had about 0.005 N m/rad of
  gravity restoring torque at the first fit, so a term of this size dominates
  them. It is consistent with the elasticity of tubing and wiring through the
  distal segment, but this result does not separate that from an error in the
  distal force law.
* **With that term available, the implausible masses disappear.** Link 3 no
  longer needs 1.13 kg and the segment-2 connector no longer climbs to its bound;
  all three links fit within 1 % of each other, as identical hardware should.
* **Three quantities still sit on bounds.** bf/l0 is at the slack limit on all
  three segments, `tendon_damping` is at 0.95 of its ceiling and `joint_damping`
  at its floor. The first says the muscles want to stiffen with contraction
  faster than a McKibben of this length can without going slack inside the
  training contraction. The other two say the arm dissipates more in its muscles
  at low pressure than the RS485 schedule provides, and less in its bearings.
  None was widened: the shape bound is the slack limit itself, and the damping
  split cannot be settled without the direct measurement in section 15.

The absolute scale still rests on `damp_b1` and the armature: masses, gains and
stiffness together are pinned to −27 / +12 %, but with all dissipation scaled
too only the lower side survives.

`digital_twin/checkpoints/canarm_mech.json` now holds this refit; the first fit's
parameters remain in sections 6–8 and in git history (`5dcea2a`).

## 10. The held-out score

Held-out validation: 90 s, 13 501 cycles, open loop. The twin sees only the
recorded pressure targets.

| | twin | joint RMS | nrmse | TLE RMS / nrmse | 7 mm RMS / nrmse | mean corr | seg 1 / 2 / 3 corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (a) | old geometry + old outer fit (start of session) | 10.850 deg | 0.967 | 9.121 / 0.762 | 11.715 / 1.069 | +0.756 | +0.850 / +0.899 / +0.520 |
| (b) | new geometry + old outer fit (a35f94a seats) | 11.746 deg | 1.045 | 9.877 / 0.824 | 12.680 / 1.156 | +0.737 | +0.836 / +0.888 / +0.485 |
| (b') | new geometry + axis fix + old outer fit | 9.691 deg | 0.869 | 6.204 / 0.532 | 11.435 / 1.037 | +0.838 | +0.960 / +0.945 / +0.610 |
| (c) | new geometry + axis fix + first mechanical fit (free per-body masses) | 5.639 deg | 0.507 | 4.852 / 0.427 | 6.033 / 0.547 | +0.952 | +0.969 / +0.976 / +0.911 |
| (d) | new geometry + axis fix + **as-built refit** (`canarm_mech.json`) | **2.124 deg** | **0.187** | **1.668 / 0.142** | **2.352 / 0.209** | **+0.993** | **+0.997 / +0.989 / +0.992** |

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RMS (b') deg | 6.56 | 4.80 | 4.93 | 8.52 | 4.21 | 9.71 | 5.54 | 10.54 | 11.68 | 9.61 | 22.62 | 17.57 |
| RMS (c) deg | 6.32 | 4.59 | 2.56 | 5.93 | 3.46 | 4.71 | 2.88 | 6.71 | 7.06 | 4.70 | 10.31 | 8.44 |
| RMS (d) deg | 2.37 | 0.86 | 1.49 | 1.95 | 3.36 | 2.92 | 1.69 | 2.49 | 1.79 | 2.03 | 3.33 | 1.22 |
| corr (b') | 0.976 | 0.931 | 0.990 | 0.944 | 0.941 | 0.949 | 0.951 | 0.940 | 0.791 | 0.715 | 0.472 | 0.462 |
| corr (c) | 0.977 | 0.952 | 0.994 | 0.953 | 0.976 | 0.981 | 0.977 | 0.970 | 0.942 | 0.934 | 0.886 | 0.881 |
| corr (d) | 0.997 | 0.996 | 0.996 | 0.998 | 0.994 | 0.976 | 0.994 | 0.994 | 0.988 | 0.991 | 0.996 | 0.994 |

The first mechanical fit took the held-out error from 9.691 to 5.639 deg
(−42 %), and the as-built refit took it on to **2.124 deg (−62 %)**; against the
twin the session began with (10.850 deg), the reduction is 80 %. Every joint
improved at both steps. Joints 10 and 11 alone account for a third of the last
step (1.18 of 3.51 deg, from 10.31 and 8.44 to 3.33 and 1.22 deg), which can
likely be attributed to the segment-3 joint stiffness of section 9: without it
those two joints had almost no restoring torque. The other ten joints account for
the remaining two thirds.

The held-out improvement (−62 %) exceeds the training one on the same windows
(−29 %). While that pattern argues against memorised windows, it does not by
itself exclude leakage, so the discipline is stated: the 13 windows are identical
between the two fits (checked line by line), none contains a validation row,
`canarm_mech.json` records `validation` as never rolled, and the refit's bf/l0
bounds were sized from training contraction only. The one input shared with the
held-out sequence is the Koopman mass prior, which predates both recordings.

Two boards stand out in the held-out pressure residual: `0x110` at 35.7 kPa RMS
(nrmse 0.86), the board that leaks from its supply side, and `0x104` at 7.9 kPa,
the board with a +1.1 psi rest offset. Both are hardware faults the twin does not
model. No candidate in either fit and no held-out rollout touched the 4000 N
force clip.

## 11. The SIM adapter

The operator's requirement was that the control window start a simulated arm
through the same interface, so that a controller cannot tell which it drives.

* **One construction differs, nothing else.** `canarm_control_gui.py` lists an
  always-present `SIM - digital twin (no hardware)` adapter (`--sim` preselects
  it; a Refresh never picks it on its own). `make_backend` returns
  `digital_twin.sim_master.SimMaster` for that label and tlelib's `Backend`
  otherwise. Scan, select, the 150 Hz cycle, targets, enables, STOP ALL and
  disconnect are the same method bodies for both, and no file under `TLE_PCB/`
  changed.
* **The twin is the fitted one.** `digital_twin/twin_params.load_twin_kwargs()`
  is the one spelling of the fitted twin for `SimArm`, `SimMaster`,
  `replay.rollout` and the deliverable.
* **Joints come through the real receiver.** `digital_twin/sim_mocap.SimMocap`
  feeds the twin's `q` through the shipped `CanArmMocap` listeners (round trip
  3.9e-16 rad) and backs a `twin` mocap source; the room viewer receives it
  through `viz_layout`'s shared array.

**The timing gave the twin away, and now does not.** The plant costs about 3.3 ms
of every 6.67 ms period, and it first ran on the GUI's interpreter lock.
`SimMaster(physics="process")` (`digital_twin/sim_process.py`, `7372387`) runs it
in a spawned child against the same `perf_counter`, with replies still stamped at
the per-id CAN latency.

| setup (24 boards selected, `0x101` at 12 psi) | cycle rate | replies heard |
| --- | --- | --- |
| real arm, same window (2026-08-20) | 145.43 Hz | 98.81 % |
| twin in the GUI process | 97.6 Hz | about 52 % |
| twin in the GUI process, after the reply-delivery fix | 64.7 Hz | 100.00 % |
| **twin in its own process** | **143.6 Hz** | **99.67 %** |

`hw_tests/gui_sim_test.py` drives the real window with `serial.Serial`, both
`CanLink.open` paths and the NatNet start poisoned, and fails if rate or replies
sit more than 5 % / 2 points from the real window: 33/33, and the in-process
twin fails it (26/28), which shows the check has teeth. `0x101` reaches 12 psi
in about 0.4 s and joint 2 swings +17.84 deg through the twin receiver. Re-run on
the as-built refit: 33/33, 142.1 Hz, 99.36 % of replies heard, joint 2
+17.65 deg.

What this does not show: a machine loaded past one free core leaves the child
behind real time, and it then misses replies as an overloaded bus would. The
process twin is not bit-identical to an in-process rollout. The twin's flow model
also drifts idle boards at a closed hold (TLE +0.10 to +0.15 psi/s, 7 mm about
−0.23 psi/s), which a careful observer of idle pressures could notice.

**A safety gap, found and left identical on both adapters.** The window does not
enforce `digital_twin/CONTRACT.md` section 8: the bars and the slider reach
40 psi and there is no antagonistic pair-sum check. A SIM-only guard would make
the two adapters behave differently, so none was added; the gap belongs to the
real window and should be closed there.

## 12. Responsiveness, like for like

`hw_tests/twin_drive_campaign.py` replays the 2026-08-21 campaign on the twin:
each board driven alone at 12 psi against a resting arm, compared joint for joint
with `hw_tests/results/axis_analysis_2026-08-21.json`. 

| | first fit | as-built refit |
| --- | --- | --- |
| boards moving the measured joint with the measured sign | 24/24 | 24/24 |
| median ratio twin / arm (range) | 1.00 (0.48–1.76) | 1.01 (0.63–2.12) |
| segment medians | 0.73 / 0.98 / 1.23 | 1.04 / 1.11 / 0.95 |
| mean absolute error | 3.45 deg | **2.29 deg** |
| `0x101` on joint 2 (arm +15.73 deg) | +17.75 deg (1.13x) | +18.95 deg (1.20x) |

The refit removes the first fit's segment-level bias, in which segment 1 was too
stiff and segment 3 too soft. What remains is per board: the twin gives two
antagonists equal and opposite swings (±18.95 deg on joint 2, ±19.02 deg on
joint 4), while the arm does not (+15.73 against −20.36 deg on joint 2). The two
outliers are `0x109` (arm +8.96, twin +19.01 deg, 2.12x) and `0x118` (arm
−21.14, twin −13.34 deg, 0.63x). An asymmetry of this kind can likely be
attributed to per-muscle differences — braid, bladder, tendon pre-tension — that
a per-segment force law cannot represent; this replay does not distinguish those
causes.

This replay replaces an earlier gravity-stiffness comparison that set a
single-muscle static figure beside the arm's co-contraction metric; the two
measured different things.

## 13. The test-order failure, attributed

Three `test_sim_master` tests failed in the full suite and passed alone. It was
neither CPU load nor a regression: after `test_sim_core` ran first, the simulated
link's delivery thread ran physics while replies waited, and a reply handed to a
second thread arrived 3–4 ms late even with garbage collection frozen. Replies
are now delivered by the thread that holds them, at their due time (`6700e29`):
0 of 300 cycles incomplete in either order, against 130–148 after `test_sim_core`
before, and `test_sim_core` then `test_sim_master` passes in 8 of 8 runs.

## 14. The video

`deliverable/twin_vs_real_validation.mp4` (1351 frames, 41.5 MB, gitignored) and
`deliverable/frame_sample.jpg` were re-rendered from the as-built refit's rollout.
The frame at 35.00 s shows the ProMax drawing in both panels at nearly the same
pose, with a mean joint error of 1.05 deg and a worst of 2.82 deg at that instant,
against 4.41 and 13.43 deg on the first fit and 7.81 and 22.62 deg in the video
the operator first saw. In the trace strip, joints 0, 4 and 8 of the twin overlay
the arm's through all 90 s. The caption glyphs no longer double: OpenCV 5.0
advances thick strokes further than thin ones, so the halo is now drawn at
one-pixel offsets (`16339ac`). The camera panel is absent
because the bench camera did not record that session.

## 15. What this does not show, and what to measure next

* **A weight.** The as-built refit pins the link structure only to x0.56–1.74,
  leaves the connector masses at the prior, and pins the overall scale only
  against `damp_b1` and the armature, both carried over. The 2.21 kg is the
  model's weight, not a scale reading.
* **Convergence.** Both fits stopped at their generation limits (step size 0.050
  and 0.055), and three of the refit's parameters sit on bounds.
* **A physical cause for segment 3's stiffness.** The fit needs the term; it does
  not say whether tubing, wiring or a distal force-law error supplies it.
* **Ring statistics.** `ring_analysis` found no ring episodes in either held-out
  trace at its RS485 band and threshold, so ring frequency and damping ratio were
  not compared.
* **More than one held-out sequence.** The score comes from one 90 s sequence of
  one family, open loop, with no closed-loop behaviour and no contact.

What to measure next, in order:

1. **Weigh one link body with its eight sleeves, and one connector** (ring,
   spacer, ring). Two scale readings pin the absolute scale the fit borrows from
   `damp_b1`.
2. **Measure passive joint stiffness.** Vent all 24 muscles, hang 100–200 g at
   the tip and at each plate, and record the mocap deflection.
3. **Measure dissipation directly.** A vented tip push-and-release gives the
   p = 0 damping; a pressurised release at two co-contraction levels gives
   `damp_b1`.
4. **Measure `AO`, `AA`, `JA` and the Y-tip position on the metal**, since each
   sets a moment arm or an inertia the fit absorbs.
5. **Refit** with those held fixed:
   `python -m digital_twin.mech_fit --start digital_twin/checkpoints/canarm_mech.json`.

---

## Files

| what | where |
| --- | --- |
| the geometry | `digital_twin/mjcf_generator.py`, `viz/mjcf_canarm.py`, renders `hw_tests/media/promax_geometry_*.png` |
| the fitted twin | `digital_twin/checkpoints/canarm_flow.npz` + `canarm_mech.json`, loaded by `digital_twin/twin_params.py` |
| the optimizer | `digital_twin/mech_fit.py`, logs `data/fit/mech_fit_*.log` and every evaluation in `data/fit/mech_fit_*_evals.jsonl` (gitignored) |
| the SIM adapter | `canarm_control_gui.py`, `digital_twin/sim_master.py`, `digital_twin/sim_process.py`, `digital_twin/sim_mocap.py`; acceptance `hw_tests/gui_sim_test.py` |
| the responsiveness replay | `hw_tests/twin_drive_campaign.py` |
| the held-out numbers | `deliverable/deliverable_validation.json`, frame `deliverable/frame_sample.jpg` |
| the binding interface | `digital_twin/CONTRACT.md` sections 3–5, 7a (`twin_params`), 7b (`mech_fit`) |
