# CAN arm: actuator/axis map, marker-frame inference, and fkine against mocap

**2026-08-21, live hardware.** Motive streaming, supply at ~20 psi, arm hung in
its monitored bay, all 24 boards on the bus.

Three things were unknown when the session started and are known now: which of
the 24 boards drives which joint, where the marker brackets sit relative to the
mechanism, and whether the forward kinematics reproduces what the cameras see.
The third one turned up a defect in the kinematics itself, which is the part of
this report worth reading twice.

---

## 0. What was run

| step | command | outcome |
|---|---|---|
| mocap census | `hw_tests/mocap_census.py --seconds 12` | 13 rigid bodies at 115 Hz: **500–505** (RS485 arm), **1008** (Kinova), **2000–2005** (this arm), four labeled markers each |
| probe | `UMArm_MOCAP/mocap_probe.py --rb-id-base 2000 --n-bodies 6 --x-mode diagonal45` | all six plates, four markers, worst marker sd 0.07 mm; two gates failed and both were *right to* — see §2 |
| drive campaign | `hw_tests/canarm_drive_campaign.py --psi 12 --poses 18` | 68 steps: 1 rest, 24 single-actuator drives with bracketing baselines, 18 multi-joint poses |
| reduction | `hw_tests/canarm_axis_analysis.py --holdout poses` | the map, the azimuths, the lengths, the FK residual |
| GUI, live | `hw_tests/canarm_gui_kinematics.py` | **11/11 checks**, Lock plates → marker receiver → 1.9 mm worst residual on screen |

The 2000–2005 id block is now **verified**, closing the three-way dispute the
2026-08-20 notes left open; the 500-block constant in `mocap_constants` was
right about the *other* arm rather than wrong.

---

## 1. Two hardware faults, found by the data

**`0x110` leaks from its supply side.** Left disabled it drifted from 0.1 to
7.0 psi across the first sweep, which put a slow ramp underneath every measured
delta. The fix is in the campaign script: unused boards are held **enabled at a
small idle setpoint** rather than disabled, so the loop vents the leak
continuously. The idle is 0.5 psi rather than 0 so the exhaust valve is not
driven for the whole campaign.

**Two boards read a non-zero pressure at rest**: `0x110` at +5.1 psi and
`0x104` at +1.1 psi. Both are recorded in
`UMArm_KINEMATICS.canarm_actuators.KNOWN_BOARD_FAULTS`. Nothing in this
campaign depends on absolute pressure, but a controller commanding `0x110` will
be short by about 5 psi.

---

## 2. The frame inference, and the thing the brief got wrong

Each plate carries four markers on radial arms about the u-joint centre. The
plates measure:

| plate | diagonals (mm) | arms (mm) | midpoint sep (mm) | crossing off-90 | out-of-plane rms (mm) |
|---|---|---|---|---|---|
| 0 | 154.7 / 152.0 | 72.1–79.9 | 4.58 | +0.44° | 0.03 |
| 1 | 149.1 / 145.3 | 70.5–78.6 | 4.56 | +0.04° | 0.16 |
| 2 | 145.5 / 153.3 | 70.9–78.5 | 2.68 | +0.05° | 0.24 |
| 3 | 150.8 / 141.2 | 66.7–78.9 | 5.18 | +0.17° | 0.04 |
| 4 | 141.7 / 146.9 | 66.9–74.9 | 4.21 | +0.03° | 0.00 |
| 5 | 149.8 / 142.7 | 68.0–79.1 | 5.37 | +0.09° | 0.23 |

The arms are unequal, so the diagonal *midpoints* sit 2.7–5.4 mm apart and the
centre has to be the intersection of the diagonal **lines**, which is exact for
radial arms of any length. The diagonals cross within 0.44° of perpendicular
and the markers are coplanar to a quarter of a millimetre, so the plane fit and
the origin are both well conditioned.

**The brief says the marker arms poke out between the actuator attachment
points, 45° from the revolute axes. On this arm they do not.** Driving each
antagonistic pair alone and measuring the axis it actually rotates about puts
the revolute axes **along the diagonals**:

| plate | the four drive axes, azimuth in the marker frame | mutual orthogonality |
|---|---|---|
| 0 | `0x108` +44.5°, `0x104` −134.6°, `0x106` −44.7°, `0x102` +135.6° | 0.68° |
| 2 | `0x10A` +44.8°, `0x10B` −45.0°, `0x10C` −136.1°, `0x109` +136.4° | 1.36° |
| 4 | `0x114` +45.7°, `0x111` +134.7°, `0x113` −44.4°, `0x112` −134.2° | 0.75° |

Four axes 90° apart to within a degree, all sitting at 45° from where the
"markers between the axes" convention would put them. That is a 45° frame
error — exactly the size that produces a smooth, repeatable, entirely wrong
`q`. `UMArm_MOCAP.canarm_frames.PLATE_AZIMUTH_DEG` carries the measured answer.

Two independent checks say the construction is sound, and neither is fitted:

* **Co-rigidity.** Plates 1/2 and 3/4 are bolted to one connector, so their
  inferred *body* frames must be identical. Measured over 68 poses: azimuth
  **0.00° ± 0.038°** and **0.00° ± 0.033°**, worst out-of-plane tilt 0.23° and
  0.40°. This also confirms the ±45° bracket family the mechanism was designed
  with, read out of the data rather than assumed into it.
* **Template registration.** Every frame is a Kabsch fit of the rest-time
  marker template; worst residual over all 68 steps and all 6 plates was
  **0.64 mm** against a 3 mm gate, and the live GUI session read 0.028 mm.
  Three of the four markers suffice, so one dropout costs nothing.

The probe's two failing gates were both correct: the span gate fired because the
nominal chain in it is the RS485 arm's (§4), and the alignment gate fired
because Motive's streamed orientation really is ~43° from the marker-derived
frame — which is the finding, not a fault.

---

## 3. The actuator/axis map

Every board pressurised alone at 12 psi against an otherwise idle arm; the joint
is the one whose marker-derived angle moved most, the sign is that motion's.

| board | joint | Δq (deg) | margin over runner-up |
|---|---|---|---|
| `0x101` | j2 `s1.u2.t3` | +15.73 | 2.9 |
| `0x102` | j1 `s1.u1.t2` | +11.67 | 2.5 |
| `0x103` | j3 `s1.u2.t4` | +20.20 | 2.8 |
| `0x104` | j0 `s1.u1.t1` | −12.91 | 2.2 |
| `0x105` | j2 `s1.u2.t3` | −20.36 | 2.4 |
| `0x106` | j1 `s1.u1.t2` | −11.99 | 2.3 |
| `0x107` | j3 `s1.u2.t4` | −15.86 | 3.0 |
| `0x108` | j0 `s1.u1.t1` | +14.20 | 2.0 |
| `0x109` | j5 `s2.u3.t2` | +8.90 | 3.9 |
| `0x10A` | j4 `s2.u3.t1` | +18.62 | 3.1 |
| `0x10B` | j5 `s2.u3.t2` | −17.06 | 3.2 |
| `0x10C` | j4 `s2.u3.t1` | −13.78 | 3.3 |
| `0x10D` | j7 `s2.u4.t4` | +18.63 | 8.0 |
| `0x10E` | j6 `s2.u4.t3` | −16.21 | 8.9 |
| `0x10F` | j7 `s2.u4.t4` | −20.70 | 7.6 |
| `0x110` | j6 `s2.u4.t3` | +17.88 | 8.8 |
| `0x111` | j9 `s3.u5.t2` | +12.62 | 8.5 |
| `0x112` | j8 `s3.u5.t1` | −9.46 | 11.6 |
| `0x113` | j9 `s3.u5.t2` | −12.61 | 9.8 |
| `0x114` | j8 `s3.u5.t1` | +12.88 | 10.2 |
| `0x115` | j11 `s3.u6.t4` | +14.30 | 105.6 |
| `0x116` | j10 `s3.u6.t3` | +15.72 | 44.3 |
| `0x117` | j11 `s3.u6.t4` | −15.83 | 135.9 |
| `0x118` | j10 `s3.u6.t3` | −21.14 | 50.2 |

The margins fall from ~100 at the tip to ~2 at the base, and that is physical
rather than noise: the arm is compliant and underactuated, so bending segment 1
changes where segments 2 and 3 hang, and their equilibrium moves with it. The
dominant joint is never in doubt.

**Result, joint → (positive, negative):**

| joint | measured | legacy table | verdict |
|---|---|---|---|
| j0 `s1.u1.t1` | +`0x108` / −`0x104` | `0x102`/`0x106` | rotated |
| j1 `s1.u1.t2` | +`0x102` / −`0x106` | `0x104`/`0x108` | rotated |
| j2 `s1.u2.t3` | +`0x101` / −`0x105` | `0x103`/`0x107` | rotated |
| j3 `s1.u2.t4` | +`0x103` / −`0x107` | `0x105`/`0x101` | rotated |
| j4–j11 | — | — | **all eight match exactly** |

The lower sixteen boards reproduce the research tree's table *including sign*.
That is also what fixes the sign convention: a robot frame turned 180° about z
would flip all sixteen at once, so the "body x nearest the volume's +x" gauge is
the same one the legacy stack used.

Segment 1 is the same four antagonistic **pairs**, each driving the other axis
of its universal joint, with the signs a **+90° rotation** implies — a pair
formerly on +x now drives +y, one formerly on +y now drives −x. That is the
signature of the replaced top regulator platform having been mounted a quarter
turn round, and it is why the legacy table must not be used on this arm.

---

## 4. Lengths

Five consecutive u-joint centre distances, read off the marker-inferred frames.
These are rigid — plate centre *is* joint centre here — so they should not
depend on pose, and did not:

| gap | measured (mm) | sd (mm) | research tree (mm) | difference |
|---|---|---|---|---|
| u1–u2 | 265.36 | 0.47 | 265.05 | +0.31 |
| u2–u3 | 72.89 | 0.03 | 73.26 | −0.37 |
| u3–u4 | 234.37 | 0.10 | 234.14 | +0.23 |
| u4–u5 | 72.99 | 0.05 | 72.49 | +0.50 |
| u5–u6 | 229.92 | 0.12 | 231.66 | −1.74 |

**The research tree's CAN table was right.** The placeholder this workspace
carried — the RS485 arm's table wearing the CAN arm's name — was wrong by 46 to
182 mm at the u-joint centres. `UMArm_KINEMATICS.canarm_params.MEASURED` is now
`True`.

---

## 5. The defect in the forward kinematics

With the azimuths and the lengths right, the fit still left a residual, and it
was not noise: it tracked a quantity the model says cannot exist.

`q_from_frames` reads a segment's distal joint as two angles out of a
three-degree-of-freedom relative rotation, and silently drops the third — a
rotation about the link axis, call it γ. On the single-actuator drives γ was
nearly constant; on the multi-joint poses it spread to ±2.7° and the u-joint
centre error tracked it at a correlation of **+0.85**.

Regressing γ on the joint angles settles what it is:

| segment | γ ≈ c · t1·t2 | residual sd | raw sd |
|---|---|---|---|
| 1 | c = −0.921 | 0.32° | 1.04° |
| 2 | c = −0.949 | 0.12° | 1.43° |
| 3 | c = −0.911 | 0.12° | 1.00° |

γ is −t1·t2 to within a tenth of a degree. That is not compliance and not
measurement error; it is the exact non-commutativity signature of **the two
proximal twists being composed in the wrong order**. In a product of
exponentials the first factor's axis is the one fixed in the proximal body, so
`exp(ξ1 t1) exp(ξ2 t2)` asserts that the proximal universal joint's x hinge is
bolted to the upper bracket. On this arm the y hinge is.

Testing all four combinations of proximal and distal order:

| proximal | distal | γ spread (sd, deg) | γ peak-to-peak (deg) |
|---|---|---|---|
| xy | xy (legacy) | 1.04 / 1.43 / 1.00 | 7.9 / 9.4 / 8.2 |
| xy | yx | 1.57 / 2.81 / 2.15 | 10.2 / 19.2 / 13.9 |
| **yx** | **xy** | **0.31 / 0.12 / 0.13** | **2.0 / 0.9 / 0.6** |
| yx | yx | 1.85 / 1.98 / 1.65 | 10.4 / 10.8 / 9.0 |

The distal pair is *not* swapped — its ξ3 really is the link-fixed axis. Only
the proximal pair is. `UMArm_KINEMATICS.fkine` now takes an `order`; the
default is the legacy `"xy"`, **bit-identical** to the Mathematica oracle and to
every number the module produced before, and the CAN arm uses `"yx"`.

---

## 6. fkine against mocap

Fitted on the 50 single-actuator and baseline steps, held out on the 18
multi-joint poses. Errors are the distance from each measured u-joint centre to
the one fkine predicts from that frame's own `q`; plate 0 is identically zero
because the chain is anchored there.

| configuration | fit rms | fit worst | **held-out rms** | held-out worst |
|---|---|---|---|---|
| RS485 azimuths + RS485 lengths (what the repo shipped) | 109.1 mm | 182.9 mm | 105.4 mm | 179.1 mm |
| RS485 azimuths + research-tree lengths | 1.29 mm | 4.52 mm | 4.04 mm | 15.8 mm |
| measured azimuths + measured lengths | 1.13 mm | 4.79 mm | 2.32 mm | 7.82 mm |
| refined azimuths + lengths, proximal order `xy` | 0.74 mm | 3.14 mm | 4.64 mm | 20.5 mm |
| **refined azimuths + lengths, proximal order `yx`** | **0.74 mm** | **3.31 mm** | **1.99 mm** | **7.01 mm** |

Per plate, on the held-out poses: 1.01 / 1.04 / 1.90 / 2.26 / 3.60 mm — the
error accumulates down the chain, as an unbranched serial model must.

**Reversing the split** — fit on the 18 poses, hold out the 24 drives — gives
0.86 mm rms / 2.97 mm worst on the held-out drives, and azimuths within 0.43°
of the shipped ones. The calibration is not an artefact of which half it saw.

Two independent estimates of the azimuth agree: the mechanism-only one (drive
axes, co-rigidity, swing-free distal axes) and the position-refined one differ
by at most **0.82°**. Both are carried in `canarm_frames`.

### What the remaining 2 mm is

Not the azimuth and not the lengths — both were fitted and the held-out set does
not improve further. Three contributors, in order:

1. **Residual γ on segment 1**, still 0.31° of spread after the order fix,
   against 0.12° on the other two. Segment 1's four proximal axes were also the
   least orthogonal in §2 at 0.68°.
2. **Out-of-plane marker wander.** A marker sitting 0.1 mm off its plate's plane
   tilts the fitted plane by about 0.1°, and no azimuth can absorb a tilt. The
   offline test bounds this contribution at under 1 mm at the tip for the
   measured wander.
3. **Settling.** Several pose steps still showed 1.0–1.6 mm of marker standard
   deviation over the 1.5 s sample window; the arm is a pneumatic pendulum and
   6 s of settle does not fully still it.

---

## 7. The control GUI

`canarm_control_gui.py` gained two things and was verified against the live
stream (`hw_tests/canarm_gui_kinematics.py`, **11/11**):

* **Lock plates** — a 3 s rest capture that mints this Motive session's plate
  locks marker-only (`x_mode="diagonal45"`, no streamed reference), writes them,
  and restarts the receiver as `CanArmMarkerMocap`. Verified by the class
  changing, not by the call returning.
* **A kinematics line** reporting the live fkine-vs-mocap residual per plate.
  The session read `u2 0.16  u3 0.26  u4 0.49  u5 0.85  u6 1.89 mm, rms 0.95`
  at a pose with 14° of joint travel, with the marker solve succeeding on
  955/955 frames and a worst template residual of 0.028 mm.

The room viewer now prefers the marker-registered receiver too, so its overlay
draws the poses the solve actually used rather than Motive's pivots.

---

## 8. Open items

* Segment 1's proximal axes are 0.68° from mutually orthogonal and its residual
  γ is 2.5× the other segments'. Worth a dedicated slow sweep before it is
  called a hardware property.
* The pose phase's settling is the largest remaining measurement term. A longer
  hold, or a settle criterion on marker sd rather than a fixed timeout, would
  tighten the held-out number.
* `0x110`'s supply-side leak and its +5.1 psi offset are worked around, not
  fixed.
* The RS485 arm has **not** been re-measured for the proximal composition order.
  Its default is unchanged and its numbers are untouched; nothing here claims
  the two arms are assembled alike.
