# Real-system data and a twin fitted to it — CAN arm, 2026-09-10

Two campaigns against the live 24-actuator UMArm ProMax, 273 867 cycles at
150 Hz, and a digital twin fitted to them. This is what was measured, what was
found, and what is still wrong.

---

## 1. The instruments agree on one motion

`hw_tests/canarm_hello.py`, 8 of 8 checks. One actuator driven alone at 12 psi:

| instrument | what it said |
| --- | --- |
| bus | `0x101` read **12.03 psi** against a 12.0 psi target |
| mocap | joint 2 (`s1.u2.t3`) moved **+16.50 deg**, the largest of the twelve |
| the map | names exactly joint 2, sign `+1`, for `0x101` |
| camera | 1.25 % of pixels changed, centred on the **left** arm in frame |

The room holds two ceiling-hung UMArms and a Kinova Gen3, so which robot the
2000–2005 marker block belongs to is not derivable offline. It is the arm with
the black tubing, not the one against the white backdrop.

## 2. The 150 Hz stack was losing 15 Hz to the GIL

The cycle held a perfect 6.667 ms median and *skipped whole periods* — 138 Hz
with 3 % of cycles missing a board. `hw_tests/canarm_cycle_profile.py` adds one
component at a time against the bring-up's 150.00 Hz baseline:

| stack | rate | periods skipped | every board answered |
| --- | --- | --- | --- |
| bare cycle | 149.91 Hz | 0.04 % | 99.91 % |
| + NatNet receiver | 139.65 Hz | 7.12 % | 97.14 % |
| + per-cycle observer | 140.66 Hz | 6.31 % | 96.64 % |
| + JSONL recorder | 141.80 Hz | 5.55 % | 96.85 % |
| + excitation | 136.81 Hz | 9.17 % | 96.78 % |
| + camera | 135.59 Hz | 10.04 % | 96.46 % |
| + frozen GC | 134.59 Hz | 10.76 % | 96.43 % |
| **+ 0.5 ms GIL switch interval** | **148.60 Hz** | **0.90 %** | **99.73 %** |

CPython's default thread-switch interval is 5 ms — **75 % of this arm's cycle
period** — so any other thread that takes the GIL can hold it for most of a
cycle. Cutting it to 0.5 ms makes nothing faster; it buys the cycle thread the
right to interrupt, which was the only thing it was short of. The collection
sessions then ran at 149.4–150.0 Hz with **0 rows dropped**.

## 3. Disabling a 7 mm board does not shut its valves

`firmware/legacy/main/main.c:517` skips the whole control body when the enable
bit is clear, and all three `gpio_set_level(V_IN/V_OUT, …)` calls sit below that
branch. A disable **freezes** the solenoids. The same tree has no link-loss
timeout — its watchdog only resets the MCP2515 — so a board disabled while
inflating goes on inflating, with nothing able to stop it but a fresh enabled
frame or power.

Measured (`hw_tests/canarm_disable_behaviour.py`): 24 boards charged to 12 psi
and disabled for 8 s, twice.

* drift repeated between the two traps to a **median of 0.003 psi/s** — it is a
  board property, not a random freeze point;
* it spanned **0.011 to 1.012 psi/s** across the arm, a factor of 88, which a
  leak alone does not explain;
* `0x109`, `0x113` and `0x117` emptied to 0.1–4.0 psi: they froze with the
  **exhaust open**, which proves the valves do not shut on a disable;
* `0x110` rose at **+0.041 psi/s**, the supply-side leak recorded in 2026-08.

None happened to freeze inflating. That is the sample, not a guarantee. Every
exit path now commands zero and waits for the arm to empty before any enable bit
goes down; `hw_tests/canarm_park.py` does it standalone and verifies it.

## 4. Mocap does not divide the cycle rate

Motive streams 120 Hz against 150, so one cycle in five repeats the previous
frame. On a 4500-cycle **rest** recording — an arm that was not moving — the
held series' `qdot` ran **1.8×** the interpolated one. The recording now keeps
the raw stream beside the per-cycle held view, and `collection/resample.py` puts
it onto the sync edges: interpolating below the cycle rate, low-pass-then-
decimating above it, so the same file survives Motive being raised past 150 Hz.

Measured on the real session: stream 119.77 Hz, cycle 149.41 Hz, offset between
`perf_counter` and `monotonic` **−1.1 µs**, 0 Motive frame gaps, 4 × 10⁻⁵ of
cycles uncovered.

## 5. A missed reply was being read as a reading

The collector fills a cycle's pressure row with zeros and overwrites the boards
that answered, so a miss lands as **zero counts** — −15.53 psi on a TLE board
and −13.44 on a 7 mm one, identically every time, in data whose real range is
0.4 to 25.6 psi. The reply-latency column was already null exactly there; only
the reader was not using it. **874 of 3.03 M board-cycles (0.029 %)** are now
carried forward from the last answered reading and marked.

## 6. What was collected

| session | cycles | minutes | phases |
| --- | --- | --- | --- |
| `session_20260910_012159` | 126 434 | 14.2 | rest, supply probe, leak probe, all 24 single-actuator staircases, 8 of 12 pair sweeps |
| `session_20260910_013843` | 147 433 | 16.7 | rest, all 12 chirps, 5 min of 12-joint random walk, 48 ringdowns, 90 s held-out validation |

The envelope clamp fired **0 times** in either. The generator satisfies the
operator's 30 psi pair rule by construction: worst single commanded pressure
25.50 psi, worst pair sum 26.00 psi.

The first campaign was cut short at 14.2 min by `WriteFile failed (OSError(22,
'The operation completed successfully.'))` from the USB-CDC driver — a write
that both failed and reported success. Every board answered normally up to that
cycle and again afterwards. The campaign now revives a dead cycle (reopen,
rescan, restart, resume), which is what carried the second campaign through the
same fault at 8.4 min and let it finish all 134 segments.

## 7. The anomaly flags were coincidences

Twelve live flags fired ("commanded but still"). The live watcher judges one
instant, and a joint in a serial chain under gravity can legitimately fail to
move in one. `hw_tests/canarm_anomaly_review.py` fits every joint's
responsiveness over the whole recording instead:

| | range | median | correlation |
| --- | --- | --- | --- |
| degrees of joint angle per psi of commanded differential | 0.529 – 0.875 | 0.720 | r 0.82 – 0.91 |

Twelve joints in one tight population, no outlier, no joint below 0.74 of the
median. **No fault.** This does not rule out a joint that moves the wrong way or
by the wrong amount — that needs the model, and §9 is where it is judged.

## 8. The proximal universal joint was routed to the wrong axis

Rolling the fitted twin against the arm gave a cross-correlation matrix between
the twin's twelve joints and the arm's that was a **permutation with three
transpositions**:

| twin joint | best-matching real joint | correlation |
| --- | --- | --- |
| j0 | **j1** | +0.93 |
| j1 | **j0** | +0.97 |
| j2, j3 | j2, j3 | +0.97, +0.98 |
| j4 | **j5** | +0.94 |
| j5 | **j4** | +0.89 |
| j6, j7 | j6, j7 | +0.97, +0.96 |
| j8 | **j9** | +0.98 |
| j9 | **j8** | +0.97 |
| j10, j11 | j10, j11 | +0.98, +0.93 |

Three swaps, all of them the **proximal** universal joint, none of them distal.
That asymmetry is the evidence: a mistake in the shared derivation would have
moved both rings. Only the proximal joint is reordered between `q` and `qpos`,
because only it composes y-then-x — the `PROXIMAL_ORDER` measured on 2026-08-21
that took `fkine`'s held-out error from 4.64 mm to 1.99 mm. The seat table was
assigning muscles by `q` index while the hinge it reached was the first declared
one. Swapping the two proximal azimuth pairs took the mean per-joint correlation
from **0.402 to 0.882**, every joint positive.

Three offline tests failed on the fix and were right to: they asserted the axis
assignment from the same closed form the table used, so table and tests were
consistently wrong and neither could catch it. They now assert what the geometry
pins and nothing more.

## 9. The twin, scored open loop on the held-out sequence

90 s, 13 501 cycles, a signal family and a seed the fit never saw. The twin is
given **only the recorded pressure targets** — never the recorded pressures,
never the recorded joints — so every divergence accumulates.

| stage | joint RMS | nrmse | note |
| --- | --- | --- | --- |
| unfitted seed | 14.766 deg | 1.289 | RS485 constants for a different actuator |
| flow net alone (routing bug) | 29.002 deg | 2.657 | pressures right, torques wrong |
| flow net + routing fix + outer fit | **10.850 deg** | **0.967** | |

split by population: **TLE 9.121 deg / nrmse 0.762**, 7 mm 11.715 / 1.069.
Mean per-joint correlation **+0.756**; segment 1 **+0.850**, segment 2 +0.899,
segment 3 +0.520.

Held-out **pressure** RMS went 20 308 Pa → **3682 Pa (0.534 psi)**, a 5.5×
improvement, over 150 shooting epochs on 25 056 windows.

**The middle row is the one worth keeping.** Fitting the pressures made the
joint error *worse*, because the unfitted twin barely pressurised and scored a
flattering number by not moving, while a correct pressure through a wrong force
law swings the arm two and a half times too far. A twin that does not move is
not a good twin; it is an unfalsifiable one.

## 10. What is still wrong, stated rather than left implicit

* **Segment 3 tracks worst** (+0.52 mean correlation against +0.85 and +0.90).
  Error accumulates down a serial chain, but the gap is large enough that the
  distal ring's geometry deserves the same scrutiny the proximal one just got.
* **No mass on this arm has been weighed.** `link_density`, `plate_mass` and
  `bracket_mass` are RS485 numbers; the 1.833 kg moving total is a *model*. Both
  the ring frequency and the damping ratio read mass, so the dissipation fit is
  absorbing a mass error and will report a good loss while doing it.
* **The ring radii are CAD, not measured.** In a twin the ring radius *is* the
  moment arm, so if CAD is wrong every predicted torque is wrong by that factor
  — and the outer fit's per-segment gain multipliers (0.020, 0.045, 0.009) are
  currently absorbing exactly that.
* **The TLE blend width (6000 Pa) is an assumption**, not a measurement, and the
  closed-hold criterion the leak fit keys off is defined by it.
* **`ring_analysis`'s band (1.5–12 Hz) and threshold (0.25 deg) are RS485
  defaults.** This arm's ring frequency has not been measured, although the
  48-episode ringdown phase in session 2 is the data to measure it from.
* **The camera stopped 298 s into the first campaign** and did not return, so
  neither session carries video. See §11.

## 11. The camera

The bench body powered itself off 298 s into the first campaign
(`MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`), and the capture card then dropped
off Windows entirely — no camera device enumerates. It needs a physical replug
or the camera powering back on; nothing in software reaches it.

What exists: `hw_tests/media/hello_101_20260910_002930.mp4` (the arm actuating,
from the verification run) and the before/after pair image beside it. The
side-by-side deliverable therefore carries two panels rather than three — the
same MuJoCo model posed at the **measured** joint angles and at the
**predicted** ones. That is the comparison that carries the claim in any case:
same renderer, same camera, same model, differing only in where the angles came
from. The camera panel shows that the middle panel is a faithful account of a
real robot rather than a plausible animation, which is worth having and is not
what the number rests on.

The recorder is now hardened against the fault: bounded retries, one reopen
attempt, then it marks itself lost and lets the session continue. Re-run
`collection/campaign.py --phases validation` once the camera is back to get the
third panel.

---

## Files

| what | where |
| --- | --- |
| the recordings | `data/session_20260910_012159/`, `data/session_20260910_013843/` (gitignored, ~300 MB) |
| the flow net | `digital_twin/checkpoints/canarm_flow.npz` |
| the mechanical fit | `digital_twin/checkpoints/canarm_outer.json` |
| the video | `deliverable/twin_vs_real_validation.mp4` (33 MB, gitignored) |
| its numbers | `deliverable/deliverable_validation.json` |
| a frame of it | `deliverable/frame_sample.jpg` |
| cycle-rate attribution | `hw_tests/results/cycle_profile_2026-09-10.json` |
| the anomaly review | `hw_tests/results/anomaly_review_2026-09-10.json` |
