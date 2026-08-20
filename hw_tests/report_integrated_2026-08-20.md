# Operator-GUI integration test — 24-board arm, 2026-08-20

**Verdict: 28 of 29 checks PASS.** The operator window connects, enumerates all
twenty-four boards, drives the 150 Hz cycle, feeds a mocap strip, spawns and
outlives a MuJoCo room viewer, stages a target without enabling anything, and
releases the port — all in one process against the real arm. The single failure
is quantitative and is the finding this session exists to report: **with the
window's pressure plot running, the cycle achieves 130–135 Hz rather than 150,
and the backend credits 95 % of replies rather than 100 %.** The boards
themselves are not implicated — they answered **24.00 replies per sync edge**,
i.e. every board on every edge, and finished the session with every diagnostic
counter at zero.

| | |
| --- | --- |
| Entry point under test | `canarm_control_gui.py`, a real Tk window driven by `hw_tests/integrated_gui_test.py` |
| Harness method | the GUI's own handlers called directly, the loop pumped with `root.update()` |
| Adapter | CANable 2.0, `16e7497-dirty github.com/normaldotcom/canable2.git`, slcan on **COM58** |
| Port resolution | `bench_env.resolve_can_port()` — VID:PID+serial match, preselected by the GUI itself |
| Host | `WS/.venv/Scripts/python.exe` (CPython 3.13.13), mujoco 3.11.0 |
| Run | 2026-08-20 18:45:11 → 18:48:40, ~3.5 min |
| Machine-readable records | `hw_tests/results/integrated_gui_2026-08-20.json`, `..._profile_2026-08-20.json` |
| Screenshots | `hw_tests/media/` (five PNGs, listed in §8) |
| Baseline | `report_can_bringup_2026-08-20.md` §4 — headless, 9003 cycles at 149.98 Hz, 100.00 % replies |

No pneumatic supply was connected. Nothing was enabled at any point, and that
is enforced rather than intended: the harness decodes the runtime table the
backend is about to send — through the backend's own `_build_targets` and
`proto.build_runtime_table`, so these are the bytes and not a reconstruction of
them — and raises if any command word carries `CONTROL_ENABLE`. That guard ran
three times, and a receive tap ORed together every compact-status flag byte all
session, yielding `0x04` (`COMMAND_SEEN`) and nothing else on all 442 560 of
them.

## 1. Acceptance, item by item

| # | Item | Result |
| --- | --- | --- |
| 1 | 24 node rows with live pressure readings | **PASS** — 0x101–0x118, all 24 numeric |
| 2a | mocap strip on `sim`: bodies visible, `q` not stale | **PASS** — 5/6 bodies at 120.0 fps, `q_stale n` (the 5-of-6 is explained in §4) |
| 2b | `live` degrades loudly rather than hanging | **PASS** — `bodies 0 / 0.0 fps / frames 0 / stale Y / q_stale Y` within 10 s, start and stop each < 5 ms |
| 3a | viewer opens and draws the arm moving with the sim `q` | **PASS** — `MuJoCo : umarm_room`, 2.87 % of pixels changed between grabs 7 s apart |
| 3b | the CAN cycle is unaffected; reply rate stays ~100 % | **FAIL** — 94.88 % at 130.1 Hz, against the headless baseline's 100.00 % at 149.98 Hz |
| 3b′ | *the viewer itself* does not measurably change the cycle | **PASS** — −1.14 pt and −4.6 Hz against the same GUI a minute earlier |
| 3c | closing the viewer leaves the CAN cycle running | **PASS** — 1998 further cycles, 45 671 further replies, `backend.running` still true |
| 4 | target staged without the enable bit, then returned to zero | **PASS** — 0x101 to 2.00 psi and back, enable clear throughout |
| 5 | clean shutdown, port released, no stray processes | **PASS** — reopened from a fresh process; no new process names this workspace |

Item 3b is split deliberately. The criterion as written compares the GUI to a
headless baseline and therefore measures the window as a whole; 3b′ compares
the window with the viewer to the same window without it, which is the question
the criterion was there to ask. The first fails, the second passes, and §3
attributes the difference.

## 2. Cycle statistics, with the viewer and without

All figures are differences across the window rather than the backend's running
totals, since a cumulative rate read sixty seconds in is diluted by every cycle
before it. Jitter is the exception: `jitter_ms_p95` and `jitter_ms_max` are
already a rolling summary of the last 600 cycles (~4 s), so the window's worst
case is the largest sample.

| Window | s | cycles | **Hz** | replies | misses | **credited %** | late | jitter p95 med / max (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline, headless** (bring-up §4) | 60.0 | 9003 | **149.98** | 216 072 | 0 | **100.00** | 0 | 0.0016 / 0.108 |
| GUI only, first 6 s of cycle | 6.0 | 830 | 138.02 | 19 107 | 813 | 95.92 | 3 | 0.210 / 8.27 |
| GUI only, immediately before the viewer | 15.0 | 2024 | **134.75** | 46 642 | 1934 | **96.02** | 32 | 0.300 / 10.55 |
| **GUI + viewer + cycle together** | 66.9 | 8701 | **130.11** | 198 126 | 10 698 | **94.88** | 176 | 0.594 / 20.30 |
| GUI only, after the viewer closed | 15.0 | 1998 | **133.15** | 45 671 | 2281 | **95.24** | 27 | 0.311 / 18.12 |

The viewer's own contribution is the difference between rows three and four:
**−4.6 Hz and −1.14 percentage points**, recovered to within 0.4 points once its
window closes. The remaining **−15 Hz and −4 points** are present whenever the
operator window is open at all, viewer or no viewer.

### The misses are not silence

The harness taps the receive thread independently of the backend and counts
every compact status that arrives, whether or not the backend credits it.

```
442 560 compact statuses over 18 440 sync edges = 24.00 per edge, 24 boards on the bus
backend credited 95.44 %
```

Every board answered every sync edge. The shortfall is entirely the backend's
reply window: `RX_WINDOW_FRAC = 0.82` of a **nominal** 6.67 ms period is 5.47 ms,
and `_publish` reads `_cycle_replies` at that instant and clears it at the next
sync. When the cycle stretches to 7.7 ms, the answers from the high-ID boards —
whose latency the bring-up measured at 3.35 ms median and 4.61 ms max at 0x118,
because CAN arbitration orders replies by node ID — arrive after the window has
already been read and are discarded unread. So the credited rate is a
measurement of **host scheduling**, not of the bus, and it degrades before any
board does.

**What this does not show:** that the arm would behave identically under
pressure. Nothing was enabled, so no board was regulating, and a 7.7 ms sync
period reaching a closed-loop controller tuned for 6.67 ms is a separate
question this session did not ask.

## 3. Where the fifteen hertz go

The window runs two periodic jobs — the node table at 10 Hz and the matplotlib
pressure plot at 4 Hz — and the way to tell them apart is to run the same bus
with each switched off in turn. Both jobs re-arm themselves through
`root.after(..., self._refresh)`, so replacing the attribute is enough to gate
one without disturbing the other or the backend. `--phase profile` does this,
20 s per condition, on the real arm:

| Condition | Hz | credited % | heard / edge | jitter p95 / max (ms) | late |
| --- | --- | --- | --- | --- | --- |
| full window: node table + 24-board plot | 136.59 | 96.46 | 24.00 | 0.197 / 11.49 | 18 |
| **node table only, plot pump idle** | **149.99** | **99.99** | 24.00 | 0.006 / 11.49 | 0 |
| **plot only, node table idle** | **134.13** | **96.15** | 23.99 | 0.140 / 37.84 | 32 |
| neither; Tk pumped, nothing redrawn | 149.94 | 99.99 | 24.00 | 0.005 / 3.27 | 0 |
| plot only, drawing a **frozen** history (no backend lock) | 136.44 | 97.54 | 24.00 | 0.134 / 9.40 | 12 |
| `Backend.history()` copies only, **no matplotlib** | 147.16 | 99.97 | 24.00 | 0.003 / 9.40 | 0 |
| full window again (drift control) | 132.51 | 96.56 | 24.00 | 0.192 / 27.33 | 58 |

The node table costs nothing measurable: with the plot idle the cycle holds
149.99 Hz and 99.99 %, which is the headless baseline. The plot costs all of it.
Within the plot, the two halves separate as well — `Backend.history()`, whose
copies are made while holding the lock the cycle thread needs to publish, costs
**2.8 Hz**, and matplotlib costs the other **13.5 Hz**. Roughly 85 % of the
deficit is therefore the redraw itself and not lock contention.

The cause is a scale change rather than a defect: `ControllerApp._replot` was
written for the eight-channel TLE manifold and now runs `axes.clear()` followed
by forty-eight fresh `plot()` calls and a `tight_layout()` four times a second,
while `Backend.history` returns `HISTORY_SECONDS = 60` of samples — 9000 points
per board — of which the plot draws at most 900. Three remedies are available:

1. **Reuse the artists.** Persistent `Line2D` objects updated with `set_data`,
   in place of `clear()` plus forty-eight `plot()` calls, is where the 13.5 Hz is.
2. **Bound the copy at the source.** A `max_points` argument on
   `Backend.history` that decimates *under* the lock would return 900 points
   instead of 9000 and recover the 2.8 Hz.
3. **Plot what is asked for.** Every board is selected by default after a scan,
   so the plot draws twenty-four whether or not anyone is watching them.

*(Supervisor addendum: remedies 1 and 2 were applied after this report was
filed and re-measured on the real bus — see §11.)*

## 4. The mocap strip

On the synthetic stream the strip reads `bodies 5   120.0 fps   frames 492
valid 492   stale n   q_stale n`, and it held `q_stale n` at 120 fps for the
whole session, including the sixty-seven seconds the viewer was up
(`frames 4392   valid 4392`).

The count is five of six rather than six, and the cause is the fixture rather
than the receiver: `sim_stream.plate_poses_from_q` places the base plate at the
mocap origin with an identity orientation by default, so that row of the pose
array is bit-identical to the identity the array was initialised with. The
strip's visibility heuristic cannot distinguish a plate that happens to sit at
the origin from a row Motive never filled, and on live data it does not have to,
because a real base plate has a real pose. The limitation is recorded in
`_mocap_line`'s comments.

On `live` the strip reads `bodies 0   0.0 fps   frames 0   valid 0   stale Y
q_stale Y` — loud, immediate and correct, since Motive is still not streaming
(see `report_mocap_2026-08-20.md`). Starting the live receiver took 4 ms and
stopping it took under 1 ms, so the dead stream costs a wait rather than a hang.
That path did not work before this session; see fix C.

## 5. Target staging without enabling

Dragging 0x101's own channel bar — through `ChannelBar._drag`, the handler the
widget binds to `<Button-1>` — to 2.00 psi, with the cycle running and the
enable bit clear:

| | before | staged | after |
| --- | --- | --- | --- |
| GUI row `Tgt` | `0.00` | `2.00` | `0.00` |
| `backend.nodes[0x101].target_psi` | 0.0 | 2.0 | 0.0 |
| table frame `0x091` slot for 0x101 | 944 counts | **1065 counts** | 944 counts |
| slot flags | 0 | **0** | 0 |
| neighbours 0x102 / 0x103 | 944 / 944 | **944 / 944** | 944 / 944 |
| 0x101 measured pressure | 936 counts (536 samples, min = max) | **936 counts (540 samples, min = max)** | — |
| 0x101 status flag OR | `0x04` | **`0x04`** | `0x04` |

The staged table frame went out as `80072904B003B003`: the `0x80` marker and
start slot 0, the all-slots mask `0x07`, then `0x0429` = 1065 for 0x101 and
`0x03B0` = 944 twice for its neighbours. The top nibble of every word is zero,
which is the enable bit clear. The board acknowledged the command
(`COMMAND_SEEN` stayed set, `ENABLED` never appeared) and its pressure did not
move by a single count in either direction.

This is the intended behaviour and it is worth stating plainly: **a target is a
number the board stores, and the enable bit is what promotes it.** The GUI's
per-channel bar and its group slider both reach `Backend.set_target` and neither
touches `set_enabled`, so staging is reachable without ever arming a channel.

## 6. What the boards recorded

`CMD_GET_CAN_DIAG` on all twenty-four boards immediately after the GUI's scan,
then `CMD_CLEAR_CAN_DIAG`, so every counter from that point belongs to this
session and to nothing before it.

**After the session: every counter zero on every board** — no errors, no
warnings, no starvation recoveries, no RX overflows, no transmit failures, no
invalid frames. The cycle's worst stall was a 36 ms gap between sync edges and
289 late cycles, and none of it registered anywhere on the arm.

Two findings from the before/after comparison:

- **The legacy boards arrived carrying `error_count = 4`,
  `warning_count = starvation_count = 202`, `last_error_reason = 0x80`**, which
  latched `STATUS_ERROR` into their compact status in an earlier run of this
  harness. **Starvation here is a ten-second timeout**, not a late sync edge:
  `WATCHDOG_subroutine` ticks every 500 ms and calls
  `CAN_recover_from_starvation` after twenty ticks
  (`firmware/legacy/main/main.c:997`). 202 recoveries ≈ 34 minutes of idle bus
  between sessions — exactly the history the bring-up report already described,
  and **not** something the GUI's stalls caused. The four errors did not
  increase across four GUI sessions.
- **`last_error_reason = 0x80` is `CAN_ERR_REASON_MERR`**
  (`firmware/legacy/include/main.h:102`), the MCP2515 message-error interrupt —
  not starvation (`0x20`). With `last_eflg = 0` and every frame counter zero,
  the most likely origin is the bus-revival transient the bring-up report
  documents at §2.

## 7. Integration bugs found and fixed

**A. The room strip sat under the taskbar (`canarm_control_gui.py`).** With
twenty-four board rows the window asks for 1091 px; Tk clamps against
`winfo_screenheight` and knows nothing about a taskbar, so on this 1080 px
display the bottom 56 px — exactly the mocap strip — fell in the 48 px the
taskbar occupies. Added `work_area()` / `fit_to_work_area()` (accounting for
the 31 px title bar `wm geometry` does not size), called at construction and
after each scan; the strip is now packed **before** the inherited body so
pack's parcel order cannot squeeze it out.

**B. A viewer closed at its own window was never noticed
(`canarm_control_gui.py`).** The button kept reading "Close viewer", so the
next press spawned a *second* viewer, and the dead child was never joined.
Added `_reap_viewer()` to the existing `_refresh_mocap` pump.

**C. The live mocap receiver could be started but never read or stopped
(`canarm_control_gui.py`).** `_build_mocap` returned whatever `start()` handed
back, and the two `start()` methods disagree: `CanArmSimStream.start` returns
`self`, `MocapRx.start` returns the `NatNetClient`. The strip read
`mocap unreadable: AttributeError`, and `_stop_mocap` left the SDK's non-daemon
threads running unreachable. Both branches now build, start and return the
receiver.

**D. The cycle status line printed a constant where a measurement belonged
(`TLE_PCB/VEMA_TLE_controller.py`).** `_refresh` displayed `P.CYCLE_HZ`, so the
line read `150 Hz` while the bus ran at 132. It now reports the achieved rate
over a rolling two seconds beside the requested one: `130.1 Hz (asked 150)`.

`canarm_control_gui.py --self-test` passes after all four.

## 8. Files

Created: `hw_tests/integrated_gui_test.py` (`--phase full` acceptance,
`--phase profile` attribution), the two JSON records, and five screenshots in
`hw_tests/media/` (`control_gui_connected.png`, `control_gui_with_viewer.png`,
`room_viewer_sim.png`, `room_viewer_sim_t2.png`,
`control_gui_target_staged.png`).

Edited: `canarm_control_gui.py` (fixes A, B, C),
`TLE_PCB/VEMA_TLE_controller.py` (fix D).

## 9. Open items

1. **The pressure plot costs the cycle 15 Hz and four points of credited reply
   rate** — measured and attributed in §3. *(Addressed post-report; see §11.)*
2. **The reply window is defined against the nominal period, not the achieved
   one** — `RX_WINDOW_FRAC * (1 / cycle_hz)` discards answers that did arrive
   whenever the cycle stretches. Fixing §3 removes the symptom without removing
   this.
3. **The legacy boards' four `MERR` errors are unexplained** — they predate
   every run in this session and did not grow across four of them.
4. **`0x104` still reads +4.13 psi at rest**, unchanged from the bring-up
   report's +4.27.
5. **The `bodies 5/6` undercount** on the synthetic stream (§4) is cosmetic and
   sim-only; a per-body presence counter on `MocapRx` would fix it for both
   streams at once.
6. **Per-board latency under host load** has no GUI-loaded counterpart yet.

## 10. Reproducing this

```
WS/.venv/Scripts/python.exe hw_tests/integrated_gui_test.py
WS/.venv/Scripts/python.exe hw_tests/integrated_gui_test.py --phase profile
```

Both need a visible desktop. The acceptance run exits 0 only when every check
passes, decodes the enable bits of the table it is about to send three separate
times, and emits no OTA command, no set-ID command and no enable bit under any
argument.

## 11. Supervisor addendum: plot remedies applied and re-measured

After this report was filed, remedies 1 and 2 of §3 were applied and the plot's
refresh halved, then `--phase profile` re-run on the real bus:

- `ControllerApp._replot` now keeps persistent `Line2D` pairs updated with
  `set_data` (no `clear()`, no per-frame `tight_layout()`; legend rebuilt only
  when the series set changes; `relim(visible_only=True)` +
  `autoscale_view(scalex=False)` for the y axis).
- `Backend.history` gained `max_points`, decimating **under the lock** via
  `itertools.islice` — the display asks for 900 points instead of copying 9000.
- `PLOT_MS` 250 → 500: at 24 boards even the artist-reuse redraw costs ~5 Hz
  at 4 Hz refresh, and a monitoring plot is not read faster than 2 Hz.
- The profile harness's `cached_history` gate now mirrors the new signature
  (the first re-run died on the mismatch *inside* `_replot`, before its
  self-re-arming `root.after` — killing the plot chain for every later
  condition; the gate also re-arms on any replot exception now).

Re-measured (20 s per condition, real bus, same guard):

| Condition | before | after |
| --- | --- | --- |
| full window (table + plot) | 136.59 Hz / 96.46 % | **145.43 Hz / 98.81 %** |
| node table only | 149.99 Hz / 99.99 % | 150.01 Hz / 100.00 % |
| plot only | 134.13 Hz / 96.15 % | 144.36 Hz / 98.40 % |
| plot only, frozen history | 136.44 Hz / 97.54 % | 145.74 Hz / 98.81 % |
| history copies only, no matplotlib | 147.16 Hz / 99.97 % | 149.28 Hz / 99.99 % |
| full window again (drift control) | 132.51 Hz / 96.56 % | 144.43 Hz / 98.91 % |

The lock-held copy cost is gone (149.3 Hz with copies only), and the operator
window recovers ~9 Hz and 2.4 points. The residual ~4.6 Hz is the rasterisation
of forty-eight 900-point lines at 2 Hz on the Tk thread; open item 2 (the reply
window defined against the nominal period) still stands and would reclassify
most of the remaining "misses" as the arrived-late answers they are.
`canarm_control_gui.py --self-test` passes after all of it.
