# Recon: data collection, camera capture, side-by-side video deliverable

Scope note up front — **what I read vs. did not**. I read in full: `side_by_side.py` (759 L), `collab_video.py` (673 L), `scene_camera.py` (480 L), `UMArm_SIM/bounce_demo.py` (282 L), `refined_campaign.py` (279 L), `digital_twin/data_schema.md`, `legacy_host/docs/sync_rules.md`, the `start/stop_data_collection` + metadata/manifest/JSONL/excitation blocks of `legacy_host/host/pc_backend/src/main.cpp`, and `UMArm_KoopmanMPPI/excitation.py`'s pressure and duty families. I read *selectively* (grep + targeted ranges): `experiments.py` (6568 L — only `TrialEngine.__init__`, `note_contact`, `_envelope`, `_chirp_psi`, the CHIRP block and the protocol constants), `intercept_drive.py` (1449 L — the `InterceptDrive` dataclass and phase law, **not** the parallel search or `--aim`), `slingshot_drive.py` (660 L — the dataclass, PLL and `psi`, **not** the search), `ft_daq.py` (839 L — constants, `Block`, `_run`, `drain`, `metadata`, `save_block`/`load_block`, **not** `start`/`stop`/`capture_bias` internals or `ati_cal.py`), `collab_replay.py` (1005 L — `BenchRun.__init__`, `zero`, `_resolve_kinova`, overlay indices; **not** `_draw_overlay`, `build_scene_xml`, `collab_scene.py`). I did **not** read: `collab_playback.py`, `collab_scene.py` (56 KB), `gui/`, `ring_frame.py`, `mount_transforms.py`, `swing_analysis.py`, `closing_speed.py`, any test file, or `UMArm_KoopmanMPPI/collect.py` beyond its header.

**The single most important structural finding:** the file you asked about, `UMArm_COLLAB/experiments.py`, is *not* the hardware collection loop. It is the **simulation** condition catalog + `TrialEngine`. The real simultaneous-hardware loop (Kinova + ATI + UMArm + mocap + camera, one process, one clock) is at
`C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_ROBOT_CONTROL/collab_hardware.py` (1724 L). That is the file to port.

---

## 1. The simultaneous-hardware collection loop

### 1.1 Process architecture and the one clock

`collab_hardware.py:1-63` states the contract: everything shares `time.perf_counter()`.

Five asynchronous producers, each on its own thread, all stamping `perf_counter`:

| producer | rate | file:line |
|---|---|---|
| bus control slice (the loop itself) | 40 Hz (`SLICE_S = 1.0/40.0`, `collab_hardware.py:98`) | `collab_hardware.py:938` |
| ATI Nano17 via NI USB-6210 | **10 000 Hz** (`DAQ_RATE_HZ = ft_daq.IMPACT_RATE_HZ`, `collab_hardware.py:124`; `ft_daq.py:174`) | `ft_daq.py:431` `_run` |
| plate mocap receiver (`PlateRx`) | asks 240 Hz, **delivers 73 Hz** (`PLATE_STREAM_HZ = 240.0`, `collab_hardware.py:196`) | `collab_hardware.py:354` `_stream` |
| ring + pad rigid-body receivers (listener-driven) | ~111 Hz | `collab_hardware.py:1417` `_body_window` |
| scene camera (Cam Link) | ~60 fps | `scene_camera.py:252` `_grab_run` |
| Gen3 bridge publish | 40 Hz (`GEN3_PUBLISH_HZ = 40.0`, `collab_hardware.py:204`) | `collab_hardware.py:472` `poll` |

**The one clock caveat you must port.** Mocap's ring buffer is stamped `time.monotonic`, everything else `perf_counter`. Both are QPC on Windows, so the offset is a constant read once (`collab_hardware.py:910-916`):

```python
t0 = time.perf_counter()
# THE OTHER CLOCK. ... both are QPC on this platform, so the offset is
# a constant and one reading of each is all that is needed
t0_mono = time.monotonic()
```

and the high-rate `q` window is rebased at save time (`collab_hardware.py:1121-1126`):

```python
w = self.plates.rx.snapshot_window(t0_mono, time.monotonic())
q_hi = {"t": np.asarray(w.t, dtype=float) - t0_mono + t0, ...}
```

**The DAQ clock is the accurate one; the host only sets zero.** `ft_daq.py:466-470` back-dates sample 0 of the first read, and every later timestamp comes from the DAQ's own sample clock (`ft_daq.py:565`):

```python
t = self._t0 + np.arange(first_kept, self._write) / self.rate_hz
```

### 1.2 What is logged per tick

`Recorder.slice()` at `collab_hardware.py:237-256` — seven columns per 40 Hz slice:

```python
def slice(self, t, phase, psi_cmd, psi_meas, q, fmag, gen3_q=None) -> None:
```

- `t` — float, seconds since `t0` (run clock)
- `phase` — str, one of `("reach","settle","bias","windup","strike","retract")` (`protocol.py:92`), from `engine.phase_at(t)`
- `psi_cmd` — (24,) float psi, commanded
- `psi_meas` — (24,) float psi, from each node's compact reply (`K.ten_pa_to_psi(reply.pressure_10pa)`, `collab_hardware.py:1010`)
- `q` — (12,) float **radians**, mocap; NaN when no fresh pose — never forward-filled at record time
- `fmag` — float, `blk.fmag.max()` over the samples drained this slice (i.e. a **max**, not a mean)
- `gen3_q` — (7,) float **degrees**, Kortex convention `[0,360)`; `np.full(7, np.nan)` when the bridge is down — explicitly *never* a fabricated zero (`collab_hardware.py:245-251`)

### 1.3 The output files per run

`Recorder.save()` at `collab_hardware.py:257-306` writes `run.npz` with these keys (arrays, `allow_pickle=False`):

| key | shape | units |
|---|---|---|
| `t`, `phase`, `psi_cmd`, `psi_meas`, `q`, `fmag`, `gen3_q` | (n,), (n,), (n,24), (n,24), (n,12), (n,), (n,7) | s / str / psi / psi / rad / N / deg |
| `ring_t`, `ring_pos`, `ring_quat`, `ring_strike`, `ring_normal` | ~111 Hz | s (perf), m, wxyz, m, unit |
| `plate_t`, `plate_pos`, `plate_quat`, `plate_q` | ~73 Hz | s, m, wxyz, rad |
| `pad_t`, `pad_pos`, `pad_quat` | ~111 Hz | the Gen3 pad at camera rate — **exists because closing speed is a derivative and 40 Hz over a 0.2 s advance is three points** (`collab_hardware.py:288-295`) |
| `q_hi_t`, `q_hi`, `q_hi_u`, `q_hi_frame` | full camera rate | rebased to `perf`; depth bounded by `mocap_constants.RING_SECONDS`, so a long run keeps only its tail |

Plus, in the same directory:
- `gauge.npz` — `ft_daq.save_block()`, `ft_daq.py:630-646`: `t`, `volts` (n,6), `wrench` (n,6), `clipped` (n,) bool, `meta` (JSON string). **Raw volts are the record; the wrench is a convenience** — `load_block` re-derives the wrench under today's calibration by default (`ft_daq.py:648-680`).
- `scene.mp4` + `scene.stamps.npy` + `scene.meta.json` — see §2.2.
- `meta.json` — `collab_hardware.py:1041-1113`, ~40 fields (see §1.5).

### 1.4 Episode segmentation

Two mechanisms, and they are different in kind:

**Between reps (the outer loop).** `stage_swing()` at `collab_hardware.py:1453-1477` runs `reps` calls to `run_condition`, each landing in its own directory `swing01`, `swing02`, …, with `REST_BETWEEN_REPS_S = 12.0` between them (`collab_hardware.py:207-210`). The rationale is measured: `|q|` L2 peaks **2.1-2.4 s after the vent**, and a windup that starts from a moving arm is a different windup.

**Within a run (the phase machine).** `protocol.py:79-92` fixes the nominal timeline:

```
T_REACH   = 2.5   T_SETTLE = 0.6   T_BIAS   = 0.5
T_WINDUP  = 1.5   T_STRIKE = 2.2   T_RETRACT = 1.5
T_PRESTRIKE = T_REACH + T_SETTLE + T_BIAS   # 3.6
T_SWING     = T_PRESTRIKE + T_WINDUP        # 5.1
T_TOTAL     = T_SWING + T_STRIKE + T_RETRACT
PHASES = ("reach", "settle", "bias", "windup", "strike", "retract")
```

Everything downstream of the collision, however, hangs off the **gauge-reported contact instant**, not off a clock. That is the design's core claim (`collab_hardware.py:16-22`): *"a clock would not transfer; a gauge does."*

### 1.5 Clock alignment across sensors — the derived zero

`t = 0` for every downstream figure and video is the **contact instant**, resolved by `BenchRun.zero()` at `collab_replay.py:374-385`:

```python
def zero(self) -> float:
    if self.contact_s is not None:
        return float(self.contact_s)
    m = self.phase == "strike"
    return float(self.t[m][0]) if m.any() else 0.0
```

Every stream is rebased to that zero in `collab_video.VideoRun.__init__` (`collab_video.py:118-160`):

```python
self.z0 = self.bench.zero()          # run-clock instant drawn as 0
self.t0 = self.bench.t0              # perf instant of run start
self.contact_rel = float(meta["contact_s"]) - self.z0
self.stamps_rel = np.load(sp) - self.t0 - self.z0      # camera frames
self.gauge_rel  = np.asarray(blk.t) - self.t0 - self.z0
```

Note the two subtractions: camera stamps and gauge stamps are absolute `perf_counter`; `bench.t` is already run-relative.

### 1.6 Safety limits and abort paths

**Pressure envelope, applied twice.** The operator's numbers are 35 psi per actuator and 35 psi summed across an antagonistic pair (`collab_hardware.py:41-44`). Enforced (a) by the engine — `TrialEngine._envelope` clips to `psi_cap` then calls `JPID.apply_pair_budget(capped, self.pair_max)`; (b) again in wire units by `ArmBus`'s own hard cap.

**The collision trigger and its gate.** `CONTACT_TRIGGER_N = 1.5` N (`collab_hardware.py:100-104`) — 35 sigma of the rig's 0.04 N/sample `|F|` noise floor. Requires `CONTACT_TRIGGER_SAMPLES = 3` consecutive samples (0.3 ms at 10 kHz) (`collab_hardware.py:120-123`). **This became a per-run default, not a constant, after a real failure:** when the Gen3 moves, its own payload puts 2.10 N through the cell at a 0.10 m/s advance — 40 % over the trigger — and on `collab_20260828_220012` that transient *was* reported as the collision, 0.7 s early, and the vent was timed from it (`collab_hardware.py:105-119`). Hence `Campaign.contact_gate_s`, which refuses any report before the plate could physically be there:

```python
if gate_s is not None and t_hit < float(gate_s):
    refused += 1
    refused_peak = max(refused_peak, float(mag[k]))
```
(`collab_hardware.py:954-967`) — and the refusals are *recorded* (`contact_reports_refused`, `contact_reports_refused_peak_n`, `collab_hardware.py:1068-1069`), because "the pad's own motion was refused as a collision" is itself a measurement.

**Geometric keep-out.** `_check_keepout` (`collab_hardware.py:1240-1256`) raises before any drive:

```python
if d < SWING_KEEPOUT_M:
    raise RuntimeError("REFUSED: the Gen3's %s is %.0f mm from the arm ...")
```
with `SWING_KEEPOUT_M = 0.38` m, `PARK_CLEARANCE_M = 0.45` m, `ARM_RADIUS_M = 0.09` m (deliberately over-claimed, since it is *subtracted*) (`collab_hardware.py:126-155`). The Gen3's own factory HOME **does not meet it** — measured 112 mm from the ring's striking face.

**The vent path, on both the normal and the exception path.** `run_condition`'s `finally` (`collab_hardware.py:1025-1027`):

```python
finally:
    self.bus.all_off(0.5)
    self.safe_arm()
```

and `safe_arm` (`collab_hardware.py:783-796`) is `vent_all(VENT_S=6.0)` → `all_off(0.5)` → `quiet(K.FAILSAFE_SILENCE_S)`. The final `quiet` is the point: silence lets **every node's own firmware failsafe** close its valves independently of this process.

**Fresh-pose refusal.** A run will not start on a stale mocap frame (`collab_hardware.py:886-890`):

```python
if q0 is None or not fresh:
    raise RuntimeError("the cameras are not delivering a fresh q — "
                       "the windup's controller has nothing to close around")
```

**Bias capture is unloaded and inside the protocol's own window** (`collab_hardware.py:896-903`): `self.daq.drain(); bias = self.daq.capture_bias(1.0)` before anything is commanded. *"A bias captured under load subtracts that load from every later reading and is invisible afterwards."*

**Two-stage protocol.** `--stage swing` (Gen3 parked, arm swings into air, establishes where the trajectory actually goes) must precede `--stage collide` (dome walked onto the *measured* strike point). The Gen3 is **never sent a simulated joint vector** — placement is Cartesian steps against the cameras, `PARK_STEP_DEG = 15.0`, max `PARK_MAX_STEPS = 8` (`collab_hardware.py:45-50`, `157-166`).

**Provenance in every recording.** `meta["reproduce"] = _provenance()` (`collab_hardware.py:1090-1097`) carries the commit, the command line, the bench frames' timestamp and the whole mounts file. `BenchRun` rebuilds the world from *that*, not from today's `bench_frames.json`, and falling back is loud (`collab_replay.py:171-183`).

---

## 2. THE VIDEO DELIVERABLE (exhaustive)

There are **two layers**. `collab_video.py` renders *one* 1920×1080 clip per view; `side_by_side.py` composites *two* such clips into a 1920×540 pair aligned at the collision. There is no ffmpeg on that machine — **cv2 `mp4v` only** (`side_by_side.py:48-50`).

### 2.1 Layer 1 — `collab_video.py`: three edits from one run directory

CLI: `python UMArm_COLLAB/collab_video.py <run_dir> [--which all|real|visualizer|mujoco|both] [--out DIR] [--fps 60]`

Global constants (`collab_video.py:68-99`):

```python
CANVAS_W, CANVAS_H = 1920, 1080
OUT_FPS = 60.0
BEFORE_S = 3.5          # leader before the collision
AFTER_VENT_S = 2.5      # window = vent_delay + 2.5, so a 3 s vent is covered whole
STRIP_H = 220           # force strip height, px
FLASH_CONTACT_S = 0.5   # red COLLISION box duration
FLASH_VENT_S = 1.5      # purple DEFLATE+RETRACT box duration
SIM_CAM = {"azimuth": 90.0, "elevation": -12.0, "distance": 1.6}
```

`SIM_CAM` is chosen to **match the Nikon's standpoint** (Gen3 enters from frame right, arm hangs centre) and is *identical* for the visualizer and the simulation so the two compare shot to shot. `lookat` is filled per run from the striking face's rest position.

**Time window** (`collab_video.py:162-175`, `VideoRun.window`):
```python
hi = (self.vent_rel if self.vent_rel is not None else 0.0) + after_vent
lo = -float(before)
lo = max(lo, float(t_rel[0]));  hi = min(hi, float(t_rel[-1]))   # never past what was recorded
```
Output timeline is `times = np.arange(lo, hi, 1.0/fps)` — a **fixed grid**. Each output frame takes the **nearest** recorded frame and the **nearest** recorded pose; nothing is interpolated, "because an interpolated pose is a pose nobody measured" (`collab_video.py:41-47`).

**Writer** (`collab_video.py:358-364`): `cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))`, raising if `not w.isOpened()`.

**`video_real.mp4`** (`make_real_video`, `collab_video.py:382-427`). The Nikon frame, cropped to `picture_rect`, letterboxed into the top `CANVAS_H - STRIP_H = 860` px with a 20 px margin (`s = min((box_h-20)/h, (CANVAS_W-20)/w)`), the force strip pasted opaque into the bottom 220 px. Banners at `(x0+14, y0+44)`.

**`video_real_visualizer.mp4`** (`make_visualizer_video`, `collab_video.py:429-522`). Recorded poses driven through the merged MuJoCo scene, `mj_forward` only — **physics off**. Renamed from `video_sim.mp4` because the old name claimed something false. Key mechanics:
- offscreen buffer is raised by string-patching the MJCF: `_raise_offbuffer` replaces `offwidth="1280" offheight="720"` with `1920`/`1080` (`collab_video.py:373-375`)
- `mujoco.Renderer(model, CANVAS_H, CANVAS_W)` — note the **(height, width)** argument order
- `canvas = np.ascontiguousarray(renderer.render()[:, :, ::-1])` — MuJoCo gives RGB, cv2 wants BGR
- real camera as **picture-in-picture**, `pip_scale = 0.36` (so 691 px wide), at `(16, 16)` with a 2 px `(200,200,200)` border
- the force strip is **alpha-blended** here, not pasted: `0.25 * roi + 0.75 * strip`
- Gen3 dimmed to `HELD_ALPHA = 0.55` when its pose was not recorded (`collab_replay.py:117-121`) — *"a still arm drawn at full opacity beside a moving one reads as 'the Gen3 held still', which is a measurement this recording did not make."*

**`video_mujoco.mp4`** (`make_mujoco_video` → `render_result_video`, `collab_video.py:529-628`). This is the twin's *prediction*: `EX.run_condition(name, overrides=ov, record_qpos=True)` re-rolls the same condition with the run's recorded overrides (`vent_delay_s`, `release_ramp_s`), and the video is rendered from the `qpos` the physics actually stepped. **Its zero is the SIM's own collision instant** — `z0 = float(res.contact_s)` — "aligning it to the rig's would claim a synchrony nobody measured" (`collab_video.py:539-542`).

`render_result_video` is deliberately exposed as a seam so a sim-only caller (the tuner, a sweep) renders a result it already holds instead of paying for a second identical simulation, and its "frame layout, camera, force strip and banners are byte-for-byte the run-dir edit's" (`collab_video.py:556-568`). It takes `before_s` to widen the leader (used for orbits).

**The force strip** (`ForceStrip`, `collab_video.py:236-330`). Drawn **once** with matplotlib Agg at exact pixel size (`figsize=(w/100, h/100), dpi=100`), axes at `fig.add_axes((0.045, 0.22, 0.945, 0.70))`, background `#181818`, trace `#ff5a5a`. Two critical details:

1. **Downsampling by per-bin extremes, not striding** — otherwise a 50 ms impact spike at 10 kHz vanishes when drawn 1900 px wide:
```python
bins = np.linspace(self.lo, self.hi, 2 * self.w + 1)
which = np.clip(np.digitize(t, bins) - 1, 0, 2 * self.w - 1)
fmax = np.full(2 * self.w, -np.inf); fmin = np.full(2 * self.w, np.inf)
np.maximum.at(fmax, which, f); np.minimum.at(fmin, which, f)
ax.fill_between(tmid[ok], fmin[ok], fmax[ok], color="#ff5a5a", alpha=0.95, lw=0)
```
The `-inf`/`+inf` init is load-bearing: `np.maximum(nan, x)` is `nan`, so a NaN-initialised bin never fills and the whole band vanishes (`collab_video.py:293-295`).

2. **`fmag_at` returns NaN outside the trace, not a clamped end value** (`collab_video.py:257-269`) — `np.interp` clamps, and on the simulated gauge that showed the last pre-withdrawal load as if still being measured seconds later.

Per frame, the strip base is copied and a white cursor line drawn at
`x = ax_x0 + (rel - lo)/(hi - lo) * (ax_x1 - ax_x0)`, with `"t %+7.3f s    |F| %6.2f"` overlaid.

**Banners** (`_banners`, `collab_video.py:335-355`) — this is what `side_by_side --verify` looks for in the pixels:
- red `(60,60,220)` filled box, `330 × 54` px at `(x, y)`, text "COLLISION" at `(x+24, y+38)`, `FONT_HERSHEY_SIMPLEX 1.0`, thickness 2, while `0.0 <= rel - contact_rel <= 0.5`
- purple `(160,80,160)` box, `460 × 54`, "DEFLATE + RETRACT", from `rel >= vent_rel`; dims to `(90,60,90)` after 1.5 s
- in the two MuJoCo edits the banner origin is `(CANVAS_W - 500, 24)`

**`make_videos`** (`collab_video.py:638-657`) writes into `<run_dir>/videos/` as `video_real.mp4`, `video_real_visualizer.mp4`, `video_mujoco.mp4`.

### 2.2 Real camera capture — `scene_camera.py`

Hardware: **Nikon Z30 → HDMI → Elgato Cam Link 4K → UVC, 1920×1080 @ ~59.94 fps**. This replaced a Hercules webcam that managed one frame per 67 ms through a ~50 ms impact.

Four constants that are measurements, not preferences:

```python
DEFAULT_INDEX = 0                            # scene_camera.py:65
DEFAULT_WIDTH = DEFAULT_HEIGHT = DEFAULT_FPS = None   # :75-77
_WINDOWS_BACKEND = "CAP_MSMF"                # :84
CAMLINK_PICTURE_RECT = (264, 28, 1656, 960)  # :98
QUEUE_FRAMES = 64                            # :107
```

- **Do not `cap.set()` the mode.** Forcing 1920×1080@59.94 *breaks* the Cam Link: the renegotiated stream runs 60 fps for ~3 s then throttles to a steady ~18, reproducibly (`scene_camera.py:67-77`). Untouched, it held 60 fps for 15 s while a 1080p mp4v encode kept pace on a second thread.
- **DirectShow cannot capture by index in this OpenCV build** — `VideoCapture(i, CAP_DSHOW)` warns and silently falls through. Media Foundation is used (`scene_camera.py:79-84`).
- `CAMLINK_PICTURE_RECT` is the Z30's live-view picture inside the HUD rim: a **3:2 inset, 1392×932**, measured 2026-08-23 by thresholding the rim. **The crop is deliberately NOT applied at record time** — the recording is the evidence; the compositor crops (`scene_camera.py:41-46`, `88-98`).
- The hand-off between grab and encode is a plain `collections.deque`, **not** a `queue.Queue`: the Queue's lock hand-off dragged the grab thread from 60 fps to 18 by GIL scheduling (`scene_camera.py:100-108`).

**Threading.** `_grab_run` (`scene_camera.py:252-281`) reads then immediately stamps `t = time.perf_counter()`, appends `("frame", seg, frame, t)`. On queue overflow it *drops and counts* and pops the stamp back off so stamps and frames stay in lockstep. `_encode_run` (`scene_camera.py:283-325`) writes into a `cv2.VideoWriter(..., "mp4v", self._fps_nominal, (w,h))` live — the hold-in-RAM design would need ~11 GB for a 30 s 1080p 60 fps run.

**Segmentation.** `cut()` (`scene_camera.py:357-370`) discards everything so far and starts a fresh segment. The trial loop calls it as the trial starts (`collab_hardware.py:883`), so `scene.mp4` is the trial and not the minutes of Gen3 placement in front of it. `max_seconds = 120.0` caps one segment, keeping its **head** (the strike is early).

**Save** (`scene_camera.py:380-427`) writes three files beside each other:
- `<path>` — the mp4 (moved by `os.replace`, hence `spool_dir=self.root` so the rename never crosses a filesystem)
- `<stem>.stamps.npy` — per-frame `perf_counter` float64 array
- `<stem>.meta.json` — `{index, requested, camera{width,height,fps_reported,fourcc,backend}, picture_rect, dropped_frames, capped_frames, grab_failures, frames, measured_fps, t_first_perf_s, t_last_perf_s, written, path}`

**Playback side.** `FrameReader` (`collab_video.py:180-224`) reads forward once, holds exactly one decoded frame, never seeks backwards, and picks the *nearest* stamp:
```python
want = int(np.searchsorted(self.stamps, rel))
want = min(max(want, 0), len(self.stamps) - 1)
if want > 0 and (abs(self.stamps[want-1] - rel) < abs(self.stamps[want] - rel)):
    want -= 1
```
The crop is applied here: `return self.frame[y0:y1, x0:x1]`.

### 2.3 Layer 2 — `side_by_side.py`: the composite

CLI (`side_by_side.py:1-9`):
```
python UMArm_COLLAB/side_by_side.py \
  --old .../video_mujoco_ref_mid_if2.mp4 --new .../video_mujoco_stroke_if2.mp4 \
  --old-ledger .../ledger.json:ref_mid_if2 --new-ledger .../ledger.json:stroke_if2 \
  --out .../side_by_side_swing_if2.mp4 --verify
```

Geometry (`side_by_side.py:71-118`): inputs are 1920×1080@60; each is downscaled to `PANE_W, PANE_H = 960, 540` (a clean 2:1 box filter, `cv2.INTER_AREA`); output is **1920×540**, with a `DIVIDER_W = 2` px `(20,20,20)` rule down the seam so the pair reads as two shots rather than one wide one.

**The alignment is derived arithmetic, not pixel search.** `contact_frame_for_row` (`side_by_side.py:223-238`) reproduces `render_result_video`'s own window exactly:

```python
contact_s = float(row["contact_s"])
lead_s = float(lead_for_row(row) if lead is None else lead)
lo = max(-lead_s, float(t_first) - contact_s)
return int(round((0.0 - lo) * float(fps)))
```

with `SWING_LEAD_S = 3.5` and `ORBIT_LEAD_S = 14.0` (mirrored from `collab_video.BEFORE_S` and `refined_campaign.ORBIT_VIDEO_BEFORE_S` rather than imported, to avoid pulling mujoco/matplotlib in to place a caption). For a run that collides later than its own lead this gives **frame 210** (swing) and **frame 840** (orbit) — but those are *the arithmetic's answer*, not constants: an orbit that strikes at 9 s opens at its own first sample and lands elsewhere entirely.

`row_is_orbit` (`side_by_side.py:196-215`) keys off the **presence** of an `"orbit"` vs `"swing"` block in the ledger row, deliberately *not* `tune.drive_name` (KMPPI circle rows keep `slingshot` there for catalog validity).

**Ledger plumbing.** `split_ledger_spec` (`side_by_side.py:252-265`) splits `"path/ledger.json:row"` on the **last** colon and only when the tail is a bare name — otherwise `C:/x/ledger.json` loses its drive letter, "which on this box is every path."

**Frame counting is verified, not trusted.** `probe_video` (`side_by_side.py:159-191`) grabs every frame (`cap.grab()` without `retrieve`, skipping colour conversion, ~0.1 s for a 540-frame 1080p file) rather than trusting `CAP_PROP_FRAME_COUNT`.

**The plan** (`plan_alignment`, `side_by_side.py:411-429`):
```python
contact = max(int(contact_old), int(contact_new))
tail = max(int(n_old) - int(contact_old), int(n_new) - int(contact_new))
return Alignment(pad_old=contact - contact_old, pad_new=contact - contact_new,
                 contact_frame=contact, n_out=contact + tail)
```
The output's collision is the **later** of the two, so neither leader is cut. The shorter side is padded by **freezing its own first or last frame — never by resampling time** — via `source_index(j, pad, n) = min(max(j - pad, 0), n - 1)`; the clamp *is* the freeze. A held frame is **labelled "held frame"** in its own pane (`draw_banner`, `side_by_side.py:473-487`), "since a frozen render and a stalled one look alike."

**Compositing** (`compose`, `side_by_side.py:515-604`). Both clips are read forward once, one decoded frame each in memory — a 1170-frame 1080p pair costs two sequential decodes rather than ~7 GB. Banner: `FONT_HERSHEY_SIMPLEX`, title at `(16, 34)` scale 0.62 thickness 2 in `(235,235,235)`, row name at `(16, 60)` scale 0.50 in `(150,150,150)`, held marker at `(16, 84)` — each over a 1 px `(25,25,25)` shadow, because the MuJoCo background behind the top-left corner is a bright grey that swallows a thin light glyph. `fit_scale` shrinks a caption that would overrun its pane. **ASCII only** — the Hershey fonts cv2 ships cannot draw an em dash, and a missing glyph renders as a box; hence `LABEL_OLD_SWING = "OLD - PID windup + shaped release"`.

Writer: `cv2.VideoWriter(path, fourcc(*"mp4v"), fps, (1920, 540))`.

**Verification in the pixels** (`contact_flash_frames`, `side_by_side.py:612-679`). Independent of the arithmetic: scan each pane's banner box for the red flash,
```python
red = ((box[:,:,2] > 140) & (box[:,:,1] < 110) & (box[:,:,0] < 110))
if int(red.sum()) >= min_pixels: hits[i].append(j)
```
The purple vent banner sits in the same corner but has blue channel 160, which the mask excludes. The box is located by fractions of the source canvas, `_FLASH_BOX_FRAC = ((1920-500)/1920, 24/1080, (1920-500+330)/1920, 78/1080)` (`side_by_side.py:130-133`).

**The one-frame caveat, documented and handled.** `np.arange(lo, hi, 1/60)` accumulates its step in floating point, so the sample meant to be exactly 0.0 misses by a few × 1e-16, and `_banners`' `0 <= rel` test rejects a negative miss. Measured on the 08-31 baselines: a swing (`lo = -3.5`) flashes at **211** for a derived 210; an orbit (`lo = -14.0`) flashes at **840** for a derived 840. The offset is a property of `lo`, so both clips of a like-with-like pairing inherit the same one and the flashes coincide. `--verify` reports `spread` rather than hiding it, and `spread == 1` is reported as "aligned to within one frame (16.7 ms)".

The report dict (returned and optionally written by `--json`) carries `out, n_frames, width, height, fps, contact_frame, pad_old, pad_new, old{...}, new{...}, warnings[]` and, with `--verify`, `verify{pane_frames, first, spread, aligned}`.

### 2.4 `UMArm_SIM/bounce_demo.py` — the GIF/plot path

A different, much lighter deliverable: three arms (`nominal`, `no_damping`, optional `legacy_overdamped`) poked and released. Schedule `SETTLE_S = 4.0`, `PUSH_S = 0.4`, `RING_S = 8.0` (×2.5 for `no_damping`), sample grid `SAMPLE_S = 0.005`, setpoints held at the campaigns' `TICK_S = 0.05` (20 Hz).

`render_gif` (`bounce_demo.py:139-186`): `mujoco.Renderer(arm.model, height=360, width=480)`, `fps = 15`, camera `lookat=(0.22,-0.44,0.55)`, `distance=1.6`, `azimuth=120.0`, `elevation=-15.0`, capture starting 1 s before the push. Written with **PIL only** — `frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0, optimize=True)`. Note it advances a *fresh* arm: "replaying two runs on one arm chains state."

Outputs: `bounce_demo.png` (dpi 130), `bounce_demo_metrics.json`, `bounce_<case>.gif`.

### 2.5 `refined_campaign.py` — the ledger the composite reads

`run_campaign` (`refined_campaign.py:186-234`) rolls each row with `record_qpos=True`, writes `ledger.json` **after every row** (crash-safe), and calls `CV.render_result_video(res, f"videos/video_mujoco_{name}.mp4", before_s=ORBIT_VIDEO_BEFORE_S if orbit else None)`. `ORBIT_VIDEO_BEFORE_S = 14.0` (`refined_campaign.py:44-47`).

`ledger_row` (`refined_campaign.py:110-184`) is the JSON schema `side_by_side` consumes. The load-bearing key is `contact_s`; also `vent_s`, `peak_contact_n`, `face_speed_at_contact_mps`, `closing_speed_at_contact_mps`, `worst_clearance_mm`, `warnings[]`, a `tune{}` block of 18 condition parameters, and exactly one of `swing{}`+`hold{}` or `orbit{}`. `_json_safe` (`refined_campaign.py:172-183`) coerces numpy types and maps non-finite floats to `null`.

---

## 3. The CAN-arm collection JSONL schema and its producer

Doc: `C:/ESP/ESP_Projects/UMArm_koopman_compliance_control_espproject/digital_twin/data_schema.md`.
Producer: `C:/ESP/ESP_Projects/UMArm_koopman_compliance_control_espproject/legacy_host/host/pc_backend/src/main.cpp`.

### 3.1 Commands

`start_data_collection` (`main.cpp:2375-2390`) takes `duration_min` (number, must be > 0), sets `collection_start_requested_`, and calls `start_loop(0)`. `stop_data_collection` (`main.cpp:2392-2395`) just sets `collection_stop_requested_`. Both emit `{"type":"collection","state":...}` lines.

Session directory: `<cwd>/real_system_data_collection/session_<timestamp_for_filename()>` (`main.cpp:2652-2653`).

### 3.2 `metadata.json` — written once at start

`write_collection_metadata_locked` (`main.cpp:2608-2647`). Exact emitted fields, in order:

`schema_version` (1), `session_tag`, `created_at_unix_s` (6 dp), `sample_rate_hz` (= `options_.rate_hz`), `duration_s`, `checkpoint_s`, `deflate_s`, `sync_semantics` (prose string, verbatim: *"runtime table sent before DLC-0 sync; firmware latches pressure ADC and promotes pending target on sync; compact reply pressure is latched ADC at sync"*), `state_fields` `["q","qdot","pressure_adc"]`, `state_dimension` **48**, `input_fields` `["target_adc"]`, `input_dimension` **24**, `pressure_units` `"adc_counts"`, `target_units` `"adc_counts"`, `selected_ids` (24 hex strings), `adc_ranges`, `pressure_calibration_source`, `pressure_calibration_default` (bool), `actuator_pairs`, `target_generation_pressure_model` (prose), `single_actuator_limit_psi`, `pair_sum_limit_psi`, `single_actuator_limit_normalized`, `pair_sum_limit_normalized`, `slew_psi_per_s_reference`, `adc_range_reference_psi`, `can_protocol`, `can_order`, `rx_window_frac`, `mocap_live`, `mocap_sim`, `mocap_server`, `mocap_local`, `mocap_rigid_ids`.

Envelope constants (`main.cpp:70-77`):
```cpp
constexpr double kCollectionCheckpointS = 60.0;
constexpr double kCollectionDeflateS    = 2.0;
constexpr double kCollectionSlewPsiPerS = 15.0;
constexpr double kCollectionAdcRangePsi = 40.0;
constexpr double kCollectionSingleLimitPsi  = 30.0;
constexpr double kCollectionPairSumLimitPsi = 35.0;
constexpr double kCollectionSingleLimitNormalized  = 30.0/40.0;  // 0.75
constexpr double kCollectionPairSumLimitNormalized = 35.0/40.0;  // 0.875
```

The 12 antagonistic pairs (doc §"verbatim", from `main.cpp:929-936`) — **not** a naive `i, i+4`:
```
(0x102,0x106) (0x104,0x108) (0x103,0x107) (0x105,0x101)
(0x10A,0x10C) (0x109,0x10B) (0x110,0x10E) (0x10D,0x10F)
(0x114,0x112) (0x111,0x113) (0x116,0x118) (0x115,0x117)
```

### 3.3 `manifest.json` — atomic-rename chunk index

`write_collection_manifest_locked` (`main.cpp:2541-2574`): written to `manifest.json.tmp`, then `remove` + `rename` — so a reader never sees a half-written index. Fields: `schema_version`, `session_tag`, `session_dir`, `state_dimension` 48, `input_dimension` 24, `total_samples`, `checkpoint_count`, `active` (bool = `collection_active_ || collection_deflating_`), `chunks[]` of `{path, reason, samples, start_cycle, end_cycle, start_time_s, end_time_s}` with `reason ∈ {checkpoint, complete, stop}`.

Chunk files are opened as `samples_chunk_NNNN_open.jsonl.tmp` (`main.cpp:2530-2539`) and renamed to `samples_chunk_NNNN_<reason>_<ts>.jsonl` on close (`main.cpp:2587`). Rollover at `kCollectionCheckpointS = 60 s` measured on `can_sync_time_s` (`main.cpp:2864-2867`). Flush every 15 samples (`main.cpp:2863`).

### 3.4 The per-cycle JSON object — exact fields as emitted

`write_collection_sample_locked` (`main.cpp:2807-2868`). One line per 150 Hz cycle, ~4.1 kB, 9000 lines per 60 s chunk. Field-by-field, in emission order:

| field | C++ source | type / unit |
|---|---|---|
| `schema_version` | literal | int, 1 |
| `cycle` | `state.cycle` | uint64 |
| `phase` | `collection_phase_locked()` | `"collecting"` \| `"deflating"` \| `"inactive"` (`main.cpp:2522-2527`) |
| `timestamp_unix_s` | `unix_seconds_now()` | double, 6 dp, **wall clock** |
| `timestamp_s` | `state.can_sync_time_s` | double, 6 dp — *identical to `can_sync_time_s`* |
| `cycle_start_time_s` | `state.cycle_start_time_s` | double, 6 dp, backend clock |
| `can_sync_time_s` | `state.can_sync_time_s` | double, 6 dp — **the alignment anchor** |
| `jitter_ms` | argument | double, 3 dp |
| `ids` | `selected_id_strings_locked()` | 24 hex strings |
| `robot_state.q` | `state.joint_current_estimate.theta` | 12 doubles, **radians**, 9 dp |
| `robot_state.qdot` | `...theta_dot` | 12 doubles, **rad/s**, 9 dp |
| `robot_state.pressure_adc` | `state.pressure_adc_filtered` | 24 uint16, **raw ADC counts** |
| `input.target_adc` | `state.target_next_sync` | 24 uint16, raw ADC counts |
| `actuator_status` | | 24 uint8 |
| `actuator_stale` | | 24 bool |
| `control_next_sync` | | 24 uint8 |
| `actuator_missed_total` | `BoardState::missed` | 24 uint64 |
| `actuator_reply_latency_ms` | `board_latency_values_locked()` | 24 doubles, 3 dp |
| `cycle_responded`, `cycle_expected` | | ints |
| `total_missed`, `unexpected_replies`, `duplicate_replies` | process-wide counters | ints |
| `joint_current_valid`, `joint_current_extrapolated` | | bool |
| `joint_current_source_error_ms`, `joint_current_extrapolation_ms` | | double, 3 dp |
| `mocap.valid`, `.stale` | | bool |
| `mocap.frame` | | int |
| `mocap.timestamp_s`, `.raw_timestamp_s`, `.received_s` | | double, 6 dp |
| `mocap.latency_ms`, `.age_ms`, `.timestamp_offset_ms`, `.frame_rate_hz` | | double, 3 dp |
| `mocap.frame_drop_count`, `.clock_sample_count`, `.clock_update_count`, `.body_count` | | ints |
| `mocap.body_ids` | `json_escape(mocap.body_ids)` | **JSON string**, not array |
| `mocap.body_centers` | | 18 doubles (6×3), 9 dp |
| `mocap.body_rotations` | | 54 doubles (6×9), 9 dp |
| `mocap.body_quaternions` | | 24 doubles (6×4), 9 dp |
| `mocap.body_points`, `.body_poses` | `json_escape(...)` | **JSON strings**, not null |

### 3.5 Discrepancies between the doc and the producer

These matter because the doc reads as a spec but the C++ is what actually writes:

1. **`board_type` is NOT emitted.** `grep -c "q_stale\|board_type" main.cpp` → **0**. It is the doc's proposed addition (`data_schema.md` §"The addition: `board_type`"). You must add it — in `metadata.json` (24 entries aligned to `selected_ids`, from byte 3 of `MSG_FW_VERSION` in reply to `CMD_GET_FW_VERSION`) and, per the doc, also per-cycle.
2. **`q_stale` is NOT emitted.** The doc says record both `mocap.stale` and `q_stale` and prefer `q_stale`. Only `mocap.stale` exists.
3. **`body_ids`, `body_points`, `body_poses` are emitted as escaped JSON *strings***, not arrays/null as the doc's example shows.
4. **No `timing_summary_<phase>.json` producer exists anywhere in the ESP project.** `grep -rn "timing_summary"` over the whole tree returns exactly one hit — `digital_twin/data_schema.md:163`. The legacy hour-run numbers quoted there (6.66666 / 6.666 / 4.306 / 25.069 ms, 0 missing cycles) came from a session directory, not from this code path. If you want that file you must write it.
5. `pressure_calibration_source`, `pressure_calibration_default`, `target_units`, `target_generation_pressure_model`, `single_actuator_limit_normalized`, `pair_sum_limit_normalized` are emitted but absent from the doc's metadata table.

---

## 4. Excitation-signal design

Two distinct bodies of work. The ESP host's is a simple bounded random walk; the RS485 Koopman work is a full 13-modality mixer with four generator revisions plus a separate **duty-space** family that maps almost exactly onto your bottom-16 PWM solenoids.

### 4.1 The ESP host's own generator (`main.cpp:2705-2730`)

A per-actuator **reflected, damped random walk in normalized pressure**, run every 150 Hz cycle:

```cpp
const double max_step = std::max(0.0001, kCollectionSlewPsiPerS
    / (kCollectionAdcRangePsi * static_cast<double>(std::max(1, options_.rate_hz))));
std::uniform_real_distribution<double> acceleration_dist(-max_step * 0.18, max_step * 0.18);
for (auto& [id, board] : boards_) {
    double velocity = collection_velocity_[id] * 0.985 + acceleration_dist(collection_rng_);
    velocity = std::max(-max_step, std::min(max_step, velocity));
    double normalized = collection_norm_[id] + velocity;
    if (normalized < 0.0) { normalized = -normalized; velocity = std::abs(velocity) * 0.35; }
    else if (normalized > kCollectionSingleLimitNormalized) {
        normalized = 2.0 * kCollectionSingleLimitNormalized - normalized;
        velocity = -std::abs(velocity) * 0.35;
    }
    collection_norm_[id] = std::clamp(normalized, 0.0, kCollectionSingleLimitNormalized);
    collection_velocity_[id] = velocity;
}
apply_collection_pair_projection_locked();
```

Numbers: `max_step = 15.0 / (40.0 * 150) = 0.0025` normalized per tick (0.1 psi/tick, 15 psi/s ceiling). Acceleration draw is ±18 % of `max_step`. Velocity decay **0.985 per tick** (≈100-tick / 0.67 s memory). Reflection at both bounds with a **0.35 restitution** on velocity. RNG is `std::mt19937 collection_rng_{std::random_device{}()}` (`main.cpp:3572`) — **unseeded and unrecorded**, so a session is not reproducible.

Pair projection (`main.cpp:2688-2702`): after the walk, any pair whose normalized sum exceeds `0.875` is **scaled down proportionally**, both sides:
```cpp
const double scale_factor = kCollectionPairSumLimitNormalized / sum;
left->second *= scale_factor;  right->second *= scale_factor;
```
Then `target = min_adc + norm * (max_adc - min_adc)`, `clamp_pressure`d to 12-bit `[0, 4095]`.

Deflate tail: `begin_collection_deflate_locked` (`main.cpp:2732-2743`) sets every target to 0 with outputs still enabled for `kCollectionDeflateS = 2.0` s, then `finish_collection_locked` disables outputs and finalizes.

**Nothing here is a chirp, a PRBS, a step, or a multisine.** It is one modality.

### 4.2 The RS485 Koopman mixer — `UMArm_KoopmanMPPI/excitation.py` (1529 L)

This is the mature design and the one worth porting. Same shape as your arm: **24 nodes, 12 antagonistic pairs** (`koopman_constants.py:45-53`), `CONTROL_HZ = K.CONTROL_TICK_HZ` (160 Hz there).

**Coordinates.** Everything is drawn in pair space `(m, d)`: `m = (agonist + antagonist)/2` co-contraction, `d = agonist - antagonist` differential. `md_to_pair_psi(m, d) = (m + d/2, m - d/2)` (`excitation.py:203-205`). The envelope is true **by construction of the draw**, not discovered by a refusal (`clamp_md`, `excitation.py:190-199`):
```python
m = np.clip(m, prm.p_min, prm.m_max)
lim = np.minimum(2.0 * (m - prm.p_min), 2.0 * (prm.p_max - m))
return m, np.clip(d, -lim, lim)
```
with `p_min = LIMITS.p_min`, `m_max = sum_max/2 = 17.5 psi`, `p_max = 35 psi`.

**Stream structure** (`build_stream_psi`, `excitation.py:833-899`): a warmup (linear ramp floor→`P0_BASELINE_PSI` over `n_warm/2`, then hold; `warmup_s = 5.0`, first second dropped as burn-in), then a concatenation of segments of drawn length `U(seg_lo_s=2.0, seg_hi_s=10.0) × CONTROL_HZ`. Each segment is with probability `regime_fraction()` a **global regime** (all 12 pairs coordinated) and otherwise a **mixer** segment (independent modality block per pair, block boundaries deliberately unaligned across pairs). Returns `(n_ticks, 24)` psi in node-id order.

**The seven per-pair modality blocks** (`_BLOCKS`, `excitation.py:315-319`), default weights from `ExcitationParams.weights()` (`excitation.py:143-148`):

| block | weight | parameters (all `U[...]` unless noted) |
|---|---|---|
| `step_hold` | 3.0 | hold `U(0.4, 2.0)` s; amp `U(-1,1) × d_max(m0)` (`:236-246`) |
| `multisine` | 2.0 | `n_tones = randint(3,7)`; **log-uniform** freqs over `[f_lo=0.05, f_hi=4.0]` Hz; random phases; normalized to unit peak; amp `U(0.2,1.0) × d_max` (`:249-258`) |
| `chirp` | 1.5 | **linear sweep** `f_lo=0.05 → f1 = U(1.0, 4.0)` Hz over the block: `phase = 2π(f_lo·t + ½(f1−f_lo)t²/T)`; amp `U(0.3,1.0) × d_max` (`:261-269`) |
| `prbs` | 1.5 | ±amp with dwell `U(0.3, 1.2)` s; amp `U(0.3,1.0) × d_max` (`:272-282`) |
| `m_walk` | 1.5 | Brownian on `m`, rate 3.0 psi/√s, reflecting; `d` held (`:294-297`) |
| `md_walk` | 1.5 | Brownian on both, rates 2.0 and 6.0 psi/√s (`:300-304`) |
| `deadband_dither` | 1.0 | sine `f = U(0.2, 2.0)` Hz, amp `U(0.2, 0.9) × pa_to_psi(K.MARGIN_PA)` — **inside the bang-bang band, so mostly no valve switches; the model must see that nothing happens there** (`:307-312`) |

`_walk` (`excitation.py:285-292`) uses **Brownian scaling** — per-step sd `rate·√dt`, so displacement over T seconds is `~rate·√T` psi — with reflection at the bounds. The comment records the bug it fixes: the first cut used `rate·dt` and every "walk" moved 0.1 psi total.

**The global regimes**, four tables by version:

- **v1** (`_REGIMES`, `:452-458`), `regime_frac = 0.35`: `dwell_hold` 2.0, `low_pressure_walk` 1.5, `cocontraction_sweep` 1.0, `settle_to_rest` 1.0, `p0_regime_dwell` 2.5.
- **v2** (default, `_REGIMES_V2`, `:474-481`), `regime_frac = 0.50`: adds `tip_orbit` at weight 3.0 (≈15 % of stream time) — twelve differentials locked to one frequency in `TIP_ORBIT_F_HZ = (0.05, 0.5)` Hz at held co-contraction `TIP_ORBIT_M_PSI = (5.0, 17.0)` psi, amplitude ramped in over `TIP_ORBIT_RAMP_S = (1.0, 2.0)` s as `clip(t/ramp,0,1)`, per-pair phase, and a coin deciding one shared frequency vs one per 4-joint arm segment. Also re-weights `dwell_hold` to draw from `[6.0, m_max]` and moves the sweep endpoints up (`SWEEP_V2_LO_PSI = (1.0, 6.0)`, `SWEEP_V2_HI_PSI = (13.0, m_max)`).
- **v3** (`:622`), `regime_frac = 0.72`: adds `dump_release` — shaped-release cycles. Bounds (`:494-508`): `DUMP_WIND_PSI = (15,32)`, `DUMP_FLOOR_PSI_HI = 8.0`, `DUMP_RAMP_S = (0.2,0.8)`, `DUMP_DRIVE_DELAY_S = (0.0,0.6)`, `DUMP_DRIVE_RAMP_S = (0.2,0.8)`, `DUMP_DRIVE_PSI = (10,32)`, `DUMP_HOLD_S = (0.5,2.0)`, `DUMP_FOLLOW_S = (0.3,1.0)`, `DUMP_RECOVER_S = (0.5,1.5)`, `DUMP_RECOVER_M_PSI = (4,12)`, `DUMP_W_MIN = 0.3`. The dump uses `_cosblend` — a raised cosine `0.5 - 0.5cos(πs/ramp)`, deliberately the same waveform family the bench release commands. The rising drive side is capped at the pair's *instantaneous* headroom `sum_max - wound(t) - 1`.
- **v4** (`:798`): adds `pump_dump` — resonant pump cycles at a period drawn to bracket the arm's measured **1.83 Hz** mode (0.4-0.7 s), amplitude grown over 2-6 cycles, released at the wound extreme with a *fast* raised cosine (0.15-0.6 s) to a 1 psi floor. A coin also draws plain hot dumps (wind 20-33 psi, no pump).

**Version is a switch that travels in the dataset meta**, because `verify_dataset`'s gate-4 re-simulation replays stored commands and only reproduces them if it draws the same revision (`excitation.py:172-186`). `regime_table()` reads through the params, not a module global, so a v1 draw stays bit-identical: it gets the original dict in the original insertion order, so `rng.choice` consumes the same number of variates.

**Wire encoding** (`to_wire`, `excitation.py:954-957`) and a **refuse-not-clamp** check at the wire (`assert_envelope`, `:960-970`) that raises rather than silently clipping.

### 4.3 The duty-space family — directly relevant to your bottom-16 PWM solenoids

`excitation.py:1031-1520`. Coordinates are `(c, s)`: `c` common mode (fill/bleed together), `s` differential, mapped by `cs_to_pair_duty` (`:1087-1104`):
```python
ag_all  = np.clip(c + 0.5 * s, -1.0, 1.0)
ant_all = np.clip(c - 0.5 * s, -1.0, 1.0)
```
The clip is the only nonlinearity, and it makes a large `s` a *saturated push-pull* rather than an illegal command.

`DutyExcitationParams` (`:1052-1084`): `c_lo = -0.30`, `c_hi = +0.60` — **asymmetric on purpose**, because fill authority collapses with pressure (+21.7 psi/s at 0 psi, +0.27 at 35) while vent authority grows, so a zero-mean duty stream drives the rack *down* and an open-loop duty dataset would carry almost no pressure. `s_max = 1.6` (past 2.0 would be fully saturated). `regime_frac = 0.45`.

Eight blocks, weights (`:1067-1072`): `duty_step_hold` 3.0, `duty_multisine` 2.0, `duty_chirp` 1.5, `duty_prbs` 1.5, `duty_ramp` 2.0, `duty_dither` 1.0, `duty_cs_walk` 1.5, `duty_pair_corr` 1.5. Three of these are new relative to the pressure family:

- `duty_ramp` (`:1159-1171`) — *"the one input shape whose pressure response separates the valve's own rate limit from the flow net's pressure dependence"*: `c` and `s` each a straight line between two independent draws.
- `duty_dither` (`:1173-1188`) — sine at `f = U(0.2, 2.0)` Hz with amp `U(0.5, 3.0) × DUTY_QUANTUM`, where `DUTY_QUANTUM = 0.08` is one servo period as a fraction of the control tick (0.5 ms of 6.25 ms). `rint(|d| * 12.5)` is zero below half a quantum, so the band alternates between a shut valve and a single pulse — the model must learn most of that band does nothing.
- `duty_pair_corr` (`:1196-1240`) — draws the two node duties jointly from a bivariate normal with correlation `ρ ~ U[-1,1]`, so fill/fill, fill/vent and vent/vent all appear at every magnitude. Pure `c` is `ρ=+1`, pure `s` is `ρ=-1`; this block visits the interior.

**The critical structural difference for a duty lane:** the envelope **cannot** be made true by construction of the draw, because a duty command is a *flow* command. It is enforced by `DutySupervisor` (`:1409+`) at the collector, tick by tick, against the *measured* pressure, by attenuating: a fill is scaled by pair headroom, a vent by height above the floor, each tapering linearly to zero over `DUTY_GUARD_TAPER_PSI = 1.5` psi (~11× the worst per-tick pressure change a fully open valve can produce), with `DUTY_GUARD_MARGIN_PSI = 0.2` held back from each bound. Without the margin a 10 min dataset peaked 0.01 psi past the 35.00 cap; with it, 34.81. Warmup is a hard fill `DUTY_WARMUP_C = 0.8` that the supervisor stops at the P0 baseline.

### 4.4 The two hardware-drive generators (`intercept_drive.py`, `slingshot_drive.py`)

Not system-ID excitations but periodic drives, and both are relevant as signal families.

**`InterceptDrive`** (`intercept_drive.py:157-296`) — open-loop fixed-period quadrature. Per universal joint group: `psi(axis1) = mid ± amp·cos(φ + phase)`, `psi(axis2) = mid ∓ amp·sin(φ + phase)`, `φ = 2π(t−t0)/period`. Six groups × `(mid, amp, phase)` plus a **second harmonic** `(amp2, phase2, sense2)` per group to cancel the linkage's systematic distortion (a co-rotating 2nd harmonic makes the circle three-cornered, a counter-rotating one elliptical). Spin-up: rate ramps `spinup_frac = 0.35 → 1` over `spinup_s` as a raised cosine, with `phase_at` integrating that profile in closed form; amplitude ramps over the same window, so the path spirals outward. Measured repeatability **1.2-1.9 mm** pass-to-pass against the joint PID's 12.25 mm RMS residual.

**`SlingshotDrive`** (`slingshot_drive.py:101-300`) — closed-loop resonant pump ("a child on a swing"). Reads one u-joint's `(q[j1], q[j2])` as a 2-vector, and commands the quadrature at that measured angle **plus a fixed lead**. The phase is carried by a **PLL, not differentiated** (`_advance`, `:222-251`): it free-runs at `seed_period_s = 1.5` while the reference pair is under `lock_amp_deg = 1.2°`, then corrects with `pll_kp = 8.0` rad/s per rad and `pll_ki = 4.0` rad/s² per rad, clamped into `period_limits = (1.0, 4.0)` s/rev. Waveform sharpening `sat(shape · cos)` — 1.0 is a sine, above flattens toward the square wave that puts most energy per cycle through a bang-bang valve. Amplitude ramp `spinup_frac = 0.35 → 1` over `spinup_s = 6.0` s. Measured: the shipped drive settles at 1.312 s/rev and 1.485 m/s peak face speed against the open-loop `tip` drive's 3.218 s/rev and 0.409 m/s.

**`CHIRP` in `experiments.py:282-286`** — the SET-4 excitation:
```python
CHIRP: dict[str, float] = dict(f0_hz=0.15, f1_hz=1.2, sweep_s=12.0, tail_s=2.0, psi_min=1.0)
CHIRP_RETRACT = 1.5
```
Linear sweep, `chirp_phase(t) = 2π(f0·t + (f1−f0)t²/(2·sweep))`, continued at `f1` after the window (`experiments.py:2500-2510`). Band is `mid ± amp` spanning `[psi_min, psi_cap]` (`chirp_pressure_band`, `:2494-2497`). Applied to the **six proximal joints only**, x fed `sin` and y fed `cos` — a 90° phased pair, i.e. a circular whirl (`_chirp_psi`, `experiments.py:3545-3564`); the distal joints hold `MIN_PSI_BASE`. The PID is fully bypassed during it. Note the documented envelope compromise: a driven pair sums to `1 + psi_cap` = 36 psi at a 35 psi cap, 1 psi over budget, and the budget takes that psi off the slack side rather than narrowing the band (which would change the excitation amplitude the condition is *defined* by).

---

## 5. Porting notes for the 24-actuator, two-population CAN arm

- **Two plants, one bus** means excitation must be drawn per population. The RS485 pressure family (`build_stream_psi`) maps to your top 8 TLE/DVP boards (setpoint-driven, envelope true by construction). The duty family (`build_duty_stream` + `DutySupervisor`) maps to your bottom 16 legacy 7 mm solenoids (flow-driven, envelope enforced tick-by-tick against measured pressure). Both produce `(n_ticks, 24)`; you will want `(n_ticks, 8)` + `(n_ticks, 16)` or one 24-wide array with a per-column lane flag.
- The RS485 arm is also 24 nodes / 12 pairs, so `PAIRS`, `cs_to_pair_duty`, `clamp_md` and `apply_pair_budget` transfer with only the pair table swapped for the `main.cpp:929-936` one.
- The video pipeline needs no ffmpeg and no external codec: OpenCV `mp4v` throughout, matplotlib Agg for the strip, PIL for GIFs.
- The side-by-side deliverable's alignment depends on exactly two things travelling in a ledger: `contact_s` and the lead the clip was rendered with. Emit both.



## KEY FILES
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_ROBOT_CONTROL/collab_hardware.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/collab_video.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/side_by_side.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/scene_camera.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_KoopmanMPPI/excitation.py
- C:/ESP/ESP_Projects/UMArm_koopman_compliance_control_espproject/legacy_host/host/pc_backend/src/main.cpp
- C:/ESP/ESP_Projects/UMArm_koopman_compliance_control_espproject/digital_twin/data_schema.md
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/ft_daq.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/collab_replay.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/refined_campaign.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/slingshot_drive.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/intercept_drive.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/experiments.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_SIM/bounce_demo.py
- C:/RUNZE_SRC/RS485_VEMA/.claude/worktrees/kmppi-collision/UMArm_COLLAB/protocol.py
- C:/ESP/ESP_Projects/UMArm_koopman_compliance_control_espproject/legacy_host/docs/sync_rules.md
- C:/ESP/ESP_Projects/UMArm_koopman_compliance_control_espproject/legacy_host/docs/mpc_integration.md

## GOTCHAS
- `UMArm_COLLAB/experiments.py` is NOT the hardware collection loop — it is the simulation condition catalog and `TrialEngine`. The real simultaneous-hardware loop (Kinova + ATI + UMArm + mocap + camera in one process) is `UMArm_ROBOT_CONTROL/collab_hardware.py:874` `Campaign.run_condition`. Porting `experiments.py` alone gets you a simulator, not a data-collection rig.
- `data_schema.md` documents `board_type` and `q_stale` as JSONL fields, but `main.cpp` emits NEITHER — `grep -c "q_stale|board_type" main.cpp` returns 0. They are proposals. You must implement them, and `board_type` is load-bearing: a TLE board legitimately sat at 0x114, so the two-population split is NOT recoverable from the id range, and reading a TLE board on the 7 mm scale (754.4 / 56.14 vs 943.75 / 60.78125) is a ~10 % error at the top of the range.
- `timing_summary_<phase>.json` has no producer anywhere in the ESP project — the only hit for that string in the entire tree is `digital_twin/data_schema.md:163`. Do not expect `main.cpp` to write it.
- In the JSONL, `mocap.body_ids`, `mocap.body_points` and `mocap.body_poses` are emitted as escaped JSON STRINGS (`json_escape(...)`), not as arrays or null as the schema doc's example shows (`main.cpp:2856-2860`). A parser written against the doc will fail on real rows.
- `can_sync_time_s` is the alignment anchor, not `timestamp_s` — though in the current writer they are literally the same value (`main.cpp:2817-2820` both emit `state.can_sync_time_s`). `timestamp_unix_s` is wall clock and a DIFFERENT epoch; `cycle_start_time_s` precedes the sync edge. Never mix raw NatNet timestamps with CAN sync timestamps (`sync_rules.md`, Mocap Clock Rule).
- The ESP host's excitation RNG is `std::mt19937{std::random_device{}()}` (`main.cpp:3572`) — unseeded and unrecorded. A collection session is NOT reproducible. The RS485 Koopman pipeline treats the excitation version + seed as data that must travel in the meta so `verify_dataset` gate-4 can re-simulate bit-exactly; port that discipline.
- Do NOT call `cap.set()` on the Elgato Cam Link. Forcing 1920x1080@59.94 breaks it: the renegotiated stream runs 60 fps for ~3 s then throttles to a steady ~18, reproducibly with no other load (`scene_camera.py:67-77`). Leave `DEFAULT_WIDTH/HEIGHT/FPS = None`.
- On Windows use `CAP_MSMF`, not `CAP_DSHOW`. This OpenCV build's DirectShow cannot capture by index at all — it warns and silently falls through (`scene_camera.py:79-84`).
- The grab→encode hand-off must be a plain `collections.deque`, not `queue.Queue`. Measured: the Queue's lock hand-off dragged the grab thread from 60 fps to 18 via GIL scheduling (`scene_camera.py:100-108`).
- `ForceStrip._draw_base` must initialise the per-bin extremes arrays with -inf/+inf, not NaN: `np.maximum(nan, x)` is nan, so a NaN-initialised bin never fills and the ENTIRE force band vanishes silently (`collab_video.py:293-295`).
- `ForceStrip.fmag_at` must return NaN outside the trace, not `np.interp`'s clamped end value — the clamp showed the last pre-withdrawal load as if it were still being measured seconds later (`collab_video.py:257-269`).
- The side-by-side collision frame is DERIVED from `contact_s` and the render lead, never hard-coded. 210 (swing) and 840 (orbit) are the arithmetic's answer only for a run whose contact is later than its own lead; a run that collides earlier opens at its own first sample and lands elsewhere (`side_by_side.py:20-32`, `223-238`).
- The burned COLLISION flash can sit ONE FRAME LATER than the derived contact frame: `np.arange(lo, hi, 1/60)` accumulates step error, so the sample meant to be exactly 0.0 misses by ~1e-16 and `_banners`' `0 <= rel` test rejects a negative miss. Measured: swing flashes at 211 for a derived 210. `--verify` reporting spread==1 is CORRECT, not a bug (`side_by_side.py:39-46`, `641-655`).
- `split_ledger_spec` must split on the LAST colon and only when the tail is a bare name — otherwise `C:/x/ledger.json` loses its drive letter, which on Windows is every path (`side_by_side.py:252-265`).
- Never resample time to align two clips. The shorter side is padded by FREEZING its first/last frame, and the frozen pane must be LABELLED 'held frame' — a frozen render and a stalled one look identical (`side_by_side.py:411-441`, `473-487`).
- `mujoco.Renderer(model, CANVAS_H, CANVAS_W)` takes (height, width) in that order, and `renderer.render()` returns RGB — you must do `[:, :, ::-1]` plus `np.ascontiguousarray` before handing it to cv2 (`collab_video.py:472`, `495`).
- The MJCF's offscreen buffer must be raised or the render silently comes back at 1280x720: `_raise_offbuffer` string-replaces `offwidth="1280" offheight="720"` in the XML (`collab_video.py:373-375`).
- cv2's Hershey fonts cannot draw an em dash or any non-ASCII glyph — a missing glyph renders as a box. All burned-in captions must be pure ASCII (`side_by_side.py:120-124`).
- The gauge bias MUST be captured unloaded, inside the protocol's own bias window, before anything is commanded. A bias captured under load subtracts that load from every later reading and is invisible afterwards (`collab_hardware.py:896-903`, `ft_daq.py:492`).
- A moving Kinova puts real newtons through the ATI cell — measured 2.10 N for a 0.10 m/s advance, 40 % over the 1.5 N trigger. On `collab_20260828_220012` that transient WAS reported as the collision, 0.7 s early, and the vent was timed from it. The fix is a per-run `contact_gate_s` that refuses any report before the plate could physically be there, plus recording the refusals (`collab_hardware.py:105-119`, `954-967`).
- The DAQ input range must be checked UNDER LOAD, not at rest. Unloaded the amplifier sits at -0.055..+0.425 V so ±1 V looks right; the first real strike drove three bridges to the ±1 V rail for 7650 samples (0.77 s of a 2 s contact). It is ±10 V now, the amplifier's own range (`ft_daq.py:133-164`).
- Store RAW volts, not the wrench. Every recording before 2026-08-29 has a wrench computed with a legacy matrix that over-reported Fx/Fy by 11.31x and Fz by 5.87x; `load_block` re-derives from volts under today's calibration by default, and picks the matrix by the RECORDING's own input range (the sensor is dual-range) (`ft_daq.py:648-680`).
- A recorded mocap `q` that is stale must be written as NaN, never forward-filled at record time — a frozen pose is exactly what a controller must not be fed silently (`collab_hardware.py:301-307`). The same goes for `gen3_q`: a seven-zero joint vector is a real and alarming pose of that arm, so absence is NaN (`collab_hardware.py:245-251`).
- `PLATE_STREAM_HZ = 240.0` is a REQUEST that delivers 73 Hz on Windows: `Event.wait`/`time.sleep` are bounded below by the ~15.6 ms system timer tick, so a 4 ms sleep takes 15 ms and the poll misses alternate camera frames. Raising the process-wide timer resolution would fix it and would also change the bus loop's timing (`collab_hardware.py:186-201`).
- Closing speed cannot be derived from the 40 Hz control slice — a derivative across a 0.2 s advance at 40 Hz is three points. The Gen3 pad needs its own listener-driven receiver at the camera rate (`pad_t`/`pad_pos`/`pad_quat`), which is what makes the pad's half of the closing speed a measurement rather than a repeat of the command (`collab_hardware.py:288-295`).
- The vent/abort path must end in SILENCE, not just zeros: `vent_all(6 s)` → `all_off(0.5)` → `quiet(FAILSAFE_SILENCE_S)`, so every node's own firmware failsafe closes its valves independently of the host. Note your bottom 16 legacy boards have NO link-loss timeout at all, per `data_schema.md` — so the host-side vent is the only thing that protects them.
- `_walk` must use Brownian scaling (per-step sd = rate·√dt), not rate·dt. The first cut used the latter and every 'random walk' moved 0.1 psi total — a jittery hold, not a walk (`excitation.py:285-292`).
- A duty/flow command's envelope CANNOT be made safe by construction of the draw the way a pressure setpoint's can. It must be enforced tick-by-tick against measured pressure by a supervisor that attenuates, with a guard taper (1.5 psi) and a margin (0.2 psi) — without the margin a 10 min dataset peaked past the 35.00 psi cap because the measured pressure is a within-tick mean and understates a rising pressure (`excitation.py:1005-1029`, `1409+`).
