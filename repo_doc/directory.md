# Where everything is — UMArm CAN-arm workspace

One line per place. Check here first; fix here in the same change that makes it wrong.

The robot: a ceiling-hung 24-actuator pneumatic UMArm on a 1 Mbit/s CAN bus reached
through a CANable 2.0 slcan dongle (VID:PID 16D0:117E, currently COM58). Top 8
actuators = TLE92464/DVP boards at `0x101–0x108`; lower 16 = legacy 7 mm boards at
`0x109-0x118`. Mocap rigid bodies **2000-2005, verified live 2026-08-21**: the
census sees 500-505 (RS485 arm), 1008 (Kinova) and 2000-2005 (this arm), four
labeled markers each. RS485 sister arm and Kinova Gen3 share the Motive volume
at 192.168.1.100.

Calibrated 2026-08-21, and all three answers live in code: the actuator/axis map
(`UMArm_KINEMATICS/canarm_actuators.py`), the per-plate marker azimuth and the
proximal-joint composition order (`UMArm_MOCAP/canarm_frames.py`), and the link
lengths (`UMArm_KINEMATICS/canarm_params.py`, `MEASURED = True`). fkine now
reproduces the measured u-joint centres to **1.99 mm RMS on held-out multi-joint
poses**; see `hw_tests/report_canarm_axis_2026-08-21.md`. Two hardware faults to
know about: `0x110` leaks from its supply side and reads +5.1 psi at rest, and
`0x104` reads +1.1 psi.

The twin, refined 2026-09-10: the ProMax geometry of Fig. 1C of
`2606.29731v1.pdf` (the paper for this arm), hinges read by name, CMA-ES-fitted
segment masses, and a SIM adapter in the operator GUI. Start at
`hw_tests/report_canarm_twin_refine_2026-09-10.md`.

## Root

| file | what it does |
|---|---|
| `bench_env.py` | the single source of ports/IPs/interpreters: CAN dongle resolved by USB identity, mocap/Kinova IPs, IDF env incantation |
| `canarm_control_gui.py` | **the operator GUI**: TLE controller (all 24 boards) + mocap status strip (sources off/sim/live/**twin**) + **Lock plates** (mints this Motive session's marker locks and swaps in the marker receiver) + a **kinematics line** (live fkine-vs-mocap u-joint centre error) + spawned MuJoCo room viewer. **`SIM - digital twin (no hardware)`** is an always-present adapter (`--sim` preselects it, a Refresh never picks it): `make_backend` builds `SimMaster(physics="process")` from `digital_twin/twin_params` instead of tlelib's `Backend`, the only call that differs; the `twin` source feeds the twin's `q` through the real receiver and the viewer through the shared array. `--self-test` opens no port. KNOWN COST: the 24-board pressure plot holds the real cycle to ~145 Hz (2026-08-20); SIM measures 141.7-143.6 Hz / 99.65 % in the same window. KNOWN GAP: the window does not enforce CONTRACT.md section 8 (bars reach 40 psi, no pair-sum check) on either adapter |
| `requirements.txt` | workspace deps; base Python 3.13 already has all but `python-can` and `cma` (CMA-ES for `digital_twin/mech_fit.py`, 4.4.4 in `.venv`) |
| `.venv/` | workspace venv (3.13 + system site-packages + python-can + cma). `.venv_kinova/` is the protobuf-3.5.1 quarantine — rebuild via `UMArm_KINOVA/setup_env.py` |
| `2606.29731v1.pdf` | the paper describing this exact arm (Zuo et al., arXiv 2606.29731); Fig. 1C/D is the source of the twin's Y / hub / bearing / u-joint geometry. **Untracked** (6.7 MB) |
| `prompts/` | the task prompt behind each work item, date-prefixed |

## Firmware

| path | what is there |
|---|---|
| `firmware/tle/` | TLE board firmware (ESP-IDF project `VEMA_MAX22200`, esp32s3). MINIMAL_BUILD is mandatory on this machine |
| `firmware/legacy/` | legacy 16-board firmware (`Valve_not_embedded_XL` v0.2.1) incl. the load-bearing `sdkconfig` that upstream gitignores |
| `firmware/images/` | prebuilt flashable images with provenance README: `legacy_7mm` (byte-identical to what runs on 0x109–0x118), `legacy_dt`, `tle` |

## Host tooling

| path | what is there |
|---|---|
| `TLE_PCB/` | the ported TLE toolkit: `tlelib/` (proto, slcan canlink, 150 Hz backend, broadcast OTA, usbflash), `VEMA_TLE_controller.py`, `VEMA_TLE_flash.py`, `tools/tle_bench.py` (hardware acceptance), `docs/` (protocol report). `PORTING.md` lists what changed and the five deadliest protocol traps |
| `legacy_host/` | the legacy portable bundle: OTA/diag/probe/calibration tools, C++ 150 Hz `pc_backend` (builds here, self-test OK), `mocap/mocap.py` bridge, the canonical `docs/can_protocol.md`, `calibration.json` |

## Mocap, kinematics, Kinova

| path | what is there |
|---|---|
| `UMArm_MOCAP/` | NatNet receiver with per-instance rigid-body base (`CanArmMocap` = 2000–2005), marker→q math, marker-frame locks, `sim_stream.py` (camera-free synthetic stream), vendored patched NatNet SDK |
| `UMArm_MOCAP/canarm_frames.py` | **the CAN arm's frame convention**: plate frames from the four markers alone, per-plate azimuth `PLATE_AZIMUTH_DEG` and `PROXIMAL_ORDER` both measured; lock minting, the co-rigidity check, `fk_residual_m`. Motive's streamed body frames sit 45 deg round from the mechanism, so this is what a controller reads. Its `templates/canarm_locks.json` is minted per Motive session and **gitignored** |
| `UMArm_KINEMATICS/` | product-of-exponentials FK, `fkine(q, params, order)`, where `order="yx"` is the CAN arm's measured proximal-joint assembly and `"xy"` the legacy default; plus `canarm_params.py` (the measured (3,10) table) and `canarm_actuators.py` (the measured actuator/axis map, with the legacy claim beside it) |
| `UMArm_KINOVA/` | the Gen3 stack: leashed driver, stdio-JSON bridge (`arm_bridge.py` on `.venv_kinova` / `bridge_client.py` anywhere), mocap body 1008 + hand-eye calibration results, `kinova_scene.py` (MJCF, bit-identical to the bench's Gen3), `setup_env.py`. See its `PORTING.md` |

## Visualizer and twin

| path | what is there |
|---|---|
| `viz/` | display-only MuJoCo room: `mjcf_canarm.py` (argument-driven arm MJCF; the CAN arm is the ProMax drawing reused from `digital_twin.mjcf_generator`, proximal hinges declared **y first = the measured `"yx"`** — the old x-first order drew live poses up to 19.9 mm off; the RS485 arm stays rod-and-disk `"xy"`; + `build_room_scene`), `multi_arm_viewer.py` (passive viewer, `mj_forward` only, q->qpos by joint name, a shared-array feed for the GUI's twin source, closing it stops nothing), `base_poses.py` (N-robot mocap mounts), `viz_layout.py` (shared array), `self_check.py` (plates vs fkine per arm) |
| `digital_twin/` | the twin, mirroring RS485 `UMArm_SIM`. **`CONTRACT.md` is binding** — every module signature is fixed there, including this session's bearing routing, q order, mass settings, `physics=`, `twin_params` (7a) and `mech_fit` (7b); `reference/` holds the 2026-09-10 read of the RS485 implementation (read it instead of the worktree); `README.md` = collect-then-fit plan, `data_schema.md` = JSONL collection schema |
| `digital_twin/mjcf_generator.py` | **the twin's geometry**: `generate_xml`/`build_model`, the **ProMax segment of Fig. 1C** — on every link a rod, two bearing hubs, two upright + two upside-down Ys 45 deg apart and 8 sleeves with their full mass; flat u-joint disks on the bracket side. 24 `pam_k` tendons (`pam_k` = board `0x100+k`) routed sleeve end -> hub bearing (`AO` 28 mm, `AA` 43.7 mm from the driven joint) -> outer-ring bracket across that joint: rest moment arm 43.104 mm (the old far-ring routing gave 46.8). **Closed-form seat azimuths** derived from `canarm_actuators.MEASURED_JOINT_PAIRS` (a35f94a's 90 deg proximal rotation undone). `order="yx"` hinges (qpos is a permutation of q — use `q_to_qpos`). Per-segment `link_mass_kg`/`bracket_mass_kg` (sleeves kept at exactly `actuator_mass`), `plate_mass`/`spacer_mass`, per-segment `joint_stiffness` (default 0); `y_tip_radius` 0.085 m is estimated from Fig. 1C, not measured. `promax_segment_elements` is the one drawing the display reuses. 1 ms timestep. `test_mjcf.py` offline |
| `digital_twin/actuator_model.py` | **the twin's plant**, pure numpy and it must never import torch: the 5->64->64->1 flow net predicting `dp/dt` from `[p, target-p, is_tle, l, ldot]`, the logistic fill/vent blend on the commanded error (per-population widths), per-node leak and gains, the anchored McKibben force law and its slack-gated pressure damping, `save`/`load` with a normalisation guard that raises, and the least-squares leak fitter. `l0` = measured `LL`; `coeff`/`bf` ratio/`damp_b1` are RS485 seeds awaiting a fit. `test_actuator_model.py` 30/30 offline |
| `digital_twin/sim_core.py` | **the twin's centre**: `SimArm` (MuJoCo + 24 nodes + one reentrant lock; `advance_to` is grid-quantised, callable from any thread, and bit-identical under racing advancers), `CanNode`/`SevenMmNode`/`TleNode` (the two firmwares, chosen by **variant byte**), `SimNode` (the plant seam — overrides only `true_psi`, `leak_pa_s`, `_integrate`). Node grid **4 ms** = 3 TLE PID steps (750 Hz) and 4 seven-mm steps (1000 Hz); `sub_step_counts` refuses any other. **`q()` returns `q` in `UMArm_KINEMATICS` order, hinges resolved by name** (`q_order`); until 2026-09-10 it returned declaration-order `qpos`, which a35f94a's seat rotation had hidden. `placeholder_xml`/`placeholder_actuator` are bring-up stand-ins never reached implicitly. `test_sim_core.py` offline |
| `digital_twin/sim_master.py` | **the twin's transport shell**: `SimMaster` subclasses `TLE_PCB/tlelib/backend.py::Backend` and replaces only `self.link`, so the 150 Hz cycle, `set_cycle_observer`, `snapshot_nodes` and `history` are the metal's own code. `physics="inline"` (default, `master.arm` in reach) or `"process"` (the GUI's SIM adapter). Per-id reply latency 1.87 ms at `0x101` -> 3.35 ms at `0x118` (64.3 us/id, CAN arbitration); each reply is delivered by the thread that holds it, at its due time — a hand-off to a second thread arrived 3-4 ms late and made `test_sim_master` depend on test order. `port` defaults to `"SIM"`; nothing here opens a serial port. `test_sim_master.py` offline, order-independent |
| `digital_twin/sim_process.py` | **the twin's plant in its own process** for `SimMaster(physics="process")`: `plant_main` free-runs the `SimArm` in a spawned child against `perf_counter`; `SimProcessLink` is the host side (table and edge in one pipe message, replies stamped `t_batch0 + reply_latency_ms`); `RemoteArm` gives the model, `data.qpos`/`q()` from shared memory and the board variants. In the GUI process the plant held the cycle to 97.6 Hz with about half the replies missed, which a controller could see. `test_sim_process.py` offline |
| `digital_twin/sim_mocap.py` | **the twin's mocap**: `SimMocap(arm, alive=)`, a `CanArmSimStream` fed from the live `SimArm`'s `q` through the real `CanArmMocap` listeners (round trip 3.9e-16 rad). Never steps physics; goes stale when its source closes; `fk_residual_m()` = twin plate sites vs fkine. No stream noise and no markers, so Lock plates stays live-only. `test_sim_mocap.py` offline |
| `digital_twin/twin_params.py` | **the fitted twin's one spelling**: `load_twin_kwargs(flow, mech)` -> the keyword arguments `SimArm`, `SimMaster`, `replay.rollout` and `twin_rollout` all take (`checkpoints/canarm_flow.npz` + `canarm_mech.json`, falling back to `canarm_outer.json`; coeff/bf, damping scalars and `mjcf` keys passed through). Logs `[twin_params] UNFITTED` for a missing part and `[twin_params] INCOMPLETE` for a file whose status is not complete; a malformed file raises. `describe()` = one-line summary. `test_twin_params.py` offline |
| `digital_twin/replay.py` | **the offline rollout**: `load_recording`/`save_recording` for the `session_*/` JSONL schema (Pa gauge out, raw counts in, population from the **variant byte**), `Recording`, `Rollout`, and `rollout()` driving a bare `SimArm` on the recording's own `can_sync_time_s` stamps -- never a nominal 6.667 ms grid. No threads, no clock; two rollouts are bit-identical. `ClampViolation` on any touch of the 4000 N pull-only clip. Target/state seams are arguments with named defaults (`stage_targets`+`sync_edge` in ADC counts first). `test_replay.py` 21/21 offline |
| `digital_twin/ring_analysis.py` | **the ringdown instrument**, pure numpy/scipy with no repo imports: `bandpass`, `find_episodes`, `fit_damped_sine`, and the single entry point `episodes_from_trace` every caller uses so a recording and a rollout cannot be measured differently. `r2` is ring-beyond-trend (detrended `ss_tot`), so it reads far lower than an ordinary r2 -- 0.30 is a real gate. Band/threshold/cycle gates are arguments; the RS485 arm's 1.5-12 Hz is a default to re-derive, not a constant. `test_ring_analysis.py` 15/15 offline |
| `digital_twin/twin_compare.py` | **the report card**: `twin_rollout` rolls a recording through the twin open loop, `align_on_sync` matches rows on `can_sync_time_s` at zero tolerance, `pick_reference_rows` picks one settled window and BOTH traces average over the same rows or the result is invalid. `compare_metrics` reports per joint and per board, each split by population (a joint whose antagonists straddle the split is `mixed`). NRMSE 1.0 = no better than 'it does not move'. `test_twin_compare.py` 17/17 offline |
| `digital_twin/fit_bounce.py` | **the dissipation fit**: Nelder-Mead over `joint_damping`, `joint_frictionloss`, `tendon_damping` and `damp_b1` (split per population -> the contract's per-segment 3-vector). Targets detected once on the real trace and shared by every candidate; the real fit is seeded with its own frequency and the twin fitted blind. A guard refuses any fit whose trajectory RMS is more than 10 % worse than the shipped baseline's, and emits `fit_bounce_GUARD_FAILED.json` rather than a checkpoint. `test_fit_bounce.py` 17/17 offline |
| `digital_twin/force_audit.py` | **five re-runnable measurements that change nothing**: `geometry` (Chou-Hannaford read backwards, plus four muscles against this arm's own 28 mm AO ring), `static`, `ring` (`f^2` affine in the coefficient scale), `replay` (Savitzky-Golay peak acceleration), `clamp` (headroom to the pull-only clip -- substituted for the RS485 audit's collision part, which this contact-free MJCF cannot carry). Every sweep goes through `scaled()`, which returns a new law; a full run asserts the audited model is bit-identical afterwards. `test_force_audit.py` 19/19 offline |
| `digital_twin/dataset.py` | **the fitting side's reader**, pure numpy and it must never import torch: `session_*/` -> `Recording` (ADC converted with each board's **own** `NodeCal` from the recorded variant byte), `TendonKinematics` (`q` -> anchored `l`/`ldot` through the twin's MJCF, **writing `q` to the hinges by name** — until 2026-09-10 each proximal pair was swapped), `make_windows` (multiple-shooting windows re-anchored at their own `p0`, never straddling an episode), `split_episodes` (**whole-episode** train/holdout -- a per-sample split at 150 Hz reports a fantasy holdout), `closed_segments`/`fit_leaks` (closed holds inferred from the commanded error, since this bus has no valve-state bit), and `write_synthetic_session` (a known plant written in the real schema, so one reader serves both). `test_dataset.py` offline |
| `digital_twin/train_actuator_net.py` | **the fit, and the only module in the package allowed to import torch**: multiple shooting/BPTT on the GPU with autograd in float64, the firmware regulator (error latched per sync edge from the simulated ADC) inside the loop, 1 ms substeps = 7 per 150 Hz cycle, full-batch Adam with `chunk` gradient accumulation for VRAM. Pipeline is leak fit -> optional warm start -> shooting -> held-out evaluation -> a checkpoint written through `ActuatorModel.save` so `load`'s unit guard checks the run. CLI `--session --out --epochs --device --holdout`, plus `--make-synthetic` for a hardware-free end-to-end run. `test_train_actuator_net.py` 12/12 offline, including **the equality test**: the numpy and torch forwards agree to 1e-13 on the same weights |
| `digital_twin/outer_fit.py` | the earlier coordinate search over three per-segment force-gain multipliers + `tendon_damping` + `joint_damping` on 15 s of `random_walk`; fitted on the rotated seats, so superseded by `mech_fit`. Its `canarm_outer.json` is `twin_params`' fallback only |
| `digital_twin/mech_fit.py` | **the mechanical fit**: CMA-ES (`cma`, 24 spawn workers, population 24) in log space on 13 contiguous training windows (178.9 s per candidate: random_walk, chirp, ringdown charge+release, staircase, pair_sweep; **never `validation`**, no sync gap over 0.25 s); objective = open-loop joint RMS through `twin_compare.twin_rollout`, family-weighted; a candidate with an invalid window scores 1000 deg + count. Masses **as built**: 8 fixed 30 g sleeves + one shared link structure, ring + spacer + ring connectors, one ring on segment 3, pulled toward the Koopman prior; rest force gain and bf/l0 per segment (bf/l0 bounds sized from training contraction only); per-segment `joint_stiffness`; three dissipation scalars. Writes `canarm_mech.running.json` every generation; `promote()` replaces `canarm_mech.json` only when complete; then mass and scaling-direction profiles. `--plan-only`, `--profile-only`, `--start`, `--max-hours`. `test_mech_fit.py` offline |
| `digital_twin/checkpoints/` | **the fitted twin** `twin_params` loads: `canarm_flow.npz` (flow net; held-out 3682 Pa as trained, 3676 Pa under the corrected kinematics), `canarm_mech.json` (the 2026-09-10 as-built refit, with bounds, budget and profiles: 2.21 kg moving, segment-3 joint stiffness 3.95 N m/rad; held out 2.124 deg / nrmse 0.187 / corr +0.993), `canarm_outer.json` (earlier search, fallback only). `canarm_mech.running.json` = a fit in progress, **gitignored** |

## Testing

| path | what is there |
|---|---|
| `hw_tests/` | hardware-in-the-loop test scripts and dated result reports (CAN bring-up, OTA, mocap, Kinova) |
| `hw_tests/can_bringup.py` | **CAN acceptance, read-only**: discovery + variant check, CAN diagnostics before/after, a 60 s 150 Hz soak with every enable bit clear, port-release check. Emits no OTA/set-ID/enable frame under any argument and asserts the enable bits are clear in the table before starting. 15/15 on 2026-08-20 |
| `hw_tests/integrated_gui_test.py` | **operator-GUI acceptance, read-only**: drives the real `canarm_control_gui.py` window against the live arm — connect/scan/cycle, mocap strip sim+live, spawned viewer for 60 s, target staged with the enable bit clear, port release. `--phase profile` gates the window's two periodic jobs on and off to attribute the cycle-rate loss. 28/29 on 2026-08-20; the one failure is the plot's cost, not the bus |
| `hw_tests/gui_sim_test.py` | **operator-GUI acceptance against the twin, offline**: the real window with `serial.Serial`, both `CanLink.open` paths and `MocapRx.start` poisoned; adapter list, SIM connect + scan (24 boards), `0x101` to 12 psi with all 24 selected, joint 2 through the twin receiver, kinematics line, viewer on the shared-array feed, STOP ALL, disconnect. Fails if cycle rate or replies heard sit more than 5 % / 2 points from the real window (145.43 Hz, 98.81 %) or the physics runs in the GUI process. 33/33 on 2026-09-10 (on the as-built refit: 142.1 Hz, 99.36 %, joint 2 +17.65 deg); `--physics inline` fails it |
| `hw_tests/twin_drive_campaign.py` | the 2026-08-21 single-board 12 psi drive campaign replayed on the twin and compared joint for joint with `results/axis_analysis_2026-08-21.json`: joint and sign per board, deflection ratio to the arm. As-built refit: 24/24 joints and signs, median ratio 1.01 (0.63-2.12), mean abs error 2.29 deg |
| `hw_tests/media/` | screenshots from the GUI tests (window, viewer, staged target), `gui_sim_*.png` (the window and viewer on the twin), and `promax_geometry_*.png` (the ProMax twin vs the old model — segment, hub close-up, arm at rest, arm bent — and the display model, for comparison with Fig. 1C) |
| `hw_tests/results/` | dated machine-readable records from the above. **Gitignored except** the records the committed calibration constants cite: `drive_2026-08-21.json` (the campaign), `axis_analysis_2026-08-21.{json,md}` (its reduction) and `gui_kinematics_2026-08-21.json` |
| `hw_tests/canarm_drive_campaign.py` | **the calibration session**: drives each of the 24 boards alone at 12 psi against a resting arm, then random multi-joint poses, recording every plate's mean marker cloud. One actuator live at a time; unused boards held enabled at 0.5 psi because `0x110` leaks; every exit path clears the enable bits |
| `hw_tests/canarm_axis_analysis.py` | **the reduction**: frame-inference health, chain lengths, the actuator/axis map, the azimuth calibration, and fkine-vs-mocap with a held-out split. Prints the constants to paste into the modules |
| `hw_tests/canarm_gui_kinematics.py` | live GUI check of Lock plates and the kinematics line; opens no serial port. 11/11 on 2026-08-21 |
| `hw_tests/report_canarm_axis_2026-08-21.md` | **the write-up**: the map, the 45 deg marker-azimuth finding, the proximal-order defect in fkine, and the error budget |
| `hw_tests/mocap_census.py` | **run this first** — wide-open NatNet receiver, enumerates every rigid-body id with per-id rates; settles which block belongs to which arm |
| `hw_tests/canarm_mocap_live.py` | CAN-arm q health + per-body dropouts + the five plate-gap chain measurement + room roster; `--sim` rehearses it without cameras |
| `hw_tests/mocap_wire_probe.py`, `mocap_sniff.py`, `mocap_discover.py` | below-the-SDK diagnostics for "run() returned true but no frames": raw multicast/unicast sockets, NAT_PING, all-interface port sweep |


## Collection and the fitted twin (2026-09-10)

| path | what is there |
|---|---|
| `collection/` | the hardware session: `safety.py` (the operator's 30 psi pair envelope, clamped then asserted, and the only place it is enforced), `excitation.py` (the designed signals, each phase justified by the parameter it makes identifiable), `recorder.py` (the 150 Hz JSONL schema plus the raw mocap stream), `resample.py` (the stream onto the sync edges, rate-agnostic), `campaign.py` (bus, mocap, camera, watchdog, bus revival, anomaly watch). `campaign.py --dry-run` opens no port and checks the whole plan |
| `UMArm_CAMERA/` | the bench camera: background grabber, pre-roll ring so a clip triggered by an anomaly contains the seconds before it, queued encoder, per-frame `.stamps.json`. `ARM_CROP` frames THIS arm, not the room -- the room has two |
| `data/session_*/` | the recordings. **Gitignored**: 273 867 cycles is 300 MB of JSONL. Families: session `012159` rest/supply/leak/staircase/pair_sweep; session `013843` rest/chirp/random_walk/ringdown/**validation** (the held-out 90 s) |
| `data/fit/` | fit logs, **gitignored**: flow net `train*.log`, `outer*.log`, `mech_fit_*.log` + `mech_fit_*_evals.jsonl` (every candidate's per-window loss), `mech_fit_*.out` |
| `digital_twin/CONTRACT.md` | **read this first**: what the twin implements, and for each of the three places it departs from the RS485 reference, what forced the departure |
| `digital_twin/reference/` | the 2026-09-10 read of the RS485 collision branch the port was made from, kept because that worktree belongs to another project and moves |
| `digital_twin/deliverable.py` | score the held-out sequence and render the side-by-side video from the SAME rollout, building the twin through `twin_params` (`--checkpoint`; `--mech` defaults to `canarm_mech.json`; `--outer` still accepted). Reports per-joint correlation and writes `deliverable/frame_sample.jpg` at 35 s |
| `digital_twin/side_by_side.py` | the three-panel compositor: bench camera, model at the measured `q`, model at the predicted `q`; caption halo drawn at one-pixel offsets because OpenCV 5.0 widens thick strokes (`test_side_by_side.py`) |
| `deliverable/` | the held-out deliverable: `deliverable_validation.json` (metrics + correlation; as-built refit 2.124 deg, nrmse 0.187, corr +0.993), `frame_sample.jpg` (frame at 35 s), `twin_vs_real_validation.mp4` (~41 MB, **gitignored**) |
| `hw_tests/canarm_hello.py` | drives one actuator and asks bus, mocap and camera what they saw -- run it first in any new session |
| `hw_tests/canarm_cycle_profile.py` | attributes cycle-rate loss to the component that causes it, one addition at a time |
| `hw_tests/canarm_disable_behaviour.py` | what a disable actually does to each board, reduced from a campaign's own leak traps |
| `hw_tests/canarm_park.py` | vent the arm and **verify** it, then park. Run after any session that ended abnormally |
| `hw_tests/canarm_anomaly_review.py` | whether a live anomaly flag is a fault or a coincidence, from every joint's responsiveness over the whole recording |
| `hw_tests/report_canarm_sysid_2026-09-10.md` | the first twin report: instruments, the GIL, disable behaviour, mocap resampling, what was collected, the first held-out score (10.850 deg). Its section 8 carries an erratum pointing at the refine report |
| `hw_tests/report_canarm_twin_refine_2026-09-10.md` | **the refine write-up**: every fit step and what was not fitted, the ProMax geometry, the qpos-as-q defect, the mass fit and what the recordings pin, the as-built refit and the weight answer, the held-out table (10.850 -> 9.691 -> 5.639 -> 2.124 deg), the SIM adapter and its timing |

## Source repos (read-only references)

- `C:\ESP\ESP_Projects\VEMA_MAX22200` (TLE firmware + TLE_PCB origin, branch TLE_PCB)
- `C:\ESP\ESP_Projects\VNEMA_MK8_PIDPWM` (legacy firmware + portable bundle origin; NEVER recursive-copy — 191 GB CSV inside)
- `C:\RUNZE_SRC\RS485_VEMA` (mocap/Kinova/twin origin; other agents work there — read-only)
- `C:\RUNZE_SRC\UMArm_dynamic_koopman_compliance` (Koopman controller). Its `runze_trying_MPC/real_system_fitting _experiment/mujoco_fit/generator.py` + `runze_trying_MPC/robot_config.py` config `original` are the ProMax MuJoCo the twin's hub routing and mass prior were checked against — read-only
