# Kinova Gen3 hardware check — 2026-08-20 (connectivity PASSED, motion deferred)

Run by the supervisor inline (`hw_tests\kinova_bridge_connectivity.py`,
`WS\.venv\Scripts\python.exe`), against the real arm at 192.168.1.10 through the
workspace bridge (`bridge_client.KinovaLink` → `arm_bridge.py` on `.venv_kinova`).
No motion command of any kind was sent.

## What passed

| step | result |
|---|---|
| offline protocol suite | `pytest UMArm_KINOVA/test_arm_bridge.py` — 40 passed |
| bridge spawn | `.venv_kinova` bridge started and reached `running` |
| `ping` | ok |
| `connect` 192.168.1.10 | ok; live snapshot returned |
| joints (deg) | `[0.004, 15.039, 180.002, 229.983, 0.021, 54.983, 90.008]` |
| tool pose (m, deg) | `[0.4615, 0.0159, 0.4333, 90.22, 0.29, 90.66]` |
| mocap half | started automatically after connect; `on=true, live=false, frames=0` — degraded loudly, did not hang |
| `mocap_solve_base` | refused with `"only 0 frames of rigid body 1008 in the last 0.50 s (need 10) — is Motive streaming and the body visible?"` — the designed failure mode, observed on real hardware |
| `disconnect` + shutdown | clean; bridge process verified exited (the non-daemon NatNet threads did not wedge it) |

The `frames=0` from the bridge's own receiver is an independent second confirmation
of the Motive streaming outage documented in `report_mocap_2026-08-20.md`.

## Deferred, with reasons

1. **Mocap body-1008 stillness capture, base solve (X vs stored), and the
   room-frame-verified jog** — blocked by the Motive outage: with no NatNet frames
   there is no room frame to verify against. Re-run once Motive streams:
   `connect` → `mocap` capture → `mocap_solve_base` (compare `vs_stored_mm` /
   `vs_stored_deg` against `results/mocap_calib_20260819_*`; centimetres means the
   cart moved) → `arm {on}` → one discrete `jog` ≤ 15 mm along a room-frame
   direction converted through the fresh X (`bridge_mocap.world_dir_to_base`,
   the R_Xᵀ rule) → verify the mocap-measured displacement → jog back → disarm.
2. **Any commanded motion at all** — the automation permission layer declined
   delegating robot-motion commands to background agents this session, so the
   supervisor limited the live test to read-only commands. The motion round trip
   (discrete `jog`, envelope-checked, ≤ 15 mm, slow) is scripted in intent above
   and safe to run interactively whenever the operator chooses.
3. **The streamed-twist held-jog path (`jog_start`/`jog_stop`)** — still has never
   run against a real arm; the first-hardware checklist lives in the RS485 repo at
   `docs\reports\real_arm_jog_and_mocap_2026-08-20.md`. Not this session's to
   execute.

## Notes

- The bridge reports state at ~8 Hz while connected; `connect` returns the snapshot
  in its own ok message, so pose+joints are one instant's `RefreshFeedback`.
- Tool configuration was not separately queried (the bridge exposes pose via
  snapshot; `kinova_arm.tool_configuration` reads the configured (10, 0, 5) mm tool
  transform — read it during the motion session and store it with any fitted data).
- Faults: none encountered; a pose move silently no-ops in a fault state — clear
  faults at http://192.168.1.10 before the motion session.
