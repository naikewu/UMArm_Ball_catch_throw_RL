# Repo log

- 2026-08-20: Workspace created from recon of VEMA_MAX22200, VNEMA_MK8_PIDPWM, and RS485_VEMA (4 recon + 5 build agents, Fable 5 supervising):
  - Scaffold + both ESP-IDF firmware trees (both compile here) + prebuilt images with provenance + `legacy_host` bundle (COM ports rerouted through `bench_env.py`; `pc_backend` builds and self-tests) + `.venv`.
  - Ported the TLE host tooling as `TLE_PCB/`, vendoring `vema_proto.py` to cut the last external dependency, repointing the flash build path at `firmware/tle/build`, verified offline (protocol constants, figures byte-identical, both GUIs construct without a port).
  - Ported `UMArm_MOCAP` + `UMArm_KINEMATICS` from RS485_VEMA; made the Motive rigid-body ID base per-instance so two arms share one NatNet stream; added `CanArmMocap` (2000–2005), `sim_stream`, and a placeholder `canarm_params` — 568 tests pass with socket constructors poisoned.
  - Ported the Kinova Gen3 stack (`UMArm_KINOVA/`), extracted `kinova_scene.py` bit-identical to the bench's Gen3, built the quarantined `.venv_kinova` (173 offline tests green).
  - Added the display-only multi-arm visualizer (`viz/`), `canarm_control_gui.py` (subclasses the TLE controller; spawned viewer; mocap strip), and the documented `digital_twin/` skeleton.
