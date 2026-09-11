# ProMax dynamic-controller interfaces

The simulation experiment requested by `prompts/20260911_1_dynamic_controller_on_sim.md`
uses the fitted twin through `digital_twin.twin_params`. The twin's own contract
remains binding. Controller outputs are Pa gauge in ascending CAN ID order;
joint arrays are radians in the measured kinematic order, never raw `qpos`.

- `controller.make_controller(name, dt=1/150, checkpoint=..., twin_kwargs=None, seed=...)`
  returns `reset(q,p_pa)` and `command(q,qdot,p_pa,q_ref,qd_ref,qdd_ref,future_q=None)`.
  Names are `pid`, `ff_pid`, and `koopman_mppi`. The returned target has 24 entries.
- `sim_env.ControlEnv(...).reset()` and `.step(targets_pa)` return dictionaries
  containing observed `q`, `qdot`, `p_pa`, and separate scoring fields `q_true`,
  `qdot_true`, `p_true_pa`. Observation timestamps are explicit.
- Feedback consumes noisy, delayed observations only. Physics truth may provide
  supervised training labels or benchmark scores, and must never enter feedback.
- The offline pressure read uses the filtered ADC word available at each sync
  instant. The simulation clock has one sync per command. CAN reply delivery
  latency and operating-system scheduling are evaluated separately in the GUI.
- Default camera assumptions are 240 Hz, independent 0.10 degree Gaussian joint
  noise, 8.33 ms delay, and a causal 40 ms velocity-filter time constant. These
  are provisional test conditions, not measured specifications of Motive.
- Every command passes the measured antagonistic-pair envelope before reaching
  the plant or Backend. ADC rounding must not push the transmitted pair over
  30 psi. A force touching the twin's 4000 N clip invalidates an experiment.
- Training, tuning, and evaluation use separate whole episodes. A checkpoint
  records its data, normalization, seed, sample period, and validation results.
- GUI control computes in a spawned process and is enabled only for SIM in this
  implementation. Worker failure or stale observations stop its commands.
- Reports distinguish simulation results from guarantees and hardware evidence.
  A perturbed twin is a robustness scenario, not a measured sim-to-real gap.
