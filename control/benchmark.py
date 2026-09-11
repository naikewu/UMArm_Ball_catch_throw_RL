"""Reproducible three-controller Soft comparison on a common noisy twin.

The simulation clock measures tracking; perf_counter measures controller cost.
These are reported separately because offline integration does not establish
that a controller meets a wall-clock deadline through GUI and CAN transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from control.controller import make_controller, pair_indices, PA_PER_PSI
from control.sim_env import ControlEnv
from control.trajectory import SoftTrajectory, tip_position


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def score_trace(trace, variants):
    active = trace["pen_down"].astype(bool)
    if not np.any(active):
        active = np.ones(len(trace["t"]), dtype=bool)
    e = trace["q_true"] - trace["q_ref"]
    tip_error = np.linalg.norm(trace["tip_true"] - trace["tip_ref"], axis=1)
    milliseconds = trace["solve_ms"]
    pressure_error = (trace["p_measured_pa"] - trace["targets_pa"]) / PA_PER_PSI
    pairs = pair_indices()
    dt = float(np.median(np.diff(trace["t"])))
    return dict(
        writing_tip_rms_mm=1000*_rms(tip_error[active]),
        writing_tip_p95_mm=1000*float(np.percentile(tip_error[active], 95)),
        writing_tip_max_mm=1000*float(tip_error[active].max()),
        writing_joint_rms_deg=float(np.rad2deg(_rms(e[active]))),
        writing_segment_joint_rms_deg=[float(np.rad2deg(_rms(e[active, s:s+4])))
                                       for s in (0, 4, 8)],
        full_tip_rms_mm=1000*_rms(tip_error),
        max_joint_deg=float(np.rad2deg(abs(trace["q_true"]).max())),
        target_max_psi=float(trace["targets_pa"].max()/PA_PER_PSI),
        target_pair_max_psi=float(trace["targets_pa"][:, pairs].sum(axis=-1).max()/PA_PER_PSI),
        pressure_tracking_rms_psi={"tle": _rms(pressure_error[:, variants == 2]),
                                   "7mm": _rms(pressure_error[:, variants != 2])},
        solver_ms={"median": float(np.median(milliseconds)),
                   "p95": float(np.percentile(milliseconds, 95)),
                   "p99": float(np.percentile(milliseconds, 99)),
                   "max": float(np.max(milliseconds))},
        solver_over_6_667ms_fraction=float(np.mean(milliseconds > 1000*dt)),
        cycles=len(trace["t"]), simulated_duration_s=float(trace["t"][-1] + dt))


def run(method, trajectory=None, *, seed=202609111, randomize=False,
        noise_std_deg=.1, checkpoint=None, controller_kwargs=None, duration_s=None):
    trajectory = SoftTrajectory() if trajectory is None else trajectory
    env = ControlEnv(seed=seed, randomize=randomize, noise_std_rad=np.deg2rad(noise_std_deg))
    obs = env.reset()
    controller = make_controller(method, dt=env.dt, seed=seed, checkpoint=checkpoint,
                                 **(controller_kwargs or {}))
    controller.reset(obs["q"], obs["p_pa"])
    horizon = int(getattr(controller, "horizon", 1))
    prediction_dt = float(getattr(controller, "prediction_dt", env.dt))
    # GPU/module initialization is not part of a running control cycle. Warm
    # it with the initial observation, then reset controller memory and RNG.
    if method == "koopman_mppi":
        q, qd, qdd = trajectory.sample(0)
        controller.command(obs["q"], obs["qdot"], obs["p_pa"], q, qd, qdd,
                           future_q=trajectory.future(0, horizon, prediction_dt))
        controller.reset(obs["q"], obs["p_pa"])
    seconds = trajectory.duration_s if duration_s is None else duration_s
    n = int(np.ceil(seconds / env.dt))
    trace = {key: np.zeros((n, dim)) for key, dim in
             (("q_true",12),("q_observed",12),("q_ref",12),("p_measured_pa",24),
              ("p_true_pa",24),("targets_pa",24),("tip_true",3),("tip_ref",3))}
    trace.update(t=np.arange(n)*env.dt, pen_down=np.zeros(n, dtype=bool), solve_ms=np.zeros(n))
    started = time.perf_counter()
    try:
        for k, t in enumerate(trace["t"]):
            q, qd, qdd = trajectory.sample(t)
            future = trajectory.future(t, horizon, prediction_dt) if horizon > 1 else None
            begin = time.perf_counter()
            targets = controller.command(obs["q"], obs["qdot"], obs["p_pa"], q, qd, qdd,
                                         future_q=future)
            trace["solve_ms"][k] = 1000*(time.perf_counter() - begin)
            for key, value in (("q_true",obs["q_true"]),("q_observed",obs["q"]),
                               ("q_ref",q),("p_measured_pa",obs["p_pa"]),
                               ("p_true_pa",obs["p_true_pa"]),("targets_pa",targets)):
                trace[key][k] = value
            trace["pen_down"][k] = trajectory.pen_down(t)
            trace["tip_ref"][k] = trajectory.tip_target(t)
            obs = env.step(targets)
        trace["tip_true"][:] = tip_position(trace["q_true"])
        meta = dict(method=method, seed=seed, randomized=randomize,
                    environment=env.meta, trajectory=trajectory.metadata,
                    wall_elapsed_s=time.perf_counter()-started,
                    max_tendon_force_n=env.max_force_n,
                    controller_options=controller_kwargs or {},
                    checkpoint=None if checkpoint is None else str(checkpoint),
                    last_controller_diagnostics=getattr(controller,"last_diagnostics",{}),
                    metrics=score_trace(trace, env.variants))
        if checkpoint is not None:
            meta["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        return trace, meta
    finally:
        env.close()


def save_run(folder, name, trace, meta):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (name + ".npz"), **trace)
    (folder / (name + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["pid", "ff_pid", "koopman_mppi"])
    parser.add_argument("--speeds", nargs="+", default=["slow", "fast"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[202609111, 202609112, 202609113])
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("deliverable/dynamic_control/benchmark"))
    args = parser.parse_args()
    results = []
    for speed in args.speeds:
        trajectory = SoftTrajectory(speed=speed)
        for seed in args.seeds:
            for method in args.methods:
                name = f"{speed}_{'perturbed' if args.randomized else 'nominal'}_{seed}_{method}"
                trace, meta = run(method, trajectory, seed=seed, randomize=args.randomized,
                                  checkpoint=args.checkpoint)
                save_run(args.out, name, trace, meta)
                results.append(dict(name=name, **meta))
                print(json.dumps({"run":name, **meta["metrics"]}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
