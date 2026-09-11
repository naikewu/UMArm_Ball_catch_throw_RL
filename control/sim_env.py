"""Deterministic CAN control experiments with delayed, sampled observations.

Physics runs on the existing 1 ms grid. Commands pass through each board's
ADC calibration and sync edge at 150 Hz; camera frames are sampled at 240 Hz.
Only the latest delivered frame is available to feedback, so neither velocity
estimation nor control uses future measurements. Truth is kept separately for
scoring and supervised labels. Noise and latency are explicit assumptions.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np

from collection.safety import PairEnvelope
from digital_twin.replay import PA_PER_PSI, counts_to_pa, pa_to_counts
from digital_twin.sim_core import SimArm
from digital_twin.twin_params import load_twin_kwargs


class ControlEnv:
    def __init__(self, dt=1 / 150, seed=20260911, randomize=False,
                 noise_std_rad=np.deg2rad(0.10), latency_s=2 / 240,
                 mocap_rate_hz=240.0, velocity_tau_s=0.04, twin_kwargs=None):
        if not 0 < dt <= 0.1 or not 0 < mocap_rate_hz <= 1000:
            raise ValueError("invalid control or camera period")
        if noise_std_rad < 0 or latency_s < 0 or velocity_tau_s <= 0:
            raise ValueError("invalid sensor settings")
        self.dt, self.seed = float(dt), int(seed)
        self.randomize = bool(randomize)
        self.noise_std_rad, self.latency_s = float(noise_std_rad), float(latency_s)
        self.camera_dt = 1 / float(mocap_rate_hz)
        self.velocity_tau_s = float(velocity_tau_s)
        self._base_kwargs = (load_twin_kwargs(log=lambda _: None) if twin_kwargs is None
                             else deepcopy(twin_kwargs))
        self.envelope = PairEnvelope()
        self.arm = None

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        kwargs = deepcopy(self._base_kwargs)
        variation = {}
        if self.randomize:
            # These ranges are robustness scenarios, not confidence intervals.
            for key in ("link_mass_kg", "bracket_mass_kg", "joint_stiffness"):
                if key in kwargs:
                    scale = self.rng.uniform(0.90, 1.10, 3)
                    kwargs[key] = (np.asarray(kwargs[key]) * scale).tolist()
                    variation[key + "_scale"] = scale.tolist()
            for key in ("joint_damping", "tendon_damping"):
                scale = float(self.rng.uniform(0.8, 1.2))
                kwargs[key] *= scale
                variation[key + "_scale"] = scale
            actuator = kwargs["actuator"]
            for key in ("fill_gain", "vent_gain", "coeff"):
                scale = self.rng.uniform(0.85, 1.15, np.asarray(getattr(actuator, key)).shape)
                setattr(actuator, key, np.asarray(getattr(actuator, key)) * scale)
                variation[key + "_scale"] = scale.tolist()
            extra_leak = self.rng.uniform(0, 150, 24)
            actuator.leak_pa_s += extra_leak
            variation["additional_leak_pa_s"] = extra_leak.tolist()
        kwargs.update(seed=self.seed, batched_actuator=True)
        self.arm = SimArm(**kwargs)
        self.ids = tuple(sorted(self.arm.nodes))
        self.variants = np.array([self.arm.nodes[b].variant for b in self.ids], dtype=int)
        self.t, self.steps, self._next_camera = 0.0, 0, self.camera_dt
        self.arm.advance_to(0.0)
        self._frames = deque()
        self._q = self.arm.q() + self.rng.normal(0, self.noise_std_rad, 12)
        self._qd = np.zeros(12)
        self._frame_time = 0.0
        self._frames.append((0.0, self._q.copy()))
        self.last_targets_pa = np.zeros(24)
        self.max_force_n = 0.0
        self.meta = dict(seed=self.seed, dt_s=self.dt, control_hz=1 / self.dt,
                         mocap_hz=1 / self.camera_dt,
                         joint_noise_std_deg=float(np.rad2deg(self.noise_std_rad)),
                         mocap_latency_s=self.latency_s,
                         velocity_filter_tau_s=self.velocity_tau_s,
                         sensor_assumptions="joint noise and latency provisional, not measured",
                         board_ids=list(self.ids), board_variants=self.variants.tolist(),
                         randomization=variation,
                         pressure_observation="filtered ADC at sync instants; reply delivery latency omitted offline",
                         plant="digital_twin.twin_params fitted ProMax; batched numpy flow")
        return self.observe()

    def observe(self):
        # Read the same filtered ADC word the immediately following sync
        # latches. Offline experiments omit CAN reply delivery wall time.
        p = np.array([counts_to_pa(self.arm.nodes[b]._wire_counts(), v)
                      for b, v in zip(self.ids, self.variants)], dtype=float)
        # The model has scalar hinge DOFs only, so qpos and qvel use the same
        # by-name address permutation. Never expose declaration-order qvel.
        qd_true = self.arm.data.qvel[self.arm._q_qposadr].copy()
        return dict(t=self.t, q=self._q.copy(), qdot=self._qd.copy(), p_pa=p,
                    q_true=self.arm.q(), qdot_true=qd_true,
                    p_true_pa=self.arm.pressures_pa(), mocap_time_s=self._frame_time,
                    mocap_age_s=self.t - self._frame_time)

    def step(self, targets_pa):
        if self.arm is None:
            raise RuntimeError("reset the environment before stepping")
        p = np.asarray(targets_pa, dtype=float)
        self.envelope.assert_safe(p / PA_PER_PSI, where="control simulation command")
        counts = np.array([int(round(float(pa_to_counts(v, variant))))
                           for v, variant in zip(p, self.variants)])
        # Quantization can push a pair past a limit by one ADC count. Remove
        # that count before transmission and assert the decoded wire target.
        wire = np.array([float(counts_to_pa(c, v)) for c, v in zip(counts, self.variants)])
        for a, b in self.envelope.pair_idx:
            while max(wire[a], 0) + max(wire[b], 0) > 30 * PA_PER_PSI + 1e-9:
                j = a if wire[a] >= wire[b] else b
                counts[j] -= 1
                wire[j] = float(counts_to_pa(counts[j], self.variants[j]))
        # Zero gauge is between two ADC integers; the firmware accepts the
        # nearest zero count. Its small negative decoded offset is not suction.
        self.envelope.assert_safe(np.maximum(wire, 0) / PA_PER_PSI,
                                  where="quantized control simulation command")
        self.last_targets_pa = wire
        self.arm.stage_targets({b: (int(c), True) for b, c in zip(self.ids, counts)})
        self.arm.sync_edge(self.t)
        self.steps += 1
        end = self.steps * self.dt
        while self._next_camera <= end + 1e-12:
            self.arm.advance_to(self._next_camera)
            frame_q = self.arm.q() + self.rng.normal(0, self.noise_std_rad, 12)
            self._frames.append((self._next_camera, frame_q))
            self._next_camera += self.camera_dt
        self.arm.advance_to(end)
        self.t = end
        while self._frames and self._frames[0][0] <= self.t - self.latency_s + 1e-12:
            stamp, q = self._frames.popleft()
            delta = stamp - self._frame_time
            if delta > 1e-12:
                alpha = -np.expm1(-delta / self.velocity_tau_s)
                self._qd += alpha * ((q - self._q) / delta - self._qd)
                self._q = q
                self._frame_time = stamp
        force = float(np.max(np.abs(self.arm.data.ctrl)))
        self.max_force_n = max(self.max_force_n, force)
        if force >= 4000 - 1e-6:
            raise RuntimeError("control experiment reached the 4000 N force clip")
        if not np.all(np.isfinite(self.arm.q())):
            raise RuntimeError("nonfinite simulation state")
        return self.observe()

    def close(self):
        if self.arm is not None:
            self.arm.all_off()
