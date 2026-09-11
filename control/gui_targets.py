"""Joint sliders and Cartesian target entry for the spawned SIM controller."""
from __future__ import annotations

import concurrent.futures
import tkinter as tk
import time
from tkinter import ttk

import numpy as np

from control.controller_process import JOINT_LIMIT_RAD


class TargetWindow:
    """A Tk view; compute and inverse kinematics run outside the Tk thread."""

    def __init__(self, parent, bridge, on_close):
        from control.trajectory import tip_position

        self.bridge, self.on_close = bridge, on_close
        self.window = tk.Toplevel(parent)
        self.window.title(f"SIM targets - {bridge.method}")
        self.window.protocol("WM_DELETE_WINDOW", on_close)
        self.window.resizable(True, True)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                              thread_name_prefix="tip-ik")
        self._pending = None
        self._closed = self._updating = False
        self._cartesian_goal = None
        self._target_revision = 0
        self.variables, self.value_labels = [], []
        content = ttk.Frame(self.window, padding=12)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Joint targets (degrees)").grid(row=0, column=0,
                                                               columnspan=3, sticky="w")
        limit = float(np.degrees(JOINT_LIMIT_RAD))
        for i in range(12):
            ttk.Label(content, text=f"S{i // 4 + 1}  q{i + 1:02d}").grid(
                row=i + 1, column=0, sticky="w", padx=(0, 8))
            var = tk.DoubleVar(value=float(np.degrees(bridge.target_q[i])))
            ttk.Scale(content, from_=-limit, to=limit, variable=var,
                      command=self._slider, length=300).grid(
                row=i + 1, column=1, sticky="ew", pady=2)
            label = ttk.Label(content, width=7, anchor="e")
            label.grid(row=i + 1, column=2, padx=(8, 0))
            self.variables.append(var)
            self.value_labels.append(label)
        content.columnconfigure(1, weight=1)
        self._labels()
        actions = ttk.Frame(content)
        actions.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Button(actions, text="Hold measured pose", command=self.hold).pack(side="left")
        ttk.Button(actions, text="Zero joint targets", command=lambda: self._set_q(np.zeros(12))).pack(
            side="left", padx=5)
        ttk.Button(actions, text="Stop controller", command=on_close).pack(side="right")

        tip = ttk.LabelFrame(content, text="Tip target in robot base frame (m)", padding=8)
        tip.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        xyz = tip_position(bridge.target_q)
        self.xyz = [tk.StringVar(value=f"{v:.5f}") for v in xyz]
        for i, axis in enumerate("XYZ"):
            ttk.Label(tip, text=axis).grid(row=0, column=2 * i, padx=(6, 3))
            ttk.Entry(tip, textvariable=self.xyz[i], width=10).grid(row=0, column=2 * i + 1)
        self.move_button = ttk.Button(tip, text="Move to target", command=self.move_tip)
        self.move_button.grid(row=1, column=0, columnspan=6, pady=(6, 8), sticky="ew")
        ttk.Label(tip, text="Nudge (mm)").grid(row=2, column=0, columnspan=2, sticky="w")
        self.step_mm = tk.StringVar(value="5")
        ttk.Spinbox(tip, from_=0.5, to=20, increment=0.5,
                    textvariable=self.step_mm, width=7).grid(row=2, column=2)
        self.nudge_buttons = []
        for i, axis in enumerate("XYZ"):
            for j, sign in enumerate((-1, 1)):
                button = ttk.Button(tip, text=f"{'-' if sign < 0 else '+'}{axis}",
                    command=lambda a=i, s=sign: self.nudge(a, s))
                button.grid(row=3, column=2 * i + j, sticky="ew", pady=(6, 0))
                self.nudge_buttons.append(button)
        self.note = tk.StringVar(value="Targets slew at 30 deg/s. Closing this window stops control.")
        ttk.Label(content, textvariable=self.note, wraplength=490).grid(
            row=15, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.status = tk.StringVar()
        ttk.Label(content, textvariable=self.status, wraplength=490).grid(
            row=16, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.tracking = tk.StringVar(value="Waiting for a measured pose ...")
        ttk.Label(content, textvariable=self.tracking, wraplength=490).grid(
            row=17, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.window.after(100, self._poll)

    def _labels(self):
        for var, label in zip(self.variables, self.value_labels):
            label.configure(text=f"{var.get():+.1f}")

    def _slider(self, _value=None):
        if self._updating:
            return
        self._cartesian_goal = None
        self._target_revision += 1
        try:
            self.bridge.set_joint_target(np.deg2rad([v.get() for v in self.variables]))
            self._labels()
            self._sync_xyz()
        except ValueError as exc:
            self.note.set(str(exc))

    def _sync_xyz(self):
        from control.trajectory import tip_position
        for var, value in zip(self.xyz, tip_position(self.bridge.target_q)):
            var.set(f"{value:.5f}")

    def _set_q(self, q):
        self._cartesian_goal = None
        self._target_revision += 1
        self.bridge.set_joint_target(q)
        self._updating = True
        try:
            for var, value in zip(self.variables, np.degrees(q)):
                var.set(float(value))
            self._labels()
            self._sync_xyz()
        finally:
            self._updating = False

    def hold(self):
        q = self.bridge.mocap.get_q()
        if q is not None:
            self._set_q(np.clip(q, -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD))

    def move_tip(self):
        if self._pending is not None:
            return
        try:
            target = np.array([float(var.get()) for var in self.xyz])
            if not np.all(np.isfinite(target)):
                raise ValueError("tip coordinates must be finite")
        except ValueError as exc:
            self.note.set(str(exc))
            return
        from control.trajectory import inverse_kinematics
        self._pending = self._executor.submit(inverse_kinematics, target,
                                              self.bridge.target_q.copy(),
                                              max_angle_rad=JOINT_LIMIT_RAD)
        self._ik_target = target
        self._ik_revision = self._target_revision
        self.move_button.configure(state="disabled")
        for button in self.nudge_buttons:
            button.configure(state="disabled")
        self.note.set("Finding a reachable joint target ...")

    def nudge(self, axis, direction):
        try:
            step = float(self.step_mm.get())
            if not np.isfinite(step) or not 0 < step <= 20:
                raise ValueError("nudge must be between 0 and 20 mm")
            self.xyz[axis].set(f"{float(self.xyz[axis].get()) + direction * step / 1000:.5f}")
        except ValueError as exc:
            self.note.set(str(exc))
            return
        self.move_tip()

    def _poll(self):
        if self._closed:
            return
        if self._pending is not None and self._pending.done():
            future, self._pending = self._pending, None
            try:
                from control.trajectory import tip_position
                q = future.result()
                if self._ik_revision != self._target_revision:
                    raise ValueError("a newer joint target superseded this IK request")
                residual = float(np.linalg.norm(tip_position(q) - self._ik_target))
                if residual > 0.002:
                    raise ValueError(f"target unreachable within bounds (residual {residual * 1000:.1f} mm)")
                self._set_q(q)
                self._cartesian_goal = self._ik_target.copy()
                self.note.set(f"Tip target accepted; kinematic residual {residual * 1000:.2f} mm.")
            except Exception as exc:
                self.note.set(f"Target unchanged: {exc}")
            self.move_button.configure(state="normal")
            for button in self.nudge_buttons:
                button.configure(state="normal")
        bridge = self.bridge
        self.status.set(f"{bridge.status} | worker PID {getattr(bridge.process, 'pid', '-')}\n"
                        f"Applied {bridge.command_hz:.1f} Hz | last solve {bridge.solve_ms:.2f} ms | "
                        f"solves over {bridge.dt * 1000:.2f} ms: {bridge.deadline_misses}")
        stamp, _, measured_q, _ = bridge.mocap.latest_control_sample()
        if measured_q is None or stamp is None or time.monotonic() - stamp > 0.15:
            self.tracking.set("Measured pose stale; tracking error unavailable.")
        else:
            from control.trajectory import tip_position
            error_deg = float(np.degrees(np.max(np.abs(measured_q - bridge.target_q))))
            line = f"Measured joint error: max {error_deg:.2f} deg"
            if self._cartesian_goal is not None:
                error_mm = float(1000 * np.linalg.norm(tip_position(measured_q) - self._cartesian_goal))
                line += f" | tip error: {error_mm:.2f} mm"
            self.tracking.set(line)
        self.window.after(100, self._poll)

    def close(self):
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()
