"""SIM-only controller process and bounded, latest-value GUI transport.

The existing Backend still owns the synchronized CAN-shaped cycle. A spawned
process computes pressures from receiver samples; a small host thread stages
the complete pressure vector. Shared snapshots overwrite old values, so a slow
optimizer cannot build a queue of commands that the robot executes later.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections import deque

import numpy as np

from collection.safety import ALL_BASES, PairEnvelope

PA_PER_PSI = 6894.757
METHODS = ("pid", "ff_pid", "koopman_mppi")
JOINT_LIMIT_RAD = np.deg2rad(25.0)
STATE_TIMEOUT_S = 0.15
COMMAND_TIMEOUT_S = 0.20


def _read(shared):
    lock = shared.get_lock()
    if not lock.acquire(timeout=0.025):
        raise RuntimeError("controller shared-state lock timed out")
    try:
        return np.frombuffer(shared.get_obj(), dtype=np.float64).copy()
    finally:
        lock.release()


def _write(shared, values):
    lock = shared.get_lock()
    if not lock.acquire(timeout=0.025):
        raise RuntimeError("controller shared-state lock timed out")
    try:
        np.frombuffer(shared.get_obj(), dtype=np.float64)[:] = values
    finally:
        lock.release()


class _StopFlag:
    """A process death cannot strand a lock inside this shutdown signal.

    Multiprocessing Event.wait owns a condition lock briefly on return. Killing
    a child there can poison the very event its parent needs for shutdown. This
    one-byte, monotone flag has no lock and sleeps in short interruptible hops.
    """
    def __init__(self, ctx):
        self.flag = ctx.RawValue("b", 0)

    def is_set(self):
        return bool(self.flag.value)

    def set(self):
        self.flag.value = 1

    def wait(self, seconds):
        end = time.perf_counter() + max(seconds, 0.0)
        while not self.is_set():
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.005))
        return self.is_set()


def validate_joint_target(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (12,) or not np.all(np.isfinite(q)):
        raise ValueError("joint target must contain 12 finite radians")
    if np.max(np.abs(q)) > JOINT_LIMIT_RAD + 1e-10:
        raise ValueError("joint targets must stay within +/-25 degrees")
    return q.copy()


def quantized_safe_targets(psi, nodes, envelope=None):
    """Stage values whose actual per-board ADC words satisfy both pressure caps.

    A command exactly at 30 psi can round upward. Evaluate decoded words with
    negative zero offsets clipped to zero, then remove counts from the larger
    antagonist until the *wire* target is safe. Calibration comes from each
    scanned NodeState, not the ID range.
    """
    envelope = envelope or PairEnvelope()
    psi = envelope.assert_safe(psi, where="controller requested pressures")
    cals = [nodes[b].cal for b in ALL_BASES]
    counts = [cal.psi_to_counts(float(value)) for cal, value in zip(cals, psi)]
    wire = np.array([max(0.0, cal.counts_to_psi(c)) for cal, c in zip(cals, counts)])
    for a, b in envelope.pair_idx:
        while wire[a] + wire[b] > 30.0 + 1e-9:
            i = a if wire[a] >= wire[b] else b
            counts[i] -= 1
            wire[i] = max(0.0, cals[i].counts_to_psi(counts[i]))
    envelope.assert_safe(wire, where="controller decoded ADC targets")
    actual = np.array([max(0.0, cal.counts_to_psi(cal.psi_to_counts(float(value))))
                       for cal, value in zip(cals, wire)])
    envelope.assert_safe(actual, where="controller before Backend staging")
    return wire


def _warm_controller(controller, obs):
    """Resolve lazy solver imports without publishing a pressure command."""
    q, p_pa, qdot = obs[2:14], obs[14:38], obs[38:50]
    controller.reset(q, p_pa)
    started = time.perf_counter()
    pressure = controller.command(q, qdot, p_pa, q, np.zeros(12), np.zeros(12))
    PairEnvelope().assert_safe(np.asarray(pressure) / PA_PER_PSI,
                               where="discarded controller warmup")
    return (time.perf_counter() - started) * 1000


def _controller_main(observation, target, command, stop, messages, method,
                     checkpoint, dt):
    """Child entry point: it never receives a backend or hardware address."""
    try:
        from control.controller import make_controller

        controller = make_controller(method, dt=dt, checkpoint=checkpoint)
        # scipy.optimize is imported lazily by the pressure allocator; in a
        # fresh GUI child its first call took240ms, versus2.4ms when warm. Run
        # it while all boards are disabled, then reset from a fresh observation
        # in the main loop. Neither this command nor its old state is emitted.
        while not stop.is_set():
            obs = _read(observation)
            if obs[0] > 0 and time.perf_counter() - obs[0] < STATE_TIMEOUT_S:
                break
            stop.wait(0.001)
        if stop.is_set():
            return
        warmup_ms = _warm_controller(controller, obs)
        messages.put(("ready", warmup_ms))
        envelope = PairEnvelope()
        q_ref = None
        qd_ref = np.zeros(12)
        last_seq = -1
        count = late = 0
        next_tick = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                stop.wait(next_tick - now)
                continue
            next_tick = max(next_tick + dt, now)
            obs = _read(observation)
            stamp, seq = obs[:2]
            if stamp <= 0 or seq == last_seq:
                continue
            if now - stamp > STATE_TIMEOUT_S:
                raise RuntimeError("mocap sample is stale in controller process")
            q, p_pa, qdot = obs[2:14], obs[14:38], obs[38:50]
            if not np.all(np.isfinite(obs)):
                raise ValueError("non-finite controller observation")
            if q_ref is None:
                controller.reset(q, p_pa)
                q_ref = q.copy()
            desired = validate_joint_target(_read(target))
            # A slider specifies position, not an instantaneous velocity step.
            # A critically damped second-order reference avoids a derivative
            # kick in PID and supplies consistent acceleration to feedforward.
            omega = 6.0
            qdd_ref = np.clip(omega ** 2 * (desired - q_ref) - 2 * omega * qd_ref,
                              -np.deg2rad(90), np.deg2rad(90))
            qd_ref = np.clip(qd_ref + qdd_ref * dt, -np.deg2rad(30), np.deg2rad(30))
            q_ref = np.clip(q_ref + qd_ref * dt, -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD)
            started = time.perf_counter()
            pressures = np.asarray(controller.command(
                q, qdot, p_pa, q_ref, qd_ref, qdd_ref), dtype=float)
            envelope.assert_safe(pressures / PA_PER_PSI,
                                 where="controller child output")
            finished = time.perf_counter()
            solve_ms = (finished - started) * 1000.0
            late += int(finished - started > dt)
            count += 1
            _write(command, np.r_[finished, stamp, count, solve_ms, late,
                                  pressures])
            last_seq = seq
    except BaseException as exc:
        try:
            messages.put_nowait(("error", f"{type(exc).__name__}: {exc}"))
        except queue.Full:
            pass
    finally:
        stop.set()


class ControllerBridge:
    """Own the compute child and retire its authority before stopping outputs.

    ``start`` requires an actual SimMaster and its bound twin receiver. Live
    adaptation is deliberately a later validation task. No hardware is opened
    by this class or by the worker. Lifecycle methods belong to the Tk thread;
    the host thread never calls a widget.
    """

    def __init__(self, backend, mocap, method="pid", *, checkpoint=None,
                 rate_hz=150.0, log=None):
        from digital_twin.sim_master import SimMaster

        if not isinstance(backend, SimMaster):
            raise ValueError("dynamic controllers are SIM-only; connect the digital twin")
        if getattr(mocap, "bound_backend", None) is not backend:
            raise ValueError("controller requires the connected twin's mocap receiver")
        if method not in METHODS:
            raise ValueError(f"unknown controller {method!r}")
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("controller rate must be positive")
        self.backend, self.mocap, self.method = backend, mocap, method
        self.checkpoint, self.dt, self.log = checkpoint, 1.0 / rate_hz, log
        self._ctx = mp.get_context("spawn")
        self._observation = self._ctx.Array("d", 50)
        self._target = self._ctx.Array("d", 12)
        self._command = self._ctx.Array("d", 29)
        self._stop = _StopFlag(self._ctx)
        self._messages = self._ctx.Queue(maxsize=8)
        self._apply_lock = threading.Lock()
        self.process = self.thread = None
        self.status = "stopped"
        self.error = ""
        self.command_hz = self.solve_ms = 0.0
        self.warmup_ms = 0.0
        self.deadline_misses = self.commands = 0
        self.worker_commands = 0
        self.solve_times_ms = deque(maxlen=8192)
        self.command_intervals_ms = deque(maxlen=8192)
        self.last_pressure_pa = np.zeros(24)
        self.target_q = np.zeros(12)
        self._started = self._first_command = 0.0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive() and not self._stop.is_set()

    def set_joint_target(self, q):
        q = validate_joint_target(q)
        _write(self._target, q)
        self.target_q = q

    def start(self):
        if self.process is not None:
            raise RuntimeError("a controller bridge can only be started once")
        nodes = self.backend.snapshot_nodes()
        if any(b not in nodes or not nodes[b].present for b in ALL_BASES):
            raise ValueError("scan all 24 twin boards before starting a controller")
        stamp, _, q, _ = self.mocap.latest_control_sample()
        if q is None or stamp is None or time.monotonic() - stamp > STATE_TIMEOUT_S:
            raise ValueError("twin mocap must be fresh before starting a controller")
        self.set_joint_target(np.clip(q, -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD))
        self.backend.stop_all()
        self.backend.set_targets(dict.fromkeys(ALL_BASES, 0.0))
        self.backend.select(ALL_BASES)
        self.backend.start_cycle()
        self.status = "loading controller"
        self._started = time.perf_counter()
        self.process = self._ctx.Process(target=_controller_main,
            args=(self._observation, self._target, self._command, self._stop,
                  self._messages, self.method, self.checkpoint, self.dt),
            name=f"canarm-controller-{self.method}", daemon=True)
        self.process.start()
        self.thread = threading.Thread(target=self._service, daemon=True,
                                       name="canarm-controller-bridge")
        self.thread.start()
        return self

    def _service(self):
        envelope = PairEnvelope()
        last_command = 0
        last_arrival = self._started
        last_observation = -1
        nodes = None
        ready = enabled = False
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                while True:
                    try:
                        kind, message = self._messages.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "error":
                        raise RuntimeError(message)
                    self.warmup_ms = float(message)
                    ready = True
                    self.status = "waiting for first command"
                if not self.process.is_alive():
                    raise RuntimeError(f"controller exited ({self.process.exitcode})")
                if now - self._started > 30 and not ready:
                    raise RuntimeError("controller did not finish loading within 30 s")
                sample_stamp, seq, q, qdot = self.mocap.latest_control_sample()
                age = (time.monotonic() - sample_stamp
                       if sample_stamp is not None else float("inf"))
                if q is None or age > STATE_TIMEOUT_S:
                    raise RuntimeError("twin mocap stopped or became stale")
                if seq != last_observation or nodes is None:
                    nodes = self.backend.snapshot_nodes()
                    p_pa = np.array([nodes[b].pressure_psi for b in ALL_BASES]) * PA_PER_PSI
                    # Acquisition timestamps cross the process boundary via
                    # their age; publication retains every camera-rate qdot.
                    _write(self._observation, np.r_[now - age, seq, q, p_pa, qdot])
                    last_observation = seq
                if now - self._started > 0.5 and any(
                    b not in nodes or now - nodes[b].last_reply_t > STATE_TIMEOUT_S
                    for b in ALL_BASES):
                    raise RuntimeError("pressure feedback stopped or became stale")
                output = _read(self._command)
                generated, observed, count, solve_ms, late = output[:5]
                if count != last_command and generated > 0:
                    if now - observed > STATE_TIMEOUT_S or now - generated > COMMAND_TIMEOUT_S:
                        raise RuntimeError("controller returned an obsolete command")
                    targets = output[5:]
                    psi = quantized_safe_targets(targets / PA_PER_PSI, nodes, envelope)
                    with self._apply_lock:
                        if self._stop.is_set():
                            break
                        self.backend.set_targets(dict(zip(ALL_BASES, psi)))
                        if not enabled:
                            self.backend.set_enabled_all(True)
                            enabled = True
                    self.last_pressure_pa = psi * PA_PER_PSI
                    self.commands += 1
                    self.worker_commands = int(count)
                    self.solve_ms = float(solve_ms)
                    self.solve_times_ms.append(self.solve_ms)
                    if last_command:
                        self.command_intervals_ms.append((now - last_arrival) * 1000)
                    self.deadline_misses = int(late)
                    if not self._first_command:
                        self._first_command = now
                    self.command_hz = ((self.commands - 1) / max(now - self._first_command, 1e-6))
                    self.status = "running"
                    last_command, last_arrival = count, now
                if enabled and now - last_arrival > COMMAND_TIMEOUT_S:
                    raise RuntimeError("controller command watchdog expired")
                if ready and not enabled and now - self._started > 30:
                    raise RuntimeError("controller produced no command")
                # Forward computed commands promptly; expensive node copies
                # above still happen only once per new camera observation.
                self._stop.wait(0.001)
            if not self.error and self.status != "stopped":
                self.status = "controller stopped"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "stopped: " + self.error
            if self.log is not None:
                self.log("[controller] " + self.status)
        finally:
            # A child sets stop on exit. Drain its final diagnostic here too,
            # because that event may end the loop before its next queue poll.
            try:
                while True:
                    kind, message = self._messages.get_nowait()
                    if kind == "error":
                        self.error = message
                        self.status = "stopped: " + message
            except queue.Empty:
                pass
            self._stop.set()
            with self._apply_lock:
                self.backend.stop_all()
                self.backend.set_targets(dict.fromkeys(ALL_BASES, 0.0))

    def stop(self):
        """Disable now, then reap children; no command can re-enable afterwards."""
        self._stop.set()
        with self._apply_lock:
            self.backend.stop_all()
            self.backend.set_targets(dict.fromkeys(ALL_BASES, 0.0))
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=1.0)
        if self.process is not None and self.process.pid is not None:
            self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        self._messages.close()
        self.status = "stopped" if not self.error else "stopped: " + self.error


__all__ = ["ControllerBridge", "METHODS", "JOINT_LIMIT_RAD", "validate_joint_target"]
