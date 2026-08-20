"""Minimal MPC-style client for the VNEMA portable synchronized communication layer.

This is the reference example for the use case the bundle is built for: a controller
(MPC, RL policy, teleop, system-ID script, ...) that needs to

  1. read the synchronized robot state ``q`` (12), ``qdot`` (12) and per-actuator
     pressure at the 150 Hz CAN sync mark, and
  2. publish a full target-pressure vector back to the actuators every cycle.

The backend (``vnema_backend.exe``) owns the real-time 150 Hz loop, the CAN
transport, the mocap clock alignment and the mocap->joint observer. This client
talks to it over a tiny line-delimited JSON protocol on stdin/stdout, so the
controller can be written in any language. See ``docs/mpc_integration.md`` for the
full protocol and the ``robot_state`` schema.

Run against the built-in simulator (no hardware, no mocap rig required)::

    python examples/mpc_client_example.py --seconds 3

Run against real hardware + live mocap::

    python examples/mpc_client_example.py --live --port COM4 \
        --mocap-server 192.168.1.100 --mocap-local 192.168.1.120 \
        --enable-outputs --seconds 10

The "controller" here is a deliberate stub: it holds every actuator at a constant
target so the wiring is obvious. Replace ``compute_targets`` with your MPC solve.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Pressure targets are 12-bit ADC counts. 0 == empty, 4095 == full scale.
ADC_MIN = 0
ADC_MAX = 4095

DEFAULT_BACKEND = Path(__file__).resolve().parent.parent / "host" / "pc_backend" / "build" / "vnema_backend.exe"
DEFAULT_MOCAP_SCRIPT = Path(__file__).resolve().parent.parent / "mocap" / "mocap.py"


@dataclass
class RobotState:
    """One 150 Hz synchronized sample, decoded from a ``robot_state`` JSON line."""

    cycle: int
    can_sync_time_s: float
    ids: list[int]
    q: list[float]
    qdot: list[float]
    pressure_adc: list[int]
    target_adc: list[int]
    joint_valid: bool
    fk_tip: list[float]
    mocap_stale: bool
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict) -> "RobotState":
        return cls(
            cycle=int(obj.get("cycle", 0)),
            can_sync_time_s=float(obj.get("can_sync_time_s", 0.0)),
            ids=[int(x) for x in obj.get("ids", [])],
            q=[float(x) for x in obj.get("joint_current_theta", [])],
            qdot=[float(x) for x in obj.get("joint_current_theta_dot", [])],
            pressure_adc=[int(x) for x in obj.get("pressure_adc_filtered", [])],
            target_adc=[int(x) for x in obj.get("target_next_sync", [])],
            joint_valid=bool(obj.get("joint_current_valid", False)),
            fk_tip=[float(x) for x in obj.get("fk_tip", [])],
            mocap_stale=bool(obj.get("mocap_stale", True)),
            raw=obj,
        )


class VnemaBackend:
    """Launch and drive ``vnema_backend.exe`` over its JSON stdin/stdout protocol.

    Copy this class into your project as the integration seam. It keeps the most
    recent ``robot_state`` available without blocking your control loop on I/O.
    """

    def __init__(self, args: list[str]) -> None:
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line buffered
        )
        self._latest: RobotState | None = None
        self._lock = threading.Lock()
        self._state_count = 0
        self._errors: queue.Queue[str] = queue.Queue()
        self._stop_reader = threading.Event()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    # ---- background readers -------------------------------------------------
    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = obj.get("type")
            if msg_type == "robot_state":
                state = RobotState.from_json(obj)
                with self._lock:
                    self._latest = state
                    self._state_count += 1
            elif msg_type == "error":
                self._errors.put(str(obj.get("message", "")))
            if self._stop_reader.is_set():
                break

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            sys.stderr.write(f"[backend stderr] {line}")

    # ---- command sending ----------------------------------------------------
    def _send(self, obj: dict) -> None:
        if self._proc.stdin is None or self._proc.poll() is not None:
            raise RuntimeError("backend process is not accepting commands")
        self._proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()

    def start_loop(self) -> None:
        self._send({"cmd": "start"})

    def enable_outputs(self, enable: bool) -> None:
        self._send({"cmd": "enable_outputs", "enable": enable})

    def set_targets(self, targets: list[int], enable: bool | None = None) -> None:
        """Publish one target-pressure vector aligned to ``robot_state.ids`` order."""
        msg: dict = {"cmd": "set_targets", "targets": [int(t) for t in targets]}
        if enable is not None:
            msg["enable"] = bool(enable)
        self._send(msg)

    def stop_loop(self) -> None:
        self._send({"cmd": "stop"})

    # ---- state access -------------------------------------------------------
    def latest(self) -> RobotState | None:
        with self._lock:
            return self._latest

    @property
    def state_count(self) -> int:
        with self._lock:
            return self._state_count

    def drain_errors(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._errors.get_nowait())
            except queue.Empty:
                break
        return out

    # ---- lifecycle ----------------------------------------------------------
    def shutdown(self, timeout: float = 5.0) -> None:
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
        self._stop_reader.set()

    def __enter__(self) -> "VnemaBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def compute_targets(state: RobotState, setpoint_adc: int) -> list[int]:
    """STUB controller -> replace with your MPC solve.

    Inputs available for the real controller:
      * ``state.q``           : 12 joint angles [rad] at the sync mark
      * ``state.qdot``        : 12 joint velocities [rad/s]
      * ``state.pressure_adc``: measured pressure per actuator [ADC counts]
      * ``state.fk_tip``      : forward-kinematics tip position [m]

    Output: one target-pressure (ADC counts) per actuator, in ``state.ids`` order.
    Here we simply hold every actuator at ``setpoint_adc``.
    """
    n = len(state.ids)
    return [max(ADC_MIN, min(ADC_MAX, setpoint_adc)) for _ in range(n)]


def build_backend_args(opts: argparse.Namespace) -> list[str]:
    args = [str(opts.backend), "--ids", opts.ids, "--rate", str(opts.rate), "--stream-state",
            "--log-dir", str(opts.log_dir)]
    if opts.live:
        args += ["--port", opts.port, "--tty-baud", str(opts.tty_baud)]
        if opts.mocap_server:
            args += ["--mocap-live", "--mocap-python", opts.mocap_python,
                     "--mocap-script", str(opts.mocap_script),
                     "--mocap-server", opts.mocap_server, "--mocap-local", opts.mocap_local,
                     "--mocap-rigid-ids", opts.mocap_rigid_ids]
        else:
            args += ["--mocap-sim"]
    else:
        args += ["--simulate-can", "--mocap-sim"]
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="MPC-style example client for the VNEMA portable comm layer")
    parser.add_argument("--backend", type=Path, default=DEFAULT_BACKEND, help="Path to vnema_backend.exe")
    parser.add_argument("--ids", default="0x101-0x118", help="Actuator ID set, e.g. 0x101-0x118")
    parser.add_argument("--rate", type=int, default=150, help="Backend cycle rate (Hz)")
    parser.add_argument("--seconds", type=float, default=3.0, help="How long to run the control loop")
    parser.add_argument("--setpoint", type=int, default=1200, help="Stub controller target pressure (ADC counts)")
    parser.add_argument("--enable-outputs", action="store_true", help="Allow the firmware to actuate (default: safe/off)")
    parser.add_argument("--log-dir", type=Path, default=Path("reports"), help="Backend CSV/report directory")
    parser.add_argument("--live", action="store_true", help="Use real SLCAN hardware instead of the simulator")
    parser.add_argument("--port", default="COM4", help="SLCAN serial port (live)")
    parser.add_argument("--tty-baud", type=int, default=2000000, help="SLCAN serial baud (live)")
    parser.add_argument("--mocap-python", default="python", help="Python used to run the mocap bridge (live)")
    parser.add_argument("--mocap-script", type=Path, default=DEFAULT_MOCAP_SCRIPT, help="mocap.py path (live)")
    parser.add_argument("--mocap-server", default="", help="Motive/NatNet server IP (live; empty -> mocap sim)")
    parser.add_argument("--mocap-local", default="127.0.0.1", help="Local interface IP for NatNet (live)")
    parser.add_argument("--mocap-rigid-ids", default="1000-1005", help="Rigid-body IDs (live)")
    opts = parser.parse_args()

    if not Path(opts.backend).exists():
        parser.error(f"backend not found: {opts.backend}\nBuild it first (see docs/install.md).")

    args = build_backend_args(opts)
    print("launching:", " ".join(args), flush=True)

    with VnemaBackend(args) as backend:
        backend.start_loop()
        if opts.enable_outputs:
            backend.enable_outputs(True)

        # Wait for the first valid synchronized state before commanding.
        deadline = time.perf_counter() + 5.0
        while backend.latest() is None and time.perf_counter() < deadline:
            time.sleep(0.005)
        if backend.latest() is None:
            print("ERROR: no robot_state received from backend", file=sys.stderr)
            return 1

        # Control loop: act once per new 150 Hz cycle.
        last_cycle = -1
        run_deadline = time.perf_counter() + opts.seconds
        commands_sent = 0
        valid_states = 0
        last_valid: RobotState | None = None
        while time.perf_counter() < run_deadline:
            state = backend.latest()
            if state is None or state.cycle == last_cycle:
                time.sleep(0.0005)
                continue
            last_cycle = state.cycle
            if state.joint_valid:
                valid_states += 1
                last_valid = state

            targets = compute_targets(state, opts.setpoint)
            backend.set_targets(targets, enable=opts.enable_outputs)
            commands_sent += 1

            if commands_sent % opts.rate == 0 and last_valid is not None:  # ~1 Hz console print
                shown = last_valid  # show the most recent VALID estimate
                q_preview = ", ".join(f"{v:+.3f}" for v in shown.q[:4])
                print(
                    f"cycle={shown.cycle} joint_valid=True "
                    f"q[0:4]=[{q_preview}] tip={['%.3f' % v for v in shown.fk_tip]} "
                    f"p[0]={shown.pressure_adc[0] if shown.pressure_adc else 'NA'}",
                    flush=True,
                )

        backend.stop_loop()
        time.sleep(0.2)
        errors = backend.drain_errors()

    print(
        f"\nSummary: received {backend.state_count} states, sent {commands_sent} target vectors, "
        f"{valid_states} cycles had a valid joint estimate."
    )
    if errors:
        print(f"backend reported {len(errors)} error message(s); first: {errors[0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
