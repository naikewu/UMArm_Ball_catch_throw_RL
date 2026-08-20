"""The host side of :mod:`UMArm_KINOVA.arm_bridge` — a real Gen3, from anywhere.

THIS MODULE IMPORTS NOTHING FROM ``kortex_api`` AND MUST NOT.  That is its whole
reason for existing: the driver lives only in ``.venv_kinova`` (protobuf 3.5.1),
the collision bench's GUI runs under the ordinary interpreter, and the two
cannot share a process.  :class:`KinovaLink` spawns the bridge on the venv's
interpreter and talks to it in JSON lines, so any ordinary-interpreter caller —
the bench panel, a notebook, a test — can drive the metal without importing the
thing that would break it.

Shaped like ``UMArm_COLLAB.gui.launcher.BenchLauncher`` on purpose, because it
is the same kind of object and the panel already knows how to use one: a child
process that can be started and stopped repeatedly while the GUI's event loop
keeps running, :meth:`send` that never blocks, :meth:`poll` that drains what
came back and advances the state, and :meth:`describe` for the status line.

    link = KinovaLink()
    link.start()                       # spawns the bridge; nothing moves
    link.send({"cmd": "connect"})      # opens the session; nothing moves
    link.send({"cmd": "arm", "on": True})
    link.send({"cmd": "jog", "d_xyz": [0, 0, 1], "speed": "slow"})

SAFETY LIVES IN THE BRIDGE, NOT HERE.  This class does not decide what is a
legal move; it forwards commands and reports state.  The envelope, the
per-command joint step limit, the arm/disarm gate and the one-motion-at-a-time
worker are all on the far side of the pipe, on the interpreter that is actually
holding the session — which is the only place a limit cannot be skipped by a
caller who forgot to use the wrapper.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)

#: Where :mod:`UMArm_KINOVA.setup_env` builds the interpreter that can import
#: the driver.  Named here rather than discovered, because a bridge started on
#: the WRONG interpreter fails with an import error the operator would have to
#: read the traceback to understand.
VENV_PYTHON = os.path.join(REPO, ".venv_kinova", "Scripts", "python.exe")
VENV_PYTHON_POSIX = os.path.join(REPO, ".venv_kinova", "bin", "python")

BRIDGE_SCRIPT = os.path.join(_HERE, "arm_bridge.py")

#: How long a bridge gets to exit politely before it is terminated, seconds.
SHUTDOWN_BUDGET_S = 3.0

#: A state older than this means the bridge has stopped answering, seconds.
#: Four publish periods at the bridge's 10 Hz, so one dropped sample is not a
#: fault and a hung session is.
STALE_S = 0.5

IDLE = "idle"
STARTING = "starting"
RUNNING = "running"
STOPPING = "stopping"


def venv_python() -> str | None:
    """The interpreter that can talk to the Gen3, or ``None`` if it is missing."""
    for path in (VENV_PYTHON, VENV_PYTHON_POSIX):
        if os.path.isfile(path):
            return path
    return None


class KinovaLink:
    """One bridge process at a time, and the last state it published.

    Not thread-safe by design: every method is meant to be called from one event
    loop, exactly as ``BenchLauncher`` is.  The reader that drains the child's
    stdout IS a thread, and it only ever puts on a queue.
    """

    def __init__(self, *, python: str | None = None,
                 script: str = BRIDGE_SCRIPT, extra_args=()):
        self.python = python
        self.script = script
        self.extra_args = list(extra_args)

        self.state = IDLE
        self.error = ""
        self.log: list[str] = []
        #: The bridge's last ``state`` message, or ``{}`` before the first one.
        self.arm_state: dict = {}
        #: When it arrived, monotonic.
        self.arm_state_t: float = 0.0

        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._reader: threading.Thread | None = None
        self._deadline: float | None = None

    # ------------------------------------------------------------------

    @property
    def live(self) -> bool:
        return self.state == RUNNING

    @property
    def busy(self) -> bool:
        return self.state in (STARTING, STOPPING)

    @property
    def connected(self) -> bool:
        return bool(self.live and self.arm_state.get("connected"))

    @property
    def armed(self) -> bool:
        """Connected AND armed — the state the panel paints red.

        Both halves, deliberately: ``armed`` alone is meaningless once the
        session has gone, and a red border on a disconnected arm trains the
        operator to ignore the border.
        """
        return bool(self.connected and self.arm_state.get("armed"))

    @property
    def fresh(self) -> bool:
        return bool(self.arm_state
                    and (time.monotonic() - self.arm_state_t) < STALE_S)

    def describe(self) -> str:
        if self.state == IDLE:
            return "real arm: not connected" + (f" — {self.error}"
                                                if self.error else "")
        if self.state == STARTING:
            return "real arm: starting the bridge..."
        if self.state == STOPPING:
            return "real arm: closing the session..."
        if not self.arm_state.get("connected"):
            return "real arm: bridge up, session closed"
        busy = self.arm_state.get("busy") or ""
        where = "ARMED" if self.arm_state.get("armed") else "safe (not armed)"
        stale = "" if self.fresh else "   NOT ANSWERING"
        return (f"real arm: {where}"
                + (f", {busy} running" if busy else "") + stale)

    # ------------------------------------------------------------------

    def start(self) -> str:
        """Spawn the bridge.  Returns a line for the log; never raises."""
        if self.state != IDLE:
            return f"the bridge is already {self.state}"
        py = self.python or venv_python()
        if py is None:
            self.error = (".venv_kinova is missing — build it with "
                          "`python UMArm_KINOVA/setup_env.py`")
            return self.error
        if not os.path.isfile(self.script):
            self.error = f"no bridge script at {self.script}"
            return self.error
        try:
            # stderr is folded into the log rather than into stdout: stdout is
            # the JSON channel and one traceback line on it would desynchronise
            # every message after it.
            self._proc = subprocess.Popen(
                [py, self.script, *self.extra_args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, cwd=REPO)
        except Exception as exc:
            self.error = f"could not start the bridge: {type(exc).__name__}: {exc}"
            self._proc = None
            return self.error
        self.error = ""
        self.arm_state = {}
        self.arm_state_t = 0.0
        self.state = STARTING
        self._reader = threading.Thread(target=self._read, name="kinova-link",
                                        daemon=True)
        self._reader.start()
        self._err_reader = threading.Thread(target=self._read_err,
                                            name="kinova-link-err", daemon=True)
        self._err_reader.start()
        return f"bridge starting on {os.path.basename(os.path.dirname(py))}"

    def _read(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    msg = {"kind": "log", "line": line}
                if not isinstance(msg, dict):
                    # A LINE THAT IS VALID JSON BUT NOT AN OBJECT.  The bridge's
                    # stdout is shared with whatever the vendored driver and
                    # kortex_api print, and a bare list or number parses to a
                    # list or an int -- on which ``msg.get`` raises, out of
                    # ``poll``, out of the panel's repaint, and out of the
                    # tkinter ``after`` callback, which is then never
                    # rescheduled.  One stray line used to freeze the whole GUI
                    # with the red border painted on.
                    msg = {"kind": "log", "line": line}
                try:
                    self._q.put_nowait(msg)
                except queue.Full:
                    pass
        except Exception:
            pass

    def _read_err(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                try:
                    self._q.put_nowait({"kind": "log", "line": f"[bridge] {line}"})
                except queue.Full:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------

    def send(self, msg: dict) -> None:
        """Write one command.  Dropped silently if no bridge is listening."""
        proc = self._proc
        if proc is None or proc.stdin is None or self.state not in (STARTING,
                                                                   RUNNING):
            return
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except Exception:
            pass

    def stop_motion(self) -> None:
        """The one command that must always get through."""
        self.send({"cmd": "stop"})

    def poll(self) -> list[dict]:
        """Drain the bridge's messages and advance the state machine."""
        msgs: list[dict] = []
        try:
            while True:
                msgs.append(self._q.get_nowait())
        except queue.Empty:
            pass

        for msg in msgs:
            if not isinstance(msg, dict):        # belt and braces
                continue
            kind = msg.get("kind")
            if kind == "state":
                self.arm_state = msg
                self.arm_state_t = time.monotonic()
                if self.state == STARTING:
                    self.state = RUNNING
                if msg.get("error"):
                    self.error = str(msg["error"])
            elif kind == "error":
                self.error = f"{msg.get('cmd', '')}: {msg.get('error', '')}"
                self.log.append(self.error)
            elif kind == "log":
                self.log.append(str(msg.get("line", "")))

        proc = self._proc
        if proc is not None and proc.poll() is not None:
            if self.state in (STARTING, RUNNING):
                self.error = self.error or "the bridge exited on its own"
            self._reap()
        elif self.state == STOPPING and self._deadline is not None:
            if time.monotonic() > self._deadline:
                try:
                    proc.terminate()
                except Exception:
                    pass
                self._reap()
        return msgs

    def request_stop(self) -> None:
        """Ask the bridge to close the session and exit.  Never blocks."""
        if self.state in (IDLE, STOPPING):
            return
        # STOP FIRST, then quit.  A bridge asked to exit while a move is in
        # flight would close the session under a moving arm; ``stop`` aborts the
        # action, and the bridge's own ``finally`` closes the session after.
        self.stop_motion()
        self.send({"cmd": "quit"})
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        self.state = STOPPING
        self._deadline = time.monotonic() + SHUTDOWN_BUDGET_S

    def _reap(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.state = IDLE
        self._deadline = None
        self.arm_state = {}

    def shutdown(self) -> None:
        """Blocking stop, for a window-close handler."""
        if self.state == IDLE:
            return
        self.request_stop()
        deadline = time.monotonic() + SHUTDOWN_BUDGET_S + 1.0
        while self.state != IDLE and time.monotonic() < deadline:
            self.poll()
            time.sleep(0.05)
        if self.state != IDLE:
            self._reap()


__all__ = ["KinovaLink", "venv_python", "VENV_PYTHON", "BRIDGE_SCRIPT",
           "IDLE", "STARTING", "RUNNING", "STOPPING", "STALE_S",
           "SHUTDOWN_BUDGET_S"]
