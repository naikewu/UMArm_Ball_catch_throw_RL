r"""``physics="process"`` -- the twin's plant in its own interpreter.

WHY THIS EXISTS.  The operator's requirement was that a controller driving the
SIM adapter cannot tell the twin from the arm.  With the plant stepped in the GUI
process, it could, from the timing alone.  ``SimCanLink`` advanced the physics
inside ``Backend``'s own ``send_batch`` call, on the cycle thread, and the fitted
twin costs about 0.5 s of CPU per simulated second -- 3.3 ms of every 6.67 ms
period, on the interpreter lock the Tk loop, the 24-board plot and the twin mocap
producer also need.  Measured 2026-09-10 in ``hw_tests/gui_sim_test.py`` with one
board at 12 psi: 87-107 Hz achieved and 35-59 % of that board's replies missed
(an independent re-run: 97.6 Hz, 256 replies and 237 misses), against the metal
window's 145.43 Hz and 98.81 % of 24 boards' replies
(``hw_tests/results/integrated_gui_profile_2026-08-20.json``, full window).
``Backend.stats``, ``missing_nodes`` and ``NodeState.consecutive_misses`` all
exposed the difference.

WHAT MOVES.  Only the plant.  ``SimMaster`` is still ``Backend`` with one member
replaced, so the cycle, the table encoder, the reply matching, the statistics
and every method a controller calls stay the metal's code; ``self.link`` is now a
:class:`SimProcessLink`.  Its host half does what a slcan link's host half does
-- encode a batch and write it, and hand the taps what comes back -- while
:func:`plant_main`, in a spawned child, owns the ``SimArm``:

* the child **free-runs** its physics against ``time.perf_counter()``.  On
  Windows that is ``QueryPerformanceCounter``, one counter for every process on
  the machine (on Linux, ``CLOCK_MONOTONIC``), so a stamp the host's cycle thread
  takes names the same instant in the child and no clock offset is estimated;
* a batch arrives in order -- table frames staged, then the sync edge, which
  advances to the child's own ``now``, latches every board, and sends the 24
  reply payloads straight back.  The edge is applied when the child receives it,
  a fraction of a millisecond after the host's write, which is also how the
  metal's boards see an edge that crossed a USB adapter;
* the host stamps each reply with ``t_batch0 + reply_latency_ms(id)`` exactly as
  the in-process link does (``SimCanLink._hand_over``), so the latency column is
  unchanged, and a reply due after the receive window is still missed;
* joint angles cross in shared memory under a sequence lock, 12 doubles, so the
  GUI's twin mocap reads the plant without a round trip;
* the TLE sync-loss failsafe needs no idle advance here: the child's clock never
  stops, whatever the host does.

WHAT THIS DOES NOT CHANGE, OR SHOW.  The physics is the same ``SimArm`` with the
same keywords; ``test_sim_process`` pins that a process twin and an in-process
twin driven by the same targets agree.  It is not bit-identical to an
in-process rollout of the same wall-clock session, because the child's advance
granularity follows its own scheduler, and nothing that needs determinism --
``replay``, ``mech_fit``, ``twin_compare`` -- goes through it.  A child that falls
behind real time (a machine loaded past one free core) answers late, and the
host then misses replies exactly as it would on an overloaded bus; nothing hides
that.  ``arm=`` and ``node_hook=`` cannot cross a process boundary and are
refused.

RS485 ORIGINAL: none -- the RS485 twin ran in a process of its own by design and
had no GUI to share an interpreter with.
"""

from __future__ import annotations

import inspect
import math
import multiprocessing as mp
import random
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from . import sim_core
from . import sim_master as SM

P = SM.P

#: s.  How long :meth:`SimProcessLink.open` waits for the child to build its
#: ``SimArm`` and report ready.  A spawned child re-imports numpy, MuJoCo and the
#: twin, and on this machine it is ready in about 2-3 s; the GUI's own window
#: module is re-imported too when the GUI is the parent.  60 s is a hung child,
#: not a slow one.
SPAWN_TIMEOUT_S = 60.0

#: s.  The child's longest wait for a command before it advances the physics to
#: ``now`` on its own.  1 ms is one physics quantum: the plant never lags wall
#: time by more than that plus one advance.
PLANT_IDLE_S = 0.001

#: s.  How long a ``snapshot_arm`` request waits for the child.
REQUEST_TIMEOUT_S = 5.0

#: Shared-memory layout, float64: sequence, sim_now, quanta_done, qpos[nq].
_SEQ, _SIM_NOW, _QUANTA, _QPOS0 = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------

def _refuse_hardware() -> None:
    """The plant process has no business with a serial port; make sure it cannot."""
    try:
        import serial  # type: ignore
    except Exception:                                        # pragma: no cover
        return

    def refuse(*_a, **_k):
        raise RuntimeError("the twin's plant process never opens a serial port")

    serial.Serial = refuse


def _publish(view: np.ndarray, arm) -> None:
    """Write sim time and ``qpos`` under a sequence lock (odd = being written)."""
    view[_SEQ] += 1.0
    view[_SIM_NOW] = arm.sim_now
    view[_QUANTA] = float(arm.quanta_done)
    view[_QPOS0:_QPOS0 + arm.model.nq] = arm.data.qpos
    view[_SEQ] += 1.0


def _replies(arm, rng: random.Random, counters: dict) -> list:
    out = []
    for base, node in sorted(arm.nodes.items()):
        if node.fault.reply_dropout and rng.random() < node.fault.reply_dropout:
            counters["dropped"] += 1
            continue
        status = node.compact_status()
        word = (status.counts & P.PRESSURE_MASK) | ((status.flags & P.FLAGS_MASK) << 12)
        out.append((base, struct.pack("<H", word)))
    return out


def _snapshot(arm) -> dict:
    with arm.lock:
        return {base: dict(p_pa=node.p_pa, true_psi=node.true_psi, action=node.action,
                           enabled=node.enabled, failsafe_active=node.failsafe_active,
                           target_pa=node.target_pa, variant=node.variant,
                           sync_loss_count=getattr(node, "sync_loss_count", None))
                for base, node in sorted(arm.nodes.items())}


def plant_main(cmd, rep, shm_name: str, xml: str, arm_kwargs: dict,
               seed: int, idle_s: float = PLANT_IDLE_S) -> None:
    """The child: build the arm, free-run it, answer batches, publish ``qpos``."""
    shm = None
    try:
        _refuse_hardware()
        try:
            from TLE_PCB.tlelib.timing import hires_clock
        except ImportError:                                  # pragma: no cover
            from tlelib.timing import hires_clock            # type: ignore
        hires_clock()
        arm = sim_core.SimArm(xml=xml, seed=seed, **arm_kwargs)
        shm = shared_memory.SharedMemory(name=shm_name)
        view = np.ndarray((_QPOS0 + arm.model.nq,), dtype=np.float64, buffer=shm.buf)
        clock = time.perf_counter
        arm.advance_to(clock())
        _publish(view, arm)
        rep.send(("ready", {
            "nodes": {int(b): (int(n.variant), bool(n.is_tle)) for b, n in arm.nodes.items()},
            "q_order": arm.q_order, "nq": int(arm.model.nq), "pid": mp.current_process().pid}))
        rng = random.Random(seed)
        counters = {"sync": 0, "dropped": 0}
        while True:
            if cmd.poll(idle_s):
                msg = cmd.recv()
                kind = msg[0]
                if kind == "stop":
                    break
                if kind == "batch":
                    _, ops, t0 = msg
                    for op in ops:
                        if op[0] == "stage":
                            arm.stage_targets(op[1])
                        elif op[0] == "edge":
                            now = clock()
                            with arm.lock:
                                arm.advance_to(now)
                                arm.sync_edge(now)
                                counters["sync"] += 1
                                replies = _replies(arm, rng, counters)
                            rep.send(("replies", t0, now, replies,
                                      counters["sync"], counters["dropped"]))
                elif kind == "snapshot":
                    rep.send(("snapshot", msg[1], _snapshot(arm)))
            arm.advance_to(clock())
            _publish(view, arm)
    except (EOFError, OSError, BrokenPipeError):
        pass                              # the host went away: nothing to answer
    except BaseException:                 # noqa: BLE001 - reported, then exit
        try:
            rep.send(("error", traceback.format_exc()))
        except Exception:                                    # pragma: no cover
            pass
    finally:
        if shm is not None:
            shm.close()


# ---------------------------------------------------------------------------
# The host side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeInfo:
    """What a scan needs of a board the child owns: its variant byte."""

    variant: int
    is_tle: bool


class _RemoteData:
    """``arm.data`` for a consumer that reads ``qpos`` (``SimMocap``)."""

    def __init__(self, arm: "RemoteArm") -> None:
        self._arm = arm

    @property
    def qpos(self) -> np.ndarray:
        return self._arm.read_qpos()


class RemoteArm:
    """The host's view of a plant that lives in :func:`plant_main`.

    Carries what the host legitimately reads of the twin without a round trip:
    the compiled ``model`` (compiled here from the very XML the child builds, for
    ``SimMocap``'s plate sites), ``data.qpos`` and :meth:`q` from shared memory,
    ``sim_now``, and ``nodes`` with each board's variant for the scan.  It has no
    ``advance_to``: nothing on the host moves this plant's clock.
    """

    def __init__(self, xml: str) -> None:
        import mujoco

        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        #: Local, guards nothing in the child: kept so a reader written against
        #: ``SimArm`` (``with arm.lock: arm.data.qpos``) runs unchanged.
        self.lock = threading.RLock()
        self.data = _RemoteData(self)
        self.nodes: dict = {}
        self.q_order = None
        self.pid = None
        self._view = None
        self._q_adr = _q_qposadr(self.model)

    # -- attach/detach, by the link ------------------------------------------
    def _attach(self, shm, info: dict) -> None:
        self._view = np.ndarray((_QPOS0 + int(info["nq"]),), dtype=np.float64,
                                buffer=shm.buf)
        self.nodes = {int(b): NodeInfo(int(v), bool(t))
                      for b, (v, t) in info["nodes"].items()}
        self.q_order = info.get("q_order")
        self.pid = info.get("pid")

    def _detach(self) -> None:
        self._view = None

    # -- reads -----------------------------------------------------------------
    def _read(self):
        view = self._view
        if view is None:
            return 0.0, 0.0, np.zeros(self.model.nq)
        for _ in range(1000):
            s1 = float(view[_SEQ])
            if int(s1) % 2:
                time.sleep(0)
                continue
            qpos = np.array(view[_QPOS0:_QPOS0 + self.model.nq], dtype=float)
            sim_now = float(view[_SIM_NOW])
            quanta = float(view[_QUANTA])
            if float(view[_SEQ]) == s1:
                return sim_now, quanta, qpos
        raise RuntimeError("the plant process kept its state mid-write for 1000 reads")

    def read_qpos(self) -> np.ndarray:
        return self._read()[2]

    def q(self) -> np.ndarray:
        """Joint angles in ``UMArm_KINEMATICS`` order, as ``SimArm.q``."""
        qpos = self.read_qpos()
        return qpos if self._q_adr is None else qpos[self._q_adr]

    @property
    def sim_now(self) -> float:
        return self._read()[0]

    @property
    def quanta_done(self) -> int:
        return int(self._read()[1])


def _q_qposadr(model):
    import mujoco

    from . import mjcf_generator as MG

    names = MG.joint_names()
    out = []
    for i in range(len(MG.QPOS_FROM_Q)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[MG.QPOS_FROM_Q[i]])
        if jid < 0:
            return None
        out.append(int(model.jnt_qposadr[jid]))
    return np.asarray(out, dtype=int)


class SimProcessLink(SM.SimCanLink):
    """``SimCanLink`` whose plant is :func:`plant_main` in a spawned child.

    The inherited routing, reply stamping, delivery thread and scan are used
    as they are; what changes is where a staged table and a sync edge go, and
    who keeps time.
    """

    def __init__(self, arm: RemoteArm, port: str = "SIM", bitrate: int = 1_000_000, *,
                 arm_kwargs: dict, seed: int = 20260910,
                 spawn_timeout_s: float = SPAWN_TIMEOUT_S, **link_kwargs) -> None:
        if link_kwargs.get("clock", time.perf_counter) is not time.perf_counter:
            raise ValueError("physics='process' needs time.perf_counter: the child "
                             "and the host must read one clock")
        if link_kwargs.get("advance", True) is not True:
            raise ValueError("physics='process' always advances: the child free-runs")
        super().__init__(arm, port, bitrate, seed=seed, **link_kwargs)
        self._arm_kwargs = dict(arm_kwargs)
        self._seed = int(seed)
        self.spawn_timeout_s = float(spawn_timeout_s)
        self._proc = None
        self._cmd = None
        self._rep = None
        self._shm = None
        self._rx = None
        self._ops = None
        self._send_lock = threading.Lock()
        self._requests: dict = {}
        self._req_seq = 0
        #: The child's traceback, if it died of an exception.
        self.plant_error = None

    # ---- lifecycle ------------------------------------------------------
    def open(self) -> None:
        if self._proc is not None:
            return
        ctx = mp.get_context("spawn")
        cmd_r, cmd_w = ctx.Pipe(duplex=False)
        rep_r, rep_w = ctx.Pipe(duplex=False)
        nq = int(self.arm.model.nq)
        shm = shared_memory.SharedMemory(create=True, size=8 * (_QPOS0 + nq))
        np.ndarray((_QPOS0 + nq,), dtype=np.float64, buffer=shm.buf)[:] = 0.0
        proc = ctx.Process(target=plant_main, name="canarm-twin-plant", daemon=True,
                           args=(cmd_r, rep_w, shm.name, self.arm.xml,
                                 self._arm_kwargs, self._seed))
        proc.start()
        cmd_r.close()
        rep_w.close()
        try:
            if not rep_r.poll(self.spawn_timeout_s):
                raise RuntimeError(f"the twin's plant process did not report ready in "
                                   f"{self.spawn_timeout_s:.0f} s")
            msg = rep_r.recv()
            if msg[0] == "error":
                raise RuntimeError("the twin's plant process failed to start:\n" + msg[1])
            if msg[0] != "ready":
                raise RuntimeError(f"unexpected first message from the plant: {msg[0]!r}")
        except BaseException:
            proc.terminate()
            proc.join(timeout=2.0)
            for c in (cmd_w, rep_r):
                c.close()
            shm.close()
            shm.unlink()
            raise
        self._proc, self._cmd, self._rep, self._shm = proc, cmd_w, rep_r, shm
        self.arm._attach(shm, msg[1])
        self.adapter_version = f"SIM (plant pid {self.arm.pid})"
        self._rx = threading.Thread(target=self._receive, name="sim-plant-rx", daemon=True)
        self._rx.start()
        super().open()                    # the delivery thread

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            super().close()
            return
        try:
            self._post(("stop",))
        except RuntimeError:
            pass
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        super().close()
        for conn in (self._cmd, self._rep):
            try:
                conn.close()
            except Exception:                                # pragma: no cover
                pass
        if self._rx is not None:
            self._rx.join(timeout=1.0)
        self.arm._detach()
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:                                    # pragma: no cover
            pass
        self._proc = self._cmd = self._rep = self._shm = self._rx = None

    @property
    def is_open(self) -> bool:
        return (self._proc is not None and self._thread is not None
                and self._proc.is_alive())

    @property
    def plant_pid(self):
        return None if self._proc is None else self._proc.pid

    # ---- transmit --------------------------------------------------------
    def _post(self, msg) -> None:
        conn = self._cmd
        if conn is None:
            raise RuntimeError("the twin's plant process is not running")
        try:
            with self._send_lock:
                conn.send(msg)
        except (OSError, EOFError, BrokenPipeError, ValueError) as exc:
            raise RuntimeError(f"the twin's plant process is gone ({exc}); "
                               f"{self.plant_error or 'no traceback reported'}") from exc

    def send(self, can_id: int, data: bytes = b"") -> None:
        self._send_ops([(int(can_id), bytes(data))], self.clock())

    def send_batch(self, frames) -> None:
        """One write, as on the metal: the tables and the edge in one message.

        Timed from the batch's first frame for the reason the in-process link
        gives -- ``Backend._publish`` measures latency from a stamp taken before
        the table is built.
        """
        self._send_ops(frames, self.clock())

    def _send_ops(self, frames, t0: float) -> None:
        ops = []
        self._ops, self._batch_t0 = ops, t0
        try:
            for can_id, data in frames:
                self._route(int(can_id), bytes(data))
        finally:
            self._ops, self._batch_t0 = None, None
        if ops:
            self._post(("batch", ops, t0))

    def _stage(self, staged: dict) -> None:
        self._ops.append(("stage", staged))

    def _sync_edge(self) -> None:
        self._ops.append(("edge",))
        self._last_edge_t = self.clock()

    def _keep_time(self) -> None:
        """Nothing to do: the child's clock runs whether or not the host drives."""

    # ---- receive ---------------------------------------------------------
    def _receive(self) -> None:
        rep = self._rep
        while True:
            try:
                msg = rep.recv()
            except (EOFError, OSError):
                break
            kind = msg[0]
            if kind == "replies":
                _, t0, _t_edge, replies, sync_count, dropped = msg
                self.sync_count = int(sync_count)
                self.dropped_replies = int(dropped)
                self._hand_over(t0, replies)
            elif kind == "snapshot":
                slot = self._requests.get(msg[1])
                if slot is not None:
                    slot[1].append(msg[2])
                    slot[0].set()
            elif kind == "error":
                self.plant_error = msg[1]
                break

    def request_snapshot(self, timeout_s: float = REQUEST_TIMEOUT_S) -> dict:
        """The child's plant truth, as ``SimMaster.snapshot_arm`` returns it."""
        self._req_seq += 1
        rid = self._req_seq
        slot = (threading.Event(), [])
        self._requests[rid] = slot
        try:
            self._post(("snapshot", rid))
            if not slot[0].wait(timeout_s):
                raise RuntimeError("the plant process did not answer a snapshot request")
            return slot[1][0]
        finally:
            self._requests.pop(rid, None)


#: ``SimArm.__init__`` keywords that are the arm's own, as opposed to MJCF
#: tunables forwarded to ``generate_xml``.
SIMARM_OWN_KEYWORDS = tuple(
    name for name, p in inspect.signature(sim_core.SimArm.__init__).parameters.items()
    if name not in ("self",) and p.kind is p.KEYWORD_ONLY)


def split_simarm_kwargs(kwargs: dict) -> tuple:
    """``(arm_kwargs, mjcf_tunables)``: what the child's ``SimArm`` takes itself."""
    own = {k: v for k, v in kwargs.items() if k in SIMARM_OWN_KEYWORDS}
    tunables = {k: v for k, v in kwargs.items() if k not in SIMARM_OWN_KEYWORDS}
    return own, tunables


def build(port: str, bitrate: int, *, seed: int, link_kwargs: dict,
          simarm_kwargs: dict) -> tuple:
    """``(RemoteArm, SimProcessLink)`` for ``SimMaster(physics="process")``."""
    own, tunables = split_simarm_kwargs(simarm_kwargs)
    if own.pop("node_hook", None) is not None:
        raise ValueError("node_hook= is a callable in this process and cannot run in "
                         "the plant process; use physics='inline'")
    own.pop("seed", None)
    xml = own.pop("xml", None)
    if xml is None:
        xml = sim_core._generated_xml(**tunables)
    elif tunables:
        raise ValueError("xml= and MJCF tunables are mutually exclusive: a handed-in "
                         f"scene cannot honour {sorted(tunables)}")
    arm = RemoteArm(xml)
    link = SimProcessLink(arm, port, bitrate, arm_kwargs=own, seed=seed, **link_kwargs)
    return arm, link


__all__ = ["SPAWN_TIMEOUT_S", "PLANT_IDLE_S", "REQUEST_TIMEOUT_S", "NodeInfo",
           "RemoteArm", "SimProcessLink", "plant_main", "split_simarm_kwargs",
           "SIMARM_OWN_KEYWORDS", "build"]
