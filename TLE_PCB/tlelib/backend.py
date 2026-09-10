"""Backend: everything between the PC and the boards.

The split the controller GUI is built on is that this module owns the bus and
the frontend owns the human. Nothing above this line touches a serial port, and
nothing in here knows what a widget is -- which is also what makes the whole
control path testable from a script with no display attached.

The core is one real-time thread driving the 150 Hz cycle:

    build target table  ->  send table frames + sync in one write
                        ->  collect compact replies for the RX window
                        ->  publish the cycle

Table frames and the sync frame go out as a single serial write. The adapter
transmits them in order, so the sync is guaranteed to follow the targets it
belongs to and follows them by about a frame time rather than by however long
the next USB transaction takes to schedule.

A second, slow duty cycles through the selected TLE boards asking for their
extended telemetry, so the UI can show valve current, controller state and the
integral term without any of that traffic touching the real-time path.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from . import native as N
from . import proto as P
from .canlink import DEFAULT_PORT as DEFAULT_CAN_PORT
from .canlink import CanLink
from .timing import hires_clock, sleep_until

HISTORY_SECONDS = 60.0
# Fraction of the cycle spent listening for replies; the rest is the margin
# that keeps the next cycle's transmit on schedule. 0.82 is the value the
# original 24-board host runs, where replies were measured between 2.9 and
# 3.5 ms after the edge -- the spread is CAN arbitration, which orders the
# answers by node ID.
RX_WINDOW_FRAC = 0.82
# How often each selected TLE board is asked for its extended telemetry.
EXTENDED_POLL_HZ = 4.0
# A board that has answered no sync in this long is reported as missing.
NODE_TIMEOUT_S = 0.5


@dataclass
class NodeState:
    base: int
    kind: str = "unknown"
    version: str = ""
    cal: P.NodeCal = field(default_factory=P.NodeCal)
    present: bool = False

    # commanded
    enabled: bool = False
    target_psi: float = 0.0

    # measured, from the compact reply
    pressure_counts: int = 0
    pressure_psi: float = 0.0
    flags: int = 0
    replies: int = 0
    misses: int = 0
    consecutive_misses: int = 0
    last_reply_t: float = 0.0
    reply_latency_ms: float = 0.0

    # extended telemetry, TLE boards only
    extended: dict = field(default_factory=dict)
    extended_t: float = 0.0

    @property
    def is_tle(self) -> bool:
        return self.kind == P.VARIANT_NAMES[P.VARIANT_TLE_DVP]

    @property
    def error(self) -> bool:
        return bool(self.flags & P.STATUS_ERROR)

    @property
    def target_counts(self) -> int:
        return self.cal.psi_to_counts(self.target_psi)


@dataclass
class CycleStats:
    cycles: int = 0
    period_ms: float = 0.0
    jitter_ms_p95: float = 0.0
    jitter_ms_max: float = 0.0
    late_cycles: int = 0
    replies: int = 0
    misses: int = 0


class Backend:
    """Owns the CAN link, the node registry and the 150 Hz cycle."""

    def __init__(self, port: str = DEFAULT_CAN_PORT, bitrate: int = 1_000_000,
                 cycle_hz: float = P.CYCLE_HZ, log=None):
        self.link = CanLink(port, bitrate)
        self.cycle_hz = cycle_hz
        self.nodes: dict[int, NodeState] = {}
        self.selected: list[int] = []
        self.stats = CycleStats()
        self.log_lines: deque[str] = deque(maxlen=500)
        self._log = log
        self._lock = threading.RLock()
        self._history: dict[int, deque] = {}
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycle_lock = threading.Lock()
        self._cycle_replies: dict[int, tuple[float, P.CompactStatus]] = {}
        self._cycle_t0 = 0.0
        self._jitter: deque[float] = deque(maxlen=600)
        self._pending_native: deque[tuple[int, bytes]] = deque()
        self._extended_cursor = 0
        self._last_extended_poll = 0.0
        #: The target table actually put on the wire this cycle,
        #: ``{base: (counts, enable)}``.  Recorded because a data collection has
        #: to log what was *commanded*, and reading ``node.target_psi`` after
        #: the fact races the thread generating the excitation: by the time an
        #: observer looks, the next target may already be staged.
        self._cycle_targets: dict[int, tuple[int, bool]] = {}
        #: Optional per-cycle observer; see :meth:`set_cycle_observer`.
        self._on_cycle = None

    # ---- lifecycle -----------------------------------------------------
    def log(self, message: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {message}"
        self.log_lines.append(stamped)
        if self._log is not None:
            self._log(stamped)

    def open(self) -> None:
        hires_clock()
        self.link.open()
        self.link.add_tap(self._on_frame)
        self.log(f"CAN link open on {self.link.port} at {self.link.bitrate // 1000} kbit/s"
                 f" (adapter {self.link.adapter_version or 'unknown'})")

    def close(self) -> None:
        self.stop_cycle()
        self.link.remove_tap(self._on_frame)
        self.link.close()

    def __enter__(self) -> "Backend":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- discovery -----------------------------------------------------
    def scan(self, bases=None) -> dict[int, NodeState]:
        """Enumerate the bus. Safe to call while the cycle is stopped."""
        found = self.link.scan(bases)
        with self._lock:
            for base, version in found.items():
                node = self.nodes.get(base) or NodeState(base=base)
                node.kind = version.variant_name
                node.version = version.version
                node.present = True
                node.cal = P.NodeCal.for_variant(version.variant, base)
                self.nodes[base] = node
                self._history.setdefault(base, deque())
            for base, node in self.nodes.items():
                if base not in found:
                    node.present = False
        self.log(f"Scan found {len(found)} board(s): " +
                 ", ".join(f"0x{b:03X} {v.variant_name} {v.version}" for b, v in found.items())
                 if found else "Scan found no boards")
        return self.snapshot_nodes()

    def select(self, bases) -> None:
        with self._lock:
            self.selected = [b for b in bases if b in self.nodes]

    # ---- commands ------------------------------------------------------
    def set_target(self, base: int, psi: float) -> None:
        with self._lock:
            node = self.nodes.get(base)
            if node is not None:
                node.target_psi = psi

    def set_targets(self, mapping) -> None:
        """Retarget many boards under **one** lock acquisition.

        A whole-arm excitation rewrites all twenty-four setpoints every cycle,
        and doing that through :meth:`set_target` takes the lock twenty-four
        times per 6.67 ms against a cycle thread that needs it to publish.
        Worse, the twenty-four writes would not land in the same cycle's table:
        a table built halfway through the loop carries part of one command and
        part of the next, which is a whole-arm state nobody asked for.

        ``mapping`` is ``{base: psi}``; unknown bases are ignored, as in
        :meth:`set_target`.
        """
        with self._lock:
            for base, psi in mapping.items():
                node = self.nodes.get(base)
                if node is not None:
                    node.target_psi = float(psi)

    def set_target_all(self, psi: float) -> None:
        with self._lock:
            for base in self.selected:
                self.nodes[base].target_psi = psi

    def set_enabled(self, base: int, on: bool) -> None:
        with self._lock:
            node = self.nodes.get(base)
            if node is not None:
                node.enabled = on
        self.log(f"0x{base:03X}: {'enabled' if on else 'disabled'}")

    def set_enabled_all(self, on: bool) -> None:
        with self._lock:
            targets = list(self.selected)
            for base in targets:
                self.nodes[base].enabled = on
        self.log(f"{len(targets)} board(s) {'enabled' if on else 'disabled'}")

    def stop_all(self) -> None:
        """Drop every board, selected or not, and hold them dropped."""
        with self._lock:
            for node in self.nodes.values():
                node.enabled = False
        # _build_targets covers every known board, so this reaches boards that
        # are not in the current selection too -- but only while the cycle is
        # running. If it is not, send the disabling table and sync once here.
        if not self._running.is_set():
            try:
                self._send_cycle(disable_everything=True)
            except Exception as exc:
                self.log(f"[ERROR] disable did not reach the bus: {exc}")
        self.log("all boards disabled")

    def send_native(self, base: int, payload: bytes) -> None:
        """Queue an extended-protocol frame; the cycle thread sends it in the
        gap after the reply window so it cannot disturb sync timing."""
        with self._cycle_lock:
            self._pending_native.append((P.ids(base).extended, payload))
        if not self._running.is_set():
            self._flush_native()

    def apply_tuning(self, base: int, params: dict | None = None) -> None:
        for payload in N.tuning_frames(params):
            self.send_native(base, payload)
        self.log(f"0x{base:03X}: tuning applied")

    def request_extended(self, base: int) -> None:
        self.send_native(base, N.build_keepalive())

    # ---- the 150 Hz cycle ----------------------------------------------
    def start_cycle(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._cycle_loop, name="sync-master", daemon=True)
        self._thread.start()
        self.log(f"sync master running at {self.cycle_hz:.0f} Hz")

    def stop_cycle(self) -> None:
        """Stop driving, and leave every board disabled.

        Gated on the thread rather than the run flag: the cycle loop clears the
        flag itself when a transmit fails, and skipping the safe-disable in
        exactly that case would walk away from a bus where every board is still
        enabled and holding its last commanded pressure.
        """
        if not self._running.is_set() and self._thread is None:
            return
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._send_cycle(disable_everything=True)
            self.log("sync master stopped, all targets disabled")
        except Exception as exc:
            self.log(f"[ERROR] sync master stopped but the safe-disable did not "
                     f"reach the bus: {exc}. Boards may still be driving")

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def _on_frame(self, t: float, can_id: int, data: bytes) -> None:
        """Reader-thread tap: match replies to the cycle they belong to."""
        status = P.parse_compact_status(can_id, data)
        if status is not None:
            with self._cycle_lock:
                self._cycle_replies[status.base] = (t, status)
            return

        base = P.status_id_to_base(can_id)
        if base is None or not data:
            return
        if data[0] in (N.TLM_PRESSURE_CONTROL, N.TLM_PRESSURE_DVP,
                       N.TLM_DVP_TIMING, N.TLM_DVP_DEBUG):
            with self._lock:
                node = self.nodes.get(base)
                if node is not None:
                    N.decode_into(data, node.extended)
                    node.extended_t = t

    def _build_targets(self, disable_everything: bool = False) -> dict[int, tuple[int, bool]]:
        """One entry per known board, not per selected board.

        A board keeps whatever control byte it was last given: leaving it out of
        the table does not turn it off, it just stops updating it, and the sync
        edge goes on promoting the last enable it saw. So de-selecting a running
        board would leave it regulating with nothing able to reach it -- and the
        sync-loss failsafe cannot help either, because the master is still
        sending syncs. Every known board is addressed every cycle; the ones not
        selected are addressed with the enable bit clear.
        """
        with self._lock:
            targets = {}
            for base, node in self.nodes.items():
                if not node.present:
                    continue
                enable = node.enabled and not disable_everything and base in self.selected
                targets[base] = (node.target_counts, enable)
            return targets

    def _send_cycle(self, disable_everything: bool = False) -> None:
        targets = self._build_targets(disable_everything)
        self._cycle_targets = targets
        if not targets:
            return
        frames = P.build_runtime_table(targets)
        frames.append(P.build_sync())
        self.link.send_batch(frames)

    def _flush_native(self) -> None:
        with self._cycle_lock:
            pending = list(self._pending_native)
            self._pending_native.clear()
        for can_id, payload in pending:
            try:
                self.link.send(can_id, payload)
            except Exception as exc:
                self.log(f"[ERROR] extended frame to 0x{can_id:03X} failed: {exc}")

    def _poll_extended(self, now: float) -> None:
        with self._lock:
            tle_nodes = [b for b in self.selected if self.nodes[b].is_tle]
        if not tle_nodes:
            return
        if (now - self._last_extended_poll) < (1.0 / (EXTENDED_POLL_HZ * len(tle_nodes))):
            return
        self._last_extended_poll = now
        self._extended_cursor = (self._extended_cursor + 1) % len(tle_nodes)
        base = tle_nodes[self._extended_cursor]
        try:
            self.link.send(P.ids(base).extended, N.build_keepalive())
        except Exception:
            pass

    def _cycle_loop(self) -> None:
        period = 1.0 / self.cycle_hz
        rx_window = period * RX_WINDOW_FRAC
        next_tick = time.perf_counter() + period
        while self._running.is_set():
            sleep_until(next_tick)
            t_sync = time.perf_counter()

            drift_ms = (t_sync - next_tick) * 1000.0
            self._jitter.append(abs(drift_ms))
            if drift_ms > period * 1000.0 * 0.5:
                self.stats.late_cycles += 1

            with self._cycle_lock:
                self._cycle_replies = {}
                self._cycle_t0 = t_sync

            try:
                self._send_cycle()
            except Exception as exc:
                self.log(f"[ERROR] cycle transmit failed: {exc}")
                self._running.clear()
                break

            sleep_until(t_sync + rx_window)
            with self._cycle_lock:
                replies = dict(self._cycle_replies)
            self._publish(t_sync, replies)

            observer = self._on_cycle
            if observer is not None:
                try:
                    observer(t_sync, self._cycle_targets, replies)
                except Exception:
                    # The traceback, not just str(exc): an observer is a
                    # caller's code, and "IndexError: index 24 is out of
                    # bounds" without a line number is a fault that costs a
                    # whole hardware session to localise.
                    import traceback as _tb
                    self.log("[ERROR] cycle observer raised, uninstalling it: "
                             + _tb.format_exc().replace("\n", " | "))
                    self._on_cycle = None

            self._flush_native()
            self._poll_extended(t_sync)

            next_tick += period
            # A long stall (a debugger, a GC pause, the OS) must not turn into
            # a burst of catch-up cycles at full rate.
            now = time.perf_counter()
            if next_tick < now:
                next_tick = now + period

    def _publish(self, t_sync: float, replies: dict[int, tuple[float, P.CompactStatus]]) -> None:
        with self._lock:
            self.stats.cycles += 1
            for base in self.selected:
                node = self.nodes[base]
                entry = replies.get(base)
                if entry is None:
                    node.misses += 1
                    node.consecutive_misses += 1
                    self.stats.misses += 1
                    continue
                t_reply, status = entry
                node.replies += 1
                node.consecutive_misses = 0
                node.last_reply_t = t_reply
                node.reply_latency_ms = (t_reply - t_sync) * 1000.0
                node.pressure_counts = status.counts
                node.pressure_psi = node.cal.counts_to_psi(status.counts)
                node.flags = status.flags
                self.stats.replies += 1

                history = self._history.setdefault(base, deque())
                history.append((t_sync, node.pressure_psi, node.target_psi))
                cutoff = t_sync - HISTORY_SECONDS
                while history and history[0][0] < cutoff:
                    history.popleft()

            if self._jitter:
                ordered = sorted(self._jitter)
                self.stats.jitter_ms_p95 = ordered[int(len(ordered) * 0.95) - 1]
                self.stats.jitter_ms_max = ordered[-1]
                self.stats.period_ms = 1000.0 / self.cycle_hz

    # ---- readout -------------------------------------------------------
    def set_cycle_observer(self, callback) -> None:
        """Install a callback fired once per cycle, on the cycle thread.

        Signature ``callback(t_sync, targets, replies)`` where ``t_sync`` is the
        ``time.perf_counter()`` stamp of the sync edge, ``targets`` is
        ``{base: (counts, enable)}`` exactly as transmitted this cycle, and
        ``replies`` is ``{base: (t_reply, CompactStatus)}`` for the boards that
        answered inside the receive window.

        This exists because a 150 Hz recording cannot be taken by polling.
        :meth:`snapshot_nodes` deep-copies twenty-four dataclasses and their
        ``extended`` dicts, and a poller runs on its own clock, so every sample
        would be both expensive and off the sync grid -- and the sync edge is
        the only instant the whole arm agrees on.

        **The callback runs inline on the cycle thread**, between the publish
        and the next tick, so anything it does is subtracted from the 6.67 ms
        budget.  It must not block: append to a deque and let another thread
        serialise.  A callback that raises is logged and then *uninstalled*,
        because a fault repeating at 150 Hz would otherwise bury the log and
        stall the bus.  Pass ``None`` to remove it.
        """
        self._on_cycle = callback

    def snapshot_nodes(self) -> dict[int, NodeState]:
        import copy

        with self._lock:
            return {base: copy.deepcopy(node) for base, node in sorted(self.nodes.items())}

    def history(self, base: int,
                max_points: int | None = None) -> list[tuple[float, float, float]]:
        """Copy of a board's (t, measured_psi, target_psi) history.

        ``max_points`` decimates *under the lock*: the full 60 s deque is 9000
        samples per board, and copying all of them while holding the lock the
        cycle thread needs to publish cost a measured 2.8 Hz on the 24-board
        bus (hw_tests/report_integrated_2026-08-20.md §3). A display can only
        show about one point per pixel column, so it should ask for ~900.
        """
        with self._lock:
            hist = self._history.get(base, ())
            if max_points is None or len(hist) <= max_points:
                return list(hist)
            step = max(1, len(hist) // max_points)
            return list(itertools.islice(hist, 0, None, step))

    def missing_nodes(self) -> list[int]:
        now = time.perf_counter()
        with self._lock:
            return [b for b in self.selected
                    if self.nodes[b].enabled and (now - self.nodes[b].last_reply_t) > NODE_TIMEOUT_S]
