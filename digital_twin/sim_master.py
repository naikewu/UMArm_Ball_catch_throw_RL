r"""``SimMaster`` -- the twin's transport shell, shaped exactly like the real one.

This module implements CONTRACT.md section 5.  The property it exists to make
true is that **a controller must not be able to discriminate between the real
robot and the twin**, and it is made true the only way that survives
maintenance: :class:`SimMaster` **subclasses**
``TLE_PCB.tlelib.backend.Backend`` and replaces one member.  The 150 Hz cycle
loop, the runtime-table construction, the reply matching, ``_publish``'s
latency and miss accounting, ``set_cycle_observer``, ``snapshot_nodes``,
``history`` and ``missing_nodes`` are all *the same code* the hardware
acceptance suite already ran against the metal on 2026-08-20.  A second
implementation that agrees with the first until it does not is exactly what
this avoids.

WHAT CHANGES: ``self.link``.  A :class:`SimCanLink` presents ``CanLink``'s
surface -- ``open``/``close``/``add_tap``/``remove_tap``/``send``/
``send_batch``/``scan``/``drain`` -- and routes frames into a
:class:`digital_twin.sim_core.SimArm` instead of down a slcan dongle.  It
decodes the runtime table with ``tlelib.proto``'s own layout rather than a
private one, so the table's ``0x80`` marker byte, its slot mask and its
12-bit-plus-flags packing are round-tripped on every cycle and a change to
either side shows up as a decode failure rather than as a silent disagreement.

THE REPLY LATENCY SPREAD IS PART OF THE INTERFACE, NOT DECORATION.  On the real
24-board bus a reply arrives 1.87 ms after the sync edge at ``0x101`` and rises
about 64 us per id step to 3.35 ms at ``0x118``.  That gradient is CAN
arbitration -- the identifier IS the priority, so the boards answer in id order
and each one waits out the frames of every lower id.  ``Backend._publish``
records it as ``node.reply_latency_ms``, so it is a column a controller can
read, and **a flat column is what would give the twin away**.  A per-id model
also reproduces the one failure that matters operationally: the boards nearest
``0x118`` are the ones whose replies fall outside a shortened receive window
first.

WHAT THIS SHELL DOES NOT MODEL, stated so nobody reads more into a clean run
than is there: bus-error and arbitration-loss recovery, the OTA broadcast path,
the extended telemetry contents on ``base + 0x400`` (the frames are accepted
and dropped), and any jitter on the latency itself -- ``reply_jitter_sd_ms``
defaults to 0 because no per-id jitter figure has been measured on this bus.
The twin's latency column is therefore *cleaner* than the metal's, which is the
one direction in which it is still distinguishable.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_master.py``, read in
detail at ``reference/sim_core.md`` section 2.
"""

from __future__ import annotations

import heapq
import random
import struct
import threading
import time

from . import sim_core

try:  # pragma: no cover - whichever of the two paths is live is exercised
    from TLE_PCB.tlelib import proto as P
    from TLE_PCB.tlelib.backend import Backend
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from TLE_PCB.tlelib import proto as P
    from TLE_PCB.tlelib.backend import Backend


# ---------------------------------------------------------------------------
# The arbitration model
# ---------------------------------------------------------------------------

#: Reply latency at ``0x101``, ms after the sync edge, on the 24-board bus.
REPLY_LATENCY_BASE_MS = 1.87

#: Reply latency at ``0x118``, ms.  The two endpoints are quoted rather than a
#: slope, because a slope alone hides which end was measured.  They imply
#: ``(3.35 - 1.87) / 23`` = 64.3 us per id step.
REPLY_LATENCY_LAST_MS = 3.35

#: Firmware version strings the modelled boards report to a scan.  Taken from
#: the trees in this workspace: ``VEMA_FIRMWARE_VERSION`` in
#: ``firmware/tle/main/pressure_controller.c``, and the legacy bundle's own
#: ``Valve_not_embedded_XL`` v0.2.1.
TLE_FIRMWARE_VERSION = "1.46"
SEVEN_MM_FIRMWARE_VERSION = "0.2.1"


def reply_latency_ms(base: int, *,
                     first_ms: float = REPLY_LATENCY_BASE_MS,
                     last_ms: float = REPLY_LATENCY_LAST_MS,
                     first_base: int = P.ACTUATOR_FIRST,
                     last_base: int = P.ACTUATOR_LAST) -> float:
    """Where in the arbitration queue this board's reply lands, ms after sync.

    Linear in the identifier, because that is what arbitration on a bus whose
    every board answers the same edge produces: the lowest id wins the first
    slot and each higher id waits out the frames below it.  The endpoints are
    arguments rather than constants so that a re-measurement replaces them at
    the call site instead of editing a module a dozen callers share.
    """
    span = max(1, int(last_base) - int(first_base))
    step = (float(last_ms) - float(first_ms)) / span
    return float(first_ms) + step * (int(base) - int(first_base))


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------

class SimCanLink:
    """``CanLink``'s surface, with a :class:`~digital_twin.sim_core.SimArm`
    behind it instead of a serial port.

    Two threads meet here and the split matters.  The **caller's** thread --
    ``Backend``'s cycle thread in practice -- runs :meth:`send_batch`, which
    advances the arm's physics, applies the sync edge and computes each board's
    reply payload from the pressure that edge latched.  A private **delivery**
    thread then hands those payloads to the taps at their arbitration times.
    That mirrors the real link, where the reader thread stamps a frame when it
    parses it, and it is what lets ``Backend``'s receive window do real work:
    a reply scheduled past the window is genuinely missed.

    The payload is built at the edge and not at delivery because that is what
    the firmware does.  ``tle_can_legacy.c::handle_sync`` latches
    ``s_latched_raw`` in the sync handler, so "the reported value belongs to the
    sync instant even though the replies themselves trickle out over the
    following milliseconds".
    """

    def __init__(self, arm, port: str = "SIM", bitrate: int = 1_000_000, *,
                 first_ms: float = REPLY_LATENCY_BASE_MS,
                 last_ms: float = REPLY_LATENCY_LAST_MS,
                 reply_jitter_sd_ms: float = 0.0,
                 clock=time.perf_counter,
                 advance: bool = True,
                 idle_advance_s: float = 0.005,
                 seed: int = 20260910) -> None:
        self.arm = arm
        self.port = port
        self.bitrate = bitrate
        self.adapter_version = "SIM"
        self.rx_count = 0
        self.rx_dropped = 0
        self.first_ms = float(first_ms)
        self.last_ms = float(last_ms)
        self.reply_jitter_sd_ms = float(reply_jitter_sd_ms)
        self.clock = clock
        #: When false the link routes frames but never moves physics -- the seam
        #: :mod:`digital_twin.replay` needs, where the caller owns the clock.
        self.advance = bool(advance)
        #: How often the delivery thread advances the arm on its own, s.
        #:
        #: Without it the twin's only clock is the sync edge, and a host that
        #: stops driving would freeze the physics -- which would make the
        #: 500 ms TLE sync-loss failsafe unobservable through this shell, since
        #: the very silence that should trip it also stops the clock that
        #: measures it.  5 ms keeps the twin within a bus period of wall time.
        #: It is not a determinism hazard: reaching ``T`` is a pure function of
        #: ``T`` (invariant C5), and a stale target is a no-op.
        self.idle_advance_s = float(idle_advance_s)

        self._rng = random.Random(seed)
        self._taps = []
        self._tap_lock = threading.Lock()
        self._pending = []                     # heap of (due, seq, can_id, data)
        self._seq = 0
        self._batch_t0 = None
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self.sync_count = 0
        self.dropped_replies = 0
        #: Frames the twin accepted and did not model -- extended telemetry
        #: requests, OTA, host control.  Counted rather than ignored, so a test
        #: that expects a modelled path can assert this stayed at zero.
        self.unmodelled_frames = 0

    # ---- lifecycle ------------------------------------------------------
    def open(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._deliver, name="sim-canlink",
                                        daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def is_open(self) -> bool:
        return self._thread is not None

    def __enter__(self) -> "SimCanLink":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- taps ------------------------------------------------------------
    def add_tap(self, fn) -> None:
        with self._tap_lock:
            self._taps.append(fn)

    def remove_tap(self, fn) -> None:
        with self._tap_lock:
            if fn in self._taps:
                self._taps.remove(fn)

    def drain(self) -> None:
        with self._cv:
            self._pending.clear()

    # ---- transmit --------------------------------------------------------
    def send(self, can_id: int, data: bytes = b"") -> None:
        self._route(int(can_id), bytes(data))

    def send_batch(self, frames) -> None:
        """One serial write on the metal, so one indivisible routing here.

        ``Backend._send_cycle`` hands the table frames and the sync as a single
        batch precisely so the adapter cannot interleave anything between them;
        routing them under one pass reproduces that guarantee rather than
        leaving a window in which a target could be staged after its own edge.
        """
        # The whole batch is timed from its first frame, not from the sync frame
        # inside it.  ``Backend._publish`` measures ``reply_latency_ms`` against
        # a stamp it takes *before* building and writing the table, which is
        # also how the 1.87/3.35 ms endpoints were measured on the metal, so the
        # twin has to reference the same instant or its whole column reads high
        # by however long the table takes to construct.
        self._batch_t0 = self.clock()
        try:
            for can_id, data in frames:
                self._route(int(can_id), bytes(data))
        finally:
            self._batch_t0 = None

    def _route(self, can_id: int, data: bytes) -> None:
        if can_id == P.ID_BROADCAST:
            if len(data) == 0:
                self._sync_edge()
            else:
                self.unmodelled_frames += 1     # broadcast OTA image data
            return
        group = can_id - P.ID_RUNTIME_TABLE_BASE
        if 0 <= group < 8 and len(data) == 8:
            self._stage_table(group, data)
            return
        self.unmodelled_frames += 1

    def _stage_table(self, group: int, data: bytes) -> None:
        """Decode one table frame with ``proto``'s own layout, and stage it.

        The ``0x80`` marker in byte 0 is what tells a board this is a table
        frame and not broadcast OTA data, whose first byte is a sequence number
        of 0..127.  Checking it here means a host that ever stops setting it
        fails loudly in the twin instead of quietly commanding nothing.
        """
        if not data[0] & P.RUNTIME_TABLE_MARKER:
            self.unmodelled_frames += 1
            return
        start_slot = data[0] & P.RUNTIME_TABLE_SLOTMASK
        mask = data[1]
        staged = {}
        for offset in range(P.RUNTIME_TABLE_SLOTS):
            if not mask & (1 << offset):
                continue
            word = struct.unpack_from("<H", data, 2 + offset * 2)[0]
            base = P.ACTUATOR_FIRST + start_slot + offset
            staged[base] = (word & P.PRESSURE_MASK,
                            bool((word >> 12) & P.CONTROL_ENABLE))
        if staged:
            self.arm.stage_targets(staged)

    def _sync_edge(self) -> None:
        """Advance physics to now, apply the edge, and queue the replies."""
        now = self.clock() if self._batch_t0 is None else self._batch_t0
        with self.arm.lock:
            if self.advance:
                self.arm.advance_to(now)
            self.arm.sync_edge(now)
            self.sync_count += 1
            replies = []
            for base, node in sorted(self.arm.nodes.items()):
                if node.fault.reply_dropout and self._rng.random() < node.fault.reply_dropout:
                    self.dropped_replies += 1
                    continue
                status = node.compact_status()
                word = (status.counts & P.PRESSURE_MASK) | ((status.flags & P.FLAGS_MASK) << 12)
                replies.append((base, struct.pack("<H", word)))
        latency = 0.0
        for base, payload in replies:
            latency = reply_latency_ms(base, first_ms=self.first_ms,
                                       last_ms=self.last_ms) / 1000.0
            if self.reply_jitter_sd_ms:
                latency += self._rng.gauss(0.0, self.reply_jitter_sd_ms / 1000.0)
            self._schedule(now + latency, base, payload)

    def _schedule(self, due: float, can_id: int, payload: bytes) -> None:
        with self._cv:
            self._seq += 1
            heapq.heappush(self._pending, (due, self._seq, can_id, payload))
            self._cv.notify()

    # ---- receive ---------------------------------------------------------
    def _deliver(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                if not self._pending:
                    self._cv.wait(self.idle_advance_s)
                    self._keep_time()
                    continue
                due = self._pending[0][0]
                remaining = due - self.clock()
                if remaining > 0.0:
                    self._cv.wait(min(remaining, self.idle_advance_s))
                    self._keep_time()
                    continue
                _, _, can_id, payload = heapq.heappop(self._pending)
            #: Stamped with the arbitration due time rather than with the
            #: delivery thread's own wake-up.  ``Backend._publish`` computes
            #: ``reply_latency_ms`` from this stamp, and stamping the wake-up
            #: would fold the host OS's scheduler jitter into a column that on
            #: the metal measures the bus.
            self.rx_count += 1
            self._fire(due, can_id, payload)

    def _keep_time(self) -> None:
        """Move the arm's clock while the bus is quiet.  See ``idle_advance_s``."""
        if not self.advance:
            return
        with self.arm.lock:
            self.arm.advance_to(self.clock())

    def _fire(self, t: float, can_id: int, data: bytes) -> None:
        with self._tap_lock:
            taps = list(self._taps)
        for tap in taps:
            tap(t, can_id, data)

    # ---- discovery -------------------------------------------------------
    def scan(self, bases=None, settle: float = 0.5,
             rounds: int = 2) -> dict:
        """Answer a firmware-version request from every modelled board.

        No settle and no rounds are needed here, and both arguments are kept
        anyway so a caller written against the real link runs unchanged: the
        real ``scan`` needs a second round because a board that has sat on a
        quiet bus parks its transmit path until something proves the bus is
        alive, and the frame that proves it is the casualty.
        """
        candidates = list(P.ALL_IDS if bases is None else bases)
        found = {}
        for base in candidates:
            node = self.arm.nodes.get(int(base))
            if node is None:                    # absent: nothing answers
                continue
            version = P.FirmwareVersion()
            version.variant = node.variant
            version.version = (TLE_FIRMWARE_VERSION if node.is_tle
                               else SEVEN_MM_FIRMWARE_VERSION)
            version.chunks = 1
            found[int(base)] = version
        return found


# ---------------------------------------------------------------------------
# The shell
# ---------------------------------------------------------------------------

class SimMaster(Backend):
    """``Backend`` with a modelled bus behind it.  Everything else is inherited.

    The constructor's positional surface is ``Backend``'s, name for name and
    default for default, with one deliberate exception: ``port`` defaults to
    ``"SIM"`` instead of the resolved COM port, so that constructing a twin can
    never touch the dongle a live campaign is using.  Everything the twin needs
    of its own is keyword-only and after it.

    ``self.nodes`` stays what ``Backend`` makes it -- the **host's** registry of
    :class:`~TLE_PCB.tlelib.backend.NodeState`, filled by :meth:`scan` and
    updated by ``_publish``.  It is deliberately not aliased onto
    ``self.arm.nodes``, which is the **board's** state: a controller reads the
    host's view, and a twin that let it read the board's would hand it the
    unfiltered plant truth and the enable bit as the board sees it, neither of
    which comes back over a real wire.  The RS485 twin could alias the two
    because there the shell owned the node objects; here ``Backend`` owns a
    separate host-side view, and keeping the two apart is what preserves the
    "cannot discriminate" property rather than breaking it.
    """

    def __init__(self, port: str = "SIM", bitrate: int = 1_000_000,
                 cycle_hz: float = P.CYCLE_HZ, log=None, *,
                 arm=None,
                 first_ms: float = REPLY_LATENCY_BASE_MS,
                 last_ms: float = REPLY_LATENCY_LAST_MS,
                 reply_jitter_sd_ms: float = 0.0,
                 clock=time.perf_counter,
                 advance: bool = True,
                 idle_advance_s: float = 0.005,
                 seed: int = 20260910,
                 **simarm_kwargs) -> None:
        super().__init__(port=port, bitrate=bitrate, cycle_hz=cycle_hz, log=log)
        if arm is None:
            arm = sim_core.SimArm(seed=seed, **simarm_kwargs)
        elif simarm_kwargs:
            raise ValueError(
                "arm= and SimArm keyword arguments are mutually exclusive: an "
                f"arm handed in cannot honour {sorted(simarm_kwargs)}")
        self.arm = arm
        #: Replaced after ``super().__init__`` rather than before, because
        #: ``Backend.__init__`` constructs its own ``CanLink``.  That
        #: constructor opens nothing -- it only records the port name and the
        #: bitrate -- so the object it builds is discarded without ever having
        #: touched the hardware.
        self.link = SimCanLink(arm, port=port, bitrate=bitrate,
                               first_ms=first_ms, last_ms=last_ms,
                               reply_jitter_sd_ms=reply_jitter_sd_ms,
                               clock=clock, advance=advance,
                               idle_advance_s=idle_advance_s, seed=seed)

    def snapshot_arm(self) -> dict:
        """Plant truth beside the host's view, for a test or a plot.

        Named apart from :meth:`snapshot_nodes` on purpose.  ``snapshot_nodes``
        is the interface a controller shares with the real robot and must stay
        byte-comparable with it; this is the twin's privilege, and any code path
        that reads it is a path that cannot run against the metal.
        """
        with self.arm.lock:
            return {base: dict(p_pa=node.p_pa,
                               true_psi=node.true_psi,
                               action=node.action,
                               enabled=node.enabled,
                               failsafe_active=node.failsafe_active,
                               target_pa=node.target_pa,
                               variant=node.variant)
                    for base, node in sorted(self.arm.nodes.items())}
