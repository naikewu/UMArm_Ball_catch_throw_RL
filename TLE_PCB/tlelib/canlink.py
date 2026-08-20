"""slcan link to the CAN bus, plus node discovery.

The adapter on this bench is a CANable-class dongle presenting a USB CDC port
(COM58 here), driven with the slcan command set over pyserial rather than
through python-can. Two rules were learned the hard way on this bench and are
enforced here:

  * The adapter is opened in NORMAL mode, never listen-only. Listen-only does
    not send ACK bits, so with a single board on the bus its CAN controller
    never sees an acknowledgement and retries forever.
  * S7 (750 kbit/s) is never emitted. The CANable 2.0 firmware sets a prescaler
    of 7 where it needs 13.33, so S7 actually puts the wire at 1.43 Mbit/s.
    Only S6 (500 k) and S8 (1 M) are offered.

Reads and writes run on separate threads without a shared lock. pyserial's
Windows backend keeps independent OVERLAPPED structures for the two
directions, so this is safe -- and it matters, because a lock held across a
blocking read would put the read timeout straight into the jitter budget of a
150 Hz transmit schedule.
"""
from __future__ import annotations

import queue
import threading
import time

import serial

from . import proto as P
from . import wsenv

BITRATES = {1_000_000: "S8", 500_000: "S6"}
# COM58 is where the dongle enumerated on this bench and remains the fallback.
# The workspace's bench_env.py, if present, resolves the port from the adapter's
# USB descriptor triple instead, which survives a replug; enumeration opens
# nothing. Every call site still takes an explicit port.
DEFAULT_PORT = wsenv.can_port("COM58")


class SlcanError(RuntimeError):
    pass


class CanLink:
    """Threaded slcan link. Raw frames in, raw frames out."""

    def __init__(self, port: str = DEFAULT_PORT, bitrate: int = 1_000_000,
                 rx_queue_depth: int = 16384):
        if bitrate not in BITRATES:
            raise SlcanError(f"unsupported bitrate {bitrate}; use {sorted(BITRATES)}")
        self.port = port
        self.bitrate = bitrate
        self.frames: queue.Queue = queue.Queue(maxsize=rx_queue_depth)
        self.adapter_version = ""
        self.rx_count = 0
        self.rx_dropped = 0
        self._ser: serial.Serial | None = None
        self._buf = bytearray()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._taps: list = []
        self._tap_lock = threading.Lock()

    # ---- lifecycle -----------------------------------------------------
    def open(self) -> None:
        # timeout=0 makes reads non-blocking. A blocking read waits out its
        # full timeout whenever fewer bytes are available than were asked for,
        # which batches arrivals onto the timeout's beat and stamps them all at
        # the end of it -- that beat is uncorrelated with the 150 Hz cycle, so
        # it smears measured reply latency uniformly across a whole cycle and
        # makes replies look like they missed their window. The reader polls
        # instead, and stamps each frame when it actually shows up.
        self._ser = serial.Serial(self.port, 115200, timeout=0, write_timeout=1.0)
        time.sleep(0.3)
        self._raw("C")            # a previous session may have left it open
        self._ser.reset_input_buffer()
        self.adapter_version = self._query("V")
        self._raw(BITRATES[self.bitrate])
        self._raw("O")            # normal mode, ACKs enabled
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="canlink-rx", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._raw("C")
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self) -> "CanLink":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ---- raw slcan -----------------------------------------------------
    def _raw(self, text: str, settle: float = 0.15) -> None:
        assert self._ser is not None
        with self._write_lock:
            self._ser.write(text.encode() + b"\r")
            self._ser.flush()
        time.sleep(settle)

    def _query(self, text: str, wait: float = 0.3) -> str:
        """Only V and E answer on this firmware; nothing else is acknowledged."""
        assert self._ser is not None
        with self._write_lock:
            self._ser.reset_input_buffer()
            self._ser.write(text.encode() + b"\r")
            self._ser.flush()
        time.sleep(wait)
        return self._ser.read(256).decode(errors="replace").strip()

    @staticmethod
    def encode(can_id: int, data: bytes) -> bytes:
        return (f"t{can_id:03X}{len(data)}" + data.hex().upper()).encode() + b"\r"

    def send(self, can_id: int, data: bytes = b"") -> None:
        if self._ser is None:
            raise SlcanError("link is closed")
        with self._write_lock:
            self._ser.write(self.encode(can_id, data))
            self._ser.flush()

    def send_batch(self, frames: list[tuple[int, bytes]]) -> None:
        """Write several frames in one serial write.

        Per-frame writes cost a USB transaction each, which dominates the cost
        of a bulk transfer; concatenating lets the CDC endpoint carry a whole
        group at once. Pacing between batches is the caller's job -- the
        adapter silently drops frames it cannot hand to the CAN controller in
        time.
        """
        if self._ser is None:
            raise SlcanError("link is closed")
        blob = b"".join(self.encode(cid, data) for cid, data in frames)
        with self._write_lock:
            self._ser.write(blob)
            self._ser.flush()

    # ---- receive -------------------------------------------------------
    def add_tap(self, fn) -> None:
        """Register fn(timestamp, can_id, data), called on the reader thread.

        Used by the sync master, which needs replies matched to the cycle they
        belong to rather than whenever a consumer gets round to the queue.
        """
        with self._tap_lock:
            self._taps.append(fn)

    def remove_tap(self, fn) -> None:
        with self._tap_lock:
            if fn in self._taps:
                self._taps.remove(fn)

    def drain(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                if self._ser is None:
                    break
                waiting = self._ser.in_waiting
                chunk = self._ser.read(waiting) if waiting else b""
            except Exception:
                break
            if chunk:
                self._buf.extend(chunk)
                self._parse()
            else:
                # Fine enough to resolve where in a 6.67 ms cycle a reply
                # landed, coarse enough not to spin a core.
                time.sleep(0.0002)

    def _parse(self) -> None:
        while b"\r" in self._buf:
            raw, _, rest = self._buf.partition(b"\r")
            self._buf = bytearray(rest)
            line = raw.decode(errors="replace").strip("\a\n ")
            if len(line) < 5 or line[0] not in "tT":
                continue
            if line[0] == "T":
                continue  # extended IDs are not used by either protocol
            try:
                can_id = int(line[1:4], 16)
                dlc = int(line[4], 16)
                data = bytes.fromhex(line[5:5 + 2 * dlc])
            except ValueError:
                continue
            if len(data) != dlc:
                continue
            item = (time.perf_counter(), can_id, data)
            self.rx_count += 1
            with self._tap_lock:
                taps = list(self._taps)
            for tap in taps:
                try:
                    tap(*item)
                except Exception:
                    pass
            try:
                self.frames.put_nowait(item)
            except queue.Full:
                # Drop oldest: a stalled consumer must never wedge the reader.
                self.rx_dropped += 1
                try:
                    self.frames.get_nowait()
                    self.frames.put_nowait(item)
                except queue.Empty:
                    pass

    # ---- request/response helpers --------------------------------------
    def collect(self, seconds: float, want=None,
                limit: int | None = None) -> list[tuple[float, int, bytes]]:
        """Gather frames for a window. `want(can_id, data)` filters if given.

        `limit` returns as soon as that many matches are in hand. Without it
        this waits out the whole window, which is right for "listen for a
        while" and badly wrong for a request/response poll -- an OTA runs one
        poll per block, so a full timeout per poll would add minutes.
        """
        out: list[tuple[float, int, bytes]] = []
        deadline = time.perf_counter() + seconds
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return out
            try:
                item = self.frames.get(timeout=min(remaining, 0.02))
            except queue.Empty:
                continue
            if want is None or want(item[1], item[2]):
                out.append(item)
                if limit is not None and len(out) >= limit:
                    return out

    def request(self, base: int, command: int, expect: int,
                timeout: float = 0.35, limit: int | None = None) -> list[bytes]:
        """Send a one-byte host-control command and collect replies.

        Returns every payload seen on that node's status ID whose first byte is
        `expect`, which is how a multi-frame reply (the version string, the
        four diagnostic frames) is gathered.
        """
        node = P.ids(base)
        # Drain first. A reply that arrived after a previous poll timed out is
        # still sitting in the queue, and without this it would be handed back
        # as the answer to this one -- during an update that is a stale view of
        # how much of a block a board has taken.
        self.drain()
        self.send(node.ctrl, P.build_simple_command(command))
        got = self.collect(timeout, lambda cid, d: cid == node.status and d and d[0] == expect, limit)
        return [data for _, _, data in got]

    def wait_ack(self, base: int, timeout: float = 2.0) -> tuple[bool, int]:
        """Wait for MSG_ACK / MSG_NACK on a node's status ID."""
        node = P.ids(base)
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False, 0
            try:
                _, can_id, data = self.frames.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if can_id != node.status or not data:
                continue
            if data[0] == P.MSG_ACK:
                return True, 0
            if data[0] == P.MSG_NACK:
                return False, data[1] if len(data) > 1 else 0

    # ---- discovery -----------------------------------------------------
    def scan(self, bases=None, settle: float = 0.5, rounds: int = 2) -> dict[int, P.FirmwareVersion]:
        """Ask every candidate actuator ID for its firmware version.

        A version reply is the discovery mechanism both board types share, so
        one scan enumerates the whole bus and says which kind each node is.

        More than one round by default. A board that has been sitting on a bus
        with no other node keeps its CAN controller's transmit path parked
        until something proves the bus is alive again, and the frame that
        proves it is the first request of the first round -- whose reply can be
        the casualty. The second round always lands.
        """
        candidates = list(P.ALL_IDS if bases is None else bases)
        by_status = {P.ids(b).status: b for b in candidates}
        found: dict[int, P.FirmwareVersion] = {}

        for _ in range(max(1, rounds)):
            self.drain()
            # Ask everyone first, then listen: a board answers in well under a
            # millisecond, and spacing the requests keeps the adapter's
            # transmit path from being handed 24 frames at once.
            for base in candidates:
                self.send(P.ids(base).ctrl, P.build_simple_command(P.CMD_GET_FW_VERSION))
                time.sleep(0.003)

            deadline = time.perf_counter() + settle
            while time.perf_counter() < deadline:
                try:
                    _, can_id, data = self.frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                base = by_status.get(can_id)
                if base is None or not data or data[0] != P.MSG_FW_VERSION:
                    continue
                found.setdefault(base, P.FirmwareVersion()).feed(data)

            if found and all(v.complete for v in found.values()):
                break

        return {base: version for base, version in sorted(found.items())}

    def ota_status(self, base: int, timeout: float = 1.0) -> P.OtaStatus | None:
        replies = self.request(base, P.CMD_GET_OTA_STATUS, P.MSG_OTA_STATUS, timeout, limit=1)
        return P.parse_ota_status(replies[-1]) if replies else None

    def can_diag(self, base: int, timeout: float = 0.5) -> P.CanDiag | None:
        node = P.ids(base)
        self.drain()
        self.send(node.ctrl, P.build_simple_command(P.CMD_GET_CAN_DIAG))
        diag_types = (P.MSG_CAN_DIAG0, P.MSG_CAN_DIAG1, P.MSG_CAN_DIAG2, P.MSG_CAN_DIAG3)
        frames: dict[int, bytes] = {}
        for _, _, data in self.collect(timeout,
                                       lambda cid, d: cid == node.status and d and d[0] in diag_types,
                                       limit=len(diag_types)):
            frames[data[0]] = data
        return P.merge_can_diag(frames) if frames else None

    def set_node_id(self, base: int, new_base: int, timeout: float = 2.0) -> bool:
        """Change a node's base ID over CAN. The board reboots to apply it."""
        self.drain()
        self.send(P.ids(base).ctrl, P.build_set_id(new_base))
        # The board applies the new ID before acknowledging, so the ACK arrives
        # on the NEW status ID.
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                _, can_id, data = self.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            if can_id == P.ids(new_base).status and data and data[0] == P.MSG_ACK:
                return True
            if can_id == P.ids(base).status and data and data[0] == P.MSG_NACK:
                return False
        return False


def list_serial_ports() -> list[tuple[str, str]]:
    from serial.tools import list_ports
    return [(p.device, f"{p.device} - {p.description}") for p in list_ports.comports()]
