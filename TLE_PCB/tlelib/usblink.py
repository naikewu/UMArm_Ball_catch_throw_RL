"""Point-to-point link to one board over its native USB-Serial/JTAG port.

The board carries the same frames on USB as it does on CAN, as text lines:

    host -> board   ">III:HHHH...\\n"
    board -> host   "<III:HHHH...\\n"

III is the 11-bit CAN ID in hex and the rest is 0..8 payload bytes as hex
pairs. Boot and warning logs share the port, so anything that is not a framed
line is kept separately rather than discarded -- the log is often the fastest
way to see what a board decided at startup.

Outbound frames are only mirrored by the firmware while a USB host is actively
talking, so a reader that wants a telemetry stream has to keep sending
something; ``keepalive()`` is that something.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

import serial

from . import native as N
from . import proto as P
from . import wsenv

# COM8 is the board's own port on this bench and remains the fallback; a
# workspace-level bench_env.py, if one exists, wins.
DEFAULT_PORT = wsenv.get("USB_PORT", "COM8")
ESP32S3_USB_VID = 0x303A
ESP32S3_USB_PID = 0x1001


class UsbLink:
    def __init__(self, port: str = DEFAULT_PORT, base: int | None = None):
        self.port = port
        # The board answers on base + 0x300 whatever we address, so the base is
        # only needed to build outbound IDs. It is discovered from the first
        # telemetry frame if not supplied.
        self.base = base
        self.frames: queue.Queue = queue.Queue(maxsize=4096)
        self.log_lines: deque[str] = deque(maxlen=400)
        self._ser: serial.Serial | None = None
        self._buf = bytearray()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._write_lock = threading.Lock()

    # ---- lifecycle -----------------------------------------------------
    def open(self) -> None:
        self._ser = serial.Serial(self.port, 115200, timeout=0.02, write_timeout=1.0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="usblink-rx", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self) -> "UsbLink":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- transport -----------------------------------------------------
    def send(self, can_id: int, payload: bytes) -> None:
        if self._ser is None:
            raise RuntimeError("USB link is closed")
        line = f">{can_id:03X}:{payload.hex().upper()}\n".encode()
        with self._write_lock:
            self._ser.write(line)
            self._ser.flush()

    def command(self, payload: bytes, base: int | None = None) -> None:
        """Send a native command. Over USB these go to base + 0x100."""
        base = base if base is not None else (self.base or P.ACTUATOR_FIRST)
        self.send(base + P.OFFSET_HOST_CTRL, payload)

    def keepalive(self) -> None:
        self.command(N.build_keepalive())

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(4096) if self._ser else b""
            except Exception:
                break
            if chunk:
                self._buf.extend(chunk)
                self._parse()
            else:
                time.sleep(0.002)

    def _parse(self) -> None:
        while b"\n" in self._buf:
            raw, _, rest = self._buf.partition(b"\n")
            self._buf = bytearray(rest)
            line = raw.decode(errors="replace").strip("\r\x00 ")
            if not line:
                continue
            if not line.startswith("<") or ":" not in line:
                self.log_lines.append(line)
                continue
            head, _, body = line[1:].partition(":")
            try:
                can_id = int(head, 16)
                data = bytes.fromhex(body)
            except ValueError:
                self.log_lines.append(line)
                continue
            if self.base is None:
                base = P.status_id_to_base(can_id)
                if base is not None:
                    self.base = base
            try:
                self.frames.put_nowait((time.perf_counter(), can_id, data))
            except queue.Full:
                try:
                    self.frames.get_nowait()
                    self.frames.put_nowait((time.perf_counter(), can_id, data))
                except queue.Empty:
                    pass

    # ---- helpers -------------------------------------------------------
    def drain(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def collect(self, seconds: float, want=None) -> list[tuple[float, int, bytes]]:
        out = []
        deadline = time.perf_counter() + seconds
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return out
            try:
                item = self.frames.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if want is None or want(item[1], item[2]):
                out.append(item)

    def read_state(self, seconds: float = 0.5) -> dict:
        """Poll once and merge whatever telemetry comes back."""
        self.drain()
        self.keepalive()
        state: dict = {}
        for _, _, data in self.collect(seconds):
            N.decode_into(data, state)
        return state

    def wait_log(self, needle: str, seconds: float = 3.0) -> str | None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            for line in list(self.log_lines):
                if needle in line:
                    return line
            time.sleep(0.05)
        return None

    def set_board_id(self, new_base: int, current_base: int | None = None,
                     settle_s: float = 3.0) -> bool:
        """Persist a new base ID over USB. The board reboots to apply it.

        The USB port is the board's own USB-Serial/JTAG peripheral, so it
        disappears and re-enumerates across the reboot; the caller has to
        reopen. Returns True once the reboot has been triggered and the port
        has come back with the new ID in its boot log.
        """
        base = current_base if current_base is not None else (self.base or P.ACTUATOR_FIRST)
        self.log_lines.clear()
        self.drain()
        self.command(N.build_set_device_id(new_base), base=base)
        # The board answers COMMAND_STATUS on the NEW status ID, then reboots
        # 100 ms later.
        acked = bool(self.collect(1.0, lambda cid, d: cid == P.ids(new_base).status
                                  and len(d) > 2 and d[0] == 0x06 and d[1] == N.CMD_SET_DEVICE_ID
                                  and d[2] == 0))
        self.close()
        time.sleep(settle_s)
        self.base = new_base
        self.open()
        return acked


def find_esp_ports() -> list[str]:
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports()
            if p.vid == ESP32S3_USB_VID and p.pid == ESP32S3_USB_PID]
