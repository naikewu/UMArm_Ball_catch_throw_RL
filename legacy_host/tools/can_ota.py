from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import tkinter as tk
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

try:
	import can
except ImportError:  # pragma: no cover
	can = None  # type: ignore[assignment]

try:
	from serial.tools import list_ports
except ImportError:  # pragma: no cover
	list_ports = None  # type: ignore[assignment]


# --- workspace paths and port resolution -----------------------------------
# Upstream, PROJECT_ROOT was the VNEMA repo root and the two images were read
# out of its build_dt/ and build_7mm/ trees. Here the images are the copies
# under firmware/images/, whose provenance is recorded in that directory's
# README; taking them from a build tree would make the flashed bytes depend on
# whoever last ran idf.py.
#
# DEFAULT_PORT was "COM4", a port that does not exist on this bench. Pinning any
# COM number repeats that mistake, since Windows assigns the number per hub port
# and the dongle moves when it is replugged, so bench_env resolves the adapter by
# its USB VID:PID and serial number instead, falling back to COM58.
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
	sys.path.insert(0, str(WS_ROOT))
try:
	import bench_env as _bench_env
except Exception:  # pragma: no cover - keep the tool usable without bench_env
	_bench_env = None

PROJECT_ROOT = WS_ROOT
DEFAULT_PORT = _bench_env.resolve_can_port() if _bench_env is not None else "COM58"
DEFAULT_CAN_BITRATE = 1_000_000
DEFAULT_TTY_BAUDRATE = 2_000_000
DEFAULT_DT_BIN = WS_ROOT / "firmware" / "images" / "legacy_dt" / "Valve_not_embedded_XL.bin"
DEFAULT_SEVEN_MM_BIN = WS_ROOT / "firmware" / "images" / "legacy_7mm" / "Valve_not_embedded_XL.bin"

DT_IDS = range(0x101, 0x109)
SEVEN_MM_IDS = range(0x109, 0x119)
EXPECTED_IDS = range(0x101, 0x119)

BROADCAST_CAN_ID = 0x090
HOST_CTRL_OFFSET = 0x100
HOST_DATA_OFFSET = 0x200
ESP_STATUS_OFFSET = 0x300

CMD_START = 0x01
CMD_END = 0x02
CMD_GET_CAN_DIAG = 0x06
CMD_GET_OTA_STATUS = 0x08
CMD_GET_FW_VERSION = 0x09

MSG_ACK = 0xAA
MSG_NACK = 0xFF
MSG_CAN_DIAG0 = 0xD0
MSG_CAN_DIAG1 = 0xD1
MSG_CAN_DIAG2 = 0xD2
MSG_CAN_DIAG3 = 0xD3
MSG_OTA_STATUS = 0xD4
MSG_FW_VERSION = 0xD5

OTA_STATUS_ACTIVE = 0x01
OTA_STATUS_SEQ_ERROR = 0x02
OTA_STATUS_WRITE_ERROR = 0x04
OTA_STATUS_BAD_FRAME = 0x08
OTA_ERROR_MASK = OTA_STATUS_SEQ_ERROR | OTA_STATUS_WRITE_ERROR | OTA_STATUS_BAD_FRAME

FRAMES_PER_BLOCK = 128
DATA_PER_FRAME = 7
BLOCK_BYTES = FRAMES_PER_BLOCK * DATA_PER_FRAME
FRAME_GAP_S = 0.0005
ACK_TIMEOUT_S = 2.0
START_ACK_TIMEOUT_S = 3.0
END_ACK_TIMEOUT_S = 4.0
STATUS_TIMEOUT_S = 1.0
VERSION_TIMEOUT_S = 0.35

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, str, float], None]


@dataclass(frozen=True)
class DeviceIds:
	base_id: int
	ctrl_id: int
	data_id: int
	stat_id: int


@dataclass(frozen=True)
class BoardUpload:
	base_id: int
	firmware_path: Path


@dataclass(frozen=True)
class OtaStatus:
	active: bool
	flags: int
	expected_seq: int
	buffer_index: int
	blocks_written: int
	error_code: int


@dataclass(frozen=True)
class FirmwareVersion:
	version: str
	variant: str
	chunks: int


def parse_int_auto(raw: str) -> int:
	return int(raw.strip(), 0)


def parse_can_ids(raw_values: Iterable[str]) -> list[int]:
	parsed: list[int] = []
	for token in raw_values:
		for part in token.split(","):
			item = part.strip()
			if not item:
				continue
			if "-" in item:
				first_raw, last_raw = item.split("-", 1)
				first = parse_int_auto(first_raw)
				last = parse_int_auto(last_raw)
				if last < first:
					raise argparse.ArgumentTypeError(f"Invalid descending CAN ID range: {item!r}")
				parsed.extend(range(first, last + 1))
			else:
				value = parse_int_auto(item)
				if value < 0 or value > 0x7FF:
					raise argparse.ArgumentTypeError(f"CAN ID {item!r} is outside standard ID range")
				parsed.append(value)
	return list(dict.fromkeys(parsed))


def expand_id_group(groups: Iterable[str]) -> list[int]:
	ids: list[int] = []
	for group in groups:
		normalized = group.strip().lower()
		if normalized in {"all", "all24"}:
			ids.extend(EXPECTED_IDS)
		elif normalized in {"dt", "big"}:
			ids.extend(DT_IDS)
		elif normalized in {"7mm", "seven", "seven-mm"}:
			ids.extend(SEVEN_MM_IDS)
		else:
			ids.extend(parse_can_ids([group]))
	return list(dict.fromkeys(ids))


def derive_ids(base_id: int) -> DeviceIds:
	return DeviceIds(
		base_id=base_id,
		ctrl_id=base_id + HOST_CTRL_OFFSET,
		data_id=base_id + HOST_DATA_OFFSET,
		stat_id=base_id + ESP_STATUS_OFFSET,
	)


def firmware_kind_for_id(base_id: int) -> str:
	if base_id in DT_IDS:
		return "DT"
	if base_id in SEVEN_MM_IDS:
		return "7mm"
	return "Custom"


def firmware_for_id(base_id: int, dt_bin: Path, seven_mm_bin: Path) -> Path:
	if base_id in DT_IDS:
		return dt_bin
	if base_id in SEVEN_MM_IDS:
		return seven_mm_bin
	raise ValueError(f"CAN ID 0x{base_id:03X} is not in the expected actuator ranges")


def list_serial_ports() -> list[str]:
	if list_ports is None:
		return [DEFAULT_PORT]
	return [port.device for port in list_ports.comports()]


def open_slcan_bus(port: str, bitrate: int, tty_baudrate: int) -> Any:
	if can is None:
		raise RuntimeError("python-can is not installed. Run: python -m pip install python-can pyserial")
	return can.interface.Bus(
		interface="slcan",
		channel=port,
		bitrate=bitrate,
		tty_baudrate=tty_baudrate,
	)


def send_can_message(bus: Any, arbitration_id: int, data: Iterable[int] | bytes) -> None:
	bus.send(can.Message(arbitration_id=arbitration_id, data=bytes(data), is_extended_id=False))


def drain_bus(bus: Any, duration_s: float = 0.05) -> None:
	deadline = time.monotonic() + duration_s
	while time.monotonic() < deadline:
		message = bus.recv(0.0)
		if message is None:
			break


def wait_for_ack(bus: Any, stat_id: int, timeout_s: float) -> tuple[bool, int]:
	deadline = time.monotonic() + timeout_s
	while time.monotonic() < deadline:
		message = bus.recv(min(0.1, max(0.0, deadline - time.monotonic())))
		if message is None or getattr(message, "is_extended_id", False):
			continue
		if message.arbitration_id != stat_id:
			continue
		data = bytes(message.data)
		if not data:
			continue
		if data[0] == MSG_ACK:
			return True, 0
		if data[0] == MSG_NACK:
			return False, data[1] if len(data) > 1 else 0
	return False, -1


def get_u16(data: bytes, index: int) -> int:
	if len(data) <= index:
		return 0
	low_byte = data[index]
	high_byte = data[index + 1] if len(data) > index + 1 else 0
	return low_byte | (high_byte << 8)


def read_ota_status(bus: Any, stat_id: int, timeout_s: float = STATUS_TIMEOUT_S) -> OtaStatus | None:
	deadline = time.monotonic() + timeout_s
	while time.monotonic() < deadline:
		message = bus.recv(min(0.1, max(0.0, deadline - time.monotonic())))
		if message is None or getattr(message, "is_extended_id", False):
			continue
		if message.arbitration_id != stat_id:
			continue
		data = bytes(message.data)
		if not data:
			continue
		if data[0] == MSG_OTA_STATUS:
			flags = data[1] if len(data) > 1 else 0
			return OtaStatus(
				active=(flags & OTA_STATUS_ACTIVE) != 0,
				flags=flags,
				expected_seq=data[2] if len(data) > 2 else 0,
				buffer_index=get_u16(data, 3),
				blocks_written=get_u16(data, 5),
				error_code=data[7] if len(data) > 7 else 0,
			)
		if data[0] == MSG_NACK:
			return OtaStatus(True, OTA_STATUS_SEQ_ERROR, data[1] if len(data) > 1 else 0, 0, 0, 0)
	return None


def request_ota_status(bus: Any, base_id: int, timeout_s: float = STATUS_TIMEOUT_S) -> OtaStatus | None:
	ids = derive_ids(base_id)
	send_can_message(bus, ids.ctrl_id, [CMD_GET_OTA_STATUS])
	return read_ota_status(bus, ids.stat_id, timeout_s)


def variant_name(value: int) -> str:
	if value == 0:
		return "7mm"
	if value == 1:
		return "DT"
	return f"unknown({value})"


def read_firmware_version(bus: Any, stat_id: int, timeout_s: float = VERSION_TIMEOUT_S) -> FirmwareVersion | None:
	deadline = time.monotonic() + timeout_s
	chunks: dict[int, str] = {}
	total_chunks: int | None = None
	variant = "unknown"
	while time.monotonic() < deadline:
		message = bus.recv(min(0.05, max(0.0, deadline - time.monotonic())))
		if message is None or getattr(message, "is_extended_id", False):
			continue
		if message.arbitration_id != stat_id:
			continue
		data = bytes(message.data)
		if len(data) < 4 or data[0] != MSG_FW_VERSION:
			continue
		chunk_index = data[1]
		total_chunks = max(1, data[2])
		variant = variant_name(data[3])
		if chunk_index >= total_chunks:
			continue
		chunks[chunk_index] = bytes(byte for byte in data[4:8] if byte != 0).decode("ascii", errors="replace")
		if len(chunks) >= total_chunks:
			break
	if not chunks or total_chunks is None:
		return None
	version = "".join(chunks.get(index, "") for index in range(total_chunks))
	return FirmwareVersion(version=version or "unknown", variant=variant, chunks=total_chunks)


def request_firmware_version(bus: Any, base_id: int, timeout_s: float = VERSION_TIMEOUT_S) -> FirmwareVersion | None:
	ids = derive_ids(base_id)
	send_can_message(bus, ids.ctrl_id, [CMD_GET_FW_VERSION])
	return read_firmware_version(bus, ids.stat_id, timeout_s)


def request_firmware_versions(bus: Any, base_ids: Iterable[int], log: LogFn = print) -> dict[int, FirmwareVersion | None]:
	versions: dict[int, FirmwareVersion | None] = {}
	for base_id in base_ids:
		version = request_firmware_version(bus, base_id)
		versions[base_id] = version
		if version is None:
			log(f"0x{base_id:03X}: firmware version unavailable")
		else:
			log(f"0x{base_id:03X}: firmware version {version.version} ({version.variant})")
	return versions


def scan_boards(bus: Any, target_ids: Iterable[int], timeout_s: float, log: LogFn = print) -> list[int]:
	ids = list(dict.fromkeys(target_ids))
	stat_to_base = {derive_ids(base_id).stat_id: base_id for base_id in ids}
	detected: set[int] = set()
	drain_bus(bus)
	for base_id in ids:
		send_can_message(bus, derive_ids(base_id).ctrl_id, [CMD_GET_CAN_DIAG])
		time.sleep(0.003)

	deadline = time.monotonic() + timeout_s
	while time.monotonic() < deadline and len(detected) < len(ids):
		message = bus.recv(min(0.1, max(0.0, deadline - time.monotonic())))
		if message is None or getattr(message, "is_extended_id", False):
			continue
		base_id = stat_to_base.get(message.arbitration_id)
		if base_id is None:
			continue
		data = bytes(message.data)
		if data and data[0] in {MSG_CAN_DIAG0, MSG_CAN_DIAG1, MSG_CAN_DIAG2, MSG_CAN_DIAG3, MSG_ACK, MSG_NACK, MSG_OTA_STATUS}:
			detected.add(base_id)

	boards = sorted(detected)
	log("Detected boards: " + (", ".join(f"0x{base_id:03X}" for base_id in boards) if boards else "none"))
	return boards


def block_is_accepted(status: OtaStatus | None, block_index: int, total_frames: int, block_len: int) -> bool:
	if status is None:
		return False
	if (status.flags & OTA_ERROR_MASK) != 0:
		return False
	if total_frames >= FRAMES_PER_BLOCK:
		return status.expected_seq == 0 and status.buffer_index == 0 and status.blocks_written >= block_index + 1
	return status.expected_seq == total_frames and status.buffer_index == block_len


def sequence_from_status(status: OtaStatus | None, total_frames: int) -> int:
	if status is None:
		return 0
	if status.expected_seq < 0 or status.expected_seq >= total_frames:
		return 0
	return status.expected_seq


def group_uploads_by_firmware(upload_items: Iterable[BoardUpload]) -> dict[Path, list[int]]:
	groups: dict[Path, list[int]] = defaultdict(list)
	for item in upload_items:
		groups[item.firmware_path].append(item.base_id)
	return dict(groups)


class UnicastCanOtaClient:
	def __init__(self, bus: Any, device_ids: DeviceIds, log: LogFn = print, cancel_event: threading.Event | None = None) -> None:
		self.bus = bus
		self.ids = device_ids
		self.log = log
		self.cancel_event = cancel_event

	def _check_cancelled(self) -> None:
		if self.cancel_event is not None and self.cancel_event.is_set():
			raise RuntimeError("Upload cancelled")

	def send_control(self, command: int) -> None:
		send_can_message(self.bus, self.ids.ctrl_id, [command])

	def send_data(self, sequence: int, payload: bytes) -> None:
		send_can_message(self.bus, self.ids.data_id, [sequence, *payload])

	def upload(self, firmware_path: Path, retries: int, progress: ProgressFn | None = None) -> bool:
		if not firmware_path.is_file():
			self.log(f"[ERROR] Firmware file not found: {firmware_path}")
			return False

		firmware_data = firmware_path.read_bytes()
		if not firmware_data:
			self.log(f"[ERROR] Firmware file is empty: {firmware_path}")
			return False

		total_blocks = math.ceil(len(firmware_data) / BLOCK_BYTES)
		self.log(
			f"0x{self.ids.base_id:03X}: unicast firmware={firmware_path} bytes={len(firmware_data)} "
			f"ctrl=0x{self.ids.ctrl_id:03X} data=0x{self.ids.data_id:03X} stat=0x{self.ids.stat_id:03X}"
		)

		if progress:
			progress(self.ids.base_id, "starting", 0.0)
		self.send_control(CMD_START)
		success, _requested_seq = wait_for_ack(self.bus, self.ids.stat_id, START_ACK_TIMEOUT_S)
		if not success:
			self.log(f"[ERROR] 0x{self.ids.base_id:03X}: no START ACK")
			return False

		for block_index in range(total_blocks):
			self._check_cancelled()
			block_start = block_index * BLOCK_BYTES
			block = firmware_data[block_start:block_start + BLOCK_BYTES]
			total_frames = math.ceil(len(block) / DATA_PER_FRAME)
			retry_count = 0
			requested_seq = 0

			while retry_count <= retries:
				self.log(
					f"0x{self.ids.base_id:03X}: block {block_index + 1}/{total_blocks}, "
					f"frames {requested_seq}..{total_frames - 1}"
				)
				for sequence in range(requested_seq, total_frames):
					self._check_cancelled()
					frame_start = sequence * DATA_PER_FRAME
					self.send_data(sequence, block[frame_start:frame_start + DATA_PER_FRAME])
					time.sleep(FRAME_GAP_S)

				if total_frames >= FRAMES_PER_BLOCK:
					success, requested_seq = wait_for_ack(self.bus, self.ids.stat_id, ACK_TIMEOUT_S)
					if success:
						break
				else:
					break

				retry_count += 1
				if requested_seq < 0 or requested_seq >= total_frames:
					requested_seq = 0
				self.log(
					f"0x{self.ids.base_id:03X}: block retry {retry_count}/{retries}, "
					f"requested seq={requested_seq}"
				)

			if retry_count > retries:
				self.log(f"[ERROR] 0x{self.ids.base_id:03X}: failed block {block_index + 1}")
				return False
			if progress:
				progress(self.ids.base_id, "receiving", (block_index + 1) / total_blocks)

		if progress:
			progress(self.ids.base_id, "finalizing", 1.0)
		self.send_control(CMD_END)
		success, _requested_seq = wait_for_ack(self.bus, self.ids.stat_id, END_ACK_TIMEOUT_S)
		if not success:
			self.log(f"[ERROR] 0x{self.ids.base_id:03X}: no END ACK")
			return False

		if progress:
			progress(self.ids.base_id, "done", 1.0)
		self.log(f"0x{self.ids.base_id:03X}: OTA successful")
		return True


class BroadcastCanOtaClient:
	def __init__(
		self,
		bus: Any,
		log: LogFn = print,
		progress: ProgressFn | None = None,
		cancel_event: threading.Event | None = None,
	) -> None:
		self.bus = bus
		self.log = log
		self.progress = progress
		self.cancel_event = cancel_event

	def _check_cancelled(self) -> None:
		if self.cancel_event is not None and self.cancel_event.is_set():
			raise RuntimeError("Upload cancelled")

	def _progress(self, base_id: int, state: str, fraction: float) -> None:
		if self.progress:
			self.progress(base_id, state, max(0.0, min(1.0, fraction)))

	def send_control(self, base_id: int, command: int) -> None:
		send_can_message(self.bus, derive_ids(base_id).ctrl_id, [command])

	def send_unicast_data(self, base_id: int, sequence: int, payload: bytes) -> None:
		send_can_message(self.bus, derive_ids(base_id).data_id, [sequence, *payload])

	def send_broadcast_data(self, sequence: int, payload: bytes) -> None:
		send_can_message(self.bus, BROADCAST_CAN_ID, [sequence, *payload])

	def poll_status(self, base_id: int) -> OtaStatus | None:
		return request_ota_status(self.bus, base_id)

	def start_boards(self, base_ids: list[int]) -> list[int]:
		failures: list[int] = []
		for base_id in base_ids:
			self._check_cancelled()
			self._progress(base_id, "starting", 0.0)
			self.log(f"0x{base_id:03X}: sending START")
			self.send_control(base_id, CMD_START)
			success, _requested_seq = wait_for_ack(self.bus, derive_ids(base_id).stat_id, START_ACK_TIMEOUT_S)
			if not success:
				self.log(f"[ERROR] 0x{base_id:03X}: no START ACK")
				self._progress(base_id, "failed", 0.0)
				failures.append(base_id)
		return failures

	def repair_board(self, base_id: int, block: bytes, start_sequence: int, total_frames: int) -> None:
		self.log(f"0x{base_id:03X}: repair frames {start_sequence}..{total_frames - 1}")
		for sequence in range(start_sequence, total_frames):
			self._check_cancelled()
			frame_start = sequence * DATA_PER_FRAME
			self.send_unicast_data(base_id, sequence, block[frame_start:frame_start + DATA_PER_FRAME])
			time.sleep(FRAME_GAP_S)
		if total_frames >= FRAMES_PER_BLOCK:
			wait_for_ack(self.bus, derive_ids(base_id).stat_id, ACK_TIMEOUT_S)

	def upload_group(self, firmware_path: Path, base_ids: Iterable[int], retries: int) -> bool:
		selected_ids = list(dict.fromkeys(base_ids))
		if not selected_ids:
			return True
		if not firmware_path.is_file():
			self.log(f"[ERROR] Firmware file not found: {firmware_path}")
			return False

		firmware_data = firmware_path.read_bytes()
		if not firmware_data:
			self.log(f"[ERROR] Firmware file is empty: {firmware_path}")
			return False

		total_blocks = math.ceil(len(firmware_data) / BLOCK_BYTES)
		self.log(
			f"Broadcast batch: boards={', '.join(f'0x{base_id:03X}' for base_id in selected_ids)} "
			f"firmware={firmware_path} bytes={len(firmware_data)} blocks={total_blocks}"
		)

		start_failures = self.start_boards(selected_ids)
		active_ids = [base_id for base_id in selected_ids if base_id not in start_failures]
		if not active_ids:
			return False

		failed_ids: set[int] = set(start_failures)
		for block_index in range(total_blocks):
			self._check_cancelled()
			block_start = block_index * BLOCK_BYTES
			block = firmware_data[block_start:block_start + BLOCK_BYTES]
			total_frames = math.ceil(len(block) / DATA_PER_FRAME)
			self.log(f"Broadcast block {block_index + 1}/{total_blocks}: {total_frames} frames")

			for sequence in range(total_frames):
				self._check_cancelled()
				frame_start = sequence * DATA_PER_FRAME
				self.send_broadcast_data(sequence, block[frame_start:frame_start + DATA_PER_FRAME])
				time.sleep(FRAME_GAP_S)

			statuses = {base_id: self.poll_status(base_id) for base_id in active_ids if base_id not in failed_ids}
			retry_count = 0
			while retry_count <= retries:
				pending = [
					base_id for base_id, status in statuses.items()
					if not block_is_accepted(status, block_index, total_frames, len(block))
				]
				if not pending:
					break
				if retry_count == retries:
					for base_id in pending:
						self.log(f"[ERROR] 0x{base_id:03X}: failed block {block_index + 1}")
						self._progress(base_id, "failed", block_index / total_blocks)
						failed_ids.add(base_id)
					break

				retry_count += 1
				for base_id in pending:
					self._check_cancelled()
					status = statuses.get(base_id)
					start_sequence = sequence_from_status(status, total_frames)
					self._progress(base_id, "repair", block_index / total_blocks)
					self.repair_board(base_id, block, start_sequence, total_frames)
					statuses[base_id] = self.poll_status(base_id)

			for base_id in active_ids:
				if base_id not in failed_ids:
					self._progress(base_id, "receiving", (block_index + 1) / total_blocks)

			if len(failed_ids) == len(selected_ids):
				return False

		for base_id in active_ids:
			if base_id in failed_ids:
				continue
			self._check_cancelled()
			self._progress(base_id, "finalizing", 1.0)
			self.log(f"0x{base_id:03X}: sending END")
			self.send_control(base_id, CMD_END)
			success, _requested_seq = wait_for_ack(self.bus, derive_ids(base_id).stat_id, END_ACK_TIMEOUT_S)
			if success:
				self._progress(base_id, "done", 1.0)
				self.log(f"0x{base_id:03X}: OTA successful")
			else:
				self._progress(base_id, "failed", 1.0)
				self.log(f"[ERROR] 0x{base_id:03X}: no END ACK")
				failed_ids.add(base_id)

		return not failed_ids


class BoardRow:
	def __init__(self, app: CanOtaGui, parent: ttk.Frame, base_id: int, row_index: int, version: FirmwareVersion | None = None) -> None:
		self.app = app
		self.base_id = base_id
		self.selected = tk.BooleanVar(value=True)
		self.kind = tk.StringVar(value=firmware_kind_for_id(base_id))
		self.version_text = tk.StringVar(value="-")
		self.custom_path = tk.StringVar(value="")
		self.state = tk.StringVar(value="detected")
		self.progress = tk.DoubleVar(value=0.0)
		self.path_text = tk.StringVar(value="")

		self.checkbox = ttk.Checkbutton(parent, variable=self.selected)
		self.checkbox.grid(row=row_index, column=0, sticky="w", padx=4, pady=2)
		ttk.Label(parent, text=f"0x{base_id:03X}").grid(row=row_index, column=1, sticky="w", padx=4, pady=2)
		ttk.Label(parent, text=firmware_kind_for_id(base_id)).grid(row=row_index, column=2, sticky="w", padx=4, pady=2)
		ttk.Label(parent, textvariable=self.version_text, width=16).grid(row=row_index, column=3, sticky="w", padx=4, pady=2)
		self.kind_combo = ttk.Combobox(parent, textvariable=self.kind, values=["DT", "7mm", "Custom"], width=8, state="readonly")
		self.kind_combo.grid(row=row_index, column=4, sticky="w", padx=4, pady=2)
		self.custom_button = ttk.Button(parent, text="Browse", command=self.choose_custom, width=8)
		self.custom_button.grid(row=row_index, column=5, sticky="w", padx=4, pady=2)
		ttk.Label(parent, textvariable=self.path_text, width=34).grid(row=row_index, column=6, sticky="w", padx=4, pady=2)
		ttk.Label(parent, textvariable=self.state, width=12).grid(row=row_index, column=7, sticky="w", padx=4, pady=2)
		self.progress_bar = ttk.Progressbar(parent, variable=self.progress, maximum=1.0, length=120)
		self.progress_bar.grid(row=row_index, column=8, sticky="ew", padx=4, pady=2)

		self.kind.trace_add("write", lambda *_args: self.refresh_path_label())
		self.custom_path.trace_add("write", lambda *_args: self.refresh_path_label())
		self.refresh_path_label()
		self.set_version(version)

	def set_version(self, version: FirmwareVersion | None) -> None:
		if version is None:
			self.version_text.set("unavailable")
		else:
			self.version_text.set(f"{version.version} {version.variant}")

	def choose_custom(self) -> None:
		path = filedialog.askopenfilename(
			title=f"Firmware for 0x{self.base_id:03X}",
			filetypes=[("Firmware binaries", "*.bin"), ("All files", "*.*")],
		)
		if path:
			self.custom_path.set(path)
			self.kind.set("Custom")

	def firmware_path(self) -> Path:
		kind = self.kind.get()
		if kind == "DT":
			return Path(self.app.dt_bin.get())
		if kind == "7mm":
			return Path(self.app.seven_mm_bin.get())
		return Path(self.custom_path.get())

	def refresh_path_label(self) -> None:
		path = self.firmware_path()
		self.path_text.set(path.name if str(path) else "")

	def set_state(self, state: str, fraction: float | None = None) -> None:
		self.state.set(state)
		if fraction is not None:
			self.progress.set(max(0.0, min(1.0, fraction)))


class CanOtaGui(tk.Tk):
	def __init__(self) -> None:
		super().__init__()
		self.title("VNEMA CAN OTA")
		self.geometry("1180x720")
		self.minsize(1020, 620)

		self.event_queue: queue.Queue[dict[str, object]] = queue.Queue()
		self.rows: dict[int, BoardRow] = {}
		self.worker_thread: threading.Thread | None = None
		self.cancel_event = threading.Event()

		self.port = tk.StringVar(value=DEFAULT_PORT)
		self.bitrate = tk.StringVar(value=str(DEFAULT_CAN_BITRATE))
		self.tty_baudrate = tk.StringVar(value=str(DEFAULT_TTY_BAUDRATE))
		self.dt_bin = tk.StringVar(value=str(DEFAULT_DT_BIN))
		self.seven_mm_bin = tk.StringVar(value=str(DEFAULT_SEVEN_MM_BIN))
		self.retries = tk.StringVar(value="3")
		self.status = tk.StringVar(value="Idle")

		self._build_ui()
		self.refresh_ports()
		self.dt_bin.trace_add("write", lambda *_args: self.refresh_all_path_labels())
		self.seven_mm_bin.trace_add("write", lambda *_args: self.refresh_all_path_labels())
		self.after(100, self.process_events)

	def _build_ui(self) -> None:
		self.columnconfigure(0, weight=1)
		self.rowconfigure(2, weight=1)
		connection = ttk.LabelFrame(self, text="Connection")
		connection.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
		for column in range(9):
			connection.columnconfigure(column, weight=0)
		connection.columnconfigure(1, weight=1)

		ttk.Label(connection, text="Port").grid(row=0, column=0, sticky="w", padx=6, pady=6)
		self.port_combo = ttk.Combobox(connection, textvariable=self.port, width=18)
		self.port_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
		ttk.Button(connection, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=6, pady=6)
		ttk.Label(connection, text="CAN bitrate").grid(row=0, column=3, sticky="w", padx=6, pady=6)
		ttk.Entry(connection, textvariable=self.bitrate, width=12).grid(row=0, column=4, padx=6, pady=6)
		ttk.Label(connection, text="TTY baud").grid(row=0, column=5, sticky="w", padx=6, pady=6)
		ttk.Entry(connection, textvariable=self.tty_baudrate, width=12).grid(row=0, column=6, padx=6, pady=6)
		self.scan_button = ttk.Button(connection, text="Scan", command=self.start_scan)
		self.scan_button.grid(row=0, column=7, padx=6, pady=6)
		self.cancel_button = ttk.Button(connection, text="Cancel", command=self.cancel_upload, state=tk.DISABLED)
		self.cancel_button.grid(row=0, column=8, padx=6, pady=6)

		firmware = ttk.LabelFrame(self, text="Firmware")
		firmware.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
		firmware.columnconfigure(1, weight=1)
		firmware.columnconfigure(4, weight=1)
		ttk.Label(firmware, text="DT").grid(row=0, column=0, sticky="w", padx=6, pady=6)
		ttk.Entry(firmware, textvariable=self.dt_bin).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
		ttk.Button(firmware, text="Browse", command=lambda: self.choose_firmware(self.dt_bin)).grid(row=0, column=2, padx=6, pady=6)
		ttk.Label(firmware, text="7mm").grid(row=0, column=3, sticky="w", padx=6, pady=6)
		ttk.Entry(firmware, textvariable=self.seven_mm_bin).grid(row=0, column=4, sticky="ew", padx=6, pady=6)
		ttk.Button(firmware, text="Browse", command=lambda: self.choose_firmware(self.seven_mm_bin)).grid(row=0, column=5, padx=6, pady=6)
		ttk.Label(firmware, text="Retries").grid(row=0, column=6, sticky="w", padx=6, pady=6)
		ttk.Entry(firmware, textvariable=self.retries, width=6).grid(row=0, column=7, padx=6, pady=6)

		boards = ttk.LabelFrame(self, text="Boards")
		boards.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
		boards.columnconfigure(0, weight=1)
		boards.rowconfigure(1, weight=1)

		toolbar = ttk.Frame(boards)
		toolbar.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
		self.expected_button = ttk.Button(toolbar, text="Add Expected", command=lambda: self.populate_boards(list(EXPECTED_IDS)))
		self.expected_button.pack(side=tk.LEFT, padx=(0, 6))
		ttk.Button(toolbar, text="Select All", command=lambda: self.set_all_selected(True)).pack(side=tk.LEFT, padx=6)
		ttk.Button(toolbar, text="Clear", command=lambda: self.set_all_selected(False)).pack(side=tk.LEFT, padx=6)
		self.upload_button = ttk.Button(toolbar, text="Upload Selected", command=self.start_upload)
		self.upload_button.pack(side=tk.RIGHT)
		ttk.Label(toolbar, textvariable=self.status).pack(side=tk.RIGHT, padx=12)

		self.board_canvas = tk.Canvas(boards, highlightthickness=0)
		scrollbar = ttk.Scrollbar(boards, orient=tk.VERTICAL, command=self.board_canvas.yview)
		self.board_table = ttk.Frame(self.board_canvas)
		self.board_table.bind("<Configure>", lambda _event: self.board_canvas.configure(scrollregion=self.board_canvas.bbox("all")))
		self.board_canvas.create_window((0, 0), window=self.board_table, anchor="nw")
		self.board_canvas.configure(yscrollcommand=scrollbar.set)
		self.board_canvas.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=6)
		scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 6), pady=6)

		log_frame = ttk.LabelFrame(self, text="Log")
		log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(6, 10))
		log_frame.columnconfigure(0, weight=1)
		log_frame.rowconfigure(0, weight=1)
		self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
		self.log_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
		log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
		log_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
		self.log_text.configure(yscrollcommand=log_scroll.set)

		self.render_board_header()

	def render_board_header(self) -> None:
		for widget in self.board_table.winfo_children():
			widget.destroy()
		headers = ["", "ID", "Detected Type", "Running Version", "Firmware", "Custom", "Image", "State", "Progress"]
		for column, text in enumerate(headers):
			ttk.Label(self.board_table, text=text, font=("Segoe UI", 9, "bold")).grid(row=0, column=column, sticky="w", padx=4, pady=4)

	def refresh_ports(self) -> None:
		ports = list_serial_ports()
		self.port_combo["values"] = ports
		if ports:
			if self.port.get() not in ports:
				self.port.set(ports[0])
		else:
			self.port.set("")

	def choose_firmware(self, variable: tk.StringVar) -> None:
		path = filedialog.askopenfilename(title="Firmware image", filetypes=[("Firmware binaries", "*.bin"), ("All files", "*.*")])
		if path:
			variable.set(path)

	def populate_boards(self, base_ids: list[int], versions: dict[int, FirmwareVersion | None] | None = None) -> None:
		versions = versions or {}
		self.rows.clear()
		self.render_board_header()
		for row_index, base_id in enumerate(sorted(dict.fromkeys(base_ids)), start=1):
			self.rows[base_id] = BoardRow(self, self.board_table, base_id, row_index, versions.get(base_id))
		self.status.set(f"{len(self.rows)} board(s)")

	def refresh_all_path_labels(self) -> None:
		for row in self.rows.values():
			row.refresh_path_label()

	def set_all_selected(self, selected: bool) -> None:
		for row in self.rows.values():
			row.selected.set(selected)

	def append_log(self, message: str) -> None:
		self.log_text.configure(state=tk.NORMAL)
		self.log_text.insert(tk.END, time.strftime("%H:%M:%S ") + message + "\n")
		self.log_text.see(tk.END)
		self.log_text.configure(state=tk.DISABLED)

	def set_busy(self, busy: bool, upload: bool = False) -> None:
		state = tk.DISABLED if busy else tk.NORMAL
		self.scan_button.configure(state=state)
		self.upload_button.configure(state=state)
		self.expected_button.configure(state=state)
		self.cancel_button.configure(state=(tk.NORMAL if upload else tk.DISABLED))

	def parse_connection(self) -> tuple[str, int, int, int]:
		port = self.port.get().strip()
		if not port:
			raise ValueError("No SLCAN COM port is selected")
		return port, int(self.bitrate.get()), int(self.tty_baudrate.get()), int(self.retries.get())

	def start_scan(self) -> None:
		try:
			port, bitrate, tty_baudrate, _retries = self.parse_connection()
		except ValueError as exception:
			messagebox.showerror("Invalid Connection", str(exception))
			return
		self.set_busy(True)
		self.status.set("Scanning")
		self.worker_thread = threading.Thread(target=self.scan_worker, args=(port, bitrate, tty_baudrate), daemon=True)
		self.worker_thread.start()

	def scan_worker(self, port: str, bitrate: int, tty_baudrate: int) -> None:
		bus = None
		try:
			self.event_queue.put({"type": "log", "message": f"Opening {port}"})
			bus = open_slcan_bus(port, bitrate, tty_baudrate)
			boards = scan_boards(bus, EXPECTED_IDS, 2.0, log=lambda message: self.event_queue.put({"type": "log", "message": message}))
			versions = request_firmware_versions(bus, boards, log=lambda message: self.event_queue.put({"type": "log", "message": message}))
			self.event_queue.put({"type": "scan_result", "boards": boards, "versions": versions})
		except Exception as exception:
			self.event_queue.put({"type": "error", "message": str(exception)})
		finally:
			if bus is not None:
				bus.shutdown()
			self.event_queue.put({"type": "idle"})

	def selected_uploads(self) -> list[BoardUpload]:
		uploads: list[BoardUpload] = []
		for base_id, row in sorted(self.rows.items()):
			if row.selected.get():
				if row.kind.get() == "Custom" and not row.custom_path.get().strip():
					raise ValueError(f"0x{base_id:03X} has no custom firmware image selected")
				path = row.firmware_path()
				if not str(path):
					raise ValueError(f"0x{base_id:03X} has no firmware image selected")
				uploads.append(BoardUpload(base_id, path))
		return uploads

	def start_upload(self) -> None:
		try:
			port, bitrate, tty_baudrate, retries = self.parse_connection()
			uploads = self.selected_uploads()
		except ValueError as exception:
			messagebox.showerror("Invalid Upload", str(exception))
			return
		if not uploads:
			messagebox.showinfo("No Boards", "No boards are selected.")
			return
		missing = sorted({str(item.firmware_path) for item in uploads if not item.firmware_path.is_file()})
		if missing:
			messagebox.showerror("Missing Firmware", "\n".join(missing))
			return

		for item in uploads:
			self.rows[item.base_id].set_state("queued", 0.0)
		self.cancel_event.clear()
		self.set_busy(True, upload=True)
		self.status.set("Uploading")
		self.worker_thread = threading.Thread(
			target=self.upload_worker,
			args=(port, bitrate, tty_baudrate, retries, uploads),
			daemon=True,
		)
		self.worker_thread.start()

	def upload_worker(self, port: str, bitrate: int, tty_baudrate: int, retries: int, uploads: list[BoardUpload]) -> None:
		bus = None
		try:
			bus = open_slcan_bus(port, bitrate, tty_baudrate)
			client = BroadcastCanOtaClient(
				bus,
				log=lambda message: self.event_queue.put({"type": "log", "message": message}),
				progress=lambda base_id, state, fraction: self.event_queue.put(
					{"type": "progress", "base_id": base_id, "state": state, "fraction": fraction}
				),
				cancel_event=self.cancel_event,
			)
			all_ok = True
			for firmware_path, base_ids in group_uploads_by_firmware(uploads).items():
				if self.cancel_event.is_set():
					raise RuntimeError("Upload cancelled")
				batch_ok = client.upload_group(firmware_path, base_ids, retries)
				all_ok = all_ok and batch_ok
			if all_ok:
				time.sleep(1.0)
				versions = request_firmware_versions(
					bus,
					[item.base_id for item in uploads],
					log=lambda message: self.event_queue.put({"type": "log", "message": message}),
				)
				self.event_queue.put({"type": "version_update", "versions": versions})
			self.event_queue.put({"type": "log", "message": "Upload complete" if all_ok else "Upload finished with failures"})
		except Exception as exception:
			self.event_queue.put({"type": "error", "message": str(exception)})
		finally:
			if bus is not None:
				bus.shutdown()
			self.event_queue.put({"type": "idle"})

	def cancel_upload(self) -> None:
		self.cancel_event.set()
		self.status.set("Cancelling")

	def process_events(self) -> None:
		try:
			while True:
				event = self.event_queue.get_nowait()
				event_type = event.get("type")
				if event_type == "log":
					self.append_log(str(event.get("message", "")))
				elif event_type == "error":
					self.append_log("[ERROR] " + str(event.get("message", "")))
					self.status.set("Error")
				elif event_type == "scan_result":
					boards = [int(item) for item in event.get("boards", [])]
					versions = event.get("versions", {})
					self.populate_boards(boards, versions if isinstance(versions, dict) else {})
				elif event_type == "version_update":
					versions = event.get("versions", {})
					if isinstance(versions, dict):
						for base_id, version in versions.items():
							row = self.rows.get(int(base_id))
							if row is not None:
								row.set_version(version)
				elif event_type == "progress":
					base_id = int(event.get("base_id", 0))
					row = self.rows.get(base_id)
					if row is not None:
						row.set_state(str(event.get("state", "")), float(event.get("fraction", 0.0)))
				elif event_type == "idle":
					self.set_busy(False)
					if self.status.get() not in {"Error"}:
						self.status.set("Idle")
		except queue.Empty:
			pass
		self.after(100, self.process_events)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CAN OTA uploader and GUI for VNEMA MK8 boards")
	parser.add_argument("--gui", action="store_true", help="Launch the Tkinter OTA GUI")
	parser.add_argument("--cli", action="store_true", help="Run command-line upload mode")
	parser.add_argument("--dry-run-gui", action="store_true", help="Validate GUI dependencies without opening a window")
	parser.add_argument("--ids", nargs="+", default=None, help="Targets: all, dt, 7mm, ranges, or IDs like 0x101 0x102")
	parser.add_argument("--port", default=DEFAULT_PORT)
	parser.add_argument("--bitrate", type=int, default=DEFAULT_CAN_BITRATE)
	parser.add_argument("--tty-baudrate", type=int, default=DEFAULT_TTY_BAUDRATE)
	parser.add_argument("--dt-bin", type=Path, default=DEFAULT_DT_BIN)
	parser.add_argument("--seven-mm-bin", type=Path, default=DEFAULT_SEVEN_MM_BIN)
	parser.add_argument("--firmware", type=Path, default=None, help="Override binary for all selected IDs")
	parser.add_argument("--retries", type=int, default=3)
	parser.add_argument("--unicast", action="store_true", help="Use per-board unicast upload instead of broadcast batches")
	parser.add_argument("--continue-on-fail", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser


def run_cli(args: argparse.Namespace) -> int:
	if args.ids is None:
		raise ValueError("Specify --ids for CLI mode, for example: --ids 0x114, --ids dt, or --ids all")
	target_ids = expand_id_group(args.ids)
	if not target_ids:
		raise ValueError("No target IDs selected")

	uploads = [
		BoardUpload(
			base_id=base_id,
			firmware_path=args.firmware if args.firmware is not None else firmware_for_id(base_id, args.dt_bin, args.seven_mm_bin),
		)
		for base_id in target_ids
	]

	print("=== VNEMA CAN OTA ===")
	print(f"Port={args.port} bitrate={args.bitrate} tty={args.tty_baudrate}")
	print("Targets:", ", ".join(f"0x{item.base_id:03X}->{item.firmware_path}" for item in uploads))

	if args.dry_run:
		for firmware_path, base_ids in group_uploads_by_firmware(uploads).items():
			print(f"Batch {firmware_path}:", ", ".join(f"0x{base_id:03X}" for base_id in base_ids))
		return 0

	bus = open_slcan_bus(args.port, args.bitrate, args.tty_baudrate)
	failures: list[int] = []
	try:
		if args.unicast:
			for item in uploads:
				client = UnicastCanOtaClient(bus, derive_ids(item.base_id))
				if not client.upload(item.firmware_path, args.retries):
					failures.append(item.base_id)
					if not args.continue_on_fail:
						break
		else:
			client = BroadcastCanOtaClient(bus)
			for firmware_path, base_ids in group_uploads_by_firmware(uploads).items():
				if not client.upload_group(firmware_path, base_ids, args.retries):
					failures.extend(base_ids)
					if not args.continue_on_fail:
						break
	finally:
		bus.shutdown()

	if failures:
		print("\nFailed:", ", ".join(f"0x{base_id:03X}" for base_id in sorted(set(failures))))
		return 1
	print("\nAll selected OTA operations completed")
	return 0


def launch_gui() -> int:
	app = CanOtaGui()
	app.mainloop()
	return 0


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()
	try:
		if args.dry_run_gui:
			ports = ", ".join(list_serial_ports()) or "none"
			print(f"GUI dependencies OK. Detected ports: {ports}")
			return 0
		if args.gui or (not args.cli and args.ids is None):
			return launch_gui()
		return run_cli(args)
	except Exception as exception:
		print(f"[ERROR] {exception}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	raise SystemExit(main())