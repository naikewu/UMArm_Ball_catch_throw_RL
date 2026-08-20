"""Wire protocol for the shared 24-actuator CAN bus.

One bus carries two kinds of board. Sixteen of them are the original
VNEMA_MK8_PIDPWM boards driving 7 mm valves; the top eight are TLE boards
driving Clippard DVP proportional valves. They speak the same protocol, and
this module is the single host-side definition of it. The firmware side lives
in ``main/tle_can_legacy.c``; the original specification is
``VNEMA_MK8_PIDPWM/portable_sync_comm_layer/docs/can_protocol.md``.

Everything is standard 11-bit CAN, DLC <= 8, little-endian, with one exception
noted at ``build_set_id``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
ID_BROADCAST = 0x090          # DLC 0 = sync edge; DLC > 0 = OTA image data
ID_RUNTIME_TABLE_BASE = 0x091  # 0x091..0x098, three actuators per frame

ACTUATOR_FIRST = 0x101
ACTUATOR_LAST = 0x118
ACTUATOR_COUNT = ACTUATOR_LAST - ACTUATOR_FIRST + 1  # 24

# Slots 0..7 are the big-valve positions the TLE manifold replaces; 8..23 stay
# on the original 7 mm boards.
TLE_IDS = range(0x101, 0x109)
SEVEN_MM_IDS = range(0x109, 0x119)
ALL_IDS = range(ACTUATOR_FIRST, ACTUATOR_LAST + 1)

OFFSET_HOST_CTRL = 0x100
OFFSET_HOST_DATA = 0x200
OFFSET_STATUS = 0x300
OFFSET_EXTENDED = 0x400   # TLE boards only; not part of the original protocol

RUNTIME_TABLE_SLOTS = 3
RUNTIME_TABLE_MARKER = 0x80
RUNTIME_TABLE_SLOTMASK = 0x1F

# ---------------------------------------------------------------------------
# Host control commands (base + 0x100) and board replies (base + 0x300)
# ---------------------------------------------------------------------------
CMD_START = 0x01
CMD_END = 0x02
CMD_SET_ID = 0x05
CMD_GET_CAN_DIAG = 0x06
CMD_CLEAR_CAN_DIAG = 0x07
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

# Compact command / status
PRESSURE_MASK = 0x0FFF
FLAGS_MASK = 0x000F
CONTROL_ENABLE = 0x01
STATUS_ENABLED = 0x01
STATUS_OTA_ACTIVE = 0x02
STATUS_COMMAND_SEEN = 0x04
STATUS_ERROR = 0x08

# Firmware variant byte in MSG_FW_VERSION. 0 and 1 are the original boards; 2
# is this project's TLE board, which the old host prints as unknown rather than
# mistaking it for either.
VARIANT_7MM = 0x00
VARIANT_DT = 0x01
VARIANT_TLE_DVP = 0x02
VARIANT_NAMES = {VARIANT_7MM: "7mm", VARIANT_DT: "DT", VARIANT_TLE_DVP: "TLE/DVP"}

# OTA framing, identical on both board types.
OTA_FRAMES_PER_BLOCK = 128
OTA_DATA_PER_FRAME = 7
OTA_BLOCK_BYTES = OTA_FRAMES_PER_BLOCK * OTA_DATA_PER_FRAME  # 896

CYCLE_HZ = 150.0


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeIds:
    base: int

    @property
    def ctrl(self) -> int:
        return self.base + OFFSET_HOST_CTRL

    @property
    def data(self) -> int:
        return self.base + OFFSET_HOST_DATA

    @property
    def status(self) -> int:
        return self.base + OFFSET_STATUS

    @property
    def extended(self) -> int:
        return self.base + OFFSET_EXTENDED

    @property
    def slot(self) -> int:
        return self.base - ACTUATOR_FIRST

    @property
    def table_id(self) -> int:
        return ID_RUNTIME_TABLE_BASE + (self.slot // RUNTIME_TABLE_SLOTS)


def ids(base: int) -> NodeIds:
    return NodeIds(base)


def status_id_to_base(status_id: int) -> int | None:
    base = status_id - OFFSET_STATUS
    return base if ACTUATOR_FIRST <= base <= ACTUATOR_LAST else None


def is_tle_slot(base: int) -> bool:
    return base in TLE_IDS


# ---------------------------------------------------------------------------
# Pressure scaling
# ---------------------------------------------------------------------------
# The compact field is 12 bits. A TLE board's sensor path is 16-bit, and the
# firmware puts raw >> 4 on the wire (main/tle_can_legacy.h, RAW_SHIFT), so one
# count is 16 raw counts, about 0.0165 psi. The 7 mm boards report their own
# 12-bit ADC directly and their psi mapping comes from the host's per-actuator
# calibration; the defaults below are the nominal transfer function documented
# in the old firmware (PBOT 760 = 0.1 psi, PCAP 3000 = 40 psi).
TLE_RAW_0_PSI = 15100.0
TLE_COUNTS_PER_PSI_RAW = 972.5
TLE_RAW_SHIFT = 4
TLE_ZERO_COUNTS = TLE_RAW_0_PSI / (1 << TLE_RAW_SHIFT)          # 943.75
TLE_COUNTS_PER_PSI = TLE_COUNTS_PER_PSI_RAW / (1 << TLE_RAW_SHIFT)  # 60.78125

LEGACY_ZERO_COUNTS = 754.4
LEGACY_COUNTS_PER_PSI = 56.14


@dataclass
class NodeCal:
    """Counts <-> psi for one actuator.

    A TLE board replacing a big-valve actuator does not inherit that slot's old
    calibration: different sensor, different plumbing. So calibration is
    per-node here and defaults by board kind, and the controller GUI can
    override it from a file the same way the original host does.
    """
    zero_counts: float = TLE_ZERO_COUNTS
    counts_per_psi: float = TLE_COUNTS_PER_PSI

    @classmethod
    def for_base(cls, base: int) -> "NodeCal":
        """Last resort: guess from the ID range. Prefer for_variant."""
        if is_tle_slot(base):
            return cls(TLE_ZERO_COUNTS, TLE_COUNTS_PER_PSI)
        return cls(LEGACY_ZERO_COUNTS, LEGACY_COUNTS_PER_PSI)

    @classmethod
    def for_variant(cls, variant: int, base: int) -> "NodeCal":
        """Scale from what the board reported, not from where it is addressed.

        A TLE board can legitimately sit anywhere in 0x101..0x118 -- it ships
        defaulting into the TLE block but an operator assigns the ID, and the
        bench board spent this session at 0x114. Choosing the transfer function
        from the ID range would then command and display it on the 7 mm scale,
        which is a 10 % error in psi at the top of the range.
        """
        if variant == VARIANT_TLE_DVP:
            return cls(TLE_ZERO_COUNTS, TLE_COUNTS_PER_PSI)
        if variant in (VARIANT_7MM, VARIANT_DT):
            return cls(LEGACY_ZERO_COUNTS, LEGACY_COUNTS_PER_PSI)
        return cls.for_base(base)

    def psi_to_counts(self, psi: float) -> int:
        return max(0, min(PRESSURE_MASK, int(round(self.zero_counts + psi * self.counts_per_psi))))

    def counts_to_psi(self, counts: float) -> float:
        return (counts - self.zero_counts) / self.counts_per_psi


def tle_counts_to_raw(counts: int) -> int:
    """Compact counts back to the 16-bit sensor scale the TLE firmware uses."""
    return (counts & PRESSURE_MASK) << TLE_RAW_SHIFT


def tle_raw_to_counts(raw: int) -> int:
    return min(PRESSURE_MASK, raw >> TLE_RAW_SHIFT)


# ---------------------------------------------------------------------------
# Compact command / status
# ---------------------------------------------------------------------------
def build_compact_command(counts: int, enable: bool = True) -> bytes:
    """Direct target for one actuator, sent to its base ID."""
    flags = CONTROL_ENABLE if enable else 0
    return struct.pack("<H", (counts & PRESSURE_MASK) | ((flags & FLAGS_MASK) << 12))


@dataclass(frozen=True)
class CompactStatus:
    base: int
    counts: int
    flags: int

    @property
    def enabled(self) -> bool:
        return bool(self.flags & STATUS_ENABLED)

    @property
    def ota_active(self) -> bool:
        return bool(self.flags & STATUS_OTA_ACTIVE)

    @property
    def command_seen(self) -> bool:
        return bool(self.flags & STATUS_COMMAND_SEEN)

    @property
    def error(self) -> bool:
        return bool(self.flags & STATUS_ERROR)


def parse_compact_status(can_id: int, data: bytes) -> CompactStatus | None:
    if len(data) != 2 or not (ACTUATOR_FIRST <= can_id <= ACTUATOR_LAST):
        return None
    value = struct.unpack("<H", data)[0]
    return CompactStatus(can_id, value & PRESSURE_MASK, (value >> 12) & FLAGS_MASK)


# ---------------------------------------------------------------------------
# Runtime target table
# ---------------------------------------------------------------------------
def build_runtime_table(targets: dict[int, tuple[int, bool]]) -> list[tuple[int, bytes]]:
    """Pack per-actuator targets into 8-byte group frames.

    ``targets`` maps base ID to (counts, enable). Only groups with at least one
    named actuator are emitted, so driving eight TLE boards costs three frames
    per cycle rather than eight.

    Returns [(can_id, payload)] in ascending group order. The 0x80 marker in
    byte 0 is what tells a board this is a table frame and not broadcast OTA
    data, whose first byte is a sequence number of 0..127.
    """
    groups: dict[int, list[tuple[int, int, bool]]] = {}
    for base, (counts, enable) in targets.items():
        if not (ACTUATOR_FIRST <= base <= ACTUATOR_LAST):
            raise ValueError(f"actuator base 0x{base:03X} outside 0x101..0x118")
        slot = base - ACTUATOR_FIRST
        groups.setdefault(slot // RUNTIME_TABLE_SLOTS, []).append(
            (slot % RUNTIME_TABLE_SLOTS, counts, enable))

    frames: list[tuple[int, bytes]] = []
    for group in sorted(groups):
        start_slot = group * RUNTIME_TABLE_SLOTS
        payload = bytearray(8)
        payload[0] = RUNTIME_TABLE_MARKER | (start_slot & RUNTIME_TABLE_SLOTMASK)
        mask = 0
        for offset, counts, enable in groups[group]:
            mask |= 1 << offset
            flags = CONTROL_ENABLE if enable else 0
            struct.pack_into("<H", payload, 2 + offset * 2,
                             (counts & PRESSURE_MASK) | ((flags & FLAGS_MASK) << 12))
        payload[1] = mask
        frames.append((ID_RUNTIME_TABLE_BASE + group, bytes(payload)))
    return frames


def build_sync() -> tuple[int, bytes]:
    """The 150 Hz edge: a zero-length frame on the broadcast ID."""
    return ID_BROADCAST, b""


# ---------------------------------------------------------------------------
# Host control
# ---------------------------------------------------------------------------
def build_set_id(new_base: int) -> bytes:
    """Persist a new base ID and reboot.

    The ID is big-endian here. Every other multi-byte field in this protocol is
    little-endian; this one field is not, and the firmware on both board types
    parses it that way, so it stays as it is.
    """
    return bytes([CMD_SET_ID, (new_base >> 8) & 0xFF, new_base & 0xFF])


def build_simple_command(command: int) -> bytes:
    return bytes([command])


def build_ota_data(sequence: int, chunk: bytes) -> bytes:
    if len(chunk) > OTA_DATA_PER_FRAME:
        raise ValueError("OTA chunk longer than 7 bytes")
    return bytes([sequence & 0xFF]) + chunk


@dataclass(frozen=True)
class OtaStatus:
    flags: int
    expected_seq: int
    buffer_index: int
    blocks_written: int
    error_code: int

    @property
    def active(self) -> bool:
        return bool(self.flags & OTA_STATUS_ACTIVE)

    @property
    def has_error(self) -> bool:
        return bool(self.flags & OTA_ERROR_MASK)


def parse_ota_status(data: bytes) -> OtaStatus | None:
    if len(data) < 8 or data[0] != MSG_OTA_STATUS:
        return None
    return OtaStatus(
        flags=data[1],
        expected_seq=data[2],
        buffer_index=struct.unpack_from("<H", data, 3)[0],
        blocks_written=struct.unpack_from("<H", data, 5)[0],
        error_code=data[7],
    )


@dataclass
class FirmwareVersion:
    version: str = ""
    variant: int = VARIANT_7MM
    chunks: int = 0
    _parts: dict[int, bytes] = field(default_factory=dict)

    @property
    def variant_name(self) -> str:
        return VARIANT_NAMES.get(self.variant, f"unknown({self.variant})")

    @property
    def complete(self) -> bool:
        return self.chunks > 0 and len(self._parts) >= self.chunks

    def feed(self, data: bytes) -> bool:
        """Merge one MSG_FW_VERSION frame. True once every chunk has arrived."""
        if len(data) < 8 or data[0] != MSG_FW_VERSION:
            return False
        self._parts[data[1]] = bytes(data[4:8])
        self.chunks = data[2]
        self.variant = data[3]
        if self.complete:
            blob = b"".join(self._parts[i] for i in sorted(self._parts))
            self.version = blob.rstrip(b"\x00").decode("ascii", errors="replace")
        return self.complete


@dataclass(frozen=True)
class CanDiag:
    """MSG_CAN_DIAG0..3 merged. Fields the TLE firmware does not keep read 0."""
    last_error_reason: int = 0
    last_eflg: int = 0
    error_count: int = 0
    warning_count: int = 0
    rx_overflow_count: int = 0
    tx_fail_count: int = 0
    invalid_frame_count: int = 0
    starvation_count: int = 0
    sync_counter: int = 0
    command_counter: int = 0
    control_byte: int = 0


def merge_can_diag(frames: dict[int, bytes]) -> CanDiag:
    d0 = frames.get(MSG_CAN_DIAG0, bytes(8))
    d1 = frames.get(MSG_CAN_DIAG1, bytes(8))
    d2 = frames.get(MSG_CAN_DIAG2, bytes(8))
    d3 = frames.get(MSG_CAN_DIAG3, bytes(8))
    u16 = lambda b, i: struct.unpack_from("<H", b, i)[0] if len(b) >= i + 2 else 0
    return CanDiag(
        last_error_reason=d0[1] if len(d0) > 1 else 0,
        last_eflg=d0[3] if len(d0) > 3 else 0,
        error_count=u16(d0, 6),
        warning_count=u16(d1, 1),
        rx_overflow_count=u16(d1, 3),
        tx_fail_count=u16(d1, 5),
        invalid_frame_count=u16(d2, 2),
        starvation_count=u16(d3, 1),
        sync_counter=u16(d3, 3),
        command_counter=u16(d3, 5),
        control_byte=d3[7] if len(d3) > 7 else 0,
    )


# ---------------------------------------------------------------------------
# Bus loading
# ---------------------------------------------------------------------------
def frame_bits(dlc: int, stuffing: bool = True) -> float:
    """Bits on the wire for one standard data frame, plus interframe space.

    A standard frame is 44 + 8*DLC bits. Of those, 34 + 8*DLC are subject to
    bit stuffing, which in the worst case inserts one bit per four. Real
    traffic stuffs far less than that, so `stuffing=True` is an upper bound and
    `stuffing=False` a lower one; the truth is near the lower end.
    """
    base = 44 + 8 * dlc + 3
    return base + (34 + 8 * dlc) / 4.0 if stuffing else base


def bus_load(n_tle: int = 8, n_legacy: int = 16, bitrate: int = 1_000_000,
             cycle_hz: float = CYCLE_HZ, stuffing: bool = True) -> float:
    """Fraction of the wire used by one steady-state control cycle.

    Per cycle the host sends one table frame per group of three actuators plus
    one sync, and every enabled board answers with a 2-byte status.
    """
    n = n_tle + n_legacy
    groups = -(-n // RUNTIME_TABLE_SLOTS)
    bits = (groups * frame_bits(8, stuffing) + frame_bits(0, stuffing)
            + n * frame_bits(2, stuffing))
    return bits * cycle_hz / bitrate
