from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import can


# --- workspace port resolution ---------------------------------------------
# Upstream hard-coded DEFAULT_PORT = "COM4". That port does not exist on this
# bench, and pinning any COM number repeats the same mistake, since Windows
# assigns the number per hub port and the dongle moves when it is replugged.
# bench_env at the workspace root resolves the adapter by its USB VID:PID and
# serial number instead, falling back to COM58.
_WS_ROOT = Path(__file__).resolve().parents[2]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))
try:
    import bench_env as _bench_env
except Exception:  # pragma: no cover - keep the tool usable without bench_env
    _bench_env = None

DEFAULT_PORT = _bench_env.resolve_can_port() if _bench_env is not None else "COM58"
DEFAULT_CAN_BITRATE = 1_000_000
DEFAULT_TTY_BAUDRATE = 2_000_000
DEFAULT_IDS = range(0x101, 0x109)

HOST_CTRL_OFFSET = 0x100
ESP_STAT_OFFSET = 0x300
CMD_GET_CAN_DIAG = 0x06
CMD_CLEAR_CAN_DIAG = 0x07
MSG_CAN_DIAG0 = 0xD0
MSG_CAN_DIAG1 = 0xD1
MSG_CAN_DIAG2 = 0xD2
MSG_CAN_DIAG3 = 0xD3

REASON_BITS = {
    0x01: "RX_OVERFLOW",
    0x02: "TX_ERROR",
    0x04: "BUS_ERROR",
    0x08: "BAD_FRAME",
    0x10: "SEND_FAIL",
    0x20: "STARVATION",
    0x40: "WARNING_ONLY",
    0x80: "MERR",
}

IRQ_BITS = {
    0x01: "RX0IF",
    0x02: "RX1IF",
    0x04: "TX0IF",
    0x08: "TX1IF",
    0x10: "TX2IF",
    0x20: "ERRIF",
    0x40: "WAKIF",
    0x80: "MERRF",
}

EFLG_BITS = {
    0x80: "RX1OVR",
    0x40: "RX0OVR",
    0x20: "TXBO",
    0x10: "TXEP",
    0x08: "RXEP",
    0x04: "TXWAR",
    0x02: "RXWAR",
    0x01: "EWARN",
}

SEND_ERRORS = {
    0: "ERROR_OK",
    1: "ERROR_FAIL",
    2: "ERROR_ALLTXBUSY",
    3: "ERROR_FAILINIT",
    4: "ERROR_FAILTX",
    5: "ERROR_NOMSG",
}


def parse_int_auto(raw: str) -> int:
    return int(raw.strip(), 0)


def parse_ids(tokens: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for token in tokens:
        for part in token.split(","):
            item = part.strip()
            if not item:
                continue
            if item.lower() == "dt":
                ids.extend(DEFAULT_IDS)
            elif "-" in item:
                first_raw, last_raw = item.split("-", 1)
                first = parse_int_auto(first_raw)
                last = parse_int_auto(last_raw)
                ids.extend(range(first, last + 1))
            else:
                ids.append(parse_int_auto(item))
    return list(dict.fromkeys(ids))


def names(mask: int, table: dict[int, str]) -> str:
    items = [name for bit, name in table.items() if mask & bit]
    return "|".join(items) if items else "none"


def u16(data: bytes | bytearray | list[int], index: int) -> int:
    return int(data[index]) | (int(data[index + 1]) << 8)


def send_command(bus: can.BusABC, base_id: int, command: int) -> None:
    bus.send(can.Message(arbitration_id=base_id + HOST_CTRL_OFFSET, data=[command], is_extended_id=False))


def collect_diag(bus: can.BusABC, ids: list[int], timeout: float) -> dict[int, dict[int, list[int]]]:
    deadline = time.monotonic() + timeout
    expected_reply_ids = {base_id + ESP_STAT_OFFSET: base_id for base_id in ids}
    frames: dict[int, dict[int, list[int]]] = {base_id: {} for base_id in ids}
    while time.monotonic() < deadline:
        if all({MSG_CAN_DIAG0, MSG_CAN_DIAG1, MSG_CAN_DIAG2, MSG_CAN_DIAG3}.issubset(board_frames) for board_frames in frames.values()):
            break
        msg = bus.recv(min(0.1, max(0.0, deadline - time.monotonic())))
        if msg is None or msg.is_extended_id or msg.arbitration_id not in expected_reply_ids or not msg.data:
            continue
        marker = int(msg.data[0])
        if marker in {MSG_CAN_DIAG0, MSG_CAN_DIAG1, MSG_CAN_DIAG2, MSG_CAN_DIAG3}:
            frames[expected_reply_ids[msg.arbitration_id]][marker] = list(msg.data)
    return frames


def print_diag(base_id: int, frames: dict[int, list[int]]) -> None:
    expected_markers = {MSG_CAN_DIAG0, MSG_CAN_DIAG1, MSG_CAN_DIAG2, MSG_CAN_DIAG3}
    d0 = frames.get(MSG_CAN_DIAG0)
    d1 = frames.get(MSG_CAN_DIAG1)
    d2 = frames.get(MSG_CAN_DIAG2)
    d3 = frames.get(MSG_CAN_DIAG3)
    if not d0:
        print(f"0x{base_id:03X}: no diagnostic reply")
        return

    reason = d0[1]
    irq = d0[2]
    eflg = d0[3]
    send_error = d0[4]
    status_flags = d0[5]
    error_count = u16(d0, 6)
    warning_count = u16(d1, 1) if d1 else 0
    rx_overflow_count = u16(d1, 3) if d1 else 0
    tx_fail_count = u16(d1, 5) if d1 else 0
    tx_all_busy_count = ((d2[1] << 8) | d1[7]) if d1 and d2 else 0
    invalid_frame_count = u16(d2, 2) if d2 else 0
    merr_count = u16(d2, 4) if d2 else 0
    errif_count = u16(d2, 6) if d2 else 0
    starvation_count = u16(d3, 1) if d3 else 0
    sync_count = u16(d3, 3) if d3 else 0
    command_count = u16(d3, 5) if d3 else 0
    active_control = d3[7] if d3 else 0

    print(f"0x{base_id:03X}:")
    missing = sorted(expected_markers - set(frames))
    if missing:
        missing_names = ",".join(f"0x{marker:02X}" for marker in missing)
        print(f"  missing_diag_frames={missing_names}")
    print(f"  last_reason=0x{reason:02X} {names(reason, REASON_BITS)}")
    print(f"  last_irq=0x{irq:02X} {names(irq, IRQ_BITS)}")
    print(f"  last_eflg=0x{eflg:02X} {names(eflg, EFLG_BITS)}")
    print(f"  last_send_error={send_error} {SEND_ERRORS.get(send_error, 'UNKNOWN')}")
    print(f"  status_flags=0x{status_flags:02X} error_count={error_count} warning_count={warning_count}")
    print(
        "  counters "
        f"rx_overflow={rx_overflow_count} tx_fail={tx_fail_count} tx_all_busy={tx_all_busy_count} "
        f"bad_frame={invalid_frame_count} merr={merr_count} errif={errif_count} starvation={starvation_count}"
    )
    print(f"  counters sync={sync_count} command={command_count} active_control=0x{active_control:02X}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query VNEMA firmware CAN diagnostic counters")
    parser.add_argument("--ids", nargs="+", default=["dt"], help="Board IDs, ranges, or 'dt' for 0x101-0x108")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--bitrate", type=int, default=DEFAULT_CAN_BITRATE)
    parser.add_argument("--tty-baudrate", type=int, default=DEFAULT_TTY_BAUDRATE)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--clear", action="store_true", help="Clear counters before reading them back")
    args = parser.parse_args()

    ids = parse_ids(args.ids)
    bus = can.interface.Bus(
        interface="slcan",
        channel=args.port,
        bitrate=args.bitrate,
        tty_baudrate=args.tty_baudrate,
    )
    try:
        for base_id in ids:
            send_command(bus, base_id, CMD_CLEAR_CAN_DIAG if args.clear else CMD_GET_CAN_DIAG)
        frames = collect_diag(bus, ids, args.timeout)
        for base_id in ids:
            print_diag(base_id, frames[base_id])
    finally:
        bus.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())