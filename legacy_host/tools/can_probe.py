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
BROADCAST_SYNC_ID = 0x090
DEFAULT_IDS = range(0x101, 0x109)


def parse_int_auto(raw: str) -> int:
    return int(raw.strip(), 0)


def parse_ids(tokens: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for token in tokens:
        for part in token.split(","):
            item = part.strip()
            if not item:
                continue
            if "-" in item:
                first_raw, last_raw = item.split("-", 1)
                first = parse_int_auto(first_raw)
                last = parse_int_auto(last_raw)
                ids.extend(range(first, last + 1))
            elif item.lower() == "dt":
                ids.extend(DEFAULT_IDS)
            else:
                ids.append(parse_int_auto(item))
    return list(dict.fromkeys(ids))


def compact_payload(target: int, flags: int) -> list[int]:
    payload = (target & 0x0FFF) | ((flags & 0x0F) << 12)
    return [payload & 0xFF, (payload >> 8) & 0xFF]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe compact VNEMA CAN replies")
    parser.add_argument("--ids", nargs="+", default=["dt"])
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--bitrate", type=int, default=DEFAULT_CAN_BITRATE)
    parser.add_argument("--tty-baudrate", type=int, default=DEFAULT_TTY_BAUDRATE)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--flags", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--tx-offset", type=lambda raw: int(raw, 0), default=0)
    parser.add_argument("--reply-offset", type=lambda raw: int(raw, 0), default=0)
    args = parser.parse_args()

    ids = parse_ids(args.ids)
    bus = can.interface.Bus(
        interface="slcan",
        channel=args.port,
        bitrate=args.bitrate,
        tty_baudrate=args.tty_baudrate,
    )
    try:
        if args.sync:
            bus.send(can.Message(arbitration_id=BROADCAST_SYNC_ID, data=[], is_extended_id=False))
        data = compact_payload(args.target, args.flags)
        for base_id in ids:
            bus.send(can.Message(arbitration_id=base_id + args.tx_offset, data=data, is_extended_id=False))

        deadline = time.monotonic() + args.timeout
        replies: dict[int, list[int]] = {}
        reply_ids = {base_id + args.reply_offset: base_id for base_id in ids}
        while time.monotonic() < deadline and len(replies) < len(ids):
            msg = bus.recv(min(0.1, max(0.0, deadline - time.monotonic())))
            if msg is None or msg.is_extended_id:
                continue
            if msg.arbitration_id in reply_ids:
                replies[reply_ids[msg.arbitration_id]] = list(msg.data)

        for base_id in ids:
            if base_id in replies:
                print(f"0x{base_id:03X}: reply {replies[base_id]}")
            else:
                print(f"0x{base_id:03X}: no reply")
    finally:
        bus.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())