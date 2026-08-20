from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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
DEFAULT_IDS = list(range(0x101, 0x109))
BROADCAST_SYNC_ID = 0x090
RUNTIME_FIRST_ID = 0x101
RUNTIME_LAST_ID = 0x118
RUNTIME_TABLE_BASE_ID = 0x091
RUNTIME_TABLE_MARKER = 0x80
RUNTIME_TABLE_SLOT_MASK = 0x1F
RUNTIME_TABLE_SLOTS = 3
PRESSURE_MASK = 0x0FFF
FLAG_MASK = 0x000F
STATUS_ERROR = 0x08


@dataclass
class Reply:
    pressure: int = 0
    status: int = 0
    latency_ms: float = 0.0
    duplicate: int = 0


@dataclass
class BoardStats:
    board_id: int
    replies: int = 0
    misses: int = 0
    late_replies: int = 0
    duplicates: int = 0
    status_errors: int = 0
    dropout_events: int = 0
    recoveries: int = 0
    consecutive_misses: int = 0
    max_consecutive_misses: int = 0
    in_dropout: bool = False
    first_dropout_cycle: int | None = None
    dropout_spans: list[tuple[int, int | None]] = field(default_factory=list)
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0

    def mark_reply(self, cycle: int, reply: Reply) -> None:
        self.replies += 1
        self.status_errors += 1 if (reply.status & STATUS_ERROR) else 0
        self.duplicates += reply.duplicate
        self.latency_sum_ms += reply.latency_ms
        self.latency_max_ms = max(self.latency_max_ms, reply.latency_ms)
        if self.in_dropout:
            self.recoveries += 1
            if self.dropout_spans and self.dropout_spans[-1][1] is None:
                self.dropout_spans[-1] = (self.dropout_spans[-1][0], cycle)
        self.in_dropout = False
        self.consecutive_misses = 0

    def mark_miss(self, cycle: int, dropout_threshold: int) -> None:
        self.misses += 1
        self.consecutive_misses += 1
        self.max_consecutive_misses = max(self.max_consecutive_misses, self.consecutive_misses)
        if self.consecutive_misses == dropout_threshold:
            self.dropout_events += 1
            self.in_dropout = True
            start_cycle = cycle - dropout_threshold + 1
            if self.first_dropout_cycle is None:
                self.first_dropout_cycle = start_cycle
            self.dropout_spans.append((start_cycle, None))

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_sum_ms / self.replies if self.replies else 0.0


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
            elif item.lower() in {"all", "all24"}:
                ids.extend(range(RUNTIME_FIRST_ID, RUNTIME_LAST_ID + 1))
            elif item.lower() in {"all23", "no10c"}:
                ids.extend(board_id for board_id in range(RUNTIME_FIRST_ID, RUNTIME_LAST_ID + 1) if board_id != 0x10C)
            elif "-" in item:
                first_raw, last_raw = item.split("-", 1)
                first = parse_int_auto(first_raw)
                last = parse_int_auto(last_raw)
                ids.extend(range(first, last + 1))
            else:
                ids.append(parse_int_auto(item))
    return list(dict.fromkeys(ids))


def compact_payload(target: int, flags: int) -> list[int]:
    payload = (target & PRESSURE_MASK) | ((flags & FLAG_MASK) << 12)
    return [payload & 0xFF, (payload >> 8) & 0xFF]


def parse_compact(data: bytes | bytearray | list[int]) -> tuple[int, int] | None:
    if len(data) != 2:
        return None
    payload = int(data[0]) | (int(data[1]) << 8)
    return payload & PRESSURE_MASK, (payload >> 12) & FLAG_MASK


def ordered_ids(ids: list[int], order: str, cycle: int) -> list[int]:
    if order == "reverse":
        return list(reversed(ids))
    if order == "rotate" and ids:
        offset = cycle % len(ids)
        return ids[offset:] + ids[:offset]
    return list(ids)


def runtime_broadcast_frames(ids: list[int], target_payload: list[int]) -> list[can.Message]:
    selected = set(ids)
    unsupported = [board_id for board_id in ids if board_id < RUNTIME_FIRST_ID or board_id > RUNTIME_LAST_ID]
    if unsupported:
        pretty = ", ".join(f"0x{board_id:03X}" for board_id in unsupported)
        raise ValueError(f"Broadcast runtime mode only supports 0x101..0x118; unsupported: {pretty}")

    frames: list[can.Message] = []
    slot_count = RUNTIME_LAST_ID - RUNTIME_FIRST_ID + 1
    for start_slot in range(0, slot_count, RUNTIME_TABLE_SLOTS):
        data = [0] * 8
        data[0] = RUNTIME_TABLE_MARKER | (start_slot & RUNTIME_TABLE_SLOT_MASK)
        mask = 0
        for offset in range(RUNTIME_TABLE_SLOTS):
            board_id = RUNTIME_FIRST_ID + start_slot + offset
            if board_id not in selected:
                continue
            mask |= 1 << offset
            payload_index = 2 + offset * 2
            data[payload_index] = target_payload[0]
            data[payload_index + 1] = target_payload[1]
        if mask:
            data[1] = mask
            frames.append(can.Message(arbitration_id=RUNTIME_TABLE_BASE_ID + start_slot // RUNTIME_TABLE_SLOTS, data=data, is_extended_id=False))
    return frames


def sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


def wait_for_reply(bus: can.BusABC, expected_id: int, deadline: float, ids: set[int], late_bucket: dict[int, int]) -> Reply | None:
    while time.perf_counter() < deadline:
        msg = bus.recv(max(0.0, min(0.0005, deadline - time.perf_counter())))
        if msg is None or msg.is_extended_id:
            continue
        parsed = parse_compact(msg.data)
        if parsed is None:
            continue
        if msg.arbitration_id == expected_id:
            pressure, status = parsed
            return Reply(pressure=pressure, status=status, latency_ms=0.0)
        if msg.arbitration_id in ids:
            late_bucket[msg.arbitration_id] = late_bucket.get(msg.arbitration_id, 0) + 1
    return None


def drain_bus(bus: can.BusABC, seconds: float) -> int:
    deadline = time.perf_counter() + seconds
    drained = 0
    while time.perf_counter() < deadline:
        msg = bus.recv(max(0.0, min(0.001, deadline - time.perf_counter())))
        if msg is not None:
            drained += 1
    return drained


def run_safe_prelude(bus: can.BusABC, ids: list[int], target_payload: list[int], mode: str) -> int:
    if mode == "none":
        return drain_bus(bus, 0.05)
    if mode == "transaction":
        id_set = set(ids)
        late_bucket: dict[int, int] = {}
        for board_id in ids:
            send_time = time.perf_counter()
            bus.send(can.Message(arbitration_id=board_id, data=target_payload, is_extended_id=False))
            wait_for_reply(bus, board_id, send_time + 0.002, id_set, late_bucket)
        bus.send(can.Message(arbitration_id=BROADCAST_SYNC_ID, data=[], is_extended_id=False))
        return drain_bus(bus, 0.05)

    for board_id in ids:
        bus.send(can.Message(arbitration_id=board_id, data=target_payload, is_extended_id=False))
    bus.send(can.Message(arbitration_id=BROADCAST_SYNC_ID, data=[], is_extended_id=False))
    if mode == "double":
        for board_id in ids:
            bus.send(can.Message(arbitration_id=board_id, data=target_payload, is_extended_id=False))
    return drain_bus(bus, 0.05)


def run_test(args: argparse.Namespace) -> tuple[Path, Path]:
    ids = parse_ids(args.ids)
    if not ids:
        raise ValueError("No IDs selected")
    id_set = set(ids)
    period_s = 1.0 / args.rate
    inter_frame_s = args.inter_frame_us / 1_000_000.0
    target_payload = compact_payload(args.target, args.flags)
    broadcast_frames = runtime_broadcast_frames(ids, target_payload) if args.mode == "broadcast" else []

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.log_dir / f"can_dropout_{stamp}_{args.mode}_{args.order}_{args.rate}hz.csv"
    report_path = args.log_dir / f"can_dropout_{stamp}_{args.mode}_{args.order}_{args.rate}hz.md"

    stats = {board_id: BoardStats(board_id) for board_id in ids}
    bus = can.interface.Bus(
        interface="slcan",
        channel=args.port,
        bitrate=args.bitrate,
        tty_baudrate=args.tty_baudrate,
    )
    try:
        drain_bus(bus, 0.2)
        prelude_drained = run_safe_prelude(bus, ids, target_payload, args.prelude)

        total_cycles = max(1, math.ceil(args.duration * args.rate))
        late_bucket: dict[int, int] = {}
        start = time.perf_counter()
        next_cycle = start
        last_print = start

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cycle",
                    "time_s",
                    "mode",
                    "order",
                    "slot",
                    "id",
                    "responded",
                    "pressure",
                    "status",
                    "latency_ms",
                    "consecutive_misses",
                    "dropout",
                    "late_replies_seen",
                    "cycle_jitter_ms",
                ],
            )
            writer.writeheader()

            for cycle in range(total_cycles):
                cycle_start = time.perf_counter()
                jitter_ms = (cycle_start - next_cycle) * 1000.0
                order_ids = ordered_ids(ids, args.order, cycle)
                replies: dict[int, Reply] = {}

                bus.send(can.Message(arbitration_id=BROADCAST_SYNC_ID, data=[], is_extended_id=False))
                if inter_frame_s > 0:
                    time.sleep(inter_frame_s)

                if args.mode == "transaction":
                    per_board_budget = max(0.0004, (period_s * args.rx_window_frac) / max(1, len(order_ids)))
                    for board_id in order_ids:
                        send_time = time.perf_counter()
                        bus.send(can.Message(arbitration_id=board_id, data=target_payload, is_extended_id=False))
                        reply = wait_for_reply(bus, board_id, send_time + per_board_budget, id_set, late_bucket)
                        if reply is not None:
                            reply.latency_ms = (time.perf_counter() - send_time) * 1000.0
                            replies[board_id] = reply
                        if inter_frame_s > 0:
                            time.sleep(inter_frame_s)
                elif args.mode == "broadcast":
                    for message in broadcast_frames:
                        bus.send(message)
                        if inter_frame_s > 0:
                            time.sleep(inter_frame_s)

                    bus.send(can.Message(arbitration_id=BROADCAST_SYNC_ID, data=[], is_extended_id=False))
                    rx_deadline = cycle_start + period_s * args.rx_window_frac
                    while time.perf_counter() < rx_deadline:
                        msg = bus.recv(max(0.0, min(0.0005, rx_deadline - time.perf_counter())))
                        if msg is None or msg.is_extended_id or msg.arbitration_id not in id_set:
                            continue
                        parsed = parse_compact(msg.data)
                        if parsed is None:
                            continue
                        pressure, status = parsed
                        if msg.arbitration_id in replies:
                            replies[msg.arbitration_id].duplicate += 1
                            continue
                        replies[msg.arbitration_id] = Reply(
                            pressure=pressure,
                            status=status,
                            latency_ms=(time.perf_counter() - cycle_start) * 1000.0,
                        )
                else:
                    for board_id in order_ids:
                        bus.send(can.Message(arbitration_id=board_id, data=target_payload, is_extended_id=False))
                        if inter_frame_s > 0:
                            time.sleep(inter_frame_s)

                    rx_deadline = cycle_start + period_s * args.rx_window_frac
                    while time.perf_counter() < rx_deadline:
                        msg = bus.recv(max(0.0, min(0.0005, rx_deadline - time.perf_counter())))
                        if msg is None or msg.is_extended_id or msg.arbitration_id not in id_set:
                            continue
                        parsed = parse_compact(msg.data)
                        if parsed is None:
                            continue
                        pressure, status = parsed
                        if msg.arbitration_id in replies:
                            replies[msg.arbitration_id].duplicate += 1
                            continue
                        replies[msg.arbitration_id] = Reply(
                            pressure=pressure,
                            status=status,
                            latency_ms=(time.perf_counter() - cycle_start) * 1000.0,
                        )

                for slot, board_id in enumerate(order_ids):
                    reply = replies.get(board_id)
                    if reply is None:
                        stats[board_id].mark_miss(cycle, args.dropout_threshold)
                    else:
                        stats[board_id].mark_reply(cycle, reply)
                    writer.writerow(
                        {
                            "cycle": cycle,
                            "time_s": f"{cycle_start - start:.6f}",
                            "mode": args.mode,
                            "order": args.order,
                            "slot": slot,
                            "id": f"0x{board_id:03X}",
                            "responded": int(reply is not None),
                            "pressure": reply.pressure if reply else "",
                            "status": reply.status if reply else "",
                            "latency_ms": f"{reply.latency_ms:.3f}" if reply else "",
                            "consecutive_misses": stats[board_id].consecutive_misses,
                            "dropout": int(stats[board_id].in_dropout),
                            "late_replies_seen": late_bucket.get(board_id, 0),
                            "cycle_jitter_ms": f"{jitter_ms:.3f}",
                        }
                    )

                if time.perf_counter() - last_print >= 1.0:
                    last_print = time.perf_counter()
                    summary = " ".join(
                        f"0x{board_id:03X}:miss={stats[board_id].misses},consec={stats[board_id].consecutive_misses}"
                        for board_id in ids
                    )
                    print(f"cycle {cycle}/{total_cycles}: {summary}")

                next_cycle = start + (cycle + 1) * period_s
                sleep_until(next_cycle)

        with report_path.open("w") as report:
            report.write("# CAN Dropout Tester Report\n\n")
            report.write(f"- Port: {args.port}\n")
            report.write(f"- Mode: {args.mode}\n")
            report.write(f"- Order: {args.order}\n")
            report.write(f"- Rate Hz: {args.rate}\n")
            report.write(f"- Duration s: {args.duration}\n")
            report.write(f"- Target: {args.target}\n")
            report.write(f"- Flags: {args.flags}\n")
            report.write(f"- Inter-frame us: {args.inter_frame_us}\n")
            report.write(f"- Prelude: {args.prelude}\n")
            report.write(f"- Prelude replies drained: {prelude_drained}\n")
            report.write(f"- CSV: {csv_path}\n\n")
            report.write("| ID | Replies | Misses | Miss % | Dropouts | Recoveries | Max Consecutive Misses | Late Replies | Status Errors | Avg Latency ms | Max Latency ms |\n")
            report.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for board_id in ids:
                board = stats[board_id]
                expected = board.replies + board.misses
                miss_pct = (board.misses / expected * 100.0) if expected else 0.0
                report.write(
                    f"| 0x{board_id:03X} | {board.replies} | {board.misses} | {miss_pct:.2f} | "
                    f"{board.dropout_events} | {board.recoveries} | {board.max_consecutive_misses} | "
                    f"{late_bucket.get(board_id, 0)} | {board.status_errors} | "
                    f"{board.avg_latency_ms:.3f} | {board.latency_max_ms:.3f} |\n"
                )
            report.write("\n## Dropout Spans\n\n")
            for board_id in ids:
                spans = stats[board_id].dropout_spans
                pretty = ", ".join(f"{start}->{end if end is not None else 'open'}" for start, end in spans) if spans else "none"
                report.write(f"- 0x{board_id:03X}: {pretty}\n")

    finally:
        bus.shutdown()

    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    return csv_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce and characterize VNEMA compact CAN dropouts safely")
    parser.add_argument("--ids", nargs="+", default=["0x101-0x108"])
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--bitrate", type=int, default=DEFAULT_CAN_BITRATE)
    parser.add_argument("--tty-baudrate", type=int, default=DEFAULT_TTY_BAUDRATE)
    parser.add_argument("--rate", type=float, default=150.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--mode", choices=["burst", "transaction", "broadcast"], default="burst")
    parser.add_argument("--order", choices=["normal", "reverse", "rotate"], default="normal")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--flags", type=int, default=0)
    parser.add_argument("--inter-frame-us", type=int, default=0)
    parser.add_argument("--prelude", choices=["double", "single", "transaction", "none"], default="double")
    parser.add_argument("--rx-window-frac", type=float, default=0.85)
    parser.add_argument("--dropout-threshold", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=Path("hardware_comm_layer") / "reports")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())