from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import can


# --- workspace port resolution ---------------------------------------------
# Upstream hard-coded DEFAULT_PORT = "COM4". That port does not exist on this
# bench, and pinning any COM number repeats the same mistake, since Windows
# assigns the number per hub port and the dongle moves when it is replugged.
# bench_env at the workspace root resolves the adapter by its USB VID:PID and
# serial number instead, falling back to COM58.
WS_ROOT = Path(__file__).resolve().parents[2]
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))
try:
    import bench_env as _bench_env
except Exception:  # pragma: no cover - keep the tool usable without bench_env
    _bench_env = None

DEFAULT_PORT = _bench_env.resolve_can_port() if _bench_env is not None else "COM58"
DEFAULT_CAN_BITRATE = 1_000_000
DEFAULT_TTY_BAUDRATE = 2_000_000
# Upstream this was the VNEMA repo root, which held both calibration.json and
# hardware_comm_layer/reports/. In this workspace the legacy host bundle owns
# both, so PROJECT_ROOT points one level shallower than before.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BROADCAST_SYNC_ID = 0x090
RUNTIME_FIRST_ID = 0x101
RUNTIME_LAST_ID = 0x118
RUNTIME_TABLE_BASE_ID = 0x091
RUNTIME_TABLE_MARKER = 0x80
RUNTIME_TABLE_SLOT_MASK = 0x1F
RUNTIME_TABLE_SLOTS = 3
PRESSURE_MASK = 0x0FFF
FLAG_MASK = 0x000F
CONTROL_ENABLE = 0x01
STATUS_ERROR = 0x08


@dataclass
class Reply:
    pressure: int
    status: int
    latency_ms: float


@dataclass
class CycleSummary:
    cycles: int = 0
    replies: dict[int, int] = field(default_factory=dict)
    misses: dict[int, int] = field(default_factory=dict)
    status_errors: dict[int, int] = field(default_factory=dict)

    def record(self, ids: list[int], replies: dict[int, Reply]) -> None:
        self.cycles += 1
        for board_id in ids:
            if board_id in replies:
                self.replies[board_id] = self.replies.get(board_id, 0) + 1
                if replies[board_id].status & STATUS_ERROR:
                    self.status_errors[board_id] = self.status_errors.get(board_id, 0) + 1
            else:
                self.misses[board_id] = self.misses.get(board_id, 0) + 1


@dataclass
class SampleStats:
    values: list[int] = field(default_factory=list)
    status_error_count: int = 0
    miss_count: int = 0
    cycle_count: int = 0

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else math.nan

    @property
    def stddev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def minimum(self) -> int | None:
        return min(self.values) if self.values else None

    @property
    def maximum(self) -> int | None:
        return max(self.values) if self.values else None

    def as_json(self) -> dict[str, Any]:
        return {
            "samples": len(self.values),
            "cycles": self.cycle_count,
            "misses": self.miss_count,
            "status_errors": self.status_error_count,
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.minimum,
            "max": self.maximum,
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
            lower = item.lower()
            if lower == "dt":
                ids.extend(range(0x101, 0x109))
            elif lower in {"all", "all24"}:
                ids.extend(range(RUNTIME_FIRST_ID, RUNTIME_LAST_ID + 1))
            elif lower in {"all23", "no10c"}:
                ids.extend(board_id for board_id in range(RUNTIME_FIRST_ID, RUNTIME_LAST_ID + 1) if board_id != 0x10C)
            elif "-" in item:
                first_raw, last_raw = item.split("-", 1)
                first = parse_int_auto(first_raw)
                last = parse_int_auto(last_raw)
                step = 1 if last >= first else -1
                ids.extend(range(first, last + step, step))
            else:
                ids.append(parse_int_auto(item))
    return list(dict.fromkeys(ids))


def id_hex(board_id: int) -> str:
    return f"0x{board_id:03X}"


def compact_payload(target: int, flags: int) -> list[int]:
    payload = (int(target) & PRESSURE_MASK) | ((int(flags) & FLAG_MASK) << 12)
    return [payload & 0xFF, (payload >> 8) & 0xFF]


def parse_compact(data: bytes | bytearray | list[int]) -> tuple[int, int] | None:
    if len(data) != 2:
        return None
    payload = int(data[0]) | (int(data[1]) << 8)
    return payload & PRESSURE_MASK, (payload >> 12) & FLAG_MASK


def clamp_adc(value: float | int) -> int:
    return max(0, min(PRESSURE_MASK, int(round(float(value)))))


def runtime_broadcast_frames(targets: dict[int, int], flags: dict[int, int]) -> list[can.Message]:
    unsupported = [board_id for board_id in targets if board_id < RUNTIME_FIRST_ID or board_id > RUNTIME_LAST_ID]
    if unsupported:
        pretty = ", ".join(id_hex(board_id) for board_id in unsupported)
        raise ValueError(f"runtime table mode supports 0x101..0x118 only; unsupported: {pretty}")

    frames: list[can.Message] = []
    slot_count = RUNTIME_LAST_ID - RUNTIME_FIRST_ID + 1
    for start_slot in range(0, slot_count, RUNTIME_TABLE_SLOTS):
        data = [0] * 8
        data[0] = RUNTIME_TABLE_MARKER | (start_slot & RUNTIME_TABLE_SLOT_MASK)
        mask = 0
        for offset in range(RUNTIME_TABLE_SLOTS):
            board_id = RUNTIME_FIRST_ID + start_slot + offset
            if board_id not in targets:
                continue
            mask |= 1 << offset
            payload = compact_payload(targets[board_id], flags.get(board_id, 0))
            payload_index = 2 + offset * 2
            data[payload_index] = payload[0]
            data[payload_index + 1] = payload[1]
        if mask:
            data[1] = mask
            frames.append(can.Message(arbitration_id=RUNTIME_TABLE_BASE_ID + start_slot // RUNTIME_TABLE_SLOTS, data=data, is_extended_id=False))
    return frames


def sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


def drain_bus(bus: can.BusABC, seconds: float) -> int:
    deadline = time.perf_counter() + seconds
    drained = 0
    while time.perf_counter() < deadline:
        msg = bus.recv(max(0.0, min(0.001, deadline - time.perf_counter())))
        if msg is not None:
            drained += 1
    return drained


def collect_replies(bus: can.BusABC, ids: set[int], deadline: float, cycle_start: float) -> dict[int, Reply]:
    replies: dict[int, Reply] = {}
    while time.perf_counter() < deadline and len(replies) < len(ids):
        msg = bus.recv(max(0.0, min(0.0005, deadline - time.perf_counter())))
        if msg is None or msg.is_extended_id or msg.arbitration_id not in ids:
            continue
        parsed = parse_compact(msg.data)
        if parsed is None or msg.arbitration_id in replies:
            continue
        pressure, status = parsed
        replies[msg.arbitration_id] = Reply(
            pressure=pressure,
            status=status,
            latency_ms=(time.perf_counter() - cycle_start) * 1000.0,
        )
    return replies


class RuntimeTableDriver:
    def __init__(self, bus: can.BusABC, ids: list[int], rate_hz: float, rx_window_frac: float) -> None:
        self.bus = bus
        self.ids = ids
        self.id_set = set(ids)
        self.period_s = 1.0 / max(1.0, rate_hz)
        self.rx_window_frac = max(0.05, min(0.98, rx_window_frac))
        self.start_time = time.perf_counter()
        self.cycle = 0

    def run_cycle(self, targets: dict[int, int], flags: dict[int, int]) -> dict[int, Reply]:
        cycle_start = time.perf_counter()
        for frame in runtime_broadcast_frames(targets, flags):
            self.bus.send(frame)
        self.bus.send(can.Message(arbitration_id=BROADCAST_SYNC_ID, data=[], is_extended_id=False))
        replies = collect_replies(
            self.bus,
            self.id_set,
            cycle_start + self.period_s * self.rx_window_frac,
            cycle_start,
        )
        self.cycle += 1
        sleep_until(cycle_start + self.period_s)
        return replies

    def run_phase(
        self,
        duration_s: float,
        targets: dict[int, int],
        flags: dict[int, int],
        sample_board_id: int | None = None,
        sample_writer: Any | None = None,
        phase: str = "hold",
    ) -> tuple[CycleSummary, SampleStats | None]:
        total_cycles = max(1, int(math.ceil(duration_s / self.period_s)))
        summary = CycleSummary()
        sample_stats = SampleStats() if sample_board_id is not None else None
        for _ in range(total_cycles):
            cycle_time_s = time.time()
            replies = self.run_cycle(targets, flags)
            summary.record(self.ids, replies)
            if sample_stats is not None and sample_board_id is not None:
                sample_stats.cycle_count += 1
                reply = replies.get(sample_board_id)
                if reply is None:
                    sample_stats.miss_count += 1
                else:
                    sample_stats.values.append(reply.pressure)
                    if reply.status & STATUS_ERROR:
                        sample_stats.status_error_count += 1
                    if sample_writer is not None:
                        sample_writer.write(
                            json.dumps(
                                {
                                    "phase": phase,
                                    "cycle": self.cycle,
                                    "timestamp_unix_s": cycle_time_s,
                                    "id": id_hex(sample_board_id),
                                    "target": targets[sample_board_id],
                                    "pressure_adc_filtered": reply.pressure,
                                    "status": reply.status,
                                    "latency_ms": reply.latency_ms,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
        return summary, sample_stats


def summarize_cycle_summary(summary: CycleSummary, ids: list[int]) -> dict[str, Any]:
    return {
        "cycles": summary.cycles,
        "replies": {id_hex(board_id): summary.replies.get(board_id, 0) for board_id in ids},
        "misses": {id_hex(board_id): summary.misses.get(board_id, 0) for board_id in ids},
        "status_errors": {id_hex(board_id): summary.status_errors.get(board_id, 0) for board_id in ids},
    }


def validate_sample_stats(label: str, board_id: int, stats: SampleStats, expected_cycles: int, min_sample_frac: float, allow_status_errors: bool) -> None:
    min_samples = max(1, int(math.floor(expected_cycles * min_sample_frac)))
    if len(stats.values) < min_samples:
        raise RuntimeError(
            f"{id_hex(board_id)} {label}: only {len(stats.values)} samples, need at least {min_samples} of {expected_cycles} cycles"
        )
    if stats.status_error_count and not allow_status_errors:
        raise RuntimeError(f"{id_hex(board_id)} {label}: saw {stats.status_error_count} status-error replies")
    if not math.isfinite(stats.mean):
        raise RuntimeError(f"{id_hex(board_id)} {label}: non-finite sample mean")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def write_summary_csv(path: Path, actuator_results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "adc_0psi_mean",
                "adc_0psi_stddev",
                "adc_0psi_samples",
                "adc_25psi_mean",
                "adc_25psi_stddev",
                "adc_25psi_samples",
                "adc_40psi_estimated",
                "adc_0psi_stored",
                "adc_40psi_stored",
            ],
        )
        writer.writeheader()
        for actuator_id, result in actuator_results.items():
            writer.writerow(
                {
                    "id": actuator_id,
                    "adc_0psi_mean": f"{result['adc_0psi']:.6f}",
                    "adc_0psi_stddev": f"{result['zero']['stddev']:.6f}",
                    "adc_0psi_samples": result["zero"]["samples"],
                    "adc_25psi_mean": f"{result['adc_25psi']:.6f}",
                    "adc_25psi_stddev": f"{result['source']['stddev']:.6f}",
                    "adc_25psi_samples": result["source"]["samples"],
                    "adc_40psi_estimated": f"{result['adc_40psi_estimated']:.6f}",
                    "adc_0psi_stored": result["adc_range"][0],
                    "adc_40psi_stored": result["adc_range"][1],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate VNEMA pressure ADC readings from 0 psi and 25 psi marks")
    parser.add_argument("--ids", nargs="+", default=["0x101-0x118"], help="Board IDs, ranges, or aliases dt/all/all24/all23/no10c")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--bitrate", type=int, default=DEFAULT_CAN_BITRATE)
    parser.add_argument("--tty-baudrate", type=int, default=DEFAULT_TTY_BAUDRATE)
    parser.add_argument("--rate", type=float, default=150.0)
    parser.add_argument("--rx-window-frac", type=float, default=0.85)
    parser.add_argument("--nominal-target", type=int, default=800)
    parser.add_argument("--deflate-target", type=int, default=0)
    parser.add_argument("--inflate-target", type=int, default=4095)
    parser.add_argument("--source-pressure-psi", type=float, default=25.0)
    parser.add_argument("--range-pressure-psi", type=float, default=40.0)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--sample-s", type=float, default=2.0)
    parser.add_argument("--between-actuators-s", type=float, default=0.5)
    parser.add_argument("--initial-nominal-s", type=float, default=5.0)
    parser.add_argument("--min-sample-frac", type=float, default=0.85)
    parser.add_argument("--min-span-adc", type=float, default=50.0)
    parser.add_argument("--allow-status-errors", action="store_true")
    parser.add_argument("--leave-enabled", action="store_true", help="Leave outputs enabled at nominal target after calibration")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "calibration.json")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned sequence without opening CAN")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ids = parse_ids(args.ids)
    if not ids:
        raise ValueError("no actuator IDs selected")
    unsupported = [board_id for board_id in ids if board_id < RUNTIME_FIRST_ID or board_id > RUNTIME_LAST_ID]
    if unsupported:
        raise ValueError("selected IDs must be in 0x101..0x118 for runtime-table calibration")

    ids = sorted(ids)
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)
    log_dir = args.log_dir if args.log_dir.is_absolute() else (PROJECT_ROOT / args.log_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sample_log_path = log_dir / f"pressure_calibration_samples_{stamp}.jsonl"
    summary_csv_path = log_dir / f"pressure_calibration_summary_{stamp}.csv"

    sequence = {
        "nominal_target_adc": clamp_adc(args.nominal_target),
        "deflate_target_adc": clamp_adc(args.deflate_target),
        "inflate_target_adc": clamp_adc(args.inflate_target),
        "source_pressure_psi": float(args.source_pressure_psi),
        "range_pressure_psi": float(args.range_pressure_psi),
        "settle_s": float(args.settle_s),
        "sample_s": float(args.sample_s),
        "initial_nominal_s": float(args.initial_nominal_s),
        "between_actuators_s": float(args.between_actuators_s),
    }
    print("Selected IDs:", ", ".join(id_hex(board_id) for board_id in ids))
    print(f"Output: {output_path}")
    print(f"Sample log: {sample_log_path}")
    if args.dry_run:
        print(json.dumps({"ids": [id_hex(board_id) for board_id in ids], "sequence": sequence}, indent=2))
        return 0

    bus: can.BusABC | None = None
    driver: RuntimeTableDriver | None = None
    actuator_results: dict[str, dict[str, Any]] = {}
    phase_summaries: dict[str, Any] = {}
    targets = {board_id: sequence["nominal_target_adc"] for board_id in ids}
    flags_enabled = {board_id: CONTROL_ENABLE for board_id in ids}
    flags_disabled = {board_id: 0 for board_id in ids}

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        bus = can.interface.Bus(
            interface="slcan",
            channel=args.port,
            bitrate=args.bitrate,
            tty_baudrate=args.tty_baudrate,
        )
        driver = RuntimeTableDriver(bus, ids, args.rate, args.rx_window_frac)
        drained = drain_bus(bus, 0.2)
        print(f"Drained {drained} stale CAN frames")

        print(f"Holding all actuators at nominal {sequence['nominal_target_adc']} ADC")
        initial_summary, _ = driver.run_phase(args.initial_nominal_s, targets, flags_enabled, phase="initial_nominal")
        phase_summaries["initial_nominal"] = summarize_cycle_summary(initial_summary, ids)
        no_reply = [board_id for board_id in ids if initial_summary.replies.get(board_id, 0) == 0]
        if no_reply:
            pretty = ", ".join(id_hex(board_id) for board_id in no_reply)
            raise RuntimeError(f"no replies during initial nominal hold from: {pretty}")

        expected_sample_cycles = max(1, int(math.ceil(args.sample_s * max(1.0, args.rate))))
        with sample_log_path.open("w", encoding="utf-8") as sample_writer:
            for index, board_id in enumerate(ids, start=1):
                actuator_label = id_hex(board_id)
                print(f"[{index}/{len(ids)}] {actuator_label}: deflate to {sequence['deflate_target_adc']} ADC")
                targets[board_id] = sequence["deflate_target_adc"]
                settle_summary, _ = driver.run_phase(args.settle_s, targets, flags_enabled, phase="deflate_settle")
                phase_summaries[f"{actuator_label}_deflate_settle"] = summarize_cycle_summary(settle_summary, ids)
                _, zero_stats = driver.run_phase(
                    args.sample_s,
                    targets,
                    flags_enabled,
                    sample_board_id=board_id,
                    sample_writer=sample_writer,
                    phase="zero_sample",
                )
                assert zero_stats is not None
                validate_sample_stats("0 psi", board_id, zero_stats, expected_sample_cycles, args.min_sample_frac, args.allow_status_errors)

                print(f"[{index}/{len(ids)}] {actuator_label}: inflate to {sequence['inflate_target_adc']} ADC")
                targets[board_id] = sequence["inflate_target_adc"]
                settle_summary, _ = driver.run_phase(args.settle_s, targets, flags_enabled, phase="inflate_settle")
                phase_summaries[f"{actuator_label}_inflate_settle"] = summarize_cycle_summary(settle_summary, ids)
                _, source_stats = driver.run_phase(
                    args.sample_s,
                    targets,
                    flags_enabled,
                    sample_board_id=board_id,
                    sample_writer=sample_writer,
                    phase="source_sample",
                )
                assert source_stats is not None
                validate_sample_stats("source pressure", board_id, source_stats, expected_sample_cycles, args.min_sample_frac, args.allow_status_errors)

                adc_0psi = zero_stats.mean
                adc_25psi = source_stats.mean
                span_25psi = adc_25psi - adc_0psi
                if span_25psi < args.min_span_adc:
                    raise RuntimeError(f"{actuator_label}: source-pressure span too small ({span_25psi:.3f} ADC counts)")
                adc_40psi = adc_0psi + span_25psi * (args.range_pressure_psi / args.source_pressure_psi)
                stored_zero = clamp_adc(adc_0psi)
                stored_full = clamp_adc(adc_40psi)
                if stored_full <= stored_zero:
                    raise RuntimeError(f"{actuator_label}: stored 40 psi endpoint is not above zero endpoint")

                actuator_results[actuator_label] = {
                    "adc_0psi": adc_0psi,
                    "adc_25psi": adc_25psi,
                    "adc_40psi_estimated": adc_40psi,
                    "adc_range": [stored_zero, stored_full],
                    "zero": zero_stats.as_json(),
                    "source": source_stats.as_json(),
                    "source_pressure_psi": args.source_pressure_psi,
                    "range_pressure_psi": args.range_pressure_psi,
                }
                print(
                    f"[{index}/{len(ids)}] {actuator_label}: adc0={adc_0psi:.2f}, "
                    f"adc25={adc_25psi:.2f}, adc40_est={adc_40psi:.2f}, stored=[{stored_zero},{stored_full}]"
                )

                targets[board_id] = sequence["nominal_target_adc"]
                if args.between_actuators_s > 0.0:
                    driver.run_phase(args.between_actuators_s, targets, flags_enabled, phase="return_nominal")

        adc_ranges = {actuator_id: result["adc_range"] for actuator_id, result in actuator_results.items()}
        payload = {
            "schema_version": 1,
            "created_at_unix_s": time.time(),
            "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "VNEMA pressure calibration from filtered compact CAN pressure replies",
            "ids": [id_hex(board_id) for board_id in ids],
            "can": {
                "port": args.port,
                "bitrate": args.bitrate,
                "tty_baudrate": args.tty_baudrate,
                "rate_hz": args.rate,
                "rx_window_frac": args.rx_window_frac,
                "protocol": "runtime_table_broadcast",
            },
            "sequence": sequence,
            "adc_ranges": adc_ranges,
            "calibration_points_psi": [0.0, float(args.source_pressure_psi)],
            "stored_range_psi": [0.0, float(args.range_pressure_psi)],
            "extrapolation": "adc_40psi = adc_0psi + (adc_25psi - adc_0psi) * 40 / 25",
            "actuators": actuator_results,
            "phase_summaries": phase_summaries,
            "logs": {
                "samples_jsonl": str(sample_log_path),
                "summary_csv": str(summary_csv_path),
            },
        }
        write_summary_csv(summary_csv_path, actuator_results)
        atomic_write_json(output_path, payload)
        print(f"Wrote {output_path}")
        print(f"Wrote {summary_csv_path}")
        return 0
    finally:
        if driver is not None:
            try:
                print("Cleanup: returning selected actuators to nominal target")
                driver.run_phase(1.0, targets, flags_enabled, phase="cleanup_nominal")
                if not args.leave_enabled:
                    print("Cleanup: disabling selected actuators")
                    driver.run_phase(0.5, targets, flags_disabled, phase="cleanup_disabled")
            except Exception as exc:
                print(f"Cleanup warning: {exc}")
        if bus is not None:
            bus.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())