from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ARM_IDS = list(range(1000, 1006))

# Upstream this pointed at a 9.4 MB vendored copy of the full NatNet SDK 4.4 for
# Windows sitting beside this script, complete with committed x64/Release build
# residue. The workspace keeps one vendored NatNet Python client instead, under
# UMArm_MOCAP/natnet_sdk, so that both mocap stacks import the same code.
WS_ROOT = Path(__file__).resolve().parents[2]
SDK_PYTHON_CLIENT = WS_ROOT / "UMArm_MOCAP" / "natnet_sdk"


@dataclass(frozen=True)
class MocapSample:
    frame: int
    timestamp_s: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


@dataclass(frozen=True)
class BodyPoint:
    body_id: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BodyPose:
    body_id: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def point(self) -> BodyPoint:
        return BodyPoint(self.body_id, self.x, self.y, self.z)


def parse_int_auto(raw: str) -> int:
    return int(raw.strip(), 0)


def parse_ids(raw: str | Iterable[str]) -> list[int]:
    tokens = [raw] if isinstance(raw, str) else list(raw)
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
                if first > last:
                    raise ValueError(f"ID range is reversed: {item}")
                ids.extend(range(first, last + 1))
            else:
                ids.append(parse_int_auto(item))
    return list(dict.fromkeys(ids))


def simulated_sample(frame: int, timestamp_s: float) -> MocapSample:
    omega_xy = 2.0 * math.pi * 0.25
    omega_z = 2.0 * math.pi * 0.10
    return MocapSample(
        frame=frame,
        timestamp_s=timestamp_s,
        x=math.sin(omega_xy * timestamp_s),
        y=math.cos(omega_xy * timestamp_s),
        z=0.25 * math.sin(omega_z * timestamp_s),
        vx=omega_xy * math.cos(omega_xy * timestamp_s),
        vy=-omega_xy * math.sin(omega_xy * timestamp_s),
        vz=0.25 * omega_z * math.cos(omega_z * timestamp_s),
    )


def simulated_body_points(sample: MocapSample, rigid_ids: list[int]) -> list[BodyPoint]:
    return [pose.point() for pose in simulated_body_poses(sample, rigid_ids)]


def simulated_body_poses(sample: MocapSample, rigid_ids: list[int]) -> list[BodyPose]:
    if not rigid_ids:
        return []
    spacing = 0.055
    center_index = (len(rigid_ids) - 1) / 2.0
    return [
        BodyPose(
            body_id=body_id,
            x=sample.x + (index - center_index) * spacing,
            y=sample.y + 0.030 * math.sin(index * 1.7 + sample.timestamp_s),
            z=sample.z + 0.025 * math.cos(index * 1.3 + sample.timestamp_s),
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        )
        for index, body_id in enumerate(rigid_ids)
    ]


def format_body_points(points: Iterable[BodyPoint]) -> str:
    return "|".join(f"{point.body_id}:{point.x:.6f}:{point.y:.6f}:{point.z:.6f}" for point in points)


def format_body_poses(poses: Iterable[BodyPose]) -> str:
    return "|".join(
        f"{pose.body_id}:{pose.x:.6f}:{pose.y:.6f}:{pose.z:.6f}:{pose.qx:.9f}:{pose.qy:.9f}:{pose.qz:.9f}:{pose.qw:.9f}"
        for pose in poses
    )


def json_record(
    sample: MocapSample,
    rigid_ids: list[int],
    body_count: int | None = None,
    body_points: Iterable[BodyPoint] | None = None,
    body_poses: Iterable[BodyPose] | None = None,
    received_s: float | None = None,
    latency_ms: float | None = None,
) -> dict[str, object]:
    received_s = time.perf_counter() if received_s is None else received_s
    visible_count = len(rigid_ids) if body_count is None else body_count
    poses = list(body_poses) if body_poses is not None else simulated_body_poses(sample, rigid_ids)
    points = list(body_points) if body_points is not None else [pose.point() for pose in poses]
    if visible_count == 0:
        points = []
        poses = []
    if latency_ms is None:
        latency_ms = (received_s - sample.timestamp_s) * 1000.0
    return {
        "type": "mocap",
        "valid": visible_count != 0,
        "stale": False,
        "frame": sample.frame,
        "timestamp_s": sample.timestamp_s,
        "received_s": received_s,
        "latency_ms": latency_ms,
        "body_count": visible_count,
        "body_ids": ",".join(str(item) for item in rigid_ids),
        "body_points": format_body_points(points),
        "body_poses": format_body_poses(poses),
        "x": sample.x,
        "y": sample.y,
        "z": sample.z,
        "vx": sample.vx,
        "vy": sample.vy,
        "vz": sample.vz,
    }


def emit_json(record: dict[str, object]) -> None:
    print(json.dumps(record, separators=(",", ":")), flush=True)


def run_simulator(seconds: float, rate_hz: float, rigid_ids: list[int], emit_as_json: bool) -> None:
    period = 1.0 / rate_hz
    deadline = time.perf_counter() + seconds
    frame = 0
    next_tick = time.perf_counter()
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        sample = simulated_sample(frame, now)
        if emit_as_json:
            emit_json(json_record(sample, rigid_ids))
        else:
            print(
                f"frame={sample.frame} t={sample.timestamp_s:.6f} ids={','.join(str(item) for item in rigid_ids)} "
                f"pos=({sample.x:.4f},{sample.y:.4f},{sample.z:.4f}) "
                f"vel=({sample.vx:.4f},{sample.vy:.4f},{sample.vz:.4f})"
            )
        frame += 1
        next_tick += period
        time.sleep(max(0.0, next_tick - time.perf_counter()))


class ArmRigidBodyTracker:
    def __init__(self, rigid_ids: list[int], emit_as_json: bool) -> None:
        self.rigid_ids = rigid_ids
        self.rigid_id_set = set(rigid_ids)
        self.emit_as_json = emit_as_json
        self.previous_center: tuple[float, float, float] | None = None
        self.previous_time_s: float | None = None
        self.last_emit_time_s = 0.0
        self.timestamp_offset_s: float | None = None
        self.frame_poses: dict[int, BodyPose] = {}

    def timestamp_latency_ms(self, timestamp_s: float, received_s: float) -> float:
        offset_s = received_s - timestamp_s
        if self.timestamp_offset_s is None:
            self.timestamp_offset_s = offset_s
        latency_ms = (offset_s - self.timestamp_offset_s) * 1000.0
        if latency_ms < -100.0 or latency_ms > 10000.0:
            self.timestamp_offset_s = offset_s
            latency_ms = 0.0
        return max(0.0, latency_ms)

    def receive_rigid_body(self, body_id: int, position: tuple[float, float, float], rotation: tuple[float, float, float, float]) -> None:
        body_id = int(body_id)
        if body_id not in self.rigid_id_set:
            return
        if len(position) < 3 or len(rotation) < 4:
            return
        self.frame_poses[body_id] = BodyPose(
            body_id,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(rotation[0]),
            float(rotation[1]),
            float(rotation[2]),
            float(rotation[3]),
        )

    def receive_frame(self, data_dict: dict[str, object]) -> None:
        matched_poses = [self.frame_poses[body_id] for body_id in sorted(self.frame_poses)]
        self.frame_poses = {}
        matched = [pose.point() for pose in matched_poses]

        frame = int(data_dict.get("frame_number", 0) or 0)
        timestamp_s = float(data_dict.get("timestamp", 0.0) or 0.0)
        if timestamp_s <= 1.0:
            timestamp_s = time.perf_counter()
        received_s = time.perf_counter()
        latency_ms = self.timestamp_latency_ms(timestamp_s, received_s)

        if not matched:
            now = time.perf_counter()
            if now - self.last_emit_time_s > 1.0:
                self.last_emit_time_s = now
                sample = MocapSample(frame, timestamp_s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                record = json_record(sample, self.rigid_ids, body_count=0, received_s=received_s, latency_ms=latency_ms)
                record["stale"] = True
                if self.emit_as_json:
                    emit_json(record)
                else:
                    print(f"frame={frame} no tracked arm rigid bodies from {self.rigid_ids}", flush=True)
            return

        matched.sort(key=lambda item: item.body_id)
        x = sum(item.x for item in matched) / len(matched)
        y = sum(item.y for item in matched) / len(matched)
        z = sum(item.z for item in matched) / len(matched)
        vx = vy = vz = 0.0
        if self.previous_center is not None and self.previous_time_s is not None:
            dt = max(1e-6, timestamp_s - self.previous_time_s)
            vx = (x - self.previous_center[0]) / dt
            vy = (y - self.previous_center[1]) / dt
            vz = (z - self.previous_center[2]) / dt
        self.previous_center = (x, y, z)
        self.previous_time_s = timestamp_s

        sample = MocapSample(frame, timestamp_s, x, y, z, vx, vy, vz)
        visible_ids = [item.body_id for item in matched]
        if self.emit_as_json:
            emit_json(
                json_record(
                    sample,
                    visible_ids,
                    body_count=len(visible_ids),
                    body_points=matched,
                    body_poses=matched_poses,
                    received_s=received_s,
                    latency_ms=latency_ms,
                )
            )
        else:
            print(
                f"frame={frame} tracked={','.join(str(item) for item in visible_ids)} "
                f"centroid=({x:.4f},{y:.4f},{z:.4f}) vel=({vx:.4f},{vy:.4f},{vz:.4f})",
                flush=True,
            )


def run_live_natnet(args: argparse.Namespace, rigid_ids: list[int]) -> int:
    if not (SDK_PYTHON_CLIENT / "NatNetClient.py").exists():
        raise RuntimeError(
            "NatNet Python client not found at "
            f"{SDK_PYTHON_CLIENT / 'NatNetClient.py'}.\n"
            "The workspace expects the vendored NatNet client (NatNetClient.py, "
            "MoCapData.py, DataDescriptions.py) in UMArm_MOCAP/natnet_sdk. "
            "Copy it there, or run this script with --simulate, which needs no "
            "SDK and no cameras."
        )
    sys.path.insert(0, str(SDK_PYTHON_CLIENT))
    from NatNetClient import NatNetClient  # type: ignore

    tracker = ArmRigidBodyTracker(rigid_ids, args.json)
    client = NatNetClient()
    client.set_client_address(args.local)
    client.set_server_address(args.server)
    client.set_use_multicast(args.multicast)
    client.set_print_level(0)
    client.rigid_body_listener = tracker.receive_rigid_body
    client.new_frame_listener = tracker.receive_frame

    if not client.run("d"):
        raise RuntimeError("NatNet client failed to start")

    status = {
        "type": "mocap_status",
        "state": "running",
        "server": args.server,
        "local": args.local,
        "multicast": args.multicast,
        "rigid_ids": ",".join(str(item) for item in rigid_ids),
    }
    if args.json:
        emit_json(status)
    else:
        print(status)

    try:
        deadline = None if args.seconds <= 0 else time.perf_counter() + args.seconds
        while deadline is None or time.perf_counter() < deadline:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        client.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VNEMA mocap smoke helper and NatNet arm bridge")
    parser.add_argument("--simulate", action="store_true", help="Emit deterministic simulated mocap samples")
    parser.add_argument("--live", action="store_true", help="Connect to Motive/NatNet and stream arm rigid-body samples")
    parser.add_argument("--json", action="store_true", help="Emit JSON-lines for the C++ backend")
    parser.add_argument("--seconds", type=float, default=2.0, help="Run duration; use 0 for live until interrupted")
    parser.add_argument("--rate", type=float, default=360.0)
    parser.add_argument("--rigid-ids", nargs="+", default=["1000-1005"], help="Arm rigid body IDs, e.g. 1000-1005")
    parser.add_argument("--server", default="127.0.0.1", help="Motive/NatNet server IP")
    parser.add_argument("--local", default="127.0.0.1", help="Local interface IP for NatNet")
    cast = parser.add_mutually_exclusive_group()
    cast.add_argument("--multicast", dest="multicast", action="store_true", default=True)
    cast.add_argument("--unicast", dest="multicast", action="store_false")
    args = parser.parse_args()

    rigid_ids = parse_ids(args.rigid_ids)
    if args.simulate:
        run_simulator(args.seconds, args.rate, rigid_ids, args.json)
        return 0
    if args.live:
        return run_live_natnet(args, rigid_ids)

    print("Use --live for NatNet arm tracking or --simulate for local validation. Default arm rigid body IDs are 1000-1005.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
