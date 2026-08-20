"""Headless bench checks for the TLE CAN work.

Runs against the real board: USB on COM8, CAN dongle on COM58. Each stage is
independent so a failure is localised rather than cascading.

    python TLE_PCB/tools/tle_bench.py usb
    python TLE_PCB/tools/tle_bench.py setid --to 0x101
    python TLE_PCB/tools/tle_bench.py scan
    python TLE_PCB/tools/tle_bench.py sync --seconds 10
    python TLE_PCB/tools/tle_bench.py latch
    python TLE_PCB/tools/tle_bench.py failsafe
    python TLE_PCB/tools/tle_bench.py ota --image build/VEMA_MAX22200.bin
    python TLE_PCB/tools/tle_bench.py all
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tlelib import native as N  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from tlelib.canlink import DEFAULT_PORT as _CAN_DEFAULT_PORT  # noqa: E402
from tlelib.canlink import CanLink  # noqa: E402
from tlelib.ota import BroadcastOta, load_image  # noqa: E402
from tlelib.usblink import DEFAULT_PORT as _USB_DEFAULT_PORT  # noqa: E402
from tlelib.usblink import UsbLink  # noqa: E402

# The bench defaults, taken from tlelib so that a workspace-level bench_env.py
# is the single place a re-enumerated port is edited. COM8 is the board's own
# USB-Serial/JTAG port, COM58 the CANable dongle; --usb and --can override.
USB_PORT = _USB_DEFAULT_PORT
CAN_PORT = _CAN_DEFAULT_PORT

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    verdict = PASS if ok else FAIL
    _results.append((verdict, name, detail))
    print(f"  [{verdict}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
def stage_usb(args) -> None:
    banner("USB control link (COM8)")
    with UsbLink(args.usb) as link:
        time.sleep(0.5)
        link.keepalive()
        time.sleep(0.3)
        state = link.read_state(1.0)
        check("telemetry over USB", bool(state),
              f"state={N.PRESSURE_STATES.get(state.get('state'), '?')} "
              f"raw={state.get('pressure_raw')} "
              f"({N.raw_to_psi(state.get('pressure_raw', 0)):.2f} psi)" if state else "no frames")
        check("board reports a base ID", link.base is not None,
              f"base=0x{link.base:03X}" if link.base else "")

        # The boot banner is only in the log if the board rebooted recently;
        # ask for it explicitly instead of relying on timing.
        idmap = link.wait_log("CAN IDs:", 0.5)
        if idmap:
            print(f"       {idmap.strip()}")
        slot = link.wait_log("Shared-bus slot", 0.5)
        if slot:
            print(f"       {slot.strip()}")


def stage_setid(args) -> None:
    banner(f"set board ID over USB -> 0x{args.to:03X}")
    with UsbLink(args.usb) as link:
        time.sleep(0.5)
        link.keepalive()
        time.sleep(0.3)
        before = link.base
        print(f"  current base: 0x{before:03X}" if before else "  current base unknown")
        if before == args.to:
            check("board already has the wanted ID", True, f"0x{args.to:03X}")
            return
        link.set_board_id(args.to, before)
        time.sleep(0.5)
        link.keepalive()
        time.sleep(0.5)
        state = link.read_state(1.0)
        check("board answers on the new ID", link.base == args.to,
              f"base=0x{link.base:03X}" if link.base else "no telemetry after reboot")
        check("controller still alive after the ID change", bool(state))


def stage_scan(args) -> None:
    banner(f"CAN node scan ({args.can})")
    with CanLink(args.can, args.bitrate) as link:
        time.sleep(0.2)
        found = link.scan()
        for base, version in found.items():
            print(f"  0x{base:03X}  {version.variant_name:8s}  fw {version.version}")
        check("at least one node answered", bool(found), f"{len(found)} node(s)")
        for base, version in found.items():
            if P.is_tle_slot(base) or version.variant == P.VARIANT_TLE_DVP:
                check(f"0x{base:03X} identifies as a TLE board",
                      version.variant == P.VARIANT_TLE_DVP, version.variant_name)

        if found:
            base = next(iter(found))
            diag = link.can_diag(base)
            check(f"0x{base:03X} answers CAN diagnostics", diag is not None,
                  f"sync={diag.sync_counter} cmd={diag.command_counter} "
                  f"rx_ovf={diag.rx_overflow_count} tx_fail={diag.tx_fail_count}" if diag else "")
            status = link.ota_status(base)
            check(f"0x{base:03X} answers OTA status", status is not None,
                  f"active={status.active} flags=0x{status.flags:02X}" if status else "")


def stage_sync(args) -> None:
    banner(f"150 Hz sync cycle for {args.seconds:.0f} s")
    with Backend(args.can, args.bitrate, log=lambda m: print("  " + m)) as backend:
        nodes = backend.scan()
        present = [b for b, n in nodes.items() if n.present]
        if not check("nodes present for the cycle", bool(present)):
            return
        backend.select(present)
        for base in present:
            backend.set_target(base, 0.0)
            backend.set_enabled(base, False)   # targets only; no valve motion

        backend.start_cycle()
        time.sleep(args.seconds)
        backend.stop_cycle()

        stats = backend.stats
        snap = backend.snapshot_nodes()
        expected = int(args.seconds * P.CYCLE_HZ)
        print(f"  cycles={stats.cycles} (expected ~{expected})")
        print(f"  jitter p95={stats.jitter_ms_p95:.3f} ms  max={stats.jitter_ms_max:.3f} ms  "
              f"late={stats.late_cycles}")
        check("cycle rate within 2 % of 150 Hz",
              abs(stats.cycles - expected) < expected * 0.02,
              f"{stats.cycles / args.seconds:.1f} Hz")
        check("cycle jitter p95 under 1 ms", stats.jitter_ms_p95 < 1.0,
              f"{stats.jitter_ms_p95:.3f} ms")

        for base, node in snap.items():
            if base not in present:
                continue
            total = node.replies + node.misses
            rate = node.replies / total if total else 0.0
            print(f"  0x{base:03X}: replies={node.replies} misses={node.misses} "
                  f"({rate * 100:.2f} %) latency={node.reply_latency_ms:.2f} ms "
                  f"p={node.pressure_psi:.2f} psi flags=0x{node.flags:02X}")
            check(f"0x{base:03X} answered every sync", rate > 0.999,
                  f"{node.misses} miss(es) in {total}")
            check(f"0x{base:03X} reply inside the cycle", node.reply_latency_ms < 6.6,
                  f"{node.reply_latency_ms:.2f} ms")
            check(f"0x{base:03X} reports no protocol error", not node.error,
                  f"flags=0x{node.flags:02X}")


def stage_latch(args) -> None:
    """The target must change on the sync edge, not when the table frame lands."""
    banner("sync-gated target latch")
    with CanLink(args.can, args.bitrate) as link:
        found = link.scan()
        if not check("a node to test", bool(found)):
            return
        base = next(iter(found))
        node = P.ids(base)
        cal = P.NodeCal.for_base(base)

        # Stage a target with the enable bit clear, so nothing drives a valve.
        counts = cal.psi_to_counts(3.0)
        link.drain()
        link.send_batch(P.build_runtime_table({base: (counts, False)}))
        quiet = link.collect(0.05, lambda cid, d: cid == base)
        check("no reply before the sync edge", not quiet,
              f"{len(quiet)} early frame(s)")

        link.send(*P.build_sync())
        replies = link.collect(0.05, lambda cid, d: cid == base)
        check("exactly one reply after the sync edge", len(replies) == 1,
              f"{len(replies)} frame(s)")
        if replies:
            status = P.parse_compact_status(replies[0][1], replies[0][2])
            check("reply is a well-formed compact status", status is not None,
                  f"counts={status.counts} flags=0x{status.flags:02X}" if status else "")
            check("command-seen flag set", status is not None and status.command_seen)

        # A second sync with nothing staged must stay silent: a board only
        # answers when its slot was addressed since the last edge.
        link.drain()
        link.send(*P.build_sync())
        silent = link.collect(0.05, lambda cid, d: cid == base)
        check("silent on an unaddressed sync", not silent, f"{len(silent)} frame(s)")


def stage_failsafe(args) -> None:
    """Losing the sync master must drop the outputs, not freeze the target."""
    banner("sync-loss failsafe")
    with CanLink(args.can, args.bitrate) as link:
        found = link.scan()
        if not check("a node to test", bool(found)):
            return
        base = next(iter(found))
        cal = P.NodeCal.for_base(base)

        # Enable at a low target, run a few cycles, then stop driving.
        counts = cal.psi_to_counts(2.0)
        for _ in range(30):
            link.send_batch(P.build_runtime_table({base: (counts, True)}) + [P.build_sync()])
            time.sleep(1.0 / P.CYCLE_HZ)
        link.drain()
        link.send_batch(P.build_runtime_table({base: (counts, True)}) + [P.build_sync()])
        replies = link.collect(0.05, lambda cid, d: cid == base)
        enabled = replies and P.parse_compact_status(replies[0][1], replies[0][2]).enabled
        check("board reports enabled while driven", bool(enabled))

        time.sleep(1.0)   # longer than TLE_LEGACY_SYNC_TIMEOUT_US (500 ms)

        link.drain()
        link.send_batch(P.build_runtime_table({base: (counts, True)}) + [P.build_sync()])
        replies = link.collect(0.05, lambda cid, d: cid == base)
        if not check("board still answering after the gap", bool(replies)):
            return
        status = P.parse_compact_status(replies[0][1], replies[0][2])
        # The first sync after the gap re-promotes the staged command, so the
        # board comes back enabled; what matters is that it dropped out during
        # the gap, which the error flag records.
        check("sync loss recorded", status.error or not status.enabled,
              f"flags=0x{status.flags:02X}")
        diag = link.can_diag(base)
        if diag:
            print(f"  diag: sync={diag.sync_counter} cmd={diag.command_counter} "
                  f"rx_ovf={diag.rx_overflow_count} tx_fail={diag.tx_fail_count} "
                  f"starvation/sync-loss={diag.starvation_count}")
            check("failsafe counter incremented", diag.starvation_count > 0,
                  f"{diag.starvation_count}")

        # Leave the board disabled.
        link.send_batch(P.build_runtime_table({base: (counts, False)}) + [P.build_sync()])


def stage_ota(args) -> None:
    banner("broadcast CAN OTA")
    image = load_image(args.image)
    print(f"  image {args.image} ({len(image)} bytes, "
          f"{-(-len(image) // P.OTA_BLOCK_BYTES)} blocks)")
    with CanLink(args.can, args.bitrate) as link:
        before = link.scan()
        if not check("nodes to update", bool(before)):
            return
        # Only boards that report themselves as TLE. On anything other than a
        # single-board bench, scan() returns the whole bus, and streaming a TLE
        # image into the sixteen 7 mm boards would take them all off the bus.
        bases = [b for b, v in before.items() if v.variant == P.VARIANT_TLE_DVP]
        skipped = [b for b in before if b not in bases]
        if skipped:
            print("  skipping non-TLE node(s): "
                  + ", ".join(f"0x{b:03X} ({before[b].variant_name})" for b in skipped))
        if not check("TLE nodes to update", bool(bases)):
            return
        ota = BroadcastOta(link, log=lambda m: print("  " + m))
        started = time.perf_counter()
        result = ota.upload(image, bases)
        elapsed = time.perf_counter() - started
        for base, ok in result.items():
            check(f"0x{base:03X} accepted the image", ok)
        print(f"  {elapsed:.1f} s total, {len(image) / elapsed / 1024:.1f} kB/s")

        print("  waiting for reboot...")
        time.sleep(6.0)
        after = link.scan()
        check("every node came back after the update",
              set(after) == set(before), f"{sorted(hex(b) for b in after)}")
        for base in bases:
            version = after.get(base)
            check(f"0x{base:03X} kept its ID and variant across the update",
                  version is not None and version.variant == before[base].variant,
                  f"{version.variant_name} fw {version.version}" if version else "did not answer")


def stage_all(args) -> None:
    stage_usb(args)
    stage_scan(args)
    stage_latch(args)
    stage_sync(args)
    stage_failsafe(args)
    if args.image:
        stage_ota(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["usb", "setid", "scan", "sync", "latch",
                                          "failsafe", "ota", "all"])
    parser.add_argument("--usb", default=USB_PORT)
    parser.add_argument("--can", default=CAN_PORT)
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--to", type=lambda s: int(s, 0), default=0x101)
    parser.add_argument("--image", default="")
    args = parser.parse_args()

    stages = {
        "usb": stage_usb, "setid": stage_setid, "scan": stage_scan, "sync": stage_sync,
        "latch": stage_latch, "failsafe": stage_failsafe, "ota": stage_ota, "all": stage_all,
    }
    stages[args.stage](args)

    failed = [r for r in _results if r[0] == FAIL]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    for _, name, detail in failed:
        print(f"  FAIL {name} -- {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
