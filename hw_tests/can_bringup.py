"""CAN bring-up verification for the 24-board arm — read/status only.

This script is the acceptance check that the whole bus is present, healthy and
answers the 150 Hz sync edge, run without commanding any actuator. Every stage
is deliberately non-actuating: the only frames it puts on the wire are firmware
version requests, CAN diagnostic requests, and the standard runtime table plus
sync edge with **every enable bit clear**. No OTA command, no set-ID command
and no enable bit is ever emitted, and the table is decoded byte-for-byte
before the cycle starts to prove it — see :func:`decode_table`.

Five stages, run in one process because each needs the previous one's result:

  1. ``scan``       — two-round discovery, per-board firmware version + variant,
                      run twice so a one-off silence is distinguishable from a
                      board that is off the bus.
  2. ``diag``       — CMD_GET_CAN_DIAG on every board, then CMD_CLEAR_CAN_DIAG,
                      so the soak's counters start from a known zero.
  3. ``soak``       — the Backend's 150 Hz sync master for ~60 s with nothing
                      selected and nothing enabled, measuring per-board reply
                      rate, reply latency and host cycle jitter.
  4. ``diag`` again — the same counters after the soak, which is where an RX
                      overflow or a transmit failure would show up.
  5. ``port``       — the adapter is reopened and closed, since the next test
                      phase needs the port free and a leaked handle is otherwise
                      invisible until that phase fails.

Reply accounting is done from this script's own tap on the receive thread
rather than from ``Backend.stats``. ``Backend._publish`` credits replies only to
boards in ``backend.selected``, and this test deliberately selects none, so the
backend's own per-node counters stay at zero by design. The tap reads
``backend._cycle_t0`` — the timestamp the cycle thread stamps immediately
before it writes the table and the sync edge — so latency is measured from the
same origin as the reference 24-board figures (2.04 ms at 0x101 rising to
3.49 ms at 0x118, the spread being CAN arbitration by node ID).

Usage, naming the workspace interpreter by absolute path:

    .venv\\Scripts\\python.exe hw_tests\\can_bringup.py
    .venv\\Scripts\\python.exe hw_tests\\can_bringup.py --soak-seconds 20
    .venv\\Scripts\\python.exe hw_tests\\can_bringup.py --port COM58
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[1]
for _p in (WS_ROOT, WS_ROOT / "TLE_PCB"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_env  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from tlelib.canlink import CanLink  # noqa: E402

# How this arm is wired: eight TLE/DVP boards on the top eight slots, sixteen
# legacy 7 mm boards below them. A deviation is reported, never tolerated — the
# variant byte is what picks a board's pressure calibration and what the OTA
# cross-flash guard keys on.
EXPECTED_VARIANT = {base: P.VARIANT_TLE_DVP for base in P.TLE_IDS}
EXPECTED_VARIANT.update({base: P.VARIANT_7MM for base in P.SEVEN_MM_IDS})

DIAG_TYPES = (P.MSG_CAN_DIAG0, P.MSG_CAN_DIAG1, P.MSG_CAN_DIAG2, P.MSG_CAN_DIAG3)


def hexid(base: int) -> str:
    return f"0x{base:03X}"


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1 — discovery
# ---------------------------------------------------------------------------
def sweep(link: CanLink, bases=None, label: str = "") -> dict:
    """One two-round firmware-version sweep, reduced to plain records."""
    t0 = time.perf_counter()
    found = link.scan(bases=bases)
    elapsed = time.perf_counter() - t0
    boards = {}
    for base, version in found.items():
        expected = EXPECTED_VARIANT.get(base)
        boards[base] = {
            "base": base,
            "version": version.version,
            "variant": version.variant,
            "variant_name": version.variant_name,
            "complete": bool(version.complete),
            "variant_expected": expected,
            "variant_ok": version.variant == expected,
        }
    candidates = list(P.ALL_IDS if bases is None else bases)
    say(f"  sweep {label}: {len(boards)}/{len(candidates)} answered in {elapsed:.2f} s")
    return {
        "label": label,
        "elapsed_s": elapsed,
        "boards": boards,
        "missing": [b for b in candidates if b not in boards],
    }


def stage_scan(link: CanLink) -> dict:
    """Discovery, run as two independent sweeps of the whole 0x101..0x118 range.

    Two sweeps rather than one because ``CanLink.scan`` already retries inside
    itself, and a board that answers one full sweep but not the other is a
    different fault from a board that answers neither. The second sweep is also
    what registers the boards with the backend, so it is not wasted work.
    """
    say("stage scan: CMD_GET_FW_VERSION across 0x101..0x118")
    first = sweep(link, label="A")
    second = sweep(link, label="B")

    boards = dict(second["boards"])
    for base, rec in first["boards"].items():
        boards.setdefault(base, rec)

    missing = [b for b in P.ALL_IDS if b not in boards]
    deviations = [b for b, rec in boards.items() if not rec["variant_ok"]]
    incomplete = [b for b, rec in boards.items() if not rec["complete"]]
    disagreement = sorted(set(first["boards"]) ^ set(second["boards"]))

    for base in sorted(boards):
        rec = boards[base]
        mark = "" if rec["variant_ok"] else "   <-- VARIANT DEVIATION"
        say(f"  {hexid(base)}  variant {rec['variant']} ({rec['variant_name']:<7})"
            f"  fw {rec['version']!r}{mark}")
    if missing:
        say(f"stage scan: MISSING {', '.join(hexid(b) for b in missing)}")
    if disagreement:
        say(f"stage scan: sweeps disagreed on {', '.join(hexid(b) for b in disagreement)}")

    return {
        "sweep_a": first,
        "sweep_b": second,
        "boards": boards,
        "missing": missing,
        "variant_deviations": deviations,
        "incomplete_versions": incomplete,
        "sweep_disagreement": disagreement,
    }


# ---------------------------------------------------------------------------
# Stages 2 and 4 — CAN diagnostics
# ---------------------------------------------------------------------------
def diag_raw(link: CanLink, base: int, command: int, timeout: float = 0.5) -> dict:
    """Send one diagnostic command and keep the four reply frames verbatim.

    ``CanLink.can_diag`` merges the four frames into a ``CanDiag`` and discards
    the bytes. The raw payloads are kept here as well, since a field the
    firmware never populates and a field that genuinely decoded to zero look
    identical once merged, and a bring-up record has to tell them apart.
    """
    node = P.ids(base)
    link.drain()
    link.send(node.ctrl, P.build_simple_command(command))
    frames: dict[int, bytes] = {}
    for _, _, data in link.collect(
            timeout,
            lambda cid, d: cid == node.status and bool(d) and d[0] in DIAG_TYPES,
            limit=len(DIAG_TYPES)):
        frames[data[0]] = bytes(data)
    merged = P.merge_can_diag(frames) if frames else None
    return {
        "base": base,
        "frames": {f"0x{k:02X}": v.hex().upper() for k, v in sorted(frames.items())},
        "frames_seen": len(frames),
        "merged": (dict(vars(merged)) if merged is not None else None),
    }


def stage_diag(link: CanLink, bases: list[int], label: str, clear: bool) -> dict:
    """CMD_GET_CAN_DIAG on every board; optionally CMD_CLEAR_CAN_DIAG after."""
    say(f"stage diag ({label}): CMD_GET_CAN_DIAG on {len(bases)} board(s)")
    read: dict[int, dict] = {}
    for base in bases:
        read[base] = diag_raw(link, base, P.CMD_GET_CAN_DIAG)
        if read[base]["frames_seen"] != 4:
            say(f"  {hexid(base)}  only {read[base]['frames_seen']}/4 diag frames returned")
    incomplete = [b for b, r in read.items() if r["frames_seen"] != 4]

    cleared: dict[int, dict] = {}
    if clear:
        say(f"stage diag ({label}): CMD_CLEAR_CAN_DIAG on {len(bases)} board(s)")
        for base in bases:
            # The firmware clears and then re-emits the four frames, so the
            # reply to the clear is itself the confirmation that it took.
            cleared[base] = diag_raw(link, base, P.CMD_CLEAR_CAN_DIAG)

    return {
        "read": {hexid(b): r for b, r in sorted(read.items())},
        "cleared": {hexid(b): r for b, r in sorted(cleared.items())},
        "incomplete": incomplete,
    }


# ---------------------------------------------------------------------------
# Stage 3 — the 150 Hz soak, nothing enabled
# ---------------------------------------------------------------------------
class SoakRecorder:
    """Receive-thread tap that credits every compact reply to its own cycle.

    Attribution is by ``backend._cycle_t0``, which the cycle thread rewrites
    once per period immediately before the table and the sync edge go out. A
    reply read while that value names cycle *n* belongs to cycle *n*, whatever
    order the consumer threads happen to run in. A float read needs no lock in
    CPython, and taking the cycle lock here would put this tap into the
    transmit thread's path.
    """

    def __init__(self, backend: Backend, bases: list[int]):
        self._backend = backend
        self.latency_ms: dict[int, list[float]] = {b: [] for b in bases}
        self.replies: dict[int, int] = {b: 0 for b in bases}
        self.duplicates: dict[int, int] = {b: 0 for b in bases}
        self.flags: dict[int, collections.Counter] = {b: collections.Counter() for b in bases}
        self.counts: dict[int, list[int]] = {b: [] for b in bases}
        self._last_cycle: dict[int, float] = {}
        # Every distinct cycle any board answered. ``Backend.stats.cycles``
        # counts publications, which happen 82 % of a period after the edge, so
        # it lags the tap by one cycle at the moment the soak ends and a board
        # can honestly show one more reply than there are counted cycles. The
        # larger of the two is the denominator, which keeps every rate at or
        # below 100 % without hiding a cycle that no board answered at all.
        self.cycle_ids: set[float] = set()
        self.foreign_frames: collections.Counter = collections.Counter()
        self.foreign_samples: dict[int, list[str]] = {}

    def __call__(self, t: float, can_id: int, data: bytes) -> None:
        status = P.parse_compact_status(can_id, data)
        if status is None or status.base not in self.replies:
            # Anything on the wire that is not a compact status: keep a few
            # payloads verbatim, since the ID alone does not say whether a
            # frame on base+0x300 was a diagnostic reply or native telemetry.
            self.foreign_frames[can_id] += 1
            samples = self.foreign_samples.setdefault(can_id, [])
            if len(samples) < 8:
                samples.append(bytes(data).hex().upper())
            return
        base = status.base
        cycle_t0 = self._backend._cycle_t0
        if self._last_cycle.get(base) == cycle_t0:
            # A second answer inside one cycle is not a second sample; count it
            # separately so the reply rate cannot be inflated by one.
            self.duplicates[base] += 1
            return
        self._last_cycle[base] = cycle_t0
        self.cycle_ids.add(cycle_t0)
        self.replies[base] += 1
        self.latency_ms[base].append((t - cycle_t0) * 1000.0)
        self.flags[base][status.flags] += 1
        self.counts[base].append(status.counts)


def decode_table(backend: Backend) -> dict:
    """Decode the table this backend would send and prove no enable bit is set.

    The enable bit is a level, not an edge, so a single table frame carrying it
    would leave a board regulating until another frame contradicted it. With no
    pneumatic supply attached nothing could move, but the check costs nothing
    and the record needs the exact bytes that went onto the wire, so they are
    decoded here and returned.
    """
    frames = P.build_runtime_table(backend._build_targets())
    slots = []
    for can_id, payload in frames:
        start_slot = payload[0] & P.RUNTIME_TABLE_SLOTMASK
        mask = payload[1]
        for offset in range(P.RUNTIME_TABLE_SLOTS):
            if not (mask >> offset) & 1:
                continue
            word = int.from_bytes(payload[2 + offset * 2:4 + offset * 2], "little")
            flags = (word >> 12) & P.FLAGS_MASK
            base = P.ACTUATOR_FIRST + start_slot + offset
            if flags & P.CONTROL_ENABLE:
                raise AssertionError(
                    f"refusing to start: table frame {hexid(can_id)} carries the enable "
                    f"bit for {hexid(base)} (word 0x{word:04X})")
            slots.append({"table_id": hexid(can_id), "base": hexid(base),
                          "counts": word & P.PRESSURE_MASK, "flags": flags})
    return {
        "frames": [{"id": hexid(cid), "payload": pl.hex().upper()} for cid, pl in frames],
        "sync": {"id": hexid(P.ID_BROADCAST), "payload": ""},
        "slots": slots,
        "enable_bits_set": 0,
    }


def stage_soak(backend: Backend, bases: list[int], seconds: float,
               drain_rx: bool = False) -> dict:
    say(f"stage soak: {seconds:.0f} s at {backend.cycle_hz:.0f} Hz, "
        f"{len(bases)} board(s) tabled, 0 selected, 0 enabled"
        + (", host RX queue drained once a second" if drain_rx else ""))

    # Pre-flight: nothing selected, nothing enabled, and the bytes that would
    # go out decoded and checked before the transmit thread starts.
    assert not backend.selected, "selection must be empty for a read-only soak"
    assert not any(n.enabled for n in backend.nodes.values()), "no node may be enabled"
    table = decode_table(backend)
    say(f"stage soak: pre-flight OK — {len(table['frames'])} table frame(s) + sync, "
        f"{len(table['slots'])} slots addressed, every enable bit clear")

    recorder = SoakRecorder(backend, bases)
    backend.link.add_tap(recorder)
    rx_count_0, rx_dropped_0 = backend.link.rx_count, backend.link.rx_dropped

    jitter_samples: list[dict] = []
    t_start = time.perf_counter()
    backend.start_cycle()
    try:
        next_sample = t_start + 1.0
        while time.perf_counter() - t_start < seconds:
            time.sleep(0.05)
            if time.perf_counter() >= next_sample:
                next_sample += 1.0
                if drain_rx:
                    # Nothing consumes CanLink.frames while the cycle runs --
                    # the backend takes its replies from the tap -- so the queue
                    # saturates and the reader starts evicting. Draining here
                    # measures whether that eviction costs anything.
                    backend.link.drain()
                s = backend.stats
                jitter_samples.append({
                    "t": round(time.perf_counter() - t_start, 3),
                    "cycles": s.cycles,
                    "jitter_p95_ms": s.jitter_ms_p95,
                    "jitter_max_ms": s.jitter_ms_max,
                    "late_cycles": s.late_cycles,
                })
            if not backend.running:
                say("stage soak: [ERROR] the cycle thread stopped early — aborting soak")
                break
    finally:
        # Detach first: reply counting must stop before the cycle counter does,
        # so a reply can never be credited to a cycle the counter has not
        # reached. The residue is at most one cycle counted as a miss.
        backend.link.remove_tap(recorder)
        cycles_published = backend.stats.cycles
        cycles = max(cycles_published, len(recorder.cycle_ids))
        wall = time.perf_counter() - t_start
        ran_to_completion = backend.running
        # stop_cycle() sends a final table with every enable bit clear, by
        # design, and does so even if the transmit thread already died.
        backend.stop_cycle()

    rx_count = backend.link.rx_count - rx_count_0
    rx_dropped = backend.link.rx_dropped - rx_dropped_0

    per_board = {}
    for base in bases:
        lat = sorted(recorder.latency_ms[base])
        replies = recorder.replies[base]
        cal = backend.nodes[base].cal
        seen = recorder.counts[base]
        per_board[base] = {
            "base": base,
            "kind": backend.nodes[base].kind,
            "replies": replies,
            "cycles": cycles,
            "misses": max(0, cycles - replies),
            "reply_rate_pct": (100.0 * replies / cycles) if cycles else 0.0,
            "duplicates": recorder.duplicates[base],
            "latency_ms_min": lat[0] if lat else None,
            "latency_ms_median": statistics.median(lat) if lat else None,
            "latency_ms_p95": lat[max(0, int(len(lat) * 0.95) - 1)] if lat else None,
            "latency_ms_max": lat[-1] if lat else None,
            "flags_histogram": {f"0x{k:X}": v for k, v in sorted(recorder.flags[base].items())},
            "counts_min": min(seen) if seen else None,
            "counts_max": max(seen) if seen else None,
            "counts_median": statistics.median(seen) if seen else None,
            # Idle pressure on the board's own reported scale, which is chosen
            # from the variant byte and not from the ID range.
            "idle_psi_median": cal.counts_to_psi(statistics.median(seen)) if seen else None,
            "cal_zero_counts": cal.zero_counts,
            "cal_counts_per_psi": cal.counts_per_psi,
        }

    jp95 = [s["jitter_p95_ms"] for s in jitter_samples] or [0.0]
    jmax = [s["jitter_max_ms"] for s in jitter_samples] or [0.0]
    result = {
        "requested_seconds": seconds,
        "wall_seconds": wall,
        "ran_to_completion": ran_to_completion,
        "cycles": cycles,
        "cycles_published": cycles_published,
        "cycles_answered": len(recorder.cycle_ids),
        "cycles_unanswered": max(0, cycles - len(recorder.cycle_ids)),
        "measured_hz": (cycles / wall) if wall else 0.0,
        "per_board": per_board,
        "host_jitter_p95_ms_worst": max(jp95),
        "host_jitter_max_ms_worst": max(jmax),
        "late_cycles": backend.stats.late_cycles,
        "link_rx_frames": rx_count,
        "link_rx_dropped": rx_dropped,
        "foreign_frames": {f"0x{k:03X}": v for k, v in sorted(recorder.foreign_frames.items())},
        "foreign_samples": {f"0x{k:03X}": v for k, v in sorted(recorder.foreign_samples.items())},
        "drained_rx": drain_rx,
        "table": table,
        "jitter_samples": jitter_samples,
    }
    say(f"stage soak: {cycles} cycles in {wall:.2f} s ({result['measured_hz']:.1f} Hz), "
        f"{result['cycles_unanswered']} unanswered, {rx_count} frames received, "
        f"{rx_dropped} evicted from the host RX queue")
    say(f"stage soak: host cycle jitter p95 {result['host_jitter_p95_ms_worst']:.4f} ms, "
        f"max {result['host_jitter_max_ms_worst']:.4f} ms, "
        f"{result['late_cycles']} late cycle(s)")
    for base in bases:
        r = per_board[base]
        if r["replies"]:
            say(f"  {hexid(base)}  {r['reply_rate_pct']:6.2f}%  misses {r['misses']:>5}  "
                f"lat med {r['latency_ms_median']:.2f} / p95 {r['latency_ms_p95']:.2f} / "
                f"max {r['latency_ms_max']:.2f} ms")
        else:
            say(f"  {hexid(base)}  SILENT for the whole soak")
    return result


# ---------------------------------------------------------------------------
# Stage 5 — the port must be free for the next test phase
# ---------------------------------------------------------------------------
def stage_port_free(port: str, bitrate: int) -> dict:
    """Reopen and close the adapter to prove nothing of ours still holds it."""
    say(f"stage port: reopening {port} to confirm it is free")
    try:
        probe = CanLink(port, bitrate)
        probe.open()
        version = probe.adapter_version
        probe.close()
        say(f"stage port: {port} reopened and closed cleanly (adapter {version!r})")
        return {"port": port, "reopened": True, "adapter_version": version, "error": None}
    except Exception as exc:
        say(f"stage port: [ERROR] {port} could not be reopened: {exc}")
        return {"port": port, "reopened": False, "adapter_version": "", "error": repr(exc)}


def sync_deltas(report: dict) -> dict:
    """How far each board's sync and command counters moved across the soak.

    Measured as a difference rather than an absolute, because
    ``CMD_CLEAR_CAN_DIAG`` was found on this bench to reset the error, warning,
    overflow, transmit-failure, invalid-frame and starvation counters but *not*
    the sync and command counters, which run free from boot on both firmwares.
    An absolute non-zero reading therefore proves nothing; the difference is
    what says the board saw this run's edges. The baseline is the reading the
    board re-emitted in answer to the clear, since that is the reading closest
    in time to the first edge of the soak.
    """
    before = (report.get("diag_before") or {})
    baseline = before.get("cleared") or before.get("read") or {}
    after = (report.get("diag_after") or {}).get("read", {})
    out = {}
    for key, rec in sorted(after.items()):
        b = (baseline.get(key) or {}).get("merged")
        a = rec.get("merged")
        if not a or not b:
            out[key] = None
            continue
        out[key] = {
            "sync_delta": a["sync_counter"] - b["sync_counter"],
            "command_delta": a["command_counter"] - b["command_counter"],
            "sync_before": b["sync_counter"],
            "sync_after": a["sync_counter"],
        }
    return out


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
def evaluate(report: dict, reply_rate_floor: float) -> dict:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    scan = report.get("scan") or {}
    missing = scan.get("missing", [P.ACTUATOR_FIRST])
    add("all 24 boards 0x101-0x118 answered discovery", not missing,
        "missing " + ", ".join(hexid(b) for b in missing) if missing
        else f"{len(scan.get('boards', {}))}/24")
    add("both discovery sweeps agreed", not scan.get("sweep_disagreement"),
        "disagreed on " + ", ".join(hexid(b) for b in scan.get("sweep_disagreement", []))
        if scan.get("sweep_disagreement") else "identical board sets")
    add("every firmware version string arrived complete", not scan.get("incomplete_versions"),
        "incomplete: " + ", ".join(hexid(b) for b in scan.get("incomplete_versions", []))
        if scan.get("incomplete_versions") else "all complete")
    add("variant 2 on 0x101-0x108 and variant 0 on 0x109-0x118",
        not scan.get("variant_deviations"),
        "DEVIATION on " + ", ".join(hexid(b) for b in scan.get("variant_deviations", []))
        if scan.get("variant_deviations") else "as expected")

    for label in ("diag_before", "diag_after"):
        d = report.get(label)
        add(f"four diagnostic frames from every board ({label.replace('_', ' ')})",
            bool(d) and not d.get("incomplete"),
            "no data" if not d else
            ("incomplete: " + ", ".join(hexid(b) for b in d["incomplete"])
             if d["incomplete"] else "4/4 from every board"))

    soak = report.get("soak") or {}
    per_board = soak.get("per_board", {})
    add("the 150 Hz cycle ran the whole soak", bool(soak.get("ran_to_completion")),
        f"{soak.get('cycles', 0)} cycles in {soak.get('wall_seconds', 0):.1f} s "
        f"({soak.get('measured_hz', 0):.1f} Hz)")
    add("every sync edge was answered by at least one board",
        bool(soak) and soak.get("cycles_unanswered") == 0,
        "no data" if not soak else f"{soak['cycles_unanswered']} unanswered cycle(s)")
    slow = [b for b, r in per_board.items() if r["reply_rate_pct"] < reply_rate_floor]
    add(f"per-board reply rate >= {reply_rate_floor:.1f} %", bool(per_board) and not slow,
        "below floor: " + ", ".join(f"{hexid(b)} {per_board[b]['reply_rate_pct']:.2f}%"
                                    for b in slow) if slow
        else f"worst {min((r['reply_rate_pct'] for r in per_board.values()), default=0.0):.2f} %")
    errored = [b for b, r in per_board.items()
               if any(int(k, 16) & P.STATUS_ERROR for k in r["flags_histogram"])]
    add("no board raised the status error flag", not errored,
        "error flag on " + ", ".join(hexid(b) for b in errored) if errored else "clean")
    enabled_seen = [b for b, r in per_board.items()
                    if any(int(k, 16) & P.STATUS_ENABLED for k in r["flags_histogram"])]
    add("no board reported itself enabled", not enabled_seen,
        "enabled bit seen on " + ", ".join(hexid(b) for b in enabled_seen)
        if enabled_seen else "every reply had the enabled bit clear")

    after = (report.get("diag_after") or {}).get("read", {})
    over = [k for k, r in after.items() if r["merged"] and r["merged"]["rx_overflow_count"]]
    txf = [k for k, r in after.items() if r["merged"] and r["merged"]["tx_fail_count"]]
    # The sync edges this run put out: one per published cycle, plus the cycle
    # in flight when the counter was read, plus the safe-disable stop_cycle
    # sends. A board must have counted at least the published cycles and no
    # more than a couple beyond, and every board must agree.
    cycles = soak.get("cycles", 0)
    deltas = report.get("sync_deltas") or {}
    short = [k for k, v in deltas.items()
             if not v or not (cycles <= v["sync_delta"] <= cycles + 5)]
    spread = {v["sync_delta"] for v in deltas.values() if v}
    add("0 RX overflows after the soak", bool(after) and not over,
        "no data" if not after else
        ("overflow on " + ", ".join(over) if over else "0 on every board"))
    add("0 TX failures after the soak", bool(after) and not txf,
        "no data" if not after else
        ("tx failures on " + ", ".join(txf) if txf else "0 on every board"))
    add(f"sync counter advanced by the soak's {cycles} cycles on every board",
        bool(deltas) and not short and len(spread) == 1,
        "no data" if not deltas else
        ("short on " + ", ".join(short) if short else
         f"+{spread.pop()} on every board, identical across the bus"))

    port = report.get("port_free") or {}
    add("adapter port free after the run", bool(port.get("reopened")),
        port.get("error") or "reopened and closed")

    return {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
            "passed": all(ok for _, ok, _ in checks)}


# ---------------------------------------------------------------------------
# Report table, emitted from the same numbers the JSON carries
# ---------------------------------------------------------------------------
def markdown_table(report: dict) -> str:
    scan = report.get("scan") or {}
    soak = report.get("soak") or {}
    before = (report.get("diag_before") or {}).get("read", {})
    after = (report.get("diag_after") or {}).get("read", {})
    deltas = report.get("sync_deltas") or {}
    rows = ["| Board | Variant | Firmware | Replies / cycles | Rate | Latency med / p95 / max (ms) "
            "| Idle psi | Sync ctr delta | Cmd ctr delta | RX ovf | TX fail | Bus err | Invalid "
            "| Starv |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for base in P.ALL_IDS:
        key = hexid(base)
        b = scan.get("boards", {}).get(base)
        s = soak.get("per_board", {}).get(base)
        a = (after.get(key) or {}).get("merged") or {}
        if b is None:
            rows.append(f"| `{key}` | **NO REPLY** |" + " — |" * 13)
            continue
        variant = f"{b['variant']} ({b['variant_name']})"
        if not b["variant_ok"]:
            variant = f"**{variant}**"
        if s and s["replies"]:
            counts = f"{s['replies']} / {s['cycles']}"
            rate = f"{s['reply_rate_pct']:.2f} %"
            lat = (f"{s['latency_ms_median']:.2f} / {s['latency_ms_p95']:.2f} / "
                   f"{s['latency_ms_max']:.2f}")
            psi = f"{s['idle_psi_median']:+.2f}"
        else:
            counts, rate, lat, psi = "0", "0 %", "—", "—"
        dl = deltas.get(key) or {}
        rows.append(
            f"| `{key}` | {variant} | `{b['version']}` | {counts} | {rate} | {lat} | {psi} "
            f"| +{dl.get('sync_delta', '—')} | +{dl.get('command_delta', '—')} "
            f"| {a.get('rx_overflow_count', '—')} | {a.get('tx_fail_count', '—')} "
            f"| {a.get('error_count', '—')} | {a.get('invalid_frame_count', '—')} "
            f"| {a.get('starvation_count', '—')} |")
    _ = before
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="override the resolved CAN adapter port")
    ap.add_argument("--bitrate", type=int, default=bench_env.BITRATE)
    ap.add_argument("--soak-seconds", type=float, default=60.0)
    ap.add_argument("--reply-rate-floor", type=float, default=99.0,
                    help="per-board reply rate below which the soak fails, in percent")
    ap.add_argument("--drain-rx", action="store_true",
                    help="drain CanLink.frames once a second during the soak; off by "
                         "default so the soak matches how the controller GUI runs")
    ap.add_argument("--json", default=None, help="where the machine-readable record lands")
    args = ap.parse_args()

    port = bench_env.resolve_can_port(args.port)
    say(f"CAN adapter: {bench_env.describe_can_port(args.port)}")
    say(f"bitrate    : {args.bitrate} bit/s")
    say("this run emits version, diagnostic, runtime-table and sync frames only; "
        "no OTA, no set-ID, no enable bit")

    report: dict = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "port": port,
        "port_description": bench_env.describe_can_port(args.port),
        "bitrate": args.bitrate,
        "soak_seconds_requested": args.soak_seconds,
        "reply_rate_floor_pct": args.reply_rate_floor,
    }

    backend = Backend(port=port, bitrate=args.bitrate,
                      log=lambda m: print(f"   backend: {m}", flush=True))
    try:
        backend.open()
        report["adapter_version"] = backend.link.adapter_version
        say(f"adapter version string: {backend.link.adapter_version!r}")

        report["scan"] = stage_scan(backend.link)
        # Register what answered. Backend.scan runs one more sweep through the
        # same link; its result is the registry the cycle actually drives.
        nodes = backend.scan()
        bases = sorted(b for b, n in nodes.items() if n.present)
        report["registered"] = [hexid(b) for b in bases]

        if not bases:
            report["fatal"] = "no board answered discovery; nothing to soak"
            say("[FATAL] no board answered discovery")
        else:
            report["diag_before"] = stage_diag(backend.link, bases, "before soak", clear=True)
            report["soak"] = stage_soak(backend, bases, args.soak_seconds,
                                        drain_rx=args.drain_rx)
            report["diag_after"] = stage_diag(backend.link, bases, "after soak", clear=False)
    finally:
        try:
            backend.close()
            say("CAN link closed")
        except Exception as exc:  # reported, never swallowed
            say(f"[ERROR] closing the link raised: {exc}")
            report["close_error"] = repr(exc)

    report["port_free"] = stage_port_free(port, args.bitrate)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["sync_deltas"] = sync_deltas(report)
    report["acceptance"] = evaluate(report, args.reply_rate_floor)

    stamp = time.strftime("%Y-%m-%d")
    out = Path(args.json) if args.json else \
        Path(__file__).resolve().parent / "results" / f"can_bringup_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    table = out.with_suffix(".md")
    table.write_text(markdown_table(report) + "\n", encoding="utf-8")
    say(f"wrote {out}")
    say(f"wrote {table}")

    print()
    print("=" * 78)
    for check in report["acceptance"]["checks"]:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['name']}: {check['detail']}")
    print("=" * 78)
    verdict = "PASSED" if report["acceptance"]["passed"] else "FAILED"
    print(f"  {verdict}")
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
