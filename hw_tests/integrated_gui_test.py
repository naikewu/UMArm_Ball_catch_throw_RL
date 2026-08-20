"""Drive the real operator GUI against the real 24-board arm, end to end.

The acceptance question this answers is not "does each part work" -- the CAN
bring-up, the viewer's render path and the sim mocap stream each have their own
check already -- it is whether the three run **in one process at the same time**
without the 150 Hz cycle noticing.  So this script builds
``canarm_control_gui.CanArmControllerApp`` in a real Tk window, pumps it with
``root.update()``, and presses its own buttons by calling the handlers the
widgets are bound to.  Every byte that reaches a board is still emitted by
``tlelib.Backend``'s cycle thread; nothing here writes to the link.

STATUS-ONLY, and it is enforced rather than intended:

* :func:`assert_enables_clear` decodes the runtime table the backend *would*
  send -- the same ``_build_targets`` / ``build_runtime_table`` pair the cycle
  thread calls -- and raises if any command word carries ``CONTROL_ENABLE``.
  It runs before the cycle starts, again while a target is staged, and again at
  the end.
* ``set_enabled`` / ``enable_selected`` / ``stop_all`` are never called, no OTA
  or set-ID command is ever built, and the Kinova is never contacted.
* A receive tap ORs together every compact-status flag byte each board sends,
  so "no board ever reported itself enabled" is evidence, not an assumption.

Phases, in order, each recorded into ``results/integrated_gui_2026-08-20.json``:

1. connect + scan (the GUI's own handlers); the boards' CAN diagnostics read
   and then cleared, so every counter from here belongs to this session; then
   the 150 Hz cycle started with every enable bit clear -- the bring-up soak's
   exact wire condition;
2. the mocap strip on ``sim``, then briefly on ``live`` to show it degrades
   loudly rather than hanging, then back to ``sim``;
3. the viewer spawned from the GUI, >= 60 s of GUI + viewer + cycle together,
   closed by posting WM_CLOSE to its own window (the operator's path);
4. one channel's target bar dragged to a small nonzero value with the enable
   bit still clear, then back to zero;
5. the diagnostics read again -- the boards' own account of the session -- then
   the window closed and the COM port's release verified from a fresh process.

``--phase profile`` runs a different experiment on the same bus: the window's
two periodic jobs are gated on and off in turn, which is how the cycle-rate
loss below 150 Hz was attributed to the 24-board matplotlib redraw rather than
to the node table, the backend lock, or the viewer.

    .venv\\Scripts\\python.exe hw_tests\\integrated_gui_test.py
    .venv\\Scripts\\python.exe hw_tests\\integrated_gui_test.py --phase profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
for _p in (WS / "TLE_PCB", WS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MEDIA = WS / "hw_tests" / "media"
RESULTS = WS / "hw_tests" / "results"

#: The board whose target bar phase 4 drags.  A TLE/DVP board, and not 0x104:
#: that one reads +4.27 psi at rest (bring-up open item 1) and a drifting
#: baseline would blunt the "pressure unchanged" comparison.
STAGE_BASE = 0x101
STAGE_PSI = 2.0

#: Baseline from ``report_can_bringup_2026-08-20.md`` sections 4 and 8, for the
#: comparison this test exists to make.  Headless, no GUI, no viewer.
BASELINE = {
    "cycles": 9003, "seconds": 60.027, "hz": 149.98,
    "reply_rate_pct": 100.00, "jitter_ms_p95": 0.0016, "jitter_ms_max": 0.108,
    "unanswered_sync_edges": 0, "late_cycles": 0,
}


# ---------------------------------------------------------------------------
# Win32 helpers -- finding, raising and closing a window that belongs to
# another process.  The viewer is deliberately a separate process, so tkinter
# knows nothing about its window and PIL can only grab screen rectangles.
# ---------------------------------------------------------------------------

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_SW_RESTORE = 9
_WM_CLOSE = 0x0010
_HWND_TOP = 0
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002


def windows_of_pid(pid: int) -> list[tuple[int, str, tuple]]:
    """Every visible top-level window owned by *pid*: (hwnd, title, rect)."""
    out: list[tuple[int, str, tuple]] = []

    def callback(hwnd, _lparam):
        owner = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and _user32.IsWindowVisible(hwnd):
            length = _user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, buf, length + 1)
            rect = wintypes.RECT()
            _user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if rect.right - rect.left > 40 and rect.bottom - rect.top > 40:
                out.append((hwnd, buf.value,
                            (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    _user32.EnumWindows(_WNDENUMPROC(callback), 0)
    return out


def raise_window(hwnd: int) -> None:
    """Best-effort foreground.  A failed raise costs a screenshot, not a test."""
    try:
        _user32.ShowWindow(hwnd, _SW_RESTORE)
        _user32.BringWindowToTop(hwnd)
        _user32.SetWindowPos(hwnd, _HWND_TOP, 0, 0, 0, 0, _SWP_NOSIZE | _SWP_NOMOVE)
        _user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def close_window(hwnd: int) -> None:
    """Ask the window manager to close it, exactly as clicking the X does."""
    _user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)


def grab(rect, name: str) -> str | None:
    """Screenshot a screen rectangle.  Returns the path written, or None."""
    try:
        from PIL import ImageGrab
    except ImportError:
        print("[warn] PIL missing; no screenshot")
        return None
    MEDIA.mkdir(parents=True, exist_ok=True)
    left, top, right, bottom = (int(v) for v in rect)
    # A bbox reaching past the desktop grabs black, which reads as a window
    # that failed to draw rather than one that is partly off-screen.
    width = _user32.GetSystemMetrics(0)
    height = _user32.GetSystemMetrics(1)
    left, top = max(left, 0), max(top, 0)
    right, bottom = min(right, width), min(bottom, height)
    if right <= left or bottom <= top:
        print(f"[warn] {name}: window is off-screen ({rect})")
        return None
    image = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
    path = MEDIA / name
    image.save(path)
    print(f"screenshot -> {path}  ({image.width}x{image.height})")
    return str(path)


def image_difference(path_a: str, path_b: str) -> dict:
    """How much two screenshots differ.  A still render differs by nothing."""
    from PIL import Image, ImageChops
    a = Image.open(path_a).convert("L")
    b = Image.open(path_b).convert("L")
    if a.size != b.size:
        return {"comparable": False, "reason": f"{a.size} vs {b.size}"}
    diff = ImageChops.difference(a, b)
    histogram = diff.histogram()
    total = sum(histogram)
    changed = sum(histogram[8:])          # 8/255 shrugs off JPEG-grade noise
    mean = sum(i * n for i, n in enumerate(histogram)) / max(total, 1)
    return {"comparable": True, "changed_fraction": changed / max(total, 1),
            "mean_abs_difference": mean}


# ---------------------------------------------------------------------------
# Safety: the bytes that would go out, decoded
# ---------------------------------------------------------------------------

def decode_table(backend) -> dict:
    """Decode the runtime table this backend would send, slot by slot.

    Calls the backend's own ``_build_targets`` and ``proto.build_runtime_table``
    -- the identical pair the cycle thread calls -- so these are the bytes on
    the wire and not a reconstruction of them.  Raises before returning if any
    slot carries the enable bit, because that bit is a level: one frame
    carrying it leaves a board regulating until another frame contradicts it.
    """
    from tlelib import proto as P

    frames = P.build_runtime_table(backend._build_targets())
    slots = {}
    for can_id, payload in frames:
        start = payload[0] & P.RUNTIME_TABLE_SLOTMASK
        mask = payload[1]
        for offset in range(P.RUNTIME_TABLE_SLOTS):
            if not (mask >> offset) & 1:
                continue
            word = int.from_bytes(payload[2 + offset * 2:4 + offset * 2], "little")
            flags = (word >> 12) & P.FLAGS_MASK
            base = P.ACTUATOR_FIRST + start + offset
            if flags & P.CONTROL_ENABLE:
                raise AssertionError(
                    f"refusing to continue: table frame 0x{can_id:03X} carries the "
                    f"enable bit for 0x{base:03X} (word 0x{word:04X})")
            slots[f"0x{base:03X}"] = {"table_id": f"0x{can_id:03X}",
                                      "counts": word & P.PRESSURE_MASK,
                                      "flags": flags}
    return {"slots": slots, "enable_bits_set": 0,
            "frames": {f"0x{cid:03X}": pl.hex().upper() for cid, pl in frames}}


def assert_enables_clear(backend, where: str) -> dict:
    """Both halves of the claim: the model says clear, and so do the bytes."""
    hot = [f"0x{b:03X}" for b, n in backend.nodes.items() if n.enabled]
    assert not hot, f"{where}: node(s) marked enabled: {hot}"
    table = decode_table(backend)
    print(f"[guard] {where}: {len(table['slots'])} slots tabled, every enable bit clear")
    return table


class StatusTap:
    """Receive-thread tap: what every board said, cheaply.

    Runs for the whole session on the same thread the backend's own tap does,
    at roughly 3600 frames/s, so it does dictionary stores and nothing else.
    The OR of the flag bytes is the record that answers "did any board ever
    report itself enabled" without keeping 200 000 frames.
    """

    def __init__(self):
        from tlelib import proto as P
        self._P = P
        self.replies: dict[int, int] = {}
        self.flag_or: dict[int, int] = {}
        self.last_counts: dict[int, int] = {}
        self.watch: int | None = None
        self.watch_counts: list[int] = []

    def __call__(self, t, can_id, data):
        status = self._P.parse_compact_status(can_id, data)
        if status is None:
            return
        base = status.base
        self.replies[base] = self.replies.get(base, 0) + 1
        self.flag_or[base] = self.flag_or.get(base, 0) | status.flags
        self.last_counts[base] = status.counts
        if base == self.watch:
            self.watch_counts.append(status.counts)

    def start_watch(self, base: int) -> None:
        self.watch_counts = []
        self.watch = base

    def stop_watch(self) -> list[int]:
        self.watch = None
        counts, self.watch_counts = self.watch_counts, []
        return counts


# ---------------------------------------------------------------------------
# Pumping the Tk loop, and reading what the GUI is showing
# ---------------------------------------------------------------------------

def pump(root, seconds: float, tick: float = 0.02, on_tick=None) -> None:
    """Run the Tk event loop for *seconds*, optionally sampling as it goes."""
    end = time.monotonic() + seconds
    next_sample = time.monotonic()
    while time.monotonic() < end:
        root.update()
        now = time.monotonic()
        if on_tick is not None and now >= next_sample:
            next_sample = now + 1.0
            on_tick()
        time.sleep(tick)
    root.update()


def stat_sample(app) -> dict:
    s = app.backend.stats
    return {"t": time.monotonic(), "cycles": s.cycles, "replies": s.replies,
            "misses": s.misses, "late_cycles": s.late_cycles,
            "jitter_ms_p95": s.jitter_ms_p95, "jitter_ms_max": s.jitter_ms_max}


def window_stats(samples: list[dict]) -> dict:
    """Summarise a run of samples as differences, never as running totals.

    The backend's counters run from ``start_cycle``, so a cumulative reply rate
    read during the viewer's lifetime is diluted by every cycle before it.  The
    difference across the window is the number the comparison needs.  Jitter is
    the exception: those two fields are already a rolling summary of the last
    600 cycles (~4 s), so the window's worst case is the max of the samples.
    """
    if len(samples) < 2:
        return {}
    a, b = samples[0], samples[-1]
    dt = b["t"] - a["t"]
    cycles = b["cycles"] - a["cycles"]
    replies = b["replies"] - a["replies"]
    misses = b["misses"] - a["misses"]
    p95 = sorted(s["jitter_ms_p95"] for s in samples)
    return {
        "seconds": round(dt, 3),
        "cycles": cycles,
        "hz": round(cycles / dt, 2) if dt else None,
        "replies": replies,
        "misses": misses,
        "reply_rate_pct": round(100.0 * replies / (replies + misses), 4)
        if (replies + misses) else None,
        "late_cycles": b["late_cycles"] - a["late_cycles"],
        "jitter_ms_p95_median": round(p95[len(p95) // 2], 4),
        "jitter_ms_p95_max": round(max(p95), 4),
        "jitter_ms_max": round(max(s["jitter_ms_max"] for s in samples), 4),
        "samples": len(samples),
    }


_STRIP = re.compile(
    r"bodies\s+(?P<bodies>\d+)\s+(?P<fps>[\d.]+) fps\s+frames (?P<frames>\d+)\s+"
    r"valid (?P<valid>\d+)\s+stale (?P<stale>[Yn])\s+q_stale (?P<q_stale>[Yn])")


def parse_strip(line: str) -> dict:
    """The mocap strip's one line, as fields.  ``{}`` when it is not that line."""
    m = _STRIP.search(line)
    if not m:
        return {"raw": line}
    d = m.groupdict()
    return {"raw": line, "bodies": int(d["bodies"]), "fps": float(d["fps"]),
            "frames": int(d["frames"]), "valid_frames": int(d["valid"]),
            "stale": d["stale"] == "Y", "q_stale": d["q_stale"] == "Y",
            "viewer_up": "viewer up" in line}


def row_readouts(app) -> dict:
    """What each board's row is showing right now, as the operator sees it."""
    out = {}
    for base, widgets in app.node_widgets.items():
        out[f"0x{base:03X}"] = {key: widgets[key].cget("text").strip()
                                for key in ("target", "pressure", "error",
                                            "latency", "miss", "flags")}
    return out


def python_processes() -> dict[int, str]:
    """Every python process and its command line.

    The command line matters: other agents run their own interpreters on this
    machine at the same time, and a PID set differenced across a three-minute
    test flags their churn as this run's leak.  Only a process whose command
    line names this workspace can be one of ours.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    found = {}
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip().isdigit():
            found[int(pid)] = cmd.strip()
    return found


def ours(cmdline: str) -> bool:
    """Whether a command line belongs to this test rather than to a neighbour."""
    low = cmdline.lower()
    return (str(WS).lower() in low
            or "integrated_gui_test" in low
            or "multiprocessing" in low and "spawn_main" in low)


# ---------------------------------------------------------------------------
# Board diagnostics: what the boards themselves recorded, before and after
# ---------------------------------------------------------------------------

DIAG_TYPES = (0xD0, 0xD1, 0xD2, 0xD3)


def read_diag(link, base: int, command: int, timeout: float = 0.5) -> dict | None:
    """One diagnostic command, merged.  ``command`` may be the clearing one.

    ``CMD_CLEAR_CAN_DIAG`` is a write, and it is the only write this test makes
    outside the runtime table.  It carries no enable bit, is not an OTA or
    set-ID command and cannot move an actuator; ``can_bringup.py`` uses it for
    the same reason, which is that a counter that runs free from boot can only
    be read as a difference.
    """
    from tlelib import proto as P

    node = P.ids(base)
    link.drain()
    link.send(node.ctrl, P.build_simple_command(command))
    frames: dict[int, bytes] = {}
    for _, _, data in link.collect(
            timeout,
            lambda cid, d: cid == node.status and bool(d) and d[0] in DIAG_TYPES,
            limit=len(DIAG_TYPES)):
        frames[data[0]] = bytes(data)
    if not frames:
        return None
    merged = P.merge_can_diag(frames)
    out = dict(vars(merged))
    out["frames_seen"] = len(frames)
    return out


def diag_sweep(link, bases, command, attempts: int = 3) -> dict:
    """One diagnostic exchange per board, retried before giving up on silence.

    A board that answers the second attempt was crowded off the bus, not
    missing, and the difference matters: this sweep's whole purpose is to say
    whether a board recorded a fault.
    """
    out = {}
    for base in bases:
        record = None
        for _ in range(attempts):
            record = read_diag(link, base, command)
            if record and record.get("frames_seen") == len(DIAG_TYPES):
                break
            time.sleep(0.05)
        out[f"0x{base:03X}"] = record
    return out


def diag_summary(sweep: dict, keys=("error_count", "warning_count",
                                    "starvation_count", "rx_overflow_count",
                                    "tx_fail_count", "invalid_frame_count")) -> dict:
    """The counters that are not zero anywhere, per board.  ``{}`` means clean."""
    out = {}
    for board, d in sweep.items():
        if not d:
            out[board] = "no reply"
            continue
        hot = {k: d[k] for k in keys if d.get(k)}
        if hot:
            hot["last_error_reason"] = f"0x{d.get('last_error_reason', 0):02X}"
            out[board] = hot
    return out


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def run(args, verdict: dict) -> dict:
    import tkinter as tk
    from tkinter import ttk

    import canarm_control_gui as CG
    import bench_env
    from tlelib import proto as P

    verdict.update({"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "python": sys.executable, "checks": []})

    def check(name: str, ok: bool, detail: str) -> bool:
        verdict["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    procs_before = python_processes()
    verdict["pids_before"] = sorted(procs_before)
    verdict["port_resolution"] = bench_env.describe_can_port()
    resolved = bench_env.resolve_can_port()
    print(f"CAN port: {verdict['port_resolution']}")

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = CG.CanArmControllerApp(root, mocap_source="off")
    root.title("CAN UMArm - integrated acceptance test")
    root.update()
    root.geometry("+40+20")
    pump(root, 0.6)

    tap = StatusTap()
    viewer_pid = None
    hwnd = title = rect = None

    try:
        # ---- 1. connect, scan, cycle -----------------------------------
        label = app.port.get()
        check("GUI preselected the resolved adapter", label.startswith(resolved),
              f"combobox {label!r}, bench_env resolved {resolved}")
        verdict["adapter_label"] = label

        t0 = time.monotonic()
        app.toggle_connect()               # opens the link AND runs scan()
        connect_s = time.monotonic() - t0
        pump(root, 0.5)
        if app.backend is None:
            raise RuntimeError("Connect failed; see the GUI log")
        app.backend.link.add_tap(tap)
        verdict["connect_seconds"] = round(connect_s, 2)
        verdict["adapter_version"] = app.backend.link.adapter_version

        rows = sorted(app.node_widgets)
        verdict["node_rows"] = [f"0x{b:03X}" for b in rows]
        verdict["scan"] = {f"0x{b:03X}": [n.kind, n.version]
                           for b, n in app.backend.snapshot_nodes().items()
                           if n.present}
        check("24 node rows built by the GUI's own scan", len(rows) == 24,
              f"{len(rows)} rows, 0x{rows[0]:03X}..0x{rows[-1]:03X}"
              if rows else "no rows")

        # What the boards recorded for the scan that has just run, and then a
        # clear, so everything counted from here on belongs to the GUI session
        # and to nothing before it.  Read-only apart from the clear itself.
        bases = sorted(app.node_widgets)
        verdict["diag_after_scan"] = diag_summary(
            diag_sweep(app.backend.link, bases, P.CMD_GET_CAN_DIAG))
        print(f"  diag after the GUI's scan: {verdict['diag_after_scan']}")
        diag_sweep(app.backend.link, bases, P.CMD_CLEAR_CAN_DIAG)
        verdict["diag_at_zero"] = diag_summary(
            diag_sweep(app.backend.link, bases, P.CMD_GET_CAN_DIAG))
        check("board diagnostics cleared to zero before the session",
              not verdict["diag_at_zero"],
              f"all {len(bases)} boards report zero error, warning, starvation, "
              f"overflow, transmit-fail and invalid-frame counts"
              if not verdict["diag_at_zero"] else str(verdict["diag_at_zero"]))

        verdict["table_before_cycle"] = assert_enables_clear(
            app.backend, "before start_cycle")
        selected = [b for b, v in app.selected.items() if v.get()]
        check("nothing enabled before the cycle starts",
              not any(n.enabled for n in app.backend.nodes.values()),
              f"{len(selected)} board(s) selected, 0 enabled")

        app.toggle_cycle()
        pump(root, 4.0)
        check("cycle running", bool(app.backend.running),
              f"backend.running={app.backend.running}, "
              f"cycles={app.backend.stats.cycles}")

        readouts = row_readouts(app)
        live = [b for b, r in readouts.items() if r["pressure"] not in ("-", "")]
        verdict["rows_at_connect"] = readouts
        check("every node row shows a live pressure reading", len(live) == 24,
              f"{len(live)}/24 rows numeric; e.g. 0x101={readouts.get('0x101', {})}")

        warm = [stat_sample(app)]
        pump(root, 6.0)
        warm.append(stat_sample(app))
        verdict["stats_gui_only"] = window_stats(warm)
        print(f"  GUI-only window: {verdict['stats_gui_only']}")

        root.attributes("-topmost", True)
        pump(root, 0.4)
        verdict["shot_connected"] = grab(gui_rect(root), "control_gui_connected.png")
        root.attributes("-topmost", False)

        # ---- 2. mocap strip: sim, live, sim -----------------------------
        app.mocap_source.set("sim")
        app.toggle_mocap()
        pump(root, 4.0)
        sim_strip = parse_strip(app.mocap_status.get())
        verdict["mocap_sim"] = sim_strip
        print(f"  strip (sim): {sim_strip.get('raw')}")
        # Five, not six: the synthetic base plate sits at the mocap origin with
        # an identity orientation, which is bit-identical to the row the pose
        # array was initialised with.  See the note in ``_mocap_line``.
        check("sim mocap: bodies visible and q not stale",
              sim_strip.get("bodies", 0) >= 5 and sim_strip.get("q_stale") is False
              and sim_strip.get("fps", 0) > 100.0,
              f"bodies={sim_strip.get('bodies')}/6 (the base plate at the origin "
              f"is indistinguishable from an unfilled row) fps="
              f"{sim_strip.get('fps')} q_stale={sim_strip.get('q_stale')} "
              f"valid={sim_strip.get('valid_frames')}")

        app.toggle_mocap()                 # stop sim
        pump(root, 0.4)
        app.mocap_source.set("live")
        log_before = app.log_text.get("1.0", "end")
        t0 = time.monotonic()
        app.toggle_mocap()                 # start live -- must not block
        start_live_s = time.monotonic() - t0
        pump(root, float(args.live_seconds))
        live_strip = parse_strip(app.mocap_status.get())
        live_log = app.log_text.get("1.0", "end")[len(log_before):]
        t0 = time.monotonic()
        if app._mocap is not None:
            app.toggle_mocap()             # stop live
        stop_live_s = time.monotonic() - t0
        pump(root, 0.4)
        verdict["mocap_live"] = {
            "strip": live_strip, "start_seconds": round(start_live_s, 3),
            "stop_seconds": round(stop_live_s, 3),
            "log": live_log.strip().splitlines()[-4:],
            "receiver_built": "[ERROR] mocap" not in live_log,
        }
        loud = (("[ERROR] mocap" in live_log)
                or (live_strip.get("q_stale") is True
                    and live_strip.get("frames", 1) == 0))
        held = start_live_s + args.live_seconds + stop_live_s
        verdict["mocap_live"]["total_seconds"] = round(held, 2)
        check("live mocap degrades loudly rather than hanging",
              loud and start_live_s < 5.0 and held <= 15.0,
              f"start {start_live_s:.2f}s, held {args.live_seconds}s, stop "
              f"{stop_live_s:.2f}s ({held:.1f}s total), "
              f"strip={live_strip.get('raw')!r}")

        app.mocap_source.set("sim")
        app.toggle_mocap()
        pump(root, 3.0)
        back = parse_strip(app.mocap_status.get())
        verdict["mocap_back_to_sim"] = back
        check("back on sim after the live trial",
              back.get("q_stale") is False and back.get("bodies", 0) >= 5,
              f"bodies={back.get('bodies')} fps={back.get('fps')} "
              f"q_stale={back.get('q_stale')}")

        # ---- 3. viewer, alongside the cycle -----------------------------
        app.rs485_var.set(False)
        app.kinova_var.set(False)
        # A proper control window, not a token one: the with-viewer numbers are
        # only interpretable against the same GUI a moment earlier.
        pre_viewer = [stat_sample(app)]
        pump(root, 15.0, on_tick=lambda: pre_viewer.append(stat_sample(app)))
        pre_viewer.append(stat_sample(app))
        verdict["stats_before_viewer"] = window_stats(pre_viewer)
        print(f"  GUI only, no viewer: {verdict['stats_before_viewer']}")

        app.toggle_viewer()
        pump(root, 1.0)
        proc = app._viewer_proc
        if proc is None:
            raise RuntimeError("viewer did not spawn; see the GUI log")
        viewer_pid = proc.pid
        verdict["viewer_pid"] = viewer_pid

        # MuJoCo's import plus the room MJCF build in a cold process is the
        # slow part; the window is not expected inside the first second.
        hwnd = None
        deadline = time.monotonic() + args.viewer_window_timeout
        while time.monotonic() < deadline:
            pump(root, 0.5)
            found = windows_of_pid(viewer_pid)
            if found:
                hwnd, title, rect = found[0]
                break
        verdict["viewer_window"] = (
            {"hwnd": hwnd, "title": title, "rect": rect} if hwnd
            else {"hwnd": None})
        check("viewer window opened", hwnd is not None,
              f"pid {viewer_pid}, title {title!r}, rect {rect}" if hwnd
              else f"no visible top-level window from pid {viewer_pid} in "
                   f"{args.viewer_window_timeout:.0f} s")

        samples = [stat_sample(app)]
        pump(root, 8.0, on_tick=lambda: samples.append(stat_sample(app)))

        shots = []
        if hwnd is not None:
            raise_window(hwnd)
            pump(root, 0.8)
            shots.append(grab(rect, "room_viewer_sim.png"))
        pump(root, 6.0, on_tick=lambda: samples.append(stat_sample(app)))
        if hwnd is not None:
            raise_window(hwnd)
            pump(root, 0.8)
            shots.append(grab(rect, "room_viewer_sim_t2.png"))
            verdict["viewer_motion"] = image_difference(shots[0], shots[1])
            m = verdict["viewer_motion"]
            check("viewer is drawing the arm moving with the sim q",
                  m.get("comparable") and m.get("changed_fraction", 0) > 0.005,
                  f"{m.get('changed_fraction', 0) * 100:.2f} % of pixels changed "
                  f"between two grabs 7 s apart, mean |diff| "
                  f"{m.get('mean_abs_difference', 0):.2f}/255")

        root.attributes("-topmost", True)
        root.lift()
        pump(root, 0.8, on_tick=lambda: samples.append(stat_sample(app)))
        verdict["cycle_line_with_viewer"] = app.status_var.get()
        verdict["strip_with_viewer"] = parse_strip(app.mocap_status.get())
        verdict["shot_with_viewer"] = grab(gui_rect(root), "control_gui_with_viewer.png")
        root.attributes("-topmost", False)

        remaining = max(0.0, args.viewer_seconds - (samples[-1]["t"] - samples[0]["t"]))
        pump(root, remaining, on_tick=lambda: samples.append(stat_sample(app)))
        samples.append(stat_sample(app))
        verdict["stats_with_viewer"] = window_stats(samples)
        print(f"  with viewer: {verdict['stats_with_viewer']}")
        check("viewer + GUI + cycle ran together for at least 60 s",
              verdict["stats_with_viewer"]["seconds"] >= 60.0
              and proc.is_alive(),
              f"{verdict['stats_with_viewer']['seconds']:.1f} s, viewer alive="
              f"{proc.is_alive()}")

        got = verdict["stats_with_viewer"]
        was = verdict["stats_before_viewer"]
        # The literal acceptance criterion, kept as its own check so that a
        # failure is recorded as a failure rather than argued away.
        check("reply rate with the viewer running matches the bring-up baseline",
              got["reply_rate_pct"] is not None and got["reply_rate_pct"] >= 99.99,
              f"{got['reply_rate_pct']:.4f} % over {got['cycles']} cycles "
              f"({got['misses']} misses) at {got['hz']:.1f} Hz vs baseline "
              f"{BASELINE['reply_rate_pct']:.2f} % at {BASELINE['hz']} Hz")
        # And the question the criterion was there to ask: is it the VIEWER.
        check("the viewer itself does not measurably change the cycle",
              abs(got["reply_rate_pct"] - was["reply_rate_pct"]) <= 3.0
              and abs(got["hz"] - was["hz"]) <= 5.0,
              f"with viewer {got['reply_rate_pct']:.2f} % at {got['hz']:.1f} Hz "
              f"vs the same GUI without it {was['reply_rate_pct']:.2f} % at "
              f"{was['hz']:.1f} Hz "
              f"(delta {got['reply_rate_pct'] - was['reply_rate_pct']:+.2f} pt, "
              f"{got['hz'] - was['hz']:+.1f} Hz)")

        # ---- 3c. close the viewer's own window --------------------------
        before_close = stat_sample(app)
        if hwnd is not None:
            close_window(hwnd)
        else:
            app.toggle_viewer()
        gone = False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            pump(root, 0.5)
            if not proc.is_alive():
                gone = True
                break
        verdict["viewer_exit_seconds"] = round(
            time.monotonic() - before_close["t"], 2)
        verdict["viewer_exitcode"] = proc.exitcode
        check("closing the viewer window ends only the viewer process", gone,
              f"exitcode={proc.exitcode} after "
              f"{verdict['viewer_exit_seconds']:.1f} s")

        after = [stat_sample(app)]
        pump(root, 15.0, on_tick=lambda: after.append(stat_sample(app)))
        after.append(stat_sample(app))
        verdict["stats_after_viewer"] = window_stats(after)
        print(f"  after viewer closed: {verdict['stats_after_viewer']}")
        adv = verdict["stats_after_viewer"]
        check("the CAN cycle keeps running after the viewer window closes",
              app.backend.running and adv["cycles"] > 1500
              and adv["replies"] > 30000,
              f"stats kept advancing: {adv['cycles']} further cycles at "
              f"{adv['hz']:.2f} Hz and {adv['replies']} further replies, "
              f"backend.running={app.backend.running}")
        check("closing the viewer did not degrade the cycle either",
              abs(adv["reply_rate_pct"] - got["reply_rate_pct"]) <= 3.0,
              f"{adv['reply_rate_pct']:.2f} % after vs "
              f"{got['reply_rate_pct']:.2f} % during "
              f"({adv['reply_rate_pct'] - got['reply_rate_pct']:+.2f} pt)")

        # the GUI's own strip and its button must have noticed the window is gone
        strip_after = parse_strip(app.mocap_status.get())
        verdict["strip_after_viewer"] = strip_after
        verdict["viewer_button_after"] = app.viewer_button.cget("text")
        check("GUI reports the viewer down and the sim stream still up",
              strip_after.get("viewer_up") is False
              and strip_after.get("q_stale") is False
              and app.viewer_button.cget("text") == "Viewer",
              f"{strip_after.get('raw', '')!r}; button reads "
              f"{app.viewer_button.cget('text')!r}")

        # ---- 4. target staging, enable bit still clear ------------------
        bar = app.node_bars[STAGE_BASE]
        node_before = app.backend.snapshot_nodes()[STAGE_BASE]
        tap.start_watch(STAGE_BASE)
        pump(root, 4.0)
        counts_before = tap.stop_watch()

        class _Event:                       # what ChannelBar._drag reads
            def __init__(self, x):
                self.x, self.y, self.delta = x, 10, 0

        tap.start_watch(STAGE_BASE)
        bar._drag(_Event(bar._to_x(STAGE_PSI)))    # the widget's own handler
        pump(root, 4.0)
        counts_staged = tap.stop_watch()
        table_staged = assert_enables_clear(app.backend, "with a target staged")
        node_staged = app.backend.snapshot_nodes()[STAGE_BASE]
        row = row_readouts(app)[f"0x{STAGE_BASE:03X}"]

        verdict["target_staging"] = {
            "base": f"0x{STAGE_BASE:03X}",
            "commanded_psi": STAGE_PSI,
            "bar_target": bar.target,
            "app_target": app.targets[STAGE_BASE],
            "backend_target_psi": node_staged.target_psi,
            "backend_target_counts": node_staged.target_counts,
            "gui_row": row,
            "table_slot_before": verdict["table_before_cycle"]["slots"]
            .get(f"0x{STAGE_BASE:03X}"),
            "table_slot_staged": table_staged["slots"].get(f"0x{STAGE_BASE:03X}"),
            "table_frame_staged": table_staged["frames"].get(
                table_staged["slots"][f"0x{STAGE_BASE:03X}"]["table_id"]),
            "enabled_flag": node_staged.enabled,
            "status_flags_or": f"0x{tap.flag_or.get(STAGE_BASE, 0):02X}",
            "pressure_counts_before": summarise(counts_before),
            "pressure_counts_staged": summarise(counts_staged),
            "neighbours_unchanged": {
                k: table_staged["slots"][k]["counts"]
                for k in ("0x102", "0x103") if k in table_staged["slots"]},
        }
        ts = verdict["target_staging"]
        check("staged target is reflected in the GUI and in the table bytes",
              abs(ts["app_target"] - STAGE_PSI) < 1e-6
              and abs(ts["backend_target_psi"] - STAGE_PSI) < 1e-6
              and ts["table_slot_staged"]["counts"] == node_staged.target_counts
              and ts["table_slot_staged"]["counts"]
              != ts["table_slot_before"]["counts"],
              f"bar/app/backend all {STAGE_PSI} psi; table slot counts "
              f"{ts['table_slot_before']['counts']} -> "
              f"{ts['table_slot_staged']['counts']} in frame "
              f"{ts['table_slot_staged']['table_id']}")

        flags_or = tap.flag_or.get(STAGE_BASE, 0)
        was, now = ts["pressure_counts_before"], ts["pressure_counts_staged"]
        drift = abs(now["mean"] - was["mean"])
        # "Unchanged" has to be judged against the sensor's own idle spread
        # rather than against zero, so the band the pressure occupied before
        # the drag is the reference, widened by a couple of counts.
        band = (was["min"] - 3, was["max"] + 3)
        in_band = band[0] <= now["min"] and now["max"] <= band[1]
        ts["idle_band"] = band
        ts["drift_counts"] = round(drift, 2)
        check("the board stayed disabled and its pressure did not move",
              not node_staged.enabled
              and ts["table_slot_staged"]["flags"] == 0
              and not (flags_or & P.STATUS_ENABLED)
              and in_band,
              f"table flags 0, status flags OR 0x{flags_or:02X} "
              f"(ENABLED bit clear), pressure counts "
              f"{was['mean']:.1f} -> {now['mean']:.1f} (drift {drift:.1f}), "
              f"staged range {now['min']}..{now['max']} inside the idle band "
              f"{band[0]}..{band[1]}")

        root.attributes("-topmost", True)
        pump(root, 0.4)
        verdict["shot_target_staged"] = grab(gui_rect(root),
                                             "control_gui_target_staged.png")
        root.attributes("-topmost", False)

        bar._drag(_Event(bar._to_x(0.0)))          # and back to zero
        pump(root, 3.0)
        table_zero = assert_enables_clear(app.backend, "after returning to zero")
        node_zero = app.backend.snapshot_nodes()[STAGE_BASE]
        verdict["target_staging"]["table_slot_after"] = \
            table_zero["slots"].get(f"0x{STAGE_BASE:03X}")
        check("target returned to zero",
              node_zero.target_psi == 0.0 and app.targets[STAGE_BASE] == 0.0
              and table_zero["slots"][f"0x{STAGE_BASE:03X}"]["counts"]
              == verdict["table_before_cycle"]["slots"][f"0x{STAGE_BASE:03X}"]["counts"],
              f"backend target {node_zero.target_psi} psi, table slot counts back "
              f"to {table_zero['slots'][f'0x{STAGE_BASE:03X}']['counts']}")

        # ---- whole-session evidence -------------------------------------
        verdict["final_cycle_line"] = app.status_var.get()
        verdict["tap"] = {
            "boards_heard": len(tap.replies),
            "total_replies": sum(tap.replies.values()),
            "flags_or": {f"0x{b:03X}": f"0x{f:02X}" for b, f in sorted(tap.flag_or.items())},
            "idle_counts": {f"0x{b:03X}": c for b, c in sorted(tap.last_counts.items())},
        }
        # The one number that separates "a board went silent" from "the host
        # stopped crediting an answer that did arrive".
        cycles_total = app.backend.stats.cycles
        heard = sum(tap.replies.values())
        per_edge = heard / max(cycles_total, 1)
        credited = app.backend.stats.replies / max(
            app.backend.stats.replies + app.backend.stats.misses, 1) * 100
        verdict["replies_per_sync_edge"] = round(per_edge, 3)
        verdict["credited_reply_rate_pct"] = round(credited, 3)
        check("every board answered every sync edge",
              per_edge >= 23.9,
              f"{heard} compact statuses over {cycles_total} sync edges = "
              f"{per_edge:.2f} per edge with 24 boards on the bus, while the "
              f"backend credited only {credited:.2f} % of them -- the shortfall "
              f"is replies landing outside its 82 % reply window, not silence")

        enabled_seen = [f"0x{b:03X}" for b, f in tap.flag_or.items()
                        if f & P.STATUS_ENABLED]
        error_seen = [f"0x{b:03X}" for b, f in tap.flag_or.items()
                      if f & P.STATUS_ERROR]
        verdict["boards_reporting_error"] = error_seen
        check("no board ever reported itself enabled",
              not enabled_seen,
              f"{heard} compact statuses from {len(tap.replies)} boards, flag "
              f"OR = {[f'0x{v:02X}' for v in sorted(set(tap.flag_or.values()))]}, "
              f"the ENABLED bit clear in every one")
        check("no board raised its error flag during the session",
              not error_seen,
              f"{len(error_seen)} board(s) latched STATUS_ERROR: "
              f"{error_seen[:4]}{'...' if len(error_seen) > 4 else ''}"
              if error_seen else "no board latched STATUS_ERROR")

        # Stop the cycle first, through the GUI's own button.  A diagnostic
        # exchange is one command frame and four replies against 3600 frames a
        # second of runtime traffic, and four of twenty-four boards' replies
        # were lost in that flood when this was read with the cycle running --
        # which reads as a missing board rather than as a crowded bus.
        app.toggle_cycle()
        pump(root, 1.5)
        verdict["cycle_stopped_line"] = app.status_var.get()
        check("the GUI stops the cycle and leaves every board disabled",
              not app.backend.running
              and not any(n.enabled for n in app.backend.nodes.values()),
              f"backend.running={app.backend.running}, status line "
              f"{app.status_var.get()!r}")

        # The boards' own account of the session, against the zero they were
        # cleared to after the scan.  This is what separates a flag the GUI
        # caused from one it inherited.
        verdict["diag_after_session"] = diag_summary(
            diag_sweep(app.backend.link, bases, P.CMD_GET_CAN_DIAG))
        print(f"  diag after the session: {verdict['diag_after_session']}")
        check("the boards recorded no bus fault across the whole session",
              not verdict["diag_after_session"],
              "every board still reports zero errors, warnings, starvations, "
              "RX overflows, transmit failures and invalid frames"
              if not verdict["diag_after_session"]
              else str(verdict["diag_after_session"]))

        verdict["log_tail"] = app.log_text.get("1.0", "end").strip().splitlines()[-25:]

        # ---- 5. shutdown -------------------------------------------------
        t0 = time.monotonic()
        app.on_close()                       # viewer down, mocap down, bus down
        verdict["close_seconds"] = round(time.monotonic() - t0, 2)
        root = None
        check("GUI closed cleanly", True, f"on_close() took {verdict['close_seconds']} s")

    finally:
        if root is not None:
            try:
                app.on_close()
            except Exception as exc:
                print(f"[warn] on_close: {type(exc).__name__}: {exc}")
        # Only ever by handle, never by a remembered PID: the number is free
        # for reuse the moment the child exits.
        stray = locals().get("proc")
        if stray is not None and stray.is_alive():
            print(f"[warn] viewer still alive at teardown; terminating {stray.pid}")
            stray.terminate()
            stray.join(timeout=5.0)

    # A different process, so the answer is about the operating system's idea
    # of the port and not about this interpreter's.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from tlelib.canlink import CanLink; "
         "l = CanLink(r'%s'); l.open(); v = l.adapter_version; l.close(); "
         "print('REOPENED', v)" % (WS / "TLE_PCB", resolved)],
        capture_output=True, text=True, timeout=60)
    verdict["port_release"] = {"returncode": probe.returncode,
                               "stdout": probe.stdout.strip(),
                               "stderr": probe.stderr.strip()[-400:]}
    check("COM port released (reopened from a fresh process)",
          probe.returncode == 0 and "REOPENED" in probe.stdout,
          probe.stdout.strip() or probe.stderr.strip()[-200:])

    time.sleep(1.0)
    procs_after = python_processes()
    new = {pid: cmd for pid, cmd in procs_after.items()
           if pid not in procs_before and pid != os.getpid()}
    strays = {pid: cmd for pid, cmd in new.items() if ours(cmd)}
    verdict["pids_after"] = sorted(procs_after)
    verdict["new_processes"] = {str(p): c for p, c in new.items()}
    verdict["strays"] = {str(p): c for p, c in strays.items()}
    check("no stray python process from this run", not strays,
          f"{len(new)} python process(es) appeared during the run and "
          f"{len(strays)} of them name this workspace"
          + (f": {list(strays)}" if strays
             else " (the rest belong to other sessions on this machine)"))

    verdict["nondaemon_threads"] = [t.name for t in threading.enumerate()
                                    if not t.daemon and t is not threading.current_thread()]
    verdict["passed"] = sum(1 for c in verdict["checks"] if c["pass"])
    verdict["total"] = len(verdict["checks"])
    verdict["result"] = ("PASSED" if verdict["passed"] == verdict["total"]
                         else "FAILED")
    return verdict


def profile(args, verdict: dict) -> dict:
    """Which half of the window costs the cycle its period, measured not guessed.

    The full run shows the 150 Hz cycle achieving ~132 Hz with ~5 % of replies
    uncredited, and the viewer accounting for almost none of it.  That leaves
    the window's own two periodic jobs -- the ten-times-a-second node table and
    the four-times-a-second matplotlib redraw -- and the way to tell them apart
    is to run the same bus with each of them switched off in turn.

    Both jobs re-arm themselves with ``root.after(..., self._refresh)``, so
    replacing the attribute is enough: the next re-arm picks up the gate, and
    the gate keeps the chain alive by re-arming itself when it declines to run
    the real job.  Nothing about the backend or the wire changes between
    conditions.
    """
    import tkinter as tk
    from tkinter import ttk

    import canarm_control_gui as CG
    import bench_env

    VTC = CG.VTC
    verdict.update({"phase": "profile", "checks": [],
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    verdict["pids_before"] = sorted(python_processes())
    resolved = bench_env.resolve_can_port()

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = CG.CanArmControllerApp(root, mocap_source="off")
    root.title("CAN UMArm - cycle cost attribution")
    pump(root, 0.6)
    tap = StatusTap()

    try:
        app.toggle_connect()
        pump(root, 0.5)
        if app.backend is None:
            raise RuntimeError("Connect failed")
        app.backend.link.add_tap(tap)
        assert_enables_clear(app.backend, "profile: before start_cycle")
        app.toggle_cycle()
        pump(root, 3.0)

        gates = {"refresh": True, "replot": True}
        real_refresh, real_replot = app._refresh, app._replot

        def gated_refresh():
            if gates["refresh"]:
                real_refresh()
            else:
                root.after(VTC.REFRESH_MS, app._refresh)

        def gated_replot():
            mode = gates["replot"]
            if mode == "full":
                try:
                    real_replot()             # re-arms itself
                except Exception as exc:      # keep the chain alive: a dead
                    print(f"[replot raised {exc!r}; chain re-armed]")
                    root.after(VTC.PLOT_MS, app._replot)
                return
            if mode == "history":
                # The lock-held half of the redraw and nothing else: the same
                # 24 calls to Backend.history(), whose copies are made while
                # holding the lock the cycle thread needs to publish, with no
                # matplotlib behind them.
                if app.backend is not None:
                    for base, var in app.selected.items():
                        if var.get():
                            app.backend.history(base)
            root.after(VTC.PLOT_MS, app._replot)

        app._refresh, app._replot = gated_refresh, gated_replot
        pump(root, 1.0)                       # let both chains pick up the gate

        real_history = app.backend.history
        frozen: dict = {}

        def cached_history(base, max_points=None, **_kw):
            # Signature mirrors Backend.history (incl. max_points, added
            # 2026-08-20): a mismatch here raises inside _replot, which dies
            # BEFORE its self-re-arming root.after — silently freezing the
            # plot for every later condition.
            hist = list(frozen.get(base, ()))
            if max_points is not None and len(hist) > max_points:
                step = max(1, len(hist) // max_points)
                return hist[::step]
            return hist

        # (label, node table on, plot mode, history from a frozen snapshot)
        conditions = [
            ("full window: node table 10 Hz + 24-board plot 4 Hz", True, "full", False),
            ("node table only, plot pump idle", True, "off", False),
            ("plot only, node table idle", False, "full", False),
            ("neither; Tk pumped but nothing redrawn", False, "off", False),
            ("plot only, drawing a frozen history (no backend lock)", False, "full", True),
            ("Backend.history() copies only, no matplotlib", False, "history", False),
            ("full window again (drift control)", True, "full", False),
        ]
        rows = []
        for name, refresh_on, replot_mode, freeze in conditions:
            if freeze:
                frozen = {b: real_history(b) for b in app.node_widgets}
                app.backend.history = cached_history
            else:
                app.backend.history = real_history
            gates["refresh"], gates["replot"] = refresh_on, replot_mode
            pump(root, 3.0)                   # settle before measuring
            heard0 = sum(tap.replies.values())
            samples = [stat_sample(app)]
            pump(root, args.profile_seconds,
                 on_tick=lambda: samples.append(stat_sample(app)))
            samples.append(stat_sample(app))
            heard1 = sum(tap.replies.values())
            row = window_stats(samples)
            row["condition"] = name
            row["heard_on_the_wire"] = heard1 - heard0
            row["replies_per_sync_edge"] = round(
                (heard1 - heard0) / max(row["cycles"], 1), 3)
            rows.append(row)
            print(f"  {name:52s} {row['hz']:7.2f} Hz  "
                  f"credited {row['reply_rate_pct']:6.2f} %  "
                  f"heard {row['replies_per_sync_edge']:5.2f}/edge  "
                  f"jitter p95 {row['jitter_ms_p95_median']:.3f} "
                  f"max {row['jitter_ms_max']:.2f} ms  late {row['late_cycles']}")
        verdict["conditions"] = rows
        verdict["result"] = "MEASURED"
    finally:
        try:
            app.on_close()
        except Exception as exc:
            print(f"[warn] on_close: {type(exc).__name__}: {exc}")
    return verdict


def summarise(counts: list[int]) -> dict:
    if not counts:
        return {"n": 0, "mean": float("nan"), "min": None, "max": None}
    return {"n": len(counts), "mean": sum(counts) / len(counts),
            "min": min(counts), "max": max(counts)}


def gui_rect(root) -> tuple:
    """The Tk window's screen rectangle including its title bar.

    ``winfo_rootx`` names the client area, so a grab built from it clips the
    frame and the window looks headless in the screenshot.  The window
    manager's own frame handle carries the outer rectangle.
    """
    try:
        hwnd = int(root.wm_frame(), 16)
        rect = wintypes.RECT()
        if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        pass
    x, y = root.winfo_rootx(), root.winfo_rooty()
    return (x, y, x + root.winfo_width(), y + root.winfo_height())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--viewer-seconds", type=float, default=66.0,
                    help="how long GUI + viewer + cycle run together")
    ap.add_argument("--live-seconds", type=float, default=10.0,
                    help="how long the strip is held on the dead live stream")
    ap.add_argument("--viewer-window-timeout", type=float, default=60.0)
    ap.add_argument("--phase", choices=["full", "profile"], default="full",
                    help="full = the acceptance run; profile = attribute the "
                         "cycle-rate loss to the window's periodic jobs")
    ap.add_argument("--profile-seconds", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(RESULTS / (f"integrated_gui_2026-08-20.json"
                                  if args.phase == "full"
                                  else "integrated_gui_profile_2026-08-20.json"))

    verdict: dict = {"result": "FAILED: did not finish"}
    try:
        (run if args.phase == "full" else profile)(args, verdict)
    except BaseException as exc:
        import traceback
        verdict["result"] = f"FAILED: {type(exc).__name__}: {exc}"
        verdict["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2, default=str))
        print(f"\nrecord -> {out}")
        failed = [c["name"] for c in verdict.get("checks", []) if not c["pass"]]
        if failed:
            print("failed checks: " + "; ".join(failed))
        print(f"RESULT: {verdict['result']}  "
              f"({verdict.get('passed', 0)}/{verdict.get('total', 0)} checks)")
        sys.stdout.flush()
        sys.stderr.flush()
    # Non-daemon SDK threads from the live-mocap trial can outlive main and
    # hold the interpreter open (measured on this bench: EXIT=124 at a 15 s
    # timeout).  The record is already on disk, so leave without waiting.
    os._exit(0 if verdict.get("result") == "PASSED" else 1)


if __name__ == "__main__":
    main()
