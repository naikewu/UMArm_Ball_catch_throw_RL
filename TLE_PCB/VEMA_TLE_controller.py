"""VEMA TLE controller -- drive every board on the bus from one window.

Frontend only. Every byte that reaches a board goes through ``tlelib.backend``,
which owns the adapter, the node registry and the 150 Hz cycle on its own
thread; this file owns widgets, and talks to the backend through method calls
and snapshots. The backend runs just as well from a script with no display (see
``tools/tle_bench.py``).

Connect, Scan bus and Disconnect run on the Tk thread and do block it -- for a
second or two while the adapter is opened and the bus enumerated. Everything
after that is non-blocking: the cycle runs on the backend's own thread and this
file only reads snapshots.

What the two halves each know:

    backend    which boards exist, what they were told, what they answered,
               and when -- targets go out in one packed table frame per three
               actuators, followed by the sync edge that makes them live
    frontend   what the operator wants, and what the last few seconds looked
               like

Every channel has its own bar, and the group slider under them moves whichever
are selected. The group move is the normal case for a manifold -- the eight
channels are one actuator group -- but it is a shortcut for setting them all,
not the only way to set any of them.

A channel's bar carries both of its numbers on one axis: the target is a line
and the measured pressure is a bar growing towards it. That is the arrangement
because the question actually being asked of the row is never what either
number is, it is whether the valve has got there yet, and that reads off the
gap without reading anything. The signed number beside it is that gap.

    python TLE_PCB/VEMA_TLE_controller.py
"""
from __future__ import annotations

import queue
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                        # noqa: E402
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure                     # noqa: E402

from tlelib import canlink                               # noqa: E402
from tlelib import native as N                           # noqa: E402
from tlelib import proto as P                            # noqa: E402
from tlelib.backend import Backend                       # noqa: E402
from tlelib.canlink import list_serial_ports             # noqa: E402

REFRESH_MS = 100
# A redraw holds the interpreter lock for tens of milliseconds, and the cycle
# thread is in the same process, so this is a trade between how live the plot
# looks and how often a sync edge goes out late. A late edge stretches one
# cycle; it does not desynchronise the boards, since they all latch on whatever
# edge arrives -- but it is still worth not doing four times a second.
PLOT_MS = 500   # was 250; at 24 boards even the set_data redraw costs the
                # 150 Hz cycle ~10 Hz at 4 Hz refresh (measured 2026-08-20,
                # hw_tests/report_integrated §11) — 2 Hz halves that for a
                # monitoring plot nobody reads faster
PLOT_WINDOW_S = 30.0
BITRATES = {"1 Mbit/s": 1_000_000, "500 kbit/s": 500_000}
TARGET_MAX_PSI = 40.0
SERIES_COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                  "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]

# A drag is quantised to this, so a commanded target is a number an operator
# can read back and repeat. It is finer than one pixel of the bar and coarser
# than the 0.016 psi the 12-bit wire field resolves, so nothing is lost.
TARGET_STEP_PSI = 0.05
WHEEL_STEP_PSI = 0.25
# Purely a readout convenience: the error number goes green inside this band so
# a settled channel can be told from a moving one across eight rows at a
# glance. Nothing in the control path knows about it.
SETTLED_PSI = 0.5
SETTLED_COLOUR = "#0a7a0a"

BAR_WIDTH = 210
BAR_HEIGHT = 22
BAR_PAD = 7          # room for the target line and its grip at either end
BAR_TOP = 5          # the grip sits above this; the track starts at it
TRACK_FILL = "#efefef"
TRACK_EDGE = "#b6b6b6"
MARK_COLOUR = "#141414"

# (heading, node_widgets key, width in characters, anchor). A width of 0 means
# natural size, which is what the bar column wants -- it is a canvas and sets
# its own width.
NODE_COLUMNS = (
    ("", None, 2, "w"),
    ("ID", None, 6, "w"),
    ("Type", None, 8, "w"),
    ("Target line / pressure bar", None, 0, "w"),
    ("Tgt", "target", 6, "e"),
    ("psi", "pressure", 7, "e"),
    ("Err", "error", 7, "e"),
    ("Lat", "latency", 5, "e"),
    ("Miss", "miss", 5, "e"),
    ("Flags", "flags", 6, "w"),
)


def _lighten(colour: str, amount: float) -> str:
    """Blend a #rrggbb colour towards white. Tk canvases have no alpha."""
    channels = (int(colour[index:index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{int(round(c + (255 - c) * amount)):02x}" for c in channels)


class ChannelBar(tk.Canvas):
    """One channel's target and pressure on a single axis.

    ttk.Scale cannot draw anything behind its trough, and a slider next to a
    number would put the two halves of one comparison in two places. Here the
    line is what was asked for, the bar is what came back, and dragging
    anywhere on the track moves the line.

    The bar's colour is the board's colour in the plot, so a row and a trace
    can be matched without reading either label.
    """

    def __init__(self, master, maximum: float, on_change, colour: str,
                 background: str, width: int = BAR_WIDTH, height: int = BAR_HEIGHT):
        super().__init__(master, width=width, height=height, bd=0,
                         highlightthickness=0, bg=background,
                         cursor="sb_h_double_arrow")
        self.maximum = maximum
        self.target = 0.0
        self.measured: float | None = None
        self._on_change = on_change
        self._measured_x: int | None = None

        self._track = self.create_rectangle(0, 0, 0, 0, fill=TRACK_FILL, outline=TRACK_EDGE)
        self._bar = self.create_rectangle(0, 0, 0, 0, fill=_lighten(colour, 0.45),
                                          outline=colour, state="hidden")
        self._ticks = [self.create_line(0, 0, 0, 0, fill=TRACK_EDGE, dash=(2, 3))
                       for _ in range(3)]
        self._line = self.create_line(0, 0, 0, 0, fill=MARK_COLOUR, width=2)
        self._grip = self.create_polygon(0, 0, 0, 0, 0, 0, fill=MARK_COLOUR, outline="")

        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<MouseWheel>", self._wheel)
        self._draw()

    # ---- geometry ------------------------------------------------------
    def _size(self) -> tuple[int, int]:
        """Configured size until the widget is mapped; winfo_* reports 1 first."""
        width = self.winfo_width()
        height = self.winfo_height()
        return (width if width > 1 else int(self["width"]),
                height if height > 1 else int(self["height"]))

    def _to_x(self, psi: float) -> float:
        width, _ = self._size()
        span = max(1, width - 2 * BAR_PAD)
        return BAR_PAD + span * min(max(psi, 0.0), self.maximum) / self.maximum

    def _to_psi(self, x: float) -> float:
        width, _ = self._size()
        span = max(1, width - 2 * BAR_PAD)
        return min(max((x - BAR_PAD) / span, 0.0), 1.0) * self.maximum

    # ---- drawing -------------------------------------------------------
    def _draw(self) -> None:
        width, height = self._size()
        right = width - BAR_PAD
        self.coords(self._track, BAR_PAD, BAR_TOP, right, height - 1)
        for index, tick in enumerate(self._ticks, start=1):
            x = BAR_PAD + (right - BAR_PAD) * index / (len(self._ticks) + 1)
            self.coords(tick, x, BAR_TOP + 1, x, height - 2)

        self._measured_x = None if self.measured is None else round(self._to_x(self.measured))
        if self._measured_x is None or self._measured_x - BAR_PAD < 1:
            self.itemconfigure(self._bar, state="hidden")
        else:
            self.itemconfigure(self._bar, state="normal")
            self.coords(self._bar, BAR_PAD, BAR_TOP + 1, self._measured_x, height - 2)

        x = self._to_x(self.target)
        self.coords(self._line, x, BAR_TOP - 1, x, height - 1)
        self.coords(self._grip, x - 4, 0, x + 4, 0, x, BAR_TOP + 1)

    # ---- input ---------------------------------------------------------
    def _drag(self, event) -> None:
        self._commit(self._to_psi(event.x))

    def _wheel(self, event) -> None:
        """One notch is finer than a pixel of track, which a drag is not."""
        self._commit(self.target + (WHEEL_STEP_PSI if event.delta > 0 else -WHEEL_STEP_PSI))

    def _commit(self, psi: float) -> None:
        psi = min(max(round(psi / TARGET_STEP_PSI) * TARGET_STEP_PSI, 0.0), self.maximum)
        if psi == self.target:
            return
        self.target = psi
        self._draw()
        self._on_change(psi)

    # ---- from the refresh ----------------------------------------------
    def set_target(self, psi: float) -> None:
        """Move the line without calling back -- the group slider and STOP."""
        psi = min(max(psi, 0.0), self.maximum)
        if psi != self.target:
            self.target = psi
            self._draw()

    def set_measured(self, psi: float | None) -> None:
        """Update the bar, comparing in pixels rather than psi.

        This is called for every board ten times a second, and most of those
        readings do not move the bar at all. A redraw that changes nothing
        still holds the interpreter lock, and the cycle thread is in the same
        process waiting to put a sync edge out on time.
        """
        x = None if psi is None else round(self._to_x(psi))
        self.measured = psi
        if x != self._measured_x:
            self._draw()


class ControllerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("VEMA TLE controller")
        root.minsize(1240, 800)

        self.backend: Backend | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self.selected: dict[int, tk.BooleanVar] = {}
        self.targets: dict[int, float] = {}
        # Fixed per board at scan time rather than per redraw, so a board's bar
        # and its trace in the plot are the same colour whatever else is
        # selected or has history.
        self.node_colour: dict[int, str] = {}
        # (monotonic time, cycles) samples, one per refresh, so the status line
        # can report the rate the cycle is ACHIEVING rather than the rate it was
        # asked for. Two seconds of them at REFRESH_MS: long enough that the
        # quotient is not dominated by which side of a 6.7 ms period each sample
        # landed on, short enough to follow a real change.
        self._rate_marks: deque[tuple[float, int]] = deque(
            maxlen=max(2, int(2000 / REFRESH_MS)))

        self._build()
        self.refresh_ports()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(REFRESH_MS, self._refresh)
        root.after(PLOT_MS, self._replot)

    # ---- layout --------------------------------------------------------
    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        self._build_link(outer)
        panes = ttk.Frame(outer)
        panes.pack(fill="both", expand=True, pady=6)
        self._build_nodes(panes)
        self._build_plot(panes)
        self._build_status(outer)

    def _build_link(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Bus", padding=8)
        frame.pack(fill="x")

        ttk.Label(frame, text="Adapter").pack(side="left")
        self.port = ttk.Combobox(frame, width=32, state="readonly")
        self.port.pack(side="left", padx=6)
        ttk.Button(frame, text="Refresh", command=self.refresh_ports).pack(side="left")

        ttk.Label(frame, text="Bitrate").pack(side="left", padx=(12, 0))
        self.bitrate = ttk.Combobox(frame, width=11, state="readonly", values=list(BITRATES))
        self.bitrate.set("1 Mbit/s")
        self.bitrate.pack(side="left", padx=6)

        self.connect_button = ttk.Button(frame, text="Connect", command=self.toggle_connect)
        self.connect_button.pack(side="left", padx=(12, 0))
        self.scan_button = ttk.Button(frame, text="Scan bus", command=self.scan, state="disabled")
        self.scan_button.pack(side="left", padx=6)
        self.cycle_button = ttk.Button(frame, text="Start 150 Hz cycle",
                                       command=self.toggle_cycle, state="disabled")
        self.cycle_button.pack(side="left", padx=6)

        stop = ttk.Button(frame, text="STOP ALL", command=self.stop_all)
        stop.pack(side="right")

    def _build_nodes(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Boards", padding=8)
        frame.pack(side="left", fill="both", expand=False)

        # Header and rows share one grid. The bar is a fixed-pixel canvas and
        # the labels either side of it are character-width, and no amount of
        # packing keeps two of those in step across two frames.
        self.node_frame = ttk.Frame(frame)
        self.node_frame.pack(fill="both", expand=True, pady=(0, 4))
        try:
            self._bar_background = ttk.Style().lookup("TFrame", "background") or self.root.cget("bg")
        except tk.TclError:
            self._bar_background = self.root.cget("bg")
        for column, (heading, _key, width, anchor) in enumerate(NODE_COLUMNS):
            ttk.Label(self.node_frame, text=heading, width=width, anchor=anchor,
                      font=("TkDefaultFont", 8, "bold")).grid(row=0, column=column,
                                                              sticky="ew", padx=2)
        self.node_widgets: dict[int, dict] = {}
        self.node_bars: dict[int, ChannelBar] = {}
        self.node_rows: list[tk.Widget] = []

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        group = ttk.LabelFrame(frame, text="All selected boards", padding=6)
        group.pack(fill="x")

        row = ttk.Frame(group)
        row.pack(fill="x")
        ttk.Label(row, text="Target").pack(side="left")
        self.target_var = tk.DoubleVar(value=0.0)
        self.target_scale = ttk.Scale(row, from_=0.0, to=TARGET_MAX_PSI, orient="horizontal",
                                      variable=self.target_var, command=self._target_moved,
                                      length=200)
        self.target_scale.pack(side="left", padx=6)
        self.target_label = ttk.Label(row, text="0.00 psi", width=10)
        self.target_label.pack(side="left")
        ttk.Label(row, text="moves every selected channel; drag a channel's own bar to "
                            "move it alone", foreground="#666").pack(side="left", padx=6)

        row = ttk.Frame(group)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Select all", command=lambda: self.select_all(True)).pack(side="left")
        ttk.Button(row, text="None", command=lambda: self.select_all(False)).pack(side="left", padx=4)
        ttk.Button(row, text="Enable", command=lambda: self.enable_selected(True)).pack(side="left", padx=(10, 0))
        ttk.Button(row, text="Disable", command=lambda: self.enable_selected(False)).pack(side="left", padx=4)

        row = ttk.Frame(group)
        row.pack(fill="x")
        ttk.Button(row, text="Apply bench tuning to TLE boards",
                   command=self.apply_tuning).pack(side="left")

    def _build_plot(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Pressure", padding=6)
        frame.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.figure = Figure(figsize=(7.0, 4.6), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("seconds")
        self.axes.set_ylabel("psi")
        self.axes.grid(alpha=0.3)
        self.figure.tight_layout()
        # Persistent Line2D pairs per board, updated with set_data. Measured on
        # the real 24-board bus (2026-08-20, hw_tests/report_integrated): a
        # clear()-and-replot cycle at 4 Hz cost the 150 Hz sync master 13.5 Hz;
        # reusing the artists is where that goes.
        self._plot_lines: dict[int, tuple] = {}
        self._plot_axes_ready = False
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.series: dict[int, tuple] = {}

    def _build_status(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Cycle", padding=6)
        frame.pack(fill="both", expand=False)
        self.status_var = tk.StringVar(value="not connected")
        ttk.Label(frame, textvariable=self.status_var, font=("Consolas", 9)).pack(anchor="w")

        self.log_text = tk.Text(frame, height=7, wrap="none", font=("Consolas", 9))
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=bar.set)
        self.log_text.pack(side="left", fill="both", expand=True, pady=(4, 0))
        bar.pack(side="right", fill="y")

    # ---- link ----------------------------------------------------------
    def refresh_ports(self) -> None:
        ports = list_serial_ports()
        self.port["values"] = [label for _, label in ports]
        for device, label in ports:
            if device == canlink.DEFAULT_PORT:
                self.port.set(label)
        if ports and not self.port.get():
            self.port.set(ports[0][1])

    def log(self, message: str) -> None:
        """Safe to call from any thread.

        The backend logs from its cycle thread (a failed transmit, an extended
        frame that would not go out), and Tk is not thread-safe, so nothing
        touches the widget except the periodic refresh on the main thread.
        """
        self._log_queue.put(message)

    def _drain_log(self) -> None:
        drained = False
        while True:
            try:
                self.log_text.insert("end", self._log_queue.get_nowait() + "\n")
                drained = True
            except queue.Empty:
                break
        if drained:
            self.log_text.see("end")

    def toggle_connect(self) -> None:
        if self.backend is not None:
            self.disconnect()
            return
        label = self.port.get()
        if not label:
            self.log("pick an adapter first")
            return
        port = label.split(" ")[0]
        try:
            self.backend = Backend(port, BITRATES[self.bitrate.get()], log=self.log)
            self.backend.open()
        except Exception as exc:
            self.log(f"[ERROR] {exc}")
            self.backend = None
            return
        self.connect_button.configure(text="Disconnect")
        for widget in (self.scan_button, self.cycle_button):
            widget.configure(state="normal")
        self.scan()

    def disconnect(self) -> None:
        if self.backend is None:
            return
        self.backend.close()
        self.backend = None
        self.connect_button.configure(text="Connect")
        for widget in (self.scan_button, self.cycle_button):
            widget.configure(state="disabled")
        self.cycle_button.configure(text="Start 150 Hz cycle")
        self.status_var.set("not connected")
        self._clear_readouts()

    def _clear_readouts(self) -> None:
        """Drop everything that was measured rather than commanded.

        _refresh stops running the moment the backend goes, so whatever was on
        screen at that instant would simply stay there -- a filled pressure bar
        on a bus nothing is talking to reads as a live measurement. The targets
        and their lines stay: those are still what the operator asked for.
        """
        for widgets in self.node_widgets.values():
            for key in ("pressure", "error", "latency", "miss", "flags"):
                widgets[key].configure(text="-", foreground="#000")
        for bar in self.node_bars.values():
            bar.set_measured(None)

    def scan(self) -> None:
        if self.backend is None:
            return
        if self.backend.running:
            self.log("stop the cycle before scanning")
            return
        nodes = self.backend.scan()
        self._render_nodes(nodes)
        self.backend.select([b for b, v in self.selected.items() if v.get()])

    def toggle_cycle(self) -> None:
        if self.backend is None:
            return
        if self.backend.running:
            self.backend.stop_cycle()
            self.cycle_button.configure(text="Start 150 Hz cycle")
            return
        chosen = [b for b, v in self.selected.items() if v.get()]
        if not chosen:
            self.log("select at least one board")
            return
        self.backend.select(chosen)
        self.backend.start_cycle()
        self.cycle_button.configure(text="Stop cycle")

    def stop_all(self) -> None:
        if self.backend is None:
            return
        self.backend.stop_all()
        self.target_var.set(0.0)
        self.target_label.configure(text="0.00 psi")
        # Every known board, not just the selected ones, and the bars with
        # them: a bar still showing 12 psi after a STOP would be reporting a
        # target no longer commanded anywhere.
        for base in self.targets:
            self._set_target(base, 0.0)

    # ---- nodes ---------------------------------------------------------
    def _render_nodes(self, nodes: dict) -> None:
        # Only what a previous render made: the header lives in the same grid
        # and is built once.
        for widget in self.node_rows:
            widget.destroy()
        self.node_rows.clear()
        self.node_widgets.clear()
        self.node_bars.clear()
        self.node_colour.clear()
        keep = dict(self.selected)
        self.selected.clear()

        index = 0
        for base, node in nodes.items():
            if not node.present:
                continue
            index += 1
            var = keep.get(base) or tk.BooleanVar(value=True)
            self.selected[base] = var
            self.targets.setdefault(base, 0.0)
            colour = SERIES_COLOURS[(index - 1) % len(SERIES_COLOURS)]
            self.node_colour[base] = colour

            check = ttk.Checkbutton(self.node_frame, variable=var,
                                    command=self._selection_changed)
            check.grid(row=index, column=0)
            ident = ttk.Label(self.node_frame, text=f"0x{base:03X}", width=6,
                              font=("Consolas", 8))
            ident.grid(row=index, column=1, sticky="w", padx=2)
            kind = ttk.Label(self.node_frame, text=node.kind, width=8, font=("Consolas", 8))
            kind.grid(row=index, column=2, sticky="w", padx=2)
            bar = ChannelBar(self.node_frame, TARGET_MAX_PSI,
                             on_change=lambda psi, b=base: self._channel_moved(b, psi),
                             colour=colour, background=self._bar_background)
            bar.set_target(self.targets[base])
            bar.grid(row=index, column=3, sticky="ew", padx=4, pady=1)
            self.node_bars[base] = bar
            self.node_rows.extend((check, ident, kind, bar))

            widgets = {}
            for column, (_heading, key, width, anchor) in enumerate(NODE_COLUMNS):
                if key is None:
                    continue
                label = ttk.Label(self.node_frame, text="-", width=width, anchor=anchor,
                                  font=("Consolas", 8))
                label.grid(row=index, column=column, sticky="ew", padx=2)
                widgets[key] = label
                self.node_rows.append(label)
            self.node_widgets[base] = widgets

        if not self.node_widgets:
            empty = ttk.Label(self.node_frame, text="no boards found")
            empty.grid(row=1, column=0, columnspan=len(NODE_COLUMNS), sticky="w", pady=4)
            self.node_rows.append(empty)

    def _selection_changed(self) -> None:
        if self.backend is not None:
            self.backend.select([b for b, v in self.selected.items() if v.get()])

    def select_all(self, on: bool) -> None:
        for var in self.selected.values():
            var.set(on)
        self._selection_changed()

    def enable_selected(self, on: bool) -> None:
        if self.backend is None:
            return
        for base, var in self.selected.items():
            if var.get():
                self.backend.set_enabled(base, on)

    def _target_moved(self, _value=None) -> None:
        """The group slider: every selected channel goes to one value."""
        psi = float(self.target_var.get())
        self.target_label.configure(text=f"{psi:.2f} psi")
        for base, var in self.selected.items():
            if var.get():
                self._set_target(base, psi)

    def _channel_moved(self, base: int, psi: float) -> None:
        """One channel's own bar. Deliberately does not touch the others."""
        self.targets[base] = psi
        if self.backend is not None:
            self.backend.set_target(base, psi)

    def _set_target(self, base: int, psi: float) -> None:
        """Command a channel and move its bar to match."""
        self.targets[base] = psi
        bar = self.node_bars.get(base)
        if bar is not None:
            bar.set_target(psi)
        if self.backend is not None:
            self.backend.set_target(base, psi)

    def apply_tuning(self) -> None:
        if self.backend is None:
            return
        snapshot = self.backend.snapshot_nodes()
        applied = 0
        for base, var in self.selected.items():
            node = snapshot.get(base)
            if var.get() and node is not None and node.is_tle:
                self.backend.apply_tuning(base)
                applied += 1
        self.log(f"tuning sent to {applied} TLE board(s)"
                 if applied else "no TLE boards selected (7 mm boards have no tuning surface here)")

    # ---- periodic ------------------------------------------------------
    def _refresh(self) -> None:
        self._drain_log()
        if self.backend is not None:
            snapshot = self.backend.snapshot_nodes()
            for base, widgets in self.node_widgets.items():
                node = snapshot.get(base)
                if node is None:
                    continue
                widgets["target"].configure(text=f"{node.target_psi:6.2f}")
                # Until a board has answered once, its pressure field is 0.0
                # because nothing has been written to it -- which is a reading
                # of zero psi to anyone looking at the row. Say so instead.
                heard = node.replies > 0
                error = node.target_psi - node.pressure_psi
                widgets["pressure"].configure(text=f"{node.pressure_psi:7.2f}" if heard else "-")
                widgets["error"].configure(
                    text=f"{error:+7.2f}" if heard else "-",
                    foreground=SETTLED_COLOUR if heard and abs(error) <= SETTLED_PSI else "#000")
                bar = self.node_bars.get(base)
                if bar is not None:
                    bar.set_measured(node.pressure_psi if heard else None)
                widgets["latency"].configure(text=f"{node.reply_latency_ms:4.1f}")
                total = node.replies + node.misses
                widgets["miss"].configure(text=f"{node.misses}" if total else "-")
                flags = "".join((
                    "E" if node.enabled else ".",
                    "O" if node.flags & P.STATUS_OTA_ACTIVE else ".",
                    "C" if node.flags & P.STATUS_COMMAND_SEEN else ".",
                    "!" if node.error else ".",
                ))
                widgets["flags"].configure(text=flags,
                                           foreground="#b00" if node.error else "#000")

            stats = self.backend.stats
            if self.backend.running:
                # MEASURED, not P.CYCLE_HZ. Printing the constant put "150 Hz"
                # on screen while the cycle thread was being held off the CPU by
                # this window's own redraw and the bus was actually being driven
                # at 132 Hz -- the one number on the line that was not a
                # measurement was the one an operator would trust first.
                self._rate_marks.append((time.monotonic(), stats.cycles))
                rate = self._measured_hz()
                total = stats.replies + stats.misses
                good = stats.replies / total * 100 if total else 0.0
                self.status_var.set(
                    (f"{rate:.1f} Hz" if rate is not None else "-- Hz")
                    + f" (asked {P.CYCLE_HZ:.0f})   cycles {stats.cycles}   "
                    f"jitter p95 {stats.jitter_ms_p95:.3f} ms / max {stats.jitter_ms_max:.3f} ms   "
                    f"late {stats.late_cycles}   replies {good:.2f} %   "
                    f"bus load ~{P.bus_load(len(self.backend.selected), 0) * 100:.1f} %")
                missing = self.backend.missing_nodes()
                if missing:
                    self.status_var.set(self.status_var.get() +
                                        "   SILENT: " + ", ".join(f"0x{b:03X}" for b in missing))
            else:
                self._rate_marks.clear()
                self.status_var.set("connected, cycle stopped")
        self.root.after(REFRESH_MS, self._refresh)

    def _measured_hz(self) -> float | None:
        """Cycles per second across the samples held, or None until there are two.

        The difference across the window rather than ``cycles / uptime``: the
        cumulative figure would keep reporting a healthy rate for a minute after
        the cycle started missing its period.
        """
        if len(self._rate_marks) < 2:
            return None
        (t0, c0), (t1, c1) = self._rate_marks[0], self._rate_marks[-1]
        return (c1 - c0) / (t1 - t0) if t1 > t0 else None

    def _replot(self) -> None:
        if self.backend is not None and self.node_widgets:
            histories = {}
            newest = 0.0
            for base, var in self.selected.items():
                if not var.get():
                    continue
                # Decimated at the source, under the backend's lock: one point
                # per pixel column is all the plot can show anyway, and copying
                # 9000 samples per board out of the lock cost the cycle thread
                # a measured 2.8 Hz on the 24-board bus.
                history = self.backend.history(base, max_points=900)
                if history:
                    histories[base] = history
                    newest = max(newest, history[-1][0])

            if not self._plot_axes_ready:
                self.axes.set_xlabel("seconds before now")
                self.axes.set_ylabel("psi")
                self.axes.set_xlim(-PLOT_WINDOW_S, 0.0)
                self.figure.tight_layout()
                self._plot_axes_ready = True

            legend_dirty = False
            for base in [b for b in self._plot_lines if b not in histories]:
                solid, dashed = self._plot_lines.pop(base)
                solid.remove()
                dashed.remove()
                legend_dirty = True
            for base, history in histories.items():
                times = [h[0] - newest for h in history]   # 0 is now: axis reads -30..0
                measured = [h[1] for h in history]
                target = [h[2] for h in history]
                lines = self._plot_lines.get(base)
                if lines is None:
                    colour = self.node_colour.get(base, SERIES_COLOURS[0])
                    solid, = self.axes.plot(times, measured, color=colour, lw=1.2,
                                            label=f"0x{base:03X}")
                    dashed, = self.axes.plot(times, target, color=colour, lw=0.8,
                                             ls="--", alpha=0.6)
                    self._plot_lines[base] = (solid, dashed)
                    legend_dirty = True
                else:
                    solid, dashed = lines
                    solid.set_data(times, measured)
                    dashed.set_data(times, target)
            if legend_dirty:
                legend = self.axes.get_legend()
                if legend is not None:
                    legend.remove()
                if histories:
                    self.axes.legend(loc="upper left", fontsize=8, ncol=4)
            if histories:
                # set_data does not autoscale; recompute y only, x is pinned.
                self.axes.relim(visible_only=True)
                self.axes.autoscale_view(scalex=False)
            self.canvas.draw_idle()
        self.root.after(PLOT_MS, self._replot)

    def on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    ControllerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
