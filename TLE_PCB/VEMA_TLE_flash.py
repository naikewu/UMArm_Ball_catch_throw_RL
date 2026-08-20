"""VEMA TLE flash -- get firmware onto boards, by wire or over the bus.

Two halves, because there are two situations.

The top half is USB. One board, cabled to this PC, flashed with esptool. This
is bring-up: a board with no firmware, a board whose both OTA slots are bad, or
a board that needs its first ID before it can join the bus at all.

The bottom half is CAN. Every board on the bus at once: one image is broadcast
and each selected board writes it, so updating eight costs what updating one
costs. Scan first to see what is out there -- the scan enumerates old 7 mm
boards as readily as TLE boards, and says which is which.

Both halves can change a board's ID, and the difference between them is who
else can be listening. Over USB the cabled board is the only thing that can
answer, so the change is between two parties. Over CAN it is not, and the
failure mode is two boards sharing one identifier -- after which neither can be
addressed apart from the other and only a USB cable gets them apart again. So
the CAN path probes the destination for a responder first and refuses to move a
board onto an ID that answers.

What neither half touches is the ID as a side effect of flashing. NVS sits
below the application in the partition layout and no image either path writes
reaches it.

    python TLE_PCB/VEMA_TLE_flash.py
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tlelib import canlink  # noqa: E402
from tlelib import ota as ota_mod  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib import usbflash  # noqa: E402
from tlelib.canlink import CanLink  # noqa: E402
from tlelib.ota import BroadcastOta, OtaCancelled, load_image  # noqa: E402
from tlelib.usblink import UsbLink  # noqa: E402

POLL_MS = 60
BITRATES = {"1 Mbit/s": 1_000_000, "500 kbit/s": 500_000}
# How long a board is given to come back after an ID change reboots it. The
# firmware acknowledges, waits 250 ms and restarts; this covers the restart
# itself and the CAN driver coming up on the other side of it.
REBOOT_SETTLE_S = 3.0


class FlashApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("VEMA TLE flash")
        root.minsize(880, 720)

        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.image_variants: frozenset[int] | None = None
        self.cancel = threading.Event()
        self.rows: dict[int, dict] = {}
        self.plan: usbflash.FlashPlan | None = None
        self.flasher: usbflash.UsbFlasher | None = None

        self._build()
        self._autoload_build()
        self.refresh_ports()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(POLL_MS, self._drain_events)

    # ---- layout --------------------------------------------------------
    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        self._build_usb(outer)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=10)
        self._build_can(outer)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=10)
        self._build_log(outer)

    def _build_usb(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="USB  -- one cabled board", padding=8)
        frame.pack(fill="x")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Port").pack(side="left")
        self.usb_port = ttk.Combobox(row, width=34, state="readonly")
        self.usb_port.pack(side="left", padx=6)
        ttk.Button(row, text="Refresh", command=self.refresh_ports).pack(side="left")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Build").pack(side="left")
        self.build_var = tk.StringVar(value="(none)")
        ttk.Label(row, textvariable=self.build_var, width=58, relief="sunken",
                  anchor="w").pack(side="left", padx=6)
        ttk.Button(row, text="Browse...", command=self.choose_build).pack(side="left")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        self.mode = tk.StringVar(value="app")
        ttk.Radiobutton(row, text="application only", value="app",
                        variable=self.mode).pack(side="left")
        ttk.Radiobutton(row, text="full (bootloader + table + otadata + app)", value="full",
                        variable=self.mode).pack(side="left", padx=(10, 0))
        self.usb_button = ttk.Button(row, text="Flash over USB", command=self.flash_usb)
        self.usb_button.pack(side="right")
        self.usb_progress = ttk.Progressbar(row, length=200, maximum=100)
        self.usb_progress.pack(side="right", padx=8)

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(6, 2))
        ttk.Label(row, text="Board ID").pack(side="left")
        self.id_var = tk.StringVar(value="0x101")
        ttk.Combobox(row, textvariable=self.id_var, width=8,
                     values=[f"0x{b:03X}" for b in P.ALL_IDS]).pack(side="left", padx=6)
        ttk.Button(row, text="Read from board", command=self.read_id).pack(side="left")
        ttk.Button(row, text="Set ID over USB", command=self.set_id).pack(side="left", padx=6)
        ttk.Label(row, text="0x101-0x108 are the TLE slots; 0x109-0x118 belong to the 7 mm boards",
                  foreground="#666").pack(side="left", padx=10)

    def _build_can(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="CAN  -- every selected board at once", padding=8)
        frame.pack(fill="both", expand=True)

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Adapter").pack(side="left")
        self.can_port = ttk.Combobox(row, width=34, state="readonly")
        self.can_port.pack(side="left", padx=6)
        ttk.Label(row, text="Bitrate").pack(side="left", padx=(10, 0))
        self.bitrate = ttk.Combobox(row, width=12, state="readonly", values=list(BITRATES))
        self.bitrate.set("1 Mbit/s")
        self.bitrate.pack(side="left", padx=6)
        self.scan_button = ttk.Button(row, text="Scan bus", command=self.scan_bus)
        self.scan_button.pack(side="left", padx=6)

        columns = ("id", "kind", "version", "state", "progress")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=9,
                                 selectmode="none")
        for name, title, width in (("id", "CAN ID", 90), ("kind", "Board", 100),
                                   ("version", "Firmware", 300), ("state", "State", 110),
                                   ("progress", "Progress", 110)):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.bind("<Button-1>", self._toggle_row)

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Button(row, text="Select all", command=lambda: self.select_all(True)).pack(side="left")
        ttk.Button(row, text="Select none", command=lambda: self.select_all(False)).pack(side="left", padx=6)
        self.ota_button = ttk.Button(row, text="Broadcast to selected", command=self.broadcast)
        self.ota_button.pack(side="right")
        self.cancel_button = ttk.Button(row, text="Cancel", command=self.cancel_operation,
                                        state="disabled")
        self.cancel_button.pack(side="right", padx=6)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=8)

        # One board at a time, and by ID rather than by the row selection above:
        # the checkboxes mean "include in the broadcast", and an ID change that
        # silently followed them could move eight boards onto one identifier.
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="Change ID   from").pack(side="left")
        self.can_id_from = ttk.Combobox(row, width=8)
        self.can_id_from.pack(side="left", padx=6)
        ttk.Label(row, text="to").pack(side="left")
        self.can_id_to = ttk.Combobox(row, width=8,
                                      values=[f"0x{b:03X}" for b in P.ALL_IDS])
        self.can_id_to.pack(side="left", padx=6)
        self.can_id_button = ttk.Button(row, text="Set ID over CAN", command=self.set_id_can)
        self.can_id_button.pack(side="left", padx=6)
        ttk.Label(row, text="the destination is probed first; a board is never moved onto "
                            "an ID that already answers",
                  foreground="#666").pack(side="left", padx=10)

    def _build_log(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Log", padding=6)
        frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(frame, height=11, wrap="none", font=("Consolas", 9))
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=bar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

    # ---- plumbing ------------------------------------------------------
    def log(self, message: str) -> None:
        self.events.put(("log", message))

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log_text.insert("end", payload + "\n")
                self.log_text.see("end")
            elif kind == "usb_progress":
                self.usb_progress["value"] = payload
            elif kind == "board_id":
                self.id_var.set(payload)
            elif kind == "nodes":
                self._render_nodes(payload)
            elif kind == "node_state":
                base, state, fraction = payload
                self._set_node_state(base, state, fraction)
            elif kind == "busy":
                self._set_busy(*payload) if isinstance(payload, tuple) else self._set_busy(payload)
        self.root.after(POLL_MS, self._drain_events)

    def _set_busy(self, busy: bool, cancellable: bool = False) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.usb_button, self.scan_button, self.ota_button,
                       self.can_id_button):
            widget.configure(state=state)
        # Only offer Cancel for work that can actually be stopped, rather than
        # a button that logs "cancel requested" and does nothing.
        self.cancel_button.configure(state="normal" if (busy and cancellable) else "disabled")

    def _run(self, fn, cancellable: bool = False) -> None:
        if self.busy:
            self.log("busy; wait for the current operation to finish")
            return
        self.cancel.clear()
        # Synchronous, not via the queue: _run is only ever called from a button
        # command, so it is already on the Tk thread, and publishing the busy
        # state asynchronously left a 60 ms window in which a second click
        # started a second worker on the same port.
        self._set_busy(True, cancellable)

        def worker():
            try:
                fn()
            except OtaCancelled:
                self.log("cancelled")
            except Exception as exc:                       # surfaced, never swallowed
                self.log(f"[ERROR] {type(exc).__name__}: {exc}")
            finally:
                self.events.put(("busy", (False, False)))

        threading.Thread(target=worker, daemon=True).start()

    # ---- USB half ------------------------------------------------------
    def refresh_ports(self) -> None:
        ports = usbflash.list_ports()
        self.usb_port["values"] = [label for _, label in ports]
        self.can_port["values"] = [label for _, label in ports]
        for device, label in ports:
            if "ESP32-S3" in label and not self.usb_port.get():
                self.usb_port.set(label)
            if device == canlink.DEFAULT_PORT and not self.can_port.get():
                self.can_port.set(label)
        if ports and not self.usb_port.get():
            self.usb_port.set(ports[0][1])
        if ports and not self.can_port.get():
            self.can_port.set(ports[0][1])

    @staticmethod
    def _device(label: str) -> str:
        return label.split(" ")[0] if label else ""

    def _autoload_build(self) -> None:
        found = usbflash.find_build()
        if found is not None:
            self._load_plan(found)
        else:
            self.log("no build found; use Browse to point at a firmware build directory")

    def _load_plan(self, build_dir: Path) -> None:
        try:
            self.plan = usbflash.read_plan(build_dir)
        except Exception as exc:
            self.log(f"[ERROR] {exc}")
            return
        self.image_variants = None
        project = ""
        if self.plan.app_file is not None and self.plan.app_file.is_file():
            image = self.plan.app_file.read_bytes()
            project = ota_mod.image_project_name(image)
            self.image_variants = ota_mod.image_variants(image)
        kind = ("/".join(sorted(P.VARIANT_NAMES.get(v, str(v)) for v in self.image_variants))
                if self.image_variants else "unrecognised")
        self.build_var.set(f"{build_dir}   ({project or '?'}, {kind}, "
                           f"{self.plan.app_size} bytes)")
        self.log(f"build loaded: {build_dir} -- project '{project}', board type {kind}")
        if not self.image_variants:
            self.log("[WARN] this image does not name a board type it belongs to; "
                     "broadcast update is disabled for it")

    def choose_build(self) -> None:
        chosen = filedialog.askdirectory(title="Firmware build directory (containing flasher_args.json)")
        if chosen:
            self._load_plan(Path(chosen))

    def flash_usb(self) -> None:
        port = self._device(self.usb_port.get())
        if not port or self.plan is None:
            self.log("pick a port and a build first")
            return
        mode = self.mode.get()

        def job():
            self.log(f"flashing {port} ({mode})")
            self.events.put(("usb_progress", 0))
            self.flasher = usbflash.UsbFlasher(
                on_line=lambda line: self.log("  " + line),
                on_progress=lambda pct: self.events.put(("usb_progress", pct)))
            try:
                code = self.flasher.flash(port, self.plan, mode)
            finally:
                self.flasher = None
            self.log("flash finished" if code == 0 else f"[ERROR] esptool exited {code}")
            if code == 0:
                self.log("the board ID is untouched: NVS sits below the application "
                         "and nothing written here reaches it")

        self._run(job, cancellable=True)

    def read_id(self) -> None:
        port = self._device(self.usb_port.get())
        if not port:
            self.log("pick a USB port first")
            return

        def job():
            with UsbLink(port) as link:
                time.sleep(0.4)
                link.keepalive()
                link.collect(1.0)
                if link.base is None:
                    self.log("[ERROR] no telemetry; is this the board's own USB port?")
                    return
                # Tk is not thread-safe, and this runs on a worker: go through
                # the event queue like every other widget update here.
                self.events.put(("board_id", f"0x{link.base:03X}"))
                self.log(f"board reports base ID 0x{link.base:03X}")

        self._run(job)

    def set_id(self) -> None:
        port = self._device(self.usb_port.get())
        if not port:
            self.log("pick a USB port first")
            return
        try:
            wanted = int(self.id_var.get(), 0)
        except ValueError:
            self.log("[ERROR] board ID must be a number, e.g. 0x101")
            return
        if not (P.ACTUATOR_FIRST <= wanted <= P.ACTUATOR_LAST):
            self.log(f"[ERROR] 0x{wanted:03X} is outside the actuator range "
                     f"0x{P.ACTUATOR_FIRST:03X}-0x{P.ACTUATOR_LAST:03X}; a board there gets "
                     f"no runtime-table slot and would never see a broadcast target")
            return

        def job():
            with UsbLink(port) as link:
                time.sleep(0.4)
                link.keepalive()
                link.collect(0.6)
                current = link.base
                self.log(f"current ID 0x{current:03X}" if current else "current ID unknown")
                if current == wanted:
                    self.log("already set")
                    return
                self.log(f"setting ID to 0x{wanted:03X}; the board reboots to apply it")
                link.set_board_id(wanted, current)
                time.sleep(0.3)
                link.keepalive()
                link.collect(1.0)
                if link.base == wanted:
                    self.log(f"board now answers as 0x{wanted:03X} "
                             f"(table group 0x{P.ids(wanted).table_id:03X})")
                else:
                    self.log("[ERROR] board did not come back with the new ID")

        self._run(job)

    # ---- CAN half ------------------------------------------------------
    def scan_bus(self) -> None:
        port = self._device(self.can_port.get())
        if not port:
            self.log("pick a CAN adapter first")
            return
        bitrate = BITRATES[self.bitrate.get()]

        def job():
            self.log(f"scanning {port} at {bitrate // 1000} kbit/s")
            with CanLink(port, bitrate) as link:
                found = link.scan()
            self.events.put(("nodes", found))
            if found:
                self.log(f"{len(found)} board(s): " + ", ".join(
                    f"0x{b:03X} {v.variant_name}" for b, v in found.items()))
            else:
                self.log("no boards answered; check wiring, termination and bitrate")

        self._run(job, cancellable=True)

    def _render_nodes(self, found: dict) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        for base, version in found.items():
            # Pre-select only what the loaded image is actually for, so the
            # default selection can never be a cross-flash.
            selected = bool(self.image_variants) and version.variant in self.image_variants
            self.rows[base] = {"selected": selected, "version": version}
            self.tree.insert("", "end", iid=str(base),
                             values=(f"{'[x]' if selected else '[ ]'} 0x{base:03X}",
                                     version.variant_name, version.version, "idle", ""))

        present = [f"0x{base:03X}" for base in found]
        self.can_id_from["values"] = present
        if self.can_id_from.get() not in present:
            self.can_id_from.set(present[0] if present else "")
        # Offer the free IDs first, but keep every ID listed: an operator
        # consolidating a bus swaps two boards through a spare slot, and the
        # second half of that is a move onto an ID that was occupied a moment
        # ago. The probe at the time of the move is what decides, not this list.
        free = [f"0x{b:03X}" for b in P.ALL_IDS if b not in found]
        self.can_id_to["values"] = free + [f"0x{b:03X}" for b in found]
        if free and self.can_id_to.get() not in free:
            self.can_id_to.set(free[0])

    def _toggle_row(self, event) -> None:
        if self.busy:
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return
        base = int(item)
        row = self.rows.get(base)
        if row is None:
            return
        row["selected"] = not row["selected"]
        mark = "[x]" if row["selected"] else "[ ]"
        self.tree.set(item, "id", f"{mark} 0x{base:03X}")

    def select_all(self, selected: bool) -> None:
        if self.busy:
            return
        for base, row in self.rows.items():
            row["selected"] = selected
            self.tree.set(str(base), "id", f"{'[x]' if selected else '[ ]'} 0x{base:03X}")

    def _set_node_state(self, base: int, state: str, fraction: float) -> None:
        item = str(base)
        if item in self.tree.get_children():
            self.tree.set(item, "state", state)
            self.tree.set(item, "progress", f"{fraction * 100:.0f} %")

    def set_id_can(self) -> None:
        """Move one board to a new ID over the bus.

        The USB path can afford to be trusting because the cabled board is the
        only thing that can answer. This one cannot: the command is unicast to
        base + 0x100, and if a second board is already living at the
        destination the two become indistinguishable the moment this one
        reboots -- same base, same control ID, same status ID, both answering
        every scan. Nothing on the bus separates them again. So the destination
        is probed before the command goes out, and the result is checked from
        both ends afterwards: the new ID has to answer and the old one has to
        have gone quiet.
        """
        port = self._device(self.can_port.get())
        if not port:
            self.log("pick a CAN adapter first")
            return
        try:
            current = int(self.can_id_from.get(), 0)
            wanted = int(self.can_id_to.get(), 0)
        except ValueError:
            self.log("[ERROR] both IDs must be numbers, e.g. 0x101")
            return
        for what, value in (("current", current), ("new", wanted)):
            if not (P.ACTUATOR_FIRST <= value <= P.ACTUATOR_LAST):
                self.log(f"[ERROR] {what} ID 0x{value:03X} is outside the actuator range "
                         f"0x{P.ACTUATOR_FIRST:03X}-0x{P.ACTUATOR_LAST:03X}. A scan only "
                         f"looks there, so a board moved outside it has to be brought "
                         f"back over USB")
                return
        if current == wanted:
            self.log(f"0x{current:03X} already has that ID")
            return
        bitrate = BITRATES[self.bitrate.get()]

        def job():
            with CanLink(port, bitrate) as link:
                before = link.scan([current, wanted])
                if wanted in before:
                    self.log(f"[ERROR] 0x{wanted:03X} is taken -- a "
                             f"{before[wanted].variant_name} board answers there. Move "
                             f"that one out of the way first, or pick a free ID")
                    return
                if current not in before:
                    self.log(f"[ERROR] nothing answers at 0x{current:03X}; scan the bus "
                             f"and use an ID that is on it")
                    return
                self.log(f"0x{current:03X} ({before[current].variant_name}) -> "
                         f"0x{wanted:03X}: the board saves the ID and reboots to apply it")
                if link.set_node_id(current, wanted):
                    self.log("acknowledged on the new status ID")
                else:
                    self.log("[WARN] no acknowledgement; checking the bus anyway")

                time.sleep(REBOOT_SETTLE_S)
                after = link.scan([current, wanted])
                if wanted in after:
                    self.log(f"board now answers as 0x{wanted:03X} "
                             f"(table group 0x{P.ids(wanted).table_id:03X})")
                else:
                    self.log(f"[ERROR] nothing answers at 0x{wanted:03X} after the reboot. "
                             f"Give it a moment and scan again before doing anything else")
                if current in after:
                    self.log(f"[WARN] 0x{current:03X} still answers -- either the change "
                             f"did not take, or a second board was already sitting there")
                self.events.put(("nodes", link.scan()))

        self._run(job)

    def cancel_operation(self) -> None:
        """Stop whatever is running: an esptool subprocess or an update."""
        self.cancel.set()
        if self.flasher is not None:
            self.flasher.cancel()
        self.log("cancel requested")

    def on_close(self) -> None:
        """Do not leave an esptool subprocess behind holding the port."""
        if self.busy:
            self.cancel_operation()
        self.root.destroy()

    def broadcast(self) -> None:
        port = self._device(self.can_port.get())
        selected = [b for b, row in self.rows.items() if row["selected"]]
        if not port or not selected:
            self.log("scan the bus and select at least one board")
            return
        if self.plan is None or self.plan.app_file is None:
            self.log("pick a build first")
            return
        bitrate = BITRATES[self.bitrate.get()]
        app_file = self.plan.app_file

        # An update is irreversible from here: each board writes the image,
        # marks it bootable and reboots into it, with no rollback configured.
        # So the check is image-versus-board, from what the scan actually read
        # back, never from the ID range -- a TLE image accepted by the sixteen
        # 7 mm boards would take every one of them off the bus until it was
        # physically reflashed over USB.
        if not self.image_variants:
            self.log("[ERROR] cannot tell which board type this image is for, so it "
                     "will not be broadcast. Load a build whose app descriptor names "
                     "a known project")
            return
        kinds = "/".join(sorted(P.VARIANT_NAMES.get(v, str(v)) for v in self.image_variants))
        wrong = [b for b in selected
                 if self.rows[b]["version"].variant not in self.image_variants]
        if wrong:
            self.log(f"[ERROR] this is a {kinds} image; "
                     + ", ".join(f"0x{b:03X} reports {self.rows[b]['version'].variant_name}"
                                 for b in wrong)
                     + ". Load the matching build, or deselect those boards")
            return

        def job():
            image = load_image(app_file)
            self.log(f"broadcasting {app_file.name} ({len(image)} bytes) to "
                     f"{len(selected)} board(s)")
            with CanLink(port, bitrate) as link:
                ota = BroadcastOta(
                    link,
                    log=self.log,
                    progress=lambda base, state, frac: self.events.put(
                        ("node_state", (base, state, frac))),
                    cancel=self.cancel)
                result = ota.upload(image, selected)
            done = [b for b, ok in result.items() if ok]
            failed = [b for b, ok in result.items() if not ok]
            self.log(f"updated {len(done)}/{len(result)}" +
                     (f"; failed: {', '.join(f'0x{b:03X}' for b in failed)}" if failed else ""))
            self.log("board IDs are unchanged: an update writes the spare application slot "
                     "and never the NVS partition that holds them")

        self._run(job)


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    FlashApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
