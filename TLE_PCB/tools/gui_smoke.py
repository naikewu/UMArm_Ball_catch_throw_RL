"""Drive both GUIs through a real session and screenshot the result.

There is no way to prove a Tk window works without opening one, so this opens
each app for real, calls the same handlers the buttons call, waits for the
worker threads to finish, and saves a PNG. Run it with a display attached.

    python TLE_PCB/tools/gui_smoke.py [outdir]
"""
from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import ImageGrab  # noqa: E402

import VEMA_TLE_controller as controller_app  # noqa: E402
import VEMA_TLE_flash as flash_app  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "figures"


def pump(root: tk.Tk, seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.02)


def shoot(root: tk.Tk, path: Path) -> None:
    root.update()
    time.sleep(0.3)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    box = (x, y, x + root.winfo_width(), y + root.winfo_height())
    ImageGrab.grab(bbox=box, all_screens=True).save(path)
    print(f"  saved {path}")


def smoke_flash() -> None:
    print("flash GUI")
    root = tk.Tk()
    app = flash_app.FlashApp(root)
    pump(root, 1.0)
    app.scan_bus()
    pump(root, 6.0)
    print("  rows:", {hex(b): r["version"].variant_name for b, r in app.rows.items()})
    # Read-only: what the ID row would offer. A smoke run must not actually
    # move a board, so the button behind these is left alone.
    print("  ID change: from", list(app.can_id_from["values"]),
          "-> first free", app.can_id_to.get() or "(none)")
    shoot(root, OUT / "gui_flash.png")
    root.destroy()


def smoke_controller() -> None:
    print("controller GUI")
    root = tk.Tk()
    app = controller_app.ControllerApp(root)
    pump(root, 1.0)
    app.toggle_connect()          # connects and scans
    pump(root, 5.0)
    print("  boards:", [hex(b) for b in app.selected])
    if app.selected:
        app.target_var.set(6.0)
        app._target_moved()
        app.toggle_cycle()
        pump(root, 4.0)

        # Then break the group apart again, through the channel bars' own drag
        # handler rather than the setter behind it -- what is being checked is
        # that a drag lands on one channel and only that one. A staircase, so
        # the screenshot shows eight different bars against eight different
        # lines instead of eight identical rows.
        for step, (base, bar) in enumerate(app.node_bars.items()):
            bar.update_idletasks()
            bar._drag(SimpleNamespace(x=bar._to_x(2.0 + 0.8 * step)))
        pump(root, 4.0)
        print("  per-channel targets:", {hex(b): round(v, 2) for b, v in app.targets.items()})

        stats = app.backend.stats
        print(f"  cycles={stats.cycles} jitter_p95={stats.jitter_ms_p95:.3f} ms "
              f"replies={stats.replies} misses={stats.misses}")
        shoot(root, OUT / "gui_controller.png")
        app.stop_all()
        app.toggle_cycle()
        pump(root, 1.0)
    app.disconnect()
    root.destroy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke_flash()
    smoke_controller()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
