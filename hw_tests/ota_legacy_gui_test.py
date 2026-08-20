"""Drive the real flash GUI to broadcast-OTA the legacy v0.2.1 image.

The user's acceptance question is "does the flash GUI update the OLD boards?",
so this script drives ``VEMA_TLE_flash.FlashApp`` itself — its scan handler,
its row selection, its cross-flash guard, its broadcast handler — with a real
Tk window pumped by ``root.update()``. The bytes on the wire are sent by the
GUI's own worker thread through ``tlelib.BroadcastOta``.

Safety properties, checked before any CMD_START leaves the host:
  * the image is ``firmware/images/legacy_7mm`` — byte-identical to what the
    sixteen boards already run, so even total success changes nothing;
  * the loaded plan must identify as project ``Valve_not_embedded_XL`` with
    variant set {0, 1}, or the script aborts;
  * the selection is asserted to contain ONLY legacy IDs (0x109–0x118); a TLE
    id in the selection aborts before the broadcast button is pressed.

Phases (run separately so each stays inside a comfortable timeout)::

    .venv\\Scripts\\python.exe hw_tests\\ota_legacy_gui_test.py --phase one
    .venv\\Scripts\\python.exe hw_tests\\ota_legacy_gui_test.py --phase fleet

``--phase one`` updates 0x109 alone; ``--phase fleet`` updates all sixteen.
Both rescan afterwards and verify every legacy board reports 0.2.1/7mm and
every TLE board still reports its pre-test version, untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "TLE_PCB"))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "vema_tle_flash", WS / "TLE_PCB" / "VEMA_TLE_flash.py")
flash_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flash_mod)

LEGACY_IDS = list(range(0x109, 0x119))
TLE_IDS = list(range(0x101, 0x109))
IMAGE_DIR = WS / "firmware" / "images" / "legacy_7mm"
MEDIA = WS / "hw_tests" / "media"


def pump(root, app, seconds=None, until_idle=False, tick=0.03):
    """Process Tk events; optionally until the app's worker finishes."""
    t0 = time.monotonic()
    while True:
        root.update()
        if until_idle and not app.busy:
            return True
        if seconds is not None and time.monotonic() - t0 >= seconds:
            return not app.busy if until_idle else True
        time.sleep(tick)


def screenshot(root, name):
    try:
        from PIL import ImageGrab
    except ImportError:
        print("PIL missing; no screenshot")
        return
    MEDIA.mkdir(parents=True, exist_ok=True)
    root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(MEDIA / name)
    print(f"screenshot -> {MEDIA / name}")


def gui_log(app):
    return app.log_text.get("1.0", "end")


def scan(root, app, timeout=30):
    app.scan_bus()
    ok = pump(root, app, seconds=timeout, until_idle=True)
    if not ok:
        raise RuntimeError("scan did not finish in time")
    return {b: r["version"] for b, r in app.rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["one", "fleet"], required=True)
    args = ap.parse_args()
    targets = [0x109] if args.phase == "one" else LEGACY_IDS

    root = tk.Tk()
    root.title(f"OTA legacy test — phase {args.phase}")
    app = flash_mod.FlashApp(root)
    verdict = {"phase": args.phase, "targets": [hex(b) for b in targets]}
    try:
        pump(root, app, seconds=0.5)

        # The GUI's own build loader (what Browse calls), pointed at the
        # legacy image directory. Overrides the TLE build it auto-loads.
        app._load_plan(IMAGE_DIR)
        pump(root, app, seconds=0.3)
        assert app.plan is not None and app.plan.app_file is not None
        assert app.image_variants == frozenset({0, 1}), app.image_variants
        assert "Valve_not_embedded_XL" in app.build_var.get()
        verdict["image"] = str(app.plan.app_file)
        verdict["image_bytes"] = app.plan.app_size

        port = app._device(app.can_port.get())
        assert port, "no CAN adapter preselected"
        verdict["port"] = port

        before = scan(root, app)
        verdict["scan_before"] = {hex(b): [v.variant, v.version]
                                 for b, v in sorted(before.items())}
        assert len(before) == 24, f"expected 24 boards, saw {len(before)}"
        tle_before = {b: before[b].version for b in TLE_IDS}

        # The guard's own pre-selection must be exactly the 16 legacy boards.
        pre = sorted(b for b, r in app.rows.items() if r["selected"])
        assert pre == LEGACY_IDS, f"guard pre-selection wrong: {pre}"

        # Narrow the selection to this phase's targets, via the GUI's state.
        app.select_all(False)
        for b in targets:
            app.rows[b]["selected"] = True
            app.tree.set(str(b), "id", f"[x] 0x{b:03X}")
        chosen = sorted(b for b, r in app.rows.items() if r["selected"])
        assert chosen == sorted(targets)
        assert not any(b in TLE_IDS for b in chosen), "TLE id in selection!"
        screenshot(root, f"flash_gui_{args.phase}_selected.png")

        t0 = time.monotonic()
        app.broadcast()
        # ~300 blocks; one board ~40 s, sixteen a few minutes.
        ok = pump(root, app, seconds=540, until_idle=True)
        verdict["broadcast_seconds"] = round(time.monotonic() - t0, 1)
        if not ok:
            raise RuntimeError("broadcast did not finish inside 540 s")
        screenshot(root, f"flash_gui_{args.phase}_done.png")

        log = gui_log(app)
        expect = f"updated {len(targets)}/{len(targets)}"
        verdict["updated_line_ok"] = expect in log
        if not verdict["updated_line_ok"]:
            tail = "\n".join(log.strip().splitlines()[-15:])
            raise RuntimeError(f"GUI did not report '{expect}'. Log tail:\n{tail}")

        # Boards reboot at CMD_END; give the last one a moment, then rescan.
        time.sleep(3.0)
        after = scan(root, app)
        verdict["scan_after"] = {hex(b): [v.variant, v.version]
                                for b, v in sorted(after.items())}
        assert len(after) == 24, f"after OTA only {len(after)} boards answer"
        for b in targets:
            v = after[b]
            assert v.variant == 0 and v.version == "0.2.1", (hex(b), v)
        for b in TLE_IDS:
            assert after[b].version == tle_before[b], (
                f"TLE board {hex(b)} version changed!")
        verdict["result"] = "PASSED"
    except BaseException as exc:
        verdict["result"] = f"FAILED: {type(exc).__name__}: {exc}"
        raise
    finally:
        out = WS / "hw_tests" / "results" / f"ota_legacy_{args.phase}_2026-08-20.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict.get("result"), indent=2))
        try:
            app.on_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
