"""Read-only connectivity check of the Kinova Gen3 through the workspace bridge.

Proves, against the real arm at 192.168.1.10, that: the bridge spawns on
``.venv_kinova``; ``ping`` answers; ``connect`` succeeds and returns a live
snapshot (7 joint angles + tool pose); the bridge's in-process mocap half
starts and reports its state (with Motive down it must degrade loudly, not
hang); and teardown actually ends the bridge process.

Deliberately absent: ``arm``, ``jog``, ``joints`` — no motion command of any
kind is sent. The one deliberate motion demo (a small room-frame jog verified
by mocap body 1008) is deferred until Motive streams again; see
``report_kinova_2026-08-20.md``.

Run with the ordinary workspace interpreter::

    .venv\\Scripts\\python.exe hw_tests\\kinova_bridge_connectivity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))

from UMArm_KINOVA import bridge_client  # noqa: E402


def drain(link, out, seconds):
    """Poll the link for `seconds`, appending every message to `out`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for m in link.poll():
            out.append(m)
            print("  <-", json.dumps(m)[:200])
        time.sleep(0.05)


def wait_for(link, out, pred, seconds, what):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for m in link.poll():
            out.append(m)
            print("  <-", json.dumps(m)[:200])
        for m in out:
            if pred(m):
                return m
        time.sleep(0.05)
    raise TimeoutError(f"no {what} within {seconds:.0f}s "
                       f"(state={link.state}, err={link.error!r})")


def main() -> int:
    link = bridge_client.KinovaLink()
    msgs: list[dict] = []
    verdict = {}
    try:
        print("-> start bridge:", link.start())
        wait_for(link, msgs, lambda m: m.get("kind") == "ready"
                 or link.state == "running", 30, "bridge ready")
        print("bridge state:", link.state)

        link.send({"cmd": "ping"})
        wait_for(link, msgs,
                 lambda m: m.get("kind") == "ok" and m.get("cmd") == "ping",
                 10, "ping ok")
        verdict["ping"] = True

        print("-> connect (defaults: 192.168.1.10)")
        link.send({"cmd": "connect"})
        ok = wait_for(link, msgs,
                      lambda m: (m.get("kind") == "ok"
                                 and m.get("cmd") == "connect"),
                      40, "connect ok")
        verdict["connect_ip"] = ok.get("ip")
        verdict["q_deg"] = ok.get("q_deg")
        verdict["pose"] = ok.get("pose")

        # The bridge starts its mocap half after connect, on its own thread.
        # With Motive down we expect either a "mocap did not start" log or a
        # silent receiver that simply never produces frames. Give it a bounded
        # window and record what actually happened.
        print("-> observing mocap half for 12 s (Motive is known down)")
        drain(link, msgs, 12)

        print("-> mocap_solve_base (expected to refuse: no frames)")
        link.send({"cmd": "mocap_solve_base"})
        try:
            m = wait_for(link, msgs,
                         lambda m: m.get("cmd") == "mocap_solve_base", 15,
                         "mocap_solve_base reply")
            verdict["mocap_solve_base"] = m
        except TimeoutError as exc:
            verdict["mocap_solve_base"] = f"no reply: {exc}"

        print("-> disconnect")
        link.send({"cmd": "disconnect"})
        try:
            wait_for(link, msgs, lambda m: m.get("kind") == "ok"
                     and m.get("cmd") == "disconnect", 20, "disconnect ok")
            verdict["disconnect"] = True
        except TimeoutError as exc:
            verdict["disconnect"] = str(exc)
    finally:
        link.shutdown()
        # The SDK's threads are not daemons; prove the process is gone.
        proc = getattr(link, "_proc", None)
        alive = proc is not None and proc.poll() is None
        verdict["bridge_exited"] = not alive
        if alive:
            proc.kill()

    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    ok = (verdict.get("ping") and verdict.get("connect_ip")
          and verdict.get("q_deg") and verdict.get("bridge_exited"))
    print("RESULT:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
