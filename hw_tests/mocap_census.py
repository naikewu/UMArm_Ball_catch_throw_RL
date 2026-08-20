"""Step 1 -- enumerate every rigid body Motive streams, with per-id frame rates.

The workspace carries three mutually contradictory claims about which block of
Motive streaming ids belongs to which arm (``UMArm_MOCAP/canarm_mocap.py``
docstring).  Nothing offline can arbitrate between them, so this script does the
one thing that can: it attaches a listener that accepts **every** id, counts what
arrives over a fixed window, and prints the roster.

It deliberately does not use :class:`~UMArm_MOCAP.mocap_rx.MocapRx`.  That class
exists to *drop* ids outside its own block -- which is correct for a control loop
and useless for a census.  The SDK is driven directly instead, so no filtering
sits between Motive and the count.

Also recorded, because step 4 (marker locks) depends on it: whether the stream
carries labeled markers at all, and which model id each labeled marker declares
in its high 16 bits.  A volume that streams rigid bodies but not labeled markers
cannot mint a lock, and that is a property of the Motive project's streaming
settings rather than of anything in this repo.

The SDK's threads are **not daemons**: a receiver started and never shut down
holds the interpreter open past any timeout.  Every path below therefore stops
the client in a ``finally``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
if _WS not in sys.path:
    sys.path.insert(0, _WS)

_SDK = os.path.join(_WS, "UMArm_MOCAP", "natnet_sdk")
if _SDK not in sys.path:
    sys.path.insert(0, _SDK)

DEFAULT_SERVER = "192.168.1.100"
DEFAULT_CLIENT = "192.168.1.120"


class Census:
    """Counts of everything the stream mentions, guarded by one lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rb_counts: dict[int, int] = collections.defaultdict(int)
        self.rb_first_t: dict[int, float] = {}
        self.rb_last_t: dict[int, float] = {}
        self.rb_last_pos: dict[int, tuple] = {}
        self.rb_zero_quat: dict[int, int] = collections.defaultdict(int)
        self.frames = 0
        self.frame_numbers: list[int] = []
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.labeled_frames = 0
        self.labeled_model_counts: dict[int, int] = collections.defaultdict(int)
        self.labeled_slots: dict[int, set] = collections.defaultdict(set)
        self.marker_set_names: dict[str, int] = collections.defaultdict(int)
        self.marker_set_sizes: dict[str, int] = {}
        self.mocap_data_frames = 0

    # -- SDK listeners, all on the SDK data thread ---------------------- #

    def on_rigid_body(self, new_id, position, quat_xyzw) -> None:
        t = time.monotonic()
        with self.lock:
            self.rb_counts[new_id] += 1
            self.rb_first_t.setdefault(new_id, t)
            self.rb_last_t[new_id] = t
            self.rb_last_pos[new_id] = tuple(float(v) for v in position)
            # Motive streams pos=(0,0,0) quat=(0,0,0,0) for a body it cannot
            # solve, and hands it to the listener before the tracking-valid bit
            # is parsed.  Counting them separates "the asset exists" from "the
            # asset is being tracked".
            if not any(float(v) for v in quat_xyzw):
                self.rb_zero_quat[new_id] += 1

    def on_mocap_data(self, mocap_data) -> None:
        with self.lock:
            self.mocap_data_frames += 1
        msd = getattr(mocap_data, "marker_set_data", None)
        sets = [] if msd is None else msd.marker_data_list
        lmd = getattr(mocap_data, "labeled_marker_data", None)
        labeled = [] if lmd is None else lmd.labeled_marker_list
        with self.lock:
            for ms in sets:
                name = ms.model_name
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                name = str(name).strip("\x00")
                self.marker_set_names[name] += 1
                try:
                    self.marker_set_sizes[name] = len(ms.marker_pos_list)
                except Exception:
                    pass
            if labeled:
                self.labeled_frames += 1
            for lm in labeled:
                model = lm.id_num >> 16
                self.labeled_model_counts[model] += 1
                self.labeled_slots[model].add(lm.id_num & 0xFFFF)

    def on_new_frame(self, data_dict) -> None:
        t = time.monotonic()
        with self.lock:
            self.frames += 1
            if self.t_first is None:
                self.t_first = t
            self.t_last = t
            fn = data_dict.get("frame_number")
            if fn is not None and len(self.frame_numbers) < 100000:
                self.frame_numbers.append(int(fn))


def run_census(seconds: float, server: str, client_ip: str,
               multicast: bool) -> dict:
    from NatNetClient import NatNetClient  # noqa: PLC0415

    cen = Census()
    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(server)
    client.set_use_multicast(multicast)
    client.rigid_body_listener = cen.on_rigid_body
    client.mocap_data_listener = cen.on_mocap_data
    client.new_frame_listener = cen.on_new_frame

    started = False
    try:
        if not client.run():
            raise RuntimeError(
                "NatNetClient.run() refused: server=%s client=%s multicast=%s"
                % (server, client_ip, multicast))
        started = True
        time.sleep(seconds)
    finally:
        # Not daemon threads.  Shut down on every path, including the raise.
        try:
            client.shutdown()
        except Exception as exc:  # pragma: no cover
            print("shutdown raised: %r" % (exc,), file=sys.stderr)
        if started:
            time.sleep(0.3)

    with cen.lock:
        span = ((cen.t_last - cen.t_first)
                if (cen.t_first is not None and cen.t_last is not None)
                else 0.0)
        out = {
            "seconds_requested": seconds,
            "server": server, "client": client_ip, "multicast": multicast,
            "frames": cen.frames,
            "span_s": span,
            "fps": (cen.frames - 1) / span if span > 0 else 0.0,
            "mocap_data_frames": cen.mocap_data_frames,
            "labeled_frames": cen.labeled_frames,
            "rigid_bodies": [],
            "labeled_models": [],
            "marker_sets": [],
            "frame_number_span": (
                [cen.frame_numbers[0], cen.frame_numbers[-1]]
                if cen.frame_numbers else None),
        }
        for rb_id in sorted(cen.rb_counts):
            n = cen.rb_counts[rb_id]
            dur = cen.rb_last_t[rb_id] - cen.rb_first_t[rb_id]
            out["rigid_bodies"].append({
                "id": rb_id,
                "count": n,
                "hz": (n - 1) / dur if dur > 0 else 0.0,
                "frac_of_frames": n / cen.frames if cen.frames else 0.0,
                "zero_quat": cen.rb_zero_quat.get(rb_id, 0),
                "last_pos": cen.rb_last_pos.get(rb_id),
            })
        for model in sorted(cen.labeled_model_counts):
            out["labeled_models"].append({
                "model_id": model,
                "marker_count": cen.labeled_model_counts[model],
                "slots": sorted(cen.labeled_slots[model]),
            })
        for name in sorted(cen.marker_set_names):
            out["marker_sets"].append({
                "name": name,
                "frames": cen.marker_set_names[name],
                "n_markers": cen.marker_set_sizes.get(name),
            })
    return out


def contiguous_blocks(ids: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive ids, as ``(first, last)`` pairs."""
    blocks: list[tuple[int, int]] = []
    for i in sorted(ids):
        if blocks and i == blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], i)
        else:
            blocks.append((i, i))
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--client", default=DEFAULT_CLIENT)
    ap.add_argument("--unicast", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    res = run_census(args.seconds, args.server, args.client,
                     not args.unicast)

    print("=" * 72)
    print("MOCAP ID CENSUS  server=%s client=%s multicast=%s  %.1f s"
          % (res["server"], res["client"], res["multicast"],
             res["seconds_requested"]))
    print("=" * 72)
    print("frames=%d  span=%.3f s  fps=%.2f  mocap_data_frames=%d  "
          "labeled_frames=%d"
          % (res["frames"], res["span_s"], res["fps"],
             res["mocap_data_frames"], res["labeled_frames"]))
    if res["frame_number_span"]:
        a, b = res["frame_number_span"]
        print("Motive frame numbers %d .. %d (delta %d)" % (a, b, b - a))
    print()
    print("RIGID BODIES")
    print("%8s %8s %9s %8s %10s  %s"
          % ("id", "count", "hz", "frac", "zero-quat", "last pos (m)"))
    for rb in res["rigid_bodies"]:
        p = rb["last_pos"]
        ps = ("(%+.4f %+.4f %+.4f)" % p) if p else "-"
        print("%8d %8d %9.2f %8.3f %10d  %s"
              % (rb["id"], rb["count"], rb["hz"], rb["frac_of_frames"],
                 rb["zero_quat"], ps))
    ids = [rb["id"] for rb in res["rigid_bodies"]]
    print()
    print("contiguous blocks: %s"
          % ", ".join("%d-%d" % b if b[0] != b[1] else "%d" % b[0]
                      for b in contiguous_blocks(ids)))
    print()
    print("LABELED MARKERS  (model id = id_num >> 16)")
    if not res["labeled_models"]:
        print("  none -- Motive is not streaming labeled markers.")
    for lm in res["labeled_models"]:
        print("  model %6d  markers=%d  slots=%s"
              % (lm["model_id"], lm["marker_count"], lm["slots"]))
    print()
    print("MARKER SETS")
    if not res["marker_sets"]:
        print("  none")
    for ms in res["marker_sets"]:
        print("  %-24s frames=%d n_markers=%s"
              % (ms["name"], ms["frames"], ms["n_markers"]))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
