"""Steps 2-5 of the CAN-arm live mocap verification, in one re-runnable pass.

The census (``mocap_census.py``) settles *which* ids exist.  This runs the arm's
own receiver against whichever block the census found, and records the four
things the campaign needs:

* **q health** -- fps, frames, valid conversions, the ``q_stale`` fraction, and
  per-joint mean and sd over the whole window.  The arm hangs unpressurised, so
  the sd is a measurement noise floor rather than motion, and a single joint
  whose sd is much larger than its neighbours' is a marker-visibility problem
  and not a compliant joint.
* **per-body dropouts** -- how many committed frames each rigid body was absent
  from, and how many it was present but unsolved (Motive streams
  ``pos=(0,0,0), quat=(0,0,0,0)`` for a body it cannot solve).  ``MocapRx`` keeps
  a row's previous pose when a frame does not mention it, so a plate that has
  silently stopped updating is indistinguishable from a plate that is not
  moving unless something counts the absences.  That counting is the only
  behaviour added here; everything else is the shipped receiver.
* **the chain** -- the five consecutive u-joint-centre gaps, from the ring's
  own ``u`` array, which is what ``UMArm_KINEMATICS.canarm_params`` must be
  written from and what the visualiser draws.
* **the room** -- whether the RS485 arm's block and the Kinova's body 1008
  stream, read-only, for the room viewer's optional arms.

``--sim`` swaps the transport for ``sim_stream.CanArmSimStream`` and leaves every
other line the same, so the analysis is exercised without cameras.  A sim run
proves the plumbing and nothing about Motive: the synthetic generator inverts
the reader's own steps with the reader's own constants, so an error shared with
the reader cancels exactly.

The SDK's threads are **not daemons**.  Every receiver below is stopped in a
``finally``, including on the raising paths, or the process wedges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
if _WS not in sys.path:
    sys.path.insert(0, _WS)

from UMArm_MOCAP import mocap_constants as mc          # noqa: E402
from UMArm_MOCAP.canarm_mocap import (                 # noqa: E402
    CANARM_N_BODIES, CANARM_RB_ID_BASE, CanArmMocap)

RS485_BLOCK_CANDIDATES = (500, 1000)


class CountingCanArmMocap(CanArmMocap):
    """``CanArmMocap`` that also tallies which bodies each frame mentioned.

    Subclassing rather than patching keeps the receiver's own single-writer
    rule intact: both hooks run on the SDK's data thread, in the SDK's own
    order (every ``rigid_body_listener`` call, then ``new_frame_listener``), so
    the per-frame set is complete at commit and no lock is needed for it.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seen_this_frame: set[int] = set()
        self._unsolved_this_frame: set[int] = set()
        self.committed_frames = 0
        self.absent_count: dict[int, int] = {}
        self.unsolved_count: dict[int, int] = {}
        self.present_count: dict[int, int] = {}
        self.dropout_events: dict[int, int] = {}
        self._was_present: dict[int, bool] = {}

    def _on_rigid_body(self, new_id, position, quat_xyzw) -> None:
        self._seen_this_frame.add(int(new_id))
        if not any(float(v) for v in quat_xyzw):
            self._unsolved_this_frame.add(int(new_id))
            # Let the parent raise-on-zero-quaternion path do its own thing;
            # suppressing the call would hide a real receiver behaviour.
        super()._on_rigid_body(new_id, position, quat_xyzw)

    def _on_new_frame(self, data_dict) -> None:
        watched = [self.rb_id_base + i for i in range(self.n_bodies)]
        watched.append(mc.KINOVA_MOCAP_STREAM_ID)
        self.committed_frames += 1
        for rb in watched:
            present = rb in self._seen_this_frame
            solved = present and rb not in self._unsolved_this_frame
            self.present_count[rb] = self.present_count.get(rb, 0) + int(present)
            self.absent_count[rb] = self.absent_count.get(rb, 0) + int(not present)
            self.unsolved_count[rb] = (self.unsolved_count.get(rb, 0)
                                       + int(present and not solved))
            # An "event" is a falling edge, not a frame: one 300-frame occlusion
            # and 300 scattered single-frame drops have the same frame count and
            # very different meanings for a control loop.
            was = self._was_present.get(rb, True)
            if was and not solved:
                self.dropout_events[rb] = self.dropout_events.get(rb, 0) + 1
            self._was_present[rb] = solved
        self._seen_this_frame.clear()
        self._unsolved_this_frame.clear()
        super()._on_new_frame(data_dict)


def _stats(rx, seconds: float, poll_hz: float = 20.0) -> dict:
    """Run for *seconds*, sampling ``get_state()`` to get a q_stale duty cycle.

    ``q_stale`` is a level, not an event, so the honest summary is the fraction
    of polls that observed it rather than a count of transitions -- and it is
    ``q_stale`` and not ``stale`` that a control loop reads, since the two
    differ exactly when frames keep arriving and stop converting.
    """
    t0 = time.monotonic()
    end = t0 + seconds
    polls = stale_polls = q_stale_polls = 0
    last_err = None
    while time.monotonic() < end:
        st = rx.get_state()
        polls += 1
        stale_polls += int(st.stale)
        q_stale_polls += int(st.q_stale)
        last_err = st.last_error or last_err
        time.sleep(1.0 / poll_hz)
    t1 = time.monotonic()
    st = rx.get_state()
    win = rx.snapshot_window(t0, t1)

    out = {
        "seconds": seconds,
        "frames": int(st.frames),
        "valid_frames": int(st.valid_frames),
        "fps": float(st.fps),
        "stale_at_end": bool(st.stale),
        "q_stale_at_end": bool(st.q_stale),
        "polls": polls,
        "stale_fraction": stale_polls / polls if polls else 1.0,
        "q_stale_fraction": q_stale_polls / polls if polls else 1.0,
        "last_error": last_err,
        "ring_len": int(st.ring_len),
        "window_n": len(win),
        "window_duration_s": float(win.duration),
        "window_fps": float(win.fps),
    }
    if len(win) >= 2:
        out["q_mean_rad"] = [float(v) for v in win.q.mean(axis=0)]
        out["q_sd_rad"] = [float(v) for v in win.q.std(axis=0, ddof=1)]
        out["q_ptp_rad"] = [float(v) for v in np.ptp(win.q, axis=0)]
        # Frame-number gaps are dropped UDP, and are invisible in the fps.
        fn = np.asarray(win.frame_no, dtype=np.int64)
        d = np.diff(fn)
        out["frame_number_gaps"] = int(np.count_nonzero(d > 1))
        out["frame_numbers_missed"] = int(np.clip(d - 1, 0, None).sum())
    return out


def _chain(rx, seconds: float) -> dict:
    """The five consecutive u-joint-centre gaps over a fresh window.

    ``MocapWindow.u`` is ``(n, 6, 3)`` -- the six centres in the mocap spatial
    frame -- so the gaps are differences of adjacent centres and are therefore
    invariant to where the volume's origin happens to sit, which is the property
    that makes them safe to write into a parameter table.
    """
    t0 = time.monotonic()
    time.sleep(seconds)
    win = rx.snapshot_window(t0, time.monotonic())
    if len(win) < 2:
        return {"n": len(win), "error": "not enough samples"}
    gaps = np.linalg.norm(np.diff(win.u, axis=1), axis=2)      # (n, 5)
    return {
        "n": len(win),
        "duration_s": float(win.duration),
        "gap_mean_m": [float(v) for v in gaps.mean(axis=0)],
        "gap_sd_m": [float(v) for v in gaps.std(axis=0, ddof=1)],
        "gap_min_m": [float(v) for v in gaps.min(axis=0)],
        "gap_max_m": [float(v) for v in gaps.max(axis=0)],
        "total_m": float(gaps.mean(axis=0).sum()),
    }


def _room(server: str, client_ip: str, multicast: bool, seconds: float) -> dict:
    """Read-only: does the RS485 block or the Kinova's 1008 stream?

    Uses the census rather than another receiver, since the question is about
    ids the CAN arm's receiver is built to drop.
    """
    from mocap_census import run_census                        # noqa: PLC0415
    res = run_census(seconds, server, client_ip, multicast)
    ids = {rb["id"]: rb for rb in res["rigid_bodies"]}
    out = {"census_frames": res["frames"], "census_fps": res["fps"],
           "ids": sorted(ids)}
    for base in RS485_BLOCK_CANDIDATES:
        present = [i for i in range(base, base + 6) if base + (i - base) in ids]
        out["rs485_block_%d" % base] = {
            "present": present,
            "complete": len(present) == 6,
        }
    k = mc.KINOVA_MOCAP_STREAM_ID
    out["kinova_1008"] = {"present": k in ids,
                          "hz": ids[k]["hz"] if k in ids else 0.0,
                          "unsolved_frames": ids[k]["zero_quat"] if k in ids else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", action="store_true",
                    help="drive the analysis from sim_stream instead of Motive")
    ap.add_argument("--rb-id-base", type=int, default=CANARM_RB_ID_BASE)
    ap.add_argument("--n-bodies", type=int, default=CANARM_N_BODIES)
    ap.add_argument("--q-seconds", type=float, default=60.0)
    ap.add_argument("--chain-seconds", type=float, default=30.0)
    ap.add_argument("--room-seconds", type=float, default=10.0)
    ap.add_argument("--server-ip", default=mc.DEFAULT_SERVER_IP)
    ap.add_argument("--client-ip", default=mc.DEFAULT_CLIENT_IP)
    ap.add_argument("--no-multicast", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    total = a.q_seconds + a.chain_seconds + 5.0
    # Ring must span the whole q window plus the chain window; the 120 Hz
    # default sizes 20 s and would silently truncate a 60 s read.
    ring = int(max(mc.RING_CAPACITY, total * 240.0))

    report: dict = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "sim" if a.sim else "live",
        "rb_id_base": a.rb_id_base,
        "n_bodies": a.n_bodies,
        "server_ip": a.server_ip, "client_ip": a.client_ip,
        "multicast": not a.no_multicast,
    }

    if a.sim:
        from UMArm_MOCAP.sim_stream import CanArmSimStream    # noqa: PLC0415

        class _SimCounting(CountingCanArmMocap, CanArmSimStream):
            """The counting hooks over the synthetic transport.

            MRO order matters: ``CountingCanArmMocap`` must precede so its
            listeners wrap, while ``start``/``stop`` come from the sim.
            """

        rx = _SimCounting(rb_id_base=a.rb_id_base, n_bodies=a.n_bodies,
                          ring_capacity=ring)
    else:
        rx = CountingCanArmMocap(
            server_ip=a.server_ip, client_ip=a.client_ip,
            use_multicast=not a.no_multicast,
            rb_id_base=a.rb_id_base, n_bodies=a.n_bodies,
            ring_capacity=ring)

    try:
        rx.start()
        # Refuse early and loudly rather than reporting 60 s of zeros.
        if rx.wait_fresh(timeout=5.0) is None:
            st = rx.get_state()
            report["error"] = (
                "no valid q within 5 s: frames=%d valid=%d stale=%s "
                "last_error=%r -- the block %d..%d is not converting"
                % (st.frames, st.valid_frames, st.stale, st.last_error,
                   a.rb_id_base, a.rb_id_base + a.n_bodies - 1))
            print(report["error"])
        else:
            print("[2/4] q health, %.0f s ..." % a.q_seconds)
            report["q"] = _stats(rx, a.q_seconds)
            print("[3/4] chain, %.0f s ..." % a.chain_seconds)
            report["chain"] = _chain(rx, a.chain_seconds)
            report["bodies"] = {
                "committed_frames": rx.committed_frames,
                "per_id": {
                    str(rb): {
                        "present": rx.present_count.get(rb, 0),
                        "absent": rx.absent_count.get(rb, 0),
                        "unsolved": rx.unsolved_count.get(rb, 0),
                        "dropout_events": rx.dropout_events.get(rb, 0),
                    }
                    for rb in sorted(set(rx.present_count) | set(rx.absent_count))
                },
            }
    finally:
        rx.stop()

    if not a.sim:
        print("[4/4] room roster ...")
        try:
            report["room"] = _room(a.server_ip, a.client_ip,
                                   not a.no_multicast, a.room_seconds)
        except Exception as exc:
            report["room"] = {"error": repr(exc)}

    print(json.dumps(report, indent=2))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("wrote %s" % a.json_out)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
