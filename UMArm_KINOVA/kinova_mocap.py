"""Reading the Gen3's mocap rigid body — Motive streaming id 1008.

``UMArm_MOCAP`` already knows about this body: ``mocap_constants`` reserves it
(:data:`~UMArm_MOCAP.mocap_constants.KINOVA_MOCAP_STREAM_ID` = 1008,
:data:`~UMArm_MOCAP.mocap_constants.KINOVA_RIGID_BODY_INDEX` = 8) and
``MocapRx._on_rigid_body`` already routes it into row 8 of the pose array.  What
the receiver does NOT do is *record* row 8: its ring holds ``q`` and the six
u-joint centres, and its marker ring holds only the arm's own seven plates
(``N_USED_RIGID_BODIES``).  The Kinova's pose is therefore live-only — readable
through ``get_homos()``, gone the next frame.

:class:`KinovaMocapRx` adds the missing ring.  It subclasses the real receiver
rather than wrapping it, which is the same move ``SimMocap`` and ``MarkerMocap``
make and for the same reason: everything a consumer touches stays the one real
implementation, so nothing about staleness, rates or threading is re-invented
here.  The override appends to its own deque and then calls straight through.

WHY AVERAGE AT ALL.  A single 120 Hz frame of a four-marker body carries
Motive's per-frame solve jitter — a few tenths of a millimetre on this rig.
Every consumer in this package samples a *stationary* arm, so averaging a second
of frames divides that by about eleven for free.  What the average must not hide
is motion: :meth:`KinovaMocapRx.capture` reports the peak-to-peak spread
alongside the mean, and a caller that finds the spread larger than the jitter it
expected has caught the arm still settling rather than fitted through it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from UMArm_MOCAP import mocap_constants as mc          # noqa: E402
from UMArm_MOCAP.mocap_rx import MocapRx               # noqa: E402
from UMArm_KINOVA.mocap_calibration import (           # noqa: E402
    project_rotation, se3)

#: Ring depth for the Kinova body, in samples.  20 s at Motive's 120 Hz, the
#: same span ``mocap_constants.RING_SECONDS`` gives the arm's own ring, so a
#: caller can ask for any window either receiver could have answered.
RING_CAPACITY = int(mc.RING_SECONDS * mc.NOMINAL_RATE_HZ)

#: A capture with fewer frames than this is refused rather than averaged.  Ten
#: frames is a twelfth of a second: below it the mean has not meaningfully beaten
#: one frame's jitter and the peak-to-peak spread is not yet evidence of
#: anything.
MIN_CAPTURE_FRAMES = 10


class KinovaMocapRx(MocapRx):
    """A :class:`~UMArm_MOCAP.mocap_rx.MocapRx` that also records body 1008."""

    def __init__(self, *args, **kwargs):
        # Set up before super().__init__, because the SDK thread the parent may
        # start could deliver a frame before this constructor's next line runs.
        self._kin_lock = threading.Lock()
        self._kin_ring: deque = deque(maxlen=RING_CAPACITY)
        self._kin_marker_ring: deque = deque(maxlen=RING_CAPACITY)
        self._kin_frames = 0
        super().__init__(*args, **kwargs)

    def _on_rigid_body(self, new_id, position, quat_xyzw) -> None:
        if new_id == mc.KINOVA_MOCAP_STREAM_ID:
            try:
                with self._kin_lock:
                    self._kin_ring.append(
                        (time.monotonic(), time.time(),
                         np.asarray(position, dtype=float).copy(),
                         np.asarray(quat_xyzw, dtype=float).copy()))
                    self._kin_frames += 1
            except Exception:            # never let the SDK thread die on us
                pass
        super()._on_rigid_body(new_id, position, quat_xyzw)

    def _on_mocap_data(self, mocap_data) -> None:
        """Also keep the four LABELED markers whose model id is 1008.

        The parent drops them, correctly: it filters to the arm's own plates
        before doing any per-marker work, so a cluttered volume cannot grow the
        120 Hz receive path.  The Kinova's four are wanted here for one reason —
        they are the only way to ask whether the operator's CAD stands describe
        the stands actually glued to the rig, which every claim about
        ``mount_transforms.PAD_MARKER_XZ_MM`` ultimately rests on.

        ``marker_id - 1`` is the stable slot within the asset, the same
        assumption ``mocap_rx`` makes and tallies for the arm's plates.  A
        marker absent from a frame leaves a NaN row rather than shortening the
        array, so a consumer counting four never silently gets three.
        """
        try:
            lmd = getattr(mocap_data, "labeled_marker_data", None)
            entries = {} if lmd is None else {
                lm.id_num & 0xFFFF: lm for lm in lmd.labeled_marker_list
                if (lm.id_num >> 16) == mc.KINOVA_MOCAP_STREAM_ID}
            if entries:
                n = max(4, max(entries))
                arr = np.full((n, 3), np.nan, dtype=float)
                for slot, lm in entries.items():
                    if 1 <= slot <= n:
                        arr[slot - 1] = lm.pos
                with self._kin_lock:
                    self._kin_marker_ring.append((time.monotonic(), arr))
        except Exception:                # never let the SDK thread die on us
            pass
        super()._on_mocap_data(mocap_data)

    def kinova_marker_window(self, t0=None, t1=None) -> list:
        """Recorded 1008 LABELED-marker samples, ``(t_mono, (n, 3) array)``."""
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        with self._kin_lock:
            return [(t, a.copy()) for t, a in self._kin_marker_ring
                    if lo <= t <= hi]

    def capture_markers(self, seconds: float = 2.0, settle: float = 0.0) -> dict:
        """Mean position of each of body 1008's markers, and its spread.

        Frames in which a marker was missing are skipped **for that marker
        only**, so one occluded ball does not throw away the other three.  The
        per-marker frame count comes back with the means, because a marker seen
        in a tenth of the frames is a marker the cameras barely have.
        """
        if settle > 0:
            time.sleep(float(settle))
        t0 = time.monotonic()
        time.sleep(float(seconds))
        rows = self.kinova_marker_window(t0=t0)
        if not rows:
            raise RuntimeError(
                "no labeled markers with model id %d arrived in %.2f s"
                % (mc.KINOVA_MOCAP_STREAM_ID, seconds))
        width = max(a.shape[0] for _, a in rows)
        stack = np.full((len(rows), width, 3), np.nan, dtype=float)
        for i, (_, a) in enumerate(rows):
            stack[i, 0:a.shape[0]] = a
        seen = np.isfinite(stack[:, :, 0])
        out = {"frames": len(rows), "n_markers": int(width),
               "seen": seen.sum(axis=0).tolist(), "points_m": [], "sd_mm": []}
        for k in range(width):
            if not seen[:, k].any():
                out["points_m"].append([float("nan")] * 3)
                out["sd_mm"].append([float("nan")] * 3)
                continue
            col = stack[seen[:, k], k, :]
            out["points_m"].append(col.mean(axis=0).tolist())
            out["sd_mm"].append((col.std(axis=0) * 1e3).tolist())
        return out

    @property
    def kinova_frames(self) -> int:
        """Frames of body 1008 seen since start.  Zero means it is not streaming."""
        with self._kin_lock:
            return int(self._kin_frames)

    def kinova_window(self, t0=None, t1=None) -> list:
        """Recorded 1008 samples with ``t0 <= t_mono <= t1`` (both monotonic)."""
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        with self._kin_lock:
            return [r for r in self._kin_ring if lo <= r[0] <= hi]

    def capture(self, seconds: float = 1.0, settle: float = 0.0) -> dict:
        """Watch body 1008 for *seconds* and return its mean pose and spread.

        *settle* seconds are waited and DISCARDED first, so a caller can put the
        settling delay and the averaging window in one call without the tail of
        a move leaking into the mean.

        The returned ``T`` is ``T_world_rb`` as a ``(4, 4)``: Motive's own frame
        for the rigid body, in the volume's coordinates.  ``pos_ptp_mm`` and
        ``ang_ptp_deg`` are the peak-to-peak spread over the window — the
        evidence that the arm was actually still.
        """
        if settle > 0:
            time.sleep(float(settle))
        t0 = time.monotonic()
        time.sleep(float(seconds))
        rows = self.kinova_window(t0=t0)
        if len(rows) < MIN_CAPTURE_FRAMES:
            raise RuntimeError(
                "only %d frames of rigid body %d in %.2f s (need %d) — is the "
                "body in the Motive project, visible to the cameras, and being "
                "streamed?" % (len(rows), mc.KINOVA_MOCAP_STREAM_ID, seconds,
                               MIN_CAPTURE_FRAMES))
        return summarise(rows)


def quat_xyzw_to_matrix(q) -> np.ndarray:
    """``(x, y, z, w)`` -> rotation matrix.

    Re-implemented here rather than imported from ``mocap_rx`` so this module
    keeps working if that helper is ever made private; it is four lines and the
    convention is the thing that matters.  Motive streams xyzw, which is the
    order this repo carries everywhere.
    """
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def summarise(rows) -> dict:
    """Mean pose and spread of a list of ``(t_mono, t_wall, pos, quat)`` rows.

    Rotations are averaged by projecting the arithmetic mean of the matrices
    back onto SO(3).  Over a window this tight — the spread is a fraction of a
    degree — the chordal mean and the geodesic mean agree to far below the
    jitter, and the projection has no branch to get wrong.
    """
    t_mono = np.array([r[0] for r in rows], dtype=float)
    t_wall = np.array([r[1] for r in rows], dtype=float)
    pos = np.array([r[2] for r in rows], dtype=float)
    mats = np.array([quat_xyzw_to_matrix(r[3]) for r in rows], dtype=float)
    R = project_rotation(mats.mean(axis=0))
    # Per-frame angle away from the mean rotation, as a spread in degrees.
    ang = np.degrees([np.arccos(np.clip((np.trace(R.T @ m) - 1.0) / 2.0,
                                        -1.0, 1.0)) for m in mats])
    return {
        "n": int(len(rows)),
        "t_mono": float(t_mono.mean()),
        "t_wall": float(t_wall.mean()),
        "duration_s": float(t_mono[-1] - t_mono[0]),
        "T": se3(R, pos.mean(axis=0)),
        "pos_m": pos.mean(axis=0).tolist(),
        "pos_sd_mm": (pos.std(axis=0) * 1e3).tolist(),
        "pos_ptp_mm": float(np.max(np.ptp(pos, axis=0)) * 1e3),
        "ang_sd_deg": float(ang.std()),
        "ang_ptp_deg": float(np.ptp(ang) if len(ang) > 1 else 0.0),
    }


def main(argv=None) -> int:
    """Print body 1008's pose, live.  Reads the network; moves nothing."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--server-ip", default=None)
    ap.add_argument("--client-ip", default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args(argv)

    kw = {}
    if args.server_ip:
        kw["server_ip"] = args.server_ip
    if args.client_ip:
        kw["client_ip"] = args.client_ip
    rx = KinovaMocapRx(**kw)
    rx.start()
    try:
        cap = rx.capture(seconds=args.seconds, settle=0.5)
    finally:
        rx.stop()
    T = cap.pop("T")
    print(json.dumps(cap, indent=2))
    print("T_world_rb =")
    for row in T:
        print("   " + "  ".join("%10.6f" % v for v in row))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
