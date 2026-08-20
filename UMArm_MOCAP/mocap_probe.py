"""Live mocap preflight, static capture and lock generation — no serial port.

Implements ``docs/marker_frame_design.md`` §6.  Run it against a (nominally)
resting arm before any campaign, and after every Motive re-calibration or
rigid-body edit::

    python UMArm_MOCAP/mocap_probe.py [--seconds 30]
        [--server-ip ... --client-ip ... --no-multicast]
        [--out-dir DIR] [--lock-out locks.json] [--allow-unlabeled]

**Reports** (always emitted, stdout + ``probe_report.json`` under ``--out-dir``):
per-plate tracked/jitter (the bit-identical repeated-pose test of
``joint_verification.repeated_plate_report``, re-stated here because this
package must not import the control repo), marker sets / labeled markers
present, per-plate marker counts and count stability, rectangle stats,
inferred-vs-streamed pose deltas, chain spans vs nominal, stream health, the
``marker_id`` correspondence check, and the occluded-marker encoding actually
observed on this Motive.

**Gates** (exit 3, named message — these are the failures that invalidate a
campaign before it starts):

* any chain span outside the +/-30 % nominal band -> *pivot placement*: re-set
  the rigid-body pivot in Motive (review finding ops-1);
* plate 0 or plate 5 streamed orientation vs marker-inferred body frame
  disagreeing by > 20 deg -> *Motive alignment*: the streamed 500/505
  orientations are load-bearing for the shipped q pipeline and nothing else
  guards them (review finding ops-0; warn from 10 deg, deltas printed exactly);
* x-candidate residual > 30 deg on any plate lock (design §4.1.3 — near the
  45 deg ambiguity boundary, a wrong-branch lock would be silently ~45 deg off).

**Lock-writing refusals** (``--lock-out`` only; reports still emitted, exit 2):
stillness failure over the window (q sd and marker sd — the probe cannot read
bus pressure, so stillness is the only rest evidence it has), any plate
tripping the repeated-pose test, labeled markers absent (unless
``--allow-unlabeled``, which stamps the regime into the JSON), any plate
without >= 1 s of all-four-markers-tracked frames (review finding ops-3 and
§4.1's lock precondition).

**Exit codes**: 0 clean; 1 stream never delivered / refused to start;
2 lock refused; 3 gate failed.  Gates dominate refusals.

**Degradation**: frame inference lives in ``marker_frame.py``, which is
imported lazily *inside* the functions that need it.  When it is missing (or
its API disagrees with this caller), the probe says so plainly and emits every
transport-level report anyway — reports-only mode.  The two inference gates
are then not evaluated, which the output states explicitly rather than
implying a pass.

Ring capacities are sized ``ceil(seconds * 1.5 * rate)`` with the rate
*measured* after a 2 s warm-up (nominal until then), so the advertised window
actually fits on a 240 Hz volume (review finding ops-10).  Multicast is the
lab default; ``--no-multicast`` is refused while another local client holds
the NatNet data port (concurrent probe+campaign is multicast-only).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import socket
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

try:  # see the note in mocap_to_q.py — both import styles must work
    from . import mocap_constants as mc
    from .mocap_rx import MarkerWindow, MocapRx, MocapWindow
except ImportError:  # pragma: no cover - flat style when run as a script
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import mocap_constants as mc  # type: ignore[no-redef]
    from mocap_rx import MarkerWindow, MocapRx, MocapWindow  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# Constants (each with its source; none are new policy)
# --------------------------------------------------------------------------

#: Rate-measurement warm-up before the rings are re-bounded (design §6).
WARMUP_S = 2.0
#: Ring headroom factor over the advertised window (review finding ops-10).
RING_HEADROOM = 1.5

#: Span band vs design nominal — same +/-30 % as
#: ``arm_constants.PLATE_CHAIN_TOL_FRAC`` (the probe must not import the
#: control repo — design D8 keeps the dependency graph one-way — so the value
#: is mirrored here with its citation).
SPAN_TOL_FRAC = 0.30

#: Inferred-vs-streamed body-frame delta thresholds for plates 0 and 5
#: (design §6 / review finding ops-0): warn at 10 deg, gate at 20 deg.
ALIGN_WARN_DEG = 10.0
ALIGN_EXIT_DEG = 20.0

#: x-candidate residual gate (design §4.1.3): past 30 deg the lock sits near
#: the 45 deg ambiguity boundary and must not be written or trusted.
X_RESIDUAL_EXIT_DEG = 30.0

#: Stillness bounds for lock generation.  The q bound mirrors
#: ``arm_constants.JOINT_PREFLIGHT_Q_SD_MAX`` (0.01 rad = 0.57 deg); the marker
#: bound is 2 mm — resting Motive marker jitter measures 0.2-1.5 mm
#: (``joint_verification.repeated_plate_report`` docstring), so 2 mm passes a
#: healthy rest and fails an arm that is visibly settling or being bumped.
PROBE_Q_SD_MAX_RAD = 0.01
PROBE_MARKER_SD_MAX_M = 0.002

#: A lock needs >= 1 s of rest frames with all four markers tracked (§4.1).
LOCK_REST_MIN_S = 1.0

#: Chain nominals fallback: the authoritative home is
#: ``UMArm_KINEMATICS.robot_params.PLATE_CHAIN_NOMINAL_M`` (design D8), which
#: is being introduced in the same campaign as this probe.  Until that package
#: is importable, these literals (identical to ``arm_constants`` and to the
#: fkine spans, ``docs/fkine_design.md`` §1) keep the span gate live.
_PLATE_CHAIN_NOMINAL_FALLBACK_M = (0.218868, 0.047871, 0.194540, 0.047382, 0.184111)

#: Bracket family angles phi_p (design §2): plates 0/2/4 are 0-deg family,
#: 1/3/5 are +45-deg CCW.  Plate 6's family is measured, not assumed — the
#: probe tries both and reports which fit (design D6; plate 6 was absent from
#: the volume on the 2026-08-11 live probe, so this path is skip-and-report).
FAMILY_PHI_RAD = {0: 0.0, 1: math.pi / 4, 2: 0.0, 3: math.pi / 4,
                  4: 0.0, 5: math.pi / 4}

#: NatNet data port, mirrored from the vendored client (``NatNetClient.py:78``)
#: for the --no-multicast refusal check; importing the SDK just for the number
#: would drag 150 kB of vendor code into every probe run.
_NATNET_DATA_PORT = 1511

EXIT_OK = 0
EXIT_NO_STREAM = 1
EXIT_LOCK_REFUSED = 2
EXIT_GATE = 3


# --------------------------------------------------------------------------
# Lazy imports of the parallel packages (design: degrade, never crash)
# --------------------------------------------------------------------------


def span_nominals() -> tuple[tuple, str]:
    """The five chain nominals and where they came from.

    Prefers ``UMArm_KINEMATICS.robot_params`` (the authoritative home, design
    D8), falls back to the local literals so the span gate never silently
    disappears.  Imported lazily: the sibling package is being written in
    parallel and its absence must not take the whole probe down.
    """
    root = os.path.dirname(_HERE)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from UMArm_KINEMATICS import robot_params  # noqa: PLC0415 (deliberate)
        return tuple(float(v) for v in robot_params.PLATE_CHAIN_NOMINAL_M), \
            "UMArm_KINEMATICS.robot_params"
    except Exception:
        return _PLATE_CHAIN_NOMINAL_FALLBACK_M, "built-in fallback literals"


def import_marker_frame():
    """The inference module, or ``None`` — it is being written in parallel."""
    try:
        from . import marker_frame  # noqa: PLC0415 (deliberate, see docstring)
        return marker_frame
    except ImportError:
        pass
    try:
        import marker_frame  # noqa: PLC0415
        return marker_frame
    except ImportError:
        return None


# --------------------------------------------------------------------------
# Small numerics
# --------------------------------------------------------------------------


def rot_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix, degrees, via the trace (clipped for noise)."""
    c = (float(np.trace(R[0:3, 0:3])) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def rz(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> scalar-last quaternion, Shepperd's branch method.

    Inverse of ``mocap_to_q.quat_xyzw_to_matrix`` (NatNet order), used only to
    write the poses CSV in the same quaternion convention Motive streams.  The
    four branches each divide by the largest of the four candidate magnitudes,
    so no branch ever divides by a vanishing quantity.
    """
    m = np.asarray(R, dtype=float)[0:3, 0:3]
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)


# --------------------------------------------------------------------------
# Transport-level analysis (pure functions over a MarkerWindow — no network,
# no marker_frame; everything here runs in reports-only mode too)
# --------------------------------------------------------------------------


def plate_positions(w: MarkerWindow) -> np.ndarray:
    """Streamed pivot positions, shape ``(n, 7, 3)``."""
    return w.streamed_poses[:, :, 0:3, 3]


def absent_plates(w: MarkerWindow) -> list[int]:
    """Plates whose streamed pose stayed the identity for the whole window.

    An identity row means Motive never placed the body once (the receiver's
    rows start at the identity and hold their last value), i.e. the body is
    absent from the volume — the expected state of plate 6 (design D6), a
    hard failure for anything else.  Distinct from the repeated-pose case,
    where a body *was* placed and then froze.
    """
    eye = np.eye(4)
    return [k for k in range(w.streamed_poses.shape[1])
            if np.array_equal(w.streamed_poses[:, k],
                              np.broadcast_to(eye, w.streamed_poses[:, k].shape))]


def repeated_pose_plates(w: MarkerWindow, absent: list[int]) -> tuple[list[int], list[int]]:
    """(untracked plates, moving plates) by the bit-identical repeated-pose test.

    Same logic as ``joint_verification.repeated_plate_report``: a tracked body
    jitters every frame (0.2-1.5 mm of marker noise), so a position that
    repeats *to the last bit* across a window in which any other plate moved is
    an untracked body, not a still one.  Unlike the simulator-facing original,
    plate 0 is not exempt here: on live hardware the base plate's markers
    jitter like everyone else's, and a frozen base is exactly the kind of
    failure a preflight exists to name.
    """
    pos = plate_positions(w)
    n_plates = pos.shape[1]
    candidates = [k for k in range(n_plates) if k not in absent]
    distinct = {k: len(np.unique(pos[:, k, :], axis=0)) for k in candidates}
    movers = [k for k in candidates if distinct[k] > 1]
    if not movers or pos.shape[0] < 8:
        return [], movers
    return [k for k in candidates if distinct[k] == 1], movers


def span_stats(w: MarkerWindow) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of the five consecutive u-joint spans, metres, over the window."""
    pos = plate_positions(w)[:, 0:6, :]
    d = np.linalg.norm(pos[:, 1:, :] - pos[:, :-1, :], axis=2)  # (n, 5)
    return d.mean(axis=0), (d.std(axis=0, ddof=1) if d.shape[0] > 1
                            else np.zeros(5))


def span_gate(mean_spans: np.ndarray, nominals: tuple,
              absent: list[int]) -> tuple[bool, list[str]]:
    """The pivot-placement gate (review finding ops-1), with per-gap headroom."""
    lines = []
    failed = False
    for i, (got, want) in enumerate(zip(mean_spans, nominals)):
        lo, hi = want * (1.0 - SPAN_TOL_FRAC), want * (1.0 + SPAN_TOL_FRAC)
        headroom_mm = min(got - lo, hi - got) * 1000.0
        ok = lo <= got <= hi
        note = ""
        involved_absent = [k for k in (i, i + 1) if k in absent]
        if involved_absent:
            note = (f"  <- plate(s) {involved_absent} never updated this window: "
                    f"tracking, not pivot placement")
        lines.append(f"  span u{i}-u{i + 1}: {got:.4f} m vs {want:.4f} m nominal "
                     f"[band {lo:.4f}..{hi:.4f}], headroom {headroom_mm:+.0f} mm"
                     f"{' OK' if ok else ' OUT OF BAND'}{note}")
        if not ok:
            failed = True
    return failed, lines


def per_plate_marker_counts(w: MarkerWindow) -> dict[int, dict]:
    """Per plate: frames seen, marker counts observed, count stability."""
    out: dict[int, dict] = {}
    for markers in w.markers:
        if markers is None:
            continue
        for plate, arr in markers.items():
            rec = out.setdefault(plate, {"frames": 0, "counts": {}})
            rec["frames"] += 1
            n = int(arr.shape[0])
            rec["counts"][n] = rec["counts"].get(n, 0) + 1
    for rec in out.values():
        rec["stable"] = len(rec["counts"]) == 1
    return out


def _four_marker_frames(w: MarkerWindow, plate: int,
                        require_tracked: bool) -> np.ndarray:
    """Stack of ``(4, 3)`` marker arrays for frames usable as rest evidence.

    ``require_tracked`` demands all four labeled tracked flags; without it
    (the ``--allow-unlabeled`` regime) a 4-marker set is taken at face value —
    which is exactly the assumption that regime stamps into its outputs.
    """
    rows = []
    for markers, flags in zip(w.markers, w.flags):
        if markers is None or plate not in markers:
            continue
        arr = markers[plate]
        if arr.shape[0] != 4:
            continue
        if require_tracked:
            if flags is None or plate not in flags or int(flags[plate].sum()) != 4:
                continue
        rows.append(arr)
    return np.stack(rows) if rows else np.empty((0, 4, 3))


def marker_position_sd(stack: np.ndarray) -> float:
    """Worst per-marker position sd (metres) over a ``(n, 4, 3)`` rest stack."""
    if stack.shape[0] < 2:
        return float("nan")
    sd = stack.std(axis=0, ddof=1)           # (4, 3)
    return float(np.linalg.norm(sd, axis=1).max())


def rectangle_stats(stack: np.ndarray) -> dict | None:
    """Rest-window rectangle sanity stats for one plate (design §6 report).

    Diagonals are identified as the two longest of the six chords — for the
    *report* that longest-chord rule is exactly the "recorded sanity stat" the
    design allows (§4.1.1); the lock itself uses the cyclic-order rule and
    lives in ``marker_frame``.  Units: mm and degrees.
    """
    if stack.shape[0] == 0:
        return None
    m = stack.mean(axis=0)                    # (4, 3) mean marker positions
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    chords = sorted(((float(np.linalg.norm(m[a] - m[b])), a, b) for a, b in pairs),
                    reverse=True)
    (d1, a1, b1), (d2, a2, b2) = chords[0], chords[1]
    mid1, mid2 = (m[a1] + m[b1]) / 2.0, (m[a2] + m[b2]) / 2.0
    v1 = (m[b1] - m[a1]) / d1
    v2 = (m[b2] - m[a2]) / d2
    crossing = math.degrees(math.acos(min(1.0, abs(float(v1 @ v2)))))
    centred = m - m.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    return {
        "frames": int(stack.shape[0]),
        "diagonal_lengths_mm": [d1 * 1000.0, d2 * 1000.0],
        "midpoint_separation_mm": float(np.linalg.norm(mid1 - mid2)) * 1000.0,
        "diagonal_crossing_deg": 90.0 - crossing if crossing > 45.0 else crossing,
        "out_of_plane_rms_mm": float(sv[2]) / math.sqrt(4.0) * 1000.0,
    }


# --------------------------------------------------------------------------
# Inference section (the only marker_frame consumer; degrades to a message)
# --------------------------------------------------------------------------

#: Lock-builder names tried, first hit wins.  ``marker_frame`` is being written
#: in parallel against the same design; this list plus the keyword fallbacks in
#: :func:`_build_lock` is the whole coupling surface, and any mismatch degrades
#: to reports-only with the exception printed rather than crashing a preflight.
_LOCK_BUILDER_NAMES = ("compute_plate_lock", "make_plate_lock",
                       "plate_lock_from_rest", "build_plate_lock", "lock_plate")
_X_RESIDUAL_ATTRS = ("x_residual_deg", "x_residual_rad", "x_residual")


def _build_lock(fn, rest_stack: np.ndarray, plate: int, u_up: np.ndarray,
                streamed_rot: np.ndarray, x_mode: str = "streamed"):
    last: Exception | None = None
    for kwargs in (dict(plate=plate, u_up=u_up, streamed_rot=streamed_rot,
                        x_mode=x_mode),
                   dict(plate=plate, u_up=u_up, streamed_rot=streamed_rot),
                   dict(plate=plate, u_up=u_up, streamed_R=streamed_rot),
                   dict(plate=plate, u_up=u_up)):
        try:
            return fn(rest_stack, **kwargs)
        except TypeError as exc:
            last = exc
    raise last if last is not None else TypeError("no lock builder call worked")


def _lock_x_residual_deg(lock) -> float | None:
    for name in _X_RESIDUAL_ATTRS:
        val = getattr(lock, name, None)
        if val is not None:
            return math.degrees(float(val)) if name.endswith("_rad") else float(val)
    return None


def inference_section(w: MarkerWindow, absent: list[int],
                      allow_unlabeled: bool, flags_known: bool,
                      x_mode: str = "streamed") -> dict:
    """Locks + inferred-vs-streamed deltas, or a clear unavailability message.

    Returns a dict with ``available`` (bool), ``message``, and when available:
    ``locks`` (plate -> lock object), ``deltas`` (plate -> dict of median
    body-frame angle delta [deg], origin delta [mm], frames, phi used),
    ``x_residuals_deg`` (plate -> float | None).
    """
    out: dict = {"available": False, "message": "", "locks": {}, "deltas": {},
                 "x_residuals_deg": {}}
    mf = import_marker_frame()
    if mf is None:
        out["message"] = ("marker_frame not importable -- inference sections "
                          "skipped (reports-only); the alignment and "
                          "x-residual gates were NOT evaluated")
        return out
    builder = None
    for name in _LOCK_BUILDER_NAMES:
        fn = getattr(mf, name, None)
        if callable(fn):
            builder = fn
            break
    if builder is None or not callable(getattr(mf, "infer_plate_frame", None)):
        out["message"] = ("marker_frame importable but exposes none of "
                          f"{_LOCK_BUILDER_NAMES} + infer_plate_frame -- "
                          "inference sections skipped (reports-only)")
        return out
    if 0 in absent or 5 in absent:
        out["message"] = ("plate 0 or 5 absent from the stream -- no u_up "
                          "reference, inference sections skipped")
        return out
    if not flags_known and not allow_unlabeled:
        out["message"] = ("labeled markers absent and --allow-unlabeled not "
                          "given -- inference on assumed markers refused")
        return out

    pos = plate_positions(w)
    # Up from the arm itself, never from world axes (design §2 / review ops-4):
    # the arm hangs, so base minus last-distal points up.
    u_up = pos[:, 0, :].mean(axis=0) - pos[:, 5, :].mean(axis=0)
    u_up = u_up / np.linalg.norm(u_up)
    mid = len(w) // 2

    try:
        for plate in range(w.streamed_poses.shape[1]):
            if plate in absent:
                continue
            stack = _four_marker_frames(w, plate, require_tracked=flags_known)
            if stack.shape[0] < 10:
                continue
            r_streamed = w.streamed_poses[mid, plate, 0:3, 0:3]
            lock = _build_lock(builder, stack, plate, u_up, r_streamed,
                               x_mode=x_mode)
            out["locks"][plate] = lock
            out["x_residuals_deg"][plate] = _lock_x_residual_deg(lock)

            # Per-frame inferred body frame vs the streamed pose.  Plate 6's
            # family is measured, not assumed (design D6): both are tried and
            # the better fit reported.
            phis = ((FAMILY_PHI_RAD[plate],) if plate in FAMILY_PHI_RAD
                    else (0.0, math.pi / 4))
            best = None
            for phi in phis:
                rz_neg = rz(-phi)
                ang, org = [], []
                for k, (markers, flags) in enumerate(zip(w.markers, w.flags)):
                    if markers is None or plate not in markers:
                        continue
                    arr = markers[plate]
                    if arr.shape[0] != 4:
                        continue
                    f = (flags[plate] if flags is not None and plate in flags
                         else np.ones(arr.shape[0], dtype=np.uint8))
                    res = mf.infer_plate_frame(arr, f, lock)
                    T = res[0] if isinstance(res, tuple) else res
                    if T is None:
                        continue
                    T = np.asarray(T, dtype=float)
                    r_body = T[0:3, 0:3] @ rz_neg
                    r_s = w.streamed_poses[k, plate, 0:3, 0:3]
                    ang.append(rot_angle_deg(r_s.T @ r_body))
                    org.append(float(np.linalg.norm(
                        w.streamed_poses[k, plate, 0:3, 3] - T[0:3, 3])) * 1000.0)
                if ang:
                    cand = {"angle_deg": float(np.median(ang)),
                            "origin_mm": float(np.median(org)),
                            "frames": len(ang), "phi_deg": math.degrees(phi)}
                    if best is None or cand["angle_deg"] < best["angle_deg"]:
                        best = cand
            if best is not None:
                out["deltas"][plate] = best
        out["available"] = True
        out["message"] = "marker_frame inference ran"
    except Exception as exc:
        # Parallel-development seam: an API drift must degrade, not crash a
        # preflight.  The gates that depend on this section are then reported
        # as NOT evaluated.
        out["available"] = False
        out["locks"].clear()
        out["deltas"].clear()
        out["message"] = (f"marker_frame unusable here ({exc!r}) -- inference "
                          "sections skipped; alignment and x-residual gates "
                          "NOT evaluated")
    return out


# --------------------------------------------------------------------------
# Lock refusals (design §6 / review finding ops-3)
# --------------------------------------------------------------------------


def lock_refusals(qwin: MocapWindow, w: MarkerWindow, absent: list[int],
                  untracked: list[int], fps: float, flags_known: bool,
                  allow_unlabeled: bool, inference: dict) -> list[str]:
    """Every reason locks.json must not be written today, each named."""
    reasons: list[str] = []
    if not inference["available"]:
        reasons.append(f"lock generation needs marker_frame: {inference['message']}")
    if len(qwin) >= 2:
        q_sd = float(qwin.q.std(axis=0, ddof=1).max())
        if q_sd > PROBE_Q_SD_MAX_RAD:
            reasons.append(
                f"stillness: worst q channel sd {q_sd:.4f} rad exceeds "
                f"{PROBE_Q_SD_MAX_RAD} rad -- the arm is not at rest (or a "
                f"plate is flapping); locks demand a rest window")
    else:
        reasons.append("stillness: no usable q over the window, so there is "
                       "no rest evidence at all")
    if untracked:
        reasons.append(f"repeated-pose test: plate(s) {untracked} reported "
                       f"bit-identical positions all window -- untracked")
    if not flags_known and not allow_unlabeled:
        reasons.append("labeled markers absent: per-marker tracked flags are "
                       "unknown; pass --allow-unlabeled to accept 4-marker "
                       "sets at face value (the regime is stamped in the JSON)")
    need = max(2, math.ceil(LOCK_REST_MIN_S * fps))
    for plate in range(w.streamed_poses.shape[1]):
        if plate in absent:
            continue        # absent plates (plate 6 today) are skip-and-report
        stack = _four_marker_frames(w, plate,
                                    require_tracked=flags_known)
        if stack.shape[0] < need:
            reasons.append(
                f"plate {plate}: only {stack.shape[0]} frames with all four "
                f"markers tracked (need >= {need} = {LOCK_REST_MIN_S:g} s at "
                f"{fps:.0f} Hz) -- fix marker visibility before locking")
            continue
        m_sd = marker_position_sd(stack)
        if not (m_sd <= PROBE_MARKER_SD_MAX_M):   # NaN-rejecting comparison
            reasons.append(
                f"plate {plate}: worst marker position sd "
                f"{m_sd * 1000.0:.2f} mm exceeds "
                f"{PROBE_MARKER_SD_MAX_M * 1000.0:.1f} mm -- not rest")
    return reasons


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def write_capture_csvs(out_dir: str, w: MarkerWindow) -> tuple[str, str]:
    """The static capture, in the campaign's long CSV formats (design §7).

    ``probe_markers.csv``: ``t_s, frame, epoch, plate, marker, x, y, z,
    tracked`` with ``tracked`` in {1, 0, ?, -} — ``?`` = labeled markers absent
    (flags unknown), ``-`` = marker sets absent that frame (one placeholder row
    with plate=-1 keeps the frame visible; *not streamed* must stay
    distinguishable from *not tracked*, review finding ops-9).
    ``probe_poses.csv``: the streamed plate poses, quaternions scalar-last as
    Motive streams them — the fkine benchmark's streamed-base regime input.
    """
    t0 = float(w.t[0]) if len(w) else 0.0
    markers_path = os.path.join(out_dir, "probe_markers.csv")
    with open(markers_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("t_s,frame,epoch,plate,marker,x,y,z,tracked\n")
        for k in range(len(w)):
            t_s = w.t[k] - t0
            frame, epoch = int(w.frame_no[k]), int(w.mapping_epoch[k])
            markers, flags = w.markers[k], w.flags[k]
            if markers is None:
                fh.write(f"{t_s:.6f},{frame},{epoch},-1,-1,nan,nan,nan,-\n")
                continue
            for plate in sorted(markers):
                arr = markers[plate]
                pf = None if flags is None else flags.get(plate)
                for j in range(arr.shape[0]):
                    tracked = "?" if pf is None else str(int(pf[j]))
                    fh.write(f"{t_s:.6f},{frame},{epoch},{plate},{j},"
                             f"{arr[j, 0]:.6f},{arr[j, 1]:.6f},{arr[j, 2]:.6f},"
                             f"{tracked}\n")
    poses_path = os.path.join(out_dir, "probe_poses.csv")
    with open(poses_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("t_s,frame,plate,qx,qy,qz,qw,x,y,z\n")
        for k in range(len(w)):
            t_s = w.t[k] - t0
            frame = int(w.frame_no[k])
            for plate in range(w.streamed_poses.shape[1]):
                T = w.streamed_poses[k, plate]
                q = matrix_to_quat_xyzw(T)
                p = T[0:3, 3]
                fh.write(f"{t_s:.6f},{frame},{plate},"
                         f"{q[0]:.7f},{q[1]:.7f},{q[2]:.7f},{q[3]:.7f},"
                         f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")
    return markers_path, poses_path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    return repr(obj)


def _lock_payload(lock):
    if dataclasses.is_dataclass(lock) and not isinstance(lock, type):
        return dataclasses.asdict(lock)
    if hasattr(lock, "__dict__"):
        return dict(vars(lock))
    return {"repr": repr(lock)}


def write_locks(path: str, inference: dict, meta: dict) -> None:
    """locks.json with the §4.1 validity binding embedded.

    Consumers (benchmark, probe re-use) re-derive the embedded rest stats from
    the first rest frames of whatever data they are given and refuse on
    mismatch beyond noise — a lock is valid only for the Motive session it was
    captured in (review finding ops-5).
    """
    payload = {
        "meta": meta,
        "plates": {str(p): _lock_payload(lock)
                   for p, lock in inference["locks"].items()},
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
        fh.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="UMArm mocap preflight probe (marker_frame_design.md sec. 6)")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="capture window length (default 30)")
    ap.add_argument("--server-ip", default=mc.DEFAULT_SERVER_IP)
    ap.add_argument("--client-ip", default=mc.DEFAULT_CLIENT_IP)
    ap.add_argument("--no-multicast", action="store_true",
                    help="unicast mode; refused while another local NatNet "
                         "client streams (concurrent probe+campaign is "
                         "multicast-only)")
    ap.add_argument("--out-dir", default=None,
                    help="write probe_report.json + capture CSVs here")
    ap.add_argument("--lock-out", default=None,
                    help="write per-plate locks JSON here (subject to the "
                         "refusal list)")
    ap.add_argument("--x-mode", choices=("streamed", "diagonal45"),
                    default="streamed",
                    help="azimuth zero of the lock frames: 'streamed' anchors "
                         "the x-axis to the streamed rest orientation (the "
                         "2026-08-11 arbitration showed the marker-arm "
                         "azimuths sit up to ~16 deg off the designed 45 deg, "
                         "so this is the mechanism-faithful default; the "
                         "offset is recorded as x_residual_deg); 'diagonal45' "
                         "is the design-spec definition, x = nearest diagonal "
                         "rotated +45 deg")
    ap.add_argument("--allow-unlabeled", action="store_true",
                    help="permit lock generation without labeled markers; "
                         "stamps the flags-unknown regime into the JSON")
    ap.add_argument("--rb-id-base", type=int, default=mc.RIGID_BODY_ID_MASK,
                    help="Motive streaming id of plate 0 (default %d, the "
                         "RS485 arm's committed block).  The CAN UMArm is "
                         "briefed as 2000; verify against Motive before "
                         "trusting either." % mc.RIGID_BODY_ID_MASK)
    ap.add_argument("--n-bodies", type=int, default=mc.N_USED_RIGID_BODIES,
                    help="how many plates that block carries (default %d; the "
                         "CAN UMArm is briefed as 6)"
                         % mc.N_USED_RIGID_BODIES)
    return ap.parse_args(argv)


def _unicast_refusal() -> str | None:
    """Best-effort concurrent-client check for --no-multicast.

    A multicast NatNet client on this PC holds UDP ``1511`` (the SDK binds it
    with SO_REUSEADDR; a plain bind then fails with WSAEADDRINUSE on Windows).
    A remote client, or a local *unicast* one (ephemeral port), is invisible to
    this check — the campaign runs on this bench PC, which is the concurrency
    case the design names.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", _NATNET_DATA_PORT))
    except OSError:
        return (f"--no-multicast refused: UDP port {_NATNET_DATA_PORT} is held "
                f"by another NatNet client on this PC (a campaign?).  "
                f"Concurrent probe+campaign is multicast-only.")
    finally:
        s.close()
    return None


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.no_multicast:
        refusal = _unicast_refusal()
        if refusal is not None:
            print(refusal)
            return EXIT_NO_STREAM

    nominal_cap = math.ceil(args.seconds * RING_HEADROOM * mc.NOMINAL_RATE_HZ)
    rx = MocapRx(server_ip=args.server_ip, client_ip=args.client_ip,
                 use_multicast=not args.no_multicast,
                 ring_capacity=nominal_cap, marker_ring_capacity=nominal_cap,
                 rb_id_base=args.rb_id_base, n_bodies=args.n_bodies)
    print(f"probe: connecting to {args.server_ip} from {args.client_ip} "
          f"({'multicast' if not args.no_multicast else 'unicast'})")
    rx.start()
    try:
        # Warm-up: measure the real rate, then re-bound the rings so the
        # advertised window actually fits (review finding ops-10).
        time.sleep(WARMUP_S)
        fps0 = rx.get_state().fps
        if fps0 > 0.0:
            cap = math.ceil(args.seconds * RING_HEADROOM * fps0)
            if cap > nominal_cap:
                rx.resize_rings(ring_capacity=cap, marker_ring_capacity=cap)
            print(f"probe: measured {fps0:.1f} Hz after {WARMUP_S:g} s warm-up; "
                  f"rings sized for {max(cap, nominal_cap)} frames")
        else:
            print(f"probe: no frames after {WARMUP_S:g} s warm-up "
                  f"(rings stay at the nominal {nominal_cap})")
        rx.clear_history()

        t0 = time.monotonic()
        end = t0 + args.seconds
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.25, remaining))
            if rx.get_state().stale and time.monotonic() - t0 > 2.0:
                print("probe: stream went stale mid-capture; reporting what "
                      "arrived")
                break
        t1 = time.monotonic()
        qwin = rx.snapshot_window(t0, t1)
        mwin = rx.snapshot_marker_window(t0, t1)
        health = rx.marker_health()
        state = rx.get_state()
    finally:
        rx.stop()

    if len(mwin) == 0:
        print("probe: no frames arrived at all -- is Motive streaming, and is "
              f"{args.client_ip} the camera-network interface?  "
              f"(last_error={state.last_error!r})")
        return EXIT_NO_STREAM

    return _report_and_gate(args, qwin, mwin, health, state)


def _report_and_gate(args, qwin: MocapWindow, mwin: MarkerWindow,
                     health, state) -> int:
    fps = mwin.fps if len(mwin) >= 2 else state.fps
    flags_known = health.labeled_frames > 0
    absent = absent_plates(mwin)
    untracked, movers = repeated_pose_plates(mwin, absent)
    nominals, nominal_src = span_nominals()
    mean_spans, sd_spans = span_stats(mwin)
    span_failed, span_lines = span_gate(mean_spans, nominals, absent)
    counts = per_plate_marker_counts(mwin)
    inference = inference_section(mwin, absent, args.allow_unlabeled,
                                  flags_known, x_mode=args.x_mode)

    print()
    print("== stream health ==")
    print(f"  frames {state.frames} (valid q {state.valid_frames}), "
          f"measured {fps:.1f} Hz, window {mwin.duration:.1f} s "
          f"({len(mwin)} marker-ring samples), stale={state.stale}")
    gaps = int(np.sum(np.diff(mwin.frame_no) > 1)) if len(mwin) >= 2 else 0
    print(f"  dropped-frame gaps in window: {gaps}; last_error={state.last_error!r}")

    print("== marker transport ==")
    print(f"  mocap_data frames {health.mocap_data_frames}, labeled-marker "
          f"frames {health.labeled_frames} "
          f"({'labeled regime' if flags_known else 'FLAGS UNKNOWN regime'})")
    print(f"  last frame: {health.last_marker_set_count} marker sets, "
          f"{health.last_labeled_marker_count} labeled markers")
    mapping_txt = {name.decode("utf-8", "replace"): plate
                   for name, plate in sorted(health.plate_names.items())}
    print(f"  mapping (epoch {health.mapping_epoch}): {mapping_txt}")
    print(f"  marker_id-1 == asset-index correspondence: "
          f"{health.id_corr_ok} ok / {health.id_corr_bad} violations "
          f"{'(assumption holds)' if health.id_corr_bad == 0 else '(FALLBACK: position matching in use)'}")
    print(f"  occluded encoding observed: occluded={health.occluded_seen}, "
          f"model_solved={health.model_solved_seen}, "
          f"point_cloud_solved={health.point_cloud_solved_seen}")

    print("== plates ==")
    if absent:
        for k in absent:
            tone = "expected (D6): skip-and-report" if k == 6 else "PROBLEM"
            print(f"  plate {k}: ABSENT from the stream all window -- {tone}")
    if untracked:
        print(f"  plate(s) {untracked}: bit-identical streamed position all "
              f"window while {movers} moved -- untracked (fix occlusion in "
              f"Motive before believing any q)")
    rect: dict[int, dict | None] = {}
    for plate in sorted(counts):
        rec = counts[plate]
        stack = _four_marker_frames(mwin, plate, require_tracked=flags_known)
        rect[plate] = rectangle_stats(stack)
        m_sd = marker_position_sd(stack)
        sd_txt = "n/a" if math.isnan(m_sd) else f"{m_sd * 1000.0:.2f} mm"
        print(f"  plate {plate}: {rec['frames']} frames, marker counts "
              f"{rec['counts']}{' (stable)' if rec['stable'] else ' (UNSTABLE)'}, "
              f"worst marker sd {sd_txt}")
        if rect[plate] is not None:
            r = rect[plate]
            print(f"    rectangle: diagonals {r['diagonal_lengths_mm'][0]:.1f}/"
                  f"{r['diagonal_lengths_mm'][1]:.1f} mm, midpoint sep "
                  f"{r['midpoint_separation_mm']:.2f} mm, crossing off-90 "
                  f"{r['diagonal_crossing_deg']:.1f} deg, out-of-plane RMS "
                  f"{r['out_of_plane_rms_mm']:.2f} mm")

    print(f"== chain spans (nominals from {nominal_src}) ==")
    for line in span_lines:
        print(line)

    print("== inference (marker_frame) ==")
    print(f"  {inference['message']}")
    align_failed = False
    align_warned = False
    x_failed = False
    if inference["available"]:
        for plate in sorted(inference["deltas"]):
            d = inference["deltas"][plate]
            xr = inference["x_residuals_deg"].get(plate)
            xr_txt = "not exposed by marker_frame" if xr is None else f"{xr:.1f} deg"
            print(f"  plate {plate}: inferred-vs-streamed body-frame delta "
                  f"{d['angle_deg']:.2f} deg, origin {d['origin_mm']:.1f} mm "
                  f"({d['frames']} frames, phi={d['phi_deg']:.0f} deg); "
                  f"x-residual {xr_txt}")
            if xr is not None and xr > X_RESIDUAL_EXIT_DEG:
                x_failed = True
            if plate in (0, 5):
                if d["angle_deg"] > ALIGN_EXIT_DEG:
                    align_failed = True
                elif d["angle_deg"] >= ALIGN_WARN_DEG:
                    align_warned = True
        if align_warned and not align_failed:
            print(f"  WARNING: plate 0/5 delta >= {ALIGN_WARN_DEG:g} deg -- "
                  f"Motive alignment is drifting; re-align before it gates")

    # ---- gates (exit 3, named) ------------------------------------------
    gate_msgs = []
    if untracked or [k for k in absent if k != 6]:
        # Found live 2026-08-12: plates 1+4 streamed bit-identical poses all
        # night and the probe said "clean" — the untracked report existed but
        # never gated, so a campaign's own stillness gate was the first thing
        # to refuse.  An untracked or absent arm plate invalidates every q
        # this volume produces; that is a gate, not a footnote.
        bad_abs = [k for k in absent if k != 6]
        gate_msgs.append(
            f"TRACKING GATE: plate(s) {sorted(set(untracked) | set(bad_abs))} "
            f"are untracked (bit-identical streamed pose) or absent from the "
            f"stream -- fix the occlusion / re-enable the asset in Motive; "
            f"no campaign should start until every arm plate jitters")
    if span_failed:
        gate_msgs.append(
            "SPAN GATE: a chain span sits outside the +/-30% nominal band -- "
            "re-set the rigid-body pivot to the marker centroid / diagonal "
            "intersection in Motive (per-gap headroom above)")
    if align_failed:
        gate_msgs.append(
            f"ALIGNMENT GATE: plate 0/5 streamed orientation vs marker-"
            f"inferred body frame disagrees by > {ALIGN_EXIT_DEG:g} deg -- "
            f"Motive rigid-body alignment moved; fix the asset alignment, do "
            f"NOT touch tubes")
    if x_failed:
        gate_msgs.append(
            f"X-RESIDUAL GATE: an x-candidate residual exceeds "
            f"{X_RESIDUAL_EXIT_DEG:g} deg -- the lock sits near the 45 deg "
            f"ambiguity boundary; check the plate's streamed orientation and "
            f"marker labels before trusting any lock")

    # ---- lock writing (refusal list, exit 2) ----------------------------
    refusals: list[str] = []
    lock_written = None
    if args.lock_out is not None:
        refusals = lock_refusals(qwin, mwin, absent, untracked, fps,
                                 flags_known, args.allow_unlabeled, inference)
        if gate_msgs:
            refusals.append("gates failed (see above); a gated volume must "
                            "not mint locks")
        if refusals:
            print("== lock writing REFUSED ==")
            for r in refusals:
                print(f"  - {r}")
        else:
            meta = {
                "captured_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": args.seconds,
                "fps_measured": fps,
                "regime": ("labeled" if flags_known
                           else "unlabeled (--allow-unlabeled)"),
                "streaming_ids_seen": [rx.rb_id_base + p
                                       for p in range(mwin.streamed_poses.shape[1])
                                       if p not in absent],
                "rb_id_base": rx.rb_id_base,
                "n_bodies": rx.n_bodies,
                "mapping": mapping_txt,
                "mapping_epoch": health.mapping_epoch,
                "per_plate_marker_counts": {str(p): counts[p]["counts"]
                                            for p in sorted(counts)},
                "rectangle_stats": {str(p): rect[p] for p in sorted(rect)
                                    if rect[p] is not None},
                "marker_id_correspondence": {"ok": health.id_corr_ok,
                                             "bad": health.id_corr_bad},
            }
            write_locks(args.lock_out, inference, meta)
            lock_written = args.lock_out
            print(f"== locks written to {args.lock_out} ==")

    # ---- files -----------------------------------------------------------
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        markers_csv, poses_csv = write_capture_csvs(args.out_dir, mwin)
        report = {
            "fps_measured": fps,
            "frames": state.frames,
            "valid_q_frames": state.valid_frames,
            "flags_known": flags_known,
            "absent_plates": absent,
            "untracked_plates": untracked,
            "spans_mean_m": mean_spans.tolist(),
            "spans_sd_m": sd_spans.tolist(),
            "span_nominals_m": list(nominals),
            "span_nominals_source": nominal_src,
            "marker_counts": {str(p): counts[p] for p in sorted(counts)},
            "rectangle_stats": {str(p): rect[p] for p in sorted(rect)},
            "mapping": mapping_txt,
            "mapping_epoch": health.mapping_epoch,
            "id_correspondence": {"ok": health.id_corr_ok,
                                  "bad": health.id_corr_bad},
            "occluded_encoding": {
                "occluded": health.occluded_seen,
                "model_solved": health.model_solved_seen,
                "point_cloud_solved": health.point_cloud_solved_seen},
            "inference": {
                "available": inference["available"],
                "message": inference["message"],
                "deltas": {str(p): d for p, d in inference["deltas"].items()},
                "x_residuals_deg": {str(p): v for p, v
                                    in inference["x_residuals_deg"].items()}},
            "gates_failed": gate_msgs,
            "lock_refusals": refusals,
            "lock_written": lock_written,
            "csv": {"markers": markers_csv, "poses": poses_csv},
        }
        report_path = os.path.join(args.out_dir, "probe_report.json")
        with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, default=_json_default)
            fh.write("\n")
        print(f"probe: wrote {report_path}, {markers_csv}, {poses_csv}")

    # ---- verdict ---------------------------------------------------------
    if gate_msgs:
        print("== PROBE FAILED (exit 3) ==")
        for m in gate_msgs:
            print(f"  {m}")
        return EXIT_GATE
    if args.lock_out is not None and refusals:
        return EXIT_LOCK_REFUSED
    print("== probe clean ==")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
