"""Live ``q`` from **marker positions**, not from Motive's rigid-body pivots.

``marker_frame_design.md`` §4 built the inference — rest-locked template plus
per-frame Kabsch registration — and the 2026-08-11 campaign verified it offline
against 23,560 frames.  It was never wired into the *live* path: until this
module, :meth:`MocapRx._solve_q` reconstructed ``q`` from the poses Motive
streams, and every closed-loop session on this arm therefore steered on the
rigid-body pivot.

WHY THAT IS NOT GOOD ENOUGH, in the operator's words: *the mocap centre
position definition may change during the auto-refine process, but the marker
position stays true.*  Motive is free to move a body's pivot whenever it
re-solves the asset, and it does so without telling any client.  A pivot that
shifts 2 mm mid-session shifts every joint downstream of it, and nothing in a
control loop can distinguish that from the arm having moved — the loop will
dutifully drive the arm to cancel a definition change.  The markers cannot move
relative to the plate they are screwed to, so a pose registered onto *them* is
the pose of the plate and nothing else.

WHAT THIS MODULE IS

* :class:`MarkerMocap` — a :class:`~UMArm_MOCAP.mocap_rx.MocapRx` whose
  ``q`` comes from :func:`~UMArm_MOCAP.marker_frame.infer_all` +
  :func:`~UMArm_MOCAP.marker_frame.q_from_frames`.  Drop-in: same
  ``start``/``stop``/``get_q``/``get_state``, so ``BusRuntime`` cannot tell the
  difference and neither can the verification harness.
* :func:`mint_locks` — build this Motive session's plate locks from a live rest
  capture, so a session needs no hand-made lock file and no operator step.
  Locks are session-scoped: Motive's asset definitions (and the marker labels
  they hand out) are, which is the whole reason a lock cannot be checked in.
* :func:`save_locks` / :func:`load_locks` — the on-disk form, which is the same
  JSON shape ``mocap_probe.py --lock-out`` writes, so the two are
  interchangeable.

WHAT IT DOES *NOT* DO.  It never falls back to the streamed pose silently.  A
frame whose markers do not solve produces ``None``, exactly as a degenerate
streamed frame does, and the receiver's ``q_stale`` flag is what a control loop
reads to notice.  Falling back would mean the loop quietly changed which
definition of "the arm" it was tracking, mid-move, which is worse than a gap.
``fallback_to_streamed=True`` exists for offline comparison work and says so.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import mocap_constants as mc
from .marker_frame import (FAMILY_PHI_RAD, PROXIMAL_ORDERS, PlateLock,
                           compute_plate_lock, infer_all, q_from_frames)
from .mocap_rx import MocapRx

#: The plates ``q_from_frames`` needs.  Plate 6 (the end effector) is absent
#: from this lab's volume and is not part of ``q``.
REQUIRED_PLATES = tuple(range(6))

#: Seconds of rest evidence :func:`mint_locks` asks for by default.  The design
#: wants >= 1 s of stillness; 3 s at 120 Hz is ~360 frames, enough that the
#: template mean is noise-free without making a session wait.
DEFAULT_REST_S = 3.0

#: Fewest usable rest frames per plate before :func:`mint_locks` refuses.  Ten
#: is ``mocap_probe``'s own floor, kept identical so the two agree about what
#: counts as evidence.
MIN_REST_FRAMES = 10

#: How still the arm must be across the rest window for the capture to count,
#: metres of per-marker standard deviation.  The measured plates sit at
#: 0.02-0.05 mm (probe, 2026-08-15), so 1 mm is three orders of margin against
#: noise and still catches an arm that is drifting or being touched.
REST_STILLNESS_SD_M = 0.001


@dataclass
class SolveStats:
    """What the marker solve has been doing, since the last :meth:`reset`.

    Read it after a run rather than trusting that "it worked": a session that
    solved 60 % of its frames from markers and let the rest lapse is a session
    whose control loop was running at 72 Hz of fresh pose, and the only place
    that shows up is here.
    """

    frames: int = 0
    solved: int = 0
    fallback: int = 0
    unsolved: int = 0
    #: plate -> gate name -> count, straight from :class:`FrameQuality.reason`.
    reasons: dict = field(default_factory=dict)
    #: plate -> most recent Kabsch RMS residual, metres.
    last_rms_m: dict = field(default_factory=dict)

    @property
    def solved_fraction(self) -> float:
        return 0.0 if self.frames == 0 else self.solved / self.frames

    def summary(self) -> str:
        worst = ""
        if self.reasons:
            flat = [(n, f"plate {p}: {r}")
                    for p, d in self.reasons.items() for r, n in d.items()]
            if flat:
                worst = "; worst gate " + max(flat)[1] + f" x{max(flat)[0]}"
        return (f"{self.solved}/{self.frames} frames solved from markers "
                f"({100.0 * self.solved_fraction:.1f} %), {self.fallback} fell "
                f"back, {self.unsolved} produced no q{worst}")


class MarkerMocap(MocapRx):
    """:class:`MocapRx` whose ``q`` is registered onto the markers.

    *locks* maps plate index -> :class:`PlateLock`; plates 0..5 must all be
    present, because ``q_from_frames`` reads all six and an identity row would
    be silently reconstructed into a joint angle rather than rejected.

    Everything else — transport, rings, health, ``capture_rest`` — is inherited
    unchanged, so the marker path is exactly one method different from the
    streamed one and the two can be compared frame-for-frame.
    """

    def __init__(self, locks: dict, *, fallback_to_streamed: bool = False,
                 phis=None, order: str = "xy", **kwargs) -> None:
        super().__init__(**kwargs)
        #: Proximal-pair composition order handed to ``q_from_frames``.  Must
        #: match whatever forward model consumes this ``q``; see
        #: ``UMArm_KINEMATICS.fkine`` for why the CAN arm needs ``"yx"``.
        self.order = str(order)
        #: Per-plate bracket azimuths handed to
        #: :func:`~UMArm_MOCAP.marker_frame.q_from_frames`; ``None`` means the
        #: ``FAMILY_PHI_RAD`` defaults, which is the RS485 arm's answer.  The
        #: CAN arm's brackets are 45 deg round from those (its marker arms lie
        #: *along* the revolute axes, measured 2026-08-21), and a wrong azimuth
        #: here does not fail — it yields a smooth, repeatable, wrong ``q`` —
        #: so the parameter exists to make the choice explicit at the call site
        #: rather than implicit in a module default.
        self.phis = None if phis is None else np.asarray(phis, dtype=float)
        if self.phis is not None and self.phis.shape != (len(REQUIRED_PLATES),):
            raise ValueError(
                f"phis must have shape ({len(REQUIRED_PLATES)},); "
                f"got {self.phis.shape}")
        self.locks = {int(p): lk for p, lk in locks.items()}
        missing = [p for p in REQUIRED_PLATES if p not in self.locks]
        if missing:
            raise ValueError(
                f"marker inference needs a lock for every plate in "
                f"{list(REQUIRED_PLATES)}; missing {missing}. Mint them with "
                f"mint_locks() against this Motive session, or run "
                f"mocap_probe.py --lock-out")
        #: Off by default and named for what it costs: see the module docstring.
        self.fallback_to_streamed = bool(fallback_to_streamed)

        self._solve_lock = threading.Lock()
        self._marker_poses: np.ndarray | None = None
        self._streamed_poses: np.ndarray | None = None
        self.stats = SolveStats()

    # ------------------------------------------------------------------

    def _solve_q(self, homos, markers, flags):
        """Markers -> plate frames -> ``q``.  Overrides the streamed path.

        Runs on the SDK thread, once per frame, and must stay cheap: it is six
        Kabsch fits of four points each, which is microseconds, plus the same
        closed-form chain ``mocap_to_q`` uses.
        """
        poses, valid, quality = infer_all(markers, flags, self.locks)
        q = None
        if all(bool(valid[p]) for p in REQUIRED_PLATES):
            q = q_from_frames(poses, self.phis, self.order)

        with self._solve_lock:
            st = self.stats
            st.frames += 1
            for plate, qual in quality.items():
                if qual.reason:
                    st.reasons.setdefault(plate, {})
                    st.reasons[plate][qual.reason] = (
                        st.reasons[plate].get(qual.reason, 0) + 1)
                if np.isfinite(qual.rms_residual_m):
                    st.last_rms_m[plate] = float(qual.rms_residual_m)
            if q is not None:
                st.solved += 1
                self._marker_poses = poses
                self._streamed_poses = np.asarray(
                    homos[0:mc.N_USED_RIGID_BODIES], dtype=float).copy()
            else:
                st.unsolved += 1

        if q is None and self.fallback_to_streamed:
            q = super()._solve_q(homos, markers, flags)
            if q is not None:
                with self._solve_lock:
                    self.stats.unsolved -= 1
                    self.stats.fallback += 1
        return q

    # ------------------------------------------------------------------

    def get_marker_poses(self) -> np.ndarray | None:
        """The last successfully inferred ``(7, 4, 4)`` plate frames, or None.

        This is what a visualiser should draw: it is the pose the controller
        actually used, not a second opinion computed later.
        """
        with self._solve_lock:
            return None if self._marker_poses is None else self._marker_poses.copy()

    def get_streamed_poses(self) -> np.ndarray | None:
        """Motive's own poses from the same frame — the comparison partner."""
        with self._solve_lock:
            return (None if self._streamed_poses is None
                    else self._streamed_poses.copy())

    def solve_stats(self) -> SolveStats:
        with self._solve_lock:
            return SolveStats(
                frames=self.stats.frames, solved=self.stats.solved,
                fallback=self.stats.fallback, unsolved=self.stats.unsolved,
                reasons={p: dict(d) for p, d in self.stats.reasons.items()},
                last_rms_m=dict(self.stats.last_rms_m))

    def reset_stats(self) -> None:
        with self._solve_lock:
            self.stats = SolveStats()


# --------------------------------------------------------------------------
# Locks
# --------------------------------------------------------------------------


def _rest_stack(window, plate: int, require_tracked: bool) -> np.ndarray:
    """``(n, 4, 3)`` of frames where this plate showed all four markers.

    Same rule as ``mocap_probe._four_marker_frames``, restated here rather than
    imported: ``mocap_probe`` is a 45 kB CLI and a control session should not
    have to import it to start.
    """
    rows = []
    for markers, flags in zip(window.markers, window.flags):
        if markers is None or plate not in markers:
            continue
        arr = np.asarray(markers[plate], dtype=float)
        if arr.shape != (4, 3) or not np.isfinite(arr).all():
            continue
        if require_tracked:
            f = None if flags is None else flags.get(plate)
            if f is None or not np.all(np.asarray(f).astype(bool)):
                continue
        rows.append(arr)
    return np.stack(rows) if rows else np.empty((0, 4, 3))


def mint_locks(rx: MocapRx, seconds: float = DEFAULT_REST_S, *,
               x_mode: str = "streamed", require_tracked: bool = True,
               log=None) -> tuple[dict, dict]:
    """Capture a rest window from a *running* receiver and lock every plate.

    Returns ``(locks, report)``.  ``report`` carries the evidence a caller
    should print or store: frames used per plate, the stillness actually
    measured, and the x-disambiguation residual — the number that told the
    2026-08-11 campaign the base plate's marker arms sit 16 deg off their
    designed azimuth.

    ``x_mode="streamed"`` is the default for the reason that campaign
    established: anchoring the lock's x to the streamed rest orientation is
    mechanism-faithful, while the design-spec "nearest diagonal + 45 deg" is
    wrong by up to 16 deg on this hardware.  Note this uses the streamed pose
    **once, at rest, for an azimuth convention only** — every per-frame solve
    afterwards reads markers alone, so a pivot that moves later cannot reach
    ``q``.

    THE ARM MUST BE AT REST AND UNTOUCHED for the whole window.  A lock built
    from a moving arm is a template of a shape the plate never has again, and
    every later frame then fails the RMS gate — which at least fails loudly.
    """
    say = log if log is not None else (lambda _m: None)
    t0 = time.monotonic()
    deadline = t0 + float(seconds)
    while time.monotonic() < deadline:
        time.sleep(0.02)
    window = rx.snapshot_marker_window(t0=t0)

    report: dict = {"seconds": float(seconds), "frames": int(len(window)),
                    "x_mode": x_mode, "plates": {}, "refusals": [], "ok": False}
    if len(window) == 0:
        report["refusals"].append("no marker frames arrived during the window")
        return {}, report

    # Up comes from the ARM, never from a world axis (design §2 / review ops-4):
    # this arm hangs, so base-plate centre minus last-distal centre points up.
    pos = window.streamed_poses[:, :, 0:3, 3]
    u_up = pos[:, 0, :].mean(axis=0) - pos[:, 5, :].mean(axis=0)
    n_up = float(np.linalg.norm(u_up))
    if not (n_up > 1e-6):
        report["refusals"].append(
            "plate 0 and plate 5 centres coincide — no up reference; is either "
            "plate absent from the volume?")
        return {}, report
    u_up = u_up / n_up
    mid = len(window) // 2

    locks: dict[int, PlateLock] = {}
    for plate in REQUIRED_PLATES:
        stack = _rest_stack(window, plate, require_tracked)
        entry: dict = {"frames": int(stack.shape[0])}
        if stack.shape[0] < MIN_REST_FRAMES:
            entry["refused"] = (f"only {stack.shape[0]} usable rest frames "
                                f"(need {MIN_REST_FRAMES})")
            report["plates"][plate] = entry
            report["refusals"].append(f"plate {plate}: {entry['refused']}")
            continue
        sd = float(np.max(np.std(stack, axis=0)))
        entry["marker_sd_m"] = sd
        if sd > REST_STILLNESS_SD_M:
            entry["refused"] = (f"marker sd {sd * 1e3:.2f} mm exceeds "
                                f"{REST_STILLNESS_SD_M * 1e3:.2f} mm — the arm "
                                f"was not still")
            report["plates"][plate] = entry
            report["refusals"].append(f"plate {plate}: {entry['refused']}")
            continue
        lock = compute_plate_lock(stack, plate, u_up,
                                  streamed_rot=window.streamed_poses[mid, plate, 0:3, 0:3],
                                  x_mode=x_mode)
        locks[plate] = lock
        entry["x_residual_deg"] = float(lock.x_residual_deg)
        entry["midpoint_separation_mm"] = float(lock.midpoint_separation_m * 1e3)
        entry["phi_deg"] = math.degrees(float(lock.phi_rad))
        report["plates"][plate] = entry
        say(f"lock plate {plate}: {stack.shape[0]} rest frames, "
            f"sd {sd * 1e3:.3f} mm, x-residual {lock.x_residual_deg:.1f} deg, "
            f"diagonal midpoints {lock.midpoint_separation_m * 1e3:.2f} mm apart")

    # Set on every path, not only this one — an early return that left the key
    # absent would read as False to a caller using .get() and raise for one using
    # [], and "the key is missing" is not a verdict either way.
    report["ok"] = not report["refusals"]
    return locks, report


def save_locks(path: str, locks: dict, meta: dict | None = None) -> None:
    """Write locks in the shape ``mocap_probe.py --lock-out`` writes."""
    payload = {"meta": dict(meta or {}),
               "plates": {str(p): lk.to_dict() for p, lk in locks.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_locks(path: str) -> dict:
    """Read a locks JSON, tolerant of both the probe's and our own layout.

    Session-scoped, so a lock file older than the running Motive session is a
    trap: the marker labels it was minted against may no longer be the labels
    arriving.  A stale lock fails loudly (every frame trips the template RMS
    gate) rather than quietly, but prefer :func:`mint_locks`.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    # "plates" is what mocap_probe --lock-out writes; "locks" is save_locks
    # above; a bare mapping is what a hand-made file tends to be.
    body = raw.get("plates") or raw.get("locks") or raw
    out = {}
    for key, value in body.items():
        try:
            plate = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and "template_m" in value:
            out[plate] = PlateLock.from_dict(value)
    if not out:
        raise ValueError(f"{path} holds no plate locks")
    return out


def compare_to_streamed(rx: MarkerMocap) -> dict | None:
    """Marker-inferred vs Motive-streamed poses for the most recent frame.

    Returns per-plate ``angle_deg`` / ``origin_mm``, or ``None`` before the
    first solved frame.  The 2026-08-11 campaign measured 0.66-4.24 deg and
    0.3-1.5 mm here; those are the *expected* disagreements (the streamed body
    frame has its own azimuth convention and its own pivot), not errors, so
    this is a change detector, not a pass/fail gate.
    """
    inferred = rx.get_marker_poses()
    streamed = rx.get_streamed_poses()
    if inferred is None or streamed is None:
        return None
    out = {}
    for plate in REQUIRED_PLATES:
        phi = FAMILY_PHI_RAD.get(plate, 0.0)
        c, s = math.cos(-phi), math.sin(-phi)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        r_body = inferred[plate, 0:3, 0:3] @ rz
        rel = streamed[plate, 0:3, 0:3].T @ r_body
        cos = (float(np.trace(rel)) - 1.0) / 2.0
        out[plate] = {
            "angle_deg": math.degrees(math.acos(min(1.0, max(-1.0, cos)))),
            "origin_mm": float(np.linalg.norm(
                streamed[plate, 0:3, 3] - inferred[plate, 0:3, 3])) * 1e3,
        }
    return out


__all__ = ["MarkerMocap", "SolveStats", "mint_locks", "save_locks",
           "load_locks", "compare_to_streamed", "REQUIRED_PLATES"]
