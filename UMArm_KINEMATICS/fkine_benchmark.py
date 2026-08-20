"""Offline fkine hardware benchmark: campaign artifacts -> parameter-vs-model verdict.

Implements ``docs/fkine_design.md`` §4 (with the §5 data-sufficiency rules).
The question this tool exists to answer, verbatim from the design: *parameter
issue or model issue* — post-fit residuals at mocap-noise level mean the CAD
lengths were off (parameters); structured residuals surviving the fit mean the
kinematic model itself has a gap, and the report names it (which plate, which
joint correlation) instead of averaging it away.

------------------------------------------------------------------------------
WHAT IT CONSUMES (per campaign pass — one ``results/<campaign>`` directory)
------------------------------------------------------------------------------
* ``joint_NN_mocap.csv`` — the recorded q and the streamed u-joint centres,
  with the per-frame ``chain_ok`` gate.  **Only chain_ok frames are used**
  (design §4: "for every recorded frame with chain_ok").
* ``joint_NN_poses.csv`` — the streamed plate poses.  Nothing else records the
  base *orientation* (review finding int-0), so this file is what regime (a)
  stands on.  Joined to the mocap CSV by ``frame`` — the two files share
  ``trace.t0`` but come from different rings, and the frame number is the
  cross-ring key (recording design §7).
* ``joint_NN_markers.csv`` + ``locks.json`` — raw marker rectangles and the
  rest-time per-plate locks, for regime (b).  Locks are validity-bound per
  marker-frame design §4.1: this consumer re-derives the rest rectangle stats
  from the head of the data it was given and **refuses** on mismatch beyond
  noise (``verify_lock_against_rest``) — a stale lock must never be laundered
  into a benchmark number.
* ``joint_NN.json`` — read leniently, for display only (verdict strings); every
  number in the report is recomputed from the CSVs.

------------------------------------------------------------------------------
THE TWO REGIMES (design §4.2-4.3)
------------------------------------------------------------------------------
* **(a) streamed**: base = the streamed body-500 pose (position *and*
  orientation from the poses CSV); targets = the streamed pivot positions of
  plates 1-5 (the mocap CSV's ``u`` columns — same rigid bodies, guaranteed
  row-aligned with the recorded q).  This is the frame the shipped pipeline
  trusts, manual-alignment error included.
* **(b) inferred**: base = the marker-inferred plate-0 **body** frame; targets
  = the marker-inferred plate origins (diagonal intersections).  The honest
  target — no Motive manual alignment anywhere.  Both regimes are evaluated
  throughout and reported side by side; plate 0's family angle is 0
  (``FAMILY_PHI_RAD``), so its bracket frame *is* its body frame.

Predicted centres come from :func:`UMArm_KINEMATICS.fkine.predict_spatial`
(rows 1-5 <-> plates 1-5; row 0 is the degenerate base point, design §1) — the
mounting model and the chain live in the port, in exactly one place, and this
tool never re-implements them (design D6).

------------------------------------------------------------------------------
THE FIT (design §4.5, decisions D2/D3)
------------------------------------------------------------------------------
Gauss-Newton on **numeric Jacobians, no scipy**, over the five chain lengths
``(span1, JD2, span2, JD3, span3)`` — the fit is 5-dimensional because
UC1 = UC2 = 0 is a *hardware fact* (plate centre = joint centre,
``robot_constants.py:23,32,41``), not because UC is unobservable — plus, for
regime (a) only, a constant body-side SO(3) correction to the streamed base
orientation (``R_base' = R_base @ exp(w^)``): the manual-alignment error model.
Regime (b) gets **no** base correction by design — needing none is the test of
the inferred frame.

The residual is exactly linear in the five lengths at fixed q (every chain
factor's rotation is length-independent; translations compose affinely), so
Gauss-Newton converges in ~1 step on the lengths and needs its iterations only
for the small rotation.  Forward differences are therefore exact where it
matters and cheap everywhere: the default ``--fit-max-frames`` subsample keeps
a full 12-joint campaign's fit around a minute; the *reported* residuals are
always computed over every loaded frame with the finished parameters.

Structured residuals are **reported, never absorbed** (design D3): the only
knobs are the five lengths and the one rotation, and anything they cannot
explain lands in the verdict section with a name.  If the fit's conditioning
is poor, the report says so and points at the §5 fallback (an annotated 5 psi
data-collection pass) instead of quietly trusting a rank-deficient solve.

------------------------------------------------------------------------------
MERGING PASSES (design §5)
------------------------------------------------------------------------------
Several campaign directories may be given (the §5 second-pass scenario).  Each
pass's base frame is fingerprinted independently — rest-window chain spans,
rest base position and orientation — and the tool **refuses to merge** passes
whose fingerprints disagree beyond noise: a moved rig or a re-created Motive
body between passes would otherwise smear two inconsistent geometries into one
meaningless fit.

------------------------------------------------------------------------------
OUTPUTS (design §4.6) and EXIT CODES
------------------------------------------------------------------------------
``<out-dir>/`` (default ``<first pass>/fkine_benchmark/``):
``fkine_benchmark.md`` + ``fkine_benchmark.json`` (every number, JSON-safe) +
PNGs (per-plate residual summary, fitted lengths, per-joint residual traces).
Report contents: per-plate residual RMS/max before/after fit for both regimes,
fitted lengths vs CAD nominal vs the campaign's measured spans, the regime-(a)
base correction, fit conditioning, the EE step (skip-and-report while body 506
is absent — 2026-08-11 probe — with q10/q11 validated through plate 5's
marker-inferred body orientation instead, design §4.4), the per-joint marker
regime table, and the verdict.  The 5 mm acceptance target is **reported, not
gated** (design §4).

Exit codes: 0 = report written; 1 = inputs missing/unusable; 2 = refusal
(stale locks or an unmergeable pass) — same "refusals are named, not warnings"
contract as the probe.

Run::

    python UMArm_KINEMATICS/fkine_benchmark.py RESULTS_DIR [RESULTS_DIR ...]
        [--locks locks.json] [--out-dir DIR] [--fit-max-frames 1500]
        [--stride 1] [--no-base-rot] [--no-plots]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

# Dual-mode import: package submodule or flat script, same pattern as fkine.py.
try:  # pragma: no cover - exercised by whichever import style the caller uses
    from . import robot_params as rp
    from .fkine import fkine, predict_spatial
except ImportError:  # pragma: no cover
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import robot_params as rp  # type: ignore[no-redef]
    from fkine import fkine, predict_spatial  # type: ignore[no-redef]

# The sibling package, for the regime-(b) machinery.  This is an *offline
# analysis* dependency, deliberately outside the D8 no-runtime-cross-import
# rule (that rule keeps robot_params/arm_constants decoupled; the benchmark is
# the consumer the marker-frame work exists for).  Repo root appended so the
# flat-script invocation finds it; appended (not prepended) out of the same
# caution as the oracle tests — nothing at the repo root may shadow stdlib or
# local names.
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
from UMArm_MOCAP.marker_frame import (  # noqa: E402
    FAMILY_PHI_RAD,
    LockRefusal,
    PlateLock,
    infer_all,
    q_from_frames,
    verify_lock_against_rest,
)
from UMArm_MOCAP.mocap_to_q import quat_xyzw_to_matrix  # noqa: E402

# --------------------------------------------------------------------------
# Constants (each with its source; none are new policy)
# --------------------------------------------------------------------------

EXIT_OK = 0
EXIT_NO_DATA = 1
EXIT_REFUSED = 2

#: Head-of-recording window treated as rest, seconds.  ``t_s`` in every
#: campaign CSV is relative to ``trace.t0`` = the start of the 3 s rest
#: capture (recording design §7), so the first second is guaranteed pre-drive;
#: it feeds the lock validity check, the rest-noise floor, and the merge
#: fingerprint.
REST_HEAD_S = 1.0

#: Minimum rest frames before a per-plate lock check / noise estimate is
#: attempted; below this the statistic is a coin flip, and the honest response
#: is a named skip, not a confident number.
REST_MIN_FRAMES = 5

#: Merge-refusal bounds (design §5: "refuses to merge passes whose rest spans
#: or base pose disagree beyond noise").  Rest marker jitter on this Motive is
#: 0.2-1.5 mm (``joint_verification.repeated_plate_report``), so 5 mm on a
#: span and 10 mm on the base position pass any healthy same-rig pair while a
#: re-seated rig or re-pivoted body (>= cm) cannot; 5 deg on the base
#: orientation passes streaming noise while a re-created Motive body (tens of
#: degrees, review finding ops-0) cannot.
MERGE_SPAN_TOL_M = 0.005
MERGE_BASE_POS_TOL_M = 0.010
MERGE_BASE_ROT_TOL_DEG = 5.0

#: Fit subsample cap.  The residual is linear in the lengths, so the fit does
#: not need every frame — it needs *excitation coverage*, which an even
#: subsample of the whole campaign preserves.  Reported residuals always use
#: every loaded frame.
FIT_MAX_FRAMES = 1500

#: Gauss-Newton controls.  Forward-difference steps: 1 um / 1 urad sit ~9
#: orders above double rounding on ~1 m coordinates and ~3 below marker noise,
#: so the numeric Jacobian is exact to far better than the data.  25
#: iterations is generous head-room for a problem that is linear in 5 of its
#: (at most) 8 parameters.
GN_MAX_ITER = 25
GN_STEP_TOL = 1e-10
FD_STEP_LEN_M = 1e-6
FD_STEP_ROT_RAD = 1e-6

#: Acceptance target (design §4: reported, not gated): post-fit RMS < 5 mm
#: across plates 1-5 over all campaign frames.
ACCEPT_RMS_M = 0.005

#: Verdict boundary: the design's "post-fit residuals at mocap-noise level"
#: is judged against the *pipeline's own* noise level — the post-fit residual
#: RMS over the rest frames, where q is constant and no model term is
#: excited.  Raw pivot jitter underestimates that level for regime (b): the
#: inferred **base orientation** jitters with the markers (~0.3 mm on a 60 mm
#: bracket is ~0.005 rad), and the 0.7 m chain levers it into millimetres at
#: plate 5 on a perfectly healthy rig.  Structure is what excitation *adds*:
#: drive-frame RMS beyond this multiple of the rest-frame RMS reads "model",
#: with the worst plate and the strongest |q| correlation named (design
#: §4.6).  Noise alone gives a ratio near 1; a real unmodelled offset only
#: shows under drive and pushes it well past 2.
STRUCTURE_RATIO = 2.0

#: Rest-noise fallback when a pass carries no usable rest frames: mid-range of
#: the 0.2-1.5 mm resting jitter this Motive measures.  Used only with a
#: warning — a measured floor always wins.
NOISE_FALLBACK_M = 0.0015

#: Verdict floor clamp: a measured rest jitter below the bottom of the
#: observed range (0.2 mm) is not evidence of a better mocap — it is a
#: too-quiet window (or a noiseless synthetic campaign, where the sd is
#: exactly 0 and an unclamped boundary would misread a perfect fit as a
#: model gap).  The clamp applies to the verdict boundary only; the measured
#: number is still reported as measured.
NOISE_FLOOR_MIN_M = 0.0002

#: Residual-vs-|q| Pearson correlation above which the verdict names the pair
#: as a model-gap candidate.  0.6 is far above what noise produces on hundreds
#: of frames (its null sd is ~1/sqrt(n)) and far below the ~1.0 a real
#: unmodelled per-plate offset shows under single-joint excitation.
CORR_NAMED = 0.6

#: Jacobian condition number past which the report recommends the design §5
#: fallback (an annotated 5 psi data-collection pass) — the parameter
#: directions are no longer all constrained by this campaign's excitation.
COND_WARN = 1e4

#: The five chain gaps, in order, as the report names them everywhere.
GAP_NAMES = ("span1", "JD2", "span2", "JD3", "span3")

_EYE4 = np.eye(4)


# --------------------------------------------------------------------------
# Small numerics
# --------------------------------------------------------------------------


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ], dtype=float)


def exp_so3(w) -> np.ndarray:
    """Rodrigues ``exp(w^)`` for a rotation vector — the base-correction map.

    The fit parametrizes the regime-(a) correction as a rotation vector
    because it is singularity-free at the origin, which is exactly where the
    correction lives (manual alignment error is a few degrees).  Below 1e-12
    rad the first-order form is exact to double precision and avoids 0/0.
    """
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + _skew(w)
    k = _skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def rot_angle_deg(r: np.ndarray) -> float:
    """Angle of a rotation matrix, degrees, trace formula clipped for noise."""
    c = (float(np.trace(np.asarray(r)[0:3, 0:3])) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


# --------------------------------------------------------------------------
# Campaign artifact loading
# --------------------------------------------------------------------------


@dataclasses.dataclass
class JointData:
    """One joint's recording, row-aligned to its ``joint_NN_mocap.csv``.

    Everything per-frame is indexed by that CSV's row order; the poses and
    markers files (different ring, possibly different frame coverage) are
    joined onto it by ``frame`` number.  ``markers``/``flags`` reproduce the
    marker-ring dict semantics exactly (``mocap_rx``): ``markers[i] = None``
    means marker sets were absent that frame (the CSV's ``-`` placeholder row,
    or the frame simply missing from the markers file); ``flags[i]`` maps only
    the plates whose tracked column was numeric — a plate recorded ``?``
    (labeled markers absent) is *omitted*, so the solver sees ``None`` flags
    for it and the flags-unknown regime is preserved end to end (design D4).
    """

    q_index: int
    t: np.ndarray                    # (n,) seconds, relative to trace.t0
    frame: np.ndarray                # (n,) int
    q: np.ndarray                    # (n, 12) radians, as recorded
    u: np.ndarray                    # (n, 6, 3) streamed pivot positions
    chain_ok: np.ndarray             # (n,) bool
    streamed: np.ndarray             # (n, 7, 4, 4) streamed plate poses
    streamed_present: np.ndarray     # (n, 7) bool — pose row found and non-identity
    have_poses: bool
    markers: list                    # n entries: dict[plate -> (m, 3)] | None
    flags: list                      # n entries: dict[plate -> (m,) uint8] | None
    have_markers: bool
    epochs: set
    meta: dict
    # Filled by infer_pass_frames:
    inferred: np.ndarray | None = None       # (n, 7, 4, 4)
    inferred_valid: np.ndarray | None = None  # (n, 7) bool
    three_marker: np.ndarray | None = None    # (n, 7) bool
    reasons: dict = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.t.shape[0])


@dataclasses.dataclass
class CampaignPass:
    """One campaign directory: its joints, its locks, and its fingerprint."""

    path: str
    joints: list
    locks: dict | None
    locks_path: str | None
    locks_meta: dict
    warnings: list
    # Merge fingerprint (design §5), from the rest heads:
    rest_spans_m: np.ndarray | None = None       # (5,)
    base_pos_m: np.ndarray | None = None         # (3,)
    base_rot: np.ndarray | None = None           # (3, 3)


def _cols(header: list, names: list) -> list:
    """Column indices by header name — the CSVs are contracts, not positions."""
    idx = {}
    for name in names:
        if name not in header:
            raise ValueError(f"missing column {name!r} (header: {header})")
        idx[name] = header.index(name)
    return [idx[n] for n in names]


def load_mocap_csv(path: str):
    """``joint_NN_mocap.csv`` -> (t, frame, q, u, chain_ok) arrays."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        q_names = [f"q{i}_rad" for i in range(12)]
        u_names = [f"u{k}_{ax}" for k in range(6) for ax in "xyz"]
        (c_t, c_f), c_q, c_u, (c_ok,) = (
            _cols(header, ["t_s", "frame"]), _cols(header, q_names),
            _cols(header, u_names), _cols(header, ["chain_ok"]))
        t, frame, q, u, ok = [], [], [], [], []
        for row in rd:
            if not row:
                continue
            t.append(float(row[c_t]))
            frame.append(int(row[c_f]))
            q.append([float(row[c]) for c in c_q])
            u.append([float(row[c]) for c in c_u])
            ok.append(row[c_ok].strip() == "1")
    return (np.asarray(t, dtype=float), np.asarray(frame, dtype=np.int64),
            np.asarray(q, dtype=float).reshape(-1, 12),
            np.asarray(u, dtype=float).reshape(-1, 6, 3),
            np.asarray(ok, dtype=bool))


def load_poses_csv(path: str) -> dict:
    """``joint_NN_poses.csv`` -> frame -> ((7, 4, 4) poses, (7,) present).

    A row whose pose is exactly the identity (quat 0,0,0,1 at the origin) is
    a body Motive never placed — the receiver's rows start at the identity
    and hold their last value (``mocap_probe.absent_plates``) — and is marked
    absent rather than treated as a real pose at the volume origin.
    """
    out: dict[int, tuple] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        cols = _cols(header, ["frame", "plate", "qx", "qy", "qz", "qw",
                              "x", "y", "z"])
        for row in rd:
            if not row:
                continue
            frame = int(row[cols[0]])
            plate = int(row[cols[1]])
            quat = np.array([float(row[c]) for c in cols[2:6]])
            pos = np.array([float(row[c]) for c in cols[6:9]])
            if frame not in out:
                out[frame] = (np.tile(_EYE4, (7, 1, 1)), np.zeros(7, dtype=bool))
            poses, present = out[frame]
            if not 0 <= plate < 7:
                continue
            identity = (np.all(quat == np.array([0.0, 0.0, 0.0, 1.0]))
                        and np.all(pos == 0.0))
            if identity:
                continue
            try:
                r = quat_xyzw_to_matrix(quat)
            except ValueError:
                continue                       # degenerate stream row: absent
            t = np.eye(4)
            t[0:3, 0:3] = r
            t[0:3, 3] = pos
            poses[plate] = t
            present[plate] = True
    return out


def load_markers_csv(path: str) -> dict:
    """``joint_NN_markers.csv`` -> frame -> (markers, flags, epoch).

    Rebuilds the marker-ring dict semantics from the long CSV: the ``-``
    placeholder row (plate -1) means marker sets were absent that frame
    (markers ``None``); a plate whose tracked column is ``?`` has unknown
    flags and is left out of the flags dict (its solver call then gets
    ``None`` flags — the flags-unknown regime, design §5); ``1``/``0`` become
    uint8 tracked flags.
    """
    acc: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        cols = _cols(header, ["frame", "epoch", "plate", "marker",
                              "x", "y", "z", "tracked"])
        for row in rd:
            if not row:
                continue
            frame = int(row[cols[0]])
            entry = acc.setdefault(frame, {"epoch": int(row[cols[1]]),
                                           "plates": {}, "absent": False})
            plate = int(row[cols[2]])
            if plate < 0:
                entry["absent"] = True
                continue
            marker = int(row[cols[3]])
            pos = (float(row[cols[4]]), float(row[cols[5]]), float(row[cols[6]]))
            tracked = row[cols[7]].strip()
            entry["plates"].setdefault(plate, []).append((marker, pos, tracked))
    out: dict[int, tuple] = {}
    for frame, entry in acc.items():
        if entry["absent"] or not entry["plates"]:
            out[frame] = (None, None, entry["epoch"])
            continue
        markers: dict[int, np.ndarray] = {}
        flags: dict[int, np.ndarray] = {}
        for plate, rows in entry["plates"].items():
            m = max(r[0] for r in rows) + 1
            arr = np.full((m, 3), np.nan)
            fl = np.zeros(m, dtype=np.uint8)
            unknown = False
            for marker, pos, tracked in rows:
                arr[marker] = pos
                if tracked == "?":
                    unknown = True
                elif tracked in ("0", "1"):
                    fl[marker] = np.uint8(tracked == "1")
                else:
                    unknown = True             # "-" on a plate row: no flag truth
            markers[plate] = arr
            if not unknown:
                flags[plate] = fl
        out[frame] = (markers, flags if flags else None, entry["epoch"])
    return out


def load_joint(dirpath: str, q_index: int) -> JointData:
    """Assemble one joint's row-aligned :class:`JointData` from its files."""
    stem = os.path.join(dirpath, f"joint_{q_index:02d}")
    t, frame, q, u, chain_ok = load_mocap_csv(stem + "_mocap.csv")
    n = t.shape[0]

    streamed = np.tile(_EYE4, (n, 7, 1, 1))
    streamed_present = np.zeros((n, 7), dtype=bool)
    have_poses = os.path.isfile(stem + "_poses.csv")
    if have_poses:
        poses = load_poses_csv(stem + "_poses.csv")
        for i in range(n):
            hit = poses.get(int(frame[i]))
            if hit is not None:
                streamed[i], streamed_present[i] = hit

    markers: list = [None] * n
    flags: list = [None] * n
    epochs: set = set()
    have_markers = os.path.isfile(stem + "_markers.csv")
    if have_markers:
        marker_map = load_markers_csv(stem + "_markers.csv")
        for i in range(n):
            hit = marker_map.get(int(frame[i]))
            if hit is not None:
                markers[i], flags[i], epoch = hit
                epochs.add(epoch)

    meta: dict = {}
    if os.path.isfile(stem + ".json"):
        try:
            with open(stem + ".json", "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            meta = {}                          # display-only; never load-bearing

    return JointData(q_index=q_index, t=t, frame=frame, q=q, u=u,
                     chain_ok=chain_ok, streamed=streamed,
                     streamed_present=streamed_present, have_poses=have_poses,
                     markers=markers, flags=flags, have_markers=have_markers,
                     epochs=epochs, meta=meta)


def load_locks(path: str) -> tuple:
    """``locks.json`` (probe format) -> ({plate: PlateLock}, meta)."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    locks = {int(p): PlateLock.from_dict(d)
             for p, d in payload.get("plates", {}).items()}
    return locks, payload.get("meta", {})


def load_pass(dirpath: str, locks_path: str | None) -> CampaignPass:
    """One campaign directory -> a loaded :class:`CampaignPass` (no inference yet)."""
    stems = sorted(glob.glob(os.path.join(dirpath, "joint_*_mocap.csv")))
    joints = []
    warnings: list = []
    for path in stems:
        m = re.search(r"joint_(\d+)_mocap\.csv$", os.path.basename(path))
        if not m:
            continue
        joints.append(load_joint(dirpath, int(m.group(1))))
    if not joints:
        raise FileNotFoundError(
            f"{dirpath}: no joint_NN_mocap.csv files — not a campaign directory")

    lp = locks_path if locks_path else os.path.join(dirpath, "locks.json")
    locks = None
    locks_meta: dict = {}
    if os.path.isfile(lp):
        locks, locks_meta = load_locks(lp)
    else:
        if locks_path:                        # an explicit path must exist
            raise FileNotFoundError(f"--locks {locks_path}: no such file")
        lp = None
        warnings.append(f"{dirpath}: no locks.json — regime (b) is unavailable "
                        f"for this pass (run the probe with --lock-out first)")

    p = CampaignPass(path=dirpath, joints=joints, locks=locks, locks_path=lp,
                     locks_meta=locks_meta, warnings=warnings)
    _fingerprint_pass(p)
    return p


# --------------------------------------------------------------------------
# Rest heads: lock validity, merge fingerprint, noise floor
# --------------------------------------------------------------------------


def _rest_mask(jd: JointData) -> np.ndarray:
    """chain_ok frames inside the guaranteed-rest head of one recording."""
    return jd.chain_ok & (jd.t <= REST_HEAD_S)


def rest_marker_stacks(p: CampaignPass) -> dict:
    """Per-plate ``(m, 4, 3)`` rest marker stacks for the lock validity check.

    Walks the pass's joints in q order and takes each plate's frames from the
    *first* joint that has enough of them — the earliest rest head is the
    closest thing this pass has to the probe's lock window.  A frame counts
    only when the plate streamed exactly four markers, all tracked (or flags
    unknown, which the design treats as assumed-usable — the regime is
    reported, not silently upgraded).
    """
    stacks: dict[int, np.ndarray] = {}
    for jd in p.joints:
        rest = np.flatnonzero(_rest_mask(jd))
        if rest.size == 0:
            continue
        per_plate: dict[int, list] = {}
        for i in rest:
            mk, fl = jd.markers[i], jd.flags[i]
            if mk is None:
                continue
            for plate, arr in mk.items():
                if plate in stacks or arr.shape[0] != 4:
                    continue
                pf = None if fl is None else fl.get(plate)
                if pf is not None and not np.all(pf == 1):
                    continue
                if not np.isfinite(arr).all():
                    continue
                per_plate.setdefault(plate, []).append(arr)
        for plate, frames in per_plate.items():
            if len(frames) >= REST_MIN_FRAMES:
                stacks[plate] = np.stack(frames)
    return stacks


def check_locks(p: CampaignPass) -> tuple:
    """The §4.1 validity binding: locks vs this pass's own rest markers.

    Returns ``(refusals, verified, skipped)``: refusal strings (stale locks —
    the caller exits 2), plates verified clean, and plates that could not be
    checked (no rest markers) — the last are *reported*, because an
    unverifiable lock is a weaker claim than a verified one, but only an
    actual mismatch refuses.
    """
    if p.locks is None:
        return [], [], []
    stacks = rest_marker_stacks(p)
    refusals, verified, skipped = [], [], []
    for plate, lock in sorted(p.locks.items()):
        stack = stacks.get(plate)
        if stack is None:
            skipped.append(plate)
            continue
        try:
            verify_lock_against_rest(lock, stack)
        except LockRefusal as exc:
            refusals.append(f"{p.path}: {exc}")
        else:
            verified.append(plate)
    return refusals, verified, skipped


def _fingerprint_pass(p: CampaignPass) -> None:
    """Rest spans + rest base pose — the §5 merge fingerprint.

    Spans come from the mocap CSV's streamed pivots over every joint's rest
    head (they are rigid-body distances, so averaging across joints is
    legitimate); the base pose comes from the poses CSV over the same frames.
    A pass without poses keeps ``base_rot = None`` and is compared on position
    and spans alone (stated in the report).
    """
    span_sum = np.zeros(5)
    span_n = 0
    pos_sum = np.zeros(3)
    pos_n = 0
    rot: np.ndarray | None = None
    for jd in p.joints:
        rest = np.flatnonzero(_rest_mask(jd))
        if rest.size < REST_MIN_FRAMES:
            continue
        d = np.linalg.norm(jd.u[rest, 1:, :] - jd.u[rest, :-1, :], axis=2)
        span_sum += d.sum(axis=0)
        span_n += rest.size
        if jd.have_poses:
            ok = rest[jd.streamed_present[rest, 0]]
            if ok.size:
                pos_sum += jd.streamed[ok, 0, 0:3, 3].sum(axis=0)
                pos_n += ok.size
                if rot is None:
                    rot = jd.streamed[ok[0], 0, 0:3, 0:3]
    if span_n:
        p.rest_spans_m = span_sum / span_n
    if pos_n:
        p.base_pos_m = pos_sum / pos_n
        p.base_rot = rot


def check_merge(passes: list) -> list:
    """Pairwise fingerprint comparison against pass 0 — refusal strings."""
    problems = []
    ref = passes[0]
    for other in passes[1:]:
        if ref.rest_spans_m is not None and other.rest_spans_m is not None:
            dspan = np.abs(other.rest_spans_m - ref.rest_spans_m)
            worst = int(np.argmax(dspan))
            if dspan[worst] > MERGE_SPAN_TOL_M:
                problems.append(
                    f"{other.path}: rest span {GAP_NAMES[worst]} differs from "
                    f"{ref.path} by {dspan[worst] * 1000.0:.1f} mm "
                    f"(> {MERGE_SPAN_TOL_M * 1000.0:.0f} mm) — the rig moved or a "
                    f"pivot was re-set between passes; benchmark them separately")
        if ref.base_pos_m is not None and other.base_pos_m is not None:
            dpos = float(np.linalg.norm(other.base_pos_m - ref.base_pos_m))
            if dpos > MERGE_BASE_POS_TOL_M:
                problems.append(
                    f"{other.path}: rest base position differs from {ref.path} by "
                    f"{dpos * 1000.0:.1f} mm (> {MERGE_BASE_POS_TOL_M * 1000.0:.0f} mm)"
                    f" — the volume or the rig moved; benchmark them separately")
        if ref.base_rot is not None and other.base_rot is not None:
            drot = rot_angle_deg(ref.base_rot.T @ other.base_rot)
            if drot > MERGE_BASE_ROT_TOL_DEG:
                problems.append(
                    f"{other.path}: rest base orientation differs from {ref.path} "
                    f"by {drot:.1f} deg (> {MERGE_BASE_ROT_TOL_DEG:.0f} deg) — a "
                    f"re-created body 500 between passes; benchmark them separately")
    return problems


def noise_floor_m(passes: list) -> tuple:
    """Measured rest jitter of the streamed pivots (plates 1-5), metres.

    Per joint, per plate: the norm of the per-axis sd over the rest head; the
    floor is the **median** across all of them, so one plate with a flaky
    marker cannot drag the verdict boundary.  Returns ``(floor, n_samples,
    measured?)`` — the fallback constant is used, flagged, when no pass has a
    usable rest head.
    """
    jitters = []
    for p in passes:
        for jd in p.joints:
            rest = np.flatnonzero(_rest_mask(jd))
            if rest.size < REST_MIN_FRAMES:
                continue
            sd = jd.u[rest].std(axis=0, ddof=1)         # (6, 3)
            jitters.extend(np.linalg.norm(sd[1:6], axis=1).tolist())
    if jitters:
        return float(np.median(jitters)), len(jitters), True
    return NOISE_FALLBACK_M, 0, False


# --------------------------------------------------------------------------
# Regime (b) inference over the recordings
# --------------------------------------------------------------------------


def infer_pass_frames(p: CampaignPass) -> None:
    """Run the marker-frame solve on every chain_ok frame of every joint.

    Fills each joint's ``inferred``/``inferred_valid``/``three_marker`` and
    the per-joint gate-reason counters (the benchmark *counts* the named
    ``None`` reasons instead of guessing — marker-frame design §4.2).  A pass
    without locks stays un-inferred and regime (b) is reported unavailable.
    """
    if p.locks is None:
        return
    for jd in p.joints:
        n = jd.n
        inferred = np.tile(_EYE4, (n, 7, 1, 1))
        valid = np.zeros((n, 7), dtype=bool)
        three = np.zeros((n, 7), dtype=bool)
        reasons: dict[str, int] = {}
        for i in range(n):
            if not jd.chain_ok[i]:
                continue
            poses, vmask, quality = infer_all(jd.markers[i], jd.flags[i], p.locks)
            inferred[i] = poses
            valid[i] = vmask
            for plate, qual in quality.items():
                if qual.reason:
                    reasons[qual.reason] = reasons.get(qual.reason, 0) + 1
                elif qual.n_usable == 3:
                    three[i, plate] = True
        jd.inferred = inferred
        jd.inferred_valid = valid
        jd.three_marker = three
        jd.reasons = reasons


# --------------------------------------------------------------------------
# Benchmark frames (one per usable recorded frame per regime)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class BenchFrame:
    """Everything one frame contributes to one regime.

    ``targets`` rows are plates 1..5 (the mocap<->fkine correspondence of
    design §1: predicted rows 1..5 of :func:`predict_spatial` line up with
    them; row 0 is the degenerate base point and carries no signal).  ``p6``
    is the plate-6 target when that body exists — ``None`` on today's volume
    (2026-08-11 probe: body 506 absent).
    """

    pass_idx: int
    joint: int
    t: float
    q: np.ndarray            # (12,)
    base: np.ndarray         # (4, 4)
    targets: np.ndarray      # (5, 3)
    mask: np.ndarray         # (5,) bool
    p6: np.ndarray | None = None


def build_frames(passes: list, regime: str) -> list:
    """All usable :class:`BenchFrame` for one regime, campaign order.

    * ``streamed``: chain_ok, poses row present, plate-0 pose present.  The
      targets are the mocap CSV's ``u`` columns — row-aligned with q by
      construction — and the base is the poses CSV's body-500 pose.
    * ``inferred``: chain_ok, plate-0 marker frame solved.  The base is that
      frame directly: plate 0's family angle is 0 (:data:`FAMILY_PHI_RAD`),
      so bracket frame == body frame and no Rz correction applies.  The
      per-plate mask is the honest validity mask of ``infer_all`` — an
      unsolved plate contributes nothing rather than an identity origin.
    """
    frames: list = []
    for pi, p in enumerate(passes):
        for jd in p.joints:
            if regime == "streamed":
                if not jd.have_poses:
                    continue
                idx = np.flatnonzero(jd.chain_ok & jd.streamed_present[:, 0])
                for i in idx:
                    p6 = (jd.streamed[i, 6, 0:3, 3]
                          if jd.streamed_present[i, 6] else None)
                    frames.append(BenchFrame(
                        pass_idx=pi, joint=jd.q_index, t=float(jd.t[i]),
                        q=jd.q[i], base=jd.streamed[i, 0],
                        targets=jd.u[i, 1:6], mask=np.ones(5, dtype=bool),
                        p6=p6))
            elif regime == "inferred":
                if jd.inferred is None:
                    continue
                idx = np.flatnonzero(jd.chain_ok & jd.inferred_valid[:, 0])
                for i in idx:
                    mask = jd.inferred_valid[i, 1:6].copy()
                    if not mask.any():
                        continue
                    p6 = (jd.inferred[i, 6, 0:3, 3]
                          if jd.inferred_valid[i, 6] else None)
                    frames.append(BenchFrame(
                        pass_idx=pi, joint=jd.q_index, t=float(jd.t[i]),
                        q=jd.q[i], base=jd.inferred[i, 0],
                        targets=jd.inferred[i, 1:6, 0:3, 3], mask=mask, p6=p6))
            else:
                raise ValueError(f"unknown regime {regime!r}")
    return frames


# --------------------------------------------------------------------------
# The fit (design §4.5: Gauss-Newton, numeric Jacobians, no scipy)
# --------------------------------------------------------------------------


def params_from_gaps(gaps) -> np.ndarray:
    """Five chain gaps -> a full ``(3, 10)`` params table (writable copy).

    The knob for each span is ``LL`` (UC and AA stay at their hardware
    values; only the *sum* moves a centre, so which term absorbs the delta is
    a bookkeeping choice and LL — the actuated length — is the one CAD is
    least sure of).  ``DEFAULT_PARAMS`` is read-only by design; the fit works
    on its own copy, never the shared table.
    """
    gaps = np.asarray(gaps, dtype=float)
    if gaps.shape != (5,):
        raise ValueError(f"gaps must have shape (5,); got {gaps.shape}")
    p = np.array(rp.DEFAULT_PARAMS)            # writable copy
    fixed = (p[:, rp.COL_UC1] + p[:, rp.COL_AA1]
             + p[:, rp.COL_AA2] + p[:, rp.COL_UC2])
    p[0, rp.COL_LL] = gaps[0] - fixed[0]
    p[1, rp.COL_LL] = gaps[2] - fixed[1]
    p[2, rp.COL_LL] = gaps[4] - fixed[2]
    p[1, rp.COL_JD] = gaps[1]
    p[2, rp.COL_JD] = gaps[3]
    return p


def _theta_split(theta: np.ndarray, fit_rot: bool):
    return (theta[0:5], theta[5:8] if fit_rot else None)


def residual_vector(frames: list, theta: np.ndarray, fit_rot: bool) -> np.ndarray:
    """Stacked 3D residuals (predicted - measured), masked plates only.

    The base correction is applied on the *body* side (``R @ exp(w^)``): a
    constant misalignment of the Motive body axes relative to the physical
    plate — exactly the manual-alignment error regime (a) is testing.
    """
    gaps, rotvec = _theta_split(theta, fit_rot)
    params = params_from_gaps(gaps)
    corr = exp_so3(rotvec) if rotvec is not None else None
    chunks = []
    for f in frames:
        base = f.base
        if corr is not None:
            base = base.copy()
            base[0:3, 0:3] = base[0:3, 0:3] @ corr
        pred = predict_spatial(base, f.q, params)[1:6]
        chunks.append(((pred - f.targets)[f.mask]).ravel())
    if not chunks:
        return np.zeros(0)
    return np.concatenate(chunks)


@dataclasses.dataclass
class FitResult:
    """One regime's finished Gauss-Newton fit, with its own conditioning."""

    fit_rot: bool
    theta: np.ndarray
    gaps_m: np.ndarray
    rotvec: np.ndarray | None
    rot_angle_deg: float
    iterations: int
    converged: bool
    n_frames: int
    n_residuals: int
    rms_before_m: float
    rms_after_m: float
    singular_values: np.ndarray
    cond: float
    param_sd_m: np.ndarray


def gauss_newton(frames: list, fit_rot: bool,
                 max_frames: int = FIT_MAX_FRAMES) -> FitResult:
    """Fit the 5 gaps (+ optional base rotation) to one regime's frames.

    Plain Gauss-Newton with forward-difference Jacobians and a halving line
    search (the problem is linear in the gaps, so the search almost never
    engages — it exists so the rotation's mild nonlinearity can't diverge).
    The normal equations are solved by ``lstsq``: a rank-deficient Jacobian
    (insufficient excitation) then yields the *minimum-norm* step rather than
    garbage, and the reported singular values say so out loud (design §5).
    """
    if len(frames) > max_frames:
        # Even subsample: excitation coverage, not recency, is what matters.
        pick = np.unique(np.round(
            np.linspace(0, len(frames) - 1, max_frames)).astype(int))
        fit_frames = [frames[i] for i in pick]
    else:
        fit_frames = list(frames)

    theta = np.array(rp.PLATE_CHAIN_NOMINAL_M
                     + ((0.0, 0.0, 0.0) if fit_rot else ()), dtype=float)
    steps = np.array([FD_STEP_LEN_M] * 5
                     + ([FD_STEP_ROT_RAD] * 3 if fit_rot else []))
    r = residual_vector(fit_frames, theta, fit_rot)
    if r.size < theta.size:
        raise ValueError(
            f"only {r.size} residual components for {theta.size} parameters "
            f"— not enough usable frames to fit")
    rms_before = float(np.sqrt(np.mean(r * r)))
    cost = float(r @ r)
    jac = np.zeros((r.size, theta.size))
    iterations = 0
    converged = False
    for _ in range(GN_MAX_ITER):
        iterations += 1
        for j in range(theta.size):
            tp = theta.copy()
            tp[j] += steps[j]
            jac[:, j] = (residual_vector(fit_frames, tp, fit_rot) - r) / steps[j]
        dx = np.linalg.lstsq(jac, -r, rcond=None)[0]
        # Halving line search: accept the first step that reduces the cost.
        alpha = 1.0
        for _ in range(8):
            r_new = residual_vector(fit_frames, theta + alpha * dx, fit_rot)
            if float(r_new @ r_new) < cost:
                break
            alpha *= 0.5
        else:
            converged = True                   # no descent left: at the optimum
            break
        theta = theta + alpha * dx
        r = r_new
        cost = float(r @ r)
        if float(np.linalg.norm(alpha * dx)) < GN_STEP_TOL:
            converged = True
            break

    sv = np.linalg.svd(jac, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0.0 else float("inf")
    # 1-sigma parameter uncertainties from the linearized normal equations —
    # honest only to the extent the residuals are noise, which is exactly what
    # the verdict section evaluates; reported for conditioning, not gating.
    dof = max(1, r.size - theta.size)
    sigma2 = cost / dof
    cov = sigma2 * np.linalg.pinv(jac.T @ jac)
    param_sd = np.sqrt(np.maximum(np.diag(cov), 0.0))

    gaps, rotvec = _theta_split(theta, fit_rot)
    rms_after = float(np.sqrt(np.mean(r * r)))
    return FitResult(
        fit_rot=fit_rot, theta=theta, gaps_m=np.array(gaps),
        rotvec=None if rotvec is None else np.array(rotvec),
        rot_angle_deg=(0.0 if rotvec is None
                       else math.degrees(float(np.linalg.norm(rotvec)))),
        iterations=iterations, converged=converged, n_frames=len(fit_frames),
        n_residuals=r.size, rms_before_m=rms_before, rms_after_m=rms_after,
        singular_values=sv, cond=cond, param_sd_m=param_sd)


# --------------------------------------------------------------------------
# Residual statistics over ALL loaded frames
# --------------------------------------------------------------------------


def per_plate_residuals(frames: list, theta: np.ndarray,
                        fit_rot: bool) -> np.ndarray:
    """``(n_frames, 5)`` residual norms in metres, NaN where a plate is masked."""
    gaps, rotvec = _theta_split(np.asarray(theta, dtype=float), fit_rot)
    params = params_from_gaps(gaps)
    corr = exp_so3(rotvec) if rotvec is not None else None
    out = np.full((len(frames), 5), np.nan)
    for k, f in enumerate(frames):
        base = f.base
        if corr is not None:
            base = base.copy()
            base[0:3, 0:3] = base[0:3, 0:3] @ corr
        pred = predict_spatial(base, f.q, params)[1:6]
        norms = np.linalg.norm(pred - f.targets, axis=1)
        out[k, f.mask] = norms[f.mask]
    return out


def residual_stats(res: np.ndarray) -> dict:
    """Per-plate and overall RMS/max (metres) from a residual-norm table."""
    stats: dict = {"per_plate": {}}
    finite = np.isfinite(res)
    for p in range(5):
        col = res[finite[:, p], p]
        stats["per_plate"][p + 1] = {
            "rms_m": float(np.sqrt(np.mean(col ** 2))) if col.size else float("nan"),
            "max_m": float(col.max()) if col.size else float("nan"),
            "n": int(col.size),
        }
    allv = res[finite]
    stats["overall_rms_m"] = (float(np.sqrt(np.mean(allv ** 2)))
                              if allv.size else float("nan"))
    stats["overall_max_m"] = float(allv.max()) if allv.size else float("nan")
    stats["n"] = int(allv.size)
    return stats


def residual_correlations(frames: list, res: np.ndarray) -> list:
    """Pearson correlation of each plate's post-fit residual with each |q_j|.

    The design's "structured residuals" detector: an unmodelled geometric
    offset shows up as residual growth *with the excitation*, and on a
    single-joint campaign that is a near-perfect correlation with the driven
    joint's |q|.  Rows: ``(plate, joint_q_index, corr)``, strongest first.
    """
    if not frames:
        return []
    qabs = np.abs(np.stack([f.q for f in frames]))       # (n, 12)
    rows = []
    for p in range(5):
        r = res[:, p]
        ok = np.isfinite(r)
        if ok.sum() < 30:
            continue
        rv = r[ok]
        if float(rv.std()) == 0.0:
            continue
        for j in range(12):
            qv = qabs[ok, j]
            if float(qv.std()) == 0.0:
                continue
            c = float(np.corrcoef(rv, qv)[0, 1])
            if math.isfinite(c):
                rows.append((p + 1, j, c))
    rows.sort(key=lambda x: -abs(x[2]))
    return rows


def rest_residual_structure(frames: list, theta, fit_rot: bool) -> tuple:
    """Split the post-fit rest residual into per-plate bias vs pooled jitter.

    At rest nothing is excited, so the residual there has exactly two parts:
    frame-to-frame **jitter** (mocap noise propagated through base inference
    and the chain — the honest "mocap-noise level" of design §4.6) and a
    per-plate constant **bias** (geometry the 5-length fit could not absorb
    even unexcited, e.g. a lateral bracket/pivot offset).  Lumping them, as a
    plain rest RMS does, would let a real rest-visible model gap inflate its
    own yardstick and read as noise.  Returns ``({plate: bias_m}, jitter_m,
    n_rest_frames)``; plates with fewer than :data:`REST_MIN_FRAMES` rest
    solves are excluded from the bias map (their mean is a coin flip).
    """
    gaps, rotvec = _theta_split(np.asarray(theta, dtype=float), fit_rot)
    params = params_from_gaps(gaps)
    corr = exp_so3(rotvec) if rotvec is not None else None
    vecs: dict[int, list] = {p: [] for p in range(5)}
    n_rest = 0
    for f in frames:
        if f.t > REST_HEAD_S:
            continue
        n_rest += 1
        base = f.base
        if corr is not None:
            base = base.copy()
            base[0:3, 0:3] = base[0:3, 0:3] @ corr
        pred = predict_spatial(base, f.q, params)[1:6]
        diff = pred - f.targets
        for p in range(5):
            if f.mask[p]:
                vecs[p].append(diff[p])
    bias: dict[int, float] = {}
    jitter_sq_sum = 0.0
    jitter_n = 0
    for p, rows in vecs.items():
        if len(rows) < REST_MIN_FRAMES:
            continue
        arr = np.stack(rows)
        mean = arr.mean(axis=0)
        bias[p + 1] = float(np.linalg.norm(mean))
        jitter_sq_sum += float(np.sum((arr - mean) ** 2))
        jitter_n += arr.shape[0]
    jitter = (math.sqrt(jitter_sq_sum / jitter_n) if jitter_n
              else float("nan"))
    return bias, jitter, n_rest


def measured_spans(passes: list) -> np.ndarray:
    """Mean chain gaps over every chain_ok frame, all passes, metres.

    These are direct plate-to-plate distances — invariant to joint angles for
    a centre-true pivot (every twist axis passes through a centre), so the
    whole campaign averages legitimately.  The fitted gaps should land on
    these; the comparison column is the report's sanity anchor.
    """
    total = np.zeros(5)
    n = 0
    for p in passes:
        for jd in p.joints:
            idx = np.flatnonzero(jd.chain_ok)
            if not idx.size:
                continue
            d = np.linalg.norm(jd.u[idx, 1:, :] - jd.u[idx, :-1, :], axis=2)
            total += d.sum(axis=0)
            n += idx.size
    return total / n if n else np.full(5, np.nan)


# --------------------------------------------------------------------------
# EE plate (506) and the q10/q11 substitute check (design §4.4)
# --------------------------------------------------------------------------


def ee_analysis(frames: list, fit: FitResult) -> dict:
    """Lever-arm fit + prediction for body 506, or the named skip.

    The lever is the mean of ``(base @ fkine(q))^-1 @ p506`` over rest frames
    — a constant 3-vector in the last-plate frame by the rigid-mount model —
    then predicted through every frame with the *fitted* parameters.  Body
    506 was absent from the volume on the 2026-08-11 probe, so on today's
    data this returns the skip record; the code path stays live for the day
    the plate returns.
    """
    with_p6 = [f for f in frames if f.p6 is not None]
    if not with_p6:
        return {"status": "skipped",
                "why": ("body 506 absent from the stream and the marker record "
                        "(matches the 2026-08-11 probe); q10/q11 are validated "
                        "through plate 5's marker-inferred body orientation "
                        "instead — see the q10/q11 section")}
    params = params_from_gaps(fit.gaps_m)
    corr = exp_so3(fit.rotvec) if fit.rotvec is not None else None

    def _ee_frame(f):
        base = f.base
        if corr is not None:
            base = base.copy()
            base[0:3, 0:3] = base[0:3, 0:3] @ corr
        return base @ fkine(f.q, params)

    rest = [f for f in with_p6 if f.t <= REST_HEAD_S]
    if len(rest) < REST_MIN_FRAMES:
        return {"status": "skipped",
                "why": (f"body 506 present but only {len(rest)} rest frames "
                        f"carry it (need >= {REST_MIN_FRAMES}) — no lever fit")}
    levers = []
    for f in rest:
        t_ee = _ee_frame(f)
        levers.append(np.linalg.solve(t_ee, np.array([*f.p6, 1.0]))[0:3])
    lever = np.mean(np.stack(levers), axis=0)
    res = np.array([float(np.linalg.norm(
        (_ee_frame(f) @ np.array([*lever, 1.0]))[0:3] - f.p6))
        for f in with_p6])
    return {"status": "fitted",
            "lever_m": [float(v) for v in lever],
            "n_rest_frames": len(rest), "n_frames": len(with_p6),
            "rms_m": float(np.sqrt(np.mean(res ** 2))),
            "max_m": float(res.max())}


def q1011_analysis(passes: list) -> dict:
    """q[10:12] recorded vs :func:`q_from_frames` on the inferred frames.

    q10/q11 move no centre (design §1), so the centre-residual fit above is
    blind to them; their only observable is plate 5's *orientation*, which the
    marker-inferred frames carry.  ``q_from_frames`` applies the family-angle
    defaults itself, so ``infer_all``'s (7, 4, 4) output feeds it directly.
    The recorded q comes from the swing-only shipped pipeline, whose known
    multi-joint twist error (design §1) is part of what this delta shows — on
    a single-joint campaign the two should agree to mocap noise.
    """
    dq = []
    n_candidate = 0
    for p in passes:
        for jd in p.joints:
            if jd.inferred is None:
                continue
            idx = np.flatnonzero(jd.chain_ok & jd.inferred_valid[:, 0:6].all(axis=1))
            n_candidate += int(idx.size)
            for i in idx:
                q_inf = q_from_frames(jd.inferred[i])
                if q_inf is not None:
                    dq.append(jd.q[i, 10:12] - q_inf[10:12])
    if not dq:
        return {"status": "skipped",
                "why": ("no chain_ok frame had all six plates marker-solved "
                        "(regime (b) unavailable or dropouts)")}
    d = np.degrees(np.stack(dq))
    return {"status": "computed", "n_frames": len(dq),
            "n_candidate_frames": n_candidate,
            "rms_deg": [float(v) for v in np.sqrt(np.mean(d ** 2, axis=0))],
            "max_deg": [float(v) for v in np.abs(d).max(axis=0)]}


# --------------------------------------------------------------------------
# Plots (matplotlib is a plotting dependency only; its absence costs the
# PNGs, never the numbers)
# --------------------------------------------------------------------------


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt
    except Exception:                          # noqa: BLE001 - plots are optional
        return None


_REGIME_LABEL = {"streamed": "(a) streamed base",
                 "inferred": "(b) inferred base"}


def write_plots(out_dir: str, regime_data: dict, frames_by_regime: dict,
                lengths: dict, warnings: list) -> list:
    """The §4.6 PNGs: residual summary, fitted lengths, per-joint traces."""
    plt = _pyplot()
    if plt is None:
        warnings.append("matplotlib unavailable — PNGs skipped "
                        "(the markdown and JSON reports carry every number)")
        return []
    written = []

    # 1. Per-plate residual RMS, before/after, both regimes.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plates = np.arange(1, 6)
    n_groups = sum(1 for r in regime_data.values() if r.get("available"))
    width = 0.8 / max(1, 2 * n_groups)
    slot = 0
    for regime, data in regime_data.items():
        if not data.get("available"):
            continue
        for phase, alpha in (("before", 0.45), ("after", 1.0)):
            vals = [data["residuals"][phase]["per_plate"][p]["rms_m"] * 1000.0
                    for p in range(1, 6)]
            ax.bar(plates + (slot - (2 * n_groups - 1) / 2.0) * width, vals,
                   width=width, alpha=alpha,
                   label=f"{_REGIME_LABEL[regime]} {phase}")
            slot += 1
    ax.axhline(ACCEPT_RMS_M * 1000.0, color="k", lw=0.8, ls="--",
               label="5 mm acceptance target")
    ax.set_xticks(plates)
    ax.set_xlabel("plate")
    ax.set_ylabel("residual RMS [mm]")
    ax.set_title("Predicted u-joint centres vs measured — before/after fit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "residuals_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 2. Fitted lengths vs CAD vs measured spans (deltas in mm).
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(5)
    cad = np.asarray(lengths["cad_m"])
    series = [("measured span", lengths.get("measured_span_m"))]
    for regime in ("streamed", "inferred"):
        g = lengths.get(f"fitted_{regime}_m")
        if g is not None:
            series.append((f"fit {_REGIME_LABEL[regime]}", g))
    width = 0.8 / len(series)
    for k, (label, vals) in enumerate(series):
        if vals is None:
            continue
        dv = (np.asarray(vals) - cad) * 1000.0
        ax.bar(x + (k - (len(series) - 1) / 2.0) * width, dv, width=width,
               label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(GAP_NAMES)
    ax.set_ylabel("length - CAD nominal [mm]")
    ax.set_title("Chain lengths: measured and fitted vs CAD")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "lengths_fit.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # 3. Per-joint residual traces (post-fit solid, pre-fit faint) per regime.
    joints = sorted({f.joint for frames in frames_by_regime.values()
                     for f in frames})
    for joint in joints:
        avail = [r for r, d in regime_data.items() if d.get("available")]
        if not avail:
            break
        fig, axes = plt.subplots(len(avail), 1, figsize=(8, 3.0 * len(avail)),
                                 sharex=True, squeeze=False)
        for row, regime in enumerate(avail):
            ax = axes[row, 0]
            frames = frames_by_regime[regime]
            sel = [k for k, f in enumerate(frames) if f.joint == joint]
            t = np.array([frames[k].t for k in sel])
            for p in range(5):
                pre = regime_data[regime]["res_before"][sel, p] * 1000.0
                post = regime_data[regime]["res_after"][sel, p] * 1000.0
                (line,) = ax.plot(t, post, lw=1.0, label=f"plate {p + 1}")
                ax.plot(t, pre, lw=0.6, alpha=0.3, color=line.get_color())
            ax.set_ylabel("residual [mm]")
            ax.set_title(f"q{joint} — {_REGIME_LABEL[regime]} "
                         f"(solid post-fit, faint pre-fit)", fontsize=9)
            if row == 0:
                ax.legend(fontsize=7, ncol=5)
        axes[-1, 0].set_xlabel("t [s]")
        fig.tight_layout()
        path = os.path.join(out_dir, f"joint_{joint:02d}_residuals.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return repr(obj)


def _mm(v: float) -> str:
    return "nan" if not math.isfinite(v) else f"{v * 1000.0:.2f}"


def build_verdict(regime_data: dict, noise_m: float, noise_measured: bool,
                  frames_by_regime: dict) -> dict:
    """The design §4.6 answer: *parameter issue or model issue*.

    Judged on the best available regime, preferring (b) — the honest target:
    it carries no manual alignment, so residuals there are geometry, not
    Motive.  The yardstick is empirical: the post-fit residual RMS over the
    **rest frames** is what "mocap-noise level" actually means once marker
    noise has propagated through base-frame inference and the chain's lever
    arms.  "Parameters" when the drive-frame RMS stays within
    :data:`STRUCTURE_RATIO` x that rest level; otherwise "model", naming the
    worst plate and the strongest residual-vs-|q| correlation — the
    which-plate/which-joint pointer the design demands.
    """
    regime = None
    for cand in ("inferred", "streamed"):
        if regime_data.get(cand, {}).get("available"):
            regime = cand
            break
    if regime is None:
        return {"call": "no-data", "explanation": "neither regime had usable frames"}
    data = regime_data[regime]
    post = data["residuals"]["after"]["overall_rms_m"]
    frames = frames_by_regime[regime]
    res = data["res_after"]
    fit_d = data["fit"]
    fit_rot = fit_d["rotvec"] is not None
    theta = (np.concatenate([fit_d["gaps_m"], fit_d["rotvec"]])
             if fit_rot else np.asarray(fit_d["gaps_m"]))
    is_rest = np.array([f.t <= REST_HEAD_S for f in frames])
    rest_vals = res[is_rest][np.isfinite(res[is_rest])]
    drive_vals = res[~is_rest][np.isfinite(res[~is_rest])]
    rest_rms = (float(np.sqrt(np.mean(rest_vals ** 2)))
                if rest_vals.size else float("nan"))
    drive_rms = (float(np.sqrt(np.mean(drive_vals ** 2)))
                 if drive_vals.size else float("nan"))
    bias_map, rest_jitter, _n_rest = rest_residual_structure(frames, theta,
                                                             fit_rot)
    # The clamps keep a too-quiet rest window (or a noiseless synthetic
    # campaign, rest jitter exactly ~0) from turning CSV rounding into "model".
    noise = max(rest_jitter if math.isfinite(rest_jitter) else 0.0,
                NOISE_FLOOR_MIN_M)
    drive_boundary = STRUCTURE_RATIO * max(
        rest_rms if math.isfinite(rest_rms) else 0.0, NOISE_FLOOR_MIN_M)
    if bias_map:
        worst_bias_plate = max(bias_map, key=bias_map.get)
        worst_bias = bias_map[worst_bias_plate]
    else:
        worst_bias_plate, worst_bias = None, float("nan")
    # Two independent structure detectors: excitation-correlated (drive RMS
    # beyond the full rest level, bias included — motion added something) and
    # rest-visible (a per-plate constant bias the fit could not absorb even
    # unexcited, judged against the jitter alone).
    drive_structured = (drive_vals.size > 0 and drive_rms > drive_boundary)
    rest_structured = (worst_bias_plate is not None
                       and worst_bias > STRUCTURE_RATIO * noise)
    corr_rows = residual_correlations(frames, res)
    top_corr = corr_rows[0] if corr_rows else None
    per_plate = data["residuals"]["after"]["per_plate"]
    worst_plate = max(per_plate,
                      key=lambda p: (per_plate[p]["rms_m"]
                                     if math.isfinite(per_plate[p]["rms_m"])
                                     else -1.0))
    verdict: dict = {
        "judged_regime": regime,
        "post_fit_rms_mm": post * 1000.0,
        "rest_rms_mm": rest_rms * 1000.0,
        "rest_jitter_mm": rest_jitter * 1000.0,
        "rest_bias_mm": {p: b * 1000.0 for p, b in bias_map.items()},
        "drive_rms_mm": drive_rms * 1000.0,
        "drive_boundary_mm": drive_boundary * 1000.0,
        "pivot_jitter_floor_mm": noise_m * 1000.0,
        "pivot_jitter_measured": noise_measured,
        "acceptance_target_mm": ACCEPT_RMS_M * 1000.0,
        "acceptance_met": bool(post < ACCEPT_RMS_M),
        "worst_plate": int(worst_plate),
        "worst_plate_rms_mm": per_plate[worst_plate]["rms_m"] * 1000.0,
        "top_correlation": (None if top_corr is None else
                            {"plate": top_corr[0], "q_index": top_corr[1],
                             "corr": top_corr[2]}),
    }
    if not drive_vals.size:
        verdict["call"] = "no-excitation"
        verdict["explanation"] = (
            "every usable frame sits in the rest head — nothing was excited, "
            "so parameter-vs-model cannot be judged; run the drives")
    elif not (drive_structured or rest_structured):
        verdict["call"] = "parameters"
        verdict["explanation"] = (
            f"drive-frame post-fit RMS {_mm(drive_rms)} mm is within "
            f"{STRUCTURE_RATIO:g}x the rest-frame residual level "
            f"({_mm(rest_rms)} mm — the pipeline's own noise, base-frame "
            f"jitter included) and no plate holds a rest bias beyond "
            f"{STRUCTURE_RATIO:g}x the rest jitter ({_mm(rest_jitter)} mm): "
            f"the fitted lengths explain everything the mocap can resolve — "
            f"a parameter issue, now fitted")
    else:
        named = []
        if rest_structured:
            named.append(
                f"plate {worst_bias_plate} holds a constant "
                f"{_mm(worst_bias)} mm residual at rest (jitter "
                f"{_mm(rest_jitter)} mm) — geometry the 5 lengths cannot "
                f"express even unexcited, e.g. a lateral pivot/bracket offset")
        if drive_structured:
            line = (f"drive-frame RMS {_mm(drive_rms)} mm exceeds "
                    f"{STRUCTURE_RATIO:g}x the rest level ({_mm(rest_rms)} mm)"
                    f" — excitation adds unmodelled geometry, worst at plate "
                    f"{worst_plate} ({_mm(per_plate[worst_plate]['rms_m'])} mm"
                    f" RMS)")
            if top_corr is not None and abs(top_corr[2]) >= CORR_NAMED:
                line += (f"; plate {top_corr[0]} residual correlates with "
                         f"|q{top_corr[1]}| at r = {top_corr[2]:+.2f} — first "
                         f"candidate: a per-plate z-offset (design §4.5 geo-6)")
            named.append(line)
        verdict["call"] = "model"
        verdict["explanation"] = (
            "structure survives the 5-length"
            + (" + base-rotation" if fit_rot else "")
            + " fit — a model gap: " + "; ".join(named))

    # Tilt arbitration (2026-08-11): a "model" call on regime (b) is only as
    # honest as its base frame.  If one global base rotation absorbs the
    # structure — the signature of a base-plate marker-plane tilt in the
    # LOCK, not of arm geometry — the call is downgraded to parameters plus
    # a measured mocap finding, with the tilt angle reported.
    fwr = data.get("fit_with_rot")
    if (verdict["call"] == "model" and regime == "inferred"
            and fwr is not None and fwr.get("converged")):
        res2 = data["res_after_rot"]
        theta2 = np.concatenate([fwr["gaps_m"], fwr["rotvec"]])
        rest2 = res2[is_rest][np.isfinite(res2[is_rest])]
        drive2 = res2[~is_rest][np.isfinite(res2[~is_rest])]
        rest2_rms = (float(np.sqrt(np.mean(rest2 ** 2)))
                     if rest2.size else float("nan"))
        drive2_rms = (float(np.sqrt(np.mean(drive2 ** 2)))
                      if drive2.size else float("nan"))
        bias2, jitter2, _ = rest_residual_structure(frames, theta2, True)
        noise2 = max(jitter2 if math.isfinite(jitter2) else 0.0,
                     NOISE_FLOOR_MIN_M)
        boundary2 = STRUCTURE_RATIO * max(
            rest2_rms if math.isfinite(rest2_rms) else 0.0, NOISE_FLOOR_MIN_M)
        worst2 = max(bias2.values()) if bias2 else float("nan")
        still_structured = (
            (drive2.size > 0 and drive2_rms > boundary2)
            or (bias2 and worst2 > STRUCTURE_RATIO * noise2))
        verdict["tilt_arbitration"] = {
            "base_tilt_deg": fwr["rot_angle_deg"],
            "post_fit_rms_mm": fwr["rms_after_m"] * 1000.0,
            "rest_rms_mm": rest2_rms * 1000.0,
            "drive_rms_mm": drive2_rms * 1000.0,
            "rest_bias_mm": {p: b * 1000.0 for p, b in bias2.items()},
            "still_structured": bool(still_structured),
        }
        if not still_structured:
            verdict["call"] = "parameters"
            verdict["mocap_base_tilt_deg"] = fwr["rot_angle_deg"]
            verdict["acceptance_met"] = bool(fwr["rms_after_m"] < ACCEPT_RMS_M)
            verdict["explanation"] = (
                f"the structure the 5 lengths could not absorb is one global "
                f"base rotation of {fwr['rot_angle_deg']:.2f} deg — the base "
                f"plate's marker plane vs the mechanism axes, a LOCK finding "
                f"(mocap), not arm geometry: with that single tilt fitted, "
                f"post-fit RMS drops to {_mm(fwr['rms_after_m'])} mm "
                f"(rest {_mm(rest2_rms)} mm, drive {_mm(drive2_rms)} mm) and "
                f"no structure detector fires — a parameter issue plus a "
                f"measured mocap base tilt")
    return verdict


def write_markdown(path: str, report: dict) -> None:
    """The human half of the §4.6 report; every number mirrors the JSON."""
    L: list = []
    add = L.append
    add("# fkine hardware benchmark")
    add("")
    add(f"Generated {report['generated_wall']} by `fkine_benchmark.py` "
        f"(docs/fkine_design.md §4).")
    add("")
    add("## Inputs")
    add("")
    for p in report["passes"]:
        add(f"* `{p['path']}` — joints {p['joints']}, "
            f"{p['frames_loaded']} frames ({p['chain_ok_frames']} chain_ok), "
            f"locks: {p['locks'] or 'none'}")
        lc = p["lock_check"]
        if lc["verified"] or lc["skipped"]:
            add(f"  * lock validity (marker-frame §4.1): verified plates "
                f"{lc['verified'] or '[]'}; unverifiable (no rest markers) "
                f"{lc['skipped'] or '[]'}")
    m = report["merge"]
    add(f"* merge check (design §5): {m['passes']} pass(es), "
        + ("fingerprints agree" if m["ok"] else "REFUSED"))
    add("")
    add("## Residuals: predicted centres vs measured, plates 1-5 [mm]")
    add("")
    add("| plate | " + " | ".join(
        f"{_REGIME_LABEL[r]} pre RMS/max | post RMS/max"
        for r in ("streamed", "inferred")
        if report["regimes"][r]["available"]) + " |")
    n_avail = sum(1 for r in ("streamed", "inferred")
                  if report["regimes"][r]["available"])
    add("|---" * (1 + 2 * n_avail) + "|")
    for plate in range(1, 6):
        cells = [f"| {plate} "]
        for r in ("streamed", "inferred"):
            if not report["regimes"][r]["available"]:
                continue
            res = report["regimes"][r]["residuals"]
            b = res["before"]["per_plate"][plate]
            a = res["after"]["per_plate"][plate]
            cells.append(f"| {_mm(b['rms_m'])} / {_mm(b['max_m'])} "
                         f"| {_mm(a['rms_m'])} / {_mm(a['max_m'])} ")
        add("".join(cells) + "|")
    for r in ("streamed", "inferred"):
        reg = report["regimes"][r]
        if reg["available"]:
            res = reg["residuals"]
            add(f"* **{_REGIME_LABEL[r]}** overall RMS: "
                f"{_mm(res['before']['overall_rms_m'])} mm before -> "
                f"{_mm(res['after']['overall_rms_m'])} mm after fit "
                f"({res['after']['n']} plate-frames)")
        else:
            add(f"* **{_REGIME_LABEL[r]}**: unavailable — {reg['why']}")
    add("")
    add("## Fitted lengths [mm]")
    add("")
    header = "| gap | CAD nominal | measured span "
    for r in ("streamed", "inferred"):
        if report["regimes"][r]["available"]:
            header += f"| fit {_REGIME_LABEL[r]} (+/- 1 sd) "
    add(header + "|")
    ncol = header.count("|")
    add("|---" * ncol + "|")
    lengths = report["lengths"]
    for i, name in enumerate(GAP_NAMES):
        row = (f"| {name} | {lengths['cad_m'][i] * 1000.0:.3f} "
               f"| {_mm(lengths['measured_span_m'][i])} ")
        for r in ("streamed", "inferred"):
            reg = report["regimes"][r]
            if reg["available"]:
                sd = reg["fit"]["param_sd_m"][i]
                row += (f"| {reg['fit']['gaps_m'][i] * 1000.0:.3f} "
                        f"(+/- {_mm(sd)}) ")
        add(row + "|")
    add("")
    add("## Base frame")
    add("")
    sa = report["regimes"]["streamed"]
    if sa["available"] and sa["fit"]["rotvec"] is not None:
        add(f"* regime (a) fitted base correction: "
            f"{sa['fit']['rot_angle_deg']:.3f} deg (constant body-side SO(3) "
            f"— the manual-alignment error estimate)")
    elif sa["available"]:
        add("* regime (a): base rotation fit disabled (--no-base-rot)")
    if report["regimes"]["inferred"]["available"]:
        add("* regime (b) gets **no** base correction by design — its post-fit "
            "residual IS the test of the marker-inferred frame")
    add("")
    add("## Fit conditioning (design §5)")
    add("")
    for r in ("streamed", "inferred"):
        reg = report["regimes"][r]
        if not reg["available"]:
            continue
        fit = reg["fit"]
        add(f"* {_REGIME_LABEL[r]}: {fit['iterations']} iterations, "
            f"converged={fit['converged']}, {fit['n_frames']} fit frames, "
            f"cond(J) = {fit['cond']:.1f}")
        if fit["cond"] > COND_WARN:
            add(f"  * **poorly conditioned** (> {COND_WARN:g}): this campaign's "
                f"excitation does not constrain every parameter — consider the "
                f"design §5 fallback (2-3 drives at 5 psi, annotated as a "
                f"data-collection pass; its MISMATCH verdicts are expected and "
                f"must be read as data, not as a map verdict)")
    add("")
    add("## EE plate (body 506)")
    add("")
    ee = report["ee"]
    if ee["status"] == "fitted":
        add(f"* lever arm (last-plate frame): "
            f"{['%.4f' % v for v in ee['lever_m']]} m from "
            f"{ee['n_rest_frames']} rest frames; residual over "
            f"{ee['n_frames']} frames: RMS {_mm(ee['rms_m'])} mm, "
            f"max {_mm(ee['max_m'])} mm")
    else:
        add(f"* skipped: {ee['why']}")
    add("")
    add("## q10/q11 (no centre moves — validated via plate 5's marker frame)")
    add("")
    qq = report["q1011"]
    if qq["status"] == "computed":
        add(f"* over {qq['n_frames']} all-plates-solved chain_ok frames: "
            f"recorded q10/q11 vs `q_from_frames` differ by RMS "
            f"{qq['rms_deg'][0]:.4f} / {qq['rms_deg'][1]:.4f} deg "
            f"(max {qq['max_deg'][0]:.4f} / {qq['max_deg'][1]:.4f} deg)")
    else:
        add(f"* skipped: {qq['why']}")
    add("")
    add("## Per-joint recording regimes")
    add("")
    add("| joint | frames | chain_ok | marker sets | labeled flags | "
        "3-marker solves | solve gate counts |")
    add("|---|---|---|---|---|---|---|")
    for row in report["per_joint"]:
        add(f"| q{row['q_index']} | {row['frames']} | {row['chain_ok']} "
            f"| {row['with_sets']} | {row['labeled']} | {row['three_marker']} "
            f"| {row['reasons'] or '-'} |")
    add("")
    add("## Verdict — parameter issue or model issue (design §4.6)")
    add("")
    v = report["verdict"]
    add(f"* **{v['call'].upper()}** (judged on regime "
        f"{_REGIME_LABEL.get(v.get('judged_regime'), 'n/a')})")
    add(f"* {v.get('explanation', '')}")
    if "rest_rms_mm" in v:
        add(f"* post-fit residual RMS: rest frames {v['rest_rms_mm']:.2f} mm "
            f"(jitter {v['rest_jitter_mm']:.2f} mm — the pipeline's noise "
            f"level) vs drive frames {v['drive_rms_mm']:.2f} mm (boundary "
            f"{v['drive_boundary_mm']:.2f} mm); streamed-pivot rest jitter "
            f"{v['pivot_jitter_floor_mm']:.2f} mm "
            f"({'measured' if v['pivot_jitter_measured'] else 'assumed'})")
        if v.get("rest_bias_mm"):
            worst = max(v["rest_bias_mm"], key=v["rest_bias_mm"].get)
            add(f"* per-plate constant rest bias [mm]: "
                + ", ".join(f"plate {p}: {b:.2f}"
                            for p, b in sorted(v["rest_bias_mm"].items()))
                + f" (worst: plate {worst})")
    if v.get("top_correlation"):
        tc = v["top_correlation"]
        add(f"* strongest residual-vs-|q| correlation: plate {tc['plate']} vs "
            f"q{tc['q_index']} (r = {tc['corr']:+.2f})")
    add(f"* acceptance target (reported, not gated): post-fit RMS "
        f"{v.get('post_fit_rms_mm', float('nan')):.2f} mm vs 5 mm -> "
        f"{'MET' if v.get('acceptance_met') else 'NOT MET'}")
    add("")
    if report["warnings"]:
        add("## Warnings")
        add("")
        for w in report["warnings"]:
            add(f"* {w}")
        add("")
    if report["plots"]:
        add("## Plots")
        add("")
        for pth in report["plots"]:
            add(f"* `{os.path.basename(pth)}`")
        add("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Offline fkine hardware benchmark (docs/fkine_design.md sec. 4)")
    ap.add_argument("campaign_dirs", nargs="+",
                    help="campaign results directories (joint_NN_*.csv inside); "
                         "several = the design sec. 5 merge, fingerprint-checked")
    ap.add_argument("--locks", default=None,
                    help="locks.json applied to every pass "
                         "(default: <pass>/locks.json when present)")
    ap.add_argument("--out-dir", default=None,
                    help="report directory (default: <first pass>/fkine_benchmark)")
    ap.add_argument("--fit-max-frames", type=int, default=FIT_MAX_FRAMES,
                    help="even subsample cap for the Gauss-Newton fit; reported "
                         "residuals always use every loaded frame")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth recorded frame at load time")
    ap.add_argument("--no-base-rot", action="store_true",
                    help="disable the regime-(a) base SO(3) correction")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the PNGs (markdown + JSON still written)")
    return ap.parse_args(argv)


def _apply_stride(p: CampaignPass, stride: int) -> None:
    if stride <= 1:
        return
    for jd in p.joints:
        sel = np.arange(0, jd.n, stride)
        jd.t, jd.frame, jd.q, jd.u = (jd.t[sel], jd.frame[sel], jd.q[sel],
                                      jd.u[sel])
        jd.chain_ok = jd.chain_ok[sel]
        jd.streamed = jd.streamed[sel]
        jd.streamed_present = jd.streamed_present[sel]
        jd.markers = [jd.markers[i] for i in sel]
        jd.flags = [jd.flags[i] for i in sel]


def main(argv=None) -> int:
    args = _parse_args(argv)
    warnings: list = []

    # ---- load ------------------------------------------------------------
    passes: list = []
    for d in args.campaign_dirs:
        try:
            p = load_pass(d, args.locks)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"fkine_benchmark: {exc}", file=sys.stderr)
            return EXIT_NO_DATA
        _apply_stride(p, args.stride)
        warnings.extend(p.warnings)
        passes.append(p)

    # ---- refusals first: stale locks, unmergeable passes -------------------
    lock_checks = []
    for p in passes:
        refusals, verified, skipped = check_locks(p)
        if refusals:
            for r in refusals:
                print(f"fkine_benchmark: REFUSED: {r}", file=sys.stderr)
            return EXIT_REFUSED
        lock_checks.append({"verified": verified, "skipped": skipped})
        if skipped and p.locks is not None:
            warnings.append(
                f"{p.path}: plates {skipped} had no all-tracked rest markers in "
                f"the first {REST_HEAD_S:g} s — their locks are unverified "
                f"against this data (marker-frame §4.1)")
    merge_problems = check_merge(passes)
    if merge_problems:
        for m in merge_problems:
            print(f"fkine_benchmark: REFUSED: {m}", file=sys.stderr)
        return EXIT_REFUSED

    # ---- inference + frames ------------------------------------------------
    for p in passes:
        infer_pass_frames(p)
    frames_by_regime = {r: build_frames(passes, r)
                        for r in ("streamed", "inferred")}
    if not any(frames_by_regime.values()):
        print("fkine_benchmark: no usable chain_ok frames in either regime — "
              "nothing to benchmark", file=sys.stderr)
        return EXIT_NO_DATA

    # ---- fits + residual tables --------------------------------------------
    theta_nominal = np.array(rp.PLATE_CHAIN_NOMINAL_M, dtype=float)
    regime_data: dict = {}
    for regime in ("streamed", "inferred"):
        frames = frames_by_regime[regime]
        if not frames:
            why = ("no poses CSV / no plate-0 streamed pose"
                   if regime == "streamed"
                   else "no locks, no marker CSVs, or plate 0 never solved")
            regime_data[regime] = {"available": False, "why": why}
            continue
        fit_rot = (regime == "streamed") and not args.no_base_rot
        try:
            fit = gauss_newton(frames, fit_rot, max_frames=args.fit_max_frames)
        except (ValueError, np.linalg.LinAlgError) as exc:
            regime_data[regime] = {"available": False, "why": f"fit failed: {exc}"}
            warnings.append(f"regime {regime}: fit failed ({exc})")
            continue
        theta_before = np.concatenate(
            [theta_nominal, np.zeros(3)]) if fit_rot else theta_nominal
        res_before = per_plate_residuals(frames, theta_before, fit_rot)
        res_after = per_plate_residuals(frames, fit.theta, fit_rot)
        regime_data[regime] = {
            "available": True,
            "n_frames": len(frames),
            "fit": {
                "gaps_m": fit.gaps_m, "rotvec": fit.rotvec,
                "rot_angle_deg": fit.rot_angle_deg,
                "iterations": fit.iterations, "converged": fit.converged,
                "n_frames": fit.n_frames, "n_residuals": fit.n_residuals,
                "rms_before_m": fit.rms_before_m,
                "rms_after_m": fit.rms_after_m,
                "singular_values": fit.singular_values, "cond": fit.cond,
                "param_sd_m": fit.param_sd_m,
            },
            "residuals": {"before": residual_stats(res_before),
                          "after": residual_stats(res_after)},
            "res_before": res_before,
            "res_after": res_after,
        }
        if regime == "inferred" and not args.no_base_rot:
            # The tilt arbitration (2026-08-11 hardware finding): regime (b)
            # is judged WITHOUT a base rotation by design — that absence is
            # the test of the inferred frame — but when it fails that test,
            # the failure must be measured, not just declared.  A second,
            # rotation-enabled fit tells tilt (one global rotation absorbs
            # the residual: a lock-frame finding — the base plate's marker
            # plane vs the mechanism axes) apart from true arm geometry
            # (nothing absorbs it).
            try:
                fit2 = gauss_newton(frames, True,
                                    max_frames=args.fit_max_frames)
                regime_data[regime]["fit_with_rot"] = {
                    "gaps_m": fit2.gaps_m, "rotvec": fit2.rotvec,
                    "rot_angle_deg": fit2.rot_angle_deg,
                    "converged": fit2.converged,
                    "rms_after_m": fit2.rms_after_m,
                }
                regime_data[regime]["res_after_rot"] = per_plate_residuals(
                    frames, fit2.theta, True)
                regime_data[regime]["residuals_with_rot"] = residual_stats(
                    regime_data[regime]["res_after_rot"])
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.append(f"regime inferred: tilt-arbitration fit "
                                f"failed ({exc})")

    # ---- sections ----------------------------------------------------------
    noise_m, noise_n, noise_measured = noise_floor_m(passes)
    if not noise_measured:
        warnings.append(f"no usable rest head in any pass — the verdict uses "
                        f"the {NOISE_FALLBACK_M * 1000.0:.1f} mm fallback noise "
                        f"floor instead of a measured one")
    lengths = {
        "names": list(GAP_NAMES),
        "cad_m": [float(v) for v in rp.PLATE_CHAIN_NOMINAL_M],
        "measured_span_m": [float(v) for v in measured_spans(passes)],
    }
    for regime in ("streamed", "inferred"):
        if regime_data[regime].get("available"):
            lengths[f"fitted_{regime}_m"] = [
                float(v) for v in regime_data[regime]["fit"]["gaps_m"]]

    ee_regime = ("inferred" if regime_data["inferred"].get("available")
                 else "streamed")
    if regime_data[ee_regime].get("available"):
        # ee_analysis re-reads fit as a FitResult-shaped dict; rebuild minimal.
        fit_d = regime_data[ee_regime]["fit"]
        fit_obj = FitResult(
            fit_rot=fit_d["rotvec"] is not None, theta=np.zeros(0),
            gaps_m=np.asarray(fit_d["gaps_m"]), rotvec=fit_d["rotvec"],
            rot_angle_deg=fit_d["rot_angle_deg"], iterations=0, converged=True,
            n_frames=0, n_residuals=0, rms_before_m=0.0, rms_after_m=0.0,
            singular_values=np.zeros(0), cond=0.0, param_sd_m=np.zeros(0))
        ee = ee_analysis(frames_by_regime[ee_regime], fit_obj)
    else:
        ee = {"status": "skipped", "why": "no regime available"}
    q1011 = q1011_analysis(passes)

    per_joint = []
    for p in passes:
        for jd in p.joints:
            per_joint.append({
                "pass": p.path, "q_index": jd.q_index, "frames": jd.n,
                "chain_ok": int(jd.chain_ok.sum()),
                "with_sets": sum(1 for m in jd.markers if m is not None),
                "labeled": sum(1 for f in jd.flags if f is not None),
                "three_marker": (int(jd.three_marker.sum())
                                 if jd.three_marker is not None else 0),
                "reasons": dict(sorted(jd.reasons.items())),
                "epochs": sorted(jd.epochs),
                "verdict": jd.meta.get("verdict", ""),
            })
            if len(jd.epochs) > 1:
                warnings.append(
                    f"q{jd.q_index}: {len(jd.epochs)} mapping epochs inside one "
                    f"recording — the Motive roster changed mid-joint; treat its "
                    f"marker-derived numbers with suspicion")

    verdict = build_verdict(regime_data, noise_m, noise_measured,
                            frames_by_regime)

    # ---- outputs -----------------------------------------------------------
    out_dir = args.out_dir or os.path.join(args.campaign_dirs[0],
                                           "fkine_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "generated_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        "design": "docs/fkine_design.md sec. 4 (rev 2)",
        "passes": [{
            "path": p.path,
            "joints": [jd.q_index for jd in p.joints],
            "frames_loaded": int(sum(jd.n for jd in p.joints)),
            "chain_ok_frames": int(sum(jd.chain_ok.sum() for jd in p.joints)),
            "locks": p.locks_path,
            "locks_meta": {k: p.locks_meta.get(k) for k in
                           ("captured_wall", "regime", "mapping_epoch")
                           if k in p.locks_meta},
            "lock_check": lock_checks[i],
            "rest_spans_m": (None if p.rest_spans_m is None
                             else [float(v) for v in p.rest_spans_m]),
            "base_pos_m": (None if p.base_pos_m is None
                           else [float(v) for v in p.base_pos_m]),
        } for i, p in enumerate(passes)],
        "merge": {"passes": len(passes), "ok": True,
                  "span_tol_m": MERGE_SPAN_TOL_M,
                  "base_pos_tol_m": MERGE_BASE_POS_TOL_M,
                  "base_rot_tol_deg": MERGE_BASE_ROT_TOL_DEG},
        "regimes": {r: {k: v for k, v in d.items()
                        if k not in ("res_before", "res_after")}
                    for r, d in regime_data.items()},
        "lengths": lengths,
        "noise": {"rest_jitter_m_median": noise_m, "n_samples": noise_n,
                  "measured": noise_measured},
        "ee": ee,
        "q1011": q1011,
        "per_joint": per_joint,
        "verdict": verdict,
        "warnings": warnings,
        "plots": [],
    }
    if not args.no_plots:
        report["plots"] = write_plots(out_dir, regime_data, frames_by_regime,
                                      lengths, warnings)
        report["warnings"] = warnings          # write_plots may append

    json_path = os.path.join(out_dir, "fkine_benchmark.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
        fh.write("\n")
    md_path = os.path.join(out_dir, "fkine_benchmark.md")
    write_markdown(md_path, report)

    print(f"fkine_benchmark: verdict {verdict.get('call', '?').upper()} — "
          f"{verdict.get('explanation', '')}")
    print(f"fkine_benchmark: wrote {md_path}, {json_path}"
          + (f", {len(report['plots'])} PNGs" if report["plots"] else ""))
    return EXIT_OK


if __name__ == "__main__":                     # pragma: no cover
    sys.exit(main())
