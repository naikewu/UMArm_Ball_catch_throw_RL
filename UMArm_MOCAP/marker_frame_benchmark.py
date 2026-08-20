"""Offline marker-frame benchmark: is inferred better than manual alignment?

Implements ``docs/marker_frame_design.md`` §8.  Consumes a probe static
capture (``probe_markers.csv`` / ``probe_poses.csv``, §6) and/or a joint
campaign's recordings (``joint_NN_markers.csv`` / ``joint_NN_poses.csv`` /
``joint_NN_mocap.csv``, §7) plus a locks JSON, and writes markdown + JSON +
PNG plots under ``--out-dir`` (default ``<campaign>/marker_frame_benchmark/``,
i.e. ``results/<campaign>/marker_frame_benchmark/`` for a campaign that lives
under ``results/``)::

    python UMArm_MOCAP/marker_frame_benchmark.py --locks locks.json
        [--probe DIR] [--campaign DIR] [--out-dir DIR]
        [--rest-seconds 1.5] [--drop-one-max-frames 400]

Everything runs offline on recorded CSVs — no serial port, no NatNet socket,
no display (matplotlib Agg).  The six report sections are the design's §8
list, numbered identically:

1. **Rectangle validity** per plate, against the 0.02 mm wand-error context.
2. **Static precision**: origin jitter sd (mm) and axis jitter (deg) over the
   rest windows — inferred vs streamed, side by side.
3. **Manual-alignment error** (the point of the feature): the constant part
   of ``R_streamed^T @ R_inferred_body`` per plate (angle + axis) and the
   pivot offset ``|p_streamed - p_inferred|`` at rest.
4. **Drop-one robustness**: synthetic drops on real 4-marker frames, plus the
   *real* dropouts encountered, keyed to the per-joint flag regime (design
   §5 / review finding ops-9 — a flags-unknown joint cannot see dropouts and
   must say so instead of reporting zero).
5. **Dynamic consistency**: q from streamed pivots (the shipped recording) vs
   q from inferred origins (same swing-only reader, marker-derived centres)
   vs ``q_from_frames`` (full frames, design §4.3); Delta-q traces; chain-span
   stability streamed vs inferred (an off-centre pivot modulates spans with
   joint angle; a diagonal-intersection origin should not).
6. **Model verification (pass/fail)**: the §2 hardware model is *assumed* by
   every section above, so it is verified last and loudly (design D7).
   Co-rigid pairs {1,2}/{3,4} (and {5,6} if plate 6 returns) must keep a
   constant relative body rotation over every drive to sub-degree; each
   drive's measured rotation axis must match the predicted revolute axis.
   A failure means the model — mounting side, family angle, or axis layout —
   is wrong, and every inferred-frame number in this report inherits that
   error: the report says so instead of presenting garbage politely.

------------------------------------------------------------------------------
LOCK VALIDITY (design §4.1 binding, review finding ops-5)
------------------------------------------------------------------------------
Before any section is computed, the locks are re-verified against the first
rest frames of **every** data source given (``marker_frame.
verify_lock_against_rest``: the lock's rigid marker template registered onto
the source's own rest-frame means, refused past the RMS noise bound).  A
mismatch beyond noise means the Motive session changed since the lock was
minted —
the benchmark **refuses** (exit 2) rather than quantifying a fiction.  A
plate whose rest data is too thin to check is reported as unverified, not
silently trusted.

------------------------------------------------------------------------------
CONVENTIONS THE NUMBERS STAND ON
------------------------------------------------------------------------------
* Body frames: ``R_body = R_inferred @ Rz(-phi_lock)`` with each plate's
  family angle taken from its lock (design §2); this is what makes co-rigid
  pairs comparable and what §4.3 consumes.
* On a non-square bracket the locked x is quantized to the +45-deg-rotated
  half-diagonal directions, a constant offset from the true bracket x (a lock
  convention, design §4.1.3).  Section 3's "constant part" absorbs it by
  construction — the *constant* rotation between streamed and inferred is
  exactly what manual alignment error plus lock convention look like, and the
  jitter/constancy numbers are unaffected.
* q is repeatable, not absolute (``mocap_constants`` header): every q variant
  in section 5 is re-zeroed on its own rest-window mean before differencing,
  so mounting-dependent zero offsets cancel and the traces compare *motion*,
  not conventions.
* Timestamps: campaign ``t_s`` shares ``trace.t0`` across one joint's CSVs;
  the probe's starts at its window head.  Cross-file alignment within a
  source is by ``frame`` number throughout (the design's own advice).

Exit codes (mirroring the probe's convention): 0 clean; 1 unusable
invocation / no data; 2 lock validity refused; 3 model verification FAILED
(reports are still written — the failure is the finding).
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import math
import os
import re
import sys
import time

import numpy as np

# Agg before pyplot, unconditionally: this runs on a headless bench PC and in
# CI (same rule and reason as joint_verification.py:135).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

_HERE = os.path.dirname(os.path.abspath(__file__))

try:  # see the note in mocap_to_q.py — both import styles must work
    from . import marker_frame as mf
    from . import mocap_constants as mc
    from .mocap_to_q import mocap_to_q, quat_xyzw_to_matrix, ujoint_angles
except ImportError:  # pragma: no cover - flat style when run as a script
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import marker_frame as mf  # type: ignore[no-redef]
    import mocap_constants as mc  # type: ignore[no-redef]
    from mocap_to_q import (  # type: ignore[no-redef]
        mocap_to_q, quat_xyzw_to_matrix, ujoint_angles)


# --------------------------------------------------------------------------
# Constants (each with its source; none are new policy)
# --------------------------------------------------------------------------

#: Rest window at the head of each campaign joint recording.  The campaign's
#: rest capture is ``arm_constants.JOINT_REST_CAPTURE_S`` = 1.5 s and every
#: joint CSV's ``t_s`` starts at that capture's start (``trace.t0``,
#: joint_verification._write_joint_files).  Mirrored, not imported: this
#: package must not depend on the control repo (design D8 keeps the
#: dependency graph one-way).  A probe capture is rest end to end (§6 refuses
#: to lock a moving arm), so probe sources ignore this and use every frame.
REST_SECONDS_DEFAULT = 1.5

#: Co-rigid constancy bound (design §8.6: "constant relative rotation ...
#: to sub-degree").  Gated on the p95 of the per-frame deviation from the
#: mean relative rotation — robust to a single glitch frame — with the max
#: reported beside it.
CORIGID_P95_MAX_DEG = 1.0

#: Rotation-axis agreement bound for the per-drive check (§8.6), applied to
#: the CROSS-TALK-AWARE residual: the measured pair delta rotation is
#: compared against the delta of a model reconstruction built from the §4.3
#: four-angle decomposition (t1/t2 from plate ORIGINS, t3/t4 from the
#: orientation residual through the 45-deg family).  Gravity really does
#: move the other angles of the measured pair during a single-agonist drive
#: (q0's within-pair cross-talk measured 0.53 on 2026-08-11 — a naive
#: single-axis compare reads that as a 26-deg "axis error" and falsifies a
#: correct model), so the naive single-axis number is reported but not
#: gated.  What IS gated is the part the model cannot express: wrong
#: families, wrong mounting sides, or positions disagreeing with
#: orientations through the model axes.  The nearest wrong answers (45-deg
#: family error, 90-deg joint-map error) sit far outside 10 deg.
AXIS_TOL_DEG = 10.0

#: Minimum peak excursion before a drive's rotation axis is worth reading.
#: Mirrors ``arm_constants.JOINT_DQ_MIN_DEG`` (2 deg) — the campaign's own
#: smallest believed motion; below it the "axis" is resting marker noise.
AXIS_MIN_EXCURSION_DEG = 2.0

#: Cap on synthetically-dropped frames per plate per source (section 4).
#: Every frame of a 12-joint campaign would be ~2 million extra solves for
#: statistics that converge within a few hundred; frames are strided, never
#: cherry-picked, and the count is reported.
DROP_ONE_MAX_FRAMES = 400

#: Minimum frames for a jitter/constancy statistic to mean anything.
MIN_WINDOW_FRAMES = 2
#: Minimum co-usable frames before a co-rigid pair is evaluated per source.
MIN_PAIR_FRAMES = 5

#: Motive wand-error context for section 1 (design §8.1): the calibrated
#: volume reports ~0.02 mm mean wand error, so rectangle stats far above the
#: marker-jitter scale are geometry (bent arms, mislabeled assets), not
#: camera noise.
WAND_ERROR_MM = 0.02

MD_NAME = "marker_frame_benchmark.md"
JSON_NAME = "marker_frame_benchmark.json"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_STALE_LOCK = 2
EXIT_MODEL_FALSIFIED = 3

#: Predicted revolute axes per within-segment joint index k = q_index % 4, in
#: the *proximal plate's body frame* (design §8.6; axis directions from the
#: fkine twists, ``docs/fkine_design.md`` §1): t1/t2 about x/y, t3/t4 about
#: (x+y)/sqrt2 and (-x+y)/sqrt2.
_S2 = 2.0 ** -0.5
PREDICTED_AXES = {
    0: np.array([1.0, 0.0, 0.0]),
    1: np.array([0.0, 1.0, 0.0]),
    2: np.array([_S2, _S2, 0.0]),
    3: np.array([-_S2, _S2, 0.0]),
}

#: Co-rigid plate pairs (design §2): same connector/EE body, Tz(-JD) apart.
CORIGID_PAIRS = ((1, 2), (3, 4), (5, 6))

#: Fixed role -> color map (Okabe-Ito, CVD-safe; one role keeps one hue in
#: every figure, never cycled — dataviz rule "color follows the entity").
COLOR_INFERRED = "#0072B2"   # blue: everything marker-inferred
COLOR_STREAMED = "#E69F00"   # orange: everything Motive-streamed
COLOR_FRAMES = "#009E73"     # green: the full-frame q variant (q_from_frames)
COLOR_GATE = "#D55E00"       # vermillion: thresholds / gates

_JOINT_MARKERS_RE = re.compile(r"^joint_(\d{2})_markers\.csv$")


# --------------------------------------------------------------------------
# Small numerics
# --------------------------------------------------------------------------


def _rz(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_angle_deg(r: np.ndarray) -> float:
    """Angle of a rotation matrix, degrees, via the trace (clipped for noise)."""
    c = (float(np.trace(np.asarray(r)[0:3, 0:3])) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def _skew_vec(r: np.ndarray) -> np.ndarray:
    """The skew-symmetric part of a rotation as a vector: 2 sin(angle) * axis."""
    return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def mean_rotation(mats) -> np.ndarray:
    """Chordal mean of rotation matrices: element-wise mean projected onto
    SO(3) by SVD (the orthogonal-Procrustes projection).  Exact for identical
    inputs, and the right notion of "constant part" for a cloud of nearby
    rotations — deviations are then pure jitter, not projection artifacts.
    """
    m = np.mean(np.asarray(mats, dtype=float), axis=0)
    u, _, vt = np.linalg.svd(m)
    d = float(np.sign(np.linalg.det(u @ vt)))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _dist_stats(vals) -> dict | None:
    """{n, median, p95, max} over the finite values, or None when empty."""
    v = np.asarray([x for x in np.asarray(vals, dtype=float).ravel()
                    if math.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95.0)), "max": float(v.max())}


def _fmt_stats(d: dict | None, nd: int = 3) -> str:
    if d is None:
        return "no data"
    return (f"med {d['median']:.{nd}f} / p95 {d['p95']:.{nd}f} / "
            f"max {d['max']:.{nd}f} (n={d['n']})")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    return repr(obj)


# --------------------------------------------------------------------------
# Ingestion: the CSV schemas of mocap_probe.write_capture_csvs and
# joint_verification._write_joint_files / _write_marker_files
# --------------------------------------------------------------------------


def read_markers_csv(path: str):
    """The long markers format -> per-frame transport-shaped structures.

    Returns ``(t, frame_no, epochs, markers, flags)`` sorted by frame number:
    ``markers[k]`` is ``dict plate -> (m, 3)`` or ``None`` (the
    ``-1,-1,nan,nan,nan,-`` placeholder row = marker sets absent that frame);
    ``flags[k]`` is ``dict plate -> (m,) uint8`` holding only the plates whose
    ``tracked`` column carried digits, or ``None`` when no plate did (labeled
    markers absent — the ``?`` regime).  This reconstructs exactly the ring
    semantics the writers flattened (design §7), with one documented loss: a
    streamed-but-empty marker dict writes zero rows, so such frames do not
    appear here at all — nothing could have been inferred from them anyway.
    """
    frames: dict[int, dict] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if header is None or header[0:2] != ["t_s", "frame"]:
            raise ValueError(f"{path}: not a markers CSV (header {header!r})")
        for row in rdr:
            if not row:
                continue
            frame = int(row[1])
            rec = frames.setdefault(frame, {"t": float(row[0]),
                                            "epoch": int(row[2]), "plates": {}})
            if row[8] == "-":                       # markers-None placeholder
                continue
            plate, marker = int(row[3]), int(row[4])
            rec["plates"].setdefault(plate, []).append(
                (marker, float(row[5]), float(row[6]), float(row[7]), row[8]))

    order = sorted(frames)
    t = np.array([frames[f]["t"] for f in order], dtype=float)
    frame_no = np.array(order, dtype=int)
    epochs = np.array([frames[f]["epoch"] for f in order], dtype=int)
    markers, flags = [], []
    for f in order:
        plates = frames[f]["plates"]
        if not plates:
            markers.append(None)
            flags.append(None)
            continue
        mdict: dict[int, np.ndarray] = {}
        fdict: dict[int, np.ndarray] = {}
        for plate, rows in plates.items():
            rows.sort(key=lambda r: r[0])
            mdict[plate] = np.array([[r[1], r[2], r[3]] for r in rows])
            chars = [r[4] for r in rows]
            if all(ch in ("0", "1") for ch in chars):
                fdict[plate] = np.array([int(ch) for ch in chars], dtype=np.uint8)
        markers.append(mdict)
        flags.append(fdict if fdict else None)
    return t, frame_no, epochs, tuple(markers), tuple(flags)


def read_poses_csv(path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The streamed-poses format -> ``frame -> ((7, 4, 4) poses, (7,) valid)``.

    A row that is exactly the identity pose (quat 0,0,0,1 at the origin — the
    receiver's never-updated initial value, and what the writers emit for an
    absent body) is left as the identity with ``valid = False``: an identity
    row must never be mistaken for a solved pose (same rule as
    ``marker_frame.infer_all``'s validity mask).
    """
    n = mc.N_USED_RIGID_BODIES
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if header is None or "qx" not in header:
            raise ValueError(f"{path}: not a poses CSV (header {header!r})")
        for row in rdr:
            if not row:
                continue
            frame, plate = int(row[1]), int(row[2])
            if not 0 <= plate < n:
                continue
            if frame not in out:
                out[frame] = (np.tile(np.eye(4), (n, 1, 1)),
                              np.zeros(n, dtype=bool))
            poses, valid = out[frame]
            quat = [float(row[3]), float(row[4]), float(row[5]), float(row[6])]
            pos = [float(row[7]), float(row[8]), float(row[9])]
            if quat == [0.0, 0.0, 0.0, 1.0] and pos == [0.0, 0.0, 0.0]:
                continue                            # absent body, stays invalid
            try:
                rot = quat_xyzw_to_matrix(quat)
            except ValueError:
                continue                            # zero quat: unsolved body
            poses[plate, 0:3, 0:3] = rot
            poses[plate, 0:3, 3] = pos
            valid[plate] = True
    return out


def read_mocap_csv(path: str) -> dict[int, tuple[np.ndarray, bool]]:
    """``joint_NN_mocap.csv`` -> ``frame -> (q (12,), chain_ok)``.

    Only the shipped q and the plate-chain gate are consumed: the u-centre
    columns duplicate the poses CSV's positions (both are written from the
    same ``_incoming`` slice) and the poses CSV is the one that also carries
    orientation.
    """
    out: dict[int, tuple[np.ndarray, bool]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if header is None or "q0_rad" not in header:
            raise ValueError(f"{path}: not a mocap CSV (header {header!r})")
        iq = header.index("q0_rad")
        ic = header.index("chain_ok")
        for row in rdr:
            if not row:
                continue
            q = np.array([float(v) for v in row[iq:iq + mc.NUM_JOINTS]])
            out[int(row[1])] = (q, row[ic] == "1")
    return out


def load_locks(path: str) -> tuple[dict[int, "mf.PlateLock"], dict]:
    """locks.json (probe §6 format) -> ``({plate: PlateLock}, meta)``."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if "plates" not in payload:
        raise ValueError(f"{path}: no 'plates' key -- not a locks JSON")
    locks = {int(p): mf.PlateLock.from_dict(d)
             for p, d in payload["plates"].items()}
    if not locks:
        raise ValueError(f"{path}: locks JSON contains no plates")
    return locks, payload.get("meta", {})


@dataclasses.dataclass
class Source:
    """One recording, on the markers CSV's timeline (the ring appends every
    frame — design §3 — so it is the master; poses/mocap join by ``frame``)."""

    name: str
    kind: str                       # "probe" | "campaign"
    q_index: int | None
    t: np.ndarray                   # (n,) seconds, file-local origin
    frame_no: np.ndarray            # (n,) int
    markers: tuple                  # per frame: dict | None
    flags: tuple                    # per frame: dict | None
    streamed: np.ndarray            # (n, 7, 4, 4)
    streamed_valid: np.ndarray      # (n, 7) bool
    q_shipped: np.ndarray | None    # (n, 12), NaN rows where no mocap frame
    chain_ok: np.ndarray | None     # (n,) bool
    rest_mask: np.ndarray           # (n,) bool
    notes: list[str]


def build_source(name: str, kind: str, q_index: int | None, markers_path: str,
                 poses_path: str | None, mocap_path: str | None,
                 rest_seconds: float) -> Source:
    notes: list[str] = []
    t, frame_no, epochs, markers, flags = read_markers_csv(markers_path)
    n = len(t)
    if len(np.unique(epochs)) > 1:
        notes.append(f"mapping epoch changed mid-recording "
                     f"(epochs {sorted(set(int(e) for e in epochs))}) -- the "
                     f"Motive roster moved; split analyses on epoch if numbers "
                     f"look bimodal (design §3)")

    streamed = np.tile(np.eye(4), (n, mc.N_USED_RIGID_BODIES, 1, 1))
    streamed_valid = np.zeros((n, mc.N_USED_RIGID_BODIES), dtype=bool)
    if poses_path is not None and os.path.exists(poses_path):
        pose_map = read_poses_csv(poses_path)
        misses = 0
        for k, fr in enumerate(frame_no):
            hit = pose_map.get(int(fr))
            if hit is None:
                misses += 1
                continue
            streamed[k], streamed_valid[k] = hit
        if misses:
            notes.append(f"{misses}/{n} marker frames had no poses-CSV row -- "
                         f"streamed-side stats skip those frames")
    else:
        notes.append("poses CSV missing -- every streamed-side comparison is "
                     "skipped for this source (nothing else records the "
                     "streamed orientations, review finding int-0)")

    q_shipped = None
    chain_ok = None
    if mocap_path is not None and os.path.exists(mocap_path):
        qmap = read_mocap_csv(mocap_path)
        q_shipped = np.full((n, mc.NUM_JOINTS), np.nan)
        chain_ok = np.zeros(n, dtype=bool)
        for k, fr in enumerate(frame_no):
            hit = qmap.get(int(fr))
            if hit is not None:
                q_shipped[k], chain_ok[k] = hit
    elif kind == "campaign":
        notes.append("mocap CSV missing -- shipped-q comparisons (section 5) "
                     "are skipped for this source")

    if kind == "probe":
        # A probe capture is rest end to end: §6 refuses locks on a moving
        # arm, and its report gates cover the residual doubt.
        rest_mask = np.ones(n, dtype=bool)
    else:
        rest_mask = t <= (t[0] + rest_seconds) if n else np.zeros(0, dtype=bool)
    return Source(name=name, kind=kind, q_index=q_index, t=t,
                  frame_no=frame_no, markers=markers, flags=flags,
                  streamed=streamed, streamed_valid=streamed_valid,
                  q_shipped=q_shipped, chain_ok=chain_ok,
                  rest_mask=rest_mask, notes=notes)


def discover_sources(probe_dir: str | None, campaign_dir: str | None,
                     rest_seconds: float) -> list[Source]:
    """Every usable recording under the given directories, probe first."""
    sources: list[Source] = []
    if probe_dir is not None:
        mpath = os.path.join(probe_dir, "probe_markers.csv")
        if os.path.exists(mpath):
            sources.append(build_source(
                "probe", "probe", None, mpath,
                os.path.join(probe_dir, "probe_poses.csv"), None,
                rest_seconds))
        else:
            print(f"benchmark: no probe_markers.csv under {probe_dir} -- "
                  f"probe source skipped")
    if campaign_dir is not None:
        found = False
        for fname in sorted(os.listdir(campaign_dir)):
            m = _JOINT_MARKERS_RE.match(fname)
            if m is None:
                continue
            found = True
            nn = m.group(1)
            sources.append(build_source(
                f"joint_{nn}", "campaign", int(nn),
                os.path.join(campaign_dir, fname),
                os.path.join(campaign_dir, f"joint_{nn}_poses.csv"),
                os.path.join(campaign_dir, f"joint_{nn}_mocap.csv"),
                rest_seconds))
        if not found:
            print(f"benchmark: no joint_NN_markers.csv under {campaign_dir} -- "
                  f"campaign contributes nothing (marker recording skipped, "
                  f"or wrong directory?)")
    return sources


# --------------------------------------------------------------------------
# Lock validity binding (design §4.1, review finding ops-5) — runs FIRST
# --------------------------------------------------------------------------


def rest_stack_for(src: Source, plate: int) -> np.ndarray:
    """``(n, 4, 3)`` rest-window frames usable as lock-validity evidence.

    Four markers, all tracked when flags are known; taken at face value when
    flags are unknown — which is exactly the assumption the ``?`` regime
    recorded at capture time (design §5).  Non-finite rows are dropped rather
    than fed to the verifier: they are dropout artifacts, not rest geometry.
    """
    rows = []
    for k in np.nonzero(src.rest_mask)[0]:
        m = src.markers[k]
        if m is None or plate not in m:
            continue
        arr = m[plate]
        if arr.shape[0] != 4 or not np.isfinite(arr).all():
            continue
        f = src.flags[k]
        pf = None if f is None else f.get(plate)
        if pf is not None and int(pf.sum()) != 4:
            continue
        rows.append(arr)
    return np.stack(rows) if rows else np.empty((0, 4, 3))


def verify_locks(locks: dict, sources: list[Source]) -> list[str]:
    """Re-derive rest stats per source and let a stale lock refuse.

    Raises :class:`marker_frame.LockRefusal` (the named §4.1 refusal) on the
    first mismatch; returns the list of checks that could **not** be run
    (plate absent / too little rest data), each of which the report states
    instead of silently trusting the lock.
    """
    unverified: list[str] = []
    for src in sources:
        for plate in sorted(locks):
            stack = rest_stack_for(src, plate)
            if stack.shape[0] < MIN_WINDOW_FRAMES:
                unverified.append(
                    f"{src.name}: plate {plate} has {stack.shape[0]} usable "
                    f"rest frames (< {MIN_WINDOW_FRAMES}) -- lock validity "
                    f"NOT re-verified against this source")
                continue
            mf.verify_lock_against_rest(locks[plate], stack)
    return unverified


# --------------------------------------------------------------------------
# The inference pass (one per source; every section reads these arrays)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Analysis:
    """Per-frame inference results for one source."""

    src: Source
    inferred: np.ndarray            # (n, 7, 4, 4), identity where unsolved
    valid: np.ndarray               # (n, 7) bool
    body: np.ndarray                # (n, 7, 3, 3): R_inferred @ Rz(-phi_lock)
    n_usable: np.ndarray            # (n, 7) int, -1 where no quality entry
    rms: np.ndarray                 # (n, 7) m, template registration RMS
    reasons: dict                   # plate -> Counter of gate reasons
    real_drop_frames: dict          # plate -> labeled frames with <4 tracked
    frames_with_sets: int
    frames_labeled: int

    @property
    def regime(self) -> str:
        """The per-joint flag regime (design §5 / review finding ops-9)."""
        if self.frames_with_sets == 0:
            return "no marker sets"
        if self.frames_labeled == 0:
            return "flags unknown (labeled markers absent)"
        if self.frames_labeled < self.frames_with_sets:
            return (f"mixed ({self.frames_labeled}/{self.frames_with_sets} "
                    f"set-frames labeled)")
        return "labeled"


def analyze_source(src: Source, locks: dict) -> Analysis:
    n = len(src.t)
    n_plates = mc.N_USED_RIGID_BODIES
    inferred = np.tile(np.eye(4), (n, n_plates, 1, 1))
    valid = np.zeros((n, n_plates), dtype=bool)
    body = np.tile(np.eye(3), (n, n_plates, 1, 1))
    n_usable = np.full((n, n_plates), -1, dtype=int)
    rms = np.full((n, n_plates), np.nan)
    reasons = {p: collections.Counter() for p in locks}
    real_drop = {p: 0 for p in locks}
    rz_neg = {p: _rz(-lock.phi_rad) for p, lock in locks.items()}

    frames_with_sets = 0
    frames_labeled = 0
    for k in range(n):
        markers, flags = src.markers[k], src.flags[k]
        if markers is not None:
            frames_with_sets += 1
        if flags is not None:
            frames_labeled += 1
        poses, v, quality = mf.infer_all(markers, flags, locks)
        inferred[k] = poses
        valid[k] = v
        for p, q in quality.items():
            n_usable[k, p] = q.n_usable
            rms[k, p] = q.rms_residual_m
            if q.reason:
                reasons[p][q.reason] += 1
            if v[p]:
                body[k, p] = poses[p][0:3, 0:3] @ rz_neg[p]
        if flags is not None:
            for p in locks:
                pf = flags.get(p)
                if pf is not None and len(pf) == 4 and int(pf.sum()) < 4:
                    real_drop[p] += 1
    return Analysis(src=src, inferred=inferred, valid=valid, body=body,
                    n_usable=n_usable, rms=rms, reasons=reasons,
                    real_drop_frames=real_drop,
                    frames_with_sets=frames_with_sets,
                    frames_labeled=frames_labeled)


# --------------------------------------------------------------------------
# Section 1: rectangle validity (design §8.1)
# --------------------------------------------------------------------------


def section_rectangle(analyses: list[Analysis], locks: dict) -> tuple[dict, list[str]]:
    plates: dict[str, dict] = {}
    md = [
        f"Per-frame template-registration RMS over every solved 4-marker "
        f"frame, against the lock's rest geometry.  Context: the calibrated "
        f"volume's wand error is ~{WAND_ERROR_MM:g} mm, so anything at the mm "
        f"scale here is bracket geometry or labeling, not camera noise "
        f"(design §8.1).",
        "",
        "| plate | 4-marker frames | registration RMS (mm)"
        " | lock diagonals (mm) | lock midpoint sep (mm) | lock oop RMS (mm)"
        " | lock skew (deg) | x ref (residual deg) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in sorted(locks):
        lock = locks[p]
        rmss, n4 = [], 0
        for a in analyses:
            sel = a.valid[:, p] & (a.n_usable[:, p] == 4)
            n4 += int(sel.sum())
            rmss.extend((a.rms[sel, p] * 1000.0).tolist())
        rms_s = _dist_stats(rmss)
        plates[str(p)] = {
            "frames_4marker": n4,
            "rms_residual_mm": rms_s,
            "lock": {
                "diag_lengths_mm": [v * 1000.0 for v in lock.diag_lengths_m],
                "midpoint_separation_mm": lock.midpoint_separation_m * 1000.0,
                "out_of_plane_rms_mm": lock.out_of_plane_rms_m * 1000.0,
                "skew_deg": lock.skew_deg,
                "x_residual_deg": lock.x_residual_deg,
                "x_ref_source": lock.x_ref_source,
            },
        }
        md.append(
            f"| {p} | {n4} | {_fmt_stats(rms_s)} | "
            f"{lock.diag_lengths_m[0] * 1000.0:.2f} / "
            f"{lock.diag_lengths_m[1] * 1000.0:.2f} | "
            f"{lock.midpoint_separation_m * 1000.0:.2f} | "
            f"{lock.out_of_plane_rms_m * 1000.0:.3f} | {lock.skew_deg:.2f} | "
            f"{lock.x_ref_source} ({lock.x_residual_deg:.1f}) |")
    return {"plates": plates, "wand_error_mm": WAND_ERROR_MM}, md


# --------------------------------------------------------------------------
# Section 2: static precision (design §8.2)
# --------------------------------------------------------------------------


def _window_deviations(origins: np.ndarray, rots: np.ndarray,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame deviations of one static window from its own mean pose:
    (origin distances [mm], rotation angles [deg]).  Computed per window and
    pooled by the caller — pooling raw poses across windows would count the
    pose *differences between joints' rest states* as jitter."""
    o_dev = np.linalg.norm(origins - origins.mean(axis=0), axis=1) * 1000.0
    r_mean = mean_rotation(rots)
    a_dev = np.array([rot_angle_deg(r_mean.T @ r) for r in rots])
    return o_dev, a_dev


def section_static(analyses: list[Analysis], locks: dict) -> tuple[dict, list[str]]:
    plates: dict[str, dict] = {}
    md = [
        "Jitter over the rest windows (probe: the whole capture; campaign: "
        "the first rest seconds of each joint), deviations taken from each "
        "window's own mean pose and pooled.  Inferred uses the marker-derived "
        "body frame; streamed uses Motive's rigid-body solve.  RMS of the "
        "per-frame deviation = the jitter sd of design §8.2.",
        "",
        "| plate | inferred origin RMS (mm) | streamed origin RMS (mm) | "
        "inferred axis RMS (deg) | streamed axis RMS (deg) | frames inf/str |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(locks):
        inf_o, inf_a, str_o, str_a = [], [], [], []
        n_inf = n_str = 0
        for a in analyses:
            sel = a.src.rest_mask & a.valid[:, p]
            if int(sel.sum()) >= MIN_WINDOW_FRAMES:
                o_dev, a_dev = _window_deviations(
                    a.inferred[sel, p, 0:3, 3], a.body[sel, p])
                inf_o.extend(o_dev.tolist())
                inf_a.extend(a_dev.tolist())
                n_inf += int(sel.sum())
            sel_s = a.src.rest_mask & a.src.streamed_valid[:, p]
            if int(sel_s.sum()) >= MIN_WINDOW_FRAMES:
                o_dev, a_dev = _window_deviations(
                    a.src.streamed[sel_s, p, 0:3, 3],
                    a.src.streamed[sel_s, p, 0:3, 0:3])
                str_o.extend(o_dev.tolist())
                str_a.extend(a_dev.tolist())
                n_str += int(sel_s.sum())

        def _rms(vals):
            return float(np.sqrt(np.mean(np.square(vals)))) if vals else None

        rec = {"inferred": {"frames": n_inf, "origin_rms_mm": _rms(inf_o),
                            "axis_rms_deg": _rms(inf_a)},
               "streamed": {"frames": n_str, "origin_rms_mm": _rms(str_o),
                            "axis_rms_deg": _rms(str_a)}}
        plates[str(p)] = rec

        def _cell(v, nd=3):
            return "no data" if v is None else f"{v:.{nd}f}"

        md.append(f"| {p} | {_cell(rec['inferred']['origin_rms_mm'])} | "
                  f"{_cell(rec['streamed']['origin_rms_mm'])} | "
                  f"{_cell(rec['inferred']['axis_rms_deg'])} | "
                  f"{_cell(rec['streamed']['axis_rms_deg'])} | "
                  f"{n_inf}/{n_str} |")
    return {"plates": plates}, md


# --------------------------------------------------------------------------
# Section 3: manual-alignment error (design §8.3 — the point of the feature)
# --------------------------------------------------------------------------


def section_alignment(analyses: list[Analysis], locks: dict) -> tuple[dict, list[str]]:
    plates: dict[str, dict] = {}
    md = [
        "The constant part of `R_streamed^T @ R_inferred_body` per plate over "
        "every rest frame (chordal mean, projected to SO(3)), and the pivot "
        "offset `|p_streamed - p_inferred|`.  This *is* the manual-alignment "
        "error Motive carries — plus, on a non-square bracket, the lock's "
        "constant x-quantization convention (design §4.1.3), which is "
        "constant by construction and therefore lands here, not in the "
        "jitter sections.",
        "",
        "| plate | frames | constant angle (deg) | axis | scatter p95 (deg) "
        "| pivot offset mean (mm) | pivot offset p95 (mm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in sorted(locks):
        d_mats, offs = [], []
        for a in analyses:
            sel = (a.src.rest_mask & a.valid[:, p]
                   & a.src.streamed_valid[:, p])
            for k in np.nonzero(sel)[0]:
                r_s = a.src.streamed[k, p, 0:3, 0:3]
                d_mats.append(r_s.T @ a.body[k, p])
                offs.append(float(np.linalg.norm(
                    a.src.streamed[k, p, 0:3, 3]
                    - a.inferred[k, p, 0:3, 3])) * 1000.0)
        if not d_mats:
            plates[str(p)] = {"frames": 0}
            md.append(f"| {p} | 0 | no data | -- | -- | -- | -- |")
            continue
        r_const = mean_rotation(d_mats)
        angle = rot_angle_deg(r_const)
        w = _skew_vec(r_const)
        wn = float(np.linalg.norm(w))
        axis = (w / wn).tolist() if wn > 1e-9 else None
        scatter = [rot_angle_deg(r_const.T @ d) for d in d_mats]
        rec = {"frames": len(d_mats),
               "angle_deg": angle,
               "axis": axis,
               "angle_scatter_p95_deg": float(np.percentile(scatter, 95.0)),
               "origin_offset_mm": {"mean": float(np.mean(offs)),
                                    "p95": float(np.percentile(offs, 95.0))}}
        plates[str(p)] = rec
        axis_txt = ("--" if axis is None
                    else "[" + ", ".join(f"{v:+.2f}" for v in axis) + "]")
        md.append(f"| {p} | {rec['frames']} | {angle:.3f} | {axis_txt} | "
                  f"{rec['angle_scatter_p95_deg']:.3f} | "
                  f"{rec['origin_offset_mm']['mean']:.2f} | "
                  f"{rec['origin_offset_mm']['p95']:.2f} |")
    return {"plates": plates}, md


# --------------------------------------------------------------------------
# Section 4: drop-one robustness (design §8.4)
# --------------------------------------------------------------------------


def section_dropone(analyses: list[Analysis], locks: dict,
                    max_frames: int) -> tuple[dict, list[str]]:
    # ---- synthetic drops on real 4-marker frames -------------------------
    synth: dict[str, dict] = {}
    for p in sorted(locks):
        per_marker = {j: {"d_origin_mm": [], "d_axis_deg": [], "gated": 0}
                      for j in range(4)}
        n_frames = 0
        for a in analyses:
            idx = np.nonzero(a.valid[:, p] & (a.n_usable[:, p] == 4))[0]
            if idx.size > max_frames:
                # Strided, never cherry-picked: keeps rest AND drive frames.
                idx = idx[::int(math.ceil(idx.size / max_frames))]
            for k in idx:
                arr = a.src.markers[k][p]
                t4 = a.inferred[k, p]
                n_frames += 1
                for j in range(4):
                    fl = np.ones(4, dtype=np.uint8)
                    fl[j] = 0
                    t3, _q = mf.infer_plate_frame(arr, fl, locks[p])
                    if t3 is None:
                        per_marker[j]["gated"] += 1
                        continue
                    per_marker[j]["d_origin_mm"].append(float(
                        np.linalg.norm(t3[0:3, 3] - t4[0:3, 3])) * 1000.0)
                    per_marker[j]["d_axis_deg"].append(
                        rot_angle_deg(t3[0:3, 0:3].T @ t4[0:3, 0:3]))
        synth[str(p)] = {
            "frames_dropped_from": n_frames,
            "markers": {str(j): {
                "n": len(per_marker[j]["d_origin_mm"]),
                "gated": per_marker[j]["gated"],
                "d_origin_mm": _dist_stats(per_marker[j]["d_origin_mm"]),
                "d_axis_deg": _dist_stats(per_marker[j]["d_axis_deg"]),
            } for j in range(4)},
        }

    # ---- real dropouts, keyed to the per-joint flag regime ---------------
    per_joint: dict[str, dict] = {}
    for a in analyses:
        jplates: dict[str, dict] = {}
        for p in sorted(locks):
            sel3 = a.valid[:, p] & (a.n_usable[:, p] == 3)
            sel4 = a.valid[:, p] & (a.n_usable[:, p] == 4)
            str_ok = a.src.streamed_valid[:, p]

            def _med_off(sel):
                sel = sel & str_ok
                if not sel.any():
                    return None
                offs = np.linalg.norm(
                    a.src.streamed[sel, p, 0:3, 3]
                    - a.inferred[sel, p, 0:3, 3], axis=1) * 1000.0
                return float(np.median(offs))

            rms3 = a.rms[sel3, p] * 1000.0
            jplates[str(p)] = {
                "real_dropout_frames": a.real_drop_frames.get(p, 0),
                "solved_3marker": int(sel3.sum()),
                "solved_4marker": int(sel4.sum()),
                "gated": dict(a.reasons.get(p, {})),
                "origin_vs_streamed_mm": {
                    "3marker_median": _med_off(sel3),
                    "4marker_median": _med_off(sel4)},
                "rms_residual_mm_median": (float(np.median(rms3))
                                           if rms3.size else None),
            }
        per_joint[a.src.name] = {
            "q_index": a.src.q_index,
            "regime": a.regime,
            "frames": len(a.src.t),
            "frames_with_sets": a.frames_with_sets,
            "frames_labeled": a.frames_labeled,
            "plates": jplates,
        }

    md = [
        "**Synthetic drops** on real 4-marker frames (strided to at most "
        f"{max_frames} frames per plate per source): each marker excluded in "
        "turn, the 3-marker solve compared to the 4-marker solve on the same "
        "frame.  Template registration is exact for any 3-subset of a rigid "
        "plate — no parallelogram assumption — so the difference is pure "
        "noise leverage (design §4.2).",
        "",
        "| plate | frames | marker | d-origin (mm) | d-axis (deg) | gated |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(locks):
        rec = synth[str(p)]
        for j in range(4):
            mrec = rec["markers"][str(j)]
            md.append(
                f"| {p} | {rec['frames_dropped_from']} | {j} | "
                f"{_fmt_stats(mrec['d_origin_mm'])} | "
                f"{_fmt_stats(mrec['d_axis_deg'])} | {mrec['gated']} |")
    md += [
        "",
        "**Real dropouts**, keyed to each joint's flag regime (design §5: a "
        "flags-unknown joint runs on 4 assumed markers, so real dropouts are "
        "*invisible* there — the regime is the honest headline, not a zero):",
        "",
    ]
    for name, jrec in per_joint.items():
        md.append(f"- **{name}** — regime: {jrec['regime']} "
                  f"({jrec['frames']} frames, {jrec['frames_with_sets']} with "
                  f"sets, {jrec['frames_labeled']} labeled)")
        for p in sorted(locks):
            prec = jrec["plates"][str(p)]
            if (prec["real_dropout_frames"] == 0 and prec["solved_3marker"] == 0
                    and not prec["gated"]):
                continue
            off3 = prec["origin_vs_streamed_mm"]["3marker_median"]
            off4 = prec["origin_vs_streamed_mm"]["4marker_median"]
            off_txt = ""
            if off3 is not None and off4 is not None:
                off_txt = (f"; origin-vs-streamed median {off3:.2f} mm on "
                           f"3-marker vs {off4:.2f} mm on 4-marker solves")
            rms_txt = ("" if prec["rms_residual_mm_median"] is None else
                       f"; median 3-marker registration RMS "
                       f"{prec['rms_residual_mm_median']:.2f} mm")
            md.append(f"  - plate {p}: {prec['real_dropout_frames']} labeled "
                      f"frames with <4 tracked, {prec['solved_3marker']} "
                      f"3-marker solves, gates {prec['gated'] or '{}'}"
                      f"{off_txt}{rms_txt}")
    return {"synthetic": synth, "per_joint": per_joint}, md


# --------------------------------------------------------------------------
# Section 5: dynamic consistency (design §8.5)
# --------------------------------------------------------------------------


def _spans(origins: np.ndarray) -> np.ndarray:
    """(m, 6, 3) plate origins -> (m, 5) consecutive u-joint spans, metres."""
    return np.linalg.norm(origins[:, 1:, :] - origins[:, :-1, :], axis=2)


def _span_stats(spans: np.ndarray) -> dict:
    if spans.shape[0] < MIN_WINDOW_FRAMES:
        return {"frames": int(spans.shape[0]), "sd_mm": None, "ptp_mm": None}
    return {"frames": int(spans.shape[0]),
            "sd_mm": (spans.std(axis=0, ddof=1) * 1000.0).tolist(),
            # np.ptp the function, not the method: ndarray.ptp was removed in
            # numpy 2.0 and this repo pins numpy>=2.0.
            "ptp_mm": (np.ptp(spans, axis=0) * 1000.0).tolist()}


def section_dynamic(analyses: list[Analysis], locks: dict, out_dir: str,
                    ) -> tuple[dict, list[str], list[str]]:
    """Returns (json, md, plot filenames).  Campaign sources only — a probe
    capture has no drive, which the report states rather than implies."""
    joints: dict[str, dict] = {}
    md: list[str] = []
    plot_files: list[str] = []
    have_all_locks = all(p in locks for p in range(6))
    campaign = [a for a in analyses if a.src.kind == "campaign"]
    if not campaign:
        md.append("No campaign sources given -- a probe static capture has "
                  "no drives, so there is nothing dynamic to compare.  "
                  "Sections 1-4 above still stand.")
        return {"joints": joints}, md, plot_files
    if not have_all_locks:
        md.append(f"Locks cover plates {sorted(locks)} but q needs all of "
                  f"0..5 -- the q comparisons are skipped (span stability is "
                  f"still reported where plates allow).")

    md += [
        "Every q variant is re-zeroed on its own rest-window mean before "
        "differencing (q is repeatable, not absolute -- mounting conventions "
        "set the zero), so the deltas below compare *motion*.  `pivot` = the "
        "shipped swing-only reader fed inferred origins + streamed "
        "orientations; `frames` = `q_from_frames` on the full inferred body "
        "frames (design §4.3).",
        "",
    ]
    lock_phis = (np.array([locks[p].phi_rad for p in range(6)])
                 if have_all_locks else None)

    for a in campaign:
        src = a.src
        name = src.name
        n = len(src.t)
        driven = src.q_index
        rec: dict = {"q_index": driven, "driven_channel": driven, "notes": []}
        joints[name] = rec

        all6 = a.valid[:, 0:6].all(axis=1)
        base_ok = src.streamed_valid[:, 0] & src.streamed_valid[:, 5]
        q_pivot = np.full((n, mc.NUM_JOINTS), np.nan)
        q_frames = np.full((n, mc.NUM_JOINTS), np.nan)
        if have_all_locks:
            for k in np.nonzero(all6 & base_ok)[0]:
                homos = src.streamed[k].copy()
                homos[0:6, 0:3, 3] = a.inferred[k, 0:6, 0:3, 3]
                q = mocap_to_q(homos)
                if q is not None:
                    q_pivot[k] = q
            for k in np.nonzero(all6)[0]:
                q = mf.q_from_frames(a.inferred[k, 0:6], phis=lock_phis)
                if q is not None:
                    q_frames[k] = q

        ship_ok = np.zeros(n, dtype=bool)
        if src.q_shipped is not None:
            ship_ok = np.isfinite(src.q_shipped).all(axis=1)
            if src.chain_ok is not None:
                ship_ok &= src.chain_ok

        def _rezero(qarr, ok, what):
            rest = ok & src.rest_mask
            if not rest.any():
                rec["notes"].append(f"no usable rest frames for {what} -- "
                                    f"its delta trace is skipped")
                return None
            return qarr - qarr[rest].mean(axis=0)

        qhat_ship = (None if src.q_shipped is None
                     else _rezero(src.q_shipped, ship_ok, "shipped q"))
        qhat_pivot = _rezero(q_pivot, np.isfinite(q_pivot).all(axis=1),
                             "pivot-inferred q") if have_all_locks else None
        qhat_frames = _rezero(q_frames, np.isfinite(q_frames).all(axis=1),
                              "frame-inferred q") if have_all_locks else None

        def _dq_stats(qhat_var, label):
            if qhat_ship is None or qhat_var is None:
                return None
            both = (np.isfinite(qhat_var).all(axis=1) & ship_ok)
            if not both.any():
                return None
            dq = qhat_var[both] - qhat_ship[both]
            out = {"frames": int(both.sum()),
                   "all_rms_rad": float(np.sqrt(np.mean(np.square(dq))))}
            if driven is not None:
                out["driven_max_rad"] = float(np.max(np.abs(dq[:, driven])))
                out["driven_max_deg"] = math.degrees(out["driven_max_rad"])
            return out

        rec["dq_pivot"] = _dq_stats(qhat_pivot, "pivot")
        rec["dq_frames"] = _dq_stats(qhat_frames, "frames")

        sel_str = src.streamed_valid[:, 0:6].all(axis=1)
        rec["spans"] = {
            "streamed": _span_stats(_spans(src.streamed[sel_str, 0:6, 0:3, 3])),
            "inferred": _span_stats(_spans(a.inferred[all6, 0:6, 0:3, 3])),
        }
        rec["frames_compared"] = (rec["dq_pivot"] or {}).get("frames", 0)

        def _dq_txt(d):
            if d is None:
                return "not computable"
            drv = ("" if "driven_max_deg" not in d else
                   f"driven max {d['driven_max_deg']:.3f} deg, ")
            return f"{drv}all-channel RMS {d['all_rms_rad']:.5f} rad (n={d['frames']})"

        md.append(f"- **{name}** (drives q{driven}): "
                  f"pivot-vs-shipped {_dq_txt(rec['dq_pivot'])}; "
                  f"frames-vs-shipped {_dq_txt(rec['dq_frames'])}")
        for side in ("streamed", "inferred"):
            st = rec["spans"][side]
            if st["sd_mm"] is None:
                md.append(f"  - spans ({side}): no data")
            else:
                md.append(f"  - spans ({side}): sd "
                          + " ".join(f"{v:.2f}" for v in st["sd_mm"])
                          + " mm; ptp "
                          + " ".join(f"{v:.2f}" for v in st["ptp_mm"])
                          + f" mm (n={st['frames']})")
        for note in rec["notes"] + src.notes:
            md.append(f"  - note: {note}")

        fname = _plot_dynamic(a, qhat_ship, qhat_pivot, qhat_frames, ship_ok,
                              all6, sel_str, out_dir)
        if fname is not None:
            plot_files.append(fname)
            md.append(f"  - ![dynamic]({fname})")
    return {"joints": joints}, md, plot_files


# --------------------------------------------------------------------------
# Section 6: model verification (design §8.6, decision D7 — pass/fail)
# --------------------------------------------------------------------------


def section_model(analyses: list[Analysis], locks: dict) -> tuple[dict, list[str]]:
    corigid: list[dict] = []
    axis_checks: list[dict] = []
    failures: list[str] = []

    # ---- co-rigid pairs: constant relative body rotation ------------------
    for a in analyses:
        for pa, pb in CORIGID_PAIRS:
            if pa not in locks or pb not in locks:
                continue        # plate 6 absent (design D6): skip-and-report
            entry = {"source": a.src.name, "pair": [pa, pb], "pass": None,
                     "note": ""}
            corigid.append(entry)
            sel = a.valid[:, pa] & a.valid[:, pb]
            entry["frames"] = int(sel.sum())
            if entry["frames"] < MIN_PAIR_FRAMES:
                entry["note"] = (f"only {entry['frames']} co-usable frames "
                                 f"(< {MIN_PAIR_FRAMES}) -- not evaluated")
                continue
            rel = np.einsum("kij,kil->kjl", a.body[sel, pa], a.body[sel, pb])
            r_mean = mean_rotation(rel)
            devs = np.array([rot_angle_deg(r_mean.T @ r) for r in rel])
            entry["constant_angle_deg"] = rot_angle_deg(r_mean)
            entry["dev_p95_deg"] = float(np.percentile(devs, 95.0))
            entry["dev_max_deg"] = float(devs.max())
            entry["pass"] = bool(entry["dev_p95_deg"] <= CORIGID_P95_MAX_DEG)
            if not entry["pass"]:
                failures.append(
                    f"co-rigid pair {{{pa},{pb}}} on {a.src.name}: relative "
                    f"body rotation wanders (p95 {entry['dev_p95_deg']:.2f} "
                    f"deg > {CORIGID_P95_MAX_DEG:g} deg) -- the mounting-side/"
                    f"family model of design §2 is violated (review geo-2)")

    # ---- per-drive rotation axis vs the predicted revolute axis -----------
    for a in analyses:
        if a.src.kind != "campaign" or a.src.q_index is None:
            continue
        j = a.src.q_index
        seg, k4 = divmod(j, 4)
        pa, pb = 2 * seg, 2 * seg + 1
        entry = {"source": a.src.name, "q_index": j, "plate_pair": [pa, pb],
                 "predicted_axis": PREDICTED_AXES[k4].tolist(), "pass": None,
                 "note": ""}
        axis_checks.append(entry)
        if pa not in locks or pb not in locks:
            entry["note"] = f"plates {pa}/{pb} not locked -- not evaluated"
            continue
        sel = a.valid[:, pa] & a.valid[:, pb]
        rest = sel & a.src.rest_mask
        if int(rest.sum()) < 1 or int(sel.sum()) < MIN_PAIR_FRAMES:
            entry["note"] = "too few co-usable/rest frames -- not evaluated"
            continue
        rel_rest = np.einsum("kij,kil->kjl", a.body[rest, pa], a.body[rest, pb])
        a_rest = mean_rotation(rel_rest)
        rel = np.einsum("kij,kil->kjl", a.body[sel, pa], a.body[sel, pb])
        delta = np.einsum("ij,kjl->kil", a_rest.T, rel)
        angles = np.array([rot_angle_deg(d) for d in delta])
        peak = float(angles.max())
        entry["peak_excursion_deg"] = peak
        if peak < AXIS_MIN_EXCURSION_DEG:
            entry["note"] = (f"peak excursion {peak:.2f} deg < "
                             f"{AXIS_MIN_EXCURSION_DEG:g} deg -- axis would "
                             f"be noise, not evaluated")
            continue
        # Accumulate the skew parts over the well-excited frames: each is
        # 2 sin(angle) * axis, so the sum is a sin-weighted axis average that
        # ignores the noise-dominated near-rest frames.
        use = angles >= max(AXIS_MIN_EXCURSION_DEG, 0.5 * peak)
        w = np.sum([_skew_vec(d) for d in delta[use]], axis=0)
        wn = float(np.linalg.norm(w))
        if not (wn > 1e-12):
            entry["note"] = "degenerate axis accumulation -- not evaluated"
            continue
        axis = w / wn
        entry["measured_axis"] = axis.tolist()
        # The naive compare against the driven joint's own axis: REPORTED,
        # not gated.  The measured pair rotation spans BOTH u-joints of the
        # segment, and gravity really moves the co-measured angles during a
        # single-agonist drive, so this number contains real physics, not
        # only model error (see AXIS_TOL_DEG's comment).
        dot = abs(float(axis @ PREDICTED_AXES[k4]))
        entry["single_axis_error_deg"] = math.degrees(math.acos(min(1.0, dot)))

        # Cross-talk-aware gate: reconstruct each frame's pair rotation from
        # the §4.3 decomposition — t1/t2 from the plate ORIGINS (positions),
        # t3/t4 from the orientation residual through the 45-deg family —
        # then push the reconstruction through the IDENTICAL delta/skew
        # pipeline and compare axis directions.  Positions predicting
        # orientations through the model's axes is the falsifiable content;
        # a wrong family/mounting/azimuth relationship cannot reconstruct.
        o_p = a.inferred[sel, pa][:, 0:3, 3]
        o_d = a.inferred[sel, pb][:, 0:3, 3]
        r_p = a.body[sel, pa]
        v = np.einsum("kji,kj->ki", r_p, o_p - o_d)   # R_P^T (o_P - o_D)
        model = np.empty_like(rel)
        bad = False
        for i in range(rel.shape[0]):
            nv = float(np.linalg.norm(v[i]))
            if not (nv > 1e-12):
                bad = True
                break
            t1, t2 = ujoint_angles(v[i] / nv)
            swing = _rx(t1) @ _ry(t2)
            m45 = mf.RZ45.T @ (swing.T @ rel[i]) @ mf.RZ45
            t3 = math.atan2(m45[2, 1], m45[1, 1])
            t4 = math.atan2(m45[0, 2], m45[0, 0])
            model[i] = swing @ mf.RZ45 @ _rx(t3) @ _ry(t4) @ mf.RZ45.T
        if bad:
            entry["note"] = "collapsed pair origins -- not evaluated"
            continue
        model_rest_sel = a.src.rest_mask[sel]
        if int(model_rest_sel.sum()) < 1:
            entry["note"] = "no rest frames among co-usable -- not evaluated"
            continue
        m_rest = mean_rotation(model[model_rest_sel])
        delta_pred = np.einsum("ij,kjl->kil", m_rest.T, model)
        w_pred = np.sum([_skew_vec(d) for d in delta_pred[use]], axis=0)
        wpn = float(np.linalg.norm(w_pred))
        if not (wpn > 1e-12):
            entry["note"] = "degenerate model reconstruction -- not evaluated"
            continue
        axis_pred = w_pred / wpn
        entry["model_axis"] = axis_pred.tolist()
        dot = abs(float(axis @ axis_pred))
        entry["axis_error_deg"] = math.degrees(math.acos(min(1.0, dot)))
        entry["pass"] = bool(entry["axis_error_deg"] <= AXIS_TOL_DEG)
        if not entry["pass"]:
            failures.append(
                f"drive q{j} ({a.src.name}): measured rotation axis of plate "
                f"{pb} relative to plate {pa} is {entry['axis_error_deg']:.1f} "
                f"deg off the model reconstruction from the four extracted "
                f"pair angles (tol {AXIS_TOL_DEG:g} deg; naive single-axis "
                f"error {entry['single_axis_error_deg']:.1f} deg) -- the §2 "
                f"axis/family model is violated for this joint")

    evaluated = [e for e in corigid + axis_checks if e["pass"] is not None]
    if not evaluated:
        verdict = "NOT EVALUATED"
    elif failures:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    md = [
        f"**Verdict: {verdict}.**  The §2 hardware model (mounting sides, "
        f"bracket families, revolute-axis layout) is *assumed* by every "
        f"number above; this section is what earns that assumption "
        f"(design D7).",
        "",
        f"Co-rigid pairs (constant relative body rotation, p95 <= "
        f"{CORIGID_P95_MAX_DEG:g} deg):",
        "",
        "| source | pair | frames | constant angle (deg) | dev p95 (deg) | "
        "dev max (deg) | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in corigid:
        if e["pass"] is None:
            md.append(f"| {e['source']} | {{{e['pair'][0]},{e['pair'][1]}}} | "
                      f"{e.get('frames', 0)} | -- | -- | -- | {e['note']} |")
        else:
            md.append(f"| {e['source']} | {{{e['pair'][0]},{e['pair'][1]}}} | "
                      f"{e['frames']} | {e['constant_angle_deg']:.3f} | "
                      f"{e['dev_p95_deg']:.3f} | {e['dev_max_deg']:.3f} | "
                      f"{'PASS' if e['pass'] else 'FAIL'} |")
    md += [
        "",
        f"Per-drive rotation axis, cross-talk-aware (gate: measured pair "
        f"delta vs the model reconstruction from the four extracted pair "
        f"angles, tol {AXIS_TOL_DEG:g} deg; the naive compare against the "
        f"driven axis alone is informational — gravity moves the pair's "
        f"other angles too, and that is physics, not model error):",
        "",
        "| drive | plates | measured | vs model recon (deg, gated) | "
        "vs driven axis alone (deg, info) | peak (deg) | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in axis_checks:
        if e["pass"] is None:
            md.append(f"| q{e['q_index']} ({e['source']}) | "
                      f"{e['plate_pair']} | -- | -- | "
                      f"{e.get('single_axis_error_deg', float('nan')):.2f} | "
                      f"{e.get('peak_excursion_deg', float('nan')):.2f} | "
                      f"{e['note']} |")
        else:
            meas = "[" + ", ".join(f"{v:+.2f}" for v in e["measured_axis"]) + "]"
            md.append(f"| q{e['q_index']} ({e['source']}) | {e['plate_pair']} "
                      f"| {meas} | {e['axis_error_deg']:.2f} | "
                      f"{e['single_axis_error_deg']:.2f} | "
                      f"{e['peak_excursion_deg']:.2f} | "
                      f"{'PASS' if e['pass'] else 'FAIL'} |")
    if failures:
        md += ["", "**MODEL FALSIFIED — the model, not this report, is wrong:**",
               ""]
        md += [f"- {f}" for f in failures]
        md += ["",
               "Every inferred-frame number above inherits this error.  Fix "
               "the §2 model (mounting side, family angle, or joint map) and "
               "re-run; do not act on sections 1-5 until this section passes."]
    return {"verdict": verdict, "corigid": corigid, "axis": axis_checks,
            "failures": failures}, md


# --------------------------------------------------------------------------
# Plots (Agg; one role = one color, thresholds drawn and labeled)
# --------------------------------------------------------------------------


def _grouped_bars(ax, plates: list[int], series: list[tuple[str, list, str]],
                  ylabel: str, title: str) -> None:
    x = np.arange(len(plates), dtype=float)
    width = 0.8 / max(1, len(series))
    for i, (label, vals, color) in enumerate(series):
        vals = [v if v is not None else np.nan for v in vals]
        ax.bar(x + (i - (len(series) - 1) / 2.0) * width, vals, width * 0.9,
               label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in plates])
    ax.set_xlabel("plate")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def plot_static(sec: dict, out_dir: str) -> str | None:
    plates = sorted(int(p) for p in sec["plates"])
    if not plates:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=110)
    for ax, key, ylabel in ((axes[0], "origin_rms_mm", "origin jitter RMS (mm)"),
                            (axes[1], "axis_rms_deg", "axis jitter RMS (deg)")):
        _grouped_bars(
            ax, plates,
            [("inferred", [sec["plates"][str(p)]["inferred"][key]
                           for p in plates], COLOR_INFERRED),
             ("streamed", [sec["plates"][str(p)]["streamed"][key]
                           for p in plates], COLOR_STREAMED)],
            ylabel, "static precision (rest windows)")
    fig.tight_layout()
    name = "static_precision.png"
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    return name


def plot_alignment(sec: dict, out_dir: str) -> str | None:
    plates = [int(p) for p in sorted(sec["plates"])
              if sec["plates"][p].get("frames", 0) > 0]
    if not plates:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=110)
    _grouped_bars(axes[0], plates,
                  [("constant angle", [sec["plates"][str(p)]["angle_deg"]
                                       for p in plates], COLOR_INFERRED)],
                  "angle (deg)", "streamed vs inferred body frame: constant part")
    _grouped_bars(axes[1], plates,
                  [("pivot offset mean",
                    [sec["plates"][str(p)]["origin_offset_mm"]["mean"]
                     for p in plates], COLOR_INFERRED)],
                  "offset (mm)", "streamed pivot vs inferred origin at rest")
    fig.tight_layout()
    name = "alignment_error.png"
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    return name


def plot_dropone(sec: dict, out_dir: str) -> str | None:
    plates = sorted(int(p) for p in sec["synthetic"])
    if not plates:
        return None
    have = any(sec["synthetic"][str(p)]["markers"][str(j)]["d_origin_mm"]
               for p in plates for j in range(4))
    if not have:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=110)
    x = np.arange(len(plates), dtype=float)
    width = 0.8 / 4.0
    # Markers are positions on one bracket, not entities: a single-hue set of
    # four bars per plate, distinguished by lightness + the marker index on
    # the axis, keeps identity without inventing four fake series colors.
    shades = ["#0072B2", "#4D94C6", "#88B7D9", "#C2DAEC"]
    for ax, key, ylabel in ((axes[0], "d_origin_mm", "median d-origin (mm)"),
                            (axes[1], "d_axis_deg", "median d-axis (deg)")):
        for j in range(4):
            vals = []
            for p in plates:
                st = sec["synthetic"][str(p)]["markers"][str(j)][key]
                vals.append(st["median"] if st else np.nan)
            ax.bar(x + (j - 1.5) * width, vals, width * 0.9,
                   label=f"marker {j}", color=shades[j])
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in plates])
        ax.set_xlabel("plate")
        ax.set_ylabel(ylabel)
        ax.set_title("drop-one vs 4-marker solve", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    name = "drop_one.png"
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    return name


def _plot_dynamic(a: Analysis, qhat_ship, qhat_pivot, qhat_frames, ship_ok,
                  all6, sel_str, out_dir: str) -> str | None:
    src = a.src
    if src.q_index is None or qhat_ship is None or len(src.t) < MIN_WINDOW_FRAMES:
        return None
    drv = src.q_index
    t = src.t - src.t[0]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), dpi=110, sharex=True)

    ax = axes[0]
    ax.plot(t[ship_ok], np.degrees(qhat_ship[ship_ok, drv]), "-",
            color=COLOR_STREAMED, lw=2, label="shipped (streamed pivots)")
    if qhat_pivot is not None:
        ok = np.isfinite(qhat_pivot[:, drv])
        ax.plot(t[ok], np.degrees(qhat_pivot[ok, drv]), "--",
                color=COLOR_INFERRED, lw=2, label="pivot (inferred origins)")
    if qhat_frames is not None:
        ok = np.isfinite(qhat_frames[:, drv])
        ax.plot(t[ok], np.degrees(qhat_frames[ok, drv]), ":",
                color=COLOR_FRAMES, lw=2, label="frames (q_from_frames)")
    ax.set_ylabel(f"q{drv} - rest (deg)")
    ax.set_title(f"{src.name}: driven channel, all three q variants",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for qhat, label, color in ((qhat_pivot, "pivot - shipped", COLOR_INFERRED),
                               (qhat_frames, "frames - shipped", COLOR_FRAMES)):
        if qhat is None:
            continue
        both = np.isfinite(qhat).all(axis=1) & ship_ok
        if both.any():
            dq = np.degrees(qhat[both, drv] - qhat_ship[both, drv])
            ax.plot(t[both], dq, "-", color=color, lw=2, label=label)
    ax.set_ylabel(f"dq{drv} (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    if sel_str.any():
        sp = _spans(src.streamed[sel_str, 0:6, 0:3, 3]) * 1000.0
        for i in range(5):
            ax.plot(t[sel_str], sp[:, i] - sp[0, i], "-", color=COLOR_STREAMED,
                    lw=1, alpha=0.7, label="streamed" if i == 0 else None)
    if all6.any():
        sp = _spans(a.inferred[all6, 0:6, 0:3, 3]) * 1000.0
        for i in range(5):
            ax.plot(t[all6], sp[:, i] - sp[0, i], "--", color=COLOR_INFERRED,
                    lw=1, alpha=0.7, label="inferred" if i == 0 else None)
    ax.set_ylabel("span change (mm)")
    ax.set_xlabel("t (s)")
    ax.set_title("chain-span stability (each span minus its first value)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    name = f"dynamic_{src.name}.png"
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    return name


def plot_model(sec: dict, out_dir: str) -> str | None:
    cor = [e for e in sec["corigid"] if e["pass"] is not None]
    axs = [e for e in sec["axis"] if e["pass"] is not None]
    if not cor and not axs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)

    ax = axes[0]
    if cor:
        labels = [f"{e['source']}\n{{{e['pair'][0]},{e['pair'][1]}}}"
                  for e in cor]
        vals = [e["dev_p95_deg"] for e in cor]
        ax.bar(np.arange(len(cor)), vals, 0.6, color=COLOR_INFERRED)
        ax.axhline(CORIGID_P95_MAX_DEG, color=COLOR_GATE, ls="--", lw=1.5)
        ax.text(0.02, CORIGID_P95_MAX_DEG, f" gate {CORIGID_P95_MAX_DEG:g} deg",
                color=COLOR_GATE, va="bottom", fontsize=8,
                transform=ax.get_yaxis_transform())
        ax.set_xticks(np.arange(len(cor)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("relative-rotation dev p95 (deg)")
        ax.set_ylim(0.0, max([CORIGID_P95_MAX_DEG * 1.5] + [v * 1.2 for v in vals]))
    ax.set_title("co-rigid constancy", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    if axs:
        labels = [f"q{e['q_index']}" for e in axs]
        vals = [e["axis_error_deg"] for e in axs]
        ax.bar(np.arange(len(axs)), vals, 0.6, color=COLOR_INFERRED)
        ax.axhline(AXIS_TOL_DEG, color=COLOR_GATE, ls="--", lw=1.5)
        ax.text(0.02, AXIS_TOL_DEG, f" gate {AXIS_TOL_DEG:g} deg",
                color=COLOR_GATE, va="bottom", fontsize=8,
                transform=ax.get_yaxis_transform())
        ax.set_xticks(np.arange(len(axs)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("axis error vs predicted (deg)")
        ax.set_ylim(0.0, max([AXIS_TOL_DEG * 1.5] + [v * 1.2 for v in vals]))
    ax.set_title("per-drive rotation axis", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    name = "model_verification.png"
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    return name


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def run_benchmark(locks: dict, locks_meta: dict, sources: list[Source],
                  out_dir: str, rest_seconds: float, drop_one_max: int,
                  unverified: list[str]) -> int:
    os.makedirs(out_dir, exist_ok=True)
    analyses = [analyze_source(src, locks) for src in sources]

    rect_j, rect_md = section_rectangle(analyses, locks)
    stat_j, stat_md = section_static(analyses, locks)
    align_j, align_md = section_alignment(analyses, locks)
    drop_j, drop_md = section_dropone(analyses, locks, drop_one_max)
    dyn_j, dyn_md, dyn_plots = section_dynamic(analyses, locks, out_dir)
    model_j, model_md = section_model(analyses, locks)

    plots = [p for p in (plot_static(stat_j, out_dir),
                         plot_alignment(align_j, out_dir),
                         plot_dropone(drop_j, out_dir))
             if p is not None]
    plots += dyn_plots
    p = plot_model(model_j, out_dir)
    if p is not None:
        plots.append(p)

    meta = {
        "generated_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "locks_meta": {k: locks_meta.get(k) for k in
                       ("captured_wall", "regime", "mapping_epoch",
                        "fps_measured") if k in locks_meta},
        "locked_plates": sorted(locks),
        "rest_seconds": rest_seconds,
        "thresholds": {"corigid_p95_max_deg": CORIGID_P95_MAX_DEG,
                       "axis_tol_deg": AXIS_TOL_DEG,
                       "axis_min_excursion_deg": AXIS_MIN_EXCURSION_DEG,
                       "wand_error_mm": WAND_ERROR_MM},
        "lock_validity_unverified": unverified,
        "sources": [{"name": a.src.name, "kind": a.src.kind,
                     "q_index": a.src.q_index, "frames": len(a.src.t),
                     "frames_with_sets": a.frames_with_sets,
                     "frames_labeled": a.frames_labeled,
                     "regime": a.regime,
                     "rest_frames": int(a.src.rest_mask.sum()),
                     "duration_s": (float(a.src.t[-1] - a.src.t[0])
                                    if len(a.src.t) else 0.0),
                     "notes": a.src.notes} for a in analyses],
    }
    exit_code = (EXIT_MODEL_FALSIFIED if model_j["verdict"] == "FAIL"
                 else EXIT_OK)
    report = {
        "meta": meta,
        "sections": {
            "rectangle_validity": rect_j,
            "static_precision": stat_j,
            "alignment_error": align_j,
            "drop_one": drop_j,
            "dynamic_consistency": dyn_j,
            "model_verification": model_j,
        },
        "plots": plots,
        "exit_code": exit_code,
    }

    lines = [
        "# Marker-frame benchmark (marker_frame_design.md sec. 8)",
        "",
        f"Generated {meta['generated_wall']}; locks over plates "
        f"{meta['locked_plates']}"
        + (f", minted {locks_meta['captured_wall']}"
           if "captured_wall" in locks_meta else "")
        + f"; model verification: **{model_j['verdict']}**.",
        "",
        "Sources (flag regime per joint, review finding ops-9):",
        "",
    ]
    for s in meta["sources"]:
        lines.append(f"- {s['name']} ({s['kind']}"
                     + (f", drives q{s['q_index']}" if s["q_index"] is not None
                        else "")
                     + f"): {s['frames']} frames over {s['duration_s']:.1f} s, "
                     f"{s['rest_frames']} rest; regime: {s['regime']}")
    if unverified:
        lines += ["", "Lock validity checks that could NOT be run "
                      "(stated, not silently trusted -- review ops-5):", ""]
        lines += [f"- {u}" for u in unverified]
    for title, md, plot in (
            ("## 1. Rectangle validity", rect_md, None),
            ("## 2. Static precision", stat_md, "static_precision.png"),
            ("## 3. Manual-alignment error", align_md, "alignment_error.png"),
            ("## 4. Drop-one robustness", drop_md, "drop_one.png"),
            ("## 5. Dynamic consistency", dyn_md, None),
            (f"## 6. Model verification -- {model_j['verdict']}", model_md,
             "model_verification.png")):
        lines += ["", title, ""]
        lines += md
        if plot is not None and plot in plots:
            lines += ["", f"![{plot}]({plot})"]

    md_path = os.path.join(out_dir, MD_NAME)
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    json_path = os.path.join(out_dir, JSON_NAME)
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
        fh.write("\n")

    print(f"benchmark: wrote {md_path}, {json_path}"
          + (f", {len(plots)} plot(s)" if plots else ""))
    if model_j["verdict"] == "FAIL":
        print("== MODEL VERIFICATION FAILED (exit 3) ==")
        for f in model_j["failures"]:
            print(f"  {f}")
        print("  The design-sec-2 hardware model, not this report, is wrong; "
              "sections 1-5 inherit the error.")
    else:
        print(f"benchmark: model verification {model_j['verdict']}")
    return exit_code


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Offline marker-frame benchmark "
                    "(marker_frame_design.md sec. 8)")
    ap.add_argument("--locks", required=True,
                    help="locks.json from the probe (--lock-out)")
    ap.add_argument("--probe", default=None,
                    help="probe static-capture directory "
                         "(probe_markers.csv / probe_poses.csv)")
    ap.add_argument("--campaign", default=None,
                    help="campaign directory (joint_NN_markers/poses/mocap "
                         "CSVs)")
    ap.add_argument("--out-dir", default=None,
                    help="report directory (default: "
                         "<campaign>/marker_frame_benchmark, falling back to "
                         "<probe>/marker_frame_benchmark)")
    ap.add_argument("--rest-seconds", type=float, default=REST_SECONDS_DEFAULT,
                    help="rest window at the head of each campaign joint "
                         f"recording (default {REST_SECONDS_DEFAULT:g} = the "
                         "campaign's JOINT_REST_CAPTURE_S)")
    ap.add_argument("--drop-one-max-frames", type=int,
                    default=DROP_ONE_MAX_FRAMES,
                    help="cap on synthetically-dropped frames per plate per "
                         f"source (default {DROP_ONE_MAX_FRAMES})")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.probe is None and args.campaign is None:
        print("benchmark: nothing to consume -- give --probe and/or "
              "--campaign (design sec. 8)")
        return EXIT_USAGE
    try:
        locks, locks_meta = load_locks(args.locks)
    except (OSError, ValueError, KeyError) as exc:
        print(f"benchmark: cannot load locks from {args.locks}: {exc}")
        return EXIT_USAGE

    sources = discover_sources(args.probe, args.campaign, args.rest_seconds)
    sources = [s for s in sources if len(s.t)]
    if not sources:
        print("benchmark: no usable marker recordings found -- nothing to do")
        return EXIT_USAGE

    # Lock validity binding FIRST (design §4.1 / review ops-5): a stale lock
    # refuses the whole run before any number is computed from it.
    try:
        unverified = verify_locks(locks, sources)
    except mf.LockRefusal as exc:
        print("== LOCK VALIDITY REFUSED (exit 2) ==")
        print(f"  {exc}")
        print("  Locks are valid only for the Motive session they were "
              "captured in; regenerate locks.json with the probe "
              "(mocap_probe.py --lock-out) and re-run.")
        return EXIT_STALE_LOCK

    out_dir = args.out_dir
    if out_dir is None:
        root = args.campaign if args.campaign is not None else args.probe
        out_dir = os.path.join(root, "marker_frame_benchmark")
    return run_benchmark(locks, locks_meta, sources, out_dir,
                         args.rest_seconds, args.drop_one_max_frames,
                         unverified)


if __name__ == "__main__":
    sys.exit(main())
