r"""Roll a real recording through the twin, open loop, and score the difference.

The twin's report card, and its being open loop is the whole point: a
closed-loop comparison lets the twin's own controller correct the model's error
and report an agreement it did not earn.  Open loop, the same commands go into
the metal and into the model and the trajectories are allowed to diverge, which
is the only arrangement in which the divergence means something.  The recording
carries the exact wire targets per cycle on a strictly increasing sync clock, so
what differs between the recorded ``q`` and the rolled-out ``q`` is the plant
model and nothing else.

ALIGNMENT IS ANCHORED ON ``can_sync_time_s`` AND ON NOTHING ELSE, and it is the
part of a port most likely to be got wrong.  Two rules carry the correctness:

1. **Both traces are re-referenced over the SAME rows** -- the intersection of
   the settled reference window with the rows that actually have a rollout
   sample and a usable pose.  Averaging the two over different sets bakes their
   difference into every later number, and the resulting bias is a constant, so
   it survives every summary statistic and looks like a modelling error.
2. **The real reference is taken in the recording's own coordinates.**  The
   RS485 version had to say "raw mocap, never ``q_raw - q_zero``", because there
   a mid-run re-zero STEPS the origin and a constant reference cannot cancel a
   step.  This schema has no per-row zero column at all -- ``robot_state.q`` is
   already the marker-derived angle in radians -- so the hazard is absent here
   by construction rather than avoided by care.  If a collector ever adds a
   re-zero column, this is the paragraph that has to change.

WHAT THIS ARM ADDS THAT THE RS485 ARM DID NOT NEED: a population split.  There
is none to port -- the RS485 file splits only temporally and per joint -- but
twenty-four boards of two variants with different valve physics and different
failsafes, scored together, yield one number that describes neither plant.  So
every metric here is reported three ways: per joint or per board, over the TLE
population, and over the 7 mm population.  **The population is read from the
variant byte the board answered, never from its id range.**  A joint whose two
antagonists come from different populations is reported as ``mixed`` rather than
assigned to one, because assigning it would put a TLE board's error into the
7 mm column.

Ported from ``C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\
UMArm_SIM\twin_compare.py`` via ``reference/mjcf_fit.md`` section 4.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import replay as R
from . import ring_analysis as RA

#: The settled window both traces are referenced over, s.  Half a second is
#: about 75 cycles at the 150 Hz sync rate -- long enough that the plate jitter
#: averages down by roughly an order of magnitude, short enough that the arm has
#: not drifted within it.
REF_WINDOW_S = 0.5

#: How many rows the fallback "last posed window" reference may use.  Eighty at
#: 150 Hz is 0.53 s, deliberately the same span as REF_WINDOW_S so the two
#: reference paths average over comparable amounts of jitter.
REF_ROWS_MAX = 80

#: A commanded target at or below this counts as idle for the purpose of finding
#: a settled pre-drive window, Pa.  Twice the 0.5 psi idle setpoint every
#: campaign on this arm parks unused boards at -- a board commanded to exactly
#: zero holds its exhaust valve open continuously, so 0.5 psi is the real floor.
#: The factor of two is headroom against the ADC round trip: a target written as
#: counts and read back in Pa lands a fraction of a count either side of the
#: setpoint, and a threshold sitting exactly on it classifies a genuinely idle
#: board as driving about half the time.
IDLE_TARGET_PA = 1.0 * R.PA_PER_PSI

#: A scoring window with fewer usable samples than this returns None rather than
#: a number.  Twenty samples is 0.13 s at the sync rate: below that an RMS is
#: dominated by whichever part of a single ring cycle the window happened to
#: catch.
MIN_SAMPLES = 20

#: Bumped whenever the rollout or the alignment changes MEANING, so a cached
#: twin from before the change cannot answer for after it.  The RS485 file's
#: comment on its own version 3 is the warning worth carrying: a version-2 cache
#: rolled under overdamped constants scored the twin four times too slow, i.e.
#: the exact opposite answer, and nothing about the cache file said so.
ALGO_VERSION = 1


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------


def _measured_joint_pairs():
    """The measured actuator/axis map, or None if this workspace is not on the path.

    Guarded because ``twin_compare`` must stay importable from inside the
    package alone -- a test that cannot import the map still tests the
    alignment, which is the part that carries the correctness.
    """
    try:
        from UMArm_KINEMATICS import canarm_actuators as CA  # noqa: WPS433
    except Exception:
        return None
    return CA.MEASURED_JOINT_PAIRS if CA.MEASURED else None


def joint_population(rec: R.Recording, *, pairs=None) -> np.ndarray:
    """``(12,)`` of ``"tle"`` / ``"7mm"`` / ``"mixed"``, from the VARIANT BYTE.

    Derived through the measured antagonistic pairs rather than assumed from the
    segment blocks, because the two answers can disagree: the id blocks put
    ``0x101``-``0x108`` on segment 1, but a board's population is whatever it
    reported, and one TLE board sat at ``0x114`` during the 2026-08 bench
    session.  Where the map is unavailable the segment fallback is used and said
    so in the result, rather than silently producing a plausible split.
    """
    pairs = pairs if pairs is not None else _measured_joint_pairs()
    by_id = {int(b): bool(t) for b, t in zip(rec.ids, rec.is_tle)}
    out = np.empty(R.N_JOINTS, dtype=object)
    if pairs is None:
        # Fallback: four joints per segment, segment 1 being the TLE platform.
        # Stated as a fallback in the metrics dict so a reader knows the split
        # came from geometry rather than from twenty-four answered variant bytes.
        for j in range(R.N_JOINTS):
            block = R.DEFAULT_IDS[(j // 4) * 8]
            out[j] = "tle" if by_id.get(int(block), j < 4) else "7mm"
        return out
    for j, (pos, neg) in enumerate(pairs[:R.N_JOINTS]):
        a, b = by_id.get(int(pos)), by_id.get(int(neg))
        if a is None or b is None:
            out[j] = "unknown"
        elif a == b:
            out[j] = "tle" if a else "7mm"
        else:
            out[j] = "mixed"
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """Which recording row corresponds to which rollout row, and how exactly.

    ``exact`` is True only when every matched pair sits at bitwise the same
    stamp.  That is the normal case and it is worth asserting rather than
    assuming: :func:`digital_twin.replay.rollout` drives on the recording's own
    ``can_sync_time_s`` values, so any drift at all means the rollout was driven
    on a resampled or nominal clock, which is a defect and not a tolerance.
    """

    rec_rows: np.ndarray      # (m,) indices into the recording
    roll_rows: np.ndarray     # (m,) indices into the rollout
    exact: bool
    max_dt_s: float
    n_unmatched: int


def align_on_sync(rec: R.Recording, roll: R.Rollout, *, tol_s: float = 0.0) -> Alignment:
    """Match recording rows to rollout rows by the sync edge itself.

    ``searchsorted`` against the rollout's own stamps plus an explicit equality
    test, rather than trusting ``Rollout.row_index``: the index is what the
    rollout believes, the stamp is what it was driven on, and the point of
    anchoring on ``can_sync_time_s`` is that the two are checked against each
    other rather than one being assumed.  Rows the rollout never reached -- a
    truncated ``t_max_s``, or a tail row that carries no recording row -- are
    dropped here rather than matched to a neighbour.
    """
    t_rec = np.asarray(rec.t_sync_s, dtype=np.float64)
    drive = roll.row_index >= 0        # tail rows are not recording rows
    t_roll = np.asarray(roll.t_s, dtype=np.float64)[drive]
    roll_ix = np.flatnonzero(drive)
    if t_roll.size == 0:
        return Alignment(np.zeros(0, np.int64), np.zeros(0, np.int64), True, 0.0,
                         int(t_rec.size))

    pos = np.searchsorted(t_roll, t_rec)
    lo = np.clip(pos - 1, 0, t_roll.size - 1)
    hi = np.clip(pos, 0, t_roll.size - 1)
    d_lo = np.abs(t_roll[lo] - t_rec)
    d_hi = np.abs(t_roll[hi] - t_rec)
    pick = np.where(d_lo <= d_hi, lo, hi)
    dt = np.minimum(d_lo, d_hi)

    keep = dt <= float(tol_s)
    rec_rows = np.flatnonzero(keep).astype(np.int64)
    roll_rows = roll_ix[pick[keep]].astype(np.int64)
    return Alignment(rec_rows=rec_rows, roll_rows=roll_rows,
                     exact=bool(np.all(dt[keep] == 0.0)),
                     max_dt_s=float(dt[keep].max()) if rec_rows.size else 0.0,
                     n_unmatched=int(np.count_nonzero(~keep)))


def pick_reference_rows(rec: R.Recording, *, window_s: float = REF_WINDOW_S,
                        rows_max: int = REF_ROWS_MAX,
                        idle_target_pa: float = IDLE_TARGET_PA,
                        explicit_window_s=None) -> "tuple[np.ndarray, str]":
    """The settled rows both traces are referenced over, and how they were chosen.

    Preference order, and each entry's reason:

    1. an ``explicit_window_s=(t0, t1)`` the caller measured itself -- an
       operator who knows where the arm was holding beats any heuristic;
    2. the first ``window_s`` of contiguous cycles in which every board is
       commanded at or below the idle setpoint and the pose is usable, i.e. the
       pre-drive hold;
    3. the LAST ``rows_max`` posed rows.  A session that starts driving at cycle
       zero has no pre-drive hold at all, and the end of a recording is a stiller
       reference than its start, where the arm is still pressurising.

    The chosen rows are returned rather than a mean, because the caller has to
    intersect them with the rows the rollout actually reached before averaging
    anything.
    """
    t = rec.t_rel_s
    if explicit_window_s is not None:
        t0, t1 = float(explicit_window_s[0]), float(explicit_window_s[1])
        rows = np.flatnonzero((t >= t0) & (t <= t1) & rec.q_valid)
        return rows, f"explicit window [{t0}, {t1}] s"

    idle = np.all(rec.target_pa <= float(idle_target_pa), axis=1) & rec.q_valid
    if np.any(idle):
        for i0, i1 in _runs(idle):
            if (t[i1 - 1] - t[i0]) >= float(window_s):
                rows = np.arange(i0, i1, dtype=np.int64)
                rows = rows[t[rows] <= t[i0] + float(window_s)]
                return rows, f"pre-drive idle hold at t_rel {float(t[i0])} s"

    posed = np.flatnonzero(rec.q_valid)
    if posed.size == 0:
        return np.zeros(0, dtype=np.int64), "no posed rows"
    return posed[-int(rows_max):], f"last {min(int(rows_max), posed.size)} posed rows"


def _runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = [int(i) + 1 for i in edges if mask[i + 1]]
    ends = [int(i) + 1 for i in edges if not mask[i + 1]]
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(int(mask.size))
    return list(zip(starts, ends))


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwinResult:
    """One open-loop rollout, aligned and re-referenced against its recording.

    Deflections rather than absolute angles, on both sides, because the twin's
    base pose and the mocap volume's origin are two different conventions and
    the difference between them is a constant nobody measured.  Referencing both
    to the same settled window removes exactly that constant and nothing else.
    """

    rec_rows: np.ndarray        # (m,) recording rows scored
    t_rel_s: np.ndarray         # (m,) s, from the recording's first sync edge
    q_sim_rad: np.ndarray       # (m, 12)
    twin_defl_rad: np.ndarray   # (m, 12) sim minus the sim reference
    real_defl_rad: np.ndarray   # (m, 12) real minus the real reference, NaN where unposed
    p_sim_pa: np.ndarray        # (m, 24) twin plant truth
    p_real_pa: np.ndarray       # (m, 24) recorded, rest offset removed
    ref_rows: np.ndarray        # (k,) the rows BOTH references averaged over
    sim_ref_rad: np.ndarray     # (12,)
    real_ref_rad: np.ndarray    # (12,)
    alignment: "Alignment | None"
    ctrl_min_n: float
    valid: bool
    reason: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.t_rel_s.shape[0])


def _invalid(reason: str, meta=None) -> TwinResult:
    z0 = np.zeros(0, dtype=np.float64)
    return TwinResult(np.zeros(0, np.int64), z0, np.zeros((0, R.N_JOINTS)),
                      np.zeros((0, R.N_JOINTS)), np.zeros((0, R.N_JOINTS)),
                      np.zeros((0, R.N_NODES)), np.zeros((0, R.N_NODES)),
                      np.zeros(0, np.int64), np.full(R.N_JOINTS, np.nan),
                      np.full(R.N_JOINTS, np.nan), None, float("nan"), False,
                      reason, dict(meta or {}))


def twin_rollout(rec: R.Recording, *, t_max_s=None, ref_window_s: float = REF_WINDOW_S,
                 explicit_ref_window_s=None, tol_s: float = 0.0,
                 **rollout_kwargs) -> TwinResult:
    """Roll ``rec`` through the twin and re-reference both traces onto one origin.

    A :class:`digital_twin.replay.ClampViolation` returns an explicitly INVALID
    result rather than a scored one.  Measured headroom on the RS485 arm was
    about 500 N against its 4000 N clip, so contact means the force law was
    driven somewhere new -- a finding, and one that must not be averaged into a
    residual as if it were ordinary model error.
    """
    try:
        roll = R.rollout(rec, t_max_s=t_max_s, **rollout_kwargs)
    except R.ClampViolation as exc:
        return _invalid(f"clamp violation: {exc}", {"t_max_s": t_max_s})

    al = align_on_sync(rec, roll, tol_s=tol_s)
    if al.rec_rows.size == 0:
        return _invalid("no recording row matched a rollout sample",
                        {"n_unmatched": al.n_unmatched})

    ref_rows, how = pick_reference_rows(rec, window_s=ref_window_s,
                                        explicit_window_s=explicit_ref_window_s)
    # THE INTERSECTION IS THE WHOLE POINT: both means must come from the same
    # rows.  A reference window that fell in a truncated tail of the rollout has
    # rows on the real side and none on the sim side, and averaging the two over
    # what each happens to have available bakes their difference into every
    # later number.
    matched = np.zeros(rec.n, dtype=bool)
    matched[al.rec_rows] = True
    ref_rows = ref_rows[matched[ref_rows] & rec.q_valid[ref_rows]]
    if ref_rows.size == 0:
        return _invalid(
            f"reference window has no rollout samples (chosen by: {how}); "
            f"re-referencing elsewhere would compare two different origins",
            {"reference_choice": how})

    rec_to_roll = np.full(rec.n, -1, dtype=np.int64)
    rec_to_roll[al.rec_rows] = al.roll_rows

    q_sim = roll.q_rad[rec_to_roll[al.rec_rows]]
    q_real = rec.q_rad[al.rec_rows].copy()
    q_real[~rec.q_valid[al.rec_rows]] = np.nan

    sim_ref = roll.q_rad[rec_to_roll[ref_rows]].mean(axis=0)
    real_ref = rec.q_rad[ref_rows].mean(axis=0)

    return TwinResult(
        rec_rows=al.rec_rows, t_rel_s=rec.t_rel_s[al.rec_rows], q_sim_rad=q_sim,
        twin_defl_rad=q_sim - sim_ref, real_defl_rad=q_real - real_ref,
        p_sim_pa=roll.p_pa[rec_to_roll[al.rec_rows]],
        p_real_pa=rec.p_pa[al.rec_rows], ref_rows=ref_rows, sim_ref_rad=sim_ref,
        real_ref_rad=real_ref, alignment=al, ctrl_min_n=float(roll.ctrl_min_n),
        valid=True, reason="",
        meta={"algo_version": ALGO_VERSION, "reference_choice": how,
              "n_ref_rows": int(ref_rows.size), "alignment_exact": bool(al.exact),
              "alignment_max_dt_s": float(al.max_dt_s),
              "n_unmatched_rows": int(al.n_unmatched),
              "rollout": dict(roll.meta), "t_max_s": t_max_s})


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------


def _score_block(err, real, *, unit_scale: float = 1.0, min_samples: int = MIN_SAMPLES):
    """Per-column RMS, NRMSE and peak error over one window.

    NRMSE is ``RMS(twin - real) / RMS(real - mean(real))``, so **1.0 means the
    twin predicts the arm no better than "it does not move"**.  That is the null
    every RMS number in this file must be read against, and it is the reason a
    raw RMS is never reported on its own here.
    """
    if err.shape[0] < int(min_samples):
        return None
    rms = np.sqrt(np.mean(err ** 2, axis=0)) * unit_scale
    spread = np.sqrt(np.mean((real - real.mean(axis=0)) ** 2, axis=0)) * unit_scale
    return {
        "n": int(err.shape[0]),
        "rms": rms,
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
        "peak": np.max(np.abs(err), axis=0) * unit_scale,
        "spread": spread,
        # 1e-9 rather than 0 so a column that genuinely never moved reports a
        # huge NRMSE instead of an inf that poisons every mean downstream.
        "nrmse": rms / np.maximum(spread, 1e-9),
        "nrmse_mean": float(np.mean(rms / np.maximum(spread, 1e-9))),
    }


def _by_population(block, labels, *, key="rms"):
    """Split a per-column block by population label, means only.

    Means only, and deliberately: a per-population maximum over four joints is
    one joint's number wearing a population's name, which is the confusion this
    split exists to remove.
    """
    if block is None:
        return {}
    out = {}
    labels = np.asarray(labels, dtype=object)
    for pop in sorted({str(v) for v in labels}):
        m = labels == pop
        if not np.any(m):
            continue
        out[pop] = {
            "n_columns": int(np.count_nonzero(m)),
            f"{key}_mean": float(np.mean(block[key][m])),
            f"{key}_max": float(np.max(block[key][m])),
            "nrmse_mean": float(np.mean(block["nrmse"][m])),
        }
    return out


def compare_metrics(rec: R.Recording, tw: TwinResult, *, min_samples: int = MIN_SAMPLES,
                    segments=None, ring_thresh_deg: float = 0.30,
                    ring_band=RA.BAND_HZ) -> dict:
    """Score a :class:`TwinResult`: per joint, per board, and split by population.

    Only rows whose pose is finite on ALL twelve joints are scored, because a
    per-joint mask would give each joint a different window and the twelve
    numbers would then not be comparable with each other.

    ``segments`` is ``[(name, t0_s, t1_s), ...]`` in the recording's relative
    clock.  The RS485 version derived these from ``benchmark_start`` events; this
    schema records no such event, so they are an argument the caller supplies
    from whatever it knows about the session.

    WHAT THIS DOES NOT SHOW.  A good score here says the twin reproduces the
    trajectory this recording drove, under this recording's pressures.  It says
    nothing about a trajectory the recording did not visit, nothing about the
    two populations' behaviour under a host stall (no recorded stall, no scored
    failsafe), and nothing about force scale -- the anchored law goes slack at a
    geometry-limited equilibrium, so a large coefficient error moves the settled
    angle by a few per cent and hides inside these residuals.
    """
    if not tw.valid:
        return {"valid": False, "reason": tw.reason, "algo_version": ALGO_VERSION}

    usable = np.all(np.isfinite(tw.real_defl_rad), axis=1)
    labels = joint_population(rec)
    board_labels = np.array(["tle" if v else "7mm" for v in rec.is_tle], dtype=object)

    def window(mask, name):
        err_q = (tw.twin_defl_rad - tw.real_defl_rad)[mask]
        real_q = tw.real_defl_rad[mask]
        err_p = (tw.p_sim_pa - tw.p_real_pa)[mask]
        real_p = tw.p_real_pa[mask]
        jb = _score_block(err_q, real_q, unit_scale=180.0 / np.pi,
                          min_samples=min_samples)
        bb = _score_block(err_p, real_p, min_samples=min_samples)
        if jb is None:
            return {"name": name, "n": int(np.count_nonzero(mask)), "scored": False,
                    "reason": f"fewer than {min_samples} usable samples"}
        return {
            "name": name,
            "scored": True,
            "n": jb["n"],
            "joints": {
                "rms_deg": jb["rms"].tolist(),
                "rms_deg_mean": jb["rms_mean"],
                "rms_deg_max": jb["rms_max"],
                "peak_err_deg": jb["peak"].tolist(),
                "nrmse": jb["nrmse"].tolist(),
                "nrmse_mean": jb["nrmse_mean"],
                "population": [str(v) for v in labels],
                "by_population": _by_population(jb, labels, key="rms"),
            },
            "boards": None if bb is None else {
                "ids": [int(v) for v in rec.ids],
                "rms_pa": bb["rms"].tolist(),
                "rms_pa_mean": bb["rms_mean"],
                "rms_pa_max": bb["rms_max"],
                "peak_err_pa": bb["peak"].tolist(),
                "nrmse": bb["nrmse"].tolist(),
                "nrmse_mean": bb["nrmse_mean"],
                "population": [str(v) for v in board_labels],
                "by_population": _by_population(bb, board_labels, key="rms"),
            },
        }

    out = {
        "valid": True,
        "algo_version": ALGO_VERSION,
        "reference_choice": tw.meta.get("reference_choice"),
        "n_ref_rows": tw.meta.get("n_ref_rows"),
        "alignment_exact": tw.meta.get("alignment_exact"),
        "alignment_max_dt_s": tw.meta.get("alignment_max_dt_s"),
        "ctrl_min_n": float(tw.ctrl_min_n),
        "population_source": "variant byte (data_schema.md board_type)",
        "overall": window(usable, "overall"),
        "segments": [],
    }
    for name, t0, t1 in (segments or []):
        m = usable & (tw.t_rel_s >= float(t0)) & (tw.t_rel_s <= float(t1))
        out["segments"].append(window(m, str(name)))

    # The ring statistics go through ring_analysis's single entry point on BOTH
    # traces with identical gates, which is the whole reason that entry point
    # exists: a frequency difference measured by two estimators is a statement
    # about the estimators.
    out["ring"] = ring_compare(tw, thresh_deg=ring_thresh_deg, band=ring_band)
    return out


def ring_compare(tw: TwinResult, *, thresh_deg: float = 0.30, band=RA.BAND_HZ,
                 r2_trust: float = RA.R2_TRUST) -> dict:
    """Ring statistics for the real trace and the twin trace, by the SAME code path."""
    if not tw.valid or tw.n < 32:
        return {"available": False,
                "reason": "invalid result or too few samples to filter"}
    t = tw.t_rel_s
    real_deg = np.degrees(tw.real_defl_rad)
    sim_deg = np.degrees(tw.twin_defl_rad)
    kw = dict(band=band, thresh=thresh_deg)
    try:
        real_eps = RA.episodes_from_trace(t, real_deg, **kw)
        sim_eps = RA.episodes_from_trace(t, sim_deg, **kw)
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "thresh_deg": float(thresh_deg),
        "band_hz": [float(band[0]), float(band[1])],
        "real": RA.ring_metrics(real_eps, r2_trust=r2_trust),
        "twin": RA.ring_metrics(sim_eps, r2_trust=r2_trust),
    }


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cache_key(rec: R.Recording, *, ckpt_path: "str | None" = None,
              custom: bool = False) -> "str | None":
    """The identity a cached twin answers for, or None when it must not be cached.

    ``None`` for any rollout with a custom actuator, custom simulator arguments
    or a truncation, because :func:`twin_for` would then serve a fit candidate as
    THE twin for that recording, and every later number would silently belong to
    a discarded parameter set.
    """
    if custom:
        return None
    parts = [f"algo{ALGO_VERSION}"]
    for p in (rec.path, ckpt_path):
        if not p:
            parts.append("-")
            continue
        try:
            st = os.stat(p)
            parts.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{p}:missing")
    return "|".join(parts)


def twin_for(session_path: str, *, cache_name: str = "twin.npz", ckpt_path=None,
             refresh: bool = False, **kwargs) -> "tuple[TwinResult, R.Recording]":
    """The canonical twin of a recording, cached beside it.

    Anything that would make the rollout non-canonical -- an actuator, extra
    simulator arguments, a ``t_max_s`` -- disables the cache entirely rather
    than writing under a different name, because a second cache file next to the
    first is a thing somebody eventually loads by accident.
    """
    rec = R.load_recording(session_path)
    custom = bool(kwargs.get("t_max_s") is not None or kwargs.get("arm") is not None
                  or kwargs.get("arm_factory") is not None
                  or any(k not in _CACHEABLE_KWARGS for k in kwargs))
    key = cache_key(rec, ckpt_path=ckpt_path, custom=custom)
    cache = os.path.join(session_path, cache_name)
    if key and not refresh and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            if str(z["cache_key"]) == key:
                meta = json.loads(str(z["meta"]))
                return TwinResult(
                    rec_rows=z["rec_rows"], t_rel_s=z["t_rel_s"],
                    q_sim_rad=z["q_sim_rad"], twin_defl_rad=z["twin_defl_rad"],
                    real_defl_rad=z["real_defl_rad"], p_sim_pa=z["p_sim_pa"],
                    p_real_pa=z["p_real_pa"], ref_rows=z["ref_rows"],
                    sim_ref_rad=z["sim_ref_rad"], real_ref_rad=z["real_ref_rad"],
                    alignment=None, ctrl_min_n=float(z["ctrl_min_n"]),
                    valid=bool(z["valid"]), reason=str(z["reason"]), meta=meta), rec
    tw = twin_rollout(rec, **kwargs)
    if key and tw.valid:
        np.savez_compressed(
            cache, cache_key=key, rec_rows=tw.rec_rows, t_rel_s=tw.t_rel_s,
            q_sim_rad=tw.q_sim_rad, twin_defl_rad=tw.twin_defl_rad,
            real_defl_rad=tw.real_defl_rad, p_sim_pa=tw.p_sim_pa,
            p_real_pa=tw.p_real_pa, ref_rows=tw.ref_rows, sim_ref_rad=tw.sim_ref_rad,
            real_ref_rad=tw.real_ref_rad, ctrl_min_n=np.float64(tw.ctrl_min_n),
            valid=np.bool_(tw.valid), reason=tw.reason,
            meta=json.dumps(tw.meta, default=str))
    return tw, rec


#: Keyword arguments a cached canonical twin may carry.  Anything else forces
#: ``cache_key`` to None; kept as a set rather than an ``in`` chain so adding an
#: argument to ``twin_rollout`` fails closed.
_CACHEABLE_KWARGS = {"ref_window_s", "explicit_ref_window_s", "tol_s"}


def main(argv=None) -> int:
    """``python -m digital_twin.twin_compare <session_dir> [--json out.json]``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    ap.add_argument("--t-max-s", type=float, default=None)
    args = ap.parse_args(argv)

    rec = R.load_recording(args.session)
    tw = twin_rollout(rec, t_max_s=args.t_max_s)
    out = compare_metrics(rec, tw)
    if not out["valid"]:
        print("INVALID:", out["reason"])
        return 1
    ov = out["overall"]
    print(f"aligned exactly: {out['alignment_exact']}  "
          f"reference: {out['reference_choice']}  n={ov.get('n')}")
    if ov.get("scored"):
        print(f"joint rms {ov['joints']['rms_deg_mean']:.3f} deg mean, "
              f"nrmse {ov['joints']['nrmse_mean']:.3f} (1.0 = no better than "
              f"'it does not move')")
        for pop, blk in ov["joints"]["by_population"].items():
            print(f"  {pop:>7}: rms {blk['rms_mean']:.3f} deg over "
                  f"{blk['n_columns']} joints, nrmse {blk['nrmse_mean']:.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REF_WINDOW_S", "REF_ROWS_MAX", "IDLE_TARGET_PA", "MIN_SAMPLES", "ALGO_VERSION",
    "Alignment", "TwinResult",
    "joint_population", "align_on_sync", "pick_reference_rows", "twin_rollout",
    "compare_metrics", "ring_compare", "cache_key", "twin_for", "main",
]
