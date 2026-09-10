r"""Offline rollouts: no threads, no wall clock, bit-identical repeats.

The instrument every fit and every regression runs through.  It drives a bare
``sim_core.SimArm`` from a recording with the wall clock removed entirely, so
two runs of the same inputs produce the same floats -- which is what makes a
change to the model visible as a diff rather than as a difference in scheduling.

The determinism is the feature, and it is easy to lose: one ``time.monotonic()``
in a control path, one dict iteration order that leaks into a sum, one thread,
and a regression becomes a thing that reproduces four times in five.  Nothing in
this module sleeps, reads a clock, or starts a thread, and every array it builds
is preallocated and filled by index rather than appended to, so the float
arithmetic does not depend on how the loop was chunked.

THE ROLLOUT IS DRIVEN ON ``can_sync_time_s`` AND ON NOTHING ELSE.  This is the
one thing that must not be simplified into a nominal 6.667 ms grid.  The sync
edge is the instant every board latches its filtered pressure and promotes its
staged target, so it is the single moment the whole arm agrees on, and it is why
``data_schema.md`` records it separately from ``timestamp_s`` and
``cycle_start_time_s``.  The collector's own timing summary on the flagship
legacy hour run reports a 6.666 ms median cycle against a 25.069 ms maximum:
replaying that on a nominal grid would place every sample after the first stall
at a time the arm was never at, and the resulting model error would be a
timebase error wearing a plant error's clothes.  The recorded jitter is the
sysid time base, deliberately, exactly as the RS485 replay kept the 8-9 ms/tick
Windows slip in its data.

THE FORCE CLIP IS AN ERROR, NOT A WARNING.  The generator's ``ctrlrange`` is
sized so the clip never engages inside the operator's 30 psi envelope, so a
rollout that touches it is running silently different physics -- fatal for a fit,
because the optimiser then tunes against a saturation rather than against the
plant.  :func:`rollout` raises :class:`ClampViolation` by default;
``assert_no_clamp=False`` records ``ctrl_min_n`` without raising, which is the
mode :mod:`digital_twin.force_audit` uses to *measure* the headroom.

THE ARM SEAM IS EXPLICIT, because ``CONTRACT.md`` section 4 fixes
``SimArm.advance_to`` and the order of operations inside a quantum but does not
fix the call a host uses to put targets on the wire or to read state back.  The
three touch points are therefore arguments with named defaults --
:func:`default_advance`, :func:`default_apply_targets`, :func:`default_sample`
-- and the defaults prefer ``sim_core``'s own pair, ``stage_targets`` in raw ADC
counts followed by ``sync_edge``, falling back to a Pa-keyed writer and finally
raising with the seam named.  Guessing instead would be silently wrong by the
ADC scale, about a factor of sixty, which reads as a badly fitted model rather
than as a unit error.  :func:`default_sample` records which accessor it used in
``Rollout.meta`` for the same reason: a rollout that quietly read zeros because
an accessor was missing is the failure mode that produces a plausible number.

Ported from ``C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\
UMArm_SIM\replay.py`` via ``reference/sim_core.md`` section 3.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Units and calibration.  CONTRACT.md section 1: Pa gauge everywhere inside the
# twin, raw ADC counts on the wire, psi only at a human boundary.
# ---------------------------------------------------------------------------

PA_PER_PSI = 6894.757

#: Variant bytes as ``data_schema.md`` records them, from the board's answer to
#: ``CMD_GET_FW_VERSION``.  The population is read from THIS and never from the
#: id range: a TLE board legitimately sat at ``0x114`` during the 2026-08 bench
#: session, and reading it on the 7 mm scale is a 10 % error in psi at the top
#: of the range.
VARIANT_7MM = 0x00
VARIANT_DT = 0x01
VARIANT_TLE_DVP = 0x02

#: ``(zero_counts, counts_per_psi)`` per variant, transcribed from
#: ``TLE_PCB/tlelib/proto.py`` (``TLE_ZERO_COUNTS``/``TLE_COUNTS_PER_PSI`` and
#: their legacy pair).  Duplicated rather than imported because ``tlelib`` is
#: only importable after a ``sys.path`` insertion of ``TLE_PCB/``, and a replay
#: that cannot run without the host bus stack is a replay that will not run in
#: CI.  ``test_replay.py`` asserts these against ``tlelib`` whenever it imports,
#: which is what keeps the copy from drifting.
VARIANT_CAL = {
    VARIANT_TLE_DVP: (943.75, 60.78125),
    VARIANT_7MM: (754.4, 56.14),
    VARIANT_DT: (754.4, 56.14),
}

#: The two boards that read a non-zero pressure with the arm at rest, measured
#: 2026-08-21 (``UMArm_KINEMATICS.canarm_actuators.KNOWN_BOARD_FAULTS``).
#: Subtracted from the MEASURED pressure and deliberately not from the target:
#: the board regulates against its own uncorrected reading, so the offset is a
#: real steady-state pressure error in the metal, and removing it from both
#: sides would hide a plant behaviour the twin is supposed to reproduce.
KNOWN_REST_OFFSET_PSI = {0x104: 1.1, 0x110: 5.1}

#: Pull-only force clip, N, from ``CONTRACT.md`` section 2.  Negative is pull.
FORCE_CLIP_N = 4000.0

#: How close to the clip counts as touching it, N.  A float rollout lands on the
#: clip exactly when it saturates, so the epsilon only guards the last ulp of the
#: comparison rather than defining a soft band.
CLAMP_EPS_N = 1e-6

#: Tail sample spacing, s, and the reason the tail exists: with no further sync
#: edges the TLE boards' 500 ms failsafe drops their outputs while the sixteen
#: legacy boards have no link-loss timeout at all and hold their last target
#: indefinitely.  The tail is how a rollout shows that asymmetry.
TAIL_DT_S = 0.05

#: Base ids in column order, ``data_schema.md``'s ``selected_ids``.
DEFAULT_IDS = tuple(range(0x101, 0x119))

N_NODES = 24
N_JOINTS = 12


class ClampViolation(RuntimeError):
    """A rollout drove a tendon command onto the pull-only force clip.

    Raised rather than warned because the physics past the clip is not the
    physics the model describes: every gradient a fit takes through a saturated
    sample is a gradient of the clip.
    """


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    """One ``session_*/`` recording, read into arrays, in the twin's units.

    ``t_sync_s`` is ``can_sync_time_s`` and is the only clock anything here
    drives on.  ``t_rel_s`` is the same clock with its first sample at zero,
    which is what every downstream plot and every alignment uses; keeping both
    means a caller never has to guess which origin a number is against.
    """

    t_sync_s: np.ndarray          # (n,)  s, the sync edge, strictly increasing
    t_rel_s: np.ndarray           # (n,)  s, t_sync_s - t_sync_s[0]
    cycle: np.ndarray             # (n,)  the collector's cycle counter
    ids: np.ndarray               # (24,) base ids, column order
    board_type: np.ndarray        # (24,) variant byte the board REPORTED
    is_tle: np.ndarray            # (24,) bool, board_type == VARIANT_TLE_DVP
    q_rad: np.ndarray             # (n, 12) rad, UMArm_KINEMATICS order
    qdot_rad_s: np.ndarray        # (n, 12) rad/s
    p_adc: np.ndarray             # (n, 24) raw counts, as recorded
    target_adc: np.ndarray        # (n, 24) raw counts, as commanded
    p_pa: np.ndarray              # (n, 24) Pa gauge, rest offset removed
    target_pa: np.ndarray         # (n, 24) Pa gauge, as commanded
    q_valid: np.ndarray           # (n,) bool, joints usable this cycle
    meta: dict = field(default_factory=dict)
    path: str = ""

    @property
    def n(self) -> int:
        return int(self.t_sync_s.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.t_sync_s[-1] - self.t_sync_s[0]) if self.n else 0.0

    @property
    def rate_hz(self) -> float:
        """From the MEDIAN gap: a mean is pulled by the stalls the timing summary reports."""
        if self.n < 2:
            return float("nan")
        return 1.0 / float(np.median(np.diff(self.t_sync_s)))

    def population_mask(self, tle: bool) -> np.ndarray:
        return self.is_tle.copy() if tle else ~self.is_tle

    def slice_time(self, t_max_s: "float | None") -> "Recording":
        """The first ``t_max_s`` seconds, by the sync clock.

        A fit runs candidates on a truncated recording to buy evaluations, so
        this exists to make that truncation one operation with one definition of
        the origin rather than a slice each caller writes for itself.
        """
        if t_max_s is None:
            return self
        keep = self.t_rel_s <= float(t_max_s)
        if not np.any(keep):
            raise ValueError(
                f"t_max_s={t_max_s} s cuts away the whole recording, whose "
                f"first sync edge is at t_rel 0 and last at {self.t_rel_s[-1]} s")
        k = int(np.count_nonzero(keep))
        return Recording(
            t_sync_s=self.t_sync_s[:k], t_rel_s=self.t_rel_s[:k],
            cycle=self.cycle[:k], ids=self.ids, board_type=self.board_type,
            is_tle=self.is_tle, q_rad=self.q_rad[:k], qdot_rad_s=self.qdot_rad_s[:k],
            p_adc=self.p_adc[:k], target_adc=self.target_adc[:k],
            p_pa=self.p_pa[:k], target_pa=self.target_pa[:k],
            q_valid=self.q_valid[:k], meta=dict(self.meta), path=self.path)


def counts_to_pa(counts, variant: int) -> np.ndarray:
    """Raw ADC counts to Pa gauge, on the calibration the BOARD's variant implies."""
    zero, per_psi = VARIANT_CAL.get(int(variant), VARIANT_CAL[VARIANT_7MM])
    return (np.asarray(counts, dtype=np.float64) - zero) / per_psi * PA_PER_PSI


def pa_to_counts(pa, variant: int) -> np.ndarray:
    zero, per_psi = VARIANT_CAL.get(int(variant), VARIANT_CAL[VARIANT_7MM])
    return np.asarray(pa, dtype=np.float64) / PA_PER_PSI * per_psi + zero


def _chunk_paths(session_dir: str) -> "list[str]":
    """Chunk files in replay order, from the manifest if there is one.

    The manifest is preferred because it also carries ``active``: a session
    whose ``active`` is still true was interrupted and its last chunk is short,
    and a bare glob would sort that in as if it were complete.  Falling back to
    a sorted glob keeps a hand-assembled directory usable, which is what the
    tests build.
    """
    man = os.path.join(session_dir, "manifest.json")
    if os.path.isfile(man):
        with open(man, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        chunks = manifest.get("chunks") or []
        if chunks:
            return [os.path.join(session_dir, os.path.basename(c["path"]))
                    for c in chunks]
    return sorted(glob.glob(os.path.join(session_dir, "samples_chunk_*.jsonl")))


def recording_from_session(path: str, *, rest_offset_psi=KNOWN_REST_OFFSET_PSI,
                           episode_field: str = "segment_index",
                           kinds=None, max_rows: "int | None" = None) -> Recording:
    """Read a session through :func:`digital_twin.dataset.load_session`.

    One path from file to arrays.  This module and ``dataset`` were written in
    parallel against the same schema document and each grew its own reader, and
    two subtly different readers is exactly how the training inputs and the
    evaluation inputs drift apart -- the fit would then be scored against a
    slightly different recording than it was fitted on, and nothing would say so.

    Delegating also means this side inherits the two corrections that only
    showed up when a reader met a real recording: missed replies carried forward
    instead of being fitted as zero counts, and ``q`` taken from the resampled
    mocap stream instead of the per-cycle held value.

    ``kinds`` selects excitation families by ``segment_kind`` -- the deliverable
    scores the held-out ``validation`` sequence and nothing else.
    """
    from digital_twin import dataset as _ds

    rec = _ds.load_session(path, episode_field=episode_field,
                           rest_offset_psi=rest_offset_psi)
    keep = np.ones(rec.n_cycles, dtype=bool)
    if kinds is not None:
        want = set(kinds)
        keep &= np.array([str(v) in want for v in rec.phase], dtype=bool)
    if max_rows is not None:
        idx = np.nonzero(keep)[0][:int(max_rows)]
        keep = np.zeros_like(keep)
        keep[idx] = True
    if not keep.any():
        raise ValueError(f"no cycles left in {path!r} after selecting "
                         f"kinds={kinds!r}")

    t = rec.can_sync_time_s[keep]
    meta = dict(rec.meta)
    meta["selected_kinds"] = list(kinds) if kinds else None
    meta["selected_cycles"] = int(keep.sum())
    return Recording(
        t_sync_s=t, t_rel_s=t - t[0], cycle=rec.cycle[keep],
        ids=np.asarray(rec.ids), board_type=np.asarray(rec.board_type),
        is_tle=np.asarray(rec.board_type) == VARIANT_TLE_DVP,
        q_rad=rec.q[keep], qdot_rad_s=rec.qdot[keep],
        p_adc=rec.pressure_adc[keep], target_adc=rec.target_adc[keep],
        p_pa=rec.pressure_pa[keep], target_pa=rec.target_pa[keep],
        q_valid=rec.q_valid[keep], meta=meta, path=os.path.abspath(path))


def load_recording(path: str, *, ids=DEFAULT_IDS,
                   rest_offset_psi=KNOWN_REST_OFFSET_PSI,
                   require_increasing_sync: bool = True,
                   max_rows: "int | None" = None) -> Recording:
    """Read a ``session_*/`` recording into arrays, in Pa gauge and radians.

    ``board_type`` is required, not optional.  Without it the population split
    is unrecoverable after the fact -- it is not derivable from the id range --
    and a recording scored as one plant reports a number describing neither.  A
    session that predates the field raises here rather than defaulting, because
    defaulting the top eight to the 7 mm calibration is a silent 10 % pressure
    error at the top of the range.

    ``require_increasing_sync`` is on because a non-increasing sync clock means
    the collector's cycles were reordered or duplicated, and a rollout driven on
    a clock that goes backwards runs :meth:`SimArm.advance_to`'s stale-target
    no-op instead of stepping, i.e. it silently drops the sample rather than
    failing.
    """
    session = os.path.abspath(path)
    if not os.path.isdir(session):
        raise FileNotFoundError(f"recording directory not found: {session}")
    meta_path = os.path.join(session, "metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    rows: "list[dict]" = []
    for cp in _chunk_paths(session):
        with open(cp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_rows is not None and len(rows) >= int(max_rows):
                    break
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    if not rows:
        raise ValueError(f"no sample rows found under {session}")

    n = len(rows)
    ids_arr = np.asarray([int(x, 16) if isinstance(x, str) else int(x)
                          for x in (rows[0].get("ids") or meta.get("selected_ids")
                                    or list(ids))], dtype=np.int64)
    if ids_arr.shape[0] != N_NODES:
        raise ValueError(f"recording names {ids_arr.shape[0]} boards, expected {N_NODES}")

    bt = rows[0].get("board_type", meta.get("board_type"))
    if bt is None:
        raise ValueError(
            f"{session} carries no board_type; the two populations cannot be "
            f"separated after the fact and the split is not recoverable from "
            f"the id range (see data_schema.md)")
    board_type = np.asarray([int(v) for v in bt], dtype=np.int64)
    if board_type.shape[0] != N_NODES:
        raise ValueError(f"board_type has {board_type.shape[0]} entries, expected {N_NODES}")

    t_sync = np.empty(n, dtype=np.float64)
    cycle = np.zeros(n, dtype=np.int64)
    q = np.full((n, N_JOINTS), np.nan, dtype=np.float64)
    qdot = np.full((n, N_JOINTS), np.nan, dtype=np.float64)
    p_adc = np.zeros((n, N_NODES), dtype=np.float64)
    tgt_adc = np.zeros((n, N_NODES), dtype=np.float64)
    q_valid = np.zeros(n, dtype=bool)

    for k, row in enumerate(rows):
        t_sync[k] = float(row["can_sync_time_s"])
        cycle[k] = int(row.get("cycle", k))
        state = row.get("robot_state") or {}
        q[k] = np.asarray(state.get("q", q[k]), dtype=np.float64)
        qdot[k] = np.asarray(state.get("qdot", qdot[k]), dtype=np.float64)
        p_adc[k] = np.asarray(state.get("pressure_adc"), dtype=np.float64)
        tgt_adc[k] = np.asarray((row.get("input") or {}).get("target_adc"),
                                dtype=np.float64)
        mocap = row.get("mocap") or {}
        # Two different questions, per data_schema.md: `stale` is "no frame at
        # all", q staleness is "frames arriving, none converting", and it is the
        # second that invalidates this sample's joints while mocap.valid can
        # still read true.  Whichever the collector wrote, both are consulted.
        ok = bool(row.get("joint_current_valid", True))
        ok = ok and not bool(row.get("q_stale", False))
        if mocap:
            ok = ok and bool(mocap.get("valid", True)) and not bool(mocap.get("stale", False))
        q_valid[k] = ok

    if require_increasing_sync:
        bad = np.flatnonzero(np.diff(t_sync) <= 0.0)
        if bad.size:
            raise ValueError(
                f"can_sync_time_s is not strictly increasing: {bad.size} of "
                f"{n - 1} gaps are non-positive, first at row {int(bad[0])} "
                f"({t_sync[bad[0]]} -> {t_sync[bad[0] + 1]})")

    p_pa = np.empty_like(p_adc)
    target_pa = np.empty_like(tgt_adc)
    offsets = dict(rest_offset_psi or {})
    for c in range(N_NODES):
        v = int(board_type[c])
        p_pa[:, c] = counts_to_pa(p_adc[:, c], v)
        target_pa[:, c] = counts_to_pa(tgt_adc[:, c], v)
        off_psi = float(offsets.get(int(ids_arr[c]), 0.0))
        if off_psi:
            p_pa[:, c] -= off_psi * PA_PER_PSI

    return Recording(
        t_sync_s=t_sync, t_rel_s=t_sync - t_sync[0], cycle=cycle, ids=ids_arr,
        board_type=board_type, is_tle=(board_type == VARIANT_TLE_DVP),
        q_rad=q, qdot_rad_s=qdot, p_adc=p_adc, target_adc=tgt_adc, p_pa=p_pa,
        target_pa=target_pa, q_valid=q_valid,
        meta={"session": meta, "n_rows": n,
              "rest_offset_psi": {int(k): float(v) for k, v in offsets.items()}},
        path=session)


def recording_from_arrays(t_sync_s, q_rad, p_adc, target_adc, board_type, *,
                          ids=DEFAULT_IDS, qdot_rad_s=None, cycle=None,
                          q_valid=None, rest_offset_psi=None,
                          meta=None, path="") -> Recording:
    """Build a :class:`Recording` from arrays, for tests and for converted data.

    The predecessor ``.npz`` format stored pressures in Pa rather than counts, so
    a forward conversion has arrays and no session directory; this is the seam
    it arrives through, and it applies exactly the same unit and offset
    arithmetic as :func:`load_recording` so a converted recording and a native
    one are scored by the same code.
    """
    t_sync_s = np.asarray(t_sync_s, dtype=np.float64)
    n = int(t_sync_s.shape[0])
    ids_arr = np.asarray([int(v) for v in ids], dtype=np.int64)
    board_type = np.asarray([int(v) for v in board_type], dtype=np.int64)
    q_rad = np.asarray(q_rad, dtype=np.float64).reshape(n, N_JOINTS)
    p_adc = np.asarray(p_adc, dtype=np.float64).reshape(n, N_NODES)
    target_adc = np.asarray(target_adc, dtype=np.float64).reshape(n, N_NODES)
    qdot = (np.zeros((n, N_JOINTS)) if qdot_rad_s is None
            else np.asarray(qdot_rad_s, dtype=np.float64).reshape(n, N_JOINTS))
    cyc = (np.arange(n, dtype=np.int64) if cycle is None
           else np.asarray(cycle, dtype=np.int64))
    valid = (np.ones(n, dtype=bool) if q_valid is None
             else np.asarray(q_valid, dtype=bool))

    p_pa = np.empty_like(p_adc)
    target_pa = np.empty_like(target_adc)
    offsets = dict(rest_offset_psi or {})
    for c in range(N_NODES):
        v = int(board_type[c])
        p_pa[:, c] = counts_to_pa(p_adc[:, c], v)
        target_pa[:, c] = counts_to_pa(target_adc[:, c], v)
        off_psi = float(offsets.get(int(ids_arr[c]), 0.0))
        if off_psi:
            p_pa[:, c] -= off_psi * PA_PER_PSI

    return Recording(
        t_sync_s=t_sync_s, t_rel_s=t_sync_s - t_sync_s[0], cycle=cyc, ids=ids_arr,
        board_type=board_type, is_tle=(board_type == VARIANT_TLE_DVP), q_rad=q_rad,
        qdot_rad_s=qdot, p_adc=p_adc, target_adc=target_adc, p_pa=p_pa,
        target_pa=target_pa, q_valid=valid,
        meta=dict(meta or {}, rest_offset_psi={int(k): float(v)
                                              for k, v in offsets.items()}),
        path=path)


def save_recording(session_dir: str, rec: Recording, *, chunk_samples: int = 9000,
                   session_tag: str = "synthetic") -> str:
    """Write a :class:`Recording` back out in the ``data_schema.md`` layout.

    Its purpose is to let a test exercise the loader on real files rather than
    on a hand-built object, so that a schema change breaks the loader test
    rather than only the production path.  It is not a collector: it writes the
    fields the twin reads and states so in ``metadata.json``.
    """
    os.makedirs(session_dir, exist_ok=True)
    ids_hex = [f"0x{int(v):03X}" for v in rec.ids]
    meta = {
        "schema_version": 2,
        "session_tag": session_tag,
        "sample_rate_hz": float(rec.rate_hz),
        "duration_s": float(rec.duration_s),
        "state_fields": ["q", "qdot", "pressure_adc"],
        "state_dimension": 48,
        "input_fields": ["target_adc"],
        "input_dimension": N_NODES,
        "pressure_units": "adc_counts",
        "selected_ids": ids_hex,
        "board_type": [int(v) for v in rec.board_type],
        "written_by": "digital_twin.replay.save_recording -- twin-readable "
                      "subset of data_schema.md, not a collector output",
    }
    with open(os.path.join(session_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    chunks = []
    k = 0
    while k < rec.n:
        hi = min(rec.n, k + int(chunk_samples))
        reason = "complete" if hi == rec.n else "checkpoint"
        name = f"samples_chunk_{len(chunks):04d}_{reason}_0.jsonl"
        with open(os.path.join(session_dir, name), "w", encoding="utf-8") as fh:
            for i in range(k, hi):
                fh.write(json.dumps({
                    "schema_version": 2,
                    "cycle": int(rec.cycle[i]),
                    "phase": "collecting",
                    "timestamp_s": float(rec.t_sync_s[i]),
                    "cycle_start_time_s": float(rec.t_sync_s[i]),
                    "can_sync_time_s": float(rec.t_sync_s[i]),
                    "ids": ids_hex,
                    "board_type": [int(v) for v in rec.board_type],
                    "robot_state": {
                        "q": [float(v) for v in rec.q_rad[i]],
                        "qdot": [float(v) for v in rec.qdot_rad_s[i]],
                        "pressure_adc": [float(v) for v in rec.p_adc[i]],
                    },
                    "input": {"target_adc": [float(v) for v in rec.target_adc[i]]},
                    "joint_current_valid": bool(rec.q_valid[i]),
                }) + "\n")
        chunks.append({"path": name, "reason": reason, "samples": hi - k,
                       "start_cycle": int(rec.cycle[k]), "end_cycle": int(rec.cycle[hi - 1]),
                       "start_time_s": float(rec.t_sync_s[k]),
                       "end_time_s": float(rec.t_sync_s[hi - 1])})
        k = hi
    with open(os.path.join(session_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"total_samples": rec.n, "checkpoint_count": len(chunks),
                   "active": False, "chunks": chunks}, fh, indent=1)
    return session_dir


# ---------------------------------------------------------------------------
# The arm seam.  Three callables, three named defaults.
# ---------------------------------------------------------------------------


def default_advance(arm, t_s: float) -> None:
    """Move the arm's clock.  ``advance_to`` is pinned by CONTRACT.md section 4."""
    arm.advance_to(float(t_s))


def default_apply_targets(arm, rec: "Recording", k: int, t_s: float, *,
                          enable: bool = True) -> None:
    """Stage row ``k``'s runtime table and fire the sync edge that promotes it.

    THE PREFERRED PATH REPLAYS THE WIRE WORDS THEMSELVES.  ``sim_core.SimArm``
    takes ``stage_targets({base: (counts, enable)})`` in raw ADC counts, which is
    what the host actually transmitted and what ``data_schema.md`` recorded, so
    the recorded ``target_adc`` goes straight back on the wire with no round
    trip through Pa.  Converting to Pa and back would re-quantise every target
    on the board's own calibration and put a fraction of a count of error into
    the one signal the rollout is driven by.

    Every board is tabled every cycle, including boards holding at idle.  That
    is what ``backend.py::_build_targets`` does on the metal, and a caller that
    tables only a selection reproduces the bus's worst failure: a de-selected
    board regulating with nothing able to reach it, and the sync-loss failsafe
    unable to help because the master is still sending edges.

    The Pa fallbacks exist for arms that do not carry the ADC seam.  An arm with
    none of the three raises with the seam named, because the failure that must
    not happen quietly is an arm whose target call takes counts being handed Pa
    -- a factor of about sixty, which looks like a badly fitted model rather
    than like a unit error.
    """
    stage = getattr(arm, "stage_targets", None)
    edge = getattr(arm, "sync_edge", None)
    if callable(stage) and callable(edge):
        counts = rec.target_adc[k]
        stage({int(base): (int(round(float(c))), bool(enable))
               for base, c in zip(rec.ids, counts)})
        edge(float(t_s))
        return
    mapping = {int(base): float(v) for base, v in zip(rec.ids, rec.target_pa[k])}
    setter = getattr(arm, "set_targets_pa", None)
    if callable(setter):
        setter(mapping, now=float(t_s))
        return
    tick = getattr(arm, "tick", None)
    if callable(tick):
        tick(mapping, now=float(t_s))
        return
    raise TypeError(
        f"{type(arm).__name__} offers none of stage_targets(...)+sync_edge(...), "
        f"set_targets_pa(mapping, now=) or tick(mapping, now=); pass "
        f"replay.rollout(..., apply_targets=<your writer>) rather than letting "
        f"replay guess the wire units")


def default_sample(arm) -> dict:
    """Read ``q`` (rad), plant pressure (Pa) and the tendon commands (N) back.

    Every accessor is tried by its contract name first and by its MuJoCo backing
    array second, so a rollout works against a partially built ``SimArm`` and
    reports in ``Rollout.meta`` which path it took -- a rollout that silently
    read zeros because an accessor was missing is the failure this guards.
    """
    out = {}

    q = getattr(arm, "q", None)
    if callable(q):
        out["q_rad"] = np.asarray(q(), dtype=np.float64)[:N_JOINTS]
        out["q_via"] = "arm.q()"
    else:
        out["q_rad"] = np.asarray(arm.data.qpos, dtype=np.float64)[:N_JOINTS]
        out["q_via"] = "data.qpos"

    for name in ("pressures_pa", "p_pa"):
        acc = getattr(arm, name, None)
        if callable(acc):
            out["p_pa"] = np.asarray(acc(), dtype=np.float64)[:N_NODES]
            out["p_via"] = f"arm.{name}()"
            break
    else:
        out["p_pa"] = np.zeros(N_NODES, dtype=np.float64)
        out["p_via"] = "unavailable"

    ctrl = getattr(arm, "ctrl_n", None)
    if callable(ctrl):
        out["ctrl_n"] = np.asarray(ctrl(), dtype=np.float64)[:N_NODES]
        out["ctrl_via"] = "arm.ctrl_n()"
    else:
        data = getattr(arm, "data", None)
        if data is not None and getattr(data, "ctrl", None) is not None:
            out["ctrl_n"] = np.asarray(data.ctrl, dtype=np.float64)[:N_NODES]
            out["ctrl_via"] = "data.ctrl"
        else:
            out["ctrl_n"] = np.zeros(N_NODES, dtype=np.float64)
            out["ctrl_via"] = "unavailable"
    return out


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rollout:
    """What the twin looked like at the instant of every sync edge, plus the tail.

    Row ``k`` is the arm as it stood when edge ``k`` hit the wire, sampled
    BEFORE that edge's targets are promoted -- mirroring the firmware, which
    builds its reply payload from the pre-edge sample.  Getting that order
    backwards makes the twin appear to react one full cycle sooner than the
    metal, which reads as a better model.
    """

    t_s: np.ndarray            # (K,)  drive stamps: recording sync edges, then tail
    row_index: np.ndarray      # (K,)  index into the recording; -1 on tail rows
    target_pa: np.ndarray      # (K, 24) Pa commanded at this edge; NaN on tail
    p_pa: np.ndarray           # (K, 24) Pa, plant truth
    q_rad: np.ndarray          # (K, 12) rad
    ctrl_n: np.ndarray         # (K, 24) N, tendon commands (negative is pull)
    ctrl_min_n: float
    clamped: bool
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.t_s.shape[0])

    @property
    def t_rel_s(self) -> np.ndarray:
        return self.t_s - self.t_s[0]

    @property
    def q_deg(self) -> np.ndarray:
        return np.degrees(self.q_rad)

    def save(self, path: str) -> str:
        """One ``.npz``, ``allow_pickle=False`` on the way back in.

        ``meta`` goes in as a JSON string for the same reason: a rollout is an
        artefact that outlives the process that wrote it, and a pickled object
        graph inside it makes reading it back depend on this module's classes
        still existing in the same shape.
        """
        np.savez_compressed(
            path, t_s=self.t_s, row_index=self.row_index, target_pa=self.target_pa,
            p_pa=self.p_pa, q_rad=self.q_rad, ctrl_n=self.ctrl_n,
            ctrl_min_n=np.float64(self.ctrl_min_n), clamped=np.bool_(self.clamped),
            meta=json.dumps(self.meta, default=str))
        return path

    @classmethod
    def load(cls, path: str) -> "Rollout":
        with np.load(path, allow_pickle=False) as z:
            return cls(t_s=z["t_s"], row_index=z["row_index"],
                       target_pa=z["target_pa"], p_pa=z["p_pa"], q_rad=z["q_rad"],
                       ctrl_n=z["ctrl_n"], ctrl_min_n=float(z["ctrl_min_n"]),
                       clamped=bool(z["clamped"]),
                       meta=json.loads(str(z["meta"])))


def assert_no_clamp(roll: Rollout, *, force_clip_n: float = FORCE_CLIP_N,
                    clamp_eps_n: float = CLAMP_EPS_N) -> None:
    """Raise :class:`ClampViolation` if the rollout touched the pull-only clip.

    Separate from :func:`rollout` so the check can also be re-run on a saved
    rollout, and so :mod:`digital_twin.force_audit` can measure the headroom
    with the check disarmed and then arm it on the same arrays.
    """
    low = float(roll.ctrl_min_n)
    if low <= -(float(force_clip_n) - float(clamp_eps_n)):
        k = int(np.argmin(np.min(roll.ctrl_n, axis=1)))
        act = int(np.argmin(roll.ctrl_n[k]))
        raise ClampViolation(
            f"tendon command reached {low} N against the {-force_clip_n} N "
            f"pull-only clip, first at rollout row {k} (t={float(roll.t_s[k])} s, "
            f"pam_{act + 1}); the physics past the clip is a saturation, not the "
            f"model, so this rollout must not be fitted or scored")


def rollout(rec: Recording, *, arm=None, arm_factory=None, tail_s: float = 0.0,
            tail_dt_s: float = TAIL_DT_S, assert_no_clamp_on_touch: bool = True,
            force_clip_n: float = FORCE_CLIP_N, clamp_eps_n: float = CLAMP_EPS_N,
            t_max_s: "float | None" = None,
            advance=default_advance, apply_targets=default_apply_targets,
            sample=default_sample, progress=None, **simarm_kwargs) -> Rollout:
    """Drive ``rec`` through a bare ``SimArm`` on its own sync stamps.

    The loop is ``advance -> sample -> apply``, in that order and no other.
    Advance first because the arm must be at the edge's instant before it is
    read; sample second so row ``k`` is the pre-edge state the firmware would
    have replied with; apply last because a target promoted before the sample
    would appear in the record one cycle early.

    A FRESH ARM PER ROLLOUT IS THE DETERMINISTIC DEFAULT.  Replaying two
    recordings on one arm chains their state, and the second one is then a
    function of the first -- which reproduces perfectly in a loop and not at all
    when either is run alone.  Pass ``arm=`` only to reuse an arm deliberately.

    ``tail_s`` keeps sampling with NO further edges.  On this arm that is not
    cosmetic: the top eight boards drop their outputs 500 ms after the last sync
    edge while the sixteen legacy boards have no link-loss timeout at all, so the
    tail is the only part of a rollout where the two populations visibly diverge
    under identical commands.
    """
    if t_max_s is not None:
        rec = rec.slice_time(t_max_s)
    n = rec.n
    if n == 0:
        raise ValueError("recording has no samples to replay")

    n_tail = int(round(float(tail_s) / float(tail_dt_s))) if tail_s else 0
    total = n + n_tail

    if arm is None:
        if arm_factory is None:
            # The population MUST come from the recorded variant bytes and not
            # from the id block: a TLE board sat at 0x114 during the 2026-08
            # bench session, and a twin that built it as a 7 mm node would give
            # it the wrong valve, the wrong failsafe and the wrong calibration.
            simarm_kwargs.setdefault(
                "variants", {int(b): int(v) for b, v in zip(rec.ids, rec.board_type)})
            arm = _default_arm_factory(**simarm_kwargs)
        else:
            arm = arm_factory(**simarm_kwargs)
    elif simarm_kwargs:
        raise TypeError(
            f"arm= was given together with {sorted(simarm_kwargs)}; those are "
            f"constructor arguments and cannot be applied to an arm that "
            f"already exists")

    t_out = np.empty(total, dtype=np.float64)
    row_index = np.full(total, -1, dtype=np.int64)
    tgt_out = np.full((total, N_NODES), np.nan, dtype=np.float64)
    p_out = np.zeros((total, N_NODES), dtype=np.float64)
    q_out = np.zeros((total, N_JOINTS), dtype=np.float64)
    ctrl_out = np.zeros((total, N_NODES), dtype=np.float64)
    via = {}

    for k in range(n):
        t_k = float(rec.t_sync_s[k])
        advance(arm, t_k)
        s = sample(arm)
        if not via:
            via = {key: s[key] for key in ("q_via", "p_via", "ctrl_via") if key in s}
        t_out[k] = t_k
        row_index[k] = k
        q_out[k] = s["q_rad"]
        p_out[k] = s["p_pa"]
        ctrl_out[k] = s["ctrl_n"]
        tgt_out[k] = rec.target_pa[k]
        apply_targets(arm, rec, k, t_k)
        if progress is not None and (k % 2000) == 0:
            progress(k, total)

    for j in range(n_tail):
        t_j = float(rec.t_sync_s[-1] + (j + 1) * float(tail_dt_s))
        advance(arm, t_j)
        s = sample(arm)
        i = n + j
        t_out[i] = t_j
        q_out[i] = s["q_rad"]
        p_out[i] = s["p_pa"]
        ctrl_out[i] = s["ctrl_n"]

    low = float(np.min(ctrl_out)) if ctrl_out.size else 0.0
    clamped = low <= -(float(force_clip_n) - float(clamp_eps_n))
    roll = Rollout(t_s=t_out, row_index=row_index, target_pa=tgt_out, p_pa=p_out,
                   q_rad=q_out, ctrl_n=ctrl_out, ctrl_min_n=low, clamped=clamped,
                   meta={"n_rows": n, "n_tail": n_tail, "tail_dt_s": float(tail_dt_s),
                         "force_clip_n": float(force_clip_n),
                         "clamp_eps_n": float(clamp_eps_n),
                         "recording_path": rec.path,
                         "arm": type(arm).__name__, "accessors": via,
                         "simarm_kwargs": {k: str(v) for k, v in simarm_kwargs.items()}})
    if assert_no_clamp_on_touch:
        assert_no_clamp(roll, force_clip_n=force_clip_n, clamp_eps_n=clamp_eps_n)
    return roll


def _default_arm_factory(**kwargs):
    """Build a ``sim_core.SimArm``, importing it only when a rollout actually needs one.

    Deferred rather than imported at module scope so that
    :func:`load_recording`, :class:`Recording` and the unit arithmetic here stay
    usable -- and testable -- while ``sim_core`` is still being written by
    another hand.  A module-level import would make every test in this package
    fail on a sibling's half-saved file.
    """
    try:
        from digital_twin import sim_core  # noqa: WPS433 -- deferred on purpose
    except Exception as exc:  # pragma: no cover - depends on a sibling module
        raise RuntimeError(
            f"digital_twin.sim_core is not importable ({exc}); pass "
            f"rollout(..., arm=) or rollout(..., arm_factory=) to replay "
            f"against something else") from exc
    return sim_core.SimArm(**kwargs)


def replay_session(path: str, **kwargs) -> Rollout:
    """Load a ``session_*/`` recording and roll it, in one call."""
    load_kw = {k: kwargs.pop(k) for k in
               ("ids", "rest_offset_psi", "require_increasing_sync", "max_rows")
               if k in kwargs}
    return rollout(load_recording(path, **load_kw), **kwargs)


def main(argv=None) -> int:
    """``python -m digital_twin.replay <session_dir> [--out roll.npz] [--tail-s 2]``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--out", default=None, help="write the rollout to this .npz")
    ap.add_argument("--tail-s", type=float, default=0.0)
    ap.add_argument("--t-max-s", type=float, default=None)
    ap.add_argument("--no-clamp-check", action="store_true",
                    help="record ctrl_min_n without raising ClampViolation")
    args = ap.parse_args(argv)

    rec = load_recording(args.session)
    print(f"{rec.n} cycles, {rec.duration_s:.3f} s, {rec.rate_hz:.4f} Hz, "
          f"{int(np.count_nonzero(rec.is_tle))} TLE boards of {N_NODES}")
    roll = rollout(rec, tail_s=args.tail_s, t_max_s=args.t_max_s,
                   assert_no_clamp_on_touch=not args.no_clamp_check)
    print(f"rollout {roll.n} rows, ctrl_min {roll.ctrl_min_n:.3f} N "
          f"(clip {-FORCE_CLIP_N} N), accessors {roll.meta['accessors']}")
    if args.out:
        print("wrote", roll.save(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PA_PER_PSI", "VARIANT_7MM", "VARIANT_DT", "VARIANT_TLE_DVP", "VARIANT_CAL",
    "KNOWN_REST_OFFSET_PSI", "FORCE_CLIP_N", "CLAMP_EPS_N", "TAIL_DT_S",
    "DEFAULT_IDS", "N_NODES", "N_JOINTS",
    "ClampViolation", "Recording", "Rollout",
    "counts_to_pa", "pa_to_counts", "load_recording", "recording_from_arrays",
    "save_recording", "default_advance", "default_apply_targets", "default_sample",
    "rollout", "replay_session", "assert_no_clamp", "main",
]
