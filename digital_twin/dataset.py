r"""Turn a ``session_*/`` recording into the arrays the flow net is fitted on.

Pure numpy, and deliberately so: ``sim_core`` steps ``actuator_model`` at 1 ms on
machines that have no CUDA, and the only module in this package allowed to import
torch is ``train_actuator_net``.  Everything here -- ADC conversion, tendon
geometry, windowing, the episode split, the leak segments -- is arithmetic on a
file and must stay runnable anywhere the twin runs.

FOUR DECISIONS ARE LOAD-BEARING, and each of them is a place where a plausible
shortcut produces a number that looks fine and is wrong.

1. **ADC is converted with the board's own ``NodeCal``, chosen from the recorded
   variant byte and never from the id range.**  ``data_schema.md`` records
   ``board_type`` per session for exactly this reason: a TLE board sat at
   ``0x114`` during the 2026-08 bench session, and reading it on the 7 mm scale
   (754.4 counts zero, 56.14 counts/psi instead of 943.75 and 60.78125) is a
   10 % error in psi at the top of the range -- about 2 psi at 20 psi, which is
   larger than the 7 mm firmware's own +-2000 Pa (+-0.29 psi) hysteresis band and
   would therefore be read as valve behaviour rather than as a unit error.

2. **Train and holdout are split by whole episode, never by sample.**  At 150 Hz
   two adjacent samples are 6.67 ms apart and the pressure autocorrelation time
   of a McKibben under bang-bang is order seconds, so a random per-sample split
   puts each holdout sample within 6.67 ms of a training sample and reports a
   holdout error that is essentially the training error.  The split here is over
   episodes, and a window may never straddle an episode boundary.

3. **The population is a per-node fact, not a per-session one.**  Anything that
   reports one number across all twenty-four boards reports a number that
   describes neither population (CONTRACT departure 2), so every array here
   carries its node index and every window carries ``is_tle`` and that node's
   own calibration.

4. **Muscle length is anchored, and it is differentiated after smoothing.**
   ``l = l0_seg + (ten_length - tendon_length0)`` is the same number the force law
   consumes.  Mocap plate noise of order 0.3 mm differentiates to roughly
   45 mm/s of spurious ``ldot`` at 150 Hz -- about a fifth of the 0.5 m/s
   normalisation scale -- so ``l`` is smoothed with the reference's centred
   3-sample mean before ``np.gradient``, and the smoothing never reaches across
   an episode boundary.

WHAT THIS MODULE DOES NOT ESTABLISH.  It does not validate the recording's
physics: a session whose mocap silently lost one plate converts to ``q`` without
complaint (``data_schema.md``, "a single untracked plate is invisible"), and
nothing here can see it.  It also does not certify the tendon geometry -- see
:class:`TendonKinematics`, whose fallback is a first-order moment-arm placeholder
with the right units and the wrong geometry.

Read ``data_schema.md`` for the file format and ``CONTRACT.md`` sections 1, 2 and
7 for the units and the feature row.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Pure numpy both ways, so this import costs nothing a rollout cannot afford.
# It is taken rather than duplicating four constants because ``actuator_model``
# is where the fit's coordinates are defined: its ``L0_SEED_M`` is this arm's
# measured rest length, its ``fit_leak_segments`` is the least squares the
# checkpoint's ``leak_pa_s`` must come from, and a second copy of either here
# would be a copy that drifts.  The division of labour is the one that module's
# leak section states: it turns a time/pressure pair into a slope, and choosing
# which stretches of a recording are closed is this module's job because only
# this module knows the schema.
try:
    from . import actuator_model as _am
except ImportError:  # pragma: no cover - depends on how the caller was started
    if _WS_ROOT_FALLBACK := os.path.dirname(os.path.dirname(os.path.abspath(__file__))):
        if _WS_ROOT_FALLBACK not in sys.path:
            sys.path.insert(0, _WS_ROOT_FALLBACK)
    from digital_twin import actuator_model as _am  # type: ignore[no-redef]

__all__ = [
    "PA_PER_PSI", "ADC_MAX", "N_NODES", "N_JOINTS",
    "VARIANT_7MM", "VARIANT_DT", "VARIANT_TLE_DVP",
    "TLE_ZERO_COUNTS", "TLE_COUNTS_PER_PSI",
    "LEGACY_ZERO_COUNTS", "LEGACY_COUNTS_PER_PSI",
    "BoardCal", "cal_for_variant", "Recording", "WindowSet",
    "load_session", "smooth3", "TendonKinematics", "MomentArmTendonGeometry",
    "muscle_traces", "make_windows", "split_episodes", "build_windows",
    "closed_segments", "node_closed_segments", "fit_leak", "fit_leaks",
    "SyntheticTruth", "write_synthetic_session",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)

#: Pa per psi.  CONTRACT section 1; psi appears only at a human boundary.
PA_PER_PSI = 6894.757

#: The wire's pressure field is 12 bits (``tlelib.proto.PRESSURE_MASK``), so a
#: simulated ADC in the shooting loop must saturate at the same place the real
#: one does rather than running off into pressures the sensor cannot report.
ADC_MAX = 0x0FFF

N_NODES = 24
N_JOINTS = 12

VARIANT_7MM = 0x00
VARIANT_DT = 0x01
VARIANT_TLE_DVP = 0x02

# The four calibration numbers.  Taken from ``TLE_PCB/tlelib/proto.py`` when it
# imports, because that file is what the collector and the controller both use
# and a second copy that drifts is worse than no copy; the literals below are
# the fallback for a checkout without ``TLE_PCB`` on the path and are the same
# numbers ``data_schema.md`` quotes.  ``test_dataset`` asserts the two agree
# whenever the import succeeds.
TLE_ZERO_COUNTS = 943.75
TLE_COUNTS_PER_PSI = 60.78125
LEGACY_ZERO_COUNTS = 754.4
LEGACY_COUNTS_PER_PSI = 56.14

_TLE_PCB = os.path.join(_WS, "TLE_PCB")
if os.path.isdir(_TLE_PCB) and _TLE_PCB not in sys.path:
    sys.path.insert(0, _TLE_PCB)
try:  # pragma: no cover - exercised on any checkout that carries TLE_PCB
    from tlelib import proto as _proto  # type: ignore

    TLE_ZERO_COUNTS = float(_proto.TLE_ZERO_COUNTS)
    TLE_COUNTS_PER_PSI = float(_proto.TLE_COUNTS_PER_PSI)
    LEGACY_ZERO_COUNTS = float(_proto.LEGACY_ZERO_COUNTS)
    LEGACY_COUNTS_PER_PSI = float(_proto.LEGACY_COUNTS_PER_PSI)
    ADC_MAX = int(_proto.PRESSURE_MASK)
    VARIANT_7MM = int(_proto.VARIANT_7MM)
    VARIANT_DT = int(_proto.VARIANT_DT)
    VARIANT_TLE_DVP = int(_proto.VARIANT_TLE_DVP)
except Exception:  # pragma: no cover
    _proto = None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoardCal:
    """One board's counts <-> Pa transfer function, and which plant it is.

    Frozen because a calibration that can be mutated after a window set has been
    built is a calibration that can disagree with the pressures already derived
    from it.
    """

    variant: int
    zero_counts: float
    counts_per_psi: float

    @property
    def is_tle(self) -> bool:
        return self.variant == VARIANT_TLE_DVP

    @property
    def pa_per_count(self) -> float:
        return PA_PER_PSI / self.counts_per_psi

    def adc_to_pa(self, adc) -> np.ndarray:
        """Counts -> Pa gauge.  Not clipped at zero: a board reading below its
        own zero (``0x110`` reads +5.1 psi at rest, ``0x104`` +1.1) is evidence
        about the sensor, and clipping it here would hide the offset from the
        leak fit that is supposed to notice it."""
        a = np.asarray(adc, dtype=np.float64)
        return (a - self.zero_counts) / self.counts_per_psi * PA_PER_PSI

    def pa_to_adc(self, pa) -> np.ndarray:
        """Pa gauge -> counts, rounded and saturated exactly as the wire does.

        The shooting loop calls this to put the sensor back inside the rollout:
        the firmware regulates on the count it read, not on the true pressure
        (CONTRACT section 4), and a rollout that skips the quantisation regulates
        on information the real board never had.
        """
        c = np.rint(self.zero_counts + np.asarray(pa, dtype=np.float64)
                    / PA_PER_PSI * self.counts_per_psi)
        return np.clip(c, 0.0, float(ADC_MAX))


def cal_for_variant(variant: int, *, base: int | None = None) -> BoardCal:
    """The board's calibration from the byte it answered with.

    Mirrors ``tlelib.proto.NodeCal.for_variant``.  ``base`` is used only for an
    unrecognised variant, where guessing from the id range is the last resort
    rather than the policy.
    """
    v = int(variant)
    if v == VARIANT_TLE_DVP:
        return BoardCal(v, TLE_ZERO_COUNTS, TLE_COUNTS_PER_PSI)
    if v in (VARIANT_7MM, VARIANT_DT):
        return BoardCal(v, LEGACY_ZERO_COUNTS, LEGACY_COUNTS_PER_PSI)
    if base is not None and 0x101 <= int(base) <= 0x108:
        return BoardCal(VARIANT_TLE_DVP, TLE_ZERO_COUNTS, TLE_COUNTS_PER_PSI)
    return BoardCal(VARIANT_7MM, LEGACY_ZERO_COUNTS, LEGACY_COUNTS_PER_PSI)


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    """Every per-cycle stream of one session, aligned on ``can_sync_time_s``.

    ``can_sync_time_s`` rather than ``timestamp_s`` because the sync edge is the
    instant every board latches its filtered pressure and promotes its staged
    target -- the one moment the whole arm agrees on (``data_schema.md``).
    """

    session_dir: str
    ids: np.ndarray             # (24,) int, CAN base ids
    board_type: np.ndarray      # (24,) int, the recorded variant byte
    cals: tuple                 # (24,) BoardCal, one per node
    cycle: np.ndarray           # (N,) int64
    can_sync_time_s: np.ndarray  # (N,) float64
    pressure_adc: np.ndarray    # (N,24) int32, raw
    target_adc: np.ndarray      # (N,24) int32, raw
    pressure_pa: np.ndarray     # (N,24) float64
    target_pa: np.ndarray       # (N,24) float64
    q: np.ndarray               # (N,12) rad
    qdot: np.ndarray            # (N,12) rad/s
    q_valid: np.ndarray         # (N,) bool
    episode: np.ndarray         # (N,) int32
    #: ``(N,)`` str -- the excitation family each cycle belongs to, from the
    #: recording's ``segment_kind``.  Kept because a holdout that reserves a
    #: whole family has to be able to name one.
    phase: np.ndarray = None
    #: ``(N,24)`` bool -- which board answered which sync edge.  False entries
    #: have had their pressure carried forward from the last answered reading;
    #: the raw file recorded them as zero counts, which is -15.53 psi on a TLE
    #: board and -13.44 on a 7 mm one, and fitting those is the failure this
    #: field exists to prevent.
    pressure_valid: np.ndarray = None
    meta: dict = None

    @property
    def n_cycles(self) -> int:
        return int(self.can_sync_time_s.shape[0])

    @property
    def is_tle(self) -> np.ndarray:
        """(24,) bool, from the recorded variant byte -- CONTRACT departure 2."""
        return self.board_type == VARIANT_TLE_DVP

    @property
    def zero_counts(self) -> np.ndarray:
        return np.array([c.zero_counts for c in self.cals], dtype=np.float64)

    @property
    def counts_per_psi(self) -> np.ndarray:
        return np.array([c.counts_per_psi for c in self.cals], dtype=np.float64)

    def episode_ids(self) -> np.ndarray:
        return np.unique(self.episode)

    def episode_slices(self) -> list:
        """Contiguous ``slice`` per episode, in recording order.

        Slices rather than index arrays because every consumer here differentiates
        or smooths along time, and a fancy-indexed copy silently permits a
        non-contiguous "episode" that would smooth across a gap.
        """
        out = []
        ep = self.episode
        if ep.size == 0:
            return out
        start = 0
        for i in range(1, ep.size + 1):
            if i == ep.size or ep[i] != ep[start]:
                out.append(slice(start, i))
                start = i
        return out


def _chunk_paths(session_dir: str) -> list:
    """Chunk files in recording order, preferring the manifest over the glob.

    The manifest is authoritative because it carries ``start_cycle`` and the
    ``reason`` tag; the glob is the fallback for a session whose collector died
    before writing the final manifest, which ``data_schema.md`` says leaves
    ``active`` true and the last chunk short.
    """
    man = os.path.join(session_dir, "manifest.json")
    if os.path.isfile(man):
        with open(man, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        chunks = m.get("chunks") or []
        if chunks:
            paths = []
            for c in chunks:
                p = c["path"]
                paths.append(p if os.path.isabs(p) else os.path.join(session_dir, p))
            return paths
    return sorted(glob.glob(os.path.join(session_dir, "samples_chunk_*.jsonl")))


def _as_int_ids(raw: Sequence) -> np.ndarray:
    out = []
    for v in raw:
        out.append(int(v, 16) if isinstance(v, str) else int(v))
    return np.asarray(out, dtype=np.int32)


#: Boards observed reading a non-zero pressure at rest on 2026-08-21
#: (``UMArm_KINEMATICS.canarm_actuators.KNOWN_BOARD_FAULTS``): ``0x110`` leaks
#: from its supply side and reads +5.1 psi, ``0x104`` reads +1.1 psi.
#: ``replay.load_recording`` subtracts these by default and this reader does
#: **not** -- see :func:`load_session`'s ``rest_offset_psi``.
KNOWN_REST_OFFSET_PSI = {0x104: 1.1, 0x110: 5.1}


def load_session(session_dir,
                 *,
                 max_gap_s: float = 0.05,
                 episode_field: str = "episode",
                 phase_splits_episode: bool = True,
                 rest_offset_psi=None,
                 n_nodes: int = N_NODES) -> Recording:
    """Read one ``session_*/`` directory into a :class:`Recording`.

    ``rest_offset_psi`` defaults to ``None``, i.e. **no rest-offset correction**,
    and that is a deliberate disagreement with ``replay.load_recording``, which
    subtracts :data:`KNOWN_REST_OFFSET_PSI` by default.  The two readers want
    different coordinates and both are right in their own place.  ``replay``
    scores the twin against physical pressure, where a board reading +5.1 psi
    with nothing in it is a sensor fault to remove.  Training wants the number
    the *firmware* acted on: both regulators compute their error from the board's
    own reported pressure, offset included, so a net fitted on offset-corrected
    pressure is fitted against an error the board never saw.  Pass
    ``KNOWN_REST_OFFSET_PSI`` here to line the two up when that is what a caller
    needs, and note that doing so changes what ``e`` means on two of the
    twenty-four boards.

    ``max_gap_s`` is the sync-edge gap that starts a new episode.  Nominal cycle
    time at 150 Hz is 6.667 ms and the flagship legacy hour-long run reports a
    worst interval of 25.069 ms with zero missing cycles (``data_schema.md``), so
    50 ms is twice the worst jitter ever recorded on this format: it cannot split
    an episode on scheduler slip alone, and it does split on a dropped run of
    seven or more cycles, which is a discontinuity the pressure state does not
    survive.

    ``phase_splits_episode`` keeps the deflate tail out of the same episode as the
    collection it follows, since the two are different excitation regimes and a
    window that straddles the transition trains on a target discontinuity that is
    an artefact of the collector rather than of the plant.

    ``episode_field`` is honoured when the collector writes one: an explicit
    episode label from the thing that generated the excitation beats any gap
    heuristic, and the heuristic is only the fallback.
    """
    session_dir = os.fspath(session_dir)
    meta_path = os.path.join(session_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"no metadata.json in {session_dir!r}; a session directory must carry "
            "metadata.json, manifest.json and its samples_chunk_*.jsonl "
            "(digital_twin/data_schema.md)")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ids = _as_int_ids(meta.get("selected_ids") or range(0x101, 0x101 + n_nodes))
    board_type = meta.get("board_type")
    if board_type is None:
        raise ValueError(
            f"{meta_path} carries no 'board_type'. The population split is not "
            "recoverable from the id range -- a TLE board sat at 0x114 during the "
            "2026-08 bench session -- so a session without it cannot be read "
            "(digital_twin/data_schema.md).")
    board_type = np.asarray(board_type, dtype=np.int32)
    if board_type.shape != (n_nodes,) or ids.shape != (n_nodes,):
        raise ValueError(
            f"metadata must carry {n_nodes} ids and {n_nodes} board_type entries; "
            f"got {ids.shape} and {board_type.shape}")
    cals = tuple(cal_for_variant(int(v), base=int(b)) for v, b in zip(board_type, ids))

    paths = _chunk_paths(session_dir)
    if not paths:
        raise FileNotFoundError(f"no samples_chunk_*.jsonl under {session_dir!r}")

    cyc, tsync, padc, tadc, qq, qd, ok, ph, epf = [], [], [], [], [], [], [], [], []
    rep_ok = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                st = rec["robot_state"]
                cyc.append(int(rec["cycle"]))
                tsync.append(float(rec["can_sync_time_s"]))
                padc.append(st["pressure_adc"])
                tadc.append(rec["input"]["target_adc"])
                # Schema 1 put q and qdot in robot_state; schema 2 moved them
                # into the mocap block, because that is where their age and
                # validity live and separating a value from its age is how a
                # stale pose gets fitted as a fresh one.  Either is read here;
                # neither is preferred to mocap_stream.jsonl, which is
                # resampled over the top below when the session has one.
                mc = rec.get("mocap") or {}
                qq.append(st.get("q") or mc.get("q") or [0.0] * N_JOINTS)
                qd.append(st.get("qdot") or mc.get("qdot") or [0.0] * N_JOINTS)
                ph.append(rec.get("phase")
                          or rec.get("segment_kind", "collecting"))
                # Which boards actually answered THIS sync edge.  A collector
                # fills the pressure row with zeros and overwrites the boards
                # that replied, so a missed reply is recorded as a reading of
                # zero counts -- which converts to -15.53 psi on a TLE board
                # and -13.44 on a 7 mm one, identically every time, in a data
                # set whose real range is 0.4 to 25.6 psi.  The reply-latency
                # column is null exactly where no reply came, so the file knows;
                # nothing but the reader was using it.
                lat = rec.get("actuator_reply_latency_ms")
                rep_ok.append([v is not None for v in lat] if lat
                              else [True] * n_nodes)
                epf.append(rec.get(episode_field))
                ok.append(_q_usable(rec))

    if not cyc:
        raise ValueError(f"{session_dir!r} contains no sample rows")

    cycle = np.asarray(cyc, dtype=np.int64)
    t = np.asarray(tsync, dtype=np.float64)
    pressure_adc = np.asarray(padc, dtype=np.int32)
    target_adc = np.asarray(tadc, dtype=np.int32)
    q = np.asarray(qq, dtype=np.float64)
    qdot = np.asarray(qd, dtype=np.float64)
    q_valid = np.asarray(ok, dtype=bool)
    replied = np.asarray(rep_ok, dtype=bool)
    phase = np.asarray(ph, dtype=object)
    explicit = np.asarray([-1 if v is None else int(v) for v in epf], dtype=np.int64)

    order = np.argsort(cycle, kind="stable")
    cycle, t = cycle[order], t[order]
    pressure_adc, target_adc = pressure_adc[order], target_adc[order]
    q, qdot, q_valid = q[order], qdot[order], q_valid[order]
    replied = replied[order]
    phase, explicit = phase[order], explicit[order]

    # Carry the last answered reading forward across a missed reply rather than
    # dropping the cycle.  One miss at 150 Hz leaves the pressure 6.7 ms stale,
    # which is far inside the sensor's own noise, whereas dropping the cycle
    # would punch a hole in every OTHER board's trace for one board's silence.
    # The mask is kept so a fit can still exclude them if it wants to.
    n_missed = int((~replied).sum())
    if n_missed:
        for j in range(n_nodes):
            col = replied[:, j]
            if col.all():
                continue
            idx = np.where(col, np.arange(col.size), 0)
            np.maximum.accumulate(idx, out=idx)
            pressure_adc[:, j] = pressure_adc[idx, j]
            # A leading run of misses has nothing to carry forward; the first
            # answered sample is the honest substitute and the mask says so.
            first = int(np.argmax(col)) if col.any() else 0
            pressure_adc[:first, j] = pressure_adc[first, j]

    if pressure_adc.shape[1] != n_nodes or target_adc.shape[1] != n_nodes:
        raise ValueError("pressure_adc/target_adc must be 24 wide per cycle")
    if q.shape[1] != N_JOINTS or qdot.shape[1] != N_JOINTS:
        raise ValueError("q/qdot must be 12 wide per cycle")

    _WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _WS_ROOT not in sys.path:
        sys.path.insert(0, _WS_ROOT)

    # Convert with each board's own calibration.  Column by column, because the
    # whole point of departure 2 is that the twenty-four columns do not share a
    # transfer function.
    pressure_pa = np.empty_like(pressure_adc, dtype=np.float64)
    target_pa = np.empty_like(target_adc, dtype=np.float64)
    offsets = dict(rest_offset_psi or {})
    for j, cal in enumerate(cals):
        off = float(offsets.get(int(ids[j]), 0.0)) * PA_PER_PSI
        pressure_pa[:, j] = cal.adc_to_pa(pressure_adc[:, j]) - off
        # The target is not offset-corrected even when the measurement is: the
        # setpoint is a number the host chose, not a number the sensor reported.
        target_pa[:, j] = cal.adc_to_pa(target_adc[:, j])

    # --- the mocap stream, if this session kept one ---------------------- #
    #
    # The per-cycle block holds the newest frame that had ARRIVED by the sync
    # edge.  Motive streams at 120 Hz against a 150 Hz cycle, so one cycle in
    # five repeats the previous frame, and a q built from those repeats
    # differentiates into a staircase: measured on a live rest recording, the
    # held series' qdot ran 1.8x the interpolated one on an arm that was not
    # moving.  Where the raw stream was recorded, it is resampled onto the sync
    # edges instead -- and a sync edge the stream does not cover is marked
    # unusable rather than filled, because an extrapolated pose across a Motive
    # dropout is a pose nobody measured.
    stream_info = None
    if os.path.exists(os.path.join(session_dir, "mocap_stream.jsonl")):
        try:
            from collection import resample as _rs
            stream_info = _rs.resample_session(session_dir, t, meta)
            q = stream_info["q"]
            qdot = stream_info["qdot"]
            q_valid = q_valid & stream_info["valid"]
        except Exception as exc:                      # pragma: no cover
            stream_info = {"error": repr(exc)}
    meta = dict(meta)
    meta["missed_replies"] = {
        "count": n_missed,
        "fraction": float(n_missed) / float(replied.size),
        "note": ("filled forward from the last answered reading; "
                 "Recording.pressure_valid marks them"),
    }
    meta["mocap_resample"] = (
        {k: v for k, v in stream_info.items()
         if k not in ("q", "qdot", "valid", "gap_s")}
        if stream_info else None)

    episode = _episode_labels(cycle, t, phase, explicit,
                              max_gap_s=max_gap_s,
                              phase_splits_episode=phase_splits_episode)

    return Recording(session_dir=session_dir, ids=ids, board_type=board_type,
                     cals=cals, cycle=cycle, can_sync_time_s=t,
                     pressure_adc=pressure_adc, target_adc=target_adc,
                     pressure_pa=pressure_pa, target_pa=target_pa,
                     q=q, qdot=qdot, q_valid=q_valid, episode=episode,
                     phase=phase, pressure_valid=replied, meta=meta)


def _q_usable(rec: dict) -> bool:
    """Whether this cycle's joints may be believed.

    ``data_schema.md`` is explicit that ``mocap.stale`` (no frame at all) and
    ``q_stale`` (frames arriving, none converting) are different questions and
    that the second is the one that invalidates ``q`` while ``mocap.valid`` can
    still read true.  Prefer ``q_stale`` where the collector writes it; fall back
    to ``joint_current_valid``; treat a row that says nothing as usable, because
    the older chunks in the legacy format carry none of these fields and refusing
    them would discard the only long recording that exists.
    """
    mocap = rec.get("mocap") or {}
    if "q_stale" in rec:
        return not bool(rec["q_stale"])
    if "q_stale" in mocap:
        return not bool(mocap["q_stale"])
    if "joint_current_valid" in rec:
        return bool(rec["joint_current_valid"])
    if "valid" in mocap:
        return bool(mocap["valid"]) and not bool(mocap.get("stale", False))
    return True


def _episode_labels(cycle, t, phase, explicit, *, max_gap_s, phase_splits_episode):
    n = cycle.size
    lab = np.zeros(n, dtype=np.int32)
    cur = 0
    for i in range(1, n):
        new = False
        if explicit[i] >= 0 or explicit[i - 1] >= 0:
            new = explicit[i] != explicit[i - 1]
        else:
            if cycle[i] != cycle[i - 1] + 1:
                new = True
            if t[i] - t[i - 1] > max_gap_s or t[i] <= t[i - 1]:
                new = True
        if phase_splits_episode and phase[i] != phase[i - 1]:
            new = True
        if new:
            cur += 1
        lab[i] = cur
    return lab


# ---------------------------------------------------------------------------
# Tendon geometry
# ---------------------------------------------------------------------------
def smooth3(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Centred 3-sample mean, edges untouched -- the reference's ``smooth3``.

    Kept as a 3-tap mean rather than anything better-behaved because the whole
    reference pipeline (warm-start midpoints, ``ldot``) was fitted through this
    exact filter, and changing the pre-filter changes the input distribution the
    net sees without changing anything that would announce it.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.shape[axis] < 3:
        return a.copy()
    out = a.copy()
    sl = [slice(None)] * a.ndim
    lo, mid, hi = list(sl), list(sl), list(sl)
    lo[axis] = slice(0, -2)
    mid[axis] = slice(1, -1)
    hi[axis] = slice(2, None)
    out[tuple(mid)] = (a[tuple(lo)] + a[tuple(mid)] + a[tuple(hi)]) / 3.0
    return out


class MomentArmTendonGeometry:
    """First-order moment-arm stand-in for the twin's real tendon routing.

    WHAT IT IS.  Actuator ``a`` drives joint ``j`` with sign ``s`` per
    ``UMArm_KINEMATICS.canarm_actuators.MEASURED_JOINT_PAIRS``; seated at radius
    ``ring_radius_m`` on its plate, its length changes to first order by
    ``dlen = -s * r * q[j]``.  Units, sign and the actuator/joint coupling are
    right, which is enough for the whole dataset/training pipeline to be exercised
    end to end offline.

    WHAT IT IS NOT.  It is not the twin's geometry.  It carries no plate offset,
    no second-order term, and no coupling between the two axes of one universal
    joint, so an ``l`` computed from it is not the ``l`` the force law will
    consume once ``digital_twin.mjcf_generator`` emits tendons.  A net trained on
    it is a net trained on the wrong input distribution, which the CONTRACT's
    unit guard cannot catch because the normalisation constants are unchanged.
    Use it to test the pipeline, and re-derive ``l`` before believing a fit.

    ``ring_radius_m`` defaults to 0.030 m, the anchor-ring radius the force audit
    cites for this arm (four muscles to a 30 mm ring); it is an argument rather
    than a constant so the placeholder cannot quietly become a fitted number.
    """

    is_placeholder = True

    def __init__(self, *, ring_radius_m: float = 0.030,
                 joint_pairs: Sequence | None = None,
                 n_nodes: int = N_NODES):
        if joint_pairs is None:
            joint_pairs = _measured_joint_pairs()
        self.ring_radius_m = float(ring_radius_m)
        self.n_nodes = int(n_nodes)
        # node index (0-based, node k = board 0x100+k+1) -> (joint, sign)
        self._joint = np.zeros(self.n_nodes, dtype=np.int32)
        self._sign = np.zeros(self.n_nodes, dtype=np.float64)
        for j, (pos, neg) in enumerate(joint_pairs):
            self._joint[int(pos) - 0x101] = j
            self._sign[int(pos) - 0x101] = +1.0
            self._joint[int(neg) - 0x101] = j
            self._sign[int(neg) - 0x101] = -1.0
        self.len0 = np.zeros(self.n_nodes, dtype=np.float64)

    def dlen(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return -self.ring_radius_m * self._sign[None, :] * q[:, self._joint]


def _measured_joint_pairs():
    if _WS not in sys.path:
        sys.path.insert(0, _WS)
    from UMArm_KINEMATICS import canarm_actuators as CA  # type: ignore

    return CA.joint_pairs()


class TendonKinematics:
    """``q`` -> per-actuator tendon excursion, through the twin's own MuJoCo model.

    Built from ``mjcf_generator.generate_xml()`` so that the ``l`` the net is
    trained on is produced by the same geometry ``sim_core`` will step.  Actuator
    ``pam_k`` is board ``0x100 + k`` (CONTRACT section 3); the tendon behind it is
    ``model.actuator_trnid[a, 0]`` and its reference length is
    ``model.tendon_length0[t]``, which is what makes the excursion *anchored* --
    ``l = l0_seg + (ten_length - tendon_length0)`` -- rather than absolute.

    ``mj_kinematics -> mj_comPos -> mj_tendon`` is the minimum pipeline that
    fills ``data.ten_length``; ``mj_forward`` would also solve the dynamics, which
    costs about an order of magnitude more per sample for a number that does not
    depend on it.
    """

    is_placeholder = False

    def __init__(self, xml: str, *, n_nodes: int = N_NODES):
        import mujoco  # imported here so a numpy-only consumer never pays for it

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.n_nodes = int(n_nodes)
        ten, len0 = [], []
        for k in range(1, self.n_nodes + 1):
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pam_{k}")
            if a < 0:
                raise ValueError(
                    f"the twin's MJCF has no actuator 'pam_{k}'; CONTRACT section 3 "
                    "requires pam_1..pam_24 with pam_k = board 0x100+k")
            t = int(self.model.actuator_trnid[a, 0])
            ten.append(t)
            len0.append(float(self.model.tendon_length0[t]))
        self.ten_ids = np.asarray(ten, dtype=np.int32)
        self.len0 = np.asarray(len0, dtype=np.float64)
        #: ``(12,)`` qpos address of ``q[i]``, resolved by hinge name, or
        #: ``None`` when the scene lacks the CAN arm's names.  See :meth:`dlen`.
        self.q_qposadr = None
        try:
            from . import mjcf_generator as _mg
        except ImportError:  # pragma: no cover
            _mg = None
        if _mg is not None:
            names = _mg.joint_names()
            found = []
            for i in range(len(_mg.QPOS_FROM_Q)):
                j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                      names[_mg.QPOS_FROM_Q[i]])
                if j < 0:
                    break
                found.append(int(self.model.jnt_qposadr[j]))
            else:
                self.q_qposadr = np.asarray(found, dtype=np.int64)

    @classmethod
    def auto(cls, *, xml: str | None = None, allow_placeholder: bool = True,
             ring_radius_m: float = 0.030):
        """The twin's geometry if it exists, the placeholder if it does not.

        ``digital_twin.mjcf_generator`` is a documented skeleton at the time of
        writing and raises ``NotImplementedError``; rather than block the whole
        fitting pipeline on it, this falls back to
        :class:`MomentArmTendonGeometry` and marks the result
        ``is_placeholder``.  Every consumer here propagates that flag into the
        checkpoint metadata, so a fit made without real tendon geometry says so
        on its own face instead of being indistinguishable from one made with it.
        """
        if xml is None:
            try:
                from . import mjcf_generator  # type: ignore
            except ImportError:  # pragma: no cover
                import mjcf_generator  # type: ignore
            try:
                xml = mjcf_generator.generate_xml()
            except NotImplementedError:
                if not allow_placeholder:
                    raise
                return MomentArmTendonGeometry(ring_radius_m=ring_radius_m)
        try:
            return cls(xml)
        except Exception:
            if not allow_placeholder:
                raise
            return MomentArmTendonGeometry(ring_radius_m=ring_radius_m)

    def dlen(self, q: np.ndarray) -> np.ndarray:
        """``(N, 24)`` excursion from ``(N, 12)`` joint angles in ``q`` order.

        ``q`` is written to the hinges by NAME.  Until 2026-09-10 it was written
        straight into ``qpos``, which on this arm swaps each proximal pair
        (``mjcf_generator.QPOS_FROM_Q``), so the twelve proximal muscles' ``l``
        and ``ldot`` came from the other axis of their u-joint.  The flow
        checkpoint of 2026-09-10 was trained on those features.  Over its own
        10 824 held-out windows it scores 3681.9 Pa under them and 3676.2 Pa
        under the corrected ones (-0.15 %), so it was kept, not retrained.
        """
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        nq = self.model.nq
        adr = self.q_qposadr
        out = np.empty((q.shape[0], self.n_nodes), dtype=np.float64)
        for i in range(q.shape[0]):
            if adr is None:
                self.data.qpos[:nq] = q[i, :nq]
            else:
                self.data.qpos[adr] = q[i, :adr.size]
            self._mj.mj_kinematics(self.model, self.data)
            self._mj.mj_comPos(self.model, self.data)
            self._mj.mj_tendon(self.model, self.data)
            out[i] = self.data.ten_length[self.ten_ids] - self.len0
        return out


#: Per-segment rest length ``l0``, m.  ``actuator_model.L0_SEED_M``, which is
#: this arm's measured ``LL`` column -- the free span between the two actuator
#: attachment points at zero joint angle, from the five plate gaps measured over
#: 68 poses on 2026-08-21.  Taken from that module rather than restated, because
#: ``l0`` is simultaneously a force-law parameter and the offset of the net's
#: ``l`` input: if the outer fit moves it, every ``l`` here must be rebuilt
#: before the net is evaluated, and two copies make that impossible to enforce.
DEFAULT_L0_PER_SEGMENT_M = _am.L0_SEED_M


def l0_per_actuator(l0_per_segment=DEFAULT_L0_PER_SEGMENT_M,
                    *, n_nodes: int = N_NODES) -> np.ndarray:
    """Expand three segment rest lengths to twenty-four actuators.

    The expansion is ``np.repeat(x, 8)`` because segment ``s`` owns boards
    ``0x101+8s .. 0x108+8s``.  That segment 0 coincides with the TLE population is
    a coincidence of this arm's wiring and not a rule: the population split is
    read from the variant byte and lives in a different index vector.
    """
    return np.repeat(np.asarray(l0_per_segment, dtype=np.float64),
                     n_nodes // len(l0_per_segment))


def muscle_traces(rec: Recording,
                  *,
                  tendon=None,
                  l0_per_segment=DEFAULT_L0_PER_SEGMENT_M,
                  smooth: bool = True):
    """Anchored muscle length and rate for every cycle and every actuator.

    Returns ``(l, ldot)``, both ``(N, 24)`` in m and m/s.  Both the smoothing and
    the differentiation are done **per episode**: a centred mean or an
    ``np.gradient`` that reaches across an episode boundary mixes two recordings
    separated by an unknown amount of wall clock, and at 150 Hz that shows up as a
    single enormous ``ldot`` sample rather than as anything a filter would reject.
    """
    if tendon is None:
        tendon = TendonKinematics.auto()
    l0 = l0_per_actuator(l0_per_segment, n_nodes=rec.pressure_pa.shape[1])
    l = np.empty((rec.n_cycles, l0.size), dtype=np.float64)
    ldot = np.empty_like(l)
    for sl in rec.episode_slices():
        seg = l0[None, :] + tendon.dlen(rec.q[sl])
        if smooth:
            seg = smooth3(seg, axis=0)
        t = rec.can_sync_time_s[sl]
        l[sl] = seg
        ldot[sl] = (np.gradient(seg, t, axis=0) if t.size >= 2
                    else np.zeros_like(seg))
    return l, ldot


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
@dataclass
class WindowSet:
    """Multiple-shooting windows: one row per (episode, node, start).

    Every row is a self-contained shooting problem re-anchored at its own recorded
    ``p0``.  That re-anchoring is what keeps a 1200-cycle recording from being one
    8-second-error-accumulating rollout, and it is why a window may not cross an
    episode boundary -- the anchor would then be a pressure from a different
    recording.
    """

    node_idx: np.ndarray        # (B,) 0-based node index, node k-1 = board 0x100+k
    episode: np.ndarray         # (B,) episode label the window came from
    is_tle: np.ndarray          # (B,) bool, from the recorded variant byte
    t: np.ndarray               # (B,K) s, the recorded sync edges
    p_rec: np.ndarray           # (B,K) Pa, the loss target
    target_pa: np.ndarray       # (B,K) Pa, the commanded setpoint
    l: np.ndarray               # (B,K) m
    ldot: np.ndarray            # (B,K) m/s
    zero_counts: np.ndarray     # (B,) that board's own cal
    counts_per_psi: np.ndarray  # (B,)

    def __len__(self) -> int:
        return int(self.node_idx.shape[0])

    @property
    def n_ticks(self) -> int:
        return int(self.t.shape[1])

    @property
    def p0(self) -> np.ndarray:
        return self.p_rec[:, 0].copy()

    def select(self, mask) -> "WindowSet":
        """Rows by boolean mask, index array or slice.  Always a new WindowSet."""
        m = mask if isinstance(mask, slice) else np.asarray(mask)
        return WindowSet(**{k: v[m] for k, v in self.__dict__.items()})

    def population(self, is_tle: bool) -> "WindowSet":
        """The windows of one plant.  CONTRACT departure 2: anything reported
        across all twenty-four boards describes neither population."""
        return self.select(self.is_tle == bool(is_tle))


def make_windows(rec: Recording,
                 l: np.ndarray,
                 ldot: np.ndarray,
                 *,
                 window_cycles: int = 300,
                 stride: int | None = None,
                 nodes: Iterable[int] | None = None,
                 require_q_valid: bool = True,
                 flush_tail: bool = True) -> WindowSet:
    """Cut every episode into equal-length shooting windows.

    ``window_cycles`` defaults to 300, which at the CAN arm's 150 Hz sync rate is
    2.0 s.  The reference used 160 ticks at 20 Hz -- an 8 s horizon -- and 8 s here
    would be 1200 cycles, seven substeps each: 8400 tape entries per window, which
    at a few thousand windows does not fit a 32 GB card.  2 s is the compromise,
    and it is a real cost, not a free one: a 2 s horizon constrains the slow limb
    of the fill (the reference measured tau_slow of order 18 s on the RS485 rig)
    only through the anchored start of the next window.  Raise it, and lower
    ``chunk`` in the trainer, when the slow limb is what is being fitted.

    ``stride`` defaults to ``window_cycles``, i.e. non-overlapping, plus one
    trailing window ending exactly at the episode end when ``flush_tail``.  The
    overlap that tail creates reweights a few cycles and is accepted as harmless,
    which is the reference's own choice.

    ``require_q_valid`` drops any window containing a cycle whose joints could not
    be believed, rather than interpolating over it: ``l`` and ``ldot`` are net
    inputs and an interpolated plate pose is a fabricated muscle length.
    """
    kw = int(window_cycles)
    if kw < 2:
        raise ValueError("window_cycles must be at least 2 -- a shooting window "
                         "with one tick has no residual")
    step = int(stride) if stride else kw
    if step < 1:
        raise ValueError("stride must be positive")
    node_list = (list(range(rec.pressure_pa.shape[1])) if nodes is None
                 else [int(n) for n in nodes])

    starts_by_ep = []
    for sl in rec.episode_slices():
        n = sl.stop - sl.start
        if n < kw:
            continue
        st = list(range(sl.start, sl.stop - kw + 1, step))
        tail = sl.stop - kw
        if flush_tail and st and st[-1] != tail:
            st.append(tail)
        if require_q_valid:
            st = [s for s in st if rec.q_valid[s:s + kw].all()]
        if st:
            starts_by_ep.append((int(rec.episode[sl.start]), st))

    if not starts_by_ep:
        raise ValueError(
            f"no episode in {rec.session_dir!r} is {kw} cycles long with usable "
            f"joints; the longest is "
            f"{max((s.stop - s.start for s in rec.episode_slices()), default=0)} "
            "cycles. Lower window_cycles or record longer episodes.")

    rows_ep, rows_start = [], []
    for ep, st in starts_by_ep:
        rows_ep.extend([ep] * len(st))
        rows_start.extend(st)
    starts = np.asarray(rows_start, dtype=np.int64)
    eps = np.asarray(rows_ep, dtype=np.int32)
    idx = starts[:, None] + np.arange(kw, dtype=np.int64)[None, :]   # (S,K)

    n_starts = starts.size
    n_sel = len(node_list)
    node_idx = np.repeat(np.asarray(node_list, dtype=np.int32), n_starts)
    episode = np.tile(eps, n_sel)
    t = np.tile(rec.can_sync_time_s[idx], (n_sel, 1))

    def gather(a):
        return np.concatenate([a[idx, j] for j in node_list], axis=0)

    return WindowSet(
        node_idx=node_idx,
        episode=episode,
        is_tle=rec.is_tle[node_idx],
        t=t,
        p_rec=gather(rec.pressure_pa),
        target_pa=gather(rec.target_pa),
        l=gather(l),
        ldot=gather(ldot),
        zero_counts=rec.zero_counts[node_idx],
        counts_per_psi=rec.counts_per_psi[node_idx],
    )


def split_episodes(windows: WindowSet,
                   *,
                   holdout_frac: float = 0.25,
                   seed: int = 20260910,
                   min_train_episodes: int = 1,
                   reserved=None):
    """Split whole episodes into train and holdout.

    By episode and never by sample.  At 150 Hz adjacent samples are 6.67 ms apart
    and are not independent, so a random per-sample split reports a holdout error
    that is essentially the training error -- a fantasy number that would make the
    guarded selection in the trainer accept anything.

    The permutation is seeded so the same session and seed give the same split,
    which is what lets two fits be compared at all.  With a single episode the
    holdout is empty and this raises rather than inventing one: a fit with no
    holdout is a fit with no evidence, and it should be an explicit choice to run
    one.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must lie strictly inside (0, 1)")
    eps = np.unique(windows.episode)
    if eps.size < 2:
        raise ValueError(
            f"only {eps.size} episode(s) in this window set; a whole-episode "
            "holdout needs at least two. Record more episodes rather than "
            "falling back to a per-sample split.")
    reserved = set(int(e) for e in (reserved or ()))
    free = np.array([e for e in eps if int(e) not in reserved])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(free) if free.size else free
    # Reserved episodes count toward the holdout rather than adding to it: the
    # fraction is a statement about how much data is withheld, and a family
    # reserved on top of a full random split withholds more than was asked.
    want = max(1, int(round(holdout_frac * eps.size)))
    n_extra = min(max(free.size - min_train_episodes, 0),
                  max(want - len(reserved), 0))
    hold = reserved | set(int(e) for e in perm[:n_extra])
    if not hold:
        hold = {int(perm[0])}
    m = np.array([int(e) in hold for e in windows.episode], dtype=bool)
    if m.all():
        raise ValueError(
            "every episode landed in the holdout; lower holdout_frac or "
            "record something outside the reserved families")
    return windows.select(~m), windows.select(m)


#: Excitation families reserved entirely for the holdout when a session has
#: them.  ``validation`` is generated from a different seed AND a different
#: signal family than anything trained on -- smooth multi-joint sinusoids rather
#: than drawn steps -- because a holdout drawn from the training distribution
#: measures interpolation and reports it as generalisation.
DEFAULT_HOLDOUT_KINDS = ("validation",)

#: What ``episode`` means by default.  ``segment_index`` gives one episode per
#: excitation segment -- 218 of them in the 2026-09-10 session -- where the
#: fallback of grouping by ``segment_kind`` gives six, and a 25 % split of six
#: groups holds out whole excitation families by accident rather than by design.
DEFAULT_EPISODE_FIELD = "segment_index"


def build_windows(session_dir,
                  *,
                  window_cycles: int = 300,
                  stride: int | None = None,
                  holdout_frac: float = 0.25,
                  seed: int = 20260910,
                  tendon=None,
                  nodes: Iterable[int] | None = None,
                  max_gap_s: float = 0.05,
                  episode_field: str = DEFAULT_EPISODE_FIELD,
                  holdout_kinds=DEFAULT_HOLDOUT_KINDS,
                  log=None):
    """One or more session directories -> ``(train, holdout, recording, l, ldot)``.

    One call so that the trainer, the tests and any later analysis all take the
    same path from file to windows; a second, subtly different reader is how the
    training inputs and the evaluation inputs drift apart.

    ``session_dir`` may be a list.  It is one, in practice, because a USB-CDC
    write failure cut the 2026-09-10 campaign in half and the second half was
    collected as its own session; concatenating the *windows* rather than the
    files is what keeps each session's own clock, calibration and episode
    numbering intact -- two recordings whose ``can_sync_time_s`` both start at
    zero cannot be concatenated as rows.

    ``holdout_kinds`` names excitation families that go to the holdout whole.
    See :data:`DEFAULT_HOLDOUT_KINDS` for why that is not the same as a larger
    random split.
    """
    dirs = ([session_dir] if isinstance(session_dir, (str, bytes, os.PathLike))
            else list(session_dir))
    if tendon is None:
        tendon = TendonKinematics.auto()

    trains, holds, recs = [], [], []
    ep_base = 0
    for d in dirs:
        rec = load_session(d, max_gap_s=max_gap_s, episode_field=episode_field)
        l, ldot = muscle_traces(rec, tendon=tendon)
        win = make_windows(rec, l, ldot, window_cycles=window_cycles,
                           stride=stride, nodes=nodes)
        # Episode ids are per-session; offset them so two sessions' episode 3
        # are not treated as the same episode by the split.
        win = _offset_episodes(win, ep_base)
        reserved = _kind_episodes(rec, holdout_kinds, ep_base)
        tr, ho = split_episodes(win, holdout_frac=holdout_frac, seed=seed,
                                reserved=reserved)
        if log is not None:
            log(f"  {os.path.basename(str(d))}: {rec.n_cycles} cycles, "
                f"{rec.episode_ids().size} episodes, {len(tr)} train / "
                f"{len(ho)} holdout windows"
                + (f", {len(reserved)} episode(s) reserved as "
                   f"{'/'.join(holdout_kinds)}" if reserved else ""))
        trains.append(tr)
        holds.append(ho)
        recs.append((rec, l, ldot))
        ep_base += int(rec.episode.max()) + 1 if rec.n_cycles else 0

    train = _concat_windows(trains)
    hold = _concat_windows(holds)
    rec, l, ldot = recs[0]
    if len(recs) > 1:
        rec.meta = dict(rec.meta)
        rec.meta["sessions"] = [str(d) for d in dirs]
        rec.meta["cycles_per_session"] = [r.n_cycles for r, _, _ in recs]
    return train, hold, rec, l, ldot


def _kind_episodes(rec, kinds, offset: int) -> set:
    """Episode ids whose rows carry one of ``kinds`` in ``segment_kind``."""
    if not kinds:
        return set()
    want = set(kinds)
    out = set()
    labels = getattr(rec, "phase", None)
    if labels is None:
        return out
    for ep in np.unique(rec.episode):
        m = rec.episode == ep
        vals = set(str(v) for v in np.unique(np.asarray(labels)[m]))
        if vals & want:
            out.add(int(ep) + offset)
    return out


def _offset_episodes(win, offset: int):
    if not offset:
        return win
    import dataclasses
    return dataclasses.replace(win, episode=win.episode + np.int32(offset))


def _concat_windows(sets):
    sets = [w for w in sets if len(w)]
    if not sets:
        raise ValueError("no windows survived the split")
    if len(sets) == 1:
        return sets[0]
    import dataclasses
    fields = {}
    for f in dataclasses.fields(sets[0]):
        vals = [getattr(w, f.name) for w in sets]
        fields[f.name] = np.concatenate(vals, axis=0)
    return dataclasses.replace(sets[0], **fields)


# ---------------------------------------------------------------------------
# Leak fit -- least squares outside the net, at the plant seam
# ---------------------------------------------------------------------------
#: Consecutive-sample noise scale on the reported pressure, Pa.  One legacy count
#: is 6894.757/56.14 = 122.8 Pa and one TLE count is 113.4 Pa; the reference
#: measured 1.7-2.7 counts of sample-to-sample spread and multiplied by sqrt(2)
#: for a difference of two samples.  400 Pa is that figure on this arm's coarser
#: legacy count, and it is an argument everywhere it is used, not a constant.
LEAK_SIGMA_PA = 400.0

#: A fitted leak above this is not a leak, Pa/s.  CONTRACT section 2 clips
#: ``leak_pa_s`` to [0, 2000]; 2000 Pa/s would empty a 20 psi muscle in 69 s.
LEAK_MAX_PA_S = 2000.0


def closed_segments(t: np.ndarray,
                    p_pa: np.ndarray,
                    target_pa: np.ndarray,
                    *,
                    band_pa: float,
                    min_duration_s: float = 2.0,
                    min_mean_pa: float = 1.0 * PA_PER_PSI,
                    max_gap_s: float = 0.2,
                    sigma_pa: float = LEAK_SIGMA_PA,
                    leak_max_pa_s: float = LEAK_MAX_PA_S) -> list:
    """Maximal runs during which the valve can be assumed shut.

    THE DEPARTURE, and its cause.  The reference read the valve state off the
    firmware's ``ST_INFLATING``/``ST_VENTING`` status bits and called a segment
    closed when neither was set.  This bus has no such bits -- ``CompactStatus``
    carries ``ENABLED``, ``OTA_ACTIVE``, ``COMMAND_SEEN`` and ``ERROR`` and none
    of them is a valve state (CONTRACT departure 1) -- so the valve state is
    inferred from the quantity both firmwares actually act on: a board whose
    commanded error sits inside its own regulation band is holding, whether that
    band is the 7 mm firmware's +-2000 Pa hysteresis or the TLE proportional
    valve's dead zone.  Pass ``band_pa`` as that population's
    ``blend_width_pa``.

    The remaining four conditions are the reference's, unchanged: a constant
    commanded target (on a ramp, "closed" reply instants alias rising pressure),
    no sampling gap over ``max_gap_s``, no sample-to-sample jump beyond
    ``3 sigma + leak_max * dt``, and a mean pressure above ``min_mean_pa`` so the
    fit is not dominated by a muscle with nothing in it.

    Returns a list of ``(start, stop)`` index pairs, half-open.
    """
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p_pa, dtype=np.float64)
    g = np.asarray(target_pa, dtype=np.float64)
    n = t.size
    if n < 3:
        return []
    dt = np.diff(t)
    dp = np.diff(p)
    lim = 3.0 * sigma_pa + leak_max_pa_s * dt
    ok = ((np.abs(g[1:] - g[:-1]) <= 1e-9)
          & (np.abs(g[:-1] - p[:-1]) <= band_pa)
          & (np.abs(g[1:] - p[1:]) <= band_pa)
          & (dt > 0.0) & (dt <= max_gap_s)
          & (np.abs(dp) <= lim))
    out = []
    i = 0
    while i < ok.size:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < ok.size and ok[j]:
            j += 1
        a, b = i, j + 1
        if t[b - 1] - t[a] >= min_duration_s and p[a:b].mean() >= min_mean_pa:
            out.append((a, b))
        i = j + 1
    return out


#: Re-exported so a caller that has segments already does not import two modules
#: to fit one slope.  ``actuator_model`` owns the least squares; this module owns
#: deciding which stretches of a recording qualify.
fit_leak = _am.fit_leak


def node_closed_segments(rec: Recording, node_idx: int, *, band_pa: float,
                         min_duration_s: float = 2.0,
                         sigma_pa: float = LEAK_SIGMA_PA,
                         leak_max_pa_s: float = LEAK_MAX_PA_S,
                         min_mean_pa: float = 1.0 * PA_PER_PSI) -> list:
    """Every guaranteed-closed ``(t, p)`` pair for one board, across all episodes.

    The shape ``actuator_model.fit_leak_segments`` consumes.  Episode by episode,
    because a segment that reached across a boundary would span an unknown amount
    of wall clock and its slope would be arbitrary.
    """
    out = []
    for sl in rec.episode_slices():
        t = rec.can_sync_time_s[sl]
        p = rec.pressure_pa[sl, node_idx]
        g = rec.target_pa[sl, node_idx]
        for a, b in closed_segments(t, p, g, band_pa=band_pa,
                                    min_duration_s=min_duration_s,
                                    min_mean_pa=min_mean_pa,
                                    sigma_pa=sigma_pa,
                                    leak_max_pa_s=leak_max_pa_s):
            out.append((t[a:b], p[a:b]))
    return out


def fit_leaks(rec: Recording,
              *,
              blend_width_pa=_am.DEFAULT_BLEND_WIDTH_PA,
              initial_pa_s=None,
              min_duration_s: float = 2.0,
              leak_clip_pa_s=_am.LEAK_CLIP_PA_S,
              sigma_pa: float = LEAK_SIGMA_PA):
    """Per-node leak, Pa/s, by duration-weighted least squares on closed holds.

    Returns ``(leak_pa_s, info)`` and never mutates anything it was handed.  A
    node with no usable closed segment **keeps** its incoming value rather than
    being zeroed: the seam subtracts the leak, so a spuriously zero leak lets an
    idle board inflate just as surely as a negative one would, and "unfitted" is
    a fact the report should carry rather than a number the fit should invent.

    ``blend_width_pa`` doubles as the per-population "the valve is shut" band --
    see :func:`closed_segments` for why this bus has to infer that rather than
    read it off a status bit.
    """
    n = rec.pressure_pa.shape[1]
    init = (np.zeros(n, dtype=np.float64) if initial_pa_s is None
            else np.asarray(initial_pa_s, dtype=np.float64).copy())
    out = init.copy()
    info = {}
    for j in range(n):
        band = float(blend_width_pa[1 if rec.is_tle[j] else 0])
        segs = node_closed_segments(rec, j, band_pa=band,
                                    min_duration_s=min_duration_s,
                                    sigma_pa=sigma_pa,
                                    leak_max_pa_s=leak_clip_pa_s[1])
        closed_s = float(sum(float(t[-1] - t[0]) for t, _ in segs))
        try:
            raw = _am.fit_leak_segments(segs, min_duration_s=min_duration_s,
                                        leak_clip_pa_s=(-np.inf, np.inf))
        except ValueError:
            info[j] = {"initial_pa_s": float(init[j]), "n_segments": 0,
                       "closed_s": closed_s, "fitted_pa_s": None}
            continue
        out[j] = float(np.clip(raw, leak_clip_pa_s[0], leak_clip_pa_s[1]))
        info[j] = {"initial_pa_s": float(init[j]), "n_segments": len(segs),
                   "closed_s": closed_s, "fitted_raw_pa_s": float(raw),
                   "fitted_pa_s": float(out[j])}
    return out, info


# ---------------------------------------------------------------------------
# The synthetic session -- one reader for both real and generated data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticTruth:
    """The plant a synthetic session is generated from, so a test can score against it.

    ``dp/dt`` is a function of the pressure, the **latched** commanded error, and
    the muscle rate:

    ``dp/dt = rate[j] * base(p, e) * (1 + ldot_beta*ldot/LDOT_SCALE) - leak[j]``

    with ``base`` a supply-limited fill above the dead band, a
    pressure-proportional vent below it, and exactly zero inside it.

    TWO PROPERTIES ARE DELIBERATE.

    *The dead band is hard, not smooth.*  Both firmwares on this bus shut the
    valve near zero error -- the 7 mm boards by their +-2000 Pa hysteresis, the
    TLE boards by their dead zone -- and it is that shut interval, not the
    regulation itself, that makes the leak fittable.  A truth without it holds
    pressure at a droop equilibrium, no segment ever decays, and the least-squares
    leak comes back as zero, which is what the first version of this generator
    did.

    *The fill is fast on purpose.*  ``fill_a_pa_s`` defaults to 2.5e5 Pa/s, the
    order of the fast fill the reference measured on the RS485 rig (~275 kPa/s).
    Against a 6.67 ms control period that is roughly 1.2 kPa of overshoot per
    refill, which is what turns the dead band into an effective hysteresis and
    gives closed decays of several seconds instead of the sub-100 ms ones a slow
    fill would leave.  Lower it and the leak fit silently loses its evidence.

    WHAT IT IS NOT.  It is not this arm's plant, and no number in it is measured
    on this arm.  It exists so the reader, the windowing, the leak fit, the
    shooting objective and the checkpoint round trip can be exercised end to end
    against a known answer, and a fit that recovers it has shown that the pipeline
    works, not that the twin is right.
    """

    rate: tuple                     # (24,) per-node flow scale, dimensionless
    leak_pa_s: tuple                # (24,) Pa/s
    fill_a_pa_s: float = 2.5e5
    vent_a_pa_s: float = 1.8e5
    p_ceil_pa: float = 30.0 * PA_PER_PSI
    ldot_beta: float = 0.25
    ldot_scale_m_s: float = 0.5
    dead_band_pa: tuple = (2000.0, 6000.0)

    def flow_pa_s(self, node_idx, p_pa, e_pa, ldot_m_s, is_tle) -> np.ndarray:
        j = np.asarray(node_idx)
        p = np.asarray(p_pa, dtype=np.float64)
        e = np.asarray(e_pa, dtype=np.float64)
        dead = np.where(np.asarray(is_tle), self.dead_band_pa[1], self.dead_band_pa[0])
        frac = np.clip(p / self.p_ceil_pa, 0.0, 1.0)
        base = np.where(e > dead, self.fill_a_pa_s * (1.0 - frac),
                        np.where(e < -dead, -self.vent_a_pa_s * frac, 0.0))
        mod = 1.0 + self.ldot_beta * np.asarray(ldot_m_s) / self.ldot_scale_m_s
        rate = np.asarray(self.rate, dtype=np.float64)[j]
        leak = np.asarray(self.leak_pa_s, dtype=np.float64)[j]
        return rate * base * mod - leak


def write_synthetic_session(out_dir,
                            *,
                            n_episodes: int = 4,
                            episode_s: float = 8.0,
                            rate_hz: float = 150.0,
                            board_type: Sequence | None = None,
                            truth: SyntheticTruth | None = None,
                            seed: int = 20260910,
                            substep_s: float = 0.001,
                            chunk_cycles: int = 4096,
                            tendon=None,
                            q_amp_rad: float = 0.30,
                            noise_counts: float = 1.2,
                            session_tag: str = "synthetic") -> str:
    """Generate a recording in the real schema, from a known plant.

    It writes ``metadata.json``, ``manifest.json`` and ``samples_chunk_*.jsonl``
    with the fields ``data_schema.md`` pins, so :func:`load_session` is the only
    reader either a real or a synthetic session ever needs.  That single-reader
    property is the point: a bespoke in-memory fixture would let the reader and
    the thing it reads drift apart, and the drift would only surface the first
    time a real session arrived.

    Everything is written raw -- ADC counts and radians -- and each board's counts
    are produced with **its own** calibration, so a reader that guesses the
    calibration from the id range gets a wrong answer here exactly as it would on
    the bench.

    The excitation is a per-node piecewise-constant target with 1-2 s dwells,
    which gives both the ramp transients the flow net needs and the >= 2 s
    constant-target holds the leak fit needs.  Joints are smooth low-frequency
    sinusoids: they exercise ``l``/``ldot`` without pretending to be a mocap
    recording, and no attempt is made to make ``q`` consistent with the pressures.

    Returns the session directory path.
    """
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if board_type is None:
        board_type = [VARIANT_TLE_DVP] * 8 + [VARIANT_7MM] * 16
    board_type = [int(b) for b in board_type]
    n = len(board_type)
    ids = [0x101 + k for k in range(n)]
    cals = [cal_for_variant(b, base=i) for b, i in zip(board_type, ids)]
    is_tle = np.array([c.is_tle for c in cals], dtype=bool)

    if truth is None:
        rate = 1.0 + 0.35 * rng.standard_normal(n)
        truth = SyntheticTruth(rate=tuple(np.clip(rate, 0.5, 1.8)),
                               leak_pa_s=tuple(rng.uniform(150.0, 500.0, n)))

    dt = 1.0 / float(rate_hz)
    k_per_ep = int(round(episode_s * rate_hz))
    nsub = max(1, int(round(dt / substep_s)))
    h = dt / nsub

    # The truth's ldot must be the ldot the *reader* derives, not an independent
    # one: otherwise a recovery test measures the gap between two definitions of
    # muscle rate rather than the trainer's ability to recover a plant.  So the
    # generator pushes q through the same geometry and the same smooth-then-
    # differentiate that ``muscle_traces`` uses.
    if tendon is None:
        tendon = MomentArmTendonGeometry(n_nodes=n)
    l0 = l0_per_actuator(n_nodes=n)

    rows = []
    cycle = 0
    t = 0.0
    node_ix = np.arange(n)
    for ep in range(n_episodes):
        # A fresh episode is a fresh recording: the wall clock jumps by more than
        # any within-episode gap so the reader's own heuristic has to find the
        # boundary, rather than the generator handing it a label it will not get
        # from a real collector.
        t += 5.0 + float(rng.uniform(0.0, 1.0))
        p = rng.uniform(0.0, 4.0, n) * PA_PER_PSI
        tgt = _synthetic_targets(rng, n, k_per_ep, dt)
        qphase = rng.uniform(0.0, 2 * math.pi, N_JOINTS)
        qfreq = rng.uniform(0.30, 0.90, N_JOINTS)
        tt_all = t + np.arange(k_per_ep) * dt
        ang = 2 * math.pi * qfreq[None, :] * tt_all[:, None] + qphase[None, :]
        q_all = q_amp_rad * np.sin(ang)
        qd_all = q_amp_rad * 2 * math.pi * qfreq[None, :] * np.cos(ang)
        l_all = smooth3(l0[None, :] + tendon.dlen(q_all), axis=0)
        ldot_all = np.gradient(l_all, tt_all, axis=0)
        # The reported pressure the regulator acts on, latched at the previous
        # sync edge.  Seeded from the initial true pressure so the first cycle
        # regulates on something a sensor could have reported.
        adc = np.array([c.pa_to_adc(p[j]) for j, c in enumerate(cals)])
        p_rep = np.array([c.adc_to_pa(adc[j]) for j, c in enumerate(cals)])
        for k in range(k_per_ep):
            tt = float(tt_all[k])
            qk, qdk = q_all[k], qd_all[k]
            ldot = ldot_all[k]
            g = tgt[k]
            # CONTRACT section 4's node-pass order, and the reason a synthetic
            # session has to reproduce it: the error is latched once per sync
            # edge from the pressure the *sensor* reported, then held while the
            # plant integrates.  Recomputing it every substep would let the valve
            # shut the instant it crossed the dead band, removing the overshoot
            # that turns that band into an effective hysteresis -- and with it
            # every closed decay the leak fit lives on.
            e = g - p_rep
            for _ in range(nsub):
                dp = truth.flow_pa_s(node_ix, p, e, ldot, is_tle)
                p = np.maximum(0.0, p + dp * h)
            adc = np.array([c.pa_to_adc(p[j]) for j, c in enumerate(cals)])
            adc = np.clip(np.rint(adc + noise_counts * rng.standard_normal(n)),
                          0, ADC_MAX)
            p_rep = np.array([c.adc_to_pa(adc[j]) for j, c in enumerate(cals)])
            tadc = np.array([c.pa_to_adc(g[j]) for j, c in enumerate(cals)])
            rows.append({
                "schema_version": 2,
                "cycle": cycle,
                "phase": "collecting",
                "timestamp_unix_s": 1.7e9 + tt,
                "timestamp_s": tt,
                "cycle_start_time_s": tt - 0.0014,
                "can_sync_time_s": tt,
                "jitter_ms": 0.0,
                "ids": [f"0x{i:03X}" for i in ids],
                "board_type": board_type,
                "robot_state": {"q": [float(v) for v in qk],
                                "qdot": [float(v) for v in qdk],
                                "pressure_adc": [int(v) for v in adc]},
                "input": {"target_adc": [int(v) for v in tadc]},
                "cycle_responded": n, "cycle_expected": n, "total_missed": 0,
                "joint_current_valid": True,
                "q_stale": False,
                "mocap": {"valid": True, "stale": False, "frame": 900000 + cycle},
            })
            cycle += 1
        # Carry the clock to the end of this episode; the next one's gap is added
        # on top of it.  Without this the episodes overlap in wall time and
        # ``can_sync_time_s`` stops being monotonic, which no reader should have
        # to tolerate and which a real collector cannot produce.
        t = float(tt_all[-1])

    chunk_paths = []
    chunks_meta = []
    for i in range(0, len(rows), chunk_cycles):
        part = rows[i:i + chunk_cycles]
        reason = "complete" if i + chunk_cycles >= len(rows) else "checkpoint"
        name = f"samples_chunk_{len(chunk_paths):04d}_{reason}_{session_tag}.jsonl"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
        chunk_paths.append(name)
        chunks_meta.append({"path": name, "reason": reason, "samples": len(part),
                            "start_cycle": part[0]["cycle"],
                            "end_cycle": part[-1]["cycle"],
                            "start_time_s": part[0]["can_sync_time_s"],
                            "end_time_s": part[-1]["can_sync_time_s"]})

    meta = {
        "schema_version": 2,
        "session_tag": session_tag,
        "created_at_unix_s": 1.7e9,
        "duration_s": len(rows) * dt,
        "sample_rate_hz": rate_hz,
        "state_fields": ["q", "qdot", "pressure_adc"],
        "state_dimension": 2 * N_JOINTS + n,
        "input_fields": ["target_adc"],
        "input_dimension": n,
        "pressure_units": "adc_counts",
        "selected_ids": [f"0x{i:03X}" for i in ids],
        "board_type": board_type,
        "adc_ranges": [[0, ADC_MAX]] * n,
        "single_actuator_limit_psi": 30.0,
        "pair_sum_limit_psi": 30.0,
        "sync_semantics": "can_sync_time_s is the sync edge; pressures are the "
                          "value latched at that edge, targets the value promoted at it",
        "mocap_live": False, "mocap_sim": True,
        "synthetic": True,
        "synthetic_truth": {"rate": list(map(float, truth.rate)),
                            "leak_pa_s": list(map(float, truth.leak_pa_s)),
                            "fill_a_pa_s": truth.fill_a_pa_s,
                            "vent_a_pa_s": truth.vent_a_pa_s,
                            "p_ceil_pa": truth.p_ceil_pa,
                            "ldot_beta": truth.ldot_beta,
                            "dead_band_pa": list(truth.dead_band_pa),
                            "seed": seed},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"total_samples": len(rows), "checkpoint_count": len(chunks_meta),
                   "active": False, "chunks": chunks_meta}, fh, indent=2)
    return out_dir


def _synthetic_targets(rng, n_nodes, k, dt):
    """Piecewise-constant per-node setpoints, psi, honouring the pair-sum rule.

    Dwells are 1-2 s: long enough that a 2 s leak segment exists inside the
    longer ones, short enough that an 8 s episode carries several transients.
    Levels are capped at 12 psi per node -- well under the operator's 30 psi
    per-line and per-pair ceiling (CONTRACT section 8) -- so the generated
    excitation could be replayed on the bench without violating the rule it is
    supposed to respect.
    """
    out = np.zeros((k, n_nodes), dtype=np.float64)
    for j in range(n_nodes):
        i = 0
        while i < k:
            dwell = int(round(rng.uniform(1.0, 2.0) / dt))
            level = float(rng.choice([0.0, 2.0, 5.0, 8.0, 12.0]))
            out[i:i + dwell, j] = level
            i += dwell
    return out * PA_PER_PSI
