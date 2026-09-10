"""Live Motive/NatNet -> ``q`` receiver, with health, history and rest capture.

Wraps the vendored NaturalPoint SDK (``natnet_sdk/``) the way the legacy
``natnet_receiver.NatNetConfigReceiver`` did — same lifecycle, same listeners,
same ID convention — and adds the three things the legacy version lacked and
that a test bench needs:

1. **Honest health.**  The legacy receiver could only tell you the last ``q`` it
   ever computed.  A dropped stream, an occluded plate or a Motive restart all
   look identical to "the arm is holding still", which is exactly the failure a
   controller must not act on.  :meth:`MocapRx.get_state` reports the frame
   number, the wall time of the last frame, a measured frame rate and a
   :attr:`MocapState.stale` flag (no frame for
   ``mocap_constants.STALE_AFTER_S``), and :meth:`MocapRx.wait_fresh` blocks for
   a frame that arrived *after* the call rather than handing back history.

2. **History.**  A ring buffer of the last ~20 s of ``(t_mono, frame_no, q,
   u_joint_positions)``.  Step responses and stiffness tests need the samples
   around an event, and the event is only recognised as interesting after it has
   happened; :meth:`MocapRx.snapshot_window` pulls the window back out.

3. **Rest capture.**  :meth:`MocapRx.capture_rest` averages ``q`` over a window
   and reports the spread, which is how the joint zero (a mounting property, not
   a constant — see :mod:`mocap_constants`) and the measurement noise floor get
   established.

4. **Marker transport** (``docs/marker_frame_design.md`` §3).  The vendored SDK
   parses every marker in the volume and then drops them; local patch 1
   (``natnet_sdk/PROVENANCE.md``) adds a ``mocap_data_listener`` that hands the
   fully parsed ``MoCapData`` to :meth:`MocapRx._on_mocap_data`, which maps
   marker sets onto arm plates, reads per-marker tracked flags off the labeled
   markers, and records everything in a **separate marker ring** —
   :meth:`MocapRx.snapshot_marker_window` pulls it back out as a
   :class:`MarkerWindow`.  The marker ring is appended **every** frame,
   including frames whose ``q`` was degenerate: the q=None frames are exactly
   the dropout-rich frames the marker-frame benchmark studies (design review
   findings int-8/ops-7).  The q ring and :class:`MocapWindow` are untouched —
   existing consumers construct :class:`MocapWindow` with exactly four kwargs
   and must keep working unchanged (design decision D5).

Threading model, which is the part worth reading before editing: the SDK owns a
data thread and calls ``rigid_body_listener`` once per plate, then (patch 1)
``mocap_data_listener`` once per frame, then ``new_frame_listener`` once per
frame, all on that thread.  Poses therefore accumulate in a scratch array that
only the SDK thread ever touches — and marker data in a scratch dict with the
same ownership rule — and are published as copies under :attr:`_lock` exactly
once per frame, so a reader can never observe a half-updated set of plates.
User callbacks run *outside* the lock — a callback that reached back into
:meth:`get_q` while we still held it would deadlock.

Known limitation, inherited and not fixable at this layer: **a single plate going
bad is invisible here.**  ``NatNetClient.__unpack_rigid_body`` calls the listener
for every rigid body in the frame before it even parses the tracking-valid bit,
and hands us only ``(id, position, quaternion)`` — so an occluded plate arrives
with Motive's last/predicted pose and a plate omitted from the frame keeps
whatever it had.  Either way the frame still converts, and ``q`` looks plausible:
one untracked plate leaves both of its neighbouring link differences
non-degenerate, so ``mocap_to_q`` has nothing to reject on, and the joints around
it quietly encode the direction to the mocap volume's origin.
:attr:`MocapState.stale` catches the whole stream dying, not one marker set
dying; that is what the raw U-joint centres in the ring buffer are for, and why
Motive's own tracking view is still the authority during a run.  Wiring the
tracking-valid bit through would mean editing vendored SDK code.

What *is* caught is the degenerate pose Motive sends for a body it cannot solve
at all (``pos=(0,0,0), quat=(0,0,0,0)``): :func:`mocap_to_q.quat_xyzw_to_matrix`
raises on it, so the plate keeps its previous pose and
:attr:`MocapState.last_error` names it, instead of an all-NaN rotation
de-rotating every link into NaN and being published as a healthy frame.

Usage::

    rx = MocapRx()                      # lab defaults from mocap_constants
    rx.start()
    q = rx.wait_fresh(timeout=2.0)      # None if Motive is not streaming
    ...
    rest = rx.capture_rest(3.0)         # rest.mean / rest.sd, 12 joints each
    rx.stop()
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:  # see the note in mocap_to_q.py — both import styles must work
    from . import mocap_constants as mc
    from .mocap_to_q import mocap_to_q, quat_xyzw_to_matrix
except ImportError:  # pragma: no cover
    import mocap_constants as mc  # type: ignore[no-redef]
    from mocap_to_q import mocap_to_q, quat_xyzw_to_matrix  # type: ignore[no-redef]

#: Frames the rate estimate averages over: ~0.5 s at 120 Hz, long enough to
#: smooth USB/UDP jitter, short enough to notice a stall promptly.
_RATE_WINDOW = 64

#: ``LabeledMarker.param`` bits (``MoCapData.py:512-516`` — the decode helpers
#: there are name-mangled private, so this module does its own bit ops on the
#: public ``.param`` attribute, exactly as the design (§3) prescribes).
_PARAM_OCCLUDED = 0x01
_PARAM_POINT_CLOUD_SOLVED = 0x02
_PARAM_MODEL_SOLVED = 0x04

#: Motive's special all-markers set: present in every frame's ``MarkerSetData``
#: alongside the per-asset sets, never an arm plate.  Marker-set names are
#: *bytes* on the wire (``MoCapData.py:126-165``), so the comparison is a bytes
#: comparison — ``"all"`` (str) would silently never match.
_ALL_SET_NAME = b"all"

#: Centroid-fallback mapping bounds (design §3): with no labeled markers to
#: group by model id, a 4-marker set may claim a plate only when its centroid
#: sits within 20 mm of that plate's streamed pivot *and* the runner-up plate
#: is at least twice as far.  The 47 mm inter-plate gaps make a sloppier
#: nearest-neighbour unsafe.
_CENTROID_WIN_M = 0.020
_CENTROID_RUNNER_UP_FACTOR = 2.0

#: Fraction of the requested window :meth:`MocapRx.capture_rest` must actually
#: have samples across before it will report a mean.  A healthy stream covers
#: ~0.97 (the shortfall is the gap before the first sample lands), so this is a
#: floor on "the stream was really running", not a tight tolerance — it exists to
#: reject the window that holds two samples from a stream that then died.
MIN_REST_COVERAGE = 0.5


# --------------------------------------------------------------------------
# Value types handed to callers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MocapState:
    """Immutable snapshot of stream health.  Cheap; poll it freely."""

    running: bool = False
    #: Frames the SDK has delivered since :meth:`MocapRx.start`.
    frames: int = 0
    #: Frames that yielded a usable ``q`` (the rest were degenerate — a plate
    #: Motive had not seen yet).
    valid_frames: int = 0
    #: Motive's own frame counter for the last frame; gaps mean dropped UDP.
    frame_number: int | None = None
    #: ``time.time()`` when the last frame arrived — for lining mocap up with
    #: other lab logs, which are wall-clock stamped.
    last_frame_wall: float | None = None
    #: ``time.monotonic()`` when the last frame arrived — for durations, which
    #: must survive a wall-clock step.
    last_frame_mono: float | None = None
    #: Frames/s measured over the last :data:`_RATE_WINDOW` frames, as of the
    #: last frame.  It does *not* decay when the stream dies — that is what
    #: :attr:`stale` is for.
    fps: float = 0.0
    #: True when no frame has arrived for ``mocap_constants.STALE_AFTER_S``
    #: (or none ever has).
    stale: bool = True
    #: ``time.monotonic()`` when the last frame that yielded a usable ``q``
    #: arrived.  Distinct from :attr:`last_frame_mono`, which advances on every
    #: frame whether or not it converted.
    last_q_mono: float | None = None
    #: True when no *valid* frame has arrived for ``STALE_AFTER_S``.  **This,
    #: not :attr:`stale`, is the field a control loop must check.**  The two
    #: differ exactly when frames keep arriving and stop converting — an
    #: untracked plate, or a marker solve that keeps failing its gates — and in
    #: that state ``get_q()`` still returns a pose, silently older every tick,
    #: while ``stale`` reads False.  A loop steering on that is steering on
    #: history.
    q_stale: bool = True
    #: Samples currently held in the ring buffer.
    ring_len: int = 0
    #: Last exception raised inside a listener, kept because an exception on the
    #: SDK thread is otherwise invisible.
    last_error: str | None = None


@dataclass(frozen=True)
class MocapWindow:
    """A slice of recorded history, transposed into arrays for plotting/fitting."""

    #: ``time.monotonic()`` per sample, shape ``(n,)``.
    t: np.ndarray
    #: Motive frame numbers, shape ``(n,)``.
    frame_no: np.ndarray
    #: Joint vectors, shape ``(n, 12)``, radians.
    q: np.ndarray
    #: U-joint centres in the mocap spatial frame, shape ``(n, 6, 3)``, metres.
    #: Kept alongside ``q`` because a suspicious ``q`` is almost always a plate
    #: problem, and the raw centres are what shows that.
    u: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        return 0.0 if len(self) < 2 else float(self.t[-1] - self.t[0])

    @property
    def fps(self) -> float:
        d = self.duration
        return 0.0 if d <= 0.0 else (len(self) - 1) / d


@dataclass(frozen=True)
class MarkerWindow:
    """A slice of recorded *marker* history (``marker_frame_design.md`` §3).

    Deliberately a separate type from :class:`MocapWindow`: marker data is
    ragged (plates drop in and out of the mapping, marker counts can change
    mid-session), so per-frame containers are the honest shape, and grafting a
    fifth field onto :class:`MocapWindow` would have broken every consumer that
    constructs it with exactly four kwargs (design decision D5).  Align the two
    windows offline by ``frame_no``, which both carry.
    """

    #: ``time.monotonic()`` per sample, shape ``(n,)``.
    t: np.ndarray
    #: Motive frame numbers, shape ``(n,)``.
    frame_no: np.ndarray
    #: Plate-mapping epoch per sample, shape ``(n,)``.  Bumped every time the
    #: marker-set -> plate mapping is re-derived (roster change in Motive), so
    #: offline analysis can split on it rather than mix rosters (review ops-6).
    mapping_epoch: np.ndarray
    #: Length-``n`` tuple.  Each entry is ``{plate: (m, 3) float array}`` in
    #: asset order — or ``None`` for a frame whose marker data did not arrive
    #: (marker sets absent, or the SDK never fired the mocap_data listener).
    #: ``None`` is *not streamed*; an empty dict is *streamed but no arm asset
    #: mapped* — the distinction the campaign CSVs spell ``-`` (review ops-9).
    markers: tuple
    #: Length-``n`` tuple.  Each entry is ``{plate: (m,) uint8 array}`` with
    #: bit0 = tracked-this-frame (labeled entry present, occluded and
    #: model-solved bits clear) — or ``None`` when labeled markers were absent
    #: that frame, i.e. flags are *unknown*, never assumed all-present
    #: (design decision D4).
    flags: tuple
    #: Streamed rigid-body poses, shape ``(n, 7, 4, 4)`` — the
    #: ``N_USED_RIGID_BODIES`` slice of the receiver's pose array at commit
    #: time.  Nothing else records the streamed base orientation the fkine
    #: benchmark's streamed-base regime needs (review finding int-0).
    streamed_poses: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        return 0.0 if len(self) < 2 else float(self.t[-1] - self.t[0])

    @property
    def fps(self) -> float:
        d = self.duration
        return 0.0 if d <= 0.0 else (len(self) - 1) / d


@dataclass(frozen=True)
class MarkerHealth:
    """Bookkeeping the live probe reports (``marker_frame_design.md`` §6).

    All counters are written only by the SDK thread and read from anywhere —
    same single-writer rule as :attr:`MocapRx._incoming`, so no lock is needed
    for these monotonically increasing ints and wholesale-swapped dicts.
    """

    #: Frames for which the (patched) SDK delivered a ``MoCapData`` object.
    mocap_data_frames: int = 0
    #: Of those, frames carrying at least one labeled marker.  When this stays
    #: zero, every flag in the marker ring is ``None`` — the *flags-unknown*
    #: regime the probe must stamp into any lock it writes.
    labeled_frames: int = 0
    #: Marker-set / labeled-marker counts of the most recent data frame.
    last_marker_set_count: int = 0
    last_labeled_marker_count: int = 0
    #: Current plate mapping, ``{bytes marker-set name: plate index}``.
    plate_names: dict = field(default_factory=dict)
    mapping_epoch: int = 0
    #: Tallies of the ``marker_id - 1 == asset-order index`` assumption, over
    #: *tracked* labeled markers only (occluded/model-solved entries may carry
    #: a model-filled position that legitimately differs from the set's, so
    #: they must not poison the check).  The probe verifies the assumption from
    #: these before trusting flags (§3: stated, then verified).
    id_corr_ok: int = 0
    id_corr_bad: int = 0
    #: Occluded-marker encoding actually observed on this Motive (design §5:
    #: server-side behaviour that varies by version/settings, so it is
    #: *recorded*, not assumed).
    occluded_seen: int = 0
    point_cloud_solved_seen: int = 0
    model_solved_seen: int = 0
    #: Samples currently held in the marker ring.
    marker_ring_len: int = 0


@dataclass(frozen=True)
class RestCapture:
    """Statistics of ``q`` while the arm was meant to be holding still."""

    #: Samples behind the statistics.  Always >= 2 — :meth:`MocapRx.capture_rest`
    #: returns ``None`` rather than build one of these from less.
    n: int
    duration: float
    #: Per-joint mean, shape ``(12,)`` — the candidate joint zero.
    mean: np.ndarray
    #: Per-joint standard deviation, shape ``(12,)`` — the measurement noise
    #: floor.  A joint whose sd is much larger than its neighbours' is a marker
    #: visibility problem, not a compliant joint.  Always a real sample sd
    #: (``ddof=1``), never a zero standing in for "too few samples".
    sd: np.ndarray
    #: Per-joint peak-to-peak, which catches a single dropout that barely moves
    #: the sd.
    ptp: np.ndarray
    window: MocapWindow = field(repr=False)


# --------------------------------------------------------------------------
# The receiver
# --------------------------------------------------------------------------


class MocapRx:
    """Motive stream -> latest ``q``, stream health, and recorded history.

    Nothing here touches the network until :meth:`start`, so importing this
    module (or constructing the object) is safe in tests and on a PC with no
    route to the mocap machine.

    WHICH RIGID BODIES.  ``rb_id_base`` and ``n_bodies`` name the block of
    Motive streaming ids this receiver claims: row ``i`` of the pose array is
    streaming id ``rb_id_base + i``, for ``i`` in ``0 .. n_bodies - 1`` (plus
    one spare row, kept from the RS485 layout).  They default to the RS485
    arm's block and are per-instance rather than module constants so that two
    arms on different bases can be received in one process against one NatNet
    stream; see :mod:`UMArm_MOCAP.canarm_mocap` for the CAN arm's subclass.
    The Kinova's row is routed separately from
    ``mocap_constants.KINOVA_MOCAP_STREAM_ID`` and is unaffected.
    """

    def __init__(self,
                 server_ip: str = mc.DEFAULT_SERVER_IP,
                 client_ip: str = mc.DEFAULT_CLIENT_IP,
                 use_multicast: bool = mc.DEFAULT_USE_MULTICAST,
                 on_q=None,
                 ring_capacity: int = mc.RING_CAPACITY,
                 marker_ring_capacity: int | None = None,
                 rb_id_base: int = mc.RIGID_BODY_ID_MASK,
                 n_bodies: int = mc.N_USED_RIGID_BODIES) -> None:
        self.server_ip = server_ip
        self.client_ip = client_ip
        self.use_multicast = use_multicast

        # --- Which block of Motive streaming ids this receiver answers to ---
        # This was a module constant (mocap_constants.RIGID_BODY_ID_MASK) read
        # at five sites.  It is per-instance so that two receivers with
        # different bases can run against one NatNet stream in one process,
        # which is exactly what a room holding two arms needs.  The Kinova's
        # 1008 stays a module constant deliberately: there is one Gen3, its id
        # is a single number rather than a block, and it lands on its own row.
        #: Motive streaming id of array row 0, i.e. ``id = rb_id_base + index``.
        self.rb_id_base = int(rb_id_base)
        #: How many of the array's rows this arm's plates occupy.  Gates the
        #: labeled-marker plate derivation and the centroid mapping fallback,
        #: neither of which may claim a row the arm does not have.
        self.n_bodies = int(n_bodies)
        if not 1 <= self.n_bodies <= mc.KINOVA_RIGID_BODY_INDEX:
            raise ValueError(
                "n_bodies must be 1..%d (the rows preceding the Kinova row in "
                "a (%d, 4, 4) pose array); got %d"
                % (mc.KINOVA_RIGID_BODY_INDEX, mc.NUM_RIGID_BODIES,
                   self.n_bodies))
        #: Rigid-body routing window width: the arm's own rows plus the one
        #: spare row the array layout reserves (see the index map in
        #: ``mocap_constants``), clamped to the rows preceding the Kinova's.
        #: At the default ``n_bodies = N_USED_RIGID_BODIES = 7`` this is 8,
        #: i.e. the pre-refactor window ``[MASK, MASK + 8)`` exactly.
        self._n_rb_rows = min(self.n_bodies + 1, mc.KINOVA_RIGID_BODY_INDEX)
        #: Optional ``callback(q)`` fired once per valid frame, on the SDK
        #: thread, outside the lock.  Keep it short: it runs in the receive path.
        #: The ``q`` handed over is the callback's own copy, so mutating it in
        #: place is safe and affects neither :meth:`get_q` nor the ring buffer.
        self.on_q = on_q

        # Written only by the SDK thread as plates arrive; never read by anyone
        # else, which is why it needs no lock.  Untracked plates stay at the
        # identity.  mocap_to_q rejects a frame only when two *adjacent* plates
        # are still there (their difference is 0/0) — one untracked plate mid-arm
        # still converts, and its joints then encode the direction to the mocap
        # origin.  See the "known limitation" note above: catching that needs
        # Motive's tracking-valid bit, which the SDK does not hand us.
        self._incoming = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)

        # --- Marker transport (marker_frame_design.md §3) -------------------
        # All of the following are written only by the SDK thread, same
        # single-writer rule as _incoming.  The scratch holds *this frame's*
        # marker data between _on_mocap_data and _on_new_frame, which consumes
        # and clears it — so a frame whose marker data did not arrive records
        # None, never a stale copy (review finding int-4).
        self._marker_scratch: dict | None = None   # plate -> (m, 3) float64
        self._flags_scratch: dict | None = None    # plate -> (m,) uint8, or None
        #: bytes marker-set name -> plate index.  Replaced wholesale on remap
        #: (never mutated in place), so unlocked readers always see a
        #: consistent mapping.
        self._plate_names: dict = {}
        self._mapping_epoch = 0
        self._remap_needed = True       # first data frame derives the mapping
        #: The §3 assumption "marker_id - 1 == asset-order index", used to place
        #: tracked flags.  The probe verifies it empirically (via the
        #: id_corr_ok/id_corr_bad tallies) and may set this False, after which
        #: flags come from per-frame position matching alone.  Position
        #: matching is also the automatic per-marker fallback whenever the
        #: assumed index does not hold that marker's exact position.
        self.marker_index_by_id = True
        # MarkerHealth counters (see the dataclass for what each means).
        self._mocap_data_frames = 0
        self._labeled_frames = 0
        self._last_marker_set_count = 0
        self._last_labeled_marker_count = 0
        self._id_corr_ok = 0
        self._id_corr_bad = 0
        self._occluded_seen = 0
        self._point_cloud_solved_seen = 0
        self._model_solved_seen = 0

        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._homos: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._seq = 0                      # bumped on every published frame
        self._frames = 0
        self._valid_frames = 0
        self._frame_number: int | None = None
        self._last_wall: float | None = None
        self._last_mono: float | None = None
        self._last_q_mono: float | None = None
        self._frame_times: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._ring: deque[tuple[float, int, np.ndarray, np.ndarray]] = deque(
            maxlen=int(ring_capacity))
        # Separate ring for marker data, appended every frame (q=None frames
        # included).  Default capacity follows the q ring, and both should be
        # sized from the *measured* rate by long-running callers (probe §6,
        # campaign §7) — the 120 Hz constant undersizes a 240 Hz volume.
        self._marker_ring: deque[tuple] = deque(
            maxlen=int(ring_capacity if marker_ring_capacity is None
                       else marker_ring_capacity))
        self._last_error: str | None = None
        self._client = None

    # ------------------------------------------------------------------ #
    # SDK listeners (all called on the SDK's data thread)
    # ------------------------------------------------------------------ #

    def _on_rigid_body(self, new_id, position, quat_xyzw) -> None:
        """One plate of one frame.  Maps Motive's streaming id to an array row.

        Ids outside the arm's block are dropped rather than trusted: a stray
        rigid body left in the Motive project must not land on a joint's row.

        Rows persist between frames, so a plate the frame does not mention keeps
        its previous pose — see the "known limitation" note in the module
        docstring before relying on that.
        """
        try:
            if new_id == mc.KINOVA_MOCAP_STREAM_ID:
                index = mc.KINOVA_RIGID_BODY_INDEX
            elif (self.rb_id_base <= new_id
                  < self.rb_id_base + self._n_rb_rows):
                index = new_id - self.rb_id_base
            else:
                return
            h = self._incoming[index]
            h[0:3, 0:3] = quat_xyzw_to_matrix(quat_xyzw)
            h[0:3, 3] = position
            h[3, 0:3] = 0.0
            h[3, 3] = 1.0
        except Exception as exc:  # never let the SDK thread die on bad data
            self._note_error(f"rigid_body {new_id}: {exc}")

    def _on_mocap_data(self, mocap_data) -> None:
        """One frame's fully parsed ``MoCapData`` (vendored patch 1) -> scratch.

        Fires between ``rigid_body_listener`` and ``new_frame_listener`` on the
        SDK thread, so by the time :meth:`_on_new_frame` commits, this frame's
        marker data is either in the scratch or provably absent.  Everything is
        filtered to mapped arm assets *before* any array is built — a cluttered
        volume must not grow the 120 Hz receive path.  Two source regimes:
        per-asset marker sets (name-mapped, flags matched from labeled
        entries) and labeled-only (this lab's observed stream: ids carry the
        plate and slot directly; see the branch below).
        """
        try:
            self._mocap_data_frames += 1
            msd = getattr(mocap_data, "marker_set_data", None)
            sets = [] if msd is None else msd.marker_data_list
            self._last_marker_set_count = len(sets)

            lmd = getattr(mocap_data, "labeled_marker_data", None)
            labeled = [] if lmd is None else lmd.labeled_marker_list
            self._last_labeled_marker_count = len(labeled)
            if labeled:
                self._labeled_frames += 1

            # Group labeled markers by model id — streaming IDs rb_id_base+i
            # are the arm's plates (mocap_constants index map); everything else
            # in the volume is dropped here, before any per-marker work.
            by_model: dict[int, dict] = {}
            for lm in labeled:
                plate = (lm.id_num >> 16) - self.rb_id_base
                if 0 <= plate < self.n_bodies:
                    by_model.setdefault(plate, {})[lm.id_num & 0xFFFF] = lm

            if not sets and not by_model:
                # Neither per-asset marker sets nor arm labeled markers in the
                # frame: leave the scratch None so the commit records "not
                # streamed", distinguishable from "streamed but nothing
                # mapped" ({}).
                return

            if not sets:
                # Labeled-only regime.  Observed live 2026-08-11: this lab's
                # Motive streams labeled markers but NO per-asset marker sets
                # (the frame header's "marker_set_count" is the SDK's
                # legacy-other-markers count, overwritten at
                # NatNetClient.py:870, and had nothing to do with marker
                # sets).  Labeled ids are *better* labels than asset order:
                # model_id names the plate outright and marker_id - 1 is the
                # stable slot, so no name mapping is needed at all and the
                # centroid fallback never runs.  An id absent this frame
                # (Motive omitted the marker) leaves a NaN row with flag 0 —
                # the same dropout the flags scheme marks, and the array stays
                # 4 rows so the lock's count gate keeps its meaning.
                if self._mapping_epoch == 0:
                    self._mapping_epoch = 1     # ids are absolute: "mapped"
                markers = {}
                flags = {}
                for plate, entries in by_model.items():
                    n = max(4, max(entries))
                    arr = np.full((n, 3), np.nan, dtype=float)
                    fl = np.zeros(n, dtype=np.uint8)
                    for marker_id, lm in entries.items():
                        idx = marker_id - 1
                        if not (0 <= idx < n):
                            continue
                        arr[idx] = lm.pos
                        if self._tally_param(lm.param):
                            fl[idx] = 1
                    markers[plate] = arr
                    flags[plate] = fl
                self._marker_scratch = markers
                self._flags_scratch = flags
                return

            suffix = getattr(mocap_data, "suffix_data", None)
            if suffix is not None and suffix.tracked_models_changed:
                # Motive's asset roster changed: names may now mean different
                # assets, so the mapping must be re-derived (review ops-6).
                self._remap_needed = True
            if self._remap_needed:
                mapping = self._derive_mapping(sets, by_model, mocap_data)
                if mapping is not None:
                    # Wholesale swap + epoch bump; the old mapping is kept when
                    # this frame cannot support a derivation (e.g. every plate
                    # momentarily occluded) — names are not reassigned between
                    # roster events, and the epoch only moves when the mapping
                    # actually does, which is what offline splitting keys on.
                    self._plate_names = mapping
                    self._mapping_epoch += 1
                    self._remap_needed = False

            name_to_set = {md.model_name: md for md in sets}  # references only
            markers: dict[int, np.ndarray] = {}
            flags: dict[int, np.ndarray] | None = {} if labeled else None
            for name, plate in self._plate_names.items():
                md = name_to_set.get(name)
                if md is None:
                    continue        # mapped asset absent from this frame
                arr = np.asarray(md.marker_pos_list, dtype=float).reshape(-1, 3)
                markers[plate] = arr
                if flags is not None:
                    flags[plate] = self._plate_flags(arr, by_model.get(plate, {}))
            self._marker_scratch = markers
            # flags stays None when no labeled markers arrived: *unknown*, an
            # explicit recorded regime, never silently "all-present" (D4/§5).
            self._flags_scratch = flags
        except Exception as exc:  # never let the SDK thread die on bad data
            self._note_error(f"mocap_data: {exc}")

    def _derive_mapping(self, sets, by_model, mocap_data) -> dict | None:
        """Marker-set name -> plate index for this frame, or ``None`` if the
        frame cannot support a derivation.

        **Primary** (labeled markers streamed — confirmed live 2026-08-11):
        a set is bound to the one model whose tracked labeled positions all
        appear, at *exact* coordinates, among the set's positions.  Both sides
        of the comparison were unpacked from the same UDP packet with the same
        ``Vector3.unpack``, so exact float equality is the correct test and no
        fuzz radius can mis-bind neighbouring plates 47 mm apart.  Only
        *tracked* labeled entries are compared: an occluded/model-solved entry
        may carry a model-filled position that legitimately differs from
        whatever the server put in the set.

        **Fallback** (no labeled markers): centroid matching, restricted to
        exactly-4-marker sets, ``b"all"`` excluded, winner < 20 mm from a
        streamed arm pivot with the runner-up at least twice as far (§3).
        """
        if by_model:
            tracked_pts = {}
            for plate, entries in by_model.items():
                pts = [tuple(lm.pos) for lm in entries.values()
                       if not (lm.param & (_PARAM_OCCLUDED | _PARAM_MODEL_SOLVED))]
                if pts:
                    tracked_pts[plate] = pts
            if not tracked_pts:
                return None
            mapping: dict = {}
            for md in sets:
                name = md.model_name
                if name == _ALL_SET_NAME:
                    continue    # superset of everything; would match every plate
                set_pts = set(map(tuple, md.marker_pos_list))
                hits = [plate for plate, pts in tracked_pts.items()
                        if all(p in set_pts for p in pts)]
                if len(hits) == 1 and name not in mapping:
                    mapping[name] = hits[0]
            return mapping or None

        rbd = getattr(mocap_data, "rigid_body_data", None)
        bodies = [] if rbd is None else rbd.rigid_body_list
        pivots = {}
        for rb in bodies:
            plate = rb.id_num - self.rb_id_base
            if 0 <= plate < self.n_bodies:
                pivots[plate] = np.asarray(rb.pos, dtype=float)
        if not pivots:
            return None
        mapping = {}
        for md in sets:
            if md.model_name == _ALL_SET_NAME or len(md.marker_pos_list) != 4:
                continue
            centroid = np.asarray(md.marker_pos_list, dtype=float).mean(axis=0)
            ranked = sorted((float(np.linalg.norm(centroid - p)), plate)
                            for plate, p in pivots.items())
            win_d, win_plate = ranked[0]
            if win_d < _CENTROID_WIN_M and (
                    len(ranked) == 1
                    or ranked[1][0] >= _CENTROID_RUNNER_UP_FACTOR * win_d):
                mapping[md.model_name] = win_plate
        return mapping or None

    def _tally_param(self, param: int) -> bool:
        """Count a labeled entry's param bits; True when it counts as tracked.

        Tracked means the occluded and model-solved bits are clear (design
        §3/§5 — a model-filled position must never launder the rigid-body
        solve into "independent" marker evidence).  Point-cloud-solved counts
        as tracked.
        """
        if param & _PARAM_OCCLUDED:
            self._occluded_seen += 1
        if param & _PARAM_POINT_CLOUD_SOLVED:
            self._point_cloud_solved_seen += 1
        if param & _PARAM_MODEL_SOLVED:
            self._model_solved_seen += 1
        return not (param & (_PARAM_OCCLUDED | _PARAM_MODEL_SOLVED))

    def _plate_flags(self, arr: np.ndarray, labeled_by_id: dict) -> np.ndarray:
        """Per-marker tracked flags for one plate's ``(m, 3)`` asset-order array.

        bit0 = tracked-this-frame: a labeled entry exists for the marker and its
        occluded and model-solved bits are clear (design §3/§5 — a model-filled
        position must never launder the rigid-body solve into "independent"
        marker evidence).  Placement uses the assumed ``marker_id - 1`` index
        when it holds *and* the positions agree exactly; otherwise per-frame
        position matching, with the id_corr tallies recording every miss so the
        probe can judge the assumption empirically.
        """
        n = arr.shape[0]
        out = np.zeros(n, dtype=np.uint8)
        for marker_id, lm in labeled_by_id.items():
            if not self._tally_param(lm.param):
                continue        # excluded from inference; its flag stays 0
            pos = np.asarray(lm.pos, dtype=float)
            idx = marker_id - 1
            if self.marker_index_by_id and 0 <= idx < n and np.array_equal(arr[idx], pos):
                self._id_corr_ok += 1
                out[idx] = 1
                continue
            if self.marker_index_by_id:
                self._id_corr_bad += 1    # the assumption testably failed here
            hit = np.flatnonzero((arr == pos).all(axis=1))
            if hit.size:
                out[int(hit[0])] = 1
        return out

    def _solve_q(self, homos, markers, flags) -> np.ndarray | None:
        """This frame's ``q``, from Motive's **streamed rigid-body poses**.

        The one extension point in the receive path, and the reason it exists
        is a documented hazard: Motive's rigid-body pivot is defined by its
        auto-refine, which moves it during a session, while the marker
        positions stay true.  Anything that needs the marker-derived pose
        overrides this — see :class:`UMArm_MOCAP.marker_mocap.MarkerMocap`,
        which registers each plate's rest template onto the observed markers
        instead.

        Kept as a method rather than a constructor callback so the override
        sees the same three things this one does — the streamed poses *and*
        this frame's markers and flags — and so a subclass can fall back to
        this implementation by calling ``super()._solve_q(...)``.
        """
        return mocap_to_q(homos)

    def _on_new_frame(self, data_dict) -> None:
        """End of one frame: convert, publish, record, notify."""
        try:
            homos = self._incoming.copy()
            # Commit contract for the marker scratch (design §3): *consume and
            # clear* first, so the next frame starts from provably-nothing and a
            # frame whose marker data did not arrive records None, never a stale
            # copy (review finding int-4).  Deep copies (dict + arrays) are made
            # here, before taking _lock, mirroring the homos.copy() above.
            markers, self._marker_scratch = self._marker_scratch, None
            flags, self._flags_scratch = self._flags_scratch, None
            epoch = self._mapping_epoch
            if markers is not None:
                markers = {p: a.copy() for p, a in markers.items()}
            if flags is not None:
                flags = {p: a.copy() for p, a in flags.items()}
            # The (7, 4, 4) slice mocap_to_q consumes, recorded raw: the fkine
            # benchmark's streamed-base regime reads base orientation from here
            # and nowhere else (review finding int-0).  Rows a frame did not
            # mention keep their previous pose — the same inherited hold-last
            # behaviour the module docstring documents for q.
            streamed_poses = homos[0:mc.N_USED_RIGID_BODIES].copy()
            q = self._solve_q(homos, markers, flags)
            now_mono = time.monotonic()
            now_wall = time.time()
            frame_no = int(data_dict.get("frame_number", -1)) if data_dict else -1

            with self._lock:
                self._frames += 1
                self._frame_number = frame_no
                self._last_mono = now_mono
                self._last_wall = now_wall
                self._frame_times.append(now_mono)
                self._homos = homos
                # The marker ring records EVERY frame, valid q or not: the
                # q=None frames are exactly the dropout-rich frames the
                # marker-frame benchmark studies (review findings int-8/ops-7).
                self._marker_ring.append(
                    (now_mono, frame_no, epoch, markers, flags, streamed_poses))
                if q is not None:
                    self._valid_frames += 1
                    self._last_q_mono = now_mono
                    self._q = q
                    self._ring.append(
                        (now_mono, frame_no, q, homos[list(mc.U_JOINT_INDICES), 0:3, 3]))
                    self._seq += 1
                    # Only a valid frame wakes wait_fresh: a caller asking for a
                    # fresh q must not be handed a frame that had none.
                    self._new_frame.notify_all()

            # A copy, for the same reason get_q() hands out one: `q` is still the
            # live object behind `_q` and behind this frame's ring entry, so an
            # in-place callback (`lambda q: q -= q0` is the obvious calibration
            # idiom) would rewrite the published value *and* silently rewrite
            # recorded history that snapshot_window has already promised.
            if q is not None and self.on_q is not None:
                self.on_q(q.copy())
        except Exception as exc:
            self._note_error(f"new_frame: {exc}")

    def _note_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def get_q(self) -> np.ndarray | None:
        """Latest joint vector (radians), or ``None`` before the first valid frame.

        A copy, so the caller cannot be handed an array the SDK thread is about
        to replace.  Note this says nothing about *age* — pair it with
        :meth:`get_state` (or use :meth:`wait_fresh`) in any loop that acts on it.
        """
        with self._lock:
            return None if self._q is None else self._q.copy()

    def latest_sample(self):
        """``(t_mono, frame_no, q, homos, seq)`` of the newest valid frame.

        One lock acquisition, no deque copy and no health arithmetic, because a
        150 Hz recorder calls this on the CAN cycle thread and everything it
        does there is subtracted from a 6.67 ms budget.  :meth:`get_q` paired
        with :meth:`get_state` answers the same question and costs two
        acquisitions plus a copy of the frame-time deque per call.

        ``q`` and ``homos`` are the receiver's own arrays, **not copies** -- the
        caller must treat them as read-only.  That is safe here and nowhere
        else: the SDK thread replaces the reference rather than mutating the
        array (see :meth:`_on_new_frame`), so a reader holding the old one keeps
        a consistent frame.  ``seq`` advances once per valid frame, so a
        recorder can tell a repeated sample from a fresh one without comparing
        timestamps.

        Returns ``(None, None, None, None, seq)`` before the first valid frame.
        """
        with self._lock:
            if self._q is None:
                return (None, None, None, None, self._seq)
            return (self._last_q_mono, self._frame_number, self._q,
                    self._homos, self._seq)

    def get_homos(self) -> np.ndarray | None:
        """Latest ``(9, 4, 4)`` plate poses in the spatial frame, or ``None``."""
        with self._lock:
            return None if self._homos is None else self._homos.copy()

    def get_state(self) -> MocapState:
        """Stream health, evaluated against the clock at call time."""
        with self._lock:
            last_mono = self._last_mono
            last_q_mono = self._last_q_mono
            times = list(self._frame_times)
            state = dict(
                running=self._client is not None,
                frames=self._frames,
                valid_frames=self._valid_frames,
                frame_number=self._frame_number,
                last_frame_wall=self._last_wall,
                last_frame_mono=last_mono,
                last_q_mono=last_q_mono,
                ring_len=len(self._ring),
                last_error=self._last_error,
            )
        fps = 0.0
        if len(times) >= 2 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        now = time.monotonic()
        stale = last_mono is None or (now - last_mono) > mc.STALE_AFTER_S
        q_stale = last_q_mono is None or (now - last_q_mono) > mc.STALE_AFTER_S
        return MocapState(fps=fps, stale=stale, q_stale=q_stale, **state)

    def wait_fresh(self, timeout: float = 1.0) -> np.ndarray | None:
        """Block for a ``q`` from a frame that arrives *after* this call.

        Returns the new ``q``, or ``None`` on timeout.  This is the correct way
        to sample the arm after commanding it: :meth:`get_q` would happily
        return a pre-command value, and at 120 Hz that is up to 8 ms of lie —
        more if the stream has stalled, which is the case that matters.

        A stopped or never-started receiver simply times out; the timeout is the
        only exit, so keep it short in an interactive loop.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            target = self._seq + 1
            while self._seq < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._new_frame.wait(remaining)
            return None if self._q is None else self._q.copy()

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    def snapshot_window(self, t0: float | None = None,
                        t1: float | None = None) -> MocapWindow:
        """Recorded samples with ``t0 <= t_mono <= t1`` (both ``time.monotonic()``).

        Either bound may be ``None`` for "open". Samples older than the ring's
        depth (``mocap_constants.RING_SECONDS``) are gone — ask for a window
        while it is still in living memory.
        """
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        with self._lock:
            rows = [r for r in self._ring if lo <= r[0] <= hi]
        n = len(rows)
        if n == 0:
            return MocapWindow(t=np.empty(0), frame_no=np.empty(0, dtype=np.int64),
                               q=np.empty((0, mc.NUM_JOINTS)), u=np.empty((0, 6, 3)))
        return MocapWindow(
            t=np.array([r[0] for r in rows], dtype=float),
            frame_no=np.array([r[1] for r in rows], dtype=np.int64),
            q=np.stack([r[2] for r in rows]),
            u=np.stack([r[3] for r in rows]),
        )

    def snapshot_marker_window(self, t0: float | None = None,
                               t1: float | None = None) -> MarkerWindow:
        """Recorded marker samples with ``t0 <= t_mono <= t1``, like
        :meth:`snapshot_window` but from the marker ring.

        Everything handed out is a copy (arrays stacked or ``.copy()``-ed, dicts
        rebuilt), for the same reason :meth:`get_q` copies: a caller mutating a
        returned array must not rewrite recorded history.  Align with a
        :class:`MocapWindow` offline by ``frame_no`` — the two rings advance
        together but the q ring skips degenerate frames.
        """
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        with self._lock:
            rows = [r for r in self._marker_ring if lo <= r[0] <= hi]
        if not rows:
            return MarkerWindow(
                t=np.empty(0), frame_no=np.empty(0, dtype=np.int64),
                mapping_epoch=np.empty(0, dtype=np.int64), markers=(), flags=(),
                streamed_poses=np.empty((0, mc.N_USED_RIGID_BODIES, 4, 4)))
        return MarkerWindow(
            t=np.array([r[0] for r in rows], dtype=float),
            frame_no=np.array([r[1] for r in rows], dtype=np.int64),
            mapping_epoch=np.array([r[2] for r in rows], dtype=np.int64),
            markers=tuple(None if r[3] is None
                          else {p: a.copy() for p, a in r[3].items()} for r in rows),
            flags=tuple(None if r[4] is None
                        else {p: a.copy() for p, a in r[4].items()} for r in rows),
            streamed_poses=np.stack([r[5] for r in rows]),
        )

    def marker_health(self) -> MarkerHealth:
        """Marker-transport bookkeeping for the probe (design §6).

        Lock-free by the same single-writer argument as :attr:`_incoming`: every
        field is either a monotonically increasing int written only by the SDK
        thread or a dict that is swapped wholesale, so a read is always
        internally consistent even if it races a frame by one count.
        """
        return MarkerHealth(
            mocap_data_frames=self._mocap_data_frames,
            labeled_frames=self._labeled_frames,
            last_marker_set_count=self._last_marker_set_count,
            last_labeled_marker_count=self._last_labeled_marker_count,
            plate_names=dict(self._plate_names),
            mapping_epoch=self._mapping_epoch,
            id_corr_ok=self._id_corr_ok,
            id_corr_bad=self._id_corr_bad,
            occluded_seen=self._occluded_seen,
            point_cloud_solved_seen=self._point_cloud_solved_seen,
            model_solved_seen=self._model_solved_seen,
            marker_ring_len=len(self._marker_ring),
        )

    def resize_rings(self, ring_capacity: int | None = None,
                     marker_ring_capacity: int | None = None) -> None:
        """Re-bound one or both rings in place, keeping the newest entries.

        Exists for the probe and campaign (design §6/§7, review finding
        ops-10): capacities must come from the *measured* frame rate, which is
        only known after a warm-up — so the rings start at a nominal size and
        get re-bounded once the rate is real.  A ``deque`` cannot change
        ``maxlen``, so each ring is rebuilt; construction from the old ring
        keeps the rightmost (newest) entries when shrinking.
        """
        with self._lock:
            if ring_capacity is not None:
                self._ring = deque(self._ring, maxlen=int(ring_capacity))
            if marker_ring_capacity is not None:
                self._marker_ring = deque(self._marker_ring,
                                          maxlen=int(marker_ring_capacity))

    def clear_history(self) -> None:
        """Drop both ring buffers, so the next window starts from a known point."""
        with self._lock:
            self._ring.clear()
            self._marker_ring.clear()

    def capture_rest(self, seconds: float = 3.0,
                     timeout: float = 2.0) -> RestCapture | None:
        """Average ``q`` over ``seconds`` of (nominally) held-still arm.

        Returns ``None`` rather than a reassuring-looking answer whenever the
        capture did not actually happen:

        * no fresh frame within ``timeout`` (the stream is not running);
        * the stream went :attr:`MocapState.stale` partway through, which aborts
          the wait immediately instead of sitting out the full ``seconds``;
        * fewer than two samples landed in the window, so there is no spread to
          report;
        * the samples span less than :data:`MIN_REST_COVERAGE` of ``seconds``,
          i.e. the stream was alive but delivering far too little to average.

        That list is the whole point of the method's return type.  A joint zero
        is adopted from ``mean`` and a noise floor from ``sd``, so the one thing
        this must never do is report ``sd = 0`` from a single sample of a stream
        that died three seconds ago — the most reassuring possible output for the
        worst possible input.

        The window is taken from the ring buffer rather than accumulated
        separately, so the raw samples behind the mean stay available for
        inspection via :attr:`RestCapture.window`.
        """
        if seconds <= 0.0:
            raise ValueError("seconds must be > 0")
        # Measured against *this* receiver's ring, not the default constant, so a
        # caller who shrank ring_capacity is told the truth rather than handed a
        # mean over whatever survived.  The ring is bounded by sample *count*, so
        # converting it to seconds needs a rate: prefer the measured one, because
        # a Prime-camera project at 240 Hz fills 2400 samples in 10 s, not 20.
        rate = self.get_state().fps or mc.NOMINAL_RATE_HZ
        ring_seconds = (self._ring.maxlen or 0) / rate
        if seconds > ring_seconds:
            raise ValueError(
                f"seconds={seconds:g} exceeds this receiver's {ring_seconds:g} s of "
                f"history ({self._ring.maxlen} samples at {rate:g} Hz); "
                "raise ring_capacity to record longer")
        if self.wait_fresh(timeout=timeout) is None:
            return None
        t0 = time.monotonic()
        # Sleep in slices so a stalled stream is noticed rather than waited out —
        # which means actually looking at the clock the stream is judged against,
        # not just waking up often.
        end = t0 + seconds
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0.0:
                break
            if self.get_state().stale:
                return None
            time.sleep(min(0.05, remaining))
        window = self.snapshot_window(t0, time.monotonic())
        # Two gates, and the second currently subsumes the first: a one-sample
        # window has `duration == 0.0` by MocapWindow's definition, so it fails
        # coverage too.  The count gate stays because it is what makes `ddof=1`
        # below well defined at the point of use, rather than by a chain of
        # reasoning through another class's property — delete the *coverage* gate
        # and a sparse-but-live stream gets through; delete *this* one and the sd
        # of a single sample becomes NaN instead of an honest refusal.
        if len(window) < 2:
            return None
        if window.duration < MIN_REST_COVERAGE * seconds:
            return None
        return RestCapture(
            n=len(window),
            duration=window.duration,
            mean=window.q.mean(axis=0),
            sd=window.q.std(axis=0, ddof=1),
            ptp=np.ptp(window.q, axis=0),
            window=window,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @staticmethod
    def _import_natnet():
        """Import the vendored SDK, adding its folder to ``sys.path`` first.

        The NaturalPoint files use flat imports of each other (``import
        MoCapData``), so their directory has to be importable as a top-level
        path — that is a property of the vendored third-party code, not a choice.
        Deferred to :meth:`start` so that merely importing this module neither
        pollutes ``sys.path`` with three generic module names nor executes 150 kB
        of SDK code in a process that only wanted the math.
        """
        sdk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natnet_sdk")
        if sdk_dir not in sys.path:
            sys.path.insert(0, sdk_dir)
        from NatNetClient import NatNetClient  # noqa: PLC0415  (deliberate, see above)
        return NatNetClient

    def start(self):
        """Connect to Motive and begin streaming.  Raises if the client refuses.

        ``NatNetClient.run()`` returning false usually means the client IP is not
        the interface on the camera network, or Motive is not streaming — both
        of which are silent failures if the return value is ignored.
        """
        if self._client is not None:
            return self._client
        natnet_client_cls = self._import_natnet()
        client = natnet_client_cls()
        client.set_client_address(self.client_ip)
        client.set_server_address(self.server_ip)
        client.set_use_multicast(self.use_multicast)
        client.new_frame_listener = self._on_new_frame
        client.rigid_body_listener = self._on_rigid_body
        # Vendored patch 1 (natnet_sdk/PROVENANCE.md).  On an unpatched SDK
        # drop this assignment still succeeds but the listener never fires —
        # the marker ring then honestly records None for every frame.
        client.mocap_data_listener = self._on_mocap_data
        if not client.run():
            raise RuntimeError(
                f"NatNet client would not start (server={self.server_ip}, "
                f"client={self.client_ip}, multicast={self.use_multicast})")
        self._client = client
        return client

    def stop(self) -> None:
        """Shut the SDK threads down.  Safe to call twice, and safe if start failed."""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.shutdown()
            except Exception as exc:  # a failed shutdown must not mask the caller's error
                self._note_error(f"shutdown: {exc}")

    def __enter__(self) -> "MocapRx":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
