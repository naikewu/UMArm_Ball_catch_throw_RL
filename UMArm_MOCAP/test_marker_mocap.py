"""Tests for the marker-derived receiver: ``q`` from markers, never from pivots.

No network, no SDK, no Motive.  :class:`MarkerMocap` extends
:class:`~UMArm_MOCAP.mocap_rx.MocapRx` at exactly one method, so the tests call
that seam (``_solve_q``) and the frame commit (``_on_new_frame``) directly — the
same way ``test_mocap_rx.py`` calls the SDK's listeners, and for the same
reason: the part we cannot test is Motive's UDP, which is the part we did not
write.  ``start()`` is never called anywhere in this file.

Poses are *forward-constructed*, which is what makes the round trip mean
something.  ``UMArm_KINEMATICS.plate_transforms`` turns a known ``q`` into six
body frames; four markers per plate are placed on **radial arms of unequal
length** in that plate's bracket frame (design §2 — the real brackets are not
parallelograms, their diagonals differ by 5.5-15 mm, and a rectangle would let a
registration bug hide behind the symmetry) and carried into the world.  Locks
are minted at one ``q``, markers generated at another, and ``_solve_q`` has to
return the second ``q``.

The other half of the file is the requirement the module exists for, in the
operator's words: *the mocap centre position definition may change during the
auto-refine process, but the marker position stays true.*  So the solve tests
feed ``_solve_q`` streamed poses that are wrong — pivots displaced by 2 mm, or
nothing but identities — and demand the same ``q`` to the last bit.

Run:  ``python -m pytest UMArm_MOCAP/test_marker_mocap.py -q``
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from UMArm_KINEMATICS.fkine import plate_transforms

from . import mocap_constants as mc
from . import mocap_rx as rx_module
from .marker_frame import FAMILY_PHI_RAD, PlateLock, compute_plate_lock
from .marker_mocap import (REQUIRED_PLATES, MarkerMocap, compare_to_streamed,
                           load_locks, mint_locks, save_locks)
from .mocap_rx import MarkerWindow, MocapRx
from .mocap_to_q import mocap_to_q

#: Round-trip budget.  The whole path is noiseless here, so the only error is
#: the Kabsch fit's cancellation against ~1 m world coordinates on a ~0.1 m
#: bracket; measured worst case over these fixtures is 3.9e-15, and 1e-11 keeps
#: orders of headroom while still catching any real defect (a wrong candidate
#: axis or mounting model moves ``q`` by whole degrees).
TOL = 1e-11

#: Family angles in plate order, read from the module's own table so the
#: synthetic markers are built with the same convention the solve reads back.
FAMILY_PHIS = tuple(FAMILY_PHI_RAD[p] for p in range(6))

#: Radial-arm radii (metres) of the synthetic brackets — unequal on purpose,
#: like the hardware (probe_20260811_live2: diagonal lengths 129.7-205.4 mm
#: with 5.5-15 mm of asymmetry, diagonal midpoints 5.5-12.8 mm apart).
ARM_RADII_M = (0.060, 0.075, 0.068, 0.090)

#: The pose the locks are minted at.  Deliberately not zeros: a straight arm
#: gives every plate the same orientation, and a solve that confused two plates
#: would still round-trip.
REST_Q = np.array([0.04, -0.03, 0.02, 0.05, -0.02, 0.06,
                   0.03, -0.04, 0.05, 0.02, -0.06, 0.01])

#: The pose the markers are generated at — far from ``REST_Q`` on every joint,
#: including q10/q11, which move no centre at all and live only in plate 5's
#: orientation.
TEST_Q = np.array([0.30, -0.25, 0.18, 0.22, -0.35, 0.28,
                   -0.20, 0.15, 0.24, -0.30, 0.19, -0.26])


# --------------------------------------------------------------------------
# Forward constructors: known q -> plate frames -> markers
# --------------------------------------------------------------------------


def rot_z(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def volume_base(rng: np.random.Generator) -> np.ndarray:
    """A random robot -> mocap-volume transform.

    Every test uses one, because nothing in this path may depend on the world
    axes: the volume gets re-created between sessions, so a solve that quietly
    leaned on world z would pass on the bench and fail after a recalibration.
    """
    a, r = np.linalg.qr(rng.normal(size=(3, 3)))
    a = a * np.sign(np.diag(r))
    if np.linalg.det(a) < 0.0:
        a[:, 0] = -a[:, 0]
    out = np.eye(4)
    out[0:3, 0:3] = a
    out[0:3, 3] = rng.uniform(-2.0, 2.0, 3)
    return out


def arm_corners(radii=ARM_RADII_M) -> np.ndarray:
    """``(4, 3)`` bracket-frame markers on radial arms of unequal length.

    Arms at 45/135/225/315 deg, i.e. the diagonals sit 45 deg off the bracket
    axes (design §2).  Label ``i``'s opposite is ``i + 2``, so both diagonal
    *lines* pass exactly through the u-joint centre however unequal the radii —
    while the diagonal *midpoints* do not, which is the asymmetry the origin
    definition exists to defeat.
    """
    return np.array([[r * math.cos(math.radians(a)),
                      r * math.sin(math.radians(a)), 0.0]
                     for r, a in zip(radii, (45.0, 135.0, 225.0, 315.0))])


def arm_plates(q, base: np.ndarray) -> np.ndarray:
    """``(6, 4, 4)`` plate body frames of pose ``q``, in the mocap volume."""
    return base @ plate_transforms(np.asarray(q, dtype=float))


def markers_from_pose(pose: np.ndarray, phi: float) -> np.ndarray:
    """The design §7 recipe: ``markers = T_p @ Rz(phi_p) @ corners``."""
    r = pose[0:3, 0:3] @ rot_z(phi)
    return arm_corners() @ r.T + pose[0:3, 3]


def arm_markers(q, base: np.ndarray) -> dict:
    """One frame's marker dict, ``plate -> (4, 3)``, as the ring records it."""
    plates = arm_plates(q, base)
    return {p: markers_from_pose(plates[p], FAMILY_PHIS[p]) for p in range(6)}


def all_tracked() -> dict:
    """Flags for a frame in which Motive saw every marker of every plate."""
    return {p: np.ones(4, dtype=np.uint8) for p in range(6)}


def arm_locks(base: np.ndarray, q_rest=REST_Q) -> dict:
    """Locks for plates 0..5 from a still rest window at ``q_rest``.

    ``u_up`` is arm-derived (plate 0's centre minus plate 5's) and ``x_mode``
    is ``"streamed"``, both exactly as :func:`mint_locks` does it, so these
    locks are the ones a session would really be holding.
    """
    plates = arm_plates(q_rest, base)
    markers = arm_markers(q_rest, base)
    u_up = plates[0][0:3, 3] - plates[5][0:3, 3]
    return {p: compute_plate_lock(np.stack([markers[p]] * 3), plate=p, u_up=u_up,
                                  streamed_rot=plates[p][0:3, 0:3],
                                  x_mode="streamed")
            for p in range(6)}


def streamed_homos(plates: np.ndarray | None = None,
                   pivot_shift: np.ndarray | None = None) -> np.ndarray:
    """The ``(9, 4, 4)`` pose array the SDK thread hands ``_solve_q``.

    ``plates=None`` leaves every row at the identity — Motive solving nothing,
    the state in which the streamed path has no answer at all.  ``pivot_shift``
    is ``(6, 3)`` of per-plate origin displacement: Motive's auto-refine moving
    the rigid-body pivots without telling any client.
    """
    homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
    if plates is not None:
        homos[0:6] = plates
        if pivot_shift is not None:
            homos[0:6, 0:3, 3] += pivot_shift
    return homos


def rest_window(base: np.ndarray, q=REST_Q, n: int = 30, drift_m: float = 0.0,
                plates_seen=tuple(range(6))) -> MarkerWindow:
    """A recorded marker window of the arm holding ``q``, as the ring hands it out.

    ``drift_m`` slides every plate along the volume's x over the window — the
    arm creeping, or an operator's hand on it — which is what the stillness
    gate has to catch.  A slide keeps each individual frame's marker geometry
    perfectly valid, so nothing but the stillness evidence can refuse it.
    """
    plates = arm_plates(q, base)
    markers = arm_markers(q, base)
    poses = streamed_homos(plates)[0:mc.N_USED_RIGID_BODIES]
    ramp = np.zeros(n) if n < 2 else np.linspace(-drift_m, drift_m, n)
    frames, flags = [], []
    for i in range(n):
        step = ramp[i] * base[0:3, 0]
        frames.append({p: markers[p] + step for p in plates_seen})
        flags.append({p: np.ones(4, dtype=np.uint8) for p in plates_seen})
    return MarkerWindow(
        t=np.arange(n, dtype=float) / mc.NOMINAL_RATE_HZ,
        frame_no=np.arange(n, dtype=np.int64),
        mapping_epoch=np.ones(n, dtype=np.int64),
        markers=tuple(frames), flags=tuple(flags),
        streamed_poses=np.tile(poses, (n, 1, 1, 1)))


class CannedRx:
    """A receiver stand-in whose marker window is already recorded.

    :func:`mint_locks` reads exactly one thing off a receiver —
    ``snapshot_marker_window`` — and spends the rest of the window asleep, so a
    fake that hands back a prepared window exercises the whole minting path
    (stillness gate, arm-derived up, per-plate lock, report) with no stream, no
    feeder thread, and no wall-clock race to lose on a busy Windows box.
    """

    def __init__(self, window: MarkerWindow) -> None:
        self.window = window
        self.t0_asked = None

    def snapshot_marker_window(self, t0=None, t1=None) -> MarkerWindow:
        self.t0_asked = t0
        return self.window


# --------------------------------------------------------------------------
# 1. Construction
# --------------------------------------------------------------------------


class TestConstruction:
    def test_a_missing_plate_lock_is_refused_and_named(self):
        """Every plate in ``REQUIRED_PLATES`` or nothing.

        ``infer_all`` returns an identity row for a plate it has no lock for,
        and ``q_from_frames`` would happily reconstruct joint angles out of
        that identity — a confident, wrong ``q`` for the joints around the
        missing plate.  The refusal has to name the plates, because the
        operator's next move is to re-run the lock capture for exactly those.
        """
        locks = arm_locks(volume_base(np.random.default_rng(1)))
        del locks[2]
        del locks[5]
        with pytest.raises(ValueError) as excinfo:
            MarkerMocap(locks)
        message = str(excinfo.value)
        assert "[2, 5]" in message
        assert "mint_locks" in message

    def test_locks_keyed_by_string_are_accepted(self):
        """``load_locks`` and the probe both hand back JSON-derived keys."""
        locks = arm_locks(volume_base(np.random.default_rng(2)))
        rx = MarkerMocap({str(p): lk for p, lk in locks.items()})
        assert sorted(rx.locks) == list(REQUIRED_PLATES)

    def test_the_fallback_is_off_unless_asked_for(self):
        """Falling back would change which definition of "the arm" the loop is
        tracking, mid-move — worse than a gap, so it is opt-in and named."""
        rx = MarkerMocap(arm_locks(volume_base(np.random.default_rng(3))))
        assert rx.fallback_to_streamed is False
        assert rx.get_q() is None and rx.solve_stats().frames == 0


# --------------------------------------------------------------------------
# 2. The round trip — the whole point of the module
# --------------------------------------------------------------------------


class TestMarkerRoundTrip:
    def test_markers_from_a_new_pose_recover_that_pose(self):
        """Locks minted at ``REST_Q``, markers generated at ``TEST_Q``, and
        ``_solve_q`` returns ``TEST_Q``.

        Everything else in this file is a variation on this line: if the
        registration, the family-angle convention or the mounting model were
        wrong, this is where it shows up, in radians rather than in a
        plausible-looking session.
        """
        base = volume_base(np.random.default_rng(11))
        rx = MarkerMocap(arm_locks(base))
        q = rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                        arm_markers(TEST_Q, base), all_tracked())
        assert q is not None
        assert np.max(np.abs(q - TEST_Q)) < TOL

    @pytest.mark.parametrize("seed", [17, 23, 29])
    def test_any_pose_round_trips_from_any_volume_orientation(self, seed):
        """A random volume transform and a random pose, per seed.

        Both halves matter: the volume is re-created between sessions (so no
        world axis may be load bearing), and the single fixed ``TEST_Q`` above
        cannot rule out a joint-order or sign coincidence.
        """
        rng = np.random.default_rng(seed)
        base = volume_base(rng)
        rx = MarkerMocap(arm_locks(base))
        for _ in range(5):
            q_test = rng.uniform(-0.45, 0.45, 12)
            q = rx._solve_q(streamed_homos(arm_plates(q_test, base)),
                            arm_markers(q_test, base), all_tracked())
            assert q is not None
            assert np.max(np.abs(q - q_test)) < TOL

    def test_flags_unknown_still_solves(self):
        """``flags=None`` is the labeled-markers-off regime (``mocap_rx`` D4):
        all four markers are assumed usable, and the solve must proceed rather
        than treat "unknown" as "absent"."""
        base = volume_base(np.random.default_rng(13))
        rx = MarkerMocap(arm_locks(base))
        q = rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                        arm_markers(TEST_Q, base), None)
        assert q is not None and np.max(np.abs(q - TEST_Q)) < TOL

    def test_one_dropped_marker_per_plate_still_solves(self):
        """Three markers determine a plate exactly (design §2), and dropouts
        are the normal condition of a real volume — a receiver that needed all
        four would spend a session mostly stale."""
        base = volume_base(np.random.default_rng(19))
        rx = MarkerMocap(arm_locks(base))
        flags = all_tracked()
        for plate in range(6):
            flags[plate][plate % 4] = 0
        q = rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                        arm_markers(TEST_Q, base), flags)
        assert q is not None and np.max(np.abs(q - TEST_Q)) < TOL


# --------------------------------------------------------------------------
# 3. The streamed poses are not used  (the operator's requirement)
# --------------------------------------------------------------------------


class TestStreamedPosesAreIgnored:
    def test_a_moved_pivot_does_not_move_q(self):
        """Motive's auto-refine displaces every pivot by 2 mm mid-session.

        Nothing in a control loop can tell that apart from the arm having
        moved, so the loop drives the arm to cancel a *definition change*.  The
        marker ``q`` here is bit-identical before and after the displacement,
        while the streamed reconstruction of the same frame moves 0.055 rad
        (3.2 deg) — which is what the displacement costs the old path.
        """
        base = volume_base(np.random.default_rng(31))
        rng = np.random.default_rng(32)
        plates = arm_plates(TEST_Q, base)
        markers, flags = arm_markers(TEST_Q, base), all_tracked()
        shift = rng.normal(size=(6, 3))
        shift *= 0.002 / np.linalg.norm(shift, axis=1)[:, None]

        rx = MarkerMocap(arm_locks(base))
        q_true = rx._solve_q(streamed_homos(plates), markers, flags)
        q_moved = rx._solve_q(streamed_homos(plates, shift), markers, flags)
        assert np.array_equal(q_true, q_moved)
        assert np.max(np.abs(q_true - TEST_Q)) < TOL

        # The displacement is genuinely poisonous — otherwise the test above
        # would pass for the wrong reason.
        streamed_true = mocap_to_q(streamed_homos(plates))
        streamed_moved = mocap_to_q(streamed_homos(plates, shift))
        assert np.max(np.abs(streamed_true - streamed_moved)) > 0.02

    def test_q_is_solved_even_when_the_streamed_poses_are_useless(self):
        """Every rigid body unsolved (all rows still identity) — the streamed
        path returns ``None`` outright, and the markers alone still carry the
        pose.  This is the strongest form of the claim: the homos argument
        contributes nothing to ``q``."""
        base = volume_base(np.random.default_rng(37))
        rx = MarkerMocap(arm_locks(base))
        junk = streamed_homos()
        assert MocapRx._solve_q(rx, junk, None, None) is None

        q = rx._solve_q(junk, arm_markers(TEST_Q, base), all_tracked())
        assert q is not None and np.max(np.abs(q - TEST_Q)) < TOL

    def test_the_override_is_what_the_receiver_publishes(self):
        """Drop-in check: ``BusRuntime`` calls ``get_q``, never ``_solve_q``.

        Driving the real frame commit with junk in the pose array and markers
        in the scratch pins that the published ``q``, the ring and the health
        flags all come from the marker path — the override is wired in, not
        merely callable.  The scratch is filled directly, standing in for the
        SDK thread: how ``_on_mocap_data`` fills it is ``test_mocap_rx``'s
        subject, and this frame's ``q`` must not depend on it.
        """
        base = volume_base(np.random.default_rng(41))
        rx = MarkerMocap(arm_locks(base))
        rx._marker_scratch = arm_markers(TEST_Q, base)
        rx._flags_scratch = all_tracked()
        rx._on_new_frame({"frame_number": 4242})

        state = rx.get_state()
        assert state.frames == 1 and state.valid_frames == 1
        assert state.frame_number == 4242 and not state.q_stale
        assert state.last_error is None
        assert np.max(np.abs(rx.get_q() - TEST_Q)) < TOL

    def test_both_pose_opinions_are_kept_for_the_same_frame(self):
        """``get_marker_poses``/``get_streamed_poses`` are the comparison pair
        the preflight reports, so they must come from one frame and be copies —
        a visualiser mutating what it drew must not rewrite what the controller
        used."""
        base = volume_base(np.random.default_rng(43))
        plates = arm_plates(TEST_Q, base)
        rx = MarkerMocap(arm_locks(base))
        rx._solve_q(streamed_homos(plates), arm_markers(TEST_Q, base),
                    all_tracked())

        inferred, streamed = rx.get_marker_poses(), rx.get_streamed_poses()
        assert inferred.shape == (mc.N_USED_RIGID_BODIES, 4, 4)
        assert np.allclose(streamed[0:6], plates)
        inferred[:] = 0.0
        assert not np.array_equal(rx.get_marker_poses(), inferred)

    def test_compare_to_streamed_reports_the_disagreement(self):
        """The change detector, not a pass/fail gate (0.66-4.24 deg and 0.3-1.5
        mm were the *expected* disagreements on 2026-08-11).  Feeding it a
        Motive that agrees perfectly must therefore report zeros — which also
        pins the ``Rz(-phi)`` family correction it applies before comparing."""
        base = volume_base(np.random.default_rng(47))
        rx = MarkerMocap(arm_locks(base))
        assert compare_to_streamed(rx) is None      # before the first solve

        rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                    arm_markers(TEST_Q, base), all_tracked())
        delta = compare_to_streamed(rx)
        assert sorted(delta) == list(REQUIRED_PLATES)
        # The angle is read through acos near 1, whose sensitivity is sqrt(eps)
        # — 2.4e-6 deg here — while the origin difference is a plain norm.
        assert max(d["angle_deg"] for d in delta.values()) < 1e-4
        assert max(d["origin_mm"] for d in delta.values()) < 1e-9


# --------------------------------------------------------------------------
# 4. Gates, stats and the opt-in fallback
# --------------------------------------------------------------------------


class TestGatesAndStats:
    def test_a_plate_with_missing_markers_yields_no_q(self):
        """One plate absent from the frame and ``q`` is ``None`` — not a ``q``
        with two joints quietly encoding the direction to the volume origin,
        which is what the streamed path does with an untracked plate."""
        base = volume_base(np.random.default_rng(51))
        rx = MarkerMocap(arm_locks(base))
        markers = arm_markers(TEST_Q, base)
        del markers[3]

        assert rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                           markers, all_tracked()) is None
        stats = rx.solve_stats()
        assert (stats.frames, stats.solved, stats.unsolved) == (1, 0, 1)
        assert stats.reasons[3] == {"no_marker_set": 1}
        assert 3 not in stats.last_rms_m           # no registration attempted
        assert stats.last_rms_m[0] < 1e-9          # the plates that did solve

    def test_a_plate_with_too_few_tracked_markers_yields_no_q(self):
        """Two markers never determine a frame (design §2), so the gate fires
        on the plate rather than the solve inventing a rotation about the
        remaining chord."""
        base = volume_base(np.random.default_rng(53))
        rx = MarkerMocap(arm_locks(base))
        flags = all_tracked()
        flags[4][0:3] = 0

        assert rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                           arm_markers(TEST_Q, base), flags) is None
        assert rx.solve_stats().reasons[4] == {"too_few_usable": 1}

    def test_a_frame_with_no_marker_data_at_all_yields_no_q(self):
        """``markers=None`` is the ring's "not streamed" entry (``mocap_rx``
        ops-9), distinct from an empty dict, and neither one may borrow the
        streamed poses that arrived alongside it."""
        base = volume_base(np.random.default_rng(57))
        rx = MarkerMocap(arm_locks(base))
        assert rx._solve_q(streamed_homos(arm_plates(TEST_Q, base)),
                           None, None) is None
        stats = rx.solve_stats()
        assert stats.unsolved == 1
        assert all(stats.reasons[p] == {"no_marker_set": 1} for p in range(6))

    def test_the_fallback_answers_from_the_streamed_poses_and_says_so(self):
        """``fallback_to_streamed=True`` exists for offline comparison work, so
        the ``q`` it returns must be the streamed one *and* be counted apart
        from the marker solves — a run that fell back for half its frames
        changed pose definition halfway, and ``fallback`` is the only place
        that shows."""
        base = volume_base(np.random.default_rng(59))
        homos = streamed_homos(arm_plates(TEST_Q, base))
        markers = arm_markers(TEST_Q, base)
        del markers[1]

        rx = MarkerMocap(arm_locks(base), fallback_to_streamed=True)
        q = rx._solve_q(homos, markers, all_tracked())
        assert q is not None and np.array_equal(q, mocap_to_q(homos))
        stats = rx.solve_stats()
        assert (stats.frames, stats.solved, stats.fallback, stats.unsolved) == (1, 0, 1, 0)
        assert stats.reasons[1] == {"no_marker_set": 1}   # why it fell back

    def test_stats_accumulate_across_frames_and_reset(self):
        """``solved_fraction`` is what a session is judged on afterwards: 60 %
        means the loop ran at 72 Hz of fresh pose on a 120 Hz stream, and this
        is the only place that is visible."""
        base = volume_base(np.random.default_rng(61))
        rx = MarkerMocap(arm_locks(base))
        homos = streamed_homos(arm_plates(TEST_Q, base))
        good, flags = arm_markers(TEST_Q, base), all_tracked()
        bad = {p: m for p, m in good.items() if p != 5}
        for markers in (good, good, good, bad):
            rx._solve_q(homos, markers, flags)

        stats = rx.solve_stats()
        assert (stats.frames, stats.solved, stats.unsolved) == (4, 3, 1)
        assert stats.solved_fraction == 0.75
        assert "3/4 frames solved from markers (75.0 %)" in stats.summary()
        assert "plate 5: no_marker_set" in stats.summary()

        rx.reset_stats()
        assert rx.solve_stats().frames == 0
        assert rx.solve_stats().solved_fraction == 0.0     # no divide by zero

    def test_solve_stats_is_a_snapshot_not_the_live_object(self):
        """It is read from another thread while the SDK thread keeps writing."""
        base = volume_base(np.random.default_rng(67))
        rx = MarkerMocap(arm_locks(base))
        homos = streamed_homos(arm_plates(TEST_Q, base))
        markers = arm_markers(TEST_Q, base)
        del markers[0]
        rx._solve_q(homos, markers, all_tracked())

        snap = rx.solve_stats()
        rx._solve_q(homos, markers, all_tracked())
        assert snap.frames == 1 and rx.solve_stats().frames == 2
        assert snap.reasons[0] == {"no_marker_set": 1}


# --------------------------------------------------------------------------
# 5. Locks on disk
# --------------------------------------------------------------------------


class TestLocksOnDisk:
    def test_save_and_load_round_trip(self, tmp_path):
        """Every field, exactly — a lock that lost its template on the way to
        disk still *looks* like a lock and fails only later, one frame at a
        time, as an RMS gate nobody is watching."""
        locks = arm_locks(volume_base(np.random.default_rng(71)))
        path = str(tmp_path / "locks.json")
        save_locks(path, locks, meta={"session": "test", "motive": "3.1"})

        loaded = load_locks(path)
        assert loaded == locks                      # frozen dataclasses, field by field
        assert MarkerMocap(loaded).locks.keys() == locks.keys()

    def test_saved_locks_are_in_the_probes_layout(self, tmp_path):
        """The documented promise is that ``save_locks`` output and
        ``mocap_probe.py --lock-out`` output are interchangeable, so the shape
        itself is part of the contract: ``meta`` alongside ``plates``, plate
        keys as strings."""
        locks = arm_locks(volume_base(np.random.default_rng(73)))
        path = str(tmp_path / "locks.json")
        save_locks(path, locks, meta={"session": "test"})

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        assert set(raw) == {"meta", "plates"}
        assert sorted(raw["plates"]) == [str(p) for p in REQUIRED_PLATES]
        assert raw["meta"] == {"session": "test"}

    def test_load_accepts_the_probe_layout_and_a_bare_mapping(self, tmp_path):
        """Three layouts reach this loader in practice — the probe's
        ``plates``, this module's own, and the hand-made bare mapping — and a
        session that cannot read the lock file it was given does not start."""
        locks = arm_locks(volume_base(np.random.default_rng(79)))
        body = {str(p): lk.to_dict() for p, lk in locks.items()}
        layouts = {
            "probe.json": {"meta": {"probe": "20260811_live2"}, "plates": body},
            "own.json": {"locks": body},
            "bare.json": body,
        }
        for name, payload in layouts.items():
            path = tmp_path / name
            path.write_text(json.dumps(payload), encoding="utf-8")
            assert load_locks(str(path)) == locks, name

    def test_a_file_without_locks_is_refused(self, tmp_path):
        """Silently returning ``{}`` would surface as the constructor's
        missing-plates error, pointing the operator at the wrong thing."""
        path = tmp_path / "notlocks.json"
        path.write_text(json.dumps({"meta": {"note": "reports only"}}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="no plate locks"):
            load_locks(str(path))

    def test_entries_that_are_not_locks_are_skipped(self, tmp_path):
        """A probe file carries non-plate keys next to the plates; picking one
        up would build a ``PlateLock`` out of metadata."""
        locks = arm_locks(volume_base(np.random.default_rng(83)))
        payload = {str(p): lk.to_dict() for p, lk in locks.items()}
        payload["captured_at"] = "2026-08-15T09:00:00"
        payload["7"] = {"note": "plate 7 was not in the volume"}
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = load_locks(str(path))
        assert sorted(loaded) == list(REQUIRED_PLATES)
        assert all(isinstance(lk, PlateLock) for lk in loaded.values())


# --------------------------------------------------------------------------
# 6. Minting locks from a rest capture
# --------------------------------------------------------------------------


class TestMintLocks:
    def test_a_still_window_locks_every_plate(self):
        """The evidence in the report is the point: a session stores it, and
        the x residual is the number that told the 2026-08-11 campaign the base
        plate's marker arms sit 16 deg off their designed azimuth.  Here the
        streamed orientation *is* the truth, so the residual is 0."""
        base = volume_base(np.random.default_rng(101))
        rx = CannedRx(rest_window(base, n=40))
        locks, report = mint_locks(rx, seconds=0.05)

        assert report["ok"] is True and report["refusals"] == []
        assert sorted(locks) == list(REQUIRED_PLATES)
        assert report["frames"] == 40 and report["x_mode"] == "streamed"
        for plate in REQUIRED_PLATES:
            entry = report["plates"][plate]
            assert entry["frames"] == 40
            assert entry["marker_sd_m"] < 1e-12     # identical frames, mean rounding
            assert entry["x_residual_deg"] < 1e-9
            assert entry["phi_deg"] == pytest.approx(math.degrees(FAMILY_PHIS[plate]))
            # The radial-arm asymmetry: recorded, never gated.  These brackets
            # separate their diagonal midpoints by 8.5 mm, inside the 5.5-12.8
            # mm the real plates measure — refusing on it was the rev-2 bug.
            assert 5.0 < entry["midpoint_separation_mm"] < 13.0

    def test_minted_locks_drive_a_correct_solve(self):
        """End to end, the way a session runs: mint at rest, construct the
        receiver from what was minted, then solve a pose that is nowhere near
        the rest one."""
        base = volume_base(np.random.default_rng(103))
        locks, report = mint_locks(CannedRx(rest_window(base, n=25)), seconds=0.05)
        assert report["ok"]

        rx = MarkerMocap(locks)
        q = rx._solve_q(streamed_homos(), arm_markers(TEST_Q, base), all_tracked())
        assert q is not None and np.max(np.abs(q - TEST_Q)) < TOL

    def test_it_refuses_when_the_arm_was_not_still(self):
        """A lock built from a moving arm is a template of a shape the plate
        never has again; every later frame then trips the RMS gate and the
        session is dead in a way that looks like bad tracking.  The gate is 1
        mm of per-coordinate sd against the 0.02-0.05 mm the real plates sit
        at, and a 10 mm creep across the window reads 2.2 mm.
        """
        base = volume_base(np.random.default_rng(107))
        rx = CannedRx(rest_window(base, n=40, drift_m=0.005))
        locks, report = mint_locks(rx, seconds=0.05)

        assert report["ok"] is False
        assert locks == {}
        assert len(report["refusals"]) == len(REQUIRED_PLATES)
        assert all("was not still" in r for r in report["refusals"])
        assert report["plates"][0]["marker_sd_m"] > 0.001

    def test_it_refuses_a_plate_with_too_little_rest_evidence(self):
        """Ten usable frames is ``mocap_probe``'s own floor, kept identical so
        the two cannot disagree about what counts as evidence."""
        base = volume_base(np.random.default_rng(109))
        rx = CannedRx(rest_window(base, n=8))
        locks, report = mint_locks(rx, seconds=0.05)

        assert report["ok"] is False and locks == {}
        assert all("usable rest frames" in r for r in report["refusals"])

    def test_it_refuses_when_no_frames_arrived(self):
        """Motive not streaming, which is the most common failure of all.  Note
        the early return leaves ``ok`` *absent* rather than False — the shipped
        caller reads it with ``report.get("ok")``, and this pins that a bare
        ``report["ok"]`` would raise on the one path that always runs when
        something is already wrong.
        """
        rx = CannedRx(rest_window(volume_base(np.random.default_rng(113)), n=0))
        locks, report = mint_locks(rx, seconds=0.05)

        assert locks == {}
        assert not report.get("ok")
        assert report["refusals"] == ["no marker frames arrived during the window"]

    def test_it_only_looks_at_frames_from_after_it_started(self):
        """The window is bounded below by the call time, so a rest capture
        cannot be satisfied by frames recorded while the arm was somewhere
        else — the ring holds ~20 s of exactly that."""
        rx = CannedRx(rest_window(volume_base(np.random.default_rng(127)), n=20))
        before = time.monotonic()
        mint_locks(rx, seconds=0.05)
        assert rx.t0_asked is not None and rx.t0_asked >= before


# --------------------------------------------------------------------------
# 7. The receiver's own change: q_stale next to stale
# --------------------------------------------------------------------------


class TestQStale:
    def test_frames_that_never_convert_are_q_stale_but_not_stale(self, monkeypatch):
        """The distinction the field was added for.

        Frames keep arriving — ``stale`` reads False and ``fps`` looks healthy —
        while nothing converts, so ``get_q()`` has nothing at all (and, once it
        has something, hands back the same pose forever).  A loop that checked
        only ``stale`` would call that "the arm is holding still".
        """
        rx = MocapRx()
        monkeypatch.setattr(rx, "_solve_q", lambda homos, markers, flags: None)
        for i in range(10):
            rx._on_new_frame({"frame_number": i})

        state = rx.get_state()
        assert state.frames == 10 and state.valid_frames == 0
        assert state.stale is False              # the stream is alive...
        assert state.q_stale is True             # ...and telling us nothing
        assert state.last_frame_mono is not None and state.last_q_mono is None
        assert rx.get_q() is None and state.ring_len == 0

    def test_a_stream_that_stops_converting_goes_q_stale_while_get_q_answers(
            self, monkeypatch):
        """The mid-run version, which is the dangerous one: ``get_q()`` keeps
        returning a pose, silently older every tick, and only ``q_stale`` says
        so.  ``last_q_mono`` stops advancing at the last frame that converted
        while ``last_frame_mono`` keeps moving.
        """
        monkeypatch.setattr(rx_module.mc, "STALE_AFTER_S", 0.05)
        rx = MocapRx()
        q_good = np.arange(mc.NUM_JOINTS, dtype=float)
        monkeypatch.setattr(rx, "_solve_q", lambda *_a: q_good.copy())
        rx._on_new_frame({"frame_number": 1})
        fresh = rx.get_state()
        assert not fresh.q_stale and fresh.last_q_mono is not None

        time.sleep(0.20)
        monkeypatch.setattr(rx, "_solve_q", lambda *_a: None)
        for i in range(2, 8):
            rx._on_new_frame({"frame_number": i})

        state = rx.get_state()
        assert state.frames == 7 and state.valid_frames == 1
        assert state.stale is False and state.q_stale is True
        assert state.last_q_mono == fresh.last_q_mono
        assert state.last_frame_mono > fresh.last_frame_mono
        assert np.array_equal(rx.get_q(), q_good)      # ...0.15 s old, and rising
