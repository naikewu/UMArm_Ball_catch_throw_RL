"""Tests for the CAN arm's marker-only plate frames.

The fixture is built the other way round from the code under test: a known
``q`` goes through :func:`UMArm_KINEMATICS.fkine.plate_transforms` to make
**body** frames, each body frame is turned into a **bracket** frame by the
per-plate azimuth, and four markers are hung off radial arms in that bracket's
plane.  Everything downstream then has a ground truth rather than only an
oracle: the round trip must return the ``q`` it started from, and fkine must
put the u-joint centres exactly where the fixture put them.

The marker azimuths are the hardware's, not a convenience.  On this arm the
arms lie along the bracket's own axes — that is the 2026-08-21 finding — so
marker ``k`` of plate ``p`` sits at body azimuth ``PLATE_AZIMUTH_DEG[p] + 45 +
90k``, which is exactly the geometry that makes ``compute_plate_lock``'s
"nearest diagonal rotated 45 deg" land on ``PLATE_AZIMUTH_DEG[p]``.  Writing
the fixture this way means a change to :data:`PLATE_AZIMUTH_DEG` that is not
also a change to the hardware fails here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from UMArm_KINEMATICS import canarm_params as cp
from UMArm_KINEMATICS.fkine import plate_transforms, ujoint_centres

from . import canarm_frames as cf

# Unequal radial arms, measured on the real plates (67-80 mm, differing by up
# to 9 mm across a diagonal).  Equal arms would make the diagonal-line origin
# and the midpoint mean agree, and the whole reason the origin is a line
# intersection is that on this hardware they do not.
ARM_RADII_M = (0.0795, 0.0721, 0.0748, 0.0705)

# Out-of-plane wander, also measured (0.03-0.24 mm RMS).  Not applied by
# default: it is the one departure from the ideal that the frame construction
# genuinely cannot absorb, since the lock's normal is the fitted plane's and no
# azimuth is a tilt.  It gets its own test rather than being smeared through
# every other one.
OUT_OF_PLANE_M = (0.00012, -0.00009, 0.00006, -0.00015)


def _rz(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def volume_base(rng=None):
    """A plausible mocap-volume pose for the arm's base plate.

    The azimuth is kept small on purpose: the lock breaks its 90 deg branch
    ambiguity against world +x, and this arm's brackets really do sit within a
    few degrees of it (measured 1.3-3.1 deg on 2026-08-21).  A base rotated
    45 deg in the volume would be a different calibration, not a harder case.
    """
    rng = np.random.default_rng(4) if rng is None else rng
    t = np.eye(4)
    t[0:3, 0:3] = _rz(math.radians(2.0))
    t[0:3, 3] = rng.uniform(-1.0, 1.0, 3)
    return t


def bracket_frames(q, base=None, azimuth_deg=None, order=None):
    """``(6, 4, 4)`` bracket frames in the volume, from a known ``q``."""
    base = volume_base() if base is None else base
    az = cf.PLATE_AZIMUTH_DEG if azimuth_deg is None else azimuth_deg
    order = cf.PROXIMAL_ORDER if order is None else order
    body = plate_transforms(q, cp.CANARM_PARAMS, order)
    out = np.tile(np.eye(4), (6, 1, 1))
    for p in range(6):
        t = base @ body[p]
        out[p, 0:3, 0:3] = t[0:3, 0:3] @ _rz(math.radians(az[p]))
        out[p, 0:3, 3] = t[0:3, 3]
    return out


def markers_from_frames(frames, out_of_plane=None):
    """Four markers per plate, on radial arms 90 deg apart in the bracket plane.

    Marker ``k`` sits at bracket azimuth ``45 + 90k``, i.e. the arms are the
    diagonals the lock looks for, and the bracket x bisects a pair of them.
    ``out_of_plane`` displaces each marker along the bracket normal; ``None``
    keeps them exactly coplanar, which is the case in which the construction is
    exact.
    """
    z = (0.0, 0.0, 0.0, 0.0) if out_of_plane is None else out_of_plane
    out = {}
    for p in range(6):
        r = frames[p][0:3, 0:3]
        o = frames[p][0:3, 3]
        pts = np.empty((4, 3))
        for k in range(4):
            a = math.radians(45.0 + 90.0 * k)
            local = np.array([ARM_RADII_M[k] * math.cos(a),
                              ARM_RADII_M[k] * math.sin(a), z[k]])
            pts[k] = o + r @ local
        out[p] = pts
    return out


REST_Q = np.array([0.04, -0.03, 0.02, 0.05, -0.02, 0.06,
                   0.03, -0.04, 0.05, 0.02, -0.06, 0.01])
BENT_Q = np.array([0.31, 0.24, -0.19, 0.27, -0.28, 0.22,
                   0.17, -0.25, 0.20, -0.30, 0.14, 0.26])


def locks_for(q=None, base=None):
    frames = bracket_frames(REST_Q if q is None else q, base)
    return cf.mint_locks(markers_from_frames(frames))


# --------------------------------------------------------------------------


class TestLockMinting:
    def test_the_lock_recovers_the_bracket_frame_it_was_built_from(self):
        """The lock is only as good as its azimuth branch: land on the wrong
        one of the four and every later ``q`` is 90 deg out while looking
        perfectly healthy."""
        base = volume_base()
        frames = bracket_frames(REST_Q, base)
        locks = locks_for(base=base)
        markers = markers_from_frames(frames)
        poses, valid, _ = cf.infer_frames(markers, None, locks)
        assert valid.all()
        assert np.allclose(poses, frames, atol=1e-9)

    def test_it_never_consults_the_streamed_orientation(self):
        """The point of this arm's pipeline: a lock must be reproducible from
        the marker file alone, because Motive's alignment is not the
        mechanism's."""
        locks = locks_for()
        assert all(lk.x_ref_source == "world_x_fallback" for lk in locks.values())
        assert all(lk.x_mode == "diagonal45" for lk in locks.values())

    def test_the_origin_is_the_diagonal_intersection_not_the_midpoint_mean(self):
        """With unequal arms the two differ by half the arm asymmetry, which on
        the real plates is 1.3-2.7 mm — far above the 0.02-0.07 mm marker
        noise, so it would show up as a bias and never as scatter."""
        frames = bracket_frames(REST_Q)
        markers = markers_from_frames(frames)
        locks = locks_for()
        poses, _, _ = cf.infer_frames(markers, None, locks)
        for p in range(6):
            centroid = markers[p].mean(axis=0)
            assert np.linalg.norm(poses[p][0:3, 3] - frames[p][0:3, 3]) < 1e-9
            assert np.linalg.norm(centroid - frames[p][0:3, 3]) > 1e-4

    def test_arm_up_comes_from_the_arm_not_from_world_z(self):
        frames = bracket_frames(REST_Q)
        u = cf.arm_up(markers_from_frames(frames))
        assert abs(np.linalg.norm(u) - 1.0) < 1e-12
        assert u[2] > 0.9          # this arm hangs base-up

    def test_a_collapsed_arm_axis_is_refused(self):
        markers = markers_from_frames(bracket_frames(REST_Q))
        markers[5] = markers[0].copy()
        with pytest.raises(ValueError):
            cf.arm_up(markers)


class TestRoundTrip:
    @pytest.mark.parametrize("q", [REST_Q, BENT_Q, np.zeros(12)])
    def test_q_survives_the_whole_pipeline(self, q):
        """Known ``q`` -> body frames -> brackets -> markers -> locks ->
        registration -> ``q``.  Exact, because every step is a rigid transform
        and the template is the fixture's own geometry."""
        base = volume_base()
        locks = locks_for(base=base)
        frames = bracket_frames(q, base)
        poses, valid, _ = cf.infer_frames(markers_from_frames(frames), None, locks)
        assert valid.all()
        assert np.allclose(cf.q_from_plate_frames(poses), q, atol=1e-9)

    def test_the_wrong_proximal_order_is_silently_wrong(self):
        """The failure mode the ``order`` parameter exists to prevent: reading
        a ``"yx"`` arm with the ``"xy"`` convention returns a finite, smooth,
        plausible ``q`` that is simply not the arm's."""
        frames = bracket_frames(BENT_Q)
        poses, _, _ = cf.infer_frames(markers_from_frames(frames), None, locks_for())
        wrong = cf.q_from_plate_frames(poses, order="xy")
        assert wrong is not None and np.isfinite(wrong).all()
        assert not np.allclose(wrong, BENT_Q, atol=1e-3)

    def test_the_wrong_azimuth_is_silently_wrong_too(self):
        frames = bracket_frames(BENT_Q)
        poses, _, _ = cf.infer_frames(markers_from_frames(frames), None, locks_for())
        wrong = cf.q_from_plate_frames(poses, azimuth_deg=cf.FAMILY_AZIMUTH_DEG)
        assert wrong is not None and np.isfinite(wrong).all()
        assert not np.allclose(wrong, BENT_Q, atol=1e-3)


class TestDropoutRobustness:
    @pytest.mark.parametrize("dropped", [0, 1, 2, 3])
    def test_three_markers_solve_exactly(self, dropped):
        """Four markers exist so that one can be lost.  Registration of a rigid
        template onto any three of them is exact, which is why there is no
        parallelogram or midpoint assumption anywhere in the solve."""
        base = volume_base()
        locks = locks_for(base=base)
        frames = bracket_frames(BENT_Q, base)
        markers = markers_from_frames(frames)
        flags = {p: np.ones(4, dtype=np.uint8) for p in range(6)}
        for p in range(6):
            flags[p][dropped] = 0
        poses, valid, quality = cf.infer_frames(markers, flags, locks)
        assert valid.all()
        assert all(q.n_usable == 3 for q in quality.values())
        assert np.allclose(cf.q_from_plate_frames(poses), BENT_Q, atol=1e-9)

    def test_two_markers_refuse_rather_than_guess(self):
        """Two points leave the rotation about their line unobservable; a frame
        returned here would be a coin flip wearing a pose."""
        locks = locks_for()
        markers = markers_from_frames(bracket_frames(BENT_Q))
        flags = {p: np.array([1, 1, 0, 0], dtype=np.uint8) for p in range(6)}
        poses, valid, quality = cf.infer_frames(markers, flags, locks)
        assert not valid.any()
        assert all(q.reason == "too_few_usable" for q in quality.values())

    def test_a_swapped_label_fails_the_template_gate(self):
        """A mid-run label swap moves marker offsets by centimetres against
        sub-millimetre noise, so it must fail loudly rather than yield a
        confident wrong frame."""
        locks = locks_for()
        markers = markers_from_frames(bracket_frames(BENT_Q))
        markers[2] = markers[2][[1, 0, 2, 3]]
        poses, valid, quality = cf.infer_frames(markers, None, locks)
        assert not valid[2]
        assert quality[2].reason == "template_rms_gate"


class TestGeometryChecks:
    def test_co_rigid_plates_agree_exactly_in_the_fixture(self):
        """Plates 1/2 and 3/4 are bolted to one connector.  This is the check
        that needs no fit and no oracle, and the one the live session passed to
        0.03 deg of standard deviation over 68 poses."""
        frames = bracket_frames(BENT_Q)
        poses, _, _ = cf.infer_frames(markers_from_frames(frames), None, locks_for())
        for az_err, tilt in cf.co_rigid_residual_deg(poses):
            assert abs(az_err) < 1e-6
            assert abs(tilt) < 1e-6

    def test_the_chain_gaps_are_the_measured_table(self):
        """Rigid distances: they must not depend on the pose, and they must be
        the numbers ``canarm_params`` carries."""
        for q in (REST_Q, BENT_Q, np.zeros(12)):
            frames = bracket_frames(q)
            poses, _, _ = cf.infer_frames(markers_from_frames(frames), None,
                                          locks_for())
            assert np.allclose(cf.chain_gaps_m(poses), cp.CANARM_PLATE_CHAIN_M,
                               atol=1e-9)

    def test_fkine_reproduces_the_fixture_exactly(self):
        """With the right azimuths, the right order and the right lengths the
        residual is numerical noise; every millimetre seen on the real arm is
        therefore hardware or measurement, not this code."""
        for q in (REST_Q, BENT_Q):
            frames = bracket_frames(q)
            poses, _, _ = cf.infer_frames(markers_from_frames(frames), None,
                                          locks_for())
            res = cf.fk_residual_m(poses)
            assert res is not None
            assert res.max() < 1e-9, res

    def test_out_of_plane_marker_wander_costs_sub_millimetre(self):
        """The honest limit of the construction.  A marker sitting 0.1 mm off
        its plate's plane tilts the fitted plane by about 0.1 deg, which no
        azimuth can absorb because an azimuth is a rotation about the normal.
        The cost accumulates down the chain and is a real part of the 2 mm the
        live arm shows; bounded here so a regression that makes it worse is
        visible."""
        base = volume_base()
        frames = bracket_frames(BENT_Q, base)
        markers = markers_from_frames(frames, OUT_OF_PLANE_M)
        locks = cf.mint_locks(markers_from_frames(bracket_frames(REST_Q, base),
                                                 OUT_OF_PLANE_M))
        poses, valid, _ = cf.infer_frames(markers, None, locks)
        assert valid.all()
        res = cf.fk_residual_m(poses)
        assert res.max() < 1e-3, res * 1000.0

    def test_fk_centres_world_matches_ujoint_centres_through_the_base(self):
        base = volume_base()
        frames = bracket_frames(BENT_Q, base)
        poses, _, _ = cf.infer_frames(markers_from_frames(frames), None,
                                      locks_for(base=base))
        body = cf.body_frames(poses)
        pred = cf.fk_centres_world(BENT_Q, body[0], cp.CANARM_PARAMS)
        local = ujoint_centres(BENT_Q, cp.CANARM_PARAMS, cf.PROXIMAL_ORDER)
        assert np.allclose(pred, local @ body[0][0:3, 0:3].T + body[0][0:3, 3])


class TestCalibrationProvenance:
    def test_the_azimuths_are_measured_and_the_two_measurements_agree(self):
        """The refined and mechanism-only azimuths are separate measurements of
        one quantity; shipping only the refined one would make the agreement
        unauditable."""
        assert cf.AZIMUTH_MEASURED is True
        assert "drive_2026-08-21" in cf.AZIMUTH_SOURCE
        a = np.array(cf.PLATE_AZIMUTH_DEG)
        b = np.array(cf.PLATE_AZIMUTH_FROM_DRIVE_AXES_DEG)
        assert np.abs(a - b).max() < 1.0

    def test_the_brackets_alternate_by_45_degrees(self):
        """The bracket family the mechanism was designed with, read back out of
        the measurement rather than assumed into it."""
        a = np.array(cf.PLATE_AZIMUTH_DEG)
        for p in range(0, 6, 2):
            gap = (a[p + 1] - a[p]) % 90.0
            assert abs(gap - 45.0) < 2.0, (p, gap)

    def test_the_proximal_order_is_the_measured_one(self):
        from UMArm_KINEMATICS.fkine import PROXIMAL_ORDERS

        assert cf.PROXIMAL_ORDER in PROXIMAL_ORDERS
        assert cf.PROXIMAL_ORDER == "yx"

    def test_locks_round_trip_through_json(self, tmp_path):
        locks = locks_for()
        path = cf.save_locks(locks, str(tmp_path / "locks.json"))
        back = cf.load_locks(path)
        assert set(back) == set(locks)
        for p in range(6):
            assert np.allclose(np.array(back[p].template_m),
                               np.array(locks[p].template_m))

    def test_missing_locks_say_how_to_make_them(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            cf.load_locks(str(tmp_path / "absent.json"))
        assert "mint_locks" in str(exc.value)
