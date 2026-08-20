"""Tests for marker-frame inference.  Pure math + one real-data acceptance test.

Layered like ``test_mocap_to_q.py``, because each layer catches what the
others cannot:

1. **Forward constructor + round trips** (:class:`TestLock`,
   :class:`TestInferPlateFrame`, :class:`TestFullPipeline`).  Marker
   quadrilaterals are *forward-constructed* from a known plate SE(3) and
   family angle (design §7's ``fake_mocap`` recipe: ``markers = T @ Rz(phi) @
   corners``), and lock+solve must recover the frame — origin and axes at
   machine precision — invariant to label permutation and rigid world motion.
   The **default geometry is the real hardware's**: four markers on radial
   arms of *unequal* length (60/75/68/90 mm here; the measured plates have
   diagonal lengths differing by 5.5-15 mm and diagonal midpoints 5.5-12.8 mm
   apart — probe_20260811_live2 ``rectangle_stats``).  A square case is kept
   where hand-checkability matters (goldens, sign conventions).

2. **Frozen goldens** (:class:`TestGoldensFrozen`).  Literal numbers,
   produced once by this implementation on this bench and frozen, pinning
   the +45 deg candidate rotation sense, the CCW family correction Rz(-phi),
   the streamed-reference disambiguation, the u_up normal sign, and the
   t3/t4 extraction basis change (design §9).  The golden plate is a
   *square*, for which the rev-3 origin redefinition (diagonal-line
   intersection instead of midpoint mean) changes nothing — the two
   definitions coincide — so every golden literal survives the template
   revision untouched, which is itself the strongest possible pin that the
   sign conventions did not move.

3. **Gates and refusals** (:class:`TestGates`, :class:`TestLockRefusals`).
   The per-frame registration gate (label swap -> ``None``, count mismatch
   -> ``None``, < 3 usable -> ``None``, NaN -> ``None``, collinear subset ->
   ``None``) and the lock refusals (diagonals far off perpendicular,
   near-horizontal normal, degenerate markers, stale-lock template mismatch
   on consume).  The 45-deg-boundary x residual is *recorded, not refused* —
   the probe owns that gate (exit 3, design §4.1.3).

4. **The kinematic seam** (:class:`TestQFromFrames`, :class:`TestCoRigid`,
   :class:`TestFullPipeline`).  ``q_from_frames`` against
   ``UMArm_KINEMATICS.plate_transforms``: machine precision for all 12
   joints, single AND multi-joint, plus the co-rigid {1,2}/{3,4} consistency
   the benchmark's model verification stands on.

5. **Acceptance on real data** (:class:`TestAcceptanceRealProbe`, skipped
   when the capture is absent).  The 2026-08-11 live probe capture — the
   very data whose 5.5-12.8 mm midpoint separations refused every lock under
   the old midpoint/parallelogram formulation — must now lock all six
   plates, solve >= 99 % of frames, and hold sub-mm origin jitter and
   drop-one agreement.

Run:  ``python -m pytest UMArm_MOCAP/test_marker_frame.py -q``
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

import numpy as np
import pytest

# The module under test, imported flat so this file works whichever way pytest
# rooted itself; the repo root goes on the path too for UMArm_KINEMATICS (a
# normal import — the package landed before this module, design §9).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import marker_frame as mf  # noqa: E402
import mocap_constants as mc  # noqa: E402
from mocap_to_q import quat_xyzw_to_matrix  # noqa: E402
from UMArm_KINEMATICS import robot_params as rp  # noqa: E402
from UMArm_KINEMATICS.fkine import plate_transforms  # noqa: E402

#: Round-trip budget for noiseless synthetic data.  The Kabsch solve carries
#: the same cancellation cost as the old diagonal math (~1 m coordinates vs a
#: ~0.1 m bracket); 1e-12 keeps orders of headroom over the observed worst
#: case while still catching any real defect (a wrong candidate or mounting
#: model moves things by whole degrees).
TOL = 1e-12

#: Frozen-golden budget: the literals pin ~13 significant digits while
#: leaving room for BLAS/matmul reassociation across numpy builds (same
#: number as test_fkine.GOLDEN_TOL).
GOLDEN_TOL = 1e-13

#: Family angles, in plate order — the design §2 table, used by every
#: synthetic-marker helper below.
FAMILY_PHIS = np.array([0.0, math.pi / 4, 0.0, math.pi / 4, 0.0, math.pi / 4])

#: Default radial-arm radii (metres) for the synthetic plates: distinct on
#: purpose, like the hardware (probe-measured diagonal lengths 129.7-205.4 mm
#: with 5.5-15 mm asymmetry per plate; these give diagonals 128 and 165 mm
#: with an 8.5 mm midpoint separation — inside the measured 5.5-12.8 mm band).
ARM_RADII_M = (0.060, 0.075, 0.068, 0.090)


# --------------------------------------------------------------------------
# Forward constructors: known frame -> marker quadrilateral
# --------------------------------------------------------------------------


def rot_z(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rodrigues(axis, angle: float) -> np.ndarray:
    """Test-local Rodrigues, deliberately independent of marker_frame's."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrix, numpy only (QR of a Gaussian matrix)."""
    a, r = np.linalg.qr(rng.normal(size=(3, 3)))
    a = a * np.sign(np.diag(r))
    if np.linalg.det(a) < 0:
        a[:, 0] = -a[:, 0]
    return a


def random_se3(rng: np.random.Generator) -> np.ndarray:
    out = np.eye(4)
    out[0:3, 0:3] = random_rotation(rng)
    out[0:3, 3] = rng.uniform(-2.0, 2.0, 3)
    return out


def arm_corners(radii=ARM_RADII_M, angles_deg=(45.0, 135.0, 225.0, 315.0),
                ) -> np.ndarray:
    """``(4, 3)`` bracket-frame markers on **radial arms of unequal length**.

    The real hardware (design §2, measured probe_20260811_live2): four
    markers on arms radiating from the u-joint centre, the diagonals sitting
    45 deg from the bracket axes — hence arm angles 45/135/225/315 — with
    per-arm radii that differ by up to 15 mm.  Label i's opposite is i+2, so
    the diagonal *lines* (0,2) and (1,3) both pass exactly through the
    origin however unequal the radii: the diagonal-line intersection IS the
    centre, while the diagonal midpoints sit millimetres away from it.
    """
    return np.array([[r * math.cos(math.radians(a)),
                      r * math.sin(math.radians(a)), 0.0]
                     for r, a in zip(radii, angles_deg)])


def rect_corners(w: float = 0.03, h: float | None = None) -> np.ndarray:
    """``(4, 3)`` square/rectangular corners, CCW about +z, centred on the
    origin — the hand-checkable special case (equal radii w*sqrt(2) at
    45/135/225/315 when h == w).  Kept for the goldens and the sign
    conventions; the asymmetric :func:`arm_corners` is the default geometry.
    """
    h = w if h is None else h
    return np.array([[w, h, 0.0], [-w, h, 0.0], [-w, -h, 0.0], [w, -h, 0.0]])


def markers_from_pose(pose: np.ndarray, phi: float,
                      corners: np.ndarray | None = None) -> np.ndarray:
    """The design §7 recipe: ``markers = T_p @ Rz(phi_p) @ corners``."""
    corners = arm_corners() if corners is None else corners
    r = pose[0:3, 0:3] @ rot_z(phi)
    return corners @ r.T + pose[0:3, 3]


def rest_stack(markers: np.ndarray, n: int = 3) -> np.ndarray:
    """A noiseless rest window: the same frame ``n`` times."""
    return np.stack([markers] * n)


def arm_locks(base: np.ndarray, corners: np.ndarray | None = None) -> dict:
    """Locks for all six plates from the q=0 rest pose under ``base``.

    u_up is arm-derived exactly as the probe computes it: plate 0 minus
    plate 5 rest positions (design §2 — never a world axis).
    """
    plates = base @ plate_transforms(np.zeros(12))
    u_up = plates[0][0:3, 3] - plates[5][0:3, 3]
    return {p: mf.compute_plate_lock(
                rest_stack(markers_from_pose(plates[p], FAMILY_PHIS[p], corners)),
                plate=p, u_up=u_up, streamed_rot=plates[p][0:3, 0:3])
            for p in range(6)}


def arm_markers(q: np.ndarray, base: np.ndarray,
                corners: np.ndarray | None = None) -> dict:
    """Marker dict (plate -> (4, 3)) for one frame of the synthetic arm."""
    plates = base @ plate_transforms(q)
    return {p: markers_from_pose(plates[p], FAMILY_PHIS[p], corners)
            for p in range(6)}


def rot_angle(r: np.ndarray) -> float:
    """Rotation angle (radians) of a 3x3 rotation matrix, noise-clipped."""
    return math.acos(min(1.0, max(-1.0, (float(np.trace(r)) - 1.0) / 2.0)))


ALL_TRACKED = np.ones(4, dtype=np.uint8)


# --------------------------------------------------------------------------
# Frozen golden data: the conventions, checkable with no reference around
# --------------------------------------------------------------------------
# Every number below is a literal, produced once by this implementation on
# this bench (scratch script, 2026-08-11) and frozen.  The inputs are exact:
# R_PLATE is the product of two Pythagorean rotations (3-4-5 and 7-24-25), so
# its entries are short decimals and the matrix is orthonormal to 1e-16.
#
# The golden plate is a SQUARE on purpose: its diagonal-line intersection and
# its diagonal-midpoint mean coincide, so the rev-3 origin redefinition (the
# template revision driven by the measured radial-arm asymmetry) changes NO
# golden literal — the pins below prove the sign conventions (45-deg sense,
# family correction, normal sign, t3/t4 basis change) survived unchanged.

#: An exactly-orthonormal, structure-free plate rotation (= GOLDEN_RZ @
#: GOLDEN_RX of test_mocap_to_q, multiplied out).
GOLDEN_R_PLATE = np.array([
    [0.6, -0.224, 0.768],
    [0.8, 0.168, -0.576],
    [0.0, 0.96, 0.28],
])

GOLDEN_ORIGIN = np.array([0.25, -0.15, 0.4])

#: The golden plate's markers: a 0.06 m square bracket at GOLDEN_R_PLATE /
#: GOLDEN_ORIGIN with family angle +45 deg (plate 1), labels deliberately
#: enumerated CW about the plate normal so the lock's u_up reversal branch is
#: exercised (label i holds corner [0, 3, 2, 1][i]).
GOLDEN_MARKERS = np.array([
    [0.2404964848608528, -0.1428723636456396, 0.44072935059634516],
    [0.27545584412271573, -0.11605887450304571, 0.4],
    [0.2595035151391472, -0.1571276363543604, 0.3592706494036549],
    [0.2245441558772843, -0.1839411254969543, 0.4],
])

#: Arm-derived up for the golden lock: tilted ~6 deg off the plate normal,
#: nothing special about it except not being an axis.
GOLDEN_U_UP = GOLDEN_R_PLATE @ np.array([0.1, -0.05, 1.0])

#: Streamed orientation with 8 deg of simulated manual-alignment error about
#: the plate z — far inside the 45 deg ambiguity, so the disambiguation must
#: shrug it off and record it as the x residual.
GOLDEN_STREAMED = GOLDEN_R_PLATE @ rot_z(math.radians(8.0))

#: The frame infer_plate_frame must return for GOLDEN_MARKERS under the
#: golden lock.  Third column = GOLDEN_R_PLATE z (the family angle lives in
#: x/y only); first two columns carry the +45 deg bracket.
GOLDEN_T = np.array([
    [0.2658721497261418, -0.5826559876977152, 0.768, 0.25],
    [0.6844793641885782, -0.44689148570989784, -0.5759999999999998, -0.15],
    [0.6788225099390854, 0.6788225099390858, 0.2800000000000002, 0.4],
    [0.0, 0.0, 0.0, 1.0],
])

#: q_from_frames golden: one segment at (t1, t2, t3, t4) = GOLDEN_Q_SEG1,
#: segments 2 and 3 straight.  GOLDEN_R_E is the distal body rotation
#: Rx(t1) Ry(t2) Rot_{(x+y)/sqrt2}(t3) Rot_{(-x+y)/sqrt2}(t4), frozen so the
#: test does not depend on any helper sharing marker_frame's conventions;
#: GOLDEN_O1 is Rx(t1) Ry(t2) @ [0, 0, -0.22].
GOLDEN_Q_SEG1 = (0.3, -0.2, 0.25, -0.15)

GOLDEN_R_E = np.array([
    [0.9918622989491775, -0.02724721208839957, -0.12436546689761523],
    [-0.0457134739938363, 0.8354733913822354, -0.5476262325598371],
    [0.11882532650351485, 0.5488549915284324, 0.8274289939660425],
])

GOLDEN_O1 = np.array([0.04370725277491347, 0.06371848507761342,
                      -0.20598453998852384])


def golden_lock() -> mf.PlateLock:
    return mf.compute_plate_lock(rest_stack(GOLDEN_MARKERS), plate=1,
                                 u_up=GOLDEN_U_UP,
                                 streamed_rot=GOLDEN_STREAMED)


def golden_chain_frames() -> np.ndarray:
    """Six *marker* frames (body @ Rz(phi_p)) for the golden one-segment q.

    Built from the frozen GOLDEN_R_E/GOLDEN_O1 literals plus exact
    translation arithmetic: with segments 2 and 3 at zero, plates 1-5 all
    share the distal body rotation and hang down its -z.
    """
    frames = np.tile(np.eye(4), (6, 1, 1))
    rot_bodies = [np.eye(3)] + [GOLDEN_R_E] * 5
    origins = [np.zeros(3), GOLDEN_O1]
    for gap in (0.048, 0.19, 0.047, 0.18):
        origins.append(origins[-1] + GOLDEN_R_E @ np.array([0.0, 0.0, -gap]))
    for p in range(6):
        frames[p, 0:3, 0:3] = rot_bodies[p] @ rot_z(FAMILY_PHIS[p])
        frames[p, 0:3, 3] = origins[p]
    return frames


# --------------------------------------------------------------------------
# 1. Design constants, pinned
# --------------------------------------------------------------------------


class TestConstantsPinned:
    def test_tolerances_are_the_design_ones(self):
        """5 mm line-intersection residual (lock, §4.1.1), 25 deg crossing
        band (lock, §4.1.1), 3 mm template RMS (per frame, §4.2), cos 60 deg
        normal-sign bound (§4.1.2).  These are design policy, not tuning
        knobs — a drift is a design change."""
        assert mf.MIDPOINT_TOL_M == 0.005
        assert mf.CROSSING_TOL_DEG == 25.0
        assert mf.TEMPLATE_RMS_TOL_M == 0.003
        assert mf.NORMAL_UP_MIN_COS == 0.5
        assert mf.LOCK_STATS_TOL_M == 0.003

    def test_dead_constants_removed(self):
        """The per-frame diagonal-length gate died with the parallelogram
        formulation (the template RMS gate subsumes it and also catches
        along-diagonal swaps); its constant must not linger misleadingly."""
        assert not hasattr(mf, "DIAG_LEN_TOL_M")

    def test_family_table(self):
        """Plates 0/2/4 = 0 deg, 1/3/5 = +45 deg; plate 6 deliberately absent
        (family measured, not assumed — design D6)."""
        assert mf.FAMILY_PHI_RAD == {0: 0.0, 1: math.pi / 4, 2: 0.0,
                                     3: math.pi / 4, 4: 0.0, 5: math.pi / 4}
        assert 6 not in mf.FAMILY_PHI_RAD

    def test_rz45_is_plus_45_and_not_the_mocap_constant(self):
        """The q_from_frames basis change is **+45 deg** about z — the
        transpose of ``mocap_constants.RZn45`` (-45 deg, applied to vectors
        in the swing-only reader).  Confusing the two is the plausible wrong
        answer, so pin both the literal and the relationship."""
        s = 0.7071067811865476                     # sin(pi/4) to double precision
        assert np.allclose(mf.RZ45, np.array([[s, -s, 0.0],
                                              [s, s, 0.0],
                                              [0.0, 0.0, 1.0]]), atol=1e-15)
        assert mf.RZ45[0, 1] < 0.0                 # +45 deg, not -45 deg
        assert np.allclose(mf.RZ45, mc.RZn45.T, atol=0)


# --------------------------------------------------------------------------
# 2. The lock
# --------------------------------------------------------------------------


class TestLock:
    def test_identity_plate_locks_the_bracket_frame(self):
        """The hand-checkable anchor: square bracket, identity pose, phi=0.
        Cyclic order is the CCW corner order, and the two ordered-diagonal
        thetas are 135 and 45 deg — the 'theta ~ +-45 for a square plate'
        note of §4.1.3, with the a->c diagonal pointing into the third
        quadrant hence 135."""
        markers = rect_corners() + np.array([0.1, -0.2, 0.05])
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0],
                                     streamed_rot=np.eye(3))
        assert lock.cyclic_order == (0, 1, 2, 3)
        assert abs(lock.diag_thetas_rad[0] - 0.75 * math.pi) < 1e-12
        assert abs(lock.diag_thetas_rad[1] - 0.25 * math.pi) < 1e-12
        assert abs(lock.x_residual_deg) < 1e-9
        assert lock.phi_rad == 0.0
        assert lock.x_ref_source == "streamed"
        assert abs(lock.diag_lengths_m[0] - 0.06 * math.sqrt(2.0)) < 1e-12
        assert abs(lock.midpoint_separation_m) < 1e-15
        assert lock.intersection_residual_m < 1e-9
        assert lock.diagonals_are_longest_chords
        assert abs(lock.n_dot_u_up - 1.0) < 1e-12
        # On a square the template IS the corner table: offsets from the
        # centre expressed in the bracket frame.
        assert np.max(np.abs(np.asarray(lock.template_m)
                             - rect_corners())) < 1e-12

    def test_asymmetric_radial_arm_plate_locks(self):
        """The real-hardware geometry (design §2 rev 3): arms of unequal
        length.  The lock must record — not refuse — the millimetre-scale
        midpoint separation (the probe measured 5.5-12.8 mm on the six real
        plates; this construction sits at 8.5 mm), keep the same theta
        conventions as the square, and measure the per-arm radii."""
        markers = arm_corners() + np.array([0.1, -0.2, 0.05])
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0],
                                     streamed_rot=np.eye(3))
        assert lock.cyclic_order == (0, 1, 2, 3)
        assert abs(lock.diag_thetas_rad[0] - 0.75 * math.pi) < 1e-12
        assert abs(lock.diag_thetas_rad[1] - 0.25 * math.pi) < 1e-12
        assert abs(lock.x_residual_deg) < 1e-9
        assert abs(lock.diag_lengths_m[0] - 0.128) < 1e-12   # 60 + 68 mm
        assert abs(lock.diag_lengths_m[1] - 0.165) < 1e-12   # 75 + 90 mm
        # The asymmetry the old formulation refused on (> 5 mm): recorded.
        assert 0.0084 < lock.midpoint_separation_m < 0.0086
        assert lock.midpoint_separation_m > mf.MIDPOINT_TOL_M
        assert lock.intersection_residual_m < 1e-9
        assert abs(lock.diagonal_crossing_deg - 90.0) < 1e-9
        for got, want in zip(lock.arm_radii_m, ARM_RADII_M):
            assert abs(got - want) < 1e-12
        # Template = the arm corners themselves (identity pose, phi 0).
        assert np.max(np.abs(np.asarray(lock.template_m)
                             - arm_corners())) < 1e-12

    def test_x_mode_streamed_anchors_the_azimuth_to_the_streamed_x(self):
        """The 2026-08-11 arbitration fix: marker arms rotated a known 16 deg
        off the designed azimuth.  diagonal45 mode (the spec) snaps x to the
        rotated diagonals — 16 deg off the mechanism — while streamed mode
        takes the streamed bracket x itself; BOTH record the 16 deg offset in
        x_residual_deg, and the tracking template is identical either way."""
        delta = math.radians(16.0)
        markers = arm_corners() @ rot_z(delta).T + np.array([0.1, -0.2, 0.05])
        kw = dict(plate=0, u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        spec = mf.compute_plate_lock(rest_stack(markers), x_mode="diagonal45", **kw)
        anch = mf.compute_plate_lock(rest_stack(markers), x_mode="streamed", **kw)
        assert spec.x_mode == "diagonal45" and anch.x_mode == "streamed"
        assert abs(spec.x_residual_deg - 16.0) < 1e-9
        assert abs(anch.x_residual_deg - 16.0) < 1e-9
        # spec's x follows the (rotated) diagonals: thetas keep the square
        # convention; anchored x is the streamed x, so the thetas absorb the
        # 16 deg instead.
        assert abs(spec.diag_thetas_rad[0] - 0.75 * math.pi) < 1e-12
        # theta maps the diagonal ONTO x: the arms moved +16 deg while the
        # anchored x stayed put, so the mapping angle SHRINKS by 16 deg.
        assert abs(anch.diag_thetas_rad[0] - (0.75 * math.pi - delta)) < 1e-12
        # Same physical markers, same origin; frames differ by exactly Rz(16).
        T_spec, q_spec = mf.infer_plate_frame(markers, np.ones(4, np.uint8), spec)
        T_anch, q_anch = mf.infer_plate_frame(markers, np.ones(4, np.uint8), anch)
        assert not q_spec.reason and not q_anch.reason
        assert np.allclose(T_spec[0:3, 3], T_anch[0:3, 3], atol=1e-12)
        rel = T_anch[0:3, 0:3].T @ T_spec[0:3, 0:3]
        assert abs(math.degrees(math.atan2(rel[1, 0], rel[0, 0])) - 16.0) < 1e-9
        # streamed mode without a streamed pose is a caller error.
        with pytest.raises(ValueError):
            mf.compute_plate_lock(rest_stack(markers), plate=0,
                                  u_up=[0.0, 0.0, 1.0], x_mode="streamed")

    def test_origin_is_the_line_intersection_not_the_midpoint_mean(self):
        """THE rev-3 change, asserted both ways: on arms radiating from a
        known centre the lock origin equals that centre to machine
        precision, while the old midpoint-mean origin (== the marker
        centroid) sits provably millimetres away — 4.1 mm here, half the
        2.8-6.4 mm bias band the real plates would have carried."""
        centre = np.array([0.1, -0.2, 0.05])
        markers = arm_corners() + centre
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0],
                                     streamed_rot=np.eye(3))
        t, q = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        assert t is not None and q.reason == ""
        assert np.max(np.abs(t[0:3, 3] - centre)) < TOL
        # The midpoint mean of 4 markers is their centroid; assert the gap.
        midpoint_mean = markers.mean(axis=0)
        gap = float(np.linalg.norm(midpoint_mean - centre))
        assert 0.004 < gap < 0.0045
        assert float(np.linalg.norm(t[0:3, 3] - midpoint_mean)) > 0.004

    def test_label_shuffle_invariance(self):
        """Same physical bracket, arbitrary asset labelling: the *inferred
        frame* must not depend on which marker got which label (§9)."""
        rng = np.random.default_rng(5)
        pose = random_se3(rng)
        u_up = pose[0:3, 0:3] @ np.array([0.0, 0.1, 1.0])
        base_markers = markers_from_pose(pose, math.pi / 4)
        lock0 = mf.compute_plate_lock(rest_stack(base_markers), plate=1,
                                      u_up=u_up, streamed_rot=pose[0:3, 0:3])
        t0, _ = mf.infer_plate_frame(base_markers, ALL_TRACKED, lock0)
        for perm in ([1, 0, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 2, 3, 0]):
            shuffled = base_markers[perm]
            lock = mf.compute_plate_lock(rest_stack(shuffled), plate=1,
                                         u_up=u_up, streamed_rot=pose[0:3, 0:3])
            t, _ = mf.infer_plate_frame(shuffled, ALL_TRACKED, lock)
            assert np.max(np.abs(t - t0)) < TOL, f"perm {perm}"

    def test_normal_sign_follows_u_up_not_label_order(self):
        """Labels enumerated CW must yield the same up-oriented frame as CCW:
        the stored cyclic order is reversed so the shoelace normal points
        along +u_up (§4.1.2)."""
        ccw = arm_corners()
        cw = ccw[[0, 3, 2, 1]]
        args = dict(plate=0, u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        lock_ccw = mf.compute_plate_lock(rest_stack(ccw), **args)
        lock_cw = mf.compute_plate_lock(rest_stack(cw), **args)
        t_ccw, _ = mf.infer_plate_frame(ccw, ALL_TRACKED, lock_ccw)
        t_cw, _ = mf.infer_plate_frame(cw, ALL_TRACKED, lock_cw)
        assert np.max(np.abs(t_cw - t_ccw)) < TOL
        assert t_cw[2, 2] > 0.99                   # z column along +u_up

    def test_up_is_arm_derived_not_world(self):
        """Hang the rig upside down and tell the lock so via u_up: the frame
        z must follow the *arm's* up (negative world z here).  This is review
        finding ops-4 made executable — world axes are never consulted."""
        flip = rodrigues([1.0, 0.0, 0.0], math.pi)
        markers = arm_corners() @ flip.T + np.array([0.0, 0.0, 2.0])
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, -1.0], streamed_rot=flip)
        t, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        assert t[2, 2] < -0.99                     # frame z points world-down
        assert lock.n_dot_u_up > 0.99              # ... which is +u_up
        assert abs(lock.u_up_vs_world_z_deg - 180.0) < 1e-9

    def test_plate6_family_is_measured_not_assumed(self):
        """Plate 6 tries both families and stores the better fit (design D6):
        markers built at +45 lock as +45, markers built at 0 lock as 0."""
        pose = np.eye(4)
        pose[0:3, 3] = [0.1, 0.2, -0.7]
        for phi_true in (0.0, math.pi / 4):
            markers = markers_from_pose(pose, phi_true)
            lock = mf.compute_plate_lock(rest_stack(markers), plate=6,
                                         u_up=[0.0, 0.0, 1.0],
                                         streamed_rot=np.eye(3))
            assert lock.phi_rad == pytest.approx(phi_true, abs=1e-12)
            assert lock.x_residual_deg < 1e-9

    def test_world_x_fallback_is_flagged(self):
        """No streamed pose -> world x-hat reference, and the lock says so
        (design §4.1.3: 'flagged in the lock')."""
        markers = arm_corners()
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=None)
        assert lock.x_ref_source == "world_x_fallback"
        assert lock.x_residual_deg < 1e-9

    def test_stats_measure_a_skewed_parallelogram(self):
        """Shear a square: still a parallelogram (midpoints coincide, lines
        cross near 90), so it locks; the skew stat must report the
        deformation and the longest-chord sanity stat stays true."""
        shear = np.array([[1.0, 0.15, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        markers = rect_corners() @ shear.T
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        assert lock.midpoint_separation_m < 1e-15
        assert lock.intersection_residual_m < 1e-9
        assert lock.skew_deg > 5.0
        assert 85.0 < lock.diagonal_crossing_deg <= 90.0
        assert lock.diagonals_are_longest_chords
        assert lock.rest_frames == 3
        # Identical frames: sd is zero up to the 1-ulp rounding of the mean.
        assert lock.rest_marker_sd_m < 1e-15

    def test_shape_and_value_errors_raise(self):
        markers = arm_corners()
        with pytest.raises(ValueError):
            mf.compute_plate_lock(markers, plate=0, u_up=[0, 0, 1])   # no stack
        with pytest.raises(ValueError):
            mf.compute_plate_lock(np.zeros((0, 4, 3)), plate=0, u_up=[0, 0, 1])
        with pytest.raises(ValueError):
            mf.compute_plate_lock(rest_stack(markers), plate=0, u_up=[0, 0])
        with pytest.raises(ValueError):
            mf.compute_plate_lock(rest_stack(markers), plate=0,
                                  u_up=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            mf.compute_plate_lock(rest_stack(markers), plate=0, u_up=[0, 0, 1],
                                  streamed_rot=np.eye(4))
        bad = rest_stack(markers).copy()
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            mf.compute_plate_lock(bad, plate=0, u_up=[0, 0, 1])


# --------------------------------------------------------------------------
# 3. Lock refusals (design §4.1) and the consume-time validity binding
# --------------------------------------------------------------------------


class TestLockRefusals:
    def test_crossing_far_off_perpendicular_refused(self):
        """Diagonal lines crossing 40 deg off perpendicular: a mislabeled
        asset, not plate geometry (the real plates cross within 0.25 deg of
        90 — probe rectangle_stats).  §4.1.1's rev-3 refusal, replacing the
        old midpoint gate that the radial-arm hardware falsified."""
        markers = arm_corners(angles_deg=(45.0, 95.0, 225.0, 275.0))
        with pytest.raises(mf.LockRefusal, match="off perpendicular"):
            mf.compute_plate_lock(rest_stack(markers), plate=0,
                                  u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))

    def test_crossing_inside_the_band_still_locks(self):
        """20 deg off perpendicular is inside the 25 deg band: locks, and the
        radial-arm origin is still exact (the lines still cross at the
        centre)."""
        markers = arm_corners(angles_deg=(45.0, 115.0, 225.0, 295.0))
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        assert abs(lock.diagonal_crossing_deg - 70.0) < 1e-9
        t, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        assert np.max(np.abs(t[0:3, 3])) < TOL

    def test_midpoint_separation_is_recorded_not_refused(self):
        """The exact failure mode of the old formulation, pinned in reverse:
        the probe capture's plates carry 5.5-12.8 mm midpoint separations
        and every one of them refused to lock (probe_report.json
        ``lock_refusals``).  The rev-3 lock must swallow an 8.5 mm
        separation and record it."""
        lock = mf.compute_plate_lock(rest_stack(arm_corners()), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        assert lock.midpoint_separation_m > mf.MIDPOINT_TOL_M

    def test_near_horizontal_plate_refused(self):
        """u_up in the plate plane: |n.u_up| = 0 < cos 60 deg, the normal
        sign would be noise (review finding geo-3)."""
        markers = arm_corners()
        with pytest.raises(mf.LockRefusal, match="near-horizontal"):
            mf.compute_plate_lock(rest_stack(markers), plate=0,
                                  u_up=[1.0, 0.0, 0.0], streamed_rot=np.eye(3))
        # 75 deg off the normal is still inside the refusal band
        # (cos 75 = 0.26 < cos 60)...
        tilt = rodrigues([0.0, 1.0, 0.0], math.radians(75.0))
        with pytest.raises(mf.LockRefusal, match="near-horizontal"):
            mf.compute_plate_lock(rest_stack(markers), plate=0,
                                  u_up=tilt @ [0.0, 0.0, 1.0],
                                  streamed_rot=np.eye(3))
        # ... and 45 deg off (cos 45 = 0.71 > cos 60) is not.
        tilt = rodrigues([0.0, 1.0, 0.0], math.radians(45.0))
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=tilt @ [0.0, 0.0, 1.0],
                                     streamed_rot=np.eye(3))
        assert lock.n_dot_u_up > 0.5

    def test_collinear_markers_refused(self):
        markers = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0],
                            [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]])
        with pytest.raises(mf.LockRefusal):
            mf.compute_plate_lock(rest_stack(markers), plate=0,
                                  u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))

    def test_45_boundary_residual_is_recorded_not_refused(self):
        """A streamed reference 40 deg off the bracket x sits past the probe's
        30 deg gate.  The lock must *record* that number (the probe exits 3 on
        it — design §4.1.3) rather than refuse: a refusal here would degrade
        the probe to reports-only instead of firing the named gate."""
        markers = rect_corners()
        streamed = rot_z(math.radians(40.0))
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=streamed)
        assert lock.x_residual_deg == pytest.approx(40.0, abs=1e-9)
        assert lock.x_residual_deg > 30.0          # what the probe gates on
        # And well inside the boundary the chosen candidate is the right one:
        # the 40-deg reference still picked the candidate nearest to it, i.e.
        # the true bracket x, so the *frame* is exact even at the boundary.
        t, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        assert np.max(np.abs(t[0:3, 0:3] - np.eye(3))) < TOL

    def test_stale_lock_template_mismatch_on_consume(self):
        """The §4.1 validity binding: a lock consumed against rest data whose
        rigid shape no longer fits the template beyond noise is refused
        (review ops-5) — one registration RMS covers scale, marker order and
        plane-shape changes at once."""
        markers = arm_corners()
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        # Same session, same data: passes (returns None, raises nothing).
        assert mf.verify_lock_against_rest(lock, rest_stack(markers)) is None
        # Sub-noise jitter: still passes.
        rng = np.random.default_rng(3)
        jittered = rest_stack(markers, n=20) + rng.normal(0.0, 2e-4, (20, 4, 3))
        mf.verify_lock_against_rest(lock, jittered)
        # A re-created rigid body with 10 % longer arms: refused.
        with pytest.raises(mf.LockRefusal, match="stale"):
            mf.verify_lock_against_rest(lock, rest_stack(markers * 1.1))
        # A re-created asset with a different marker order: the template no
        # longer registers onto the renamed shape — refused.
        with pytest.raises(mf.LockRefusal, match="stale"):
            mf.verify_lock_against_rest(lock, rest_stack(markers[[1, 0, 2, 3]]))

    def test_verify_shape_errors_raise_value_error(self):
        markers = arm_corners()
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        with pytest.raises(ValueError):
            mf.verify_lock_against_rest(lock, np.zeros((3, 5, 3)))


# --------------------------------------------------------------------------
# 4. The per-frame solve: round trips (template registration, design §4.2)
# --------------------------------------------------------------------------


class TestInferPlateFrame:
    def test_round_trip_random_poses(self):
        """Lock+solve recovers the constructed frame: origin and rotation at
        machine precision, for every family, over random SE(3) poses — on
        the asymmetric radial-arm geometry (template registration has no
        parallelogram assumption to lean on)."""
        rng = np.random.default_rng(11)
        worst = 0.0
        for plate in range(6):
            for _ in range(20):
                pose = random_se3(rng)
                u_up = pose[0:3, 0:3] @ np.array([0.05, -0.02, 1.0])
                markers = markers_from_pose(pose, FAMILY_PHIS[plate])
                lock = mf.compute_plate_lock(rest_stack(markers), plate=plate,
                                             u_up=u_up,
                                             streamed_rot=pose[0:3, 0:3])
                t, q = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
                assert t is not None and q.reason == ""
                want_r = pose[0:3, 0:3] @ rot_z(FAMILY_PHIS[plate])
                worst = max(worst,
                            float(np.max(np.abs(t[0:3, 0:3] - want_r))),
                            float(np.max(np.abs(t[0:3, 3] - pose[0:3, 3]))))
        assert worst < TOL, f"worst round-trip error {worst:g}"

    def test_body_frame_is_inferred_times_rz_minus_phi(self):
        """The family correction the whole §4.3 pipeline stands on:
        ``R_body = R_inferred @ Rz(-phi_p)`` recovers the plate body rotation
        for a +45-family plate."""
        rng = np.random.default_rng(13)
        pose = random_se3(rng)
        markers = markers_from_pose(pose, math.pi / 4)
        lock = mf.compute_plate_lock(rest_stack(markers), plate=1,
                                     u_up=pose[0:3, 0:3] @ [0.0, 0.0, 1.0],
                                     streamed_rot=pose[0:3, 0:3])
        t, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        r_body = t[0:3, 0:3] @ rot_z(-math.pi / 4)
        assert np.max(np.abs(r_body - pose[0:3, 0:3])) < TOL

    def test_invariant_to_rigid_world_motion(self):
        """Move the whole volume: the inferred frame moves with it exactly —
        the per-frame solve consults nothing but the markers (design §4
        preamble), so this must be a pure equivariance."""
        rng = np.random.default_rng(17)
        pose = np.eye(4)
        pose[0:3, 3] = [0.1, 0.2, 0.3]
        markers = markers_from_pose(pose, 0.0)
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        t0, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        for _ in range(10):
            w = random_se3(rng)
            moved = markers @ w[0:3, 0:3].T + w[0:3, 3]
            t, _ = mf.infer_plate_frame(moved, ALL_TRACKED, lock)
            assert t is not None
            assert np.max(np.abs(t - w @ t0)) < TOL

    def test_noise_tolerance_at_the_measured_level(self):
        """0.05 mm marker noise (the top of the probe-measured resting
        jitter, 0.02-0.05 mm): origin within 0.1 mm, axes within 0.1 deg,
        and no gate fires — on the asymmetric geometry."""
        rng = np.random.default_rng(19)
        pose = random_se3(rng)
        u_up = pose[0:3, 0:3] @ np.array([0.0, 0.0, 1.0])
        clean = markers_from_pose(pose, math.pi / 4)
        lock = mf.compute_plate_lock(
            np.stack([clean + rng.normal(0.0, 5e-5, (4, 3)) for _ in range(60)]),
            plate=1, u_up=u_up, streamed_rot=pose[0:3, 0:3])
        want_r = pose[0:3, 0:3] @ rot_z(math.pi / 4)
        for _ in range(50):
            noisy = clean + rng.normal(0.0, 5e-5, (4, 3))
            t, q = mf.infer_plate_frame(noisy, ALL_TRACKED, lock)
            assert t is not None and q.reason == ""
            assert np.linalg.norm(t[0:3, 3] - pose[0:3, 3]) < 1e-4
            assert rot_angle(want_r.T @ t[0:3, 0:3]) < math.radians(0.1)
            assert q.rms_residual_m < 2e-4

    def test_quality_fields_on_a_clean_solve(self):
        markers = arm_corners() + np.array([0.0, 0.0, 0.5])
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        t, q = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        assert t is not None
        assert q.n_usable == 4 and q.used == (True, True, True, True)
        assert q.n_flagged_out == 0
        assert q.rms_residual_m < 1e-12
        assert q.reason == ""


# --------------------------------------------------------------------------
# 5. Drop-one robustness (design §4.2: exact for ANY geometry)
# --------------------------------------------------------------------------


class TestDropOne:
    def test_every_3_subset_equals_the_4_marker_solve(self):
        """On a noiseless plate of ANY geometry — here the asymmetric
        radial-arm one — every 3-marker registration is exact, so it equals
        the 4-marker answer to machine precision.  This is the claim the
        parallelogram formulation could only make for parallelograms
        (design §9 rev 3)."""
        rng = np.random.default_rng(23)
        for _ in range(10):
            pose = random_se3(rng)
            markers = markers_from_pose(pose, math.pi / 4)
            lock = mf.compute_plate_lock(rest_stack(markers), plate=1,
                                         u_up=pose[0:3, 0:3] @ [0.0, 0.0, 1.0],
                                         streamed_rot=pose[0:3, 0:3])
            t4, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
            for drop in range(4):
                flags = ALL_TRACKED.copy()
                flags[drop] = 0
                t3, q3 = mf.infer_plate_frame(markers, flags, lock)
                assert t3 is not None
                assert q3.n_usable == 3 and q3.n_flagged_out == 1
                assert not q3.used[drop]
                assert q3.rms_residual_m < 1e-12
                assert np.max(np.abs(t3 - t4)) < TOL, f"dropped {drop}"

    def test_3_subsets_under_noise_stay_close(self):
        """0.05 mm noise on 3-marker solves: the lever arm shrinks, so allow
        2x the 4-marker budget — still well under anything kinematically
        visible."""
        rng = np.random.default_rng(27)
        pose = random_se3(rng)
        clean = markers_from_pose(pose, 0.0)
        lock = mf.compute_plate_lock(
            np.stack([clean + rng.normal(0.0, 5e-5, (4, 3)) for _ in range(60)]),
            plate=0, u_up=pose[0:3, 0:3] @ [0.0, 0.0, 1.0],
            streamed_rot=pose[0:3, 0:3])
        want_r = pose[0:3, 0:3]
        for trial in range(25):
            noisy = clean + rng.normal(0.0, 5e-5, (4, 3))
            flags = ALL_TRACKED.copy()
            flags[trial % 4] = 0
            t3, q3 = mf.infer_plate_frame(noisy, flags, lock)
            assert t3 is not None and q3.n_usable == 3
            assert np.linalg.norm(t3[0:3, 3] - pose[0:3, 3]) < 2e-4
            assert rot_angle(want_r.T @ t3[0:3, 0:3]) < math.radians(0.2)

    def test_flagged_out_marker_with_nan_position_still_solves(self):
        """Design D4 semantics end to end: an *excluded* marker's position is
        never read, so Motive zeroing/NaN-ing an occluded marker cannot
        poison the 3-marker solve."""
        markers = arm_corners() + np.array([0.2, 0.0, 0.4])
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        t4, _ = mf.infer_plate_frame(markers, ALL_TRACKED, lock)
        broken = markers.copy()
        broken[2] = np.nan
        flags = np.array([1, 1, 0, 1], dtype=np.uint8)
        t3, q3 = mf.infer_plate_frame(broken, flags, lock)
        assert t3 is not None and q3.reason == ""
        assert np.max(np.abs(t3 - t4)) < TOL


# --------------------------------------------------------------------------
# 6. Per-frame gates (design §4.2, review findings geo-4 / int-11)
# --------------------------------------------------------------------------


class TestGates:
    @pytest.fixture()
    def locked(self):
        markers = arm_corners() + np.array([0.1, -0.2, 0.5])
        lock = mf.compute_plate_lock(rest_stack(markers), plate=0,
                                     u_up=[0.0, 0.0, 1.0], streamed_rot=np.eye(3))
        return markers, lock

    def test_any_label_swap_fires_the_template_rms_gate(self, locked):
        """The headline gate (geo-4): a mid-run swap of ANY two labels breaks
        the template fit by centimetres against 0.02-0.15 mm noise.  Note the
        along-diagonal swaps (0,2)/(1,3), which the old diagonal-length gate
        was structurally blind to — the registration residual catches them
        too."""
        markers, lock = locked
        for swap in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)):
            swapped = markers.copy()
            swapped[[swap[0], swap[1]]] = swapped[[swap[1], swap[0]]]
            t, q = mf.infer_plate_frame(swapped, ALL_TRACKED, lock)
            assert t is None, f"swap {swap} was not gated"
            assert q.reason == "template_rms_gate"
            assert q.rms_residual_m > mf.TEMPLATE_RMS_TOL_M

    def test_count_mismatch_returns_none(self, locked):
        """Marker-set count != the lock template's 4 rows: asset-order
        indexing may have shifted, so the conservative answer is no frame
        (int-11)."""
        markers, lock = locked
        t, q = mf.infer_plate_frame(markers[0:3], None, lock)
        assert t is None and q.reason == "marker_count_mismatch"
        five = np.vstack([markers, markers[0] + 0.01])
        t, q = mf.infer_plate_frame(five, None, lock)
        assert t is None and q.reason == "marker_count_mismatch"

    def test_fewer_than_3_usable_returns_none(self, locked):
        markers, lock = locked
        t, q = mf.infer_plate_frame(markers, np.array([1, 1, 0, 0], np.uint8),
                                    lock)
        assert t is None and q.reason == "too_few_usable"
        assert q.n_usable == 2 and q.n_flagged_out == 2

    def test_nan_in_a_used_marker_returns_none(self, locked):
        markers, lock = locked
        broken = markers.copy()
        broken[1, 2] = np.nan
        t, q = mf.infer_plate_frame(broken, ALL_TRACKED, lock)
        assert t is None and q.reason == "non_finite_marker"
        broken[1, 2] = np.inf
        t, q = mf.infer_plate_frame(broken, ALL_TRACKED, lock)
        assert t is None and q.reason == "non_finite_marker"

    def test_collinear_usable_subset_returns_none(self, locked):
        """The reflection/degeneracy guard (§4.2): three usable markers on a
        line leave the rotation about that line unobservable — ``None``
        (named), never a garbage frame.  Fires *before* the RMS gate so the
        reported reason is the true one."""
        markers, lock = locked
        garbage = markers.copy()
        garbage[0] = [0.0, 0.0, 0.5]
        garbage[1] = [0.05, 0.0, 0.5]
        garbage[2] = [0.10, 0.0, 0.5]
        flags = np.array([1, 1, 1, 0], dtype=np.uint8)
        t, q = mf.infer_plate_frame(garbage, flags, lock)
        assert t is None and q.reason == "degenerate_markers"
        # Fully collapsed markers: same guard.
        collapsed = np.tile(np.array([0.1, 0.2, 0.3]), (4, 1))
        t, q = mf.infer_plate_frame(collapsed, ALL_TRACKED, lock)
        assert t is None and q.reason == "degenerate_markers"

    def test_rms_threshold_is_respected(self, locked):
        """A 20 mm bend fails the 3 mm gate; a 1 mm bend passes with the
        residual honestly recorded — the gate is a threshold, not a cliff at
        zero."""
        markers, lock = locked
        bent = markers.copy()
        bent[0] += np.array([0.020, 0.0, 0.0])
        t, q = mf.infer_plate_frame(bent, ALL_TRACKED, lock)
        assert t is None and q.reason == "template_rms_gate"
        assert q.rms_residual_m > mf.TEMPLATE_RMS_TOL_M
        mild = markers.copy()
        mild[0] += np.array([0.001, 0.0, 0.0])
        t, q = mf.infer_plate_frame(mild, ALL_TRACKED, lock)
        assert t is not None and q.reason == ""
        assert 1e-5 < q.rms_residual_m < mf.TEMPLATE_RMS_TOL_M

    def test_flags_none_means_assumed_markers(self, locked):
        """Flags-unknown regime (§5): inference still runs on 4 assumed
        markers — the *caller* tags the regime, this function must not
        refuse."""
        markers, lock = locked
        t, q = mf.infer_plate_frame(markers, None, lock)
        assert t is not None and q.n_usable == 4

    def test_bad_shapes_raise(self, locked):
        markers, lock = locked
        with pytest.raises(ValueError):
            mf.infer_plate_frame(markers[:, 0:2], ALL_TRACKED, lock)
        with pytest.raises(ValueError):
            mf.infer_plate_frame(markers, np.ones(3, np.uint8), lock)


# --------------------------------------------------------------------------
# 7. infer_all: the frame-level wrapper
# --------------------------------------------------------------------------


class TestInferAll:
    def test_shapes_and_validity_mask(self):
        base = np.eye(4)
        locks = arm_locks(base)
        markers = arm_markers(np.zeros(12), base)
        poses, valid, quality = mf.infer_all(markers, None, locks)
        assert poses.shape == (mc.N_USED_RIGID_BODIES, 4, 4)
        assert valid.shape == (mc.N_USED_RIGID_BODIES,)
        assert valid[0:6].all() and not valid[6]       # no plate-6 lock (D6)
        assert np.array_equal(poses[6], np.eye(4))     # untouched row
        assert all(quality[p].reason == "" for p in range(6))

    def test_missing_plate_and_absent_markers(self):
        base = np.eye(4)
        locks = arm_locks(base)
        markers = arm_markers(np.zeros(12), base)
        del markers[3]
        poses, valid, quality = mf.infer_all(markers, None, locks)
        assert not valid[3] and quality[3].reason == "no_marker_set"
        assert np.array_equal(poses[3], np.eye(4))
        assert valid[[0, 1, 2, 4, 5]].all()
        # Marker sets absent for the whole frame (markers=None ring entry):
        poses, valid, quality = mf.infer_all(None, None, locks)
        assert not valid.any()
        assert all(q.reason == "no_marker_set" for q in quality.values())

    def test_flags_dict_is_consumed_per_plate(self):
        base = np.eye(4)
        locks = arm_locks(base)
        markers = arm_markers(np.zeros(12), base)
        flags = {p: ALL_TRACKED.copy() for p in range(6)}
        flags[2][0] = 0                                # one dropout on plate 2
        flags[4] = np.zeros(4, np.uint8)               # plate 4 fully occluded
        poses, valid, quality = mf.infer_all(markers, flags, locks)
        assert valid[2] and quality[2].n_usable == 3
        assert not valid[4] and quality[4].reason == "too_few_usable"

    def test_out_of_range_lock_raises(self):
        base = np.eye(4)
        locks = arm_locks(base)
        locks[9] = locks[0]
        with pytest.raises(ValueError):
            mf.infer_all(arm_markers(np.zeros(12), base), None, locks)


# --------------------------------------------------------------------------
# 8. Frozen goldens (NO reference implementation exists — these ARE the pin)
# --------------------------------------------------------------------------


class TestGoldensFrozen:
    def test_golden_lock_values(self):
        """CW-labelled markers, +45 family, 8 deg of simulated alignment
        error: the stored cyclic order is the *reversed* enumeration (normal
        sign followed u_up, not the label order), the ordered-diagonal thetas
        are 135/45 deg, and the disambiguation recorded exactly the 8 deg it
        shrugged off.  All literals unchanged by the rev-3 template revision:
        on a square the diagonal-line intersection IS the midpoint mean."""
        lock = golden_lock()
        assert lock.cyclic_order == (0, 3, 2, 1)
        assert abs(lock.diag_thetas_rad[0] - 2.3561944901923444) < 1e-12
        assert abs(lock.diag_thetas_rad[1] - 0.7853981633974483) < 1e-12
        assert lock.x_residual_deg == pytest.approx(8.0, abs=1e-9)
        assert lock.phi_rad == pytest.approx(math.pi / 4, abs=1e-15)
        assert lock.n_dot_u_up == pytest.approx(0.9938079899999063, abs=1e-12)

    def test_golden_template(self):
        """The rev-3 addition, pinned against exact arithmetic: the golden
        markers are the CW-permuted square corners rigidly placed at
        GOLDEN_T, so the template (offsets in the locked frame) must be
        exactly that corner table — label i holding corner [0, 3, 2, 1][i]."""
        lock = golden_lock()
        assert np.max(np.abs(np.asarray(lock.template_m)
                             - rect_corners()[[0, 3, 2, 1]])) < 1e-12
        assert lock.midpoint_separation_m < 1e-12   # square: midpoints agree
        assert lock.intersection_residual_m < 1e-9

    def test_golden_frame(self):
        """The headline golden: frozen markers -> frozen SE(3), now solved by
        template registration instead of diagonal midpoints — same literals,
        because the geometry is exact either way."""
        t, q = mf.infer_plate_frame(GOLDEN_MARKERS, ALL_TRACKED, golden_lock())
        assert t is not None and q.reason == ""
        assert np.max(np.abs(t - GOLDEN_T)) < GOLDEN_TOL

    def test_golden_family_correction(self):
        """Rz(-phi) recovers the frozen body rotation — pinning the CCW sense
        of the family correction against literals, not against rot_z."""
        t, _ = mf.infer_plate_frame(GOLDEN_MARKERS, ALL_TRACKED, golden_lock())
        s = 0.7071067811865476
        rz_m45 = np.array([[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]])
        assert np.max(np.abs(t[0:3, 0:3] @ rz_m45 - GOLDEN_R_PLATE)) < GOLDEN_TOL

    def test_golden_normal_sign(self):
        """The frame z is the *up-oriented* plate normal: positive dot with
        u_up, literal third column."""
        t, _ = mf.infer_plate_frame(GOLDEN_MARKERS, ALL_TRACKED, golden_lock())
        assert np.max(np.abs(t[0:3, 2] - GOLDEN_R_PLATE[:, 2])) < GOLDEN_TOL
        assert float(t[0:3, 2] @ GOLDEN_U_UP) > 0.0

    def test_golden_q_from_frames(self):
        """Frozen chain frames -> the exact (t1, t2, t3, t4) they encode.
        Pins the t3/t4 basis change (Rz(+45) conjugation) and the Rz(-phi)
        body conversion inside q_from_frames, via literals only."""
        q = mf.q_from_frames(golden_chain_frames())
        assert q is not None
        want = np.zeros(12)
        want[0:4] = GOLDEN_Q_SEG1
        assert np.max(np.abs(q - want)) < GOLDEN_TOL

    def test_golden_r_e_provenance(self):
        """Re-derive the frozen distal rotation longhand — explicit twist
        axes (x+y)/sqrt2 and (-x+y)/sqrt2, test-local Rodrigues — so the
        literal cannot quietly rot into a self-consistent fiction."""
        t1, t2, t3, t4 = GOLDEN_Q_SEG1
        s2 = 2.0 ** -0.5
        r_e = (rodrigues([1, 0, 0], t1) @ rodrigues([0, 1, 0], t2)
               @ rodrigues([s2, s2, 0], t3) @ rodrigues([-s2, s2, 0], t4))
        assert np.max(np.abs(r_e - GOLDEN_R_E)) < 1e-15
        o1 = rodrigues([1, 0, 0], t1) @ rodrigues([0, 1, 0], t2) @ np.array(
            [0.0, 0.0, -0.22])
        assert np.max(np.abs(o1 - GOLDEN_O1)) < 1e-15

    def test_golden_q_is_sensitive_to_the_basis_change(self, monkeypatch):
        """Proof the golden has teeth: conjugating by Rz(-45) instead of
        Rz(+45) (the plausible wrong answer — it is what mocap_constants
        carries) must move the recovered t3/t4 by a lot."""
        monkeypatch.setattr(mf, "RZ45", mf.RZ45.T.copy())
        q = mf.q_from_frames(golden_chain_frames())
        want = np.zeros(12)
        want[0:4] = GOLDEN_Q_SEG1
        assert np.max(np.abs(q - want)) > 0.05

    def test_golden_markers_provenance(self):
        """The frozen markers really are the golden pose's +45 bracket with
        CW labels — re-derived from the exact R_PLATE literal."""
        cw = rect_corners()[[0, 3, 2, 1]]
        rebuilt = cw @ (GOLDEN_R_PLATE @ rot_z(math.pi / 4)).T + GOLDEN_ORIGIN
        assert np.max(np.abs(rebuilt - GOLDEN_MARKERS)) < 1e-15
        # And R_PLATE is exactly orthonormal (Pythagorean construction).
        assert np.max(np.abs(GOLDEN_R_PLATE.T @ GOLDEN_R_PLATE - np.eye(3))) < 1e-15


# --------------------------------------------------------------------------
# 9. Co-rigid consistency (design §2 / benchmark §8.6's self-check)
# --------------------------------------------------------------------------


class TestCoRigid:
    def test_pairs_agree_frame_by_frame(self):
        """Plates {1,2} and {3,4} share a rigid connector: their inferred
        *body* frames (Rz(-phi) applied) must have identity relative rotation
        and a Tz(-JD) offset at every q — the free self-check the benchmark
        uses (§8.6), here proven on synthetic segments including dropouts,
        with the real asymmetric-arm geometry."""
        rng = np.random.default_rng(29)
        base = random_se3(rng)
        locks = arm_locks(base)
        jd = rp.segment_jds()
        for trial in range(15):
            q = rng.uniform(-0.5, 0.5, 12)
            markers = arm_markers(q, base)
            flags = {p: ALL_TRACKED.copy() for p in range(6)}
            if trial % 3 == 1:                       # exercise the 3-marker path
                flags[1][int(rng.integers(4))] = 0
                flags[4][int(rng.integers(4))] = 0
            poses, valid, _ = mf.infer_all(markers, flags, locks)
            assert valid[0:6].all()
            for a, b, gap in ((1, 2, jd[1]), (3, 4, jd[2])):
                ra = poses[a][0:3, 0:3] @ rot_z(-FAMILY_PHIS[a])
                rb = poses[b][0:3, 0:3] @ rot_z(-FAMILY_PHIS[b])
                assert rot_angle(ra.T @ rb) < 1e-7
                d_body = ra.T @ (poses[b][0:3, 3] - poses[a][0:3, 3])
                assert np.max(np.abs(d_body - [0.0, 0.0, -gap])) < TOL


# --------------------------------------------------------------------------
# 10. q_from_frames vs the kinematics port (design §4.3 / §9)
# --------------------------------------------------------------------------


class TestQFromFrames:
    @pytest.mark.parametrize("joint", range(12))
    @pytest.mark.parametrize("amp", [0.3, -0.3, 0.05])
    def test_single_joint_machine_precision(self, joint, amp):
        """All 12 joints from body frames alone — including q10/q11, which
        live only in plate 5's rotation."""
        q = np.zeros(12)
        q[joint] = amp
        q_rec = mf.q_from_frames(plate_transforms(q), phis=np.zeros(6))
        assert q_rec is not None
        assert np.max(np.abs(q_rec - q)) < TOL

    def test_multi_joint_machine_precision(self):
        """The §4.3 claim that separates this from mk8: multi-joint q is
        exact by construction (no swing-only alignment loses the link-axis
        twist), so the documented mk8 error budget simply does not appear."""
        rng = np.random.default_rng(31)
        worst = 0.0
        for _ in range(200):
            q = rng.uniform(-0.6, 0.6, 12)
            q_rec = mf.q_from_frames(plate_transforms(q), phis=np.zeros(6))
            assert q_rec is not None
            worst = max(worst, float(np.max(np.abs(q_rec - q))))
        assert worst < TOL, f"multi-joint error {worst:g}"

    def test_default_phis_undo_the_marker_families(self):
        """Marker frames (body @ Rz(phi_p)) with the default family table:
        q_from_frames' own Rz(-phi) conversion must land on the same q."""
        rng = np.random.default_rng(37)
        for _ in range(50):
            q = rng.uniform(-0.6, 0.6, 12)
            body = plate_transforms(q)
            frames = body.copy()
            for p in range(6):
                frames[p, 0:3, 0:3] = body[p, 0:3, 0:3] @ rot_z(FAMILY_PHIS[p])
            q_rec = mf.q_from_frames(frames)          # phis=None -> defaults
            assert q_rec is not None
            assert np.max(np.abs(q_rec - q)) < TOL

    def test_invariant_to_rigid_world_motion(self):
        """Only relative rotations and body-frame differences are read, so a
        rigid motion of the whole volume changes nothing."""
        rng = np.random.default_rng(41)
        q = rng.uniform(-0.4, 0.4, 12)
        frames = plate_transforms(q)
        q0 = mf.q_from_frames(frames, phis=np.zeros(6))
        for _ in range(10):
            w = random_se3(rng)
            q_rec = mf.q_from_frames(w @ frames, phis=np.zeros(6))
            assert np.max(np.abs(q_rec - q0)) < TOL

    def test_seven_row_input_ignores_the_ee_row(self):
        """The (7, 4, 4) infer_all output drops straight in; row 6 (EE plate,
        identity when uninferred) must not influence q."""
        q = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, -0.3, 0.2,
                      -0.1, 0.3, 0.15, -0.25])
        frames6 = plate_transforms(q)
        frames7 = np.concatenate([frames6, np.eye(4)[None]], axis=0)
        a = mf.q_from_frames(frames6, phis=np.zeros(6))
        b = mf.q_from_frames(frames7, phis=np.zeros(6))
        assert np.array_equal(a, b)

    def test_degenerate_frames_give_none(self):
        q = np.zeros(12)
        frames = plate_transforms(q)
        # Collapsed P-D pair: both u3 plates at the same point.
        collapsed = frames.copy()
        collapsed[3, 0:3, 3] = collapsed[2, 0:3, 3]
        assert mf.q_from_frames(collapsed, phis=np.zeros(6)) is None
        # Non-finite rotation.
        broken = frames.copy()
        broken[1, 0, 0] = np.nan
        assert mf.q_from_frames(broken, phis=np.zeros(6)) is None
        # Non-finite origin.
        broken = frames.copy()
        broken[4, 0:3, 3] = np.inf
        assert mf.q_from_frames(broken, phis=np.zeros(6)) is None

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            mf.q_from_frames(np.tile(np.eye(4), (5, 1, 1)))
        with pytest.raises(ValueError):
            mf.q_from_frames(np.zeros((6, 3, 3)))
        with pytest.raises(ValueError):
            mf.q_from_frames(plate_transforms(np.zeros(12)), phis=np.zeros(5))


# --------------------------------------------------------------------------
# 11. The full pipeline: markers -> locks -> frames -> q (the §9 headline)
# --------------------------------------------------------------------------


class TestFullPipeline:
    def test_markers_to_q_machine_precision(self):
        """End to end, exactly as the benchmark will run it: rest markers
        under an arbitrary base -> locks (u_up from the arm), moving markers
        -> infer_all -> q_from_frames, recovering fkine-generated q to
        machine precision — single and multi-joint, any base pose, on the
        asymmetric radial-arm geometry."""
        rng = np.random.default_rng(43)
        worst = 0.0
        for _ in range(5):
            base = random_se3(rng)
            locks = arm_locks(base)
            cases = [rng.uniform(-0.5, 0.5, 12) for _ in range(4)]
            for j in (0, 5, 10, 11):                 # a few single-joint cases
                q = np.zeros(12)
                q[j] = 0.3
                cases.append(q)
            for q in cases:
                poses, valid, quality = mf.infer_all(
                    arm_markers(q, base), None, locks)
                assert valid[0:6].all(), quality
                q_rec = mf.q_from_frames(poses[0:6])
                assert q_rec is not None
                worst = max(worst, float(np.max(np.abs(q_rec - q))))
        assert worst < TOL, f"full-pipeline error {worst:g}"

    def test_pipeline_survives_one_dropout_per_plate(self):
        """One marker flagged out on every plate at once: still exact —
        drop-one exactness holds for any geometry under template
        registration, composed through the whole chain."""
        rng = np.random.default_rng(47)
        base = random_se3(rng)
        locks = arm_locks(base)
        q = rng.uniform(-0.4, 0.4, 12)
        flags = {p: ALL_TRACKED.copy() for p in range(6)}
        for p in range(6):
            flags[p][int(rng.integers(4))] = 0
        poses, valid, quality = mf.infer_all(arm_markers(q, base), flags, locks)
        assert valid[0:6].all()
        assert all(quality[p].n_usable == 3 for p in range(6))
        q_rec = mf.q_from_frames(poses[0:6])
        assert np.max(np.abs(q_rec - q)) < TOL

    def test_pipeline_reports_a_gated_plate_honestly(self):
        """A label swap mid-chain: that plate goes invalid with the named
        gate, no q is computed from the poisoned pose array."""
        base = np.eye(4)
        locks = arm_locks(base)
        q = np.zeros(12)
        markers = arm_markers(q, base)
        markers[3] = markers[3][[1, 0, 2, 3]]           # swapped labels
        poses, valid, quality = mf.infer_all(markers, None, locks)
        assert not valid[3]
        assert quality[3].reason == "template_rms_gate"
        assert valid[[0, 1, 2, 4, 5]].all()


# --------------------------------------------------------------------------
# 12. Lock serialization (locks.json round trip)
# --------------------------------------------------------------------------


class TestLockJson:
    def test_to_dict_is_asdict(self):
        """The probe writes ``dataclasses.asdict(lock)``; ``to_dict`` must be
        the same payload so both paths produce identical locks.json."""
        lock = golden_lock()
        assert lock.to_dict() == dataclasses.asdict(lock)

    def test_json_round_trip_preserves_the_lock(self):
        """to_dict -> json -> from_dict is the identity, and the round-tripped
        lock still drives an exact solve (the template survives)."""
        lock = golden_lock()
        rebuilt = mf.PlateLock.from_dict(json.loads(json.dumps(lock.to_dict())))
        assert rebuilt == lock
        t0, _ = mf.infer_plate_frame(GOLDEN_MARKERS, ALL_TRACKED, lock)
        t1, _ = mf.infer_plate_frame(GOLDEN_MARKERS, ALL_TRACKED, rebuilt)
        assert np.array_equal(t0, t1)

    def test_lock_payload_is_json_serializable_as_is(self):
        """No numpy scalars/arrays may leak into the dataclass: the probe
        dumps it with a default= fallback that would silently stringify
        them, and from_dict would then not round-trip."""
        payload = golden_lock().to_dict()
        text = json.dumps(payload)                  # no default= on purpose
        assert isinstance(json.loads(text), dict)


# --------------------------------------------------------------------------
# 13. Acceptance on the real capture (probe_20260811_live2)
# --------------------------------------------------------------------------
# The measured hardware that drove rev 3: 30 s of static capture at 120 Hz,
# six plates, four labeled markers each, all tracked.  Its probe_report.json
# records diagonal midpoint separations of 5.5-12.8 mm — every plate REFUSED
# to lock under the midpoint/parallelogram formulation ("plate 0: diagonal
# midpoints disagree by 11.61 mm").  The template formulation must lock and
# solve this data cleanly; the numbers below are the acceptance bar.

_PROBE_DIR = os.path.join(_ROOT, "UMArm_ROBOT_CONTROL", "logs",
                          "probe_20260811_live2")
_PROBE_MARKERS_CSV = os.path.join(_PROBE_DIR, "probe_markers.csv")
_PROBE_POSES_CSV = os.path.join(_PROBE_DIR, "probe_poses.csv")

_probe_present = (os.path.exists(_PROBE_MARKERS_CSV)
                  and os.path.exists(_PROBE_POSES_CSV))


def _load_probe_capture():
    """Long-format CSVs -> dense per-frame arrays for plates 0-5.

    Returns ``(pos, trk, streamed_pos, streamed_rot)``: marker positions
    ``(n, 6, 4, 3)`` (NaN where absent), tracked flags ``(n, 6, 4)`` uint8,
    and per-plate mean streamed position / mean-quaternion rotation from the
    poses CSV (the lock inputs: u_up = plate0 - plate5, streamed_rot as the
    x-disambiguation reference — exactly what the probe hands the lock)."""
    raw = np.genfromtxt(_PROBE_MARKERS_CSV, delimiter=",", skip_header=1)
    frames = raw[:, 1].astype(np.int64)
    plates = raw[:, 3].astype(int)
    marks = raw[:, 4].astype(int)
    tracked = raw[:, 8]
    uf, inv = np.unique(frames, return_inverse=True)
    n = uf.size
    pos = np.full((n, 6, 4, 3), np.nan)
    trk = np.zeros((n, 6, 4), dtype=np.uint8)
    sel = (plates >= 0) & (plates < 6) & (marks >= 0) & (marks < 4)
    pos[inv[sel], plates[sel], marks[sel]] = raw[sel, 5:8]
    tr = np.where(np.isfinite(tracked), tracked, 0.0)
    trk[inv[sel], plates[sel], marks[sel]] = tr[sel].astype(np.uint8)

    praw = np.genfromtxt(_PROBE_POSES_CSV, delimiter=",", skip_header=1)
    streamed_pos, streamed_rot = {}, {}
    for p in range(6):
        rows = praw[praw[:, 2] == p]
        streamed_pos[p] = rows[:, 7:10].mean(axis=0)
        quats = rows[:, 3:7]
        # Hemisphere-align before averaging (q and -q are the same rotation).
        quats = np.where((quats @ quats[0])[:, None] < 0.0, -quats, quats)
        streamed_rot[p] = quat_xyzw_to_matrix(quats.mean(axis=0))
    return pos, trk, streamed_pos, streamed_rot


@pytest.fixture(scope="module")
def solved():
    """Locks + full-window per-frame solves + strided drop-one over the real
    probe capture, computed once for the module (only requested by
    :class:`TestAcceptanceRealProbe`, which is skipped when the capture is
    absent)."""
    pos, trk, streamed_pos, streamed_rot = _load_probe_capture()
    n = pos.shape[0]
    u_up = streamed_pos[0] - streamed_pos[5]         # the arm hangs (design §2)
    locks, results = {}, {}
    for p in range(6):
        rest = (trk[:, p].sum(axis=1) == 4) & np.isfinite(
            pos[:, p]).all(axis=(1, 2))
        locks[p] = mf.compute_plate_lock(
            pos[rest, p], plate=p, u_up=u_up,
            streamed_rot=streamed_rot[p])
        origins = np.full((n, 3), np.nan)
        rots = np.full((n, 3, 3), np.nan)
        rms = np.full(n, np.nan)
        ok = np.zeros(n, dtype=bool)
        for k in range(n):
            t, q = mf.infer_plate_frame(pos[k, p], trk[k, p], locks[p])
            if t is not None:
                ok[k] = True
                origins[k] = t[0:3, 3]
                rots[k] = t[0:3, 0:3]
                rms[k] = q.rms_residual_m
        # Drop-one, strided (every 4th solved frame keeps ~900 samples
        # per marker — plenty for a p95 — at a quarter of the cost).
        drop_dorg = {j: [] for j in range(4)}
        drop_dax = {j: [] for j in range(4)}
        drop_gated = {j: 0 for j in range(4)}
        for k in np.flatnonzero(ok)[::4]:
            for j in range(4):
                fl = trk[k, p].copy()
                fl[j] = 0
                t3, _q3 = mf.infer_plate_frame(pos[k, p], fl, locks[p])
                if t3 is None:
                    drop_gated[j] += 1
                    continue
                drop_dorg[j].append(float(np.linalg.norm(
                    t3[0:3, 3] - origins[k])))
                drop_dax[j].append(math.degrees(rot_angle(
                    t3[0:3, 0:3].T @ rots[k])))
        results[p] = dict(ok=ok, origins=origins, rms=rms,
                          drop_dorg=drop_dorg, drop_dax=drop_dax,
                          drop_gated=drop_gated)
    return dict(n=n, locks=locks, results=results)


@pytest.mark.skipif(not _probe_present,
                    reason="real probe capture probe_20260811_live2 not present")
class TestAcceptanceRealProbe:
    def test_every_plate_locks(self, solved):
        """All six plates lock — the data the old formulation refused
        wholesale — and the locks carry the measured asymmetry: midpoint
        separations inside the probe-reported 5.5-12.8 mm band, diagonal
        lines within 1 deg of perpendicular, intersection residual at the
        noise floor."""
        locks = solved["locks"]
        assert sorted(locks) == [0, 1, 2, 3, 4, 5]
        for p, lock in locks.items():
            # probe_report.json rectangle_stats: 5.48-12.75 mm — every plate
            # past the old 5 mm refusal.
            assert 0.004 < lock.midpoint_separation_m < 0.014, f"plate {p}"
            assert abs(lock.diagonal_crossing_deg - 90.0) < 1.0, f"plate {p}"
            assert lock.intersection_residual_m < 1e-3, f"plate {p}"
            assert lock.out_of_plane_rms_m < 2e-4, f"plate {p}"  # <= 0.15 mm meas.

    def test_solve_rate(self, solved):
        """>= 99 % of the 3601 frames solve on every plate (the capture is
        clean: all markers tracked throughout)."""
        for p, res in solved["results"].items():
            rate = float(res["ok"].mean())
            assert rate >= 0.99, f"plate {p}: solve rate {rate:.4f}"

    def test_origin_jitter(self, solved):
        """Static capture: per-plate origin jitter (RMS deviation from the
        mean) under 0.5 mm — the acceptance bar; the measured marker sd is
        0.02-0.15 mm, so most of the budget is headroom."""
        for p, res in solved["results"].items():
            o = res["origins"][res["ok"]]
            dev = o - o.mean(axis=0)
            jitter = float(np.sqrt(np.mean(np.sum(dev * dev, axis=1))))
            assert jitter < 5e-4, f"plate {p}: origin jitter {jitter * 1e3:.3f} mm"

    def test_rms_residual(self, solved):
        """Per-plate registration RMS residual p95 under 1 mm: the plates are
        rigid and the template captures their true (non-parallelogram)
        shape, so the residual is marker noise, not model error."""
        for p, res in solved["results"].items():
            r = res["rms"][res["ok"]]
            p95 = float(np.percentile(r, 95.0))
            assert p95 < 1e-3, f"plate {p}: rms p95 {p95 * 1e3:.3f} mm"

    def test_drop_one_agreement(self, solved):
        """Each 3-subset solve agrees with the 4-marker solve: p95 origin
        difference < 1 mm and p95 axis difference < 0.5 deg, per plate per
        dropped marker, with nothing gated — the one-dropout robustness the
        design promises (§2), now exact-by-construction on asymmetric arms."""
        for p, res in solved["results"].items():
            for j in range(4):
                assert res["drop_gated"][j] == 0, f"plate {p} marker {j}"
                dorg = np.asarray(res["drop_dorg"][j])
                dax = np.asarray(res["drop_dax"][j])
                assert dorg.size > 100
                p95_o = float(np.percentile(dorg, 95.0))
                p95_a = float(np.percentile(dax, 95.0))
                assert p95_o < 1e-3, (
                    f"plate {p} marker {j}: drop-one origin p95 "
                    f"{p95_o * 1e3:.3f} mm")
                assert p95_a < 0.5, (
                    f"plate {p} marker {j}: drop-one axis p95 {p95_a:.3f} deg")
