"""Tests for the forward-kinematics port.  Pure math: no sockets, no serial.

Four independent things are checked, because each catches what the others
cannot (the layering follows ``UMArm_MOCAP/test_mocap_to_q.py``):

1. **Structure** (:class:`TestParams` .. :class:`TestMountingModel`) — facts
   the design derives from the geometry and that need no oracle: the distal
   u-joint centre is invariant to t3/t4 (both axes pass through it), the q=0
   chain offsets are the structure doc's u-joint heights, the segment spans
   are **bit-exact** to ``PLATE_CHAIN_NOMINAL_M`` (the cross-package test also
   asserts ``arm_constants``' literal copy agrees, imported lazily so no
   runtime dependency appears — design D8), and the per-plate mounting model
   holds: plates 0/2/4 ignore their own joint, plate 5's rotation carries
   q[10:12], co-rigid pairs {1,2}/{3,4} have identity relative rotation.

2. **Frozen goldens** (:class:`TestGoldens`) — literal q -> literal numbers,
   blessed against the legacy ``fkine_mk5`` once (agreement 2.2e-16) and
   frozen, so a fresh clone with no access to the OneDrive tree still reports
   a *red* suite if a sign, a factor order or a parameter drifts.

3. **Round trip** (:class:`TestRoundTrip`) — synthetic plate poses built from
   :func:`plate_transforms` fed to the *shipped* ``UMArm_MOCAP.mocap_to_q``:
   single-joint q comes back at machine precision for **all 12 joints**
   (q10/q11 through plate 5's z-column — the mounting model is what makes that
   work), invariant to an arbitrary SE(3) base.  Multi-joint, mk8's swing-only
   extraction loses link-axis twist; the documented error budget (design §1:
   <= 3.7e-4 rad at |q| <= 0.05) is pinned as *present*, not fixed — a
   machine-precision multi-joint result here would mean the test lost its
   teeth, not that the code got better.

4. **Oracle equivalence** (:class:`TestOracleExponentials`,
   :class:`TestOracleFkine`) — against the legacy tree, imported by
   ``sys.path`` **for the test only**.  Two gates, deliberately separate:
   ``exponentials_mk5``/``robot_constants`` are numpy-only and skip only when
   the OneDrive tree is absent; ``kinematics_mp`` imports scipy at module
   scope, so the ``fkine_mk5`` comparison additionally skips where scipy is
   missing.  The legacy root is *appended* to ``sys.path``, never prepended:
   it contains ``NatNetClient.py``/``MoCapData.py`` name-twins of the vendored
   SDK, and the repo's copies must keep winning.

Run:  ``python -m pytest UMArm_KINEMATICS/test_fkine.py -q``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Repo root on the path so both sibling packages import the same way pytest
# finds this file — from the root or from inside the package directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from UMArm_KINEMATICS import robot_params as rp  # noqa: E402
from UMArm_KINEMATICS.fkine import (  # noqa: E402
    fkine,
    plate_transforms,
    predict_spatial,
    segment_transform,
    segment_twists,
    twist_exp,
    ujoint_centres,
)
from UMArm_MOCAP.mocap_to_q import mocap_to_q  # noqa: E402

# --------------------------------------------------------------------------
# The oracle: the legacy implementation, TEST ONLY
# --------------------------------------------------------------------------
# Read-only reference repo.  Imported here and nowhere else; when the path is
# wrong or the folder is gone the oracle tests skip and the rest still run.
# APPENDED to sys.path (not prepended): the legacy root carries top-level
# NatNetClient.py / MoCapData.py / DataDescriptions.py, the same module names
# the vendored SDK in UMArm_MOCAP/natnet_sdk uses, and the repo's own copies
# must always shadow the research tree's.

LEGACY_ROOT = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Umich", "PHD_Courses", "Research", "project2023_variable_stiffness",
    "UMArm_compliance_TRO",
)

_legacy_exp = None          # exponentials_mk5 — numpy-only
_legacy_rc = None           # robot_constants — numpy-only
_legacy_km = None           # kinematics_mp — imports scipy at module scope
_exp_why = ""
_km_why = ""
if os.path.isdir(LEGACY_ROOT):
    if LEGACY_ROOT not in sys.path:
        sys.path.append(LEGACY_ROOT)
    try:
        import exponentials_mk5 as _legacy_exp  # noqa: E402
        import robot_constants as _legacy_rc  # noqa: E402
    except Exception as exc:  # pragma: no cover - syntax drift, moved files, ...
        _legacy_exp = _legacy_rc = None
        _exp_why = f"legacy numpy-only import failed: {exc}"
    try:
        import kinematics_mp as _legacy_km  # noqa: E402
    except Exception as exc:  # scipy missing is the expected reason
        _legacy_km = None
        _km_why = f"legacy kinematics_mp import failed (scipy?): {exc}"
else:
    _exp_why = _km_why = f"legacy repo not found at {LEGACY_ROOT}"

needs_oracle = pytest.mark.skipif(_legacy_exp is None, reason=_exp_why)
needs_fkine_oracle = pytest.mark.skipif(_legacy_km is None, reason=_km_why)

#: Oracle equivalence budget from the design spec (§3).  Recon measured the
#: actual gap at ~4e-16 (a Rodrigues product against a Mathematica closed
#: form), so 1e-12 is nearly four orders of margin, not a fudge factor.
ORACLE_TOL = 1e-12

#: Round-trip budget: "machine precision" per the design.  Measured worst case
#: is 5.0e-16 with an identity base and 3.8e-15 with a random SE(3) base, so
#: 1e-12 keeps three orders of headroom while still catching any real defect
#: (a wrong mounting side moves q by whole degrees, not femtoradians).
TRIP_TOL = 1e-12


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


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


def build_homos(q, base=None) -> np.ndarray:
    """``(7, 4, 4)`` pose array for ``mocap_to_q``, all rows from the port.

    Rows 0..5 are ``base @ plate_transforms(q)`` — origin *and* orientation,
    so plate 5's z-column carries q[10:12] exactly as the mounting model says.
    ``mocap_to_q`` demands at least 7 rows (``mc.N_USED_RIGID_BODIES``) but
    reads only positions 0..5 and rotations 0 and 5; row 6 (the EE plate,
    body 506 — absent from the live volume) is filled with the last-plate pose
    so the array is a legal, finite SE(3) stack.
    """
    base = np.eye(4) if base is None else np.asarray(base, dtype=float)
    plates = plate_transforms(q)
    homos = np.tile(np.eye(4), (7, 1, 1))
    homos[0:6] = base @ plates
    homos[6] = base @ fkine(q)
    return homos


def assert_se3(t: np.ndarray, atol: float = 1e-14) -> None:
    r = t[0:3, 0:3]
    assert np.allclose(r.T @ r, np.eye(3), atol=atol)
    assert np.linalg.det(r) > 0.0
    assert np.array_equal(t[3], np.array([0.0, 0.0, 0.0, 1.0]))


#: q=0 u-joint centre heights (metres): the structure doc's values
#: (``UMArm_structure.html`` §2, quoted in ``arm_constants.py:536-540``),
#: written as the exact partial sums of the five chain gaps.
Q0_CENTRE_Z = np.array([0.0, -0.218868, -0.266739, -0.461279, -0.508661, -0.692772])


# --------------------------------------------------------------------------
# 1. The parameter table and the chain nominals
# --------------------------------------------------------------------------


class TestParams:
    def test_table_is_the_legacy_literals(self):
        """The whole (3, 10) table, frozen.  Bit-equal, not close: these are
        transcribed literals and any rounding means a typo, not a tolerance."""
        expected = np.array([
            [0.10, 0.05, 0.0, 0.0, 0.0285, 0.0285, 0.03, 0.03, 0.161868, 0.0],
            [0.05, 0.05, 0.0, 0.0, 0.0285, 0.0285, 0.03, 0.03, 0.137540, 0.047871],
            [0.05, 0.05, 0.0, 0.0, 0.0285, 0.0285, 0.03, 0.03, 0.127111, 0.047382],
        ])
        assert rp.DEFAULT_PARAMS.shape == (3, 10)
        assert np.array_equal(rp.DEFAULT_PARAMS, expected)
        assert rp.PARAM_COLUMNS == ("JA1", "JA2", "UC1", "UC2", "AA1", "AA2",
                                    "AO1", "AO2", "LL", "JD")

    def test_uc_is_zero_hardware_fact(self):
        """UC1 = UC2 = 0: plate centre *is* joint centre
        (``robot_constants.py:23,32,41``).  The 5-parameter benchmark fit
        leans on this, so it is pinned as data, not assumed in code."""
        assert np.all(rp.DEFAULT_PARAMS[:, rp.COL_UC1] == 0.0)
        assert np.all(rp.DEFAULT_PARAMS[:, rp.COL_UC2] == 0.0)

    def test_spans_and_jds(self):
        assert np.array_equal(rp.segment_spans(), [0.218868, 0.194540, 0.184111])
        assert np.array_equal(rp.segment_jds(), [0.0, 0.047871, 0.047382])

    def test_plate_chain_is_bit_exact_to_the_nominals(self):
        """The float sums land on the same doubles as the literals — checked
        with ``==``, because the design says *exactly*, not *approximately*."""
        assert rp.plate_chain_m() == rp.PLATE_CHAIN_NOMINAL_M
        assert rp.PLATE_CHAIN_NOMINAL_M == (0.218868, 0.047871, 0.194540,
                                            0.047382, 0.184111)

    def test_cross_package_equality_with_arm_constants(self):
        """Design D8: this module is the authoritative home; ``arm_constants``
        keeps a literal copy and **this test** is the only coupling between
        them — imported lazily, so no runtime dependency exists in either
        direction and a probe machine never needs the bus stack.

        PORT NOTE (this workspace).  ``UMArm_ROBOT_CONTROL`` is the RS485 bus
        stack and is deliberately not carried here, so the cross-check has no
        second copy to compare against and skips.  It is kept rather than
        deleted because the CAN workspace will grow its own bus package, and
        the moment that package restates the nominal chain the coupling this
        test guards comes back.
        """
        arm_constants = pytest.importorskip(  # noqa: PLC0415 - deliberate
            "UMArm_ROBOT_CONTROL.arm_constants",
            reason="UMArm_ROBOT_CONTROL (the RS485 bus stack) is not part of "
                   "this workspace; nothing here keeps a second copy of the "
                   "nominal chain")
        assert tuple(arm_constants.PLATE_CHAIN_NOMINAL_M) == rp.PLATE_CHAIN_NOMINAL_M

    def test_default_table_is_read_only(self):
        """The table is a default argument everywhere; a fit that edited it in
        place would silently re-zero every other caller."""
        with pytest.raises(ValueError):
            rp.DEFAULT_PARAMS[0, rp.COL_LL] = 1.0

    def test_as_params_validation(self):
        assert rp.as_params(None) is rp.DEFAULT_PARAMS
        with pytest.raises(ValueError):
            rp.as_params(np.zeros((3, 9)))
        with pytest.raises(ValueError):
            rp.as_params(np.zeros(10))


# --------------------------------------------------------------------------
# 2. twist_exp: one Rodrigues factor
# --------------------------------------------------------------------------


class TestTwistExp:
    def test_zero_angle_is_exact_identity(self):
        """sin(0) and 1-cos(0) are exactly 0.0, so this holds bitwise — which
        is what makes the q=0 structure facts below exact rather than close."""
        for xi in segment_twists(rp.DEFAULT_PARAMS[0]):
            assert np.array_equal(twist_exp(xi, 0.0), np.eye(4))

    def test_results_are_se3_with_the_right_rotation(self):
        """Each factor rotates by exactly theta about its own axis: the axis is
        an eigenvector, and trace(R) = 1 + 2 cos(theta)."""
        rng = np.random.default_rng(3)
        for row in rp.DEFAULT_PARAMS:
            for xi in segment_twists(row):
                for _ in range(20):
                    theta = rng.uniform(-1.5, 1.5)
                    g = twist_exp(xi, theta)
                    assert_se3(g)
                    w = xi[3:6]
                    assert np.allclose(g[0:3, 0:3] @ w, w, atol=1e-14)
                    assert abs(np.trace(g[0:3, 0:3]) - (1.0 + 2.0 * np.cos(theta))) < 1e-13

    def test_axis_points_stay_fixed(self):
        """The u-joint centres are on the axes, so they must not move: the
        proximal centre under xi1/xi2, the distal centre under xi3/xi4.  This
        is the local form of the u_dist-invariance fact the design leans on."""
        row = rp.DEFAULT_PARAMS[0]
        xi1, xi2, xi3, xi4 = segment_twists(row)
        uc1 = row[rp.COL_UC1]
        l_dist = row[rp.COL_AA1] + row[rp.COL_AA2] + row[rp.COL_LL] + uc1
        prox = np.array([0.0, 0.0, -uc1, 1.0])
        dist = np.array([0.0, 0.0, -l_dist, 1.0])
        for theta in (-1.2, -0.3, 0.4, 1.0):
            assert np.allclose(twist_exp(xi1, theta) @ prox, prox, atol=1e-15)
            assert np.allclose(twist_exp(xi2, theta) @ prox, prox, atol=1e-15)
            assert np.allclose(twist_exp(xi3, theta) @ dist, dist, atol=1e-15)
            assert np.allclose(twist_exp(xi4, theta) @ dist, dist, atol=1e-15)

    def test_axis_directions_are_the_design_ones(self):
        """xi1 = +x, xi2 = +y, xi3 = (x+y)/sqrt2, xi4 = (-x+y)/sqrt2 — the 45 deg
        bracket lives *here* and nowhere else (design §1: no Rz(45) between
        segments)."""
        s = 2.0 ** (-0.5)
        xi = segment_twists(rp.DEFAULT_PARAMS[1])
        assert np.array_equal(xi[0][3:6], [1.0, 0.0, 0.0])
        assert np.array_equal(xi[1][3:6], [0.0, 1.0, 0.0])
        assert np.array_equal(xi[2][3:6], [s, s, 0.0])
        assert np.array_equal(xi[3][3:6], [-s, s, 0.0])

    def test_bad_input_raises(self):
        with pytest.raises(ValueError):
            twist_exp(np.zeros(5), 0.1)
        with pytest.raises(ValueError):
            twist_exp([0.0, 0.0, 0.0, 2.0, 0.0, 0.0], 0.1)  # non-unit axis


# --------------------------------------------------------------------------
# 3. segment_transform: the explicit 4-product
# --------------------------------------------------------------------------


class TestSegmentTransform:
    def test_is_the_ordered_four_product(self):
        """Factor order is joint order (design D1) — and the order *matters*:
        the reversed product differs by ~0.2, so this is not vacuous."""
        row = rp.DEFAULT_PARAMS[0]
        q4 = np.array([0.3, -0.4, 0.5, 0.2])
        xi1, xi2, xi3, xi4 = segment_twists(row)
        longhand = (twist_exp(xi1, q4[0]) @ twist_exp(xi2, q4[1])
                    @ twist_exp(xi3, q4[2]) @ twist_exp(xi4, q4[3]))
        assert np.array_equal(segment_transform(row, q4), longhand)
        reversed_ = (twist_exp(xi4, q4[3]) @ twist_exp(xi3, q4[2])
                     @ twist_exp(xi2, q4[1]) @ twist_exp(xi1, q4[0]))
        assert np.max(np.abs(longhand - reversed_)) > 1e-3

    def test_t1_only_is_a_pure_x_rotation(self):
        """With UC1 = 0 the proximal axes pass through the origin, so a lone t1
        is Rx(t1) with zero translation — a hand-checkable anchor case."""
        t = 0.37
        g = segment_transform(rp.DEFAULT_PARAMS[0], [t, 0.0, 0.0, 0.0])
        c, s = np.cos(t), np.sin(t)
        rx = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, c, -s, 0.0],
                       [0.0, s, c, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
        assert np.allclose(g, rx, atol=1e-15)

    def test_u_dist_invariant_to_t3_t4(self):
        """Both distal axes pass through the distal centre, so t3/t4 cannot
        move it (design §1, recon-verified at 1.1e-16).  This is why q[10:12]
        moves no mocap centre and must be read from plate 5's orientation."""
        rng = np.random.default_rng(11)
        for row in rp.DEFAULT_PARAMS:
            l_dist = (row[rp.COL_AA1] + row[rp.COL_AA2] + row[rp.COL_LL]
                      + row[rp.COL_UC1])
            p = np.array([0.0, 0.0, -l_dist, 1.0])
            for _ in range(50):
                t1, t2 = rng.uniform(-1.0, 1.0, 2)
                ref = segment_transform(row, [t1, t2, 0.0, 0.0]) @ p
                for _ in range(5):
                    t3, t4 = rng.uniform(-1.0, 1.0, 2)
                    got = segment_transform(row, [t1, t2, t3, t4]) @ p
                    assert np.max(np.abs(got - ref)) < 1e-14

    def test_bad_q4_raises(self):
        with pytest.raises(ValueError):
            segment_transform(rp.DEFAULT_PARAMS[0], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------
# 4. The chain: fkine / centres / plates / predict_spatial structure
# --------------------------------------------------------------------------


class TestChainStructure:
    def test_q0_is_a_pure_z_drop(self):
        """fkine(0) = Tz(-0.692772): identity rotation (exactly — every factor
        is the identity at q=0) and the design's headline translation."""
        t = fkine(np.zeros(12))
        assert np.array_equal(t[0:3, 0:3], np.eye(3))
        assert np.allclose(t[0:3, 3], [0.0, 0.0, -0.692772], atol=1e-15)

    def test_q0_centre_heights_match_the_structure_doc(self):
        c = ujoint_centres(np.zeros(12))
        assert c.shape == (6, 3)
        assert np.array_equal(c[:, 0], np.zeros(6))     # exactly on the z axis
        assert np.array_equal(c[:, 1], np.zeros(6))
        assert np.allclose(c[:, 2], Q0_CENTRE_Z, atol=1e-15)

    def test_q0_consecutive_gaps_are_the_chain_nominals(self):
        """The five |u_{k+1} - u_k| distances at rest ARE PLATE_CHAIN_NOMINAL_M
        — the invariant the campaign's chain gate measures on live mocap."""
        c = ujoint_centres(np.zeros(12))
        gaps = np.linalg.norm(np.diff(c, axis=0), axis=1)
        assert np.allclose(gaps, rp.PLATE_CHAIN_NOMINAL_M, atol=1e-15)

    def test_gaps_are_q_invariant(self):
        """Rigid bodies: consecutive centre distances cannot depend on q.  (The
        one non-rigid pair, u2-u3 and u4-u5, spans the JD spacer — also rigid.)"""
        rng = np.random.default_rng(7)
        for _ in range(50):
            c = ujoint_centres(rng.uniform(-0.6, 0.6, 12))
            gaps = np.linalg.norm(np.diff(c, axis=0), axis=1)
            assert np.allclose(gaps, rp.PLATE_CHAIN_NOMINAL_M, atol=1e-13)

    def test_fkine_is_the_plate5_pose(self):
        """gst0 is a pure translation, so T_3 and the distal body share their
        rotation, and T_3's origin is the sixth centre: fkine == plates[5]."""
        rng = np.random.default_rng(13)
        for _ in range(25):
            q = rng.uniform(-0.6, 0.6, 12)
            assert np.allclose(fkine(q), plate_transforms(q)[5], atol=1e-14)
            assert np.allclose(fkine(q)[0:3, 3], ujoint_centres(q)[5], atol=1e-14)

    def test_predict_spatial_is_base_times_centres(self):
        rng = np.random.default_rng(17)
        for _ in range(10):
            q = rng.uniform(-0.5, 0.5, 12)
            base = random_se3(rng)
            got = predict_spatial(base, q)
            assert got.shape == (6, 3)
            centres = ujoint_centres(q)
            for k in range(6):
                want = base @ np.array([*centres[k], 1.0])
                assert np.allclose(got[k], want[0:3], atol=1e-13)

    def test_predict_spatial_identity_base_is_robot_frame(self):
        q = np.zeros(12)
        assert np.allclose(predict_spatial(np.eye(4), q), ujoint_centres(q), atol=0)

    def test_predict_spatial_ee_lever(self):
        """The optional 7th row: a lever arm in the last-plate frame, through
        fkine — at rest a pure -z lever lands straight below the sixth centre."""
        got = predict_spatial(np.eye(4), np.zeros(12), ee_lever_m=[0.0, 0.0, -0.14])
        assert got.shape == (7, 3)
        assert np.allclose(got[6], [0.0, 0.0, -0.692772 - 0.14], atol=1e-15)

    def test_wrong_shapes_raise(self):
        with pytest.raises(ValueError):
            fkine(np.zeros(11))
        with pytest.raises(ValueError):
            fkine(np.zeros((12, 1)))
        with pytest.raises(ValueError):
            ujoint_centres(np.zeros(13))
        with pytest.raises(ValueError):
            plate_transforms([0.0] * 10)
        with pytest.raises(ValueError):
            predict_spatial(np.eye(3), np.zeros(12))
        with pytest.raises(ValueError):
            predict_spatial(np.eye(4), np.zeros(12), ee_lever_m=[0.0, 0.0])
        with pytest.raises(ValueError):
            fkine(np.zeros(12), params=np.zeros((2, 10)))


# --------------------------------------------------------------------------
# 5. The mounting model (design D6 / review finding geo-1)
# --------------------------------------------------------------------------


class TestMountingModel:
    def test_all_plate_poses_are_se3(self):
        rng = np.random.default_rng(19)
        for _ in range(10):
            for t in plate_transforms(rng.uniform(-0.6, 0.6, 12)):
                assert_se3(t)

    def test_plate0_is_identity_for_all_q(self):
        """JD1 = UC1_1 = 0: the base plate is the robot frame itself, which is
        why streamed body 500 is a degenerate prediction target (design §1)."""
        rng = np.random.default_rng(23)
        for _ in range(10):
            assert np.array_equal(plate_transforms(rng.uniform(-0.6, 0.6, 12))[0],
                                  np.eye(4))

    @pytest.mark.parametrize("plate,joint_slice", [(0, slice(0, 2)),
                                                   (2, slice(4, 6)),
                                                   (4, slice(8, 10))])
    def test_proximal_plates_ignore_their_own_joint(self, plate, joint_slice):
        """Plates 0/2/4 sit on the *proximal* body of their u-joint: the joint
        rotates away from them, so their pose is bitwise independent of its two
        angles.  A wrong mounting side fails this by whole degrees."""
        rng = np.random.default_rng(29)
        for _ in range(10):
            q = rng.uniform(-0.5, 0.5, 12)
            ref = plate_transforms(q)[plate]
            q2 = q.copy()
            q2[joint_slice] = rng.uniform(-0.5, 0.5, 2)
            assert np.array_equal(plate_transforms(q2)[plate], ref)

    def test_plate5_rotation_carries_q10_q11(self):
        """The u6 signal: q[10:12] moves *no centre* — only plate 5's
        orientation, which is exactly where mk8 reads it
        (``kinematics_mp.py:889-899``)."""
        q = np.zeros(12)
        rest = plate_transforms(q)
        q[10], q[11] = 0.3, -0.2
        moved = plate_transforms(q)
        # Orientation moved by the joint angle scale...
        dr = rest[5][0:3, 0:3].T @ moved[5][0:3, 0:3]
        angle = np.arccos(np.clip((np.trace(dr) - 1.0) / 2.0, -1.0, 1.0))
        assert angle > 0.3
        # ...while every centre stayed put (u_dist invariance, chain-wide).
        assert np.allclose(ujoint_centres(q), ujoint_centres(np.zeros(12)), atol=1e-14)

    @pytest.mark.parametrize("plate,tslice", [(1, slice(2, 4)),
                                              (3, slice(6, 8)),
                                              (5, slice(10, 12))])
    def test_distal_plates_rotate_with_their_joint(self, plate, tslice):
        q = np.zeros(12)
        ref = plate_transforms(q)[plate]
        q[tslice] = [0.25, -0.15]
        got = plate_transforms(q)[plate]
        dr = ref[0:3, 0:3].T @ got[0:3, 0:3]
        assert np.arccos(np.clip((np.trace(dr) - 1.0) / 2.0, -1.0, 1.0)) > 0.2

    def test_co_rigid_pairs_have_identity_relative_rotation(self):
        """Plates {1,2} and {3,4} share a rigid connector; in this model their
        body rotations are *the same matrix* (the +45 deg between bracket
        families lives in the marker bracket, not the body frame — marker-frame
        design §2), and their origins differ by exactly Tz(-JD) in the shared
        body frame.  The synthetic-marker generator and the benchmark's
        co-rigid self-check both stand on this."""
        rng = np.random.default_rng(31)
        jd = rp.segment_jds()
        for _ in range(25):
            plates = plate_transforms(rng.uniform(-0.6, 0.6, 12))
            for a, b, gap in ((1, 2, jd[1]), (3, 4, jd[2])):
                ra = plates[a][0:3, 0:3]
                rb = plates[b][0:3, 0:3]
                assert np.allclose(ra.T @ rb, np.eye(3), atol=1e-13)
                d_body = ra.T @ (plates[b][0:3, 3] - plates[a][0:3, 3])
                assert np.allclose(d_body, [0.0, 0.0, -gap], atol=1e-13)

    def test_plate_origins_are_the_ujoint_centres(self):
        rng = np.random.default_rng(37)
        for _ in range(10):
            q = rng.uniform(-0.6, 0.6, 12)
            assert np.array_equal(plate_transforms(q)[:, 0:3, 3], ujoint_centres(q))


# --------------------------------------------------------------------------
# 6. Frozen goldens  (blessed against fkine_mk5 once — agreement 2.2e-16)
# --------------------------------------------------------------------------

#: A literal q with every joint excited, no symmetry, all four twist types.
GOLDEN_Q = np.array([0.10, -0.20, 0.30, -0.10,
                     0.20, 0.10, -0.30, 0.20,
                     -0.10, 0.30, 0.15, -0.25])

#: ujoint_centres(GOLDEN_Q), robot frame, metres.
GOLDEN_CENTRES = np.array([
    [0.0, 0.0, 0.0],
    [0.043482359092453474, 0.021414788178448017, -0.21343357917400704],
    [0.0460797329659603, 0.03926358578573946, -0.25777663628484654],
    [0.03617602813324756, 0.14569437452352313, -0.4203196011871756],
    [0.03747298496855411, 0.15646434053072356, -0.46644312727930237],
    [-0.012031483942272249, 0.17920366589598447, -0.6423098124723481],
])

#: fkine(GOLDEN_Q) — the plate-5 pose.  Matches the legacy
#: ``fkine_mk5(params, GOLDEN_Q)`` to 2.2e-16 (re-checked in TestOracleFkine
#: wherever the tree is reachable, so the literal cannot quietly rot).
GOLDEN_FKINE = np.array([
    [0.9787646082224917, 0.0807554508404127, 0.18841018775701532, -0.012031483942272249],
    [0.0004720857919882149, 0.9182400522222433, -0.39602396850430266, 0.17920366589598447],
    [-0.20498687476537877, 0.3877031901525362, 0.8987027414665371, -0.6423098124723481],
    [0.0, 0.0, 0.0, 1.0],
])

#: Frozen-golden budget.  The literals were produced by this implementation on
#: this bench (and cross-blessed against the oracle at 2.2e-16); 1e-13 leaves
#: room for BLAS/matmul reassociation across numpy builds while still pinning
#: every sign, factor order and constant to ~12 significant digits.
GOLDEN_TOL = 1e-13


class TestGoldens:
    def test_golden_centres(self):
        assert np.max(np.abs(ujoint_centres(GOLDEN_Q) - GOLDEN_CENTRES)) < GOLDEN_TOL

    def test_golden_fkine(self):
        assert np.max(np.abs(fkine(GOLDEN_Q) - GOLDEN_FKINE)) < GOLDEN_TOL

    def test_golden_last_plate_consistency(self):
        """The two goldens agree with each other through the plate model."""
        plates = plate_transforms(GOLDEN_Q)
        assert np.max(np.abs(plates[5] - GOLDEN_FKINE)) < GOLDEN_TOL
        assert np.max(np.abs(plates[:, 0:3, 3] - GOLDEN_CENTRES)) < GOLDEN_TOL


# --------------------------------------------------------------------------
# 7. Round trip through the shipped mocap_to_q
# --------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("joint", range(12))
    @pytest.mark.parametrize("amp", [0.3, -0.3, 0.05])
    def test_single_joint_machine_precision(self, joint, amp):
        """All 12 joints, including q10/q11 — the two that exist only in
        plate 5's orientation.  Recovery at machine precision is the design's
        single-joint claim (§1), and it is what the hardware campaign's
        per-joint verdicts stand on."""
        q = np.zeros(12)
        q[joint] = amp
        q_rec = mocap_to_q(build_homos(q))
        assert q_rec is not None
        assert np.max(np.abs(q_rec - q)) < TRIP_TOL

    @pytest.mark.parametrize("joint", range(12))
    def test_single_joint_invariant_to_base_se3(self, joint):
        """``predict_spatial``'s correspondence in reverse: pre-multiplying
        every plate by one SE(3) is 'the same arm, rig moved', and q must not
        care (mocap_to_q de-rotates by body 500)."""
        rng = np.random.default_rng(1000 + joint)
        q = np.zeros(12)
        q[joint] = 0.25
        q_rec = mocap_to_q(build_homos(q, base=random_se3(rng)))
        assert q_rec is not None
        assert np.max(np.abs(q_rec - q)) < TRIP_TOL

    def test_q_zero_round_trips(self):
        q_rec = mocap_to_q(build_homos(np.zeros(12)))
        assert q_rec is not None
        assert np.max(np.abs(q_rec)) < TRIP_TOL

    def test_multi_joint_swing_only_error_is_the_documented_one(self):
        """mk8 loses link-axis twist on multi-joint frames — a property of the
        swing-only extraction, not a porting bug (design §1: <= 3.7e-4 rad at
        |q| <= 0.05; measured 2.3e-4 for this seed).  Both bounds matter: the
        upper pins the budget the campaign tolerances were built from, the
        lower proves this test still exercises the lossy path — machine
        precision here would mean the homos stopped carrying multi-joint
        coupling, i.e. the test broke, not the code improved."""
        rng = np.random.default_rng(20260811)
        worst = 0.0
        for _ in range(100):
            q = rng.uniform(-0.05, 0.05, 12)
            q_rec = mocap_to_q(build_homos(q))
            assert q_rec is not None
            worst = max(worst, float(np.max(np.abs(q_rec - q))))
        assert worst < 5e-4, f"multi-joint error {worst:g} exceeds the documented budget"
        assert worst > 1e-9, "multi-joint became exact — the lossy path is no longer exercised"

    def test_homos_shape_satisfies_mocap_to_q(self):
        """mocap_to_q wants >= 7 rows (row 6 = EE plate, present but unread);
        pin that build_homos delivers them, so a future edit cannot silently
        hand it a 6-row array (which raises) or non-finite filler."""
        homos = build_homos(GOLDEN_Q)
        assert homos.shape == (7, 4, 4)
        assert np.isfinite(homos).all()


# --------------------------------------------------------------------------
# 8. Oracle equivalence — numpy-only part (exponentials + params)
# --------------------------------------------------------------------------


@needs_oracle
class TestOracleExponentials:
    def test_params_table_matches_legacy(self):
        legacy = np.array([_legacy_rc.param_link0, _legacy_rc.param_link1,
                           _legacy_rc.param_link2], dtype=float)
        assert np.array_equal(rp.DEFAULT_PARAMS, legacy)

    def test_500_random_q_against_get_e1234(self):
        """The headline equivalence from the design spec (§3): the readable
        four-factor product against the Mathematica closed form, all three
        segments, |q| up to 1 rad (past the arm's real travel)."""
        rng = np.random.default_rng(20260811)
        worst = 0.0
        worst_at = None
        for i in range(500):
            seg = int(rng.integers(0, 3))
            q4 = rng.uniform(-1.0, 1.0, 4)
            mine = segment_transform(rp.DEFAULT_PARAMS[seg], q4)
            theirs = _legacy_exp.get_e1234(rp.DEFAULT_PARAMS[seg], q4)
            d = float(np.max(np.abs(mine - theirs)))
            if d > worst:
                worst, worst_at = d, (i, seg)
        assert worst < ORACLE_TOL, f"max |dT| = {worst:g} at draw/segment {worst_at}"

    def test_single_axis_cases_against_get_e1234(self):
        """One joint at a time, each segment — the cases whose failure names
        the broken twist rather than 'something in the product'."""
        for seg in range(3):
            for j in range(4):
                for amp in (-0.7, 0.3):
                    q4 = np.zeros(4)
                    q4[j] = amp
                    mine = segment_transform(rp.DEFAULT_PARAMS[seg], q4)
                    theirs = _legacy_exp.get_e1234(rp.DEFAULT_PARAMS[seg], q4)
                    assert np.max(np.abs(mine - theirs)) < ORACLE_TOL, \
                        f"segment {seg} twist {j + 1} amp {amp}"


# --------------------------------------------------------------------------
# 9. Oracle equivalence — fkine_mk5 (legacy tree AND scipy)
# --------------------------------------------------------------------------


@needs_fkine_oracle
class TestOracleFkine:
    def test_500_random_q_against_fkine_mk5(self):
        rng = np.random.default_rng(8112026)
        worst = 0.0
        for _ in range(500):
            q = rng.uniform(-1.0, 1.0, 12)
            theirs = _legacy_km.fkine_mk5(rp.DEFAULT_PARAMS, q)
            d = float(np.max(np.abs(fkine(q) - theirs)))
            worst = max(worst, d)
        assert worst < ORACLE_TOL, f"max |dT| = {worst:g}"

    def test_golden_still_matches_the_oracle(self):
        """The frozen literals are only as good as their provenance — where the
        legacy tree is reachable, confirm GOLDEN_FKINE is still what it
        computes, so the golden cannot rot into a self-consistent fiction."""
        theirs = _legacy_km.fkine_mk5(rp.DEFAULT_PARAMS, GOLDEN_Q)
        assert np.max(np.abs(GOLDEN_FKINE - theirs)) < GOLDEN_TOL

    def test_q0_matches_the_oracle(self):
        theirs = _legacy_km.fkine_mk5(rp.DEFAULT_PARAMS, np.zeros(12))
        assert np.max(np.abs(fkine(np.zeros(12)) - theirs)) < ORACLE_TOL
