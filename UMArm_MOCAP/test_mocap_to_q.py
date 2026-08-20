"""Tests for the mocap -> ``q`` conversion.  Pure math: no sockets, no serial.

Four independent things are checked, because each catches what the others
cannot:

1. **Round trip against ground truth** (:class:`TestRoundTrip`).  Plate poses are
   *forward-constructed* from a known ``q`` — link vectors built from the angle
   convention, chained through the same align/45 deg steps, then turned into
   U-joint centres — and the reader must recover that ``q``.

   Read what this does and does not prove.  :func:`poses_from_q` inverts the
   reader using *the same* ``mc.RZn45``, the same :func:`rotation_aligning`, the
   same ``k % 2`` parity and the same index constants, so a shared convention
   error cancels exactly and the round trip still passes.  Verified: flipping
   ``RZn45`` to +45 deg fails 0 of the 75 round-trip cases (and 0 of all 92
   oracle-free tests as this file stood before) while moving the oracle
   comparison by 3.1157 rad.  The round trip pins *self-consistency and
   invariance* — that ``q`` does not depend on base pose or link length, and that
   the reader has no internal contradiction.  It is not a convention check.

2. **Conventions pinned to frozen numbers** (:class:`TestConventionsPinned`).
   This is what actually catches a shared convention error, and it needs no
   oracle: the ``RZn45``/``V_BASE`` values and the golden ``(homos -> q)`` pairs
   are hard-coded decimal literals, blessed against the legacy implementation
   once and frozen here.  It is the reason a fresh clone with no access to the
   legacy tree still reports a *red* suite if a constant or a parity drifts.

3. **Equivalence with the legacy implementation** (:class:`TestOracle`).  The
   live stack's numbers come from ``UMArm_compliance_TRO`` today; a re-write is
   only safe if it is numerically indistinguishable from what the lab has been
   calibrating against.  The legacy module is imported by ``sys.path`` insert
   **for the test only** — nothing at runtime reaches into that repo.  These
   tests *skip* when that tree is unreachable, which is the normal case on a
   deployment machine — hence item 2 carrying the conventions.

4. **The scipy replacements in isolation** (:class:`TestRotationAligning`,
   :class:`TestAgainstScipy`), since :func:`mocap_to_q.rotation_aligning` is the
   one piece of new math and its near-antiparallel branch is barely reachable
   through the pose-level tests.  The scipy comparisons are gated on *scipy*,
   not on the legacy tree, because scipy is a normal dependency of this repo's
   test environment while the OneDrive research folder is not.

Run:  ``python -m pytest UMArm_MOCAP/test_mocap_to_q.py -v``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# The module under test, imported flat so this file works whichever way pytest
# rooted itself.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mocap_constants as mc  # noqa: E402
from mocap_to_q import (  # noqa: E402
    link_vectors,
    mocap_to_q,
    quat_xyzw_to_matrix,
    rotation_aligning,
    ujoint_angles,
    unit_from_ujoint_angles,
)

# --------------------------------------------------------------------------
# The oracle: the legacy implementation, TEST ONLY
# --------------------------------------------------------------------------
# Read-only reference repo.  It is imported here and nowhere else; if this path
# is wrong or the folder is gone, the oracle tests skip and the rest still run,
# so the suite stays useful on a machine that only has this repo checked out.

LEGACY_MOCAP_DIR = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Umich", "PHD_Courses", "Research", "project2023_variable_stiffness",
    "UMArm_compliance_TRO", "mocap_to_config",
)

_oracle = None
_oracle_why = ""
if os.path.isdir(LEGACY_MOCAP_DIR):
    if LEGACY_MOCAP_DIR not in sys.path:
        sys.path.insert(0, LEGACY_MOCAP_DIR)
    try:
        from mocap_to_config import mocap_to_config as _oracle  # noqa: E402
        import mocap_config_constants as _legacy_mc  # noqa: E402
    except Exception as exc:  # scipy missing, syntax drift, ...
        _oracle_why = f"legacy import failed: {exc}"
else:
    _oracle_why = f"legacy repo not found at {LEGACY_MOCAP_DIR}"

needs_oracle = pytest.mark.skipif(_oracle is None, reason=_oracle_why)

# scipy is a dependency of this repo's *test* environment, not of the runtime and
# not of the legacy tree.  The two scipy-replacement checks are gated on it alone,
# so they still run on a machine that has no access to the research folder — the
# quaternion component order in particular is something mocap_to_q calls "the one
# place where getting the order wrong yields a plausible-looking but silently
# wrong base frame", and it must not be checked only where the oracle happens to
# live.
try:
    from scipy.spatial.transform import Rotation as _ScipyRotation
except ImportError:  # pragma: no cover
    _ScipyRotation = None
needs_scipy = pytest.mark.skipif(_ScipyRotation is None, reason="scipy not installed")

#: Equivalence budget from the design spec.  The two implementations differ only
#: in rounding (a Rodrigues construction instead of a quaternion round trip, and
#: radians throughout instead of degrees-then-radians), so the real gap is ~1e-14
#: and this threshold is three orders of margin, not a fudge factor.
TOL = 1e-9


# --------------------------------------------------------------------------
# Forward construction: known q -> plate poses
# --------------------------------------------------------------------------


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrix, numpy only (QR of a Gaussian matrix)."""
    a, r = np.linalg.qr(rng.normal(size=(3, 3)))
    a = a * np.sign(np.diag(r))          # fix QR's sign ambiguity -> uniform
    if np.linalg.det(a) < 0:
        a[:, 0] = -a[:, 0]
    return a


def frame_with_z(z: np.ndarray) -> np.ndarray:
    """Some rotation matrix whose third column is the unit vector ``z``.

    Only that column is read (it is plate 5's body z = the last link direction),
    but a whole valid rotation is built so the pose array is a legal SE(3).
    """
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def poses_from_q(q, rot_base=None, base_pos=None, lengths=None) -> np.ndarray:
    """Build a ``(9, 4, 4)`` pose array that a perfect reader turns back into ``q``.

    Inverts the reader step by step.  For joint ``k`` the reader computes
    ``v_k = [RZn45] @ align(rv_{k-1} -> +z) @ rv_k`` and reads the angles off
    ``v_k``; so given the wanted angles we take ``v_k =
    unit_from_ujoint_angles(...)`` and undo the two rotations (transpose, both
    being rotations) to get the link direction ``rv_k`` in the base frame.  The
    U-joint centres then follow by walking the chain: ``pos[k+1] = pos[k] -
    R_base @ (L_k * rv_k)``, the minus sign being the reader's proximal-minus-
    distal convention.

    A random base rotation/position and unequal link lengths are supported on
    purpose: ``q`` must be invariant to where the mocap volume's origin is and
    to how long the links are.

    **This helper cannot pin a convention.**  It undoes the reader's steps with
    the reader's own ``mc.RZn45``, its own :func:`rotation_aligning` and its own
    ``k % 2 == 1`` parity, so any error shared with the reader cancels exactly and
    the round trip still succeeds.  What it proves is invariance and internal
    consistency.  The conventions themselves are pinned by the frozen literals in
    :class:`TestConventionsPinned`; keep the two jobs separate rather than
    trusting this function with both.
    """
    q = np.asarray(q, dtype=float)
    assert q.shape == (mc.NUM_JOINTS,)
    rot_base = np.eye(3) if rot_base is None else np.asarray(rot_base, dtype=float)
    base_pos = np.zeros(3) if base_pos is None else np.asarray(base_pos, dtype=float)
    lengths = np.full(6, 0.15) if lengths is None else np.asarray(lengths, dtype=float)

    rv = np.empty((6, 3))
    for k in range(6):
        v = unit_from_ujoint_angles(q[2 * k], q[2 * k + 1])
        if k == 0:
            rv[0] = v
            continue
        if k % 2 == 1:                                  # u2, u4, u6 carry RZn45
            v = mc.RZn45.T @ v
        rv[k] = rotation_aligning(rv[k - 1], mc.V_BASE).T @ v

    homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
    homos[mc.IDX_BASE, 0:3, 0:3] = rot_base
    pos = base_pos.copy()
    homos[0, 0:3, 3] = pos
    for k in range(5):                                  # centres 1..5
        pos = pos - rot_base @ (lengths[k] * rv[k])
        homos[k + 1, 0:3, 3] = pos
    # Plate 5's orientation encodes the sixth link direction; the EE plate is
    # placed along it so the array is physically sensible (mk8 ignores its
    # position).
    last = rot_base @ rv[5]
    homos[mc.IDX_U3_DISTAL, 0:3, 0:3] = frame_with_z(last)
    homos[mc.IDX_END_EFFECTOR, 0:3, 3] = pos - lengths[5] * last
    return homos


def random_pose_set(rng: np.random.Generator) -> np.ndarray:
    """A wholly random ``(9, 4, 4)`` pose array — geometry with no arm in it.

    Deliberately unphysical: random orientations and positions drive the angle
    formulas over their full range (including ``|theta| > pi/2`` and the
    quadrant boundaries) instead of only the tiny neighbourhood of straight that
    the real arm visits.  Equivalence has to hold there too, or a future
    refactor could break the interesting cases unnoticed.  Matches the sampling
    the legacy README used for its own verification.
    """
    homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
    for i in range(mc.NUM_RIGID_BODIES):
        homos[i, 0:3, 0:3] = random_rotation(rng)
        homos[i, 0:3, 3] = rng.uniform(-0.5, 0.5, 3)
    return homos


def structured_q_cases() -> list[tuple[str, np.ndarray]]:
    """Named ``q`` vectors: zeros, single-joint excursions, then random combos."""
    cases: list[tuple[str, np.ndarray]] = [("zeros", np.zeros(mc.NUM_JOINTS))]
    for j in range(mc.NUM_JOINTS):
        for sign in (+1.0, -1.0):
            q = np.zeros(mc.NUM_JOINTS)
            q[j] = sign * 0.4
            cases.append((f"joint{j:02d}{'+' if sign > 0 else '-'}0.4", q))
    rng = np.random.default_rng(4242)
    for i in range(40):
        cases.append((f"combo{i:02d}", rng.uniform(-0.4, 0.4, mc.NUM_JOINTS)))
    # Big excursions: past the arm's real travel, still inside the branch cuts.
    for i in range(10):
        cases.append((f"wide{i:02d}", rng.uniform(-1.0, 1.0, mc.NUM_JOINTS)))
    return cases


#: Built once, so the parametrize ids and values cannot drift apart.
STRUCTURED_CASES = structured_q_cases()
STRUCTURED_IDS = [name for name, _ in STRUCTURED_CASES]


# --------------------------------------------------------------------------
# Frozen golden data: the conventions, checkable with no oracle present
# --------------------------------------------------------------------------
# Every number below is a literal, produced once against the legacy
# implementation (agreement 2.2e-16) and frozen.  Nothing here calls
# `poses_from_q`, `rotation_aligning` or `mc.RZn45`, so these values pin the
# convention chain end to end: the proximal-minus-distal link sign, V_BASE, the
# 45 deg offset's sign, which joints carry it, the atan2 formulas, and the base
# de-rotation.  If one of those changes, this fails on a machine that has never
# heard of the legacy repo.

#: Link vectors, expressed in the base frame, as exact sixteenths.  Small
#: integers so the input is unambiguous and reproducible bit for bit.
GOLDEN_LINK_STEPS = np.array([
    [1.0, 2.0, 8.0],
    [-2.0, 1.0, 7.0],
    [3.0, -1.0, 9.0],
    [0.0, 2.0, 6.0],
    [-1.0, -3.0, 8.0],
]) / 16.0

#: Plate 5's body z, i.e. the sixth link direction, in the base frame.
GOLDEN_LAST_Z = np.array([2.0, -3.0, 7.0])

#: A base orientation with no special structure, built from two Pythagorean
#: (3-4-5 and 7-24-25) rotations so the literals are exact and the matrix is
#: orthonormal to 1.1e-16.
GOLDEN_RZ = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
GOLDEN_RX = np.array([[1.0, 0.0, 0.0], [0.0, 0.28, -0.96], [0.0, 0.96, 0.28]])

#: The ``q`` the golden frame must produce, to the last bit the oracle gave.
GOLDEN_Q = np.array([
    -0.244978663126864, 0.12067855313100972,
    -0.2225253417762494, -0.3463766216924711,
    0.2759969371726707, 0.5911820499034528,
    -0.5255287373269761, 0.08559782799181304,
    0.6805212246672148, -0.11651106281837897,
    0.30621527361697326, 0.22192222186675847,
])


def golden_homos(rot_base=None, base_pos=None) -> np.ndarray:
    """The golden pose array, walked straight down the chain of link steps.

    Deliberately *not* built with :func:`poses_from_q`: the only convention this
    shares with the reader is "a link vector is the proximal centre minus the
    distal one", which is why a shared error in the align/45 deg steps cannot
    hide here.  ``rot_base`` rotates the whole rig, which must leave ``q``
    unchanged.
    """
    rot_base = np.eye(3) if rot_base is None else np.asarray(rot_base, dtype=float)
    base_pos = np.zeros(3) if base_pos is None else np.asarray(base_pos, dtype=float)
    homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
    homos[mc.IDX_BASE, 0:3, 0:3] = rot_base
    pos = base_pos.copy()
    homos[mc.IDX_BASE, 0:3, 3] = pos
    for k in range(5):
        pos = pos - rot_base @ GOLDEN_LINK_STEPS[k]
        homos[k + 1, 0:3, 3] = pos
    last = rot_base @ GOLDEN_LAST_Z
    homos[mc.IDX_U3_DISTAL, 0:3, 0:3] = frame_with_z(last)
    homos[mc.IDX_END_EFFECTOR, 0:3, 3] = pos - 0.1 * last / np.linalg.norm(last)
    return homos


#: A frame that drives :func:`rotation_aligning` into its exactly-antiparallel
#: branch twice (links 2 and 5 point at -z while their predecessors point at +z),
#: with successors that are *not* parallel to z — so the arbitrary axis pick
#: actually reaches ``q`` and can be pinned.  See
#: :meth:`TestConventionsPinned.test_antiparallel_axis_choice_is_pinned`.
ANTIPARALLEL_DIRECTIONS = [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
    np.array([1.0, 2.0, 1.0]) / np.sqrt(6.0),
    np.array([-1.0, 1.0, 2.0]) / np.sqrt(6.0),
    np.array([0.0, 0.0, -1.0]),
    np.array([2.0, -1.0, 1.0]) / np.sqrt(6.0),
]

#: ``q`` for :func:`antiparallel_homos`, frozen from the oracle (agreement
#: 1.1e-16).  Joints 4-5 and 10-11 are the ones the axis pick moves.
ANTIPARALLEL_Q = np.array([
    0.0, -0.0,
    -3.141592653589793, -0.0,
    -2.0344439357957027, -0.4205343352839652,
    -0.5513515966154474, -0.9434723594246988,
    -2.677945044588987, -0.4205343352839652,
    -2.5261129449194057, -1.0471975511965979,
])


def antiparallel_homos(length: float = 0.15) -> np.ndarray:
    """Pose array whose link directions are :data:`ANTIPARALLEL_DIRECTIONS`."""
    homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
    pos = np.zeros(3)
    homos[mc.IDX_BASE, 0:3, 3] = pos
    for k in range(5):
        pos = pos - length * ANTIPARALLEL_DIRECTIONS[k]
        homos[k + 1, 0:3, 3] = pos
    homos[mc.IDX_U3_DISTAL, 0:3, 0:3] = frame_with_z(ANTIPARALLEL_DIRECTIONS[5])
    homos[mc.IDX_END_EFFECTOR, 0:3, 3] = pos - length * ANTIPARALLEL_DIRECTIONS[5]
    return homos


# --------------------------------------------------------------------------
# 1. The angle convention itself
# --------------------------------------------------------------------------


class TestAngleConvention:
    def test_straight_is_zero(self):
        """A link continuing along +z reads (0, 0) — the definition of the zero."""
        t1, t2 = ujoint_angles(mc.V_BASE)
        assert abs(t1) < 1e-15 and abs(t2) < 1e-15

    def test_inverse_round_trips(self):
        rng = np.random.default_rng(7)
        for _ in range(2000):
            t1, t2 = rng.uniform(-1.4, 1.4, 2)
            v = unit_from_ujoint_angles(t1, t2)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-15
            r1, r2 = ujoint_angles(v)
            assert abs(r1 - t1) < 1e-14 and abs(r2 - t2) < 1e-14

    def test_scale_invariance(self):
        """Angles are read off a direction, so length must not matter."""
        v = np.array([0.3, -0.7, 0.6])
        for s in (1e-6, 0.5, 1.0, 1e3):
            assert np.allclose(ujoint_angles(s * v), ujoint_angles(v), atol=1e-15)

    def test_sign_facts_match_the_design_note(self):
        """+theta1 tips the distal end toward +y; +theta2 tips it toward -x.

        The link vector points proximal-ward (base minus distal), so the distal
        end moves along ``-v``.  Pinning this here is what makes the sign facts
        in ``docs/umarm_mocap_design.md`` checkable rather than folklore — and it
        is independent of the forward construction below, which is derived from
        the same inverse.
        """
        distal = -unit_from_ujoint_angles(0.3, 0.0)
        assert distal[1] > 0.25 and abs(distal[0]) < 1e-12   # toward +y
        distal = -unit_from_ujoint_angles(0.0, 0.3)
        assert distal[0] < -0.25 and abs(distal[1]) < 1e-12  # toward -x


# --------------------------------------------------------------------------
# 1b. The conventions, pinned to frozen numbers (NO ORACLE REQUIRED)
# --------------------------------------------------------------------------


class TestConventionsPinned:
    """The convention checks that survive on a machine without the legacy tree.

    Everything in :class:`TestOracle` skips there — 80 of this suite's tests —
    and the round trip is convention-blind by construction, so without this class
    a fresh clone with a sign-flipped ``RZn45`` reports a fully green suite.
    Verified: with ``RZn45`` flipped to +45 deg, every test here fails.
    """

    def test_v_base_is_plus_z(self):
        assert np.array_equal(mc.V_BASE, np.array([0.0, 0.0, 1.0]))

    def test_rzn45_is_minus_45_about_z(self):
        """The *sign* is the whole point: +45 deg is the plausible wrong answer.

        A -45 deg z-rotation has ``+sin(45)`` in the upper right, i.e.
        ``RZn45[0, 1] > 0``.  Spelled out as literals rather than as
        ``cos(-pi/4)`` so this cannot agree with a flipped constant by
        construction.
        """
        s = 0.7071067811865476                      # sin(pi/4) to double precision
        assert np.allclose(mc.RZn45, np.array([[s, s, 0.0],
                                               [-s, s, 0.0],
                                               [0.0, 0.0, 1.0]]), atol=1e-15)
        assert mc.RZn45[0, 1] > 0.0                 # -45 deg, not +45 deg
        assert np.allclose(mc.RZn45.T @ mc.RZn45, np.eye(3), atol=1e-15)
        assert np.linalg.det(mc.RZn45) > 0.0

    def test_index_map_and_sizes(self):
        """Lifted values; a silent renumbering would put a joint on a stray body."""
        assert mc.NUM_JOINTS == 12
        assert mc.NUM_RIGID_BODIES == 9
        assert mc.N_USED_RIGID_BODIES == 7
        assert mc.RIGID_BODY_ID_MASK == 500
        assert mc.KINOVA_MOCAP_STREAM_ID == 1008
        assert mc.KINOVA_RIGID_BODY_INDEX == 8
        assert (mc.IDX_BASE, mc.IDX_U1_DISTAL, mc.IDX_U2_PROXIMAL, mc.IDX_U2_DISTAL,
                mc.IDX_U3_PROXIMAL, mc.IDX_U3_DISTAL, mc.IDX_END_EFFECTOR) == \
               (0, 1, 2, 3, 4, 5, 6)
        assert mc.U_JOINT_INDICES == (0, 1, 2, 3, 4, 5)
        assert mc.MIN_LINK_NORM == 1e-10

    def test_golden_frame(self):
        """The headline: a frozen pose array must give a frozen ``q``."""
        q = mocap_to_q(golden_homos())
        assert q is not None
        assert np.max(np.abs(q - GOLDEN_Q)) < TOL

    def test_golden_frame_is_invariant_to_the_base_pose(self):
        """Same arm, rig moved and turned: identical ``q``, still the frozen one."""
        rot_base = GOLDEN_RZ @ GOLDEN_RX
        assert np.allclose(rot_base.T @ rot_base, np.eye(3), atol=1e-15)
        q = mocap_to_q(golden_homos(rot_base=rot_base,
                                    base_pos=np.array([0.3, -1.2, 2.0])))
        assert q is not None
        assert np.max(np.abs(q - GOLDEN_Q)) < TOL

    def test_golden_link_vectors_point_the_way_the_docs_say(self):
        """Proximal minus distal, de-rotated into the base frame, unit length."""
        rot_base = GOLDEN_RZ @ GOLDEN_RX
        rv = link_vectors(golden_homos(rot_base=rot_base))
        assert rv is not None
        assert np.allclose(np.linalg.norm(rv, axis=1), 1.0, atol=1e-15)
        for k in range(5):
            want = GOLDEN_LINK_STEPS[k] / np.linalg.norm(GOLDEN_LINK_STEPS[k])
            assert np.allclose(rv[k], want, atol=1e-14)
        assert np.allclose(rv[5], GOLDEN_LAST_Z / np.linalg.norm(GOLDEN_LAST_Z),
                           atol=1e-14)

    def test_rzn45_parity_is_the_odd_joints(self):
        """u2/u4/u6 carry the offset — ``k % 2 == 1``, not ``k % 2 == 0``.

        Applying the offset to the wrong half of the joints is exactly the sort of
        error the round trip cannot see, so pin it by rebuilding ``q`` here with
        the parity written out longhand and comparing to the frozen vector.
        """
        rv = link_vectors(golden_homos())
        expect = []
        for k in range(6):
            v = rv[0] if k == 0 else rotation_aligning(rv[k - 1], mc.V_BASE) @ rv[k]
            if k in (1, 3, 5):                      # u2, u4, u6 — longhand
                v = mc.RZn45 @ v
            expect.extend(ujoint_angles(v))
        assert np.max(np.abs(np.array(expect) - GOLDEN_Q)) < TOL

    def test_quaternion_component_order_is_pinned(self):
        """Scalar **last**, pinned without scipy and without the legacy tree.

        :func:`quat_xyzw_to_matrix` calls this "the one place where getting the
        order wrong yields a plausible-looking but silently wrong base frame", and
        a scalar-first misread of this quaternion moves the matrix by 1.60 — a
        rotation that is still perfectly orthonormal, so nothing but a frozen
        value catches it.  Every component differs in magnitude, so no
        permutation of them can agree by accident.
        """
        quat = np.array([0.2004312214748382, -0.4008624429496764,
                         0.5010780536870955, 0.7403089972462])
        expected = np.array([
            [0.17646017189113938, -0.902595881101925, -0.3926607736379958],
            [0.58121448476676, 0.4174962191425132, -0.6984888185926934],
            [0.7943875190569522, -0.10496467224521938, 0.5982732545810436],
        ])
        got = quat_xyzw_to_matrix(quat)
        assert np.max(np.abs(got - expected)) < 1e-12
        assert np.allclose(got.T @ got, np.eye(3), atol=1e-14)
        # A scalar-first misread would also be a valid rotation matrix — which is
        # exactly why it needs pinning rather than an invariant check.
        misread = quat_xyzw_to_matrix(np.roll(quat, 1))
        assert np.max(np.abs(misread - expected)) > 1.0

    def test_never_returns_a_non_finite_q(self):
        """Finite input can still overflow, so the *output* is what gets gated.

        Positions of +-1e308 are finite, their difference overflows to inf, the
        norm is inf and ``inf >= MIN_LINK_NORM`` is True — so neither the input
        check nor the length check sees anything wrong, and only the finiteness
        check on the result keeps a NaN direction from reaching the caller.
        """
        homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
        for i in range(6):
            homos[i, 0:3, 3] = [1e308 if i % 2 == 0 else -1e308, 0.0, 0.0]
        assert np.isfinite(homos[0:6, 0:3, 3]).all()      # the input really is finite
        with np.errstate(over="ignore", invalid="ignore"):
            assert link_vectors(homos) is None
            assert mocap_to_q(homos) is None

    def test_antiparallel_axis_choice_is_pinned(self):
        """The arbitrary 180 deg axis reaches ``q``, and which one is frozen.

        ``_orthogonal_axis``' pick is mathematically arbitrary but it is *not*
        unobservable: with a predecessor at -z and a successor that is not
        parallel to z, a different pick swings ``q`` by radians.  The frozen
        vector is scipy's choice, so an edit to ``_orthogonal_axis`` fails here
        even with no legacy tree and no scipy.
        """
        homos = antiparallel_homos()
        rv = link_vectors(homos)
        assert np.allclose(rv[1], -mc.V_BASE, atol=1e-15)   # branch really taken
        assert np.allclose(rv[4], -mc.V_BASE, atol=1e-15)
        assert abs(rv[2] @ mc.V_BASE) < 0.9                 # successor not along z
        assert abs(rv[5] @ mc.V_BASE) < 0.9
        q = mocap_to_q(homos)
        assert q is not None
        assert np.max(np.abs(q - ANTIPARALLEL_Q)) < TOL

    def test_a_different_antiparallel_axis_would_be_caught(self, monkeypatch):
        """Proof the frame above is sensitive — the gap the old frame had.

        The previous version of this check used a frame in which every link was
        parallel to its predecessor, so ``R @ v`` was ``+z`` for *any* 180 deg
        axis and substituting one changed ``q`` by exactly zero.  Show that this
        frame does not have that hole.
        """
        import mocap_to_q as m
        baseline = mocap_to_q(antiparallel_homos())
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)):
            monkeypatch.setattr(m, "_orthogonal_axis", lambda _a, _k=axis: _k)
            perturbed = m.mocap_to_q(antiparallel_homos())
            assert np.max(np.abs(perturbed - baseline)) > 1.0


# --------------------------------------------------------------------------
# 2. The scipy replacement
# --------------------------------------------------------------------------


class TestRotationAligning:
    def test_maps_from_onto_to(self):
        rng = np.random.default_rng(11)
        for _ in range(2000):
            v = rng.normal(size=3)
            target = rng.normal(size=3)
            if min(np.linalg.norm(v), np.linalg.norm(target)) < 1e-6:
                continue
            r = rotation_aligning(v, target)
            assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
            assert np.linalg.det(r) > 0
            assert np.allclose(r @ (v / np.linalg.norm(v)),
                               target / np.linalg.norm(target), atol=1e-12)

    def test_is_the_minimal_rotation(self):
        """No rotation about the target axis: the swing must be the short one."""
        rng = np.random.default_rng(12)
        for _ in range(500):
            v = rng.normal(size=3)
            r = rotation_aligning(v, mc.V_BASE)
            angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
            expected = np.arccos(np.clip(
                v @ mc.V_BASE / np.linalg.norm(v), -1.0, 1.0))
            assert abs(angle - expected) < 1e-9

    def test_parallel_is_identity(self):
        assert np.allclose(rotation_aligning(mc.V_BASE, mc.V_BASE), np.eye(3), atol=0)
        assert np.allclose(rotation_aligning(3.0 * mc.V_BASE, mc.V_BASE), np.eye(3), atol=0)

    def test_antiparallel_branch(self):
        """Exactly backwards: axis is undetermined, so check the invariants.

        The answer must still be a proper rotation that lands on the target; for
        the ``+z`` target the deterministic pick is the y axis, i.e.
        ``diag(-1, 1, -1)`` — the same choice the legacy scipy call made, which
        is why the oracle agrees here too.
        """
        r = rotation_aligning(-mc.V_BASE, mc.V_BASE)
        assert np.allclose(r @ (-mc.V_BASE), mc.V_BASE, atol=1e-15)
        assert np.allclose(r, np.diag([-1.0, 1.0, -1.0]), atol=1e-15)
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-15)
        assert np.linalg.det(r) > 0

    def test_near_antiparallel_stays_accurate(self):
        """Just off backwards is the ill-conditioned regime the branch exists for."""
        for eps in (1e-2, 1e-4, 1e-6, 1e-9, 1e-11):
            v = np.array([eps, 0.7 * eps, -1.0])
            r = rotation_aligning(v, mc.V_BASE)
            assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
            assert np.allclose(r @ (v / np.linalg.norm(v)), mc.V_BASE, atol=1e-12)

    def test_zero_length_rejected(self):
        with pytest.raises(ValueError):
            rotation_aligning(np.zeros(3), mc.V_BASE)


# --------------------------------------------------------------------------
# 3. Round trip: known q -> poses -> q
# --------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("name,q_true", STRUCTURED_CASES, ids=STRUCTURED_IDS)
    def test_identity_base(self, name, q_true):
        q = mocap_to_q(poses_from_q(q_true))
        assert q is not None
        assert np.max(np.abs(q - q_true)) < TOL

    def test_invariant_to_base_pose_and_link_lengths(self):
        """The whole point of de-rotating by the base: ``q`` must not care where
        the rig sits, how it is turned, or how long the links are."""
        rng = np.random.default_rng(99)
        worst = 0.0
        for _ in range(200):
            q_true = rng.uniform(-0.4, 0.4, mc.NUM_JOINTS)
            homos = poses_from_q(
                q_true,
                rot_base=random_rotation(rng),
                base_pos=rng.uniform(-2.0, 2.0, 3),
                lengths=rng.uniform(0.05, 0.4, 6),
            )
            q = mocap_to_q(homos)
            assert q is not None
            worst = max(worst, float(np.max(np.abs(q - q_true))))
        assert worst < TOL, f"max round-trip error {worst:g}"

    def test_link_vectors_are_unit_and_ordered(self):
        homos = poses_from_q(np.zeros(mc.NUM_JOINTS))
        rv = link_vectors(homos)
        assert rv.shape == (6, 3)
        assert np.allclose(np.linalg.norm(rv, axis=1), 1.0, atol=1e-15)
        # A straight, all-zero-q arm hangs base-up: every link vector is +z.
        assert np.allclose(rv, np.tile(mc.V_BASE, (6, 1)), atol=1e-15)


# --------------------------------------------------------------------------
# 4. Degenerate frames
# --------------------------------------------------------------------------


class TestDegenerate:
    def test_untracked_plates_give_none(self):
        """All plates still at the identity = Motive has not seen the arm yet."""
        homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
        assert mocap_to_q(homos) is None

    def test_zero_first_link_gives_none(self):
        homos = poses_from_q(np.zeros(mc.NUM_JOINTS))
        homos[1, 0:3, 3] = homos[0, 0:3, 3]
        assert mocap_to_q(homos) is None

    @needs_oracle
    def test_zero_first_link_matches_legacy(self):
        homos = poses_from_q(np.zeros(mc.NUM_JOINTS))
        homos[1, 0:3, 3] = homos[0, 0:3, 3]
        assert _oracle(homos) is None and mocap_to_q(homos) is None

    def test_zero_inner_link_gives_none_where_legacy_gives_nan(self):
        """Documented deviation: we reject *any* collapsed link, legacy only the first.

        The legacy code normalises the inner links unguarded, so a plate that
        drops out mid-chain yields ``0/0`` and silently poisons four joints with
        NaN — which a PID loop would happily act on.  Returning ``None`` for the
        whole frame is a strict superset of legacy's ``None`` cases and cannot
        change any well-formed frame's result.
        """
        homos = poses_from_q(np.zeros(mc.NUM_JOINTS))
        homos[3, 0:3, 3] = homos[2, 0:3, 3]
        assert mocap_to_q(homos) is None
        if _oracle is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                legacy = _oracle(homos)
            assert legacy is not None and not np.all(np.isfinite(legacy))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            mocap_to_q(np.tile(np.eye(4), (mc.N_USED_RIGID_BODIES - 1, 1, 1)))
        with pytest.raises(ValueError):
            mocap_to_q(np.zeros((9, 3, 3)))

    def test_zero_norm_quaternion_raises(self):
        """Motive streams ``quat=(0,0,0,0)`` for a body it cannot solve.

        Normalising that silently gives an all-NaN matrix — a bare
        ``RuntimeWarning``, no exception — so it has to fail loudly here, which is
        also what ``Rotation.from_quat`` did.
        """
        with pytest.raises(ValueError):
            quat_xyzw_to_matrix([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            quat_xyzw_to_matrix([np.nan, 0.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            quat_xyzw_to_matrix([np.inf, 0.0, 0.0, 1.0])

    def test_nan_base_rotation_gives_none(self):
        """The frame that used to be published as perfectly healthy NaN.

        A non-finite base rotation leaves every *raw* link length finite, so the
        ``MIN_LINK_NORM`` gate never sees anything wrong; only the de-rotated
        output is NaN.  ``nan < 1e-10`` is False, so the old gate waved it
        through and a 12-NaN ``q`` reached the caller.
        """
        homos = golden_homos()
        homos[mc.IDX_BASE, 0:3, 0:3] = np.nan
        assert link_vectors(homos) is None
        assert mocap_to_q(homos) is None

    def test_nan_anywhere_used_gives_none(self):
        for idx, sl in ((mc.IDX_U2_PROXIMAL, (slice(0, 3), 3)),
                        (mc.IDX_U3_DISTAL, (slice(0, 3), 2)),
                        (mc.IDX_BASE, (slice(0, 3), 3))):
            homos = golden_homos()
            homos[idx][sl] = np.nan
            assert mocap_to_q(homos) is None, f"plate {idx} NaN was accepted"

    def test_infinite_position_gives_none(self):
        homos = golden_homos()
        homos[mc.IDX_U1_DISTAL, 0:3, 3] = np.inf
        assert mocap_to_q(homos) is None

    def test_one_untracked_plate_is_NOT_rejected(self):
        """Pins the documented hole, so the docstrings stay honest.

        A *single* plate left at the identity keeps both of its neighbouring
        differences non-degenerate, so nothing here can reject the frame: joints
        9-12 silently become the direction from plate 4 to the mocap origin.
        Closing it needs a plausibility bound on link length, which would also
        make this function disagree with the oracle on the random pose sets
        :class:`TestOracle` is built from — so the behaviour is documented rather
        than changed, and pinned here so the two cannot drift apart.
        """
        homos = golden_homos(base_pos=np.array([1.0, 2.0, 3.0]))
        homos[mc.IDX_U3_DISTAL] = np.eye(4)          # plate 505 never seen
        q = mocap_to_q(homos)
        assert q is not None and np.all(np.isfinite(q))
        # The last link is now plate 5's identity z, i.e. +z in the base frame.
        rv = link_vectors(homos)
        assert np.allclose(rv[5], mc.V_BASE, atol=1e-15)

    def test_two_adjacent_untracked_plates_are_rejected(self):
        """The case the rejection really covers: neighbours collapsed together."""
        homos = golden_homos(base_pos=np.array([1.0, 2.0, 3.0]))
        homos[mc.IDX_U3_PROXIMAL] = np.eye(4)
        homos[mc.IDX_U3_DISTAL] = np.eye(4)
        assert mocap_to_q(homos) is None


# --------------------------------------------------------------------------
# 5. Oracle equivalence
# --------------------------------------------------------------------------


@needs_oracle
class TestOracle:
    def test_constants_match(self):
        """A drifted constant would show up as a plausible, wrong ``q``."""
        assert mc.NUM_JOINTS == _legacy_mc.NUM_JOINTS
        assert mc.NUM_RIGID_BODIES == _legacy_mc.NUM_RIGID_BODIES
        assert mc.N_USED_RIGID_BODIES == _legacy_mc.N_USED_RIGID_BODIES
        assert mc.RIGID_BODY_ID_MASK == _legacy_mc.RIGID_BODY_ID_MASK
        assert mc.KINOVA_MOCAP_STREAM_ID == _legacy_mc.KINOVA_MOCAP_STREAM_ID
        assert mc.KINOVA_RIGID_BODY_INDEX == _legacy_mc.KINOVA_RIGID_BODY_INDEX
        assert mc.MIN_LINK_NORM == _legacy_mc.MIN_LINK_NORM
        # Bit-identical, not just close: same expression, same rounding.
        assert np.array_equal(mc.V_BASE, _legacy_mc.V_BASE)
        assert np.array_equal(mc.RZn45, _legacy_mc.RZn45)

    def test_500_random_pose_sets(self):
        """The headline equivalence check from the design spec."""
        rng = np.random.default_rng(20260811)
        worst = 0.0
        worst_at = None
        for i in range(500):
            homos = random_pose_set(rng)
            mine = mocap_to_q(homos)
            theirs = _oracle(homos)
            assert mine is not None and theirs is not None
            d = float(np.max(np.abs(mine - theirs)))
            if d > worst:
                worst, worst_at = d, i
        assert worst < TOL, f"max |dq| = {worst:g} at random pose set #{worst_at}"

    @pytest.mark.parametrize("name,q_true", STRUCTURED_CASES, ids=STRUCTURED_IDS)
    def test_structured_cases(self, name, q_true):
        """Equivalence on physically constructed arms, not only random noise."""
        homos = poses_from_q(q_true, rot_base=random_rotation(np.random.default_rng(5)),
                             base_pos=np.array([0.3, -1.2, 2.0]),
                             lengths=np.array([0.15, 0.04, 0.15, 0.04, 0.15, 0.10]))
        mine = mocap_to_q(homos)
        theirs = _oracle(homos)
        assert mine is not None and theirs is not None
        assert np.max(np.abs(mine - theirs)) < TOL

    def test_exactly_antiparallel_link_matches_legacy(self):
        """The doubled-back frame: even the arbitrary branch agrees with scipy.

        Uses :func:`antiparallel_homos`, whose successors are *not* parallel to
        their antiparallel predecessors — so the 180 deg axis pick actually
        reaches ``q`` and a divergence from scipy's tie-break would show up here.
        The frame this test used to build had every link parallel to the one
        before it, which made ``R @ v == +z`` for any perpendicular axis and left
        the comparison vacuous.
        """
        homos = antiparallel_homos()
        mine = mocap_to_q(homos)
        theirs = _oracle(homos)
        assert mine is not None and theirs is not None
        assert np.max(np.abs(mine - theirs)) < TOL

    def test_doubled_back_frame_matches_legacy(self):
        """The all-parallel doubled-back frame, kept for its own sake.

        Weaker than the test above (the axis pick cancels here), but it is the
        degenerate shape a folded-up arm actually produces, so equivalence on it
        is still worth pinning.
        """
        homos = poses_from_q(np.zeros(mc.NUM_JOINTS))
        # Reverse links 2..6 so every align call sees prev = +z, current = -z.
        for i in range(2, 6):
            homos[i, 0:3, 3] = homos[0, 0:3, 3] + np.array([0.0, 0.0, 0.15 * (i - 1)])
        homos[mc.IDX_U3_DISTAL, 0:3, 0:3] = frame_with_z(np.array([0.0, 0.0, -1.0]))
        mine = mocap_to_q(homos)
        theirs = _oracle(homos)
        assert mine is not None and theirs is not None
        assert np.max(np.abs(mine - theirs)) < TOL

    def test_golden_values_still_match_the_oracle(self):
        """The frozen literals are only as good as their provenance — recheck it.

        :class:`TestConventionsPinned` runs everywhere but can only compare
        against numbers baked in at authoring time.  Where the legacy tree *is*
        reachable, confirm those numbers are still what it produces, so the
        goldens cannot quietly rot into a self-consistent fiction.
        """
        for homos, want in ((golden_homos(), GOLDEN_Q),
                            (golden_homos(rot_base=GOLDEN_RZ @ GOLDEN_RX,
                                          base_pos=np.array([0.3, -1.2, 2.0])), GOLDEN_Q),
                            (antiparallel_homos(), ANTIPARALLEL_Q)):
            theirs = _oracle(homos)
            assert theirs is not None
            assert np.max(np.abs(theirs - want)) < TOL


# --------------------------------------------------------------------------
# 6. The scipy replacements, gated on scipy rather than on the legacy tree
# --------------------------------------------------------------------------


@needs_scipy
class TestAgainstScipy:
    """What ``rotation_aligning``/``quat_xyzw_to_matrix`` replaced, still checked.

    These lived inside :class:`TestOracle` and so skipped on any machine without
    the research folder — including the deployment machines this package exists
    for.  scipy is what they actually need.
    """

    def test_quat_to_matrix_matches_scipy(self):
        """``mocap_rx`` builds each plate's rotation from Motive's ``[x,y,z,w]``
        quaternion; the legacy receiver used ``Rotation.from_quat``.  Same
        component order (**scalar last**), same normalisation, or every pose is
        subtly wrong in a way nothing else here would notice.
        """
        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(2000):
            quat = rng.normal(size=4)
            if np.linalg.norm(quat) < 1e-6:
                continue
            expected = _ScipyRotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
            worst = max(worst, float(np.max(np.abs(quat_xyzw_to_matrix(quat) - expected))))
        assert worst < 1e-12, f"max |dR| = {worst:g}"

    def test_zero_quaternion_raises_like_scipy(self):
        """Motive's "cannot solve this body" pose must fail, not become NaN."""
        with pytest.raises(ValueError):
            _ScipyRotation.from_quat([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            quat_xyzw_to_matrix([0.0, 0.0, 0.0, 0.0])

    def test_align_matches_scipy_over_the_sphere(self):
        rng = np.random.default_rng(17)
        worst = 0.0
        for _ in range(2000):
            a, b = rng.normal(size=3), rng.normal(size=3)
            if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-6:
                continue
            expected = _ScipyRotation.align_vectors(b, a)[0].as_matrix()
            worst = max(worst, float(np.max(np.abs(rotation_aligning(a, b) - expected))))
        assert worst < 1e-9, f"max |dR| = {worst:g}"

    def test_antiparallel_tie_break_matches_scipy(self):
        """Pins the claim ``test_antiparallel_branch`` only asserted in prose.

        That test hard-codes ``diag(-1, 1, -1)`` and its docstring calls it "the
        same choice the legacy scipy call made" — true, but nothing invoked scipy
        to find out, so a change on either side would have passed.
        """
        expected = _ScipyRotation.align_vectors(mc.V_BASE, -mc.V_BASE)[0].as_matrix()
        assert np.allclose(rotation_aligning(-mc.V_BASE, mc.V_BASE), expected, atol=1e-14)
        assert np.allclose(expected, np.diag([-1.0, 1.0, -1.0]), atol=1e-14)
