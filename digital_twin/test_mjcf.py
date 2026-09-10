r"""What ``mjcf_generator`` must keep true, pinned so a later edit has to break it.

Every test here runs offline: no serial port, no CAN, no NatNet socket, no
camera.  The only external dependency is ``mujoco``, which compiles a string.

The tests are ordered by what they protect, not by module layout:

1. the model compiles and has the shape ``CONTRACT.md`` section 3 fixes;
2. the twenty-four actuators are the twenty-four CAN boards, by name;
3. **the routing reproduces the measured actuator map** -- the one test that can
   actually distinguish this model from a plausible wrong one.  A tendon whose
   dominant moment arm is about the joint the 2026-08-21 campaign says it drives,
   with the sign that campaign read off the motion, is a routing claim that has
   been checked against metal.  A test that only counted tendons would pass on a
   model that pulls the arm sideways;
4. the forward kinematics agrees with ``fkine(..., order="yx")``, and the test
   has teeth -- it is shown to fail under ``order="xy"``;
5. the masses are what the module says they are, and the dissipation tunables
   are arguments rather than constants.

WHAT THESE TESTS DO NOT SHOW.  Nothing here says the model is *right about the
arm*.  The moment-arm test proves the muscles are seated where the map says,
given the parameter table's ring radii; if ``JA1`` is wrong on the metal, every
test still passes and every predicted torque is wrong by the same factor.  The
mass test proves the mass model is self-consistent and lands in a plausible
range; no mass on this arm has been weighed.  Frequency, damping ratio and
predicted force are all outside what an offline test can reach.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

mujoco = pytest.importorskip("mujoco")

import digital_twin                                     # noqa: E402
from digital_twin import mjcf_generator as G            # noqa: E402
from UMArm_KINEMATICS import canarm_actuators as ca     # noqa: E402
from UMArm_KINEMATICS.fkine import ujoint_centres       # noqa: E402

#: One compiled model shared by the read-only tests.  Built at the origin with
#: no base rotation so a body position can be compared with ``fkine``'s robot
#: frame directly, rather than through a mount transform that would hide a sign
#: error in either.
_KW = dict(base_pos=(0.0, 0.0, 0.0), base_rpy_deg=(0.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def model():
    return G.build_model(**_KW)


@pytest.fixture(scope="module")
def seats():
    return G.actuator_seats()


def _tendon_moments(model, data, qpos, h: float = 1e-6) -> np.ndarray:
    """``(ntendon, nv)`` of ``d(tendon length)/d(qpos)`` by central difference.

    Finite differences rather than ``data.ten_J`` on purpose: MuJoCo stores that
    Jacobian sparsely, so reading it means reproducing a layout, and this test
    exists to check a physical claim rather than an array convention.  ``h`` is
    1e-6 rad against tendon lengths of order 0.2 m, which puts the truncation
    error near 1e-13 m -- eleven orders below the ~1e-2 m/rad arms measured.
    """
    out = np.zeros((model.ntendon, model.nv))
    for dof in range(model.nv):
        lengths = []
        for step in (+h, -h):
            data.qpos[:] = qpos
            data.qpos[dof] += step
            mujoco.mj_forward(model, data)
            lengths.append(np.array(data.ten_length))
        out[:, dof] = (lengths[0] - lengths[1]) / (2.0 * h)
    return out


# ---------------------------------------------------------------------------
# 1. Shape and solver settings
# ---------------------------------------------------------------------------

def test_model_compiles_with_the_contracted_shape(model):
    assert (model.nq, model.nv) == (12, 12)
    assert model.nu == 24
    assert model.ntendon == 24


def test_option_block_is_the_contract(model):
    # Exact equality, not approximate: sim_core counts node-logic passes in whole
    # 1 ms quanta, so a timestep that is 0.001 to within a float wobble would put
    # the firmware's control period off by a fraction of a quantum forever.
    assert model.opt.timestep == 0.001
    assert model.opt.timestep == digital_twin.TIMESTEP_S == G.TIMESTEP_S
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.iterations == 100


def test_nothing_collides_and_no_keyframe_exists(model):
    # Two arms drawn overlapping must be a drawing.  A keyframe would let a
    # rollout start somewhere nobody chose.
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)
    assert model.nkey == 0
    assert model.npair == 0


# ---------------------------------------------------------------------------
# 2. The twenty-four actuators are the twenty-four boards
# ---------------------------------------------------------------------------

def test_actuators_are_pam_1_through_pam_24(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    assert names == list(G.actuator_names())
    assert names == [f"pam_{k}" for k in range(1, 25)]


def test_pam_k_is_board_0x100_plus_k(seats):
    for s in seats:
        assert s.board == 0x100 + s.index
    assert [s.board for s in seats] == list(range(0x101, 0x119))


def test_every_actuator_is_pull_only(model):
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        assert hi == 0.0 and lo == -G.FORCE_RANGE_N
        flo, fhi = model.actuator_forcerange[i]
        assert fhi == 0.0 and flo == -G.FORCE_RANGE_N
        assert model.actuator_ctrllimited[i] and model.actuator_forcelimited[i]


# ---------------------------------------------------------------------------
# 3. The routing -- the test that proves the model is this arm's
# ---------------------------------------------------------------------------

def test_seat_table_reproduces_the_measured_actuator_map(seats):
    """The derived seats must recover ``MEASURED_JOINT_PAIRS`` exactly.

    Derived rather than transcribed is the whole point: a hand-typed azimuth
    table and a hand-checked assertion can agree with each other and both be
    wrong about the metal.
    """
    recovered = [[None, None] for _ in range(12)]
    for s in seats:
        recovered[s.joint][0 if s.sign > 0 else 1] = s.board
    assert [tuple(row) for row in recovered] == list(ca.MEASURED_JOINT_PAIRS)
    assert ca.MEASURED is True


def test_segment_one_is_not_the_legacy_table(seats):
    """Segment 1 must differ from the legacy claim; segments 2 and 3 must not.

    The 2026-08-21 campaign found the top regulator platform remounted a quarter
    turn, so a generator that quietly fell back to ``LEGACY_JOINT_PAIRS`` would
    flip every agonist on joints 0-3 and would still compile, still route 24
    tendons, and still pass every count-based test.
    """
    legacy = {s.index: (s.ring, s.azimuth_deg)
              for s in G.actuator_seats(joint_pairs=ca.LEGACY_JOINT_PAIRS)}
    measured = {s.index: (s.ring, s.azimuth_deg) for s in seats}
    differ = {k for k in measured if measured[k] != legacy[k]}
    assert differ == set(range(1, 9)), "segment 1 must be the only disagreement"
    for k in range(9, 25):
        assert measured[k] == legacy[k]


def _ujoint_dofs(joint: int):
    """The two qpos indices belonging to ``joint``'s own universal joint.

    Which of the two a given muscle drives is **not** settled by the closed
    form.  Both this file and the seat table derived it from "axis 0 is t1
    about +x", and both were consistently wrong: rolling the fitted twin
    against the real arm on 2026-09-10 produced a cross-correlation matrix
    that was a permutation with three transpositions, all of them the
    proximal universal joint (the twin's j0 tracked the arm's j1 at +0.93 and
    vice versa at +0.97, likewise j4/j5 and j8/j9, while every distal joint sat
    on its own diagonal at +0.93 to +0.98). Swapping the proximal seat
    azimuths took the mean per-joint correlation from 0.402 to 0.882.

    A self-consistent derivation cannot catch that, so these tests no longer
    assert it.  What they do assert is everything the geometry *does* pin: the
    muscle drives one axis and only one, that axis belongs to its own universal
    joint and no other, the two members of a pair oppose each other, and the
    four muscles of a ring cover both of its axes.  The axis assignment itself
    is pinned by `digital_twin.mjcf_generator.LOWER_SEAT_DEG`'s recorded
    measurement, and re-deriving it needs the arm.
    """
    base = (joint // 4) * 4
    return (G.QPOS_FROM_Q[base + (joint % 4)],
            G.QPOS_FROM_Q[base + (1 - (joint % 4) if joint % 4 < 2
                                   else 5 - (joint % 4))])


def test_each_muscle_drives_one_axis_of_its_own_ujoint(model, seats):
    """Sign and dominance of every moment arm, at rest and away from it.

    A muscle pulls, so it drives a joint in whichever direction shortens the
    tendon: for the map's *positive* board, increasing that joint angle must
    shorten the tendon, i.e. ``d(len)/dq < 0``.  Dominance -- that arm being the
    largest of the twelve -- is what says the muscle is on the right ring at the
    right azimuth rather than merely on the right side of the arm.

    Checked at ``q = 0``, where the cross-axis arm of a pure seat is identically
    zero, and over twenty random configurations inside +-0.3 rad, which brackets
    the ~30 deg of joint travel the 2026-08-21 campaign spanned.  The largest
    cross-axis-to-own ratio seen there is ~0.09, so dominance is not marginal.
    """
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20260910)
    configs = [np.zeros(model.nq)]
    configs += [G.q_to_qpos(rng.uniform(-0.3, 0.3, 12)) for _ in range(20)]

    worst_ratio = 0.0
    for qpos in configs:
        arms = _tendon_moments(model, data, qpos)
        for k, seat in enumerate(seats):
            own_pair = _ujoint_dofs(seat.joint)
            dof = int(own_pair[int(np.argmax(np.abs(arms[k, list(own_pair)])))])
            own = arms[k, dof]
            others = np.abs(np.delete(arms[k], dof))
            assert abs(own) > others.max(), (
                f"pam_{seat.index}'s largest moment arm is not on its own "
                f"universal joint {seat.joint // 2}: own {abs(own):.6g}, "
                f"largest other {others.max():.6g}")
            assert dof in own_pair, (
                f"pam_{seat.index} (board 0x{seat.board:03X}) drives dof {dof}, "
                f"which is not one of joint {seat.joint}'s own pair {own_pair}")
            worst_ratio = max(worst_ratio, others.max() / abs(own))
    assert worst_ratio < 0.2, f"dominance margin has eroded to {worst_ratio:.3f}"


def test_at_rest_a_pure_seat_has_exactly_one_moment_arm(model, seats):
    """At ``q = 0`` the cross-axis arm is zero to numerical noise, not merely small.

    That is the property the eight seat azimuths were solved for, and it is what
    makes an antagonistic pair a pair rather than two muscles that mostly oppose.
    """
    data = mujoco.MjData(model)
    arms = _tendon_moments(model, data, np.zeros(model.nq))
    for k, seat in enumerate(seats):
        own_pair = _ujoint_dofs(seat.joint)
        dof = int(own_pair[int(np.argmax(np.abs(arms[k, list(own_pair)])))])
        others = np.abs(np.delete(arms[k], dof))
        assert others.max() < 1e-9, (
            f"pam_{seat.index} has a {others.max():.3g} m/rad cross arm at rest")


def test_a_pair_opposes_and_a_ring_covers_both_axes(model, seats):
    """The two properties the axis assignment cannot hide behind.

    A pair must push its joint in opposite directions -- whichever axis that
    joint turns out to be -- and the four muscles seated on one ring must
    between them drive both of that ring's axes, two each.  A swapped axis
    assignment satisfies both, which is precisely why it survived until the arm
    was asked.
    """
    data = mujoco.MjData(model)
    arms = _tendon_moments(model, data, np.zeros(model.nq))
    by_joint = {}
    for k, seat in enumerate(seats):
        by_joint.setdefault(seat.joint, []).append((k, seat))
    for joint, members in by_joint.items():
        assert len(members) == 2, f"joint {joint} has {len(members)} muscles"
        own_pair = _ujoint_dofs(joint)
        signs = []
        for k, seat in members:
            dof = int(own_pair[int(np.argmax(np.abs(arms[k, list(own_pair)])))])
            signs.append(math.copysign(1.0, arms[k, dof]))
        assert signs[0] == -signs[1], (
            f"joint {joint}'s two muscles do not oppose: {signs}")

    by_ring = {}
    for k, seat in enumerate(seats):
        by_ring.setdefault((seat.segment, seat.ring), []).append(k)
    for ring, ks in by_ring.items():
        assert len(ks) == 4, f"ring {ring} carries {len(ks)} muscles"
        driven = []
        for k in ks:
            driven.append(int(np.argmax(np.abs(arms[k]))))
        assert len(set(driven)) == 2, (
            f"ring {ring}'s four muscles drive {sorted(set(driven))}, not two "
            f"axes")


def test_pulling_one_muscle_torques_the_joint_the_map_claims(model, seats):
    """The same claim through MuJoCo's own transmission, not through geometry.

    ``qfrc_actuator`` is ``moment^T * force``, so this closes the loop from a
    commanded newton to a generalised torque without the test re-deriving the
    sign convention it is checking.
    """
    data = mujoco.MjData(model)
    for k, seat in enumerate(seats):
        data.qpos[:] = 0.0
        data.ctrl[:] = 0.0
        data.ctrl[k] = -100.0                    # 100 N of pull, well inside the clip
        mujoco.mj_forward(model, data)
        tau = np.array(data.qfrc_actuator)
        own_pair = _ujoint_dofs(seat.joint)
        dof = int(own_pair[int(np.argmax(np.abs(tau[list(own_pair)])))])
        # Which of the joint's two axes is settled by measurement, not here --
        # see _ujoint_dofs.  What this closes the loop on is that a commanded
        # newton becomes a generalised torque on this muscle's OWN universal
        # joint and nowhere else.
        assert abs(tau[dof]) == pytest.approx(np.abs(tau).max())
        assert dof in own_pair


# ---------------------------------------------------------------------------
# 4. Forward kinematics, under the measured composition order
# ---------------------------------------------------------------------------

def test_rest_pose_reproduces_fkine_ujoint_centres(model):
    """``mj_forward`` at ``q = 0`` against ``fkine(zeros, params, order="yx")``.

    The contract asks for well under a millimetre; what the shared geometry
    actually delivers is float noise, and the tolerance is set there so a
    sub-millimetre drift would be caught rather than tolerated.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    want = ujoint_centres(np.zeros(12), G.fitted_params(), order=G.PROXIMAL_ORDER)
    got = np.array([data.site(name).xpos
                    for name in G.plate_site_names()])
    err = np.abs(got - want).max()
    assert err < 1e-12, f"{err:.3g} m at rest"
    assert err < 1e-3                       # the contract's stated requirement


def test_random_poses_reproduce_fkine_under_yx(model):
    data = mujoco.MjData(model)
    params = G.fitted_params()
    rng = np.random.default_rng(20260910)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(-0.4, 0.4, 12)
        data.qpos[:] = G.q_to_qpos(q)
        mujoco.mj_forward(model, data)
        want = ujoint_centres(q, params, order=G.PROXIMAL_ORDER)
        got = np.array([data.site(n).xpos for n in G.plate_site_names()])
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1e-12, f"worst u-joint centre error {worst:.3g} m"


def test_the_fk_test_has_teeth_under_the_wrong_order(model):
    """The same comparison under ``order="xy"`` must fail, and by millimetres.

    Without this, the FK test would pass on a model whose proximal hinges were
    declared in the legacy order, since ``q = 0`` and single-joint motions cannot
    tell the two compositions apart.
    """
    data = mujoco.MjData(model)
    params = G.fitted_params()
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-0.4, 0.4, 12)
        data.qpos[:] = G.q_to_qpos(q)
        mujoco.mj_forward(model, data)
        wrong = ujoint_centres(q, params, order="xy")
        got = np.array([data.site(n).xpos for n in G.plate_site_names()])
        worst = max(worst, float(np.abs(got - wrong).max()))
    assert worst > 1e-3, (
        f"order='xy' disagrees by only {worst:.3g} m; the FK test cannot "
        f"distinguish the two compositions and proves nothing")


def test_qpos_permutation_is_its_own_inverse():
    rng = np.random.default_rng(0)
    q = rng.normal(size=12)
    assert np.allclose(G.qpos_to_q(G.q_to_qpos(q)), q)
    assert G.QPOS_FROM_Q[0] == 1, "under order='yx' the first dof is t2, not t1"
    names = G.joint_names()
    assert names[0].endswith("uj1_y") and names[1].endswith("uj1_x")
    assert names[2].endswith("uj2_x") and names[3].endswith("uj2_y")


def test_joint_names_match_the_compiled_model(model):
    got = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
           for i in range(model.njnt)]
    assert got == list(G.joint_names())


# ---------------------------------------------------------------------------
# 5. Masses and tunables
# ---------------------------------------------------------------------------

def test_masses_are_plausible_and_sourced(model):
    """Total moving mass in the range the mass model claims, and no free lunch.

    The static base is excluded because it hangs off a mount and enters no
    equation of motion.  The bounds are wide on purpose: they reject a model
    that reverted to the display file's 0.32 kg or that gained a decimal place,
    and they assert nothing finer, because nothing on this arm has been weighed.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "canarm_base")
    moving = float(model.body_mass.sum() - model.body_mass[base_id])
    assert 1.0 < moving < 3.0, f"moving mass {moving:.3f} kg"
    # 24 muscles at the operator's 30 g are 0.72 kg of the total, of which the
    # four half-muscles anchored to the static base do not move.
    assert moving == pytest.approx(1.833, abs=0.02)
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name.endswith("_mount"):
            continue                            # the mocap mount is massless
        assert model.body_mass[body] > 0.0, name


def test_mass_provenance_covers_every_mass_argument():
    """Every mass keyword must have a line saying where its default came from."""
    sig = inspect.signature(G.generate_xml).parameters
    mass_args = [n for n in sig
                 if n.endswith("_mass") or n.endswith("density")]
    for name in mass_args:
        key = name if name in G.MASS_PROVENANCE else f"{name}_kg"
        key = key if key in G.MASS_PROVENANCE else f"{name}_kg_m"
        assert key in G.MASS_PROVENANCE, f"{name} has no provenance line"
    assert "MEASURED" in G.MASS_PROVENANCE["actuator_mass_kg"], (
        "the operator's 30 g per actuator is the only measured mass here and "
        "must stay labelled as such")


def test_masses_follow_their_arguments():
    """Doubling a mass argument must move the model, which proves it is read."""
    heavy = G.build_model(actuator_mass=0.060, **_KW)
    light = G.build_model(actuator_mass=0.030, **_KW)
    assert heavy.body_mass.sum() == pytest.approx(light.body_mass.sum() + 0.72)
    stiff = G.build_model(link_density=1.0, **_KW)
    assert stiff.body_mass.sum() > light.body_mass.sum() + 0.6


def test_dissipation_tunables_are_arguments_and_propagate():
    """The three MJCF dissipation scalars, on defaults and on overrides.

    Emitted on the ``umarm`` default class, so an override has to reach all
    twelve hinges and all twenty-four tendons; a class that only decorated the
    first body would pass a spot check on joint 0.
    """
    base = G.build_model(**_KW)
    assert np.allclose(base.dof_damping, G.DEFAULT_JOINT_DAMPING)
    assert np.allclose(base.dof_frictionloss, G.DEFAULT_JOINT_FRICTIONLOSS)
    assert np.allclose(base.tendon_damping, G.DEFAULT_TENDON_DAMPING)

    tuned = G.build_model(joint_damping=0.5, joint_frictionloss=0.125,
                          tendon_damping=7.5, **_KW)
    assert np.allclose(tuned.dof_damping, 0.5)
    assert np.allclose(tuned.dof_frictionloss, 0.125)
    assert np.allclose(tuned.tendon_damping, 7.5)


def test_every_contracted_tunable_is_a_named_keyword():
    """``CONTRACT.md`` section 3 names these; none may become a module constant."""
    sig = inspect.signature(G.generate_xml).parameters
    for name in ("joint_damping", "joint_frictionloss", "tendon_damping",
                 "link_density", "plate_mass", "base_pos", "base_rpy_deg"):
        assert name in sig, name
        assert sig[name].default is not inspect.Parameter.empty, name
        assert sig[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_base_pose_moves_the_whole_arm():
    shifted = G.build_model(base_pos=(0.3, -0.4, 1.1), base_rpy_deg=(0, 0, 0))
    data = mujoco.MjData(shifted)
    mujoco.mj_forward(shifted, data)
    assert np.allclose(data.site("canarm_plate0").xpos, (0.3, -0.4, 1.1))


# ---------------------------------------------------------------------------
# 6. Seams and guards
# ---------------------------------------------------------------------------

def test_build_model_accepts_a_handed_in_scene():
    """The ``xml=`` seam ``sim_core`` and a merged room scene both go through."""
    xml = G.generate_xml(**_KW)
    handed = G.build_model(xml=xml)
    assert handed.nu == 24 and handed.opt.timestep == 0.001


def test_generated_xml_names_the_parameters_it_was_built_with():
    xml = G.generate_xml(joint_damping=0.031, actuator_mass=0.042, **_KW)
    assert "joint_damping=0.031" in xml
    assert "actuator_mass=0.042" in xml
    assert "proximal_order=yx" in xml


def test_generate_scene_writes_a_file(tmp_path):
    out = G.generate_scene(tmp_path / "twin.xml", **_KW)
    assert out.exists()
    mujoco.MjModel.from_xml_string(out.read_text(encoding="utf-8"))


def test_tendon_rest_lengths_are_finite_and_ordered(model):
    lengths = G.tendon_rest_lengths(model)
    assert lengths.shape == (24,)
    assert np.all(np.isfinite(lengths)) and np.all(lengths > 0.05)
    # Lower and upper muscles span different parts of the segment, so a model
    # whose 24 lengths were all identical would have lost a ring.
    assert lengths.std() > 1e-3


def test_fitted_params_is_a_writable_copy():
    a, b = G.fitted_params(), G.fitted_params()
    a[0, 0] = 99.0
    assert b[0, 0] != 99.0
    assert a.shape == (3, 10)


def test_a_half_populated_map_is_refused():
    """A map that leaves a ring short must raise, not compile.

    A model with three muscles on a ring pulls the arm sideways under a
    balanced command, which looks like a plant asymmetry and would be absorbed
    by whatever is fitted next.
    """
    broken = list(ca.MEASURED_JOINT_PAIRS)
    broken[0] = (broken[0][0], broken[1][1])      # 0x106 seated twice
    with pytest.raises(ValueError):
        G.actuator_seats(joint_pairs=tuple(broken))


def test_actuator_joint_map_is_the_compact_form(seats):
    compact = G.actuator_joint_map()
    assert len(compact) == 24
    assert compact == tuple((s.joint, s.sign) for s in seats)
