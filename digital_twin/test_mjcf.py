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
4. **the routing is the ProMax's** (Fig. 1C of ``2606.29731v1.pdf``): every
   tendon runs actuator end -> hub bearing ``AA`` from the joint it drives ->
   outer-ring bracket across that joint, its moment arm at rest is the closed
   form ``AA*JA/sqrt((JA-AO)^2+AA^2)``, and the sleeves and Ys ride the link
   body.  The far-ring routing this replaced on 2026-09-10 fails section 4 by
   centimetres, not by tolerances;
5. the forward kinematics agrees with ``fkine(..., order="yx")``, and the test
   has teeth -- it is shown to fail under ``order="xy"``;
6. the per-segment mass model does what it says, and the dissipation tunables
   are arguments rather than constants.

WHAT THESE TESTS DO NOT SHOW.  Nothing here says the model is *right about the
arm*.  The moment-arm tests prove the muscles are routed where the map and the
figure say, given the parameter table's ring radii and hub heights; if ``JA``,
``AO`` or ``AA`` is wrong on the metal, every test still passes and every
predicted torque is wrong by the same factor.  The Y tip radius is estimated from
the figure, and the tests pin only that it is honoured.  The mass tests prove the
mass model is self-consistent; no mass on this arm has been weighed.  Frequency,
damping ratio and predicted force are all outside what an offline test can reach.
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

#: m.  The generator writes coordinates with 10 significant digits, so a site
#: placed by ``cos``/``sin`` at a 47 mm radius is exact to about 1e-11 m.  Chain
#: positions are short decimals and stay at float noise (the FK tests hold them
#: to 1e-12); trig-placed routing sites are held to this.
_XML_TOL_M = 1e-9


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
    1e-6 rad against tendon lengths of order 0.1 m, which puts the truncation
    error near 1e-13 m -- eleven orders below the ~4e-2 m/rad arms measured.
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


def _id(model, kind, name: str) -> int:
    i = mujoco.mj_name2id(model, kind, name)
    assert i >= 0, f"{name} is not in the model"
    return int(i)


def _body(model, name: str) -> int:
    return _id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _across_body(seat) -> str:
    """The body on the far side of the joint this muscle drives."""
    if seat.ring == "upper":
        return f"canarm_seg{seat.segment}_plate2"
    return "canarm_base" if seat.segment == 1 else f"canarm_seg{seat.segment - 1}_plate2"


def _driven_plate(seat) -> str:
    """The plate site at the centre of the u-joint this muscle drives."""
    n = seat.segment
    return f"canarm_plate{2 * (n - 1) if seat.ring == 'lower' else 2 * n - 1}"


def _ring_numbers(seat):
    g = G.segment_geometry()[seat.segment - 1]
    if seat.ring == "lower":
        return g, g.aa1, g.ja1, g.ao1
    return g, g.aa2, g.ja2, g.ao2


def _geom_mass(xml: str, name: str) -> float:
    """One geom's ``mass`` attribute, read from the document.

    From the XML because a compiled ``MjModel`` keeps only body masses: the
    compiler folds each geom's mass into its body's inertia and discards it.
    """
    import xml.etree.ElementTree as ET

    for geom in ET.fromstring(xml).iter("geom"):
        if geom.get("name") == name:
            return float(geom.get("mass"))
    raise AssertionError(f"{name} is not in the document")


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

    The weaker of the two axis claims, kept because a failure here separates
    "wrong u-joint" from "wrong axis of the right u-joint".  Which axis is
    asserted exactly by
    :func:`test_each_muscle_drives_the_hinge_its_measured_joint_lives_on`.

    An earlier version of this docstring said the closed form ("axis 0 is t1
    about +x") had been proven wrong by the arm.  It had not.  The 2026-09-10
    permutation between the twin's joints and the arm's (j0<->j1, j4<->j5,
    j8<->j9) came from ``SimArm.q()`` returning ``qpos`` as ``q``; rotating the
    proximal seats 90 deg hid it and put every proximal muscle on the wrong
    world axis.  Both are undone, and the evidence is written out under
    ``mjcf_generator.LOWER_SEAT_DEG``.
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
    cross-axis-to-own ratio seen there is 0.095 under the bearing routing (0.082
    under the far-ring routing it replaced), so dominance is not marginal.
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


def test_each_muscle_drives_the_hinge_its_measured_joint_lives_on(model, seats):
    """The axis assignment itself: ``qpos[QPOS_FROM_Q[joint]]``, with the map's sign.

    ``_ujoint_dofs`` stops at "one of its own u-joint's two axes", and that
    tolerance is what let a 90 deg rotation of every proximal seat survive from
    commit ``a35f94a`` until 2026-09-10.  The rotation hid a read error --
    ``SimArm.q()`` returned ``qpos`` as ``q`` -- and both are now fixed, so the
    assignment is asserted exactly: at ``q = 0`` each muscle's largest moment
    arm is on the hinge that carries its measured joint, and increasing that
    joint shortens the positive board's tendon (``d(len)/dq < 0``).  ``q[j]``
    lives at ``qpos[QPOS_FROM_Q[j]]`` because the plate sites reproduce
    ``fkine(order="yx")`` under :func:`mjcf_generator.q_to_qpos` to 4.4e-16 m,
    so this pins a physical axis and not a naming convention.
    """
    data = mujoco.MjData(model)
    arms = _tendon_moments(model, data, np.zeros(model.nq))
    names = G.joint_names()
    for k, seat in enumerate(seats):
        dof = int(np.argmax(np.abs(arms[k])))
        want = G.QPOS_FROM_Q[seat.joint]
        assert dof == want, (
            f"pam_{seat.index} (board 0x{seat.board:03X}) drives {names[dof]}; "
            f"its measured joint q{seat.joint} lives on {names[want]}")
        assert math.copysign(1.0, arms[k, dof]) == -seat.sign, (
            f"pam_{seat.index} drives q{seat.joint} the wrong way: "
            f"d(len)/dq = {arms[k, dof]:+.6g} m/rad for sign {seat.sign:+d}")


def test_the_a35f94a_seat_rotation_fails_the_axis_test(monkeypatch):
    """Teeth: the proximal table that shipped from ``a35f94a`` fails on all 12.

    That table scored 11.746 deg of held-out joint RMS against 9.691 deg for
    the closed form (same flow net, same outer fit, bearing routing), so a test
    that let it back in would be letting back 2 deg of error.
    """
    monkeypatch.setattr(G, "LOWER_SEAT_DEG",
                        {(0, +1): 180.0, (0, -1): 0.0, (1, +1): 90.0, (1, -1): 270.0})
    swapped = G.actuator_seats()
    model = G.build_model(seats=swapped, **_KW)
    data = mujoco.MjData(model)
    arms = _tendon_moments(model, data, np.zeros(model.nq))
    wrong = [s.index for k, s in enumerate(swapped) if s.ring == "lower"
             and int(np.argmax(np.abs(arms[k]))) != G.QPOS_FROM_Q[s.joint]]
    assert len(wrong) == 12, f"only {len(wrong)} of 12 proximal muscles fail: {wrong}"


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
        # Which of the joint's two axes is asserted exactly by
        # test_each_muscle_drives_the_hinge_its_measured_joint_lives_on.  What
        # this closes the loop on is that a commanded newton becomes a
        # generalised torque on this muscle's OWN universal joint and nowhere
        # else.
        assert abs(tau[dof]) == pytest.approx(np.abs(tau).max())
        assert dof in own_pair


# ---------------------------------------------------------------------------
# 4. The ProMax routing and drawing (Fig. 1C)
# ---------------------------------------------------------------------------

def test_every_link_side_routing_site_lies_AA_from_the_joint_it_drives(model, seats):
    """The far-ring regression guard.

    A proximal muscle's bearing is on the TOP hub, ``AA1`` below the proximal
    centre it drives; a distal muscle's is on the BOTTOM hub, ``AA2`` above the
    distal centre.  Until 2026-09-10 each proximal tendon ran to a ring
    ``AA1 + LL`` below its joint (222 mm on segment 1) and each distal tendon to
    a ring ``AA1`` below the proximal centre, i.e. ``LL + AA2`` above its joint;
    both are more than 140 mm from where this test demands the bearing, so no
    tolerance can let that routing back in.  Checked in world coordinates at
    ``q = 0`` against the plate site at the driven joint's centre, which is the
    distance the claim is about, and in the link body's own frame.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for s in seats:
        g, aa, _ja, ao = _ring_numbers(s)
        names = G.routing_site_names(s)
        sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, names["brg"])
        assert model.site_bodyid[sid] == _body(model, f"canarm_seg{s.segment}_link")
        centre = data.site(_driven_plate(s)).xpos
        brg = data.site_xpos[sid]
        # below the proximal joint (-AA1) or above the distal joint (+AA2)
        want_dz = -aa if s.ring == "lower" else +aa
        assert brg[2] - centre[2] == pytest.approx(want_dz, abs=_XML_TOL_M), (
            f"pam_{s.index}'s bearing is {1e3 * (brg[2] - centre[2]):.2f} mm from "
            f"the joint it drives, not {1e3 * want_dz:.2f} mm")
        assert math.hypot(brg[0] - centre[0], brg[1] - centre[1]) == \
            pytest.approx(ao, abs=_XML_TOL_M)
        az = math.degrees(math.atan2(brg[1] - centre[1], brg[0] - centre[0]))
        assert abs((az - s.azimuth_deg + 180.0) % 360.0 - 180.0) < 1e-6
        far_dz = -(g.aa1 + g.ll) if s.ring == "lower" else +(g.ll + g.aa2)
        assert abs((brg[2] - centre[2]) - far_dz) > 0.14


def test_each_tendon_runs_actuator_end_then_bearing_then_bracket(model, seats):
    """Three sites, in that order, on the bodies the ProMax puts them on.

    The actuator end and the bearing are rigid with the link, so the first span
    is a constant; the bracket sits on the outer ring across the driven joint --
    the PARENT for a proximal muscle, the distal plate body for a distal one --
    in that joint's plane at the ring radius.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for s in seats:
        _g, _aa, ja, _ao = _ring_numbers(s)
        tid = _id(model, mujoco.mjtObj.mjOBJ_TENDON, f"t_pam_{s.index}")
        adr, num = int(model.tendon_adr[tid]), int(model.tendon_num[tid])
        assert num == 3
        assert all(int(t) == int(mujoco.mjtWrap.mjWRAP_SITE)
                   for t in model.wrap_type[adr:adr + 3])
        got = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(i))
               for i in model.wrap_objid[adr:adr + 3]]
        names = G.routing_site_names(s)
        assert got == [names["end"], names["brg"], names["brk"]]
        link = _body(model, f"canarm_seg{s.segment}_link")
        ids = [int(i) for i in model.wrap_objid[adr:adr + 3]]
        assert [int(model.site_bodyid[i]) for i in ids] == \
            [link, link, _body(model, _across_body(s))]
        centre = data.site(_driven_plate(s)).xpos
        brk = data.site_xpos[ids[2]]
        assert brk[2] == pytest.approx(centre[2], abs=_XML_TOL_M)
        assert math.hypot(*(brk[:2] - centre[:2])) == pytest.approx(ja, abs=_XML_TOL_M)


def test_at_rest_each_moment_arm_is_the_bearing_closed_form(model, seats):
    """One axis only, the measured sign, and ``AA*JA/sqrt((JA-AO)^2+AA^2)`` to 1e-6.

    The sign is the map's: the positive board's tendon shortens as the joint it
    drives turns positive.  The magnitude is 43.105 mm on this arm; the far-ring
    routing gave 46.8 mm, 3.7 mm outside the tolerance.
    """
    data = mujoco.MjData(model)
    arms = _tendon_moments(model, data, np.zeros(model.nq))
    for k, s in enumerate(seats):
        _g, aa, ja, ao = _ring_numbers(s)
        want = G.bearing_moment_arm_m(aa, ja, ao)
        nonzero = np.flatnonzero(np.abs(arms[k]) > 1e-9)
        assert len(nonzero) == 1, f"pam_{s.index} moves dofs {nonzero.tolist()}"
        dof = int(nonzero[0])
        assert dof in _ujoint_dofs(s.joint)
        assert arms[k, dof] == pytest.approx(-s.sign * want, abs=1e-6), (
            f"pam_{s.index}: {1e3 * arms[k, dof]:+.4f} mm/rad, want "
            f"{-1e3 * s.sign * want:+.4f}")
    assert G.bearing_moment_arm_m(0.0437125, 0.047, 0.028) == \
        pytest.approx(0.043105, abs=5e-6)


def test_the_q_dependent_length_is_the_bearing_to_bracket_span(model, seats):
    """``ten_length`` = the rigid sleeve-to-bearing span + |bearing - bracket|.

    The first term must not move with ``q`` -- it is what cancels in
    ``ten_length - tendon_length0``, which every consumer of tendon length uses.
    """
    data = mujoco.MjData(model)
    rng = np.random.default_rng(11)
    rigid0 = None
    for trial in range(8):
        q = np.zeros(12) if trial == 0 else rng.uniform(-0.4, 0.4, 12)
        data.qpos[:] = G.q_to_qpos(q)
        mujoco.mj_forward(model, data)
        rigid = []
        for k, s in enumerate(seats):
            n = G.routing_site_names(s)
            end, brg, brk = (data.site(n[key]).xpos for key in ("end", "brg", "brk"))
            rigid.append(float(np.linalg.norm(brg - end)))
            span = float(np.linalg.norm(brk - brg))
            assert data.ten_length[k] == pytest.approx(rigid[-1] + span, abs=1e-12)
        if rigid0 is None:
            rigid0 = np.array(rigid)
        assert np.allclose(rigid, rigid0, atol=1e-12)


def test_sleeves_ys_hubs_and_rod_ride_the_link_body(model, seats):
    """On the ProMax everything but the u-joint outer rings is rigid with the rod.

    The model this replaced split each muscle in half between a u-joint plate and
    the link, which is what the 2026-09-10 video showed as Y arms connected to
    the u-joint.
    """
    xml = G.generate_xml(**_KW)
    for s in seats:
        link = _body(model, f"canarm_seg{s.segment}_link")
        for part in ("sleeve", "yarm", "ytip", "ybrace", "bearing"):
            gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM,
                      f"canarm_s{s.segment}_a{s.index}_{part}")
            assert model.geom_bodyid[gid] == link, (s.index, part)
        gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM,
                  f"canarm_s{s.segment}_a{s.index}_bracket")
        assert model.geom_bodyid[gid] == _body(model, _across_body(s))
        assert _geom_mass(xml, f"canarm_s{s.segment}_a{s.index}_sleeve") > 0.0
    for n in (1, 2, 3):
        link = _body(model, f"canarm_seg{n}_link")
        for part in ("rod", "hub_top", "hub_bot"):
            gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"canarm_seg{n}_{part}")
            assert model.geom_bodyid[gid] == link
        parent = "canarm_base" if n == 1 else f"canarm_seg{n - 1}_plate2"
        assert model.geom_bodyid[_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     f"canarm_seg{n}_plate1_geom")] == _body(model, parent)
        assert model.geom_bodyid[_id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     f"canarm_seg{n}_plate2_geom")] == \
            _body(model, f"canarm_seg{n}_plate2")
    sleeve_bodies = {int(model.geom_bodyid[i]) for i in range(model.ngeom)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                     .endswith("_sleeve")}
    assert sleeve_bodies == {_body(model, f"canarm_seg{n}_link") for n in (1, 2, 3)}


def test_the_two_y_tip_azimuth_sets_sit_45_deg_apart(model, seats):
    """Upright Ys at 45/135/225/315, upside-down at 0/90/180/270, tips outside the disks.

    Also that the upright tips surround the proximal u-joint and the upside-down
    tips the distal one, at the tip radius asked for -- the geometry that puts
    each u-joint inside a Y cone.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for n in (1, 2, 3):
        up, down = [], []
        for s in (s for s in seats if s.segment == n):
            tip = data.site(G.routing_site_names(s)["tip"]).xpos
            centre = data.site(_driven_plate(s)).xpos
            # the tip sits in the plane of the joint at the OTHER end of the
            # segment from the joint the muscle drives
            other = (f"canarm_plate{2 * n - 1}" if s.ring == "lower"
                     else f"canarm_plate{2 * (n - 1)}")
            plane = data.site(other).xpos
            assert tip[2] == pytest.approx(plane[2] - (G.DEFAULT_Y_TIP_Z_OFFSET_M
                                                       if s.ring == "lower"
                                                       else -G.DEFAULT_Y_TIP_Z_OFFSET_M),
                                           abs=_XML_TOL_M)
            r = math.hypot(tip[0] - centre[0], tip[1] - centre[1])
            assert r == pytest.approx(G.DEFAULT_Y_TIP_RADIUS_M, abs=_XML_TOL_M)
            assert r > G.segment_geometry()[n - 1].ja1 + G.DEFAULT_ACTUATOR_RADIUS_M
            az = round(math.degrees(math.atan2(tip[1], tip[0])), 6) % 360.0
            (down if s.ring == "lower" else up).append(az)
        assert sorted(up) == [45.0, 135.0, 225.0, 315.0]
        assert sorted(down) == [0.0, 90.0, 180.0, 270.0]
        gap = min(min(abs(a - b) % 360.0, 360.0 - abs(a - b) % 360.0)
                  for a in up for b in down)
        assert gap == pytest.approx(45.0)


def test_sleeves_hang_clear_of_the_ujoint_disks(model):
    """No sleeve capsule reaches a u-joint disk, with 1 mm to spare, at rest."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
             for i in range(model.ngeom)]
    sleeves = [i for i, n in enumerate(names) if n.endswith("_sleeve")]
    disks = [i for i, n in enumerate(names)
             if n.endswith("_plate1_geom") or n.endswith("_plate2_geom")]
    assert len(sleeves) == 24 and len(disks) == 6
    worst = np.inf
    for i in sleeves:
        c, axis = data.geom_xpos[i], data.geom_xmat[i].reshape(3, 3)[:, 2]
        radius, half = model.geom_size[i, 0], model.geom_size[i, 1]
        pts = c[None, :] + np.linspace(-half, half, 101)[:, None] * axis[None, :]
        for j in disks:
            dc, dax = data.geom_xpos[j], data.geom_xmat[j].reshape(3, 3)[:, 2]
            big_r, h = model.geom_size[j, 0], model.geom_size[j, 1]
            rel = pts - dc[None, :]
            dz = np.abs(rel @ dax)
            rho = np.linalg.norm(rel - np.outer(rel @ dax, dax), axis=1)
            gap = np.hypot(np.maximum(rho - big_r, 0.0), np.maximum(dz - h, 0.0))
            worst = min(worst, float(gap.min()) - radius)
    assert worst > 1e-3, f"a sleeve comes within {1e3 * worst:.2f} mm of a disk"


# ---------------------------------------------------------------------------
# 5. Forward kinematics, under the measured composition order
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
# 6. Masses and tunables
# ---------------------------------------------------------------------------

def _body_mass(model, name: str) -> float:
    return float(model.body_mass[_body(model, name)])


def test_masses_are_plausible_and_sourced(model):
    """Total moving mass is the default mass model's, and no body is massless.

    The static base is excluded because it hangs off a mount and enters no
    equation of motion.  CHANGED 2026-09-10, with the routing: the default used
    to be the RS485 carry-over component model at 1.833 kg, with half of every
    muscle anchored to a u-joint plate.  On the ProMax the sleeve hangs from a Y
    that is rigid with the rod, so that split was wrong, and the default is now
    the Koopman ProMax model's per-segment prior -- links 0.70/0.50/0.50 kg,
    brackets 0.20/0.20/0.10 kg -- plus the 20 g tip stub, 2.22 kg in all.  The
    number asserts the model is what the module says; nothing has been weighed.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "canarm_base")
    moving = float(model.body_mass.sum() - model.body_mass[base_id])
    assert 1.0 < moving < 3.0, f"moving mass {moving:.3f} kg"
    want = (sum(G.DEFAULT_LINK_MASS_KG) + sum(G.DEFAULT_BRACKET_MASS_KG)
            + G.DEFAULT_TIP_MASS_KG)
    assert moving == pytest.approx(want, abs=1e-9)
    assert moving == pytest.approx(2.22, abs=0.005)
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name.endswith("_mount"):
            continue                            # the mocap mount is massless
        assert model.body_mass[body] > 0.0, name


def test_mass_provenance_covers_every_mass_argument():
    """Every mass keyword must have a line saying where its default came from."""
    sig = inspect.signature(G.generate_xml).parameters
    mass_args = [n for n in sig
                 if n.endswith("_mass") or n.endswith("_mass_kg")
                 or n.endswith("density")]
    assert {"link_mass_kg", "bracket_mass_kg", "y_mass", "hub_mass",
            "spacer_mass"} <= set(mass_args)
    for name in mass_args:
        key = name if name in G.MASS_PROVENANCE else f"{name}_kg"
        key = key if key in G.MASS_PROVENANCE else f"{name}_kg_m"
        assert key in G.MASS_PROVENANCE, f"{name} has no provenance line"
    assert "MEASURED" in G.MASS_PROVENANCE["actuator_mass_kg"], (
        "the operator's 30 g per actuator is the only measured mass here and "
        "must stay labelled as such")
    for key in ("link_mass_kg", "bracket_mass_kg"):
        assert "Koopman" in G.MASS_PROVENANCE[key]
        assert "NOT measured" in G.MASS_PROVENANCE[key]


def test_masses_follow_their_arguments():
    """Doubling a component mass must move the component model, which proves it is read.

    Run with both totals set to ``None``, because with a total given a component
    argument sets a proportion and not a mass -- which the second half pins.
    """
    free = dict(link_mass_kg=None, bracket_mass_kg=None, **_KW)
    heavy = G.build_model(actuator_mass=0.060, **free)
    light = G.build_model(actuator_mass=0.030, **free)
    # all 24 sleeves ride link bodies now, so every one of them counts
    assert heavy.body_mass.sum() == pytest.approx(light.body_mass.sum() + 0.72)
    stiff = G.build_model(link_density=1.0, **free)
    assert stiff.body_mass.sum() > light.body_mass.sum() + 0.6

    pinned = G.build_model(**_KW)
    pinned_heavy = G.build_model(actuator_mass=0.060, **_KW)
    assert pinned_heavy.body_mass.sum() == pytest.approx(pinned.body_mass.sum())
    # CHANGED 2026-09-10: under a pinned link total the sleeve keeps its own
    # mass exactly -- the one number measured on this arm -- and the structure
    # (rod, hubs, Ys) gives way.  The old rule scaled the sleeve with the total
    # and put 43 g sleeves on the 0.70 kg prior link.
    span1 = G.segment_geometry()[0].span
    structure = (G.DEFAULT_LINK_DENSITY_KG_M * span1 + 2 * G.DEFAULT_HUB_MASS_KG
                 + 4 * G.DEFAULT_Y_MASS_KG)
    for act in (0.030, 0.060):
        xml = G.generate_xml(actuator_mass=act, **_KW)
        assert _geom_mass(xml, "canarm_s1_a1_sleeve") == pytest.approx(act, rel=1e-12)
        assert _geom_mass(xml, "canarm_seg1_hub_top") == pytest.approx(
            G.DEFAULT_HUB_MASS_KG * (G.DEFAULT_LINK_MASS_KG[0] - 8 * act) / structure,
            rel=1e-9)


def test_per_segment_mass_totals_are_honoured():
    """``link_mass_kg`` and ``bracket_mass_kg`` land on the bodies they name.

    Three forms (tuple, one number, ``None``), the default, the proportional
    spread, and a refusal of a malformed total.
    """
    links, brackets = (0.9, 0.6, 0.4), (0.3, 0.25, 0.15)
    m = G.build_model(link_mass_kg=links, bracket_mass_kg=brackets, **_KW)
    for n in (1, 2, 3):
        assert _body_mass(m, f"canarm_seg{n}_link") == pytest.approx(links[n - 1], abs=1e-9)
        tip = G.DEFAULT_TIP_MASS_KG if n == 3 else 0.0
        assert _body_mass(m, f"canarm_seg{n}_plate2") == \
            pytest.approx(brackets[n - 1] + tip, abs=1e-9)

    one = G.build_model(link_mass_kg=0.55, bracket_mass_kg=0.2, **_KW)
    for n in (1, 2, 3):
        assert _body_mass(one, f"canarm_seg{n}_link") == pytest.approx(0.55, abs=1e-9)

    default = G.build_model(**_KW)
    for n in (1, 2, 3):
        assert _body_mass(default, f"canarm_seg{n}_link") == \
            pytest.approx(G.DEFAULT_LINK_MASS_KG[n - 1], abs=1e-9)

    geo = G.segment_geometry()
    sleeves = 8 * G.DEFAULT_ACTUATOR_MASS_KG
    structure = [G.DEFAULT_LINK_DENSITY_KG_M * g.span + 2 * G.DEFAULT_HUB_MASS_KG
                 + 4 * G.DEFAULT_Y_MASS_KG for g in geo]
    free = G.build_model(link_mass_kg=None, bracket_mass_kg=None, **_KW)
    for n in (1, 2, 3):
        assert _body_mass(free, f"canarm_seg{n}_link") == pytest.approx(
            structure[n - 1] + sleeves, abs=1e-9)
    # The sleeves keep the measured 30 g under a pinned total; the structure
    # takes the rest in its component proportions (CHANGED 2026-09-10).
    pinned_xml = G.generate_xml(link_mass_kg=links, bracket_mass_kg=brackets, **_KW)
    free_xml = G.generate_xml(link_mass_kg=None, bracket_mass_kg=None, **_KW)
    for xml in (pinned_xml, free_xml):
        assert _geom_mass(xml, "canarm_s1_a1_sleeve") == pytest.approx(
            G.DEFAULT_ACTUATOR_MASS_KG, rel=1e-12)
    assert _geom_mass(pinned_xml, "canarm_seg1_rod") == pytest.approx(
        G.DEFAULT_LINK_DENSITY_KG_M * geo[0].span * (links[0] - sleeves) / structure[0],
        rel=1e-9)

    with pytest.raises(ValueError):
        G.generate_xml(link_mass_kg=(0.5, 0.5), **_KW)
    with pytest.raises(ValueError):
        G.generate_xml(bracket_mass_kg=-0.1, **_KW)
    with pytest.raises(ValueError, match="sleeves"):
        G.generate_xml(link_mass_kg=(0.9, 0.24, 0.5), **_KW)


def test_joint_stiffness_lands_on_its_segments_four_hinges_and_defaults_to_none():
    """The passive-stiffness tunable added for the 2026-09-10 physical refit.

    Zero by default and then **absent from the document**, so every model built
    before the term existed is reproduced; per segment otherwise, on that
    segment's proximal and distal hinges and no others, with the spring at rest
    at ``q = 0``.  Negative and ``None`` are refused.
    """
    base = G.build_model(**_KW)
    assert np.all(base.jnt_stiffness == 0.0)
    assert "stiffness=" not in G.generate_xml(**_KW).split("<worldbody>")[1]

    k = (0.4, 1.5, 2.25)
    m = G.build_model(joint_stiffness=k, **_KW)
    names = G.joint_names()
    for i, name in enumerate(names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert m.jnt_stiffness[jid] == pytest.approx(k[i // 4], rel=1e-12), name
        assert m.qpos_spring[m.jnt_qposadr[jid]] == 0.0
    one = G.build_model(joint_stiffness=0.7, **_KW)
    assert np.allclose(one.jnt_stiffness, 0.7)
    assert "joint_stiffness=(0.4, 1.5, 2.25)" in G.generate_xml(joint_stiffness=k, **_KW)

    for bad in (-0.1, (0.1, 0.2), None, (0.1, float("nan"), 0.2)):
        with pytest.raises(ValueError):
            G.generate_xml(joint_stiffness=bad, **_KW)

    # It is a restoring torque: a hinge displaced with no gravity comes back.
    data = mujoco.MjData(m)
    data.qpos[:] = 0.1
    mujoco.mj_forward(m, data)
    assert np.all(data.qfrc_passive < 0.0)


def test_mass_and_drawing_tunables_reach_the_model_through_simarm():
    """``SimArm(**mjcf_tunables)`` is the seam a fitted mass model arrives through."""
    from digital_twin import sim_core as sc

    arm = sc.SimArm(actuator=sc.placeholder_actuator(),
                    link_mass_kg=(0.8, 0.6, 0.45), bracket_mass_kg=0.25,
                    y_tip_radius=0.09)
    assert _body_mass(arm.model, "canarm_seg1_link") == pytest.approx(0.8, abs=1e-9)
    assert _body_mass(arm.model, "canarm_seg2_plate2") == pytest.approx(0.25, abs=1e-9)
    assert "y_tip_radius=0.09" in arm.xml


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
    """``CONTRACT.md`` section 3 names these; none may become a module constant.

    The per-segment mass totals and the three drawing tunables joined the list
    on 2026-09-10, because a fit drives the first and a figure estimated the
    second.
    """
    sig = inspect.signature(G.generate_xml).parameters
    for name in ("joint_damping", "joint_frictionloss", "tendon_damping",
                 "link_density", "plate_mass", "base_pos", "base_rpy_deg",
                 "link_mass_kg", "bracket_mass_kg", "y_tip_radius",
                 "y_tip_z_offset", "actuator_radius", "joint_stiffness"):
        assert name in sig, name
        assert sig[name].default is not inspect.Parameter.empty, name
        assert sig[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_base_pose_moves_the_whole_arm():
    shifted = G.build_model(base_pos=(0.3, -0.4, 1.1), base_rpy_deg=(0, 0, 0))
    data = mujoco.MjData(shifted)
    mujoco.mj_forward(shifted, data)
    assert np.allclose(data.site("canarm_plate0").xpos, (0.3, -0.4, 1.1))


# ---------------------------------------------------------------------------
# 7. Seams and guards
# ---------------------------------------------------------------------------

def test_build_model_accepts_a_handed_in_scene():
    """The ``xml=`` seam ``sim_core`` and a merged room scene both go through."""
    xml = G.generate_xml(**_KW)
    handed = G.build_model(xml=xml)
    assert handed.nu == 24 and handed.opt.timestep == 0.001


def test_generated_xml_names_the_parameters_it_was_built_with():
    xml = G.generate_xml(joint_damping=0.031, actuator_mass=0.042,
                         link_mass_kg=(0.7, 0.5, 0.45), **_KW)
    assert "joint_damping=0.031" in xml
    assert "actuator_mass=0.042" in xml
    assert "link_mass_kg=(0.7, 0.5, 0.45)" in xml
    assert "proximal_order=yx" in xml
    assert "routing=promax_bearing" in xml


def test_generate_scene_writes_a_file(tmp_path):
    out = G.generate_scene(tmp_path / "twin.xml", **_KW)
    assert out.exists()
    mujoco.MjModel.from_xml_string(out.read_text(encoding="utf-8"))


def test_tendon_rest_lengths_are_the_rigid_span_plus_the_bearing_span(model, seats):
    """Rest lengths are the rigid sleeve span plus the closed-form bearing span.

    CHANGED 2026-09-10.  This used to assert ``lengths.std() > 1e-3`` so that a
    model which lost a ring would show up.  Under the ProMax bearing routing the
    eight muscles of a segment are symmetric by construction and the segments
    differ only through ``LL``, so the 24 lengths span 1.3 mm and a spread test
    would be testing ``LL``.  Which hub and which body each tendon routes through
    -- the ring distinction -- is pinned by
    ``test_each_tendon_runs_actuator_end_then_bearing_then_bracket``; what is
    asserted here is the length itself.
    """
    lengths = G.tendon_rest_lengths(model)
    assert lengths.shape == (24,)
    assert np.all(np.isfinite(lengths)) and np.all(lengths > 0.05)
    for k, s in enumerate(seats):
        _g, aa, ja, ao = _ring_numbers(s)
        n = G.routing_site_names(s)
        end = model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, n["end"])]
        brg = model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, n["brg"])]
        assert lengths[k] == pytest.approx(
            float(np.linalg.norm(brg - end)) + math.hypot(ja - ao, aa), abs=_XML_TOL_M)


def test_fitted_params_is_a_writable_copy():
    a, b = G.fitted_params(), G.fitted_params()
    a[0, 0] = 99.0
    assert b[0, 0] != 99.0
    assert a.shape == (3, 10)


def test_params_from_chain_reproduces_the_measured_table():
    """The display draws from the chain; for the measured chain it is this table."""
    from UMArm_KINEMATICS import canarm_params as cp

    assert np.allclose(G.params_from_chain(cp.CANARM_PLATE_CHAIN_M),
                       cp.CANARM_PARAMS, atol=1e-15)


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
