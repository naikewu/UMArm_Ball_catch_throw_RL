"""Kinova Gen3 forward kinematics, damped-least-squares IK and reaches.

PROVENANCE: ported from ``C:/RUNZE_SRC/UMArm_maxi_collab/sim/kinova_control.py``
(read-only example).  The DLS core, the seed ladder, the collision re-seed and
the quintic reach are that file's; what is new here is:

* **IK sees the world where it actually is.**  The example zeroed the whole
  ``qpos`` on its scratch data, so its collision check tested the Gen3 against a
  UMArm hanging at rest no matter what the UMArm was doing.  Here the scratch
  data inherits the live ``qpos`` AND the live ``mocap_pos``/``mocap_quat``, and
  only the seven Gen3 addresses are overwritten.  The mocap half is not
  optional in this scene: both robots hang off mocap bodies, and a fresh
  ``MjData`` initialises those from the model's compile-time poses — so without
  it, every nudge after the operator moved a base was solved in a world where
  neither robot had moved.
* **Full-pose IK** alongside the example's position + face-normal form, because
  the GUI's task-space nudges have to hold orientation while they move.
* **Task-space nudges** (:meth:`KinovaArm.nudge`), the operator-facing verb: a
  step in world x/y/z or a turn about a world axis, resolved to joint angles.

On the real rig this module's callers become kortex high-level actions —
``reach_joint_angles`` for :class:`KinovaReach`, Cartesian ``reach_pose`` for
:func:`solve_pose_ik` — so the seam is kept narrow on purpose: everything here
takes joint angles in and gives joint angles out, and nothing but
:class:`KinovaReach.update` ever writes ``data.ctrl``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

#: The seven Gen3 joints, in chain order.
JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))

#: Mechanical limits, radians.  Joints 1/3/5/7 are continuous on the Gen3; the
#: even ones are limited.  Keyed by 0-based index into :data:`JOINT_NAMES`.
JOINT_LIMITS = {1: (-2.24, 2.24), 3: (-2.57, 2.57), 5: (-2.09, 2.09)}

#: Actuator ctrlranges the MJCF pins for the limited joints (slightly wider
#: than the mechanical range, as mujoco_menagerie ships them).
CTRL_LIMITS = {1: (-2.2497294058206907, 2.2497294058206907),
               3: (-2.5795966344476193, 2.5795966344476193),
               5: (-2.0996310901491784, 2.0996310901491784)}

#: The example's ``demo_start`` keyframe — the Gen3's own retract pose, folded
#: back.  Kept because it is the manufacturer's named pose, but NOT what this
#: scene starts in: ``joint_4`` sits 0.022 rad from its -2.57 stop there, so
#: half the task-space nudges an operator tries from it are genuinely
#: unreachable while holding the pad's orientation.  Starting a bench on a
#: joint stop is a bad first minute.
RETRACT_Q = np.array([0.0, -0.34906585, 3.14159265, -2.54818071,
                      0.0, -0.87266463, 1.57079633], dtype=float)

#: **The pose the scene starts in**: elbow forward, pad presented toward the
#: UMArm, every joint comfortably mid-range.  It is the example's first IK seed,
#: which was chosen for exactly that property.
#:
#: ``joint_7`` IS -pi/2, NOT the example's +pi/2, and the half turn is the
#: camera-up request applied to the parking pose.  The wrist roll moves the pad
#: 0.45 mm and nothing else -- ``pad_center`` is on the tool axis -- while it
#: takes the vision module from 104.8 deg off vertical (pointing below the
#: horizon) to 75.2 deg (above it).  75 deg is as good as this pose gets: the
#: pad normal at home is 14.8 deg off vertical, and the camera sweeps a cone
#: about that normal, so the best any roll can do is 90 - 14.8.
HOME_Q = np.array([0.0, 0.6, np.pi, -1.8, 0.0, -1.0, -np.pi / 2], dtype=float)

#: IK seeds, tried in order.  Forward-facing postures first: joint_1 near zero
#: puts the elbow over the front of the base, toward the UMArm.  All five carry
#: HOME_Q's wrist roll, so the ladder starts from a camera-up posture and the
#: six-DOF stage of :func:`solve_normal_ik` has a short way to travel.
IK_SEEDS = (
    np.array([0.0, 0.6, np.pi, -1.8, 0.0, -1.0, -np.pi / 2]),
    np.array([0.0, 0.9, np.pi, -1.4, 0.0, -0.6, -np.pi / 2]),
    np.array([0.0, 0.3, np.pi, -2.0, 0.0, -1.2, -np.pi / 2]),
    np.array([0.6, 0.7, np.pi, -1.6, 0.0, -0.9, -np.pi / 2]),
    np.array([-0.6, 0.7, np.pi, -1.6, 0.0, -0.9, -np.pi / 2]),
)

#: Name of the site the IK drives by default: the pole of the contact dome.
TOOL_SITE = "pad_center"

# ---------------------------------------------------------------------------
# Which way up the end effector is carried
# ---------------------------------------------------------------------------
#
# The operator asked for the Gen3's integrated wrist camera to point generally
# up while the bench runs, so the printed end effector presents the same face to
# the motion-capture cameras every trial: "it does not need to align perfectly,
# but it has to be in the ballpark."
#
# There are two things that could be called "where the camera points", and only
# one of them is free.  MEASURED off ``bracelet_with_vision_link.stl``: the
# vision module is a boss on the bracelet's **-Y** side (its centroid sits at
# (0.0005, -0.0587, -0.0496) m, i.e. the unit direction off the tool axis is
# (0.009, -1.000, 0.000)), and its lens faces are the front plane at
# z = -0.0644, so the cameras LOOK along the bracelet's -Z -- which is the tool
# axis, which is wherever the pad is pointing.  During a horizontal strike that
# is horizontal by definition and no choice of joint angles can raise it.
#
# What IS free is the roll about that axis, and it is exactly what decides which
# side of the wrist the module sits on.  That is the readable request, it is the
# one that matters for mocap, and it is satisfiable: the free spin of the
# five-DOF pad IK is precisely this degree of freedom.

#: Direction from the tool axis to the vision module, in the TOOL SITE's frame.
#: ``pad_mount`` turns the pad 180 deg about the tool z and ``collision_pad_face``
#: turns it 180 deg about x again, so the bracelet's -Y arrives here as -Y as
#: well; ``test_kinova_kinematics`` re-derives it from the compiled scene rather
#: than trusting that chain of two flips.
CAMERA_DIR_TOOL = np.array([0.0, -1.0, 0.0])

#: How far the camera may be off vertical before ``camera_up_warning`` says so.
#: "In the ballpark" -- a marker plate is not a lens, and the mocap volume does
#: not care about 30 deg.
CAMERA_UP_WARN_DEG = 45.0


def camera_dir_world(model, data, site: str = TOOL_SITE) -> np.ndarray:
    """Unit world direction from the tool axis to the Gen3's vision module."""
    _, rot = site_pose(model, data, site)
    return rot @ CAMERA_DIR_TOOL


def camera_up_deg(rot) -> float:
    """Angle between the camera direction and world +z, degrees, given ``rot``.

    *rot* is the tool site's 3x3 world rotation.  0 means the module points
    straight up; 180 means it is hanging underneath.
    """
    c = np.asarray(rot, dtype=float).reshape(3, 3) @ CAMERA_DIR_TOOL
    c = c / max(float(np.linalg.norm(c)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(c[2], -1.0, 1.0))))


def camera_up_rotation(target_normal) -> np.ndarray:
    """The tool rotation that puts the pad on *target_normal*, camera as up as
    it can be.

    The five-DOF pad task fixes the tool's +z and leaves the spin about it free,
    so the camera direction sweeps a cone about the normal and the best it can
    do is the point of that cone nearest vertical.  With the camera fixed in the
    tool frame as ``c = a*z + b*u`` the achievable maximum is
    ``a*(n.zhat) + b*|zhat - (zhat.n) n|``, reached when ``u`` is aligned with
    the projection of ``zhat`` onto the plane perpendicular to ``n`` -- and for
    this end effector ``a = 0``, so a HORIZONTAL pad normal lets the camera
    point exactly up and a vertical one cannot do better than horizontal.

    Returns a proper rotation matrix; the degenerate case (a normal within a
    thousandth of vertical, where the projection vanishes) falls back to any
    perpendicular, because at that point every spin is equally bad.
    """
    n = np.asarray(target_normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    u = z - float(z @ n) * n
    if float(np.linalg.norm(u)) < 1e-3:
        u = np.cross(n, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(u)) < 1e-6:
            u = np.cross(n, np.array([0.0, 1.0, 0.0]))
    u = u / float(np.linalg.norm(u))
    # camera = R @ (0, -1, 0) = -R[:, 1], and we want that to be u.
    col1 = -u
    col0 = np.cross(col1, n)
    return np.column_stack([col0, col1, n])


def _spin_error(rot, n_des) -> np.ndarray:
    """Rotation-vector error about *n_des* that turns the camera toward up.

    Only the component ABOUT the pad normal, because the other two rotational
    degrees of freedom belong to the normal task itself and must not be fought
    over.  Added to the five-DOF residual with a small weight, it biases the
    null space; it is never part of the convergence test.
    """
    n = np.asarray(n_des, dtype=float)
    des = camera_up_rotation(n)
    c_cur = np.asarray(rot, dtype=float) @ CAMERA_DIR_TOOL
    c_des = des @ CAMERA_DIR_TOOL
    a = c_cur - float(c_cur @ n) * n
    b = c_des - float(c_des @ n) * n
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return np.zeros(3)
    a, b = a / na, b / nb
    ang = math.atan2(float(n @ np.cross(a, b)), float(a @ b))
    return n * ang

#: Root of the Gen3 subtree, for the collision check.
KINOVA_ROOT_BODY = "kinova_base"


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------

def joint_addresses(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(qpos addresses, dof addresses, actuator ids)`` for the seven joints."""
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES]
    if any(j < 0 for j in jids):
        raise ValueError("model has no Kinova joints — wrong scene?")
    qadr = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
    vadr = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)
    aids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                     for n in JOINT_NAMES], dtype=int)
    if np.any(aids < 0):
        raise ValueError("model has no Kinova position actuators")
    return qadr, vadr, aids


def subtree_geoms(model, root_body: str = KINOVA_ROOT_BODY) -> set[int]:
    """Every geom id under *root_body*, pad and gauge included."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root < 0:
        return set()
    in_tree = set()
    for bid in range(model.nbody):
        b = bid
        while b != 0:
            if b == root:
                in_tree.add(bid)
                break
            b = model.body_parentid[b]
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in in_tree}


def cross_robot_geoms(model, kinova_geoms=None) -> set[int]:
    """The non-Gen3 geoms the scene deliberately lets the pad touch.

    Derived from the model's own ``<pair>`` list rather than from a hard-coded
    name list: ``collab_scene._add_contacts`` pairs each of the pad's colliders
    against each UMArm strike target, so any pair with exactly one end inside
    the Gen3 subtree names, at its other end, a thing the pad is *supposed* to
    hit.  Adding a strike target to the scene therefore exempts it here with no
    second edit.

    Contacts between the Gen3 and one of these are not collisions to be refused
    (see :func:`_pose_collides`); self-collision, the floor and the stand are
    not in any ``<pair>`` and so are unaffected.
    """
    if kinova_geoms is None:
        kinova_geoms = subtree_geoms(model)
    out: set[int] = set()
    for pi in range(model.npair):
        g1 = int(model.pair_geom1[pi])
        g2 = int(model.pair_geom2[pi])
        in1, in2 = g1 in kinova_geoms, g2 in kinova_geoms
        if in1 != in2:
            out.add(g2 if in1 else g1)
    return out


#: How far inside a mechanical limit an IK *seed* is placed.  Only seeds: see
#: :func:`clamp_to_limits`.
SEED_MARGIN = 0.05


def clamp_to_limits(q, margin: float = 0.0) -> np.ndarray:
    """Clamp the limited joints inside their range, keeping *margin* radians.

    THE DEFAULT MARGIN IS ZERO, and that is a fix rather than a preference.  The
    Gen3's retract pose (:data:`RETRACT_Q`) parks ``joint_4`` at -2.5482 rad
    against a -2.57 limit, i.e. 0.022 rad from the stop.  The example's IK
    clamped every iterate 0.05 rad inside the range, so seeding from a live pose
    anywhere near a stop immediately *moved* the arm away from where it was, and
    then every DLS step that tried to come back was clipped.  The solver stalled
    15 mm short of a 30 mm nudge and reported failure, on a target that is
    trivially reachable.  (The scene now starts at :data:`HOME_Q` instead, which
    is mid-range everywhere — but a solver that only works away from the stops
    is a solver that will fail the first time the operator drives to one.)

    Iteration therefore uses the true mechanical limits.  The margin still
    exists for *seeds* (:data:`SEED_MARGIN`), where starting hard against a stop
    wastes the seed, and ``KinovaArm.set_command`` separately clamps commands to
    the actuators' own ``ctrlrange``.
    """
    q = np.asarray(q, dtype=float).copy()
    for idx, (lo, hi) in JOINT_LIMITS.items():
        q[idx] = float(np.clip(q[idx], lo + margin, hi - margin))
    return q


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def site_pose(model, data, site: str = TOOL_SITE) -> tuple[np.ndarray, np.ndarray]:
    """``(position, 3x3 rotation)`` of a site in world coordinates."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"no site named {site!r}")
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def _rot_error(r_cur, r_des) -> np.ndarray:
    """Rotation vector taking *r_cur* to *r_des* (small-angle safe)."""
    e = r_des @ r_cur.T
    v = np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]])
    s = np.linalg.norm(v)
    c = np.clip((np.trace(e) - 1.0) * 0.5, -1.0, 1.0)
    if s < 1e-9:
        # Either aligned, or 180 deg apart; the latter needs the axis from the
        # symmetric part rather than the (vanishing) antisymmetric one.
        if c > 0.0:
            return np.zeros(3)
        w, vec = np.linalg.eigh(e + np.eye(3))
        return np.pi * vec[:, int(np.argmax(w))]
    return v / s * float(np.arctan2(s * 0.5, c))


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

def _live_state(data, model):
    """The caller state a scratch ``MjData`` has to inherit to be comparable.

    ``qpos`` is the obvious half.  ``mocap_pos``/``mocap_quat`` are the half that
    was missed, and missing it is not subtle: BOTH robots in this scene hang off
    mocap bodies, and a fresh ``MjData`` initialises those from the model's
    COMPILE-TIME body poses.  So the moment the operator moved a base, IK was
    solving in a world where the Gen3 and the arm were still at the coordinates
    the XML was generated with — it would happily return joint angles that put
    the pad at the right place in the wrong world.
    """
    if data is None:
        return (np.zeros(model.nq), None, None)
    return (np.asarray(data.qpos).copy(),
            np.asarray(data.mocap_pos).copy() if model.nmocap else None,
            np.asarray(data.mocap_quat).copy() if model.nmocap else None)


def _restore(scratch, live) -> None:
    qpos, mpos, mquat = live
    scratch.qpos[:] = qpos
    if mpos is not None:
        scratch.mocap_pos[:] = mpos
        scratch.mocap_quat[:] = mquat


def _dls_solve(model, scratch, qadr, vadr, sid, q_seeds, residual_fn,
               kinova_geoms, iters, damp, check_collisions,
               live, exempt_geoms=frozenset()) -> np.ndarray | None:
    """The shared DLS loop.  *residual_fn(pos, rot) -> (residual6, converged)``."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for si, seed in enumerate(q_seeds):
        # Seed 0 is the caller's current pose and is taken as given — nudging it
        # off a legally-parked joint is what SEED_MARGIN must not do.  The
        # ladder seeds are hand-written and get the margin.
        q = clamp_to_limits(seed, 0.0 if si == 0 else SEED_MARGIN)
        for it in range(iters):
            _restore(scratch, live)
            scratch.qpos[qadr] = q
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)

            pos = scratch.site_xpos[sid].copy()
            rot = scratch.site_xmat[sid].reshape(3, 3).copy()
            res, converged = residual_fn(pos, rot)
            if converged:
                if check_collisions and _pose_collides(
                        model, scratch, qadr, q, kinova_geoms, live,
                        exempt_geoms):
                    break                       # valid but jammed: next seed
                return q

            mujoco.mj_jacSite(model, scratch, jacp, jacr, sid)
            j = np.vstack([jacp[:, vadr], jacr[:, vadr]])        # 6 x 7
            # The example's schedule: damp hard while the error is large, then
            # loosen for the last third so near-reach targets still polish in.
            lam = damp if it < 2 * iters // 3 else 0.015
            dq = j.T @ np.linalg.solve(j @ j.T + lam ** 2 * np.eye(6), res)
            q = clamp_to_limits(q + np.clip(dq, -0.2, 0.2))
    return None


def _pose_collides(model, scratch, qadr, q, kinova_geoms, live,
                   exempt_geoms=frozenset()) -> bool:
    """True if the Gen3 at *q* is interpenetrating anything it must not.

    IK is collision-blind; without this, it happily returns
    elbow-through-the-mast solutions that jam the position servos against their
    force limit and look, on screen, like a broken robot.

    *exempt_geoms* is what the operator asked for: TOUCHING THE UMARM IS THE
    JOB.  The threshold below is 10 um of penetration, so refusing every contact
    made pressing the pad against a cover — the whole point of the bench —
    indistinguishable from an unreachable target, and the GUI answered a nudge
    towards the arm with "move the base".  Contacts whose non-Gen3 end is one of
    these geoms (in practice
    :func:`cross_robot_geoms`, i.e. the scene's declared strike targets) are
    skipped.  Everything else still refuses: Gen3 self-collision, the floor and
    the Gen3's own stand are what this check was written for, and none of them
    appear in a cross-robot ``<pair>``.  The default is empty so a caller that
    forgets to pass it gets the strict old behaviour rather than a silent
    licence to drive through the arm.
    """
    _restore(scratch, live)
    scratch.qpos[qadr] = q
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    for ci in range(scratch.ncon):
        con = scratch.contact[ci]
        if con.dist >= -1e-5:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        in1, in2 = g1 in kinova_geoms, g2 in kinova_geoms
        if not (in1 or in2):
            continue
        # in1 == in2 means both ends are the Gen3's own: a self-collision, which
        # nothing exempts.
        if in1 != in2 and (g2 if in1 else g1) in exempt_geoms:
            continue
        return True
    return False


#: Weight on the camera-up bias inside the five-DOF solve.  Small on purpose:
#: it steers the null space and must never out-shout the pose task, and the
#: convergence test does not look at it at all, so a pose that cannot get the
#: camera up still converges — it just converges with the camera wherever the
#: seed left it.
CAMERA_SPIN_GAIN = 0.35


def solve_normal_ik(model, target_pos, target_normal, q_init, *,
                    site: str = TOOL_SITE, data=None, iters: int = 300,
                    damp: float = 0.05, pos_tol: float = 4e-3,
                    ang_tol: float = 0.03,
                    check_collisions: bool = True,
                    exempt_geoms=None,
                    camera_up: bool = True,
                    seeds=None) -> np.ndarray | None:
    """Place *site* at *target_pos* with its +z along *target_normal*.

    The example's ``solve_pad_ik``: five degrees of freedom constrained, the
    spin about the pad normal left free, which is right for a flat pad.  Runs on
    scratch state; the caller's ``data`` is never written.  Returns ``q(7)`` or
    ``None`` if no seed converged to a pose free of the collisions that matter —
    which does NOT include touching the UMArm; see :func:`_pose_collides`.

    DEVIATION FROM THE EXAMPLE: *q_init* is tried FIRST, not last.  The example
    placed the pad once per trial from a retract pose, so starting at a
    known-good posture was the better bet.  Here the same call also has to track
    — nudge the pad 10 mm, push it into the cover, follow a strike — and a
    solver that prefers a fixed seed ladder answers "move 10 mm" by flinging the
    elbow through a different solution branch.  Seeding from where the arm is
    keeps small moves small; when that fails to converge the ladder is still
    there, one wasted attempt later.

    *exempt_geoms* overrides the default exemption set (the scene's declared
    strike targets).  Pass ``frozenset()`` to keep collision checking but demand
    a pose that touches NOTHING — which is what the D6 withdrawal wants, since a
    re-home that ends inside the arm is not an operator asking for contact.

    *camera_up* spends the free spin about the pad normal on pointing the Gen3's
    vision module upward (see :data:`CAMERA_DIR_TOOL`), which is what the
    operator asked for so the printed end effector faces the mocap cameras the
    same way every trial.  It is done in two stages and NEITHER of them can cost
    a solution: the full six-DOF pose is tried first, with the camera exactly
    where :func:`camera_up_rotation` puts it, and if no seed reaches that the
    five-DOF solve runs as before with a small extra residual about the normal
    (:data:`CAMERA_SPIN_GAIN`) that biases the spin without entering the
    convergence test.  Pass ``camera_up=False`` for the old behaviour.
    """
    scratch = mujoco.MjData(model)
    live = _live_state(data, model)
    qadr, vadr, _ = joint_addresses(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"no site named {site!r}")
    geoms = subtree_geoms(model) if check_collisions else set()
    if not check_collisions:
        exempt = set()
    elif exempt_geoms is None:
        exempt = cross_robot_geoms(model, geoms)
    else:
        exempt = frozenset(int(g) for g in exempt_geoms)
    tgt = np.asarray(target_pos, dtype=float)
    n_des = np.asarray(target_normal, dtype=float)
    n_des = n_des / max(np.linalg.norm(n_des), 1e-12)

    if seeds is None:
        seeds = [np.asarray(q_init, dtype=float)] + list(IK_SEEDS)

    if camera_up:
        # Stage one: ask for the whole pose, camera included.  A six-DOF task on
        # a seven-DOF arm still has a null space, so this succeeds far more
        # often than it looks like it should; when it does, the camera is exactly
        # where it was asked for rather than merely biased toward it.
        # A TIGHTER POSITION TOLERANCE THAN THE FALLBACK'S, deliberately.  Both
        # branches "converge" anywhere inside their tolerance, and on a stiff
        # contact a millimetre of standoff is worth tens of newtons -- measured
        # at up to 92 N on one battery condition when the two branches landed at
        # opposite ends of a 4 mm window.  Asking stage one for a quarter of the
        # window means turning the camera up cannot move the pad by more than
        # the five-DOF answer already might.
        q = solve_pose_ik(model, tgt, camera_up_rotation(n_des), q_init,
                          site=site, data=data, iters=iters, damp=damp,
                          pos_tol=0.25 * pos_tol, ang_tol=ang_tol,
                          check_collisions=check_collisions,
                          exempt_geoms=exempt_geoms, seeds=list(seeds))
        if q is not None:
            return q

    def residual(pos, rot):
        z_cur = rot[:, 2]
        pos_err = tgt - pos
        rot_err = np.cross(z_cur, n_des)
        ok = (np.linalg.norm(pos_err) < pos_tol
              and np.arccos(np.clip(float(z_cur @ n_des), -1, 1)) < ang_tol)
        if camera_up:
            rot_err = rot_err + CAMERA_SPIN_GAIN * _spin_error(rot, n_des)
        return np.concatenate([pos_err, 0.5 * rot_err]), ok

    q = _dls_solve(model, scratch, qadr, vadr, sid, seeds, residual, geoms,
                   iters, damp, check_collisions, live, exempt)
    if q is not None or not camera_up:
        return q

    # STAGE THREE, and the reason the docstring can promise that the camera
    # preference never costs a solution.  Stage two's extra residual is small,
    # but it is still a term the solver is driving to zero, and near a joint
    # stop it can keep the pose error from settling inside the tolerance at all.
    # So a failure is retried with the preference off before it is reported as
    # "unreachable" -- measured: without this, the re-home withdrawal on
    # ``s0_tip_p30`` stopped having a solution.
    def plain(pos, rot):
        z_cur = rot[:, 2]
        pos_err = tgt - pos
        ok = (np.linalg.norm(pos_err) < pos_tol
              and np.arccos(np.clip(float(z_cur @ n_des), -1, 1)) < ang_tol)
        return np.concatenate([pos_err, 0.5 * np.cross(z_cur, n_des)]), ok

    return _dls_solve(model, scratch, qadr, vadr, sid, seeds, plain, geoms,
                      iters, damp, check_collisions, live, exempt)


def solve_pose_ik(model, target_pos, target_rot, q_init, *,
                  site: str = TOOL_SITE, data=None, iters: int = 200,
                  damp: float = 0.05, pos_tol: float = 2e-3,
                  ang_tol: float = 0.02, check_collisions: bool = True,
                  exempt_geoms=None, seeds=None) -> np.ndarray | None:
    """Full six-DOF IK: place *site* at *target_pos* with orientation *target_rot*.

    What the GUI's task-space nudges need — moving the pad 10 mm along world x
    must not let the wrist spin to get there.  Seeded from *q_init* first, since
    a nudge is by definition a small move from where the arm already is.  A step
    that pushes the pad into the UMArm is allowed: poking the arm is the bench's
    purpose, so only self-collision, the floor and the stand refuse.

    *exempt_geoms* is here for one caller: :func:`solve_normal_ik`'s camera-up
    stage, which must exempt exactly what the five-DOF fallback would exempt or
    the two stages would disagree about which poses exist.
    """
    scratch = mujoco.MjData(model)
    live = _live_state(data, model)
    qadr, vadr, _ = joint_addresses(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise ValueError(f"no site named {site!r}")
    geoms = subtree_geoms(model) if check_collisions else set()
    if not check_collisions:
        exempt = set()
    elif exempt_geoms is None:
        exempt = cross_robot_geoms(model, geoms)
    else:
        exempt = frozenset(int(g) for g in exempt_geoms)
    tgt = np.asarray(target_pos, dtype=float)
    r_des = np.asarray(target_rot, dtype=float).reshape(3, 3)

    def residual(pos, rot):
        pos_err = tgt - pos
        rot_err = _rot_error(rot, r_des)
        ok = (np.linalg.norm(pos_err) < pos_tol
              and np.linalg.norm(rot_err) < ang_tol)
        return np.concatenate([pos_err, rot_err]), ok

    if seeds is None:
        seeds = [np.asarray(q_init, dtype=float)] + list(IK_SEEDS)
    return _dls_solve(model, scratch, qadr, vadr, sid, seeds, residual, geoms,
                      iters, damp, check_collisions, live, exempt)


# ---------------------------------------------------------------------------
# Base-relative kinematics - what the real rig actually knows
# ---------------------------------------------------------------------------
#
# ON THE RIG THE BASE POSE IS NOT MEASURED.  Nothing is stuck to the Gen3's
# plinth; the cameras see four markers glued into the collision pad, i.e. the
# END of the chain.  So the known quantities are (a) the pad's pose in the mocap
# world and (b) the joint angles the robot reports, and the base is *derived*:
#
#     T_world_base  =  T_world_pad  @  inv( T_base_pad(q) )
#
# Everything below exists to make that one line true and testable.  The second
# factor is pure forward kinematics of the robot's own chain and cannot depend
# on where the base is, which is exactly why :func:`fk_in_base` divides the base
# transform out rather than reading world coordinates.


def _scratch_from(model, data):
    """A scratch ``MjData`` carrying the caller's world (qpos + both mounts)."""
    scratch = mujoco.MjData(model)
    if data is not None:
        scratch.qpos[:] = np.asarray(data.qpos)
        if model.nmocap:
            scratch.mocap_pos[:] = np.asarray(data.mocap_pos)
            scratch.mocap_quat[:] = np.asarray(data.mocap_quat)
    return scratch


def fk_in_base(model, q, site: str = TOOL_SITE, data=None,
               scratch=None) -> tuple:
    """``(position, 3x3 rotation)`` of *site* **in the Gen3's base frame**.

    Independent of where the base is, by construction: the base's own world
    transform is divided out.  That is what makes this the same quantity the
    real robot's kortex forward kinematics returns, and what lets the base be
    solved for rather than assumed.
    """
    if scratch is None:
        scratch = _scratch_from(model, data)
    qadr, _, _ = joint_addresses(model)
    scratch.qpos[qadr] = np.asarray(q, dtype=float)
    mujoco.mj_kinematics(model, scratch)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, KINOVA_ROOT_BODY)
    if sid < 0 or bid < 0:
        raise ValueError("model is missing the Kinova base or the tool site")
    r_base = scratch.xmat[bid].reshape(3, 3)
    return (r_base.T @ (scratch.site_xpos[sid] - scratch.xpos[bid]),
            r_base.T @ scratch.site_xmat[sid].reshape(3, 3))


def base_from_tool(model, q, tool_pos_world, tool_rot_world,
                   site: str = TOOL_SITE, data=None, scratch=None) -> tuple:
    """Where the Gen3's base must be, given a measured tool pose and *q*.

    The back-propagation the rig runs on every mocap frame:
    ``T_world_base = T_world_tool @ inv(T_base_tool(q))``.  Returns
    ``(position, MuJoCo scalar-first quaternion)``, ready for a mocap mount.
    """
    p_bt, r_bt = fk_in_base(model, q, site=site, data=data, scratch=scratch)
    r_wt = np.asarray(tool_rot_world, dtype=float).reshape(3, 3)
    r_wb = r_wt @ r_bt.T
    p_wb = np.asarray(tool_pos_world, dtype=float) - r_wb @ p_bt
    return p_wb, mat_to_quat(r_wb)


def mat_to_quat(m) -> np.ndarray:
    """Rotation matrix -> MuJoCo scalar-first quaternion (Shepperd's branches)."""
    m = np.asarray(m, dtype=float)
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=float)
    return q / np.linalg.norm(q)


#: Prefix of the four marker sites the scene draws on the pad.
MARKER_SITE_PREFIX = "pad_marker_"
#: The pad body's origin as a site - the frame the marker coordinates live in.
PAD_FRAME_SITE = "pad_frame"


def marker_points_world(model, data, prefix: str = MARKER_SITE_PREFIX,
                        n: int = 4) -> np.ndarray:
    """The modelled marker centres in world coordinates, ``(n, 3)``.

    In simulation this stands in for what the cameras report; the order matches
    :func:`UMArm_COLLAB.mount_transforms.marker_points_pad_mm`, and the order is
    the correspondence the pose fit depends on.
    """
    out = []
    for i in range(n):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{i}")
        if sid < 0:
            raise ValueError(f"no site {prefix}{i} - wrong scene?")
        out.append(data.site_xpos[sid].copy())
    return np.asarray(out)


# ---------------------------------------------------------------------------
# The operator-facing verbs
# ---------------------------------------------------------------------------

class KinovaArm:
    """Read/write handle on the Gen3 inside a compiled scene.

    Holds no physics state of its own: every method takes the live ``data`` (or
    writes ``data.ctrl``), so this object is safe to build once and keep.  It is
    NOT thread-safe on its own — callers hold :attr:`UMArm_SIM.sim_core.SimArm.lock`
    exactly as they would around any other read of the shared model.
    """

    def __init__(self, model, site: str = TOOL_SITE):
        self.model = model
        self.site = site
        self.qadr, self.vadr, self.act_ids = joint_addresses(model)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.sid < 0:
            raise ValueError(f"no site named {site!r}")
        #: Reused rather than reallocated: the commanded-pose FK below runs on
        #: every nudge, and the base solve runs on every mocap frame.
        self._scratch = mujoco.MjData(model)

    # -- state ---------------------------------------------------------
    def q(self, data) -> np.ndarray:
        return np.asarray(data.qpos)[self.qadr].copy()

    def command(self, data) -> np.ndarray:
        return np.asarray(data.ctrl)[self.act_ids].copy()

    def set_command(self, data, q) -> None:
        """Write the seven position-servo targets, clamped to their ctrlrange."""
        q = np.asarray(q, dtype=float).copy()
        for idx, (lo, hi) in CTRL_LIMITS.items():
            q[idx] = float(np.clip(q[idx], lo, hi))
        data.ctrl[self.act_ids] = q

    def hold(self, data) -> None:
        """Freeze the servos wherever the joints currently are."""
        data.ctrl[self.act_ids] = np.asarray(data.qpos)[self.qadr]

    def tool_pose(self, data) -> tuple[np.ndarray, np.ndarray]:
        """Where the pad face IS - the measured pose."""
        return site_pose(self.model, data, self.site)

    def commanded_tool_pose(self, data) -> tuple[np.ndarray, np.ndarray]:
        """Where the pad face is being ASKED to be: FK of the servo targets.

        THIS, NOT :meth:`tool_pose`, IS WHAT A NUDGE STEPS FROM.  A position
        servo settles a little short of its target, so a nudge computed from the
        MEASURED pose bakes that shortfall into the new command and asks for it
        again on the next click.  Measured on this scene with the Gen3's gravity
        compensation off, four clicks of "+X" walked the pad 22 mm DOWNWARDS
        while the operator was only ever asking for horizontal motion - the arm
        drooped 4 mm under the 0.6 kg pad, the nudge measured the drooped pose,
        and the droop was re-added every time.  Stepping from the command makes
        repeated nudges accumulate exactly what was asked, and turns any
        steady-state tracking error into a constant offset instead of a ratchet.
        """
        s = self._scratch
        s.qpos[:] = np.asarray(data.qpos)
        if self.model.nmocap:
            s.mocap_pos[:] = np.asarray(data.mocap_pos)
            s.mocap_quat[:] = np.asarray(data.mocap_quat)
        s.qpos[self.qadr] = np.asarray(data.ctrl)[self.act_ids]
        mujoco.mj_kinematics(self.model, s)
        return (s.site_xpos[self.sid].copy(),
                s.site_xmat[self.sid].reshape(3, 3).copy())

    def fk_in_base(self, q, data=None) -> tuple[np.ndarray, np.ndarray]:
        """The pad face in the Gen3's own base frame - see :func:`fk_in_base`."""
        return fk_in_base(self.model, q, site=self.site, data=data,
                          scratch=self._scratch)

    # -- task space ----------------------------------------------------
    def nudge(self, data, d_xyz=(0.0, 0.0, 0.0), d_rpy_deg=(0.0, 0.0, 0.0),
              **ik_kw) -> np.ndarray | None:
        """Step the tool by *d_xyz* metres and *d_rpy_deg* degrees, in WORLD axes.

        Returns the new joint vector, or ``None`` when the step is unreachable
        or would collide — in which case the caller must leave the command
        alone, which is what the GUI does (the button does nothing and says so).

        A REFUSAL HERE IS REAL, and relaxing the task does not rescue it.  The
        obvious fallback — give up the spin about the flat pad's own normal and
        re-solve the same step as a five-DOF task — was implemented, measured and
        removed: over a 26-direction sweep from four postures it converted
        exactly zero refusals.  Near a stop the binding constraint is a *joint*
        (the retract pose sits 0.022 rad from ``joint_4``'s limit and some
        diagonals need it to go further), and freeing the wrist's roll does not
        unlock a shoulder.  So the honest answer to a refused nudge is to move
        the base, not to quietly do something else and call it the same move.

        With the scene's :data:`HOME_Q` start pose all 26 directions are reachable;
        the refusals only appear once the arm has been driven onto a stop.

        The step is taken from the COMMANDED pose, not the measured one - see
        :meth:`commanded_tool_pose` for the 22 mm of downward drift that cost.
        """
        pos, rot = self.commanded_tool_pose(data)
        d = np.deg2rad(np.asarray(d_rpy_deg, dtype=float))
        rx, ry, rz = (_axis_rot(i, d[i]) for i in range(3))
        target_rot = (rz @ ry @ rx) @ rot
        target_pos = pos + np.asarray(d_xyz, dtype=float)
        # ...and seeded from the commanded joint angles for the same reason: a
        # seed at the measured pose pulls the solution back toward the droop.
        return solve_pose_ik(self.model, target_pos, target_rot,
                             self.command(data), site=self.site, data=data,
                             **ik_kw)

    def place_pad(self, data, target_pos, target_normal, **ik_kw):
        """The protocol's verb: put the pad face at a point, facing a direction."""
        return solve_normal_ik(self.model, target_pos, target_normal,
                               self.q(data), site=self.site, data=data, **ik_kw)


def _axis_rot(axis: int, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    m = np.eye(3)
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    m[a, a] = c
    m[a, b] = -s
    m[b, a] = s
    m[b, b] = c
    return m


class KinovaReach:
    """A quintic joint-space reach on the seven position servos.

    The sim stand-in for kortex ``reach_joint_angles``: zero velocity and zero
    acceleration at both ends, so the pad is placed without shaking the cart.
    Driven off ``data.time`` rather than wall time, so it behaves identically
    under a paused or fast-forwarded sim.

    A reach may carry WAYPOINTS -- see :meth:`start` -- and then it is a chain
    of quintics that comes to rest at each of them.
    """

    def __init__(self, model):
        self.model = model
        self.qadr, self.vadr, self.act_ids = joint_addresses(model)
        self._q0: np.ndarray | None = None
        self._q1: np.ndarray | None = None
        #: ``(K+1, 7)`` -- the start pose followed by every waypoint -- and the
        #: seconds allotted to each of the K segments.  ``None`` when idle.
        self._pts: np.ndarray | None = None
        self._seg: np.ndarray | None = None
        self._t0 = 0.0
        self._dur = 1.0

    @property
    def active(self) -> bool:
        return self._pts is not None

    def start(self, data, q_target, duration_s: float) -> None:
        """Begin a reach to a pose, or along a path.

        *q_target* is either a ``(7,)`` pose -- one quintic, the behaviour this
        class has always had -- or a ``(K, 7)`` array of waypoints, in which
        case *duration_s* is split between the K segments IN PROPORTION TO
        SUMMED JOINT TRAVEL and each segment is its own quintic.

        THE GEN3 COMES TO REST AT EVERY WAYPOINT, and that is a choice against
        the obvious alternative of blending through the via without stopping.
        A blend is smoother and quicker, but the path it traces is no longer
        the piecewise quintic that :func:`kinova_path.plan_leg` swept for
        clearance -- it cuts the corner by an amount that depends on the
        approach speed, which is exactly the geometry the check was run on.
        Stopping at the via is what makes the swept check and the commanded
        motion the same object.  The cost is measured and small: a via adds one
        acceleration and one deceleration inside a window whose length does not
        change, so a 2.5 s reach with one via runs two 1.25 s-ish quintics and
        the peak joint rate rises by the ratio of the segments' travels.

        Splitting BY TRAVEL rather than evenly is what stops a 5 cm stand-off
        via at the end of a 400-degree traverse from being given half the
        window: each segment gets the time its own motion needs.
        """
        q = np.asarray(q_target, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        self._q0 = np.asarray(data.qpos)[self.qadr].copy()
        self._q1 = q[-1].copy()
        self._pts = np.vstack([self._q0[None, :], q])
        travel = np.abs(np.diff(self._pts, axis=0)).sum(axis=1)
        if float(travel.sum()) <= 1e-12:
            travel = np.ones(len(travel))
        self._dur = max(float(duration_s), 1e-3)
        self._seg = self._dur * travel / float(travel.sum())
        self._t0 = float(data.time)

    def cancel(self) -> None:
        self._q1 = None
        self._pts = None
        self._seg = None

    def update(self, data) -> bool:
        """Stream the servo targets.  True while the reach is still running."""
        if self._pts is None:
            return False
        t = float(data.time) - self._t0
        acc = 0.0
        for i, span in enumerate(self._seg):
            if t < acc + span:
                s = max(0.0, (t - acc) / max(float(span), 1e-9))
                alpha = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
                a, b = self._pts[i], self._pts[i + 1]
                data.ctrl[self.act_ids] = a + alpha * (b - a)
                return True
            acc += float(span)
        data.ctrl[self.act_ids] = self._pts[-1]
        self._pts = None
        self._seg = None
        self._q1 = None
        return False


__all__ = [
    "JOINT_NAMES", "JOINT_LIMITS", "CTRL_LIMITS", "HOME_Q", "IK_SEEDS",
    "TOOL_SITE", "KINOVA_ROOT_BODY", "RETRACT_Q", "SEED_MARGIN",
    "joint_addresses", "subtree_geoms", "cross_robot_geoms",
    "clamp_to_limits", "site_pose",
    "solve_normal_ik", "solve_pose_ik", "KinovaArm", "KinovaReach",
    "fk_in_base", "base_from_tool", "mat_to_quat", "marker_points_world",
    "MARKER_SITE_PREFIX", "PAD_FRAME_SITE",
]
