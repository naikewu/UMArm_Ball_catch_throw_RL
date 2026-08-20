"""Where each robot's base pose comes from — sliders today, cameras on the rig.

PROVENANCE: ported from ``C:\\RUNZE_SRC\\RS485_VEMA\\UMArm_COLLAB\\base_poses.py``
(``BasePose``, ``BasePoseSource``, ``ManualBasePoses``, ``MocapBasePoses``,
``MountWriter``).  The one structural change is the one the original could not
avoid: its ``poses()`` returns a fixed ``(umarm, kinova)`` 2-tuple, and this room
holds three robots of which two are optional.  Here ``poses()`` returns a **map
keyed by robot name**, and a source that knows nothing about a robot simply omits
it.  The collision-bench source ``KinovaBaseFromEE`` is deliberately not ported:
it solves a Gen3 base from markers on a printed strike pad that does not exist in
this workspace.

Every robot hangs off a MuJoCo **mocap body** rather than off baked-in
``pos``/``quat`` attributes.  That is not a rendering convenience.  A mocap body
is literally the thing MuJoCo provides for "a pose an external tracker writes",
it accepts a full six degrees of freedom (an MJCF generator can only bake
position and yaw), and writing it costs one array assignment per frame.  So the
same scene and the same viewer serve a simulated room and a measured one; only
which :class:`BasePoseSource` is plugged in changes.

TWO CORRECTIONS SIT BETWEEN A MARKER PLATE AND A ROBOT BASE, and both are here
because both are measured, not derived:

* :data:`BASE_CORRECTION` per robot — the mechanical base frame expressed in the
  base plate's *marker* frame.  On the RS485 arm the preflight fits it
  (``UMArm_ROBOT_CONTROL/preflight.py:690``, ``fit_base_correction``) and the
  measured numbers are a 1.78 deg plate-plane tilt and marker arms 16.1 deg off
  their designed azimuth; ignoring it grows from 1.1 mm at the base to 9.4 mm at
  the tip, absorbing it leaves 1.5 mm.  **Nothing has measured the CAN arm's**,
  so it ships as the identity and it is a knob, not a constant.
* The Gen3's is the same idea and has never been measured on either workspace
  (the RS485 file says so at its lines 29-35).  Any claim about where the Gen3's
  base is in the room frame inherits that unmeasured gap.

STALENESS IS NOT COSMETIC.  A plate Motive loses keeps its previous row forever —
``MocapRx`` says so — so a base pose that has stopped updating looks exactly like
a base that is not moving.  :meth:`MocapBasePoses.poses` therefore reports
``fresh=False`` and hands back the fallback rather than silently handing back
history, and a consumer freezes the mount and says so instead of drawing a robot
at a base pose from a minute ago.

``state.stale``, NOT ``state.q_stale``, IS WHAT A BASE POSE CHECKS.  This is the
opposite of the rule a control loop follows, and it is deliberate in both
directions: a base pose is a rigid-body row and keeps updating on frames where
the ARM's joint solve fails, whereas ``q`` on such a frame is silently older
every tick.  Carried over exactly as written at ``base_poses.py:185-186``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:                                    # both import styles, as elsewhere here
    from . import transforms as TF
except ImportError:                     # pragma: no cover
    import transforms as TF             # type: ignore[no-redef]

__all__ = [
    "MOUNT_BODY", "BASE_CORRECTION", "IDENTITY_POSE",
    "BasePose", "BasePoseSource", "ManualBasePoses", "MocapBasePoses",
    "MountWriter", "row_to_pose",
]

#: Name of each robot's mocap body in the merged scene.  ``mjcf_canarm`` emits
#: these; :class:`MountWriter` looks them up **by name** rather than assuming
#: mocap index 0/1/2, because MuJoCo assigns mocap ids in body-declaration order
#: — which the scene composer controls today and could quietly change tomorrow,
#: and which already varies with which optional robots are included.
MOUNT_BODY = {
    "canarm": "canarm_mount",
    "rs485": "rs485_mount",
    "kinova": "kinova_mount",
}

#: ``(4, 4)`` mechanical-base-in-marker-frame corrections.  All identity, all
#: unmeasured.  See the module docstring.
BASE_CORRECTION = {name: np.eye(4) for name in MOUNT_BODY}


@dataclass(frozen=True)
class BasePose:
    """One base: position (m), MuJoCo scalar-first quaternion, and its age."""

    pos: np.ndarray
    quat: np.ndarray
    #: False when the source could not vouch for this being a *current* reading.
    fresh: bool = True
    #: Human-readable reason when :attr:`fresh` is False.  It is a string rather
    #: than an enum because it ends up in front of an operator, and "mocap
    #: stale" and "Kinova row unusable (untracked)" are different problems with
    #: different fixes.
    note: str = ""

    @staticmethod
    def from_xyz_rpy(xyz, rpy_deg=(0.0, 0.0, 0.0), **kw) -> "BasePose":
        return BasePose(np.asarray(xyz, dtype=float),
                        TF.rpy_to_quat(rpy_deg), **kw)

    def as_matrix(self) -> np.ndarray:
        return TF.pose_matrix(self.pos, self.quat)

    def stale(self, note: str) -> "BasePose":
        """The same pose, disowned: same numbers, ``fresh=False``, a reason."""
        return BasePose(self.pos, self.quat, False, note)


#: What a source returns for a robot it has never had a reading for.  The origin
#: with an identity rotation, and flagged not fresh so nobody mistakes it for a
#: robot that happens to be parked at the world origin.
IDENTITY_POSE = BasePose(np.zeros(3), np.array(TF.IDENTITY_QUAT, dtype=float),
                         False, "no reading")


class BasePoseSource:
    """Interface: give me every base pose you know about, keyed by robot name.

    Deliberately tiny.  A consumer calls :meth:`poses` once per tick and writes
    whatever comes back into ``data.mocap_pos`` / ``data.mocap_quat``; it never
    learns whether the numbers came from a slider or a camera.

    Returning a **map** rather than a tuple is what lets a room drop a robot: a
    source that does not know about the Gen3 omits the key, and
    :meth:`MountWriter.apply` writes the mounts it was given and leaves the rest
    of the scene alone.
    """

    def poses(self) -> dict[str, BasePose]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ManualBasePoses(BasePoseSource):
    """Base poses the operator sets — the offline source, and the GUI's.

    Thread-safe by being trivially small: a setter replaces a whole tuple, so a
    reader either sees the old pose or the new one, never a half-written one.
    """

    def __init__(self, poses: dict | None = None, **kwargs):
        """*poses* maps robot name to ``(xyz, rpy_deg)`` or to just ``xyz``."""
        self._poses: dict[str, tuple] = {}
        for name, value in dict(poses or {}, **kwargs).items():
            self.set(name, *_split_xyz_rpy(value))

    def set(self, robot: str, xyz=None, rpy_deg=None) -> None:
        cur_xyz, cur_rpy = self._poses.get(robot, ((0.0, 0.0, 0.0),
                                                   (0.0, 0.0, 0.0)))
        self._poses[robot] = (
            tuple(float(v) for v in (cur_xyz if xyz is None else xyz)),
            tuple(float(v) for v in (cur_rpy if rpy_deg is None else rpy_deg)))

    def nudge(self, robot: str, d_xyz=(0, 0, 0), d_rpy_deg=(0, 0, 0)) -> None:
        xyz, rpy = self._poses.get(robot, ((0.0,) * 3, (0.0,) * 3))
        self.set(robot, np.add(xyz, d_xyz), np.add(rpy, d_rpy_deg))

    def get(self, robot: str) -> tuple:
        """``(xyz, rpy_deg)`` as last set — what a slider reads back."""
        return self._poses.get(robot, ((0.0,) * 3, (0.0,) * 3))

    def robots(self) -> tuple:
        return tuple(self._poses)

    def poses(self) -> dict[str, BasePose]:
        return {name: BasePose.from_xyz_rpy(xyz, rpy)
                for name, (xyz, rpy) in self._poses.items()}


class MocapBasePoses(BasePoseSource):
    """Base poses from the cameras.

    *rows* maps a robot name to ``(adapter, row_index)``: the adapter is
    anything carrying ``MocapRx``'s ``get_homos()`` and ``get_state()``, and the
    row index is which of its ``(9, 4, 4)`` rows carries that robot's base.  For
    either UMArm that is ``mocap_constants.IDX_BASE`` (0) of its own receiver;
    for the Gen3 it is ``KINOVA_RIGID_BODY_INDEX`` (8) of whichever receiver is
    listening, since ``MocapRx`` routes streaming id 1008 into row 8 regardless
    of which arm's block the receiver is bound to.

    Two adapters bound to different id blocks may appear here at once — that is
    exactly the arrangement a room holding a 2000-series CAN arm and a
    500-or-1000-series RS485 arm needs, and it is why the block base became a
    constructor argument on ``MocapRx`` rather than staying a module constant.

    *fallback* is used whenever a row cannot be trusted — before the first
    frame, or once the stream goes stale.  It is normally the
    :class:`ManualBasePoses` the caller was already using, so losing the cameras
    parks a robot where the operator last put it rather than at the origin.
    """

    def __init__(self, rows: dict, fallback: BasePoseSource | None = None, *,
                 corrections: dict | None = None):
        self.rows = dict(rows)
        self.fallback = fallback
        self.corrections = dict(BASE_CORRECTION)
        self.corrections.update(corrections or {})

    def poses(self) -> dict[str, BasePose]:
        fb = self.fallback.poses() if self.fallback is not None else {}
        out: dict[str, BasePose] = {}
        # A cache, because two robots commonly read two rows of ONE adapter and
        # calling get_homos() twice would compare rows from two frames.
        reads: dict[int, tuple] = {}
        for name, (adapter, index) in self.rows.items():
            key = id(adapter)
            if key not in reads:
                reads[key] = _read_adapter(adapter)
            homos, note = reads[key]
            base = fb.get(name, IDENTITY_POSE)
            if homos is None:
                out[name] = base.stale(note or "no mocap frame")
                continue
            out[name] = row_to_pose(homos, index, self.corrections.get(name),
                                    base, f"{name} base row unusable")
        for name, pose in fb.items():           # robots the cameras do not see
            out.setdefault(name, pose)
        return out


def _read_adapter(adapter) -> tuple:
    """``(homos | None, note)`` for one mocap adapter, never raising.

    A dead SDK thread is not fatal to a display; it is a reason to say so and
    keep drawing the last pose the operator chose.
    """
    try:
        homos = adapter.get_homos()
        state = adapter.get_state()
        # `stale`, not `q_stale`: a base pose is a rigid-body row, and it keeps
        # updating even on frames where the ARM's joint solve fails.
        if state.stale:
            return None, "mocap stale"
        return homos, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def row_to_pose(homos, index: int, correction, fallback: BasePose,
                note: str) -> BasePose:
    """One ``(4, 4)`` row -> a :class:`BasePose`, or the fallback if it is junk.

    Motive publishes an unsolvable body as an all-zero pose, and a row that has
    never been seen is the identity the array was initialised with.  Both are
    indistinguishable from "the robot is at the world origin", so both are
    refused rather than believed.
    """
    if homos is None or index is None or index >= len(homos):
        return fallback.stale(note)
    t = np.asarray(homos[index], dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        return fallback.stale(note)
    if np.allclose(t, np.eye(4)) or np.allclose(t[0:3, 3], 0.0):
        return fallback.stale(note + " (untracked)")
    world = t @ (np.eye(4) if correction is None
                 else np.asarray(correction, dtype=float))
    return BasePose(world[0:3, 3].copy(), TF.mat_to_quat(world[0:3, 0:3]))


def _split_xyz_rpy(value) -> tuple:
    """Accept ``xyz``, ``(xyz, rpy_deg)`` or a :class:`BasePose`-ish pair."""
    if isinstance(value, BasePose):
        raise TypeError("ManualBasePoses takes xyz / (xyz, rpy_deg), not a "
                        "BasePose: a BasePose carries a quaternion, and the "
                        "slider this class exists for is in degrees")
    seq = list(value)
    if len(seq) == 2 and all(hasattr(v, "__len__") for v in seq):
        return tuple(seq[0]), tuple(seq[1])
    return tuple(seq), (0.0, 0.0, 0.0)


@dataclass
class MountWriter:
    """Applies base poses to a compiled scene's mocap bodies.

    Built once against a model; :meth:`apply` is what runs per frame.  Holding
    the ids rather than looking them up each time matters at 160 Hz, and looking
    them up by NAME rather than by index matters because MuJoCo assigns mocap
    ids in body-declaration order and this scene's declaration order depends on
    which optional robots were included.

    A robot absent from the scene gets id ``-1`` and is skipped, so the same
    writer serves a room with one robot and a room with three.
    """

    ids: dict
    last: dict | None = field(default=None, repr=False)

    @classmethod
    def from_model(cls, model, bodies: dict | None = None) -> "MountWriter":
        import mujoco

        bodies = MOUNT_BODY if bodies is None else bodies
        ids = {}
        for name, body in bodies.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            ids[name] = -1 if bid < 0 else int(model.body_mocapid[bid])
        return cls(ids=ids)

    def present(self) -> tuple:
        """The robots this model actually carries a mount for."""
        return tuple(n for n, i in self.ids.items() if i >= 0)

    def apply(self, data, poses: dict) -> None:
        """Write every mount named in *poses* that this model has.

        A pose that is not ``fresh`` is written anyway — it is the fallback,
        which is a real pose the operator chose — but the caller can see
        ``fresh`` and say so.  Freezing is the caller's decision, not this
        method's, because "freeze" and "draw the operator's parked pose" look
        identical on screen and only one of them is honest about why.
        """
        for name, pose in poses.items():
            mid = self.ids.get(name, -1)
            if mid < 0:
                continue
            data.mocap_pos[mid] = np.asarray(pose.pos, dtype=float)
            quat = np.asarray(pose.quat, dtype=float)
            if float(np.abs(quat).sum()) > 1e-9:
                data.mocap_quat[mid] = quat
        self.last = dict(poses)
