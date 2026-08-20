"""The room's window: every robot in it, as MuJoCo draws them.

THIS IS A DISPLAY, NOT A PLANT WINDOW.  **Closing it stops nothing and vents
nothing.**  That is a deliberate reversal of the rule both RS485 viewers follow —
``UMArm_CONTROL/gui/viewer.py:23-28`` and ``UMArm_COLLAB/gui/viewer.py:12-14``
both say, correctly for their own bench, that closing the 3D window is the
operator saying they are done with the arm, and the backend's vent hangs off that
event.  Here the CAN arm is driven by ``TLE_PCB/tlelib``'s 150 Hz cycle in
another process entirely, the window is a second opinion about where the arm is,
and inheriting that rule would mean **opening a display arms a shutdown path
nobody asked for**.  A viewer process that exits sets no event that any bus
thread is waiting on.

THE INVARIANTS, carried verbatim from the RS485 pattern:

1. ``mujoco.viewer.launch_passive`` — a passive viewer, so this process owns the
   render loop and nothing is stepping behind it.
2. Per frame, ``data.qpos`` and ``data.mocap_pos`` / ``data.mocap_quat`` are
   **overwritten**, never integrated towards.
3. ``mujoco.mj_forward`` ONLY — **never** ``mj_step``.  Forward kinematics from
   the angles it was handed, so nothing the renderer does can perturb what it is
   showing, and a dropped frame costs a frame.
4. ``scn.ngeom = 0`` at the top of **every** overlay pass.  ``user_scn`` is
   additive and ``sync()`` does not clear it, so an overlay that appends fills
   the scene with the session's history until it hits MuJoCo's geom cap.
5. The render loop is nested inside a **rebuild loop keyed on
   ``viz_layout.GENERATION``**.  ``launch_passive`` is bound to the model it was
   handed and MuJoCo will not recompile a live one, so a robot arriving or
   leaving tears the window down and opens a fresh one.  It blinks, and that is
   the honest cost of changing a compiled model.
6. The viewer loads its **own** model.  ``mjModel``/``mjData`` do not cross a
   process boundary.

WHY THE OVERLAY IS THE POINT.  The rendered arm is forward kinematics of ``q``;
the spheres are where the cameras say the plates are.  They are the same six
physical points, so the two agreeing *on screen* is the whole pose pipeline —
markers, plate registration, joint reconstruction, length table — verified at a
glance, live, while the arm moves.  When they separate you can see which joint.

Read the overlay with two caveats the RS485 workspace measured and paid for.
The lock frame agrees with Motive's streamed plate frame only to 0.37-3.67 deg,
which at a 100 mm marker radius is 5 mm, so a drawn marker landing millimetres
off a reported one is a frame convention rather than a fault.  And the CAN arm's
chain lengths are placeholders (``UMArm_KINEMATICS.canarm_params.MEASURED`` is
``False``), so until they are measured a separation that grows down the chain is
the length table, not the joints.

TWO FEEDS, ONE RENDERER.

* :class:`SharedArrayFeed` — reads ``viz.viz_layout``'s flat block of doubles, so
  a producer in another process can drive this window.
* :class:`MocapFeed` — reads mocap adapters **in this process**.  This is what
  the control GUI uses first: the GUI spawns a viewer process, and that process
  builds its own receiver.  A ``CanArmMocap`` cannot cross a process boundary
  (it owns SDK threads and a socket) and must not be shared with a control loop
  anyway, so the child constructing its own is the only arrangement that is both
  possible and honest.

Both are polled by one reader thread, so a mocap read that blocks costs a stale
frame rather than a frozen window.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_WS_ROOT = Path(__file__).resolve().parents[1]
if str(_WS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WS_ROOT))

try:
    from . import base_poses as BP
    from . import mjcf_canarm as MJ
    from . import viz_layout as VZ
except ImportError:                                          # pragma: no cover
    import base_poses as BP                                  # type: ignore
    import mjcf_canarm as MJ                                 # type: ignore
    import viz_layout as VZ                                  # type: ignore

__all__ = [
    "RobotFrame", "Feed", "SharedArrayFeed", "MocapFeed", "FeedReader",
    "run_viewer", "viewer_main", "render_once", "smoke_render",
    "DEFAULT_FPS",
]

#: Redraw rate.  60 is what both RS485 viewers use and is comfortably above the
#: 120 Hz mocap rate divided by any number an eye can resolve.
DEFAULT_FPS = 60.0

#: Overlay colours.  The model draws itself, so these only have to read as
#: "measured": one hue for the plate origins, a dimmer one for the measured
#: chain between them, and the conventional red/green/blue triad for axes.
_PLATE_RGBA = (1.0, 0.35, 0.1, 1.0)
_CHAIN_RGBA = (1.0, 0.6, 0.15, 0.85)
_MOUNT_RGBA = {"canarm": (1.0, 0.45, 0.15, 0.9),
               "rs485": (0.55, 0.85, 0.35, 0.9),
               "kinova": (0.35, 0.75, 1.0, 0.9)}
_AXIS_RGBA = ((0.95, 0.2, 0.2, 0.9), (0.2, 0.9, 0.3, 0.9), (0.3, 0.5, 1.0, 0.9))
_PLATE_R = 0.010
_AXIS_LEN = 0.035
_AXIS_R = 0.0025
_MOUNT_AXIS_LEN = 0.08
_MOUNT_R = 0.012


@dataclass
class RobotFrame:
    """One robot's contribution to one drawn frame.

    Every field is independently optional, because the robots in this room are
    known to different degrees: the CAN arm may have joints from the bus and no
    mocap, the RS485 arm may have mocap and no bus, and the Gen3 may have
    neither.  ``None`` means "do not touch what is already in the model", which
    is what makes a partial feed render a partial answer rather than an arm
    snapped to zero.
    """

    q: object = None                 #: (nq,) radians, or None
    mount_pos: object = None         #: (3,) metres, or None
    mount_quat: object = None        #: (4,) scalar-first, or None
    fresh: bool = True               #: False when the mount is a frozen fallback
    plates: object = None            #: (n, 4, 4) measured plate poses, or None
    plates_ok: bool = False          #: False disables this robot's overlay
    note: str = ""                   #: why, when something is not fresh


class Feed:
    """Interface: give me the current frame for every robot you know about."""

    def read(self) -> dict:
        """``{robot_name: RobotFrame}``.  Called from the reader thread only."""
        raise NotImplementedError

    def generation(self) -> float:
        """Bumped when the scene must be rebuilt.  See invariant 5."""
        return 0.0

    def close(self) -> None:
        pass


class SharedArrayFeed(Feed):
    """Reads the ``viz_layout`` block another process publishes.

    No lock, by construction — see :func:`viz_layout.make_array` for why a torn
    frame is preferred to a producer that waits on a repaint.
    """

    def __init__(self, arr, robots=None):
        self.arr = arr
        self.robots = tuple(robots or VZ.ROBOTS)

    def read(self) -> dict:
        import numpy as np

        out = {}
        for name in self.robots:
            blk = VZ.read_robot(self.arr, name)
            n = VZ.N_PLATES[name]
            plates = (np.asarray(blk["plates"], dtype=float).reshape(n, 4, 4)
                      if n and blk["plates_ok"] else None)
            out[name] = RobotFrame(q=blk["q"], mount_pos=blk["mount_pos"],
                                   mount_quat=blk["mount_quat"],
                                   fresh=blk["fresh"], plates=plates,
                                   plates_ok=blk["plates_ok"])
        return out

    def generation(self) -> float:
        return float(self.arr[VZ.GENERATION])


class MocapFeed(Feed):
    """Reads mocap adapters directly, in this process.

    *adapters* maps a robot name to anything carrying ``MocapRx``'s surface:
    ``get_q()``, ``get_state()``, ``get_homos()`` and — on the marker-based
    receivers — ``get_marker_poses()``.  Feature detection is duck-typed, the
    way ``UMArm_CONTROL/gui/backend.py:349`` already does it, so a streamed-pose
    receiver and a marker-registered one both work and the second simply draws a
    better overlay.

    ``get_marker_poses`` is PREFERRED where it exists, and the reason is worth
    keeping: it returns the poses the solve actually used, not a second opinion
    computed later, and Motive is free to move a rigid body's pivot when it
    re-solves an asset without telling any client — which is the whole reason
    the marker path exists.

    *base_source* is a :class:`viz.base_poses.BasePoseSource`; when it is absent
    one is built from the same adapters, reading each arm's row
    ``IDX_BASE`` with the manual poses as the fallback.  ``q_stale`` gates the
    JOINTS and ``stale`` gates the MOUNT — the two flags differ exactly when
    frames keep arriving and stop converting, and in that state the joints are
    silently older every tick while the base row is still current.
    """

    def __init__(self, adapters: dict, *, base_source=None, fallback=None,
                 hold_stale_q: bool = True):
        self.adapters = dict(adapters)
        self.fallback = fallback
        self.hold_stale_q = bool(hold_stale_q)
        if base_source is None:
            if adapters:
                rows = {name: (ad, _base_row(name))
                        for name, ad in self.adapters.items()}
                base_source = BP.MocapBasePoses(rows, fallback)
            else:
                # No cameras at all is a normal case, not a degraded one: the
                # room still has to draw every robot somewhere, and where the
                # operator put them is the best answer available.
                base_source = fallback
        self.base_source = base_source

    def read(self) -> dict:
        import numpy as np

        poses = self.base_source.poses() if self.base_source is not None else {}
        out = {}
        for name in set(self.adapters) | set(poses):
            adapter = self.adapters.get(name)
            q = plates = None
            ok = False
            note = ""
            if adapter is not None:
                try:
                    state = adapter.get_state()
                    q = adapter.get_q()
                    if state.q_stale:
                        # The joints stopped converting.  Hold the last pose but
                        # say so; a drawn arm that quietly freezes reads exactly
                        # like an arm that is holding still.
                        note = "q stale"
                        if not self.hold_stale_q:
                            q = None
                    getter = getattr(adapter, "get_marker_poses", None)
                    raw = getter() if callable(getter) else adapter.get_homos()
                    if raw is not None:
                        n = VZ.N_PLATES.get(name, 0)
                        arr = np.asarray(raw, dtype=float)
                        if n and arr.shape[0] >= n:
                            plates = arr[:n]
                            ok = bool(np.isfinite(plates).all())
                except Exception as exc:
                    note = f"{type(exc).__name__}: {exc}"
            pose = poses.get(name)
            out[name] = RobotFrame(
                q=q,
                mount_pos=None if pose is None else pose.pos,
                mount_quat=None if pose is None else pose.quat,
                fresh=True if pose is None else pose.fresh,
                plates=plates, plates_ok=ok,
                note=note or ("" if pose is None else pose.note))
        return out

    def close(self) -> None:
        if self.base_source is not None:
            self.base_source.close()


def _base_row(robot: str) -> int:
    """Which ``get_homos()`` row carries *robot*'s base.

    Row 0 for either arm — its own receiver's ``IDX_BASE`` — and row 8 for the
    Gen3, which ``MocapRx`` routes streaming id 1008 into regardless of which
    arm's id block the receiver is bound to.
    """
    try:
        from UMArm_MOCAP import mocap_constants as mc
        return (mc.KINOVA_RIGID_BODY_INDEX if robot == "kinova" else mc.IDX_BASE)
    except Exception:                                        # pragma: no cover
        return 8 if robot == "kinova" else 0


@dataclass
class FeedReader:
    """A thread that keeps the newest frame from a feed, and never blocks a draw.

    The render loop must hit its period whatever the feed is doing: a NatNet
    read that stalls, or a shared array on a page that has been swapped out,
    must cost a repeated frame rather than a window that stops responding.  So
    the feed is polled here and the render loop takes whatever is on the shelf.
    """

    feed: Feed
    period: float = 1.0 / 240.0
    _latest: dict = field(default_factory=dict)
    _gen: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: object = None
    _errors: int = 0

    def start(self) -> "FeedReader":
        self.pump()                      # one synchronous read, so frame 0 is real
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="viz-feed-reader")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def pump(self) -> None:
        """One read.  Public because the offline smoke test drives it by hand."""
        try:
            frames = self.feed.read()
            gen = self.feed.generation()
        except Exception:
            self._errors += 1
            return
        with self._lock:
            self._latest = frames
            self._gen = gen

    def latest(self) -> tuple:
        with self._lock:
            return dict(self._latest), self._gen

    def _run(self) -> None:
        while not self._stop.is_set():
            self.pump()
            self._stop.wait(self.period)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _triad(mujoco, np, scn, pos, rot, rgba, length, radius, ball) -> None:
    """A sphere plus three axis capsules, if there is room left in the scene."""
    if scn.ngeom >= scn.maxgeom - 4:
        return
    eye = np.eye(3).flatten()
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([ball, 0.0, 0.0]), pos, eye,
                        np.array(rgba, dtype=np.float32))
    scn.ngeom += 1
    if rot is None:
        return
    for axis in range(3):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), eye,
                            np.array(_AXIS_RGBA[axis], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             pos, pos + length * rot[:, axis])
        scn.ngeom += 1


def draw_overlay(mujoco, np, scn, frames: dict) -> int:
    """Rewrite the overlay from this frame.  Returns the geom count used.

    ``scn.ngeom = 0`` FIRST, every pass, no exceptions: ``user_scn`` is additive
    and ``sync()`` does not clear it (invariant 4).
    """
    scn.ngeom = 0
    for name, frame in frames.items():
        # The mount, always: a triad where this robot's base is being written.
        if frame.mount_pos is not None:
            pos = np.asarray(frame.mount_pos, dtype=float)
            quat = (None if frame.mount_quat is None
                    else np.asarray(frame.mount_quat, dtype=float))
            rot = None
            if quat is not None and float(np.abs(quat).sum()) > 1e-9:
                m = np.zeros(9)
                mujoco.mju_quat2Mat(m, quat / np.linalg.norm(quat))
                rot = m.reshape(3, 3)
            if np.isfinite(pos).all():
                rgba = _MOUNT_RGBA.get(name, (0.8, 0.8, 0.8, 0.9))
                if not frame.fresh:
                    # A frozen mount is drawn faint rather than not drawn: the
                    # robot IS still somewhere, and a base that vanishes reads
                    # as a robot that left rather than as a stream that stopped.
                    rgba = (rgba[0], rgba[1], rgba[2], 0.25)
                _triad(mujoco, np, scn, pos, rot, rgba, _MOUNT_AXIS_LEN,
                       _MOUNT_R * 0.35, _MOUNT_R)

        # The measured plates, for whoever has them.
        if not frame.plates_ok or frame.plates is None:
            continue
        poses = np.asarray(frame.plates, dtype=float)
        origins = []
        for t in poses:
            if t.shape != (4, 4) or not np.isfinite(t).all():
                continue
            o = t[0:3, 3]
            origins.append(o)
            _triad(mujoco, np, scn, o, t[0:3, 0:3], _PLATE_RGBA, _AXIS_LEN,
                   _AXIS_R, _PLATE_R)
        for a, b in zip(origins, origins[1:]):     # the measured chain itself
            if scn.ngeom >= scn.maxgeom - 1:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                np.zeros(3), np.eye(3).flatten(),
                                np.array(_CHAIN_RGBA, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, a, b)
            scn.ngeom += 1
    return int(scn.ngeom)


# ---------------------------------------------------------------------------
# One frame, and the loop around it
# ---------------------------------------------------------------------------

def _address_book(mujoco, model) -> tuple:
    """``({robot: [qpos addresses]}, MountWriter)`` for a compiled scene.

    BY NAME, not by index.  Which qpos address a robot's block starts at depends
    on which optional robots were included, so a viewer that assumed "the Gen3's
    seven follow the arm's twelve" would drive the RS485 arm with Gen3 angles in
    a room that has no RS485 arm.
    """
    qadr = {}
    for robot in VZ.ROBOTS:
        addrs = []
        for jname in MJ.robot_joint_names(model, robot):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                addrs.append(int(model.jnt_qposadr[jid]))
        if addrs:
            qadr[robot] = addrs
    return qadr, BP.MountWriter.from_model(model)


def render_once(mujoco, np, model, data, qadr, writer, frames: dict,
                scn=None) -> None:
    """Write one frame into *data*, run forward kinematics, refresh the overlay.

    The whole per-frame contract lives here so the offline smoke test can drive
    exactly the code the window drives, with a mock viewer in place of a GL
    context.  Nothing in it opens anything.
    """
    for robot, addrs in qadr.items():
        q = frames.get(robot, RobotFrame()).q
        if q is None:
            continue
        for i, adr in enumerate(addrs):
            if i < len(q):
                value = float(q[i])
                # A NaN reaching qpos poisons the whole model for the rest of
                # the session -- mj_forward propagates it into every xpos and
                # nothing puts it back.  One frame of a bad mocap solve must not
                # cost the window.
                if value == value:
                    data.qpos[adr] = value

    poses = {}
    for robot, frame in frames.items():
        if frame.mount_pos is None:
            continue
        quat = (np.asarray(frame.mount_quat, dtype=float)
                if frame.mount_quat is not None
                else np.array([1.0, 0.0, 0.0, 0.0]))
        poses[robot] = BP.BasePose(np.asarray(frame.mount_pos, dtype=float),
                                   quat, frame.fresh, frame.note)
    if poses:
        writer.apply(data, poses)

    # mj_forward ONLY.  Never mj_step (invariant 3).
    mujoco.mj_forward(model, data)
    if scn is not None:
        draw_overlay(mujoco, np, scn, frames)


def _render_until(mujoco, np, model, data, reader, stop_evt, period,
                  generation, launch) -> bool:
    """One window's lifetime.  True when it ended because the scene changed."""
    qadr, writer = _address_book(mujoco, model)
    with launch(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running() and not (stop_evt is not None and stop_evt.is_set()):
            frames, gen = reader.latest()
            if gen != generation:
                return True                    # rebuild: invariant 5
            render_once(mujoco, np, model, data, qadr, writer, frames,
                        scn=getattr(v, "user_scn", None))
            v.sync()
            time.sleep(period)
    return False


def run_viewer(feed: Feed, *, include_rs485: bool = False,
               include_kinova: bool = False, fps: float = DEFAULT_FPS,
               stop_evt=None, launch=None, scene_kwargs=None) -> None:
    """Open the window and render until it closes.  Returns; stops nothing else.

    *launch* defaults to :func:`mujoco.viewer.launch_passive` and exists as a
    seam so the offline test can substitute a viewer object with no GL context.
    """
    import mujoco
    import numpy as np

    if launch is None:
        import mujoco.viewer
        launch = mujoco.viewer.launch_passive

    period = 1.0 / float(fps)
    reader = FeedReader(feed).start()
    try:
        while not (stop_evt is not None and stop_evt.is_set()):
            _, generation = reader.latest()
            xml = MJ.build_room_scene(include_rs485=include_rs485,
                                      include_kinova=include_kinova,
                                      **dict(scene_kwargs or {}))
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            if not _render_until(mujoco, np, model, data, reader, stop_evt,
                                 period, generation, launch):
                break                          # the operator closed the window
    except Exception as exc:                   # a missing display is not fatal
        print(f"[viewer] closed: {type(exc).__name__}: {exc}")
    finally:
        reader.stop()
        feed.close()
        # DELIBERATELY NOTHING ELSE.  No stop_evt.set(), no vent, no disable.
        # See the module docstring: this window is a display.


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def viewer_main(viz_arr, stop_evt, opts: dict) -> None:
    """Spawned-process target.  Builds its own feed, its own model, its own window.

    Imports are function-local: this is a spawn target, the child re-imports the
    module, and paying MuJoCo's import cost in the parent too would double
    startup for nothing.

    *opts* selects the feed:

    ``{"feed": "shared"}``
        read *viz_arr*, which the parent published.
    ``{"feed": "mocap", "mocap": {"kind": "sim"}}``
        build a ``CanArmSimStream`` here — a real receiver fed by a synthetic
        producer thread, opening no socket.
    ``{"feed": "mocap", "mocap": {"kind": "live", ...}}``
        build a ``CanArmMocap`` here and ``start()`` it.  **This is the only
        path in this file that touches the network**, it happens only on
        explicit request, and it happens in the child so that a viewer crash
        cannot take a receiver down with a control loop attached to it.
    """
    feed = None
    try:
        spec = dict(opts.get("mocap") or {})
        if str(opts.get("feed", "shared")) == "shared" and viz_arr is not None:
            feed = SharedArrayFeed(viz_arr)
        else:
            feed = MocapFeed(_build_adapters(spec),
                             fallback=_manual_fallback(opts))
        run_viewer(feed,
                   include_rs485=bool(opts.get("include_rs485", False)),
                   include_kinova=bool(opts.get("include_kinova", False)),
                   fps=float(opts.get("fps", DEFAULT_FPS)),
                   stop_evt=stop_evt,
                   scene_kwargs=opts.get("scene_kwargs"))
    except Exception as exc:                                 # pragma: no cover
        print(f"[viewer] failed to start: {type(exc).__name__}: {exc}")
    finally:
        for name in ("adapters",):
            for adapter in getattr(feed, name, {}).values():
                try:
                    adapter.stop()
                except Exception:
                    pass


def _build_adapters(spec: dict) -> dict:
    """The mocap adapters named in *spec*, started.  ``{}`` when there are none.

    A receiver that cannot be built is not fatal: the room still draws every
    robot whose joints came from somewhere else, and the operator gets a window
    with no overlay rather than no window.
    """
    kind = str(spec.get("kind", "none")).lower()
    if kind in ("", "none", "off"):
        return {}
    try:
        if kind == "sim":
            from UMArm_MOCAP.sim_stream import CanArmSimStream
            rx = CanArmSimStream(rate_hz=float(spec.get("rate_hz", 120.0)))
        else:
            from UMArm_MOCAP.canarm_mocap import CanArmMocap
            rx = CanArmMocap(**{k: v for k, v in spec.items()
                                if k not in ("kind", "rate_hz")})
        rx.start()
        return {"canarm": rx}
    except Exception as exc:
        print(f"[viewer] no mocap: {type(exc).__name__}: {exc}")
        return {}


def _manual_fallback(opts: dict):
    """Where robots sit when nothing is measuring them."""
    return BP.ManualBasePoses({
        "canarm": opts.get("canarm_mount", MJ.DEFAULT_CANARM_MOUNT),
        "rs485": opts.get("rs485_mount", MJ.DEFAULT_RS485_MOUNT),
        "kinova": opts.get("kinova_mount", MJ.DEFAULT_KINOVA_MOUNT),
    })


# ---------------------------------------------------------------------------
# Offline smoke test
# ---------------------------------------------------------------------------

class _MockViewer:
    """A ``launch_passive`` stand-in with a real ``MjvScene`` and no GL context.

    ``MjvScene`` is a plain data structure — ``mjv_initGeom`` and
    ``mjv_connector`` fill arrays in it and neither touches OpenGL — so the
    overlay code under test here is the same code the window runs, geom for
    geom.  What this cannot exercise is ``sync()`` and the window itself, which
    is the part that needs a display and is therefore the part a headless check
    has to stop short of.
    """

    def __init__(self, model, frames: int = 5, maxgeom: int = 2000):
        import mujoco

        self.user_scn = mujoco.MjvScene(model, maxgeom=maxgeom)
        self.frames = int(frames)
        self.syncs = 0
        self.max_ngeom = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self) -> bool:
        return self.frames > 0

    def sync(self) -> None:
        self.frames -= 1
        self.syncs += 1
        self.max_ngeom = max(self.max_ngeom, int(self.user_scn.ngeom))


def smoke_render(feed: Feed, *, frames: int = 5, include_rs485: bool = False,
                 include_kinova: bool = False, fps: float = 240.0) -> dict:
    """Run the real render loop against :class:`_MockViewer`.  Opens no window.

    Returns ``{"frames", "max_ngeom", "nq", "robots"}``.  ``max_ngeom`` is the
    one number worth asserting on: an overlay that forgot ``scn.ngeom = 0``
    grows it every pass, so a value that is stable across frames is the
    invariant holding.
    """
    import mujoco
    import numpy as np

    made = {}

    def launch(model, data, **_kw):
        made["viewer"] = _MockViewer(model, frames=frames)
        return made["viewer"]

    run_viewer(feed, include_rs485=include_rs485,
               include_kinova=include_kinova, fps=fps, launch=launch)
    v = made.get("viewer")
    return {"frames": 0 if v is None else v.syncs,
            "max_ngeom": 0 if v is None else v.max_ngeom,
            "robots": tuple(feed.read())}


def _self_test(argv=None) -> int:                            # pragma: no cover
    """``python -m viz.multi_arm_viewer --self-test`` — no window, no socket."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rs485", action="store_true")
    ap.add_argument("--kinova", action="store_true")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args(argv)

    from UMArm_MOCAP.sim_stream import CanArmSimStream

    rx = CanArmSimStream(rate_hz=200.0).start()
    try:
        time.sleep(0.2)
        feed = MocapFeed({"canarm": rx}, fallback=_manual_fallback({}))
        out = smoke_render(feed, frames=args.frames,
                           include_rs485=args.rs485, include_kinova=args.kinova)
        print(f"smoke_render: {out}")
        arr = VZ.make_array()
        shared = smoke_render(SharedArrayFeed(arr), frames=3)
        print(f"shared-array feed: {shared}")
    finally:
        rx.stop()
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_self_test())
