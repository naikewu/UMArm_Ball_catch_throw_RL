"""Everything in ``viz/`` that can be checked without cameras or a display.

Opens NO COM port, NO socket and NO window.  What it does exercise: the scene
composer in all four flag combinations, the display model against
``UMArm_KINEMATICS.fkine``, both feeds, the render loop against a mock viewer
holding a real ``MjvScene``, the ``scn.ngeom`` reset invariant, the
freeze-on-stale semantics of the base-pose sources, and the promise that closing
the viewer sets nothing.

    python viz/self_check.py            # exits 0 when everything passes

A script rather than a pytest module, because the assertions worth reading are
the numeric ones — the fkine agreement, the geom count, the chain reproduction —
and a passing pytest prints none of them.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from viz import mjcf_canarm as MJ, viz_layout as VZ, base_poses as BP
from viz import multi_arm_viewer as MAV

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + str(detail)) if detail else ""))
    if not cond:
        fails.append(name)


print("(a) build_room_scene, all four flag combinations")
canarm_xml_ref = None
for rs in (False, True):
    for kv in (False, True):
        xml = MJ.build_room_scene(include_rs485=rs, include_kinova=kv)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        nj = {r: len(MJ.robot_joint_names(m, r)) for r in VZ.ROBOTS}
        check("rs485=%-5s kinova=%-5s compiles + mj_forward" % (rs, kv), True,
              "nq=%d nmocap=%d canarm/rs485/kinova joints=%d/%d/%d  kinova: %s"
              % (m.nq, m.nmocap, nj["canarm"], nj["rs485"], nj["kinova"], MJ.LAST_KINOVA_NOTE))
        check("  canarm always 12 joints", nj["canarm"] == 12)
        check("  rs485 present iff asked", (nj["rs485"] == 12) == rs)
        # The CAN arm must be unaffected by who else is in the room: same qpos
        # addresses, same plate positions.  Compared as model structure rather
        # than as text, because a grafted Gen3 re-serialises the whole document
        # and reformats whitespace it does not change.
        adr = tuple(int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                    for n in MJ.joint_names("canarm_"))
        pl = np.array([d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)]
                       for n in MJ.plate_site_names("canarm_")])
        if canarm_xml_ref is None:
            canarm_xml_ref = (adr, pl)
        check("  canarm qpos block and plate poses unchanged by the options",
              adr == canarm_xml_ref[0] and np.allclose(pl, canarm_xml_ref[1]),
              "qposadr %s" % (adr,))

print()
print("(a2) the display model against UMArm_KINEMATICS.fkine")
import importlib
K = importlib.import_module("UMArm_KINEMATICS.fkine")
from UMArm_KINEMATICS import canarm_params as cp
from UMArm_MOCAP import canarm_frames as CF
m = mujoco.MjModel.from_xml_string(MJ.build_room_scene(canarm_mount=((0, 0, 0), (0, 0, 0))))
d = mujoco.MjData(m)
sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in MJ.plate_site_names("canarm_")]
# q is written by joint NAME, which is what the viewer does: the CAN arm declares
# its proximal y hinge first, so q order and qpos order differ.
qadr_c = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in MJ.joint_names("canarm_")]
check("the display's CAN-arm order is the measured one",
      MJ.CANARM_PROXIMAL_ORDER == CF.PROXIMAL_ORDER == "yx",
      "display %s, canarm_frames %s" % (MJ.CANARM_PROXIMAL_ORDER, CF.PROXIMAL_ORDER))
check("  and qpos is therefore not q for the CAN arm", qadr_c[:2] == [1, 0], "qposadr %s" % qadr_c)
rng = np.random.default_rng(7)
worst = wrong = 0.0
for _ in range(200):
    q = rng.uniform(-0.6, 0.6, 12)
    d.qpos[qadr_c] = q
    mujoco.mj_forward(m, d)
    P = np.array([d.site_xpos[i] for i in sids])
    worst = max(worst, float(np.abs(P - np.asarray(
        K.ujoint_centres(q, params=cp.CANARM_PARAMS, order=MJ.CANARM_PROXIMAL_ORDER))).max()))
    wrong = max(wrong, float(np.abs(P - np.asarray(
        K.ujoint_centres(q, params=cp.CANARM_PARAMS, order="xy"))).max()))
check("plate sites reproduce fkine.ujoint_centres(order='yx') over 200 random q", worst < 1e-12,
      "max |err| = %.3e m" % worst)
check("  and the check has teeth: order='xy' is off by millimetres", wrong > 1e-3,
      "max |err| under 'xy' = %.1f mm" % (1e3 * wrong))
gnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
count = {s: sum(1 for g in gnames if g.startswith("canarm_") and g.endswith("_" + s))
         for s in ("sleeve", "yarm", "bearing", "bracket", "tendon")}
check("the CAN arm is drawn as the ProMax: 24 sleeves, Y arms, bearings, brackets, tendon stubs",
      all(v == 24 for v in count.values()) and m.ntendon == 0 and m.nu == 0,
      "%s ntendon=%d nu=%d style: %s" % (count, m.ntendon, m.nu, MJ.LAST_ARM_STYLE_NOTE.get("canarm_")))
link1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "canarm_seg1_link")
check("  every segment-1 sleeve and Y arm rides the link body, not a u-joint plate",
      all(int(m.geom_bodyid[i]) == link1 for i, g in enumerate(gnames)
          if g.startswith("canarm_s1_") and (g.endswith("_sleeve") or g.endswith("_yarm"))))
mr = mujoco.MjModel.from_xml_string(MJ.build_room_scene(
    include_rs485=True, canarm_mount=((0, 0, 1.0), (0, 0, 0)), rs485_mount=((0, 0, 0), (0, 0, 0))))
dr = mujoco.MjData(mr)
qadr_r = [int(mr.jnt_qposadr[mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in MJ.joint_names("rs485_")]
sr = [mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_SITE, n) for n in MJ.plate_site_names("rs485_")]
worst_r = 0.0
for _ in range(50):
    q = rng.uniform(-0.6, 0.6, 12)
    dr.qpos[qadr_r] = q
    mujoco.mj_forward(mr, dr)
    P = np.array([dr.site_xpos[i] for i in sr])
    worst_r = max(worst_r, float(np.abs(P - np.asarray(K.ujoint_centres(q))).max()))
rs_names = [mujoco.mj_id2name(mr, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(mr.ngeom)]
rs_sleeves = sum(1 for g in rs_names if g.startswith("rs485_")
                 and (g.endswith("_sleeve") or g.endswith("_yarm") or g.endswith("_bearing")))
check("the RS485 arm keeps the legacy order 'xy' and its rod-and-disk drawing",
      worst_r < 1e-12 and rs_sleeves == 0,
      "max |err| = %.3e m, rs485 ProMax geoms = %d" % (worst_r, rs_sleeves))
gaps = None
d.qpos[:12] = 0
mujoco.mj_forward(m, d)
P = np.array([d.site_xpos[i] for i in sids])
gaps = np.linalg.norm(np.diff(P, axis=0), axis=1)
check("plate gaps equal the chain argument", np.abs(gaps - np.array(MJ.DEFAULT_CANARM_CHAIN_M)).max() < 1e-12,
      "max |err| = %.3e m" % float(np.abs(gaps - np.array(MJ.DEFAULT_CANARM_CHAIN_M)).max()))
custom = (0.30, 0.06, 0.26, 0.06, 0.25)
m2 = mujoco.MjModel.from_xml_string(MJ.build_room_scene(canarm_chain_m=custom, canarm_mount=((0, 0, 0), (0, 0, 0))))
d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
s2 = [mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_SITE, n) for n in MJ.plate_site_names("canarm_")]
g2 = np.linalg.norm(np.diff(np.array([d2.site_xpos[i] for i in s2]), axis=0), axis=1)
check("chain_m is a real argument (a longer arm builds)", np.abs(g2 - np.array(custom)).max() < 1e-12,
      "max |err| = %.3e m" % float(np.abs(g2 - np.array(custom)).max()))

print()
print("(b) sim_stream -> CanArmMocap-shaped q -> the viewer's in-process feed")
from UMArm_MOCAP.sim_stream import CanArmSimStream
rx = CanArmSimStream(rate_hz=200.0).start()
try:
    time.sleep(0.3)
    st = rx.get_state()
    q = rx.get_q()
    check("sim stream publishes through the real receiver",
          q is not None and len(q) == 12 and not st.q_stale,
          "fps=%.1f frames=%d valid=%d q_stale=%s" % (st.fps, st.frames, st.valid_frames, st.q_stale))
    feed = MAV.MocapFeed({"canarm": rx}, fallback=MAV._manual_fallback({}))
    frames = feed.read()
    check("MocapFeed yields a RobotFrame with q and plates",
          frames["canarm"].q is not None and frames["canarm"].plates_ok
          and np.asarray(frames["canarm"].plates).shape == (6, 4, 4))

    # the render loop body, against a mock viewer holding a real MjvScene
    out5 = MAV.smoke_render(MAV.MocapFeed({"canarm": rx}, fallback=MAV._manual_fallback({})), frames=5)
    out20 = MAV.smoke_render(MAV.MocapFeed({"canarm": rx}, fallback=MAV._manual_fallback({})), frames=20)
    check("render loop runs the requested number of frames", out5["frames"] == 5 and out20["frames"] == 20,
          "%s / %s" % (out5, out20))
    check("scn.ngeom is reset every pass (invariant 4)", out5["max_ngeom"] == out20["max_ngeom"],
          "5 frames -> %d geoms, 20 frames -> %d" % (out5["max_ngeom"], out20["max_ngeom"]))

    # q actually reaches qpos, and mj_forward moves the model
    xml = MJ.build_room_scene()
    mm = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(mm)
    qadr, writer = MAV._address_book(mujoco, mm)
    scn = mujoco.MjvScene(mm, maxgeom=500)
    MAV.render_once(mujoco, np, mm, dd, qadr, writer, feed.read(), scn=scn)
    got = np.array([dd.qpos[a] for a in qadr["canarm"]])       # q order, by name
    want = np.asarray(feed.read()["canarm"].q)
    check("q reaches data.qpos through render_once", np.allclose(got, want, atol=0.2),
          "|dq| max %.4f (the stream moves between the two reads)" % float(np.abs(got - want).max()))
    check("mocap mount written from the base source", float(np.abs(dd.mocap_pos).sum()) > 0.0,
          "mocap_pos = %s" % np.round(np.asarray(dd.mocap_pos), 3).tolist())
    check("overlay drew geoms", int(scn.ngeom) > 0, "ngeom=%d" % int(scn.ngeom))

    # NaN guard
    bad = dict(feed.read())
    bad["canarm"] = MAV.RobotFrame(q=[float("nan")] * 12)
    keep = np.array(dd.qpos[:12])
    MAV.render_once(mujoco, np, mm, dd, qadr, writer, bad, scn=scn)
    check("a NaN q does not poison qpos", np.allclose(np.array(dd.qpos[:12]), keep))
finally:
    rx.stop()

print()
print("(b2) the shared-array feed path")
arr = VZ.make_array()
VZ.write_robot(arr, "canarm", q=np.linspace(0.01, 0.12, 12), mount_pos=(0.1, 0.2, 1.3),
               mount_quat=(1, 0, 0, 0), fresh=True,
               plates=np.tile(np.eye(4), (6, 1, 1)) * 1.0, plates_ok=False)
blk = VZ.read_robot(arr, "canarm")
check("write_robot / read_robot round trip", np.allclose(blk["q"], np.linspace(0.01, 0.12, 12))
      and np.allclose(blk["mount_pos"], (0.1, 0.2, 1.3)) and blk["fresh"] and not blk["plates_ok"])
sfeed = MAV.SharedArrayFeed(arr)
mm = mujoco.MjModel.from_xml_string(MJ.build_room_scene()); dd = mujoco.MjData(mm)
qadr, writer = MAV._address_book(mujoco, mm)
MAV.render_once(mujoco, np, mm, dd, qadr, writer, sfeed.read())
check("shared-array q reaches qpos, joint by joint by name",
      np.allclose(np.array([dd.qpos[a] for a in qadr["canarm"]]), np.linspace(0.01, 0.12, 12)))
check("shared-array mount reaches mocap_pos", np.allclose(np.asarray(dd.mocap_pos)[0], (0.1, 0.2, 1.3)))
arr[VZ.GENERATION] = 7.0
check("generation is readable through the feed", sfeed.generation() == 7.0)
r = MAV.smoke_render(MAV.SharedArrayFeed(arr), frames=4)
check("shared-array feed drives the render loop", r["frames"] == 4, r)

print()
print("(b3) closing the viewer stops nothing")
ev = threading.Event()
MAV.smoke_render(MAV.SharedArrayFeed(VZ.make_array()), frames=2)


def launch(model, data, **kw):
    return MAV._MockViewer(model, frames=2)


MAV.run_viewer(MAV.SharedArrayFeed(VZ.make_array()), stop_evt=ev, launch=launch)
check("run_viewer never sets the caller's stop event", not ev.is_set())

print()
print("(c) base_poses semantics")


class _Stale:
    class _S:
        stale = True
        q_stale = True
    def get_state(self): return self._S()
    def get_homos(self): return np.tile(np.eye(4), (9, 1, 1))


class _Live:
    class _S:
        stale = False
        q_stale = True          # frames arriving, joints not converting
    def get_state(self): return self._S()
    def get_homos(self):
        h = np.tile(np.eye(4), (9, 1, 1)).astype(float)
        h[0, 0:3, 3] = (1.0, 2.0, 3.0)
        return h


manual = BP.ManualBasePoses({"canarm": ((0.5, 0.0, 1.2), (0, 0, 0))})
src = BP.MocapBasePoses({"canarm": (_Stale(), 0)}, manual)
p = src.poses()["canarm"]
check("stale mocap freezes onto the fallback, flagged", (not p.fresh) and np.allclose(p.pos, (0.5, 0, 1.2)),
      "note=%r" % p.note)
src = BP.MocapBasePoses({"canarm": (_Live(), 0)}, manual)
p = src.poses()["canarm"]
check("a live base row is used even when q_stale is True (stale, not q_stale)",
      p.fresh and np.allclose(p.pos, (1.0, 2.0, 3.0)))
src = BP.MocapBasePoses({"canarm": (_Stale(), 0), "kinova": (_Stale(), 8)}, manual)
check("poses() is an N-map keyed by robot name", set(src.poses()) == {"canarm", "kinova"})

print()
print("(d) MountWriter by name, with robots absent")
mm = mujoco.MjModel.from_xml_string(MJ.build_room_scene(include_rs485=True))
w = BP.MountWriter.from_model(mm)
check("present() reports only the robots the scene carries", set(w.present()) == {"canarm", "rs485"},
      w.ids)

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("ALL CHECKS PASSED")
