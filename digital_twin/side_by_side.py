"""The deliverable: the twin and the real arm, same input sequence, side by side.

Three panels, and the middle one is the reason there are three.

  ``REAL CAMERA``    what the bench camera saw
  ``MEASURED (mocap)``   the same MuJoCo model posed at the ``q`` mocap measured,
                     ``mj_forward`` only, **no physics**
  ``TWIN (predicted)``   the same model at the ``q`` the twin *rolled out* from
                     the recorded pressure targets alone

Comparing the camera against a rendered twin is a comparison of two different
things — one is a photograph through an uncalibrated lens, the other a
projection through a camera pose that was matched by eye.  A disagreement there
could be the model or could be the lens, and the picture cannot say which.  The
middle panel removes that ambiguity: it is the *same renderer, same camera, same
model* as the right panel, differing only in where the joint angles came from.
Any difference between the middle and the right is the twin's error and nothing
else.  The left panel is then doing the job it is actually good at — showing
that the middle panel is a faithful account of a real robot rather than a
plausible animation.

The joint strip along the bottom carries the number the eye cannot judge: the
measured and predicted angle of a chosen joint, with a cursor at the current
frame, so a reader can see *how far* apart they are and not only *that* they
are.

Alignment is arithmetic, not search.  Every clip is written with a
``.stamps.json`` beside it holding one ``time.monotonic`` stamp per frame, and
the recording stamps every cycle's sync edge on a clock the session's metadata
relates to that one.  Each output frame therefore picks the **nearest** recorded
camera frame and the **nearest** recorded sample; nothing is interpolated,
because an interpolated pose is a pose nobody measured.

Usage::

    .venv\\Scripts\\python.exe -m digital_twin.side_by_side \\
        --session data/session_20260910_012201 \\
        --checkpoint data/fit/actuator_net.npz \\
        --out deliverable/twin_vs_real.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Panel size, and therefore the output size (three panels plus the strip).
#: Portrait panels because the arm is: a landscape frame of a ceiling-hung arm
#: is mostly wall.
PANEL_W, PANEL_H = 460, 600

#: Height of the joint-trace strip under the panels.
STRIP_H = 220

#: Output frame rate.  The arm's ring is a few Hz and the camera clips are
#: written at 15 fps, so 15 is the rate at which no panel is invented.
OUT_FPS = 15.0

#: Where the MuJoCo camera is put.  Matched **by eye** against a bench frame,
#: not calibrated: the bench camera's intrinsics were never measured, and a
#: pose fitted to an unknown focal length would be precise about the wrong
#: thing.  This is why the quantitative claim rests on the middle-versus-right
#: comparison, which shares this camera exactly, rather than on left-versus-right.
DEFAULT_CAM = {"azimuth": 90.0, "elevation": -8.0, "distance": 2.1,
               "lookat": (0.0, 0.0, 0.72)}

#: The FourCC.  This bench has no H.264 encoder (see UMArm_CAMERA.camera).
FOURCC = "mp4v"

_BG = (24, 24, 24)
_FG = (235, 235, 235)


def _cv2():
    import cv2
    return cv2


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def find_clip(session_dir: str, *, kind: str = "validation"):
    """The longest clip whose name carries ``kind``, and its stamps.

    Longest rather than first because the campaign takes a short context clip
    at many segment boundaries and a continuous one across the validation
    sequence; the deliverable wants the continuous one, and its length is what
    distinguishes it without depending on the naming staying the way it is.
    """
    media = os.path.join(session_dir, "media")
    best = None
    for path in sorted(glob.glob(os.path.join(media, f"*{kind}*.mp4"))):
        stamps = os.path.splitext(path)[0] + ".stamps.json"
        if not os.path.exists(stamps):
            continue
        with open(stamps, encoding="utf-8") as fh:
            meta = json.load(fh)
        if best is None or meta.get("frames", 0) > best[1].get("frames", 0):
            best = (path, meta)
    return best


def load_metadata(session_dir: str) -> dict:
    with open(os.path.join(session_dir, "metadata.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Rendering one arm
# --------------------------------------------------------------------------


class ArmRenderer:
    """A MuJoCo offscreen renderer posed at whatever ``q`` it is handed.

    Physics is never stepped here: both the measured and the predicted panel
    are ``mj_forward`` on a supplied ``q``.  The twin's physics already ran, in
    :mod:`digital_twin.replay`; rendering it again with physics on would be a
    second, different rollout.
    """

    def __init__(self, *, width=PANEL_W, height=PANEL_H, cam=None, xml=None):
        import mujoco
        from digital_twin import mjcf_generator as MG

        self.mujoco = mujoco
        self.MG = MG
        xml = xml if xml is not None else MG.generate_xml()
        # The offscreen buffer defaults smaller than a panel in most MJCFs;
        # a renderer larger than the buffer silently returns a cropped image.
        xml = self._raise_offbuffer(xml, width, height)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        c = dict(DEFAULT_CAM)
        c.update(cam or {})
        self.cam.azimuth = c["azimuth"]
        self.cam.elevation = c["elevation"]
        self.cam.distance = c["distance"]
        self.cam.lookat[:] = c["lookat"]
        self.opt = mujoco.MjvOption()

    @staticmethod
    def _raise_offbuffer(xml: str, width: int, height: int) -> str:
        import re
        if "offwidth" in xml:
            xml = re.sub(r'offwidth="\d+"', f'offwidth="{max(width, 640)}"', xml)
            xml = re.sub(r'offheight="\d+"', f'offheight="{max(height, 640)}"', xml)
            return xml
        return xml.replace(
            "<visual>",
            f'<visual><global offwidth="{max(width, 640)}" '
            f'offheight="{max(height, 640)}"/>', 1) if "<visual>" in xml else \
            xml.replace("</mujoco>",
                        f'<visual><global offwidth="{max(width, 640)}" '
                        f'offheight="{max(height, 640)}"/></visual></mujoco>', 1)

    def render(self, q) -> np.ndarray:
        """BGR image of the arm at joint vector ``q`` (12, radians)."""
        qpos = self.MG.q_to_qpos(np.asarray(q, dtype=float))
        n = min(len(qpos), self.model.nq)
        self.data.qpos[:n] = qpos[:n]
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam, self.opt)
        # MuJoCo returns RGB; cv2 wants BGR.
        return np.ascontiguousarray(self.renderer.render()[:, :, ::-1])


class ClipReader:
    """Forward-only reader that hands back the frame nearest a wanted time.

    Forward-only and single-buffered on purpose: seeking backwards in a
    long-GOP mp4 costs a keyframe search per call, and the compositor asks for
    monotonically increasing times.
    """

    def __init__(self, path: str, stamps_meta: dict):
        cv2 = _cv2()
        self.cap = cv2.VideoCapture(path)
        self.stamps = np.asarray(stamps_meta["t_mono_s"], dtype=float)
        self.idx = -1
        self.frame = None
        self.size = tuple(stamps_meta.get("size", (PANEL_W, PANEL_H)))

    def at(self, t_mono: float):
        want = int(np.searchsorted(self.stamps, t_mono))
        want = min(max(want, 0), len(self.stamps) - 1)
        if want > 0 and abs(self.stamps[want - 1] - t_mono) < \
                abs(self.stamps[want] - t_mono):
            want -= 1
        while self.idx < want:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.idx += 1
            self.frame = frame
        return self.frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


# --------------------------------------------------------------------------
# The joint strip
# --------------------------------------------------------------------------


def render_strip(t, q_real, q_sim, joints, *, width, height):
    """A static plot of the chosen joints; the cursor is drawn per frame.

    Drawn once with matplotlib's Agg backend at exact pixel size and then
    copied per frame, because re-rendering a plot 1350 times is most of the
    run time and none of the information.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("#181818")
    ax = fig.add_axes((0.055, 0.20, 0.93, 0.72))
    ax.set_facecolor("#181818")
    colors = ("#5aa9ff", "#ff9a5a", "#7ee07e", "#ff5a5a")
    for k, j in enumerate(joints):
        c = colors[k % len(colors)]
        ax.plot(t, np.degrees(q_real[:, j]), color=c, lw=1.4,
                label=f"j{j} measured")
        ax.plot(t, np.degrees(q_sim[:, j]), color=c, lw=1.4, ls="--",
                label=f"j{j} twin")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_xlabel("time (s)", color=_FG_hex(), fontsize=8)
    ax.set_ylabel("joint angle (deg)", color=_FG_hex(), fontsize=8)
    ax.tick_params(colors=_FG_hex(), labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(color="#333333", lw=0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=len(joints),
              facecolor="#222222", edgecolor="#444444", labelcolor=_FG_hex())
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return np.ascontiguousarray(buf[:, :, ::-1]), ax.get_position()


def _FG_hex():
    return "#ebebeb"


# --------------------------------------------------------------------------
# The compositor
# --------------------------------------------------------------------------


def compose(out_path: str, *, t, q_real, q_sim, clip=None, clip_meta=None,
            joints=(0, 4, 8), cam=None, fps: float = OUT_FPS,
            clock_offset_s: float = 0.0, title: str = "",
            metrics: dict | None = None) -> dict:
    """Write the three-panel video.  Returns what it actually wrote.

    ``t`` is in the recording's ``can_sync_time_s``; ``clock_offset_s`` carries
    it to the camera's ``monotonic`` timebase, which the session's metadata
    supplies rather than this function assuming they are the same.
    """
    cv2 = _cv2()
    n_panels = 3 if clip is not None else 2
    W = PANEL_W * n_panels
    H = PANEL_H + STRIP_H
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    strip, axpos = render_strip(t, q_real, q_sim, joints,
                                width=W, height=STRIP_H)
    ax_x0 = int(axpos.x0 * W)
    ax_x1 = int(axpos.x1 * W)

    renderer = ArmRenderer(cam=cam)
    reader = None
    if clip is not None:
        reader = ClipReader(clip, clip_meta)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*FOURCC),
                             fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a writer for {out_path}")

    t0, t1 = float(t[0]), float(t[-1])
    times = np.arange(t0, t1, 1.0 / fps)
    written = 0
    try:
        for tt in times:
            k = int(np.searchsorted(t, tt))
            k = min(max(k, 0), len(t) - 1)
            canvas = np.zeros((H, W), dtype=np.uint8)
            canvas = np.dstack([canvas] * 3)
            canvas[:, :] = _BG

            panels = []
            if reader is not None:
                frame = reader.at(tt + clock_offset_s)
                if frame is None:
                    frame = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                panels.append(("REAL CAMERA", _fit(frame, PANEL_W, PANEL_H)))
            panels.append(("MEASURED (mocap)", renderer.render(q_real[k])))
            panels.append(("TWIN (predicted)", renderer.render(q_sim[k])))

            for i, (label, img) in enumerate(panels):
                x = i * PANEL_W
                canvas[0:PANEL_H, x:x + PANEL_W] = img
                _label(cv2, canvas, label, (x + 12, 30))
                if i:
                    canvas[0:PANEL_H, x:x + 2] = (70, 70, 70)

            canvas[PANEL_H:H, :] = strip
            frac = 0.0 if t1 <= t0 else (tt - t0) / (t1 - t0)
            cx = int(ax_x0 + frac * (ax_x1 - ax_x0))
            cv2.line(canvas, (cx, PANEL_H + int(0.06 * STRIP_H)),
                     (cx, PANEL_H + int(0.94 * STRIP_H)), (255, 255, 255), 1)

            err = np.degrees(np.abs(q_real[k] - q_sim[k]))
            _label(cv2, canvas,
                   f"t {tt - t0:6.2f} s    mean |q err| {err.mean():5.2f} deg"
                   f"    worst {err.max():5.2f} deg",
                   (12, PANEL_H - 14), scale=0.52)
            if title:
                _label(cv2, canvas, title, (12, PANEL_H - 38), scale=0.5)
            writer.write(canvas)
            written += 1
    finally:
        writer.release()
        if reader is not None:
            reader.release()

    return {"path": out_path, "frames": written, "fps": fps,
            "size": [W, H], "seconds": written / fps,
            "bytes": os.path.getsize(out_path) if os.path.exists(out_path) else 0,
            "metrics": metrics or {}}


def _fit(img, w, h):
    """Letterbox ``img`` into a ``w x h`` panel without distorting it."""
    cv2 = _cv2()
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = _BG
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def _label(cv2, canvas, text, org, scale: float = 0.6):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _FG, 1, cv2.LINE_AA)
