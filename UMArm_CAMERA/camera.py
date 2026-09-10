"""Threaded bench-camera grabber with a pre-roll ring buffer.

The camera is a mirrorless body feeding a USB capture card, so it enumerates as
an ordinary 1920x1080 UVC device and OpenCV can drive it.  Two properties of
that path shape this module.

First, ``VideoCapture.read()`` blocks for a frame period.  Calling it from the
campaign's loop would couple a 60 Hz USB device to a 150 Hz control cycle, so
the grab runs on its own thread and the loop only ever touches a lock-protected
deque.

Second, the camera overlays its own viewfinder OSD -- exposure, aperture, ISO --
in a letterbox around the picture.  :data:`CONTENT_CROP` names the picture
inside that letterbox, measured from a probe frame on 2026-09-10, so clips carry
the arm rather than the camera's status bar.

Clips are written by a second thread from a queue, since ``VideoWriter.write``
is an encode and must not sit in the grab path either.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:  # cv2 is optional so the campaign can run camera-free on a bench without one
    import cv2
except Exception:  # pragma: no cover - exercised only on a machine without cv2
    cv2 = None


#: Index of the capture card as OpenCV enumerates it.  Probed 2026-09-10:
#: index 0 is the only device that opens, at 1920x1080 and 59.94 fps.
CAMERA_INDEX = 0

#: Native capture size requested from the device.
CAPTURE_SIZE = (1920, 1080)

#: The whole picture inside the camera's viewfinder letterbox, ``(x0, y0, x1,
#: y1)`` in native pixels.  Measured from the 2026-09-10 probe frame: the
#: picture spans x 260-1660 and y 24-936, with the exposure/ISO status bar
#: below it.  This is the *room* view, kept for context shots.
CONTENT_CROP = (260, 24, 1660, 936)

#: The CAN arm alone, ``(x0, y0, x1, y1)`` in native pixels, and the crop every
#: clip actually uses.  **The room holds two ceiling-hung UMArms**: the RS485
#: sister arm stands against the white backdrop on the right of frame and the
#: Kinova Gen3 sits below it, so a clip of the whole room is ambiguous about
#: which robot moved.  The window was located by driving ``0x101`` alone and
#: taking the bounding box of the changed pixels (``hw_tests/canarm_hello.py``,
#: 2026-09-10), then widened to hold a full-range swing: the arm's 0.87 m of
#: chain covers 255 px, so +-0.3 m of tip travel is +-88 px and this window
#: leaves 120 px on each side of the resting chain.
ARM_CROP = (700, 150, 1220, 830)

#: Size clips are encoded at -- three quarters of :data:`ARM_CROP`, which keeps
#: every plate and marker bracket resolvable while holding a 15 s clip near
#: 5 MB.  Portrait, because the arm is.
CLIP_SIZE = (390, 510)

#: Clip frame rate.  The camera delivers ~60 fps; the arm's ringdown is a few
#: Hz, so 15 fps still oversamples the motion by 5x and cuts the file to a
#: quarter of the native rate.
CLIP_FPS = 15.0

#: FourCC for the writer.  This bench has **no H.264 encoder**: OpenCV's FFmpeg
#: build defers H.264 to ``openh264-2.5.0-win64.dll``, which is absent, and the
#: machine has no route to fetch it, so ``avc1`` fails to initialise inside a
#: worker thread and silently costs a clip.  ``mp4v`` (MPEG-4 Part 2) is
#: therefore the default rather than the fallback: roughly 3x the bytes of
#: H.264 for the same picture, which is what :data:`ARM_CROP` buys back.
CLIP_FOURCC = "mp4v"

#: Consecutive failed grabs after which the device is treated as gone.  The
#: bench camera is a mirrorless body, and on 2026-09-10 it dropped out 298 s
#: into a session with ``MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`` -- the body
#: had gone to sleep.  The first version of this module retried every 5 ms and
#: OpenCV logged three lines per attempt, so a sleeping camera produced 4882
#: stderr lines and a thread spinning at 200 Hz against a 150 Hz control cycle.
#: Sixty failures is one second of a dead device, which is long enough that a
#: transient USB hiccup is not mistaken for a dropout.
GRAB_FAIL_LIMIT = 60

#: Seconds to wait before the single reopen attempt, and how long a reopened
#: device is given to deliver a frame.  One attempt, not a loop: a body that
#: has powered itself off will not come back until someone touches it, and a
#: reopen loop is the same busy-wait in slower clothing.
REOPEN_DELAY_S = 2.0

#: Seconds of history the ring buffer holds, so a clip triggered by an anomaly
#: still contains the moments before it was recognised.  The ring stores frames
#: already cropped and downscaled to :data:`CLIP_SIZE` and decimated to
#: :data:`CLIP_FPS`, so 8 s costs 8 x 15 x 390 x 510 x 3 B = 72 MB -- affordable,
#: and the decimation is what makes it so.
PRE_ROLL_S = 8.0


@dataclass
class CameraState:
    """What the grabber has done so far, for a status line or an assertion."""

    frames: int = 0
    dropped: int = 0
    fps: float = 0.0
    last_grab_monotonic: float = 0.0
    opened: bool = False
    clips_written: int = 0
    error: str = ""
    #: Names of clips finished so far, newest last.
    clip_paths: list = field(default_factory=list)
    #: True once the device stopped answering and the reopen did not help.
    #: The campaign keeps recording; it just stops getting pictures.
    lost: bool = False
    #: Consecutive grab failures at the moment the state was read.
    consecutive_failures: int = 0
    #: How many times the device was reopened over the session.
    reopens: int = 0


def silence_opencv_logging() -> bool:
    """Stop OpenCV writing to stderr from C++.  True if it took.

    Worth its own function because the name moved: OpenCV 5 exposes it as
    ``cv2.utils.logging.setLogLevel`` and has no module-level ``setLogLevel``,
    so the obvious call fails with ``AttributeError`` -- and wrapped in a
    try/except, that failure is invisible while the log keeps flooding.
    """
    if cv2 is None:
        return False
    try:
        from cv2.utils import logging as _cvlog
        _cvlog.setLogLevel(_cvlog.LOG_LEVEL_SILENT)
        return True
    except Exception:
        pass
    try:
        cv2.setLogLevel(0)          # OpenCV 4 spelling
        return True
    except Exception:
        return False


def probe_cameras(max_index: int = 6) -> list:
    """Enumerate openable capture devices without leaving one open.

    Returns a list of ``(index, backend_name, width, height, fps)``.  Used by
    the acceptance test and by :func:`BenchCamera.__init__` when its configured
    index fails, so a replugged capture card is a warning rather than a stop.
    """
    if cv2 is None:
        return []
    found = []
    backends = ((cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"))
    for idx in range(max_index):
        for backend, name in backends:
            cap = cv2.VideoCapture(idx, backend)
            try:
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                found.append((idx, name,
                              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              float(cap.get(cv2.CAP_PROP_FPS))))
                break  # one working backend per index is enough
            finally:
                cap.release()
    return found


class BenchCamera:
    """Background grabber with a pre-roll ring buffer and queued clip writing.

    ``start()`` opens the device and spawns two threads: a grabber that pushes
    cropped, downscaled, monotonic-stamped frames into a bounded deque, and a
    writer that drains a clip queue.  Neither ever blocks the caller.

    The class is usable as a context manager, and ``stop()`` is idempotent, so
    every exit path in a campaign closes the device exactly once.
    """

    def __init__(self, index: int = CAMERA_INDEX, *, crop=ARM_CROP,
                 size=CLIP_SIZE, fps: float = CLIP_FPS,
                 pre_roll_s: float = PRE_ROLL_S, fourcc: str = CLIP_FOURCC,
                 enabled: bool = True):
        self.index = index
        self.crop = crop
        self.size = size
        self.fps = fps
        self.pre_roll_s = pre_roll_s
        self.fourcc = fourcc
        #: ``False`` runs the whole campaign camera-free; every method becomes a
        #: no-op and :attr:`state` reports ``opened=False``.  A bench without a
        #: capture card must still be able to collect data.
        self.enabled = enabled and cv2 is not None

        self._cap = None
        self._ring = deque(maxlen=max(2, int(round(pre_roll_s * fps)) + 4))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._grab_thread = None
        self._write_thread = None
        self._clip_q: "queue.Queue" = queue.Queue()
        self._state = CameraState()
        #: Set while a clip is being *extended* live; the grabber appends to it.
        self._active_clip = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, timeout_s: float = 5.0) -> bool:
        """Open the device and start grabbing.  Returns whether frames arrived.

        A camera that does not open is reported, not raised: the campaign's
        primary product is the JSONL recording, and losing the video must not
        lose the data.
        """
        if not self.enabled:
            return False
        if self._grab_thread is not None:
            return self._state.opened
        try:
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._state.error = f"camera index {self.index} would not open"
                self._cap = None
                self.enabled = False
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
            # A one-frame internal buffer keeps the ring's newest frame recent;
            # the default depth would hand us frames a couple of periods stale.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - device-specific
            self._state.error = f"camera open failed: {exc!r}"
            self.enabled = False
            return False

        # A device fault must not be able to flood the log: OpenCV writes three
        # lines per failed grab straight to stderr from C++, and a redirected
        # stderr makes each of those a blocking write.  The call lives in
        # ``cv2.utils.logging`` in OpenCV 5 -- ``cv2.setLogLevel`` does not
        # exist there, and a try/except around the wrong name silences the
        # silencer instead of the log.
        silence_opencv_logging()

        self._stop.clear()
        self._grab_thread = threading.Thread(target=self._grab_loop,
                                             name="bench-cam-grab", daemon=True)
        self._write_thread = threading.Thread(target=self._write_loop,
                                              name="bench-cam-write", daemon=True)
        self._grab_thread.start()
        self._write_thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._state.frames > 5:
                self._state.opened = True
                return True
            time.sleep(0.05)
        self._state.error = "camera opened but delivered no frames"
        return False

    def stop(self) -> None:
        """Stop grabbing, finish every queued clip, release the device."""
        if self._grab_thread is None and self._cap is None:
            return
        self.end_clip()
        self._stop.set()
        for t in (self._grab_thread, self._write_thread):
            if t is not None:
                t.join(timeout=10.0)
        self._grab_thread = None
        self._write_thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._state.opened = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- state -------------------------------------------------------------- #

    @property
    def state(self) -> CameraState:
        with self._lock:
            st = CameraState(frames=self._state.frames,
                             dropped=self._state.dropped,
                             fps=self._state.fps,
                             last_grab_monotonic=self._state.last_grab_monotonic,
                             opened=self._state.opened,
                             clips_written=self._state.clips_written,
                             error=self._state.error,
                             clip_paths=list(self._state.clip_paths))
        return st

    def latest_frame(self):
        """The newest frame, or ``None``.  Copied, so the caller may draw on it."""
        with self._lock:
            if not self._ring:
                return None
            return self._ring[-1][1].copy()

    # -- clips -------------------------------------------------------------- #

    def clip(self, path: str, *, pre_s: float = 0.0, post_s: float = 0.0,
             label: str = "") -> bool:
        """Write a clip covering ``pre_s`` of history and ``post_s`` of future.

        ``pre_s`` is clamped to what the ring actually holds.  When ``post_s``
        is zero the clip is written immediately from history alone; otherwise
        the grabber keeps appending until the post-roll elapses and the writer
        thread encodes it then.  Returns ``False`` when the camera is absent.
        """
        if not self.enabled or not self._state.opened:
            return False
        pre_s = max(0.0, min(pre_s, self.pre_roll_s))
        now = time.monotonic()
        with self._lock:
            head = [(t, f) for (t, f) in self._ring if t >= now - pre_s]
        if post_s <= 0.0:
            self._clip_q.put((path, head, label))
            return True
        self._active_clip = {"path": path, "frames": head, "label": label,
                             "until": now + post_s}
        return True

    def end_clip(self) -> None:
        """Finish an open-ended clip early and queue it for encoding."""
        active = self._active_clip
        if active is None:
            return
        self._active_clip = None
        self._clip_q.put((active["path"], active["frames"], active["label"]))

    def drain(self, timeout_s: float = 60.0) -> None:
        """Block until every queued clip is on disk."""
        if not self.enabled:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._clip_q.unfinished_tasks == 0 and self._active_clip is None:
                return
            time.sleep(0.1)

    # -- threads ------------------------------------------------------------ #

    def _reopen(self) -> bool:
        """One attempt to bring the device back, after it stopped answering.

        Returns whether it delivered a frame.  On failure the camera is marked
        lost and every method becomes a no-op: the campaign's primary product
        is the JSONL recording, and losing the video must not lose the data.
        """
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        time.sleep(REOPEN_DELAY_S)
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            try:
                cap = cv2.VideoCapture(self.index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self._cap = cap
                        with self._lock:
                            self._state.reopens += 1
                        return True
                cap.release()
            except Exception:
                pass
        return False

    def _grab_loop(self) -> None:
        x0, y0, x1, y1 = self.crop
        fails = 0
        n_since = 0
        t_since = time.monotonic()
        # The camera runs at ~60 fps but clips are written at CLIP_FPS, so the
        # ring is decimated at grab time.  Decimating here rather than at the
        # writer is what keeps the pre-roll's memory bounded.
        period = 1.0 / max(1e-6, self.fps)
        next_keep = 0.0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception as exc:  # pragma: no cover - device-specific
                ok, frame = False, None
                with self._lock:
                    self._state.error = f"grab failed: {exc!r}"
            now = time.monotonic()
            if not ok or frame is None:
                fails += 1
                with self._lock:
                    self._state.dropped += 1
                    self._state.consecutive_failures = fails
                if fails < GRAB_FAIL_LIMIT:
                    # Back off as failures accumulate rather than hammering the
                    # device: a retry every 5 ms is a 200 Hz thread competing
                    # with a 150 Hz control cycle for the GIL, which is the
                    # opposite of what a camera fault should cost.
                    time.sleep(min(0.2, 0.005 * fails))
                    continue
                if self._reopen():
                    fails = 0
                    with self._lock:
                        self._state.error = "device reopened after a dropout"
                        self._state.consecutive_failures = 0
                    continue
                with self._lock:
                    self._state.lost = True
                    self._state.opened = False
                    self._state.error = (
                        f"camera stopped answering after {self._state.frames} "
                        f"frames and did not reopen; the session continues "
                        f"without video")
                self.enabled = False
                self.end_clip()
                return
            fails = 0
            if now < next_keep:
                continue
            next_keep = max(now, next_keep + period)
            try:
                cropped = frame[y0:y1, x0:x1]
                small = cv2.resize(cropped, self.size,
                                   interpolation=cv2.INTER_AREA)
            except Exception:  # pragma: no cover - malformed frame
                with self._lock:
                    self._state.dropped += 1
                continue
            with self._lock:
                self._ring.append((now, small))
                self._state.frames += 1
                self._state.last_grab_monotonic = now
                n_since += 1
                if now - t_since >= 1.0:
                    self._state.fps = n_since / (now - t_since)
                    n_since = 0
                    t_since = now
            active = self._active_clip
            if active is not None:
                active["frames"].append((now, small))
                if now >= active["until"]:
                    self.end_clip()

    def _write_loop(self) -> None:
        while not self._stop.is_set() or not self._clip_q.empty():
            try:
                path, frames, label = self._clip_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._encode(path, frames, label)
                with self._lock:
                    self._state.clips_written += 1
                    self._state.clip_paths.append(path)
            except Exception as exc:  # pragma: no cover - encoder-specific
                with self._lock:
                    self._state.error = f"clip write failed: {exc!r}"
            finally:
                self._clip_q.task_done()

    def _encode(self, path: str, frames, label: str) -> None:
        if not frames:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # The stamps are written beside the clip and they are not optional.
        # A grabber that decimates a jittery 60 fps source to a nominal 15
        # produces a file whose frame index is only approximately its time, and
        # a side-by-side that assumed otherwise would drift by a frame every few
        # seconds -- which reads as the twin lagging the arm, a claim the
        # recording does not support.  Written as JSON rather than .npy so the
        # file is readable without numpy and by anything.
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*self.fourcc),
                                 self.fps, self.size)
        if not writer.isOpened():
            # Not a formality: ``avc1`` reports a usable writer from the main
            # thread on this bench and then fails to initialise from the
            # encoder thread, so a clip is lost unless the fallback is real.
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, self.size)
        t0 = frames[0][0]
        try:
            for t, frame in frames:
                out = frame
                if label:
                    out = frame.copy()
                    cv2.putText(out, f"{label}  t={t - t0:6.2f}s",
                                (8, self.size[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(out, f"{label}  t={t - t0:6.2f}s",
                                (8, self.size[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
                writer.write(out)
            with open(os.path.splitext(path)[0] + ".stamps.json", "w",
                      encoding="utf-8") as fh:
                json.dump({"t_mono_s": [float(t) for t, _ in frames],
                           "fps_nominal": self.fps,
                           "size": list(self.size),
                           "crop": list(self.crop),
                           "label": label,
                           "frames": len(frames)}, fh)
        finally:
            writer.release()
