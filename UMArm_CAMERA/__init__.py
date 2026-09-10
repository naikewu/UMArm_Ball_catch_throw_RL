"""Bench camera capture for the CAN arm.

One mirrorless camera looks at the arm through a USB capture card, enumerating
as a 1920x1080 UVC device.  This package owns the two things a data-collection
campaign needs from it: a background grabber that never blocks the 150 Hz
control loop, and a ring buffer deep enough that a clip can start *after* the
event that motivated it.

The second property is the one that matters.  An anomaly -- a commanded
pressure that moves no joint -- is only recognised once the pressure has
already been held for a second or two, by which point a recorder that started
on the trigger would have missed the whole thing.  :class:`BenchCamera` keeps
the last :data:`PRE_ROLL_S` seconds resident, so ``clip()`` writes the past as
well as the future.
"""

from .camera import (
    BenchCamera,
    CameraState,
    CAMERA_INDEX,
    CLIP_FOURCC,
    CLIP_FPS,
    CLIP_SIZE,
    ARM_CROP,
    CONTENT_CROP,
    PRE_ROLL_S,
    probe_cameras,
    silence_opencv_logging,
)

__all__ = [
    "BenchCamera",
    "CameraState",
    "CAMERA_INDEX",
    "CLIP_FOURCC",
    "CLIP_FPS",
    "CLIP_SIZE",
    "ARM_CROP",
    "CONTENT_CROP",
    "PRE_ROLL_S",
    "probe_cameras",
    "silence_opencv_logging",
]
