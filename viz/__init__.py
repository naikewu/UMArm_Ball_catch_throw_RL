"""Display-only visualisation of the room: the CAN arm, and whoever else is in it.

Nothing in this package commands a robot.  It builds an MJCF of whatever the
room currently holds, writes joint angles and mount poses into it every frame,
calls :func:`mujoco.mj_forward`, and draws.  The distinction is not stylistic:
in the RS485 workspace the 3D window *is* the plant — closing it vents the arm —
and inheriting that rule here would mean opening a display arms a shutdown path
nobody asked for.  Closing this window closes a window.

Four modules, deliberately separable:

``transforms``
    The three rotation conversions the rest of the package needs, written out so
    a scene builds on a clone carrying nothing but numpy.
``viz_layout``
    The flat block of doubles a producer writes and the viewer reads.  Backend
    and viewer agree on the layout by *importing* it, never by being edited at
    the same time.
``base_poses``
    Where a robot's base pose comes from — sliders today, cameras on the rig —
    behind one interface that returns a name-keyed map of poses.
``mjcf_canarm``
    The CAN arm's display model and the room scene that merges the optional
    robots into it.
``multi_arm_viewer``
    The passive viewer process, its feeds, and the measured-plate overlay.
"""

from __future__ import annotations

__all__ = ["transforms", "viz_layout", "base_poses", "mjcf_canarm",
           "multi_arm_viewer"]
