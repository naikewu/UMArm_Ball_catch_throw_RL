r"""``SimMocap`` -- plate poses from the twin, pushed through the REAL receiver.

A producer thread that streams the simulated plate poses at 120 Hz through the
shipped ``MocapRx`` machinery, by subclassing it, so that a ``q`` obtained from
the twin has genuinely been through the id routing, the quaternion conversion,
``mocap_to_q``, the publish-under-lock and the ring buffer.  A twin that hands a
controller a ``q`` it computed itself has silently removed the entire mocap
pipeline from the thing under test.

**THIS ONE IS ALREADY HALF-DONE, ELSEWHERE.**  ``UMArm_MOCAP/sim_stream.py``
(``CanArmSimStream``) is exactly this idea for the CAN arm's block of rigid
bodies: a real ``CanArmMocap`` fed by a producer thread computing plate poses
from a ``q(t)``, opening no socket.  What it does not have is a physics source
-- its ``q_of_t`` is an analytic sweep.  The port here is to hand it
``sim_core``'s ``q`` instead of ``sweep_q``, which is a constructor argument
that already exists.

So this module is small and its docstring is most of it.  Resist re-solving the
injection problem: ``sim_stream.inject_frame`` already respects a receiver's
``rb_id_base``, so a receiver bound to the wrong block sees nothing, exactly as
it would live.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_mocap.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 20 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_mocap.py"

_TODO = (
    "digital_twin.sim_mocap is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


class SimMocap:
    """Not implemented.  See the module docstring -- and read
    ``UMArm_MOCAP/sim_stream.py`` first, which is most of it."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_TODO)
