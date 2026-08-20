r"""``SimMaster`` -- the twin's transport shell, shaped exactly like the real one.

The piece that makes "a controller cannot tell the twin from the arm" true at
the interface rather than only in the physics.  The RS485 version **subclasses**
its own ``fake_arm.FakeMaster``, so the TDMA grid and the fault plumbing are the
*same code* the calibration suites proved, rather than a second implementation
that agrees with the first until it does not.

THE CAN ARM'S EQUIVALENT IS ``TLE_PCB/tlelib/backend.Backend``, and the same
rule applies: subclass or wrap it, do not reimplement it.  Concretely --

* the 150 Hz cycle is table frames on 0x091-0x098, then ``0x090`` with DLC 0 as
  the sync edge, then replies gathered inside ``RX_WINDOW_FRAC = 0.82`` of the
  period;
* ``_build_targets`` addresses **every known board every cycle**, not just the
  selected ones, because the enable bit is a LEVEL: dropping a board from the
  table leaves it regulating with nothing able to reach it, and the sync-loss
  failsafe cannot help because the master is still sending edges.  A twin that
  tables only the selection cannot produce the real bus's worst failure;
* reply latency on the real 24-board bus runs 2.04 ms at 0x101 to 3.49 ms at
  0x118, about 65 us per id step, from CAN arbitration.  The 1.07 ms median in
  the TLE report is a ONE-BOARD measurement and adds to that spread rather than
  replacing it.  A twin that models one latency models a one-board bench.

The seam to fill is a ``CanLink``-shaped object whose frames reach modelled
boards instead of a slcan dongle, so that ``Backend`` itself is unmodified.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_master.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 8.7 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_master.py"

_TODO = (
    "digital_twin.sim_master is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


class SimMaster:
    """Not implemented.  See the module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_TODO)
