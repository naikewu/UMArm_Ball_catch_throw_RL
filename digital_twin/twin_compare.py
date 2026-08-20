r"""Roll a real recording through the twin, open loop, and score the difference.

The twin's report card, and its being open loop is the whole point: a
closed-loop comparison lets the twin's own controller correct the model's error
and report an agreement it did not earn.  Open loop, the same commands go into
the metal and into the model and the trajectories are allowed to diverge, which
is the only arrangement in which the divergence means something.

Scores worth keeping from the RS485 version: per-joint angle error over the
recording, per-actuator pressure error, and the ring statistics
(``ring_analysis``) applied IDENTICALLY to both -- the same bandpass, the same
episode finder, the same damped-sine fit -- because a comparison in which the
recording and the rollout are measured by different instruments is not one.

FOR THE CAN ARM, one addition the RS485 arm did not need: the recording carries
twenty-four boards of two variants, and the two populations have different valve
physics and different failsafes.  Scoring them together yields one number that
describes neither.  Split by the variant the board reported.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\twin_compare.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 24 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\twin_compare.py"

_TODO = (
    "digital_twin.twin_compare is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


def compare(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def main(argv=None):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)
