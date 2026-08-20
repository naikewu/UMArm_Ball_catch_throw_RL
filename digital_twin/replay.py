r"""Offline rollouts: no threads, no wall clock, bit-identical repeats.

The instrument every fit and every regression runs through.  It drives
``sim_core`` from a recorded command sequence with the wall clock removed
entirely, so two runs of the same inputs produce the same floats -- which is
what makes a change to the model visible as a diff rather than as a difference
in scheduling.

The determinism is the feature, and it is easy to lose: one
``time.monotonic()`` in a control path, one dict iteration order that leaks into
a sum, one thread, and a regression becomes a thing that reproduces four times
in five.

FOR THE CAN ARM the command sequence is the runtime target table -- eight frames
of three actuators each, then the sync edge on ``0x090`` with DLC 0 -- rather
than the RS-485 node writes.  Its recorded form is the JSONL of
``data_schema.md``.  The replay must also honour the two-population failsafe
(see ``sim_core``), because a recording that contains a host stall contains the
moment the top eight released and the bottom sixteen did not.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\replay.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 15 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\replay.py"

_TODO = (
    "digital_twin.replay is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


def rollout(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def load_recording(path):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)
