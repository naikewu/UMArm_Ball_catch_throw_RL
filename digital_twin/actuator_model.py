r"""The valve-and-muscle model: flow in, force out.  **Do not inherit the RS485 fit.**

Three pieces in the RS485 original, and the split is worth keeping: a small
neural flow network (2x32 tanh, plain numpy, hand-written and gradient-checked
backward pass), an anchored McKibben force law, and per-node scalars for fill
gain, vent gain and leak.  Plus the trainers that fit them.

THE FIT DOES NOT TRANSFER, AND THIS IS THE MOST IMPORTANT SENTENCE IN THIS
PACKAGE.  The RS485 checkpoint was fitted to a 7 mm solenoid pair switched by a
PWM node: flow is set by a duty cycle against a fixed orifice.  The CAN arm's
top eight boards are TLE92464 proportional valves under current control -- flow
is set by a coil current of 0 to 182.7 mA (host codes clamped to 116 of a
120-code budget, ``I_mA = code * 200/127``), shaped by an inlet/outlet pair with
their own open and max codes, a slew limit, a dead zone and an optional hardware
dither.  The two plants share a bladder and share nothing else.  A checkpoint
from one loaded into the other fits the trajectory it was fitted on and diverges
everywhere else, which is the failure mode that looks most like success.

WHAT DOES TRANSFER: the anchored force law's *shape* (the same braid on the same
bladder), the practice of keeping per-node scalars rather than one global gain
(the boards are not identical, and pretending they are pushes the difference
into the flow net where it is invisible), and the gradient check -- a
hand-written backward pass that is never checked is a plausible-looking model of
the wrong function.

Fit against data collected with the schema in ``data_schema.md``, not against
data collected in whatever shape was convenient at the time.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\actuator_model.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 52 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\actuator_model.py"

_TODO = (
    "digital_twin.actuator_model is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


class ActuatorModel:
    """Not implemented.  See the module docstring."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_TODO)


def load_checkpoint(path):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def train(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)
