r"""Is the force law overtuned?  Five re-runnable measurements that change nothing.

An instrument, not a step in a pipeline: it reads the fitted model and reports
five independent measurements of whether the force law is doing work the rest of
the model should be doing.  It writes nothing back, which is what makes it safe
to run after every fit and honest when it disagrees with one.

Keep that property in the port.  An audit that can adjust what it audits stops
being evidence the first time somebody is in a hurry.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\force_audit.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 24 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\force_audit.py"

_TODO = (
    "digital_twin.force_audit is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


def audit(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def main(argv=None):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)
