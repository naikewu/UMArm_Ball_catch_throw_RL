r"""Fit the dissipation scalars to a recording's ring episodes.

Four numbers, fitted against the episodes ``ring_analysis`` finds: the spine's
viscous damping and its Coulomb friction, and the muscle damping schedule's
intercept and pressure slope.  Small, and the smallness is deliberate -- a
dissipation model with many knobs fits any ringdown and predicts none.

THE LESSON THE RS485 FIT PAID FOR, worth carrying whatever the CAN arm's numbers
turn out to be: dissipation belongs where the physics puts it.  Its previous
values (joint damping 2.25, frictionloss 0.70) were engineering guesses a 20 Hz
system-identification campaign could never see past, and the 0.7 N.m Coulomb
floor alone exceeded the peak elastic torque of a one-degree oscillation -- so
the twin could not oscillate at all.  Moving the dissipation into
pressure-scheduled damping of the muscle itself, leaving only small spine terms,
is what made the measured rings reproducible.

There is a known cost, recorded rather than hidden: with the small friction the
model has no deflated-muscle passive elasticity to hold the chain during
single-muscle drives, and that campaign regressed from 12/12 to 8/12.  The metal
has that elasticity; the model does not.  Do not restore a ring-killing friction
term to paper over it.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\fit_bounce.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 19 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\fit_bounce.py"

_TODO = (
    "digital_twin.fit_bounce is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


def fit(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def main(argv=None):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)
