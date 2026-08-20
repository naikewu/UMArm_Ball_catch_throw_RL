r"""The CAN arm's simulation MJCF -- the single source of the twin's geometry.

The display model in ``viz/mjcf_canarm.py`` answers "where are the links, given
q" and deliberately carries no actuators, no tendons and no fitted dynamics.
This one is the other half: the same kinematic chain plus the twenty-four muscle
tendons, the masses, and the small number of MuJoCo-side tunables the twin is
allowed to fit.  They are separate files because they fail differently -- a
display that is slightly wrong is a picture, a twin that is slightly wrong is a
controller tuned against a lie -- and because the display must stay cheap enough
to compile in a spawned process sixty times a second.

WHAT CHANGES FROM THE RS485 GENERATOR:

* **The chain is an argument, not a module constant.**  The RS485 file pins
  ``FITTED_CHAIN_M`` at module scope and exposes only base pose and three
  damping scalars, so a different-sized arm cannot be built without editing it.
  ``viz.mjcf_canarm.build_arm_xml`` already took the other road (``chain_m=``)
  and this generator must too, along with the 24-tendon ring table.
* **The lengths are unmeasured.**  ``UMArm_KINEMATICS.canarm_params.MEASURED``
  is ``False`` and its numbers are the RS485 arm's, unscaled, wearing this
  arm's name.  Fitting anything against them produces a shape, not a length.
* **The actuator ring azimuths must be re-derived.**  The RS485 table
  (``UMArm_ROBOT_CONTROL.joint_actuator_map.SEAT_DEG_BY_LOCAL_ID``, three
  independent derivations that superseded a legacy note carrying a 180-degree
  confusion) is what makes ``pam_i`` be node ``i``'s muscle.  The CAN arm's
  board ids are 0x101-0x118 and its seating is its own question; assuming the
  RS485 table transfers is assuming the manifold was plumbed the same way.
* **The timestep stays 1 ms** -- see ``digital_twin.TIMESTEP_S`` and design
  decision 1 in the package docstring.

WHAT TRANSFERS UNCHANGED: the topology (three segments, universal joints as two
stacked hinges at one origin, distal axes ``(+-1, 1, 0)/sqrt(2)``, straight
two-site tendons), which is the same mechanism, and the practice of citing the
provenance of every fitted number inline beside it.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\mjcf_generator.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 27 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\mjcf_generator.py"

_TODO = (
    "digital_twin.mjcf_generator is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


def generate_xml(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def generate_scene(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)


def fitted_params(*args, **kwargs):
    """Not implemented.  See the module docstring."""
    raise NotImplementedError(_TODO)
