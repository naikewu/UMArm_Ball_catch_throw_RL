"""The CAN arm's digital twin — **a documented skeleton, not an implementation**.

Every module in this package raises :class:`NotImplementedError` and carries two
things worth more than the code that is missing: the role its RS485 counterpart
plays, rewritten for this arm's bus and firmware, and an absolute path to that
counterpart so the port is a read rather than a search.

WHY THE STRUCTURE IS MIRRORED AND THE CODE IS NOT.  ``UMArm_SIM`` cannot be
lifted.  ``sim_core``, ``actuator_model``, ``sim_master``, ``sim_mocap`` and
``replay`` all import ``UMArm_ROBOT_CONTROL.{arm_constants, fake_arm,
fake_mocap}`` at **module level**, and those three are precisely the files a
different bus (CAN at 1 Mbit/s, not RS-485), a different firmware (ESP-IDF
``VEMA_MAX22200`` and ``Valve_not_embedded_XL``, not the mk8 node) and a
different valve (TLE92464 / DVP proportional, not the 7 mm solenoid pair)
invalidate.  Mirroring the layout keeps the design decisions; copying the code
would keep the assumptions.

THE GOAL, unchanged from the RS485 design document and worth restating because
every module decision follows from it: *a controller must not be able to
discriminate between the real robot and the simulator*.  Same interface, same
timing, same node behaviour.  Anything that makes the twin easier to write at the
cost of that property is not a shortcut, it is a different project.

TWO DESIGN DECISIONS ARE CARRIED IN FROM THE RS485 TWIN because they were paid
for and are not re-derivable from the code that will replace them:

1. **The 1 ms timestep is locked.**  ``sim_core`` counts node-logic passes in
   whole quanta (``NODE_LOGIC_EVERY``), so halving the timestep silently halves
   the firmware's control period.  Anything discretising a lag or a dissipation
   must read ``model.opt.timestep`` rather than assume it.
2. **``sim_core`` must take ``xml=``.**  A merged multi-robot scene — the CAN
   arm, the RS485 arm and the Gen3 in one room — is composed elsewhere
   (``viz.mjcf_canarm.build_room_scene`` does the display version today) and
   handed in.  Re-authoring the arm inside the merge is how the two copies drift.

WHAT MUST NOT BE INHERITED: the RS485 arm's **actuator fits**.  Its
``actuator_model`` carries a flow network and a McKibben force law fitted to a
7 mm solenoid pair driven by a PWM node.  The CAN arm's top eight boards are
TLE92464 proportional valves under a DVP current-control law with its own gains,
dither and slew shaping; the bottom sixteen are the legacy 7 mm boards on the
same bus.  A fitted checkpoint from one plant loaded into the other produces a
twin that is confidently wrong and reports a good residual on the trajectory it
was fitted on.  See ``README.md`` for the collection-then-fit plan that replaces
inheriting them.

RS485 originals: ``C:\\RUNZE_SRC\\RS485_VEMA\\UMArm_SIM\\``, design document
``C:\\RUNZE_SRC\\RS485_VEMA\\docs\\simulator_design.md``.
"""

from __future__ import annotations

#: Physics quantum, seconds.  See design decision 1 in the module docstring.
#: Declared here rather than in ``mjcf_generator`` so that a caller can assert
#: on it without building a scene.
TIMESTEP_S = 0.001

#: The RS485 twin, for the port.  Absolute, because this workspace and that one
#: are different repositories on the same machine and neither is inside the
#: other.
RS485_SIM_DIR = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM"
RS485_DESIGN_DOC = r"C:\RUNZE_SRC\RS485_VEMA\docs\simulator_design.md"

__all__ = ["TIMESTEP_S", "RS485_SIM_DIR", "RS485_DESIGN_DOC"]
