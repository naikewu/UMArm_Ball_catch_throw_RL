r"""``SimArm`` -- MuJoCo model and data, plus one firmware model per board.

The twin's centre.  It owns a compiled model, its data, twenty-four node models
standing in for the boards' control loops, and exactly ONE mutex, so that
``advance_to`` is callable from any thread and a caller can never observe a
half-stepped state.

TWO DECISIONS ARE LOAD-BEARING AND MUST SURVIVE THE PORT:

1. **``advance_to`` is grid-quantised, and the grid is 1 ms.**  Node logic runs
   every ``NODE_LOGIC_EVERY`` whole quanta, so the firmware's control period is
   expressed as an integer count of timesteps rather than as a duration.
   Halving the timestep therefore halves the modelled control period silently,
   with nothing raising and every rollout still looking plausible.  Read
   ``model.opt.timestep``; never assume it.
2. **It takes ``xml=``.**  A merged multi-robot scene is composed by somebody
   else -- ``viz.mjcf_canarm.build_room_scene`` for the display, and whatever
   the twin's own merge becomes -- and handed in.  The RS485 workspace states
   this at ``UMArm_COLLAB/collab_scene.py:1-7``: the merge exists *because*
   ``SimArm`` accepts a scene, rather than the arm being re-authored inside the
   merge.

WHAT CHANGES FOR THE CAN ARM.  The node model is not the mk8 node.  The top
eight boards run ``pressure_controller.c``'s DVP current-control law against
TLE92464 half-bridges -- 16-bit PID gains, dither, slew and flow shaping, a
120-code current budget of which 4 are reserved for the hardware dither overlay
-- and the bottom sixteen run ``Valve_not_embedded_XL`` with a 7 mm solenoid
pair.  So there are two node models, and which one a board gets is chosen from
the variant byte the board REPORTED, never from its id range: a TLE board
legitimately sat at 0x114 during the 2026-08 bench session, and reading it on
the 7 mm calibration is a 10 % error at the top of the range.

THE FAILSAFE IS ASYMMETRIC AND THE TWIN MUST REPRODUCE THAT.  A TLE board drops
its outputs 500 ms after the last sync edge; the sixteen legacy boards have no
link-loss timeout and hold their last commanded pressure indefinitely.  A twin
that models one rule for all twenty-four cannot show what a host stall actually
does to the arm.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_core.py``
"""

from __future__ import annotations

#: The file this module must be ported from, absolute.  Read it before writing
#: anything here; it is 24 kB of behaviour that was measured rather than
#: designed.
RS485_ORIGINAL = r"C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_core.py"

_TODO = (
    "digital_twin.sim_core is a documented skeleton.  Port it from "
    + RS485_ORIGINAL
    + " -- see this module's docstring for what changes for the CAN arm."
)


class SimArm:
    """Not implemented.  See the module docstring."""

    def __init__(self, *args, xml=None, **kwargs):
        # ``xml=`` is in the signature deliberately, even unimplemented:
        # it is the seam a merged multi-robot scene arrives through, and a
        # port that adds it later tends to add it wrongly.
        raise NotImplementedError(_TODO)

    def advance_to(self, t_s):
        raise NotImplementedError(_TODO)
