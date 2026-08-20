"""UMArm forward kinematics: joint vector ``q`` -> plate poses and centres.

Two layers, deliberately separable (design ``docs/fkine_design.md`` §2):

* :mod:`robot_params` — the mk5 geometric parameter table, lifted from the
  legacy ``robot_constants.py``, plus the **authoritative**
  ``PLATE_CHAIN_NOMINAL_M`` chain nominals (design D8).  Pure numpy, no
  intra-repo imports, so the mocap probe can depend on it without dragging in
  anything else.
* :mod:`fkine` — the product-of-exponentials chain, oracle-equivalent to the
  legacy ``kinematics_mp.fkine_mk5``, plus the per-plate mounting model
  (:func:`fkine.plate_transforms`, design D6) that every synthetic-pose
  consumer must go through.

* :mod:`canarm_params` — the CAN arm's table, added by this workspace.  Same
  topology, **placeholder lengths**, and a ``MEASURED`` flag that is still
  ``False``.  It is a second table rather than an edit to ``DEFAULT_PARAMS``,
  which is the whole reason ``fkine`` takes ``params=None`` everywhere.

numpy only — no scipy anywhere (design D2).  This is the *forward* complement
of ``UMArm_MOCAP.mocap_to_q`` (mocap -> q); the round-trip between the two is
pinned in ``test_fkine.py``.  The offline hardware benchmark (design §4) is a
separate module and a later stage.
"""

from __future__ import annotations

from . import canarm_params, robot_params
from .fkine import (
    fkine,
    plate_transforms,
    predict_spatial,
    segment_transform,
    segment_twists,
    twist_exp,
    ujoint_centres,
)
from .robot_params import DEFAULT_PARAMS, PLATE_CHAIN_NOMINAL_M

__all__ = [
    "robot_params",
    "canarm_params",
    "DEFAULT_PARAMS",
    "PLATE_CHAIN_NOMINAL_M",
    "twist_exp",
    "segment_twists",
    "segment_transform",
    "fkine",
    "ujoint_centres",
    "plate_transforms",
    "predict_spatial",
]
