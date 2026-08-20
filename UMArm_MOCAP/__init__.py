"""UMArm motion-capture front end: Motive/NatNet poses -> joint vector ``q``.

Two layers, deliberately separable:

* :mod:`mocap_to_q` — the geometry.  Pure numpy, stateless, no network, no
  scipy; this is the part with a test oracle and the part other repos import.
* :mod:`mocap_rx` — the live stream.  Wraps the vendored NaturalPoint SDK in
  ``natnet_sdk/`` and maintains the latest ``q`` plus stream health and history.

Nothing in this package opens a socket until :meth:`mocap_rx.MocapRx.start` is
called, so importing it is safe anywhere, including in tests.

Added by this workspace, on top of the RS485 original:

* :mod:`canarm_mocap` — receivers bound to the CAN arm's block of Motive rigid
  bodies (briefed as 2000-2005, **not yet verified live**).  Which block a
  receiver claims is a per-instance property of :class:`mocap_rx.MocapRx`
  rather than a module constant, which is what lets two arms with different
  bases be received in one process against one NatNet stream.
* :mod:`sim_stream` — a synthetic producer that drives a real receiver's real
  listeners in-process, so the whole mocap -> ``q`` chain can be developed with
  no cameras and no UDP.

Neither is imported here: ``canarm_mocap`` pulls in the marker stack and
``sim_stream`` pulls in the probe, and this package's promise is that importing
it costs nothing.  Import them by name.

See ``docs/umarm_mocap_design.md`` for the design, ``natnet_sdk/PROVENANCE.md``
for where the third-party SDK came from.
"""

from __future__ import annotations

from . import mocap_constants
from .mocap_rx import MocapRx, MocapState, MocapWindow, RestCapture
from .mocap_to_q import (
    link_vectors,
    mocap_to_q,
    quat_xyzw_to_matrix,
    rotation_aligning,
    ujoint_angles,
    unit_from_ujoint_angles,
)

__all__ = [
    "mocap_constants",
    "mocap_to_q",
    "link_vectors",
    "ujoint_angles",
    "unit_from_ujoint_angles",
    "rotation_aligning",
    "quat_xyzw_to_matrix",
    "MocapRx",
    "MocapState",
    "MocapWindow",
    "RestCapture",
]
