"""Frames, Motive IDs and geometry constants for the mocap -> ``q`` pipeline.

Every value here is *lifted*, not invented: the Motive rigid-body streaming IDs,
the rigid-body ordering and the 45 deg hardware offset all describe hardware and
a Motive project that already exist in the lab.  Each block therefore cites the
legacy file it came from, so that a future reader can diff against the source of
truth instead of guessing which side is stale.

Legacy sources (read-only reference repo ``UMArm_compliance_TRO``, branch
``main``, 2026-05):

* ``mocap_to_config/mocap_config_constants.py`` — sizes, ID mask, ``V_BASE``,
  ``RZn45``, ``MIN_LINK_NORM`` (itself distilled from ``kinematics_mp.py``,
  ``mocap_natnet_receiver_routine.py`` and ``robot_constants.py``).
* ``mocap_to_config/natnet_receiver.py`` — default server/client IPs and the
  multicast default.

The one thing that is *not* a constant is the joint zero: the raw angles depend
on how the marker plates happen to be screwed on, so a straight arm does not
read all zeros.  Treat ``q`` as repeatable, not absolute, and let calibration
own the offset (see ``MocapRx.capture_rest``).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------

#: 3 segments x 2 universal joints x 2 DOF.  Legacy: ``mc.NUM_JOINTS``.
NUM_JOINTS = 12

#: Rows in the pose array the receiver maintains.  Indices 0..7 are the arm's
#: plates, index 8 is the (optional) Kinova arm — kept so a pose array built for
#: the legacy stack drops straight in.  Legacy: ``mc.NUM_RIGID_BODIES``.
NUM_RIGID_BODIES = 9

#: The reconstruction consumes indices 0..6 only; :func:`mocap_to_q.mocap_to_q`
#: rejects a shorter array.  Legacy: ``mc.N_USED_RIGID_BODIES``.
N_USED_RIGID_BODIES = 7

# --------------------------------------------------------------------------
# Rigid-body index map  (legacy: mocap_config_constants.py header comment)
# --------------------------------------------------------------------------
# Motive streaming id = RIGID_BODY_ID_MASK + array index, i.e. 500 + index.
#
#   idx  id   plate                        what the math uses
#   ---  ---  ---------------------------  ----------------------------------
#    0   500  base plate                   position + ROTATION (base frame)
#    1   501  seg-1 distal U-joint centre  position
#    2   502  seg-2 proximal U-joint       position
#    3   503  seg-2 distal U-joint         position
#    4   504  seg-3 proximal U-joint       position
#    5   505  seg-3 distal U-joint         position + ROTATION (z = last link)
#    6   506  end-effector plate           position (carried, unused by mk8)
#    7   507  spare                        read into the array, unused
#    8  1008  Kinova arm                   not part of the UMArm q
#
# Index 5's *orientation* is load bearing: the sixth U-joint has no further
# centre to difference against, so its link direction is that plate's body z.

RIGID_BODY_ID_MASK = 500
KINOVA_MOCAP_STREAM_ID = 1008
KINOVA_RIGID_BODY_INDEX = 8

#: Array index of each named plate, for call sites that would otherwise carry
#: bare integers.
IDX_BASE = 0
IDX_U1_DISTAL = 1
IDX_U2_PROXIMAL = 2
IDX_U2_DISTAL = 3
IDX_U3_PROXIMAL = 4
IDX_U3_DISTAL = 5
IDX_END_EFFECTOR = 6

#: The six universal-joint centres, proximal to distal.  ``u_joint_positions``
#: in the :class:`mocap_rx.MocapRx` ring buffer is ``pos[U_JOINT_INDICES]``.
U_JOINT_INDICES = (0, 1, 2, 3, 4, 5)

# --------------------------------------------------------------------------
# Geometry  (legacy: mocap_config_constants.V_BASE / RZn45 / MIN_LINK_NORM)
# --------------------------------------------------------------------------

#: The robot base z-axis in base coordinates.  Universal-joint angles are read
#: after rotating the *previous* link onto this axis, so "+z" is the zero-ish
#: pose direction: the arm hangs base-up and link vectors (proximal minus
#: distal centre) point +z at rest.
V_BASE = np.array([0.0, 0.0, 1.0], dtype=float)

_pi = np.pi

#: -45 deg rotation about z.  Alternating universal joints are mounted 45 deg
#: apart, so the "even" joints (u2, u4, u6) need their de-rotated link vector
#: turned back into their own measurement frame before the angles are read.
#: Rotating the *frame* by +45 deg means rotating the *vector* by -45 deg, hence
#: the negative angle.  Written with the same expression shapes as the legacy
#: file so the matrix is bit-identical to the oracle's.
RZn45 = np.array([
    [np.cos(-_pi / 4.0), -np.sin(-_pi / 4.0), 0.0],
    [np.sin(-_pi / 4.0), np.cos(-_pi / 4.0), 0.0],
    [0.0, 0.0, 1.0],
], dtype=float)

#: Shortest link vector (metres) still treated as real data.  Below this the
#: direction is pure noise — an un-tracked plate leaves its pose at the
#: identity, so the difference of two centres collapses to zero — and the frame
#: is reported as "no data" (``None``) rather than normalised into garbage.
MIN_LINK_NORM = 1e-10

# --------------------------------------------------------------------------
# NatNet transport defaults  (legacy: natnet_receiver.NatNetConfigReceiver)
# --------------------------------------------------------------------------

#: Motive host PC.
DEFAULT_SERVER_IP = "192.168.1.100"
#: This PC, i.e. the interface Motive's stream is received on.  Must be the
#: address of the NIC on the camera network, not a VPN/Wi-Fi address, or the
#: multicast join silently attaches to the wrong interface.
DEFAULT_CLIENT_IP = "192.168.1.120"
#: The lab's Motive project streams multicast; unicast needs Motive changed too.
DEFAULT_USE_MULTICAST = True

# --------------------------------------------------------------------------
# Live-stream health  (new here; the legacy receiver had no staleness notion)
# --------------------------------------------------------------------------

#: Motive's usual streaming rate.  Only used to size buffers and to seed the
#: rate estimate — nothing assumes it.
NOMINAL_RATE_HZ = 120.0

#: No frame for this long means the stream is stale: at 120 Hz that is ~30
#: missed frames, far more than jitter, and short enough that a controller
#: notices before the arm has moved appreciably.
STALE_AFTER_S = 0.25

#: Ring-buffer depth: ~20 s of history at :data:`NOMINAL_RATE_HZ`.  Long enough
#: to hold a whole step-response or rest capture and pull it out afterwards,
#: small enough (12 floats + 18 floats per entry) to be free.
RING_SECONDS = 20.0
RING_CAPACITY = int(RING_SECONDS * NOMINAL_RATE_HZ)  # 2400
