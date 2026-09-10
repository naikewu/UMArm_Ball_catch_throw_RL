"""The pressure envelope, and the one place it is enforced.

The operator's rule for this arm is two inequalities: no single line above
30 psi, and no antagonistic pair whose two pressures sum above 30 psi.  The
second is the binding one — a pair is two muscles pulling against each other
across one universal joint, and their sum is what loads the joint bearing and
the plate hardware whether or not either line is individually modest.

Everything a campaign puts on the wire passes through :class:`PairEnvelope`.
It clamps, and then :meth:`assert_safe` checks the clamped vector again
immediately before transmission.  The second check is not redundant: the clamp
is arithmetic on a generated vector, the assertion is a guard against a code
path that skipped the clamp, and those are different failures.  A campaign that
could reach the bus without passing the assertion is a defect regardless of
whether it happens to generate safe numbers today.

The floor is not zero.  A board commanded to exactly zero holds its exhaust
valve open continuously, which wears the valve and heats the driver for no
benefit, and — the reason that actually matters here — ``0x110`` leaks from its
supply side, so a board that stops regulating keeps filling.  Every board is
held enabled at :data:`IDLE_PSI` instead, which parks the loop just off the
exhaust stop while venting that leak continuously.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS not in sys.path:
    sys.path.insert(0, _WS)

from UMArm_KINEMATICS import canarm_actuators as ACT  # noqa: E402

#: Hard ceiling on any single commanded pressure, psi.  The operator's rule.
SINGLE_MAX_PSI = 30.0

#: Hard ceiling on the sum of an antagonistic pair's two commanded pressures,
#: psi.  The operator's rule, and the binding one of the two.
PAIR_SUM_MAX_PSI = 30.0

#: What a campaign should generate to, leaving the hard ceiling as a guard
#: rather than as an operating point.  Two psi of headroom covers the two boards
#: whose sensors read high at rest (``0x110`` +5.1 psi, ``0x104`` +1.1 psi): a
#: board that reads high regulates to a *lower* true pressure than commanded, so
#: the offset is safe in sign, but generating flush against the ceiling would
#: leave nothing for a calibration that drifts the other way.
WORKING_PAIR_SUM_PSI = 28.0

#: Idle setpoint for a board that is not being excited, psi.  See the module
#: docstring for why it is not zero.
IDLE_PSI = 0.5

#: Boards, proximal to distal, as ``base`` CAN ids.
ALL_BASES = tuple(range(0x101, 0x119))

#: ``base -> index into a 24-vector``.  The vector's order is the id order,
#: which is also the order the collection schema's ``ids`` array uses.
BASE_INDEX = {b: i for i, b in enumerate(ALL_BASES)}


class PairEnvelope:
    """Clamp and check a 24-vector of commanded pressures.

    The pair table comes from :func:`UMArm_KINEMATICS.canarm_actuators.joint_pairs`,
    which returns the **measured** map and refuses to return the legacy one.
    That refusal is load-bearing here: the legacy table describes the top
    regulator platform as it was wired before 2026-08, and clamping against the
    wrong pairing would enforce the rule on pairs that are not pairs while
    leaving the real ones unbounded.
    """

    def __init__(self, *, pair_sum_max_psi: float = WORKING_PAIR_SUM_PSI,
                 single_max_psi: float = SINGLE_MAX_PSI,
                 idle_psi: float = IDLE_PSI):
        if pair_sum_max_psi > PAIR_SUM_MAX_PSI:
            raise ValueError(
                f"pair_sum_max_psi={pair_sum_max_psi} exceeds the operator's "
                f"{PAIR_SUM_MAX_PSI} psi rule for this arm")
        if single_max_psi > SINGLE_MAX_PSI:
            raise ValueError(
                f"single_max_psi={single_max_psi} exceeds the operator's "
                f"{SINGLE_MAX_PSI} psi rule for this arm")
        self.pair_sum_max_psi = float(pair_sum_max_psi)
        self.single_max_psi = float(single_max_psi)
        self.idle_psi = float(idle_psi)
        self.pairs = ACT.joint_pairs()
        #: ``(12, 2)`` of indices into a 24-vector, joint order.
        self.pair_idx = np.array(
            [[BASE_INDEX[pos], BASE_INDEX[neg]] for pos, neg in self.pairs],
            dtype=int)
        seen = sorted(int(i) for row in self.pair_idx for i in row)
        if seen != list(range(24)):
            raise RuntimeError(
                "the measured actuator map does not cover all 24 boards "
                "exactly once; the envelope cannot be trusted")
        #: Number of times :meth:`clamp` actually had to move a value.  A
        #: campaign whose generator is correct should finish with this at zero,
        #: so a non-zero count is a finding about the generator, not a success.
        self.clamped_single = 0
        self.clamped_pair = 0

    # -- the envelope ------------------------------------------------------ #

    def clamp(self, psi24) -> np.ndarray:
        """Return a copy of ``psi24`` that satisfies both inequalities.

        A pair over the sum limit is scaled down **proportionally** rather than
        truncated on its larger member, because the pair's difference is the
        joint torque and its sum is the co-contraction: scaling preserves the
        commanded torque direction and gives up only stiffness, while
        truncating one member would silently rotate the command.
        """
        psi = np.clip(np.asarray(psi24, dtype=float).copy(),
                      self.idle_psi, self.single_max_psi)
        self.clamped_single += int(np.count_nonzero(
            np.asarray(psi24, dtype=float) > self.single_max_psi))
        a = self.pair_idx[:, 0]
        b = self.pair_idx[:, 1]
        total = psi[a] + psi[b]
        over = total > self.pair_sum_max_psi
        if np.any(over):
            self.clamped_pair += int(np.count_nonzero(over))
            # Scale the part of each pressure that sits above the idle floor,
            # so the clamp cannot drive a board below the floor that keeps the
            # leaking board venting.
            floor2 = 2.0 * self.idle_psi
            head = np.maximum(total - floor2, 1e-9)
            room = max(self.pair_sum_max_psi - floor2, 0.0)
            scale = np.where(over, np.minimum(1.0, room / head), 1.0)
            psi[a] = self.idle_psi + (psi[a] - self.idle_psi) * scale
            psi[b] = self.idle_psi + (psi[b] - self.idle_psi) * scale
        return psi

    def assert_safe(self, psi24, *, where: str = "") -> np.ndarray:
        """Raise unless ``psi24`` satisfies the operator's rule.

        Checked against the **hard** ceilings, not the working ones, so a
        campaign may legitimately generate above its own working limit but
        nothing may ever reach the wire above the operator's.
        """
        psi = np.asarray(psi24, dtype=float)
        if psi.shape != (24,):
            raise AssertionError(f"{where}: expected a 24-vector, got {psi.shape}")
        if not np.all(np.isfinite(psi)):
            raise AssertionError(f"{where}: non-finite commanded pressure")
        bad = np.nonzero(psi > SINGLE_MAX_PSI + 1e-9)[0]
        if bad.size:
            raise AssertionError(
                f"{where}: single-line limit: " + ", ".join(
                    f"0x{ALL_BASES[i]:03X}={psi[i]:.2f} psi" for i in bad))
        if np.any(psi < -1e-9):
            raise AssertionError(f"{where}: negative commanded pressure")
        total = psi[self.pair_idx[:, 0]] + psi[self.pair_idx[:, 1]]
        bad = np.nonzero(total > PAIR_SUM_MAX_PSI + 1e-9)[0]
        if bad.size:
            raise AssertionError(
                f"{where}: pair-sum limit: " + ", ".join(
                    f"joint {j} (0x{self.pairs[j][0]:03X}+0x{self.pairs[j][1]:03X})"
                    f"={total[j]:.2f} psi" for j in bad))
        return psi

    def safe(self, psi24, *, where: str = "") -> np.ndarray:
        """Clamp, then assert.  The only function a campaign should call."""
        return self.assert_safe(self.clamp(psi24), where=where)

    # -- helpers a generator wants ----------------------------------------- #

    def idle_vector(self) -> np.ndarray:
        """Every board at :data:`IDLE_PSI`."""
        return np.full(24, self.idle_psi, dtype=float)

    def from_pairs(self, co_psi, diff_psi) -> np.ndarray:
        """Build a 24-vector from per-joint co-contraction and differential.

        ``co_psi[j]`` is the pair's sum and ``diff_psi[j]`` its difference, so
        the positive-driving board gets ``(co + diff) / 2``.  This is the
        coordinate the mechanism actually works in — the difference sets the
        joint torque and the sum sets its stiffness — and generating in it means
        the sum constraint is satisfied by construction rather than by clamping.
        """
        co = np.asarray(co_psi, dtype=float)
        diff = np.asarray(diff_psi, dtype=float)
        psi = np.full(24, self.idle_psi, dtype=float)
        psi[self.pair_idx[:, 0]] = 0.5 * (co + diff)
        psi[self.pair_idx[:, 1]] = 0.5 * (co - diff)
        return np.maximum(psi, self.idle_psi)

    def pair_sums(self, psi24) -> np.ndarray:
        psi = np.asarray(psi24, dtype=float)
        return psi[self.pair_idx[:, 0]] + psi[self.pair_idx[:, 1]]
