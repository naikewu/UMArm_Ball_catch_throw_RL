"""Excitation signals, designed backwards from what the fit has to see.

A campaign is not a demonstration.  Each phase below exists because some
parameter of the twin is unidentifiable without it, and the phase is shaped by
what makes that parameter observable rather than by what looks impressive on
the camera.

What the flow net has to learn is a map from *(pressure, commanded error,
population, muscle length, muscle rate)* to ``dp/dt``.  Everything about the
design follows from wanting that map's input distribution covered:

* **pressure** is covered by the staircase, which visits each level and dwells;
* **commanded error** is covered by steps, which is the only way to see a large
  error, and by the random walk, which is the only way to see the small ones in
  their natural proportion;
* **muscle length and rate** are covered by driving the arm to genuinely
  different poses — a campaign run from one pose learns a flow model that is
  secretly a function of that pose;
* **the two populations** are covered by running every phase on both, never by
  running one and scaling.

Two further phases exist for parameters the flow net does not carry.  The pair
sweeps identify the static torque-vs-differential map at three co-contraction
levels, which is what the anchored McKibben force law's segment constants are
fitted against.  The ringdowns identify the dissipation, and they are the phase
most easily got wrong: the RS485 arm's 20 Hz system-identification campaign
could not see past its own damping constants, and it took 53 ringdown episodes
to find out why.  A ring is only visible in a *release*, so the release here is
a step to idle rather than a ramp.

Every generator returns commanded pressures in **psi** as a 24-vector in CAN id
order, and every one of them is built in the pair coordinate — co-contraction
and differential — so the operator's pair-sum rule holds by construction rather
than by clamping.  :mod:`collection.safety` still clamps and asserts; that is a
guard against this module, not a substitute for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .safety import IDLE_PSI, PairEnvelope

#: Number of joints, and therefore of antagonistic pairs.
N_JOINTS = 12


@dataclass
class Segment:
    """One contiguous piece of a campaign, with the label the recording carries.

    ``duration_s`` is nominal: the driver runs on its own clock and the
    recording's own ``can_sync_time_s`` is what a rollout is aligned to, so a
    segment that runs 20 ms long is a fact about the session rather than a
    defect.  ``kind`` is what the training split groups on — the split is by
    whole episode, never by sample, because adjacent samples at 150 Hz are not
    independent and a random sample split reports a fantasy holdout error.
    """

    name: str
    kind: str
    duration_s: float
    #: ``f(t_rel_s) -> psi24``.  Called from the drive thread at the cycle rate.
    fn: object
    #: Free-form, copied verbatim into the recording's segment index.
    meta: dict = field(default_factory=dict)
    #: Boards this segment deliberately excites, for the anomaly watcher.
    driven: tuple = ()


def _const(psi24):
    arr = np.asarray(psi24, dtype=float)
    return lambda t: arr


# --------------------------------------------------------------------------
# Phase A — rest, and the trapped-air leak probe
# --------------------------------------------------------------------------


def rest(env: PairEnvelope, *, seconds: float = 30.0) -> list:
    """Every board regulating to idle, arm hanging wherever gravity puts it.

    This is the session's joint zero, and it has to be re-taken every session
    because **the joint zero is a mounting property, not a constant**: a
    straight arm does not read all zeros, and ``q`` is repeatable rather than
    absolute.  It is also the sensor bias record — two boards read high at rest
    and the amount is a per-session number.
    """
    return [Segment("rest", "rest", seconds, _const(env.idle_vector()),
                    meta={"purpose": "session joint zero and sensor bias"})]


def leak_probe(env: PairEnvelope, *, charge_psi: float = 12.0,
               charge_s: float = 6.0, trap_s: float = 8.0,
               repeats: int = 2) -> list:
    """Charge every board equally, shut its valves, and watch the drift.

    The leak term is fitted outside the net by least squares on segments where
    both valves are shut, and on this arm those segments have to be *made*: a
    board holding a constant target is in a bang-bang limit cycle, not closed,
    and a limit cycle's closed-conditioned mean slope is approximately zero no
    matter what the leak is.  Dropping the enable bit shuts both valves, which
    traps the air and turns the leak into the only thing moving.

    Charging all twenty-four to the same pressure keeps every pair balanced —
    the sum is ``2 * charge_psi``, comfortably inside the rule — and stiffens
    the arm without moving it, so the trapped-drift window is also a clean
    co-contraction datapoint.

    ``0x110`` is expected to go the *other way*: it leaks from its supply side
    and inflates when it stops being regulated, drifting 0.1 to 7.0 psi over the
    2026-08-21 sweep.  Measuring that is the point; it is not a failure.
    """
    segs = []
    charge = env.from_pairs(np.full(N_JOINTS, 2.0 * charge_psi),
                            np.zeros(N_JOINTS))
    for r in range(repeats):
        segs.append(Segment(f"leak_charge_{r}", "leak", charge_s,
                            _const(charge),
                            meta={"charge_psi": charge_psi}))
        # The trap itself is a *disable*, which the driver performs; the
        # generator holds the same target so the board resumes where it left
        # off when the enable bit comes back.
        segs.append(Segment(f"leak_trap_{r}", "leak", trap_s, _const(charge),
                            meta={"disable_all": True,
                                  "purpose": "both valves shut, drift = leak"}))
        segs.append(Segment(f"leak_vent_{r}", "leak", 5.0,
                            _const(env.idle_vector()), meta={}))
    return segs


# --------------------------------------------------------------------------
# Phase B — what the supply can actually deliver
# --------------------------------------------------------------------------


def supply_probe(env: PairEnvelope, *, bases=(0x101, 0x109),
                 top_psi: float = 26.0, step_psi: float = 4.0,
                 hold_s: float = 3.0, vent_s: float = 5.0) -> list:
    """Staircase one board of each population to find the achievable ceiling.

    The workspace's own notes disagree with the operator about the supply: the
    2026-08 campaign was written against a bench regulated to about 20 psi and
    capped itself at 14, while this session's brief says the inlet is at about
    40 psi.  Which is true changes what every later phase can ask for, and the
    difference is visible in one measurement — the pressure at which the
    measured value stops following the commanded one.

    Running it on one TLE board and one 7 mm board rather than on one of them
    also separates a supply limit, which both would share, from a valve limit,
    which only one would show.
    """
    from .safety import BASE_INDEX
    segs = []
    for base in bases:
        i = BASE_INDEX[base]
        levels = list(np.arange(step_psi, top_psi + 1e-9, step_psi))
        for psi in levels:
            v = env.idle_vector()
            v[i] = psi
            segs.append(Segment(f"supply_{base:03X}_{psi:.0f}", "supply",
                                hold_s, _const(v),
                                meta={"base": f"0x{base:03X}", "psi": float(psi)},
                                driven=(base,)))
        segs.append(Segment(f"supply_{base:03X}_vent", "supply", vent_s,
                            _const(env.idle_vector()), driven=(base,)))
    return segs


# --------------------------------------------------------------------------
# Phase C — one actuator at a time
# --------------------------------------------------------------------------


def single_staircase(env: PairEnvelope, *, bases=None, top_psi: float = 24.0,
                     n_steps: int = 6, hold_s: float = 2.5,
                     vent_s: float = 4.0) -> list:
    """Each actuator alone, up a staircase and back to idle.

    One actuator live at a time makes the pair sum equal to that one pressure,
    so this phase cannot approach the rule even in principle.  It is the phase
    that identifies each board's own fill and vent character — the per-node
    ``fill_gain`` and ``vent_gain`` are exactly what distinguishes twenty-four
    nominally identical actuators, and nothing that drives them together can
    separate them.

    The staircase dwells rather than ramping because the fit needs both halves:
    the transient after each edge gives ``dp/dt`` at a large commanded error,
    and the dwell gives it at a small one, at a known pressure.
    """
    from .safety import ALL_BASES, BASE_INDEX
    bases = tuple(ALL_BASES) if bases is None else tuple(bases)
    levels = np.linspace(top_psi / n_steps, top_psi, n_steps)
    segs = []
    for base in bases:
        i = BASE_INDEX[base]
        for psi in levels:
            v = env.idle_vector()
            v[i] = float(psi)
            segs.append(Segment(f"stair_{base:03X}_{psi:.0f}", "staircase",
                                hold_s, _const(v),
                                meta={"base": f"0x{base:03X}", "psi": float(psi)},
                                driven=(base,)))
        segs.append(Segment(f"stair_{base:03X}_vent", "staircase", vent_s,
                            _const(env.idle_vector()),
                            meta={"base": f"0x{base:03X}", "psi": 0.0},
                            driven=(base,)))
    return segs


# --------------------------------------------------------------------------
# Phase D — the antagonistic map
# --------------------------------------------------------------------------


def pair_sweeps(env: PairEnvelope, *, joints=None,
                co_levels=(10.0, 18.0, 26.0), sweep_s: float = 12.0,
                settle_s: float = 2.0) -> list:
    """Per joint, sweep the differential at several fixed co-contraction sums.

    Holding the sum and moving the difference is the experiment that separates
    the two things a pair does.  The difference is the joint torque, so the
    sweep traces the static torque-vs-differential curve the force law's segment
    constants are fitted against; the sum is the co-contraction, so repeating
    the sweep at three sums says how that curve stiffens — which is the part a
    single-actuator campaign cannot see at all, since one muscle alone is never
    opposed.

    A triangle rather than a sine, because a triangle spends equal time at every
    differential and a sine spends most of its time at the extremes.
    """
    joints = tuple(range(N_JOINTS)) if joints is None else tuple(joints)
    segs = []
    for j in joints:
        for co in co_levels:
            amp = co - 2.0 * IDLE_PSI  # the widest differential this sum allows

            def fn(t, j=j, co=co, amp=amp, T=sweep_s):
                # Triangle over one full period: 0 -> +amp -> -amp -> 0.
                phase = (t / T) % 1.0
                tri = 4.0 * abs(phase - 0.5) - 1.0     # +1 at phase 0, -1 at 0.5
                # Every untouched joint sits at a pair sum of 2*IDLE, which
                # from_pairs turns back into IDLE on both of its boards.
                co_v = np.full(N_JOINTS, 2.0 * IDLE_PSI)
                diff_v = np.zeros(N_JOINTS)
                co_v[j] = co
                diff_v[j] = amp * tri
                return env.from_pairs(co_v, diff_v)

            pos, neg = env.pairs[j]
            segs.append(Segment(f"sweep_j{j}_co{co:.0f}", "pair_sweep",
                                sweep_s, fn,
                                meta={"joint": j, "co_psi": float(co)},
                                driven=(pos, neg)))
        segs.append(Segment(f"sweep_j{j}_rest", "pair_sweep", settle_s,
                            _const(env.idle_vector()), meta={"joint": j}))
    return segs


# --------------------------------------------------------------------------
# Phase E — chirps
# --------------------------------------------------------------------------


def chirps(env: PairEnvelope, *, joints=None, co_psi: float = 18.0,
           amp_psi: float = 7.0, f0: float = 0.15, f1: float = 4.0,
           sweep_s: float = 18.0, settle_s: float = 3.0) -> list:
    """Logarithmic frequency sweep of one joint's differential.

    The band is chosen around what this arm is expected to do rather than around
    what the valves can pass.  The RS485 arm rang at 1.83 Hz with a damping
    ratio near 0.054; this arm is longer and heavier, so its first mode should
    sit lower, and a sweep from 0.15 to 4 Hz brackets that expectation by better
    than a factor of two on each side.  If the ring turns out to be outside this
    band the sweep will say so, which is the point of bracketing rather than
    centring.

    Logarithmic rather than linear so the low decade — where the arm's own
    dynamics live — gets as many cycles as the high one, instead of being
    crossed in the first second.
    """
    joints = tuple(range(N_JOINTS)) if joints is None else tuple(joints)
    segs = []
    for j in joints:
        def fn(t, j=j, T=sweep_s):
            # Instantaneous phase of a log sweep, integrated in closed form so
            # the signal is continuous at the segment boundary.
            k = math.log(f1 / f0)
            ph = 2.0 * math.pi * f0 * T / k * (math.exp(k * min(t / T, 1.0)) - 1.0)
            co_v = np.full(N_JOINTS, 2.0 * IDLE_PSI)
            diff_v = np.zeros(N_JOINTS)
            co_v[j] = co_psi
            diff_v[j] = amp_psi * math.sin(ph)
            return env.from_pairs(co_v, diff_v)

        pos, neg = env.pairs[j]
        segs.append(Segment(f"chirp_j{j}", "chirp", sweep_s, fn,
                            meta={"joint": j, "f0_hz": f0, "f1_hz": f1,
                                  "co_psi": co_psi, "amp_psi": amp_psi},
                            driven=(pos, neg)))
        segs.append(Segment(f"chirp_j{j}_rest", "chirp", settle_s,
                            _const(env.idle_vector()), meta={"joint": j}))
    return segs


# --------------------------------------------------------------------------
# Phase F — the operating regime
# --------------------------------------------------------------------------


def random_walk(env: PairEnvelope, *, seed: int, seconds: float = 300.0,
                hold_lo_s: float = 0.25, hold_hi_s: float = 1.5,
                co_levels=(8.0, 14.0, 20.0, 26.0),
                diff_frac: float = 0.9, block_s: float = 30.0) -> list:
    """All twelve joints retargeted at once, at random, forever.

    This is the phase that matters most to the flow net, and the reason is
    distributional rather than dynamic.  Every other phase visits its part of
    the input space on a grid the designer chose; this one visits the *joint*
    distribution of pressure, commanded error, muscle length and muscle rate
    that the arm actually occupies when many joints move together.  A model
    fitted only on one-at-a-time data is a model that has never seen the arm
    loaded, and the muscle length input is exactly what carries that loading.

    Hold durations are drawn rather than fixed so the excitation has no period
    for the fit to lock onto: a fixed hold makes every trace share a spectrum,
    and a net can match that spectrum without matching the dynamics.

    Cut into ``block_s`` segments so the train/holdout split has whole episodes
    to work with, and so the camera has a natural place to take a clip.
    """
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(round(seconds / block_s)))
    segs = []
    for b in range(n_blocks):
        # Pre-draw the schedule so the segment function is a pure lookup: it is
        # called from the drive thread at 150 Hz and must not allocate or draw.
        t_edges = [0.0]
        while t_edges[-1] < block_s:
            t_edges.append(t_edges[-1] + float(rng.uniform(hold_lo_s, hold_hi_s)))
        n = len(t_edges)
        co = np.asarray(co_levels)[rng.integers(0, len(co_levels), size=(n, N_JOINTS))]
        diff = rng.uniform(-1.0, 1.0, size=(n, N_JOINTS)) * diff_frac * (
            co - 2.0 * IDLE_PSI)
        table = np.stack([env.from_pairs(co[k], diff[k]) for k in range(n)])
        edges = np.asarray(t_edges)

        def fn(t, edges=edges, table=table):
            k = int(np.searchsorted(edges, t, side="right")) - 1
            return table[min(max(k, 0), table.shape[0] - 1)]

        segs.append(Segment(f"walk_{b:03d}", "random_walk", block_s, fn,
                            meta={"block": b, "seed": seed,
                                  "n_targets": int(n)},
                            driven=tuple(range(0x101, 0x119))))
    return segs


# --------------------------------------------------------------------------
# Phase G — ringdowns
# --------------------------------------------------------------------------


def ringdowns(env: PairEnvelope, *, seed: int, episodes: int = 48,
              charge_s: float = 2.5, ring_s: float = 4.0,
              co_psi: float = 22.0) -> list:
    """Load a random pose, then release one joint in a single step.

    The dissipation scalars are fitted to ring episodes, and a ring is only
    visible when the arm is *let go*.  Each episode charges a randomly chosen
    joint's pair to a large differential against a stiff co-contraction, holds
    long enough for the transient to settle, and then drops that pair to idle in
    one cycle — a step, not a ramp, because a ramp slower than the mode being
    measured excites the mode with the ramp's own spectrum and the estimate then
    reports the ramp.

    The other eleven joints are held at a modest random co-contraction rather
    than at idle, so the released joint rings against a configuration rather
    than against a limp chain; the dissipation the fit wants is the one the arm
    has in use.

    Note what this does **not** measure: with the release applied at the
    pressure setpoint, the muscle vents through the same valve that was filling
    it, so the early ring is contaminated by the vent transient.  The estimator
    is fitted to the tail for that reason, and the episode is long enough to
    have one.
    """
    rng = np.random.default_rng(seed)
    segs = []
    for e in range(episodes):
        j = int(rng.integers(0, N_JOINTS))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        bg_co = np.full(N_JOINTS, float(rng.uniform(6.0, 14.0)))
        bg_diff = rng.uniform(-0.5, 0.5, size=N_JOINTS) * (bg_co - 2.0 * IDLE_PSI)
        co = bg_co.copy()
        diff = bg_diff.copy()
        co[j] = co_psi
        diff[j] = sign * (co_psi - 2.0 * IDLE_PSI) * 0.95
        charged = env.from_pairs(co, diff)
        released = env.from_pairs(bg_co, bg_diff)
        pos, neg = env.pairs[j]
        segs.append(Segment(f"ring_{e:03d}_charge", "ringdown_charge",
                            charge_s, _const(charged),
                            meta={"episode": e, "joint": j, "sign": sign},
                            driven=(pos, neg)))
        segs.append(Segment(f"ring_{e:03d}_release", "ringdown", ring_s,
                            _const(released),
                            meta={"episode": e, "joint": j, "sign": sign,
                                  "released_joint": j},
                            driven=(pos, neg)))
    return segs


# --------------------------------------------------------------------------
# Phase H — the held-out validation sequence
# --------------------------------------------------------------------------


def validation(env: PairEnvelope, *, seed: int, seconds: float = 90.0,
               period_s: float = 6.0, co_psi: float = 20.0,
               block_s: float = 30.0) -> list:
    """A slow, large, smooth multi-joint motion, reserved for scoring.

    Held out from every fit and used for two things: the open-loop score, and
    the side-by-side video.  Those two purposes agree on what it should look
    like — large amplitude, so a millimetre of model error is not what the eye
    is asked to judge, and slow, so the comparison is of trajectories rather
    than of two things blurring past.

    Smooth rather than stepped, and generated from a different seed and a
    different *family* than :func:`random_walk`, because a holdout drawn from
    the training distribution measures interpolation and calls it generalisation.
    """
    rng = np.random.default_rng(seed)
    phase0 = rng.uniform(0.0, 2.0 * math.pi, size=N_JOINTS)
    # Incommensurate rates, so the pose never repeats inside the window.
    rate = 1.0 + 0.37 * np.arange(N_JOINTS) / N_JOINTS
    n_blocks = max(1, int(round(seconds / block_s)))
    segs = []
    for b in range(n_blocks):
        def fn(t, b=b, T=period_s, phase0=phase0, rate=rate, B=block_s):
            tt = b * B + t
            co_v = np.full(N_JOINTS, co_psi)
            diff_v = (co_psi - 2.0 * IDLE_PSI) * 0.9 * np.sin(
                2.0 * math.pi * tt / T * rate + phase0)
            return env.from_pairs(co_v, diff_v)

        segs.append(Segment(f"valid_{b:03d}", "validation", block_s, fn,
                            meta={"block": b, "seed": seed,
                                  "period_s": period_s, "co_psi": co_psi},
                            driven=tuple(range(0x101, 0x119))))
    return segs


# --------------------------------------------------------------------------
# The campaign
# --------------------------------------------------------------------------


def full_campaign(env: PairEnvelope, *, seed: int = 20260910,
                  scale: float = 1.0) -> list:
    """Every phase, in the order each one's instrument needs.

    The order is not arbitrary.  Rest first, because the joint zero it records
    is what every later pose is read against and it must be taken before the
    arm has been cycled.  The supply probe second, because what it finds bounds
    every pressure asked for afterwards.  The leak probe early, while the
    actuators are still cold — a leak measured after forty minutes of cycling is
    a leak at a different seal temperature.  The validation sequence last, so
    the recording it produces is the arm as it was at the end of the session
    rather than at the start, which is the harder case for the twin to match.

    ``scale`` shortens every duration proportionally for a rehearsal.  It does
    **not** reduce the number of segments: a short run should exercise the same
    code path as a long one, and the phase that breaks is never the one that got
    dropped.
    """
    def s(x):
        return max(0.5, x * scale)

    segs = []
    segs += rest(env, seconds=s(30.0))
    segs += supply_probe(env, hold_s=s(3.0), vent_s=s(5.0))
    segs += leak_probe(env, charge_s=s(6.0), trap_s=s(8.0), repeats=2)
    segs += single_staircase(env, hold_s=s(2.5), vent_s=s(4.0))
    segs += pair_sweeps(env, sweep_s=s(12.0), settle_s=s(2.0))
    segs += chirps(env, sweep_s=s(18.0), settle_s=s(3.0))
    segs += random_walk(env, seed=seed, seconds=s(300.0), block_s=s(30.0))
    segs += ringdowns(env, seed=seed + 1, episodes=48,
                      charge_s=s(2.5), ring_s=s(4.0))
    segs += rest(env, seconds=s(15.0))
    segs += validation(env, seed=seed + 2, seconds=s(90.0), block_s=s(30.0))
    return segs


def total_seconds(segments) -> float:
    return float(sum(sg.duration_s for sg in segments))
