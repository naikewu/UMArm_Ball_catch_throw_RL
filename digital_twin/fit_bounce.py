r"""Fit the dissipation scalars to a recording's ring episodes.

Four physical quantities, fitted against the episodes :mod:`ring_analysis`
finds: the spine's viscous damping and its Coulomb friction, and the muscle
damping schedule's intercept and pressure slope.  Small, and the smallness is
deliberate -- a dissipation model with many knobs fits any ringdown and predicts
none.

THE LESSON THE RS485 FIT PAID FOR, worth carrying whatever this arm's numbers
turn out to be: dissipation belongs where the physics puts it.  Its previous
values (joint damping 2.25 N.m.s/rad, frictionloss 0.70 N.m) were engineering
guesses a 20 Hz system-identification campaign could never see past, and the
0.7 N.m Coulomb floor alone exceeded the peak elastic torque of a one-degree
oscillation -- measured stick threshold 0.56-1.11 deg per joint at 15 psi, up to
3.3 deg at 5 psi -- so the twin could not oscillate at all, at any
parameterisation of the rest of the model.  Moving the dissipation into
pressure-scheduled damping of the muscle itself, leaving only small spine terms,
is what made the measured rings reproducible.

There is a known cost, recorded rather than hidden: with the small friction the
model has no deflated-muscle passive elasticity to hold the chain during
single-muscle drives, and that campaign regressed from 12/12 to 8/12 with
frozen-plate aborts.  The metal has that elasticity; the model does not.  Do not
restore a ring-killing friction term to paper over it.

WHAT THIS ARM ADDS: the pressure slope splits by population.  ``joint_damping``
and ``joint_frictionloss`` are SPINE terms -- bearings and rod flex -- and stay
global, because a valve cannot change how a bearing dissipates.  ``damp_b1`` is
a muscle property and the two populations are two valves on the same braid, so
it is fitted once per population and expanded into the contract's per-segment
3-vector by :func:`damp_b1_vector`.  ``tendon_damping`` stays global because
``mjcf_generator.generate_xml`` carries exactly one tendon-class default, not
one per segment; splitting it is a generator change, not a fit change, and is
stated here rather than silently wanted.

STIFFNESS IS NOT FITTED HERE.  The anchored force law already carries a
stiffness proportional to pressure (``dF/dl = -6 coeff p l``), so a fit that
touched it would trade against the force scale the sysid campaign pinned.  With
dissipation removed the RS485 model rang within 2 % of its own tangent-stiffness
mode and about 10 % of the measured frequency; a residual frequency error of
that size is a stiffness or mass finding, not a damping one, and
:mod:`digital_twin.force_audit` is where it gets chased.

Ported from ``C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\
UMArm_SIM\fit_bounce.py`` via ``reference/mjcf_fit.md`` section 3.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import replay as R
from . import ring_analysis as RA

# ---------------------------------------------------------------------------
# Target selection gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingGates:
    """Which detected episodes are ring, and which are step-settle.

    ``zeta_max`` is the gate that does the actual separating: a closed-loop
    settle has a damping ratio around 0.4 and above, a structural ring on the
    RS485 arm had a median of 0.054.  The frequency band is wide open ON PURPOSE
    for the same reason -- narrowing it would reject a genuine low mode from a
    longer arm, which is exactly what this arm is expected to have.

    ``r2_min`` is 0.30 against :mod:`ring_analysis`'s ring-beyond-trend r2, not
    against an ordinary mean-referenced one.  It is not a loose gate wearing a
    tight number: 0.30 kept 15 of the RS485 arm's 24 candidate episodes,
    including every visually verified ring.  Raising it back toward
    mean-referenced values to widen a thin target set is how a fit ends up
    trained on step settles.
    """

    zeta_max: float = 0.25
    r2_min: float = 0.30
    freq_hz: "tuple[float, float]" = (0.8, 6.0)
    amp_deg: "tuple[float, float]" = (0.4, 6.0)


#: Detection threshold for the fit's own target selection, deg.  Higher than
#: :data:`ring_analysis.THRESH_DEG` because a fit target has to be a ring worth
#: fitting, not merely a ring worth reporting; the RS485 fit used the same
#: 0.30 deg against a 0.25 deg reporting floor.
TARGET_THRESH_DEG = 0.30

#: Loss weights on the amplitude, frequency and damping-ratio terms.  Damping
#: carries four times the others because it is the quantity being fitted: the
#: amplitude and frequency terms are there to stop the optimiser buying a
#: matching zeta with a twin that rings at the wrong size or the wrong rate.
W_AMP, W_FREQ, W_ZETA = 1.0, 1.0, 4.0

#: Loss charged for a candidate whose rollout raised or came back invalid.  Far
#: above any loss a working candidate produces, so Nelder-Mead walks away from
#: the region rather than exploring it, and recorded with its reason so a fit
#: that failed everywhere is distinguishable from one that merely fitted badly.
FAILED_LOSS = 1.0e3


# ---------------------------------------------------------------------------
# The search space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitSpace:
    """Bounds and the log/linear choice for each scalar, with its physics.

    ============================  =========================  ===============
    scalar                        unit                       searched in
    ============================  =========================  ===============
    ``joint_damping``             N.m.s/rad per hinge        log
    ``joint_frictionloss``        N.m per hinge              linear
    ``tendon_damping``            N.s/m per tendon           log
    ``damp_b1_*``                 N.s/m per Pa               log
    ============================  =========================  ===============

    Log for the three that span decades, because a linear simplex step that is
    sensible near the top of a 0.005-1.5 range is smaller than the whole bottom
    of it.  Linear for the Coulomb term, whose lower bound is exactly zero -- a
    quantity whose physical value may be zero cannot be searched in log space.

    ``damp_b1`` is the pressure slope of the McKibben's viscous damping, affine
    in pressure after Reynolds/Repperger et al. 2003, and attributed by Tondu &
    Lopez 2000 to inter-fibre kinetic friction.  The RS485 fit landed at
    9.9e-4 N.s/m/Pa, which is about 55 N.s/m per muscle at the roughly 8 psi its
    arm rang at -- against a p = 0 base of 1.0 N.s/m, i.e. the base is a rounding
    error at any pressure the arm actually rings at, and that is why the base is
    chosen rather than fitted.
    """

    joint_damping: "tuple[float, float]" = (0.005, 1.5)
    joint_frictionloss: "tuple[float, float]" = (0.0, 0.35)
    tendon_damping: "tuple[float, float]" = (0.05, 30.0)
    damp_b1: "tuple[float, float]" = (1e-5, 3e-3)
    split_damp_b1: bool = True

    def names(self) -> "tuple[str, ...]":
        if self.split_damp_b1:
            return ("joint_damping", "joint_frictionloss", "tendon_damping",
                    "damp_b1_tle", "damp_b1_7mm")
        return ("joint_damping", "joint_frictionloss", "tendon_damping", "damp_b1")

    def _bound(self, name: str) -> "tuple[float, float]":
        return getattr(self, "damp_b1" if name.startswith("damp_b1") else name)

    def _is_log(self, name: str) -> bool:
        return name != "joint_frictionloss"

    def to_vector(self, params: dict) -> np.ndarray:
        x = []
        for name in self.names():
            lo, hi = self._bound(name)
            v = float(np.clip(float(params[name]), lo, hi))
            x.append(math.log(max(v, 1e-30)) if self._is_log(name) else v)
        return np.asarray(x, dtype=np.float64)

    def from_vector(self, x) -> dict:
        """Decode and CLIP into bounds.

        Nelder-Mead is unconstrained, so clipping here rather than penalising is
        what keeps the loss surface flat outside the box instead of steering the
        simplex with a penalty gradient that has nothing to do with the plant.
        """
        out = {}
        for name, v in zip(self.names(), np.asarray(x, dtype=np.float64)):
            lo, hi = self._bound(name)
            val = math.exp(float(v)) if self._is_log(name) else float(v)
            out[name] = float(np.clip(val, lo, hi))
        return out


def damp_b1_vector(params: dict, *, tle_segments=(0,)) -> np.ndarray:
    """Expand the fitted slope(s) into the contract's per-segment ``(3,)`` vector.

    ``tle_segments`` defaults to segment 1 alone because the measured id blocks
    put the eight TLE/DVP boards there and the sixteen legacy boards on segments
    2 and 3, so on this arm the population split and the segment split coincide.
    It is an argument rather than a constant precisely because that coincidence
    is a fact about how the arm is currently addressed, not a law: a reflashed or
    re-addressed board breaks it, and the population must then be read from the
    variant byte and this mapping supplied by the caller.
    """
    if "damp_b1" in params:
        return np.full(3, float(params["damp_b1"]), dtype=np.float64)
    tle = float(params["damp_b1_tle"])
    other = float(params["damp_b1_7mm"])
    v = np.full(3, other, dtype=np.float64)
    for s in tle_segments:
        v[int(s)] = tle
    return v


#: Where the search starts.  ``tendon_damping`` is seeded at 2.0 N.s/m rather
#: than at 1.0 because of a documented Nelder-Mead trap: the initial simplex
#: steps a coordinate whose seed is exactly zero by only 2.5e-4, and
#: ``log(1.0)`` IS zero -- seeding the base there leaves it effectively
#: unexplored while the run still reports a converged fit.
SEED_PARAMS = {
    "joint_damping": 0.03,
    "joint_frictionloss": 0.02,
    "tendon_damping": 2.0,
    "damp_b1_tle": 2.0e-4,
    "damp_b1_7mm": 2.0e-4,
}

#: The point the guard is measured against.  These are the constants the RS485
#: twin SHIPPED before its 2026-08-17 fit, kept here as a deliberately bad
#: reference rather than as a suggestion: the guard exists to prove a fit did not
#: buy its ring by wrecking the trajectory match, and it has to be measured
#: against what actually shipped.  Replace it with this generator's own defaults
#: as soon as ``mjcf_generator`` carries fitted ones.
BASELINE_PARAMS = {
    "joint_damping": 2.25,
    "joint_frictionloss": 0.70,
    "tendon_damping": 50.0,
    "damp_b1_tle": 0.0,
    "damp_b1_7mm": 0.0,
}

#: How much worse than the baseline the fitted trajectory match may be before the
#: fit is refused, as a fraction.  Ten per cent, because the fit is allowed to
#: trade a little trajectory accuracy for a ring it could not otherwise produce,
#: and not allowed to trade a lot.
GUARD_TOL = 0.10


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def ring_targets(t_s, q_deg, *, gates=RingGates(), thresh_deg: float = TARGET_THRESH_DEG,
                 band=RA.BAND_HZ, max_nan_frac: float = 0.2,
                 max_episodes: "int | None" = None) -> "list[RA.Episode]":
    """The episodes a fit is allowed to be scored on.

    Ordered by descending in-band amplitude by :func:`ring_analysis.episodes_from_trace`,
    so ``max_episodes`` keeps the loudest rather than the earliest -- a fit
    budget spent on the quietest episodes in a recording is spent on the ones
    closest to the plate jitter floor.
    """
    eps = RA.episodes_from_trace(t_s, q_deg, band=band, thresh=thresh_deg,
                                 max_nan_frac=max_nan_frac)
    keep = []
    for e in eps:
        f = e.fit
        if f is None or not np.isfinite(f.r2):
            continue
        if f.r2 < gates.r2_min or f.zeta > gates.zeta_max:
            continue
        if not (gates.freq_hz[0] <= f.freq_hz <= gates.freq_hz[1]):
            continue
        if not (gates.amp_deg[0] <= f.amp <= gates.amp_deg[1]):
            continue
        keep.append(e)
    return keep[:int(max_episodes)] if max_episodes else keep


def episode_features(t_s, q_deg, ep: RA.Episode, *, seed_f0: bool,
                     band=RA.BAND_HZ) -> dict:
    """Band RMS and a damped-sine fit for one trace over one episode's window.

    THE SEEDING IS ASYMMETRIC AND THAT ASYMMETRY IS THE POINT.  The real trace is
    re-fitted seeded with its own detected frequency; the twin is fitted blind.
    A seeded fit confines the frequency to +-60 % of the seed, so seeding the
    twin with the real frequency would clamp the very number the loss compares
    and silently understate any frequency mismatch -- the twin would then look
    correct in frequency by construction.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    y = np.asarray(q_deg, dtype=np.float64)[:, ep.joint] if np.ndim(q_deg) == 2 \
        else np.asarray(q_deg, dtype=np.float64)
    filled, nan_frac = RA.interp_nans(t_s, y)
    m = (t_s >= ep.t0) & (t_s <= ep.t1)
    out = {"joint": int(ep.joint), "t0": float(ep.t0), "t1": float(ep.t1),
           "n": int(np.count_nonzero(m)), "nan_frac": float(nan_frac)}
    if out["n"] < 6:
        out.update(band_rms=float("nan"), fit=None)
        return out
    out["band_rms"] = RA.band_rms(t_s, filled, ep.t0, ep.t1, band=band)
    f0 = (1.0 / ep.period_s) if seed_f0 else None
    out["fit"] = RA.fit_damped_sine(t_s[m], filled[m], f0=f0).as_dict()
    return out


def score_candidate(t_real_s, q_real_deg, t_sim_s, q_sim_deg, episodes, *,
                    w_amp: float = W_AMP, w_freq: float = W_FREQ,
                    w_zeta: float = W_ZETA, trust_r2: float = RingGates().r2_min,
                    band=RA.BAND_HZ) -> "tuple[float, list[dict]]":
    """Mean per-episode loss between a real trace and a twin trace.

    ``log(s_amp / r_amp) ** 2`` on the amplitude so that a twin twice as loud and
    a twin half as loud are charged equally -- a linear ratio charges the
    over-damped direction, which is the failure this fit exists to escape, far
    less than the under-damped one.

    WHEN THE TWIN'S OWN FIT IS UNTRUSTED (``r2`` below ``trust_r2`` -- over-damped
    or barely moving) ONLY THE AMPLITUDE TERM CONTRIBUTES, on purpose.  Amplitude
    already punishes a dead twin hard through the log of a near-zero ratio, and
    charging fabricated frequencies from garbage fits would steer the optimiser
    with noise.
    """
    rows: "list[dict]" = []
    total = 0.0
    for ep in episodes:
        rf = episode_features(t_real_s, q_real_deg, ep, seed_f0=True, band=band)
        sf = episode_features(t_sim_s, q_sim_deg, ep, seed_f0=False, band=band)
        r_amp = float(rf.get("band_rms", float("nan")))
        s_amp = float(sf.get("band_rms", float("nan")))
        row = {"joint": int(ep.joint), "t0": float(ep.t0), "t1": float(ep.t1),
               "real_band_rms_deg": r_amp, "twin_band_rms_deg": s_amp,
               "real_fit": rf.get("fit"), "twin_fit": sf.get("fit")}
        if not (np.isfinite(r_amp) and np.isfinite(s_amp)) or r_amp <= 0.0:
            row.update(loss=float(FAILED_LOSS), terms={}, reason="no band amplitude")
            rows.append(row)
            total += FAILED_LOSS
            continue
        # 1e-9 deg is four orders below the 0.25 deg plate jitter floor, so it
        # only keeps the logarithm finite for a twin that did not move at all.
        term_amp = math.log(max(s_amp, 1e-9) / r_amp) ** 2
        terms = {"amp": term_amp}
        loss = w_amp * term_amp
        sfit, rfit = sf.get("fit"), rf.get("fit")
        if sfit and rfit and sfit["r2"] >= float(trust_r2) and rfit["freq_hz"] > 0:
            term_freq = math.log(max(sfit["freq_hz"], 1e-9) / rfit["freq_hz"]) ** 2
            term_zeta = (sfit["zeta"] - rfit["zeta"]) ** 2
            terms.update(freq=term_freq, zeta=term_zeta)
            loss += w_freq * term_freq + w_zeta * term_zeta
        else:
            terms["untrusted_twin_fit"] = True
        row.update(loss=float(loss), terms=terms)
        rows.append(row)
        total += loss
    return (total / max(len(episodes), 1)), rows


# ---------------------------------------------------------------------------
# Candidates and the fit
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    params: dict
    loss: float
    q_rms_deg: float
    reason: str = ""
    rows: list = field(default_factory=list)


@dataclass
class BounceFit:
    """What a fit produced, including everything needed to refuse it."""

    best: Candidate
    baseline: Candidate
    seed: dict
    guard_ok: bool
    guard_tol: float
    n_episodes: int
    n_evals: int
    episodes: list = field(default_factory=list)
    history: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "best": {"params": self.best.params, "loss": self.best.loss,
                     "q_rms_deg": self.best.q_rms_deg, "reason": self.best.reason},
            "baseline": {"params": self.baseline.params, "loss": self.baseline.loss,
                         "q_rms_deg": self.baseline.q_rms_deg,
                         "reason": self.baseline.reason},
            "seed": self.seed,
            "guard_ok": self.guard_ok,
            "guard_tol": self.guard_tol,
            "n_episodes": self.n_episodes,
            "n_evals": self.n_evals,
            "episodes": [e.as_dict() for e in self.episodes],
            "meta": self.meta,
            "episode_rows": self.best.rows,
        }


def _align_exact(t_real, t_sim):
    """Rows present in both traces, matched at bitwise the same stamp.

    Exact rather than nearest because a candidate rollout is driven on the
    recording's own ``can_sync_time_s`` values; anything else means the candidate
    was rolled on a different clock, and a nearest-neighbour match would absorb
    that defect into the residual instead of exposing it.
    """
    t_real = np.asarray(t_real, dtype=np.float64)
    t_sim = np.asarray(t_sim, dtype=np.float64)
    pos = np.searchsorted(t_sim, t_real)
    ok = (pos < t_sim.size) & (t_sim[np.clip(pos, 0, t_sim.size - 1)] == t_real)
    return np.flatnonzero(ok), pos[ok]


def _q_rms_deg(t_real, q_real_deg, t_sim, q_sim_deg) -> float:
    """Trajectory match, deg, over the rows and joints both traces actually have."""
    ir, isim = _align_exact(t_real, t_sim)
    if ir.size == 0:
        return float("nan")
    err = np.asarray(q_sim_deg)[isim] - np.asarray(q_real_deg)[ir]
    good = np.isfinite(err)
    if not np.any(good):
        return float("nan")
    return float(np.sqrt(np.mean(err[good] ** 2)))


def default_rollout_fn(params: dict, rec: R.Recording, *, t_max_s=None,
                       tle_segments=(0,)) -> "tuple[np.ndarray, np.ndarray]":
    """Roll one candidate and return ``(t_rel_s, q_deg)``.

    Builds a FRESH ``ActuatorModel`` per candidate and hands the three MJCF
    scalars to ``SimArm`` as constructor arguments, which is the seam
    ``CONTRACT.md`` section 3 requires ("every tunable an argument with a named
    default, never a module constant read at call time") -- a fit that mutated a
    module constant would leave the last candidate's dissipation behind in every
    later rollout in the process.
    """
    try:
        from digital_twin import actuator_model as AM  # noqa: WPS433 -- deferred
    except Exception as exc:  # pragma: no cover - depends on a sibling module
        raise RuntimeError(
            f"digital_twin.actuator_model is not importable ({exc}); pass "
            f"fit(..., rollout_fn=) to fit against something else") from exc
    model = AM.ActuatorModel.fresh(
        is_tle=rec.is_tle,
        damp_b1=damp_b1_vector(params, tle_segments=tle_segments))
    roll = R.rollout(rec, t_max_s=t_max_s, actuator=model,
                     joint_damping=float(params["joint_damping"]),
                     joint_frictionloss=float(params["joint_frictionloss"]),
                     tendon_damping=float(params["tendon_damping"]))
    return roll.t_s - roll.t_s[0], np.degrees(roll.q_rad)


def fit(rec: R.Recording, *, rollout_fn=default_rollout_fn, space=FitSpace(),
        seed_params=None, baseline_params=None, gates=RingGates(),
        thresh_deg: float = TARGET_THRESH_DEG, band=RA.BAND_HZ,
        budget: int = 60, t_max_s: "float | None" = 60.0,
        max_episodes: "int | None" = None, guard_tol: float = GUARD_TOL,
        weights=(W_AMP, W_FREQ, W_ZETA), out_dir: "str | None" = None,
        progress=None) -> BounceFit:
    """Fit the dissipation scalars against ``rec``'s ring episodes.

    Method, and why each step is where it is:

    1. detect targets on the REAL trace once, before any rollout, so every
       candidate is scored on the same episodes and a candidate cannot improve
       its loss by changing which episodes exist;
    2. refuse outright if no episode survives the gates, or if ``t_max_s``
       leaves none inside the fit window -- an empty target set makes the loss
       identically zero for every candidate, and a fit that reports convergence
       on an empty set is worse than no fit;
    3. evaluate the baseline, deliberately bypassing the fit-space clipping,
       because the guard must be measured against constants that actually
       shipped and those sit outside the search box;
    4. Nelder-Mead on the encoded vector to ``budget`` evaluations;
    5. take the BEST EVALUATED POINT, not scipy's simplex centroid -- on a
       sixty-evaluation budget those differ;
    6. apply the guard.

    ON A GUARD FAILURE NOTHING FITTABLE IS EMITTED.  A fit that bought its ring
    by wrecking the trajectory match must not become a checkpoint anyone can load
    by accident; the forensics go to ``out_dir`` and the caller gets
    ``guard_ok=False``.
    """
    w_amp, w_freq, w_zeta = weights
    seed = dict(SEED_PARAMS if seed_params is None else seed_params)
    base = dict(BASELINE_PARAMS if baseline_params is None else baseline_params)
    if not space.split_damp_b1:
        for d in (seed, base):
            if "damp_b1" not in d:
                d["damp_b1"] = d.pop("damp_b1_7mm", d.get("damp_b1_tle", 0.0))
                d.pop("damp_b1_tle", None)

    fit_rec = rec.slice_time(t_max_s) if t_max_s is not None else rec
    # Blanked once, here, and reused for both target detection and scoring: an
    # unposed sample scored as if it carried an angle contributes the mocap
    # dropout's shape to the loss.
    q_real_deg = np.degrees(rec.q_rad)
    q_real_deg[~rec.q_valid] = np.nan
    q_fit_deg = q_real_deg[:fit_rec.n]
    episodes = ring_targets(rec.t_rel_s, q_real_deg, gates=gates,
                            thresh_deg=thresh_deg, band=band,
                            max_episodes=max_episodes)
    if not episodes:
        raise ValueError(
            f"no ring episode survived the gates (zeta<={gates.zeta_max}, "
            f"r2>={gates.r2_min}, {gates.freq_hz} Hz, {gates.amp_deg} deg) on a "
            f"{rec.duration_s:.1f} s recording at {rec.rate_hz:.2f} Hz; refusing "
            f"to fit rather than reporting a loss that is identically zero")
    horizon = float(fit_rec.t_rel_s[-1])
    inside = [e for e in episodes if e.t1 <= horizon]
    if not inside:
        raise ValueError(
            f"t_max_s={t_max_s} s leaves no episode inside the fit window "
            f"(earliest episode ends at {min(e.t1 for e in episodes):.2f} s); "
            f"an empty target set makes every candidate's loss identically 0")
    episodes = inside

    history: "list[Candidate]" = []

    def evaluate(params: dict) -> Candidate:
        try:
            t_sim, q_sim = rollout_fn(params, fit_rec, t_max_s=None)
        except R.ClampViolation as exc:
            cand = Candidate(dict(params), FAILED_LOSS, float("nan"),
                             f"clamp violation: {exc}")
            history.append(cand)
            return cand
        except Exception as exc:  # a diverged candidate is data, not a crash
            cand = Candidate(dict(params), FAILED_LOSS, float("nan"), repr(exc))
            history.append(cand)
            return cand
        loss, rows = score_candidate(fit_rec.t_rel_s, q_fit_deg, t_sim, q_sim,
                                     episodes, w_amp=w_amp, w_freq=w_freq,
                                     w_zeta=w_zeta, trust_r2=gates.r2_min, band=band)
        q_rms = _q_rms_deg(fit_rec.t_rel_s, q_fit_deg, t_sim, q_sim)
        cand = Candidate(dict(params), float(loss), q_rms, "", rows)
        history.append(cand)
        if progress is not None:
            progress(len(history), cand)
        return cand

    baseline = evaluate(base)

    x0 = space.to_vector(seed)
    n_before = len(history)

    def objective(x):
        return evaluate(space.from_vector(x)).loss

    minimize(objective, x0, method="Nelder-Mead",
             options={"maxfev": int(budget), "xatol": 0.05, "fatol": 5e-3})

    searched = history[n_before:]
    best = min(searched, key=lambda c: c.loss) if searched else baseline
    guard_ok = bool(np.isfinite(baseline.q_rms_deg)
                    and np.isfinite(best.q_rms_deg)
                    and best.q_rms_deg <= baseline.q_rms_deg * (1.0 + float(guard_tol)))

    out = BounceFit(best=best, baseline=baseline, seed=seed, guard_ok=guard_ok,
                    guard_tol=float(guard_tol), n_episodes=len(episodes),
                    n_evals=len(searched), episodes=episodes,
                    history=[{"params": c.params, "loss": c.loss,
                              "q_rms_deg": c.q_rms_deg, "reason": c.reason}
                             for c in history],
                    meta={"recording": rec.path, "t_max_s": t_max_s,
                          "fit_window_s": horizon, "rate_hz": float(rec.rate_hz),
                          "gates": gates.__dict__, "weights": list(weights),
                          "split_damp_b1": bool(space.split_damp_b1),
                          "n_tle_boards": int(np.count_nonzero(rec.is_tle))})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        name = "fit_bounce.json" if guard_ok else "fit_bounce_GUARD_FAILED.json"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(out.as_dict(), fh, indent=1, default=str)
    return out


def main(argv=None) -> int:
    """``python -m digital_twin.fit_bounce <session_dir> [--budget 60] [--out DIR]``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--t-max-s", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-split", action="store_true",
                    help="one damp_b1 for all three segments, as the RS485 fit had")
    args = ap.parse_args(argv)

    rec = R.load_recording(args.session)
    res = fit(rec, budget=args.budget, t_max_s=args.t_max_s, out_dir=args.out,
              space=FitSpace(split_damp_b1=not args.no_split))
    print(f"{res.n_episodes} episodes, {res.n_evals} evaluations")
    print("baseline loss", res.baseline.loss, "q_rms", res.baseline.q_rms_deg, "deg")
    print("best     loss", res.best.loss, "q_rms", res.best.q_rms_deg, "deg")
    for k, v in res.best.params.items():
        print(f"  {k} = {v}")
    print("guard", "OK" if res.guard_ok else
          "FAILED -- no checkpoint emitted; the ring was bought with trajectory error")
    return 0 if res.guard_ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RingGates", "TARGET_THRESH_DEG", "W_AMP", "W_FREQ", "W_ZETA", "FAILED_LOSS",
    "FitSpace", "SEED_PARAMS", "BASELINE_PARAMS", "GUARD_TOL",
    "Candidate", "BounceFit",
    "damp_b1_vector", "ring_targets", "episode_features", "score_candidate",
    "default_rollout_fn", "fit", "main",
]
