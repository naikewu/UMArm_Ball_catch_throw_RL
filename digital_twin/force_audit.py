r"""Is the force law overtuned?  Five re-runnable measurements that change nothing.

An instrument, not a step in a pipeline: it reads the fitted model and reports
five independent measurements of whether the force law is doing work the rest of
the model should be doing.  It writes nothing back, which is what makes it safe
to run after every fit and honest when it disagrees with one.

KEEP THAT PROPERTY.  An audit that can adjust what it audits stops being evidence
the first time somebody is in a hurry.  Every coefficient sweep here goes through
:func:`scaled`, which returns a NEW :class:`ForceLaw` and never touches the one
it was handed; ``test_force_audit.py`` asserts that the audited model's arrays
are bit-identical before and after a full run.

THE FIVE PARTS, and what each can and cannot settle:

1. ``geometry`` -- pure arithmetic on the law's own constants, read backwards
   through Chou-Hannaford into a braid.  *Can show the modelled muscle cannot
   physically exist on this arm's anchor rings.  Cannot say what the right force
   is.*
2. ``static`` -- one muscle, fixed pressure, settle, at several coefficient
   scales.  The anchored law goes slack at ``l = Bf/sqrt(3)``, so the
   equilibrium is geometry-limited rather than force-limited: on the RS485 arm a
   sixteen-fold coefficient change moved the settled angle by a few per cent.
   *A pressure-versus-angle campaign is blind to the force scale by
   construction, which is why this part exists to be reported rather than
   believed.*
3. ``ring`` -- free ringdown frequency against the coefficient scale, fitting
   ``f^2`` affine in the scale so the gravity share and the elastic share
   separate.  *The one measurement that constrains the scale, and on the RS485
   arm it pulled the opposite way from everything else.*
4. ``replay`` -- peak joint acceleration through a real recording at two
   coefficient scales.  *Acceleration is the closest thing to a force
   measurement this repo has.*
5. ``clamp`` -- headroom between the peak commanded tendon force and the
   pull-only clip.

**Part five is a re-derivation, not a port, and the substitution is deliberate.**
The RS485 audit's fifth part is a collision trial, and this twin's MJCF is
contact-free by contract -- ``contype``/``conaffinity`` zero everywhere, no
contact block, no keyframe -- so a collision measurement here would be measuring
a scene that does not exist.  The clamp headroom answers the same class of
question the collision part answered (is the law being driven somewhere the
model does not describe?) on a model this arm actually has.

Ported from ``C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision\
UMArm_SIM\force_audit.py`` via ``reference/mjcf_fit.md`` section 5.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import savgol_filter

from . import replay as R
from . import ring_analysis as RA

PARTS = ("geometry", "static", "ring", "replay", "clamp")

#: The anchor-ring radii this arm's tendons are routed on, m, from
#: ``UMArm_KINEMATICS.canarm_params.CANARM_PARAMS``: ``JA1 = JA2 = 0.047`` on the
#: plates and ``AO1 = AO2 = 0.028`` on the link discs, the same on all three
#: segments.  The RS485 arm's 0.100 / 0.050 / 0.030 do not transfer, and the
#: display model's ``DEFAULT_PLATE_RADII`` are cosmetic RS485 leftovers -- in a
#: twin the ring radius IS the moment arm, so using the display values would
#: inflate every plate-ring joint torque by 17 % and every link-ring one by 96 %.
ANCHOR_RADII_M = {"JA1": 0.047, "JA2": 0.047, "AO1": 0.028, "AO2": 0.028}

#: The operator's mass figures, kg.  Thirty grams per actuator times twenty-four
#: is the only published mass on this arm; the structure's distribution has NOT
#: been measured here, and the display model carries 0.3204 kg of moving mass
#: against the RS485 twin's 1.261 kg on a SHORTER arm.  Both frequency and
#: damping ratio read mass, so this is the single largest unmeasured input to
#: anything in this file, and every ring number below has to be read with it.
OPERATOR_ACTUATOR_KG = 0.030
N_ACTUATORS = 24

#: The operator's per-line pressure cap, psi, from ``CONTRACT.md`` section 8.
#: The pair rule is the same number summed over an antagonistic pair, so a
#: single-muscle envelope of 30 psi is the worst case a clamp audit must clear.
ENVELOPE_PSI = 30.0

#: Coefficient scales the sweeps run at.  Spanning a factor of eight because the
#: RS485 audit's finding was that a factor of sixteen moved the settled angle by
#: only a few per cent -- a narrower sweep would have reported "no effect" and
#: been read as "the scale is right".
DEFAULT_SCALES = (0.25, 0.5, 1.0, 2.0)

#: Savitzky-Golay derivative window, samples, and polynomial order.  Seventeen
#: samples is 113 ms at this arm's 150 Hz sync rate (the RS485 audit used the
#: same seventeen at 159 Hz, i.e. 107 ms).  Raw differentiation is unusable on
#: mocap-derived joint angles: one frame of the RS485 2026-08-16 recording
#: carries 11 rad/s and a quarter of its rows repeat the previous mocap frame.
SAVGOL_WINDOW = 17
SAVGOL_POLY = 3

#: How much of the start of a replay to discard before differentiating, s.  The
#: first samples carry the rollout's own start transient -- the arm begins at
#: qpos0 with whatever pressure the first sync edge commands -- and a peak
#: acceleration taken there is a measurement of the initial condition.
SAVGOL_SKIP_S = 1.0


@dataclass(frozen=True)
class ForceLaw:
    """The anchored McKibben law's constants, per segment.

    ``f = coeff * p * (Bf^2 - 3 l^2)``, clipped to ``[-FORCE_CLIP_N, 0]`` --
    negative is pull, and a McKibben cannot push.  Frozen so a sweep cannot
    scale the audited law in place; :func:`scaled` returns a new one.
    """

    coeff: np.ndarray   # (3,) N per Pa per m^2
    bf: np.ndarray      # (3,) m, the braid's fixed fibre length
    l0: np.ndarray      # (3,) m, the anchored rest length
    source: str = ""

    @property
    def slack_len_m(self) -> np.ndarray:
        """``Bf/sqrt(3)``: the length at which the law's force crosses zero.

        BELOW it the expression turns positive -- a push -- and the pull-only
        clip zeroes it, so a muscle shorter than this carries no force at any
        pressure.  That is where the equilibrium stops being a property of the
        pressure and becomes a property of the geometry, and it is the reason a
        static pressure-versus-angle campaign cannot see the force scale.
        """
        return self.bf / math.sqrt(3.0)


def scaled(law: ForceLaw, coeff_scale: float = 1.0, bf_scale: float = 1.0) -> ForceLaw:
    """A NEW law with the coefficients scaled.  The argument is never modified."""
    return replace(law, coeff=np.asarray(law.coeff, dtype=np.float64) * float(coeff_scale),
                   bf=np.asarray(law.bf, dtype=np.float64) * float(bf_scale),
                   source=f"{law.source} (coeff x{coeff_scale}, bf x{bf_scale})")


def force_law_from(model) -> ForceLaw:
    """Read a :class:`ForceLaw` out of an ``ActuatorModel``, by attribute name.

    By attribute rather than by import so this file never has to construct one,
    which is what lets the audit run against a checkpoint loaded by somebody
    else and against a hand-built law in a test by the same code.
    """
    return ForceLaw(coeff=np.asarray(model.coeff, dtype=np.float64).reshape(3),
                    bf=np.asarray(model.bf, dtype=np.float64).reshape(3),
                    l0=np.asarray(model.l0, dtype=np.float64).reshape(3),
                    source=getattr(model, "source", type(model).__name__))


def force_n(law: ForceLaw, psi: float, l_m=None) -> np.ndarray:
    """Per-segment force, N, at ``psi`` and length ``l_m`` (default: the rest length)."""
    l = np.asarray(law.l0 if l_m is None else l_m, dtype=np.float64)
    p_pa = float(psi) * R.PA_PER_PSI
    f = law.coeff * p_pa * (law.bf ** 2 - 3.0 * l * l)
    return np.clip(f, -R.FORCE_CLIP_N, 0.0)


def stiffness_n_per_m(law: ForceLaw, psi: float, l_m=None) -> np.ndarray:
    """``|dF/dl| = |6 coeff p l|``, N/m.

    Reported beside every force because the two cannot be scaled apart: the
    lever ``F/(dF/dl) = (3 l^2 - Bf^2) / (6 l)`` is fixed by the geometry alone,
    so any change to ``coeff`` moves force and stiffness together and no
    coefficient exists that fixes one without moving the other.
    """
    l = np.asarray(law.l0 if l_m is None else l_m, dtype=np.float64)
    return np.abs(6.0 * law.coeff * float(psi) * R.PA_PER_PSI * l)


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------


def audit_geometry(law: ForceLaw, *, anchor_radii_m=ANCHOR_RADII_M,
                   psis=(8.0, 25.0, ENVELOPE_PSI), stiffness_psi: float = 8.0,
                   muscle_diameters_mm=(12.0, 15.0, 20.0, 25.0)) -> dict:
    """Read Chou-Hannaford backwards: what braid does this law describe?

    ``coeff = 1/(4 pi Nf^2)`` gives the fibre turn count, ``D0 = Bf/(pi Nf)`` the
    braid diameter at zero angle, ``cos(theta) = l0/Bf`` the braid angle at rest,
    and the rest diameter follows as ``D0 sin(theta)``.  Then the check that
    matters on this arm: four muscles share each anchor ring, so their combined
    cross-section is compared with the area of the ring they are seated on.

    WHAT THIS CANNOT SHOW.  Nothing here is a force measurement.  A law can
    describe a perfectly plausible braid and still be wrong by a factor in
    ``coeff``, because ``coeff`` and ``Nf`` are the same number read two ways.
    """
    coeff = np.asarray(law.coeff, dtype=np.float64)
    bf = np.asarray(law.bf, dtype=np.float64)
    l0 = np.asarray(law.l0, dtype=np.float64)

    nf = np.sqrt(1.0 / (4.0 * np.pi * coeff))
    d0 = bf / (np.pi * nf)
    cos_theta = np.clip(l0 / bf, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    d_rest = d0 * np.sin(theta)

    ao_m = float(anchor_radii_m["AO1"])
    four_muscle_mm2 = 4.0 * np.pi * (d_rest * 1000.0 / 2.0) ** 2
    ring_mm2 = np.pi * (ao_m * 1000.0) ** 2

    out = {
        "available": True,
        "source": law.source,
        "turns_nf": nf.tolist(),
        "braid_diameter_zero_angle_m": d0.tolist(),
        "braid_angle_rest_deg": np.degrees(theta).tolist(),
        "rest_diameter_m": d_rest.tolist(),
        "bladder_volume_ml": (np.pi * (d_rest / 2.0) ** 2 * l0 * 1e6).tolist(),
        "slack_length_m": law.slack_len_m.tolist(),
        "rest_length_m": l0.tolist(),
        # Above one, the four muscles seated on a ring do not fit inside it, so
        # either the modelled diameter or the ring radius is wrong.
        "overfill_on_AO_ring": (four_muscle_mm2 / ring_mm2).tolist(),
        "anchor_radii_m": dict(anchor_radii_m),
        "muscle_kg_total": OPERATOR_ACTUATOR_KG * N_ACTUATORS,
    }
    for psi in psis:
        out[f"force_at_{psi:g}psi_n"] = force_n(law, psi).tolist()
    out[f"stiffness_at_{stiffness_psi:g}psi_n_per_m"] = \
        stiffness_n_per_m(law, stiffness_psi).tolist()
    # Force and stiffness cannot be scaled apart; the ratio is geometry alone.
    out["force_over_stiffness_lever_m"] = \
        ((3.0 * l0 ** 2 - bf ** 2) / (6.0 * l0)).tolist()
    out["coeff_scale_for_diameter"] = {
        f"{d:g}mm": (((float(d) / 1000.0) / d_rest) ** 2).tolist()
        for d in muscle_diameters_mm
    }
    return out


# ---------------------------------------------------------------------------
# 2. static
# ---------------------------------------------------------------------------


def audit_static(law: ForceLaw, *, settle_fn=None, scales=DEFAULT_SCALES,
                 psis=(8.0, 25.0), nodes=(1, 9, 17)) -> dict:
    """Settled deflection of one muscle at several coefficient scales.

    ``settle_fn(law, psi, node) -> max_abs_q_deg`` is the seam a caller supplies;
    with none this part reports itself unavailable rather than inventing a
    number, because a static settle needs a stepped model and this module owns
    no simulator.

    THE VERDICT IS STRUCTURAL AND DOES NOT DEPEND ON THE NUMBERS.  The anchored
    law goes slack at ``l = Bf/sqrt(3)``, so past that length the muscle carries
    no force at any pressure and the equilibrium is set by where the geometry
    runs out.  ``spread_frac`` -- how much the settled angle moves across the
    whole coefficient sweep -- is therefore the reportable quantity, and on the
    RS485 arm a sixteen-fold sweep moved it by a few per cent.
    """
    if settle_fn is None:
        return {"available": False,
                "reason": "no settle_fn supplied; a static settle needs a stepped "
                          "model and force_audit owns no simulator",
                "slack_length_m": law.slack_len_m.tolist()}
    rows = []
    for psi in psis:
        for node in nodes:
            angles = []
            for s in scales:
                angles.append(float(settle_fn(scaled(law, s), float(psi), int(node))))
            a = np.asarray(angles, dtype=np.float64)
            span = float(np.max(a) - np.min(a))
            rows.append({
                "psi": float(psi), "node": int(node), "scales": list(scales),
                "max_abs_q_deg": a.tolist(),
                # Fraction of the mean the whole sweep moves the angle by: the
                # number that says whether a static campaign can see the scale.
                "spread_frac": span / max(float(np.mean(np.abs(a))), 1e-12),
            })
    return {"available": True, "rows": rows, "scales": list(scales),
            "verdict": "geometry-limited, not force-limited, wherever spread_frac "
                       "is small: the settle is set by the slack length, so a "
                       "pressure-versus-angle campaign is blind to the force scale"}


# ---------------------------------------------------------------------------
# 3. ring
# ---------------------------------------------------------------------------


def audit_ring(law: ForceLaw, *, poke_fn=None, scales=DEFAULT_SCALES,
               real_ring_hz: "float | None" = None, poke_n: float = 2.0) -> dict:
    """Free-ringdown frequency against the coefficient scale.

    ``poke_fn(law, poke_n) -> (t_s, q_deg)`` is the seam; the returned trace goes
    through :func:`ring_analysis.episodes_from_trace`, the SAME entry point every
    other measurement in this package uses, so the frequency here and the
    frequency in :mod:`digital_twin.twin_compare` are the same quantity.

    ``f^2`` is fitted affine in the coefficient scale because the two restoring
    terms add in stiffness, not in frequency: gravity contributes a scale-free
    intercept and the muscle elasticity a term proportional to ``coeff``.  The
    intercept's share at unit scale is therefore how much of the twin's ring is
    the pendulum rather than the muscle, and ``scale_for_real_hz`` inverts the
    line -- reported only when a measured frequency is supplied, because this
    arm's ring frequency has not been measured and a default would be the RS485
    arm's 1.83 Hz quietly answering for a longer, heavier mechanism.

    ``poke_n`` is 2.0 N rather than a large impulse because ring damping is
    amplitude-dependent: the RS485 twin read a damping ratio of 0.099 at 3.9 deg
    and 0.059 at 23.9 deg, while its real episodes were 0.46-2.9 deg.  Auditing
    at an amplitude the arm never reaches audits a different mode.
    """
    if poke_fn is None:
        return {"available": False,
                "reason": "no poke_fn supplied; a free ringdown needs a stepped "
                          "model and force_audit owns no simulator"}
    rows = []
    for s in scales:
        t, q_deg = poke_fn(scaled(law, s), poke_n)
        eps = RA.episodes_from_trace(np.asarray(t), np.asarray(q_deg))
        m = RA.ring_metrics(eps)
        rows.append({"scale": float(s), "freq_hz": m["median_freq_hz"],
                     "zeta": m["median_zeta"], "n_trusted": m["n_trusted"]})
    good = [r for r in rows if np.isfinite(r["freq_hz"])]
    out = {"available": True, "rows": rows, "poke_n": float(poke_n),
           "scales": list(scales)}
    if len(good) < 2:
        out["reason"] = "fewer than two scales produced a trusted ring fit"
        return out
    x = np.asarray([r["scale"] for r in good], dtype=np.float64)
    y = np.asarray([r["freq_hz"] for r in good], dtype=np.float64) ** 2
    a, b = np.polyfit(x, y, 1)
    out.update(f2_slope_hz2_per_scale=float(a), f2_intercept_hz2=float(b),
               gravity_share_at_x1=float(b / (a + b)) if (a + b) != 0 else float("nan"))
    if real_ring_hz is not None and a != 0.0:
        out["scale_for_real_hz"] = float((float(real_ring_hz) ** 2 - b) / a)
        out["real_ring_hz"] = float(real_ring_hz)
    else:
        out["scale_for_real_hz"] = None
        out["reason_no_scale"] = ("this arm's ring frequency has not been measured; "
                                  "pass real_ring_hz once a recording gives one")
    return out


# ---------------------------------------------------------------------------
# 4. replay
# ---------------------------------------------------------------------------


def savgol_derivatives(t_s, q, *, window: int = SAVGOL_WINDOW, poly: int = SAVGOL_POLY,
                       skip_s: float = SAVGOL_SKIP_S):
    """First and second time derivatives of ``q`` ``(n, nj)``, Savitzky-Golay.

    Raw differencing is unusable on mocap-derived angles -- one frame of the
    RS485 2026-08-16 recording carries 11 rad/s and a quarter of its rows repeat
    the previous mocap frame, so a first difference alternates between zero and
    twice the true rate.  A cubic Savitzky-Golay over 17 samples differentiates
    the local polynomial instead, which is why the same window is used on the
    real trace and on the twin: an asymmetric smoothing would compare a filtered
    arm with an unfiltered model.

    Non-finite rows are dropped BEFORE filtering, and the first ``skip_s`` are
    discarded, so the returned arrays are shorter than the input.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    good = np.all(np.isfinite(q), axis=1) & np.isfinite(t_s)
    t_s, q = t_s[good], q[good]
    if t_s.size == 0:
        raise ValueError("no finite rows to differentiate")
    keep = t_s >= (t_s[0] + float(skip_s))
    t_s, q = t_s[keep], q[keep]
    w = int(window)
    if w % 2 == 0:
        w += 1  # savgol requires an odd window; rounding up keeps the span
    if t_s.size <= w:
        raise ValueError(
            f"{t_s.size} usable samples is not more than the {w}-sample "
            f"Savitzky-Golay window; widen the trace or narrow the window")
    dt = float(np.median(np.diff(t_s)))
    d1 = savgol_filter(q, w, poly, deriv=1, delta=dt, axis=0, mode="interp")
    d2 = savgol_filter(q, w, poly, deriv=2, delta=dt, axis=0, mode="interp")
    return t_s, d1, d2


def audit_replay(law: ForceLaw, rec: R.Recording, *, rollout_fn=None,
                 scales=(0.5, 1.0), window: int = SAVGOL_WINDOW,
                 skip_s: float = SAVGOL_SKIP_S) -> dict:
    """Peak joint velocity and acceleration through a real recording, per coefficient scale.

    Acceleration is the closest thing to a force measurement this repo has: the
    settled angle is geometry-limited and therefore blind to the scale, but the
    rate at which the arm gets there is not.  Ratios twin-over-real are reported
    per joint and as a median, because a single joint's ratio is one plate's
    mocap quality as much as it is the model's.

    THIS DELIBERATELY DOES NOT USE A CACHED TWIN.  On the RS485 arm an
    algorithm-2 cache scored the twin four times too SLOW -- the exact opposite
    answer -- and nothing in the cache file said which algorithm wrote it.

    ``rollout_fn(law, rec) -> (t_s, q_deg)`` is the seam.
    """
    if rollout_fn is None:
        return {"available": False,
                "reason": "no rollout_fn supplied; force_audit owns no simulator "
                          "and must not read a cached twin, which can carry the "
                          "opposite answer from a superseded algorithm"}
    q_real = np.degrees(rec.q_rad)
    q_real[~rec.q_valid] = np.nan
    t_r, dr1, dr2 = savgol_derivatives(rec.t_rel_s, q_real, window=window,
                                       skip_s=skip_s)
    peak_r1 = np.max(np.abs(dr1), axis=0)
    peak_r2 = np.max(np.abs(dr2), axis=0)

    rows = []
    for s in scales:
        t_s, q_sim = rollout_fn(scaled(law, s), rec)
        _, ds1, ds2 = savgol_derivatives(t_s, q_sim, window=window, skip_s=skip_s)
        p1 = np.max(np.abs(ds1), axis=0)
        p2 = np.max(np.abs(ds2), axis=0)
        r1 = p1 / np.maximum(peak_r1, 1e-12)
        r2 = p2 / np.maximum(peak_r2, 1e-12)
        rows.append({
            "scale": float(s),
            "qdot_ratio_median": float(np.median(r1)),
            "qddot_ratio_median": float(np.median(r2)),
            "qddot_ratio_per_joint": r2.tolist(),
            "twin_peak_qddot_deg_s2": p2.tolist(),
        })
    return {"available": True, "rows": rows,
            "real_peak_qdot_deg_s": peak_r1.tolist(),
            "real_peak_qddot_deg_s2": peak_r2.tolist(),
            "window_samples": int(window),
            "window_ms": float(window / max(rec.rate_hz, 1e-9) * 1000.0),
            "skip_s": float(skip_s),
            "n_samples_used": int(t_r.size)}


# ---------------------------------------------------------------------------
# 5. clamp
# ---------------------------------------------------------------------------


def audit_clamp(law: ForceLaw, *, roll: "R.Rollout | None" = None,
                envelope_psi: float = ENVELOPE_PSI,
                force_clip_n: float = R.FORCE_CLIP_N,
                scales=DEFAULT_SCALES) -> dict:
    """Headroom between the law's peak pull and the pull-only clip.

    Two independent readings.  The closed-form one asks what the law produces at
    the operator's per-line cap with the muscle at its rest length -- the
    strongest pull the law can be asked for inside the envelope -- and reports it
    against the clip.  The measured one, when a rollout is handed in, reports the
    most negative tendon command that rollout actually reached.

    A clip contact is a finding, not a nuisance: the generator's ``ctrlrange`` is
    sized so the clip never engages inside the envelope, so touching it means the
    force law was driven somewhere the model does not describe, and every sample
    past it is a gradient of the saturation rather than of the plant.
    """
    out = {
        "available": True,
        "envelope_psi": float(envelope_psi),
        "force_clip_n": float(force_clip_n),
        "closed_form": [],
    }
    for s in scales:
        f = force_n(scaled(law, s), envelope_psi)
        peak = float(np.min(f))
        out["closed_form"].append({
            "scale": float(s),
            "peak_force_n": peak,
            "headroom_n": float(force_clip_n) + peak,
            "touches_clip": bool(peak <= -(float(force_clip_n) - R.CLAMP_EPS_N)),
        })
    if roll is not None:
        out["measured"] = {
            "ctrl_min_n": float(roll.ctrl_min_n),
            "headroom_n": float(force_clip_n) + float(roll.ctrl_min_n),
            "clamped": bool(roll.clamped),
            "n_rows": int(roll.n),
        }
    else:
        out["measured"] = None
        out["reason_no_measurement"] = ("no rollout supplied; the closed-form "
                                        "reading bounds the law but not what a "
                                        "trajectory actually commanded")
    return out


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def audit(law: ForceLaw, *, parts=PARTS, rec: "R.Recording | None" = None,
          roll: "R.Rollout | None" = None, settle_fn=None, poke_fn=None,
          rollout_fn=None, scales=DEFAULT_SCALES,
          real_ring_hz: "float | None" = None) -> dict:
    """Run the requested parts and return one dict.  Nothing is written anywhere.

    A part whose seam was not supplied reports ``available: False`` with its
    reason instead of being silently omitted, so a caller diffing two audits sees
    that a measurement stopped being taken rather than seeing its number vanish.
    """
    before = (law.coeff.copy(), law.bf.copy(), law.l0.copy())
    out = {"parts": list(parts), "source": law.source,
           "scales": [float(s) for s in scales]}
    if "geometry" in parts:
        out["geometry"] = audit_geometry(law)
    if "static" in parts:
        out["static"] = audit_static(law, settle_fn=settle_fn, scales=scales)
    if "ring" in parts:
        out["ring"] = audit_ring(law, poke_fn=poke_fn, scales=scales,
                                 real_ring_hz=real_ring_hz)
    if "replay" in parts:
        if rec is None:
            out["replay"] = {"available": False,
                             "reason": "no recording supplied"}
        else:
            out["replay"] = audit_replay(law, rec, rollout_fn=rollout_fn)
    if "clamp" in parts:
        out["clamp"] = audit_clamp(law, roll=roll, scales=scales)
    # The audit must be evidence, so it proves it left its subject alone rather
    # than only promising to.
    after = (law.coeff, law.bf, law.l0)
    out["model_unmodified"] = all(np.array_equal(a, b) for a, b in zip(before, after))
    if not out["model_unmodified"]:  # pragma: no cover - a defect, not a state
        raise RuntimeError(
            "force_audit modified the law it was auditing; every sweep must go "
            "through scaled(), which returns a new ForceLaw")
    return out


def main(argv=None) -> int:
    """``python -m digital_twin.force_audit <checkpoint.npz> [--parts geometry,clamp]``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="geometry and clamp only -- the two that need no simulator")
    args = ap.parse_args(argv)

    from digital_twin import actuator_model as AM  # deferred: sibling module
    law = force_law_from(AM.ActuatorModel.load(args.checkpoint))
    parts = ("geometry", "clamp") if args.quick else tuple(args.parts.split(","))
    out = audit(law, parts=parts)
    print(json.dumps(out, indent=1, default=str))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARTS", "ANCHOR_RADII_M", "OPERATOR_ACTUATOR_KG", "N_ACTUATORS", "ENVELOPE_PSI",
    "DEFAULT_SCALES", "SAVGOL_WINDOW", "SAVGOL_POLY", "SAVGOL_SKIP_S",
    "ForceLaw", "scaled", "force_law_from", "force_n", "stiffness_n_per_m",
    "audit_geometry", "audit_static", "audit_ring", "savgol_derivatives",
    "audit_replay", "audit_clamp", "audit", "main",
]
