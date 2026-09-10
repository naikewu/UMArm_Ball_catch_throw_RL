"""Fit the mechanical half, after the flow net has fixed the pneumatic half.

The twin has two halves and only one of them is a network.  The flow net turns
a commanded pressure into a pressure, and on this arm it does that well: 150
shooting epochs took the held-out pressure RMS from 20.3 kPa to about 1.4 kPa.
The force law then turns that pressure into a torque, and **its constants are
still the RS485 arm's** -- a different braid, a different bladder, a different
ring radius.  Nothing in the flow fit touches them.

That combination has a signature worth naming, because it looks like a
regression.  Before the flow net was fitted the twin barely pressurised, so the
arm hung near its rest pose and scored a *flattering* 14.8 deg of joint RMS by
not moving.  Once the pressures were right the same wrong force law converted
them into torques that swing the arm about two and a half times too far, and
the joint RMS went **up** to 29.0 deg while the thing being fitted got fourteen
times better.  A twin that does not move is not a good twin; it is an
unfalsifiable one.

So this module fits what the flow net cannot reach, in the coordinate the
mechanism actually has:

``coeff``          per segment, the force law's gain -- degrees of swing per psi
``tendon_damping`` global, the muscle's own dissipation
``joint_damping``  global, the bearing's

Multiplicative and in log space, exactly as the reference's ``outer_fit`` does,
so a factor of two up and a factor of two down are the same distance and the
search cannot walk a coefficient negative.

**It is fitted on the training families and scored on the held-out one.**
Fitting these three numbers on the sequence the twin is then judged by would be
the same mistake as fitting the net on its own holdout, and it is easier to
make here because the outer fit has so few parameters that it feels harmless.

Usage::

    .venv\\Scripts\\python.exe -m digital_twin.outer_fit \\
        --session data/session_20260910_013843 \\
        --checkpoint digital_twin/checkpoints/canarm_flow.npz \\
        --out digital_twin/checkpoints/canarm_outer.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from digital_twin import replay as R  # noqa: E402
from digital_twin import twin_compare as TC  # noqa: E402
from digital_twin.actuator_model import ActuatorModel  # noqa: E402

#: The parameters searched, in order, with the bounds of their log multiplier.
#: Far wider than the reference's +-30 %, and the width is a measurement rather
#: than caution.  Its seeds were that arm's own fitted values; these are another
#: arm's, and a sweep of the force gain against this arm's own deflections put
#: the seed **about sixty-five times too strong** -- at a multiplier of 1.0 the
#: twin's 95th-percentile deflection was 48.5 deg against the real arm's 17.7
#: and it spent 6 % of its samples pinned against a 40 deg joint limit, while at
#: 0.015 the two matched at 18.7 against 17.7. A +-3x bracket would have
#: searched entirely inside the saturated region, where the objective is nearly
#: flat because more torque and less torque both end at the stop -- which is
#: exactly what the first search found, and why the bounds are stated here with
#: the evidence rather than tuned quietly.
PARAMS = (
    ("coeff_seg1", math.log(1 / 300.0), math.log(2.0)),
    ("coeff_seg2", math.log(1 / 300.0), math.log(2.0)),
    ("coeff_seg3", math.log(1 / 300.0), math.log(2.0)),
    ("tendon_damping", math.log(0.1), math.log(10.0)),
    ("joint_damping", math.log(0.1), math.log(10.0)),
)

#: Where the coordinate search starts, as a log multiplier per parameter.  Not
#: zero for the force gain: starting at the seed puts the first round inside the
#: saturated region and every evaluation there is uninformative.  0.02 is the
#: order the deflection sweep found, and the search refines from it.
X0 = (math.log(0.02), math.log(0.02), math.log(0.02), 0.0, 0.0)

#: Seconds of training data each objective evaluation rolls.  A rollout costs
#: about 0.85 s of wall clock per second of recording, so this sets the search's
#: budget directly: 20 s of arm is about 17 s per evaluation.
EVAL_SECONDS = 20.0


def objective(rec, model, x, *, log=None):
    """Joint RMS in degrees for one log-multiplier vector.

    A rollout that touches the force clip returns a large finite loss rather
    than raising: the search must be able to step away from a saturated corner,
    and a raise would end it there.
    """
    m = ActuatorModel(
        net=model.net, is_tle=model.is_tle,
        fill_gain=model.fill_gain, vent_gain=model.vent_gain,
        blend_width_pa=model.blend_width_pa, leak_pa_s=model.leak_pa_s,
        coeff=model.coeff * np.exp(x[:3]),
        bf=model.bf, l0=model.l0, damp_b1=model.damp_b1)
    kwargs = dict(actuator=m,
                  tendon_damping=float(np.exp(x[3])),
                  joint_damping=0.026 * float(np.exp(x[4])))
    try:
        tw = TC.twin_rollout(rec, **kwargs)
    except Exception as exc:
        if log:
            log(f"    rollout raised: {exc!r}")
        return 1e6
    if not tw.valid:
        return 1e6
    err = tw.twin_defl_rad - tw.real_defl_rad
    good = np.isfinite(err)
    if good.sum() < 100:
        return 1e6
    return float(np.degrees(np.sqrt(np.nanmean(err[good] ** 2))))


def coordinate_search(rec, model, *, rounds: int = 2, points: int = 5,
                      log=print):
    """Coordinate descent in log space, refining the bracket each round.

    Nelder-Mead was the reference's choice and it is the better one with a
    generous evaluation budget; a rollout here costs seventeen seconds, so a
    coordinate sweep that spends its evaluations on a grid it can reason about
    beats a simplex that spends them exploring.  Five points per parameter per
    round is coarse on purpose: the seeds are another arm's and the first round
    is looking for the right factor, not the right percent.
    """
    x = np.array(X0, dtype=float)
    best = objective(rec, model, x, log=log)
    log("  start: " + ", ".join(f"{n} x{np.exp(x[i]):.3g}" for i, (n, _, _) in enumerate(PARAMS)) + f" -> {best:.3f} deg")
    for r in range(rounds):
        span = 1.0 / (r + 1.0)
        for i, (name, lo, hi) in enumerate(PARAMS):
            grid = np.clip(x[i] + np.linspace(-1.0, 1.0, points) *
                           span * (hi - lo) * 0.5, lo, hi)
            vals = []
            for g in grid:
                if abs(g - x[i]) < 1e-9:
                    vals.append(best)
                    continue
                trial = x.copy()
                trial[i] = g
                vals.append(objective(rec, model, trial, log=log))
            k = int(np.argmin(vals))
            if vals[k] < best - 1e-6:
                x[i] = grid[k]
                best = vals[k]
            log(f"  round {r} {name:<15s} x{np.exp(x[i]):5.2f}  "
                f"best {best:.3f} deg")
    return x, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True, action="append")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="digital_twin/checkpoints/canarm_outer.json")
    ap.add_argument("--fit-kinds", default="random_walk",
                    help="excitation families the search is allowed to see; "
                         "NEVER the one it will be scored on")
    ap.add_argument("--eval-seconds", type=float, default=EVAL_SECONDS)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--points", type=int, default=5)
    args = ap.parse_args(argv)

    kinds = [k.strip() for k in args.fit_kinds.split(",") if k.strip()]
    model = ActuatorModel.load(args.checkpoint)

    rec = None
    for sess in args.session:
        try:
            r = R.recording_from_session(sess, kinds=kinds)
        except Exception:
            continue
        rec = r
        break
    if rec is None:
        print(f"no session carries {kinds}")
        return 1
    rec = rec.slice_time(args.eval_seconds)
    print(f"fitting on {kinds} from {rec.path}: {rec.n} cycles, "
          f"{rec.duration_s:.1f} s")

    t0 = time.perf_counter()
    x, best = coordinate_search(rec, model, rounds=args.rounds,
                                points=args.points)
    took = time.perf_counter() - t0

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "fit_kinds": kinds,
        "fit_seconds_of_arm": rec.duration_s,
        "search_seconds": took,
        "multipliers": {name: float(np.exp(x[i]))
                        for i, (name, _, _) in enumerate(PARAMS)},
        "coeff": (model.coeff * np.exp(x[:3])).tolist(),
        "tendon_damping": float(np.exp(x[3])),
        "joint_damping": 0.026 * float(np.exp(x[4])),
        "fit_joint_rms_deg": best,
        "note": ("fitted on the training families only; the held-out "
                 "validation sequence was never rolled during this search"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"search took {took / 60:.1f} min; best {best:.3f} deg on the fit "
          f"slice")
    print("multipliers: " + ", ".join(
        f"{k} x{v:.2f}" for k, v in result["multipliers"].items()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
