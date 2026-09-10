"""Score the twin on the held-out sequence and render the side-by-side video.

One command, because the two halves of the deliverable have to be the same
rollout.  Scoring one rollout and filming another would let a good number and a
bad picture coexist without either being wrong, and the whole point of the video
is that a reader can check the number against their own eyes.

What it does, in order:

1. read the **held-out** ``validation`` segment -- a sequence generated from a
   different seed and a different signal family than anything trained on;
2. roll it through the twin **open loop**: the twin sees only the recorded
   pressure targets, never the recorded pressures and never the recorded joints,
   so every divergence accumulates rather than being corrected;
3. score it, per joint and split by population, against the mocap the same
   sequence produced;
4. render three panels -- the bench camera, the model posed at the *measured*
   joint angles, and the model posed at the *predicted* ones -- and write the
   metrics into the frame so the picture carries its own evidence.

The camera panel is dropped, with a note, when the bench camera did not record
that session.  The comparison that carries the claim is the middle panel against
the right one: same renderer, same camera, same model, differing only in where
the joint angles came from.

Usage::

    .venv\\Scripts\\python.exe -m digital_twin.deliverable \\
        --session data/session_20260910_013843 \\
        --checkpoint digital_twin/checkpoints/canarm_flow.npz \\
        --out-dir deliverable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_WS, os.path.join(_WS, "TLE_PCB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from digital_twin import replay as R  # noqa: E402
from digital_twin import side_by_side as SBS  # noqa: E402
from digital_twin import twin_compare as TC  # noqa: E402

PA_PER_PSI = 6894.757


def load_outer(path):
    """The mechanical multipliers, or ``None``.

    Kept as a separate file from the checkpoint because they are fitted by a
    different procedure against a different objective: the checkpoint is the
    flow net, fitted on pressure residuals, and these are the force law and the
    dissipation, fitted on joint residuals. Merging them would make it possible
    to load half of a fit without noticing.
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def score(rec, *, checkpoint=None, outer=None, t_max_s=None, log=print) -> dict:
    """Open-loop rollout and metrics.  Returns the result and the numbers."""
    kwargs = {}
    if checkpoint:
        from digital_twin.actuator_model import ActuatorModel
        model = ActuatorModel.load(checkpoint)
        if outer:
            model = ActuatorModel(
                net=model.net, is_tle=model.is_tle,
                fill_gain=model.fill_gain, vent_gain=model.vent_gain,
                blend_width_pa=model.blend_width_pa, leak_pa_s=model.leak_pa_s,
                coeff=np.asarray(outer["coeff"], dtype=float),
                bf=model.bf, l0=model.l0, damp_b1=model.damp_b1)
            kwargs["tendon_damping"] = float(outer["tendon_damping"])
            kwargs["joint_damping"] = float(outer["joint_damping"])
            log("outer fit applied: " + ", ".join(
                f"{k} x{v:.3g}" for k, v in outer["multipliers"].items()))
        kwargs["actuator"] = model
        log(f"using {checkpoint}")
        m = getattr(model, "meta", None) or {}
        if m.get("holdout_rms_psi") is not None:
            log(f"  its own fit reported a {m['holdout_rms_psi']:.3f} psi "
                f"holdout pressure RMS over {m.get('holdout_windows')} windows")
    else:
        log("no checkpoint: rolling the UNFITTED seed model. Its constants are "
            "the RS485 arm's, for a different actuator, so this scores the "
            "geometry and the firmware model and nothing else.")
    log(f"rolling {rec.n} cycles ({rec.duration_s:.1f} s at "
        f"{rec.rate_hz:.2f} Hz) through the twin, open loop")
    t0 = time.perf_counter()
    tw = TC.twin_rollout(rec, t_max_s=t_max_s, **kwargs)
    log(f"  rollout took {time.perf_counter() - t0:.1f} s")
    if not tw.valid:
        return {"valid": False, "reason": tw.reason}, tw
    out = TC.compare_metrics(rec, tw)
    return out, tw


def summarise(out: dict, log=print) -> None:
    if not out.get("valid"):
        log("INVALID: " + str(out.get("reason")))
        return
    ov = out["overall"]
    if not ov.get("scored"):
        log("nothing scored: " + str(ov))
        return
    j = ov["joints"]
    log(f"  joint RMS {j['rms_deg_mean']:.3f} deg mean over "
        f"{ov.get('n')} cycles")
    log(f"  nrmse {j['nrmse_mean']:.3f}  "
        f"(1.0 means no better than predicting 'it does not move')")
    for pop, blk in j.get("by_population", {}).items():
        log(f"    {pop:>7}: rms {blk['rms_mean']:.3f} deg over "
            f"{blk['n_columns']} joints, nrmse {blk['nrmse_mean']:.3f}")
    p = ov.get("pressures")
    if p:
        log(f"  pressure RMS {p.get('rms_pa_mean', float('nan')) / PA_PER_PSI:.3f} psi mean")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--outer", default=None,
                    help="canarm_outer.json from digital_twin.outer_fit -- the "
                         "force law and dissipation, which the flow fit does "
                         "not reach")
    ap.add_argument("--out-dir", default="deliverable")
    ap.add_argument("--kind", default="validation",
                    help="excitation family to score and film")
    ap.add_argument("--t-max-s", type=float, default=None)
    ap.add_argument("--joints", default="0,4,8",
                    help="joints drawn in the trace strip")
    ap.add_argument("--fps", type=float, default=SBS.OUT_FPS)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rec = R.recording_from_session(args.session, kinds=[args.kind])
    print(f"held-out {args.kind}: {rec.n} cycles, {rec.duration_s:.1f} s, "
          f"{rec.rate_hz:.2f} Hz")

    outer = load_outer(args.outer)
    out, tw = score(rec, checkpoint=args.checkpoint, outer=outer,
                    t_max_s=args.t_max_s)
    summarise(out)

    report = {
        "session": os.path.abspath(args.session),
        "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
        "outer": outer,
        "kind": args.kind,
        "cycles": int(rec.n),
        "seconds": float(rec.duration_s),
        "metrics": out,
    }

    if not args.no_video and out.get("valid"):
        # Deflections, not absolute angles: the twin's base pose and the mocap
        # volume's origin are different conventions and the constant between
        # them is not something either side measured.  twin_rollout removes
        # exactly that constant by referencing both to the same settled window.
        t = tw.t_rel_s
        q_real = tw.real_defl_rad
        q_sim = tw.twin_defl_rad
        good = np.isfinite(q_real).all(axis=1)
        t, q_real, q_sim = t[good], q_real[good], q_sim[good]

        clip = SBS.find_clip(args.session, kind=args.kind)
        meta = SBS.load_metadata(args.session)
        if clip is None:
            print("no camera clip for this session -- the video will carry the "
                  "measured and predicted panels only. The bench camera "
                  "powered itself off during collection; see the session "
                  "report's camera_error.")
        joints = tuple(int(v) for v in args.joints.split(","))
        vid = SBS.compose(
            os.path.join(args.out_dir, f"twin_vs_real_{args.kind}.mp4"),
            t=t, q_real=q_real, q_sim=q_sim,
            clip=clip[0] if clip else None,
            clip_meta=clip[1] if clip else None,
            joints=joints, fps=args.fps,
            clock_offset_s=0.0,
            title=(f"open loop, held-out {args.kind}; the twin sees only the "
                   f"recorded pressure targets"),
            metrics=out.get("overall", {}))
        print(f"wrote {vid['path']}: {vid['frames']} frames, "
              f"{vid['seconds']:.1f} s, {vid['bytes'] / 1e6:.1f} MB")
        report["video"] = vid

    path = os.path.join(args.out_dir, f"deliverable_{args.kind}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
