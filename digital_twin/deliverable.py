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
   sequence produced, and add the per-joint correlation beside the RMS;
4. render three panels -- the bench camera, the model posed at the *measured*
   joint angles, and the model posed at the *predicted* ones -- and write the
   metrics into the frame so the picture carries its own evidence;
5. save one frame near :data:`FRAME_AT_S` as ``frame_sample.jpg``, so the picture
   can be checked without playing 33 MB of mp4.

THE TWIN IT SCORES is the one ``twin_params.load_twin_kwargs`` builds from the
flow checkpoint and the mechanical file, which is also the twin the operator
GUI's SIM adapter runs.  Until 2026-09-10 this module rebuilt the
``ActuatorModel`` itself from ``--outer`` and copied two damping scalars across,
so a mass, a ``bf`` or a friction in a newer file would have been dropped here
silently while the GUI used it.  ``--outer`` still works; it is now simply
another mechanical file handed to the same loader.

The camera panel is dropped, with a note, when the bench camera did not record
that session.  The comparison that carries the claim is the middle panel against
the right one: same renderer, same camera, same model, differing only in where
the joint angles came from.

Usage::

    .venv\\Scripts\\python.exe -m digital_twin.deliverable \\
        --session data/session_20260910_013843 --out-dir deliverable

``--checkpoint`` defaults to ``checkpoints/canarm_flow.npz`` (``none`` rolls the
unfitted seed net) and ``--mech`` to ``checkpoints/canarm_mech.json``, falling
back to ``canarm_outer.json`` exactly as ``twin_params`` does.
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
from digital_twin import twin_params as TP  # noqa: E402

PA_PER_PSI = 6894.757

#: s.  Where ``frame_sample.jpg`` is taken.  35 s is inside the second of the
#: three 30 s validation blocks, well clear of the start where both traces sit
#: near their reference and every twin looks good.
FRAME_AT_S = 35.0

#: The mechanical file's keys worth copying into the deliverable report.  The
#: whole file is not copied: its per-generation history and its bound evidence
#: belong to the checkpoint, and the report names the checkpoint's path.
MECH_SUMMARY_KEYS = ("kind", "date", "status", "coeff", "bf", "tendon_damping",
                     "joint_damping", "joint_frictionloss", "mjcf",
                     "moving_mass_kg", "rest_gain_n_per_psi", "bf_over_l0",
                     "multipliers", "fit_kinds", "fit_joint_rms_deg", "note")


def load_outer(path):
    """A parsed mechanical JSON, or ``None``.  Kept for callers of the old name."""
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def twin_for(checkpoint, mech, *, log=print):
    """``(kwargs, mech_doc)``: the twin through ``twin_params``, and the file it read.

    *checkpoint* ``None`` or ``"none"`` rolls ``ActuatorModel.fresh``, which the
    loader logs as UNFITTED.  *mech* is any file in the twin-parameter schema,
    ``canarm_outer.json`` included.
    """
    flow = None if checkpoint in (None, "", "none", "seed") else checkpoint
    kwargs = TP.load_twin_kwargs(flow=flow, mech=mech, log=log)
    path = kwargs.provenance.get("mech")
    doc = load_outer(path) if path else None
    return kwargs, doc


def mech_summary(doc):
    if not doc:
        return None
    out = {k: doc[k] for k in MECH_SUMMARY_KEYS if k in doc}
    fit = doc.get("fit") or {}
    if fit:
        out["fit"] = {k: fit[k] for k in ("loss_deg", "start_loss_deg", "generations_done",
                                          "evaluations", "seconds_of_arm_rolled",
                                          "elapsed_h") if k in fit}
    return out


def correlation(tw) -> dict:
    """Pearson r between the twin's deflection and the arm's, per joint.

    Over the rows with a finite pose on all twelve joints, the rows
    ``compare_metrics`` scores.  Correlation reads shape and phase and is blind
    to scale: a twin that moves twice as far, in step, scores 1.0.  That is why it
    sits beside the RMS and never replaces it.
    """
    real = np.asarray(tw.real_defl_rad, dtype=float)
    sim = np.asarray(tw.twin_defl_rad, dtype=float)
    use = np.all(np.isfinite(real), axis=1) & np.all(np.isfinite(sim), axis=1)
    r = []
    for j in range(real.shape[1]):
        a, b = sim[use, j], real[use, j]
        r.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0
                 else float("nan"))
    r = np.asarray(r)
    return {"per_joint": r.tolist(), "mean": float(np.nanmean(r)),
            "per_segment": [float(np.nanmean(r[i:i + 4])) for i in (0, 4, 8)],
            "n": int(use.sum())}


def score(rec, twin_kwargs, *, t_max_s=None, log=print):
    """Open-loop rollout and metrics.  Returns ``(metrics, TwinResult)``."""
    log(f"rolling {rec.n} cycles ({rec.duration_s:.1f} s at "
        f"{rec.rate_hz:.2f} Hz) through the twin, open loop")
    t0 = time.perf_counter()
    tw = TC.twin_rollout(rec, t_max_s=t_max_s, **twin_kwargs)
    log(f"  rollout took {time.perf_counter() - t0:.1f} s")
    if not tw.valid:
        return {"valid": False, "reason": tw.reason}, tw
    out = TC.compare_metrics(rec, tw)
    out["correlation"] = correlation(tw)
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
    c = out.get("correlation")
    if c:
        log(f"  mean per-joint correlation {c['mean']:+.3f}; segments "
            + " / ".join(f"{v:+.3f}" for v in c["per_segment"]))
    p = ov.get("pressures")
    if p:
        log(f"  pressure RMS {p.get('rms_pa_mean', float('nan')) / PA_PER_PSI:.3f} psi mean")


def write_frame(video_path: str, t_s: float, out_path: str):
    """One frame of the rendered video near *t_s*, as a JPEG.  ``None`` if unreadable."""
    cv2 = SBS._cv2()
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or SBS.OUT_FPS)
        idx = int(round(float(t_s) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return {"path": out_path, "frame": idx, "t_s": idx / fps,
            "bytes": os.path.getsize(out_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--checkpoint", default=str(TP.DEFAULT_FLOW),
                    help="the flow net; 'none' rolls the unfitted seed net")
    ap.add_argument("--mech", default=str(TP.DEFAULT_MECH),
                    help="the mechanical file (twin_params schema): masses, force "
                         "law, dissipation; canarm_mech.json from mech_fit")
    ap.add_argument("--outer", default=None,
                    help="legacy spelling of --mech for canarm_outer.json; when "
                         "given it is used as the mechanical file")
    ap.add_argument("--out-dir", default="deliverable")
    ap.add_argument("--kind", default="validation",
                    help="excitation family to score and film")
    ap.add_argument("--t-max-s", type=float, default=None)
    ap.add_argument("--joints", default="0,4,8",
                    help="joints drawn in the trace strip")
    ap.add_argument("--fps", type=float, default=SBS.OUT_FPS)
    ap.add_argument("--frame-at-s", type=float, default=FRAME_AT_S)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rec = R.recording_from_session(args.session, kinds=[args.kind])
    print(f"held-out {args.kind}: {rec.n} cycles, {rec.duration_s:.1f} s, "
          f"{rec.rate_hz:.2f} Hz")

    mech = args.outer if args.outer else args.mech
    kwargs, mech_doc = twin_for(args.checkpoint, mech)
    out, tw = score(rec, kwargs, t_max_s=args.t_max_s)
    summarise(out)

    prov = getattr(kwargs, "provenance", {}) or {}
    report = {
        "session": os.path.abspath(args.session),
        "twin": {"describe": TP.describe(kwargs), "flow": prov.get("flow"),
                 "mech": prov.get("mech"),
                 "mech_fallback_from": prov.get("mech_fallback_from")},
        "mech": mech_summary(mech_doc),
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
        frame = write_frame(vid["path"], args.frame_at_s,
                            os.path.join(args.out_dir, "frame_sample.jpg"))
        if frame:
            print(f"wrote {frame['path']} (frame {frame['frame']}, t {frame['t_s']:.2f} s)")
        report["frame_sample"] = frame

    path = os.path.join(args.out_dir, f"deliverable_{args.kind}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
