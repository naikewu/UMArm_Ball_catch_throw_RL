"""Move the Gen3 through a small ball, watch it in the cameras, solve the pair.

    .venv_kinova/Scripts/python.exe UMArm_KINOVA/calibrate_mocap.py --dry-run
    .venv_kinova/Scripts/python.exe UMArm_KINOVA/calibrate_mocap.py --yes
    .venv_kinova/Scripts/python.exe UMArm_KINOVA/calibrate_mocap.py \
        --analyze-only UMArm_KINOVA/results/<run>/campaign.json

THE CAMPAIGN.  Twenty-one poses inside a **20 cm-diameter ball** centred on the
arm's Home pose, which is the operator's stated limit and is enforced by
:class:`~UMArm_KINOVA.kinova_arm.SafeKinovaArm` rather than merely respected by
the plan.  The poses are of two kinds, and the split is what makes the fit
identifiable:

* **thirteen pure translations** — the axes, the face diagonals and two
  off-plane points, all at the Home orientation.  Every pair of these answers
  the operator's question directly (:func:`~UMArm_KINOVA.mocap_calibration.axis_fidelity`):
  the lever arm cancels, so a commanded displacement and a measured one can be
  compared with no fitted quantity in between.
* **eight orientation changes** — +-15 deg about the tool's own x and y, +-20 deg
  about its z, and two mixed poses off-centre.  These exist because a campaign
  of translations alone cannot see the marker body's rotation OR the lever arm
  from the tool to it: both cancel out of every pure translation, and the fit
  would come back with a beautiful residual and an arbitrary ``Y``.  Applied in
  the TOOL frame so the tool origin stays at the ball's centre and the envelope
  never binds on a rotation.

EVERY POSE IS RECORDED AS IT WAS ACHIEVED, not as it was asked for.  The solve
consumes the arm's own ``tool_pose`` feedback; the commanded pose is kept beside
it only so the report can separate "the arm did not go where it was told" from
"the cameras and the arm disagree about where it went".

SAFETY, in the order it acts:

1. ``--radius`` is validated against the envelope before anything is contacted,
   and ``--dry-run`` reports every pose that would not fit a ball of that radius
   about the pose it assumes — so an over-reaching plan is a refusal at the desk.
   The plan is checked AGAIN against the real armed centre once the session is
   open, because only then is the centre known;
2. every move is speed-limited (``--speed``, default 30 mm/s) and blocking;
3. every move's arrival is verified to 2 mm / 1 deg, and a miss aborts that pose
   rather than being averaged into the fit;
4. a failed pose returns the arm to the ENVELOPE CENTRE — not to Home, which
   under ``--no-home`` is somewhere else entirely — and the campaign continues
   without that pose;
5. the arm is sent Home at the end from a ``finally``, whatever happened, in a
   fresh session in case the old one is what broke.

Run ``--dry-run`` first: it prints the whole plan with each pose's distance from
the centre and touches neither the arm nor the network.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from UMArm_KINOVA import mocap_calibration as MC        # noqa: E402
from UMArm_KINOVA.kinova_arm import (                    # noqa: E402
    ArrivalError, DEFAULT_ENVELOPE_R_M, DEFAULT_IP, EnvelopeViolation,
    SafeKinovaArm, envelope_violations, pose_delta, tool_configuration)
from UMArm_KINOVA.vendor.kinova_driver import (          # noqa: E402
    SE3_to_pose, pose_to_SE3)

RESULTS_DIR = os.path.join(_HERE, "results")

#: Translation radius of the campaign, metres.  The envelope allows 100 mm; the
#: plan stops at 70 mm so that a refusal means a mistake rather than a rounding
#: error at the wall, and the two diagonal families stay inside it too.
PLAN_R_M = 0.070

#: Wrist excursions, degrees, applied in the TOOL frame.  Large enough that
#: Motive's ~0.02 deg orientation jitter is four hundred times smaller, small
#: enough that Home stays far from a joint limit.
PLAN_ROT_DEG = 15.0
PLAN_TWIST_DEG = 20.0

#: Seconds to wait after a move before the cameras are believed.  The Gen3's
#: own settling is a few tens of milliseconds; the rest is the printed bracket
#: and its markers, which are the flexible part of this assembly.
SETTLE_S = 1.0
#: Seconds of mocap averaged per pose, and arm-feedback samples per pose.
CAPTURE_S = 1.0
ARM_SAMPLES = 20

#: Peak-to-peak spread that says the arm had not settled when it was measured.
#: Motive's own jitter on this body was measured at 0.08 mm and 0.09 deg
#: peak-to-peak over two still seconds (2026-08-19), so 0.5 mm is six times the
#: noise floor and still tight enough to catch a bracket that is still ringing.
STILL_POS_PTP_MM = 0.5
STILL_ANG_PTP_DEG = 0.5


def build_plan(r_m: float = PLAN_R_M, rot_deg: float = PLAN_ROT_DEG,
               twist_deg: float = PLAN_TWIST_DEG) -> list:
    """``[(label, (dx, dy, dz), (rx, ry, rz)), ...]`` relative to the centre.

    Translations are in the BASE frame; rotations are applied in the TOOL frame
    (``T = T_centre_translated @ Rot``), so they turn the tool about itself and
    leave its origin where the translation put it.
    """
    d = r_m
    # EXACTLY 1/sqrt(2), not 0.71: at 0.71 the four face diagonals sit at
    # 1.0041 r, so a campaign run at the envelope's own radius would put four
    # poses 0.4 mm OUTSIDE it and be refused before it started.
    s = r_m * (0.5 ** 0.5)               # face diagonals, same radius
    # The off-plane and mixed poses use TWO components of t each, so they land
    # at r * sqrt(2/3) = 0.82 r.  Deliberately inside the shell: a cloud whose
    # every point is on one sphere is a worse-conditioned set for a translation
    # solve than one with an interior.
    t = r_m * (1.0 / 3.0) ** 0.5
    plan = [
        ("centre", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("x+", (+d, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("x-", (-d, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("y+", (0.0, +d, 0.0), (0.0, 0.0, 0.0)),
        ("y-", (0.0, -d, 0.0), (0.0, 0.0, 0.0)),
        ("z+", (0.0, 0.0, +d), (0.0, 0.0, 0.0)),
        ("z-", (0.0, 0.0, -d), (0.0, 0.0, 0.0)),
        ("xy++", (+s, +s, 0.0), (0.0, 0.0, 0.0)),
        ("xy-+", (-s, +s, 0.0), (0.0, 0.0, 0.0)),
        ("xy+-", (+s, -s, 0.0), (0.0, 0.0, 0.0)),
        ("xy--", (-s, -s, 0.0), (0.0, 0.0, 0.0)),
        ("xz++", (+t, 0.0, +t), (0.0, 0.0, 0.0)),
        ("yz--", (0.0, -t, -t), (0.0, 0.0, 0.0)),
        ("rx+", (0.0, 0.0, 0.0), (+rot_deg, 0.0, 0.0)),
        ("rx-", (0.0, 0.0, 0.0), (-rot_deg, 0.0, 0.0)),
        ("ry+", (0.0, 0.0, 0.0), (0.0, +rot_deg, 0.0)),
        ("ry-", (0.0, 0.0, 0.0), (0.0, -rot_deg, 0.0)),
        ("rz+", (0.0, 0.0, 0.0), (0.0, 0.0, +twist_deg)),
        ("rz-", (0.0, 0.0, 0.0), (0.0, 0.0, -twist_deg)),
        ("mix+", (+t, +t, 0.0), (+10.0, -10.0, 0.0)),
        ("mix-", (-t, 0.0, -t), (-10.0, +10.0, +15.0)),
    ]
    return plan


def plan_poses(centre_pose, plan) -> list:
    """``[(label, pose6), ...]`` — the plan resolved against a centre pose."""
    T0 = pose_to_SE3(centre_pose)
    out = []
    for label, dpos, drot in plan:
        T = T0.copy()
        T[0:3, 3] = T0[0:3, 3] + np.asarray(dpos, dtype=float)
        if any(abs(v) > 1e-12 for v in drot):
            Trot = MC.se3(MC.rot_exp(np.deg2rad(drot)), [0.0, 0.0, 0.0])
            T = T @ Trot
        out.append((label, list(SE3_to_pose(T))))
    return out


# --------------------------------------------------------------------------- #
# The campaign
# --------------------------------------------------------------------------- #
def run_campaign(ip: str = DEFAULT_IP, server_ip=None, client_ip=None,
                 r_m: float = PLAN_R_M, speed_ms: float = 0.030,
                 settle_s: float = SETTLE_S, capture_s: float = CAPTURE_S,
                 go_home: bool = True, log=print) -> dict:
    """Drive the arm through the plan and return the raw record."""
    from UMArm_KINOVA.kinova_mocap import KinovaMocapRx

    kw = {}
    if server_ip:
        kw["server_ip"] = server_ip
    if client_ip:
        kw["client_ip"] = client_ip
    rx = KinovaMocapRx(**kw)
    rx.start()
    record = {
        "captured_wall": datetime.datetime.now().isoformat(timespec="seconds"),
        "ip": ip, "plan_r_m": float(r_m), "speed_ms": float(speed_ms),
        "settle_s": float(settle_s), "capture_s": float(capture_s),
        "mocap": {"server_ip": rx.server_ip, "client_ip": rx.client_ip,
                  "multicast": bool(rx.use_multicast),
                  "stream_id": 1008},
        "samples": [], "skipped": [],
    }
    try:
        with SafeKinovaArm(ip=ip) as arm:
            if go_home:
                log("sending the arm Home ...")
                arm.move_home()
            centre = arm.arm_envelope()
            record["centre_pose"] = list(centre)
            record["envelope_r_m"] = arm.envelope_r_m
            record["tool_configuration"] = tool_configuration(arm)
            record["joints_home_deg"] = arm.get_joint_angles()
            log("envelope centre %s, radius %.0f mm"
                % (" ".join("%.4f" % v for v in centre[0:3]),
                   arm.envelope_r_m * 1e3))

            poses = plan_poses(centre, build_plan(r_m))
            # Check the WHOLE plan against the REAL centre before moving
            # anything further.  ``main`` has already rejected a radius that
            # cannot fit; this catches a centre that is not where it was assumed.
            for label, pose in poses:
                arm.check_envelope(pose)

            for i, (label, pose) in enumerate(poses):
                log("[%2d/%2d] %-6s -> %s"
                    % (i + 1, len(poses), label,
                       " ".join("%8.4f" % v for v in pose)))
                try:
                    arm.move_to_pose(pose, speed_ms=speed_ms, name=label)
                except (ArrivalError, EnvelopeViolation) as exc:
                    log("        SKIPPED: %s" % exc)
                    record["skipped"].append({"label": label,
                                              "commanded": list(pose),
                                              "why": str(exc)})
                    # Recover to the ENVELOPE CENTRE, not to Home.  They are the
                    # same point in the normal case and are not under --no-home,
                    # where Home is a pose with no relation to the armed
                    # envelope — throwing the arm there would both contradict
                    # the flag and leave the region the operator authorised.
                    try:
                        arm.move_to_pose(centre, speed_ms=speed_ms,
                                         name="recover")
                    except (ArrivalError, EnvelopeViolation) as back:
                        log("        could not return to the centre: %s" % back)
                        record["skipped"][-1]["recovery"] = str(back)
                    continue
                mocap = rx.capture(seconds=capture_s, settle=settle_s)
                snap = arm.averaged_snapshot(n=ARM_SAMPLES)
                still = (mocap["pos_ptp_mm"] <= STILL_POS_PTP_MM
                         and mocap["ang_ptp_deg"] <= STILL_ANG_PTP_DEG)
                T = mocap.pop("T")
                record["samples"].append({
                    "label": label,
                    "commanded_pose": list(pose),
                    "arm_pose": snap["pose"],
                    "arm_pose_sd": snap["pose_sd"],
                    "arm_pos_ptp_m": snap["pos_ptp_m"],
                    "joints_deg": snap["joints_deg"],
                    "rb_T_world": [list(map(float, row)) for row in T],
                    "mocap": mocap,
                    "still": bool(still),
                })
                d, ang = pose_delta(pose, snap["pose"])
                log("        arrived %.2f mm / %.3f deg off; mocap %d frames, "
                    "ptp %.3f mm / %.3f deg%s"
                    % (d * 1e3, ang, mocap["n"], mocap["pos_ptp_mm"],
                       mocap["ang_ptp_deg"], "" if still else "  << NOT STILL"))
    finally:
        # The return home is in a FINALLY, because the docstring promises the
        # arm is parked "whatever happened" and a promise on the normal path is
        # not that promise.  It opens its own session: whatever brought us here
        # may have been the session itself.  Its own failure is reported and
        # swallowed, so it cannot mask the exception that caused the abort.
        if go_home:
            try:
                log("returning Home ...")
                with SafeKinovaArm(ip=ip) as parking:
                    parking.move_home()
            except Exception as exc:                # noqa: BLE001 - see above
                log("COULD NOT RETURN HOME: %s: %s — park the arm by hand"
                    % (type(exc).__name__, exc))
                record["park_failed"] = "%s: %s" % (type(exc).__name__, exc)
        rx.stop()
    record["mocap"]["frames_seen"] = rx.kinova_frames
    return record


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #
def _stacks(record, use_commanded: bool = False):
    """``(labels, M, N)`` — arm poses and mocap poses as ``(n, 4, 4)`` stacks."""
    rows = [s for s in record["samples"] if s.get("still", True)]
    labels = [s["label"] for s in rows]
    key = "commanded_pose" if use_commanded else "arm_pose"
    M = np.array([pose_to_SE3(s[key]) for s in rows])
    N = np.array([np.array(s["rb_T_world"], dtype=float) for s in rows])
    return labels, M, N


def analyse(record: dict) -> dict:
    """Fit both transforms and answer the questions the campaign was run for."""
    labels, M, N = _stacks(record)
    res = MC.fit(M, N)
    X, Y = res["X_world_base"], res["Y_tool_rb"]

    # The operator's question, on the pure-translation pairs only.
    fid = MC.axis_fidelity(M, N, X[0:3, 0:3])
    # The same test against what was COMMANDED rather than what the arm
    # reported reaching: the difference between the two is the arm's own
    # tracking error, and keeping them apart is the only way to attribute a
    # disagreement to the right machine.
    _, Mc, Nc = _stacks(record, use_commanded=True)
    fid_cmd = MC.axis_fidelity(Mc, Nc, X[0:3, 0:3])
    # How well the arm reached what it was asked for, on its own terms.
    track = [pose_delta(s["commanded_pose"], s["arm_pose"])
             for s in record["samples"] if s.get("still", True)]
    track_mm = np.array([t[0] for t in track]) * 1e3
    track_deg = np.array([t[1] for t in track])

    out = {
        "n_samples": int(M.shape[0]),
        "n_skipped": len(record.get("skipped", [])),
        "labels": labels,
        "X_world_base": X.tolist(),
        "Y_tool_rb": Y.tolist(),
        "Y_translation_mm": (Y[0:3, 3] * 1e3).tolist(),
        "Y_lever_arm_mm": float(np.linalg.norm(Y[0:3, 3]) * 1e3),
        "Y_rotation_rpy_deg": _rpy_deg(Y[0:3, 0:3]),
        "X_translation_m": X[0:3, 3].tolist(),
        "X_rotation_rpy_deg": _rpy_deg(X[0:3, 0:3]),
        "X_yaw_deg": float(np.degrees(np.arctan2(X[1, 0], X[0, 0]))),
        "residual": res["residual"],
        "residual_closed_form": res["residual_closed_form"],
        "closed_form_info": res["closed_form_info"],
        "observability": res["observability"],
        "axis_fidelity_arm_reported": fid,
        "axis_fidelity_commanded": fid_cmd,
        "axis_fidelity_by_axis": MC.axis_fidelity_by_axis(M, N, X[0:3, 0:3]),
        "scale_fit": _jsonable(MC.fit_with_scale(M, N)),
        "arm_tracking": {
            "pos_rms_mm": float(np.sqrt(np.mean(track_mm ** 2))),
            "pos_max_mm": float(track_mm.max()),
            "ang_rms_deg": float(np.sqrt(np.mean(track_deg ** 2))),
            "ang_max_deg": float(track_deg.max()),
        },
        "base_recovery": MC.base_recovery_spread(M, N, Y, X),
        "mocap_noise": {
            "pos_ptp_mm_max": float(max(s["mocap"]["pos_ptp_mm"]
                                        for s in record["samples"])),
            "ang_ptp_deg_max": float(max(s["mocap"]["ang_ptp_deg"]
                                         for s in record["samples"])),
        },
        "per_sample": [
            {"label": labels[i],
             "pos_mm": res["residual"]["pos_mm"][i],
             "ang_deg": res["residual"]["ang_deg"][i]}
            for i in range(len(labels))],
    }
    out["leave_one_out"] = _leave_one_out(M, N)
    return out


def _jsonable(d: dict) -> dict:
    """Turn the arrays in a fit result into lists, leaving scalars alone."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in d.items()}


def _rpy_deg(R) -> list:
    """Extrinsic-xyz Euler angles, degrees — the convention this repo uses."""
    from scipy.spatial.transform import Rotation

    return [float(v) for v in Rotation.from_matrix(R).as_euler("xyz",
                                                               degrees=True)]


def _leave_one_out(M, N) -> dict:
    """Refit without each sample and predict the one left out.

    An in-sample residual can always be driven down by a model with enough
    freedom; neither of the two numbers below can.

    * ``pos_*`` - how far the refit places the rigid body of the pose it never
      saw.  That is what the calibration will actually be asked to do.
    * ``base_*`` - how far the BASE, back-solved from that one unseen frame
      through the refit's own ``Y``, lands from the refit's own ``X``.  This is
      the out-of-sample version of
      :func:`~UMArm_KINOVA.mocap_calibration.base_recovery_spread`, which uses a
      ``Y`` fitted from the very frames it then back-solves, and it is the
      number the "do we need a base marker body" question turns on.
    """
    errs, base = [], []
    n = M.shape[0]
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        try:
            r = MC.fit(M[keep], N[keep])
        except ValueError:
            continue
        P = r["X_world_base"] @ M[i] @ r["Y_tool_rb"]
        errs.append(float(np.linalg.norm(N[i, 0:3, 3] - P[0:3, 3]) * 1e3))
        Xi = MC.base_from_single_sample(M[i], N[i], r["Y_tool_rb"])
        base.append(float(np.linalg.norm(Xi[0:3, 3]
                                         - r["X_world_base"][0:3, 3]) * 1e3))
    errs = np.array(errs) if errs else np.zeros(1)
    base = np.array(base) if base else np.zeros(1)
    return {"pos_rms_mm": float(np.sqrt(np.mean(errs ** 2))),
            "pos_max_mm": float(errs.max()),
            "per_sample_mm": errs.tolist(),
            "base_rms_mm": float(np.sqrt(np.mean(base ** 2))),
            "base_max_mm": float(base.max())}


def print_report(rec: dict, an: dict, log=print) -> None:
    log("")
    log("=" * 74)
    log("KINOVA GEN3 <-> MOCAP CALIBRATION   %s" % rec["captured_wall"])
    log("=" * 74)
    log("samples used %d, skipped %d, mocap body 1008, %d frames seen"
        % (an["n_samples"], an["n_skipped"], rec["mocap"].get("frames_seen", 0)))
    tc = rec.get("tool_configuration", {})
    if tc.get("available"):
        log("arm tool transform (m, deg): %s   payload %.3f kg"
            % (" ".join("%.4f" % v for v in tc["tool_transform_m_deg"]),
               tc["tool_mass_kg"]))
    log("")
    log("-- observability " + "-" * 56)
    o = an["observability"]
    log("  relative rotation up to %.1f deg, axis spread %.1f deg, tool cloud "
        "radius %.0f mm" % (o["max_relative_rot_deg"],
                            o["rotation_axis_spread_deg"],
                            o["tool_radius_m"] * 1e3))
    for note in o["notes"]:
        log("  ! " + note)
    if not o["notes"]:
        log("  both transforms are fully determined by this campaign")
    log("")
    log("-- fitted transforms " + "-" * 52)
    log("  Y = T_tool_rb   translation %s mm  (|lever| %.2f mm)"
        % (" ".join("%8.3f" % v for v in an["Y_translation_mm"]),
           an["Y_lever_arm_mm"]))
    log("                  rotation    %s deg (extrinsic xyz)"
        % " ".join("%8.3f" % v for v in an["Y_rotation_rpy_deg"]))
    log("  X = T_world_base translation %s m"
        % " ".join("%8.4f" % v for v in an["X_translation_m"]))
    log("                  rotation    %s deg, yaw %.3f deg"
        % (" ".join("%8.3f" % v for v in an["X_rotation_rpy_deg"]),
           an["X_yaw_deg"]))
    log("")
    log("-- how well the model explains the data " + "-" * 33)
    r = an["residual"]
    log("  position   RMS %.2f mm   max %.2f mm" % (r["pos_rms_mm"],
                                                    r["pos_max_mm"]))
    log("  orientation RMS %.3f deg  max %.3f deg" % (r["ang_rms_deg"],
                                                      r["ang_max_deg"]))
    lo = an["leave_one_out"]
    log("  leave-one-out prediction  RMS %.2f mm   max %.2f mm"
        % (lo["pos_rms_mm"], lo["pos_max_mm"]))
    log("  mocap's own jitter, worst pose: %.3f mm / %.3f deg peak-to-peak"
        % (an["mocap_noise"]["pos_ptp_mm_max"],
           an["mocap_noise"]["ang_ptp_deg_max"]))
    log("")
    log("-- does +x mean +x " + "-" * 54)
    for name, key in (("arm-reported", "axis_fidelity_arm_reported"),
                      ("commanded   ", "axis_fidelity_commanded")):
        f = an[key]
        if not f.get("pairs"):
            log("  %s: %s" % (name, f.get("note")))
            continue
        log("  %s: %d translation pairs, scale %.6f +- %.6f (%+.0f ppm), "
            "direction %.3f deg mean / %.3f deg worst, residual %.2f mm RMS"
            % (name, f["n_pairs"], f["scale_mean"], f["scale_sd"],
               f["scale_ppm_from_unity"], f["angle_mean_deg"],
               f["angle_max_deg"], f["error_rms_mm"]))
    ax = an["axis_fidelity_by_axis"]
    for name in "xyz":
        a = ax[name]
        if not a["n_pairs"]:
            continue
        log("    along base %s: %2d pairs, %+.0f ppm, direction %.3f deg"
            % (name, a["n_pairs"], a["scale_ppm_from_unity"],
               a["angle_mean_deg"]))
    log("    the three axes agree to %.0f ppm against a %.0f ppm threshold "
        "-> %s.  A COMMON scale is one ruler against the other; three genuinely "
        "different ones would be a distortion"
        % (ax["spread_ppm"], ax["significance_threshold_ppm"],
           "SIGNIFICANT" if ax["spread_is_significant"] else "not significant"))
    sf = an["scale_fit"]
    log("  fitting a scale as a 13th parameter gives %.6f (%+.0f ppm) and "
        "moves the residual %.2f -> %.2f mm RMS"
        % (sf["scale"], sf["scale_ppm_from_unity"],
           sf["pos_rms_mm_without_scale"], sf["pos_rms_mm"]))
    t = an["arm_tracking"]
    log("  the arm reached what it was told to %.2f mm RMS / %.3f deg RMS"
        % (t["pos_rms_mm"], t["ang_rms_deg"]))
    log("")
    log("-- do we need a marker body on the BASE " + "-" * 33)
    b = an["base_recovery"]
    log("  base solved from ONE frame + FK wanders %.2f mm RMS (%.2f mm worst)"
        % (b["pos_rms_mm"], b["pos_max_mm"]))
    log("  ... and %.2f mm RMS (%.2f worst) when the frame is one the fit never "
        "saw, which is the number to quote"
        % (lo["base_rms_mm"], lo["base_max_mm"]))
    log("  and %.3f deg RMS (%.3f deg worst) about the jointly fitted base"
        % (b["ang_rms_deg"], b["ang_max_deg"]))
    log("  per-axis sd of the recovered base origin: %s mm"
        % " ".join("%.2f" % v for v in b["centre_sd_mm"]))
    log("")
    log("-- per pose " + "-" * 61)
    for row in an["per_sample"]:
        log("  %-6s  %6.2f mm  %6.3f deg" % (row["label"], row["pos_mm"],
                                             row["ang_deg"]))
    for s in rec.get("skipped", []):
        log("  %-6s  SKIPPED: %s" % (s["label"], s["why"]))
    log("=" * 74)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--server-ip", default=None, help="Motive host")
    ap.add_argument("--client-ip", default=None, help="this PC's camera-net NIC")
    ap.add_argument("--radius", type=float, default=PLAN_R_M,
                    help="plan radius, metres (envelope allows 0.100)")
    ap.add_argument("--speed", type=float, default=0.030, help="m/s")
    ap.add_argument("--settle", type=float, default=SETTLE_S)
    ap.add_argument("--capture", type=float, default=CAPTURE_S)
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; touch nothing")
    ap.add_argument("--no-home", action="store_true",
                    help="use the current pose as the centre instead of Home")
    ap.add_argument("--analyze-only", default=None,
                    help="re-analyse a saved campaign.json")
    ap.add_argument("--yes", action="store_true",
                    help="required to move the arm")
    args = ap.parse_args(argv)

    if args.analyze_only:
        with open(args.analyze_only, encoding="utf-8") as fh:
            rec = json.load(fh)
        an = analyse(rec)
        print_report(rec, an)
        out = os.path.join(os.path.dirname(args.analyze_only), "analysis.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(an, fh, indent=1)
        print("wrote " + out)
        return 0

    if args.radius > DEFAULT_ENVELOPE_R_M:
        print("--radius %.4f m is outside the %.4f m envelope; refused before "
              "anything was contacted." % (args.radius, DEFAULT_ENVELOPE_R_M))
        return 2

    if args.dry_run:
        centre = [0.4615, 0.0159, 0.4337, 90.16, 0.27, 90.65]
        print("DRY RUN — nothing is contacted.  Centre assumed at Home:")
        print("  " + " ".join("%8.4f" % v for v in centre))
        poses = plan_poses(centre, build_plan(args.radius))
        for i, (label, pose) in enumerate(poses):
            d = float(np.linalg.norm(np.array(pose[0:3])
                                     - np.array(centre[0:3])))
            _, ang = pose_delta(centre, pose)
            print("  [%2d] %-6s %s   %5.1f mm from centre, %5.1f deg turned"
                  % (i + 1, label, " ".join("%8.4f" % v for v in pose),
                     d * 1e3, ang))
        # The dry run used to print the plan and say nothing about whether it
        # FITS, which is the one question it exists to answer before the arm is
        # touched.
        bad = envelope_violations(centre, [p for _, p in poses])
        if bad:
            print("  REFUSED: %d pose(s) leave the %.0f mm envelope about this "
                  "centre:" % (len(bad), DEFAULT_ENVELOPE_R_M * 1e3))
            for i, d in bad:
                print("    [%2d] %-6s %.4f m" % (i + 1, poses[i][0], d))
            return 2
        print("  all %d poses fit a %.0f mm ball about this centre"
              % (len(poses), DEFAULT_ENVELOPE_R_M * 1e3))
        return 0

    if not args.yes:
        print("this MOVES the arm.  Re-run with --yes once the workspace is "
              "clear, or --dry-run to see the plan.")
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out or os.path.join(RESULTS_DIR, "mocap_calib_" + stamp)
    os.makedirs(outdir, exist_ok=True)

    rec = run_campaign(ip=args.ip, server_ip=args.server_ip,
                       client_ip=args.client_ip, r_m=args.radius,
                       speed_ms=args.speed, settle_s=args.settle,
                       capture_s=args.capture, go_home=not args.no_home)
    path = os.path.join(outdir, "campaign.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote " + path)

    if len(rec["samples"]) < 3:
        print("only %d usable poses — nothing to fit" % len(rec["samples"]))
        return 2
    an = analyse(rec)
    print_report(rec, an)
    with open(os.path.join(outdir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(an, fh, indent=1)
    print("wrote " + os.path.join(outdir, "analysis.json"))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
