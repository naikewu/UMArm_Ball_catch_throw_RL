"""Reduce a drive campaign: actuator/axis map, plate azimuths, FK-vs-mocap.

Reads a ``canarm_drive_campaign.py`` record and answers the three questions the
campaign was run to answer, in an order where each one's evidence is visible
before the next one uses it:

1. **Is the frame inference sound?**  Lock statistics, per-frame registration
   residual, and the co-rigidity check — plates 1/2 and 3/4 are bolted to one
   connector, so their inferred body frames must agree, and nothing about the
   agreement is fitted.  This section can fail before any conclusion is drawn.
2. **Which joint does each board drive, and which way?**  The joint whose
   marker-derived angle moves most when that board alone is pressurised, with
   the margin over the runner-up printed so a marginal call is visible as one.
3. **Where do the marker brackets sit, and does fkine then match mocap?**  The
   per-plate azimuth is measured from the *mechanism* — the rotation axis the
   actuators actually excite — and then the two intra-segment relative
   azimuths and the five link lengths are refined against the u-joint centres
   fkine predicts.  Both numbers are reported, and so is their disagreement,
   because a calibration that only reports its own residual cannot be checked.

The azimuth calibration is what the whole thing turns on: this arm's marker
arms lie **along** the revolute axes rather than 45 deg from them, and 45 deg
of frame error yields a ``q`` that is smooth, repeatable and wrong.

USAGE::

    python hw_tests/canarm_axis_analysis.py                       # newest record
    python hw_tests/canarm_axis_analysis.py --source <file.json>
    python hw_tests/canarm_axis_analysis.py --holdout poses       # fit on singles
    python hw_tests/canarm_axis_analysis.py --mint-locks          # write locks.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(_HERE)
if _WS not in sys.path:
    sys.path.insert(0, _WS)

from UMArm_KINEMATICS import canarm_actuators as ca  # noqa: E402
from UMArm_KINEMATICS import robot_params as rp  # noqa: E402
from UMArm_KINEMATICS.fkine import ujoint_centres  # noqa: E402
from UMArm_MOCAP import canarm_frames as cf  # noqa: E402
from UMArm_MOCAP import marker_frame as mf  # noqa: E402

D2R = math.pi / 180.0

#: Proximal-pair composition order used throughout this reduction.  Module
#: level rather than threaded through every helper because the reader and the
#: forward model must never disagree about it inside one run; ``--order``
#: rebinds it once, before anything is computed.
ORDER = cf.PROXIMAL_ORDER

#: The CAN arm's geometry as the research tree records it
#: (``UMARM_Variable_Stiffness_Oct2025/robot_constants.py:20-45``,
#: ``param_link0/1/2``).  Used as the starting table: ``UC``, ``AA`` and the
#: travel/offset columns are CAD numbers this campaign has no way to improve,
#: while ``LL`` and ``JD`` are what the measured chain gaps replace.
LEGACY_CANARM_TABLE = np.array([
    # JA1    JA2    UC1  UC2  AA1        AA2        AO1    AO2    LL        JD
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.177625, 0.0],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.146715, 0.07326],
    [0.047, 0.047, 0.0, 0.0, 0.0437125, 0.0437125, 0.028, 0.028, 0.144235, 0.07249],
], dtype=float)


# --------------------------------------------------------------------------
# Small numerics
# --------------------------------------------------------------------------


def _rz(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_axis_angle(r):
    """Unit axis and angle of a rotation matrix, radians."""
    ang = math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = float(np.linalg.norm(v))
    return (v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])), ang


def circular_offset(angles_deg, spacing, expect):
    """Least-squares ``o`` with every ``a + o == expect`` modulo ``spacing``.

    The four axes of one universal joint are 90 deg apart and their labelling is
    arbitrary, so the quantity to fit is a single offset against a modular
    lattice rather than four independent angles.  Averaging on the unit circle
    of the wrapped angle does that without ever choosing which axis is which.
    """
    a = np.asarray(angles_deg, dtype=float)
    k = 360.0 / spacing
    o = -math.degrees(np.angle(np.exp(1j * (np.radians(a) - math.radians(expect)) * k).mean())) / k
    resid = []
    for v in a:
        e = (v + o - expect) % spacing
        resid.append(e if e < spacing / 2 else e - spacing)
    return float(o), np.asarray(resid)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def newest_record() -> str:
    hits = sorted(glob.glob(os.path.join(_HERE, "results", "drive_*.json")))
    if not hits:
        raise FileNotFoundError(
            "no hw_tests/results/drive_*.json; run hw_tests/canarm_drive_campaign.py")
    return hits[-1]


def step_markers(step) -> dict | None:
    out = {}
    for p in range(cf.N_PLATES):
        v = step["plates"][str(p)]
        if v is None:
            return None
        out[p] = np.asarray(v["mean_m"], dtype=float)
    return out


def load(source: str):
    with open(source, encoding="utf-8") as fh:
        rec = json.load(fh)
    steps, dropped = [], 0
    for s in rec["steps"]:
        m = step_markers(s)
        if m is None:
            dropped += 1
            continue
        s["_markers"] = m
        steps.append(s)
    rec["steps"] = steps
    rec["_dropped"] = dropped
    return rec


# --------------------------------------------------------------------------
# Section 1: is the frame inference sound?
# --------------------------------------------------------------------------


def build_frames(steps, locks):
    """``(n, 6, 4, 4)`` inferred bracket frames plus the per-step worst RMS."""
    poses, worst = [], []
    for s in steps:
        f = np.tile(np.eye(4), (cf.N_PLATES, 1, 1))
        w = 0.0
        for p in range(cf.N_PLATES):
            t, qual = mf.infer_plate_frame(s["_markers"][p], None, locks[p])
            if t is None:
                raise RuntimeError(
                    f"{s['label']}: plate {p} would not solve ({qual.reason}, "
                    f"rms {qual.rms_residual_m * 1000:.2f} mm)")
            f[p] = t
            w = max(w, float(qual.rms_residual_m))
        poses.append(f)
        worst.append(w)
    return np.array(poses), np.array(worst)


def report_frames(rec, locks, poses, worst, out):
    print("== frame inference ==")
    print(f"  {len(poses)} usable steps"
          + (f" ({rec['_dropped']} dropped: a plate was never fully tracked)"
             if rec["_dropped"] else ""))
    for p in range(cf.N_PLATES):
        L = locks[p]
        print(f"  plate {p}: diagonals {L.diag_lengths_m[0]*1000:6.1f}/"
              f"{L.diag_lengths_m[1]*1000:6.1f} mm, arms "
              f"{min(L.arm_radii_m)*1000:.1f}-{max(L.arm_radii_m)*1000:.1f} mm, "
              f"midpoint sep {L.midpoint_separation_m*1000:4.2f} mm, "
              f"crossing off-90 {90.0-L.diagonal_crossing_deg:+.2f} deg, "
              f"out-of-plane rms {L.out_of_plane_rms_m*1000:.2f} mm")
    print(f"  template registration residual over all steps and plates: "
          f"max {worst.max()*1000:.3f} mm (gate "
          f"{mf.TEMPLATE_RMS_TOL_M*1000:.1f} mm)")
    out["frames"] = {
        "steps": len(poses),
        "dropped": rec["_dropped"],
        "max_template_rms_m": float(worst.max()),
        "locks": {str(p): locks[p].to_dict() for p in range(cf.N_PLATES)},
    }


def report_corigid(poses, azimuth_deg, out, tag):
    rows = np.array([cf.co_rigid_residual_deg(f, azimuth_deg) for f in poses])
    print(f"  co-rigidity ({tag} azimuths) -- these plates share one connector, "
          f"so both columns must be zero:")
    for i, (a, b) in enumerate(cf.CO_RIGID_PAIRS):
        az, tilt = rows[:, i, 0], rows[:, i, 1]
        print(f"    plates {a}/{b}: body-frame azimuth {az.mean():+7.3f} "
              f"+- {az.std():.3f} deg,  out-of-plane {tilt.max():.3f} deg worst")
    out.setdefault("co_rigid", {})[tag] = [
        {"pair": list(pair),
         "azimuth_mean_deg": float(rows[:, i, 0].mean()),
         "azimuth_sd_deg": float(rows[:, i, 0].std()),
         "out_of_plane_max_deg": float(rows[:, i, 1].max())}
        for i, pair in enumerate(cf.CO_RIGID_PAIRS)]
    return rows


def bracket_corigid_offsets(poses) -> list:
    """``phi_{2i+2} - phi_{2i+1}``, straight off the *bracket* frames.

    Independent of any azimuth: the two brackets of one connector differ by a
    fixed rotation about their shared normal, and measuring it is what lets the
    distal plate's azimuth be pinned to its co-rigid neighbour's rather than to
    a noisier drive measurement.
    """
    out = []
    for a, b in cf.CO_RIGID_PAIRS:
        vals = [math.degrees(math.atan2(*(f[a][0:3, 0:3].T @ f[b][0:3, 0:3])[[1, 0], 0]))
                for f in poses]
        out.append(float(np.mean(vals)))
    return out


# --------------------------------------------------------------------------
# Section 2 and 3: the drive analysis
# --------------------------------------------------------------------------


def link_dir(f, seg, phi_p_deg=0.0):
    """Unit link direction of one segment, in its proximal plate's frame."""
    p, dd = 2 * seg, 2 * seg + 1
    r = f[p][0:3, 0:3] @ _rz(-phi_p_deg * D2R)
    v = r.T @ (f[p][0:3, 3] - f[dd][0:3, 3])
    return v / np.linalg.norm(v)


def swing_free_rel(f, seg, phi_p_deg, order=ORDER):
    """``R3(t3) R4(t4) Rz(phi_D)`` -- the distal rotation, swing removed.

    The distal pair's axes are only meaningful once the proximal pair's swing
    is taken out, because pressurising a distal actuator also shifts the
    proximal joint's gravity equilibrium by several degrees on this arm.
    """
    p, dd = 2 * seg, 2 * seg + 1
    r = f[p][0:3, 0:3] @ _rz(-phi_p_deg * D2R)
    v = r.T @ (f[p][0:3, 3] - f[dd][0:3, 3])
    v /= np.linalg.norm(v)
    t1, t2 = mf.swing_angles(v, order)
    prox = (_rx(t1) @ _ry(t2)) if order == "xy" else (_ry(t2) @ _rx(t1))
    return prox.T @ r.T @ f[dd][0:3, 0:3]


def _neighbour_baseline(idx_list, k):
    """Indices of the steps bracketing a drive step, for the local baseline.

    Local rather than global because the arm creeps: a McKibben that has just
    been vented is a little longer than one that has been idle for a minute,
    and differencing against the two neighbours removes the ramp that a single
    campaign-wide rest capture would leave in every delta.
    """
    return [i for i in (k - 1, k + 1) if i in idx_list]


def analyse_drives(steps, poses):
    """Per single-actuator step, the tilt and rotation each segment shows."""
    singles = [i for i, s in enumerate(steps) if s.get("kind") == "single"]
    baselines = {i for i, s in enumerate(steps)
                 if s.get("kind") in ("baseline", "rest")}
    out = {}
    for k in singles:
        nb = [i for i in (k - 1, k + 1) if i in baselines]
        if not nb:
            continue
        rows = []
        for seg in range(3):
            v0 = sum(link_dir(poses[i], seg) for i in nb)
            v0 = v0 / np.linalg.norm(v0)
            c = np.cross(v0, link_dir(poses[k], seg))
            rows.append((math.degrees(math.asin(min(1.0, float(np.linalg.norm(c))))),
                         math.degrees(math.atan2(c[1], c[0]))))
        out[int(steps[k]["base"])] = {"step": k, "baselines": nb, "tilt": rows}
    return out


def distal_axes(steps, poses, drives, seg, drivers, phi_p_deg):
    res = []
    for base in sorted(drivers):
        d = drives.get(base)
        if d is None:
            continue
        g = sum(swing_free_rel(poses[i], seg, phi_p_deg) for i in d["baselines"])
        u, _s, vt = np.linalg.svd(g)
        g = u @ vt
        ax, ang = rot_axis_angle(g.T @ swing_free_rel(poses[d["step"]], seg, phi_p_deg))
        res.append((base, math.degrees(ang),
                    math.degrees(math.atan2(ax[1], ax[0])) % 180.0))
    return res


def calibrate_azimuths(steps, poses, drives, out):
    """The mechanism-measured per-plate azimuth, plus the evidence for it."""
    phi = np.zeros(cf.N_PLATES)
    prox_drivers, dist_drivers = {}, {}
    print("\n== plate azimuth, measured from the mechanism ==")
    print("  proximal plates: the rotation axis of the intra-segment link tilt.")
    ev_prox = {}
    for seg in range(3):
        block = set(ca.SEGMENT_BLOCKS[seg])
        cand = sorted(((drives[b]["tilt"][seg][0], drives[b]["tilt"][seg][1], b)
                       for b in drives if b in block), reverse=True)
        prox_drivers[seg] = {b for _, _, b in cand[:4]}
        dist_drivers[seg] = block - prox_drivers[seg]
        o, resid = circular_offset([a for _, a, _ in cand[:4]], 90.0, 0.0)
        phi[2 * seg] = o
        print(f"    plate {2*seg}: " + "  ".join(
            f"0x{b:03X} {m:5.2f} deg @ {a:+7.2f}" for m, a, b in cand[:4]))
        print(f"             phi {o:+7.2f} deg; the four axes are mutually "
              f"orthogonal to {np.abs(resid).max():.2f} deg; the other four "
              f"boards of this block tilt this link by at most "
              f"{max(m for m, _, _ in cand[4:]):.2f} deg")
        ev_prox[seg] = {"drivers": {f"0x{b:03X}": {"tilt_deg": m, "azimuth_deg": a}
                                    for m, a, b in cand[:4]},
                        "phi_deg": o,
                        "orthogonality_resid_deg": float(np.abs(resid).max()),
                        "non_driver_max_tilt_deg": float(max(m for m, _, _ in cand[4:]))}

    print("  distal plates 1 and 3: co-rigid with the next proximal plate.")
    co = bracket_corigid_offsets(poses)
    for i, (a, b) in enumerate(cf.CO_RIGID_PAIRS):
        phi[a] = phi[b] - co[i]
        print(f"    plate {a}: bracket offset to plate {b} is {co[i]:+.3f} deg "
              f"-> phi {phi[a]:+7.2f} deg")

    print("  plate 5: no co-rigid partner, so its own distal rotation axes.")
    ax5 = distal_axes(steps, poses, drives, 2, dist_drivers[2], phi[4])
    o, resid = circular_offset([a for _, _, a in ax5], 90.0, 45.0)
    phi[5] = o
    print("    " + "  ".join(f"0x{b:03X} {m:5.2f} deg @ {a:6.2f}" for b, m, a in ax5))
    print(f"             phi {o:+7.2f} deg, worst residual {np.abs(resid).max():.2f} deg")

    print("  cross-check: the same measurement on plates 1 and 3, which "
          "co-rigidity already pinned --")
    cross = {}
    for seg in (0, 1):
        axd = distal_axes(steps, poses, drives, seg, dist_drivers[seg], phi[2 * seg])
        o2, r2 = circular_offset([a for _, _, a in axd], 90.0, 45.0)
        diff = (o2 - phi[2 * seg + 1] + 45.0) % 90.0 - 45.0
        print(f"    plate {2*seg+1}: distal axes give {o2:+7.2f} deg "
              f"(residual {np.abs(r2).max():.2f}) against co-rigidity's "
              f"{phi[2*seg+1]:+7.2f} -- they differ by {diff:+.2f} deg")
        cross[2 * seg + 1] = {"distal_axis_phi_deg": o2, "corigid_phi_deg": float(phi[2*seg+1]),
                              "difference_deg": float(diff)}

    # Resolve the 90 deg branch: the body x nearest the volume's +x at rest.
    for p in range(cf.N_PLATES):
        best, bd = phi[p], -2.0
        for k in range(4):
            cand_phi = phi[p] + 90.0 * k
            x = (poses[0][p][0:3, 0:3] @ _rz(-cand_phi * D2R))[:, 0]
            if x[0] > bd:
                bd, best = x[0], cand_phi
        phi[p] = (best + 180.0) % 360.0 - 180.0
    print(f"  branch resolved by the gauge -- {cf.AZIMUTH_GAUGE}:")
    print("    phi (deg) " + " ".join(f"{v:+8.3f}" for v in phi))
    print("    body x world azimuth (deg) " + " ".join(
        f"{math.degrees(math.atan2(*(poses[0][p][0:3, 0:3] @ _rz(-phi[p]*D2R))[[1, 0], 0])):+7.2f}"
        for p in range(cf.N_PLATES)))
    out["azimuth_axis_deg"] = phi.tolist()
    out["azimuth_evidence"] = {"proximal": ev_prox, "corigid_bracket_offsets_deg": co,
                               "plate5_distal_axes": [[f"0x{b:03X}", m, a] for b, m, a in ax5],
                               "cross_check": cross}
    return phi, prox_drivers, dist_drivers


def actuator_map(steps, poses, drives, phi, out):
    print("\n== actuator / axis map ==")
    qs = {}
    rows = []
    for base in sorted(drives):
        d = drives[base]
        qd = cf.q_from_plate_frames(poses[d["step"]], phi, ORDER)
        qb = np.mean([cf.q_from_plate_frames(poses[i], phi, ORDER)
                      for i in d["baselines"]], axis=0)
        dq = qd - qb
        qs[base] = dq
        order = np.argsort(-np.abs(dq))
        rows.append((base, int(order[0]), float(np.degrees(dq[order[0]])),
                     int(order[1]), float(np.degrees(dq[order[1]])),
                     float(abs(dq[order[0]]) / max(abs(dq[order[1]]), 1e-12))))
    print(f"  {'board':6} {'joint':6} {'name':10} {'dq (deg)':>9} "
          f"{'runner-up':>12} {'margin':>7}")
    for base, j, v, j2, v2, ratio in rows:
        print(f"  0x{base:03X}  j{j:<5d} {ca.JOINT_NAMES[j]:10} {v:+9.2f} "
              f"  j{j2:<2d} {v2:+6.2f} {ratio:7.2f}")

    pairs = {}
    for base, j, v, _j2, _v2, _r in rows:
        pairs.setdefault(j, {})[base] = v
    measured = []
    complaints = []
    for j in range(12):
        got = pairs.get(j, {})
        pos = [b for b, v in got.items() if v > 0]
        neg = [b for b, v in got.items() if v < 0]
        if len(pos) == 1 and len(neg) == 1:
            measured.append((pos[0], neg[0]))
        else:
            measured.append(None)
            complaints.append(f"joint {j} ({ca.JOINT_NAMES[j]}): "
                              f"{len(pos)} positive, {len(neg)} negative drivers")
    print("\n  antagonistic pairs, joint -> (positive, negative):")
    legacy = ca.LEGACY_JOINT_PAIRS
    for j, pair in enumerate(measured):
        if pair is None:
            print(f"    j{j:<2d} {ca.JOINT_NAMES[j]:10} NOT RESOLVED")
            continue
        same = tuple(pair) == tuple(legacy[j])
        same_set = set(pair) == set(legacy[j])
        verdict = ("matches the legacy table" if same else
                   "same pair, opposite sign" if same_set else
                   f"legacy said 0x{legacy[j][0]:03X}/0x{legacy[j][1]:03X}")
        print(f"    j{j:<2d} {ca.JOINT_NAMES[j]:10} "
              f"+0x{pair[0]:03X} / -0x{pair[1]:03X}   {verdict}")
    if complaints:
        print("  UNRESOLVED: " + "; ".join(complaints))
    out["actuator_map"] = {
        "rows": [{"base": f"0x{b:03X}", "joint": j, "name": ca.JOINT_NAMES[j],
                  "dq_deg": v, "runner_up_joint": j2, "runner_up_dq_deg": v2,
                  "margin": r} for b, j, v, j2, v2, r in rows],
        "pairs": [None if p is None else [f"0x{p[0]:03X}", f"0x{p[1]:03X}"]
                  for p in measured],
        "legacy_pairs": [[f"0x{a:03X}", f"0x{b:03X}"] for a, b in legacy],
        "unresolved": complaints,
        "dq_deg": {f"0x{b:03X}": np.degrees(v).tolist() for b, v in qs.items()},
    }
    return measured


# --------------------------------------------------------------------------
# Section 3: fkine against mocap
# --------------------------------------------------------------------------


def params_from_gaps(gaps, base=None):
    p = np.array(LEGACY_CANARM_TABLE if base is None else base, dtype=float)
    p[:, rp.COL_LL] = (np.asarray(gaps, dtype=float)[[0, 2, 4]]
                       - (p[:, rp.COL_AA1] + p[:, rp.COL_AA2]))
    p[:, rp.COL_JD] = [0.0, float(gaps[1]), float(gaps[3])]
    return p


def fk_errors(poses, phi, params, order=None):
    """``(n, 6)`` distance from each measured plate centre to fkine's."""
    order = ORDER if order is None else order
    ph = np.radians(np.asarray(phi, dtype=float))
    out = np.empty((len(poses), cf.N_PLATES))
    for k, f in enumerate(poses):
        q = mf.q_from_frames(np.concatenate([f, np.eye(4)[None]]), ph, order)
        c = ujoint_centres(q, params, order)
        r0 = f[0][0:3, 0:3] @ _rz(-float(ph[0]))
        out[k] = np.linalg.norm(c @ r0.T + f[0][0:3, 3] - f[0:cf.N_PLATES, 0:3, 3],
                                axis=1)
    return out


def rms(e):
    return float(np.sqrt((np.asarray(e) ** 2).mean()))


def refine(poses_fit, phi0, gaps0, co, verbose=True):
    """Refine the two intra-segment relative azimuths and the five lengths.

    Only those seven numbers are identifiable from centre positions, and saying
    so is half the point of doing it this way:

    * a rotation applied to **every** plate at once turns the robot frame about
      z and moves no predicted centre, so the overall azimuth is a gauge, fixed
      afterwards by the mechanism rather than by this fit;
    * the co-rigid offsets ``phi_2 - phi_1`` and ``phi_4 - phi_3`` are rigid-body
      measurements, held fixed here rather than re-fitted;
    * ``phi_5`` moves no centre at all -- the last universal joint's two angles
      rotate the tip plate and nothing else -- so the fit cannot see it and does
      not pretend to.
    """
    def expand(free):
        p = np.zeros(cf.N_PLATES)
        p[0], p[2], p[4] = free[0:3]
        p[1] = p[2] - co[0]
        p[3] = p[4] - co[1]
        p[5] = p[4] + (phi0[5] - phi0[4])
        return p

    def cost(free, gaps):
        return rms(fk_errors(poses_fit, expand(free), params_from_gaps(gaps)))

    free = np.array([phi0[0], phi0[2], phi0[4]], dtype=float)
    gaps = np.array(gaps0, dtype=float)
    best = cost(free, gaps)
    if verbose:
        print(f"  starting from the measured azimuths and gaps: "
              f"rms {best*1000:.3f} mm")
    for _sweep in range(4):
        for k in range(3):
            for step in (1.0, 0.25, 0.05):
                while True:
                    moved = False
                    for s in (+step, -step):
                        t = free.copy()
                        t[k] += s
                        c = cost(t, gaps)
                        if c < best - 1e-9:
                            best, free, moved = c, t, True
                    if not moved:
                        break
        for k in range(5):
            for step in (3e-4, 6e-5):
                while True:
                    moved = False
                    for s in (+step, -step):
                        t = gaps.copy()
                        t[k] += s
                        c = cost(free, t)
                        if c < best - 1e-12:
                            best, gaps, moved = c, t, True
                    if not moved:
                        break
    phi = expand(free)
    # Re-gauge: a common rotation is free, so spend it on agreeing with the
    # mechanism-measured azimuths of the three proximal plates.
    shift = float(np.mean([phi0[p] - phi[p] for p in (0, 2, 4)]))
    phi = phi + shift
    return phi, gaps, best


def report_fk(poses_fit, poses_val, phi_axis, phi_fit, gaps_meas, gaps_fit, out,
              val_label):
    other = "xy" if ORDER == "yx" else "yx"
    variants = [
        ("RS485 azimuths + RS485 lengths",
         np.array(cf.FAMILY_AZIMUTH_DEG), np.array(rp.DEFAULT_PARAMS), ORDER),
        ("RS485 azimuths + legacy CAN lengths",
         np.array(cf.FAMILY_AZIMUTH_DEG), LEGACY_CANARM_TABLE, ORDER),
        ("measured azimuths + measured lengths",
         phi_axis, params_from_gaps(gaps_meas), ORDER),
        (f"refined azimuths + refined lengths, proximal order {other!r}",
         phi_fit, params_from_gaps(gaps_fit), other),
        (f"refined azimuths + refined lengths, proximal order {ORDER!r}",
         phi_fit, params_from_gaps(gaps_fit), ORDER),
    ]
    print("\n== fkine against mocap: u-joint centre error ==")
    print("  plate 0 is identically zero -- the chain is anchored there, so the "
          "signal is plates 1-5.")
    rows = []
    for name, ph, pa, od in variants:
        e = fk_errors(poses_fit, ph, pa, od)
        per = np.sqrt((e ** 2).mean(axis=0)) * 1000.0
        line = {"name": name, "fit_rms_mm": rms(e) * 1000.0,
                "fit_max_mm": float(e.max()) * 1000.0,
                "fit_per_plate_rms_mm": per.tolist()}
        print(f"  {name}")
        print(f"    per-plate rms (mm) " + " ".join(f"{v:7.3f}" for v in per)
              + f"   overall {rms(e)*1000:6.3f}  worst {e.max()*1000:6.2f}")
        if poses_val is not None and len(poses_val):
            ev = fk_errors(poses_val, ph, pa, od)
            line["val_rms_mm"] = rms(ev) * 1000.0
            line["val_max_mm"] = float(ev.max()) * 1000.0
            print(f"    held out ({val_label}, {len(poses_val)} poses): "
                  f"rms {rms(ev)*1000:6.3f} mm, worst {ev.max()*1000:6.2f} mm")
        rows.append(line)
    out["fk"] = rows
    return rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def emit_constants(phi_fit, phi_axis, gaps_fit, gaps_meas, pairs, source):
    print("\n== constants to write into the modules ==")
    print("UMArm_MOCAP/canarm_frames.py:")
    print("    PLATE_AZIMUTH_DEG = (" + ", ".join(f"{v:.3f}" for v in phi_fit) + ")")
    print(f"    PROXIMAL_ORDER = {ORDER!r}")
    print("UMArm_KINEMATICS/canarm_params.py -- u-joint centre gaps, metres:")
    print("    CANARM_PLATE_CHAIN_M = (" + ", ".join(f"{v:.6f}" for v in gaps_fit) + ")")
    p = params_from_gaps(gaps_fit)
    print("    LL = " + ", ".join(f"{v:.6f}" for v in p[:, rp.COL_LL])
          + "   JD = " + ", ".join(f"{v:.6f}" for v in p[:, rp.COL_JD]))
    print("UMArm_KINEMATICS/canarm_actuators.py:")
    print("    MEASURED_JOINT_PAIRS = (")
    for i in range(0, 12, 4):
        print("        " + " ".join(
            f"(0x{a:03X}, 0x{b:03X})," for a, b in pairs[i:i+4]))
    print("    )")
    print(f"    MEASURED_SOURCE = {os.path.relpath(source, _WS)!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None)
    ap.add_argument("--holdout", choices=("none", "poses", "singles"),
                    default="poses",
                    help="which step kind to keep out of the refinement")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--order", choices=("xy", "yx"), default=None,
                    help="proximal-pair composition order; default is "
                         "UMArm_MOCAP.canarm_frames.PROXIMAL_ORDER")
    ap.add_argument("--mint-locks", action="store_true",
                    help="write the rest-window locks to canarm_frames.DEFAULT_LOCK_PATH")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args(argv)
    if args.order:
        global ORDER
        ORDER = args.order

    source = args.source or newest_record()
    print(f"analysis: proximal composition order {ORDER!r}")
    print(f"analysis: {os.path.relpath(source, _WS)}")
    rec = load(source)
    steps = rec["steps"]
    if not steps:
        print("analysis: no step carried all six plates fully tracked")
        return 1

    rest = next((s for s in steps if s.get("kind") == "rest"), steps[0])
    locks = cf.mint_locks(rest["_markers"])
    poses, worst = build_frames(steps, locks)
    out = {"schema": "canarm_axis_analysis/1", "source": os.path.relpath(source, _WS),
           "campaign_created": rec.get("created_local")}
    report_frames(rec, locks, poses, worst, out)

    gaps_all = np.array([cf.chain_gaps_m(f) for f in poses])
    print("\n== chain: consecutive u-joint centre distances ==")
    print("  measured (mm) " + " ".join(f"{v:8.3f}" for v in gaps_all.mean(0) * 1000)
          + "   sd " + " ".join(f"{v:5.3f}" for v in gaps_all.std(0) * 1000))
    legacy_chain = np.array([
        LEGACY_CANARM_TABLE[0, rp.COL_AA1] + LEGACY_CANARM_TABLE[0, rp.COL_AA2]
        + LEGACY_CANARM_TABLE[0, rp.COL_LL],
        LEGACY_CANARM_TABLE[1, rp.COL_JD],
        LEGACY_CANARM_TABLE[1, rp.COL_AA1] + LEGACY_CANARM_TABLE[1, rp.COL_AA2]
        + LEGACY_CANARM_TABLE[1, rp.COL_LL],
        LEGACY_CANARM_TABLE[2, rp.COL_JD],
        LEGACY_CANARM_TABLE[2, rp.COL_AA1] + LEGACY_CANARM_TABLE[2, rp.COL_AA2]
        + LEGACY_CANARM_TABLE[2, rp.COL_LL]])
    print("  legacy CAN table (mm) " + " ".join(f"{v:8.3f}" for v in legacy_chain * 1000)
          + "   difference " + " ".join(
              f"{v:+6.2f}" for v in (gaps_all.mean(0) - legacy_chain) * 1000))
    out["chain"] = {"measured_m": gaps_all.mean(0).tolist(),
                    "sd_m": gaps_all.std(0).tolist(),
                    "legacy_m": legacy_chain.tolist()}

    drives = analyse_drives(steps, poses)
    if len(drives) < 24:
        print(f"\nanalysis: only {len(drives)} of 24 boards have a usable "
              f"single-actuator step")
    phi_axis, prox, dist = calibrate_azimuths(steps, poses, drives, out)
    report_corigid(poses, phi_axis, out, "measured")
    pairs = actuator_map(steps, poses, drives, phi_axis, out)

    kinds = np.array([s.get("kind", "") for s in steps])
    if args.holdout == "poses":
        fit_mask = kinds != "pose"
        val_label = "multi-joint poses"
    elif args.holdout == "singles":
        fit_mask = kinds != "single"
        val_label = "single-actuator drives"
    else:
        fit_mask = np.ones(len(steps), dtype=bool)
        val_label = ""
    poses_fit = poses[fit_mask]
    poses_val = poses[~fit_mask] if args.holdout != "none" else None
    gaps_meas = gaps_all[fit_mask].mean(0)
    co = bracket_corigid_offsets(poses)

    if args.no_refine:
        phi_fit, gaps_fit = phi_axis, gaps_meas
    else:
        print(f"\n== refinement against the predicted centres "
              f"({int(fit_mask.sum())} poses, holding out {val_label or 'nothing'}) ==")
        phi_fit, gaps_fit, best = refine(poses_fit, phi_axis, gaps_meas, co)
        print(f"  refined rms {best*1000:.3f} mm")
        print("  azimuth (deg): measured " + " ".join(f"{v:+8.3f}" for v in phi_axis))
        print("                 refined  " + " ".join(f"{v:+8.3f}" for v in phi_fit))
        print("                 they disagree by " + " ".join(
            f"{v:+6.2f}" for v in (phi_fit - phi_axis))
            + f"  (worst {np.abs(phi_fit - phi_axis).max():.2f} deg)")
        print("  gaps (mm): measured " + " ".join(f"{v:8.3f}" for v in gaps_meas * 1000))
        print("             refined  " + " ".join(f"{v:8.3f}" for v in gaps_fit * 1000))
    out["azimuth_refined_deg"] = np.asarray(phi_fit).tolist()
    out["chain_refined_m"] = np.asarray(gaps_fit).tolist()

    report_corigid(poses, phi_fit, out, "refined")
    report_fk(poses_fit, poses_val, phi_axis, phi_fit, gaps_meas, gaps_fit, out,
              val_label)
    emit_constants(phi_fit, phi_axis, gaps_fit, gaps_meas,
                   [p for p in pairs if p is not None], source)

    if args.mint_locks:
        path = cf.save_locks(locks)
        print(f"\nanalysis: wrote {os.path.relpath(path, _WS)}")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"analysis: wrote {args.json_out}")
    if args.md_out:
        write_markdown(args.md_out, out)
        print(f"analysis: wrote {args.md_out}")
    return 0


def write_markdown(path, out) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = out["actuator_map"]["rows"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# CAN arm actuator/axis map and frame calibration\n\n")
        fh.write(f"Source: `{out['source']}` ({out.get('campaign_created')})\n\n")
        fh.write("| board | joint | name | dq (deg) | runner-up | margin |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['base']} | j{r['joint']} | {r['name']} | "
                     f"{r['dq_deg']:+.2f} | j{r['runner_up_joint']} "
                     f"{r['runner_up_dq_deg']:+.2f} | {r['margin']:.2f} |\n")
        fh.write("\n| joint | measured +/- | legacy +/- |\n|---|---|---|\n")
        for j, (m, l) in enumerate(zip(out["actuator_map"]["pairs"],
                                       out["actuator_map"]["legacy_pairs"])):
            fh.write(f"| j{j} | {'/'.join(m) if m else 'unresolved'} | "
                     f"{'/'.join(l)} |\n")
        fh.write("\n## Plate azimuth (deg)\n\n")
        fh.write("| plate | measured | refined |\n|---|---|---|\n")
        for p, (a, b) in enumerate(zip(out["azimuth_axis_deg"],
                                       out["azimuth_refined_deg"])):
            fh.write(f"| {p} | {a:+.3f} | {b:+.3f} |\n")
        fh.write("\n## fkine against mocap\n\n")
        fh.write("| variant | fit rms (mm) | fit worst (mm) | held-out rms (mm) |\n")
        fh.write("|---|---|---|---|\n")
        for r in out["fk"]:
            fh.write(f"| {r['name']} | {r['fit_rms_mm']:.3f} | "
                     f"{r['fit_max_mm']:.2f} | "
                     f"{r.get('val_rms_mm', float('nan')):.3f} |\n")


if __name__ == "__main__":
    raise SystemExit(main())
