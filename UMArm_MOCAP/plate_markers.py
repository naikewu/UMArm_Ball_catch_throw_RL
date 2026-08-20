"""Where the UMArm's existing mocap markers sit on each plate.

The arm has carried motion-capture markers since long before the collision
bench: four per plate, on **radial arms of unequal length** about the u-joint
centre, six plates (0-5) in the volume today.  ``marker_frame`` turns them into
plate frames and ``mocap_to_q`` turns those into ``q``; what neither of them
does is hand back the marker *positions* as geometry, because neither needs to.

This module does, for one purpose: so the merged scene can DRAW them.  A
picture with the printed parts' new markers in one colour and the arm's
existing markers in another is the cheapest possible check that the model and
the cameras are talking about the same points — and the only one that catches a
mistake before a run rather than after it.

WHERE THE NUMBERS COME FROM.  Not from CAD; the arm's marker plates were never
drawn.  They are the **rest templates** of
:class:`~UMArm_MOCAP.marker_frame.PlateLock`, minted from a 30 s still capture
on 2026-08-11 and committed at :data:`DEFAULT_TEMPLATE_PATH`.  Each is that
plate's four marker offsets from the lock origin, in the locked plate frame, in
metres, in asset-label order.  A lock is exactly the thing the per-frame solve
registers against, so these ARE the points the pipeline believes in.

TWO THINGS THEY ARE NOT, and a reader who forgets either will over-read a
picture:

* **The origin is the diagonal-line intersection, not a CAD datum.**  It lands
  on the u-joint centre because the arms are radial, which is why hanging the
  markers off a simulated plate body — whose origin is the kinematic u-joint
  centre — puts them in the right place.  It is not a machined feature and
  nothing measured it to better than the fit.
* **The frame agrees with the streamed plate frame only to a few degrees.**  The
  committed locks were minted with ``x_mode="streamed"``, so their azimuth zero
  IS the streamed rest orientation; that capture's own inferred-vs-streamed body
  delta was **0.37 to 3.00 deg** per plate (``campaign_2026-08-11_recal/probe/
  probe_report.json``, ``inference/deltas/*/angle_deg``), and a live probe on
  2026-08-19 read 0.66 to 3.67 deg.  At a 100 mm arm radius three degrees is
  5 mm, so a drawn marker landing a few millimetres off a reported one is the
  frame convention, not a fault.

:func:`verify_against_live` re-measures the templates from a running receiver
and reports the disagreement, which is the honest way to find out whether the
committed capture still describes the arm.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: The committed locks.  Written by ``mocap_probe.py --lock-out`` during the
#: 2026-08-11 recalibration campaign and kept somewhere *tracked* rather than
#: under an untracked ``logs/``, precisely so that something like this module
#: can depend on it.
#:
#: PORT NOTE (this workspace).  In ``RS485_VEMA`` this pointed at
#: ``UMArm_ROBOT_CONTROL/results/campaign_2026-08-11_recal/locks.json``.  That
#: campaign directory is 1.2 GB and is not carried here; the 15 kB file itself
#: is, byte for byte, as ``templates/rs485_locks_example.json`` (SHA-256
#: dbb1fd8f7adefa5f92025daaa828eaffea7d6e87cbddb6bcef5f063fdddc942b).  It
#: describes **the RS485 arm's** plates and that Motive asset roster, so it is
#: the default here only because the RS485 arm is the one whose locks have been
#: minted.  The CAN arm's locks live at
#: :data:`UMArm_MOCAP.canarm_mocap.DEFAULT_TEMPLATE_PATH` and do not exist yet;
#: :func:`load_templates` raises rather than falling back, which is what stops
#: an unminted CAN arm from being drawn with the RS485 arm's marker geometry.
DEFAULT_TEMPLATE_PATH = os.path.join(
    _HERE, "templates", "rs485_locks_example.json")

#: Plates the volume actually carries.  Plate 6 (the end-effector plate) is
#: absent from the Motive project — the probe reports it as "ABSENT from the
#: stream all window -- expected (D6)" — so nothing here invents one.
PLATES = (0, 1, 2, 3, 4, 5)


def load_templates(path: str | None = None) -> dict:
    """``{plate: (4, 3) array in metres}`` from a committed locks file.

    Raises rather than falling back.  A missing or malformed template file
    would otherwise mean an arm drawn with no markers at all, which looks
    exactly like an arm whose markers are correctly at the origin.
    """
    path = path or DEFAULT_TEMPLATE_PATH
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    plates = doc.get("plates", doc)
    out = {}
    for key, lock in plates.items():
        if not isinstance(lock, dict) or "template_m" not in lock:
            continue
        t = np.asarray(lock["template_m"], dtype=float)
        if t.shape != (4, 3) or not np.isfinite(t).all():
            raise ValueError(
                "%s: plate %s has a template of shape %s, expected (4, 3) of "
                "finite numbers" % (path, key, t.shape))
        out[int(key)] = t
    if not out:
        raise ValueError("%s carries no plate templates" % path)
    return out


def template_meta(path: str | None = None) -> dict:
    """Provenance of a locks file: when it was captured, and how well.

    Carried into the scene's own record so that a figure can say which capture
    it was drawn from without anyone having to remember.
    """
    path = path or DEFAULT_TEMPLATE_PATH
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    meta = dict(doc.get("meta", {}))
    plates = doc.get("plates", {})
    meta["path"] = path
    meta["plates"] = sorted(int(k) for k in plates)
    meta["x_mode"] = {int(k): v.get("x_mode") for k, v in plates.items()}
    meta["x_residual_deg"] = {int(k): float(v.get("x_residual_deg", float("nan")))
                              for k, v in plates.items()}
    meta["arm_radii_mm"] = {
        int(k): [round(float(r) * 1e3, 2) for r in v.get("arm_radii_m", ())]
        for k, v in plates.items()}
    return meta


def plate_marker_points_m(plate: int, templates: dict | None = None) -> np.ndarray:
    """One plate's four marker offsets, metres, in that plate's own frame."""
    templates = load_templates() if templates is None else templates
    if plate not in templates:
        raise KeyError("no template for plate %d; the file carries %s"
                       % (plate, sorted(templates)))
    return np.asarray(templates[plate], dtype=float)


def arm_radii_mm(plate: int, templates: dict | None = None) -> list:
    """Each marker's distance from the plate origin, millimetres.

    The four are deliberately unequal — that is what makes a plate's pose
    resolvable without an ambiguity — so a set that comes back nearly equal is
    a sign the wrong file was loaded.
    """
    t = plate_marker_points_m(plate, templates)
    return [float(v) for v in np.linalg.norm(t, axis=1) * 1e3]


def verify_against_live(rx, seconds: float = 3.0,
                        templates: dict | None = None) -> dict:
    """Re-measure the templates from a running receiver and compare.

    *rx* is a started :class:`~UMArm_MOCAP.mocap_rx.MocapRx`.  The arm must be
    at rest for the window, for the same reason minting a lock needs it: a
    template taken from a moving plate is a shape the plate never has again.

    Compares SHAPE, not pose: each plate's live marker cloud is rigidly
    registered onto the committed template (Kabsch), and what is reported is
    the residual after that best fit plus the four inter-marker distances.  A
    frame convention cannot move those, so a disagreement here is a marker that
    has physically moved — knocked, re-glued, or relabelled — which is the only
    thing worth being told about.
    """
    import time

    from UMArm_MOCAP.marker_frame import _kabsch          # noqa: WPS437

    templates = load_templates() if templates is None else templates
    t0 = time.monotonic()
    time.sleep(float(seconds))
    window = rx.snapshot_marker_window(t0=t0)
    out = {"frames": int(len(window)), "seconds": float(seconds), "plates": {}}
    if len(window) == 0:
        out["refusal"] = "no marker frames arrived during the window"
        return out
    # Keep ONE mapping epoch.  Motive renumbers its assets when the roster
    # changes, and averaging across a change mixes two different plate
    # assignments into one "shape" that belongs to neither.  The last epoch is
    # the current one.
    epoch = int(window.mapping_epoch.max())
    keep = window.mapping_epoch == epoch
    out["mapping_epoch"] = epoch
    out["frames_in_epoch"] = int(keep.sum())
    if keep.sum() < len(window):
        out["note"] = ("the marker mapping changed mid-window; only the %d "
                       "frames of epoch %d were used"
                       % (int(keep.sum()), epoch))
    markers = [m for m, k in zip(window.markers, keep) if k]
    for plate in sorted(templates):
        rows = [m[plate] for m in markers
                if m is not None and plate in m and m[plate].shape == (4, 3)
                and np.isfinite(m[plate]).all()]
        if len(rows) < 10:
            out["plates"][plate] = {"frames": len(rows),
                                    "note": "too few complete frames"}
            continue
        live = np.mean(np.asarray(rows, dtype=float), axis=0)
        tmpl = templates[plate]
        fit = _kabsch(tmpl, live)
        if fit is None:
            out["plates"][plate] = {
                "frames": len(rows),
                "note": "the live markers are collinear or collapsed — no "
                        "rigid fit exists, so nothing can be compared"}
            continue
        R, t, _rms = fit
        res = np.linalg.norm(live - (tmpl @ R.T + t), axis=1)
        # The live radii are measured about the template ORIGIN carried through
        # the fit, not about the live centroid.  Those are different points —
        # the lock origin is the diagonal-line intersection and sits several
        # millimetres from the centroid on these plates — so measuring one
        # against the other would print two columns that disagree by that
        # offset and invite the reader to call it a moved marker.
        origin = t                                   # template origin, mapped
        out["plates"][plate] = {
            "frames": len(rows),
            "fit_residual_mm": [float(v * 1e3) for v in res],
            "fit_rms_mm": float(np.sqrt(np.mean(res ** 2)) * 1e3),
            "committed_radii_mm": [round(v, 2) for v in arm_radii_mm(plate,
                                                                     templates)],
            "live_radii_mm": [round(float(v), 2) for v in
                              np.linalg.norm(live - origin, axis=1) * 1e3],
            "edge_delta_mm": _edge_delta_mm(tmpl, live),
        }
    good = [p["fit_rms_mm"] for p in out["plates"].values()
            if "fit_rms_mm" in p]
    out["worst_fit_rms_mm"] = float(max(good)) if good else float("nan")
    return out


def _edge_delta_mm(a, b) -> list:
    """Change in each of the six inter-marker distances, millimetres.

    Pose-free by construction, which is the point: this is what a moved marker
    changes and a frame convention cannot.
    """
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            da = float(np.linalg.norm(a[i] - a[j]))
            db = float(np.linalg.norm(b[i] - b[j]))
            out.append(round((db - da) * 1e3, 3))
    return out


def main(argv=None) -> int:
    """Print the committed templates, and optionally check them live.

    ``python UMArm_MOCAP/plate_markers.py``          — print what is committed
    ``python UMArm_MOCAP/plate_markers.py --live``   — also compare against the
                                                       running mocap stream
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("--path", default=None)
    ap.add_argument("--live", action="store_true",
                    help="compare against the live stream (arm must be at rest)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--server-ip", default=None)
    ap.add_argument("--client-ip", default=None)
    args = ap.parse_args(argv)

    templates = load_templates(args.path)
    meta = template_meta(args.path)
    print("templates from %s" % meta["path"])
    print("captured %s, %.1f s, regime %s"
          % (meta.get("captured_wall"), meta.get("seconds", float("nan")),
             meta.get("regime")))
    for plate in sorted(templates):
        r = arm_radii_mm(plate, templates)
        print("  plate %d  arm radii %s mm   x_mode %s (residual %.1f deg)"
              % (plate, " ".join("%6.2f" % v for v in r),
                 meta["x_mode"].get(plate), meta["x_residual_deg"].get(plate)))
        for row in templates[plate]:
            print("        %s mm" % " ".join("%8.2f" % (v * 1e3) for v in row))
    if not args.live:
        return 0

    from UMArm_MOCAP.mocap_rx import MocapRx

    kw = {}
    if args.server_ip:
        kw["server_ip"] = args.server_ip
    if args.client_ip:
        kw["client_ip"] = args.client_ip
    rx = MocapRx(**kw)
    rx.start()
    try:
        rep = verify_against_live(rx, seconds=args.seconds,
                                  templates=templates)
    finally:
        rx.stop()
    print("")
    print("live check: %d frames over %.1f s" % (rep["frames"], rep["seconds"]))
    if "refusal" in rep:
        print("  REFUSED: " + rep["refusal"])
        return 2
    for plate in sorted(rep["plates"]):
        p = rep["plates"][plate]
        if "fit_rms_mm" not in p:
            print("  plate %d: %s" % (plate, p["note"]))
            continue
        print("  plate %d: rigid-fit residual %.2f mm RMS, per-marker %s mm"
              % (plate, p["fit_rms_mm"],
                 " ".join("%.2f" % v for v in p["fit_residual_mm"])))
        print("            inter-marker distance change %s mm"
              % " ".join("%+.2f" % v for v in p["edge_delta_mm"]))
    print("  worst plate: %.2f mm RMS" % rep["worst_fit_rms_mm"])
    return 0


__all__ = ["DEFAULT_TEMPLATE_PATH", "PLATES", "load_templates",
           "template_meta", "plate_marker_points_m", "arm_radii_mm",
           "verify_against_live"]


if __name__ == "__main__":       # pragma: no cover
    import sys

    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    raise SystemExit(main())
