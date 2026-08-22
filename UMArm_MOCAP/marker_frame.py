"""Marker radial-arm plates -> full plate frames, and plate frames -> ``q``.  Pure numpy.

Implements the frame-inference math of ``docs/marker_frame_design.md`` (§4,
rev 3, standing on the hardware model of §2 and the dropout semantics of §5).
Two phases, deliberately asymmetric:

* a rest-time **lock** per plate (:func:`compute_plate_lock` ->
  :class:`PlateLock`) that turns the plate's conventions into a
  label-addressed recipe **and captures the rigid marker template** every
  later frame is registered against.  This is the only place the streamed
  orientation or an "up" direction is ever consulted;
* a stateless **per-frame solve** (:func:`infer_plate_frame`,
  :func:`infer_all`) that rigidly registers the lock's template onto the
  observed markers (Kabsch, det(R)=+1), reading origin *and* orientation off
  the markers alone, robust to one dropped marker, gated so a mid-run label
  swap becomes an honest ``None`` instead of a silently wrong frame (review
  finding geo-4);

plus the body-chain-correct **frame-based q** (:func:`q_from_frames`, design
§4.3) the benchmark compares against the shipped pipeline.

Nothing here touches ``mocap_to_q`` (design D2): its angle convention
(:func:`mocap_to_q.ujoint_angles`) is imported and reused, never modified, so
the 12-joint verification keeps running on the proven path while this module
earns trust offline and inside the probe.

------------------------------------------------------------------------------
THE GEOMETRY, IN ONE PARAGRAPH
------------------------------------------------------------------------------
Each plate carries four markers on **radial arms of unequal length** about
the u-joint centre, the diagonals sitting 45 deg away from the bracket's
revolute axes (design §2).  Measured on the real hardware (live probe,
``probe_20260811_live2``, 3601 frames at 120 Hz, 0.02-0.15 mm marker sd):
the quadrilaterals are **not parallelograms** — per plate the two diagonal
lengths differ by 5.5-15 mm (e.g. plate 0: 205.4 vs 196.5 mm) and the
diagonal *midpoints* sit 5.5-12.8 mm apart — yet the markers are coplanar
(out-of-plane RMS <= 0.15 mm) and the diagonal **lines** cross within 0.25
deg of perpendicular.  So "the intersection of the two diagonals is the
centre" means the intersection of the diagonal *lines* (exact for radial
arms of any length), never the midpoint mean, which is biased by half the
arm-radius asymmetry.  At rest the lock fits the marker plane, sorts the
labels into a cyclic order (diagonals are the *non-adjacent* pairs), takes
the **origin from the least-squares in-plane intersection of the diagonal
lines**, orients the plane normal against an **arm-derived** up (never world
axes — the volume was re-created, world axes are exactly what may have
moved, review finding ops-4), picks the bracket x among the four
half-diagonals-rotated-+45-deg candidates using the streamed orientation as
a coarse reference (manual alignment error is a few degrees, far inside the
45 deg ambiguity), and stores the **template**: each marker's offset from
the origin expressed in the locked frame.  Per frame, the solve rigidly
registers the template's usable subset onto the observed markers — optimal
under noise for any marker geometry, exact for any 3-subset with no
parallelogram assumption anywhere — and never consults anything but the
markers, so nothing can flip mid-run however far the arm moves.

------------------------------------------------------------------------------
FAMILIES AND BODY FRAMES
------------------------------------------------------------------------------
Bracket families (design §2): plates 0/2/4 are 0-deg brackets, plates 1/3/5
are +45-deg-CCW; plate 6's family is measured at lock time, not assumed
(design D6).  The frame this module infers is the **marker bracket** frame;
every kinematic consumer converts to the **body** frame as

    R_body = R_inferred @ Rz(-phi_p)

which is what makes the co-rigid pairs {1,2}/{3,4} agree identically (the
+45 deg family difference is in the bracket, not the body —
``UMArm_KINEMATICS.plate_transforms`` gives those pairs *identity* relative
body rotation) and what :func:`q_from_frames` consumes.

------------------------------------------------------------------------------
CONTRACT
------------------------------------------------------------------------------
Positions are metres, angles radians unless a name says ``_deg``.  Marker
arrays are ``(4, 3)`` in asset-label order; flags are ``(4,)`` uint8 with
1 = tracked (labeled entry present, occluded/model-solved bits clear — the
transport already collapses the Motive param bits, ``mocap_rx``), 0 =
excluded, ``None`` = flags unknown (labeled markers absent; inference still
runs on 4 assumed markers per §5, and the *caller* owns tagging that regime).
Shape errors raise ``ValueError``; data-driven lock refusals raise
:class:`LockRefusal` (a ``ValueError`` with a named reason — the probe
catches it, prints it, and degrades to reports-only, which is the designed
behaviour for a failed preflight); a per-frame solve never raises on bad
geometry, it returns ``(None, quality-with-reason)``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

# Dual-mode import: this module is used both as a package submodule
# (``from UMArm_MOCAP.marker_frame import infer_plate_frame``) and as a flat
# module by the probe and lab scripts that put this directory on sys.path.
# Same pattern as mocap_to_q; neither style should be the only one that works.
try:  # pragma: no cover - exercised by whichever import style the caller uses
    from . import mocap_constants as mc
    from .mocap_to_q import ujoint_angles
except ImportError:  # pragma: no cover
    import mocap_constants as mc  # type: ignore[no-redef]
    from mocap_to_q import ujoint_angles  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# Constants (each from the design; none are new policy)
# --------------------------------------------------------------------------

#: Lock-time bound on the in-plane least-squares intersection residual of the
#: two diagonal *lines* (design §4.1.1).  For genuinely crossing lines the
#: residual is numerically zero; 5 mm of residual means near-parallel or
#: otherwise non-crossing "diagonals" — a mislabeled asset, not noise (rest
#: marker jitter is 0.02-0.15 mm on the 2026-08-11 probe).  NOTE: this is a
#: *line-intersection* bound; the diagonal **midpoints** of the real radial-arm
#: plates legitimately sit 5.5-12.8 mm apart (probe_20260811_live2) and are
#: recorded as a stat, never gated.
MIDPOINT_TOL_M = 0.005

#: Lock-time bound on how far off perpendicular the two diagonal lines may
#: cross (design §4.1.1).  The real plates cross within 0.25 deg of 90
#: (probe_20260811_live2 ``rectangle_stats``); more than 25 deg off means the
#: cyclic order paired an edge as a "diagonal" — a mislabeled asset.
CROSSING_TOL_DEG = 25.0

#: Per-frame registration gate (design §4.2, review finding geo-4): the RMS
#: residual of the Kabsch fit of the lock's template onto the observed
#: markers.  3 mm sits far above marker noise (0.02-0.15 mm) and far below
#: the >= 5 mm effect of any label swap or shifted labeling, so a swapped
#: frame fails here loudly instead of yielding a silently wrong frame.
TEMPLATE_RMS_TOL_M = 0.003

#: Lock refusal bound on |n_hat . u_up| (design §4.1.2, review finding geo-3):
#: below cos(60 deg) the rest plate is near-horizontal relative to the
#: arm-derived up and the normal *sign* would be decided by noise.
NORMAL_UP_MIN_COS = 0.5

#: Consume-time lock validity bound (design §4.1 "lock validity binding",
#: review finding ops-5): the lock's template registered onto the consumer's
#: own rest-frame means must fit with RMS residual below this.  3 mm = twice
#: the worst resting marker jitter observed on this Motive (1.5 mm), and well
#: below the >= cm-scale shape change a re-created rigid body produces — so
#: healthy same-session data passes and a stale lock cannot.
LOCK_STATS_TOL_M = 0.003

#: Bracket family angles phi_p (design §2): plates 0/2/4 are the 0-deg
#: family, 1/3/5 the +45-deg-CCW family.  Plate 6 is deliberately absent —
#: its family is measured at lock time, both candidates tried (design D6).
FAMILY_PHI_RAD = {0: 0.0, 1: math.pi / 4.0, 2: 0.0, 3: math.pi / 4.0,
                  4: 0.0, 5: math.pi / 4.0}

#: Below this norm a direction (projected diagonal, plane normal, x average)
#: is treated as degenerate.  Bracket geometry is ~1e-2 m, so 1e-12 m is
#: eight orders below any real chord and can only mean collapsed input.
_DIR_MIN_NORM = 1e-12

#: Registration degeneracy bound: the second singular value of the Kabsch
#: cross-covariance must carry at least this fraction of the first, else the
#: usable subset is (numerically) collinear and the rotation about that line
#: is unobservable — the reflection guard of design §4.2.  On the real plates
#: (arms at ~90 deg, radii 60-103 mm) the ratio is O(1); only genuinely
#: collinear/collapsed input falls under 1e-6.
_COLLINEAR_SV_RATIO = 1e-6

_pi = np.pi

#: +45 deg rotation about z: the basis change that maps the distal twist axes
#: (x+y)/sqrt(2), (-x+y)/sqrt(2) onto x, y in :func:`q_from_frames` (design
#: §4.3).  NOT the same object as ``mocap_constants.RZn45`` — that is the
#: **-45 deg** matrix the swing-only reader applies to *vectors*; this one
#: conjugates a residual *rotation*, and the sign is pinned by the frozen
#: goldens in ``test_marker_frame.py``.
RZ45 = np.array([
    [np.cos(_pi / 4.0), -np.sin(_pi / 4.0), 0.0],
    [np.sin(_pi / 4.0), np.cos(_pi / 4.0), 0.0],
    [0.0, 0.0, 1.0],
], dtype=float)


class LockRefusal(ValueError):
    """A lock (or lock-validity check) refused for a named, data-driven reason.

    Subclass of ``ValueError`` so callers that only know the house rule
    ("shape and value errors raise, nothing print-and-return-None") still
    catch it.  The probe's inference section wraps lock building in a broad
    ``except`` and degrades to reports-only with the message printed — for a
    preflight that *is* the designed response to a refused lock.
    """


# --------------------------------------------------------------------------
# Small numerics
# --------------------------------------------------------------------------


def _rz(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a **unit** axis (callers guarantee unit)."""
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _unit(v: np.ndarray) -> np.ndarray | None:
    """``v`` normalized, or ``None`` when degenerate.

    Uses the NaN-rejecting ``not (n >= tol)`` idiom of
    ``mocap_to_q.link_vectors``: any comparison against NaN is False, so a
    NaN norm falls into the rejection branch instead of being waved through.
    """
    n = float(np.linalg.norm(v))
    if not (n >= _DIR_MIN_NORM):
        return None
    return v / n


def _in_plane(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    """Component of ``v`` perpendicular to the unit normal ``n_hat``."""
    return v - (v @ n_hat) * n_hat


def _shoelace_normal(pts: np.ndarray) -> np.ndarray:
    """Unnormalized polygon normal over cyclically ordered points (design
    §4.1.2): ``sum_i (m_i - c) x (m_{i+1} - c)``.

    The sum is translation-invariant for a closed cycle, but the points are
    centred first anyway: marker coordinates are ~1 m while the bracket is
    ~1e-1 m, and centring keeps the cross products out of the cancellation
    regime.
    """
    c = pts.mean(axis=0)
    p = pts - c
    total = np.zeros(3)
    for i in range(p.shape[0]):
        total += np.cross(p[i], p[(i + 1) % p.shape[0]])
    return total


def _signed_angle(u: np.ndarray, v: np.ndarray, n_hat: np.ndarray) -> float:
    """Signed angle from ``u`` to ``v`` about ``n_hat`` (all in-plane, u/v unit)."""
    return math.atan2(float(n_hat @ np.cross(u, v)), float(u @ v))


def _kabsch(src: np.ndarray, dst: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Rigid registration ``dst_i ~= R @ src_i + t`` (design §4.2), or ``None``.

    Kabsch via SVD of the cross-covariance, constrained to a **proper**
    rotation (det(R) = +1) by flipping the smallest singular direction when
    the raw solution is a reflection — the standard correction, which on a
    planar point set (the plates are coplanar to 0.15 mm) picks the rotation
    branch instead of the mirror.  Returns ``None`` when the point sets are
    (numerically) collinear or collapsed: the rotation about that line is
    unobservable, so no honest frame exists (:data:`_COLLINEAR_SV_RATIO`).

    Returns ``(R, t, rms)`` with ``rms`` the root-mean-square residual of the
    fit over the given points — the per-frame geometry gate's number.
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    h = (src - src_c).T @ (dst - dst_c)
    try:
        u, s, vt = np.linalg.svd(h)
    except np.linalg.LinAlgError:  # pragma: no cover - non-finite inputs only
        return None
    # NaN-rejecting comparison: a NaN singular value must land here too.
    if not (s[1] >= max(_DIR_MIN_NORM, _COLLINEAR_SV_RATIO * float(s[0]))):
        return None
    d = float(np.linalg.det(vt.T @ u.T))
    flip = np.array([1.0, 1.0, 1.0 if d > 0.0 else -1.0])
    r = (vt.T * flip) @ u.T
    t = dst_c - r @ src_c
    res = dst - (src @ r.T + t)
    rms = float(np.sqrt(np.mean(np.sum(res * res, axis=1))))
    return r, t, rms


def _as_rest_stack(rest_stack) -> np.ndarray:
    stack = np.asarray(rest_stack, dtype=float)
    if stack.ndim != 3 or stack.shape[1:] != (4, 3) or stack.shape[0] < 1:
        raise ValueError(
            f"rest stack must have shape (n >= 1, 4, 3); got {stack.shape}")
    if not np.isfinite(stack).all():
        raise ValueError("rest stack contains non-finite marker positions")
    return stack


# --------------------------------------------------------------------------
# The lock (design §4.1)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlateLock:
    """One plate's rest-time recipe: everything the per-frame solve needs.

    All fields are plain Python scalars/tuples so ``dataclasses.asdict``
    (what the probe writes into locks.json) round-trips through JSON without
    custom encoders; :meth:`from_dict` accepts the JSON-decoded form (lists
    where tuples were).

    Recipe fields — consumed by :func:`infer_plate_frame`:

    * ``template_m`` — the rigid marker **template**, ``(4, 3)``: each
      marker's rest offset from the lock origin (the in-plane least-squares
      intersection of the two diagonal lines), expressed in the locked plate
      frame, asset-label order.  The per-frame solve registers this shape
      onto the observed markers (Kabsch), which is what makes any 3-marker
      subset exactly solvable on the real non-parallelogram plates (design
      §4.2 — no midpoint assumption anywhere).
    * ``phi_rad`` — the plate's bracket family angle; body frame consumers
      apply ``R_inferred @ Rz(-phi_rad)``.

    Definition / audit-trail fields (how the template frame was chosen):

    * ``cyclic_order`` — the four asset labels in cyclic order around the
      plate, oriented so the shoelace normal over it pointed along +u_up at
      rest.  Diagonals are the non-adjacent pairs ``(order[0], order[2])``
      and ``(order[1], order[3])``.
    * ``diag_thetas_rad`` — per *ordered* diagonal (a->c, then b->d), the
      signed in-plane angle about the normal that maps the diagonal direction
      onto the locked bracket x (theta ~ +-45 deg, measured not assumed —
      design §4.1.3).
    * ``diag_lengths_m`` — rest diagonal lengths.  Unequal on the real
      plates: they differ by 5.5-15 mm (probe_20260811_live2).

    Evidence fields — recorded for the probe report, the locks.json validity
    binding (review finding ops-5) and post-hoc debugging: the x
    disambiguation residual and reference source (§4.1.3; the probe gates at
    30 deg), the up-axis evidence (§4.1.2 — u_up as used, its dot with the
    normal, and its angle to world z), and the geometry stats (§4.1.4):
    ``midpoint_separation_m`` (5.5-12.8 mm on the real plates — the bias the
    midpoint-mean origin *would* have carried), ``intersection_residual_m``
    (the in-plane line-intersection residual, ~0 for crossing lines),
    ``arm_radii_m`` (per-marker distance from the origin), crossing angle,
    out-of-plane RMS, skew, and the rest-window capture stats
    :func:`verify_lock_against_rest` compares against.
    """

    plate: int
    phi_rad: float
    cyclic_order: tuple[int, int, int, int]
    template_m: tuple
    diag_thetas_rad: tuple[float, float]
    diag_lengths_m: tuple[float, float]
    midpoint_separation_m: float
    intersection_residual_m: float
    arm_radii_m: tuple[float, float, float, float]
    x_residual_deg: float
    x_ref_source: str
    #: "diagonal45" (user spec: x = nearest diagonal + 45 deg) or "streamed"
    #: (x = streamed bracket x, one-time rest constant; forced into existence
    #: by the 2026-08-11 finding that the base plate's marker arms sit ~16 deg
    #: off the designed azimuth — see compute_plate_lock's docstring).
    x_mode: str
    u_up: tuple[float, float, float]
    n_dot_u_up: float
    u_up_vs_world_z_deg: float
    out_of_plane_rms_m: float
    diagonal_crossing_deg: float
    skew_deg: float
    diagonals_are_longest_chords: bool
    rest_frames: int
    rest_marker_sd_m: float
    rest_markers_mean: tuple

    def to_dict(self) -> dict:
        """JSON-ready dict — exactly what the probe's ``asdict`` path emits."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlateLock":
        """Inverse of :meth:`to_dict`, tolerant of JSON's tuples-become-lists."""
        return cls(
            plate=int(d["plate"]),
            phi_rad=float(d["phi_rad"]),
            cyclic_order=tuple(int(i) for i in d["cyclic_order"]),
            template_m=tuple(tuple(float(x) for x in row)
                             for row in d["template_m"]),
            diag_thetas_rad=tuple(float(v) for v in d["diag_thetas_rad"]),
            diag_lengths_m=tuple(float(v) for v in d["diag_lengths_m"]),
            midpoint_separation_m=float(d["midpoint_separation_m"]),
            intersection_residual_m=float(d["intersection_residual_m"]),
            arm_radii_m=tuple(float(v) for v in d["arm_radii_m"]),
            x_residual_deg=float(d["x_residual_deg"]),
            x_ref_source=str(d["x_ref_source"]),
            x_mode=str(d.get("x_mode", "diagonal45")),
            u_up=tuple(float(v) for v in d["u_up"]),
            n_dot_u_up=float(d["n_dot_u_up"]),
            u_up_vs_world_z_deg=float(d["u_up_vs_world_z_deg"]),
            out_of_plane_rms_m=float(d["out_of_plane_rms_m"]),
            diagonal_crossing_deg=float(d["diagonal_crossing_deg"]),
            skew_deg=float(d["skew_deg"]),
            diagonals_are_longest_chords=bool(d["diagonals_are_longest_chords"]),
            rest_frames=int(d["rest_frames"]),
            rest_marker_sd_m=float(d["rest_marker_sd_m"]),
            rest_markers_mean=tuple(tuple(float(x) for x in row)
                                    for row in d["rest_markers_mean"]),
        )


def compute_plate_lock(rest_stack, plate: int, u_up, streamed_rot=None,
                       phi_rad: float | None = None,
                       x_mode: str = "diagonal45") -> PlateLock:
    """Build a :class:`PlateLock` from a rest window (design §4.1).

    ``rest_stack``: ``(n, 4, 3)`` marker positions, asset-label order, from
    frames with **all four markers tracked** (the caller — probe §6 — owns
    that filtering plus the >= 1 s stillness evidence).
    ``u_up``: arm-derived up, ``normalize(p_plate0 - p_plate5)`` at rest —
    never a world axis (review finding ops-4).  Normalized here.
    ``streamed_rot``: the plate's streamed ``(3, 3)`` orientation, the coarse
    x-disambiguation reference; ``None`` falls back to world x-hat and flags
    the lock (``x_ref_source``).
    ``phi_rad``: family-angle override for tests; by default plates 0-5 use
    :data:`FAMILY_PHI_RAD` and any other plate (6) has **both** candidates
    tried, keeping the better-fitting one (design D6).
    ``x_mode``: where the azimuth ZERO of the lock frame comes from.
    ``"diagonal45"`` (the user's spec): x = nearest diagonal rotated +45 deg,
    streamed orientation used only to pick among the 90-deg-spaced
    candidates.  ``"streamed"``: x = the streamed bracket x itself, projected
    in-plane; the diagonals then only define the template and the per-frame
    tracking.  The 2026-08-11 arbitration forced the second mode into
    existence: the base plate's marker arms sit ~16 deg off the designed
    45-deg azimuth (x_residual_deg records exactly that offset), so a
    diagonal-derived x misrepresents the mechanism's axes while the manual
    Motive alignment was shown mechanism-faithful (segment-1 drives:
    axis-vs-model 33/25/18 deg with diagonal frames, 2-7 deg with streamed
    frames).  The azimuth is a one-time rest constant either way — per-frame
    tracking never trusts Motive.  ``"streamed"`` requires ``streamed_rot``.

    Raises ``ValueError`` on shape/finite problems and :class:`LockRefusal`
    on data-driven refusals: diagonal lines crossing more than
    :data:`CROSSING_TOL_DEG` off perpendicular, or an in-plane
    line-intersection residual above :data:`MIDPOINT_TOL_M` (both mean a
    mislabeled asset — the real plates cross within 0.25 deg of 90,
    probe_20260811_live2), a near-horizontal rest plate (|n.u_up| < cos 60
    deg — the normal sign would be noise, review finding geo-3), or a
    degenerate/collinear marker set.

    Deliberately **not** refused here: a diagonal-midpoint separation of
    several mm — that is the measured truth of the radial-arm plates
    (5.5-12.8 mm, recorded as ``midpoint_separation_m``), and refusing on it
    is exactly the bug this revision removes.  Likewise an x residual near
    the 45 deg ambiguity boundary is *recorded* (``x_residual_deg``) and
    gated by the probe at 30 deg with exit 3 (design §4.1.3).
    """
    stack = _as_rest_stack(rest_stack)
    u = np.asarray(u_up, dtype=float)
    if u.shape != (3,):
        raise ValueError(f"u_up must have shape (3,); got {u.shape}")
    u = _unit(u)
    if u is None:
        raise ValueError("u_up is zero/non-finite; cannot orient the normal")
    if streamed_rot is not None:
        streamed_rot = np.asarray(streamed_rot, dtype=float)
        if streamed_rot.shape != (3, 3):
            raise ValueError(
                f"streamed_rot must have shape (3, 3); got {streamed_rot.shape}")
        if not np.isfinite(streamed_rot).all():
            raise ValueError("streamed_rot contains non-finite values")

    mbar = stack.mean(axis=0)                    # (4, 3) mean marker positions
    centroid = mbar.mean(axis=0)
    centred = mbar - centroid

    # 1. Cyclic order: plane fit (SVD), project, sort by angle about the
    # centroid.  The SVD normal's *sign* is arbitrary and never used — the
    # orientation step below fixes handedness against u_up, so this sort only
    # has to produce *a* consistent cycle.  Anchored at the smallest label so
    # the stored order is deterministic (goldens depend on it).
    sv = np.linalg.svd(centred, compute_uv=True)
    vt = sv[2]
    ang = np.arctan2(centred @ vt[1], centred @ vt[0])
    order = np.argsort(ang)
    order = np.roll(order, -int(np.argmin(order)))
    a, b, c, d = (int(i) for i in order)

    # Diagonals = the non-adjacent pairs of the cyclic order (design §4.1.1).
    # Origin = the least-squares intersection of the two diagonal LINES,
    # solved in-plane (basis vt[0]/vt[1] of the fitted plane) — exact for
    # radial arms of unequal length, unlike the midpoint mean, which the real
    # plates put 2.8-6.4 mm off centre (half of the measured 5.5-12.8 mm
    # midpoint separations, probe_20260811_live2).
    e1, e2 = vt[0], vt[1]
    pts2 = centred @ np.stack([e1, e2], axis=1)  # (4, 2) in-plane coordinates
    u1 = _unit(pts2[c] - pts2[a])
    u2 = _unit(pts2[d] - pts2[b])
    if u1 is None or u2 is None:
        raise LockRefusal(
            f"plate {plate}: a diagonal is degenerate after in-plane projection")
    crossing_deg = math.degrees(math.acos(min(1.0, abs(float(u1 @ u2)))))
    if not (crossing_deg >= 90.0 - CROSSING_TOL_DEG):  # NaN-rejecting
        raise LockRefusal(
            f"plate {plate}: diagonal lines cross at {crossing_deg:.1f} deg, "
            f"more than {CROSSING_TOL_DEG:.0f} deg off perpendicular "
            f"(the real plates cross within 0.25 deg of 90) -- mislabeled "
            f"asset; re-check the Motive marker set before locking")
    proj1 = np.eye(2) - np.outer(u1, u1)
    proj2 = np.eye(2) - np.outer(u2, u2)
    sol = np.linalg.lstsq(proj1 + proj2,
                          proj1 @ pts2[a] + proj2 @ pts2[b], rcond=None)[0]
    inter_resid = math.sqrt(
        (float(np.sum((proj1 @ (sol - pts2[a])) ** 2))
         + float(np.sum((proj2 @ (sol - pts2[b])) ** 2))) / 2.0)
    if not (inter_resid <= MIDPOINT_TOL_M):            # NaN-rejecting
        raise LockRefusal(
            f"plate {plate}: diagonal lines do not cross -- in-plane "
            f"intersection residual {inter_resid * 1000.0:.2f} mm "
            f"(> {MIDPOINT_TOL_M * 1000.0:.0f} mm); mislabeled asset or "
            f"degenerate geometry, re-check the Motive marker set")
    origin = centroid + sol[0] * e1 + sol[1] * e2

    # Midpoint separation: recorded as a stat (the asymmetry the origin
    # definition exists to defeat), never gated — see the docstring.
    midpoint_sep = float(np.linalg.norm(
        (mbar[a] + mbar[c]) / 2.0 - (mbar[b] + mbar[d]) / 2.0))

    # 2. Normal sign: shoelace over the cyclic order, oriented against the
    # arm-derived up.  Reversing the cyclic order exactly negates the
    # shoelace sum; the origin is direction-free (lines, not rays) so it
    # does not move.
    n_raw = _shoelace_normal(mbar[order])
    n_hat = _unit(n_raw)
    if n_hat is None:
        raise LockRefusal(
            f"plate {plate}: markers are collinear/collapsed -- no plane normal")
    dot_up = float(n_hat @ u)
    if not (abs(dot_up) >= NORMAL_UP_MIN_COS):     # NaN-rejecting comparison
        raise LockRefusal(
            f"plate {plate}: rest plate is near-horizontal "
            f"(|n.u_up| = {abs(dot_up):.3f} < cos 60 deg = {NORMAL_UP_MIN_COS}) "
            f"-- the normal sign would be noise; re-pose the arm and re-lock")
    if dot_up < 0.0:
        order = np.roll(order[::-1], -int(np.argmin(order[::-1])))
        a, b, c, d = (int(i) for i in order)
        n_hat = _unit(_shoelace_normal(mbar[order]))
        dot_up = float(n_hat @ u)

    # 3. x lock: candidates are the four half-diagonal directions, projected
    # in-plane, each rotated +45 deg about the (now up-oriented) normal.
    diag1 = mbar[c] - mbar[a]                     # ordered a -> c
    diag2 = mbar[d] - mbar[b]                     # ordered b -> d
    d1_ip = _unit(_in_plane(diag1, n_hat))
    d2_ip = _unit(_in_plane(diag2, n_hat))
    if d1_ip is None or d2_ip is None:
        raise LockRefusal(
            f"plate {plate}: a diagonal is degenerate after in-plane projection")
    r45n = _rot_about(n_hat, _pi / 4.0)
    candidates = [r45n @ v for v in (d1_ip, -d1_ip, d2_ip, -d2_ip)]

    if phi_rad is not None:
        phis = (float(phi_rad),)
    elif plate in FAMILY_PHI_RAD:
        phis = (FAMILY_PHI_RAD[plate],)
    else:
        # Plate 6: family measured, not assumed (design D6) — both tried,
        # better residual wins.
        phis = (0.0, _pi / 4.0)
    if x_mode not in ("diagonal45", "streamed"):
        raise ValueError(f"x_mode must be 'diagonal45' or 'streamed'; got {x_mode!r}")
    if x_mode == "streamed" and streamed_rot is None:
        raise ValueError("x_mode='streamed' requires streamed_rot")
    x_ref_source = "streamed" if streamed_rot is not None else "world_x_fallback"

    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for phi in phis:                # (residual_deg, candidate_x, ref_ip, phi)
        # Reference = the streamed body x rotated by the family angle about
        # the streamed body z: the bracket x the lock is trying to find,
        # to within the manual-alignment error (a few degrees << 45).
        bracket_x_local = np.array([math.cos(phi), math.sin(phi), 0.0])
        ref = (streamed_rot @ bracket_x_local if streamed_rot is not None
               else bracket_x_local)
        ref_ip = _unit(_in_plane(ref, n_hat))
        if ref_ip is None:
            raise LockRefusal(
                f"plate {plate}: x reference is perpendicular to the marker "
                f"plane -- cannot disambiguate the bracket x")
        dots = [float(cand @ ref_ip) for cand in candidates]
        k = int(np.argmax(dots))
        residual_deg = math.degrees(math.acos(min(1.0, max(-1.0, dots[k]))))
        if best is None or residual_deg < best[0]:
            best = (residual_deg, candidates[k], ref_ip, phi)
    x_residual_deg, x_cand, x_ref_ip, phi_used = best
    # In streamed mode the reference IS the x-axis and the residual records
    # the measured arm-azimuth offset from the designed 45 deg; in diagonal45
    # mode the candidate is the x-axis and the residual is the
    # disambiguation margin.  Same number, two readings — the mode says which.
    x_hat = x_ref_ip if x_mode == "streamed" else x_cand
    phi_used = float(phi_used)

    # Per-ordered-diagonal locked angles: rotate the diagonal by theta about
    # the normal and you land on the bracket x.  Definition and audit trail
    # of the template frame (§4.1.3) — the per-frame solve itself registers
    # the template, which is what makes any 3-marker subset solvable.
    theta1 = _signed_angle(d1_ip, x_hat, n_hat)
    theta2 = _signed_angle(d2_ip, x_hat, n_hat)

    # 4. Geometry stats (design §4.1.4), all measured, none assumed.
    len1 = float(np.linalg.norm(diag1))
    len2 = float(np.linalg.norm(diag2))
    chords = sorted(float(np.linalg.norm(mbar[i] - mbar[j]))
                    for i in range(4) for j in range(i + 1, 4))
    diagonals_longest = min(len1, len2) >= chords[4] - 1e-12
    corners = mbar[order]
    skew_deg = 0.0
    for i in range(4):
        s_in = _unit(corners[i] - corners[i - 1])
        s_out = _unit(corners[(i + 1) % 4] - corners[i])
        if s_in is None or s_out is None:
            raise LockRefusal(f"plate {plate}: a quadrilateral side is degenerate")
        corner_deg = math.degrees(math.acos(
            min(1.0, max(-1.0, -float(s_in @ s_out)))))
        skew_deg = max(skew_deg, abs(90.0 - corner_deg))
    # Same expression as the probe's rectangle_stats: the smallest singular
    # value is sqrt(sum of squared out-of-plane distances) over 4 markers.
    out_of_plane_rms = float(sv[1][2]) / 2.0
    arm_radii = tuple(float(np.linalg.norm(mbar[i] - origin)) for i in range(4))

    if stack.shape[0] >= 2:
        per_marker_sd = stack.std(axis=0, ddof=1)          # (4, 3)
        rest_sd = float(np.linalg.norm(per_marker_sd, axis=1).max())
    else:
        rest_sd = float("nan")

    # 5. The template: each marker's offset from the origin, expressed in the
    # locked plate frame (columns x, y = n x x, z = n).  The template is what
    # the per-frame solve registers (design §4.1.1 "Template" / §4.2); its
    # out-of-plane z components carry the true rest shape, not an idealized
    # flat one.
    r_lock = np.column_stack([x_hat, np.cross(n_hat, x_hat), n_hat])
    template = (mbar - origin) @ r_lock

    return PlateLock(
        plate=int(plate),
        phi_rad=phi_used,
        cyclic_order=(a, b, c, d),
        template_m=tuple(tuple(float(x) for x in row) for row in template),
        diag_thetas_rad=(float(theta1), float(theta2)),
        diag_lengths_m=(len1, len2),
        midpoint_separation_m=midpoint_sep,
        intersection_residual_m=float(inter_resid),
        arm_radii_m=arm_radii,
        x_residual_deg=float(x_residual_deg),
        x_ref_source=x_ref_source,
        x_mode=x_mode,
        u_up=tuple(float(v) for v in u),
        n_dot_u_up=float(dot_up),
        u_up_vs_world_z_deg=math.degrees(math.acos(
            min(1.0, max(-1.0, float(u[2]))))),
        out_of_plane_rms_m=out_of_plane_rms,
        diagonal_crossing_deg=crossing_deg,
        skew_deg=skew_deg,
        diagonals_are_longest_chords=bool(diagonals_longest),
        rest_frames=int(stack.shape[0]),
        rest_marker_sd_m=rest_sd,
        rest_markers_mean=tuple(tuple(float(x) for x in row) for row in mbar),
    )


def verify_lock_against_rest(lock: PlateLock, rest_stack,
                             tol_m: float = LOCK_STATS_TOL_M) -> None:
    """Refuse a stale lock (design §4.1 validity binding, review finding ops-5).

    Registers the lock's **template** onto the mean of the rest frames the
    *consumer* was given (rigid Kabsch, the same fit the per-frame solve
    uses) and raises :class:`LockRefusal` when the RMS residual exceeds
    ``tol_m``.  One number covers every rigid-invariant shape change: a
    re-created Motive asset with different marker order, arm lengths, or
    plane shape all show up as template misfit, while any rigid motion of
    the healthy plate fits exactly.  Locks are valid only for the Motive
    session they were captured in; any rigid-body edit requires regeneration.
    """
    stack = _as_rest_stack(rest_stack)
    mbar = stack.mean(axis=0)
    template = np.asarray(lock.template_m, dtype=float)
    fit = _kabsch(template, mbar)
    if fit is None:
        raise LockRefusal(
            f"plate {lock.plate}: lock is stale for this data -- rest markers "
            f"are degenerate/collinear, template registration impossible -- "
            f"regenerate locks.json for this Motive session")
    rms = fit[2]
    if not (rms <= tol_m):                          # NaN-rejecting comparison
        raise LockRefusal(
            f"plate {lock.plate}: lock is stale for this data -- template "
            f"registration RMS {rms * 1000.0:.2f} mm over the rest window "
            f"(tol {tol_m * 1000.0:.1f} mm) -- regenerate locks.json for "
            f"this Motive session")


# --------------------------------------------------------------------------
# The per-frame solve (design §4.2)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FrameQuality:
    """Per-frame solve diagnostics (design §4.2 "quality fields").

    ``rms_residual_m`` is the Kabsch registration's RMS residual over the
    usable markers — the one number that says how rigidly this frame's
    markers still match the lock-time template (0.02-0.15 mm of it is plain
    marker noise on this Motive; :data:`TEMPLATE_RMS_TOL_M` gates it).  NaN
    when no registration was attempted (count/usable/finite gates), and
    carried even on a gated frame when the registration itself was what
    fired — the benchmark reads the number instead of guessing.

    The design also names "whether any used marker was model-solved": under
    the labeled-flags regime that is **structurally always False** — the
    transport already collapses Motive's param bits into the single tracked
    flag, and a model-solved marker arrives as flag 0, i.e. excluded (design
    D4/§5).  ``n_flagged_out`` carries the information that survives that
    collapse: how many markers the flags excluded this frame.  When flags are
    unknown (``None``) nothing is excluded and the *caller* owns tagging the
    flags-unknown regime.

    ``reason`` is empty for a successful solve, else the named gate that
    returned ``None`` — the benchmark counts these instead of guessing.
    """

    n_usable: int = 0
    used: tuple = (False, False, False, False)
    n_flagged_out: int = 0
    rms_residual_m: float = float("nan")
    reason: str = ""


def infer_plate_frame(markers, flags, lock: PlateLock,
                      ) -> tuple[np.ndarray | None, FrameQuality]:
    """One plate, one frame: markers -> ``(4, 4)`` SE(3), or ``(None, why)``.

    ``markers``: ``(m, 3)`` marker-set positions in asset-label order.
    ``flags``: ``(m,)`` uint8, 1 = tracked (see module contract), or ``None``
    = flags unknown -> all markers assumed usable (design §5).
    Returns ``(T, quality)`` — ``T`` is ``None`` whenever a gate fired, and
    ``quality.reason`` names which one.

    The solve is **rigid registration** (design §4.2): Kabsch, det(R) = +1,
    of the lock template's usable subset onto the observed usable markers,
    minimizing ``sum_i ||R @ template_i + t - observed_i||^2``.  Because the
    template is expressed *in* the locked plate frame, the registered pose
    IS the plate frame: ``T = [[R, t], [0, 1]]`` maps lock frame to world.
    Optimal under noise for any marker geometry and exact for any 3-subset
    of the real radial-arm plates — no parallelogram or midpoint assumption
    anywhere (the measured plates have diagonal midpoints 5.5-12.8 mm apart,
    probe_20260811_live2).  Never consults world axes, streamed poses or
    ``u_up``: everything directional comes from the template, so nothing can
    flip mid-run (design §4 preamble).

    Gate order (each is a design citation, not defensiveness):

    * count != the lock template's 4 rows -> ``None``: asset-order indexing
      may have shifted (§4.2, review finding int-11);
    * < 3 usable -> ``None`` (§2: three markers suffice, two never);
    * non-finite usable marker -> ``None`` (degenerate-input guard, §4.2);
    * collinear/collapsed usable subset -> ``None``: the rotation is
      unobservable and Kabsch's reflection correction meaningless (§4.2
      "reflection guard");
    * registration RMS residual above :data:`TEMPLATE_RMS_TOL_M` -> ``None``:
      the label-swap/geometry gate (review finding geo-4) — a swap moves
      offsets by >= cm against 0.02-0.15 mm noise, and unlike the old
      diagonal-length check it also catches a swap *along* one diagonal.
    """
    m = np.asarray(markers, dtype=float)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError(f"markers must have shape (m, 3); got {m.shape}")
    template = np.asarray(lock.template_m, dtype=float)
    if m.shape[0] != template.shape[0]:
        return None, FrameQuality(reason="marker_count_mismatch")
    if flags is None:
        usable = np.ones(4, dtype=bool)
    else:
        f = np.asarray(flags)
        if f.shape != (4,):
            raise ValueError(
                f"flags must have shape (4,) to match markers; got {f.shape}")
        usable = f.astype(bool)
    n_usable = int(usable.sum())
    n_flagged_out = 4 - n_usable
    used_tuple = tuple(bool(x) for x in usable)
    if n_usable < 3:
        return None, FrameQuality(n_usable=n_usable, used=used_tuple,
                                  n_flagged_out=n_flagged_out,
                                  reason="too_few_usable")
    if not np.isfinite(m[usable]).all():
        return None, FrameQuality(n_usable=n_usable, used=used_tuple,
                                  n_flagged_out=n_flagged_out,
                                  reason="non_finite_marker")

    fit = _kabsch(template[usable], m[usable])
    if fit is None:
        return None, FrameQuality(n_usable=n_usable, used=used_tuple,
                                  n_flagged_out=n_flagged_out,
                                  reason="degenerate_markers")
    r, t_vec, rms = fit
    if not (rms <= TEMPLATE_RMS_TOL_M):             # NaN-rejecting comparison
        return None, FrameQuality(n_usable=n_usable, used=used_tuple,
                                  n_flagged_out=n_flagged_out,
                                  rms_residual_m=rms,
                                  reason="template_rms_gate")

    t = np.eye(4)
    t[0:3, 0:3] = r
    t[0:3, 3] = t_vec
    if not np.isfinite(t).all():
        return None, FrameQuality(n_usable=n_usable, used=used_tuple,
                                  n_flagged_out=n_flagged_out,
                                  rms_residual_m=rms,
                                  reason="non_finite_result")
    return t, FrameQuality(n_usable=n_usable, used=used_tuple,
                           n_flagged_out=n_flagged_out,
                           rms_residual_m=rms,
                           reason="")


def infer_all(frame_markers, frame_flags, locks: dict,
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    """All plates of one frame -> ``(poses, valid, quality)`` (design §4.2).

    ``frame_markers``/``frame_flags`` are one marker-ring entry's dicts
    (plate -> array), either possibly ``None`` (marker sets absent / flags
    unknown, ``mocap_rx`` semantics); ``locks`` maps plate -> lock.

    ``poses`` is ``(N_USED_RIGID_BODIES, 4, 4)`` — the shape of the
    ``MocapRx._incoming`` slice ``mocap_to_q`` consumes — identity where
    nothing was inferred; ``valid`` is the explicit ``(N,)`` bool mask (an
    identity row must never be mistaken for a solved base pose); ``quality``
    maps each *locked* plate to its :class:`FrameQuality`, including the
    named reason when it produced nothing.
    """
    n = mc.N_USED_RIGID_BODIES
    poses = np.tile(np.eye(4), (n, 1, 1))
    valid = np.zeros(n, dtype=bool)
    quality: dict[int, FrameQuality] = {}
    for plate, lock in locks.items():
        plate = int(plate)
        if not 0 <= plate < n:
            raise ValueError(f"lock for plate {plate} is outside 0..{n - 1}")
        if frame_markers is None or plate not in frame_markers:
            quality[plate] = FrameQuality(reason="no_marker_set")
            continue
        flags = None if frame_flags is None else frame_flags.get(plate)
        t, q = infer_plate_frame(frame_markers[plate], flags, lock)
        quality[plate] = q
        if t is not None:
            poses[plate] = t
            valid[plate] = True
    return poses, valid, quality


# --------------------------------------------------------------------------
# Frame-based q (design §4.3 — benchmark variant, parallel to mocap_to_q)
# --------------------------------------------------------------------------


#: Proximal-pair composition orders, mirroring
#: ``UMArm_KINEMATICS.fkine.PROXIMAL_ORDERS`` -- the reader and the forward
#: model must agree about which hinge of the proximal universal joint is bolted
#: to the upper bracket, or ``q`` and ``fkine(q)`` describe different machines.
PROXIMAL_ORDERS = ("xy", "yx")


def swing_angles(v, order: str = "xy") -> tuple[float, float]:
    """Link direction -> the proximal pair ``(t1, t2)``, in the given order.

    ``"xy"`` inverts ``Rx(t1) Ry(t2) zhat == v`` and is exactly
    :func:`mocap_to_q.ujoint_angles`, reused rather than restated.  ``"yx"``
    inverts ``Ry(t2) Rx(t1) zhat == v``, whose closed form falls out of
    ``Ry(t2) Rx(t1) zhat = [sin t2 cos t1, -sin t1, cos t2 cos t1]``::

        t1 = asin(-v_y)          t2 = atan2(v_x, v_z)

    The two agree to first order and diverge as ``t1 * t2``, which is exactly
    the term the CAN arm's 2026-08-21 measurement found in the residual (see
    ``UMArm_KINEMATICS.fkine``).  ``v`` need not be unit length for ``"xy"``;
    ``"yx"`` needs it, so it is normalised here.
    """
    if order == "xy":
        return ujoint_angles(v)
    if order != "yx":
        raise ValueError(f"order must be one of {PROXIMAL_ORDERS}; got {order!r}")
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not (n > 0.0):
        raise ValueError("cannot read swing angles off a zero-length direction")
    v = v / n
    return (float(math.asin(max(-1.0, min(1.0, -v[1])))),
            float(math.atan2(v[0], v[2])))


def q_from_frames(frames, phis=None, order: str = "xy") -> np.ndarray | None:
    """Full plate frames -> the 12-DOF ``q``, exact for multi-joint poses.

    The body-chain-correct recipe of design §4.3 (rev 1's previous-plate
    version was geometrically wrong for the odd joints — review finding
    geo-0).  Per segment ``i`` with proximal plate ``P = 2i`` and distal
    plate ``D = 2i + 1``, both first converted to **body** frames
    ``R_body = R_inferred @ Rz(-phi_p)``:

    1. proximal joint: ``v = normalize(R_P^T (o_P - o_D))`` — proximal minus
       distal points +z at rest, the ``mocap_to_q`` link convention — and
       ``(t1, t2) = swing_angles(v, order)``, which inverts the proximal pair's
       composition exactly, so step 2 is a clean residual either way.
    2. distal joint: the residual ``R45 = (Rx(t1) Ry(t2))^T R_P^T R_D``
       equals ``exp(t3 A1) exp(t4 A2)`` with A1/A2 the 45-deg twist axes
       (x+y)/sqrt2 and (-x+y)/sqrt2; the basis change
       ``M = Rz(45)^T R45 Rz(45)`` maps those axes onto x and y, so
       ``M = Rx(t3) @ Ry(t4)`` and ``t3 = atan2(M[2,1], M[1,1])``,
       ``t4 = atan2(M[0,2], M[0,0])``.

    No swing-only alignment appears anywhere, so multi-joint ``q`` is exact
    by construction (machine-precision fkine round trip for all 12 joints,
    pinned in the tests), q10/q11 come from plate 5's body frame, and the
    whole thing is invariant to any rigid motion of the volume — only
    relative rotations and body-frame differences are ever read.

    ``order`` selects the proximal pair's composition and must match the
    forward model's (``UMArm_KINEMATICS.fkine``): ``"xy"`` is the legacy
    default, ``"yx"`` is what the CAN arm measured.  Getting it wrong does not
    fail -- it leaves a ``-t1*t2`` twist in the residual that the two-angle
    distal reader silently drops, and the error surfaces downstream as a
    u-joint centre that fkine puts millimetres away from where mocap sees it.

    ``frames``: ``(>= 6, 4, 4)`` inferred plate frames (rows beyond 5
    ignored).  ``phis``: per-plate family angles, length 6; ``None`` means
    the :data:`FAMILY_PHI_RAD` defaults — pass zeros when the frames are
    already body frames (e.g. straight from
    ``UMArm_KINEMATICS.plate_transforms``).

    Returns ``None`` for a degenerate frame set (non-finite values, or a
    collapsed P-D pair below ``mocap_constants.MIN_LINK_NORM``) — same
    honesty contract as ``mocap_to_q``.  Shape errors raise ``ValueError``.
    """
    fr = np.asarray(frames, dtype=float)
    if fr.ndim != 3 or fr.shape[1:] != (4, 4) or fr.shape[0] < 6:
        raise ValueError(
            f"frames must have shape (>= 6, 4, 4); got {fr.shape}")
    if phis is None:
        phi_arr = np.array([FAMILY_PHI_RAD[p] for p in range(6)])
    else:
        phi_arr = np.asarray(phis, dtype=float)
        if phi_arr.shape != (6,):
            raise ValueError(f"phis must have shape (6,); got {phi_arr.shape}")
    if not (np.isfinite(fr[0:6]).all() and np.isfinite(phi_arr).all()):
        return None

    r_body = [fr[p, 0:3, 0:3] @ _rz(-float(phi_arr[p])) for p in range(6)]
    origins = fr[0:6, 0:3, 3]

    q = np.empty(12, dtype=float)
    for i in range(3):
        p_idx, d_idx = 2 * i, 2 * i + 1
        v = r_body[p_idx].T @ (origins[p_idx] - origins[d_idx])
        n = float(np.linalg.norm(v))
        # NaN-rejecting comparison, as in mocap_to_q.link_vectors: a NaN norm
        # must reject the frame, and `n < MIN_LINK_NORM` would wave it through.
        if not (n >= mc.MIN_LINK_NORM):
            return None
        t1, t2 = swing_angles(v / n, order)
        prox = (_rx(t1) @ _ry(t2)) if order == "xy" else (_ry(t2) @ _rx(t1))
        r45 = prox.T @ r_body[p_idx].T @ r_body[d_idx]
        mtx = RZ45.T @ r45 @ RZ45
        q[4 * i + 0] = t1
        q[4 * i + 1] = t2
        q[4 * i + 2] = math.atan2(mtx[2, 1], mtx[1, 1])
        q[4 * i + 3] = math.atan2(mtx[0, 2], mtx[0, 0])

    # Final invariant, mirroring mocap_to_q: never hand a non-finite q to a
    # consumer that would happily act on it.
    if not np.isfinite(q).all():
        return None
    return q
