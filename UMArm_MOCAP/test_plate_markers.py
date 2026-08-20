"""Tests for :mod:`UMArm_MOCAP.plate_markers`.

Pure files and pure math: no sockets, no cameras, no arm.  Two kinds of claim
are checked, and they fail for different reasons:

1. **What the committed capture actually contains**
   (:class:`TestCommittedTemplates`, :class:`TestCommittedGeometry`,
   :class:`TestCommittedMeta`).  These read
   :data:`plate_markers.DEFAULT_TEMPLATE_PATH` — the tracked
   ``campaign_2026-08-11_recal/locks.json`` — and re-derive, from the marker
   coordinates alone, every property the module's docstring asserts about
   them: six plates, metres, unequal radial arms in a 60-115 mm band, four
   coplanar markers, diagonals crossing square, and an origin sitting on the
   diagonal *lines* rather than at the centroid.  A failure here means the
   file the drawing code depends on changed, or the wrong file was put in its
   place; it does not mean the code is wrong.

   Where a property is also *stored* in the lock (``arm_radii_m``,
   ``out_of_plane_rms_m``, ``diagonal_crossing_deg``), the re-derived value is
   compared against it.  :func:`plate_markers.load_templates` reads only
   ``template_m``, so the stored statistics are an independent witness: they
   agree only if the array handed back is the same geometry the probe
   measured, which is exactly the "did I load the right field of the right
   file?" question.

2. **What the code does with a file that is not that one**
   (:class:`TestPlateMarkerPoints`, :class:`TestMalformedFiles`,
   :class:`TestSyntheticTemplates`).  These build their own JSON in
   ``tmp_path`` — a hand-checkable square plate whose radii are known in
   closed form, a wrong-shaped template, a non-finite one, an empty one, an
   absent one — and pin that the loader *raises* rather than returning an
   empty dict.  The distinction matters at the drawing end: an arm rendered
   with no markers looks exactly like an arm whose markers are correctly at
   the origin, so silence is the one failure mode that cannot be seen in the
   picture the module exists to produce.

:func:`plate_markers.verify_against_live` is **not** covered.  It needs a
started receiver and a still arm, i.e. the cameras, which this suite is
required to run without.  Its pose-free helper :func:`_edge_delta_mm` is
exercised directly instead, since that one is pure math.

Run:  ``python -m pytest UMArm_MOCAP/test_plate_markers.py -q``
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
import os
import sys

import numpy as np
import pytest

# The module under test, imported flat so this file works whichever way pytest
# rooted itself; the repo root goes on the path too, matching the rest of the
# UMArm_MOCAP suite.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plate_markers as pm  # noqa: E402
from marker_frame import CROSSING_TOL_DEG  # noqa: E402
from mocap_probe import X_RESIDUAL_EXIT_DEG  # noqa: E402

# --------------------------------------------------------------------------
# Budgets.  Each is a stated multiple of what the committed capture measures,
# so a failure means a changed file rather than a tight threshold.
# --------------------------------------------------------------------------

#: Out-of-plane budget, millimetres.  The worst committed plate (4) deviates
#: 0.169 mm from its own best-fit plane and the best (2) deviates 0.012 mm;
#: 0.25 mm clears the worst by 1.5x while still refusing anything that is not
#: a flat four-marker plate.  The module docstring's "~0.2 mm" is this number.
COPLANAR_TOL_MM = 0.25

#: How far from 90 deg the diagonals may cross.  Committed worst is 0.285 deg
#: (plate 1).  Note this is 90x tighter than :data:`CROSSING_TOL_DEG`, the
#: 25 deg at which ``compute_plate_lock`` refuses outright: the lock's gate
#: admits a badly skewed quadrilateral, whereas the real plates are square to
#: a third of a degree, and it is the second fact a wrong file would break.
CROSSING_TOL_DEG_MEASURED = 0.5

#: Radial-arm band, millimetres.  Committed radii span 62.09-113.18 mm across
#: the six plates.  The wide bracket is deliberate: its job is to catch a
#: units error (a millimetre-valued file reads as 62 000) or a different
#: plate entirely (the printed pad's markers sit ~25 mm out), not to re-pin
#: numbers the geometry tests already re-derive.
RADIUS_MIN_MM = 50.0
RADIUS_MAX_MM = 130.0

#: Minimum radius change under a cyclic relabel, millimetres — see
#: :meth:`TestCommittedGeometry.test_radii_break_the_cyclic_ambiguity`.  The
#: committed worst case is 9.32 mm (plate 5 under a two-step shift) and the
#: rest-marker standard deviation over the capture is at most 0.073 mm, so
#: 5 mm sits 60x above the noise and 1.9x below the weakest real plate.
RELABEL_SEPARATION_MM = 5.0


# --------------------------------------------------------------------------
# Test-local geometry, deliberately independent of the module and of
# marker_frame: nothing below calls anything that plate_markers calls.
# --------------------------------------------------------------------------


def best_fit_plane(points: np.ndarray):
    """``(centroid, unit normal)`` of the least-squares plane through *points*.

    Plain SVD of the centred coordinates; the normal is signed to +z so that
    comparisons against the template's own z axis have a defined sense.
    """
    c = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - c)
    n = vt[2]
    return c, n * (1.0 if n[2] >= 0.0 else -1.0)


def out_of_plane_mm(points: np.ndarray) -> np.ndarray:
    """Each point's signed distance from the best-fit plane, millimetres."""
    c, n = best_fit_plane(points)
    return (points - c) @ n * 1e3


def plane_basis(points: np.ndarray):
    """Two in-plane unit vectors spanning the best-fit plane of *points*."""
    c = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - c)
    return vt[0], vt[1]


def point_line_distance(p, a, b) -> float:
    """Distance from *p* to the infinite line through *a* and *b*."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d / np.linalg.norm(d)
    w = np.asarray(p, dtype=float) - np.asarray(a, dtype=float)
    return float(np.linalg.norm(w - np.dot(w, d) * d))


def rodrigues(axis, angle_rad: float) -> np.ndarray:
    """Test-local Rodrigues rotation, so the spin test borrows no repo code."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return (np.eye(3) + math.sin(angle_rad) * k
            + (1.0 - math.cos(angle_rad)) * (k @ k))


def chord_lengths(points: np.ndarray) -> dict:
    """``{(i, j): |p_i - p_j|}`` for the six chords of a four-marker plate."""
    return {(i, j): float(np.linalg.norm(points[i] - points[j]))
            for i, j in itertools.combinations(range(4), 2)}


def square_template(half_mm: float = 50.0) -> list:
    """A hand-checkable plate: a square of half-diagonal *half_mm*, in metres.

    Every radius is exactly ``half_mm`` and every cyclic relabel maps the set
    onto itself — the degenerate case the real plates avoid, used below as a
    negative control for the resolvability test.
    """
    r = half_mm * 1e-3
    return [[r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), 0.0]
            for a in (45.0, 135.0, 225.0, 315.0)]


def write_locks(tmp_path, plates: dict, meta: dict | None = None,
                name: str = "locks.json", allow_nan: bool = False) -> str:
    """Write a locks-shaped JSON file into *tmp_path* and return its path."""
    doc = {"meta": dict(meta or {}), "plates": plates}
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, allow_nan=allow_nan))
    return path


@pytest.fixture(scope="module")
def committed() -> dict:
    """The committed templates, loaded once for the whole module."""
    return pm.load_templates()


@pytest.fixture(scope="module")
def committed_doc() -> dict:
    """The raw locks JSON, for cross-checks against fields the loader ignores."""
    with open(pm.DEFAULT_TEMPLATE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# 1. What load_templates hands back
# --------------------------------------------------------------------------


class TestCommittedTemplates:
    """The shape, type and units of the committed template set."""

    def test_default_path_exists_and_is_the_tracked_template_file(self):
        """The module's whole premise is that this file is *committed*.

        Failing means the dependency moved out from under the drawing code —
        most likely into ``logs/`` (untracked), which is the exact mistake the
        module's docstring says a tracked location exists to prevent.  A fresh
        clone would then draw an arm with no markers at all.

        PORT NOTE (this workspace).  Upstream this asserted the path ended
        ``results/campaign_2026-08-11_recal/locks.json``; that 1.2 GB campaign
        directory is not carried here, and the 15 kB locks file itself lives at
        ``UMArm_MOCAP/templates/rs485_locks_example.json`` instead, byte for
        byte.  The claim under test is unchanged — the file is tracked, it is
        not under ``logs/``, and it is the RS485 arm's 2026-08-11 mint — only
        where it sits has moved.  Every other test in this file reads its
        *contents* and is untouched.
        """
        assert os.path.isfile(pm.DEFAULT_TEMPLATE_PATH)
        parts = pm.DEFAULT_TEMPLATE_PATH.replace("\\", "/").split("/")
        assert "logs" not in parts
        assert parts[-1] == "rs485_locks_example.json"
        assert parts[-2] == "templates"

    def test_six_plates_each_four_by_three_and_finite(self, committed):
        """Six plates, ``(4, 3)`` and finite, is the contract the scene draws.

        A plate short of four markers, a transposed ``(3, 4)`` array, or a NaN
        would each reach MuJoCo as a site at an undefined position; failing
        here says the loader's own shape gate let something through.
        """
        assert len(committed) == 6
        for plate, t in committed.items():
            assert isinstance(t, np.ndarray), plate
            assert t.shape == (4, 3), plate
            assert t.dtype == np.float64, plate
            assert np.isfinite(t).all(), plate

    def test_keys_are_plate_integers_matching_PLATES(self, committed):
        """Keys are ``int``, and they are exactly plates 0-5.

        JSON object keys are strings; if the ``int(key)`` conversion were
        dropped, ``templates[0]`` would raise ``KeyError`` for every caller.
        The set equality also pins the docstring's claim that plate 6, the
        end-effector plate, is ABSENT from the Motive project and that nothing
        here invents one — a seventh plate appearing means someone did.
        """
        assert all(isinstance(k, int) for k in committed)
        assert sorted(committed) == list(pm.PLATES) == [0, 1, 2, 3, 4, 5]
        assert 6 not in committed

    def test_coordinates_are_metres(self, committed):
        """Every marker sits 50-130 mm from its plate origin, i.e. in metres.

        This is the units gate.  A file in millimetres would put every radius
        near 62 000 and a file in some other frame would put them nowhere near
        a plate; either would be drawn without complaint and would look, in
        the merged scene, like an arm that had exploded.
        """
        for plate, t in committed.items():
            radii_mm = np.linalg.norm(t, axis=1) * 1e3
            assert radii_mm.min() > RADIUS_MIN_MM, (plate, radii_mm)
            assert radii_mm.max() < RADIUS_MAX_MM, (plate, radii_mm)

    def test_radii_span_the_measured_sixty_to_hundred_fifteen_band(
            self, committed):
        """Across the six plates the radii genuinely occupy the 60-115 mm band.

        Per-plate bounds alone would pass on six identical plates.  The real
        arm tapers — the base plate's arms reach 113 mm while the distal ones
        come in to 62 mm — so the *spread across plates* is itself evidence
        that all six plates were read, and not one plate copied six times.
        """
        radii = np.array([r for p in pm.PLATES
                          for r in pm.arm_radii_mm(p, committed)])
        assert 60.0 <= radii.min() <= 70.0, radii.min()
        assert 105.0 <= radii.max() <= 120.0, radii.max()
        assert radii.max() - radii.min() > 40.0

    def test_arm_radii_mm_are_the_template_row_norms(self, committed):
        """``arm_radii_mm`` is exactly the row norms of the template, in mm.

        Re-derived rather than copied: a factor-of-1000 slip, an axis-wrong
        norm (``axis=0`` gives three numbers, not four), or a radius measured
        from the centroid instead of the origin would each survive a
        golden-number comparison against a value produced the same way.
        """
        for plate, t in committed.items():
            got = pm.arm_radii_mm(plate, committed)
            want = np.linalg.norm(t, axis=1) * 1e3
            assert len(got) == 4, plate
            assert all(isinstance(v, float) for v in got), plate
            assert np.allclose(got, want, rtol=0, atol=1e-12), plate

    def test_arm_radii_mm_agree_with_the_locks_own_stored_radii(
            self, committed, committed_doc):
        """The re-derived radii match the lock's stored ``arm_radii_m``.

        ``load_templates`` never reads that field, so it is an independent
        witness written by the probe at capture time.  Agreement to a
        micrometre says the ``template_m`` array handed to the scene is the
        same geometry the probe measured and reported; disagreement would mean
        the two fields describe different plates, which no shape check catches.
        """
        for plate in pm.PLATES:
            stored = np.asarray(
                committed_doc["plates"][str(plate)]["arm_radii_m"],
                dtype=float) * 1e3
            derived = np.asarray(pm.arm_radii_mm(plate, committed))
            assert np.allclose(derived, stored, rtol=0, atol=1e-9), plate


# --------------------------------------------------------------------------
# 2. The geometry the docstring claims
# --------------------------------------------------------------------------


class TestCommittedGeometry:
    """Coplanarity, square diagonals, unequal arms, and where the origin is."""

    def test_radii_break_the_cyclic_ambiguity(self, committed):
        """No plate's four radii survive a cyclic relabel unchanged.

        This is the resolvability claim in ``arm_radii_mm``'s docstring, stated
        as the thing that actually matters.  Four markers on arms of *equal*
        length at 90 deg spacing map onto themselves under a quarter-turn
        relabel, so the plate's azimuth is ambiguous to 90 deg and a pose
        solve can lock onto the wrong branch.  What forbids that is not that
        every pair of radii differs — plate 2 has two arms within 0.12 mm —
        but that no rotation of the *sequence* reproduces it.  Failing means
        a set of near-equal radii came back, i.e. the wrong file was loaded.
        """
        for plate in pm.PLATES:
            r = np.asarray(pm.arm_radii_mm(plate, committed))
            separations = [float(np.abs(r - np.roll(r, k)).max())
                           for k in (1, 2, 3)]
            assert min(separations) > RELABEL_SEPARATION_MM, (plate, r,
                                                              separations)

    def test_a_square_plate_would_fail_that_test(self, tmp_path):
        """Negative control: the degenerate square scores exactly zero.

        Without it the previous test proves nothing — a threshold no geometry
        can fail is not a test.  A square plate is invariant under every
        cyclic relabel, so its separation is 0 mm against the 5 mm budget.
        """
        path = write_locks(tmp_path, {"0": {"template_m": square_template()}})
        r = np.asarray(pm.arm_radii_mm(0, pm.load_templates(path)))
        separations = [float(np.abs(r - np.roll(r, k)).max()) for k in (1, 2, 3)]
        assert min(separations) == pytest.approx(0.0, abs=1e-12)
        assert min(separations) < RELABEL_SEPARATION_MM

    def test_markers_are_coplanar(self, committed):
        """All four markers sit within 0.25 mm of one plane.

        The committed locks were minted from a still capture of a rigid plate,
        so the four markers are coplanar to the fit; the module's docstring
        puts that at ~0.2 mm.  A plate whose markers no longer lie in a plane
        has been knocked, and the plane is what the drawn plate body is hung
        on — an out-of-plane marker would be drawn floating off the bracket.
        """
        for plate, t in committed.items():
            dev = np.abs(out_of_plane_mm(t))
            assert dev.max() < COPLANAR_TOL_MM, (plate, dev)

    def test_out_of_plane_scatter_matches_the_locks_own_statistic(
            self, committed, committed_doc):
        """The re-derived out-of-plane RMS equals the lock's stored one.

        Two computations of the same quantity, one here from the coordinates
        and one written by the probe over 3600 frames.  They agree to a
        nanometre only if the template really is the rest shape the probe
        summarised, which is a sharper statement than "it is flat".
        """
        for plate, t in committed.items():
            derived = float(np.sqrt(np.mean(out_of_plane_mm(t) ** 2)))
            stored = committed_doc["plates"][str(plate)][
                "out_of_plane_rms_m"] * 1e3
            assert derived == pytest.approx(stored, abs=1e-6), plate

    def test_the_two_longest_chords_are_the_opposite_pairs(self, committed):
        """Markers 0-2 and 1-3 are the diagonals, not the sides.

        The template rows arrive in cyclic order, so the diagonals *should* be
        the opposite pairs; re-deriving them as the two longest of the six
        chords proves the ordering rather than assuming it.  A relabelled or
        re-sorted template would put a side where a diagonal belongs, and the
        crossing and origin tests below would then be measuring the wrong
        lines while still, on a near-square plate, nearly passing.
        """
        for plate, t in committed.items():
            chords = chord_lengths(t)
            longest = sorted(chords, key=chords.get, reverse=True)[:2]
            assert set(longest) == {(0, 2), (1, 3)}, (plate, chords)

    def test_diagonals_cross_within_half_a_degree_of_square(self, committed):
        """The diagonals meet at 90 deg to better than half a degree.

        The plates are machined square even though the arms are unequal, which
        is what lets a diagonal pair define an in-plane frame at all.  The
        budget here is 50x tighter than ``CROSSING_TOL_DEG``, the angle at
        which ``compute_plate_lock`` refuses: that gate is there to catch a
        mislabeled asset, whereas this test is asserting what the real plates
        measure, and only a different plate would fail it.
        """
        assert CROSSING_TOL_DEG_MEASURED < CROSSING_TOL_DEG / 10.0
        for plate, t in committed.items():
            d02 = t[2] - t[0]
            d13 = t[3] - t[1]
            cos = abs(float(np.dot(d02, d13))) / (np.linalg.norm(d02)
                                                  * np.linalg.norm(d13))
            crossing_deg = math.degrees(math.acos(min(1.0, cos)))
            assert abs(crossing_deg - 90.0) < CROSSING_TOL_DEG_MEASURED, (
                plate, crossing_deg)

    def test_crossing_angle_matches_the_locks_own_statistic(
            self, committed, committed_doc):
        """The re-derived crossing angle equals the lock's stored one.

        Agreement is to 1e-3 deg rather than to machine precision because the
        probe forms the angle from the fitted in-plane diagonal directions
        while this test forms it from the raw template rows; the two differ by
        the out-of-plane component, which is 2e-6 deg on the worst plate.  A
        wrong file moves the angle by whole degrees, so the loose budget costs
        the check nothing.
        """
        for plate, t in committed.items():
            d02 = t[2] - t[0]
            d13 = t[3] - t[1]
            cos = abs(float(np.dot(d02, d13))) / (np.linalg.norm(d02)
                                                  * np.linalg.norm(d13))
            derived = math.degrees(math.acos(min(1.0, cos)))
            stored = committed_doc["plates"][str(plate)]["diagonal_crossing_deg"]
            assert derived == pytest.approx(stored, abs=1e-3), plate

    def test_origin_is_the_diagonal_intersection(self, committed):
        """The origin lies on both diagonal LINES, in-plane, to machine zero.

        This is the first of the docstring's two disclaimers, stated
        positively: the template origin is where the two diagonal lines cross,
        which lands on the u-joint centre only because the arms are radial.
        Projected into the plate's own plane the origin is on both lines to
        better than a nanometre; in 3-D the residual is the plate's
        out-of-plane thickness and nothing more.  Failing means the templates
        were re-centred on something else, at which point hanging them off a
        simulated plate body — whose origin IS the kinematic u-joint centre —
        no longer puts them in the right place.
        """
        for plate, t in committed.items():
            e0, e1 = plane_basis(t)
            xy = np.stack([t @ e0, t @ e1], axis=1)
            origin_xy = np.zeros(2)
            for i, j in ((0, 2), (1, 3)):
                d = point_line_distance(origin_xy, xy[i], xy[j])
                assert d * 1e3 < 1e-6, (plate, i, j, d * 1e3)
            for i, j in ((0, 2), (1, 3)):
                d = point_line_distance(np.zeros(3), t[i], t[j])
                assert d * 1e3 < COPLANAR_TOL_MM, (plate, i, j, d * 1e3)

    def test_origin_is_neither_the_centroid_nor_a_diagonal_midpoint(
            self, committed):
        """The origin sits millimetres from the centroid and from both midpoints.

        The complement of the previous test, and the reason it is not vacuous:
        with unequal arms the diagonal lines still cross at the origin while
        their *midpoints* do not, so the centroid is several millimetres away.
        A template quietly re-centred on the centroid would still satisfy a
        loose "near the middle" check; it would move every drawn marker by
        that offset and shift nothing else, which is invisible in a picture
        unless it is looked for.
        """
        for plate, t in committed.items():
            centroid_mm = float(np.linalg.norm(t.mean(axis=0))) * 1e3
            assert centroid_mm > 2.0, (plate, centroid_mm)
            for i, j in ((0, 2), (1, 3)):
                midpoint_mm = float(np.linalg.norm((t[i] + t[j]) / 2.0)) * 1e3
                assert midpoint_mm > 2.0, (plate, i, j, midpoint_mm)

    def test_plane_normal_is_the_template_z_axis(self, committed):
        """The plate plane is the template's own z = 0 plane, to 0.05 deg.

        The scene hangs these points on a plate body whose z is the bracket
        normal.  Were the template's plane tilted in its own frame, every
        marker would be drawn on a cocked disc; the committed locks put the
        best-fit normal within 0.021 deg of +z, so 0.05 deg is a factor of two
        of headroom on the worst plate.
        """
        for plate, t in committed.items():
            _c, n = best_fit_plane(t)
            tilt_deg = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
            assert tilt_deg < 0.05, (plate, tilt_deg)
            assert np.abs(t[:, 2]).max() * 1e3 < COPLANAR_TOL_MM, plate

    def test_three_degrees_of_azimuth_moves_a_marker_about_five_millimetres(
            self, committed):
        """The docstring's "three degrees is 5 mm" holds on the real geometry.

        This is the second disclaimer, made concrete: the lock frame agrees
        with the streamed plate frame only to a few degrees, so a drawn marker
        landing millimetres from a reported one is the frame convention rather
        than a fault.  Spinning the base plate's template 3 deg about its own
        normal — the plate whose arms reach 113 mm — moves its markers
        4.8-5.9 mm, which brackets the quoted 5 mm.  A reader who takes a 5 mm
        disagreement in the merged scene as a mounting error is over-reading it.
        """
        t = committed[0]
        _c, n = best_fit_plane(t)
        moved = t @ rodrigues(n, math.radians(3.0)).T
        disp_mm = np.linalg.norm(moved - t, axis=1) * 1e3
        assert 4.0 < disp_mm.min() < 6.0, disp_mm
        assert 4.0 < disp_mm.max() < 7.0, disp_mm
        # The chord identity the docstring's arithmetic is the small-angle
        # form of: a point at perpendicular radius r swept through theta moves
        # 2 r sin(theta / 2).  Exact, so it holds to machine precision, and it
        # says the displacement is set by the arm length and nothing else.
        r_perp_mm = np.linalg.norm(t - np.outer(t @ n, n), axis=1) * 1e3
        assert disp_mm == pytest.approx(
            r_perp_mm * 2.0 * math.sin(math.radians(1.5)), rel=1e-12)
        assert 92.0 < r_perp_mm.min() and r_perp_mm.max() < 114.0


# --------------------------------------------------------------------------
# 3. Provenance
# --------------------------------------------------------------------------


class TestCommittedMeta:
    """What ``template_meta`` reports about where the numbers came from."""

    def test_capture_provenance_is_the_2026_08_11_still(self):
        """The capture is the 30 s, 120 fps, labeled still of 2026-08-11.

        A figure drawn from these templates should be able to name its source
        without anyone remembering it.  Failing means the committed locks were
        re-minted from a different capture — which is legitimate, but the
        module docstring, ``base_poses`` and every figure caption that quotes
        the date then describe a capture that no longer exists.
        """
        meta = pm.template_meta()
        stamp = datetime.datetime.fromisoformat(meta["captured_wall"])
        assert stamp.date() == datetime.date(2026, 8, 11)
        assert meta["seconds"] == pytest.approx(30.0)
        assert meta["regime"] == "labeled"
        assert 118.0 < meta["fps_measured"] < 122.0

    def test_frame_count_reconciles_with_seconds_and_rate(self):
        """Every plate contributes 4 markers for the full window.

        Re-derived rather than asserted: 3600 frames at the measured 119.99 fps
        IS the claimed 30 s, and each plate saw four markers in every one of
        them.  A plate that dropped markers mid-capture would have a second
        count key, and its template would then be a mean over a shape the
        plate only sometimes had.
        """
        meta = pm.template_meta()
        counts = meta["per_plate_marker_counts"]
        assert sorted(int(k) for k in counts) == list(pm.PLATES)
        for plate, per_count in counts.items():
            assert list(per_count) == ["4"], (plate, per_count)
            frames = per_count["4"]
            assert frames / meta["fps_measured"] == pytest.approx(
                meta["seconds"], rel=1e-3), plate

    def test_path_and_plate_list_are_carried(self):
        """``path`` and ``plates`` say which file was read and what was in it."""
        meta = pm.template_meta()
        assert meta["path"] == pm.DEFAULT_TEMPLATE_PATH
        assert meta["plates"] == list(pm.PLATES)
        assert all(isinstance(p, int) for p in meta["plates"])

    def test_one_x_mode_per_plate_and_all_are_streamed(self):
        """Every plate's azimuth zero is the streamed rest orientation.

        The module docstring rests its "agrees with the streamed plate frame
        only to a few degrees" disclaimer on ``x_mode == "streamed"``.  Were a
        plate minted in ``diagonal45`` mode instead, its azimuth zero would be
        the nearest diagonal plus 45 deg, and on the base plate that differs
        from the streamed orientation by the ~16 deg recorded below — a 30 mm
        error at a 113 mm arm, ten times the discrepancy a reader is told to
        ignore.
        """
        meta = pm.template_meta()
        assert sorted(meta["x_mode"]) == list(pm.PLATES)
        assert set(meta["x_mode"].values()) == {"streamed"}

    def test_one_x_residual_per_plate_within_the_probe_gate(self):
        """Each plate carries a finite x residual, and the capture passed.

        ``x_residual_deg`` is how far a plate's marker arms sit from the
        designed 45-deg azimuth, not the frame disagreement itself; the probe
        refuses a capture above ``X_RESIDUAL_EXIT_DEG``.  The base plate's
        ~16 deg is the 2026-08-11 finding that forced ``x_mode="streamed"``
        into existence, so a run where plate 0 came back near zero would mean
        the committed locks are no longer that capture.
        """
        meta = pm.template_meta()
        residuals = meta["x_residual_deg"]
        assert sorted(residuals) == list(pm.PLATES)
        for plate, value in residuals.items():
            assert isinstance(value, float), plate
            assert math.isfinite(value), plate
            assert 0.0 <= value < X_RESIDUAL_EXIT_DEG, (plate, value)
        assert 12.0 < residuals[0] < 20.0, residuals[0]
        assert max(residuals.values()) == residuals[0]

    def test_meta_radii_agree_with_arm_radii_mm(self, committed):
        """``meta["arm_radii_mm"]`` is ``arm_radii_mm`` rounded to 0.01 mm.

        The two are computed from different fields of the file — the meta from
        the stored ``arm_radii_m``, the function from ``template_m`` — so a
        disagreement means one plate's stored summary and its stored geometry
        have drifted apart, which nothing else in the pipeline would report.
        """
        meta = pm.template_meta()
        for plate in pm.PLATES:
            derived = [round(v, 2) for v in pm.arm_radii_mm(plate, committed)]
            assert meta["arm_radii_mm"][plate] == pytest.approx(derived,
                                                                abs=1e-9), plate

    def test_meta_does_not_refuse_a_file_that_load_templates_would(
            self, tmp_path):
        """``template_meta`` is deliberately weaker than ``load_templates``.

        Pinned because the asymmetry is a trap rather than a symmetry: a
        metadata-only file makes ``load_templates`` raise "carries no plate
        templates", while ``template_meta`` returns happily with
        ``plates == []``.  Anything that records provenance BEFORE loading the
        geometry — a figure caption, a scene record — therefore cannot use the
        meta call as its file check.  A lock missing its own residual comes
        back as NaN by the same tolerance, not as an error.
        """
        meta_only = os.path.join(str(tmp_path), "meta_only.json")
        with open(meta_only, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"regime": "labeled"}}, fh)
        meta = pm.template_meta(meta_only)
        assert meta["plates"] == []
        assert meta["x_mode"] == {}
        with pytest.raises(ValueError):
            pm.load_templates(meta_only)

        partial = write_locks(tmp_path, {"0": {"template_m": square_template()}},
                              name="no_residual.json")
        meta = pm.template_meta(partial)
        assert meta["x_mode"] == {0: None}
        assert math.isnan(meta["x_residual_deg"][0])
        assert meta["arm_radii_mm"] == {0: []}

    def test_meta_reads_an_alternate_path(self, tmp_path):
        """``template_meta(path)`` reports the file it was handed, not the default.

        A figure that quotes the default provenance while having been drawn
        from a one-off capture is worse than a figure with no provenance.
        """
        path = write_locks(
            tmp_path,
            {"3": {"template_m": square_template(), "x_mode": "diagonal45",
                   "x_residual_deg": 7.5, "arm_radii_m": [0.05] * 4}},
            meta={"captured_wall": "2026-01-02T03:04:05", "regime": "labeled"})
        meta = pm.template_meta(path)
        assert meta["path"] == path
        assert meta["plates"] == [3]
        assert meta["x_mode"] == {3: "diagonal45"}
        assert meta["x_residual_deg"][3] == pytest.approx(7.5)
        assert meta["arm_radii_mm"][3] == [50.0] * 4
        assert meta["captured_wall"] == "2026-01-02T03:04:05"


# --------------------------------------------------------------------------
# 4. Lookup behaviour
# --------------------------------------------------------------------------


class TestPlateMarkerPoints:
    """Selecting one plate, and what happens when it is not there."""

    def test_returns_that_plate_unchanged(self, committed):
        """The points handed back are the loaded template for that plate."""
        for plate, t in committed.items():
            got = pm.plate_marker_points_m(plate, committed)
            assert got.shape == (4, 3)
            assert np.array_equal(got, t), plate

    def test_missing_plate_raises_KeyError_naming_what_is_available(
            self, committed):
        """Asking for plate 6 raises, and the message lists 0-5.

        Plate 6 is the end-effector plate, absent from the Motive project, so
        this is the mistake a caller will actually make.  A silent ``None`` or
        an empty array would be drawn as a marker at the plate origin — a
        plausible-looking point that no camera ever saw.  The message must
        name what IS present, because the useful next question is always
        "then which plates do I have?".
        """
        with pytest.raises(KeyError) as excinfo:
            pm.plate_marker_points_m(6, committed)
        message = str(excinfo.value)
        assert "6" in message
        for plate in pm.PLATES:
            assert str(plate) in message
        assert "0, 1, 2, 3, 4, 5" in message

    def test_a_supplied_dict_is_not_silently_topped_up_from_the_default_file(
            self, committed):
        """A caller-supplied ``templates`` is the whole world for that call.

        Plate 0 is in the committed file, so a lookup that fell back to disk
        when the supplied dict lacked it would succeed here and quietly mix
        two captures in one picture.  It must raise instead.
        """
        partial = {3: committed[3]}
        assert 0 in pm.load_templates()
        with pytest.raises(KeyError):
            pm.plate_marker_points_m(0, partial)

    def test_lookup_does_not_mutate_the_supplied_dict(self, committed):
        """Reading a plate leaves the caller's dict exactly as it was.

        The scene loads once and reads six times; a lookup with a side effect
        would make the sixth plate depend on the order the first five were
        drawn in.
        """
        supplied = {p: committed[p].copy() for p in pm.PLATES}
        before = {p: supplied[p].copy() for p in supplied}
        for plate in pm.PLATES:
            pm.plate_marker_points_m(plate, supplied)
            pm.arm_radii_mm(plate, supplied)
        assert sorted(supplied) == list(pm.PLATES)
        for plate in pm.PLATES:
            assert np.array_equal(supplied[plate], before[plate]), plate

    def test_default_argument_reads_the_committed_file(self, committed):
        """Omitting ``templates`` falls back to the committed capture.

        The convenience path the CLI and the scene both take; if it silently
        loaded something else, every drawn marker would move together and the
        picture would still look self-consistent.
        """
        assert np.array_equal(pm.plate_marker_points_m(2), committed[2])
        assert pm.arm_radii_mm(4) == pytest.approx(
            pm.arm_radii_mm(4, committed))


# --------------------------------------------------------------------------
# 5. Files that are not the committed one
# --------------------------------------------------------------------------


class TestSyntheticTemplates:
    """A hand-checkable file, to pin the math independently of the capture."""

    def test_square_plate_radii_are_the_closed_form(self, tmp_path):
        """A square of half-diagonal 50 mm gives four radii of exactly 50 mm.

        The committed-file tests compare the code against a measurement; this
        one compares it against arithmetic anyone can do on paper, so a shared
        error in how both the file and the loader treat units cannot hide.
        """
        path = write_locks(tmp_path, {"0": {"template_m": square_template(50.0)}})
        templates = pm.load_templates(path)
        assert sorted(templates) == [0]
        radii = pm.arm_radii_mm(0, templates)
        assert radii == pytest.approx([50.0] * 4, abs=1e-9)

    def test_a_bare_plate_dict_without_the_plates_wrapper_still_loads(
            self, tmp_path):
        """A file that is already ``{plate: lock}`` is accepted.

        ``load_templates`` falls back to the document itself when there is no
        ``"plates"`` key, which is what lets a hand-extracted single lock be
        passed straight in.  Pinned because it is deliberate tolerance and
        would otherwise look like an accident worth "fixing".
        """
        path = os.path.join(str(tmp_path), "bare.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"2": {"template_m": square_template(60.0)}}, fh)
        templates = pm.load_templates(path)
        assert sorted(templates) == [2]
        assert pm.arm_radii_mm(2, templates) == pytest.approx([60.0] * 4,
                                                              abs=1e-9)

    def test_entries_without_a_template_are_skipped(self, tmp_path):
        """A non-lock entry alongside a real plate is ignored, not fatal.

        This is how a ``"meta"`` sibling survives the bare-dict fallback above.
        Read the cost honestly: a genuine lock written *without* its
        ``template_m`` is skipped by the same branch and reported nowhere, so
        five plates can come back from a six-plate file and only the count
        would show it.  The all-junk case still raises — see below — which is
        what keeps the failure from being total silence.
        """
        path = write_locks(tmp_path, {
            "0": {"template_m": square_template()},
            "1": {"plate": 1, "note": "lock refused, no template"},
            "meta_like": ["not", "a", "lock"],
        })
        templates = pm.load_templates(path)
        assert sorted(templates) == [0]


class TestMalformedFiles:
    """The loader raises; it never returns an empty set of markers.

    Every test here exists for one reason, stated in ``load_templates``: an
    arm drawn with no markers looks exactly like an arm whose markers are
    correctly at the origin, so a silent empty result is undetectable in the
    only output this module has.
    """

    def test_wrong_shaped_template_raises_ValueError(self, tmp_path):
        """A three-marker template is refused, naming the file and the plate.

        Three markers is the realistic corruption — a dropped marker during
        the capture — and it is the one that would otherwise reach numpy as a
        ``(3, 3)`` array that broadcasts against a ``(4, 3)`` expectation
        without complaint in some code paths.
        """
        path = write_locks(tmp_path, {
            "0": {"template_m": [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0],
                                 [-0.05, 0.0, 0.0]]}})
        with pytest.raises(ValueError) as excinfo:
            pm.load_templates(path)
        message = str(excinfo.value)
        assert "expected (4, 3)" in message
        assert "(3, 3)" in message
        assert path in message
        assert "plate 0" in message

    def test_ragged_template_raises(self, tmp_path):
        """A row with two coordinates instead of three is refused.

        Numpy builds an object array from ragged rows, so the shape gate must
        catch it before ``np.linalg.norm`` does something surprising with it.
        """
        path = write_locks(tmp_path, {
            "0": {"template_m": [[0.05, 0.0, 0.0], [0.0, 0.05], [-0.05, 0.0, 0.0],
                                 [0.0, -0.05, 0.0]]}})
        with pytest.raises((ValueError, TypeError)):
            pm.load_templates(path)

    def test_non_finite_template_raises_ValueError(self, tmp_path):
        """A NaN coordinate is refused rather than drawn.

        A NaN reaches MuJoCo as a site position and is not necessarily an
        error there; it is a marker that vanishes or lands somewhere absurd.
        The finiteness gate is the only place it can be caught with the file
        name still in hand.
        """
        rows = square_template()
        rows[1][2] = float("nan")
        path = write_locks(tmp_path, {"0": {"template_m": rows}},
                           allow_nan=True)
        with pytest.raises(ValueError) as excinfo:
            pm.load_templates(path)
        assert "finite" in str(excinfo.value)

    def test_empty_plates_object_raises_ValueError(self, tmp_path):
        """``{"plates": {}}`` raises instead of returning ``{}``."""
        path = write_locks(tmp_path, {})
        with pytest.raises(ValueError) as excinfo:
            pm.load_templates(path)
        assert "no plate templates" in str(excinfo.value)
        assert path in str(excinfo.value)

    def test_document_with_no_plates_key_raises_ValueError(self, tmp_path):
        """A metadata-only file raises through the bare-dict fallback.

        Without the ``"plates"`` key the loader iterates the document itself,
        finds only a ``"meta"`` entry with no template, skips it, and must
        then refuse — the path by which a truncated or half-written capture
        would otherwise come back empty.
        """
        path = os.path.join(str(tmp_path), "meta_only.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"captured_wall": "2026-01-01T00:00:00"}}, fh)
        with pytest.raises(ValueError) as excinfo:
            pm.load_templates(path)
        assert "no plate templates" in str(excinfo.value)

    def test_missing_file_raises_OSError(self, tmp_path):
        """An absent locks file raises ``FileNotFoundError``.

        The realistic case is a fresh clone or a moved campaign directory.
        Failing to raise would mean the scene renders, silently, without any
        of the arm's markers — the exact picture that is supposed to be the
        check.
        """
        path = os.path.join(str(tmp_path), "does_not_exist.json")
        with pytest.raises(OSError):
            pm.load_templates(path)
        with pytest.raises(OSError):
            pm.template_meta(path)

    def test_unparseable_json_raises(self, tmp_path):
        """A truncated file raises ``JSONDecodeError``, not an empty dict."""
        path = os.path.join(str(tmp_path), "truncated.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"plates": {"0": {"template_m": [[0.0, 0.0,')
        with pytest.raises(json.JSONDecodeError):
            pm.load_templates(path)


# --------------------------------------------------------------------------
# 6. The one pure-math helper of the live path
# --------------------------------------------------------------------------


class TestEdgeDelta:
    """``_edge_delta_mm``: the pose-free comparison ``verify_against_live`` uses.

    The live check itself needs a receiver and is out of scope here, but the
    quantity it reports is pure geometry and is worth pinning: it is what
    tells a moved marker apart from a frame convention.
    """

    def test_identical_clouds_give_six_zeros(self, committed):
        """A cloud compared with itself changes no inter-marker distance."""
        out = pm._edge_delta_mm(committed[0], committed[0])
        assert len(out) == 6
        assert out == pytest.approx([0.0] * 6, abs=1e-9)

    def test_a_rigid_transform_changes_nothing(self, committed):
        """Rotating and translating the cloud leaves all six edges unchanged.

        This is the property the whole live check is built on: a frame
        convention cannot move an inter-marker distance, so anything it does
        report is a marker that physically moved.
        """
        t = committed[4]
        moved = t @ rodrigues([0.3, -0.7, 0.5], 1.1).T + np.array([1.0, -2.0, 3.0])
        out = pm._edge_delta_mm(t, moved)
        assert out == pytest.approx([0.0] * 6, abs=1e-9)

    def test_one_moved_marker_shows_in_its_three_edges(self, committed):
        """Pushing marker 0 out by 1 mm changes exactly its three edges.

        Six edges, three of which touch marker 0.  A displacement along its
        own radius changes those three by up to a millimetre and the opposite
        three by nothing, which is how the report localises the damage.
        """
        t = committed[2].copy()
        radial = t[0] / np.linalg.norm(t[0])
        moved = t.copy()
        moved[0] = t[0] + radial * 1e-3
        out = pm._edge_delta_mm(t, moved)
        touching, opposite = out[0:3], out[3:6]
        assert opposite == pytest.approx([0.0] * 3, abs=1e-9)
        assert max(abs(v) for v in touching) > 0.3
        assert max(abs(v) for v in touching) <= 1.0 + 1e-9
