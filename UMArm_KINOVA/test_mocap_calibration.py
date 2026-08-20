"""Synthetic-truth tests for the robot-world / hand-eye solve.

Every test builds a campaign from a KNOWN ``X`` and ``Y``, runs the solver, and
asks whether it came back.  No hardware, no kortex_api, no mocap — the point is
that the arithmetic is checked somewhere the answer is not in question, so a
disagreement on the rig is about the rig.

Run with the repo's ordinary interpreter::

    python -m pytest UMArm_KINOVA/test_mocap_calibration.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UMArm_KINOVA import mocap_calibration as MC  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: a plausible rig, and campaigns of varying richness
# --------------------------------------------------------------------------- #
#: A base parked 1.2 m from the volume origin and yawed 27 deg, and a marker
#: body 60 mm off the tool with a 40 deg twist.  Both are arbitrary; being
#: arbitrary is the point, since a solver that only works for small transforms
#: would pass on identity-ish truth.
X_TRUE = MC.se3(MC.rot_exp(np.deg2rad([3.0, -2.0, 27.0])), [-1.20, 0.35, 0.02])
Y_TRUE = MC.se3(MC.rot_exp(np.deg2rad([12.0, -35.0, 18.0])), [0.021, -0.008, 0.061])


def _tool_poses(translations, rotations_deg):
    """``T_base_tool`` for a grid of offsets and wrist orientations."""
    base_R = MC.rot_exp(np.deg2rad([180.0, 0.0, 90.0]))
    home = np.array([0.46, 0.02, 0.43])
    out = []
    for t in translations:
        for r in rotations_deg:
            out.append(MC.se3(base_R @ MC.rot_exp(np.deg2rad(r)), home + t))
    return np.array(out)


TRANSLATIONS = [(0, 0, 0), (0.07, 0, 0), (-0.07, 0, 0), (0, 0.07, 0),
                (0, -0.07, 0), (0, 0, 0.07), (0, 0, -0.07),
                (0.05, 0.05, 0.0), (-0.04, 0.03, -0.05)]
ROTATIONS = [(0, 0, 0), (12, 0, 0), (0, 12, 0), (0, 0, 12), (-10, 8, 0)]


def _observe(M, X=X_TRUE, Y=Y_TRUE, pos_noise_m=0.0, rot_noise_deg=0.0,
             seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for Mi in M:
        N = X @ Mi @ Y
        if pos_noise_m:
            N[0:3, 3] += rng.normal(0.0, pos_noise_m, 3)
        if rot_noise_deg:
            N[0:3, 0:3] = MC.rot_exp(
                np.deg2rad(rng.normal(0.0, rot_noise_deg, 3))) @ N[0:3, 0:3]
        out.append(N)
    return np.array(out)


# --------------------------------------------------------------------------- #
# The SE(3) helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deg", [0.0, 0.5, 30.0, 90.0, 179.0, 179.999])
def test_rot_log_exp_round_trip(deg):
    axis = np.array([0.3, -0.9, 0.31])
    axis = axis / np.linalg.norm(axis)
    R = MC.rot_exp(axis * np.deg2rad(deg))
    assert MC.angle_between_deg(R, MC.rot_exp(MC.rot_log(R))) < 1e-6


def test_rot_log_at_pi_is_a_real_axis():
    """The near-pi branch must return an axis, not a zero or a NaN."""
    for axis in np.eye(3):
        R = MC.rot_exp(axis * np.pi)
        w = MC.rot_log(R)
        assert np.isfinite(w).all()
        assert abs(np.linalg.norm(w) - np.pi) < 1e-6
        assert abs(abs(float(np.dot(w / np.pi, axis))) - 1.0) < 1e-6


def test_se3_inv_is_an_inverse():
    T = MC.se3(MC.rot_exp([0.3, -0.2, 1.1]), [0.4, -0.9, 0.2])
    assert np.allclose(T @ MC.se3_inv(T), np.eye(4), atol=1e-12)


def test_project_rotation_rejects_a_reflection():
    R = MC.project_rotation(np.diag([1.0, 1.0, -1.0]))
    assert np.linalg.det(R) > 0


# --------------------------------------------------------------------------- #
# The solve, on noiseless data
# --------------------------------------------------------------------------- #
def test_exact_recovery_without_noise():
    M = _tool_poses(TRANSLATIONS, ROTATIONS)
    N = _observe(M)
    res = MC.fit(M, N)
    assert MC.angle_between_deg(res["X_world_base"][0:3, 0:3],
                                X_TRUE[0:3, 0:3]) < 1e-6
    assert MC.angle_between_deg(res["Y_tool_rb"][0:3, 0:3],
                                Y_TRUE[0:3, 0:3]) < 1e-6
    assert np.linalg.norm(res["X_world_base"][0:3, 3] - X_TRUE[0:3, 3]) < 1e-9
    assert np.linalg.norm(res["Y_tool_rb"][0:3, 3] - Y_TRUE[0:3, 3]) < 1e-9
    assert res["residual"]["pos_rms_mm"] < 1e-6
    assert res["observability"]["ok"]


def test_closed_form_alone_is_already_close():
    """The polish must be a refinement, not the thing that makes it work."""
    M = _tool_poses(TRANSLATIONS, ROTATIONS)
    N = _observe(M)
    X0, Y0, _ = MC.closed_form_fit(M, N)
    assert MC.angle_between_deg(X0[0:3, 0:3], X_TRUE[0:3, 0:3]) < 1e-6
    assert np.linalg.norm(Y0[0:3, 3] - Y_TRUE[0:3, 3]) < 1e-9


def test_three_samples_suffice_when_they_rotate():
    M = _tool_poses([(0, 0, 0), (0.06, 0.0, 0.0), (0.0, 0.05, -0.03)],
                    [(0, 0, 0)])
    M = np.array([M[0], M[1] @ MC.se3(MC.rot_exp(np.deg2rad([15, 0, 0])), [0, 0, 0]),
                  M[2] @ MC.se3(MC.rot_exp(np.deg2rad([0, 14, 0])), [0, 0, 0])])
    N = _observe(M)
    res = MC.fit(M, N)
    assert res["residual"]["pos_rms_mm"] < 1e-6


def test_fewer_than_three_samples_refuses():
    M = _tool_poses([(0, 0, 0), (0.05, 0, 0)], [(0, 0, 0)])
    with pytest.raises(ValueError):
        MC.fit(M, _observe(M))


def test_mismatched_stacks_refuse():
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    with pytest.raises(ValueError):
        MC.fit(M, _observe(M)[:-1])


# --------------------------------------------------------------------------- #
# Observability: the thing a good residual cannot tell you
# --------------------------------------------------------------------------- #
def test_pure_translation_is_reported_as_unobservable():
    """No rotation anywhere: Ry is arbitrary and the report must say so."""
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    N = _observe(M)
    obs = MC.observability(M, N)
    assert not obs["ok"]
    assert "unobservable" in " ".join(obs["notes"])
    res = MC.fit(M, N)
    # It still fits the DATA beautifully — which is exactly the trap.
    assert res["residual"]["pos_rms_mm"] < 1e-6
    assert not res["observability"]["ok"]
    assert "refusal" in res["closed_form_info"]


def test_single_rotation_axis_is_flagged():
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0), (10, 0, 0), (20, 0, 0)])
    obs = MC.observability(M, _observe(M))
    assert not obs["ok"]
    assert "one axis" in " ".join(obs["notes"])


def test_tiny_workspace_is_flagged():
    M = _tool_poses([(0, 0, 0), (0.004, 0, 0), (0, 0.003, 0)], ROTATIONS)
    obs = MC.observability(M, _observe(M))
    assert any("leverage" in n for n in obs["notes"])


# --------------------------------------------------------------------------- #
# Noise: the residual must track the noise that was put in
# --------------------------------------------------------------------------- #
def test_residual_tracks_injected_noise():
    M = _tool_poses(TRANSLATIONS, ROTATIONS)
    quiet = MC.fit(M, _observe(M, pos_noise_m=0.0002, rot_noise_deg=0.05,
                               seed=1))
    loud = MC.fit(M, _observe(M, pos_noise_m=0.0020, rot_noise_deg=0.50,
                              seed=1))
    assert quiet["residual"]["pos_rms_mm"] < 0.5
    assert loud["residual"]["pos_rms_mm"] > 3.0 * quiet["residual"]["pos_rms_mm"]
    # and the truth is still recovered to well inside the noise
    assert np.linalg.norm(quiet["Y_tool_rb"][0:3, 3] - Y_TRUE[0:3, 3]) < 0.002


def test_refine_never_makes_it_worse():
    M = _tool_poses(TRANSLATIONS, ROTATIONS)
    N = _observe(M, pos_noise_m=0.0005, rot_noise_deg=0.2, seed=7)
    res = MC.fit(M, N)
    assert res["residual"]["pos_rms_mm"] <= \
        res["residual_closed_form"]["pos_rms_mm"] + 1e-9


# --------------------------------------------------------------------------- #
# The operator's own question
# --------------------------------------------------------------------------- #
def test_axis_fidelity_is_perfect_on_perfect_data():
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    N = _observe(M)
    out = MC.axis_fidelity(M, N, X_TRUE[0:3, 0:3])
    assert out["n_pairs"] > 10
    assert abs(out["scale_mean"] - 1.0) < 1e-9
    assert out["angle_max_deg"] < 1e-4


def test_axis_fidelity_catches_a_scale_error():
    """A volume calibrated 1 % small must show up as a 1 % scale."""
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    N = _observe(M)
    centre = N[:, 0:3, 3].mean(axis=0)
    N[:, 0:3, 3] = centre + 1.01 * (N[:, 0:3, 3] - centre)
    out = MC.axis_fidelity(M, N, X_TRUE[0:3, 0:3])
    assert abs(out["scale_mean"] - 1.01) < 1e-6


def test_axis_fidelity_skips_pairs_that_rotated():
    M = _tool_poses([(0, 0, 0), (0.07, 0, 0)], [(0, 0, 0), (15, 0, 0)])
    out = MC.axis_fidelity(M, _observe(M), X_TRUE[0:3, 0:3])
    # four poses, six pairs, but only the two same-orientation pairs qualify
    assert out["n_pairs"] == 2


# --------------------------------------------------------------------------- #
# Do we need a marker body on the base?
# --------------------------------------------------------------------------- #
def test_single_frame_base_solve_is_exact_without_noise():
    M = _tool_poses(TRANSLATIONS, ROTATIONS)
    N = _observe(M)
    for i in (0, 5, 20):
        Xi = MC.base_from_single_sample(M[i], N[i], Y_TRUE)
        assert np.linalg.norm(Xi - X_TRUE) < 1e-9


def test_base_recovery_spread_grows_with_noise():
    M = _tool_poses(TRANSLATIONS, ROTATIONS)
    quiet = MC.fit(M, _observe(M, pos_noise_m=0.0002, seed=3))
    loud = MC.fit(M, _observe(M, pos_noise_m=0.0020, seed=3))
    sq = MC.base_recovery_spread(M, _observe(M, pos_noise_m=0.0002, seed=3),
                                 quiet["Y_tool_rb"], quiet["X_world_base"])
    sl = MC.base_recovery_spread(M, _observe(M, pos_noise_m=0.0020, seed=3),
                                 loud["Y_tool_rb"], loud["X_world_base"])
    assert sl["pos_rms_mm"] > 3.0 * sq["pos_rms_mm"]

# --------------------------------------------------------------------------- #
# The refusals added after the 2026-08-19 review
# --------------------------------------------------------------------------- #
def test_one_rotation_axis_is_REFUSED_not_answered():
    """Two turns of one wrist joint are two rotations and ONE axis.

    The refusal used to count rotations while its message promised "at least
    two different axes", so a campaign that turned only one joint got a
    confident answer from a rank-deficient cross-covariance: one arbitrary
    member of a one-parameter family, with nothing to say it was.
    """
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0), (10, 0, 0), (20, 0, 0), (-14, 0, 0)])
    Ry, info = MC.hand_eye_rotation(M, _observe(M))
    assert info["pairs_used"] > 10, "the count test alone would have passed"
    assert info["axis_spread_deg"] < MC.MIN_AXIS_SPREAD_DEG
    assert "refusal" in info
    assert "one axis" in info["refusal"]
    assert np.allclose(Ry, np.eye(3)), "a refusal must not return a guess"


def test_two_axes_are_enough_and_are_not_refused():
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0), (12, 0, 0), (0, 12, 0)])
    Ry, info = MC.hand_eye_rotation(M, _observe(M))
    assert "refusal" not in info
    assert info["axis_spread_deg"] > MC.MIN_AXIS_SPREAD_DEG
    assert MC.angle_between_deg(Ry, Y_TRUE[0:3, 0:3]) < 1e-6


def test_axis_spread_treats_a_rotation_axis_as_a_LINE():
    """+12 and -12 deg about one joint are the same axis, not opposite ones.

    Comparing signed directions would call that pair 180 deg apart and pronounce
    a single-axis campaign richly determined, which is exactly backwards.
    """
    assert MC.axis_spread_deg([[0, 0, 1], [0, 0, -1]]) == pytest.approx(0.0,
                                                                       abs=1e-9)
    assert MC.axis_spread_deg([[0, 0, 1], [0, 1, 0]]) == pytest.approx(90.0,
                                                                       abs=1e-9)


def test_a_frozen_rigid_body_is_skipped_not_scored_as_perfect():
    """``max(-1, min(1, nan))`` is +1.0 in Python, so 0/0 read as a perfect match.

    The failure it hides is the one that matters most: the arm moved, the
    cameras did not, and the report said the directions agreed exactly.
    """
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    N = _observe(M)
    N[3][0:3, 3] = N[2][0:3, 3]          # two poses, one measured position
    out = MC.axis_fidelity(M, N, X_TRUE[0:3, 0:3])
    assert all(not (r["i"] == 2 and r["j"] == 3) for r in out["pairs"])
    assert np.isfinite(out["angle_max_deg"])
    assert np.isfinite(out["scale_mean"])


def test_per_axis_scale_carries_its_own_uncertainty():
    """A spread of three correlated means is not a finding until it beats noise."""
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    clean = MC.axis_fidelity_by_axis(M, _observe(M), X_TRUE[0:3, 0:3])
    assert clean["spread_ppm"] < 1.0
    assert not clean["spread_is_significant"]

    noisy = MC.axis_fidelity_by_axis(
        M, _observe(M, pos_noise_m=5e-4, seed=5), X_TRUE[0:3, 0:3])
    assert noisy["per_axis_uncertainty_ppm"] > 0.0
    # whatever the draw, the verdict must be consistent with its own numbers
    assert noisy["spread_is_significant"] == (
        noisy["spread_ppm"] > noisy["significance_threshold_ppm"])
    assert noisy["significance_threshold_ppm"] >= MC.SCALE_SIGNIFICANCE_FLOOR_PPM


def test_a_real_per_axis_distortion_IS_significant():
    """Stretch one axis by 1 % and the verdict has to change."""
    M = _tool_poses(TRANSLATIONS, [(0, 0, 0)])
    N = _observe(M)
    centre = N[:, 0:3, 3].mean(axis=0)
    # stretch along the WORLD image of the base x axis, so exactly one of the
    # three per-axis groups moves
    axis = X_TRUE[0:3, 0:3] @ np.array([1.0, 0.0, 0.0])
    for i in range(N.shape[0]):
        d = N[i, 0:3, 3] - centre
        N[i, 0:3, 3] = centre + d + 0.01 * float(np.dot(d, axis)) * axis
    out = MC.axis_fidelity_by_axis(M, N, X_TRUE[0:3, 0:3])
    assert out["spread_ppm"] > 5000.0
    assert out["spread_is_significant"]


def test_observability_reports_a_radius_and_says_so():
    M = _tool_poses([(0, 0, 0), (0.07, 0, 0), (-0.07, 0, 0)], ROTATIONS)
    obs = MC.observability(M, _observe(M))
    assert "tool_radius_m" in obs and "tool_span_m" not in obs
    # three collinear points at +-70 mm: the RADIUS is 70 mm, not the 140 mm span
    assert obs["tool_radius_m"] == pytest.approx(0.070, abs=1e-9)
