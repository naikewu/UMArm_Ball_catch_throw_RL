"""Tests for the offline fkine benchmark.  Pure files + math: no sockets.

The strategy follows the design's own logic (``docs/fkine_design.md`` §4): the
benchmark's job is to recover geometry from campaign artifacts, so the tests
*synthesize a campaign from the port itself* — a known q trajectory pushed
through ``UMArm_KINEMATICS.fkine`` with a deliberately perturbed truth table —
write byte-format-faithful CSVs/locks, and assert the Gauss-Newton fit gets
the perturbation back:

1. **Noiseless recovery** (:class:`TestSyntheticCampaign`): truth = the CAD
   table with segment 2's span +4 mm; both regimes must recover every chain
   gap to < 0.5 mm (the task's bound; the noiseless fit actually lands within
   microns), the report files (md + JSON + PNGs) must exist, the verdict must
   read "parameters" (that is what a pure length perturbation *is*), the EE
   section must be the named skip (body 506 absent), and the regime-(a) base
   correction must come back ~0 (the synthetic streamed base is exact).

2. **Noisy recovery**: 0.3 mm Gaussian noise on every marker and every
   streamed pivot; the fit still recovers the perturbed span to < 0.5 mm —
   the averaging the design counts on, demonstrated.

3. **Refusals** (:class:`TestRefusals`): a second pass whose base moved 5 cm
   is refused (exit 2, design §5 merge fingerprint); a tampered locks.json is
   refused by the §4.1 validity binding *before* any number is produced.

Fixture CSVs are written in the campaign writers' exact formats (header,
column order, precision, ``tracked`` alphabet, the identity row for the
absent plate 6) so the benchmark's parsers are tested against the real
contract, not a convenient one.

Run:  ``python -m pytest UMArm_KINEMATICS/test_fkine_benchmark.py -q``
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

import numpy as np
import pytest

# Repo root on the path so both sibling packages import the same way pytest
# finds this file — from the root or from inside the package directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from UMArm_KINEMATICS import fkine_benchmark as fb  # noqa: E402
from UMArm_KINEMATICS import robot_params as rp  # noqa: E402
from UMArm_KINEMATICS.fkine import plate_transforms  # noqa: E402
from UMArm_MOCAP import marker_frame as mf  # noqa: E402
from UMArm_MOCAP.mocap_probe import matrix_to_quat_xyzw  # noqa: E402

#: Family angles in plate order (marker-frame design §2) — the synthetic
#: marker recipe's phi_p.
FAMILY_PHIS = (0.0, math.pi / 4, 0.0, math.pi / 4, 0.0, math.pi / 4)

#: Square bracket corners, CCW about +z (see test_marker_frame.rect_corners:
#: on a *square* the lock's x candidates land exactly on the bracket axes, so
#: synthetic round trips are machine-precision instead of carrying the
#: non-square lock-convention offset).
RECT = np.array([
    [0.03, 0.03, 0.0],
    [-0.03, 0.03, 0.0],
    [-0.03, -0.03, 0.0],
    [0.03, -0.03, 0.0],
])

#: The task's recovery bound: the deliberately perturbed span must come back
#: to better than half a millimetre.
RECOVER_TOL_M = 5e-4


def rot_z(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def markers_from_pose(pose: np.ndarray, phi: float) -> np.ndarray:
    """Design §7 recipe: ``markers = T_p @ Rz(phi_p) @ corners``."""
    r = pose[0:3, 0:3] @ rot_z(phi)
    return RECT @ r.T + pose[0:3, 3]


def truth_table() -> np.ndarray:
    """The CAD table with segment 2's LL (hence its span) +4 mm — the
    perturbation the fit must find.  A writable copy: DEFAULT_PARAMS is
    deliberately read-only."""
    t = np.array(rp.DEFAULT_PARAMS)
    t[1, rp.COL_LL] += 0.004
    return t


def base_pose() -> np.ndarray:
    """A structure-free synthetic mocap base: rotated and offset, so nothing
    accidentally aligns with world axes (the product of two Pythagorean
    rotations used across this repo's test suites)."""
    t = np.eye(4)
    t[0:3, 0:3] = np.array([
        [0.6, -0.224, 0.768],
        [0.8, 0.168, -0.576],
        [0.0, 0.96, 0.28],
    ])
    t[0:3, 3] = np.array([0.4, -0.2, 1.1])
    return t


def joint_trajectory(q_index: int, fps: float, rest_s: float, drive_s: float,
                     amp: float) -> np.ndarray:
    """One campaign joint: rest at q=0, then a half-sine sweep on one joint —
    the single-joint excitation shape of the real campaign (design §5)."""
    n_rest = int(round(rest_s * fps))
    n_drive = int(round(drive_s * fps))
    qs = np.zeros((n_rest + n_drive, 12))
    td = np.arange(n_drive) / fps
    qs[n_rest:, q_index] = amp * np.sin(math.pi * td / drive_s)
    return qs


def synth_pass(dirpath: str, joints=(0, 1, 4, 10), noise: float = 0.0,
               seed: int = 0, fps: float = 30.0, rest_s: float = 1.0,
               drive_s: float = 1.5, amp: float = 0.25,
               base: np.ndarray | None = None,
               plate5_offset_body=None) -> np.ndarray:
    """Write one synthetic campaign pass; returns the truth chain gaps.

    Everything is generated from the port itself under the perturbed truth
    table: mocap CSV u columns and poses CSV from ``plate_transforms`` (their
    origins ARE ``ujoint_centres``), markers from the §7 bracket recipe,
    locks from a rest window through the real ``compute_plate_lock``.  With
    ``noise`` > 0, iid Gaussian position noise lands on every streamed pivot
    and every marker — the recorded q stays the true q, as on hardware where
    it is recorded upstream of this tool.

    ``plate5_offset_body``: a deliberate model gap — every *measured*
    plate-5 datum (pivot, streamed pose, all four markers) is rigidly
    displaced by this body-frame vector, i.e. the physical bracket does not
    sit on the u-joint centre.  A lateral (x/y) offset is inexpressible by
    the five chain lengths, so the verdict must read "model".
    """
    rng = np.random.default_rng(seed)
    truth = truth_table()
    base = base_pose() if base is None else base
    p5_off = (np.zeros(3) if plate5_offset_body is None
              else np.asarray(plate5_offset_body, dtype=float))
    os.makedirs(dirpath, exist_ok=True)

    def jitter(shape):
        return rng.normal(0.0, noise, shape) if noise > 0.0 else np.zeros(shape)

    # Locks from a synthetic rest window (the probe's job on hardware), with
    # the same noise regime as the campaign so the §4.1 validity check sees
    # statistically identical data.
    plates0 = base @ plate_transforms(np.zeros(12), truth)
    plates0[5, 0:3, 3] += plates0[5, 0:3, 0:3] @ p5_off
    u_up = plates0[0][0:3, 3] - plates0[5][0:3, 3]
    locks = {}
    for p in range(6):
        stack = np.stack([markers_from_pose(plates0[p], FAMILY_PHIS[p])
                          + jitter((4, 3)) for _ in range(12)])
        locks[p] = mf.compute_plate_lock(stack, plate=p, u_up=u_up,
                                         streamed_rot=plates0[p][0:3, 0:3])
    with open(os.path.join(dirpath, "locks.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump({"meta": {"captured_wall": "synthetic", "regime": "labeled"},
                   "plates": {str(p): dataclasses.asdict(lk)
                              for p, lk in locks.items()}}, fh, indent=2)

    for j in joints:
        qs = joint_trajectory(j, fps, rest_s, drive_s, amp)
        n = qs.shape[0]
        mocap_rows, pose_rows, marker_rows = [], [], []
        for i in range(n):
            t_s = i / fps
            frame = j * 100000 + i
            plates = base @ plate_transforms(qs[i], truth)
            plates[5, 0:3, 3] += plates[5, 0:3, 0:3] @ p5_off
            u_noisy = plates[:, 0:3, 3] + jitter((6, 3))
            mocap_rows.append(
                f"{t_s:.4f},{frame},"
                + ",".join(f"{v:.6f}" for v in qs[i]) + ","
                + ",".join(f"{v:.6f}" for v in u_noisy.reshape(-1)) + ",1")
            for plate in range(7):
                if plate == 6:      # body 506 absent: the identity placeholder
                    pose_rows.append(f"{t_s:.6f},{frame},6,"
                                     "0.0000000,0.0000000,0.0000000,1.0000000,"
                                     "0.000000,0.000000,0.000000")
                    continue
                quat = matrix_to_quat_xyzw(plates[plate])
                pos = u_noisy[plate]
                pose_rows.append(
                    f"{t_s:.6f},{frame},{plate},"
                    + ",".join(f"{v:.7f}" for v in quat) + ","
                    + ",".join(f"{v:.6f}" for v in pos))
            for plate in range(6):
                mk = markers_from_pose(plates[plate], FAMILY_PHIS[plate]) \
                    + jitter((4, 3))
                for k in range(4):
                    marker_rows.append(
                        f"{t_s:.6f},{frame},0,{plate},{k},"
                        f"{mk[k, 0]:.6f},{mk[k, 1]:.6f},{mk[k, 2]:.6f},1")
        stem = os.path.join(dirpath, f"joint_{j:02d}")
        with open(stem + "_mocap.csv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("t_s,frame," + ",".join(f"q{i}_rad" for i in range(12))
                     + "," + ",".join(f"u{k}_{ax}" for k in range(6)
                                      for ax in "xyz") + ",chain_ok\n")
            fh.write("\n".join(mocap_rows) + "\n")
        with open(stem + "_poses.csv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("t_s,frame,plate,qx,qy,qz,qw,x,y,z\n")
            fh.write("\n".join(pose_rows) + "\n")
        with open(stem + "_markers.csv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("t_s,frame,epoch,plate,marker,x,y,z,tracked\n")
            fh.write("\n".join(marker_rows) + "\n")
        with open(stem + ".json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"q_index": j, "verdict": "MATCH"}, fh)

    return np.array(rp.plate_chain_m(truth))


def run_benchmark(dirs, out_dir, extra=()):
    argv = [str(d) for d in dirs] + ["--out-dir", str(out_dir)] + list(extra)
    return fb.main(argv)


# --------------------------------------------------------------------------
# Unit-level checks on the fit plumbing
# --------------------------------------------------------------------------


class TestFitPlumbing:
    def test_params_from_gaps_round_trip(self):
        """The gap->table map must invert plate_chain_m exactly: the fit's
        parameter vector and the report's length names are the same object."""
        gaps = np.array([0.221, 0.046, 0.198, 0.049, 0.181])
        params = fb.params_from_gaps(gaps)
        assert np.allclose(rp.plate_chain_m(params), gaps, atol=1e-15)
        # ...and the default gaps reproduce the default table's chain.
        assert rp.plate_chain_m(fb.params_from_gaps(
            np.array(rp.PLATE_CHAIN_NOMINAL_M))) == rp.PLATE_CHAIN_NOMINAL_M

    def test_params_from_gaps_never_touches_default(self):
        before = rp.DEFAULT_PARAMS.copy()
        fb.params_from_gaps(np.array([0.3, 0.1, 0.3, 0.1, 0.3]))
        assert np.array_equal(rp.DEFAULT_PARAMS, before)

    def test_exp_so3(self):
        r = fb.exp_so3(np.array([0.0, 0.0, math.pi / 2]))
        assert np.allclose(r, rot_z(math.pi / 2), atol=1e-15)
        assert np.allclose(fb.exp_so3(np.zeros(3)), np.eye(3), atol=1e-16)


# --------------------------------------------------------------------------
# The synthetic campaign: recovery + report artifacts
# --------------------------------------------------------------------------


class TestSyntheticCampaign:
    def _check_recovery(self, tmp_path, noise: float, seed: int):
        camp = tmp_path / "campaign"
        out = tmp_path / "report"
        truth_gaps = synth_pass(str(camp), noise=noise, seed=seed)
        assert run_benchmark([camp], out) == fb.EXIT_OK

        with open(out / "fkine_benchmark.json", "r", encoding="utf-8") as fh:
            report = json.load(fh)
        for regime in ("streamed", "inferred"):
            reg = report["regimes"][regime]
            assert reg["available"], regime
            fitted = np.asarray(reg["fit"]["gaps_m"])
            err = np.abs(fitted - truth_gaps)
            assert err.max() < RECOVER_TOL_M, (
                f"{regime}: gap errors {err * 1000.0} mm")
            # The perturbed gap (span2, index 2) moved 4 mm off CAD; the fit
            # must have *followed* it, not just stayed near the start point.
            assert abs(fitted[2] - rp.PLATE_CHAIN_NOMINAL_M[2]) > 0.003
        return report, out

    def test_noiseless_recovery_and_report(self, tmp_path):
        report, out = self._check_recovery(tmp_path, noise=0.0, seed=0)

        # Report files (design §4.6): markdown + JSON + the PNGs.
        assert (out / "fkine_benchmark.md").is_file()
        assert (out / "residuals_summary.png").is_file()
        assert (out / "lengths_fit.png").is_file()
        for j in (0, 1, 4, 10):
            assert (out / f"joint_{j:02d}_residuals.png").is_file()

        # A pure length perturbation IS a parameter issue — the verdict must
        # say so, and the 5 mm acceptance target is met post-fit.
        assert report["verdict"]["call"] == "parameters"
        assert report["verdict"]["acceptance_met"] is True

        # The synthetic streamed base is exact: the fitted regime-(a) SO(3)
        # correction must come back ~zero (quat CSV rounding is ~1e-7 rad).
        assert report["regimes"]["streamed"]["fit"]["rot_angle_deg"] < 0.05

        # EE step: skip-and-report while body 506 is absent (design §4.4).
        assert report["ee"]["status"] == "skipped"
        assert "506" in report["ee"]["why"]

        # q10/q11 substitute check: joint 10 was driven, all plates solved,
        # recorded q vs q_from_frames agree to marker-CSV rounding.
        assert report["q1011"]["status"] == "computed"
        assert max(report["q1011"]["rms_deg"]) < 0.01

        # Lock validity binding ran and verified every locked plate.
        assert report["passes"][0]["lock_check"]["verified"] == list(range(6))

        # Noiseless residuals collapse post-fit in both regimes.
        for regime in ("streamed", "inferred"):
            res = report["regimes"][regime]["residuals"]
            assert res["after"]["overall_rms_m"] < 5e-5
            assert res["before"]["overall_rms_m"] > 1e-3   # the 4 mm truth gap

        # The markdown names the verdict and the fitted-lengths section.
        md = (out / "fkine_benchmark.md").read_text(encoding="utf-8")
        assert "PARAMETERS" in md
        assert "Fitted lengths" in md
        assert "506" in md

    def test_noisy_recovery(self, tmp_path):
        """0.3 mm iid noise on every pivot and marker: averaging over the
        campaign still pins the perturbed span to < 0.5 mm, and the verdict
        stays 'parameters' (nothing structured was injected)."""
        report, _ = self._check_recovery(tmp_path, noise=3e-4, seed=7)
        assert report["verdict"]["call"] == "parameters"
        assert report["noise"]["measured"] is True
        # The measured floor should be in the ballpark of the injected noise
        # (norm of a 3-axis 0.3 mm sd is ~0.52 mm).
        floor = report["noise"]["rest_jitter_m_median"]
        assert 1e-4 < floor < 1.5e-3

    def test_model_gap_named(self, tmp_path):
        """The verdict must have teeth: a 10 mm *lateral* plate-5 bracket
        offset is a geometry the five chain lengths cannot express (at rest
        every length acts along the chain z), so it must survive the fit and
        flip the verdict to 'model' — this is the design §4.6 'structured
        residuals get reported, not absorbed' contract (D3)."""
        camp = tmp_path / "campaign"
        out = tmp_path / "report"
        synth_pass(str(camp), joints=(0, 8), plate5_offset_body=(0.010, 0.0, 0.0))
        assert run_benchmark([camp], out) == fb.EXIT_OK
        with open(out / "fkine_benchmark.json", "r", encoding="utf-8") as fh:
            report = json.load(fh)
        v = report["verdict"]
        assert v["call"] == "model"
        # The rest-visible bias detector names plate 5, and the explanation
        # says the gap is constant at rest — the lateral-offset signature.
        assert v["worst_plate"] == 5 or "plate 5" in v["explanation"]
        assert max(v["rest_bias_mm"].values()) > 5.0
        # Note the pooled acceptance number can still be < 5 mm here
        # (sqrt(10mm^2 / 5 plates) = 4.5 mm) — which is exactly why the
        # verdict judges structure, not the pooled RMS: the per-plate number
        # must expose the gap even when the pooled target reads "met".
        assert v["worst_plate_rms_mm"] > 5.0


# --------------------------------------------------------------------------
# Refusals: merge fingerprints and stale locks
# --------------------------------------------------------------------------


class TestRefusals:
    def test_merge_refusal_on_moved_base(self, tmp_path, capsys):
        """Design §5: two passes whose rest base positions disagree by 5 cm
        must not be merged — that is a moved rig, not more data."""
        d1, d2 = tmp_path / "pass1", tmp_path / "pass2"
        synth_pass(str(d1), joints=(0,))
        moved = base_pose()
        moved[0, 3] += 0.05
        synth_pass(str(d2), joints=(0,), base=moved)
        rc = run_benchmark([d1, d2], tmp_path / "report")
        assert rc == fb.EXIT_REFUSED
        err = capsys.readouterr().err
        assert "REFUSED" in err
        assert "base position" in err

    def test_stale_locks_refused(self, tmp_path, capsys):
        """Marker-frame §4.1 validity binding: a lock whose rigid marker
        template no longer registers onto this campaign's own rest markers is
        refused before any benchmark number exists (a re-created Motive asset
        must not be laundered into a residual table).  The tamper scales the
        template by 15 % — a ~6 mm registration RMS against the 3 mm bound."""
        camp = tmp_path / "campaign"
        synth_pass(str(camp), joints=(0,))
        locks_path = camp / "locks.json"
        with open(locks_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for plate in payload["plates"].values():
            plate["template_m"] = [[1.15 * v for v in row]
                                   for row in plate["template_m"]]
        with open(locks_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2)
        rc = run_benchmark([camp], tmp_path / "report")
        assert rc == fb.EXIT_REFUSED
        err = capsys.readouterr().err
        assert "stale" in err

    def test_missing_campaign_dir(self, tmp_path):
        assert run_benchmark([tmp_path / "nope"], tmp_path / "r") == fb.EXIT_NO_DATA


if __name__ == "__main__":                     # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
