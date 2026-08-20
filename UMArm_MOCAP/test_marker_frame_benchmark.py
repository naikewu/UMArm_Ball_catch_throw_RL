"""Tests for the offline marker-frame benchmark (design §8).  No sockets.

The fixture is a **tiny synthetic campaign written through the exact CSV
schemas** the real writers emit (``mocap_probe.write_capture_csvs`` /
``joint_verification._write_marker_files``): markers at %.6f, quaternions
scalar-last at %.7f, the same headers, the same ``tracked`` alphabet — so the
benchmark's parsers are exercised against the byte format they will meet on
the bench, quantization included.  Geometry comes from
``UMArm_KINEMATICS.plate_transforms`` + the design §7 marker recipe
(``markers = T_p @ Rz(phi_p) @ corners``, square corners so solves are exact
up to the CSV's ~1e-6 m quantization — see test_marker_frame.rect_corners for
why squares).

What is pinned, per the stage brief:

* the tool ingests a synthetic campaign end to end and writes markdown +
  JSON + PNGs (exit 0, model verification PASS);
* a stale lock (minted from a differently-sized bracket) is **refused** with
  exit 2 and no report written — the §4.1 validity binding;
* drop-one stats exist per plate per marker and are quantization-small on
  square noiseless rectangles;
* the per-joint flag-regime keying of the dropout section (a labeled joint
  reports its real dropout; regime string present);
* a campaign whose recording drives a different joint than its filename
  claims **fails model verification loudly** (exit 3, FAIL in the markdown) —
  design D7's "the model, not the report, is wrong";
* probe-only ingestion works (sections 5/6-axis degrade to notes, exit 0).

Run:  ``python -m pytest UMArm_MOCAP/test_marker_frame_benchmark.py -q``
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import marker_frame as mf  # noqa: E402
import marker_frame_benchmark as mfb  # noqa: E402
from mocap_probe import matrix_to_quat_xyzw  # noqa: E402
from UMArm_KINEMATICS.fkine import plate_transforms  # noqa: E402

FAMILY_PHIS = np.array([0.0, math.pi / 4, 0.0, math.pi / 4, 0.0, math.pi / 4])


# --------------------------------------------------------------------------
# Synthetic-campaign builder (the design §7 recipe, written as CSV text)
# --------------------------------------------------------------------------


def rot_z(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rect_corners(w: float = 0.03) -> np.ndarray:
    """Square bracket corners, CCW about +z (see test_marker_frame for why
    exact tests need squares: non-squares quantize the locked x)."""
    return np.array([[w, w, 0.0], [-w, w, 0.0], [-w, -w, 0.0], [w, -w, 0.0]])


def markers_from_pose(pose: np.ndarray, phi: float) -> np.ndarray:
    r = pose[0:3, 0:3] @ rot_z(phi)
    return rect_corners() @ r.T + pose[0:3, 3]


#: A structure-free base: rotation about z plus an off-origin translation, so
#: nothing accidentally aligns with world axes.
BASE = np.eye(4)
BASE[0:3, 0:3] = rot_z(0.3)
BASE[0:3, 3] = [0.5, -0.2, 1.0]

DT_S = 0.05                      # 20 Hz synthetic rate, compact but plural


def make_locks(scale: float = 1.0) -> dict[int, mf.PlateLock]:
    """Locks from the q=0 rest pose, u_up arm-derived exactly as the probe
    computes it.  ``scale`` != 1 mints locks for a *differently sized*
    bracket — the stale-lock fixture."""
    plates = BASE @ plate_transforms(np.zeros(12))
    u_up = plates[0][0:3, 3] - plates[5][0:3, 3]
    locks = {}
    for p in range(6):
        markers = (rect_corners() * scale) @ (
            plates[p][0:3, 0:3] @ rot_z(FAMILY_PHIS[p])).T + plates[p][0:3, 3]
        locks[p] = mf.compute_plate_lock(
            np.stack([markers] * 3), plate=p, u_up=u_up,
            streamed_rot=plates[p][0:3, 0:3])
    return locks


def write_locks_json(path: str, locks: dict) -> None:
    payload = {"meta": {"captured_wall": "2026-08-11T12:00:00",
                        "regime": "labeled", "mapping_epoch": 0},
               "plates": {str(p): dataclasses.asdict(lock)
                          for p, lock in locks.items()}}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def q_trajectory(drive_channel: int, n_rest: int, n_drive: int,
                 amp: float = 0.3) -> list[np.ndarray]:
    qs = [np.zeros(12) for _ in range(n_rest)]
    for i in range(n_drive):
        q = np.zeros(12)
        q[drive_channel] = amp * (i + 1) / n_drive
        qs.append(q)
    return qs


def write_recording(dirpath: str, stem: str, qs: list[np.ndarray],
                    with_mocap: bool, drop: tuple[int, int, int] | None = None,
                    placeholder_frame: int | None = None,
                    twist: tuple[tuple[int, ...], int, float] | None = None) -> None:
    """One recording (probe or joint) in the writers' exact byte formats.

    ``drop`` = (frame_index, plate, marker): that marker's tracked flag is
    written 0 (a real labeled dropout; the position stays honest, as a
    visible-but-excluded marker's would).  ``placeholder_frame`` inserts one
    markers-absent frame (the ``-1,-1,nan,nan,nan,-`` row) to exercise the
    parser against the design's "not streamed" regime.  ``twist`` =
    (plates, channel, ratio): those plates' MARKER rectangles get an extra
    rotation of ``ratio * q[channel]`` about their own normal while their
    origins (and the streamed poses) stay put — the orientation-disagrees-
    with-positions fault the model-verification gate exists to catch; pass
    a whole co-rigid pair so only the axis check fires.
    """
    m_lines = ["t_s,frame,epoch,plate,marker,x,y,z,tracked"]
    p_lines = ["t_s,frame,plate,qx,qy,qz,qw,x,y,z"]
    q_lines = (["t_s,frame," + ",".join(f"q{i}_rad" for i in range(12)) + ","
                + ",".join(f"u{k}_{ax}" for k in range(6) for ax in "xyz")
                + ",chain_ok"] if with_mocap else None)
    for k, q in enumerate(qs):
        t_s = k * DT_S
        plates = BASE @ plate_transforms(q)
        if placeholder_frame is not None and k == placeholder_frame:
            m_lines.append(f"{t_s:.6f},{k},0,-1,-1,nan,nan,nan,-")
        else:
            for p in range(6):
                extra = 0.0
                if twist is not None and p in twist[0]:
                    extra = twist[2] * float(q[twist[1]])
                markers = markers_from_pose(plates[p], FAMILY_PHIS[p] + extra)
                for j in range(4):
                    tracked = "1"
                    if drop is not None and (k, p, j) == drop:
                        tracked = "0"
                    m_lines.append(
                        f"{t_s:.6f},{k},0,{p},{j},{markers[j, 0]:.6f},"
                        f"{markers[j, 1]:.6f},{markers[j, 2]:.6f},{tracked}")
        for p in range(7):
            T = plates[p] if p < 6 else np.eye(4)   # plate 6 absent (D6)
            quat = matrix_to_quat_xyzw(T)
            pos = T[0:3, 3]
            p_lines.append(f"{t_s:.6f},{k},{p},{quat[0]:.7f},{quat[1]:.7f},"
                           f"{quat[2]:.7f},{quat[3]:.7f},{pos[0]:.6f},"
                           f"{pos[1]:.6f},{pos[2]:.6f}")
        if q_lines is not None:
            u = plates[0:6, 0:3, 3].reshape(-1)
            q_lines.append(f"{t_s:.4f},{k},"
                           + ",".join(f"{v:.6f}" for v in q) + ","
                           + ",".join(f"{v:.6f}" for v in u) + ",1")
    with open(os.path.join(dirpath, f"{stem}_markers.csv"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(m_lines) + "\n")
    with open(os.path.join(dirpath, f"{stem}_poses.csv"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(p_lines) + "\n")
    if q_lines is not None:
        with open(os.path.join(dirpath, f"{stem}_mocap.csv"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(q_lines) + "\n")


def build_campaign(tmp_path, joint_name: str = "joint_00",
                   drive_channel: int = 0, lock_scale: float = 1.0,
                   drop: tuple[int, int, int] | None = (10, 2, 1)):
    """Campaign dir + locks path.  Rest = 6 frames (0..0.25 s), drive = 12
    frames ramping the channel to 0.3 rad; ``--rest-seconds 0.28`` in the
    runs below keeps the drive out of the rest window."""
    camp = tmp_path / "campaign"
    camp.mkdir()
    qs = q_trajectory(drive_channel, n_rest=6, n_drive=12)
    write_recording(str(camp), joint_name, qs, with_mocap=True, drop=drop)
    locks_path = tmp_path / "locks.json"
    write_locks_json(str(locks_path), make_locks(scale=lock_scale))
    return str(camp), str(locks_path)


REST_ARG = ["--rest-seconds", "0.28"]


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------


class TestCampaignIngestion:
    def test_end_to_end_reports_written(self, tmp_path):
        """Exit 0, all report files exist (default out dir under the
        campaign), all six sections present, model verification PASS."""
        camp, locks = build_campaign(tmp_path)
        rc = mfb.main(["--locks", locks, "--campaign", camp] + REST_ARG)
        assert rc == mfb.EXIT_OK
        out = os.path.join(camp, "marker_frame_benchmark")
        assert os.path.exists(os.path.join(out, mfb.MD_NAME))
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        for sec in ("rectangle_validity", "static_precision",
                    "alignment_error", "drop_one", "dynamic_consistency",
                    "model_verification"):
            assert sec in report["sections"], sec
        assert report["sections"]["model_verification"]["verdict"] == "PASS"
        for png in ("static_precision.png", "alignment_error.png",
                    "drop_one.png", "dynamic_joint_00.png",
                    "model_verification.png"):
            assert png in report["plots"]
            assert os.path.getsize(os.path.join(out, png)) > 0

    def test_numbers_are_quantization_small_on_truth_data(self, tmp_path):
        """Streamed == truth here, so every 'error' the benchmark reports is
        the CSV quantization floor (~1e-6 m, ~1e-5 deg): alignment constants
        near zero, jitter near zero, dq near zero."""
        camp, locks = build_campaign(tmp_path)
        out = str(tmp_path / "out")
        rc = mfb.main(["--locks", locks, "--campaign", camp,
                       "--out-dir", out] + REST_ARG)
        assert rc == mfb.EXIT_OK
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        align = report["sections"]["alignment_error"]["plates"]
        for p in range(6):
            assert align[str(p)]["angle_deg"] < 0.1
            assert align[str(p)]["origin_offset_mm"]["mean"] < 0.05
        static = report["sections"]["static_precision"]["plates"]
        for p in range(6):
            assert static[str(p)]["inferred"]["origin_rms_mm"] < 0.05
            assert static[str(p)]["inferred"]["axis_rms_deg"] < 0.05
        dyn = report["sections"]["dynamic_consistency"]["joints"]["joint_00"]
        assert dyn["dq_pivot"]["driven_max_rad"] < 1e-3
        assert dyn["dq_frames"]["driven_max_rad"] < 1e-3
        rect = report["sections"]["rectangle_validity"]["plates"]
        for p in range(6):
            assert rect[str(p)]["rms_residual_mm"]["median"] < 0.01

    def test_dropout_section_is_keyed_per_joint_regime(self, tmp_path):
        """The labeled regime and the one real dropout (frame 10, plate 2,
        marker 1) are reported per joint (design §5 / review ops-9)."""
        camp, locks = build_campaign(tmp_path, drop=(10, 2, 1))
        out = str(tmp_path / "out")
        assert mfb.main(["--locks", locks, "--campaign", camp,
                         "--out-dir", out] + REST_ARG) == mfb.EXIT_OK
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        jrec = report["sections"]["drop_one"]["per_joint"]["joint_00"]
        assert jrec["regime"] == "labeled"
        assert jrec["frames_labeled"] == jrec["frames_with_sets"]
        plate2 = jrec["plates"]["2"]
        assert plate2["real_dropout_frames"] == 1
        assert plate2["solved_3marker"] == 1
        # The 3-marker registration residual sits at the CSV quantization
        # floor on noiseless synthetic data.
        assert plate2["rms_residual_mm_median"] < 0.01


class TestDropOneStats:
    def test_synthetic_drop_one_is_exact_on_squares(self, tmp_path):
        """Every 3-marker solve equals the 4-marker solve on a noiseless
        square bracket, so the pooled stats sit at the CSV quantization
        floor -- per plate, per marker, with nothing gated."""
        camp, locks = build_campaign(tmp_path, drop=None)
        out = str(tmp_path / "out")
        assert mfb.main(["--locks", locks, "--campaign", camp,
                         "--out-dir", out] + REST_ARG) == mfb.EXIT_OK
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        synth = report["sections"]["drop_one"]["synthetic"]
        for p in range(6):
            rec = synth[str(p)]
            assert rec["frames_dropped_from"] == 18       # 6 rest + 12 drive
            for j in range(4):
                mrec = rec["markers"][str(j)]
                assert mrec["n"] == 18 and mrec["gated"] == 0
                assert mrec["d_origin_mm"]["median"] < 0.05
                assert mrec["d_axis_deg"]["median"] < 0.05


class TestLockValidity:
    def test_stale_lock_is_refused(self, tmp_path):
        """Locks minted from a 15 %-bigger bracket: the re-derived diagonal
        lengths disagree by ~13 mm >> the 3 mm binding tolerance, so the run
        refuses (exit 2) and writes no report (design §4.1 / review ops-5)."""
        camp, locks = build_campaign(tmp_path, lock_scale=1.15)
        out = str(tmp_path / "out")
        rc = mfb.main(["--locks", locks, "--campaign", camp,
                       "--out-dir", out] + REST_ARG)
        assert rc == mfb.EXIT_STALE_LOCK
        assert not os.path.exists(os.path.join(out, mfb.MD_NAME))
        assert not os.path.exists(os.path.join(out, mfb.JSON_NAME))


class TestModelVerification:
    def test_wrong_driven_joint_is_a_map_error_not_a_model_error(self, tmp_path):
        """A recording that claims to drive q0 (filename joint_00) but whose
        frames rotate about y (q1) is a JOINT-MAP error -- the campaign's own
        argmax verdict catches it -- not a hardware-model error: the motion is
        perfectly expressible by the pair's axes, so the cross-talk-aware
        gate PASSES while the naive single-axis number (reported, not gated)
        shows the full 90 deg.  This pins the semantic split the 2026-08-11
        hardware run forced: gravity moves the pair's other angles during a
        real single-agonist drive (q0 cross-talk 0.53), and a gate that
        cannot tell physics from model error falsifies correct hardware."""
        camp, locks = build_campaign(tmp_path, joint_name="joint_00",
                                     drive_channel=1, drop=None)
        out = str(tmp_path / "out")
        rc = mfb.main(["--locks", locks, "--campaign", camp,
                       "--out-dir", out] + REST_ARG)
        assert rc == mfb.EXIT_OK
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        model = report["sections"]["model_verification"]
        assert model["verdict"] == "PASS"
        axis = [e for e in model["axis"] if e["pass"] is not None]
        assert len(axis) == 1
        assert axis[0]["single_axis_error_deg"] == pytest.approx(90.0, abs=1.0)
        assert axis[0]["axis_error_deg"] < 1.0

    def test_orientation_position_inconsistency_fails_loudly(self, tmp_path):
        """The inexpressible case: plate 1's ORIENTATION twists about its own
        normal while the pair's ORIGINS say a pure x-axis (q0) swing.  No
        combination of the pair's four in-plane revolute axes reproduces
        that, so the model reconstruction diverges from the measured delta:
        FAIL, exit 3, named in the markdown -- reports still written (the
        failure IS the finding, design D7)."""
        camp = tmp_path / "campaign"
        camp.mkdir()
        qs = q_trajectory(0, n_rest=6, n_drive=12)
        write_recording(str(camp), "joint_00", qs, with_mocap=True,
                        twist=((1, 2), 0, 0.5))
        locks_path = tmp_path / "locks.json"
        write_locks_json(str(locks_path), make_locks())
        out = str(tmp_path / "out")
        rc = mfb.main(["--locks", str(locks_path), "--campaign", str(camp),
                       "--out-dir", out] + REST_ARG)
        assert rc == mfb.EXIT_MODEL_FALSIFIED
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        model = report["sections"]["model_verification"]
        assert model["verdict"] == "FAIL"
        axis = [e for e in model["axis"] if e["pass"] is not None]
        assert len(axis) == 1
        assert axis[0]["axis_error_deg"] > mfb.AXIS_TOL_DEG
        assert model["failures"]
        with open(os.path.join(out, mfb.MD_NAME), encoding="utf-8") as fh:
            md = fh.read()
        assert "FAIL" in md and "MODEL FALSIFIED" in md
        # Co-rigid constancy is still fine -- only the axis check fired
        # (the twist is between plates 0 and 1; pair {1,2} twists together).
        assert all(e["pass"] for e in model["corigid"]
                   if e["pass"] is not None)


class TestProbeOnly:
    def test_probe_static_capture_ingests(self, tmp_path):
        """Probe-only run: static sections computed, dynamic section is an
        explicit note (no drives), placeholder markers-absent frame parsed,
        exit 0."""
        probe = tmp_path / "probe"
        probe.mkdir()
        qs = [np.zeros(12) for _ in range(10)]
        write_recording(str(probe), "probe", qs, with_mocap=False,
                        placeholder_frame=4)
        locks_path = tmp_path / "locks.json"
        write_locks_json(str(locks_path), make_locks())
        out = str(tmp_path / "out")
        rc = mfb.main(["--locks", str(locks_path), "--probe", str(probe),
                       "--out-dir", out])
        assert rc == mfb.EXIT_OK
        with open(os.path.join(out, mfb.JSON_NAME), encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["sections"]["dynamic_consistency"]["joints"] == {}
        src = report["meta"]["sources"][0]
        assert src["kind"] == "probe"
        # 10 frames, one of them the markers-absent placeholder.
        assert src["frames"] == 10 and src["frames_with_sets"] == 9
        static = report["sections"]["static_precision"]["plates"]
        assert static["0"]["inferred"]["origin_rms_mm"] is not None
        # Co-rigid pairs evaluate on a static window too (trivially constant).
        cor = report["sections"]["model_verification"]["corigid"]
        assert any(e["pass"] for e in cor if e["pass"] is not None)

    def test_no_inputs_is_a_usage_error(self, tmp_path):
        locks_path = tmp_path / "locks.json"
        write_locks_json(str(locks_path), make_locks())
        assert mfb.main(["--locks", str(locks_path)]) == mfb.EXIT_USAGE
