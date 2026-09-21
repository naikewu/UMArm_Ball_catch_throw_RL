"""Pure-NumPy task specs for UMARM catch variants.

This module deliberately avoids MuJoCo, JAX, and MJX imports so catch task
distributions can be reused by feasibility checks, previews, and later training
without changing the existing strike task path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Range = tuple[float, float]
Vec3 = tuple[float, float, float]

GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class CatchTaskSpec:
    name: str
    notes: str
    modes: tuple[str, ...]
    mode_probabilities: tuple[float, ...]
    intercept_x_range_m: Range
    intercept_y_range_m: Range
    intercept_z_range_m: Range
    lob_intercept_time_s: Range
    flat_intercept_time_s: Range
    mixed_intercept_time_s: Range
    max_resample_attempts: int = 128
    # Lob-mode launch envelope (defaults preserve the historical hardcoded box). Arrival
    # DIRECTION at the glove is launch-determined (vy is ballistically invariant), and the
    # glove-orientation gate only accepts arrivals reasonably far from vertical.
    lob_launch_x_m: Range = (0.03, 0.09)
    lob_launch_y_m: Range = (-0.24, -0.16)
    lob_launch_z_m: Range = (0.26, 0.34)
    lob_launch_vx_mps: Range = (-0.05, 0.05)
    lob_launch_vy_mps: Range = (0.30, 0.50)
    lob_launch_vz_mps: Range = (1.5, 2.3)


CATCH_DIVERSE = CatchTaskSpec(
    name="catch_diverse",
    notes=(
        "Catch-and-throw construction distribution mixing FEASIBLE_V2-like "
        "upward lobs, flat fast line drives, and intermediate trajectories. "
        "Samples are generated from physically consistent ballistics into the "
        "arm vicinity and include an explicit intercept state."
    ),
    modes=("lob_arc", "flat_fast", "mixed"),
    mode_probabilities=(0.40, 0.30, 0.30),
    intercept_x_range_m=(0.035, 0.115),
    intercept_y_range_m=(-0.055, 0.075),
    intercept_z_range_m=(0.18, 0.34),
    lob_intercept_time_s=(0.34, 0.52),
    flat_intercept_time_s=(0.16, 0.25),
    mixed_intercept_time_s=(0.24, 0.42),
)


GLOVE_FEASIBLE_V1 = CatchTaskSpec(
    name="glove_feasible_v1",
    notes=(
        "Catchable distribution RE-DERIVED FOR THE GLOVE PLANT (2026-07-04, "
        "results/glove_catch_envelope_20260704/glove_catch_envelope.npz). Rollout-backed "
        "acceptance geometry: the Koopman-MPC reacher on the glove+wrist arm brings the "
        "pocket within 0.05 m of 86% of IK-reachable front-workspace points (median 0.13 s, "
        "p90 0.29 s from the ready pose); statically holdable everywhere below z~0.45 "
        "(gravity torque inside the pressure-limited torque polytope, residual <= 0.5 Nm; "
        "z >= 0.5 is infeasible). The lob acceptance box (x 0-0.13, y -0.13..0.08, "
        "z 0.16-0.36) lies inside that envelope, so what FEASIBLE_V2 got wrong for the "
        "heavier glove plant is ARRIVAL TIME, not position: this spec keeps the "
        "FEASIBLE_V2-like upward-lob launches but floors the intercept time at 0.42 s "
        "(>= p90 reach 0.29 s + lead safety margin), giving the arm time to arrive."
    ),
    modes=("lob_arc",),
    mode_probabilities=(1.0,),
    intercept_x_range_m=(0.0, 0.13),
    intercept_y_range_m=(-0.13, 0.08),
    intercept_z_range_m=(0.16, 0.36),
    lob_intercept_time_s=(0.42, 0.58),
    flat_intercept_time_s=(0.42, 0.58),
    mixed_intercept_time_s=(0.42, 0.58),
    # Launch envelope: same intercept box, but from further back with FEASIBLE_V2-like
    # horizontal speed so the ARRIVAL DIRECTION matches the envelope's nominal incoming
    # (~(0,-1.5,-2.0), ~37 deg from vertical) — the glove-orientation gate rejects
    # near-vertical arrivals (the default 0.3-0.5 m/s vy lob arrives ~12 deg from plumb).
    lob_launch_y_m=(-0.85, -0.55),
    lob_launch_z_m=(0.06, 0.16),
    lob_launch_vy_mps=(1.2, 1.7),
    lob_launch_vz_mps=(2.0, 3.0),
)


CATCH_TASK_SPECS: dict[str, CatchTaskSpec] = {
    spec.name: spec for spec in (CATCH_DIVERSE, GLOVE_FEASIBLE_V1)
}


def _sample_uniform(rng: np.random.Generator, lo_hi: Range, n: int) -> np.ndarray:
    return rng.uniform(float(lo_hi[0]), float(lo_hi[1]), size=int(n))


def _ballistic_initial_from_intercept(
    intercept_pos: np.ndarray,
    incoming_vel: np.ndarray,
    intercept_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial ball state whose state at ``intercept_time`` is requested."""
    t = np.asarray(intercept_time, dtype=np.float64).reshape(-1, 1)
    gravity = np.asarray([0.0, 0.0, -GRAVITY_MPS2], dtype=np.float64).reshape(1, 3)
    ball_vel = np.asarray(incoming_vel, dtype=np.float64) - gravity * t
    ball_pos = np.asarray(intercept_pos, dtype=np.float64) - np.asarray(incoming_vel, dtype=np.float64) * t
    ball_pos += 0.5 * gravity * t * t
    return ball_pos, ball_vel


def _sample_lob_mode(
    rng: np.random.Generator,
    spec: CatchTaskSpec,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Preserve the current FEASIBLE_V2-like upward launch envelope, then choose
    # a descending reachable intercept time from that ballistic trajectory.
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    remaining = int(n)
    attempts = 0
    gravity = np.asarray([0.0, 0.0, -GRAVITY_MPS2], dtype=np.float64).reshape(1, 3)
    while remaining > 0 and attempts < int(spec.max_resample_attempts):
        attempts += 1
        batch = max(remaining * 4, remaining + 16)
        p0 = np.stack(
            [
                _sample_uniform(rng, spec.lob_launch_x_m, batch),
                _sample_uniform(rng, spec.lob_launch_y_m, batch),
                _sample_uniform(rng, spec.lob_launch_z_m, batch),
            ],
            axis=-1,
        )
        v0 = np.stack(
            [
                _sample_uniform(rng, spec.lob_launch_vx_mps, batch),
                _sample_uniform(rng, spec.lob_launch_vy_mps, batch),
                _sample_uniform(rng, spec.lob_launch_vz_mps, batch),
            ],
            axis=-1,
        )
        t = _sample_uniform(rng, spec.lob_intercept_time_s, batch)
        pi = p0 + v0 * t.reshape(-1, 1) + 0.5 * gravity * t.reshape(-1, 1) ** 2
        vi = v0 + gravity * t.reshape(-1, 1)
        ok = (
            (pi[:, 0] >= spec.intercept_x_range_m[0])
            & (pi[:, 0] <= spec.intercept_x_range_m[1])
            & (pi[:, 1] >= spec.intercept_y_range_m[0])
            & (pi[:, 1] <= spec.intercept_y_range_m[1])
            & (pi[:, 2] >= spec.intercept_z_range_m[0])
            & (pi[:, 2] <= spec.intercept_z_range_m[1])
            & (vi[:, 2] <= -0.45)
        )
        keep = np.flatnonzero(ok)[:remaining]
        if keep.size:
            chunks.append((p0[keep], v0[keep], pi[keep], vi[keep], t[keep]))
            remaining -= int(keep.size)

    if remaining > 0:
        # Deterministic fallback: pick a reachable descending z target and solve
        # the descending branch time for FEASIBLE_V2-like launches.
        p0 = np.stack(
            [
                _sample_uniform(rng, spec.lob_launch_x_m, remaining),
                _sample_uniform(rng, spec.lob_launch_y_m, remaining),
                _sample_uniform(rng, spec.lob_launch_z_m, remaining),
            ],
            axis=-1,
        )
        v0 = np.stack(
            [
                _sample_uniform(rng, spec.lob_launch_vx_mps, remaining),
                _sample_uniform(rng, spec.lob_launch_vy_mps, remaining),
                _sample_uniform(rng, spec.lob_launch_vz_mps, remaining),
            ],
            axis=-1,
        )
        z_target = _sample_uniform(rng, (0.18, 0.34), remaining)
        disc = np.maximum(v0[:, 2] ** 2 - 2.0 * GRAVITY_MPS2 * (z_target - p0[:, 2]), 1e-9)
        t = (v0[:, 2] + np.sqrt(disc)) / GRAVITY_MPS2
        pi = p0 + v0 * t.reshape(-1, 1) + 0.5 * gravity * t.reshape(-1, 1) ** 2
        vi = v0 + gravity * t.reshape(-1, 1)
        chunks.append((p0, v0, pi, vi, t))

    ball_pos = np.concatenate([c[0] for c in chunks], axis=0)[:n]
    ball_vel = np.concatenate([c[1] for c in chunks], axis=0)[:n]
    intercept_pos = np.concatenate([c[2] for c in chunks], axis=0)[:n]
    incoming_vel = np.concatenate([c[3] for c in chunks], axis=0)[:n]
    return ball_pos, ball_vel, intercept_pos, incoming_vel


def _sample_intercept_mode(
    rng: np.random.Generator,
    spec: CatchTaskSpec,
    mode: str,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intercept_pos = np.stack(
        [
            _sample_uniform(rng, spec.intercept_x_range_m, n),
            _sample_uniform(rng, spec.intercept_y_range_m, n),
            _sample_uniform(rng, spec.intercept_z_range_m, n),
        ],
        axis=-1,
    )
    if mode == "flat_fast":
        intercept_time = _sample_uniform(rng, spec.flat_intercept_time_s, n)
        incoming_vel = np.stack(
            [
                _sample_uniform(rng, (-0.25, 0.25), n),
                _sample_uniform(rng, (1.35, 2.45), n),
                _sample_uniform(rng, (-0.18, 0.18), n),
            ],
            axis=-1,
        )
    elif mode == "mixed":
        intercept_time = _sample_uniform(rng, spec.mixed_intercept_time_s, n)
        incoming_vel = np.stack(
            [
                _sample_uniform(rng, (-0.18, 0.18), n),
                _sample_uniform(rng, (0.55, 1.35), n),
                _sample_uniform(rng, (-0.85, 0.45), n),
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"unknown intercept-mode sampler {mode!r}")

    ball_pos, ball_vel = _ballistic_initial_from_intercept(intercept_pos, incoming_vel, intercept_time)
    return ball_pos, ball_vel, intercept_pos, incoming_vel, intercept_time


def sample_catch_tasks(
    spec: CatchTaskSpec,
    seed: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample catch ballistics.

    Returns ``(ball_pos, ball_vel, intercept_pos, incoming_vel, intercept_time,
    mode_id)``. ``ball_pos`` and ``ball_vel`` are the initial ball state.
    ``incoming_vel`` is the velocity at the intercept.
    """
    rng = np.random.default_rng(int(seed))
    count = int(n)
    if count < 0:
        raise ValueError("n must be non-negative")
    if count == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, empty, empty, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    probs = np.asarray(spec.mode_probabilities, dtype=np.float64)
    probs = probs / np.sum(probs)
    mode_id = rng.choice(len(spec.modes), size=count, p=probs).astype(np.int32)

    ball_pos = np.zeros((count, 3), dtype=np.float64)
    ball_vel = np.zeros((count, 3), dtype=np.float64)
    intercept_pos = np.zeros((count, 3), dtype=np.float64)
    incoming_vel = np.zeros((count, 3), dtype=np.float64)
    intercept_time = np.zeros((count,), dtype=np.float64)

    for mode_index, mode in enumerate(spec.modes):
        idx = np.flatnonzero(mode_id == mode_index)
        if idx.size == 0:
            continue
        if mode == "lob_arc":
            p0, v0, pi, vi = _sample_lob_mode(rng, spec, int(idx.size))
            t = (vi[:, 2] - v0[:, 2]) / -GRAVITY_MPS2
        else:
            # Rejection keeps generated initial balls in a plausible launcher
            # region while preserving the requested intercept envelope.
            remaining = int(idx.size)
            chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            attempts = 0
            while remaining > 0 and attempts < int(spec.max_resample_attempts):
                attempts += 1
                batch = max(remaining * 3, remaining + 8)
                p0, v0, pi, vi, t = _sample_intercept_mode(rng, spec, mode, batch)
                ok = (
                    (p0[:, 1] >= -0.58)
                    & (p0[:, 1] <= -0.08)
                    & (p0[:, 2] >= 0.035)
                    & (p0[:, 2] <= 0.34)
                    & (np.linalg.norm(vi[:, :2], axis=1) >= 0.45)
                )
                keep = np.flatnonzero(ok)[:remaining]
                if keep.size:
                    chunks.append((p0[keep], v0[keep], pi[keep], vi[keep], t[keep]))
                    remaining -= int(keep.size)
            if remaining > 0:
                p0, v0, pi, vi, t = _sample_intercept_mode(rng, spec, mode, remaining)
                chunks.append((p0, v0, pi, vi, t))
            p0 = np.concatenate([c[0] for c in chunks], axis=0)[: idx.size]
            v0 = np.concatenate([c[1] for c in chunks], axis=0)[: idx.size]
            pi = np.concatenate([c[2] for c in chunks], axis=0)[: idx.size]
            vi = np.concatenate([c[3] for c in chunks], axis=0)[: idx.size]
            t = np.concatenate([c[4] for c in chunks], axis=0)[: idx.size]

        ball_pos[idx] = p0
        ball_vel[idx] = v0
        intercept_pos[idx] = pi
        incoming_vel[idx] = vi
        intercept_time[idx] = t

    return (
        ball_pos.astype(np.float64),
        ball_vel.astype(np.float64),
        intercept_pos.astype(np.float64),
        incoming_vel.astype(np.float64),
        intercept_time.astype(np.float64),
        mode_id.astype(np.int32),
    )


def summarize_catch_samples(
    spec: CatchTaskSpec,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    intercept_pos: np.ndarray,
    incoming_vel: np.ndarray,
    intercept_time: np.ndarray,
    mode_id: np.ndarray,
) -> dict[str, object]:
    arrays = {
        "initial_pos_m": np.asarray(ball_pos, dtype=np.float64),
        "initial_vel_mps": np.asarray(ball_vel, dtype=np.float64),
        "intercept_pos_m": np.asarray(intercept_pos, dtype=np.float64),
        "incoming_vel_at_intercept_mps": np.asarray(incoming_vel, dtype=np.float64),
    }
    summary: dict[str, object] = {"task_name": spec.name, "n": int(len(mode_id)), "modes": {}}
    for mode_index, mode in enumerate(spec.modes):
        summary["modes"][mode] = int(np.sum(np.asarray(mode_id) == mode_index))
    for name, arr in arrays.items():
        summary[name] = {
            "min": [float(v) for v in np.min(arr, axis=0)] if arr.size else [0.0, 0.0, 0.0],
            "max": [float(v) for v in np.max(arr, axis=0)] if arr.size else [0.0, 0.0, 0.0],
        }
    t = np.asarray(intercept_time, dtype=np.float64)
    speed = np.linalg.norm(np.asarray(incoming_vel, dtype=np.float64), axis=1)
    summary["intercept_time_s"] = {"min": float(np.min(t)), "max": float(np.max(t))}
    summary["incoming_speed_mps"] = {"min": float(np.min(speed)), "max": float(np.max(speed))}
    return summary
