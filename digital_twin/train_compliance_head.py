"""Legacy finite-time force-response fit for the fitted CAN-arm twin.

For local static compliance at the controlled plate centre, use
``python -m digital_twin.train_tangent_compliance``. This legacy collector
does not establish equilibrium and reads the 50 mm tip stub rather than
the plate centre; its historical outputs must not be treated as static C.

The labels come from the current ``digital_twin.twin_params`` MuJoCo model:
for each sampled joint/pressure state, the script freezes the pressure, applies
small +/- Cartesian forces, integrates for a fixed duration, and estimates

    Cx(q, p) = d tip_xyz / d force_xyz       [m/N].

The fitted head is a ridge-regression surrogate from ``(q, p)`` to the six
symmetric entries of ``Cx``.  It is intentionally independent of the older
Koopman-compliance package, because this file is meant to follow the latest
outer digital twin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from control.controller import PA_PER_PSI, pair_indices, project_pressures
from digital_twin.compliance_head import feature_matrix as _feature_matrix
from digital_twin.compliance_head import matrix_from_components as _matrix_from_components
from digital_twin.sim_core import SimArm
from digital_twin.twin_params import describe, load_twin_kwargs


COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def _sample_pressures(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample pressure commands in the controller's legal pair envelope."""
    pairs = pair_indices()
    out = np.zeros((n, 24), dtype=np.float64)
    for k in range(n):
        p = np.zeros(24, dtype=np.float64)
        for a, b in pairs:
            pair_sum = rng.uniform(2.0, 30.0) * PA_PER_PSI
            frac = rng.beta(1.7, 1.7)
            p[a] = pair_sum * frac
            p[b] = pair_sum * (1.0 - frac)
        out[k] = project_pressures(p)
    return out


def _set_state(arm: SimArm, q: np.ndarray, p_pa: np.ndarray) -> None:
    arm.data.qpos[:] = 0.0
    arm.data.qvel[:] = 0.0
    if arm._q_qposadr is None:
        arm.data.qpos[:12] = q
    else:
        arm.data.qpos[arm._q_qposadr] = q
    for node in arm.nodes.values():
        node.p_pa = float(p_pa[node.index])
        psi = node.p_pa / PA_PER_PSI
        node.raw = int(max(0, min(node.adc.adc_max, round(node.adc.counts_of(psi)))))
        node.filtered_raw = float(node.raw)
        node._filter_primed = True
    arm.data.ctrl[:] = 0.0
    arm.data.xfrc_applied[:] = 0.0
    arm._mujoco.mj_forward(arm.model, arm.data)


def _frozen_step(arm: SimArm, p_pa: np.ndarray, force: np.ndarray | None) -> None:
    if force is None:
        arm.data.xfrc_applied[:] = 0.0
    else:
        arm.data.xfrc_applied[:] = 0.0
        tip_body = int(arm.model.site("canarm_tip").bodyid[0])
        arm.data.xfrc_applied[tip_body, :3] = force
    dlen = arm.data.ten_length[arm._ten_ids] - arm._ten_len0
    arm.data.ctrl[arm._act_ids] = arm.actuator.force_n(p_pa, dlen)
    if arm.tendon_damping_const is None:
        length = arm._l0_per_act + dlen
        arm.model.tendon_damping[arm._ten_ids] = arm.actuator.tendon_damping_n_s_m(
            p_pa, base=arm._tendon_damping_base, l_m=length)
    else:
        arm.model.tendon_damping[arm._ten_ids] = float(arm.tendon_damping_const)
    arm._mujoco.mj_step(arm.model, arm.data)


def _settled_tip(arm: SimArm, q: np.ndarray, p_pa: np.ndarray, *,
                 force: np.ndarray | None, settle_s: float) -> np.ndarray:
    _set_state(arm, q, p_pa)
    steps = max(1, int(round(settle_s / arm.dt)))
    for _ in range(steps):
        _frozen_step(arm, p_pa, force)
    arm._mujoco.mj_forward(arm.model, arm.data)
    return np.array(arm.data.site("canarm_tip").xpos, dtype=np.float64)


def collect_dataset(samples: int, q_range_rad: float, force_n: float,
                    settle_s: float, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    kwargs = load_twin_kwargs(log=lambda _: None)
    kwargs.update(seed=seed, batched_actuator=True)
    arm = SimArm(**kwargs)
    q = rng.uniform(-q_range_rad, q_range_rad, size=(samples, 12))
    p = _sample_pressures(rng, samples)
    C = np.empty((samples, 3, 3), dtype=np.float64)
    residual = np.empty(samples, dtype=np.float64)
    axes = np.eye(3, dtype=np.float64)
    for i in range(samples):
        base = _settled_tip(arm, q[i], p[i], force=None, settle_s=settle_s)
        for j in range(3):
            f = axes[j] * force_n
            plus = _settled_tip(arm, q[i], p[i], force=f, settle_s=settle_s)
            minus = _settled_tip(arm, q[i], p[i], force=-f, settle_s=settle_s)
            C[i, :, j] = (plus - minus) / (2.0 * force_n)
        C[i] = 0.5 * (C[i] + C[i].T)
        residual[i] = float(np.linalg.norm(_settled_tip(
            arm, q[i], p[i], force=None, settle_s=settle_s) - base))
        if (i + 1) % max(1, samples // 10) == 0:
            print(f"collected {i + 1}/{samples}", flush=True)
    meta = {
        "kind": "latest_outer_twin_tip_compliance_dataset",
        "label": "legacy finite-time force response at the tip stub; not static plate compliance",
        "units": {"q": "rad", "p": "Pa gauge", "compliance": "m/N"},
        "samples": int(samples),
        "q_range_rad": float(q_range_rad),
        "force_n": float(force_n),
        "settle_s": float(settle_s),
        "seed": int(seed),
        "twin": describe(kwargs),
    }
    return {"q": q.astype(np.float32), "p_pa": p.astype(np.float32),
            "compliance": C.astype(np.float32),
            "settle_residual_m": residual.astype(np.float32)}, meta


def _components(C: np.ndarray) -> np.ndarray:
    return np.stack([C[:, 0, 0], C[:, 1, 1], C[:, 2, 2],
                     C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]], axis=1)


def fit_head(dataset: dict, seed: int, ridge: float) -> tuple[dict, dict]:
    q, p, C = dataset["q"], dataset["p_pa"], dataset["compliance"]
    X, feature_meta = _feature_matrix(q, p)
    y = _components(C)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_train = max(1, int(round(0.8 * X.shape[0])))
    tr, va = idx[:n_train], idx[n_train:]
    x_mean = X[tr].mean(axis=0)
    x_std = X[tr].std(axis=0)
    x_std[x_std < 1e-9] = 1.0
    y_mean = y[tr].mean(axis=0)
    y_std = y[tr].std(axis=0)
    y_std[y_std < 1e-12] = 1.0
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std
    A = Xn[tr].T @ Xn[tr] + float(ridge) * np.eye(Xn.shape[1])
    B = Xn[tr].T @ yn[tr]
    coef = np.linalg.solve(A, B)
    pred = Xn @ coef * y_std + y_mean
    C_pred = _matrix_from_components(pred)
    num = np.linalg.norm(C_pred - C, axis=(1, 2))
    den = np.maximum(np.linalg.norm(C, axis=(1, 2)), 1e-12)
    rel = num / den
    metrics = {
        "train_relF_median": float(np.median(rel[tr])),
        "train_relF_p90": float(np.percentile(rel[tr], 90)),
        "holdout_relF_median": float(np.median(rel[va])) if len(va) else None,
        "holdout_relF_p90": float(np.percentile(rel[va], 90)) if len(va) else None,
        "holdout_frames": int(len(va)),
        "ridge": float(ridge),
    }
    fit = {
        "coef": coef.astype(np.float32),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }
    meta = {"kind": "latest_outer_twin_polynomial_compliance_head",
            "input": "q12 rad + p24 Pa",
            "output_components": COMPONENTS,
            "feature_meta": feature_meta,
            "metrics": metrics}
    return fit, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--q-range-rad", type=float, default=np.deg2rad(18.0))
    parser.add_argument("--force-n", type=float, default=0.10)
    parser.add_argument("--settle-s", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/compliance_fit"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--head", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset or args.out_dir / "latest_twin_compliance_dataset.npz"
    head_path = args.head or args.out_dir / "latest_twin_compliance_head.npz"

    dataset, dataset_meta = collect_dataset(
        samples=args.samples,
        q_range_rad=args.q_range_rad,
        force_n=args.force_n,
        settle_s=args.settle_s,
        seed=args.seed,
    )
    np.savez_compressed(dataset_path, **dataset, meta=json.dumps(dataset_meta))
    fit, fit_meta = fit_head(dataset, seed=args.seed, ridge=args.ridge)
    np.savez_compressed(head_path, **fit, meta=json.dumps(fit_meta))
    print(f"wrote dataset: {dataset_path}")
    print(f"wrote head: {head_path}")
    print(json.dumps(fit_meta["metrics"], indent=2))


if __name__ == "__main__":
    main()
