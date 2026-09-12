"""Fit polynomial joint stiffness from the current twin, then use C=J K^-1 J.T."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from control.controller import PA_PER_PSI, pair_indices
from control.trajectory import tip_jacobian
from digital_twin.tangent_compliance import TangentCompliance

DEFAULT_TANGENT_HEAD = Path(__file__).resolve().parents[1] / "data/compliance_tangent_3000/head.npz"
TRIU = np.triu_indices(12)


def features(q, p):
    q, p = np.broadcast_arrays(np.asarray(q)[..., :, None], np.asarray(p)[..., None, :] / PA_PER_PSI)
    q, p = q[..., :, 0], p[..., 0, :]
    return np.concatenate((np.ones(q.shape[:-1] + (1,)), q,
                           (q[..., :, None] * q[..., None, :])[..., TRIU[0], TRIU[1]],
                           p, (q[..., :, None] * p[..., None, :]).reshape(q.shape[:-1] + (288,))), axis=-1)


def unpack(values):
    K = np.zeros(values.shape[:-1] + (12, 12))
    K[..., TRIU[0], TRIU[1]] = values
    K[..., TRIU[1], TRIU[0]] = values
    return K


class TangentComplianceHead:
    def __init__(self, path=DEFAULT_TANGENT_HEAD):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as f:
            self.coef, self.mean, self.scale = f["coef"], f["mean"], f["scale"]
            self.meta = json.loads(f["meta"].item())
        if self.meta["schema"] != "plate_tangent_stiffness_v1":
            raise ValueError("Wrong compliance head definition")

    def stiffness(self, q, p):
        return unpack(((features(q, p) - self.mean) / self.scale) @ self.coef)

    def predict(self, q, p):
        K = self.stiffness(q, p)
        J = tip_jacobian(q)
        return J @ np.linalg.solve(K, np.swapaxes(J, -1, -2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--mean-min-psi", type=float, default=2.)
    parser.add_argument("--mean-max-psi", type=float, default=14.)
    parser.add_argument("--out", type=Path, default=DEFAULT_TANGENT_HEAD.parent)
    args = parser.parse_args()
    if not 0 < args.mean_min_psi < args.mean_max_psi < 15 or args.samples < 10:
        parser.error("require 0 < mean min < mean max < 15 psi, and at least 10 samples")
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    rng = np.random.default_rng(20260913)
    exact = TangentCompliance()
    q = rng.uniform(-.10, .10, (args.samples, 12))
    p = np.zeros((args.samples, 24))
    pairs = pair_indices()
    means = rng.uniform(args.mean_min_psi, args.mean_max_psi, (args.samples, 12))
    diff = rng.uniform(-3, 3, means.shape)
    diff = np.clip(diff, -2*means, 2*means)
    p[:, pairs[:, 0]] = (means + diff / 2) * PA_PER_PSI
    p[:, pairs[:, 1]] = (means - diff / 2) * PA_PER_PSI
    K = np.empty((args.samples, 12, 12))
    for i in range(args.samples):
        K[i] = exact.stiffness(q[i], p[i])
        if (i + 1) % 300 == 0:
            print(f"tangent labels {i + 1}/{args.samples}", flush=True)
    X = features(q, p)
    n = int(.8 * len(q))
    mean, scale = X[:n].mean(0), X[:n].std(0)
    mean[0], scale[0] = 0., 1.
    scale[scale < 1e-10] = 1.
    X = (X - mean) / scale
    y = K[:, TRIU[0], TRIU[1]]
    coef = np.linalg.solve(X[:n].T @ X[:n] + 1e-5 * np.eye(X.shape[1]), X[:n].T @ y[:n])
    Kfit = unpack(X @ coef)
    J = tip_jacobian(q)
    C = J @ np.linalg.solve(K, np.swapaxes(J, -1, -2))
    Cfit = J @ np.linalg.solve(Kfit, np.swapaxes(J, -1, -2))
    error = np.linalg.norm(Cfit - C, axis=(1, 2)) / np.linalg.norm(C, axis=(1, 2))
    metrics = {"holdout_relF_median": float(np.median(error[n:])),
               "holdout_relF_p90": float(np.percentile(error[n:], 90)),
               "holdout_relF_max": float(error[n:].max()),
               "holdout_frames": len(q) - n,
               "minimum_fitted_stiffness_eigenvalue": float(np.linalg.eigvalsh(Kfit).min())}
    root = Path(__file__).resolve().parents[1]
    provenance = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in (
        "digital_twin/checkpoints/canarm_mech.json", "digital_twin/checkpoints/canarm_flow.npz",
        "digital_twin/tangent_compliance.py", "digital_twin/actuator_model.py")}
    meta = dict(schema="plate_tangent_stiffness_v1", definition="frozen pressure, local static tangent with constant balancing torque",
                endpoint="final plate centre, arm base axes", units="m/N", samples=len(q),
                q_range_rad=.10, pair_mean_psi=[args.mean_min_psi, args.mean_max_psi],
                pair_difference_psi=[-3, 3], nonnegative_lines=True,
                source_sha256=provenance, metrics=metrics)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "dataset.npz", q=q, p_pa=p, K=K, C=C)
    np.savez_compressed(args.out / "head.npz", coef=coef, mean=mean, scale=scale, meta=json.dumps(meta))
    (args.out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
