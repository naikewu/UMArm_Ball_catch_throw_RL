"""Runtime evaluator for the latest fitted-twin tip-compliance head."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from control.controller import PA_PER_PSI, pair_indices


COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
DEFAULT_HEAD = Path(__file__).resolve().parents[1] / "data" / "compliance_fit_latest_3000" / "latest_twin_compliance_head.npz"


def feature_matrix(q: np.ndarray, p_pa: np.ndarray) -> tuple[np.ndarray, dict]:
    q = np.asarray(q, dtype=np.float64)
    p_pa = np.asarray(p_pa, dtype=np.float64)
    if q.shape[-1] != 12 or p_pa.shape[-1] != 24:
        raise ValueError("expected q[...,12] and p_pa[...,24]")
    flat_q = q.reshape((-1, 12))
    flat_p = p_pa.reshape((-1, 24))
    p_psi = flat_p / PA_PER_PSI
    pairs = pair_indices()
    pair_mean = p_psi[:, pairs].mean(axis=2)
    pair_diff = p_psi[:, pairs[:, 0]] - p_psi[:, pairs[:, 1]]
    inv_mean = 1.0 / np.clip(pair_mean, 0.25, None)
    blocks = [np.ones((flat_q.shape[0], 1)), flat_q, flat_q * flat_q,
              p_psi, p_psi * p_psi, pair_mean, inv_mean, pair_diff]
    for seg in range(3):
        js = slice(4 * seg, 4 * seg + 4)
        blocks.append(flat_q[:, js] * pair_mean[:, 2 * seg:2 * seg + 1])
        blocks.append(flat_q[:, js] * pair_mean[:, 2 * seg + 1:2 * seg + 2])
    X = np.concatenate(blocks, axis=1)
    meta = {
        "feature_blocks": [
            "1", "q12", "q12_squared", "p24_psi", "p24_psi_squared",
            "pair_mean12_psi", "inv_pair_mean12_1_per_psi", "pair_diff12_psi",
            "segment_local_q_times_pair_mean",
        ],
        "feature_dim": int(X.shape[1]),
    }
    return X, meta


def matrix_from_components(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    C = np.zeros(y.shape[:-1] + (3, 3), dtype=np.float64)
    C[..., 0, 0], C[..., 1, 1], C[..., 2, 2] = y[..., 0], y[..., 1], y[..., 2]
    C[..., 0, 1] = C[..., 1, 0] = y[..., 3]
    C[..., 0, 2] = C[..., 2, 0] = y[..., 4]
    C[..., 1, 2] = C[..., 2, 1] = y[..., 5]
    return C


class TipComplianceHead:
    """Polynomial/ridge surrogate Cx(q,p) in m/N for MPC cost evaluation."""

    def __init__(self, path: Path | str = DEFAULT_HEAD):
        self.path = Path(path)
        payload = np.load(self.path, allow_pickle=True)
        self.coef = payload["coef"].astype(np.float64)
        self.x_mean = payload["x_mean"].astype(np.float64)
        self.x_std = payload["x_std"].astype(np.float64)
        self.y_mean = payload["y_mean"].astype(np.float64)
        self.y_std = payload["y_std"].astype(np.float64)
        raw_meta = payload["meta"].item()
        self.meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta

    def predict(self, q: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        p_pa = np.asarray(p_pa, dtype=np.float64)
        shape = np.broadcast_shapes(q.shape[:-1], p_pa.shape[:-1])
        qb = np.broadcast_to(q, shape + (12,))
        pb = np.broadcast_to(p_pa, shape + (24,))
        X, _ = feature_matrix(qb, pb)
        y = ((X - self.x_mean) / self.x_std) @ self.coef * self.y_std + self.y_mean
        return matrix_from_components(y).reshape(shape + (3, 3))
