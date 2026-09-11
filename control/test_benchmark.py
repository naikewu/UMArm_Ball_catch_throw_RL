"""Benchmark records identify the model and the wire commands actually used."""
import hashlib

import numpy as np

from control import benchmark
from control.controller import PA_PER_PSI
from control.trajectory import tip_position


class _Hold:
    duration_s = .04
    metadata = {"kind": "test hold"}

    def sample(self, t):
        return np.zeros(12), np.zeros(12), np.zeros(12)

    def future(self, t, H, dt):
        return np.zeros((H, 12))

    def pen_down(self, t):
        return False

    def tip_target(self, t):
        return tip_position(np.zeros(12))


def test_default_checkpoint_hash_wire_targets_and_sampler_warmup(tmp_path, monkeypatch):
    path = tmp_path/"weights.pt"
    path.write_bytes(b"test checkpoint identified before mocked controller construction")
    draws = []

    class Controller:
        horizon = 1
        model_meta = {"schema": "test-only"}

        def __init__(self):
            self.rng = np.random.default_rng(83)

        def reset(self, q, p):
            pass

        def command(self, *args, **kwargs):
            draws.append(self.rng.normal())
            return np.full(24, 2.34567*PA_PER_PSI)

    monkeypatch.setattr(benchmark, "DEFAULT_CHECKPOINT", path)
    monkeypatch.setattr(benchmark, "make_controller", lambda *a, **k: Controller())
    trace, meta = benchmark.run("koopman_mppi", _Hold())
    assert meta["checkpoint"] == str(path.resolve())
    assert meta["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["model_metadata"] == Controller.model_meta
    assert draws[0] == draws[1], "warmup must not consume experiment sampler noise"
    assert np.max(np.abs(trace["wire_targets_pa"]-trace["targets_pa"])) > 0
    assert np.max(np.abs(trace["wire_targets_pa"]-trace["targets_pa"])) < .02*PA_PER_PSI
    assert trace["qdot_observed"].shape == (6, 12)
    assert np.all(trace["mocap_time_s"] <= np.maximum(trace["t"]-2/240, 0)+1e-12)
    assert "control/observation.py" in meta["source_sha256"]
