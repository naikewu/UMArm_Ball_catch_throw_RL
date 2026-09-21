from dataclasses import asdict
from types import SimpleNamespace
import json

import numpy as np
import pytest

from teacher_rl.data import write_json
from teacher_rl.env import ACTION_SIZE, OBS_SIZE
from teacher_rl.generalized_teacher import make_scenario_manifest
from teacher_rl.improved_bc import (acceptance, balanced_quality_weights, train,
    validate_dataset)
from teacher_rl.improved_teacher import (IMPROVED_SCHEMA, ImprovedRecipe,
    improved_fingerprint)
from teacher_rl.model import load_checkpoint
from teacher_rl.reachable_collection import collector_fingerprint, constrain_manifest


def _dataset(path):
    path.mkdir(parents=True, exist_ok=True)
    manifest = constrain_manifest(make_scenario_manifest(5, 30001, 101), .315)
    recipe = ImprovedRecipe(catch_match=.22, throw_force_gain_n_per_m=0.,
        throw_radius_gain_m_per_m=.2)
    contract = dict(schema=IMPROVED_SCHEMA, source_hash=improved_fingerprint(),
        collector_source_hash=collector_fingerprint(),
        manifest_sha256=manifest["manifest_sha256"], recipe=asdict(recipe),
        selection_file="missing-for-unit-test.json",
        selection_source_hash=improved_fingerprint(), batched_actuator=True)
    write_json(path / "scenario_manifest.json", manifest)
    write_json(path / "dataset_contract.json", contract)
    for scenario in manifest["scenarios"]:
        seed = scenario["seed"]
        phases = np.arange(6, dtype=np.float32)
        masks = np.zeros((6, ACTION_SIZE), dtype=np.float32)
        masks[0, :5] = 1
        masks[0, 9] = 1
        masks[1:3, 4] = 1
        masks[3, 4:9] = 1
        data_name = f"episode_{seed}_unit.npz"
        np.savez_compressed(path / data_name,
            observations=np.zeros((6, OBS_SIZE), dtype=np.float32),
            teacher_actions=np.zeros((6, ACTION_SIZE), dtype=np.float32),
            masks=masks, phases=phases)
        write_json(path / f"episode_{seed}_unit.json", dict(
            schema=IMPROVED_SCHEMA, source_hash=improved_fingerprint(),
            name="selected", seed=seed, scenario=scenario, recipe=asdict(recipe),
            batched_actuator=True, split=scenario["split"], samples=6,
            data=data_name, result=dict(hit15=True, captured=True, weld_peak_n=300.)))
    return manifest


def test_balanced_quality_weights_equalize_phases_and_keep_quality():
    import torch
    data = (torch.zeros(4, 1), torch.zeros(4, 1), torch.ones(4, 1),
        torch.tensor([0, 0, 1, 1]), torch.tensor([1., .25, 1., 1.]))
    weights = balanced_quality_weights(data)
    assert weights[0] == pytest.approx(4 * weights[1])
    assert weights[:2].sum() == pytest.approx(.625)
    assert weights[2:].sum() == pytest.approx(1.)


def test_v15_dataset_contract_and_training_smoke(tmp_path):
    manifest = _dataset(tmp_path / "data")
    validated = validate_dataset(tmp_path / "data")
    assert validated["manifest"]["manifest_sha256"] == manifest["manifest_sha256"]
    args = SimpleNamespace(data=tmp_path / "data", out=tmp_path / "bc",
        epochs=2, batch=4, lr=3e-4, seed=7, threads=1)
    train(args)
    _, checkpoint = load_checkpoint(args.out / "bc_best.pt",
        expected_schema=IMPROVED_SCHEMA)
    assert checkpoint["kind"] == "v15_bc"
    assert checkpoint["dataset"]["episodes"] == 5


def test_v15_dataset_rejects_student_trajectory(tmp_path):
    _dataset(tmp_path)
    path = next(tmp_path.glob("episode_*.json"))
    row = json.loads(path.read_text(encoding="utf-8"))
    row["checkpoint_sha256"] = "student"
    write_json(path, row)
    with pytest.raises(ValueError, match="student/DAgger"):
        validate_dataset(tmp_path)


def test_formal_acceptance_thresholds():
    teacher = dict(captured_rate=.90, hit15_rate=.89)
    bc = dict(captured_rate=.88, hit15_rate=.85, mean_landing_error_m=.06)
    assert acceptance(teacher, bc, .82)["passed"]
    bc["captured_rate"] = .86
    result = acceptance(teacher, bc, .82)
    assert not result["passed"]
    assert not result["checks"]["capture_drop_le_3pp"]
