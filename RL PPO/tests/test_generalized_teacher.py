import json

import pytest

from teacher_rl.data import load_dataset, write_json
from teacher_rl.generalized_teacher import (FIXED_SCENARIO, GENERALIZED_SCHEMA,
    PARAMETER_RANGES, GeneralizedTeacherEnv, make_scenario_manifest)


def test_generalized_manifest_is_deterministic_split_and_bounded():
    first = make_scenario_manifest()
    second = make_scenario_manifest()
    assert first == second
    assert first["train_episodes"] == 205
    assert first["validation_episodes"] == 51
    assert len({row["seed"] for row in first["scenarios"]}) == 256
    assert {row["split"] for row in first["scenarios"]} == {"train", "validation"}
    for row in first["scenarios"]:
        for name, (low, high) in PARAMETER_RANGES.items():
            assert low <= row[name] <= high
        for name, value in FIXED_SCENARIO.items():
            assert row[name] == value


def test_generalized_environment_uses_scenario_distribution():
    scenario = make_scenario_manifest(5)["scenarios"][0]
    env = GeneralizedTeacherEnv(scenario)
    assert env.config.launch_speed == scenario["launch_speed"]
    assert env.config.target_distance == scenario["target_distance"]
    assert env.distribution.v_jit == 0
    assert env.distribution.ang_jit == scenario["launch_angle_jitter_deg"]
    assert env.distribution.pos_jit == scenario["launch_position_jitter_m"]


def test_v14_data_cannot_enter_older_training(tmp_path):
    write_json(tmp_path / "episode_1.json", dict(schema=GENERALIZED_SCHEMA, source_hash="v14"))
    with pytest.raises(ValueError, match="mismatch"):
        load_dataset(tmp_path)


def test_manifest_hash_detects_changes():
    manifest = make_scenario_manifest(10)
    changed = json.loads(json.dumps(manifest))
    changed["scenarios"][0]["launch_speed"] += .01
    assert changed != manifest
