from teacher_rl.generalized_teacher import make_scenario_manifest
from teacher_rl.reachable_collection import _reachable, constrain_manifest


def test_reachability_constraint_preserves_valid_rows_and_repairs_invalid_rows():
    manifest = make_scenario_manifest(5)
    manifest["scenarios"][0]["launch_speed"] = 4.5
    manifest["scenarios"][0]["launch_distance"] = 2.3
    original_second = dict(manifest["scenarios"][1])
    constrained = constrain_manifest(manifest, .314541)
    repaired = constrained["scenarios"][0]
    assert repaired["launch_distance"] < 2.3
    assert _reachable(repaired["launch_speed"], repaired["launch_distance"], -.185459)
    assert constrained["scenarios"][1] == original_second
    assert len(constrained["reachability_constraint"]["adjusted_scenarios"]) == 1
