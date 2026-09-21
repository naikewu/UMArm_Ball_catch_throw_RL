import numpy as np

from rl_ppo.catch_throw_task import sample_task
from rl_ppo.train_catch_throw import curriculum_transition, difficulty_label, evaluation_score


def test_stage_one_requires_updates_and_capture_rate() -> None:
    passing = {
        "capture_rate": .85,
        "stable_capture_rate": .85,
        "success_rate": .85,
        "mean_first_contact_relative_speed_mps": 1.80,
        "mean_grasp_event_relative_speed_mps": .60,
        "unsafe_force_rate": 0.0,
        "mean_predicted_entry_radial_m": .020,
    }
    assert curriculum_transition(
        1, 1, 19, passing, minimum_updates=20, success_threshold=.80
    ) == (1, 1)
    assert curriculum_transition(
        1, 1, 20, {**passing, "mean_predicted_entry_radial_m": .026},
        minimum_updates=20, success_threshold=.80,
    ) == (2, 1)
    assert curriculum_transition(
        1, 1, 20, passing, minimum_updates=20, success_threshold=.80
    ) == (2, 1)


def test_stage_two_advances_through_three_sublevels() -> None:
    evaluation = {
        "capture_rate": .80,
        "stable_capture_rate": .80,
        "success_rate": .80,
        "mean_grasp_event_relative_speed_mps": .50,
        "mean_grasp_event_alignment": .70,
        "mean_grasp_event_radial_offset_m": .020,
        "unsafe_force_rate": 0.0,
    }
    assert curriculum_transition(
        2, 1, 20, evaluation, minimum_updates=20, success_threshold=.80
    ) == (2, 2)
    assert curriculum_transition(
        2, 2, 20, evaluation, minimum_updates=20, success_threshold=.80
    ) == (2, 3)
    assert curriculum_transition(
        2, 3, 20, evaluation, minimum_updates=20, success_threshold=.80
    ) == (3, 3)
    assert difficulty_label(2, 3) == "2C"


def test_stage_two_quality_metrics_block_premature_promotion() -> None:
    noisy_capture = {
        "capture_rate": .90,
        "stable_capture_rate": .90,
        "success_rate": .90,
        "mean_grasp_event_relative_speed_mps": .51,
        "mean_grasp_event_alignment": .50,
        "mean_grasp_event_radial_offset_m": .030,
        "unsafe_force_rate": 0.0,
    }
    assert curriculum_transition(
        2, 2, 20, noisy_capture, minimum_updates=20, success_threshold=.80
    ) == (2, 2)


def test_auto_volume_curriculum_uses_stable_capture_not_soft_catch_gate() -> None:
    automatic_capture = {
        "capture_rate": .85,
        "stable_capture_rate": .85,
        "success_rate": .85,
        "mean_grasp_event_relative_speed_mps": 1.5,
        "mean_grasp_event_alignment": .30,
        "mean_grasp_event_radial_offset_m": .024,
        "unsafe_force_rate": 0.0,
    }
    assert curriculum_transition(
        2, 1, 20, automatic_capture,
        minimum_updates=20, success_threshold=.80,
        grasp_mode="auto_volume",
    ) == (2, 2)
    assert curriculum_transition(
        2, 1, 20, {**automatic_capture, "unsafe_force_rate": .05},
        minimum_updates=20, success_threshold=.80,
        grasp_mode="auto_volume",
    ) == (2, 1)


def test_best_checkpoint_score_prefers_capture_then_quality() -> None:
    higher_capture_poor_alignment = {
        "capture_rate": .90,
        "release_rate": 0.0,
        "success_rate": .90,
        "mean_grasp_event_relative_speed_mps": .40,
        "mean_grasp_event_alignment": .49,
        "mean_grasp_event_radial_offset_m": .016,
        "unsafe_force_rate": 0.0,
    }
    lower_capture_better_alignment = {
        **higher_capture_poor_alignment,
        "capture_rate": .80,
        "success_rate": .80,
        "mean_grasp_event_alignment": .59,
    }
    assert evaluation_score(higher_capture_poor_alignment) > evaluation_score(
        lower_capture_better_alignment
    )


def test_best_checkpoint_prefers_grasp_gate_quality_before_rendezvous_noise() -> None:
    baseline = {
        "stage": 2,
        "stage2_level": 1,
        "capture_rate": .80,
        "stable_capture_rate": .80,
        "release_rate": 0.0,
        "success_rate": .80,
        "grasp_event_physical_compatible_rate": .25,
        "mean_grasp_event_gate_score": 2.0,
        "mean_grasp_event_relative_speed_mps": .60,
        "mean_grasp_event_alignment": .70,
        "mean_grasp_event_radial_offset_m": .020,
        "mean_rendezvous_relative_speed_mps": 1.0,
        "unsafe_force_rate": 0.0,
    }
    worse_gate_better_rendezvous = {
        **baseline,
        "mean_grasp_event_gate_score": 3.0,
        "mean_rendezvous_relative_speed_mps": .8,
    }
    assert evaluation_score(baseline) > evaluation_score(
        worse_gate_better_rendezvous
    )


def test_stage_two_task_offset_matches_sublevel() -> None:
    home = np.array([.2, -.1, .5])
    ranges = {1: (.025, .050), 2: (.040, .075), 3: (.055, .100)}
    for level in range(1, 4):
        task = sample_task(
            np.random.default_rng(100 + level), home,
            curriculum_stage=2, stage2_level=level,
        )
        lateral_radius = np.linalg.norm(task.intercept_world_m[:2] - home[:2])
        assert ranges[level][0] <= lateral_radius <= ranges[level][1]
        assert task.stage2_level == level
