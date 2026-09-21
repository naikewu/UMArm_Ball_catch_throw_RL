from dataclasses import replace

import pytest

from teacher_rl.generalized_teacher import make_scenario_manifest
from teacher_rl.improved_teacher import ImprovedRecipe, ImprovedTeacherEnv, candidates


def test_throw_force_is_target_conditioned_and_bounded():
    recipe = ImprovedRecipe(throw_force_gain_n_per_m=10.)
    assert recipe.throw_force(1.3) == pytest.approx(8.5)
    assert recipe.throw_force(1.45) == pytest.approx(10.)
    assert recipe.throw_force(1.7) == pytest.approx(12.5)
    assert replace(recipe, throw_force_gain_n_per_m=30.).throw_force(2.) == 14.
    assert replace(recipe, throw_force_gain_n_per_m=-10.).throw_force(1.7) == pytest.approx(7.5)


def test_throw_radius_is_target_conditioned_and_bounded():
    recipe = ImprovedRecipe(throw_radius_gain_m_per_m=.4)
    assert recipe.throw_radius(1.45) == pytest.approx(.62)
    assert recipe.throw_radius(1.7) == pytest.approx(.72)
    assert recipe.throw_radius(1.3) == pytest.approx(.56)


def test_candidates_keep_an_explicit_baseline():
    recipes = candidates()
    assert recipes["baseline"].catch_aim_mode == "baseline"
    assert recipes["baseline"].catch_match == .3
    assert recipes["baseline"].throw_force_gain_n_per_m == 0.
    assert recipes["match20"].catch_match == .2
    assert recipes["force_neg10"].throw_force_gain_n_per_m == -10.
    assert recipes["radius20"].throw_radius_gain_m_per_m == .2
    assert recipes["match20_radius20"].catch_match == .2


def test_improved_environment_uses_recipe_and_scenario():
    scenario = make_scenario_manifest(5)["scenarios"][0]
    recipe = ImprovedRecipe(catch_match=.4, throw_force_gain_n_per_m=10.)
    env = ImprovedTeacherEnv(scenario, recipe)
    assert env.recipe.match == .4
    assert env.throw_force_n == recipe.throw_force(scenario["target_distance"])
    assert env.distribution.ang_jit == scenario["launch_angle_jitter_deg"]


@pytest.mark.parametrize("updates", [
    {"catch_aim_mode": "unknown"}, {"catch_match": .1}, {"catch_lead_s": -.1},
    {"catch_offset_scale": 2.}, {"catch_aim_alpha": 0.},
    {"catch_ik_iterations": 0},
    {"catch_ik_damping": 1.}, {"throw_force_gain_n_per_m": -31.},
    {"throw_radius_gain_m_per_m": 2.},
])
def test_improved_recipe_rejects_invalid_values(updates):
    with pytest.raises(ValueError):
        replace(ImprovedRecipe(), **updates)
