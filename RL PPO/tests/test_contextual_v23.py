from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.contextual_env import (CONTEXT_NAMES, ContextualEnv, ContextualHandover,
    base_config, bounded_parameters, measured_context, normalized_action, episode_utility)
from teacher_rl.trajectory_envelope_rl import EnvelopeHandover


@pytest.mark.parametrize("action", [[np.nan, 0.], [0., np.inf], [0.], [1.01, 0.]])
def test_unsafe_or_malformed_actions_are_rejected(action):
    with pytest.raises(ValueError):
        bounded_parameters(action)


def test_action_boundaries_preserve_validated_envelope():
    for force in [1., 1.2, 1.4]:
        for radius in [1., 1.09, 1.18]:
            parameters = bounded_parameters(normalized_action(force, radius))
            assert parameters == pytest.approx(dict(envelope_force_scale=force,
                envelope_radius_scale=radius))
    from dataclasses import replace
    for action in [np.array([1., 1.], dtype=np.float32), np.array([-1., -1.], dtype=np.float32)]:
        replace(base_config(), **bounded_parameters(action))


def observable_fixture():
    observation = dict(q_meas=np.zeros(12), qd_hat=np.ones(12),
        p_view_pa=np.ones(24) * 6894.757293168, held=True, t_win=1.23,
        ball_hat=dict(pos=np.array([1., 2., 3.]), vel=np.array([4., 5., 6.])))
    model = SimpleNamespace(jac=lambda q: (np.ones(3), np.ones((3, 12)), None))
    return observation, model


def test_context_uses_only_observable_values_and_excludes_seed_and_outcome():
    obs, model = observable_fixture()
    first = measured_context(model, obs, [1.5, .2])
    obs.update(seed=123, actual_landing=[100., 100.], ball_true=[999.] * 3)
    np.testing.assert_array_equal(first, measured_context(model, obs, [1.5, .2]))
    assert first.shape == (len(CONTEXT_NAMES),)
    assert first[-1] == 36.


def test_decision_is_latched_once_at_first_held_control_tick(monkeypatch):
    calls = []
    env = SimpleNamespace(trajectory_decision=None)
    def choose(obs):
        calls.append(obs["t_win"])
        env.trajectory_decision = dict(action=[0., 0.])
    env.choose_trajectory = choose
    handover = object.__new__(ContextualHandover)
    handover.env = env
    monkeypatch.setattr(EnvelopeHandover, "command", lambda self, obs: ("pressure", {}))
    for held, time in [(False, 1.), (True, 1.01), (True, 1.02), (False, 2.)]:
        handover.command(dict(held=held, t_win=time))
    assert calls == [1.01]


def test_selection_updates_both_drive_and_release_configs_without_changing_original():
    obs, model = observable_fixture()
    env = object.__new__(ContextualEnv)
    env.model, env.target = model, np.array([1.5, .2])
    env.selector = lambda context: np.array([1., -1.])
    initial = base_config()
    env.continuous = initial
    env.handover, env.releaser = SimpleNamespace(), SimpleNamespace()
    env.choose_trajectory(obs)
    assert env.handover.config is env.releaser.config is env.continuous
    assert initial.envelope_force_scale == 1.25
    assert env.continuous.envelope_force_scale == pytest.approx(1.4)
    assert env.trajectory_decision["time_s"] == 1.23


def test_predicted_success_cannot_outscore_actual_safe_success():
    safe = dict(captured=True, released=True, hit15=True, grip_broken=False,
        max_joint_deg=33., landing_error_m=.1, catch_to_release_s=15.)
    never_released = dict(safe, released=False, landing_error_m=None,
        calibrated_release=dict(min_calibrated_error_m=0.))
    unsafe = dict(safe, max_joint_deg=38., landing_error_m=0.)
    assert episode_utility(safe) > episode_utility(never_released)
    assert episode_utility(safe) > episode_utility(unsafe)


def test_ancestral_training_and_selection_seeds_cannot_enter_final_test():
    from teacher_rl.contextual_training import assert_fresh
    ancestry = dict(training_seeds=[11], selection_seeds=[12],
        nested=dict(manifest=dict(scenarios=[dict(seed=13)])))
    for seed in [11, 12, 13]:
        with pytest.raises(ValueError, match="overlap"):
            assert_fresh(dict(scenarios=[dict(seed=seed)]), ancestry)
    assert_fresh(dict(scenarios=[dict(seed=14)]), ancestry)


def test_failed_teacher_cannot_start_formal_ppo(tmp_path):
    from teacher_rl.contextual_training import require_gate
    with pytest.raises(ValueError, match="Missing bc-evaluate"):
        require_gate(tmp_path, tmp_path / "teacher.pt", "bc-evaluate")


def test_context_model_roundtrip_preserves_actual_selection(tmp_path):
    from teacher_rl.contextual_model import ContextPolicy, save_model, load_model
    model = ContextPolicy([[-1., -1.], [1., 1.]])
    context = np.linspace(-.1, .1, len(CONTEXT_NAMES)).astype(np.float32)
    action, info = model.decide(context)
    path = tmp_path / "model.pt"
    save_model(path, model, kind="context_bc", contract={})
    loaded, _ = load_model(path)
    following, other = loaded.decide(context)
    np.testing.assert_array_equal(action, following)
    assert other == info


def test_compact_outcome_teacher_reports_interpretable_predictions(tmp_path):
    from teacher_rl.contextual_model import OutcomeTeacher, save_model, load_model
    model = OutcomeTeacher([[-1., -1.], [0., 0.], [1., 1.]])
    assert sum(parameter.numel() for parameter in model.parameters()) < 5000
    context = np.linspace(-.2, .2, len(CONTEXT_NAMES)).astype(np.float32)
    action, info = model.decide(context)
    assert action.shape == (2,)
    assert 0. <= info["predicted_release_probability"] <= 1.
    assert 0. <= info["predicted_hit15_probability"] <= 1.
    path = tmp_path / "outcome.pt"
    save_model(path, model, kind="outcome_teacher", contract={})
    restored, payload = load_model(path)
    np.testing.assert_array_equal(action, restored.decide(context)[0])
    assert payload["kind"] == "outcome_teacher"


def test_constant_contextual_action_reproduces_v22_closed_loop():
    from dataclasses import asdict
    from teacher_rl.contextual_rl import worker, DEFAULT_ANCHOR, DEFAULT_CALIBRATION
    from teacher_rl.envelope_teacher_rl import worker as v22_worker
    from teacher_rl.improved_rl import read_json
    from pathlib import Path
    scenario = read_json(Path("teacher_runs/v22_envelope_teacher/smoke_v1/v15_bc_episodes.json"))[0]["scenario"]
    config = base_config(tolerance=.08)
    anchor, calibration = str(DEFAULT_ANCHOR.resolve()), str(DEFAULT_CALIBRATION.resolve())
    reference = v22_worker((scenario, anchor, calibration, asdict(config)))
    candidate = worker((scenario, anchor, calibration, asdict(config), normalized_action(1.25, 1.)))
    for key in ("captured", "released", "hit15", "grip_broken"):
        assert reference["result"][key] == candidate["result"][key]
    for key in ("max_joint_deg", "landing_error_m", "capture_time", "release_time", "pressure_integral_psi_s"):
        expected = reference["result"][key]
        assert candidate["result"][key] == (None if expected is None else pytest.approx(expected, abs=1e-8))


def test_release_estimator_is_warmed_once_per_tick_before_probe_window():
    from teacher_rl.contextual_calibration import warm_before_release
    calls = []
    class State:
        lead = .01
        def __call__(self, obs):
            calls.append(("state", obs["t"]))
            return np.ones(3), np.ones(3)
    original = SimpleNamespace(t_release=None, ball_state=State(),
        _rates=lambda t, v: calls.append(("rate", t)))
    controller = SimpleNamespace(t_hand=1.)
    obs = dict(t=3.5, t_win=1.5, held=True)
    stamp = warm_before_release(original, controller, obs, 2., None)
    warm_before_release(original, controller, obs, 2., stamp)
    assert calls == [("state", 3.5), ("rate", 1.5)]
    # Once the release window opens the ordinary path owns the update.
    warm_before_release(original, controller, dict(obs, t=5., t_win=3.), 2., stamp)
    assert len(calls) == 2


def test_kernel_release_roundtrip_and_real_feature_schema():
    from teacher_rl.contextual_kernel import fit_kernel, KernelCalibration
    rng = np.random.default_rng(4123)
    features = rng.normal(size=(40, 19))
    target = .02 * features[:, :2] + .005 * features[:, :2] ** 2
    data = [dict(features=x, raw_landing=np.zeros(2), actual_landing=y)
            for x, y in zip(features, target)]
    model = fit_kernel(data, 4., .1)
    restored = KernelCalibration.from_dict(model.to_dict())
    for x in features[:3]:
        np.testing.assert_allclose(model.predict_landing([1., 2.], x),
                                   restored.predict_landing([1., 2.], x), atol=1e-12)
    bad = model.to_dict()
    bad["kernel_length"] = float("nan")
    with pytest.raises(ValueError, match="Invalid kernel"):
        KernelCalibration.from_dict(bad)


def test_ppo_updates_from_recorded_behavior_and_resumes_without_promoting_failed_selection(tmp_path, monkeypatch):
    """Synthetic rollouts test the optimizer/resume contract, not robot success."""
    import torch
    from dataclasses import asdict
    from teacher_rl import contextual_training as training
    from teacher_rl.contextual_model import ContextPolicy, save_model, load_model
    from teacher_rl.improved_teacher import ImprovedRecipe
    torch.manual_seed(819)
    anchor, calibration = tmp_path / "anchor", tmp_path / "calibration"
    anchor.write_text("synthetic anchor", encoding="utf-8")
    calibration.write_text("synthetic calibration", encoding="utf-8")
    checkpoint = tmp_path / "bc.pt"
    initial = ContextPolicy([[-1., -1.], [1., 1.]])
    contract = dict(runtime_hash=training.runtime_hash(), anchor_sha256=training.file_hash(anchor),
                    calibration_sha256=training.file_hash(calibration))
    save_model(checkpoint, initial, kind="context_bc", contract=contract)
    monkeypatch.setattr(training, "require_gate", lambda *args: {"contract": {}})
    monkeypatch.setattr(training, "load_anchor", lambda path: (None, {"recipe": asdict(ImprovedRecipe())}))
    monkeypatch.setattr(training, "scenarios", lambda count, seed, *args: {
        "scenarios": [{"seed": seed + i} for i in range(count)]})
    monkeypatch.setattr(training, "previous_seeds", lambda path: set())
    monkeypatch.setattr(training, "report", lambda rows, baseline: {
        "eligible": False, "task_nonregression": False})
    stochastic_batches = []
    def synthetic_rollout(manifest, anchor, calibration, model_path, output, workers, *, stochastic=False, salt=0):
        model = None if model_path is None else load_model(model_path)[0]
        if stochastic:
            stochastic_batches.append([s["seed"] for s in manifest["scenarios"]])
        rows = []
        for i, scene in enumerate(manifest["scenarios"]):
            torch.manual_seed(scene["seed"] + salt)
            context = np.linspace(-.3, .3, len(CONTEXT_NAMES), dtype=np.float32) + .01 * i
            decision = None if model is None else dict(context=context.tolist(), **model.decide(context, stochastic)[1])
            rows.append(dict(seed=scene["seed"], decision=decision, result={"hit15": False},
                             utility=0. if decision is None else 30. * (2 * decision["index"] - 1)))
        return rows
    monkeypatch.setattr(training, "evaluate", synthetic_rollout)
    args = SimpleNamespace(prerequisite=tmp_path, checkpoint=checkpoint, init=anchor,
        calibration=calibration, out=tmp_path / "ppo", updates=2, episodes=8, seed=91001,
        validation_seed=92001, design_seed=901, workers=1)
    training.ppo(args)
    latest, payload = load_model(args.out / "ppo_latest.pt")
    assert payload["update"] == 2 and payload["optimizer"]["state"]
    assert len(stochastic_batches) == 2
    assert set(stochastic_batches[0]).isdisjoint(stochastic_batches[1])
    assert any(not torch.equal(a, b) for a, b in zip(initial.actor.parameters(), latest.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(initial.critic.parameters(), latest.critic.parameters()))
    assert not (args.out / "ppo_candidate.pt").exists()
    assert not (args.out / "ppo_accepted.pt").exists()
    training.ppo(args)
    assert len(stochastic_batches) == 2
    resumed, _ = load_model(args.out / "ppo_latest.pt")
    for key, value in latest.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
