from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_ppo.ppo import PPOConfig, RolloutBuffer
from teacher_rl.data import rollout
from teacher_rl.env import OBS_SIZE
from teacher_rl.generalized_teacher import make_scenario_manifest
from teacher_rl.improved_teacher import ImprovedRecipe, ImprovedTeacherEnv
from teacher_rl.improved_rl import (compare, ensure_disjoint, episode,
    selection_score, training_scenarios, update_policy)
from teacher_rl.model import IntentPolicy, load_checkpoint, save_checkpoint
from teacher_rl.residual_env import QualityConfig, ResidualEnv, policy_mask, quality_cost


def test_quality_penalty_is_bounded_and_failure_remains_worse():
    config = QualityConfig()
    worst = dict(weld_peak_n=1e6, impact_weld_impulse_ns=1e6,
        relative_capture_speed_m_s=1e6, pressure_integral_psi_s=1e6, contact_impulse_ns=1e6)
    assert sum(quality_cost(worst, config).values()) == pytest.approx(6.)
    assert sum(quality_cost(worst, QualityConfig(weight=0)).values()) == 0
    assert sum(quality_cost(dict(relative_capture_speed_m_s=None), config).values()) == 0
    for kwargs in (dict(weight=-1), dict(weight=2), dict(pressure=float('nan')), dict(weld_peak=10)):
        with pytest.raises(ValueError):
            QualityConfig(**kwargs)


def test_sampler_is_repeatable_reachable_and_retains_full_range():
    rows = training_scenarios(80, 100001, 100, .25, .315)
    assert rows == training_scenarios(80, 100001, 100, .25, .315)
    focused = [r for r in rows if r['sampling_group'] == 'mid_speed']
    assert len(focused) == 20
    assert all(5 <= r['launch_speed'] <= 5.25 for r in focused)
    assert min(r['launch_speed'] for r in rows) < 4.75
    assert max(r['launch_speed'] for r in rows) > 5.25
    assert len({r['seed'] for r in rows}) == 80
    from teacher_rl.reachable_collection import _reachable
    assert all(_reachable(r['launch_speed'], r['launch_distance'], -.185) for r in rows)
    with pytest.raises(ValueError, match='overlap'):
        ensure_disjoint(dict(demo=[1, 2], validation=[2, 3]))


def _row(seed, **updates):
    scenario = dict(seed=seed, launch_speed=5., launch_distance=2., target_distance=1.5, target_azimuth=20.)
    result = dict(captured=True, released=True, hit15=True, hit30=True,
        landing_error_m=.05, weld_peak_n=400., impact_weld_impulse_ns=3.,
        relative_capture_speed_m_s=4.6, pressure_integral_psi_s=270.,
        contact_impulse_ns=.1, contact_peak_n=10.)
    result.update(updates)
    return dict(seed=seed, scenario=scenario, result=result)


def test_selection_rejects_failed_tasks_and_worse_precision():
    baseline = [_row(i) for i in range(50)]
    softer = [_row(i, weld_peak_n=320., impact_weld_impulse_ns=2.4) for i in range(50)]
    report = compare(softer, baseline)
    assert report['eligible'] and report['quality_improved']
    assert selection_score(report) > selection_score(compare(baseline, baseline))
    lost = softer[:47] + [_row(i, captured=False, released=False, hit15=False,
        hit30=False, landing_error_m=None, weld_peak_n=0.) for i in range(47, 50)]
    assert not compare(lost, baseline)['eligible']
    inaccurate = [_row(i, landing_error_m=.065, weld_peak_n=100.) for i in range(50)]
    assert not compare(inaccurate, baseline)['eligible']
    assert not compare([_row(i, pressure_integral_psi_s=300.) for i in range(50)], baseline)['eligible']
    with pytest.raises(ValueError, match='matching'):
        compare(softer[:-1], baseline)


def test_ppo_updates_actor_and_critic_but_not_anchor_or_normalization():
    torch.set_num_threads(1)
    torch.manual_seed(55)
    model, anchor = IntentPolicy(), IntentPolicy()
    anchor.load_state_dict(model.state_dict())
    frozen = {k: v.clone() for k, v in anchor.state_dict().items()}
    initial_actor = model.actor[-1].weight.detach().clone()
    initial_critic = model.critic[-1].weight.detach().clone()
    buffer = RolloutBuffer(12, OBS_SIZE, 10)
    masks = policy_mask(np.ones(10, dtype=np.float32))
    assert list(np.flatnonzero(masks == 0)) == [5, 6]
    for i in range(12):
        obs = torch.randn(1, OBS_SIZE) * .01
        mask = np.zeros(10, dtype=np.float32) if i == 11 else masks
        with torch.no_grad():
            _, raw, lp, value = model.act(obs, action_mask=torch.tensor(mask))
        buffer.add(obs[0], raw[0], lp.item(), .2*i, i in (5, 11), value.item(), mask)
    config = PPOConfig(update_epochs=2, minibatch_size=4)
    buffer.finish(0, config)
    data = (torch.tensor(buffer.observations), torch.zeros(12, 10),
        torch.tensor(buffer.action_masks), torch.tensor([0]*6+[3]*6), torch.ones(12))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    metrics = update_policy(model, anchor, optimizer, buffer, config, data, 1., .05)
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(initial_actor, model.actor[-1].weight)
    assert not torch.equal(initial_critic, model.critic[-1].weight)
    for key, value in anchor.state_dict().items():
        torch.testing.assert_close(value, frozen[key])
    torch.testing.assert_close(model.obs_mean, frozen['obs_mean'])
    torch.testing.assert_close(model.obs_std, frozen['obs_std'])


def test_zero_quality_preserves_v15_full_closed_loop():
    torch.set_num_threads(1)
    recipe = ImprovedRecipe(catch_match=.22, throw_force_gain_n_per_m=0., throw_radius_gain_m_per_m=.2)
    scenario = make_scenario_manifest(5, 93001)['scenarios'][0]
    model = IntentPolicy()
    original, expected = rollout(ImprovedTeacherEnv(scenario, recipe), scenario['seed'], model)
    residual, actual = rollout(ResidualEnv(scenario, recipe, QualityConfig(weight=0)), scenario['seed'], model)
    np.testing.assert_array_equal(original['trace150'], residual['trace150'])
    np.testing.assert_array_equal(original['observations'], residual['observations'])
    np.testing.assert_array_equal(original['rewards'], residual['rewards'])
    assert actual['landing_error_m'] == expected['landing_error_m']


def test_episode_worker_uses_model_schema_and_preserves_terminal_gae(monkeypatch):
    import teacher_rl.improved_rl as module
    torch.set_num_threads(1)
    model = IntentPolicy()
    monkeypatch.setattr(module, 'load_checkpoint', lambda path, expected_schema: (model, {}))

    class FakeEnv:
        def __init__(self, *args):
            self.steps = 0

        def reset(self, seed):
            return np.zeros(OBS_SIZE, dtype=np.float32)

        def mask(self):
            return np.ones(10, dtype=np.float32)

        def step(self, action):
            self.steps += 1
            return np.zeros(OBS_SIZE, dtype=np.float32), 1., self.steps == 2, {'hit15': True}

    monkeypatch.setattr(module, 'ResidualEnv', FakeEnv)
    row = episode(({'seed': 42}, asdict(ImprovedRecipe()), 'policy.pt', 'schema',
        asdict(QualityConfig()), True, 99))
    assert len(row['samples']) == 2
    assert row['samples'][-1][4] is True
    assert row['samples'][0][6][5] == row['samples'][0][6][6] == 0


def test_resume_matches_uninterrupted_updates_and_rejects_contract_change(tmp_path, monkeypatch):
    import random
    from concurrent.futures import Future
    import teacher_rl.improved_rl as module
    from teacher_rl.improved_bc import validate_dataset
    from teacher_rl.improved_teacher import IMPROVED_SCHEMA, improved_fingerprint
    from teacher_rl.residual_env import RL_SCHEMA
    from test_improved_bc import _dataset

    torch.set_num_threads(1)
    _dataset(tmp_path / 'data')
    validated = validate_dataset(tmp_path / 'data')
    anchor = IntentPolicy()
    anchor.schema = IMPROVED_SCHEMA
    init = tmp_path / 'bc.pt'
    save_checkpoint(init, anchor, kind='v15_bc', source_hash=improved_fingerprint(),
        dataset=validated['inventory'], dataset_contract=validated['contract'],
        recipe=validated['contract']['recipe'], demonstration_seeds=validated['demonstration_seeds'],
        selection_seeds=[])

    class ImmediatePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, job):
            future = Future()
            future.set_result(function(job))
            return future

    def fake_episode(job):
        scenario, _, checkpoint, schema, _, stochastic, seed = job
        result = _row(scenario['seed'])
        result['scenario'] = scenario
        result['wall_s'] = 0.
        result['samples'] = []
        if not stochastic:
            return result
        rng = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            model, _ = load_checkpoint(checkpoint, expected_schema=schema)
            for i in range(4):
                obs = torch.full((1, OBS_SIZE), i * .01)
                mask = policy_mask(np.ones(10, dtype=np.float32))
                with torch.no_grad():
                    _, raw, lp, value = model.act(obs, action_mask=torch.tensor(mask))
                result['samples'].append((obs[0].numpy(), raw[0].numpy(), lp.item(),
                    float(i), i == 3, value.item(), mask))
        finally:
            torch.set_rng_state(rng)
        return result

    monkeypatch.setattr(module, 'ProcessPoolExecutor', ImmediatePool)
    monkeypatch.setattr(module, 'episode', fake_episode)
    monkeypatch.setattr(module, '_grasp_height', lambda *args: .315)
    args = SimpleNamespace(init=init, data=tmp_path/'data', out=tmp_path/'full', resume=None,
        updates=2, episodes_per_update=2, workers=1, threads=1, eval_every=1,
        eval_episodes=5, eval_seed=2000001, seed=100001, design_seed=101,
        random_seed=21, lr=3e-5, critic_lr=3e-4, bc_weight=1., kl_weight=.05,
        quality_weight=1., focus_fraction=.25)

    def seed_all():
        torch.manual_seed(21)
        np.random.seed(21)
        random.seed(21)

    seed_all()
    module.train(args)
    args.out, args.updates = tmp_path/'resumed', 1
    seed_all()
    module.train(args)
    args.resume, args.updates = args.out/'ppo_latest.pt', 2
    module.train(args)
    full, _ = load_checkpoint(tmp_path/'full'/'ppo_latest.pt', expected_schema=RL_SCHEMA)
    resumed, checkpoint = load_checkpoint(args.resume, expected_schema=RL_SCHEMA)
    assert checkpoint['update'] == 2
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    args.quality_weight = .5
    with pytest.raises(ValueError, match='configuration changed'):
        module.train(args)
