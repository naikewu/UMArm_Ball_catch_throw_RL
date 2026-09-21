import numpy as np
import torch

from teacher_rl.env import ACTOR_SIZE, OBS_SIZE, LIMITS, bounded_intent, TaskConfig, TeacherEnv
from teacher_rl.model import IntentPolicy, save_checkpoint, load_checkpoint
from teacher_rl.__main__ import phase_weights, ppo_update
from rl_ppo.ppo import RolloutBuffer, PPOConfig
from teacher_rl.data import dataset_inventory, write_json, collect_one


def test_intent_limits_and_disabled_phases():
    teacher = np.linspace(-.3, .3, 10)
    mask = np.array([1, 0]*5)
    np.testing.assert_allclose(bounded_intent(teacher, teacher, mask, 1), teacher)
    applied = bounded_intent(np.ones(10), teacher, mask, .25)
    assert np.all(np.abs(applied-teacher) <= LIMITS*.25*mask + 1e-9)
    np.testing.assert_allclose(bounded_intent(-np.ones(10), teacher, mask, 0), teacher)


def test_actor_cannot_see_privileged_state(tmp_path):
    model = IntentPolicy()
    obs = torch.zeros(2, OBS_SIZE)
    obs[1, ACTOR_SIZE:] = 5
    torch.testing.assert_close(model.distribution(obs).mean[0], model.distribution(obs).mean[1])
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model)
    restored, _ = load_checkpoint(path)
    torch.testing.assert_close(restored.distribution(obs).mean, model.distribution(obs).mean)


def test_phase_balancing_and_ppo_anchor():
    torch.set_num_threads(1)
    model = IntentPolicy()
    anchor = IntentPolicy()
    anchor.load_state_dict(model.state_dict())
    cfg = PPOConfig(update_epochs=1, minibatch_size=4)
    buffer = RolloutBuffer(8, OBS_SIZE, 10)
    for i in range(8):
        obs = np.full(OBS_SIZE, .01*i, dtype=np.float32)
        mask = np.array([1]*5+[0]*5, dtype=np.float32)
        with torch.no_grad():
            _, raw, lp, value = model.act(torch.tensor(obs[None]), action_mask=torch.tensor(mask))
        buffer.add(obs, raw[0], lp.item(), .5, i==7, value.item(), mask)
    buffer.finish(0, cfg)
    data = (torch.tensor(buffer.observations), torch.zeros(8,10),
            torch.tensor(buffer.action_masks), torch.tensor([0]*6+[3]*2), torch.ones(8))
    weights = phase_weights(data)
    torch.testing.assert_close(weights[:6].sum(), weights[6:].sum())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = ppo_update(model, anchor, optimizer, buffer, cfg, data, 1., .05)
    assert all(np.isfinite(v) for v in metrics.values())


def test_environment_sensor_observation_and_teacher_zero_residual():
    env = TeacherEnv(TaskConfig(authority=0.))
    obs = env.reset(401)
    assert obs.shape == (OBS_SIZE,)
    for _ in range(3):
        teacher = env.teacher_action()
        obs, reward, done, info = env.step(teacher)
        np.testing.assert_allclose(env.last_intent, teacher)
        assert np.isfinite(obs).all() and np.isfinite(reward) and not done
    assert not info["captured"]


def test_inventory_ignores_incomplete_episodes_and_detects_changes(tmp_path):
    before = dataset_inventory(tmp_path)
    write_json(tmp_path / "episode_401.json", {"seed": 401, "samples": 3})
    complete = dataset_inventory(tmp_path)
    assert complete["episodes"] == 1
    assert complete["manifest_sha256"] != before["manifest_sha256"]
    write_json(tmp_path / "episode_402.tmp.json", {"seed": 402})
    assert dataset_inventory(tmp_path) == complete
    write_json(tmp_path / "episode_401.json", {"seed": 401, "samples": 4})
    assert dataset_inventory(tmp_path)["manifest_sha256"] != complete["manifest_sha256"]


def test_collection_reuses_completed_episode(tmp_path, monkeypatch):
    import teacher_rl.data as data
    from dataclasses import asdict

    class FakeEnv:
        def __init__(self, config):
            self.plant = type("Plant", (), {"log_events": []})()
            self.kw = type("Twin", (), {"provenance": {}})()

    calls = []

    def fake_rollout(env, seed, policy, beta, perturb):
        calls.append(seed)
        return {"observations": np.zeros((3, OBS_SIZE))}, {"captured": True}

    monkeypatch.setattr(data, "TeacherEnv", FakeEnv)
    monkeypatch.setattr(data, "rollout", fake_rollout)
    monkeypatch.setattr(data, "fingerprint", lambda: "test-source")
    job = (405, asdict(TaskConfig()), str(tmp_path), None, 0., 0.)
    first = collect_one(job)
    assert first == collect_one(job)
    assert first["split"] == "validation"
    assert calls == [405]
