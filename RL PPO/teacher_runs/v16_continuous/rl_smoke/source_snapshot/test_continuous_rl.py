from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_rl.continuous_env import (ACTOR_DIM, OBS_DIM, ContinuousConfig,
    ContinuousHandover, ContinuousRelease)
from teacher_rl.continuous_model import ResidualPolicy, load_policy, save_policy, update_policy
from rl_ppo.ppo import PPOConfig, RolloutBuffer


def test_configuration_rejects_unsafe_nonfinite_values():
    for kwargs in (dict(blend_s=0),dict(pressure_drop_psi=10),dict(gain_scale=float('nan')),
            dict(spinup_s=-1),dict(release_min_s=100)):
        with pytest.raises(ValueError):
            ContinuousConfig(**kwargs)


def test_handover_preserves_motion_and_never_calls_stop():
    class Model:
        D = np.zeros((24,12))
        def hold_psi(self, p):
            return np.full(24,16.)
        def jac(self, q):
            return np.array([.2,0.,.4]), np.eye(3,12), None
    class Drive:
        p0 = np.full(12,16.)
        r_final = .62
        c0 = np.array([0.,0.,.4])
        def reset(self):
            self.st = {}
        def psi(self, t, t0, q, qd):
            self.st.update(r=.2, F_tip=np.ones(3))
            return np.full(24,14.)
    class Catch:
        def command(self, obs):
            return np.full(24,18.), {}
        def idle_psi(self):
            return np.full(24,18.)
        def stop(self, *args, **kwargs):
            raise AssertionError("Must not stop before orbit")
        brake = stop
    controller = ContinuousHandover(Catch(),Drive(),Model(),ContinuousConfig())
    obs = dict(t_win=1.,held=False,q_meas=np.zeros(12),qd_hat=np.ones(12))
    controller.command(obs)
    obs.update(t_win=1.01,held=True)
    pressure,_ = controller.command(obs)
    assert controller.t_hand == 1.01
    assert controller.mode == 'throw'
    assert controller.settle_s == 0
    assert np.allclose(controller.drv.st['vel'],1.)
    assert controller.drv.r0 == pytest.approx(.2)
    assert np.allclose(pressure,18.)
    obs['t_win'] += .1
    pressure,_ = controller.command(obs)
    assert np.allclose(pressure,16.)
    obs['t_win'] += .2
    pressure,_ = controller.command(obs)
    assert np.allclose(pressure,14.)


def test_release_gate_replaces_legacy_wait_not_prediction():
    class Original:
        def __call__(self, plant, obs):
            return self.t_min
    original = Original()
    wrapper = ContinuousRelease(original,SimpleNamespace(t_hand=1.),ContinuousConfig())
    wrapper.t_min = 15.
    assert wrapper(None,{}) == 3.5
    assert original.t_min == 3.5


def test_zero_residual_and_no_privileged_actor_leak():
    model = ResidualPolicy()
    obs = torch.randn(5,OBS_DIM)
    assert torch.equal(model.distribution(obs).mean,torch.zeros(5,4))
    with torch.no_grad():
        model.actor[-1].weight.normal_()
    alternative = obs.clone()
    alternative[:,ACTOR_DIM:] += 10.
    assert torch.equal(model.distribution(obs).mean,model.distribution(alternative).mean)


def test_masked_dimensions_do_not_change_probability():
    model = ResidualPolicy()
    obs = torch.zeros(2,OBS_DIM)
    raw = torch.zeros(2,4)
    masks = torch.tensor([[0.,1.,0.,0.],[0.,0.,0.,0.]])
    before = model.evaluate(obs,raw,masks)[0]
    raw[:,0] = 3.
    assert torch.equal(before,model.evaluate(obs,raw,masks)[0])
    assert before[1] == 0


def test_checkpoint_contract_and_update_diagnostics(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = ResidualPolicy()
    buffer = RolloutBuffer(16,OBS_DIM,4)
    for index in range(16):
        obs = torch.randn(1,OBS_DIM)
        with torch.no_grad():
            _,raw,lp,value = model.act(obs,action_mask=torch.ones(4))
        buffer.add(obs[0],raw[0],lp.item(),float(index%3),index%8==7,value.item(),np.ones(4))
    config = PPOConfig(update_epochs=1,minibatch_size=8,target_kl=1.)
    buffer.finish(0.,config)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    result = update_policy(model,optimizer,buffer,config)
    assert result['active_samples'] == 16
    assert np.isfinite(result['explained_variance'])
    assert result['actor_grad_norm']>0
    assert result['critic_grad_norm']>0
    path = tmp_path/'policy.pt'
    save_policy(path,model,source_hash='source',anchor_sha256='anchor')
    loaded,_ = load_policy(path,'source','anchor')
    assert torch.equal(model.actor[-1].weight,loaded.actor[-1].weight)
    with pytest.raises(ValueError):
        load_policy(path,'changed','anchor')
