from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_rl.continuous_env import (ACTOR_DIM, OBS_DIM, ContinuousConfig,
    ContinuousEnv, ContinuousHandover, ContinuousRelease)
from teacher_rl.improved_teacher import ImprovedTeacherEnv
from teacher_rl.soft_env import SoftRecipe
from teacher_rl.continuous_model import ResidualPolicy, load_policy, save_policy, update_policy
from teacher_rl import continuous_rl
from teacher_rl.data import write_json
from teacher_rl.continuous_release import BallTrajectoryPrediction, MeasuredBallState, TimestampedValve
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


def test_residuals_are_inactive_after_handover():
    env = object.__new__(ContinuousEnv)
    env.capture_time = None
    env.handover = SimpleNamespace(t_hand=None,blend_used_s=.2)
    assert np.array_equal(env.residual_mask(),np.ones(4))
    env.handover.t_hand = 1.
    assert not env.residual_mask().any()
    env.handover.t_hand = None
    env.capture_time = 1.
    assert not env.residual_mask().any()
    env.continuous = ContinuousConfig()
    env.improved_recipe = SimpleNamespace(catch_match=.22)
    env.last_residual = np.array([.1,.2,.3,.4],dtype=np.float32)
    env.recipe = SoftRecipe(match=.22)
    env.base_gains = dict(kp=1.,kd=2.,ki=3.)
    env.catch = SimpleNamespace()
    before = env.last_residual.copy()
    env._set_residual(-np.ones(4))
    assert np.array_equal(env.last_residual,before)


def test_reset_restores_nominal_recipe_before_rebuilding_controller(monkeypatch):
    class StopBeforePhysics(Exception):
        pass
    def parent_reset(env,seed):
        assert env.recipe == SoftRecipe(match=.22)
        raise StopBeforePhysics
    monkeypatch.setattr(ImprovedTeacherEnv,'reset',parent_reset)
    env = object.__new__(ContinuousEnv)
    env.recipe = SoftRecipe(match=.3,pressure_drop_psi=2.,catch_gain_scale=.6)
    env.improved_recipe = SimpleNamespace(catch_match=.22)
    env.last_residual = np.ones(4,dtype=np.float32)
    with pytest.raises(StopBeforePhysics):
        env.reset(1)
    assert not env.last_residual.any()


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


def test_selection_never_trades_success_or_impact_for_composite_score():
    result = dict(eligible=True,task_nonregression=True,impact_peak_ratio=.99,impact_impulse_ratio=.99)
    assert continuous_rl.candidate_eligible(result)
    assert not continuous_rl.candidate_eligible(dict(result,task_nonregression=False))
    assert not continuous_rl.candidate_eligible(dict(result,impact_peak_ratio=1.001))
    assert not continuous_rl.candidate_eligible(dict(result,impact_impulse_ratio=1.001))


@pytest.mark.parametrize('delay,pause,eligible',[(.011,.035,True),(.021,.035,False),(.011,.101,False)])
def test_report_rejects_delayed_handover_or_sustained_pause(monkeypatch,delay,pause,eligible):
    def comparison(rows,baseline):
        return dict(summary=dict(hit15=1,captured=1),baseline=dict(hit15=1,captured=1),
            checks=dict(task=True),eligible=True)
    monkeypatch.setattr(continuous_rl,'compare',comparison)
    result = dict(hit15=True,continuous_config={},catch_to_orbit_s=delay,
        longest_postcatch_pause_s=pause,impact_weld_peak_n=400.,impact_weld_impulse_ns=3.)
    rows = [dict(seed=1,result=result)]
    assert continuous_rl.report(rows,rows)['eligible'] is eligible


def test_previous_v16_development_and_smoke_seeds_are_excluded(tmp_path,monkeypatch):
    monkeypatch.setattr(continuous_rl,'DEFAULT_RL_OUT',tmp_path/'absent_v15')
    monkeypatch.setattr(continuous_rl,'DEFAULT_OUT',tmp_path)
    (tmp_path/'pilot').mkdir()
    (tmp_path/'smoke/evaluation_5').mkdir(parents=True)
    write_json(tmp_path/'pilot/pilot_contract.json',dict(manifest=dict(scenarios=[dict(seed=10)])))
    config = dict(validation_manifest=dict(scenarios=[dict(seed=30)]),seed=20,episodes_per_update=5)
    write_json(tmp_path/'smoke/run_config.json',config)
    write_json(tmp_path/'smoke/update_0002.json',{})
    write_json(tmp_path/'smoke/evaluation_5/evaluation_contract.json',dict(manifest=dict(scenarios=[dict(seed=40)])))
    assert continuous_rl.previous_seeds() == {10,30,40,*range(20,30)}
    assert continuous_rl.previous_seeds(tmp_path/'smoke') == {10}


def test_ball_release_estimator_uses_only_timestamped_sensor_frames():
    fallback = (np.ones(3)*99.,np.ones(3)*99.)
    estimator = MeasuredBallState(lambda obs:fallback,16)
    for frame in range(1,17):
        time = (frame-1)/240.
        obs = dict(t=time+.008,ball_frame_id=frame,
            ball_meas=np.array([1.+2.*time+3.*time**2,0.,.4]),
            privileged_ball_velocity=np.full(3,1000.))
        p,v = estimator(obs)
    now = obs['t']
    assert np.allclose(p,[1.+2.*now+3.*now**2,0.,.4],atol=1e-8)
    assert np.allclose(v,[2.+6.*now,0.,0.],atol=1e-8)
    estimator(obs)
    assert len(estimator.fit.buf) == 16
    p,v = estimator(dict(obs,t=obs['t']+.1))
    assert np.array_equal(v,fallback[1])


def test_valve_schedule_matches_observation_clock():
    class Plant:
        sim_now = 2.
        def gripper_open(self,t=None,**kwargs):
            return kwargs
    valve = TimestampedValve(Plant(),2.+1/150)
    assert valve.gripper_open(vent_in_s=.005)['vent_in_s'] == pytest.approx(.005+1/150)
    assert valve.gripper_open(due_in_s=.02)['due_in_s'] == pytest.approx(.02+1/150)
    assert TimestampedValve(Plant(),1.).gripper_open(vent_in_s=.005)['vent_in_s'] == .005


def test_future_release_prediction_keeps_attached_ball_acceleration():
    estimator = MeasuredBallState(lambda obs:(np.zeros(3),np.zeros(3)),16)
    for frame in range(1,17):
        time = (frame-1)/240.
        estimator(dict(t=time+.008,ball_frame_id=frame,
            ball_meas=np.array([1.+2.*time+3.*time**2,0.,.4])))
    releaser = SimpleNamespace(speed_gain=1.,az_bias=0.,elev_bias=0.,z_floor=.04,
        predict_from=lambda p,v,d:(np.zeros(3),np.zeros(3),np.zeros(3)))
    predictor = BallTrajectoryPrediction(estimator,releaser)
    target,velocity,position = predictor(None,None,.02)
    future = 15/240.+.008+.02
    assert np.allclose(position,[1.+2.*future+3.*future**2,0.,.4],atol=1e-8)
    assert np.allclose(velocity,[2.+6.*future,0.,0.],atol=1e-8)
    assert target[0]>position[0]
    assert target[2] == .04
    estimator.lead = None
    assert np.array_equal(predictor(None,None,.02)[0],np.zeros(3))
