from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_rl.buffered_env import (ACTION_NAMES, ACTOR_DIM, OBS_DIM, BufferedConfig,
    BufferedEnv, BufferedHandover, _CatchPressure, buffer_weight, gain_scale)
from teacher_rl.buffered_model import ResidualPolicy, save_policy, load_policy, update_policy
from teacher_rl import buffered_rl
from teacher_rl.soft_env import SoftRecipe
from rl_ppo.ppo import PPOConfig, RolloutBuffer


def control_env():
    env = object.__new__(BufferedEnv)
    env.continuous = BufferedConfig()
    env.last_residual = np.zeros(7, dtype=np.float32)
    env.filtered_residual = np.zeros(7, dtype=np.float32)
    env.last_control_time = None
    env.improved_recipe = SimpleNamespace(catch_match=.22)
    env.recipe = SoftRecipe(match=.22)
    env.base_gains = {"kp": np.array(25.), "kd": np.array(1.), "ki": np.array(15.)}
    env.catch = SimpleNamespace()
    env.handover = SimpleNamespace(t_hand=None, blend_used_s=.2)
    env.capture_time = env.release_time = None
    env.obs = {"t_win": 0.}
    return env


@pytest.mark.parametrize("kwargs", [dict(pressure_authority_psi=4.), dict(filter_s=0.),
    dict(buffer_s=.15, buffer_fade_s=.15), dict(gain_scale=.8), dict(pressure_drop_psi=1.),
    dict(predict_ball_motion=True), dict(match_authority=float("nan"))])
def test_config_rejects_ambiguous_or_unsafe_controls(kwargs):
    with pytest.raises(ValueError):
        BufferedConfig(**kwargs)


@pytest.mark.parametrize("sign", [-1., 0., 1.])
def test_signed_pressure_has_no_dead_half_and_restores_schedule(sign):
    class Catch:
        p_hold = 18.
        def command(self, obs):
            return np.full(24, self.p_hold), {}
    env = control_env()
    env.filtered_residual[1] = sign
    env.soft_window = lambda t: 1.
    catch = Catch()
    command, _ = _CatchPressure(env, catch).command({"t_win": 1.})
    assert command.mean() == 18. + sign
    assert catch.p_hold == 18.


def test_catch_kp_kd_are_independent_and_ki_is_fixed():
    env = control_env()
    env.last_residual[2] = -1.
    for i in range(150):
        env.prepare_control_tick({"t_win": i/150.})
    assert env.catch.kp == pytest.approx(25.*.7, rel=1e-5)
    assert env.catch.kd == 1.
    assert env.catch.ki == 15.
    env.last_residual[3] = 1.
    for i in range(150,300):
        env.prepare_control_tick({"t_win": i/150.})
    assert env.catch.kp == pytest.approx(25.*.7, rel=1e-5)
    assert env.catch.kd == pytest.approx(1.3, rel=1e-5)
    assert env.catch.ki == 15.


def test_residual_filter_rate_and_zero_action():
    env = control_env()
    env.prepare_control_tick({"t_win": 0.})
    assert not env.filtered_residual.any()
    assert env.catch.kp == 25.
    env.last_residual[[1,5]] = -1.
    old = env.filtered_residual.copy()
    env.prepare_control_tick({"t_win": 1/150.})
    assert np.all(env.filtered_residual[[1,5]] < 0.)
    assert np.max(np.abs(env.filtered_residual-old))*env.continuous.pressure_authority_psi <= 20/150.+1e-7


def test_phase_masks_prepare_then_update_then_freeze_buffer():
    env = control_env()
    assert np.array_equal(env.residual_mask(), np.ones(7))
    env.capture_time = 1.
    env.handover.t_hand = 1.
    env.obs["t_win"] = 1.1
    assert np.array_equal(env.residual_mask(), [0,0,0,0,0,1,1])
    env._set_residual(np.ones(7))
    assert np.array_equal(env.last_residual, [0,0,0,0,0,1,1])
    env.obs["t_win"] = 1.25
    env._set_residual(-np.ones(7))
    assert not env.residual_mask().any()
    assert np.array_equal(env.last_residual, [0,0,0,0,0,1,1])
    assert buffer_weight(.20,env.continuous) == pytest.approx(1.)
    assert buffer_weight(.25,env.continuous) == pytest.approx(.5)
    assert buffer_weight(.30,env.continuous) == 0.


def toy_handover(pressure_action, damping_action):
    class Model:
        D = np.zeros((24,12))
        def hold_psi(self,p):
            return np.repeat(p,2)
        def jac(self,q):
            return np.array([.2,0.,.4]), np.eye(3,12), None
    class Drive:
        r_final = .62
        c0 = np.array([0.,0.,.4])
        def __init__(self):
            self.set_p0_pair(np.full(12,16.))
        def set_p0_pair(self,p):
            self.p0 = np.asarray(p).copy()
            self.kp, self.kd = 2.*self.p0, .2*self.p0
        def reset(self):
            self.st = {}
        def psi(self,t,t0,q,qd):
            self.st.update(r=.2,F_tip=np.ones(3))
            p = np.repeat(self.p0,2)
            p[::2] += self.kd
            p[1::2] -= self.kd
            return p
    class Catch:
        def idle_psi(self):
            return np.full(24,16.)
        def stop(self,*args,**kwargs):
            raise AssertionError("No stop allowed")
        brake = stop
    env = SimpleNamespace(filtered_residual=np.array([0,0,0,0,0,pressure_action,damping_action]),
        prepare_control_tick=lambda obs: None)
    drive = Drive()
    controller = BufferedHandover(env,Catch(),drive,Model(),BufferedConfig())
    obs = dict(t_win=1.,held=True,q_meas=np.zeros(12),qd_hat=np.ones(12))
    controller.command(obs)
    obs["t_win"] = 1.1
    pressure,_ = controller.command(obs)
    assert np.all(drive.p0 == 16.)
    assert np.allclose(drive.kd,3.2)
    return controller,pressure


def test_buffer_reaches_orbit_pressure_and_damping_without_accumulation():
    _,zero = toy_handover(0.,0.)
    controller,plus = toy_handover(1.,1.)
    assert plus.mean() > zero.mean()
    assert (plus[::2]-plus[1::2]).mean() > (zero[::2]-zero[1::2]).mean()
    obs = dict(t_win=1.4,held=True,q_meas=np.zeros(12),qd_hat=np.ones(12))
    controller.command(obs)
    assert controller.env.effective_buffer_pressure == 0.
    assert controller.env.effective_buffer_kd == 1.
    assert controller.t_hand == 1.


def test_actor_privileged_isolation_and_new_checkpoint_schema(tmp_path):
    torch.set_num_threads(1)
    model = ResidualPolicy()
    obs = torch.randn(3,OBS_DIM)
    assert model.distribution(obs).mean.shape == (3,7)
    with torch.no_grad():
        model.actor[-1].weight.normal_()
    changed = obs.clone()
    changed[:,ACTOR_DIM:] += 100.
    assert torch.equal(model.distribution(obs).mean,model.distribution(changed).mean)
    path = tmp_path/"policy.pt"
    save_policy(path,model,source_hash="s",anchor_sha256="a")
    assert load_policy(path,"s","a")[0].actor[-1].out_features == len(ACTION_NAMES)
    with pytest.raises(ValueError):
        load_policy(path,"old_source","a")
    from teacher_rl.continuous_model import load_policy as old_loader
    with pytest.raises(ValueError):
        old_loader(path,"s","a")


def test_ppo_updates_with_buffer_only_samples():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = ResidualPolicy()
    buffer = RolloutBuffer(16,OBS_DIM,7)
    mask = np.array([0,0,0,0,0,1,1], dtype=np.float32)
    for index in range(16):
        obs = torch.randn(1,OBS_DIM)
        with torch.no_grad():
            _,raw,lp,value = model.act(obs,action_mask=torch.tensor(mask))
        buffer.add(obs[0],raw[0],lp.item(),float(index%3),index%8==7,value.item(),mask)
    config = PPOConfig(update_epochs=1,minibatch_size=8,target_kl=1.)
    buffer.finish(0.,config)
    metrics = update_policy(model,torch.optim.Adam(model.parameters(),lr=1e-4),buffer,config)
    assert metrics["active_samples"] == 16
    assert metrics["actor_grad_norm"] > 0.
    assert np.isfinite(metrics["explained_variance"])


def test_selection_checks_p95_and_primary_impact_separately():
    report = dict(eligible=True,task_nonregression=True,impact_peak_ratio=.99,
        impact_impulse_ratio=.99,impact_peak_p95_ratio=1.)
    assert buffered_rl.candidate_eligible(report)
    assert not buffered_rl.candidate_eligible(dict(report,impact_peak_p95_ratio=1.01))
    assert not buffered_rl.candidate_eligible(dict(report,impact_impulse_ratio=1.001))


def test_gain_scale_endpoints():
    assert gain_scale(-1.,.3,.1) == .7
    assert gain_scale(0.,.3,.1) == 1.
    assert gain_scale(1.,.3,.1) == 1.1


def test_smoke_cannot_be_used_for_long_training(monkeypatch):
    monkeypatch.setattr("sys.argv", ["buffered_rl", "train", "--config", "absent.json",
        "--smoke", "--updates", "3"])
    with pytest.raises(SystemExit) as error:
        buffered_rl.main()
    assert error.value.code == 2


def test_formal_training_rejects_unapproved_baseline_before_rollout(monkeypatch):
    monkeypatch.setattr(buffered_rl,"load_anchor",lambda path:(None,{}))
    monkeypatch.setattr(buffered_rl,"source_hash",lambda:"s")
    monkeypatch.setattr(buffered_rl,"file_hash",lambda path:"a")
    monkeypatch.setattr(buffered_rl,"read_json",lambda path:dict(eligible=False,schema=buffered_rl.SCHEMA,
        pilot_source_hash="s",anchor_sha256="a"))
    with pytest.raises(ValueError,match="matching source"):
        buffered_rl.train(SimpleNamespace(init=None,config=None,smoke=False))


def test_primary_metrics_include_captured_throw_failures(monkeypatch):
    def comparison(rows,baseline):
        return dict(summary=dict(hit15=1,captured=2,released=2),baseline=dict(hit15=1,captured=2,released=2),
            checks=dict(task=True),eligible=True)
    monkeypatch.setattr(buffered_rl,"compare",comparison)
    common = dict(captured=True,continuous_config={},catch_to_orbit_s=.01,longest_postcatch_pause_s=.02,
        max_joint_deg=35.,impact_weld_peak_n=400.,impact_weld_impulse_ns=3.,buffer_weld_peak_n=20.,
        buffer_weld_impulse_ns=1.)
    baseline = [dict(seed=1,result=dict(common,hit15=True)),dict(seed=2,result=dict(common,hit15=False))]
    rows = [baseline[0],dict(seed=2,result=dict(baseline[1]["result"],impact_weld_peak_n=500.,buffer_weld_impulse_ns=2.))]
    result = buffered_rl.report(rows,baseline)
    assert result["paired_captures"] == 2
    assert result["impact_peak_ratio"] == 1.125
    assert not result["checks"]["buffer_weld_impulse_ns_guard"]
    assert not result["eligible"]


def test_seed_isolation_includes_v16_and_v17(tmp_path,monkeypatch):
    from teacher_rl.data import write_json
    monkeypatch.setattr(buffered_rl,"DEFAULT_OUT",tmp_path)
    monkeypatch.setattr(buffered_rl,"DEFAULT_RL_OUT",tmp_path/"absent")
    monkeypatch.setattr(buffered_rl,"v16_previous_seeds",lambda:{10,20})
    directory = tmp_path/"pilot"
    directory.mkdir()
    write_json(directory/"pilot_contract.json",dict(manifest=dict(scenarios=[dict(seed=30)])))
    assert buffered_rl.previous_seeds() == {10,20,30}


def test_reset_clears_filter_buffer_and_recipe_before_rebuilding(monkeypatch):
    from teacher_rl.improved_teacher import ImprovedTeacherEnv
    class StopBeforePhysics(Exception):
        pass
    def reset(env,seed):
        assert not env._ready
        assert env.recipe == SoftRecipe(match=.22)
        assert not env.last_residual.any()
        assert not env.filtered_residual.any()
        assert env.buffer_peak == env.buffer_impulse == 0.
        assert env.effective_catch_pressure == env.effective_buffer_pressure == 0.
        assert env.effective_buffer_kd == 1.
        assert env.control_trace == []
        assert env.last_control_time is None
        raise StopBeforePhysics
    monkeypatch.setattr(ImprovedTeacherEnv,"reset",reset)
    env = control_env()
    env.last_residual[:] = 1.
    env.filtered_residual[:] = -1.
    env.recipe = SoftRecipe(match=.3,catch_gain_scale=.6,pressure_drop_psi=2.)
    env.buffer_peak = env.buffer_impulse = 100.
    with pytest.raises(StopBeforePhysics):
        env.reset(17)


@pytest.mark.parametrize("action", [np.zeros(4),np.full(7,np.nan),np.full(7,1.01)])
def test_bad_actions_cannot_change_control_state(action):
    env = control_env()
    with pytest.raises(ValueError):
        env._set_residual(action)
    assert not env.last_residual.any()
    assert not env.filtered_residual.any()


def test_zero_action_restores_gains_without_multiplicative_drift():
    env = control_env()
    env.last_residual[2:4] = -1.
    for i in range(150):
        env.prepare_control_tick({"t_win":i/150.})
    env.last_residual[:] = 0.
    for i in range(150,450):
        env.prepare_control_tick({"t_win":i/150.})
    assert env.catch.kp == pytest.approx(25.,abs=1e-6)
    assert env.catch.kd == pytest.approx(1.,abs=1e-6)
    assert env.catch.ki == 15.
