from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_rl.exploration_env import (ExplorationConfig, ExplorationEnv, ExploringRelease,
    ExploringHandover, ACTOR_DIM, OBS_DIM, CONTROL_COLUMNS, RESIDUAL_SIZE)
from teacher_rl.exploration_model import ResidualPolicy, load_policy, save_policy, update_policy
from teacher_rl.buffered_env import BufferedHandover
from teacher_rl import exploration_rl
from rl_ppo.ppo import PPOConfig, RolloutBuffer


def release_fixture(vote=0.,fresh=True,angle=0.,speed=2.,stage="throw"):
    class Original:
        t_release = None
        target = np.array([1.,0.])
        lat = .02
        tick = 1/150
        sub = 2
        horizon = .01
        depart_fn = None
        m_ball = .5
        v_min = 1.5
        min_elev = -.1
        best = (np.inf,None)
        omega = 0.
        def _rates(self,t,v):
            self.rate_calls = getattr(self,"rate_calls",0)+1
        def predict_from(self,p,v,delay):
            return np.array([1.2,0.,.04]),np.array([speed,0.,.5]),p
    class Estimator:
        lead = .01 if fresh else None
        def __call__(self,obs):
            return np.array([0.,0.,.5]),np.array([speed,0.,.5])
    class Plant:
        sim_now = 5.
        def gripper_open(self,t=None,**kwargs):
            self.commands = getattr(self,"commands",[])+[kwargs]
    original = Original()
    original.ball_state = Estimator()
    env = SimpleNamespace(continuous=ExplorationConfig(stage=stage),handover=SimpleNamespace(t_hand=0.),
        last_residual=np.r_[np.zeros(9),vote])
    obs = dict(t=5.006,t_win=3.006,held=True,q_meas=np.full(12,np.radians(angle)))
    return ExploringRelease(original,env),Plant(),obs


@pytest.mark.parametrize("kwargs",[dict(stage="scratch"),dict(force_authority_n=5.),
    dict(radius_authority_m=.1),dict(abort_joint_deg=50.),dict(vote_threshold=float("nan"))])
def test_configuration_bounds(kwargs):
    with pytest.raises(ValueError):
        ExplorationConfig(**kwargs)


def test_policy_can_release_outside_advisor_tolerance_and_only_once():
    wrapper,plant,obs = release_fixture(vote=.8)
    wrapper.refresh(obs)
    wrapper(plant,obs)
    wrapper(plant,dict(obs,t=5.012,t_win=3.012))
    assert len(plant.commands)==1
    assert plant.commands[0]["due_in_s"]==pytest.approx(.026)
    assert wrapper.state_at_cmd["origin"]=="policy"
    assert wrapper.state_at_cmd["landing_err_pred"]==pytest.approx(.2)
    assert wrapper.counts["policy_releases"]==1


def test_zero_vote_uses_advisor_veto_blocks_and_cache_is_idempotent():
    wrapper,plant,obs = release_fixture()
    wrapper(plant,obs)
    assert not hasattr(plant,"commands")
    assert wrapper.original.rate_calls==1
    wrapper.refresh(obs)
    assert wrapper.original.rate_calls==1
    wrapper.original.predict_from = lambda p,v,delay:(np.array([1.,0.,.04]),np.array([2.,0.,.5]),p)
    wrapper.env.last_residual[9] = -.8
    wrapper(plant,dict(obs,t=5.012,t_win=3.012))
    assert not hasattr(plant,"commands")
    wrapper.env.last_residual[9] = 0.
    wrapper(plant,dict(obs,t=5.018,t_win=3.018))
    assert wrapper.state_at_cmd["origin"]=="advisor"


@pytest.mark.parametrize("kwargs",[dict(fresh=False),dict(angle=37.),dict(speed=.1)])
def test_positive_vote_cannot_bypass_guards(kwargs):
    wrapper,plant,obs = release_fixture(vote=.8,**kwargs)
    wrapper(plant,obs)
    assert not hasattr(plant,"commands")
    assert wrapper.counts["blocked_request_ticks"]==1


def test_no_release_before_earliest_time_or_without_ball():
    wrapper,plant,obs = release_fixture(vote=.8)
    wrapper.env.handover.t_hand = 3.
    wrapper(plant,obs)
    assert not hasattr(plant,"commands")
    wrapper.env.handover.t_hand = 0.
    wrapper(plant,dict(obs,t=5.012,t_win=3.012,held=False))
    assert not hasattr(plant,"commands")


def test_initial_release_noise_is_rare_but_authority_is_preserved():
    model = ResidualPolicy()
    obs = torch.zeros(1,OBS_DIM)
    distribution = model.distribution(obs)
    threshold = np.arctanh(ExplorationConfig().vote_threshold)
    probability = 1.-distribution.cdf(torch.full((1,RESIDUAL_SIZE),threshold))[0,9]
    assert 0. < probability < .001
    wrapper,plant,obs = release_fixture(vote=1.)
    wrapper(plant,obs)
    assert wrapper.counts["policy_releases"]==1


def test_throw_stage_does_not_change_catch_and_flight_masks():
    env = object.__new__(ExplorationEnv)
    env.continuous = ExplorationConfig()
    env.capture_time = env.release_time = None
    env.releaser = SimpleNamespace(t_release=None)
    env.handover = SimpleNamespace(t_hand=None)
    env.obs = dict(t_win=1.)
    assert not env.residual_mask().any()
    env.capture_time = env.handover.t_hand = 1.
    env.obs["t_win"] = 1.1
    assert np.array_equal(env.residual_mask(),[0,0,0,0,0,0,0,1,1,0])
    env.obs["t_win"] = 1.6
    assert np.array_equal(env.residual_mask(),[0,0,0,0,0,0,0,1,1,1])
    env.releaser.t_release = 1.6
    assert not env.residual_mask().any()


def test_orbit_residual_applied_after_nominal_override_and_restored(monkeypatch):
    def parent(controller,obs):
        return controller.drv.F_max,controller.drv.r_final
    monkeypatch.setattr(BufferedHandover,"command",parent)
    controller = object.__new__(ExploringHandover)
    controller.env = SimpleNamespace(throw_force_n=10.,throw_radius_m=.62,
        continuous=ExplorationConfig(),filtered_residual=np.r_[np.zeros(7),1.,-1.,0.])
    controller.drv = SimpleNamespace(F_max=10.,r_final=.62)
    assert controller.command({})==pytest.approx((11.,.60))
    assert controller.drv.F_max==10.
    assert controller.drv.r_final==.62


def test_actor_shapes_privileged_isolation_and_schema(tmp_path):
    torch.set_num_threads(1)
    model = ResidualPolicy()
    obs = torch.randn(3,OBS_DIM)
    assert model.distribution(obs).mean.shape==(3,10)
    assert not model.distribution(obs).mean.any()
    with torch.no_grad():
        model.actor[-1].weight.normal_()
    changed = obs.clone()
    changed[:,ACTOR_DIM:] += 50.
    assert torch.equal(model.distribution(obs).mean,model.distribution(changed).mean)
    path = tmp_path/"policy.pt"
    save_policy(path,model,source_hash="s",anchor_sha256="a")
    assert load_policy(path,"s","a")[0].actor[-1].out_features==10
    from teacher_rl.buffered_model import load_policy as old_load
    with pytest.raises(ValueError):
        old_load(path,"s","a")
    assert len(CONTROL_COLUMNS)==32


def test_ppo_release_samples_have_finite_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(18)
    model = ResidualPolicy()
    buffer = RolloutBuffer(20,OBS_DIM,RESIDUAL_SIZE)
    mask = np.r_[np.zeros(7),np.ones(3)].astype(np.float32)
    for i in range(20):
        obs = torch.randn(1,OBS_DIM)
        with torch.no_grad():
            _,raw,lp,value = model.act(obs,action_mask=torch.tensor(mask))
        buffer.add(obs[0],raw[0],lp.item(),float(i%3),i%10==9,value.item(),mask)
    cfg = PPOConfig(update_epochs=1,minibatch_size=10,target_kl=1.)
    buffer.finish(0.,cfg)
    metrics = update_policy(model,torch.optim.Adam(model.parameters(),lr=1e-4),buffer,cfg)
    assert metrics["active_samples"]==20
    assert np.isfinite(metrics["actor_grad_norm"]) and metrics["actor_grad_norm"]>0.


def test_research_score_penalizes_aborts_before_hits():
    def row(hit,abort):
        return dict(result=dict(hit15=hit,hit30=hit,captured=True,released=True,landing_error_m=.02,
            safety_abort=abort,max_joint_deg=46. if abort else 34.,weld_peak_n=400.,
            impact_weld_impulse_ns=3.,relative_capture_speed_m_s=4.,pressure_integral_psi_s=200.,
            contact_impulse_ns=0.,contact_peak_n=0.))
    assert exploration_rl.research_score([row(False,False)]) > exploration_rl.research_score([row(True,True)])


@pytest.mark.parametrize("arguments",[["explore","--updates","1000"],
    ["explore","--episodes-per-update","2"],["explore","--eval-episodes","2"],
    ["evaluate","--episodes","2"]])
def test_budget_validation(monkeypatch,arguments):
    monkeypatch.setattr("sys.argv",["exploration_rl",*arguments])
    with pytest.raises(SystemExit) as exc:
        exploration_rl.main()
    assert exc.value.code==2
