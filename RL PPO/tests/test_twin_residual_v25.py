from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from teacher_rl import env as _bootstrap  # installs the sibling CAN source paths
from teacher_rl.contextual_env import CONTEXT_NAMES
from teacher_rl.generalized_teacher import make_scenario_manifest
from teacher_rl.twin_gate import ConservativeGate
from teacher_rl.twin_residual_env import (HOLD_ACTION, JOINT_ACTIONS,
    OBSERVATION_NAMES, TwinResidualConfig, TwinResidualEnv)
from teacher_rl.twin_residual_model import OBS_SIZE, TwinResidualPolicy
from teacher_rl.twin_residual_rl import (_assert_disjoint, _finish_paired,
    _hybrid_report, _branch_score)


def _result(*, captured=True, released=True, hit=True, utility_error=.05,
            release_s=8.):
    return dict(captured=captured, released=released, hit15=hit, hit30=hit,
        grip_broken=False, max_joint_deg=25., landing_error_m=(utility_error if released else None),
        catch_to_release_s=(release_s if released else None), catch_to_orbit_s=.01,
        longest_postcatch_pause_s=.02, contact_peak_n=20., weld_peak_n=100.,
        impact_weld_impulse_ns=1., relative_capture_speed_m_s=2.,
        pressure_integral_psi_s=100., contact_impulse_ns=.1,
        calibrated_release=dict(min_calibrated_error_m=.05))


def test_v25_has_joint_actions_and_no_release_head():
    model = TwinResidualPolicy(hidden=16)
    assert len(OBSERVATION_NAMES) == OBS_SIZE == 59
    assert len(CONTEXT_NAMES) == 39
    assert len(JOINT_ACTIONS) == 9
    assert not hasattr(model, "release_head")
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.ones(9, dtype=bool)
    mask[HOLD_ACTION] = False
    action, _, _ = model.act_motion(observation, mask, deterministic=True)
    assert action != HOLD_ACTION


def test_boundary_no_effect_joint_actions_are_masked():
    env = object.__new__(TwinResidualEnv)
    env.continuous = SimpleNamespace(envelope_force_scale=1., envelope_radius_scale=1.)
    env.twin_dynamic = TwinResidualConfig()
    mask = env.valid_action_mask()
    assert not mask[JOINT_ACTIONS.index((-1, -1))]
    assert not mask[JOINT_ACTIONS.index((-1, 0))]
    assert not mask[JOINT_ACTIONS.index((0, -1))]
    assert mask[HOLD_ACTION]
    assert mask[JOINT_ACTIONS.index((1, 1))]


def test_nominal_receives_full_paired_credit_without_sequence_discount():
    observations = [np.zeros(OBS_SIZE, dtype=np.float32) for _ in range(4)]
    samples = [dict(phase=0, value=.1, reward=0., observation=observations[0])]
    samples += [dict(phase=1, value=0., reward=0., observation=value,
                     action=HOLD_ACTION)
                for value in observations[1:]]
    candidate = _result(utility_error=.02, release_s=4.)
    baseline = _result(utility_error=.14, release_s=10.)
    expected = np.clip((
        __import__("teacher_rl.contextual_env", fromlist=["episode_utility"]).episode_utility(candidate)-
        __import__("teacher_rl.contextual_env", fromlist=["episode_utility"]).episode_utility(baseline))/40., -2., 2.)
    _finish_paired(samples, candidate, baseline, .97, .95)
    assert samples[0]["reward"] == pytest.approx(expected)
    assert samples[0]["return_value"] == pytest.approx(expected)
    assert samples[0]["advantage"] == pytest.approx(expected-.1)


def test_zero_takeover_cannot_vacuously_pass():
    scenarios = make_scenario_manifest(5, 880001, 20261130)["scenarios"]
    rows = []
    for scenario in scenarios:
        result = _result()
        result["initial_policy_mode"] = "v15_passthrough"
        rows.append(dict(seed=scenario["seed"], scenario=scenario,
            result=result, baseline=_result(), gate=dict(takeover=False),
            paired_utility_delta=0.))
    report = _hybrid_report(rows, minimum_coverage=0., require_time=False)
    assert report["takeover_episodes"] == 0
    assert not report["checks"]["nonzero_takeover"]
    assert not report["passed"]


class _ConstantMember(nn.Module):
    def __init__(self, delta, logit):
        super().__init__()
        self.delta, self.logit = float(delta), float(logit)

    def forward(self, value):
        return torch.tensor([[self.delta, self.logit]], dtype=torch.float32).repeat(len(value), 1)


def test_gate_uses_lower_confidence_bounds():
    mean = np.zeros(len(CONTEXT_NAMES), dtype=np.float32)
    scale = np.ones_like(mean)
    confident = ConservativeGate([_ConstantMember(.3, 4.) for _ in range(3)],
        mean, scale, delta_threshold=0., success_threshold=.8)
    assert confident.decision(mean)["takeover"]
    uncertain = ConservativeGate([_ConstantMember(.5, 4.), _ConstantMember(-.5, -4.)],
        mean, scale, delta_threshold=0., success_threshold=.5)
    assert not uncertain.decision(mean)["takeover"]


def test_stage_seed_overlap_is_rejected():
    with pytest.raises(ValueError, match="scenario leak"):
        _assert_disjoint(train={1, 2}, validation={2, 3})


def test_branch_score_does_not_reuse_historical_minimum():
    class Snapshot:
        def restore(self):
            env.done = False
            env.obs = {"t_win": 0., "q_meas": np.zeros(12)}
            env.releaser.current_preview = None

    class Releaser:
        def __init__(self):
            self.original = SimpleNamespace(t_release=None)
            self.current_preview = None
            self.audit = {"min_calibrated_error_m": .001}

        def refresh(self, observation):
            pass

    class Env:
        def __init__(self):
            self.releaser = Releaser()
            self.obs = {"t_win": 0., "q_meas": np.zeros(12)}
            self.done = False

        def apply_joint_action(self, action):
            self.action = action

        def dynamic_due(self):
            return False

        def step(self, action):
            self.obs["t_win"] = .1
            self.releaser.current_preview = {"calibrated_error_m": .2}
            self.done = True
            result = dict(released=False, hit15=False, landing_error_m=None,
                          max_joint_deg=10., grip_broken=False)
            return None, 0., True, result

    env = Env()
    score = _branch_score(env, Snapshot(), HOLD_ACTION, .5)
    assert score["minimum_calibrated_error_m"] == pytest.approx(.2)
