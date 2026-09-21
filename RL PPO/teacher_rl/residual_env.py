"""V15 reward instrumentation; the validated teacher dynamics stay intact."""
from dataclasses import asdict, dataclass

import numpy as np

from .improved_teacher import ImprovedTeacherEnv


RL_SCHEMA = "can_generalized_residual_v15"
FIXED_ACTIONS = (5, 6)


@dataclass(frozen=True)
class QualityConfig:
    weight: float = 1.
    weld_peak: float = .8
    impact_impulse: float = .6
    relative_speed: float = .8
    pressure: float = .4
    contact_impulse: float = .4

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("quality weights must be finite and nonnegative")
        if self.weight > 1.:
            raise ValueError("quality weight must be <= 1 for success-first training")
        if sum(v for k, v in asdict(self).items() if k != "weight") > 3.:
            raise ValueError("quality coefficients must sum to <= 3")


def quality_cost(summary, config):
    scales = (("weld_peak", "weld_peak_n", 400.),
        ("impact_impulse", "impact_weld_impulse_ns", 3.),
        ("relative_speed", "relative_capture_speed_m_s", 5.),
        ("pressure", "pressure_integral_psi_s", 270.),
        ("contact_impulse", "contact_impulse_ns", 1.))
    return {name: config.weight * getattr(config, name) *
        float(np.clip((summary.get(key) or 0.) / scale, 0., 2.))
        for name, key, scale in scales}


def policy_mask(mask):
    result = np.asarray(mask, dtype=np.float32).copy()
    result[..., list(FIXED_ACTIONS)] = 0.
    return result


class ResidualEnv(ImprovedTeacherEnv):
    def __init__(self, scenario, recipe, quality=QualityConfig()):
        super().__init__(scenario, recipe)
        self.quality = quality

    def reset(self, seed):
        self.previous_quality_cost = 0.
        self.task_return = self.quality_penalty = 0.
        return super().reset(seed)

    def step(self, action):
        observation, reward, done, summary = super().step(action)
        terms = quality_cost(summary, self.quality)
        cumulative = sum(terms.values())
        penalty = cumulative - self.previous_quality_cost
        self.previous_quality_cost = cumulative
        self.task_return += reward
        self.quality_penalty += penalty
        summary.update(task_return=self.task_return, quality_penalty=self.quality_penalty,
            quality_cost_terms=terms, rl_return=self.task_return-self.quality_penalty)
        return observation, float(reward-penalty), done, summary
