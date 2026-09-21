"""V19 continuous handover baseline with a proactive joint-range governor."""
from dataclasses import asdict, dataclass

import numpy as np

from .buffered_env import BufferedConfig, BufferedEnv, BufferedHandover
from .continuous_env import ContinuousRelease

SCHEMA = "can_safe_continuous_baseline_v19"


def smoothstep(value):
    value = float(np.clip(value, 0., 1.))
    return value * value * (3. - 2. * value)


@dataclass(frozen=True)
class SafeContinuousConfig(BufferedConfig):
    """A zero-residual continuous controller with only nominal safety shaping."""
    governor_start_deg: float = 30.
    governor_band_deg: float = 8.
    force_start_scale: float = .60
    force_ramp_s: float = .60
    radius_ramp_s: float = .60

    def __post_init__(self):
        super().__post_init__()
        for name, low, high in (
                ("governor_start_deg", 24., 36.), ("governor_band_deg", 3., 12.),
                ("force_start_scale", .30, 1.), ("force_ramp_s", 0., 1.5),
                ("radius_ramp_s", 0., 1.5)):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.governor_start_deg + self.governor_band_deg > 42.:
            raise ValueError("joint governor range must begin before 42 degrees")


class SafeHandover(BufferedHandover):
    """Keep V16's immediate handover while starting its built-in governor earlier."""
    def __init__(self, env, catch, drive, model, config):
        super().__init__(env, catch, drive, model, config)
        self.q_cap = np.radians(config.governor_start_deg)
        self.q_band = np.radians(config.governor_band_deg)

    def command(self, obs):
        cfg = self.config
        nominal_force, nominal_radius = self.drv.F_max, self.drv.r_final
        active = bool(obs["held"]) or self.t_hand is not None
        force_scale = radius_scale = 1.
        if active:
            age = 0. if self.t_hand is None else max(0., float(obs["t_win"]) - self.t_hand)
            force_scale = cfg.force_start_scale + (1. - cfg.force_start_scale) * smoothstep(
                age / max(cfg.force_ramp_s, 1e-9))
            self.drv.F_max = nominal_force * force_scale
            # On the first held tick ContinuousHandover measures r0.  Preserve that
            # measurement, then expand smoothly towards the nominal target radius.
            if self.t_hand is not None:
                radius_scale = smoothstep(age / max(cfg.radius_ramp_s, 1e-9))
                self.drv.r_final = self.drv.r0 + radius_scale * (nominal_radius - self.drv.r0)
        self.env.applied_force_scale = force_scale
        self.env.applied_radius_scale = radius_scale
        try:
            result = super().command(obs)
        finally:
            self.drv.F_max, self.drv.r_final = nominal_force, nominal_radius
        if self.t_hand is not None:
            self.env.min_joint_governor = min(self.env.min_joint_governor, float(self.gov_q))
            self.env.min_force_scale = min(self.env.min_force_scale, force_scale)
            self.env.min_radius_scale = min(self.env.min_radius_scale, radius_scale)
        return result


class SafeContinuousEnv(BufferedEnv):
    """V19 pilot environment. Actions remain zero; it is not a PPO environment."""
    def __init__(self, scenario, recipe, anchor, config=SafeContinuousConfig()):
        if not isinstance(config, SafeContinuousConfig):
            raise TypeError("SafeContinuousEnv requires SafeContinuousConfig")
        super().__init__(scenario, recipe, anchor, config)

    def reset(self, seed):
        self.min_joint_governor = self.min_force_scale = self.min_radius_scale = 1.
        self.applied_force_scale = self.applied_radius_scale = 1.
        super().reset(seed)
        old = self.handover
        self.handover = SafeHandover(self, old.catch, self.drive, self.model, self.continuous)
        self.controller.controller = self.handover
        # Retain BufferedEnv's sensor-fitted state object, while binding release
        # timing to the replacement handover controller.
        self.releaser = ContinuousRelease(self.releaser.original, self.handover, self.continuous)
        return self.observation()

    def summary(self):
        result = super().summary()
        result.update(safe_continuous_config=asdict(self.continuous), safety_abort=False,
            joint_governor_min=self.min_joint_governor, force_scale_min=self.min_force_scale,
            radius_scale_min=self.min_radius_scale)
        return result
