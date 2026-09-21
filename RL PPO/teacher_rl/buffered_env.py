"""V17 signed pressure, independent gains and a real post-catch orbit buffer."""
from dataclasses import asdict, dataclass, replace

import numpy as np

from .continuous_env import ContinuousConfig, ContinuousEnv, ContinuousHandover, ContinuousRelease
from .continuous_release import MeasuredBallState
from .env import ACTOR_SIZE, OBS_SIZE, DT, PA_PER_PSI
from .improved_teacher import ImprovedTeacherEnv
from .soft_env import SoftRecipe

SCHEMA = "can_buffered_residual_v17"
ACTION_NAMES = ("match", "catch_pressure", "catch_kp", "catch_kd", "blend",
                "buffer_pressure", "buffer_kd")
RESIDUAL_SIZE = len(ACTION_NAMES)
ACTOR_DIM = ACTOR_SIZE + 2 * RESIDUAL_SIZE + 2
OBS_DIM = OBS_SIZE + 2 * RESIDUAL_SIZE + 2
CONTROL_COLUMNS = (["time_s"] + ["requested_" + n for n in ACTION_NAMES] +
    ["filtered_" + n for n in ACTION_NAMES] + ["catch_pressure_delta_psi", "catch_kp_scale",
    "catch_kd_scale", "buffer_envelope", "buffer_pressure_delta_psi", "buffer_kd_scale",
    "command_pressure_mean_psi", "measured_pressure_mean_psi", "clipped_fraction",
    "command_pressure_min_psi", "command_pressure_max_psi"])


@dataclass(frozen=True)
class BufferedConfig(ContinuousConfig):
    pressure_authority_psi: float = 1.
    match_authority: float = .06
    filter_s: float = .05
    pressure_slew_psi_s: float = 20.
    buffer_s: float = .30
    buffer_fade_s: float = .10
    release_ball_frames: int = 24
    align_release_clock: bool = True
    release_azimuth_bias_deg: float = 0.
    release_tolerance_m: float = .04

    def __post_init__(self):
        super().__post_init__()
        for name, lo, hi in (("pressure_authority_psi", .25, 3.), ("match_authority", .01, .10),
                ("filter_s", .02, .15), ("pressure_slew_psi_s", 5., 40.),
                ("buffer_s", .15, .5), ("buffer_fade_s", .05, .2)):
            value = getattr(self, name)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid {name}: expected [{lo}, {hi}]")
        if self.buffer_fade_s >= self.buffer_s:
            raise ValueError("buffer fade must be shorter than buffer duration")
        if self.pressure_drop_psi != 0. or self.gain_scale != 1. or self.predict_ball_motion:
            raise ValueError("V17 replaces one-sided pressure/shared gains; failed future extrapolation is disabled")


def gain_scale(action, down, up):
    return 1. + float(action) * (up if action >= 0 else down)


def buffer_weight(age, config):
    if age is None or age < 0. or age >= config.buffer_s:
        return 0.
    x = np.clip((config.buffer_s - age) / config.buffer_fade_s, 0., 1.)
    return float(x * x * (3. - 2. * x))


class _CatchPressure:
    """Apply signed co-contraction after the legacy pressure schedule, before allocation."""
    def __init__(self, env, catch):
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "catch", catch)

    def __getattr__(self, name):
        return getattr(self.catch, name)

    def __setattr__(self, name, value):
        setattr(self.catch, name, value)

    def command(self, obs):
        nominal = self.catch.p_hold
        delta = self.env.filtered_residual[1] * self.env.continuous.pressure_authority_psi
        delta *= self.env.soft_window(obs["t_win"])
        self.env.effective_catch_pressure = float(delta)
        self.catch.p_hold = nominal + delta
        try:
            return self.catch.command(obs)
        finally:
            self.catch.p_hold = nominal


class BufferedHandover(ContinuousHandover):
    def __init__(self, env, catch, drive, model, config):
        super().__init__(catch, drive, model, config)
        self.env = env

    def command(self, obs):
        env = self.env
        env.prepare_control_tick(obs)
        if not obs["held"] and self.t_hand is None:
            return super().command(obs)
        age = 0. if self.t_hand is None else obs["t_win"] - self.t_hand
        weight = buffer_weight(age, self.config)
        pressure = weight * self.config.pressure_authority_psi * env.filtered_residual[5]
        damping = 1. + weight * (gain_scale(env.filtered_residual[6], .2, .3) - 1.)
        env.effective_buffer_pressure, env.effective_buffer_kd = float(pressure), float(damping)
        env.buffer_envelope = weight
        # Restore nominal values after every tick: the parent sets p0/gains again in phase 3.
        p0, kp, kd, hold = self.drv.p0.copy(), self.drv.kp.copy(), self.drv.kd.copy(), self.hold.copy()
        self.drv.set_p0_pair(p0 + pressure)
        self.drv.kp, self.drv.kd = kp, kd * damping
        self.hold = self.model.hold_psi(self.drv.p0)
        try:
            return super().command(obs)
        finally:
            self.drv.set_p0_pair(p0)
            self.drv.kp, self.drv.kd, self.hold = kp, kd, hold


class BufferedEnv(ContinuousEnv):
    def __init__(self, scenario, recipe, anchor, config=BufferedConfig()):
        self.continuous, self.anchor = config, anchor
        self.last_residual = np.zeros(RESIDUAL_SIZE, dtype=np.float32)
        self.filtered_residual = np.zeros(RESIDUAL_SIZE, dtype=np.float32)
        self._ready = False
        ImprovedTeacherEnv.__init__(self, scenario, recipe)

    def reset(self, seed):
        self._ready = False
        self.recipe = SoftRecipe(match=self.improved_recipe.catch_match)
        self.last_residual[:] = self.filtered_residual[:] = 0.
        self.motion_trace, self.control_trace = [], []
        self.quality_previous = self.task_return = self.quality_penalty = 0.
        self.postcatch_low_speed_s = self.pause_run_s = self.pause_longest_s = 0.
        self.buffer_peak = self.buffer_impulse = 0.
        self.last_motion_t = self.last_control_time = self.last_logged_tick = None
        self.effective_catch_pressure = self.effective_buffer_pressure = self.buffer_envelope = 0.
        self.effective_buffer_kd = 1.
        ImprovedTeacherEnv.reset(self, seed)
        old = self.controller.controller
        old.catch.catch = _CatchPressure(self, old.catch.catch)
        self.handover = BufferedHandover(self, old.catch, self.drive, self.model, self.continuous)
        self.drive.k_r, self.drive.r_final = self.continuous.radial_damping, self.throw_radius_m
        self.controller.controller = self.handover
        self.releaser = ContinuousRelease(self.releaser, self.handover, self.continuous)
        if self.continuous.release_ball_frames:
            original = self.releaser.original
            original.ball_state = MeasuredBallState(original.ball_state, self.continuous.release_ball_frames)
        self.base_gains = {name: np.copy(getattr(self.catch, name)) for name in ("kp", "kd", "ki")}
        self._ready = True
        return self.observation()

    def residual_mask(self):
        if self.capture_time is None and self.handover.t_hand is None:
            # Buffer settings may be prepared before contact, then adjusted during the buffer.
            return np.ones(RESIDUAL_SIZE, dtype=np.float32)
        mask = np.zeros(RESIDUAL_SIZE, dtype=np.float32)
        start = self.handover.t_hand if self.handover.t_hand is not None else self.capture_time
        age = max(0., self.obs["t_win"] - start)
        if self.release_time is None and age < self.continuous.buffer_s - self.continuous.buffer_fade_s:
            mask[5:7] = 1.
        return mask

    def _set_residual(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (RESIDUAL_SIZE,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("expected seven bounded finite residuals")
        self.last_residual = np.where(self.residual_mask() > 0, action, self.last_residual)

    def prepare_control_tick(self, obs):
        self.effective_catch_pressure = 0.
        t = float(obs["t_win"])
        dt = DT if self.last_control_time is None else max(0., t - self.last_control_time)
        self.last_control_time = t
        change = dt / (self.continuous.filter_s + dt) * (self.last_residual - self.filtered_residual)
        bound = self.continuous.pressure_slew_psi_s * dt / self.continuous.pressure_authority_psi
        change[[1, 5]] = np.clip(change[[1, 5]], -bound, bound)
        self.filtered_residual += change
        a = self.filtered_residual
        match = np.clip(self.improved_recipe.catch_match + self.continuous.match_offset +
            self.continuous.match_authority * a[0], .15, .49)
        self.recipe = replace(self.recipe, match=float(match))
        self.catch.kp = self.base_gains["kp"] * gain_scale(a[2], .3, .1)
        self.catch.kd = self.base_gains["kd"] * gain_scale(a[3], .2, .3)
        self.catch.ki = self.base_gains["ki"].copy()
        if self.handover.t_hand is None:
            self.handover.blend_used_s = float(np.clip(self.continuous.blend_s + .12 * a[4], .05, .5))

    def observation(self):
        original = self._vector()
        age = 0. if self.capture_time is None else max(0., self.obs["t_win"] - self.capture_time)
        extra = np.r_[self.last_residual, self.filtered_residual, min(age, 10.) / 10.,
            float(self.capture_time is not None)]
        return np.r_[original[:ACTOR_SIZE], extra, original[ACTOR_SIZE:]].astype(np.float32)

    def _physics_metrics(self, t):
        dt = max(0., t - self.last_hook_time)
        super()._physics_metrics(t)
        if not self._ready:
            return
        age = None if self.capture_physics_time is None else t - 2. - self.capture_physics_time
        if age is not None and .05 < age <= .30:
            force = self.plant.weld_force_n()
            self.buffer_peak = max(self.buffer_peak, force)
            self.buffer_impulse += force * dt
        if self.last_logged_tick != self.tick and self.p_prev is not None:
            self.last_logged_tick = self.tick
            raw = self.p_prev + .1 * self.dp
            command = np.clip(raw, 1., 30.)
            self.control_trace.append([self.obs["t_win"], *self.last_residual.tolist(),
                *self.filtered_residual.tolist(), self.effective_catch_pressure,
                gain_scale(self.filtered_residual[2], .3, .1), gain_scale(self.filtered_residual[3], .2, .3),
                self.buffer_envelope, self.effective_buffer_pressure, self.effective_buffer_kd,
                float(command.mean()), float(np.mean(self.obs["p_view_pa"])) / PA_PER_PSI,
                float(np.mean(raw != command)), float(command.min()), float(command.max())])

    def quality_cost(self, result):
        costs = super().quality_cost(result)
        weight = self.continuous.quality_weight
        costs.update(buffer_peak=weight * .5 * float(np.clip(self.buffer_peak / 400., 0., 2.)),
            buffer_impulse=weight * .5 * float(np.clip(self.buffer_impulse / 3., 0., 2.)))
        return costs

    def summary(self):
        return dict(super().summary(), buffered_config=asdict(self.continuous),
            buffer_weld_peak_n=self.buffer_peak, buffer_weld_impulse_ns=self.buffer_impulse,
            action_names=list(ACTION_NAMES))
