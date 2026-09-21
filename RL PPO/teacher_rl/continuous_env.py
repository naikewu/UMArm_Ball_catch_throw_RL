"""V16 continuous catch-to-orbit handover, isolated from frozen V15 sources."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .env import ACTOR_SIZE, OBS_SIZE, DT, PA_PER_PSI
from .improved_teacher import ImprovedTeacherEnv
from .soft_env import SoftRecipe
from can_whirl import CatchThenWhirl
from .continuous_release import BallTrajectoryPrediction, MeasuredBallState, TimestampedValve


SCHEMA = "can_continuous_residual_v16"
RESIDUAL_SIZE = 4
ACTOR_DIM = ACTOR_SIZE + RESIDUAL_SIZE + 2
OBS_DIM = OBS_SIZE + RESIDUAL_SIZE + 2


@dataclass(frozen=True)
class ContinuousConfig:
    blend_s: float = .20
    spinup_s: float = 2.0
    release_min_s: float = 2.5
    release_tolerance_m: float = .08
    release_ball_frames: int = 0
    predict_ball_motion: bool = False
    align_release_clock: bool = False
    release_azimuth_bias_deg: float = -3.5
    radial_damping: float = 6.
    match_offset: float = 0.
    pressure_drop_psi: float = 0.
    gain_scale: float = 1.
    quality_weight: float = 1.

    def __post_init__(self):
        bounds = dict(blend_s=(.05,.5), spinup_s=(.5,8.), release_min_s=(.5,10.),
            radial_damping=(2.,12.), match_offset=(-.06,.10), pressure_drop_psi=(0.,5.),
            gain_scale=(.6,1.1), quality_weight=(0.,2.), release_tolerance_m=(.04,.15),
            release_azimuth_bias_deg=(-6.,6.))
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"invalid {name}: expected [{lo}, {hi}]")
        if self.release_ball_frames not in (0,12,16,24,32):
            raise ValueError("release_ball_frames must be 0,12,16,24 or 32")
        if not isinstance(self.align_release_clock,bool):
            raise ValueError("align_release_clock must be boolean")
        if not isinstance(self.predict_ball_motion,bool) or (self.predict_ball_motion and not self.release_ball_frames):
            raise ValueError("predict_ball_motion requires sensor fitting")


class ContinuousHandover(CatchThenWhirl):
    """Start the orbit from measured motion; never call catch.stop()/brake()."""
    def __init__(self, catch, drive, model, config):
        super().__init__(catch, drive, model, q_cap_deg=36., settle_s=0.)
        self.config = config
        self.last_pressure = None
        self.last_aux = {}
        self.handover_pressure = None
        self.blend_used_s = config.blend_s

    def command(self, obs):
        t = float(obs["t_win"])
        self.drv.k_r = self.config.radial_damping
        if self.mode != "throw" and not obs["held"]:
            pressure, aux = self.catch.command(obs)
            self.last_pressure = pressure.copy()
            self.last_aux = aux
            return pressure, aux
        if self.t_hand is None:
            self.set_mode("throw")
            q, qd = np.asarray(obs["q_meas"]), np.asarray(obs["qd_hat"])
            tip, jac, _ = self.model.jac(q)
            self.drv.reset()
            self.drv.r0 = float(np.clip(np.linalg.norm((tip-self.drv.c0)[:2]), .03, self.drv.r_final))
            self.drv.spinup = self.config.spinup_s
            self.drv.st.update(tip_prev=tip.copy(), t_prev=0., vel=jac @ qd,
                qd=qd.copy(), r_bar=self.drv.r0)
            self.t_hand = self.t_settle = t
            self.handover_pressure = (self.last_pressure.copy() if self.last_pressure is not None
                else self.catch.idle_psi().copy())
            self.drv.st["p_prev"] = self.handover_pressure.copy()
        pressure, aux = super().command(obs)
        fraction = float(np.clip((t-self.t_hand)/self.blend_used_s, 0., 1.))
        weight = fraction*fraction*(3.-2.*fraction)
        pressure = (1.-weight)*self.handover_pressure + weight*pressure
        self.last_pressure = pressure.copy()
        self.last_aux = dict(aux, continuous_blend=weight, handover_time=self.t_hand)
        return pressure, self.last_aux


class ContinuousRelease:
    """Replace only V15's fixed 6+8 s gate, retaining its ballistic release model."""
    def __init__(self, original, controller, config):
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "config", config)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __setattr__(self, name, value):
        setattr(self.original, name, value)

    def __call__(self, plant, obs):
        handover = self.controller.t_hand
        self.original.t_min = np.inf if handover is None else handover+self.config.release_min_s
        self.original.tol = self.config.release_tolerance_m
        self.original.az_bias = np.radians(self.config.release_azimuth_bias_deg)
        target = TimestampedValve(plant,obs["t"]) if self.config.align_release_clock else plant
        return self.original(target, obs)


class ContinuousEnv(ImprovedTeacherEnv):
    """Frozen BC task execution plus four physical, catch-only residual controls.

    Residuals: match +/- .06; pressure drop +/- 2 psi; gain +/- .2;
    handover blend +/- .12 s. No throw action is learned in this first stage.
    """
    def __init__(self, scenario, recipe, anchor, config=ContinuousConfig()):
        self.continuous = config
        self.anchor = anchor
        self.last_residual = np.zeros(RESIDUAL_SIZE, dtype=np.float32)
        self._ready = False
        super().__init__(scenario, recipe)

    def reset(self, seed):
        self._ready = False
        # A reused environment must not inherit the last episode's gain/pressure recipe.
        self.recipe = SoftRecipe(match=self.improved_recipe.catch_match)
        self.last_residual[:] = 0.
        self.motion_trace = []
        self.quality_previous = self.task_return = self.quality_penalty = 0.
        self.postcatch_low_speed_s = 0.
        self.pause_run_s = self.pause_longest_s = 0.
        self.last_motion_t = None
        super().reset(seed)
        old = self.controller.controller
        self.handover = ContinuousHandover(old.catch, self.drive, self.model, self.continuous)
        self.drive.k_r = self.continuous.radial_damping
        self.drive.r_final = self.throw_radius_m
        self.controller.controller = self.handover
        self.releaser = ContinuousRelease(self.releaser, self.handover, self.continuous)
        if self.continuous.release_ball_frames:
            original = self.releaser.original
            estimator = MeasuredBallState(original.ball_state,self.continuous.release_ball_frames)
            original.ball_state = estimator
            if self.continuous.predict_ball_motion:
                original.predict_from = BallTrajectoryPrediction(estimator,original)
        self.base_gains = {name: getattr(self.catch, name).copy() if isinstance(getattr(self.catch, name), np.ndarray)
            else getattr(self.catch, name) for name in ("kp", "kd", "ki")}
        self._ready = True
        self._set_residual(np.zeros(RESIDUAL_SIZE))
        return self.observation()

    def _observe(self):
        result = super()._observe()
        result["ball_frame_id"] = int(self.plant._ball_frame_id)
        return result

    def residual_mask(self):
        if self.capture_time is None and self.handover.t_hand is None:
            return np.ones(RESIDUAL_SIZE, dtype=np.float32)
        # Orbit control no longer calls the catch pressure/gain controller.
        return np.zeros(RESIDUAL_SIZE, dtype=np.float32)

    def _set_residual(self, action):
        config = self.continuous
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (RESIDUAL_SIZE,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1.000001):
            raise ValueError("expected four bounded finite residuals")
        mask = self.residual_mask()
        self.last_residual = np.where(mask > 0, a, self.last_residual)
        a = self.last_residual
        match = float(np.clip(self.improved_recipe.catch_match+config.match_offset+.06*a[0], .15, .49))
        drop = float(np.clip(config.pressure_drop_psi+2.*a[1], 0., 6.))
        self.recipe = replace(self.recipe, match=match, pressure_drop_psi=drop)
        gain = float(np.clip(config.gain_scale+.2*a[2], .5, 1.2))
        self.recipe = replace(self.recipe, catch_gain_scale=gain)
        for name, base in self.base_gains.items():
            setattr(self.catch, name, base*gain)
        if self.handover.t_hand is None:
            self.handover.blend_used_s = float(np.clip(config.blend_s+.12*a[3], .05, .5))

    def observation(self):
        original = self._vector()
        age = 0. if self.capture_time is None else max(0., self.obs["t_win"]-self.capture_time)
        extra = np.r_[self.last_residual, min(age,10.)/10., float(self.capture_time is not None)]
        return np.r_[original[:ACTOR_SIZE], extra, original[ACTOR_SIZE:]].astype(np.float32)

    def step(self, action):
        self._set_residual(action)
        original = self._vector()
        with torch.no_grad():
            intent = self.anchor.distribution(torch.tensor(original[None])).mean.tanh()[0].numpy()
        # V15 overwrites radius/drive: make their observation-history entries fixed too.
        intent[5:7] = self.teacher_action()[5:7]
        _, task_reward, done, result = super().step(intent)
        self.task_return += task_reward
        costs = self.quality_cost(result)
        cumulative = sum(costs.values())
        penalty = cumulative-self.quality_previous
        self.quality_previous = cumulative
        self.quality_penalty += penalty
        result.update(task_return=self.task_return, quality_penalty=self.quality_penalty,
            quality_cost_terms=costs, rl_return=self.task_return-self.quality_penalty)
        return self.observation(), float(task_reward-penalty), done, result

    def quality_cost(self, result):
        weight = self.continuous.quality_weight
        # Absolute engineering scales, not ratios to a sparse/near-zero BC metric.
        return {name: weight*factor*float(np.clip((result.get(key) or 0.)/scale, 0., 2.))
            for name, key, factor, scale in (
                ("impact_peak", "impact_weld_peak_n", 2., 400.),
                ("impact_impulse", "impact_weld_impulse_ns", 2., 3.),
                ("capture_speed", "relative_capture_speed_m_s", 1., 5.),
                ("contact", "contact_impulse_ns", .25, 1.),
                ("pause", "postcatch_low_speed_s", .5, 1.))}

    def _physics_metrics(self, t):
        super()._physics_metrics(t)
        if not self._ready:
            return
        age = None if self.capture_physics_time is None else t-2.-self.capture_physics_time
        near_impact = ((age is not None and age <= .2) or
            (age is None and abs(t-2.-self.launch.t_arr_known_s) <= .2))
        if not near_impact and self.last_motion_t is not None and t-self.last_motion_t < DT-1e-8:
            return
        dt = DT if self.last_motion_t is None else t-self.last_motion_t
        self.last_motion_t = t
        speed = float(np.linalg.norm(self.previous_tip_velocity))
        if age is not None and .15 <= age <= 2.0 and self.release_time is None:
            slow = speed < .15
            self.postcatch_low_speed_s += dt*slow
            self.pause_run_s = self.pause_run_s+dt if slow else 0.
            self.pause_longest_s = max(self.pause_longest_s, self.pause_run_s)
        aux = self.handover.last_aux
        self.motion_trace.append([t-2., speed, self.plant.weld_force_n(),
            float(self.plant.ball_held()), self.phase(),
            float(np.mean(self.obs["p_view_pa"]))/PA_PER_PSI,
            float(np.mean(self.p_prev)) if self.p_prev is not None else 0.,
            float(aux.get("tau_ff_max",0.)),float(aux.get("tau_fb_max",0.)),
            float(aux.get("saturated_fraction",0.))])

    def summary(self):
        result = super().summary()
        handover = getattr(self, "handover", None)
        t_hand = None if handover is None else handover.t_hand
        result.update(continuous_config=asdict(self.continuous), orbit_start_time=t_hand,
            postcatch_low_speed_s=self.postcatch_low_speed_s,
            longest_postcatch_pause_s=self.pause_longest_s,
            catch_to_orbit_s=None if t_hand is None or self.capture_physics_time is None
                else t_hand-self.capture_physics_time,
            catch_to_release_s=None if self.capture_time is None or self.release_time is None
                else self.release_time-self.capture_time)
        return result
