"""Versioned teacher recipes; preserve the V12 plant and catch geometry."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .data import fingerprint
from .env import TeacherEnv, TaskConfig, PA_PER_PSI, bounded_intent

SOFT_SCHEMA = "can_soft_teacher_v13"


@dataclass(frozen=True)
class SoftRecipe:
    match: float = .3
    acceleration_s: float = .4
    pressure_drop_psi: float = 0.
    soft_pre_s: float = .20
    soft_post_s: float = .25
    pressure_ramp_s: float = .10
    deceleration_s: float = .3
    return_s: float = 1.
    brake_kd: float = 8.
    catch_gain_scale: float = 1.

    def __post_init__(self):
        limits = dict(match=(.15,.49), acceleration_s=(.2,.8), pressure_drop_psi=(0.,6.),
            soft_pre_s=(.05,.5), soft_post_s=(.05,.6), pressure_ramp_s=(.03,.2),
            deceleration_s=(.15,.8), return_s=(.5,2.), brake_kd=(3.,12.), catch_gain_scale=(.5,1.2))
        for key, (low, high) in limits.items():
            value = getattr(self,key)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be finite and in [{low}, {high}]")


def soft_fingerprint():
    h = hashlib.sha256(fingerprint().encode())
    for name in ("soft_env.py", "soft_teacher.py", "data.py", "model.py", "__main__.py"):
        p = Path(__file__).with_name(name)
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def pressure_window(t, arrival, recipe):
    if arrival is None or not np.isfinite(arrival):
        return 0.
    start, end = arrival-recipe.soft_pre_s, arrival+recipe.soft_post_s
    def smooth(x):
        x = float(np.clip(x,0.,1.))
        return x*x*(3.-2.*x)
    return smooth((t-start)/recipe.pressure_ramp_s)*smooth((end-t)/recipe.pressure_ramp_s)


class _ScheduledCatch:
    """Apply the pressure schedule after stop() has updated the catch controller."""
    def __init__(self, env, catch):
        self.env, self.catch = env, catch

    def __getattr__(self, name):
        return getattr(self.catch, name)

    def __setattr__(self, name, value):
        if name in ("env", "catch"):
            object.__setattr__(self,name,value)
        else:
            setattr(self.catch,name,value)

    def command(self, obs):
        env, ctl, r = self.env, self.catch, self.env.recipe
        if env.phase() == 0:
            ctl.plan.v_c = (r.match + .2*env.action_delta[3])*env.launch.v_arr_nom_m_s
            ctl.plan.t_acc = r.acceleration_s
            ctl.plan.amp = ctl.plan.v_c*(ctl.plan.t_c+.5*r.acceleration_s)
        nominal = 18. if env.controller.mode != "throw" else 16.
        ctl.p_hold = nominal-r.pressure_drop_psi*env.soft_window(obs["t_win"])+8*env.action_delta[4]
        ctl.p_lo = None
        return ctl.command(obs)


class SoftTeacherEnv(TeacherEnv):
    def __init__(self, recipe=SoftRecipe(), config=TaskConfig()):
        self.recipe = recipe
        super().__init__(config)

    def reset(self, seed):
        self.action_delta = np.zeros(10)
        self.capture_physics_time = None
        self.relative_capture_speed = None
        self.capture_tip = self.previous_tip = self.previous_ball_velocity = self.previous_tip_velocity = None
        self.weld_impulse = self.impact_weld_peak = self.impact_weld_impulse = self.buffer_displacement = 0.
        self.weld_peak_time = self.contact_peak_time = None
        self.schedule_trace = []
        self.frozen_arrival = None
        super().reset(seed)
        self.controller.t_dec = self.recipe.deceleration_s
        self.controller.t_ret = self.recipe.return_s
        self.controller.brake_kd = self.recipe.brake_kd
        for name in ("kp", "kd", "ki"):
            setattr(self.catch,name,getattr(self.catch,name)*self.recipe.catch_gain_scale)
        self.controller.catch = _ScheduledCatch(self,self.catch)
        self.previous_tip = self.plant.grasp_centre()[0].copy()
        self.previous_ball_velocity = self.plant.ball_vel().copy()
        self.previous_tip_velocity = np.zeros(3)
        return self._vector()

    def soft_window(self, t):
        arrival = self.frozen_arrival
        if arrival is None:
            arrival = self.catch.t_arr if np.isfinite(self.catch.t_arr) else self.launch.t_arr_known_s
        return pressure_window(t,arrival,self.recipe)

    def teacher_action(self):
        result = super().teacher_action()
        result[3] = (self.recipe.match-.3)/.2
        if self.phase() in (0,1,2):
            nominal = 18. if self.controller.mode != "throw" else 16.
            result[4] = (nominal-self.recipe.pressure_drop_psi*self.soft_window(self.obs["t_win"])-16.)/8.
        return np.clip(result,-.999,.999).astype(np.float32)

    def step(self, action):
        teacher = self.teacher_action()
        self.action_delta = bounded_intent(action,teacher,self.mask(),self.config.authority)-teacher
        return super().step(action)

    def _physics_metrics(self, t):
        dt = max(0.,t-self.last_hook_time)
        old_weld, old_contact = self.weld_peak, self.contact_peak
        super()._physics_metrics(t)
        tip = self.plant.grasp_centre()[0].copy()
        velocity = (tip-self.previous_tip)/dt if dt > 0 and self.previous_tip is not None else np.zeros(3)
        force = self.plant.weld_force_n()
        self.weld_impulse += force*dt
        if self.weld_peak > old_weld:
            self.weld_peak_time = t-2.
        if self.contact_peak > old_contact:
            self.contact_peak_time = t-2.
        if self.capture_physics_time is None and self.plant.ball_held():
            self.capture_physics_time, self.capture_tip = t-2., tip.copy()
            if self.previous_ball_velocity is not None:
                self.relative_capture_speed = float(np.linalg.norm(self.previous_ball_velocity-self.previous_tip_velocity))
            self.frozen_arrival = self.catch.t_arr if np.isfinite(self.catch.t_arr) else self.launch.t_arr_known_s
        if self.capture_physics_time is not None:
            age = t-2.-self.capture_physics_time
            if age <= .05:
                self.impact_weld_peak = max(self.impact_weld_peak,force)
                self.impact_weld_impulse += force*dt
            if age <= 1.:
                self.buffer_displacement = max(self.buffer_displacement,float(np.linalg.norm(tip-self.capture_tip)))
        self.previous_tip, self.previous_tip_velocity = tip, velocity
        self.previous_ball_velocity = self.plant.ball_vel().copy()

    def _observe(self):
        obs = super()._observe()
        if hasattr(self,"catch") and self.tick > 300:
            self.schedule_trace.append([obs["t_win"], self.catch.p_hold,
                self.catch.plan.v_c if hasattr(self.catch.plan,"v_c") else 0.])
        return obs

    def summary(self):
        return dict(super().summary(), recipe=asdict(self.recipe),
            capture_physics_time=self.capture_physics_time,
            relative_capture_speed_m_s=self.relative_capture_speed,
            weld_impulse_ns=self.weld_impulse, impact_weld_peak_n=self.impact_weld_peak,
            impact_weld_impulse_ns=self.impact_weld_impulse,
            buffer_displacement_m=self.buffer_displacement,
            weld_peak_time=self.weld_peak_time, contact_peak_time=self.contact_peak_time)
