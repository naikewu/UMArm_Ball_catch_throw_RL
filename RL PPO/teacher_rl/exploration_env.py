"""V18 bounded online exploration of throw force, radius and release decisions."""
from dataclasses import asdict, dataclass

import numpy as np

from .buffered_env import BufferedConfig, BufferedEnv, BufferedHandover, gain_scale
from .buffered_env import CONTROL_COLUMNS as BUFFER_COLUMNS
from .continuous_release import TimestampedValve
from .env import ACTOR_SIZE, OBS_SIZE, DT

SCHEMA = "can_online_release_v18"
ACTION_NAMES = ("match", "catch_pressure", "catch_kp", "catch_kd", "blend_disabled",
    "buffer_pressure", "buffer_kd", "throw_force", "throw_radius", "release_vote")
RESIDUAL_SIZE = len(ACTION_NAMES)
RELEASE_FEATURES = 10
ACTOR_DIM = ACTOR_SIZE + 2*RESIDUAL_SIZE + 2 + RELEASE_FEATURES
OBS_DIM = OBS_SIZE + 2*RESIDUAL_SIZE + 2 + RELEASE_FEATURES
CONTROL_COLUMNS = (["time_s"] + ["requested_"+n for n in ACTION_NAMES] +
    ["filtered_"+n for n in ACTION_NAMES] + BUFFER_COLUMNS[15:])
EXPLORATION_COLUMNS = ("time_s", "force_n", "radius_m", "release_vote", "release_allowed",
    "release_requested", "predicted_error_m", "measured_joint_max_deg")


@dataclass(frozen=True)
class ExplorationConfig(BufferedConfig):
    stage: str = "throw"
    pressure_authority_psi: float = .5
    match_authority: float = .02
    force_authority_n: float = 1.
    radius_authority_m: float = .02
    request_min_s: float = .5
    vote_threshold: float = .75
    release_joint_deg: float = 36.
    abort_joint_deg: float = 45.
    safety_penalty: float = 30.
    holding_cost_per_s: float = .05

    def __post_init__(self):
        super().__post_init__()
        if self.stage not in ("catch", "throw", "joint"):
            raise ValueError("stage must be catch, throw or joint")
        for name, low, high in (("force_authority_n", .1, 2.), ("radius_authority_m", .005, .04),
                ("request_min_s", .3, 2.5), ("vote_threshold", .1, .8),
                ("release_joint_deg", 30., 36.), ("abort_joint_deg", 38., 45.),
                ("safety_penalty", 10., 100.), ("holding_cost_per_s", 0., .2)):
            if not np.isfinite(getattr(self,name)) or not low <= getattr(self,name) <= high:
                raise ValueError(f"invalid {name}")


class ExploringRelease:
    """A policy can request early release or veto the ballistic advisor, within state guards.

    The advisor is evaluated once per observation; no simulator ball truth is used here.
    A neutral vote retains its error/minimum-time rule, plus the V18 joint/sensor guard.
    """
    def __init__(self, original, env):
        object.__setattr__(self,"original",original)
        object.__setattr__(self,"env",env)
        object.__setattr__(self,"cache_time",None)
        object.__setattr__(self,"candidate",None)
        object.__setattr__(self,"features",np.zeros(RELEASE_FEATURES,dtype=np.float32))
        object.__setattr__(self,"allowed",False)
        object.__setattr__(self,"counts",dict(policy_requests=0,policy_releases=0,advisor_releases=0,
            veto_ticks=0,blocked_request_ticks=0))

    def __getattr__(self,name):
        return getattr(self.original,name)

    def __setattr__(self,name,value):
        if name in ("cache_time","candidate","features","allowed"):
            object.__setattr__(self,name,value)
        else:
            setattr(self.original,name,value)

    def refresh(self,obs):
        if self.cache_time == obs["t"]:
            return
        self.cache_time = obs["t"]
        self.allowed, self.candidate = False, None
        self.features[:] = 0.
        if not obs["held"] or self.original.t_release is not None:
            return
        original, cfg = self.original, self.env.continuous
        original.az_bias = np.radians(cfg.release_azimuth_bias_deg)
        p,v = original.ball_state(obs)
        if not np.isfinite(np.r_[p,v]).all():
            return
        original._rates(float(obs["t_win"]),v)
        # Fit freshness is exposed by the timestamped measured-ball estimator.
        fresh = getattr(original.ball_state,"lead",None) is not None
        start = self.env.handover.t_hand
        age = -1. if start is None else obs["t_win"]-start
        qmax = float(np.degrees(np.abs(obs["q_meas"]).max()))
        self.features[:3] = np.clip(v/6.,-2.,2.)
        self.features[8:] = [float(fresh),float(age >= cfg.request_min_s)]
        if age < cfg.request_min_s:
            return
        candidates = []
        ahead = int(round(original.horizon/(original.tick/original.sub)))
        for k in range(original.sub+1+ahead):
            vent = k*original.tick/original.sub
            delay = original.lat
            if original.depart_fn is not None:
                for _ in range(2):
                    _,pred_v,_ = original.predict_from(p,v,vent+delay)
                    accel = abs(original.omega)*float(np.hypot(pred_v[0],pred_v[1]))
                    delay = float(original.depart_fn(original.m_ball*np.hypot(accel,9.81)))
            land,pred_v,_ = original.predict_from(p,v,vent+delay)
            error = float(np.linalg.norm(land[:2]-original.target))
            speed = float(np.linalg.norm(pred_v))
            elevation = float(np.arctan2(pred_v[2],np.hypot(pred_v[0],pred_v[1])))
            if np.isfinite(np.r_[land,pred_v,error,delay]).all():
                candidates.append(dict(k=k,vent=vent,delay=vent+delay,departure=delay,
                    land=land,velocity=pred_v,error=error,speed=speed,elevation=elevation))
        if not candidates:
            return
        best = min(candidates,key=lambda c:c["error"])
        now = [c for c in candidates if c["k"] <= original.sub and
            c["speed"] >= original.v_min and c["elevation"] >= original.min_elev]
        current = min(now,key=lambda c:c["error"]) if now else None
        self.allowed = bool(fresh and qmax <= cfg.release_joint_deg and current is not None)
        self.candidate = dict(best=best,current=current,age=age)
        self.features[3:5] = np.clip((best["land"][:2]-original.target)/2.,-2.,2.)
        self.features[5:8] = [min(best["delay"],.5)/.5,float(self.allowed),float(best["k"] <= original.sub)]
        if best["error"] < original.best[0]:
            original.best = (best["error"],float(obs["t_win"]))

    def __call__(self,plant,obs):
        self.refresh(obs)
        if self.original.t_release is not None or not obs["held"]:
            return
        vote = float(self.env.last_residual[9])
        cfg = self.env.continuous
        if vote < -cfg.vote_threshold:
            self.counts["veto_ticks"] += 1
            return
        requested = vote > cfg.vote_threshold
        if requested:
            self.counts["policy_requests"] += 1
        if not self.allowed:
            self.counts["blocked_request_ticks"] += int(requested)
            return
        if requested:
            choice = self.candidate["current"]
        else:
            choice = self.candidate["best"]
            if (self.candidate["age"] < cfg.release_min_s or choice["k"] > self.original.sub or
                    choice["error"] > cfg.release_tolerance_m or choice["speed"] < self.original.v_min or
                    choice["elevation"] < self.original.min_elev):
                return
        valve = TimestampedValve(plant,obs["t"])
        if self.original.depart_fn is None:
            valve.gripper_open(due_in_s=choice["delay"])
        else:
            valve.gripper_open(vent_in_s=choice["vent"])
        self.original.t_release = float(obs["t_win"])
        self.original.vent_in, self.original.depart_pred = choice["vent"],choice["departure"]
        self.original.state_at_cmd = dict(landing_pred=choice["land"].tolist(),landing_err_pred=choice["error"],
            v=choice["speed"],elev_pred=choice["elevation"],due_in_s=choice["delay"],
            vent_in_s=choice["vent"],depart_pred_s=choice["departure"],origin="policy" if requested else "advisor")
        self.counts["policy_releases" if requested else "advisor_releases"] += 1


class ExploringHandover(BufferedHandover):
    def command(self,obs):
        # Applied after the V15 adaptive wrapper overwrites its nominal orbit parameters.
        nominal_force, nominal_radius = self.env.throw_force_n,self.env.throw_radius_m
        force,radius = self.drv.F_max,self.drv.r_final
        cfg = self.env.continuous
        self.drv.F_max = float(np.clip(nominal_force+cfg.force_authority_n*self.env.filtered_residual[7],7.,14.))
        self.drv.r_final = float(np.clip(nominal_radius+cfg.radius_authority_m*self.env.filtered_residual[8],.50,.74))
        self.env.applied_force, self.env.applied_radius = self.drv.F_max,self.drv.r_final
        try:
            return super().command(obs)
        finally:
            self.drv.F_max,self.drv.r_final = force,radius


class ExplorationEnv(BufferedEnv):
    def __init__(self,scenario,recipe,anchor,config=ExplorationConfig()):
        super().__init__(scenario,recipe,anchor,config)
        self.last_residual = np.zeros(RESIDUAL_SIZE,dtype=np.float32)
        self.filtered_residual = np.zeros(RESIDUAL_SIZE,dtype=np.float32)

    def reset(self,seed):
        self.exploration_ready = False
        self.safety_abort = False
        self.exploration_trace = []
        self.last_exploration_tick = None
        self.applied_force,self.applied_radius = self.throw_force_n,self.throw_radius_m
        super().reset(seed)
        old = self.handover
        self.handover = ExploringHandover(self,old.catch,self.drive,self.model,self.continuous)
        self.controller.controller = self.handover
        self.releaser = ExploringRelease(self.releaser.original,self)
        self.exploration_ready = True
        return self.observation()

    def residual_mask(self):
        mask = np.zeros(RESIDUAL_SIZE,dtype=np.float32)
        if self.release_time is not None or self.releaser.t_release is not None:
            return mask
        catch = self.capture_time is None and self.handover.t_hand is None
        if self.continuous.stage in ("catch","joint"):
            if catch:
                mask[[0,1,2,3,5,6]] = 1.
            else:
                start = self.handover.t_hand if self.handover.t_hand is not None else self.capture_time
                if self.obs["t_win"]-start < self.continuous.buffer_s-self.continuous.buffer_fade_s:
                    mask[5:7] = 1.
        if not catch and self.continuous.stage in ("throw","joint"):
            mask[7:9] = 1.
            start = self.handover.t_hand
            if start is not None and self.obs["t_win"]-start >= self.continuous.request_min_s:
                # Includes veto decisions even while a positive request is currently blocked.
                mask[9] = 1.
        return mask

    def _set_residual(self,action):
        action = np.asarray(action,dtype=np.float32)
        if action.shape != (RESIDUAL_SIZE,) or not np.isfinite(action).all() or np.any(np.abs(action)>1.000001):
            raise ValueError("expected ten bounded V18 actions")
        self.last_residual = np.where(self.residual_mask()>0,action,self.last_residual)

    def prepare_control_tick(self,obs):
        super().prepare_control_tick(obs)
        a = self.filtered_residual
        self.catch.kp = self.base_gains["kp"]*gain_scale(a[2],.1,.05)
        self.catch.kd = self.base_gains["kd"]*gain_scale(a[3],.05,.1)
        self.handover.blend_used_s = self.continuous.blend_s

    def observation(self):
        original = self._vector()
        features = np.zeros(RELEASE_FEATURES,dtype=np.float32)
        if getattr(self,"exploration_ready",False):
            self.releaser.refresh(self.obs)
            features = self.releaser.features
        age = 0. if self.capture_time is None else max(0.,self.obs["t_win"]-self.capture_time)
        extra = np.r_[self.last_residual,self.filtered_residual,min(age,10.)/10.,
            float(self.capture_time is not None),features]
        return np.r_[original[:ACTOR_SIZE],extra,original[ACTOR_SIZE:]].astype(np.float32)

    def _physics_metrics(self,t):
        super()._physics_metrics(t)
        if not getattr(self,"exploration_ready",False):
            return
        if self.control_trace:
            self.control_trace[-1][22] = gain_scale(self.filtered_residual[2],.1,.05)
            self.control_trace[-1][23] = gain_scale(self.filtered_residual[3],.05,.1)
        if self.max_q >= self.continuous.abort_joint_deg:
            self.safety_abort = True
            # End at the next control boundary without modifying the simulated state.
            self.deadline = min(self.deadline,self.obs["t_win"])
        if self.last_exploration_tick != self.tick:
            self.last_exploration_tick = self.tick
            candidate = self.releaser.candidate
            error = -1. if candidate is None else candidate["best"]["error"]
            self.exploration_trace.append([self.obs["t_win"],self.applied_force,self.applied_radius,
                float(self.last_residual[9]),float(self.releaser.allowed),
                float(self.last_residual[9]>self.continuous.vote_threshold),error,
                float(np.degrees(np.abs(self.obs["q_meas"]).max()))])

    def quality_cost(self,result):
        costs = super().quality_cost(result)
        costs["joint_excursion"] = 2.*float(np.clip((self.max_q-36.)/9.,0.,2.))
        costs["safety_abort"] = self.continuous.safety_penalty*float(self.safety_abort)
        age = 0. if self.capture_time is None else max(0.,
            (self.release_time if self.release_time is not None else self.obs["t_win"])-self.capture_time)
        costs["holding_time"] = self.continuous.holding_cost_per_s*age
        return costs

    def summary(self):
        result = super().summary()
        result.update(exploration_config=asdict(self.continuous),action_names=list(ACTION_NAMES),
            safety_abort=self.safety_abort,release_decisions=dict(self.releaser.counts)
                if isinstance(self.releaser,ExploringRelease) else {})
        return result
