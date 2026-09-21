"""Measured-ball release state and timestamp-aligned valve scheduling for V16."""
import numpy as np

from can_episode import FrameFit
from can_release import landing


class MeasuredBallState:
    """Use timestamped sensor frames, never the simulator's privileged ball state."""
    def __init__(self, fallback, frames):
        self.fallback = fallback
        self.fit = FrameFit(k=frames,frame_dt=1./240.,order=3)
        self.last_frame = None
        self.lead = None

    def __call__(self, obs):
        # Maintain the original offset estimate for diagnostics and initial fallback.
        nominal = self.fallback(obs)
        frame = obs.get("ball_frame_id",-1)
        measured = np.asarray(obs.get("ball_meas",[np.nan]*3),dtype=float)
        if frame >= 0 and measured.shape == (3,) and np.isfinite(measured).all():
            self.fit.push(frame,measured)
            self.last_frame = frame
        if self.last_frame is None or len(self.fit.buf)<self.fit.k:
            self.lead = None
            return nominal
        age = obs["t"]-(self.last_frame-1)*self.fit.dt
        if not 0. <= age <= .04:
            self.lead = None
            return nominal
        self.lead = age
        position,velocity,_ = self.fit.state(lead_s=age)
        return position,velocity


class BallTrajectoryPrediction:
    """Extrapolate attached-ball acceleration before applying free-flight ballistics."""
    def __init__(self, estimator, releaser):
        self.estimator,self.releaser = estimator,releaser
        self.fallback = releaser.predict_from

    def __call__(self, p, v, delay):
        if self.estimator.lead is None:
            return self.fallback(p,v,delay)
        position,velocity,_ = self.estimator.fit.state(lead_s=self.estimator.lead+delay)
        speed = np.linalg.norm(velocity)*self.releaser.speed_gain
        azimuth = np.arctan2(velocity[1],velocity[0])+self.releaser.az_bias
        elevation = np.arctan2(velocity[2],np.hypot(*velocity[:2]))+self.releaser.elev_bias
        velocity = speed*np.array([np.cos(elevation)*np.cos(azimuth),
            np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
        target,_ = landing(position,velocity,z_floor=self.releaser.z_floor)
        return target,velocity,position


class TimestampedValve:
    """Schedule a valve command relative to the observation's reference time."""
    def __init__(self, plant, command_time):
        self.plant = plant
        self.lead = max(0.,float(command_time)-float(plant.sim_now))

    def __getattr__(self, name):
        return getattr(self.plant,name)

    def gripper_open(self, t=None, *, due_in_s=None, vent_in_s=None):
        if vent_in_s is not None:
            return self.plant.gripper_open(t,vent_in_s=vent_in_s+self.lead)
        if due_in_s is not None:
            return self.plant.gripper_open(t,due_in_s=due_in_s+self.lead)
        return self.plant.gripper_open(t)
