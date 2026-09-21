"""Thirty-Hz bounded intent policy over the original 150-Hz CAN controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "umarm-mjx-rl-catch-and-place-pivot" / "mjx_experiments"
os.environ["UMARM_CAN_REPO"] = str(ROOT)
sys.path[:0] = [str(ROOT), str(SOURCE / "umarm_can"), str(SOURCE / "umarm_mk5")]

from can_plant import CanPlant, PA_PER_PSI
from can_model import CanModel
from can_episode import FrameFit
from can_catch import BallKalman, CatchSwing, MovingCloser, ball_eta, G_W
from can_whirl import OrbitDrive, CatchThenWhirl
from can_release import ReleaserModel, make_depart_fn
from digital_twin.twin_params import load_twin_kwargs
from launches_throw import ThrowDistribution

SCHEMA = "can_teacher_intent_v1"
ACTOR_SIZE = 82
OBS_SIZE = ACTOR_SIZE + 31
ACTION_SIZE = 10
DT = 1 / 150
PHASES = ("approach", "capture", "settle", "spinup", "release", "flight")
# Position, speed-match ratio, cocontraction, radius, drive, damping, azimuth, close.
LIMITS = np.array([.1, .1, .1, .25, .125, .1, .2, .25, .1, 1/6])


@dataclass(frozen=True)
class TaskConfig:
    target_distance: float = 1.5
    orbit_radius: float = .62
    target_azimuth: float = 20.
    launch_distance: float = 2.
    launch_speed: float = 5.
    mass: float = .5
    ball_radius: float = .04
    authority: float = .25
    quality_weight: float = 0.

    def __post_init__(self):
        if not 0 <= self.authority <= 1 or self.quality_weight < 0:
            raise ValueError("authority must be in [0,1]; quality weight must be nonnegative")


def bounded_intent(action, teacher, mask, authority):
    action = np.asarray(action, dtype=float)
    if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
        raise ValueError("expected ten finite intent values")
    return teacher + authority * np.clip(action - teacher, -LIMITS, LIMITS) * mask


class TeacherEnv:
    def __init__(self, config=TaskConfig()):
        self.config = config
        self.kw = load_twin_kwargs(log=lambda _: None)
        self.model = CanModel(self.kw, mount_yaw_deg=45., ball_mass_kg=config.mass)
        c, a, _ = self.model.fk(np.zeros(12))
        self.distribution = ThrowDistribution(c, a, v0_nom=config.launch_speed,
            dist_m=config.launch_distance, ball_mass_kg=config.mass, ball_radius_m=config.ball_radius)

    def reset(self, seed):
        self.seed = int(seed)
        self.launch = self.distribution.draw(self.seed)
        L, cfg = self.launch, self.config
        self.catch = CatchSwing(self.model, L.dir_known, cruise=(.3 * L.v_arr_nom_m_s, .4, .15),
            p_hold_psi=18., t_arr_known_win=L.t_arr_known_s, lead_s=.1, aim_correct=True)
        self.drive = OrbitDrive(self.model, p0_pair=np.full(12, 16.), r_final_m=cfg.orbit_radius,
            spinup_s=8., F_max_n=10., k_r=6., spin_sense=1.)
        self.controller = CatchThenWhirl(self.catch, self.drive, self.model, q_cap_deg=36., settle_s=6.)
        self.controller.brake_lag_s = .17
        self.controller.t_dec, self.controller.brake_kd, self.controller.t_ret = .3, 8., 1.
        self.closer = MovingCloser(self.model, .069)
        az = np.radians(cfg.target_azimuth)
        self.target = np.asarray(L.aim)[:2] + cfg.target_distance * np.array([np.cos(az), np.sin(az)])
        self.releaser = ReleaserModel(self.model, self.target, tol_m=.15,
            depart_fn=make_depart_fn(168.), ball_mass_kg=cfg.mass, az_bias_deg=-3.5,
            harmonic=False, horizon_s=.1, z_floor=cfg.ball_radius)
        self.plant = CanPlant(twin_kwargs=self.kw, mount_yaw_deg=45., mocap_range_m=2.5,
            grasp_tol_m=.05, seed=seed, ball_mass_kg=cfg.mass, ball_radius_m=cfg.ball_radius,
            target_xy=self.target)
        self.fit = FrameFit(k=32, frame_dt=self.plant.mocap_dt, order=3)
        self.kalman = BallKalman(sigma_pos=self.plant.ball_noise)
        self.last_ball_frame = -1
        self.tick = 300
        self.capture_time = self.release_time = self.landing_error = None
        self.launched = self.done = False
        self.last_intent = np.zeros(10)
        self.p_prev, self.dp = None, np.zeros(24)
        self.deadline = L.t_arr_known_s + .05 + 6 + 8 + 12 + 1.5
        self.contact_peak = self.contact_impulse = self.weld_peak = self.post_contact_peak = 0.
        self.pressure_integral = self.slew_integral = self.max_q = 0.
        self.trace = []
        self.plant.advance_to(0.)
        for k in range(300):
            t = (k + 1) * DT
            self.plant.command_pa(self.controller.idle_psi() * PA_PER_PSI, t)
            self.plant.advance_to(t)
        self.last_hook_time = self.plant.sim_now
        self.plant.hook_after_step = self._physics_metrics
        self.obs = self._observe()
        return self._vector()

    def phase(self):
        if self.release_time is not None:
            return 5
        if self.releaser.t_release is not None:
            return 4
        if self.capture_time is None:
            return 0
        if self.controller.mode != "throw":
            return 1
        if self.controller.t_hand is None:
            return 2
        return 3

    def mask(self):
        mask = np.zeros(10, dtype=np.float32)
        phase = self.phase()
        if phase == 0:
            mask[:5] = 1
            mask[9] = 1
        elif phase in (1, 2):
            mask[4] = 1
        elif phase == 3:
            mask[4:9] = 1
        return mask

    def teacher_action(self):
        aim = self.catch.aim_shift.copy()
        bh = self.obs["ball_hat"]
        eta = ball_eta(bh, self.catch.c0, self.catch.a0)
        if self.phase() == 0 and np.isfinite(eta) and eta > 0 and self.catch._vis >= 3:
            d = bh["pos"] + bh["vel"] * eta + .5 * G_W * eta**2 - self.catch.c0
            d -= self.catch.a0 * (d @ self.catch.a0)
            aim = d * min(1., .15 / max(np.linalg.norm(d), 1e-12))
        # Normalized physical intent, including nonconstant geometric teacher labels.
        p0 = 18. if self.controller.mode != "throw" else 16.
        return np.clip(np.r_[aim / .15, 0., (p0 - 16) / 8,
            (self.config.orbit_radius - .55) / .3, 0., 0., -.35, 0.], -.999, .999).astype(np.float32)

    def _observe(self):
        pl = self.plant
        for fid, q in pl._q_frames_new:
            self.fit.push(fid, q)
        pl._q_frames_new.clear()
        t = (self.tick + 1) * DT
        q_frame = pl._q_frame.copy()
        age = float(np.clip(t - (pl._q_frame_id - 1) * pl.mocap_dt, 0, 4 * pl.mocap_dt))
        fit = self.fit.state(lead_s=age)
        bfid, bm = pl._ball_frame_id, pl._ball_frame.copy()
        if bfid >= 0 and bfid != self.last_ball_frame and np.all(np.isfinite(bm)):
            self.kalman.update((bfid - 1) * pl.mocap_dt, bm)
            self.last_ball_frame = bfid
        bs = self.kalman.state_at(t) if self.kalman.x is not None and pl._ball_visible else None
        return dict(t=t, t_win=t-2., q_meas=fit[0] if fit else q_frame,
            q_frame=q_frame, qd_hat=fit[1] if fit else np.zeros(12),
            ball_meas=bm, ball_hat=dict(visible=bs is not None,
                pos=bs[0] if bs else None, vel=bs[1] if bs else None),
            p_view_pa=pl.p_view_pa(), mocap_age_s=age, held=pl.ball_held())

    def _vector(self):
        o = self.obs
        c, J, _ = self.model.jac(o["q_meas"])
        bh = o["ball_hat"]
        relative = bh["pos"]-c if bh["visible"] else np.zeros(3)
        velocity = bh["vel"] if bh["visible"] else np.zeros(3)
        clock = [o["t_win"]/30, (self.launch.t_arr_known_s-o["t_win"])/2,
                 float(bh["visible"]), float(o["held"])]
        actor = np.r_[o["q_meas"], o["qd_hat"]/8, o["p_view_pa"]/(30*PA_PER_PSI),
            relative/2.5, velocity/6, c, (J @ o["qd_hat"])/6, self.target/2,
            np.eye(6)[self.phase()], clock, self.last_intent]
        # Only the critic receives the actual plant state and constraint force.
        privileged = np.r_[self.plant.q(), self.plant.qd_true()/8,
            self.plant.ball_pos()/2.5, self.plant.ball_vel()/6, self.plant.weld_force_n()/400]
        result = np.r_[actor, privileged].astype(np.float32)
        if result.shape != (OBS_SIZE,) or not np.all(np.isfinite(result)):
            raise RuntimeError("invalid teacher observation")
        return result

    def _physics_metrics(self, t):
        pl = self.plant
        dt = max(0., t - self.last_hook_time)
        self.last_hook_time = t
        contacts = pl.ball_contacts()
        force = sum(f for name, f in contacts if name != "floor")
        self.contact_peak = max(self.contact_peak, force)
        self.contact_impulse += force * dt
        self.weld_peak = max(self.weld_peak, pl.weld_force_n())
        self.max_q = max(self.max_q, float(np.degrees(np.abs(pl.q()).max())))
        if self.capture_time is not None and not pl.ball_held() and self.release_time is None:
            self.release_time = t - 2.
        if self.release_time is not None:
            self.post_contact_peak = max(self.post_contact_peak, force)
            if self.landing_error is None and any(name == "floor" for name, _ in contacts):
                self.landing_error = float(np.linalg.norm(pl.ball_pos()[:2] - self.target))

    def step(self, action):
        if self.done:
            raise RuntimeError("reset after episode end")
        teacher = self.teacher_action()
        mask = self.mask()
        applied = bounded_intent(action, teacher, mask, self.config.authority)
        delta = applied - teacher
        was_caught, was_released = self.capture_time is not None, self.release_time is not None
        impulse_before = self.contact_impulse
        peak_before = self.weld_peak
        pressure_before = self.pressure_integral
        reward = 0.
        for _ in range(5):
            o, pl, ctl = self.obs, self.plant, self.controller
            t, tw = o["t"], o["t_win"]
            phase = self.phase()
            if phase == 0:
                speed = (.3 + .2 * delta[3]) * self.launch.v_arr_nom_m_s
                self.catch.plan.v_c = speed
                self.catch.plan.amp = speed * (.15 + .5 * .4)
                shift = np.linalg.solve(self.catch._J0.T @ self.catch._J0 + 1e-3*np.eye(12),
                    self.catch._J0.T @ (.15 * delta[:3]))
                self.catch.q_hold0 = shift
                self.catch.p_hold = 18. + 8 * delta[4]
                self.closer.lat = .069 + .03 * delta[9]
            elif phase in (1, 2):
                self.catch.p_hold = (18. if phase == 1 else 16.) + 8*delta[4]
            elif phase == 3:
                self.drive.set_p0_pair(np.full(12, 16. + 8*delta[4]))
                ctl.hold = self.model.hold_psi(self.drive.p0)
                self.drive.r_final = self.config.orbit_radius + .3*delta[5]
                self.drive.F_max = 10. + 5*delta[6]
                self.drive.k_r = 6. + 4*delta[7]
                self.releaser.az_bias = np.radians(-3.5 + 10*delta[8])
            psi, aux = ctl.command(o)
            if self.p_prev is not None:
                self.dp += DT/(DT+.03)*((psi-self.p_prev)/DT-self.dp)
                self.slew_integral += float(np.mean(np.abs(psi-self.p_prev)))
            self.p_prev = psi.copy()
            output = np.clip(psi + .1*self.dp, 1., 30.)
            self.pressure_integral += float(output.mean()) * DT
            pl.command_pa(output*PA_PER_PSI, t)
            if tw >= self.launch.t_launch_s and not self.launched:
                pl.launch_ball(self.launch.pos, self.launch.vel)
                self.launched = True
            if self.capture_time is None:
                self.closer(pl, o)
                if pl.ball_held():
                    self.capture_time = tw
            else:
                if ctl.mode != "throw" and tw >= self.capture_time + .05:
                    ctl.set_mode("throw")
                    self.releaser.t_min = tw + 6. + 8.
                if ctl.mode == "throw":
                    self.releaser(pl, o)
            pl.advance_to(t)
            self.trace.append(np.r_[tw, phase, o["q_meas"], o["qd_hat"],
                o["p_view_pa"]/PA_PER_PSI, psi, output, pl.q(), pl.qd_true(),
                pl.ball_pos(), pl.ball_vel(), pl.weld_force_n(), pl.pad_contact_n()])
            self.tick += 1
            self.obs = self._observe()
            miss = self.capture_time is None and tw > self.launch.t_arr_known_s + 1.
            self.done = self.landing_error is not None or tw >= self.deadline or miss or pl.grip_broken
            if self.done:
                break
        if not was_caught and self.capture_time is not None:
            reward += 5.
        if not was_released and self.release_time is not None and self.releaser.t_release is not None:
            reward += 5.
        if self.done:
            if self.landing_error is not None:
                reward += 20*np.exp(-(self.landing_error/.3)**2) + 20*float(self.landing_error <= .15)
            else:
                reward -= 10.
        # Small residual regularizer; quality costs are disabled for success-first training.
        reward -= .002 * float(np.square(delta).sum())
        cost = .02*(self.contact_impulse-impulse_before) + .002*max(0., self.weld_peak-peak_before)
        cost += .0001*(self.pressure_integral-pressure_before)
        reward -= self.config.quality_weight * cost
        self.last_intent = applied
        return self._vector(), float(reward), self.done, self.summary()

    def summary(self):
        return dict(seed=self.seed, config=asdict(self.config), captured=self.capture_time is not None,
            released=self.release_time is not None and self.releaser.t_release is not None,
            hit15=self.landing_error is not None and self.landing_error <= .15,
            hit30=self.landing_error is not None and self.landing_error <= .30,
            landing_error_m=self.landing_error, capture_time=self.capture_time,
            release_time=self.release_time, contact_peak_n=self.contact_peak,
            contact_impulse_ns=self.contact_impulse, weld_peak_n=max(self.weld_peak,self.plant.grip_force_peak_n),
            post_release_contact_n=self.post_contact_peak, max_joint_deg=self.max_q,
            pressure_integral_psi_s=self.pressure_integral, pressure_variation_psi=self.slew_integral,
            grip_broken=self.plant.grip_broken)
