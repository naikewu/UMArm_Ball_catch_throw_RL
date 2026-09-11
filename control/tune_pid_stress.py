"""Independent large-angle PID stress tuning; no Soft glyph or scoring seeds.

The acceptance gates measure bounded excursions and settling in these runs.
They are empirical gates, not a robust-stability proof for the delayed plant.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools
import json
from pathlib import Path
import numpy as np

from .controller import make_controller,PA_PER_PSI
from .trajectory import tip_position


def blend(t,duration):
    x=np.clip(t/duration,0.,1.)
    b=10*x**3-15*x**4+6*x**5
    bd=(30*x**2-60*x**3+30*x**4)/duration if 0<t<duration else 0.
    bdd=(60*x-180*x*x+120*x**3)/(duration*duration) if 0<t<duration else 0.
    return b,bd,bdd


class StressTrajectory:
    duration_s=28.
    def __init__(self,seed=73101,initial_deg=0.):
        rng=np.random.default_rng(seed)
        self.q0=np.deg2rad(initial_deg*np.resize([1.,-1.,-.6,.6],12))
        self.a=np.deg2rad(np.resize([14.,10.,-12.,10.],12))
        self.b=-self.a
        self.omega=2*np.pi*np.resize([.18,.31,.47],12)
        self.phase=rng.uniform(-np.pi,np.pi,12)
        self.amplitude=np.deg2rad(np.resize([14.,10.,12.],12))

    def _sine(self,t):
        a=self.omega*t+self.phase
        return self.amplitude*np.sin(a),self.amplitude*self.omega*np.cos(a),-self.amplitude*self.omega**2*np.sin(a)

    @staticmethod
    def _transition(t,duration,start,end,ed=None,edd=None):
        ed=np.zeros(12) if ed is None else ed
        edd=np.zeros(12) if edd is None else edd
        b,bd,bdd=blend(t,duration)
        return start+b*(end-start),bd*(end-start)+b*ed,bdd*(end-start)+2*bd*ed+b*edd

    def sample(self,t):
        z=np.zeros(12)
        if t<3:return self._transition(t,3,self.q0,self.a)
        if t<5:return self.a,z,z
        if t<8:return self._transition(t-5,3,self.a,self.b)
        if t<10:return self.b,z,z
        if t<13:return self._transition(t-10,3,self.b,*self._sine(t-10))
        if t<20:return self._sine(t-10)
        if t<23:
            q,qd,qdd=self._sine(t-10);b,bd,bdd=blend(t-20,3)
            return (1-b)*q,(1-b)*qd-bd*q,(1-b)*qdd-2*bd*qd-bdd*q
        return z,z,z


def run_case(config,seed,randomize,initial_deg,speed=1.):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from .sim_env import ControlEnv
    from .observation import CausalJointObserver
    import mujoco
    env=ControlEnv(seed=seed,randomize=randomize)
    env.reset();trajectory=StressTrajectory(seed,initial_deg)
    env.arm.data.qpos[env.arm._q_qposadr]=trajectory.q0
    env.arm.data.qvel[:]=0
    mujoco.mj_forward(env.arm.model,env.arm.data)
    env._q=trajectory.q0+env.rng.normal(0,env.noise_std_rad,12)
    env._qd=np.zeros(12);env._frames=deque([(0.,env._q.copy())])
    env.observer=CausalJointObserver(tau_s=env.velocity_tau_s)
    env.observer.update(0.,env._q)
    obs=env.observe();controller=make_controller("pid",**config)
    controller.reset(obs["q"],obs["p_pa"])
    duration=23/speed+5
    n=round(duration/env.dt)
    truth=np.zeros((n,12));ref=np.zeros_like(truth);targets=np.zeros((n,24))
    try:
        for k in range(n):
            q,qd,qdd=trajectory.sample(k*env.dt*speed)
            qd=qd*speed;qdd=qdd*speed*speed
            p=controller.command(obs["q"],obs["qdot"],obs["p_pa"],q,qd,qdd)
            truth[k]=obs["q_true"];ref[k]=q;targets[k]=p
            obs=env.step(p)
    finally:env.close()
    errors=np.rad2deg(truth-ref)
    active=slice(round(3/speed/env.dt),round(23/speed/env.dt))
    settled=np.rad2deg(truth[-round(2/env.dt):])
    tip_error=np.linalg.norm(tip_position(truth)-tip_position(ref),axis=1)*1000
    out=dict(seed=seed,randomize=randomize,initial_deg=initial_deg,speed=speed,
             joint_rms_deg=float(np.sqrt(np.mean(errors[active]**2))),
             tip_rms_mm=float(np.sqrt(np.mean(tip_error[active]**2))),
             max_joint_deg=float(np.rad2deg(abs(truth).max())),
             settling_rms_deg=float(np.sqrt(np.mean(settled**2))),
             settling_max_std_deg=float(settled.std(axis=0).max()),
             command_max_psi=float(targets.max()/PA_PER_PSI),
             segment_rms_deg=[float(np.sqrt(np.mean(errors[active,s:s+4]**2))) for s in (0,4,8)])
    out["passes"]=out["max_joint_deg"]<24 and out["settling_rms_deg"]<1.5 and out["settling_max_std_deg"]<.6
    return out


def candidate(task):
    config,cases=task
    rows=[]
    for case in cases:
        try:rows.append(run_case(config,**case))
        except Exception as e:rows.append(dict(**case,passes=False,error=f"{type(e).__name__}: {e}"))
    passes=all(r["passes"] for r in rows)
    objective=max((r.get("joint_rms_deg",1e6)+r.get("tip_rms_mm",1e6)/100) for r in rows)
    return dict(config=config,passes=passes,objective=objective,cases=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("data/control_pid_stress.json"))
    p.add_argument("--workers",type=int,default=16)
    p.add_argument("--configs",type=Path)
    p.add_argument("--confirm",action="store_true")
    p.add_argument("--fast-focused",action="store_true")
    args=p.parse_args()
    configs=(json.loads(args.configs.read_text()) if args.configs else
             [dict(kp=kp,ki=ki,kd=kd) for kp,ki,kd in itertools.product([45,65,85,110],[8,20],[2.5,5,8])]+[dict(kp=150,ki=30,kd=12)])
    cases=[dict(seed=73101,randomize=False,initial_deg=0.),
           dict(seed=73102,randomize=True,initial_deg=3.),
           dict(seed=73103,randomize=True,initial_deg=-3.)]
    if args.confirm:
        cases=[dict(seed=s,randomize=r,initial_deg=q,speed=v) for s,r,q,v in
               [(73201,False,0.,1.),(73202,True,4.,1.),(73203,True,-4.,1.),
                (73204,False,2.,1.8),(73205,True,-2.,1.5),(73206,True,0.,1.5)]]
    if args.fast_focused:cases=cases[-3:]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(candidate,(c,cases)) for c in configs]):
            row=f.result();results.append(row)
            print(json.dumps(row),flush=True)
    results.sort(key=lambda r:(not r["passes"],r["objective"]))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(trajectory="28s independently specified10–14deg coupled holds, reversals, multisines, settling",acceptance="max|q|<24deg; last2s jointRMS<1.5deg,maxjointSD<.6deg",objective="worst-case joint RMSdeg + tip RMSmm/100",cases=cases,results=results),indent=2),encoding="utf-8")


if __name__=="__main__":main()
