"""Controller gain selection on joint multisines, separate from Soft scoring."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import numpy as np
from control.trajectory import tip_position


class TuningTrajectory:
    duration_s = 12.0
    metadata = {"kind": "joint multisine tuning, not Soft", "duration_s": 12.0,
                "frequency_hz": [0.12, 0.23, 0.37], "amplitude_deg": 6.0}

    def sample(self, t):
        omega = 2*np.pi*np.resize([.12, .23, .37], 12)
        phase = np.arange(12)*.71
        amp = np.deg2rad(6)
        x = np.clip(t/2, 0, 1)
        b = 10*x**3-15*x**4+6*x**5
        bd = (30*x**2-60*x**3+30*x**4)/2 if t<2 else 0
        bdd = (60*x-180*x**2+120*x**3)/4 if t<2 else 0
        a = omega*t+phase
        q = amp*np.sin(a)
        qd = amp*omega*np.cos(a)
        qdd = -amp*omega**2*np.sin(a)
        return b*q, bd*q+b*qd, bdd*q+2*bd*qd+b*qdd

    def future(self,t,H,dt):
        return np.stack([self.sample(t+(k+1)*dt)[0] for k in range(H)])

    def tip_target(self,t):
        return tip_position(self.sample(t)[0])

    def pen_down(self,t):
        return np.asarray(t)>=2


def candidate(task):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from control.benchmark import run
    method, config = task
    try:
        trace, meta = run(method, TuningTrajectory(), seed=6102, controller_kwargs=config)
    except (RuntimeError, ValueError, AssertionError) as exc:
        return dict(method=method,config=config,error=str(exc),objective=1e9)
    metrics = meta["metrics"]
    return dict(method=method, config=config, metrics=metrics,
                objective=metrics["writing_tip_rms_mm"]+15*metrics["writing_joint_rms_deg"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=["pid","ff_pid"],default="pid")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--refine",action="store_true")
    p.add_argument("--out",type=Path,default=Path("data/control_tuning.json"))
    args=p.parse_args()
    if args.method=="pid":
        if args.refine:
            configs=[dict(kp=kp,ki=ki,kd=kd) for kp,ki,kd in
                     itertools.product([100,150,220],[30,60],[12,20])]
        else:
            configs=[dict(kp=kp,ki=ki,kd=kd) for kp,ki,kd in
                     itertools.product([150,300,600],[30,100],[3,9])]
            configs.append(dict(kp=45,ki=8,kd=2.5))
    else:
        configs=[dict(kp=kp,ki=8,kd=kd,preview=preview) for kp,kd,preview in
                 itertools.product([25,100],[2,5],[.0,.035,.070])]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(candidate,(args.method,c)) for c in configs]):
            r=f.result(); results.append(r)
            print(json.dumps(r),flush=True)
    results.sort(key=lambda r:r["objective"])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(trajectory=TuningTrajectory.metadata,
                                      seed=6102,results=results),indent=2),encoding="utf-8")


if __name__=="__main__": main()
