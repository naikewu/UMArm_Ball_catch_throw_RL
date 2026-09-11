"""Equal-pressure step comparison of the fitted twin's two valve populations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from control.sim_env import ControlEnv
from digital_twin.replay import PA_PER_PSI


def run():
    env=ControlEnv(seed=9162)
    obs=env.reset()
    rows=[]
    for k in range(1500):
        t=k*env.dt
        target=5 if t<2 or t>=6 else 12
        rows.append((t,target,obs["p_pa"].copy()/PA_PER_PSI))
        obs=env.step(np.full(24,target*PA_PER_PSI))
    t=np.array([r[0] for r in rows]); p=np.stack([r[2] for r in rows])
    command=np.array([r[1] for r in rows])
    result={"kind":"equal common-mode pressure steps on the fitted twin",
            "stimulus_psi":[5,12,5], "step_times_s":[2,6],
            "environment":env.meta,"populations":{},
            "limitation":"Twin predictions from the fitted valve model; this is not a new hardware measurement."}
    for label,mask in (("tle",env.variants==2),("7mm",env.variants!=2)):
        traces=p[:,mask]; metrics=[]
        for j in range(traces.shape[1]):
            y=traces[:,j]
            pre=float(np.mean(y[(t>=1.5)&(t<2)])); final=float(np.mean(y[(t>=5)&(t<6)]))
            def crossing(level,edge=2,stop=6,rising=True):
                ids=np.flatnonzero((t>=edge)&(t<stop)&((y>=level) if rising else (y<=level)))
                return float(t[ids[0]]-edge) if len(ids) else None
            t10=crossing(pre+.1*(final-pre)); t90=crossing(pre+.9*(final-pre))
            down=crossing(5+.1*(final-5),6,10,False)
            metrics.append({"rise_10_90_ms":None if t10 is None or t90 is None else 1000*(t90-t10),
                            "step_to_90_ms":None if t90 is None else 1000*t90,
                            "vent_to_10_ms":None if down is None else 1000*down,
                            "steady_bias_psi":final-12,
                            "steady_sd_psi":float(np.std(y[(t>=5)&(t<6)]))})
        result["populations"][label]={"boards":np.asarray(env.ids)[mask].tolist(),
              "per_board":metrics,"median":{k:float(np.median([m[k] for m in metrics if m[k] is not None]))
                                             for k in metrics[0]}}
    env.close()
    return t,command,p,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("deliverable/dynamic_control"))
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    t,command,p,result=run()
    (args.out/"valve_probe.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,3.6),layout="constrained")
    ax.plot(t,command,"--",c="0.35",label="Target")
    for label,mask,color in (("TLE/DVP",np.array(result["environment"]["board_variants"])==2,"#177e89"),
                             ("7 mm",np.array(result["environment"]["board_variants"])!=2,"#b56832")):
        ax.plot(t,p[:,mask].mean(axis=1),label=label,c=color)
        ax.fill_between(t,p[:,mask].min(axis=1),p[:,mask].max(axis=1),color=color,alpha=.16)
    ax.set(xlabel="Simulated time (s)",ylabel="Reported gauge pressure (psi)",
           title="Equal pressure steps | fitted ProMax twin")
    ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.savefig(args.out/"valve_probe.png",dpi=180)
    plt.close(fig)
    print(json.dumps({k:v["median"] for k,v in result["populations"].items()}))


if __name__=="__main__":main()
