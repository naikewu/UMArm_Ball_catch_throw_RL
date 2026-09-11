"""Parallel, hardware-free pressure/joint excitation through the shared sensors."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import numpy as np
from .controller import PA_PER_PSI, differential_to_pressure, project_pressures, make_controller, pair_indices

FAMILIES=("prbs","chirp","multisine","common_mode","ringdown","joint_reference","coupled")


def summarize_campaign(directory):
    """Recompute coverage and safety from saved arrays, independent of trainer."""
    import hashlib
    from .controller import pair_indices
    result={"episodes":0,"transitions":0,"seconds":0.,"families":{},
            "per_joint_min_deg":np.full(12,np.inf),"per_joint_max_deg":np.full(12,-np.inf),
            "per_joint_max_speed_rad_s":np.zeros(12),"max_command_psi":0.,
            "max_pair_command_psi":0.,"command_cap_violations":0,"variant_layouts":[]}
    for path in sorted(Path(directory).glob("episode_*.npz")):
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d["meta"]));x=d["truth"];u=d["action"]
        result["episodes"]+=1;result["transitions"]+=len(u);result["seconds"]+=meta["seconds"]
        family=meta["family"];result["families"][family]=result["families"].get(family,0)+1
        result["per_joint_min_deg"]=np.minimum(result["per_joint_min_deg"],np.rad2deg(x[:,:12].min(axis=0)))
        result["per_joint_max_deg"]=np.maximum(result["per_joint_max_deg"],np.rad2deg(x[:,:12].max(axis=0)))
        result["per_joint_max_speed_rad_s"]=np.maximum(result["per_joint_max_speed_rad_s"],np.abs(x[:,12:24]).max(axis=0))
        pairsum=u[:,pair_indices()].sum(axis=-1)/PA_PER_PSI
        result["max_command_psi"]=max(result["max_command_psi"],float(u.max()/PA_PER_PSI))
        result["max_pair_command_psi"]=max(result["max_pair_command_psi"],float(pairsum.max()))
        result["command_cap_violations"]+=int(np.sum(np.any((u<-.1)|(u>30*PA_PER_PSI+.1),axis=1)|np.any(pairsum>30.0001,axis=1)))
        layout=meta["environment"]["board_variants"]
        if layout not in result["variant_layouts"]: result["variant_layouts"].append(layout)
    for key,value in result.items():
        if isinstance(value,np.ndarray):result[key]=value.tolist()
    repo=Path(__file__).resolve().parents[1]
    result["twin_checkpoints"]={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (repo/"digital_twin/checkpoints/canarm_flow.npz",repo/"digital_twin/checkpoints/canarm_mech.json")}
    result["collection_feedback"]={"kp_psi_per_rad":25,"ki_psi_per_rad_s":8,"kd_psi_s_per_rad":2,"preview_s":.035}
    result["limitations"]="Sampled coverage is not exhaustive modal coverage; parameter ranges and sensor noise are assumed robustness scenarios, not measured confidence intervals. No hardware ran."
    return result


def collect_episode(task):
    index,seed,seconds,out,family_override=task
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from .sim_env import ControlEnv
    rng=np.random.default_rng(seed)
    family=family_override or FAMILIES[index%len(FAMILIES)]
    env=ControlEnv(seed=seed,randomize=(index%4!=0))
    obs=env.reset()
    # Explicit collection gains preserve this campaign when benchmark defaults
    # are subsequently tuned. These are the pre-benchmark exploration gains.
    controller=(make_controller("ff_pid",kp=25.,ki=8.,kd=2.,preview=.035)
                if family in ("joint_reference","disturbance") else None)
    if controller: controller.reset(obs["q"],obs["p_pa"])
    n=round(seconds/env.dt)
    state=np.zeros((n+1,48),np.float32)
    truth=np.zeros_like(state)
    actions=np.zeros((n,24),np.float32)
    phase=rng.uniform(0,2*np.pi,12); freqs=rng.uniform(.12,1.2,12)
    amp=rng.uniform(3,14,12)
    diff=np.zeros(12)
    previous=np.full(24,2*PA_PER_PSI)
    force_amp=rng.uniform(.3,1.5) if family=="disturbance" else 0.
    force_dir=rng.normal(size=3) if family=="disturbance" else np.zeros(3)
    if family=="disturbance": force_dir/=np.linalg.norm(force_dir)
    tip_body=int(env.arm.model.site("canarm_tip").bodyid[0])
    def vector(o): return np.r_[o["q"],o["qdot"],o["p_pa"]]
    def true_vector(o): return np.r_[o["q_true"],o["qdot_true"],o["p_true_pa"]]
    state[0]=vector(obs); truth[0]=true_vector(obs)
    for k in range(n):
        t=k*env.dt; base=6.
        if family=="disturbance":
            pulse=max(0.,np.sin(2*np.pi*.25*t))**4
            env.arm.data.xfrc_applied[tip_body,:3]=force_dir*force_amp*pulse
        if family=="prbs":
            if k%int(rng.choice([15,30,60,120]))==0: diff=rng.choice([-1,0,1],12)*amp
        elif family=="chirp":
            diff=amp*np.sin(2*np.pi*(.04*t+1.0*t*t/seconds)+phase)
        elif family=="multisine":
            diff=amp*(.65*np.sin(2*np.pi*freqs*t+phase)+.35*np.sin(2*np.pi*2.7*freqs*t-phase))
        elif family=="common_mode":
            base=7+5*np.sin(2*np.pi*.22*t)
            diff=amp*.6*np.sin(2*np.pi*freqs*t+phase)
        elif family=="ringdown":
            diff=amp*rng.choice([-1,1],12) if k%450==0 else diff
            if k%450>130: diff=np.zeros(12); base=.4
        elif family=="coupled":
            diff=amp*np.sin(2*np.pi*.3*t+phase)+4*np.sin(2*np.pi*1.8*t)
            base=3+3*(1+np.sin(t))
        if controller:
            omega=2*np.pi*freqs*.45
            qr=np.deg2rad(amp)*np.sin(omega*t+phase)*min(t/2,1)
            qdr=np.deg2rad(amp)*omega*np.cos(omega*t+phase)
            qddr=-np.deg2rad(amp)*omega**2*np.sin(omega*t+phase)
            target=controller.command(obs["q"],obs["qdot"],obs["p_pa"],qr,qdr,qddr)
        else:
            target=differential_to_pressure(diff*PA_PER_PSI,base*PA_PER_PSI)
        # Rich excitation stays bounded; reduce input near the training envelope.
        target=project_pressures(previous+np.clip(target-previous,-2*PA_PER_PSI,2*PA_PER_PSI))
        if np.max(abs(obs["q_true"]))>.65: target=np.full(24,.3*PA_PER_PSI)
        obs=env.step(target)
        previous=target
        state[k+1]=vector(obs); truth[k+1]=true_vector(obs); actions[k]=target
    meta={"episode":index,"seed":seed,"family":family,"dt_s":env.dt,"seconds":seconds,
          "state_order":"q12,qdot12,p24_Pa_gauge","pair_indices":pair_indices().tolist(),"environment":env.meta,
          "max_q_deg":float(np.rad2deg(np.max(abs(truth[:,:12]))))}
    meta["max_force_n"]=env.max_force_n
    meta["command_cap_violations"]=0
    if family=="disturbance": meta["external_force"]={"amplitude_n":float(force_amp),"direction":force_dir.tolist(),"shape":"positive sine^4 pulses .25Hz applied at distal body"}
    path=Path(out)/f"episode_{index:04d}.npz"
    np.savez_compressed(path,state=state,truth=truth,action=actions,meta=json.dumps(meta))
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("control/data/campaign_20260911"))
    p.add_argument("--episodes",type=int,default=84)
    p.add_argument("--seconds",type=float,default=20)
    p.add_argument("--workers",type=int,default=24)
    p.add_argument("--seed",type=int,default=20260911)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--family",choices=FAMILIES+("disturbance",))
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    tasks=[(i,args.seed+1009*i,args.seconds,str(args.out),args.family) for i in range(args.start,args.start+args.episodes)]
    metadata=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(collect_episode,t) for t in tasks]):
            m=f.result(); metadata.append(m)
            print(f"episode {m['episode']} {m['family']}: max |q|={m['max_q_deg']:.1f} deg",flush=True)
    existing={m["episode"]:m for m in metadata}
    for path in args.out.glob("episode_*.npz"):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d["meta"])); existing[m["episode"]]=m
    metadata=sorted(existing.values(),key=lambda m:m["episode"])
    manifest={"kind":"fitted CAN digital-twin training campaign","episodes":metadata,
              "total_seconds":sum(m["seconds"] for m in metadata),"seed":args.seed,
              "assumptions":"240Hz mocap .10deg Gaussian noise, two-frame delay, causal velocity; synchronized150Hz ADC; bounded domain randomization, not fitted uncertainty"}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (args.out/"summary.json").write_text(json.dumps(summarize_campaign(args.out),indent=2),encoding="utf-8")
    print(json.dumps({"episodes":len(metadata),"seconds":manifest["total_seconds"]}))


if __name__=="__main__": main()
