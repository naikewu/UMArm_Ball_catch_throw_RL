"""GPU Koopman fitting with whole-episode splits and multi-step validation.

Use --warm-start old.pt --anchor-strength 0.01 for a second fit on recorded
real transitions in the same state/action NPZ schema. The original scaler is
preserved, and an optional frozen encoder limits the number of adapted weights.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from .controller import CAP_PA, PA_PER_PSI, DEFAULT_CHECKPOINT, pair_indices
from .koopman import LiftedDynamics, SCHEMA, load_model


def load_episodes(directory):
    episodes=[]
    layout=None
    for path in sorted(Path(directory).glob("episode_*.npz")):
        with np.load(path,allow_pickle=False) as data:
            x=np.asarray(data["state"],dtype=np.float32)
            u=np.asarray(data["action"],dtype=np.float32)
            meta=json.loads(str(data["meta"]))
        if x.shape!=(len(u)+1,48) or u.shape[1:]!=(24,) or not np.isfinite(x).all() or not np.isfinite(u).all():
            raise ValueError(f"invalid state/action episode {path}")
        if abs(meta["dt_s"]-1/150)>1e-10: raise ValueError("expected150Hz episodes")
        if meta.get("state_order")!="q12,qdot12,p24_Pa_gauge":
            raise ValueError(f"state ordering / units are missing or wrong in {path}")
        if meta.get("pair_indices")!=pair_indices().tolist():
            raise ValueError(f"recorded measured antagonist map mismatch: {path}")
        env=meta.get("environment",{})
        if env.get("board_ids")!=list(range(0x101,0x119)):
            raise ValueError(f"record all24 nodes in ascending CAN order: {path}")
        variants=env.get("board_variants")
        if not isinstance(variants,list) or len(variants)!=24 or any(v not in (0,2) for v in variants):
            raise ValueError(f"missing / unsupported recorded board variants: {path}")
        if layout is None:layout=variants
        if variants!=layout:raise ValueError("one model requires one documented valve layout")
        if env.get("control_hz")!=150 or env.get("mocap_hz")!=240 or not env.get("pressure_observation"):
            raise ValueError(f"sensor sampling metadata must describe150Hz pressure/240Hz mocap: {path}")
        for key in ("joint_noise_std_deg","mocap_latency_s","velocity_filter_tau_s"):
            if key not in env or not np.isfinite(env[key]) or env[key]<0:
                raise ValueError(f"missing / invalid sensor contract {key}: {path}")
        if np.any(u<-.1) or np.any(u>CAP_PA+.1) or np.any(u[:,pair_indices()].sum(axis=-1)>CAP_PA+.1):
            raise ValueError(f"unsafe command pressure or wrong units: {path}")
        episodes.append(dict(x=x,u=u,meta=meta,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len(episodes)<7: raise ValueError("at least7 complete episodes required")
    return episodes


def split_episodes(episodes):
    """Stratified complete episodes; never split adjacent150Hz samples."""
    groups={}
    for e in episodes: groups.setdefault(e["meta"]["family"],[]).append(e)
    split={"train":[],"validation":[],"test":[]}
    for family,items in sorted(groups.items()):
        rng=np.random.default_rng(20260911+sum(map(ord,family)))
        items=[items[i] for i in rng.permutation(len(items))]
        nv=max(1,len(items)//6); nt=max(1,len(items)//6)
        if len(items)<3: raise ValueError(f"need at least3 episodes for {family}")
        split["train"]+=items[:len(items)-nv-nt]
        split["validation"]+=items[len(items)-nv-nt:len(items)-nt]
        split["test"]+=items[len(items)-nt:]
    return split


def tensor_episodes(episodes,device):
    return [(torch.tensor(e["x"],device=device),torch.tensor(e["u"],device=device)) for e in episodes]


def windows(episodes,batch,horizon,rng):
    # Stack equal length episodes once on device; each window stays in one row.
    xs,us=episodes
    ids=torch.tensor(rng.integers(0,len(xs),batch),device=xs.device)
    starts=torch.tensor(rng.integers(0,xs.shape[1]-horizon,batch),device=xs.device)
    offsets=torch.arange(horizon+1,device=xs.device)
    x=xs[ids[:,None],starts[:,None]+offsets[None,:]]
    u=us[ids[:,None],starts[:,None]+offsets[None,:-1]]
    return x,u


def stack_episodes(episodes,device):
    lengths={len(e["u"]) for e in episodes}
    if len(lengths)!=1: raise ValueError("training batch requires equal-length episodes; reslice whole episodes first")
    return torch.tensor(np.stack([e["x"] for e in episodes]),device=device),torch.tensor(np.stack([e["u"] for e in episodes]),device=device)


@torch.no_grad()
def ridge_init(model,episodes,ridge=.5):
    x=torch.cat([x[:-1] for x,u in episodes])[::2]
    y=torch.cat([x[1:] for x,u in episodes])[::2]
    u=torch.cat([u for x,u in episodes])[::2]/CAP_PA
    z=model.encode(x).double(); target=model.encode(y).double()
    design=torch.cat((z,u.double(),torch.ones((len(z),1),device=z.device,dtype=z.dtype)),dim=1)
    gram=design.T@design
    weights=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0],device=z.device,dtype=z.dtype),design.T@(target-z)).float()
    model.A.copy_(torch.eye(model.zdim,device=z.device)+weights[:model.zdim].T)
    model.B.copy_(weights[model.zdim:-1].T)
    model.bias.copy_(weights[-1])


@torch.no_grad()
def evaluate(model,episodes,horizons=(1,5,15,24,30,75),max_windows=2048):
    device=model.A.device
    H=max(horizons); predictions={h:[] for h in horizons}; persistence={h:[] for h in horizons}
    populations=None
    per_episode=[]
    for e in episodes:
        x=torch.tensor(e["x"],device=device); u=torch.tensor(e["u"],device=device)
        starts=torch.arange(0,len(u)-H,max(1,(len(u)-H)//max(1,max_windows//len(episodes))),device=device)
        z=model.encode(x[starts]); errors={}
        for h in range(1,H+1):
            z=model.step(z,u[starts+h-1])
            if h in horizons:
                err=(model.decode(z)-x[starts+h]).cpu().numpy()
                predictions[h].append(err)
                persistence[h].append((x[starts]-x[starts+h]).cpu().numpy())
                errors[str(h)]=float(np.rad2deg(np.sqrt(np.mean(err[:,:12]**2))))
        per_episode.append({"episode":e["meta"]["episode"],"family":e["meta"]["family"],"joint_rmse_deg":errors})
        variants=e["meta"]["environment"]["board_variants"]
        # Environment metadata stores hexadecimal node keys and variant bytes.
        if isinstance(variants,dict):
            populations=np.array([variants.get(f"0x{b:03x}",variants.get(str(b),0))==2 for b in range(0x101,0x119)])
        else: populations=np.array(variants)==2
    def metrics(err):
        values={"joint_rmse_deg":float(np.rad2deg(np.sqrt(np.mean(err[:,:12]**2)))),
                "velocity_rmse_rad_s":float(np.sqrt(np.mean(err[:,12:24]**2))),
                "pressure_rmse_pa":float(np.sqrt(np.mean(err[:,24:]**2))),
                "per_joint_rmse_deg":np.rad2deg(np.sqrt(np.mean(err[:,:12]**2,axis=0))).tolist()}
        if populations is not None:
            for label,mask in (("tle",populations),("seven_mm",~populations)):
                values[f"pressure_{label}_rmse_pa"]=float(np.sqrt(np.mean(err[:,24:][:,mask]**2))) if mask.any() else None
        return values
    return {"horizons":{str(h):{"model":metrics(np.concatenate(predictions[h])),"persistence":metrics(np.concatenate(persistence[h]))} for h in horizons},"episodes":per_episode}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("control/data/campaign_20260911"))
    p.add_argument("--out",type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--horizon",type=int,default=30)
    p.add_argument("--device",default="cuda")
    p.add_argument("--warm-start",type=Path)
    p.add_argument("--freeze-encoder",action="store_true")
    p.add_argument("--anchor-strength",type=float,default=0)
    p.add_argument("--lr",type=float,default=None,help="default2e-4; warm-start2e-5")
    args=p.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); torch.manual_seed(20260911); np.random.seed(20260911)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(args.device); episodes=load_episodes(args.data); splits=split_episodes(episodes)
    train_tensor=tensor_episodes(splits["train"],device)
    if args.warm_start:
        model,oldmeta=load_model(args.warm_start,device=device)
        if oldmeta.get("variants")!=episodes[0]["meta"]["environment"]["board_variants"]:
            raise ValueError("warm-start valve layout differs from recorded variant bytes")
    else:
        model=LiftedDynamics().to(device)
        with torch.no_grad():
            x=torch.cat([x for x,u in train_tensor])
            model.xmean.copy_(x.mean(0))
            floor=torch.tensor(np.r_[np.full(12,.03),np.full(12,.15),np.full(24,5000.)],device=device)
            model.xscale.copy_(torch.maximum(x.std(0),floor))
        ridge_init(model,train_tensor)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters(): parameter.requires_grad_(False)
    anchors={n:p.detach().clone() for n,p in model.named_parameters()}
    learning_rate=args.lr if args.lr is not None else (2e-5 if args.warm_start else 2e-4)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=learning_rate/10)
    tensor_train=stack_episodes(splits["train"],device)
    baseline=evaluate(model,splits["validation"],horizons=(1,5,15,30),max_windows=768)
    print("initial",json.dumps(baseline["horizons"]),flush=True)
    best=baseline["horizons"]["30"]["model"]["joint_rmse_deg"]+.00002*baseline["horizons"]["30"]["model"]["pressure_rmse_pa"]
    best_state=copy.deepcopy(model.state_dict()); best_epoch=0; history=[]
    running_meta={"schema":SCHEMA,"status":"running","dt_s":1/150,"state_order":"q12,qdot12,p24_Pa_gauge",
                  "pair_indices":pair_indices().tolist(),"variants":episodes[0]["meta"]["environment"]["board_variants"]}
    torch.save({"config":model.config(),"state_dict":model.state_dict(),"meta":running_meta},args.out.with_suffix(".running.pt"))
    rng=np.random.default_rng(20260911)
    weights=torch.tensor(np.r_[np.full(12,3.),np.full(12,.15),np.full(24,.4)],device=device)
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for _ in range(args.steps):
            x,u=windows(tensor_train,args.batch,args.horizon,rng)
            z=model.encode(x[:,0]); loss=0
            for h in range(args.horizon):
                z=model.step(z,u[:,h])
                target=(x[:,h+1]-model.xmean)/model.xscale
                loss=loss+((z[:,:48]-target).square()*weights).mean()
                if h in (0,4,14,29):
                    target_lift=model.encode(x[:,h+1]).detach()
                    loss=loss+.02*(z[:,48:]-target_lift[:,48:]).square().mean()
            loss=loss/args.horizon
            if args.anchor_strength:
                loss=loss+args.anchor_strength*sum((param-anchors[n]).square().mean() for n,param in model.named_parameters())
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); model.eval()
        val=evaluate(model,splits["validation"],horizons=(1,15,30),max_windows=768)
        score=val["horizons"]["30"]["model"]["joint_rmse_deg"]+.00002*val["horizons"]["30"]["model"]["pressure_rmse_pa"]
        row={"epoch":epoch+1,"loss":float(np.mean(losses)),"validation_score":score,
             "validation_h30_joint_deg":val["horizons"]["30"]["model"]["joint_rmse_deg"],
             "validation_h30_pressure_pa":val["horizons"]["30"]["model"]["pressure_rmse_pa"]}
        history.append(row); print(json.dumps(row),flush=True)
        if score<best and np.isfinite(score):
            best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            torch.save({"config":model.config(),"state_dict":best_state,"meta":dict(running_meta,best_epoch=best_epoch)},args.out.with_suffix(".running.pt"))
    model.load_state_dict(best_state)
    validation=evaluate(model,splits["validation"]); test=evaluate(model,splits["test"])
    meta={"schema":SCHEMA,"status":"complete","dt_s":1/150,"state_order":"q12,qdot12,p24_Pa_gauge",
          "pair_indices":pair_indices().tolist(),"pressure_scale_pa":CAP_PA,
          "population_source":"per-episode environment variant byte, never CAN id range",
          "variants":episodes[0]["meta"]["environment"]["board_variants"],
          "sensor_contract":{k:v for k,v in episodes[0]["meta"]["environment"].items() if k not in ("seed","randomization")},
          "batch_size":args.batch,"optimizer_steps_per_epoch":args.steps,"optimizer":"AdamW,cosine",
          "learning_rate":learning_rate,"minimum_learning_rate":learning_rate/10,
          "seed":20260911,"best_epoch":best_epoch,"epochs":args.epochs,"horizon":args.horizon,
          "device":torch.cuda.get_device_name(device) if device.type=="cuda" else str(device),
          "torch_version":torch.__version__,"elapsed_s":time.perf_counter()-started,
          "warm_start":str(args.warm_start) if args.warm_start else None,
          "freeze_encoder":args.freeze_encoder,"anchor_strength":args.anchor_strength,
          "data":{s:[{"episode":e["meta"]["episode"],"family":e["meta"]["family"],"sha256":e["sha256"]} for e in es] for s,es in splits.items()},
          "sample_count":sum(len(e["u"]) for e in episodes),
          "limitations":"Simulation-trained; no new physical-arm validation. Finite dictionary and bounded innovation are not stability guarantees."}
    torch.save({"config":model.config(),"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},"meta":meta},args.out)
    report={"metadata":meta,"initial_validation":baseline,"validation":validation,"test":test,"history":history}
    args.out.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("saved",str(args.out),"best_epoch",best_epoch,"elapsed",meta["elapsed_s"],flush=True)


if __name__=="__main__": main()
