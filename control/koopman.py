"""State-inclusive, controlled Koopman dictionary and pressure-space MPPI.

z=[normalised q, qdot, p, learned observables]; z_next=A z+B u+b.
The identity coordinates prevent an encoder from losing physical state. A
low-rank bilinear term lets the input gain depend on the current state. This
is a finite learned approximation, not a claim of Koopman-invariant closure.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .controller import (PA_PER_PSI, CAP_PA, DEFAULT_CHECKPOINT,
                         FeedforwardPIDController, pair_indices,
                         project_pressures, assert_pressures)

SCHEMA = "canarm_control_koopman_v1"


class LiftedDynamics(nn.Module):
    def __init__(self, hidden=96, learned=48, rank=8):
        super().__init__()
        self.hidden,self.learned,self.rank = hidden,learned,rank
        self.zdim = 48+learned
        self.encoder=nn.Sequential(nn.Linear(48,hidden),nn.Tanh(),nn.Linear(hidden,learned),nn.Tanh())
        self.A=nn.Parameter(torch.eye(self.zdim))
        self.B=nn.Parameter(torch.zeros(self.zdim,24))
        self.bias=nn.Parameter(torch.zeros(self.zdim))
        self.V=nn.Parameter(torch.randn(self.zdim,rank)/np.sqrt(max(rank,1)))
        self.W=nn.Parameter(torch.zeros(24*rank,self.zdim))
        self.register_buffer("xmean",torch.zeros(48))
        self.register_buffer("xscale",torch.ones(48))

    def encode(self,x):
        xn=(x-self.xmean)/self.xscale
        return torch.cat((xn,self.encoder(xn)),dim=-1)

    def step(self,z,u):
        un=u/CAP_PA
        cross=((z@self.V).unsqueeze(-2)*un.unsqueeze(-1)).flatten(-2)
        return z@self.A.T+un@self.B.T+self.bias+cross@self.W

    def decode(self,z):
        return z[...,:48]*self.xscale+self.xmean

    def config(self):
        return dict(hidden=self.hidden,learned=self.learned,rank=self.rank)


def load_model(path=DEFAULT_CHECKPOINT, device="cpu", dt=1/150):
    payload=torch.load(Path(path),map_location=device,weights_only=False)
    meta=payload["meta"]
    if meta.get("schema")!=SCHEMA or abs(meta.get("dt_s",0)-dt)>1e-10:
        raise ValueError("Koopman checkpoint schema or control period mismatch")
    if meta.get("status")!="complete":
        raise ValueError("Koopman checkpoint is not marked complete")
    if meta.get("state_order") != "q12,qdot12,p24_Pa_gauge" or meta.get("pair_indices")!=pair_indices().tolist():
        raise ValueError("Koopman checkpoint state units or measured pair map mismatch")
    if meta.get("pressure_scale_pa",CAP_PA)!=CAP_PA:
        raise ValueError("Koopman checkpoint pressure normalization mismatch")
    model=LiftedDynamics(**payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()) or torch.any(model.xscale<=0):
        raise ValueError("Koopman checkpoint has nonfinite weights or invalid state scales")
    model.eval()
    return model,meta


class NumpyPredictor:
    """Same learned weights as torch, with small CPU GEMMs for low solve latency."""
    def __init__(self,model):
        self.w={k:v.detach().cpu().numpy().astype(np.float64) for k,v in model.state_dict().items()}
        self.rank=model.rank

    def encode(self,x):
        w=self.w
        xn=(x-w["xmean"])/w["xscale"]
        h=np.tanh(xn@w["encoder.0.weight"].T+w["encoder.0.bias"])
        h=np.tanh(h@w["encoder.2.weight"].T+w["encoder.2.bias"])
        return np.concatenate((xn,h),axis=-1)

    def step(self,z,u):
        w=self.w; un=u/CAP_PA
        cross=((z@w["V"])[...,None,:]*un[...,None]).reshape(z.shape[:-1]+(24*self.rank,))
        return z@w["A"].T+un@w["B"].T+w["bias"]+cross@w["W"]

    def decode(self,z):
        return z[...,:48]*self.w["xscale"]+self.w["xmean"]


class KoopmanMPPIController(FeedforwardPIDController):
    """MPPI plans12 antagonist differences on every synchronized150Hz observation.

The fitted feedforward PID supplies an action prior; weighted sampled
corrections are chosen by the freshly trained model. A bounded innovation
observer estimates slow model mismatch from observed state transitions.
"""
    def __init__(self,dt=1/150,checkpoint=DEFAULT_CHECKPOINT,twin_kwargs=None,
                 seed=20260911,horizon=16,samples=64,adapt=True,
                 action_regularization=.1,noise_psi=.5,temperature_min=.0001,
                 preview_actions=False,**kwargs):
        super().__init__(dt=dt,twin_kwargs=twin_kwargs,**kwargs)
        from threadpoolctl import threadpool_limits
        self._blas=threadpool_limits(limits=1,user_api="blas")
        model,self.model_meta=load_model(checkpoint,dt=dt)
        variants=[self.dynamics.arm.nodes[b].variant for b in sorted(self.dynamics.arm.nodes)]
        if self.model_meta.get("variants")!=variants:
            raise ValueError("Koopman checkpoint valve variants differ from the supplied twin")
        self.predictor=NumpyPredictor(model)
        self.horizon,self.samples=int(horizon),int(samples)
        self.prediction_dt=self.dt
        self.seed=int(seed)
        self.rng=np.random.default_rng(seed)
        self.adapt=bool(adapt)
        self.action_regularization=float(action_regularization)
        self.noise_psi=float(noise_psi)
        self.temperature_min=float(temperature_min)
        self.preview_actions=bool(preview_actions)
        self.reset(np.zeros(12),np.zeros(24))

    def reset(self,q,p_pa):
        super().reset(q,p_pa)
        if hasattr(self,"seed"):self.rng=np.random.default_rng(self.seed)
        self.correction=np.zeros((getattr(self,"horizon",24),12))
        self.innovation=np.zeros(48)
        self.predicted=None

    def command(self,q,qdot,p_pa,q_ref,qd_ref,qdd_ref,future_q=None):
        started=time.perf_counter()
        prior=super().command(q,qdot,p_pa,q_ref,qd_ref,qdd_ref,future_q)
        x=np.concatenate((q,qdot,p_pa))
        model=self.predictor; H,K=self.horizon,self.samples
        if self.adapt and self.predicted is not None:
            bound=np.r_[np.full(12,.002),np.full(12,.04),np.full(24,1000.)]
            self.innovation=.96*self.innovation+.04*np.clip(x-self.predicted,-bound,bound)
        refs=(np.asarray(future_q,dtype=float) if future_q is not None else
              np.asarray(q_ref)+(np.arange(1,H+1)*self.dt)[:,None]*np.asarray(qd_ref))
        if len(refs)<H:
            refs=np.concatenate((refs,np.repeat(refs[-1:],H-len(refs),axis=0)))
        refs=refs[:H]
        self.correction[:-1]=self.correction[1:]
        self.correction[-1]=self.correction[-2]
        # Four temporal knots suppress sample-to-sample pressure chatter.
        knots=self.rng.normal(size=(K,4,12))*self.noise_psi*PA_PER_PSI
        ts=np.linspace(0,3,H)
        lo=np.minimum(ts.astype(int),2); alpha=ts-lo
        noise=knots[:,lo]*(1-alpha)[None,:,None]+knots[:,lo+1]*alpha[None,:,None]
        delta=.75*self.correction[None,:,:]+noise
        delta[0]=0 # The prior is explicitly a candidate, not presumed superior.
        delta[1]=self.correction
        prior_h=np.repeat(prior[None,:],H,axis=0)
        if self.preview_actions:
            future_p=[]
            for t in np.linspace(0,(H-1)*self.dt,4):
                ahead=t+self.preview
                qr=np.asarray(q_ref)+ahead*np.asarray(qd_ref)+.5*ahead*ahead*np.asarray(qdd_ref)
                qdr=np.asarray(qd_ref)+ahead*np.asarray(qdd_ref)
                target,_=self.dynamics.allocate(qr,qdr,qdd_ref,p_pa)
                future_p.append(target)
            future_p=np.asarray(future_p)
            prior_h+=future_p[lo]*(1-alpha)[:,None]+future_p[lo+1]*alpha[:,None]-future_p[0]
        actions=np.repeat(prior_h[None,:,:],K,axis=0)
        pairs=pair_indices()
        actions[:,:,pairs[:,0]]+=delta/2
        actions[:,:,pairs[:,1]]-=delta/2
        actions=project_pressures(actions)
        z=np.repeat(model.encode(x)[None,:],K,axis=0)
        cost=np.zeros(K)
        normalized_innovation=self.innovation/model.w["xscale"]
        for h in range(H):
            z=model.step(z,actions[:,h])
            z[:,:48]+=normalized_innovation
            xp=model.decode(z)
            err=xp[:,:12]-refs[h]
            cost+=(np.mean((err/.08)**2,axis=1)+.015*np.mean((xp[:,12:24]-qd_ref)**2,axis=1))*(2 if h==H-1 else 1)
        cost/=H
        cost+=self.action_regularization*np.mean((delta/(5*PA_PER_PSI))**2,axis=(1,2))
        cost+=.004*np.mean(((actions[:,0]-p_pa)/(5*PA_PER_PSI))**2,axis=1)
        cost=np.nan_to_num(cost,nan=1e12,posinf=1e12,neginf=1e12)
        temperature=max(self.temperature_min,.25*float(np.std(cost)))
        weights=np.exp(np.clip(-(cost-cost.min())/temperature,-60,0)); weights/=weights.sum()
        self.correction=np.einsum("k,khd->hd",weights,delta)
        result=project_pressures(np.einsum("k,kd->d",weights,actions[:,0]))
        self.predicted=model.decode(model.step(model.encode(x),result))
        assert_pressures(result)
        self.last_p=result
        self.last_diagnostics={"solve_ms":1000*(time.perf_counter()-started),
            "effective_samples":float(1/(weights@weights)),"prior_cost":float(cost[0]),
            "best_cost":float(cost.min()),"innovation_q_rad":float(np.linalg.norm(self.innovation[:12]))}
        return result
