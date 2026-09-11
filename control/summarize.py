"""Aggregate paired benchmark trials and plot held-out Koopman prediction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def aggregate(folder):
    records=[]
    for path in sorted(Path(folder).glob("*.json")):
        record=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record,dict) and "method" in record and "metrics" in record:
            records.append(record)
    groups={}
    hashes={}
    for record in records:
        method=record["method"]
        speed=round(record["trajectory"]["speed_m_s"],8)
        condition="perturbed" if record["randomized"] else "nominal"
        key=(condition,speed,method)
        groups.setdefault(key,[]).append(record)
        for file,sha in record.get("source_sha256",{}).items():
            if file in hashes and hashes[file]!=sha:
                raise ValueError(f"source changed between benchmark runs: {file}")
            hashes[file]=sha
        if record["metrics"]["target_pair_max_psi"]>30+1e-8 or record["max_tendon_force_n"]>=4000-1e-6:
            raise ValueError("invalid experiment envelope")
    rows=[]
    for (condition,speed,method),group in sorted(groups.items()):
        row=dict(condition=condition,speed_m_s=speed,method=method,
                 seeds=sorted(g["seed"] for g in group),n=len(group))
        for name in ("writing_tip_rms_mm","writing_joint_rms_deg","writing_tip_p95_mm",
                     "solver_over_6_667ms_fraction"):
            values=np.array([g["metrics"][name] for g in group])
            row[name]={"mean":float(values.mean()),"sd":float(values.std(ddof=1)) if len(values)>1 else 0,
                       "min":float(values.min()),"max":float(values.max())}
        row["solver_p95_ms_mean"]=float(np.mean([g["metrics"]["solver_ms"]["p95"] for g in group]))
        row["solver_max_ms"]=max(g["metrics"]["solver_ms"]["max"] for g in group)
        row["maximum_joint_deg"]=max(g["metrics"]["max_joint_deg"] for g in group)
        row["maximum_pair_psi"]=max(g["metrics"]["target_pair_max_psi"] for g in group)
        row["maximum_tendon_force_n"]=max(g["max_tendon_force_n"] for g in group)
        rows.append(row)
    for condition,speed,_ in groups:
        expected=[set(g["seed"] for g in groups.get((condition,speed,method),[]))
                  for method in ("pid","ff_pid","koopman_mppi")]
        if not all(seeds==expected[0] and len(seeds)>0 for seeds in expected):
            raise ValueError("controller comparisons must use the same seeds")
    return dict(trials=len(records),rows=rows,source_sha256=hashes,
                interpretation="Paired seeded simulation trials; sample SD, not a stability guarantee.")


def training_figure(report_path,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report=json.loads(Path(report_path).read_text(encoding="utf-8"))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    history=report["history"]
    axes[0].plot([h["epoch"] for h in history],
                 [h["validation_h30_joint_deg"] for h in history],c="#177e89")
    axes[0].axvline(report["metadata"]["best_epoch"],c=".5",ls="--",label="Selected checkpoint")
    axes[0].set(xlabel="Training epoch",ylabel="200 ms joint RMSE (degrees)",
                title="Whole-episode validation")
    axes[0].legend(frameon=False)
    horizons=report["test"]["horizons"]
    x=np.array([int(h) for h in horizons])*1000/150
    for name,label,color in (("model","Learned Koopman","#177e89"),
                              ("persistence","Hold current state","#b56832")):
        axes[1].plot(x,[r[name]["joint_rmse_deg"] for r in horizons.values()],
                     marker="o",label=label,c=color)
    axes[1].set(xlabel="Prediction horizon (ms)",ylabel="Joint RMSE (degrees)",
                title="Untouched test episodes")
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("ProMax Koopman training | simulated sensor and plant dynamics")
    fig.savefig(Path(out)/"koopman_training.png",dpi=180)
    fig.savefig(Path(out)/"koopman_training.svg")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=Path("deliverable/dynamic_control/benchmark"))
    p.add_argument("--out",type=Path,default=Path("deliverable/dynamic_control"))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    result=aggregate(args.folder)
    (args.out/(args.folder.name+"_aggregate.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    training_figure("control/checkpoints/canarm_koopman.json",args.out)
    print(json.dumps(result["rows"],indent=2))


if __name__=="__main__":main()
