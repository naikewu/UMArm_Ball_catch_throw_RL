"""Plot measured V16 handover timing and post-catch motion from the pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,default=Path("teacher_runs/v16_continuous/pilot_v2"))
    args = parser.parse_args()
    baseline = read(args.pilot/"v15_bc_episodes.json")
    current = read(args.pilot/"continuous_episodes.json")
    reference = {r["seed"]:r for r in baseline}
    pairs = [(reference[r["seed"]]["result"],r["result"]) for r in current
        if r["result"]["hit15"] and reference[r["seed"]]["result"]["hit15"]]
    metrics = {}
    for key in ("catch_to_orbit_s","catch_to_release_s","impact_weld_peak_n",
            "impact_weld_impulse_ns","landing_error_m","pressure_integral_psi_s"):
        a = float(np.mean([a[key] for a,b in pairs]))
        b = float(np.mean([b[key] for a,b in pairs]))
        metrics[key] = dict(v15_bc=a,v16_continuous=b,change_percent=100*(b/a-1))
    selected = [r for r in current if r["result"]["hit15"]]
    pause = max(r["result"]["longest_postcatch_pause_s"] for r in selected)
    result = dict(paired_hit15=len(pairs),metrics=metrics,longest_postcatch_pause_s=pause,
        note="Development pilot, not independent evidence of RL improvement")
    (args.pilot/"continuity_analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig,axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,title in ((axs[0,0],"catch_to_orbit_s","Catch to orbit start"),
            (axs[0,1],"catch_to_release_s","Catch to release")):
        values = [metrics[key]["v15_bc"],metrics[key]["v16_continuous"]]
        bars = ax.bar(["V15 BC","V16 continuous BC"],values,color=["#757575","#138875"])
        ax.bar_label(bars,labels=[f"{v:.3f} s" for v in values],padding=4)
        ax.set(title=title,ylabel="Seconds",ylim=(0,max(values)*1.2))
    for row in selected:
        trace = np.asarray(row["motion_trace"])
        age = trace[:,0]-row["result"]["capture_physics_time"]
        mask = (age>=-.15)&(age<=2.)
        axs[1,0].plot(age[mask],trace[mask,1],alpha=.65,linewidth=1)
        impact = (age>=-.03)&(age<=.08)
        axs[1,1].plot(age[impact]*1000,trace[impact,2],alpha=.65,linewidth=1)
    axs[1,0].axhline(.15,color="#bc4353",linestyle="--",label="Low-speed threshold")
    axs[1,0].set(title="V16 post-catch motion (successful episodes)",xlabel="Seconds from capture",ylabel="Tip speed (m/s)")
    axs[1,0].legend()
    axs[1,1].set(title="Initial grasp-constraint load (not improved yet)",xlabel="Milliseconds from capture",ylabel="Weld force (N)")
    for ax in axs.flat:
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("V16 development pilot: continuous handover, frozen BC task policy")
    fig.savefig(args.pilot/"continuity_comparison.png",dpi=160)
    plt.close(fig)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
