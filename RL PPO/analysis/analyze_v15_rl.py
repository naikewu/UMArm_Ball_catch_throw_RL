"""Read-only V15 run audit, paired uncertainty and reproducible diagnostic plots.

Run from RL PPO: python analysis/analyze_v15_rl.py
Only analysis artifacts are written; checkpoints and fingerprinted code are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUALITY_KEYS = ("weld_peak_n", "impact_weld_impulse_ns", "relative_capture_speed_m_s",
    "pressure_integral_psi_s", "contact_impulse_ns")
QUALITY_WEIGHTS = np.array([.30, .25, .25, .15, .05])
MEASURES = (*QUALITY_KEYS, "impact_weld_peak_n", "contact_peak_n",
    "weld_impulse_ns", "buffer_displacement_m", "pressure_variation_psi",
    "landing_error_m")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_pairs(baseline, candidate):
    reference = {r["seed"]: r for r in baseline}
    if len(reference) != len(candidate) or set(reference) != {r["seed"] for r in candidate}:
        raise ValueError("comparison requires matching unique scenario seeds")
    for r in candidate:
        if r["scenario"] != reference[r["seed"]]["scenario"]:
            raise ValueError("scenario mismatch")
    pairs = [(reference[r["seed"]]["result"], r["result"]) for r in candidate]
    hit_pairs = [(a, b) for a, b in pairs if a["hit15"] and b["hit15"]]
    rng = np.random.default_rng(20260918)
    indices = rng.integers(len(pairs), size=(4000, len(pairs)))
    from scipy.stats import binomtest
    success = {}
    for key in ("captured", "released", "hit15", "hit30"):
        delta = np.array([int(b[key])-int(a[key]) for a, b in pairs])
        lost, gained = int((delta < 0).sum()), int((delta > 0).sum())
        success[key] = dict(bc_count=sum(a[key] for a, _ in pairs),
            rl_count=sum(b[key] for _, b in pairs), lost=lost, gained=gained,
            lost_seeds=[r["seed"] for r in candidate
                if reference[r["seed"]]["result"][key] and not r["result"][key]],
            gained_seeds=[r["seed"] for r in candidate
                if not reference[r["seed"]]["result"][key] and r["result"][key]],
            delta_rate=float(delta.mean()),
            paired_bootstrap_delta_ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist(),
            mcnemar_exact_p=float(binomtest(gained, lost+gained).pvalue) if lost+gained else 1.)
    result = dict(episodes=len(pairs), paired_hit15=len(hit_pairs), success=success,
        quality_population="same scenarios where both policies hit15",
        caveat="paired percentile bootstrap, conditional on common successes; not a proof of equivalence")
    if not hit_pairs:
        return result
    hit_indices = rng.integers(len(hit_pairs), size=(4000, len(hit_pairs)))
    metrics = {}
    for key in MEASURES:
        a = np.array([a[key] for a, b in hit_pairs], dtype=float)
        b = np.array([b[key] for a, b in hit_pairs], dtype=float)
        difference = b-a
        metrics[key] = dict(bc_mean=float(a.mean()), rl_mean=float(b.mean()),
            mean_delta=float(difference.mean()),
            percent_change=float(100*(b.mean()/a.mean()-1)) if a.mean() else None,
            paired_delta_ci95=np.quantile(difference[hit_indices].mean(1), [.025, .975]).tolist(),
            bc_p95=float(np.percentile(a, 95)), rl_p95=float(np.percentile(b, 95)),
            bc_max=float(a.max()), rl_max=float(b.max()),
            bc_nonzero=int((a > 0).sum()), rl_nonzero=int((b > 0).sum()))
    components, replicate_scores = {}, np.zeros(len(hit_indices))
    for key, weight in zip(QUALITY_KEYS, QUALITY_WEIGHTS):
        a = np.array([a[key] for a, b in hit_pairs])
        b = np.array([b[key] for a, b in hit_pairs])
        ratio = (b.mean()+1e-6)/(a.mean()+1e-6)
        components[key] = dict(weight=float(weight), ratio=float(ratio),
            improvement_percentage_points=float(100*weight*(1-ratio)))
        replicate_scores += weight*(b[hit_indices].mean(1)+1e-6)/(a[hit_indices].mean(1)+1e-6)
    result.update(metrics=metrics, quality_components=components,
        quality_score=float(sum(v["weight"]*v["ratio"] for v in components.values())),
        quality_score_ci95=np.quantile(replicate_scores, [.025, .975]).tolist(),
        all_episode_contact_max_bc=max(a["contact_peak_n"] for a, _ in pairs),
        all_episode_contact_max_rl=max(b["contact_peak_n"] for _, b in pairs))
    return result


def offline_actions(root):
    import torch
    from teacher_rl.data import load_dataset
    from teacher_rl.env import LIMITS
    from teacher_rl.improved_teacher import IMPROVED_SCHEMA, improved_fingerprint
    from teacher_rl.model import load_checkpoint
    from teacher_rl.residual_env import RL_SCHEMA, policy_mask
    torch.set_num_threads(1)
    dataset = load_dataset(root.parent / "dataset_1000_formal", expected_schema=IMPROVED_SCHEMA,
        source_hash=improved_fingerprint())["validation"]
    observations, labels, masks, phases, _ = dataset
    masks = policy_mask(masks)
    paths = {"bc": (root.parent / "bc_v1_formal" / "bc_best.pt", IMPROVED_SCHEMA),
        "best": (root / "ppo_best.pt", RL_SCHEMA), "latest": (root / "ppo_latest.pt", RL_SCHEMA)}
    scale = {0: (.15*1000, "mm"), 1: (.15*1000, "mm"), 2: (.15*1000, "mm"),
        3: (.2, "match_ratio"), 4: (8., "psi"), 7: (4., "k_r"),
        8: (10., "degree"), 9: (.03*1000, "ms")}
    report = {"population": "held-out BC demonstration observations, NOT on-policy RL states",
        "retained_samples": len(observations), "models": {}}
    for name, (path, schema) in paths.items():
        model, _ = load_checkpoint(path, expected_schema=schema)
        model.eval()
        with torch.inference_mode():
            prediction = torch.cat([model.distribution(t).mean.tanh()
                for t in torch.as_tensor(observations).split(4096)]).numpy()
        error = prediction-labels
        effective = .25*np.clip(error, -LIMITS, LIMITS)
        stats = {}
        for phase in (0, 1, 2, 3):
            stats[str(phase)] = {}
            for index, (gain, units) in scale.items():
                selected = (phases == phase) & (masks[:, index] > 0)
                if not selected.any():
                    continue
                values = effective[selected, index]*gain
                stats[str(phase)][str(index)] = dict(unit=units, count=int(selected.sum()),
                    signed_mean=float(values.mean()), mean_abs=float(np.abs(values).mean()),
                    p95_abs=float(np.percentile(np.abs(values), 95)),
                    permitted_abs_max=float(.25*LIMITS[index]*gain),
                    clipped_proposal_fraction=float((np.abs(error[selected, index]) >= LIMITS[index]).mean()))
        report["models"][name] = stats
    return report


def plots(updates, validations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    x = [u["update"] for u in updates]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, key, title in zip(axs.flat, ("value_loss", "anchor_kl", "approx_kl", "bc_loss"),
            ("Critic MSE loss", "KL to frozen BC", "PPO old/new approximate KL", "BC auxiliary masked MSE")):
        ax.plot(x, [u["metrics"][key] for u in updates], color="#11776b")
        ax.set(title=title, xlabel="PPO update")
        ax.grid(alpha=.2)
    fig.suptitle("V15 RL v1: optimization diagnostics")
    fig.savefig(output / "training_metrics.png", dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    x = [v["update"] for v in validations]
    for key, label, color in (("captured_rate", "Capture", "#11776b"),
            ("hit15_rate", "Hit within 15 cm", "#b13e54")):
        axs[0, 0].plot(x, [100*v["report"]["summary"][key] for v in validations], "o-", label=label, color=color)
    axs[0, 0].set(title="Fixed 80-scenario validation", ylabel="Percent")
    axs[0, 0].legend()
    axs[0, 1].plot(x, [100*v["report"]["summary"]["mean_landing_error_m"] for v in validations], "o-", color="#a77916")
    axs[0, 1].set(title="Mean landing error (landed episodes)", ylabel="cm")
    for key, weight in zip(QUALITY_KEYS, QUALITY_WEIGHTS):
        values = [100*weight*(1-v["report"]["paired_quality_ratios"][key]) for v in validations]
        axs[1, 0].plot(x, values, "o-", label=key.replace("_", " "))
    axs[1, 0].axhline(0, color="gray", lw=.7)
    axs[1, 0].set(title="Contribution to composite score improvement", ylabel="Percentage points")
    axs[1, 0].legend(fontsize=7)
    axs[1, 1].plot(x, [100*v["report"]["worst_parameter_bin_hit15_rate"] for v in validations], "o-", color="#636363")
    axs[1, 1].axhline(80, color="#b13e54", ls="--", label="Absolute gate")
    axs[1, 1].set(title="Worst one-dimensional parameter bin", ylabel="Hit15 percent")
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel("PPO update")
    fig.suptitle("V15 RL v1: checkpoint selection uses this validation set")
    fig.savefig(output / "validation_metrics.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("teacher_runs/v15_improved/rl_v1_formal"))
    parser.add_argument("--skip-actions", action="store_true")
    args = parser.parse_args()
    root = args.run
    output = root / "analysis_v1"
    output.mkdir(parents=True, exist_ok=True)
    updates = [read(p) for p in sorted(root.glob("update_*.json"))]
    validations = [read(p) for p in sorted(root.glob("validation_*.json"))]
    summary = read(root / "training_summary.json")
    if [u["update"] for u in updates] != list(range(1, summary["completed_updates"]+1)):
        raise ValueError("update log sequence is incomplete")
    config = read(root / "run_config.json")
    rows = [r for u in updates for r in u["rows"]]
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate training seeds")
    if not all(np.isfinite(v) for u in updates for v in u["metrics"].values()):
        raise ValueError("nonfinite training metric")
    baseline = validations[0]["rows"]
    best = next(v for v in validations if v["update"] == summary["best_update"])
    report = dict(training=summary, decision_steps=sum(u["metrics"]["samples"] for u in updates),
        active_decision_steps=sum(u["metrics"]["active_samples"] for u in updates),
        kl_early_stops=[u["update"] for u in updates if u["metrics"]["kl_early_stop"]],
        training_captured=sum(r["result"]["captured"] for r in rows),
        training_hit15=sum(r["result"]["hit15"] for r in rows),
        validation_best=compare_pairs(baseline, best["rows"]),
        validation_latest=compare_pairs(baseline, validations[-1]["rows"]),
        validations=[dict(update=v["update"], **{k: v["report"][k] for k in
            ("summary", "quality_score", "eligible", "worst_parameter_bin_hit15_rate")}) for v in validations])
    bins = {}
    for name, selected in (("full_range", [r for r in rows if r["scenario"]["sampling_group"] == "full_range"]),
            ("mid_speed", [r for r in rows if r["scenario"]["sampling_group"] == "mid_speed"])):
        bins[name] = dict(episodes=len(selected), captured=sum(r["result"]["captured"] for r in selected),
            hit15=sum(r["result"]["hit15"] for r in selected))
    report["training_groups"] = bins
    report["training_quarters"] = []
    for start in (1, 26, 51, 76):
        selected = [u for u in updates if start <= u["update"] < start+25]
        group = [r["result"] for u in selected for r in u["rows"]]
        report["training_quarters"].append(dict(start=start, end=start+24, episodes=len(group),
            hit15=sum(r["hit15"] for r in group), captured=sum(r["captured"] for r in group),
            metric_means={key: float(np.mean([u["metrics"][key] for u in selected])) for key in
                ("value_loss", "anchor_kl", "approx_kl", "bc_loss")},
            mean_task_return=float(np.mean([r["task_return"] for r in group])),
            mean_quality_penalty=float(np.mean([r["quality_penalty"] for r in group]))))
    successful = [r["result"] for r in rows if r["result"]["hit15"]]
    report["quality_cost_mean_successes"] = {k: float(np.mean([r["quality_cost_terms"][k] for r in successful]))
        for k in successful[0]["quality_cost_terms"]}
    report["gae_direct_trace_factor_14s"] = (config["ppo"]["gamma"]*config["ppo"]["gae_lambda"])**420
    actions_path = output / "offline_action_audit.json"
    if not args.skip_actions or not actions_path.exists():
        actions_path.write_text(json.dumps(offline_actions(root), indent=2, allow_nan=False), encoding="utf-8")
    final = root / "evaluation_200"
    if (final / "comparison.json").is_file():
        report["independent_evaluation"] = read(final / "comparison.json")
        reference = read(final / "bc_episodes.json")
        candidate = read(final / "rl_episodes.json")
        report["independent_paired"] = compare_pairs(reference, candidate)
        test_seeds = {r["seed"] for r in reference}
        forbidden = set(seeds) | set(config["excluded_seeds"]) | {r["seed"] for r in baseline}
        if test_seeds & forbidden:
            raise ValueError("independent evaluation leaks previous seeds")
        report["test_seed_disjoint_verified"] = True
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plots(updates, validations, output)
    print(json.dumps(dict(output=str(output), completed_updates=len(updates),
        decision_steps=report["decision_steps"], best_update=summary["best_update"],
        independent_evaluation_available="independent_paired" in report)))


if __name__ == "__main__":
    main()
