"""Development-only held-out action-choice audit of the V23 paired sweep.

No policy is promoted here. Whole captured scenarios form five outer folds;
hyperparameters are selected only within each outer training partition.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from teacher_rl.contextual_model import normalization
from teacher_rl.improved_rl import read_json
from teacher_rl.data import write_json


def predict(train_x, train_y, test_x, mask, length, penalty):
    mean, scale = normalization(train_x)
    x = ((train_x - mean.numpy()) / scale.numpy())[:, mask]
    test = ((test_x - mean.numpy()) / scale.numpy())[:, mask]
    square = np.maximum(0., (x * x).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2. * x @ x.T)
    cross = np.maximum(0., (test * test).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2. * test @ x.T)
    center = train_y.mean(0)
    weights = np.linalg.solve(np.exp(-square / (2. * length ** 2)) + penalty * np.eye(len(x)), train_y - center)
    return center + np.exp(-cross / (2. * length ** 2)) @ weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("teacher_runs/v23_contextual/warm_sweep_v1"))
    parser.add_argument("--out", type=Path, default=Path("teacher_runs/v23_contextual/warm_selector_diagnostic.json"))
    args = parser.parse_args()
    contract = read_json(args.data / "sweep_contract.json")
    x, y, seeds, hits = [], [], [], []
    for scenario in contract["manifest"]["scenarios"]:
        rows = [read_json(args.data / "episodes" / f"{scenario['seed']}_a{i:03d}.json") for i in range(len(contract["actions"]))]
        decision = rows[0]["result"]["trajectory_decision"]
        if decision is None:
            continue
        x.append(decision["context"])
        y.append([row["utility"] for row in rows])
        hits.append([row["result"]["hit15"] for row in rows])
        seeds.append(scenario["seed"])
    x, y, hits = np.asarray(x), np.asarray(y), np.asarray(hits)
    folds = np.array_split(np.random.default_rng(20261031).permutation(len(x)), 5)
    masks = {"target": [0, 1], "target_q_velocity": list(range(26)), "all_observed": list(range(x.shape[1]))}
    candidates = [(name, length, penalty) for name in masks for length in (1., 3., 10.) for penalty in (.1, 1., 10.)]
    decisions, constant_decisions, reports = {}, {}, []
    for outer, test in enumerate(folds):
        train_folds = [part for i, part in enumerate(folds) if i != outer]
        train = np.concatenate(train_folds)
        scores = []
        for name, length, penalty in candidates:
            utilities = []
            for inner, validation in enumerate(train_folds):
                fit = np.concatenate([part for i, part in enumerate(train_folds) if i != inner])
                predictions = predict(x[fit], y[fit], x[validation], masks[name], length, penalty)
                utilities.extend(y[validation, predictions.argmax(-1)].tolist())
            scores.append(np.mean(utilities))
        name, length, penalty = candidates[int(np.argmax(scores))]
        selected = predict(x[train], y[train], x[test], masks[name], length, penalty).argmax(-1)
        constant = int(y[train].mean(0).argmax())
        decisions.update(zip(test.tolist(), selected.tolist()))
        constant_decisions.update((int(i), constant) for i in test)
        reports.append(dict(fold=outer, features=name, length=length, penalty=penalty,
            outer_scenarios=len(test), hit15=int(hits[test, selected].sum()),
            utility=float(y[test, selected].mean())))
    selected = np.array([decisions[i] for i in range(len(x))])
    constant = np.array([constant_decisions[i] for i in range(len(x))])
    idx = np.arange(len(x))
    oracle = y.argmax(-1)
    payload = dict(purpose="DEVELOPMENT DIAGNOSTIC ONLY; not independent closed-loop qualification",
        total_scenarios=len(contract["manifest"]["scenarios"]), captured_contexts=len(x),
        fitted_choices=dict(hit15=int(hits[idx, selected].sum()), mean_utility=float(y[idx, selected].mean())),
        fold_training_best_constant=dict(hit15=int(hits[idx, constant].sum()), mean_utility=float(y[idx, constant].mean())),
        posthoc_oracle=dict(hit15=int(hits[idx, oracle].sum()), mean_utility=float(y[idx, oracle].mean())),
        folds=reports, scenario_seeds=seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
