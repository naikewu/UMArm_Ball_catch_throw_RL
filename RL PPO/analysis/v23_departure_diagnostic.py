"""Offline nested CV: predict physical departure correction, then ballistics."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from teacher_rl.improved_rl import read_json
from teacher_rl.release_calibration_rl import calibration_rows, stratified_fold, prediction_metrics, grouped_metrics, acceptance_checks


def raw_departure(features):
    x = np.asarray(features)
    p = np.array([x[0], x[1], x[2]]) + .5 * (x[3:6] + x[8:11]) * x[12]
    return np.r_[p, x[8:11]]


def fit_predict(train, holdout, length, penalty):
    x = np.vstack([row["features"] for row in train])
    y = np.vstack([row["departure_delta"] for row in train])
    xt = np.vstack([row["features"] for row in holdout])
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    z, zt = (x-mean)/scale, (xt-mean)/scale
    a = np.c_[np.ones(len(x)), z]
    ridge = np.eye(a.shape[1]) * .25
    ridge[0,0] = 0.
    beta = solve(a.T @ a + ridge, a.T @ y, assume_a="pos")
    pred = np.c_[np.ones(len(xt)), zt] @ beta
    if length:
        k = np.exp(-np.sum((z[:,None]-z[None])**2, axis=-1)/(2*length**2))
        kt = np.exp(-np.sum((zt[:,None]-z[None])**2, axis=-1)/(2*length**2))
        pred += kt @ solve(k+penalty*np.eye(len(x)), y-a@beta, assume_a="pos")
    result = []
    for row, delta in zip(holdout, pred):
        departed = raw_departure(row["features"]) + delta
        p, v = departed[:3], departed[3:]
        flight = (v[2]+np.sqrt(max(v[2]**2+2*9.81*(p[2]-.04),0.)))/9.81
        landing_rel_target = p[:2]+v[:2]*flight
        corrected = row["raw_landing"] + landing_rel_target - row["features"][6:8]
        result.append(dict(trajectory=row["trajectory"], probe_age_s=row["probe_age_s"],
            raw_landing=row["raw_landing"], actual_landing=row["actual_landing"], calibrated_landing=corrected))
    return result


def main():
    root = Path("teacher_runs/v23_contextual/warm_calibration_supplement_v1")
    rows = read_json(root / "controlled_release_episodes.json")
    data = calibration_rows(rows)
    by_seed = {row["seed"]:row for row in rows}
    for row in data:
        result = by_seed[row["seed"]]["result"]
        actual = result["offline_actual_departure"]
        p = np.asarray(actual["position_m"]).copy()
        p[:2] -= result["target_xy"]
        row["departure_delta"] = np.r_[p, actual["velocity_m_s"]] - raw_departure(row["features"])
    folds = {i:[row for row in data if stratified_fold(row,36)==i] for i in range(5)}
    choices = [(0.,1.)]+[(length, penalty) for length in (2.,4.,8.) for penalty in (.01,.1,1.)]
    predictions, reports = [], []
    for outer in range(5):
        development = [i for i in range(5) if i!=outer]
        scores=[]
        for length, penalty in choices:
            inner=[]
            for heldout in development:
                train=[row for i in development if i!=heldout for row in folds[i]]
                inner.extend(fit_predict(train,folds[heldout],length,penalty))
            scores.append(prediction_metrics(inner)["calibrated_mean_m"])
        length,penalty=choices[int(np.argmin(scores))]
        train=[row for i in development for row in folds[i]]
        predicted=fit_predict(train,folds[outer],length,penalty)
        report=dict(fold=outer,length=length,penalty=penalty,**prediction_metrics(predicted))
        predictions.extend(predicted)
        reports.append(report)
        print(report,flush=True)
    pooled=prediction_metrics(predictions)
    trajectories,phases=grouped_metrics(predictions,"trajectory"),grouped_metrics(predictions,"probe_age_s")
    result=dict(pooled=pooled,folds=reports,by_trajectory=trajectories,by_phase=phases,
        checks=acceptance_checks(pooled,reports,trajectories,phases),diagnostic_only=True)
    output=Path("teacher_runs/v23_contextual/warm_diagnostic/departure_nested_cv.json")
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(dict(pooled=pooled,checks=result["checks"]),flush=True)


if __name__ == "__main__":
    main()
