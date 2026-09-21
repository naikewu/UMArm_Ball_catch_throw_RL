"""V23 warmed release data supplementation and nested-CV nonlinear calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_calibration import source_hash as warm_source_hash, worker as probe_worker
from .contextual_rl import previous_seeds
from .data import write_json
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe
from .release_calibration import RidgeLandingCalibration, fit_ridge
from .release_calibration_rl import (calibration_rows, stratified_fold, evaluate_model,
    prediction_metrics, grouped_metrics, acceptance_checks)
from .trajectory_envelope_rl import trajectories, PHASES, EnvelopeProbeConfig

CAMPAIGN_SCHEMA = "can_warmed_kernel_calibration_v23"


def source_hash():
    digest = hashlib.sha256(warm_source_hash().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


class KernelCalibration(RidgeLandingCalibration):
    def predict_delta(self, features):
        normalized = (np.asarray(features, dtype=float) - self.mean) / self.scale
        distances = np.sum((self.centres - normalized) ** 2, axis=-1)
        return super().predict_delta(features) + np.exp(-distances / (2 * self.length ** 2)) @ self.weights

    def to_dict(self):
        return dict(super().to_dict(), kernel_centres=self.centres.tolist(),
            kernel_weights=self.weights.tolist(), kernel_length=self.length, kernel_penalty=self.penalty)

    @classmethod
    def from_dict(cls, payload):
        base = RidgeLandingCalibration.from_dict(payload)
        model = cls(base.mean, base.scale, base.coefficients, base.ridge_lambda)
        centres, weights = np.asarray(payload["kernel_centres"]), np.asarray(payload["kernel_weights"])
        length, penalty = float(payload["kernel_length"]), float(payload["kernel_penalty"])
        if (centres.ndim != 2 or centres.shape[1] != len(base.mean) or weights.shape != (len(centres), 2)
                or not np.isfinite(np.r_[centres.ravel(), weights.ravel(), length, penalty]).all()
                or length <= 0. or penalty <= 0.):
            raise ValueError("Invalid kernel calibration")
        for name, value in (("centres", centres), ("weights", weights), ("length", length), ("penalty", penalty)):
            object.__setattr__(model, name, value)
        return model


def fit_kernel(data, length, penalty):
    x = np.vstack([row["features"] for row in data])
    y = np.vstack([row["actual_landing"] - row["raw_landing"] for row in data])
    base = fit_ridge(x, y)
    z = (x - base.mean) / base.scale
    residual = y - np.vstack([base.predict_delta(feature) for feature in x])
    kernel = np.exp(-np.sum((z[:, None] - z[None]) ** 2, axis=-1) / (2 * length ** 2))
    weights = solve(kernel + penalty * np.eye(len(x)), residual, assume_a="pos")
    return KernelCalibration.from_dict(dict(**base.to_dict(), kernel_centres=z.tolist(),
        kernel_weights=weights.tolist(), kernel_length=length, kernel_penalty=penalty))


def supplement(args):
    contract = read_json(args.data / "collection_contract.json")
    if contract["source_hash"] != warm_source_hash():
        raise ValueError("Original warmed collection source changed")
    original = read_json(args.data / "controlled_release_episodes.json")
    data = calibration_rows(original)
    # Select missing cells by sample COUNT only, before collecting their labels.
    deficient = [(name, age) for name in contract["trajectories"] for age in PHASES
        if sum(row["trajectory"] == name and row["probe_age_s"] == age for row in data) < 4]
    if not deficient:
        raise ValueError("No undersampled trajectory/phase cells need supplementation")
    _, payload = load_anchor(args.init)
    manifest = scenarios(len(deficient) * args.repeats, args.seed, args.design_seed,
                         ImprovedRecipe(**payload["recipe"]))
    cells = [(name, age) for name in contract["trajectories"] for age in PHASES]
    assignments = []
    for index, scenario in enumerate(manifest["scenarios"]):
        name, age = deficient[index % len(deficient)]
        repeat = contract["repeats"] + index // len(deficient)
        scenario["scenario_id"] = repeat * len(cells) + cells.index((name, age))
        assignments.append((name, age))
    identity = dict(schema=CAMPAIGN_SCHEMA, warm_source_hash=warm_source_hash(),
        base_contract_sha256=file_hash(args.data / "collection_contract.json"), manifest=manifest,
        assignments=assignments, repeats=args.repeats, anchor_sha256=file_hash(args.init))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "supplement_contract.json"
    if path.exists():
        old = read_json(path)
        if old != dict(identity, assignments=[list(item) for item in assignments]):
            raise ValueError("Supplement contract changed")
    elif {row["seed"] for row in manifest["scenarios"]} & previous_seeds(args.out):
        raise ValueError("Supplement seeds overlap")
    write_json(path, identity)
    output = args.out / "episodes"
    output.mkdir(exist_ok=True)
    rows, jobs = {}, []
    for scenario, (name, age) in zip(manifest["scenarios"], assignments):
        path = output / f"episode_{scenario['seed']}.json"
        if path.exists():
            rows[scenario["seed"]] = read_json(path)
        else:
            config = EnvelopeProbeConfig(**trajectories()[name], probe_name=name, probe_age_s=age)
            jobs.append((scenario, str(args.init.resolve()), asdict(config)))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for future in as_completed([pool.submit(probe_worker, job) for job in jobs]):
            row = future.result()
            rows[row["seed"]] = row
            write_json(output / f"episode_{row['seed']}.json", row)
    combined = original + [rows[s["seed"]] for s in manifest["scenarios"]]
    write_json(args.out / "controlled_release_episodes.json", combined)
    print(dict(original=len(original), supplemented=len(rows), total=len(combined)), flush=True)


def fit(args):
    contract = read_json(args.data / "supplement_contract.json")
    if contract["warm_source_hash"] != warm_source_hash() or contract["anchor_sha256"] != file_hash(args.init):
        raise ValueError("Supplemented calibration data are stale")
    rows = read_json(args.data / "controlled_release_episodes.json")
    data = calibration_rows(rows)
    if len(data) < 150:
        raise ValueError("Too few calibration labels")
    folds = {i: [row for row in data if stratified_fold(row, 36) == i] for i in range(5)}
    candidates = [(length, penalty) for length in (1., 2., 4., 8.) for penalty in (.01, .1, 1.)]
    predictions, reports = [], []
    for outer in range(5):
        dev = [i for i in range(5) if i != outer]
        scores = []
        for length, penalty in candidates:
            inner_predictions = []
            for inner in dev:
                train = [row for i in dev if i != inner for row in folds[i]]
                inner_predictions.extend(evaluate_model(fit_kernel(train, length, penalty), folds[inner])[1])
            scores.append(prediction_metrics(inner_predictions)["calibrated_mean_m"])
        length, penalty = candidates[int(np.argmin(scores))]
        train = [row for i in dev for row in folds[i]]
        metric, predicted = evaluate_model(fit_kernel(train, length, penalty), folds[outer])
        reports.append(dict(fold=outer, length=length, penalty=penalty, **metric))
        predictions.extend(predicted)
    pooled = prediction_metrics(predictions)
    by_trajectory, by_phase = grouped_metrics(predictions, "trajectory"), grouped_metrics(predictions, "probe_age_s")
    checks = acceptance_checks(pooled, reports, by_trajectory, by_phase)
    counts = {f"{name}|{age:g}": sum(row["trajectory"] == name and row["probe_age_s"] == age for row in data)
        for name in trajectories() for age in PHASES}
    checks["every_cell_rows_ge_4"] = all(count >= 4 for count in counts.values())
    checks["all_labels_warmed"] = all(row["result"].get("calibration_probe", {}).get("warm_history", {}).get("rate_frames", 0) >= 20
        for row in rows if row["result"].get("calibration_probe", {}).get("scheduled"))
    checks["all_joint_angles_le_36"] = all(row["result"]["max_joint_deg"] <= 36. for row in rows)
    checks["no_grip_broken"] = not any(row["result"]["grip_broken"] for row in rows)
    # Deployment hyperparameters selected on all development folds; the above
    # reported errors are outer holdouts with separate inner model selection.
    scores = []
    for length, penalty in candidates:
        predicted = []
        for fold in range(5):
            train = [row for i in range(5) if i != fold for row in folds[i]]
            predicted.extend(evaluate_model(fit_kernel(train, length, penalty), folds[fold])[1])
        scores.append(prediction_metrics(predicted)["calibrated_mean_m"])
    length, penalty = candidates[int(np.argmin(scores))]
    model = fit_kernel(data, length, penalty)
    payload = dict(**model.to_dict(), campaign_schema=CAMPAIGN_SCHEMA, fit_source_hash=source_hash(),
        anchor_sha256=file_hash(args.init), data_sha256=file_hash(args.data / "controlled_release_episodes.json"),
        calibration_seeds=[row["seed"] for row in rows], deployment_fit_rows=len(data),
        validation_protocol="nested_5fold_outer_4fold_inner_warmed_v23", accepted=all(checks.values()), checks=checks,
        cross_validation=dict(pooled=pooled, folds=reports, by_trajectory=by_trajectory, by_phase=by_phase, cell_rows=counts))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "release_calibration_v23_kernel.json", payload)
    print(dict(accepted=payload["accepted"], pooled=pooled, checks=checks), flush=True)


def load_calibration(path, anchor):
    payload = read_json(path)
    if payload.get("campaign_schema") != CAMPAIGN_SCHEMA:
        from .contextual_calibration import load_calibration as load_linear
        return load_linear(path, anchor)
    if (not payload.get("accepted") or payload.get("fit_source_hash") != source_hash() or
            payload.get("anchor_sha256") != file_hash(anchor) or not all(payload.get("checks", {}).values())):
        raise ValueError("Warmed kernel calibration failed or does not match source/anchor")
    return KernelCalibration.from_dict(payload), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("supplement", "fit"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=27110001)
    parser.add_argument("--design-seed", type=int, default=20261022)
    args = parser.parse_args()
    (supplement if args.command == "supplement" else fit)(args)


if __name__ == "__main__":
    main()
