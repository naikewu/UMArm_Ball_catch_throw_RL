"""V23 development-only probe of preserving PD while limiting orbit drive.

Legacy q-governor fades all differential pressures, including stabilizing PD.
This experiment limits the feedforward drive before allocation, retaining PD.
It is not qualified for demonstration collection or PPO.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from .buffered_rl import DEFAULT_ANCHOR, scenarios
from .contextual_calibration import WarmContextualEnv
from .contextual_env import base_config, normalized_action, episode_utility
from .contextual_kernel import load_calibration, source_hash as kernel_hash
from .contextual_warm_sweep import worker as old_worker
from .data import write_json
from .envelope_teacher_env import EnvelopeTeacherConfig
from .envelope_teacher_rl import report
from .improved_rl import _init_worker, file_hash, load_anchor, read_json
from .improved_teacher import ImprovedRecipe


class FeedbackPreservingEnv(WarmContextualEnv):
    def __init__(self, *args, cutoff_deg=30., **kwargs):
        self.cutoff_deg = cutoff_deg
        super().__init__(*args, **kwargs)

    def reset(self, seed):
        super().reset(seed)
        self.handover.q_cap = None
        original_plan = self.drive._plan
        def guarded_plan(q, time):
            # Reserve a 6-degree smooth band and a short velocity look-ahead.
            predicted = np.asarray(q) + .10 * np.asarray(self.drive.st["qd"])
            excursion = np.degrees(np.maximum(np.abs(q), np.abs(predicted))).max()
            factor = float(np.clip((self.cutoff_deg - excursion) / 6., 0., 1.))
            nominal = self.drive.F_max
            self.drive.F_max *= factor
            try:
                return original_plan(q, time)
            finally:
                self.drive.F_max = nominal
        self.drive._plan = guarded_plan
        return self.observation()


def worker(job):
    scenario, anchor_path, calibration_path, config, action, cutoff = job
    if cutoff is None:
        return old_worker((scenario, anchor_path, calibration_path, config, action))
    torch.set_num_threads(1)
    anchor, payload = load_anchor(Path(anchor_path))
    calibration, _ = load_calibration(Path(calibration_path), Path(anchor_path))
    env = FeedbackPreservingEnv(scenario, ImprovedRecipe(**payload["recipe"]), anchor,
        calibration, lambda context: np.asarray(action), EnvelopeTeacherConfig(**config), cutoff_deg=cutoff)
    env.reset(scenario["seed"])
    while not env.done:
        _, _, _, result = env.step(np.zeros(7, dtype=np.float32))
    result["governor_probe"] = dict(cutoff_deg=cutoff, pd_preserved=True, qualified=False)
    return dict(seed=scenario["seed"], scenario=scenario, result=result, utility=episode_utility(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--calibration", type=Path, default=Path("teacher_runs/v23_contextual/warm_kernel_v1/release_calibration_v23_kernel.json"))
    parser.add_argument("--out", type=Path, default=Path("teacher_runs/v23_contextual/governor_probe_v1"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    _, payload = load_anchor(args.init)
    manifest = scenarios(args.episodes, 28100001, 20261030, ImprovedRecipe(**payload["recipe"]))
    config = base_config()
    action = normalized_action(1.4, 1.09).tolist()
    variants = dict(v15_bc=None, original=None, preserve_pd_q26=26., preserve_pd_q30=30., preserve_pd_q34=34.)
    identity = dict(source_hash=hashlib.sha256((kernel_hash()+Path(__file__).read_text()).encode()).hexdigest(),
        anchor_sha256=file_hash(args.init), calibration_sha256=file_hash(args.calibration),
        manifest=manifest, config=asdict(config), action=action, variants=variants, purpose="development_only")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "probe_contract.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Probe source/config changed; use a new directory")
    write_json(path, identity)
    rows = {name: [] for name in variants}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {}
        for scenario in manifest["scenarios"]:
            for name, cutoff in variants.items():
                path = args.out / f"{name}_{scenario['seed']}.json"
                if path.exists():
                    rows[name].append(read_json(path))
                else:
                    data = None if name == "v15_bc" else asdict(config)
                    futures[pool.submit(worker, (scenario, str(args.init.resolve()), str(args.calibration.resolve()), data, action, cutoff))] = (name,path)
        for future in as_completed(futures):
            name, path = futures[future]
            row = future.result()
            rows[name].append(row)
            write_json(path, row)
    comparisons = {name: report(group, rows["v15_bc"]) for name, group in rows.items() if name != "v15_bc"}
    write_json(args.out / "comparison.json", comparisons)
    print({name: dict(summary=value["summary"], max_joint=value["max_joint_deg"],checks=value["checks"])
           for name,value in comparisons.items()}, flush=True)


if __name__ == "__main__":
    main()
