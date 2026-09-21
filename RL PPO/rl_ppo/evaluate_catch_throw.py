"""Evaluate a hierarchical PPO checkpoint on catch-and-throw episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .catch_throw_env import CatchThrowEnv, CatchThrowEnvConfig
from .catch_throw_task import DEFAULT_TARGET_OFFSET_WORLD_M
from .networks import ActorCritic
from .train_catch_throw import COMPATIBLE_CHECKPOINT_SCHEMAS


def _metric(info: dict, record_key: str, metric_key: str) -> float | None:
    record = info.get(record_key)
    if not record:
        return None
    value = record.get(metric_key)
    return None if value is None else float(value)


def _mean(records: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--out", type=Path, default=Path("eval/catch_throw"))
    parser.add_argument("--controller", choices=("koopman_mppi", "ff_pid"), default="koopman_mppi")
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=DEFAULT_TARGET_OFFSET_WORLD_M)
    parser.add_argument("--target-radius", type=float, default=.060)
    parser.add_argument("--curriculum-stage", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--stage2-level", type=int, choices=range(1, 4), default=3)
    parser.add_argument(
        "--grasp-mode", choices=("auto_volume", "physical_dwell"),
        help="override the grasp mode stored in the checkpoint",
    )
    parser.add_argument("--auto-grasp-radius", type=float)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") not in COMPATIBLE_CHECKPOINT_SCHEMAS:
        raise ValueError("checkpoint schema does not match the current catch-and-throw model")
    if payload.get("observation_size") != CatchThrowEnv.observation_size:
        raise ValueError("checkpoint observation schema does not match this environment")
    model = ActorCritic(CatchThrowEnv.observation_size, CatchThrowEnv.action_size)
    model.load_state_dict(payload["model"])
    model.eval()
    checkpoint_config = payload.get("config", {})
    grasp_mode = args.grasp_mode or checkpoint_config.get(
        "grasp_mode", "physical_dwell"
    )
    auto_grasp_radius = (
        args.auto_grasp_radius
        if args.auto_grasp_radius is not None
        else checkpoint_config.get("auto_grasp_radius")
    )
    env = CatchThrowEnv(CatchThrowEnvConfig(seed=args.seed, controller=args.controller,
                                             target_offset_world_m=tuple(args.target_offset),
                                             target_radius_m=args.target_radius,
                                             grasp_mode=grasp_mode,
                                             auto_grasp_radius_m=auto_grasp_radius,
                                             curriculum_stage=args.curriculum_stage,
                                             stage2_level=args.stage2_level))
    records = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                with torch.no_grad():
                    action, _, _, _ = model.act(
                        torch.as_tensor(observation[None], dtype=torch.float32),
                        deterministic=True,
                        action_mask=torch.as_tensor(env.action_mask()),
                    )
                observation, reward, terminated, truncated, info = env.step(action[0].numpy())
                total_reward += reward
            records.append({"episode": episode, "return": total_reward, "success": terminated,
                            "captured": info["captured"], "released": info["released"],
                            "stable_capture": info["stable_capture"],
                            "physical_contact_occurred": info[
                                "physical_contact_occurred"
                            ],
                            "compatible_contact_occurred": info[
                                "compatible_contact_occurred"
                            ],
                            "auto_capture": info["auto_capture_occurred"],
                            "physical_capture": info["physical_capture_occurred"],
                            "physical_contact_at_grasp": info["physical_contact_at_grasp"],
                            "grasp_event_physical_compatible": info[
                                "physical_compatible_at_grasp"
                            ],
                            "target_distance_m": info["target_distance_m"],
                            "predicted_landing_error_m": info["predicted_landing_error_m"],
                            "release_velocity_error_mps": _metric(
                                info, "release_event_metrics", "velocity_error_mps"
                            ),
                            "release_landing_error_m": _metric(
                                info, "release_event_metrics", "landing_error_m"
                            ),
                            "release_flight_time_s": _metric(
                                info, "release_event_metrics", "flight_time_s"
                            ),
                            "minimum_contact_relative_speed_mps": info["minimum_contact_relative_speed_mps"],
                            "last_contact_velocity_alignment": info["last_contact_velocity_alignment"],
                            "last_contact_radial_offset_m": info["last_contact_radial_offset_m"],
                            "first_contact_relative_speed_mps": _metric(
                                info, "first_physical_contact_metrics", "relative_speed_mps"
                            ),
                            "first_contact_alignment": _metric(
                                info, "first_physical_contact_metrics", "velocity_alignment"
                            ),
                            "first_contact_radial_offset_m": _metric(
                                info, "first_physical_contact_metrics", "radial_offset_m"
                            ),
                            "first_contact_velocity_error_mps": _metric(
                                info, "first_physical_contact_metrics", "cup_velocity_error_mps"
                            ),
                            "first_contact_reference_planning_error_mps": _metric(
                                info, "first_physical_contact_metrics",
                                "reference_planning_velocity_error_mps"
                            ),
                            "first_contact_mppi_tracking_error_mps": _metric(
                                info, "first_physical_contact_metrics",
                                "mppi_tracking_velocity_error_mps"
                            ),
                            "best_pregrasp_relative_speed_mps": _metric(
                                info, "best_compatible_pregrasp_contact_metrics", "relative_speed_mps"
                            ),
                            "best_pregrasp_alignment": _metric(
                                info, "best_compatible_pregrasp_contact_metrics", "velocity_alignment"
                            ),
                            "best_pregrasp_radial_offset_m": _metric(
                                info, "best_compatible_pregrasp_contact_metrics", "radial_offset_m"
                            ),
                            "best_pregrasp_gate_score": _metric(
                                info, "best_compatible_pregrasp_contact_metrics", "gate_score"
                            ),
                            "grasp_event_relative_speed_mps": _metric(
                                info, "grasp_event_metrics", "relative_speed_mps"
                            ),
                            "grasp_event_alignment": _metric(
                                info, "grasp_event_metrics", "velocity_alignment"
                            ),
                            "grasp_event_radial_offset_m": _metric(
                                info, "grasp_event_metrics", "radial_offset_m"
                            ),
                            "grasp_event_axial_offset_m": _metric(
                                info, "grasp_event_metrics", "axial_offset_m"
                            ),
                            "grasp_event_gate_score": _metric(
                                info, "grasp_event_metrics", "gate_score"
                            ),
                            "grasp_event_velocity_error_mps": _metric(
                                info, "grasp_event_metrics", "cup_velocity_error_mps"
                            ),
                            "grasp_event_reference_planning_error_mps": _metric(
                                info, "grasp_event_metrics",
                                "reference_planning_velocity_error_mps"
                            ),
                            "grasp_event_mppi_tracking_error_mps": _metric(
                                info, "grasp_event_metrics",
                                "mppi_tracking_velocity_error_mps"
                            ),
                            "rendezvous_position_error_m": _metric(
                                info, "rendezvous_event_metrics", "position_error_m"
                            ),
                            "rendezvous_timing_error_s": _metric(
                                info, "rendezvous_event_metrics", "timing_error_s"
                            ),
                            "rendezvous_velocity_error_mps": _metric(
                                info, "rendezvous_event_metrics", "velocity_error_mps"
                            ),
                            "rendezvous_relative_speed_mps": _metric(
                                info, "rendezvous_event_metrics",
                                "ball_cup_relative_speed_mps"
                            ),
                            "rendezvous_alignment": _metric(
                                info, "rendezvous_event_metrics", "velocity_alignment"
                            ),
                            "precontact_entry_radial_m": info[
                                "precontact_entry_radial_m"
                            ],
                            "precontact_predicted_contact_radial_m": info[
                                "precontact_predicted_contact_radial_m"
                            ],
                            "contact_failure_radial": info["contact_failure_radial"],
                            "contact_failure_axial": info["contact_failure_axial"],
                            "contact_failure_speed": info["contact_failure_speed"],
                            "contact_failure_alignment": info["contact_failure_alignment"],
                            "unsafe_force": info["unsafe_force"],
                            "grasp_jump_m": info["grasp_jump_m"],
                            "minimum_capture_volume_error_m": info[
                                "minimum_capture_volume_error_m"
                            ],
                            "maximum_grasp_dwell_fraction": info[
                                "maximum_grasp_dwell_fraction"
                            ],
                            "peak_contact_force_n": env.peak_force_n,
                            "contact_impulse_ns": info["contact_impulse_ns"],
                            "mechanical_work_abs_j": info["mechanical_work_abs_j"],
                            "gripper_energy_j": info["gripper_energy_j"],
                            "energy_proxy_j": info["energy_proxy_j"]})
    finally:
        env.close()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(args.checkpoint), "episodes": args.episodes,
               "success_rate": float(np.mean([row["success"] for row in records])),
               "capture_rate": float(np.mean([row["captured"] for row in records])),
               "stable_capture_rate": float(np.mean([
                   row["stable_capture"] for row in records
               ])),
               "physical_contact_rate": float(np.mean([
                   row["physical_contact_occurred"] for row in records
               ])),
               "compatible_contact_rate": float(np.mean([
                   row["compatible_contact_occurred"] for row in records
               ])),
               "auto_capture_rate": float(np.mean([
                   row["auto_capture"] for row in records
               ])),
               "physical_capture_rate": float(np.mean([
                   row["physical_capture"] for row in records
               ])),
               "physical_contact_at_grasp_rate": float(np.mean([
                   row["physical_contact_at_grasp"] for row in records
               ])),
               "grasp_event_physical_compatible_rate": float(np.mean([
                   row["grasp_event_physical_compatible"] for row in records
               ])),
               "release_rate": float(np.mean([row["released"] for row in records])),
               "mean_return": float(np.mean([row["return"] for row in records])),
               "mean_target_distance_m": float(np.mean([row["target_distance_m"] for row in records])),
               "mean_release_velocity_error_mps": _mean(records, "release_velocity_error_mps"),
               "mean_release_landing_error_m": _mean(records, "release_landing_error_m"),
               "mean_release_flight_time_s": _mean(records, "release_flight_time_s"),
               "mean_first_contact_relative_speed_mps": _mean(records, "first_contact_relative_speed_mps"),
               "mean_first_contact_alignment": _mean(records, "first_contact_alignment"),
               "mean_first_contact_radial_offset_m": _mean(records, "first_contact_radial_offset_m"),
               "mean_first_contact_velocity_error_mps": _mean(records, "first_contact_velocity_error_mps"),
               "mean_first_contact_reference_planning_error_mps": _mean(
                   records, "first_contact_reference_planning_error_mps"
               ),
               "mean_first_contact_mppi_tracking_error_mps": _mean(
                   records, "first_contact_mppi_tracking_error_mps"
               ),
               "mean_best_pregrasp_relative_speed_mps": _mean(records, "best_pregrasp_relative_speed_mps"),
               "mean_best_pregrasp_alignment": _mean(records, "best_pregrasp_alignment"),
               "mean_best_pregrasp_radial_offset_m": _mean(records, "best_pregrasp_radial_offset_m"),
               "mean_best_pregrasp_gate_score": _mean(records, "best_pregrasp_gate_score"),
               "mean_grasp_event_relative_speed_mps": _mean(records, "grasp_event_relative_speed_mps"),
               "mean_grasp_event_alignment": _mean(records, "grasp_event_alignment"),
               "mean_grasp_event_radial_offset_m": _mean(records, "grasp_event_radial_offset_m"),
               "mean_grasp_event_axial_offset_m": _mean(records, "grasp_event_axial_offset_m"),
               "mean_grasp_event_gate_score": _mean(records, "grasp_event_gate_score"),
               "mean_grasp_event_velocity_error_mps": _mean(records, "grasp_event_velocity_error_mps"),
               "mean_grasp_event_reference_planning_error_mps": _mean(
                   records, "grasp_event_reference_planning_error_mps"
               ),
               "mean_grasp_event_mppi_tracking_error_mps": _mean(
                   records, "grasp_event_mppi_tracking_error_mps"
               ),
               "mean_rendezvous_position_error_m": _mean(
                   records, "rendezvous_position_error_m"
               ),
               "mean_rendezvous_timing_error_s": _mean(
                   records, "rendezvous_timing_error_s"
               ),
               "mean_rendezvous_velocity_error_mps": _mean(
                   records, "rendezvous_velocity_error_mps"
               ),
               "mean_rendezvous_relative_speed_mps": _mean(
                   records, "rendezvous_relative_speed_mps"
               ),
               "mean_rendezvous_alignment": _mean(records, "rendezvous_alignment"),
               "mean_minimum_capture_volume_error_m": _mean(
                   records, "minimum_capture_volume_error_m"
               ),
               "mean_maximum_grasp_dwell_fraction": _mean(
                   records, "maximum_grasp_dwell_fraction"
               ),
               "unsafe_force_rate": float(np.mean([row["unsafe_force"] for row in records])),
               "curriculum_stage": args.curriculum_stage,
               "stage2_level": args.stage2_level,
               "grasp_mode": grasp_mode,
               "auto_grasp_radius_m": env._auto_grasp_radial_limit(),
               "contact_failure_counts": {
                   name: int(sum(row[f"contact_failure_{name}"] for row in records))
                   for name in ("radial", "axial", "speed", "alignment")
               },
               "physical_contact": True,
               "grasp_model": (
                   "capture-volume auto attach, then runtime rigid weld"
                   if grasp_mode == "auto_volume"
                   else "physical contact plus pose/velocity dwell, then runtime rigid weld"
               ),
               "energy_metric": "absolute muscle boundary-work plus assumed finger open/close energy; not electrical supply energy",
               "records": records}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
