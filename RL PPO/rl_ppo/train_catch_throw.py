"""Train hierarchical PPO for three-finger physical catch-and-throw."""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .catch_throw_env import CatchThrowEnv, CatchThrowEnvConfig
from .catch_throw_task import DEFAULT_TARGET_OFFSET_WORLD_M
from .networks import ActorCritic
from .ppo import PPOConfig, RolloutBuffer, update


CHECKPOINT_SCHEMA = "umarm_three_finger_far_horizontal_catch_throw_hierarchical_ppo_v11"
COMPATIBLE_CHECKPOINT_SCHEMAS = {
    CHECKPOINT_SCHEMA,
    "umarm_three_finger_far_horizontal_catch_throw_hierarchical_ppo_v10",
}


def save_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer,
                    *, update_index: int, config: dict, curriculum_stage: int,
                    stage2_level: int, stage_updates: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "observation_size": CatchThrowEnv.observation_size,
        "action_size": CatchThrowEnv.action_size, "update": update_index,
        "curriculum_stage": curriculum_stage, "stage2_level": stage2_level,
        "stage_updates": stage_updates, "config": config,
    }, path)


def _mean(rows: deque[dict] | list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _metric(info: dict, record_key: str, metric_key: str) -> float | None:
    record = info.get(record_key)
    if not record:
        return None
    value = record.get(metric_key)
    return None if value is None else float(value)


def _quality_gate_passed(evaluation: dict, *, success_threshold: float) -> bool:
    speed = evaluation.get("mean_grasp_event_relative_speed_mps")
    alignment = evaluation.get("mean_grasp_event_alignment")
    radial = evaluation.get("mean_grasp_event_radial_offset_m")
    unsafe_rate = evaluation.get("unsafe_force_rate")
    return bool(
        evaluation["capture_rate"] >= success_threshold
        and speed is not None and speed <= .50
        and alignment is not None and alignment >= .65
        and radial is not None and radial <= .025
        and unsafe_rate == 0.0
    )


def evaluation_score(evaluation: dict) -> tuple[float, ...]:
    """Lexicographic rank for ppo_best.pt deterministic checkpoints."""
    def low_metric(key: str) -> float:
        value = evaluation.get(key)
        return -float(value) if value is not None else -1e9

    stage = float(evaluation.get("stage", 0.0))
    stage2_level = float(evaluation.get("stage2_level", 0.0))
    raw_capture_rate = float(evaluation.get("capture_rate", 0.0))
    capture_rate = float(evaluation.get("stable_capture_rate", raw_capture_rate))
    release_rate = float(evaluation.get("release_rate", 0.0))
    success_rate = float(evaluation.get("success_rate", 0.0))
    physical_quality_rate = float(evaluation.get(
        "grasp_event_physical_compatible_rate", 0.0
    ))
    speed = evaluation.get("mean_grasp_event_relative_speed_mps")
    alignment = evaluation.get("mean_grasp_event_alignment")
    radial = evaluation.get("mean_grasp_event_radial_offset_m")
    unsafe_rate = float(evaluation.get("unsafe_force_rate", 1.0))
    quality_ready = float(capture_rate >= .80 and unsafe_rate == 0.0)
    speed_deficit = max(0.0, float(speed) - .50) if speed is not None else 1e6
    alignment_deficit = max(0.0, .65 - float(alignment)) if alignment is not None else 1e6
    radial_deficit = max(0.0, float(radial) - .025) if radial is not None else 1e6

    return (
        stage,
        stage2_level,
        success_rate if stage >= 4.0 else release_rate if stage >= 3.0 else capture_rate,
        raw_capture_rate,
        release_rate,
        success_rate,
        -float(evaluation.get("mean_release_landing_error_m") or 1e6),
        -float(evaluation.get("mean_release_velocity_error_mps") or 1e6),
        low_metric("mean_minimum_capture_volume_error_m"),
        float(evaluation.get("mean_maximum_grasp_dwell_fraction") or 0.0),
        physical_quality_rate,
        quality_ready,
        low_metric("mean_grasp_event_gate_score"),
        -alignment_deficit,
        -speed_deficit,
        -radial_deficit,
        low_metric("mean_grasp_event_relative_speed_mps"),
        float(evaluation.get("mean_grasp_event_alignment") or -1e9),
        low_metric("mean_grasp_event_radial_offset_m"),
        low_metric("mean_rendezvous_relative_speed_mps"),
        float(evaluation.get("mean_rendezvous_alignment") or -1e9),
        low_metric("mean_rendezvous_position_error_m"),
        low_metric("mean_rendezvous_velocity_error_mps"),
        -unsafe_rate,
        low_metric("mean_contact_impulse_ns"),
        low_metric("mean_energy_proxy_j"),
    )


def difficulty_label(stage: int, stage2_level: int) -> str:
    return f"2{chr(64 + stage2_level)}" if stage == 2 else str(stage)


def curriculum_transition(stage: int, stage2_level: int, stage_updates: int,
                          evaluation: dict | None, *, minimum_updates: int,
                          success_threshold: float,
                          grasp_mode: str = "physical_dwell") -> tuple[int, int]:
    """Return the next curriculum, evaluated only at PPO update boundaries."""
    if evaluation is None or stage_updates < minimum_updates or stage >= 5:
        return stage, stage2_level
    stable_capture_rate = float(evaluation.get(
        "stable_capture_rate", evaluation.get("success_rate", evaluation["capture_rate"])
    ))
    unsafe_rate = float(evaluation.get("unsafe_force_rate", 1.0))
    if grasp_mode == "auto_volume" and stage <= 2:
        passed = stable_capture_rate >= success_threshold and unsafe_rate == 0.0
        if not passed:
            return stage, stage2_level
        if stage == 2 and stage2_level < 3:
            return stage, stage2_level + 1
        return stage + 1, 1 if stage == 1 else stage2_level
    if stage == 1:
        first_speed = evaluation.get("mean_first_contact_relative_speed_mps")
        grasp_speed = evaluation.get("mean_grasp_event_relative_speed_mps")
        passed = (
            stable_capture_rate >= success_threshold
            and first_speed is not None and float(first_speed) <= 2.15
            and grasp_speed is not None and float(grasp_speed) <= .65
            and unsafe_rate == 0.0
        )
    elif stage == 2:
        speed = evaluation.get("mean_grasp_event_relative_speed_mps")
        alignment = evaluation.get("mean_grasp_event_alignment")
        radial = evaluation.get("mean_grasp_event_radial_offset_m")
        speed_limit = (.60, .58, .55)[stage2_level - 1]
        alignment_limit = (.58, .63, .66)[stage2_level - 1]
        radial_limit = (.027, .026, .025)[stage2_level - 1]
        passed = (
            stable_capture_rate >= success_threshold
            and speed is not None and float(speed) <= speed_limit
            and alignment is not None and float(alignment) >= alignment_limit
            and radial is not None and float(radial) <= radial_limit
            and unsafe_rate == 0.0
        )
    elif stage == 3:
        passed = evaluation["release_rate"] >= success_threshold
    else:
        passed = evaluation["success_rate"] >= success_threshold
    if not passed:
        return stage, stage2_level
    if stage == 2 and stage2_level < 3:
        return stage, stage2_level + 1
    return stage + 1, 1 if stage == 1 else stage2_level


def _environment_config(args: argparse.Namespace, *, seed: int, stage: int,
                        stage2_level: int) -> CatchThrowEnvConfig:
    return CatchThrowEnvConfig(
        seed=seed, controller=args.controller,
        mppi_samples=args.mppi_samples, mppi_horizon=args.mppi_horizon,
        mppi_tip_weight=getattr(args, "mppi_tip_weight", 1.0),
        mppi_tip_velocity_weight=getattr(args, "mppi_tip_velocity_weight", .25),
        mppi_joint_velocity_weight=getattr(args, "mppi_joint_velocity_weight", .05),
        mppi_compliance_weight=getattr(args, "mppi_compliance_weight", 1.0),
        mppi_correction_blend=getattr(args, "mppi_correction_blend", .20),
        target_offset_world_m=tuple(args.target_offset), target_radius_m=args.target_radius,
        capture_hold_s=getattr(args, "capture_hold_s", .10),
        grasp_mode=getattr(args, "grasp_mode", "auto_volume"),
        auto_grasp_radius_m=getattr(args, "auto_grasp_radius", None),
        curriculum_stage=stage, stage2_level=stage2_level,
    )


def evaluate_policy(model: ActorCritic, args: argparse.Namespace, device: torch.device,
                    *, stage: int, stage2_level: int) -> dict[str, float | int | None]:
    """Run a repeatable held-out evaluation without perturbing the training env."""
    evaluation_seed = args.seed + 1_000_000 + stage * 10_000 + stage2_level * 1_000
    env = CatchThrowEnv(_environment_config(
        args, seed=evaluation_seed, stage=stage, stage2_level=stage2_level
    ))
    records: list[dict] = []
    try:
        for episode in range(args.curriculum_eval_episodes):
            observation, _ = env.reset(evaluation_seed + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                tensor = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
                mask = torch.as_tensor(env.action_mask(), dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, _, _, _ = model.act(
                        tensor, deterministic=True, action_mask=mask
                    )
                observation, _, terminated, truncated, info = env.step(
                    action[0].cpu().numpy()
                )
            first_contact = info["first_physical_contact_metrics"] or {}
            best_pregrasp = info["best_compatible_pregrasp_contact_metrics"] or {}
            grasp_event = info["grasp_event_metrics"] or {}
            rendezvous_event = info["rendezvous_event_metrics"] or {}
            release_event = info["release_event_metrics"] or {}
            records.append({
                "success": int(terminated),
                "physical_contact": int(info["physical_contact_occurred"]),
                "captured": int(info["captured"]),
                "stable_capture": int(info["stable_capture"]),
                "auto_capture": int(info["auto_capture_occurred"]),
                "physical_capture": int(info["physical_capture_occurred"]),
                "physical_contact_at_grasp": int(info["physical_contact_at_grasp"]),
                "grasp_event_physical_compatible": int(
                    info["physical_compatible_at_grasp"]
                ),
                "released": int(info["released"]),
                "capture_radial_offset_m": (
                    grasp_event.get("radial_offset_m") if info["captured"] else None
                ),
                "contact_radial_offset_m": info["last_contact_radial_offset_m"],
                "contact_relative_speed_mps": info["minimum_contact_relative_speed_mps"],
                "contact_alignment": info["last_contact_velocity_alignment"],
                "first_contact_relative_speed_mps": first_contact.get("relative_speed_mps"),
                "first_contact_alignment": first_contact.get("velocity_alignment"),
                "first_contact_radial_offset_m": first_contact.get("radial_offset_m"),
                "first_contact_velocity_error_mps": first_contact.get("cup_velocity_error_mps"),
                "first_contact_reference_planning_error_mps": first_contact.get(
                    "reference_planning_velocity_error_mps"
                ),
                "first_contact_mppi_tracking_error_mps": first_contact.get(
                    "mppi_tracking_velocity_error_mps"
                ),
                "best_pregrasp_relative_speed_mps": best_pregrasp.get("relative_speed_mps"),
                "best_pregrasp_alignment": best_pregrasp.get("velocity_alignment"),
                "best_pregrasp_radial_offset_m": best_pregrasp.get("radial_offset_m"),
                "best_pregrasp_gate_score": best_pregrasp.get("gate_score"),
                "grasp_event_relative_speed_mps": grasp_event.get("relative_speed_mps"),
                "grasp_event_alignment": grasp_event.get("velocity_alignment"),
                "grasp_event_radial_offset_m": grasp_event.get("radial_offset_m"),
                "grasp_event_axial_offset_m": grasp_event.get("axial_offset_m"),
                "grasp_event_gate_score": grasp_event.get("gate_score"),
                "grasp_event_cup_displacement_m": grasp_event.get(
                    "cup_displacement_from_home_m"
                ),
                "grasp_event_velocity_error_mps": grasp_event.get("cup_velocity_error_mps"),
                "grasp_event_reference_planning_error_mps": grasp_event.get(
                    "reference_planning_velocity_error_mps"
                ),
                "grasp_event_mppi_tracking_error_mps": grasp_event.get(
                    "mppi_tracking_velocity_error_mps"
                ),
                "rendezvous_position_error_m": rendezvous_event.get(
                    "position_error_m"
                ),
                "rendezvous_timing_error_s": rendezvous_event.get("timing_error_s"),
                "rendezvous_velocity_error_mps": rendezvous_event.get(
                    "velocity_error_mps"
                ),
                "rendezvous_relative_speed_mps": rendezvous_event.get(
                    "ball_cup_relative_speed_mps"
                ),
                "rendezvous_alignment": rendezvous_event.get(
                    "velocity_alignment"
                ),
                "predicted_entry_radial_m": info["precontact_entry_radial_m"],
                "predicted_contact_radial_m": info[
                    "precontact_predicted_contact_radial_m"
                ],
                "minimum_capture_volume_error_m": info[
                    "minimum_capture_volume_error_m"
                ],
                "maximum_grasp_dwell_fraction": info[
                    "maximum_grasp_dwell_fraction"
                ],
                "maximum_cup_displacement_m": info["maximum_cup_displacement_m"],
                "failure_radial": int(info["contact_failure_radial"]),
                "failure_axial": int(info["contact_failure_axial"]),
                "failure_speed": int(info["contact_failure_speed"]),
                "failure_alignment": int(info["contact_failure_alignment"]),
                "unsafe_force": int(info["unsafe_force"]),
                "contact_impulse_ns": info["contact_impulse_ns"],
                "energy_proxy_j": info["energy_proxy_j"],
                "reference_velocity_saturation_fraction": info[
                    "reference_velocity_saturation_fraction"
                ],
                "release_velocity_error_mps": release_event.get("velocity_error_mps"),
                "release_landing_error_m": release_event.get("landing_error_m"),
                "release_flight_time_s": release_event.get("flight_time_s"),
            })
    finally:
        env.close()
    return {
        "stage": stage,
        "stage2_level": stage2_level,
        "grasp_mode": getattr(args, "grasp_mode", "auto_volume"),
        "episodes": len(records),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "physical_contact_rate": float(np.mean([row["physical_contact"] for row in records])),
        "capture_rate": float(np.mean([row["captured"] for row in records])),
        "stable_capture_rate": float(np.mean([row["stable_capture"] for row in records])),
        "auto_capture_rate": float(np.mean([row["auto_capture"] for row in records])),
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
        "mean_capture_radial_offset_m": _mean(records, "capture_radial_offset_m"),
        "mean_contact_radial_offset_m": _mean(records, "contact_radial_offset_m"),
        "mean_contact_relative_speed_mps": _mean(records, "contact_relative_speed_mps"),
        "mean_contact_alignment": _mean(records, "contact_alignment"),
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
        "mean_grasp_event_axial_offset_m": _mean(
            records, "grasp_event_axial_offset_m"
        ),
        "mean_grasp_event_gate_score": _mean(records, "grasp_event_gate_score"),
        "mean_grasp_event_cup_displacement_m": _mean(
            records, "grasp_event_cup_displacement_m"
        ),
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
        "mean_predicted_entry_radial_m": _mean(records, "predicted_entry_radial_m"),
        "mean_predicted_contact_radial_m": _mean(
            records, "predicted_contact_radial_m"
        ),
        "mean_minimum_capture_volume_error_m": _mean(
            records, "minimum_capture_volume_error_m"
        ),
        "mean_maximum_grasp_dwell_fraction": _mean(
            records, "maximum_grasp_dwell_fraction"
        ),
        "mean_maximum_cup_displacement_m": _mean(
            records, "maximum_cup_displacement_m"
        ),
        "unsafe_force_rate": float(np.mean([row["unsafe_force"] for row in records])),
        "mean_contact_impulse_ns": _mean(records, "contact_impulse_ns"),
        "mean_energy_proxy_j": _mean(records, "energy_proxy_j"),
        "mean_reference_velocity_saturation_fraction": _mean(
            records, "reference_velocity_saturation_fraction"
        ),
        "mean_release_velocity_error_mps": _mean(records, "release_velocity_error_mps"),
        "mean_release_landing_error_m": _mean(records, "release_landing_error_m"),
        "mean_release_flight_time_s": _mean(records, "release_flight_time_s"),
        "radial_failure_count": int(sum(row["failure_radial"] for row in records)),
        "axial_failure_count": int(sum(row["failure_axial"] for row in records)),
        "speed_failure_count": int(sum(row["failure_speed"] for row in records)),
        "alignment_failure_count": int(
            sum(row["failure_alignment"] for row in records)
        ),
        "radial_failure_rate": float(np.mean([row["failure_radial"] for row in records])),
        "axial_failure_rate": float(np.mean([row["failure_axial"] for row in records])),
        "speed_failure_rate": float(np.mean([row["failure_speed"] for row in records])),
        "alignment_failure_rate": float(
            np.mean([row["failure_alignment"] for row in records])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=100_000,
                        help="PPO decisions, each containing five 150 Hz MPPI steps")
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/catch_throw_v11_auto_grasp"))
    parser.add_argument("--controller", choices=("koopman_mppi", "ff_pid"), default="koopman_mppi")
    parser.add_argument("--mppi-samples", type=int, default=16)
    parser.add_argument("--mppi-horizon", type=int, default=8)
    parser.add_argument("--mppi-tip-weight", type=float, default=1.0)
    parser.add_argument("--mppi-tip-velocity-weight", type=float, default=.25)
    parser.add_argument("--mppi-joint-velocity-weight", type=float, default=.05)
    parser.add_argument("--mppi-compliance-weight", type=float, default=1.0)
    parser.add_argument("--mppi-correction-blend", type=float, default=.20)
    parser.add_argument("--capture-hold-s", type=float, default=.10)
    parser.add_argument(
        "--grasp-mode", choices=("auto_volume", "physical_dwell"),
        default="auto_volume",
        help="auto-attach inside the capture volume or require physical dwell",
    )
    parser.add_argument(
        "--auto-grasp-radius", type=float,
        help="override the curriculum capture-volume radius in metres",
    )
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=DEFAULT_TARGET_OFFSET_WORLD_M)
    parser.add_argument("--target-radius", type=float, default=.060)
    parser.add_argument("--start-stage", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--start-stage2-level", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--fixed-stage", action="store_true",
                        help="disable deterministic-evaluation curriculum advancement")
    parser.add_argument("--stage-min-updates", type=int, default=20)
    parser.add_argument("--stage-success-threshold", type=float, default=.80)
    parser.add_argument("--curriculum-eval-every", type=int, default=10)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="initialize model weights from a compatible V9/V10/V11 checkpoint")
    args = parser.parse_args()
    if args.total_steps <= 1 or args.rollout_steps <= 1:
        parser.error("total and rollout steps must exceed one")
    if args.stage_min_updates < 1 or not 0 < args.stage_success_threshold <= 1:
        parser.error("invalid curriculum advancement settings")
    if args.curriculum_eval_every < 1 or args.curriculum_eval_episodes < 1:
        parser.error("invalid deterministic evaluation settings")
    if min(args.mppi_tip_weight, args.mppi_tip_velocity_weight,
           args.mppi_joint_velocity_weight, args.mppi_compliance_weight) < 0:
        parser.error("MPPI weights must be non-negative")
    if not 0 <= args.mppi_correction_blend <= 1:
        parser.error("MPPI correction blend must be in [0, 1]")
    if args.capture_hold_s <= 0:
        parser.error("capture hold must be positive")
    if args.auto_grasp_radius is not None and args.auto_grasp_radius <= 0:
        parser.error("auto grasp radius must be positive")
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        parser.error(f"initial checkpoint does not exist: {args.init_checkpoint}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    env = CatchThrowEnv(_environment_config(
        args, seed=args.seed, stage=args.start_stage,
        stage2_level=args.start_stage2_level,
    ))
    observation, _ = env.reset(args.seed)
    model = ActorCritic(env.observation_size, env.action_size).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if (int(payload.get("observation_size", -1)) != env.observation_size
                or int(payload.get("action_size", -1)) != env.action_size):
            raise ValueError("initial checkpoint dimensions do not match the V11 environment")
        model.load_state_dict(payload["model"])
        print(
            f"Initialized policy weights from {args.init_checkpoint} "
            f"(update {payload.get('update', 'unknown')}).",
            flush=True,
        )
    ppo_config = PPOConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
    run_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    run_config["ppo"] = vars(ppo_config)
    history: list[dict] = []
    recent_episodes: deque[dict] = deque(maxlen=50)
    episode_return = 0.0
    episode_length = 0
    episodes = curriculum_successes = final_task_successes = 0
    stage_updates = 0
    last_evaluation: dict | None = None
    last_evaluation_update: int | None = None
    best_evaluation_score: tuple[float, ...] | None = None
    best_evaluation_update: int | None = None
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    save_checkpoint(
        args.out_dir / "ppo_initial.pt", model, optimizer,
        update_index=0, config=run_config,
        curriculum_stage=env.curriculum_stage,
        stage2_level=env.stage2_level, stage_updates=0,
    )
    baseline_evaluation = evaluate_policy(
        model, args, device, stage=env.curriculum_stage,
        stage2_level=env.stage2_level,
    )
    last_evaluation = baseline_evaluation
    last_evaluation_update = 0
    best_evaluation_score = evaluation_score(baseline_evaluation)
    best_evaluation_update = 0
    save_checkpoint(
        args.out_dir / "ppo_best.pt", model, optimizer,
        update_index=0, config=run_config,
        curriculum_stage=env.curriculum_stage,
        stage2_level=env.stage2_level, stage_updates=0,
    )
    print(json.dumps({
        "baseline_evaluation": baseline_evaluation,
        "best_evaluation_update": best_evaluation_update,
    }), flush=True)
    interrupted = False

    try:
        for update_index, start in enumerate(
                range(0, args.total_steps, args.rollout_steps), start=1):
            rollout_stage = env.curriculum_stage
            rollout_stage2_level = env.stage2_level
            steps = min(args.rollout_steps, args.total_steps - start)
            buffer = RolloutBuffer(steps, env.observation_size, env.action_size)
            for _ in range(steps):
                tensor = torch.as_tensor(observation[None], dtype=torch.float32, device=device)
                action_mask = env.action_mask()
                mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, raw_action, log_prob, value = model.act(
                        tensor, action_mask=mask_tensor
                    )
                next_observation, reward, terminated, truncated, info = env.step(
                    action[0].cpu().numpy()
                )
                done = terminated or truncated
                buffer.add(
                    observation, raw_action[0].cpu().numpy(), float(log_prob.item()),
                    reward, float(done), float(value.item()), action_mask,
                )
                observation = next_observation
                episode_return += reward
                episode_length += 1
                if done:
                    episodes += 1
                    curriculum_successes += int(terminated)
                    final_success = bool(
                        info["released"]
                        and info["target_distance_m"] <= info["target_radius_m"]
                    )
                    final_task_successes += int(final_success)
                    first_contact = info["first_physical_contact_metrics"] or {}
                    best_pregrasp = info["best_compatible_pregrasp_contact_metrics"] or {}
                    grasp_event_metrics = info["grasp_event_metrics"] or {}
                    rendezvous_event_metrics = info["rendezvous_event_metrics"] or {}
                    release_event_metrics = info["release_event_metrics"] or {}
                    recent_episodes.append({
                        "return": episode_return, "length": episode_length,
                        "curriculum_success": int(terminated),
                        "final_success": int(final_success),
                        "physical_contact": int(info["physical_contact_occurred"]),
                        "compatible_contact": int(info["compatible_contact_occurred"]),
                        "captured": int(info["captured"]),
                        "stable_capture": int(info["stable_capture"]),
                        "auto_capture": int(info["auto_capture_occurred"]),
                        "physical_capture": int(info["physical_capture_occurred"]),
                        "physical_contact_at_grasp": int(
                            info["physical_contact_at_grasp"]
                        ),
                        "grasp_event_physical_compatible": int(
                            info["physical_compatible_at_grasp"]
                        ),
                        "released": int(info["released"]),
                        "peak_force_n": env.peak_force_n,
                        "impulse_ns": info["contact_impulse_ns"],
                        "contact_relative_speed_mps": info["minimum_contact_relative_speed_mps"],
                        "contact_alignment": info["last_contact_velocity_alignment"],
                        "contact_radial_offset_m": info["last_contact_radial_offset_m"],
                        "first_contact_relative_speed_mps": first_contact.get("relative_speed_mps"),
                        "first_contact_alignment": first_contact.get("velocity_alignment"),
                        "first_contact_radial_offset_m": first_contact.get("radial_offset_m"),
                        "first_contact_velocity_error_mps": first_contact.get("cup_velocity_error_mps"),
                        "first_contact_reference_planning_error_mps": first_contact.get(
                            "reference_planning_velocity_error_mps"
                        ),
                        "first_contact_mppi_tracking_error_mps": first_contact.get(
                            "mppi_tracking_velocity_error_mps"
                        ),
                        "best_pregrasp_relative_speed_mps": best_pregrasp.get("relative_speed_mps"),
                        "best_pregrasp_alignment": best_pregrasp.get("velocity_alignment"),
                        "best_pregrasp_radial_offset_m": best_pregrasp.get("radial_offset_m"),
                        "best_pregrasp_gate_score": best_pregrasp.get("gate_score"),
                        "grasp_event_relative_speed_mps": grasp_event_metrics.get("relative_speed_mps"),
                        "grasp_event_alignment": grasp_event_metrics.get("velocity_alignment"),
                        "grasp_event_radial_offset_m": grasp_event_metrics.get("radial_offset_m"),
                        "grasp_event_axial_offset_m": grasp_event_metrics.get("axial_offset_m"),
                        "grasp_event_gate_score": grasp_event_metrics.get("gate_score"),
                        "grasp_event_cup_displacement_m": grasp_event_metrics.get(
                            "cup_displacement_from_home_m"
                        ),
                        "grasp_event_velocity_error_mps": grasp_event_metrics.get("cup_velocity_error_mps"),
                        "grasp_event_reference_planning_error_mps": grasp_event_metrics.get(
                            "reference_planning_velocity_error_mps"
                        ),
                        "grasp_event_mppi_tracking_error_mps": grasp_event_metrics.get(
                            "mppi_tracking_velocity_error_mps"
                        ),
                        "rendezvous_position_error_m": rendezvous_event_metrics.get(
                            "position_error_m"
                        ),
                        "rendezvous_timing_error_s": rendezvous_event_metrics.get(
                            "timing_error_s"
                        ),
                        "rendezvous_velocity_error_mps": rendezvous_event_metrics.get(
                            "velocity_error_mps"
                        ),
                        "rendezvous_relative_speed_mps": rendezvous_event_metrics.get(
                            "ball_cup_relative_speed_mps"
                        ),
                        "rendezvous_alignment": rendezvous_event_metrics.get(
                            "velocity_alignment"
                        ),
                        "predicted_entry_radial_m": info[
                            "precontact_entry_radial_m"
                        ],
                        "predicted_contact_radial_m": info[
                            "precontact_predicted_contact_radial_m"
                        ],
                        "minimum_capture_volume_error_m": info[
                            "minimum_capture_volume_error_m"
                        ],
                        "maximum_grasp_dwell_fraction": info[
                            "maximum_grasp_dwell_fraction"
                        ],
                        "maximum_cup_displacement_m": info[
                            "maximum_cup_displacement_m"
                        ],
                        "failure_radial": int(info["contact_failure_radial"]),
                        "failure_axial": int(info["contact_failure_axial"]),
                        "failure_speed": int(info["contact_failure_speed"]),
                        "failure_alignment": int(info["contact_failure_alignment"]),
                        "unsafe_force": int(info["unsafe_force"]),
                        "reference_velocity_saturation_fraction": info[
                            "reference_velocity_saturation_fraction"
                        ],
                        "grasp_jump_m": info["grasp_jump_m"],
                        "energy_proxy_j": info["energy_proxy_j"],
                        "landing_error_m": info["predicted_landing_error_m"],
                        "release_velocity_error_mps": release_event_metrics.get(
                            "velocity_error_mps"
                        ),
                        "release_landing_error_m": release_event_metrics.get(
                            "landing_error_m"
                        ),
                        "release_flight_time_s": release_event_metrics.get(
                            "flight_time_s"
                        ),
                    })
                    observation, _ = env.reset(args.seed + episodes)
                    episode_return, episode_length = 0.0, 0

            with torch.no_grad():
                last_value = float(model.critic(
                    torch.as_tensor(observation[None], dtype=torch.float32, device=device)
                ).item())
            buffer.finish(last_value, ppo_config)
            metrics = update(model, optimizer, buffer, ppo_config, device)
            stage_updates += 1

            evaluation = None
            if stage_updates % args.curriculum_eval_every == 0:
                evaluation = evaluate_policy(
                    model, args, device, stage=rollout_stage,
                    stage2_level=rollout_stage2_level,
                )
                last_evaluation = evaluation
                last_evaluation_update = update_index
                score = evaluation_score(evaluation)
                if best_evaluation_score is None or score > best_evaluation_score:
                    best_evaluation_score = score
                    best_evaluation_update = update_index
                    save_checkpoint(
                        args.out_dir / "ppo_best.pt", model, optimizer,
                        update_index=update_index, config=run_config,
                        curriculum_stage=env.curriculum_stage,
                        stage2_level=env.stage2_level, stage_updates=stage_updates,
                    )
            next_stage, next_level = rollout_stage, rollout_stage2_level
            if not args.fixed_stage:
                next_stage, next_level = curriculum_transition(
                    rollout_stage, rollout_stage2_level, stage_updates, evaluation,
                    minimum_updates=args.stage_min_updates,
                    success_threshold=args.stage_success_threshold,
                    grasp_mode=args.grasp_mode,
                )
            promoted = (next_stage, next_level) != (rollout_stage, rollout_stage2_level)
            evaluation_fields = {
                f"deterministic_eval_{key}": value
                for key, value in (last_evaluation or {}).items()
            }
            row = {
                "update": update_index, "steps": start + steps, "episodes": episodes,
                "curriculum_stage": rollout_stage,
                "stage2_level": rollout_stage2_level,
                "difficulty": difficulty_label(rollout_stage, rollout_stage2_level),
                "grasp_mode": args.grasp_mode,
                "auto_grasp_radius_m": env._auto_grasp_radial_limit(),
                "stage_updates": stage_updates,
                "promoted": promoted,
                "next_curriculum_stage": next_stage,
                "next_stage2_level": next_level,
                "last_evaluation_update": last_evaluation_update,
                "best_evaluation_update": best_evaluation_update,
                "curriculum_success_rate": curriculum_successes / max(episodes, 1),
                "final_task_success_rate": final_task_successes / max(episodes, 1),
                "recent_success_rate": _mean(recent_episodes, "curriculum_success"),
                "recent_physical_contact_rate": _mean(recent_episodes, "physical_contact"),
                "recent_compatible_contact_rate": _mean(recent_episodes, "compatible_contact"),
                "recent_capture_rate": _mean(recent_episodes, "captured"),
                "recent_stable_capture_rate": _mean(recent_episodes, "stable_capture"),
                "recent_auto_capture_rate": _mean(recent_episodes, "auto_capture"),
                "recent_physical_capture_rate": _mean(
                    recent_episodes, "physical_capture"
                ),
                "recent_physical_contact_at_grasp_rate": _mean(
                    recent_episodes, "physical_contact_at_grasp"
                ),
                "recent_grasp_event_physical_compatible_rate": _mean(
                    recent_episodes, "grasp_event_physical_compatible"
                ),
                "recent_release_rate": _mean(recent_episodes, "released"),
                "mean_completed_return": _mean(recent_episodes, "return"),
                "mean_completed_length": _mean(recent_episodes, "length"),
                "mean_peak_force_n": _mean(recent_episodes, "peak_force_n"),
                "mean_contact_impulse_ns": _mean(recent_episodes, "impulse_ns"),
                "mean_contact_relative_speed_mps": _mean(
                    recent_episodes, "contact_relative_speed_mps"
                ),
                "mean_contact_alignment": _mean(recent_episodes, "contact_alignment"),
                "mean_contact_radial_offset_m": _mean(
                    recent_episodes, "contact_radial_offset_m"
                ),
                "mean_first_contact_relative_speed_mps": _mean(
                    recent_episodes, "first_contact_relative_speed_mps"
                ),
                "mean_first_contact_alignment": _mean(
                    recent_episodes, "first_contact_alignment"
                ),
                "mean_first_contact_radial_offset_m": _mean(
                    recent_episodes, "first_contact_radial_offset_m"
                ),
                "mean_first_contact_velocity_error_mps": _mean(
                    recent_episodes, "first_contact_velocity_error_mps"
                ),
                "mean_first_contact_reference_planning_error_mps": _mean(
                    recent_episodes, "first_contact_reference_planning_error_mps"
                ),
                "mean_first_contact_mppi_tracking_error_mps": _mean(
                    recent_episodes, "first_contact_mppi_tracking_error_mps"
                ),
                "mean_best_pregrasp_relative_speed_mps": _mean(
                    recent_episodes, "best_pregrasp_relative_speed_mps"
                ),
                "mean_best_pregrasp_alignment": _mean(
                    recent_episodes, "best_pregrasp_alignment"
                ),
                "mean_best_pregrasp_radial_offset_m": _mean(
                    recent_episodes, "best_pregrasp_radial_offset_m"
                ),
                "mean_best_pregrasp_gate_score": _mean(
                    recent_episodes, "best_pregrasp_gate_score"
                ),
                "mean_grasp_event_relative_speed_mps": _mean(
                    recent_episodes, "grasp_event_relative_speed_mps"
                ),
                "mean_grasp_event_alignment": _mean(
                    recent_episodes, "grasp_event_alignment"
                ),
                "mean_grasp_event_radial_offset_m": _mean(
                    recent_episodes, "grasp_event_radial_offset_m"
                ),
                "mean_grasp_event_axial_offset_m": _mean(
                    recent_episodes, "grasp_event_axial_offset_m"
                ),
                "mean_grasp_event_gate_score": _mean(
                    recent_episodes, "grasp_event_gate_score"
                ),
                "mean_grasp_event_cup_displacement_m": _mean(
                    recent_episodes, "grasp_event_cup_displacement_m"
                ),
                "mean_grasp_event_velocity_error_mps": _mean(
                    recent_episodes, "grasp_event_velocity_error_mps"
                ),
                "mean_grasp_event_reference_planning_error_mps": _mean(
                    recent_episodes, "grasp_event_reference_planning_error_mps"
                ),
                "mean_grasp_event_mppi_tracking_error_mps": _mean(
                    recent_episodes, "grasp_event_mppi_tracking_error_mps"
                ),
                "mean_rendezvous_position_error_m": _mean(
                    recent_episodes, "rendezvous_position_error_m"
                ),
                "mean_rendezvous_timing_error_s": _mean(
                    recent_episodes, "rendezvous_timing_error_s"
                ),
                "mean_rendezvous_velocity_error_mps": _mean(
                    recent_episodes, "rendezvous_velocity_error_mps"
                ),
                "mean_rendezvous_relative_speed_mps": _mean(
                    recent_episodes, "rendezvous_relative_speed_mps"
                ),
                "mean_rendezvous_alignment": _mean(
                    recent_episodes, "rendezvous_alignment"
                ),
                "mean_predicted_entry_radial_m": _mean(
                    recent_episodes, "predicted_entry_radial_m"
                ),
                "mean_predicted_contact_radial_m": _mean(
                    recent_episodes, "predicted_contact_radial_m"
                ),
                "mean_minimum_capture_volume_error_m": _mean(
                    recent_episodes, "minimum_capture_volume_error_m"
                ),
                "mean_maximum_grasp_dwell_fraction": _mean(
                    recent_episodes, "maximum_grasp_dwell_fraction"
                ),
                "mean_maximum_cup_displacement_m": _mean(
                    recent_episodes, "maximum_cup_displacement_m"
                ),
                "recent_radial_failure_rate": _mean(recent_episodes, "failure_radial"),
                "recent_axial_failure_rate": _mean(recent_episodes, "failure_axial"),
                "recent_speed_failure_rate": _mean(recent_episodes, "failure_speed"),
                "recent_alignment_failure_rate": _mean(
                    recent_episodes, "failure_alignment"
                ),
                "recent_unsafe_force_rate": _mean(recent_episodes, "unsafe_force"),
                "mean_grasp_jump_m": _mean(recent_episodes, "grasp_jump_m"),
                "mean_energy_proxy_j": _mean(recent_episodes, "energy_proxy_j"),
                "mean_reference_velocity_saturation_fraction": _mean(
                    recent_episodes, "reference_velocity_saturation_fraction"
                ),
                "mean_predicted_landing_error_m": _mean(
                    recent_episodes, "landing_error_m"
                ),
                "mean_release_velocity_error_mps": _mean(
                    recent_episodes, "release_velocity_error_mps"
                ),
                "mean_release_landing_error_m": _mean(
                    recent_episodes, "release_landing_error_m"
                ),
                "mean_release_flight_time_s": _mean(
                    recent_episodes, "release_flight_time_s"
                ),
                **evaluation_fields,
                **metrics,
            }
            history.append(row)
            line = json.dumps(row)
            print(line, flush=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

            if promoted:
                env.set_curriculum(next_stage, next_level)
                observation, _ = env.reset(args.seed + episodes + update_index * 100_000)
                episode_return, episode_length = 0.0, 0
                recent_episodes.clear()
                stage_updates = 0
                last_evaluation = None
                last_evaluation_update = None
            save_checkpoint(
                args.out_dir / "ppo_latest.pt", model, optimizer,
                update_index=update_index, config=run_config,
                curriculum_stage=env.curriculum_stage,
                stage2_level=env.stage2_level, stage_updates=stage_updates,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted; preserving the latest completed update.", flush=True)
    finally:
        env.close()

    report = {
        "schema": CHECKPOINT_SCHEMA, "config": run_config,
        "elapsed_s": time.perf_counter() - started, "episodes": episodes,
        "curriculum_successes": curriculum_successes,
        "final_task_successes": final_task_successes,
        "best_evaluation_update": best_evaluation_update,
        "interrupted": interrupted, "history": history,
    }
    (args.out_dir / "training_history.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
