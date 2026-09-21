"""Read-only release estimator audit on stored V16 scenes; never retunes control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teacher_rl.continuous_env import ContinuousConfig, ContinuousEnv
from teacher_rl.improved_rl import load_anchor, read_json
from teacher_rl.improved_teacher import ImprovedRecipe
from can_release import landing


def episode(row):
    torch.set_num_threads(1)
    anchor,payload = load_anchor(Path("teacher_runs/v15_improved/bc_v1_formal/bc_best.pt"))
    env = ContinuousEnv(row["scenario"],ImprovedRecipe(**payload["recipe"]),anchor,
        ContinuousConfig(**row["result"]["continuous_config"]))
    env.reset(row["seed"])
    audit = dict(seed=row["seed"])
    original = env.releaser.original
    state_fn = original.ball_state
    last_state = {}
    predictions = []
    prediction_fn = original.predict_from
    audited_time = None
    gate_counts = dict(ticks=0,future_minimum=0,low_speed=0,low_elevation=0,outside_tolerance=0)
    best_current = None

    def prediction(p,v,delay):
        answer = prediction_fn(p,v,delay)
        predictions.append(answer)
        return answer

    original.predict_from = prediction

    def state(obs):
        predictions.clear()
        p,v = state_fn(obs)
        last_state.update(time=obs["t_win"],position=p.copy(),velocity=v.copy(),
            true_position=env.plant.ball_pos(),true_velocity=env.plant.ball_vel())
        return p,v

    original.ball_state = state
    hook = env.plant.hook_after_step

    def inspect(t):
        nonlocal audited_time,best_current
        hook(t)
        if predictions and audited_time != last_state["time"]:
            audited_time = last_state["time"]
            # The existing spring-delay solver calls predict twice, then once per candidate.
            candidates = predictions[2::3] if original.depart_fn is not None else predictions
            errors = [float(np.linalg.norm(item[0][:2]-original.target)) for item in candidates]
            k = int(np.argmin(errors))
            velocity = candidates[k][1]
            speed = float(np.linalg.norm(velocity))
            elevation = float(np.arctan2(velocity[2],np.hypot(velocity[0],velocity[1])))
            gate_counts["ticks"] += 1
            gate_counts["future_minimum"] += int(k > original.sub)
            gate_counts["low_speed"] += int(speed < original.v_min)
            gate_counts["low_elevation"] += int(elevation < original.min_elev)
            gate_counts["outside_tolerance"] += int(errors[k] > original.tol)
            if k <= original.sub and speed >= original.v_min and elevation >= original.min_elev:
                if best_current is None or errors[k] < best_current["error_m"]:
                    best_current = dict(error_m=errors[k],time_s=audited_time,speed_m_s=speed)
        if original.t_release is not None and "command" not in audit:
            command = original.state_at_cmd
            delay = command["due_in_s"]
            predicted,pred_v,pred_p = original.predict_from(last_state["position"],last_state["velocity"],delay)
            audit["command"] = dict(**last_state,delay=delay,omega=original.omega,edot=original.edot,
                predicted_land=predicted,predicted_departure_velocity=pred_v,predicted_departure_position=pred_p)
        if env.release_time is not None and "departure" not in audit:
            audit["departure"] = dict(time=t-2.,position=env.plant.ball_pos(),velocity=env.plant.ball_vel())
    env.plant.hook_after_step = inspect
    while not env.done:
        _,_,_,result = env.step(np.zeros(4,dtype=np.float32))
    for key in ("landing_error_m","capture_time","release_time"):
        actual, expected = result[key], row["result"][key]
        matches = actual is expected if actual is None or expected is None else np.isclose(actual,expected,atol=1e-8,rtol=1e-8)
        if not matches:
            raise RuntimeError(f"Diagnostic rollout does not reproduce stored baseline: {key}")
    audit["result"] = result
    audit["best_predicted_error_m"] = float(original.best[0]) if np.isfinite(original.best[0]) else None
    audit["best_prediction_time_s"] = original.best[1]
    audit["release_gate_counts"] = gate_counts
    audit["best_current_tick_minimum"] = best_current
    if "departure" in audit:
        d,c = audit["departure"],audit["command"]
        audit["errors"] = dict(velocity_estimate_m_s=np.linalg.norm(c["velocity"]-c["true_velocity"]),
            position_estimate_m=np.linalg.norm(c["position"]-c["true_position"]),
            departure_delay_s=d["time"]-c["time"]-c["delay"],
            departure_velocity_m_s=np.linalg.norm(d["velocity"]-c["predicted_departure_velocity"]),
            departure_position_m=np.linalg.norm(d["position"]-c["predicted_departure_position"]),
            predicted_vs_actual_landing_m=np.linalg.norm(c["predicted_land"][:2]-result["landing_xy"]),
            free_flight_vs_actual_landing_m=np.linalg.norm(landing(d["position"],d["velocity"])[0][:2]-result["landing_xy"]))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("teacher_runs/v16_continuous/rl_smoke/validation_0000.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[9200001,9200005,9200008,9200010])
    args = parser.parse_args()
    source = args.source
    stored = read_json(source)
    rows = [r for r in (stored["rows"] if isinstance(stored,dict) else stored) if r["seed"] in args.seeds]
    if {r["seed"] for r in rows} != set(args.seeds):
        raise ValueError("Requested seeds are missing from stored source")
    with ProcessPoolExecutor(max_workers=min(4,len(rows))) as pool:
        results = list(pool.map(episode,rows))
    output = source.parent/"release_diagnostics.json"
    output.write_text(json.dumps(results,indent=2,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else float(x)),encoding="utf-8")
    print(json.dumps([dict(seed=r["seed"],best_predicted_error_m=r["best_predicted_error_m"],
        release_gate_counts=r["release_gate_counts"],best_current_tick_minimum=r["best_current_tick_minimum"],
        **r.get("errors",{})) for r in results]))
