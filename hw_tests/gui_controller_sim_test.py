"""Exercise dynamic targets in the real Tk window, with hardware opens poisoned.

    .venv/Scripts/python.exe hw_tests/gui_controller_sim_test.py

This is functional acceptance against SIM. Timing is measured and reported;
only a separate unloaded run establishes whether the 150 Hz deadline is met.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))


def _wait(root, condition, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update_idletasks()
        root.update()
        if condition():
            return True
        time.sleep(0.01)
    return bool(condition())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="pid,ff_pid,koopman_mppi")
    parser.add_argument("--out", type=Path, default=WS / "control/results/gui_acceptance.json")
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--jog-settle-s", type=float, default=9.0)
    args = parser.parse_args(argv)
    from hw_tests.gui_sim_test import ATTEMPTS, poison, pump
    poison()
    import tkinter as tk
    import canarm_control_gui as GUI
    from collection.safety import PairEnvelope

    root = tk.Tk()
    app = GUI.CanArmControllerApp(root, prefer_sim=True)
    records = []
    report = {"sim_only": True, "fresh_plant_per_method": True,
              "jog_settle_s": args.jog_settle_s, "methods": records, "failures": []}
    report["source_sha256"] = {
        name: hashlib.sha256((WS / name).read_bytes()).hexdigest()
        for name in ("control/controller.py", "control/koopman.py",
                     "control/controller_process.py", "control/observation.py",
                     "digital_twin/sim_mocap.py", "control/checkpoints/canarm_koopman.pt")
        if (WS / name).exists()}
    try:
        # No connect first: a Start press must not construct a hardware backend.
        app.toggle_controller()
        assert app._controller is None and app.backend is None
        app.toggle_connect()
        assert app.backend is not None
        assert _wait(root, lambda: app._twin_q() is not None)
        assert app._mocap.rate_hz == 240
        assert app._mocap.joint_noise_std_rad > 0
        methods = args.methods.split(",")
        for index, method in enumerate(methods):
            if index:
                app.disconnect()
                app._reap_twin_mocap()
                app.toggle_connect()
                assert _wait(root, lambda: app._mocap.latest_control_sample()[2] is not None)
            record = {"method": method}
            records.append(record)
            app.controller_method.set(method)
            app.toggle_controller()
            bridge = app._controller
            assert bridge is not None, "controller start refused"
            assert _wait(root, lambda: bridge.commands >= 20 or not bridge.running), bridge.status
            assert bridge.commands >= 20, bridge.status
            assert bridge.process.pid != os.getpid()
            window = app._target_window
            assert window is not None and len(window.variables) == 12
            record["child_pid"] = bridge.process.pid
            cycle_t0 = time.perf_counter()
            cycle_n0 = app.backend.stats.cycles
            reply_n0, miss_n0 = app.backend.stats.replies, app.backend.stats.misses
            q0 = app._mocap.get_q().copy()
            slider_target = 5.0
            record["joint2_target_deg"] = slider_target
            window.variables[1].set(slider_target)
            window._slider()
            samples = []
            for _ in range(150):
                pump(root, 0.02, tick=0.005)
                stamp, seq, q_sample, qd_sample = app._mocap.latest_control_sample()
                samples.append(np.r_[time.perf_counter(), q_sample, qd_sample,
                                     bridge.last_pressure_pa])
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out.with_name(args.out.stem + f"_{method}_trace.npz"),
                                samples=np.asarray(samples), columns="time,q12,qdot12,target_pa24")
            assert bridge.running, bridge.status
            moved = float(np.degrees(app._mocap.get_q()[1] - q0[1]))
            record["joint2_motion_deg"] = moved
            record["joint2_start_deg"] = float(np.degrees(q0[1]))
            record["joint2_final_deg"] = float(np.degrees(app._mocap.get_q()[1]))
            record["joint2_last_half_second_mean_deg"] = float(np.degrees(np.mean(np.array(samples)[-25:, 2])))
            record["joint2_range_deg"] = np.degrees(np.array(samples)[:, 2]).take(
                [np.argmin(np.array(samples)[:, 2]), np.argmax(np.array(samples)[:, 2])]).tolist()
            record["measured_backend_hz"] = ((app.backend.stats.cycles - cycle_n0) /
                                              (time.perf_counter() - cycle_t0))
            error0 = abs(record["joint2_start_deg"] - slider_target)
            error1 = abs(record["joint2_final_deg"] - slider_target)
            assert (error1 < error0 - 0.5 or error1 < 1.0), (
                f"joint slider did not approach target: {error0:.3f} -> {error1:.3f} deg error")
            app._channel_moved(0x101, 40.0)
            app._set_target(0x101, 40.0)
            nodes = app.backend.snapshot_nodes()
            envelope = PairEnvelope()
            envelope.assert_safe([nodes[b].target_psi for b in sorted(nodes)])
            record["manual_override_refused"] = True
            from control.trajectory import tip_position
            # Start the jog from the observed pose, as an operator's Hold does.
            # Ground-truth twin q is used only to score motion below.
            window.hold()
            previous_target = bridge.target_q.copy()
            tip_before = tip_position(app.backend.arm.q())
            window.step_mm.set("10")
            window.nudge(0, 1)
            tip_requested = window._ik_target.copy()
            assert _wait(root, lambda: window._pending is None, 10), "tip IK did not finish"
            record["tip_message"] = window.note.get()
            assert "accepted" in window.note.get(), window.note.get()
            assert np.linalg.norm(bridge.target_q - previous_target) > 1e-5
            tip_samples = []
            jog_samples = []
            jog_end = time.monotonic() + args.jog_settle_s
            while time.monotonic() < jog_end:
                pump(root, 0.04, tick=0.005)
                true_tip = tip_position(app.backend.arm.q())
                tip_samples.append(true_tip)
                stamp, seq, observed_q, observed_qdot = app._mocap.latest_control_sample()
                jog_samples.append(np.r_[time.perf_counter(), observed_q, observed_qdot,
                                         bridge.target_q, bridge.last_pressure_pa, true_tip])
            np.savez_compressed(args.out.with_name(args.out.stem + f"_{method}_jog_trace.npz"),
                                samples=np.asarray(jog_samples), target_tip_m=tip_requested,
                                columns="time,q12,qdot12,requested_q12,target_pa24,truth_tip3")
            tip_after = np.mean(tip_samples[-10:], axis=0)
            before_error = float(np.linalg.norm(tip_before - tip_requested))
            after_error = float(np.linalg.norm(tip_after - tip_requested))
            record.update(tip_jog_mm=10.0, tip_requested_m=tip_requested.tolist(),
                          tip_before_m=tip_before.tolist(), tip_after_m=tip_after.tolist(),
                          tip_error_before_mm=1000 * before_error,
                          tip_error_after_mm=1000 * after_error,
                          tip_motion_mm=float(1000 * np.linalg.norm(tip_after - tip_before)))
            assert after_error < before_error, (
                f"tip jog did not reduce error: {before_error * 1000:.2f} -> {after_error * 1000:.2f} mm")
            assert bridge.running, bridge.status
            record.update(applied_hz=bridge.command_hz, last_solve_ms=bridge.solve_ms,
                          startup_warmup_ms=bridge.warmup_ms,
                          deadline_misses=bridge.deadline_misses, commands=bridge.commands,
                          backend_hz=((app.backend.stats.cycles - cycle_n0) /
                                      (time.perf_counter() - cycle_t0)),
                          backend_reply_fraction=(
                              (app.backend.stats.replies - reply_n0) /
                              max(1, app.backend.stats.replies - reply_n0 +
                                     app.backend.stats.misses - miss_n0)),
                          mocap_hz=app._mocap.get_state().fps)
            record["solve_ms_p50_p95_p99_max"] = np.percentile(
                list(bridge.solve_times_ms), [50, 95, 99, 100]).tolist()
            record["applied_interval_ms_p50_p95_p99_max"] = np.percentile(
                list(bridge.command_intervals_ms), [50, 95, 99, 100]).tolist()
            record["worker_commands"] = bridge.worker_commands
            if args.screenshots and index == len(methods) - 1:
                from hw_tests.gui_sim_test import grab_root
                window.window.withdraw()
                root.attributes("-topmost", True)
                pump(root, 0.10, tick=0.005)
                record["operator_screenshot"] = grab_root(root, "gui_dynamic_operator.png")
                root.attributes("-topmost", False)
                window.window.deiconify()
                window.window.attributes("-topmost", True)
                window.window.lift()
                pump(root, 0.10, tick=0.005)
                record["targets_screenshot"] = grab_root(window.window, "gui_dynamic_targets.png")
                window.window.attributes("-topmost", False)
            if index == 0:
                window.on_close()
                record["shutdown"] = "target window close"
            elif index == 1:
                bridge.process.terminate()
                assert _wait(root, lambda: app._controller is None, 3)
                record["shutdown"] = "worker terminated"
            else:
                app._mocap.stop()
                assert _wait(root, lambda: app._controller is None, 3)
                record["shutdown"] = "mocap stopped"
            assert not bridge.process.is_alive()
            assert not any(n.enabled for n in app.backend.snapshot_nodes().values())
            assert all(n.target_psi == 0 for n in app.backend.snapshot_nodes().values())
            record["all_boards_disabled"] = True
            if index >= 2 and index + 1 < len(methods):
                app._stop_mocap()
                app.mocap_source.set("twin")
                app.toggle_mocap()
                assert _wait(root, lambda: not app._mocap.get_state().q_stale)
        app.disconnect()
        assert app.backend is None and app._controller is None
        assert not any(ATTEMPTS.values()), ATTEMPTS
        report["hardware_open_attempts"] = ATTEMPTS
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        report["gui_log"] = app.log_text.get("1.0", "end")
    finally:
        app.on_close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
