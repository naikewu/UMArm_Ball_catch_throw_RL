"""Offline process/watchdog checks; the fake plant never opens a port."""
from __future__ import annotations

import os
import time
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from control.controller_process import (ControllerBridge, validate_joint_target,
                                        quantized_safe_targets)
from digital_twin.sim_master import SimMaster
from TLE_PCB.tlelib.proto import NodeCal


class _Backend(SimMaster):
    def __init__(self):
        # Only the transport interface is needed for watchdog tests; no MuJoCo
        # or board object is constructed, and no hardware code is invoked.
        self._fake_running = False
        self.enabled = False
        self.targets = {}
        self.selected = []
        self.pressure_stale = False
        self.stop_calls = 0

    def snapshot_nodes(self):
        stamp = time.perf_counter() - (2.0 if self.pressure_stale else 0.0)
        return {b: SimpleNamespace(present=True, pressure_psi=0.0, last_reply_t=stamp,
                                  cal=NodeCal.for_variant(2 if b % 2 else 1, b))
                for b in range(0x101, 0x119)}

    @property
    def running(self):
        return self._fake_running

    def select(self, bases):
        self.selected = list(bases)

    def set_targets(self, mapping):
        self.targets = dict(mapping)

    def start_cycle(self):
        self._fake_running = True

    def set_enabled_all(self, on):
        self.enabled = on

    def stop_all(self):
        self.enabled = False
        self.stop_calls += 1


class _Mocap:
    def __init__(self, backend):
        self.bound_backend = backend
        self.stale = False
        self.seq = 0

    def latest_sample(self):
        self.seq += 1
        return (time.monotonic() - (2.0 if self.stale else 0.0), 1,
                np.zeros(12), None, self.seq)

    def latest_control_sample(self):
        stamp, _, q, _, seq = self.latest_sample()
        return stamp, seq, q, np.zeros(12)


def _until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


class ProcessTests(unittest.TestCase):
    def test_quantized_pair_caps_use_each_node_calibration(self):
        from collection.safety import PairEnvelope
        envelope = PairEnvelope()
        nodes = _Backend().snapshot_nodes()
        requested = np.zeros(24)
        requested[envelope.pair_idx[:, 0]] = 30.0
        wire = quantized_safe_targets(requested, nodes)
        self.assertTrue(np.all(wire <= 30))
        envelope.assert_safe(wire)
        for i, b in enumerate(sorted(nodes)):
            cal = nodes[b].cal
            self.assertAlmostEqual(max(0, cal.counts_to_psi(cal.psi_to_counts(wire[i]))), wire[i])

    def test_reject_live_adapter_before_any_action(self):
        with self.assertRaisesRegex(ValueError, "SIM-only"):
            ControllerBridge(object(), object())

    def test_joint_targets_refuse_nan_shape_and_outside_bounds(self):
        for value in (np.zeros(11), np.full(12, np.nan), np.full(12, np.deg2rad(26))):
            with self.assertRaises(ValueError):
                validate_joint_target(value)

    def test_wrong_receiver_refused(self):
        with self.assertRaisesRegex(ValueError, "mocap"):
            ControllerBridge(_Backend(), _Mocap(_Backend()))

    def test_stale_start_does_not_enable(self):
        backend = _Backend()
        mocap = _Mocap(backend)
        mocap.stale = True
        bridge = ControllerBridge(backend, mocap)
        with self.assertRaisesRegex(ValueError, "fresh"):
            bridge.start()
        self.assertFalse(backend.enabled)

    def test_cleanup_handles_process_or_thread_start_failure(self):
        backend = _Backend()
        bridge = ControllerBridge(backend, _Mocap(backend))
        bridge.process = bridge._ctx.Process()
        bridge.thread = threading.Thread()
        bridge.stop()
        self.assertFalse(backend.enabled)
        self.assertTrue(all(v == 0 for v in backend.targets.values()))

    def test_real_child_runs_then_stale_sample_disables(self):
        backend = _Backend()
        mocap = _Mocap(backend)
        bridge = ControllerBridge(backend, mocap).start()
        try:
            self.assertNotEqual(bridge.process.pid, os.getpid())
            self.assertTrue(_until(lambda: bridge.commands >= 3 or not bridge.running), bridge.status)
            self.assertGreaterEqual(bridge.commands, 3, bridge.status)
            self.assertTrue(backend.enabled)
            self.assertEqual(len(backend.targets), 24)
            mocap.stale = True
            self.assertTrue(_until(lambda: not bridge.running, 2))
            self.assertFalse(backend.enabled)
            self.assertIn("mocap", bridge.error)
        finally:
            bridge.stop()

    def test_process_death_disables_and_releases_commands(self):
        backend = _Backend()
        bridge = ControllerBridge(backend, _Mocap(backend)).start()
        try:
            self.assertTrue(_until(lambda: bridge.commands >= 2 or not bridge.running), bridge.status)
            self.assertGreaterEqual(bridge.commands, 2, bridge.status)
            bridge.process.terminate()
            bridge.process.join(2)
            self.assertTrue(_until(lambda: not bridge.running, 2))
            self.assertFalse(backend.enabled)
            self.assertTrue(all(v == 0 for v in backend.targets.values()))
        finally:
            bridge.stop()

    def test_pressure_staleness_disables(self):
        backend = _Backend()
        bridge = ControllerBridge(backend, _Mocap(backend)).start()
        try:
            self.assertTrue(_until(lambda: bridge.commands >= 2 or not bridge.running), bridge.status)
            self.assertGreaterEqual(bridge.commands, 2, bridge.status)
            backend.pressure_stale = True
            self.assertTrue(_until(lambda: not bridge.running, 2))
            self.assertFalse(backend.enabled)
            self.assertIn("pressure", bridge.error)
        finally:
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
