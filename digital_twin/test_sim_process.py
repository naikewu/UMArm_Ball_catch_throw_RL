r"""``SimMaster(physics="process")``: the plant in a child, the interface unchanged.

The property under test is CONTRACT.md section 5's, extended to timing: a
controller must not be able to tell the twin from the arm, and with the plant
stepped on the host's interpreter lock it could, from ``Backend.stats`` alone.
So these pin that the host process steps no physics, that the plant child is
spawned, answers and is reaped, that what crosses back is the same reply stream
the in-process link produces (latency column, misses, the TLE failsafe), and
that the two placements drive the same plant to the same place.

Each test that opens a link spawns a child, which costs about 2-3 s of import
and model build; they are kept few.  Nothing here opens a serial port: the host
side never constructs a ``CanLink``, and the child replaces ``serial.Serial``
with a refusal before it builds anything.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from digital_twin import sim_core as sc
from digital_twin import sim_master as sm
from digital_twin import sim_process as spr
from TLE_PCB.tlelib import proto as P
from TLE_PCB.tlelib.backend import Backend

PSI = sc.PA_PER_PSI


def _actuator():
    return sc.placeholder_actuator(is_tle=[b in P.TLE_IDS for b in sc.ALL_IDS])


def make_process_master(**kwargs):
    kwargs.setdefault("actuator", _actuator())
    return sm.SimMaster(physics="process", **kwargs)


def _run(master, targets_psi, run_s, *, select_all=True):
    master.scan()
    master.select(list(sc.ALL_IDS) if select_all else list(targets_psi))
    for base, psi in targets_psi.items():
        master.set_target(base, psi)
        master.set_enabled(base, True)
    seen = []
    master.set_cycle_observer(lambda t, targets, replies: seen.append(len(replies)))
    master.start_cycle()
    time.sleep(run_s)
    master.stop_cycle()
    return seen


def test_the_process_twin_is_still_backend_and_refuses_what_cannot_cross():
    master = make_process_master()
    assert isinstance(master, Backend)
    assert isinstance(master.link, spr.SimProcessLink)
    assert isinstance(master.link, sm.SimCanLink)
    assert isinstance(master.arm, spr.RemoteArm)
    assert master.link.port == "SIM" and not master.link.is_open
    assert not hasattr(master.arm, "advance_to"), "nothing on the host moves the plant"
    assert master.arm.model.nq == 12

    with pytest.raises(ValueError, match="physics"):
        sm.SimMaster(physics="thread", actuator=_actuator())
    with pytest.raises(ValueError, match="cannot cross"):
        sm.SimMaster(physics="process", arm=sc.SimArm(actuator=_actuator()))
    with pytest.raises(ValueError, match="node_hook"):
        make_process_master(node_hook=lambda now: None)
    with pytest.raises(ValueError, match="perf_counter"):
        make_process_master(clock=time.monotonic)

    own, tunables = spr.split_simarm_kwargs({"actuator": 1, "absent": (0x101,),
                                             "link_mass_kg": 0.9, "joint_damping": 0.1})
    assert set(own) == {"actuator", "absent"}
    assert set(tunables) == {"link_mass_kg", "joint_damping"}


def test_a_plant_that_fails_to_build_says_why_and_leaves_nothing_running():
    master = make_process_master(timestep_s=-1.0)
    with pytest.raises(RuntimeError, match="timestep_s must be positive"):
        master.open()
    assert not master.link.is_open and master.link._proc is None


def test_the_cycle_runs_through_the_plant_and_the_host_steps_no_physics():
    """One spawn carries the transport checks, to keep the file's wall time down."""
    master = make_process_master(absent=(0x107,))
    master.open()
    try:
        assert master.link.is_open and master.link.plant_pid is not None
        found = master.scan()
        assert 0x107 not in found and len(found) == 23
        assert found[0x101].is_tle and found[0x101].version == sm.TLE_FIRMWARE_VERSION
        assert found[0x110].kind == P.VARIANT_NAMES[P.VARIANT_7MM]

        q0 = master.arm.q()
        quanta0 = master.arm.quanta_done
        t0 = time.perf_counter()
        seen = _run(master, {0x101: 10.0, 0x10A: 10.0}, 1.5)
        elapsed = time.perf_counter() - t0
        snap = master.snapshot_nodes()
        truth = master.snapshot_arm()
        q1 = master.arm.q()

        # The plant's clock ran in the child, at wall-clock pace.
        assert master.arm.quanta_done - quanta0 == pytest.approx(elapsed * 1000.0, rel=0.15)
        # The reply stream: every edge answered by all 23 boards, the metal's column.
        assert len(seen) >= int(1.5 * P.CYCLE_HZ * 0.9)
        complete = sum(1 for n in seen if n == 23)
        assert complete >= len(seen) - 2, f"{len(seen) - complete} incomplete cycles"
        assert master.stats.misses <= 0.01 * (master.stats.replies + master.stats.misses)
        present = [b for b in sc.ALL_IDS if b != 0x107]
        lat = np.array([snap[b].reply_latency_ms for b in present])
        assert (np.diff(lat) > 0.0).all()
        assert lat[0] == pytest.approx(sm.REPLY_LATENCY_BASE_MS, abs=0.15)
        assert lat[-1] == pytest.approx(sm.REPLY_LATENCY_LAST_MS, abs=0.15)
        assert master.link.sync_count >= len(seen)
        # The plant moved, and the host saw it through the reply and shared memory.
        for base in (0x101, 0x10A):
            assert 7.0 < snap[base].pressure_psi < 13.0
            assert snap[base].flags & P.STATUS_COMMAND_SEEN
        assert truth[0x101]["true_psi"] > 7.0 and truth[0x101]["enabled"] is False
        assert float(q1[2] - q0[2]) > 1e-3, "0x101 drives joint 2 positive"

        # A silent host: edges stop without the safe-disable.  The child's clock
        # keeps running, so the TLE failsafe trips and the 7 mm board holds.
        master.set_enabled(0x101, True)
        master.set_enabled(0x10A, True)
        master.start_cycle()
        time.sleep(0.3)
        master._running.clear()
        master._thread.join(timeout=1.0)
        master._thread = None
        time.sleep(0.8)
        truth = master.snapshot_arm()
        assert truth[0x101]["failsafe_active"] is True and truth[0x101]["enabled"] is False
        assert truth[0x10A]["failsafe_active"] is False and truth[0x10A]["enabled"] is True
    finally:
        pid = master.link.plant_pid
        proc = master.link._proc
        master.close()
    assert not master.link.is_open
    assert proc is not None and not proc.is_alive(), f"plant pid {pid} outlived close()"


def test_the_twin_mocap_reads_a_process_plant_through_the_real_receiver():
    from digital_twin.sim_mocap import SimMocap

    master = make_process_master()
    master.open()
    rx = None
    try:
        master.scan()
        rx = SimMocap(master.arm, alive=lambda: master.link.is_open).start()
        q = rx.wait_fresh(timeout=3.0)
        assert q is not None
        np.testing.assert_allclose(q, master.arm.q(), atol=1e-3)
        res = rx.fk_residual_m()
        assert res is not None and float(np.max(res)) < 1e-4
    finally:
        if rx is not None:
            rx.stop()
        master.close()


def test_the_two_placements_drive_the_plant_to_the_same_place():
    """Same targets, same plant: inline and process agree at the settled state.

    Not bit-identical -- each advances on its own thread's wake-ups against the
    wall clock -- so the comparison is the settled pressure and joint angle after
    2 s at 10 psi on one board, where both have long stopped moving.
    """
    out = {}
    for physics in ("inline", "process"):
        master = sm.SimMaster(physics=physics, actuator=_actuator())
        master.open()
        try:
            q0 = master.arm.q()
            _run(master, {0x101: 10.0}, 2.0)
            out[physics] = (master.snapshot_nodes()[0x101].pressure_psi,
                            np.degrees(master.arm.q() - q0))
        finally:
            master.close()
    p_in, dq_in = out["inline"]
    p_pr, dq_pr = out["process"]
    assert p_in == pytest.approx(p_pr, abs=0.5)
    np.testing.assert_allclose(dq_in, dq_pr, atol=0.5)
    assert abs(dq_in[2]) > 0.05
