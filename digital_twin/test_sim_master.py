r"""The shell, as assertions.  ``reference/sim_core.md`` section 7.5-7.6 is the source.

The property under test throughout is CONTRACT.md section 5's: **a controller
must not be able to discriminate between the real robot and the twin**.  So the
first group pins the interface against ``TLE_PCB.tlelib.backend.Backend``
itself, by ``inspect.signature`` rather than by eye, and the rest exercises the
transport semantics a controller can actually observe -- the per-id reply
latency, what the runtime table addresses, what a scan reports, and what happens
when the host stops driving.

Nothing here opens a serial port.  ``SimMaster`` defaults ``port`` to ``"SIM"``
and replaces ``Backend``'s link before anything is opened, and
:func:`test_no_serial_port_is_ever_opened` asserts exactly that -- a live
data-collection campaign is on COM58 while these run.

Wall-clock cost: the cycle thread is the real one at 150 Hz, so each test that
starts it pays its own duration.  They are kept under a second apiece.
"""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np
import pytest

from digital_twin import sim_core as sc
from digital_twin import sim_master as sm
from TLE_PCB.tlelib import proto as P
from TLE_PCB.tlelib.backend import Backend

PSI = sc.PA_PER_PSI

#: Long enough for the 150 Hz cycle to publish a few dozen times and for every
#: board's reply to have been matched at least once.  Short enough that the
#: whole file stays inside a few seconds of wall time.
RUN_S = 0.5


def make_master(**kwargs):
    """A twin on the real generated scene, with its own controlled actuator.

    The actuator is the bring-up stand-in rather than a fresh checkpoint,
    because these tests assert *transport* behaviour and want a plant whose fill
    rate is a known constant -- an untrained net's is whatever its random
    weights give, and a shell test failing because a checkpoint changed would be
    a test failing for the wrong reason.
    """
    kwargs.setdefault("actuator", sc.placeholder_actuator(
        is_tle=[b in P.TLE_IDS for b in sc.ALL_IDS]))
    return sm.SimMaster(**kwargs)


def _drive(master, targets_psi, run_s=RUN_S):
    """Scan, select, command, run the real cycle, stop cleanly."""
    master.open()
    master.scan()
    master.select(list(sc.ALL_IDS))
    for base, psi in targets_psi.items():
        master.set_target(base, psi)
        master.set_enabled(base, True)
    master.start_cycle()
    time.sleep(run_s)
    master.stop_cycle()
    return master.snapshot_nodes()


# ===========================================================================
# M1-M4 -- interface identity
# ===========================================================================

SHARED_METHODS = ("log", "open", "close", "scan", "select", "set_target",
                  "set_target_all", "set_enabled", "set_enabled_all",
                  "stop_all", "send_native", "apply_tuning", "request_extended",
                  "start_cycle", "stop_cycle", "set_cycle_observer",
                  "snapshot_nodes", "history", "missing_nodes")


def test_m1_every_shared_method_signature_equals_the_real_backends():
    """Pinned with ``inspect.signature``, not by eye.

    The twin inherits all of these today, so the assertion is trivially true --
    which is the point.  It fails the day somebody overrides one and quietly
    changes a parameter name or a default, and that is exactly the change a
    controller would trip over on the real bus and not in the twin.
    """
    for name in SHARED_METHODS:
        real = inspect.signature(getattr(Backend, name))
        twin = inspect.signature(getattr(sm.SimMaster, name))
        assert twin == real, f"{name} drifted: {twin} != {real}"
    assert isinstance(sm.SimMaster.running, property)
    assert issubclass(sm.SimMaster, Backend)


def test_m1b_the_constructors_positional_surface_is_the_backends():
    """Same names, kinds and defaults, with ``port`` deliberately excepted.

    ``port`` defaults to ``"SIM"`` instead of the resolved COM port so that
    constructing a twin can never reach for the dongle a live campaign is using.
    Everything the twin needs of its own is keyword-only and comes after.
    """
    real = inspect.signature(Backend.__init__).parameters
    twin = inspect.signature(sm.SimMaster.__init__).parameters
    for name in ("port", "bitrate", "cycle_hz", "log"):
        assert name in twin
        assert twin[name].kind == real[name].kind
        if name != "port":
            assert twin[name].default == real[name].default
    assert twin["port"].default == "SIM"
    assert twin["bitrate"].default == 1_000_000
    assert twin["cycle_hz"].default == P.CYCLE_HZ

    extra = [p for p in twin.values()
             if p.name not in ("self", "port", "bitrate", "cycle_hz", "log")]
    assert all(p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD) for p in extra)


def test_m2_the_cycle_observer_fires_once_per_cycle_with_the_documented_shapes():
    """``set_cycle_observer`` is the only way to record at 150 Hz.

    ``snapshot_nodes`` deep-copies twenty-four dataclasses and a poller runs on
    its own clock, so every polled sample is both expensive and off the sync
    grid -- and the sync edge is the only instant the whole arm agrees on.
    """
    master = make_master()
    seen = []
    master.set_cycle_observer(lambda t, targets, replies: seen.append(
        (t, dict(targets), dict(replies))))
    _drive(master, {0x101: 10.0, 0x10A: 10.0})
    master.close()

    assert len(seen) >= int(RUN_S * P.CYCLE_HZ * 0.7)
    assert len(seen) == master.stats.cycles

    t, targets, replies = seen[-1]
    assert isinstance(t, float)
    assert set(targets) == set(sc.ALL_IDS)
    counts, enable = targets[0x101]
    assert isinstance(counts, int) and isinstance(enable, bool)
    assert set(replies) == set(sc.ALL_IDS)
    t_reply, status = replies[0x101]
    assert isinstance(status, P.CompactStatus)
    assert t_reply > t

    # Monotone sync stamps, one bus period apart.
    stamps = np.array([row[0] for row in seen])
    assert (np.diff(stamps) > 0.0).all()
    assert np.median(np.diff(stamps)) == pytest.approx(1.0 / P.CYCLE_HZ, rel=0.2)


def test_m3_the_reply_latency_rises_with_the_identifier_and_is_not_flat():
    """CAN arbitration orders the answers by node id, and a flat column would
    give the twin away.

    The column reads a few tens of microseconds above the model because
    ``Backend`` stamps ``t_sync`` before it builds and writes the table, and the
    twin's table build costs real time exactly as the host's does.
    """
    master = make_master()
    snap = _drive(master, {0x101: 8.0})
    master.close()

    latencies = np.array([snap[b].reply_latency_ms for b in sc.ALL_IDS])
    assert (np.diff(latencies) > 0.0).all()
    assert latencies[0] == pytest.approx(sm.REPLY_LATENCY_BASE_MS, abs=0.15)
    assert latencies[-1] == pytest.approx(sm.REPLY_LATENCY_LAST_MS, abs=0.15)
    assert latencies[-1] - latencies[0] == pytest.approx(
        sm.REPLY_LATENCY_LAST_MS - sm.REPLY_LATENCY_BASE_MS, abs=0.05)

    steps = np.diff(latencies)
    assert steps.mean() == pytest.approx(0.0643, abs=0.005)     # ms per id step
    assert steps.std() < 0.01                                   # linear, not noise

    for base in (P.ACTUATOR_FIRST, 0x110, P.ACTUATOR_LAST):
        assert snap[base].reply_latency_ms >= sm.reply_latency_ms(base)


def test_m3b_the_latency_endpoints_are_arguments_not_baked_in():
    """A re-measurement replaces them at the call site, not by editing a module
    a dozen callers share."""
    assert sm.reply_latency_ms(0x101) == pytest.approx(1.87)
    assert sm.reply_latency_ms(0x118) == pytest.approx(3.35)
    assert sm.reply_latency_ms(0x102) - sm.reply_latency_ms(0x101) == \
        pytest.approx((3.35 - 1.87) / 23.0)
    assert sm.reply_latency_ms(0x109, first_ms=1.0, last_ms=1.0) == 1.0


def test_m4_every_known_board_is_tabled_every_cycle_not_just_the_selection():
    """De-selecting a board must not leave it regulating out of reach.

    A board keeps whatever control byte it was last given, so omitting it from
    the table does not turn it off -- it stops updating it, and the sync edge
    goes on promoting the last enable the board saw.  The sync-loss failsafe
    cannot help either, because the master is still sending edges.  So the
    unselected boards are addressed with the enable bit clear rather than left
    out.
    """
    master = make_master()
    master.open()
    master.scan()
    master.select([0x101, 0x102])
    master.set_enabled(0x101, True)
    master.set_target(0x101, 9.0)

    tabled = master._build_targets()
    assert set(tabled) == set(sc.ALL_IDS)
    assert tabled[0x101][1] is True
    assert tabled[0x102][1] is False          # selected, but not enabled
    assert tabled[0x118][1] is False          # not even selected
    master.close()


# ===========================================================================
# Transport semantics
# ===========================================================================

def test_no_serial_port_is_ever_opened():
    """A live campaign owns COM58 while this suite runs."""
    master = make_master()
    assert isinstance(master.link, sm.SimCanLink)
    assert master.link.port == "SIM"
    assert not hasattr(master.link, "_ser")
    master.open()
    assert master.link.is_open
    master.close()
    assert not master.link.is_open


def test_the_runtime_table_round_trips_through_protos_own_encoder():
    """Encoded by ``build_runtime_table``, decoded by the link, staged on the arm.

    Using ``proto``'s encoder rather than a private one means the ``0x80``
    marker, the slot mask and the 12-bit-plus-flags packing are exercised on
    every cycle, so a change on either side shows up as a decode failure instead
    of as a silent disagreement about what was commanded.
    """
    master = make_master()
    arm = master.arm
    wanted = {0x101: (1234, True), 0x105: (7, False), 0x118: (4095, True)}
    frames = P.build_runtime_table(wanted)
    assert all(data[0] & P.RUNTIME_TABLE_MARKER for _, data in frames)

    master.link.send_batch(frames)
    for base, (counts, enable) in wanted.items():
        node = arm.nodes[base]
        assert node.pending_target_counts == counts
        assert bool(node.pending_control & P.CONTROL_ENABLE) is enable
        assert node.active_target_counts == 0        # nothing promoted yet
    assert master.link.unmodelled_frames == 0

    master.link.send(*P.build_sync())
    for base, (counts, _) in wanted.items():
        assert arm.nodes[base].active_target_counts == counts


def test_scan_reports_the_variant_byte_and_the_cal_follows_it():
    """A TLE board sat at 0x114 during the 2026-08 bench session.

    ``Backend.scan`` builds each node's ``NodeCal`` with ``for_variant``, so the
    twin only has to answer honestly for the host to land on the right transfer
    function -- and choosing it from the id range instead would be a 10 % error
    in psi at the top of the range.
    """
    variants = dict(sc.NOMINAL_VARIANTS)
    variants[0x114] = P.VARIANT_TLE_DVP
    master = make_master(variants=variants)
    master.open()
    found = master.scan()
    master.close()

    assert len(found) == 24
    assert found[0x114].kind == P.VARIANT_NAMES[P.VARIANT_TLE_DVP]
    assert found[0x114].is_tle
    assert found[0x114].cal.counts_per_psi == pytest.approx(P.TLE_COUNTS_PER_PSI)
    assert found[0x114].version == sm.TLE_FIRMWARE_VERSION

    assert found[0x110].kind == P.VARIANT_NAMES[P.VARIANT_7MM]
    assert found[0x110].cal.counts_per_psi == pytest.approx(P.LEGACY_COUNTS_PER_PSI)
    assert found[0x110].version == sm.SEVEN_MM_FIRMWARE_VERSION


def test_an_absent_board_answers_nothing_anywhere():
    master = make_master(absent=(0x107,))
    master.open()
    found = master.scan()
    assert 0x107 not in found
    master.select(list(sc.ALL_IDS))
    master.set_enabled(0x101, True)
    master.set_target(0x101, 6.0)
    master.start_cycle()
    time.sleep(RUN_S)
    master.stop_cycle()
    snap = master.snapshot_nodes()
    master.close()

    assert 0x107 not in snap
    assert snap[0x101].replies > 0
    assert snap[0x101].consecutive_misses == 0


def test_reply_dropout_is_a_transport_fault_and_leaves_the_plant_alone():
    """Faults split by layer: the plant never sees a lost reply.

    The board goes on regulating to its promoted target; only the host's view of
    it goes dark.  Mixing the two layers is what makes a plant fault impossible
    to reproduce offline in :mod:`digital_twin.replay`, which has no transport.
    """
    master = make_master(faults={0x10A: sc.NodeFault(reply_dropout=1.0)})
    snap = _drive(master, {0x109: 10.0, 0x10A: 10.0})
    master.close()

    assert snap[0x10A].replies == 0
    assert snap[0x10A].misses >= master.stats.cycles - 1
    assert snap[0x109].replies > 0
    assert master.link.dropped_replies >= snap[0x10A].misses

    # The plant did the same thing on both boards regardless.
    truth = master.snapshot_arm()
    assert truth[0x10A]["true_psi"] == pytest.approx(truth[0x109]["true_psi"],
                                                    abs=0.5)


def test_a_closed_loop_through_the_shell_moves_the_pressure_the_host_sees():
    master = make_master()
    snap = _drive(master, {0x101: 10.0, 0x10A: 10.0}, run_s=1.0)
    master.close()

    for base in (0x101, 0x10A):
        assert 7.0 < snap[base].pressure_psi < 13.0
        assert snap[base].flags & P.STATUS_COMMAND_SEEN
    idle = snap[0x118]
    assert abs(idle.pressure_psi) < 1.0
    assert not idle.flags & P.STATUS_ENABLED


def test_stop_cycle_leaves_every_board_disabled_on_the_modelled_bus():
    """The safe-disable has to reach the boards, not just the host's registry.

    ``stop_cycle`` is gated on the thread and not on the run flag precisely
    because the cycle loop clears that flag itself when a transmit fails --
    skipping the safe-disable in that case would walk away from a bus where
    every board is still enabled and holding its last commanded pressure.
    """
    master = make_master()
    _drive(master, {0x101: 10.0, 0x10A: 10.0})
    truth = master.snapshot_arm()
    master.close()

    assert all(not row["enabled"] for row in truth.values())
    assert all(row["action"] == sc.ACTION_NONE
               for base, row in truth.items() if base in P.TLE_IDS)
    # And the air is still in the arm: a disable is not a vent.
    assert truth[0x101]["true_psi"] > 3.0
    assert truth[0x10A]["true_psi"] > 3.0


def test_a_silent_host_trips_the_tle_failsafe_and_never_the_seven_mm_one():
    """The safety asymmetry, observed through the shell rather than the arm.

    The cycle thread is stopped **without** ``stop_cycle``'s safe-disable, which
    is what a crashed host looks like from the boards' side: edges simply stop.
    The twin keeps its own clock through the link's idle advance, which is the
    only reason this is observable at all -- with the sync edge as the sole
    clock, the very silence that trips the failsafe would also stop the clock
    measuring it.
    """
    master = make_master()
    master.open()
    master.scan()
    master.select(list(sc.ALL_IDS))
    for base in (0x101, 0x10A):
        master.set_target(base, 10.0)
        master.set_enabled(base, True)
    master.start_cycle()
    time.sleep(0.3)

    master._running.clear()                  # the host dies; no safe-disable
    master._thread.join(timeout=1.0)
    master._thread = None
    time.sleep(0.8)                          # well past TLE_SYNC_TIMEOUT_S
    truth = master.snapshot_arm()
    tle, seven = master.arm.nodes[0x101], master.arm.nodes[0x10A]
    master.link.close()

    assert tle.failsafe_active is True
    assert tle.enabled is False
    assert tle.sync_loss_count == 1
    assert truth[0x101]["true_psi"] > 1.0    # a trap, not a vent

    assert seven.failsafe_active is False
    assert seven.enabled is True
    assert seven.sync_loss_count == 0
    assert seven.target_pa == pytest.approx(10.0 * PSI, rel=0.02)


def test_the_reply_carries_the_pressure_latched_at_the_edge():
    """One host cycle is one logical instant.

    The board latches its filtered reading in the sync handler, so a reply that
    arrives 3.35 ms later still describes the edge.  A twin that sampled at
    transmit time would hand the host twenty-four readings from twenty-four
    different instants and quietly break every cross-actuator fit.
    """
    master = make_master()
    master.open()
    master.scan()
    master.select(list(sc.ALL_IDS))
    node = master.arm.nodes[0x101]
    master.arm.advance_to(time.perf_counter())
    node.p_pa = 9.0 * PSI
    node._sample_adc()

    captured = {}
    master.set_cycle_observer(
        lambda t, targets, replies: captured.setdefault("first", dict(replies)))
    master.link.send_batch([P.build_sync()])
    latched = node.latched_counts
    node.p_pa = 25.0 * PSI                     # the plant moves after the edge
    node._sample_adc()
    time.sleep(0.05)
    master.close()

    assert node.compact_status().counts == latched
    assert node.cal.counts_to_psi(latched) == pytest.approx(9.0, abs=0.1)


def test_a_reply_already_due_when_physics_returns_is_heard_on_that_edge():
    """No second thread stands between a due reply and the host.

    The link is never opened, so no delivery thread exists, and the physics is
    made to outlast the whole 1.87-3.35 ms arbitration spread -- as one 6.67 ms
    advance of the fitted twin does.  Every reply is therefore due when the
    physics returns, and all 24 must reach the tap inside ``send_batch``, stamped
    with their arbitration times.  Before the fix they sat on the heap for a
    delivery thread that, in a live window, had to win the interpreter lock back
    from the thread that had just spent it on physics.
    """
    master = make_master()
    link, arm = master.link, master.arm
    heard = []
    link.add_tap(lambda t, can_id, data: heard.append((t, can_id)))
    real_advance = arm.advance_to

    def slow_advance(t):
        out = real_advance(t)
        time.sleep(0.006)
        return out

    arm.advance_to = slow_advance
    arm.advance_to(time.perf_counter())            # pin the origin
    t0 = time.perf_counter()
    link._batch_t0 = t0
    try:
        link._route(P.ID_BROADCAST, b"")
    finally:
        link._batch_t0 = None

    assert [can_id for _t, can_id in heard] == list(sc.ALL_IDS)
    stamps = np.array([t for t, _ in heard]) - t0
    np.testing.assert_allclose(stamps * 1000.0,
                               [sm.reply_latency_ms(b) for b in sc.ALL_IDS], atol=1e-9)
    assert link.rx_count == 24 and not link._pending


def test_a_reply_not_yet_due_is_fired_at_its_time_by_the_thread_that_holds_it():
    """Delivery never waits on a second thread's wake-up, and never fires early.

    Fast physics this time, so every reply is still in the future when the edge
    has been applied.  With no delivery thread in existence (the link is not
    opened), all 24 must still reach the tap before ``send_batch`` returns, each
    no earlier than its arbitration time.  Before the fix they waited on the
    heap for a delivery thread that ran a median 3.4-4.1 ms late once
    ``test_sim_core`` had run in the same interpreter.
    """
    master = make_master()
    link, arm = master.link, master.arm
    heard = []
    link.add_tap(lambda t, can_id, data: heard.append((t, time.perf_counter(), can_id)))
    arm.advance_to(time.perf_counter())
    t0 = time.perf_counter()
    link._batch_t0 = t0
    try:
        link._route(P.ID_BROADCAST, b"")
    finally:
        link._batch_t0 = None
    returned = time.perf_counter()

    assert [can_id for _t, _w, can_id in heard] == list(sc.ALL_IDS)
    for stamp, wall, _ in heard:
        assert wall >= stamp, "a reply was heard before its arbitration time"
    assert returned - t0 >= sm.reply_latency_ms(P.ACTUATOR_LAST) / 1000.0
    assert not link._pending


def test_a_reply_due_past_the_hold_is_left_for_the_delivery_thread():
    """What keeps a reply due after the receive window a miss, as on the metal."""
    master = make_master(first_ms=1.0, last_ms=9.0)
    link, arm = master.link, master.arm
    assert link.inline_hold_s == pytest.approx(min(sm.INLINE_HOLD_S,
                                                   0.82 / P.CYCLE_HZ))
    heard = []
    link.add_tap(lambda t, can_id, data: heard.append(can_id))
    arm.advance_to(time.perf_counter())
    link.send(*P.build_sync())
    early = {b for b in sc.ALL_IDS
             if sm.reply_latency_ms(b, first_ms=1.0, last_ms=9.0) / 1000.0
             <= link.inline_hold_s}
    assert set(heard) == early and 0 < len(early) < 24
    assert len(link._pending) == 24 - len(early)
    link.open()
    time.sleep(0.05)
    link.close()
    assert sorted(heard) == list(sc.ALL_IDS)


def test_the_delivery_thread_leaves_physics_to_the_edges_while_the_host_drives():
    """Only a quiet bus is kept in time by the link's own thread.

    The idle advance exists so a silent host still trips the TLE failsafe.  It
    used to run on every wake-up of the delivery thread, including the ones
    waiting to deliver a reply, under the condition replies are scheduled with.
    """
    master = make_master()
    arm = master.arm
    real_advance = arm.advance_to
    during_cycle = []

    def spy(t):
        if master.running and master.stats.cycles > 2:
            during_cycle.append(threading.current_thread().name)
        return real_advance(t)

    arm.advance_to = spy
    snap = _drive(master, {0x101: 8.0}, run_s=0.6)
    master.close()

    assert snap[0x101].replies > 0
    assert "sim-canlink" not in during_cycle, (
        f"{during_cycle.count('sim-canlink')} advances on the delivery thread while "
        f"the host was driving")
    assert "sync-master" in during_cycle


def test_history_decimates_under_the_lock():
    """A display can only show about one point per pixel column.

    Copying the full 60 s deque while holding the lock the cycle thread needs to
    publish cost a measured 2.8 Hz on the 24-board bus, so a caller asks for the
    points it can draw.
    """
    master = make_master()
    snap = _drive(master, {0x101: 6.0})
    full = master.history(0x101)
    small = master.history(0x101, max_points=10)
    master.close()

    # One row per REPLY, not per cycle: ``_publish`` charges a miss and moves on
    # rather than writing a placeholder, so a gap in the history is a gap on the
    # bus and a consumer must not read it as an even time base.
    assert len(full) == snap[0x101].replies
    assert 0 < len(full) <= master.stats.cycles
    assert 0 < len(small) <= len(full)
    assert len(small) <= len(full) // max(1, len(full) // 10) + 1
    assert all(len(row) == 3 for row in full)


def test_the_link_counts_what_it_did_not_model_rather_than_swallowing_it():
    """Extended telemetry and OTA are accepted and dropped, and say so.

    A test that expects a modelled path can then assert this stayed at zero,
    instead of discovering afterwards that its frames went nowhere.
    """
    master = make_master()
    master.open()
    assert master.link.unmodelled_frames == 0
    master.request_extended(0x101)             # base + 0x400, not modelled
    master.link.send(P.ID_BROADCAST, b"\x00\x01\x02")   # broadcast OTA data
    master.close()
    assert master.link.unmodelled_frames == 2


def test_a_handed_in_arm_and_simarm_kwargs_are_mutually_exclusive():
    arm = sc.SimArm(actuator=sc.placeholder_actuator())
    with pytest.raises(ValueError, match="mutually exclusive"):
        sm.SimMaster(arm=arm, absent=(0x101,))


def test_the_twins_own_readout_is_named_apart_from_the_shared_one():
    """``snapshot_arm`` is plant truth and ``snapshot_nodes`` is the host's view.

    Any code path that reads the first is a path that cannot run against the
    metal, so the two must not share a name -- and the host's view must never
    carry the unfiltered plant.
    """
    master = make_master()
    master.open()
    master.scan()
    master.arm.nodes[0x101].p_pa = 17.0 * PSI
    host = master.snapshot_nodes()
    truth = master.snapshot_arm()
    master.close()

    assert "p_pa" not in vars(host[0x101])
    assert host[0x101].pressure_psi == 0.0     # nothing has replied yet
    assert truth[0x101]["true_psi"] == pytest.approx(17.0)
    assert set(truth) == set(host)
