r"""The protocol, as assertions.  ``reference/sim_core.md`` section 7 is the source.

The RS485 twin's design note is blunt that its sim/real protocol "is not a
document -- it is the set of assertions in ``test_sim_core.py``".  So the C1-C5
clock set and the P1-P5 plant seam are ported here unchanged in meaning, with
only the node grid renumbered from 5 ms to this arm's 4 ms, and the
firmware-semantics group is rewritten around what ``firmware/tle/`` and
``firmware/legacy/`` actually do.

Nothing here touches hardware: no serial port, no CAN, no socket, no camera.
The default arm is built on the real ``mjcf_generator`` scene and a fresh
``actuator_model``, because this file is the integration seam and a break in
either of those should be loud.  The tests that pin the **seam's own
arithmetic** -- P1, P2 and P5 -- hand in a controlled stand-in instead, since a
leak subtraction measured through a fitted net measures the net.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from digital_twin import sim_core as sc

PSI = sc.PA_PER_PSI


def _is_tle_vector(variants=None):
    variants = sc.NOMINAL_VARIANTS if variants is None else variants
    return [variants.get(b, 0) == 0x02 for b in sc.ALL_IDS]


#: A body the generated scene is known to carry, for the concurrent reader in
#: C5.  Named here rather than inline so a rename in ``mjcf_generator`` shows up
#: as one edit and not as a puzzling KeyError in a threading test.
READER_BODY = "canarm_seg2_link"


def make_arm(*, variants=None, **kwargs):
    """One arm on the real generated scene and the real actuator model.

    Both are somebody else's modules, and depending on them here is deliberate:
    this suite is the integration seam, so a change in either that breaks
    :class:`~digital_twin.sim_core.SimArm`'s bindings should fail loudly rather
    than be hidden behind a stand-in.  The tests that pin the **seam's own
    arithmetic** hand in a controlled actuator instead, because measuring a leak
    subtraction through a fitted net measures the net.
    """
    return sc.SimArm(variants=variants, **kwargs)


# ===========================================================================
# C1-C5 -- the clock.  reference/sim_core.md section 7.1, ported verbatim.
# ===========================================================================

def test_c1_accumulator_conserves_sub_quantum_hops():
    """1000 hops of 1.5 ms consume exactly 1500 quanta, not 1000.

    Per-hop flooring would lose a third of the wall time, and it would lose it
    silently -- every trajectory still looks plausible, just slow.
    """
    arm = make_arm()
    arm.advance_to(0.0)
    for k in range(1, 1001):
        arm.advance_to(k * 0.0015)
    assert arm.quanta_done == 1500
    assert arm.sim_now == pytest.approx(1.5, abs=1e-9)


def test_c2_stale_target_is_a_noop():
    """A target behind the furthest one seen is a no-op; so is one equal to it.

    This is what makes racing advancers order-independent, so it is asserted on
    both the strictly-behind and the exactly-equal case.
    """
    arm = make_arm()
    arm.advance_to(10.0)
    arm.advance_to(10.05)
    consumed = arm.quanta_done
    arm.advance_to(10.05)
    assert arm.quanta_done == consumed
    arm.advance_to(10.02)
    assert arm.quanta_done == consumed
    arm.advance_to(9.0)
    assert arm.quanta_done == consumed


def test_c3_first_advance_pins_the_origin_and_consumes_nothing():
    arm = make_arm()
    assert arm.quanta_done == 0
    arm.advance_to(1234.5)
    assert arm.quanta_done == 0
    assert arm.sim_now == 1234.5


def test_c4_node_logic_runs_on_whole_quanta():
    """Passes fall at quanta 0, 4, 8, ... and the grid is a whole 4 ms."""
    arm = make_arm()
    arm.advance_to(0.0)
    arm.advance_to(0.025)
    assert arm.quanta_done == 25
    assert arm.node_passes == 7          # quanta 0, 4, 8, 12, 16, 20, 24
    arm.advance_to(0.028)
    assert arm.node_passes == 7          # quanta 25..27 are off the grid
    arm.advance_to(0.032)
    # Passes fall at the START of a quantum, so 32 consumed quanta have run the
    # passes at indices 0..28 and not yet the one at 32.
    assert arm.node_passes == 8
    arm.advance_to(0.033)
    assert arm.node_passes == 9

    assert sc.NODE_LOGIC_EVERY * arm.model.opt.timestep == pytest.approx(0.004)
    assert arm.node_grid_s == pytest.approx(0.004)


def test_c5_two_racing_advancers_are_bit_identical_to_one_serial_advancer():
    """Determinism, not exclusion, is what makes any-thread advancing safe.

    Two advancer threads on incommensurate hops (3.1 ms and 7.3 ms) plus a
    reader thread hammering ``q()`` and ``body_pose()`` must land on **exactly**
    the state one serial advancer reaches -- ``np.array_equal``, not a
    tolerance.  A tolerance here would pass on a twin whose node passes had
    drifted by one, which is the failure this test exists to catch.
    """
    t_end = 0.8

    def drive(arm):
        arm.advance_to(0.0)
        arm.stage_targets({b: (arm.nodes[b].cal.psi_to_counts(12.0), True)
                           for b in arm.nodes})
        arm.sync_edge(0.0)

    serial = make_arm()
    drive(serial)
    serial.advance_to(t_end)

    racing = make_arm()
    drive(racing)
    stop = threading.Event()

    def advancer(hop):
        t = 0.0
        while t < t_end:
            t = min(t + hop, t_end)
            racing.advance_to(t)

    def reader():
        while not stop.is_set():
            racing.q()
            racing.body_pose(READER_BODY)

    threads = [threading.Thread(target=advancer, args=(0.0031,)),
               threading.Thread(target=advancer, args=(0.0073,)),
               threading.Thread(target=reader)]
    for th in threads[:2]:
        th.start()
    threads[2].start()
    for th in threads[:2]:
        th.join()
    stop.set()
    threads[2].join()

    assert serial.quanta_done == 800
    assert racing.quanta_done == 800
    assert np.array_equal(serial.data.qpos, racing.data.qpos)
    assert np.array_equal(serial.data.qvel, racing.data.qvel)
    for base in serial.nodes:
        a, b = serial.nodes[base], racing.nodes[base]
        assert a.p_pa == b.p_pa
        assert a.raw == b.raw
        assert a.action == b.action
        assert a.failsafe_active == b.failsafe_active


# ===========================================================================
# P1-P5 -- the plant seam.  reference/sim_core.md section 7.2.
# ===========================================================================

def test_p1_pressure_is_clamped_nonnegative_every_substep_and_stays_there():
    """A monster leak cannot drive ``p_pa`` negative, then or on any later step."""
    actuator = sc.placeholder_actuator(is_tle=_is_tle_vector(),
                                       fill_gain=0.0, vent_gain=0.0,
                                       leak_pa_s=1.0e6)
    arm = make_arm(actuator=actuator)
    for node in arm.nodes.values():
        node.p_pa = 5.0 * PSI
    arm.advance_to(0.0)
    arm.advance_to(0.1)
    assert all(n.p_pa == 0.0 for n in arm.nodes.values())
    arm.advance_to(0.3)
    assert all(n.p_pa == 0.0 for n in arm.nodes.values())


def test_p2_leak_seam_arithmetic_is_exact_to_the_pascal():
    """With the net zeroed, the closed branch is exactly ``-leak``.

    And over 2.0 s the plant integrates **1.996 s**, not 2.0: passes fall at
    quanta 0, 4, ..., 1996 and the pass at quantum 0 moves nothing.  Any leak
    arithmetic validated against this twin has to use ``(N-1) * node_grid``.
    """
    leak = 500.0
    actuator = sc.placeholder_actuator(is_tle=_is_tle_vector(),
                                       fill_gain=0.0, vent_gain=0.0,
                                       leak_pa_s=leak)
    arm = make_arm(actuator=actuator)
    start = 10_000.0
    for node in arm.nodes.values():
        node.p_pa = start
    arm.advance_to(0.0)
    arm.advance_to(2.0)

    integrated_s = (arm.node_passes - 1) * arm.node_grid_s
    assert integrated_s == pytest.approx(1.996, abs=1e-12)
    for node in arm.nodes.values():
        assert node.p_pa == pytest.approx(start - leak * integrated_s, abs=1e-6)


def test_p3_leak_fault_overrides_the_per_node_scalar():
    actuator = sc.placeholder_actuator(is_tle=_is_tle_vector(), leak_pa_s=700.0)
    arm = make_arm(actuator=actuator,
                   faults={0x109: sc.NodeFault(leak=True)})
    assert arm.nodes[0x10A].leak_pa_s == pytest.approx(700.0)
    assert arm.nodes[0x109].leak_pa_s == pytest.approx(sc.LEAK_FAULT_PA_S)

    louder = make_arm(actuator=actuator, faults={0x109: sc.NodeFault(leak=True)},
                      leak_fault_pa_s=4321.0)
    assert louder.nodes[0x109].leak_pa_s == pytest.approx(4321.0)


def test_p4_adc_coupling_is_pinned_and_the_sampler_is_untouched():
    """``true_psi`` is the whole coupling; the sampler below it is the parent's.

    Asserted on both populations, because they differ in every term: the TLE
    board's raw field is 16-bit and reaches the wire shifted right by four,
    while the 7 mm board's is the 12-bit field itself.
    """
    quiet_tle = sc.AdcSpec(**{**vars(sc.TLE_ADC), "noise_sd_counts": 0.0})
    quiet_7mm = sc.AdcSpec(**{**vars(sc.SEVEN_MM_ADC), "noise_sd_counts": 0.0})
    arm = make_arm(tle_adc=quiet_tle, seven_mm_adc=quiet_7mm)

    for base, spec in ((0x101, quiet_tle), (0x10A, quiet_7mm)):
        node = arm.nodes[base]
        node.p_pa = 12.0 * PSI
        node._sample_adc()
        expected = round(spec.zero_counts + 12.0 * spec.counts_per_psi)
        assert node.raw == expected
        # First sample primes the EMA, so filtered == raw exactly.
        assert node.filtered_raw == float(expected)
        # The relation is pinned exactly; the psi it lands on is one ADC
        # quantisation away from 12.0, and nothing here may hide that.
        assert node.pressure_pa == pytest.approx(
            (expected - spec.zero_counts) / spec.counts_per_psi * PSI, rel=1e-12)
        half_count_pa = 0.5 / spec.counts_per_psi * PSI
        assert abs(node.pressure_pa - 12.0 * PSI) <= half_count_pa
        assert node._wire_counts() == expected >> spec.wire_shift

    assert arm.nodes[0x101].adc.wire_shift == 4
    assert arm.nodes[0x10A].adc.wire_shift == 0


def test_p5_stuck_vent_blocks_the_flow_and_leaves_the_status_honest():
    """The firmware does not know its exhaust is blocked, so it must still say
    it is venting -- and ``outlet_open_s`` must keep counting the energised coil.

    ``stuck_vent`` is adapted rather than ported: the reference flipped the flow
    model's valve input, which Departure 1 removed from this arm's model, so
    here it blocks the outward half of ``dp`` instead.
    """
    actuator = sc.placeholder_actuator(is_tle=_is_tle_vector(), leak_pa_s=0.0)
    arm = make_arm(actuator=actuator, faults={0x10A: sc.NodeFault(stuck_vent=True)})
    healthy, stuck = arm.nodes[0x109], arm.nodes[0x10A]
    for node in (healthy, stuck):
        node.p_pa = 20.0 * PSI

    arm.advance_to(0.0)
    arm.stage_targets({0x109: (healthy.cal.psi_to_counts(0.0), True),
                       0x10A: (stuck.cal.psi_to_counts(0.0), True)})
    arm.sync_edge(0.0)
    arm.advance_to(0.5)

    assert healthy.p_pa < 20.0 * PSI - 1000.0
    assert stuck.p_pa == pytest.approx(20.0 * PSI, abs=1e-6)
    assert stuck.action == sc.ACTION_OUTLET
    assert stuck.outlet_open_s > 0.4
    assert stuck.air_out_psi == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# The node grid -- CONTRACT.md section 4's "integer count of quanta"
# ===========================================================================

def test_node_logic_period_is_a_whole_count_of_both_firmware_periods():
    """4 ms is the smallest grid on which neither loop is truncated.

    750 Hz and 1000 Hz have a 4 ms common period, so a 4 ms node pass carries
    exactly three TLE PID steps and four 7 mm bang-bang steps.
    """
    counts = sc.sub_step_counts(sc.NODE_LOGIC_EVERY * 0.001)
    assert counts == {sc.TLE_PID_HZ: 3, sc.SEVEN_MM_TICK_HZ: 4}
    assert sc.NODE_LOGIC_EVERY * 0.001 * sc.TLE_PID_HZ == pytest.approx(3.0)
    assert sc.NODE_LOGIC_EVERY * 0.001 * sc.SEVEN_MM_TICK_HZ == pytest.approx(4.0)

    for smaller in (1, 2, 3):
        with pytest.raises(ValueError, match="whole number"):
            sc.sub_step_counts(smaller * 0.001)
    # The reference's own 5 ms grid is exactly the trap: it is a whole number of
    # 7 mm periods and 3.75 of a TLE PID period.
    with pytest.raises(ValueError, match="750 Hz"):
        sc.sub_step_counts(0.005)
    assert sc.sub_step_counts(0.005, strict=False)[sc.TLE_PID_HZ] == 4


def test_simarm_refuses_a_fractional_node_grid_and_can_be_told_not_to():
    with pytest.raises(ValueError, match="whole number"):
        make_arm(node_logic_every=5)
    arm = make_arm(node_logic_every=5, strict_node_grid=False)
    assert arm.node_grid_s == pytest.approx(0.005)


def test_timestep_is_read_back_from_the_compiled_model():
    arm = make_arm()
    assert arm.dt == float(arm.model.opt.timestep) == pytest.approx(0.001)
    faster = make_arm(timestep_s=1.0 / 3000.0, node_logic_every=12)
    assert faster.dt == pytest.approx(1.0 / 3000.0)
    assert faster.node_grid_s == pytest.approx(0.004)
    with pytest.raises(ValueError, match="positive"):
        make_arm(timestep_s=0.0)


# ===========================================================================
# Firmware semantics -- read out of firmware/tle/ and firmware/legacy/
# ===========================================================================

def test_target_is_promoted_at_the_sync_edge_and_not_before():
    """Boards stage what the table carries; the DLC-0 edge promotes it."""
    arm = make_arm()
    node = arm.nodes[0x101]
    counts = node.cal.psi_to_counts(15.0)

    arm.stage_targets({0x101: (counts, True)})
    assert node.pending_target_counts == counts
    assert node.active_target_counts == 0
    assert node.enabled is False

    arm.sync_edge(0.0)
    assert node.active_target_counts == counts
    assert node.enabled is True
    # The wire carries counts, so the promoted target is 15 psi rounded onto
    # this board's own calibration and not 15 psi exactly.
    assert node.target_pa == pytest.approx(
        node.cal.counts_to_psi(counts) * PSI, rel=1e-12)
    assert abs(node.target_pa - 15.0 * PSI) <= PSI / node.cal.counts_per_psi

    # A second table, not yet promoted, must not disturb the live target.
    arm.stage_targets({0x101: (node.cal.psi_to_counts(3.0), True)})
    assert node.active_target_counts == counts
    arm.sync_edge(0.01)
    assert node.active_target_counts == node.cal.psi_to_counts(3.0)


def test_the_sync_edge_latches_the_pressure_the_reply_will_carry():
    """The reported value belongs to the sync instant, not to transmit time."""
    arm = make_arm()
    node = arm.nodes[0x101]
    arm.advance_to(0.0)
    node.p_pa = 9.0 * PSI
    node._sample_adc()
    arm.sync_edge(0.0)
    latched = node.latched_counts

    node.p_pa = 25.0 * PSI          # the plant moves after the edge
    node._sample_adc()
    assert node.compact_status().counts == latched
    arm.sync_edge(0.01)
    assert node.compact_status().counts != latched


def test_tle_failsafe_fires_at_500_ms_and_the_seven_mm_one_never_does():
    """The asymmetry is a safety property, so it is asserted from both sides.

    ``TLE_LEGACY_SYNC_TIMEOUT_US`` is 500 ms; the legacy tree has no equivalent,
    which this test pins by running a 7 mm board three seconds past the point a
    TLE board would have dropped out.
    """
    arm = make_arm()
    tle, seven = arm.nodes[0x101], arm.nodes[0x10A]
    arm.advance_to(0.0)
    arm.stage_targets({b: (arm.nodes[b].cal.psi_to_counts(10.0), True)
                       for b in (0x101, 0x10A)})
    arm.sync_edge(0.0)

    arm.advance_to(0.4)
    assert tle.failsafe_active is False
    assert tle.enabled is True

    arm.advance_to(0.52)
    assert tle.failsafe_active is True
    assert tle.enabled is False
    assert tle.action == sc.ACTION_NONE
    assert tle.current_code == 0.0
    assert tle.sync_loss_count == 1

    arm.advance_to(3.5)
    assert seven.failsafe_active is False
    assert seven.enabled is True
    assert seven.sync_loss_count == 0
    assert seven.target_pa == pytest.approx(10.0 * PSI, rel=1e-3)


def test_the_tle_failsafe_clears_the_pending_control_byte_too():
    """A stale table replayed after the trap must not re-enable the board.

    ``tle_can_legacy_service`` clears ``s_pending_control`` as well as
    ``s_active_control``, so recovery costs the host a fresh table frame.  A
    twin that cleared only the active byte would recover on the next edge and
    hide a real operational cost.
    """
    arm = make_arm()
    tle = arm.nodes[0x101]
    arm.advance_to(0.0)
    arm.stage_targets({0x101: (tle.cal.psi_to_counts(10.0), True)})
    arm.sync_edge(0.0)
    arm.advance_to(0.6)
    assert tle.failsafe_active is True

    arm.sync_edge(0.6)                       # an edge, but no fresh table
    assert tle.enabled is False
    arm.stage_targets({0x101: (tle.cal.psi_to_counts(10.0), True)})
    arm.sync_edge(0.61)
    assert tle.enabled is True
    assert tle.failsafe_active is False


def test_a_seven_mm_board_freezes_its_valves_when_the_enable_bit_is_dropped():
    """``main.c``'s disable branch is a bare ``continue`` above the GPIO writes.

    So dropping the enable bit stops updating the solenoids rather than
    de-energising them.  Reproduced, not fixed: with no link-loss timeout either,
    this is how a 7 mm board ends up inflating with nobody watching it.
    """
    arm = make_arm()
    node = arm.nodes[0x10A]
    arm.advance_to(0.0)
    arm.stage_targets({0x10A: (node.cal.psi_to_counts(15.0), True)})
    arm.sync_edge(0.0)
    arm.advance_to(0.05)
    assert node.action == sc.ACTION_INLET

    arm.stage_targets({0x10A: (node.cal.psi_to_counts(15.0), False)})
    arm.sync_edge(0.05)
    arm.advance_to(0.10)
    assert node.enabled is False
    assert node.action == sc.ACTION_INLET             # frozen, not shut
    assert node.inlet_open_s > 0.09

    # The host's explicit stop is the path that does shut them.
    node.all_off()
    assert node.action == sc.ACTION_NONE


def test_a_tle_board_drops_its_outputs_when_the_enable_bit_is_dropped():
    """The other half of the same asymmetry: ``set_enabled(false)`` on a TLE
    board forces every output off on a path that does not wait for the loop."""
    arm = make_arm()
    node = arm.nodes[0x101]
    arm.advance_to(0.0)
    arm.stage_targets({0x101: (node.cal.psi_to_counts(15.0), True)})
    arm.sync_edge(0.0)
    arm.advance_to(0.05)
    assert node.action == sc.ACTION_INLET

    arm.stage_targets({0x101: (node.cal.psi_to_counts(15.0), False)})
    arm.sync_edge(0.05)
    assert node.action == sc.ACTION_NONE
    assert node.current_code == 0.0


def test_the_population_is_read_from_the_variant_byte_not_the_id_range():
    """A TLE board sat at 0x114 during the 2026-08 bench session.

    Reading it off the address would command and display it on the 7 mm scale,
    a 10 % error in psi at the top of the range.
    """
    variants = dict(sc.NOMINAL_VARIANTS)
    variants[0x114] = 0x02
    variants[0x101] = 0x00
    arm = make_arm(variants=variants)

    assert isinstance(arm.nodes[0x114], sc.TleNode)
    assert arm.nodes[0x114].sync_timeout_s == sc.TLE_SYNC_TIMEOUT_S
    assert arm.nodes[0x114].cal.counts_per_psi == pytest.approx(P_TLE_COUNTS)

    assert isinstance(arm.nodes[0x101], sc.SevenMmNode)
    assert arm.nodes[0x101].sync_timeout_s is None
    assert arm.nodes[0x101].cal.counts_per_psi == pytest.approx(P_LEGACY_COUNTS)


P_TLE_COUNTS = 60.78125
P_LEGACY_COUNTS = 56.14


def test_the_seven_mm_hysteresis_steps_with_the_target_in_adc_counts():
    """5 to 8 counts, i.e. 614 to 982 Pa -- not the RS485 arm's 2000 Pa.

    ``blend_width_pa``'s 7 mm default in CONTRACT.md section 2 is the reference
    arm's ``MARGIN_PA``; these boards regulate two to three times tighter, and
    the trainer wants that number from here rather than from the contract.
    """
    tuning = sc.SevenMmTuning()
    assert tuning.deadband_counts(999.0) == 5.0
    assert tuning.deadband_counts(1000.0) == 6.0
    assert tuning.deadband_counts(1499.0) == 6.0
    assert tuning.deadband_counts(1500.0) == 7.0
    assert tuning.deadband_counts(1999.0) == 7.0
    assert tuning.deadband_counts(2000.0) == 8.0

    node = sc.make_node(0x10A, 0x00, adc=sc.SEVEN_MM_ADC)
    node.active_target_counts = 0
    assert node.deadband_pa() == pytest.approx(5.0 / P_LEGACY_COUNTS * PSI, rel=1e-9)
    assert node.deadband_pa() == pytest.approx(614.0, abs=1.0)
    node.active_target_counts = 2500
    assert node.deadband_pa() == pytest.approx(982.0, abs=1.0)


def test_the_tle_current_budget_reserves_four_codes_for_the_hardware_dither():
    """120 codes of headroom, four of them the dither overlay's, 116 for the DC.

    The Clippard DVP is rated 190 mA continuous.  Full scale is code 127 at
    200 mA, so the firmware's hard ceiling is 120 (189 mA) -- but the TLE92464's
    dither generator adds its triangle *on top of* the setpoint register, so a
    DC code of 120 with the default overlay would peak around 195 mA, above the
    very limit the ceiling exists to enforce.
    """
    t = sc.TleTuning()
    assert t.safe_max_code == 120
    assert t.dither_peak_max_code == 4
    assert t.host_max_code == 116
    assert sc.TleTuning.code_to_ma(116) == pytest.approx(182.7, abs=0.1)
    assert sc.TleTuning.code_to_ma(120) == pytest.approx(189.0, abs=0.1)
    assert sc.TleTuning.code_to_ma(127) == pytest.approx(200.0, abs=0.1)

    node = sc.make_node(0x101, 0x02, adc=sc.TLE_ADC,
                        tuning=sc.TleTuning(max_code=127))
    node.active_control = 0x01
    node.enabled = True
    node.active_target_counts = node.cal.psi_to_counts(28.0)
    for _ in range(400):
        node.regulate()
    assert node.current_code <= t.host_max_code
    assert node.current_ma <= 182.8


def test_the_tle_slew_limit_bounds_the_code_change_per_update():
    """``dvp_slew_code`` = 4 codes per current update, and it bounds every step
    including the first one out of the crack code."""
    t = sc.TleTuning()
    node = sc.make_node(0x101, 0x02, adc=sc.TLE_ADC)
    node.enabled = True
    node.active_control = 0x01
    node.active_target_counts = node.cal.psi_to_counts(25.0)
    previous = 0.0
    seen_inlet = False
    for _ in range(60):
        node.regulate()
        if node.action == sc.ACTION_INLET:
            if seen_inlet:
                assert abs(node.current_code - previous) <= t.slew_code + 1e-9
            seen_inlet = True
            previous = node.current_code
    assert seen_inlet
    assert node.current_code >= t.open_code


def test_the_tle_split_range_has_a_coast_band_between_inlet_and_outlet():
    """Between the output dead zone and ``-vent_threshold_pm`` neither valve
    drives; the plant's own leak does the venting.  Removing that band is what
    put an inlet/outlet relay limit cycle at low setpoints on the bench."""
    node = sc.make_node(0x101, 0x02, adc=sc.TLE_ADC)
    t = node.tuning

    node._prepare_output(1.0)
    assert node.action == sc.ACTION_INLET
    node._prepare_output(-(t.vent_threshold_pm - 1.0))
    assert node.action == sc.ACTION_NONE
    assert node.current_code == 0.0
    node._prepare_output(-(t.vent_threshold_pm + 1.0))
    assert node.action == sc.ACTION_OUTLET
    assert node.current_code >= t.outlet_open_code


def test_the_step_order_is_integrate_sample_failsafe_regulate():
    """The regulator must act on the pressure the SENSOR reported.

    Asserted by giving the node a ridiculous ADC offset: the plant is empty and
    the target is zero, so a regulator reading truth would sit in its dead zone,
    while one reading the sensor sees a full bladder and vents.
    """
    liar = sc.AdcSpec(**{**vars(sc.SEVEN_MM_ADC),
                         "zero_counts": sc.SEVEN_MM_ADC.zero_counts + 1500.0,
                         "noise_sd_counts": 0.0})
    node = sc.make_node(0x10A, 0x00, adc=liar)
    node.enabled = True
    node.active_control = 0x01
    node.active_target_counts = 0
    node.p_pa = 0.0
    node.step(0.004, 0.0, n_sub=4)
    assert node.action == sc.ACTION_OUTLET
    assert node.p_pa == 0.0             # clamped, and the plant never went below


def test_absent_nodes_are_absent_everywhere_and_produce_no_force():
    arm = make_arm(absent=(0x105,))
    assert 0x105 not in arm.nodes
    assert arm.psi_of(0x105) == 0.0
    assert arm.pressures_pa()[0x105 - sc.ACTUATOR_FIRST] == 0.0

    arm.advance_to(0.0)
    arm.nodes[0x104].p_pa = 20.0 * PSI
    arm.advance_to(0.01)
    aid = arm._act_ids[0x105 - sc.ACTUATOR_FIRST]
    neighbour = arm._act_ids[0x104 - sc.ACTUATOR_FIRST]
    assert arm.data.ctrl[aid] == 0.0
    assert arm.data.ctrl[neighbour] < 0.0


def test_muscle_force_is_written_every_quantum_and_is_pull_only():
    """The force law runs at the physics grid, not at the node grid.

    A force held across four ``mj_step`` calls is a fourfold zero-order-hold
    error in the one channel the arm is driven through.  One board is
    pressurised rather than all twenty-four, because a uniformly pressurised
    arm is in equilibrium -- every antagonist balances -- and would hold its
    tendon lengths whatever the update rate, which is exactly the case this
    test must not accidentally measure.
    """
    arm = make_arm()
    arm.advance_to(0.0)
    hot = arm.nodes[0x101]
    hot.p_pa = 18.0 * PSI
    # From the SECOND quantum on: MuJoCo's integrator advances velocity before
    # position, so with the arm starting at rest ``qpos`` is still exactly its
    # qpos0 after one step and the tendon lengths have not moved yet.
    arm.advance_to(0.002)
    first = np.array(arm.data.ctrl[arm._act_ids])
    arm.advance_to(0.003)                  # still inside the same node pass
    second = np.array(arm.data.ctrl[arm._act_ids])
    assert arm.node_passes == 1
    assert not np.array_equal(first, second)
    assert (first <= 0.0).all() and (second <= 0.0).all()
    assert (first >= -4000.0).all()        # the pull-only clip, CONTRACT.md s.2
    assert first[hot.index] < 0.0
    assert first[arm.nodes[0x118].index] == 0.0


def test_tendon_damping_follows_pressure_and_is_refreshed_at_the_node_grid():
    arm = make_arm()
    base = np.array(arm._tendon_damping_base)
    arm.advance_to(0.0)
    hot, cold = arm.nodes[0x101], arm.nodes[0x102]
    hot.p_pa = 20.0 * PSI
    cold.p_pa = 0.0
    arm.advance_to(0.02)

    # Asked of the model rather than pinned to a formula: the damping law is a
    # checkpoint's business (CONTRACT.md section 2 gives it a slack gate), and a
    # test that restated it here would pass only for today's checkpoint.
    dlen = arm.data.ten_length[arm._ten_ids] - arm._ten_len0
    predicted = arm.actuator.tendon_damping_n_s_m(
        arm.pressures_pa(), base=base, l_m=arm.actuator.l0_per_act + dlen)
    live = arm.model.tendon_damping[arm._ten_ids]
    assert np.allclose(live, predicted, rtol=1e-9, atol=1e-12)
    assert live[hot.index] > live[cold.index]


def test_all_off_traps_the_air_it_finds_rather_than_venting_it():
    """E-stop is a valve state, not a vent.  Venting is a separate act.

    Run on the controlled stand-in, whose closed branch is exactly zero, so that
    "trapped" is an equality rather than a tolerance.  On a fitted checkpoint it
    would not be: the reference records the bench fit **inflating** a settled
    25.15 psi rack to 27.36 psi over 10 s of ALL_OFF while the shipped synthetic
    one drifted down.  So a rollout that wants an empty arm has to vent past the
    closed branch's own fixed point BEFORE shutting the valves, and no test may
    pin the sign of that drift -- ask the model what it predicts instead.
    """
    actuator = sc.placeholder_actuator(is_tle=_is_tle_vector(), leak_pa_s=0.0)
    arm = make_arm(actuator=actuator)
    arm.advance_to(0.0)
    arm.stage_targets({b: (arm.nodes[b].cal.psi_to_counts(12.0), True)
                       for b in arm.nodes})
    arm.sync_edge(0.0)
    arm.advance_to(0.4)
    hot = {b: n.p_pa for b, n in arm.nodes.items()}
    assert min(hot.values()) > 5.0 * PSI
    assert any(n.action == sc.ACTION_INLET for n in arm.nodes.values())

    arm.all_off()
    assert all(n.action == sc.ACTION_NONE for n in arm.nodes.values())
    assert all(n.enabled is False for n in arm.nodes.values())
    arm.advance_to(1.4)
    # A whole second later, and nothing has re-opened a valve on its own.
    assert all(n.action == sc.ACTION_NONE for n in arm.nodes.values())
    for b, n in arm.nodes.items():
        assert n.p_pa == pytest.approx(hot[b], abs=1e-6)


def test_batched_and_scalar_flow_paths_agree():
    """``batched_actuator`` must be a speed choice and never a physics choice."""
    kwargs = dict(actuator=sc.placeholder_actuator(is_tle=_is_tle_vector()))
    scalar = make_arm(**kwargs)
    batched = make_arm(actuator=sc.placeholder_actuator(is_tle=_is_tle_vector()),
                       batched_actuator=True)
    for arm in (scalar, batched):
        arm.advance_to(0.0)
        arm.stage_targets({b: (arm.nodes[b].cal.psi_to_counts(9.0), True)
                           for b in arm.nodes})
        arm.sync_edge(0.0)
        arm.advance_to(0.2)
    assert np.allclose(scalar.pressures_pa(), batched.pressures_pa(), atol=1e-9)


def test_air_accounting_charges_realized_change_on_an_open_valve_only():
    """Charged on realized change and only on an open valve.

    The stand-in actuator is used because the figure being asserted is a *rate*
    -- how much air a 0.3 s fill costs -- and on an untrained checkpoint that
    rate is whatever the random weights happen to give.
    """
    arm = make_arm(actuator=sc.placeholder_actuator(is_tle=_is_tle_vector()))
    node = arm.nodes[0x10A]
    arm.advance_to(0.0)
    arm.stage_targets({0x10A: (node.cal.psi_to_counts(12.0), True)})
    arm.sync_edge(0.0)
    arm.advance_to(0.3)

    totals = arm.air_totals()
    i = node.index
    assert totals["air_in_psi"][i] > 5.0
    assert totals["inlet_open_s"][i] > 0.2
    assert totals["valve_switches"][i] >= 1
    # A node that never opened a valve is charged nothing at all.
    idle = arm.nodes[0x118].index
    assert totals["air_in_psi"][idle] == 0.0
    assert totals["inlet_open_s"][idle] == 0.0


def test_a_scene_without_the_pam_actuators_is_refused_by_name():
    bare = ('<mujoco><worldbody><body name="b">'
            '<joint type="hinge" axis="0 1 0"/>'
            '<geom type="sphere" size="0.05"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match=r"pam_1"):
        sc.SimArm(xml=bare, actuator=sc.placeholder_actuator())


def test_the_default_arm_builds_against_the_real_generator_and_model():
    """No ``xml=`` and no ``actuator=``: the integration seam, asserted.

    ``SimArm`` binds by name, so this fails the moment
    ``digital_twin.mjcf_generator`` stops emitting ``pam_1``..``pam_24`` or
    ``digital_twin.actuator_model`` stops offering the section 2 surface -- which
    is the whole reason to assert it here rather than to hand in a stand-in.
    """
    arm = sc.SimArm()
    assert len(arm.nodes) == 24
    assert arm.dt == pytest.approx(0.001)
    assert len(arm._act_ids) == 24
    assert len(set(arm._ten_ids.tolist())) == 24
    assert (arm._ten_len0 > 0.0).all()
    arm.advance_to(0.0)
    arm.advance_to(0.02)
    assert arm.quanta_done == 20
    assert (arm.data.ctrl[arm._act_ids] <= 0.0).all()


def test_the_bring_up_stand_ins_still_build_a_working_arm():
    """The fallback path, kept alive on purpose.

    ``placeholder_xml`` and ``placeholder_actuator`` exist so this module can be
    exercised on a machine where the other two are absent or regressed.  A
    fallback nobody runs is a fallback that does not work, so it is run here --
    and the test says out loud that the geometry is not this arm.
    """
    arm = sc.SimArm(xml=sc.placeholder_xml(),
                    actuator=sc.placeholder_actuator(is_tle=_is_tle_vector()))
    assert len(arm.nodes) == 24
    arm.advance_to(0.0)
    arm.stage_targets({0x10A: (arm.nodes[0x10A].cal.psi_to_counts(12.0), True)})
    arm.sync_edge(0.0)
    arm.advance_to(0.5)
    assert 8.0 < arm.psi_of(0x10A) < 14.0
