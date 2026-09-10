r"""``SimArm`` -- MuJoCo, twenty-four firmware models, and exactly one mutex.

This module implements CONTRACT.md section 4.  Three layers, and the split is
the whole design:

* :class:`CanNode` models the **firmware** and nothing else.  It stages a
  target, promotes it at the sync edge, latches the pressure the reply will
  carry, samples an ADC, runs a control law and reports a status byte.  It
  carries a stand-in plant so that it is exercisable on its own.
* :class:`SimNode` replaces **only the plant** -- three members,
  :attr:`~SimNode.true_psi`, :attr:`~SimNode.leak_pa_s` and
  :meth:`~SimNode._integrate` -- so every firmware semantic below is literally
  the same code for the real-plant twin and for the standalone node.  That
  discipline is ported unchanged from ``reference/sim_core.md`` section 1.3,
  where it is what kept the RS485 twin's firmware behaviour from drifting away
  from its own node model.
* :class:`SimArm` owns the compiled model, its data, the twenty-four nodes and
  one reentrant lock, and exposes a grid-quantised :meth:`SimArm.advance_to`
  callable from any thread.

TWO NODE FLAVOURS, CHOSEN BY THE VARIANT BYTE, NEVER BY THE ID RANGE.
:class:`TleNode` is a TLE92464/DVP board (variant ``0x02``) and
:class:`SevenMmNode` a legacy 7 mm board (variant ``0x00``/``0x01``).  A TLE
board legitimately sat at ``0x114`` during the 2026-08 bench session, and
reading it on the 7 mm calibration is a 10 % error in psi at the top of the
range (``tlelib/proto.py::NodeCal.for_variant``).

THE FAILSAFE IS ASYMMETRIC AND THAT ASYMMETRY IS A SAFETY PROPERTY.  Read out
of the firmware rather than taken from the docs, 2026-09-10:

* ``firmware/tle/main/tle_can_legacy.c::tle_can_legacy_service`` drops a TLE
  board's outputs ``TLE_LEGACY_SYNC_TIMEOUT_US`` = 500 ms after the last sync
  edge, and clears *both* the active and the pending control byte, so a later
  sync edge cannot re-enable the board until the host sends a fresh table
  frame carrying the enable bit.
* ``firmware/legacy/main/`` has **no link-loss timeout at all**, which this
  module confirmed in the source rather than taking on trust.  The only
  time-based guard there is ``main.c::WATCHDOG_subroutine``, which after 20
  half-second ticks with no CAN traffic calls
  ``can_routine.c::CAN_recover_from_starvation`` -- and that function resets
  the MCP2515's interrupt and error state and nothing else.  It does not touch
  ``active_control_byte`` or ``active_target_pressure``.  A 7 mm board whose
  host stops therefore regulates to its last commanded pressure indefinitely.
* Worse, and also confirmed in the source: clearing a 7 mm board's enable bit
  does not de-energise its valves.  ``main.c`` line 517 reads
  ``if ((ctrl & 0x01) == 0) { vTaskDelay(1); continue; }`` and the three
  ``gpio_set_level(V_IN/V_OUT, ...)`` calls live only inside the branch below
  it, so a disable **freezes the solenoids in whatever state they were last
  driven to**.  A board that was inflating when the host dropped its enable
  bit goes on inflating.  Both behaviours are reproduced here rather than
  fixed, because a twin that quietly repairs them cannot show what a host
  stall does to this arm.

WHERE THE FIRMWARE'S REGULATOR DOES AND DOES NOT REACH THE PLANT.  Under
CONTRACT.md Departure 1 the flow model's input is the commanded error
``e = target - measured`` in Pa, not a valve state and not a coil current.  So
the regulator's *output* -- the DVP current code, the bang-bang valve state --
is modelled, accounted and reported, but it does not drive the flow; the target
promoted at the sync edge does.  The consequence bounds what any result from
this twin shows and is worth stating plainly: **retuning a firmware gain here
changes the telemetry and the air accounting and leaves the pressure trajectory
untouched.**  What the twin does capture is the part a host-side controller can
see -- when a target takes effect, what the reply carries, and what happens when
the boards stop hearing the host.

RS485 ORIGINAL: ``C:\RUNZE_SRC\RS485_VEMA\UMArm_SIM\sim_core.py``, read in
detail at ``reference/sim_core.md``.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - whichever of the two paths is live is exercised
    from TLE_PCB.tlelib import proto as P
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from TLE_PCB.tlelib import proto as P


# ---------------------------------------------------------------------------
# Units and bus layout
# ---------------------------------------------------------------------------

#: Pa per psi.  CONTRACT.md section 1; psi appears only at a human boundary.
PA_PER_PSI = 6894.757

N_NODES = P.ACTUATOR_COUNT                      # 24
ACTUATOR_FIRST = P.ACTUATOR_FIRST               # 0x101
ALL_IDS = tuple(P.ALL_IDS)

#: The shipped population layout: eight TLE/DVP boards then sixteen 7 mm ones.
#: This is a **default for a bus nobody has scanned**, not a measurement.  Under
#: CONTRACT.md Departure 2 the population is read from the variant byte a board
#: answered; :class:`digital_twin.sim_master.SimMaster` does exactly that, and
#: this table only decides what the modelled boards *report*.
NOMINAL_VARIANTS = {
    **{b: P.VARIANT_TLE_DVP for b in P.TLE_IDS},
    **{b: P.VARIANT_7MM for b in P.SEVEN_MM_IDS},
}

#: Regulated supply, psi gauge.  The operator runs this arm's manifold at 40 psi
#: -- ``tlelib/proto.py`` calibrates the 7 mm ADC's full scale to 40 psi, and the
#: TLE firmware's ``PRESSURE_DVP_TARGET_MAX_RAW`` = 54000 raw is 40.1 psi on its
#: own transfer function.  It is the asymptote :meth:`CanNode._integrate` fills
#: toward; the fitted flow net does not use it.
SUPPLY_PSI = 40.0

#: Per-line command ceiling, psi.  CONTRACT.md section 8's safety rule is
#: enforced in the target generator and asserted again before transmission; this
#: constant exists so a node can be asked what it should never have been sent.
MAX_COMMAND_PSI = 30.0


# ---------------------------------------------------------------------------
# The node grid
# ---------------------------------------------------------------------------

#: 1 ms quanta between node-logic passes.  **Four, and four is forced.**
#:
#: The two firmwares run their control loops at different rates, both read from
#: source on 2026-09-10: the TLE PID at ``PRESSURE_PID_HZ`` = 750 Hz
#: (``firmware/tle/main/pressure_controller.c``) and the 7 mm bang-bang loop at
#: 1000 Hz (``firmware/legacy/main/main.c``'s ``vTaskDelay(1)`` with
#: ``CONFIG_FREERTOS_HZ=1000`` in that tree's ``sdkconfig``; the "2000us (2ms)
#: window" comment beside it predates that setting and is stale).  CONTRACT.md
#: section 4 requires the firmware's control period to be an integer count of
#: quanta.  ``n`` ms is a whole number of 750 Hz periods only when ``0.75 * n``
#: is an integer, so ``n`` must be a multiple of 4 -- and 4 ms is then also
#: exactly four 7 mm periods.  Four is the smallest grid on which **neither**
#: population's control period is truncated.
NODE_LOGIC_EVERY = 4

#: TLE PID rate, Hz -- ``PRESSURE_PID_HZ`` in ``pressure_controller.c``.
TLE_PID_HZ = 750.0

#: 7 mm bang-bang rate, Hz -- one FreeRTOS tick at ``CONFIG_FREERTOS_HZ=1000``.
SEVEN_MM_TICK_HZ = 1000.0

#: Sync-loss failsafe, s.  ``TLE_LEGACY_SYNC_TIMEOUT_US`` = 500 ms, TLE only.
TLE_SYNC_TIMEOUT_S = 0.5

#: The bus cycle, Hz.  ``tlelib.proto.CYCLE_HZ``; 6.667 ms, which is 6.667
#: quanta and 1.667 node passes and therefore lands off both grids.
CYCLE_HZ = P.CYCLE_HZ

#: Leak imposed by ``NodeFault(leak=True)``, Pa/s.  Ported from the reference's
#: ``LEAK_FAULT_PA_PER_S``: it is roughly four times the RS485 rig's ~720 Pa/s
#: natural leak, chosen so a fault is unambiguous against a 1 %/s-of-15-psi
#: watch threshold (~1030 Pa/s).  It has **not** been re-measured on this arm;
#: the one known leaker here is ``0x110``, which reads +5.1 psi at rest.
LEAK_FAULT_PA_S = 3000.0

#: Forgiveness in :meth:`SimArm.advance_to`, in quanta.  Reference section 1.5:
#: without it a target meaning "exactly n quanta" consumes ``n-1`` whenever the
#: float sum lands one ulp low, which silently changes every trajectory.
_QUANTUM_EPS = 1e-9


# ---------------------------------------------------------------------------
# Sensor front ends
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdcSpec:
    """One board's pressure front end, in the raw counts its own ADC produces.

    The two populations do not share a sensor path and cannot share a spec.  A
    TLE board carries a 16-bit LTC1864 whose raw reading is shifted right by
    ``TLE_LEGACY_RAW_SHIFT`` = 4 to fit the 12-bit compact field, so the wire
    field and the raw field differ by a factor of 16; a 7 mm board reports its
    own 12-bit ADC directly and ``wire_shift`` is 0 for it.

    ``filter_shift`` and ``filter_hz`` describe the fixed-gain exponential
    average the firmware applies before the control law sees the reading, whose
    time constant is ``2**filter_shift / filter_hz``.  Modelling it matters
    because that lag is most of the phase the regulator has to fight.
    """

    zero_counts: float
    counts_per_psi: float
    adc_max: int
    noise_sd_counts: float
    wire_shift: int
    filter_shift: int
    filter_hz: float

    @property
    def filter_tau_s(self) -> float:
        return (1 << self.filter_shift) / self.filter_hz

    @property
    def noise_psi(self) -> float:
        return self.noise_sd_counts / self.counts_per_psi

    def counts_of(self, psi: float) -> float:
        return self.zero_counts + psi * self.counts_per_psi

    def psi_of(self, counts: float) -> float:
        return (counts - self.zero_counts) / self.counts_per_psi


#: The TLE board's front end.  ``TLE_RAW_0_PSI`` = 15100 counts at 0 psi gauge
#: and ``TLE_COUNTS_PER_PSI_RAW`` = 972.5 counts/psi are ``tlelib/proto.py``'s,
#: which is where ``NodeCal``'s 943.75 / 60.78125 compact-field numbers come
#: from (both divided by 16).  The noise figure is the firmware's own: fw 1.46's
#: comment beside ``s_adc_avg_shift`` records a **white sigma of 9.5 raw counts**
#: unfiltered and 1.1 counts after the shift-3 EMA, measured on the bench.  9.5
#: counts is 0.0098 psi, under a fifth of the 49-raw dead zone.
TLE_ADC = AdcSpec(zero_counts=P.TLE_RAW_0_PSI,
                  counts_per_psi=P.TLE_COUNTS_PER_PSI_RAW,
                  adc_max=65535,
                  noise_sd_counts=9.5,
                  wire_shift=P.TLE_RAW_SHIFT,
                  filter_shift=3,
                  filter_hz=TLE_PID_HZ)

#: The 7 mm board's front end.  Zero and span are ``tlelib.proto``'s legacy
#: defaults, 754.4 counts and 56.14 counts/psi, the transfer function the old
#: firmware documents (PBOT 760 = 0.1 psi, PCAP 3000 = 40 psi).
#:
#: **The noise figure is an assumption, not a measurement on this arm.**  The
#: legacy firmware never states one; 2.0 counts is the middle of the 1.7-2.7
#: count spread the RS485 rack's 12-bit ESP32 front end measured, and it is
#: carried here only because a noiseless ADC lets the bang-bang law sit exactly
#: on a threshold forever.  Re-measure it from a resting board's reply stream
#: before quoting any number that depends on it.
SEVEN_MM_ADC = AdcSpec(zero_counts=P.LEGACY_ZERO_COUNTS,
                       counts_per_psi=P.LEGACY_COUNTS_PER_PSI,
                       adc_max=4095,
                       noise_sd_counts=2.0,
                       wire_shift=0,
                       filter_shift=2,
                       filter_hz=SEVEN_MM_TICK_HZ)


# ---------------------------------------------------------------------------
# Firmware tuning, transcribed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TleTuning:
    """The DVP loop's shipped constants, from ``pressure_controller.c`` fw 1.46.

    Every field is a named default rather than a module constant read at call
    time, so a caller sweeping a gain does it by constructing a tuning rather
    than by mutating global state a second node is also reading.

    The current budget is the part worth reading twice.  ``safe_max_code`` = 120
    is a hard ceiling the firmware never emits past, because the Clippard DVP is
    rated 190 mA continuous and full-scale code 127 would be 200 mA at
    ``I_mA = code * 200/127``.  But the DC setpoint is not the whole coil
    current: the TLE92464's hardware dither generator overlays a triangle *on
    top of* the setpoint register, so ``dither_peak_max_code`` = 4 (about
    6.3 mA) is reserved out of the 120 and every host-supplied DC code is
    clamped to the remaining ``host_max_code`` = 116, which is 182.7 mA -- still
    comfortably above the valve's ~180 mA saturation, so no authority is lost.
    """

    kp16: float = 2048.0
    ki16: float = 2500.0
    kd16: float = 2500.0
    #: Dead zone / soft zone, raw counts.  ``PRESSURE_DEFAULT_DEADBAND_RAW`` =
    #: 49 raw is about 0.05 psi; inside it the P and I contributions fade
    #: linearly to ``soft_zone_min_gain`` so a high-gain hold stays quiet.
    deadband_raw: float = 49.0
    soft_zone_min_gain: float = 64.0 / 256.0
    #: Inlet current range, codes.  51 is the measured inlet crack at 80.3 mA
    #: (2026-08-10 sweep; codes 38..50 are dead); 110 is about 173 mA.
    open_code: int = 51
    max_code: int = 110
    #: Outlet range.  The outlet cracks at 46 but has no authority against the
    #: leak until about 55, and saturates by 90 -- both measured.
    outlet_open_code: int = 57
    outlet_max_code: int = 96
    #: The hard budget, and the slice of it reserved for the dither overlay.
    safe_max_code: int = 120
    dither_peak_max_code: int = 4
    #: Slew limit, codes per current update at ``current_update_hz``.
    slew_code: int = 4
    code_jump_max: int = 2
    current_update_hz: float = 1500.0
    #: Split range.  The outlet engages only below ``-vent_threshold_pm``; in
    #: between the loop coasts and the plant leak does the venting, which is
    #: what removes the inlet/outlet relay limit cycle near low setpoints.
    vent_threshold_pm: float = 20.0
    output_deadband_pm: float = 0.0
    #: Dead-time compensation: the error is formed against the pressure
    #: projected this many PID steps ahead on the filtered rate.  Ten steps is
    #: 13.3 ms, which the firmware records as a 27x cut in small-step overshoot
    #: for about 9 ms of sine lag.
    deadtime_steps: int = 10
    deriv_lp_div: float = 3.0
    d_fade_band_raw: float = 486.0
    #: Pressure-adaptive gain schedule: commanded gains are "gains at 5 psi"
    #: (raw 19963) and every product is scaled by ``ref/raw``, floored at 0.25x
    #: and capped at 2x, because the plant's small-signal gain grows about
    #: proportionally with absolute reading.
    gain_ref_raw: float = 19963.0
    gain_sched_min_raw: float = 4096.0
    gain_sched_min: float = 0.25
    gain_sched_max: float = 2.0
    #: Leak make-up feedforward, fitted to the measured steady hold integral
    #: (514 raw at 5 psi, 2261 at 25 psi): ``ff = 77 + 23*(target-0psi)/256``.
    ff_base_raw: float = 77.0
    ff_gain_q8: float = 23.0
    ff_scale: float = 30.0 / 128.0
    #: The old plant's leak model over-seeds this bench by 2.7x, so the seed is
    #: scaled to land a touch below make-up and approach target from below.
    fs_seed_scale: float = 17.0 / 128.0
    fs_capture_raw: float = 389.0
    fs_bulk_raw: float = 972.0
    fs_rate_thresh: float = 2.0
    zero_psi_raw: float = 15100.0
    control_raw_scale: float = 4096.0
    integral_limit_raw: float = 8192.0
    #: Integral floor.  On a leak-down plant the steady output is always a
    #: positive inlet feed, so a symmetric integrator winds deeply negative
    #: through a down-step vent and then has to swing all the way back,
    #: undershooting on arrival.  Zero is the shipped mode.
    integral_floor_raw: float = 0.0
    integral_reset_min_raw: float = 2048.0
    target_max_raw: float = 54000.0

    @property
    def host_max_code(self) -> int:
        """The DC ceiling left once the dither overlay's slice is reserved."""
        return self.safe_max_code - self.dither_peak_max_code

    @staticmethod
    def code_to_ma(code: float) -> float:
        """``I_mA = code * 200/127`` -- the TLE92464 ICC full-scale mapping."""
        return code * 200.0 / 127.0


@dataclass(frozen=True)
class SevenMmTuning:
    """The 7 mm bang-bang law, from ``firmware/legacy/main/main.c``.

    The hysteresis is **not** a fixed pascal figure, and that is the single most
    load-bearing correction this module makes.  ``main.c`` computes it in the
    board's own 12-bit ADC counts and steps it with the target::

        #define DEADBAND 5
        if      (tp < 1000) deadband = DEADBAND;       /* below  4.4 psi */
        else if (tp < 1500) deadband = DEADBAND + 1;   /* below 13.3 psi */
        else if (tp < 2000) deadband = DEADBAND + 2;   /* below 22.2 psi */
        else                deadband = DEADBAND + 3;

    At ``LEGACY_COUNTS_PER_PSI`` = 56.14 that is 5 counts = 0.089 psi =
    **614 Pa** at the bottom of the range and 8 counts = 0.143 psi = **982 Pa**
    at the top.  CONTRACT.md section 2 quotes +/-2000 Pa as ``blend_width_pa``'s
    7 mm default; that is the RS485 arm's ``MARGIN_PA``, and these boards are
    between two and three times tighter.  The contract already labels that
    number an assumption to be re-measured rather than a fit, so nothing here
    contradicts it -- but a blend width of 2000 Pa describes a hysteresis this
    firmware does not have, and the trainer should be told.
    """

    deadband_base_counts: float = 5.0
    #: ``(target_counts_below, extra_counts)``, in the order the firmware tests.
    deadband_steps: tuple = ((1000.0, 0.0), (1500.0, 1.0), (2000.0, 2.0))
    deadband_top_extra: float = 3.0

    def deadband_counts(self, target_counts: float) -> float:
        for below, extra in self.deadband_steps:
            if target_counts < below:
                return self.deadband_base_counts + extra
        return self.deadband_base_counts + self.deadband_top_extra


# ---------------------------------------------------------------------------
# Faults, split by layer
# ---------------------------------------------------------------------------

@dataclass
class NodeFault:
    """What is wrong with one board, split by the layer that can observe it.

    ``leak`` and ``stuck_vent`` are **plant** faults and live at the seam in
    :class:`SimNode`.  ``absent`` and ``reply_dropout`` are **transport** faults
    and live in the shell (:mod:`digital_twin.sim_master`).  The reference's
    checklist item 8 is blunt about why they must not be mixed: a plant fault
    that hides in the transport cannot be reproduced offline by
    :mod:`digital_twin.replay`, which has no transport.
    """

    leak: bool = False
    stuck_vent: bool = False
    absent: bool = False
    reply_dropout: float = 0.0


# Valve actions.  Numerically equal to the TLE firmware's ``pressure_action_t``
# for the two states that matter; the twin does not model SETTLING.
ACTION_NONE = 0
ACTION_INLET = 1
ACTION_OUTLET = 2


class CanNode:
    """One board's firmware, on this bus's protocol.

    The 150 Hz cycle both firmwares implement, from
    ``firmware/tle/main/tle_can_legacy.h``'s header comment and reproduced step
    for step by ``firmware/legacy/main/can_routine.c``:

    1. the host sends the runtime table frames and boards **stage** what they
       receive -- nothing changes yet;
    2. the host sends the DLC-0 sync on ``0x090``;
    3. every board promotes its staged target to the live one and, in the same
       handler, latches the filtered pressure it will report, so the reported
       value belongs to the sync instant even though the replies themselves
       trickle out over the following milliseconds;
    4. boards whose slot was active reply with compact status on their base id.

    Steps 1 and 3 are :meth:`stage_command` and :meth:`sync_edge`.  The control
    law is **not** on that grid: unlike the RS485 node, which regulated at the
    bus tick, both of these firmwares run their loops on their own free-running
    timers.  That removes the reference's sharpest gotcha -- there, the bare-arm
    path and the shell path sampled differently, and porting only the bare one
    silently bought a low-fidelity twin -- and it is why this module has no
    ``sample_now``/``apply_now`` pair.
    """

    #: Seconds of sync silence before the outputs drop, or ``None`` for a board
    #: with no link-loss timeout.  Overridden per flavour.
    sync_timeout_s = None

    #: Firmware control-loop rate, Hz.  Overridden per flavour.
    control_hz = SEVEN_MM_TICK_HZ

    def __init__(self, base: int, variant: int, adc: AdcSpec, *,
                 fault: NodeFault | None = None,
                 rng: random.Random | None = None,
                 supply_psi: float = SUPPLY_PSI,
                 plant_tau_s: float = 0.35,
                 max_command_psi: float = MAX_COMMAND_PSI) -> None:
        self.base = int(base)
        self.variant = int(variant)
        self.adc = adc
        self.cal = P.NodeCal.for_variant(self.variant, self.base)
        self.fault = fault if fault is not None else NodeFault()
        self._rng = rng if rng is not None else random.Random(20260910 + self.base)
        self.supply_psi = float(supply_psi)
        self.plant_tau_s = float(plant_tau_s)
        self.max_command_psi = float(max_command_psi)

        #: THE plant state, one scalar per node, Pa gauge.
        self.p_pa = 0.0

        # -- staged by a table frame, promoted by the sync edge --------------
        self.pending_target_counts = 0
        self.pending_control = 0
        self.active_target_counts = 0
        self.active_control = 0
        self.enabled = False

        # -- what the reply carries.  Latched at the edge, not at transmit ---
        self.latched_counts = 0

        # -- the sensor path -------------------------------------------------
        self.raw = int(round(adc.zero_counts))
        self.filtered_raw = float(adc.zero_counts)
        self._filter_primed = False

        # -- status -----------------------------------------------------------
        self.command_seen = False
        self.error_sticky = False
        self.failsafe_active = False
        self.sync_seen_ever = False
        self.sync_count = 0
        self.command_count = 0
        self.sync_loss_count = 0
        self.last_sync_s = -math.inf
        self._now_s = 0.0

        # -- valve bookkeeping.  Reported and accounted; not a plant input ---
        self.action = ACTION_NONE
        self.valve_switches = 0
        self.inlet_open_s = 0.0
        self.outlet_open_s = 0.0
        self.air_in_psi = 0.0
        self.air_out_psi = 0.0

        # -- muscle state, written by SimArm before every node pass ----------
        self._l_m = 0.0
        self._ldot_m_s = 0.0

        # -- consumed by SimNode._integrate when SimArm batches the forward --
        self._net_flow_cache = None

    # -- identity -----------------------------------------------------------
    @property
    def index(self) -> int:
        """0-based actuator index: ``pam_{index+1}`` drives this board."""
        return self.base - ACTUATOR_FIRST

    @property
    def is_tle(self) -> bool:
        return self.variant == P.VARIANT_TLE_DVP

    # -- the plant seam.  SimNode overrides exactly these three -------------
    @property
    def true_psi(self) -> float:
        return self.p_pa / PA_PER_PSI

    @property
    def leak_pa_s(self) -> float:
        return LEAK_FAULT_PA_S if self.fault.leak else 0.0

    def _integrate(self, dt: float) -> None:
        """Stand-in plant: first order toward the open valve's asymptote.

        Deliberately not the fitted flow net.  It exists so that
        :class:`CanNode` is exercisable without a checkpoint, in the same role
        the RS485 ``FakeNode``'s double exponential played, and
        :class:`SimNode` replaces it wholesale.  Anything measured on it
        describes this expression and not the arm.
        """
        self._net_flow_cache = None
        if dt <= 0.0:
            return
        self._accrue_open_time(dt)
        p0 = self.p_pa
        action = self.action
        if action == ACTION_OUTLET and self.fault.stuck_vent:
            action = ACTION_NONE
        if action == ACTION_INLET:
            asymptote = self.supply_psi * PA_PER_PSI
        elif action == ACTION_OUTLET:
            asymptote = 0.0
        else:
            asymptote = self.p_pa
        alpha = 1.0 - math.exp(-dt / self.plant_tau_s)
        self.p_pa += alpha * (asymptote - self.p_pa)
        self.p_pa = max(0.0, self.p_pa - self.leak_pa_s * dt)
        self._accrue_air(p0)

    # -- accounting ---------------------------------------------------------
    def _accrue_open_time(self, dt: float) -> None:
        if self.action == ACTION_INLET:
            self.inlet_open_s += dt
        elif self.action == ACTION_OUTLET:
            self.outlet_open_s += dt

    def _accrue_air(self, p0: float) -> None:
        """Charge only REALIZED pressure changes, and only on an open valve.

        Nothing is charged on the closed branch: a fitted net's closed-branch
        flow is an artifact of the fit, not air the operator paid for.  A stuck
        vent therefore costs nothing here while ``outlet_open_s`` still counts
        the energised coil, which is the honest split.
        """
        dp = self.p_pa - p0
        if self.action == ACTION_INLET and dp > 0.0:
            self.air_in_psi += dp / PA_PER_PSI
        elif self.action == ACTION_OUTLET:
            self.air_out_psi += abs(dp) / PA_PER_PSI

    def _switch_action(self, action: int) -> None:
        """Unconditional, on purpose.

        Any refusal rule -- a minimum dwell, a rate limit -- belongs in the
        decision layer that calls this and never here, or the safety paths
        (:meth:`all_off`, the failsafe trap) become refusable.  Reference
        section 1.11.
        """
        if action != self.action:
            self.action = action
            self.valve_switches += 1

    # -- the muscle, written by SimArm --------------------------------------
    def set_muscle_state(self, l_m: float, ldot_m_s: float) -> None:
        self._l_m = float(l_m)
        self._ldot_m_s = float(ldot_m_s)

    # -- clocks --------------------------------------------------------------
    def _stamp_now(self, now: float) -> None:
        """Monotone maximum of the sync mark and the arm's grid time.

        Two grids write this node's clock: the sync edge, at the host's own mark
        ``t``, and the node pass, at ``SimArm.sim_now``, which trails the last
        mark by up to one quantum.  At 150 Hz the period is 6.667 ms, which is
        6.667 quanta of 1 ms and 1.667 node passes, so the mark is routinely
        ahead of the grid.  Taking the max keeps the failsafe window on a clock
        that never runs backwards, which would otherwise make a just-elapsed
        timeout read negative for one pass.
        """
        now = float(now)
        if now > self._now_s:
            self._now_s = now

    # -- the wire ------------------------------------------------------------
    def stage_command(self, counts: int, enable: bool) -> None:
        """A runtime-table slot arriving.  Stages; changes nothing yet."""
        self.pending_target_counts = int(counts) & P.PRESSURE_MASK
        self.pending_control = P.CONTROL_ENABLE if enable else 0
        self.command_seen = True
        self.command_count += 1

    def sync_edge(self, now: float) -> None:
        """The synchronised state transition; everything simultaneous is here.

        Ported from ``tle_can_legacy.c::handle_sync`` and
        ``can_routine.c::apply_sync_edge``, which agree: latch the filtered
        pressure, promote the staged target, promote the staged control byte.
        The enable bit is a **level**, not an edge -- a loop that stopped itself
        is re-armed while the host is still commanding it.
        """
        self._stamp_now(now)
        self.last_sync_s = self._now_s
        self.sync_seen_ever = True
        self.sync_count += 1

        self.latched_counts = self._wire_counts()
        self.active_target_counts = self.pending_target_counts
        self.active_control = self.pending_control

        want_enabled = bool(self.active_control & P.CONTROL_ENABLE)
        if want_enabled and not self.enabled:
            self.failsafe_active = False
        self.enabled = want_enabled
        self.on_enable_changed(want_enabled)

    def on_enable_changed(self, enabled: bool) -> None:
        """Hook for what a flavour does when the enable level moves."""

    def all_off(self) -> None:
        """Host-commanded stop: drop the enable and shut both valves.

        This is the twin's e-stop path and it **traps** the air it finds rather
        than venting it, exactly as the reference's ALL_OFF does.  Venting is a
        separate act, and a rollout that wants an empty arm has to command a
        vent target and wait for it before shutting the valves.
        """
        self.enabled = False
        self.active_control = 0
        self.pending_control = 0
        self._switch_action(ACTION_NONE)

    # -- the sensor ----------------------------------------------------------
    def _sample_adc(self) -> None:
        """Noise, quantise, rail-clamp, then the firmware's own EMA.

        The truncation at 3 sigma is the reference's and is kept: an untruncated
        Gaussian puts a one-in-a-thousand sample far enough out to flip a
        bang-bang decision, and a rare flipped decision is exactly the kind of
        difference that cannot be attributed afterwards.
        """
        limit = 3.0 * self.adc.noise_sd_counts
        noise = self._rng.gauss(0.0, self.adc.noise_sd_counts) if limit > 0.0 else 0.0
        noise = max(-limit, min(limit, noise))
        raw = self.adc.counts_of(self.true_psi) + noise
        self.raw = int(max(0, min(self.adc.adc_max, round(raw))))
        if not self._filter_primed:
            self.filtered_raw = float(self.raw)
            self._filter_primed = True
        else:
            gain = 1.0 / (1 << self.adc.filter_shift)
            self.filtered_raw += gain * (float(self.raw) - self.filtered_raw)

    def _wire_counts(self) -> int:
        """The 12-bit compact field, from the filtered raw reading.

        Truncated, not rounded, because ``compact_from_raw`` is a shift.
        """
        counts = int(self.filtered_raw) >> self.adc.wire_shift
        return max(0, min(P.PRESSURE_MASK, counts))

    # -- readout -------------------------------------------------------------
    @property
    def pressure_pa(self) -> float:
        """What the firmware's control law believes, Pa gauge.

        Measured, not true: the regulator acts on the reading the sensor
        reported, and so does everything downstream of it here.
        """
        return self.adc.psi_of(self.filtered_raw) * PA_PER_PSI

    @property
    def target_pa(self) -> float:
        """The promoted target, Pa gauge, on this board's own calibration."""
        return self.cal.counts_to_psi(self.active_target_counts) * PA_PER_PSI

    @property
    def plant_target_pa(self) -> float:
        """The target the **plant** sees, Pa gauge.

        Under CONTRACT.md Departure 1 the flow model's only command channel is
        ``e = target - measured``.  A board that is disabled or failsafed has
        both valves shut, and the closed-valve condition in that coordinate is
        ``e = 0``, so this returns the node's own true pressure then.  That is
        not the same as "no flow": the reference is explicit that a fitted net's
        closed branch has its own non-zero fixed point and that a trapped rack
        walks toward it, in a direction that depends on the checkpoint -- the
        bench fit inflated a settled 25.15 psi rack to 27.36 psi over 10 s while
        the shipped synthetic one drifted down.  Never pin the sign of that
        drift in a test; ask the model what it predicts and check against that.
        """
        if self._regulating:
            return self.target_pa
        return self.p_pa

    @property
    def _regulating(self) -> bool:
        return self.enabled and not self.failsafe_active

    def status_flags(self) -> int:
        """The four-bit compact status.  ``build_status_flags`` in both trees.

        ``STATUS_ERROR`` is sticky-until-read on the real boards, and clearing
        it on read reproduces that: a poller that drops a reply loses the error.
        That is a property of the protocol and not a defect of the twin.
        """
        flags = 0
        if self.error_sticky:
            flags |= P.STATUS_ERROR
            self.error_sticky = False
        if self.active_control & P.CONTROL_ENABLE:
            flags |= P.STATUS_ENABLED
        if self.command_seen:
            flags |= P.STATUS_COMMAND_SEEN
        return flags & P.FLAGS_MASK

    def compact_status(self) -> P.CompactStatus:
        return P.CompactStatus(self.base, self.latched_counts, self.status_flags())

    # -- the step ------------------------------------------------------------
    def step(self, dt: float, now: float, n_sub: int = 1) -> None:
        """Integrate, sample, check the failsafe, regulate -- in that order.

        The order is immovable and is the reference's checklist item 5.  The
        regulator must act on the pressure the sensor reported, not on the true
        one, so the ADC sample has to fall between the plant and the control
        law; and the failsafe has to fall between the sample and the control
        law, so a board that has just timed out does not get one more regulation
        pass out of the same tape.

        ``n_sub`` is how many of this firmware's own control periods fall inside
        one node pass.  :class:`SimArm` picks the node grid so that it is a whole
        number for both populations.  Sub-step edges are computed by difference
        from a shared endpoint, so they sum to ``dt`` exactly rather than to
        ``dt`` plus ``n_sub`` roundings -- which is what keeps the leak
        arithmetic exact to the pascal.
        """
        self._stamp_now(now)
        n_sub = max(1, int(n_sub))
        edge = 0.0
        for k in range(n_sub):
            nxt = dt if k == n_sub - 1 else dt * (k + 1) / n_sub
            self._integrate(nxt - edge)
            edge = nxt
            self._sample_adc()
            self._check_failsafe()
            if self._regulating:
                self.regulate()
            else:
                self.on_not_regulating()

    def _check_failsafe(self) -> None:
        """The sync-loss trap, or nothing at all if this board has none."""
        timeout = self.sync_timeout_s
        if timeout is None:
            return
        if not self.sync_seen_ever or not self.enabled:
            return
        if (self._now_s - self.last_sync_s) < timeout:
            return
        self.sync_loss_count += 1
        self.error_sticky = True
        self.failsafe_active = True
        self.enabled = False
        self.active_control = 0
        # The firmware clears the **pending** control byte too, so a later sync
        # edge replaying a stale table cannot re-enable the board: the host has
        # to send a fresh table frame carrying the enable bit.
        self.pending_control = 0
        self.on_failsafe()

    def on_failsafe(self) -> None:
        self._switch_action(ACTION_NONE)

    def on_not_regulating(self) -> None:
        """What the valves do while the loop is not running.  Flavour-specific."""
        self._switch_action(ACTION_NONE)

    def regulate(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError("CanNode has no control law; use a flavour")

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (f"<{type(self).__name__} 0x{self.base:03X} "
                f"{self.true_psi:.2f} psi target {self.target_pa / PA_PER_PSI:.2f} psi "
                f"{'ENABLED' if self.enabled else 'off'}"
                f"{' FAILSAFE' if self.failsafe_active else ''}>")


class SevenMmNode(CanNode):
    """A legacy 7 mm board: bang-bang, and no link-loss timeout whatsoever.

    ``firmware/legacy/main/main.c::clippard_7mm_valve_bangbang_ctrl``, in full::

        if (current_clean_pressure < (tp - deadband))       V_IN=1, V_OUT=0;
        else if (current_clean_pressure > (tp + deadband))  V_IN=0, V_OUT=1;
        else                                                V_IN=0, V_OUT=0;

    with ``deadband`` stepping 5 to 8 ADC counts with the target (see
    :class:`SevenMmTuning`).  Two behaviours around that loop are safety
    properties and are reproduced rather than repaired:

    * :attr:`sync_timeout_s` is ``None``.  Nothing in that tree clears the
      control byte on silence.
    * :meth:`on_not_regulating` **holds the valves where they were**.  The
      firmware's disable branch is a bare ``continue`` above the GPIO writes, so
      a board that was inflating when its enable bit went away goes on
      inflating.
    """

    sync_timeout_s = None
    control_hz = SEVEN_MM_TICK_HZ

    def __init__(self, base: int, variant: int, adc: AdcSpec, *,
                 tuning: SevenMmTuning | None = None, **kwargs) -> None:
        super().__init__(base, variant, adc, **kwargs)
        self.tuning = tuning if tuning is not None else SevenMmTuning()

    def regulate(self) -> None:
        deadband = self.tuning.deadband_counts(float(self.active_target_counts))
        target = float(self.active_target_counts)
        clean = self.filtered_raw          # 12-bit already; wire_shift is 0 here
        if clean < target - deadband:
            self._switch_action(ACTION_INLET)
        elif clean > target + deadband:
            self._switch_action(ACTION_OUTLET)
        else:
            self._switch_action(ACTION_NONE)

    def on_not_regulating(self) -> None:
        """Freeze, do not shut.  See the class docstring; this is the firmware."""
        return

    def deadband_pa(self) -> float:
        """The live hysteresis in pascals, for a caller that wants to quote it."""
        counts = self.tuning.deadband_counts(float(self.active_target_counts))
        return counts / self.adc.counts_per_psi * PA_PER_PSI


class TleNode(CanNode):
    """A TLE92464/DVP board: the onboard proportional loop, plus the 500 ms trap.

    The law is ``pressure_controller.c``'s DVP branch at fw 1.46, in floating
    point and in the 16-bit raw counts the firmware works in:

    * a dead-time-compensated error, the pressure projected ``deadtime_steps``
      PID steps ahead on the filtered rate;
    * a pressure-adaptive gain schedule, ``gain_ref_raw / raw`` clamped to
      ``[0.25, 2]``;
    * a soft zone that fades P and I to a quarter inside the 49-raw dead zone;
    * a derivative faded to zero at the setpoint so it stops chewing the valve
      current in the noise-only band;
    * conditional integration -- frozen while the drive already saturates in the
      error's direction, while the pressure ceiling blocks positive drive, while
      a further-negative integral could only wind against the coast band, and
      while the flow-shaping deceleration is active -- with the floor at zero
      and the limit at twice full output;
    * split-range output: inlet above the output dead zone, outlet below
      ``-vent_threshold_pm``, and a coast band between them where both valves
      are shut and the plant's own leak does the venting;
    * an ``[open, max]`` current mapping per channel and a slew limit of
      ``slew_code`` codes per current update.

    WHAT IS NOT REPRODUCED, because it bounds how far a result from this node
    travels: the firmware's fixed-point truncation -- Q8 gains, a Q16 integral,
    Q9 code fractions and a moving-window sub-LSB dither -- is done in float
    here, so the stick zones and one-LSB limit cycles those quantisations
    produce do not appear.  Neither does the fw 1.41 reference generator
    (``vmax_up``/``vmax_down`` are 0 in the shipped build, so the PID does the
    shaping) nor the distance-scaled deceleration caps (``fs_decel_up`` and
    ``fs_decel_down`` are 0, because the cap released discontinuously at the
    0.4 psi capture band and measurably degraded 10 psi steps).  Flow shaping is
    enabled in the shipped build **solely** for its scaled integral seed, and
    that seed is the only part of it modelled here.
    """

    sync_timeout_s = TLE_SYNC_TIMEOUT_S
    control_hz = TLE_PID_HZ

    def __init__(self, base: int, variant: int, adc: AdcSpec, *,
                 tuning: TleTuning | None = None, **kwargs) -> None:
        super().__init__(base, variant, adc, **kwargs)
        self.tuning = tuning if tuning is not None else TleTuning()
        self.integral_raw = 0.0
        self.deriv_filt = 0.0
        self._prev_raw = None
        self._seed_target = None
        self._fs_rising = False
        self.output_permille = 0.0
        self.current_code = 0.0
        self.output_channel = ACTION_NONE

    # -- the enable level ----------------------------------------------------
    def on_enable_changed(self, enabled: bool) -> None:
        if not enabled:
            self._drop_outputs()
            return
        self._seed_integral(float(self.active_target_counts * (1 << self.adc.wire_shift)))

    def on_failsafe(self) -> None:
        self._drop_outputs()

    def on_not_regulating(self) -> None:
        self._drop_outputs()

    def all_off(self) -> None:
        super().all_off()
        self._drop_outputs()

    def _drop_outputs(self) -> None:
        self._switch_action(ACTION_NONE)
        self.output_channel = ACTION_NONE
        self.current_code = 0.0
        self.output_permille = 0.0

    def _seed_integral(self, target_raw: float) -> None:
        """Seed the integrator at the measured steady leak make-up.

        ``dvp_makeup_seed_q16`` was fitted to 514 raw at 5 psi and 2261 raw at
        25 psi, then scaled by ``fs_seed_scale`` = 17/128 because the old plant's
        leak model over-seeds this bench by a factor of 2.7.  Seeding rather than
        integrating up to the make-up is what makes the slew-to-hold handoff
        bumpless: without it every step ends in a sag, an undershoot and a ring.
        """
        t = self.tuning
        above = max(0.0, target_raw - t.zero_psi_raw)
        seed = (t.ff_base_raw + t.ff_gain_q8 * above / 256.0) * t.fs_seed_scale
        self.integral_raw = min(max(seed, t.integral_floor_raw), t.integral_limit_raw)
        self._seed_target = target_raw

    # -- the loop -------------------------------------------------------------
    def regulate(self) -> None:
        t = self.tuning
        target_raw = float(self.active_target_counts) * (1 << self.adc.wire_shift)
        p_raw = self.filtered_raw

        if self._seed_target is None or abs(target_raw - self._seed_target) > 0.5:
            self._fs_rising = target_raw > p_raw
            self._seed_integral(target_raw)

        # Derivative, low-passed at ``deriv_lp_div`` per PID step.
        delta = 0.0 if self._prev_raw is None else (p_raw - self._prev_raw)
        self._prev_raw = p_raw
        self.deriv_filt += (delta - self.deriv_filt) / t.deriv_lp_div

        predicted = p_raw + self.deriv_filt * t.deadtime_steps
        error = target_raw - predicted
        meas_error = target_raw - p_raw
        abs_meas = abs(meas_error)

        # Stale-integral backstop, gated on the MEASURED error: a predicted
        # error can flip sign on a transient rate and dump the leak make-up
        # spuriously, which is the one thing the standing integral must survive.
        reset_at = max(4.0 * t.deadband_raw, t.integral_reset_min_raw)
        if abs_meas >= reset_at and (meas_error * self.integral_raw) < 0.0:
            self.integral_raw = 0.0

        soft = self._soft_zone_gain(abs(error))
        sched = min(max(t.gain_ref_raw / max(p_raw, t.gain_sched_min_raw),
                        t.gain_sched_min), t.gain_sched_max)

        deriv = 0.0
        if t.kd16:
            # D_raw = -(kd16 * deriv_q12) >> 16 with deriv_q12 = rate * 4096.
            deriv = -(t.kd16 * self.deriv_filt / 16.0) * sched
            if t.d_fade_band_raw > 0.0 and abs_meas < t.d_fade_band_raw:
                deriv *= abs_meas / t.d_fade_band_raw

        prop = error * (t.kp16 / 256.0) * soft * sched
        above = max(0.0, target_raw - t.zero_psi_raw)
        feedforward = (t.ff_base_raw + t.ff_gain_q8 * above / 256.0) * t.ff_scale

        fs_decel = (abs_meas > t.fs_capture_raw
                    and ((self._fs_rising and meas_error > 0.0)
                         or (not self._fs_rising and meas_error < 0.0))
                    and (abs_meas > t.fs_bulk_raw
                         or abs(self.deriv_filt) > t.fs_rate_thresh))

        if t.ki16:
            # P+I only in the saturation test: with D in it, a braking
            # derivative desaturates the sum and the integrator instantly
            # recharges to the rail, cancelling the brake.
            presat = prop + feedforward + self.integral_raw
            saturated = ((error > 0.0 and presat >= t.control_raw_scale)
                         or (error < 0.0 and presat <= -t.control_raw_scale))
            ceiling_blocks = p_raw >= t.target_max_raw and error > 0.0
            vent_raw = t.vent_threshold_pm * t.control_raw_scale / 1000.0
            dead_raw = t.output_deadband_pm * t.control_raw_scale / 1000.0
            coast_windup = error < 0.0 and dead_raw >= presat >= -vent_raw
            if not (saturated or ceiling_blocks or coast_windup or fs_decel):
                self.integral_raw += error * t.ki16 * soft * sched / 65536.0
                self.integral_raw = min(max(self.integral_raw, t.integral_floor_raw),
                                        t.integral_limit_raw)

        drive = prop + feedforward + self.integral_raw + deriv
        permille = drive * 1000.0 / t.control_raw_scale
        if p_raw >= t.target_max_raw and permille > 0.0:
            permille = 0.0
        self._prepare_output(permille)

    def _soft_zone_gain(self, abs_error: float) -> float:
        t = self.tuning
        if t.deadband_raw <= 0.0 or abs_error >= t.deadband_raw:
            return 1.0
        span = 1.0 - t.soft_zone_min_gain
        return t.soft_zone_min_gain + span * abs_error / t.deadband_raw

    def _prepare_output(self, permille: float) -> None:
        """Split range, per-channel current span, then the slew limit."""
        t = self.tuning
        permille = min(max(permille, -1000.0), 1000.0)
        self.output_permille = permille

        if permille > t.output_deadband_pm:
            action, open_code, max_code = ACTION_INLET, t.open_code, t.max_code
            magnitude = permille
        elif permille < -(t.vent_threshold_pm + t.output_deadband_pm):
            action = ACTION_OUTLET
            open_code, max_code = t.outlet_open_code, t.outlet_max_code
            magnitude = min(1000.0, -permille - t.vent_threshold_pm)
        else:
            # The coast band: neither valve drives and the leak bleeds down.
            self._switch_action(ACTION_NONE)
            self.output_channel = ACTION_NONE
            self.current_code = 0.0
            return

        if self.output_channel != action:
            self.current_code = 0.0
        self._switch_action(action)
        self.output_channel = action

        max_code = min(max(max_code, open_code), t.host_max_code)
        desired = open_code + (max_code - open_code) * magnitude / 1000.0
        desired = min(max(desired, open_code), max_code)
        if self.current_code <= 0.0:
            self.current_code = float(open_code)
        if desired > self.current_code:
            self.current_code = min(desired, self.current_code + t.slew_code)
        else:
            self.current_code = max(desired, self.current_code - t.slew_code)

    @property
    def current_ma(self) -> float:
        """Commanded DC coil current, mA.  Excludes the dither overlay, whose
        peak is budgeted separately at ``dither_peak_max_code``."""
        return TleTuning.code_to_ma(self.current_code)


# ---------------------------------------------------------------------------
# The plant seam
# ---------------------------------------------------------------------------

class SimNode(CanNode):
    """The plant seam: override ONLY the plant, exactly as the reference does.

    Three members and nothing else -- :attr:`true_psi` pins the ADC coupling,
    :attr:`leak_pa_s` carries the fault override, and :meth:`_integrate` runs
    the fitted flow model minus the leak with a ``p >= 0`` clamp after every
    substep.  Every firmware semantic above is inherited unchanged, which is the
    property that keeps the twin's node and the standalone node from drifting
    apart.

    It is mixed in ahead of the flavour, so a real node is
    ``class _SimTleNode(SimNode, TleNode)`` and takes the seam from the left and
    the control law from the right.
    """

    def __init__(self, *args, actuator=None,
                 leak_fault_pa_s: float = LEAK_FAULT_PA_S, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if actuator is None:
            raise ValueError(
                "SimNode needs an actuator model; see CONTRACT.md section 2")
        self.actuator = actuator
        self.leak_fault_pa_s = float(leak_fault_pa_s)

    @property
    def true_psi(self) -> float:
        """The whole plant-to-sensor coupling, and deliberately nothing more."""
        return self.p_pa / PA_PER_PSI

    @property
    def leak_pa_s(self) -> float:
        if self.fault.leak:
            return self.leak_fault_pa_s
        return float(self.actuator.leak_pa_s[self.index])

    def _integrate(self, dt: float) -> None:
        """``dp = net(p, e, l, ldot) - leak``, clamped at zero every substep.

        The clamp is inside the substep and not once per node pass because a
        learned ``dp/dt`` integrated through a vent goes negative otherwise, and
        a McKibben cannot go below ambient.  Reference invariant P1.

        ``stuck_vent`` is adapted rather than ported.  The reference flipped the
        flow model's *valve input*, which this arm's model does not have --
        Departure 1 replaced the valve one-hot with the commanded error.  Here it
        blocks the outward half of the flow instead, and leaves ``action``, the
        status byte and ``outlet_open_s`` honest: the firmware does not know its
        exhaust is blocked, so it must still report that it is venting.
        """
        cached, self._net_flow_cache = self._net_flow_cache, None
        if dt <= 0.0:
            return
        self._accrue_open_time(dt)

        p0 = self.p_pa
        if cached is None:
            net = float(self.actuator.net_flow_pa_s(
                self.index + 1, self.p_pa, self.plant_target_pa,
                self._l_m, self._ldot_m_s))
        else:
            net = float(cached)
        dp = net - self.leak_pa_s
        if self.fault.stuck_vent and dp < 0.0:
            dp = 0.0
        self.p_pa = max(0.0, self.p_pa + dp * dt)
        self._accrue_air(p0)


class _SimSevenMmNode(SimNode, SevenMmNode):
    """A 7 mm board with the fitted plant behind it."""


class _SimTleNode(SimNode, TleNode):
    """A TLE/DVP board with the fitted plant behind it."""


def make_node(base: int, variant: int, *, actuator=None, **kwargs) -> CanNode:
    """Build the node flavour the **variant byte** calls for, never the id range.

    CONTRACT.md Departure 2: a TLE board legitimately sat at ``0x114`` during the
    2026-08 bench session, and a twin that reads the population off the address
    would command and display it on the 7 mm scale -- a 10 % error in psi at the
    top of the range.

    With ``actuator=None`` the node keeps :meth:`CanNode._integrate`'s stand-in
    plant, which is what makes a firmware-only test possible; passing an
    actuator swaps in the :class:`SimNode` seam.
    """
    tle = int(variant) == P.VARIANT_TLE_DVP
    adc = kwargs.pop("adc", TLE_ADC if tle else SEVEN_MM_ADC)
    if actuator is None:
        cls = TleNode if tle else SevenMmNode
        return cls(base, variant, adc, **kwargs)
    cls = _SimTleNode if tle else _SimSevenMmNode
    return cls(base, variant, adc, actuator=actuator, **kwargs)


# ---------------------------------------------------------------------------
# Node-grid arithmetic
# ---------------------------------------------------------------------------

def sub_step_counts(node_grid_s: float, *, strict: bool = True,
                    rates: tuple = (TLE_PID_HZ, SEVEN_MM_TICK_HZ)) -> dict:
    """How many firmware control periods fall inside one node pass, per rate.

    ``strict`` refuses a grid on which either count is fractional, and that
    refusal is the point.  The reference's own warning is that changing the node
    grid for a plant identified around a particular control period "would
    silently desynchronize the sim from the trained actuator net"; a fractional
    count is precisely how that happens with nothing raising.
    """
    counts = {}
    for hz in rates:
        exact = node_grid_s * hz
        n = int(round(exact))
        if abs(exact - n) > 1e-9 or n < 1:
            if strict:
                raise ValueError(
                    f"a node grid of {node_grid_s * 1e3:.4f} ms carries "
                    f"{exact:.4f} periods of a {hz:.0f} Hz firmware loop, which "
                    f"is not a whole number; pick a grid that is a common "
                    f"multiple of {1e3 / TLE_PID_HZ:.4f} ms and "
                    f"{1e3 / SEVEN_MM_TICK_HZ:.4f} ms -- 4 ms is the smallest -- "
                    f"or pass strict_node_grid=False and record the grid the fit "
                    f"was made on in the checkpoint metadata")
            n = max(1, n)
        counts[hz] = n
    return counts


def _l0_per_act(actuator) -> np.ndarray:
    """The 24 anchored rest lengths, however the checkpoint spells them.

    CONTRACT.md section 2 fits ``l0`` per segment, three of them; the reference
    expands it 8x into ``l0_per_act``.  Accepting either spelling keeps this
    module from breaking on a naming choice in a file it does not own.
    """
    if hasattr(actuator, "l0_per_act"):
        return np.asarray(actuator.l0_per_act, dtype=float)
    return np.repeat(np.asarray(actuator.l0, dtype=float), N_NODES // 3)


# ---------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------

class SimArm:
    """MuJoCo plus twenty-four nodes plus one reentrant lock.

    THE CLOCK IS THE CONTRACT.  :meth:`advance_to` reproduces
    ``reference/sim_core.md`` section 7.1 invariants C1-C5 unchanged: the first
    call pins the origin and consumes nothing, a target at or behind the
    furthest one seen is a no-op, the sub-quantum remainder accumulates across
    calls, and there is one part in ``1e9`` of a quantum of forgiveness.  Those
    four together are what make two racing advancer threads with incommensurate
    hops produce bit-identical ``data.qpos`` against one serial advancer:
    determinism, not exclusion, is what makes interleaving safe.

    THE ORDER INSIDE ONE QUANTUM IS FIXED, and CONTRACT.md section 4 writes it
    out.  Node logic first, when the quantum is on the node grid; then the force
    from tendon length; then one ``mj_step``.  The force is written **every**
    quantum and not every node pass, because the muscle's length changes every
    physics step and a force held across four of them is a fourfold
    zero-order-hold error in the one channel the arm is driven through.
    """

    def __init__(self, *,
                 actuator=None,
                 xml: str | None = None,
                 variants: dict | None = None,
                 ids: tuple = ALL_IDS,
                 absent: tuple = (),
                 faults: dict | None = None,
                 timestep_s: float | None = None,
                 node_logic_every: int | None = None,
                 strict_node_grid: bool = True,
                 tle_adc: AdcSpec = TLE_ADC,
                 seven_mm_adc: AdcSpec = SEVEN_MM_ADC,
                 tle_tuning: TleTuning | None = None,
                 seven_mm_tuning: SevenMmTuning | None = None,
                 supply_psi: float = SUPPLY_PSI,
                 leak_fault_pa_s: float = LEAK_FAULT_PA_S,
                 tendon_damping_const: float | None = None,
                 batched_actuator: bool = False,
                 seed: int = 20260910,
                 node_hook=None,
                 **mjcf_tunables) -> None:
        import mujoco

        self._mujoco = mujoco
        self.seed = int(seed)

        self.variants = dict(NOMINAL_VARIANTS if variants is None else variants)
        self.actuator = (actuator if actuator is not None
                         else _default_actuator(self.variants, ids, seed=self.seed))
        self._l0_per_act = _l0_per_act(self.actuator)

        if xml is None:
            xml = _generated_xml(**mjcf_tunables)
        elif mjcf_tunables:
            raise ValueError(
                "xml= and MJCF tunables are mutually exclusive: a handed-in scene "
                f"cannot honour {sorted(mjcf_tunables)}")
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        if timestep_s is not None:
            if float(timestep_s) <= 0.0:
                raise ValueError("timestep_s must be positive")
            self.model.opt.timestep = float(timestep_s)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        #: Read back from the compiled model, never assumed.  A generator that
        #: quietly ships a different ``<option timestep>`` would otherwise
        #: rescale the node grid with nothing raising.
        self._dt = float(self.model.opt.timestep)

        self.node_logic_every = int(NODE_LOGIC_EVERY if node_logic_every is None
                                    else node_logic_every)
        if self.node_logic_every < 1:
            raise ValueError("node_logic_every must be >= 1")
        self.node_grid_s = self.node_logic_every * self._dt
        self._sub_counts = sub_step_counts(self.node_grid_s, strict=strict_node_grid)

        #: THE mutex.  Reentrant so a consumer can hold it across
        #: advance-then-read without a second lock object appearing.
        self.lock = threading.RLock()

        self.batched_actuator = bool(batched_actuator)
        self.tendon_damping_const = tendon_damping_const
        self.node_hook = node_hook

        self._bind_actuators()
        self._build_nodes(ids, absent, faults, tle_adc, seven_mm_adc,
                          tle_tuning, seven_mm_tuning, supply_psi, leak_fault_pa_s)

        self._t_origin = None
        self._t_target = None
        self._accum = 0.0
        self.quanta_done = 0
        self.node_passes = 0
        self._last_node_quantum = 0

    # -- construction --------------------------------------------------------
    def _bind_actuators(self) -> None:
        """Resolve ``pam_1``..``pam_24`` by NAME, and capture the qpos0 anchors.

        By name and not by index, because the merged multi-robot scene puts other
        robots' actuators in the same model and an index that happens to be right
        in the single-arm scene silently drives a Gen3 joint in the room.
        """
        mj = self._mujoco
        missing, act_ids, ten_ids = [], [], []
        for k in range(1, N_NODES + 1):
            aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"pam_{k}")
            if aid < 0:
                missing.append(f"pam_{k}")
                continue
            act_ids.append(aid)
            ten_ids.append(int(self.model.actuator_trnid[aid, 0]))
        if missing:
            raise ValueError(
                "the scene is missing tendon actuators " + ", ".join(missing)
                + " -- CONTRACT.md section 3 names them pam_1..pam_24, where pam_k "
                  "is board 0x100 + k")
        self._act_ids = np.asarray(act_ids, dtype=int)
        self._ten_ids = np.asarray(ten_ids, dtype=int)
        #: The compiler's qpos0 tendon lengths.  The anchored muscle length is
        #: ``l0_per_act + (ten_length - ten_len0)``, so this is the reference the
        #: force law is anchored to, and it has to be captured before anything
        #: moves.
        self._ten_len0 = np.array(self.model.tendon_length0[self._ten_ids],
                                  dtype=float)
        #: Captured before any pressure-scheduled overwrite.
        self._tendon_damping_base = np.array(
            self.model.tendon_damping[self._ten_ids], dtype=float)

    def _build_nodes(self, ids, absent, faults, tle_adc, seven_mm_adc,
                     tle_tuning, seven_mm_tuning, supply_psi,
                     leak_fault_pa_s) -> None:
        absent = {int(b) for b in absent}
        faults = dict(faults or {})
        self.nodes = {}
        for base in ids:
            base = int(base)
            if base in absent:
                continue
            variant = int(self.variants.get(base, P.VARIANT_7MM))
            tle = variant == P.VARIANT_TLE_DVP
            kwargs = dict(
                adc=tle_adc if tle else seven_mm_adc,
                fault=faults.get(base, NodeFault()),
                # One RNG stream per board, seeded by its own id, so a node's
                # sensor character is a function of (seed, id) alone.  Sharing
                # one stream would make each node's noise depend on how many
                # other nodes were built first, and C5's bit-identical racing
                # threads would then depend on construction order.
                rng=random.Random(self.seed + base),
                supply_psi=supply_psi,
                actuator=self.actuator,
                leak_fault_pa_s=leak_fault_pa_s,
            )
            if tle and tle_tuning is not None:
                kwargs["tuning"] = tle_tuning
            if not tle and seven_mm_tuning is not None:
                kwargs["tuning"] = seven_mm_tuning
            self.nodes[base] = make_node(base, variant, **kwargs)

    # -- the clock ------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def sim_now(self) -> float:
        if self._t_origin is None:
            return 0.0
        return self._t_origin + self.quanta_done * self._dt

    def advance_to(self, t: float) -> float:
        """Advance to ``t`` in whole quanta.  Callable from any thread.

        Four properties, and together they are the determinism contract:

        * the **first** call pins the origin and consumes nothing;
        * a target at or behind the furthest one seen is a **no-op**, which is
          what makes racing advancers order-independent;
        * the sub-quantum remainder **accumulates across calls**, so 1000 hops
          of 1.5 ms consume 1500 quanta and not 1000 -- per-hop flooring would
          lose a third of the wall time;
        * ``+ 1e-9`` quanta of forgiveness, so a float sum landing one ulp low
          does not consume ``n-1``.
        """
        t = float(t)
        with self.lock:
            if self._t_target is None:
                self._t_origin = t
                self._t_target = t
                return self.sim_now
            if t <= self._t_target:
                return self.sim_now
            self._accum += t - self._t_target
            self._t_target = t
            n = int(self._accum / self._dt + _QUANTUM_EPS)
            if n > 0:
                self._accum -= n * self._dt
                for _ in range(n):
                    self._step_quantum()
            return self.sim_now

    def _step_quantum(self) -> None:
        if self.quanta_done % self.node_logic_every == 0:
            self._node_pass()
        dlen = self.data.ten_length[self._ten_ids] - self._ten_len0
        self.data.ctrl[self._act_ids] = self.actuator.force_n(self.pressures_pa(), dlen)
        self._mujoco.mj_step(self.model, self.data)
        self.quanta_done += 1

    def _node_pass(self) -> None:
        """One firmware-grid pass, over every present node.

        ``node_dt`` is measured in consumed quanta rather than assumed to be the
        grid, so the **first** pass integrates ``dt = 0`` and no plant time is
        lost or double-counted.  A consequence worth knowing before validating
        any leak arithmetic against this: advancing ``T`` seconds integrates
        ``T - node_grid_s`` of plant, because the pass at quantum 0 moves
        nothing.
        """
        node_dt = (self.quanta_done - self._last_node_quantum) * self._dt
        self._last_node_quantum = self.quanta_done
        now = self.sim_now

        dlen = self.data.ten_length[self._ten_ids] - self._ten_len0
        l = self._l0_per_act + dlen
        ldot = self.data.ten_velocity[self._ten_ids]

        if self.batched_actuator and node_dt > 0.0:
            self._batch_flows(l, ldot)

        for node in self.nodes.values():
            i = node.index
            node.set_muscle_state(float(l[i]), float(ldot[i]))
            node.step(node_dt, now, self._sub_counts[node.control_hz])

        if self.tendon_damping_const is None:
            self.model.tendon_damping[self._ten_ids] = \
                self.actuator.tendon_damping_n_s_m(
                    self.pressures_pa(), base=self._tendon_damping_base, l_m=l)
        else:
            self.model.tendon_damping[self._ten_ids] = float(self.tendon_damping_const)

        self.node_passes += 1
        if self.node_hook is not None:
            self.node_hook(now)

    def _batch_flows(self, l, ldot) -> None:
        """One 24-row forward pass instead of twenty-four scalar ones.

        The per-node cache it fills is consumed by :meth:`SimNode._integrate`.
        ``stuck_vent`` therefore has to be applied at the seam and not here, or
        a fault would vanish the moment batching was switched on -- the
        reference's own port made that mistake once and it cost a debugging
        session.
        """
        p = self.pressures_pa()
        target = np.zeros(N_NODES, dtype=float)
        for node in self.nodes.values():
            target[node.index] = node.plant_target_pa
        flows = np.asarray(self.actuator.net_flow_pa_s_batch(p, target, l, ldot),
                           dtype=float)
        for node in self.nodes.values():
            node._net_flow_cache = float(flows[node.index])

    # -- the wire, on the bare-arm path ---------------------------------------
    def stage_targets(self, targets: dict) -> None:
        """Stage one runtime table.  ``{base: (counts, enable)}``, as transmitted.

        A board left out of the table is **not** turned off: it keeps whatever
        control byte it was last given, and the next sync edge promotes it again.
        ``backend.py::_build_targets`` addresses every known board every cycle
        for exactly this reason, and a caller that tables only a selection
        reproduces the real bus's worst failure -- a de-selected board regulating
        with nothing able to reach it, and the sync-loss failsafe unable to help
        because the master is still sending edges.
        """
        with self.lock:
            for base, entry in targets.items():
                node = self.nodes.get(int(base))
                if node is None:
                    continue
                counts, enable = entry
                node.stage_command(int(counts), bool(enable))

    def sync_edge(self, now: float | None = None) -> float:
        """The DLC-0 broadcast on ``0x090``: latch and promote, on every board."""
        with self.lock:
            t = self.sim_now if now is None else float(now)
            for node in self.nodes.values():
                node.sync_edge(t)
            return t

    def all_off(self) -> None:
        """Drop every board.  Traps the air it finds; it does not vent."""
        with self.lock:
            for node in self.nodes.values():
                node.all_off()

    # -- readout ---------------------------------------------------------------
    def pressures_pa(self) -> np.ndarray:
        """Plant truth, ``(24,)`` Pa.  An absent node reads exactly 0."""
        p = np.zeros(N_NODES, dtype=float)
        for node in self.nodes.values():
            p[node.index] = node.p_pa
        return p

    def psi_of(self, base: int) -> float:
        """The one unlocked read in this class -- a single float, truth psi.

        Everything else that touches ``mjData`` or node state takes the lock.  Do
        not add unlocked reads casually; this one is here because a status strip
        polls it at display rate, and on the reference rig taking the mutex for
        that measurably delayed the node grid.
        """
        node = self.nodes.get(int(base))
        return 0.0 if node is None else node.p_pa / PA_PER_PSI

    def q(self) -> np.ndarray:
        with self.lock:
            return np.array(self.data.qpos, dtype=float)

    def body_pose(self, name: str):
        mj = self._mujoco
        with self.lock:
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"no body named {name!r} in this scene")
            return (np.array(self.data.xpos[bid], dtype=float),
                    np.array(self.data.xquat[bid], dtype=float))

    def air_totals(self) -> dict:
        """Five ``(24,)`` arrays.  Changes no dynamics; every plant carries it."""
        out = {k: np.zeros(N_NODES, dtype=float)
               for k in ("air_in_psi", "air_out_psi", "inlet_open_s", "outlet_open_s")}
        out["valve_switches"] = np.zeros(N_NODES, dtype=int)
        for node in self.nodes.values():
            i = node.index
            out["air_in_psi"][i] = node.air_in_psi
            out["air_out_psi"][i] = node.air_out_psi
            out["inlet_open_s"][i] = node.inlet_open_s
            out["outlet_open_s"][i] = node.outlet_open_s
            out["valve_switches"][i] = node.valve_switches
        return out


# ---------------------------------------------------------------------------
# Dependencies another agent is writing.  CONTRACT.md sections 2 and 3.
# ---------------------------------------------------------------------------

_DEPENDENCY_HINT = (
    "digital_twin.{mod} is still the documented skeleton, so SimArm cannot "
    "build {arg}= for you. Pass {arg}= explicitly, or use "
    "digital_twin.sim_core.{fallback} for bring-up -- which is a stand-in and "
    "not this arm.")


def _generated_xml(**tunables) -> str:
    from . import mjcf_generator

    try:
        return mjcf_generator.generate_xml(**tunables)
    except NotImplementedError as exc:
        raise NotImplementedError(
            _DEPENDENCY_HINT.format(mod="mjcf_generator", arg="xml",
                                    fallback="placeholder_xml()")) from exc


def _default_actuator(variants: dict, ids, *, seed: int):
    from . import actuator_model

    is_tle = np.zeros(N_NODES, dtype=bool)
    for base in ids:
        idx = int(base) - ACTUATOR_FIRST
        if 0 <= idx < N_NODES:
            is_tle[idx] = (int(variants.get(int(base), P.VARIANT_7MM))
                           == P.VARIANT_TLE_DVP)
    try:
        return actuator_model.ActuatorModel.fresh(is_tle=is_tle, seed=seed)
    except (NotImplementedError, AttributeError) as exc:
        raise NotImplementedError(
            _DEPENDENCY_HINT.format(mod="actuator_model", arg="actuator",
                                    fallback="placeholder_actuator()")) from exc


# ---------------------------------------------------------------------------
# Bring-up stand-ins.  NEVER reached implicitly.
# ---------------------------------------------------------------------------

def placeholder_xml(*, joint_damping: float = 0.026,
                    joint_frictionloss: float = 0.025,
                    tendon_damping: float = 1.0,
                    link_length_m: float = 0.23,
                    link_radius_m: float = 0.030,
                    moment_arm_m: float = 0.045,
                    link_density: float = 260.0,
                    joint_range_rad: float = 1.2,
                    timestep_s: float = 0.001,
                    force_range: str = "-4000 0",
                    base_pos: tuple = (0.0, 0.0, 1.25)) -> str:
    """A six-u-joint chain with 24 tendon actuators, for bring-up ONLY.

    THIS IS NOT THIS ARM.  The link lengths are uniform, the plates are absent,
    the inertias are a capsule's, and the routing is a clean orthogonal pair per
    joint rather than the measured manifold.  It exists so that
    :class:`SimArm`'s clock, node grid and plant seam can be tested while
    ``digital_twin.mjcf_generator`` is still a skeleton, and :class:`SimArm`
    never reaches it on its own -- a caller has to name it.  Any number measured
    on this geometry describes this function.

    What it does reproduce, because those are what :class:`SimArm` binds
    against: twenty-four actuators named ``pam_1``..``pam_24`` where ``pam_k`` is
    board ``0x100 + k``, routed onto the twelve joints by
    ``UMArm_KINEMATICS.canarm_actuators.MEASURED_JOINT_PAIRS`` so the
    positive/negative sense of each pair is the measured one; a 1 ms timestep
    with ``implicitfast``; and a ``-4000 0`` force range so the pull-only clip
    is the same clip :mod:`digital_twin.replay` asserts against.
    """
    from UMArm_KINEMATICS.canarm_actuators import joint_pairs

    pairs = joint_pairs()
    # azimuth index -> (x, y) unit offset.  Joint 2i turns about y and is driven
    # by the +x / -x pair; joint 2i+1 turns about x and is driven by +y / -y.
    azimuth = {0: (1.0, 0.0), 1: (-1.0, 0.0), 2: (0.0, 1.0), 3: (0.0, -1.0)}
    site_of = {}
    for j, (pos_base, neg_base) in enumerate(pairs):
        uj, axis = j // 2, j % 2
        site_of[pos_base] = (uj, 0 if axis == 0 else 2)
        site_of[neg_base] = (uj, 1 if axis == 0 else 3)

    L, R, A = link_length_m, link_radius_m, moment_arm_m
    body_open, body_close = [], []
    for uj in range(6):
        anchor_z = 0.0 if uj == 0 else -L
        insertions = "".join(
            f'      <site name="b{uj}_{k}" pos="{A * ux:.5f} {A * uy:.5f} '
            f'-0.02" size="0.004"/>\n'
            for k, (ux, uy) in azimuth.items())
        body_open.append(
            f'    <body name="seg{uj}" pos="0 0 {anchor_z:.5f}">\n'
            f'      <joint name="q{2 * uj}" type="hinge" axis="0 1 0" '
            f'limited="true" range="{-joint_range_rad} {joint_range_rad}" '
            f'damping="{joint_damping}" frictionloss="{joint_frictionloss}"/>\n'
            f'      <joint name="q{2 * uj + 1}" type="hinge" axis="1 0 0" '
            f'limited="true" range="{-joint_range_rad} {joint_range_rad}" '
            f'damping="{joint_damping}" frictionloss="{joint_frictionloss}"/>\n'
            f'      <geom type="capsule" fromto="0 0 0 0 0 {-L:.5f}" '
            f'size="{R}" density="{link_density}"/>\n'
            f'{insertions}')
        body_close.append('    </body>\n')

    # The anchor sites for u-joint ``uj`` live on its PARENT body, so they are
    # emitted before the child body opens.
    nested = ""
    for uj in reversed(range(6)):
        anchors = "".join(
            f'      <site name="a{uj}_{k}" pos="{A * ux:.5f} {A * uy:.5f} '
            f'{(0.0 if uj == 0 else -L) + 0.02:.5f}" size="0.004"/>\n'
            for k, (ux, uy) in azimuth.items())
        nested = anchors + body_open[uj] + nested + body_close[uj]

    tendons, actuators = [], []
    for base in ALL_IDS:
        k = base - 0x100
        uj, site = site_of[base]
        tendons.append(
            f'    <spatial name="t_pam_{k}" damping="{tendon_damping}" '
            f'width="0.002">\n'
            f'      <site site="a{uj}_{site}"/>\n'
            f'      <site site="b{uj}_{site}"/>\n'
            f'    </spatial>\n')
        actuators.append(
            f'    <motor name="pam_{k}" tendon="t_pam_{k}" gear="1" '
            f'ctrlrange="{force_range}" forcerange="{force_range}"/>\n')

    bx, by, bz = base_pos
    return (
        '<mujoco model="canarm_placeholder">\n'
        f'  <option timestep="{timestep_s}" integrator="implicitfast" '
        f'iterations="100" gravity="0 0 -9.81"/>\n'
        '  <worldbody>\n'
        f'    <body name="canarm_base" pos="{bx} {by} {bz}" quat="0 1 0 0">\n'
        + nested +
        '    </body>\n'
        '  </worldbody>\n'
        '  <tendon>\n' + "".join(tendons) + '  </tendon>\n'
        '  <actuator>\n' + "".join(actuators) + '  </actuator>\n'
        '</mujoco>\n')


class placeholder_actuator:                       # noqa: N801 - a factory, named
    """A closed-form stand-in for :class:`digital_twin.actuator_model.ActuatorModel`.

    THIS IS NOT A FIT.  It implements CONTRACT.md section 2's method surface with
    smooth expressions chosen so that a fill to 20 psi takes a few hundred
    milliseconds and a McKibben pulls rather than pushes -- enough to exercise
    :class:`SimArm`'s clock, seam and force path, and nothing more.  Two things
    it deliberately does **not** have, and both matter when a real checkpoint
    arrives:

    * its closed branch is exactly zero at ``e = 0``.  A fitted net's is not,
      and a trapped rack walks toward that branch's own fixed point in a
      direction that depends on the checkpoint.  No test written against this
      class may assume a trapped rack holds still.
    * its per-node gains are uniform, so it cannot show the per-board spread
      that CONTRACT.md keeps ``fill_gain`` and ``vent_gain`` per node for.
    """

    #: Contract section 2's normalisation constants, so a checkpoint's unit
    #: guard and this stand-in cannot silently disagree about the scale.
    P_SCALE_PA = 30.0 * PA_PER_PSI
    E_SCALE_PA = 30.0 * PA_PER_PSI
    DP_SCALE_PA_S = 1.0e5
    FORCE_CLIP_N = 4000.0

    def __init__(self, *, is_tle=None, fill_gain: float = 5.0,
                 vent_gain: float = 5.0, leak_pa_s: float = 700.0,
                 blend_width_pa: tuple = (2000.0, 6000.0),
                 coeff: float = 0.035, bf: float = 0.36, l0: float = 0.24,
                 damp_b1: float = 9.9e-4) -> None:
        n = N_NODES
        self.is_tle = (np.zeros(n, dtype=bool) if is_tle is None
                       else np.asarray(is_tle, dtype=bool))
        self.fill_gain = np.full(n, float(fill_gain))
        self.vent_gain = np.full(n, float(vent_gain))
        self.leak_pa_s = np.full(n, float(leak_pa_s))
        self.blend_width_pa = np.asarray(blend_width_pa, dtype=float)
        self.l0_per_act = np.full(n, float(l0))
        self.coeff_per_act = np.full(n, float(coeff))
        self.bf2_per_act = np.full(n, float(bf) ** 2)
        self.damp_b1_per_act = np.full(n, float(damp_b1))

    # -- the gain blend, verbatim from CONTRACT.md section 2 -----------------
    def gain(self, node_idx, e_pa):
        pop = self.is_tle[node_idx].astype(int)
        width = self.blend_width_pa[pop]
        s = 1.0 / (1.0 + np.exp(-np.asarray(e_pa, dtype=float) / width))
        return s * self.fill_gain[node_idx] + (1.0 - s) * self.vent_gain[node_idx]

    # -- flow ----------------------------------------------------------------
    def net_flow_pa_s(self, node_id, p_pa, target_pa, l_m, ldot_m_s) -> float:
        idx = np.asarray([int(node_id) - 1])
        e = np.asarray([float(target_pa) - float(p_pa)])
        return float(self._net(idx, e)[0])

    def net_flow_pa_s_batch(self, p_pa, target_pa, l_m, ldot_m_s) -> np.ndarray:
        idx = np.arange(N_NODES)
        e = np.asarray(target_pa, dtype=float) - np.asarray(p_pa, dtype=float)
        return self._net(idx, e)

    def _net(self, idx, e_pa) -> np.ndarray:
        return self.gain(idx, e_pa) * self.DP_SCALE_PA_S * np.tanh(
            e_pa / self.E_SCALE_PA)

    def flow_pa_s(self, node_id, p_pa, target_pa, l_m, ldot_m_s) -> float:
        return (self.net_flow_pa_s(node_id, p_pa, target_pa, l_m, ldot_m_s)
                - float(self.leak_pa_s[int(node_id) - 1]))

    # -- force ----------------------------------------------------------------
    def force_n(self, p_pa, dlen_m) -> np.ndarray:
        l = self.l0_per_act + np.asarray(dlen_m, dtype=float)
        f = self.coeff_per_act * np.asarray(p_pa, dtype=float) * (
            self.bf2_per_act - 3.0 * l * l)
        return np.clip(f, -self.FORCE_CLIP_N, 0.0)

    def tendon_damping_n_s_m(self, p_pa, base, l_m) -> np.ndarray:
        return np.asarray(base, dtype=float) + self.damp_b1_per_act * np.asarray(
            p_pa, dtype=float)
