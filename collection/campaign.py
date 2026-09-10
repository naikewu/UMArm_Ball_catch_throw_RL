"""Drive the CAN arm through a designed excitation and record everything.

This is the hardware session.  It owns the bus, the mocap receiver, the camera
and the recording, walks :mod:`collection.excitation`'s segment list, and
watches for the one class of fault that is invisible in the numbers alone: a
commanded pressure that arrives at the board while the joint it should move
does not move.

Three things are non-negotiable and are implemented as such rather than as
conventions.

**The pressure rule.** Every target vector passes :meth:`PairEnvelope.safe`,
which clamps and then asserts against the operator's hard ceilings.  The
assertion is checked again in :meth:`Campaign._on_cycle` immediately before
the bulk write, so no code path reaches the bus unchecked.

**Unused boards stay enabled at idle.** ``0x110`` leaks from its supply side and
a disabled board stops venting it, which would put a slow ramp underneath every
measurement.  The only place the enable bits go down deliberately is the leak
probe, where shutting both valves *is* the measurement.

**Every exit path vents first, then stops the cycle.**  Clearing a 7 mm board's
enable bit does **not** de-energise its solenoids.  ``firmware/legacy/main/main.c``
skips the whole control body when the enable bit is clear, and all three
``gpio_set_level(V_IN/V_OUT, ...)`` calls sit below that branch, so a disable
*freezes* the valves where they were; that tree also has no link-loss timeout,
so a board disabled while inflating goes on inflating with nothing able to stop
it but a fresh enabled frame or power.

Measured on this arm on 2026-09-10: all twenty-four boards charged to 12 psi and
then disabled for 8 s, twice.  The per-board drift repeated to within 3 % between
the two traps, and it ranged from -0.01 to -1.08 psi/s -- two orders of
magnitude, which a leak alone does not explain.  ``0x109``, ``0x113`` and
``0x117`` emptied to 0.1-4.1 psi, i.e. they froze with the **exhaust open**;
``0x110`` rose at +0.041 psi/s, the known supply-side leak.  None happened to
freeze inflating, but that is the sample, not a guarantee.

So :meth:`Campaign._vent_and_disable` commands zero and *waits for the arm to
empty* before any enable bit goes down.  The last command a board sees is then a
step down, which leaves it venting or in its deadband -- both safe directions to
freeze in.  This is the same rule the RS485 twin's protocol states as "vent past
the closed branch's fixed point BEFORE shutting the valves".

Usage::

    .venv\\Scripts\\python.exe -m collection.campaign --out-root data
    .venv\\Scripts\\python.exe -m collection.campaign --scale 0.05 --dry-run
    .venv\\Scripts\\python.exe -m collection.campaign --phases rest,supply_probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

import numpy as np

_WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_WS, os.path.join(_WS, "TLE_PCB"),
           os.path.join(_WS, "UMArm_MOCAP", "natnet_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_env  # noqa: E402
from tlelib import proto as P  # noqa: E402
from tlelib.backend import Backend  # noqa: E402
from tlelib.timing import fine_gil_handoff, hires_clock  # noqa: E402
from UMArm_CAMERA import BenchCamera  # noqa: E402
from UMArm_KINEMATICS import canarm_actuators as ACT  # noqa: E402
from UMArm_MOCAP.canarm_mocap import CanArmMocap  # noqa: E402

from . import excitation as EX  # noqa: E402
from .recorder import Recorder  # noqa: E402
from .safety import ALL_BASES, BASE_INDEX, PairEnvelope  # noqa: E402

#: Nominal rate the excitation is evaluated and pushed at.  It is not the
#: period of a timer: the excitation is evaluated **inside the cycle observer**,
#: once per sync edge, so this is the bus's own rate by construction.
#:
#: The first version of this file paced a separate drive thread with
#: ``time.sleep`` and it cost 9 Hz of cycle rate -- 140.7 Hz measured over a
#: 28173-row rehearsal, with dt reaching 42.7 ms -- because a Python thread
#: waking on a 6.67 ms period competes with the cycle thread for the GIL and
#: Windows' scheduler grants neither of them the resolution that period needs.
#: Driving from the observer removes the thread, removes the clock skew between
#: the excitation and the table it lands in, and puts every target on a real
#: sync edge instead of near one.
DRIVE_HZ = 150.0

#: Seconds of camera clip taken at each trigger, and how often an untriggered
#: clip is taken anyway.  A clip is about 4 MB, so a fifty-minute session with a
#: clip every two minutes lands near 100 MB -- enough to see the arm through the
#: whole campaign without the video outweighing the data it documents.
CLIP_S = 12.0
CLIP_EVERY_S = 120.0
CLIP_PRE_S = 3.0


class AnomalyWatcher:
    """Flags a commanded motion that the arm did not make.

    The failure this exists for is specific: the pneumatics work, the board
    reports the pressure it was asked for, and the joint does not move.  On this
    arm that is a disconnected or burst muscle, a fouled linkage, or a marker
    plate that Motive has lost — and all three look identical in the pressure
    log, which is why the camera is the instrument that resolves them.

    The test is deliberately conservative.  It fires only when the commanded
    *differential* on a pair changed by more than ``dpsi_min``, the measured
    pressures followed it to within ``follow_tol``, and the joint's angle then
    moved less than ``dq_min``.  Requiring the pressure to have followed is what
    separates this from the ordinary case of a board that simply could not
    reach its target, which is a different finding and is counted separately.

    What it does **not** catch: a joint that moves the wrong way, or by the
    wrong amount.  Both need the fitted model to judge, and the point of this
    watcher is to flag things during the session, while the camera can still be
    pointed at them.
    """

    def __init__(self, env: PairEnvelope, *, dpsi_min: float = 6.0,
                 dq_min_deg: float = 0.8, follow_tol_psi: float = 2.5,
                 settle_s: float = 1.2, cooldown_s: float = 25.0):
        self.env = env
        self.dpsi_min = dpsi_min
        self.dq_min = np.radians(dq_min_deg)
        self.follow_tol = follow_tol_psi
        self.settle_s = settle_s
        self.cooldown_s = cooldown_s
        self.events: list = []
        self._pending = None
        self._last_fire = -1e9

    def observe(self, t: float, cmd_psi, meas_psi, q, segment: str) -> dict | None:
        """Feed one sample; returns an event dict when one fires."""
        if q is None:
            return None
        cmd = np.asarray(cmd_psi, float)
        meas = np.asarray(meas_psi, float)
        q = np.asarray(q, float)
        a = self.env.pair_idx[:, 0]
        b = self.env.pair_idx[:, 1]
        diff = cmd[a] - cmd[b]

        if self._pending is None:
            self._pending = {"t": t, "diff": diff.copy(), "q": q.copy(),
                             "cmd": cmd.copy(), "segment": segment}
            return None

        p = self._pending
        if t - p["t"] < self.settle_s:
            return None

        ddiff = diff - p["diff"]
        dq = q - p["q"]
        # Only judge joints whose command actually moved, and only when the
        # boards did what they were told.
        moved_cmd = np.abs(ddiff) >= self.dpsi_min
        followed = (np.abs(meas[a] - cmd[a]) <= self.follow_tol) & \
                   (np.abs(meas[b] - cmd[b]) <= self.follow_tol)
        stuck = moved_cmd & followed & (np.abs(dq) < self.dq_min)
        self._pending = {"t": t, "diff": diff.copy(), "q": q.copy(),
                         "cmd": cmd.copy(), "segment": segment}
        if not np.any(stuck) or (t - self._last_fire) < self.cooldown_s:
            return None
        self._last_fire = t
        joints = [int(j) for j in np.nonzero(stuck)[0]]
        ev = {
            "kind": "commanded_but_still",
            "t_s": round(float(t), 3),
            "segment": segment,
            "joints": joints,
            "joint_names": [ACT.JOINT_NAMES[j] for j in joints],
            "boards": [[f"0x{self.env.pairs[j][0]:03X}",
                        f"0x{self.env.pairs[j][1]:03X}"] for j in joints],
            "commanded_differential_change_psi":
                [round(float(ddiff[j]), 2) for j in joints],
            "joint_change_deg": [round(float(np.degrees(dq[j])), 3)
                                 for j in joints],
        }
        self.events.append(ev)
        return ev


class Campaign:
    """One hardware session, from port open to port release."""

    def __init__(self, args):
        self.args = args
        self.env = PairEnvelope(pair_sum_max_psi=args.pair_sum_max,
                                single_max_psi=args.single_max)
        self.segments = self._build_segments()
        self.be = None
        self.rx = None
        self.cam = None
        self.rec = None
        self.watcher = AnomalyWatcher(self.env)
        self._cmd_psi = self.env.idle_vector()
        self._stop = threading.Event()
        self._seg_lock = threading.Lock()
        self._seg = None
        self._seg_t0 = 0.0
        self._enable_all = True
        self._meas_psi = np.zeros(24, dtype=float)
        self._last_rows = 0
        self._cal_zero = np.zeros(24, dtype=float)
        self._cal_scale = np.ones(24, dtype=float)
        self.notes: list = []
        self.clips: list = []
        self.stats = {"drive_ticks": 0, "drive_late": 0}

    # -- plan --------------------------------------------------------------- #

    def _build_segments(self) -> list:
        env, a = self.env, self.args
        if a.phases:
            wanted = [w.strip() for w in a.phases.split(",") if w.strip()]
            segs = []
            for w in wanted:
                fn = getattr(EX, w, None)
                if fn is None:
                    raise SystemExit(f"unknown phase '{w}'")
                segs += fn(env) if w not in ("random_walk", "ringdowns",
                                             "validation") \
                    else fn(env, seed=a.seed)
            return segs
        return EX.full_campaign(env, seed=a.seed, scale=a.scale)

    # -- the drive thread --------------------------------------------------- #

    def _on_cycle(self, t_sync, targets, replies) -> None:
        """The one per-cycle observer: record this edge, then command the next.

        Both jobs live here because both want exactly this clock.  The
        recording has to be stamped at the sync edge, since that is the only
        instant the whole arm agrees on, and the excitation has to be evaluated
        against the same edge or the recorded target and the recorded pressure
        belong to different times.

        Everything in this path is bounded: the recorder appends one tuple, the
        segment function is a table lookup or a dozen operations on 12-element
        arrays, and ``set_targets`` takes the backend lock once.  Nothing here
        formats, opens a file, or waits.
        """
        try:
            self._observe(t_sync, targets, replies)
        except Exception:
            # NOTHING in this path may escape unrecorded.  Backend uninstalls
            # an observer that raises, which takes the recording AND the
            # excitation down together: on 2026-09-10 exactly that happened
            # 83 s into a session and the campaign walked another four minutes
            # of excitation into a file nobody was writing.  The whole body
            # therefore sits inside this one handler rather than having the
            # safety-critical part inside a narrower one -- the first version
            # guarded only the target generation, and the fault was elsewhere.
            self.rec.observer_error = traceback.format_exc()
            self._stop.set()
            try:
                self.be.set_targets({b: self.env.idle_psi for b in ALL_BASES})
            except Exception:
                pass
            raise

    def _observe(self, t_sync, targets, replies) -> None:
        self.rec.on_cycle(t_sync, targets, replies)

        # The measured pressures the anomaly watcher needs, taken from the
        # replies already in hand rather than from snapshot_nodes(), which
        # deep-copies twenty-four dataclasses under the lock the cycle thread
        # is about to need again.
        meas = self._meas_psi
        for k, base in enumerate(ALL_BASES):
            rep_k = replies.get(base)
            if rep_k is not None:
                meas[k] = self._cal_scale[k] * (rep_k[1].counts - self._cal_zero[k])

        seg = self._seg
        if seg is None or self._stop.is_set():
            return
        psi = seg.fn(t_sync - self._seg_t0)
        psi = self.env.safe(psi, where=f"segment {seg.name}")
        # The second assertion the module docstring promises: nothing reaches
        # the bus without it, including a target from a generator that forgot
        # to clamp.  A failure propagates to :meth:`_on_cycle`'s handler, which
        # stops the session and idles the arm in this same cycle.
        self.env.assert_safe(psi, where="pre-transmit")
        self._cmd_psi = psi
        self.be.set_targets({b: float(psi[i]) for i, b in enumerate(ALL_BASES)})
        self.stats["drive_ticks"] += 1

    # -- helpers ------------------------------------------------------------ #

    def _measured_psi(self) -> np.ndarray:
        """The latest reported pressures in psi, without touching the backend.

        Filled by :meth:`_on_cycle` from the replies it already holds.  Reading
        them through ``snapshot_nodes()`` would deep-copy twenty-four
        dataclasses under the cycle thread's own lock, which is the cost that
        took 15 Hz off this arm when the operator GUI did it for a plot.
        """
        return self._meas_psi.copy()

    def _vent_and_disable(self, *, timeout_s: float = 12.0,
                          floor_psi: float = 1.5) -> dict:
        """Command zero, wait for the arm to empty, and only then disable.

        Returns what it observed, for the session report.  See the module
        docstring: a 7 mm board freezes its solenoids where the disable found
        them, so the only safe moment to clear an enable bit is one where every
        board is venting or shut.  Commanding zero and waiting guarantees that;
        clearing the bit straight from a hold does not.

        The timeout is not a failure path.  A board that will not empty --
        ``0x110`` inflates from its supply side and never will -- is reported
        and left; waiting longer would not help, and refusing to disable would
        leave the whole arm regulating instead.
        """
        if self.be._on_cycle is None:          # noqa: SLF001 - see below
            # Without the observer nothing updates _meas_psi and the wait below
            # is a loop over a constant.  Refuse rather than report a vent that
            # was never observed.
            raise RuntimeError(
                "_vent_and_disable needs the cycle observer installed; without "
                "it the pressures it waits on never change")
        self.be.set_targets({b: 0.0 for b in ALL_BASES})
        self.be.set_enabled_all(True)
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            time.sleep(0.1)
            if float(np.nanmax(self._meas_psi)) <= floor_psi:
                break
        held = {f"0x{b:03X}": round(float(self._meas_psi[i]), 2)
                for i, b in enumerate(ALL_BASES)
                if self._meas_psi[i] > floor_psi}
        self._set_enable_all(False)
        if held:
            print("campaign: vented before disabling; still above "
                  f"{floor_psi} psi: {held}")
        else:
            print(f"campaign: arm vented below {floor_psi} psi, then disabled")
        return {"seconds_waited": round(timeout_s - (deadline - time.perf_counter()), 2),
                "still_pressurised_psi": held}

    def _set_enable_all(self, on: bool) -> None:
        # set_enabled_all takes the lock once and logs once; the per-board call
        # would take it twenty-four times and print twenty-four lines.
        self.be.set_enabled_all(on)
        self._enable_all = on

    def _clip(self, tag: str, *, post_s: float = CLIP_S,
              pre_s: float = CLIP_PRE_S) -> None:
        if self.cam is None or not self.cam.state.opened:
            return
        path = os.path.join(self.media_dir, f"{tag}.mp4")
        if self.cam.clip(path, pre_s=pre_s, post_s=post_s, label=tag):
            self.clips.append(path)

    # -- the session -------------------------------------------------------- #

    def run(self) -> int:
        a = self.args
        total = EX.total_seconds(self.segments)
        print(f"campaign: {len(self.segments)} segments, "
              f"{total:.0f} s ({total / 60.0:.1f} min) of excitation")
        kinds = {}
        for sg in self.segments:
            kinds[sg.kind] = kinds.get(sg.kind, 0.0) + sg.duration_s
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<18} {v:7.1f} s")
        if a.dry_run:
            self._dry_run()
            return 0

        # Before any of the other threads exist.  With CPython's default 5 ms
        # switch interval this stack runs the cycle at 143 Hz and skips 4.7 %
        # of periods outright; at 0.5 ms it runs at 150.00 Hz and skips none.
        # See tlelib.timing.fine_gil_handoff for the measurement.
        hires_clock()
        old_swi = fine_gil_handoff()
        print(f"campaign: GIL switch interval {old_swi * 1e3:.1f} ms -> "
              f"{sys.getswitchinterval() * 1e3:.2f} ms")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(a.out_root, f"session_{stamp}")
        self.media_dir = os.path.join(self.session_dir, "media")
        os.makedirs(self.media_dir, exist_ok=True)
        print(f"campaign: session -> {self.session_dir}")

        self.cam = BenchCamera(enabled=not a.no_camera)
        cam_up = self.cam.start()
        print(f"campaign: camera {'up' if cam_up else 'UNAVAILABLE: ' + (self.cam.state.error or 'disabled')}")

        self.rx = CanArmMocap(server_ip=a.server_ip, client_ip=a.client_ip,
                              ring_capacity=8000, marker_ring_capacity=4000)
        self.rx.start()
        try:
            deadline = time.monotonic() + 6.0
            while self.rx.get_state().frames < 20 and time.monotonic() < deadline:
                time.sleep(0.1)
            st = self.rx.get_state()
            if st.frames < 20:
                print("campaign: no mocap frames; refusing to drive blind")
                return 1
            print(f"campaign: mocap up, {st.fps:.1f} Hz")

            port = bench_env.resolve_can_port(a.port)
            # Without a sink the Backend's errors go only to a deque nobody
            # reads, which is how a dead cycle thread stayed invisible.
            self.be = Backend(port=port, bitrate=bench_env.BITRATE,
                              log=lambda m: print("   backend: " + m, flush=True))
            self.be.open()
            try:
                return self._run_bus()
            finally:
                try:
                    self.be.stop_cycle()
                finally:
                    self.be.close()
                    print("campaign: bus stopped, every enable bit clear, "
                          "port closed")
        finally:
            self.rx.stop()
            if self.cam is not None:
                self.cam.drain(timeout_s=60.0)
                self.cam.stop()

    def _run_bus(self) -> int:
        a = self.args
        found = self.be.scan()
        present = sorted(b for b, n in found.items() if n.present)
        if len(present) != 24:
            print(f"campaign: {len(present)} of 24 boards answered; stopping")
            return 1
        board_type = []
        cals = {}
        for b in ALL_BASES:
            node = found[b]
            variant = (P.VARIANT_TLE_DVP if node.kind ==
                       P.VARIANT_NAMES[P.VARIANT_TLE_DVP] else P.VARIANT_7MM)
            board_type.append(int(variant))
            cals[b] = node.cal
            i = BASE_INDEX[b]
            self._cal_zero[i] = node.cal.zero_counts
            self._cal_scale[i] = 1.0 / node.cal.counts_per_psi
        n_tle = sum(1 for v in board_type if v == P.VARIANT_TLE_DVP)
        print(f"campaign: 24 boards, {n_tle} TLE/DVP, {24 - n_tle} 7 mm")

        meta = {
            "session_tag": a.tag,
            "actuator_pairs_measured": [[f"0x{p:03X}", f"0x{n:03X}"]
                                        for p, n in self.env.pairs],
            "actuator_pairs_source": ACT.MEASURED_SOURCE,
            "joint_names": list(ACT.JOINT_NAMES),
            "single_actuator_limit_psi": self.env.single_max_psi,
            "pair_sum_limit_psi": self.env.pair_sum_max_psi,
            "idle_psi": self.env.idle_psi,
            "drive_hz": DRIVE_HZ,
            "known_board_faults": {f"0x{k:03X}": v
                                   for k, v in ACT.KNOWN_BOARD_FAULTS.items()},
            "mocap_rigid_ids": list(range(2000, 2006)),
            "segments_planned": [
                {"name": sg.name, "kind": sg.kind, "duration_s": sg.duration_s,
                 "meta": sg.meta,
                 "driven": [f"0x{b:03X}" for b in sg.driven]}
                for sg in self.segments],
        }
        self.rec = Recorder(self.session_dir, ids=ALL_BASES,
                            board_type=board_type, cals=cals, mocap=self.rx,
                            metadata=meta)

        self.be.select(list(ALL_BASES))
        for b in ALL_BASES:
            self.be.set_target(b, 0.0)
            self.be.set_enabled(b, False)
        self.be.start_cycle()
        time.sleep(0.5)
        self._set_enable_all(True)
        self.be.set_targets({b: self.env.idle_psi for b in ALL_BASES})
        time.sleep(2.0)

        self.rec.start()
        with self._seg_lock:
            self._seg = None
            self._seg_t0 = time.perf_counter()
        self.be.set_cycle_observer(self._on_cycle)
        t_session = time.perf_counter()
        last_clip = -1e9
        rc = 0
        try:
            for idx, sg in enumerate(self.segments):
                if self._runaway_guard():
                    print("campaign: STOPPING -- a board ran away while "
                          "disabled")
                    rc = 1
                    break
                why = self._watchdog(idx)
                if why:
                    print("campaign: STOPPING -- " + why)
                    self.notes.append("watchdog: " + why)
                    rc = 1
                    break
                if self._stop.is_set():
                    print("campaign: the cycle observer stopped the session")
                    self.notes.append("observer stop: "
                                      + (self.rec.observer_error or "safety"))
                    rc = 1
                    break
                with self._seg_lock:
                    self._seg = sg
                    self._seg_t0 = time.perf_counter()
                self.rec.segment = sg.name
                self.rec.segment_kind = sg.kind
                self.rec.segment_index = idx

                # The leak probe is the one phase that deliberately drops the
                # enable bits: both valves shut is what makes the drift a leak.
                want_enabled = not bool(sg.meta.get("disable_all"))
                if want_enabled != self._enable_all:
                    self._set_enable_all(want_enabled)

                now = time.perf_counter()
                if (now - last_clip) > CLIP_EVERY_S or sg.kind == "validation":
                    tag = f"{idx:04d}_{sg.name}"
                    self._clip(tag, post_s=(sg.duration_s if sg.kind ==
                                            "validation" else CLIP_S))
                    last_clip = now

                deadline = now + sg.duration_s
                next_watch = 0.0
                while time.perf_counter() < deadline and not self._stop.is_set():
                    time.sleep(0.02)
                    # snapshot_nodes() deep-copies twenty-four dataclasses under
                    # the lock the cycle thread needs to publish, and that cost
                    # is exactly what took 15 Hz off this arm's cycle rate when
                    # the operator GUI did it for a plot.  Four times a second
                    # is plenty to catch a joint that is not moving at all.
                    tnow = time.perf_counter()
                    if tnow >= next_watch:
                        next_watch = tnow + 0.25
                        self._watch(tnow - t_session, sg)

                if (idx % 25) == 0 or idx == len(self.segments) - 1:
                    el = time.perf_counter() - t_session
                    print(f"  [{el / 60.0:5.1f} min] seg {idx + 1}/"
                          f"{len(self.segments)} {sg.name:<24} "
                          f"rows={self.rec.rows_written} "
                          f"anomalies={len(self.watcher.events)}")
        except KeyboardInterrupt:
            print("campaign: interrupted by operator")
            rc = 130
        finally:
            self._stop.set()
            # The observer stays installed through the vent.  It is what
            # fills _meas_psi, so uninstalling it first leaves the wait loop
            # reading an array frozen at the last commanded pose -- a check
            # that can only ever time out and report the pose it started from,
            # which is exactly what the 2026-09-10 session reported.
            self.vent_report = self._vent_and_disable()
            self.be.set_cycle_observer(None)
            if self.rec is not None:
                self.rec.close()
            self._write_session_report(t_session)
        return rc

    #: A board that climbs past this while its enable bit is clear has frozen
    #: with its inlet open, which the firmware gives no way to stop except a
    #: fresh enabled frame.  Twelve psi over the leak probe's 12 psi charge is
    #: far outside the natural drift measured on 2026-09-10 (worst +0.041 psi/s
    #: on 0x110, so 0.33 psi over an 8 s trap) and still well inside the
    #: operator's 30 psi rule.
    RUNAWAY_PSI = 24.0

    def _runaway_guard(self) -> bool:
        """True if a disabled board is inflating, after re-enabling the arm.

        The recovery is the point: re-enabling restores the regulator, which is
        the only thing that can shut a frozen inlet valve short of power.
        """
        if self._enable_all:
            return False
        if float(np.nanmax(self._meas_psi)) < self.RUNAWAY_PSI:
            return False
        bad = {f"0x{b:03X}": round(float(self._meas_psi[i]), 2)
               for i, b in enumerate(ALL_BASES)
               if self._meas_psi[i] >= self.RUNAWAY_PSI}
        self.notes.append(f"RUNAWAY while disabled: {bad}")
        self.be.set_targets({b: self.env.idle_psi for b in ALL_BASES})
        self._set_enable_all(True)
        return True

    #: How many times a dead cycle may be revived before the session gives up.
    #: The fault this exists for is the USB-CAN dongle's, not the arm's: on
    #: 2026-09-10 a 33-minute campaign was cut off at 14.2 min by
    #: ``WriteFile failed (OSError(22, 'The operation completed successfully.'))``
    #: from the CDC driver -- a write that both failed and reported success,
    #: which is a driver hiccup and not a bus problem.  Every board answered
    #: normally up to that cycle and again afterwards.  Three attempts, because
    #: a fault that survives three reopens is not a hiccup.
    MAX_BUS_REVIVALS = 3

    def _revive_bus(self) -> bool:
        """Reopen the adapter and restart the cycle, mid-session.

        Returns whether the cycle is running again.  The recording is *not*
        restarted: it keeps the same files and the same ``t0``, so the gap
        appears in ``can_sync_time_s`` as the gap it was rather than being
        papered over -- a reader can see exactly which cycles are missing.

        While the cycle is down the arm is not being commanded.  The 7 mm
        boards go on regulating their last target indefinitely (no link-loss
        timeout) and the TLE boards drop their outputs after 500 ms, so the arm
        holds part of a pose; the first thing this does after the cycle is back
        is command idle, before the excitation resumes.
        """
        self.stats["bus_revivals"] = self.stats.get("bus_revivals", 0) + 1
        n = self.stats["bus_revivals"]
        print(f"campaign: reviving the bus, attempt {n}/{self.MAX_BUS_REVIVALS}")
        try:
            self.be.set_cycle_observer(None)
        except Exception:
            pass
        try:
            self.be.close()
        except Exception:
            pass
        time.sleep(1.5)
        try:
            port = bench_env.resolve_can_port(self.args.port)
            self.be = Backend(port=port, bitrate=bench_env.BITRATE,
                              log=lambda m: print("   backend: " + m, flush=True))
            self.be.open()
            found = self.be.scan()
            present = sorted(b for b, nd in found.items() if nd.present)
            if len(present) != 24:
                print(f"campaign: only {len(present)} boards after the reopen")
                return False
            self.be.select(list(ALL_BASES))
            for b in ALL_BASES:
                self.be.set_target(b, self.env.idle_psi)
                self.be.set_enabled(b, False)
            self.be.start_cycle()
            time.sleep(0.4)
            self._set_enable_all(True)
            self.be.set_targets({b: self.env.idle_psi for b in ALL_BASES})
            time.sleep(0.6)
            self.rec.observer_error = ""
            self.be.set_cycle_observer(self._on_cycle)
            self.notes.append(f"bus revived (attempt {n}) after a transmit "
                              f"failure; the recording carries the gap")
            print("campaign: bus back, cycle running, excitation resumed")
            return True
        except Exception:
            print("campaign: revival failed: " + traceback.format_exc())
            return False

    def _watchdog(self, idx: int) -> str:
        """Why the session must stop, or ``""`` if it may continue.

        Three ways a session can be dead while still looking busy, all of them
        seen on this bench:

        * the **cycle thread exited** -- ``Backend._cycle_loop`` clears its run
          flag and breaks on a transmit failure, after which nothing reaches
          the bus and nothing says so;
        * the **observer was uninstalled** -- ``Backend`` removes an observer
          that raises, so the recording and the excitation stop together while
          the segment walk carries on;
        * the **writer thread died** -- rows keep being queued and dropped.

        Checked at every segment boundary, which on this plan is at worst every
        18 s, because the cost of finding out late is the whole session.
        """
        if not self.be.running:
            tail = " | ".join(list(self.be.log_lines)[-4:])
            if self.stats.get("bus_revivals", 0) < self.MAX_BUS_REVIVALS:
                print("campaign: the cycle thread exited -- " + tail)
                if self._revive_bus():
                    self._last_rows = self.rec.rows_written
                    return ""
            return ("the 150 Hz cycle thread has exited and could not be "
                    "revived; nothing is reaching the bus. Backend log tail: "
                    + tail)
        if self.rec.writer_error:
            return "the recorder's writer thread died:\n" + self.rec.writer_error
        if self.rec.observer_error:
            return ("the cycle observer raised and was uninstalled:\n"
                    + self.rec.observer_error)
        if idx > 0 and self.rec.rows_written <= self._last_rows:
            return (f"no rows were recorded during segment {idx} "
                    f"(still {self.rec.rows_written}); the recording has "
                    f"stalled even though the cycle is running")
        self._last_rows = self.rec.rows_written
        return ""

    def _watch(self, t: float, sg) -> None:
        if sg.meta.get("disable_all"):
            return
        meas = self._measured_psi()
        q = self.rx.get_q()
        ev = self.watcher.observe(t, self._cmd_psi, meas, q, sg.name)
        if ev is not None:
            names = ", ".join(ev["joint_names"])
            print(f"  ANOMALY at {t:7.1f} s in {sg.name}: commanded but still "
                  f"-- {names}")
            tag = f"anomaly_{len(self.watcher.events):03d}_{sg.name}"
            # The pre-roll is what makes this useful: by the time the settle
            # window has elapsed the motion that should have happened is
            # already in the past.
            self._clip(tag, pre_s=6.0, post_s=6.0)
            ev["clip"] = os.path.join(self.media_dir, f"{tag}.mp4")

    def _write_session_report(self, t_session: float) -> None:
        elapsed = time.perf_counter() - t_session
        cam = self.cam.state if self.cam is not None else None
        report = {
            "session_dir": self.session_dir,
            "elapsed_s": round(elapsed, 1),
            "rows_written": self.rec.rows_written if self.rec else 0,
            "cycles_observed": self.rec.cycles if self.rec else 0,
            "rows_dropped": self.rec.dropped if self.rec else 0,
            "drive_ticks": self.stats["drive_ticks"],
            "drive_late": self.stats["drive_late"],
            "bus_revivals": self.stats.get("bus_revivals", 0),
            "envelope_clamped_single": self.env.clamped_single,
            "envelope_clamped_pair": self.env.clamped_pair,
            "anomalies": self.watcher.events,
            "clips": [p for p in (cam.clip_paths if cam else [])],
            "camera_frames": cam.frames if cam else 0,
            "camera_dropped": cam.dropped if cam else 0,
            "camera_error": cam.error if cam else "camera disabled",
            "camera_lost": bool(cam.lost) if cam else True,
            "camera_reopens": cam.reopens if cam else 0,
            "recorder_writer_error": self.rec.writer_error if self.rec else "",
            "recorder_observer_error": self.rec.observer_error if self.rec else "",
            "cycle_still_running_at_end": bool(self.be.running) if self.be else False,
            "notes": self.notes,
            "vent_before_disable": getattr(self, "vent_report", None),
        }
        path = os.path.join(self.session_dir, "session_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"campaign: {report['rows_written']} rows in {elapsed / 60.0:.1f} "
              f"min, {len(self.watcher.events)} anomalies, "
              f"{len(report['clips'])} clips")
        print(f"campaign: report -> {path}")

    # -- rehearsal ---------------------------------------------------------- #

    def _dry_run(self) -> None:
        """Evaluate every segment on a fine grid and check the envelope.

        Opens no port.  This is what proves the generator satisfies the
        operator's rule *before* the arm is asked to do it, rather than relying
        on the clamp to rescue a bad design at run time -- a clamp that fires is
        a finding about the generator, and the point of the rehearsal is to see
        that finding without the hardware.
        """
        worst_single = 0.0
        worst_pair = 0.0
        n = 0
        for sg in self.segments:
            for t in np.linspace(0.0, sg.duration_s, 61):
                psi = np.asarray(sg.fn(float(t)), float)
                self.env.assert_safe(self.env.clamp(psi),
                                     where=f"dry-run {sg.name} t={t:.2f}")
                worst_single = max(worst_single, float(psi.max()))
                worst_pair = max(worst_pair, float(self.env.pair_sums(psi).max()))
                n += 1
        print(f"campaign: dry run OK over {n} evaluated instants")
        print(f"  worst single commanded pressure {worst_single:.2f} psi "
              f"(hard limit 30.0)")
        print(f"  worst pair sum               {worst_pair:.2f} psi "
              f"(hard limit 30.0)")
        print(f"  clamp fired: {self.env.clamped_single} single, "
              f"{self.env.clamped_pair} pair "
              f"({'clean' if not (self.env.clamped_single or self.env.clamped_pair) else 'THE GENERATOR NEEDED RESCUING'})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default=os.path.join(_WS, "data"))
    ap.add_argument("--tag", default="canarm-sysid")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shorten every segment proportionally (rehearsal)")
    ap.add_argument("--phases", default=None,
                    help="comma-separated excitation function names, "
                         "instead of the full campaign")
    ap.add_argument("--pair-sum-max", type=float, default=28.0)
    ap.add_argument("--single-max", type=float, default=30.0)
    ap.add_argument("--port", default=None)
    ap.add_argument("--server-ip", default=bench_env.MOCAP_SERVER_IP)
    ap.add_argument("--client-ip", default=bench_env.MOCAP_CLIENT_IP)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the envelope over the whole plan; open no port")
    args = ap.parse_args()
    return Campaign(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
