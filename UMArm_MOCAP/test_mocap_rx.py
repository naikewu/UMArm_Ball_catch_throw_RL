"""Tests for the live receiver's plumbing, with a synthetic frame source.

No sockets, no Motive, no SDK: the NaturalPoint client's whole contract with us
is "call ``rigid_body_listener`` once per plate, then ``new_frame_listener`` once
per frame, on my thread", so the tests call those two listeners directly.  That
covers everything that can actually be wrong on the bench — the streaming-id map,
publishing a frame atomically, staleness, the ring buffer, ``wait_fresh``,
``capture_rest`` — while the part we cannot test (Motive's UDP) is exactly the
part we did not write.

Frames are built **quaternion first** (random unit quaternion -> pose) because
that is the direction Motive works in, and it lets the expected poses be built
with the same :func:`mocap_to_q.quat_xyzw_to_matrix` the receiver uses, without
a matrix-to-quaternion helper that would itself need testing.

Run:  ``python -m pytest UMArm_MOCAP/test_mocap_rx.py -v``
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mocap_constants as mc  # noqa: E402
import mocap_rx as rx_module  # noqa: E402
from mocap_rx import MarkerWindow, MocapRx, MocapWindow  # noqa: E402
from mocap_to_q import mocap_to_q, quat_xyzw_to_matrix  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic Motive frames
# --------------------------------------------------------------------------


def make_frame(rng: np.random.Generator, ids=None) -> dict:
    """One frame as Motive would deliver it: ``{streaming_id: (pos, quat_xyzw)}``."""
    if ids is None:
        ids = [mc.RIGID_BODY_ID_MASK + i for i in range(mc.N_USED_RIGID_BODIES)]
    frame = {}
    for sid in ids:
        quat = rng.normal(size=4)
        frame[sid] = (rng.uniform(-0.5, 0.5, 3), quat / np.linalg.norm(quat))
    return frame


def expected_homos(frame: dict) -> np.ndarray:
    """The pose array the receiver should end up holding for ``frame``."""
    homos = np.tile(np.eye(4), (mc.NUM_RIGID_BODIES, 1, 1)).astype(float)
    for sid, (pos, quat) in frame.items():
        index = (mc.KINOVA_RIGID_BODY_INDEX if sid == mc.KINOVA_MOCAP_STREAM_ID
                 else sid - mc.RIGID_BODY_ID_MASK)
        homos[index, 0:3, 0:3] = quat_xyzw_to_matrix(quat)
        homos[index, 0:3, 3] = pos
    return homos


def push(rx: MocapRx, frame: dict, frame_no: int, mocap=None) -> None:
    """Drive the SDK listeners exactly as ``NatNetClient`` would.

    With ``mocap`` given, the (patched) ``mocap_data_listener`` fires between
    the rigid bodies and the new-frame callback — the real SDK's order, and the
    order the marker scratch's commit contract depends on.
    """
    for sid, (pos, quat) in frame.items():
        rx._on_rigid_body(sid, pos, quat)
    if mocap is not None:
        rx._on_mocap_data(mocap)
    rx._on_new_frame({"frame_number": frame_no})


# --------------------------------------------------------------------------
# Synthetic MoCapData (design §3 / tests §9: tiny stand-in objects with the
# same attributes the receiver reads — the vendored SDK is not imported)
# --------------------------------------------------------------------------


def make_mocap(sets: dict, labeled=None, rigid_bodies=None,
               tracked_models_changed: bool = False) -> SimpleNamespace:
    """A ``MoCapData``-shaped object.

    ``sets``: ``{bytes name: [(x, y, z), ...]}`` — marker sets in asset order.
    ``labeled``: ``[(id_num, (x, y, z), param), ...]`` or None for "labeled
    marker streaming off" (flags then unknown).
    ``rigid_bodies``: ``{streaming_id: (x, y, z)}`` for the centroid fallback.
    """
    marker_sets = SimpleNamespace(marker_data_list=[
        SimpleNamespace(model_name=name, marker_pos_list=[tuple(p) for p in pts])
        for name, pts in sets.items()])
    labeled_data = SimpleNamespace(labeled_marker_list=[] if labeled is None else [
        SimpleNamespace(id_num=i, pos=tuple(p), size=0.0, param=par, residual=0.0)
        for (i, p, par) in labeled])
    body_data = SimpleNamespace(rigid_body_list=[
        SimpleNamespace(id_num=i, pos=tuple(p), rot=(0.0, 0.0, 0.0, 1.0))
        for i, p in (rigid_bodies or {}).items()])
    return SimpleNamespace(
        marker_set_data=marker_sets,
        labeled_marker_data=labeled_data,
        rigid_body_data=body_data,
        suffix_data=SimpleNamespace(tracked_models_changed=tracked_models_changed),
    )


def square_markers(cx: float, cy: float, cz: float, half: float = 0.05) -> list:
    """Four corners of an axis-aligned square about (cx, cy, cz)."""
    return [(cx - half, cy - half, cz), (cx + half, cy - half, cz),
            (cx + half, cy + half, cz), (cx - half, cy + half, cz)]


def make_arm_mocap(rng=None, tracked_models_changed: bool = False,
                   labeled: bool = True, param: int = 0) -> SimpleNamespace:
    """Seven mapped plates plus the ``b"all"`` superset and a clutter set.

    Labeled ids follow Motive's encoding: ``model_id = 500 + plate`` in the
    high 16 bits, ``marker_id = asset index + 1`` in the low 16.
    """
    sets: dict = {}
    lm = []
    for plate in range(mc.N_USED_RIGID_BODIES):
        pts = square_markers(0.2 * plate, 0.0, -0.1 * plate)
        sets[b"plate%d" % plate] = pts
        for j, p in enumerate(pts):
            lm.append((((mc.RIGID_BODY_ID_MASK + plate) << 16) | (j + 1), p, param))
    sets[b"all"] = [p for pts in sets.values() for p in pts]
    sets[b"table_clutter"] = [(9.0, 9.0, 9.0), (9.1, 9.0, 9.0), (9.0, 9.1, 9.0),
                              (9.1, 9.1, 9.0), (9.05, 9.05, 9.0)]
    return make_mocap(sets, labeled=lm if labeled else None,
                      tracked_models_changed=tracked_models_changed)


class Feeder:
    """Background thread pushing frames at a rate, standing in for the SDK thread."""

    def __init__(self, rx: MocapRx, frame: dict, hz: float = 200.0):
        self.rx, self.frame, self.period = rx, frame, 1.0 / hz
        self.n = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.n += 1
            push(self.rx, self.frame, self.n)
            time.sleep(self.period)

    def __enter__(self) -> "Feeder":
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------


class TestConstructionIsInert:
    def test_no_state_and_no_client_before_start(self):
        """Constructing must not touch the network — tests and offline PCs rely on it."""
        rx = MocapRx()
        assert rx.get_q() is None and rx.get_homos() is None
        state = rx.get_state()
        assert not state.running and state.frames == 0 and state.stale
        assert state.frame_number is None and state.last_frame_wall is None

    def test_stop_without_start_is_safe(self):
        rx = MocapRx()
        rx.stop()
        rx.stop()
        assert rx.get_state().last_error is None

    def test_defaults_come_from_constants(self):
        rx = MocapRx()
        assert rx.server_ip == mc.DEFAULT_SERVER_IP
        assert rx.client_ip == mc.DEFAULT_CLIENT_IP
        assert rx.use_multicast is mc.DEFAULT_USE_MULTICAST


class TestFramePublication:
    def test_q_and_homos_match_the_conversion(self):
        rng = np.random.default_rng(1)
        rx = MocapRx()
        frame = make_frame(rng)
        push(rx, frame, 77)

        want = expected_homos(frame)
        assert np.array_equal(rx.get_homos(), want)
        assert np.array_equal(rx.get_q(), mocap_to_q(want))
        state = rx.get_state()
        assert state.frames == 1 and state.valid_frames == 1
        assert state.frame_number == 77 and not state.stale
        assert state.last_error is None

    def test_reads_are_copies(self):
        """A caller must not be able to corrupt the receiver's own state."""
        rx = MocapRx()
        push(rx, make_frame(np.random.default_rng(2)), 1)
        q = rx.get_q()
        q[:] = 99.0
        assert not np.any(rx.get_q() == 99.0)
        homos = rx.get_homos()
        homos[:] = 0.0
        assert rx.get_homos() is not None and np.any(rx.get_homos() != 0.0)

    def test_kinova_and_stray_ids(self):
        """Ids outside the arm's block must not land on a joint's row."""
        rng = np.random.default_rng(3)
        rx = MocapRx()
        frame = make_frame(rng)
        frame[mc.KINOVA_MOCAP_STREAM_ID] = (np.array([1.0, 2.0, 3.0]),
                                            np.array([0.0, 0.0, 0.0, 1.0]))
        push(rx, frame, 1)
        homos = rx.get_homos()
        assert np.allclose(homos[mc.KINOVA_RIGID_BODY_INDEX, 0:3, 3], [1.0, 2.0, 3.0])

        before = rx.get_homos()
        rx._on_rigid_body(999, np.array([9.0, 9.0, 9.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        rx._on_rigid_body(mc.RIGID_BODY_ID_MASK + mc.KINOVA_RIGID_BODY_INDEX,
                          np.array([9.0, 9.0, 9.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        push(rx, {}, 2)
        assert np.array_equal(rx.get_homos(), before)

    def test_degenerate_frame_counts_but_does_not_publish_q(self):
        """Before every plate has been seen once, frames arrive but ``q`` does not.

        This is the startup state on a real bench, and the reason
        :attr:`MocapState.valid_frames` exists next to ``frames``: a caller that
        only watched ``frames`` would think the stream was healthy while
        ``get_q()`` was still ``None``.
        """
        rng = np.random.default_rng(4)
        base_only = {mc.RIGID_BODY_ID_MASK: make_frame(rng)[mc.RIGID_BODY_ID_MASK]}
        rx = MocapRx()
        push(rx, base_only, 1)
        assert rx.get_q() is None
        state = rx.get_state()
        assert state.frames == 1 and state.valid_frames == 0 and state.ring_len == 0
        assert not state.stale                      # frames *are* arriving

    def test_omitted_plate_holds_its_last_pose(self):
        """Pins the inherited "hold last pose" behaviour, hazard and all.

        The SDK's per-plate listener carries no tracking-valid bit (see the note
        in ``mocap_rx``), so a plate that stops updating is indistinguishable
        from one that is not moving, and the pose array keeps its last value.
        One dropped plate therefore yields a plausible ``q`` rather than a
        ``None`` — which is why the ring buffer records the raw U-joint centres.
        """
        rng = np.random.default_rng(4)
        rx = MocapRx()
        good = make_frame(rng)
        push(rx, good, 1)
        q_good = rx.get_q()

        push(rx, {mc.RIGID_BODY_ID_MASK: good[mc.RIGID_BODY_ID_MASK]}, 2)
        assert np.array_equal(rx.get_q(), q_good)
        assert rx.get_state().valid_frames == 2

    def test_listener_survives_malformed_data(self):
        """An exception on the SDK thread would kill the stream silently."""
        rx = MocapRx()
        rx._on_rigid_body(mc.RIGID_BODY_ID_MASK, np.zeros(3),
                          np.array([1.0, 2.0, 3.0]))   # a 3-vector where a quat belongs
        state = rx.get_state()
        assert state.last_error is not None and "rigid_body" in state.last_error

    def test_on_q_callback(self):
        seen = []
        rx = MocapRx(on_q=seen.append)
        push(rx, make_frame(np.random.default_rng(5)), 1)
        assert len(seen) == 1 and np.array_equal(seen[0], rx.get_q())

    def test_callback_gets_its_own_copy(self):
        """An in-place callback must not rewrite published state or history.

        ``on_q=lambda q: q -= q0`` is the obvious way to subtract a calibrated
        joint zero, and the argument used to be the very array stored in ``_q``
        and appended to the ring — so that idiom silently rewrote the published
        value *and* a sample ``snapshot_window`` had already recorded, with no way
        to tell afterwards.  ``get_q``/``get_homos`` have always copied; this path
        now does too.
        """
        rx = MocapRx()
        rx.on_q = lambda q: q.__isub__(np.full(mc.NUM_JOINTS, 100.0))
        push(rx, make_frame(np.random.default_rng(21)), 1)

        published = rx.get_q()
        recorded = rx.snapshot_window().q[0]
        expected = mocap_to_q(expected_homos(make_frame(np.random.default_rng(21))))
        assert np.array_equal(published, expected)
        assert np.array_equal(recorded, expected)

    def test_unsolved_body_is_rejected_and_reported(self):
        """Motive's ``pos=(0,0,0), quat=(0,0,0,0)`` for a body it cannot solve.

        The SDK fires ``rigid_body_listener`` for every body *before* it parses
        the tracking-valid bit, so this pose reaches us as if it were data.
        Normalising a zero quaternion used to yield an all-NaN rotation with only
        a ``RuntimeWarning``, which de-rotated every link into NaN and published
        ``q = [nan] * 12`` with ``valid_frames`` incremented, ``stale`` False and
        ``last_error`` None — a control loop reading ``wait_fresh()`` would drive
        NaN into valve commands with no indication anything was wrong.
        """
        rng = np.random.default_rng(22)
        rx = MocapRx()
        good = make_frame(rng)
        push(rx, good, 1)
        q_good = rx.get_q()
        assert q_good is not None and np.all(np.isfinite(q_good))

        bad = dict(good)
        bad[mc.RIGID_BODY_ID_MASK] = (np.zeros(3), np.zeros(4))   # base plate occluded
        push(rx, bad, 2)

        state = rx.get_state()
        assert state.last_error is not None and "rigid_body" in state.last_error
        # The plate kept its previous pose, so q stays finite and usable...
        assert np.all(np.isfinite(rx.get_q()))
        # ...and nothing non-finite ever reached the ring buffer.
        assert np.all(np.isfinite(rx.snapshot_window().q))

    def test_nan_pose_never_publishes_a_q(self):
        """Belt and braces: a NaN that gets in some other way still cannot publish.

        ``mocap_to_q`` gates on finiteness, so even a NaN written straight into the
        pose array (bypassing the quaternion check) is a rejected frame, not a
        plausible one.
        """
        rng = np.random.default_rng(23)
        rx = MocapRx()
        push(rx, make_frame(rng), 1)
        rx._incoming[mc.IDX_BASE, 0:3, 0:3] = np.nan
        push(rx, {}, 2)
        assert np.all(np.isfinite(rx.get_q()))       # still the last good frame
        state = rx.get_state()
        assert state.frames == 2 and state.valid_frames == 1

    def test_one_untracked_plate_still_publishes(self):
        """Pins the documented hole so the module docstring stays honest.

        A plate that is never mentioned leaves both neighbouring differences
        non-degenerate, so ``mocap_to_q`` has nothing to reject on and the joints
        around it encode the direction to the mocap origin.  ``stale`` catches the
        whole stream dying, not one marker set dying — that is what the raw
        U-joint centres in the ring buffer are for.
        """
        rng = np.random.default_rng(24)
        rx = MocapRx()
        ids = [mc.RIGID_BODY_ID_MASK + i for i in range(mc.N_USED_RIGID_BODIES)
               if i != mc.IDX_U3_DISTAL]
        push(rx, make_frame(rng, ids=ids), 1)
        q = rx.get_q()
        assert q is not None and np.all(np.isfinite(q))
        state = rx.get_state()
        assert state.valid_frames == 1 and not state.stale and state.last_error is None
        # The give-away is in the recorded centres, not in q: plate 5 sits at the
        # mocap origin because Motive never placed it.
        assert np.array_equal(rx.snapshot_window().u[0][mc.IDX_U3_DISTAL], np.zeros(3))

    def test_callback_may_call_back_in(self):
        """Callbacks run outside the lock; a re-entrant one must not deadlock."""
        results = []
        rx = MocapRx()
        rx.on_q = lambda q: results.append(rx.get_state().frame_number)
        push(rx, make_frame(np.random.default_rng(6)), 42)
        assert results == [42]


class TestStaleness:
    def test_fresh_then_stale(self, monkeypatch):
        rx = MocapRx()
        push(rx, make_frame(np.random.default_rng(7)), 1)
        assert not rx.get_state().stale
        # Shrink the window instead of sleeping 0.25 s: the flag is computed
        # against the clock at call time, which is the behaviour under test.
        monkeypatch.setattr(rx_module.mc, "STALE_AFTER_S", 0.0)
        assert rx.get_state().stale

    def test_rate_estimate(self):
        rng = np.random.default_rng(8)
        rx = MocapRx()
        frame = make_frame(rng)
        with Feeder(rx, frame, hz=200.0):
            time.sleep(0.4)
            fps = rx.get_state().fps
        # Windows' sleep granularity makes the exact rate unpredictable; what
        # matters is that the estimate is a real measurement, not the nominal.
        assert 20.0 < fps < 400.0


class TestWaitFresh:
    def test_times_out_without_a_stream(self):
        rx = MocapRx()
        t0 = time.monotonic()
        assert rx.wait_fresh(timeout=0.05) is None
        assert time.monotonic() - t0 >= 0.05

    def test_ignores_history(self):
        """The whole point: a q from before the call must not satisfy the wait."""
        rx = MocapRx()
        push(rx, make_frame(np.random.default_rng(9)), 1)
        assert rx.get_q() is not None
        assert rx.wait_fresh(timeout=0.05) is None

    def test_returns_the_next_frame(self):
        rng = np.random.default_rng(10)
        rx = MocapRx()
        frame = make_frame(rng)
        with Feeder(rx, frame, hz=100.0):
            q = rx.wait_fresh(timeout=2.0)
        assert q is not None
        assert np.allclose(q, mocap_to_q(expected_homos(frame)))


class TestHistory:
    def test_ring_records_q_and_u_joint_centres(self):
        rng = np.random.default_rng(11)
        rx = MocapRx()
        frames = [make_frame(rng) for _ in range(5)]
        for i, f in enumerate(frames):
            push(rx, f, 100 + i)

        window = rx.snapshot_window()
        assert len(window) == 5
        assert window.q.shape == (5, mc.NUM_JOINTS)
        assert window.u.shape == (5, 6, 3)
        assert list(window.frame_no) == [100, 101, 102, 103, 104]
        assert np.all(np.diff(window.t) >= 0)
        last = expected_homos(frames[-1])
        assert np.allclose(window.q[-1], mocap_to_q(last))
        assert np.allclose(window.u[-1], last[list(mc.U_JOINT_INDICES), 0:3, 3])

    def test_window_bounds_are_inclusive_and_open_ended(self):
        rng = np.random.default_rng(12)
        rx = MocapRx()
        for i in range(6):
            push(rx, make_frame(rng), i)
            time.sleep(0.002)
        all_t = rx.snapshot_window().t
        mid = all_t[2]
        assert len(rx.snapshot_window(t0=mid)) == 4
        assert len(rx.snapshot_window(t1=mid)) == 3
        assert len(rx.snapshot_window(mid, mid)) == 1
        assert len(rx.snapshot_window(all_t[-1] + 1.0, all_t[-1] + 2.0)) == 0

    def test_empty_window_has_the_right_shapes(self):
        """Callers index ``.q[:, j]`` unconditionally; an empty window must not
        blow up with a 1-D array."""
        window = MocapRx().snapshot_window()
        assert len(window) == 0
        assert window.q.shape == (0, mc.NUM_JOINTS) and window.u.shape == (0, 6, 3)
        assert window.duration == 0.0 and window.fps == 0.0

    def test_ring_is_bounded(self):
        rng = np.random.default_rng(13)
        rx = MocapRx(ring_capacity=10)
        frame = make_frame(rng)
        for i in range(25):
            push(rx, frame, i)
        window = rx.snapshot_window()
        assert len(window) == 10
        assert list(window.frame_no) == list(range(15, 25))  # oldest dropped

    def test_clear_history(self):
        rx = MocapRx()
        push(rx, make_frame(np.random.default_rng(14)), 1)
        rx.clear_history()
        assert len(rx.snapshot_window()) == 0
        assert rx.get_q() is not None      # clearing history is not clearing q


class TestCaptureRest:
    def test_returns_none_without_a_stream(self):
        assert MocapRx().capture_rest(seconds=0.05, timeout=0.05) is None

    def test_mean_and_spread_of_a_held_pose(self):
        rng = np.random.default_rng(15)
        rx = MocapRx()
        frame = make_frame(rng)
        q_true = mocap_to_q(expected_homos(frame))
        with Feeder(rx, frame, hz=200.0):
            rest = rx.capture_rest(seconds=0.25, timeout=1.0)
        assert rest is not None and rest.n > 5
        assert np.allclose(rest.mean, q_true, atol=1e-12)
        # Every frame carried the same pose, so the spread is nothing but the
        # rounding in the mean (ptp is exactly zero; sd is not, because it
        # subtracts a mean that is one ulp off the common value).
        assert np.all(rest.ptp == 0.0)
        assert np.all(rest.sd < 1e-12)
        assert len(rest.window) == rest.n
        assert 0.0 < rest.duration <= 0.3

    def test_rejects_windows_it_cannot_hold(self):
        rx = MocapRx()
        with pytest.raises(ValueError):
            rx.capture_rest(seconds=mc.RING_SECONDS + 1.0)
        with pytest.raises(ValueError):
            rx.capture_rest(seconds=0.0)

    def test_capacity_guard_uses_the_measured_rate(self):
        """The ring is bounded by sample *count*, so seconds need a rate.

        With Motive at 240 Hz — a standard Prime-camera rate — a 2400-sample ring
        holds 10 s, not the 20 s the nominal 120 Hz implies.  The guard used to
        divide by the constant and so accepted a 15 s window it could not hold,
        while telling the caller they had 20 s of history.
        """
        rx = MocapRx(ring_capacity=2400)
        frame = make_frame(np.random.default_rng(30))
        # Hand-build a rate estimate of ~240 Hz without needing a real 240 Hz feed.
        push(rx, frame, 1)
        now = time.monotonic()
        with rx._lock:
            rx._frame_times.clear()
            for i in range(64):
                rx._frame_times.append(now - (63 - i) / 240.0)
        assert 200.0 < rx.get_state().fps < 280.0

        with pytest.raises(ValueError, match="exceeds"):
            rx.capture_rest(seconds=15.0, timeout=0.01)
        # And the message must quote the real capacity, not the nominal one.
        try:
            rx.capture_rest(seconds=15.0, timeout=0.01)
        except ValueError as exc:
            assert "20 s" not in str(exc)

    def test_returns_none_when_the_stream_dies_mid_capture(self):
        """The failure this method exists to refuse to paper over.

        One frame lands inside the window, then the stream stops.  The old loop
        watched nothing but the clock (despite a comment claiming otherwise), so
        it sat out the full three seconds and reported ``n=1``, ``duration=0.0``,
        ``mean`` = that single sample and ``sd = ptp = zeros(12)`` — a noise floor
        of exactly zero, adopted as a joint zero, from a stream that had been dead
        for three seconds.
        """
        rng = np.random.default_rng(31)
        rx = MocapRx()
        frame = make_frame(rng)

        def two_then_die():
            time.sleep(0.02)
            push(rx, frame, 1)          # satisfies wait_fresh
            time.sleep(0.02)
            push(rx, frame, 2)          # lands inside the window, then silence
        threading.Thread(target=two_then_die, daemon=True).start()

        t0 = time.monotonic()
        assert rx.capture_rest(seconds=3.0, timeout=1.0) is None
        # And it noticed rather than waiting the whole window out.
        assert time.monotonic() - t0 < 2.0
        assert rx.get_state().stale

    def test_returns_none_rather_than_a_zero_noise_floor(self, monkeypatch):
        """A single sample cannot carry a spread, so it is not reported as one.

        ``sd`` used to fall back to ``zeros(12)`` at ``n == 1``, which is the most
        reassuring possible output for the worst possible input: a noise floor of
        exactly zero, taken straight into a calibration file.
        """
        rng = np.random.default_rng(32)
        rx = MocapRx()
        frame = make_frame(rng)

        def one_inside():
            time.sleep(0.02)
            push(rx, frame, 1)          # satisfies wait_fresh
            time.sleep(0.02)
            push(rx, frame, 2)          # the lone sample inside the window
        threading.Thread(target=one_inside, daemon=True).start()
        # Generous stale threshold, so staleness cannot be what rejects this.
        monkeypatch.setattr(rx_module.mc, "STALE_AFTER_S", 100.0)
        assert rx.capture_rest(seconds=0.3, timeout=1.0) is None
        assert len(rx.snapshot_window()) == 2   # samples exist; the summary refused

    def test_returns_none_when_the_stream_is_alive_but_far_too_sparse(self, monkeypatch):
        """Live, not stale, enough samples to average — and still not a capture.

        A Motive project misconfigured to a few Hz, or a stream that resumes for
        the last instant of the window, gives a handful of samples spanning almost
        none of the requested time.  ``mean`` over that is not the rest pose the
        caller asked for, so it is refused rather than returned with an honest-
        but-easily-ignored ``duration``.
        """
        rng = np.random.default_rng(34)
        rx = MocapRx()
        frame = make_frame(rng)

        def a_few_then_silence():
            time.sleep(0.02)
            for i in range(4):
                push(rx, frame, i)
                time.sleep(0.01)
        threading.Thread(target=a_few_then_silence, daemon=True).start()
        monkeypatch.setattr(rx_module.mc, "STALE_AFTER_S", 100.0)

        rest = rx.capture_rest(seconds=1.0, timeout=1.0)
        assert rest is None
        window = rx.snapshot_window()
        assert len(window) >= 3                      # n >= 2, so the count gate passed
        assert window.duration < 0.5                 # coverage is what refused

    def test_healthy_capture_covers_the_window(self):
        """The guard must not fire on a real stream: coverage is ~0.97."""
        rng = np.random.default_rng(33)
        rx = MocapRx()
        frame = make_frame(rng)
        with Feeder(rx, frame, hz=200.0):
            rest = rx.capture_rest(seconds=0.25, timeout=1.0)
        assert rest is not None
        assert rest.n >= 2
        assert rest.duration >= rx_module.MIN_REST_COVERAGE * 0.25
        assert np.all(np.isfinite(rest.sd))


class TestMarkerTransport:
    """The §3 transport: mapping, flags, commit contract, epochs."""

    def test_primary_mapping_groups_by_model_id(self):
        """Labeled model ids 500+i bind set names to plates; ``b"all"`` and
        clutter sets never map, so a crowded volume cannot land on a plate."""
        rx = MocapRx()
        push(rx, make_frame(np.random.default_rng(40)), 1, mocap=make_arm_mocap())
        w = rx.snapshot_marker_window()
        assert len(w) == 1
        markers = w.markers[0]
        assert set(markers) == set(range(mc.N_USED_RIGID_BODIES))
        for plate in range(mc.N_USED_RIGID_BODIES):
            want = np.asarray(square_markers(0.2 * plate, 0.0, -0.1 * plate))
            assert np.array_equal(markers[plate], want)
            assert np.array_equal(w.flags[0][plate],
                                  np.ones(4, dtype=np.uint8))
        health = rx.marker_health()
        assert set(health.plate_names.values()) == set(range(mc.N_USED_RIGID_BODIES))
        assert b"all" not in health.plate_names
        assert b"table_clutter" not in health.plate_names

    def test_commit_contract_consumes_and_clears(self):
        """A frame whose marker data did not arrive records None — never the
        previous frame's dict (review finding int-4)."""
        rng = np.random.default_rng(41)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        push(rx, make_frame(rng), 2)                       # no mocap_data fired
        push(rx, make_frame(rng), 3, mocap=make_arm_mocap())
        w = rx.snapshot_marker_window()
        assert len(w) == 3
        assert w.markers[0] is not None and w.flags[0] is not None
        assert w.markers[1] is None and w.flags[1] is None
        assert w.markers[2] is not None

    def test_marker_ring_records_q_none_frames(self):
        """The dropout-rich frames are the point (review int-8/ops-7): a frame
        with no usable q still lands in the marker ring, with streamed poses."""
        rng = np.random.default_rng(42)
        base_only = {mc.RIGID_BODY_ID_MASK: make_frame(rng)[mc.RIGID_BODY_ID_MASK]}
        rx = MocapRx()
        push(rx, base_only, 7, mocap=make_arm_mocap())
        state = rx.get_state()
        assert state.valid_frames == 0 and state.ring_len == 0     # q ring: nothing
        w = rx.snapshot_marker_window()
        assert len(w) == 1 and w.frame_no[0] == 7
        assert w.markers[0] is not None
        assert w.streamed_poses.shape == (1, mc.N_USED_RIGID_BODIES, 4, 4)
        assert np.array_equal(w.streamed_poses[0, 0], expected_homos(base_only)[0])

    def test_mapping_epoch_bumps_on_tracked_models_changed(self):
        rng = np.random.default_rng(43)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        push(rx, make_frame(rng), 2, mocap=make_arm_mocap())
        push(rx, make_frame(rng), 3,
             mocap=make_arm_mocap(tracked_models_changed=True))
        push(rx, make_frame(rng), 4, mocap=make_arm_mocap())
        w = rx.snapshot_marker_window()
        assert list(w.mapping_epoch) == [1, 1, 2, 2]

    def test_epoch_zero_until_a_mapping_is_derivable(self):
        """No labeled markers and no streamed bodies -> nothing to map against:
        the frame records an empty dict (streamed, nothing identified) at epoch
        0, and the derivation retries — it must not give up for the session."""
        rng = np.random.default_rng(44)
        rx = MocapRx()
        unmappable = make_mocap({b"plate0": square_markers(0.0, 0.0, 0.0)})
        push(rx, make_frame(rng), 1, mocap=unmappable)
        push(rx, make_frame(rng), 2, mocap=make_arm_mocap())
        w = rx.snapshot_marker_window()
        assert w.markers[0] == {} and w.mapping_epoch[0] == 0
        assert len(w.markers[1]) == mc.N_USED_RIGID_BODIES
        assert w.mapping_epoch[1] == 1

    def test_mapped_asset_absent_from_a_frame_is_just_missing(self):
        rng = np.random.default_rng(45)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        partial = make_arm_mocap()
        del partial.marker_set_data.marker_data_list[3]    # plate 3's set gone
        push(rx, make_frame(rng), 2, mocap=partial)
        w = rx.snapshot_marker_window()
        assert set(w.markers[1]) == set(range(mc.N_USED_RIGID_BODIES)) - {3}

    def test_flags_follow_the_param_bits(self):
        """occluded (0x01) and model-solved (0x04) clear the flag; point-cloud-
        solved (0x02) is normal tracking; a marker missing from the labeled
        list is untracked (design §5 — model fills must not launder the
        rigid-body solve into independent marker evidence)."""
        rng = np.random.default_rng(46)
        mocap = make_arm_mocap()
        lm = mocap.labeled_marker_data.labeled_marker_list
        # Plate 0's four labeled entries are the first four, marker_id 1..4.
        lm[0].param = 0x01                       # occluded
        lm[1].param = 0x04                       # model-solved
        lm[2].param = 0x02                       # point-cloud-solved = tracked
        del lm[3]                                # absent from labeled data
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=mocap)
        w = rx.snapshot_marker_window()
        assert np.array_equal(w.flags[0][0], np.array([0, 0, 1, 0], dtype=np.uint8))
        # ...while other plates are untouched.
        assert np.array_equal(w.flags[0][1], np.ones(4, dtype=np.uint8))
        health = rx.marker_health()
        assert health.occluded_seen == 1 and health.model_solved_seen == 1
        assert health.point_cloud_solved_seen == 1

    def test_flags_unknown_when_labeled_markers_absent(self):
        """Markers still recorded (the mapping persists across the regime
        change), flags None: unknown is a recorded state, never all-present."""
        rng = np.random.default_rng(47)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        push(rx, make_frame(rng), 2, mocap=make_arm_mocap(labeled=False))
        w = rx.snapshot_marker_window()
        assert w.flags[0] is not None
        assert w.markers[1] is not None and len(w.markers[1]) == 7
        assert w.flags[1] is None

    def test_marker_id_mismatch_falls_back_to_position_matching(self):
        """The §3 assumption (marker_id - 1 == asset index) is checked per
        marker against exact positions; where it fails, the flag lands on the
        position-matched index and the miss is tallied for the probe."""
        rng = np.random.default_rng(48)
        pts = square_markers(0.0, 0.0, 0.0)
        # One labeled marker claiming marker_id 1 (index 0) but carrying the
        # position of asset index 2.
        labeled = [((mc.RIGID_BODY_ID_MASK << 16) | 1, pts[2], 0)]
        mocap = make_mocap({b"plate0": pts}, labeled=labeled)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=mocap)
        w = rx.snapshot_marker_window()
        assert np.array_equal(w.flags[0][0], np.array([0, 0, 1, 0], dtype=np.uint8))
        health = rx.marker_health()
        assert health.id_corr_bad == 1 and health.id_corr_ok == 0

    def test_centroid_fallback_is_restricted(self):
        """No labeled markers ever seen: only an exactly-4-marker set whose
        centroid wins by <20 mm with the runner-up >=2x farther may map (§3 —
        the 47 mm inter-plate gaps make sloppy nearest-neighbour unsafe)."""
        rng = np.random.default_rng(49)
        bodies = {mc.RIGID_BODY_ID_MASK: (0.0, 0.0, 0.0),
                  mc.RIGID_BODY_ID_MASK + 1: (0.03, 0.0, 0.0)}
        sets = {
            b"good": square_markers(0.0, 0.0, 0.0),          # centroid == pivot 0
            b"five": square_markers(0.03, 0.0, 0.0)          # right count wrong...
                     + [(0.03, 0.0, 0.01)],                  # ...5 markers
            b"far": square_markers(0.06, 0.0, 0.0),          # 60 mm from best
            b"between": square_markers(0.012, 0.0, 0.0),     # runner-up < 2x
            b"all": square_markers(0.0, 0.0, 0.0) + square_markers(0.03, 0.0, 0.0),
        }
        mocap = make_mocap(sets, labeled=None, rigid_bodies=bodies)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=mocap)
        w = rx.snapshot_marker_window()
        assert set(w.markers[0]) == {0}
        assert rx.marker_health().plate_names == {b"good": 0}
        assert w.flags[0] is None                            # unlabeled regime

    def test_malformed_mocap_data_is_survived_and_reported(self):
        """An exception on the SDK thread would kill the stream silently —
        same rule as the rigid-body listener."""
        rng = np.random.default_rng(50)
        rx = MocapRx()
        bad = SimpleNamespace(marker_set_data=SimpleNamespace(marker_data_list=42))
        rx._on_mocap_data(bad)
        state = rx.get_state()
        assert state.last_error is not None and "mocap_data" in state.last_error
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())   # still alive
        assert len(rx.snapshot_marker_window()) == 1

    def test_marker_health_counts_frames_and_correspondence(self):
        rng = np.random.default_rng(51)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        push(rx, make_frame(rng), 2, mocap=make_arm_mocap(labeled=False))
        health = rx.marker_health()
        assert health.mocap_data_frames == 2 and health.labeled_frames == 1
        assert health.last_marker_set_count == 9       # 7 plates + b"all" + clutter
        assert health.last_labeled_marker_count == 0   # last frame was unlabeled
        assert health.id_corr_ok == 28 and health.id_corr_bad == 0
        assert health.marker_ring_len == 2


    # ---------------------------------------------------------------- #
    # Labeled-only regime: what this lab's Motive actually streams
    # (2026-08-11 live probe) — labeled markers, NO per-asset marker sets.
    # ---------------------------------------------------------------- #

    def test_labeled_only_regime_builds_plates_from_ids(self):
        """model_id names the plate and marker_id-1 the slot: full arrays,
        known flags, epoch 1, and no name mapping involved at all."""
        rng = np.random.default_rng(46)
        full = make_arm_mocap()
        labeled_only = make_mocap({}, labeled=[
            (lm.id_num, lm.pos, lm.param)
            for lm in full.labeled_marker_data.labeled_marker_list])
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=labeled_only)
        w = rx.snapshot_marker_window()
        markers, flags = w.markers[0], w.flags[0]
        assert set(markers) == set(range(mc.N_USED_RIGID_BODIES))
        for plate in range(mc.N_USED_RIGID_BODIES):
            want = np.asarray(square_markers(0.2 * plate, 0.0, -0.1 * plate))
            assert np.array_equal(markers[plate], want)
            assert np.array_equal(flags[plate], np.ones(4, dtype=np.uint8))
        assert w.mapping_epoch[0] == 1
        assert rx.marker_health().plate_names == {}    # no names were needed

    def test_labeled_only_missing_id_is_a_nan_row_with_flag_zero(self):
        """An id Motive omitted this frame is a dropout: NaN row, flag 0, and
        the array stays 4 rows so the lock's count gate keeps its meaning."""
        rng = np.random.default_rng(47)
        pts = square_markers(0.0, 0.0, 0.0)
        labeled = [(((mc.RIGID_BODY_ID_MASK + 0) << 16) | mid, pts[mid - 1], 0)
                   for mid in (1, 2, 4)]                    # id 3 absent
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_mocap({}, labeled=labeled))
        w = rx.snapshot_marker_window()
        arr, fl = w.markers[0][0], w.flags[0][0]
        assert arr.shape == (4, 3) and fl.shape == (4,)
        assert np.isnan(arr[2]).all() and fl[2] == 0
        assert np.array_equal(fl, np.array([1, 1, 0, 1], dtype=np.uint8))
        assert np.array_equal(arr[[0, 1, 3]], np.asarray(pts)[[0, 1, 3]])

    def test_labeled_only_occluded_entry_keeps_position_but_flag_zero(self):
        """A model-filled (occluded) position stays visible to the benchmark
        but is never 'tracked' — D4: don't launder the rigid-body solve."""
        rng = np.random.default_rng(48)
        pts = square_markers(0.0, 0.0, 0.0)
        labeled = [(((mc.RIGID_BODY_ID_MASK + 0) << 16) | (j + 1), p,
                    0x01 if j == 0 else 0) for j, p in enumerate(pts)]
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_mocap({}, labeled=labeled))
        w = rx.snapshot_marker_window()
        arr, fl = w.markers[0][0], w.flags[0][0]
        assert np.array_equal(fl, np.array([0, 1, 1, 1], dtype=np.uint8))
        assert np.array_equal(arr[0], np.asarray(pts[0]))   # position kept
        assert rx.marker_health().occluded_seen == 1

    def test_neither_sets_nor_labeled_records_not_streamed(self):
        """Empty sets AND no arm labeled markers -> scratch stays None (the
        'not streamed' regime), not an empty dict."""
        rng = np.random.default_rng(49)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_mocap({}, labeled=None))
        w = rx.snapshot_marker_window()
        assert w.markers[0] is None and w.flags[0] is None
        assert w.mapping_epoch[0] == 0


class TestMarkerWindow:
    def test_slicing_bounds_are_inclusive_and_open_ended(self):
        rng = np.random.default_rng(52)
        rx = MocapRx()
        for i in range(6):
            push(rx, make_frame(rng), i, mocap=make_arm_mocap())
            time.sleep(0.002)
        w = rx.snapshot_marker_window()
        assert len(w) == 6 and len(w.markers) == 6 and len(w.flags) == 6
        mid = w.t[2]
        assert len(rx.snapshot_marker_window(t0=mid)) == 4
        assert len(rx.snapshot_marker_window(t1=mid)) == 3
        assert len(rx.snapshot_marker_window(mid, mid)) == 1
        assert len(rx.snapshot_marker_window(w.t[-1] + 1.0, w.t[-1] + 2.0)) == 0

    def test_empty_window_has_the_right_shapes(self):
        w = MocapRx().snapshot_marker_window()
        assert len(w) == 0
        assert w.streamed_poses.shape == (0, mc.N_USED_RIGID_BODIES, 4, 4)
        assert w.markers == () and w.flags == ()
        assert w.mapping_epoch.shape == (0,)
        assert w.duration == 0.0 and w.fps == 0.0

    def test_snapshot_hands_out_copies(self):
        """Mutating a returned window must not rewrite recorded history."""
        rng = np.random.default_rng(53)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        w = rx.snapshot_marker_window()
        w.markers[0][0][:] = 99.0
        w.flags[0][0][:] = 99
        w.streamed_poses[:] = 0.0
        again = rx.snapshot_marker_window()
        assert not np.any(again.markers[0][0] == 99.0)
        assert not np.any(again.flags[0][0] == 99)
        assert np.any(again.streamed_poses != 0.0)

    def test_marker_ring_capacity_defaults_to_the_q_ring(self):
        assert MocapRx()._marker_ring.maxlen == mc.RING_CAPACITY
        assert MocapRx(ring_capacity=10)._marker_ring.maxlen == 10
        rx = MocapRx(ring_capacity=10, marker_ring_capacity=33)
        assert rx._ring.maxlen == 10 and rx._marker_ring.maxlen == 33

    def test_resize_rings_keeps_the_newest_entries(self):
        rng = np.random.default_rng(54)
        rx = MocapRx(ring_capacity=10)
        for i in range(12):
            push(rx, make_frame(rng), i, mocap=make_arm_mocap())
        rx.resize_rings(ring_capacity=5, marker_ring_capacity=5)
        assert list(rx.snapshot_window().frame_no) == list(range(7, 12))
        assert list(rx.snapshot_marker_window().frame_no) == list(range(7, 12))
        assert rx._ring.maxlen == 5 and rx._marker_ring.maxlen == 5

    def test_clear_history_clears_both_rings(self):
        rng = np.random.default_rng(55)
        rx = MocapRx()
        push(rx, make_frame(rng), 1, mocap=make_arm_mocap())
        rx.clear_history()
        assert len(rx.snapshot_window()) == 0
        assert len(rx.snapshot_marker_window()) == 0


class TestMocapWindowUntouched:
    """Design decision D5: the q ring and MocapWindow are provably unaffected —
    ``test_joint_verification.py`` constructs MocapWindow with exactly four
    kwargs, and that must keep working with no fifth field to supply."""

    def test_four_field_construction_and_shape(self):
        n = 3
        w = MocapWindow(t=np.arange(n) / 120.0, frame_no=np.arange(n),
                        q=np.zeros((n, mc.NUM_JOINTS)), u=np.zeros((n, 6, 3)))
        assert len(w) == n
        assert len(dataclasses.fields(MocapWindow)) == 4

    def test_q_ring_entries_still_have_four_elements(self):
        rx = MocapRx()
        push(rx, make_frame(np.random.default_rng(56)), 1, mocap=make_arm_mocap())
        assert len(rx._ring[0]) == 4
        assert len(rx._marker_ring[0]) == 6      # the new ring is the wide one


class TestSdkImport:
    def test_vendored_sdk_is_importable_and_never_auto_loaded(self):
        """The SDK must be present and loadable, but only when asked for.

        Importing ``mocap_rx`` must not pull in 150 kB of vendor code or add
        three generic top-level names (``NatNetClient``, ``MoCapData``,
        ``DataDescriptions``) to ``sys.path``/``sys.modules`` for a process that
        only wanted the math.  Loading the class object opens nothing — the SDK
        creates its sockets in ``run()``, which nothing here calls.
        """
        sdk_dir = os.path.join(_HERE, "natnet_sdk")
        for name in ("NatNetClient.py", "MoCapData.py", "DataDescriptions.py"):
            assert os.path.isfile(os.path.join(sdk_dir, name))

        cls = MocapRx._import_natnet()
        assert cls.__name__ == "NatNetClient"
        assert hasattr(cls, "run") and hasattr(cls, "shutdown")
        assert sdk_dir in sys.path      # the import added it, lazily
