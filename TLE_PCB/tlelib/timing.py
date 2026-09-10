"""Deadline sleeping accurate enough for a 150 Hz transmit schedule.

The control cycle is 6.667 ms. Windows' default scheduler tick is 15.6 ms, and
time.sleep() cannot be trusted below about a millisecond even with CPython's
high-resolution timer, so the pacing here hands the bulk of a wait to the OS
and spins the last part. `hires_clock()` additionally asks Windows for a 1 ms
tick for the lifetime of the process, which is what keeps the coarse part of
the wait from overshooting.
"""
from __future__ import annotations

import sys
import time

_hires_active = False


def hires_clock() -> bool:
    """Request a 1 ms system timer on Windows. True if it was granted.

    Idempotent, and never released: the period is process-global and a bench
    tool holding it for its lifetime is the intent.
    """
    global _hires_active
    if _hires_active:
        return True
    if sys.platform != "win32":
        _hires_active = True
        return True
    try:
        import ctypes

        _hires_active = ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
    except Exception:
        _hires_active = False
    return _hires_active


#: What :func:`fine_gil_handoff` sets the interpreter's thread-switch interval
#: to, in seconds.  CPython's default is 5 ms, which is **75 % of this arm's
#: 6.667 ms cycle period**: a thread that takes the GIL and does not release it
#: at a bytecode boundary can hold it for most of a cycle, and the cycle thread
#: then wakes a whole period late.
GIL_SWITCH_INTERVAL_S = 0.0005


def fine_gil_handoff(interval_s: float = GIL_SWITCH_INTERVAL_S) -> float:
    """Shorten the interpreter's thread-switch interval; return the old value.

    Call this in any process that runs the 150 Hz cycle **alongside other
    Python threads** — a mocap receiver, a recorder, a camera grabber.  With
    the default 5 ms interval the cycle keeps a perfect 6.667 ms median and
    then skips whole periods, which is the signature of losing the GIL race
    rather than of running slowly.

    Measured on the live 24-board arm, 15 s per point, with a NatNet receiver,
    the JSONL recorder, a per-cycle excitation and the bench camera all running
    (``hw_tests/canarm_cycle_profile.py``, 2026-09-10):

    ==================  =========  ==========  ===========
    switch interval     rate       cycles      every board
                                   skipped     answered
    ==================  =========  ==========  ===========
    5 ms (CPython)      143.0 Hz   4.66 %      97.25 %
    0.5 ms              150.0 Hz   0.00 %      99.96 %
    ==================  =========  ==========  ===========

    What this does **not** do is make anything faster: the same work is done in
    the same time, and the shorter interval costs a few more GIL hand-offs a
    second.  It buys the cycle thread the right to interrupt, which is the only
    thing it was short of.
    """
    old = sys.getswitchinterval()
    sys.setswitchinterval(float(interval_s))
    return old


def sleep_until(deadline: float) -> None:
    """Block until perf_counter() reaches `deadline`."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)  # yield without leaving the scheduler's mercy
