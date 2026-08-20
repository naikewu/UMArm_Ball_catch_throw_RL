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
