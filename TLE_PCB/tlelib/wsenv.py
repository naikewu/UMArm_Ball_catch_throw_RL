"""Optional workspace-level defaults, read from ``<workspace>/bench_env.py``.

Added when this tooling moved into the UMArm workspace. That workspace keeps a
single file naming the bench's devices and interpreters, and the CAN dongle in
particular has already been renamed twice by Windows -- the literals ``COM31``
and ``COM4`` in two older repositories both name ports that no longer exist.
``bench_env.resolve_can_port()`` ranks the enumerated ports by the adapter's USB
descriptor triple instead, which is stable across a replug. Enumeration opens
nothing.

Nothing here is required. Every value this module supplies is a *default* that
the caller already overrides at the call site, so a missing, older, or broken
``bench_env.py`` degrades to the values measured on this bench rather than
raising. That choice keeps ``TLE_PCB/`` runnable on its own, which is what the
port to this workspace was for.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

_env: Any = None
_tried = False


def module() -> Any:
    """The workspace ``bench_env`` module, or None if there is none."""
    global _env, _tried
    if _tried:
        return _env
    _tried = True
    if (WORKSPACE_ROOT / "bench_env.py").is_file():
        if str(WORKSPACE_ROOT) not in sys.path:
            sys.path.insert(0, str(WORKSPACE_ROOT))
        try:
            import bench_env  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover - a broken bench_env must not
            _env = None     # take the whole tool chain down with it
        else:
            _env = bench_env
    return _env


def get(name: str, default):
    """``bench_env.<name>`` if it exists and is not None, otherwise ``default``."""
    env = module()
    if env is None:
        return default
    value = getattr(env, name, None)
    return default if value is None else value


def can_port(default: str) -> str:
    """The CAN adapter's COM device name.

    Prefers ``bench_env.resolve_can_port()``, which consults an explicit
    ``VEMA_CAN_PORT`` environment variable, then a ``list_ports`` scan ranked by
    the dongle's VID/PID/serial, and only then its own fallback. The scan
    enumerates; it does not open anything. Any failure falls back to ``default``,
    which is the port this bench used when the tooling was ported.
    """
    env = module()
    resolve = getattr(env, "resolve_can_port", None) if env is not None else None
    if resolve is None:
        return str(get("CAN_PORT", default))
    try:
        resolved = resolve()
    except Exception:  # pragma: no cover - enumeration is best-effort
        return default
    return str(resolved) if resolved else default
