"""The lock-free array a producer publishes and the viewer reads.

One flat block of doubles in a ``multiprocessing.Array(lock=False)``, in the
style ``C:\\RUNZE_SRC\\RS485_VEMA\\UMArm_COLLAB\\gui\\viz.py`` established: the
renderer would rather read a torn frame than make a bus tick wait on a repaint,
and every field here is either independently meaningful or re-sent sixty times a
second.

Kept in its own module so a producer and the viewer agree on the layout by
**importing it**, rather than by both being edited at the same time.  That is the
whole point of the file, and it is why the offsets below are computed from one
table instead of being written out as a ladder of hand-added constants: the RS485
original had to renumber five constants every time a field was inserted, and a
producer built against the old numbering wrote a plausible frame into the wrong
slots.

WHAT CHANGED FROM THE RS485 VERSION, and why.  That layout named its two robots
in its field names (``UMARM_Q``, ``KINOVA_Q``, ``UMARM_MOUNT``, ``KINOVA_MOUNT``)
and hard-coded the pair.  This room holds three — the CAN arm, the RS485 arm and
the Gen3 — and the CAN arm has to render when the other two are absent, so the
layout is generated from :data:`ROBOTS` and addressed by name.  Adding a fourth
robot is one row in that table.

THE LAYOUT IS FIXED-SIZE REGARDLESS OF WHO IS PRESENT.  A room with no Kinova
still carries the Kinova's block, unwritten and flagged not-fresh.  Sizing the
array to the present robots would make the offsets depend on a runtime decision
that the two processes make independently, which is the one thing this module
exists to prevent.

Blocks, per robot, in :data:`ROBOTS` order:

===================  ======  ==========================================
field                length  meaning
===================  ======  ==========================================
``Q``                ``nq``  joint angles, radians, model qpos order
``MOUNT``            7       base pose: ``pos(3)`` + scalar-first ``quat(4)``
``FRESH``            1       1.0 when the mount came from a live source
``PLATES_OK``        1       1.0 when the plate block below is real
``PLATES``           16·n    ``(n_plates, 4, 4)`` row-major measured poses
===================  ======  ==========================================

then two globals:

``SEQ``
    publish counter, so the viewer can tell a stalled producer from a still arm.
``GENERATION``
    rebuild counter.  ``launch_passive`` is bound to the model it was handed and
    MuJoCo will not recompile a live one, so anything that changes geometry —
    a robot arriving or leaving, a re-fitted chain — is published by bumping
    this and letting the viewer tear its window down and open a fresh one.

The plate block is not in the RS485 original; it is here because the overlay this
viewer draws is the point of the view.  The measured plate poses are what the
cameras say, the rendered arm is forward kinematics of ``q``, and the two
agreeing on screen is the whole pose pipeline verified at a glance.  A producer
that has no plates leaves ``PLATES_OK`` at zero and the overlay simply does not
draw, which is the honest thing for a robot whose joints are known and whose
markers are not.
"""

from __future__ import annotations

__all__ = [
    "ROBOTS", "NQ", "N_PLATES", "MOUNT_LEN",
    "Q_OFF", "MOUNT_OFF", "FRESH_OFF", "PLATES_OK_OFF", "PLATES_OFF",
    "SEQ", "GENERATION", "VIZ_LEN",
    "q_slice", "mount_slice", "fresh_index", "plates_ok_index", "plates_slice",
    "make_array", "write_robot", "read_robot", "describe",
]

#: The room's roster, in the order their blocks appear in the array.  The CAN
#: arm is first because it is the one robot that is always present.
ROBOTS = ("canarm", "rs485", "kinova")

#: Degrees of freedom per robot.  The two UMArms are 3 segments x 2 universal
#: joints x 2 DOF; the Gen3 is a 7-DOF serial chain.
NQ = {"canarm": 12, "rs485": 12, "kinova": 7}

#: Marker plates a robot's mocap adapter can report, and therefore how much room
#: its overlay block needs.  Six for either UMArm — base plate plus five joint
#: plates, ids ``<base>+0 .. <base>+5``.  **Zero for the Gen3**: nothing is stuck
#: to its plinth, and the pad markers on its tool flange are a different
#: quantity (they solve the base, they do not decorate the chain).
N_PLATES = {"canarm": 6, "rs485": 6, "kinova": 0}

#: ``pos(3)`` + scalar-first ``quat(4)``.
MOUNT_LEN = 7

Q_OFF: dict[str, int] = {}
MOUNT_OFF: dict[str, int] = {}
FRESH_OFF: dict[str, int] = {}
PLATES_OK_OFF: dict[str, int] = {}
PLATES_OFF: dict[str, int] = {}

_cursor = 0
for _name in ROBOTS:
    Q_OFF[_name] = _cursor
    _cursor += NQ[_name]
    MOUNT_OFF[_name] = _cursor
    _cursor += MOUNT_LEN
    FRESH_OFF[_name] = _cursor
    _cursor += 1
    PLATES_OK_OFF[_name] = _cursor
    _cursor += 1
    PLATES_OFF[_name] = _cursor
    _cursor += 16 * N_PLATES[_name]

#: Publish counter.  Monotonic; a viewer watching it stop knows the producer
#: died rather than that the arm is holding still.
SEQ = _cursor
_cursor += 1

#: Scene rebuild counter.  See the module docstring.
GENERATION = _cursor
_cursor += 1

#: Total length, in doubles.
VIZ_LEN = _cursor
del _cursor, _name


# ---------------------------------------------------------------------------
# Addressing helpers.  Slices rather than raw offsets, so a caller cannot
# accidentally write one element past a block and land in its neighbour.
# ---------------------------------------------------------------------------

def _check(robot: str) -> str:
    if robot not in NQ:
        raise KeyError(f"unknown robot {robot!r}; the roster is {ROBOTS}")
    return robot


def q_slice(robot: str) -> slice:
    """Where *robot*'s joint angles live."""
    _check(robot)
    return slice(Q_OFF[robot], Q_OFF[robot] + NQ[robot])


def mount_slice(robot: str) -> slice:
    """Where *robot*'s base pose lives, as ``pos(3) + quat(4)``."""
    _check(robot)
    return slice(MOUNT_OFF[robot], MOUNT_OFF[robot] + MOUNT_LEN)


def fresh_index(robot: str) -> int:
    """The single double carrying *robot*'s mount freshness flag."""
    return FRESH_OFF[_check(robot)]


def plates_ok_index(robot: str) -> int:
    """The single double carrying *robot*'s plate-block validity flag."""
    return PLATES_OK_OFF[_check(robot)]


def plates_slice(robot: str) -> slice:
    """Where *robot*'s measured plate poses live, row-major ``(n, 4, 4)``."""
    _check(robot)
    return slice(PLATES_OFF[robot], PLATES_OFF[robot] + 16 * N_PLATES[robot])


# ---------------------------------------------------------------------------
# Producing and consuming
# ---------------------------------------------------------------------------

def make_array():
    """A fresh ``multiprocessing.Array('d', VIZ_LEN, lock=False)``.

    NO LOCK, deliberately, and inherited from the RS485 original: a producer
    running a bus at 150 Hz must never block on a repaint, and every field is
    re-sent at the frame rate, so the worst a torn read costs is one frame drawn
    from two instants.  The alternative — a lock — moves a rendering hiccup onto
    the wire, where it becomes a late sync edge.
    """
    import multiprocessing as mp

    return mp.Array("d", VIZ_LEN, lock=False)


def write_robot(arr, robot: str, *, q=None, mount_pos=None, mount_quat=None,
                fresh: bool | None = None, plates=None,
                plates_ok: bool | None = None) -> None:
    """Publish one robot's block.  Every argument is optional and independent.

    Nothing here is atomic across fields, by design (see :func:`make_array`).
    Fields *within* a frame are written joints-first, flags-last, so a reader
    that samples mid-write and trusts the flag sees the flag of the older frame
    against the joints of the newer one — stale-consistent rather than
    torn-inconsistent.
    """
    _check(robot)
    if q is not None:
        off = Q_OFF[robot]
        for i in range(min(len(q), NQ[robot])):
            arr[off + i] = float(q[i])
    if plates is not None and N_PLATES[robot]:
        off = PLATES_OFF[robot]
        flat = [float(v) for row in plates for col in row for v in col]
        for i in range(min(len(flat), 16 * N_PLATES[robot])):
            arr[off + i] = flat[i]
    if mount_pos is not None:
        off = MOUNT_OFF[robot]
        for i in range(3):
            arr[off + i] = float(mount_pos[i])
    if mount_quat is not None:
        off = MOUNT_OFF[robot] + 3
        for i in range(4):
            arr[off + i] = float(mount_quat[i])
    if plates_ok is not None:
        arr[PLATES_OK_OFF[robot]] = 1.0 if plates_ok else 0.0
    if fresh is not None:
        arr[FRESH_OFF[robot]] = 1.0 if fresh else 0.0


def read_robot(arr, robot: str) -> dict:
    """One robot's block as plain Python lists plus two bools.

    Returns ``{"q", "mount_pos", "mount_quat", "fresh", "plates", "plates_ok"}``.
    ``plates`` is a flat list of ``16 * n_plates`` doubles; the caller reshapes,
    because numpy is not a dependency of this module and a viewer already has it.
    """
    _check(robot)
    qo, mo = Q_OFF[robot], MOUNT_OFF[robot]
    po = PLATES_OFF[robot]
    return {
        "q": [float(arr[qo + i]) for i in range(NQ[robot])],
        "mount_pos": [float(arr[mo + i]) for i in range(3)],
        "mount_quat": [float(arr[mo + 3 + i]) for i in range(4)],
        "fresh": float(arr[FRESH_OFF[robot]]) > 0.5,
        "plates_ok": float(arr[PLATES_OK_OFF[robot]]) > 0.5,
        "plates": [float(arr[po + i]) for i in range(16 * N_PLATES[robot])],
    }


def describe() -> str:
    """The layout as a table — for a log line, and for a human diffing two runs."""
    lines = [f"viz_layout: {VIZ_LEN} doubles, robots {list(ROBOTS)}"]
    for name in ROBOTS:
        lines.append(
            f"  {name:<8} q[{Q_OFF[name]}:{Q_OFF[name] + NQ[name]}] "
            f"mount[{MOUNT_OFF[name]}:{MOUNT_OFF[name] + MOUNT_LEN}] "
            f"fresh[{FRESH_OFF[name]}] "
            f"plates_ok[{PLATES_OK_OFF[name]}] "
            f"plates[{PLATES_OFF[name]}:{PLATES_OFF[name] + 16 * N_PLATES[name]}]")
    lines.append(f"  SEQ[{SEQ}] GENERATION[{GENERATION}]")
    return "\n".join(lines)


if __name__ == "__main__":       # pragma: no cover - a convenience, not a test
    print(describe())
