# Where these files came from

Copied 2026-08-19 from

```
C:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\
UMARM_Variable_Stiffness_Oct2025\kinova_arm_driver\
```

which the operator describes as the lab's verified working Kinova connector.

| file here | file there | changed? |
|---|---|---|
| `kinova_driver.py` | `kinova_driver.py` | **one line** — see below |
| `utilities.py` | `utilities.py` | verbatim |
| `AGENT_GUIDE.md` | `AGENT_GUIDE.md` | verbatim |
| `../wheels/kortex_api-2.6.0.post3-py3-none-any.whl` | `kinova/…whl` | verbatim (byte-identical) |

`utilities.py` and the wheel are Kinova's own, redistributed under the BSD
3-Clause licence that ships with the Kortex API; `kinova_driver.py` is the lab's
wrapper over them.

## The one edit

`kinova_driver.py` originally read

```python
import utilities  # shipped in this folder
```

which resolves only when the file is run as a loose script from its own
directory.  It now tries the packaged import first and falls back to the
original:

```python
try:
    from . import utilities
except ImportError:                    # pragma: no cover - loose-script path
    import utilities
```

`test_kinova_env.py` asserts that this is the *only* difference against the
origin copy whenever that copy is reachable, so a future re-vendoring that
silently drops a fix fails a test instead of a run.

## Why the wheel is committed

`kortex_api` is not on PyPI.  The wheel is 165 kB, pure Python
(`py3-none-any`), and without it this package cannot be installed at all on a
machine that has never had the lab's other project checked out.  See
`../SETUP.md` for the install, including the protobuf patch Python ≥ 3.10
needs.
