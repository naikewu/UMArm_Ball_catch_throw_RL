r"""``load_twin_kwargs`` -- the fitted twin, as the keyword arguments every seam takes.

WHY ONE LOADER.  Four places build a twin: ``sim_core.SimArm``,
``sim_master.SimMaster`` (which the operator GUI's SIM adapter constructs),
``replay.rollout`` and ``twin_compare.twin_rollout``.  All four accept the same
keywords, since the last three forward theirs to ``SimArm``.  Before this module
each caller applied the fitted artefacts itself, and only one did:
``deliverable.score`` rebuilt the ``ActuatorModel`` with the outer fit's
``coeff`` and copied two damping scalars across.  A GUI constructing
``SimMaster()`` bare would therefore have driven the unfitted seed model while
the report card scored the fitted one.  A controller tuned against one twin and
judged against another is being judged against neither, so the fitted twin now
has exactly one spelling and every caller reads it from here.

THE TWO FILES, AND WHY THEY STAY TWO.  The flow checkpoint is fitted on pressure
residuals by multiple shooting; the mechanical file is fitted on joint residuals
by a different procedure against a different objective.  ``deliverable.load_outer``
already kept them apart for that reason -- merging them would make it possible to
load half of a fit without noticing -- and this loader keeps them apart too, and
reports per part which file it found.

WHAT A MISSING FILE MEANS.  Each part falls back to its own seed and the log line
starts with :data:`UNFITTED`, so a twin that is silently the RS485 arm's constants
cannot happen through this path.  A file that is present but malformed -- a JSON
that does not parse, a ``coeff`` that is not three positive numbers, a damping
that is negative, an ``mjcf`` key that ``generate_xml`` does not take -- raises
instead.  A broken fit replaced quietly by seeds produces a twin that is
confidently wrong, which is worse than a twin that refuses to start.

DEFAULTS AND THEIR PROVENANCE.

* :data:`DEFAULT_FLOW` -- ``checkpoints/canarm_flow.npz``, the 5-64-64-1 flow net
  fitted by multiple shooting on the 2026-09-10 sessions (held-out pressure RMS
  20308 -> 3682 Pa, ``hw_tests/report_canarm_sysid_2026-09-10.md``).
* :data:`DEFAULT_MECH` -- ``checkpoints/canarm_mech.json``, the mechanical fit
  that adds segment masses to the force-law gain and the dissipation scalars.
  When it is absent the loader falls back to :data:`FALLBACK_MECH`,
  ``checkpoints/canarm_outer.json``, the earlier ``outer_fit`` coordinate search
  (gain multipliers 0.020/0.045/0.009, ``tendon_damping`` 0.316,
  ``joint_damping`` 0.0082), which fitted no mass at all.

THE JSON SCHEMA, fixed by the shared twin-parameter contract::

    {"coeff": [3], "bf": [3] (optional),
     "tendon_damping": float, "joint_damping": float,
     "joint_frictionloss": float (optional),
     "mjcf": {generate_xml keyword: value} (optional),
     ...free-form provenance keys, ignored here...}

WHAT THIS DOES NOT DO.  It does not check that the flow checkpoint and the
mechanical file were fitted against each other; a ``coeff`` fitted on one net and
applied to another is accepted, since both files name no shared fit id to compare.
It does not judge whether an ``mjcf`` value is physically plausible, only whether
``generate_xml`` accepts the keyword.  It does not choose board variants either:
``SimMaster`` builds the nominal eight TLE/sixteen 7 mm layout, and
``replay.rollout`` takes the layout from the recording's variant bytes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

#: Where the committed fits live.
CHECKPOINT_DIR = _HERE / "checkpoints"

#: The flow net.  See the module docstring for its provenance.
DEFAULT_FLOW = CHECKPOINT_DIR / "canarm_flow.npz"

#: The mechanical fit this loader prefers.
DEFAULT_MECH = CHECKPOINT_DIR / "canarm_mech.json"

#: What :data:`DEFAULT_MECH`'s absence falls back to: the outer fit, which
#: carries the force-law gain and two damping scalars and no mass.
FALLBACK_MECH = CHECKPOINT_DIR / "canarm_outer.json"

#: Top-level scalars of the mechanical JSON that become ``generate_xml``
#: tunables.  The first two are required, the third optional.
REQUIRED_MECH_SCALARS = ("tendon_damping", "joint_damping")
OPTIONAL_MECH_SCALARS = ("joint_frictionloss",)
MECH_SCALARS = REQUIRED_MECH_SCALARS + OPTIONAL_MECH_SCALARS

#: The prefix of every line that reports a part running on its seed.  A test
#: or an operator greps for this one string.
UNFITTED = "[twin_params] UNFITTED"

#: The prefix of the line that reports a mechanical file whose ``status`` is not
#: ``complete`` -- a fit's best-so-far checkpoint, or one a run abandoned.  It is
#: loaded (a best-so-far twin is a legitimate thing to look at), but never
#: quietly: until 2026-09-10 ``mech_fit`` rewrote ``canarm_mech.json`` itself
#: every generation, and the GUI's SIM adapter then drove a half-fitted twin
#: with no line saying so.  ``mech_fit`` now writes progress beside it.
INCOMPLETE = "[twin_params] INCOMPLETE"

_AUTO = object()


class TwinKwargs(dict):
    """A ``dict`` of ``SimArm`` keywords that also remembers where they came from.

    A dict subclass rather than a dict carrying extra keys, because every key of
    this mapping is forwarded to ``SimArm`` and an unknown key there raises.
    ``**kwargs`` unpacking reads only the mapping; :attr:`provenance` travels
    beside it for :func:`describe` and for a caller that wants to record which
    files a run used.
    """

    def __init__(self, *args, provenance: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.provenance = dict(provenance or {})


def nominal_is_tle() -> np.ndarray:
    """``(24,)`` bool, the shipped eight-TLE/sixteen-7 mm layout.

    The same layout ``sim_core.NOMINAL_VARIANTS`` builds a ``SimArm`` with when
    nobody hands it variants, so a seed model made here matches the arm the GUI's
    SIM adapter constructs.
    """
    from . import sim_core as SC

    return np.array([SC.NOMINAL_VARIANTS.get(int(b)) == SC.P.VARIANT_TLE_DVP
                     for b in SC.ALL_IDS], dtype=bool)


def load_twin_kwargs(flow=DEFAULT_FLOW, mech=DEFAULT_MECH, *, log=print,
                     fallback_mech=_AUTO, validate: bool = True) -> TwinKwargs:
    """The keyword arguments that build the fitted twin.

    Returns a :class:`TwinKwargs` accepted unchanged by ``SimArm``,
    ``SimMaster``, ``replay.rollout`` and ``twin_compare.twin_rollout``:

    * ``"actuator"`` -- the flow checkpoint's ``ActuatorModel`` with ``coeff``
      (and ``bf``, when the mechanical file carries one) replaced;
    * ``"tendon_damping"``, ``"joint_damping"`` and, when present,
      ``"joint_frictionloss"``;
    * every key of the mechanical file's ``"mjcf"`` dict, passed through
      untouched.

    *fallback_mech* defaults to :data:`FALLBACK_MECH` when *mech* is
    :data:`DEFAULT_MECH` and to nothing otherwise, so a caller who names a
    specific mechanical file and mistypes it gets the seed mechanics and a loud
    line rather than a different fit.  *validate* renders the tunables through
    ``mjcf_generator.generate_xml`` once, which builds a string and compiles
    nothing, so an unknown ``mjcf`` key is refused here with the file's name on
    it rather than later inside a ``SimArm`` constructor.
    """
    say = print if log is None else log
    prov: dict = {}
    model = _load_flow(flow, say, prov)
    kwargs = TwinKwargs()
    # The same dict, not a copy: the mechanical part below still writes to it.
    kwargs.provenance = prov

    path = _resolve_mech(mech, fallback_mech, say, prov)
    if path is not None:
        data = _read_mech(path)
        model = _apply_force_law(model, data)
        for key in MECH_SCALARS:
            if key in data:
                kwargs[key] = float(data[key])
        for key, value in dict(data.get("mjcf") or {}).items():
            kwargs[key] = value
        prov["mech"] = str(path)
        prov["mech_mjcf_keys"] = sorted(dict(data.get("mjcf") or {}))
        if data.get("note"):
            prov["mech_note"] = str(data["note"])
        status = data.get("status")
        if status is not None and not str(status).startswith("complete"):
            prov["mech_status"] = str(status)
            say(f"{INCOMPLETE} mechanical fit: {path} has status {str(status)!r}. "
                f"This is a fit's best-so-far or an abandoned run, not a finished "
                f"fit; the twin below is built from it anyway")
    kwargs["actuator"] = model

    if validate:
        _validate_tunables(kwargs, path)
    say(describe(kwargs))
    return kwargs


def describe(kwargs) -> str:
    """One line: which files, and every number that differs from a seed.

    Accepts a plain dict too, in which case the file names are simply absent.
    """
    prov = getattr(kwargs, "provenance", None) or {}
    parts = []
    if "flow" in prov:
        flow = prov["flow"]
        text = "flow " + (Path(flow).name if flow else "UNFITTED seed net")
        if prov.get("flow_holdout_rms_psi") is not None:
            text += f" (holdout {prov['flow_holdout_rms_psi']:.3f} psi)"
        parts.append(text)
    if "mech" in prov:
        mech = prov["mech"]
        parts.append("mech " + (Path(mech).name if mech
                                else "UNFITTED generator defaults")
                     + (f" (INCOMPLETE: {prov['mech_status']})"
                        if prov.get("mech_status") else ""))
    actuator = kwargs.get("actuator")
    if actuator is not None:
        parts.append("coeff " + "/".join(f"{float(c):.4g}"
                                         for c in np.asarray(actuator.coeff)))
    for key in sorted(k for k in kwargs if k != "actuator"):
        value = kwargs[key]
        parts.append(f"{key} {value:.4g}" if isinstance(value, float)
                     else f"{key} {value}")
    return "twin: " + (" | ".join(parts) if parts else "seed model, no files")


# ---------------------------------------------------------------------------
# The parts
# ---------------------------------------------------------------------------

def _load_flow(flow, say, prov: dict):
    from .actuator_model import ActuatorModel

    if flow is not None and Path(flow).is_file():
        # ActuatorModel.load raises on a normalisation mismatch, and that is
        # left to propagate: a present checkpoint that does not load is a
        # broken fit, not a missing one.
        model = ActuatorModel.load(str(flow))
        prov["flow"] = str(Path(flow))
        meta = getattr(model, "meta", None) or {}
        if meta.get("holdout_rms_psi") is not None:
            prov["flow_holdout_rms_psi"] = float(meta["holdout_rms_psi"])
        return model
    say(f"{UNFITTED} flow net: no checkpoint at {flow}. The twin runs "
        f"ActuatorModel.fresh -- a small random net, unit gains, zero leak, and "
        f"the RS485 arm's coeff/bf/damp_b1 seeds -- which does not pressurise "
        f"like this arm (seed model on the held-out sequence: 14.766 deg joint "
        f"RMS, nrmse 1.289)")
    prov["flow"] = None
    return ActuatorModel.fresh(is_tle=nominal_is_tle())


def _resolve_mech(mech, fallback_mech, say, prov: dict):
    if fallback_mech is _AUTO:
        fallback_mech = (FALLBACK_MECH
                         if mech is not None and _same_path(mech, DEFAULT_MECH)
                         else None)
    if mech is not None and Path(mech).is_file():
        return Path(mech)
    if fallback_mech is not None and Path(fallback_mech).is_file():
        say(f"[twin_params] no mechanical fit at {mech}; falling back to "
            f"{fallback_mech}")
        prov["mech_fallback_from"] = str(mech)
        return Path(fallback_mech)
    say(f"{UNFITTED} mechanics: no file at {mech}"
        + (f" nor at {fallback_mech}" if fallback_mech is not None else "")
        + ". The force-law coeff/bf stay at the flow checkpoint's own values and "
          "every damping and mass is mjcf_generator's default -- the RS485 "
          "carry-overs hw_tests/report_canarm_sysid_2026-09-10.md lists")
    prov["mech"] = None
    return None


def _read_mech(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be an object; got "
                         f"{type(data).__name__}")

    _positive_vec(data.get("coeff"), "coeff", path)
    if "bf" in data:
        _positive_vec(data["bf"], "bf", path)
    for key in REQUIRED_MECH_SCALARS:
        if key not in data:
            raise ValueError(f"{path}: '{key}' is required by the twin-parameter "
                             f"contract and is missing")
    for key in MECH_SCALARS:
        if key in data:
            _non_negative(data[key], key, path)

    mjcf = data.get("mjcf", {})
    if mjcf is None:
        mjcf = {}
    if not isinstance(mjcf, dict):
        raise ValueError(f"{path}: 'mjcf' must be an object of generate_xml "
                         f"keywords; got {type(mjcf).__name__}")
    for key in mjcf:
        if not isinstance(key, str):
            raise ValueError(f"{path}: 'mjcf' key {key!r} is not a string")
        if key in MECH_SCALARS or key == "actuator":
            # Two spellings of one tunable in one file leave no right answer
            # about which one the author meant, so neither is picked.
            raise ValueError(f"{path}: '{key}' appears both at the top level "
                             f"and inside 'mjcf'; keep one")
    return data


def _apply_force_law(model, data: dict):
    """A new ``ActuatorModel`` with ``coeff`` (and ``bf``) replaced.

    Rebuilt through the constructor exactly as ``deliverable.score`` rebuilt it
    for the 2026-09-10 validation numbers, so the model this loader returns is
    the one those numbers describe, and so the constructor's shape checks run on
    the new vectors.
    """
    from .actuator_model import ActuatorModel

    new = ActuatorModel(
        net=model.net, is_tle=model.is_tle,
        fill_gain=model.fill_gain, vent_gain=model.vent_gain,
        blend_width_pa=model.blend_width_pa, leak_pa_s=model.leak_pa_s,
        coeff=np.asarray(data["coeff"], dtype=float),
        bf=(np.asarray(data["bf"], dtype=float) if "bf" in data else model.bf),
        l0=model.l0, damp_b1=model.damp_b1)
    new.meta = getattr(model, "meta", None)
    return new


def _validate_tunables(kwargs: dict, path) -> None:
    from . import mjcf_generator as MG

    tunables = {k: v for k, v in kwargs.items() if k != "actuator"}
    try:
        MG.generate_xml(**tunables)
    except TypeError as exc:
        raise ValueError(
            f"{path}: a tunable is not accepted by mjcf_generator.generate_xml "
            f"({exc}); the 'mjcf' dict must name generate_xml keywords") from exc


def _positive_vec(value, name: str, path) -> None:
    arr = np.asarray(value if value is not None else [], dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{path}: '{name}' must be three per-segment numbers; "
                         f"got {value!r}")
    if not (np.all(np.isfinite(arr)) and np.all(arr > 0.0)):
        raise ValueError(f"{path}: '{name}' must be finite and positive; got "
                         f"{arr.tolist()}")


def _non_negative(value, name: str, path) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: '{name}' must be a number; got {value!r}") from exc
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{path}: '{name}' must be finite and non-negative; "
                         f"got {v}")


def _same_path(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:                                          # pragma: no cover
        return str(a) == str(b)


__all__ = [
    "CHECKPOINT_DIR", "DEFAULT_FLOW", "DEFAULT_MECH", "FALLBACK_MECH",
    "REQUIRED_MECH_SCALARS", "OPTIONAL_MECH_SCALARS", "MECH_SCALARS",
    "UNFITTED", "INCOMPLETE", "TwinKwargs", "nominal_is_tle", "load_twin_kwargs",
    "describe",
]
