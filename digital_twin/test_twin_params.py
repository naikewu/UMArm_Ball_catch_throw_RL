"""``twin_params``: the fitted twin has one spelling, and a missing part says so.

Offline and hardware-free.  The committed checkpoints under
``digital_twin/checkpoints/`` are the inputs; every file this suite writes goes to
pytest's ``tmp_path``.  What is pinned:

* the committed flow checkpoint and mechanical file are applied key for key;
* each missing part falls back to its own seed with an ``UNFITTED`` line, and the
  default mechanical file falls back to the outer fit while an explicitly named
  one does not;
* a present but malformed file raises instead of falling back;
* ``mjcf`` keys pass through untouched, and an unknown one is refused early;
* the returned mapping builds ``SimArm`` and ``SimMaster`` and rolls a recording.

What it does not show: that the fitted twin moves like the arm.  That is
``twin_compare``'s question, on the held-out sequence.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from digital_twin import twin_params as TP
from digital_twin.actuator_model import ActuatorModel


def _logger():
    lines: list[str] = []
    return lines, lines.append


def _mech_in_use():
    return TP.DEFAULT_MECH if TP.DEFAULT_MECH.is_file() else TP.FALLBACK_MECH


def _good_mech(**overrides) -> dict:
    data = {"coeff": [0.0034, 0.0054, 0.0011],
            "tendon_damping": 0.3, "joint_damping": 0.008,
            "fit_note": "free-form provenance, ignored by the loader"}
    data.update(overrides)
    return data


def _write(tmp_path, data, name="mech.json"):
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data),
                    encoding="utf-8")
    return path


def _existing_tunable():
    """A ``generate_xml`` keyword that exists at the time the suite runs.

    Chosen by signature rather than hard-coded, because the generator's mass
    arguments are being reworked in parallel and this suite tests the
    pass-through, not any particular name.
    """
    from digital_twin import mjcf_generator as MG

    names = set(inspect.signature(MG.generate_xml).parameters)
    names |= set(inspect.signature(MG.build_arm_xml).parameters)
    for key in ("joint_armature", "tip_mass", "plate_mass"):
        if key in names:
            return key
    pytest.skip("no known scalar tunable on generate_xml to pass through")


# ---------------------------------------------------------------------------
# The committed fit
# ---------------------------------------------------------------------------

def test_the_defaults_name_the_committed_files():
    assert TP.DEFAULT_FLOW.name == "canarm_flow.npz" and TP.DEFAULT_FLOW.is_file()
    assert TP.DEFAULT_MECH.name == "canarm_mech.json"
    assert TP.FALLBACK_MECH.name == "canarm_outer.json"
    assert _mech_in_use().is_file()


def test_the_committed_fit_is_applied_key_for_key():
    lines, log = _logger()
    kw = TP.load_twin_kwargs(log=log)
    data = json.loads(_mech_in_use().read_text(encoding="utf-8"))
    flow = ActuatorModel.load(str(TP.DEFAULT_FLOW))

    act = kw["actuator"]
    np.testing.assert_array_equal(act.coeff, np.asarray(data["coeff"], dtype=float))
    np.testing.assert_array_equal(
        act.bf, np.asarray(data["bf"], dtype=float) if "bf" in data else flow.bf)
    # Everything the mechanical file does not name is the checkpoint's own.
    np.testing.assert_array_equal(act.net.W1, flow.net.W1)
    np.testing.assert_array_equal(act.leak_pa_s, flow.leak_pa_s)
    np.testing.assert_array_equal(act.l0, flow.l0)
    for key in TP.REQUIRED_MECH_SCALARS:
        assert kw[key] == float(data[key])
    if "joint_frictionloss" in data:
        assert kw["joint_frictionloss"] == float(data["joint_frictionloss"])
    else:
        assert "joint_frictionloss" not in kw
    for key, value in (data.get("mjcf") or {}).items():
        assert kw[key] == value
    assert not any(line.startswith(TP.UNFITTED) for line in lines)
    assert kw.provenance["flow"] and kw.provenance["mech"]


# ---------------------------------------------------------------------------
# Missing parts: seeds, loudly
# ---------------------------------------------------------------------------

def test_a_missing_flow_checkpoint_is_the_seed_model_and_says_so(tmp_path):
    lines, log = _logger()
    kw = TP.load_twin_kwargs(flow=tmp_path / "absent.npz", log=log)
    assert any(line.startswith(TP.UNFITTED) and "flow" in line for line in lines)
    assert kw["actuator"].meta is None
    np.testing.assert_array_equal(kw["actuator"].is_tle, TP.nominal_is_tle())
    assert kw.provenance["flow"] is None
    # The mechanics are a separate part and still load.
    assert "tendon_damping" in kw


def test_a_missing_mechanical_file_is_the_generator_default_and_says_so(tmp_path):
    lines, log = _logger()
    kw = TP.load_twin_kwargs(mech=tmp_path / "absent.json", log=log)
    assert any(line.startswith(TP.UNFITTED) and "mechanics" in line for line in lines)
    assert not any(k in kw for k in TP.MECH_SCALARS)
    flow = ActuatorModel.load(str(TP.DEFAULT_FLOW))
    np.testing.assert_array_equal(kw["actuator"].coeff, flow.coeff)
    assert kw.provenance["mech"] is None


def test_the_default_mechanical_file_falls_back_to_the_outer_fit(tmp_path):
    lines, log = _logger()
    kw = TP.load_twin_kwargs(mech=tmp_path / "canarm_mech.json",
                             fallback_mech=TP.FALLBACK_MECH, log=log)
    outer = json.loads(TP.FALLBACK_MECH.read_text(encoding="utf-8"))
    assert kw["tendon_damping"] == float(outer["tendon_damping"])
    np.testing.assert_array_equal(kw["actuator"].coeff, outer["coeff"])
    assert any("falling back" in line for line in lines)
    assert not any(line.startswith(TP.UNFITTED) for line in lines)


def test_an_explicitly_named_missing_file_does_not_borrow_the_outer_fit(tmp_path):
    lines, log = _logger()
    kw = TP.load_twin_kwargs(mech=tmp_path / "typo.json", log=log)
    assert "tendon_damping" not in kw
    assert any(line.startswith(TP.UNFITTED) for line in lines)


# ---------------------------------------------------------------------------
# Present files: applied, or refused
# ---------------------------------------------------------------------------

def test_mjcf_keys_and_optional_scalars_pass_through_untouched(tmp_path):
    key = _existing_tunable()
    data = _good_mech(bf=[0.02, 0.02, 0.02], joint_frictionloss=0.02,
                      mjcf={key: 0.0123})
    kw = TP.load_twin_kwargs(mech=_write(tmp_path, data), log=None)
    assert kw[key] == 0.0123
    assert kw["joint_frictionloss"] == 0.02
    np.testing.assert_array_equal(kw["actuator"].bf, [0.02, 0.02, 0.02])
    assert "fit_note" not in kw
    assert kw.provenance["mech_mjcf_keys"] == [key]


@pytest.mark.parametrize("bad", [
    "{ this is not json",
    _good_mech(coeff=[0.003, 0.005]),
    _good_mech(coeff=[0.003, -0.005, 0.001]),
    _good_mech(bf=[0.02, 0.02]),
    _good_mech(tendon_damping=-1.0),
    _good_mech(joint_frictionloss=float("nan")),
    {k: v for k, v in _good_mech().items() if k != "joint_damping"},
    _good_mech(mjcf=[1, 2, 3]),
    _good_mech(mjcf={"tendon_damping": 0.5}),
])
def test_a_malformed_file_raises_rather_than_falling_back(tmp_path, bad):
    with pytest.raises(ValueError):
        TP.load_twin_kwargs(mech=_write(tmp_path, bad), log=None)


def test_an_unknown_mjcf_key_is_refused_before_any_arm_is_built(tmp_path):
    path = _write(tmp_path, _good_mech(mjcf={"no_such_tunable_xyz": 1.0}))
    with pytest.raises(ValueError, match="no_such_tunable_xyz"):
        TP.load_twin_kwargs(mech=path, log=None)


# ---------------------------------------------------------------------------
# The seams accept it
# ---------------------------------------------------------------------------

def _short_recording(n: int = 45):
    """A 0.3 s all-idle recording in the real schema's arrays, no file."""
    from digital_twin import replay as R
    from digital_twin import sim_core as SC

    ids = np.array(list(SC.ALL_IDS), dtype=int)
    variants = np.array([SC.NOMINAL_VARIANTS[int(b)] for b in ids], dtype=int)
    t = np.arange(n, dtype=float) / 150.0
    zeros24 = np.zeros((n, 24))
    return R.Recording(
        t_sync_s=t, t_rel_s=t - t[0], cycle=np.arange(n), ids=ids,
        board_type=variants, is_tle=variants == SC.P.VARIANT_TLE_DVP,
        q_rad=np.zeros((n, 12)), qdot_rad_s=np.zeros((n, 12)),
        p_adc=zeros24.astype(int), target_adc=zeros24.astype(int),
        p_pa=zeros24.copy(), target_pa=zeros24.copy(),
        q_valid=np.ones(n, dtype=bool))


def test_the_kwargs_build_simarm_and_simmaster_and_roll_a_recording():
    from digital_twin import replay as R
    from digital_twin import sim_core as SC
    from digital_twin import sim_master as SM

    kw = TP.load_twin_kwargs(log=None)
    arm = SC.SimArm(**kw)
    assert arm.actuator is kw["actuator"]
    np.testing.assert_allclose(arm.model.dof_damping, kw["joint_damping"])

    master = SM.SimMaster(**kw)
    assert not master.link.is_open            # constructed, never opened
    np.testing.assert_array_equal(master.arm.actuator.coeff, kw["actuator"].coeff)

    rec = _short_recording()
    roll = R.rollout(rec, **kw)
    assert roll.n == rec.n
    assert np.isfinite(roll.q_rad).all() and np.isfinite(roll.p_pa).all()


def test_describe_is_one_line_naming_both_files():
    kw = TP.load_twin_kwargs(log=None)
    line = TP.describe(kw)
    assert "\n" not in line
    assert TP.DEFAULT_FLOW.name in line and _mech_in_use().name in line
    assert TP.describe({"joint_damping": 0.01}).startswith("twin: ")
