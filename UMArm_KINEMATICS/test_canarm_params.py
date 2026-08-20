"""Tests for the CAN arm's parameter table.

Two kinds of claim, and they are deliberately different in character:

1. **Structure**, which is real and testable now — a ``(3, 10)`` table with the
   same topology as the RS485 arm, ``UC1 = UC2 = 0`` (plate centre is joint
   centre), read-only, and accepted by every ``fkine`` entry point without an
   edit to any of them.  That last point is the port's whole premise: nothing
   in ``fkine`` hard-codes a length.
2. **Provenance**, which is the honest half — the lengths are placeholders and
   the module says so loudly enough that a caller needing real numbers gets a
   refusal rather than a plausible float.  When a live session finally measures
   this arm, ``test_measured_flag_still_says_placeholder`` is the test that
   fails and asks to be updated, which is the point of writing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import canarm_params as cp
from . import robot_params as rp
from .fkine import fkine, plate_transforms, ujoint_centres


class TestStructure:
    def test_shape_and_topology_match_the_rs485_table(self):
        """Three segments x two u-joints x two DOF is what makes the port a new
        table rather than new code."""
        assert cp.CANARM_PARAMS.shape == rp.DEFAULT_PARAMS.shape == (3, 10)
        assert rp.as_params(cp.CANARM_PARAMS) is not None

    def test_plate_centre_is_joint_centre(self):
        """``UC1 = UC2 = 0`` is the condition under which ``plate_chain_m``
        equals the plate-to-plate distances the mocap gate measures.  If the
        CAN arm's plates sit off the joint centres this stops being true, and
        the gate's interpretation changes even though the code still runs."""
        assert np.all(cp.CANARM_PARAMS[:, rp.COL_UC1] == 0.0)
        assert np.all(cp.CANARM_PARAMS[:, rp.COL_UC2] == 0.0)

    def test_the_table_is_read_only(self):
        """It is a default argument all over ``fkine``; an in-place fit would
        silently re-zero every other caller."""
        with pytest.raises(ValueError):
            cp.CANARM_PARAMS[0, rp.COL_LL] = 1.0

    def test_every_fkine_entry_point_accepts_it_unedited(self):
        q = np.linspace(-0.2, 0.2, 12)
        assert fkine(q, cp.CANARM_PARAMS).shape == (4, 4)
        assert ujoint_centres(q, cp.CANARM_PARAMS).shape == (6, 3)
        assert plate_transforms(q, cp.CANARM_PARAMS).shape == (6, 4, 4)

    def test_the_chain_is_five_positive_gaps(self):
        chain = cp.CANARM_PLATE_CHAIN_M
        assert len(chain) == 5
        assert all(g >= 0.0 for g in chain)
        assert chain == rp.plate_chain_m(cp.CANARM_PARAMS)


class TestProvenance:
    def test_measured_flag_still_says_placeholder(self):
        """Fails the day someone measures the arm, which is when this file
        needs its lengths and its citation updated together."""
        assert cp.MEASURED is False

    def test_require_measured_refuses_and_names_the_work(self):
        with pytest.raises(RuntimeError) as exc:
            cp.require_measured()
        assert "PLACEHOLDER" in str(exc.value)
        assert "MEASURED" in str(exc.value)

    def test_the_lengths_are_the_rs485_table_scaled(self):
        """Stated as a test so the placeholder cannot drift into looking like
        an independent measurement."""
        expected = np.array(rp.DEFAULT_PARAMS, dtype=float)
        for col in cp.LENGTH_COLUMNS:
            expected[:, col] *= cp.PLACEHOLDER_SCALE
        assert np.array_equal(cp.CANARM_PARAMS, expected)

    def test_the_non_length_columns_ride_along_untouched(self):
        """``JA*``/``AO*`` are consumed by neither ``fkine`` nor a scale
        factor; they exist so a row can go to the legacy oracle intact."""
        for col in (rp.COL_JA1, rp.COL_JA2, rp.COL_AO1, rp.COL_AO2):
            assert np.array_equal(cp.CANARM_PARAMS[:, col],
                                  rp.DEFAULT_PARAMS[:, col])
