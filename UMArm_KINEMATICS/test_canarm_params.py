"""Tests for the CAN arm's parameter table.

Two kinds of claim, and they are deliberately different in character:

1. **Structure**, which is real and testable now — a ``(3, 10)`` table with the
   same topology as the RS485 arm, ``UC1 = UC2 = 0`` (plate centre is joint
   centre), read-only, and accepted by every ``fkine`` entry point without an
   edit to any of them.  That last point is the port's whole premise: nothing
   in ``fkine`` hard-codes a length.
2. **Provenance**, which since 2026-08-21 is a measurement rather than a
   placeholder.  The tests below pin what was measured and what was not: the
   five u-joint centre distances came off the arm, the ``UC``/``AA``/``JA``/
   ``AO`` columns are CAD numbers a mocap campaign cannot see, and the table
   must reproduce ``CANARM_PLATE_CHAIN_M`` exactly — a table and a chain tuple
   that disagree would be two answers to one question.
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
    def test_the_lengths_are_measured_and_cited(self):
        """``MEASURED`` and its citation move together or not at all."""
        assert cp.MEASURED is True
        assert "drive_2026-08-21" in cp.MEASURED_SOURCE
        assert cp.require_measured() is cp.CANARM_PARAMS

    def test_the_table_reproduces_the_measured_chain_exactly(self):
        """The table and the chain tuple are two spellings of one measurement.
        ``LL`` is derived as ``span - (AA1 + AA2)``, so the round trip has to
        land on the same doubles; a mismatch means someone edited one of them."""
        chain = rp.plate_chain_m(cp.CANARM_PARAMS)
        assert np.allclose(chain, cp.CANARM_PLATE_CHAIN_M, atol=5e-7)

    def test_it_is_not_the_rs485_table_in_disguise(self):
        """The placeholder this file used to carry was exactly that, and it put
        the last u-joint centre 182 mm from where the cameras see it."""
        assert not np.allclose(cp.CANARM_PARAMS[:, rp.COL_LL],
                               rp.DEFAULT_PARAMS[:, rp.COL_LL], atol=1e-3)
        legacy_chain = rp.plate_chain_m(rp.DEFAULT_PARAMS)
        assert abs(sum(cp.CANARM_PLATE_CHAIN_M) - sum(legacy_chain)) > 0.15

    def test_it_agrees_with_the_research_tree_to_two_millimetres(self):
        """The substantive finding: the legacy CAN table was right.  Stated as
        a bound rather than an equality because the measurement is the
        authority and the CAD table is the corroboration, not the other way
        round."""
        legacy = rp.plate_chain_m(cp.LEGACY_CANARM_PARAMS)
        diff = np.abs(np.array(cp.CANARM_PLATE_CHAIN_M) - np.array(legacy))
        assert diff.max() < 2e-3, diff

    def test_the_cad_columns_match_the_research_tree_exactly(self):
        """``UC``/``AA``/``JA``/``AO`` are transcribed, not fitted; only ``LL``
        and ``JD`` carry the measurement."""
        for col in (rp.COL_JA1, rp.COL_JA2, rp.COL_UC1, rp.COL_UC2,
                    rp.COL_AA1, rp.COL_AA2, rp.COL_AO1, rp.COL_AO2):
            assert np.array_equal(cp.CANARM_PARAMS[:, col],
                                  cp.LEGACY_CANARM_PARAMS[:, col]), col
