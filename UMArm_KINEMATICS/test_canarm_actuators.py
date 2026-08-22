"""Tests for the CAN arm's actuator/axis map.

The map is data, so most of what can go wrong with it is structural: a board
named twice, a joint with two positive drivers, a pair split across segments.
Those are the checks below, plus the two substantive claims the 2026-08-21
campaign made — segments 2 and 3 reproduce the legacy table exactly, and
segment 1 is the same four pairs rotated by 90 deg.

The second claim is worth pinning as an *equality* rather than as prose,
because it is the difference between "the top platform was rewired" and "the
measurement disagreed with the legacy file", and only the first of those is
consistent with sixteen other boards agreeing.
"""

from __future__ import annotations

import pytest

from . import canarm_actuators as ca


class TestStructure:
    def test_twelve_pairs_of_twenty_four_distinct_boards(self):
        pairs = ca.joint_pairs()
        assert len(pairs) == 12
        flat = [b for pair in pairs for b in pair]
        assert len(flat) == 24
        assert len(set(flat)) == 24
        assert set(flat) == set(range(0x101, 0x119))

    def test_each_pair_lives_on_one_segment(self):
        """An antagonistic pair is two tubes on one joint; the two boards
        driving it are on that segment's own regulator platform, and the
        campaign confirmed all 24 that way."""
        for j, (pos, neg) in enumerate(ca.joint_pairs()):
            block = ca.SEGMENT_BLOCKS[j // 4]
            assert pos in block and neg in block, (j, hex(pos), hex(neg))

    def test_base_to_joint_is_the_inverse_and_signs_are_opposite(self):
        table = ca.base_to_joint()
        assert len(table) == 24
        for j, (pos, neg) in enumerate(ca.joint_pairs()):
            assert table[pos] == (j, +1)
            assert table[neg] == (j, -1)

    def test_the_joint_names_line_up_with_the_q_ordering(self):
        assert len(ca.JOINT_NAMES) == 12
        for j, name in enumerate(ca.JOINT_NAMES):
            seg, ujoint, axis = name.split(".")
            assert seg == f"s{j // 4 + 1}"
            assert axis == f"t{j % 4 + 1}"
            assert ujoint == f"u{2 * (j // 4) + (0 if j % 4 < 2 else 1) + 1}"


class TestLegacyTable:
    def test_the_legacy_composition_is_transcribed_correctly(self):
        """Guards the transcription itself: the legacy file stores an address
        list and an index table, and this module stores their composition."""
        assert len(ca.LEGACY_ADDRESS_LIST) == 24
        assert len(set(ca.LEGACY_ADDRESS_LIST)) == 24
        for j, (pos_i, neg_i) in enumerate(ca.LEGACY_JOINT_LIST_INDEX):
            assert ca.LEGACY_JOINT_PAIRS[j] == (ca.LEGACY_ADDRESS_LIST[pos_i],
                                                ca.LEGACY_ADDRESS_LIST[neg_i])

    def test_segments_two_and_three_reproduce_the_legacy_table_exactly(self):
        """Sixteen boards agreeing, pairing and sign, is also what fixes the
        sign convention: a robot frame turned 180 deg about z would flip every
        one of them at once."""
        for j in range(4, 12):
            assert ca.MEASURED_JOINT_PAIRS[j] == ca.LEGACY_JOINT_PAIRS[j], j

    def test_segment_one_is_the_same_pairs_rotated_by_ninety_degrees(self):
        """A quarter turn maps a pair on +x to +y and a pair on +y to -x.  In
        the joint ordering ``(t1, t2)`` and ``(t3, t4)`` that is: the old t1
        pair becomes the t2 pair unchanged, and the old t2 pair becomes the t1
        pair with its two boards swapped."""
        for lo in (0, 2):
            old_a, old_b = ca.LEGACY_JOINT_PAIRS[lo], ca.LEGACY_JOINT_PAIRS[lo + 1]
            new_a, new_b = ca.MEASURED_JOINT_PAIRS[lo], ca.MEASURED_JOINT_PAIRS[lo + 1]
            assert new_b == old_a, (lo, new_b, old_a)
            assert new_a == (old_b[1], old_b[0]), (lo, new_a, old_b)

    def test_the_two_tables_use_the_same_boards(self):
        assert (sorted(b for p in ca.MEASURED_JOINT_PAIRS for b in p)
                == sorted(b for p in ca.LEGACY_JOINT_PAIRS for b in p))


class TestProvenance:
    def test_measured_and_cited(self):
        assert ca.MEASURED is True
        assert "drive_2026-08-21" in ca.MEASURED_SOURCE
        assert ca.joint_pairs() is ca.MEASURED_JOINT_PAIRS

    def test_the_legacy_table_needs_asking_for(self):
        """It describes a regulator platform that no longer exists; a caller
        that wants it has to say so."""
        assert ca.joint_pairs(allow_legacy=True) is ca.MEASURED_JOINT_PAIRS
        assert ca.LEGACY_JOINT_PAIRS != ca.MEASURED_JOINT_PAIRS

    def test_an_unmeasured_map_refuses(self, monkeypatch):
        """The state this module shipped in before the campaign, and the state
        the next re-plumbing puts it back into."""
        monkeypatch.setattr(ca, "MEASURED", False)
        with pytest.raises(RuntimeError, match="has not been measured"):
            ca.joint_pairs()
        assert ca.joint_pairs(allow_legacy=True) is ca.LEGACY_JOINT_PAIRS

    def test_the_known_faults_name_boards_that_exist(self):
        for base, note in ca.KNOWN_BOARD_FAULTS.items():
            assert 0x101 <= base <= 0x118
            assert note
        assert 0x110 in ca.KNOWN_BOARD_FAULTS
