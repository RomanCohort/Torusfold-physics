"""Level 0's pair vote, and the one source that never reached it.

isrnaclong_pipeline's Level 0 fuses four pair sources into the hard-restraint set. The
caller-supplied secondary_structure was not one of them: run_2013nt.py computes the MUSES
consensus, writes it to test_2013nt_ss.txt, passes it in as secondary_structure -- and Level 0
never read the argument. It recomputed its own sources and voted among those.

The fix admits the input as a fourth voter rather than as an authority. That distinction is the
whole point: for L > 500 the upstream text is linear ViennaRNA alone (multisource_ss drops its
Nussinov source above 500 nt, and _vienna_fold_consensus folds without md.circ=1), while every
other Level 0 source folds with circ=1. A single linear-model vote must not make a pair hard.

These tests pin the vote, including the pre-existing quirk that pf_high_set is admitted
outright -- the >=2 threshold never applied to it.

Run: python -m pytest tests/test_level0_pair_vote.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

isrnaclong = pytest.importorskip("torusfold.scheme2.isrnaclong")

fuse = isrnaclong._fuse_pair_sources


def test_a_lone_input_ss_pair_does_not_become_hard():
    """One linear-model vote is not enough to override the circular evidence."""
    _, hard_set, _, stats = fuse(set(), set(), set(), {(5, 40)})
    assert (5, 40) not in hard_set, (
        "a pair named only by the caller's (linear) secondary structure became hard")
    assert stats["n_input_ss_hard"] == 0
    assert stats["n_input_ss_single"] == 1


def test_input_ss_pairs_with_a_second_source_become_hard():
    """Two votes make it hard, and the diagnostics report that the input supplied the second."""
    _, hard_set, _, stats = fuse(set(), {(5, 40)}, set(), {(5, 40)})
    assert (5, 40) in hard_set
    assert stats["n_input_ss_hard"] == 1


def test_pf_high_pairs_are_hard_regardless_of_the_vote():
    """Pre-existing behaviour, kept: the P>0.9 tier is admitted outright."""
    _, hard_set, _, _ = fuse({(1, 30)}, set(), set(), set())
    assert (1, 30) in hard_set


def test_mfe_and_divide_still_need_both_to_agree():
    """The threshold discriminates between mfe/divide: either alone is one vote, not two."""
    _, hard_alone, _, _ = fuse(set(), {(1, 30)}, set(), set())
    assert (1, 30) not in hard_alone
    _, hard_both, _, _ = fuse(set(), {(1, 30)}, {(1, 30)}, set())
    assert (1, 30) in hard_both


def test_an_input_that_adds_nothing_to_mfe_is_dropped_entirely():
    """run_2013nt.py's fallback route hands in this very fold, so counting it would let one
    source supply the second vote for its own pairs and turn every MFE pair hard."""
    _, hard_set, _, stats = fuse(set(), {(1, 30)}, set(), {(1, 30)})
    assert (1, 30) not in hard_set, (
        "the input duplicated mfe_set and was counted twice, turning the pair hard")
    assert stats["n_input_ss_raw"] == 1 and stats["n_input_ss"] == 0

    # a strict subset is the same situation
    _, hard_set, _, stats = fuse(set(), {(1, 30), (2, 31)}, set(), {(1, 30)})
    assert stats["n_input_ss"] == 0


def test_an_input_carrying_new_pairs_votes_on_all_of_them():
    """With multisource_ss working the input is a LINEAR fold: a pair both models find is two
    independent models agreeing, which is what the >=2 threshold is for."""
    _, hard_set, _, stats = fuse(set(), {(1, 30), (2, 31)}, set(), {(1, 30), (4, 50)})
    assert (1, 30) in hard_set, "linear and circular agreed, but the vote did not count it"
    assert (2, 31) not in hard_set       # circular only -> one vote
    assert (4, 50) not in hard_set       # linear only  -> one vote
    assert stats["n_input_ss_raw"] == 2 and stats["n_input_ss"] == 2


def test_admitting_a_fourth_voter_never_un_hardens_a_pair():
    """The input can only add hard pairs; it must not drop ones the first three agreed on."""
    pf, mfe, div = {(1, 30)}, {(1, 30)}, {(2, 31)}
    _, hard_before, _, _ = fuse(pf, mfe, div, set())
    _, hard_after, _, _ = fuse(pf, mfe, div, {(3, 32)})
    assert hard_before <= hard_after
    assert (3, 32) not in hard_after
