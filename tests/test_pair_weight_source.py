"""Which quantity weights the base-pair restraints, and which alphabet it is computed in.

Both of these were measured before they were changed, and both measurements are recorded at
their call sites. These tests keep the decisions from silently reverting.

1. isrnaclong_pipeline's pair weights. It used to overwrite every method-agreement weight with
   compute_rcm_score(...)['confidence']. Measured against the WC pairs of the structure
   database (scripts/measure_pair_weight_quality.py): AUC 0.5019 with a 95 percent CI of
   [0.4939, 0.5093] against geometry-matched negatives, sequence-specific component +0.0021
   with a CI spanning zero, and 88.84 percent of 2330 true pairs scoring exactly 0.0. Since
   pair_w multiplies the WC spring stiffness, that switched the restraint off for about 89
   percent of the pairs. The switch now defaults to False.

2. rcm.py's complement table. It mapped A -> T, which is DNA. Fed an RNA sequence it made
   A-U visible in one orientation and invisible in the other.

Run: python -m pytest tests/test_pair_weight_source.py
"""
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

rcm = pytest.importorskip("torusfold.scheme2.rcm")
isrnaclong = pytest.importorskip("torusfold.scheme2.isrnaclong")


def test_pair_weights_do_not_default_to_the_rcm_confidence():
    """The RCM confidence is at chance and is zero for most true pairs, so it is opt-in."""
    sig = inspect.signature(isrnaclong.isrnaclong_pipeline)
    assert "use_rcm_reweight" in sig.parameters, (
        "the switch is gone; without it the RCM reweighting cannot be turned off")
    assert sig.parameters["use_rcm_reweight"].default is False, (
        "the default flipped back to True, which overwrites the method-agreement weights with "
        "a quantity that is 0 for 88.84 percent of true pairs and scores AUC 0.5019")


def test_the_complement_table_is_rna_not_dna():
    comp = rcm._COMPLEMENT
    assert comp["A"] == "U", f"_COMPLEMENT maps A -> {comp['A']!r}; that is the DNA complement"
    assert comp["U"] == "A"
    assert comp["C"] == "G" and comp["G"] == "C"


def test_an_au_pair_is_visible_in_both_orientations():
    """The DNA table made exactly one orientation of every A-U match invisible."""
    au = rcm.rcm_crossing("A", "U", 1)[0]
    ua = rcm.rcm_crossing("U", "A", 1)[0]
    gc = rcm.rcm_crossing("G", "C", 1)[0]
    cg = rcm.rcm_crossing("C", "G", 1)[0]
    assert au == 1 and ua == 1, f"A-U is asymmetric: A,U -> {au}, U,A -> {ua}"
    assert gc == 1 and cg == 1, f"C-G is asymmetric: G,C -> {gc}, C,G -> {cg}"


def test_gu_is_left_alone_on_purpose():
    """Whether a wobble counts as a crossing is a modelling choice, not an alphabet error.

    The alphabet fix turned A -> T into A -> U and stopped there. If someone later decides
    G-U should count, this test is the record that it was a decision and not an oversight --
    change it deliberately or not at all.
    """
    assert rcm.rcm_crossing("G", "U", 1)[0] == 0
    assert rcm.rcm_crossing("U", "G", 1)[0] == 0
