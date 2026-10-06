# -*- coding: utf-8 -*-
"""The chi edge repair: it must not move anything when there is nothing to pair, and it must survive a
structure it cannot help. No GPU, no pipeline, no 2OIU input -- a 12-residue synthetic A-form trace.

WHY THE FIRST TEST IS THE IMPORTANT ONE. repair_base_placement changes atom coordinates in place, and it is
called from the all-atom stage of every refinement that sets TORUSFOLD_HBOND_REPAIR=1. A version that
"repairs" bases when the pair list is empty would silently rotate every base of every product, which no
measurement in findings Parts 28-30 would catch: those all have pairs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
pytest.importorskip("torch")

from torusfold.scheme2.aform_from_template import (          # noqa: E402
    reconstruct_all_atom, repair_base_placement)


def _helix(n=12, rise=2.8, twist=32.0):
    """A synthetic A-form-like P trace in Angstrom, one P per residue."""
    p = np.zeros((n, 3))
    for i in range(n):
        th = np.radians(twist * i)
        p[i] = [10.0 * np.cos(th), 10.0 * np.sin(th), rise * i]
    return p


def _coords(st):
    return np.array([[a.xyz[0], a.xyz[1], a.xyz[2]] for a in st.atoms], dtype=float)


def test_no_pairs_moves_nothing():
    seq = "AUGC" * 3
    st = reconstruct_all_atom(_helix(len(seq)), seq)
    before = _coords(st)
    n = repair_base_placement(st, seq, [])
    assert n == 0
    assert np.allclose(before, _coords(st)), "an empty pair list must leave every atom where it was"


def test_a_pair_is_handled_without_crashing():
    seq = "AUGC" * 3
    st = reconstruct_all_atom(_helix(len(seq)), seq)
    before = _coords(st)
    n = repair_base_placement(st, seq, [(0, 9)], passes=1)
    after = _coords(st)
    assert n >= 0
    assert after.shape == before.shape
    assert np.all(np.isfinite(after))
    # the sugar/phosphate atoms must not move: only base atoms rotate
    C1 = st.residue_atom_index[0]["C1'"]
    assert np.allclose(before[C1], after[C1]), "chi rotation must leave C1' fixed"


def test_unknown_base_letter_is_ignored():
    seq = "AUGC" * 3
    st = reconstruct_all_atom(_helix(len(seq)), seq)
    before = _coords(st)
    n = repair_base_placement(st, seq, [(0, 0), (1, 1)])
    assert n == 0
    assert np.allclose(before, _coords(st))
