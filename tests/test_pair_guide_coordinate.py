# -*- coding: utf-8 -*-
"""Which atoms does the pair guide actually read?

THE DEFECT THIS PINS. `cg_energy_forces` applies its pair guide to the P beads
(`pos_nm[:, P(pi)] - pos_nm[:, P(pj)]`, torch_cgsim.py:1401) while the constant it compares
against, `PAIR_NN = 1.00 nm`, is documented as "pairing target on N beads; native
0.954 +/- 0.115 nm" (:377). The BPP term twenty lines below reads `NN(pi)`/`NN(pj)` (:1438) --
the asymmetry inside one function is the evidence.

`_sigmoid_f` charges `+k*softplus((r - r0)/width)`: nothing when the pair is closer than r0,
a linearly growing pull when it is farther. So the guide pulls the PHOSPHATES of every paired
residue toward 10 A. Measured on 2OIU: the crystal's paired P-P distance is 18.16 A (its N-N
is 9.86 A, dead on target), and the run outputs' paired P-P collapses to 8.4 A with minima at
3.96 A -- two phosphates 4 A apart. That is the mechanism behind the 9 A offset and the
2500 kJ/mol energy gap between the experimental structure and the field's own output
(`scripts/crystal_energy_2oiu.py`).

WHAT THE TEST DOES. The guide's contribution is isolated by zeroing `K_PAIR_GUIDE` and
differencing, so every other term cancels -- including BPP, which also reads N beads. Then the
same state is perturbed twice, once moving only P beads and once moving only N beads:

  * moving the P beads of a paired pair apart must NOT change the guide's energy;
  * moving the N beads apart MUST.

The first assertion fails on the shipped code and passes after `P(` -> `NN(`. The second keeps
the test from being satisfied by a term that reads nothing at all.

Needs torch.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")
C = pytest.importorskip("torusfold.scheme2.torch_cgsim")

L = 6
PAIR = (0, 5)


def _state(p_beads=None, n_beads=None):
    """A flat 6-residue trace in nm, 3 beads each, with one declared pair."""
    pos = torch.zeros(1, 3 * L, 3, dtype=torch.float32)
    for i in range(L):
        pos[0, 3 * i + 0] = torch.tensor([i * 0.59, 0.0, 0.0])      # P
        pos[0, 3 * i + 1] = torch.tensor([i * 0.59, 0.0, 0.3])      # C4'
        pos[0, 3 * i + 2] = torch.tensor([i * 0.59, 0.0, 0.6])      # N
    # the pair's N beads at exactly the guide's target, so the guide starts neutral
    pos[0, 3 * PAIR[1] + 2] = torch.tensor([0.0, 1.0, 0.6])
    if p_beads is not None:
        pos[0, 3 * PAIR[0] + 0] = torch.tensor(p_beads, dtype=torch.float32)
        pos[0, 3 * PAIR[1] + 0] = torch.tensor(p_beads, dtype=torch.float32)
    if n_beads is not None:
        pos[0, 3 * PAIR[0] + 2] = torch.tensor([0.0, 0.0, 0.6])
        pos[0, 3 * PAIR[1] + 2] = torch.tensor(n_beads, dtype=torch.float32)
    return pos


def _guide_energy(pos):
    """The pair guide's own contribution: everything else cancels in the difference."""
    pairs = torch.tensor([[PAIR[0], PAIR[1]]], dtype=torch.long)
    pair_w = torch.tensor([1.0], dtype=torch.float32)
    old = C.K_PAIR_GUIDE
    try:
        e_on, _ = C.cg_energy_forces(pos, pairs, pair_w, lam=1.0)
        C.K_PAIR_GUIDE = 0.0
        e_off, _ = C.cg_energy_forces(pos, pairs, pair_w, lam=1.0)
    finally:
        C.K_PAIR_GUIDE = old
    return float(e_on.sum() - e_off.sum())


def test_the_guide_reads_the_n_beads_not_the_p_beads():
    base = _state()
    # perturb ONLY the P beads of the pair: 3 nm apart instead of one backbone step
    p_moved = _state()
    p_moved[0, 3 * PAIR[0] + 0] = torch.tensor([0.0, 0.0, 0.0])
    p_moved[0, 3 * PAIR[1] + 0] = torch.tensor([0.0, 3.0, 0.0])
    # perturb ONLY the N beads of the pair: 4 nm apart instead of at the target
    n_moved = _state(n_beads=(0.0, 4.0, 0.6))

    g_base = _guide_energy(base)
    g_p = _guide_energy(p_moved)
    g_n = _guide_energy(n_moved)

    assert g_n > g_base + 1.0, (
        f"moving the paired N beads apart must be charged by the guide; "
        f"base={g_base:.3f} n_moved={g_n:.3f}")
    # relative, not absolute: the other terms still depend on the P beads, and float32
    # accumulation of a ~15-term sum differs in its last bits when they move.
    tol = 1e-3 * max(abs(g_base), 1.0)
    assert abs(g_p - g_base) < tol, (
        f"the pair guide must not read the P beads. Moving ONLY the paired P beads changed "
        f"its energy by {g_p - g_base:+.3f} kJ/mol, more than the {tol:.4f} float noise "
        f"(base {g_base:.3f}, p_moved {g_p:.3f}). torch_cgsim.py:1401 used to compute this "
        f"term from P(pi)/P(pj) against PAIR_NN, which is documented as the N-bead target -- "
        f"see this file's docstring for what that did on 2OIU.")


def test_the_charge_at_the_target_is_k_ln2_not_zero():
    """`_sigmoid_f` is `k*softplus((r-r0)/w)`, and softplus(0) = ln 2.

    A pair sitting exactly at PAIR_NN is therefore charged K*ln2 ~ 14.4 kJ/mol, not 0. That is
    a property of the form, not a defect -- this test exists so that a later edit cannot quietly
    change the floor, and so that the floor is not mistaken for a residual bug (it is what an
    earlier version of this test wrongly asserted to be zero).
    """
    pos = _state()
    dist = float(torch.linalg.norm(pos[0, 3 * PAIR[0] + 2] - pos[0, 3 * PAIR[1] + 2]))
    assert abs(dist - float(C.PAIR_NN)) < 0.01, f"precondition: the pair sits at {dist:.4f} nm"
    expected = float(C.K_PAIR_GUIDE) * float(np.log(2.0))
    got = _guide_energy(pos)
    assert abs(got - expected) < 0.05, (
        f"at exactly r0 the term should charge K*ln2 = {expected:.3f} kJ/mol, got {got:.3f}")


def test_the_guide_pulls_a_distant_pair_in():
    """The sign, so a future edit cannot flip the term back to its old behaviour.

    `_sigmoid_f`'s docstring records that x used to be (r0 - dist)/width, which "pulled
    hardest when the pair was already too close and did nothing at long range".
    """
    pos = _state(n_beads=(0.0, 3.0, 0.6))
    assert _guide_energy(pos) > 1.0, "a pair beyond the target must be charged"


def test_paired_phosphates_are_not_pulled_to_the_n_target():
    """The end-to-end statement: native geometry must not be charged for its P-P distance.

    In A-form RNA the P-P distance across a Watson-Crick pair is ~18 A while the N-N distance
    is ~9.9 A (measured on 2OIU). The guide's job is the second number, so a pair with N-N at
    the target must cost the same whether its P-P is native (1.8 nm) or collapsed (0.59 nm).
    """
    native = _state()
    native[0, 3 * PAIR[0] + 0] = torch.tensor([-0.9, 0.0, 0.0])
    native[0, 3 * PAIR[1] + 0] = torch.tensor([0.9, 0.0, 0.0])
    native[0, 3 * PAIR[0] + 2] = torch.tensor([0.0, 0.0, 0.6])
    native[0, 3 * PAIR[1] + 2] = torch.tensor([0.0, 1.0, 0.6])
    p_dist = float(torch.linalg.norm(native[0, 3 * PAIR[0] + 0] - native[0, 3 * PAIR[1] + 0]))
    assert abs(p_dist - 1.8) < 0.05, "precondition: P-P at 1.8 nm"

    floor = float(C.K_PAIR_GUIDE) * float(np.log(2.0))
    got = _guide_energy(native)
    assert got < floor + 0.05, (
        f"a pair with N-N at the target and P-P at native A-form geometry was charged "
        f"{got:.3f} kJ/mol against a floor of {floor:.3f}. On the shipped code this read "
        f"83.6 kJ/mol because the term was measuring P-P against the N-N target.")
