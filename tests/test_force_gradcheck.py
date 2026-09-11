"""Finite-difference gradcheck for the CG force paths.

Rewritten twice, and the history is the point.

The first version reported the max over six probes of |F_analytic - F_fd| divided by a
single force component plus 1e-6. That denominator can pass through zero, so the statistic
blew up on its own: diagnose_cap_gradcheck.py catches it returning exactly 1.0000 for two
structures, meaning the finite difference was zero. Its value across the arc was 11.42,
7.17, 5.85, 3.09, 6.51 -- no trend, because there was nothing to trend.

The second version used the relative L2 norm over every coordinate, which is sound, and it
failed at 1.07. That turned out not to be a wrong force. diagnose_energy_discontinuity.py
shows the full-path energy is DISCONTINUOUS in the coordinates: cg_energy_forces rebuilds its
GB/SA pair list from a hard 1.0 nm cell filter on every call, so a pair crossing it changes
the energy by 2.0 * exp(-1.0/0.304)/1.0 = 0.074 kJ/mol with no corresponding term in the
gradient. Finite differences of a step function return the step size divided by 2h.

So the full path cannot be checked this way at any h, and asserting on it was asserting
something impossible. What the suite checks now:

  * the thirteen analytic terms, through cg_force_terms, which do have continuous gradients
  * the force cap, which is a constraint on the output and gets its own test
  * the GB discontinuity itself, as a measured fact, so that fixing it is a visible event

Skipped automatically when torch / torch_cgsim is unavailable.
"""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

tc = pytest.importorskip("torusfold.scheme2.torch_cgsim")
torch = pytest.importorskip("torch")
ft = pytest.importorskip("cg_force_terms")


def _make_system(L=9):
    """A physical chain in float64: 0.59 nm P-P steps with sensible C4' and N offsets.

    The original helper used torch.rand(1, 18, 3) * 1.2 + 0.3 for L = 6, i.e. eighteen beads
    inside a 1.2 nm box, so every bead was inside the clash cutoff of every other.
    """
    pos = torch.zeros(1, 3 * L, 3, dtype=torch.float64)
    for i in range(L):
        base = torch.tensor([0.59 * i, 0.0, 0.0], dtype=torch.float64)
        pos[0, 3 * i] = base
        pos[0, 3 * i + 1] = base + torch.tensor([0.12, 0.37, 0.0], dtype=torch.float64)
        pos[0, 3 * i + 2] = pos[0, 3 * i + 1] + torch.tensor(
            [0.07, 0.30, 0.13], dtype=torch.float64)
    pairs = torch.tensor([[0, 4], [1, 6], [2, 5]], dtype=torch.long)
    pair_w = torch.ones(pairs.shape[0], dtype=torch.float64)
    return pos, pairs, pair_w


def _rel_l2(F_an, F_fd):
    num = float(torch.linalg.norm(F_an - F_fd))
    den = max(float(torch.linalg.norm(F_fd)), 1e-12)
    return num / den


def _terms_energy_forces(pos, pairs, pair_w):
    cl = tc.GPUCellList(cell_size=1.5)
    cl.build(pos)
    F, e = ft.term_energies_forces(pos, pairs, pair_w, cell_list=cl)
    return sum(e.values()), torch.tensor(sum(F.values()), dtype=torch.float64)


def test_analytic_terms_match_their_energy_gradient():
    """The thirteen terms have continuous gradients and can be checked properly."""
    pos, pairs, pair_w = _make_system()
    h = 1e-5
    fd = torch.zeros_like(pos)
    for i in range(pos.shape[1]):
        for d in range(3):
            hi = pos.clone(); hi[:, i, d] += h
            lo = pos.clone(); lo[:, i, d] -= h
            fd[:, i, d] = -(_terms_energy_forces(hi, pairs, pair_w)[0]
                            - _terms_energy_forces(lo, pairs, pair_w)[0]) / (2.0 * h)
    _e, F = _terms_energy_forces(pos, pairs, pair_w)
    rel = _rel_l2(F, fd)
    assert rel < 0.02, (
        f"the thirteen analytic terms deviate from their own energy gradient "
        f"(relative L2 {rel:.4f})")


def test_the_force_cap_actually_caps():
    """The cap is a constraint on the output vector, so it gets its own test rather than
    hiding inside a consistency check. A bead whose raw force exceeds 200 comes back at 200,
    and a bead under it is untouched."""
    pos, pairs, pair_w = _make_system()
    _, F_raw = tc.cg_energy_forces(pos, pairs, pair_w, force_cap=None)
    _, F_cap = tc.cg_energy_forces(pos, pairs, pair_w, force_cap=200.0)
    m_raw = torch.linalg.norm(F_raw, dim=-1)
    m_cap = torch.linalg.norm(F_cap, dim=-1)
    assert float(m_cap.max()) <= 200.0 + 1e-6
    over = m_raw > 200.0
    if over.any():
        assert torch.allclose(m_cap[over], torch.full_like(m_cap[over], 200.0), atol=1e-4)
    under = ~over
    if under.any():
        assert torch.allclose(F_cap[under], F_raw[under], atol=1e-9)


def test_gb_energy_is_discontinuous_which_is_why_it_cannot_be_gradchecked():
    """A measured defect, not an aspiration.

    cg_energy_forces rebuilds the GB/SA pair list from a hard 1.0 nm cell filter each call, so
    a pair entering or leaving changes the energy by a finite amount with no term in the
    gradient. Moving one bead by 1e-4 nm should make the forward and backward differences
    ASYMMETRIC; the asymmetry is the jump. If a switching function is ever added the
    asymmetry should collapse and this test should be replaced by a real gradcheck of the
    full path.
    """
    pos, pairs, pair_w = _make_system()
    h = 1e-4

    def E(p):
        return float(tc.cg_energy_forces(p, pairs, pair_w, force_cap=None,
                                         cell_list=None)[0].reshape(-1)[0])

    worst = 0.0
    for i in range(pos.shape[1]):
        hi = pos.clone(); hi[:, i, 0] += h
        lo = pos.clone(); lo[:, i, 0] -= h
        ep, e0, em = E(hi), E(pos), E(lo)
        worst = max(worst, abs((ep - e0) - (e0 - em)) / 2.0)
    assert worst > 1e-6, (
        "the GB energy is now smooth under a 1e-4 nm displacement; if a switching function "
        "was added, replace this test with a real finite-difference gradcheck of the full "
        "path")
