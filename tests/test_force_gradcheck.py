"""Finite-difference gradcheck for the no-autograd explicit-force paths.

Verifies that the forces returned by cg_forces_explicit_batched and
cg_energy_forces match the (central finite-difference) gradient of their own
energy: F = -dE/dx, term-level bugs included.

Skipped automatically when torch / torch_cgsim is unavailable (the CI smoke
suite runs without torch).
"""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

tc = pytest.importorskip("torusfold.scheme2.torch_cgsim")


def _make_system(L=6, seed=7):
    """Small 3-bead system (P/C4'/N per residue) at clash-free distances."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    B, N = 1, 3 * L
    pos = (torch.rand(B, N, 3) * 1.2 + 0.3).to(torch.float32)  # nm
    pairs = torch.tensor([[0, 2], [1, 3], [2, 5], [0, 4]], dtype=torch.long)  # residue-index pairs
    pair_w = torch.ones(pairs.shape[0], dtype=torch.float32)  # (P,) 1-D, as expected by the force functions
    return pos, pairs, pair_w


def _fd_max_errors(force_fn, pos, pairs, pair_w, h=1e-3):
    """Central-difference forces on a few probe coordinates; returns (max_abs, max_rel)."""
    B, N, _ = pos.shape
    probe = [(0, 0), (0, 1), (0, 2), (1, 0), (N // 2, 1), (N - 1, 2)]
    fd = {}
    for (i, d) in probe:
        hi = pos.clone(); hi[:, i, d] += h
        lo = pos.clone(); lo[:, i, d] -= h
        fd[(i, d)] = -(force_fn(hi, pairs, pair_w)[0] - force_fn(lo, pairs, pair_w)[0]) / (2.0 * h)
    _, F = force_fn(pos, pairs, pair_w)
    max_abs, max_rel = 0.0, 0.0
    for (i, d), fnum in fd.items():
        a = float((F[:, i, d] - fnum).abs().max())
        denom = float(F[:, i, d].abs().max()) + 1e-6
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, a / denom)
    return max_abs, max_rel


@pytest.mark.parametrize("force_fn", [tc.cg_forces_explicit_batched, tc.cg_energy_forces])
def test_explicit_force_matches_own_energy_gradient(force_fn):
    pos, pairs, pair_w = _make_system()
    max_abs, max_rel = _fd_max_errors(force_fn, pos, pairs, pair_w)
    # Generous float32 tolerance: catches term-level bugs (wrong sign, missing
    # terms, wrong indices) without being fragile to sigmoid saturation.
    assert max_rel < 0.2, (
        f"{force_fn.__name__}: explicit force deviates from its own energy "
        f"gradient (max_rel={max_rel:.4f}, max_abs={max_abs:.5f} kJ/mol/nm)"
    )
