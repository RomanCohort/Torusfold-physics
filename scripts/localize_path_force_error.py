"""Where does the path-level force stop matching the path-level energy gradient?

With the rewritten gradcheck the relative L2 error is 1.07 for cg_energy_forces with its cap
disabled, which is larger than the gradient itself and cannot be noise. Yet gradcheck_per_term
says all thirteen analytic terms agree with their own energies to 5.6e-07. So the discrepancy
is in something outside those thirteen, and cg_energy_forces has a second cap there:

    gb_f = gb_f * torch.clamp(50.0 / _gb_f_mag, max=1.0)

applied to the GB/SA gradient per atom. This separates the two contributions: it finite
differences the thirteen-term energy and the full energy separately, and compares each against
its own force.

Run: python scripts/localize_path_force_error.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import cg_force_terms as FT
import torusfold.scheme2.torch_cgsim as C
import test_force_gradcheck as G

pos, pairs, pair_w = G._make_system()
L = pos.shape[1] // 3
print(f"test chain, L = {L}, {len(pairs)} pairs, float64")
print()


def E_full(p):
    return float(C.cg_energy_forces(p, pairs, pair_w, force_cap=None)[0].reshape(-1)[0])


def E_terms(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    _F, e = FT.term_energies_forces(p, pairs, pair_w, cell_list=cl)
    return float(sum(e.values()))


def F_full(p):
    return C.cg_energy_forces(p, pairs, pair_w, force_cap=None)[1]


def F_terms(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    F, _e = FT.term_energies_forces(p, pairs, pair_w, cell_list=cl)
    return torch.tensor(sum(F.values()), dtype=torch.float64)


def fd(Efn, h=1e-5):
    out = torch.zeros_like(pos)
    for i in range(pos.shape[1]):
        for d in range(3):
            hi = pos.clone(); hi[:, i, d] += h
            lo = pos.clone(); lo[:, i, d] -= h
            out[:, i, d] = -(Efn(hi) - Efn(lo)) / (2.0 * h)
    return out


def report(name, F_an, F_fd):
    num = float(torch.linalg.norm(F_an - F_fd))
    den = max(float(torch.linalg.norm(F_fd)), 1e-12)
    m = float((F_an - F_fd).abs().max())
    print(f"{name:34s} rel L2 {num/den:9.4f}   max abs {m:10.2f}")


print(f"{'quantity':34s} {'relative L2':>9s}   {'max abs':>10s}")
print("-" * 62)
report("13 analytic terms", F_terms(pos), fd(E_terms))
report("full path, cap disabled", F_full(pos), fd(E_full))
print()
print("and the difference, which isolates GB/SA + Manning:")
Fg_an = F_full(pos) - F_terms(pos)
Fg_fd = fd(E_full) - fd(E_terms)
report("GB/SA + Manning (by difference)", Fg_an, Fg_fd)
print()
print(f"13-term energy   {E_terms(pos):14.4f}")
print(f"full energy      {E_full(pos):14.4f}")
print(f"GB/SA + Manning  {E_full(pos) - E_terms(pos):14.4f}")
print()
gm = torch.linalg.norm(Fg_an, dim=-1)
print(f"per-atom GB/SA + Manning force magnitude: median {float(gm.median()):.3f}, "
      f"max {float(gm.max()):.1f} kJ/mol/nm")
print(f"the inner clamp in cg_energy_forces is 50.0; atoms at it: "
      f"{int((gm > 49.999).sum())} of {3*L}")
