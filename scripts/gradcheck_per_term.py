"""Per-term gradcheck: which analytic force does not match its own energy gradient?

tests/test_force_gradcheck.py reports a single max_rel for a whole path, on a system built
from torch.rand. That cannot localise anything: the worst of six probe ratios is dominated
by whichever term happens to be largest at that probe, so zeroing bpp or the WC pair moves
the number not at all. attribute_gradcheck_to_term.py shows that directly.

This checks each term on its own, on a real structure rather than random noise. For every
term, the term's own energy is differenced on one coordinate and compared with the term's
own force as the library computes it. A term whose force is right passes regardless of how
large it is or how the other terms behave.

Terms 1-13 come from cg_force_terms, which calls the library's own helpers. GB/SA and
Manning are not included: they are produced by autograd inside cg_energy_forces, so their
force is -dE/dx by construction and cannot be the source of an inconsistency.

Run: python scripts/gradcheck_per_term.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import cg_force_terms as FT
import torusfold.scheme2.torch_cgsim as C

structs = B.load_structures(limit=4)
s = max(structs, key=lambda x: len(x["pos"]))
L = len(s["pos"])
ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float64)
print(f"{s['name']}, L = {L}, {len(ij)} WC pairs")
print()


def terms_at(pos_nm):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos_nm)
    return FT.term_energies_forces(pos_nm, ij, pw, cell_list=cl)


base = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
# term_energies_forces returns (terms, energies); an earlier version unpacked these the
# wrong way round and then indexed a float
terms_F, _ = terms_at(base)

# probe coordinates: a spread across the chain and all three bead slots
n_pt = 3 * L
probe = [(i, d) for i in range(0, n_pt, max(1, n_pt // 12)) for d in (0, 1, 2)][:36]
probe = [(i, d) for (i, d) in probe if i < n_pt]

h = 1e-5
names = list(terms_F.keys())
worst_overall = {}
for name in names:
    worst_rel, worst_abs = 0.0, 0.0
    for (i, d) in probe:
        hi = base.clone(); hi[0, i, d] += h
        lo = base.clone(); lo[0, i, d] -= h
        e_hi = terms_at(hi)[1][name]
        e_lo = terms_at(lo)[1][name]
        fd = -(e_hi - e_lo) / (2 * h)
        an = float(terms_F[name][i, d])
        abs_err = abs(an - fd)
        denom = max(abs(an), abs(float(fd)), 1e-6)
        worst_rel = max(worst_rel, abs_err / denom)
        worst_abs = max(worst_abs, abs_err)
    worst_overall[name] = worst_rel
print()
print("(the table body is printed below, sorted by the worst relative error)")
print()
rows = sorted(names, key=lambda n: -worst_overall[n])
print(f"{'term':22s} {'max |F|':>10s} {'worst rel err':>14s}")
print("-" * 50)
for name in rows:
    print(f"{name:22s} {float(np.abs(terms_F[name]).max()):10.2f} "
          f"{worst_overall[name]:14.3e}")
print("-" * 50)
print()
print("force and energy are in kJ/mol/nm. A term with a correct analytic gradient shows a")
print("relative error at the level of the difference step; a term whose force is an")
print("approximation shows it as a fraction.")
