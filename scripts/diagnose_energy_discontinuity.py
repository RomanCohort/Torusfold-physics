"""Is the full-path energy discontinuous at the finite-difference scale?

localize_path_force_error.py isolates the discrepancy to the GB/SA + Manning block: the
thirteen analytic terms agree with their own FD gradient to a relative L2 of 0.0036, the full
path does not, and the difference has a relative L2 of 1.0007 with a maximum absolute error of
3750 kJ/mol/nm -- while the GB block's own force is at most 2.8 kJ/mol/nm per atom and its
energy is -7.29 kJ/mol. An energy that differs by 7.29 while its gradient differs by 3750 is
not a gradient problem.

The suspicion is that the energy is discontinuous. cg_energy_forces rebuilds the GB pair list
from the current coordinates on every call:

    p_coords = gb_pos[0]
    p_cell = torch.floor(p_coords / GB_CUTOFF).long()
    p_pairs = torch.nonzero(p_valid, as_tuple=False)

so a pair crossing the 1.0 nm cell filter enters or leaves and the energy jumps by a finite
amount, which central differences then read as an enormous derivative.

This checks it directly: for the coordinate with the largest discrepancy, print the energy at
pos, pos+h and pos-h against what the analytic force predicts.

Run: python scripts/diagnose_energy_discontinuity.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import torusfold.scheme2.torch_cgsim as C
import test_force_gradcheck as G

pos, pairs, pair_w = G._make_system()
L = pos.shape[1] // 3
h = 1e-5


def E(p, cap=None):
    return float(C.cg_energy_forces(p, pairs, pair_w, force_cap=cap)[0].reshape(-1)[0])


def F(p, cap=None):
    return C.cg_energy_forces(p, pairs, pair_w, force_cap=cap)[1]


F_an = F(pos, None)
worst = []
for i in range(pos.shape[1]):
    for d in range(3):
        hi = pos.clone(); hi[:, i, d] += h
        lo = pos.clone(); lo[:, i, d] -= h
        fd = -(E(hi, None) - E(lo, None)) / (2 * h)
        an = float(F_an[0, i, d])
        worst.append((abs(an - fd), i, d, an, fd))
worst.sort(reverse=True)

print(f"L = {L}, h = {h} nm, {3*L} coordinates probed")
print()
print(f"{'bead':>5s} {'axis':>5s} {'analytic F':>13s} {'FD':>13s} {'difference':>12s}")
print("-" * 54)
for diff, i, d, an, fd in worst[:6]:
    print(f"{i:5d} {d:5d} {an:13.3f} {fd:13.3f} {diff:12.3f}")
print()
i, d = worst[0][1], worst[0][2]
hi = pos.clone(); hi[:, i, d] += h
lo = pos.clone(); lo[:, i, d] -= h
e0, ep, em = E(pos, None), E(hi, None), E(lo, None)
print(f"worst coordinate: bead {i}, axis {d}")
print(f"  E(pos)     {e0:16.6f}")
print(f"  E(pos+h)   {ep:16.6f}   difference {ep-e0:+16.6f}")
print(f"  E(pos-h)   {em:16.6f}   difference {em-e0:+16.6f}")
print(f"  (E(+h) - E(-h)) / 2h  {-(ep-em)/(2*h):14.3f}")
print(f"  analytic force        {float(F_an[0,i,d]):14.3f}")
print()
# the same probe with the thirteen terms only
import cg_force_terms as FT


def E13(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    _F, e = FT.term_energies_forces(p, pairs, pair_w, cell_list=cl)
    return float(sum(e.values()))


print("the same coordinate on the thirteen-term energy, for contrast:")
print(f"  E13(pos)   {E13(pos):16.6f}")
print(f"  E13(+h)    {E13(hi):16.6f}   difference {E13(hi)-E13(pos):+16.6f}")
print(f"  E13(-h)    {E13(lo):16.6f}   difference {E13(lo)-E13(pos):+16.6f}")
print()
print("a jump of order 1e-2 kJ/mol between +h and -h at h = 1e-5 reads as a force of order")
print("1e3. If the full-path differences are that size while the thirteen-term ones scale with h,")
print("the full path is discontinuous and no finite-difference check of it can mean anything.")
