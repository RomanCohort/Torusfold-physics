"""The clash mask, tested on a bead pair that is actually in the neighbour set.

The first attempt used beads 0 and 1, which are two of the three beads of residue 0 and are
excluded by get_pair_info's seq_near filter (|bead_i - bead_j| <= 2). Nothing changing there
was the filter working, not a bug. This uses beads 0 and 9, three residues apart.

Run: python scripts/diagnose_clash_mask.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

L = 12
gen = torch.Generator().manual_seed(3)
pos = torch.cat([torch.tensor([[[1.0 * i, 0.0, 0.0],
                                [1.0 * i + 0.2, 0.3, 0.1],
                                [1.0 * i + 0.3, 0.5, 0.35]]], dtype=torch.float64)
                 for i in range(L)], dim=1)
pos = pos.repeat(2, 1, 1)
pos = pos + torch.randn(pos.shape, generator=gen, dtype=torch.float64) * 0.9
pos[:, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
pos[:, 9] = torch.tensor([6.0, 0.0, 0.0], dtype=torch.float64)
A, Bi = 0, 9
print(f"bead indices {A} and {Bi}: |difference| = {abs(A-Bi)} > 2, so they are in the set")
print(f"distance in replica 0: {float(torch.linalg.norm(pos[0,A]-pos[0,Bi])):.4f} nm")
print(f"distance in replica 1: {float(torch.linalg.norm(pos[1,A]-pos[1,Bi])):.4f} nm")
print()


def clash(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    pi, pj, _d, _r = cl.get_pair_info(p)
    has = any((int(a) == A and int(b) == Bi) or (int(a) == Bi and int(b) == A)
              for a, b in zip(pi, pj))
    e, _f = C._clash_f(p, cl, C.K_CLASH, C.CLASH_DIST)
    return e.detach().numpy(), has


print(f"CLASH_DIST = {C.CLASH_DIST} nm")
e, has = clash(pos)
print(f"baseline: energy {e}, pair present in the set: {has}")
print()

p1 = pos.clone()
p1[1, A] = p1[1, Bi] + torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
e, has = clash(p1)
print(f"replica 1 compacted to 0.05 nm, replica 0 left at "
      f"{float(torch.linalg.norm(p1[0,A]-p1[0,Bi])):.4f} nm")
print(f"  energy {e}   pair present: {has}")
print(f"  a clash of 0.05 nm between beads {A} and {Bi} in replica 1 contributes "
      f"{e[1]:.6f} kJ/mol")
print()

p0 = pos.clone()
p0[0, A] = p0[0, Bi] + torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
e, has = clash(p0)
print(f"replica 0 compacted to 0.05 nm for comparison")
print(f"  energy {e}   pair present: {has}")
print()
print("If the second line is 0.0 while the first is large, the mask decided from replica 0")
print("is suppressing the term for replica 1, which matters because the pipeline runs 64")
print("replicas and only one of them has to clash.")
