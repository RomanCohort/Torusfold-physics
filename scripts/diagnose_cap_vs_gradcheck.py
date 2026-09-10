"""Does the force cap make F = -dE/dx impossible, by construction?

After the angle, dihedral and BSJ-contact gradients were corrected, a per-term
finite-difference check passes for all thirteen terms of cg_energy_forces to 5.6e-07 or
better. The path-level check in tests/test_force_gradcheck.py still fails at max_rel
7.1708. Something outside the terms must be breaking the identity.

The candidate is the cap at the end of cg_energy_forces:

    total_F = total_F * torch.clamp(200.0 / |total_F|, max=1.0)

That rescales the total force vector whenever its magnitude exceeds 200 kJ/mol/nm. A
rescaled vector is not the gradient of any energy, so wherever the cap is active the
returned force cannot equal -dE/dx -- not because a term is wrong, but because the cap is
a constraint applied to the output rather than a term in the energy. With the corrected
dihedral gradient this should be the common case rather than the exception.

This checks the correlation: for a range of geometries, how much of the force is capped
against how far the path-level check misses.

Run: python scripts/diagnose_cap_vs_gradcheck.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C
import test_force_gradcheck as G

CAP = 200.0


def capped_fraction(pos, pairs, pair_w):
    _, F = C.cg_energy_forces(pos, pairs, pair_w)
    m = torch.linalg.norm(F, dim=-1)
    return float((m > CAP - 1e-3).float().mean())


def path_max_rel(pos, pairs, pair_w):
    a, _ = G._fd_max_errors(C.cg_energy_forces, pos, pairs, pair_w)
    return a


print("the test's own random system, then progressively stretched copies of it")
print(f"{'case':30s} {'capped beads':>13s} {'path max_rel':>13s}")
print("-" * 60)
pos0, pairs, pair_w = G._make_system()
cases = [("test system as built", pos0.clone())]
for s in (1.5, 3.0, 6.0, 12.0):
    cases.append((f"test system stretched x{s}", pos0.clone() * s))
for name, p in cases:
    try:
        print(f"{name:30s} {capped_fraction(p, pairs, pair_w):13.3f} "
              f"{path_max_rel(p, pairs, pair_w):13.4f}")
    except Exception as exc:
        print(f"{name:30s}   {type(exc).__name__}: {exc}")
print()

structs = B.load_structures(limit=6)
print("real structures, same two numbers")
print(f"{'structure':30s} {'capped beads':>13s} {'path max_rel':>13s}")
print("-" * 60)
for s in structs[:6]:
    L = len(s["pos"])
    p = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    if len(ij) == 0:
        ij = torch.zeros((0, 2), dtype=torch.long)
    w = torch.ones(len(ij), dtype=torch.float64)
    print(f"{s['name'] + ' (L=' + str(L) + ')':30s} {capped_fraction(p, ij, w):13.3f} "
          f"{path_max_rel(p, ij, w):13.4f}")
print()
print("If max_rel is near zero exactly when no bead is capped, the cap is the cause and the")
print("test cannot be satisfied while the cap fires. That is a design question -- a constraint")
print("on the output vector is not a term in the energy -- rather than a term-level bug.")
