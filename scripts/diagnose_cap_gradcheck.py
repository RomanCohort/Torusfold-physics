"""Is the force cap what breaks F = -dE/dx?

Every one of the thirteen terms passes its own finite-difference check to 5.6e-07 or better,
and the GB/SA and Manning blocks derive their force by autograd on their own energy. Yet
tests/test_force_gradcheck.py fails at the path level, with max_rel bouncing between 11.42,
7.17, 5.85, 3.09 and 6.51 as the field changed -- no monotone trend, which is what a fragile
statistic looks like rather than a term that is getting worse.

The remaining candidate is the cap. Rescaling the summed force vector is nonlinear and is not
a term in the energy, so wherever it fires the returned force cannot equal -dE/dx, whatever
the terms do. cg_energy_forces now takes force_cap=None to switch it off, which makes the
question answerable.

Run: python scripts/diagnose_cap_gradcheck.py
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

CAP = 200.0


def errs(pos, pairs, pair_w, cap):
    def fn(p, pr, pw):
        return C.cg_energy_forces(p, pr, pw, force_cap=cap)
    fn.__name__ = f"cg_energy_forces(force_cap={cap})"
    return G._fd_max_errors(fn, pos, pairs, pair_w)


pos0, pairs, pair_w = G._make_system()
print("the test's own random system")
print(f"{'force_cap':>12s} {'max_rel':>10s} {'max_abs':>10s} {'capped beads':>14s}")
print("-" * 50)
for cap in (200.0, None):
    a, r = errs(pos0, pairs, pair_w, cap)
    _, F = C.cg_energy_forces(pos0, pairs, pair_w, force_cap=cap)
    m = torch.linalg.norm(F, dim=-1)
    capped = float((np.abs(m.numpy() - CAP) < 1e-6).mean())
    tag = "  (cap disabled)" if cap is None else ""
    print(f"{str(cap):>12s} {r:10.4f} {a:10.4f} {capped:13.3f}{tag}")
print()
print("real structures, same comparison")
print(f"{'structure':14s} {'cap=200':>10s} {'cap=None':>10s} {'capped frac':>12s}")
print("-" * 50)
import boltzmann_bonded as B
for s in B.load_structures(limit=6):
    L = len(s["pos"])
    p = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    if len(ij) == 0:
        ij = torch.zeros((0, 2), dtype=torch.long)
    w = torch.ones(len(ij), dtype=torch.float64)
    try:
        a1, r1 = errs(p, ij, w, 200.0)
        a2, r2 = errs(p, ij, w, None)
        _, F = C.cg_energy_forces(p, ij, w, force_cap=200.0)
        frac = float((np.abs(torch.linalg.norm(F, dim=-1).numpy() - CAP) < 1e-6).mean())
        print(f"{s['name'] + ' (L=' + str(L) + ')':14s} {r1:10.4f} {r2:10.4f} {frac:12.3f}")
    except Exception as exc:
        print(f"{s['name']:14s} {type(exc).__name__}: {exc}")
