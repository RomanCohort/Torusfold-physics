"""Is the CG backbone bond target the thing pulling the relaxation away from the crystal?

cgRNASP is coarse-grained too, and its own CG model (simRNA/Vfold-style 3-bead) is what
BOND_P_NEXT was presumably tuned against. Measure |P[i+1]-P[i]| in the crystal, in the
reconstruction, and after a CG-only relaxation.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
from torusfold.scheme2.aform_from_template import real_cg_beads
from torusfold.scheme2.torch_cgsim import GPUCellList, cg_energy_forces, BOND_P_NEXT

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
pairs = [(a, b) for a, b in partner.items() if a < b]

ref = np.zeros((L, 3, 3))
for i in range(L):
    r = res[order[i]]
    gly = "N9" if base_of[i] in "AG" else "N1"
    ref[i, 0] = r["P"]; ref[i, 1] = r["C4'"]; ref[i, 2] = r[gly]
start = real_cg_beads(ps, seq, pairs=pairs)


def pp(beads):
    P = beads[:, 0]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return d.mean(), d.std()


print(f"BOND_P_NEXT = {BOND_P_NEXT:.3f} nm = {BOND_P_NEXT * 10:.2f} A")
print()
print(f"{'structure':34s} {'mean |P-P|':>12s} {'sd':>8s}   {'vs target':>11s}")
for name, b in (("1EHZ crystal", ref), ("our reconstruction", start)):
    m, sd = pp(b)
    print(f"{name:34s} {m:12.3f} {sd:8.3f}   {m - BOND_P_NEXT * 10:+11.3f}")


def force_cg(beads):
    x = torch.tensor(beads.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)
    pi = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pj = torch.tensor([b for _, b in pairs], dtype=torch.long)
    pw = torch.ones(len(pairs), dtype=torch.float64)
    cl = GPUCellList(cell_size=1.5)
    cl.build(x)
    _, f = cg_energy_forces(x, torch.stack([pi, pj], 1), pw, cell_list=cl)
    return f.reshape(3 * L, 3).numpy() * 10.0


def unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


beads = start.copy()
for _ in range(200):
    d = unit(force_cg(beads))
    nd = np.linalg.norm(d, axis=1, keepdims=True)
    beads = (beads.reshape(3 * L, 3) + 0.02 * d / np.maximum(nd, 1e-12)).reshape(L, 3, 3)
m, sd = pp(beads)
print(f"{'after 200 CG steps':34s} {m:12.3f} {sd:8.3f}   {m - BOND_P_NEXT * 10:+11.3f}")
print()

# how much of the CG force is the backbone term alone?
x = torch.tensor(start.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)
cl = GPUCellList(cell_size=1.5); cl.build(x)
empty = torch.zeros((0, 2), dtype=torch.long)
_, f_all = cg_energy_forces(x, torch.stack([torch.tensor([a for a, _ in pairs]),
                                            torch.tensor([b for _, b in pairs])], 1),
                            torch.ones(len(pairs), dtype=torch.float64), cell_list=cl)
_, f_nobond = None, None

import torusfold.scheme2.torch_cgsim as TCS
_bond_saved = TCS._bb_k if hasattr(TCS, "_bb_k") else None
print("CG force at the starting point, backbone-bond term zeroed for comparison")
x2 = x.clone()

# recompute with the backbone bond constant zeroed via a monkeypatch of the module global
orig = getattr(TCS, "K_BB", None)
import inspect
src = inspect.getsource(TCS.cg_energy_forces)
print("  (skipped a full decomposition: _bb_k is a local in cg_energy_forces, not a module global)")
