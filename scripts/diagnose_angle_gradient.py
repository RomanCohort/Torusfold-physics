"""Is _angle_f's gradient wrong, and does the textbook form fix it?

gradcheck_per_term.py finds the angle term's force disagreeing with its own energy
gradient by 1.927 relative -- the force is wrong in sign, magnitude or both -- while ten of
the thirteen terms agree to 1e-6 or better. This derives the gradient independently and
checks the derivation the same way, so the diagnosis does not rest on reading the code.

For v1 = p0 - p1, v2 = p2 - p1, cos = (v1.v2)/(|v1||v2|):

    dcos/dx0 = v2/(n1 n2) - cos * v1/n1^2
    dcos/dx2 = v1/(n1 n2) - cos * v2/n2^2
    dcos/dx1 = -(dcos/dx0 + dcos/dx2)

Run: python scripts/diagnose_angle_gradient.py
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
base = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)


def angle_energy(pos):
    return FT.term_energies_forces(
        pos, ij, pw, cell_list=C.GPUCellList(cell_size=1.5))[1]["angle P-P-P"]


def textbook_angle_force(pos):
    """F = -dE/dx for E = 0.5*k*(cos - c0)^2, with the gradient derived above."""
    P = lambda i: 3 * i
    Lc = pos.shape[1] // 3
    a = torch.arange(Lc - 2)
    p0, p1, p2 = pos[:, P(a)], pos[:, P(a + 1)], pos[:, P(a + 2)]
    v1, v2 = p0 - p1, p2 - p1
    n1 = torch.linalg.norm(v1, dim=-1, keepdim=True).clamp(min=1e-12)
    n2 = torch.linalg.norm(v2, dim=-1, keepdim=True).clamp(min=1e-12)
    cos = (v1 * v2).sum(-1, keepdim=True) / (n1 * n2)
    dE = C.K_ANGLE * (cos - np.cos(C.ANGLE_PPP))            # dE/dcos
    d0 = v2 / (n1 * n2) - cos * v1 / (n1 ** 2)
    d2 = v1 / (n1 * n2) - cos * v2 / (n2 ** 2)
    d1 = -(d0 + d2)
    F = torch.zeros_like(pos)
    F[:, P(a)] -= (dE * d0).squeeze(-1)
    F[:, P(a + 1)] -= (dE * d1).squeeze(-1)
    F[:, P(a + 2)] -= (dE * d2).squeeze(-1)
    return F


lib_F = FT.term_energies_forces(
    base, ij, pw, cell_list=C.GPUCellList(cell_size=1.5))[0]["angle P-P-P"]
mine_F = textbook_angle_force(base).reshape(3 * L, 3).numpy()

h = 1e-5
n_pt = 3 * L
probe = [(i, d) for i in range(0, n_pt, max(1, n_pt // 14)) for d in (0, 1, 2)][:36]
probe = [(i, d) for (i, d) in probe if i < n_pt]

res = {"library _angle_f": lib_F, "textbook form": mine_F}
print(f"{'gradient':18s} {'worst rel err':>14s} {'worst abs err':>14s}")
print("-" * 50)
for name, F in res.items():
    wr, wa = 0.0, 0.0
    for (i, d) in probe:
        hi = base.clone(); hi[0, i, d] += h
        lo = base.clone(); lo[0, i, d] -= h
        fd = -(float(angle_energy(hi)) - float(angle_energy(lo))) / (2 * h)
        an = float(F[i, d])
        wr = max(wr, abs(an - fd) / max(abs(an), abs(fd), 1e-6))
        wa = max(wa, abs(an - fd))
    print(f"{name:18s} {wr:14.3e} {wa:14.3e}")
print()
d = np.abs(lib_F - mine_F)
print(f"library minus textbook: max |difference| {float(d.max()):.2f} kJ/mol/nm, "
      f"max |library| {float(np.abs(lib_F).max()):.2f}")
print(f"correlation of the two force fields: "
      f"{float(np.corrcoef(lib_F.reshape(-1), mine_F.reshape(-1))[0, 1]):+.4f}")
