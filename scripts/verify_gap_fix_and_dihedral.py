"""Did the gap fix take, and what is the dihedral force scale on clean data?

Two things at once. First, whether boltzmann_bonded.load_structures now only emits runs of
consecutively numbered residues. Second, the dihedral force magnitude and where it comes
from, on that cleaned data.

The mechanism under test: the dihedral gradient carries a factor 1/|n0|^2 with
n0 = b0 x b1, so |n0| = |b0||b1| sin(theta) with theta the angle between consecutive bonds.
When three phosphorus atoms line up, sin(theta) -> 0 and the gradient diverges. That would
make the divergence geometric rather than a property of restraining cos instead of phi --
both forms share dphi/dx.

Run: python scripts/verify_gap_fix_and_dihedral.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
structs = B.load_structures(limit=N)
print(f"{len(structs)} chains after the gap fix, lengths "
      f"{min(len(s['pos']) for s in structs)}-{max(len(s['pos']) for s in structs)}")
print()

# After the fix every structure is a run of consecutive residues, so the P-P distance
# should sit in the native band with no cross-gap outliers.
allbb = []
for s in structs:
    P = s["pos"][:, 0, :]
    allbb.append(np.linalg.norm(P[1:] - P[:-1], axis=-1))
allbb = np.concatenate(allbb)
print(f"P-P step distances: n={len(allbb)}, mean {allbb.mean():.3f} nm, "
      f"sd {allbb.std():.3f}, min {allbb.min():.3f}, max {allbb.max():.3f}")
print(f"  steps over 0.90 nm (a gap would look like this): {int((allbb > 0.90).sum())}")
print()

rows = []
for s in structs:
    L = len(s["pos"])
    if L < 6:
        continue
    p = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
    P = torch.arange(L) * 3
    with torch.enable_grad():
        pr = p[:, P].detach().clone().requires_grad_(True)
        b0 = pr[:, 1:-2] - pr[:, :-3]
        b1 = pr[:, 2:-1] - pr[:, 1:-2]
        b2 = pr[:, 3:] - pr[:, 2:-1]
        n0 = torch.linalg.cross(b0, b1, dim=-1)
        n1 = torch.linalg.cross(b1, b2, dim=-1)
        cosd = ((n0 * n1).sum(-1) /
                (torch.linalg.norm(n0, dim=-1) * torch.linalg.norm(n1, dim=-1)))
        (0.5 * C.K_DIH * (cosd.clamp(-1 + 1e-6, 1 - 1e-6) - np.cos(C.DIH_PPPP)) ** 2).sum().backward()
        g = pr.grad[0]
    n0n = torch.linalg.norm(n0, dim=-1)[0].detach().numpy()
    for i in range(L - 3):
        idx = [i, i + 1, i + 2, i + 3]
        fmag = float(torch.linalg.norm(g[idx], dim=-1).max())
        rows.append((fmag, float(n0n[i])))

a = np.array(rows)
f, n0n = a[:, 0], a[:, 1]
print(f"{len(f)} dihedral windows")
print(f"|F| median {np.median(f):.1f}, p95 {np.percentile(f,95):.1f}, max {f.max():.1f} kJ/mol/nm")
print(f"|F| over the 200 cap: {int((f>200).sum())} ({(f>200).mean()*100:.1f} percent)")
print()
print("is the force explained by the size of |n0| = |b0||b1| sin(theta)?")
for lo, hi in [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]:
    m = (n0n >= lo) & (n0n < hi)
    if m.sum():
        print(f"  |n0| in [{lo:.2f},{hi:.2f}): {int(m.sum()):5d} windows, "
              f"|F| median {np.median(f[m]):10.1f}, max {f[m].max():12.1f}")
print()
print("the seven largest, with the |n0| that should drive them")
print(f"{'|F|':>12s} {'|n0|':>9s} {'1/|n0|^2':>11s}")
print("-" * 34)
for i in np.argsort(-f)[:7]:
    print(f"{f[i]:12.1f} {n0n[i]:9.4f} {1.0/max(n0n[i],1e-9)**2:11.0f}")
