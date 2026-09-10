"""Verify that the CG beads are now taken from real geometry.

Compares the previous fabrication (C4' and N as offsets along the backbone direction)
with real_cg_beads (1EHZ template reconstruction) on the same P trace.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
from torusfold.scheme2.aform_from_template import real_cg_beads

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
pairs = [(a, b) for a, b in partner.items() if a < b]

beads = real_cg_beads(ps, seq, pairs=pairs)
print(f"real_cg_beads -> {beads.shape}  (L={L})")
print()

# old fabrication, same rule as torch_gpu_refine.py before the change
d = np.zeros_like(ps)
d[1:] = ps[1:] - ps[:-1]
d[0] = d[1]
bb = d / np.linalg.norm(d, axis=1, keepdims=True)
old_c4 = ps + bb * 3.4
old_n = ps + bb * (-1.5)

def stats(name, P, C4, NB):
    d_pc = np.linalg.norm(P - C4, axis=1)
    d_cn = np.linalg.norm(C4 - NB, axis=1)
    v = C4 - P
    ang = np.degrees(np.arccos(np.clip(
        np.sum(v * bb, axis=1) / (np.linalg.norm(v, axis=1) + 1e-12), -1, 1)))
    print(f"  {name:12s} |P-C4'| {d_pc.mean():6.3f}+/-{d_pc.std():5.3f}   "
          f"|C4'-N| {d_cn.mean():6.3f}+/-{d_cn.std():5.3f}   "
          f"P->C4' vs backbone {ang.mean():6.1f} deg")

print("intra-residue geometry")
stats("real", beads[:, 0], beads[:, 1], beads[:, 2])
stats("old offsets", ps, old_c4, old_n)
print()
print("  force field targets: |P-C4'| = 3.90 A, |C4'-N| = 3.35 A   (torch_cgsim.py:124,125)")
print("  1EHZ measured     : |P-C4'| = 3.887 +/- 0.081, |C4'-N| = 3.428 +/- 0.266 A")
print()
print(f"  base-typed bead: real beads carry N9 for purines and N1 for pyrimidines,")
print(f"  {sum(1 for b in base_of if b in 'AG')} purines / {sum(1 for b in base_of if b in 'CU')} pyrimidines")
print()
err = []
for i in range(L):
    r = res[order[i]]
    gly = "N9" if base_of[i] in "AG" else "N1"
    if all(a in r for a in ("P", "C4'", gly)):
        err.append(np.linalg.norm(beads[i, 2] - r[gly]))
print(f"  bead N against the crystal N9/N1, after the reconstruction places it: "
      f"mean {np.mean(err):.3f} +/- {np.std(err):.3f} A")
