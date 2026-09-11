"""Two different sets are called "non-bonded", and only one of them is the field's.

  set A, residue index gap >= 3.  This is the set scripts/calibrate_excluded_volume.py used to
      place CLASH_SIGMA at the P-P minimum 0.3975 nm. Its own closest pair is 0.3333 nm.
  set B, bead index gap >= 3.  This is the set the excluded volume actually acts on: both call
      paths drop |i-j| <= 2, and |i-j| is a BEAD index difference. Set B contains set A plus
      everything between beads 3i+1/3i+2 and beads 3(i+1)+0/3(i+1)+1 -- the pairs that cross one
      backbone link. scripts/measure_pair_clash_bsj_constants.py measured its closest at 0.3092 nm
      over 2022024 pairs.

A single sigma of 0.3975 cannot be right for set B without pressing on native contacts, because
set B reaches 0.3092. Sizing that is the point of this script: the earlier 0.0034 percent figure
was computed over set A, which is the wrong set for that question.

Run: python scripts/measure_type_pair_excluded_volume.py
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B   # noqa: E402

KBT = 2.494
SHIPPED_SIGMA = 0.3975
ATOM = ("P", "C4'", "N9/N1")

sets = {"A: residue gap >= 3": {}, "B: bead gap >= 3": {}}
nchain = 0
for f in sorted(B.DATA.glob("*.pdb")):
    for rec in B._chain_residues(str(f), with_names=True):
        beads = rec[0]
        nchain += 1
        L = beads.shape[0]
        if L < 4:
            continue
        flat = beads.reshape(L * 3, 3)
        types = np.arange(L * 3) % 3
        d = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
        i, j = np.triu_indices(L * 3, k=1)
        for label, keep in (
                ("A: residue gap >= 3", np.abs(i // 3 - j // 3) >= 3),
                ("B: bead gap >= 3", (j - i) >= 3)):
            ii, jj = i[keep], j[keep]
            ti, tj = types[ii], types[jj]
            for a in range(3):
                for b in range(a, 3):
                    sel = ((ti == a) & (tj == b)) | ((ti == b) & (tj == a))
                    if sel.any():
                        sets[label].setdefault((a, b), []).append(d[ii[sel], jj[sel]])

print(f"chains {nchain}")
for label in sets:
    print()
    print(f"=== {label} ===")
    print(f"{'type pair':14s} {'N':>8s} {'min':>7s} {'p0.1':>7s} {'< 0.3975':>9s} {'share':>9s}")
    tot = 0
    inside = 0
    for ab in sorted(sets[label]):
        v = np.concatenate(sets[label][ab])
        n_in = int((v < SHIPPED_SIGMA).sum())
        tot += len(v)
        inside += n_in
        print(f"{ATOM[ab[0]]+'-'+ATOM[ab[1]]:14s} {len(v):8d} {v.min():7.4f} "
              f"{np.percentile(v,0.1):7.4f} {n_in:9d} {100.0*n_in/len(v):8.5f}%")
    print(f"{'TOTAL':14s} {tot:8d} {'':7s} {'':7s} {inside:9d} {100.0*inside/tot:8.5f}%")
print()
print("Set B is the one that matters. Every pair counted in its '< 0.3975' column is a distance the")
print("database contains for that bead-type pair and that the shipped wall nevertheless pushes")
print("outward. The per-type minima are what a per-type sigma would use instead.")
