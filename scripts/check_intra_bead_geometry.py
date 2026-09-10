"""Are the force field's intra-residue bead distances right?

k_INTRA restrains |P-C4'| to BOND_P_C4 = 3.90 A and |C4'-N| to BOND_C4_N = 3.35 A
(torch_cgsim.py:103,124,125), but the C4' and N beads are fabricated at
initialisation. Measure the real distances on 1EHZ.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_1ehz

order, res, base_of, ps, partner, meta = truth_1ehz.load()
names = {"ACGU".index("A"): "N9"}
pc, cn, pn = [], [], []
for i, k in enumerate(order):
    r = res[k]
    gly = "N9" if base_of[i] in "AG" else "N1"
    if not all(a in r for a in ("P", "C4'", gly)):
        continue
    P, C4, NB = r["P"], r["C4'"], r[gly]
    pc.append(float(np.linalg.norm(P - C4)))
    cn.append(float(np.linalg.norm(C4 - NB)))
    pn.append(float(np.linalg.norm(P - NB)))

f = lambda v: f"{np.mean(v):6.3f} +/- {np.std(v):5.3f}  (min {np.min(v):6.3f}, max {np.max(v):6.3f})"
print(f"n = {len(pc)} residues")
print()
print("quantity        measured (1EHZ)                          force-field target")
print(f"|P-C4'|         {f(pc)}     0.390 nm = 3.90 A")
print(f"|C4'-N|         {f(cn)}     0.335 nm = 3.35 A")
print(f"|P-N|           {f(pn)}     (not restrained)")
print()
print("spread relative to the target:")
for nm, v, t in (("|P-C4'|", pc, 3.90), ("|C4'-N|", cn, 3.35)):
    d = np.array(v) - t
    print(f"  {nm}: mean offset {d.mean():+.3f} A, sd {d.std():.3f} A, "
          f"fraction within 0.3 A of target {np.mean(np.abs(d) <= 0.3):.1%}")
