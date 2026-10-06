"""The CG stage's own paired-bead distance: is the pairing target itself wrong?

The chain of evidence so far:
  * aform_from_template already implements the orientation fix I was going to
    propose (pairs -> base faces its partner) and on the real 2OIU product it took
    key contacts 1/12 -> 3/12, stacking unchanged at 58.3%.
  * Run on THIS run's trace with all 82 pairs supplied, recall on the checkpoint
    pairs is still 0.0%, with 51-57 of 82 partners beyond 12 A -- a positional
    gap, which its own docstring says no twist can close.
  * From the other session's concurrent fix in torch_cgsim.py: PAIR_NN = 1.00 nm
    "pairing target on N beads; native 0.954 +/- 0.115 nm", and the guide term
    used to read the P beads instead, giving "run outputs, paired P-P 8.4 A".

So the question is what the CG output's PAIRED BEAD distance actually is. The
pipeline's own `pair_rate` counts P-P < 12 A and reports 0.55, but P-P across a
Watson-Crick pair is ~18 A in A-form, so that metric cannot be reading what the
pairing term steers. Measure the real distances instead.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
sys.path.insert(0, str(REPO / "src"))

RUN = REPO / "results" / "immuno_full"
KB = 0.008314462618


def dist_matrix(P: np.ndarray) -> np.ndarray:
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)


print("=" * 78)
print("the CG trace and its own pairing list")
print("=" * 78)
cg = RUN / "_final_cg_for_aa.pdb"
P = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
              for l in cg.read_text(encoding="utf-8", errors="replace").splitlines()
              if l.startswith(("ATOM", "HETATM"))])
L = len(P)
ck = json.loads((RUN / "_checkpoint.json").read_text(encoding="utf-8"))
pairs = [(int(p[0]), int(p[1])) for p in (ck.get("pairs") or [])]
print(f"  {cg.name}: {L} beads")
print(f"  checkpoint pairs: {len(pairs)} (0-based, matching the sequence index)")

D = dist_matrix(P)

# 0-based pairs index into the residue list directly
d_pair = np.array([D[i, j] for i, j in pairs])
# consecutive backbone beads
d_bb = np.array([D[i, i + 1] for i in range(L - 1)])
# a random baseline: everything that is not a listed pair and not adjacent
mask = np.ones((L, L), dtype=bool)
np.fill_diagonal(mask, False)
for i, j in pairs:
    mask[i, j] = mask[j, i] = False
for i in range(L - 1):
    mask[i, i + 1] = mask[i + 1, i] = False
d_other = D[mask]

print()
print("=" * 78)
print("measured bead-bead distances")
print("=" * 78)
print(f"  {'set':<34} {'n':>6} {'min':>7} {'median':>8} {'max':>8}")
for label, a in (("consecutive backbone bead", d_bb),
                 ("checkpoint PAIRS", d_pair),
                 ("non-pair, non-adjacent", d_other)):
    print(f"  {label:<34} {len(a):>6} {a.min():>7.2f} {np.median(a):>8.2f} {a.max():>8.2f}")

print()
print("  reference values from the code's own comments:")
print("    PAIR_NN = 1.00 nm = 10.00 A   the CG pairing target on N beads")
print("    native N-N 0.954 +/- 0.115 nm =  9.54 +/- 1.15 A")
print("    A-form paired P-P                ~18 A")
print("    A-form consecutive P-P          5.9-7.1 A")

# what the pipeline's pair_rate counts
print()
print("=" * 78)
print("what the pipeline's `pair_rate` is actually counting")
print("=" * 78)
for thr in (5.0, 8.0, 10.0, 12.0, 15.0, 18.0):
    frac = float((d_pair < thr).mean())
    print(f"  fraction of checkpoint pairs with P-P < {thr:>4.1f} A : {frac:6.1%}")
print()
print("  the run reported pair_rate = 0.549 at Level 1.5 and 0.561 at Level 2")
print("  (see _plots/07_validation.json). Compare with the row for 12.0 A.")

print()
print("=" * 78)
print("what the Metropolis criterion would see on these pairs")
print("=" * 78)
t_lo, t_hi, n_t = 300.0, 1000.0, 8
temps = np.geomspace(t_lo, t_hi, n_t)
beta0, beta1 = 1.0 / (KB * temps[0]), 1.0 / (KB * temps[1])
print(f"  ladder          : {np.round(temps, 1).tolist()}")
print(f"  |beta0-beta1|   : {abs(beta0-beta1):.6f}")
print(f"  the pairing excess on the listed pairs, in kJ/mol, is what a swap")
print(f"  would have to survive. Largest observed P-P deviation from the 10 A")
print(f"  target: {abs(d_pair - 10.0).max():.1f} A, a length not an energy --")
print(f"  converting needs the force constant, which is K_PAIR.")

print()
print("=" * 78)
print("CONCLUSION")
print("=" * 78)
n_far = int((d_pair > 12.0).sum())
print(f"""
  {n_far} of {len(pairs)} checkpoint pairs have their two beads further apart than
  12 A in the CG output -- and {int((d_pair > 18.0).sum())} are beyond 18 A, which is the
  distance a Watson-Crick pair SHOULD have. The pairing term is a harmonic on
  this distance, so those pairs are being pulled, and the rest of the structure
  is what resists.

  That is the wall: not the base orientation (already implemented, and it works
  on a product where the partners are close), but the fact that the CG structure
  does not hold the paired beads at pairing distance in the first place.
""")
