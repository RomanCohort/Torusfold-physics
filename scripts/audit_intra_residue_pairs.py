"""Every bead pair with a sequence gap of 1, 2 or 3: which one has a term, and what does the
database say?

This replaces an earlier version that HAND-LISTED six pair classes. A hand-written table cannot
find the class its author forgot, and it had: (N9/N1(i), C4'(i+1)) is a gap-2 pair with no bonded
term and no excluded volume, exactly like the three the list did contain. Enumerating the classes
from the bead indexing instead of from memory is the whole point of this file.

Bead indexing is flat: bead 3i+0 = P(i), 3i+1 = C4'(i), 3i+2 = N9/N1(i), bead 3i+3 = P(i+1). A
class is (residue offset, atom of the lower bead, atom of the higher bead).

The excluded volume skips |i-j| <= 2 in both the cell list and the O(N^2) mask, so for those pairs
the only possible restraint is a bonded term. The 3-bead nucleotide has two, so the classes below
with no bonded owner are the ones the field assigns nothing at all: they have no equilibrium and
no cost, and a minimiser or a thermal trajectory can drive them through zero.

Run: python scripts/audit_intra_residue_pairs.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B   # noqa: E402

KBT = 2.494            # kJ/mol at 300 K, the kBT the shipped k = kBT/sd^2 table is built on
ATOM = ("P", "C4'", "N9/N1")
GAPS = (1, 2, 3)

# The bonded terms the field HAS, as (residue offset, atom of the lower bead, atom of the higher).
# Derived from the index sets each path builds; nothing here is from memory.
BONDED = {
    (0, 0, 1): "K_INTRA_PC",     # P(i)   - C4'(i)
    (0, 1, 2): "K_INTRA_CN",     # C4'(i) - N9/N1(i)
    (0, 0, 2): "K_INTRA_PN",     # P(i)   - N9/N1(i)
    (1, 0, 0): "K_BB_BOND",      # P(i)   - P(i+1)
    (1, 1, 0): "K_LINK_CP",      # C4'(i) - P(i+1)
    (1, 2, 0): "K_LINK_NP",      # N9/N1(i) - P(i+1)
    (1, 2, 1): "K_LINK_NC",      # N9/N1(i) - C4'(i+1)
}

structs = B.load_structures()
print(f"chains {len(structs)}")

# accumulate every gap-1..3 pair by class, straight from the bead indexing
acc = {}
for s in structs:
    p = s["pos"]                      # (L, 3, 3)
    L = p.shape[0]
    nb = 3 * L
    for r in range(L):
        for a in range(3):
            b1 = 3 * r + a
            for gap in GAPS:
                b2 = b1 + gap
                if b2 >= nb:
                    continue
                r2, a2 = divmod(b2, 3)
                d = float(np.linalg.norm(p[r, a] - p[r2, a2]))
                acc.setdefault((r2 - r, a, a2), []).append(d)

print()
print(f"{'class':30s} {'gap':>4s} {'owner':>12s} {'wall':>5s} {'N':>6s} {'min':>7s} "
      f"{'p1':>7s} {'mean':>7s} {'max':>7s} {'sd':>7s} {'k':>9s} {'floor':>7s}")
print("-" * 118)
holes = []
for (dr, a1, a2), vals in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    v = np.array(vals)
    gap = 3 * dr + (a2 - a1)
    owner = BONDED.get((dr, a1, a2))
    wall = "yes" if gap >= 3 else "skip"
    lab = f"{ATOM[a1]}(i)-{ATOM[a2]}(i+{dr})"
    sd = v.std()
    k = KBT / sd ** 2
    print(f"{lab:30s} {gap:4d} {str(owner):>12s} {wall:>5s} {len(v):6d} {v.min():7.4f} "
          f"{np.percentile(v, 1):7.4f} {v.mean():7.4f} {v.max():7.4f} {sd:7.4f} "
          f"{k:9.1f} {np.sqrt(k * KBT):7.1f}")
    if owner is None and gap < 3:
        holes.append((lab, gap, v.min(), v.mean(), sd, k))

print()
if holes:
    print("NO TERM AT ALL (no bonded owner and the excluded volume skips the gap):")
    for lab, gap, mn, mean, sd, k in holes:
        print(f"    {lab:26s} gap {gap}   min {mn:.4f}  mean {mean:.4f}  sd {sd:.4f}"
              f"   k = kBT/sd^2 = {k:9.1f}   r0 = {mean:.4f}   floor {np.sqrt(k * KBT):.1f}")
else:
    print("no unguarded classes remain")
print()
print("Classes with gap >= 3 are covered by the excluded volume and need no bonded term.")
print("The closure link -- C4'(L-1)-P(0), N9/N1(L-1)-P(0), N9/N1(L-1)-C4'(0) -- is not in this")
print("table: no linear chain has it, so K_BSJ is what holds it, and the link constants above are")
print("deliberately not wrapped onto it.")
