"""Can a CG-level statistical potential be counted from what we have on hand?

Beads follow cgRNASP's definition: P, C4', and N9 (purine) or N1 (pyrimidine).
Distances are collected for every residue pair, split by bead-pair channel and by
residue separation, then binned. The question is occupancy: how many (channel,
separation, distance-bin) cells actually receive counts.
"""
import sys, collections
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz

BINS = np.arange(0.0, 30.5, 1.0)          # 0..30 A, 1 A bins
NB = len(BINS) - 1
BEADS = ("P", "S")                        # backbone phosphate, C4'


def bead_set(residue, base):
    out = {"P": residue.get("P"), "S": residue.get("C4'")}
    out["B"] = residue.get("N9" if base in "AG" else "N1")
    return out


def separation_class(d):
    if d <= 3:
        return "short(1-3)"
    if d <= 10:
        return "medium(4-10)"
    return "long(>10)"


def collect(structures):
    """structures: list of (order, res, base_of)."""
    cells = collections.Counter()
    by_channel = collections.Counter()
    n_pairs = 0
    for order, res, base_of in structures:
        L = len(order)
        bpos = [bead_set(res[k], base_of[i]) for i, k in enumerate(order)]
        for i in range(L):
            for j in range(i + 1, L):
                sep = j - i
                cls = separation_class(sep)
                bi = bpos[i]
                bj = bpos[j]
                for ni, vi in bi.items():
                    if vi is None:
                        continue
                    for nj, vj in bj.items():
                        if vj is None:
                            continue
                        ti = base_of[i]
                        tj = base_of[j]
                        ta = ni if ni != "B" else ti
                        tb = nj if nj != "B" else tj
                        key = tuple(sorted((ta, tb)))
                        d = float(np.linalg.norm(vi - vj))
                        if d >= BINS[-1]:
                            continue
                        k = int(np.searchsorted(BINS, d, side="right") - 1)
                        cells[(key, cls, k)] += 1
                        by_channel[key] += 1
                        n_pairs += 1
    return cells, by_channel, n_pairs


order, res, base_of, ps, partner, meta = truth_1ehz.load()
structures = [(order, res, base_of)]
cells, by_channel, n = collect(structures)
print(f"input: {meta['n_residues']} residues, chains with data: 1")
print(f"bead-pair distances collected: {n}")
print()
print(f"{'channel':10s} {'count':>8s}")
for k in sorted(by_channel, key=lambda x: -by_channel[x]):
    print(f"{k[0] + '-' + k[1]:10s} {by_channel[k]:8d}")
print()

classes = sorted({c for (_, c, _) in cells})
channels = sorted({k for (k, _, _) in cells})
print(f"{'separation':14s} {'cells used':>11s} {'cells total':>12s} {'occupancy':>10s} {'median count':>13s}")
tot_used = tot_all = 0
for cls in classes:
    used = sum(1 for (k, c, b) in cells if c == cls)
    allc = len(channels) * NB
    med = float(np.median([cells.get((k, cls, b), 0) for k in channels for b in range(NB)]))
    tot_used += used
    tot_all += allc
    print(f"{cls:14s} {used:11d} {allc:12d} {used / allc:10.1%} {med:13.0f}")
print()
print(f"overall occupancy: {tot_used}/{tot_all} = {tot_used / tot_all:.1%}")
print()
print("Zero-count share, and what it would take to fill it:")
tot_cells = len(channels) * len(classes) * NB
zero = tot_cells - sum(1 for v in cells.values() if v > 0)
print(f"  distinct cells: {tot_cells}, non-empty: {tot_cells - zero}, empty: {zero} "
      f"({zero / tot_cells:.1%})")
for target in (10, 50, 100):
    per_struct = n / max(1, sum(1 for v in cells.values() if v > 0))
    print(f"  to average {target:3d} counts over the non-empty cells: "
          f"~{target / max(per_struct, 1e-9):,.0f} comparable structures")
