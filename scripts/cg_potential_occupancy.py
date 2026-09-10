"""Occupancy broken down by channel group.

The naive "counts per non-empty cell" average is dominated by the backbone channels
(P-P, P-S, S-S), which carry no sequence information. What a potential needs is occupancy
in the base-carrying channels, and in the short-range separation class.
"""
import sys, collections
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz

BINS = np.arange(0.0, 30.5, 1.0)
NB = len(BINS) - 1
BASES = set("ACGU")


def group(a, b):
    ab = {a, b}
    if ab <= {"P", "S"}:
        return "backbone-backbone"
    if ab & BASES and ab <= ({"P", "S"} | BASES) and not (a in BASES and b in BASES):
        return "backbone-base"
    return "base-base"


def sep_class(d):
    return "short(1-3)" if d <= 3 else ("medium(4-10)" if d <= 10 else "long(>10)")


order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
bead = []
for i, k in enumerate(order):
    r = res[k]
    bead.append({"P": r.get("P"), "S": r.get("C4'"),
                 "B": r.get("N9" if base_of[i] in "AG" else "N1")})

cells = collections.Counter()
channels = collections.Counter()
total = 0
for i in range(L):
    for j in range(i + 1, L):
        cls = sep_class(j - i)
        for ni, vi in bead[i].items():
            if vi is None:
                continue
            for nj, vj in bead[j].items():
                if vj is None:
                    continue
                ta = ni if ni != "B" else base_of[i]
                tb = nj if nj != "B" else base_of[j]
                g = group(ta, tb)
                channels[g] += 1
                d = float(np.linalg.norm(vi - vj))
                if d < BINS[-1]:
                    cells[(g, cls, int(np.searchsorted(BINS, d, side="right") - 1))] += 1
                    total += 1

print(f"one structure, {L} residues, {total} bead-pair distances")
print()
groups = ["backbone-backbone", "backbone-base", "base-base"]
nch = {"backbone-backbone": 3, "backbone-base": 8, "base-base": 10}
print(f"{'channel group':18s} {'dist':>7s} {'chan':>5s} {'cells':>6s} {'used':>6s} {'occup':>7s} "
      f"{'median':>7s} {'mean':>7s}")
for g in groups:
    ncell = nch[g] * 3 * NB
    used = sum(1 for (gg, c, b), v in cells.items() if gg == g and v > 0)
    vals = [cells.get((g, c, b), 0) for c in ("short(1-3)", "medium(4-10)", "long(>10)")
            for b in range(NB)]
    print(f"{g:18s} {channels[g]:7d} {nch[g]:5d} {ncell:6d} {used:6d} {used / ncell:7.1%} "
          f"{np.median(vals):7.0f} {np.mean(vals):7.2f}")
print()
print("the same, but only the short-range class (where cgRNASP's added resolution sits)")
print(f"{'channel group':18s} {'used':>6s} {'total':>6s} {'occup':>7s} {'median':>7s}")
for g in groups:
    ncell = nch[g] * NB
    vals = [cells.get((g, "short(1-3)", b), 0) for b in range(NB)]
    used = sum(1 for v in vals if v > 0)
    print(f"{g:18s} {used:6d} {ncell:6d} {used / ncell:7.1%} {np.median(vals):7.0f}")
print()
print("what one structure contributes to the base-base channel, per separation class")
for c in ("short(1-3)", "medium(4-10)", "long(>10)"):
    v = [cells.get(("base-base", c, b), 0) for b in range(NB)]
    print(f"  {c:14s} total {sum(v):5d} over {NB} bins   non-empty {sum(1 for x in v if x):2d}")
print()
print("So the useful channels and the useful separations are exactly the sparse ones.")
print("Counts alone are not the whole problem either: one tRNA gives one fold's worth of")
print("statistics, so more copies of similar structures would bias the reference state")
print("rather than fill the cells. cgRNASP used 191 non-redundant RNA structures")
print("(RNA 3D Hub non-redundant Release 3.102), with 0.3 A bins - three times finer")
print("than the 1 A used above, so the occupancy here is the generous case.")