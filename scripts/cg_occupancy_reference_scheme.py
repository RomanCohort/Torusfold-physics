"""Occupancy of the cgRNASP cell grid, counted exactly as cgRNASP.c does.

Scheme taken from the reference implementation shipped in Tan-group/cgRNASP:
  k1=0, k2=1, k3=2, k4=4          separation boundaries (residue number difference)
  Rc1=5.0, Rc2=9.0, Rc3=13.0, Rc4=24.0
  intervals1=17, intervals2=30, intervals3=43, intervals4=80    (bin = 0.3 A)
  12 bead types: {A,U,C,G} x {P, C4', N9 or N1}
  class by |dresnum|: 1 / 2 / 3-4 / >=5 (different chain also counts as long range)
  pairs within one residue (same chain AND same residue number) are excluded

Counted over the 191 native structures in rsRNASP/Training_set, i.e. the very set the
published potentials were derived from.
"""
import sys, collections
from pathlib import Path

import numpy as np

DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")
K = (0, 1, 2, 4)
INTERVALS = (17, 30, 43, 80)
TYPES = [("A", "P"), ("A", "C4'"), ("A", "N9"),
         ("U", "P"), ("U", "C4'"), ("U", "N1"),
         ("C", "P"), ("C", "C4'"), ("C", "N1"),
         ("G", "P"), ("G", "C4'"), ("G", "N9")]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}


def parse(path):
    """Return (coords[12] as arrays, residx[12], chain[12]) for one PDB."""
    rec = collections.defaultdict(list)
    resmap = {}
    order = []
    for line in open(path, errors="replace"):
        if not line.startswith("ATOM"):
            continue
        resn = line[17:20].strip()
        atom = line[12:16].strip()
        key = (line[21], line[22:27].strip())
        if key not in resmap:
            resmap[key] = len(order)
            order.append(key)
        t = (resn, atom)
        if t in TYPE_INDEX:
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            rec[TYPE_INDEX[t]].append((resmap[key], line[21], xyz))
    coords, ridx, ch = [], [], []
    for i in range(12):
        v = rec.get(i, [])
        coords.append(np.array([x[2] for x in v], float) if v else np.zeros((0, 3)))
        ridx.append(np.array([x[0] for x in v], int) if v else np.zeros(0, int))
        ch.append(np.array([x[1] for x in v]) if v else np.zeros(0, dtype="U1"))
    return coords, ridx, ch, len(order)


def sep_class(dsep, same_chain):
    if not same_chain or dsep > K[3]:
        return 3
    if dsep > K[0] and dsep <= K[1]:
        return 0
    if dsep > K[1] and dsep <= K[2]:
        return 1
    if dsep > K[2] and dsep <= K[3]:
        return 2
    return None


cells = collections.Counter()
n_struct = 0
n_res_total = 0
files = sorted(DATA.glob("*.pdb"))
for f in files:
    try:
        coords, ridx, ch, nres = parse(f)
    except Exception:
        continue
    if nres == 0:
        continue
    n_struct += 1
    n_res_total += nres
    for i in range(12):
        if len(ridx[i]) == 0:
            continue
        for j in range(12):
            if len(ridx[j]) == 0:
                continue
            d = np.linalg.norm(coords[i][:, None, :] - coords[j][None, :, :], axis=-1)
            ds = np.abs(ridx[i][:, None] - ridx[j][None, :])
            sc = ch[i][:, None] == ch[j][None, :]
            same_res = (ds == 0) & sc
            for c, iv in enumerate(INTERVALS):
                cand = np.zeros(d.shape, bool)
                if c == 3:
                    cand |= (~sc) | (ds > K[3])
                elif c == 0:
                    cand |= sc & (ds > K[0]) & (ds <= K[1])
                elif c == 1:
                    cand |= sc & (ds > K[1]) & (ds <= K[2])
                else:
                    cand |= sc & (ds > K[2]) & (ds <= K[3])
                cand &= ~same_res
                if not cand.any():
                    continue
                bins = (d[cand] / 0.3).astype(int)
                ok = bins < iv
                for b in bins[ok]:
                    cells[(i, j, c, int(b))] += 1

total_cells = 144 * sum(INTERVALS)
nonempty = len(cells)
grand = sum(cells.values())
print(f"structures parsed      : {n_struct} / {len(files)}")
print(f"residues in total      : {n_res_total}")
print(f"bead-pair counts placed: {grand:,}")
print()
print(f"cell grid (12x12 x {sum(INTERVALS)} bins x 4 classes) : {total_cells:,}")
print(f"non-empty cells                             : {nonempty:,} ({nonempty / total_cells:.1%})")
print()
print(f"{'class':>6s} {'sep':>6s} {'bins':>5s} {'cells':>7s} {'used':>7s} {'occup':>7s} {'median':>7s}")
for c, (lo, hi) in enumerate([("1", "1"), ("2", "2"), ("3", "4"), (">=5", ">=5")]):
    nc = 144 * INTERVALS[c]
    vals = [cells.get((i, j, c, b), 0) for i in range(12) for j in range(12)
            for b in range(INTERVALS[c])]
    used = sum(1 for v in vals if v > 0)
    print(f"{c + 1:>6d} {lo + '-' + hi:>6s} {INTERVALS[c]:5d} {nc:7d} {used:7d} "
          f"{used / nc:7.1%} {float(np.median(vals)):7.0f}")
print()
allv = np.array([v for v in cells.values()])
print(f"counts per non-empty cell: median {np.median(allv):.0f}, mean {allv.mean():.1f}, "
      f"max {allv.max()}, cells with <10 counts {int((allv < 10).sum())} "
      f"({(allv < 10).mean():.1%} of non-empty)")
