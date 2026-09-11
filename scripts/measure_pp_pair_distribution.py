"""The observable the excluded volume and the CPU's electrostatics should be judged against.

Two open decisions are currently being argued against 5.90 angstrom -- the P(i)-P(i+1) bond
target. That is a target, not a criterion: any knob can be turned until the mean lands there,
and turning a knob to hit a number is how this field acquired most of its constants in the first
place.

The database does contain the criterion, and it is a distribution rather than a mean: for beads
that are NOT bonded and NOT 1-3 or 1-4 neighbours, how close do P atoms actually get? That
distribution is what an excluded volume has to reproduce, and its lower edge is what sets the
repulsion's range.

Run: python scripts/measure_pp_pair_distribution.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B          # noqa: E402

EDGES = np.array([0.0, 0.28, 0.30, 0.31, 0.32, 0.34, 0.36, 0.40, 0.45, 0.50, 0.60,
                  0.80, 1.00, 1.50, 2.00, 5.00])

rows = []
for f in sorted((REPO.parent / "torusfold-cgdata" / "rsRNASP" / "Training_set").glob("*.pdb")):
    for beads, _pairs, _names in B._chain_residues(str(f), with_names=True):
        P = beads[:, 0, :]
        n = len(P)
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        i, j = np.triu_indices(n)
        gap = np.abs(i - j)
        keep = gap >= 3
        rows.append(d[i[keep], j[keep]])

v = np.concatenate(rows)
print(f"non-bonded P-P pairs (|i-j| >= 3) over the whole database: {v.size} pairs")
print(f"  191 files, chains from boltzmann_bonded._chain_residues")
print()
print(f"{'statistic':22s} {'nm':>9s}")
print("-" * 33)
for name, q in (("minimum", 0.0), ("0.001 percentile", 0.1), ("0.1 percentile", 0.1),
                ("1 percentile", 1.0), ("5 percentile", 5.0), ("median", 50.0),
                ("mean", None)):
    if q is None:
        print(f"{name:22s} {v.mean():9.4f}")
    elif q == 0.0:
        print(f"{name:22s} {v.min():9.4f}")
    else:
        print(f"{name:22s} {np.percentile(v, q):9.4f}")
print()
print("=== how many pairs are below each candidate distance ===")
print(f"{'below (nm)':>11s} {'pairs':>10s} {'share':>9s}")
print("-" * 33)
for c in (0.295, 0.300, 0.305, 0.309, 0.310, 0.320, 0.340, 0.360, 0.400):
    n = int((v < c).sum())
    print(f"{c:11.3f} {n:10d} {100 * n / v.size:8.4f}%")
print()
print("=== the shape of the lower edge ===")
print(f"{'bin (nm)':>16s} {'count':>10s} {'share':>9s}")
print("-" * 37)
h, _ = np.histogram(v, bins=EDGES)
for k in range(len(h)):
    print(f"{EDGES[k]:7.3f}-{EDGES[k+1]:<7.3f} {h[k]:10d} {100 * h[k] / v.size:8.4f}%")
print()
print("Reading it: the field's clash cutoff is 0.300 nm and its clash spring pushes only below")
print("that. The database's own lower edge is the number that cutoff should reproduce, and the")
print("counts just above it say how much room there is before an excluded volume starts changing")
print("the structures it is supposed to be fitting.")
