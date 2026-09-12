"""Is the block-to-block drift one common direction (a field slow mode) or per-chain noise?

This is the zero-cost half of the drift-attribution question. It reads the block tables that
ibi_round0.py already wrote (the "=== stationarity ===" section of each log) and asks, for every
chain and every coordinate, whether the block sim/ref value moves monotonically as the window
advances -- and whether the chains agree on WHICH coordinate and WHICH direction.

The two hypotheses put different signatures on those tables:

  * a FIELD slow mode (one specific coordinate that relaxes slowly) -> every chain drifts in the
    SAME coordinate, in a consistent direction (the sign of the slope agrees across chains).
  * a SAMPLING limitation (each chain relaxing from its own native start) -> each chain drifts
    in a different coordinate and/or a different direction; the block J is a random walk.

Judgment is by the sign of a least-squares slope of sim/ref against block index, cross-checked
with Spearman rho (rank correlation), which is robust to the odd block. With 8 blocks the power
is low per chain, so the decisive number is the AGREEMENT of the sign across the five chains, not
any single chain's slope.

This script runs no simulation and writes no log; it is a report over logs that already exist.
It prints everything it reads, so every number can be checked against the log by eye.

Run: python scripts/analyze_drift_trend.py
"""
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# The five chains of E2 (same protocol: 8 replicas, 100000 steps, burn 20000, 8 blocks,
# friction 0.1), plus E1b (idx 0 at 200000 steps, burn 40000) for the longer window.
CHAINS = [
    ("1L2X  idx0  100k", RESULTS / "E1_stationarity.log"),
    ("1Q96  idx1  100k", RESULTS / "E2_idx1.log"),
    ("3MJA  idx2  100k", RESULTS / "E2_idx2.log"),
    ("4PCJ  idx3  100k", RESULTS / "E2_idx3.log"),
    ("4RZD  idx4  100k", RESULTS / "E2_idx4.log"),
    ("1L2X  idx0  200k", RESULTS / "E1b_stationarity_200k.log"),
]

COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")

# block table row: "    1   40.0-  60.0     400  0.0958  1.129  0.999  1.023  0.903  0.721  1.002"
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)-\s*([\d.]+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    rx = x.argsort().argsort()
    ry = y.argsort().argsort()
    # Pearson on ranks == Spearman rho
    return float(np.corrcoef(rx, ry)[0, 1])


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def parse_blocks(path):
    blocks = {c: [] for c in COORDS}
    blocks["J"] = []
    blocks["frames"] = []
    in_table = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("=== stationarity"):
            in_table = True
            continue
        if not in_table:
            continue
        m = ROW.match(line)
        if not m:
            if blocks["J"]:
                break
            continue
        g = m.groups()
        blocks["frames"].append(int(g[3]))
        blocks["J"].append(float(g[4]))
        for i, c in enumerate(COORDS):
            blocks[c].append(float(g[5 + i]))
    return blocks


def main():
    print(f"{'chain':>20s}  {'coord':>9s}  {'slope':>8s} {'rho':>6s} {'sign':>5s} "
          f"  block sim/ref values (first->last)")
    print("-" * 100)
    # table of slope sign per (chain, coordinate) for the agreement test
    signs = {c: [] for c in COORDS}
    drift_coord_per_chain = {}
    for label, path in CHAINS:
        if not path.exists():
            print(f"{label}: MISSING {path}")
            continue
        b = parse_blocks(path)
        n = len(b["J"])
        if n < 3:
            continue
        x = np.arange(n)
        # which coordinate drifts most for this chain, by |slope| on sim/ref
        per_coord = {}
        for c in COORDS:
            s = slope(x, b[c])
            r = spearman(x, b[c])
            per_coord[c] = (s, r)
            signs[c].append(np.sign(s) if s != 0 else 0)
            vals = " ".join(f"{v:.3f}" for v in b[c])
            print(f"{label:>20s}  {c:>9s}  {s:+.4f} {r:+6.2f} "
                  f"{'+' if s > 0 else ('-' if s < 0 else '0'):>5s}  {vals}")
        drift_coord = max(COORDS, key=lambda c: abs(per_coord[c][0]))
        s, r = per_coord[drift_coord]
        jv = " ".join(f"{v:.4f}" for v in b["J"])
        js = slope(x, b["J"])
        jr = spearman(x, b["J"])
        print(f"{label:>20s}  {'J':>9s}  {js:+.4f} {jr:+6.2f} "
              f"{'+' if js > 0 else ('-' if js < 0 else '0'):>5s}  {jv}")
        drift_coord_per_chain[label] = (drift_coord, s)
        print("-" * 100)

    print()
    print("=== cross-chain agreement: for each coordinate, the sign of the slope per chain ===")
    print(f"{'coord':>9s}  {'signs (5 E2 chains)':>28s}  {'consistent?':>12s}")
    print("-" * 60)
    for c in COORDS:
        s = signs[c][:5]  # the five 100k chains; the 200k entry is listed separately
        pos = sum(1 for v in s if v > 0)
        neg = sum(1 for v in s if v < 0)
        consistent = (pos == 0 or neg == 0)
        print(f"{c:>9s}  {str(s):>28s}  {'yes' if consistent else 'no':>12s}")

    print()
    print("=== which coordinate drifts most, per chain ===")
    for label, (c, s) in drift_coord_per_chain.items():
        print(f"  {label:>24s}: {c:>9s}  (slope {s:+.4f})")


if __name__ == "__main__":
    main()
