"""Does the drift direction reproduce across seeds? (deterministic relaxation vs noise)

The friction control already showed the block-to-block drift is a sampling limitation: at the
same field/seed/window, friction 0.1 -> 5.0 moves the joint J by ~20 percent, which an
equilibrium quantity may not do. The remaining question is whether the drift is DETERMINISTIC
(the ensemble mean relaxing from the deposited folded start, with zero velocity, no matter the
noise) or STOCHASTIC (a particular noise realisation).

The discriminator is a second seed. Every replica in a run starts from the same deposited
structure with zero velocity; the seed changes only the Langevin noise. If seed 7 reproduces the
downward block trend in bb_bond/stack that seed 20260218 shows, the drift is the deterministic
relaxation of the ensemble mean (8 replicas x 400 frames averages the noise down). If it does
not, the drift is noise-driven and the "common direction" across chains is weaker than it looks.

This reads two logs that differ ONLY in seed -- results/E1_stationarity.log (seed 20260218,
friction 0.1) and results/F_seed7_seed7.log (seed 7, friction 0.1) -- and prints, per coordinate,
the whole-window sim/ref, the block range, and the block slope at each seed. The decisive columns
are bb_bond and stack: same downward sign at both seeds = deterministic; disagreeing sign = noise.

Run: python scripts/compare_seed_drift.py
"""
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)-\s*([\d.]+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_blocks(path):
    blocks = {c: [] for c in COORDS}
    blocks["J"] = []
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
        blocks["J"].append(float(g[4]))
        for i, c in enumerate(COORDS):
            blocks[c].append(float(g[5 + i]))
    return blocks


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0


def main():
    a = parse_blocks(RESULTS / "E1_stationarity.log")
    b = parse_blocks(RESULTS / "F_seed7_seed7.log")
    n = min(len(a["J"]), len(b["J"]))
    x = np.arange(n)

    print(f"seed comparison (friction 0.1, {n} blocks, 40-200 ps, 1L2X, idx 0)")
    print(f"{'coord':>10s} {'s20260218':>10s} {'seed7':>10s} {'range s20':>9s} {'range s7':>9s} "
          f"{'slope s20':>10s} {'slope s7':>10s}  same sign?")
    print("-" * 92)
    agree = 0
    for c in COORDS:
        va = np.array(a[c][:n], dtype=float)
        vb = np.array(b[c][:n], dtype=float)
        ra, rb = va.max() - va.min(), vb.max() - vb.min()
        wa, wb = va.mean(), vb.mean()
        s1, s2 = slope(x, va), slope(x, vb)
        same = (np.sign(s1) == np.sign(s2)) and s1 != 0 and s2 != 0
        agree += int(same)
        print(f"{c:>10s} {wa:10.3f} {wb:10.3f} {ra:8.3f} {rb:8.3f} {s1:+10.4f} {s2:+10.4f}  "
              f"{'yes' if same else 'no'}")
    print("-" * 92)
    print(f"slope sign agrees across seeds on {agree}/6 coordinates")
    print()
    print("bb_bond and stack are the two that drifted the same way across all five E2 chains.")
    print("Same downward sign here at a second seed -> deterministic ensemble-mean relaxation.")
    print("Disagreeing sign -> the cross-chain common direction is (partly) a noise coincidence.")


if __name__ == "__main__":
    main()
