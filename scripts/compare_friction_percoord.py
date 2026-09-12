"""Per-coordinate friction comparison of the block-to-block drift.

The total joint J averages the six coordinates together, which hides exactly the signal that
decides the drift-attribution question. The cross-chain trend analysis
(scripts/analyze_drift_trend.py) found bb_bond and stack drifting the same way in all five
chains while the other coordinates wander, and a teammate's isolated-torsion measurement
(scripts/*, Agent C) found that `dihedral` is a genuinely SLOW coordinate: at K_DIH=7.2 the
effective phi spring is ~0.355 kJ/mol/rad^2, giving a ~56 ps period at m=110, so 40-200 ps may
not sample it at all.

This script makes the friction control per-coordinate instead of per-J. It reads two runs that
differ ONLY in friction -- results/E1_stationarity.log (friction 0.1) and
results/F_fric5.0_seed20260218.log (friction 5.0), same field, same seed, same 40-200 ps window
-- and prints, for each coordinate: the whole-window sim/ref at each friction, the block spread
(max/min over the 8 disjoint blocks) at each friction, and the block slope at each friction.

The reading rule, per coordinate:
  * block spread that GROWS with friction -> the coordinate is overdamped at high friction and
    undersampled (a slow coordinate); its drift is relaxation, not a field slow mode.
  * block spread that COLLAPSES with friction -> a fast coordinate whose ringing the high
    friction damps out; it was near equilibrium already.
  * whole-window sim/ref that MOVES with friction -> the run is not equilibrated (equilibrium is
    friction-independent), i.e. sampling limitation, not a shifted equilibrium.

For dihedral specifically the target is NOT sim/ref = 1.0: the kBT/sigma^2 criterion fails for a
cosine coordinate, and the measure-correct 1-D is 0.596 of the pooled reference (Agent C). So
dihedral sim/ref converging downward toward ~0.6 is the chain approaching ITS OWN equilibrium,
not the field narrowing it.

Run: python scripts/compare_friction_percoord.py
"""
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")
# measure-correct 1-D target, as a multiple of the pooled reference sigma, per coordinate.
# 1.0 everywhere the kBT/sigma^2 criterion holds; 0.596 for dihedral (cosine coordinate, Agent C).
TARGET = {"bb_bond": 1.0, "intra_pc": 1.0, "intra_cn": 1.0, "angle": 1.0,
          "dihedral": 0.596, "stack": 1.0}

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)-\s*([\d.]+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_blocks(path):
    blocks = {c: [] for c in COORDS}
    in_table = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("=== stationarity"):
            in_table = True
            continue
        if not in_table:
            continue
        m = ROW.match(line)
        if not m:
            if blocks["bb_bond"]:
                break
            continue
        g = m.groups()
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
    lo = parse_blocks(RESULTS / "E1_stationarity.log")
    hi = parse_blocks(RESULTS / "F_fric5.0_seed20260218.log")
    n = min(len(lo["bb_bond"]), len(hi["bb_bond"]))
    x = np.arange(n)

    print(f"per-coordinate friction comparison ({n} blocks, 40-200 ps, 1L2X, seed 20260218)")
    print("whole-window sim/ref from each log's summary table; block spread = max/min over blocks.")
    print()
    print(f"{'coord':>10s} {'target':>7s} {'sim/ref g0.1':>12s} {'sim/ref g5.0':>12s} "
          f"{'range g0.1':>11s} {'range g5.0':>11s} {'slope g0.1':>11s} {'slope g5.0':>11s}  read")
    print("-" * 112)

    # whole-window sim/ref, read from the summary table of each log (not recomputed from blocks)
    # These are printed in the log as the "sim/ref" column of the per-coordinate table.
    whole = {}
    for tag, path in (("g0.1", RESULTS / "E1_stationarity.log"),
                      ("g5.0", RESULTS / "F_fric5.0_seed20260218.log")):
        whole[tag] = {}
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            # "bb_bond      0.0470   0.0471   0.0527    1.121   1.118 ..."
            m = re.match(r"^\s*(" + "|".join(COORDS) + r")\s+\S+\s+\S+\s+\S+\s+(\S+)\s", line)
            if m:
                whole[tag][m.group(1)] = float(m.group(2))

    for c in COORDS:
        a = np.array(lo[c][:n], dtype=float)
        b = np.array(hi[c][:n], dtype=float)
        ra, rb = a.max() - a.min(), b.max() - b.min()
        wa, wb = whole["g0.1"][c], whole["g5.0"][c]
        s1, s2 = slope(x, a), slope(x, b)
        # reading
        if rb > 1.5 * ra and abs(s2) > abs(s1):
            read = "SLOW/overdamped: block range and slope grow with friction"
        elif rb < 0.6 * ra:
            read = "fast: friction damps the ringing"
        elif abs(wb - wa) <= 0.02:
            read = "friction-invariant: near equilibrium at both"
        elif c == "dihedral":
            read = "moves with friction; target=%.3f, both above -> not converged" % TARGET[c]
        else:
            read = "moves with friction -> not equilibrated"
        print(f"{c:>10s} {TARGET[c]:7.3f} {wa:12.3f} {wb:12.3f} "
              f"{ra:10.3f} {rb:10.3f} {s1:+11.4f} {s2:+11.4f}  {read}")

    print()
    print("dihedral target 0.596: the pooled reference sigma (0.588) is wrong for a cosine")
    print("coordinate; its true 1-D is ~0.35. A dihedral drifting DOWN toward 0.6 is converging,")
    print("not being squeezed by the field.")


if __name__ == "__main__":
    main()
