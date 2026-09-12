"""How much of the dihedral table's near-vertical "needle wall" is a binning artifact?

The shipped dihedral table spans lo=-1.03999, hi=1.04000 with binw=0.01733, so its
steepest bin (117, slope 1049.8 kJ/mol per unit cos) straddles q=1.0: its range is
[0.9876, 1.0050], of which [1.0, 1.0050] is NONPHYSICAL (cos cannot exceed 1). The raw
observations are min=-0.999992, max=1.000000 -- zero mass beyond |q|=1 -- so that
nonphysical half of bin 117 is empty and carried by the pseudo-count alone, while the next
bin (118) is entirely nonphysical and equally pseudo-count. The near-vertical wall is
therefore the jump from a full peak bin (117) to an empty pseudo-count bin (118) that only
exists because the table extends 0.04 past the physical edge.

This script rebuilds the dihedral table from the same 6386 raw observations three ways and
measures the steepest-bin slope and the real max |F| (autograd, evaluated on 2OIU and the
pool) for each:

  1. 现状       lo/hi = data range +- 2% pad, 120 bins  (reproduces the shipped table)
  2. [lo,hi]    lo=-1.0, hi=+1.0, 120 bins             (binw = 0.01667)
  3. [lo,hi]    lo=-1.0, hi=+1.0, 240 and 480 bins

Re-binning is a NEW local function here; boltzmann_bonded.py and the shipped NPZ are not
touched. The table is rebuilt with the same _table_from_values recipe (pseudo=0.5,
U = -kBT ln((counts+pseudo)/...), shifted to min 0), and evaluated through
B.mixed_energy(which=["dihedral"]) after B.prepare, so the force is measured by the same
autograd path the production decision would use.

Criterion (from team-lead): if re-binning to the physical range -- or finer bins -- drops
max |F| below force_cap=5000, re-binning is the cheap fix and the cap principle is not
challenged. If none of the three does, the needle is the data's honest shape.

Run: python scripts/check_table_rebin.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
FORCE_CAP = 5000.0
GMAX = 10.6916                        # max |dq/dx|, from check_table_force_law.py (1L2X)
KBT = B.KBT
PSEUDO = 0.5


def build_table(v, lo, hi, nbins):
    """Same recipe as boltzmann_bonded._table_from_values, on an explicit support."""
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    p = (counts + PSEUDO) / (counts.sum() + PSEUDO * nbins)
    U = -KBT * np.log(p)
    U = U - U.min()
    return {"lo": float(lo), "hi": float(hi), "binw": float((hi - lo) / nbins),
            "U": U, "centre": 0.5 * (edges[:-1] + edges[1:]),
            "counts": counts, "sigma": float(v.std())}


def max_force(table, structs):
    """Autograd max |F| of the dihedral table term, evaluated on real geometries."""
    tt = {"dihedral": table}
    B.prepare(tt)
    maxf = 0.0
    who = ""
    for s in structs:
        L = len(s["pos"])
        pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64,
                           requires_grad=True)
        E = B.mixed_energy(pos, tt, which=["dihedral"])
        F = -torch.autograd.grad(E, pos)[0]
        m = float(F.reshape(-1, 3).norm(dim=-1).max())
        if m > maxf:
            maxf, who = m, s["name"]
    return maxf, who


def main():
    # raw observations, the same 6386
    structs = B.load_structures(limit=400)
    vals = []
    for s in structs:
        pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
        vals.append(B.coords_of(pos, "dihedral").reshape(-1).numpy())
    v = np.concatenate(vals)
    pool = [s for s in structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
    oiu = [s for s in structs if s["name"] == "2OIU"]
    targets = pool + oiu

    print("=" * 74)
    print("raw observations and the nonphysical half of the peak bin")
    print("=" * 74)
    print(f"  n = {len(v)}, min = {v.min():.6f}, max = {v.max():.6f}")
    print(f"  |q| > 1 (floating-point overshoot): {(np.abs(v) > 1.0).sum()}")
    lo_c, binw_c = -1.03999, 0.01733
    bin117 = (lo_c + 117 * binw_c, lo_c + 118 * binw_c)
    print(f"  shipped bin 117 = [{bin117[0]:.4f}, {bin117[1]:.4f}]")
    print(f"    observations in physical half [0.9876, 1.0]: "
          f"{(v >= bin117[0]) & (v <= 1.0)} -> {((v >= bin117[0]) & (v <= 1.0)).sum()}")
    print(f"    observations in nonphysical half (1.0, 1.0050]: "
          f"{((v > 1.0) & (v <= bin117[1])).sum()} (all pseudo-count)")

    print()
    print("=" * 74)
    print("three tables: steepest-bin slope and max |F|")
    print("=" * 74)
    print(f"  multiplier max|dq/dx| = {GMAX:.4f} /nm; force_cap = {FORCE_CAP}")

    # 1. 现状: reproduce the shipped fit (support = data range +- 2% pad)
    lo1 = float(v.min() - 0.02 * (v.max() - v.min()))
    hi1 = float(v.max() + 0.02 * (v.max() - v.min()))
    t1 = build_table(v, lo1, hi1, 120)
    # 2 / 3. physical range
    t2 = build_table(v, -1.0, 1.0, 120)
    t3a = build_table(v, -1.0, 1.0, 240)
    t3b = build_table(v, -1.0, 1.0, 480)

    rows = [
        ("现状  [-1.04, 1.04] 120", t1),
        ("物理  [-1.00, 1.00] 120", t2),
        ("物理  [-1.00, 1.00] 240", t3a),
        ("物理  [-1.00, 1.00] 480", t3b),
    ]
    print(f"  {'table':24s} {'binw':>8s} {'max |slope|':>12s} {'xGmax|F|':>10s} "
          f"{'/cap':>7s} {'real max|F|':>12s} {'/cap':>7s} {'at'}")
    print("  " + "-" * 70)
    for label, t in rows:
        U = np.asarray(t["U"], float)
        slope = np.diff(U) / t["binw"]
        ms = float(np.abs(slope).max())
        mf, who = max_force(t, targets)
        print(f"  {label:24s} {t['binw']:8.5f} {ms:12.2f} {ms * GMAX:10.1f} "
              f"{ms * GMAX / FORCE_CAP:7.3f} {mf:12.2f} {mf / FORCE_CAP:7.3f}   {who}")

    print()
    print("  slope is dU/dq (kJ/mol per unit cos). 'xGmax|F|' is team-lead's requested proxy")
    print("  (max slope times the fixed geometric factor); 'real max|F|' is the actual autograd")
    print("  force on 2OIU and the pool. Only the real force is the physical quantity.")

    # confirm the shipped-table reproduction matches the NPZ byte-for-byte
    z = np.load(NPZ)
    shipped_U = z["dihedral__U"]
    print()
    print("  shipped-table reproduction check: "
          f"{'MATCH' if np.allclose(np.asarray(t1['U']), shipped_U) else 'MISMATCH'}"
          f" (refit support=None vs results/boltzmann_tables_clean.npz)")

    # where does the max slope sit in each table, and how many obs are in the steepest bin?
    print()
    print("=" * 74)
    print("where the steepest slope sits, and how much real mass the steep bin carries")
    print("=" * 74)
    for label, t in rows:
        U = np.asarray(t["U"], float)
        slope = np.diff(U) / t["binw"]
        i = int(np.argmax(np.abs(slope)))
        lo_b = t["lo"] + i * t["binw"]
        hi_b = lo_b + t["binw"]
        cnt = int(t["counts"][i])
        print(f"  {label:24s} steepest bin {i}: [{lo_b:+.4f}, {hi_b:+.4f}] "
              f"obs={cnt:5d}  slope={slope[i]:+9.1f}")


if __name__ == "__main__":
    main()
