"""Transferability error bars, second attempt -- this time with the production pool and a shape check.

TWO BUGS IN THE FIRST ATTEMPT, both mine:
  1. the pool was built with "len(pairs) >= 8", which is the SMALL branch's filter; the production loop's
     "all" branch loads every chain (IBI_LOOP_POOL=all -> B.load_structures(limit=5000) with no filter),
     which is where its 867 comes from against my 609.
  2. a chain whose "pos" is not (L, 3, 3) silently contributed garbage to the pooled means: the first
     attempt read bb_bond = 1.6 where the direct geometry of the same chains is 0.58 nm. Every chain is
     now shape-normalised AND sanity-checked (median P-P step within 0.4-1.2 nm) or skipped and counted.

What it measures, on the deposited P traces the loop was fitted to, with no sampling:
  A. END EFFECTS  -- per-chain (interior - all) shift in mean and sd, bucketed by length.
  B. CLOSURE      -- interior marginals bucketed by end-to-end extension R_ee / R_max; a circle is the
                     R_ee -> 0 extreme, so any trend with R_ee is the size of the closure error.
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ["TORUSFOLD_RSRNASP"] = str(REPO / "_cgdata" / "combined")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

COORDS = ("bb_bond", "angle", "dihedral", "stack")
TRIM = 3
BOND_STEP = 0.59


def p_trace(s):
    """(L,3) nm P trace, or None if the record's shape or scale is not what the field assumes."""
    p = np.asarray(s["pos"], dtype=float)
    if p.ndim != 3:
        return None, "ndim"
    if p.shape[1] == 3 and p.shape[0] >= 4:
        pass
    elif p.shape[0] == 3 and p.shape[1] >= 4:
        p = np.transpose(p, (1, 0, 2))
    else:
        return None, f"shape {p.shape}"
    tr = p[:, 0, :]
    step = float(np.median(np.linalg.norm(tr[1:] - tr[:-1], axis=1)))
    if not (0.4 <= step <= 1.2):
        return None, f"P-P step {step:.2f} nm"
    return tr, None


def main():
    import torch
    import boltzmann_bonded as B

    print("dataset:", os.environ["TORUSFOLD_RSRNASP"])
    pool = B.load_structures(limit=5000)          # the "all" branch: no pairs filter
    print(f"chains loaded: {len(pool)}   (production loop: 867)")
    rows, skipped = [], {}
    for s in pool:
        tr, why = p_trace(s)
        if tr is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        L = tr.shape[0]
        if L < 2 * TRIM + 6:
            skipped["too short"] = skipped.get("too short", 0) + 1
            continue
        # THE BUG THAT COST THE FIRST TWO ATTEMPTS: tensor.repeat(1, 3, 1) TILES the trace
        # (P0..Pn,P0..Pn,P0..Pn) instead of interleaving it (P0,P0,P0,P1,P1,P1,...), so bead slot 3i
        # was not residue i's P and every coordinate came out at the wrong scale. The field stores
        # (L, 3, 3) residue-major, so a bare trace has to be interleaved with np.repeat(..., axis=0).
        t = torch.tensor(np.repeat(tr, 3, axis=0), dtype=torch.float64).reshape(1, 3 * L, 3)
        rec = {"L": L, "ext": float(np.linalg.norm(tr[-1] - tr[0])) / ((L - 1) * BOND_STEP)}
        for c in COORDS:
            v = B.coords_of(t, c).reshape(-1).numpy()
            rec[c] = (float(v.mean()), float(v.std()),
                      float(v[TRIM:-TRIM].mean()), float(v[TRIM:-TRIM].std()))
        rows.append(rec)
    print(f"used: {len(rows)}   skipped: {skipped}")
    bb = float(np.median([r["bb_bond"][0] for r in rows]))
    print(f"SELF-CHECK: median pooled bb_bond = {bb:.4f} nm (the field's tables assume ~0.59)\n")

    print("A. END EFFECTS -- per chain (interior - all), median over chains")
    print(f"   {'L bucket':>12} {'n':>4} | " + " | ".join(f"{c:^26s}" for c in COORDS))
    print(f"   {'':>12} {'':>4} | " + " | ".join(f"{'dmean   dsd   |  interior sd':^26s}" for c in COORDS))
    for lo, hi in ((0, 60), (60, 150), (150, 400), (400, 10 ** 9)):
        sel = [r for r in rows if lo < r["L"] <= hi]
        if not sel:
            continue
        cells = []
        for c in COORDS:
            dm = float(np.median([r[c][2] - r[c][0] for r in sel]))
            ds = float(np.median([r[c][3] - r[c][1] for r in sel]))
            sd = float(np.median([r[c][3] for r in sel]))
            cells.append(f"{dm:+.4f} {ds:+.4f}  |   {sd:.4f}   ")
        label = f"{lo}-{'plus' if hi >= 10 ** 9 else hi}"
        print(f"   {label:>12} {len(sel):>4} | " + " | ".join(cells))

    print("\nB. CLOSURE -- interior statistics by extension (L >= 60)")
    print(f"   {'R_ee/R_max':>12} {'n':>4} | " + " | ".join(f"{c:^22s}" for c in COORDS))
    for lo, hi in ((0.0, 0.15), (0.15, 0.35), (0.35, 0.6), (0.6, 1.01)):
        sel = [r for r in rows if lo <= r["ext"] < hi and r["L"] >= 60]
        if not sel:
            continue
        cells = []
        for c in COORDS:
            m = float(np.median([r[c][2] for r in sel]))
            s = float(np.median([r[c][3] for r in sel]))
            cells.append(f"{m:+.4f} +/- {s:.4f}  ")
        print(f"   {f'{lo:.2f}-{hi:.2f}':>12} {len(sel):>4} | " + " | ".join(cells))

    import ibi_bonded as I
    ref = I.load_clean_tables(str(REPO / "results" / "refit_smooth5.npz"))
    print("\n   target (refit_smooth5) stored sigmas: " +
          ", ".join(f"{c}={float(ref[c]['sigma']):.4f}" for c in COORDS))


if __name__ == "__main__":
    main()
