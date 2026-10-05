"""Paired analysis of the stacking scan: same chains, same seeds, with and without the base-level term.

The scan writes one npz per setting, and each holds the PER-CHAIN coordinates as well as the pooled ones
(results/plan_c/_stack_sum_w*_eps*.npz). Because the two settings run the same chains with the same seeds,
the comparison is paired: for every chain the base-level total-variation distances and the trace joint J are
computed twice, and what is reported is the difference, its mean and standard deviation over chains, and how
many chains move the way the term is supposed to move them.

Nothing here re-samples. Usage:
    python scripts/analyze_stack_sum_paired.py <baseline.npz> <treatment.npz>
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import measure_base_stacking as M          # noqa: E402

COORDS = ("nb_dist", "rise", "theta")
GRID = {"nb_dist": (0.3, 1.6), "rise": (-1.0, 1.0), "theta": (0.0, 180.0)}


def tv(a, b, lo, hi, nbins=80):
    ha, _ = np.histogram(a[np.isfinite(a)], bins=nbins, range=(lo, hi))
    hb, _ = np.histogram(b[np.isfinite(b)], bins=nbins, range=(lo, hi))
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return 0.5 * float(np.abs(ha - hb).sum())


def load(path):
    with np.load(path) as z:
        # Chain names contain underscores (10ZT_1), so the key is split by PREFIX and SUFFIX rather than
        # by index -- the first attempt used split("_", 2)[1] and asked the archive for "chain_10ZT_...".
        names = sorted({k[len("chain_"):-len("_" + c)]
                        for k in z.files if k.startswith("chain_")
                        for c in COORDS if k.endswith("_" + c)})
        per = {nm: {c: z[f"chain_{nm}_{c}"] for c in COORDS} for nm in names}
        return per, float(z["j"][0]), names


def main():
    base, j_base, names = load(sys.argv[1])
    treat, j_treat, _n2 = load(sys.argv[2])
    tgt = np.load(REPO / "results" / "plan_c" / "_base_coords_planes.npz")
    target = {c: tgt[f"impl_{c}"] for c in COORDS}

    print("paired over %d chains: %s" % (len(names), " ".join(names)))
    print("%-9s %-22s %-22s %14s %10s" % ("coord", "baseline (mean +- sd)",
                                          "with the term (mean +- sd)", "paired change", "chains better"))
    for c in COORDS:
        tv_b = np.array([tv(base[nm][c], target[c], *GRID[c]) for nm in names])
        tv_t = np.array([tv(treat[nm][c], target[c], *GRID[c]) for nm in names])
        mean_b = np.array([np.nanmean(base[nm][c]) for nm in names])
        mean_t = np.array([np.nanmean(treat[nm][c]) for nm in names])
        lower = (tv_t < tv_b).sum()
        print("%-9s mean %8.4f +- %-7.4f  mean %8.4f +- %-7.4f  %+8.4f +- %-6.4f  %2d/%d"
              % (c, mean_b.mean(), mean_b.std(ddof=1), mean_t.mean(), mean_t.std(ddof=1),
                 (mean_t - mean_b).mean(), (mean_t - mean_b).std(ddof=1), lower, len(names)))
        print("%-9s TV   %8.4f +- %-7.4f  TV   %8.4f +- %-7.4f  %+8.4f +- %-6.4f  %2d/%d"
              % ("", tv_b.mean(), tv_b.std(ddof=1), tv_t.mean(), tv_t.std(ddof=1),
                 (tv_t - tv_b).mean(), (tv_t - tv_b).std(ddof=1), lower, len(names)))
        if c == "rise":
            p5_b = np.array([np.nanpercentile(base[nm][c], 5) for nm in names])
            p5_t = np.array([np.nanpercentile(treat[nm][c], 5) for nm in names])
            print("%-9s rise 5th pct  %+8.4f +- %-7.4f  %+8.4f +- %-7.4f  %+8.4f +- %-6.4f  %2d/%d"
                  % ("", p5_b.mean(), p5_b.std(ddof=1), p5_t.mean(), p5_t.std(ddof=1),
                     (p5_t - p5_b).mean(), (p5_t - p5_b).std(ddof=1), int((p5_t > p5_b).sum()), len(names)))
    print("\ntrace joint J (mean over chains, from the scan's own log): baseline %.4f   with the term"
          " %.4f   change %+.4f" % (j_base, j_treat, j_treat - j_base))
    print("  J is stored as a mean over chains, not per chain, so this one number is not paired; every"
          " number above it is.")


if __name__ == "__main__":
    main()
