"""What would stratifying the Boltzmann tables by base identity actually buy?

boltzmann_bonded.fit() pools every observation of a coordinate into one histogram, so the
six bonded potentials it builds are sequence averaged. This script refits each coordinate
once per base-identity group (boltzmann_bonded.fit_stratified) and prices the result against
two things: the pooled table, and a null in which the base letters are permuted.

The null is the point. A group table fitted on a few hundred observations differs from the
pooled table no matter what the bases are doing, because a histogram of 300 counts over 120
bins is mostly counting noise. Permuting the label vector reassigns which observation gets
which base WITHOUT changing any group size, so the null is fitted on exactly the same
partition sizes as the real labelling, and the same label names are available in every
replicate -- which is what lets the per-group test in section 4b match a group to its own
null distribution rather than to a size-matched stranger.

Sections:
  0  the pooled path, re-fitted and diffed against results/boltzmann_tables_clean.npz
  1  what there is to stratify: chains, observations, groups per coordinate
  2  per coordinate: every group, its support, its k, and what fell back to pooled
  3  sensitivity of the support floor, and what the floor rejects on real data
  4  null control: real labelling vs permuted labelling
  4b per-group test: which individual groups beat their own relabelling null
  5  verdict, computed from the numbers above

Run: python scripts/refit_tables_stratified.py [n_chains] [n_perm]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B          # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
NPZ_LIMIT = 96                        # the limit refit_tables_clean.py fitted at
NBINS, PSEUDO = 120, 0.5
LIMIT = None if len(sys.argv) < 2 or sys.argv[1].lower() in ("none", "all", "0") else int(sys.argv[1])
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 40
GATES = (100, 200, 400, 800, 1200)


def summarise(v, lab, pooled, min_obs, nbins=NBINS, pseudo=PSEUDO):
    """Per-group numbers for one coordinate under one labelling, plus pooled deltas.

    dU is the mean absolute free-energy difference from the pooled table over the bins this
    group actually occupies. Empty bins are excluded because the pseudo-count, not the data,
    sets the group table there. The histogram is called with the same bins and range
    _table_from_values used, so the occupancy mask lines up with the stored U.
    """
    groups, counts, fallback = B.stratify_values(
        v, lab, pooled, nbins=nbins, pseudo=pseudo, min_obs=min_obs)
    rows = []
    for g, t in groups.items():
        vg = v[lab == g]
        c, _ = np.histogram(vg, bins=nbins, range=(t["lo"], t["hi"]))
        occ = c > 0
        dU = np.abs(t["U"][occ] - pooled["U"][occ]) if occ.any() else np.zeros(1)
        rows.append({"g": g, "n": counts[g], "sigma": t["sigma"], "k": t["k"],
                     "pf": t["pseudo_frac"], "empty": t["empty"],
                     "dU": float(dU.mean()), "dUmax": float(dU.max())})
    return rows, counts, fallback


def spread(rows):
    """Scale-free summaries of one labelling. Returns None when nothing is supported."""
    if not rows:
        return None
    k = np.array([r["k"] for r in rows])
    s = np.array([r["sigma"] for r in rows])
    n = np.array([r["n"] for r in rows], dtype=float)
    d = np.array([r["dU"] for r in rows])
    return {"ng": len(rows), "kratio": float(k.max() / k.min()),
            "sd_log_sigma": float(np.log(s).std()),
            "wdU": float((n * d).sum() / n.sum()), "maxdU": float(d.max()),
            "nmin": int(n.min()), "nmax": int(n.max())}


def sigmas_by_label(rows):
    """{label: sigma} for the groups that reached the floor, keyed by label."""
    return {r["g"]: r["sigma"] for r in rows}


# ------------------------------------------------------------------ 0. pooled
print("=== 0. the pooled path, re-fitted and diffed against the stored tables ===")
stored = np.load(NPZ)
t0 = B.fit(B.load_structures(limit=NPZ_LIMIT))
print(f"refit at limit={NPZ_LIMIT}, the same limit refit_tables_clean.py used")
print()
print(f"{'coordinate':12s} {'fields':>6s} {'max |refit - stored|':>21s} {'n':>7s}")
print("-" * 52)
worst, nfields = 0.0, 0
for name in B.COORDS:
    d = 0.0
    for f in ("lo", "hi", "binw", "U", "centre"):
        d = max(d, float(np.max(np.abs(np.asarray(t0[name][f]) - stored[name + "__" + f]))))
        nfields += 1
    worst = max(worst, d)
    print(f"{name:12s} {5:6d} {d:21.3e} {t0[name]['n']:7d}")
print("-" * 52)
print(f"maximum deviation over all {nfields} stored arrays = {worst:.3e}")
print()
print("Tolerance: 0.0, exact equality, not a slack bound. The only change to the pooled")
print("path is that its arithmetic moved into _table_from_values, so any non-zero value")
print("here would be a behaviour change rather than floating-point dust.")
print()

# ------------------------------------------------------------- 1. inventory
structs = B.load_structures(limit=LIMIT, with_names=True)
print("=== 1. what there is to stratify ===")
print(f"{len(structs)} chains loaded (limit={LIMIT})")
print()
vals, labs = {}, {}
for c in B.COORDS:
    vals[c], labs[c] = [], []
for s in structs:
    pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
    for c in B.COORDS:
        v = B.coords_of(pos, c).reshape(-1).numpy()
        lab = B.labels_for(s["names"], c)
        assert len(lab) == len(v), (c, len(lab), len(v))
        vals[c].append(v)
        labs[c].append(lab)
vals = {c: np.concatenate(a) for c, a in vals.items()}
labs = {c: np.concatenate(a) for c, a in labs.items()}
pooled = B.fit(structs)
print(f"{'coordinate':12s} {'labelling':14s} {'groups':>6s} {'n_obs':>7s} {'pooled sigma':>12s} {'pooled k':>9s}")
print("-" * 66)
for c in B.COORDS:
    sd = float(vals[c].std())
    print(f"{c:12s} {B.DEFAULT_SCHEME[c]:14s} {len(np.unique(labs[c])):6d} {len(vals[c]):7d} {sd:12.4f} {B.KBT / sd ** 2:9.1f}")
print()
print("The labelling is per coordinate, not shared: a bond has two flanking bases, an")
print("angle three, the dihedral four. B.DEFAULT_SCHEME records the choice and the reason.")
print()

# ---------------------------------------------------- 2. groups at the floor
print(f"=== 2. every group at the support floor min_obs = {B.MIN_OBS} ===")
real_rows, preal = {}, {}
for c in B.COORDS:
    rows, counts, fb = summarise(vals[c], labs[c], pooled[c], B.MIN_OBS)
    rows.sort(key=lambda r: -r["n"])
    real_rows[c], preal[c] = rows, counts
    pf = [r["pf"] for r in rows]
    print()
    print(f"--- {c}  ({B.DEFAULT_SCHEME[c]}) ---")
    print(f"{'group':6s} {'n':>6s} {'sigma':>8s} {'k=kBT/s^2':>10s} {'pseudo_frac':>11s} {'emptybins':>10s} {'mean|dU|':>10s} {'max':>8s}")
    for r in rows:
        print(f"{r['g']:6s} {r['n']:6d} {r['sigma']:8.4f} {r['k']:10.1f} {r['pf']:11.3f} {r['empty']:5d}/{len(pooled[c]['U']):<4d} {r['dU']:10.3f} {r['dUmax']:8.3f}")
    sp = spread(rows)
    ks = np.array([r["k"] for r in rows]) if rows else np.array([np.nan])
    print(f"  supported {len(rows)}/{len(counts)} groups; observation counts {int(min(counts.values()))} to {int(max(counts.values()))}")
    if rows:
        print(f"  implied k across groups: {ks.min():.1f} to {ks.max():.1f}, ratio {sp['kratio']:.3f}, sd(log sigma) {sp['sd_log_sigma']:.4f}")
        print(f"  pseudo-count share of the probability mass: {min(pf):.3f} to {max(pf):.3f}")
        print(f"  observation-weighted mean |dU| away from pooled: {sp['wdU']:.3f} kJ/mol = {sp['wdU'] / B.KBT:.3f} kBT, worst group {sp['maxdU']:.3f}")
    else:
        print("  no group reaches the floor; this coordinate stays pooled")
    if fb:
        print("  FALLBACK to pooled -- " + ", ".join(f"{g}(n={counts[g]})" for g in sorted(fb, key=lambda x: -counts[x])))
    else:
        print("  fallback: none, every group is supported")
print()

# ------------------------------------------------------- 3. gate sensitivity
print("=== 3. sensitivity of the support floor ===")
print("cells are supported/total groups; the floor is a judgement call, so here is how the")
print("answer moves with it. The k ratio grows with the floor because only the large groups")
print("survive, and small groups are where the apparent spread comes from.")
print()
print(f"{'coordinate':12s} {'labelling':14s} " + " ".join(f"{g:>7d}" for g in GATES))
print("-" * 70)
for c in B.COORDS:
    cells = []
    for g in GATES:
        rows, counts, fb = summarise(vals[c], labs[c], pooled[c], g)
        sp = spread(rows)
        cells.append(f"{sp['ng']:>3d}/{len(counts):<3d}" if sp else f"{0:>3d}/{len(counts):<3d}")
    print(f"{c:12s} {B.DEFAULT_SCHEME[c]:14s} " + " ".join(f"{x:>7s}" for x in cells))
print()
print("k ratio at each floor:")
for c in B.COORDS:
    ratios = []
    for g in GATES:
        rows, _, _ = summarise(vals[c], labs[c], pooled[c], g)
        sp = spread(rows)
        ratios.append(f"{g}:{sp['kratio']:.3f}" if sp else f"{g}:--")
    print(f"  {c:12s} " + "  ".join(ratios))
print()
print("and the fallback the floor produces on real data, at floor=400 and floor=800:")
for c in B.COORDS:
    for g in (400, 800):
        rows, counts, fb = summarise(vals[c], labs[c], pooled[c], g)
        if fb:
            print(f"  {c:12s} floor={g:4d}  pooled-priced: " + ", ".join(f"{x}(n={counts[x]})" for x in sorted(fb, key=lambda y: -counts[y])))
        else:
            print(f"  {c:12s} floor={g:4d}  pooled-priced: none")
print()

# ------------------------------------------------------------ 4. null control
print(f"=== 4. null control: {NPERM} permutations of the base-letter vector ===")
print("The label vector is permuted globally, so every group keeps its exact size and its")
print("own name, and the partition is refitted identically. Only the base-to-geometry")
print("assignment is gone, which is exactly the hypothesis being tested.")
print()
rng = np.random.default_rng(11)
print(f"{'coordinate':12s} {'labelling':14s} {'sd log sigma: real [null p5,p95]':>34s} {'k ratio: real [null p5,p95]':>30s} {'mean |dU|: real [null p5,p95]':>32s} {'signal frac':>11s} {'p':>7s}")
print("-" * 138)
verdicts = {}
for c in B.COORDS:
    real_sp = spread(real_rows[c])
    nulls, null_sig = [], {g: [] for g in sigmas_by_label(real_rows[c])}
    for _ in range(NPERM):
        lab = rng.permutation(labs[c])
        rows, counts, fb = summarise(vals[c], lab, pooled[c], B.MIN_OBS)
        assert set(counts) == set(preal[c]), "the null changed the partition"
        nulls.append(spread(rows))
        for g, sg in sigmas_by_label(rows).items():
            null_sig[g].append(sg)
    if real_sp is None or any(x is None for x in nulls):
        print(f"{c:12s} {B.DEFAULT_SCHEME[c]:14s} no group reaches the floor; no null to run")
        verdicts[c] = None
        continue
    z = {k: np.array([x[k] for x in nulls]) for k in real_sp}
    p_sd = (1 + int((z["sd_log_sigma"] >= real_sp["sd_log_sigma"]).sum())) / (1 + NPERM)
    # How much of the observed spread is left after subtracting the variance a relabelling
    # of the same partition produces? That subtraction is the only honest way to read a
    # spread off a few hundred observations per group.
    null_var = float(np.mean(z["sd_log_sigma"] ** 2))
    excess = max(0.0, real_sp["sd_log_sigma"] ** 2 - null_var)
    sig_frac = float(np.sqrt(excess) / real_sp["sd_log_sigma"])
    verdicts[c] = {"p_sd": p_sd, "sig_frac": sig_frac, "null_sd": float(np.sqrt(null_var)),
                   "sd_ratio": real_sp["sd_log_sigma"] / z["sd_log_sigma"].mean(),
                   "wdU_ratio": real_sp["wdU"] / z["wdU"].mean() if z["wdU"].mean() else float("nan"),
                   "null_sig": {g: np.array(v) for g, v in null_sig.items()}}
    sd_lo, sd_hi = np.percentile(z["sd_log_sigma"], [5, 95])
    kr_lo, kr_hi = np.percentile(z["kratio"], [5, 95])
    du_lo, du_hi = np.percentile(z["wdU"], [5, 95])
    sd = f"{real_sp['sd_log_sigma']:6.4f} [{sd_lo:.4f},{sd_hi:.4f}]"
    kr = f"{real_sp['kratio']:7.3f} [{kr_lo:.3f},{kr_hi:.3f}]"
    du = f"{real_sp['wdU']:7.3f} [{du_lo:.3f},{du_hi:.3f}]"
    print(f"{c:12s} {B.DEFAULT_SCHEME[c]:14s} {sd} {kr} {du} {sig_frac:11.3f} {p_sd:7.3f}")
print()
print("signal frac is sqrt(1 - null_variance / real_variance) on sd(log sigma): the share of")
print("the observed stiffness spread that a relabelling of the same partition does not")
print("reproduce. 0 means the spread is counting noise; 1 means every bit of it is the bases.")
print(f"p is one-sided on sd(log sigma) with {NPERM} permutations, so its floor is {1 / (1 + NPERM):.4f}.")
print()

# ---------------------------------------------------- 4b. per-group testing
print("=== 4b. per-group test against the group's own relabelling null ===")
print("For each label the null gives the distribution of that label's sigma under random")
print("assignment, so a group is compared with a same-size, same-name relabelling of itself.")
print("d is |log sigma_group - median_null(log sigma)|, and the p-value is the share of null")
print("draws reaching d. Bonferroni over the groups tested in that coordinate is applied,")
print("because with 16 groups at alpha=0.05 roughly one false positive is expected by chance.")
print()
print(f"{'coordinate':12s} {'groups':>6s} {'alpha=0.05/k':>12s} {'significant':>11s}  which")
print("-" * 78)
sig_counts = {}
for c in B.COORDS:
    v = verdicts.get(c)
    if v is None:
        print(f"{c:12s} unsupported; no test")
        sig_counts[c] = 0
        continue
    rows = real_rows[c]
    ng = len(rows)
    alpha = 0.05 / ng
    hits = []
    for r in rows:
        nl = np.log(v["null_sig"][r["g"]])
        med = float(np.median(nl))
        d_real = abs(float(np.log(r["sigma"])) - med)
        d_null = np.abs(nl - med)
        p_g = (1 + int((d_null >= d_real).sum())) / (1 + NPERM)
        if p_g < alpha:
            hits.append(f"{r['g']}({r['k']:.0f}, p={p_g:.4f})")
    sig_counts[c] = len(hits)
    shown = ", ".join(hits) if hits else "none"
    print(f"{c:12s} {ng:6d} {alpha:12.4f} {len(hits):11d}  {shown}")
print()
print("A group in the list is one whose implied stiffness the base letters really do set, at")
print("the size this database gives it. A group absent from the list is not shown to be equal")
print("to the pooled value, only to be indistinguishable from it at this sample size -- which")
print("is the honest reading, and is why the pooled table stays the default.")
print()

# ------------------------------------------------------------------ 5. verdict
print("=== 5. verdict, read off the numbers above ===")
for c in B.COORDS:
    v = verdicts.get(c)
    sp = spread(real_rows[c])
    if v is None or sp is None:
        print(f"  {c:12s} unsupported at min_obs={B.MIN_OBS}")
        continue
    same = "no" if v["p_sd"] > 0.05 else "yes"
    print(f"  {c:12s} k ratio {sp['kratio']:.3f}  spread vs null {v['sd_ratio']:.2f}x  of which signal {100 * v['sig_frac']:.1f} pct  table shift vs null {v['wdU_ratio']:.2f}x  groups independently significant {sig_counts[c]}/{len(real_rows[c])}  p={v['p_sd']:.4f}  separable from noise: {same}")
print()