"""The clean instrument: what two SAMPLED ensembles say about each other, and its floor.

WHY THIS EXISTS, measured 2026-09-23 (docs/archive/plan_c_c2_stabilization.md sections 3.4-3.5). The
instrument the first two passes used to ask whether a self-consistent loop is closing was
fit_implied_ln_ratio -- the Boltzmann distribution a FIELD implies, compared with the histogram the
sampler produced. Its floor is small (0.007 / 0.009 / 0.035 median ln-units for bb_bond, angle,
dihedral when the field IS the histogram's exact inverse), but the field that PRODUCED an ensemble
scores 0.218 / 1.721 / 0.514 on it, i.e. the sampled ensemble is not the Boltzmann distribution of
its own field and cannot be: the sampled Hamiltonian also carries the wall potential
(table_wall:2000) and every coupling to the rest of the chain. Anything the proxy says about a fit
is therefore read through a confound that is larger than the effect.

A comparison of two SAMPLED ensembles has no such confound: both sides were produced by the same
kind of object -- one sampler, one full Hamiltonian, one field each -- so a difference between them
is the field difference plus sampling noise, and nothing else. That is the instrument this module
is: six numbers per coordinate, computed from the per-round ensembles the loop already stores, with
the noise floor measured from the data itself.

WHAT IT REPORTS between ensembles a and b on a common grid (the table's own bins):
  tv             0.5 * sum |p_b - p_a| over the mass-bearing bins, renormalised to their mass
  ln_mean        mass-weighted mean |ln(p_b/p_a)| over those bins -- the loop's own stationarity
  ln_max         max |ln(p_b/p_a)| over them
  dmean_sig      (<q>_b - <q>_a) / sigma, the shift of the mean in units of the coordinate's spread
  dstd_sig       (sigma_b - sigma_a) / sigma
  dq05/dq50/dq95_sig   quantile shifts, same units -- robust to a single noisy bin, which is what
                 ln_max is not

THE FLOOR comes from three places and only one of them needs new sampling (see the to-run list in
the doc):

  1. EXACT ZERO. Same field, same seed, bit for bit. Measured on this run: C0_r1 and C1_r1 are
     identical arrays (both sampled fields/run1/start.npz with seed + 1000 + i), and the refused
     first attempt's C2s8_r1 equals the second attempt's C2s8_r1 although the fit code changed in
     between -- the sampler does not depend on the fit.
  2. FROM THE DATA. The arm whose field moved least between two rounds has a round-to-round distance
     that is almost all noise: apply the instrument to every adjacent pair in every arm, keep the
     pairs whose injected step was below a threshold, and the smallest of those is the tightest
     empirical bound this dataset can give. Measured: C0's last pairs.
  3. POISSON BOUND. What each statistic would read if the frames were independent, by Monte Carlo on
     the observed histogram (draw two Poisson samples per bin). This is a LOWER bound on the true
     floor, because stride-5 frames are correlated and the effective sample size is smaller than N.

Run (offline only; it reads results/plan_c and writes nothing else):

    python scripts/plan_c_instrument.py --run1 results/plan_c/plan_c_run1.json \\
        --c2stab results/plan_c/plan_c_c2stab.json --ensembles results/plan_c/ensembles_*.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_bonded as I                # noqa: E402
import plan_c_loop as L               # noqa: E402

COORDS = ("bb_bond", "angle", "dihedral")
# Below this injected step (mass-weighted std, kJ/mol) a round-to-round distance is taken as
# floor-like: the field barely moved, so what is left is noise. Measured range of the arms' applied
# steps is 0.15 (C0 r4) to 4.5 (C2s4's dihedral), so 1 kJ/mol selects the settled end.
FLOOR_STEP_MAX = 1.0
LN_FLOOR = 1e-6


def load_ensembles(path):
    """(key -> counts) for a stored ensemble file; the convention is <arm>_r<round>__<coord>."""
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.endswith("__n_total") or "__n_total__" in k or "__n_outside__" in k:
            continue
        if "__" in k.split("__", 1)[1]:
            # <name>__chain<i>__<coord> and <name>__blocks__<coord> are the decomposition, read by
            # chains_of/blocks_of below. A caller that asked for the pooled ensemble must not
            # silently receive a decomposed one, so they are skipped here rather than merged.
            continue
        arm_round, coord = k.split("__", 1)
        out.setdefault(arm_round, {})[coord] = np.asarray(z[k], dtype=float)
    return out


def chains_of(z, arm_round, coord):
    """{chain index: counts} for one name-round and coordinate, or {} if it was not stored.

    The pooled histogram is one number; this is the distribution behind it, which is what turns a
    floor from a bracket into a line (docs/archive/plan_c_c2_stabilization.md 4.6).
    """
    pre = f"{arm_round}__chain"
    out = {}
    for k in z.files:
        if not k.startswith(pre) or not k.endswith(f"__{coord}"):
            continue
        mid = k[len(pre):-len(coord) - 2]
        if mid.endswith("_blocks") or not mid.isdigit():
            continue
        out[int(mid)] = np.asarray(z[k], dtype=float)
    return out


def blocks_of(z, arm_round, coord):
    """(nblocks, nbins) counts summed over chains, or None when the run did not store them."""
    key = f"{arm_round}__blocks__{coord}"
    return np.asarray(z[key], dtype=float) if key in z.files else None


def jackknife(values):
    """Delete-one jackknife over a list of block-level estimates: mean and the variance estimate.

    var_jack = (n-1)/n * sum((theta_b - theta_bar)^2) over the n delete-one replicates. It answers
    the question a pooled pair cannot: how much of this distance is the particular window that was
    sampled, at the size of the whole window rather than of one block.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return {"n": int(v.size), "mean": float(v.mean()) if v.size else float("nan"),
                "std_jackknife": float("nan")}
    return {"n": int(v.size), "mean": float(v.mean()),
            "std_jackknife": float(np.sqrt((v.size - 1) / v.size * ((v - v.mean()) ** 2).sum()))}


def metrics(a, b, centre, ln_floor=LN_FLOOR, reps=200, seed=20260923, pseudo=I.DEFAULT_PSEUDO):
    """The instrument. a, b: counts on the same bins; centre: bin centres; sigma from the pooled data.

    Everything is computed on the bins the TWO ensembles together have mass in (0.5*(p_a+p_b) above
    ln_floor), renormalised to that mass, so a bin that only one side visited cannot dominate a
    ratio and a bin neither side visited cannot enter at all. This is deliberately the convention
    plan_c_loop._stationarity uses for ln_mean -- so this module reproduces the loop's own record --
    with four more statistics beside it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"ensembles on different grids: {a.shape} vs {b.shape}")
    na, nb = a.sum(), b.sum()
    if na <= 0 or nb <= 0:
        raise ValueError("an ensemble with no observations")
    # THE PSEUDOCOUNT POLICY IS ibi_bonded's, CALLED RATHER THAN REIMPLEMENTED, and that is a
    # measurement decision. Dividing the counts by their own total instead leaves every bin that
    # neither ensemble visited at p = 0, which clip(1e-300) turns into a |ln ratio| of about 690
    # with the visit-one-side weight still attached to it: measured on C2s8's bb_bond r1 -> r2 that
    # reads 2.684 where the loop's own stationarity (and this function, corrected) reads 1.129, i.e.
    # a factor of 2.4 out of nothing. The loop normalises with (counts + 0.5) / (N + 0.5 * nbins),
    # so no bin is ever at zero and the ratio there is the ln of the two totals.
    pa = I.probability_from_counts(a, pseudo=pseudo)
    pb = I.probability_from_counts(b, pseudo=pseudo)
    w = 0.5 * (pa + pb)
    m = w > ln_floor
    if not m.any():
        raise ValueError("no bin carries mass in both ensembles")
    wa, wb = pa[m] / pa[m].sum(), pb[m] / pb[m].sum()

    lr = np.abs(np.log(np.clip(pb[m], 1e-300, None) / np.clip(pa[m], 1e-300, None)))
    rw = 0.5 * (wa + wb)
    tv = 0.5 * float(np.abs(wb - wa).sum())

    c = np.asarray(centre, dtype=float)[m]
    ma, mb = float((wa * c).sum()), float((wb * c).sum())
    sig = float(np.sqrt(0.5 * ((wa * (c - ma) ** 2).sum() + (wb * (c - mb) ** 2).sum())))
    ca, cb = np.cumsum(wa), np.cumsum(wb)

    def quant(cdf, q):
        return float(c[min(int(np.searchsorted(cdf, q)), len(c) - 1)])

    out = {"tv": tv,
           "ln_mean": float((rw * lr).sum()),
           "ln_max": float(lr.max()),
           "n_bins": int(m.sum()),
           "n_a": int(na), "n_b": int(nb),
           "dmean_sig": (mb - ma) / sig if sig > 0 else float("nan"),
           "dstd_sig": 0.0, "dq05_sig": 0.0, "dq50_sig": 0.0, "dq95_sig": 0.0}
    if sig > 0:
        sa = float(np.sqrt((wa * (c - ma) ** 2).sum()))
        sb = float(np.sqrt((wb * (c - mb) ** 2).sum()))
        out["dstd_sig"] = (sb - sa) / sig
        out["dq05_sig"] = (quant(cb, 0.05) - quant(ca, 0.05)) / sig
        out["dq50_sig"] = (quant(cb, 0.50) - quant(ca, 0.50)) / sig
        out["dq95_sig"] = (quant(cb, 0.95) - quant(ca, 0.95)) / sig
    # The Poisson floor for THIS pair, computed where the data is rather than assumed: two draws of
    # the same histogram's counts. Returned as the 95th percentile over the replicates, which is the
    # number a single comparison should be read against.
    if reps:
        rng = np.random.default_rng(seed)
        draws = {"tv": [], "ln_mean": [], "ln_max": [], "dq50_sig": []}
        lam = 0.5 * (a + b)
        for _ in range(int(reps)):
            ra = rng.poisson(lam).astype(float)
            rb = rng.poisson(lam).astype(float)
            try:
                d = metrics(ra, rb, np.asarray(centre, dtype=float), ln_floor, reps=0)
            except ValueError:
                continue
            for k in draws:
                draws[k].append(d[k])
        out["poisson_p95"] = {k: float(np.percentile(v, 95)) for k, v in draws.items() if v}
    return out


def step_std(field_a, field_b, counts, coord):
    """The mass-weighted std of the step the field between two rounds actually took, kJ/mol.

    This is the applied-step column plan_c_loop records per round, recomputed from the two field
    files so a pair can be classified as floor-like or not without trusting the json.
    """
    ua = np.asarray(field_a[coord]["U"], dtype=float)
    ub = np.asarray(field_b[coord]["U"], dtype=float)
    p = I.probability_from_counts(np.asarray(counts, dtype=float), pseudo=I.DEFAULT_PSEUDO)
    d = ub - ua
    mean = float((p * d).sum())
    return float(np.sqrt((p * (d - mean) ** 2).sum()))

# --------------------------------------------------------------------------- datasets
OUT = REPO / "results" / "plan_c"
DATASETS = {
    "run1": {"ensembles": OUT / "ensembles_run1.npz", "json": OUT / "plan_c_run1.json",
             "fields": OUT / "fields" / "run1"},
    "c2stab": {"ensembles": OUT / "ensembles_c2stab.npz", "json": OUT / "plan_c_c2stab.json",
               "fields": OUT / "fields" / "c2stab"},
    "dihedral": {"ensembles": OUT / "ensembles_dihedral.npz", "json": OUT / "plan_c_dihedral.json",
                 "fields": OUT / "fields" / "dihedral"},
}

# THE LINE, not zero. Measured 2026-09-24 (scripts/plan_c_same_field_floor.py, 614 s on 6 workers):
# the distance between two independent trajectories under ONE field, pooled over the same seven
# chains every arm uses. A round-to-round distance means nothing until it is read against this, and
# every table below prints the ratio.
SAME_FIELD_FLOOR = {
    "ln_mean": {"bb_bond": 0.0471, "angle": 0.0521, "dihedral": 0.0589},
    "tv": {"bb_bond": 0.0235, "angle": 0.0260, "dihedral": 0.0294},
    "source": "results/plan_c/same_field_floor.json (pooled, C2s8_r4 field, 2026-09-24)",
}


def record_lookup(doc):
    """(arm, round) -> the round's record, and the coordinate fitting diagnostic beside it."""
    out = {}
    for arm, rec in (doc or {}).get("arms", {}).items():
        for r in rec.get("rounds", []):
            out[(arm, int(r["round"]))] = r
    return out


def pairs_for(ens, arm):
    """(round_a, round_b) for the consecutive rounds this arm has ensembles for."""
    rounds = sorted(int(k.split("_r")[1]) for k in ens if k.startswith(arm + "_r"))
    return [(r, r + 1) for r in rounds if (r + 1) in rounds]


def analyse(datasets, step_max=FLOOR_STEP_MAX, reps=200):
    """Every adjacent-round pair in every arm of every dataset, with the polluted reading beside it.

    ALIGNMENT. The clean number for (r -> r+1) and the polluted number recorded at round r are both
    statements about the field written at the END of round r: the first says how far the ensemble
    then moved, the second how well the fit that produced that field represents the ensemble it was
    fitted to. Reading them as two instruments watching the same update is what section 3 is about.
    """
    rows = []
    for tag, cfg in datasets.items():
        ens = load_ensembles(cfg["ensembles"])
        doc = json.load(open(cfg["json"])) if Path(cfg["json"]).exists() else {}
        lookup = record_lookup(doc)
        for arm in sorted({k.split("_r")[0] for k in ens}):
            for ra, rb in pairs_for(ens, arm):
                fa = L.load_field(cfg["fields"] / f"{arm}_r{ra}.npz")
                fb = L.load_field(cfg["fields"] / f"{arm}_r{rb}.npz")
                for c in COORDS:
                    if c not in ens[f"{arm}_r{ra}"] or c not in ens[f"{arm}_r{rb}"]:
                        continue
                    a, b = ens[f"{arm}_r{ra}"][c], ens[f"{arm}_r{rb}"][c]
                    m = metrics(a, b, fa[c]["centre"], reps=reps)
                    step = step_std(fa, fb, b, c)
                    rec = lookup.get((arm, ra), {})
                    polluted = (rec.get("fit") or {}).get(c, {}).get(
                        "fit_implied_ln_ratio_median")
                    rows.append({"tag": tag, "arm": arm, "ra": ra, "rb": rb, "coord": c,
                                 "step_std": step, "polluted_ln_median": polluted, **m})
    return rows


def floor_rows(rows, step_max=FLOOR_STEP_MAX):
    """The empirical floor: pairs whose injected step was small enough that noise is what is left."""
    out = {}
    for c in COORDS:
        sel = [r for r in rows if r["coord"] == c and r["step_std"] <= step_max]
        sel.sort(key=lambda r: r["ln_mean"])
        out[c] = {"n": len(sel),
                  "ln_mean_min": sel[0]["ln_mean"] if sel else None,
                  "ln_mean_median": float(np.median([r["ln_mean"] for r in sel])) if sel else None,
                  "tv_min": min([r["tv"] for r in sel]) if sel else None,
                  "tv_median": float(np.median([r["tv"] for r in sel])) if sel else None,
                  "poisson_p95_ln_mean": float(np.median(
                      [r["poisson_p95"]["ln_mean"] for r in sel
                       if "poisson_p95" in r and "ln_mean" in r["poisson_p95"]])) if sel else None,
                  "poisson_p95_tv": float(np.median(
                      [r["poisson_p95"]["tv"] for r in sel
                       if "poisson_p95" in r and "tv" in r["poisson_p95"]])) if sel else None,
                  "members": [(r["tag"], r["arm"], r["ra"], r["rb"], round(r["step_std"], 2))
                              for r in sel[:8]]}
    return out


def trend(vals):
    """-1, 0, +1 from the first and last finite value; the crudest honest reading of a series."""
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    if len(v) < 2:
        return 0
    d = v[-1] - v[0]
    return 0 if abs(d) < 1e-12 else (1 if d > 0 else -1)


def opposite(rows, arm, coord, tag=None):
    """Do the clean and the polluted series disagree about the direction of travel?

    The clean series is the round-to-round distance, the polluted one the round-r fit diagnostic;
    both are read oldest-first. A pair of opposite non-zero trends is the thing section 3 reports.
    """
    sel = [r for r in rows if r["arm"] == arm and r["coord"] == coord
           and (tag is None or r["tag"] == tag)]
    sel.sort(key=lambda r: r["ra"])
    tc = trend([r["ln_mean"] for r in sel])
    tp = trend([r["polluted_ln_median"] for r in sel])
    return tc, tp, (tc * tp == -1)


def print_table(rows, arms=None, tag=None):
    arms = arms or sorted({r["arm"] for r in rows})
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm and (tag is None or r["tag"] == tag)]
        if not sel:
            continue
        print(f"\n== {arm} (clean instrument | x floor | polluted)")
        print(f"{'coord':10s} {'r->r+1':>7s} {'step kJ/mol':>11s} {'TV':>7s} {'ln_mean':>8s} "
              f"{'x floor':>8s} {'ln_max':>7s} {'dq50/sig':>9s} {'dq95/sig':>9s} {'| polluted':>11s}")
        for r in sorted(sel, key=lambda x: (x["coord"], x["ra"])):
            pol = r["polluted_ln_median"]
            pols = f"{pol:11.3f}" if isinstance(pol, (int, float)) else "        n/a"
            fl = SAME_FIELD_FLOOR["ln_mean"].get(r["coord"])
            ratio = f"{r['ln_mean'] / fl:8.2f}" if fl else "     n/a"
            print(f"{r['coord']:10s} {str(r['ra'])+'->'+str(r['rb']):>7s} "
                  f"{r['step_std']:11.3f} {r['tv']:7.3f} {r['ln_mean']:8.3f} {ratio} "
                  f"{r['ln_max']:7.2f} {r['dq50_sig']:9.3f} {r['dq95_sig']:9.3f} {pols}")


def decomposed_report(sets, reps=0):
    """Per-chain spread and block jackknife for every stored pair that carries the decomposition.

    TWO QUESTIONS, ONE SECTION. Per chain: is the pooled distance one number or seven, and does the
    pooled value scale like sqrt(chains) -- which is what section 4.2 assumed when it used a 150k
    block floor to bound a 1.2M window comparison. Per block: what does a delete-one jackknife at
    full window size say, which needs no second trajectory at all. Both are printed only where the
    run stored them; a pooled-only ensembles file prints nothing rather than a guess.
    """
    for tag, cfg in sets.items():
        z = np.load(cfg["ensembles"])
        ens = load_ensembles(cfg["ensembles"])
        for arm in sorted({k.split("_r")[0] for k in ens}):
            for ra, rb in pairs_for(ens, arm):
                for c in COORDS:
                    ka, kb = f"{arm}_r{ra}", f"{arm}_r{rb}"
                    if c not in ens.get(ka, {}) or c not in ens.get(kb, {}):
                        continue
                    ca, cb = chains_of(z, ka, c), chains_of(z, kb, c)
                    ba, bb = blocks_of(z, ka, c), blocks_of(z, kb, c)
                    if not ca and ba is None:
                        continue
                    bits = []
                    if ca and cb and sorted(ca) == sorted(cb):
                        centre = np.arange(len(next(iter(ca.values()))), dtype=float)
                        per = [metrics(ca[i], cb[i], centre, reps=reps)["ln_mean"]
                               for i in sorted(ca)]
                        pooled = metrics(ens[ka][c], ens[kb][c], centre, reps=0)["ln_mean"]
                        bits.append(
                            f"per-chain n={len(per)} ln_mean {min(per):.4f}/{float(np.median(per)):.4f}"
                            f"/{max(per):.4f}, pooled {pooled:.4f} "
                            f"(x sqrt(n) = {pooled * np.sqrt(len(per)):.4f})")
                    if ba is not None and bb is not None:
                        centre = np.arange(ba.shape[1], dtype=float)
                        jk_ln, jk_tv = [], []
                        for b in range(ba.shape[0]):
                            d = metrics(np.delete(ba, b, axis=0).sum(axis=0),
                                        np.delete(bb, b, axis=0).sum(axis=0), centre, reps=0)
                            jk_ln.append(d["ln_mean"])
                            jk_tv.append(d["tv"])
                        j = jackknife(jk_ln)
                        bits.append(f"block jackknife n={j['n']} ln_mean "
                                    f"{j['mean']:.4f} +- {j['std_jackknife']:.4f}, "
                                    f"TV {jackknife(jk_tv)['std_jackknife']:.4f}")
                    if bits:
                        print(f"   {tag:6s} {arm:5s} {c:9s} {ra}->{rb}: " + " | ".join(bits))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default="run1,c2stab")
    ap.add_argument("--reps", type=int, default=200, help="Poisson replicates for the floor")
    ap.add_argument("--out", default=str(OUT / "instrument.json"))
    ap.add_argument("--floor-step-max", type=float, default=FLOOR_STEP_MAX)
    args = ap.parse_args(argv)
    sets = {k: DATASETS[k] for k in args.datasets.split(",") if k in DATASETS}

    print("== exact zero: the same field and the same seed, twice")
    z = load_ensembles(DATASETS["run1"]["ensembles"])
    for c in COORDS:
        same = bool((z["C0_r1"][c] == z["C1_r1"][c]).all())
        print(f"   C0_r1 vs C1_r1 {c:9s} identical arrays: {same}")
    ref = OUT / "ensembles_c2stab_refused.npz"
    if ref.exists():
        zr = load_ensembles(ref)
        zs = load_ensembles(DATASETS["c2stab"]["ensembles"])
        for c in COORDS:
            if "C2s8_r1" in zr and "C2s8_r1" in zs:
                same = bool((zr["C2s8_r1"][c] == zs["C2s8_r1"][c]).all())
                print(f"   refused C2s8_r1 vs this run's C2s8_r1 {c:9s} identical: {same}"
                      f"   (the fit code changed between the two)")

    rows = analyse(sets, step_max=args.floor_step_max, reps=args.reps)
    fl = floor_rows(rows, args.floor_step_max)
    print(f"\n== floor: adjacent pairs whose injected step was <= {args.floor_step_max:g} kJ/mol")
    print(f"{'coord':10s} {'n':>3s} {'ln_mean min':>12s} {'ln_mean med':>12s} {'TV min':>8s} "
          f"{'TV med':>8s} {'poisson p95 ln':>15s} {'poisson p95 TV':>15s}")
    for c, f in fl.items():
        def s(v, w=12, p=3):
            return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else "n/a".rjust(w)
        print(f"{c:10s} {f['n']:3d} {s(f['ln_mean_min'])} {s(f['ln_mean_median'])} "
              f"{s(f['tv_min'], 8)} {s(f['tv_median'], 8)} "
              f"{s(f['poisson_p95_ln_mean'], 15)} {s(f['poisson_p95_tv'], 15)}")
    for c, f in fl.items():
        print(f"   {c}: members {f['members']}")

    print_table(rows)
    print("\n== decomposed: per-chain spread and block jackknife, where the run stored them")
    decomposed_report(sets, reps=0)
    print("\n== do the two instruments agree about the direction of travel?")
    for tag in sorted({r["tag"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["tag"] == tag}):
            for c in COORDS:
                tc, tp, opp = opposite(rows, arm, c, tag)
                if tc == 0 and tp == 0:
                    continue
                print(f"   {tag:6s} {arm:5s} {c:9s} clean {'+' if tc>0 else '-' if tc<0 else '0'}"
                      f"  polluted {'+' if tp>0 else '-' if tp<0 else '0'}"
                      f"   {'OPPOSITE' if opp else 'same'}")

    Path(args.out).write_text(json.dumps({"rows": rows, "floor": fl}, indent=1, default=float),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
