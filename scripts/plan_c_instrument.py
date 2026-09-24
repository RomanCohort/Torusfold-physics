"""The clean instrument: what two SAMPLED ensembles say about each other, and its floor.

WHY THIS EXISTS, measured 2026-09-23 (docs/plan_c_c2_stabilization.md sections 3.4-3.5). The
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
        arm_round, coord = k.split("__", 1)
        out.setdefault(arm_round, {})[coord] = np.asarray(z[k], dtype=float)
    return out


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
        print(f"\n== {arm} (clean instrument | polluted)")
        print(f"{'coord':10s} {'r->r+1':>7s} {'step kJ/mol':>11s} {'TV':>7s} {'ln_mean':>8s} "
              f"{'ln_max':>7s} {'dq50/sig':>9s} {'dq95/sig':>9s} {'| polluted':>11s}")
        for r in sorted(sel, key=lambda x: (x["coord"], x["ra"])):
            pol = r["polluted_ln_median"]
            pols = f"{pol:11.3f}" if isinstance(pol, (int, float)) else "        n/a"
            print(f"{r['coord']:10s} {str(r['ra'])+'->'+str(r['rb']):>7s} "
                  f"{r['step_std']:11.3f} {r['tv']:7.3f} {r['ln_mean']:8.3f} {r['ln_max']:7.2f} "
                  f"{r['dq50_sig']:9.3f} {r['dq95_sig']:9.3f} {pols}")


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
