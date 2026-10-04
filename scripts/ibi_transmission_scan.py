"""Which coordinates does the field actually move? Transmission, measured off the record.

The loop's whole premise is that a table-side change reaches the sampled marginal. Arm A showed that is
not always true (its dihedral table moved while the sampled dihedral did not), and the production
campaign's Part 9 numbers showed the same thing on the edge. This script measures the relation directly
from what is already on disk: for every round, the table that round sampled under (tables_r{r}.npz) and
the pooled histogram those chains produced (tasks_r{r}/counts__*.npz).

Reported per coordinate and round:
  * sigma of the table's own implied distribution, exp(-U/kBT) over the bin centres;
  * sigma of the pooled simulated distribution, on the same centres;
  * their ratio (sim/implied) -- 1.0 means the table is self-consistent with what the pool does;
  * outer-5-percent mass on both sides, and each against the fixed reference (refit_smooth5).

Then, per coordinate, a least-squares line through (implied sigma, sampled sigma) over the rounds of
EACH run separately, because that slope is the loop's effective gain: a table change of d_sigma reaches
the ensemble as slope * d_sigma. A slope near 0 means the coordinate is pinned by the coupling and no
table can move it; a slope near 1 means the table is the only thing setting it. The line's intercept is
what the pool would sample if the table were infinitely narrow, i.e. the model's own width floor.

What this is NOT: the slope is a two-point-per-round correlation over a single trajectory of table
states, not a designed experiment. It is quoted with its round count and its spread, and the projection
it implies ("implied sigma needed to reach the target") is an extrapolation of a nonlinear relation,
labelled as one.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import ibi_core as IC          # noqa: E402

REF = REPO / "results" / "refit_smooth5.npz"
COORDS = ("bb_bond", "angle", "dihedral", "stack")
KBT = 2.494
# The campaign record is resolved rather than hard-coded: it was moved out of results/ on
# 2026-10-01 and only partly copied back (see ibi_core.campaign_root). need_tasks=True because this
# scan sums per-chain histograms, which the stray copy does not have.
RUNS = {
    "campaign": (IC.campaign_root(need_tasks=True), range(0, 9)),
    "armA": (REPO / "results" / "ibi_armA", range(0, 2)),
}


def pooled_counts(round_dir, coord):
    tot = None
    n = 0
    for f in sorted(round_dir.glob("*.npz")):
        with np.load(f) as z:
            key = f"counts__{coord}"
            if key not in z:
                continue
            c = z[key].astype(np.float64)
        tot = c if tot is None else tot + c
        n += 1
    return tot, n


def sigma_of(p, centre):
    m = float((p * centre).sum())
    return float(np.sqrt(max((p * (centre - m) ** 2).sum(), 0.0)))


def edge_mass(p, frac=0.05):
    k = max(1, int(len(p) * frac))
    return float(p[:k].sum() + p[-k:].sum())


def implied(rec):
    U = np.asarray(rec["U"], dtype=float)
    p = np.exp(-(U - U.min()) / KBT)
    p = p / p.sum()
    centre = np.asarray(rec["lo"]) + np.asarray(rec["binw"]) * (np.arange(len(U)) + 0.5)
    return p, centre


def ls_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icept), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([slope, icept])
    ss = float(((y - pred) ** 2).sum())
    st = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(icept), (1.0 - ss / st) if st > 0 else float("nan")


def main():
    ref_tables = IC.load_tables(str(REF))
    ref = {}
    print("Reference (refit_smooth5) implied sigma / edge mass:", flush=True)
    for c in COORDS:
        p, _ = implied(ref_tables[c])
        ref[c] = (sigma_of(p, np.asarray(ref_tables[c]["centre"], float)), edge_mass(p))
        print("   %-9s sigma %.5f  edge %.4f" % (c, ref[c][0], ref[c][1]), flush=True)

    series = {}
    for name, (root, rounds) in RUNS.items():
        print("\n########## %s (%s) ##########" % (name, root.name), flush=True)
        print("%5s %-9s %10s %10s %8s %9s %9s %9s"
              % ("round", "coord", "implied", "sampled", "sim/imp", "tbl_edge", "smp_edge", "chain") , flush=True)
        for r in rounds:
            tf = root / f"tables_r{r}.npz"
            cd = root / f"tasks_r{r}"
            if not tf.exists() or not cd.exists():
                print("  round %d: missing %s" % (r, tf.name if not tf.exists() else cd.name), flush=True)
                continue
            tabs = IC.load_tables(str(tf))
            for c in COORDS:
                cnt, _n = pooled_counts(cd, c)
                if cnt is None:
                    continue
                centre = np.asarray(tabs[c]["centre"], dtype=float)
                p_sim = cnt / cnt.sum()
                s_sim = sigma_of(p_sim, centre)
                p_imp, cen_imp = implied(tabs[c])
                s_imp = sigma_of(p_imp, np.asarray(tabs[c]["centre"], float))
                series.setdefault((name, c), []).append((s_imp, s_sim))
                print("%5d %-9s %10.5f %10.5f %8.3f %9.4f %9.4f %9d"
                      % (r, c, s_imp, s_sim, s_sim / s_imp if s_imp else float("nan"),
                         edge_mass(p_imp), edge_mass(p_sim), int(cnt.sum())), flush=True)

    print("\n########## transmission (slope of sampled sigma against implied sigma) ##########", flush=True)
    print("%-10s %-9s %6s %8s %9s %9s %10s"
          % ("run", "coord", "points", "slope", "intercept", "R^2", "need_implied"), flush=True)
    for (name, c), pts in sorted(series.items()):
        if len(pts) < 2:
            continue
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        slope, icept, r2 = ls_line(x, y)
        tgt = ref[c][0]
        need = (tgt - icept) / slope if (slope == slope and abs(slope) > 1e-9) else float("nan")
        print("%-10s %-9s %6d %8.3f %9.4f %9.3f %10s"
              % (name, c, len(pts), slope, icept, r2,
                 ("%.4f" % need) if need == need else "n/a"), flush=True)
        print("           target sigma %.5f  -> the pooled width is %.5f, floor at implied=0 is %.5f"
              % (tgt, y[-1], icept), flush=True)


if __name__ == "__main__":
    main()
