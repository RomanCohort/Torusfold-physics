"""Phase 1: the (locality, ridge) frontier, offline, on the ensemble the arms actually cycled on.

Inputs are the ones the arms' round-4 fit saw: the round-4 ensemble (sampled under C2s8_r3) and the
field C2s8_r3 itself, so the step a family "wants" here is comparable with the step the arms wanted
at round 4 -- the cheb8 column IS that step, recomputed through plan_c_loop._chebyshev_fit_stable.

Two things get swept together because they are not independent. Locality (the number of basis
functions m) is the axis the project is about, and the RIDGE is what decides whether a given m is
usable at all: measured 2026-09-24, moment_correction's relative ridge (lambda = ridge_rel x
trace(A^T W A)/m) falls like 1/m for B-splines while the design's largest eigenvalue does not, so by
m=48 the system is effectively unregularised and the solve returns values past 1e100. The ridge is
therefore taken relative to the LARGEST EIGENVALUE here, which means the same thing at every m, and
its magnitude is swept rather than assumed.

Nothing here samples. Output: a table per coordinate and results/plan_c/basis_family.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ibi_bonded as I                # noqa: E402
import plan_c_basis as PB             # noqa: E402
import plan_c_loop as L               # noqa: E402

PC = REPO / "results" / "plan_c"
FIELD_DIR = PC / "fields" / "c2stab"
COORDS = ("bb_bond", "angle", "dihedral")
M_LIST = (8, 16, 32, 64, 128)
RIDGE_LIST = (1e-6, 1e-4, 1e-3, 1e-2)
# A variant is usable for phase 2 if it can express the target (residual below a third of kBT), its
# penalised system is numerically safe (condition under 1e8), and it carries the edge mass (the
# implied edge mass within 0.05 of the target's). kBT = 2.494 kJ/mol.
RESID_MAX = 0.35 * float(np.asarray(2.494))
COND_MAX = 1e8
EDGE_TOL = 0.05


def usable(row):
    return (row["resid_rms_kJ"] <= RESID_MAX and row["cond_fit"] <= COND_MAX
            and abs(row["edge_gap"]) <= EDGE_TOL)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PC / "basis_family.json"))
    args = ap.parse_args(argv)

    ens = np.load(PC / "ensembles_c2stab.npz")
    fields = {r: L.load_field(FIELD_DIR / f"C2s8_r{r}.npz") for r in (3, 4)}
    payload = {"source": {"ensemble": "C2s8_r4", "old_field": "C2s8_r3",
                          "note": "the round-4 fit's own inputs; cheb8 reproduces it",
                          "kBT": float(2.494), "resid_max": RESID_MAX, "cond_max": COND_MAX,
                          "edge_tol": EDGE_TOL},
               "coords": {}}

    for c in COORDS:
        tab_old = fields[3][c]
        counts = ens[f"C2s8_r4__{c}"]
        target = L._target_from_counts(counts)
        U_cheb, _d = L._chebyshev_fit_stable(tab_old, target, counts, 8)
        p = I.probability_from_counts(counts.astype(float))
        lo, hi = float(tab_old["lo"]), float(tab_old["hi"])
        edge_t = PB.edge_mass(p, tab_old["centre"], lo, hi)
        rows = {}

        print(f"\n== {c}: edge target mass {edge_t:.3f}; usable = resid <= {RESID_MAX:.2f} kJ/mol, "
              f"cond_fit <= {COND_MAX:.0e}, |edge gap| <= {EDGE_TOL}")
        print(f"{'family':10s} {'m':>4s} {'ridge':>8s} {'resid kJ':>9s} {'cond_fit':>10s} "
              f"{'step kJ':>8s} {'x cheb':>7s} {'edge fit':>9s} {'edge gap':>9s} {'max|dU|':>8s} {'use':>4s}")
        for m in M_LIST:
            for ridge in RIDGE_LIST:
                A, _desc = PB.design(f"bspline{m}", tab_old["centre"], lo, hi)
                U_new, diag = PB.fit(A, tab_old, counts, U_target=target, ridge_rel=ridge,
                                     ridge_form="eig")
                row = PB.diagnose(tab_old, counts, U_new, tab_old["U"], U_cheb=U_cheb, diag=diag)
                rows[f"bspline{m}_r{ridge:g}"] = row
                ok = usable(row)
                print(f"bspline   {m:4d} {ridge:8.0e} {row['resid_rms_kJ']:9.3f} "
                      f"{row['cond_fit']:10.3g} {row['step_rms_kJ']:8.3f} "
                      f"{row.get('corr_with_cheb', float('nan')):+7.3f} {row['edge_implied_new']:9.3f} "
                      f"{row['edge_gap']:+9.3f} {row['max_abs_dU']:8.2f} {'yes' if ok else '':>4s}")
        # the two references, at the ridge the arms use (trace/m) and at the arm's own inputs
        for name, kw in (("cheb8", {}), ("cheb16", {}), ("cheb32", {}), ("table", {})):
            A, desc = PB.design(name, tab_old["centre"], lo, hi)
            U_new, diag = PB.fit(A, tab_old, counts, U_target=target, ridge_rel=1e-1,
                                 ridge_form="trace")
            row = PB.diagnose(tab_old, counts, U_new, tab_old["U"], U_cheb=U_cheb, diag=diag)
            rows[name + "_armridge"] = row
            print(f"{desc.split()[0]:10s} {row['family_n_basis']:4d} {'trace/m':>8s} "
                  f"{row['resid_rms_kJ']:9.3f} {row['cond_fit']:10.3g} {row['step_rms_kJ']:8.3f} "
                  f"{row.get('corr_with_cheb', float('nan')):+7.3f} {row['edge_implied_new']:9.3f} "
                  f"{row['edge_gap']:+9.3f} {row['max_abs_dU']:8.2f} {'':>4s}")
        payload["coords"][c] = rows

        # what the frontier says: the best residual among the usable variants, and where the wall is
        us = {k: v for k, v in rows.items() if k.startswith("bspline") and usable(v)}
        if us:
            best = min(us.items(), key=lambda kv: kv[1]["resid_rms_kJ"])
            m_best = int(best[1]["family_n_basis"])
            print(f"   -> usable variants: {len(us)}; best residual {best[1]['resid_rms_kJ']:.3f} "
                  f"kJ/mol at m={m_best}, ridge={best[1]['ridge_rel']:g}")
        else:
            print("   -> NO usable B-spline variant: the ridge/locality frontier is empty here")

    Path(args.out).write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
