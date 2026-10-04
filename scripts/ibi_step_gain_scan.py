"""Two offline questions the record can answer without sampling: what gain makes a step neutral, and
what a damped step does to the edge mass.

WHY. Arm A applied two full-strength steps and both misfired (docs/ibi_loop_and_oxrna_findings.md
Part 10): the angle's self-consistent refit installed a table whose implied sigma is 0.685 against a
target of 0.322, and the dihedral's B-spline step took its own implied sigma from 0.625 to 0.519 while
its sampled width barely moved. Damping is the obvious response, and this script sizes it from the data
rather than by taste, by REPLAYING each rule at a ladder of gains on the same pooled histograms the
round used:

    angle     plan_c_basis.fit, Chebyshev K=8, trace-relative ridge 1e-1, support 1e-3, taper 2 decades,
              U_target = -kBT ln p_ens (the ensemble the round itself produced).
    dihedral  ibi_bonded.moment_correction, B-spline m=16, eigenvalue-relative ridge 1e-3,
              target = p_ref (the deposited reference; this rule never leaves it).

Two numbers come out of the ladder, and they are different numbers:

  * the NEUTRAL gain -- where the fitted table's own implied sigma equals the width of the distribution
    the fit was asked to reproduce. At gain 1 a rule may overshoot its own target; the ratio is the
    rule's error, and it is a property of the fit, not of the physics.
  * the EDGE GAP at each gain -- whether a smaller step preserves the outer-5-percent mass the basis
    project made the loop's order parameter.

VERIFIED AGAINST THE RECORD, not asserted: gain 1 here must reproduce the table arm A actually applied
(tables_r1), whose implied sigma is 0.68491 on the angle and 0.51862 on the dihedral. Both are printed
beside the replay's, and a mismatch would mean this script is not replaying the same rule.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B        # noqa: E402
import ibi_bonded as I              # noqa: E402
import ibi_core as IC               # noqa: E402
import plan_c_basis as PB           # noqa: E402

ARM = REPO / "results" / "ibi_armA"
REF = REPO / "results" / "refit_smooth5.npz"
KBT = 2.494
K_CHEB = 8          # IBI_LOOP_CORRECTION_K default
M_BSPLINE = 16      # IBI_LOOP_RULE_BSPLINE_M default
RIDGE_C2S = 1e-1    # IBI_LOOP_RULE_C2S_RIDGE default
RIDGE_BSP = 1e-3    # IBI_LOOP_RULE_RIDGE default
SUPPORT_FRAC_C2S = 1e-3
TAPER_DECADES_C2S = 2.0
GAINS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def implied(rec):
    U = np.asarray(rec["U"], dtype=float)
    p = np.exp(-(U - U.min()) / KBT)
    p = p / p.sum()
    return p, np.asarray(rec["centre"], dtype=float)


def sigma_of(p, centre):
    m = float((p * centre).sum())
    return float(np.sqrt(max((p * (centre - m) ** 2).sum(), 0.0)))


def edge_mass(p, frac=0.05):
    k = max(1, int(len(p) * frac))
    return float(p[:k].sum() + p[-k:].sum())


def pooled(round_dir, coord):
    cnt = n_tot = n_out = 0
    for f in sorted(round_dir.glob("*.npz")):
        with np.load(f) as z:
            cnt = cnt + z[f"counts__{coord}"].astype(np.float64)
            n_tot += int(z[f"n_total__{coord}"])
            n_out += int(z[f"n_outside__{coord}"])
    return cnt, n_tot, n_out


def main():
    ref_tables = IC.load_tables(str(REF))
    ref_p = {}
    for c in ("angle", "dihedral"):
        p, _ = implied(ref_tables[c])
        ref_p[c] = p
    print("reference implied sigma: angle %.5f  dihedral %.5f"
          % (sigma_of(ref_p["angle"], np.asarray(ref_tables["angle"]["centre"], float)),
             sigma_of(ref_p["dihedral"], np.asarray(ref_tables["dihedral"]["centre"], float))), flush=True)

    t0 = IC.load_tables(str(ARM / "tables_r0.npz"))
    t1 = IC.load_tables(str(ARM / "tables_r1.npz"))

    for coord in ("angle", "dihedral"):
        cnt, n_tot, n_out = pooled(ARM / "tasks_r0", coord)
        old = t0[coord]
        centre = np.asarray(old["centre"], dtype=float)
        lo, hi = float(old["lo"]), float(old["hi"])
        s_old = sigma_of(*implied(old))
        s_app = sigma_of(*implied(t1[coord]))
        print("\n===== %s =====" % coord, flush=True)
        print("   sampled %d observations, %.4f%% outside the support"
              % (n_tot, 100.0 * n_out / max(n_tot, 1)), flush=True)
        print("   table it sampled under: implied sigma %.5f   |   applied at gain 1: %.5f"
              % (s_old, s_app), flush=True)

        if coord == "angle":
            p_ens = I.probability_from_counts(np.asarray(cnt, dtype=float), pseudo=I.DEFAULT_PSEUDO)
            U_tgt = -B.KBT * np.log(np.clip(p_ens, 1e-300, None))
            U_tgt = U_tgt - U_tgt.min()
            p_tgt = p_ens / p_ens.sum()
            A, _x = I._chebyshev_design(centre, lo, hi, K_CHEB)
            print("   the rule's own target is p_ens, sigma %.5f  (so a faithful fit reproduces THAT)"
                  % sigma_of(p_tgt, centre), flush=True)

            def step(g):
                U_new, diag = PB.fit(A, old, cnt, U_target=U_tgt, ridge_rel=RIDGE_C2S,
                                     ridge_form="trace", gain=g,
                                     support_frac=SUPPORT_FRAC_C2S,
                                     taper_decades=TAPER_DECADES_C2S)
                U_new = np.asarray(U_new, dtype=float)
                p = np.exp(-(U_new - U_new.min()) / KBT)
                p = p / p.sum()
                return U_new, p, float(np.abs(U_new - np.asarray(old["U"], float)).max())

        else:
            def step(g):
                A = PB.design_bspline(centre, lo, hi, M_BSPLINE)
                res = I.moment_correction(old, cnt, n_tot, n_out, ref_p[coord], K=M_BSPLINE,
                                          gain=g, design=A, ridge_rel=RIDGE_BSP,
                                          ridge_form="eig")
                U_new = np.asarray(res.table["U"], dtype=float)
                p = np.exp(-(U_new - U_new.min()) / KBT)
                p = p / p.sum()
                return U_new, p, float(np.abs(res.dU).max())

        print("   %5s %12s %10s %10s %11s" % ("gain", "implied_sig", "edge", "max|dU|", "vs target"), flush=True)
        for g in GAINS:
            _U, p, dmax = step(g)
            s = sigma_of(p, centre)
            gap = edge_mass(p) - edge_mass(ref_p[coord])
            note = ""
            if coord == "angle":
                note = "sigma_ens %.3f" % (s / sigma_of(p_tgt, centre))
            else:
                note = "gap %+.4f" % gap
            print("   %5.2f %12.5f %10.4f %10.3f   %s" % (g, s, edge_mass(p), dmax, note), flush=True)


if __name__ == "__main__":
    main()
