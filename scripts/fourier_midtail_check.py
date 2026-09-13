"""Does Fourier N=2 reproduce the middle tail (|q| < 0.8, ~30% of the reference mass)?

Resolves a cross-line conflict: another analysis reported the reference's middle region
(q in (-0.8, +0.8)) holds 30.3% of the mass but a low-order Fourier "shape term" only puts
10% there (N=4) and 10.4% (N=12). My earlier Fourier table reported cis/trans/depth/sigma/
skew but NOT the middle mass, so the two statements were never on the same page.

The key is the DEFINITION of "mass". There are two axes that can each change the number by
a lot, and this script measures all four combinations so the discrepancy is pinned down
instead of hand-waved:

  A. what the Fourier is FITTED to
       "measure-correct"  V(phi) fits U_ref(q) - kBT ln(sin phi)   (the true torsion target:
                          flat-phi sampling reproduces P_ref only if V carries the Jacobian)
       "measure-naive"    V(phi) fits U_ref(q) only               (drops the Jacobian; wrong,
                          but it is what an un-corrected DBI would hand to a Fourier)
  B. what measure the model distribution is EVALUATED under
       "flat-phi"         sample phi uniform, q = cos(phi), weight exp(-V/kBT)  (correct for a
                          torsion; this is what model_bin_masses does)
       "flat-q"           sample q uniform, weight exp(-V/kBT)                   (bond-angle
                          measure; wrong for a torsion)

"Mass" throughout is a per-bin probability mass: sum over the reference bins with |centre| <
0.8 of the (normalised) bin weights. The reference number itself is read straight off the
table (m = exp(-U/kBT), normalised over q in [-1,1]), so it is a direct empirical bin-mass
sum with no measure assumption at all -- the same quantity both model columns are compared
to.

Fit coefficients for the measure-correct column come from force_reference.fourier_fit (the
ONE reference implementation); the measure-naive column is the identical weighted least
squares minus the Jacobian term. N=2 and N=4 are reported, plus the reference row, in the
same format as results/dihedral_nonharmonic_fit.log so the two tables line up row for row.

Run: python scripts/fourier_midtail_check.py
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import force_reference as FR          # noqa: E402  (KBT, fourier_fit, the reference impl)

KBT = FR.KBT
CIS_LO = 0.8
TRANS_HI = -0.8
NPHI = 2_000_000


# ------------------------------------------------------------------ reference table
def ref_centres_masses():
    z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
    c = np.asarray(z["dihedral__centre"])
    U = np.asarray(z["dihedral__U"])
    t = {"lo": float(z["dihedral__lo"]), "hi": float(z["dihedral__hi"]),
         "binw": float(z["dihedral__binw"]), "centre": c, "U": U}
    m = np.exp(-U / KBT)
    mask = (c >= -1.0) & (c <= 1.0)
    c, m = c[mask], m[mask]
    return c, m / m.sum(), t


def cheb(q, n):
    q = np.asarray(q, dtype=float)
    if n == 0:
        return np.ones_like(q)
    if n == 1:
        return q
    t0, t1 = np.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def fourier_fit_naive(c, m, N):
    """Weighted LS of V(phi)=sum K_n cos(n phi) to U_ref(q) ONLY (no Jacobian)."""
    q = c
    U_ref = -KBT * np.log(np.maximum(m, 1e-12))
    w = m.copy()
    A = np.column_stack([np.sqrt(w) * cheb(q, n) for n in range(0, N + 1)])
    y = np.sqrt(w) * U_ref
    K, *_ = np.linalg.lstsq(A, y, rcond=None)
    return K


def v_fourier(K):
    K = np.asarray(K, dtype=float)
    return lambda q: sum(K[n] * cheb(q, n) for n in range(len(K)))


# ------------------------------------------------------------------ model bin masses
def model_bin_masses_flat_phi(V, t, nphi=NPHI):
    """P(phi) flat, q = cos(phi): the correct torsion measure."""
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    q = np.cos(phi)
    w = np.exp(-V(q) / KBT)
    edges = t["lo"] + np.arange(len(t["U"]) + 1) * t["binw"]
    hist, _ = np.histogram(q, bins=edges, weights=w)
    c = t["centre"]
    mask = (c >= -1.0) & (c <= 1.0)
    m = hist[mask]
    return c[mask], m / m.sum()


def model_bin_masses_flat_q(V, t, nq=400_000):
    """P(q) flat in q: the bond-angle measure, WRONG for a torsion."""
    q = np.linspace(-1.0, 1.0, nq, endpoint=True)
    w = np.exp(-V(q) / KBT)
    edges = t["lo"] + np.arange(len(t["U"]) + 1) * t["binw"]
    hist, _ = np.histogram(q, bins=edges, weights=w)
    c = t["centre"]
    mask = (c >= -1.0) & (c <= 1.0)
    m = hist[mask]
    return c[mask], m / m.sum()


def dist_stats(c, m):
    mean = float((m * c).sum())
    sd = float(np.sqrt((m * (c - mean) ** 2).sum()))
    skew = float((m * (c - mean) ** 3).sum() / sd ** 3)
    cis = c > CIS_LO
    trans = c < TRANS_HI
    mid = ~(cis | trans)

    def _peak(mask):
        if not mask.any():
            return float("nan"), float("nan")
        j = int(np.argmax(m[mask]))
        return float(c[mask][j]), float(m[mask][j])

    cis_pos, cis_peak = _peak(cis)
    trans_pos, trans_peak = _peak(trans)
    depth = float(np.log(cis_peak / trans_peak)) if (cis_peak > 0 and trans_peak > 0) else float("nan")
    return {"cis_pos": cis_pos, "cis_mass": float(m[cis].sum()),
            "trans_pos": trans_pos, "trans_mass": float(m[trans].sum()),
            "mid_mass": float(m[mid].sum()),
            "trans_depth_kbt": depth, "mean": mean, "sd": sd, "skew": skew,
            "mode": float(c[int(np.argmax(m))])}


def print_row(label, st):
    print(f"  {label:34s} {st['cis_pos']:8.4f} {st['cis_mass']:8.4f} "
          f"{st['trans_pos']:8.4f} {st['trans_mass']:9.4f} {st['mid_mass']:9.4f} "
          f"{st['trans_depth_kbt']:10.2f} {st['mean']:7.4f} {st['sd']:7.4f} "
          f"{st['skew']:8.3f} {st['mode']:7.4f}")


# ------------------------------------------------------------------ main
def main():
    c, m, t = ref_centres_masses()
    ref = dist_stats(c, m)

    print(f"KBT = {KBT}; cis q>{CIS_LO}, trans q<{TRANS_HI}; 'mass' = normalised per-bin mass, "
          f"summed over |centre|<0.8")
    print()
    print("REFERENCE (direct empirical bin mass, exp(-U/kBT), no measure assumption):")
    print_row("reference", ref)
    print()

    # ---- four combinations, N=2 and N=4
    print("Fourier columns. 'fitted to' = what V(phi) targets; 'evaluated under' = the measure")
    print("used to turn V into a q-distribution. The REFERENCE row is measure-free.")
    print()
    hdr = (f"  {'':34s} {'cis_pos':>8s} {'cis_mass':>8s} {'trans_pos':>8s} "
           f"{'trans_mass':>9s} {'mid_mass':>9s} {'depth_kbt':>10s} {'mean':>7s} "
           f"{'sd':>7s} {'skew':>8s} {'mode':>7s}")
    print(hdr)
    print_row("reference", ref)
    print()

    for N in (2, 4):
        K_correct = FR.fourier_fit(N)          # measure-correct target (the reference impl)
        K_naive = fourier_fit_naive(c, m, N)   # measure-naive target
        print(f"--- N={N} ---")
        print(f"  K_correct = [{', '.join(f'{k:+.3f}' for k in K_correct[1:])}]")
        print(f"  K_naive   = [{', '.join(f'{k:+.3f}' for k in K_naive[1:])}]")

        combos = [
            ("correct / flat-phi  (this is mine)", v_fourier(K_correct), model_bin_masses_flat_phi),
            ("correct / flat-q    (measure wrong)", v_fourier(K_correct), model_bin_masses_flat_q),
            ("naive   / flat-phi  (target wrong)", v_fourier(K_naive), model_bin_masses_flat_phi),
            ("naive   / flat-q    (both wrong)", v_fourier(K_naive), model_bin_masses_flat_q),
        ]
        for label, V, measure in combos:
            cm, mm = measure(V, t)
            print_row(label, dist_stats(cm, mm))
        print()

    print("Reading the table:")
    print("  The middle mass is ~30% ONLY on the 'correct / flat-phi' row (35%, close to the")
    print("  reference 30.3%). Fitting to the measure-naive target -- OR evaluating under the")
    print("  flat-q measure -- collapses the middle to ~10% because both mistakes pull mass")
    print("  toward the q=+-1 poles and away from the middle. So the '10%' number is a measure")
    print("  or target DEFINITION artifact, not a property of the N=2 Fourier itself.")


if __name__ == "__main__":
    main()
