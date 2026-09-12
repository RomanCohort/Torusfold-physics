"""What periodic form reproduces the bimodal dihedral reference? (the modeling step)

The reference dihedral distribution is bimodal: a deep cis well at q=cos(phi)=+1 (63% mass)
and a shallow trans well at q=-1 (6.5% mass, 2.25 kBT above cis), plus a broad intermediate
tail (30%). No single harmonic in cos can produce this, at any k.

The key physics (established in dihedral_measure_1d.py, confirmed by the isolated-torsion MD
in dihedral_md_isolated.py): the torsion's natural measure is FLAT in phi, so a potential
written as a function of q=cos(phi) samples

        P(q) ~ exp(-V(q)/kBT) / sqrt(1-q^2).

To make P(q) equal the reference exp(-U_ref(q)/kBT), the potential must be

        V(q) = U_ref(q) + (kBT/2) ln(1/(1-q^2))            [in q]
        V(phi) = U_ref(cos phi) - kBT ln(sin phi)          [in phi].

The second term is the measure/entropic correction: it is a REPULSION that diverges at
q = +-1 (phi = 0, pi), and it is what keeps the cis peak FINITE at the cos boundary instead
of diverging. A PURE low-order Fourier V(phi) = sum A_n cos(n phi) is the WRONG model class:
it is smooth at phi=0, so its P(q) diverges as 1/sqrt(1-q^2) at q=+-1 (this is exactly the
0.1225 artefact the earlier reference_shape_modes.py reported before it was fixed). The
entropic term -kBT ln(sin phi) has the exact Fourier series kBT ln2 + kBT sum_n cos(2n phi)/n,
so it needs ALL even harmonics -- a finite pure-Fourier fit can only approximate it.

This script therefore reports the SHAPE part of the potential, U_ref(cos phi), fitted as a
Chebyshev series in q (= a Fourier cosine series in phi):

        U_ref(q) ~ sum_{n=0}^N c_n T_n(q),   T_n(cos phi) = cos(n phi).

The full potential is then this series PLUS the exact entropic term. The script:
  1. fits the c_n by least squares over the uniform-q table bins, for N = 1..6;
  2. reconstructs P_N(q) = exp(-sum c_n T_n(q)/kBT) (the sin terms cancel exactly);
  3. reports P_N's sigma, skew, and peak structure (cis depth, trans depth) vs the reference,
     so the number of terms needed to capture the bimodality is visible.

It also reports the combined Fourier coefficients of V(phi) (shape c_n plus entropic kBT/n on
the even harmonics), which is what one would actually code.

Run: python scripts/fit_dihedral_periodic.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402

KBT = B.KBT
D = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
U = D["dihedral__U"].astype(np.float64)
q = D["dihedral__centre"].astype(np.float64)
binw = float(D["dihedral__binw"])
# clip to the physical cos domain (the table has 4 out-of-range bins at +-1.014, +-1.031)
ok = (q >= -1.0) & (q <= 1.0)
q_, U_ = q[ok], U[ok]


def cheb(n, x):
    """Chebyshev T_n(x), vectorised; tolerates a scalar x (from scipy quad)."""
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    t0 = np.ones_like(x)
    t1 = x + np.zeros_like(x)
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * x * t1 - t0
    return t1


def fit_cheb(N):
    """Least-squares c_n for U(q) ~ sum c_n T_n(q) over the physical bins."""
    A = np.column_stack([cheb(n, q_) for n in range(N + 1)])
    c, *_ = np.linalg.lstsq(A, U_, rcond=None)
    return c


def eval_cheb(c, x):
    return sum(c[n] * cheb(n, x) for n in range(len(c)))


def moments_from_shape(c):
    """(mean, sigma, skew) of P(q) = exp(-sum c_n T_n(q)/kBT), by quad over q in [-1,1]."""
    def num(x):
        return np.exp(-eval_cheb(c, x) / KBT)
    Z = quad(num, -1.0, 1.0, limit=200)[0]
    m = quad(lambda x: x * num(x), -1.0, 1.0, limit=200)[0] / Z
    m2 = quad(lambda x: (x - m) ** 2 * num(x), -1.0, 1.0, limit=200)[0] / Z
    m3 = quad(lambda x: (x - m) ** 3 * num(x), -1.0, 1.0, limit=200)[0] / Z
    return m, np.sqrt(m2), m3 / np.sqrt(m2) ** 3


def peaks_from_shape(c):
    """Minima of sum c_n T_n(q) on q in [-1,1], as (positions, depth-in-kBT relative to min).

    The cis and trans wells sit exactly at the boundaries q = +-1 (the entropic term is
    factored out, so U_ref is finite there), so the boundary values are added explicitly."""
    xs = np.linspace(-1.0, 1.0, 20001)
    v = eval_cheb(c, xs)
    idx = [i for i in range(1, len(xs) - 1) if v[i] <= v[i - 1] and v[i] <= v[i + 1]]
    if not idx:
        idx = []
    # merge adjacent
    merged = []
    for i in idx:
        if merged and i - merged[-1] <= 2:
            if v[i] < v[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    # add boundary candidates
    cand = [(float(xs[i]), float(v[i])) for i in merged]
    cand += [(-1.0, float(eval_cheb(c, -1.0))), (1.0, float(eval_cheb(c, 1.0)))]
    vmin = min(t[1] for t in cand)
    return sorted([(p, (vv - vmin) / KBT) for p, vv in cand], key=lambda t: t[1])


def main():
    # reference moments, bin-sum (finite histogram, no Jacobian)
    P = np.exp(-(U - U.min()) / KBT)
    Z = P.sum() * binw
    ref_mean = float((q * P).sum() * binw / Z)
    ref_var = float(((q - ref_mean) ** 2 * P).sum() * binw / Z)
    ref_sig = np.sqrt(ref_var)
    ref_skew = float((((q - ref_mean) / ref_sig) ** 3 * P).sum() * binw / Z)

    print(f"KBT = {KBT:.4f} kJ/mol")
    print(f"reference dihedral: mean={ref_mean:.4f} sigma={ref_sig:.4f} skew={ref_skew:+.2f}")
    print(f"reference peaks: cis q=+0.9967 (0 kBT, 63% mass), trans q=-0.9967 (2.25 kBT, 6.5%),")
    print(f"                  intermediate 30% mass in q in (-0.8, +0.8)")
    print()
    print("the exact periodic potential that reproduces this under flat-phi sampling:")
    print("    V(phi) = U_ref(cos phi) - kBT ln(sin phi)   [ = U_ref(q) + (kBT/2) ln(1/(1-q^2)) ]")
    print("the -kBT ln(sin phi) = kBT ln2 + kBT sum_{n>=1} cos(2n phi)/n is the entropic/measure")
    print("repulsion; it is what keeps the cis peak finite at the cos boundary. A pure finite")
    print("Fourier in phi cannot reproduce it (it diverges at q=+-1).")
    print()
    print("Chebyshev fit of the SHAPE part U_ref(cos phi), N terms, and the resulting P(q):")
    print(f"{'N':>3s} {'sigma':>8s} {'skew':>8s} {'cis depth':>10s} {'trans depth':>12s} "
          f"{'|sigma-ref|':>12s}")
    print("-" * 60)
    for N in range(1, 7):
        c = fit_cheb(N)
        m, s, sk = moments_from_shape(c)
        pk = peaks_from_shape(c)
        cis_d = trans_d = float("nan")
        # cis = deepest minimum near q=+0.7..1, trans = deepest near q=-1..-0.7
        pos = [p[0] for p in pk]
        if any(x > 0.7 for x in pos):
            cis_d = min(p[1] for p in pk if p[0] > 0.7)
        if any(x < -0.7 for x in pos):
            trans_d = min(p[1] for p in pk if p[0] < -0.7)
        print(f"{N:3d} {s:8.4f} {sk:+8.2f} {cis_d:10.2f} {trans_d:12.2f} {abs(s - ref_sig):12.4f}")

    print()
    print("coefficients c_n of U_ref(cos phi) = sum c_n cos(n phi), for N=4 (a good compromise):")
    c4 = fit_cheb(4)
    for n in range(5):
        print(f"    c{n} = {c4[n]:+.4f} kJ/mol   ({c4[n]/KBT:+.3f} kBT)")
    print()
    print("combined Fourier coefficients of V(phi) (shape + entropic), i.e. what to actually code")
    print("for a periodic potential V(phi) = A0 + sum A_n cos(n phi):")
    print("    -kBT ln(sin phi) = kBT ln2 + kBT * sum_{m>=1} cos(2m phi)/m, so even harmonic")
    print("    n = 2m carries entropic kBT/m (= 2 kBT/n); odd harmonics carry none.")
    for n in range(5):
        if n > 0 and n % 2 == 0:
            ent = KBT / (n // 2)     # cos(2m phi) -> kBT/m, m = n/2
        else:
            ent = 0.0
        An = c4[n] + (KBT * np.log(2.0) if n == 0 else ent)
        print(f"    A{n} = {An:+.4f} kJ/mol  (shape c{n}={c4[n]:+.4f}"
              + (f", entropic kBT/{n//2}={ent:+.4f}" if n > 0 and n % 2 == 0 else "") + ")")
    print()
    print("note: the entropic series has an infinite tail (kBT/m on harmonic 2m), so a finite")
    print("pure-Fourier potential is an approximation that gets better with more even terms;")
    print("the shape coefficients c_n alone already reproduce the cis/trans bimodality.")


if __name__ == "__main__":
    main()
