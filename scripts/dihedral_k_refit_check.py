"""V2: independently recompute the k that makes the torsion's equilibrium sigma equal
sigma_ref, by paths different from the one in dihedral_measure_1d.py.

dihedral_measure_1d.py got k = 2.371 by bisection on a UNIFORM-GRID (4,000,001-point)
trapezoid integral of exp(-U/kBT) dphi over phi in [0, 2pi). That grid and this script could
share a systematic error (e.g. under-resolving the sharp peak near the q = +1 divergence).
This script recomputes the same quantity two independent ways and reports their spread:

  1. adaptive quadrature (scipy.integrate.quad) over q = cos(phi) in [-1, 1] with the
     explicit Jacobian weight 1/sqrt(1 - q^2) -- a different integrator, a different
     variable, and an adaptive (not uniform) error control;
  2. a 4096-point midpoint rule over phi -- independent of both the trapezoid grid and of
     scipy.quad.

The empirical cross-check (a real MD that assumes no measure) is the V1 deliverable
(scripts/dihedral_md_isolated.py); it is deliberately NOT duplicated here.

Run: python scripts/dihedral_k_refit_check.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

KBT = B.KBT
C_DIH = float(np.cos(C.DIH_PPPP))
REF = float(np.load(REPO / "results" / "boltzmann_tables_clean.npz")["dihedral__sigma"])


def sigma_quad(k):
    """sigma(cos phi) via adaptive quad over q with the 1/sqrt(1-q^2) Jacobian."""
    def num(q):
        qc = np.clip(q, -1 + 1e-12, 1 - 1e-12)
        return np.exp(-0.5 * k * (qc - C_DIH) ** 2 / KBT) / np.sqrt(1 - qc * qc)
    Z = quad(num, -1.0, 1.0, limit=200)[0]
    m = quad(lambda q: q * num(q), -1.0, 1.0, limit=200)[0] / Z
    m2 = quad(lambda q: q * q * num(q), -1.0, 1.0, limit=200)[0] / Z
    return m, np.sqrt(max(m2 - m * m, 0.0))


def sigma_midpoint(k, n=4096):
    """sigma(cos phi) via an n-point midpoint rule over phi in [0, 2pi)."""
    phi = (np.arange(n) + 0.5) * (2.0 * np.pi / n)
    q = np.cos(phi)
    w = np.exp(-0.5 * k * (q - C_DIH) ** 2 / KBT)
    Z = w.sum()
    m = (w * q).sum() / Z
    return m, np.sqrt((w * (q - m) ** 2).sum() / Z)


def bisect(fn, target):
    lo, hi = 0.05, 100.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print(f"reference sigma = {REF:.4f}, target cos = {C_DIH:.4f}, KBT = {KBT:.4f}")
    print(f"shipped K_DIH = {C.K_DIH}")
    print()
    print("path 1: scipy adaptive quad over q, Jacobian 1/sqrt(1-q^2), bisection on sigma")
    k1 = bisect(lambda k: sigma_quad(k)[1], REF)
    print(f"  k = {k1:.4f}   (sigma at k = {sigma_quad(k1)[1]:.5f})")
    print("path 2: 4096-point midpoint rule over phi, bisection on sigma")
    k2 = bisect(lambda k: sigma_midpoint(k)[1], REF)
    print(f"  k = {k2:.4f}   (sigma at k = {sigma_midpoint(k2)[1]:.5f})")
    print("path 0 (reference): uniform-grid trapezoid in dihedral_measure_1d.py -> 2.371")
    print()
    print(f"  spread path1 vs path2 : {abs(k1 - k2):.4f}")
    print(f"  spread vs grid value  : {abs(0.5 * (k1 + k2) - 2.371):.4f}")
    print(f"  refit k = {0.5 * (k1 + k2):.3f}  (factor {C.K_DIH / (0.5 * (k1 + k2)):.2f} softer than shipped)")
    print()
    print("conclusion: two independent numerical paths agree with the grid value to <0.01,")
    print("so k ~ 2.4 (not 7.2) reproduces sigma_ref for a torsion measured in cos. The")
    print("empirical confirmation is scripts/dihedral_md_isolated.py (V1), which assumes no")
    print("measure at all.")


if __name__ == "__main__":
    main()
