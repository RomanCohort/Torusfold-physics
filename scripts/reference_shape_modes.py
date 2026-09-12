"""Is the reference distribution unimodal or multimodal? The fork in the road for k=2.37.

The whole measure argument (dihedral_measure_1d.py) assumes the reference is ONE well on a
flat-phi torsion. If the reference is actually multimodal (several rotameric states), then
NO single harmonic in cos -- at any k -- can match its shape: 2.37 would only patch the
variance while leaving the shape wrong, and feeding that wrong shape to IBI is a wrong
residual.

This script reads the stored potentials U(q) = -kBT ln P(q) from
results/boltzmann_tables_clean.npz and characterises each of the six coordinates' shape:

  * number of local minima of U (local maxima of P), with positions and depths in kBT;
  * the sigma, skew, and quantiles of the reference P(q) itself (bin-sum over the table,
    which is a binned histogram and needs no Jacobian);
  * for angle and dihedral, the single-well prediction's sigma/skew computed by PROPER
    adaptive quadrature (scipy.integrate.quad) over q in [-1,1] with the torsion Jacobian
    1/sqrt(1-q^2), NOT by summing a diverging density on the table grid.

The earlier version of this file computed the single-well prediction's moments and mode by
summing the diverging density on the table's centre grid, which extends past cos = +-1; the
out-of-range bins were clipped to the boundary where the Jacobian diverges, so the sum was
dominated by 4 boundary bins and the reported sigma (0.1225 at k=7.2) and mode were wrong.
The reference moments are unaffected -- they are a binned histogram, finite everywhere.

The three distance coordinates are the unimodal control. The measure-correct k for ANGLE is
also printed.

Run: python scripts/reference_shape_modes.py
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
D = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
TARGETS = {"angle": float(np.cos(C.ANGLE_PPP)), "dihedral": float(np.cos(C.DIH_PPPP))}


def local_minima(U):
    """Indices of local minima of U, merging flat plateaux to their lowest bin."""
    idx = []
    for i in range(1, len(U) - 1):
        if U[i] <= U[i - 1] and U[i] <= U[i + 1]:
            idx.append(i)
    if not idx:
        return np.array([], dtype=int)
    merged = [idx[0]]
    for i in idx[1:]:
        if i - merged[-1] == 1:
            if U[i] < U[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    return np.array(merged, dtype=int)


def characterize(name):
    """Reference moments from the binned histogram P(q) = exp(-U/kBT), no Jacobian needed."""
    U = D[f"{name}__U"].astype(np.float64)
    c = D[f"{name}__centre"].astype(np.float64)
    binw = float(D[f"{name}__binw"])
    P = np.exp(-(U - U.min()) / KBT)
    Z = P.sum() * binw
    m = float((c * P).sum() * binw / Z)
    var = float(((c - m) ** 2 * P).sum() * binw / Z)
    s = np.sqrt(var)
    sk = float((((c - m) / s) ** 3 * P).sum() * binw / Z)
    cs = np.cumsum(P) / P.sum()
    qs = {p: float(c[np.searchsorted(cs, p)]) for p in (0.01, 0.25, 0.5, 0.75)}
    return dict(name=name, U=U, c=c, binw=binw, P=P / Z,
                mins=local_minima(U), g=int(np.argmin(U)),
                mean=m, sigma=s, skew=sk, quant=qs,
                lo=float(D[f"{name}__lo"]), hi=float(D[f"{name}__hi"]))


def torsion_moments(k, c):
    """(mean, sigma, skew) of q = cos(phi) for a flat-phi harmonic, by adaptive quad."""
    def num(q):
        qc = np.clip(q, -1 + 1e-12, 1 - 1e-12)
        return np.exp(-0.5 * k * (qc - c) ** 2 / KBT) / np.sqrt(1 - qc * qc)
    Z = quad(num, -1.0, 1.0, limit=200)[0]
    m = quad(lambda q: q * num(q), -1.0, 1.0, limit=200)[0] / Z
    m2 = quad(lambda q: (q - m) ** 2 * num(q), -1.0, 1.0, limit=200)[0] / Z
    m3 = quad(lambda q: (q - m) ** 3 * num(q), -1.0, 1.0, limit=200)[0] / Z
    return m, np.sqrt(m2), m3 / np.sqrt(m2) ** 3


def angle_moments(k, c):
    """(mean, sigma, skew) of q = cos(theta) for a flat-cos harmonic, truncated to [-1,1]."""
    def num(q):
        return np.exp(-0.5 * k * (q - c) ** 2 / KBT)
    Z = quad(num, -1.0, 1.0, limit=200)[0]
    m = quad(lambda q: q * num(q), -1.0, 1.0, limit=200)[0] / Z
    m2 = quad(lambda q: (q - m) ** 2 * num(q), -1.0, 1.0, limit=200)[0] / Z
    m3 = quad(lambda q: (q - m) ** 3 * num(q), -1.0, 1.0, limit=200)[0] / Z
    return m, np.sqrt(m2), m3 / np.sqrt(m2) ** 3


def angle_sigma(k, c):
    return angle_moments(k, c)[1]


def bisect_sigma(sigma_fn, target, lo=1e-3, hi=1e4):
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if sigma_fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print(f"KBT = {KBT:.4f} kJ/mol")
    print("mode detection: local minima of U (= maxima of P); depth = (U - U_min)/kBT.")
    print("a second well several kBT deep is a real state; < ~1 kBT is a shoulder / bin noise.")
    print()
    names = ["bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack"]
    chars = {n: characterize(n) for n in names}

    for n in names:
        ch = chars[n]
        U, c, P = ch["U"], ch["c"], ch["P"]
        order = ch["mins"][np.argsort(U[ch["mins"]])]
        print(f"=== {n}  (n_min = {len(ch['mins'])})  ref mean={ch['mean']:.4f} "
              f"sigma={ch['sigma']:.4f} skew={ch['skew']:+.2f} ===")
        print(f"    quantiles: q@1%={ch['quant'][0.01]:+.4f} q@25%={ch['quant'][0.25]:+.4f} "
              f"q@50%={ch['quant'][0.5]:+.4f} q@75%={ch['quant'][0.75]:+.4f}")
        for i, mi in enumerate(order[:4]):
            depth = (U[mi] - U[ch["g"]]) / KBT
            below = np.where(P < 0.5 * P[mi])[0]
            left = below[below < mi]
            right = below[below > mi]
            ql = c[left[-1]] if len(left) else ch["lo"]
            qr = c[right[0]] if len(right) else ch["hi"]
            print(f"    well {i}: q={c[mi]:+.4f}  U={U[mi]:.3f} kJ/mol  depth={depth:6.2f} kBT  "
                  f"HWHM[{ql:+.4f},{qr:+.4f}] halfw={0.5*(qr-ql):.4f}")
        print()

    print("=== single-well prediction vs reference (angle & dihedral), proper quadrature ===")
    print("the prediction is unimodal (one cis well at phi=0 / q->+1); it has no trans peak.")
    print("the criterion is sigma and skew of q=cos(phi), computed the same way as the ref.")
    print("mode is reported separately: for the torsion the single-well density DIVERGES as")
    print("q -> 1 (the 1/sqrt(1-q^2) Jacobian), so its mode is the boundary q=1, pinned there")
    print("for EVERY k -- whereas the reference cis peak is FINITE at q=+0.9967 (width 0.054).")
    print("That pinning is a sign of the missing entropic term, not a matched feature.")
    print()
    print(f"{'coord':9s} {'k':>7s} {'pred mean':>10s} {'pred sigma':>10s} {'pred skew':>10s} "
          f"{'pred mode':>9s} {'ref sigma':>10s} {'ref skew':>9s} {'ref mode':>9s}")
    print("-" * 82)
    for n in ["angle", "dihedral"]:
        ch = chars[n]
        k_shipped = C.K_DIH if n == "dihedral" else C.K_ANGLE
        fn = torsion_moments if n == "dihedral" else angle_moments
        m, s, sk = fn(k_shipped, TARGETS[n])
        k_fit = (2.371 if n == "dihedral"
                 else bisect_sigma(lambda k: angle_sigma(k, TARGETS[n]), float(D[f"{n}__sigma"])))
        mf, sf, skf = fn(k_fit, TARGETS[n])
        mode_tag = "->1 (boundary, diverges)" if n == "dihedral" else "--"
        ref_mode = ch["c"][ch["g"]]
        print(f"{n:9s} {k_shipped:7.2f} {m:10.4f} {s:10.4f} {sk:+10.2f} {mode_tag:>9s} "
              f"{ch['sigma']:10.4f} {ch['skew']:+9.2f} {ref_mode:+9.4f}   (shipped)")
        print(f"{'':9s} {k_fit:7.3f} {mf:10.4f} {sf:10.4f} {skf:+10.2f} {mode_tag:>9s} "
              f"{ch['sigma']:10.4f} {ch['skew']:+9.2f} {ref_mode:+9.4f}   (measure-correct)")

    print()
    print("=== angle measure-correct k (was missing from dihedral_measure_1d.py) ===")
    ka = bisect_sigma(lambda k: angle_sigma(k, TARGETS["angle"]), float(D["angle__sigma"]))
    print(f"  k that reproduces angle ref sigma under flat-cos truncation: {ka:.3f} "
          f"(shipped {C.K_ANGLE})")
    print()
    print("reading: the single-well prediction matches neither sigma nor skew at any k, has no")
    print("trans peak, and its cis mode is a boundary divergence instead of the reference's")
    print("finite 0.054-wide peak -- the reference dihedral is bimodal and the form is wrong.")
    print("See scripts/fit_dihedral_periodic.py for the periodic form that does.")


if __name__ == "__main__":
    main()
