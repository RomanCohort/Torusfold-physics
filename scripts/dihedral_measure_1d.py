"""The single-coordinate ("1-D") equilibrium of the angle and dihedral restraints, done right.

The IBI pipeline reports a "1-D sigma" for each coordinate as sqrt(kBT/k), and because the
shipped stiffness is k = kBT/sigma_ref^2, that 1-D sigma equals the reference sigma "by
construction" (see scripts/ibi_round0.py, the block headed "The single-coordinate
prediction"). The claim underneath is that a harmonic restraint E = 0.5*k*(q - c)^2 on one
coordinate q has equilibrium variance kBT/k.

That claim needs a flat measure dq. It holds for a BOND ANGLE (the angle between two bond
vectors carries solid-angle measure sin(theta) dtheta = d(cos theta), i.e. flat in cos). It
does NOT hold for a TORSION: the dihedral is an angle on the circle with measure dphi (flat in
phi), so the projection onto q = cos(phi) picks up the Jacobian 1/sqrt(1 - q^2), which is
singular at q = +-1. The dihedral target DIH_PPPP = acos(0.975) = 12.84 deg sits 0.224 rad
from the phi = 0 boundary, so the well is one-sided: ~5.7 kBT deep toward phi = pi and
essentially flat toward phi = 0. The equilibrium of cos(phi) is therefore shifted and
narrowed relative to the Gaussian that sqrt(kBT/k) describes.

This script measures that effect two independent ways:
  1. quadrature: the exact equilibrium moment integral under the correct measure, and
  2. a 1-D overdamped Langevin trajectory in the coordinate itself (phi for the dihedral),
     which is exactly the "single independent coordinate" the 1-D prediction idealises.

It also inverts the problem: what k would reproduce sigma_ref under the correct measure.

The dihedral is the point; the angle is included because it shows the SAME defect class
(truncation of the cos-domain at cos = -1) but a DIFFERENT cause (flat-cos measure is correct,
so the truncation is the whole story there). Both say the "1-D = ref by construction" number
is an upper bound that a shallow, one-sided well near a cos-boundary cannot actually reach.

Run: python scripts/dihedral_measure_1d.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402  (KBT lives here)
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

KBT = B.KBT

# Simulated sigmas, measured in results/E1_stationarity.log (1L2X, 40-200 ps, 8 replicas):
#   angle     0.2685    dihedral 0.4165
SIM = {"angle": 0.2685, "dihedral": 0.4165}


def quadrature_flat_cos(k, c):
    """Equilibrium moments of q = cos over the flat-cos measure (bond angle). q in [-1, 1]."""
    q = np.linspace(-1.0, 1.0, 4_000_001)
    w = np.exp(-0.5 * k * (q - c) ** 2 / KBT)
    Z = w.sum()
    m = (w * q).sum() / Z
    s = np.sqrt((w * (q - m) ** 2).sum() / Z)
    return float(m), float(s)


def quadrature_flat_phi(k, c):
    """Equilibrium moments of q = cos(phi) over the flat-phi measure (torsion). phi in [0, 2pi)."""
    phi = np.linspace(0.0, 2.0 * np.pi, 4_000_001)
    q = np.cos(phi)
    w = np.exp(-0.5 * k * (q - c) ** 2 / KBT)
    Z = w.sum()
    m = (w * q).sum() / Z
    s = np.sqrt((w * (q - m) ** 2).sum() / Z)
    return float(m), float(s)


def langevin_dihedral(k, c, n_steps=2_000_000, dt=0.002, seed=20260218):
    """Overdamped Langevin on phi in [0, 2pi) under U = 0.5*k*(cos phi - c)^2.

    Samples exp(-U/kBT) dphi, the measure a single independent torsion carries. gamma = 1,
    so D = kBT. Returns the mean and sigma of cos(phi) over the second half of the run.
    """
    rng = np.random.default_rng(seed)
    phi = 1.0  # arbitrary start; the well is shallow and mixing is fast
    burn = n_steps // 2
    qs = np.empty(n_steps - burn)
    j = 0
    for i in range(n_steps):
        # dU/dphi = -k*(cos phi - c)*sin phi, so -dt*dU/dphi = +dt*k*(cos-c)*sin
        phi += dt * k * (np.cos(phi) - c) * np.sin(phi) + np.sqrt(2.0 * KBT * dt) * rng.standard_normal()
        phi = np.mod(phi, 2.0 * np.pi)
        if i >= burn:
            qs[j] = np.cos(phi)
            j += 1
    return float(qs.mean()), float(qs.std())


def k_for_sigma(k, c, target_sigma, measure):
    """The k whose equilibrium sigma equals target_sigma, by bisection on the measure."""
    lo, hi = 1e-3, 1e4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = measure(mid, c)[1]
        if s > target_sigma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    rows = [
        ("angle", C.K_ANGLE, np.cos(C.ANGLE_PPP)),
        ("dihedral", C.K_DIH, np.cos(C.DIH_PPPP)),
    ]
    z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")

    print(f"KBT = {KBT:.4f} kJ/mol")
    print()
    print("equilibrium sigma of cos, one coordinate alone, under each candidate measure:")
    print(f"{'coord':9s} {'k':>6s} {'target c':>9s} {'sqrt(kBT/k)':>12s} "
          f"{'flat-cos':>9s} {'flat-phi':>9s} {'ref sigma':>10s} {'sim sigma':>10s}")
    print("-" * 80)
    for name, k, c in rows:
        naive = np.sqrt(KBT / k)
        mc, sc = quadrature_flat_cos(k, c)
        mp, sp = quadrature_flat_phi(k, c)
        ref = float(z[f"{name}__sigma"])
        print(f"{name:9s} {k:6.1f} {c:9.3f} {naive:12.4f} {sc:9.4f} {sp:9.4f} "
              f"{ref:10.4f} {SIM[name]:10.4f}")

    print()
    print("dihedral, the point of this script:")
    k = C.K_DIH
    c = np.cos(C.DIH_PPPP)
    ref = float(z["dihedral__sigma"])
    m_q, s_q = quadrature_flat_phi(k, c)
    print(f"  k={k}  target c={c:.3f}  (phi0={np.arccos(c):.4f} rad = {np.degrees(np.arccos(c)):.2f} deg)")
    print(f"  well depth toward phi=pi : {0.5 * k * (c + 1.0) ** 2:.2f} kJ/mol = "
          f"{0.5 * k * (c + 1.0) ** 2 / KBT:.2f} kBT")
    print(f"  well depth toward phi=0  : {0.5 * k * (1.0 - c) ** 2:.4f} kJ/mol = "
          f"{0.5 * k * (1.0 - c) ** 2 / KBT:.4f} kBT")
    print(f"  sqrt(kBT/k)               : {np.sqrt(KBT / k):.4f}  <- what ibi_round0 uses as 1-D sigma")
    print(f"  quadrature (flat phi)     : mean cos = {m_q:.4f}, sigma = {s_q:.4f}")
    ml, sl = langevin_dihedral(k, c)
    print(f"  1-D Langevin (flat phi)   : mean cos = {ml:.4f}, sigma = {sl:.4f}")
    print(f"  ref sigma                 : {ref:.4f}")
    print(f"  sim sigma (E1)            : {SIM['dihedral']:.4f}")
    print(f"  => correct 1-D sim/ref    : {s_q / ref:.3f}  (the 'by construction' number was 1.000)")
    print(f"  => measured sim vs 1-D    : {SIM['dihedral'] / s_q:.3f}  (coupling widens, not narrows)")
    print()
    k_fix = k_for_sigma(k, c, ref, quadrature_flat_phi)
    print(f"  k that WOULD reproduce ref sigma under the flat-phi measure: {k_fix:.3f} "
          f"(shipped {k:.1f})")

    print()
    print("angle, for contrast (flat-cos measure is the right one here, so the truncation at")
    print("cos = -1 is the whole story):")
    ka = C.K_ANGLE
    ca = np.cos(C.ANGLE_PPP)
    ra = float(z["angle__sigma"])
    ma_q, sa_q = quadrature_flat_cos(ka, ca)
    print(f"  k={ka}  target c={ca:.3f}  sqrt(kBT/k)={np.sqrt(KBT / ka):.4f}  "
          f"truncated-cos sigma={sa_q:.4f}  ref={ra:.4f}  sim={SIM['angle']:.4f}")
    print(f"  => correct 1-D sim/ref : {sa_q / ra:.3f}    measured sim vs 1-D : {SIM['angle'] / sa_q:.3f}")


if __name__ == "__main__":
    main()
