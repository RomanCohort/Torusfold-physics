"""V3: how many of the six bonded coordinates have a wrong 1-D baseline?

The pipeline's single-coordinate prediction is sigma_1D = sqrt(kBT/k). For a harmonic
E = 0.5*k*(q - c)^2 that is the equilibrium sigma ONLY when q carries a flat measure dq and
the domain is unbounded. The dihedral investigation (dihedral_measure_1d.py) showed that a
TORSION does not: the correct measure is flat in phi, and the projection onto q = cos(phi)
picks up the Jacobian 1/sqrt(1 - q^2), which diverges at the well's own target q = +0.975.
This script measures the same defect for all six coordinates in the IBI table, with one
uniform criterion: the ratio of the measure-correct equilibrium sigma to the sqrt(kBT/k)
baseline, and to the reference sigma.

The coordinates fall into three measure classes:
  * distances (bb_bond, intra_pc, intra_cn, stack): a 3-D bond length r carries the measure
    r^2 dr (solid angle integrated out). Correct equilibrium of r is
    P(r) ∝ r^2 exp(-0.5 k (r - r0)^2 / kBT). The well is narrow and centred far from 0, so
    the correction is ~(sigma/r0)^2 -- expected to be small, and we verify that.
  * bond angle (angle): the angle theta carries solid-angle measure sin(theta) dtheta, which
    is FLAT in cos(theta). So the cos measure is right, but the domain cos in [-1, 1]
    truncates the Gaussian when the target sits near cos = -1 (target -0.866, sigma 0.298).
  * torsion (dihedral): flat in phi, NOT flat in cos(phi); Jacobian 1/sqrt(1-q^2) diverges at
    q = +1, and the shipped target +0.975 sits 0.224 rad from that divergence, with a well
    that is 5.63 kBT deep toward phi = pi and 0.0009 kBT toward phi = 0.

stack has K_STACK = 0 (it is a derived coordinate, an exact identity of two bonds and the
angle), so it has no harmonic restraint and no sqrt(kBT/k) baseline; it is reported as N/A
and is excluded from the "how many baselines are wrong" count.

Run: python scripts/measure_all_coords_1d.py
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


def distance_sigma(k, r0):
    """Equilibrium sigma of a 3-D bond length under the r^2 dr measure, by adaptive quad."""
    def num(r):
        return r ** 2 * np.exp(-0.5 * k * (r - r0) ** 2 / KBT)
    def den(r):
        return num(r)
    Z = quad(den, 0.0, 3.0 * r0)[0]
    m = quad(lambda r: r * num(r), 0.0, 3.0 * r0)[0] / Z
    m2 = quad(lambda r: r * r * num(r), 0.0, 3.0 * r0)[0] / Z
    return m, np.sqrt(max(m2 - m * m, 0.0))


def angle_cos_sigma(k, c):
    """Equilibrium sigma of cos(theta) under the flat-cos measure, domain cos in [-1, 1]."""
    def num(q):
        return np.exp(-0.5 * k * (q - c) ** 2 / KBT)
    Z = quad(num, -1.0, 1.0)[0]
    m = quad(lambda q: q * num(q), -1.0, 1.0)[0] / Z
    m2 = quad(lambda q: q * q * num(q), -1.0, 1.0)[0] / Z
    return m, np.sqrt(max(m2 - m * m, 0.0))


def dihedral_cos_sigma(k, c):
    """Equilibrium sigma of cos(phi) under the flat-phi measure, via the Jacobian
    1/sqrt(1-q^2) over q in [-1, 1]. This is the same integral as quadrature_flat_phi in
    dihedral_measure_1d.py but integrated by a different (adaptive, weighted) method."""
    def num(q):
        qc = np.clip(q, -1 + 1e-12, 1 - 1e-12)
        return np.exp(-0.5 * k * (qc - c) ** 2 / KBT) / np.sqrt(1 - qc * qc)
    Z = quad(num, -1.0, 1.0)[0]
    m = quad(lambda q: q * num(q), -1.0, 1.0)[0] / Z
    m2 = quad(lambda q: q * q * num(q), -1.0, 1.0)[0] / Z
    return m, np.sqrt(max(m2 - m * m, 0.0))


def main():
    z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
    # simulated sigmas, results/E1_stationarity.log (1L2X, 40-200 ps)
    sim = {"bb_bond": 0.0527, "intra_pc": 0.0111, "intra_cn": 0.0083,
           "angle": 0.2685, "dihedral": 0.4165, "stack": 0.1188}

    specs = [
        ("bb_bond", "distance", C.K_BB, C.BOND_P_NEXT),
        ("intra_pc", "distance", C.K_INTRA_PC, C.BOND_P_C4),
        ("intra_cn", "distance", C.K_INTRA_CN, C.BOND_C4_N),
        ("angle", "angle-cos", C.K_ANGLE, np.cos(C.ANGLE_PPP)),
        ("dihedral", "torsion-cos", C.K_DIH, np.cos(C.DIH_PPPP)),
        ("stack", "distance", C.K_STACK, C.STACK_R0),
    ]

    print(f"KBT = {KBT:.4f} kJ/mol")
    print("measure-correct 1-D equilibrium vs the sqrt(kBT/k) baseline, all six coordinates:")
    print(f"{'coord':10s} {'measure':11s} {'k':>9s} {'sqrt(kBT/k)':>12s} "
          f"{'correct 1-D':>12s} {'ref sigma':>10s} {'1D/ref':>7s} {'correct/ref':>11s}")
    print("-" * 84)
    for name, meas, k, tgt in specs:
        naive = np.sqrt(KBT / k) if k > 0 else float("nan")
        ref = float(z[f"{name}__sigma"])
        if k == 0:
            m, s = float("nan"), float("nan")
        elif meas == "distance":
            m, s = distance_sigma(k, tgt)
        elif meas == "angle-cos":
            m, s = angle_cos_sigma(k, tgt)
        else:
            m, s = dihedral_cos_sigma(k, tgt)
        cr = s / ref if (s == s and ref > 0) else float("nan")
        nr = naive / ref if naive == naive else float("nan")
        print(f"{name:10s} {meas:11s} {k:9.1f} {naive:12.4f} {s:12.4f} "
              f"{ref:10.4f} {nr:7.3f} {cr:11.3f}")

    print()
    print("reading the table:")
    print("  sqrt(kBT/k)/ref ~ 1.000 for every restrained coordinate is BY CONSTRUCTION: the")
    print("  shipped k is kBT/sigma_ref^2, so that column is a tautology, not a prediction.")
    print("  'correct 1-D/ref' is the measure-correct single-coordinate ceiling. A value well")
    print("  below 1.000 means the baseline was wrong for that coordinate's measure/domain.")
    print()
    print("  distances: the r^2 dr correction is (sigma/r0)^2 -- sub-percent. The three bond")
    print("  lengths have correct/ref ~ 1.000 and are NOT affected.")
    print("  angle: flat-cos measure is correct, but cos in [-1,1] truncates the well near")
    print("  cos=-1 (target -0.866). Affected, mildly: correct/ref ~ 0.69.")
    print("  dihedral: torsion Jacobian 1/sqrt(1-q^2) diverges at the well's own target q=+0.975.")
    print("  Affected, strongly: correct/ref ~ 0.60.")
    print("  stack: K_STACK = 0, no restraint, no baseline -- N/A.")


if __name__ == "__main__":
    main()
