"""Why is the dihedral force as large as it is? Measured, not derived.

3q attributed it to the cos-restraint going singular as the dihedral approaches 0 or 180
degrees, on the grounds that a restraint quadratic in cos(phi) has gradient magnitude
proportional to k|cos(phi) - c|/(b sin phi). Working the chain rule out again says the
opposite: dE/dx = k(cos(phi)-c) * (-sin(phi)) * dphi/dx, and dphi/dx goes like 1/(|b| sin
theta) with theta the BOND angle, so sin(phi) sits in the numerator and cancels. The
singularity of dphi/dx is at collinear bonds, not at a planar dihedral.

Rather than build a functional-form change on a derivation I have not checked, this measures
which geometric quantity the large forces actually track. For every four-atom window it
records the exact force magnitude alongside the dihedral, both bond angles, and the three
bond lengths, then reports how strongly the force correlates with each candidate.

Run: python scripts/diagnose_dihedral_large_force.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

structs = B.load_structures(limit=10)
rows = []
for s in structs:
    L = len(s["pos"])
    if L < 6:
        continue
    p = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
    P = torch.arange(L) * 3
    # exact force, from the library's own current implementation
    _, F = C._dihedral_f(p, C.K_DIH, np.cos(C.DIH_PPPP))

    for i in range(L - 3):
        r = [p[0, P[i + k]] for k in range(4)]
        b0, b1, b2 = r[1] - r[0], r[2] - r[1], r[3] - r[2]
        n0 = torch.linalg.cross(b0, b1, dim=-1)
        n1 = torch.linalg.cross(b1, b2, dim=-1)
        n0n = float(torch.linalg.norm(n0))
        n1n = float(torch.linalg.norm(n1))
        if n0n < 1e-9 or n1n < 1e-9:
            continue
        cosd = float((n0 * n1).sum() / (n0n * n1n))
        phi = float(np.degrees(np.arccos(np.clip(cosd, -1, 1))))
        nb0, nb1, nb2 = (float(torch.linalg.norm(b)) for b in (b0, b1, b2))
        th1 = float(np.degrees(np.arccos(np.clip(
            float((b0 * b1).sum()) / (nb0 * nb1), -1, 1))))
        th2 = float(np.degrees(np.arccos(np.clip(
            float((b1 * b2).sum()) / (nb1 * nb2), -1, 1))))
        idx = [P[i + k] for k in range(4)]
        fmag = float(torch.linalg.norm(F[0, idx], dim=-1).max())
        rows.append((fmag, phi, cosd, th1, th2, nb0, nb1, nb2))

a = np.array(rows)
f, phi, cosd, th1, th2, nb0, nb1, nb2 = (a[:, i] for i in range(8))
print(f"{len(a)} dihedral windows from {len(structs)} structures")
print(f"|F| median {np.median(f):.1f}, p95 {np.percentile(f,95):.1f}, max {f.max():.1f} kJ/mol/nm")
print()

cands = {
    "dihedral |cos(phi)|": np.abs(cosd),
    "1/sin(dihedral)": 1.0 / np.maximum(np.sin(np.radians(phi)), 1e-6),
    "1/sin(theta1)": 1.0 / np.maximum(np.sin(np.radians(th1)), 1e-6),
    "1/sin(theta2)": 1.0 / np.maximum(np.sin(np.radians(th2)), 1e-6),
    "1/|b1|": 1.0 / nb1,
    "1/(|b1| sin theta1)": 1.0 / (nb1 * np.maximum(np.sin(np.radians(th1)), 1e-6)),
    "1/(|b1| sin theta2)": 1.0 / (nb1 * np.maximum(np.sin(np.radians(th2)), 1e-6)),
}
print(f"{'quantity':24s} {'corr with |F|':>14s} {'corr with log|F|':>17s}")
print("-" * 58)
for name, v in cands.items():
    c1 = float(np.corrcoef(v, f)[0, 1])
    c2 = float(np.corrcoef(np.log(np.maximum(v, 1e-9)), np.log(f))[0, 1])
    print(f"{name:24s} {c1:14.3f} {c2:17.3f}")
print()
print("a flat cos-restraint gradient would give |F| ~ k|cos(phi)-c|/(|b| sin theta), so")
print("the candidate that tracks |F| is the one whose correlation is near 1")
print()

k = C.K_DIH
worst = int(np.argmax(f))
print("the ten largest-force windows")
print(f"{'|F|':>9s} {'dihedral':>9s} {'|cos-c|':>8s} {'theta1':>7s} {'theta2':>7s} "
      f"{'|b0|':>6s} {'|b1|':>6s} {'|b2|':>6s}")
print("-" * 68)
for i in np.argsort(-f)[:10]:
    print(f"{f[i]:9.1f} {phi[i]:9.2f} {abs(cosd[i]-np.cos(C.DIH_PPPP)):8.3f} "
          f"{th1[i]:7.2f} {th2[i]:7.2f} {nb0[i]:6.3f} {nb1[i]:6.3f} {nb2[i]:6.3f}")
print()
simple = k * np.abs(cosd - np.cos(C.DIH_PPPP)) / (nb1 * np.maximum(np.sin(np.radians(th1)), 1e-6))
print(f"the simple estimate k|cos(phi)-c|/(|b1| sin theta1): median {np.median(simple):.1f}, "
      f"max {simple.max():.1f}")
print(f"actual:                                              median {np.median(f):.1f}, "
      f"max {f.max():.1f}")
print(f"ratio actual/estimate: median {np.median(f/simple):.2f}, "
      f"max {(f/simple).max():.2f}")
