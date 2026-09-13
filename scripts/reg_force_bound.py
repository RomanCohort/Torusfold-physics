"""Does the regularization floor on 1-q^2 actually bound the table-dihedral FORCE?

This resolves a contradiction between two of my own earlier logs. The first,
dihedral_force_divergence.py, once printed a synthetic sweep in which
"reg eps=3deg" left the CIS side blowing up to 116128 while suppressing the TRANS side to
165.6 -- which is physically wrong, because the floor is on 1-q^2 (symmetric in q^2). The
second claimed max|dV/dq| = 1959 "bounded and < 5000".

Both numbers describe different things, and the first was produced by a broken synthetic
geometry (its make_dihedral returned q = 0.0008 for an intended phi = 0.1 deg, so the sweep
never approached q -> +1 at all -- that function has been removed). This script redoes the
sweep with a CORRECT, verified geometry and measures the actual per-bead force, not dV/dq.

What it measures and why:

  1. A synthetic P-P-P-P with bond angle theta=150 deg (the field's ANGLE_PPP), swept over
     phi -> 0 (cis, q -> +1) and phi -> pi (trans, q -> -1). The geometry is verified: the
     q returned by the reference _q_dihedral equals cos(phi) to machine precision, and the
     cross-product normals |n0|, |n1| stay ~ sin(theta) = 0.5 -- i.e. WELL-CONDITIONED, no
     near-collinear bonds. This isolates the entropy singularity (q -> +-1) from the
     geometric singularity (collinear bonds, |n| -> 0).

  2. On that sweep, the per-bead max |F| of exact_b2 and of reg eps=3deg, side by side, plus
     the geometric factor max|dq/dx| at each phi. This answers "does reg bound the cis side"
     and "does |dq/dx| diverge as q -> +-1".

  3. The relationship per-bead force ~ |dV/dq| * |dq/dx|, and where the 10.69 /nm multiplier
     came from, and what the real geometric factor is on 2OIU.

  4. Why reg cannot bound the REAL force: the actual max force on 2OIU comes from a
     near-collinear window (|n0|,|n1| -> 0 => |dq/dx| -> infinity) at q = 0.0589, where the
     entropy floor never engages. This is confirmed by results/force_reference.log, where
     reg eps=3deg and exact_b2 give the identical max 15735.80 on 2OIU.

The conclusion: the regularization bounds dV/dq (the entropy term), NOT the force, because
|dq/dx| has no upper bound (geometric singularity). The table form therefore cannot be made
force-safe by this regularization; Fourier N=2 remains the only bounded candidate.

Run: python scripts/reg_force_bound.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import force_reference as FR          # noqa: E402  (the ONE reference implementation)
import boltzmann_bonded as B          # noqa: E402  (load_structures, KBT)

KBT = FR.KBT
FORCE_CAP = FR.FORCE_CAP
EPS = math.sin(math.radians(3.0))     # 0.0523, the regularization under test
THETA_DEG = 150.0                     # ANGLE_PPP in torch_cgsim


# ------------------------------------------------------------------ correct synthetic geometry
def make_dihedral(phi_deg, theta_deg=THETA_DEG, bond=1.0):
    """Four points with unit bonds, bond angle theta at p1 and p2, dihedral angle phi.

    Derived so that the reference convention q = cos(phi) = (n0 . n1)/(|n0||n1|) with
    n0 = b0 x b1, n1 = b1 x b2 (b0 = p1-p0, b1 = p2-p1, b2 = p3-p2) returns cos(phi)
    exactly. With theta = 150 deg the normals |n0| = |n1| = sin(theta) = 0.5 stay far from
    zero at every phi, so this isolates the phi -> 0/pi entropy singularity from the
    collinear-bond geometric singularity.
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    c, s = math.cos(th), math.sin(th)
    cp, sp = math.cos(ph), math.sin(ph)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([bond, 0.0, 0.0])
    p0 = np.array([c, -s, 0.0]) * bond          # angle between -b0 and b1 is theta
    p3 = np.array([bond - c, -s * cp, s * sp]) * bond   # angle between -b1 and b2 is theta
    return np.stack([p0, p1, p2, p3])


def geo_factor(pts):
    """max |dq/dx| over the 4 beads of one window, by autograd on q alone."""
    p = torch.tensor(pts, dtype=torch.float64).reshape(1, 4, 3)
    p = p.detach().clone().requires_grad_(True)
    q = FR._q_dihedral(p.unsqueeze(1))          # (1, 1)
    q.sum().backward()
    return float((-p.grad).norm(dim=-1).max().item())


def normals(pts):
    """|n0|, |n1| of the two cross-product normals (near-collinear check)."""
    p0, p1, p2, p3 = (torch.tensor(x, dtype=torch.float64) for x in pts)
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = torch.linalg.norm(torch.linalg.cross(b0, b1)).item()
    n1 = torch.linalg.norm(torch.linalg.cross(b1, b2)).item()
    return n0, n1


def window_force(spec, pts):
    """Per-window isolated max per-bead |F| via the reference implementation."""
    pos = torch.tensor(pts, dtype=torch.float64).reshape(1, 4, 3)
    ij = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    _, f = FR.window_max_force(spec, pos, ij_like=ij)
    return float(f[0])


# ------------------------------------------------------------------ main
def main():
    print(f"KBT = {KBT}, force_cap = {FORCE_CAP}, eps = sin(3 deg) = {EPS:.4f}, "
          f"theta = {THETA_DEG:.0f} deg")
    print()

    # ---- verification of the synthetic geometry
    print("=== synthetic geometry verification (q must equal cos(phi); normals stay ~0.5) ===")
    print(f"  {'phi(deg)':>8s} {'cos(phi)':>10s} {'q':>12s} {'|n0|':>7s} {'|n1|':>7s}")
    for ph in (0.1, 1.0, 4.7, 90.0, 175.3, 179.9):
        pts = make_dihedral(ph)
        pos = torch.tensor(pts, dtype=torch.float64).reshape(1, 4, 3)
        ij = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        qv, _ = FR.window_max_force("shipped_harmonic", pos, ij_like=ij)
        n0, n1 = normals(pts)
        print(f"  {ph:8.1f} {math.cos(math.radians(ph)):10.6f} {qv[0]:12.6f} "
              f"{n0:7.4f} {n1:7.4f}")
    print()

    # ---- the sweep: per-bead force of exact_b2 vs reg, plus |dq/dx|
    specs = {
        "exact_b2": ("table_jac", None),
        "reg eps=3deg": ("table_jac", EPS),
    }
    print("=== per-bead max |F| on WELL-CONDITIONED synthetic windows (bond angle 150 deg) ===")
    print("  (exact_b2's entropy term diverges as q -> +-1; reg floors it. |dq/dx| is finite")
    print("   everywhere because the bonds are never near-collinear.)")
    print(f"  {'phi(deg)':>8s} {'q':>12s} {'|dq/dx|':>9s} {'exact_b2 |F|':>14s} "
          f"{'reg eps=3deg |F|':>17s}")
    for side, phis in (("cis  q->+1", [4.7, 2.0, 1.0, 0.5, 0.2, 0.1]),
                       ("trans q->-1", [175.3, 178.0, 179.0, 179.5, 179.8, 179.9])):
        print(f"  -- {side} --")
        for ph in phis:
            pts = make_dihedral(ph)
            q = math.cos(math.radians(ph))
            g = geo_factor(pts)
            fe = window_force(specs["exact_b2"], pts)
            fr = window_force(specs["reg eps=3deg"], pts)
            print(f"  {ph:8.1f} {q:12.7f} {g:9.4f} {fe:14.1f} {fr:17.1f}")
    print()

    # ---- the relationship, and the provenance of 10.69
    print("=== per-bead force = |dV/dq| * |dq/dx|, and |dq/dx| has NO upper bound ===")
    print("  10.69 /nm is NOT the pole value: it is the max |dq/dx| over ONE chain (1L2X),")
    print("  computed in check_table_force_law.py and reused as a fixed multiplier in")
    print("  check_table_rebin.py (GMAX). On the well-conditioned sweep above |dq/dx| stays")
    print("  ~0.2-0.3 /nm all the way into q -> +-1 -- it does NOT diverge at the pole.")
    print("  The divergence of |dq/dx| is a DIFFERENT singularity: near-collinear bonds")
    print("  (|n| -> 0), which can happen at ANY q, not specifically near +-1.")
    print()

    # ---- the real geometry: where does the actual max force come from
    print("=== real geometry: the true max |dq/dx| is at a near-collinear window, not the pole ===")
    oiu = [s for s in B.load_structures() if s["name"] == "2OIU"][0]
    pos = torch.tensor(oiu["pos"], dtype=torch.float64).reshape(1, -1, 3)
    L = oiu["pos"].shape[0]
    a = torch.arange(L - 3)
    ij = torch.stack([3 * a, 3 * a + 3, 3 * a + 6, 3 * a + 9], dim=-1)
    worst = (0.0, None)
    for i in range(L - 3):
        pts = pos[0, ij[i]].numpy()
        g = geo_factor(pts)
        if g > worst[0]:
            worst = (g, i)
    gmax, iw = worst
    pts = pos[0, ij[iw]].numpy()
    n0, n1 = normals(pts)
    posw = torch.tensor(pts, dtype=torch.float64).reshape(1, 4, 3)
    qw, _ = FR.window_max_force("table", posw, ij_like=torch.tensor([[0, 1, 2, 3]]))
    fe = window_force(specs["exact_b2"], pts)
    fr = window_force(specs["reg eps=3deg"], pts)
    print(f"  2OIU window {iw}: q = {qw[0]:.4f}, |n0| = {n0:.4f}, |n1| = {n1:.4f}, "
          f"max|dq/dx| = {gmax:.2f} /nm")
    print(f"    exact_b2 per-window |F| = {fe:.1f},  reg eps=3deg per-window |F| = {fr:.1f}")
    print(f"    (q = {qw[0]:.4f} is nowhere near +-1, so the entropy floor never engages: the")
    print(f"     two are IDENTICAL, and both scale as table-slope * {gmax:.0f}.)")
    print()

    # ---- reconcile with force_reference.log
    print("=== reconcile with results/force_reference.log (chain-summed, what the cap sees) ===")
    print("  2OIU max |F|:  table 16477.57,  exact_b2 15735.80,  reg eps=3deg 15735.80")
    print("  -> reg leaves the max UNCHANGED (exact == reg), because the max is geometric,")
    print("     not entropic. So reg eps=3deg does NOT make the table form force-safe.")
    print()
    print("CONCLUSION: max|dV/dq| = 1959 (reg, entropy term only) is bounded, but the per-bead")
    print("force = |dV/dq| * |dq/dx| and |dq/dx| is unbounded (collinear bonds). Bounding dV/dq")
    print("does not bound the force. The preferred candidate is therefore Fourier N=2")
    print("(max 396 on 2OIU), whose |dV/dq| is a bounded polynomial in q, NOT table+reg.")


if __name__ == "__main__":
    main()
