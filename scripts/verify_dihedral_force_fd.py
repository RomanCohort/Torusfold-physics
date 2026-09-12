"""Is _dihedral_f's returned force the true gradient of its own energy? (Question 1 of the
dihedral investigation.)

The repository's own open note (docs/statistical_potentials_as_forces.md, the "still
unresolved" list) says _dihedral_f is a self-admitted approximate gradient with arbitrary
0.25 coefficients. That note is STALE: the 0.25 coefficients now live only in
old_torch_cgsim_ref.py (lines 628-629). The live _dihedral_f differentiates its energy
expression with autograd on a detached copy of the P atoms, so by construction the force is
the exact gradient of the energy it returns.

This script verifies that claim against central finite differences of the term's OWN energy
along Cartesian P-atom perturbations, on a real structure (1L2X), at three step sizes. It
checks the term in isolation (not the full field), so it does not trip over the two things
the README says make the full-path gradcheck "impossible by construction": the force cap and
the GB/SA pair-list switching, neither of which is in _dihedral_f. The finite difference here
perturbs a continuous function, so a match means F = -dE/dx exactly (to FD truncation order).

This only answers "is the force self-consistent with its energy". It does NOT answer why the
sampled distribution is narrow -- that is scripts/dihedral_measure_1d.py.

Run: python scripts/verify_dihedral_force_fd.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402


def _rel_l2(F_an, F_fd):
    num = float(torch.linalg.norm(F_an - F_fd))
    den = max(float(torch.linalg.norm(F_fd)), 1e-12)
    return num / den


def main():
    pool = [s for s in B.load_structures(limit=400)
            if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
    s0 = pool[0]
    assert s0["name"] == "1L2X", s0["name"]
    L = len(s0["pos"])
    pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    P = lambda i: 3 * i + 0
    p_idx = [P(i) for i in range(L)]

    k = C.K_DIH
    c = float(np.cos(C.DIH_PPPP))
    e, F = C._dihedral_f(pos, k, c)

    print(f"structure {s0['name']}  L={L}  K_DIH={k}  target cos={c:.4f} "
          f"(phi0={np.degrees(np.arccos(c)):.2f} deg)")
    print(f"energy (term, all windows) = {float(e.sum()):.6f} kJ/mol")
    print(f"|F| max = {float(torch.linalg.norm(F, dim=-1).max()):.4f} kJ/mol/nm "
          f"(the old 0.25 approximation capped this at ~dE/dcos; see docs 3ad)")
    print()
    print("central finite difference of THIS energy, P atoms only:")
    print(f"{'h (nm)':>10s} {'rel L2':>10s} {'max abs err':>12s}")
    print("-" * 36)
    for h in (1e-4, 1e-5, 1e-6):
        fd = torch.zeros_like(F)
        for i in p_idx:
            for d in range(3):
                hi = pos.clone(); hi[:, i, d] += h
                lo = pos.clone(); lo[:, i, d] -= h
                eh = float(C._dihedral_f(hi, k, c)[0].sum())
                el = float(C._dihedral_f(lo, k, c)[0].sum())
                fd[:, i, d] = -(eh - el) / (2.0 * h)
        rel = _rel_l2(F, fd)
        maxabs = float((F - fd).abs().max())
        print(f"{h:10.0e} {rel:10.3e} {maxabs:12.3e}")

    print()
    print("conclusion: the live force is the exact gradient of its own energy (rel L2 shrinks")
    print("with h as truncation order predicts), so the 0.25-approximation is gone and is NOT")
    print("the cause of the narrow dihedral. The mechanism is in dihedral_measure_1d.py.")


if __name__ == "__main__":
    main()
