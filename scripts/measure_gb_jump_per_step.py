"""How big is the GB discontinuity over a whole integration step?

measure_gb_discontinuity.py moved ONE bead by 1e-4 nm and found a jump of at most 0.001953
kJ/mol, 0.0008 kBT, and called it negligible. But an integration step moves every bead, and
the jump comes from pairs crossing the 1.0 nm cell filter, so it should grow with the number
of beads moving. That caveat was left open.

This measures it properly. For a random displacement u applied to all beads at once,

    E(pos+u) - E(pos-u) = 2 F.u + jump

where F.u is the smooth first-order term the analytic force predicts. Expanding about pos,
E(+u) - E(-u) = 2 grad(E).u + O(d^3) = -2 F.u + O(d^3), so the residual is (E(+u) - E(-u)) +
2 F.u. An earlier version of this script used a minus sign there and therefore measured
4 |F.u|, which is why the residual came out at exactly twice the "smooth" column and scaled
linearly with the step instead of staying at rounding level.

Run: python scripts/measure_gb_jump_per_step.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS = (1e-4, 1e-3, 1e-2)
TRIALS = 12
structs = [s for s in B.load_structures(limit=200) if len(s["pairs"]) >= 4][:N]

print(f"{len(structs)} structures, {TRIALS} random directions per step size, all beads moved")
print(f"kBT = {B.KBT} kJ/mol at 300 K")
print()
print(f"{'structure':12s} {'L':>4s} {'step nm':>9s} {'|F.u| smooth':>14s} "
      f"{'jump median':>13s} {'jump max':>11s} {'max/kBT':>9s}")
print("-" * 76)
rng = np.random.default_rng(41)
worst_overall = 0.0
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float64)

    def E(p):
        return float(C.cg_energy_forces(p, ij, pw, force_cap=None,
                                        cell_list=None)[0].reshape(-1)[0])

    F = C.cg_energy_forces(pos, ij, pw, force_cap=None, cell_list=None)[1]
    for d in STEPS:
        smooth, jump = [], []
        for _ in range(TRIALS):
            u = torch.tensor(rng.normal(size=pos.shape), dtype=torch.float64)
            u = u / torch.linalg.norm(u.reshape(-1)) * (d * np.sqrt(3 * L))
            ep, em = E(pos + u), E(pos - u)
            # E(+u) - E(-u) = -2 F.u + O(d^3), so the smooth prediction is MINUS 2 F.u
            s_term = float((F * u).sum()) * 2.0
            smooth.append(abs(s_term))
            jump.append(abs((ep - em) + s_term))
        smooth = np.array(smooth); jump = np.array(jump)
        worst_overall = max(worst_overall, jump.max())
        print(f"{s['name']:12s} {L:4d} {d:9.0e} {np.median(smooth):14.4f} "
              f"{np.median(jump):13.6f} {jump.max():11.6f} {jump.max()/B.KBT:9.4f}")
print("-" * 76)
print(f"worst jump anywhere: {worst_overall:.6f} kJ/mol = {worst_overall/B.KBT:.4f} kBT")
print()
print("smooth is |2 F.u|, the first-order term the analytic force predicts; jump is what is")
print("left of E(+u) - E(-u) after removing it. For a smooth potential the jump is at")
print("rounding level and does not grow with the step.")
