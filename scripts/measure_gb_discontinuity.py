"""How large is the GB cutoff discontinuity on realistic structures?

diagnose_energy_discontinuity.py shows the full-path energy is discontinuous: moving bead 0
by 1e-5 nm in a test chain changes the GB contribution by -0.074 kJ/mol in one direction and
0.000 in the other, and 2.0 * exp(-1.0/0.304)/1.0 = 0.074, exactly one pair entering the
1.0 nm cell filter that cg_energy_forces rebuilds from the coordinates on every call.

Within a call the pair list feeds both the energy and the gradient, so they agree with each
other. What is discontinuous is the potential. For the integrator that is mostly survivable
because it only ever sees the analytic force, but the REMD exchange criterion uses the energy,
and a step in the energy is a step in the acceptance ratio.

One pair crossing contributes 0.074 kJ/mol, which is 3 percent of kBT at 300 K. The question
is how many pairs sit near the cutoff at once. This measures the total jump over a small
displacement for real structures.

Run: python scripts/measure_gb_discontinuity.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
h = 1e-4          # 1e-4 nm, where the discontinuity should dwarf the smooth part
structs = [s for s in B.load_structures(limit=200) if len(s["pairs"]) >= 4][:N]

print(f"{len(structs)} structures, displacement {h} nm, kBT = {B.KBT} kJ/mol")
print()
print(f"{'structure':14s} {'L':>4s} {'smooth (kJ/mol)':>16s} {'jump (kJ/mol)':>14s} "
      f"{'jump/kBT':>9s}")
print("-" * 64)
jumps = []
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float64)

    def E(p):
        return float(C.cg_energy_forces(p, ij, pw, force_cap=None,
                                        cell_list=None)[0].reshape(-1)[0])

    hi = pos.clone(); hi[0, 3, 0] += h
    lo = pos.clone(); lo[0, 3, 0] -= h
    ep, e0, em = E(hi), E(pos), E(lo)
    smooth = abs((ep - e0) - (e0 - em)) / 2.0     # asymmetry is the jump
    jumps.append(smooth)
    print(f"{s['name'] + ' (L=' + str(L) + ')':14s} {L:4d} {abs(ep-e0):16.6f} "
          f"{smooth:14.6f} {smooth/B.KBT:9.4f}")
print("-" * 64)
jumps = np.array(jumps)
print(f"median jump {np.median(jumps):.6f} kJ/mol = {np.median(jumps)/B.KBT:.4f} kBT")
print(f"max jump    {jumps.max():.6f} kJ/mol = {jumps.max()/B.KBT:.4f} kBT")
print()
print("The asymmetry between the forward and backward difference is the discontinuity; the")
print("symmetric part is the smooth derivative and scales with h. Only one bead is moved, so")
print("the total over a trajectory is larger.")
