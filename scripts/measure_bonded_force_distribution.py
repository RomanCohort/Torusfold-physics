"""What is the per-bead bonded force distribution now?

Two statistics were being conflated. verify_gap_fix_and_dihedral.py reported 2323.7 as the
median over windows of the MAXIMUM atom force in that window. funnel_vs_stiffness.py
reported 2.9 as the median over beads of the summed bonded force. They measure different
things and both were quoted as "the" median. This reports the distribution properly, per
bead and per atom, on the cleaned data.

Also worth separating: how many beads carry a force over the 200 kJ/mol/nm cap. A few beads
above it can cover most of the windows, because each atom appears in up to four.

Run: python scripts/measure_bonded_force_distribution.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import cg_force_terms as FT
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
structs = B.load_structures(limit=N)
BONDED = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
          "angle P-P-P", "dihedral P-P-P-P", "stacking P-P")
CAP = 200.0

tots, per_term = [], {k: [] for k in BONDED}
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2) if s["pairs"] else \
        torch.zeros((0, 2), dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    F, _ = FT.term_energies_forces(pos, ij, torch.ones(len(ij), dtype=torch.float64),
                                   cell_list=cl)
    tot = sum(F[k] for k in BONDED)
    tots.append(np.linalg.norm(tot, axis=1))
    for k in BONDED:
        per_term[k].append(np.linalg.norm(F[k], axis=1))

tot = np.concatenate(tots)
print(f"{len(structs)} cleaned structures, {len(tot)} beads (P/C4'/N triples)")
print()
print("summed bonded force per bead (kJ/mol/nm), before the cap:")
print(f"  median {np.median(tot):10.2f}   p95 {np.percentile(tot,95):10.2f}   "
      f"p99 {np.percentile(tot,99):10.2f}   max {tot.max():12.2f}")
print(f"  beads over the 200 cap: {int((tot>CAP).sum())} of {len(tot)} "
      f"({(tot>CAP).mean()*100:.2f} percent)")
print()
print(f"{'term':22s} {'median':>9s} {'p95':>10s} {'p99':>11s} {'max':>12s} "
      f"{'over cap':>9s}")
print("-" * 78)
for k in BONDED:
    v = np.concatenate(per_term[k])
    print(f"{k:22s} {np.median(v):9.2f} {np.percentile(v,95):10.2f} "
          f"{np.percentile(v,99):11.2f} {v.max():12.2f} {int((v>CAP).sum()):9d}")
print()
print("only the P atoms carry the angle and dihedral terms, so those two rows are")
print("statistics over the P subset rather than over every bead.")
