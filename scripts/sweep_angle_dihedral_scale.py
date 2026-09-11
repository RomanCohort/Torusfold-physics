"""Keeping some bending stiffness while getting the forces under the cap.

ablate_backbone_terms.py shows that removing the backbone angle and dihedral together takes
the funnel to a perfect rank 1.000 and takes the beads over the 200 kJ/mol/nm cap from 27.1
percent to 1.9, while removing either alone changes almost nothing. But removing both leaves
a chain with no bending stiffness at all -- the local terms only hold bond lengths and
intra-bead distances, so the backbone would be a freely jointed chain. For a folding pipeline
that is its own problem: the search space explodes and the chain has no physical persistence
length.

The grid that matters is therefore both terms kept but scaled down. ablate_backbone_terms.py
only swept them one at a time, so the two-dimensional cell was missing. This fills it.

Force metrics only: no decoys are generated, so this is much cheaper than the funnel run.

Run: python scripts/sweep_angle_dihedral_scale.py [n_structs]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CAP = 200.0
FACTORS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

structs = B.load_structures(limit=N)
print(f"{len(structs)} structures")
print()


def forces(ws_dih, ws_ang):
    out = []
    for s in structs:
        L = len(s["pos"])
        pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
        ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2) if s["pairs"] else \
            torch.zeros((0, 2), dtype=torch.long)
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        F, _e = FT.term_energies_forces(pos, ij,
                                        torch.ones(len(ij), dtype=torch.float64),
                                        cell_list=cl)
        tot = (F["bb bond P-P"] + F["intra P-C4'"] + F["intra C4'-N"]
               + F["stacking P-P"]
               + ws_ang * F["angle P-P-P"] + ws_dih * F["dihedral P-P-P-P"])
        out.append(np.linalg.norm(tot, axis=1))
    return np.concatenate(out)


print("percentage of beads over the 200 kJ/mol/nm cap")
print(f"{'angle \\ dihedral':>16s} " + "".join(f"{f:>9g}" for f in FACTORS))
print("-" * 72)
for wa in FACTORS:
    cells = []
    for wd in FACTORS:
        f = forces(wd, wa)
        cells.append((f > CAP).mean() * 100)
    print(f"{wa:>16g} " + "".join(f"{c:8.1f}%" for c in cells))
print()
print("p95 of the per-bead bonded force, kJ/mol/nm")
print(f"{'angle \\ dihedral':>16s} " + "".join(f"{f:>9g}" for f in FACTORS))
print("-" * 72)
for wa in FACTORS:
    cells = []
    for wd in FACTORS:
        cells.append(float(np.percentile(forces(wd, wa), 95)))
    print(f"{wa:>16g} " + "".join(f"{c:9.0f}" for c in cells))
print()
print("for reference the full field is angle=dihedral=1.0, and the row 'off' in")
print("ablate_backbone_terms.py corresponds to 0.0 in both.")
