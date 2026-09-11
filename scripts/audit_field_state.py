"""State of the force field: every live constant, and whether the four paths are the same field."""
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402
import cg_force_terms as FT           # noqa: E402

# Comparing the five entry points IS this script's job, so it opts into the guard that
# _alternate_field puts on the four non-production ones. Until this line existed the script died
# at the comparison -- the guard is not a bug to work around, but a script whose whole purpose is
# the comparison has to say out loud that it is crossing it. The ratio it reports is the finding.
C.ALLOW_ALTERNATE_FIELDS = True

GROUP = {
    "bonded": ("K_BB", "BOND_P_NEXT", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
               "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
               "K_ANGLE", "ANGLE_PPP", "K_DIH", "DIH_PPPP", "K_STACK", "STACK_R0"),
    "nonbonded": ("K_PAIR", "PAIR_NN", "K_PAIR_GUIDE", "K_BPP", "K_CLASH", "CLASH_SIGMA"),
    "closure": ("K_BSJ", "K_BSJ_GUIDE", "K_BSJ_CONTACT"),
    "numerics": ("GB_FORCE_CAP", "GB_CUTOFF", "GB_SWITCH_ON", "GB_SWITCH_OFF"),
}
for g, names in GROUP.items():
    print(f"--- {g} ---")
    for n in names:
        v = getattr(C, n, "<MISSING>")
        print(f"    {n:16s} {v}")
print(f"--- numerics ---")
print(f"    {'force_cap':16s} {inspect.signature(C.cg_energy_forces).parameters['force_cap'].default}")
print(f"    {'CLASH_DIST':16s} {getattr(C, 'CLASH_DIST', '<MISSING>')}  (retired)")
print()

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"]); NB = 3 * L
pos = torch.tensor(s0["pos"].reshape(1, NB, 3), dtype=torch.float64)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float64)


def cl():
    c = C.GPUCellList(cell_size=1.5)
    c.build(pos)
    return c


print(f"five paths on {s0['name']} (L={L}, {len(ij)} WC pairs), same geometry, force_cap=None:")
print("  cg_energy_forces is the production field; the other four are different potentials under")
print("  the same constant names, and this script forces ALLOW_ALTERNATE_FIELDS to reach them.")
vals = {
    "cg_energy_forces": float(C.cg_energy_forces(pos, ij, pw, cell_list=cl(), force_cap=None)[0]),
    "cg_forces_explicit_batched": float(C.cg_forces_explicit_batched(pos, ij, pw, cell_list=cl())[0]),
    "cg_forces_explicit": float(C.cg_forces_explicit(pos, ij, pw, cell_list=cl())[0]),
    "cg_energy_3bead": float(C.cg_energy_3bead(pos, ij, pw)[0]),
}
base = vals["cg_energy_forces"]
for k, v in vals.items():
    print(f"    {k:28s} {v:12.4f}   ratio to cg_energy_forces {v / base if base else float('nan'):.4f}")

F, e = FT.term_energies_forces(pos, ij, pw, cell_list=cl())
print()
print(f"    decomposition has {len(e)} terms, sum {sum(e.values()):.4f}")
print("    " + ", ".join(sorted(e)))
print()

pos2 = pos.clone()
pos2[0, 5] += 0.03
for path, fn in (("cg_energy_forces", lambda p: float(C.cg_energy_forces(p, ij, pw, cell_list=cl(), force_cap=None)[0])),
                 ("cg_forces_explicit_batched", lambda p: float(C.cg_forces_explicit_batched(p, ij, pw, cell_list=cl())[0])),
                 ("cg_forces_explicit", lambda p: float(C.cg_forces_explicit(p, ij, pw, cell_list=cl())[0]))):
    print(f"    {path:28s} responds to a 0.03 nm move: {fn(pos2) - fn(pos):+.4f}")
