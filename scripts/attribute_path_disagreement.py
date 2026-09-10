"""What exactly differs between the two force paths?

3j reported that cg_energy_forces and cg_forces_explicit_batched disagree by about 18
percent on the same structure, against a comment claiming they are consistent. Reading the
two functions side by side says the disagreement is not a numerical one: the batched path
does not implement the same field. It appears to be missing the pair guide, the BSJ contact
and the bpp term, it derives the dihedral force by autograd where the unified path uses an
acknowledged approximation, its GB coefficient is 0.365 against the unified path's
2.0 * 0.73, and it carries two extra Mg terms.

This tests that behaviourally instead of by reading. Each suspect constant is zeroed and
both paths are asked whether their energy moved; whichever path responds is the one that
implements the term.

Run: python scripts/attribute_path_disagreement.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

structs = B.load_structures(limit=4)
s = max(structs, key=lambda x: len(x["pos"]))
L = len(s["pos"])
pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float64)
cl = C.GPUCellList(cell_size=1.5)
cl.build(pos)
print(f"{s['name']}, L = {L}, {len(ij)} WC pairs")
print()


def energies():
    e1 = float(C.cg_energy_forces(pos, ij, pw, cell_list=cl)[0])
    e2 = float(C.cg_forces_explicit_batched(pos, ij, pw, cell_list=cl)[0])
    return e1, e2


base1, base2 = energies()
print(f"cg_energy_forces            {base1:12.2f} kJ/mol")
print(f"cg_forces_explicit_batched  {base2:12.2f} kJ/mol")
print(f"difference                  {base2 - base1:12.2f}  "
      f"({(base2 - base1) / abs(base1) * 100:+.1f} percent)")
print()

# Each probe zeroes every name the constant might be exported under, so a frozen snapshot
# in the batched path is covered too.
PROBES = [
    ("bpp",            ["K_BPP", "_K_BPP"]),
    ("pair guide",     ["K_PAIR_GUIDE", "_K_PAIR_GUIDE"]),
    ("BSJ contact",    ["K_BSJ_CONTACT", "_K_BSJ_CONTACT"]),
    ("Mg / K_MG",      ["K_MG", "_K_MG"]),
    ("clash",          ["K_CLASH", "_K_CLASH"]),
    ("WC pair",        ["K_PAIR", "_K_PAIR"]),
    ("angle",          ["K_ANGLE", "_K_ANGLE"]),
]
print(f"{'term zeroed':16s} {'unified moved?':>16s} {'batched moved?':>16s}")
print("-" * 52)
verdict = {}
for label, names in PROBES:
    saved = {n: getattr(C, n) for n in names if hasattr(C, n)}
    try:
        for n in saved:
            setattr(C, n, 0.0)
        a1, a2 = energies()
    finally:
        for n, v in saved.items():
            setattr(C, n, v)
    u_moved = abs(a1 - base1) > 1e-6 * max(abs(base1), 1.0)
    b_moved = abs(a2 - base2) > 1e-6 * max(abs(base2), 1.0)
    verdict[label] = (u_moved, b_moved)
    print(f"{label:16s} {str(u_moved):>16s} {str(b_moved):>16s}")
print()
print("'moved' means zeroing that constant changed the path's reported energy, i.e. the path")
print("implements the term. A term that moves one path but not the other is implemented in")
print("one and absent from the other.")
print()

# The GB coefficient, computed from each path's own formula on the same geometry.
with torch.no_grad():
    P = torch.arange(L) * 3
    gb_p = pos[:, P, :]
    d = torch.linalg.norm(gb_p[:, :, None, :] - gb_p[:, None, :, :], dim=-1).clamp(min=1e-6)
    eye = torch.eye(L, dtype=torch.float64).unsqueeze(0)
    ml = 1.0 - eye
    ion = C.C_MG_DEFAULT * 2.0 + C.C_NA_DEFAULT
    ld = 0.304 / np.sqrt(max(ion, 1e-6))
    exp_gb = torch.exp(-d / ld) * ml / d
    sa_overlap = (1.0 - d / 0.58).clamp(min=0.0) * ml
    e_unified = 2.0 * (0.73 * exp_gb
                       + 2.12e-2 * 4 * np.pi * 0.0225 * sa_overlap
                       - C.C_MG_DEFAULT * torch.exp(-d / 0.3) * ml).sum()
    e_batched_style = 0.365 * (torch.exp(-d / ld) * ml / d).sum()
print("the GB term as each path writes it, on this geometry (kJ/mol):")
print(f"  2.0 * 0.73 * sum(exp(-r/ld)/r)   {float(2.0 * 0.73 * exp_gb.sum()):12.2f}")
print(f"  0.365 * sum(exp(-r/ld)/r)        {float(e_batched_style):12.2f}")
print(f"  ratio                            "
      f"{float((2.0 * 0.73 * exp_gb.sum()) / e_batched_style):12.2f}")
print()
print("the unified path also folds the SASA and Mg exponentials into the same sum; the")
print("batched path adds a separate K_MG term and a separate -0.5*c_mg*sum term.")
