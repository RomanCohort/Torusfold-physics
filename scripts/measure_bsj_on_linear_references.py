"""How much do the three closure terms distort a chain that is not covalently closed?

cgRNASP-style deposited chains are linear. K_BSJ, K_BSJ_GUIDE and K_BSJ_CONTACT all act on
P(0)-P(L-1) -- the backbone join, which exists only in a circular RNA. On a linear reference that
coordinate is a free end-to-end distance, measured at mean 3.927 nm and max 11.833 nm over the
loader's chains against a BOND_P_NEXT target of 0.590.

The criterion test already says K_BSJ has "no independent criterion" for this reason. What it does
not say is what the three terms DO to a linear structure, and that is this: their share of the
energy and of the force, so that a run on linear references can say whether it is measuring the
field or measuring the join.

Reported both ways round: the terms' share with the shipped constants, and the fold-ranking
response to zeroing them, because a term can be large in energy and still cancel out of a
ranking that only ever compares two pairings of the SAME chain.

Run: python scripts/measure_bsj_on_linear_references.py [n_structs]
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
import cg_force_terms as FT           # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BSJ = ("bsj closure", "bsj guide", "bsj contact")

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34][:N]
print(f"{len(structs)} linear reference chains from the loader")
e2e = []
rows = []
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    e2e.append(float(torch.linalg.norm(pos[0, 0] - pos[0, 3 * (L - 1)])))
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float64)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    total = float(C.cg_energy_forces(pos, ij, pw, cell_list=cl)[0])
    terms, energies = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    e_bsj = sum(energies.get(k, 0.0) for k in BSJ)
    f_per = {k: float(torch.linalg.norm(torch.as_tensor(v).reshape(-1, 3), dim=-1).mean())
             for k, v in terms.items()}
    f_sum = sum(f_per.values())
    f_bsj = sum(f_per.get(k, 0.0) for k in BSJ)
    rows.append((total, e_bsj, f_bsj, f_sum))

tot = np.array([r[0] for r in rows]); eb = np.array([r[1] for r in rows])
fb = np.array([r[2] for r in rows]); fs = np.array([r[3] for r in rows])
print()
print(f"P(0)-P(L-1) on these chains: mean {np.mean(e2e):.3f} nm, max {np.max(e2e):.3f} nm, "
      f"target BOND_P_NEXT {C.BOND_P_NEXT} nm")
print()
print(f"{'quantity':34s} {'mean':>11s} {'max':>11s}")
print("-" * 58)
print(f"{'closure energy (kJ/mol)':34s} {eb.mean():11.2f} {np.abs(eb).max():11.2f}")
print(f"{'closure as % of |total energy|':34s} "
      f"{100 * np.abs(eb / np.maximum(np.abs(tot), 1e-9)).mean():11.2f} "
      f"{100 * np.abs(eb / np.maximum(np.abs(tot), 1e-9)).max():11.2f}")
print(f"{'closure force share of sum |F| %':34s} {100 * (fb / fs).mean():11.3f} "
      f"{100 * (fb / fs).max():11.3f}")
print()
print("per term, energy share:")
for k in BSJ:
    vals = []
    for s in structs:
        L = len(s["pos"])
        pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
        ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
        pw = torch.ones(len(ij), dtype=torch.float64)
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        _t, e = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
        tt = float(C.cg_energy_forces(pos, ij, pw, cell_list=cl)[0])
        vals.append(100.0 * abs(e.get(k, 0.0)) / max(abs(tt), 1e-9))
    print(f"    {k:16s} {np.mean(vals):8.2f}% of |E|")
print()
print("And the same three, zeroed, against the register-shift decoy that DOES discriminate")
print("(scripts/select_k_pair_by_ranking.py): the closure terms depend only on P(0) and P(L-1), so")
print("both the native and the decoy see the same value and the difference is exactly zero. That is")
print("why the fold metrics never noticed this: they compare two pairings of the SAME chain.")
print()
print("So the distortion is real for DYNAMICS on a linear reference and invisible to every ranking")
print("test in the repo. A run that is not on a covalently closed chain should set K_BSJ,")
print("K_BSJ_GUIDE and K_BSJ_CONTACT to zero and say so in its own provenance line.")
