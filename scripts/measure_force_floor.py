"""What does the field actually produce, once the workload artifact is removed?

The per-term attribution found bsj closure owning the worst bead in 9 of 25 native structures,
with a 0.338 mean share of that bead's force. But bsj closure restrains |P(0)-P(L-1)| to 0.590 nm
and the reference structures are LINEAR deposited chains whose ends are 3.927 nm apart on average
and up to 11.833. So on these structures the term is pure artifact: it is the right term for a
circular molecule and the wrong one for a linear test case, and every full-field measurement made
on a linear structure carries it.

This measures the field with K_BSJ set to zero, which is the honest way to ask "what does this
field produce on a linear structure". The answer is the height the force cap has to cover.

Both configurations are run on the same structures and the same geometry; the only difference is
that one constant.

Run: python scripts/measure_force_floor.py [n_structs] [n_steps]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
K_BSJ_SHIPPED = C.K_BSJ

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 4][:NS]
print(f"{len(structs)} native structures, cap off")
print(f"K_BSJ shipped = {K_BSJ_SHIPPED}, and the e2e distance of these linear references is "
      f"3.927 nm mean / 11.833 max against a 0.590 target")
print()
print(f"{'configuration':16s} {'max |F|':>10s} {'p99.9':>9s} {'p99':>9s} {'median':>9s} "
      f"{'>200':>8s}")
print("-" * 68)
res = {}
for label, kbsj in (("K_BSJ as shipped", K_BSJ_SHIPPED), ("K_BSJ = 0", 0.0)):
    C.K_BSJ = kbsj
    mags = []
    worst_owner = Counter()
    per_term = {}
    import cg_force_terms as T
    for s in structs:
        L = len(s["pos"])
        pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
        ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
        pw = torch.ones(len(ij), dtype=torch.float32)
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=None)
        m = torch.linalg.norm(f.reshape(-1, 3), dim=-1).numpy()
        mags.append(m)
        terms, _en = T.term_energies_forces(pos, ij, pw, cell_list=cl)
        total = np.zeros_like(terms[list(terms)[0]])
        for k, v in terms.items():
            total += v
            per_term[k] = max(per_term.get(k, 0.0), float(np.linalg.norm(v, axis=1).max()))
        w = int(np.argmax(np.linalg.norm(total, axis=1)))
        worst_owner[max(terms, key=lambda k: float(np.linalg.norm(terms[k][w])))] += 1
    m = np.concatenate(mags)
    res[label] = (m, per_term, worst_owner)
    print(f"{label:16s} {m.max():10.1f} {np.percentile(m, 99.9):9.1f} "
          f"{np.percentile(m, 99):9.1f} {np.median(m):9.1f} "
          f"{100 * (m > 200).mean():7.2f}%")
C.K_BSJ = K_BSJ_SHIPPED
print()

print("=== with K_BSJ = 0, who owns the worst bead now ===")
for k, cnt in res["K_BSJ = 0"][2].most_common():
    print(f"  {k:24s} {cnt:3d}/{len(structs)}")
print()
print("=== with K_BSJ = 0, the largest single-bead force per term ===")
print(f"{'term':26s} {'max |F|':>10s}   {'shipped K_BSJ':>14s}")
print("-" * 56)
main, off = res["K_BSJ as shipped"][1], res["K_BSJ = 0"][1]
for k in sorted(off, key=lambda x: -off[x]):
    print(f"{k:26s} {off[k]:10.1f}   {main.get(k, 0.0):14.1f}")
print()
print("The difference between the two columns is the artifact. What is left in the right-hand")
print("column is the height the force cap has to cover on a linear structure.")
