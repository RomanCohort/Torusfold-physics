"""Which term owns the force that is already thousands per bead at NATIVE geometry?

The excluded volume was rebuilt from the database and it works -- a six-fold improvement in the
closest approach -- but it cannot set a floor, because the force cap clips it. The cap clips it
because the field already produces more than the cap at native, undamaged geometry: measured
above, max|F| 6033 kJ/mol/nm with the cap off, and 38 of 162 beads at the cap with it on. And the
collapsing pair's trajectory is IDENTICAL at K_CLASH = 0 and at the criterion value, i.e. the
repulsion is not what drives that collapse.

So before the cap or the collapse can be fixed, this has to be answered: at native geometry, with
nothing damaged yet, which term is producing thousands of kJ/mol/nm?

Run: python scripts/find_dominant_force_term.py [n_structs]
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
import cg_force_terms as T            # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 4][:NS]
print(f"{len(structs)} native structures, cap off, per-term decomposition")
print()

owner = Counter()
per_term_max = {}
per_term_share_at_worst = {}
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float32)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    terms, _e = T.term_energies_forces(pos, ij, pw, cell_list=cl)

    total = np.zeros_like(terms[list(terms)[0]])
    for k, v in terms.items():
        total += v
        m = np.linalg.norm(v, axis=1).max()
        per_term_max[k] = max(per_term_max.get(k, 0.0), float(m))
    mag = np.linalg.norm(total, axis=1)
    worst = int(np.argmax(mag))
    who, best = None, -1.0
    for k, v in terms.items():
        c = float(np.linalg.norm(v[worst]))
        share = c / max(float(mag[worst]), 1e-12)
        per_term_share_at_worst[k] = per_term_share_at_worst.get(k, 0.0) + share
        if c > best:
            best, who = c, k
    owner[who] += 1

n = len(structs)
print(f"=== which term is largest at the worst bead of each structure ===")
for k, cnt in owner.most_common():
    print(f"  {k:24s} {cnt:3d}/{n}")
print()
print(f"=== largest single-bead force each term ever produces ===")
print(f"{'term':26s} {'max |F| (kJ/mol/nm)':>21s}")
print("-" * 50)
for k in sorted(per_term_max, key=lambda x: -per_term_max[x]):
    print(f"{k:26s} {per_term_max[k]:21.2f}")
print()
print(f"=== mean share of the worst bead's force, per term ===")
print(f"{'term':26s} {'mean share':>12s}")
print("-" * 40)
for k in sorted(per_term_share_at_worst, key=lambda x: -per_term_share_at_worst[x]):
    print(f"{k:26s} {per_term_share_at_worst[k] / n:11.3f}")
print()
print("The decomposition duplicates terms that have no library helper, so read the ranking and")
print("not the absolute values; cg_force_terms.py is the copy that has already drifted four times.")
