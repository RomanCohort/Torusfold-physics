"""Three scale choices with no criterion: how much of the force do they actually carry?

tests/test_pair_clash_bsj_criterion.py says K_BSJ_GUIDE has "no criterion ... its scale is an
annealing-schedule choice", and K_PAIR_GUIDE and K_BSJ_CONTACT are in the same position: the
guide terms are softplus/sigmoid, strictly monotone in distance, so there is no equilibrium point
and no curvature for kBT/sigma^2 to match. That is a statement about WHY they have no criterion.
This is the statement about whether it matters: their share of the force the field applies.

If a term carries a negligible share at native geometry, then its scale is not merely unmeasured,
it is unmeasurable and inert, and zeroing it removes an unmeasured number without changing the
physics. If it carries a real share, then it needs a criterion and the honest thing is to say it
still does not have one.

Reported per term: mean and max |F| over beads, and the share of the total |F| summed over terms.
Ablation is reported too -- the same geometry with that one constant set to zero -- because a term
can be small in magnitude and still be the only thing holding something apart.

Run: python scripts/measure_guide_term_share.py [n_structs]
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
WATCH = ("pair guide", "bsj guide", "bsj contact", "bpp", "wc pair N-N", "clash")

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34][:N]
print(f"{len(structs)} structures")
print()

acc = {}
totals = []
e_tot = []
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float64)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    terms, energies = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    e_tot.append(float(C.cg_energy_forces(pos, ij, pw, cell_list=cl)[0]))
    per = {k: float(torch.linalg.norm(torch.as_tensor(v).reshape(-1, 3), dim=-1).mean())
           for k, v in terms.items()}
    ssum = sum(per.values())
    totals.append(ssum)
    for k, v in per.items():
        acc.setdefault(k, []).append(100.0 * v / ssum if ssum else 0.0)

print(f"per-term mean |F| as a share of the summed per-term mean |F|, over {len(structs)} structures")
print(f"{'term':22s} {'share %':>9s} {'min':>8s} {'max':>8s}")
print("-" * 52)
rows = sorted(acc.items(), key=lambda kv: -np.mean(kv[1]))
for k, v in rows:
    mark = "  <- no criterion" if k in ("pair guide", "bsj guide", "bsj contact") else ""
    print(f"{k:22s} {np.mean(v):9.4f} {np.min(v):8.4f} {np.max(v):8.4f}{mark}")
print()
watched = {k: float(np.mean(v)) for k, v in rows if k in WATCH}
print("the three without a criterion, together: "
      f"{sum(watched.get(k, 0.0) for k in ('pair guide', 'bsj guide', 'bsj contact')):.4f} % of the "
      f"summed per-term |F|")
print()

print("ablation: does zeroing them change the ENERGY at native geometry?")
print(f"{'constant':22s} {'value':>9s} {'dE mean':>11s} {'dE max':>11s} {'|dE| as % of |E|':>18s}")
print("-" * 76)
for name in ("K_PAIR_GUIDE", "K_BSJ_GUIDE", "K_BSJ_CONTACT"):
    saved = getattr(C, name)
    ds = []
    try:
        for s in structs:
            L = len(s["pos"])
            pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
            ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
            pw = torch.ones(len(ij), dtype=torch.float64)
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            base = float(C.cg_energy_forces(pos, ij, pw, cell_list=cl)[0])
            setattr(C, name, 0.0)
            cl2 = C.GPUCellList(cell_size=1.5)
            cl2.build(pos)
            off = float(C.cg_energy_forces(pos, ij, pw, cell_list=cl2)[0])
            setattr(C, name, saved)
            ds.append(base - off)
    finally:
        setattr(C, name, saved)
    ds = np.array(ds)
    denom = np.mean(np.abs(e_tot))
    print(f"{name:22s} {saved:9.1f} {ds.mean():11.4f} {np.abs(ds).max():11.4f} "
          f"{100 * np.abs(ds).max() / denom:18.5f}")
print()
print("A share far below the smallest term that does have a criterion, and an ablation that moves")
print("almost no energy, together mean the scale is inert rather than merely unmeasured. That is a")
print("reason to remove the number, not a reason to guess one.")
