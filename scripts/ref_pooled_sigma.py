"""The three reference denominators a pooled sim/ref could be scored against.

Every IBI round-0 residual so far divides the simulated sigma by the npz table's `__sigma`,
which refit_tables_clean.py fitted on 96 cleaned chains. But the question being asked is
"one simulated chain vs 126 pooled chains", so the honest like-for-like denominator for a
POOLED simulation is the concatenation of all deposited chains (126), not the 96-chain fit.
And the structures actually simulated are the 7 that pass the pool filter
(len(pairs) >= 8, 24 <= len(pos) <= 34), so a third candidate is that 7-chain sub-pool.

This prints all three per coordinate and how far apart they are, so the pooled comparison
can state exactly which denominator it used. It measures only; it simulates nothing.

Run: python scripts/ref_pooled_sigma.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"


def pooled_sd(structs):
    """std of the concatenation of one coordinate's observations over all chains."""
    out = {}
    vals = {c: [] for c in B.COORDS}
    for s in structs:
        pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
        for c in B.COORDS:
            vals[c].append(B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64))
    for c in B.COORDS:
        v = np.concatenate(vals[c])
        out[c] = (float(v.std()), int(v.size))
    return out


z = np.load(NPZ)
tab_sigma = {c: float(z[f"{c}__sigma"]) for c in B.COORDS}

all_structs = B.load_structures()
pool = [s for s in all_structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]

sd_all, n_all = {}, {}
for c in B.COORDS:
    sd_all[c], n_all[c] = pooled_sd(all_structs)[c]
sd_pool, n_pool = {}, {}
for c in B.COORDS:
    sd_pool[c], n_pool[c] = pooled_sd(pool)[c]

print(f"{len(all_structs)} chains in the full database; {len(pool)} pass the pool filter")
print()
print(f"{'coordinate':10s} {'table sd':>9s} {'126-chain sd':>12s} {'7-chain sd':>11s} "
      f"{'table vs 126':>13s} {'table vs 7':>11s} {'n126':>7s} {'n7':>6s}")
print("-" * 88)
for c in B.COORDS:
    t = tab_sigma[c]
    a = sd_all[c]
    p = sd_pool[c]
    print(f"{c:10s} {t:9.4f} {a:12.4f} {p:11.4f} "
          f"{100 * (t / a - 1):+12.2f}% {100 * (t / p - 1):+10.2f}% "
          f"{n_all[c]:7d} {n_pool[c]:6d}")
print()
print("'table vs 126' is the gap between the 96-chain fit the sampler scores against and the")
print("126-chain pooled sd the question is about. Positive means the table is WIDER than the")
print("126-chain pool, so a pooled sim/ref scored against the table reads slightly LOW.")
