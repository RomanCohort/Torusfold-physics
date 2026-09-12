"""Why is the 7-chain sim-pool's native intra_cn sd (0.0059) so much narrower than the
126-chain pooled sd (0.0081) and the table sigma (0.0083)?

intra_cn is the C4'-N distance inside one residue, which physically should not depend on chain
LENGTH. So the 7-chain pool being ~28% narrower is not obviously a length effect and needs a real
cause before R_7 is used as the like-for-like denominator. The pool filter is
len(pairs) >= 8 and 24 <= len(pos) <= 34, i.e. it selects SHORT, FOLDED chains. Both axes can
move the distribution: folding pairs the base and pins the glycosidic bond (C4'-N), and the
database's longer chains have more unpaired single-stranded stretch. This splits each residue's
intra_cn observation by (a) whether its residue is in a Watson-Crick pair, and (b) chain length,
and reports the sd of each group so the cause is identified rather than assumed.

Run: python scripts/inspect_intra_cn_pool.py > results/inspect_intra_cn_pool.log
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402


def per_chain(structs):
    """For each chain: intra_cn values for paired residues and for unpaired residues."""
    rows = []
    for s in structs:
        pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
        v = B.coords_of(pos, "intra_cn").reshape(-1).numpy()   # one value per residue
        L = len(s["pos"])
        paired = np.zeros(L, dtype=bool)
        for a, b in s["pairs"]:
            paired[a] = True
            paired[b] = True
        rows.append((len(s["pos"]), len(s["pairs"]), v[paired], v[~paired]))
    return rows


def sd(x):
    x = np.asarray(x, dtype=float)
    return float(x.std()) if x.size > 1 else float("nan"), int(x.size)


all_structs = B.load_structures()
pool7 = [s for s in all_structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]

rows_all = per_chain(all_structs)
rows_pool = per_chain(pool7)

# pooled sd over all paired vs unpaired residues, whole database
vp = np.concatenate([r[2] for r in rows_all if r[2].size])
vu = np.concatenate([r[3] for r in rows_all if r[3].size])
vp_sd, vp_n = sd(vp)
vu_sd, vu_n = sd(vu)

print(f"{len(all_structs)} chains in database; {len(pool7)} pass the sim filter")
print()
print("whole database, intra_cn pooled sd split by pairing:")
print(f"  paired residues   : sd={vp_sd:.4f}  n={vp_n}")
print(f"  unpaired residues : sd={vu_sd:.4f}  n={vu_n}")
print(f"  paired fraction   : {vp_n / (vp_n + vu_n):.3f}")
print()

# pooled sd over the 7 sim-pool chains, same split
vpp = np.concatenate([r[2] for r in rows_pool if r[2].size])
vpu = np.concatenate([r[3] for r in rows_pool if r[3].size])
vpp_sd, vpp_n = sd(vpp)
vpu_sd, vpu_n = sd(vpu)
print("7-chain sim pool, intra_cn pooled sd split by pairing:")
print(f"  paired residues   : sd={vpp_sd:.4f}  n={vpp_n}")
print(f"  unpaired residues : sd={vpu_sd:.4f}  n={vpu_n}")
print(f"  paired fraction   : {vpp_n / (vpp_n + vpu_n):.3f}")
print()

# whole-pool intra_cn sd (all residues) vs whole-database, for the record
all_v = np.concatenate([np.concatenate([r[2], r[3]]) for r in rows_all])
pool_v = np.concatenate([np.concatenate([r[2], r[3]]) for r in rows_pool])
print(f"whole pooled intra_cn sd: database {all_v.std():.4f} (n={all_v.size}), "
      f"7-chain pool {pool_v.std():.4f} (n={pool_v.size})")
print()

# per-chain sd and length, to show whether the 7 chains are themselves anomalous or just short
print("per-chain intra_cn sd (paired sd, unpaired sd), L, pairs:")
print(f"{'name':8s} {'L':>3s} {'pairs':>5s} {'paired sd':>10s} {'unpaired sd':>12s}")
print("-" * 46)
for s in pool7:
    pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
    v = B.coords_of(pos, "intra_cn").reshape(-1).numpy()
    L = len(s["pos"])
    paired = np.zeros(L, dtype=bool)
    for a, b in s["pairs"]:
        paired[a] = paired[b] = True
    ps, pn = sd(v[paired])
    us, un = sd(v[~paired])
    print(f"{s['name'][:8]:8s} {L:3d} {len(s['pairs']):5d} {ps:10.4f}({pn:3d}) {us:12.4f}({un:3d})")
print()
print("If the 7 chains' paired sd is close to the whole-database paired sd, then their narrow")
print("pooled sd comes from the paired FRACTION (the filter selects folded chains), not from any")
print("residue being abnormal. That is a folding-composition effect, not a chain-length effect.")
