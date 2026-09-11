"""Is the stacking term a restraint on stacking, or a redundant function of two other terms?

The term is E = 0.5 * K_STACK * (|P(i) - P(i+2)| - STACK_R0)^2 with K_STACK = 500 and
STACK_R0 = 1.125 nm. Geometrically, with b_i = P(i+1) - P(i), b_{i+1} = P(i+2) - P(i+1) and
cos_a the cosine of the P-P-P angle as the code defines it (v1 = P(i)-P(i+1),
v2 = P(i+2)-P(i+1)),

    |P(i) - P(i+2)|^2 = |b_i|^2 + |b_{i+1}|^2 - 2 |b_i| |b_{i+1}| cos_a

which is exact. Both bond lengths are restrained by the bb bond term and cos_a by the angle
term, so this coordinate is a function of two quantities that already have their own
restraints -- with a third, separately tuned spring constant.

That matters beyond tidiness. The model has no base beads with orientation: the beads are
P, C4' and a point N9/N1, with no plane, no normal, and no rise or twist. So the field cannot
express base stacking at all, and a term named for it that restrains a backbone-derived
distance is not doing the job its name implies.

This checks the identity numerically and then ablates the term the way 3t ablated the angle
and dihedral.

Run: python scripts/assess_stacking_redundancy.py [n_structs]
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
CAP = 200.0
CACHE = Path(__file__).resolve().parent.parent / "results" / "rcm_weights.npz"
W = np.load(CACHE)["w"] if CACHE.exists() else np.full(50, 0.35)
SIGMAS = (0.3, 0.6, 1.0)
N_DECOY = 2
BONDED = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
          "angle P-P-P", "dihedral P-P-P-P", "stacking P-P")

structs = [s for s in B.load_structures(limit=300) if len(s["pairs"]) >= 4][:N]

# ---- the identity ----------------------------------------------------------
lhs, rhs = [], []
for s in structs:
    P = s["pos"][:, 0, :]
    for i in range(len(P) - 2):
        b0 = P[i + 1] - P[i]
        b1 = P[i + 2] - P[i + 1]
        v1 = P[i] - P[i + 1]
        v2 = P[i + 2] - P[i + 1]
        cos_a = float((v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        lhs.append(np.linalg.norm(P[i + 2] - P[i]) ** 2)
        rhs.append(np.linalg.norm(b0) ** 2 + np.linalg.norm(b1) ** 2
                   - 2 * np.linalg.norm(b0) * np.linalg.norm(b1) * cos_a)
lhs, rhs = np.array(lhs), np.array(rhs)
err = np.abs(lhs - rhs)
print(f"{len(lhs)} i,i+2 windows over {len(structs)} structures")
print(f"|P(i)-P(i+2)|^2 by direct measurement     mean {lhs.mean():.6f}")
print(f"the same from the two bonds and cos_a      mean {rhs.mean():.6f}")
print(f"max |difference| {err.max():.3e} nm^2, max relative {np.max(err/np.maximum(lhs,1e-9)):.3e}")
print()

# ---- how much of the spread does the identity explain ----------------------
fit = np.polyfit(np.sqrt(np.maximum(rhs, 1e-12)), np.sqrt(np.maximum(lhs, 1e-12)), 1)
pred = np.polyval(fit, np.sqrt(np.maximum(rhs, 1e-12)))
ss = 1 - ((np.sqrt(lhs) - pred) ** 2).sum() / ((np.sqrt(lhs) - np.sqrt(lhs).mean()) ** 2).sum()
print(f"R^2 of |P(i)-P(i+2)| predicted from the bonds and the angle: {ss:.6f}")
print()

# ---- ablation --------------------------------------------------------------
gen = np.random.default_rng(23)
decoys = []
for s in structs:
    o = []
    for sig in SIGMAS:
        for _ in range(N_DECOY):
            o.append(gen.normal(0, sig / 10.0, s["pos"].shape))
    decoys.append(o)


def evaluate(s, wstack, w):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(w, dtype=torch.float64)
    F, E = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    e = sum(E[k] for k in BONDED if k != "stacking P-P") + wstack * E["stacking P-P"]
    f = sum(F[k] for k in BONDED if k != "stacking P-P") + wstack * F["stacking P-P"]
    return e, np.linalg.norm(f, axis=1)


for label, ws in (("stacking x1 (shipped)", 1.0), ("stacking x0.1", 0.1),
                  ("stacking off", 0.0)):
    ranks, gaps, mags = [], [], []
    for s, o in zip(structs, decoys):
        w = np.asarray(W)[gen.integers(0, len(W), len(s["pairs"]))]
        e0, f0 = evaluate(s, ws, w)
        cand = [e0]
        for off in o:
            d = dict(s)
            d["pos"] = s["pos"] + off
            cand.append(evaluate(d, ws, w)[0])
        v = np.array(cand)
        ranks.append(int((v < v[0]).sum()) + 1)
        gaps.append(float(v.min() - v[0]))
        mags.append(f0)
    m = np.concatenate(mags)
    print(f"{label:24s} rank {np.mean(ranks):5.3f}  gap {np.mean(gaps):9.1f}  "
          f"p95 |F| {np.percentile(m,95):8.2f}  over cap {(m>CAP).mean()*100:5.2f}%")
print()
print("the bonded subset only: these six terms are what 3t ablated, and the stacking term is")
print("scaled for the ablation.")
