"""Is the stacking term compensating for an under-stiff backbone bond?

assess_stacking_redundancy.py established two things. The coordinate is an exact function of
the two P-P bond lengths and the P-P-P angle cosine, both of which already have restraints,
with R^2 of 1.000000 and a residual at machine precision. And ablating it leaves the funnel
rank and gap untouched at 1.000 and 0.0 while taking the bonded subset's cap saturation from
4.02 to 0.58 percent.

So by construction it cannot carry information the bond and angle terms do not, and the only
way it could still be doing work is by holding a coordinate that another term restrains too
loosely. The candidate is the bond term: 3s derived k = kBT/sigma^2 = 1129.2 for it against a
shipped 500, a factor of 2.26, and that criterion was applied to the angle and the dihedral
but never to the bond.

If that is what is happening, stiffening the bond should let the stacking term go without
cost. This runs the four combinations.

Run: python scripts/stacking_vs_bond_stiffness.py [n_structs]
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
gen = np.random.default_rng(29)
decoys = []
for s in structs:
    o = []
    for sig in SIGMAS:
        for _ in range(N_DECOY):
            o.append(gen.normal(0, sig / 10.0, s["pos"].shape))
    decoys.append(o)

K_BB_MATCH = float(B.KBT / np.concatenate([
    np.linalg.norm(s["pos"][1:, 0, :] - s["pos"][:-1, 0, :], axis=-1) for s in structs
]).std() ** 2)
print(f"K_BB shipped {C.K_BB}, kBT/sigma^2 = {K_BB_MATCH:.1f}")
print()


def evaluate(s, w):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(w, dtype=torch.float64)
    F, E = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    e = sum(E[k] for k in BONDED)
    f = np.linalg.norm(sum(F[k] for k in BONDED), axis=1)
    return e, f


saved = {n: getattr(C, n) for n in ("K_BB", "_K_BB", "K_STACK", "_K_STACK")
         if hasattr(C, n)}
print(f"{'K_BB':>9s} {'stacking':>10s} {'rank':>7s} {'gap':>9s} {'p95 |F|':>10s} "
      f"{'over cap':>9s}")
print("-" * 60)
try:
    for kbb in (C.K_BB, K_BB_MATCH):
        for kst in (1.0, 0.0):
            for n in ("K_BB", "_K_BB"):
                if n in saved:
                    setattr(C, n, kbb)
            for n in ("K_STACK", "_K_STACK"):
                if n in saved:
                    setattr(C, n, 500.0 * kst)
            ranks, gaps, mags = [], [], []
            for s, o in zip(structs, decoys):
                g = np.random.default_rng(31)
                w = np.asarray(W)[g.integers(0, len(W), len(s["pairs"]))]
                e0, f0 = evaluate(s, w)
                cand = [e0]
                for off in o:
                    d = dict(s)
                    d["pos"] = s["pos"] + off
                    cand.append(evaluate(d, w)[0])
                v = np.array(cand)
                ranks.append(int((v < v[0]).sum()) + 1)
                gaps.append(float(v.min() - v[0]))
                mags.append(f0)
            m = np.concatenate(mags)
            print(f"{kbb:9.1f} {('on' if kst else 'off'):>10s} {np.mean(ranks):7.3f} "
                  f"{np.mean(gaps):9.1f} {np.percentile(m,95):10.2f} "
                  f"{(m>CAP).mean()*100:8.2f}%")
finally:
    for n, v in saved.items():
        setattr(C, n, v)
print("-" * 60)
print()
print("bonded subset only. If stiffening the bond lets the stacking term go without the")
print("rank or the gap moving, the stacking term was holding what the bond held too loosely.")
