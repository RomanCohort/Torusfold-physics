"""Ablate the backbone angle and dihedral and see whether anything gets worse.

The argument, from the measurements rather than from taste. The dihedral cosine has a pooled
spread of 0.588 over the cosine range, while the spread of the per-structure MODE is only
0.148. By the law of total variance the within-structure spread is therefore
sqrt(0.588^2 - 0.148^2) = 0.57, essentially the whole width. What varies is not the typical
value from structure to structure but the thing inside one structure -- a single RNA has
helices and loops, and which one a residue sits in is decided by the fold, not by local
geometry. A bonded term cannot pin that, and restraining it hard to the pooled mode is what
3r measured as 26 percent of beads sitting on the force cap.

So: if the field still ranks a native structure above its decoys with the angle and the
dihedral removed or softened, those terms are doing no useful work and the stiffness is pure
damage.

Two things are measured for each setting, because a term can be useless in the funnel and
still be load-bearing for the forces:
  * the funnel rank over held-out structures, six decoys each
  * the per-bead bonded force, and how much of it the 200 kJ/mol/nm cap would absorb

Run: python scripts/ablate_backbone_terms.py [n_train] [n_test]
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

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 24
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SIGMAS = (0.3, 0.6, 1.0)
N_DECOY = 2
SEED = 20260218
CAP = 200.0

BASE = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
        "angle P-P-P", "dihedral P-P-P-P", "stacking P-P")
ABLATABLE = {"angle": "angle P-P-P", "dihedral": "dihedral P-P-P-P",
             "stack": "stacking P-P"}
FACTORS = (1.0, 0.3, 0.1, 0.03, 0.0)

allstructs = B.load_structures(limit=N_TRAIN + N_TEST)
TEST = allstructs[N_TRAIN:N_TRAIN + N_TEST]
print(f"{len(TEST)} held-out structures, {N_DECOY} decoys per sigma at {SIGMAS}")
print()


def evaluate(s, weights):
    """(bonded energy, per-bead bonded force) with each term scaled by weights[name]."""
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2) if s["pairs"] else \
        torch.zeros((0, 2), dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    F, E = FT.term_energies_forces(pos, ij, torch.ones(len(ij), dtype=torch.float64),
                                   cell_list=cl)
    e = 0.0
    f = None
    for name, key in ABLATABLE.items():
        w = weights.get(name, 1.0)
        e += w * E[key]
        f = w * F[key] if f is None else f + w * F[key]
    for key in BASE:
        if key in ABLATABLE.values():
            continue
        e += E[key]
        f = F[key] if f is None else f + F[key]
    return e, np.linalg.norm(f, axis=1)


rng = np.random.default_rng(SEED)
decoys = []
for s in TEST:
    o = []
    for sig in SIGMAS:
        for _ in range(N_DECOY):
            o.append(rng.normal(0, sig / 10.0, s["pos"].shape))
    decoys.append(o)

SETTINGS = [("full", {})]
for term in ("dihedral", "angle"):
    for fac in FACTORS[:-1]:
        SETTINGS.append((f"{term} x{fac:g}", {term: fac}))
    SETTINGS.append((f"{term} off", {term: 0.0}))
SETTINGS.append(("dihedral+angle off", {"dihedral": 0.0, "angle": 0.0}))
SETTINGS.append(("dihedral+angle+stack off",
                 {"dihedral": 0.0, "angle": 0.0, "stack": 0.0}))

print(f"{'setting':26s} {'rank':>7s} {'gap':>10s} {'p95 |F|':>11s} {'max |F|':>11s} "
      f"{'over cap':>9s}")
print("-" * 80)
force_samples = TEST[:12]
rows = {}
for label, weights in SETTINGS:
    ranks, gaps = [], []
    for s, o in zip(TEST, decoys):
        e0, _ = evaluate(s, weights)
        cand = [e0]
        for off in o:
            d = dict(s)
            d["pos"] = s["pos"] + off
            cand.append(evaluate(d, weights)[0])
        v = np.array(cand)
        ranks.append(int((v < v[0]).sum()) + 1)
        gaps.append(float(v.min() - v[0]))
    fs = np.concatenate([evaluate(s, weights)[1] for s in force_samples])
    rows[label] = (float(np.mean(ranks)), float(np.mean(gaps)),
                   float(np.percentile(fs, 95)), float(fs.max()),
                   float((fs > CAP).mean() * 100))
    r = rows[label]
    print(f"{label:26s} {r[0]:7.3f} {r[1]:10.1f} {r[2]:11.1f} {r[3]:11.1f} {r[4]:8.1f}%")
print("-" * 80)
print()
print(f"rank 1.000 = the native is the lowest of {1 + len(SIGMAS)*N_DECOY} candidates in")
print("every held-out structure; higher is worse. 'gap' is min(decoy)-native, so negative")
print("means a decoy beat the native.")
print(f"'over cap' is the fraction of beads whose bonded force exceeds {CAP:.0f} kJ/mol/nm,")
print("i.e. the fraction the cap in cg_energy_forces would truncate.")
