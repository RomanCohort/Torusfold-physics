"""Energy funnel test: harmonic bonded terms against Boltzmann-inverted ones.

The test that all of this exists to pass. Li & Chen's review (PMC13110076) calls it
top-down Energy Funnel Optimization -- "the energy landscape should guide folding toward
native state" -- and notes it "requires the generation of a well-designed set of decoy
structures".

Before the recalibration the shipped field failed it badly: over 32 held-out structures a
native structure ranked 6.75th of 7 candidates, with the best random perturbation scoring
19592.6 kJ/mol LOWER. This asks whether replacing the bonded springs with tabulated
U = -kBT ln P does better than the best harmonic version available.

Four energies are compared on the same structures and the same decoys:
  harmonic   the six bonded terms as the library computes them, with the recalibrated
             constants (STACK_R0 1.125, DIH cos 0.975)
  boltzmann  the same six coordinates from -kBT ln P_ref, fitted on the FIT half
  full       cg_energy_forces, the whole field, harmonic bonded terms included
  swapped    full, with the six harmonic bonded energies replaced by the Boltzmann ones

Note on scale: -kBT ln P has a depth of a few kBT (the tables bottom out around 15-40
kJ/mol across their range) while the harmonic terms with k around 500 produce hundreds of
kJ/mol at a few tenths of a nm. Swapping them in is therefore not a like-for-like
substitution, and that mismatch is a result to report, not a detail to hide.

Run: python scripts/funnel_test_boltzmann.py [n_fit] [n_test]
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

N_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SIGMAS = (0.3, 0.6, 1.0)
N_DECOY = 2
SEED = 20260214

BONDED = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
          "angle P-P-P", "dihedral P-P-P-P", "stacking P-P")

allstructs = B.load_structures(limit=N_FIT + N_TEST)
FIT = allstructs[:N_FIT]
TEST = allstructs[N_FIT:N_FIT + N_TEST]
print(f"fitted on {len(FIT)} structures, tested on {len(TEST)} held out")

tables = B.prepare(B.fit(FIT))
print(f"tables built: {', '.join(B.COORDS)}")
print()


def unpack(s):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    pairs = s["pairs"]
    ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2) if pairs else \
        torch.zeros((0, 2), dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    return pos, ij, cl


def energies(s):
    pos, ij, cl = unpack(s)
    pw = torch.ones(len(ij), dtype=torch.float64)
    e_full = float(C.cg_energy_forces(pos, ij, pw, cell_list=cl)[0])
    _, terms_e = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    e_harm = sum(terms_e[k] for k in BONDED)
    e_boltz = float(B.energy(pos, tables))
    return e_harm, e_boltz, e_full, e_full - e_harm + e_boltz


rng = np.random.default_rng(SEED)
keys = ("harmonic", "boltzmann", "full", "swapped")
ranks = {k: [] for k in keys}
margins = {k: [] for k in keys}
scale = {k: [] for k in keys}

for s in TEST:
    base = energies(s)
    cands = {k: [base[i]] for i, k in enumerate(keys)}
    for sig in SIGMAS:
        for _ in range(N_DECOY):
            d = dict(s)
            d["pos"] = s["pos"] + rng.normal(0, sig / 10.0, s["pos"].shape)
            e = energies(d)
            for i, k in enumerate(keys):
                cands[k].append(e[i])
    for k in keys:
        v = np.array(cands[k])
        ranks[k].append(int((v < v[0]).sum()) + 1)
        margins[k].append(float(v.min() - v[0]))
        scale[k].append(abs(v[0]))

print(f"{'energy':12s} {'native rank':>12s} {'mean margin':>14s} {'native |E| scale':>17s}")
print("-" * 60)
for k in keys:
    print(f"{k:12s} {np.mean(ranks[k]):12.2f} {np.mean(margins[k]):14.1f} "
          f"{np.mean(scale[k]):17.1f}")
print()
print(f"rank 1.00 = the native is always the lowest of {1 + len(SIGMAS)*N_DECOY} candidates;")
print("higher is worse. margin is how much lower the best decoy scores, so a large negative")
print("margin means the native lost badly. scale is |E(native)| in kJ/mol, shown because the")
print("four energies are not on a comparable scale.")
