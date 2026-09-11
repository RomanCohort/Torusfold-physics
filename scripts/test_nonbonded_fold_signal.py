"""Can the nonbonded terms tell a correct fold from a wrong one?

The ablation left folding resting on the nonbonded terms, and those have never been
validated. The Gaussian-bead decoys used so far cannot test this: they break bonds and
create clashes, so any term that prices local geometry catches them, and the pairing terms
are not exercised at all.

This uses a length-matched decoy instead. The same structure, the same residues, the same
geometry -- only which residue is paired with which is scrambled. Nothing about local
geometry changes, so every bonded term is blind to the difference by construction. If the
pairing-dependent terms still put the correct assignment lower, they carry fold information.
If they do not, the ablations in 3t merely moved the problem down a level.

The second question is the share of the energy. A term can separate the two assignments and
still be irrelevant if it is a fraction of a percent of the total, so each term's
contribution to the total is reported alongside.

Run: python scripts/test_nonbonded_fold_signal.py [n_structs]
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
MIN_PAIRS = 6
SEED = 20260219

PAIR_TERMS = ("wc pair N-N", "bpp", "pair guide")
GEOM_TERMS = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
              "angle P-P-P", "dihedral P-P-P-P", "stacking P-P", "bsj closure",
              "bsj guide", "bsj contact", "clash")

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= MIN_PAIRS][:N]
print(f"{len(structs)} structures with at least {MIN_PAIRS} WC pairs")
print()

rng = np.random.default_rng(SEED)


def derange(pairs):
    """Same residues, same pair count, partners permuted so no pair is left intact."""
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    for _ in range(200):
        perm = rng.permutation(len(b))
        if all(perm[i] != i for i in range(len(b))):
            break
    out = []
    for i, j in enumerate(perm):
        lo, hi = (a[i], b[j]) if a[i] < b[j] else (b[j], a[i])
        out.append((lo, hi))
    return out


def energies(s, pairs):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    full, _ = C.cg_energy_forces(pos, ij, torch.ones(len(ij), dtype=torch.float64),
                                 cell_list=cl)
    _, e = FT.term_energies_forces(pos, ij, torch.ones(len(ij), dtype=torch.float64),
                                   cell_list=cl)
    return float(full), e


deltas = {k: [] for k in PAIR_TERMS + GEOM_TERMS}
magnitudes = {k: [] for k in PAIR_TERMS + GEOM_TERMS}
full_delta, totals, n_wins = [], [], 0
per_struct = []
for s in structs:
    if len(s["pairs"]) < MIN_PAIRS:
        continue
    nat_f, nat_e = energies(s, s["pairs"])
    wrong_f, wrong_e = energies(s, derange(s["pairs"]))
    for k in deltas:
        deltas[k].append(wrong_e[k] - nat_e[k])
        magnitudes[k].append(abs(nat_e[k]))
    full_delta.append(wrong_f - nat_f)
    totals.append(abs(nat_f))
    per_struct.append((s["name"], len(s["pairs"]), wrong_f - nat_f))
    if wrong_f > nat_f:
        n_wins += 1

n = len(full_delta)
print(f"{n} structures; wrong-pairing minus native, kJ/mol")
print()
print(f"{'term':22s} {'mean delta':>12s} {'favours native':>15s} "
      f"{'mean |E|':>11s} {'share':>8s}")
print("-" * 74)
allterms = PAIR_TERMS + GEOM_TERMS
for k in allterms:
    d = np.array(deltas[k])
    # an earlier version printed mean|delta| / mean|total| under the heading "share of
    # total", which is a sensitivity not a share. Both are shown now.
    mag = float(np.mean(magnitudes[k]))
    share = mag / float(np.mean(totals)) * 100
    tag = "  <- pairing" if k in PAIR_TERMS else ""
    print(f"{k:22s} {d.mean():12.1f} {int((d > 0).sum()):>10d}/{n:<4d} "
          f"{mag:11.1f} {share:7.2f}%{tag}")
print("-" * 82)
d = np.array(full_delta)
print(f"{'FULL (includes GB/SA)':22s} {d.mean():12.1f} {n_wins:>15d}/{n:<12d}")
print()
print("a positive delta means the term scores the correct pairing LOWER, i.e. it has fold")
print("signal. 'share of total' is the term's mean |energy| over the full energy's.")
print()
print("the five most decisive structures by total:")
per_struct.sort(key=lambda t: t[2])
for name, npr, dd in per_struct[:2]:
    print(f"  {name:10s} {npr:3d} pairs  wrong-native {dd:+12.1f}")
for name, npr, dd in per_struct[-3:]:
    print(f"  {name:10s} {npr:3d} pairs  wrong-native {dd:+12.1f}")
