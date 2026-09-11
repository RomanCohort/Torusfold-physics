"""Can the score pick the right fold over a PLAUSIBLE wrong one?

3u showed the pairing terms respond to the pairing assignment, 24/24, but the decoy was
random re-pairing, which flings partners far apart and makes the pairing energy trivially
large. That shows responsiveness and sign, not discrimination.

The near-native decoy for RNA is a register shift: slide the partner strand by one or two
positions along a helix. The same residues stay paired to members of the same helix, so every
distance stays in a plausible band and nothing about local geometry is absurd. What changes
is which base sits opposite which -- the chemistry, and in a real molecule the fold.

This measures whether the score prefers the correct register, per term, alongside the
C1'-C1' distance distribution to show how near-native the decoy actually is.

Run: python scripts/test_register_shift_decoys.py [n_structs]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MIN_RUN = 3
SHIFTS = (1, 2)
PAIR_TERMS = ("wc pair N-N", "bpp", "pair guide")
OTHER_TERMS = ("bb bond P-P", "intra P-C4'", "intra C4'-N", "angle P-P-P",
               "dihedral P-P-P-P", "stacking P-P", "bsj closure", "bsj guide",
               "bsj contact", "clash")


def helix_runs(pairs):
    """Runs of consecutive stacking pairs, a(k) increasing as b(k) decreases."""
    ps = sorted(pairs, key=lambda p: p[0])
    if not ps:
        return []
    runs, cur = [], [ps[0]]
    for p in ps[1:]:
        a0, b0 = cur[-1]
        a1, b1 = p
        if a1 == a0 + 1 and b1 == b0 - 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    return [r for r in runs if len(r) >= MIN_RUN]


def shift(pairs, s):
    """Shift every run's partner strand by s, dropping the tail of each run."""
    out, kept = [], 0
    for run in helix_runs(pairs):
        a = [p[0] for p in run]
        b = [p[1] for p in run]
        for i in range(len(a) - s):
            j = b[i + s]
            if a[i] == j:
                continue
            out.append((min(a[i], j), max(a[i], j)))
        kept += len(a) - s
    if kept < 3:
        return None
    if len(set(out)) != len(out):
        return None
    return sorted(out)


def c1_distances(s, pairs):
    """C1'-C1' distances need C1', which the loader does not keep; use P-P instead.

    P-P across a pair is about twice C1'-C1' but it is the same comparison between native
    and decoy, which is what matters here.
    """
    P = s["pos"][:, 0, :]
    return np.array([np.linalg.norm(P[a] - P[b]) for a, b in pairs])


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


allstructs = B.load_structures(limit=400)
usable = []
for s in allstructs:
    if len(helix_runs(s["pairs"])) == 0:
        continue
    d = shift(s["pairs"], 1)
    if d is not None and len(d) >= 3:
        usable.append(s)
    if len(usable) >= N:
        break
print(f"{len(usable)} structures with at least one helix run of {MIN_RUN}+ pairs")
print()

deltas = {k: [] for k in PAIR_TERMS + OTHER_TERMS}
full_delta, wins = [], 0
nat_nn, dec_nn = [], []
for s in usable:
    nat_f, nat_e = energies(s, s["pairs"])
    nat_nn.append(c1_distances(s, s["pairs"]))
    for sh in SHIFTS:
        d = shift(s["pairs"], sh)
        if d is None:
            continue
        dec_f, dec_e = energies(s, d)
        dec_nn.append(c1_distances(s, d))
        for k in deltas:
            deltas[k].append(dec_e[k] - nat_e[k])
        full_delta.append(dec_f - nat_f)
        if dec_f > nat_f:
            wins += 1

n = len(full_delta)
nat_nn = np.concatenate(nat_nn)
dec_nn = np.concatenate(dec_nn)
print(f"{n} native/decoy comparisons over shifts {SHIFTS}")
print(f"P-P across a pair: native {nat_nn.mean():.3f} +/- {nat_nn.std():.3f} nm, "
      f"decoy {dec_nn.mean():.3f} +/- {dec_nn.std():.3f} nm")
print(f"  decoy partners outside 0.8-1.4 nm: {(~((dec_nn > 0.8) & (dec_nn < 1.4))).mean()*100:.1f}%"
      f"  (native {(~((nat_nn > 0.8) & (nat_nn < 1.4))).mean()*100:.1f}%)")
print()
print(f"{'term':22s} {'mean delta':>12s} {'favours native':>15s}")
print("-" * 54)
for k in PAIR_TERMS + OTHER_TERMS:
    d = np.array(deltas[k])
    tag = "  <- pairing" if k in PAIR_TERMS else ""
    print(f"{k:22s} {d.mean():12.1f} {int((d > 0).sum()):>10d}/{n:<4d}{tag}")
print("-" * 54)
d = np.array(full_delta)
print(f"{'FULL (includes GB/SA)':22s} {d.mean():12.1f} {wins:>10d}/{n:<4d}")
print()
print("a positive delta means the term scores the correct register LOWER.")
print("This decoy keeps partners inside the same helix, so it cannot be dismissed on")
print("geometry alone -- which is the point.")
