"""Does the statistical-potential-derived field carry sequence information?

The claim under test. The field's targets and stiffnesses came from Boltzmann-inverted
distributions (scripts/boltzmann_bonded.py). cgRNASP-style statistical potentials are indexed
by base-type pair, so "derived from statistical potentials" is often read as "sequence
dependent". Ours were fitted by pooling every observation into one histogram, so the question
is whether pooling destroyed real information or whether there was none to destroy.

Part A: for each fitted coordinate, how much of its variance is explained by base identity
        (eta^2), against a shuffled-label null.
Part B: with coordinates and pairs fixed, zeroing pair_w -- the only channel the pipeline
        ever fills from the sequence -- how many of the 13 terms move?
Part C: the sigma that sets K_BPP, decomposed into within-base-type and between-base-type.

Run: python scripts/test_sequence_dependence.py [n_files]
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import cg_force_terms as T            # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")
WCP = B.WCP


def label_sets(names):
    """{coordinate: [(label_vector, label_name), ...]} for one chain."""
    L = len(names)
    n = np.array(names)
    out = {
        "bb_bond":  [(n[:-1] + n[1:], "step i,i+1")],
        "intra_pc": [(n, "base i")],
        "intra_cn": [(n, "base i")],
        "angle":    [(n[:-2] + n[1:-1] + n[2:], "triplet"),
                     (n[1:-1], "base i+1")],
        "dihedral": [(n[1:-2] + n[2:-1], "central step")],
        "stack":    [(n[:-2] + n[2:], "step i,i+2")],
    }
    for k, v in out.items():
        for lab, _ in v:
            assert len(lab) == {"bb_bond": L - 1, "intra_pc": L, "intra_cn": L,
                                "angle": L - 2, "dihedral": L - 3,
                                "stack": L - 2}[k], (k, len(lab), L)
    return out


def eta2(vals, labs, rng, n_null=30):
    """Fraction of variance explained by the label, plus a shuffled-label null."""
    def raw(lab):
        mu = vals.mean()
        sst = ((vals - mu) ** 2).sum()
        if sst <= 0:
            return 0.0
        ssb = 0.0
        for g in np.unique(lab):
            m = lab == g
            ssb += m.sum() * (vals[m].mean() - mu) ** 2
        return float(ssb / sst)
    real = raw(labs)
    null = float(np.mean([raw(rng.permutation(labs)) for _ in range(n_null)]))
    return real, null


# ------------------------------------------------------------------ part A
vals = defaultdict(list)
labs = defaultdict(lambda: defaultdict(list))
nn_by_pair = defaultdict(list)
nn_all = []
files = sorted(glob.glob(str(DATA / "*.pdb")))[:NF]
used = 0
for f in files:
    for beads, pairs, names in B._chain_residues(f, with_names=True):
        pos = torch.tensor(beads.reshape(1, -1, 3), dtype=torch.float64)
        ls = label_sets(names)
        for coord in B.COORDS:
            v = B.coords_of(pos, coord).reshape(-1).numpy()
            vals[coord].append(v)
            for lab, lname in ls[coord]:
                labs[coord][lname].append(lab)
        for a, b in pairs:
            r = float(np.linalg.norm(beads[a, 2] - beads[b, 2]))
            nn_all.append(r)
            nn_by_pair["".join(sorted((names[a], names[b])))].append(r)
        used += 1

print(f"{used} chains from {len(files)} files")
print()
print("=== Part A: variance of each fitted coordinate explained by base identity ===")
print(f"{'coordinate':10s} {'labeling':16s} {'n':>7s} {'eta^2':>8s} {'shuffled':>9s} {'ratio':>7s}")
print("-" * 62)
rng = np.random.default_rng(7)
for coord in B.COORDS:
    v = np.concatenate(vals[coord])
    for lname, chunks in labs[coord].items():
        lab = np.concatenate(chunks)
        real, null = eta2(v, lab, rng)
        print(f"{coord:10s} {lname:16s} {len(v):7d} {real:8.4f} {null:9.4f} "
              f"{(real / null if null > 0 else float('nan')):7.2f}")
print()
print("  The tables are fitted on the pooled column, so eta^2 is the fraction of the")
print("  distribution's shape that pooling threw away. A ratio near 1 means the base")
print("  letters in that label carry no information about the coordinate.")

# ------------------------------------------------------------------ part B
print()
print("=== Part B: zeroing pair_w, with coordinates and pairs untouched ===")
structs = [s for s in B.load_structures(limit=300) if len(s["pairs"]) >= 4][:10]
acc = defaultdict(float)
moved = defaultdict(int)
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    ones = torch.ones(len(ij), dtype=torch.float32)
    zero = torch.zeros(len(ij), dtype=torch.float32)
    f1, e1 = T.term_energies_forces(pos, ij, ones)
    f0, e0 = T.term_energies_forces(pos, ij, zero)
    for k in e1:
        acc[k] += abs(e1[k] - e0[k])
        # a force component moves if it differs at all, so compare the raw arrays
        moved[k] += int((np.asarray(f1[k], dtype=np.float64)
                         != np.asarray(f0[k], dtype=np.float64)).sum())
for k in sorted(acc, key=lambda x: -acc[x]):
    print(f"{k:18s} dE={acc[k]:14.6f}   force components that move: {moved[k]:5d}")
print()
print("  Zero force components means the term is bit-identical with and without the")
print("  sequence weight, i.e. the field cannot see the sequence through it.")

# ------------------------------------------------------------------ part C
print()
print("=== Part C: the sigma behind K_BPP, split by base pair type ===")
nn_all = np.array(nn_all)
print(f"pooled: n={len(nn_all)} mean={nn_all.mean():.4f} sigma={nn_all.std():.4f} nm")
print()
print(f"{'pair':6s} {'n':>6s} {'mean':>8s} {'sigma':>8s} {'K_BPP if this sigma':>20s}")
print("-" * 54)
gm, gn = [], []
for k in sorted(nn_by_pair):
    a = np.array(nn_by_pair[k])
    gm.append(a.mean())
    gn.append(len(a))
    print(f"{k:6s} {len(a):6d} {a.mean():8.4f} {a.std():8.4f} "
          f"{0.6 * B.KBT / a.std():20.1f}")
gm = np.array(gm)
gn = np.array(gn)
mu = nn_all.mean()
s_between = float(np.sqrt((gn * (gm - mu) ** 2).sum() / gn.sum()))
w = []
for k in sorted(nn_by_pair):
    a = np.array(nn_by_pair[k])
    w.append(((a - a.mean()) ** 2).sum())
s_within = float(np.sqrt(sum(w) / gn.sum()))
print()
print(f"sigma_between base-pair types = {s_between:.4f} nm")
print(f"sigma_within  base-pair types = {s_within:.4f} nm")
print(f"pooled                        = {nn_all.std():.4f} nm")
print(f"shipped K_BPP = 0.6*kBT/{nn_all.std():.4f} = {0.6 * B.KBT / nn_all.std():.1f}"
      f"   (torch_cgsim has {C.K_BPP})")
print()
gu_au = float(np.mean(nn_by_pair["GU"]) - np.mean(nn_by_pair["AU"]))
print(f"  base-pair type explains {100 * (s_between / nn_all.std()) ** 2:.1f} percent of the")
print(f"  N-N variance. The largest mean shift, GU minus AU, is {gu_au:.4f} nm = "
      f"{gu_au / nn_all.std():.2f} pooled sigma.")
print("  Pooling is not free here: G-U wobble pairs really are wider, and the flat")
print("  K_BPP prices them as if they were not.")
