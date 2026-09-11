"""Measuring the field on a legitimately closed structure, so the BSJ terms are honest.

3aa found that every whole-field number in this document includes the BSJ closure term,
which restrains P(0)-P(L-1) to a backbone bond length, while the test structures are linear
and have their ends far apart. That term therefore fires on a violation that would not exist
in the real construct, and it now shows as 90.7 percent of the energy budget -- an artefact
of the test geometry, not a property of the field.

A circular RNA is just a chain whose first and last residues are backbone-adjacent. So a
linear structure can be renumbered cyclically: pick a cut k, make old residue k the new
residue 0, and old residue k-1 the new residue L-1. The two ends are then genuinely bonded
and the closure restraint is satisfied, with no change to the geometry or the fold.

That is not a real circRNA -- the back-splice junction imposes a topological constraint that
changes the fold, and this is only a relabelling -- but it removes the specific artefact,
and it separates three things that were being conflated: the covalent closure, which should
now be satisfied; the BSJ guide, which pulls the two ends toward the WC pairing distance; and
the BSJ contact, which pulls residues symmetric about the junction together.

Run: python scripts/circular_field_measurement.py [n_structs]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CAP = 200.0
CACHE = Path(__file__).resolve().parent.parent / "results" / "rcm_weights.npz"
W = np.load(CACHE)["w"] if CACHE.exists() else np.full(50, 0.35)

structs = [s for s in B.load_structures(limit=300) if len(s["pairs"]) >= 4][:N]
gen = np.random.default_rng(17)


def renumber(s, k):
    """Rotate residue indices so old k becomes 0 and old k-1 becomes L-1."""
    L = len(s["pos"])
    order = [(k + m) % L for m in range(L)]
    pos = s["pos"][order]
    pos = pos * 1.0
    pairs = []
    for i, j in s["pairs"]:
        a, b = (i - k) % L, (j - k) % L
        pairs.append((min(a, b), max(a, b)))
    return {"name": s["name"], "pos": pos, "pairs": sorted(set(pairs))}


def per_term(s, w):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(w, dtype=torch.float64)
    F, E = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    full, ff = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
    return F, E, ff.reshape(3 * L, 3).numpy()


def collect(fn, label):
    acc_e, acc_f, allf = {}, {}, []
    for s in structs:
        s2 = fn(s)
        npr = len(s2["pairs"])
        w = np.asarray(W)[gen.integers(0, len(W), npr)]
        F, E, ff = per_term(s2, w)
        for k in E:
            acc_e.setdefault(k, []).append(abs(E[k]))
            acc_f.setdefault(k, []).append(np.linalg.norm(F[k], axis=1))
        allf.append(np.linalg.norm(ff, axis=1))
    allf = np.concatenate(allf)
    print(f"{label}: mean |F| {allf.mean():.2f}, p95 {np.percentile(allf,95):.2f}, "
          f"on cap {(np.abs(allf-CAP)<1e-6).mean()*100:.2f}%")
    print(f"  {'term':22s} {'mean |E|':>11s} {'p95 |F|':>10s} {'max |F|':>10s} "
          f"{'over cap':>9s}")
    print("  " + "-" * 66)
    rows = sorted(acc_e, key=lambda k: -np.mean(acc_e[k]))
    for k in rows:
        f = np.concatenate(acc_f[k])
        print(f"  {k:22s} {np.mean(acc_e[k]):11.1f} {np.percentile(f,95):10.2f} "
              f"{f.max():10.2f} {(f>CAP).mean()*100:8.2f}%")
    print()
    return allf


print(f"{len(structs)} structures, weights drawn from the measured RCM sample")
print()
print(f"{'':22s} {'linear':>12s} {'cyclically closed':>18s}")
print("-" * 54)
lin_d, clo_d = [], []
for s in structs:
    lin_d.append(np.linalg.norm(s["pos"][0] - s["pos"][-1]))
print(f"{'P(0)-P(L-1) in nm':22s} {np.mean(lin_d):12.3f} "
      f"{'(same structure)' if False else '':>18s}")
print()
collect(lambda s: s, "linear (what every earlier number used)")
collect(lambda s: renumber(s, int(gen.integers(3, len(s["pos"]) - 3))),
        "cyclically closed at a random cut")
