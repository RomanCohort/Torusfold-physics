"""Can the deposited-structure database supply an IBI target at all?

IBI corrects a potential until the SIMULATED distribution matches the REFERENCE distribution.
That only means something if the reference is the equilibrium distribution of the same system.
ours is a pool over 191 different deposited structures, one conformation each. So the width of
that pool is not a fluctuation width -- it is spread between different molecules, sequences and
conformers. An ANOVA decomposition says how much of it is which:

    var_pooled = var(structure means) + mean(structure variances)
                 \______between______/   \_____within_____/

If between dominates, the pooled sigma is a cross-molecule spread and k = kBT/sigma^2 is not a
spring constant. If within dominates, the objection is weak.

Also counts multi-MODEL files: an NMR ensemble would give a genuine within-molecule spread, and
that is the only thing in the database that could set a stiffness.

Run: python scripts/decompose_database_variance.py [n_files]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")
files = sorted(glob.glob(str(DATA / "*.pdb")))[:NF]

per_struct = {c: [] for c in B.COORDS}       # per structure: (mean, var, n)
multi = []
for f in files:
    with open(f) as fh:
        nmodel = sum(1 for line in fh if line.startswith("MODEL"))
    if nmodel > 1:
        multi.append((Path(f).stem, nmodel))
    for beads, _pairs in B._chain_residues(f):
        pos = torch.tensor(beads.reshape(1, -1, 3), dtype=torch.float64)
        for c in B.COORDS:
            v = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
            if v.size >= 2:
                per_struct[c].append((float(v.mean()), float(v.var(ddof=1)), v.size))

print(f"{len(files)} files, {len(per_struct['bb_bond'])} chains")
print(f"multi-MODEL files: {len(multi)}" +
      ("  " + ", ".join(f"{n}({m})" for n, m in multi[:10]) if multi else ""))
print()
print("=== variance decomposition of every fitted coordinate ===")
print(f"{'coordinate':10s} {'chains':>6s} {'obs':>7s} {'sigma_pool':>11s} "
      f"{'sigma_between':>14s} {'sigma_within':>13s} {'between share':>14s}")
print("-" * 80)
for c in B.COORDS:
    ps = per_struct[c]
    means = np.array([m for m, _v, _n in ps])
    vars_ = np.array([v for _m, v, _n in ps])
    ns = np.array([n for _m, _v, n in ps])
    s_between = means.std(ddof=1)
    s_within = float(np.sqrt((vars_ * (ns - 1)).sum() / (ns - 1).sum()))
    s_pool = float(np.sqrt(s_between ** 2 + s_within ** 2))
    share = s_between ** 2 / s_pool ** 2
    print(f"{c:10s} {len(ps):6d} {int(ns.sum()):7d} {s_pool:11.4f} "
          f"{s_between:14.4f} {s_within:13.4f} {share:14.3f}")
print()
print("  between share near 1  -> the pooled sigma is a cross-molecule spread,")
print("                           and k = kBT/sigma^2 is not a spring constant.")
print("  between share near 0  -> the width is local, and the objection is weak.")
print()
print("=== the shipped stiffnesses against each candidate sigma ===")
print(f"{'coordinate':10s} {'K in library':>13s} {'kBT/s_between^2':>16s} "
      f"{'kBT/s_within^2':>16s} {'kBT/s_pool^2':>14s}")
print("-" * 74)
import torusfold.scheme2.torch_cgsim as C   # noqa: E402
K = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA, "intra_cn": C.K_INTRA,
     "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}
for c in B.COORDS:
    ps = per_struct[c]
    means = np.array([m for m, _v, _n in ps])
    vars_ = np.array([v for _m, v, _n in ps])
    ns = np.array([n for _m, _v, n in ps])
    s_between = means.std(ddof=1)
    s_within = float(np.sqrt((vars_ * (ns - 1)).sum() / (ns - 1).sum()))
    s_pool = float(np.sqrt(s_between ** 2 + s_within ** 2))
    print(f"{c:10s} {K[c]:13.1f} {B.KBT / s_between ** 2:16.1f} "
          f"{B.KBT / s_within ** 2:16.1f} {B.KBT / s_pool ** 2:14.1f}")
print()
print("  sigma_pool here is the ANOVA combination, not the np.std of the concatenation;")
print("  refit_tables_clean.py used the latter, which is the same quantity when the")
print("  per-structure counts are equal.")
