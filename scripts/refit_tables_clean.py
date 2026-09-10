"""Refit the Boltzmann tables on cleaned data, and derive k = kBT/sigma^2 for the locals.

Two things at once, because they share the same measurement.

The tables in 3o/3p were fitted through boltzmann_bonded.load_structures, which kept only
A/C/G/U and so joined across residue-numbering gaps -- 93 of 7354 consecutive pairs were
residues apart and were being counted as backbone bonds, angles and dihedrals. The loader
now splits at gaps, so every table number reported so far is stale.

Alongside that, the criterion for the harmonic terms. A harmonic restraint of stiffness k on
a coordinate with equilibrium variance sigma^2 produces a distribution of variance kBT/k, so
reproducing the observed spread requires k = kBT/sigma^2. That is the only criterion so far
that comes from the data rather than from tuning, and 3p showed the six shipped constants
disagree with it in both directions, spanning about 5400-fold.

Saves results/boltzmann_tables_clean.npz so the sampler does not refit each run.

Run: python scripts/refit_tables_clean.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
OUT = Path(__file__).resolve().parent.parent / "results" / "boltzmann_tables_clean.npz"
KBT = B.KBT

K_HARM = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA, "intra_cn": C.K_INTRA,
          "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}
LOCAL = ("bb_bond", "intra_pc", "intra_cn")

structs = B.load_structures(limit=N)
print(f"{len(structs)} cleaned chains, lengths "
      f"{min(len(s['pos']) for s in structs)}-{max(len(s['pos']) for s in structs)}")
print()

values = {name: [] for name in B.COORDS}
for s in structs:
    pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
    for name in B.COORDS:
        values[name].append(B.coords_of(pos, name).reshape(-1).numpy())
values = {k: np.concatenate(v) for k, v in values.items()}

print(f"{'coordinate':12s} {'n':>7s} {'sigma':>9s} {'kBT/sigma^2':>12s} "
      f"{'shipped k':>10s} {'shipped/that':>13s}")
print("-" * 70)
ratios = {}
for name in B.COORDS:
    v = values[name]
    sd = float(v.std())
    k_match = KBT / sd ** 2
    ratios[name] = K_HARM[name] / k_match
    print(f"{name:12s} {len(v):7d} {sd:9.4f} {k_match:12.1f} {K_HARM[name]:10.1f} "
          f"{ratios[name]:13.4f}")
print()
lo = min(ratios.values())
hi = max(ratios.values())
print(f"shipped k over the data-derived k spans {lo:.4f} to {hi:.1f}, a factor of {hi/lo:,.0f}")
print("values below 1 mean the shipped spring is too soft to reproduce the observed spread;")
print("values above 1 mean it is too stiff.")
print()
print("the three local coordinates are the ones proposed to keep a harmonic form:")
for name in LOCAL:
    v = values[name]
    print(f"  {name:10s} k {K_HARM[name]:7.1f} -> {KBT/v.std()**2:9.1f}  "
          f"({ratios[name]:.4f}x)")
print()

tables = B.prepare(B.fit(structs))
print(f"{'coordinate':12s} {'argmin U':>10s} {'data mode':>10s} {'U range':>9s} "
      f"{'empty bins':>11s}")
print("-" * 58)
for name in B.COORDS:
    t = tables[name]
    v = values[name]
    h, e = np.histogram(v, bins=60)
    mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    argmin = float(t["centre"][int(np.argmin(t["U"]))])
    print(f"{name:12s} {argmin:10.3f} {mode:10.3f} "
          f"{t['U'].max() - t['U'].min():9.1f} {t['empty']:6d}/{len(t['U']):<4d}")
print()
print("U range is the table's depth in kJ/mol; a few kBT is expected (kBT = "
      f"{KBT:.2f} kJ/mol at 300 K)")
print()

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT,
         **{f"{name}__{field}": np.asarray(tables[name][field])
            for name in B.COORDS for field in ("lo", "hi", "binw", "U", "centre")},
         **{f"{name}__kBT_over_sigma2": np.asarray(KBT / values[name].std() ** 2)
            for name in B.COORDS},
         **{f"{name}__sigma": np.asarray(values[name].std()) for name in B.COORDS})
print(f"saved {OUT}")
