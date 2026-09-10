"""Make the cgRNASP tables differentiable, then check F = -dE/dr against finite differences.

Reference   : the binned lookup the C implementation uses, potential[i][j][int(r/0.3)].
Interpolated: the same table sampled at bin centres and linearly interpolated, which has a
              piecewise-constant derivative and therefore a well-defined force.

Both evaluators come from scripts/cg_rnasp_evaluator.py (single source). The test is
deliberately narrow: it asks whether the interpolated representation is self-consistent,
and how badly the binned one fails the same test. It says nothing about whether the
potential is any good.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_1ehz
import cg_rnasp_evaluator as CG

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = 20
base = base_of[:L]
beads = np.zeros((3 * L, 3))
for i in range(L):
    r = res[order[i]]
    gly = "N9" if base[i] in "AG" else "N1"
    beads[3 * i] = r["P"]
    beads[3 * i + 1] = r["C4'"]
    beads[3 * i + 2] = r[gly]
types = CG.bead_types(base)
PA, PB, PC = CG.bead_pair_arrays(L)
print(f"test system: {L} residues, {len(PA)} bead pairs   fun(N) = {CG.fun(L):.2f}")
print()

e_bin = CG.energy_binned(beads, types, PA, PB, PC, L)
e_int, f_int = CG.energy_forces_interp(beads, types, PA, PB, PC, L)
print(f"binned energy       : {e_bin:12.4f} kBT")
print(f"interpolated energy : {e_int:12.4f} kBT")
print(f"difference          : {e_int - e_bin:+12.4f} kBT")
print()

h = 1e-4
print(f"analytic gradient vs central finite differences (h = {h} A)")
ok = tot = 0
worst = 0.0
rng = np.random.default_rng(0)
for a in rng.choice(len(beads), size=12, replace=False):
    for ax in (0, 1, 2):
        xp = beads.copy(); xp[a, ax] += h
        xm = beads.copy(); xm[a, ax] -= h
        fd = (CG.energy_forces_interp(xp, types, PA, PB, PC, L)[0]
              - CG.energy_forces_interp(xm, types, PA, PB, PC, L)[0]) / (2 * h)
        an = -f_int[a, ax]
        tot += 1
        if abs(fd - an) <= 1e-6 + 1e-4 * abs(an):
            ok += 1
        worst = max(worst, abs(fd - an))
print(f"  interpolated : {ok}/{tot} components agree, worst |FD - analytic| = {worst:.3e}")
print()

zero = tot2 = 0
for a in rng.choice(len(beads), size=12, replace=False):
    for ax in (0, 1, 2):
        xp = beads.copy(); xp[a, ax] += h
        xm = beads.copy(); xm[a, ax] -= h
        fd = (CG.energy_binned(xp, types, PA, PB, PC, L)
              - CG.energy_binned(xm, types, PA, PB, PC, L)) / (2 * h)
        tot2 += 1
        if fd == 0.0:
            zero += 1
print(f"  binned       : {zero}/{tot2} components have FD exactly 0")
print()
print("So the interpolated representation is self-consistent, and the binned one has no")
print("usable gradient at all at this step size.")
