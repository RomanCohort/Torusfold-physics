"""Is the stored table the fixed point of the function that reads it?

boltzmann_bonded.fit() stores U = -kBT ln p_bin, a POINT weight at each bin centre.
boltzmann_bonded._sample() evaluates that table as a PIECEWISE-LINEAR function, and the integral
of a piecewise-linear density over a bin is not the point value at its centre. So the density the
simulator actually samples is not the histogram the table was fitted to, and an IBI round against
a perfect simulator would still move the table by that difference alone.

This measures the difference, so that a first-round dU from ibi_round0 can be read for what it
contains rather than assumed to be the coupling correction.

Run: python scripts/check_table_convention.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B          # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
z = np.load(NPZ)
KBT = B.KBT
GRID = 40001

print(f"table: {NPZ.name}")
print()
print(f"{'coordinate':10s} {'nbin':>5s} {'core gap kJ/mol':>16s} {'core gap kBT':>13s} "
      f"{'all-bin max kJ/mol':>19s} {'all-bin max kBT':>16s}")
print("-" * 86)
for coord in B.COORDS:
    t = {"lo": float(z[f"{coord}__lo"]), "hi": float(z[f"{coord}__hi"]),
         "binw": float(z[f"{coord}__binw"]), "U": z[f"{coord}__U"],
         "centre": z[f"{coord}__centre"]}
    B.prepare({coord: t})
    n = len(t["U"])
    edges = np.concatenate([[t["centre"][0] - t["binw"] / 2],
                           0.5 * (t["centre"][:-1] + t["centre"][1:]),
                           [t["centre"][-1] + t["binw"] / 2]])

    # exact bin integrals of the piecewise-linear table, by fine composite grid per bin
    p_int = np.zeros(n)
    for k in range(n):
        q = np.linspace(edges[k], edges[k + 1], GRID // n + 2)
        u = B._sample(torch.tensor(q, dtype=torch.float64), t).numpy()
        p_int[k] = np.trapezoid(np.exp(-u / KBT), q)

    p_point = np.exp(-(t["U"] - t["U"].min()) / KBT)
    p_int_n = p_int / p_int.sum()
    p_point_n = p_point / p_point.sum()

    with np.errstate(divide="ignore"):
        gap = KBT * np.log(p_int_n / p_point_n)

    # the 99 percent mass core, so a near-empty tail bin cannot set the headline number
    order = np.argsort(-p_point_n)
    core_idx = order[:max(1, int(np.searchsorted(np.cumsum(p_point_n[order]), 0.99)) + 1)]
    core = np.abs(gap[core_idx]).max()

    print(f"{coord:10s} {n:5d} {core:16.3f} {core / KBT:13.3f} "
          f"{np.abs(gap).max():19.3f} {np.abs(gap).max() / KBT:16.3f}")
print()
print("This is the part of a first-round dU that is a binning convention, not coupling.")
