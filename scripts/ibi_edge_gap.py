"""The edge-gap diagnostic the basis project recommends, applied to the production record.

The basis arms (2026-09-27) say the quantity that predicts whether a fit will cycle is not its
residual but its EDGE GAP: the outer 5 percent mass of the fitted field's implied distribution
against the target's. The only cycling arm had -0.378 (it delivered less than half the target's edge
mass); the four that converged sat inside |gap| <= 0.012, including the one with the worst residual.

If that is the right order parameter, it should say something about the production loop too, where
the angle did not cycle but DRIFTED: its moment residual stopped falling, its table's implied sigma
fell 4 percent per round while the sampled sigma fell 1.4, and the table walked in one direction
(Part 8). Both quantities come off the files on disk, so this costs nothing.

gap_r = (mass in the outer 5 percent of the table's own implied distribution)
      - (mass in the outer 5 percent of the reference distribution p_ref)
and the same for the SAMPLED histogram, which is what the sampler actually delivers.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import ibi_bonded as I                # noqa: E402
import ibi_core as IC                 # noqa: E402

OUT = REPO / "results" / "ibi_relax"
REF = REPO / "results" / "refit_smooth5.npz"
COORDS = ("bb_bond", "angle", "dihedral")
ROUNDS = (5, 6, 7, 8)
KBT = 2.494


def outer_mass(p, k):
    k = max(1, int(k))
    return float(p[:k].sum() + p[-k:].sum())


def implied_dist(tab_coord):
    U = np.asarray(tab_coord["U"], dtype=float)
    p = np.exp(-(U - U.min()) / KBT)
    return p / p.sum()


def sampled_dist(round_n, coord, tab):
    tot = None
    for f in sorted((OUT / f"tasks_r{round_n}").glob("*.npz")):
        with np.load(f) as z:
            c = z[f"counts__{coord}"].astype(np.float64)
        tot = c if tot is None else tot + c
    return tot / tot.sum()


def main():
    ref_tables = I.load_clean_tables(REF)
    n_bins = len(np.asarray(ref_tables["bb_bond"]["U"]))
    k = n_bins // 20
    print(f"outer 5 percent = {k} bins at each end; reference edge mass, and the gaps against it")
    print()
    hdr = (f"{'rnd':>3} | {'coord':9s} | {'ref edge':>8} | {'table edge':>10} {'gap_table':>10} | "
           f"{'sampled edge':>12} {'gap_sampled':>12}")
    print(hdr)
    print("-" * len(hdr))
    for c in COORDS:
        ref_edge = outer_mass(I.probability_from_table(ref_tables[c]), k)
        for rnd in ROUNDS:
            tabs = IC.load_tables(str(OUT / f"tables_r{rnd}.npz"))
            t_edge = outer_mass(implied_dist(tabs[c]), k)
            s_edge = outer_mass(sampled_dist(rnd, c, tabs), k)
            print(f"{rnd:3d} | {c:9s} | {ref_edge:8.4f} | {t_edge:10.4f} "
                  f"{t_edge - ref_edge:+10.4f} | {s_edge:12.4f} {s_edge - ref_edge:+12.4f}")
        print()


if __name__ == "__main__":
    main()
