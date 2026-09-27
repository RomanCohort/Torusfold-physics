"""Is the angle's table narrowing toward a fixed point, or is the update being absorbed?

THE OBSERVATION (commit 3072a36, Part 7). Under the moment operator at full pool the angle's table
implied sigma falls every round -- 0.15112, 0.14390, 0.13583, 0.13065 over rounds 5-8 -- while the
other three coordinates move by under 2 percent, and the simulated marginal's ratio to it rises to
2.4. Two readings fit the same numbers and they call for opposite things:

  FIXED POINT: the sampled marginal is not the table's own Boltzmann distribution -- it is that
  distribution convolved with the rest of the Hamiltonian. Matching the sampled histogram's moments
  to the reference's then asks the table to compensate, by itself, for a width the coupling adds, so
  a narrow table IS the fixed point and the ratio is a statement about the coupling.

  DRIFT: the operator's moment residual stops falling while its corrections keep being applied
  (|d<T>|max 0.0470 -> 0.0223 -> 0.0214, ratio 0.96 in the last step, against the histogram's own
  noise floor of order 1e-4 at 3.3e8 samples). An update that no longer moves the sampled moments
  but keeps moving the table has no fixed point at all: the correction is absorbed, and the table
  narrows until something else stops it.

THE DISCRIMINATOR, and it needs no sampling: does the SAMPLED width keep falling with the table?
If yes, the update is working and the sequence is a slow convergence; if the sampled width flattens
while the table keeps narrowing, the correction is being absorbed and the loop has no fixed point on
this coordinate. Everything below is read off the task files and tables already on disk.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import ibi_core as IC          # noqa: E402

OUT = REPO / "results" / "ibi_relax"
ARCH = OUT / "table_operator_archive"
REF = REPO / "results" / "refit_smooth5.npz"
COORDS = ("bb_bond", "angle", "dihedral", "stack")
ROUNDS = (5, 6, 7, 8)


def pooled(round_n, coord, tab):
    """Pooled histogram moments of one round: sigma, edge mass (outer 5 percent), n, n_outside."""
    tot = None
    n_out = 0
    for f in sorted((OUT / f"tasks_r{round_n}").glob("*.npz")):
        with np.load(f) as z:
            c = z[f"counts__{coord}"].astype(np.float64)
            n_out += int(z[f"n_outside__{coord}"])
        tot = c if tot is None else tot + c
    centre = np.asarray(tab[coord]["centre"], dtype=np.float64)
    p = tot / tot.sum()
    mean = float((p * centre).sum())
    sig = float(np.sqrt(max((p * (centre - mean) ** 2).sum(), 0.0)))
    k = max(1, len(p) // 20)
    return sig, float(p[:k].sum() + p[-k:].sum()), int(tot.sum()), n_out


def main():
    ref = IC.load_tables(str(REF))
    print("reference (the fixed target every round is compared against):")
    for c in COORDS:
        print(f"   {c:9s} stored sigma {float(ref[c]['sigma']):.5f}  implied "
              f"{IC.implied_sigma(ref[c]):.5f}")
    print()
    hdr = (f"{'rnd':>3} | {'coord':9s} | {'sigma_sim':>9} {'edge':>6} {'n_outside':>9} | "
           f"{'implied':>8} {'stored':>8} | {'sim/impl':>8} {'sim/stored':>10}")
    print(hdr)
    print("-" * len(hdr))
    for rnd in ROUNDS:
        tabs = IC.load_tables(str(OUT / f"tables_r{rnd}.npz"))
        for c in COORDS:
            sig, edge, n, n_out = pooled(rnd, c, tabs)
            imp = IC.implied_sigma(tabs[c])
            sto = float(tabs[c]["sigma"])
            print(f"{rnd:3d} | {c:9s} | {sig:9.5f} {edge:6.3f} {n_out:9d} | {imp:8.5f} "
                  f"{sto:8.5f} | {sig/imp:8.3f} {sig/sto:10.3f}")
        print()
    # WHERE the angle table narrows: the U profile of the four tables, on the round-5 grid.
    t0 = IC.load_tables(str(OUT / "tables_r5.npz"))["angle"]
    x = np.asarray(t0["centre"], dtype=np.float64)
    U = {r: np.asarray(IC.load_tables(str(OUT / f"tables_r{r}.npz"))["angle"]["U"], dtype=np.float64)
         for r in ROUNDS}
    print("angle table: U at fixed points of the support, and the round-to-round change (kJ/mol)")
    idx = [0, len(x) // 10, len(x) // 4, len(x) // 2, 3 * len(x) // 4, 9 * len(x) // 10, len(x) - 1]
    print("   bin      x(nm)   " + "".join(f"{('U_r%d' % r):>10}" for r in ROUNDS))
    for i in idx:
        print(f"   {i:5d} {x[i]:9.4f}   " + "".join(f"{U[r][i]:10.3f}" for r in ROUNDS))
    print()
    deltas = {r: U[r] - U[r - 1] for r in ROUNDS[1:]}
    for r in ROUNDS[1:]:
        d = deltas[r]
        print(f"   U_r{r} - U_r{r-1}: max {d.max():8.3f}  min {d.min():8.3f}  "
              f"at the mode {d[np.argmin(np.abs(x - x[np.argmax(-U[r])]))]:8.3f}  "
              f"at the edges {d[idx[0]]:7.3f} / {d[idx[-1]]:7.3f}")
    # DRIFT OR CYCLE: a fixed-point iteration that overshoots alternates (corr ~ -1); one that has
    # no fixed point walks in the same direction every round (corr ~ +1) on top of the overshoot.
    print()
    print("   the step vectors: correlation between consecutive steps, and the mean drift")
    ks = ROUNDS[1:]
    for a, b in zip(ks, ks[1:]):
        print(f"     corr(U_r{a}-U_r{a-1}, U_r{b}-U_r{b-1}) = "
              f"{float(np.corrcoef(deltas[a], deltas[b])[0, 1]):+.3f}")
    drift = (U[ROUNDS[-1]] - U[ROUNDS[0]]) / (len(ROUNDS) - 1)
    # The scatter of the individual steps about that mean drift: small scatter means the loop is
    # walking in ONE direction (no fixed point); large scatter on top of it means a cycle with a
    # drift superimposed.
    spread = np.array([deltas[r] - drift for r in ks])
    print(f"     mean drift per round: max |{np.abs(drift).max():.3f}| at bin "
          f"{int(np.argmax(np.abs(drift)))} (x={x[int(np.argmax(np.abs(drift)))]:.3f}), "
          f"rms {float(np.sqrt((drift ** 2).mean())):.3f}")
    print(f"     scatter of the individual steps about that drift: rms "
          f"{float(np.sqrt((spread ** 2).mean())):.3f} per bin, "
          f"largest {np.abs(spread).max():.3f}")


if __name__ == "__main__":
    main()
