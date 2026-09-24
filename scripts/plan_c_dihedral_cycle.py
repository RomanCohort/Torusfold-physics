"""Why the dihedral limit-cycles: the refit's own step is a growing 2-cycle, not noise.

Reads only what is on disk -- results/plan_c/fields/c2stab/C2s8_r{0..4}.npz (the fields the loop
injected) and results/plan_c/ensembles_c2stab.npz (the pooled histogram each round produced). No
sampling, so it answers a question the sampled runs raised without costing a core-hour.

MEASURED 2026-09-24, C2s8, K=8, gain 1.0, per coordinate:

  coord      round  |dU|max   rms_p   corr(step_r, step_r-1)  modes  edge_mass
  bb_bond        1    23.227   2.572              --            213     0.0061
                 2     9.144   1.410          -0.163            309     0.1468
                 3     3.232   0.522          +0.268            323     0.1293
                 4     1.660   0.229          -0.279            324     0.1133
  angle          1    24.211   2.082              --            143     0.1149
                 2    12.067   1.129          -0.211            179     0.1657
                 3    10.380   0.439          +0.242            137     0.1832
                 4     7.060   0.346          +0.442             74     0.2023
  dihedral       1    13.429   4.080              --              4     0.4949
                 2     6.384   2.716          -0.365              2     0.3703
                 3     7.794   3.418          -0.853              2     0.4214
                 4     9.363   4.058          -0.936              2     0.3136

(rms_p is the mass-weighted rms of the injected step in kJ/mol; corr is the Pearson correlation
between the shape of this round's step and the previous round's, on the same bin grid; modes are
local maxima holding at least 10 percent of the peak density; edge_mass is the mass in the outer 5
percent of the support on each side.)

WHAT IT SAYS. The dihedral's step does not decorrelate: by round 3 it is almost exactly MINUS the
step before it (-0.853, then -0.936), and its mass-weighted size GROWS (2.72 -> 3.42 -> 4.06 kJ/mol,
i.e. past one kBT) while bb_bond's and angle's fall to 0.23 and 0.35. A shape correlation near -1
with a growing magnitude is an unstable period-2 orbit of the refit map -- the fit overshoots, the
sampler moves the population the other way, the next fit overshoots back -- and it is not the
signature of sampling noise, which would decorrelate the shapes (corr ~ 0) instead.

WHY THE DIHEDRAL AND NOT THE OTHERS. Its target is not the unimodal-ish shape the other two have:
the generated dihedral marginal is effectively BIMODAL (2 local maxima above 10 percent of the peak
against 74-324 for the other coordinates) and 31-49 percent of its mass sits in the outer 5 percent
of the support, against 11-20 percent for angle and 0.6-15 percent for bb_bond. A K=4-8 Chebyshev
replacement, which is smooth and global, has to move that mass back and forth between the two
regions each round: the period-2 orbit is what a full-strength replacement of a bimodal target by a
smooth low-order basis looks like.

=> The fix the evidence points at is damping the dihedral's replacement (a gain below 1/|lambda|;
the measured growth ratio is about 1.2, so anything below ~0.8 leaves the unstable region and the
0.1-0.2 band that the gain sweep already flagged sits far inside it), or replacing the dihedral's
update with the table inversion the production loop uses, which reaches 1.015 on the full pool.
"""
import numpy as np

FIELDS = "results/plan_c/fields/c2stab/C2s8_r%d.npz"
ENSM = "results/plan_c/ensembles_c2stab.npz"
COORDS = ("bb_bond", "angle", "dihedral")
ROUNDS = (1, 2, 3, 4)


def U_of(z, c):
    for key in (f"{c}__U", f"U__{c}", f"{c}_U"):
        if key in z.files:
            return np.asarray(z[key], dtype=float)
    raise KeyError(f"no U for {c} in {list(z.files)[:8]}")


def main():
    z = np.load(ENSM, allow_pickle=True)
    fields = {r: np.load(FIELDS % r, allow_pickle=True) for r in (0,) + ROUNDS}
    for c in COORDS:
        Us = {r: U_of(fields[r], c) for r in fields}
        n_bins = len(Us[0])
        print(f"=== {c} ===")
        print(f"  {'r':>2} {'|dU|max':>9} {'rms_p':>8} {'corr(step_r, step_r-1)':>23} "
              f"{'modes':>6} {'edge_mass':>10} {'outside':>9}")
        prev = None
        for r in ROUNDS:
            cnt = np.asarray(z[f"C2s8_r{r}__{c}"], dtype=float)
            n_tot = int(z[f"C2s8_r{r}__n_total__{c}"])
            n_out = int(z[f"C2s8_r{r}__n_outside__{c}"])
            p = cnt / cnt.sum() if cnt.sum() else cnt
            dU = Us[r] - Us[r - 1]
            mean = float((p * dU).sum())
            rms = float(np.sqrt(max((p * (dU - mean) ** 2).sum(), 0.0)))
            corr = float("nan") if prev is None else float(np.corrcoef(dU, prev)[0, 1])
            peak = p.max()
            loc = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > 0.10 * peak))[0] + 1
            k = max(1, n_bins // 20)
            print(f"  {r:>2} {np.abs(dU).max():9.3f} {rms:8.3f} {corr:23.3f} "
                  f"{len(loc):6d} {float(p[:k].sum() + p[-k:].sum()):10.4f} {n_out/n_tot:9.5f}")
            prev = dU
        print()


if __name__ == "__main__":
    main()
