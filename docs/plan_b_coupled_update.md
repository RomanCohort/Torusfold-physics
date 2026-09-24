# Plan B — change the update operator, not the sampler

*One of two independent tracks opened on 2026-09-21, after four rounds of the closed IBI loop on the
867-chain pool. Plan C (`plan_c_selfconsistent_target.md`) changes the TARGET; this one keeps the
target and changes how the correction is computed. They are separate experiments and neither
depends on the other.*

## What the four rounds measured

| round | chains | bb_bond | angle | dihedral | stack | pooled J | per-chain median J | updates (max|dU| kJ/mol) |
| --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| 0 | 867 | 0.957 | 1.299 | 1.233 | 1.164 | 0.167 | 0.1745 | bb 2.84 / ang 3.00 / dih 5.62 |
| 1 | 867 | 0.862 | 1.265 | 1.072 | 1.107 | 0.139 | 0.1567 | bb 1.70 / ang 3.94 / dih 1.82 |
| 2 | 867 | 0.855 | 1.248 | 1.023 | 1.094 | 0.123 | 0.1536 | bb 0.48 / ang 3.02 / dih 0.77 |
| 3 | 866 | 0.856 | 1.228 | 1.005 | 1.083 | 0.112 | 0.1498 | (round 3 closing) |

`sim/ref = sigma_sim / sigma_ref` from the pooled histograms; the target is 1.000 per coordinate.

Two signatures, and they are different failures:

- **dihedral converged** (1.233 -> 1.005, correction 5.62 -> 0.77 kJ/mol). At this level of the
  field the dihedral is nearly decoupled from the rest, and a 1-D marginal inversion handles a
  nearly decoupled coordinate well.
- **angle oscillates without trend** (correction 3.00 -> 3.94 -> 3.02) and **bb_bond converges to
  a fixed point that is not the reference** (correction 2.84 -> 1.70 -> 0.48, ratio frozen at
  0.856). A coupled 1-D iteration is expected to do exactly this: `ibi_bonded`'s own synthetic
  test says a coupled system (J_COUPLED = 2.2) needs about four rounds, and nothing in the
  operator says the fixed point must be the reference.

## Why the current operator cannot close the gap

```
plan_update(table, hist, p_ref) -> dU = kBT * ln(P_sim(q) / P_ref(q))     per bin, per coordinate
```

Three properties of that operator matter here.

1. **It is one-dimensional.** Each coordinate is corrected against its own marginal, while the
   other two coordinates are being corrected at the same time under the same field. The correction
   each coordinate receives is therefore computed against a distribution that the other corrections
   are about to change.
2. **It is non-parametric.** 1000 numbers per coordinate, so the correction carries the histogram's
   Poisson noise; that noise is a random force field and is why the tables are smoothed with
   `SMOOTH_WIDTH = 5` and why friction had to go to 1.0 (unsmoothed: 588 K against a 300 K target).
3. **Its force is piecewise constant**, with an honest maximum of about 3.3x the shipped force cap
   (`dihedral_table_decision.md`), i.e. the operator can ask for forces the field is not allowed to
   deliver.

## The proposed operator: relative entropy on a low-order basis

Write each replaced term as a short expansion instead of a table,

    U_c(q) = sum_k c_k * phi_k(q),      phi_k = Chebyshev polynomials on the coordinate's own domain

and update the coefficients, not the bins:

    c_k  <-  c_k + eta_k * ( <phi_k>_sim - <phi_k>_ref )

That difference IS the gradient of the relative entropy S = int P_sim ln(P_sim / P_ref), so this is
gradient descent on the quantity the loop is trying to reduce, with three practical consequences:

- **It uses MOMENTS, not histograms.** A handful of running sums per chain instead of 1000 bins, so
  there is no binning noise to smooth away and no `min_bin_obs` / `unvisited_reference_support`
  refusals to dodge (both are artefacts of binning).
- **Coupling is handled where it lives.** `<phi_k>_sim` is measured under the FULL field with the
  other coordinates' corrections in force, so a coordinate whose partner moves is corrected for the
  moved partner on the next round rather than fighting it.
- **The potential is smooth by construction**, so the force is bounded by the basis's own
  derivative — the force-cap question becomes a property of the fitted function instead of a
  property of the bin width.

The same idea is already in the repository twice: `cg_potentials` has a `fourier:K1=..,K2=..` spec,
and the hand-built Fourier N=2 dihedral arm was the one that fixed dihedral across five chains
(0.659 -> 1.060). What this plan adds is the UPDATE (coefficients from ensemble averages) and
extending it to angle and bb_bond.

## The cheapest decisive experiment

Seven-chain pool (`IBI_LOOP_POOL=small`), six rounds, everything else as round 3 had it:

| arm | operator |
| :-- | :-- |
| A (control) | the table path, as it runs today |
| B1 | Fourier/Chebyshev K=4 on angle and dihedral, table on bb_bond |
| B2 | K=8 on all three, with the existing `table_wall:2000` wall on bb_bond |

**Cost**: the 7-chain pool ran six rounds in well under an hour in the earlier gain calibration, so
this is a few hours for all three arms, against 18 h per full round.

**Acceptance, written before the run**: angle's correction series becomes monotone (today it is
3.00 -> 3.94 -> 3.02) AND no coordinate's sim/ref moves further than 0.02 per round after round 4
AND the pooled J does not rise. If angle still oscillates at K=4, the failure is not the operator's
resolution — go to K=8 and then to the conditional (2-D) form.

**What would falsify the plan**: a basis that cannot represent the marginal. The dihedral's
reference marginal is bimodal (cis/trans), so K >= 4 is the minimum and the fit residual of the
basis against the reference table must be reported per coordinate before any sampling happens.
That check is free and it is the first thing to do.

## What it needs in code

| piece | where | size |
| :-- | :-- | :-- |
| basis spec + evaluate + derivative | `cg_potentials.py` (`fourier` already exists for the cosine coordinates; bb_bond needs a distance basis) | small |
| accumulate `<phi_k>` per task instead of (or next to) the histogram | `ibi_core.run_round` | small |
| the coefficient update + its step size | `ibi_bonded` (next to `plan_update`) | medium |
| driver wiring | `ibi_loop` (`UPDATED` gains a per-coordinate operator choice) | small |
| tests | equivalence where it must be equivalent, monotonicity of the objective on synthetic data | medium |

## Risks

- **Representation.** A short basis cannot reproduce a marginal with a sharp feature; measure the
  residual against the reference table first (free).
- **Step size.** `eta` is the analogue of the gain and needs the same kind of calibration the gain
  table got; the moments make the objective measurable per round, which is what makes that cheap.
- **It does not fix the target.** If the reference is itself the wrong thing to chase (Plan C's
  position), a better operator converges to the wrong place faster. Run both.

## First free check — done 2026-09-21, and it revises the plan

Before any sampling, the obvious question: can a short Chebyshev basis in the coordinate represent
the reference tables at all? Fit by weighted least squares (weights = the reference probability),
on `refit_smooth5.npz`:

| coordinate | K=2 | K=4 | K=8 | K=12 | max \|ΔU\| inside 2 sigma (best K) |
| :-- | --: | --: | --: | --: | --: |
| bb_bond | 0.721 | 0.687 | 0.323 | **0.112** | 0.61 |
| angle | 0.944 | 0.681 | 0.325 | **0.261** | **15.5** |
| dihedral | 0.906 | 0.556 | 0.510 | **0.502** | **21.6** (and it gets WORSE with K) |
| stack | 1.487 | 0.765 | 0.255 | **0.164** | 1.19 |

(kJ/mol; 0.1 kBT = 0.25 kJ/mol. Two numbers matter and they say different things.)

1. **In the region where samples live, a short basis is enough.** The probability-weighted residual
   at K=12 is 0.11-0.50 kJ/mol = **0.04-0.20 kBT** for all four coordinates — smaller than the
   Poisson noise the tables already carry (`SMOOTH_WIDTH` exists for exactly that noise).
2. **Between the peaks it is not.** The unweighted maximum inside the 2 sigma window reaches 15-22
   kJ/mol (6-9 kBT) for angle and dihedral, and for dihedral it GROWS with K — the signature of a
   polynomial chasing a kink. The tables are piecewise linear by construction, and the coordinates
   are bimodal (a narrow cis peak against a broad trans basin), so the barrier region is a feature
   no low-order polynomial resolves. It is also the region the marginal constrains least.

**So the plan changes to B': keep the table's shape and make the CORRECTION low-order.**

    U_c(q) = table_c(q) + sum_k d_k * phi_k(q)
    d_k   <- d_k + eta_k * ( <phi_k>_sim - <phi_k>_ref )        (relative-entropy gradient)

The table keeps the peaks and the barrier (its 1000 bins buy shape, which is why the table path
currently reaches dihedral 1.005 while the hand-built Fourier N=2 arm reached 1.060); the correction
is a handful of coefficients, so the update is a moment-matching step with no binning noise and the
step size has a measurable objective. The free check above is what selects this variant: fitting the
whole potential with a basis throws away the shape that is already working.


## The ridge, swept at last — measured 2026-09-24 (867 chains, no sampling)

The step above is a Newton step against the simulation's own covariance, and the one number in it
that decides how much of a nearly-dependent direction it is allowed to act on is the ridge. It has
been hard-coded at `1e-3 * trace(Cov)/n` since the toy test that showed a 1e-8 ridge turning a
matching simulation's rounding-level moment difference into ~1 kJ/mol. Plan C's parametric arm was
stabilised by sweeping exactly this quantity (60.6 -> 23.2 kJ/mol of `max|dU|` between ridge 1e-3 and
1e-1 while the ensemble-weighted step barely moved), so it was worth asking whether the production
operator sits in a ridge-sensitive regime. It does not, and the sweep is now a parameter
(`ibi_bonded.DEFAULT_RIDGE_REL`, default unchanged) plus `scripts/ibi_moment_ridge_sweep.py`, which
reads the 867 task files of a finished round and calls the shipped function — no sampling.

Round 5, pooled over all 867 chains, K=8, gain 1.0 (all four columns in kJ/mol):

| ridge | bb_bond max\|dU\| | angle max\|dU\| | dihedral max\|dU\| | bb_bond rms | angle rms | dihedral rms |
| --: | --: | --: | --: | --: | --: | --: |
| 0 | 0.261 | 2.734 | 0.338 | 0.0145 | 0.4196 | 0.0557 |
| 1e-3 (shipped) | 0.216 | 1.634 | 0.336 | 0.0142 | 0.4187 | 0.0555 |
| 1e-2 | 0.089 | 0.914 | 0.324 | 0.0134 | 0.4136 | 0.0540 |
| 1e-1 | 0.033 | 0.664 | 0.243 | 0.0114 | 0.3799 | 0.0451 |
| 3e-1 | 0.025 | 0.610 | 0.158 | 0.0094 | 0.3330 | 0.0363 |

Four things fall out of it:

1. **`|d<T>|max` — the diagnostic the divergence guard is fed — is EXACTLY ridge-invariant**
   (0.00197 / 0.06675 / 0.00871 for the three coordinates at every ridge and every K). It is computed
   from the moment difference before the solve, so the guard's convergence signal and this choice are
   independent. What rounds 6-8 report about convergence does not depend on the ridge.
2. **The ensemble-weighted step is a fraction of kBT and does not care about the ridge.** rms step
   0.011-0.42 kJ/mol = 0.005-0.17 kBT across the whole sweep; going from the shipped 1e-3 to 1e-1
   moves it by 10-20 percent while shrinking `max|dU|` by 2.5-6.5x. The headline is tail
   oscillation, as in Plan C; for bb_bond the thin-tail share of the step's variance is 18 percent at
   the shipped ridge and 3.7 percent at 1e-1, and for dihedral it is zero.
3. **The shipped combination sits inside the stable region, and K is what would leave it.** At K=16
   the same histogram wants 91 kJ/mol on angle at ridge 0 and still 2.7 at 1e-3, against 2.7 and 0.66
   at K=8: the high-order Chebyshev functions are nearly collinear under a peaked density, and the
   ridge is the only thing standing between them and the sampler. If K ever grows, the ridge has to
   grow with it — and `max_step_kbt` (20 kBT = 50 kJ/mol) is the refusal that would catch a mistake.
4. **The step does not depend on the field being corrected at all.** The archived table-operator
   `tables_r5.npz` and the replayed moment-operator one differ by up to 9 kJ/mol in U, and every
   number in the table above is identical for both: the Newton step is a function of the bin
   geometry, the simulation's histogram and p_ref, and of nothing else. That is why the sweep is
   valid against a round whose field came from the other operator.


