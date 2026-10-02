# Plan C, second task -- making the parametrised fit (C2) stable

*Sequel to plan_c_selfconsistent_target.md. The first pass measured C0/C1 and lost C2 in
round one: max|dU| came back at 52.6 / 44.6 / 45.7 kJ/mol against the control's 8.0 / 2.9 / 5.6,
and the diagnosis written there is that a Chebyshev projection of -kBT ln p is unbounded where
the empirical density is thin and the loop injects it with nothing in the way.*

This file is what came out of acting on that diagnosis: what the instability actually was, the
five instruments now between the projection and the sampler, and what the arm does once they are
in place. Code: scripts/plan_c_loop.py, arms C2s / C2s2 / C2s4 / C2s8; the plain C2 arm is
untouched, so the old result stays reproducible.

## 1. What the instability actually was

Two faults, both measured on the real round-1 ensembles of the first c2stab attempt, and neither
of them the support problem the first pass suspected.

**(a) The weighted least squares was solved with one factor of the weight missing.** The intended
problem is min ||diag(w) A c - diag(w) U_target||_2 with w = sqrt(p), whose normal equations are
(A^T W^2 A) c = A^T W^2 U_target. The first version of the stabilised fit used A^T W U_target --
one factor of w short -- which puts the pseudo-count bins back in at sqrt(p) instead of p. On the
real data that alone gave coefficients of -720..+122 kJ/mol where the weighted Gram matrix is
perfectly regular (eigenvalues 0.005..1.73, condition 25-347), and a field that wanted a
2167-4078 kJ/mol step. The spectrum was never the problem; the residual was. The plain C2 arm
never showed this because np.linalg.lstsq(..., rcond=None) truncates the small singular
directions silently -- which is also why its first round came back as very nearly a constant.

**(b) A fit can leave the representable range, and that used to kill the run.** With the gain at
30 on the smoke protocol, ibi_bonded.bin_probabilities_from_U raised (U spans too much for
exp(-U/kBT) to be representable) *before* any record was written: the whole run died, exit 1, no
json. That is now a refusal: the previous table is kept, the round is recorded with the step that
was wanted (the message carries its max|dU|), and a C2s arm stops there.

## 2. The five instruments

Ordered by how much each one moves the answer.

1. **Support cut** (--support-frac, 1e-3 of the modal probability). Rows below the cut are
   dropped from the fit (weight 0). The basis is not masked: masking the *correction* would put a
   step in U and a delta function in its force.
2. **Taper** (--taper-decades, 2 decades). The correction is multiplied by a smoothstep in ln p
   that is 1 above the cut and 0 two decades below it, so where the ensemble has no mass the
   previous field stands. Measured on the real round-1 ensemble, the fitted polynomial dives in
   that region (bb_bond reaches -44 kJ/mol against a stored table whose own minimum is 0.2):
   without the taper the refit digs a well in a tail the sampler can still reach.
3. **Relative ridge** (--ridge, default 1e-1, lambda = ridge x trace(A^T W A) / K). Kept in
   ibi_bonded.moment_correction's own relative form, so C2s and Plan B's operator are fitted the
   same way; the *value* is measured, see below.
4. **Mass gauge.** The mass-weighted mean of the step is subtracted under the ensemble's own
   probability. A constant in U has no force anywhere, so this changes nothing the sampler does
   and makes max|dU| mean the same thing in every arm instead of reporting where the polynomial
   bottomed out -- the failure mode the first pass measured as 'a constant +49 kJ/mol with a
   spread of 1.5'.
5. **Guard.** ibi_bonded.divergence_check (patience 3, growth 2), the same instrument ibi_loop
   feeds Plan B's moment norm, fed the *force-relevant* size of the step: the mass-weighted
   standard deviation over the ensemble's own mass, not max|dU|. A refusal keeps the previous
   table, records the step that was refused, and stops the arm.

--c2-gain (default 1.0) stays as the loop's own damping knob for a fixed-point iteration that
starts walking; nothing in the results below needed it.

### The ridge is the one that matters, and its value is measured

Sweep on the real round-1 ensembles, K=8; 'applied std' is the mass-weighted standard deviation
of the step, i.e. its force-relevant size, and all numbers are kJ/mol.

| ridge | lambda | U_fit range | max dU | applied std bb/ang/dih |
| --: | --: | --: | --: | --: |
| 1e-3 | 4.7e-4 | -44.1 .. 11.1 | 60.6 | 1.69 / 1.09 / 4.63 |
| 1e-2 | 4.7e-3 | -26.4 .. 6.5 | 42.9 | 1.98 / 1.28 / 4.53 |
| **1e-1** | 4.7e-2 | -5.5 .. 5.3 | 23.2 | 2.58 / 2.09 / 4.08 |
| 1e+0 | 4.7e-1 | 0.8 .. 7.9 | 16.9 | 3.07 / 2.86 / 3.94 |

The force-relevant size of the step barely moves across three decades of ridge -- on this
ensemble the field really is 1-5 kJ/mol away from the ensemble's own Boltzmann inverse -- while
max|dU| falls by a factor of 3.5 as the tail stops oscillating. 1e-1 is the knee: the smallest
ridge whose fit stays inside the range of the target it is fitting (the target's own span is
0-22 kJ/mol), so the result is a fit rather than an extrapolation. For scale, the loop's own
operators are allowed up to DEFAULT_MAX_STEP_KBT = 20 kBT = 50 kJ/mol, so a 23 kJ/mol tail is
inside the convention the rest of the loop already runs under.

### K is not what was wrong

At the chosen ridge the sweep gives 18.9-33.2 kJ/mol of max|dU| at K=2, 29.6-33.2 at K=4 and
18.3-60.6 at K=8, with 1.1-4.6 kJ/mol of applied std in all three: the outcome is set by the ridge
and the support/taper, not by the order of the basis. The arm is run at all three anyway, because
'K=2 would have been enough' is a claim the run has to make rather than the sweep.

## 3. What the arm does now -- measured

Record: results/plan_c/plan_c_c2stab.json, fields under results/plan_c/fields/c2stab/.

### 3.1 C2s8, four rounds, same pool and seed as run1's C0/C1

max dU is the step the fit asked for; applied std is its mass-weighted standard deviation over
the ensemble's own mass (the force-relevant size, recomputed for C0/C1 from their fields on disk
in 3.2); stat mean is the mass-weighted mean of |ln p_r / p_(r-1)| between consecutive rounds'
ensembles, TV the total-variation distance between the same two, and ret the native-retention
instrument (deposited -> sampled-mean RMSD, A). No arm hit the guard.

| r | max dU bb/ang/dih | applied std bb/ang/dih | stat mean bb/ang/dih | TV bb/ang/dih | ret pool / held |
| --: | --: | --: | --: | --: | --: |
| 1 | 23.2 / 24.2 / 13.4 | 2.58 / 2.09 / 4.08 | -- | -- | 0.39 / 0.52 |
| 2 | 9.1 / 12.1 / 6.4 | 1.41 / 1.13 / 2.72 | 1.13 / 0.31 / 1.04 | 0.40 / 0.14 / 0.44 | 0.39 / 0.51 |
| 3 | 3.2 / 10.4 / 7.8 | 0.52 / 0.44 / 3.42 | 0.37 / 0.15 / 0.65 | 0.18 / 0.07 / 0.30 | 0.40 / 0.51 |
| 4 | 1.7 / 7.1 / 9.4 | 0.23 / 0.35 / 4.06 | 0.11 / 0.10 / 0.82 | 0.06 / 0.05 / 0.36 | 0.40 / 0.49 |

**The iteration has a fixed point in two of three coordinates, and the third is a limit cycle.**
bb_bond's applied step falls 2.58 -> 0.23 kJ/mol (11x) and its ensemble-to-ensemble distance falls
1.13 -> 0.11 (10x, TV 0.40 -> 0.06); the angle falls 2.09 -> 0.35 (6x) with TV 0.14 -> 0.05. The
**dihedral does not settle**: 4.08 / 2.72 / 3.42 / 4.06 kJ/mol with TV 0.44 / 0.30 / 0.36 -- the
refit wants about the same correction every round and the ensemble keeps moving by a third of its
mass each time. That is a limit cycle rather than a divergence (the guard needs a rise of 2x over
three consecutive rounds and never fired), and it is the one thing in Plan C that is still open on
this pool.

**Retention does not move under any of it**: 0.39-0.40 A (pool) and 0.49-0.52 A (holdout) across
all four rounds, against the shared start field's 0.399 / 0.500 and tables_r3's 0.411 / 0.486 -- a
refit that abandons the deposited marginal (below) still holds native geometry exactly as well as
the control, which is the acceptance criterion the plan sets.

**J_dep rises to 0.44, and that is the design rather than a regression.** The self-consistent
target is the ensemble the field produces, so the deposited marginal's residual is not what this
loop descends; C1 shows the same rise in miniature (0.137 -> 0.197), and the first pass already
argued that the criterion has to be retention plus self-consistency rather than J.

### 3.2 All arms on one footing: the step that was actually written

Recomputed from fields/run1/*.npz and ensembles_run1.npz, so C0 and C1 -- which predate the
mass-weighted columns -- get the same column as C2s. kJ/mol:

| arm | r | applied std bb/ang/dih |
| :-- | --: | --: |
| C0 (deposited target, plan_update) | 2 | 0.54 / 0.62 / 0.48 |
| | 3 | 0.31 / 0.54 / 0.41 |
| | 4 | 0.22 / 0.56 / 0.15 |
| C1 (self-consistent target, smoothed table) | 2 | 1.43 / 0.50 / 0.67 |
| | 3 | 1.07 / 0.42 / 0.33 |
| | 4 | 0.64 / 0.24 / 0.19 |
| C2s8 (self-consistent target, stabilised Chebyshev) | 1 | 2.58 / 2.09 / 4.08 |
| | 2 | 1.41 / 1.13 / 2.72 |
| | 3 | 0.52 / 0.44 / 3.42 |
| | 4 | 0.23 / 0.35 / 4.06 |

A refit moves the field further than a marginal correction does (2-4 kJ/mol against 0.15-0.6),
which is the point of the arm: the refit is what makes the target self-consistent, and it is the
only arm whose step keeps shrinking round after round.

### 3.3 K

C2s4 ran the same protocol, ridge and taper as C2s8 at K=4. Beside it, the sampled trajectories
(applied std = mass-weighted std of the step, kJ/mol):

| arm | r | applied std bb/ang/dih | stat mean bb/ang/dih | TV bb/ang/dih |
| :-- | --: | --: | --: | --: |
| C2s8 | 1 | 2.58 / 2.09 / 4.08 | -- | -- |
| | 2 | 1.41 / 1.13 / 2.72 | 1.13 / 0.31 / 1.04 | 0.40 / 0.14 / 0.44 |
| | 3 | 0.52 / 0.44 / 3.42 | 0.37 / 0.15 / 0.65 | 0.18 / 0.07 / 0.30 |
| | 4 | 0.23 / 0.35 / 4.06 | 0.11 / 0.10 / 0.82 | 0.06 / 0.05 / 0.36 |
| C2s4 | 1 | 2.57 / 2.14 / 4.28 | -- | -- |
| | 2 | 0.95 / 1.09 / 4.35 | 0.95 / 0.41 / 1.02 | 0.35 / 0.17 / 0.45 |
| | 3 | 0.39 / 0.90 / 3.50 | 0.28 / 0.29 / 1.07 | 0.14 / 0.14 / 0.43 |
| | 4 | 0.21 / 0.51 / 4.48 | 0.11 / 0.16 / 1.15 | 0.05 / 0.08 / 0.45 |

The two orders agree to about 15 per cent on every coordinate and every round, and they agree about
the conclusion: bb_bond and the angle settle, the dihedral does not -- C2s4's dihedral is if
anything the worse case (4.28 / 4.35 / 3.50 / 4.48 kJ/mol with TV 0.45 / 0.43 / 0.45, a flat limit
cycle from round two on).

K=2 was measured **offline** rather than sampled, on the same round-4 ensembles, and the sweep goes
to K=32 because the question is whether the basis can represent the histogram at all. Median
|ln(p_implied/p_hist)| (the instrument of 3.4, which is the representation question):

| coord | K=2 | K=4 | K=8 | K=12 | K=16 | K=24 | K=32 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| bb_bond | 0.241 | 0.225 | 0.284 | 0.268 | 0.289 | 0.256 | 0.252 |
| angle | 2.230 | 2.285 | 2.168 | 1.684 | 1.666 | 2.111 | 2.373 |
| dihedral | 0.837 | 0.876 | 1.399 | 1.445 | 1.392 | 1.543 | 1.429 |

Past K=8 the proxy does not improve and the wanted step grows (max|dU| 9.4 kJ/mol at K=8 against
17.4 at K=32 for the dihedral): more basis functions buy oscillation, not representation. K=2 is
mildly better for the dihedral (0.837 against 0.514 for leaving the field alone -- still worse than
doing nothing) and indistinguishable for the others, which is why the sampled C2s2 arm was started,
stopped after its first field and is reported here as a sweep rather than as a fourth trajectory.

### 3.4 Is the window a sample of the field at all? -- measured, and it bounds the proxy

The first pass's headline was that changing the target did not move retention. Before any number
above can be read as being about the *target*, the cheaper question has to be answered: at 2 ps of
burn and a 10 ps window, is the ensemble a sample of the field that produced it?

run_round cuts the window into disjoint equal-time blocks, so the drift inside it is measurable
from one round. Eight blocks of 1.25 ps over the seven-chain pool, 8 replicas, about 150k
observations per block, adjacently-paired ln-ratio medians:

| coord | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 1-last (TV) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| bb_bond | 0.110 | 0.117 | 0.116 | 0.104 | 0.100 | 0.106 | 0.102 | 0.261 (0.149) |
| angle | 0.161 | 0.132 | 0.138 | 0.131 | 0.142 | 0.134 | 0.123 | 0.267 (0.131) |
| dihedral | 0.161 | 0.143 | 0.143 | 0.136 | 0.132 | 0.134 | 0.158 | 0.400 (0.183) |

**No trend**: block 1 vs 2 differs by exactly as much as block 7 vs 8, and the first-to-last
difference is the sqrt(7) accumulation of seven independent differences (0.13 x 2.65 = 0.35), which
is what a random walk of noise looks like and not what a relaxation looks like. The absolute level
(0.10-0.16) is the correlated-sampling noise of a stride-5 window, not the Poisson level of 150k
independent frames. So the window IS stationary under the C protocol -- worth knowing, because it
means the arms' ensembles are usable samples of their fields and the plan_c_run1 conclusion was
not an equilibration artefact.

**That control also bounds what the implied-distribution proxy can claim.**
fit_implied_ln_ratio compares the sampled histogram with the Boltzmann distribution the field
implies, and the two are not supposed to agree exactly: the sampled Hamiltonian also carries the
wall potential (table_wall:2000) and every coupling to the rest of the chain, so a bonded
coordinate's marginal is not exp(-U_coord/kBT). Measured on round 4, median ln-units:

| field | bb_bond | angle | dihedral |
| :-- | --: | --: | --: |
| U = the histogram's own inverse (floor of the instrument) | 0.007 | 0.009 | 0.035 |
| U = the field that PRODUCED this ensemble (unchanged) | 0.218 | 1.721 | 0.514 |
| U = the stabilised fit, gain 1 | 0.284 | 2.168 | 1.399 |

The floor is what makes the instrument worth quoting at all; the middle row is what makes it
confounded -- the field that produced the ensemble cannot reproduce it either, by a median factor
of e^1.7 on the angle, and no fit can be blamed for a gap the sampling Hamiltonian opens on its
own. It is reported as a **warning about the fit, not as proof**: on this proxy the refit reads
worse than leaving the field alone for bb_bond and the angle, and better for the dihedral only when
strongly damped (3.5). The clean statements stay the sampled ones: 3.1's self-consistency distances
and the retention instrument.

### 3.5 Does any amount of refit help? -- gain sweep against the unchanged field

Same proxy, same round-4 ensemble, the stabilised fit at K=8 and ridge 1e-1, varying only the
damping of the step against the field that produced the ensemble (gain 0 = leave it alone):

| gain | bb_bond | angle | dihedral | step std bb/ang/dih |
| --: | --: | --: | --: | --: |
| 0.00 | 0.218 | 1.721 | 0.514 | 0 / 0 / 0 |
| 0.10 | 0.224 | 1.756 | 0.376 | 0.02 / 0.04 / 0.41 |
| 0.20 | 0.228 | 1.796 | **0.363** | 0.05 / 0.07 / 0.81 |
| 0.35 | 0.234 | 1.854 | 0.487 | 0.08 / 0.12 / 1.42 |
| 0.50 | 0.242 | 1.916 | 0.541 | 0.12 / 0.17 / 2.03 |
| 1.00 | 0.284 | 2.168 | 1.399 | 0.23 / 0.35 / 4.06 |

For bb_bond and the angle the proxy rises monotonically with the gain: on this ensemble a refit
never improves it, at any damping. For the dihedral it improves up to gain 0.2 (0.514 -> 0.363) and
is destroyed by a full step. With 3.4's caveat in mind, that says the arm's full-strength
replacement step is the wrong size for two of the three coordinates, and the one coordinate where a
damped refit helps is the one whose sampled self-consistency (3.1) does not settle -- two different
instruments pointing at the same coordinate.

## 4. The clean instrument -- two SAMPLED ensembles, with its floor

Everything in this section is offline: it reads the stored ensembles under results/plan_c and runs
no sampler. Tool: scripts/plan_c_instrument.py, tests tests/test_plan_c_instrument.py,
machine-readable record results/plan_c/instrument.json, printed tables instrument.txt.

### 4.1 Why the old instrument could not be repaired

Section 3.4 measured the confound and it is larger than any effect it was used to judge: the field
that PRODUCED an ensemble reads 0.218 / 1.721 / 0.514 median ln-units on fit_implied_ln_ratio
against an instrument floor of 0.007 / 0.009 / 0.035, because the sampled Hamiltonian carries the
wall potential and every coupling to the rest of the chain, so a bonded coordinate's marginal is
not the Boltzmann factor of its own term. The replacement compares two SAMPLED ensembles: both
sides came out of the same sampler under the same full Hamiltonian, so a difference between them is
the field difference plus sampling noise and nothing else. Six numbers per coordinate: tv,
mass-weighted ln_mean (which is exactly the loop's own stationarity -- the instrument reproduces
the json's recorded values to 1e-9, which is what its golden-value test pins), ln_max, and
dmean_sig / dstd_sig / dq05|50|95_sig, the moment and quantile shifts in units of the coordinate's
own spread. ln_max is quoted but never used for a verdict: it reads 690 on a bin nobody visited,
which is the same trap the previous instrument fell into.

### 4.2 The floor (question 1)

| floor part | bb_bond | angle | dihedral |
| :-- | --: | --: | --: |
| exact zero: one field, one seed, twice | 0 | 0 | 0 |
| empirical, adjacent pairs with step <= 1 kJ/mol: ln_mean min / median | 0.068 / 0.178 | 0.073 / 0.133 | 0.070 / 0.123 |
| same pairs: TV min / median | 0.034 / 0.087 | 0.036 / 0.066 | 0.035 / 0.061 |
| Poisson-independent p95 (a lower bound): ln_mean / TV | 0.037 / 0.019 | 0.028 / 0.014 | 0.027 / 0.013 |

The exact zero is measured twice and both halves are worth stating: C0_r1 and C1_r1 are identical
arrays (same field, same seed, so the two arms' first rounds are one trajectory), and the refused
first attempt's C2s8_r1 is identical to this run's -- identical although the fitting code changed
in between, which is the check that the sampler does not depend on the fit.

**MEASURED 2026-09-24: the floor is a LINE, not a bracket.** To-run item 1 asked for two independent
trajectories under ONE field -- the one C2s8's last round ended on -- and 614 s of wall clock on 6
workers bought it (7 chains x 8 replicas x 5000 steps x 2 seeds; scripts/plan_c_same_field_floor.py,
results/plan_c/same_field_floor.json):

| coord | pooled ln_mean | pooled TV | per-chain ln_mean min / median / max | per-chain TV median |
| :-- | --: | --: | --: | --: |
| bb_bond | **0.0471** | 0.0235 | 0.1081 / 0.1180 / 0.1474 | 0.0587 |
| angle | **0.0521** | 0.0260 | 0.1470 / 0.1709 / 0.1978 | 0.0830 |
| dihedral | **0.0589** | 0.0294 | 0.1153 / 0.1580 / 0.1750 | 0.0785 |

Seven chains is seven points, so the honest reading of the spread is min/median/max rather than a
variance -- and they behave exactly as independent chains should: the per-chain median is 0.95, 1.24
and 1.01 times the pooled value times sqrt(7), which is the sqrt(N) scaling section 4.2 assumed when
it bounded a 1.2M-observation window comparison with a 150k-observation block floor. So the bracket
below was conservative in the right direction (its minimum, 0.068-0.073, sits above the line) and is
REPLACED by the line for any pooled adjacent-round comparison at this protocol.

The empirical floor comes from the nine (bb_bond), ten (angle) and six (dihedral) adjacent pairs
whose injected step was at most 1 kJ/mol, i.e. the arms that had all but stopped moving. Its
members include C0's last three pairs, whose field changes are the smallest in the whole dataset
(0.15-0.56 kJ/mol applied std), so their round-to-round distance is almost all noise. The correlated
sampling of a stride-5 window puts the real floor 2-3x above the independent bound, and the bracket
to quote for a full-window adjacent-round comparison is **ln_mean 0.03-0.09, TV 0.015-0.04**. C0 --
an arm whose target is fixed and whose corrections have gone to 0.2-0.6 kJ/mol -- reads 0.068-0.115
over its last two rounds: that is what a settled arm looks like on this instrument.

### 4.3 C2s8 on the clean instrument (question 2)

| coord | r 1->2 | r 2->3 | r 3->4 | floor bracket | verdict |
| :-- | --: | --: | --: | --: | :-- |
| bb_bond (ln_mean) | 1.129 | 0.373 | 0.114 | 0.068 .. 0.178 | converges, ends AT the floor |
| bb_bond (TV) | 0.405 | 0.182 | 0.056 | 0.034 .. 0.087 | same |
| angle (ln_mean) | 0.305 | 0.154 | 0.102 | 0.073 .. 0.133 | converges, ends AT the floor |
| angle (TV) | 0.138 | 0.074 | 0.050 | 0.036 .. 0.066 | same |
| dihedral (ln_mean) | 1.035 | 0.649 | 0.816 | 0.070 .. 0.123 | 6.6-11.7x the floor, last step RISES |
| dihedral (TV) | 0.442 | 0.300 | 0.362 | 0.035 .. 0.061 | same |

The moment columns give the shape rather than the amplitude, and they say the same thing: bb_bond's
quantile shifts decay (dq50/sigma 0.069, 0.031, 0.047; dq95/sigma 0.741, -0.003, -0.057) and so do
the angle's (0.148, 0.049, 0.033), while **the dihedral's alternate sign every round** (-1.126,
+0.558, -0.598): the distribution is orbiting, not settling. At K=4 the same instrument reads the
dihedral 1.023 / 1.070 / 1.149 -- rising at both orders.

### 4.4 Side by side with the polluted instrument (question 3)

Paired so that both columns are about the field written at the END of round r: the clean number is
the distance from ensemble r to r+1, the polluted one is fit_implied_ln_ratio_median recorded at
round r.

| arm | coord | clean ln_mean 1->2 / 2->3 / 3->4 | polluted r1 / r2 / r3 | reading |
| :-- | :-- | --: | --: | :-- |
| C2s8 | bb_bond | 1.129 / 0.373 / 0.114 | 0.957 / 0.499 / 0.196 | agree: both fall |
| C2s8 | angle | 0.305 / 0.154 / 0.102 | 0.974 / 0.631 / **1.722** | OPPOSITE: the proxy says the fit is getting worse while the ensembles close |
| C2s8 | dihedral | 1.035 / 0.649 / 0.816 | 1.004 / 0.822 / 1.234 | opposite by first-vs-last; neither shows convergence |
| C2s4 | angle | 0.407 / 0.291 / 0.156 | 1.081 / 0.641 / **2.153** | opposite, same as K=8 |
| C2s4 | dihedral | 1.023 / 1.070 / 1.149 | 1.492 / 1.277 / 1.780 | agree on the limit cycle |
| C1 | dihedral | 0.268 / 0.131 / 0.070 | 0.074 / 0.090 / 0.097 | opposite: the clean side falls to the floor |
| C1 | angle | 0.155 / 0.132 / 0.083 | 0.027 / 0.025 / 0.025 | agree (both flat or falling) |
| C0 | all three | 0.073-0.269, falling | not recorded | the polluted column does not exist for C0 |

**On every arm where both columns exist, at least one coordinate has the proxy rising while the
sampled distance falls** -- the angle in both C2 arms, the dihedral in C1. That is why the
instrument had to change: not that it was noisy, but that its sign was wrong where it mattered, and
a loop steered by it would have spent the next round on a coordinate that was already converging.

### 4.5 What this does to the section 3 conclusions (question 4)

**Both survive, and one of them gets stronger.**

* bb_bond and the angle having a fixed point: **the trend stands, the wording does not.** Against
  the measured line their last steps (0.114 and 0.102) are 2.42x and 1.96x the floor (0.0471 and
  0.0521), not inside it -- the bracket's upper half had been inflated by the small field changes of
  the pairs it was taken from. So: converging, ten-fold down from round 1, with a last step still
  twice the noise; not yet at rest.
* the dihedral being a limit cycle stands, and the measured floor makes it STRONGER: 13.84x the
  same-field floor (0.816 against 0.0589) at the last step, rising rather than falling, quantile
  shifts alternating sign, and the same shape at K=4 (1.023 / 1.070 / 1.149) where the proxy was the
  only witness before.
* What changes is the EVIDENCE, not the verdict: the earlier statement rested on an instrument whose
  own confound (0.218 / 1.721 / 0.514 for the field that produced the ensemble) is larger than the
  effects being judged. The verdict is unchanged, which is itself worth recording -- the confound
  was large enough to reverse a sign, and it happened not to reverse these.

### 4.6 To-run list -- two of the three are DONE

1. **The true floor. DONE 2026-09-24.** Two independent trajectories under ONE field with different
   seeds, same protocol as the arms: 7 chains x 8 replicas x 5000 steps, twice, 6 worker processes.
   Predicted 1.2 core-hours and about 11 minutes; measured 614 s of wall clock (1.0 core-hours) and
   the answer is the table at the top of 4.2: the bracket is replaced by a line, 0.047-0.059 pooled.
2. **Per-chain and per-block counts. DONE as code 2026-09-24** -- see 4.7 for what it answers and for
   the one thing that went wrong with this run's blocks.
3. **A production-burn control. Still to run, and deliberately not run yet.** The same-field pair at
   burn 20,000 (40 ps) instead of 1,000 (2 ps) would say whether the short burn leaves a residue the
   block floor does not see. The repo's own measurement (ibi_loop's docstring) says the transient
   exceeds 40 ps from deposited starts, which is why this is a control rather than an assumption.
   Decision rule agreed with the operator: if the 2 ps burn's residue is below the measured floor,
   this does not need to be spent.

### 4.7 The decomposition: per chain, per block (to-run item 2)

plan_c_loop used to store the POOLED histogram only, one number per coordinate per round, and that is
why the only floor sections 4.2 could build was a bracket. Since 2026-09-24
(plan_c_loop.store_ensembles) every round also stores:

* `<arm>_r<round>__chain<i>__<coord>` -- one histogram per chain, so a pooled distance becomes seven and
  gets a median with a spread (measured on the calibration run: pooled 0.047-0.059 against per-chain
  0.108-0.198, which is the sqrt(7) of independent chains, ratios 0.95 / 1.24 / 1.01);
* `<arm>_r<round>__blocks__<coord>` -- the window cut into equal-time blocks, summed over chains, which
  is what a delete-one jackknife at FULL WINDOW size needs. plan_c_instrument.jackknife implements it
  and is tested; the instrument's decomposed_report prints the per-chain spread and the jackknife for
  any pair that carries them.

Cost: 7 chains x 6 coordinates x 1000 bins x 8 bytes = 336 KB per round, eight times the pooled
48 KB, and nothing in the fit path reads any of it.

**What went wrong with this run's blocks, and what it cost.** The calibration run stored its blocks
in the script's first key layout; the analysis script then re-saved the file twice on top of itself,
and the second re-save nested a phantom chain axis into the block array (shape (7,7,8,1000) where
(7,8,1000) was meant), so every chain appeared to carry chain 0's blocks. The COUNTRIES -- the
per-chain histograms the floor is computed from -- were unaffected and were verified against the
pooled totals; only the block decomposition was lost, so 4.2 quotes no jackknife and the analysis
prints "not available" for it rather than a number. The re-save now uses np.stack with a shape
assertion and refuses to write an array whose per-chain rows are all identical, which is the
signature of exactly this duplication -- and the reason it took a re-save bug rather than a sampling
budget to lose it is that the counts were on disk, which is the whole point of item 2.

### 4.8 Reproduce

```
python scripts/plan_c_instrument.py                     # both datasets, tables + instrument.json
python scripts/plan_c_instrument.py --datasets c2stab --reps 1000
python scripts/plan_c_same_field_floor.py               # sampling: 614 s on 6 workers
python scripts/plan_c_same_field_floor.py --reuse       # re-analysis from the stored counts
pytest tests/test_plan_c_instrument.py -q
```

## 5. The dihedral 2-cycle: three arms that try to break it -- measured

### 5.1 The cycle, and why damping is the standard remedy

The mechanism was measured separately (scripts/plan_c_dihedral_cycle.py, commit f991274) and is not
re-derived here: the dihedral's refit step does not decorrelate. corr(step_r, step_{r-1}) is -0.85 at
round 3 and -0.94 at round 4 while its amplitude GROWS (2.72 -> 3.42 -> 4.06 kJ/mol, above 1 kBT),
against 0.23 / 0.35 for bb_bond and angle, whose correlations sit near zero. A shape correlation
approaching -1 with a growing amplitude is an unstable 2-cycle of the refit map -- sampling noise
would decorrelate, a contraction would decay -- and the reason it is the dihedral is in its shape:
its marginal is bimodal (2 local maxima above ten per cent of the peak against 74-324 elsewhere) and
31-49 per cent of its mass sits in the outer five per cent of the support, so a smooth global basis
can only move mass from one edge region to the other, round after round.

Damping is the standard remedy for a fixed-point iteration that walks: the growth ratio is about 1.2,
so 1/|lambda| is about 0.83 and a gain below that should land inside the stability boundary. The
three arms ask whether it does, and whether the production rule is already inside it.

### 5.2 What the arms are

| arm | bb_bond | angle | dihedral |
| :-- | :-- | :-- | :-- |
| D02 | stabilised refit, gain 1.0 | stabilised refit, gain 1.0 | stabilised refit, **gain 0.2** |
| D05 | stabilised refit, gain 1.0 | stabilised refit, gain 1.0 | stabilised refit, **gain 0.5** |
| Dtbl | stabilised refit, gain 1.0 | stabilised refit, gain 1.0 | **table inversion** (ibi_bonded.plan_update, the production rule, gain 1.0) |

Everything else is identical to C2s8 -- same pool, same seeds, same protocol, same ridge, taper and
guard -- so a difference between the arms is the dihedral's update rule and nothing else. Each arm
stop on its first refusal, as the fitted arms do.

**An offline pre-check gives Dtbl a positive prior** (results/plan_c/_precheck_dihedral.py, applying
plan_update to the same round-4 ensemble the refit saw): the table inversion wants a MILDER dihedral
step there -- mass-weighted rms 2.99 against 4.06 kJ/mol -- and it is positively correlated with the
refit step (+0.55), i.e. the same shape with less amplitude rather than a different direction. The
same pre-check on the other two coordinates says why the production loop looks the way it does: for
the angle the table step is 2.27 against the refit's 0.35 kJ/mol and anti-correlated (-0.77), and for
bb_bond 6.03 against 0.23.
### 5.3 Measured: four dihedral rules, four rounds each

All four arms share the pool, the seeds, the protocol, the ridge, the taper and the guard; they
differ only in what the dihedral does. C2s8 is the reference (gain 1.0 refit, from the c2stab run);
the rms column is the mass-weighted std of the step the rule wanted, corr is
corr(step_r, step_{r-1}) over the mass-bearing bins, and the clean column is the distance between
consecutive SAMPLED ensembles against the same-field floor of section 4.2.

| arm (dihedral rule) | rms kJ/mol, r1 -> r4 | corr, r2 / r3 / r4 | clean ln_mean (x floor), r2 / r3 / r4 | TV r4 | edge mass r4 | ret pool / holdout |
| :-- | :-- | --: | --: | --: | --: | --: |
| C2s8: refit, gain 1.0 | 4.08 / 2.72 / 3.42 / **4.06** | -0.37 / -0.85 / **-0.94** | 1.035 (17.6x) / 0.649 (11.0x) / **0.816 (13.8x)** | 0.36 | 0.43 | 0.40 / 0.49 |
| D05: refit, gain 0.5 | 2.04 / 1.18 / 0.59 / **0.32** | +0.78 / +0.89 / **+0.79** | 0.515 (8.7x) / 0.279 (4.7x) / **0.148 (2.5x)** | 0.073 | 0.396 | 0.391 / 0.491 |
| D02: refit, gain 0.2 | 0.82 / 0.67 / 0.56 / **0.44** | +0.97 / +0.99 / **+0.99** | 0.284 (4.8x) / 0.191 (3.2x) / **0.114 (1.9x)** | 0.057 | 0.466 | 0.388 / 0.512 |
| Dtbl: table inversion, gain 1.0 | 1.44 / 1.01 / 0.88 / **0.60** | +0.66 / +0.97 / **+0.96** | 0.343 (5.8x) / **0.083 (1.4x)** / 0.099 (1.7x) | 0.050 | 0.301 | 0.395 / 0.513 |

Every number in the table is from results/plan_c/plan_c_dihedral.json and ensembles_dihedral.npz,
and the same table is reproduced by results/plan_c/_analyse_dihedral.py. No arm refused: the guard
never fired, so these are four runs that behaved rather than four runs that were stopped.

### 5.4 Did the cycle break? -- yes, at every damping, and the table rule is already inside it

**The anti-correlation is gone in all three dihedral arms.** corr goes from -0.85 / -0.94 (gain 1.0,
the cycle) to +0.78 / +0.89 / +0.79 at gain 0.5, +0.97 / +0.99 / +0.99 at gain 0.2, and
+0.66 / +0.97 / +0.96 for the table inversion. One caveat on the criterion as it was written:
'corr -> 0, decorrelated' is not what a converging iteration looks like -- a contraction repeats the
SAME step with a shrinking amplitude, which reads corr near +1, while ~0 is the signature of
noise-dominated steps. What the cycle was made of is the SIGN, and the sign flipped.

**The amplitude falls monotonically in all three**, against rising at gain 1.0: 0.82 -> 0.67 -> 0.56
-> 0.44 (D02), 2.04 -> 1.18 -> 0.59 -> 0.32 (D05), 1.44 -> 1.01 -> 0.88 -> 0.60 (Dtbl), against
4.08 -> 2.72 -> 3.42 -> 4.06 (C2s8).

**And the clean distance walks into the floor**: D02 ends at 1.9x, D05 at 2.5x and Dtbl at 1.7x the
same-field floor, against 13.8x for the undamped refit -- where 1.0 means the ensemble moved no more
than two independent trajectories under one field do.

**Dtbl is the fastest to the floor and the only rule that changes the SHAPE.** It reaches 1.40x at
round 3 and holds 1.69x at round 4, and its edge mass falls 0.495 -> 0.337 -> 0.315 -> 0.301 while
the damped refits leave it at 0.396-0.466. That is the mechanism the cycle was made of: the smooth
global basis can only shuttle the 31-49 per cent of dihedral mass living in the outer five per cent
of the support from one edge region to the other, while the table inversion reshapes the marginal
where it is bimodal. The production loop's choice of the table rule for the dihedral is therefore
not a historical accident: it is the one that can fix the thing that was wrong.

**Retention never moved in any arm**: pool 0.385-0.400 and holdout 0.482-0.522 across all sixteen
arm-rounds, against the shared start field's 0.399 / 0.500 and tables_r3's 0.411 / 0.486. The
criterion was that it must not leave 0.39-0.40, and no arm did.

**bb_bond and the angle were not hurt by changing their partner.** Under all three D arms they
behave like C2s8's (monotone fall, correlations drifting positive) and end at 2.3-2.8x (bb_bond) and
1.8-2.1x (angle) the floor. A small cross-effect is worth recording: D02's and Dtbl's bb_bond end at
2.79x / 2.31x against C2s8's 2.42x, and their angle at 1.83x / 1.88x against 1.96x -- the coupled
coordinates settle a little further once the dihedral stops cycling, which is what a coupled
fixed-point iteration should do.

### 5.5 What this settles, and what to do with it

1. **The dihedral's rule should be the table inversion** (which production already uses), or a refit
   damped to gain <= 0.5. A full-gain refit is the only one of the four that cycles.
2. **If a uniform operator is wanted later** -- Plan B's moment operator is the live candidate -- the
   dihedral needs either a gain at or below 0.5 or a basis that can represent a bimodal marginal
   (a mixture, or a periodic basis, not K=8 Chebyshev over the whole support). The acceptance test is
   not 'the step got smaller': it is corr(step_r, step_{r-1}) >= 0 with a monotone amplitude, the
   clean distance reaching the same-field floor, and retention inside its band.
3. **The 2-cycle is a diagnosable failure, not a mysterious one**: it shows up as a sign flip in this
   correlation together with a growing amplitude and an edge mass the basis cannot move. All three
   are cheap columns, and the loop now records two of them (edge_mass and step_shape_corr_prev) on
   every round of every arm.
## 6. Reproduce

```
set TORUSFOLD_RSRNASP=C:/baidunetdiskdownload/torusfold-hybrid/_cgdata/combined
python scripts/plan_c_loop.py --rounds 4 --arms C2s8,C2s4,C2s2 --tag c2stab --preflight-retention 0
python scripts/plan_c_loop.py --rounds 4 --nrep 1 --nsteps 200 --relax 200 --pool-size 2 --holdout 1 --retention-steps 200 --retention-relax 200 --tag smoke5 --arms C2s2 --c2-gain 30 --workers 2 --preflight-retention 0
```

The pool depends on WHICH DATABASE IS MOUNTED (build_pool's own docstring says so): without
TORUSFOLD_RSRNASP pointing at _cgdata/combined the 24-34 band has eight chains instead of eleven
and the run stops with 'the 24-34 band has 8 chains, asked for 7 + 4' -- which is what the first
launch of this task did. With it set, build_pool(7, 4) returns exactly the chains
plan_c_run1.json was run on, and tests/test_plan_c_loop.py pins that list.

Records: results/plan_c/plan_c_c2stab.json (this run), plan_c_c2stab_refused.json (the first
attempt, refused on all three coordinates in round one -- kept as evidence),
ensembles_c2stab.npz (every round's ensemble), fields/c2stab/ (every field).

Two notes on the records. plan_c_c2stab.log is NOT usable: the tee process was stopped with the
run, so the file holds only the stderr warnings of the refused first attempt -- the json is the
record. And every table above comes from a scratch script under results/plan_c/_*.py, uncommitted
by design: _k_sweep.py, _ridge_with_taper.py, _gain_sweep.py, _floor.py and _equilibration.py
reproduce sections 3.3-3.5, while _spectrum.py and _why_blowup.py reproduce section 1.
