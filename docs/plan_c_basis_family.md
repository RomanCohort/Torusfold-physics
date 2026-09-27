# The basis family project: which correction family can carry a bimodal, edge-piled marginal at gain 1.0?

*Opened 2026-09-24, from the result in docs/plan_c_c2_stabilization.md section 5. Code:
scripts/plan_c_basis.py (the families and the shared fit), scripts/plan_c_basis_family.py (phase 1),
tests/test_plan_c_basis.py. Phase-2 arms are named B08/B16/B32/B64 and live in plan_c_loop.py.*

## 1. The question, and why it is not arm-tuning

Four arms already made this a controlled experiment. On ONE coordinate (the dihedral), ONE ensemble,
at gain 1.0:

* a smooth global **K=8 Chebyshev refit cycles** -- corr(step_r, step_{r-1}) -0.94, amplitude growing
  2.72 -> 3.42 -> 4.06 kJ/mol, 13.8x the same-field floor, edge mass stuck at 0.43;
* the **1000-bin table inversion converges** and is the only rule that MOVES the edge mass
  (0.495 -> 0.301), reaching 1.40x the floor by round 3;
* damping the refit to gain 0.5 or 0.2 also converges (2.5x and 1.9x the floor), but that buys
  stability by taking smaller steps, not by being able to represent the target.

The dihedral's marginal is bimodal with 31 per cent of its mass in the outer five per cent of the
support, and a global basis can only shuttle that mass between the two edge regions. So the question
is a real one: **is there a family that is low dimensional, smooth, and still able to carry that
shape at full strength?** Two consumers: Plan C's self-consistent loop, and **Plan B's moment
operator**, which fits every coordinate with the same K=8 Chebyshev and will hit the same wall the
moment its target becomes a self-consistent ensemble.

The axis is LOCALITY, and the two known outcomes are its ends. In between sits a one-parameter
family of cubic B-splines with m basis functions; m* -- the knot count at which the cycle disappears
-- is the answer, and it is also what the moment operator should be using.

## 2. Phase 1: the frontier, measured offline

### 2.1 What was swept, and on what

The inputs are the ones the arms' round-4 fit saw: the round-4 ensemble (sampled under C2s8_r3) and
the field C2s8_r3 itself. The step a family "wants" here is therefore directly comparable with what
the arms wanted at round 4, and the cheb8 column IS that step, recomputed through
plan_c_loop._chebyshev_fit_stable on the same inputs. Nothing in phase 1 samples.

The fit is shared and unchanged across families: support cut at 1e-3 of the modal probability,
sqrt(p) weighting, a smoothstep taper in ln p two decades below the cut, the mass gauge, and a ridge.
`scripts/plan_c_basis.py`'s Chebyshev path is pinned to the loop's own function by
tests/test_plan_c_basis.py, which is what makes "only the design changed" true rather than assumed.

### 2.2 The dihedral frontier

resid = mass-weighted RMS of the fitted field against the target -kBT ln p, in the bins the ensemble
visits; edge gap = the fitted field's implied edge mass minus the target's (target 0.313); step rms =
the mass-weighted std of the step the family wants; cond_fit = condition number of the matrix
actually inverted. kJ/mol, kBT = 2.494.

| family | residual | edge gap | step rms | cond_fit | usable |
| :-- | --: | --: | --: | --: | :-- |
| Chebyshev K=8 | 2.907 | -0.120 | 4.058 | 11 | no |
| Chebyshev K=16 | 2.871 | -0.200 | 3.704 | 30 | no |
| Chebyshev K=32 | 2.873 | -0.163 | 3.688 | 55 | no |
| B-spline m=8 | 1.966 | +0.015 | 3.093 | 30-41 | no (residual) |
| **B-spline m=16**, ridge 1e-3 | **0.231** | -0.018 | 2.201 | 121 | yes |
| B-spline m=32, ridge 1e-3 | 0.244 | -0.021 | 2.202 | 821 | yes |
| B-spline m=64, ridge 1e-4 | 0.113 | -0.024 | 2.212 | 1.0e4 | yes |
| B-spline m=128, ridge 1e-4 | 0.102 | -0.025 | 2.214 | 1.0e6 | yes |
| table (1000 bins, fully local) | 1.163 | -0.125 | 2.405 | 497 | (see note) |

The table row's residual is not zero because the FIT weights only the bins above the support cut
while the residual is measured down to LN_RATIO_FLOOR: the pseudo-count bins in between are counted
in the residual and not in the fit. That is a definition difference, not a failed fit.

### 2.3 What the frontier says -- two of the three guesses were wrong

**Chebyshev's problem is shape, not resolution.** K=8, 16 and 32 give the same residual to three
digits (2.907 / 2.871 / 2.873) and all three leave the edge mass 40 per cent short (-0.12 to -0.20 of
a target that holds 0.313). Adding functions to a global basis does not help, because the shape it
cannot make is a pile-up at the ends.

**The locality threshold is LOW: m* sits between 8 and 16, not at 32-64.** m=8 is still a failure
(residual 1.97 kJ/mol, 0.79 kBT -- 8.5x worse than m=16), and from m=16 up the residual only creeps
from 0.23 to 0.10 as m goes to 128. The marginal return beyond m=16 is small, which is the useful
answer for Plan B': a low-dimensional, smooth, local family that can carry a bimodal edge-piled
marginal needs about sixteen basis functions, not sixty-four.

**And the step a family wants is smaller when it can express the target**: 4.06 kJ/mol at K=8 against
2.20 for every usable B-spline, with a shape correlation of +0.72..+0.81 against the Chebyshev step
(the same general direction, less amplitude needed).

### 2.4 The ridge-scaling trap that had to be fixed first

moment_correction's relative ridge is lambda = ridge_rel x trace(A^T W A)/m. For a Chebyshev design
that is about 0.43 x eig_max, so it behaves like an eigenvalue-relative ridge. For a B-spline it does
NOT, and the measurement on this very ensemble is:

| m | 8 | 16 | 32 | 64 | 128 |
| :-- | --: | --: | --: | --: | --: |
| trace/m | 0.0607 | 0.0288 | 0.0145 | 0.00736 | 0.00373 |
| eig_max | 0.141 | 0.0925 | 0.0845 | 0.0825 | 0.0761 |
| trace/m / eig_max | 0.430 | 0.312 | 0.172 | 0.0891 | 0.0490 |

trace/m falls by 16x while eig_max falls by 1.9x, so the same ridge_rel regularises **8.8x less** at
m=128 than at m=8: the ridge tuned on a Chebyshev design stops doing anything exactly where the
design needs it, and by m=48 the weighted system is numerically singular (the solve returned values
past 1e100 in the first sweep). Phase 1 therefore defaults to an eigenvalue-relative ridge, and the
same lesson is applied to the roughness penalty. This is a Plan-B' finding in its own right: the
moment operator cannot simply swap its Chebyshev design for a spline one without rescaling its ridge.

### 2.5 The other two coordinates (controls, not the target)

**bb_bond: B-splines win outright.** Residual 0.08-0.31 kJ/mol against Chebyshev's 1.06, and an edge
gap of +-0.004 against **+0.238** -- the Chebyshev field puts three times the target's mass at the
edges of the bond distribution while every B-spline from m=8 up carries it. Every (m, ridge) point
swept is usable here.

**angle: a different disease.** No family carries its edge mass (the gap stays +0.10..+0.29 for every
m and every ridge) and its residual swings between 0.4 and 2.2 depending on the pair, so the angle is
not the dihedral's problem with a different basis -- something else is wrong with the angle's target,
and it is left as its own thread rather than folded into this project.

### 2.6 Selection for phase 2

Four arms at gain 1.0, dihedral only, everything else identical to the D arms and C2s8, so a
difference is the basis and nothing else. Each one is a test of a different part of the frontier:

| arm | dihedral | phase-1 prediction | why this one |
| :-- | :-- | :-- | :-- |
| B08 | B-spline m=8, ridge 1e-3 | should still cycle | below the measured threshold (resid 1.97); this is the arm that makes the m* claim falsifiable |
| B16 | m=16, ridge 1e-3 | should stop cycling | the threshold itself (resid 0.23, cond 121) |
| B32 | m=32, ridge 1e-3 | should stop, with margin | just inside (resid 0.24, cond 821) |
| B64 | m=64, ridge 1e-4 | should stop | the finest end phase 1 still called usable (resid 0.11, cond 1e4) |

## 3. Phase 2: the four arms

### 3.1 The four arms, against C2s8

Every number below is offline: the arms' records for the step diagnostics, the stored ensembles for
the clean instrument. The r1 edge gap and residual are recomputed from each arm's OWN round-1 inputs
(the start field and the round-1 ensemble), and the recomputed r1 field is checked elementwise
against the field the arm wrote, so a mismatch cannot pass silently.

| arm | replicas | r1 residual kJ | **r1 edge gap** | corr r2 / r3 / r4 | rms r1 -> r4 | monotone | clean x floor r2 / r3 / r4 | ret pool |
| :-- | --: | --: | --: | --: | --: | :-- | --: | --: |
| C2s8 Chebyshev K=8 | 8 | 3.849 | **-0.378** | -0.37 / -0.85 / -0.94 | 4.08 / 2.72 / 3.42 / **4.06** | **no** | 17.6 / 11.0 / **13.8** | 0.389-0.401 |
| B08 B-spline m=8 | 6 | 0.643 | **+0.012** | +0.71 / +0.90 / +0.92 | 1.62 / 1.08 / 0.59 / 0.42 | yes | 6.6 / 3.1 / **2.4** | 0.386-0.408 |
| B16 m=16 | 6 | 0.508 | +0.002 | +0.79 / +0.81 / +0.94 | 1.65 / 1.21 / 0.67 / 0.36 | yes | 7.5 / 3.9 / **2.1** | 0.385-0.394 |
| B32 m=32 | 6 | 0.492 | -0.005 | +0.83 / +0.91 / +0.86 | 1.64 / 1.22 / 0.64 / 0.38 | yes | 7.4 / 3.8 / **2.2** | 0.388-0.396 |
| B64 m=64 | 6 | 0.433 | -0.002 | +0.91 / +0.88 / +0.85 | 1.62 / 1.30 / 0.68 / 0.58 | yes | 7.0 / 3.9 / **3.3** | 0.393-0.394 |

C2s8's corr column is not in its own record -- that diagnostic was added to run_arm after that run --
so it is the value from commit f991274, independently recomputed here for the last step (-0.99).

### 3.2 Verdict: m*=16 is refuted, and the edge gap is what predicts the cycle

**The negative control does not cycle.** B08 (m=8), which phase 1 predicted would cycle because its
residual is 1.97 kJ/mol, has a correlation sequence of +0.71 / +0.90 / +0.92 with no sign flip, a
monotonically falling step (1.62 -> 0.42 kJ/mol) and a clean distance walking from 6.6x to 2.4x the
same-field floor. The phase-1 prediction rested on the residual and it was wrong; the m*=16 claim is
retracted here, by its own control.

**What separates the one cycler from the four that converge is the edge gap.** C2s8's round-1 fit
delivers an implied edge mass 0.378 below its target's -- less than half of it -- while all four
B-spline arms are within |gap| <= 0.012. The residual does NOT order them: among the four that
converge it spans 0.43 to 0.64 with no relation to cycling, and **B08 has the worst residual of the
four and converges most cleanly**. The discriminating evidence is therefore B08 rather than the
Chebyshev arm: a family that cannot carry the target's edge mass keeps asking for more correction
there every round, which is what becomes the sign flip, while a family that can carry it converges
whatever its order and a high residual only slows it down.

**The claim, with its limits.** "Any family that puts the edge mass right does not cycle, whatever its
order" is supported by five arms -- with the honest caveat that there is only ONE cycler, so any
quantity separating C2s8 from the other four "predicts" it; the arm that makes the claim
discriminating rather than vacuous is B08. Phase 1's Chebyshev sweep also shows bad edge gaps at K=16
and K=32 (-0.200, -0.163), but neither was run as an arm, so the K-dependence has no arm-level
evidence and is not claimed.

**Protocol note, since the arms are 6 replicas and C2s8 is 8.** What that affects: the statistics the
FIT sees (about 25 per cent fewer frames per round, so the residual and gap columns are slightly
noisier than C2s8's -- against a separation of 0.012 versus 0.378, no threat) and the variance of
sim/ref. What it does NOT affect: the retention instrument, which is its own 6000-step run and
independent of --nrep, so the 0.39-0.42 / 0.48-0.52 band comparison stays valid; and the qualitative
cycle-versus-spiral reading.

**For Plan B', the actionable form is one cheap diagnostic**: before trusting an update, compare the
fitted field's implied edge mass (the outer five per cent of the support at each end) with the
target's. A deficit means the family cannot carry the shape the target needs, and no ridge or order
will fix it -- the update will cycle. The number is offline, costs one bin-integration per coordinate
per round, and is available as plan_c_basis.implied_edge / edge_mass.

### 3.3 What this does to phase 1's selection

The four arms still answer what they were selected for, but the reason for the recommendation
changes: with the edge gap right, the ORDER should be chosen for the residual, which is what decides
how fast the loop closes. m=16 is the cheapest order that is both (residual 0.51 kJ/mol at round 1,
gap +0.002); m=8 is cheaper but leaves 0.64 on the table; m=64 buys 0.43 and costs nothing in
stability once the ridge is eigenvalue-relative, yet nothing in the arm data shows it converging
better than m=16 (3.3x against 2.1x the floor at round 4).


## 4. The angle arms (C)

*Running now as scheduled task plan_c_angle, started 2026-09-27 14:39 (two arms, 6 workers each).
This section is filled from results/plan_c/plan_c_CA16.json, plan_c_CBdep.json and their ensembles
when they land; the design and its reasons are already fixed:*

- **CA16** -- the angle refitted on a B-spline m=16 (ridge 1e-3, eigenvalue-relative), bb_bond and the
  dihedral unchanged at C2s8's K=8 refit. This asks whether the BASIS was what limited the angle:
  phase 1 measured its residual against -kBT ln p at 1.1-2.2 kJ/mol for a Chebyshev design against
  0.4-0.6 for a local one.
- **CBdep** -- the angle on the production rule for that coordinate (target = the deposited marginal
  p_ref, ibi_bonded.plan_update, gain 1.0), everything else unchanged. This asks whether the TARGET
  is what limits it: the deposited reference's sigma is 0.32176, and on the full pool the angle's
  sampled sigma stalls about 9 per cent wider than its target while the table's implied sigma falls
  about 4 per cent per round and the sampled one only 1.4 (Part 8, commit 3cccd4f).

Why the self-consistent target was NOT used for CBdep, although the task suggested it: it is already
what every C2s/B/D arm uses for every coordinate -- _target_from_counts builds -kBT ln p of the arm's
own ensemble -- so switching to it would change nothing. The deposited marginal is the other
well-defined target and the only one whose sigma can be compared round after round.

Why not the surgical variant (down-weight the edge mass): phase 1 measured the angle's implied edge
mass to be 1.5-2.4x its target's for every family, so for the angle the fit over-delivers the edges
rather than failing to reach them; the deficit the edge-gap hypothesis is about belongs to the
dihedral. Down-weighting would treat a symptom the measurement does not show.

Both arms are read with the clean instrument (round-to-round ensemble distance against the same-field
floor 0.0521 for the angle) AND with the two pooled quantities the task names: the moment residual
|d<T>|max against the target, and sigma_sim against sigma_target.

## 5. The full-pool validation: proposal (NOT launched -- this is the account the operator approves)

### 5.1 What would be validated

The one thing the seven-chain project cannot answer is whether the edge-gap prescription matters at
full pool scale, where the production campaign's converged fields live. The candidate change is
narrow by construction: **replace the refit basis for the coordinates whose edge gap is bad with a
B-spline m=16 (gain 1.0, eigenvalue-relative ridge), and leave everything else exactly as the
production campaign ended** -- same operator for the other coordinates, same target, same protocol,
same checkpoints.

### 5.2 Step 0, before any sampling: measure the full-pool edge gap (offline, about 15 minutes)

Phase 1 measured edge gaps on the seven-chain pool. The full pool's are computable from what is
already on disk and cost no sampling: each round stores per-chain histograms in
results/ibi_relax/tasks_r<N>/<idx>.npz (READ-ONLY), so summing them gives the pooled ensemble per
round, and tables_r<N>.npz gives the fields. The number to produce is, per coordinate per round of
the finished campaign, the implied edge mass of the field minus the ensemble's edge mass.

That is what decides whether this proposal is worth 1,350 core-hours at all: **if the production
table/moment rule already carries the edges at full pool** -- which the seven-chain Dtbl arm suggests
it might, since the table rule reached 1.4x the floor there -- then the basis change is unnecessary
and the answer is no. If instead a coordinate shows the Chebyshev disease (a gap of a few tenths),
the arm below is the cheapest test of the fix.

### 5.3 The arm, if step 0 says go

| item | value |
| :-- | :-- |
| chains | 867 (the same pool, so the tables stay comparable) |
| rounds | 3 (the seven-chain convergence took 3-4; 9 rounds of the production campaign is the fallback if 3 is ambiguous) |
| operator | the moment operator for every coordinate, as the production campaign ended, EXCEPT the one(s) step 0 flags, which are refitted on a B-spline m=16 |
| ridge | eigenvalue-relative, 1e-3, for the refit; the moment operator's covariance needs its own rescaling (failure mode 3) |
| acceptance | (1) edge gap |gap| <= 0.05 on the flagged coordinate; (2) |d<T>|max falls towards the noise floor instead of stalling; (3) sigma_sim closes on sigma_target; (4) retention inside 0.39-0.42 / 0.48-0.52; (5) the round-to-round clean distance falls below the subset floor's multiple |
| cost | 1 round = 450 core-hours measured (867 chains x 32,500 steps; the sum of per-chain times is 448-461 core-h); 3 rounds = 1,350 core-h |
| wall clock | 14 h per round at 32 workers is the floor and the campaign's own rounds took 15.9-18.2 h, so 3 rounds is 48-55 h (2.0-2.3 days), plus about 1 h for step 0 and the subset floor |
| extra measurement | a same-field floor on a SUBSET (20 chains spanning the length range, 2 trajectories: about 1.2 core-h), because a full-pool floor would cost 900 core-h and is not worth a calibration number |

### 5.4 What would make it fail, and what each failure means

1. **Step 0 shows no bad edge gap at full pool.** Then the seven-chain disease does not survive the
   full pool's heterogeneity, the edge-gap prescription is a seven-chain artefact, and the honest
   answer is that the production rule never needed the basis change. Cost of finding out: 15 minutes.
2. **Exactly one coordinate needs it.** A per-coordinate rule is then a judgement call, not a
   technical one: the machinery already exists (plan_c_loop's per-coordinate specs), the cost is
   complexity in the operator table, and the alternative -- leave that coordinate on the table rule,
   which is what production did -- is demonstrably adequate on the seven-chain pool (1.4-1.7x the
   floor). Recommendation in that case: do not open a per-coordinate basis rule for one coordinate;
   record the measurement and move on.
3. **The moment operator's covariance does not accept the new basis without its own ridge.** The
   eigenvalue-relative rescaling was derived on the refit's Gram, not on moment_correction's
   covariance. Offline check before the arm: build the B-spline design, form the operator's own
   matrix, and read its condition number against ridge -- cheap, and it belongs in step 0.
4. **The edge gap is chain-dependent.** The pool mixes 21-residue chains with a 2,929-residue one, and
   the bimodality that produces an edge pile-up is a per-chain property. The per-chain histograms are
   already stored, so step 0 reports the gap's distribution across chains, not just its pooled value;
   if only a handful of chains show it, the arm should run on those rather than on 867.
5. **Wall-clock risk.** 48-55 h of machine time for a change whose seven-chain effect is about 2x the
   floor is a real cost; the campaign is checkpointed per task, so it can be stopped between rounds
   without losing a round, and 3 rounds at 18 h is the number to approve.

### 5.5 What it would cost NOT to run it

Nothing breaks: the production campaign finished, converged, and its fields are the record. What is
lost is the only arm-level evidence that a local basis helps where the edge gap is bad at full pool
scale. Given that step 0 is 15 minutes and the arm is two days, the recommendation is: **run step 0,
then decide.** If step 0 shows the disease, the arm is worth it; if not, the seven-chain basis project
closes with its answer recorded, and Plan B' keeps the edge-gap diagnostic, which is the durable part
of this line of work.
## 6. Reproduce

```
python scripts/plan_c_basis_family.py                 # phase 1, offline, writes basis_family.json
pytest tests/test_plan_c_basis.py -q
python scripts/plan_c_loop.py --rounds 4 --arms B08,B16,B32,B64 --tag basis --preflight-retention 0
python results/plan_c/_analyse_basis.py               # the per-arm table against the floor
```

Records: results/plan_c/basis_family.json (phase 1), plan_c_basis.json + ensembles_basis.npz +
fields/basis/ (phase 2). The same-field floor these arms are read against is
0.0471 / 0.0521 / 0.0589 ln_mean for bb_bond / angle / dihedral (results/plan_c/same_field_floor.json).
