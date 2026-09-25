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

PHASE2

## 4. Reproduce

```
python scripts/plan_c_basis_family.py                 # phase 1, offline, writes basis_family.json
pytest tests/test_plan_c_basis.py -q
python scripts/plan_c_loop.py --rounds 4 --arms B08,B16,B32,B64 --tag basis --preflight-retention 0
python results/plan_c/_analyse_basis.py               # the per-arm table against the floor
```

Records: results/plan_c/basis_family.json (phase 1), plan_c_basis.json + ensembles_basis.npz +
fields/basis/ (phase 2). The same-field floor these arms are read against is
0.0471 / 0.0521 / 0.0589 ln_mean for bb_bond / angle / dihedral (results/plan_c/same_field_floor.json).
