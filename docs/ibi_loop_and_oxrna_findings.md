# The IBI loop, and what the oxRNA / RhoFold comparison actually measured

One session's measurements, written down because six of them overturn something that was
believed or written earlier. Every number here comes from a run whose log is named; the raw
artifacts are under `results/` (in-repo) and `ox_runs/` (outside the repo).

The two halves are independent: the first is about the force field's own iteration, the second
about the architecture claim (search stage vs refinement stage) that `force_field_comparison.md`
raises.

---

## Part 1 — The IBI loop now exists and has run

### 1.1 What was built

| piece | file | what it does |
| :-- | :-- | :-- |
| injection point | `src/.../torch_cgsim.py` (`cg_energy_forces`, keyword-only `angle_potential` / `dihedral_potential`) | run the full field with two terms replaced. `None` (default) is bit-identical to before — checked by an 8000-step regression that reproduces `results/newfield_8x8000.log` exactly, and by `tests/test_table_potential_injection.py` |
| potential construction | `scripts/cg_potentials.py` | spec string -> callable. Holds no energy expression of its own; `force_reference._v_fn` is the single site |
| sampling core | `scripts/ibi_core.py` | ONE copy of the loop (`§3ba`: 采样环不能有两份). `ibi_round0.py` and `pool_sim_bonded.py` both became callers |
| on-disk handoff | `ibi_round0.py --write=DIR` | per-coordinate npz (`counts/n/n_outside/lo/hi/nbins/U/centre/block_counts/values`) + `manifest.json` |
| update half | `scripts/ibi_update.py` | the first caller `plan_update` has ever had |

### 1.2 Three defects found while wiring it

1. **`force_reference.dihedral_force` had no `enable_grad` block.** Inside `torch.no_grad()` —
   which is exactly where the sampler evaluates forces — `requires_grad_(True)` does not re-enable
   tracking and `backward()` raises. It never surfaced because the function had only ever been
   called from `main()` in normal grad mode.
2. **`--table` did not reach the potential.** `force_reference` caches its table on the module, so
   a round would have SAMPLED under the old potential while BINNING against the new table — the
   one failure that makes a round look converged for the wrong reason. Fixed with
   `cg_potentials.use_table_file`, called before the potentials are built.
3. **`K_BSJ_CONTACT` was missing from `ibi_round0.py`'s `_FINGERPRINT`.** A round-0 run patched it
   to zero and the manifest recorded three of the four BSJ constants. `ibi_remd_residual.py` has
   had it all along — the duplicated-list drift that file's own comment warns about.

### 1.3 Round 0 -> round 1: the first real IBI iteration

idx 0 (1L2X), 8 replicas x 100000 steps, burn 20000, tables as potentials.

| coordinate | round 0 | round 1 | |
| :-- | --: | --: | :-- |
| **dihedral** | 1.348 | **1.115** | **improved (the update was asked for this)** |
| angle | 1.451 | 1.525 | worse |
| stack | 1.462 | 1.567 | worse |
| bb_bond | 1.094 | 1.117 | worse |
| intra_pc | 1.040 | 1.068 | worse |
| intra_cn | 1.027 | 1.051 | worse |
| **joint J** | **0.2011** | **0.2011** | **unchanged** |

**The update worked on its target and paid for it elsewhere.** That is the coupling mechanism
seen all evening from the other side: the four hand-built arms (below) showed it when a human
picked the spec; here `plan_update` itself produces it. It matches `ibi_bonded`'s own synthetic
test, where the uncoupled case converges in one round and the coupled case (`J_COUPLED = 2.2`)
takes four — so one round being insufficient is the expected behaviour, not a verdict on the
method. Round 1's block spread is 17.2% (E1b, no injection, was 12.3%).

### 1.4 The four hand-built arms, and why manual swaps have a ceiling

Five chains (idx 0-4), `K_BSJ=K_BSJ_GUIDE=K_BSJ_CONTACT=K_BPP=0`, 8 x 100000 steps, pooled:

| coordinate | shipped | Fourier dihedral | both, naive | both, measure-corrected |
| :-- | --: | --: | --: | --: |
| **dihedral** | 0.659 | **1.060** | 1.063 | 1.099 |
| angle | 0.812 | 0.808 | **1.204** | **1.182** |
| stack | 0.861 | 0.855 | **1.232** | **1.207** |
| bb_bond | 1.063 | 1.062 | 1.054 | 1.052 |
| intra_pc | 1.004 | 1.006 | 1.010 | 1.006 |
| intra_cn | 1.000 | 1.006 | 1.004 | 1.006 |
| **joint J** | 0.140 | **0.083** | 0.087 | 0.085 |

Two results worth keeping:

- **Fourier N=2 fixes dihedral, across chains.** 0.659 -> 1.060, with four of the five chains
  landing within a few percent of 1.0. Joint J falls 40%, reproducing the single-chain -45% from
  earlier in the day. It is not a one-chain anecdote.
- **Swapping more coordinates made it worse, not better**, and the measure-corrected spec did not
  rescue it (it pulled dihedral back down from 1.063 to 1.099 while leaving angle and stack
  essentially where they were). The failure has the same *shape* as the naive-vs-B2 distinction
  `dihedral_table_decision.md` records — but that resemblance is a coincidence, not the
  mechanism. **Manual swap has a ceiling, and it is not a spec-selection problem.**

`scripts/cg_potentials.py --check` reports 0.000e+00 over 18 coordinate/spec/dtype combinations,
including a `torch.no_grad()` pass.

---

## Part 2 — oxRNA on 2OIU, and the architecture claim

### 2.1 What the survey's experiment actually asks

`force_field_comparison.md` proposes: build 2OIU in oxRNA, run it, compare to the crystal. If it
lands in the same neighbourhood as the Level-2 relaxation (**1.83 Å, 17 min**) for a fraction of
the cost, then our field's job is local refinement.

### 2.2 oxRNA runs, and is ~400x faster than our field

10^6 steps of oxRNA MD on the 71-nt circular 2OIU: **41.6 s on one CPU core** (0.0416 ms/step).
Our field is 61 steps/s on the same box, so 10^6 steps would be 4.6 hours. That is the concrete
number behind "worth having as an engine", and it did not need a GPU: the run was CPU.

### 2.3 But the measurement is not comparable to 1.83 Å, for two reasons

**(a) The conversion floor.** `tacoxDNA`'s `PDB_oxDNA.py` rebuilds a CG representation from PDB
atoms, and that step alone costs:

| structure | RMSD right after conversion (no relaxation) |
| :-- | --: |
| 2OIU (circular, L=71) | **1.919 Å** |
| 1L2X (linear, L=27) | **5.757 Å** |

**Already at or above the 1.83 Å criterion, before any dynamics.** This is the number that
matters most here: it says the oxRNA side's RMSD floor is set by the modelling path, not by the
engine, and that comparing oxRNA to Level-2's number requires a better path (build the topology
from sequence + secondary structure rather than reverse-engineering it from crystal atoms).

**(b) Level-2's 1.83 Å is a relaxation, not a prediction.** Its starting point IS the crystal.
oxRNA's run here started from the crystal too, but then ran free MD for 10^6 steps.

### 2.4 What the run did

`RNA_relax` (oxDNA's non-physical constant-force backbone, needed because the crystal-derived
configuration has bonded distances outside the FENE well) then free MD:

| stage | RMSD vs crystal |
| :-- | --: |
| as converted | 1.919 Å |
| after relax | 2.123 Å |
| **first MD snapshot (10^5 steps)** | **17.840 Å** |
| end (10^6 steps) | 16.311 Å |

**It does not drift — it jumps and then sits.** The jump happens inside the first 10^5 steps, and
the remaining 9x10^5 show no trend (14.0-17.9 Å). So ~16 Å is a stable state of oxRNA for this
chain, reached quickly.

**Caveat, unresolved:** `RNA_relax` uses an explicitly unphysical potential (oxDNA prints
`Using unphysical backbone`). It guarantees bond lengths, not that the result is anywhere near
equilibrium under the real RNA potential — so real MD relaxes hard at step one. How much of the
16 Å is the relax recipe versus oxRNA's landscape is **not separated**.

Linear control (1L2X), same protocol: converted 5.757 -> relax 5.890 -> **9.155 Å** at 10^6 steps.
So the jump is not specific to circular topology, but circular jumps much further (+15.7 Å
against +4.1 Å).

### 2.5 The predictor side: RhoFold works, and the ensemble is worse

**Correcting two earlier mistakes of this session.** RhoFold+ does not fail on Windows — it needs
`RHOFOLD_ROOT`, which was unset. Same for the other two predictors:

| predictor | where it lives | env var that was unset |
| :-- | :-- | :-- |
| RhoFold+ | `C:\Users\...\deploy\IGEM疑难服务\tools\RhoFold` (508 MB checkpoint) | `RHOFOLD_ROOT` |
| trRNA2 | runner `C:\Torusfold-physics\scripts\_trrna2_runner.py`, weights `C:\trRNA2\models\params` | `TRRNA2_RUNNER` |
| RNAbpFlow | `C:\Torusfold-physics\RNAbpFlow\checkpoint\RNA3DB.ckpt` | `RNABPFLOW_ROOT` |

All three are installed and all three run (the `comfyui` conda env carries torch 2.12+rocm7.13
and a working Radeon 8060S; RhoFold uses it, trRNA2 forces CPU to dodge a ROCm InstanceNorm
crash).

Measured on 2OIU from sequence + ViennaRNA circular secondary structure only (the crystal was
never shown to the model):

| search stage | RMSD vs crystal (P atoms) | confidence |
| :-- | --: | --: |
| RhoFold alone | **12.896 Å** | 0.780 |
| three-predictor ensemble | **22.143 Å** | 0.881 |

**The ensemble is worse, and its confidence is higher.** Coordinate fusion pulled the structure
toward self-consistency among the three predictions rather than toward the truth; the
`distance calibration: RMSE 20.44 -> 1.50 A` line is the suspicious part — that is coordinates
being fitted to a predicted distance matrix that may itself be wrong. (Part of the 22.1 Å is the
`post-relaxation (5000 steps)` that runs after fusion and was not separated.)

### 2.6 The architecture question, restated

Once the numbers are side by side, the criterion as written does not hold:

| | starting point | RMSD vs crystal |
| :-- | :-- | --: |
| RhoFold / pipeline Level 1 (search stage) | sequence + SS | 12.9-22.1 Å |
| oxRNA free MD | crystal | 16.3 Å |
| **Level-2 (refinement stage)** | **crystal** | **1.83 Å** |

Level-2 reaches 1.83 Å because it starts from the answer. No predictor can, and 1.83 Å is
therefore not a "search stage" bar. **The coherent form of the architecture claim is: the search
stage supplies ~13-22 Å, and the refinement stage closes the rest.** Only the second half has
been demonstrated (from the crystal). The untested link is *refinement started from a coarse
structure* — and both endpoints for that test now exist on disk
(`ox_runs/rhofold_2oiu/2oiu_rhofold_fullatom.pdb`, and the pipeline's own `assembled.pdb`).

---

## Part 3 — Two negatives worth not re-measuring

- **REMD does not help.** 24 replicas, 2D REST2 x T-REMD, 660 ps per replica: last 8 of 24 blocks
  spread 23.1%, identical to the no-exchange fixture's 23.1%, and *worse* than the plain 400 ps
  no-exchange run (12.3%, trendless). The cold rung holds one replica out of 24, so its effective
  sample is an eighth of the no-exchange 8-replica run and the exchange gain does not cover it.
- **The GPU does not help this workload.** `comfyui`'s ROCm torch picks the Radeon 8060S, and the
  same 2000-step run measures 60.9 steps/s against CPU's 55.8 — inside the noise. The box is
  81 CG particles; the per-step fixed cost dominates, and `GPUCellList` builds its neighbour table
  in numpy on the CPU every step by design (to avoid a ROCm crash). Speeding this up needs the
  neighbour table on the GPU or a much larger system, not a device switch.

---

## Part 4 — Round 0 over 867 chains, and the twelve chains that refused it

The closed loop (Part 1) ran on the full pool for the first time: 867 chains, one replica each,
32500 steps = 65 ps, burn 20000, 33 workers x 1 thread, friction 1.0, wall_k 2000, gain
bb_bond 1 / angle 0.3 / dihedral 1, reference `results/refit_smooth5.npz`. Round 0 took
**75461 s (21.0 h)** and wrote `results/ibi_full/tables_r1.npz`.

| coordinate | status | max\|dU\| | kBT | outside support |
| :-- | :-- | --: | --: | --: |
| dihedral | applied | 5.6201 | 2.2534 | 0.0000% |
| angle | applied | 1.1624 | 0.4661 | 0.0028% |
| bb_bond | **refused** (support_drift) | 3.1204 | 1.2512 | **1.0204%** |
| stack | carried | -- | -- | 2.5210% |

Joint residual over the 867 chains: median **0.1716**, min 0.0300, p75 0.2154, max 2.0542, and it
rises with chain length (L<=34 0.157, 34-60 0.139, 60-120 0.162, 120-300 0.181, >300 0.205). All
three updated coordinates report converged=False, which is what one round should report. Verified
array by array: `tables_r1.npz` differs from `tables_r0.npz` only where the round said it would
-- angle 1.1235 and dihedral 5.8016 as stored (the applied values, after smoothing), bb_bond,
stack, intra_pc and intra_cn bit-identical at 0.0000.

**The refusal is twelve chains, not the field.** 12 of the 867 carry 99.48 percent of every
out-of-support bb_bond count; without them the pooled fraction is 0.0027 percent, 370x below the 1
percent the update allows. 41.2 percent of chains never leave the support, and by length band it
is 0.0015 percent below 60 residues against 1.40 at 60-120 and 2.38 at 120-300. The twelve are
9JHD_3(60), 8D8K_31(114/162/164), 7OYB_1(138), 7QVP_7(119/713), 9AXT_1(195), 9BH5_7(192),
8I9W_2(233), 6XU7_62(595), 9J9I_1(440), at J 1.31-2.05 against a pool median of 0.17.

**They do not start broken; they come apart.** Two candidate causes were measured and both are
dead: the numbering-gap pseudo-bond (the database does contain 17 such chains -- worst 7R6Q_10 at
7.224 nm -- and none of them is one of the twelve), and a modified-nucleotide base atom (all
twelve are canonical ACGU). What they have is a deposited configuration the field cannot hold.
`scripts/diagnose_chain_meltdown.py` ran the loop's own sampling core on the three shortest of
them and a length-matched clean control each, 65 ps, burn 0, 13 blocks of 5 ps:

| run | L | round-0 outside | E at deposit | max\|F\| | closest approach | outside at 5 ps | outside at 65 ps | sigma_sim/ref at 65 ps | bb_bond max at 65 ps |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| BAD 9JHD_3 | 60 | 99.98% | 186919 | 5000 (cap) | 0.026 | 11.9% | 100.0% | 46.1 | 13.6 nm |
| control 5XYM_6 | 60 | 0.00% | 46623 | 5000 (cap) | 0.152 | 3.3% | 0.0% | 0.96 | 0.77 nm |
| BAD 8D8K_31 | 114 | 95.76% | 233206 | 5000 (cap) | 0.021 | 7.4% | 100.0% | 47.0 | 16.5 nm |
| BAD 7QVP_7 | 119 | 83.16% | 207501 | 5000 (cap) | 0.023 | 6.1% | 95.8% | 42.3 | 13.4 nm |
| control 7OYB_1 | 114 | 0.00% | 10639 | 4021 | 0.191 | 0.2% | 0.0% | 0.91 | 0.77 nm |
| control 7OLC_3 | 119 | 0.00% | 7623 | 4780 | 0.249 | 0.1% | 0.0% | 0.95 | 0.77 nm |

(energies in kJ/mol, distances in nm)

The blocks give the mechanism. The deposited state is already at the force cap -- 200000 kJ/mol
against 10000 for the controls -- the beads interpenetrate to 0.02 nm, deep inside the clash wall
where the cap makes the restoring force a constant, and the chain melts inside 5-15 ps. The bond
goes first because the table's edge slopes are exactly 0.0: outside the support the wall is the
entire restoring force, and a clipped wall does not depend on how much further out the pair goes.

**The loop has no minimization step, and that is the difference.** `ibi_core.run_round` starts
Langevin at 300 K from the deposited coordinates; nothing screens a start state that is 20x the
field's energy scale. The twelve melt for a property of the entry point, not of the potential --
and they contaminate the pooled histograms of every coordinate, not only bb_bond's.

Also new: `ibi_loop.py --start-round=N`. Round 1 was killed at its first task (the log ends at
the round header, and the process was gone by morning) and the driver could not start from where
it stopped, so continuing would have cost the 21 hours again. A resume loads `tables_r{N}.npz`,
keeps p_ref from the reference -- refusing when the two do not share bins, measured: resuming the
867 run without `IBI_LOOP_REF` is refused, 1000 bins against the table of record's 120 -- and
rebuilds plan_update's divergence history from round0..N-1.json, because an empty history cannot
fire the rule at all.

### The entry relaxation, measured

`ibi_core.relax_positions` descends the injected Hamiltonian before the sampler's first step --
same field, same cap, normalised direction, halving line search, SHAKE after every accepted step,
and deterministic, so a seed's trajectory is unchanged. Two arms, same chain, same seed, same
32500 steps, burn 0, 13 blocks of 5 ps (`scripts/diagnose_chain_meltdown.py 32500 13 1 {0,1500}`):

| arm | E at deposit | E after relaxation | max\|F\| after | whole-run outside | closest approach | sigma_sim/ref, blocks 1-12 | bb_bond max |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| **9JHD_3, no relaxation** | 186919 | -- | 5000 (cap) | **70.71%** | **0.026 nm** | 4.7 -> 46.1 | 13.6 nm |
| **9JHD_3, relax 1500** | 186901 | **92512** | 5000 (cap) | **0.85%** | **0.193 nm** | 0.86 - 1.05 | 0.79 nm |
| control 5XYM_6, no relaxation | 46623 | -- | 5000 | 0.254% | 0.152 nm | 0.88 - 0.96 | 0.77 nm |
| control 5XYM_6, relax 1500 | 46619 | 14648 | 4545 (left the cap) | 0.254% | 0.297 nm | 0.84 - 0.99 | 0.77 nm |

**The melt is gone.** The bad chain samples a normal bb_bond afterwards (median 0.615 nm, max 0.79,
sigma ratio about 1.0), and its FIRST block is the only one outside the support: 11.0 percent over
0-5 ps and 0.000 percent in every block from 5 to 65 ps. The loop's own protocol burns 40 ps before
it bins, so that transient would not reach the histogram at all; the number that matters for the 1
percent refusal is the whole-run 0.85 percent against round 0's 1.0204 percent, which the update
accepts.

**The relaxation is not converged, and that is the number to carry forward.** All 1500 steps were
accepted, not one rejected; E only halved; max|F| never left the cap, so the descent was still
going when it ran out of steps. The chain ends at 92512 kJ/mol against the control's 14648 -- the
deposited state of 9JHD_3 is genuinely far from anything this field's capped descent reaches
quickly. 1500 steps cost about 1500 field evaluations, roughly 5 percent of one chain's round;
10000 would cost about 30 percent and might leave the cap. What the length should be is a cost
decision, and these arms bound it from below: 1500 is already enough to stop the melt.

**A chain that was fine is unaffected.** The control's sampled distribution is the same with and
without the relaxation (0.2542 percent in both arms, identical block table), so this is a repair
of the twelve, not a change of protocol for the 855.

The entry state is recorded for every chain now: `_sample_one` returns `entry = {energy_0,
max_force_0, at_cap_0}` and `relax`, and the loop's parent copies both into the round json. The
rounds after this one will therefore carry the distribution a gate would be set from -- how many
of the 867 start with the force already clipped -- which round 0 could not answer and which had to
be re-derived by hand from the out-of-support counts.

**Decision, 2026-09-16.** Round 0 is to be redone under the relaxation. The run that is meant to
produce the table is `results/ibi_relax` -- 867 chains, 33 workers, `IBI_LOOP_RELAX=1500`, the
same reference (`refit_smooth5.npz`) and the same gains -- and `results/ibi_full` stays as the
record of what round 0 looked like WITHOUT it. The loop was stopped for this, so the round-1
sampling that was in flight is abandoned rather than finished: its bb_bond column could only have
been refused again, and the twelve melting chains contaminate every coordinate's pooled
histogram, not just the bond's. The suite after the relaxation, the entry record and the resume
path: 190 passed.

## Part 5 — Four rounds on the full pool, and a metric that was flooring itself

The closed loop ran rounds 0-3 on all 867 chains (32 workers, longest-first dispatch after round 2,
entry relaxation 1500 steps, reference `refit_smooth5.npz`). Round timings: 21.0, 9.2 and 18.3 h for
rounds 0-2, and round 3 closed at 21:23 on 2026-09-21; the driver then relaunched itself to eight
rounds.

| round | bb_bond | angle | dihedral | stack | pooled J | per-chain median J |
| --: | --: | --: | --: | --: | --: | --: |
| 0 | 0.957 | 1.299 | 1.233 | 1.164 | 0.167 | 0.1745 |
| 1 | 0.862 | 1.265 | 1.072 | 1.107 | 0.139 | 0.1567 |
| 2 | 0.855 | 1.248 | 1.023 | 1.094 | 0.123 | 0.1536 |
| 3 | 0.856 | 1.228 | 1.005 | 1.083 | 0.112 | 0.1498 |

and the corrections, all applied, all three coordinates every round:

| round | bb_bond | angle | dihedral |
| --: | --: | --: | --: |
| 0 | 2.84 | 3.00 | 5.62 |
| 1 | 1.70 | 3.94 | 1.82 |
| 2 | 0.48 | 3.02 | 0.77 |
| 3 | **0.35** | **4.27** | **0.38** |

Two coordinates converge monotonically; **angle does not** -- 3.00, 3.94, 3.02, 4.27 with no trend,
and by round 3 it is the largest correction in the loop.

### 5.1 More than half of that J is a denominator artefact

`sim_ref_ratio` divides the simulated sigma by `table["sigma"]`, and that sigma is
`float(v.std())` -- the PLAIN standard deviation of every observation in the database
(boltzmann_bonded.py:485). The table's support, however, is built from a ROBUST sigma,
`(q99.9 - q0.1) / 6.58`, times `SUPPORT_SIGMA = 4` -- a window that excludes exactly the
non-physical tail the loader describes: "44 observations of 132695 are above 0.8 nm; a P-P bond of
7 nm does not exist".

Normalise `exp(-U/kBT)` on each table's own bins and compare that distribution's sigma with the
stored one:

| coordinate | sigma_data (J's denominator) | sigma implied by the table | ratio |
| :-- | --: | --: | --: |
| bb_bond | 0.0633 | **0.0540** | **0.853** |
| angle | 0.3203 | 0.3218 | 1.005 |
| dihedral | 0.6299 | 0.6247 | 0.992 |
| stack | 0.1390 | 0.1268 | 0.913 |

**So sim/ref for bb_bond cannot exceed 0.853 whatever the field does** -- and that is where both
operators settle: round 3 measures 0.856 against the data's sigma, a 0.4 percent error against the
table's. `|ln 0.853| = 0.159` is a floor inside every J this project has quoted for the bond, stack
adds 0.091, and the four-coordinate J therefore cannot go below **0.0659** by construction.

Measured against the target the loop is actually aiming at, round 3 is:

| coordinate | sim/ref (data sigma) | sim/ref (table sigma) | verdict |
| :-- | --: | --: | :-- |
| **bb_bond** | 0.856 | **1.004** | **at its target** |
| angle | 1.232 | **1.226** | **+23 percent off its table -- the one real gap** |
| **dihedral** | 1.006 | **1.015** | **at its target** |
| stack | 1.085 | 1.189 | +19 percent, and it has no injection path |

Round 3's J of 0.112 is therefore about 0.066 of artefact and 0.046 of physics, and the physics is
one coordinate.

### 5.2 The moment operator fixes the one that is broken

`docs/archive/plan_b_coupled_update.md`'s operator ran head to head against the marginal inversion on the
seven-chain pool: six rounds, same seed, same 16 replicas, same reference, only the operator
different (8 workers per arm, K=8). The comparison metric is operator-independent -- the Chebyshev
moment difference pooled over the chains, computed offline from the stored histograms:

| round | angle \|dT\|max, table | angle \|dT\|max, moments | angle sim/ref, table | angle sim/ref, moments |
| --: | --: | --: | --: | --: |
| 0 | 0.1335 | 0.1335 | 1.259 | 1.259 |
| 1 | 0.1090 | 0.0780 | 1.221 | 1.246 |
| 2 | 0.1069 | 0.0565 | 1.222 | 1.188 |
| 3 | 0.0820 | 0.0483 | 1.225 | 1.178 |
| 4 | 0.0859 | 0.0450 | 1.205 | 1.141 |
| 5 | 0.0861 | **0.0351** | 1.242 | **1.135** |

Both arms ran all six rounds (exit 0). The same two arms, with every coordinate measured against
the table's OWN sigma -- the fair number, since 5.1 showed the stored sigma is inflated for bb_bond
and stack:

| arm | round | bb_bond | angle | dihedral | stack |
| :-- | --: | --: | --: | --: | --: |
| table | 0 | 1.101 | 1.253 | 1.240 | 1.242 |
| table | 3 | 1.004 | 1.219 | 1.009 | 1.171 |
| table | 5 | **1.004** | **1.199** | **1.015** | 1.158 |
| moments | 0 | 1.101 | 1.253 | 1.240 | 1.242 |
| moments | 3 | 0.998 | 1.172 | 0.994 | 1.137 |
| moments | 5 | **1.000** | **1.130** | **0.970** | 1.110 |

Read off it:

- **bb_bond ends on target under both** (1.004 and 1.000) -- the two operators agree there because
  both are already at the table's own distribution, which is what 5.1 says and what the stalled
  correction (0.48 -> 0.35 kJ/mol) says.
- **dihedral is on target under both** (1.015 and 0.970); the moment operator drifts slightly PAST
  the reference where the inversion settles on it.
- **angle is where they separate**: the inversion stalls at 1.199 with a ringing correction
  (0.082-0.086), the moment operator walks it to **1.130** and is still falling at round 5.
- **stack, which neither arm updates, improves under both** (1.242 -> 1.158 / 1.110), i.e. the
  coupling is doing part of the work and the moment operator's step is a little easier on it.

The verdict against the pre-registered acceptance ("angle's correction series becomes monotone") is
that **the moment operator passes and the marginal inversion does not**. The same evidence says the
operators are equivalent on the two coordinates that were never broken, so this is a change to make
for one coordinate, not a different theory of the field.

### 5.3 What this changes

- **The metric first.** `sim_ref_ratio` should report against the table's implied sigma (a second
  key beside `sigma`), or the fit should use the same robust scale it builds the support from.
  Every J this project has quoted for bb_bond and stack carries the floor.
- **The field is closer to converged than J said**: bb_bond and dihedral are within 1.5 percent of
  their targets, and the residual is angle.
- **Angle's fix is demonstrated and it is the operator, not more rounds**: six rounds of the
  marginal inversion moved it 0.1335 -> 0.0861; six rounds of the moment operator, 0.1335 -> 0.0351.
- **The reference's outliers belong at the loader**, where they are described: a 7 nm distance is
  not a bond, and letting it into `v.std()` inflates the denominator the whole loop is judged by.

## What is open

1. **IBI convergence.** Round 1 of the 867-chain loop is running now (resumed under
   `tables_r1.npz`; round 0's numbers are in Part 4). Whether J descends over rounds 1-3 is the
   question, and angle's correction at gain 0.3 is the coordinate being watched -- it oscillated
   at gain 1.0 (6.21, 8.72, 9.65, 10.46 kJ/mol over four rounds on the 7-chain pool).
2. **The refinement-from-coarse-structure test.** Materials exist; not run.
3. **The oxRNA jump's provenance.** Relax artifact or landscape? Needs a run that skips
   `RNA_relax` or equilibrates properly under the real potential first.
4. **`bb_bond` is still too wide and in the wrong direction** (1.05-1.13 against a single-chain
   ceiling of 0.998, `§3bb`). Unexplained, and no arm moved it. Part 4 adds a separate mechanism
   to keep out of that number: the twelve melting chains are 50-100 percent outside the support on
   their own, so any pooled bb_bond ratio that includes them is measuring the melt as well.
5. **The entry relaxation exists and is measured, but its length is not settled.** 1500 steps on
   the worst chain stops the melt and does not touch a clean chain; it also does not converge (E
   186901 -> 92512 kJ/mol, max|F| still at the cap). Whether to spend 30 percent of a round on
   10000 steps to leave the cap, and whether to pair it with a gate on `at_cap_0` that drops a
   chain loudly instead, are open. What is NOT open any more: round 0's pooled histograms were
   sampled from starts that melt twelve chains, so a run whose tables are meant to be the table
   has to redo round 0 under the relaxation.
5. **The Fourier/tabulated split.** Fourier fixes dihedral but cannot enter the IBI loop (the
   update is defined on a table, and the table's honest max force is 3.3x cap). Whichever way the
   loop goes has to resolve this.


## Part 6 — The operator switch that did not happen, and how it was caught (2026-09-23)

`results/ibi_relax/switch_when_round4_done.ps1` fired on schedule at 16:42 on 2026-09-22: it
archived rounds 0-4, killed the python tree, ran `schtasks /run /tn ibi_relax_loop` and logged
"relaunched: rounds 0-4 replay under the moment operator". Seventeen hours later the banner of the
process it had "relaunched" reads

    gain bb_bond=1 angle=0.3 dihedral=1, relax 5000 steps, operator table

so rounds 5 and 6 were sampled AND updated by the table operator with the old 0.3 angle gain, and
the moment operator never ran at full pool. The process table says what happened:

| process | pid | started |
| :-- | --: | :-- |
| driver (python, `ibi_loop.py`) | 38044 | 2026-09-22 16:42:02 — 20 s BEFORE the switch's own `schtasks /run` |
| its parent, `cmd.exe` | 9068 | **2026-09-21 21:23:24** — the launcher instance of the PREVIOUS run |
| its parent, `svchost.exe` | 3876 | Task Scheduler |

`taskkill /PID <python> /T /F` kills the target and its CHILDREN. The `cmd.exe` running
run_task.cmd is the PARENT, so it survived; when its python child died, cmd resumed reading the
batch file from the byte offset it had saved — in a file that had been edited since (the moment
block was added at 10:56 that morning, changing its length) — and re-executed the python line from
the environment it had built at 21:23 the day before. That environment is the tell: the banner's
`angle=0.3` is not the default (the default follows `IBI_LOOP_GAIN=1.0`), it is the value the OLD
launcher carried, and `operator` fell back to its `table` default because the old environment had no
`IBI_LOOP_OPERATOR` at all.

Two rules follow, and both are cheap:

1. **STOP THE TASK, NOT THE PYTHON.** Kill the `cmd.exe` parent (`taskkill /PID <cmd> /T /F`),
   verify no python running `ibi_loop.py` is left, and wait for
   `Get-ScheduledTask ibi_relax_loop` to read Ready before `schtasks /run`. A launcher killed at the
   python level comes back with yesterday's environment, which is worse than not restarting at all.
2. **READ THE BANNER AFTER EVERY RESTART.** `operator table` against a launcher that sets
   `IBI_LOOP_OPERATOR=moments` is the entire symptom; nothing else in the log names the operator
   unless it is moments (each coordinate prints `moments |d<T>|max=...` when it is).

Also fixed while diagnosing: run_task.cmd was written with bare LF line endings, which cmd.exe
tolerates in a short file but is exactly the condition under which its byte-offset resume is unsafe.
It is now CRLF, and a probe of the real layout confirms what a fresh cmd reads out of it:
`OPERATOR=moments GAIN_ANGLE=1.0 K=8 RELAX=5000 ATTEMPTS=60 POOL=all`.

**What the restart bought, beyond the fix.** The relaunched run (2026-09-23 23:30:38, nine rounds)
replays rounds 0-5 under the moment operator from the SAME task files the table operator consumed, so
the two operators are compared on identical evidence and no sampling:

| round | table: bb_bond / angle / dihedral | moments: bb_bond / angle / dihedral |
| --: | --: | --: |
| 0 | 2.84 / 3.00 / 5.62 | 2.63 / 2.56 / 3.74 |
| 1 | 1.70 / 3.94 / 1.82 | 1.06 / 2.69 / 0.83 |
| 2 | 0.48 / 3.02 / 0.77 | 0.15 / 2.44 / 0.39 |
| 3 | 0.35 / 4.27 / 0.38 | 0.15 / 2.08 / 0.39 |
| 4 | 0.72 / 4.31 / 0.34 | 0.31 / 1.89 / 0.38 |
| 5 | 0.62 / 3.72 / 0.27 | 0.22 / 1.63 / 0.34 |

(max|dU| in kJ/mol; `scripts/ibi_round_report.py` prints this table plus the per-chain residuals.)
The table operator's angle rings between 3.0 and 4.3 for six rounds with no trend; the moment
operator walks it down monotonically, 2.56 -> 2.69 -> 2.44 -> 2.08 -> 1.89 -> 1.63, which is the
seven-chain A/B's finding reproduced at 867 chains. TWO CAVEATS, both load-bearing: the replayed
rounds 1-5 are retrospective — their histograms were sampled under the TABLE operator's fields, so
the moments sequence is a counterfactual one — and only rounds 6-8, which sample under the
moments-rebuilt tables, are moments rounds in the full sense. The 548 round-6 chains sampled under
the table operator's `tables_r6.npz` were moved to `results/ibi_relax/tasks_r6_tableop_discarded/`
rather than mixed into a round whose field had changed.


## Part 7 — The table-sigma denominator moves because the TABLE moves: round 6's angle number (2026-09-25)

`sim_ref_table` = sigma_sim / sigma_implied_by_the_table was added to remove the floor the stored
(outlier-inflated) sigma puts under bb_bond and stack. It does that, and it has a failure mode of its
own that round 6 walked into. The decomposition, from the pooled task histograms and the table files
already on disk:

| angle | stored sigma | implied sigma | pooled sigma_sim | sim/implied | sim/stored |
| :-- | --: | --: | --: | --: | --: |
| table operator r5 (sampled in round 5) | 0.32027 | 0.25731 | 0.38847 | 1.510 | 1.213 |
| moment replay r5 | 0.32027 | 0.15112 | — | — | — |
| moment r6 (sampled in round 6) | 0.32027 | 0.14390 | 0.35903 | 2.495 | 1.121 |
| moment r7 (sampling now) | 0.32027 | 0.13583 | — | — | — |

The per-chain median of that ratio looks like a regression from round 5 to round 6 (1.373 -> 2.257,
with the per-chain J_table median 0.234 -> 0.328). It is not one. The SIMULATION's own width improved
(sigma_sim/stored 1.213 -> 1.121, stored-sigma per-chain J median 0.1458 -> 0.1273); what moved is
the table: the moment operator's replay narrowed angle's implied width by 41 percent in one step
(0.257 -> 0.151) and it has kept narrowing (0.144, then 0.136).

**Why it narrows is not a defect of the operator either.** sigma_implied is the width the table ALONE
would produce, and the sampled marginal is not the table's own Boltzmann distribution — it is the
table convolved with the rest of the Hamiltonian. A moment-matching step that compares the sampled
histogram against the reference's moments is therefore asking the table to make up, by itself, for a
width the coupling contributes. The table it settles on is narrower than the target, and the ratio
is then a statement about the coupling rather than about the field.

**The asymmetry is the finding.** The same replay left the other three coordinates alone — implied
sigma bb_bond 0.04804 -> 0.04843, dihedral 0.47592 -> 0.46641, stack 0.12683 -> 0.12683, all inside
2 percent — against angle's 41 percent. So the angle is the coordinate whose sampled marginal its own
table explains least, which is the same coordinate that has needed a different operator, a different
gain, and six rounds of attention since round 0. Rule for reading the metric: quote the implied sigma
beside `sim_ref_table`, and do not use the ratio as a field-quality number for a coupled coordinate.
For bb_bond, dihedral and stack the table's own sigma remains the better denominator; for angle it
is not.

The moment operator's own convergence signal moved the other way over the same pair of rounds and is
the one to trust: `|d<T>|max` 0.0280 / 0.0470 / 0.0366 at round 6 against 0.0020 / 0.0667 / 0.0087
at round 5 (bb_bond / angle / dihedral), with the estimated relative-entropy drop staying below
0.014 kBT per round.


## Part 8 — The angle has no fixed point under the moment operator (2026-09-27)

Part 7 left one observation unexplained: under the moment operator the angle's table implied sigma
falls every round (0.15112, 0.14390, 0.13583, 0.13065 over rounds 5-8) while the other three
coordinates move by under 2 percent. `scripts/ibi_angle_narrowing.py` separates the two readings
that fit those numbers -- a coupling-compensated FIXED POINT, or an update that is being ABSORBED --
using only the task files and tables on disk. The answer is the second one, and three independent
signatures say so.

**1. The sampled marginal does not follow the table.** Pooled over all 867 chains, per round:

| round | angle sigma_sim | implied sigma | sim/implied | edge mass | |d<T>|max |
| --: | --: | --: | --: | --: | --: |
| 5 | 0.38847 | 0.15112 | 2.571 | 0.105 | 0.0667 |
| 6 | 0.35903 | 0.14390 | 2.495 | 0.111 | 0.0470 |
| 7 | 0.35413 | 0.13583 | 2.607 | 0.115 | 0.0223 |
| 8 | 0.34930 | 0.13065 | 2.674 | 0.114 | 0.0214 |

The reference's own angle sigma is 0.32176. So the sampled width sits 9 percent ABOVE the target and
falls 1.4 percent per round, while the table's implied width sits 2.5x BELOW the target and falls 4
percent per round: the two are separating, not meeting. A fixed point would have both rates going to
zero together; neither does.

**2. The operator's own residual stops falling.** |d<T>|max 0.0470 -> 0.0223 -> 0.0214: the last
ratio is 0.96 against 0.47 before it, and 0.021 is about 200x the histogram's own noise floor (of
order 1e-4 at 3.3e8 samples per coordinate-round). A correction of 1.68 kJ/mol that no longer reduces
the quantity it is computed from is a correction the coupling is absorbing.

**3. The step direction becomes CONSISTENT, which is the opposite of the dihedral's cycle.** The
angle's consecutive table steps correlate +0.176 then **+0.896**; the dihedral's 2-cycle read -0.853
then -0.936. A positive correlation that grows means the loop is walking in one direction, not
overshooting back and forth: the mean drift is +1.098 kJ/mol per round at bin 999 (x = 0.909, the
right edge) with rms 0.490 across the support, and the scatter about that drift is 0.442 -- a drift
with an overshoot superimposed, not a cycle.

**The mechanism is the one the basis project found from the other side.** Measured offline on the
seven-chain pool, the angle's target carries edge mass that no family can move (the fitted-minus-
target edge gap stays +0.10..+0.29 for Chebyshev, B-splines and the table alike), and the K=8
Chebyshev residual on it is 0.4-2.2 kJ/mol depending on ridge. So the operator keeps asking for the
same thing, the basis cannot deliver it, and the table walks: the correction is absorbed by the
coupling instead of changing the sampled edge.

**What this means for the next campaign.** Both operators fail on the angle at full pool, in
different ways: the table operator rings between 3.0 and 4.3 kJ/mol for six rounds, and the moment
operator drifts without a fixed point. That is one coordinate, and it is the coordinate whose
sampled marginal its own table explains least -- the coupling dominates it (Part 7). Note the
cross-link to Plan C: there the angle converged (its clean inter-round distance fell to 2.0-2.4x the
same-field floor) on a SELF-CONSISTENT target, on a seven-chain pool. The pooled deposited marginal
for the angle may simply be unattainable through this coupling, in which case no operator will close
it and the target, not the update, is what has to change for this coordinate.


## Part 9 — The edge-gap diagnostic transfers to the full pool, and it names both pathologies (2026-09-27)

The basis project's phase 2 ended with a cheap quantity replacing an expensive one: what predicts
whether a fit will cycle is not its residual but its EDGE GAP -- the outer 5 percent mass of the
fitted field's own implied distribution against the target's. The only cycling arm (Chebyshev K=8 on
the dihedral) had gap -0.378; the four that converged sat inside |gap| <= 0.012, including the two
with the worst residuals. `scripts/ibi_edge_gap.py` applies that quantity to the production record,
where it needs no sampling: the tables and the pooled histograms are both on disk.

| round | coordinate | reference edge | table's own edge | gap | sampled edge | gap |
| --: | :-- | --: | --: | --: | --: | --: |
| 5 | angle | 0.1178 | 0.1538 | **+0.0359** | 0.1048 | -0.0131 |
| 6 | angle | 0.1178 | 0.1621 | **+0.0442** | 0.1113 | -0.0065 |
| 7 | angle | 0.1178 | 0.1713 | **+0.0535** | 0.1148 | -0.0030 |
| 8 | angle | 0.1178 | 0.1750 | **+0.0571** | 0.1144 | -0.0034 |
| 5 | dihedral | 0.3243 | 0.1699 | **-0.1543** | 0.3183 | -0.0060 |
| 6 | dihedral | 0.3243 | 0.1747 | -0.1496 | 0.3612 | +0.0369 |
| 7 | dihedral | 0.3243 | 0.1598 | -0.1645 | 0.3320 | +0.0077 |
| 8 | dihedral | 0.3243 | 0.1515 | **-0.1728** | 0.3228 | -0.0015 |
| 5-8 | bb_bond | 0.0006 | 0.0006-0.0009 | within +-0.0003 | 0.0007 | within +-0.0003 |

(outer 5 percent = the outermost 50 of the table's 1000 bins at each end; the reference is
`refit_smooth5.npz`, the same fixed target every round.)

Three readings, and the first two are the point:

1. **The dihedral's table under-delivers its edges by half, and it worsens every round**
   (-0.154 -> -0.173). That is the same sign and the same order as the deficit the basis arms
   measured on the seven-chain pool and then REMOVED by swapping Chebyshev K=8 for a B-spline basis,
   which is also what stopped that arm's cycle. The production loop ran K=8 on the dihedral for all
   nine rounds, so it carried the pathology the whole time, and the diagnostic transfers from seven
   chains to 867.
2. **The angle's table over-delivers its edges, monotonically** (+0.036 -> +0.057, about 1.5x the
   reference), while its SAMPLED edge moves the other way, closing from -0.013 to -0.003. The table
   is walked outward by corrections the simulation does not take -- Part 8's absorbed update, seen
   by a diagnostic that never looks at the moment residual.
3. **The tables and the sampled marginals disagree, and that is the coupling again.** The sampled
   edge mass sits within 0.04 of the reference for all three coordinates at every round, while the
   tables are off by 0.06 (angle) and 0.17 (dihedral). A table's own implied distribution is not the
   distribution the chain samples (Part 7), which is why a TABLE-side edge gap predicts a cycle: the
   fit is trying to place mass where the coupling will not let it stay.

Not claimed here: the bb_bond gap is zero at full pool (the +0.238 the basis arms measured for
Chebyshev was a seven-chain number), and there is no arm-level evidence for K=16 or K=32.

## Part 10 — The A-round verdict: a full-strength replacement overshoots, and one refusal starves another coordinate (2026-10-04)

Parts 8 and 9 diagnosed the production loop's cycle from its own record. Arm A is the intervention
those diagnoses implied: a fresh two-round loop on the same 867-chain pool with one rule per
coordinate, each chosen by a different measurement.

| coordinate | rule | why that rule |
| :-- | :-- | :-- |
| bb_bond | moments | its table already sits within 0.0003 of the target's edge; keep the change smallest |
| angle | selfconsistent | Part 8: the moment operator is absorbed by the coupling |
| dihedral | bspline16 | Part 9's edge-gap arm: B-spline m=16 held \|gap\| <= 0.012 where Chebyshev K=8 was -0.378 |

Round 0 ran 17.05 h, round 1 17.00 h (61 383 + 61 210 s), 867 of 867 chains in both, and both criteria
the basis project set for "the cycle is broken" fail on the round 1 tables:

| round | bb_bond | angle | dihedral | per-chain J median | J_all median |
| --: | :-- | --: | --: | --: | --: |
| 0 | refused | applied, max \|dU\| 23.94 kJ/mol (9.60 kBT) | applied, 4.70 (1.89 kBT) | 0.1623 | 0.1716 |
| 1 | refused | applied, 9.95 (3.99 kBT) | applied, 2.63 (1.06 kBT) | **0.2343** | **0.3043** |

Round 0's table columns are the starting tables by construction (every gap exactly zero, every implied
sigma equal to the target); the informative columns are round 1's, measured on the tables round 0 applied
and sampled by the chains that ran under them.

| round | coordinate | sampled sigma | table implied | sim/implied | edge gap |
| --: | :-- | --: | --: | --: | --: |
| 0 | angle | 0.41746 | 0.32176 | 1.297 | 0.0000 |
| 0 | dihedral | 0.77650 | 0.62471 | 1.243 | 0.0000 |
| 1 | angle | 0.54072 | **0.68491** | **0.789** | **+0.3393** |
| 1 | dihedral | 0.74991 | 0.51862 | 1.446 | **-0.1368** |

(targets, refit_smooth5: angle sigma 0.32176, dihedral 0.62471.)

1. **The angle's full-strength replacement overshot by a factor 2.1 in implied width, and the sampler
   followed it 30 percent.** Round 0's self-consistent update -- maximum |dU| = 9.6 kBT, the largest
   update this campaign has ever applied -- installed a table whose implied sigma is 0.685 against a
   0.322 target, and the pool then sampled 0.417 -> 0.541. A rule that is right when the ensemble
   matches the table (Part 8) is not safe when it does not: the update is a ratio of two histograms
   that are 30 percent apart in width, and nothing in the rule caps how far one step may go.
2. **Correcting the angle starved the dihedral.** Round 1's dihedral update was fitted to chains whose
   angle marginal had just moved; its implied sigma fell 0.625 -> 0.519 (edge gap -0.137) while its
   sampled width barely moved (0.7765 -> 0.7499). That is Part 9's signature read in the opposite
   direction, and it is the coupling: the two applied updates are not independent, so an arm that lets
   each coordinate take its own step can have one coordinate's overshoot disable the other's correction.
3. **bb_bond never updated at all.** Both rounds refused it on support drift -- 1.02 then 1.08 percent
   of observations outside [0.373417, 0.809522] against a 1 percent gate (3 385 037 and 3 594 522 of
   331 737 500). The gate exists so the moments are not taken on a truncated distribution, and it did
   its job; the cost is that the one coordinate whose rule was already known to work contributed
   nothing, in an arm whose whole point was to let the working rules work.

Two lessons, both applicable without a new arm, and the second is the more expensive one:

- **Blend, do not replace.** `U_new = (1 - g) U_old + g U_fit` with g in 0.1-0.5 bounds a single step
  by the field it is a correction to, which is exactly what fails when the ensemble and the table
  disagree by 30 percent. The production loop's own moment operator is implicitly a small-g step (it
  fits a correction, not a distribution); the self-consistent and B-spline rules are not.
- **A basis that fixes the seven-chain pool does not automatically fix the full one.** The B-spline
  family that measured +0.012 at seven chains measured -0.137 here, the same sign and order as the
  production loop's Chebyshev K=8 (-0.154 to -0.173). Whatever the seven-chain pool leaves out, it is
  not only the basis order -- so the cheap pool is a screen for a *rule*, not a stand-in for the pool.

## Part 12 — The criterion was the defect: the campaign converged on the ensemble while its tables churned (2026-10-04)

Part 10 judged arm A on table-side quantities and it failed. The obvious next move was another arm. The
cheaper move was to ask what the table-side quantities are a proxy FOR, and the record was already on
disk: nine campaign rounds, each with the table it sampled under (tables_r{r}.npz) and the pooled
histogram it produced (tasks_r{r}, 867 chains). scripts/ibi_transmission_scan.py regresses one against
the other -- sampled sigma against the table's own implied sigma -- and that slope is the loop's
effective gain, per coordinate:

| coordinate | slope | R2 | intercept (pooled sigma at an infinitely narrow table) | target sigma |
| :-- | --: | --: | --: | --: |
| bb_bond | 1.111 | 0.94 | 0.0004 | 0.0540 |
| angle | 0.311 | 0.70 | **0.3253** | **0.3218** |
| dihedral | 0.970 | 0.82 | 0.159 | 0.625 |
| stack | 0.019 | 0.00 | 0.146 | 0.127 |

1. **The angle was at the model's floor, and the loop kept pushing a table nobody was sampling.** Its
   line says a table change reaches the ensemble at about a third of its size, and that the pooled width
   cannot go below 0.3253 -- one percent ABOVE the 0.3218 target. The campaign drove the angle's implied
   sigma 0.3218 -> 0.1307 (-59 percent) while the sampled sigma came down 0.4159 -> 0.3493 (that is all
   it could do) and the table kept walking away from the target. Nothing was wrong with the angle.
2. **bb_bond is the one coordinate the loop FINISHED, at slope 1.11**, and arm A refused it both
   rounds on a 1 percent support gate. A knob with unit gain was locked out to protect the moments from
   a tail whose bias is bounded by f*d = 0.0102 x 0.18 nm = 0.0018 nm, 3.4 percent of the target sigma.
3. **The dihedral's two estimates disagree, and the disagreement is the coupling.** 0.97 over the
   campaign, 0.251 in arm A -- in the same table-move range (-17 percent implied). The difference
   between the two runs is what the ANGLE did at the same time: the campaign narrowed the angle's
   implied sigma, arm A widened it 2.1x. So a dihedral step measured while the angle moves is
   measuring both.

THE VERDICT, ON THE SAMPLED MARGINAL (scripts/ibi_verdict.py, criteria |ln(sigma_sampled/sigma_target)|
<= 0.10 and |edge gap| <= 0.05, both taken from the record's own scatter and from the basis project's
frozen edge tolerance):

| round | bb_bond | angle | dihedral |
| --: | :-- | :-- | :-- |
| 0 | sampled 0.0606 (FAIL 0.122) | 0.4159 (FAIL 0.260) | 0.7770 (FAIL 0.218) |
| 1 | **PASS** 0.0546 (0.011) | 0.4050 (FAIL 0.231) | **PASS** 0.6753 (0.078) |
| 3 | PASS (0.004) | 0.3944 (FAIL 0.204) | PASS (0.015) |
| 7 | PASS (0.003) | **PASS** 0.3541 (0.096) | PASS (0.024) |
| 8 | PASS 0.05405 (0.001) | PASS 0.34930 (0.082) | PASS 0.61074 (0.023) |

**Every controlled coordinate of the production campaign passes at round 8, on the sampled marginal the
field is actually asked to reproduce**, and every one of them passed at least five rounds before the
campaign was stopped. The stack is excluded as DERIVED, not judged: it has no table, its implied sigma
is identical to ten digits in all nine rounds, and its pooled width tracks the angle's.

So the "limit cycle" of Parts 8-10 is a TABLE-side phenomenon. The loop's convergence monitors watch the
size of the update -- max|dU|, the moment norm, the divergence guard -- and an update can shrink, ring,
or grow while the ensemble it describes sits still. Part 8 measured the same thing from one side (the
moment operator narrowing a marginal the sampler ignores); Part 9 saw it in the edge gap; this is the
same fact stated as a number that can be used: **transmission, per coordinate, and where its line
crosses the target.**

WHAT CHANGED IN THE CODE, each because of a measurement above:

* `IBI_LOOP_SUPPORT_GATE` (default 0.01, unchanged; the arms set 0.03): the gate was already a
  parameter of both operators, and the driver now passes it and prints it in the header. The
  out-of-support fraction was always recorded; now it is recorded for coordinates that are REFUSED and
  for coordinates that are FROZEN too, which is where arm A's bb_bond lived.
* Every round json now carries a `marginals` block per coordinate -- sampled sigma and edge, the
  target's sigma and edge, the table's implied sigma and edge before and after, the ratios, and the two
  edge gaps. Verified against the record: it reproduces arm A's applied tables to the digit (angle
  implied 0.68491, dihedral 0.51862).
* `scripts/ibi_verdict.py` judges an arm on the sampled marginal and reports the table-side numbers as
  diagnostics, with the transmission slope beside them, and `scripts/ibi_step_gain_scan.py` replayed
  each rule across a gain ladder to size the damping rather than guess it: the angle's self-consistent
  refit overshoots its OWN target by 1.64x at gain 1 (implied sigma 0.685 against the sigma 0.417 of the
  ensemble it was fitted to) and is neutral at gain 0.3 (0.41281 against 0.41746); the dihedral's table
  edge gap is monotone in gain, -0.016 at 0.1 and -0.137 at 1.0, so a half step costs half the edge
  mass arm A lost.
* `ibi_core.campaign_root()`: the campaign record is no longer at results/ibi_relax -- a housekeeping
  pass parked it on 2026-10-01 and only part of it came back under _strays -- so the scripts that read
  it resolve it instead of hard-coding a path that stopped existing. Eight of them still hard-code it;
  the list is in the resolver's docstring.

WHAT IS RUNNING, and what each one decides. Two arms, launched together, neither a repeat of arm A:

* `results/plan_c/run_dih7.cmd` -- seven chains, three rounds, production protocol, the angle and
  bb_bond FROZEN, the dihedral alone at B-spline m=16 and gain 0.5 with the gate at 0.03. This is the
  designed experiment for point 3: with nothing else moving, the dihedral's own transmission is
  measurable, and its edge gap either holds (a stable table exists) or does not.
  **RESULT (2026-10-05, 33.5 min for all three rounds).** With the angle and bb_bond frozen the dihedral's
  transmission is **0.802 (R2 0.937)** over the three rounds -- against arm A's 0.251, and near the
  campaign's confounded 0.970. So arm A's "the dihedral does not respond" was the ANGLE cancelling it, and
  the coupling of Parts 8-10 now has a number instead of an explanation. Two more readings from the same
  arm: the step size falls monotonically (2.74 -> 2.06 -> 1.65 kJ/mol) and the implied-sigma ratio climbs
  towards 1 (0.918 -> 0.954 -> 0.967), i.e. **a table-side fixed point exists once the finished
  coordinates are frozen** -- which the full-pool campaign never had; and the pooled dihedral width came
  down 0.7587 -> 0.7051 -> 0.6994 against a 0.6247 target, with the first step transmitting 1:1 and the
  second only 0.22, so the response saturates before the target. The two frozen coordinates score FAIL on
  the sampled criterion in this arm (bb_bond 0.0600 vs 0.0540, angle 0.3964 vs 0.3218): the seven-chain
  pool samples different marginals from the full pool, which is Part 10's second lesson showing up as a
  construction detail rather than a result.
* `results/plan_c/run_ab2oiu10.cmd` -- the 2OIU A/B through the shipped GPU refiner, TEN draws per arm
  instead of one, so the dihedral effect (|ln(sd/target)| 0.635 analytic against 0.307 fitted in the
  single draw) becomes a paired mean +- sd with a win count. The arms share the input and the draw
  index, so the paired difference ln(sd_tables/sd_analytic) cancels the scatter the single draw carried.

NOT CLAIMED HERE. The sampled-side pass is at the POOLED equilibrium marginal, which is what the tables
were fitted to; it is not a claim about any single structure's geometry -- the 2OIU repeats are that
claim, and they are running. The dihedral's transmission is still two numbers (0.97 and 0.25) until the
frozen-angle arm reports. And the tolerances, 10 percent in width and 0.05 in edge, are choices read off
the record's own round-to-round scatter (1-3 percent) and the basis project's frozen edge tolerance;
they are not derived.

## Part 13 — Measured: the stacking bottleneck is base placement, not the CG force field (2026-10-05)

Part 12 ended with the field passing on every coordinate it controls, and with stacking still
unexplained: K_STACK = 0, the scored stack coordinate is the algebraic P(i)-P(i+2) distance, and every
base in the product is placed by a 1EHZ template plus an axis heuristic rather than by anything the CG
stage sampled. The question that decides where to spend the next month is which of those actually costs
stacking, so it was measured rather than argued (scripts/measure_base_stacking.py).

WHAT IS MEASURED. A HELICAL STEP is a WC pair (i, j) whose next pair along the helix (i+1, j-1) also
exists -- the only place stacking is expected, since sequence-adjacent bases across a bulge are not
supposed to stack. For each step: the centroid separation, the angle between the base-plane normals
theta, the RISE (separation along the mean normal) and the TWIST (in-plane rotation between the two
glycosidic directions). A step counts as STACKED when its rise is 2.5-4.0 A and theta <= 30 degrees.
Pairs are detected geometrically from the H-bond contacts (A N1-U N3 / A N6-U O4, G N1-C N3 / G O6-C N4 /
G N2-C O2, 3.6 A cutoff), so the same instrument applies to a deposit, to a reconstruction and to a CG
product with no second source of truth.

THE INSTRUMENT, CALIBRATED ON CRYSTALS, over 20 fragments of _cgdata/combined:

| | steps | stacked | rise | theta | twist |
| :-- | --: | --: | --: | --: | --: |
| crystal | 226 | 98.8% +- 5.4 | 3.353 +- 0.075 A | 8.3 deg | 31.3 +- 2.4 deg |
| reconstruction, trace EXACT | 226 | **43.1% +- 22.7** | **2.496 +- 0.274 A** | 16.9 deg | 29.3 +- 2.8 deg |

Two pitfalls found while building it, both recorded because each produced a wrong answer first: the
plane normal comes out of an SVD with an arbitrary sign, so a perfectly stacked pair can read as
theta = 180 degrees (2OIU's crystal scored 50 percent stacked until the normals were sign-aligned); and
the criterion has to be the rise, not the centroid separation, because A-form bases are 4.2-4.8 A apart
along a helix whose rise is 3.4 A, so a 4.5 A separation cutoff rejects correct steps.

(A) THE TEMPLATE COSTS MORE THAN THE TRACE DOES. With the trace taken from the crystal itself, i.e. zero
trace error, reconstruction loses **57 percent of the stacking** (98.8 -> 43.1), puts the bases 0.9 A too
close (rise 3.35 -> 2.50 A), doubles the normal angle, and **keeps 10-20 percent of the WC contacts**
(2OIU: 0 of 12; 4QK9: 2 of 22; 2QBZ_1: 4 of 21) -- the shipped CG-to-all-atom step reproduces neither the
rise nor the pairing geometry of the structure it is reconstructing.

(B) TRACE ERROR COMPOUNDS IT, measured by perturbing 2OIU's P trace (RMSD quoted after Kabsch):

| trace RMSD | stacked | rise | theta | WC contacts kept |
| --: | --: | --: | --: | --: |
| 0.000 A | 58.3% | 2.49 A | 15.5 deg | 0/12 |
| 0.777 A | 33.3% | 2.49 A | 22.7 deg | 2/12 |
| 1.734 A | 25.0% | 2.18 A | 31.0 deg | 2/12 |
| 2.736 A | 0.0% | 2.47 A | 32.2 deg | 1/12 |
| 3.602 A | 8.3% | 3.44 A | 48.8 deg | 0/12 |
| 5.013 A | 8.3% | 3.41 A | 55.4 deg | 0/12 |

The note in docs/NOTES.md that "a 1.5 A trace error halves the reconstructed stacking fraction" is now a
curve, and it was optimistic in the middle: 0.78 A already costs a third, and by 2.7 A the stacking is
gone; past 3.6 A the number sits at the uncorrelated floor.

(C) WHAT OUR OWN PRODUCTS RECONSTRUCT TO. The ten-draw 2OIU A/B products are P traces of a 300 K CG
ensemble, with no restraint to the deposit (torch_gpu_refine is a sampler: pre-fold, REMD, physical
relaxation -- free to drift). They sit **8.95-11.19 A** from the crystal trace (Kabsch) while keeping
their circular closure (BSJ 0.638-0.718 nm against the crystal's 0.592), and reconstruct to **0-25 percent
stacked**, theta 39-59 degrees, 0-1 of 12 WC contacts. The drift is expected of this stage and is not a
defect; the stacking it carries is nonetheless gone by then, and (A) says most of it was already gone
before the drift.

WHAT THIS DECIDES. The binding constraint on stacking is the CG-to-all-atom BASE PLACEMENT, not the CG
force field: with a perfect trace, the reconstruction alone loses more than half the stacking and almost
all the pairing contacts. So the ordering of work is

1. **base placement first, and it is not a force-field change**: place each base by superposing the
   template on the SAMPLED per-residue frame (P, C4', base site) instead of on the P trace plus an axis
   heuristic, and optionally solve the base orientation against the pair list. The machinery already
   exists -- aform_from_template.real_cg_beads reads those three beads per residue from exactly this
   reconstruction, and its own docstring says fabricated beads "carry no base identity, so no
   base-specific quantity can be expressed on them". This is the cheapest change on the list and it is
   upstream of every stacking number we are unhappy with.
2. **then a base-level stacking term in the CG field** (Part 12's predecessor analysis: the term exists as
   e_stack = 0.5*K_STACK*lam*(d_st - STACK_R0)^2 on the P(i)-P(i+2) distance with K_STACK = 0, i.e. an
   isotropic backbone proxy switched off; oxRNA, at the same three sites per nucleotide, puts it on a
   dedicated stacking site between neighbouring units and gets orientation dependence from the rigid
   body). It only pays off together with (1): today the CG base beads' geometry never reaches the product.
3. **trace accuracy stays worth having** -- 0.78 A costs a third of the stacking -- but it is the part we
   are already doing, and Part 12 says it is the part that is already inside tolerance.

NOT CLAIMED. The WC criterion is strict (all key contacts within 3.6 A); it is calibrated on crystals,
which pass 98.8 percent, so it is a fair test of a structure that should be paired, but it does not say
an amber refinement downstream cannot recover pairing -- that is testable with amber_refine on these same
two structures and it was not run here. (C) measures what a free 300 K CG ensemble reconstructs to, not
how well the pipeline relaxes a deposit; a restrained protocol is a different experiment. And the 20
fragments are the first 20 with an RNA chain of at most 300 nucleotides, not a designed sample.

## Part 14 — Reconstructing from the SAMPLED base frame recovers over half the loss, and needs a field term to finish (2026-10-05)

Part 13 said the reconstruction, not the field, was the binding constraint on stacking. The fix that
follows from it is small: the model already carries three beads per residue (P, C4', N9/N1) and its
intra-residue geometry is rigid by SHAKE, so those three points ARE the base frame -- the roll about the
backbone is information, not something to guess. `aform_from_template.reconstruct_all_atom_from_beads`
fits the 1EHZ template's own P / C4' / N9-N1 onto the sampled beads (a three-point Kabsch per residue,
which fixes the rotation including the roll). The shipped `reconstruct_all_atom` is untouched, and the
pipeline only uses the new path when asked (`cg_frame_allatom=True`).

MEASURED OVER THE SAME 20 FRAGMENTS, same pairs, same instrument:

| reconstruction | stacked | sep | rise | theta | twist |
| :-- | --: | --: | --: | --: | --: |
| crystal (reference) | 98.8% +- 5.4 | 4.271 A | 3.345 +- 0.071 A | 8.2 deg | 31.1 deg |
| P-trace template (shipped) | **43.9% +- 22.7** | 4.180 A | **2.466 +- 0.244 A** | 16.7 deg | 28.8 deg |
| sampled bead frame (new) | **75.7% +- 15.5** | 4.252 A | **3.032 +- 0.148 A** | 15.3 deg | 31.9 deg |

One fragment in full, because the contacts matter as much as the stacking (38JD_1, 20 helical steps):
crystal 100 percent, shipped path 50 percent with 2 of 25 WC contacts kept, bead frame **85 percent with
11 of 25**. So the reconstruction change recovers **roughly half of what the template loses** (43.9 -> 75.7
against a 98.8 crystal), fixes the rise from 0.9 A short to 0.3 A short, and multiplies the surviving
pairing contacts by about five. The residual gap is the template's own rigid-residue idealisation, not the
roll: with three points per residue the base PLANE is still inferred from the template.

AND THEN IT STOPS, for a reason worth stating plainly. On a REAL CG state -- the new `cg_bead_sink` hands
out the 3-bead state the sampler actually moved, which the interface used to slice to P and discard
(torch_gpu_refine line ~236; the state existed all along, REMD carries it across rounds) -- the bead-frame
reconstruction does not deliver: one short 2OIU draw (4 replicas x 5000 steps) landed 7.74 A from the
crystal trace, and there the shipped path gives 0 percent stacked (theta 53.9 deg) and the bead-frame path
gives 16.7 percent with a twist of 100 +- 54 deg. A twist of 100 degrees is not a reconstruction artefact:
it says the SAMPLED base frames are not physically oriented relative to each other. They are not, because
nothing in the field asks them to be -- `K_STACK = 0`, there is no base-orientation term, and the roll is
held only by the K_LINK_* links against a pair potential that does not care about it.

So the two changes are a PAIR, and the measurement says neither works alone:

  * the reconstruction change is necessary -- without it nothing base-level can reach the product, and it
    is worth 32 percentage points of stacking on ideal input;
  * a base-level orientation/stacking term is necessary too -- without it the sampled frames the
    reconstruction consumes are arbitrary, and the reconstruction faithfully reproduces arbitrariness.

THAT also reframes what the next arm should fit. The base-level coordinates (rise, twist, normal angle,
base-base distance) are now measurable on BOTH sides of the loop: the target from the crystal database
with this instrument, and what the current field produces from the bead sink on a real run. That pair of
distributions is the loop's input, and it can be measured before anything is parameterised -- the same
order that made Part 12 possible.

NOT CLAIMED. The real-run comparison is ONE short draw at 7.74 A drift, with the twist spread at +- 54 deg,
so it is a demonstration that the sampled roll is currently meaningless and not a converged measurement of
it; the ten-draw 2OIU products were produced before the sink existed and carry no beads. The 75.7 percent
is on ideal input (the crystal's own rigid units), which is the right number for validating a
reconstruction and the wrong number for claiming a product.

## Part 15 — The base-level marginals, measured: what a stacking term would have to fix (2026-10-05)

Part 14 ended with a pair of conclusions: the reconstruction must consume the sampled base frame, and the
field must have a term that makes that frame mean something. The input to such a term is a pair of
distributions, and both sides are now measurable (`scripts/measure_base_coords.py`). Every coordinate is
a function of the three beads the model carries, because a CG potential can only be a function of those:

    nb_dist   |N_i - N_{i+1}|                base-base separation
    cc_dist   |C4'_i - C4'_{i+1}|            sugar-sugar separation (control: this one IS restrained)
    rise      (N_{i+1} - N_i) . n_mean        separation along the mean triangle normal
    theta     angle(n_i, n_{i+1})             triangle-normal angle, sign-aligned first
    twist     in-plane rotation of P->N about n_mean,   n_i = normalize((C4'_i - P_i) x (N_i - P_i))

TARGET: every consecutive pair over 20 crystal fragments (1625 pairs; the helical subset is 609).
SAMPLED: `ibi_core.run_round` -- the sampler the loop itself uses, production tables installed -- on the
loader's own 3-bead chains, 5000 steps each at stride 25, four chains, 17 920 pair observations.

| coordinate | TV (target vs sampled) | target mean +- sd | sampled mean +- sd | target's OWN internal TV |
| :-- | --: | --: | --: | --: |
| nb_dist | **0.534** | 0.570 +- 0.198 nm | **0.703 +- 0.171 nm** | 0.118 |
| cc_dist | 0.490 | 0.625 +- 0.120 nm | 0.614 +- 0.087 nm | 0.096 |
| rise | **0.551** | 0.366 +- 0.193 nm | 0.337 +- **0.350 nm** | 0.068 |
| theta | 0.464 | 28.6 +- 21.1 deg | **49.4 +- 20.9 deg** | 0.099 |
| twist | 0.387 | 44.1 +- 34.2 deg | 51.9 +- 42.7 deg | 0.114 |

FOUR READINGS, and the third is the one that says what the term is for:

1. **Neighbouring base frames are much more mutually rotated than the crystals**: theta 49.4 against 28.6
   degrees. Nothing in the current field couples the two triangles' orientations.
2. **The bases sit further apart, not closer**: nb_dist 0.703 against 0.570 nm. A stacking term is an
   attraction between bases, and there is none.
3. **The rise is nearly symmetric about zero, where the target is a sharp forward offset.** The target's
   rise is 0.366 +- 0.193 nm with a 5th percentile at +0.02; the model's is 0.337 +- **0.350** with a 5th
   percentile at **-0.43 nm**, i.e. the model has no preference for one base lying OVER its neighbour
   rather than beside or under it. That asymmetry IS stacking, and it is the quantity a base-level term
   has to create. Its spread alone (sd 0.35 against 0.19) is the size of the effect.
4. **The reference is not ambiguous, which is what makes the gap readable.** The target's own internal
   spread -- all consecutive pairs against the helical subset, i.e. loop residues against stacked ones --
   is TV 0.07-0.12, five times smaller than the 0.39-0.55 gap to the model. So the disagreement is a model
   deficiency and not an artefact of how "stacked" was defined.

ONE CAVEAT ON THE INSTRUMENT, measured on the same 20 fragments: the triangle P-C4'-N is not the base
plane. Its normal is off the true base-plane normal (from the ring atoms) by a mean of **20.2 degrees**,
median 16.2. Both sides of the table above use the same proxy, so the comparison is fair, but a term fitted
on this theta carries that systematic, and a target built from ring atoms would be the honest one to fit
against if the model ever gains a bead that can express a plane.

TWO INCIDENTAL FINDINGS, both recorded because both cost a run:

* **The 24-34 residue band is 31 chains, and the loader's order is NOT stable across processes.** Measured:
  two calls in the same session returned different members first. `ibi_loop`'s "small" pool takes the first
  N of that list, so its seven-chain arms are not necessarily the same seven chains between runs -- which
  matters for comparing `run_dih7` against the earlier seven-chain arms. The measurement script sorts by
  name before it indexes anything.
* **A chain in the band hangs `run_round` reproducibly** -- twice, at the sixth chain of a
  nondeterministically ordered pool, with five complete chains' work lost the first time. The defence is
  structural and now in place: one process per chain under a 420 s timeout, so a hang costs one chain. The
  chain itself is not identified, because the order was not stable enough to name it.

## Part 16 — A base-level stacking term, built from the target: the model can move, the term as shaped cannot (2026-10-05)

THE INGREDIENTS, each a measurement:

* **A bead-to-plane map that is exact and cheap** (`base_frames.py`). A CG potential can only see the beads,
  and the base PLANE is not determined by three points -- it takes the rigid template to say where the ring
  sits. Because the template is rigid, the plane normal is a FIXED linear combination of the three template
  vectors e1 = C4'-P, e2 = N-P, e3 = e1 x e2, so the map is a few cross products per residue, no per-step
  reconstruction and no per-step SVD. It reproduces the template's own ring plane to 0.000 degrees per base,
  and ONE pooled triple covers all four bases to 6.573 degrees -- base identity is not needed, which matters
  because the sampler does not carry it (load_structures returns name, pairs, pos).
  Two traps, both measured: the sign of an SVD plane normal is arbitrary, and without a convention
  (normal along +e3) the pooled triple misses by 77.5 degrees; and the map is scale-sensitive, because the
  cross-product term scales with length squared, so feeding nm beads to Angstrom coefficients reads 60.9
  degrees instead of 17.6.
* **The target, in the units the map gives** (`measure_base_coords.py --planes`, 20 fragments, 1625
  consecutive pairs): d = 0.570 +- 0.198 nm, rise = 0.328 +- 0.188 nm with a 5th percentile at +0.06 (the
  target is ONE-SIDED), theta = 24.5 +- 22.2 deg (median 15.9). Taken through the SAME map, so these are the
  numbers a bead-level term can be held to; the ring-atom version is d = 0.570, rise = 0.334 +- 0.216,
  theta = 18.5 +- 20.4, and the map itself is 17.6 degrees (median 11.2) off the ring planes on crystals.
* **The term** (`base_stacking.py`, wired into `cg_energy_forces` as an opt-in injection, gradient verified
  against a finite difference to 6 decimal places): E = -eps * SUM_i Gd(d) * Gr(rise) * Gt(theta), each
  factor a Gaussian well at the measured target value with the measured spread as its width.

THE SCAN, two chains, 5000 steps each, production tables, everything else fixed:

| eps (kJ/mol) | d (nm) | rise (nm), 5th pct | theta (deg) | trace J |
| --: | --: | --: | --: | --: |
| target | 0.570 | 0.328, +0.06 | 24.5 | -- |
| 0 | 0.750 | 0.332, -0.33 | 51.4 | 0.139 |
| 4 | 0.727 | 0.279, -0.43 | 50.9 | 0.189 |
| 10 | 0.700 | 0.266, -0.52 | 46.4 | 0.119 |

It moves d and theta in the right direction and by 10-20 percent of what is needed, does nothing useful for
the one-sidedness of the rise, and the trace J shows no systematic damage at this sample size. So the term
"works" and is far too weak -- and the reason is not the one I expected.

THE REASON, measured twice.

1. **The model CAN reorient a base.** Rotating one residue's rigid unit about the local P-P axis, against the
   four springs that hold it to its neighbours, costs (scripts/measure_base_rotation_cost.py, 1Q96):

   | rotation | 5 deg | 10 deg | 20 deg | 30 deg | 45 deg |
   | :-- | --: | --: | --: | --: | --: |
   | cost | 0.0 | 1.3 | 6.2 | 12.4 | 20.4 kJ/mol |
   | in kBT | 0.0 | 0.5 | 2.5 | 5.0 | 8.2 |

   A 20-30 degree reorientation costs 2.5-5 kBT, which is affordable. The roll is a SOFT coordinate in this
   parameterisation, so the obstacle is not the rigid-link network -- K_LINK_NP is 5477 kJ/mol/nm^2, but at
   the equilibrium geometry a rotation about the local axis barely stretches anything, and the cost is
   second order in the angle.
2. **The term's effective strength is not eps.** The three factors MULTIPLY, and each is <= 1, so the
   attraction is eps times a product that is small exactly where repair is needed: at the sampled theta of
   51.4 degrees the orientation factor with a 25 degree scale is exp(-(51.4/25)^2) = 0.016. With the distance
   factor 0.44 and the rise factor ~1, a nominal eps of 10 kJ/mol acts as **0.07 kJ/mol per pair** -- against
   the ~6 kJ/mol that a 20 degree reorientation costs.

So the shape is wrong, not the physics: a reward that vanishes in the configurations that need changing has
almost no gradient there, and the scan confirms it. The fix is a form whose scale IS eps -- an additive or
pairwise-attraction form with a broad orientation factor (oxRNA's stacking is a distance attraction times an
orientation factor, not a product of three narrow wells), after which the same scan is the test. That the
base frame is soft is what makes this worth doing rather than a dead end: the model can carry stacking, the
current term cannot reach it.

## Part 17 — The shape was the whole story: independent penalties create the stacking asymmetry (2026-10-05)

Part 16's fix was wrong, and the way it was caught is worth recording. Rewriting the reward as
`eps * (1 - Gd*Gr*Gt)` changes nothing dynamically, because `SUM(1 - f) = N - SUM(f)`: the two differ by a
CONSTANT, so their gradients -- and therefore the forces -- are identical. The sampler said so before I
noticed: the reward and "penalty" scans produced **bit-identical trajectories** (same bead coordinates to
five decimals, same frame counts, same per-chain J) while both differed from no term at all. Identical
output is a measurement.

What actually limits the product form is the GRADIENT, not the value: with a 25 degree orientation scale,
f_theta at the sampled theta of 51 degrees is 0.016 and its slope is `f * 2*theta/theta_c^2`, about 0.0026
per degree. Measured directly at an unstacked configuration (every second residue rotated 70 degrees about
its local axis, mean neighbour-normal angle 59 degrees), with eps = 10 kJ/mol:

| form | E per pair | max\|F\| |
| :-- | --: | --: |
| reward (eps * Gd Gr Gt) | -0.010 kJ/mol | 7.4 |
| penalty (eps * (1 - Gd Gr Gt)) | 9.990 | **7.4** (identical, as the algebra says) |
| **sum (eps * [(1-Gd) + (1-Gr) + (1-Gt)])** | 20.599 | **111.5** |

A sum of INDEPENDENT penalties has full-strength slope wherever any one coordinate misses its well, which
the product cannot. The orientation scale is widened to 60 degrees for the same reason (a factor that is
0.016 where repair is needed cannot pull anything in); that width is the one modelling choice in the term
and it is recorded as one, while the wells' centres stay at the measured target values.

THE SCAN, sum form, two chains, 5000 steps, production tables, everything else fixed:

| eps (kJ/mol) | d (nm) | rise mean, 5th pct | theta (deg) | TV d / rise / theta | trace J |
| --: | --: | --: | --: | --: | --: |
| target | 0.570 ± 0.198 | 0.328, **+0.06** | 24.5 | -- | -- |
| 0 | 0.750 | 0.332, -0.33 | 51.4 | 0.590 / 0.528 / 0.547 | 0.139 |
| 5 | 0.657 | 0.302, -0.42 | 41.8 | 0.519 / 0.424 / 0.444 | 0.147 |
| 20 | **0.621** | **0.317, +0.02** | **30.1** | 0.572 / **0.255** / **0.290** | 0.215 |

**The stacking asymmetry is created.** The rise's 5th percentile moves from -0.33 nm to **+0.02 nm** against
a target of +0.06 -- i.e. the model goes from having no preference for one base lying over its neighbour to
having the right one-sided preference, and the rise's total-variation distance halves (0.528 -> 0.255). The
base-base distance moves 0.750 -> 0.621 against a 0.570 target and the neighbour-normal angle 51.4 -> 30.1
against 24.5, both in the right direction.

Two things this run does NOT settle, both visible in the same table and both the next knobs:

* **the distance well is over-constraining its WIDTH**: at eps = 20 the sampled d has sd 0.122 against the
  target's 0.198, so the d distance's total variation gets WORSE (0.519 -> 0.572) even as its mean gets
  better. The three penalties are summed with equal weight; the measurement says they should not be.
* **the trace pays for it**: the joint J of the backbone coordinates rises 0.139 -> 0.147 -> 0.215 as the
  base term strengthens. Two chains, so this is not a resolved comparison -- but it is the expected effect:
  the trace tables were fitted with no base-level term present, so a stacking term pulls the trace away from
  its own fitted marginals. Either the term is reweighted, or the trace tables are refit with the term
  present, which is what the loop machinery exists for.

So: the model can carry stacking, the term can be built from measurements, and the shape of the term -- not
its strength -- was what stood in the way for two attempts.

## Part 18 — Ten chains, paired: the term works, and the trace does not pay for it (2026-10-05)

Part 17's two open items were the distance well over-constraining its own width at equal weights, and the
trace J rising with the strength. Both were weight problems. Discounting the distance penalty to 0.3 (the
rise and the orientation are the coordinates the model cannot express at all today; the base-base distance
at least has the pair and link network pulling on it) and raising the strength fixes both at once. Ten
chains of the 24-34 residue band, 5000 steps each, production tables, SAME chains and SAME seeds with and
without the term, so every number below is paired
(`scripts/analyze_stack_sum_paired.py`, weights 0.3/1/1, eps = 45 kJ/mol = 18 kBT per pair):

| coordinate | without the term | with the term | paired change | chains improved |
| :-- | --: | --: | --: | --: |
| base-base distance | 0.7164 +- 0.0359 nm | **0.5906 +- 0.0306** | -0.1258 +- 0.0267 | **10/10** |
| ... its TV vs target | 0.5506 +- 0.0547 | **0.4255 +- 0.0398** | -0.1251 +- 0.0595 | **10/10** |
| rise, mean | 0.3208 +- 0.0665 | 0.3197 +- 0.0254 | -0.0011 +- 0.0704 | TV: 10/10 |
| rise, TV vs target | 0.5418 +- 0.0418 | **0.2717 +- 0.0310** | **-0.2702 +- 0.0449** | **10/10** |
| **rise, 5th percentile** | **-0.3267 +- 0.1566** | **+0.1625 +- 0.2225** | **+0.4891 +- 0.2250** | **9/10** |
| neighbour-normal angle | 49.6469 +- 2.2567 deg | **18.7428 +- 1.0735** | -30.9041 +- 2.4238 | **10/10** |
| ... its TV vs target | 0.5072 +- 0.0336 | **0.2357 +- 0.0281** | -0.2715 +- 0.0289 | **10/10** |
| trace joint J (mean) | 0.1269 | **0.1234** | -0.0035 | -- |

THREE READINGS.

1. **The stacking asymmetry is created, and every chain agrees.** The rise's 5th percentile goes from
   -0.327 +- 0.157 nm to +0.163 +- 0.223, a paired change of +0.489 +- 0.225 that is twice its own spread,
   against the crystal target's +0.06. The model starts with no preference for one base lying over its
   neighbour rather than beside or under it and ends with the right one, slightly overdone.
2. **The trace does not pay for it, once the weights are right.** Joint J: 0.1269 without the term, 0.1234
   with it. That is the opposite of the two-chain equal-weight indication (0.139 -> 0.215), and it says the
   trace cost was the distance penalty's, not the stacking term's -- i.e. the base-base distance at full
   weight was fighting the backbone's own pair and link network, which is exactly the double-counting the
   weighting removes.
3. **The strength is slightly past optimal, and that is visible as overshoot rather than as damage.** theta
   lands at 18.7 +- 1.1 against a 24.5 target (TV 0.236, the best of the three coordinates) and the rise's
   5th percentile at +0.16 against +0.06. A scan of eps over 25-35 is the obvious way to land nearer the
   target; the interesting question -- whether the term survives being fitted by the loop against an
   ensemble that contains it -- now has an instrument: `IBI_LOOP_BASE_STACK=45` plus the form and weights,
   with everything else the loop already does.

Sample and caveats: ten chains from the 24-34 residue band (the band holds 31, and its loader order is not
stable across processes -- Part 15), 5000 steps, one seed per chain, and the target is the crystal-expressible
marginal through the same bead-to-plane map, not the ring-atom one. "Chains improved" is a sign count, not a
p-value; the paired means being several times their standard deviations is what carries the result.

## Part 19 — Landing the strength on the target, and the loop holding the trace with the term on (2026-10-05)

TWO THINGS RUN TOGETHER: a strength scan to land the term on the measured target, and a seven-chain loop arm
with the term ON to ask the question the loop exists for -- with a base-level term pulling the trace, can the
trace tables still match their reference marginals, and do they stop drifting.

THE STRENGTH SCAN, ten chains each, same chains and seeds as the baseline, so every column is paired:

| eps (kJ/mol) | base-base distance | rise, 5th percentile | neighbour-normal angle | trace J |
| --: | --: | --: | --: | --: |
| target | 0.5704 +- 0.1980 | **+0.06** | 24.51 +- 22.22 | -- |
| 0 (off) | 0.7164 +- 0.0359 | -0.3267 +- 0.1566 | 49.6469 +- 2.2567 | 0.1269 |
| 25 | 0.6061 +- 0.0359 | -0.0022 +- 0.2957 (8/10) | **24.7943 +- 2.4062** | 0.1369 |
| **35** | 0.5977 +- 0.0308 | **+0.0842 +- 0.2517 (9/10)** | 21.8752 +- 2.7995 | **0.1215** |
| 45 | 0.5906 +- 0.0306 | +0.1625 +- 0.2225 (9/10) | 18.7428 +- 1.0735 | 0.1234 |

The term's parameters now land ON the target rather than near it: at 25 kJ/mol the orientation angle matches
to 0.28 degrees (24.79 against 24.51) but the rise's one-sidedness is only just achieved (-0.002), at 35 the
rise lands nearest the target (+0.084 against +0.06) with a mild orientation overshoot, and at 45 both
overshoot. **35 kJ/mol per pair is the setting to carry**: it is the closest on the coordinate the model could
not express at all, and it is also the only strength at which the trace joint J ends BELOW the baseline
(0.1215 against 0.1269).

THE LOOP ARM, seven chains, three rounds, production protocol, `IBI_LOOP_BASE_STACK=45` (run before the scan
above chose 35), moment operator on all three trace coordinates, support gate 0.03:

| round | bb_bond sampled / target (\|ln\|) | angle | dihedral | per-chain J median |
| --: | :-- | :-- | :-- | --: |
| 0 | 0.06553 / 0.05402 (0.193) | 0.41631 / 0.32176 (0.258) | 0.73043 / 0.62471 (0.156) | 0.1552 |
| 1 | 0.05359 / 0.05402 (**0.008**) | 0.29619 / 0.32176 (0.083) | 0.50814 / 0.62471 (0.207 FAIL) | 0.1757 |
| 2 | 0.05571 / 0.05402 (**0.031**) | 0.30452 / 0.32176 (**0.055**) | 0.65178 / 0.62471 (**0.042**) | **0.1477** |

**Every trace coordinate is inside tolerance by round 2, with the base-level term on**, and the dihedral's
update falls from 4.19 and 5.30 kJ/mol in rounds 0-1 to **0.99 kJ/mol in round 2** with the moment norm at
0.0502 -- the table is settling rather than ringing. The per-chain J median comes back to 0.1477, the level the
nine-round campaign had at its own second round. So the loop's question is answered for the trace: a
base-level term does not prevent the trace tables from reaching their reference marginals, and the machinery
already in `ibi_loop.py` (the `marginals` block, the support gate, the divergence guard) is what shows it.

WHAT THE LOOP CANNOT SEE YET, and it is the next wiring step: the loop bins the SIX coordinates in
`boltzmann_bonded.COORDS` -- bb_bond, intra_pc, intra_cn, angle, dihedral, stack -- and none of them is a
base-level quantity. So this arm can report that the trace is fine with the term on, but it cannot report
whether the base-level marginals are, or fit them: `base_dist`, `base_rise` and `base_theta` have to become
coordinates of the sampler's own binning loop (their bead-only definitions and their measured targets are in
Parts 15-18) before the loop can close on them the way it closes on the trace. That is a table, a target
measurement, and three lines in the binning loop -- the same shape as every other coordinate in this field.

## Part 20 — The loop scores the base level now, and immediately falsifies the strength I had chosen (2026-10-05)

THE WIRING. Three things, all opt-in, and the shipped path is bit-identical without them:

* **The coordinates** (`boltzmann_bonded.coords_of`): `base_dist` = |N(i+1) - N(i)|, `base_rise` = that
  vector's projection on the mean base-plane normal, `base_cos` = the folded cosine between the two
  normals, all through the same rigid map the term uses. They are kept OUT of `COORDS` on purpose: `COORDS`
  is what every table file must contain, and growing that tuple makes `results/refit_smooth5.npz`
  unloadable -- measured, `load_tables` raises "missing 18 key(s)" the moment it does. Instead
  `scored_coords()` returns `COORDS` plus `BASE_COORDS` when `IBI_SCORE_BASE=1`, and `ibi_core` asks for
  the scored list rather than the constant in the eighteen places that used to read it directly.
* **The targets** (`scripts/build_base_level_ref.py`): the crystal marginals for the three coordinates,
  measured by calling the sampler's OWN `coords_of` on crystal beads -- there is no second implementation
  to drift -- on fixed wide grids (the support has to cover what the SAMPLER does, not only what the
  crystals do, or the out-of-support gate trips and the round reports a truncated view). 40 fragments,
  2991 consecutive pairs: base_dist 0.5626 +- 0.1836 nm, base_rise 0.3286 +- 0.1770 nm, base_cos 0.8534 +-
  0.2394. Output is a MERGED reference (every shipped entry copied verbatim plus the three), so an arm
  points `IBI_LOOP_REF` at it and gets all nine.
* **The scoreboard** (`ibi_loop`): the three are CARRIED -- measured and scored, never injected, because
  their dynamics come from `base_stacking`'s term and not from a 1-D table -- and every structure now
  carries a third joint residual, `joint_J_base`, beside `joint_J` (which stays the mean over the
  CONTROLLED trace coordinates, so earlier runs remain comparable) and `joint_J_all`.

THE ARM, seven chains, two rounds, production protocol, term on at the strength Part 19 chose (35 kJ/mol
per pair, weights 0.3/1/1):

| round | joint_J (trace) | **joint_J_base** | base_dist sim/ref | base_rise sim/ref | base_cos sim/ref |
| --: | --: | --: | --: | --: | --: |
| 0 | 0.2280 | **1.0937** | 0.664 | **0.249** | **0.266** |
| 1 | **0.0784** | **0.9178** | 0.805 | **0.246** | **0.254** |

**The term is 3-4 times too strong, and the loop's own measure is what says so.** sim/ref here is
sigma_sim / sigma_ref, so base_rise and base_cos are pinned to a QUARTER of their crystal widths. The
5000-step scan that chose 35 could not see this: over a short window the same term looked close to target,
because the pooled statistics of a 2.5 ps window are not the equilibrium width. The loop samples 65 ps and
scores sigma, and it is right.

AND THE RIGHT STRENGTH IS DERIVABLE, not another scan: a penalty of curvature 2*eps/sigma^2 pins the
coordinate to a width that scales as sqrt(kBT*sigma^2/(2*eps)), so sigma_sim = sqrt(eps_scan/eps)*sigma_scan
and matching the target needs **eps = eps_scan * (sim/ref)^2**: for base_rise 35 * 0.246^2 = **2.1**, for
base_cos 2.3, and for base_dist 22.7. So the rise and the orientation want ~2 kJ/mol per pair while the
base-base distance wants ~23 -- a factor of eleven between them, which is the same "the three penalties
should not carry equal weight" finding the scan saw on the other side, now with a number. The term's
weights and strength are one calibration, and the loop is finally the instrument that can do it: a
two-round arm at 2 kJ/mol with the distance weight raised is the next run, and it is 25 minutes.

THE SECOND STRENGTH, run at 2.3 kJ/mol with the weights unchanged (0.3/1/1):

| round | joint_J (trace) | **joint_J_base** | base_dist sim/ref | base_rise sim/ref | base_cos sim/ref |
| --: | --: | --: | --: | --: | --: |
| 0 | 0.2182 | **0.3013** | 0.877 | **1.77** | 1.14 |
| 1 | 0.1848 | **0.3158** | 0.925 | **1.92** | 1.19 |
| (for scale) 35 kJ/mol, round 1 | 0.0784 | 0.9178 | 0.805 | 0.246 | 0.254 |

**joint_J_base falls from 0.92 to 0.30**, and the error has changed sign and shape: the rise is now 1.8-1.9
times TOO WIDE, the orientation 1.15-1.19 times too wide, and the base-base distance is still slightly
narrow (0.88-0.93). One strength cannot fix three coordinates that want different ones, which is the
per-coordinate weighting the scans kept asking for -- and now it can be DERIVED rather than searched:
applying sigma ~ 1/sqrt(eps) per coordinate, the rise wants 2.3 * 1.85^2 = 7.9, the orientation
2.3 * 1.17^2 = 3.1, and the distance 0.69 * 0.9^2 = 0.56. With eps = 7.9 that is **w = (0.07, 1, 0.39)**,
which is the third arm, running.

Worth stating plainly, because it is the point of this whole step: **the loop's sigma-based scoreboard is
now the calibration instrument.** The first strength was chosen from a 5000-step scan and was wrong by a
factor of fifteen; the second was derived from the loop's own ratios and cut the residual by three; the
third is derived the same way, per coordinate. Every one of those numbers came from a two-round, seven-chain
arm that takes twenty-five minutes.

## Part 21 — The calibration converges, and one of its own derivations was wrong for a good reason (2026-10-05)

Three two-round, seven-chain arms, everything else identical, only the term's strength and weights moving:

| arm | (eps, weights) | joint_J (trace) | **joint_J_base** | base_dist | base_rise | base_cos |
| :-- | :-- | --: | --: | --: | --: | --: |
| 1 | (35, 0.3/1/1) | 0.078 | **0.918** | 0.805 | 0.246 | 0.254 |
| 2 | (2.3, 0.3/1/1) | 0.185 | **0.316** | 0.925 | 1.92 | 1.19 |
| 3 | (7.9, 0.07/1/0.39) | 0.193 | **0.254** | 0.834 | 1.50 | **1.07** |
| target | -- | -- | **0** | 1.0 | 1.0 | 1.0 |

(round 1 of each arm; the round-0 numbers are within a few percent of these.)

THREE READINGS.

1. **The orientation is at its target**: base_cos's sigma is 1.07 of the crystal's, from 0.254 at the first
   strength -- a factor of four in the term's strength and a factor of eleven in its weight, arrived at by
   two derivations and not by a search.
2. **The rise is still 1.4-1.5 times too wide**, and the lever is known and linear in the effective
   strength: 7.9 * 1.45^2 = 16.6 kJ/mol, i.e. w_r = 2 with eps kept at 7.9, or eps = 16.6 with w_d and w_t
   rescaled to hold their effective strengths where they are. That is one more 25-minute arm.
3. **The base-base distance does not follow the law, and that is a finding rather than a failure.** Its
   sigma ratio moved 0.805 -> 0.925 -> 0.834 while the term's effective strength on it moved 10.5 -> 0.69 ->
   0.55, i.e. by a factor of nineteen with no monotone response: the distance's width is set by the
   backbone's own pair and link network, not by the base-level term. The derivation in Part 20 that put the
   distance at "22.7 kJ/mol" assumed the 1/sqrt(eps) relation applies to it; it does not, and the arm that
   was meant to confirm that number is what showed it.

ONE OBSERVATION ABOUT THE TRACE, stated because it would otherwise read as a win: the trace's own J is
BEST in the arm with the strongest base term (0.078 at eps = 35 against 0.19 at 7.9). That is not better
physics -- a term strong enough to pin the base frames to a quarter of their width removes degrees of
freedom the trace tables were fitted WITHOUT, so the trace is being scored against a more constrained, and
therefore easier, ensemble. The honest reading is that the trace J is comparable across these arms only
because the term is weak enough not to dominate, and that a base-level term at a physically sensible
strength leaves the trace where Part 19 found it.

WHAT IS LEFT, in one line each: the rise's last factor of 1.5 (one arm); the distance, which is a backbone
property and should probably be dropped from the term or given a cosmetic weight rather than tuned (its
0.07 already makes it almost inactive, and the result is 0.83); and then the question this all served --
whether the field that carries stacking also reproduces the standard this project is judged on, retention.

## Part 22 — The fourth calibration, and the retention answer (2026-10-05)

THE FOURTH ARM: eps = 16.6 kJ/mol, weights (0.0, 1, 0.19) -- the rise's derived lever applied, the
orientation's effective strength held where it was, and the distance term DROPPED rather than tuned (a
parameter the measurement says does not control its own coordinate should not carry a value that looks
tuned).

| arm | (eps, weights) | joint_J (trace) | **joint_J_base** | base_dist | base_rise | base_cos |
| :-- | :-- | --: | --: | --: | --: | --: |
| 1 | (35, 0.3/1/1) | 0.078 | 0.918 | 0.805 | 0.246 | 0.254 |
| 2 | (2.3, 0.3/1/1) | 0.185 | 0.316 | 0.925 | 1.92 | 1.19 |
| 3 | (7.9, 0.07/1/0.39) | 0.193 | 0.254 | 0.834 | 1.50 | 1.07 |
| **4** | **(16.6, 0/1/0.19)** | 0.118 | **0.190** | 0.738 | **1.21** | **0.90** |
| target | -- | -- | 0 | 1.0 | 1.0 | 1.0 |

joint_J_base falls 0.918 -> 0.316 -> 0.254 -> **0.190**: a factor of five from the first strength, with two
of the three coordinates within 20 percent of the crystal width (rise 1.21 wide, orientation 0.90) and the
third confirmed once more not to be under the term's control -- **dropping the distance penalty from 0.07 to
0.0 moved its sigma ratio from 0.834 to 0.738**, the wrong way for a term being removed, which is what "this
coordinate's width is the backbone's" looks like when it is tested rather than argued.

AND THEN THE QUESTION THIS ARC WAS FOR. Does a field that carries stacking still keep a chain near the
geometry it was deposited in? The retention instrument (`measure_native_retention.py`, seven
length-stratified chains to 400 residues, 12 ps, 5000-step relaxation) run twice on the SAME tables, the
only difference being the base-level term:

| arm | median dep->mean | median spread | chains > 10 A | per-chain range |
| :-- | --: | --: | --: | --: |
| term OFF | **0.46 A** | 0.19 A | **0 of 7** | 0.22 - 1.47 A |
| term ON (7.9, 0.07/1/0.39) | **0.43 A** | 0.18 A | **0 of 7** | 0.22 - 1.45 A |

**Retention is unchanged.** The stacking term costs nothing on the standard this project is judged by, at
the strength the calibration landed on, and the two arms agree chain by chain to within 0.03 A on six of
seven. That is the answer the whole base-level arc was pointing at: the coordinates the crystals actually
constrain can be brought to their targets without giving up the property the field was already good at.

Caveats, because they bound the claim: one strength (the calibrated one, not the 16.6 of the fourth arm),
seven chains from a length-stratified sample rather than the full pool, 12 ps, and the tables are the merged
reference rather than the campaign's own `tables_r9` -- so the numbers compare the two arms to EACH OTHER,
which is what the question needed, and not to the historical baseline directly.

## Part 23 — The product step, and a negative result that bounds Parts 13-14 (2026-10-05)

The step that was supposed to convert all of the above into a better PRODUCT: build the all-atom structure from
the SAMPLED bead frame instead of from the P trace, and compare the two products on their own base planes.
Two infrastructure pieces were needed first, and one of them was broken:

* **`write_allatom_pdb` did not exist.** `torch_gpu_refine`'s bead-frame branch (`cg_frame_allatom`)
  imported it from `isrnacirc_wrapper`, where it has never been, so the branch raised ImportError, was
  caught, and fell back to the P-trace path WITHOUT SAYING SO -- the feature was dead on arrival and only a
  warning that nothing printed would have told anyone. Written now, beside the reconstruction that produces
  an AllAtomStructure, in the same PDB columns the CG writer uses.
* **The shipped CG-to-all-atom step is an external binary, `CG_to_allatom.exe`, and it is NOT PRESENT on
  this machine** (`_CG_TO_AA_EXE` is empty). `cg_to_allatom` therefore raises, the exception is caught,
  and the "all-atom" product of the shipped path is the P trace itself. So the in-tree reconstruction is not
  an alternative to the shipped path here -- it is the only all-atom path there is.

THE COMPARISON, one 2OIU chain, 5000 steps under the production tables WITH the term at the Part 22 setting
(eps 16.6, w 0/1/0.19), then both structures built from the SAME final state:

| | base_dist | base_rise (mean, 5th pct) | base_cos |
| :-- | --: | --: | --: |
| crystal | 0.5306 nm | +0.3186 (+0.2110) nm | 0.9008 |
| **sampled, term ON** | **0.5381** | **+0.3034 (+0.2074)** | **0.8599** |

**The field's own beads sit on the crystal's base-level geometry almost exactly** -- which is the cleanest
evidence yet that the term does what it was built to do, measured directly on the beads rather than through
a reconstruction.

| product (same state) | stacked | rise | normal angle | WC contacts kept |
| :-- | --: | --: | --: | --: |
| A: P trace (shipped in-tree path) | **0.0%** | 1.48 +- 0.66 A | 29.0 deg | 3/12 |
| B: sampled bead frame (new) | **0.0%** | **0.73 +- 0.34 A** | 28.1 deg | 5/12 |
| crystal | 100% | 3.40 +- 0.13 A | 7.8 deg | 12/12 |

**Neither product is stacked, and the bead frame is WORSE than the P trace here** -- the opposite of what
the ideal-input measurement in Part 14 found (43.9 -> 75.7 percent). The reason is in the same table: this
run's trace ended **21.5 A** from the deposit, so its local backbone geometry is far from A-form.

WHY THAT BREAKS THE BEAD-FRAME PATH, and it is a real bound on Parts 13-14. `reconstruct_all_atom_from_beads`
fits a RIGID 1EHZ residue onto the three sampled beads. When those three beads form an A-form-like triangle,
that fit reproduces the base plane; when the triangle is distorted, the best-fit rotation is set by the
SHAPE MISMATCH between the template's triangle and the sampled one, and the base plane can end up far from
where the beads say it should be. The field can therefore know the right base geometry (the beads above say
it does) while the reconstruction cannot express it. The in-tree reconstruction is only as good as the local
trace geometry it is handed -- which is exactly what Part 13 measured when it varied the trace RMSD and found
stacking gone by 2.7 A.

WHAT THIS MEANS FOR THE NEXT MOVE, and it is a change of plan rather than a failure: threading the sampled
beads into the product is worth doing on a trace that is already good (which is the refinement use case, and
where the ideal-input gain is 32 points of stacking), and it is NOT the fix for a field whose trace has
drifted. The fix for that is a model whose base frame is the model's OWN -- the beads themselves defining the
plane, or a fifth bead that carries a normal -- which is the architectural question from the stacking
discussion, and it is now the measured bottleneck rather than a suspicion.


## Part 11 — The delivered tables through the shipped refiner: 2OIU, fitted against analytic (2026-10-04)

The pipeline switch in `torch_gpu_refine` (TORUSFOLD_CG_TABLES, 2026-10-01) made this the first
end-to-end test of the field as delivered. Both arms run the same shipped refiner on this machine's GPU
from the same crystal P trace, the same secondary structure and the same pair list; the only difference
is which field the CG energy calls take. 2OIU, 71 nt, 27 pairs, input BSJ 0.592 nm. The production
table mixes sources by measurement: bb_bond and dihedral from the campaign's round 9, angle from
refit_smooth5, all three on one shared support (`scripts/build_production_tables.py`).

| arm | wall | final energy (its own field) | BSJ |
| :-- | --: | --: | --: |
| fitted tables (`results/production_tables.npz`) | 629.5 s | 708.417 | 0.687 nm |
| analytic field | 622.0 s | 730.05 | 0.649 nm |

The energies are not comparable across arms; the geometry is. P-trace widths (all four coordinates are
functions of the P trace alone) against the fit target:

| coordinate | target sd | analytic | tables | \|ln(sd/target)\| an / tab |
| :-- | --: | --: | --: | --: |
| bb_bond | 0.0633 | 0.0042 | 0.0045 | 2.720 / 2.648 |
| angle | 0.3203 | 0.4155 | 0.4188 | 0.260 / 0.268 |
| dihedral | 0.6299 | 0.3337 | 0.4634 | **0.635 / 0.307** |
| stack | 0.1390 | 0.1729 | 0.1730 | 0.219 / 0.219 |

1. **The one coordinate the tables were fitted to change is the one that moved.** The dihedral's width
   over the analytic field goes 0.334 -> 0.463, halving its log error; no other coordinate moves by more
   than 0.004 in sd (and stack, which is algebraic in bb_bond and angle and has no table of its own,
   moves the least of all -- an internal consistency check that the raw P-trace statistics are being
   read the same way in both arms). The fitted field does what it was fitted to do through the shipped
   GPU path, and it leaves the dihedral 26 percent narrow against a target the free refinement never
   reaches.
2. **Two of the four rows are not field verdicts.** The bb_bond row is dominated by the refiner's own
   bond restraints at this stage, so both arms sit at 0.004 against a 0.063 target; the ln ratios there
   say nothing about the field. The angle row is the analytic field's own error, and the angle table it
   used is the refit target's, so a table-side gain was not expected and none appears.
3. **One draw per arm, and it is reported as one.** REMD velocities are unseeded, and the two products
   differ by 13.5 A P-only Kabsch RMSD. That number bounds nothing by itself -- it says the refinement
   is not deterministic, not that the fields disagree -- and it is why this is recorded as a single
   observation of a real effect (the dihedral is far outside any draw-to-draw scatter this pipeline has
   shown) rather than as an effect size.

CORRECTION (2026-10-05, ten draws per arm). The assumption in point 3 -- "the dihedral is far outside any
draw-to-draw scatter this pipeline has shown" -- is false, and ten draws say so. The same two arms, same
protocol, run ten times each (`--draws=10`), analysed as a PAIRED difference because the arms share the
input geometry and the draw index:

| coordinate | target sd | analytic mean +- sd | tables mean +- sd | paired ln(tab/an) | draws closer |
| :-- | --: | --: | --: | --: | --: |
| bb_bond | 0.0633 | 0.00428 +- 0.00071 | 0.00456 +- 0.00060 | +0.0683 +- 0.2245 | 6/10 |
| angle | 0.3203 | 0.43247 +- 0.04528 | 0.42623 +- 0.04748 | -0.0151 +- 0.1415 | 4/10 |
| dihedral | 0.6299 | 0.38285 +- 0.03188 | 0.38561 +- 0.03967 | **+0.0057 +- 0.1318** | **6/10** |
| stack | 0.1390 | 0.18085 +- 0.02457 | 0.17436 +- 0.02371 | -0.0365 +- 0.1706 | 5/10 |

The dihedral effect that the single draw measured (0.3337 -> 0.4634, |ln(sd/target)| 0.635 -> 0.307) is
**+0.0057 +- 0.1318, six draws out of ten**: indistinguishable from zero, with the draw-to-draw scatter
(sd 0.032-0.040 nm on a 0.38 nm width, i.e. 8-10 percent) an order of magnitude larger than the paired
mean. The single draw was one sample, and the honest reading of Part 11's point 1 is that the fitted field
has NOT been shown to change this stage's geometry at all.

What the ten draws ALSO show, and it is a statement about the stage rather than the field: these products
sit far from the marginals the tables were fitted to -- angle 0.426 against 0.320, dihedral 0.386 against
0.630 (40 percent narrow), stack 0.174 against 0.139, bb_bond 0.0046 against 0.063 (the refiner's own bond
restraints, as Part 11 said). A short, free, restraint-dominated refinement is not an equilibrium sampler
for those marginals; Part 12's pooled pass is where the tables are validated, and this stage is not a
second measurement of it. That distinction was missing from Part 11 and is the reason its one-draw claim
looked like a field result.

Chebyshev was a seven-chain number), and there is no arm-level evidence for K=16 or K=32.




