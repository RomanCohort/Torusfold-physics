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

`docs/plan_b_coupled_update.md`'s operator ran head to head against the marginal inversion on the
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

The inversion stalls and rings at 0.082-0.086; the moment operator falls monotonically to 0.035 and
angle walks from 1.259 to 1.135. Dihedral reaches about 1.0 under both. bb_bond plateaus at
0.853-0.856 under BOTH, which 5.1 explains: the two operators agree because both are already at the
table's own distribution.

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
