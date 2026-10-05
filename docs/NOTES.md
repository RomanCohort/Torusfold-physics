# Development notes: what didn't work, and known limitations

> Honest engineering log for the next team. These are *measured* dead ends and
> open problems, not solved features. Everything here reflects actual runs.

## TriRNASP as a force field term was dropped

We tried adding the TriRNASP three-body statistical potential
(Tan-group/TriRNASP) as an extra energy term in the coarse-grained (CG) force
field. It was **disabled** (`use_trirnasp=False` in the isRNAcircLong runner)
for two reasons:

1. **Scale mismatch / conflict.** TriRNASP is a *log-odds statistical potential*,
   not a physical energy. At `trirnasp_scale = 0.002` it still contributed
   ~11 % of the total energy — i.e. its raw magnitude was ~50× the CG terms.
   The two potentials disagreed in their minima (idealized CG geometry vs.
   PDB statistics), producing a frustrated landscape that made the composite
   hard to anneal and hypersensitive to the coupling scale.
2. **Resolution mismatch.** TriRNASP triplets are defined near all-atom
   resolution; our CG beads are P / C4′ / N. Mapping one onto the other is
   fragile and, if wrong by one atom order, yields spurious forces.

Lesson: **keep physics potentials in the force field and statistical potentials
as post-hoc scorers** (rsRNASP is already used only as a score, not a force).

Note: `trirnasp_scorer.py` had a broken duplicated method — `score_from_pdb`
contained dead code copied from `full_scoring` that referenced out-of-scope
`n_atoms` / `atom_coords` (would `NameError`). Fixed by delegating it to the
correct `score_pdb()` implementation (it was never called in-tree).

### Direction we converged on (for a future pass)

- Apply statistical corrections **sparsely in space** — only on loop / BSJ /
  non-canonical (NCM) regions, and zero in A-form stems where the CG + A-form
  template already encodes the answer.
- The high-value novelty is **running many REMD/REST2 replicas as a batched
  tensor on GPU** (`torch_cgsim.py`, `BatchedREMD2D`), not the force-field
  complexity. Sparse terms are what make the replica batch fit in GPU memory.

### Open validation items (please do these before claiming convergence)

- **2OIU recovery**: does the sparse CG field + batched-replica sampler return
  a structure within RMSD of the crystal? If `E_stat` is not minimized near the
  native structure, the potential mapping/reference state is broken.
- **Replica-exchange correctness under tensor batching**: acceptance rates
  (~10–30 %), replica round-trips, and no bias introduced by the batched
  exchange.

## Far / long-range pair closure is still imperfect

Distal (topologically far) pairing distances come out too large in early runs;
RL MCTS + steering forces were added to pull them toward Watson–Crick
geometry (C1′–C1′ ~10.5 Å) but this remains the main accuracy bottleneck for
long sequences. Not fully solved.

## CG planar collapse (fixed)

Initialization collapsed to a flat disc (`z ≈ 0.6 Å`) because CG started with
`z = 0` and the energy surface flattened it. Fixed by injecting helical `z`
into the CG initialization and a `CustomExternalForce` `z`-restraint during
refinement. Keep the restraint on; do not "clean up" the `z` patch.

## Amber refinement: dihedral terms used wrong atoms (fixed)

In the all-atom amber path, the `alpha`/`zeta` dihedrals were computed with
cross-residue atoms of the *same* residue instead of the *neighboring*
residue, causing 18/20 outlier dihedrals. Fixed; verify per-residue dihedral
deviations stay within ~5° of ideal.

## Modules referenced but not published yet

`predict_3d_allatom` (legacy aggregator in `scheme2/__init__.py`) references
`cg_forcefield` and `modification_aware`, which are not part of this snapshot
(under active development). The main `isrnaclong_pipeline()` path does not
depend on them.

## External ML predictors (not vendored)

RhoFold+, trRosettaRNA2 and RNAbpFlow are third-party model checkpoints we
invoke as subprocesses; none of their weights are in this repository. See
`docs/DEPLOY_EXTERNAL.md`.

## Experimental validation

No wet-lab or experimentally-determined ground-truth validation has been
performed yet for the 2013-nt construct. Validation is limited to internal
consistency + a handful of published-sequence tests. Treat predicted
structures as hypotheses until the 2OIU recovery and replica-exchange checks
above are completed.

## 2026-09-09 — GPU platform notes (internal)

Measured on an AMD Ryzen AI MAX 395 (Radeon 8060S iGPU): ROCm compatibility
was poor and required extensive patching (a large time cost); iGPU throughput
measured far below an RTX 3080. The CPU path is therefore the reference for
all measured runs. GPU full-configuration wall time (~14 d estimated on this
APU; ≈2 d on an NVIDIA A100 at the ≈8× estimate) is not yet measured.

## 2026-09-09 — 2OIU force-field integrity test (update)

Completed: starting from the 2OIU X-ray structure, the Level-2 relaxation
(CPU) took 17 min and ended at RMSD 1.83 Å vs the crystal — the force field
does not distort known structures. Note this is a force-field integrity test
from the X-ray structure, not a from-sequence "recovery" benchmark; the
from-sequence recovery question above remains open.

## 2026-09-09 — verification backlog & status update

- **Replica-exchange acceptance (measured):** 30–50% on recent runs (the
  "~10–30 %" figure above is outdated). Dated update — retest after any
  exchange-criterion change.
- **Explicit-force vs autograd cross-check (TODO):** the autograd
  implementation (`cg_forces_autograd`) is the free ground truth; add a
  short-trajectory force comparison (max relative error) before claiming the
  no-autograd forces are verified. "Runs fine" is not a gradient check.
- **Async CPU force injection latency (scoped):** only active with
  `use_trirnasp_force=True` (preview, off by default) — no stale-force risk
  on the default path. No timing claim is made in public docs.
- **RL scheduling evidence (roadmap):** the RL controller is deliberately
  conservative (design choice, see README implementation notes); its benefit
  over rule-based scheduling / fixed budgets is not yet quantified — plan a
  short-sequence energy-vs-time comparison.


## 2026-10-01 — All-atom path: three wiring defects, and what the CG level owes stacking

Found while taking **2OIU** (the only resolved circular RNA this project has: 71 nt, BSJ P0-P71 =
5.9 A) through "crystal P trace -> reconstruction -> amber14-OL3". Three separate things stop that
chain. All three are cheap to fix, and all three are silent until they are hit:

1. **`aform_from_template` gives EVERY residue a third phosphate oxygen (OP3).** A phosphodiester
   phosphate has two non-bridging oxygens (OP1/OP2) plus the two bridging ones (O5' of its own
   residue, O3' of the previous one), so the exported structure carries one oxygen too many per
   residue. The force field reports it ONE RESIDUE AT A TIME — "No template found for residue N (G).
   The set of atoms is similar to G, but has 1 O atom too many" — which reads like a terminal problem
   and is not: on 2OIU it is 23 extra atoms over 71 residues, and filtering OP3 fixes the whole chain
   (1550 -> 1527 atoms). **As it stands, `aform_from_template` output cannot be fed to
   `amber_refine` without that filter.** The fix belongs upstream, in the reconstruction.
2. **amber14's `G5`/`A3` are the DEPHOSPHORYLATED termini.** A phosphorylated 5' end is an internal
   residue as far as the templates are concerned — OpenMM's own matcher says so ("the set of atoms is
   similar to G") once OP3 is gone. Renaming the ends `G5`/`A3` therefore makes matching fail, and so
   does writing them into a file and reading it back, because `PDBFile` normalises "G5" to "G". The
   combination that works is plain internal names plus
   `createSystem(..., ignoreExternalBonds=True)`, which is what the project's own circular builder
   already does.
3. **`Simulation()` does not carry coordinates.** Without
   `sim.context.setPositions(modeller.positions)` the first `minimizeEnergy()` answers "Particle
   positions have not been set".

With those three, 2OIU goes reconstruction -> the repository's circular builder (2297 atoms, 770 H) ->
`createSystem(amber14-all.xml + implicit/obc1.xml)` (2297 particles, 6 forces) with no further
complaint. **What is still missing is a RELAXED starting structure**: straight from the crystal trace
the reconstruction enters OpenMM at **4.1e25 kJ/mol** of atomic overlap and NaNs during annealing,
while the shipped benchmark CG-refines first and records e0 ~1e14 (`scripts/benchmark_2oiu.py`).
That component (`openmm_gpu_refine`) produced no output in our run and is the one open step between
this wiring and an all-atom marginal on the product's topology.

**What the CG level owes, measured separately** (`docs/archive/cg_allatom_interface.md`): the field's four
scored coordinates are local P-trace quantities, and `stack` is an exact function of `bb_bond` and
the P-P-P angle. On the 867 fragments the local marginals are length-independent to 4-13 percent over
a 7x length range, the end effect is 4-5 percent of the sd below L=60 (under 0.2 percent past L=400),
and **forcing the end-to-end distance to zero — the closure a circle imposes — moves them by under 0.5
percent**. Fitting on linear fragments and applying the field to circular RNA is therefore sound, and
the only topology-dependent part is the two terminal residues a circle does not have. The same
measurement shows the all-atom level carries the base-plane physics but INHERITS the trace: a 1.5 A
error in the CG P trace halves the reconstructed stacking fraction (0.59 -> 0.29 on 1ET4). Lesson:
**the CG owes a correct P trace; stacking is the all-atom level's job, and `stack` should not be a CG
scoring target** — which is also why `torch_cgsim` sets its spring to zero.


## 2026-10-01 — Hand-building the coarse-grained field: the whole arc in one place

*Written because the process is spread over a timeline, four topic documents and a hundred code
comments, and the next person needs the story in one read. The topic docs remain the detailed record:
`docs/timeline.md` (the 134-commit arc), `docs/statistical_potentials_as_forces.md` (the reasoning),
`docs/dihedral_table_decision.md`, `docs/force_field_comparison.md` (against IsRNA2/IsRNAcirc),
`docs/ibi_loop_and_oxrna_findings.md` (the iteration campaign), `docs/archive/plan_b_coupled_update.md` and
`docs/archive/plan_c_basis_family.md` (the two open tracks), `docs/archive/cg_allatom_interface.md`.*

### 1. It began as springs pinned to target values

The field was a list of "equilibrium value plus harmonic stiffness" per coordinate, and the stiffnesses
had no criterion worth the name. Phase 5 of the timeline replaced every one of them:

| constant | before | after | what the data said |
| :-- | --: | --: | :-- |
| `K_ANGLE` | 600 | 28.1 | 21x too stiff |
| `K_DIH` | 500 | 7.2 | 69x too stiff |
| `K_BPP` | 600 | 13.4 | 45x too stiff |
| `K_BB` | 500 | 1122.4 | too SOFT, opposite direction |
| `K_STACK` | 500 | **0** | exactly redundant with bond + angle (verified to 8.5e-16) |
| `force_cap` | 200 | 5000 | the cap was clipping physics, not noise |

Three of those had no criterion at all, and four short-range backbone pairs had no term of any kind.
The lesson is not "stiffer is wrong": one was too soft and one did not belong in the field at all.

### 2. Then the functional form turned out to be wrong, not just the numbers

The bonded coordinates are skewed. The P-P-P-P pseudo-torsion has its mode at -22.5 degrees with only
5.1 percent of observations within 30 degrees of 180, so a harmonic spring to any single value pins the
structure where most native configurations are not. The field moved to **Direct Boltzmann Inversion**,
`U(q) = -kBT ln P_ref(q)` over a table, with three consequences that are all in the code: forces come
from **autograd** on `U(q(x))` rather than a hand-derived approximate gradient (the explicit path had
an admitted approximation and its gradcheck failed at HEAD); outside the fitted support a C1-continuous
quadratic **wall** keeps the coordinate from drifting off the sampled range; and the tables are
**sequence averaged** (base identity only picks the N9-vs-N1 bead and the Watson-Crick pairs), with a
stratified refit available to test whether that averaging costs anything.

### 3. What had to be repaired before that could be believed

Boltzmann inversion inherits every error upstream of it, and the arc found several: the all-atom
reconstruction was Kabsch-superposing onto **degenerate anchors** (and the anchor file had never been
committed); the 1EHZ "truth set" was **missing HETATM modified residues**, so every comparison against
it was against an incomplete structure; the residue loader **joined across chain gaps**; `F = -dE/dx`
itself was broken in one path and a dihedral mechanism was described wrongly and retracted; and the
three-bead model is **nominal** -- the C4' and base beads are grown from P rather than taken from
geometry, which is why cgRNASP-level statistics cannot simply be injected into our coordinates, and why
the piecewise-constant tabulated potentials of that family are blocked on differentiability and not on
data. Two more routes were tried and dropped with reasons recorded: TriRNASP as an energy term (scale
mismatch, resolution mismatch) and the hand-built Fourier dihedral (cannot enter a table-based loop).

### 4. The refit series, and where the tables came from

`refit_191` -> `refit_439` -> `refit_full` -> `refit_full_1000` -> `refit_smooth5`, 191 deposited
PDB entries and 126 gap-free chains at the start, the full 867-chain pool at the end. Every step is a
support/robustness decision: a robust sigma from (q99.9 - q0.1)/6.58 rather than the plain standard
deviation, a support cut at 4 robust sigmas, pseudo-counts and a smoothing width whose convention
differs between the two code paths (bins vs +-bins, and getting it wrong smooths twice as hard).

### 5. The iteration era: 9 rounds, 867 chains, 200,109 s

The IBI loop (`scripts/ibi_loop.py`) closed the loop: sample under `tables_r<N>`, invert, repeat. What
it produced, with the numbers that matter:

* **Two update operators.** The marginal inversion (`plan_update`) and the relative-entropy step on a
  low-order correction (`moment_correction`). The marginal one **rings** on the angle at full pool
  (3.00 / 3.94 / 3.02 / 4.27 / 4.31 / 3.72 kJ/mol over rounds 0-5, no decay) while settling bb_bond and
  the dihedral; the moment operator decays the angle's correction but drives the table's own implied
  sigma from 0.3218 down to 0.1307 over nine rounds while the SAMPLED sigma falls only 16 percent --
  an update the coupling absorbs rather than a converging one.
* **The metric had to be rebuilt twice.** The joint residual J averaged over four coordinates carries a
  **0.0659 floor by construction** (the stored sigmas are outlier-inflated for bb_bond 0.853 and stack
  0.913 of their tables' own implied ones), and `stack` is an exact function of the other two, so it is
  now reported (as a consistency check) and excluded from the scored J. Against the tables' own sigma,
  bb_bond and the dihedral were **at target by round 3** (1.004, 1.015) while the pooled J still looked
  bad -- a reminder that a residual is a number about an instrument before it is a number about a field.
* **The instruments that came out of it** (all cheap, all reused since): the same-field floor for any
  two-ensemble comparison (0.0471 / 0.0521 / 0.0589 pooled, measured from two independent trajectories
  under one field); the **edge gap** -- the outer-5-percent mass of a fitted field's implied distribution
  against its target -- which is what predicts whether a fit cycles, where the residual does not (the
  only cycling arm read -0.378 while four converging arms sat inside 0.012, including the one with the
  worst residual); a drift-versus-cycle test (a consistent step direction means no fixed point, a sign
  flip every round means a 2-cycle); and the local-geometry transfer measurement behind
  `docs/archive/cg_allatom_interface.md`.
* **Two plans, two answers.** Plan B' (change the operator) works on the coordinates that are not
  coupling-dominated. Plan C (change the target to the ensemble the field itself produces) buys **no
  retention** on a seven-chain pool (0.39-0.42 A against a 0.40-0.41 A baseline), though it does stop
  the angle's self-motion (0.94x the floor) at the cost of drifting away from the deposited reference --
  which is what a self-consistent target means. And the parametric (Chebyshev) refit is numerically
  fragile until weighted correctly, tapered at the tails, and ridged relative to the largest eigenvalue.

### 6. Where the field stands now

* **What the CG level owns**: the P trace. Its four scored coordinates are local functions of it, and on
  the 867 fragments those local marginals are length-independent to 4-13 percent, with the closure a
  circle imposes worth under 0.5 percent (see `docs/archive/cg_allatom_interface.md`).
* **What it does not own**: stacking. `stack` = |P(i) - P(i+2)| is algebraically derived from the bond
  and the pseudo-angle, the three-bead residue has no plane, normal, rise or twist to express stacking
  with, and the all-atom level carries that physics -- while inheriting the trace (a 1.5 A trace error
  halves the reconstructed stacking fraction). It should not be a CG scoring target.
* **What is still open**: the angle has no fixed point under either operator at full pool; whether the
  pooled deposited marginal is something a physical 300 K ensemble produces at all has never been
  measured (the all-atom route to it is wired but needs a relaxed starting structure); and the
  parameterisation method is still one-shot rather than iterative, which is the one structural
  difference from IsRNA2/IsRNAcirc that the comparison document calls "the sharpest".
* **The sampling gap, for scale**: their calibration is 10 replicas x 50 ns x 3 = 1,500 ns; ours is
  24 replicas x 0.08 ns = 1.92 ns of production sampling (1.28 ns in the calibration harness). The
  exchange scheme is comparable or better; the LENGTH is 625x shorter per replica.

### 7. The lessons worth keeping

1. **A criterion beats a target.** Every stiffness that came from "this looks right" was off by 20-70x
   in one direction or the other; the ones that came from a measured distribution were not.
2. **The functional form is part of the parameters.** Skewed coordinates cannot be pinned by springs,
   and a basis that cannot express the target's shape (the dihedral's edge mass) will cycle no matter
   how it is tuned.
3. **Check the instrument before believing the number.** Two of this field's headline metrics were
   later shown to be dominated by their own denominators or by an uncontrollable coordinate.
4. **Do not score what the model cannot control.** `stack` beside three controllable coordinates made
   J monotonically worse-looking while the field improved.
5. **Measure the coupling, not just the marginal.** The coordinates that resisted are the ones whose
   sampled marginal is not their own table's Boltzmann distribution; the ones that behaved are the ones
   where that approximation holds.
6. **Record the dead ends with their numbers.** TriRNASP, the Fourier dihedral, the Chebyshev refit's
   first weighted least squares, the operator switch that never took effect, the launcher's LF-only
   byte layout -- each cost real time and each is now a paragraph instead of a rediscovery.


## 2026-10-01 — The fitted field reaches the pipeline, and it runs on the GPU

The tables the loop fitted were **never what the production pipeline sampled**: `torch_gpu_refine.py`
called `cg_energy_forces(pos, pairs, pw)` with no potentials at all — the analytic field inside
`torch_cgsim` — while everything Boltzmann-inverted lived in the calibration harness. The table
potentials were moreover cpu-only by an explicit deferral in `cg_potentials.make_potential`:
*"supporting cuda means moving the table tensors and the interpolation together, not just this call, so
it is left until something needs it"*.

**Now.** `TORUSFOLD_CG_TABLES=<file>` makes the pipeline's CG stage take the tabulated potentials
(unset = the analytic field, unchanged), and the tables **follow the coordinates' device**:
`force_reference.table_for` returns the installed record with only `U` moved, cached per device, and
`use_table_file` clears the per-device copies when it installs a new one. The cpu numbers are
**bit-identical**, pinned in `tests/test_table_potential_device.py` against goldens captured before
the change.

**Which table.** `results/production_tables.npz`, composed by `scripts/build_production_tables.py`
from the campaign: **bb_bond and dihedral from the converged 9-round tables**, **angle from the
pre-campaign refit** (every campaign update made it worse: implied sigma 0.3218 -> 0.1307 while the
sampled sigma fell only 16 percent), and **stack not injected at all** (an exact function of the other
two; `torch_cgsim`'s stack spring is zero for the same reason). The builder asserts the two sources
share `lo`/`binw`, so the mix cannot silently combine different grids.

**GPU verification, on this machine**, with the ROCm torch build (`2.12.0a0+rocm7.13.0a20260313`,
AMD Radeon 8060S): `scripts/verify_cg_gpu.py` evaluates the tabulated field on one structure on cpu
and on cuda — **E identical to the last bit, forces agreeing to 3.7e-16 relative**. The cpu build of
torch (circrna3d env) has no CUDA and skips that half.

**What this does not settle**: the shipped dihedral table is the campaign's, so it carries the
edge-mass deficit (-0.18 at full pool) that the basis work showed is a Chebyshev problem rather than a
rounds problem; the angle's table is the pre-campaign one, i.e. the best available and not a converged
object. Both would be refit by the per-coordinate rules that work argues for.

---

## 2026-10-04 — The criterion was the defect: what the field delivers, and what the loop was chasing

Two more arms ran, both failed their stated criteria, and the failure turned out to be the criteria.
The full account is Part 12 of `docs/ibi_loop_and_oxrna_findings.md`; this is the part a reader of the
field needs.

**The measurement that reframed it.** For each of the campaign's nine rounds, the table that round
sampled under and the pooled histogram it produced are both on disk, so the loop's effective gain can be
regressed per coordinate (`scripts/ibi_transmission_scan.py`, sampled sigma against the table's own
implied sigma):

| coordinate | slope | R2 | pooled sigma at an infinitely narrow table | target sigma |
| :-- | --: | --: | --: | --: |
| bb_bond | 1.111 | 0.94 | 0.0004 | 0.0540 |
| angle | 0.311 | 0.70 | **0.3253** | **0.3218** |
| dihedral | 0.970 (campaign) / 0.251 (arm A) | 0.82 | 0.159 | 0.625 |
| stack | 0.019 | 0.00 | 0.146 | 0.127 |

The angle's line says its pooled width cannot fall below 0.3253 — **one percent above its target**. The
campaign drove its implied sigma 0.3218 -> 0.1307 while the ensemble followed a third of the way; that
coordinate was finished and the loop could not tell. The dihedral's two slopes disagree because the two
runs moved the ANGLE in opposite directions: a coordinate whose marginal is not its own table's
Boltzmann distribution cannot be measured with another coordinate moving.

**The verdict on the shipped field** (`scripts/ibi_verdict.py`, criteria |ln(sigma_sampled/sigma_target)|
<= 0.10 and |edge gap| <= 0.05): bb_bond, angle and dihedral all PASS at the campaign's round 8 —
sampled 0.05405 / 0.34930 / 0.61074 against targets 0.05402 / 0.32176 / 0.62471, with edge gaps of
+0.0000 / -0.0034 / -0.0015. bb_bond and the dihedral passed at round 1; the angle from round 7. `stack`
is DERIVED (no table, algebraic in the other two) and is excluded rather than judged. **So the tables in
`results/production_tables.npz` reproduce the target's pooled local-geometry marginals to within 8
percent in width and 0.003 in edge mass — and the nine rounds of table churn after round 1 bought the
ensemble almost nothing.**

**What that does not mean.** The pass is at the *pooled equilibrium* marginal, which is what the tables
were fitted to. It is not a statement about any single structure's geometry: that is what the 2OIU A/B
measures, and its single draw showed the fitted dihedral field halving the log width error
(|ln(sd/target)| 0.635 -> 0.307) while the two products differed by 13.5 Å from REMD's unseeded start.
Ten draws per arm are running now, analysed as a PAIRED difference so the scatter cancels.

**Which of the failures were real.** Three, and all three are now handled in code rather than by taste:
the support gate (arm A refused bb_bond, the one coordinate with unit gain, at 1.02 percent against a
1 percent gate — now `IBI_LOOP_SUPPORT_GATE`, printed in the header, with the out-of-support fraction
recorded for refused and frozen coordinates too); the undamped replacement step (the angle's
self-consistent refit overshoots its own target by 1.64x at gain 1 and is neutral at 0.3, measured in
`scripts/ibi_step_gain_scan.py`; the dihedral's edge loss is monotone in gain, -0.016 at 0.1 against
-0.137 at 1.0); and the missing per-round record (every round json now carries a `marginals` block —
sampled and implied sigma and edge, before and after, with the edge gaps).

**Still open.** Whether a table-side fixed point exists at all: an arm with the angle and bb_bond frozen
and only the dihedral stepping is running (seven chains, gain 0.5), which is the first clean measurement
of one coordinate's own transmission. And the record itself moved — the campaign's 0.44 GB sat at
`results/ibi_relax` for nine rounds and was parked elsewhere on 2026-10-01; `ibi_core.campaign_root()`
resolves it now, and eight scripts still hard-code the old path.

**Correction (added the same day).** The 2OIU A/B was reported above from ONE draw per arm. Ten
draws per arm, analysed as a paired difference, give the dihedral effect as **+0.006 +- 0.132 with 6 of 10
draws closer to target** -- indistinguishable from zero, against a draw-to-draw scatter of 8-10 percent.
The single draw was one sample, and the fitted field has NOT been shown to change this stage's geometry.
## 2026-10-05 (3) — The base-level marginals, on both sides, measured

`scripts/measure_base_coords.py` measures the same bead-only coordinates on the crystals and on the
field's own sampler (production tables, the loader's 3-bead chains, 5000 steps each). Every coordinate is
a function of the three beads the model carries, so a fitted potential could use any of them directly.

| coordinate | target (20 fragments, 1625 pairs) | sampled (4 chains, 17 920 pairs) |
| :-- | :-- | :-- |
| base-base distance | 0.570 +- 0.198 nm | **0.703 +- 0.171 nm** |
| rise along the mean triangle normal | 0.366 +- 0.193 nm | 0.337 +- **0.350 nm** (5th percentile **-0.43**) |
| triangle-normal angle | 28.6 +- 21.1 deg | **49.4 +- 20.9 deg** |
| twist | 44.1 +- 34.2 deg | 51.9 +- 42.7 deg |

Total-variation distances are 0.39-0.55, against 0.07-0.12 for the target's OWN internal spread (all
consecutive pairs against the helical subset) -- so the gap is five times the reference's own ambiguity and
is a model deficiency, not a definition artefact. The three things a stacking term would have to fix, in
the order the numbers put them: **align neighbouring base frames** (theta 49 against 29 deg), **pull the
bases together** (0.70 against 0.57 nm), and above all **make the rise one-sided** -- the target has one
base lying over its neighbour at +0.37 nm, the model is symmetric about zero with a 5th percentile at
-0.43 nm, i.e. no preference at all for over rather than under. That asymmetry is stacking.

Instrument caveat: the P-C4'-N triangle normal is 20.2 deg (median 16.2) off the true base-plane normal.
Both sides use the same proxy so the comparison is fair, but a fitted term would inherit the systematic.

Two incidental findings, both recorded because both cost a run: the 24-34 residue band holds 31 chains and
the loader's order is not stable across processes (so the loop's seven-chain arms may not be the same seven
chains between runs), and one chain in that band hangs `run_round` reproducibly -- twice at the sixth
chain of an unordered pool, with five completed chains' work lost the first time. The sampler now runs one
process per chain under a timeout, so a hang costs one chain.

The same ten draws also show that these products sit far from the fitted marginals (dihedral 0.386 against
a 0.630 target, angle 0.426 against 0.320) because a short free refinement with bond restraints is not an
equilibrium sampler for them -- the tables are validated at the loop's pooled level (the entry above),
not here.

---

## 2026-10-05 — Stacking, measured: the bottleneck is base placement, not the force field

We argued for months about whether the CG field is too simple to carry stacking. It was measurable, so we
measured it (`scripts/measure_base_stacking.py`; full tables in Part 13 of
`docs/ibi_loop_and_oxrna_findings.md`). A helical step is a WC pair whose next pair along the helix also
exists; a step is stacked when its rise is 2.5-4.0 A and the angle between base-plane normals is <= 30
degrees. The instrument passes its own calibration: over 20 crystal fragments from `_cgdata/combined`,
**98.8 +- 5.4 percent of helical steps are stacked, rise 3.35 +- 0.08 A, normal angle 8.3 deg**.

Three results, and the first is the one that changes the plan:

* **The template loses more than the trace does.** Take the P trace from the crystal itself — zero trace
  error — and `reconstruct_all_atom` still loses **57 percent of the stacking** (98.8 -> 43.1 percent),
  puts the bases **0.9 A too close** (rise 3.35 -> 2.50 A), and keeps only **10-20 percent of the WC
  contacts** (2OIU 0/12, 4QK9 2/22). The shipped CG-to-all-atom step reproduces neither the rise nor the
  pairing geometry of the structure it is given.
* **Trace error compounds it**: 0.78 A RMSD costs a third of the stacking, 1.7 A two thirds, and by
  2.7 A it is gone. The note that "a 1.5 A trace error halves the reconstructed stacking fraction" was
  optimistic in the middle of the range.
* **Our own products are past that point anyway**: the ten-draw 2OIU A/B traces sit **8.95-11.19 A** from
  the deposit (they keep their circular closure, BSJ 0.64-0.72 nm against 0.59) and reconstruct to 0-25
  percent stacked. That is expected of a free 300 K sampler stage, not a defect — but nothing downstream
  can put the stacking back.

**What this decides.** The next month should not start with a new CG energy term. It should start with
**base placement**: superpose the template on the SAMPLED per-residue frame (P, C4', base site) instead of
on the P trace plus an axis heuristic, and optionally solve the base orientation against the pair list.
The three beads already exist in the model and `real_cg_beads` already reads them out of this very
reconstruction — whose own docstring says fabricated beads "carry no base identity, so no base-specific
quantity can be expressed on them". A base-level stacking term (oxRNA does it at the same three sites per
nucleotide, on a dedicated stacking site, with orientation from the rigid body) is the step AFTER that
one, because today nothing the CG base beads do can reach the product.

**Not claimed**: the WC test is strict (all key contacts within 3.6 A) and calibrated on crystals; amber
refinement downstream might recover pairing, which is testable with `amber_refine` and was not run here.
And a restrained refinement protocol is a different experiment from the free sampler measured above.

---

## 2026-10-05 (2) — The reconstruction fix works on ideal input and stops on real input, which is the useful answer

Acting on the entry above: `aform_from_template.reconstruct_all_atom_from_beads` now fits the 1EHZ
template onto the three beads the model already carries (P, C4', N9/N1) instead of guessing the base roll
from the P trace, and `torch_gpu_refine` hands those beads out through `cg_bead_sink` -- they always
existed (REMD carries the full 3-bead state across rounds) but the interface sliced them to P and threw
the rest away. Both are opt-in; the shipped product is unchanged.

**On ideal input it recovers half the loss.** Same 20 fragments, same pairs, same instrument:

| reconstruction | stacked | rise | twist |
| :-- | --: | --: | --: |
| crystal | 98.8% | 3.345 A | 31.1 deg |
| P-trace template (shipped) | **43.9%** | **2.466 A** | 28.8 deg |
| sampled bead frame (new) | **75.7%** | **3.032 A** | 31.9 deg |

and the surviving Watson-Crick contacts go from 2/25 to 11/25 on the fragment checked in full. The
residual gap is the template's rigid-residue idealisation -- with three points per residue the base PLANE
is still inferred.

**On a real CG state it does nothing yet, and the reason is the field.** One short 2OIU draw (4 replicas x
5000 steps) landed 7.74 A from the crystal trace, where the shipped path gives 0 percent stacked and the
bead-frame path gives 16.7 percent with a **twist of 100 +- 54 deg**. That is not a reconstruction
artefact: the sampled base frames are simply not oriented relative to each other, because nothing asks them
to be -- `K_STACK = 0`, no base-orientation term, and the roll is held only by the K_LINK_* links against
a pair potential that does not care.

**So the two changes are a pair.** The reconstruction is necessary (without it no base-level quantity can
reach the product) and insufficient (without a field term it faithfully reproduces arbitrary rolls). What
is now available, and was not before, is the ability to measure BOTH sides of a base-level IBI: the target
from the crystal database with `scripts/measure_base_stacking.py`, and what the current field produces
from the bead sink on a real run. That is the next thing to do, and it is a measurement rather than a
parameterisation.

## 2026-10-05 (4) — A stacking term from the measured target: the model can move, the term cannot reach it

Built one, from measurements rather than taste: `base_frames.py` (the base plane as a FIXED linear
combination of the three bead vectors, 0.000 deg exact per base, 6.573 deg with one pooled triple, no
per-step reconstruction) and `base_stacking.py` (E = -eps * SUM Gd(d) Gr(rise) Gt(theta), each well at a
measured target value with a measured width, wired into `cg_energy_forces` as an opt-in injection,
gradient verified to 6 decimals). The target, through that same map: d = 0.570 +- 0.198 nm, rise =
0.328 +- 0.188 nm with a 5th percentile at +0.06, theta = 24.5 +- 22.2 deg.

Scanned at eps = 0, 4, 10 kJ/mol per pair, two chains, 5000 steps:

| eps | d (nm) | rise 5th pct | theta (deg) | trace J |
| --: | --: | --: | --: | --: |
| target | 0.570 | +0.06 | 24.5 | -- |
| 0 | 0.750 | -0.33 | 51.4 | 0.139 |
| 4 | 0.727 | -0.43 | 50.9 | 0.189 |
| 10 | 0.700 | -0.52 | 46.4 | 0.119 |

It moves d and theta 10-20 percent of the way and does nothing for the one-sidedness. Two measurements say
why, and the first is the good news: **rotating one residue's rigid unit about the local P-P axis costs only
1.3 kJ/mol at 10 degrees, 6.2 at 20, 12.4 at 30** -- the roll is a SOFT coordinate, so the rigid-link
network is not the obstacle. The obstacle is the term's shape: the three factors multiply and each is <= 1,
and at the sampled theta of 51 degrees the orientation factor is exp(-(51.4/25)^2) = 0.016, so a nominal
10 kJ/mol acts as **0.07 kJ/mol per pair** against the ~6 kJ/mol a 20 degree reorientation costs. The reward
vanishes exactly where repair is needed.

## 2026-10-05 (5) — Ten chains, paired: the term works and the trace does not pay for it

The two open items from the entry above were both weight problems. Discounting the distance penalty to 0.3
(the rise and the orientation are the coordinates the model cannot express at all today; the base-base
distance at least has the pair and link network pulling on it) and raising the strength fixes both at once.
Ten chains, 5000 steps, production tables, SAME chains and SAME seeds with and without the term, so every
number is paired (weights 0.3/1/1, eps = 45 kJ/mol = 18 kBT per pair):

| coordinate | without | with the term | paired change | chains improved |
| :-- | --: | --: | --: | --: |
| base-base distance | 0.7164 +- 0.0359 nm | **0.5906 +- 0.0306** | -0.1258 +- 0.0267 | **10/10** |
| rise, TV vs target | 0.5418 +- 0.0418 | **0.2717 +- 0.0310** | **-0.2702 +- 0.0449** | **10/10** |
| **rise, 5th percentile** | **-0.3267 +- 0.1566** | **+0.1625 +- 0.2225** | **+0.4891 +- 0.2250** | **9/10** |
| neighbour-normal angle | 49.65 +- 2.26 deg | **18.74 +- 1.07** | -30.90 +- 2.42 | **10/10** |
| trace joint J | 0.1269 | **0.1234** | -0.0035 | -- |

Three readings. **The stacking asymmetry is created and every chain agrees**: the rise's 5th percentile
goes from -0.327 nm to +0.163 (paired +0.489 +- 0.225, twice its own spread) against a +0.06 target -- the
model starts with no preference for one base lying over its neighbour and ends with the right one, slightly
overdone. **The trace does not pay**, once the weights are right: J 0.1269 -> 0.1234, the opposite of the
two-chain equal-weight indication, which says the trace cost was the distance penalty's rather than the
stacking term's (at full weight it was fighting the backbone's own pair and link network). And **the
strength is slightly past optimal**, visible as overshoot rather than damage: theta lands at 18.7 against a
24.5 target, the rise's 5th percentile at +0.16 against +0.06.

## 2026-10-05 (6) — The strength lands on the target at 35 kJ/mol, and a loop arm holds the trace with the term on

Ten chains each, same chains and seeds as the baseline, so every number is paired:

| eps | base-base distance | rise 5th pct | neighbour-normal angle | trace J |
| --: | --: | --: | --: | --: |
| target | 0.5704 +- 0.1980 | **+0.06** | 24.51 +- 22.22 | -- |
| 0 (off) | 0.7164 +- 0.0359 | -0.3267 +- 0.1566 | 49.6469 +- 2.2567 | 0.1269 |
| 25 | 0.6061 +- 0.0359 | -0.0022 +- 0.2957 | **24.7943 +- 2.4062** | 0.1369 |
| **35** | 0.5977 +- 0.0308 | **+0.0842 +- 0.2517** | 21.8752 +- 2.7995 | **0.1215** |
| 45 | 0.5906 +- 0.0306 | +0.1625 +- 0.2225 | 18.7428 +- 1.0735 | 0.1234 |

The parameters now LAND on the target: at 25 the orientation angle matches to 0.28 degrees but the rise is
only just one-sided; at 35 the rise lands closest (+0.084 against +0.06) with a mild orientation overshoot and
the trace J is the only one BELOW the baseline; at 45 both overshoot. **35 kJ/mol per pair is the setting to
carry.**

And a seven-chain loop arm with the term ON (45, run before the scan chose 35), three rounds, production
protocol: the three trace coordinates are all inside tolerance by **round 2** (bb_bond |ln| 0.031, angle 0.055,
dihedral 0.042), the dihedral's update falls from 4.19 and 5.30 kJ/mol to **0.99** with the moment norm at
0.0502, and the per-chain J median returns to 0.1477 -- the level the nine-round campaign had at its own second
round. So a base-level term does not stop the trace tables from reaching their reference marginals.

What the loop cannot see yet, and it is the next wiring step: it bins the six coordinates of
`boltzmann_bonded.COORDS` and none is a base-level quantity, so this arm can report the trace but cannot
measure or fit the base-level marginals. `base_dist`, `base_rise` and `base_theta` have to become
coordinates of the sampler's own binning loop -- their definitions and their measured targets are in Parts
15-18 -- and then the loop can close on them the way it closes on the trace.

Next: scan eps over 25-35 to land nearer the target, then the question that now has an instrument -- does the
term survive being fitted by the loop against an ensemble that contains it (`IBI_LOOP_BASE_STACK=45` plus
form and weights, with everything else the loop already does).

**Correction and the result (same day).** The rewrite tried first -- `eps * (1 - Gd*Gr*Gt)` -- is
dynamically IDENTICAL to the reward, because `SUM(1-f) = N - SUM(f)` differs by a constant; the sampler
proved it by producing bit-identical trajectories for the two forms. The limiter is the gradient, not the
value: at an unstacked configuration (mean neighbour-normal angle 59 deg, eps = 10) the reward and the
penalty both give max|F| = 7.4 while `eps * [(1-Gd) + (1-Gr) + (1-Gt)]` gives **111.5**.

With that sum form, two chains, eps = 0 / 5 / 20 kJ/mol per pair:

| eps | d (nm) | rise mean, 5th pct | theta (deg) | trace J |
| --: | --: | --: | --: | --: |
| target | 0.570 | 0.328, **+0.06** | 24.5 | -- |
| 0 | 0.750 | 0.332, -0.33 | 51.4 | 0.139 |
| 5 | 0.657 | 0.302, -0.42 | 41.8 | 0.147 |
| 20 | 0.621 | **0.317, +0.02** | 30.1 | 0.215 |

**The one-sidedness appears** -- the rise's 5th percentile goes from -0.33 nm to +0.02 against a +0.06
target, and the rise's TV halves. Two things are left open and both are in the table: the distance well
over-constrains its WIDTH at eps = 20 (sd 0.122 against the target's 0.198, so d's TV gets worse while its
mean gets better -- the three penalties should not carry equal weight), and the trace J rises with the
strength (0.139 -> 0.215 on two chains) because the trace tables were fitted with no base-level term
present. Together they are the next measurement: per-coordinate weights, and/or a trace refit with the term
on -- which is what the loop exists for.

Next: an additive / pairwise-attraction form whose scale IS eps and whose orientation factor is broad (oxRNA
uses a distance attraction times an orientation factor, not a product of three narrow wells), then the same
scan. The model can carry stacking; this term cannot reach it.

**Caveats, stated**: the real-run comparison is one short draw at 7.74 A drift with a +- 54 deg twist
spread -- a demonstration, not a converged measurement; the ten-draw 2OIU products predate the sink and
carry no beads; and 75.7 percent is a reconstruction validation on ideal input, not a product claim.



