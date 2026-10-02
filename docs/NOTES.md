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



