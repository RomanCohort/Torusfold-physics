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

**What the CG level owes, measured separately** (`docs/cg_allatom_interface.md`): the field's four
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

