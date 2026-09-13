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

## What is open

1. **IBI convergence.** Round 1 is one round. Whether J descends over rounds 2-4 is being measured
   (two chains, `ox_runs/ibi_iterate.sh`); the synthetic test says coupled systems need ~4.
2. **The refinement-from-coarse-structure test.** Materials exist; not run.
3. **The oxRNA jump's provenance.** Relax artifact or landscape? Needs a run that skips
   `RNA_relax` or equilibrates properly under the real potential first.
4. **`bb_bond` is still too wide and in the wrong direction** (1.05-1.13 against a single-chain
   ceiling of 0.998, `§3bb`). Unexplained, and no arm moved it.
5. **The Fourier/tabulated split.** Fourier fixes dihedral but cannot enter the IBI loop (the
   update is defined on a table, and the table's honest max force is 3.3x cap). Whichever way the
   loop goes has to resolve this.
