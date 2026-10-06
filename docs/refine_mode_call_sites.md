# The refine_mode call sites: who calls torch_gpu_refine, and which of them want the refinement

WHAT THIS IS. An inventory of every caller of `torch_gpu_refine`
(`src/torusfold/scheme2/torch_gpu_refine.py`) in this repository, classified the way the
`refine_mode` argument asks for it:

    REFINEMENT  the call is HANDED a structure and the structure is expected to be KEPT
    FOLD        the call is handed a sequence, or a trace used only as a STARTING POINT
                for sampling, and a new structure is expected

WHY THE DISTINCTION IS NOT COSMETIC. Measured end to end on 2OIU, 71 nt, with
TORUSFOLD_BASE_STACK=16.6:0,1,0.19 and the beads read from the deposit
(docs/ibi_loop_and_oxrna_findings.md Parts 24-27; the deposit's own base-frame cosine is 0.901):

| protocol                  | base-frame cosine | helical steps stacked | trace vs the deposit |
|---------------------------|-------------------|-----------------------|----------------------|
| `refine_mode="fold"` (the shipped default) | 0.650 | 0.0 percent   | 8.0-10.5 A           |
| `refine_mode="refine"`                   | 0.907 | 50.0 percent  | 1.62 A               |

A base-level stacking term worth ~6 kBT per pair at 300 K is worth ~1.5 at the top of the folding
path's 1000 K ladder, so the folding path scrambles the base frames BY CONSTRUCTION and the
refinement does not.

HOW THEY WERE FOUND. `grep -rn torch_gpu_refine` over the repository and `grep -rn "refine("`
over `src/` and `run_2013nt.py`. That yields FOUR direct call sites and two further entry points
that reach one of them indirectly.

NOTHING HERE IS CHANGED. As of 2026-10-06 every call site passes NONE of `refine_mode` /
`refine_steps` / `refine_temp`, so all of them run the built-in default, which is the folding
protocol. That is why the environment route (`TORUSFOLD_REFINE_MODE`, `TORUSFOLD_REFINE_STEPS`,
`TORUSFOLD_REFINE_TEMP`, resolved by `_resolve_refine_settings` at
`torch_gpu_refine.py:95`) exists at all: it is the only way to switch a call site without editing
it.

## Summary

| # | call site | is handed | it is | switch it? |
|---|-----------|-----------|-------|------------|
| 1 | `src/torusfold/scheme2/isrnaclong.py:1822` | the round's input | REFINEMENT | YES |
| 2 | `run_2013nt.py:99` | sequence + SS, reaches #1 | REFINEMENT by proxy | YES, env only |
| 3 | `scripts/benchmark_2oiu_field_ab.py:105` | the crystal trace | FOLD | NO |
| 4 | `scripts/measure_cg_frame_gain.py:63` | the crystal trace | FOLD (probe) | NO as written |
| 5 | `scripts/rerun_2oiu_fixed_field.py:111` | the crystal trace | FOLD | NO as written |

In one line each: #1 is the production path whose whole job is to improve a structure it was given;
#2 is the same call reached from a driver with no argument for it (the environment is its only
route); #3 holds the protocol fixed because the FIELD is its independent variable; #4's numbers are
properties of the protocol it was measured under; #5 mirrors #3 deliberately so its acceptance
numbers stay comparable with the recorded old-field baselines. The sections below carry the detail.

---

## 1. `src/torusfold/scheme2/isrnaclong.py:1822` -- the production Level-2 refinement: REFINEMENT

WHAT IT PASSES. `refine_input, round_dir, sequence, secondary_structure, name=f"remd_r{round_idx}"`,
`nstep=max(100000, n_steps)`, `use_remd=True`, `remd_n_replicas=_remd_reps`,
`use_multistage_remd=True`, `verbose=verbose`, `use_physical_relax=True`,
`skip_minimal_fold=(round_idx > 0)`, `use_trirnasp=False`, `use_trirnasp_force=False`,
`trirnasp_scale=0.002`, `trirnasp_update_freq=1000`, `use_staged_tri=True`,
`tri_stage_config={CG-only 1000 / Ramp-up 1500 / Tri-guided 2500}`, `use_adaptive_tri_weight=True`,
`on_report=_snapshot_cb`. It passes NEITHER `refine_mode`, `refine_steps`, `refine_temp`, NOR
`seed`, `bead_source_pdb`, `cg_frame_allatom` or `cg_bead_sink`. The function object is bound
at `isrnaclong.py:1794` (with `openmm_gpu_refine` as the fallback at `:1797`).

WHAT THE INPUT IS. A structure, in every round:

* round 0: `merged_aa.pdb` (this run's own all-atom Level-1 assembly, `:1745-1752`) if it is
  fresh, else `vfold3d/assembled.pdb` (`:1768`), else a P-only trace written from the in-memory
  coordinates (`:1781`);
* rounds > 0: the previous round's best CG coordinates written to `_prev_round_cg.pdb` (`:1787`),
  else the previous round's output PDB (`:1791`).

WHY IT IS A REFINEMENT. The result is read straight back into the loop
(`coords_relaxed = _read_pdb_p_coords(pdb_out)`, `:1889`) and carried into the next round, and the
documented purpose of the level (`:1728-1733`) is to improve the supplied fold. Nothing in the call
wants new sampling: a 1000 K re-anneal is the opposite of "keep the structure you were given".
This is also the call the production entry points reach (README.md:89; `run_2013nt.py:99`;
`serve.py:2373`).

TWO THINGS TO KNOW BEFORE SWITCHING IT, both measured elsewhere and neither an objection:

* THE BEADS ARE FABRICATED WHEN THE INPUT IS P-ONLY, which is what rounds > 0 get:
  `_prev_round_cg.pdb` is written by `_write_coords_pdb` with P atoms only, and `beads_from_pdb`
  needs P, C4' and a base atom on every residue (it returns None when any is missing), so the branch
  falls back to `real_cg_beads` -- the heuristic beads whose base-frame cosine is 0.607 against the
  deposit's 0.901 (findings Part 25). A round fed the previous all-atom product, or round 0's
  `merged_aa.pdb`, does have the atoms, but they are the in-tree TEMPLATE reconstruction, not the
  deposit's frames. The measured 0.907 / 50.0 percent end-to-end number used beads READ FROM THE
  DEPOSIT, so `refine_mode="refine"` alone changes the protocol, not the quality of the initial
  state. Closing that gap means also passing `bead_source_pdb` = the full-atom file the round's P
  trace came from; `_read_p_coords` is NOT chain-aware (measured: 1527 P for a 71-nt chain), which
  is why that argument exists and why it must point at the right file.
* THE PROGRESS CALLBACK GOES QUIET. `on_report` is attached to the `BatchedREMD2D` instance
  (`torch_gpu_refine.py:662`), and refine mode never constructs it, so `_snapshot_cb` receives no
  frames. The GUI's live structure panel (`serve.py` streams that callback) would hold the input
  coordinates for the whole call instead of following the trajectory. If the panel matters, this is
  the argument that decides how the switch is made, not whether.

WHAT BECOMES INERT IF IT IS SWITCHED. `nstep`, `use_remd`, `remd_n_replicas`, `remd_n_steps`,
`use_multistage_remd`, `use_potential_refine` (guard at `torch_gpu_refine.py:489`),
`use_physical_relax` (guard at `:732`), `skip_minimal_fold`, `use_staged_tri`,
`tri_stage_config`, `use_adaptive_tri_weight` and `on_report` are all read only by the folding
path. The call would then run `refine_steps` (1000) steps at `refine_temp` (300 K) through
`ibi_core.run_round` and go straight to the all-atom step. That is the point of the mode, but it is
worth saying out loud because the arguments above LOOK like the controls of the run.

## 2. `run_2013nt.py:99` -- the end-to-end demo driver: REFINEMENT, by proxy

`isrnaclong_pipeline(...)` is called with sequence, SS, `max_seg_len=200`, `n_relax_rounds=20`,
`nrep=16`, `use_pyrosetta=True`, `resume=True` and so on (`:99-125`); it reaches #1, which is
the refinement. The pipeline's signature has no refine knob
(`isrnaclong_pipeline` is at `isrnaclong.py:551`, and `serve.py:1008-1019` builds the GUI form
from that signature), so this file cannot pass `refine_mode` even if the parent decides it should:
for this entry point the environment variable is the only route in, and the run will print
`[Torch GPU] refinement settings CHANGED BY THE ENVIRONMENT: ...` when it takes it.

## 3. `scripts/benchmark_2oiu_field_ab.py:105` -- the analytic-vs-tables A/B: FOLD, keep it

WHAT IT PASSES. `str(p_pdb), str(work), seq, ss, name="2oiu_" + arm`, `nstep=20000`,
`use_remd=True`, `remd_n_replicas=8`, `remd_n_steps=20000`, `use_multistage_remd=True`,
`use_potential_refine=True`, `verbose=False`, `skip_cg_to_allatom=True`, `seed=_seed`.
`p_pdb` is the deposited crystal P trace, written from `_2oiu_input.json` at `:132-138`.

WHY IT IS A FOLD. The crystal is the STARTING POINT, not a restraint: the script's independent
variable is the field (it sets or clears `TORUSFOLD_CG_TABLES` per arm at `:96-99`) and its
per-arm output is the DRIFT from the crystal plus the within-arm scatter -- the 7.56 / 7.90 A
figures that motivated seeding, and the twenty structures under `results/plan_c/ab_2oiu/`. Those
numbers are properties of the folding protocol. Switching this arm to refine would change the
protocol under the A/B and make every recorded comparison meaningless, without any label in the
output saying which protocol produced which arm.

WHAT WOULD DECIDE IT. Only a change of question: if the A/B were re-asked as "how well does the
fixed field KEEP a supplied structure", refine mode is the right protocol and the arm names,
acceptance numbers and the recorded baselines all have to be re-measured under it.

## 4. `scripts/measure_cg_frame_gain.py:63` -- the bead-frame reconstruction probe: FOLD

WHAT IT PASSES. `str(CG_IN), str(WORK), seq, spec["ss"], name="2oiu_cgframe"`, `nstep=2000`,
`use_remd=True`, `remd_n_replicas=4`, `remd_n_steps=5000`, `use_multistage_remd=False`,
`use_potential_refine=True`, `verbose=True`, `skip_cg_to_allatom=True`, `cg_bead_sink=sink`.
`CG_IN` is `results/plan_c/ab_2oiu/crystal_p.pdb` (`:38`), i.e. the crystal trace again.

WHY IT IS A FOLD. The structure is a starting point for one REAL sampled CG state; the script then
compares two all-atom reconstructions OF THAT STATE -- the shipped P-trace template against
`reconstruct_all_atom_from_beads` -- and its docstring writes the protocol down as "deliberately
SHORT ... the pre-fold and the physical relaxation on is enough". Its reported percentages
(25.0 for the P-trace path, 58.3 for the sampled frame in findings Parts 25-26) belong to the
folding protocol and would change under refine mode while the labels stayed the same.

WHAT WOULD DECIDE IT. What the number is quoted FOR. If it is "what the sampler's own bead frame is
worth for reconstruction", the protocol is part of the measurement and is best left as measured. If
it becomes "what the refinement's bead frame is worth", the run should say refine and the label has
to carry the protocol with it.

## 5. `scripts/rerun_2oiu_fixed_field.py:111` -- the defect-17 acceptance run: FOLD, keep it

WHAT IT PASSES. `str(p_pdb), str(outdir), seq, ss, name=f"2oiu_fixed_draw{d}"`,
`nstep=args.nstep` (default 20000), `use_remd=True`, `remd_n_replicas=args.replicas` (8),
`remd_n_steps=args.remd_n_steps` (20000), `use_multistage_remd=not args.no_multistage`,
`use_potential_refine=True`, `verbose=False`, `skip_cg_to_allatom=True`, `seed=seed`.
Again the input is the crystal trace written at `:85-91`, one draw per seed.

WHY IT IS A FOLD, and it is the closest call in this file. The crystal IS supplied and the reported
acceptance IS closeness to it (paired P-P near 18 A, Kabsch RMSD "well below the old field's 9.03 A
mean", `:21-23`). But the driver says explicitly that it "mirrors the call the old A/B made
(`scripts/benchmark_2oiu_field_ab.py`) so the two are comparable" (`:6-11`), and the reference
numbers it is judged against (8.4 A mean paired P-P, 8.0-10.5 A from the crystal, 9.03 A mean RMSD,
`:102-103`) were all produced by the folding protocol. Refine mode would very likely report a
better RMSD and destroy the comparison the run exists to make.

WHAT WOULD DECIDE IT. The acceptance criterion, not the input. If the claim being tested is "the
fixed field lands closer to the experiment than the old field did", the protocol must stay as
recorded. If the claim becomes "the closest achievable structure to the deposit", switch it and
re-measure the old-field baseline under the same protocol.

---

## References that are NOT call sites

* `scripts/crystal_energy_2oiu.py:245` imports `_cg_potential_kwargs` only. It is an energy
  instrument; no refinement is run, and it has no call to classify.
* `tests/test_seed_reproducibility.py:33` (import) and `:94-101` (an `inspect.signature` check
  that `seed` is last and defaults to None). Any change to the entry point's signature has to keep
  this test green; it does not call the refiner.
* `src/torusfold/scheme2/physical_relaxation.py:84` is a list, inside a docstring, of the callers
  of `relax_structure` -- torch_gpu_refine is the CALLER there, not a callee.
* `scripts/benchmark_bead_frame_product.py:8` and `scripts/test_real_cg_beads.py:25` mention the
  module in a docstring and a comment. The former deliberately uses `ibi_core.run_round` instead
  ("why the loop's sampler rather than the refiner").
* The prose references in `docs/` (`ibi_loop_and_oxrna_findings.md`, `NOTES.md`,
  `statistical_potentials_as_forces.md`, `REPRODUCTION_RESOURCES.md`, the archive audit) describe
  the module and are not call sites.

## What a switch reaches, and what it does not

* THE ENVIRONMENT ROUTE IS PER PROCESS AND HITS ALL FOUR DIRECT CALL SITES AT ONCE. None of them
  passes any of the three arguments, so `TORUSFOLD_REFINE_MODE=refine` in a shell that runs the
  production pipeline AND one of the measurement scripts switches both. A switch meant for #1 has to
  be scoped to that process (or the variables unset again) or the scripts in #3-#5 will report
  numbers from a protocol their labels do not name.
* THE PRINT IS NOT GATED ON `verbose`. #3 and #5 run with `verbose=False`, and they still print
  `[Torch GPU] refinement settings CHANGED BY THE ENVIRONMENT: ...` when the environment changed a
  value. A protocol switch that a run cannot report is worse than no switch.
* THE OPENMM FALLBACK BRANCH IS NOT REACHABLE BY THE ENVIRONMENT. If the torch import fails,
  `isrnaclong.py:1797` binds `openmm_gpu_refine`, whose signature has no `refine_mode`, so that
  run folds whatever the environment says. The switch controls the torch path, and only that.
* `TORUSFOLD_SEED` DOES NOT REACH THE REFINEMENT'S SAMPLER. The refine branch passes the RAW
  `seed` argument to `_refine_langevin` (`torch_gpu_refine.py:475`), which falls back to its own
  `20261005` (`:276`) when that is None: a refine-mode run through #1 is therefore reproducible
  by default, and `TORUSFOLD_SEED` varies every OTHER stochastic stage but not this one. Both are
  defensible; they are not the same policy, and a caller switching #1 should know which one it gets.
* THE OTHER REFINERS ARE OUT OF SCOPE. `amber_refine`, `pyrosetta_refine`, `openmm_amber_refine`
  and the OpenMM REMD path in `openmm_gpu_refiner.py` take no such argument and are not affected by
  any of this.
