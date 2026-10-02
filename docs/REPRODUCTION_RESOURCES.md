# TorusFold-Hybrid — Resources & Reproduction

> Judge-facing reference (English). All figures were measured on the team's
> reference machine, September 2026. Figures marked [TBD] are measurements
> still running or not yet collected — they are never estimated.
>
> **2026-10-01 — what changed here, and why.** This page described a four-step
> access path in which three of the four steps could not be taken. The demo
> sequence and the predicted structure were git-ignored, no release existed, and
> `scripts/benchmark_2oiu.py` read an input that nothing in the repository wrote.
> The audit note at `docs/pipeline_audit_2026-09-13.md:1022` had already recorded
> the consequence for the Level 0 path. That is fixed: the outputs are in
> `artifacts/`, the crystal structure is too, and `scripts/verify_headline.py`
> re-derives the published numbers from committed files with numpy alone. One
> claim did not survive being checked and is corrected in §5 (pair satisfaction).
> Nothing else in this document was re-measured, so the [TBD] markers still stand.

## 1. Reference hardware

| Item | Value |
|---|---|
| Platform | AMD Ryzen AI MAX 395 (APU, unified memory), Windows |
| GPU mode | GPU-accelerated path of the pipeline (REMD/MetaD/torch CG); peak unified-memory use ≈ **60 GB** |
| CPU mode | CPU-only path (OpenMM CPU platform, multi-replica parallel); peak RAM ≈ **30 GB** |
| Wall time, low config (CPU) | ≈ **7 h** — 2,013 nt end-to-end (REST2 ×8 × 20,000 steps; REMD ×2 × 8 relax rounds), measured on an **earlier build** (see provenance note below) |
| Wall time, full config (GPU, default) | ≈ **14 days** (estimated — not measured) on this APU; highly GPU-dependent, ≈ **8× faster on an NVIDIA A100** (≈ 2 days) |

*Provenance of the ≈ 7 h figure. It was measured on an **earlier internal build**
of the pipeline, which ran the OpenMM CPU path. That build is no longer in this
repository, so the figure is a historical measurement rather than something
reproducible from the current checkout: the argument values passed at the time
(`n_rest2_replicas=8`, `rest2_nsteps=20000`, `nrep=2`, `n_relax_rounds=8`) are
**not** the values in the checked-in `run_2013nt.py`, which carries a higher
configuration (`n_rest2_replicas=16`, `rest2_nsteps=100000`, `nrep=16`,
`n_relax_rounds=20`). The pre-built viewer structure (Level 4.9, PPR repaired)
is attributed to that run. A re-measurement on the current code is pending, so
the two wall times are not directly comparable. The ≈ 14 days GPU estimate has
not been measured either. Per-stage step counts for the current pipeline are
tabulated in §2.1.*
Why is the CPU path the measured reference? Two reasons. First, the
pipeline is heavily vectorized (batched NumPy/Torch kernels, multi-replica
parallelism), and the reference CPU — AMD Ryzen AI MAX 395 (Zen 5) — has wide
SIMD/AVX-512 support that these vectorized kernels exploit well, so the CPU
path is unusually strong on this specific chip. Second, on the same APU the
GPU-accelerated path required substantial ROCm-specific adaptation and its
measured throughput did not beat the CPU path — so all measured reference
runs use CPU. On discrete NVIDIA hardware (e.g. A100), the GPU path is
expected to run ≈8× faster (estimate, not yet measured).*

## 2. Runtime profiles of the headline demo (2,013 nt circRNA)

| Mode | Peak memory | Wall time (low config) | Notes |
|---|---|---|---|
| GPU path (full config, default) | ≈ 60 GB (unified memory) | ≈ 14 days (estimated, not measured); ≈ 2 days on an NVIDIA A100 (est., 8×) | Default is the full configuration; wall time highly GPU-dependent |
| GPU path (low config) | ≈ 60 GB (unified memory) | [TBD — not run] | Low-config timing was only measured on the CPU path |
| CPU path (low config) | ≈ 30 GB | ≈ 7 h (REST2 ×8 × 20,000 steps; REMD ×2 × 8 rounds), measured on an earlier build | Runs without a discrete GPU; wall time not yet re-measured on the current code |

The ≈7 h wall time was measured with the low configuration above and is
dominated by the **Level-2 REMD sampling** stage (2 replicas × 8 relax rounds)
together with the Level-4 REST2 run (8 replicas × 20,000 steps). Wall time
scales with the number of parallel replicas and per-replica steps; configure
the call arguments in `run_2013nt.py`.
### 2.1 Per-stage step counts and physical time (current pipeline, 2,013 nt)

Every number below is read from the current source at the listed locations, so
this table describes **this** checkout — not the earlier build behind the ≈ 7 h
figure. "MD steps" counts integrator steps only: minimization iterations and
gradient-optimizer steps are not molecular dynamics and have no physical time.

| Stage | Source | Nominal steps | Step kind | MD steps | Physical time |
|---|---|---:|---|---:|---:|
| Level 1 cross-segment post-relaxation | `segmented_vfold3d.py:1588,1592` | 3,000 | minimize 3,000 + MD 600 + minimize 1,500 | 600 | 1.2 ps |
| Level 1.5 global CG relaxation | `isrnaclong.py:1059,1062` | 5,000 | minimize 5,000 + MD 1,000 + minimize 2,500 | 1,000 | 2.0 ps |
| Level 2 pre-fold | `torch_gpu_refine.py:141-157` | 6 × 2,000 | **Adam optimizer** (lr = 1e-3) — not MD, no timestep | 0 | — |
| Level 2 REMD (inner) | `torch_gpu_refine.py:176-177` | 8 × 5,000 | Langevin REMD, 8 temperatures × 8 Hamiltonians (64 replicas) | 40,000 | 80 ps/replica |
| Level 2.3 5-bead annealing | `fivebead_folding.py:304,316,323,330` | 13,000 × _s | Langevin MD; `_s = max(1, L/200) = 10.065` at L = 2,013 | 130,845 | 130.8 ps |
| Level 2.5b post-conversion relaxation | `isrnaclong.py:1684-1689` | 3,000 | minimize 3,000 + MD 600 | 600 | 1.2 ps |
| Level 3.5 metadynamics | `isrnaclong.py:522,1893` | 200,000 | well-tempered MetaD, 8 replicas | 200,000 | 400 ps/replica |
| Level 4 REST2 | `isrnaclong.py:2023-2040` | `rest2_nsteps` | batched 2D-REMD, 8 temperatures × 8 Hamiltonians (64 replicas) | 20,000 / 100,000 | 40 / 200 ps/replica |
| Level 5 AMBER RNA.OL3 | `amber_refine.py:684,765-767`; `isrnaclong.py:2117-2121` | 3,000 | L-BFGS minimization + MD 150 | 150 | 0.3 ps |

Timesteps, each read from source: `torch_cgsim.py:1548,1778` = 2 fs;
`physical_relaxation.py:95` = 2 fs; `metadynamics_gpu.py:107` = 2 fs;
`fivebead_folding.py:272` = 1 fs; `amber_refine.py:684` = 2 fs.

**Total genuine MD: ≈ 0.66 ns per replica** at the low configuration
(`rest2_nsteps=20000`, 8 Level-2 rounds); **≈ 0.94 ns per replica** at the
checked-in defaults (`rest2_nsteps=100000`, up to 20 Level-2 rounds — early
stopping truncates). Two caveats for any write-up. First, replicas must not be
summed into a "total sampling time": report per-replica physical time and the
replica count separately. Second, both figures sit one to two orders of
magnitude below the 10–50 ns regime in which short MD refinement has been
reported to help a *good* starting model, and far below the >50 ns regime where
refinement is reported to drift (CASP15 refinement benchmark, PMC12513224). So
the physics stages here are local repair and constraint enforcement — clash
relief, BSJ closure, pairing restraints, A-form torsions — not conformational
search. The search is carried by the CG fold and the segmented ensemble.

Level 3 (RL fine-tuning) trains a policy and runs no MD; Level 2.6 (PyRosetta)
is conditional; Level 5.5 (PPR repair) is a geometric repair pass with no MD.

## 3. Four-level access for judges (no one needs to run the full pipeline)

| Level | What you get | Effort | Where |
|---|---|---|---|
| L0 — Verify | Every number the viewer displays, re-derived from committed files | `python scripts/verify_headline.py` — seconds, **numpy only** | `scripts/verify_headline.py`; the same data in `artifacts/2013nt/quality.json`, which names the entries that do *not* come back and why |
| L1 — View | Interactive 3D of the predicted 2,013 nt structure (42,831 atoms, Level 4.9 PPR repaired) | Open a file in a browser | `docs/circrna_3d_viewer.html` (fully self-contained: structure + 3Dmol.js + pako inlined; opens offline, no network needed) |
| L2 — Download | Full-atom PDB + the demo sequence + the 2OIU crystal structure | 1 click, already in the repository | `artifacts/2013nt/isrnaclong_final.pdb`, `artifacts/2013nt/sequence.txt`, `artifacts/2oiu/2OIU.pdb`. Archived copies (per-level outputs, quality JSON, DOI) are still to come at Wiki Freeze / Zenodo |
| L3 — Force-field check | 2OIU (≈100 nt; only experimentally resolved circRNA): X-ray structure → Level-2 relaxation → RMSD vs crystal | **17 min (CPU), measured** — final RMSD **1.83 Å**; needs OpenMM + ViennaRNA | `python scripts/benchmark_2oiu.py`, reading the committed crystal structure. Evidence that the force field does not distort known structures |
| L4 — Full run | End-to-end 2,013 nt all-atom structure | Memory 30–60 GB depending on path; wall time must come from a fresh measurement (see §1 note) | `python run_2013nt.py` — it falls back to the committed demo sequence, so it starts; adjust the call arguments for a shorter run (see README and §2.1) |

*Why L0 comes first. A level that says "reproduce the headline result" and then
asks for 30 GB and a day is not a reproduction path, it is a description of the
authors' machine. What a reader can actually check is a number, and the numbers are
now checkable: `verify_headline.py` re-reads the committed structure and the
committed IBI histograms with numpy and reports 20 of 32 checks reproducing, with
the other 12 localised to one coordinate (`intra_pc`, off by 2.2-4.2x in all
twelve committed runs) and explained in its own output. That ratio being published
is the point; a clean 32/32 would have meant the checks were too weak.*

## 4. External tools (why setup is non-trivial, and what it costs)

TorusFold-Hybrid is an orchestrator: physics core in this repo; the sequence
predictors are external subprocesses configured via environment variables
(see README and DEPLOY_EXTERNAL.md for versions/URLs).

| Tool | Required for | Disk footprint (approx.) |
|---|---|---|
| ViennaRNA ≥ 2.6 | SS folding (Level 0) | small |
| OpenMM ≥ 8.0 | all MD stages | small |
| PyTorch ≥ 2.0 | RL / GPU CG path | ~2–3 GB |
| RhoFold+ / trRosettaRNA2 / RNAbpFlow | Level 1 ensemble | RNAbpFlow checkpoint alone ≈ 1.3 GB |
| isRNAcirc (Windows binaries) | CG→all-atom | small |
| PyRosetta / structRFM / Infernal+RFAM | optional stages | several GB |

First-time environment setup is the dominant cost (hours), not the run itself.
All machine-specific paths are environment variables — no hard-coded paths.

## 5. Known-limitation cross-check (single source of truth)

- Viewer stat panel (Level 4.9 PPR repaired structure): BSJ closure 5.898 Å;
  bond RMSD 0.0082 Å; 3dRNAscore 27.18; global RMSD 1.23 Å. **All figures are
  internal self-consistency metrics of the PPR-repaired model — none are
  comparisons against predicted base pairs or experimental ground truth.**
  `artifacts/2013nt/quality.json` carries each of them with the recipe that
  re-derives it, if one exists.
- **Correction, measured 2026-10-01: "pair satisfaction 100.0%" is not what
  `pdb_analyzer.compute_pair_satisfaction` returns.** An earlier version of this
  file attributed the panel's 100.0% to that function. Run on the committed
  structure it returns **47.2%**: 5,532 P-P pairs within 15 Å, 1,723 of them
  Watson–Crick complements, 814 of those within 12 Å. Three different quantities in
  this codebase are called "pair rate" and only one of them is structure-inferred:

  | Quantity | Definition | Value on the delivered structure |
  |---|---|---|
  | `pdb_analyzer.compute_pair_satisfaction` | WC-complement pairs found within 15 Å, fraction of them within 12 Å — inferred from the structure alone | **47.2%** |
  | `isrnaclong._compute_pair_rate(coords, pairs)` | fraction of the *predicted* pair list with P-P < 12 Å; the pipeline prints it as "CG pair_rate (P-P<12A)" | the 100% family |
  | `isrnaclong._validate_structure` → `pair_rate` | the same at 15 Å | the 100% family |

  The panel's 100.0% is consistent with the second and third, not with the first.
  They are not the same measurement and should not share a label: the second and
  third score a pair list against the geometry it was derived from, so they are near
  tautological on any structure that is used to generate its own pairs. Whether the
  panel label, this file, or neither changes is a maintainer decision; the
  measurement is here so the decision is made on a number. It is re-run by
  `scripts/verify_headline.py` on every invocation.
- Experimental ground truth: only one circRNA structure exists (PDB 2OIU).
  2OIU is used as a force-field validity test: starting from the X-ray
  structure, the Level-2 relaxation takes 17 min (CPU) and ends at
  RMSD 1.83 Å vs the crystal — the force field does not distort known
  structures. This is not a from-sequence prediction benchmark, and no
  de-novo accuracy claim is made from it.
- Historical engineering log with measured dead-ends: docs/NOTES.md (dated).

## 6. One-liner for the wiki / software page

"Run anywhere with 30 GB RAM (CPU) or ~60 GB unified memory (GPU); the
2,013 nt demo takes ≈ 7 h with the documented low configuration (README shows
which arguments to set) — or skip the run entirely:
inspect the embedded 3D structure in your browser, download the PDB, and
see the 2OIU force-field check (17 min, RMSD 1.83 Å): known structures are
not distorted by the relaxation stages."
