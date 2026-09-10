# TorusFold-Hybrid

_Team JLU-FBH · iGEM 2026 · Oncology Village — structure-prediction engine of the CirCure project_

> *"We choose to go to the Moon in this decade, not because it is easy, but
> because it is hard."* — J. F. Kennedy, 1962
>
> We chose to predict the 3D structure of long circular RNA for the same
> reason — because it is hard. To our knowledge, no one has yet produced an
> end-to-end all-atom fold of a circRNA beyond ~2,000 nt. At iGEM meetups, few
> teams reach for circRNA as a drug vehicle — not because circRNA is a poor
> idea, but because, without a structure-prediction tool, the design feels out
> of control. circRNA drug design (immunogenicity control, dsRNA/ssRNA exposure
> ratio, IRES translation efficiency) is evaluated through 3D coordinates:
> IRES-element accessibility and solvent-accessible surface area (SASA) do not
> exist without a structure. TorusFold exists so a future team can choose
> circRNA confidently — without wondering whether the job is even doable.

**The gap.** The 1980s–90s ribozyme era put RNA at the centre of biology, but
RNA 3D structure determination then fell two decades behind proteins:
RNA-only PDB entries remain a tiny fraction, and for circular RNA exactly
**one** complete structure exists (PDB: 2OIU). Template-based prediction
inherited the drought — 3dRNA yields locally irrational models on circular
constructs (PLoS Comput. Biol. 2024, doi:10.1371/journal.pcbi.1012293) —
while physics-based *de novo* folding stayed out of reach for 1000+ nt
circular RNA. That is the gap TorusFold fills.

Physics-based **circRNA 3D structure prediction**. A multi-predictor ensemble
(RhoFold+ · trRosettaRNA2 · RNAbpFlow) feeds a coarse-grained folding engine
that is relaxed with OpenMM molecular dynamics, replica-exchange and
metadynamics, then reconstructed to all atoms and refined under the Amber14-OL3
force field — producing experimentally plausible models for long (1000+ nt)
circular RNA. (An RL-based sampling scheduler exists as a preview feature —
see Implementation notes below.)

> This repository is the official software deliverable of Team JLU-FBH
> (iGEM 2026, Oncology Village). It is the structure-prediction engine of
> CirCure, our multi-epitope circRNA vaccine platform for triple-negative
> breast cancer — see the wiki Software page for how predicted 3D models
> feed the CirCure design loop. All source code lives on `main`; a
> release is created automatically at Wiki Freeze as the judging artifact.
> If you use an AI assistant here, please read
> [.claude/RESPONSIBLE_AI_USE.md](.claude/RESPONSIBLE_AI_USE.md) first.

## Visual overview

<p align="center">
  <img src="docs/images/torusfold_architecture.png" alt="TorusFold end-to-end pipeline" width="100%"/>
  <br/>
  <em>TorusFold end-to-end pipeline: circRNA sequence → multi-source secondary-structure consensus → base-pair constraints → segmented 3D ensemble prediction (RhoFold+ · trRosettaRNA2 · RNAbpFlow) → RL-guided coarse-grained folding & sampling → all-atom reconstruction → Amber14-OL3 refinement.</em>
</p>

<p align="center">
  <img src="docs/images/cg_forcefield.png" alt="Multi-resolution coarse-grained representation" width="100%"/>
  <br/>
  <em>Multi-resolution coarse-grained representation. The physics core folds each window at increasing resolution (3-bead → 5-bead → all-atom), activating more detailed interaction terms at each level.</em>
</p>

## Description

circRNA back-splicing creates covalently closed circular RNA molecules whose
three-dimensional folds are almost entirely uncharacterized — only one circRNA
has an experimentally resolved structure to date (PDB: 2OIU). We predict full
all-atom 3D models for long circRNA sequences (e.g. the 2013 nt TNBC-targeting
construct in `run_2013nt.py`) without requiring any experimental restraints.

**How it works**

```
sequence
  └─ Level 0   MUSES multi-source secondary-structure consensus
               (multisource_ss / ViennaRNA fallback)
               PF base-pair-probability fusion
               + NCM non-canonical pair detection
  └─ Level 1   segmented 3D prediction, ≤200 nt per chunk
               ensemble_predict: RhoFold+ · trRosettaRNA2 · RNAbpFlow
               region-adaptive weights (stem / BSJ / loop),
               Kabsch stitching, NCM distance back-inference
  └─ Level 2   RL-guided coarse-grained close (RL MCTS + steering forces)
               iterative relaxation, REST2 × T-REMD replica exchange,
               well-tempered metadynamics,
               PyRosetta conditional refinement
  └─ Level 5.5 PPR base-pair hydrogen-bond repair
  └─ all-atom  1EHZ-crystal-template reconstruction (aform_from_template)
               Amber14-OL3 constrained minimization (amber_refine)
  └─ output    PDB + per-residue confidence + immune/structure fingerprints
```

The main entry point is `isrnaclong_pipeline()` in
[`src/torusfold/scheme2/isrnaclong.py`](src/torusfold/scheme2/isrnaclong.py);
segmented prediction is in `segmented_vfold3d.py`; the physics/refinement core
in `openmm_gpu_refiner.py`, `rest2_remd_2d.py`, `metadynamics_sampler.py`,
`torch_cgsim.py` and `torch_gpu_refine.py`.

**Features**

- Three-predictor ensemble with region-adaptive weights
- NCM (non-canonical pairing) detection as fourth evidence source
- RL MCTS guiding long-range (far) pair closure
- REST2 × T-REMD and well-tempered metadynamics enhanced sampling
- 1EHZ-crystal-template all-atom reconstruction + Amber14-OL3 refinement
- Atomic checkpoint resume, per-level success guards
- Web UI with SSE log streaming and Mol* 3D viewer
- Per-residue confidence + immune/structure fingerprints

## Installation

```bash
git clone <this-repo>
cd torusfold-hybrid
pip install -e .

# Secondary-structure folding needs ViennaRNA (provides the `RNA` module):
conda install -c conda-forge viennarna      # or: pip install ".[ss]"
```

Optional extras: `[gpu]` (PyTorch CG path), `[ml]` (multi-task heads /
circRNA library), `[plot]` (analysis scripts), `[pyrosetta]` (full-atom
refinement; Linux/WSL).

> **Scope note.** This repository ships the physics/refinement core plus the
> web frontend. The sequence/structure predictors it consults — RhoFold+,
> trRosettaRNA2, RNAbpFlow and the DivideFold SS subprocess — are *external*
> tools. Point to the first three via the environment variables below;
> DivideFold is resolved from a `DivideFold-main/` checkout next to this
> repository (see the AI/model disclosure table). Two legacy aggregator modules
> referenced by `predict_3d_allatom` (`cg_forcefield`, `modification_aware`)
> are not yet published and are under active development.
>
> Full install/deploy instructions for every external tool (ViennaRNA, OpenMM,
> the three predictors, isRNAcirc, TriRNASP, PyRosetta, structRFM, Rfam):
> see [docs/DEPLOY_EXTERNAL.md](docs/DEPLOY_EXTERNAL.md).

## Usage

**End-to-end demo — 2013 nt circRNA.** Put the target sequence (plain text,
`T`→`U` handled) in `sequence.txt`, then:

```bash
python run_2013nt.py
# → output_2013nt/isrnaclong_final.pdb
```

Run flags (RL close, REST2 replicas, REMD rounds, PyRosetta, PPR repair,
pseudo-MSA fallback) are configured as call arguments in `run_2013nt.py`.

**Shorter runs, and the provenance of the ≈7 h figure.** The pre-built viewer
structure was produced by an **earlier internal build** of the pipeline (OpenMM
CPU path) that is no longer in this repository, so its ≈7 h wall time is a
historical measurement, not a figure reproducible from this checkout. The
checked-in `run_2013nt.py` carries a higher configuration (`n_rest2_replicas=16`,
`rest2_nsteps=100000`, `nrep=16`, `n_relax_rounds=20`). To run shorter, pass
smaller values explicitly — e.g. `n_rest2_replicas=8`, `rest2_nsteps=20000`,
`nrep=2`, `n_relax_rounds=8` — and budget the wall time from a fresh measurement.
Per-stage step counts for the current code are tabulated in
`docs/REPRODUCTION_RESOURCES.md` §2.1.

**Web server.**

```bash
python serve.py            # default port 8877
# open http://127.0.0.1:8877  (SSE log stream + Predict API + Mol* 3D viewer)
```

Pre-built interactive demo of the 2013 nt prediction:
`docs/circrna_3d_viewer.html`.

**External predictors & runtime paths** are configured through environment
variables (no hard-coded machine paths in this repository):

| Variable | Purpose |
|---|---|
| `RNABPFLOW_ROOT` | RNAbpFlow checkout dir (`checkpoint/RNA3DB.ckpt`, `inference_rocm.py`) |
| `RNABPFLOW_PYTHON` | Python used to launch RNAbpFlow (default: current interpreter) |
| `TRRNA2_RUNNER` | path to the trRosettaRNA2 runner script |
| `RHOFOLD_ROOT` | RhoFold+ checkout dir (`pretrained/rhofold_pretrained_params.pt`) |
| `ISRNACIRC_BIN_DIR` | dir containing `CG_to_allatom.exe` + DLLs (Windows, ASCII path) |
| `CG_TO_ALLATOM_COEFF` | isRNAcirc CG→all-atom coefficient dir |
| `TF_STRUCTRFM_MODEL` | pretrained structRFM checkpoint for the multi-task heads |
| `TF_SCHEME2_SRC` | optional extra source dir injected into the server `sys.path` |
| `PPR_INPUT_PDB` / `PPR_OUT_PDB` | input / output PDB for the PPR repair script |
| `RFAM_CM` | `Rfam.cm` path for cmsearch-based MSA (optional) |
| `TF_DIVIDEFOLD_ROOT` | DivideFold checkout dir (default: `DivideFold-main/` next to this repo) |
| `TF_DIVIDEFOLD_PYTHON` | Python used to launch the DivideFold SS subprocess (default: current interpreter) |
| `TF_DIVIDEFOLD_RUNNER` | path to the DivideFold SS runner script (default: `scripts/_dd_runner.py`) |

**Repository layout**

```
├── run_2013nt.py          end-to-end 2013 nt demo
├── serve.py               SSE + Predict API + web viewer
├── src/torusfold/
│   ├── scheme2/           pipeline core (47 modules): folding, RL, REMD/MetaD,
│   │                      NCM detection, ensemble predictor wrappers, Amber refine
│   ├── circrna_library/   CIF/PDB ingest + circular QC (gemmi)
│   └── web/               browser frontend (Mol*, live logs, prediction panel)
├── scripts/               curated analysis & diagnostics
└── docs/                  architecture, viewer, library notes
```

## Data and large files

This repository holds **source code only**. Model weights, datasets and heavy
predictor checkouts are *not* committed:

- RNAbpFlow (~1.3 GB checkpoints) and the TriRNASP statistical potential are
  third-party tools installed separately.
- Prediction outputs (`output_*`), model weights and caches are git-ignored.
- Datasets and trained models used for the iGEM season will be archived on
  Zenodo and linked here (DOI added at Wiki Freeze).

**Planned asset — a circRNA design-element library.** Beyond raw structures,
we plan to publish a curated, schema-documented element library that dissects
predicted structures into reusable design elements — back-splice junction
geometries, IRES-containing domains, exposure/immunogenicity-relevant motifs —
each with its 3D context. The goal is that teams designing circRNA payloads
can look up "how does this element fold" the way they look up a sequence
motif today. First entries ship with the Wiki Freeze release / Zenodo
archive; the underlying data layer (`circrna_library/`) already enforces
provenance separation between experimental structures, experimental
constraints, and physics-generated hypotheses.

## Contributing

Issues and merge requests are welcome. This is an active research codebase;
please keep changes focused and add tests under the project conventions when
possible. Every contributor remains responsible for the accuracy of their
commits — see [.claude/RESPONSIBLE_AI_USE.md](.claude/RESPONSIBLE_AI_USE.md).

## Authors and acknowledgment

Team JLU-FBH (iGEM 2026). Primary developer: Ziyi Yan. Built in the Dry Lab at
Jilin University.

## For judges & non-experts (quick tour)

You do **not** need to run the full pipeline to evaluate this tool:

1. **See a real predicted structure in seconds** — open
   [`docs/circrna_3d_viewer.html`](docs/circrna_3d_viewer.html) in any browser:
   it loads the pre-built 2013 nt prediction with no install (hover, rotate,
   recolor).
2. **Try the web UI** — `python serve.py` serves the Mol* viewer with live
   logs and a Predict API at `http://127.0.0.1:8877`.
3. **Reproduce the headline result** — install (below), put the 2013 nt
   sequence in `sequence.txt`, run `python run_2013nt.py`, and inspect
   `output_2013nt/isrnaclong_final.pdb`.

Running the full ensemble needs external predictors and (ideally) a GPU — see
[docs/DEPLOY_EXTERNAL.md](docs/DEPLOY_EXTERNAL.md).

Hardware requirements, measured runtime profiles (≈7 h CPU for the 2,013 nt
demo, dominated by the Level-2 REMD sampling; GPU-mode notes) and a
three-level access guide — view-only / download / reproduce — are in
[docs/REPRODUCTION_RESOURCES.md](docs/REPRODUCTION_RESOURCES.md).

## AI / model disclosure

This software **calls machine-learning RNA structure predictors as external
tools**; it does **not** train them, and it ships no trained weights:

| Model | Source | Used for | Weight/data provenance |
|---|---|---|---|
| RhoFold+ | Wang et al., *Nat. Methods* 2024 | per-chunk 3D prediction (`rhofold_wrapper`) | external checkpoint |
| trRosettaRNA2 | Li et al., *Nat. Commun.* 2021;12:5934 | per-chunk 3D prediction (`trrna2_wrapper`) | external checkpoint |
| RNAbpFlow | Bhattacharya-Lab/RNAbpFlow | 3D flow prediction / distance evidence (`ensemble_predictor`) | `RNA3DB.ckpt` (trained on RNA3DB/bpRNA), archived separately |
| structRFM | inspired by Zhai et al., *Nat. Commun.* 2024 | optional multi-task heads (`multitask_heads`) | external checkpoint |
| DivideFold | external RNA folding predictor — Omnes L, Angel E, Tahi F. *DivideFold+: an AI-based tool for RNA secondary structure prediction...* J Mol Biol. 2026;438(18):169865 (doi:10.1016/j.jmb.2026.169865); team checkout named `DivideFold-main` | SS evidence for the Level 0 consensus (`ss_divide`, CPU subprocess) | external checkout (`DivideFold-main`; `TF_DIVIDEFOLD_ROOT`) |

The folding/refinement core of this repository is **physics-based
(zero-training)**: CG MD, REST2×T-REMD, metadynamics and Amber14-OL3 — no
learned model. No fine-tuning data is committed. Development used an AI coding
assistant; see [.claude/RESPONSIBLE_AI_USE.md](.claude/RESPONSIBLE_AI_USE.md)
for the team's responsibility policy.

**Evaluation & limitations.** No experimental (wet-lab) validation has been
performed yet for the demo construct. A 2OIU force-field integrity test
(X-ray structure → Level-2 relaxation, 17 min CPU, final RMSD 1.83 Å) is
completed and shows the force field does not distort known structures;
independent validation is otherwise pending (2OIU from-sequence recovery and
replica-exchange acceptance checks — dated status in
[docs/NOTES.md](docs/NOTES.md), which also records known dead ends; please
read it before extending the code).


## Implementation notes (why it works)

Short answers to the questions a computational reviewer usually asks. Each
entry points to the module that implements it.

1. **Batched replica exchange on one tensor.** In `torch_cgsim.py`
   (`BatchedREMD` / `BatchedREMD2D`), all replicas live in a single
   `(B, N, 3)` tensor and a Metropolis swap is a **tensor-index permutation**
   — no coordinate copies, no process pipes. The classical CPU alternative is
   one OpenMM process per replica (~200 ms/step) with pipe-based swaps.
   Energy and forces are computed by the same explicit-force function, so the
   swap criterion and the dynamics are consistent by construction.
2. **Asynchronous CPU-side force injection.** Statistical-potential forces
   (TriRNASP, optional preview feature) are evaluated on a CPU process pool
   (`_refresh_cpu_forces_async`) and injected into the GPU force cache;
   when replicas are exchanged the force cache is permuted together with the
   coordinates (`_swap_tri_force_cache`), so replica identity stays
   consistent across the swap.
3. **No-autograd explicit forces.** Forces follow the TorchMD recipe:
   analytic, autograd-free force evaluation with a cell-list neighbor table
   (O(N) memory instead of O(N²)). Because each force function returns energy
   and forces together, the two can never drift apart, and `_require_finite`
   guards every stage. An autograd implementation is kept alongside
   (`cg_forces_autograd`) for cross-checking.
4. **Analytic CV gradients for metadynamics.** The GPU metadynamics sampler
   (`metadynamics_gpu.py`) computes CVs (e.g. radius of gyration) with
   hand-written analytic gradients and deposits well-tempered hills as batched
   tensor updates — no autograd on the sampling loop.
5. **RL-based sampling scheduling — preview feature.** A reinforcement-
   learning controller can suggest sampling budgets and pair weights
   (nstep / pair_weights). It starts from a heuristic policy (ViennaRNA run at
   scale on long circBase sequences as a weak-but-available prior) and is
   refined by online learning; it is deliberately conservative and never
   overrides the physics. Like the TriRNASP statistical potential, it is a
   documented preview idea rather than a headline claim — see NOTES.md.
7. **Length scaling of the sampling budget.** For very long chains the total
   budget is halved at L > 500 and halved again at L > 1,000
   (`isrnaclong.py`, length scaling). Rationale: sampling bottlenecks are
   local — segments are fixed at 200 nt and each segment relaxes
   independently of total length, while long-range pairing is handled by
   constraints plus the temperature ladder — so the total need grows slower
   than linearly. This is a working heuristic: it is guarded by early
   stopping and energy gates, and a formal scaling benchmark is on the
   roadmap.
8. **Chunk fusion is confidence-weighted, with bidirectional context.**
   Segment predictions (200 nt, 30 nt overlap, raised to ease boundary
   effects) are fused by confidence-weighted assembly (Kabsch alignment,
   `segmented_vfold3d.py`), then refined with bidirectional context
   (NLP-style: each boundary is corrected by neighbors on both sides, ±50 nt
   with linear decay, `_refine_with_context`) before the final global
   relaxation.
9. **NCM candidates are filtered by a thermodynamic cost.** Non-canonical
   pair candidates survive only if their local folding cost is affordable:
   `ncm_thermo_filter.py` refolds the local window with and without the
   candidate and suppresses it when ΔΔG > 2.0 kcal/mol, down-weights to 0.5
   between 1.0 and 2.0 kcal/mol. The thresholds are empirical, but the
   quantity being thresholded is a physical free-energy cost.

## Data formats & synthetic-biology standards

Input: FASTA-like plain sequence (`sequence.txt`, `T`→`U` handled);
secondary-structure strings (dot-bracket). Output: PDB (all-atom) + JSON
metrics. This is a **structure-prediction** tool, so it does not emit SBOL /
genetic-design constructs; where a circRNA is later cloned for wet-lab work the
PDB is the geometry reference and the sequence can be re-exported to FASTA /
GenBank as needed.

## Testing & CI

- `tests/test_smoke.py` — compiles every Python file and imports the
  `torusfold.scheme2` package (needs only numpy).
- `.gitlab-ci.yml` — runs the smoke suite on every push to keep `main` green.
- Run locally: `pip install -e . && python -m pytest -q tests`.

## License

[Apache-2.0](LICENSE). External components retain their own licenses
(RNAbpFlow, TriRNASP, ViennaRNA, Mol*, RhoFold+, trRosettaRNA2, OpenMM).
