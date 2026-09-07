# TorusFold-Hybrid

_Team JLU-FBH · iGEM 2026 Software & AI_

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

Physics-based **circRNA 3D structure prediction**. A multi-predictor ensemble
(RhoFold+ · trRosettaRNA2 · RNAbpFlow) feeds an RL-guided coarse-grained folding
engine that is relaxed with OpenMM molecular dynamics, replica-exchange and
metadynamics, then reconstructed to all atoms and refined under the Amber14-OL3
force field — producing experimentally plausible models for long (1000+ nt)
circular RNA.

> This repository is the official software deliverable of Team JLU-FBH
> (iGEM 2026, Software & AI village). All source code lives on `main`; a
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
> web frontend. The three sequence predictors (RhoFold+, trRosettaRNA2,
> RNAbpFlow) are *external* tools invoked as subprocesses — point to your own
> checkouts via the environment variables below. Two legacy aggregator modules
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
pseudo-MSA fallback) are configured at the top of `run_2013nt.py`.

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

## AI / model disclosure

This software **calls machine-learning RNA structure predictors as external
tools**; it does **not** train them, and it ships no trained weights:

| Model | Source | Used for | Weight/data provenance |
|---|---|---|---|
| RhoFold+ | Wang et al., *Nat. Methods* 2024 | per-chunk 3D prediction (`rhofold_wrapper`) | external checkpoint |
| trRosettaRNA2 | Li et al., *Nat. Commun.* 2021;12:5934 | per-chunk 3D prediction (`trrna2_wrapper`) | external checkpoint |
| RNAbpFlow | Bhattacharya-Lab/RNAbpFlow | 3D flow prediction / distance evidence (`ensemble_predictor`) | `RNA3DB.ckpt` (trained on RNA3DB/bpRNA), archived separately |
| structRFM | inspired by Zhai et al., *Nat. Commun.* 2024 | optional multi-task heads (`multitask_heads`) | external checkpoint |

The folding/refinement core of this repository is **physics-based
(zero-training)**: CG MD, REST2×T-REMD, metadynamics and Amber14-OL3 — no
learned model. No fine-tuning data is committed. Development used an AI coding
assistant; see [.claude/RESPONSIBLE_AI_USE.md](.claude/RESPONSIBLE_AI_USE.md)
for the team's responsibility policy.

**Evaluation & limitations.** No experimental (wet-lab) validation has been
performed yet for the demo construct. Independent validation is pending:
2OIU crystal-structure recovery and replica-exchange acceptance checks. Known
dead ends and open problems are recorded in
[docs/NOTES.md](docs/NOTES.md) — please read it before extending the code.

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
