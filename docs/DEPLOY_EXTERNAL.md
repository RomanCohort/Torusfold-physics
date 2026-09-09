# Deploying external tools

TorusFold-Hybrid is an *orchestrator*: the physics/refinement core lives in
this repository, but the sequence predictors and statistical potentials it
calls are separate projects. This page documents how to obtain and configure
each one. No machine-specific path is hard-coded anywhere — everything is set
through environment variables (also listed in the [README](../README.md)).

Quick map:

| Tool | Role in pipeline | Needed by | Required? |
|---|---|---|---|
| ViennaRNA | secondary-structure folding (`RNA` module) | Level 0 `multisource_ss` / fallback | **yes** (demo) |
| OpenMM | MD relaxation, REMD, MetaD, PPR | Level 2+ physics core | **yes** |
| PyTorch | RL MCTS, GPU CG MD (`torch_cgsim`) | RL close / GPU path | yes (GPU) |
| RhoFold+ | per-chunk 3D predictor | `rhofold_wrapper` (Level 1) | for ensemble |
| trRosettaRNA2 | per-chunk 3D predictor | `trrna2_wrapper` (Level 1) | for ensemble |
| RNAbpFlow | 3D flow predictor (distance/BSJ evidence) | `ensemble_predictor` (Level 1) | for ensemble |
| isRNAcirc | CG→all-atom + force-field binaries (Windows) | `isrnacirc_wrapper` | optional |
| TriRNASP | three-body statistical potential | `trirnasp_*` energy terms | optional (off by default) |
| PyRosetta | conditional full-atom refinement (Linux/WSL) | `pyrosetta_refine` | optional |
| structRFM | multi-task DL heads checkpoint | `multitask_heads` | optional |
| Infernal/cmsearch | Rfam MSA for chunk prediction | pseudo-MSA/real-MSA | optional |

## Core Python dependencies

```bash
# OpenMM >= 8.0  (MD / replica-exchange engine)
conda install -c conda-forge openmm        # or: pip install openmm

# ViennaRNA >= 2.6  (provides the `RNA` module)
conda install -c conda-forge viennarna     # or: pip install ".[ss]"

# PyTorch >= 2.0  (RL + GPU CG path)
pip install torch                           # or: pip install ".[gpu]"
```

Install the package itself:

```bash
pip install -e .
```

## Ensemble predictors (Level 1)

These are invoked as subprocesses by
`src/torusfold/scheme2/{ensemble_predictor,rhofold_wrapper,trrna2_wrapper}.py`.
Point each wrapper at your checkout with the environment variable shown.

### RNAbpFlow — `RNABPFLOW_ROOT`

- Source: <https://github.com/Bhattacharya-Lab/RNAbpFlow>
  (development checkout pinned at `e8b1c07`).
- The wrapper expects the model weights at
  `<RNABPFLOW_ROOT>/checkpoint/RNA3DB.ckpt` and runs
  `<RNABPFLOW_ROOT>/inference_rocm.py` (`RNABPFLOW_ROOT` unset → clear error).
- Input/output directory layout under a temp dir is handled by
  `ensemble_predictor.py` (FASTA + map + distance maps → `Predictions/seq/Sample_0.pdb`).

```bash
export RNABPFLOW_ROOT=/path/to/RNAbpFlow      # checkout with checkpoint/RNA3DB.ckpt
export RNABPFLOW_PYTHON=python                # interpreter that can run RNAbpFlow
```

> TODO(iGEM): archive `RNA3DB.ckpt` on Zenodo and link it here (DOI).

### RhoFold+ — `RHOFOLD_ROOT`

`rhofold_wrapper.py` loads the `rhofold` package and the pretrained checkpoint
`<RHOFOLD_ROOT>/pretrained/rhofold_pretrained_params.pt`.

```bash
export RHOFOLD_ROOT=/path/to/RhoFold         # package importable + pretrained/ params
```

> **Upstream:** RhoFold+ is under active development and several releases
> coexist — consult the original paper for the canonical code release and
> pretrained weights: Wang W, Lv G, Tang W, et al. *RhoFold+: accurate RNA 3D
> structure prediction via deep learning.* Nat. Methods, 2024. Point
> `RHOFOLD_ROOT` at the matching checkout.

### trRosettaRNA2 — `TRRNA2_RUNNER`

`trrna2_wrapper.py` runs a small *runner script* (in a CPU-only environment) via
subprocess. Point the runner at your trRosettaRNA2 install:

```bash
export TRRNA2_RUNNER=/path/to/trrna2_runner.py
```

> **Upstream:** trRosettaRNA2 is actively maintained with many versions —
> consult the original paper for the canonical release: Li S, et al.
> *trRosettaRNA: automated prediction of RNA 3D structure with transformer
> network.* Nat. Commun. 2021;12:5934. Provide a runner script that invokes
> the matching trRosettaRNA2 install.

## DivideFold (optional Level-0 SS evidence)

DivideFold is an external RNA folding predictor used as third-party evidence
in the Level-0 secondary-structure consensus (`ss_divide` in
`scheme2/isrnaclong.py`, invoked as a pure-CPU subprocess with GPU devices
hidden). It is fully optional — if it fails, the pipeline falls back to the
ViennaRNA-based two-source vote.

- Source: DivideFold+ — Omnes L, Angel E, Tahi F. *DivideFold+: an
  AI-based tool for RNA secondary structure prediction with subdomains
  identification and visualization and data augmentation*. J Mol Biol.
  2026;438(18):169865. doi:10.1016/j.jmb.2026.169865. The team's checkout is
  named `DivideFold-main` (same lineage).
- Configuration (environment variables, no hard-coded paths):
  | Variable | Purpose |
  |---|---|
  | `TF_DIVIDEFOLD_ROOT` | checkout dir (default: `DivideFold-main/` next to this repo) |
  | `TF_DIVIDEFOLD_PYTHON` | interpreter for the SS subprocess (default: current interpreter) |
  | `TF_DIVIDEFOLD_RUNNER` | runner script (default: `scripts/_dd_runner.py` in this repo — note: the runner is not yet committed; until it is, set this to your local copy or the stage is skipped with a logged warning) |

## Statistical potentials & refiners (optional)

- Source: <https://github.com/Tan-group/TriRNASP>
  (development checkout pinned at `69e999a`).
- Wrappers in this repo: `trirnasp_openmm.py`, `trirnasp_torch.py`,
  `trirnasp_wrapper.py`. Off by default (`use_trirnasp=False`).

### isRNAcirc — CG→all-atom binaries (Windows)

`isrnacirc_wrapper.py` shells out to `CG_to_allatom.exe` (ASCII path only —
the exe cannot load DLLs from a path containing non-ASCII characters):

```bash
export ISRNACIRC_BIN_DIR="C:/path/to/IsRNAcirc/standalone/bin"   # contains CG_to_allatom.exe
export CG_TO_ALLATOM_COEFF="C:/path/to/IsRNA2/coeff"             # coefficient dir
```

> **Upstream:** Xiao M, Sun Y, Li Y, et al. *IsRNAcirc: prediction of circular
> RNA 3D structures via coarse-grained molecular dynamics simulations.*
> J. Chem. Theory Comput. 2023. Standalone binaries for several versions are in
> circulation — use the release described in the paper (Windows-only exe + DLLs,
> ASCII path required).

### PyRosetta — conditional full-atom refinement (Linux/WSL)

Optional; used by `pyrosetta_refine.py`. Install per your PyRosetta license
(<https://www.pyrosetta.org>).

### structRFM — multi-task DL heads

`multitask_heads.py` loads a pretrained structRFM checkpoint for the
SS / pair / BSJ / clash heads:

```bash
export TF_STRUCTRFM_MODEL=/path/to/structrfm/model/dir
```

> **Upstream:** structRFM-style multi-task heads (see
> `src/torusfold/scheme2/multitask_heads.py`), inspired by Zhai et al.,
> *Nature Communications* 2024. The pretrained checkpoint is a moving target —
> point `TF_STRUCTRFM_MODEL` at a compatible structRFM release.

### Infernal / Rfam (optional MSA)

`run_2013nt.py` accepts `RFAM_CM` for cmsearch-based covariance models:

```bash
export RFAM_CM=/path/to/Rfam.cm
```

## Verifying a deployment

```bash
# environment sanity (imports resolve)
python -c "import RNA, openmm; print('fold+MD OK')"

# wrappers fail loudly when a tool is not configured (clear FileNotFoundError):
python -c "from torusfold.scheme2 import rhofold_wrapper, ensemble_predictor, trrna2_wrapper, isrnacirc_wrapper"
```

Then run the end-to-end demo (see [README](../README.md)):

```bash
# put the sequence in sequence.txt, then:
python run_2013nt.py
```
