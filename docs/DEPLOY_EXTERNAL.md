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
export ISRNACIRC_ROOT="C:/path/to/IsRNAcirc/standalone"          # optional; $BIN_DIR/.. is assumed
```

The coefficient directory is `Data/data/IsRNA2/` in the distributed package — *not*
`Data/`. Pointing `CG_TO_ALLATOM_COEFF` at `Data/` looks right and fails at run time
with `Wrong coeffDIR`, because the five `AA_*.dat` templates are one level below it.
`python tools/configure_deps.py discover` finds them wherever they are and prints which
of isRNAcirc's sub-steps work; `python tools/configure_deps.py write` records the result
in `.env.local` / `activate_deps.bat`. `IsRNAcirc.exe` (the CPU MD refinement binary) is
**not** needed: it is absent from the distributed `bin/` and nothing in this repository
invokes it — Level 2 refinement runs through the Torch field or `openmm_gpu_refiner.py`.
An earlier version of `configure_deps.py` required it and reported a complete install as
`BLOCKING`.

> **Upstream:** Jiang H, Xu Y, Tong Y, Zhang D, Zhou R. *IsRNAcirc: 3D structure
> prediction of circular RNAs based on coarse-grained molecular dynamics simulation.*
> PLoS Comput. Biol. 2024;20(11):e1012293. doi:10.1371/journal.pcbi.1012293.
> PMC11542809, PMID 39466881. Code: `github.com/DongZhangRNA/IsRNAcirc`.
>
> The same DOI is cited in `constraint_solver.py:260` as "Jiang et al., PLOS Comp
> Biol 2024". An earlier version of this file attributed IsRNAcirc to Xiao M, Sun Y,
> Li Y, et al., J. Chem. Theory Comput. 2023; a search finds no such paper, and the
> DOI above is the method paper the authors link their own code from.
> Standalone binaries for several versions are in circulation — use the release
> described in the paper (Windows-only exe + DLLs, ASCII path required).

### PyRosetta — conditional full-atom refinement (Linux/WSL)

Optional; used by `pyrosetta_refine.py`, which Level 2.6 calls as a Python import
inside WSL.

> **PyRosetta is licensed software. Obtain a licence from
> <https://www.pyrosetta.org> before installing it.**
>
> It is free of charge for academic and other non-commercial use, but that still
> requires a licence key that you request yourself. Commercial use requires a paid
> licence. It is not on PyPI and not on conda-forge, so nothing in this repository
> can download or install it, and this repository neither bundles it nor ships a
> licence for it.
>
> Running an unlicensed copy is not a configuration this project supports. If you
> are not covered by a licence, leave PyRosetta out: Level 2.6 is **skipped, not
> failed**, without it, and the rest of the pipeline is unaffected.

Check whether it is available, and see the licence note, with either:

```bash
python tools/configure_deps.py discover     # the PyRosetta row carries the note
python tools/install_deps.py --skip-downloads --skip-git
```

### structRFM — multi-task DL heads

`multitask_heads.py` loads a pretrained structRFM checkpoint for the
SS / pair / BSJ / clash heads. It is off by default (`use_structrfm=False`).

```bash
export TF_STRUCTRFM_MODEL=/path/to/structRFM      # a HuggingFace checkpoint directory
```

The directory must contain `config.json`, `tokenizer.json` and either
`model.safetensors` or `pytorch_model.bin`. `tools/configure_deps.py` checks exactly
that and reports the weights filename and size, because a directory with the
checkpoint's *name* and none of its contents is what a killed download leaves behind.

> **Upstream:** Zhu H, Li R, Zhang F, Tang F, Ye T, Li X, Gu Y, Xiong P, Zhou S K.
> *A fully-open structure-guided RNA foundation model for robust structural and
> functional inference.* bioRxiv 2025.08.06.668731, doi:10.1101/2025.08.06.668731.
> Code `github.com/heqin-zhu/structRFM`, weights `huggingface.co/heqin-zhu/structRFM`
> (MIT, `model.safetensors`, 344 MB), also released as
> `github.com/heqin-zhu/structRFM/releases/latest/download/structRFM_checkpoint.tar.gz`
> (319 MB). This section previously cited "Zhai et al., *Nature Communications* 2024";
> there is no such paper, and the model card names Zhu as the author of both the
> repository and the checkpoint.
>
> **Getting it, on a slow link.** `python tools/fetch_structrfm.py` downloads the
> checkpoint into `TorusFold-tools/structRFM` beside the repository — the place
> `configure_deps.py` searches first. It exists because `huggingface_hub` did not work
> here: the installed version has moved to Xet, the `HF_HUB_ENABLE_HF_TRANSFER` route is
> deprecated, and Xet against a mirror does not move. Ranged GETs through
> `hf-mirror.com` do, and in parallel they are much faster — 69 KB/s on one connection
> versus 369 KB/s on eight, measured on the same 344 MB file, which is the difference
> between 84 and 16 minutes. The assembled file is checked against the Hub's published
> LFS sha256 and set aside as `.BAD` on mismatch.
>
> **ModelScope does not have it.** Checked 2026-10-14 against
> `PUT modelscope.cn/api/v1/dolphin/models`: `structRFM`, `struct-rfm`, `heqin`,
> `BPfold` and `trRosettaRNA` all return `TotalCount: 0`, while `RNAErnie` returns 3 —
> so the query works and the answer is genuinely no. On a network where the Hub is
> slow, `HF_ENDPOINT=https://hf-mirror.com` is the route that works.
>
> **A torch built without distributed disables this whole path.** structRFM is loaded
> through `transformers`, and transformers 5.x imports torch DTensor while building
> `GenerationMixin`, so under a torch where `hasattr(torch._C, "_c10d_init")` is False
> (the ROCm build used here is one) `import transformers` succeeds and
> `from transformers import AutoModel` raises
> `ModuleNotFoundError: No module named 'torch._C._distributed_c10d'`. The checkpoint
> can be present, complete, and still unusable. `tools/configure_deps.py` reports this
> as a `BROKEN transformers` row with the failing statement, not as `OK`.

### RNAbpFlow — 3D flow prediction (ensemble member)

Upstream is a hydra-free script with no command-line arguments: `inference.py` hardcodes
`CONFIG_FILE_PATH = "configs/inference.yaml"` and drives everything from that file plus
the working directory. This repository used to require `inference_rocm.py`, a team-held
adaptation that adds a CLI; that file is not in any upstream release and there is no URL
for it, so a checkout upstream considers complete was reported as a missing tool.

Two routes, tried in this order by `ensemble_predictor._run_rnabpflow`:

1. `$RNABPFLOW_ROOT/inference_rocm.py` — used unchanged when present.
2. `tools/rnabpflow_runner.py` — this repository's adapter. It writes
   `configs/inference.yaml` under the caller's scratch directory, puts the checkout on
   `sys.path`, installs the torch shims below, and calls `inference.run_inference()`.

```bash
export RNABPFLOW_ROOT=/path/to/RNAbpFlow
export RNABPFLOW_PYTHON=/path/to/python    # must be the GPU interpreter
```

Verified here at 71 nt / 1 sample / 50 timesteps: 38 s on the ROCm GPU, producing
`Predictions/seq/Sample_0.pdb` and a `(71, 3)` coordinate array through
`_run_rnabpflow` itself. The 203 MB checkpoint is `checkpoint/RNA3DB.ckpt`, from Zenodo
record 18305861.

### `torusfold.torch_compat` — why the two optional predictors need a shim here

The GPU torch on the reference machine is a ROCm build compiled **without distributed
support**:

    torch._C._has_distributed()      -> False
    torch.distributed.is_available() -> False
    torch.cuda.get_device_name(0)    -> AMD Radeon(TM) 8060S Graphics

The GPU is fine; distributed is absent. Two optional predictors import into that gap
incidentally — neither performs a collective:

* **structRFM**: `transformers.generation.utils` → `continuous_batching` →
  `from torch.distributed.tensor.device_mesh import DeviceMesh` → `torch/distributed/
  tensor/__init__.py` → … → `from torch._C._distributed_c10d import (...)`, which does
  not exist. `import transformers` succeeds; `from transformers import AutoModel` raises
  `ModuleNotFoundError: No module named 'torch._C._distributed_c10d'`.
* **RNAbpFlow**: its model is a `pytorch_lightning` module, and `pytorch_lightning` →
  `torchmetrics` → `torchmetrics.functional.text.bert` → `from transformers import
  AutoModel` hits the same wall, so the model class cannot be imported at all.

`src/torusfold/torch_compat.py` installs two stand-ins, each only when the real thing is
truly absent, and each refusing rather than guessing:

| shim | why it is safe | how it refuses |
|---|---|---|
| `torch.distributed.tensor.device_mesh` | `continuous_batching/distributed.py` uses `DeviceMesh` for one annotation plus code guarded by `dist.is_available()`, which is False | constructing the placeholder raises `RuntimeError` |
| `torch_scatter` | RNAbpFlow calls exactly two functions, both `dim=0` with a 1-D index, which `Tensor.index_add_` computes exactly (error 0.0 against an explicit loop) | any other `dim` or `reduce` raises `NotImplementedError` |

No package in the interpreter is modified, and the GPU stays in use. `configure_deps.py`
reports this in two places at once, deliberately: the plain import really does fail
(which matters to anything else on the machine, ComfyUI's nodes included), and
`+ reachable through torusfold.torch_compat` records that this repository can still get
there.

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
