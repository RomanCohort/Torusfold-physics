# -*- coding: utf-8 -*-
"""Install the external tools TorusFold delegates to, as far as is automatable.

This is the engine behind the "Set up environment" button. It never claims more
than it can do: every component is classified by what is actually possible from
the machine it runs on, and anything it cannot fetch is reported with the exact
manual step instead of a generic failure.

What is automatable, and from where — measured, not assumed:

  pip packages           pypi (a mirror is fine)      freesasa, gemmi, ...
  RhoFold+ source        github.com/ml4bio/RhoFold
  RNAbpFlow source       github.com/Bhattacharya-Lab/RNAbpFlow
  RNAbpFlow checkpoint   zenodo.org record 18305861   532 MB, direct link
  RhoFold+ checkpoint    huggingface.co                485 MB, often blocked
  trRosettaRNA2          no upstream URL in this repository — manual
  DivideFold             no upstream URL in this repository — manual
  isRNAcirc binaries     distributed as a zip by the authors — manual

The distinction matters because huggingface.co is unreachable from some
networks while zenodo.org is reachable from the same machine, so "download the
checkpoint" is not one task: one of them works here and the other does not.

Every step prints one JSON line per event so a caller can stream progress:

    {"event": "step",     "id": "...", "label": "...", "status": "start|ok|skip|fail", ...}
    {"event": "progress", "id": "...", "done": 1234, "total": 5678, "pct": 21.7}
    {"event": "summary",  "installed": [...], "manual": [...]}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Where downloaded tools are placed. Kept out of the repository itself so a
# checkout stays source-only, which is what the project's README promises.
TOOLS_DIR = os.environ.get("TORUSFOLD_TOOLS_DIR") or os.path.join(
    os.path.dirname(REPO), "TorusFold-tools")

PYPI_MIRROR = os.environ.get("TORUSFOLD_PIP_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple")

# ── Where model weights come from ────────────────────────────────────────────
#
# huggingface.co is unreachable from some networks — measured on the machine this
# was written on: huggingface.co and huggingface.co.cn both fail to connect,
# while hf-mirror.com serves the same repository layout and returned
# `bytes 0-1023/508317553` for the RhoFold checkpoint, i.e. the identical file.
#
# A mirror is a different host, not a different model: hf-mirror.com proxies the
# HuggingFace repository tree, so the same `resolve/main/<file>` path resolves.
# That is why one optional base-URL swap is enough, and why the checksum the
# library records can still be checked against the official hash.
MODEL_SOURCES = [
    {
        "id": "huggingface",
        "label": "HuggingFace (official)",
        "base": "https://huggingface.co",
        "note": "the upstream host; unreachable from some networks",
    },
    {
        "id": "hf-mirror",
        "label": "hf-mirror.com (mirror)",
        "base": "https://hf-mirror.com",
        "note": "proxies the same HuggingFace repository tree",
    },
    {
        "id": "modelscope",
        "label": "ModelScope",
        "base": "https://www.modelscope.cn/models",
        "note": "checked: the RhoFold weights are not hosted there, so this source "
                "works only for models that are",
    },
]
DEFAULT_SOURCE = os.environ.get("TORUSFOLD_MODEL_SOURCE", "auto")
# The second element is the path after the base, so each checkpoint can name its
# own HuggingFace repository and file.
MODEL_PATHS = {
    "rhofold_ckpt": ("cuhkaih/rhofold", "rhofold_pretrained_params.pt"),
}

## ── The Python environment itself ────────────────────────────────────────────
#
# The pipeline is a set of imports before it is a set of tools: without numpy,
# scipy, ViennaRNA, torch and OpenMM there is nothing to start. Those are the
# same kind of dependency as the external tools and belong in the same setup
# pass, so this file handles both.
#
# Names are per-ecosystem: conda and pip do not always agree (ViennaRNA is `RNA`
# to Python, `viennarna` to conda, and `ViennaRNA` to pip).
PIPELINE_PACKAGES = [
    # (import name, conda name, pip name, required, why)
    ("numpy", "numpy", "numpy", True, "core numerics"),
    ("scipy", "scipy", "scipy", True, "KD-tree neighbour search"),
    ("RNA", "viennarna", "ViennaRNA", True, "secondary structure + partition function"),
    ("torch", "pytorch", "torch", True, "RL and the GPU coarse-grained path"),
    ("openmm", "openmm", "openmm", True, "MD, replica exchange, minimisation"),
    ("matplotlib", "matplotlib", "matplotlib", False, "RhoFold+ imports it by name"),
    ("ml_collections", "ml_collections", "ml_collections", False, "RhoFold+ config"),
    ("Bio", "biopython", "biopython", False, "RhoFold+ MSA handling"),
    ("tree", "dm-tree", "dm-tree", False, "RhoFold+ tree utilities"),
    ("einops", "einops", "einops", False, "RhoFold+ tensor rearrangement"),
    ("gemmi", "gemmi", "gemmi", False, "CIF/PDB ingest"),
    ("freesasa", "freesasa", "freesasa", False, "per-residue SASA"),
    ("pandas", "pandas", "pandas", False, "analysis scripts"),
]

# Where interpreters are looked for, best guess first. A GPU torch build is
# preferred because RNAbpFlow is launched with `--device cuda` and will not run
# without one — so this is a capability question, not a speed preference.
CONDA_ROOTS = [
    os.environ.get("CONDA_ROOT") or "",
    r"C:\ana",
    r"C:\ProgramData\anaconda3",
    r"C:\ProgramData\miniconda3",
    os.path.expanduser(r"~\anaconda3"),
    os.path.expanduser(r"~\miniconda3"),
    r"C:\ProgramData\miniforge3",
]
CONDA_ENV_NAMES = [
    os.environ.get("TORUSFOLD_CONDA_ENV") or "",
    "comfyui", "circrna3d", "bio", "base",
]
ENV_PROBE = (
    "import importlib.util as u, json\n"
    "def safe(m):\n"
    "    try: return u.find_spec(m)\n"
    "    except Exception: return None\n"
    "names = %r\n"
    "missing = [m for m in names if safe(m) is None]\n"
    "gpu = None\n"
    "try:\n"
    "    import torch\n"
    "    gpu = {'torch': getattr(torch, '__version__', '?'),\n"
    "           'hip': getattr(getattr(torch, 'version', None), 'hip', None),\n"
    "           'cuda': bool(torch.cuda.is_available())}\n"
    "except Exception as exc:\n"
    "    gpu = {'error': repr(exc)[:120]}\n"
    "print(json.dumps({'missing': missing, 'torch': gpu}))\n"
)


def _conda_exe() -> Optional[str]:
    """The conda launcher, if this machine has one. Off PATH is normal: the
    standard installer does not add it, so the well-known locations are checked."""
    for root in CONDA_ROOTS:
        if not root or not os.path.isdir(root):
            continue
        for rel in (os.path.join("Scripts", "conda.exe"),
                    os.path.join("condabin", "conda.bat"),
                    os.path.join("Scripts", "conda.bat")):
            candidate = os.path.join(root, rel)
            if os.path.isfile(candidate):
                return candidate
    found = shutil.which("conda") or shutil.which("conda.bat")
    return found


def _conda_env_pythons() -> List[str]:
    """Every interpreter conda knows about, newest env first is not knowable, so
    the conventional names are enumerated and the rest discovered by globbing."""
    exe = _conda_exe()
    paths: List[str] = []
    roots = [r for r in CONDA_ROOTS if r and os.path.isdir(r)]
    for root in roots:
        envs_dir = os.path.join(root, "envs")
        for name in CONDA_ENV_NAMES:
            if not name:
                continue
            cand = root if name == "base" else os.path.join(envs_dir, name)
            py = os.path.join(cand, "python.exe")
            if os.path.isfile(py) and py not in paths:
                paths.append(py)
        if os.path.isdir(envs_dir):
            for name in sorted(os.listdir(envs_dir)):
                py = os.path.join(envs_dir, name, "python.exe")
                if os.path.isfile(py) and py not in paths:
                    paths.append(py)
    if exe:
        pass  # enumeration above already covers the standard layouts
    return paths


def probe_interpreter(python: str, timeout: float = 180.0) -> Dict:
    """What this interpreter can and cannot import, and whether its torch has a GPU."""
    entry = {"path": python, "usable": False, "missing": None, "torch": None}
    names = [p[0] for p in PIPELINE_PACKAGES]
    try:
        p = subprocess.run([python, "-c", ENV_PROBE % (names,)], capture_output=True,
                           text=True, timeout=timeout)
        if p.returncode != 0 or not p.stdout.strip():
            entry["error"] = (p.stderr or "no output").strip().splitlines()[-1:][0][:160]
            return entry
        data = json.loads(p.stdout.strip().splitlines()[-1])
        entry["missing"] = data["missing"]
        entry["torch"] = data["torch"]
        required = [pkg[0] for pkg in PIPELINE_PACKAGES if pkg[3]]
        entry["missing_required"] = [m for m in data["missing"] if m in required]
        entry["usable"] = not entry["missing_required"]
        entry["complete"] = not data["missing"]
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        entry["error"] = str(exc)[:160]
    return entry


def survey_environments(extra: Optional[List[str]] = None) -> Dict:
    """Every interpreter worth considering, and which one to use.

    The current interpreter is probed first and wins ties, because being run by an
    interpreter is weak evidence that it is the intended one; a GPU-capable one
    wins outright, because RNAbpFlow's `--device cuda` needs it.
    """
    seen: List[str] = []
    for py in [sys.executable] + list(extra or []) + _conda_env_pythons():
        if py and py not in seen:
            seen.append(py)
    probes = [probe_interpreter(py) for py in seen]
    usable = [p for p in probes if p.get("usable")]
    gpu = [p for p in usable if (p.get("torch") or {}).get("cuda")]
    chosen = (gpu or usable or [None])[0]
    return {"interpreters": probes, "chosen": chosen,
            "conda": _conda_exe(), "count": len(probes)}


def _conda_env_for(python: str) -> Optional[str]:
    """The conda environment directory this interpreter belongs to, or None.

    A conda distribution on PATH is not evidence that a given interpreter is a
    conda environment. `conda install -p` was previously aimed at whatever
    directory the interpreter sat in, so on a machine that has conda AND a normal
    python.org installation, choosing the latter made conda target
    `C:\\Python312` — not an environment, which either errors out or creates one
    there. The interpreter has to actually live inside a conda prefix, or conda
    is the wrong installer for it.
    """
    if not python:
        return None
    target = os.path.normcase(os.path.abspath(python))
    for root in CONDA_ROOTS:
        if not root:
            continue
        base = os.path.normcase(os.path.abspath(root))
        # base environment
        if target == os.path.join(base, "python.exe"):
            return root
        # a named environment directly under <root>\envs
        envs = os.path.join(base, "envs")
        if target.startswith(os.path.normcase(os.path.abspath(envs)) + os.sep):
            rest = target[len(os.path.normcase(os.path.abspath(envs))) + 1:]
            name = rest.split(os.sep)[0]
            if name:
                return os.path.join(envs, name)
    return None


def install_into_environment(python: str, missing: List[str], use_conda: bool = True) -> Dict:
    """Install the named imports into one interpreter.

    conda first for the scientific stack, because mixing a pip torch into a conda
    environment is how you get two BLAS libraries and a segfault; pip as the
    fallback because a couple of these packages are not on the conda channels.

    conda is only used when the target interpreter is itself a conda environment.
    For a plain Python installation there is nothing for conda to target and pip
    is the only correct installer — which matters because ViennaRNA and dm-tree
    have no conda-forge Windows build at all, so pip does the work either way.
    """
    by_import = {p[0]: p for p in PIPELINE_PACKAGES}
    result = {"installed": [], "failed": []}
    install_with_conda = _conda_exe() if use_conda else None
    conda_env = _conda_env_for(python) if install_with_conda else None
    for name in missing:
        spec = by_import.get(name)
        if not spec:
            continue
        _, conda_name, pip_name, required, why = spec
        step_start("pkg:" + name, "%s (%s)" % (pip_name, why))
        cmd = None
        if install_with_conda and conda_env:
            cmd = [install_with_conda, "install", "-y", "-p", conda_env,
                   "-c", "conda-forge", conda_name]
        try:
            if cmd:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if p.returncode == 0:
                    step_start("pkg:" + name, pip_name, "ok", note="conda")
                    result["installed"].append(name)
                    continue
            p = subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check",
                                "-i", PYPI_MIRROR, pip_name],
                               capture_output=True, text=True, timeout=3600)
        except (subprocess.TimeoutExpired, OSError) as exc:
            step_start("pkg:" + name, pip_name, "fail", note=str(exc)[:160])
            result["failed"].append({"name": name, "why": str(exc)[:160]})
            continue
        if p.returncode == 0:
            step_start("pkg:" + name, pip_name, "ok", note="pip")
            result["installed"].append(name)
        else:
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-2:]
            step_start("pkg:" + name, pip_name, "fail", note=" | ".join(tail)[:200])
            result["failed"].append({"name": name, "why": " | ".join(tail)[:200]})
    return result

PIP_PACKAGES = [
    # (import name, pip name, why it is needed, required)
    ("freesasa", "freesasa", "per-residue SASA for the immune heuristic", False),
    ("gemmi", "gemmi", "CIF/PDB ingest in circrna_library", False),
]

# Source is fetched as a release tarball over HTTPS, not `git clone`.
#
# Measured on the machine this was written on: `git ls-remote` succeeds and
# `git clone` fails with "Recv failure: Connection was reset", while a plain
# HTTPS GET of the same repository's tarball from codeload.github.com completes
# (RhoFold 29.4 MB, RNAbpFlow 0.86 MB). The git wire protocol is blocked in a way
# that ordinary HTTPS is not, so a clone-based installer reports a network
# failure for a tool it could have installed.
GIT_TOOLS = [
    {
        "id": "rhofold",
        "label": "RhoFold+ (source)",
        "tarball": "https://codeload.github.com/ml4bio/RhoFold/tar.gz/refs/heads/main",
        "repo": "https://github.com/ml4bio/RhoFold",
        "dest": "RhoFold",
        "env": "RHOFOLD_ROOT",
        "probe": os.path.join("rhofold", "config.py"),
        "spec": "rhofold",
    },
]

# RNAbpFlow is deliberately NOT here, and the reason is worth stating.
#
# The wrapper runs `inference_rocm.py` with argparse-style flags
# (`--name --checkpoint --device`). The upstream repository ships only
# `inference.py`, which takes no command-line arguments at all, so it cannot
# stand in for it — verified by fetching the upstream tarball and reading both
# files. The working build on the team's machine is 7,017 bytes against the
# upstream 4,628: it is a locally modified variant.
#
# Downloading the upstream source and calling it installed would produce exactly
# the failure this tool exists to prevent — "installed" but the predictor still
# absent from the ensemble. So RNAbpFlow is listed as a manual step instead.
RNABPFLOW_NOTE = (
    "The pipeline runs inference_rocm.py with --name/--checkpoint/--device. The "
    "public repository ships inference.py, which accepts no arguments, so the "
    "team's ROCm-modified checkout is required. Its checkpoints are on Zenodo "
    "(record 18305861)."
)

# Machine-readable install sources, kept only for what they can genuinely
# deliver. RNAbpFlow's checkpoint is still useful even though its source is not
# downloadable: a team checkout placed by hand still needs the 532 MB archive.
CHECKPOINTS = [
    {
        "id": "rnabpflow_ckpt",
        "label": "RNAbpFlow checkpoint (532 MB, from Zenodo)",
        "url": "https://zenodo.org/api/records/18305861/files/model.tar.gz/content",
        "into": "RNAbpFlow/checkpoint",
        "needs_tool": "rnabpflow",
        # The archive holds checkpoint/RNA3DB.ckpt; verified by extraction.
        "expect": os.path.join("checkpoint", "RNA3DB.ckpt"),
        "archive": "tar.gz",
        # This one is looked up in the team checkout rather than under TOOLS_DIR,
        # because the source is not downloadable and a manual placement is the
        # normal case.
        "tool_dirs": ["RNAbpFlow"],
        "also_search": ["RNAbpFlow"],
    },
    {
        "id": "rhofold_ckpt",
        "label": "RhoFold+ checkpoint (485 MB)",
        "url": "https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt",
        "into": "RhoFold/pretrained",
        "filename": "rhofold_pretrained_params.pt",
        "needs_tool": "rhofold",
        "expect": os.path.join("pretrained", "rhofold_pretrained_params.pt"),
        "archive": None,
    },
]

# Things that exist on the user's machine and in a paper, but have no
# machine-readable download source in this repository. Listed so the report can
# say exactly what to do rather than leaving a gap.
MANUAL_TOOLS = [
    {
        "id": "rnabpflow",
        "label": "RNAbpFlow (ROCm build)",
        "why": RNABPFLOW_NOTE,
        "env": "RNABPFLOW_ROOT (+ RNABPFLOW_PYTHON)",
    },
    {
        "id": "isrnacirc",
        "label": "isRNAcirc binaries",
        "why": "Distributed as IsRNAcirc_standalone.zip by the authors (Dong Zhang, "
               "Zhejiang University); the Windows build needs CG_to_allatom.exe plus "
               "its matching DLLs in one ASCII-only directory, and Data/ from the same "
               "release.",
        "env": "ISRNACIRC_BIN_DIR (+ ISRNACIRC_ROOT, CG_TO_ALLATOM_COEFF)",
    },
    {
        "id": "trrna2",
        "label": "trRosettaRNA2",
        "why": "No upstream URL recorded in this repository, and the local copy has no "
               "git remote. Ask the team for the checkout, then point TRRNA2_RUNNER at "
               "its runner script.",
        "env": "TRRNA2_RUNNER",
    },
    {
        "id": "dividefold",
        "label": "DivideFold",
        "why": "No upstream URL recorded in this repository. A majority of the tool is "
               "GitHub-hosted; the team checkout is named DivideFold-main. The runner "
               "script (scripts/_dd_runner.py) ships with this repository.",
        "env": "TF_DIVIDEFOLD_ROOT",
    },
    # ── Scoring functions ────────────────────────────────────────────────────
    # These are listed for a reason beyond "no download source": none of them is
    # wired into the pipeline at all. The report used to omit them entirely, so a
    # reader could reasonably conclude that the three score rows in the interface
    # were backed by something. They are not — see the "not integrated" notes.
    {
        "id": "rsrnasp1",
        "label": "rsRNASP1 (RNA statistical potential)",
        "why": "Source is public at https://github.com/Tan-group/rsRNASP1, but it builds "
               "with a Makefile and gcc for Linux and ships no Windows binary and no pip "
               "package. NOT INTEGRATED: isrnaclong.py writes \"rsrnasp1\": None into the "
               "result, so the interface's rsRNASP1 row can never show a value. Building "
               "it under WSL and calling it would be new work, not configuration.",
        "env": "(none — no wrapper reads an environment variable for this)",
        "integrated": False,
    },
    {
        "id": "dfire",
        "label": "DFIRE (RNA scoring function)",
        "why": "No download URL recorded in this repository. NOT INTEGRATED: nothing in "
               "the pipeline computes it and no result key named DFIRE is emitted, so the "
               "interface's DFIRE row can never show a value.",
        "env": "(none)",
        "integrated": False,
    },
    {
        "id": "3drnascore",
        "label": "3dRNAscore",
        "why": "The authors distribute it as a web server and a REST API "
               "(http://melolab.org/supmat.html) rather than a local binary. NOT "
               "INTEGRATED: no result key named 3drnascore is emitted, so the interface's "
               "3dRNAscore row can never show a value.",
        "env": "(none)",
        "integrated": False,
    },
    {
        "id": "lociparse",
        "label": "lociPARSE (pMoL / pNuL)",
        "why": "Bundled with this repository as src/torusfold/scheme2/loci_quality.py and "
               "implemented in numpy, so it needs no installation and is the one quality "
               "score that does produce a value. TORUSFOLD_LOCIPARSE overrides its "
               "location if the module is vendored elsewhere.",
        "env": "TORUSFOLD_LOCIPARSE (optional)",
        "integrated": True,
    },
]


def emit(**kw) -> None:
    """One JSON line per event, flushed, so a caller can stream it live."""
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def find_local_copy(tool: Dict) -> Optional[str]:
    """Is this tool already installed somewhere on this machine?

    Checked before downloading, because on a machine that already ran the
    pipeline the tool is present and re-fetching it to prove that would be
    wasteful. The existing tools/configure_deps.py search is reused rather than
    duplicated — it already knows where each tool hides and validates what it
    finds, and duplicating that logic here is how the two would drift apart.
    """
    try:
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import configure_deps as cd
    except Exception:                                   # noqa: BLE001
        return None
    # configure_deps keys differ from ours for one entry, so ask by its own key.
    spec = next((t for t in cd.TOOLS if t["key"] == tool.get("spec", tool["id"])), None)
    if not spec:
        return None
    # Cheap and specific first: a sibling directory, and the paths already known
    # to be in use. Only then the bounded drive scan. Ordering matters because the
    # scan has to be short enough that the installer stays responsive, and on a
    # machine that already has the tools the first two checks find them.
    parent = os.path.dirname(REPO)
    for candidate in (os.path.join(parent, tool["dest"]), os.path.join(REPO, tool["dest"])):
        if tool.get("probe") and os.path.isfile(os.path.join(candidate, tool["probe"])):
            return candidate
    for env_name in (tool.get("env"),):
        existing = os.environ.get(env_name or "")
        if existing and os.path.isdir(existing):
            if not tool.get("probe") or os.path.isfile(os.path.join(existing, tool["probe"])):
                return existing

    roots = [parent, REPO]
    drive = os.path.splitdrive(REPO)[0] + os.sep
    if os.path.isdir(drive):
        roots.append(drive)
    for hit in cd._search_roots(spec, roots, max_depth=4, budget=6000, seconds=20.0):
        # Only accept a copy the wrappers could actually use.
        if tool.get("probe") and not os.path.isfile(os.path.join(hit, tool["probe"])):
            continue
        return hit
    return None


def step_start(tid, label, status="start", **extra):
    emit(event="step", id=tid, label=label, status=status, **extra)


# ── pip ──────────────────────────────────────────────────────────────────────

def install_pip(python: str, only_missing: bool = True) -> Dict:
    result = {"installed": [], "skipped": [], "failed": []}
    for module, pip_name, why, required in PIP_PACKAGES:
        step_start("pip:" + pip_name, "Python package: %s (%s)" % (pip_name, why))
        import importlib.util
        have = False
        try:
            have = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            have = False
        if have and only_missing:
            step_start("pip:" + pip_name, pip_name, "skip", note="already importable")
            result["skipped"].append(pip_name)
            continue
        cmd = [python, "-m", "pip", "install", "--disable-pip-version-check",
               "-i", PYPI_MIRROR, pip_name]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (subprocess.TimeoutExpired, OSError) as exc:
            step_start("pip:" + pip_name, pip_name, "fail", note=str(exc)[:200])
            result["failed"].append({"name": pip_name, "why": str(exc)[:200]})
            continue
        if p.returncode == 0:
            step_start("pip:" + pip_name, pip_name, "ok")
            result["installed"].append(pip_name)
        else:
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
            step_start("pip:" + pip_name, pip_name, "fail", note=" | ".join(tail)[:300])
            result["failed"].append({"name": pip_name, "why": " | ".join(tail)[:300]})
    return result


# ── git ──────────────────────────────────────────────────────────────────────

def fetch_source(tool: Dict) -> Dict:
    """Get a tool's source as a tarball and unpack it into TOOLS_DIR."""
    dest = os.path.join(TOOLS_DIR, tool["dest"])
    step_start(tool["id"], tool["label"])
    if tool.get("probe") and _exists(os.path.join(dest, tool["probe"])):
        step_start(tool["id"], tool["label"], "skip", path=dest, note="already present")
        return {"ok": True, "path": dest, "skipped": True}

    os.makedirs(TOOLS_DIR, exist_ok=True)
    archive = os.path.join(TOOLS_DIR, "_%s.tar.gz" % tool["dest"])
    got = _download(tool["tarball"], archive, tool["id"], tool["label"])
    if not got["ok"]:
        step_start(tool["id"], tool["label"], "fail", note=got["why"])
        return got

    # A GitHub tarball wraps everything in "<repo>-<ref>/", so extract to a
    # staging directory and move the single top-level folder into place. That
    # keeps the probe paths (which are relative to the tool root) correct.
    staging = os.path.join(TOOLS_DIR, "_staging_%s" % tool["dest"])
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in m.name.split("/"):
                    raise ValueError("unsafe path in archive: %s" % m.name)
            tf.extractall(staging)
    except (tarfile.TarError, OSError, ValueError) as exc:
        step_start(tool["id"], tool["label"], "fail",
                   note="extract failed: %s" % str(exc)[:200])
        return {"ok": False, "why": "extract failed: %s" % str(exc)[:200]}

    entries = [e for e in os.listdir(staging) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(staging, entries[0])):
        source_root = os.path.join(staging, entries[0])
    else:
        source_root = staging
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(source_root, dest)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        os.remove(archive)
    except OSError:
        pass

    ok = not tool.get("probe") or _exists(os.path.join(dest, tool["probe"]))
    step_start(tool["id"], tool["label"], "ok" if ok else "fail", path=dest,
               note="" if ok else "unpacked, but %s is missing" % tool["probe"])
    return {"ok": ok, "path": dest}


# ── downloads ────────────────────────────────────────────────────────────────

def _download(url: str, dest: str, tid: str, label: str,
              attempts: int = 4, stall_seconds: float = 45.0) -> Dict:
    """Stream a URL to a file, resuming and retrying on a broken transfer.

    A single-shot urlopen is not good enough for a 500 MB checkpoint: measured on
    the machine this was written on, hf-mirror.com delivers ~6 MB/s and then drops
    the connection partway, so a one-shot GET is a coin flip on whether the file
    arrives. This keeps a .part file and reissues a ranged request from the last
    byte received, up to `attempts` times.

    `stall_seconds` is a no-progress timeout rather than a total one. A total
    timeout penalises a slow-but-working connection and still cannot tell it apart
    from a dead one; measuring the gap between successful reads can.
    """
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last_error = None

    for attempt in range(1, attempts + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"User-Agent": "TorusFold-setup/1.0"}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=stall_seconds) as r:
                # 206 means the resume was honoured; 200 means the server ignored
                # the range and is sending the whole file again, so start over.
                if have and r.status == 200:
                    have = 0
                    os.remove(tmp)
                total = int(r.headers.get("Content-Length") or 0) + have
                f = open(tmp, "ab" if have else "wb")
                try:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        emit(event="progress", id=tid, done=have, total=total,
                             pct=round(100.0 * have / total, 1) if total else None,
                             mb=round(have / 1048576.0, 1),
                             attempt=attempt)
                finally:
                    f.close()
            if total and have < total:
                # The read loop ended without error but short of the expected size,
                # which is how a truncated response presents itself.
                last_error = "incomplete: %d of %d bytes" % (have, total)
                continue
            os.replace(tmp, dest)
            return {"ok": True, "path": dest, "bytes": os.path.getsize(dest),
                    "attempts": attempt}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            if attempt < attempts:
                emit(event="step", id=tid, label=label, status="info",
                     note="transfer dropped (%s); resuming from %.1f MB, attempt %d/%d"
                          % (last_error, have / 1048576.0, attempt + 1, attempts))
                time.sleep(min(2 ** attempt, 15))
    return {"ok": False, "why": last_error or "download failed"}


def _source_url(source: Dict, ck: Dict) -> Optional[str]:
    """The download URL for a checkpoint under one source.

    Only checkpoints hosted on HuggingFace can be redirected: RNAbpFlow's archive
    lives on Zenodo, which has its own record id and is not part of this choice.
    """
    entry = MODEL_PATHS.get(ck["id"])
    if not entry:
        return None
    repo, filename = entry
    if source["id"] == "modelscope":
        return "%s/%s/resolve/master/%s" % (source["base"], repo, filename)
    return "%s/%s/resolve/main/%s" % (source["base"], repo, filename)


def _source_reachable(source: Dict, ck: Dict, timeout: float = 12.0) -> bool:
    """Does this source serve the file? A ranged GET, so nothing large moves.

    One kilobyte is requested because a HEAD is not reliable across these hosts
    and a plain GET would transfer 508 MB just to answer the question.
    """
    url = _source_url(source, ck)
    if not url:
        return False
    req = urllib.request.Request(url, headers={
        "User-Agent": "TorusFold-setup/1.0", "Range": "bytes=0-1023"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(1024)
            return r.status in (200, 206)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def resolve_source_choice(choice: str, ck: Dict, log=None) -> Optional[Dict]:
    """Turn the requested source into a concrete host for this download.

    "auto" probes in declared order and uses the first that answers, which is the
    behaviour that makes a single button work on both a connected network and a
    restricted one. An explicit choice is honoured without probing: if the user
    picked one, silently falling back to another would hide the failure they are
    trying to diagnose.
    """
    if choice and choice not in ("auto", "", None):
        picked = next((s for s in MODEL_SOURCES if s["id"] == choice), None)
        if picked is None:
            if log:
                log("unknown source '%s'; using auto" % choice)
            return resolve_source_choice("auto", ck, log)
        return picked
    for source in MODEL_SOURCES:
        if _source_reachable(source, ck):
            if log:
                log("source: %s (%s)" % (source["id"], source["base"]))
            return source
    return None


def fetch_checkpoint(ck: Dict, source_choice: str = None) -> Dict:
    label = ck["label"]
    tid = ck["id"]
    step_start(tid, label)

    # Where this checkpoint belongs. For a tool whose source is downloadable it is
    # under TOOLS_DIR; for one that has to be placed by hand, the installed copy
    # is found anywhere on the machine, because that is where it will actually be.
    roots = []
    if ck.get("also_search"):
        for spec_key in ck["also_search"]:
            found = detect_tool(spec_key) or find_local_copy({"id": spec_key, "spec": spec_key,
                                                              "dest": spec_key})
            if found:
                roots.append(found)
    roots.append(os.path.join(TOOLS_DIR, ck["into"].split("/")[0]))
    roots.append(os.path.join(REPO, ck["into"].split("/")[0]))

    tool_root = None
    for r in roots:
        if r and os.path.isdir(r):
            tool_root = r
            break
    if tool_root is None:
        step_start(tid, label, "skip",
                   note="its tool is not present anywhere; place the tool first")
        return {"ok": False, "why": "tool not present"}

    expect = os.path.join(tool_root, ck["expect"])
    if _exists(expect):
        step_start(tid, label, "skip", path=expect, note="already present")
        return {"ok": True, "path": expect, "skipped": True}

    local = None
    if ck["id"] == "rhofold_ckpt":
        # The checkpoint comes from HuggingFace, which some networks block. Known
        # locations are checked before any search, because on this machine the
        # copy is already in an obvious place and a scan would only cost time.
        known = [
            os.path.join(os.path.splitdrive(REPO)[0] + os.sep, "Users",
                         os.environ.get("USERNAME", ""), "deploy"),
        ]
        for cand in known:
            hit = _find_named(cand, "rhofold_pretrained_params.pt", depth=6, seconds=15.0)
            if hit:
                local = hit
                break
        if not local:
            local = _search_local_checkpoint("rhofold_pretrained_params.pt")
    elif ck["id"] == "rnabpflow_ckpt":
        local = _search_local_checkpoint("RNA3DB.ckpt")

    if local:
        step_start(tid, label, "skip", note="copying from %s" % local)
        try:
            os.makedirs(os.path.dirname(expect), exist_ok=True)
            shutil.copy2(local, expect)
            step_start(tid, label, "ok", path=expect, note="copied locally")
            return {"ok": True, "path": expect, "copied_from": local}
        except OSError as exc:
            step_start(tid, label, "fail", note="copy failed: %s" % exc)

    # The download URL is chosen per source rather than read off the checkpoint,
    # so a mirror can be selected without touching the checkpoint definition.
    url = ck["url"]
    if ck["id"] in MODEL_PATHS:
        chosen = resolve_source_choice(source_choice or DEFAULT_SOURCE, ck,
                                       log=lambda m: emit(event="step", id=tid,
                                                          label=label, status="info", note=m))
        if chosen is None:
            step_start(tid, label, "fail",
                       note="no model source answered; tried %s. Pick one explicitly "
                            "with --source, or copy the file in by hand."
                            % ", ".join(s["id"] for s in MODEL_SOURCES))
            return {"ok": False, "why": "no model source reachable"}
        url = _source_url(chosen, ck) or url
        step_start(tid, label, "start", source=chosen["id"], url=url)

    # Where the file lands has to be exactly where the check will look for it.
    # `expect` is relative to the tool root, and `into` already names the parent,
    # so the download target is derived from `expect` rather than from the URL's
    # basename or a separate `filename` field — two sources for one path is how
    # this downloaded successfully and then reported failure.
    target = os.path.join(tool_root, ck["expect"])
    if os.path.isdir(target):
        target = os.path.join(target, os.path.basename(ck.get("filename") or url))
    os.makedirs(os.path.dirname(target), exist_ok=True)

    got = _download(url, target, tid, label)
    if not got["ok"]:
        step_start(tid, label, "fail", note=got["why"])
        return got

    if ck.get("archive") == "tar.gz":
        step_start(tid, label, "start", note="extracting")
        try:
            mode = "r:gz"
            with tarfile.open(target, mode) as tf:
                # Guard against path traversal in the archive.
                for m in tf.getmembers():
                    if m.name.startswith("/") or ".." in m.name.split("/"):
                        raise ValueError("unsafe path in archive: %s" % m.name)
                tf.extractall(tool_root)
        except (tarfile.TarError, OSError, ValueError) as exc:
            step_start(tid, label, "fail", note="extract failed: %s" % str(exc)[:200])
            return {"ok": False, "why": "extract failed: %s" % str(exc)[:200]}
        try:
            os.remove(target)
        except OSError:
            pass

    ok = _exists(expect)
    step_start(tid, label, "ok" if ok else "fail", path=expect,
               note="" if ok else "expected %s after install" % ck["expect"])
    return {"ok": ok, "path": expect}


def _find_named(root: str, filename: str, depth: int = 5,
                seconds: float = 15.0) -> Optional[str]:
    """Depth- and time-bounded search for one file name under one root."""
    if not os.path.isdir(root):
        return None
    deadline = time.time() + seconds
    base_depth = root.rstrip("\\/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", "node_modules")]
        if dirpath.count(os.sep) - base_depth >= depth or time.time() > deadline:
            dirnames[:] = []
            continue
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def _search_local_checkpoint(filename: str, budget: int = 8000,
                             seconds: float = 20.0) -> Optional[str]:
    """Look for a checkpoint the machine already has, before downloading it.

    Bounded twice, because this runs inside an installer the user is watching:
    a directory budget caps the work on a machine with huge trees, and a wall
    clock caps it on a slow disk. An unbounded walk here took minutes on the
    machine this was written on and made the progress bar look stuck — the one
    thing an installer must not do. Returning None is cheap; the caller then
    downloads, which is bounded and reports progress.
    """
    deadline = time.time() + seconds
    roots = [os.path.dirname(REPO), REPO,
             os.path.join(os.path.splitdrive(REPO)[0] + os.sep, "Torusfold-physics"),
             os.path.join(os.path.splitdrive(REPO)[0] + os.sep, "tmp")]
    visited = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "node_modules", "pkgs", "envs")]
            visited += 1
            if visited > budget or time.time() > deadline:
                return None
            if dirpath.count(os.sep) - root.count(os.sep) > 5:
                dirnames[:] = []
                continue
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


# ── orchestration ────────────────────────────────────────────────────────────

def run(python: str, skip_downloads: bool = False, skip_git: bool = False,
        source: str = None, install_missing: bool = False) -> Dict:
    summary = {"installed": [], "skipped": [], "failed": [], "paths": {}, "manual": [],
               "model_source": source or DEFAULT_SOURCE,
               "environment": None}

    emit(event="step", id="plan", label="Planning", status="start",
         tools_dir=TOOLS_DIR, python=python, model_source=source or DEFAULT_SOURCE)
    emit(event="step", id="plan", label="Planning", status="ok")

    # ── the Python environment ──────────────────────────────────────────────
    # Done first: if no interpreter can run the pipeline, nothing after this
    # matters, and saying so early is the most useful thing the report can do.
    step_start("env", "Python environment")
    survey = survey_environments()
    chosen = survey["chosen"]
    for probe in survey["interpreters"]:
        if not probe.get("usable"):
            continue
        t = probe.get("torch") or {}
        step_start("env", os.path.basename(os.path.dirname(probe["path"])),
                   "info",
                   note="usable%s%s" % (" (GPU)" if t.get("cuda") else "",
                                        "" if not probe.get("missing")
                                        else "; optional missing: " + ", ".join(probe["missing"])))
    if chosen is None:
        step_start("env", "Python environment", "fail",
                   note="no interpreter can import the required packages "
                        "(numpy, scipy, RNA, torch, openmm)")
    else:
        missing = chosen.get("missing") or []
        installable = [m for m in missing if m != "freesasa" or True]
        if not missing:
            step_start("env", "Python environment", "skip",
                       note="%s has everything" % chosen["path"])
            summary["skipped"].append("environment")
        elif not install_missing:
            step_start("env", "Python environment", "info",
                       note="%s is missing: %s — re-run with install-missing to "
                            "add them" % (chosen["path"], ", ".join(missing)))
        else:
            step_start("env", "Python environment", "start",
                       note="installing %d package(s) into %s"
                            % (len(installable), chosen["path"]))
            res = install_into_environment(chosen["path"], installable,
                                           use_conda=bool(survey.get("conda")))
            summary["installed"] += res["installed"]
            summary["failed"] += res["failed"]
            recheck = probe_interpreter(chosen["path"])
            summary["environment"] = {"path": chosen["path"],
                                      "complete": recheck.get("complete"),
                                      "missing": recheck.get("missing"),
                                      "gpu": (recheck.get("torch") or {}).get("cuda")}
            step_start("env", "Python environment",
                       "ok" if recheck.get("usable") else "fail",
                       note="missing after install: %s"
                            % (", ".join(recheck.get("missing") or []) or "none"))
        summary["paths"]["TORUSFOLD_PYTHON"] = chosen["path"]
        summary["environment"] = summary["environment"] or {
            "path": chosen["path"], "complete": not missing,
            "missing": missing, "gpu": (chosen.get("torch") or {}).get("cuda")}
        if not summary["environment"].get("complete"):
            summary["manual"].append({
                "id": "python-packages",
                "label": "Python packages in %s" % os.path.basename(
                    os.path.dirname(chosen["path"])),
                "why": "missing: %s. Install with conda-forge or pip, then re-run "
                       "the setup." % ", ".join(summary["environment"].get("missing") or []),
                "env": "TORUSFOLD_PYTHON",
                "detected": None,
            })

    pip_res = install_pip(python)
    summary["installed"] += pip_res["installed"]
    summary["skipped"] += pip_res["skipped"]
    summary["failed"] += pip_res["failed"]

    if not skip_git:
        for tool in GIT_TOOLS:
            local = find_local_copy(tool)
            if local:
                step_start(tool["id"], tool["label"], "skip", path=local,
                           note="already installed on this machine")
                summary["skipped"].append(tool["id"])
                summary["paths"][tool["env"]] = local
                continue
            res = fetch_source(tool)
            if res.get("ok"):
                summary["installed"].append(tool["id"])
                summary["paths"][tool["env"]] = res["path"]
            else:
                summary["failed"].append({"name": tool["id"], "why": res.get("why", "")})

    if not skip_downloads:
        for ck in CHECKPOINTS:
            # A checkpoint goes where its tool is. If the tool is not on the
            # machine at all, fetching 532 MB into an empty directory would help
            # nobody, so the step is skipped with that reason.
            res = fetch_checkpoint(ck, source_choice=source)
            if res.get("ok") and res.get("skipped"):
                summary["skipped"].append(ck["id"])
            elif res.get("ok"):
                summary["installed"].append(ck["id"])
            elif res.get("why") == "tool not present":
                step_start(ck["id"], ck["label"], "skip",
                           note="its tool is not installed; place the tool first")
                summary["skipped"].append(ck["id"])
            else:
                summary["failed"].append({"name": ck["id"], "why": res.get("why", "")})

    for tool in MANUAL_TOOLS:
        entry = dict(tool)
        entry["detected"] = detect_tool(tool["id"])
        summary["manual"].append(entry)

    # A tool that is present but was reported as a manual step is not a missing
    # dependency, and the report should not imply it is.
    present = [m["id"] for m in summary["manual"] if m["detected"]]
    summary["present_but_manual"] = present

    emit(event="summary", **summary)
    return summary


def detect_tool(key: str) -> Optional[str]:
    """Run configure_deps' own search for one tool and return where it lives.

    Reused rather than reimplemented so the installer and the report agree: two
    independent searches would eventually disagree about the same machine, and
    the user would be told a tool is missing by one screen and present by the
    other.
    """
    try:
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import configure_deps as cd
        drive = os.path.splitdrive(REPO)[0] + os.sep
        spec = next((t for t in cd.TOOLS if t["key"] == key), None)
        if not spec:
            return None
        roots = [os.path.dirname(REPO), REPO]
        if os.path.isdir(drive):
            roots.append(drive)
        hits = cd._search_roots(spec, roots)
        if not hits:
            return None
        if key == "isrnacirc":
            # A directory containing the binaries is not enough. Require both a
            # loader-clean executable AND its Data/ directory: launching proves
            # only that the DLLs resolved, and a copy on this very machine
            # launches, parses every residue as GUA, and exits 1 with no Data/.
            for h in hits:
                lay = cd._isrnacirc_layout(h)
                if not lay.get("cg_to_aa") or not lay.get("data"):
                    continue
                ok, _ = cd._run_smoke(lay["cg_to_aa"])
                if ok:
                    return h
            return None
        return hits[0]
    except Exception:                                    # noqa: BLE001
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to install pip packages into")
    ap.add_argument("--skip-downloads", action="store_true",
                    help="do not fetch model checkpoints")
    ap.add_argument("--skip-git", action="store_true",
                    help="do not clone tool repositories")
    ap.add_argument("--tools-dir", default=None,
                    help="where to place downloaded tools "
                         "(default: a TorusFold-tools directory beside the repo)")
    ap.add_argument("--source", default=None, choices=[s["id"] for s in MODEL_SOURCES] + ["auto"],
                    help="where to fetch model weights from. 'auto' probes each host "
                         "and uses the first that answers; name one to force it")
    ap.add_argument("--install-missing", action="store_true",
                    help="install packages the chosen interpreter lacks, instead of "
                         "only reporting them")
    ap.add_argument("--env-only", action="store_true",
                    help="survey the Python environment and stop")
    ap.add_argument("--list-sources", action="store_true",
                    help="show the model hosts and whether each answers right now, then exit")
    args = ap.parse_args(argv)

    if args.list_sources:
        print("Model sources (a 1 KB ranged request decides reachability):")
        probe_ck = CHECKPOINTS[0] if CHECKPOINTS else None
        for s in MODEL_SOURCES:
            ck = next((c for c in CHECKPOINTS if c["id"] in MODEL_PATHS), None)
            ok = _source_reachable(s, ck) if ck else False
            print("  %-14s %-34s %-11s %s" % (
                s["id"], s["base"], "reachable" if ok else "no answer", s["note"]))
        print("\n  default: %s" % DEFAULT_SOURCE)
        print("  use:     --source <id>   or set TORUSFOLD_MODEL_SOURCE")
        return 0

    if args.env_only:
        survey = survey_environments()
        print("Python environment")
        print("=" * 78)
        print("  conda: %s" % (survey.get("conda") or "not found"))
        for probe in survey["interpreters"]:
            t = probe.get("torch") or {}
            mark = "USABLE " if probe.get("usable") else "unusable"
            gpu = "GPU" if t.get("cuda") else "cpu"
            star = "  <-- chosen" if survey["chosen"] and \
                probe["path"] == survey["chosen"]["path"] else ""
            print("  %s %-8s %s%s" % (mark, gpu, probe["path"], star))
            if probe.get("missing"):
                print("           missing: %s" % ", ".join(probe["missing"]))
            if probe.get("error"):
                print("           error: %s" % probe["error"])
        if not survey["chosen"]:
            print("\n  No interpreter can import the required packages.")
            print("  Create one, for example:")
            print("    conda create -n torusfold -c conda-forge python=3.11 numpy scipy "
                  "viennarna openmm pytorch matplotlib")
        else:
            print("\n  chosen: %s" % survey["chosen"]["path"])
            print("  re-run with --install-missing to add anything it lacks")
        return 0

    # Rebind the module-level destination rather than declaring it global in a
    # function that already reads it: a `global` statement after the name has been
    # used is a SyntaxError.
    if args.tools_dir:
        globals()["TOOLS_DIR"] = args.tools_dir

    summary = run(args.python, skip_downloads=args.skip_downloads, skip_git=args.skip_git,
                  source=args.source, install_missing=args.install_missing)

    # A human-readable tail for the terminal, after the JSON stream.
    print("\n" + "=" * 70)
    print("Installed : %s" % (", ".join(summary["installed"]) or "nothing"))
    print("Skipped   : %s" % (", ".join(str(s) for s in summary["skipped"]) or "nothing"))
    if summary["failed"]:
        print("Failed    :")
        for f in summary["failed"]:
            print("   %-22s %s" % (f.get("name"), f.get("why", "")))
    if summary["paths"]:
        print("Paths to set:")
        for k, v in summary["paths"].items():
            print("   %-22s %s" % (k, v))
    # Scoring functions that the interface displays but the pipeline never
    # computes. Kept out of the manual list above because "NOT FOUND" would be the
    # wrong message: installing them changes nothing until something calls them.
    _unintegrated = [m for m in summary["manual"] if m.get("integrated") is False]
    if _unintegrated:
        print("Shown in the interface but not computed by the pipeline:")
        for m in _unintegrated:
            print("   %-16s [not integrated]" % m["id"])
            print("        %s" % m.get("why", ""))
    # Bundled with the repository: nothing to install, and listing it as a manual
    # step would print "NOT FOUND" for something that needs no finding.
    _bundled = [m for m in summary["manual"] if m.get("integrated") is True]
    if _bundled:
        print("Bundled with this repository (no installation needed):")
        for m in _bundled:
            print("   %-16s [in-tree]" % m["id"])
            print("        %s" % m.get("why", ""))
    _manual = [m for m in summary["manual"]
               if m.get("integrated") is None]
    print("Manual steps still required:")
    for m in _manual:
        state = ("already present at %s" % m["detected"]) if m.get("detected") else "NOT FOUND"
        print("   %-16s [%s]" % (m["id"], state))
        if not m.get("detected"):
            print("        %s" % m["why"])
            print("        set: %s" % m["env"])
    print("=" * 70)
    print("\nNow run:  python tools/configure_deps.py write")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
