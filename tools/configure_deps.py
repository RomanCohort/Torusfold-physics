# -*- coding: utf-8 -*-
"""Discover and configure the external tools TorusFold delegates to.

TorusFold's own code is physics-based and self-contained; what it cannot ship is
the third-party machinery it calls. This module knows where each of those tools
expects to be found, looks for it on this machine, and can write the environment
variables the wrappers read.

It deliberately does NOT contain any machine-specific path. Every candidate is
either derived from the repository layout, read from an existing environment
variable, or discovered by searching the directories given on the command line.
Anything it cannot find is reported as missing with the reason, never guessed at.

Usage
-----
    python tools/configure_deps.py discover            # what is present, what is not
    python tools/configure_deps.py discover --roots D: E:
    python tools/configure_deps.py write               # write .env.local + a .bat
    python tools/configure_deps.py check               # verify the wrappers resolve

The written files are git-ignored by convention (`.env.local`, `activate_deps.bat`)
so a personal machine layout never lands in the repository.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The package root, for the one check that has to import a wrapper rather than
# inspect a path (see the wsl_only branch in discover()).
SRC = os.path.join(REPO, "src")

# Publish .env.local before anything reads a tool variable. Without this the
# wrapper check below imports the wrappers with an empty environment, they build
# relative paths, and the report claims RhoFold+ and RNAbpFlow are missing while
# both checkpoints are present at the recorded locations.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _env
    _APPLIED_ENV = _env.apply(REPO)
except ImportError:                     # pragma: no cover - file always ships
    _APPLIED_ENV = []

# ── What each tool is, and how to tell whether it is present ─────────────────
#
# `probe` is a path relative to the tool's root. It is chosen to be the file the
# wrapper itself looks for, so "found here" and "the wrapper will find it" are
# the same statement. Where a tool has no single required file, `probe` is None
# and `extra_checks` decides.

TOOLS = [
    {
        "key": "isrnacirc",
        "title": "isRNAcirc (CG -> all-atom, CG refinement)",
        "env": ["ISRNACIRC_BIN_DIR", "ISRNACIRC_ROOT", "CG_TO_ALLATOM_COEFF"],
        "probe": None,
        "levels": "Level 2 / 2.5 CG->all-atom; the isRNAcirc CG fold path",
        "required": True,
        # This one is asymmetric in practice: the binaries and the Data directory
        # are frequently in different places, and a copy of CG_to_allatom.exe can
        # be present and still unable to run. Both facts are handled by
        # `isrnacirc_layout` and `smoke` below rather than assumed away.
        "smoke": "CG_to_allatom.exe",
        "note": "CG_to_allatom.exe needs its matching DLLs beside it; a bare exe "
                "exits with 0xC0000135 (DLL not found).",
        "hints": ["isRNAcirc", "isrnacirc", "cg2aa_bin", "isrna_data", "isrnacirc_data"],
        "extra_dirs": ["bin", ""],
    },
    {
        "key": "rhofold",
        "title": "RhoFold+ (ensemble member)",
        "env": ["RHOFOLD_ROOT"],
        "probe": os.path.join("pretrained", "rhofold_pretrained_params.pt"),
        "levels": "Level 1 segmented 3D prediction",
        "required": False,
        "note": "The wrapper also puts the root on sys.path, so the rhofold package "
                "must be importable from the interpreter running the pipeline.",
        "hints": ["RhoFold", "rhofold"],
    },
    {
        "key": "rnabpflow",
        "title": "RNAbpFlow (ensemble member)",
        "env": ["RNABPFLOW_ROOT", "RNABPFLOW_PYTHON"],
        # "Usable" here means the wrapper can invoke it, which is decided by this
        # script and nothing else. The 1.3 GB of checkpoints are data the installer
        # fetches separately, and requiring them for the path to count would report
        # a working checkout as missing.
        "probe": "inference_rocm.py",
        "levels": "Level 1 segmented 3D prediction",
        "required": False,
        "note": "The upstream repository ships inference.py, which takes no "
                "arguments; the wrapper runs inference_rocm.py with flags, so that "
                "file is the requirement. Checkpoints come from Zenodo record 18305861.",
        "hints": ["RNAbpFlow", "rnabpflow"],
    },
    {
        "key": "dividefold",
        "title": "DivideFold (secondary-structure evidence)",
        "env": ["TF_DIVIDEFOLD_ROOT", "TF_DIVIDEFOLD_PYTHON", "TF_DIVIDEFOLD_RUNNER"],
        "probe": None,
        "levels": "Level 0 consensus",
        "required": False,
        "note": "Also resolved from DivideFold-main/ next to the repository.",
        "default_root": os.path.join(os.path.dirname(REPO), "DivideFold-main"),
        "hints": ["DivideFold-main", "DivideFold"],
    },
    {
        "key": "trrna2",
        "title": "trRosettaRNA2 (ensemble member)",
        "env": ["TRRNA2_RUNNER"],
        "probe": None,
        "levels": "Level 1 segmented 3D prediction",
        "required": False,
        "note": "A runner SCRIPT, not a directory: the wrapper executes it directly.",
        "hints": ["trRosettaRNA2", "trrna2", "trRosettaRNA", "trRNA2"],
        # The search finds the checkout directory, but the wrapper runs this file,
        # so the value written to the environment has to be the file itself.
        "script_names": ["_trrna2_runner.py", "trrna2_runner.py", "run_trrna2.py"],
    },
    {
        "key": "structrfm",
        "title": "structRFM checkpoint (multi-task heads)",
        "env": ["TF_STRUCTRFM_MODEL"],
        "probe": None,
        "levels": "optional multi-task heads",
        "required": False,
        "note": "Opt-in only; loaded through transformers AutoModel.",
        "hints": ["structRFM", "structrfm"],
    },
    {
        "key": "rsrnasp1",
        "title": "rsRNASP1 (all-atom quality score)",
        "env": ["TORUSFOLD_RSRNASP"],
        "probe": None,
        "levels": "final quality score on the all-atom structure",
        "required": False,
        # Deliberately resolvable two ways, and neither is a path on this machine:
        # the binary is a Linux build that only runs inside WSL, so `resolved` is a
        # WSL-side directory and the usual filesystem checks do not apply.
        "note": "Linux-only C++ build (Makefile + gcc). Runs via WSL; point "
                "TORUSFOLD_RSRNASP at the checkout as a WSL path. Without it the "
                "rsRNASP1 row in the interface reports n/a rather than a score.",
        "hints": ["rsRNASP1", "rsrnasp1"],
        "wsl_only": True,
    },
    {
        "key": "pyrosetta",
        "title": "PyRosetta (Level 2.6 full-atom refinement)",
        "env": [],
        "probe": None,
        "levels": "Level 2.6 conditional PyRosetta refinement",
        "required": False,
        # Licensed software that cannot be fetched, and it lives on the Linux side
        # of WSL, so neither the filesystem probe nor the search applies. Before
        # this entry existed, PyRosetta appeared in no report at all: a user could
        # run every check this repository offers and never learn that Level 2.6
        # needed something they have to obtain under a licence.
        "note": "Licensed software — obtain it yourself from https://www.pyrosetta.org "
                "(free for academic/non-commercial use, but a licence key is still "
                "required; commercial use is paid). Not on PyPI and not on "
                "conda-forge, so no installer here can fetch it. Runs as a Python "
                "import inside WSL. Level 2.6 is skipped, not failed, without it.",
        "hints": ["pyrosetta", "PyRosetta"],
        "wsl_only": True,
        "wsl_probe": "_pyrosetta",
    },
]

# Python packages the pipeline imports directly. `module` is what to try.
PACKAGES = [
    ("numpy", "numpy", "core numerics", True),
    ("scipy", "scipy", "KD-tree neighbour search (SASA, clash, pairs)", True),
    ("RNA", "ViennaRNA", "secondary structure + partition function", True),
    ("torch", "PyTorch", "RL + the GPU coarse-grained path", True),
    ("openmm", "OpenMM", "MD, replica exchange, minimisation", True),
    ("gemmi", "gemmi", "CIF/PDB ingest in circrna_library", False),
    ("freesasa", "FreeSASA", "per-residue SASA in the immune heuristic", False),
    ("transformers", "transformers", "structRFM multi-task heads", False),
    ("pandas", "pandas", "analysis scripts", False),
]

# Packaged tools that are separate executables rather than importable modules.
BINARIES = [
    ("wsl", "WSL", "Level 2.6 PyRosetta refinement (Windows path)", False),
    ("cmsearch", "Infernal cmsearch", "Rfam-based MSA (optional)", False),
]


def _which(name: str) -> Optional[str]:
    from shutil import which
    return which(name)


# What the pipeline itself must be able to import. `RNA` is ViennaRNA; the rest
# are used by the folding and refinement core. rhofold's own dependencies
# (matplotlib, ml_collections, biopython, dm-tree, einops) are listed because
# RhoFold+ is imported IN-PROCESS, so a missing one there fails the whole
# predictor — it is not a separate environment's problem.
PIPELINE_IMPORTS = ["numpy", "scipy", "RNA", "torch", "openmm",
                    "matplotlib", "ml_collections", "Bio", "tree", "einops"]

# Interpreters worth testing, in preference order. The names are generic
# conventions (conda envs live under anaconda3/envs or ana/envs); none of them is
# a hard-coded requirement, and any failure just moves on to the next.
#
# A plain python.org installation is included deliberately and is listed FIRST.
# The whole dependency set the pipeline needs has Windows wheels on PyPI —
# numpy, scipy, ViennaRNA, OpenMM and torch all publish win_amd64 builds for
# CPython 3.10 through 3.14 — so conda is a convenience, not a requirement. An
# earlier version of this list contained only conda paths, which meant a machine
# with a normal Python installation found nothing and was told to install conda.
CANDIDATE_INTERPRETERS = [
    # python.org installs, including the per-user default location.
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python314\python.exe"),
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python313\python.exe"),
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe"),
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Python311\python.exe"),
    r"C:\Python314\python.exe",
    r"C:\Python313\python.exe",
    r"C:\Python312\python.exe",
    r"C:\Python311\python.exe",
    # The official "py" launcher, which resolves whatever is actually installed.
    os.path.expanduser(r"~\AppData\Local\Programs\Python\Launcher\py.exe"),
    # conda distributions, in the layouts seen in practice.
    r"C:\ana\envs\comfyui\python.exe",
    r"C:\ana\envs\circrna3d\python.exe",
    r"C:\ana\envs\bio\python.exe",
    r"C:\ana\python.exe",
    r"C:\anaconda3\python.exe",
    r"C:\ProgramData\anaconda3\python.exe",
    r"C:\miniconda3\python.exe",
    r"C:\ProgramData\miniconda3\python.exe",
    os.path.expanduser(r"~\miniconda3\python.exe"),
    os.path.expanduser(r"~\anaconda3\python.exe"),
    os.path.expanduser(r"~\miniforge3\python.exe"),
    os.path.expanduser(r"~\.conda\python.exe"),
]


def find_conda() -> Optional[str]:
    """Where conda lives, if it does. Used only to report the options.

    Returns the conda executable path, or None on a machine without a conda
    distribution. Nothing in the pipeline requires conda; this exists so the
    setup advice can name the right command for the machine it is running on.
    """
    from shutil import which
    found = which("conda")
    if found:
        return found
    for exe in (
        r"C:\ana\Scripts\conda.exe",
        r"C:\anaconda3\Scripts\conda.exe",
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        r"C:\miniconda3\Scripts\conda.exe",
        r"C:\ProgramData\miniconda3\Scripts\conda.exe",
        os.path.expanduser(r"~\anaconda3\Scripts\conda.exe"),
        os.path.expanduser(r"~\miniconda3\Scripts\conda.exe"),
        os.path.expanduser(r"~\miniforge3\Scripts\conda.exe"),
    ):
        if os.path.isfile(exe):
            return exe
    return None


def check_interpreters() -> List[Dict]:
    """Which Python can run the pipeline, and does it also have GPU torch?

    Worth testing rather than assuming, because the environments differ in ways
    that change results rather than just speed: on this machine one has CPU-only
    torch and the other a ROCm build. RNAbpFlow is launched with `--device cuda`,
    which the ROCm build satisfies and the CPU build rejects, so the same
    sequence yields one predictor in one environment and two in the other — and
    the two-predictor ensemble passed validation where the one-predictor run
    failed it.
    """
    import subprocess
    out = []
    # A tiny helper keeps the probe readable and avoids nested-quote escaping.
    helper = (
        "import importlib.util as u, json, sys\n"
        "def safe(m):\n"
        "    try: return u.find_spec(m)\n"
        "    except Exception: return None\n"
        "missing = [m for m in %r if safe(m) is None]\n"
        "gpu = None\n"
        "try:\n"
        "    import torch\n"
        "    gpu = {'torch': torch.__version__, 'hip': getattr(torch.version, 'hip', None),\n"
        "           'cuda_available': bool(torch.cuda.is_available())}\n"
        "except Exception as exc:\n"
        "    gpu = {'error': repr(exc)}\n"
        "print(json.dumps({'missing': missing, 'torch': gpu,\n"
        "                  'py': list(sys.version_info[:3]),\n"
        "                  'exe': sys.executable}))\n"
    ) % (PIPELINE_IMPORTS,)

    for exe in CANDIDATE_INTERPRETERS:
        if not os.path.isfile(exe):
            continue
        entry = {"path": exe, "usable": False, "missing": None, "torch": None,
                 "py": None, "resolved": None}
        try:
            p = subprocess.run([exe, "-c", helper], capture_output=True, text=True,
                               timeout=180)
            if p.returncode == 0 and p.stdout.strip():
                last = p.stdout.strip().splitlines()[-1]
                data = json.loads(last)
                entry["missing"] = data["missing"]
                entry["torch"] = data["torch"]
                entry["py"] = data.get("py")
                # py.exe is a launcher: report which interpreter it actually chose,
                # otherwise the table shows two "py.exe" rows that look identical.
                entry["resolved"] = data.get("exe")
                entry["usable"] = not data["missing"]
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            entry["error"] = str(exc)[:120]
        out.append(entry)
    return out


def pick_interpreter(interpreters: List[Dict]) -> Optional[Dict]:
    """The best usable interpreter: one with GPU torch first, else any usable."""
    usable = [i for i in interpreters if i.get("usable")]
    if not usable:
        return None
    gpu = [i for i in usable if (i.get("torch") or {}).get("cuda_available")]
    return (gpu or usable)[0]


def check_packages() -> List[Dict]:
    out = []
    for module, label, why, required in PACKAGES:
        spec = None
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        version = None
        if spec is not None:
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, "__version__", None)
                if version is None and module == "RNA":
                    version = getattr(mod, "version", lambda: None)()
            except Exception:                      # noqa: BLE001 — import side effects
                version = None
        out.append({
            "module": module, "label": label, "why": why, "required": required,
            "found": spec is not None, "version": str(version) if version else None,
        })
    return out


def check_binaries() -> List[Dict]:
    out = []
    for name, label, why, required in BINARIES:
        path = _which(name)
        out.append({"name": name, "label": label, "why": why,
                    "required": required, "found": bool(path), "path": path})
    return out


def _looks_like_tool(path: str, tool: Dict) -> Tuple[bool, str]:
    """Does `path` look like this tool's installation root?

    Returns (ok, reason). The probe file is authoritative: it is what the wrapper
    reads, so its presence is the only evidence that matters. `require` lists
    files the wrapper additionally needs to actually run — a directory can satisfy
    `probe` and still be unusable, which is exactly what happened with an
    upstream RNAbpFlow checkout: it had the checkpoint's parent directory but not
    the modified inference script the wrapper invokes.
    """
    if not os.path.isdir(path):
        return False, "not a directory"
    probe = tool.get("probe")
    if probe:
        if not os.path.isfile(os.path.join(path, probe)):
            return False, "missing " + probe
    for needed in tool.get("require", []) or []:
        if not os.path.isfile(os.path.join(path, needed)):
            return False, "missing " + needed
    if probe:
        return True, probe
    # No single probe: accept the directory if it contains anything hint-like.
    for hint in tool.get("hints", []):
        if os.path.exists(os.path.join(path, hint)):
            return True, hint
    entries = os.listdir(path)
    if entries:
        return True, "%d entries" % len(entries)
    return False, "empty directory"


def _run_smoke(exe: str, timeout: float = 25.0) -> Tuple[bool, str]:
    """Can this executable actually start?

    Necessary because presence is not capability for native Windows binaries: a
    copy can be there and immediately exit with a loader error, most often
    0xC0000135 (a required DLL is not beside it).

    This only proves the binary loads. It does not prove the tool will work —
    CG_to_allatom.exe loads cleanly and then refuses to run without its five
    coefficient files, which is a separate check (`_isrnacirc_coeff`).
    """
    import subprocess
    if not os.path.isfile(exe):
        return False, "not present"
    try:
        p = subprocess.run([exe], capture_output=True, text=True, timeout=timeout,
                           cwd=os.path.dirname(exe) or None)
    except subprocess.TimeoutExpired:
        # A program that hangs waiting for arguments has loaded successfully.
        return True, "loads (waited for arguments)"
    except OSError as exc:
        return False, "cannot launch: %s" % exc
    rc = p.returncode & 0xFFFFFFFF
    known = {
        0xC0000135: "STATUS_DLL_NOT_FOUND — a required DLL is missing from its folder",
        0xC000007B: "STATUS_INVALID_IMAGE_FORMAT — wrong architecture or a 32/64-bit mismatch",
        0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND — DLL version mismatch",
    }
    if rc in known:
        return False, known[rc]
    return True, "runs (exit %s)" % p.returncode


# The files CG_to_allatom.exe opens before it will do anything. Reading
# CG_to_allatom.h: it opens coeffDIR+"AA_baseA.dat" first and prints
# "Wrong coeffDIR" if that fails, then G, C, U and AA_backbone.dat.
CG_COEFF_FILES = ("AA_baseA.dat", "AA_baseG.dat", "AA_baseC.dat",
                  "AA_baseU.dat", "AA_backbone.dat")


def find_cg_coeff_dir(root: str) -> Optional[str]:
    """Where CG_to_allatom's coefficient files are, found rather than assumed.

    The templates are NOT in `Data/`, which is what both the binary's own
    documented default (`../Data/data/`) and an earlier version of this script
    assumed. On the distributed package they are in `Data/data/IsRNA2/`, so a
    correctly installed isRNAcirc still fails with "Wrong coeffDIR" — a message
    that names the directory and not the missing files, which is why this looked
    like a broken binary for some time.
    """
    if not root or not os.path.isdir(root):
        return None

    def has_all(path):
        return os.path.isdir(path) and all(
            os.path.isfile(os.path.join(path, name)) for name in CG_COEFF_FILES
        )

    if has_all(root):
        return root
    for rel in (("Data", "data", "IsRNA2"), ("data", "IsRNA2"),
                ("data", "data", "IsRNA2"), ("Data", "data"), ("data",)):
        candidate = os.path.join(root, *rel)
        if has_all(candidate):
            return candidate
    # Bounded search, for a copy that has been rearranged.
    for base, dirs, _files in os.walk(root):
        if base[len(root):].count(os.sep) > 3:
            dirs[:] = []
            continue
        if has_all(base):
            return base
    return None


def _isrnacirc_layout(root: str) -> Dict[str, Optional[str]]:
    """Resolve isRNAcirc's three pieces, which are often in different folders.

    The distribution is meant to be one tree (`bin/IsRNAcirc.out` +
    `Data/`), but assembled installs end up split: the executables in one place,
    `Data/` in another. Each piece is located independently and reported
    separately, because a partial layout is still worth telling the user about —
    it explains which sub-step fails.
    """
    layout = {"bin_dir": None, "cg_to_aa": None, "isrnacirc_exe": None, "data": None}

    def consider(dirpath: str) -> None:
        for name in ("CG_to_allatom.exe",):
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and layout["cg_to_aa"] is None:
                layout["cg_to_aa"] = p
                layout["bin_dir"] = dirpath
        for name in ("IsRNAcirc.exe", "IsRNAcirc.out"):
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and layout["isrnacirc_exe"] is None:
                layout["isrnacirc_exe"] = p
        if layout["data"] is None and os.path.isdir(os.path.join(dirpath, "Data")):
            layout["data"] = os.path.join(dirpath, "Data")
        if layout["data"] is None and os.path.isdir(os.path.join(dirpath, "data")):
            layout["data"] = os.path.join(dirpath, "data")

    if not os.path.isdir(root):
        return layout
    consider(root)
    for sub in ("bin", "CGtools", "cg2aa_bin"):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path):
            consider(sub_path)
    return layout


def _resolve_script(root: str, names: List[str]) -> Optional[str]:
    """Turn a checkout directory into the runner script the wrapper executes.

    The search below finds *directories*, but some wrappers run a file directly
    (TRRNA2_RUNNER is executed as argv[0]). Writing the directory into that
    variable produces "the file exists in the report but the run still fails",
    which is exactly the class of mistake this tool exists to prevent.
    """
    if not os.path.isdir(root):
        return root if os.path.isfile(root) else None
    for name in names:
        direct = os.path.join(root, name)
        if os.path.isfile(direct):
            return direct
    # Fall back to a shallow scan: the runner is not always named identically.
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "models")]
            for fn in filenames:
                if fn.lower().endswith(".py") and "runner" in fn.lower():
                    return os.path.join(dirpath, fn)
    except OSError:
        pass
    return None


def _search_roots(tool: Dict, roots: List[str], max_depth: int = 4,
                  budget: int = 20000, seconds: float = 25.0) -> List[str]:
    """Breadth-limited search for a tool root under `roots`.

    Deliberately shallow, name-filtered, and bounded in BOTH directory count and
    wall time. A full disk walk on a machine with conda environments takes minutes
    and finds nothing extra, because these tools live in directories named after
    themselves.

    The budget counter is incremented for every directory examined, before the
    depth test. An earlier version incremented it only inside the depth check, so
    on a drive root — where the first children are already at max_depth — the
    counter never advanced and the budget never applied. That turned an installer
    step into a multi-minute stall with no output.
    """
    hints = [h.lower() for h in tool.get("hints", [])]
    if not hints:
        return []
    deadline = time.time() + seconds
    found: List[str] = []
    seen = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip("\\/").count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            seen += 1
            if seen > budget or time.time() > deadline:
                return found
            depth = dirpath.rstrip("\\/").count(os.sep) - base_depth
            if depth >= max_depth:
                dirnames[:] = []
                continue
            # Do not descend into these: huge, and they never hold these tools.
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in ("__pycache__", "node_modules", ".git",
                                                "pkgs", "envs", "site-packages",
                                                "wheels", "conda-bld", "Library")]
            for d in list(dirnames):
                if any(h in d.lower() for h in hints):
                    candidate = os.path.join(dirpath, d)
                    ok, _ = _looks_like_tool(candidate, tool)
                    if ok:
                        found.append(candidate)
    return found


def discover(roots: List[str], verbose: bool = False) -> Dict:
    result = {"tools": [], "packages": check_packages(), "binaries": check_binaries()}
    result["interpreters"] = check_interpreters()
    result["interpreter"] = pick_interpreter(result["interpreters"])

    for tool in TOOLS:
        entry = dict(tool)
        env_value = None
        for name in tool["env"]:
            v = os.environ.get(name)
            if v and os.path.exists(v):
                env_value = v
                break
        resolved, source, reason = None, None, ""

        # A WSL-only tool cannot be resolved from here. Its path is a POSIX path
        # inside the Linux distribution — /home/... does not exist as far as this
        # process is concerned — so exists(), _looks_like_tool and the filesystem
        # search are all the wrong question. Ask the wrapper that has to invoke it,
        # which probes inside WSL where the binary actually lives.
        if tool.get("wsl_only"):
            probe = None
            try:
                sys.path.insert(0, SRC)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                # Which prober to use is the tool's own choice: two WSL-only tools
                # are checked in completely different ways, one by running a binary
                # and one by importing a Python module.
                prober = tool.get("wsl_probe")
                if prober == "_pyrosetta":
                    import _pyrosetta as _pj
                    probe = _pj.probe()
                else:
                    from torusfold.scheme2 import rsrnasp_quality as _rs
                    probe = _rs.available()
            except Exception as exc:                     # noqa: BLE001
                probe = {"available": False, "why": "check failed: %s" % exc}
            if probe.get("available"):
                # rsRNASP1 reports `root`; the PyRosetta prober reports `path`.
                entry["resolved"] = probe.get("root") or probe.get("path")
                entry["source"] = "wsl"
                entry["reason"] = "reachable inside %s" % (probe.get("distro") or "WSL")
            else:
                entry["reason"] = probe.get("why", "not available")
            # The licence text travels with the result, so any report that prints a
            # PyRosetta row can also print what obtaining it involves.
            if probe.get("licence"):
                entry["licence"] = probe["licence"]
            entry["wsl_probe"] = probe
            entry.update({"usable": bool(probe.get("available"))})
            result["tools"].append(entry)
            continue

        if env_value:
            ok, why = _looks_like_tool(env_value, tool)
            if ok:
                resolved, source, reason = env_value, "environment", why
            else:
                reason = "env points at %s but it %s" % (env_value, why)

        if not resolved and tool.get("default_root"):
            ok, why = _looks_like_tool(tool["default_root"], tool)
            if ok:
                resolved, source, reason = tool["default_root"], "repo layout", why
            elif not reason:
                reason = "default location %s: %s" % (tool["default_root"], why)

        candidates = []
        if not resolved:
            candidates = _search_roots(tool, roots)
            # Rank by explicit intent, then by preference. An environment variable
            # the user already set is the strongest signal, and a copy this tool
            # downloaded into its own staging directory is the weakest — that one
            # is a fallback, not a replacement for a working checkout elsewhere on
            # the machine.
            staged = os.path.normcase(os.path.join(os.path.dirname(REPO), "TorusFold-tools"))
            usable = []
            for hit in candidates:
                ok, _why = _looks_like_tool(hit, tool)
                if not ok:
                    continue
                rank = 2
                if os.path.normcase(hit).startswith(staged):
                    rank = 3
                usable.append((rank, hit))
            usable.sort(key=lambda t: t[0])
            if usable:
                resolved, source = usable[0][1], "search"
                reason = "found under %s" % os.path.dirname(usable[0][1])
            elif not reason:
                reason = "not found in the searched roots"

        # isRNAcirc gets a deeper look: its pieces are separate files, and the
        # interesting question is which of them work, not whether a folder exists.
        #
        # It only counts as resolved when the conversion binary actually loads AND
        # a Data/ directory is present, because both are needed for the CG->all-atom
        # step the pipeline calls. A folder holding a copy that dies at load time is
        # reported as PARTIAL, not as found — otherwise the report sends the user to
        # configure an environment variable that cannot help.
        if tool["key"] == "isrnacirc" and resolved:
            layout = _isrnacirc_layout(resolved)
            entry["layout"] = layout
            cg_ok = False
            if layout["cg_to_aa"]:
                cg_ok, why = _run_smoke(layout["cg_to_aa"])
                entry["cg_to_allatom_usable"] = cg_ok
                entry["cg_to_allatom_why"] = why
            else:
                entry["cg_to_allatom_usable"] = False
                entry["cg_to_allatom_why"] = "no CG_to_allatom.exe in this tree"
            if layout["isrnacirc_exe"]:
                ok, why = _run_smoke(layout["isrnacirc_exe"])
                entry["isrnacirc_usable"] = ok
                entry["isrnacirc_why"] = why
            else:
                entry["isrnacirc_usable"] = False
                entry["isrnacirc_why"] = "no IsRNAcirc.exe/.out in this tree"

            # Locate the coefficient files, and treat their absence as a failure.
            #
            # This is the check that matters. `Data/` existing proves nothing: on
            # the distributed package Data/ is present, CG_to_allatom.exe loads
            # cleanly, and the conversion still cannot run because the five
            # AA_*.dat templates live in Data/data/IsRNA2/, a directory the
            # binary's own default does not point at. That combination was
            # previously reported as present-and-usable while every conversion
            # failed with "Wrong coeffDIR".
            coeff = find_cg_coeff_dir(resolved)
            entry["cg_coeff_dir"] = coeff
            if not cg_ok:
                resolved = None
                source = None
                entry["reason"] = ("CG_to_allatom.exe %s (in %s)"
                                   % (entry["cg_to_allatom_why"],
                                      layout["bin_dir"] or "?"))
            elif not coeff:
                resolved = None
                source = None
                entry["reason"] = (
                    "CG_to_allatom.exe launches but its coefficient files (%s) "
                    "are nowhere in the tree, so the conversion cannot run: it "
                    "exits with \"Wrong coeffDIR\". They are usually in "
                    "Data/data/IsRNA2/."
                    % ", ".join(CG_COEFF_FILES[:2]))
            elif not layout["data"]:
                # Launching and having coefficients is still not the whole tool:
                # the MD refinement path needs Data/ as well.
                entry["reason"] = ("CG->all-atom ready (coeff: %s); Data/ is "
                                   "absent so the isRNAcirc refinement path is not"
                                   % coeff)
            elif not entry.get("isrnacirc_usable"):
                # The CG->all-atom path works; the MD refinement path does not.
                # That is a partial result, and saying so is more useful than
                # calling the whole tool missing.
                entry["reason"] = ("CG->all-atom usable (coeff: %s); IsRNAcirc "
                                   "refinement %s"
                                   % (coeff, entry.get("isrnacirc_why", "unusable")))
            else:
                entry["reason"] = "CG->all-atom and refinement both usable"

        # Some wrappers execute a FILE rather than point at a directory. Resolve
        # that here, so what gets written to the environment is runnable.
        if resolved and tool.get("script_names"):
            script = _resolve_script(resolved, tool["script_names"])
            if script:
                entry["script"] = script
                entry["reason"] = "runner script: %s" % script
            else:
                entry["reason"] = ("found %s but no runner script in it "
                                   "(%s)" % (resolved, ", ".join(tool["script_names"])))

        entry.update({"resolved": resolved, "source": source, "reason": reason,
                      "candidates": candidates[:8],
                      "env_to_set": (tool["env"][0] if resolved else None)})
        if verbose:
            print("  %-12s %-9s %s" % (tool["key"], source or "MISSING", resolved or reason))
        result["tools"].append(entry)

    return result


def env_for(result: Dict) -> Dict[str, str]:
    """The environment variables that make the found tools reachable.

    The subprocess interpreters are taken from the interpreter the checker picked,
    not from sys.executable. That matters: RNAbpFlow is invoked with
    `--device cuda`, so it only runs under a torch build that can satisfy that —
    the GPU-capable one. Pointing these at whichever Python happened to start the
    configurator would silently drop a predictor from the ensemble.
    """
    env: Dict[str, str] = {}
    chosen = (result.get("interpreter") or {}).get("path") or sys.executable
    # The interpreter itself, so the launcher and any CLI run use the one that can
    # actually import the pipeline. Written even when nothing else is found: it is
    # the single most useful variable here.
    env["TORUSFOLD_PYTHON"] = chosen
    for t in result["tools"]:
        if not t.get("resolved"):
            continue
        key = t["key"]
        root = t["resolved"]
        if key == "isrnacirc":
            layout = t.get("layout") or {}
            # ISRNACIRC_ROOT is the tree the wrapper expands from: it looks for
            # bin/ and Data/ underneath it. Going up one level from Data/ gives
            # that tree.
            if layout.get("data"):
                env["ISRNACIRC_ROOT"] = os.path.dirname(layout["data"])
            elif layout.get("bin_dir"):
                env["ISRNACIRC_ROOT"] = os.path.dirname(layout["bin_dir"])
            if layout.get("cg_to_aa"):
                env["ISRNACIRC_BIN_DIR"] = layout["bin_dir"]
            # The coefficient directory, NOT Data/. This used to be set to Data/,
            # which is a real directory and therefore looked correct, while the
            # five AA_*.dat files it must contain are one level further down in
            # Data/data/IsRNA2/. The wrapper passed Data/ straight through and
            # every conversion failed with "Wrong coeffDIR".
            coeff = t.get("cg_coeff_dir")
            if coeff:
                env["CG_TO_ALLATOM_COEFF"] = coeff
        elif key == "rhofold":
            env["RHOFOLD_ROOT"] = root
        elif key == "rnabpflow":
            env["RNABPFLOW_ROOT"] = root
            env["RNABPFLOW_PYTHON"] = chosen
        elif key == "dividefold":
            env["TF_DIVIDEFOLD_ROOT"] = root
            env["TF_DIVIDEFOLD_PYTHON"] = chosen
        elif key == "trrna2":
            # The wrapper runs this path, so it must be the script, not the folder.
            env["TRRNA2_RUNNER"] = t.get("script") or root
        elif key == "structrfm":
            env["TF_STRUCTRFM_MODEL"] = root
    return env


def read_env_local(path: str) -> Dict[str, str]:
    """Parse a previously written .env.local back into a dict."""
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    try:
        with io.open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def merge_env(path: str, new: Dict[str, str]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Combine a new scan with the previously recorded one.

    A narrower scan must not erase a broader one. `configure_deps.py write --fast`
    only looks beside the repository, so on a machine whose tools live elsewhere it
    finds a subset — and writing that subset verbatim replaced a complete set of
    variables with two, silently breaking every tool it dropped. The launcher uses
    --fast on first run, so this happened on the normal path.

    Existing values win only when the new scan has nothing for that key; a tool
    found by the new scan replaces the old path for it, because that reflects where
    the tool is now.
    """
    previous = read_env_local(path)
    merged = dict(previous)
    merged.update(new)
    added = sorted(k for k in new if k not in previous)
    kept = sorted(k for k in previous if k not in new)
    return merged, added, kept


def write_env_local(env: Dict[str, str], path: str) -> None:
    lines = [
        "# Generated by tools/configure_deps.py - machine-specific, do not commit.",
        "# Sourced by activate_deps.bat / activate_deps.sh.",
    ]
    for k in sorted(env):
        lines.append("%s=%s" % (k, env[k]))
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_activate_bat(env: Dict[str, str], path: str) -> None:
    """Write the environment as `set` lines.

    Encoded as UTF-8 WITH a byte-order mark, unlike the other files this module
    writes. On a machine whose user profile is not ASCII — a Chinese user name is
    the common case — the paths themselves contain non-ASCII bytes, and cmd.exe
    reads a .bat as bytes in the console's active code page. Without a BOM it
    decodes them as the OEM page and the variables come out mangled; with one it
    switches to UTF-8 and they survive. A .bat with no non-ASCII paths is
    unaffected either way, so this is safe for everyone.
    """
    lines = [
        "@echo off",
        "REM Generated by tools/configure_deps.py -- machine-specific, do not commit.",
        "REM Run this before starting the server or a CLI run: it only sets variables.",
        "REM Written as UTF-8 with a BOM so non-ASCII paths survive cmd.exe.",
    ]
    for k in sorted(env):
        lines.append('set "%s=%s"' % (k, env[k]))
    lines.append('echo TorusFold external tools configured (%d variables).' % len(env))
    with io.open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def _wrap(text: str, width: int) -> List[str]:
    """Wrap a note onto lines of at most `width`, on word boundaries."""
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)
    return lines or [""]


def _no_interpreter_advice() -> List[str]:
    """What to do when no candidate interpreter can run the pipeline.

    Deliberately does not lead with "install Anaconda". Every dependency the
    pipeline imports — numpy, scipy, ViennaRNA, OpenMM, torch and the RhoFold+
    helpers — publishes a Windows wheel on PyPI for CPython 3.10-3.14, so a
    python.org installer plus pip is sufficient. conda is offered second, for a
    machine that already has it, because it is genuinely convenient for a
    PyTorch build matched to a particular GPU.
    """
    out = []
    out.append("")
    out.append("  No interpreter here can run the pipeline.")
    out.append("")
    out.append("  Option 1 - plain Python (no Anaconda needed, ~25 MB download)")
    out.append("    1. Install Python 3.11 or newer from https://www.python.org/downloads/")
    out.append("       Tick \"Add python.exe to PATH\" in the installer.")
    out.append("    2. Open a NEW terminal, then run:")
    out.append("         python -m pip install --upgrade pip")
    out.append("         python -m pip install numpy scipy ViennaRNA openmm \\")
    out.append("             matplotlib ml_collections biopython dm-tree einops \\")
    out.append("             gemmi freesasa pandas transformers")
    out.append("         python -m pip install torch")
    out.append("       Every one of those has a Windows wheel; no compiler is required.")
    out.append("    3. For an AMD GPU, install the ROCm build of torch instead of the")
    out.append("       default one - see https://pytorch.org/get-started/locally/")
    out.append("    4. Re-run:  start.bat --setup")
    conda = find_conda()
    out.append("")
    if conda:
        out.append("  Option 2 - the conda you already have  (%s)" % conda)
        out.append("        conda create -n torusfold -c conda-forge python=3.11 \\")
        out.append("            numpy scipy openmm pytorch matplotlib pandas \\")
        out.append("            biopython einops gemmi")
        out.append("        conda activate torusfold")
        out.append("        pip install ViennaRNA dm-tree freesasa transformers")
        out.append("      ViennaRNA is not on conda-forge for Windows and dm-tree has no")
        out.append("      conda-forge build, so those two come from pip either way.")
    else:
        out.append("  Option 2 - Anaconda / Miniconda")
        out.append("      Not installed on this machine. It is not required: Option 1")
        out.append("      installs everything. If you prefer conda anyway, get Miniconda")
        out.append("      from https://docs.conda.io/en/latest/miniconda.html and then")
        out.append("      run the commands in Option 1 with `pip` from that environment.")
    out.append("")
    out.append("  Either way, finish with:  python tools\\configure_deps.py write")
    return out


def render(result: Dict) -> str:
    out = []
    tools = result["tools"]
    ok = [t for t in tools if t.get("resolved")]
    missing_req = [t for t in tools if not t.get("resolved") and t.get("required")]
    missing_opt = [t for t in tools if not t.get("resolved") and not t.get("required")]

    out.append("")
    out.append("External tools")
    out.append("-" * 74)
    for t in tools:
        if t.get("resolved"):
            mark = "OK     "
        elif t.get("layout") and (t["layout"].get("cg_to_aa") or t["layout"].get("isrnacirc_exe")):
            # Pieces are present but at least one cannot run. Saying "MISSING"
            # here would hide the more useful fact.
            mark = "PARTIAL"
        elif t["required"]:
            mark = "MISSING"
        else:
            mark = "absent "
        out.append("  %s %-38s %s" % (mark, t["title"][:38], t.get("source") or ""))
        if t.get("resolved"):
            out.append("          %s" % t["resolved"])
        elif t.get("layout"):
            lay = t["layout"]
            if lay.get("cg_to_aa"):
                out.append("          CG_to_allatom.exe : %s" % lay["cg_to_aa"])
                out.append("            %s" % t.get("cg_to_allatom_why", ""))
            if lay.get("isrnacirc_exe"):
                out.append("          IsRNAcirc         : %s" % lay["isrnacirc_exe"])
                out.append("            %s" % t.get("isrnacirc_why", ""))
            if lay.get("data"):
                out.append("          Data/             : %s" % lay["data"])
            if not lay.get("data"):
                out.append("          Data/             : not found")
        else:
            out.append("          %s" % t.get("reason", ""))
        if t.get("candidates") and len(t["candidates"]) > 1:
            out.append("          also found: %s" % ", ".join(t["candidates"][1:4]))
        # A tool that is licensed rather than downloadable says so here, in the
        # report, not only in a document: "absent" and "you must obtain a licence"
        # are different instructions and the second is easy to miss.
        if t.get("licence"):
            for line in _wrap(t["licence"], 68):
                out.append("          ! %s" % line)

    out.append("")
    out.append("Interpreters")
    out.append("-" * 74)
    for i in result.get("interpreters", []):
        chosen = result.get("interpreter") or {}
        mark = "CHOSEN " if i["path"] == chosen.get("path") else ("usable " if i.get("usable") else "unusable")
        torch_info = i.get("torch") or {}
        gpu = "GPU" if torch_info.get("cuda_available") else "cpu"
        ver = torch_info.get("torch") or torch_info.get("error") or "no torch"
        pyv = ".".join(str(x) for x in i["py"]) if i.get("py") else "?"
        out.append("  %s %-48s py%-7s %s" % (mark, i["path"], pyv, gpu))
        # py.exe is a launcher, so the path in the table is not the interpreter
        # that ran the probe. Name the one it resolved to.
        if i.get("resolved") and os.path.normcase(i["resolved"]) != os.path.normcase(i["path"]):
            out.append("          -> %s" % i["resolved"])
        if i.get("missing"):
            out.append("          missing: %s" % ", ".join(i["missing"]))
        out.append("          torch: %s" % ver)
    if not result.get("interpreter"):
        out.extend(_no_interpreter_advice())

    out.append("")
    out.append("Python packages in the running interpreter")
    out.append("-" * 74)
    for p in result["packages"]:
        mark = "OK     " if p["found"] else ("MISSING" if p["required"] else "absent ")
        ver = (" %s" % p["version"]) if p["version"] else ""
        out.append("  %s %-16s %-28s%s" % (mark, p["label"][:16], p["why"][:28], ver))

    out.append("")
    out.append("Command-line tools")
    out.append("-" * 74)
    for b in result["binaries"]:
        mark = "OK     " if b["found"] else ("MISSING" if b["required"] else "absent ")
        out.append("  %s %-20s %s" % (mark, b["label"][:20], b.get("path") or b["why"]))

    out.append("")
    out.append("Summary")
    out.append("-" * 74)
    out.append("  tools found   : %d / %d" % (len(ok), len(tools)))
    if missing_req:
        out.append("  BLOCKING      : %s" % ", ".join(t["key"] for t in missing_req))
    if missing_opt:
        out.append("  missing (opt) : %s" % ", ".join(t["key"] for t in missing_opt))
    pk = [p for p in result["packages"] if not p["found"] and p["required"]]
    if pk:
        out.append("  MISSING PACKAGES: %s" % ", ".join(p["label"] for p in pk))
    out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["discover", "write", "check", "json"])
    ap.add_argument("--roots", nargs="*", default=None,
                    help="Directories to search for the tools (default: the repo's "
                         "parent and the current drive roots given by --drives).")
    ap.add_argument("--drives", nargs="*", default=None,
                    help="Drive letters to search, e.g. --drives D: E:")
    ap.add_argument("--json", dest="json_out", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="only look beside the repository, skipping the drive "
                         "search. Used by the launcher, which must not spend "
                         "minutes before the server starts.")
    args = ap.parse_args(argv)

    roots = list(args.roots or [])
    if not roots:
        roots.append(os.path.dirname(REPO))
        roots.append(REPO)
        if args.drives:
            roots.extend(args.drives)
        elif not args.fast:
            # The drive the repository lives on. Skipped with --fast because the
            # launcher runs this on every start and a drive scan is minutes.
            drive = os.path.splitdrive(REPO)[0] + os.sep
            if os.path.isdir(drive):
                roots.append(drive)

    if args.action == "check":
        return _check_wrappers()

    result = discover(roots)
    if args.json_out or args.action == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(render(result))
    if args.action == "write":
        env = env_for(result)
        env_local = os.path.join(REPO, ".env.local")
        bat = os.path.join(REPO, "activate_deps.bat")
        merged, added, kept = merge_env(env_local, env)
        write_env_local(merged, env_local)
        write_activate_bat(merged, bat)
        print("Wrote %d variables to:" % len(merged))
        print("  %s" % env_local)
        print("  %s" % bat)
        if added:
            print("  new      : %s" % ", ".join(added))
        if kept:
            print("  kept from an earlier scan: %s" % ", ".join(kept))
        print("\nRun `activate_deps.bat` in a new terminal, then start the server.")
        if not merged:
            print("\nNothing was found to configure — see the MISSING rows above.")
        elif args.fast and kept:
            print("\nThis was a --fast scan (repository neighbourhood only). The "
                  "variables above were kept from an earlier full scan; run "
                  "`write` without --fast to search the whole drive again.")
    return 0


def _check_wrappers() -> int:
    """Ask the real wrappers whether they can resolve their tools."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    print("Wrapper resolution, as the pipeline sees it")
    print("-" * 74)
    bad = 0
    checks = [
        ("isRNAcirc CG_to_allatom.exe",
         lambda: __import__("torusfold.scheme2.isrnacirc_wrapper",
                            fromlist=["x"])._CG_TO_AA_EXE or "(unset)"),
        ("RhoFold+ checkpoint",
         lambda: __import__("torusfold.scheme2.rhofold_wrapper",
                            fromlist=["x"])._RHOFOLD_CKPT or "(unset)"),
        ("RNAbpFlow checkpoint",
         lambda: os.path.join(os.environ.get("RNABPFLOW_ROOT", ""),
                              "checkpoint", "RNA3DB.ckpt")),
    ]
    for label, fn in checks:
        try:
            value = fn()
        except Exception as exc:                   # noqa: BLE001
            print("  ERROR  %-28s %s" % (label, exc))
            bad += 1
            continue
        exists = os.path.exists(value) if value and value != "(unset)" else False
        print("  %s %-28s %s" % ("OK   " if exists else "MISS ", label, value))
        if not exists:
            bad += 1
    print()
    print("  %d of %d unresolved" % (bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
