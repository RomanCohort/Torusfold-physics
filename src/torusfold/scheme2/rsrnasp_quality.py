"""rsRNASP1 quality score for all-atom RNA structures.

rsRNASP1 (Lou, Zheng, Yu, Tan & Tan, Biophys J 2025, 124:2740-2753) is a
distance- and dihedral-dependent statistical potential with residue separation,
used here as an absolute quality check on the final all-atom structure.

Why this module exists at all: the interface has always shown an `rsRNASP1` row
and the pipeline has always written `"rsrnasp1": None` into its result, so the row
could never display anything. The scorer is now actually run.

**Lower (more negative) is better.** This is a raw statistical potential, not a
0-1 score, so it is only meaningful against other structures of the same length —
the same caveat that applies to lociPARSE's pMoL. Two reference points measured
with this build:

    1a9nR (crystal, 27 nt)   -3146.58
    1h4sT (crystal, 61 nt)   -7757.55
    this pipeline's 139 nt prediction from a 0.40-confidence ensemble  +2289.27

The positive score on our own output is the honest reading: the structure is a
low-confidence prediction, and the potential says so. It is not a calibration
error.

Requirements, and why they are unusual
--------------------------------------
rsRNASP1 is C++ with a Makefile and ships **Linux binaries only** — no Windows
build, no pip package. On Windows it is run through WSL, which is why this module
shells out instead of importing anything.

    rsRNASP1 compiled under WSL, e.g.
        git clone https://github.com/Tan-group/rsRNASP1 && cd rsRNASP1 && make
    TORUSFOLD_RSRNASP set to that directory as a WSL path (default
        /home/<user>/tools/rsRNASP1 is also tried), and its energy files either at
        <root>/data/energyfiles or wherever rsRNASP_RNA_HOME points.

Returns None, never raises, when any of that is missing: a quality score is not
worth failing a completed prediction over. `available()` reports which piece is
absent so the dependency report can say so precisely.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Optional

# The energy files this build needs, as the binary looks for them.
#
# Joined with "/" rather than os.path.join: these are POSIX paths inside WSL, and
# os.path.join on Windows would produce "/home/u/tools/rsRNASP1\data\..." — a path
# that does not exist. Nothing here is opened by this process, only by bash.
_REQUIRED_DATA = (
    "data/energyfiles/dist/nonlocal.dat",
    "data/energyfiles/dist/local.dat",
)
_BINARY_REL = "bin/rsRNASP1"


def _pjoin(*parts: str) -> str:
    """Join POSIX path components for use inside WSL."""
    cleaned = [p.strip("/") for p in parts if p]
    if not cleaned:
        return "/"
    head = parts[0]
    joined = "/".join(cleaned)
    return "/" + joined if head.startswith("/") else joined

_DEFAULT_ROOT = "~/tools/rsRNASP1"
_probe_cache: Optional[Dict] = None


def _wsl_distro() -> Optional[str]:
    """A usable WSL distribution, or None.

    Prefers whatever WSL reports as default, because that is where a user will
    have compiled rsRNASP1.
    """
    try:
        p = subprocess.run(["wsl.exe", "-e", "bash", "-lc", "echo $WSL_DISTRO_NAME"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    name = (p.stdout or "").strip().splitlines()[-1:] or [""]
    return name[0] or None


def _run_bash(command: str, timeout: float = 300.0):
    """Run one bash command inside WSL. Returns (ok, stdout+stderr)."""
    try:
        p = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out after %.0fs" % timeout
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)[:200]
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    return p.returncode == 0, out


def _root_hint() -> str:
    return (os.environ.get("TORUSFOLD_RSRNASP", "").strip() or _DEFAULT_ROOT)


def _expand_in_wsl(path: str) -> str:
    """Expand a leading ~ using WSL's own HOME.

    Not expanded by the shell here: the root is quoted into the command string, and
    a quoted `~` is literal, so `~/tools/rsRNASP1` failed to match a directory that
    existed. On Windows `os.path.expanduser` would expand it to the *Windows* home
    directory instead, which WSL cannot see.
    """
    if not path.startswith("~"):
        return path
    ok, out = _run_bash("echo $HOME", timeout=60)
    home = (out.strip().splitlines()[-1] if ok and out.strip() else "")
    if not home:
        return path
    return home + path[1:]


def _to_wsl_path(win_path: str) -> Optional[str]:
    """Convert a Windows path to its /mnt/... form, or None.

    Done with wslpath rather than by hand so that a path on a drive WSL does not
    have mounted fails here, loudly, instead of producing a plausible path that
    the binary then cannot open.
    """
    ok, out = _run_bash('wslpath -a -u "%s"' % win_path.replace('"', '\\"'), timeout=60)
    if not ok or not out:
        return None
    line = out.splitlines()[-1].strip()
    return line or None


def _probe() -> Dict:
    """Is the whole chain present? Cached: this runs on every report and poll."""
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    result = {"available": False, "why": "", "root": None, "distro": None}
    if os.name != "nt":
        result["why"] = "not Windows; run the binary directly"
        _probe_cache = result
        return result

    distro = _wsl_distro()
    if not distro:
        result["why"] = "WSL is not available (rsRNASP1 ships Linux builds only)"
        _probe_cache = result
        return result
    result["distro"] = distro

    root = _expand_in_wsl(_root_hint())
    # The binary and the energy files are checked inside WSL, where they are used.
    wanted = [_BINARY_REL] + list(_REQUIRED_DATA)
    checks = " && ".join('test -e "%s"' % _pjoin(root, rel) for rel in wanted)
    ok, _out = _run_bash("set -e; " + checks, timeout=90)
    if not ok:
        result["why"] = ("rsRNASP1 not found at %s (needs bin/rsRNASP1 and "
                         "data/energyfiles). Build it under WSL, or point "
                         "TORUSFOLD_RSRNASP at it." % root)
        _probe_cache = result
        return result

    result["available"] = True
    result["root"] = root
    _probe_cache = result
    return result


def available() -> Dict:
    """Whether this scorer can run, and why not if it cannot."""
    return dict(_probe())


def _parse_score(text: str) -> Optional[float]:
    """Pull the score out of the binary's output.

    It prints one `<path> <score>` line per input, e.g.
        example/1a9nR.pdb -3146.575662
    """
    for line in reversed(text.splitlines()):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return float(parts[-1])
        except ValueError:
            continue
    return None


def score_pdb(pdb_path: str) -> Optional[float]:
    """rsRNASP1 score for one PDB, or None if it cannot be computed.

    Needs a full-atom structure with canonical RNA atom names. A coarse-grained
    P-only file is not scoreable and will produce a nonsense number rather than an
    error, so callers must pass the all-atom result.
    """
    if not pdb_path or not os.path.isfile(pdb_path):
        return None
    probe = _probe()
    if not probe["available"]:
        return None

    wsl_pdb = _to_wsl_path(os.path.abspath(pdb_path))
    if not wsl_pdb:
        return None

    root = probe["root"]
    # rsRNASP_RNA_HOME, NOT the documented -d flag.
    #
    # `-d` looks like the clean way to pass the energy-file directory, but this
    # build aborts on it: "terminate called after throwing an instance of
    # 'std::logic_error' what(): basic_string: construction from null is not
    # valid", exit 134. Setting the environment variable gives the correct score.
    # Exporting it also means the answer does not depend on the user's shell.
    cmd = ('export rsRNASP_RNA_HOME=%s; exec %s "%s"'
           % (root, _pjoin(root, _BINARY_REL), wsl_pdb))
    ok, out = _run_bash(cmd, timeout=600)
    if not ok:
        return None
    return _parse_score(out)


def score_batch(pdb_paths: List[str]) -> Dict[str, Optional[float]]:
    """Scores for several PDBs, keyed by the path given.

    One binary invocation for all of them: process start dominates the cost of a
    score (0.3 s for 139 nt), so a per-file call would be nearly all overhead.
    """
    out: Dict[str, Optional[float]] = {p: None for p in pdb_paths}
    probe = _probe()
    if not probe["available"]:
        return out
    root = probe["root"]
    resolved = {}
    for path in pdb_paths:
        wsl_path = _to_wsl_path(os.path.abspath(path))
        if wsl_path:
            resolved[wsl_path] = path
    if not resolved:
        return out
    # Same environment-variable approach as score_pdb, for the same reason: -d aborts.
    cmd = 'export rsRNASP_RNA_HOME=%s; exec %s %s' % (
        root, _pjoin(root, _BINARY_REL),
        " ".join('"%s"' % p for p in resolved))
    ok, text = _run_bash(cmd, timeout=900)
    if not ok:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        key = " ".join(parts[:-1])
        if key in resolved:
            out[resolved[key]] = value
    return out
