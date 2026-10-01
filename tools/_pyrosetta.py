# -*- coding: utf-8 -*-
"""Detect PyRosetta, and say plainly that it needs a licence.

PyRosetta is not a dependency that can be fetched. It is distributed by the Gray
lab under a licence that has to be obtained and, for anything other than
non-commercial academic use, paid for. It is also not a Python package on PyPI and
not on conda-forge, so there is no install command this repository could run even
if it wanted to.

It runs on the Linux side of WSL (`wsl.exe ... python3 -c "import pyrosetta"`), so
the path is a POSIX path inside the distribution and the usual Windows filesystem
probes do not apply. Detection therefore asks WSL directly, the same way rsRNASP1
is detected.

Used by tools/configure_deps.py and tools/install_deps.py so the licence
requirement appears in the reports a user actually runs, rather than only in a
document.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Optional

# Shown wherever PyRosetta is reported. Short, and states the actionable part: an
# unlicensed copy is not a configuration problem, and this repository does not
# ship or fetch one.
LICENCE_NOTE = (
    "PyRosetta is licensed software, not a free download. Get it from "
    "https://www.pyrosetta.org (academic/non-commercial use is free but still "
    "requires a licence key you obtain yourself; commercial use requires a paid "
    "licence). This repository neither bundles nor fetches it. Level 2.6 is "
    "skipped, not failed, when it is absent."
)

# The import that proves it is usable. Importing PyRosetta is expensive (it loads
# the Rosetta libraries), so the probe is cached and only run when a report asks.
_IMPORT = "import pyrosetta; print(pyrosetta.__file__)"


def _run_in_wsl(command: str, timeout: float = 180.0):
    """Run one bash command inside WSL. Returns (ok, output)."""
    try:
        p = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)[:200]
    return p.returncode == 0, ((p.stdout or "") + (p.stderr or "")).strip()


def _distro() -> Optional[str]:
    ok, out = _run_in_wsl("echo $WSL_DISTRO_NAME", timeout=60)
    if not ok or not out:
        return None
    return (out.strip().splitlines() or [""])[-1] or None


def _candidate_pythons() -> list:
    """Interpreters to try inside WSL, most likely first.

    A --user install lands in the distribution's own python3, not in a conda
    environment, and that is the case on this machine. Conda environments are still
    covered by a glob, because a user who installed PyRosetta with conda would have
    it there.
    """
    return [
        "python3",
        '"$HOME/miniforge3/bin/python"',
        '"$HOME/miniforge3/envs/pyrosetta/bin/python"',
        # Any conda env, without needing to know its name.
        '"$(ls -d "$HOME"/miniforge3/envs/*/bin/python 2>/dev/null | head -1)"',
    ]


_probe_cache: Optional[Dict] = None


def probe(refresh: bool = False) -> Dict:
    """Is PyRosetta usable, where from, and what are the licence terms.

    Cached: the import is slow and the answer does not change while the server
    runs. `available` is False with a stated `why` when it cannot be found, so a
    report can print the reason rather than a bare "missing".
    """
    global _probe_cache
    if _probe_cache is not None and not refresh:
        return dict(_probe_cache)

    result = {"available": False, "why": "", "path": None, "distro": None,
              "licence": LICENCE_NOTE}

    if os.name != "nt":
        result["why"] = "not Windows; probe PyRosetta directly"
        _probe_cache = result
        return dict(result)

    distro = _distro()
    if not distro:
        result["why"] = "WSL is not available (PyRosetta runs on the Linux side)"
        _probe_cache = result
        return dict(result)
    result["distro"] = distro

    for py in _candidate_pythons():
        ok, out = _run_in_wsl('%s -c %s' % (py, _shell_quote(_IMPORT)), timeout=240)
        if ok and out and "pyrosetta" in out.lower():
            result["available"] = True
            result["path"] = out.strip().splitlines()[-1]
            result["interpreter"] = py
            break
    if not result["available"]:
        result["why"] = ("PyRosetta does not import from python3 or from a conda "
                         "environment in %s. Install it per your licence: %s"
                         % (distro, "https://www.pyrosetta.org"))

    _probe_cache = result
    return dict(result)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def status_line() -> str:
    """One line for a startup banner or a report."""
    p = probe()
    if p["available"]:
        return "ok        %s" % (p.get("path") or "")
    return "%-9s %s" % ("not found", p.get("why", ""))
