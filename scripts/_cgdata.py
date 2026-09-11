"""Where the bundled data lives, resolved once so every caller agrees.

Two datasets ship inside the working folder, because a copied directory has to be self-contained
and the tools that read them were originally pointed at an absolute Windows path outside it:

    _cgdata/rsRNASP/Training_set     191 PDB files, the deposited-structure database
    _cgdata/cgRNASP                  the cgRNASP reference implementation's own tables

Resolution order, for each of them:

    1. the environment variable TORUSFOLD_RSRNASP / TORUSFOLD_CGRNASP, if set;
    2. the copy inside this repository, if it exists;
    3. the original absolute path, so the machine every measurement was taken on keeps working
       with no environment at all.

The third fallback is deliberately last and deliberately kept: every number in
docs/statistical_potentials_as_forces.md was produced by reading path 3, and a resolver that
silently preferred something else would put those measurements and the code that cites them on
different data.

This module exists because the same literal had been copied into seventeen scripts. Importing a
resolver is the fix; seventeen literals is the bug.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RSRNASP_ENV = "TORUSFOLD_RSRNASP"
CGRNASP_ENV = "TORUSFOLD_CGRNASP"


def _resolve(env, bundled, original):
    override = os.environ.get(env)
    if override:
        return Path(override)
    if bundled.is_dir():
        return bundled
    return Path(original)


def rsrnasp():
    """The deposited-structure database, 191 PDB files."""
    return _resolve(RSRNASP_ENV, REPO / "_cgdata" / "rsRNASP" / "Training_set",
                    r"D:\torusfold-cgdata\rsRNASP\Training_set")


def cgrnasp():
    """The cgRNASP reference implementation's data directory."""
    return _resolve(CGRNASP_ENV, REPO / "_cgdata" / "cgRNASP",
                    r"D:\torusfold-cgdata\cgRNASP\cgRNASP\data")
