"""lociPARSE quality scores for CG structures.

lociPARSE (Bhattacharya Lab, J. Chem. Inf. Model. 2024, 64(22):8655-8664) predicts a
per-nucleotide lDDT (pNuL) and a molecular lDDT (pMoL) for an RNA 3D structure with no
reference structure required.

Why it fits this pipeline, measured rather than assumed:

  * It builds its local nucleotide frames from exactly P, C4' and the glycosidic N (N9 for
    purines, N1 for pyrimidines) -- see lociPARSE/feature_generation.py:119-123. Those are
    the three beads of this pipeline's CG model.
  * Verified on 2OIU: 3054 atoms -> pMoL 0.80; reduced to those three atoms per residue
    (426 atoms) -> pMoL 0.80, unchanged. It really does use only those atoms, so a CG
    structure is in-distribution for it, not a degraded input.
  * Cost, one CPU core: 142 nt in 0.09 s, 568 nt in 0.26 s. Linear in length, so a 200-230 nt
    chunk costs about 0.1 s.
  * It discriminates on our own material: 1L2X crystal scores 0.64, the CG structure this
    pipeline produced for the same 27-nt sequence scores 0.52. (n = 1; this is a direction,
    not a validated discrimination claim -- the paper's 30-target benchmark is.)

CAVEAT, and it decides how the score may be used: **do not compare pMoL across lengths.**
Three crystals score 0.64 (1L2X, 27 nt), 0.75 (R1108, 69 nt), 0.80 (2OIU, 142 nt). Compare
candidates of the same length, or the same structure before and after a change. A
cross-chunk ranking would be comparing different lengths and is not supported by this.

lociPARSE pins numpy==1.22.3 / torch==1.12.0 in its setup.py, which predate Python 3.11; it
is not installable on this machine's interpreters. The pins are spurious -- the package
imports only torch, numpy and tqdm -- so this module loads it from a source checkout
instead. Point TORUSFOLD_LOCIPARSE at the directory CONTAINING the lociPARSE package.

lociPARSE is GPL-3.0. This module imports it; it does not vendor it. Confirm the licence
question before shipping anything that bundles it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np

_LOCI = None          # cached lociparse() instance
_LOCI_FAILED = False  # so a missing dependency is not retried on every call


def _load():
    """Import and construct lociparse(), or return None. Cached either way."""
    global _LOCI, _LOCI_FAILED
    if _LOCI is not None or _LOCI_FAILED:
        return _LOCI
    root = os.environ.get("TORUSFOLD_LOCIPARSE")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from lociPARSE import lociparse
        _LOCI = lociparse()
    except Exception:
        _LOCI_FAILED = True
        _LOCI = None
    return _LOCI


def available() -> bool:
    """True when lociPARSE can be imported and its weights loaded."""
    return _load() is not None


def _write_pseudo_pdb(path: str, sequence: str, p, c4, n) -> None:
    """One residue per line-group, three atoms: P, C4', and N9 (purine) or N1 (pyrimidine).

    Every other atom of a real structure is absent. That is deliberate: the model uses only
    these three, and 2OIU scores identically with and without the rest.
    """
    lines, serial = [], 1
    purines = ("A", "G")
    for i, base in enumerate(sequence):
        b = {"T": "U", "t": "u"}.get(base, base).upper()
        if b not in "ACGU":
            b = "A"
        nname = "N9" if b in purines else "N1"
        for aname, xyz in (("P", p[i]), ("C4'", c4[i]), (nname, n[i])):
            x, y, z = (float(v) for v in xyz)
            lines.append(f"ATOM  {serial:5d} {aname:<4s}{b:>3s} A{i + 1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
            serial += 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _score_file(path: str) -> Optional[Dict]:
    lp = _load()
    if lp is None:
        return None
    try:
        s = lp.score(path)
        return {"pMoL": float(s.pMoL.value), "pNuL": [float(v) for v in s.pNuL.values]}
    except Exception:
        return None


def score_cg(sequence: str, p, c4, n) -> Optional[Dict]:
    """Score a CG structure from its three bead arrays.

    Args:
        sequence: one-letter sequence, same length as the bead arrays
        p, c4, n: (L, 3) arrays, in whatever length unit the structure is in -- lociPARSE's
            features are distance-based and it is trained on Angstrom PDBs, so pass
            Angstroms unless you have a reason not to.

    Returns {"pMoL": float, "pNuL": list[float]} or None if lociPARSE is unavailable.
    """
    p = np.asarray(p, dtype=float)
    c4 = np.asarray(c4, dtype=float)
    n = np.asarray(n, dtype=float)
    L = len(sequence)
    if not (p.shape == c4.shape == n.shape == (L, 3)):
        raise ValueError(f"bead arrays must all be ({L}, 3); got "
                         f"{p.shape}, {c4.shape}, {n.shape}")
    fd, tmp = tempfile.mkstemp(suffix=".pdb", prefix="lociparse_")
    os.close(fd)
    try:
        _write_pseudo_pdb(tmp, sequence, p, c4, n)
        return _score_file(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def score_pdb(path: str) -> Optional[Dict]:
    """Score an existing PDB file. Returns None if lociPARSE is unavailable."""
    return _score_file(path)
