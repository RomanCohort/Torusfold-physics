# -*- coding: utf-8 -*-
"""trrna2_wrapper.py — trRosettaRNA2 inference via subprocess.

trRNA2 needs a dedicated Python environment (CPU-only, to avoid ROCm MIOpen
crashes); it is invoked through the _trrna2_runner.py subprocess.
"""
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


_RUNNER = os.environ.get("TRRNA2_RUNNER", "")
_TRRNA2_PYTHON = sys.executable  # use the current interpreter; the runner forces CPU internally


@dataclass
class TrRNA2Result:
    coords: object  # (L, 3) numpy array
    dist: object    # (L, L) distance matrix
    confidence: float


def trrna2_predict_chunk(
    sequence: str,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    num_recycles: int = 1,
    verbose: bool = False,
) -> TrRNA2Result:
    """Predict 3D structure using trRosettaRNA2 via subprocess.

    Args:
        sequence: RNA sequence
        output_dir: output directory
        name: output name prefix
        num_recycles: number of recycling iterations
        verbose: print progress

    Returns:
        TrRNA2Result with coords and confidence
    """
    if not os.path.exists(_RUNNER):
        raise FileNotFoundError(f"trRNA2 runner not found: {_RUNNER}")

    # Create temp MSA file (single sequence = pseudo MSA)
    # Unique suffix: when several chunks run in parallel, a shared tmp_msa name
    # would be truncated by other processes, so parse_a3m crashes on a half-written
    # or empty file. A uuid keeps the file private to each process.
    import uuid
    _uniq = f"{uuid.uuid4().hex[:8]}"
    tmp_msa = os.path.join(
        tempfile.gettempdir(),
        f"trrna2_{name or 'chunk'}_{os.getpid()}_{_uniq}.msa",
    )
    with open(tmp_msa, "w") as f:
        f.write(f">seq\n{sequence}\n")

    out_base = os.path.join(
        output_dir or tempfile.gettempdir(),
        f"{name or 'trrna2_out'}_{os.getpid()}_{_uniq}"
    )

    try:
        cmd = [
            _TRRNA2_PYTHON, _RUNNER,
            "--msa", tmp_msa,
            "--out", out_base,
            "--num-recycles", str(num_recycles),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=None,
        )

        if result.returncode != 0:
            raise RuntimeError(f"trRNA2 failed: {result.stderr[:300]}")

        # Load outputs
        import numpy as np
        coords_path = out_base + "_coords.npy"
        dist_path = out_base + "_dist.npy"

        coords = None
        dist = None
        if os.path.exists(coords_path):
            coords = np.load(coords_path)  # (N, 3) all-atom
            # Extract P atoms (every 23rd)
            L = len(sequence)
            if coords.shape[0] >= L:
                atoms_per_res = coords.shape[0] // L
                coords = coords[::atoms_per_res][:L].astype(np.float32)
        if os.path.exists(dist_path):
            dist = np.load(dist_path)

        # Confidence from distance matrix quality
        confidence = 0.5
        if dist is not None:
            # Simple confidence: how many strong distances
            strong = np.sum(dist < 15.0) / (dist.shape[0] * dist.shape[1])
            confidence = min(0.9, max(0.1, strong))

        return TrRNA2Result(coords=coords, dist=dist, confidence=confidence)

    finally:
        # Cleanup temp MSA
        if os.path.exists(tmp_msa):
            os.unlink(tmp_msa)
