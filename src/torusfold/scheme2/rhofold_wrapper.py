"""
rhofold_wrapper.py — RhoFold+ RNA 3D structure prediction wrapper.

Global model cache: the model is loaded on the first call and reused afterwards
(the 11 chunks load it only once).
"""
import os
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

_RHOFOLD_ROOT = os.environ.get("RHOFOLD_ROOT", "")
_RHOFOLD_CKPT = os.path.join(_RHOFOLD_ROOT, "pretrained", "rhofold_pretrained_params.pt")

# ── Global model cache ──
_cached_model = None
_cached_device = None


def _get_model(device: str = "auto"):
    """Load and cache the RhoFold+ model (loaded only once)."""
    global _cached_model, _cached_device
    import torch
    import sys

    if _cached_model is not None and _cached_device == device:
        return _cached_model

    if _RHOFOLD_ROOT not in sys.path:
        sys.path.insert(0, _RHOFOLD_ROOT)

    from rhofold.rhofold import RhoFold
    from rhofold.config import rhofold_config

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model = RhoFold(rhofold_config)
    ckpt = torch.load(_RHOFOLD_CKPT, map_location=device_obj)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device_obj)

    _cached_model = model
    _cached_device = device
    return model


def rhofold_predict_chunk(
    sequence: str,
    secondary_structure: Optional[str] = None,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    msa_path: Optional[str] = None,
    verbose: bool = False,
    device: str = "auto",
    boundary_pairs: Optional[List[Tuple[int, int, str]]] = None,
) -> np.ndarray:
    """Predict 3D structure using RhoFold+.

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket)
        output_dir: output directory for PDB
        name: output name
        msa_path: MSA file path (optional)
        verbose: print details
        device: device string ("auto", "cuda", "cpu")
        boundary_pairs: list of Level 1 boundary-constraint pairs
            [(global_i, global_j, edge_type)]. When provided, distance-constraint
            relaxation is applied with OpenMM after the RhoFold prediction.

    Returns: (coords, confidence) tuple.
        coords: (L, 3) C1' coordinates in Angstroms
        confidence: mean pLDDT [0, 1]
    Raises on failure.
    """
    import torch
    import tempfile

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _get_model(device)
    device_obj = torch.device(device)

    from rhofold.utils.alphabet import get_features

    # Prepare FASTA + MSA files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        f.write(f">seq\n{sequence}\n")
        fas_path = f.name

    if msa_path and os.path.exists(msa_path):
        msa_file = msa_path
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".a3m", delete=False) as f:
            f.write(f">seq\n{sequence}\n")
            msa_file = f.name

    try:
        data_dict = get_features(fas_path, msa_file)
    finally:
        os.unlink(fas_path)
        if msa_file != msa_path:
            os.unlink(msa_file)

    # Move to device
    tokens = data_dict["tokens"].to(device_obj)
    rna_fm_tokens = data_dict["rna_fm_tokens"].to(device_obj)
    seq = data_dict["seq"]

    # Forward pass
    with torch.no_grad():
        outputs = model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq)

    output = outputs[-1]

    # Extract per-residue coordinates
    L = len(sequence)
    c1_coords = output["cords_c1'"]
    if isinstance(c1_coords, list) and len(c1_coords) > 0:
        coords = c1_coords[-1].squeeze(0).cpu().numpy()[:L].astype(np.float32)
    else:
        all_coords = output["cord_tns_pred"][0].squeeze(0).cpu().numpy()
        atoms_per_res = all_coords.shape[0] // L
        coords = all_coords[::atoms_per_res][:L].astype(np.float32) if atoms_per_res > 0 else all_coords[:L].astype(np.float32)

    # Extract confidence from plddt
    plddt = output.get("plddt", None)
    if plddt is not None and isinstance(plddt, (tuple, list)) and len(plddt) > 0:
        confidence = float(plddt[0].squeeze().mean())
    else:
        confidence = 0.5

    # Level 1 boundary-constraint relaxation
    if boundary_pairs:
        try:
            from .boundary_constraints import apply_boundary_constraints_to_coords
            coords = apply_boundary_constraints_to_coords(coords, boundary_pairs)
            if verbose:
                print(f"  [Boundary] Applied {len(boundary_pairs)} boundary constraints")
        except Exception as e:
            if verbose:
                print(f"  [Boundary] Constraint relaxation failed: {e}")

    # Save PDB if requested
    if output_dir and name:
        os.makedirs(output_dir, exist_ok=True)
        pdb_path = os.path.join(output_dir, f"{name}.pdb")
        _write_pdb(coords, sequence, pdb_path)

    return coords, confidence


def _write_pdb(coords, sequence, output_path):
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    with open(output_path, "w") as f:
        f.write("HEADER    RhoFold+ prediction\n")
        for i, (coord, base) in enumerate(zip(coords, sequence)):
            resname = base_map.get(base.upper(), "UNK")
            x, y, z = coord
            f.write(
                f"ATOM  {i+1:5d}  P   {resname} A{i+1:4d}"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n"
            )
        f.write("END\n")
