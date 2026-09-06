"""
isrnacirc_wrapper.py — Python wrapper for isRNAcirc binaries.

Wraps CG_to_allatom.exe to convert CG P-only PDB to all-atom PDB.
"""
import os
import subprocess
import tempfile
from pathlib import Path


# isRNAcirc standalone root
# CG_to_allatom.exe does not support non-ASCII paths (GBK-encoded, native Windows exe)
# exe + DLL must stay together under an ASCII-only path (Windows DLL loader rejects non-ASCII paths)
# ISRNACIRC_BIN_DIR env var points to an ASCII-only directory holding CG_to_allatom.exe + its DLLs
_ISRNACIRC_BIN_DIR = os.environ.get("ISRNACIRC_BIN_DIR", "")

# exe: looked up in the env-specified directory (blank when unset; cg_to_allatom() gives a clear error)
_CG_TO_AA_EXE = os.path.join(_ISRNACIRC_BIN_DIR, "CG_to_allatom.exe") \
    if _ISRNACIRC_BIN_DIR else ""

# coeff: must be an ASCII-only path; the exe rejects non-ASCII paths
_COEFF_DIR = os.environ.get("CG_TO_ALLATOM_COEFF", "")


def _write_cg_pdb(coords_A, sequence, output_path):
    """Write CG P-only PDB (one P atom per residue).

    CG_to_allatom.exe expects P-only PDB format:
    ATOM serial P resname chain resid x y z

    Args:
        coords_A: (L, 3) P coordinates in Angstroms
        sequence: RNA sequence string (ACGU)
        output_path: output PDB file path
    """
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    with open(output_path, "w") as f:
        f.write("HEADER    CG P-only structure\n")
        for i, (coord, base) in enumerate(zip(coords_A, sequence)):
            resname = base_map.get(base.upper(), "UNK")
            x, y, z = coord
            f.write(
                f"ATOM  {i+1:5d}  P   {resname} A{i+1:4d}"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n"
            )
        f.write("END\n")


def cg_to_allatom(cg_pdb_path, output_pdb_path, sequence=None):
    """Convert CG P-only PDB to all-atom PDB using CG_to_allatom.exe.

    Args:
        cg_pdb_path: input CG PDB (P atoms only)
        output_pdb_path: output all-atom PDB path
        sequence: optional, not used by exe but kept for API compat

    Returns:
        output_pdb_path on success, raises on failure
    """
    if not os.path.exists(_CG_TO_AA_EXE):
        raise FileNotFoundError(
            f"CG_to_allatom.exe not found at {_CG_TO_AA_EXE}. "
            f"Set ISRNACIRC_ROOT env var to isRNAcirc standalone root."
        )

    cmd = [_CG_TO_AA_EXE, cg_pdb_path, output_pdb_path, _COEFF_DIR]
    # DLL search: the bin/ folder holds MSVC/FFTW runtime DLLs, which need to be on PATH
    _bin_dir = os.path.dirname(_CG_TO_AA_EXE)
    _env = os.environ.copy()
    _existing_path = _env.get("PATH", "")
    _env["PATH"] = _bin_dir + os.pathsep + _ISRNACIRC_BIN + os.pathsep + _existing_path
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=None, env=_env,
        cwd=_bin_dir if os.path.isdir(_bin_dir) else None,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CG_to_allatom failed (ret={result.returncode}): "
            f"{result.stderr.strip()}"
        )

    if not os.path.exists(output_pdb_path):
        raise FileNotFoundError(
            f"CG_to_allatom produced no output: {output_pdb_path}"
        )

    return output_pdb_path


def isrnacirc_refine(cg_pdb_path, output_dir, sequence,
                     nstep=10000, nstep_close=1000, nstru=10):
    """Run isRNAcirc.exe for CG refinement (Type=1, user-provided PDB).

    Args:
        cg_pdb_path: input CG PDB
        output_dir: output directory
        sequence: RNA sequence
        nstep: MD steps
        nstep_close: close-range MD steps
        nstru: number of output structures

    Returns:
        path to best output PDB
    """
    isrnacirc_exe = os.path.join(_ISRNACIRC_ROOT, "bin", "IsRNAcirc.exe")
    if not os.path.exists(isrnacirc_exe):
        raise FileNotFoundError(f"IsRNAcirc.exe not found: {isrnacirc_exe}")

    os.makedirs(output_dir, exist_ok=True)

    # Write .2d file
    ss = "." * len(sequence)  # placeholder
    dotbracket_file = os.path.join(output_dir, "input.2d")
    with open(dotbracket_file, "w") as f:
        f.write(f"{sequence}\n")
        f.write(f"{ss} 0\n")

    # Write config.txt
    config_file = os.path.join(output_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write(f"Nstep {nstep}\n")
        f.write(f"Nstep_close {nstep_close}\n")
        f.write(f"Nstru {nstru}\n")
        f.write("PS 0.3\nPE 0.9\nRMSD_Cut 7.5\nNout 1\n")

    # Run
    cmd = [isrnacirc_exe, _ISRNACIRC_ROOT + "/Data/",
           dotbracket_file, output_dir, "output", config_file,
           "1", cg_pdb_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)

    if result.returncode != 0:
        raise RuntimeError(f"IsRNAcirc failed: {result.stderr.strip()}")

    # Find output PDB
    for f in sorted(Path(output_dir).glob("*.pdb"), key=os.path.getmtime, reverse=True):
        return str(f)
    raise FileNotFoundError(f"No output PDB in {output_dir}")
