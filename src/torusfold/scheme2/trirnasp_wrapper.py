"""
TriRNASP ctypes wrapper — 粗粒度 RNA 统计势.

TriRNASP 是 Tan 组开发的 3-bead RNA 统计势 (P/C4'/N).
原始代码: https://github.com/Tan-group/TriRNASP
编译: gcc -shared -o TriRNASP.dll TriRNASP.c -O3

用法:
    wrapper = TriRNASPWrapper("./TriRNASP.dll")
    energy = wrapper.score_3bead(sequence, coords)  # coords: (N, 3, 3) in Angstrom
"""
import os
import ctypes
import numpy as np
from pathlib import Path
from typing import Optional


class TriRNASPWrapper:
    """TriRNASP ctypes wrapper."""

    def __init__(self, lib_path: Optional[str] = None):
        """Load TriRNASP shared library.

        Args:
            lib_path: path to TriRNASP.dll (Linux: .so). If None, auto-detect.
        """
        if lib_path is None:
            # Auto-detect: look for .dll (Windows) or .so (Linux) in external/TriRNASP/
            base = Path(__file__).parent.parent / "external" / "TriRNASP"
            if (base / "TriRNASP.dll").exists():
                lib_path = str(base / "TriRNASP.dll")
            elif (base / "TriRNASP.so").exists():
                lib_path = str(base / "TriRNASP.so")
            else:
                raise FileNotFoundError(
                    f"TriRNASP library not found in {base}. "
                    f"Compile with: gcc -shared -o TriRNASP.dll TriRNASP.c -O3"
                )

        self.lib = ctypes.CDLL(lib_path)

        # Setup function signatures
        # double trirnasp_score_3bead(const char* seq, const double* coords, int n_nt)
        self.lib.trirnasp_score_3bead.restype = ctypes.c_double
        self.lib.trirnasp_score_3bead.argtypes = [
            ctypes.c_char_p,  # sequence
            ctypes.POINTER(ctypes.c_double),  # coords (N*3*3 doubles)
            ctypes.c_int,  # n_nt
        ]

        # void trirnasp_score_3bead_with_gradient(
        #     const char* seq, const double* coords, int n_nt,
        #     double* energy, double* gradient)
        self.lib.trirnasp_score_3bead_with_gradient.restype = None
        self.lib.trirnasp_score_3bead_with_gradient.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),  # energy output
            ctypes.POINTER(ctypes.c_double),  # gradient output
        ]

        print(f"    [TriRNASP] Loaded from {lib_path}")

    def score_3bead(self, sequence: str, coords: np.ndarray) -> float:
        """Compute TriRNASP energy for 3-bead model.

        Args:
            sequence: RNA sequence (ACGU), length N
            coords: (N, 3, 3) array in Angstroms.
                    coords[i, 0, :] = P bead position
                    coords[i, 1, :] = C4' bead position
                    coords[i, 2, :] = N bead position

        Returns:
            energy in kcal/mol
        """
        n_nt = len(sequence)
        assert coords.shape == (n_nt, 3, 3), f"Expected ({n_nt}, 3, 3), got {coords.shape}"

        # Flatten coords to 1D array (C-order)
        coords_flat = coords.flatten().astype(np.float64)
        coords_ptr = coords_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        energy = self.lib.trirnasp_score_3bead(
            sequence.encode('utf-8'),
            coords_ptr,
            n_nt
        )

        return energy

    def score_3bead_with_gradient(self, sequence: str, coords: np.ndarray):
        """Compute TriRNASP energy + gradient (force) for 3-bead model.

        Args:
            sequence: RNA sequence (ACGU), length N
            coords: (N, 3, 3) array in Angstroms

        Returns:
            (energy, gradient) tuple. gradient shape: (N, 3, 3) in kcal/mol/Å
        """
        n_nt = len(sequence)
        assert coords.shape == (n_nt, 3, 3)

        coords_flat = coords.flatten().astype(np.float64)
        coords_ptr = coords_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        energy_out = ctypes.c_double(0.0)
        energy_ptr = ctypes.pointer(energy_out)

        grad_flat = np.zeros(n_nt * 3 * 3, dtype=np.float64)
        grad_ptr = grad_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        self.lib.trirnasp_score_3bead_with_gradient(
            sequence.encode('utf-8'),
            coords_ptr,
            n_nt,
            energy_ptr,
            grad_ptr
        )

        gradient = grad_flat.reshape((n_nt, 3, 3))

        return energy_out.value, gradient


# ─────────────────────────────────────────────────────────────
# Test: if run as script, compile and test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    lib_path = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        wrapper = TriRNASPWrapper(lib_path)

        # Test with random 10-nt RNA
        seq = "AUGCAUGCAU"
        coords = np.random.rand(10, 3, 3) * 10.0  # random coords in [0, 10] Å

        energy = wrapper.score_3bead(seq, coords)
        print(f"Test: E = {energy:.3f} kcal/mol")

        energy, grad = wrapper.score_3bead_with_gradient(seq, coords)
        print(f"Test: E = {energy:.3f} kcal/mol, |grad| = {np.linalg.norm(grad):.3f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo compile TriRNASP:")
        print("  cd external/TriRNASP")
        print("  gcc -shared -o TriRNASP.dll TriRNASP.c -O3 -lm")
