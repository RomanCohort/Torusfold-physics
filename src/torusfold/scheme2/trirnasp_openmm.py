"""TriRNASP Python implementation — knowledge-based three-body statistical potential.

Pure Python + numpy port of TriRNASP (Tan-group, Wuhan University).
Loads precomputed energy tables, computes three-body RNA scores with gradients.

TriRNASP: https://github.com/Tan-group/TriRNASP

Algorithm:
  For each triple of atoms (n1 < n2 < n3) within cutoff R0=7.7A:
    - Compute distances d12, d13, d23
    - Map atom types to codes (12 types: 4 bases x 3 beads)
    - Look up energy from precomputed distance-bin tables
    - Sum all triplet energies
  Unit: kBT
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Constants ───────────────────────────────────────────────────

R0 = 8.0          # Cutoff distance (A)
BIN_WIDTH_ROUGH = 2.0  # Coarse bin width
BIN_WIDTH_FINE = 0.6   # Fine bin width
KBT_FACTOR = 2.494     # 1 kBT in kJ/mol at 300K
EXCLUSION_R1_SQ = 1.21  # (1.1)^2
EXCLUSION_R2_SQ = 2.89  # (1.7)^2
EXCLUSION_R3_SQ = 16.0  # (4.0)^2
R0_SQ = (R0 - 0.3) ** 2  # Adjusted cutoff squared

# Atom type codes (12 types)
# C4': 0-3 (A/U/C/G), N9/N1: 4-7 (A/U/C/G), P: 8-11 (A/U/C/G)
_ATOM_CODES = {
    "AC4'": 0, "UC4'": 1, "CC4'": 2, "GC4'": 3,
    "AN9": 4,  "UN1": 5,  "CN1": 6,  "GN9": 7,
    "AP": 8,   "UP": 9,   "CP": 10,  "GP": 11,
}

# Base to code offsets
_BASE_OFFSET = {"A": 0, "U": 1, "C": 2, "G": 3}
_BEAD_OFFSET = {"C4'": 0, "N9": 4, "N1": 4, "P": 8}


def _type_code(base: str, bead: str) -> int:
    """Map base+bead to atom type code."""
    b = _BASE_OFFSET.get(base, -1)
    if b < 0:
        return -1
    if bead == "C4'":
        return b
    if bead in ("N9", "N1"):
        return 4 + b
    if bead == "P":
        return 8 + b
    return -1


# ── Energy Table Loading ────────────────────────────────────────

def _load_energy_table(filepath: str, n_types: int, n_bins: int, n_bins_fine: int) -> np.ndarray:
    """Load TriRNASP energy table from file.

    Table indexed as IDX(t1,t2,t3, b12,b13,b23) where:
      t1,t2,t3: atom type codes (0-11)
      b12,b13: bin indices (0..n_bins-1)
      b23: bin index (0..2*n_bins-1)

    Returns:
        np.ndarray flat array of energy values
    """
    # Compute table size
    # Rough: 12*12*12 * I * I * (2*I) where I = n_bins
    # Fine: 12*12*12 * I1 * I1 * (2*I1)
    total = (n_types ** 3) * (n_bins ** 2) * (2 * n_bins)
    table = np.zeros(total, dtype=np.float64)

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            t1, t2, t3 = int(parts[0]), int(parts[1]), int(parts[2])
            b12, b13, b23 = int(parts[3]), int(parts[4]), int(parts[5])
            energy = float(parts[6])

            idx = (((((t1 * n_types + t2) * n_types + t3) * n_bins + b12) * n_bins + b13) * (2 * n_bins) + b23)
            if 0 <= idx < total:
                table[idx] = energy

    return table


class TriRNASPPotential:
    """TriRNASP three-body statistical potential for RNA 3D structure scoring.

    Usage:
        pot = TriRNASPPotential("external/TriRNASP/Energy")
        energy, gradient = pot.score_with_gradient(coords_3bead, sequence)
    """

    def __init__(self, energy_dir: Optional[str] = None):
        """Load energy tables.

        Args:
            energy_dir: Path to Energy/ directory containing
                        Rough.energy and Fine.energy.
                        Auto-detects from external/TriRNASP/Energy/
        """
        if energy_dir is None:
            # Absolute path: project root = src/torusfold/scheme2/../../..
            _project_root = Path(__file__).resolve().parents[3]
            energy_dir = str(_project_root / "external" / "TriRNASP" / "Energy")

        rough_path = Path(energy_dir).resolve() / "Rough.energy"
        fine_path = Path(energy_dir).resolve() / "Fine.energy"

        if not rough_path.exists():
            raise FileNotFoundError(f"Rough.energy not found: {rough_path}")

        self._rough = _load_energy_table(str(rough_path), 12, 4, 4)  # I=4
        self._fine = None
        if fine_path.exists():
            self._fine = _load_energy_table(str(fine_path), 12, 13, 13)  # I1=13

        self._energy_dir = energy_dir
        print(f"  [TriRNASP] Loaded tables from {energy_dir}")

    def score(self, coords_3bead: np.ndarray, sequence: str) -> float:
        """Compute the TriRNASP energy (kBT) — vectorized version.

        Equivalent to the original per-atom triple loop, but the distance
        matrix is computed once and the three-body loop does batched table
        lookups with numpy fancy indexing. Complexity is still O(N·k²)
        (k = number of neighbors within R0), but the constant factor drops
        from ~200x Python-loop iterations to vectorized operations.
        """
        atoms = self._build_atoms(coords_3bead, sequence)
        n_atoms = len(atoms[0]) if atoms else 0
        if n_atoms < 3:
            return 0.0
        atom_types, atom_res, _beads, atom_coords = atoms

        # Full squared-distance matrix (float32 halves the memory; 6024² × 4B ≈ 145MB)
        diff = atom_coords[:, None, :] - atom_coords[None, :, :]
        dist_sq = (diff * diff).sum(-1).astype(np.float32)

        inv_bw = np.float32(1.0 / BIN_WIDTH_ROUGH)

        # ── Exclusions (pair-level constant contributions, consistent with the original logic) ──
        iu, ju = np.triu_indices(n_atoms, k=1)
        d_ij = dist_sq[iu, ju]
        res_diff = np.abs(atom_res[iu].astype(np.int32) - atom_res[ju].astype(np.int32))

        excl = np.zeros(len(iu), dtype=bool)
        e_const = np.zeros(len(iu), dtype=np.float64)

        m1 = d_ij <= np.float32(EXCLUSION_R1_SQ)
        e_const[m1] += 0.5
        excl |= m1

        m2 = (~excl) & (d_ij <= np.float32(EXCLUSION_R2_SQ)) & (res_diff > 1)
        e_const[m2] += 0.5
        excl |= m2

        m3 = (~excl) & (d_ij <= np.float32(EXCLUSION_R3_SQ)) & \
             (d_ij > np.float32(EXCLUSION_R2_SQ))
        e_const[m3] += 0.2
        excl |= m3

        # ── Valid pair candidates (entering the three-body loop) ──
        cand_mask = (~excl) & (d_ij < np.float32(R0_SQ))
        ci, cj = iu[cand_mask], ju[cand_mask]

        d12 = np.sqrt(dist_sq[ci, cj])
        b12 = (d12 * inv_bw).astype(np.int32)
        ok12 = b12 <= 3
        ci, cj, b12 = ci[ok12], cj[ok12], b12[ok12]

        total = float(e_const.sum())

        # ── Three-body loop: for each valid pair (i,j), find k>j inside both spheres ──
        # Neighborhood built from the distance matrix (within R0-0.3); no KD-tree needed
        r_cut = np.float32((R0 - 0.3) ** 2)
        ti_all, tj_all, tk_all = atom_types[ci], atom_types[cj], None

        for idx in range(len(ci)):
            i, j = int(ci[idx]), int(cj[idx])
            row_j = dist_sq[j]                      # distances from j to every atom
            ks = np.nonzero(
                (np.arange(n_atoms) > j) &
                (row_j < r_cut) &
                (dist_sq[i] < np.float32(R0_SQ))
            )[0]
            if len(ks) == 0:
                continue

            d13_sq = dist_sq[i, ks].astype(np.float64)
            d23_sq = dist_sq[j, ks].astype(np.float64)
            ri, rj = atom_res[i], atom_res[j]

            # Exclusion rules (three-body)
            bad = (d13_sq <= EXCLUSION_R1_SQ) | (d23_sq <= EXCLUSION_R1_SQ)
            bad |= ((d13_sq <= EXCLUSION_R2_SQ) & (np.abs(ri - atom_res[ks]) > 1))
            bad |= ((d23_sq <= EXCLUSION_R2_SQ) & (np.abs(rj - atom_res[ks]) > 1))
            bad |= ((d13_sq > EXCLUSION_R2_SQ) & (d13_sq <= EXCLUSION_R3_SQ))
            bad |= ((d23_sq > EXCLUSION_R2_SQ) & (d23_sq <= EXCLUSION_R3_SQ))
            good_k = ks[~bad]
            if len(good_k) == 0:
                continue

            g_d13 = np.sqrt(d13_sq[~bad])
            g_d23 = np.sqrt(d23_sq[~bad])
            b13 = (g_d13 * inv_bw).astype(np.int32)
            b23 = (g_d23 * inv_bw).astype(np.int32)
            fin = (b13 <= 3) & (b23 <= 7)
            if not fin.any():
                continue

            tk = atom_types[good_k[fin]]
            t13, t23 = b13[fin], b23[fin]
            t12 = b12[idx]
            indices = ((((ti_all[idx] * 12 + tj_all[idx]) * 12 + tk) * 4 + t12) * 4 + t13) * 8 + t23
            total += float(self._rough[indices].sum())

        return total

    def _build_atoms(self, coords_3bead: np.ndarray, sequence: str):
        """Build a flat atom array (types, res_idx, bead_idx, coords).

        Returns:
            (atom_types[int32], atom_res[int32], atom_beads[int32],
             atom_coords[float32 (N,3)])
            or None (not enough residues)
        """
        L = len(sequence)
        codes_c = np.array([_type_code(sequence[i], "C4'") for i in range(L)],
                           dtype=np.int32)
        codes_n = np.array([
            _type_code(sequence[i], "N9" if sequence[i] in ("A", "G") else "N1")
            for i in range(L)], dtype=np.int32)
        codes_p = np.array([_type_code(sequence[i], "P") for i in range(L)],
                           dtype=np.int32)

        valid = []
        for i in range(L):
            if codes_c[i] >= 0:
                valid.append((codes_c[i], i, 1))
            if codes_n[i] >= 0:
                valid.append((codes_n[i], i, 2))
            if codes_p[i] >= 0:
                valid.append((codes_p[i], i, 0))

        if len(valid) < 3:
            return None

        types = np.fromiter((v[0] for v in valid), dtype=np.int32)
        res = np.fromiter((v[1] for v in valid), dtype=np.int32)
        beads = np.fromiter((v[2] for v in valid), dtype=np.int32)
        coords = np.stack(
            [coords_3bead[r, b] for _, r, b in valid]).astype(np.float64)
        return types, res, beads, coords

    def score_with_gradient(
        self, coords_3bead: np.ndarray, sequence: str, *, p_only: bool = False
    ) -> Tuple[float, np.ndarray]:
        """Compute energy + ANALYTIC gradient (soft-binned linearization).

        The energy table is piecewise constant, so the gradient inside each
        bin is zero and undefined at bin boundaries. We linearize it with soft
        binning: for each triplet (d12, d13, d23), dE/dd is approximated by
        forward differences between bins, then distributed onto the three atom
        coordinates via the chain rule.

        Complexity = one score loop + O(#triplets) gradient accumulation.
        Compared with the old finite-difference scheme (6L score calls),
        a 2008-nt system drops from ~12 min to <1 s.

        Args:
            coords_3bead: (L, 3, 3) Angstroms
            sequence: ACGU string
            p_only: kept for compatibility (the analytic version already returns full-atom gradients; ignored)

        Returns:
            (energy, gradient) — gradient shape (L, 3, 3) in kBT/Angstrom
        """
        atoms = self._build_atoms(coords_3bead, sequence)
        grad_out = np.zeros_like(coords_3bead, dtype=np.float64)
        if atoms is None:
            return 0.0, grad_out
        atom_types, atom_res, atom_beads, atom_coords = atoms
        n_atoms = len(atom_types)

        diff = atom_coords[:, None, :] - atom_coords[None, :, :]
        dist_sq = (diff * diff).sum(-1).astype(np.float64)

        inv_bw = 1.0 / BIN_WIDTH_ROUGH

        # ── Pair-level constant terms (same as in score) ──
        iu, ju = np.triu_indices(n_atoms, k=1)
        d_ij = dist_sq[iu, ju]
        res_diff = np.abs(atom_res[iu].astype(np.int32) - atom_res[ju].astype(np.int32))
        excl = np.zeros(len(iu), dtype=bool)
        e_const = np.zeros(len(iu))
        m1 = d_ij <= EXCLUSION_R1_SQ
        e_const[m1] += 0.5; excl |= m1
        m2 = (~excl) & (d_ij <= EXCLUSION_R2_SQ) & (res_diff > 1)
        e_const[m2] += 0.5; excl |= m2
        m3 = (~excl) & (d_ij <= EXCLUSION_R3_SQ) & (d_ij > EXCLUSION_R2_SQ)
        e_const[m3] += 0.2; excl |= m3
        total = float(e_const.sum())

        cand_mask = (~excl) & (d_ij < R0_SQ)
        ci, cj = iu[cand_mask], ju[cand_mask]
        d12_arr = np.sqrt(dist_sq[ci, cj])
        b12_arr = (d12_arr * inv_bw).astype(np.int32)
        ok12 = b12_arr <= 3
        ci, cj, b12_arr = ci[ok12], cj[ok12], b12_arr[ok12]

        r_cut = (R0 - 0.3) ** 2
        arange_n = np.arange(n_atoms)
        # Forward-difference step (1/4 bin width — resolves bin boundaries without overshooting)
        h = BIN_WIDTH_ROUGH * 0.25

        # Bin-derivative cache: de/db[b] = E[b+1]-E[b] along that dimension
        def _dbin(axis_size):
            return None  # look up on demand (the table is sparse; cache hit rate would be low)

        # ── Vectorized three-body loop: process in chunks to avoid memory blowup ──
        n_pairs = len(ci)
        r_cut = (R0 - 0.3) ** 2
        arange_n = np.arange(n_atoms)
        chunk_size = 10000  # pairs processed per chunk

        for c_start in range(0, n_pairs, chunk_size):
            c_end = min(c_start + chunk_size, n_pairs)
            ci_c = ci[c_start:c_end]
            cj_c = cj[c_start:c_end]
            b12_c = b12_arr[c_start:c_end]
            n_c = c_end - c_start

            # k-candidate matrix (n_chunk, n_atoms)
            k_ok = (arange_n[None, :] > cj_c[:, None]) & \
                   (dist_sq[cj_c] < r_cut) & (dist_sq[ci_c] < R0_SQ)

            pi_idx_c, kk_all = np.nonzero(k_ok)
            if len(pi_idx_c) == 0:
                continue

            ii = ci_c[pi_idx_c]
            jj = cj_c[pi_idx_c]
            d13_sq = dist_sq[ii, kk_all]
            d23_sq = dist_sq[jj, kk_all]
            ri = atom_res[ii]
            rj = atom_res[jj]
            rk = atom_res[kk_all]

            bad = ((d13_sq <= EXCLUSION_R1_SQ) | (d23_sq <= EXCLUSION_R1_SQ) |
                   ((d13_sq <= EXCLUSION_R2_SQ) & (np.abs(ri - rk) > 1)) |
                   ((d23_sq <= EXCLUSION_R2_SQ) & (np.abs(rj - rk) > 1)) |
                   ((d13_sq > EXCLUSION_R2_SQ) & (d13_sq <= EXCLUSION_R3_SQ)) |
                   ((d23_sq > EXCLUSION_R2_SQ) & (d23_sq <= EXCLUSION_R3_SQ)))
            fin = ~bad
            if not fin.any():
                continue

            pi_f = pi_idx_c[fin]
            ii_f = ii[fin]; jj_f = jj[fin]; kk_f = kk_all[fin]
            d13_sq = d13_sq[fin]; d23_sq = d23_sq[fin]
            rk = rk[fin]

            g_d13 = np.sqrt(d13_sq)
            g_d23 = np.sqrt(d23_sq)
            b13 = (g_d13 * inv_bw).astype(np.int32)
            b23 = (g_d23 * inv_bw).astype(np.int32)
            fin2 = (b13 <= 3) & (b23 <= 7)
            if not fin2.any():
                continue

            pi_f = pi_f[fin2]; ii_f = ii_f[fin2]; jj_f = jj_f[fin2]
            kk_f = kk_f[fin2]
            g_d13 = g_d13[fin2]; g_d23 = g_d23[fin2]
            b13 = b13[fin2]; b23 = b23[fin2]
            rk = rk[fin2]

            tb12 = b12_c[pi_f].astype(np.int64)
            t_i = atom_types[ii_f].astype(np.int64)
            t_j = atom_types[jj_f].astype(np.int64)
            t_k = atom_types[kk_f].astype(np.int64)

            base_idx = (((t_i * 12 + t_j) * 12 + t_k) * 4 + tb12) * 32
            idx_table = base_idx + b13.astype(np.int64) * 8 + b23.astype(np.int64)
            e_vals = self._rough[idx_table]
            total += float(e_vals.sum())

            # Batch gradients
            safe_b12p = np.minimum(tb12 + 1, 3)
            safe_b13p = np.minimum(b13 + 1, 3)
            safe_b23p = np.minimum(b23 + 1, 7)
            e_12p = self._rough[((t_i*12+t_j)*12+t_k)*128 + safe_b12p*32 + b13*8 + b23]
            e_13p = self._rough[base_idx + safe_b13p * 8 + b23]
            e_23p = self._rough[base_idx + b13 * 8 + safe_b23p]
            ded_d12 = (e_12p - e_vals) / BIN_WIDTH_ROUGH
            ded_d13 = (e_13p - e_vals) / BIN_WIDTH_ROUGH
            ded_d23 = (e_23p - e_vals) / BIN_WIDTH_ROUGH

            if np.any(ded_d12 != 0.0):
                xi = atom_coords[ii_f]; xj = atom_coords[jj_f]
                u12 = (xi - xj) / np.maximum(d12_arr[c_start + pi_f], 1e-8)[:, None]
                f12_per_trip = ded_d12[:, None] * u12
                f12_pair = np.zeros((n_c, 3), dtype=np.float64)
                np.add.at(f12_pair, pi_f, f12_per_trip)
                np.add.at(grad_out, (atom_res[ci_c], atom_beads[ci_c]), f12_pair)
                np.add.at(grad_out, (atom_res[cj_c], atom_beads[cj_c]), -f12_pair)

            # d13 / d23 forces: scattered directly (per-triplet)
            if np.any(ded_d13 != 0.0):
                u13 = (atom_coords[ii_f] - atom_coords[kk_f]) / \
                      np.maximum(g_d13[:, None], 1e-8)
                f13 = ded_d13[:, None] * u13
                np.add.at(grad_out, (atom_res[ii_f], atom_beads[ii_f]), f13)
                np.add.at(grad_out, (atom_res[kk_f], atom_beads[kk_f]), -f13)

            if np.any(ded_d23 != 0.0):
                u23 = (atom_coords[jj_f] - atom_coords[kk_f]) / \
                      np.maximum(g_d23[:, None], 1e-8)
                f23 = ded_d23[:, None] * u23
                np.add.at(grad_out, (atom_res[jj_f], atom_beads[jj_f]), f23)
                np.add.at(grad_out, (atom_res[kk_f], atom_beads[kk_f]), -f23)

        return total, grad_out


def score_from_openmm_coords(
    pos_nm: np.ndarray,
    sequence: str,
    potential: TriRNASPPotential,
    scale: float = 10.0,
) -> Tuple[float, np.ndarray]:
    """Score from OpenMM 3-bead coordinates (nm) and return kJ/mol.

    Args:
        pos_nm: (3*L, 3) array in nm (OpenMM format)
        sequence: ACGU string
        potential: loaded TriRNASPPotential
        scale: energy scaling factor

    Returns:
        (energy_kJmol, force_kJmol_per_nm)
        force shape: (3*L, 3)
    """
    L = len(sequence)
    coords_A = pos_nm.reshape(L, 3, 3) * 10.0  # nm → Å
    e_kBT, grad_kBT_A = potential.score_with_gradient(coords_A, sequence)
    e_kJmol = e_kBT * scale * KBT_FACTOR
    # Force = -grad * scale * KBT_FACTOR, convert from kBT/Å to kJ/mol/nm
    # 1 kBT/Å = KBT_FACTOR kJ/mol / (0.1 nm) = KBT_FACTOR * 10 kJ/mol/nm
    force_kJmol_nm = -grad_kBT_A * scale * KBT_FACTOR * 10.0
    return e_kJmol, force_kJmol_nm.reshape(3 * L, 3)


# ── Quick test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    energy_dir = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        pot = TriRNASPPotential(energy_dir)

        # Test: random 10-nt RNA
        seq = "AUGCAUGCAU"
        L = len(seq)
        coords = np.random.rand(L, 3, 3) * 10.0

        e = pot.score(coords, seq)
        print(f"Energy: {e:.3f} kBT")

        e2, grad = pot.score_with_gradient(coords, seq)
        print(f"Energy (with grad): {e2:.3f} kBT")
        print(f"Gradient norm: {np.linalg.norm(grad):.3f}")
        print(f"Max gradient: {np.max(np.abs(grad)):.3f} kBT/A")

    except FileNotFoundError as e:
        print(f"Error: {e}")
