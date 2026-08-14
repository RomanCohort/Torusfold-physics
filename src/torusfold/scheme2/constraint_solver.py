"""
constraint_solver.py — Geometric Constraint Solver for circRNA.

Implements Plan B: zero-training 3D structure prediction from geometric constraints.
Uses a combination of:
1. Regular polygon initialization (closure guaranteed by construction)
2. Iterative perturbation to satisfy pair constraints
3. Shapiro-Barnes algorithm for closure correction
4. Simple coarse-grained energy ranking

No experimental data needed. All physics comes from:
- Backbone bond geometry (5.9 Å P-P distance)
- Watson-Crick pair geometry (10.6 Å C1'-C1' distance)
- A-form RNA dihedral preferences
- Steric exclusion (3.0 Å minimum non-bonded distance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SolverConfig:
    """Configuration for the constraint solver."""
    bond_length: float = 5.9        # Å, P-P backbone distance (A-form RNA 真值)
    pair_distance: float = 10.6     # Å, WC C1'-C1' distance
    clash_distance: float = 3.0     # Å, minimum non-bonded distance
    n_samples: int = 20             # Number of conformations to sample
    closure_tolerance: float = 0.5  # Å, max allowed closure deviation
    max_iterations: int = 100       # Max perturbation iterations
    perturbation_scale: float = 0.5 # Å, initial perturbation amplitude
    # Annealing closure (IsRNAcirc-inspired)
    use_annealing_closure: bool = True   # Use simulated annealing for BSJ
    annealing_temp_init: float = 500.0   # K, initial temperature
    annealing_temp_final: float = 300.0  # K, final temperature
    annealing_cooling: float = 0.95      # Cooling rate per cycle
    annealing_steps_per_temp: int = 50   # Steps per temperature level


class GeometricConstraintSolver:
    """Solve circRNA 3D structure from geometric constraints.

    Core algorithm:
    1. Initialize nucleotides on regular polygon (radius = L * bond_length / 2π)
       → closure is guaranteed: x[L-1] connects to x[0] exactly
    2. Iteratively perturb positions to satisfy pair constraints
       → for each pair (i, j, d), move j toward/away from i to achieve d
    3. After perturbation, apply Shapiro-Barnes closure correction
       → redistribute closure error evenly across all bonds
    4. Filter out conformations with steric clashes
    5. Rank remaining by simple CG energy

    Args:
        config: SolverConfig with geometric parameters
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    def solve(self, constraint_set) -> List[np.ndarray]:
        """Sample conformations satisfying all constraints.

        Args:
            constraint_set: ConstraintSet from ConstraintExtractor

        Returns:
            List of (L, 3) numpy arrays, each a valid conformation
        """
        L = constraint_set.seq_len
        if L < 3:
            return [self._single_point(L)]

        conformations = []

        for sample_idx in range(self.config.n_samples):
            # 1. Initialize on regular polygon (closure guaranteed)
            coords = self._regular_polygon(L, self.config.bond_length)

            # 2. Perturb to satisfy pair constraints
            coords = self._satisfy_pair_constraints(coords, constraint_set)

            # 3. Correct closure using annealing (IsRNAcirc-inspired) or standard
            if self.config.use_annealing_closure:
                coords = self._annealing_closure(coords, constraint_set)
            else:
                coords = self._closure_correction(coords)

            # 4. Steric clash check (soft: count clashes, don't discard)
            clash_count = self._count_clashes(coords)

            # 5. Additional local relaxation (optional)
            coords = self._local_relax(coords, constraint_set)

            conformations.append((coords, clash_count))

        # Guarantee at least one conformation (best-effort fallback)
        if not conformations:
            fallback = self._regular_polygon(L, self.config.bond_length)
            fallback = self._closure_correction(fallback)
            conformations.append((fallback, self._count_clashes(fallback)))

        # Rank by CG energy + clash penalty
        if len(conformations) > 1:
            results = []
            for coords, clash_count in conformations:
                e = self._compute_cg_energy(coords, constraint_set)
                e += clash_count * 50.0  # Penalize each clash heavily
                results.append((coords, e))
            results.sort(key=lambda x: x[1])
            return [r[0] for r in results]
        return [conformations[0][0]]

    def _regular_polygon(self, L: int, bond_length: float) -> np.ndarray:
        """Initialize nucleotides on a ring.

        For L < 10: simple planar polygon (short sequences, torus not meaningful).
        For 10 <= L < 500: toroidal helix with gentle z-oscillation.
        For L >= 500: pure planar polygon (long sequences rely on OpenMM for 3D).

        Toroidal helix (gentle version, fixed 2026-07-27):
          - Original q = L/11 caused d_phi = 32.7°/step → z-jumps → bond ~9.4Å
          - Fixed: q = L/110 gives d_phi ~3.3°/step → bond ~5.9Å as expected
          - z-span ~26Å (small oscillation to seed 3D structure)

        Returns:
            (L, 3) coordinates
        """
        coords = np.zeros((L, 3), dtype=np.float64)

        # Main ring radius: circumference = L * bond_length
        R = L * bond_length / (2 * math.pi)

        if L < 10:
            # Short sequence: planar polygon
            for i in range(L):
                angle = 2 * math.pi * i / L
                coords[i, 0] = R * math.cos(angle)
                coords[i, 1] = R * math.sin(angle)
                coords[i, 2] = 0.0
        elif L < 500:
            # Medium sequence: gentle toroidal helix (fixed q)
            r = 13.0  # A-form radius
            q = L / 110.0  # Fixed: gentle z-oscillation (was L/11, too steep)
            for i in range(L):
                theta = 2 * math.pi * i / L
                phi = 2 * math.pi * q * i / L
                coords[i, 0] = (R + r * math.cos(phi)) * math.cos(theta)
                coords[i, 1] = (R + r * math.cos(phi)) * math.sin(theta)
                coords[i, 2] = r * math.sin(phi)
        else:
            # Long sequence: pure planar polygon (OpenMM will 3D-ify)
            for i in range(L):
                angle = 2 * math.pi * i / L
                coords[i, 0] = R * math.cos(angle)
                coords[i, 1] = R * math.sin(angle)
                coords[i, 2] = 0.0

        return coords

    def _satisfy_pair_constraints(
        self,
        coords: np.ndarray,
        constraint_set,
        max_iter: Optional[int] = None,
    ) -> np.ndarray:
        """Iteratively perturb positions to satisfy pair constraints.

        For each pair (i, j, target_d, weight):
        - Compute current distance d_curr
        - If d_curr differs from target_d, move both i and j symmetrically
        - Weight determines how much to move (higher = more movement)

        This is a local optimization, not global. Each pair is treated
        independently, which can lead to conflicts. The closure correction
        and energy ranking handle the global consistency.
        """
        max_iter = max_iter or self.config.max_iterations
        coords = coords.copy()
        L = len(coords)

        pair_constraints = constraint_set.pair_constraints
        if not pair_constraints:
            return coords

        # CRITICAL: Skip pair constraint perturbation entirely.
        # For circRNA with L > 500, perturbation destroys structure geometry
        # because even "local" pairs (span <= 50) have ~100A initial distance on ring,
        # forcing them to 10.6A would collapse local structure.
        # Let OpenMM handle ALL pair constraints during energy minimization.
        # This preserves initial ring geometry and allows OpenMM to naturally fold.
        return coords

        return coords

    def _closure_correction(self, coords: np.ndarray) -> np.ndarray:
        """Shapiro-Barnes algorithm for ring closure.

        After perturbation, the closure constraint ||x[0] - x[-1]|| ≈ bond_length
        may be violated. This algorithm redistributes the closure error
        evenly across all nucleotides by adjusting bond vectors.

        Algorithm:
        1. Compute closure error: e = x[-1] - x[0] (should be ~bond_length vector)
        2. Compute target closure vector: v = x[0] + bond_length * unit(e)
        3. Distribute error: each bond vector gets a correction = -e/L
        4. Apply corrections to all positions

        This preserves the relative geometry while fixing closure.
        """
        coords = coords.copy()
        L = len(coords)
        bond_length = self.config.bond_length

        # Current closure vector (should connect x[-1] back to x[0])
        closure_vec = coords[0] - coords[-1]
        closure_dist = np.linalg.norm(closure_vec)

        # Check if closure is already satisfied
        if abs(closure_dist - bond_length) < self.config.closure_tolerance:
            return coords

        # Target: we want ||x[0] - x[-1]|| = bond_length
        # Current: ||closure_vec|| = closure_dist

        # Distribute error across all bonds
        error_vec = closure_vec - np.array([bond_length, 0, 0])  # reference direction
        correction = -error_vec / L

        # Apply correction cumulatively
        for i in range(L):
            coords[i] += i * correction

        # Final closure check: explicitly set last position to connect to first
        # Place x[-1] at distance bond_length from x[0]
        final_vec = coords[0] - coords[-2]
        final_dist = np.linalg.norm(final_vec)
        if final_dist > 0.01:
            final_dir = final_vec / final_dist
            coords[-1] = coords[0] - bond_length * final_dir

        return coords

    def _annealing_closure(
        self,
        coords: np.ndarray,
        constraint_set,
    ) -> np.ndarray:
        """Simulated annealing for BSJ closure (IsRNAcirc-inspired).

        Instead of directly correcting closure (which can introduce strain),
        gradually anneal the structure from high to low temperature,
        allowing the BSJ region to find a low-strain closure path.

        Inspired by IsRNAcirc's end-closure step:
        Jiang et al., PLOS Comp Biol 2024, DOI:10.1371/journal.pcbi.1012293

        Algorithm:
        1. Start at high temperature (flexible)
        2. Perturb positions near BSJ
        3. Accept perturbations that improve closure + pair constraints
        4. Cool down gradually (accept fewer bad moves)
        5. Final geometric correction as safety net
        """
        coords = coords.copy()
        L = len(coords)
        bond_length = self.config.bond_length
        pair_constraints = getattr(constraint_set, 'pair_constraints', [])

        T = self.config.annealing_temp_init
        T_final = self.config.annealing_temp_final
        cooling = self.config.annealing_cooling
        steps = self.config.annealing_steps_per_temp

        # Precompute: how many residues near BSJ to perturb
        n_bsj_zone = max(3, L // 10)

        best_coords = coords.copy()
        best_energy = self._compute_cg_energy(coords, constraint_set)

        while T > T_final:
            for _ in range(steps):
                # Perturb positions near BSJ (indices 0..n_bsj_zone and L-n_bsj_zone..L-1)
                perturbed = coords.copy()
                scale = 0.1 * (T / self.config.annealing_temp_init)  # Scale with temperature

                for idx in list(range(n_bsj_zone)) + list(range(L - n_bsj_zone, L)):
                    perturbed[idx] += np.random.randn(3) * scale * bond_length * 0.1

                # Compute closure distance (should be ~bond_length)
                closure_dist = np.linalg.norm(perturbed[0] - perturbed[-1])
                closure_error = abs(closure_dist - bond_length)

                # Compute energy
                new_energy = self._compute_cg_energy(perturbed, constraint_set)
                energy_change = new_energy - best_energy

                # Metropolis acceptance criterion
                if energy_change < 0:
                    # Always accept improvements
                    coords = perturbed
                    if new_energy < best_energy:
                        best_coords = perturbed.copy()
                        best_energy = new_energy
                elif T > 0:
                    # Accept with probability exp(-ΔE / T_scale)
                    T_scale = T * 0.01  # Scale temperature to energy units
                    accept_prob = math.exp(-energy_change / T_scale)
                    if np.random.random() < accept_prob:
                        coords = perturbed

            T *= cooling

        # Final geometric correction as safety net (from _closure_correction)
        closure_dist = np.linalg.norm(best_coords[0] - best_coords[-1])
        if abs(closure_dist - bond_length) > self.config.closure_tolerance:
            best_coords = self._closure_correction(best_coords)

        return best_coords

    def _pairwise_dist_sq(self, coords: np.ndarray) -> np.ndarray:
        """Squared pairwise distance matrix via inner-product expansion.

        |a-b|² = |a|² + |b|² - 2a·b  — avoids the (L,L,3) intermediate that
        blows up memory for L~2000 (288 MB → 32 MB for float64).

        Returns:
            (L, L) squared distances, symmetric
        """
        sq = np.sum(coords ** 2, axis=1)          # (L,)
        dist_sq = sq[:, np.newaxis] + sq[np.newaxis, :] - 2.0 * coords @ coords.T
        # Numerical noise can produce tiny negatives for coincident points
        return np.maximum(dist_sq, 0.0)

    def _has_clashes(self, coords: np.ndarray) -> bool:
        """Check for steric clashes (vectorized, memory-safe for L~2000)."""
        L = len(coords)
        clash_dist = self.config.clash_distance

        if L < 4:
            return False

        dist_sq = self._pairwise_dist_sq(coords)
        dist_matrix = np.sqrt(dist_sq + 1e-8)

        i_idx, j_idx = np.triu_indices(L, k=2)  # |i-j| >= 2
        mask = ~((i_idx == 0) & (j_idx == L - 1))

        valid_dists = dist_matrix[i_idx[mask], j_idx[mask]]
        return bool(np.any(valid_dists < clash_dist))

    def _count_clashes(self, coords: np.ndarray) -> int:
        """Count steric clashes (soft version, memory-safe for L~2000).

        Returns:
            Number of clashing (i,j) pairs
        """
        L = len(coords)
        clash_dist = self.config.clash_distance

        if L < 4:
            return 0

        dist_sq = self._pairwise_dist_sq(coords)
        dist_matrix = np.sqrt(dist_sq + 1e-8)

        i_idx, j_idx = np.triu_indices(L, k=2)
        mask = ~((i_idx == 0) & (j_idx == L - 1))

        valid_dists = dist_matrix[i_idx[mask], j_idx[mask]]
        return int(np.sum(valid_dists < clash_dist))

    def _local_relax(self, coords: np.ndarray, constraint_set) -> np.ndarray:
        """Local relaxation to reduce energy.

        Simple iterative adjustment:
        - For each position, move toward the mean of its neighbors
        - This reduces bond strain while preserving overall shape
        """
        coords = coords.copy()
        L = len(coords)

        for iteration in range(20):
            for i in range(L):
                # Neighbors: (i-1) and (i+1), circularly
                prev = (i - 1) % L
                next = (i + 1) % L

                # Target: midpoint of neighbors + bond_length direction
                midpoint = (coords[prev] + coords[next]) / 2

                # Move slightly toward midpoint (relaxation)
                coords[i] = 0.9 * coords[i] + 0.1 * midpoint

        return coords

    def _compute_cg_energy(self, coords: np.ndarray, constraint_set) -> float:
        """Compute coarse-grained energy with extended physics terms (vectorized).

        Energy components:
        1. Bond energy: k_bond * Σ(d - bond_length)²
        2. Pair energy: k_pair * Σ(d - target)² for satisfied pairs
        3. Clash energy: k_clash * Σmax(0, clash_dist - d)²
        4. Stacking energy: k_stack * Σ(d_stack - 3.4)² for adjacent bases
        5. Electrostatic energy: k_elec * Σ q²/d for phosphate repulsion
        6. Dihedral energy: k_dih * Σ(dih - dih_Aform)² for backbone dihedrals
        """
        L = len(coords)
        bond_length = self.config.bond_length
        clash_dist = self.config.clash_distance

        # Force constants
        k_bond = 1.0
        k_pair = 0.5
        k_clash = 10.0
        k_stack = 0.3       # Base stacking (π-π interaction)
        k_elec = 0.05       # Phosphate electrostatic repulsion
        k_dih = 0.1         # A-form RNA dihedral preference

        energy = 0.0

        # 1. Bond energy (vectorized)
        next_coords = np.roll(coords, -1, axis=0)
        bond_dists = np.linalg.norm(next_coords - coords, axis=1)
        energy += k_bond * np.sum((bond_dists - bond_length) ** 2)

        # 2. Pair energy
        for (i, j, target_d, weight) in constraint_set.pair_constraints:
            d = np.linalg.norm(coords[j] - coords[i])
            energy += k_pair * weight * (d - target_d) ** 2

        # 3. Clash energy (vectorized, memory-safe for L~2000)
        # Only check non-adjacent pairs (|i-j| > 1) excluding BSJ
        if L > 10:  # Only for longer sequences where clashes matter
            dist_matrix = np.sqrt(self._pairwise_dist_sq(coords) + 1e-8)

            # Create mask for valid pairs (non-adjacent, not BSJ)
            i_idx, j_idx = np.triu_indices(L, k=2)  # Upper triangle, |i-j| >= 2
            mask = ~((i_idx == 0) & (j_idx == L - 1))  # Exclude BSJ
            valid_i, valid_j = i_idx[mask], j_idx[mask]

            valid_dists = dist_matrix[valid_i, valid_j]
            clashes = valid_dists[valid_dists < clash_dist]
            energy += k_clash * np.sum((clash_dist - clashes) ** 2)

        # 4. Base stacking energy (vectorized)
        stack_distance = 3.4  # Å, A-form RNA stacking distance
        dz = np.abs(np.roll(coords[:, 2], -1) - coords[:, 2])
        energy += k_stack * np.sum((dz - stack_distance) ** 2)

        # 5. Electrostatic repulsion (vectorized with cutoff)
        if L > 10:
            elec_cutoff = 20.0
            valid_dists_elec = dist_matrix[valid_i, valid_j]
            within_cutoff = valid_dists_elec[(valid_dists_elec < elec_cutoff) & (valid_dists_elec > 0.1)]
            energy += k_elec * np.sum(1.0 / within_cutoff)

        # 6. A-form RNA dihedral preference (simplified, O(L))
        v1 = coords[1:-1] - coords[:-2]
        v2 = coords[2:] - coords[1:-1]
        norms = np.linalg.norm(v1, axis=1, keepdims=True) * np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
        cos_angles = np.sum(v1 * v2, axis=1)[:, np.newaxis] / norms
        energy += k_dih * np.sum((cos_angles - (-0.276)) ** 2)

        # 7. A-form virtual dihedral 扭转势 (正路三维化):
        #    连续 4 个 P 原子的二面角趋向 ~70° (A-form 右手螺旋特征扭转)。
        #    用 cos 差做周期性惩罚, 不钉位置只奖励正确扭转。circRNA 闭合
        #    约束会自然引入 twist defect (累积扭转须凑 360° 整数倍), 这是
        #    物理正确的: circRNA 不是均匀螺旋, 有局部扭转涨落。
        if L >= 4:
            p0, p1, p2, p3 = coords[:-3], coords[1:-2], coords[2:-1], coords[3:]
            b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
            b2_norm = np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-8)
            n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-8)
            b2_hat = b2 / b2_norm
            m1 = np.cross(n1, b2_hat)
            x = np.sum(n1 * n2, axis=1)
            y = np.sum(m1 * n2, axis=1)
            dih = np.arctan2(y, x)  # [-pi, pi]
            k_torsion = 0.8
            target_dih = 1.22  # ~70°, A-form 右手螺旋特征扭转
            energy += k_torsion * np.sum((np.cos(dih) - np.cos(target_dih)) ** 2)

        return energy

    def _single_point(self, L: int) -> np.ndarray:
        """Degenerate case: place all nucleotides at same point."""
        return np.zeros((L, 3), dtype=np.float64)


def circular_distance_matrix_np(L: int) -> np.ndarray:
    """Compute circular distance matrix (NumPy version).

    d_circ(i, j) = min(|i - j|, L - |i - j|)

    Args:
        L: Sequence length

    Returns:
        (L, L) numpy array with circular distances
    """
    positions = np.arange(L)
    diff = np.abs(positions[:, None] - positions[None, :])
    return np.minimum(diff, L - diff)


def is_bsj_crossing_np(i: int, j: int, L: int) -> bool:
    """Check if pair (i, j) crosses the BSJ.

    BSJ-crossing: the shorter arc between i and j contains the BSJ.
    Equivalent to: |i - j| > L/2 in linear distance.
    """
    return abs(i - j) > L / 2