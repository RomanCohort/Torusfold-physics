# -*- coding: utf-8 -*-
"""trrna2_calibrator.py - calibrate RhoFold coordinates with the trRNA2 distance matrix.

Core idea:
  trRNA2 outputs a distance matrix (L x L); RhoFold outputs 3D coordinates.
  Use the trRNA2 distance matrix as a restraint and calibrate the RhoFold
  coordinates by minimizing the distance discrepancy.

Public API:
  calibrate_with_distance_matrix() - coordinate-optimization calibration
  trrna2_rhofold_ensemble()       - dual-engine ensemble prediction
  integrated_predict_chunk()      - unified entry point
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ── Constants ──
P_P_BOND = 5.9          # P-P bond length (Angstrom)
WC_TARGET_DIST = 20.0   # Watson-Crick C1'-C1' target distance
WC_TOLERANCE = 3.0      # WC distance tolerance
MAX_BOND_DEVIATION = 1.5  # max bond-length deviation (Angstrom)


# ── Data classes ──
@dataclass
class CalibrationResult:
    """Calibration result."""
    coords: np.ndarray          # (L, 3) calibrated coordinates
    initial_coords: np.ndarray  # (L, 3) original RhoFold coordinates
    dist_matrix: np.ndarray     # (L, L) trRNA2 distance matrix
    rmse_before: float          # distance RMSE before calibration
    rmse_after: float           # distance RMSE after calibration
    n_iterations: int           # actual number of iterations
    success: bool               # whether the optimization succeeded
    confidence: float           # calibration confidence


# ── Utility functions ──

def _pairwise_distance(coords: np.ndarray) -> np.ndarray:
    """Compute the pairwise distance matrix (L, L)."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _find_wc_pairs_from_ss(ss: str) -> List[Tuple[int, int]]:
    """Find WC-paired positions (0-based) from a secondary-structure string.

    Standard bracket matching: '(' pairs with ')', nesting supported.
    """
    pairs = []
    stack = []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def _find_wc_pairs(sequence: str) -> List[Tuple[int, int]]:
    """Infer WC pairs from the sequence (only when it is dot-bracket format).

    Returns an empty list if the sequence is made of bases such as ATGCNU.
    """
    if any(ch in '().' for ch in sequence):
        return _find_wc_pairs_from_ss(sequence)
    return []


def _rmse_distance(coords: np.ndarray, target_dist: np.ndarray) -> float:
    """Compute the RMSE between the coordinate distance matrix and the target distance matrix."""
    pred_dist = _pairwise_distance(coords)
    return float(np.sqrt(np.mean((pred_dist - target_dist) ** 2)))


# ── Core calibration ──

def calibrate_with_distance_matrix(
    coords_rhofold: np.ndarray,
    dist_trrna2: np.ndarray,
    sequence: str,
    secondary_structure: Optional[str] = None,
    n_iterations: int = 200,
    bond_weight: float = 200.0,
    wc_weight: float = 50.0,
    dist_weight: float = 1.0,
    verbose: bool = False,
) -> CalibrationResult:
    """Calibrate RhoFold coordinates with the trRNA2 distance matrix.

    Optimizes the coordinates with scipy.optimize.minimize to minimize
    |d_ij(coords) - dist_trrna2[i,j]| while preserving backbone bond lengths
    and secondary-structure restraints.

    Args:
        coords_rhofold: (L, 3) RhoFold-predicted coordinates (Angstrom)
        dist_trrna2: (L, L) trRNA2-predicted distance matrix (Angstrom)
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket). If None, brackets are inferred from the sequence.
        n_iterations: maximum number of optimization iterations
        bond_weight: bond-length restraint weight
        wc_weight: WC-pairing restraint weight
        dist_weight: distance-matrix fitting weight
        verbose: print the optimization progress

    Returns:
        A CalibrationResult with calibrated coordinates and quality metrics
    """
    L = len(sequence)
    if coords_rhofold.shape != (L, 3):
        raise ValueError(f"coords shape {coords_rhofold.shape} != ({L}, 3)")
    if dist_trrna2.shape != (L, L):
        raise ValueError(f"dist shape {dist_trrna2.shape} != ({L}, {L})")

    # Find WC pairs
    if secondary_structure:
        wc_pairs = _find_wc_pairs_from_ss(secondary_structure)
    else:
        wc_pairs = _find_wc_pairs(sequence)

    # Compute the initial RMSE
    rmse_before = _rmse_distance(coords_rhofold, dist_trrna2)

    # If already accurate enough, return early to avoid degradation
    if rmse_before < 0.5:
        return CalibrationResult(
            coords=coords_rhofold.copy(),
            initial_coords=coords_rhofold.copy(),
            dist_matrix=dist_trrna2.copy(),
            rmse_before=rmse_before,
            rmse_after=rmse_before,
            n_iterations=0,
            success=True,
            confidence=0.95,
        )

    # Target bond length (adjacent-residue P-P distance)
    bond_targets = np.full(L - 1, P_P_BOND)

    # WC target distance
    wc_targets = np.array([WC_TARGET_DIST] * len(wc_pairs))

    def objective_stage1(x):
        """Stage 1: distance-matrix fitting (unconstrained)."""
        coords = x.reshape(L, 3)
        pred_dist = _pairwise_distance(coords)
        return np.mean((pred_dist - dist_trrna2) ** 2)

    def objective_stage2(x):
        """Stage 2: restrained refinement (distance + bond length + WC)."""
        coords = x.reshape(L, 3)

        # 1. Distance-matrix fitting
        pred_dist = _pairwise_distance(coords)
        dist_loss = np.mean((pred_dist - dist_trrna2) ** 2)

        # 2. Bond-length restraint (adjacent residues) - quartic penalty
        bond_dists = np.sqrt(np.sum((coords[1:] - coords[:-1]) ** 2, axis=1))
        bond_dev = bond_dists - bond_targets
        bond_loss = np.mean(bond_dev ** 4)

        # 3. WC-pairing restraint
        wc_loss = 0.0
        if wc_pairs:
            wc_dists = np.array([
                np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                for i, j in wc_pairs
            ])
            wc_dev = wc_dists - wc_targets
            wc_loss = np.mean(wc_dev ** 4)

        return dist_weight * dist_loss + bond_weight * bond_loss + wc_weight * wc_loss

    # L-BFGS-B with box constraints
    # L-BFGS-B bounds
    bounds = []
    for i in range(L):
        for j in range(3):
            center = coords_rhofold[i, j]
            bounds.append((center - 30.0, center + 30.0))

    x0 = coords_rhofold.flatten().copy()

    # Stage 1: distance-matrix fitting (fast convergence to the rough layout)
    s1_iters = max(n_iterations // 2, 50)
    result1 = minimize(
        objective_stage1, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': s1_iters, 'ftol': 1e-6, 'gtol': 1e-4},
    )

    # Stage 2: restrained refinement (apply bond-length/WC restraints on the stage-1 result)
    s2_iters = max(n_iterations // 2, 50)
    result2 = minimize(
        objective_stage2, result1.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': s2_iters, 'ftol': 1e-6, 'gtol': 1e-4},
    )

    calibrated_coords = result2.x.reshape(L, 3)
    rmse_after = _rmse_distance(calibrated_coords, dist_trrna2)
    total_nit = result1.nit + result2.nit

    # Confidence: RMSE improvement ratio
    if rmse_before > 1e-6:
        improvement = 1.0 - (rmse_after / rmse_before)
        confidence = min(0.95, max(0.1, 0.5 + 0.5 * improvement))
    else:
        improvement = 0.0
        confidence = 0.9

    if verbose:
        print(f"  [Calibrate] RMSE: {rmse_before:.2f} -> {rmse_after:.2f} "
              f"({improvement*100:.1f}% improved), iter={total_nit}, "
              f"conf={confidence:.3f}")

    return CalibrationResult(
        coords=calibrated_coords,
        initial_coords=coords_rhofold.copy(),
        dist_matrix=dist_trrna2.copy(),
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        n_iterations=total_nit,
        success=result2.success,
        confidence=confidence,
    )


# ── Dual-engine ensemble ──

def trrna2_rhofold_ensemble(
    sequence: str,
    secondary_structure: Optional[str] = None,
    msa_path: Optional[str] = None,
    n_trrna2_samples: int = 1,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    verbose: bool = False,
    device: str = "auto",
) -> Tuple[np.ndarray, float, float]:
    """Run RhoFold once for coordinates, then trRNA2 for the distance matrix,
    then calibrate the output.

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket)
        msa_path: MSA file path (optional)
        n_trrna2_samples: number of trRNA2 samples (average the distance matrices)
        output_dir: output directory
        name: output-name prefix
        verbose: print detailed information
        device: device ("auto", "cuda", "cpu")

    Returns:
        (coords, confidence, dist_quality)
        coords: (L, 3) calibrated coordinates
        confidence: calibration confidence
        dist_quality: distance-matrix quality metric (0-1)
    """
    from .rhofold_wrapper import rhofold_predict_chunk
    from .trrna2_wrapper import trrna2_predict_chunk

    L = len(sequence)

    # Step 1: RhoFold prediction
    if verbose:
        print("  [Ensemble] Step 1: RhoFold+ prediction...")

    rf_coords, rf_conf = rhofold_predict_chunk(
        sequence, secondary_structure,
        output_dir=output_dir, name=f"{name}_rhofold" if name else None,
        msa_path=msa_path, verbose=verbose, device=device,
    )

    # Step 2: trRNA2 distance matrix (average over repeated samples)
    if verbose:
        print(f"  [Ensemble] Step 2: trRNA2 distance matrix ({n_trrna2_samples} samples)...")

    dist_accum = None
    tr_conf_accum = 0.0

    for s in range(n_trrna2_samples):
        tr_result = trrna2_predict_chunk(
            sequence,
            output_dir=output_dir,
            name=f"{name}_tr{s}" if name else f"tr{s}",
            num_recycles=3,
            verbose=verbose,
        )

        if tr_result.dist is not None:
            if dist_accum is None:
                dist_accum = tr_result.dist.copy()
            else:
                dist_accum += tr_result.dist
            tr_conf_accum += tr_result.confidence

    if dist_accum is None:
        # trRNA2 returned no distance matrix; fall back to a RhoFold-only prediction
        if verbose:
            print("  [Ensemble] trRNA2 no distance matrix, using RhoFold alone")
        return rf_coords, rf_conf, 0.0

    dist_avg = dist_accum / n_trrna2_samples
    tr_conf_avg = tr_conf_accum / n_trrna2_samples

    # Step 3: Calibration
    if verbose:
        print("  [Ensemble] Step 3: Distance matrix calibration...")

    cal_result = calibrate_with_distance_matrix(
        rf_coords, dist_avg, sequence, secondary_structure,
        verbose=verbose,
    )

    # Combined confidence: RhoFold + trRNA2 + calibration improvement
    combined_conf = 0.4 * rf_conf + 0.3 * tr_conf_avg + 0.3 * cal_result.confidence

    # Distance quality: RMSE improvement ratio
    if cal_result.rmse_before > 0:
        dist_quality = min(1.0, 1.0 - cal_result.rmse_after / cal_result.rmse_before)
    else:
        dist_quality = 0.5

    if verbose:
        print(f"  [Ensemble] Done: conf={combined_conf:.3f}, "
              f"dist_quality={dist_quality:.3f}")

    return cal_result.coords, combined_conf, dist_quality


# ── Unified entry point ──

def integrated_predict_chunk(
    sequence: str,
    secondary_structure: Optional[str] = None,
    msa_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    use_calibration: bool = True,
    n_trrna2_samples: int = 1,
    verbose: bool = False,
    device: str = "auto",
    boundary_pairs: Optional[List[Tuple[int, int, str]]] = None,
) -> Tuple[np.ndarray, float, dict]:
    """Unified prediction interface that calls both RhoFold and trRNA2 for calibration.

    Compatible with the calling convention of segmented_vfold3d.py.

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket)
        msa_path: MSA file path (optional)
        output_dir: output directory
        name: output-name prefix
        use_calibration: whether to enable trRNA2 calibration (False degrades to pure RhoFold)
        n_trrna2_samples: number of trRNA2 samples
        verbose: print detailed information
        device: device
        boundary_pairs: list of Level-1 boundary restraint pairs

    Returns:
        (coords, confidence, metadata)
        coords: (L, 3) C1' coordinates (Angstrom)
        confidence: combined confidence [0, 1]
        metadata: dict with detailed information
            - method: method used
            - rmse_before: pre-calibration RMSE (calibration mode only)
            - rmse_after: post-calibration RMSE (calibration mode only)
            - dist_quality: distance quality (calibration mode only)
    """
    metadata = {"method": "rhofold"}

    if not use_calibration:
        # Pure RhoFold mode
        from .rhofold_wrapper import rhofold_predict_chunk
        coords, conf = rhofold_predict_chunk(
            sequence, secondary_structure,
            output_dir=output_dir, name=name,
            msa_path=msa_path, verbose=verbose, device=device,
            boundary_pairs=boundary_pairs,
        )
        metadata["method"] = "rhofold_only"
        return coords, conf, metadata

    # Calibration mode
    try:
        coords, conf, dist_quality = trrna2_rhofold_ensemble(
            sequence, secondary_structure,
            msa_path=msa_path,
            n_trrna2_samples=n_trrna2_samples,
            output_dir=output_dir, name=name,
            verbose=verbose, device=device,
        )
        metadata["method"] = "rhofold+trrna2_calibrated"
        metadata["dist_quality"] = dist_quality
    except Exception as e:
        if verbose:
            print(f"  [Integrated] Calibration failed: {e}, falling back to RhoFold")
        from .rhofold_wrapper import rhofold_predict_chunk
        coords, conf = rhofold_predict_chunk(
            sequence, secondary_structure,
            output_dir=output_dir, name=name,
            msa_path=msa_path, verbose=verbose, device=device,
            boundary_pairs=boundary_pairs,
        )
        metadata["method"] = "rhofold_fallback"
        metadata["fallback_reason"] = str(e)

    # Level-1 boundary-restraint relaxation (may still be applied after calibration)
    if boundary_pairs and metadata["method"] != "rhofold_only":
        try:
            from .boundary_constraints import apply_boundary_constraints_to_coords
            coords = apply_boundary_constraints_to_coords(coords, boundary_pairs)
            if verbose:
                print(f"  [Boundary] Applied {len(boundary_pairs)} boundary constraints post-calibration")
        except Exception as e:
            if verbose:
                print(f"  [Boundary] Post-calibration constraint failed: {e}")

    return coords, conf, metadata
