"""
overlap_confidence.py - Level-1 overlap-region confidence assessment

Evaluate the consistency of adjacent chunk overlap regions in segmented
prediction:
  - RMSD of two chunks over the overlap -> confidence score
  - multi-sample consistency -> flexible-region detection
  - per-residue confidence map -> Level-2.3 force-field parameterization

Confidence tiers:
  RMSD < 3 A  -> high confidence (force_scale=1.0)
  3-8 A       -> medium confidence (force_scale=0.5)
  > 8 A       -> flexible region (force_scale=0.1, left to the force field)

Public API:
  evaluate_overlap_confidence()  - evaluate a single chunk-pair overlap
  multi_sample_confidence()      - multi-sample consistency assessment
  segment_confidence_map()       - per-residue confidence map
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Constants ──
HIGH_CONF_THRESHOLD = 3.0      # RMSD < 3 A -> high confidence
FLEXIBLE_THRESHOLD = 8.0       # RMSD > 8 A -> flexible region
DEFAULT_CONFIDENCE_TEMP = 3.0  # sigmoid temperature


@dataclass
class OverlapConfidence:
    """Overlap-region confidence evaluation result.

    Attributes:
        rmsd: RMSD of the overlap region (Angstrom)
        confidence: confidence [0, 1], 1.0/(1+rmsd/temp)
        is_flexible: whether this is a flexible region (rmsd > 8 A)
        force_scale: force-field scale factor (high/medium/flexible)
        n_residues: number of residues in the overlap region
    """
    rmsd: float
    confidence: float
    is_flexible: bool
    force_scale: float
    n_residues: int

    @property
    def tier(self) -> str:
        """Confidence tier: high / medium / flexible."""
        if self.rmsd < HIGH_CONF_THRESHOLD:
            return "high"
        elif self.rmsd < FLEXIBLE_THRESHOLD:
            return "medium"
        else:
            return "flexible"


def _kabsch_rmsd(moving: np.ndarray, target: np.ndarray) -> float:
    """Compute the RMSD after Kabsch alignment.

    Args:
        moving: (N, 3) coordinates to align
        target: (N, 3) reference coordinates

    Returns:
        aligned RMSD (Angstrom)
    """
    assert moving.shape == target.shape, (
        f"Shape mismatch: {moving.shape} vs {target.shape}"
    )
    n = moving.shape[0]
    if n == 0:
        return 0.0

    # Center the coordinates
    centroid_m = moving.mean(axis=0)
    centroid_t = target.mean(axis=0)
    m = moving - centroid_m
    t = target - centroid_t

    # SVD for the optimal rotation
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T

    # Align + RMSD
    m_aligned = (R @ m.T).T
    diff = m_aligned - t
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def evaluate_overlap_confidence(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    overlap_indices: np.ndarray,
    temperature: float = DEFAULT_CONFIDENCE_TEMP,
) -> OverlapConfidence:
    """Evaluate the consistency of two chunks over the overlap region.

    Extracts the overlap coordinates of coords_a and coords_b, performs an
    optimal Kabsch alignment, computes the RMSD, and maps it to a confidence.

    Args:
        coords_a: chunk_a predicted coordinates (L_a, 3), including the overlap
        coords_b: chunk_b predicted coordinates (L_b, 3), including the overlap
        overlap_indices: global indices of the overlap region in the full chain
        temperature: sigmoid temperature; smaller is stricter

    Returns:
        An OverlapConfidence evaluation result
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    overlap_indices = np.asarray(overlap_indices, dtype=np.int64)

    n_residues = len(overlap_indices)
    if n_residues < 3:
        # Overlap too short for a reliable evaluation
        return OverlapConfidence(
            rmsd=0.0, confidence=1.0, is_flexible=False,
            force_scale=1.0, n_residues=n_residues,
        )

    # Extract the overlap-region coordinates - the passed coords are assumed to already be
    # the coordinates of the corresponding chunk. overlap_indices are global indices, but
    # coords_a/b are local to each chunk, so the caller must ensure overlap_indices map to
    # the correct local indices. If overlap_indices are global and a chunk starts at some
    # offset, the indices are used directly as valid indices into both coordinate sets.
    # In practice, the mapping should be applied before calling this function.

    # If coords_a and coords_b are already aligned to the same overlap region:
    if coords_a.shape[0] == coords_b.shape[0]:
        ol_a = coords_a
        ol_b = coords_b
    else:
        # Index by overlap_indices (the coords must be long enough)
        max_idx = max(overlap_indices.max() + 1, 0)
        if coords_a.shape[0] >= max_idx and coords_b.shape[0] >= max_idx:
            ol_a = coords_a[overlap_indices]
            ol_b = coords_b[overlap_indices]
        else:
            # Fallback: align by the min length
            common = min(coords_a.shape[0], coords_b.shape[0], n_residues)
            ol_a = coords_a[:common]
            ol_b = coords_b[:common]

    # Kabsch RMSD
    rmsd = _kabsch_rmsd(ol_a, ol_b)

    # Confidence: sigmoid mapping
    confidence = 1.0 / (1.0 + rmsd / max(temperature, 1e-6))

    # Flexible-region decision
    is_flexible = rmsd > FLEXIBLE_THRESHOLD

    # Force-field scale factor
    if rmsd < HIGH_CONF_THRESHOLD:
        force_scale = 1.0
    elif rmsd < FLEXIBLE_THRESHOLD:
        # Linear interpolation: 3 A -> 1.0, 8 A -> 0.5
        force_scale = 1.0 - 0.5 * (rmsd - HIGH_CONF_THRESHOLD) / (
            FLEXIBLE_THRESHOLD - HIGH_CONF_THRESHOLD
        )
    else:
        force_scale = 0.1

    return OverlapConfidence(
        rmsd=float(rmsd),
        confidence=float(confidence),
        is_flexible=is_flexible,
        force_scale=float(force_scale),
        n_residues=n_residues,
    )


def multi_sample_confidence(
    sequence: str,
    predict_fn: Callable[[str, int], np.ndarray],
    overlap_indices: np.ndarray,
    n_samples: int = 5,
    temperature: float = DEFAULT_CONFIDENCE_TEMP,
) -> OverlapConfidence:
    """Evaluate overlap-region consistency across multiple samples.

    Runs n_samples predictions on the same sequence with different random seeds
    and computes the spread of the overlap coordinates; high consistency leads
    to high confidence, low consistency to a flexible region.

    Rationale: if a region is structurally determined (e.g., a stem), repeated
    predictions should converge; if it is flexible (e.g., a loop), they diverge.

    Args:
        sequence: RNA sequence
        predict_fn: prediction function (sequence, seed) -> (L, 3) coordinates
        overlap_indices: global indices of the overlap region
        n_samples: number of samples
        temperature: sigmoid temperature

    Returns:
        An OverlapConfidence result based on sampling consistency
    """
    overlap_indices = np.asarray(overlap_indices, dtype=np.int64)
    n_residues = len(overlap_indices)

    if n_residues < 3 or n_samples < 2:
        return OverlapConfidence(
            rmsd=0.0, confidence=1.0, is_flexible=False,
            force_scale=1.0, n_residues=n_residues,
        )

    # Multi-sample predictions
    samples: List[np.ndarray] = []
    for i in range(n_samples):
        seed = 42 + i  # fixed seeds keep results reproducible; different seeds add diversity
        coords = predict_fn(sequence, seed)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape[0] > overlap_indices.max():
            samples.append(coords[overlap_indices])
        elif coords.shape[0] > 0:
            # Coordinates too short; take the valid part
            valid = min(coords.shape[0], n_residues)
            samples.append(coords[:valid])

    if len(samples) < 2:
        return OverlapConfidence(
            rmsd=0.0, confidence=0.5, is_flexible=False,
            force_scale=0.5, n_residues=n_residues,
        )

    # Compute pairwise RMSD
    rmsds: List[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            min_len = min(samples[i].shape[0], samples[j].shape[0])
            if min_len >= 3:
                rmsd = _kabsch_rmsd(samples[i][:min_len], samples[j][:min_len])
                rmsds.append(rmsd)

    if not rmsds:
        return OverlapConfidence(
            rmsd=0.0, confidence=0.5, is_flexible=False,
            force_scale=0.5, n_residues=n_residues,
        )

    # Use the mean RMSD as the consistency metric
    mean_rmsd = float(np.mean(rmsds))
    confidence = 1.0 / (1.0 + mean_rmsd / max(temperature, 1e-6))
    is_flexible = mean_rmsd > FLEXIBLE_THRESHOLD

    if mean_rmsd < HIGH_CONF_THRESHOLD:
        force_scale = 1.0
    elif mean_rmsd < FLEXIBLE_THRESHOLD:
        force_scale = 1.0 - 0.5 * (mean_rmsd - HIGH_CONF_THRESHOLD) / (
            FLEXIBLE_THRESHOLD - HIGH_CONF_THRESHOLD
        )
    else:
        force_scale = 0.1

    return OverlapConfidence(
        rmsd=mean_rmsd,
        confidence=confidence,
        is_flexible=is_flexible,
        force_scale=force_scale,
        n_residues=n_residues,
    )


def segment_confidence_map(
    chunks: List[Dict],
    chunk_coords: List[np.ndarray],
    overlaps: List[Dict],
    full_length: int,
) -> np.ndarray:
    """Build a per-residue confidence map.

    Iterates over all chunks and overlap regions, computing a confidence for
    each residue:
    - non-overlap regions: use the chunk's own quality score (from _score_chunk_quality)
    - overlap regions: use the confidence mapped from the two-chunk RMSD
    - residues not covered by any chunk: default 0.0

    The output feeds Level-2.3 force-field parameterization: force_scale per residue.

    Args:
        chunks: list of segmentation info (from split_sequence)
        chunk_coords: P coordinates of each chunk
        overlaps: list of overlap info, each containing:
            - "indices": global indices of the overlap region
            - "chunk_a": left chunk index
            - "chunk_b": right chunk index
        full_length: length of the full sequence

    Returns:
        (full_length,) per-residue confidence [0, 1]
    """
    confidence = np.zeros(full_length, dtype=np.float64)
    weight = np.zeros(full_length, dtype=np.float64)

    # 1. Non-overlap regions: the chunk's own quality score
    for idx, (seg, coords) in enumerate(zip(chunks, chunk_coords)):
        start = seg["start"]
        end = seg["end"]
        length = end - start
        seg_coords = coords[:length] if len(coords) >= length else coords

        # Simple quality metric: smaller coordinate variance means higher certainty
        if seg_coords.shape[0] >= 3:
            # Use the variance of adjacent-residue distances as the certainty metric
            diffs = np.diff(seg_coords, axis=0)
            bond_lengths = np.linalg.norm(diffs, axis=1)
            # Normal P-P bond length ~5.9 A; smaller variance means higher certainty
            if len(bond_lengths) > 1:
                bl_var = float(np.var(bond_lengths))
                # Mapping: variance 0 -> 1.0, variance 5 -> 0.3
                chunk_conf = max(0.3, 1.0 - bl_var / 5.0)
            else:
                chunk_conf = 0.5
        else:
            chunk_conf = 0.5

        for i in range(start, min(start + len(seg_coords), full_length)):
            confidence[i] += chunk_conf
            weight[i] += 1.0

    # 2. Overlap regions: two-chunk RMSD -> confidence (higher weight)
    for ol in overlaps:
        ol_indices = np.asarray(ol["indices"], dtype=np.int64)
        idx_a = ol["chunk_a"]
        idx_b = ol["chunk_b"]

        if idx_a >= len(chunk_coords) or idx_b >= len(chunk_coords):
            continue

        coords_a = chunk_coords[idx_a]
        coords_b = chunk_coords[idx_b]
        seg_a = chunks[idx_a]
        seg_b = chunks[idx_b]

        # Map global overlap indices to chunk-local coordinates
        ol_local_a = ol_indices - seg_a["start"]
        ol_local_b = ol_indices - seg_b["start"]

        # Extract the overlap-region coordinates
        valid_a = (ol_local_a >= 0) & (ol_local_a < coords_a.shape[0])
        valid_b = (ol_local_b >= 0) & (ol_local_b < coords_b.shape[0])
        valid = valid_a & valid_b

        if valid.sum() < 3:
            continue

        ol_coords_a = coords_a[ol_local_a[valid]]
        ol_coords_b = coords_b[ol_local_b[valid]]

        ol_conf = evaluate_overlap_confidence(
            ol_coords_a, ol_coords_b,
            np.arange(valid.sum()),
        )

        # Overlap-region confidence overrides with 2x weight
        for i, global_idx in enumerate(ol_indices[valid]):
            confidence[global_idx] = ol_conf.confidence
            weight[global_idx] = 2.0  # overlap regions get a higher weight

    # Normalize
    weight = np.maximum(weight, 1e-8)
    confidence = confidence / weight

    return np.clip(confidence, 0.0, 1.0)


def force_scale_from_confidence(confidence: float) -> float:
    """Map a confidence score to a force-field scale factor.

    Args:
        confidence: confidence in [0, 1]

    Returns:
        force_scale: 0.1 (flexible) ~ 1.0 (high confidence)
    """
    if confidence >= 0.75:
        return 1.0
    elif confidence >= 0.3:
        # Linear interpolation: 0.3 -> 0.5, 0.75 -> 1.0
        return 0.5 + 0.5 * (confidence - 0.3) / 0.45
    else:
        return 0.1
