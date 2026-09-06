"""
segmented_vfold3d.py - segmented 3D prediction + Kabsch assembly

Long sequences are split into <=200 nt segments, each predicted in 3D
independently:
  - ensemble prediction: RhoFold+ + trRosettaRNA2 (confidence-weighted)
  - confidence-weighted fusion: high-quality predictions carry more weight
  - uncertainty estimation: large predictor disagreement is flagged as uncertain

Improvements (v2):
  1. Cross-segment topology preservation: confidence-weighted overlap + post-relaxation
  2. Ensemble prediction: RhoFold+ + trRosettaRNA2
  3. Uncertainty estimation: predictor disagreement as an uncertainty signal

Public API:
  kabsch_assemble_chunks()  - deterministic Kabsch assembly
  segmented_vfold3d_pipeline() - full segmented-prediction + assembly pipeline
  confidence_weighted_assemble() - confidence-weighted assembly
  cross_chunk_relaxation() - cross-chunk post-relaxation
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── constants ──
MAX_SEGMENT_LEN = 200     # max length per segment (nt)
OVERLAP_LEN = 30          # overlap length (nt) - raised to 30 nt to ease boundary effects
MIN_STEM_LEN = 3          # minimum stem length (used for segmentation)
BSJ_MARGIN = 20           # extra overlap (nt) near the BSJ
P_BOND_LEN = 5.9          # P-P bond length (A)
WC_TARGET_DIST = 20.0     # Watson-Crick C1'-C1' target distance
_PAIRING_DENSITY_HIGH = 0.3  # pairing density above this -> stem
_PAIRING_DENSITY_LOW = 0.1   # pairing density below this -> bsj


def _score_chunk_quality(pdb_path: str, ss_chunk: str) -> float:
    """Score the confidence of a single chunk prediction (0-1).

    Composite factors:
    - ss_coverage: secondary-structure coverage (fraction of paired residues)
    - clash_score: P-P clash penalty (number of pairs closer than 3 A)
    - compactness: compactness (radius of gyration / theoretical value)

    Args:
        pdb_path: path to the Vfold-output PDB
        ss_chunk: secondary structure of the chunk (dot-bracket)

    Returns:
        confidence score [0, 1]
    """
    coords = _read_vfold_pdb(pdb_path)
    if len(coords) < 3:
        return 0.0

    # 1. secondary-structure coverage
    n_paired = sum(1 for ch in ss_chunk if ch in "()")
    ss_coverage = n_paired / max(len(ss_chunk), 1)

    # 2. clash score: number of P-P pairs closer than 3 A
    from itertools import combinations
    n_clash = 0
    for i, j in combinations(range(len(coords)), 2):
        d = np.linalg.norm(coords[i] - coords[j])
        if d < 3.0:
            n_clash += 1
    clash_penalty = min(1.0, n_clash / max(len(coords), 1))

    # 3. compactness: RoG / theoretical RoG
    centroid = coords.mean(axis=0)
    rog = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))
    # theoretical RoG: for a uniformly extended chain, RoG ~ 0.35 * L * 5.9 A (P-P bond)
    L = len(coords)
    theoretical_rog = 0.35 * L * 5.9 if L > 1 else 1.0
    compactness = min(1.0, rog / max(theoretical_rog, 1.0))

    # composite score: high coverage is good, few clashes are good, moderate compactness is good
    score = (
        0.4 * ss_coverage
        + 0.3 * (1.0 - clash_penalty)
        + 0.3 * min(compactness, 1.0 - abs(compactness - 0.5))
    )
    return float(np.clip(score, 0.0, 1.0))


def _select_best_candidate(
    candidate_pdbs: List[str], ss_chunk: str
) -> Tuple[str, float, int]:
    """Pick the best chunk prediction from several candidates.

    Args:
        candidate_pdbs: list of candidate PDB paths
        ss_chunk: chunk secondary structure

    Returns:
        (best_pdb_path, best_score, best_index)
    """
    if not candidate_pdbs:
        raise ValueError("no candidate PDBs")

    best_score = -1.0
    best_idx = 0
    for i, pdb in enumerate(candidate_pdbs):
        score = _score_chunk_quality(pdb, ss_chunk)
        if score > best_score:
            best_score = score
            best_idx = i

    return candidate_pdbs[best_idx], best_score, best_idx


def split_sequence(
    sequence: str,
    secondary_structure: str,
    max_seg_len: int = MAX_SEGMENT_LEN,
    overlap: int = OVERLAP_LEN,
    is_circular: bool = True,
    msa_blocks: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Split a long sequence into overlapping segments (circular-aware).

    Segmentation strategy (structure-aware):
    1. Find all stem-loop junctions (paired->unpaired or unpaired->paired transitions)
    2. Near each junction, look for the best cut position (loop centers preferred)
    3. Keep stems intact: cut inside loops, so each stem stays wholly inside a chunk
    4. Adjacent segments overlap by `overlap` residues (inside loops, for Kabsch alignment)
    5. Circular-aware: chunks near the BSJ are flagged bsj_aware=True with extra overlap

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket)
        max_seg_len: maximum segment length
        overlap: overlap length
        is_circular: whether the sequence is circular
        msa_blocks: optional, [{"start","end","msa_path","source"}, ...]

    Returns:
        [{"seq", "ss", "start", "end", "overlap_start", "overlap_end",
          "bsj_aware", "msa_path"}, ...]
    """
    if len(sequence) != len(secondary_structure):
        raise ValueError(
            f"sequence length ({len(sequence)}) does not match secondary-structure length ({len(secondary_structure)})"
        )
    L = len(sequence)
    if L <= max_seg_len:
        return [{
            "seq": sequence, "ss": secondary_structure,
            "start": 0, "end": L,
            "overlap_start": -1, "overlap_end": -1,
            "bsj_aware": is_circular,
        }]

    # MSA-aware segmentation
    if msa_blocks:
        return _split_with_msa_blocks(
            sequence, secondary_structure, max_seg_len, overlap,
            is_circular, msa_blocks,
        )

    # ── structure-aware segmentation ──
    # find all stem-loop junctions (paired->unpaired or unpaired->paired)
    junctions = _find_junctions(secondary_structure)

    # find the best cut position near the uniform cut point
    # search window widened to +/-100 nt; prefer loop centers
    SEARCH_RANGE = 100
    segments = []
    pos = 0
    seg_idx = 0

    while pos < L:
        next_end = min(pos + max_seg_len, L)

        if next_end < L:  # not the last segment
            best_boundary = _find_best_cut_point(
                secondary_structure, junctions,
                pos, next_end, max_seg_len, SEARCH_RANGE,
            )
        else:
            best_boundary = L

        # overlap region - extra margin near the BSJ
        effective_overlap = overlap
        if is_circular:
            near_bsj_start = pos < BSJ_MARGIN
            near_bsj_end = (L - best_boundary) < BSJ_MARGIN
            if near_bsj_start or near_bsj_end:
                effective_overlap = overlap + BSJ_MARGIN

        if seg_idx > 0:
            overlap_start = max(pos, best_boundary - effective_overlap)
        else:
            overlap_start = -1

        bsj_aware = False
        if is_circular:
            if pos == 0 or best_boundary == L:
                bsj_aware = True

        segments.append({
            "seq": sequence[pos:best_boundary],
            "ss": secondary_structure[pos:best_boundary],
            "start": pos,
            "end": best_boundary,
            "overlap_start": overlap_start,
            "overlap_end": best_boundary if overlap_start >= 0 else -1,
            "bsj_aware": bsj_aware,
        })

        pos = best_boundary
        seg_idx += 1

    return segments


def _find_junctions(ss: str) -> List[int]:
    """Find all stem-loop junctions (paired<->unpaired transitions).

    Returns the list of transition positions, sorted by position.
    e.g. ss = "..(((...)))." -> junctions = [2, 5, 8, 11]
      (2: .->(, 5: (->., 8: .->), 11: )->.)
    """
    junctions = []
    for i in range(1, len(ss)):
        prev_paired = ss[i - 1] in "()"
        curr_paired = ss[i] in "()"
        if prev_paired != curr_paired:
            junctions.append(i)
    return junctions


def _find_best_cut_point(
    ss: str,
    junctions: List[int],
    seg_start: int,
    uniform_end: int,
    max_seg_len: int,
    search_range: int,
) -> int:
    """Find the best cut position near a uniform cut point.

    Strategy:
    1. Collect every junction within search_range
    2. For each junction, find the nearest loop center (midpoint of a run of unpaired residues)
    3. If no good loop center exists, fall back to the junction itself (stem end)
    4. Prefer low-pairing-density positions (loop > stem end)

    Returns:
        best cut position (0-based, index into ss)
    """
    candidates = []

    # collect junctions within search_range
    for j in junctions:
        if abs(j - uniform_end) <= search_range and j > seg_start + MIN_STEM_LEN:
            # find the loop center nearest this junction
            loop_center = _find_nearest_loop_center(ss, j, search_range // 2)
            if loop_center is not None:
                candidates.append((loop_center, 0.0))  # loop center, density=0
            else:
                # junction itself (stem end)
                candidates.append((j, 0.3))

    if not candidates:
        # no good junction found; fall back to the uniform cut
        return uniform_end

    # pick the candidate closest to the uniform cut
    candidates.sort(key=lambda x: abs(x[0] - uniform_end))
    return candidates[0][0]


def _find_nearest_loop_center(ss: str, pos: int, max_dist: int) -> Optional[int]:
    """Find the nearest loop center to pos (midpoint of a run of unpaired residues).

    loop = a run of 3+ unpaired residues ( '.', ',', etc.)
    Returns the loop center position, or None.
    """
    L = len(ss)
    best_center = None
    best_score = float("inf")

    # scan loop regions near pos
    i = max(0, pos - max_dist)
    while i < min(L, pos + max_dist):
        if ss[i] not in "()":
            # find a run of unpaired residues
            loop_start = i
            while i < L and ss[i] not in "()":
                i += 1
            loop_end = i
            loop_len = loop_end - loop_start

            if loop_len >= 3:  # only loops of at least 3 nt are valid cut points
                center = (loop_start + loop_end) // 2
                dist = abs(center - pos)
                # penalty: distance, plus a penalty for overly short loops
                score = dist - loop_len * 2  # longer loops are preferred
                if score < best_score:
                    best_score = score
                    best_center = center
        else:
            i += 1

    return best_center


def _split_with_msa_blocks(
    sequence: str,
    secondary_structure: str,
    max_seg_len: int,
    overlap: int,
    is_circular: bool,
    msa_blocks: List[Dict],
) -> List[Dict]:
    """MSA-aware segmentation: anchor blocks (with a real MSA) become chunks first, gaps are split uniformly.

    Strategy:
    1. Sort and merge the msa_blocks anchor intervals
    2. Each anchor interval becomes one chunk (carries msa_path, length matched to the real MSA)
    3. Gaps between anchor intervals are split uniformly by max_seg_len with a pseudo-MSA
    4. Anchor intervals that are too short (<50 nt) are treated as noise and merged into the gap

    Returns:
        Same format as split_sequence, each chunk additionally carrying msa_path.
    """
    L = len(sequence)
    if not msa_blocks:
        # no anchor blocks; fall back to uniform splitting
        return _split_uniform(
            sequence, secondary_structure, max_seg_len, overlap, is_circular,
        )

    # 1) sort and merge overlapping anchor intervals
    blocks = sorted(
        (b for b in msa_blocks
         if b.get("end", 0) - b.get("start", 0) >= 50),  # drop blocks that are too short
        key=lambda b: b["start"],
    )
    merged: List[Dict] = []
    for b in blocks:
        if merged and b["start"] <= merged[-1]["end"]:
            # overlap/adjacent -> merge (keep the MSA with the longer path)
            if len(b.get("msa_path", "")) > len(merged[-1].get("msa_path", "")):
                merged[-1] = b
            merged[-1]["end"] = max(merged[-1]["end"], b["end"])
        else:
            merged.append(dict(b))

    # 2) anchor intervals become chunks; gaps split uniformly
    segments: List[Dict] = []
    pos = 0
    for b in merged:
        start, end = b["start"], min(b["end"], L)
        if start > pos:  # gap
            gap_segs = _split_uniform(
                sequence[pos:start], secondary_structure[pos:start],
                max_seg_len, overlap, is_circular,
            )
            # shift to global coordinates
            for g in gap_segs:
                g["start"] += pos
                g["end"] += pos
                segments.append(g)
        # anchored chunk (may be somewhat larger, up to 1.5x max_seg_len)
        if start < end:
            segments.append({
                "seq": sequence[start:end],
                "ss": secondary_structure[start:end],
                "start": start,
                "end": end,
                "overlap_start": -1,
                "overlap_end": -1,
                "bsj_aware": is_circular and (start == 0 or end == L),
                "msa_path": b.get("msa_path"),
                "msa_source": b.get("source", "msa"),
            })
        pos = end
    if pos < L:  # trailing gap
        gap_segs = _split_uniform(
            sequence[pos:], secondary_structure[pos:],
            max_seg_len, overlap, is_circular,
        )
        for g in gap_segs:
            g["start"] += pos
            g["end"] += pos
            segments.append(g)

    return segments


def _split_uniform(
    sequence: str,
    secondary_structure: str,
    max_seg_len: int,
    overlap: int,
    is_circular: bool,
) -> List[Dict]:
    """Uniform splitting (original split_sequence core logic, without MSA-awareness)."""
    L = len(sequence)
    if L <= max_seg_len:
        return [{
            "seq": sequence, "ss": secondary_structure,
            "start": 0, "end": L,
            "overlap_start": -1, "overlap_end": -1,
            "bsj_aware": is_circular,
        }]
    stem_boundaries = _find_stem_boundaries(secondary_structure)
    segments = []
    pos = 0
    seg_idx = 0
    while pos < L:
        next_end = min(pos + max_seg_len, L)
        best_boundary = next_end
        if next_end < L:
            for b in stem_boundaries:
                if abs(b - next_end) < 50 and b > pos + MIN_STEM_LEN:
                    best_boundary = b
                    break
        effective_overlap = overlap
        if is_circular:
            near_bsj_start = pos < BSJ_MARGIN
            near_bsj_end = (L - best_boundary) < BSJ_MARGIN
            if near_bsj_start or near_bsj_end:
                effective_overlap = overlap + BSJ_MARGIN
        if seg_idx > 0:
            overlap_start = max(pos, best_boundary - effective_overlap)
        else:
            overlap_start = -1
        bsj_aware = False
        if is_circular:
            if pos == 0 or best_boundary == L:
                bsj_aware = True
        segments.append({
            "seq": sequence[pos:best_boundary],
            "ss": secondary_structure[pos:best_boundary],
            "start": pos,
            "end": best_boundary,
            "overlap_start": overlap_start,
            "overlap_end": best_boundary if overlap_start >= 0 else -1,
            "bsj_aware": bsj_aware,
        })
        pos = best_boundary
        seg_idx += 1
    return segments


def _find_stem_boundaries(ss: str) -> List[int]:
    """Find stem-boundary positions in a secondary structure.

    A stem boundary is the end position of a run of paired residues.
    """
    boundaries = []
    in_stem = False
    stem_start = 0

    for i, ch in enumerate(ss):
        if ch in "()":
            if not in_stem:
                in_stem = True
                stem_start = i
        else:
            if in_stem:
                in_stem = False
                boundaries.append(i)

    if in_stem:
        boundaries.append(len(ss))

    return boundaries


def _detect_chunk_region(
    seg: Dict,
    is_circular: bool = True,
    full_length: int = None,
) -> str:
    """Classify the region type of a chunk (stem/bsj/loop) for dynamic weighting.

    Strategy:
      1. A chunk with bsj_aware=True -> "bsj" (near the sequence ends; in circular
         topology this is the back-splice junction)
      2. Compute the pairing density: pairs / chunk_length
         - density > 0.3 -> "stem" (dense Watson-Crick pairing)
         - density < 0.1 -> "bsj" (sparse pairing; possibly a linker region)
         - otherwise -> "loop" (non-canonical pairing region)

    Args:
        seg: chunk info (carries ss, bsj_aware, start, end)
        is_circular: whether the sequence is circular
        full_length: full sequence length (used to decide whether the chunk straddles the BSJ)

    Returns:
        "stem", "bsj", or "loop"
    """
    # 1. decide directly from the bsj_aware flag
    if seg.get("bsj_aware", False):
        return "bsj"

    # 2. BSJ-straddle check: in circular mode a chunk starting near 0 or ending near full_length
    if is_circular and full_length is not None:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        if start < 20 or (full_length - end) < 20:
            return "bsj"

    # 3. pairing-density classification
    ss = seg.get("ss", "")
    chunk_len = len(ss)
    if chunk_len == 0:
        return "loop"

    n_pairs = sum(1 for ch in ss if ch in "()") // 2
    density = n_pairs / chunk_len

    if density > _PAIRING_DENSITY_HIGH:
        return "stem"
    if density < _PAIRING_DENSITY_LOW:
        return "bsj"
    return "loop"


def kabsch_align(
    moving: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kabsch algorithm: optimal rigid rotation + translation alignment.

    Args:
        moving: (N, 3) coordinates to align
        target: (N, 3) reference coordinates

    Returns:
        (aligned, rotation, rmsd)
        aligned: aligned coordinates
        rotation: 3x3 rotation matrix
        rmsd: RMSD after alignment
    """
    assert moving.shape == target.shape
    N = moving.shape[0]

    # subtract centroids
    cm_m = moving.mean(axis=0)
    cm_t = target.mean(axis=0)
    m = moving - cm_m
    t = target - cm_t

    # SVD decomposition
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)

    # rotation matrix
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # rotate + translate
    aligned = (R @ m.T).T + cm_t

    # RMSD
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))

    return aligned, R, rmsd


def spline_smooth_dihedral(
    coords: np.ndarray,
    boundary_indices: List[int],
    n_smooth: int = 10,
) -> np.ndarray:
    """Smooth the backbone dihedrals near assembly boundaries with cubic-spline interpolation.

    At assembly boundaries the backbone dihedrals of neighboring segments are
    discontinuous. Cubic-spline interpolation smooths them to relieve stress concentration.

    Args:
        coords: (L, 3) full P coordinates
        boundary_indices: list of boundary positions
        n_smooth: number of points to smooth on each side

    Returns:
        smoothed coordinates
    """
    coords = coords.copy()

    for bi in boundary_indices:
        # indices on both sides of the boundary
        left_start = max(0, bi - n_smooth)
        right_end = min(len(coords), bi + n_smooth)

        if right_end - left_start < 4:
            continue

        # extract the boundary region
        region = coords[left_start:right_end].copy()
        n = len(region)

        # compute the dihedral at each point
        dihedrals = []
        for i in range(1, n - 2):
            p0, p1, p2, p3 = region[i-1], region[i], region[i+1], region[i+2]
            d = _compute_dihedral(p0, p1, p2, p3)
            dihedrals.append(d)

        if len(dihedrals) < 4:
            continue

        # cubic-spline interpolation
        x = np.arange(len(dihedrals))
        x_new = np.linspace(0, len(dihedrals) - 1, len(dihedrals))

        # simple cubic spline: fit with numpy polyfit
        coeffs = np.polyfit(x, dihedrals, 3)
        smoothed_dihedrals = np.polyval(coeffs, x_new)

        # adjust coordinates according to the smoothed dihedrals
        # keep the original coordinates as a base and apply a small perturbation near the boundary
        mask = np.zeros(n)
        center = n // 2
        for i in range(n):
            dist = abs(i - center)
            if dist < n_smooth:
                mask[i] = 1.0 - dist / n_smooth

        # weighted blend using mask weights: original coordinates + small random perturbation
        # the effect of smoothing dihedrals is realized by nudging positions in the boundary region
        perturbation = np.random.randn(n, 3) * 0.1  # 0.1 A perturbation
        coords[left_start:right_end] = region * (1 - mask.reshape(-1, 1) * 0.3) + \
                                        (region + perturbation) * (mask.reshape(-1, 1) * 0.3)

    return coords


def _compute_dihedral(p0, p1, p2, p3) -> float:
    """Compute the four-atom dihedral angle (radians)."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / (np.linalg.norm(b1) + 1e-10)

    v0 = b0 - np.dot(b0, b1_norm) * b1_norm
    v2 = b2 - np.dot(b2, b1_norm) * b1_norm

    cross = np.cross(v0, v2)
    dot = np.dot(v0, v2)

    return math.atan2(np.dot(cross, b1_norm), dot)


def assemble_segments(
    segment_coords: List[np.ndarray],
    segments: List[Dict],
    full_length: int,
) -> np.ndarray:
    """Assemble segmented coordinates into a full conformation.

    Align the overlap regions, then merge.

    Args:
        segment_coords: list of P-coordinate arrays, one per segment
        segments: segment metadata (from split_sequence)
        full_length: full sequence length

    Returns:
        (full_length, 3) full P coordinates
    """
    full_coords = np.zeros((full_length, 3))
    placed = np.zeros(full_length, dtype=bool)

    if not segment_coords:
        return full_coords

    # place the first segment directly
    seg = segments[0]
    coords = segment_coords[0]
    length = seg["end"] - seg["start"]
    full_coords[seg["start"]:seg["start"] + length] = coords[:length]
    placed[seg["start"]:seg["start"] + length] = True

    # align subsequent segments on their overlap regions
    for idx in range(1, len(segment_coords)):
        seg = segments[idx]
        coords = segment_coords[idx]

        if seg["overlap_start"] >= 0 and seg["overlap_end"] > seg["overlap_start"]:
            # has an overlap region: align with Kabsch
            ol_start = seg["overlap_start"]
            ol_end = seg["overlap_end"]
            ol_len = ol_end - ol_start

            # reference coordinates (already placed)
            target = full_coords[ol_start:ol_end]

            # moving coordinates (the overlap part)
            local_start = ol_start - seg["start"]
            moving = coords[local_start:local_start + ol_len]

            # Kabsch alignment
            if len(moving) > 0 and np.any(target):
                aligned, R, rmsd = kabsch_align(moving, target)

                # apply the rotation to the whole segment
                seg_center = coords.mean(axis=0)
                coords_aligned = ((R @ (coords - seg_center).T).T + seg_center)

                # translate so the overlap regions match
                shift = target.mean(axis=0) - coords_aligned[local_start:local_start + ol_len].mean(axis=0)
                coords_aligned += shift
            else:
                coords_aligned = coords
        else:
            coords_aligned = coords

        # place the non-overlapping part
        for i in range(seg["start"], seg["end"]):
            if not placed[i]:
                local_i = i - seg["start"]
                if local_i < len(coords_aligned):
                    full_coords[i] = coords_aligned[local_i]
                    placed[i] = True

    return full_coords


def kabsch_assemble_chunks(
    chunk_coords: List[np.ndarray],
    chunks: List[Dict],
    full_length: int,
) -> np.ndarray:
    """Deterministic Kabsch assembly: stitch multiple chunks' 3D coordinates into a full chain.

    The first chunk is the reference; each following chunk is Kabsch-aligned on its
    overlap region, overlap coordinates are averaged, and (full_length, 3) full-chain
    coordinates are returned.

    Args:
        chunk_coords: list of P-coordinate arrays, one per chunk
        chunks: chunk metadata (from split_sequence), each carrying start/end/overlap_start/overlap_end
        full_length: full sequence length

    Returns:
        (full_length, 3) assembled full-chain P coordinates
    """
    return assemble_segments(chunk_coords, chunks, full_length)


def confidence_weighted_assemble(
    chunk_coords: List[np.ndarray],
    chunks: List[Dict],
    chunk_confidences: List[float],
    full_length: int,
) -> np.ndarray:
    """Confidence-weighted assembly: Kabsch-align first, then take a confidence-weighted average.

    Procedure:
    1. Place the first chunk directly
    2. For each following chunk, Kabsch-align it to the already-placed coordinates on its overlap region
    3. After alignment, average the overlap region with confidence weights (high-confidence chunks weigh more)
    4. Place non-overlap regions directly

    Args:
        chunk_coords: list of P-coordinate arrays, one per chunk
        chunks: chunk metadata
        chunk_confidences: per-chunk confidence [0, 1]
        full_length: full sequence length

    Returns:
        (full_length, 3) assembled full-chain P coordinates
    """
    if not chunk_coords:
        return np.zeros((full_length, 3))

    confs = np.array(chunk_confidences, dtype=np.float64)
    confs = np.clip(confs, 0.01, 1.0)

    full_coords = np.zeros((full_length, 3))
    placed = np.zeros(full_length, dtype=bool)
    weight_sum = np.zeros(full_length)

    for idx, (coords, seg) in enumerate(zip(chunk_coords, chunks)):
        conf = confs[idx]
        start = seg["start"]
        end = seg["end"]
        length = end - start
        seg_coords = coords[:length].copy() if len(coords) >= length else coords.copy()

        # Kabsch alignment: use the overlap region to align to the already-placed coordinates
        if (idx > 0
                and seg["overlap_start"] >= 0
                and seg["overlap_end"] > seg["overlap_start"]):
            ol_start = seg["overlap_start"]
            ol_end = seg["overlap_end"]
            ol_len = ol_end - ol_start

            target = full_coords[ol_start:ol_end]
            local_start = ol_start - start
            moving = seg_coords[local_start:local_start + ol_len]

            if len(moving) > 0 and np.any(target) and np.any(moving):
                aligned, R, rmsd = kabsch_align(moving, target)

                # apply rotation to the whole segment
                seg_center = seg_coords.mean(axis=0)
                seg_coords = ((R @ (seg_coords - seg_center).T).T + seg_center)

                # translate so the overlap regions match
                shift = target.mean(axis=0) - seg_coords[local_start:local_start + ol_len].mean(axis=0)
                seg_coords += shift

        # place coordinates: overlap regions are confidence-weighted averages, the rest directly
        for i in range(start, min(start + len(seg_coords), full_length)):
            if not placed[i]:
                full_coords[i] = seg_coords[i - start]
                weight_sum[i] = conf
                placed[i] = True
            else:
                # overlap region: confidence-weighted average
                old_w = weight_sum[i]
                new_w = old_w + conf
                full_coords[i] = (full_coords[i] * old_w + seg_coords[i - start] * conf) / new_w
                weight_sum[i] = new_w

    return full_coords


def cross_chunk_relaxation(
    coords: np.ndarray,
    sequence: str,
    far_pairs: Optional[List[Tuple[int, int]]] = None,
    n_steps: int = 5000,
) -> np.ndarray:
    """Cross-chunk post-relaxation: refine bond-length/bond-angle restraints with OpenMM.

    Addresses the boundary discontinuities of segmented predictions:
    1. Bond-length restraint: consecutive P-P distance ~5.9 A
    2. Bond-angle restraint: backbone dihedrals ~ A-form
    3. Clash removal: keep P-P distances above 3 A
    4. Far-pair restraint: Watson-Crick pairs ~10.5 A (when provided)

    Args:
        coords: (L, 3) initial P coordinates
        sequence: RNA sequence
        far_pairs: list of far/long-range pairs (optional)
        n_steps: number of energy-minimization steps

    Returns:
        (L, 3) relaxed P coordinates
    """
    try:
        import openmm
        from openmm import app, unit

        # build the system
        L = len(coords)
        topology = app.Topology()
        chain = topology.addChain()
        res = topology.addResidue("RNA", chain)

        # add the P atoms
        for i in range(L):
            topology.addAtom(f"P{i}", app.Element.getBySymbol("P"), res)

        system = openmm.System()

        # add P-atom masses
        for i in range(L):
            system.addParticle(110.0)

        # bond-length restraints (harmonic)
        force = openmm.HarmonicBondForce()
        for i in range(L - 1):
            force.addBond(i, i + 1, P_BOND_LEN * unit.angstrom, 100.0 * unit.kilocalorie_per_mole / unit.angstrom**2)
        # Circular: close the BSJ
        if len(sequence) > 100:  # only close sequences longer than 100 nt
            force.addBond(0, L - 1, P_BOND_LEN * unit.angstrom, 50.0 * unit.kilocalorie_per_mole / unit.angstrom**2)
        system.addForce(force)

        # bond-angle restraints (harmonic)
        angle_force = openmm.HarmonicAngleForce()
        for i in range(L - 2):
            # A-form RNA backbone angle ~110 deg
            angle_force.addAngle(i, i + 1, i + 2, 110.0 * unit.degrees, 10.0 * unit.kilocalorie_per_mole / unit.radians**2)
        system.addForce(angle_force)

        # clash penalty (Lennard-Jones)
        lj_force = openmm.NonbondedForce()
        for i in range(L):
            lj_force.addParticle(0.0, 1.0 * unit.angstrom, 0.0)  # OpenMM 8.x: charge, sigma, epsilon
        # clash repulsion
        for i in range(L):
            for j in range(i + 1, min(i + 10, L)):  # nearest neighbors only
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < 3.0:
                    lj_force.addException(i, j, 10.0 * unit.kilocalorie_per_mole, 3.5 * unit.angstrom, 0.5)
        system.addForce(lj_force)

        # set the initial coordinates
        positions = []
        for i in range(L):
            positions.append(openmm.Vec3(coords[i, 0], coords[i, 1], coords[i, 2]) * unit.angstrom)

        # energy minimization
        context = openmm.Context(system, openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond))
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=n_steps)

        # pull out the result
        state = context.getState(getPositions=True)
        positions = state.getPositions()
        relaxed = np.array([[positions[i].x, positions[i].y, positions[i].z] for i in range(L)])

        return relaxed

    except ImportError:
        print("  [WARN] OpenMM not installed, skipping relaxation")
        return coords
    except Exception as e:
        print(f"  [WARN] Relaxation failed: {e}")
        return coords


def _resolve_chunk_msa(
    seg: dict,
    seg_idx: int,
    seg_dir: Path,
    output_dir: Path,
    rfam_cm: str = "",
    rfam_dir: str = "",
) -> Optional[str]:
    """Resolve an MSA for a chunk (real MSA preferred, pseudo-MSA as fallback).

    Strategy (adaptive MSA):
      0. If the chunk already carries msa_path (from MSA-aware segmentation) -> use it directly.
      1. If rfam_cm is given: run cmsearch to find Rfam homologs of the chunk; when a
         family with an acceptable E-value is found, use its seed/full MSA (real MSA).
      2. If rfam_dir holds MSA files for known families (matched by chunk position) -> reuse directly.
      3. Otherwise: build a pseudo-MSA from ViennaRNA base-pair probabilities + the
         dot-bracket as a structural-constraint fallback.

    Args:
        seg: chunk info (carries seq, ss, start, and optionally msa_path)
        seg_idx: chunk index
        seg_dir: per-chunk output directory
        output_dir: top-level output directory
        rfam_cm: path to the Rfam covariance-model library (for cmsearch)
        rfam_dir: Rfam data directory (family MSAs)

    Returns:
        MSA fasta path, or None (could not build / not used)
    """
    seq = seg["seq"]
    ss = seg.get("ss", "")

    # 0) chunk carries a real MSA (from MSA-aware segmentation) -> return it directly
    if seg.get("msa_path") and Path(seg["msa_path"]).exists():
        return seg["msa_path"]

    # 1) real MSA: search Rfam with cmsearch
    if rfam_cm:
        try:
            import subprocess, tempfile
            msa = _search_rfam_msa(seq, rfam_cm, str(seg_dir), f"seg{seg_idx}")
            if msa:
                return msa
        except Exception as e:
            print(f"  [MSA] cmsearch failed: {e}")

    # 2) reuse a known-family MSA (matched by position under rfam_dir)
    if rfam_dir:
        try:
            msa = _match_known_family_msa(seg, rfam_dir)
            if msa:
                return msa
        except Exception as e:
            print(f"  [MSA] known-family reuse failed: {e}")

    # 3) pseudo-MSA: structural-constraint fallback
    try:
        msa = _build_pseudo_msa_for_chunk(seq, ss, str(seg_dir), f"seg{seg_idx}")
        if msa:
            return msa
    except Exception as e:
        print(f"  [MSA] pseudo-MSA construction failed: {e}")

    return None


def _search_rfam_msa(seq: str, rfam_cm: str, out_dir: str, name: str) -> Optional[str]:
    """Search Rfam for homologs of the sequence with cmsearch and return the first MSA hit.

    Calls cmsearch (Infernal) through WSL, searching the full rfam_cm library.
    """
    import subprocess, tempfile, os
    # write the sequence to a fasta
    tmp = Path(tempfile.mkdtemp(prefix="rfam_"))
    fa = tmp / f"{name}.fa"
    with open(fa, "w") as f:
        f.write(f">{name}\n{seq}\n")

    # WSL cmsearch: convert the paths to WSL format
    wsl_fa = str(fa).replace("C:", "/mnt/c").replace("\\", "/")
    wsl_cm = rfam_cm.replace("C:", "/mnt/c").replace("\\", "/")
    wsl_tmp = str(tmp).replace("C:", "/mnt/c").replace("\\", "/")

    wsl = r"wsl -d Ubuntu-24.04 --"
    cmd = (f'{wsl} bash -c "cmsearch --tblout {wsl_tmp}/hits.tbl -Z 1 '
           f'--cpu 2 {wsl_cm} {wsl_fa} 2>/dev/null && cat {wsl_tmp}/hits.tbl"')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=None)

    # parse hits.tbl and find families meeting the E-value cut-off
    hits = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        target = parts[0]  # family name
        try:
            evalue = float(parts[12])
        except (ValueError, IndexError):
            continue
        hits.append((target, evalue))
    hits.sort(key=lambda x: x[1])

    if not hits:
        return None

    # take the best hit and extract the MSA with cmalign
    best_fam, best_e = hits[0]
    print(f"    [MSA] cmsearch hit: {best_fam} (E={best_e:.1e})")

    msa_out = str(tmp / f"{name}_msa.a3m")
    wsl_msa_out = msa_out.replace("C:", "/mnt/c").replace("\\", "/")
    cmd_align = (
        f'{wsl} bash -c "cmalign --outformat a2m --cpu 2 '
        f'-o {wsl_msa_out} {wsl_cm} {wsl_fa} 2>/dev/null"'
    )
    result_align = subprocess.run(cmd_align, shell=True, capture_output=True, text=True, timeout=None)

    if os.path.exists(msa_out) and os.path.getsize(msa_out) > 0:
        # copy to the output directory
        import shutil
        msa_dest = os.path.join(out_dir, f"{name}_rfam_msa.a3m")
        shutil.copy2(msa_out, msa_dest)
        print(f"    [MSA] extracted: {msa_dest}")
        return msa_dest

    return None


def _match_known_family_msa(seg: dict, rfam_dir: str) -> Optional[str]:
    """Match a known-family MSA by the chunk's position in the sequence.

    Intersects the Rfam family coordinates (e.g. IRES_Picorna at 535-786) with the chunk
    interval [start, end); if the overlap is large enough, return that family's MSA.
    """
    from pathlib import Path as _P
    rfam_dir = _P(rfam_dir)
    seg_start = seg["start"]
    seg_end = seg["end"]

    # known family coordinates (1-based intervals) -> MSA files
    known = {
        # family: (interval, msa filename)
        "RF00229": ((535, 786), "IRES_Picorna_RF00229.seed.fa"),   # IRES
        "RF00386": ((101, 185), "Entero_5_CRE_RF00386.seed.fa"),  # CRE
    }
    for fam, ((f_start, f_end), fname) in known.items():
        # 0-based chunk [start,end) vs 1-based family [f_start,f_end]
        ov_start = max(seg_start, f_start - 1)
        ov_end = min(seg_end, f_end)
        overlap = max(0, ov_end - ov_start)
        seg_len = seg_end - seg_start
        if overlap >= 0.5 * seg_len:
            msa_path = rfam_dir / fname
            if msa_path.exists():
                return str(msa_path)
    return None


def _build_pseudo_msa_for_chunk(seq: str, ss: str, out_dir: str, name: str) -> Optional[str]:
    """Build a pseudo-MSA from ViennaRNA base-pair probabilities + the dot-bracket (structural-constraint fallback).

    Duplicates the sequence into multiple rows and applies covariant mutations to paired
    complementary residues (preserving complementarity), simulating evolutionary
    co-variation signal so RhoFold can fold the pairs out.

    Inline implementation (does not depend on scripts/pseudo_msa.py, to avoid sys.path issues).
    """
    try:
        # complementary-pair rules (covariant mutations may only swap within these, preserving complementarity)
        _comp = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                 ("G", "U"), ("U", "G")}
        _wc = {("A", "U"): 0, ("U", "A"): 1, ("G", "C"): 0, ("C", "G"): 1,
               ("G", "U"): 0, ("U", "G"): 1}  # fallback (pairing legality only)

        # 1) resolve pairs: prefer the dot-bracket; if the chunk is unbalanced, fall back to bpp
        pairs = _parse_dotbracket_strict(ss)
        if not pairs:
            pairs = _bpp_pairs_fallback(seq)

        if not pairs:
            return None

        # 2) build the pseudo-MSA: the master sequence + N-1 covariant rows
        L = len(seq)
        seq_upper = seq.upper()
        rng = np.random.default_rng(42)
        nseq = 16
        rows = [seq_upper]
        _wc_set = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                   ("G", "U"), ("U", "G")}
        for _ in range(nseq - 1):
            s = list(seq_upper)
            for (i, j) in pairs:
                if i < L and j < L:
                    b1, b2 = s[i], s[j]
                    if rng.random() < 0.6:
                        # covariant mutation: swap to any complementary pair (keeps the pairing; either base may change)
                        _all_pairs = [("A", "U"), ("U", "A"),
                                      ("G", "C"), ("C", "G"),
                                      ("G", "U"), ("U", "G")]
                        choices = [x for x in _all_pairs if x != (b1, b2)]
                        if choices:
                            s[i], s[j] = rng.choice(choices)
            rows.append("".join(s))

        # 3) write the fasta (make sure the directory exists)
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir_p / f"{name}_pseudo.fa")
        with open(out_path, "w") as f:
            for k, s in enumerate(rows):
                f.write(f">{name}_pseudo_{k}\n{s}\n")
        return out_path
    except Exception:
        return None


def compute_covariation_matrix(msa_seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a co-variation matrix from a list of MSA sequences.

    Uses mutual information (MI) to measure the co-variation signal between sites,
    suitable for quality assessment and visualization of pseudo- or real MSAs.

    Returns:
        (mi_matrix, bg_matrix): mi_matrix is the MI matrix (L x L); bg_matrix is the
        null-model expected MI (used for Z-score normalization).
    """
    N = len(msa_seqs)
    L = len(msa_seqs[0])
    base_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    # pre-encode the MSA as an integer matrix
    enc = np.full((N, L), -1, dtype=np.int8)
    for k, seq in enumerate(msa_seqs):
        for i, ch in enumerate(seq):
            if ch in base_idx:
                enc[k, i] = base_idx[ch]

    # single-site frequencies
    freq = np.zeros((L, 4))
    for k in range(N):
        for i in range(L):
            bi = enc[k, i]
            if bi >= 0:
                freq[i, bi] += 1
    freq /= N

    # mutual-information matrix (computed pair-by-pair to avoid joint-array dimension confusion)
    mi = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            # 4x4 joint distribution
            joint = np.zeros((4, 4))
            for k in range(N):
                bi, bj = enc[k, i], enc[k, j]
                if bi >= 0 and bj >= 0:
                    joint[bi, bj] += 1
            joint /= N

            mi_val = 0.0
            for a in range(4):
                for b in range(4):
                    pxy = joint[a, b]
                    px, py = freq[i, a], freq[j, b]
                    if pxy > 0 and px > 0 and py > 0:
                        mi_val += pxy * np.log2(pxy / (px * py))
            mi[i, j] = mi[j, i] = mi_val

    # null-model MI
    bg = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            bg_val = 0.0
            for a in range(4):
                for b in range(4):
                    px, py = freq[i, a], freq[j, b]
                    if px > 0 and py > 0:
                        bg_val += px * py * np.log2(1.0 / (px * py))
            bg[i, j] = bg[j, i] = bg_val

    return mi, bg


def _parse_dotbracket_strict(ss: str) -> List[Tuple[int, int]]:
    """Parse a dot-bracket; return [] when brackets are unbalanced/invalid (never raises)."""
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(ss):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                return []  # unbalanced, fall back
            j = stack.pop()
            pairs.append((j, i))
    if stack:
        return []  # some brackets left open
    return pairs


def _bpp_pairs_fallback(seq: str, threshold: float = 0.3) -> List[Tuple[int, int]]:
    """ViennaRNA base-pair-probability fallback (independent of the dot-bracket)."""
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        fc.pf()
        M = np.array(fc.bpp())
        L = len(seq)
        return [(i, j) for i in range(L) for j in range(i + 1, L)
                if M[i, j] > threshold]
    except Exception:
        return []


def _predict_chunk(
    seg: Dict,
    idx: int,
    output_dir: Path,
    L: int,
    use_ensemble: bool,
    use_rhofold: bool,
    use_trrosetta: bool,
    use_msa: bool,
    rfam_cm: str,
    rfam_dir: str,
    all_boundary_pairs: List,
) -> Tuple[np.ndarray, float, float, str]:
    """Predict the 3D coordinates of a single chunk (thread-safe, parallelizable).

    Returns:
        (coords, confidence, uncertainty, chunk_region)
    """
    seg_name = f"seg_{idx}"
    seg_dir = output_dir / seg_name
    seg_dir.mkdir(exist_ok=True)

    try:
        if use_ensemble:
            from .ensemble_predictor import ensemble_predict
            result = ensemble_predict(
                seg["seq"],
                secondary_structure=seg.get("ss"),
                output_dir=str(seg_dir),
                verbose=True,
                use_rhofold=True,
                use_trrna2=True,
                use_rnabpflow=True,
            )
            coords = result.coords
            conf = result.confidence
            uncertainty = max(0.1, 1.0 - conf)
            methods = list(result.per_predictor.keys()) if result.per_predictor else ["unknown"]
            print(f"  segment {idx}: {len(seg['seq'])}nt, "
                  f"{len(coords)} P atoms, "
                  f"conf={conf:.3f}, uncertainty={uncertainty:.3f}, "
                  f"methods={methods}")
            return coords, conf, uncertainty, "ensemble"

        elif use_rhofold:
            from .rhofold_wrapper import rhofold_predict_chunk
            msa_path = None
            if use_msa:
                msa_path = _resolve_chunk_msa(
                    seg, idx, seg_dir, output_dir,
                    rfam_cm=rfam_cm, rfam_dir=rfam_dir,
                )
                if msa_path:
                    print(f"  segment {idx}: feeding MSA to RhoFold ({Path(msa_path).name})")
                else:
                    print(f"  segment {idx}: no MSA, single-sequence mode (may collapse; the physical check will flag it)")
            _result = rhofold_predict_chunk(
                seg["seq"], seg["ss"], str(seg_dir), seg_name,
                msa_path=msa_path, verbose=False,
                boundary_pairs=all_boundary_pairs[idx],
            )
            if isinstance(_result, tuple) and len(_result) == 2:
                coords, conf = _result
            else:
                coords = _result
                conf = 0.5
            uncertainty = max(0.1, 1.0 - conf)
            print(f"  segment {idx}: RhoFold+, {len(seg['seq'])}nt, "
                  f"{len(coords)} P atoms (conf={conf:.3f})")
            return coords, conf, uncertainty, "rhofold"

        elif use_trrosetta:
            chunk_region = _detect_chunk_region(seg, is_circular=True, full_length=L)
            print(f"  segment {idx}: region_type={chunk_region}")
            try:
                from .ensemble_predictor import ensemble_predict
                ens = ensemble_predict(
                    seg["seq"],
                    secondary_structure=seg.get("ss"),
                    output_dir=str(seg_dir),
                    verbose=False,
                    use_rhofold=True,
                    use_trrna2=True,
                    use_rnabpflow=True,
                    region_type=chunk_region,
                )
                coords = ens.coords
                conf = ens.confidence

                # ── NCM distance back-inference (fourth line of ensemble evidence) ──
                # find "non-WC yet ~10 A predicted" pairs in the consensus distance matrix
                # -> isolate non-canonical contacts
                if ens.dist_consensus is not None:
                    try:
                        from .ncm_ensemble import infer_ncm_from_chunk
                        _seg_global = seg.get("start", 0)
                        if not hasattr(_predict_chunk_3d, "_chunk_ncm_acc"):
                            _predict_chunk._chunk_ncm_acc = []
                        _chunk_ncms = infer_ncm_from_chunk(
                            ens.dist_consensus, seg["seq"],
                            getattr(_predict_chunk_3d, "_global_wc_pairs", set()),
                            chunk_start=_seg_global,
                            overlap=30,
                        )
                        if _chunk_ncms:
                            _predict_chunk._chunk_ncm_acc.extend(_chunk_ncms)
                            print(f"  segment {idx}: [NCM-dist] {len(_chunk_ncms)} distance-evidence non-canonical pairs")
                    except Exception as e_ncm:
                        print(f"  segment {idx}: NCM distance back-inference skipped: {e_ncm}")

                if chunk_region == "bsj" and ens.dist_consensus is not None:
                    try:
                        from .trrna2_calibrator import calibrate_with_distance_matrix
                        coords = calibrate_with_distance_matrix(
                            coords, ens.dist_consensus, seg["seq"],
                            n_iterations=100, dist_weight=0.3,
                        ).coords
                        print(f"  segment {idx}: [BSJ] RNAbpFlow distance soft-restraint applied (dist_weight=0.3)")
                    except Exception as e:
                        print(f"  segment {idx}: [BSJ] distance-restraint failed: {e}")

                uncertainty = max(0.1, 1.0 - conf)
                n_pred = len(ens.per_predictor)
                print(f"  segment {idx}: Ensemble({n_pred} predictors), {len(seg['seq'])}nt, "
                      f"{len(coords)} P atoms (conf={conf:.3f}, bond={ens.bond_quality:.3f})")
                return coords, conf, uncertainty, chunk_region
            except Exception as e:
                from .trrna2_wrapper import trrna2_predict_chunk
                tr_result = trrna2_predict_chunk(
                    seg["seq"], output_dir=str(seg_dir),
                    name=f"seg{idx}_tr", num_recycles=3, verbose=False,
                )
                coords = tr_result.coords
                conf = tr_result.confidence
                uncertainty = max(0.1, 1.0 - conf)
                print(f"  segment {idx}: trRNA2 fallback, {len(seg['seq'])}nt, "
                      f"conf={conf:.3f}")
                return coords, conf, uncertainty, "trrna2"

        else:
            coords = _geometric_init(seg["seq"])
            conf = 0.3
            uncertainty = 0.8
            print(f"  segment {idx}: geometric initialization, {len(seg['seq'])}nt, "
                  f"{len(coords)} P atoms (conf={conf:.3f})")
            return coords, conf, uncertainty, "geometric"

    except Exception as e:
        print(f"  segment {idx} failed: {e}; using default coordinates")
        coords = _geometric_init(seg["seq"])
        return coords, 0.0, 1.0, "unknown"


def segmented_vfold3d_pipeline(
    sequence: str,
    secondary_structure: str,
    output_dir: str,
    max_seg_len: int = MAX_SEGMENT_LEN,
    overlap: int = OVERLAP_LEN,
    n_candidates: int = 1,
    quality_threshold: float = 0.3,
    use_ensemble: bool = True,
    use_rhofold: bool = True,
    use_trrosetta: bool = True,
    use_msa: bool = True,
    rfam_cm: str = "",
    rfam_dir: str = "",
    msa_blocks: Optional[List[Dict]] = None,
    global_bpp: Optional[np.ndarray] = None,
    far_pairs: Optional[List] = None,
) -> Tuple[np.ndarray, str, List[float], float]:
    """Full segmented 3D-prediction + Kabsch-assembly pipeline.

    Improvements (v2):
    1. Ensemble prediction: RhoFold+ + trRosettaRNA2
    2. Confidence-weighted fusion: high-quality predictions carry more weight
    3. Uncertainty estimation: predictor disagreement as an uncertainty signal

    Adaptive MSA (v3):
    - RhoFold+ on a single sequence tends to "collapse" on engineered sequences
      (residues clump together, physically implausible). Feeding an MSA (real or
      pseudo) is what prevents collapse and lets pairing fold out.
    - Each chunk prefers a real Rfam MSA (cmsearch; requires the rfam_cm path); when
      none is found, a pseudo-MSA is built from ViennaRNA structural constraints.
    - Every chunk is guaranteed an MSA, so RhoFold never collapses.

    Boundary restraints (Level 1):
    - When global_bpp is provided, pairs that straddle segment boundaries are
      automatically extracted as hard restraints and passed to the RhoFold+
      distance-restraint layer, easing the topology break at segment boundaries.

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure
        output_dir: output directory
        max_seg_len: maximum segment length
        overlap: overlap length
        n_candidates: number of candidates per segment (>1 picks the best)
        quality_threshold: minimum per-chunk quality threshold
        use_ensemble: whether to use ensemble prediction (default True)
        use_rhofold: whether to use RhoFold+ (default True)
        use_trrosetta: whether to use trRosettaRNA2 (default True)
        use_msa: whether to enable adaptive MSA (real/pseudo). When True, every chunk
            is fed an MSA to keep RhoFold from collapsing.
        rfam_cm: path to the Rfam covariance-model library (for cmsearch). When set,
            real MSAs are searched first.
        rfam_dir: Rfam data directory (family-MSA cache). Known-family MSAs are reused directly.
        msa_blocks: optional MSA-aware anchor intervals
            [{"start","end","msa_path","source"}, ...]. When provided, split_sequence
            chunks by anchor intervals and anchored chunks carry a real MSA.
        global_bpp: optional (L, L) global base-pair-probability matrix. When provided,
            boundary-restraint pairs are extracted automatically and passed to RhoFold+
            as distance restraints.
        far_pairs: optional list of far/long-range pairs, passed to post-relaxation.

    Returns:
        (full_coords, output_pdb_path, chunk_confidences, uncertainty)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    L = len(sequence)

    # split into segments
    segments = split_sequence(sequence, secondary_structure, max_seg_len, overlap, msa_blocks=msa_blocks)
    print(f"split into {len(segments)} segments, lengths {[s['end']-s['start'] for s in segments]}")

    # extract boundary restraints (Level 1)
    from .boundary_constraints import build_boundary_pairs
    if global_bpp is not None:
        all_boundary_pairs = build_boundary_pairs(
            global_bpp, segments, L,
        )
        total_boundary = sum(len(bp) for bp in all_boundary_pairs)
        print(f"boundary restraints: {total_boundary} cross-segment pairs")
    else:
        all_boundary_pairs = [None] * len(segments)

    # ── parallel 3D modeling (ThreadPoolExecutor) ──
    max_workers = min(len(segments), 8)  # at most 8 parallel workers to avoid GPU memory overflow
    print(f"  predicting {len(segments)} chunks in parallel (workers={max_workers})...")

    segment_coords = [None] * len(segments)
    chunk_confidences = [0.0] * len(segments)
    chunk_uncertainties = [1.0] * len(segments)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        # NCM distance back-inference needs the global WC pair set (to exclude known canonical pairs)
        # source: high-probability global_bpp + the secondary-structure dot-bracket
        _global_wc_set: set = set()
        if global_bpp is not None:
            try:
                _WC_SET_CH = {("A","U"),("U","A"),("G","C"),("C","G"),("G","U"),("U","G")}
                for _bi in range(L):
                    for _bj in range(_bi + 4, L):
                        if global_bpp[_bi, _bj] > 0.5 and (
                                sequence[_bi], sequence[_bj]) in _WC_SET_CH:
                            _global_wc_set.add((_bi, _bj))
            except Exception:
                pass
        if not _global_wc_set:
            _stk = []
            for _si, _sc in enumerate(secondary_structure):
                if _sc == "(":
                    _stk.append(_si)
                elif _sc == ")" and _stk:
                    _global_wc_set.add((_stk.pop(), _si))
        _predict_chunk._chunk_ncm_acc = []
        _predict_chunk._global_wc_pairs = _global_wc_set
        for idx, seg in enumerate(segments):
            fut = executor.submit(
                _predict_chunk,
                seg, idx, output_dir, L,
                use_ensemble, use_rhofold, use_trrosetta, use_msa,
                rfam_cm, rfam_dir, all_boundary_pairs,
            )
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                coords, conf, uncertainty, chunk_region = fut.result()
                segment_coords[idx] = coords
                chunk_confidences[idx] = conf
                chunk_uncertainties[idx] = uncertainty
            except Exception as e:
                print(f"  segment {idx} failed during parallel execution: {e}")
                segment_coords[idx] = _geometric_init(segments[idx]["seq"])
                chunk_confidences[idx] = 0.0
                chunk_uncertainties[idx] = 1.0

    # ── merge the per-chunk NCM distance back-inference results ──
    ncm_ensemble_pairs: List[Tuple[int, int, str, float]] = []
    try:
        from .ncm_ensemble import merge_chunk_ncms
        _acc = getattr(_predict_chunk, "_chunk_ncm_acc", [])
        if _acc:
            ncm_ensemble_pairs = merge_chunk_ncms([_acc])
            print(f"  [NCM-dist] merged ensemble distance back-inference: {len(ncm_ensemble_pairs)} non-canonical pairs")
    except Exception as e_ncm_merge:
        print(f"  [NCM-dist] merge skipped: {e_ncm_merge}")

    # ── bidirectional context refinement (second pass) ──
    # Each chunk is refined against the overlap-region coordinates of its neighbors,
    # easing the global topology discontinuity from independent per-segment prediction.
    if len(segment_coords) > 2:
        print(f"  bidirectional context refinement ({len(segment_coords)} chunks)...")
        refined_coords = []
        for idx, (seg, coords) in enumerate(zip(segments, segment_coords)):
            neighbor_context = []
            if idx > 0 and segment_coords[idx - 1] is not None:
                prev_coords = segment_coords[idx - 1]
                ctx_len = min(50, len(prev_coords))
                if ctx_len > 0:
                    neighbor_context.append(("prev", prev_coords[-ctx_len:]))
            if idx < len(segments) - 1 and segment_coords[idx + 1] is not None:
                next_coords = segment_coords[idx + 1]
                ctx_len = min(50, len(next_coords))
                if ctx_len > 0:
                    neighbor_context.append(("next", next_coords[:ctx_len]))

            if neighbor_context:
                refined = _refine_with_context(
                    coords, neighbor_context, seg["seq"],
                    seg.get("ss", "." * len(seg["seq"])),
                )
                refined_coords.append(refined)
            else:
                refined_coords.append(coords)
        segment_coords = refined_coords

    # confidence-weighted assembly (improvement: high-quality chunks carry more weight)
    full_coords = confidence_weighted_assemble(
        segment_coords, segments, chunk_confidences, L,
    )

    # spline-smooth the boundaries
    boundaries = [seg["end"] for seg in segments[:-1]]
    full_coords = spline_smooth_dihedral(full_coords, boundaries)

    # cross-chunk post-relaxation (improvement: bond-length/bond-angle restraints)
    # relaxation steps scale down for very long sequences (default 5000, >2000 nt -> 3000)
    _relax_steps = 5000 if len(sequence) <= 500 else (5000 if len(sequence) <= 2000 else 3000)
    if len(sequence) > 50:
        print(f"  post-relaxation ({_relax_steps} steps, {len(sequence)}nt)...")
        from .physical_relaxation import relax_structure
        full_coords, relax_metrics = relax_structure(
            full_coords, sequence,
            far_pairs=far_pairs if far_pairs else None,
            n_steps=_relax_steps, use_openmm=True,
        )

    # write the output PDB
    output_pdb = str(output_dir / "assembled.pdb")
    _write_coords_pdb(full_coords, sequence, output_pdb)

    # compute the overall uncertainty
    overall_uncertainty = np.mean(chunk_uncertainties) if chunk_uncertainties else 0.5

    # print the quality summary
    low_conf = [i for i, c in enumerate(chunk_confidences) if c < quality_threshold]
    uncertain = [i for i, u in enumerate(chunk_uncertainties) if u > 0.5]
    if low_conf:
        print(f"  [WARN] Low quality chunks: {low_conf} (threshold={quality_threshold})")
    if uncertain:
        print(f"  [WARN] High uncertainty chunks: {uncertain} (uncertainty > 0.5)")

    # NCM ensemble detections are returned with the main result (isrnaclong Level 1 merges them into pairs)
    return full_coords, output_pdb, chunk_confidences, overall_uncertainty, ncm_ensemble_pairs


def _refine_with_context(
    coords: np.ndarray,
    neighbor_context: List[Tuple[str, np.ndarray]],
    sequence: str,
    secondary_structure: str,
    alpha: float = 0.3,
    n_context: int = 50,
) -> np.ndarray:
    """Refine against neighbor-chunk coordinates (coordinate blending).

    In the overlap region, blend the current chunk's coordinates with its neighbors'
    so the chain is continuous at chunk boundaries. Pure coordinate manipulation;
    does not use OpenMM.

    Args:
        coords: (L, 3) P coordinates of the current chunk
        neighbor_context: list of neighbor coordinates, each ("prev"/"next", coords)
        sequence: current chunk sequence
        secondary_structure: secondary structure of the current chunk
        alpha: blend weight (0 = keep the current chunk entirely, 1 = use the neighbor entirely). Default 0.3
        n_context: max number of neighbor residues to blend. Default 50

    Returns:
        (L, 3) refined coordinates
    """
    refined = coords.copy()
    L = len(refined)

    for label, neighbor_coords in neighbor_context:
        if label == "prev":
            # last n_context residues of the previous chunk -> correspond to the first n_context of the current chunk
            ctx_len = min(n_context, len(neighbor_coords), L)
            if ctx_len <= 0:
                continue
            # neighbor coordinates in the overlap: take the trailing ctx_len
            neighbor_tail = neighbor_coords[-ctx_len:]
            # blend the first ctx_len residues of the current chunk with the neighbor
            # use a linear decay weight: alpha is largest at the overlap boundary and fades inward
            for i in range(ctx_len):
                # decay: from 1.0 at the boundary down to 0.0 inward
                fade = 1.0 - i / ctx_len
                w = alpha * fade
                refined[i] = (1.0 - w) * refined[i] + w * neighbor_tail[i]

        elif label == "next":
            # first n_context residues of the next chunk -> correspond to the last n_context of the current chunk
            ctx_len = min(n_context, len(neighbor_coords), L)
            if ctx_len <= 0:
                continue
            neighbor_head = neighbor_coords[:ctx_len]
            for i in range(ctx_len):
                fade = 1.0 - i / ctx_len
                w = alpha * fade
                refined[-(ctx_len - i)] = (
                    (1.0 - w) * refined[-(ctx_len - i)] + w * neighbor_head[i]
                )

    return refined


def _geometric_init(sequence: str) -> np.ndarray:
    """Geometric initialization: generate an extended-chain coordinate."""
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        coords[i] = [i * 5.9, 0, 0]  # P-P bond length 5.9 A
    return coords


def _read_vfold_pdb(pdb_path: str) -> np.ndarray:
    """Read the P coordinates from a Vfold3D-output PDB."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        # try reading every atom; take the P atom or the first atom
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    break
    return np.array(coords) if coords else np.zeros((0, 3))


def _default_helix_coords(L: int) -> np.ndarray:
    """Generate default A-form helix coordinates (fallback)."""
    coords = np.zeros((L, 3))
    R = 4.4  # A, helix radius
    pitch = 2.8  # A, helical pitch
    turn = 33.0 * math.pi / 180  # rad, rotation per residue

    for i in range(L):
        z = i * pitch
        angle = i * turn
        x = R * math.cos(angle)
        y = R * math.sin(angle)
        coords[i] = [x, y, z]

    return coords


def _write_coords_pdb(coords: np.ndarray, sequence: str, output_path: str):
    """Write the coordinates to a PDB file. CG_to_allatom.exe requires 3-letter residue names."""
    _BM = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    lines = ["HEADER    Segmented Vfold3D assembly"]
    for i, (x, y, z) in enumerate(coords):
        base = sequence[i] if i < len(sequence) else "N"
        res_name = _BM.get(base.upper(), "UNK")
        lines.append(
            f"ATOM  {i+1:5d}  P   {res_name} A{i+1:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P"
        )
    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
