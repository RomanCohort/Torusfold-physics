"""ncm_ensemble.py - infer non-canonical pairing (NCM) from ensemble distance matrices.

Idea: the consensus distance matrices of RNAbpFlow/trRNA2 are geometric evidence -
if a non-WC base pair is consistently predicted by two independent models at
~10A (WC geometry), that position most likely forms a non-canonical pair
(Hoogsteen/shear or other 3D contacts).

Complementary to the sequence heuristics in ncm_detector.py:
  - sequence heuristics: detect tandem/miniloop motifs embedded in regular secondary structure
  - distance inference: detect isolated 3D contacts with no sequence context

Usage (call after Level 0 or after merging segmented_vfold3d chunks):
    from torusfold.scheme2.ncm_ensemble import infer_ncm_from_distances
    ncm_pairs = infer_ncm_from_distances(
        dist_consensus, sequence, wc_pairs, chunk_offset=seg_start)
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

# WC target distance window: non-WC pairs predicted inside this window are NCM candidates
NCM_DIST_LO = 8.0    # Å
NCM_DIST_HI = 12.5   # Å
# Minimum sequence separation (excludes adjacent residues and short loops)
MIN_SEQ_GAP = 4
# Distance-to-confidence mapping: d=10.0A peaks confidence; the farther the deviation, the lower
NCM_REF_DIST = 10.2
NCM_CONF_MAX = 0.65   # distance-evidence cap (below the 0.7 used for tandem motifs)
NCM_CONF_MIN = 0.40


def _dist_confidence(d: float) -> float:
    """Map a predicted distance to a confidence. Gaussian kernel centered at NCM_REF_DIST."""
    sigma = 1.5
    c = np.exp(-((d - NCM_REF_DIST) ** 2) / (2 * sigma * sigma))
    return float(NCM_CONF_MIN + (NCM_CONF_MAX - NCM_CONF_MIN) * c)


def infer_ncm_from_distances(
    dist_matrix: np.ndarray,
    sequence: str,
    wc_pairs: Set[Tuple[int, int]],
    *,
    known_pairs: Optional[Set[Tuple[int, int]]] = None,
    min_gap: int = MIN_SEQ_GAP,
    max_pairs: int = 200,
) -> List[Tuple[int, int, str, float]]:
    """Infer non-canonical pairing from a consensus distance matrix.

    Args:
        dist_matrix: (L,L) consensus distance matrix (A), from trRNA2/RNAbpFlow
        sequence: ACGU string (chunk-local sequence)
        wc_pairs: known WC pairs {(i,j)}, used for exclusion
        known_pairs: other known pairs (hard/soft restraints), also excluded
        min_gap: minimum sequence separation
        max_pairs: maximum number of pairs returned (sorted by confidence, then truncated)

    Returns:
        [(gi, gj, "ENSEMBLE_DIST", confidence)] - global indices
        (a caller that passes chunk_offset translates the indices; this function
        returns local indices - see the offset-parameter variant below)
    """
    if dist_matrix is None or len(dist_matrix.shape) != 2:
        return []
    L = min(dist_matrix.shape[0], len(sequence))
    if L < min_gap * 2:
        return []

    seq_arr = np.frombuffer(sequence[:L].encode(), dtype=np.uint8)
    b1 = seq_arr[:, None]
    b2 = seq_arr[None, :]

    # neither WC nor G-U wobble
    is_canonical = np.zeros((L, L), dtype=bool)
    for a, b in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                 ("G", "U"), ("U", "G")]:
        is_canonical |= (b1 == ord(a)) & (b2 == ord(b))

    jj, ii = np.meshgrid(np.arange(L), np.arange(L))
    cand = (~is_canonical) & (jj > ii + min_gap)

    # distance window
    dist_ok = (dist_matrix >= NCM_DIST_LO) & (dist_matrix <= NCM_DIST_HI)
    cand &= dist_ok

    # exclude known pairs
    exclude = set(wc_pairs or [])
    if known_pairs:
        exclude |= set(known_pairs)

    idx_i, idx_j = np.nonzero(cand)
    results = []
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        if (min(i, j), max(i, j)) in exclude:
            continue
        # average both orientations (the distance matrix may be asymmetric)
        d = 0.5 * (float(dist_matrix[i, j]) + float(dist_matrix[j, i]))
        conf = _dist_confidence(d)
        results.append((i, j, "ENSEMBLE_DIST", round(conf, 3)))

    # sort by descending confidence, then truncate
    results.sort(key=lambda x: -x[3])
    return results[:max_pairs]


def infer_ncm_from_chunk(
    ens_dist_consensus: np.ndarray,
    seg_seq: str,
    global_wc_pairs: Set[Tuple[int, int]],
    chunk_start: int,
    overlap: int = 30,
    **kwargs,
) -> List[Tuple[int, int, str, float]]:
    """Chunk variant: local indices -> global indices.

    Args:
        ens_dist_consensus: consensus distance matrix within the chunk
        seg_seq: chunk sequence
        global_wc_pairs: global WC pair set
        chunk_start: start position of this chunk in the full sequence
        overlap: width of the chunk overlap; detections in the overlap are
            down-weighted (boundary effect)

    Returns:
        [(global i, global j, type, conf)]
    """
    L = len(seg_seq)
    # global WC pairs -> chunk-local
    local_wc = set()
    for gi, gj in global_wc_pairs:
        li, lj = gi - chunk_start, gj - chunk_start
        if 0 <= li < L and 0 <= lj < L:
            local_wc.add((li, lj))

    local_ncms = infer_ncm_from_distances(ens_dist_consensus, seg_seq, local_wc, **kwargs)

    out = []
    for li, lj, etype, conf in local_ncms:
        gi, gj = li + chunk_start, lj + chunk_start
        # down-weight the overlap region (within overlap width of either end)
        in_overlap = (li < overlap) or (lj >= L - overlap)
        if in_overlap:
            conf = round(conf * 0.7, 3)
        out.append((gi, gj, etype, conf))
    return out


def merge_chunk_ncms(
    chunk_ncm_lists: List[List[Tuple[int, int, str, float]]],
    iou_dedup: int = 0,
) -> List[Tuple[int, int, str, float]]:
    """Merge NCM detections from multiple chunks: per pair, keep the highest confidence.

    Args:
        chunk_ncm_lists: list of [(gi,gj,type,conf)] lists, one per chunk
        iou_dedup: reserved argument (for future near-neighbor deduplication)

    Returns:
        merged list, sorted by descending confidence
    """
    best: dict = {}
    for lst in chunk_ncm_lists:
        for gi, gj, etype, conf in lst:
            key = (min(gi, gj), max(gi, gj))
            if key not in best or conf > best[key][3]:
                best[key] = (key[0], key[1], etype, conf)
    merged = list(best.values())
    merged.sort(key=lambda x: -x[3])
    return merged


# -- self-test ----------------------------------------------------

if __name__ == "__main__":
    # build 20nt with an A-G Hoogsteen pair at (3,14) at ~10.2A
    rng = np.random.default_rng(42)
    L = 20
    seq = "".join(rng.choice(list("ACGU"), L))
    # force seq[3]='A', seq[14]='G'
    seq = seq[:3] + "A" + seq[4:14] + "G" + seq[15:]

    dist = rng.uniform(20, 80, (L, L))  # far by default
    np.fill_diagonal(dist, 0)
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= 1:
                dist[i, j] = abs(i - j) * 5.9
    # place the NCM pair at short distance
    dist[3, 14] = dist[14, 3] = 10.2

    wc = {(0, 19), (1, 18)}
    res = infer_ncm_from_distances(dist, seq, wc)
    hit = [r for r in res if r[0] == 3 and r[1] == 14]
    assert hit, f"expected NCM at (3,14), got {res[:5]}"
    print(f"[PASS] self-test: {hit}")
