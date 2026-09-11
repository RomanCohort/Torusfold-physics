"""
rcm.py - Reverse Complementary Match (RCM) computation module.

Fast complex cumulative-sum algorithm based on CircCNNs
(Wang & Liang, 2024) for detecting reverse complementary matches between
introns flanking the BSJ of a circRNA.

Core idea:
  nucleotide -> complex mapping (A=1, T=-1, C=1j, G=-1j)
  cumulative-sum vectors to score all kmers quickly
  outer-product broadcasting to detect RCM kmer pairs between two sequences

Applications:
  - crossing RCM: upstream vs downstream introns around the BSJ -> promotes back-splicing
  - within RCM: within a single intron -> promotes linear splicing (competing)
  - RCM score injected into the pipeline as a base-pair confidence weight

Reference: Wang & Liang, Scientific Reports 14:18982 (2024)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


# ── nucleotide -> complex mapping ──
_BASE_MAP = {
    'A': 1.0,   'a': 1.0,
    'T': -1.0,  't': -1.0,  'U': -1.0, 'u': -1.0,
    'C': 1j,    'c': 1j,
    'G': -1j,   'g': -1j,
    'N': 0.0,   'n': 0.0,
}

# Watson-Crick complement. RNA, not DNA: this maps A -> U. It used to map A -> T, which is
# the DNA complement, and fed an RNA sequence the prefilter accepted A-U while _validate_rcm
# then rejected it in one of the two orientations -- rcm_crossing('A','U',1) returned 1 and
# rcm_crossing('U','A',1) returned 0. Measured effect of the alphabet, done caller-side by
# rewriting U to T: +0.0340 AUC [0.0212, 0.0465], p=0.000, on the looser negative set. T is
# kept so a malformed input containing T complements rather than going unmatched.
_COMPLEMENT = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C', 'T': 'A',
               'a': 'u', 'u': 'a', 'c': 'g', 'g': 'c', 't': 'a',
               'N': 'N', 'n': 'n'}


def _seq_to_complex(seq: str) -> np.ndarray:
    """Convert a sequence to a complex vector. O(L)."""
    return np.array([_BASE_MAP.get(c, 0.0) for c in seq], dtype=np.complex64)


def _kmer_scores(seq: str, k: int) -> np.ndarray:
    """Compute cumulative scores of all length-k kmers in a sequence. O(L).

    Uses a cumulative sum so that score[i] = sum(seq[i:i+k]) is obtained
    quickly via a cumsum difference. Returns a (L-k+1,) complex vector in
    which each element is the cumulative score of one kmer.
    """
    # Pad the sequence with a leading 'N' (value 0) so that cumsum[0] = 0
    aug = np.concatenate([[0.0 + 0j], _seq_to_complex(seq)])
    cum = aug.cumsum()
    # score[i] = cum[i+k] - cum[i], i.e. the sum over a length-k window
    return cum[k:] - cum[:-k]


def _validate_rcm(seq1: str, seq2: str, i: int, j: int, k: int) -> int:
    """Base-by-base check of whether the kmers at positions i and j are reverse complements. Returns the mismatch count."""
    mismatches = 0
    for t in range(k):
        if seq1[i + t] != _COMPLEMENT.get(seq2[j + k - 1 - t], 'N'):
            mismatches += 1
            if mismatches > 0:  # return as soon as a mismatch is found (original logic)
                return mismatches
    return mismatches


def rcm_crossing(
    seq_upstream: str,
    seq_downstream: str,
    k: int = 7,
    max_mismatch: int = 0,
) -> Tuple[int, np.ndarray]:
    """Detect RCM kmer pairs between two sequences (crossing RCM).

    Original CircCNNs two-step algorithm:
      1. cumulative-sum outer-product fast prefilter (|real|+|imag| <= threshold)
      2. base-by-base validation (exact reverse-complement check)

    Args:
        seq_upstream: intron sequence upstream of the BSJ
        seq_downstream: intron sequence downstream of the BSJ
        k: kmer length
        max_mismatch: allowed mismatches (0 = fully complementary)

    Returns:
        (n_rcm_pairs, distribution_5x5)
    """
    L1, L2 = len(seq_upstream), len(seq_downstream)
    if L1 < k or L2 < k:
        return 0, np.zeros((5, 5), dtype=np.float64)

    # Step 1: cumulative-sum outer-product fast prefilter
    s1 = _kmer_scores(seq_upstream, k)
    s2 = _kmer_scores(seq_downstream, k)
    combo = s1.reshape(-1, 1) + s2.reshape(1, -1)
    score_mat = np.abs(combo.real) + np.abs(combo.imag)
    candidates = np.where(score_mat <= max_mismatch)

    # Step 2: base-by-base validation
    valid_rows, valid_cols = [], []
    for r, c in zip(candidates[0], candidates[1]):
        if _validate_rcm(seq_upstream, seq_downstream, int(r), int(c), k) <= max_mismatch:
            valid_rows.append(r)
            valid_cols.append(c)

    n_pairs = len(valid_rows)

    # 5x5 distribution matrix
    dist = np.zeros((5, 5), dtype=np.float64)
    if n_pairs > 0:
        r_bins = np.clip((np.array(valid_rows) / max(1, len(s1) - 1) * 5).astype(int), 0, 4)
        c_bins = np.clip((np.array(valid_cols) / max(1, len(s2) - 1) * 5).astype(int), 0, 4)
        for rb, cb in zip(r_bins, c_bins):
            dist[rb, cb] += 1.0

    return n_pairs, dist


def rcm_within(
    seq: str,
    k: int = 7,
    max_mismatch: int = 0,
) -> Tuple[int, np.ndarray]:
    """Detect RCM kmer pairs within a single sequence (within RCM).

    Original CircCNNs two-step algorithm + upper-triangle dedup.

    Args:
        seq: intron sequence
        k: kmer length
        max_mismatch: allowed mismatches

    Returns:
        (n_rcm_pairs, distribution_5x5)
    """
    L = len(seq)
    if L < 2 * k:
        return 0, np.zeros((5, 5), dtype=np.float64)

    s = _kmer_scores(seq, k)

    # Step 1: cumulative-sum outer-product prefilter (upper triangle only, i < j)
    combo = s.reshape(-1, 1) + s.reshape(1, -1)
    score_mat = np.abs(combo.real) + np.abs(combo.imag)
    candidates = np.where(np.triu(score_mat <= max_mismatch, k=1))

    # Step 2: base-by-base validation
    valid_rows, valid_cols = [], []
    for r, c in zip(candidates[0], candidates[1]):
        if _validate_rcm(seq, seq, int(r), int(c), k) <= max_mismatch:
            valid_rows.append(r)
            valid_cols.append(c)

    n_pairs = len(valid_rows)

    dist = np.zeros((5, 5), dtype=np.float64)
    if n_pairs > 0:
        r_bins = np.clip((np.array(valid_rows) / max(1, len(s) - 1) * 5).astype(int), 0, 4)
        c_bins = np.clip((np.array(valid_cols) / max(1, len(s) - 1) * 5).astype(int), 0, 4)
        for rb, cb in zip(r_bins, c_bins):
            dist[rb, cb] += 1.0

    return n_pairs, dist


def compute_rcm_score(
    seq_upstream: str,
    seq_downstream: str,
    kmer_lengths: Optional[List[int]] = None,
    max_mismatch: int = 0,
) -> dict:
    """Compute the composite RCM score: crossing + within(upstream) + within(downstream).

    Corresponds to the input features of RCM_triCNN in the paper.

    Args:
        seq_upstream: intron sequence upstream of the BSJ
        seq_downstream: intron sequence downstream of the BSJ
        kmer_lengths: list of k values, default [5, 7, 9, 11, 13]
        max_mismatch: allowed mismatches

    Returns:
        dict with keys:
            'crossing_total': int, total number of crossing RCM kmer pairs (sum over all k)
            'within_up_total': int, total upstream within-RCM count
            'within_down_total': int, total downstream within-RCM count
            'crossing_dists': list of (k, 5x5 matrix), crossing distribution per k
            'within_up_dists': list of (k, 5x5 matrix)
            'within_down_dists': list of (k, 5x5 matrix)
            'confidence': float, composite confidence [0, 1]
    """
    if kmer_lengths is None:
        kmer_lengths = [5, 7, 9, 11, 13]

    crossing_total = 0
    within_up_total = 0
    within_down_total = 0
    crossing_dists = []
    within_up_dists = []
    within_down_dists = []

    for k in kmer_lengths:
        n_cross, d_cross = rcm_crossing(seq_upstream, seq_downstream, k, max_mismatch)
        n_up, d_up = rcm_within(seq_upstream, k, max_mismatch)
        n_down, d_down = rcm_within(seq_downstream, k, max_mismatch)

        crossing_total += n_cross
        within_up_total += n_up
        within_down_total += n_down
        crossing_dists.append((k, d_cross))
        within_up_dists.append((k, d_up))
        within_down_dists.append((k, d_down))

    # Composite confidence: more crossing is better, more within is worse (competing)
    # confidence = crossing / (crossing + within_up + within_down + 1)
    total = crossing_total + within_up_total + within_down_total
    confidence = crossing_total / max(1, total)

    return {
        'crossing_total': crossing_total,
        'within_up_total': within_up_total,
        'within_down_total': within_down_total,
        'crossing_dists': crossing_dists,
        'within_up_dists': within_up_dists,
        'within_down_dists': within_down_dists,
        'confidence': confidence,
    }


def rcm_pair_weight(
    seq_upstream: str,
    seq_downstream: str,
    base_weight: float = 1.0,
    kmer_lengths: Optional[List[int]] = None,
) -> float:
    """Compute the RCM-weighted weight for a single base pair.

    Used to inject into the pipeline as a replacement for, or a supplement to,
    the ViennaRNA BPP confidence.

    Args:
        seq_upstream: intron upstream of the BSJ
        seq_downstream: intron downstream of the BSJ
        base_weight: base weight
        kmer_lengths: list of k values

    Returns:
        weighted weight = base_weight x (1 + crossing_confidence)
    """
    result = compute_rcm_score(seq_upstream, seq_downstream, kmer_lengths)
    return base_weight * (1.0 + result['confidence'])


def rcm_density_score(
    seq_upstream: str,
    seq_downstream: str,
    kmer_lengths: Optional[List[int]] = None,
    max_mismatch: int = 0,
) -> dict:
    """Length-normalised form of the composite RCM statistics.

    compute_rcm_score()'s confidence is a ratio of kmer-match counts. Both the numerator
    and the denominator grow with the length of the flanking window, and the pipeline picks
    that window as min(200, len(sequence) // 4) -- 5 to 30 nt on the structures this was
    measured on -- so two pairs scored at different window lengths are not compared on the
    same scale. This function returns the same counts together with the number of kmer
    pairs that were actually scanned, so the match rate can be read instead of the count.

    The opportunity counts mirror the guards in rcm_crossing() and rcm_within() exactly:
    crossing scans (L1-k+1)*(L2-k+1) pairs when both flanks are at least k long, and
    within scans the upper triangle when one flank is at least 2k long and contributes
    nothing at all below that. Counting opportunities the functions never look at would
    make the density a different quantity than "matches per comparison made".

    'confidence' is recomputed here with the same expression compute_rcm_score() uses, so a
    caller can verify the two agree; the existing function is left untouched and no
    existing caller changes behaviour.

    Returns a dict with 'crossing_total', 'within_up_total', 'within_down_total',
    'crossing_comparisons', 'within_comparisons', 'crossing_density' (crossing matches per
    crossing comparison, 0.0 when there are none to make), 'flank_up', 'flank_down',
    'per_k' (per kmer length: the four counts and the two opportunity counts) and
    'confidence' (identical to compute_rcm_score(...)['confidence']).
    """
    if kmer_lengths is None:
        kmer_lengths = [5, 7, 9, 11, 13]

    l1, l2 = len(seq_upstream), len(seq_downstream)
    crossing_total = 0
    within_up_total = 0
    within_down_total = 0
    crossing_cmp = 0
    within_cmp = 0
    per_k = {}

    for k in kmer_lengths:
        n_cross, _ = rcm_crossing(seq_upstream, seq_downstream, k, max_mismatch)
        n_up, _ = rcm_within(seq_upstream, k, max_mismatch)
        n_down, _ = rcm_within(seq_downstream, k, max_mismatch)

        c_cmp = max(0, l1 - k + 1) * max(0, l2 - k + 1)
        # rcm_within() returns (0, zeros) below 2k, so no comparison is made there
        up_cmp = max(0, l1 - k + 1) * max(0, l1 - k) // 2 if l1 >= 2 * k else 0
        dn_cmp = max(0, l2 - k + 1) * max(0, l2 - k) // 2 if l2 >= 2 * k else 0

        crossing_total += n_cross
        within_up_total += n_up
        within_down_total += n_down
        crossing_cmp += c_cmp
        within_cmp += up_cmp + dn_cmp
        per_k[k] = {
            'crossing': n_cross, 'crossing_comparisons': c_cmp,
            'within_up': n_up, 'within_down': n_down,
            'within_comparisons': up_cmp + dn_cmp,
        }

    total = crossing_total + within_up_total + within_down_total

    return {
        'crossing_total': crossing_total,
        'within_up_total': within_up_total,
        'within_down_total': within_down_total,
        'crossing_comparisons': crossing_cmp,
        'within_comparisons': within_cmp,
        'crossing_density': crossing_total / crossing_cmp if crossing_cmp else 0.0,
        'flank_up': l1,
        'flank_down': l2,
        'per_k': per_k,
        'confidence': crossing_total / max(1, total),
    }


# ── Self-test ──
if __name__ == "__main__":
    import time

    # Two fully complementary sequences
    seq1 = "AUCGAUCGAUCGAUCG"
    seq2 = "CGAUCGAUCGAUCGAU"  # reverse complement of seq1

    print("=== RCM self-test ===")
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2} (reverse complement of seq1)")

    t0 = time.time()
    result = compute_rcm_score(seq1, seq2)
    t1 = time.time()

    print(f"\nResults:")
    print(f"  crossing RCM pairs: {result['crossing_total']}")
    print(f"  within(up) pairs:   {result['within_up_total']}")
    print(f"  within(down) pairs: {result['within_down_total']}")
    print(f"  confidence: {result['confidence']:.3f}")
    print(f"  elapsed: {(t1-t0)*1000:.1f}ms")

    # Random sequences (should have few RCMs)
    rng = np.random.default_rng(42)
    bases = "AUCG"
    rand_seq1 = "".join(rng.choice(list(bases), 1000))
    rand_seq2 = "".join(rng.choice(list(bases), 1000))

    t0 = time.time()
    result_rand = compute_rcm_score(rand_seq1, rand_seq2)
    t1 = time.time()

    print(f"\nRandom sequences (L=1000):")
    print(f"  crossing RCM pairs: {result_rand['crossing_total']}")
    print(f"  within(up) pairs:   {result_rand['within_up_total']}")
    print(f"  within(down) pairs: {result_rand['within_down_total']}")
    print(f"  confidence: {result_rand['confidence']:.3f}")
    print(f"  elapsed: {(t1-t0)*1000:.1f}ms")
