"""
multisource_ss.py - multi-source secondary-structure consensus prediction

Inspired by: structRFM's MUSES (Multiple Sequence & Structure Prediction Fusion)
- weighted average of 5 predictors -> MEA decoding -> consensus SS

Simplified: uses ViennaRNA + structural-consistency weighting, no extra dependencies

Usage:
    from .multisource_ss import multisource_consensus_ss

    # input: sequence + optional bpp matrix
    ss_consensus, bpp_fused = multisource_consensus_ss(sequence, bpp_vienna)
"""

import numpy as np
from typing import Optional, Tuple, List


def _vienna_fold_consensus(sequence: str) -> Tuple[str, np.ndarray]:
    """Single source: ViennaRNA fold -> (ss_string, bpp_matrix)"""
    try:
        import RNA
        fc = RNA.fold_compound(sequence)
        ss, mfe = fc.mfe()
        L = len(sequence)
        bpp_0 = np.zeros((L, L), dtype=np.float32)

        # ViennaRNA 2.x returns the base-pair probability matrix as an (L+1, L+1) array, 1-based,
        # with row/column 0 unused. The loop here used to index that array as if it were a
        # per-position partner vector (`bp[i] > 0`, `int(bp[i]) > i`), which raises
        #
        #     TypeError: '>' not supported between instances of 'tuple' and 'int'
        #
        # and the `except` below turned that into (None, None) on every single call. So this
        # source never contributed anything. With the Nussinov source also gated behind L <= 500
        # at :161, multisource_consensus_ss returned an all-dots structure for every sequence
        # longer than 500 nt -- and run_2013nt.py:67, seeing no pairs, silently fell through to
        # its own ViennaRNA fold. Measured on ViennaRNA 2.7.2.
        # bpp() needs the partition function to have been computed; called on a fresh
        # fold_compound it returns an empty tuple rather than raising, which is why the shape
        # check below is a check and not an assumption.
        fc.pf()
        bp = np.asarray(fc.bpp(), dtype=np.float64)
        if bp.ndim != 2 or bp.shape[0] < L + 1 or bp.shape[1] < L + 1:
            return None, None
        bp = bp[1:L + 1, 1:L + 1]

        # Give each base its most likely partner above the diagonal, which is what the original
        # loop was reaching for. The result stays a binary contact map, as before.
        for i in range(L):
            row = bp[i].copy()
            row[:i + 1] = 0.0
            j = int(np.argmax(row))
            if row[j] > 0.0:
                bpp_0[i, j] = 1.0
                bpp_0[j, i] = 1.0
        return ss, bpp_0
    except ImportError:
        # Deliberately narrow. This used to be `except (ImportError, Exception)`, which is what
        # kept the TypeError above invisible for as long as it existed: a broken source looked
        # exactly like an absent one. Anything other than a missing ViennaRNA now propagates,
        # and run_2013nt.py:65 already catches it and reports "MUSES unavailable".
        return None, None


def _ss_to_contact_matrix(ss: str) -> np.ndarray:
    """SS string -> contact matrix (0/1)"""
    L = len(ss)
    contact = np.zeros((L, L), dtype=np.float32)
    stack = []
    bracket_map = {'(': ')', '[': ']', '{': '}', '<': '>'}
    close_to_open = {v: k for k, v in bracket_map.items()}

    for i, ch in enumerate(ss):
        if ch in bracket_map:
            stack.append((ch, i))
        elif ch in close_to_open:
            if stack and stack[-1][0] == close_to_open[ch]:
                _, j = stack.pop()
                contact[j, i] = 1.0
                contact[i, j] = 1.0
    return contact


def _consensus_from_bpp_list(bpp_list: List[np.ndarray],
                               weights: Optional[List[float]] = None) -> np.ndarray:
    """Weighted average of multi-source bpp -> consensus bpp"""
    if weights is None:
        weights = [1.0] * len(bpp_list)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    result = np.zeros_like(bpp_list[0], dtype=np.float32)
    for bpp, w in zip(bpp_list, weights):
        result += w * bpp
    return result


def _confidence_weighted_consensus(ss_list: List[str],
                                    bpp_list: List[np.ndarray]) -> Tuple[str, np.ndarray]:
    """Confidence-weighted consensus:
    - each predictor's bpp matrix uses base-pair probability as its confidence
    - higher-confidence predictors get larger weights
    """
    L = len(ss_list[0])
    # Per-predictor confidence: mean of the pairing probabilities above 5 percent.
    #
    # This used to read np.diag(bpp), the diagonal of the base-pair matrix. That diagonal is
    # identically zero -- a base does not pair with itself -- so the `np.any(diag > 0.05)` branch
    # could never fire, every predictor came out at the 0.01 floor, and the "confidence-weighted"
    # consensus below was equal weighting. Use the off-diagonal (upper triangle) instead.
    confidences = []
    _iu = np.triu_indices(L, k=1)
    for bpp in bpp_list:
        if bpp is not None:
            vals = np.asarray(bpp[:L, :L])[_iu]
            conf = float(np.mean(vals[vals > 0.05])) if np.any(vals > 0.05) else 0.0
            confidences.append(max(conf, 0.01))
        else:
            confidences.append(0.01)

    # normalize the weights
    confidences = np.array(confidences, dtype=np.float32)
    weights = confidences / confidences.sum()

    # weighted bpp
    fused_bpp = _consensus_from_bpp_list(bpp_list, weights.tolist())

    # consensus SS: majority vote (only over Vienna-format symbols)
    consensus = list(ss_list[0])
    for pos in range(L):
        votes = {}
        for ss in ss_list:
            ch = ss[pos] if pos < len(ss) else '.'
            votes[ch] = votes.get(ch, 0) + 1
        # tie-break: paired > unpaired
        paired = {k: v for k, v in votes.items() if k != '.'}
        if paired:
            best_paired = max(paired, key=paired.get)
            if paired[best_paired] > votes.get('.', 0):
                consensus[pos] = best_paired
            else:
                consensus[pos] = '.'
        else:
            consensus[pos] = '.'

    return ''.join(consensus), fused_bpp


def multisource_consensus_ss(
    sequence: str,
    bpp_vienna: Optional[np.ndarray] = None,
    use_nussinov_fallback: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    Multi-source secondary-structure consensus prediction

    Simplified MUSES: ViennaRNA + Nussinov (optional) + confidence weighting

    Args:
        sequence: RNA sequence (ACGU)
        bpp_vienna: optional precomputed ViennaRNA bpp matrix
        use_nussinov_fallback: whether to use Nussinov as the second predictor

    Returns:
        ss_consensus: consensus secondary-structure string
        bpp_fused: fused bpp matrix (L, L)
    """
    L = len(sequence)
    ss_list = []
    bpp_list = []

    # Source 1: ViennaRNA fold
    if bpp_vienna is not None:
        # reconstruct SS from bpp
        ss_v = '.' * L
        stack = []
        for i in range(L):
            for j in range(i + 3, L):  # min loop size = 3
                if bpp_vienna[i, j] > 0.5:
                    ss_v = ss_v[:i] + '(' + ss_v[i+1:j] + ')' + ss_v[j+1:]
                    break
        ss_list.append(ss_v)
        bpp_list.append(bpp_vienna)
    else:
        ss_v, bpp_v = _vienna_fold_consensus(sequence)
        if ss_v is not None:
            ss_list.append(ss_v)
            bpp_list.append(bpp_v)

    # Source 2: Nussinov (simple maximum matching)
    if use_nussinov_fallback and L <= 500:
        ss_n, bpp_n = _nussinov_fold(sequence)
        ss_list.append(ss_n)
        bpp_list.append(bpp_n)

    if len(ss_list) == 0:
        return '.' * L, np.zeros((L, L), dtype=np.float32)

    if len(ss_list) == 1:
        return ss_list[0], bpp_list[0]

    return _confidence_weighted_consensus(ss_list, bpp_list)


def _nussinov_fold(sequence: str) -> Tuple[str, np.ndarray]:
    """Nussinov maximum-matching algorithm (simple baseline)"""
    L = len(sequence)
    can_pair = np.zeros((L, L), dtype=bool)

    # RNA base-pairing rules
    pairs = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}
    for i in range(L):
        for j in range(i + 4, L):  # min loop = 4
            if (sequence[i], sequence[j]) in pairs:
                can_pair[i, j] = True

    # DP
    dp = np.zeros((L, L), dtype=int)
    for length in range(5, L + 1):
        for i in range(L - length + 1):
            j = i + length - 1
            dp[i, j] = dp[i, j-1]
            for k in range(i, j - 3):
                if can_pair[k, j]:
                    score = 1 + dp[i, k-1] if k > i else 1
                    if k > i:
                        score += dp[i, k-1]
                    if k + 1 < j:
                        score += dp[k+1, j-1]
                    dp[i, j] = max(dp[i, j], score)

    # traceback
    ss = ['.' ] * L
    traceback_stack = [(0, L - 1)]
    while traceback_stack:
        i, j = traceback_stack.pop()
        if i >= j:
            continue
        if dp[i, j] == dp[i, j-1]:
            traceback_stack.append((i, j-1))
            continue
        found = False
        for k in range(i, j - 3):
            if can_pair[k, j]:
                score = 1
                if k > i:
                    score += dp[i, k-1]
                if k + 1 < j:
                    score += dp[k+1, j-1]
                if score == dp[i, j]:
                    ss[k] = '('
                    ss[j] = ')'
                    if k > i:
                        traceback_stack.append((i, k-1))
                    if k + 1 < j:
                        traceback_stack.append((k+1, j-1))
                    found = True
                    break
        if not found:
            traceback_stack.append((i, j-1))

    # build the bpp matrix
    bpp = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        if ss[i] == '(':
            for j in range(i+1, L):
                if ss[j] == ')':
                    bpp[i, j] = 1.0
                    bpp[j, i] = 1.0
                    break

    return ''.join(ss), bpp
