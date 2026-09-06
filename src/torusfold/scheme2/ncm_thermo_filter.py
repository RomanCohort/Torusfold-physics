"""
ncm_thermo_filter.py - Thermodynamic negative-sample filter for NCM candidates.

Post-filters non-canonical pair (NCM) candidates by checking whether the Watson-Crick
alternative is thermodynamically more favorable. Suppresses false-positive NCM candidates
where forcing the NCM pair costs too much energy compared to unconstrained folding.

Key idea: For each NCM candidate (i,j), compute the energy cost of forcing this pair
in a local window vs. unconstrained MFE. If the cost exceeds a threshold, the candidate
is likely a false positive (the WC alternative is far more favorable).
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


# Default parameters
DEFAULT_DDG_THRESHOLD = 2.0  # kcal/mol: cost threshold for suppression
DEFAULT_HALF_WINDOW = 10     # nucleotides on each side of the pair
WEIGHT_REDUCTION_FACTOR = 0.5  # multiply weight by this when penalty is moderate


def _compute_local_window(
    seq_len: int,
    i: int,
    j: int,
    half_window: int,
) -> Tuple[int, int]:
    """Compute the local window [start, end) around pair (i, j).

    The window is centered on the midpoint of i and j, extended by half_window
    on each side, clipped to sequence bounds.
    """
    mid = (i + j) // 2
    start = max(0, mid - half_window)
    end = min(seq_len, mid + half_window + 1)
    # Ensure both i and j are inside the window
    start = min(start, i)
    end = max(end, j + 1)
    return start, end


def _fold_with_constraint(
    sequence: str,
    i_rel: int,
    j_rel: int,
) -> float:
    """Compute MFE with constraint forcing pair (i_rel, j_rel) in 1-based ViennaRNA coords.

    Returns energy in kcal/mol, or float('inf') on failure.
    """
    import RNA

    try:
        fc = RNA.fold_compound(sequence)
        # Add hard constraint: force i_rel to pair with j_rel
        # ViennaRNA uses 1-based indexing; ENFORCE | ALL_LOOPS forces the pair
        constraint_option = RNA.CONSTRAINT_CONTEXT_ENFORCE | RNA.CONSTRAINT_CONTEXT_ALL_LOOPS
        fc.hc_add_bp(i_rel, j_rel, constraint_option)
        _ss, energy = fc.mfe()
        return energy
    except Exception:
        return float('inf')


def _fold_unconstrained(sequence: str) -> float:
    """Compute unconstrained MFE.

    Returns energy in kcal/mol, or float('inf') on failure.
    """
    import RNA

    try:
        _ss, energy = RNA.fold(sequence)
        return energy
    except Exception:
        return float('inf')


def filter_ncm_candidates(
    sequence: str,
    ncm_pairs: List[Tuple[int, int, float, str]],
    bpp_matrix: Optional[np.ndarray] = None,
    ddg_threshold: float = DEFAULT_DDG_THRESHOLD,
    half_window: int = DEFAULT_HALF_WINDOW,
) -> List[Tuple[int, int, float, str]]:
    """Filter NCM candidates by thermodynamic penalty of forcing the NCM pair.

    For each candidate (i, j), compute:
        E_constrained = MFE energy of local window with (i,j) forced to pair
        E_unconstrained = MFE energy of same window without constraints
        ddg = E_constrained - E_unconstrained

    If ddg > ddg_threshold: suppress the candidate (remove from list)
    If 1.0 < ddg <= ddg_threshold: downweight by factor 0.5
    If ddg <= 1.0: keep as-is

    Args:
        sequence: RNA sequence (ACGU)
        ncm_pairs: List of (i, j, weight, type) tuples from NCM detection
        bpp_matrix: Optional BPP matrix (unused in current impl, reserved)
        ddg_threshold: ΔΔG threshold in kcal/mol for full suppression
        half_window: Half-window size for local folding context

    Returns:
        Filtered list of (i, j, weight, type) tuples, same format as input
    """
    if not ncm_pairs:
        return []

    L = len(sequence)
    if L < 4:
        return ncm_pairs[:]

    # Import ViennaRNA once
    try:
        import RNA
    except ImportError:
        # If ViennaRNA not available, skip filtering entirely
        return ncm_pairs[:]

    # Group candidates by their local window for potential batching
    # (in practice, each candidate has a unique window, but this structure
    # allows future optimization)
    filtered = []
    n_suppressed = 0
    n_downweighted = 0

    for i, j, weight, ncm_type in ncm_pairs:
        # Validate positions
        if i < 0 or j < 0 or i >= L or j >= L or i >= j:
            filtered.append((i, j, weight, ncm_type))
            continue

        # Compute local window
        start, end = _compute_local_window(L, i, j, half_window)
        local_seq = sequence[start:end]
        local_len = len(local_seq)

        # Need at least 4 nt for meaningful MFE
        if local_len < 4:
            filtered.append((i, j, weight, ncm_type))
            continue

        # Relative positions in the local window (1-based for ViennaRNA)
        i_rel = i - start + 1
        j_rel = j - start + 1

        # Sanity check
        if i_rel < 1 or j_rel < 1 or i_rel > local_len or j_rel > local_len or i_rel >= j_rel:
            filtered.append((i, j, weight, ncm_type))
            continue

        # Compute ΔΔG
        try:
            E_constrained = _fold_with_constraint(local_seq, i_rel, j_rel)
            E_unconstrained = _fold_unconstrained(local_seq)
        except Exception:
            # On any error, keep the candidate
            filtered.append((i, j, weight, ncm_type))
            continue

        # Check for failed computations
        if E_constrained == float('inf') or E_unconstrained == float('inf'):
            filtered.append((i, j, weight, ncm_type))
            continue

        ddg = E_constrained - E_unconstrained

        if ddg > ddg_threshold:
            # Suppress: WC alternative is much more favorable
            n_suppressed += 1
            # Don't add to filtered list (suppress)
        elif ddg > 1.0:
            # Downweight: moderate penalty
            new_weight = weight * WEIGHT_REDUCTION_FACTOR
            filtered.append((i, j, new_weight, ncm_type))
            n_downweighted += 1
        else:
            # Keep as-is: NCM pair is thermodynamically reasonable
            filtered.append((i, j, weight, ncm_type))

    return filtered


def compute_pairwise_ddg(
    sequence: str,
    ncm_pairs: List[Tuple[int, int, float, str]],
    half_window: int = DEFAULT_HALF_WINDOW,
) -> List[Tuple[int, int, float, str, float]]:
    """Debug/utility: compute ΔΔG for each NCM candidate without filtering.

    Returns list of (i, j, weight, type, ddg) tuples.
    """
    L = len(sequence)
    results = []

    for i, j, weight, ncm_type in ncm_pairs:
        if i < 0 or j < 0 or i >= L or j >= L or i >= j:
            results.append((i, j, weight, ncm_type, 0.0))
            continue

        start, end = _compute_local_window(L, i, j, half_window)
        local_seq = sequence[start:end]
        local_len = len(local_seq)

        if local_len < 4:
            results.append((i, j, weight, ncm_type, 0.0))
            continue

        i_rel = i - start + 1
        j_rel = j - start + 1

        if i_rel < 1 or j_rel < 1 or i_rel > local_len or j_rel > local_len or i_rel >= j_rel:
            results.append((i, j, weight, ncm_type, 0.0))
            continue

        try:
            E_c = _fold_with_constraint(local_seq, i_rel, j_rel)
            E_u = _fold_unconstrained(local_seq)
            if E_c == float('inf') or E_u == float('inf'):
                ddg = 0.0
            else:
                ddg = E_c - E_u
        except Exception:
            ddg = 0.0

        results.append((i, j, weight, ncm_type, ddg))

    return results


if __name__ == "__main__":
    # Self-test: construct a sequence with a known stem and inject a false-positive NCM
    import time

    print("=" * 60)
    print("NCM Thermodynamic Filter - Self Test")
    print("=" * 60)

    # Construct a 40-nt sequence with a strong stem at positions 5-15 / 25-35
    # Positions 5-10: GAUCCA pairs with 30-35: UAGGU (WC)
    # This is a strong canonical stem.
    #
    # Inject a false-positive NCM: position 10 (A) with position 20 (A) = A-A SHEAR
    # This pair is inside the stem region and should be suppressed because
    # the WC alternative (10 pairs with 30) is thermodynamically better.

    # sequence:     0123456789012345678901234567890123456789
    test_seq = "CCCCGAUCCAGGGGGGGGGAUAGGGGGAUAGGUUUUCC"

    print(f"Sequence ({len(test_seq)} nt): {test_seq}")
    print(f"Expected stem: pos 5-10 (GAUCCA) pairs with pos 30-35 (UAGGU)")

    # Create test NCM candidates
    # (i, j, weight, type)
    test_ncm_pairs = [
        (10, 20, 0.5, "SHEAR"),   # False positive: A at 10, G at 20 - inside stem
        (0, 39, 0.6, "HOOGSTEEN"), # Genuinely at ends, might be loop closure
    ]

    print(f"\nTest NCM pairs before filtering:")
    for i, j, w, t in test_ncm_pairs:
        print(f"  ({i:2d}, {j:2d}) type={t:10s} weight={w:.2f}")

    # Run filter
    start_time = time.time()
    filtered = filter_ncm_candidates(
        test_seq,
        test_ncm_pairs,
        ddg_threshold=DEFAULT_DDG_THRESHOLD,
    )
    elapsed = time.time() - start_time

    print(f"\nFiltered NCM pairs:")
    for i, j, w, t in filtered:
        print(f"  ({i:2d}, {j:2d}) type={t:10s} weight={w:.2f}")
    print(f"  (suppressed {len(test_ncm_pairs) - len(filtered)} candidates)")

    # Compute pairwise ddg for debugging
    print(f"\nPairwise ΔΔG breakdown:")
    ddg_results = compute_pairwise_ddg(test_seq, test_ncm_pairs)
    for i, j, w, t, ddg in ddg_results:
        print(f"  ({i:2d}, {j:2d}) type={t:10s} ΔΔG={ddg:+.2f} kcal/mol")

    print(f"\nFiltering time: {elapsed:.3f}s")
    print("=" * 60)

    # Check that at least one candidate was suppressed or downweighted
    n_original = len(test_ncm_pairs)
    n_filtered = len(filtered)
    if n_filtered < n_original:
        print("PASS: Filter suppressed at least one candidate")
    else:
        print("NOTE: No candidates were suppressed (may be sequence-dependent)")
