"""
ncm_detector.py - Non-Canonical Motif Detector & Global 2D Pair Assembly.

Level 0 module: ViennaRNA partition function → WC pairing probability matrix
+ heuristic non-canonical base pair detection (Leontis-Westhof classification).

Output format: List[Tuple[int, int, str, float]]
  (i, j, edge_type, confidence)
  edge_type: "WC", "HOOGSTEEN", "SUGAR", "STACK", "SHEAR"
  confidence: 0.0 - 1.0
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# ---------- Leontis-Westhof edge classification ----------
# RNA bases have three edges: Watson-Crick (W), Hoogsteen (H), Sugar (S)
# WC pairs: W-W
# Non-canonical pairs defined by edge combinations

# Canonical WC pairs
_WC_PAIRS: Set[Tuple[str, str]] = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
}
_GU_WOBBLE: Set[Tuple[str, str]] = {("G", "U"), ("U", "G")}

# Leontis-Westhof non-canonical classification
# Format: (base1, base2) → [(edge1, edge2, type_name), ...]
# Edges: W=Watson-Crick, H=Hoogsteen, S=Sugar
_NCM_TYPES: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {
    # A pairs
    ("A", "G"): [("H", "H", "HOOGSTEEN"), ("S", "S", "SUGAR")],
    ("A", "A"): [("H", "S", "SHEAR"), ("S", "H", "SHEAR")],
    ("A", "C"): [("H", "W", "HOOGSTEEN")],
    # G pairs
    ("G", "A"): [("H", "H", "HOOGSTEEN"), ("S", "S", "SUGAR")],
    ("G", "G"): [("H", "S", "SHEAR"), ("S", "H", "SHEAR"), ("H", "H", "HOOGSTEEN")],
    ("G", "U"): [("W", "W", "WC")],  # wobble
    # U pairs
    ("U", "U"): [("H", "H", "HOOGSTEEN"), ("S", "S", "SUGAR")],
    ("U", "C"): [("H", "H", "HOOGSTEEN")],
    # C pairs
    ("C", "A"): [("W", "H", "HOOGSTEEN")],
    ("C", "C"): [("H", "H", "HOOGSTEEN"), ("S", "S", "SUGAR")],
    ("C", "U"): [("H", "H", "HOOGSTEEN")],
}

# Tandem motif patterns: consecutive WC pairs flanking a non-canonical pair
# (i-1,j+1) WC, (i,j) non-canonical, (i+1,j-1) WC
_TANDEM_MIN_FLANK =1  # minimum WC pairs on each side

# Miniloop closure: non-canonical pair closing a loop of size 1-5
_MINILOOP_MAX_SIZE = 5
_MINILOOP_MIN_SIZE =1

# Stacking: consecutive pairs (i,j), (i+1,j-1) both non-canonical
_STACK_MIN_LEN =2

# Probability thresholds
WC_HARD_THRESHOLD =0.9
WC_SOFT_THRESHOLD =0.5
NCM_BASE_CONFIDENCE =0.5  # base confidence for non-canonical pairs


def _get_pf_matrix(sequence: str) -> Tuple[np.ndarray, float]:
    """Compute base pair probability matrix via ViennaRNA partition function.

    Returns:
        pp_matrix: np.ndarray shape (L, L), pp_matrix[i,j] = P(i pairs with j)
        free_energy: float, ensemble free energy in kcal/mol
    """
    import RNA

    md = RNA.md()
    md.circ =1  # circular RNA
    fc = RNA.fold_compound(sequence, md)
    ss, mfe = fc.pf()  # partition function folding
    free_energy = fc.mean_bp_distance()  # not used but available

    # Extract pair probabilities from fold_compound
    L = len(sequence)
    pp_matrix = np.zeros((L, L), dtype=np.float64)

    # ViennaRNA provides zscore, but we need the probability matrix
    # Use export_bppm to get base pair probability matrix
    # The bp_prob array is 1-indexed and flattened
    bp_prob = np.array(fc.bpp(), dtype=np.float64)

    # bp_prob is a 1D array of size L*(L+1)/2, stored as:
    # bpp[i,j] for j > i, at index i*L + j - i*(i+1)/2
    # Actually ViennaRNA stores it as a matrix accessed via fc.bpp[i][j]
    for i in range(L):
        for j in range(i +1, L):
            # ViennaRNA uses 1-based indexing
            p = fc.bpp()[i +1][j +1]
            pp_matrix[i, j] = p
            pp_matrix[j, i] = p

    return pp_matrix, mfe


def _is_wc(b1: str, b2: str) -> bool:
    """Check if (b1, b2) is a Watson-Crick pair (including G-U wobble)."""
    return (b1, b2) in _WC_PAIRS or (b1, b2) in _GU_WOBBLE


def _get_ncm_type(b1: str, b2: str) -> Optional[str]:
    """Get the Leontis-Westhof NCM type for a non-canonical pair.

    Returns the most likely edge combination type, or None if not a known NCM.
    """
    if (b1, b2) in _NCM_TYPES:
        # Return the first (most common) type
        return _NCM_TYPES[(b1, b2)][0][2]
    return None


def _build_seq_masks(sequence: str):
    """Build the sequence base-pair mask matrices (vectorized foundation).

    Returns:
        (is_nonwc, ncm_code)
        is_nonwc: (L,L) bool, True = base pair that is neither WC nor wobble (valid on the upper triangle i<j)
        ncm_code: (L,L) int8, NCM type code (0=none, 1=HOOGSTEEN, 2=SUGAR, 3=SHEAR, 4=STACK)
    """
    seq_arr = np.frombuffer(sequence.encode(), dtype=np.uint8)
    b1 = seq_arr[:, None]
    b2 = seq_arr[None, :]
    L = len(sequence)

    is_nonwc = np.ones((L, L), dtype=bool)
    for a, b in list(_WC_PAIRS) + list(_GU_WOBBLE):
        is_nonwc &= ~((b1 == ord(a)) & (b2 == ord(b)))
    # upper triangle j > i (self and below-diagonal entries are meaningless); each
    # detector enforces its own min-loop constraint
    # (tandem uses j>i+3? no - the tandem flanks sit on both sides, so the NCM gap
    #  itself can be as small as 2; the original Python range(i+4, L) required gap>=4,
    #  here we stay consistent and allow gap>=2 to permit miniloop-closure-type tandems;
    #  shear/miniloop apply their own filtering afterwards)
    jj, ii = np.meshgrid(np.arange(L), np.arange(L))
    is_nonwc &= jj > ii
    np.fill_diagonal(is_nonwc, False)

    # NCM type code: take the first Leontis-Westhof type of the base pair
    type_map = {"HOOGSTEEN": 1, "SUGAR": 2, "SHEAR": 3}
    ncm_code = np.zeros((L, L), dtype=np.int8)
    for (a, b), combos in _NCM_TYPES.items():
        etype = combos[0][2]
        code = type_map.get(etype, 0)
        if code and etype != "WC":
            mask = (b1 == ord(a)) & (b2 == ord(b))
            cur = ncm_code[mask]
            # do not overwrite a higher-priority type already present (larger value wins)
            ncm_code[mask] = np.maximum(cur, code)

    return is_nonwc, ncm_code


def _detect_tandem_ncms(
    sequence: str,
    pp_matrix: np.ndarray,
    wc_pairs: Set[Tuple[int, int]],
    L: int,
) -> List[Tuple[int, int, str, float]]:
    """Detect tandem non-canonical motifs: WC-flanked non-canonical pairs.

    Pattern: (i-k, j+k) WC, (i, j) NCM, (i+k, j-k) WC for k in 1..N

    Vectorized implementation: WC pair set -> mask matrix; flank counts are
    accumulated by shifting. O(L^2) numpy vs. the original O(L^2*flank) pure
    Python; for 2013nt this drops runtime from minutes to seconds.
    """
    if L < 8 or pp_matrix is None:
        return []

    is_nonwc, ncm_code = _build_seq_masks(sequence)

    # WC mask matrix (symmetric): wc_mask[i,j] = True means (i,j) is a high-confidence WC pair
    wc_mask = np.zeros((L, L), dtype=bool)
    for (i, j) in wc_pairs:
        wc_mask[i, j] = True
        wc_mask[j, i] = True

    # flank_up[i,j] = number of consecutive k (k>=1) for which (i-k, j+k) is WC (stops at the first break)
    #   i.e. flank_up[i,j] += wc_mask[i-k, j+k]; shift: new[i,j] = old[i-1, j+1]
    # flank_dn[i,j] = number of consecutive k for which (i+k, j-k) is WC
    #   i.e. flank_dn[i,j] += wc_mask[i+k, j-k]; shift: new[i,j] = old[i+1, j-1]
    flank_up = np.zeros((L, L), dtype=np.int32)
    flank_dn = np.zeros((L, L), dtype=np.int32)

    max_flank = min(L // 4, 20)  # beyond 20 layers the confidence is already capped; no need to continue
    for _k in range(1, max_flank + 1):
        # layer_u[i,j] = wc_mask[i-_k, j+_k]
        layer_u = np.zeros((L, L), dtype=bool)
        if 2 * _k <= L - 1:
            layer_u[_k:L - _k, _k:L - _k] = wc_mask[0:L - 2 * _k, 2 * _k:L]
        # layer_d[i,j] = wc_mask[i+_k, j-_k]
        layer_d = np.zeros((L, L), dtype=bool)
        if 2 * _k <= L - 1:
            layer_d[_k:L - _k, _k:L - _k] = wc_mask[2 * _k:L, 0:L - 2 * _k]

        if _k == 1:
            flank_up += layer_u
            flank_dn += layer_d
            prev_u, prev_d = layer_u, layer_d
        else:
            cont_u = prev_u & layer_u
            cont_d = prev_d & layer_d
            flank_up += cont_u
            flank_dn += cont_d
            prev_u, prev_d = cont_u, cont_d
            if not cont_u.any() and not cont_d.any():
                break

    # candidates = non-WC with at least one flank and gap>=3 (tandem minimum separation,
    # same as the original Python range(i+4,L))
    jj, ii = np.meshgrid(np.arange(L), np.arange(L))
    gap_ok = jj > ii + 3
    cand = is_nonwc & gap_ok & ((flank_up + flank_dn) >= _TANDEM_MIN_FLANK) & (ncm_code > 0)

    idx_i, idx_j = np.nonzero(cand)
    ncms = []
    code_to_type = {1: "HOOGSTEEN", 2: "SUGAR", 3: "SHEAR"}
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        fw = int(flank_up[i, j] + flank_dn[i, j])
        conf = min(NCM_BASE_CONFIDENCE + 0.1 * fw, 0.7)
        ncms.append((i, j, code_to_type[int(ncm_code[i, j])], conf))

    return ncms


def _detect_miniloop_ncms(
    sequence: str,
    pp_matrix: np.ndarray,
    wc_pairs: Set[Tuple[int, int]],
    L: int,
) -> List[Tuple[int, int, str, float]]:
    """Detect non-canonical pairs closing miniloops (size 1-5).

    Vectorized: the small-loop window j-i in [2,6] covers only 5 diagonals,
    so they are sliced directly.
    """
    if L < 8 or pp_matrix is None:
        return []

    is_nonwc, ncm_code = _build_seq_masks(sequence)
    code_to_type = {1: "HOOGSTEEN", 2: "SUGAR", 3: "SHEAR"}

    wc_set = wc_pairs if isinstance(wc_pairs, set) else set(wc_pairs)

    ncms = []
    seen: Set[Tuple[int, int]] = set()
    for gap in range(_MINILOOP_MIN_SIZE + 1, _MINILOOP_MAX_SIZE + 2):
        diag = np.diag(is_nonwc, k=gap)          # (L-gap,) bool
        code_diag = np.diag(ncm_code, k=gap)     # (L-gap,)
        prob_diag = np.diag(pp_matrix, k=gap)    # (L-gap,)
        for off in np.nonzero(diag & (code_diag > 0) & (prob_diag > 0.01))[0]:
            i = int(off)
            j = i + gap
            if (i, j) in wc_set or (i, j) in seen:
                continue
            conf = min(NCM_BASE_CONFIDENCE + float(prob_diag[off]) * 0.5, 0.7)
            ncms.append((i, j, code_to_type[int(code_diag[off])], conf))
            seen.add((i, j))

    return ncms


def _detect_stack_ncms(
    sequence: str,
    pp_matrix: np.ndarray,
    ncm_pairs: Set[Tuple[int, int]],
    L: int,
) -> List[Tuple[int, int, str, float]]:
    """Detect stacking non-canonical pairs: consecutive NCMs."""
    ncms = []
    ncm_set: Set[Tuple[int, int]] = set()

    # Sort NCM pairs by position
    sorted_ncms = sorted(ncm_pairs)

    for idx in range(len(sorted_ncms) -1):
        i1, j1 = sorted_ncms[idx]
        i2, j2 = sorted_ncms[idx +1]

        # Check if consecutive: i2 = i1+1 and j2 = j1-1
        if i2 == i1 +1 and j2 == j1 -1:
            b1, b2 = sequence[i2], sequence[j2]
            ncm_type = _get_ncm_type(b1, b2)
            if ncm_type and (i2, j2) not in ncm_set:
                conf = min(NCM_BASE_CONFIDENCE +0.15, 0.7)  # stacking bonus
                ncms.append((i2, j2, ncm_type, conf))
                ncm_set.add((i2, j2))

    return ncms


def _detect_shear_ncms(
    sequence: str,
    pp_matrix: np.ndarray,
    L: int,
) -> List[Tuple[int, int, str, float]]:
    """Detect shear base pairs: A-A or G-G pairs with the SHEAR edge type.

    Vectorized: outer-product base mask + probability threshold in a single filter.
    """
    if L < 8 or pp_matrix is None:
        return []

    seq_arr = np.frombuffer(sequence.encode(), dtype=np.uint8)
    b1 = seq_arr[:, None]
    b2 = seq_arr[None, :]

    shear_mask = np.zeros((L, L), dtype=bool)
    for a in ("A", "G"):
        shear_mask |= (b1 == ord(a)) & (b2 == ord(a))

    jj, ii = np.meshgrid(np.arange(L), np.arange(L))
    shear_mask &= jj > ii + 3

    cand = shear_mask & (pp_matrix > 0.01)
    idx_i, idx_j = np.nonzero(cand)

    ncms = [
        (int(i), int(j), "SHEAR", min(NCM_BASE_CONFIDENCE + float(pp_matrix[i, j]) * 0.3, 0.65))
        for i, j in zip(idx_i.tolist(), idx_j.tolist())
    ]
    return ncms


def global_pair_assembly(
    sequence: str,
    msa_path: Optional[str] = None,
    *,
    wc_hard: float = WC_HARD_THRESHOLD,
    wc_soft: float = WC_SOFT_THRESHOLD,
    min_prob: float =0.01,
) -> List[Tuple[int, int, str, float]]:
    """Global 2D pair assembly: WC pairs + non-canonical motifs.

    Main entry point for Level0 pairing. Combines ViennaRNA partition function
    probabilities with heuristic NCM detection.

    Args:
        sequence: RNA sequence (ACGU), circular
        msa_path: Optional MSA file path (not used yet, reserved)
        wc_hard: WC probability threshold for hard constraints (default0.9)
        wc_soft: WC probability threshold for soft constraints (default0.5)
        min_prob: Minimum probability to consider a pair (default0.01)

    Returns:
        List of (i, j, edge_type, confidence) tuples:
          - i, j: 0-based residue indices
          - edge_type: "WC", "HOOGSTEEN", "SUGAR", "STACK", "SHEAR"
          - confidence: 0.0-1.0 (WC uses probability, NCMs use heuristic)
    """
    L = len(sequence)
    if L <4:
        return []

    # Step1: ViennaRNA partition function
    pp_matrix, mfe = _get_pf_matrix(sequence)

    # Step2: Extract WC pairs with probability thresholds
    result: List[Tuple[int, int, str, float]] = []
    wc_pairs: Set[Tuple[int, int]] = set()

    for i in range(L):
        for j in range(i +1, L):
            prob = pp_matrix[i, j]
            if prob < min_prob:
                continue
            b1, b2 = sequence[i], sequence[j]

            if _is_wc(b1, b2):
                wc_pairs.add((i, j))
                # Confidence = probability for WC pairs
                conf = prob
                result.append((i, j, "WC", conf))

    # Step3: Heuristic NCM detection
    # 3a: Tandem NCMs (WC-flanked)
    tandem_ncms = _detect_tandem_ncms(sequence, pp_matrix, wc_pairs, L)

    # 3b: Miniloop closure NCMs
    miniloop_ncms = _detect_miniloop_ncms(sequence, pp_matrix, wc_pairs, L)

    # 3c: Shear pairs (A-A, G-G)
    shear_ncms = _detect_shear_ncms(sequence, pp_matrix, L)

    # Merge all NCMs, dedup
    ncm_set: Set[Tuple[int, int]] = set()
    all_ncms: List[Tuple[int, int, str, float]] = []

    for ncm_list in [tandem_ncms, miniloop_ncms, shear_ncms]:
        for i, j, etype, conf in ncm_list:
            if (i, j) not in ncm_set:
                all_ncms.append((i, j, etype, conf))
                ncm_set.add((i, j))

    # 3d: Stacking detection on merged NCMs
    stack_ncms = _detect_stack_ncms(sequence, pp_matrix, ncm_set, L)
    for i, j, etype, conf in stack_ncms:
        if (i, j) not in ncm_set:
            all_ncms.append((i, j, etype, conf))
            ncm_set.add((i, j))

    result.extend(all_ncms)

    # Sort by position
    result.sort(key=lambda x: (x[0], x[1]))

    return result


def detect_ncms_from_bpp(
    sequence: str,
    pp_matrix: np.ndarray,
    hard_pairs: list,
    apply_thermo_filter: bool = True,
) -> list:
    """Detect non-canonical pairs from an existing BPP matrix, avoiding a redundant ViennaRNA partition-function call.

    Args:
        sequence: RNA sequence
        pp_matrix: BPP matrix already computed at Level 0 (L x L, symmetric)
        hard_pairs: list of already-determined hard-restraint pairs [(i, j, w), ...]

    Returns:
        soft-restraint list [(i, j, weight), ...] (non-WC non-canonical pairs)
    """
    L = len(sequence)
    if pp_matrix is None or L < 4:
        return []

    # extract the WC pair set from pp_matrix (P > wc_hard)
    wc_pairs: Set[Tuple[int, int]] = set()
    for i in range(L):
        for j in range(i + 1, L):
            prob = pp_matrix[i, j] if pp_matrix.shape == (L, L) else 0
            if prob < WC_HARD_THRESHOLD:
                continue
            b1, b2 = sequence[i], sequence[j]
            if _is_wc(b1, b2):
                wc_pairs.add((i, j))

    # run NCM detection (reuse the existing functions; only pp_matrix + wc_pairs are needed)
    tandem_ncms = _detect_tandem_ncms(sequence, pp_matrix, wc_pairs, L)
    miniloop_ncms = _detect_miniloop_ncms(sequence, pp_matrix, wc_pairs, L)
    shear_ncms = _detect_shear_ncms(sequence, pp_matrix, L)

    ncm_set: Set[Tuple[int, int]] = set()
    all_ncms = []
    for ncm_list in [tandem_ncms, miniloop_ncms, shear_ncms]:
        for i, j, etype, conf in ncm_list:
            if (i, j) not in ncm_set:
                all_ncms.append((i, j, etype, conf))
                ncm_set.add((i, j))

    stack_ncms = _detect_stack_ncms(sequence, pp_matrix, ncm_set, L)
    for i, j, etype, conf in stack_ncms:
        if (i, j) not in ncm_set:
            all_ncms.append((i, j, etype, conf))
            ncm_set.add((i, j))

    # Filter: return only non-WC pairs that are not in the hard restraints
    # Return format: (i, j, weight, type) - type lets downstream code distinguish
    # HOOGSTEEN/SUGAR/SHEAR/STACK; different types have different geometric restraints
    # (e.g. SHEAR is a translated/sheared pair, so the WC target distance does not apply)
    hard_set = set((i, j) for i, j, *_ in hard_pairs)
    result = [(i, j, w, etype) for i, j, etype, w in all_ncms
              if etype != "WC" and (i, j) not in hard_set]

    # Thermodynamic filter: suppress NCM candidates where WC alternative is far more favorable
    if apply_thermo_filter and result:
        try:
            from .ncm_thermo_filter import filter_ncm_candidates
            result = filter_ncm_candidates(sequence, result, bpp_matrix=pp_matrix)
        except ImportError:
            # If ncm_thermo_filter not available, skip filtering
            pass

    return result


def get_wc_matrix(
    sequence: str,
    threshold: float = WC_SOFT_THRESHOLD,
) -> np.ndarray:
    """Get WC-only pairing probability matrix (for downstream use).

    Returns:
        np.ndarray shape (L, L) with WC pair probabilities above threshold.
    """
    pp_matrix, _ = _get_pf_matrix(sequence)
    L = len(sequence)
    wc_mat = np.zeros((L, L), dtype=np.float64)

    for i in range(L):
        for j in range(i +1, L):
            if pp_matrix[i, j] >= threshold:
                b1, b2 = sequence[i], sequence[j]
                if _is_wc(b1, b2):
                    wc_mat[i, j] = pp_matrix[i, j]
                    wc_mat[j, i] = pp_matrix[i, j]

    return wc_mat


def format_pair_output(pairs: List[Tuple[int, int, str, float]]) -> str:
    """Pretty-print pair assembly results."""
    lines = [f"Total pairs: {len(pairs)}"]
    wc_count = sum(1 for _, _, et, _ in pairs if et == "WC")
    ncm_count = len(pairs) - wc_count
    lines.append(f"  WC pairs: {wc_count}")
    lines.append(f"  NCM pairs: {ncm_count}")

    # NCM type breakdown
    ncm_types = {}
    for _, _, et, _ in pairs:
        if et != "WC":
            ncm_types[et] = ncm_types.get(et,0) +1
    if ncm_types:
        lines.append(f"  NCM breakdown: {ncm_types}")

    lines.append("\nFirst20 pairs:")
    for i, (pos_i, pos_j, etype, conf) in enumerate(pairs[:20]):
        lines.append(f"  [{i:3d}] ({pos_i:4d}, {pos_j:4d}) {etype:10s} conf={conf:.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick self-test with a99nt sequence
    test_seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    print(f"Sequence length: {len(test_seq)}")
    print(f"Sequence: {test_seq[:50]}...")

    pairs = global_pair_assembly(test_seq)
    print(format_pair_output(pairs))
