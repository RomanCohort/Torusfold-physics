"""ncm_fusion.py — Hierarchical fusion layer for non-canonical motif (NCM) prediction.

Layer 0 (recall):  vectorized scan + motif features -> candidate set
Layer 1 (evidence): each candidate gets an evidence VECTOR
Layer 2 (decision): lightweight RandomForest learns optimal combination

Feature vector (9-dim):
  [conf, bpp_prob, mean_dist_pred, seq_gap,
   type_hoogsteen, type_sugar, type_shear, type_ensemble_dist,
   is_flanked_by_wc]

Usage:
    # Bootstrap training (no external annotations needed):
    from torusfold.scheme2.ncm_fusion import bootstrap_weak_labels, train_model
    train, X, y = bootstrap_weak_labels(sequence, bpp_matrix, hard_pairs)
    model = train_model(X, y)

    # Inference:
    from torusfold.scheme2.ncm_fusion import fuse_candidates
    fused = fuse_candidates(candidates, context)

    # Integration point (will wire into isrnaclong.py Level 0 later):
    from torusfold.scheme2.ncm_fusion import fuse_level0_candidates
    all_pairs = fuse_level0_candidates(sequence, hard_pairs, bpp_matrix)
"""
from __future__ import annotations

import os
import sys

# Ensure project src/ is on sys.path so torusfold is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Feature dimension
FEATURE_DIM = 9

# NCM type to one-hot index (4 types)
_TYPE_TO_IDX = {
    "HOOGSTEEN": 0,
    "SUGAR": 1,
    "SHEAR": 2,
    "ENSEMBLE_DIST": 3,
}

# Default model path
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
DEFAULT_MODEL_PATH = os.path.join(_DATA_DIR, "ncm_fusion_model.pkl")

# Hand-tuned fallback weights (preserves current behavior when no model exists)
_FALLBACK_WEIGHTS = {
    "heuristic_conf": 0.55,
    "ensemble_dist_conf": 0.30,
    "thermo_penalty": 0.15,
}

# Watson-Crick pairs
_WC_PAIRS: Set[Tuple[str, str]] = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
}
_GU_WOBBLE: Set[Tuple[str, str]] = {("G", "U"), ("U", "G")}


def _is_wc(b1: str, b2: str) -> bool:
    return (b1, b2) in _WC_PAIRS or (b1, b2) in _GU_WOBBLE


def extract_features(
    candidate: Tuple[int, int, str, float],
    context: Dict,
) -> np.ndarray:
    """Extract 9-dim feature vector for an NCM candidate.

    Args:
        candidate: (i, j, ncm_type, confidence)
            - i, j: 0-based residue indices
            - ncm_type: "HOOGSTEEN"/"SUGAR"/"SHEAR"/"ENSEMBLE_DIST"/"STACK"
            - confidence: heuristic confidence from detector
        context: dict with keys:
            - "bpp": np.ndarray (L,L) base pair probability matrix
            - "sequence": str, RNA sequence
            - "wc_pairs": set of (i,j) Watson-Crick pairs
            - "ens_dist_pairs" (optional): list of (i,j,type,conf) from ensemble
            - "thermo_penalty" (optional): float, thermodynamic penalty [0,1]

    Returns:
        np.ndarray shape (FEATURE_DIM,) with 9 features:
          [conf, bpp_prob, mean_dist_pred, seq_gap,
           type_hoogsteen, type_sugar, type_shear, type_ensemble_dist,
           is_flanked_by_wc]
    """
    i, j, ncm_type, conf = candidate
    bpp = context["bpp"]
    sequence = context["sequence"]
    wc_pairs = context.get("wc_pairs", set())
    ens_dist_pairs = context.get("ens_dist_pairs", [])

    L = len(sequence)

    # Feature 0: heuristic confidence
    f_conf = float(conf)

    # Feature 1: base pair probability
    f_bpp = float(bpp[i, j]) if 0 <= i < L and 0 <= j < L and bpp is not None else 0.0

    # Feature 2: mean distance prediction from ensemble (-1 if unavailable)
    f_dist = -1.0
    if ens_dist_pairs:
        dist_vals = [
            float(ep[3]) for ep in ens_dist_pairs
            if ep[0] == i and ep[1] == j
        ]
        if dist_vals:
            f_dist = float(np.mean(dist_vals))

    # Feature 3: sequence gap
    f_gap = float(j - i)

    # Features 4-7: type one-hot (4 dims)
    f_type = np.zeros(4, dtype=np.float64)
    if ncm_type in _TYPE_TO_IDX:
        f_type[_TYPE_TO_IDX[ncm_type]] = 1.0
    elif ncm_type == "STACK":
        # STACK doesn't have its own one-hot; leave as zeros
        pass

    # Feature 8: is flanked by WC pairs (both sides)
    f_flanked = 0.0
    if i > 0 and j < L - 1:
        if (i - 1, j + 1) in wc_pairs and (i + 1, j - 1) in wc_pairs:
            f_flanked = 1.0

    features = np.array([
        f_conf, f_bpp, f_dist, f_gap,
        f_type[0], f_type[1], f_type[2], f_type[3],
        f_flanked,
    ], dtype=np.float64)

    return features


def _check_sklearn():
    """Check if sklearn is available; raise ImportError with clear message if not."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        return True
    except ImportError:
        return False


def train_from_annotations(
    annotation_dir: str,
    bpp_matrices: Dict[str, np.ndarray],
    sequences: Dict[str, str],
) -> Tuple[Optional[object], np.ndarray, np.ndarray]:
    """Train fusion model from external annotations (FR3D-style or CSV).

    Expected CSV format (one file per sequence or combined):
        seq_id,i,j,label
        1A,10,50,1
        1A,20,45,0
    label: 1 = true NCM, 0 = false positive

    Args:
        annotation_dir: path to directory containing CSV annotation files
        bpp_matrices: {seq_id: np.ndarray (L,L)} base pair probability matrices
        sequences: {seq_id: str} RNA sequences

    Returns:
        (model, X, y) or (None, empty, empty) if sklearn unavailable or no data
    """
    if not _check_sklearn():
        warnings.warn("sklearn not available; training blocked. Install scikit-learn.")
        return None, np.empty((0, FEATURE_DIM)), np.empty(0, dtype=int)

    import joblib
    from sklearn.ensemble import RandomForestClassifier

    all_features = []
    all_labels = []

    for fname in os.listdir(annotation_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(annotation_dir, fname)
        with open(fpath, "r") as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                seq_id, i_str, j_str, label_str = parts[0], parts[1], parts[2], parts[3]
                if seq_id not in bpp_matrices or seq_id not in sequences:
                    continue
                i, j, label = int(i_str), int(j_str), int(label_str)
                seq = sequences[seq_id]
                bpp = bpp_matrices[seq_id]
                L = len(seq)

                # Build WC pairs for this sequence
                wc_pairs = set()
                for wi in range(L):
                    for wj in range(wi + 1, L):
                        if _is_wc(seq[wi], seq[wj]) and bpp[wi, wj] > 0.9:
                            wc_pairs.add((wi, wj))

                # Determine NCM type from sequence
                b1, b2 = seq[i], seq[j]
                ncm_type = "HOOGSTEEN"  # default
                from torusfold.scheme2.ncm_detector import _get_ncm_type
                detected = _get_ncm_type(b1, b2)
                if detected:
                    ncm_type = detected

                candidate = (i, j, ncm_type, float(bpp[i, j]))
                context = {
                    "bpp": bpp,
                    "sequence": seq,
                    "wc_pairs": wc_pairs,
                }
                feat = extract_features(candidate, context)
                all_features.append(feat)
                all_labels.append(label)

    if not all_features:
        return None, np.empty((0, FEATURE_DIM)), np.empty(0, dtype=int)

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=int)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42
    )
    clf.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
    joblib.dump(clf, DEFAULT_MODEL_PATH)
    print(f"[ncm_fusion] Model saved to {DEFAULT_MODEL_PATH}")

    return clf, X, y


def bootstrap_weak_labels(
    sequence: str,
    bpp_matrix: np.ndarray,
    hard_pairs: list,
    *,
    ens_dist_pairs: Optional[List[Tuple[int, int, str, float]]] = None,
    min_flank: int = 3,
    bpp_pos_thresh: float = 0.5,
    bpp_neg_thresh: float = 0.01,
    max_negatives: int = 500,
) -> Tuple[Optional[object], np.ndarray, np.ndarray]:
    """Generate weak training labels WITHOUT external annotations.

    Positives: tandem-detected pairs with flank >= min_flank AND high bpp (> bpp_pos_thresh)
    Negatives: random non-WC pairs with bpp < bpp_neg_thresh, far from any WC pair

    This gives a bootstrapped model immediately usable for inference.

    Args:
        sequence: RNA sequence (ACGU)
        bpp_matrix: (L,L) base pair probability matrix
        hard_pairs: list of (i, j, w) hard constraint pairs
        ens_dist_pairs: optional list of (i, j, type, conf) from ensemble
        min_flank: minimum WC flank for positive labels
        bpp_pos_thresh: BPP threshold for positive candidates
        bpp_neg_thresh: BPP threshold for negative candidates
        max_negatives: maximum number of negative samples

    Returns:
        (model, X, y) or (None, empty, empty) if sklearn unavailable
    """
    if not _check_sklearn():
        warnings.warn("sklearn not available; bootstrap blocked. Install scikit-learn.")
        return None, np.empty((0, FEATURE_DIM)), np.empty(0, dtype=int)

    from sklearn.ensemble import RandomForestClassifier

    L = len(sequence)

    # Build WC pairs set
    wc_pairs: Set[Tuple[int, int]] = set()
    for i in range(L):
        for j in range(i + 1, L):
            if _is_wc(sequence[i], sequence[j]) and bpp_matrix[i, j] > 0.9:
                wc_pairs.add((i, j))

    # --- POSITIVE LABELS ---
    # Tandem-detected pairs with flank >= min_flank AND high bpp
    from torusfold.scheme2.ncm_detector import _build_seq_masks, _get_ncm_type

    is_nonwc, ncm_code = _build_seq_masks(sequence)
    code_to_type = {1: "HOOGSTEEN", 2: "SUGAR", 3: "SHEAR"}

    # Vectorized flank count (simplified: just check immediate flanks for speed)
    pos_candidates = []
    for gap in range(5, min(L, 50)):
        for i in range(L - gap):
            j = i + gap
            if not is_nonwc[i, j] or ncm_code[i, j] == 0:
                continue
            bpp_val = float(bpp_matrix[i, j])
            if bpp_val < bpp_pos_thresh:
                continue
            # Count flanks (simplified: check 1..min_flank)
            flank = 0
            for k in range(1, min_flank + 1):
                if (i - k, j + k) in wc_pairs:
                    flank += 1
                else:
                    break
            if flank < min_flank:
                continue
            ncm_t = code_to_type.get(int(ncm_code[i, j]), "HOOGSTEEN")
            pos_candidates.append((i, j, ncm_t, bpp_val))

    # --- NEGATIVE LABELS ---
    # Random non-WC pairs with low bpp, far from WC pairs
    neg_candidates = []
    wc_set = set(wc_pairs)
    rng = np.random.RandomState(42)
    attempts = 0
    while len(neg_candidates) < max_negatives and attempts < max_negatives * 10:
        attempts += 1
        i = int(rng.randint(0, L - 4))
        j = int(rng.randint(i + 4, L))
        if (i, j) in wc_set or (j, i) in wc_set:
            continue
        bpp_val = float(bpp_matrix[i, j])
        if bpp_val >= bpp_neg_thresh:
            continue
        # Check it's not flanked by WC (would be a false negative)
        flanked = False
        for k in range(1, 4):
            if (i - k, j + k) in wc_pairs and (i + k, j - k) in wc_pairs:
                flanked = True
                break
        if flanked:
            continue
        b1, b2 = sequence[i], sequence[j]
        ncm_t = _get_ncm_type(b1, b2) or "HOOGSTEEN"
        neg_candidates.append((i, j, ncm_t, bpp_val))

    # Combine
    all_candidates = [(c, 1) for c in pos_candidates] + [(c, 0) for c in neg_candidates]
    if not all_candidates:
        return None, np.empty((0, FEATURE_DIM)), np.empty(0, dtype=int)

    context = {
        "bpp": bpp_matrix,
        "sequence": sequence,
        "wc_pairs": wc_pairs,
        "ens_dist_pairs": ens_dist_pairs or [],
    }

    X = np.array([extract_features(c, context) for c, _ in all_candidates], dtype=np.float64)
    y = np.array([lab for _, lab in all_candidates], dtype=int)

    print(f"[ncm_fusion] Bootstrap: {len(pos_candidates)} positives, {len(neg_candidates)} negatives")

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=42
    )
    clf.fit(X, y)

    # Save
    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
    import joblib
    joblib.dump(clf, DEFAULT_MODEL_PATH)
    print(f"[ncm_fusion] Bootstrap model saved to {DEFAULT_MODEL_PATH}")

    return clf, X, y


def _fallback_confidence(candidate: Tuple[int, int, str, float], context: Dict) -> float:
    """Hand-tuned weighted average fallback when no trained model exists."""
    i, j, ncm_type, conf = candidate
    bpp = context.get("bpp")
    ens_dist_pairs = context.get("ens_dist_pairs", [])

    heuristic = float(conf)
    bpp_val = float(bpp[i, j]) if bpp is not None and 0 <= i < bpp.shape[0] and 0 <= j < bpp.shape[1] else 0.0
    ens_conf = 0.0
    if ens_dist_pairs:
        vals = [float(ep[3]) for ep in ens_dist_pairs if ep[0] == i and ep[1] == j]
        if vals:
            ens_conf = float(np.mean(vals))

    w = _FALLBACK_WEIGHTS
    fused = (
        w["heuristic_conf"] * heuristic
        + w["ensemble_dist_conf"] * ens_conf
        + w["thermo_penalty"] * bpp_val
    )
    return float(np.clip(fused, 0.0, 1.0))


def fuse_candidates(
    candidates: List[Tuple[int, int, str, float]],
    context: Dict,
    model_path: str = DEFAULT_MODEL_PATH,
) -> List[Tuple[int, int, str, float]]:
    """Fuse NCM candidates using learned RandomForest or hand-tuned fallback.

    Args:
        candidates: list of (i, j, ncm_type, confidence)
        context: dict with bpp, sequence, wc_pairs, (optional) ens_dist_pairs
        model_path: path to saved sklearn model

    Returns:
        list of (i, j, ncm_type, fused_confidence)
    """
    if not candidates:
        return []

    # Try loading model
    model = None
    if os.path.exists(model_path) and _check_sklearn():
        import joblib
        try:
            model = joblib.load(model_path)
        except Exception as e:
            warnings.warn(f"[ncm_fusion] Failed to load model: {e}")

    results = []
    if model is not None:
        X = np.array([extract_features(c, context) for c in candidates], dtype=np.float64)
        proba = model.predict_proba(X)
        # Class 1 = true NCM
        if proba.shape[1] == 2:
            fused_confs = proba[:, 1]
        else:
            fused_confs = proba[:, 0]
        for (i, j, ncm_type, _orig_conf), fc in zip(candidates, fused_confs):
            results.append((i, j, ncm_type, float(fc)))
    else:
        for c in candidates:
            i, j, ncm_type, _orig_conf = c
            fc = _fallback_confidence(c, context)
            results.append((i, j, ncm_type, fc))

    return results


def fuse_level0_candidates(
    sequence: str,
    hard_pairs: list,
    bpp_matrix: np.ndarray,
    ens_dist_pairs: Optional[List[Tuple[int, int, str, float]]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> List[Tuple[int, int, str, float]]:
    """Integration point: run NCM detection + ensemble merge + fusion.

    This will be wired into isrnaclong.py Level 0 later.

    Args:
        sequence: RNA sequence (ACGU)
        hard_pairs: list of (i, j, w) hard constraint pairs
        bpp_matrix: (L,L) base pair probability matrix
        ens_dist_pairs: optional ensemble distance NCM candidates
        model_path: path to trained fusion model

    Returns:
        Fused NCM candidates as (i, j, ncm_type, fused_confidence)
    """
    from torusfold.scheme2.ncm_detector import detect_ncms_from_bpp

    # Layer 0: heuristic detection
    ncm_raw = detect_ncms_from_bpp(sequence, bpp_matrix, hard_pairs)
    # Convert from (i, j, weight, type) to (i, j, type, conf)
    candidates = [(i, j, etype, w) for i, j, w, etype in ncm_raw]

    # Merge ensemble distance candidates if available
    if ens_dist_pairs:
        seen = {(c[0], c[1]) for c in candidates}
        for ep in ens_dist_pairs:
            if (ep[0], ep[1]) not in seen:
                candidates.append(ep)
                seen.add((ep[0], ep[1]))

    # Build WC pairs set for context
    L = len(sequence)
    wc_pairs: Set[Tuple[int, int]] = set()
    for i in range(L):
        for j in range(i + 1, L):
            if _is_wc(sequence[i], sequence[j]) and bpp_matrix[i, j] > 0.9:
                wc_pairs.add((i, j))

    context = {
        "bpp": bpp_matrix,
        "sequence": sequence,
        "wc_pairs": wc_pairs,
        "ens_dist_pairs": ens_dist_pairs or [],
    }

    # Layer 2: fusion
    return fuse_candidates(candidates, context, model_path=model_path)


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ncm_fusion.py — self-test")
    print("=" * 60)

    # Build a structured 300nt sequence with tandem stems and NCM motifs
    # Layout: stem1 [WC 14→40, 13→41, ..., 0→54] NCM(15,39) WC [16→38, 17→37, ...]
    #         + stem2, stem3 for additional structure
    np.random.seed(42)
    seq_len = 300

    seq_list = list("N" * seq_len)
    bpp = np.zeros((seq_len, seq_len), dtype=np.float64)
    wc_pairs = set()

    # Stem 1a: positions 0-13 paired with 54-41 (descending j)
    # This ensures (14,40) and (16,38) flank the NCM at (15,39)
    for k in range(14):
        i, j = k, 54 - k   # (0,54), (1,53), ..., (13,41)
        if k % 2 == 0:
            seq_list[i], seq_list[j] = "G", "C"
        else:
            seq_list[i], seq_list[j] = "A", "U"
        bpp[i, j] = bpp[j, i] = 0.95
        wc_pairs.add((i, j))

    # Flanking WC pairs for the tandem NCM
    # (14, 40) — left flank
    seq_list[14], seq_list[40] = "G", "C"
    bpp[14, 40] = bpp[40, 14] = 0.95
    wc_pairs.add((14, 40))
    # (16, 38) — right flank
    seq_list[16], seq_list[38] = "G", "C"
    bpp[16, 38] = bpp[38, 16] = 0.95
    wc_pairs.add((16, 38))

    # The tandem NCM: (15, 39) = A-A (SHEAR type)
    seq_list[15], seq_list[39] = "A", "A"
    bpp[15, 39] = bpp[39, 15] = 0.6

    # Stem 2: positions 60-69 paired with 90-99
    for k in range(10):
        i, j = 60 + k, 90 + k
        if k % 2 == 0:
            seq_list[i], seq_list[j] = "G", "C"
        else:
            seq_list[i], seq_list[j] = "U", "A"
        bpp[i, j] = bpp[j, i] = 0.92
        wc_pairs.add((i, j))

    # Stem 3: positions 200-207 paired with 220-227
    for k in range(8):
        i, j = 200 + k, 220 + k
        if k % 2 == 0:
            seq_list[i], seq_list[j] = "G", "C"
        else:
            seq_list[i], seq_list[j] = "A", "U"
        bpp[i, j] = bpp[j, i] = 0.88
        wc_pairs.add((i, j))

    # Fill remaining with random bases
    for idx in range(seq_len):
        if seq_list[idx] == "N":
            seq_list[idx] = np.random.choice(list("ACGU"))

    seq = "".join(seq_list)

    hard_pairs = [(i, j, 1.0) for i, j in list(wc_pairs)[:30]]

    print(f"Sequence length: {seq_len}")
    print(f"WC pairs: {len(wc_pairs)}")
    print(f"Hard pairs (top 30): {len(hard_pairs)}")

    # Test feature extraction
    test_candidate = (15, 39, "SHEAR", 0.6)
    test_context = {
        "bpp": bpp,
        "sequence": seq,
        "wc_pairs": wc_pairs,
        "ens_dist_pairs": [(15, 39, "ENSEMBLE_DIST", 0.55)],
    }
    feat = extract_features(test_candidate, test_context)
    print(f"\nFeature vector shape: {feat.shape}")
    print(f"Feature vector: {feat}")
    assert feat.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {feat.shape}"

    # Test bootstrap training
    print("\n--- Bootstrap weak label training ---")
    model, X, y = bootstrap_weak_labels(seq, bpp, hard_pairs)
    if model is not None:
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        print(f"Model trained: {X.shape[0]} samples ({n_pos} pos, {n_neg} neg), {X.shape[1]} features")

        # Test inference with the trained model
        print("\n--- Inference test (with trained model) ---")
        test_candidates = [
            (15, 39, "SHEAR", 0.6),     # the tandem NCM we planted
            (5, 35, "HOOGSTEEN", 0.5),  # random pair
            (16, 38, "WC", 0.95),       # WC pair (should stay high)
            (100, 150, "SUGAR", 0.45),  # isolated pair
        ]
        fused = fuse_candidates(test_candidates, test_context)
        print(f"\n{'Pos':>5s} {'Type':>14s} {'Original':>10s} {'Fused':>10s}")
        print("-" * 45)
        for (i, j, t, orig), (_, _, _, fc) in zip(test_candidates, fused):
            print(f"({i:3d},{j:3d}) {t:>14s} {orig:>10.3f} {fc:>10.3f}")

        # Delete model so fallback path is also tested
        os.remove(DEFAULT_MODEL_PATH)

    # Test fallback path (no model)
    print("\n--- Inference test (fallback, no model) ---")
    test_candidates_fb = [
        (15, 39, "SHEAR", 0.6),
        (5, 35, "HOOGSTEEN", 0.5),
    ]
    fused_fb = fuse_candidates(test_candidates_fb, test_context)
    for (i, j, t, orig), (_, _, _, fc) in zip(test_candidates_fb, fused_fb):
        print(f"  {t:>14s}: {orig:.3f} -> {fc:.3f}")

    # Test integration point
    print("\n--- fuse_level0_candidates test ---")
    fused_pairs = fuse_level0_candidates(seq, hard_pairs, bpp)
    print(f"Fused pairs returned: {len(fused_pairs)}")
    for p in fused_pairs[:8]:
        print(f"  ({p[0]:3d}, {p[1]:3d}) {p[2]:>14s} conf={p[3]:.3f}")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)
