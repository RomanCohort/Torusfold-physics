"""immune_heuristic.py - scheme2 geometric-heuristic immune fingerprints (10 indicators).

Until the cloud weights for the trained scheme10 ImmuneFingerprintHeads neural
heads arrive, compute the 10 immune indicators from all-atom coordinates +
sequence + ViennaRNA base pairing via geometric heuristics, so the frontend
Coloring Scheme dropdown recovers pkr/m6a/tlr7/rigi/nlrp3/sponge etc. Field names
align with export.py's default_per_res / default_scalar; replace once weights are available.

Does not depend on whether the amber refinement ran - it works on the reconstructed
all-atom coordinates (ideal A-form geometry). Lower accuracy, but enough for figure coloring.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _freesasa_per_residue(structure, L: int) -> np.ndarray:
    """Per-residue SASA (A^2) via FreeSASA Shrake-Rupley.

    Feeds all-atom coordinates to FreeSASA and aggregates by residueNumber.
    On failure (missing library / anomalous coordinates) returns None; the caller
    falls back to the C1' nearest-neighbor proxy.
    """
    try:
        import freesasa  # type: ignore
    except ImportError:
        return None

    try:
        atom_names = [a.atom_name for a in structure.atoms]
        res_names = [a.res_name for a in structure.atoms]
        res_nums = [str(a.res_seq) for a in structure.atoms]
        chains = ['A'] * len(structure.atoms)
        xs = [float(a.xyz[0]) for a in structure.atoms]
        ys = [float(a.xyz[1]) for a in structure.atoms]
        zs = [float(a.xyz[2]) for a in structure.atoms]

        struct = freesasa.Structure()
        struct.addAtoms(atom_names, res_names, res_nums, chains, xs, ys, zs)
        classifier = freesasa.Classifier()
        struct.setRadiiWithClassifier(classifier)
        result = freesasa.calc(struct)

        # residueAreas: {chain: {resNum_str: ResidueArea}}
        res_areas = result.residueAreas()
        sasa = np.zeros(L, dtype=np.float32)
        chain_areas = res_areas.get('A', {})
        for i in range(L):
            ra = chain_areas.get(str(i + 1))
            if ra is not None:
                sasa[i] = float(ra.total)
        return sasa
    except Exception:
        return None


def _c1p_nn_sasa(structure, L: int) -> np.ndarray:
    """C1' nearest-neighbor distance proxy for SASA (fallback; larger exposure -> larger value)."""
    c1_xyzs = np.array([
        structure.atoms[structure.residue_atom_index[i]["C1'"]].xyz
        for i in range(L)
    ], dtype=np.float64)
    if L > 1:
        d = np.linalg.norm(c1_xyzs[:, None] - c1_xyzs[None], axis=2)
        np.fill_diagonal(d, 1e9)
        return np.clip(d.min(axis=1) / 15.0, 0.0, 1.0).astype(np.float32)
    return np.ones(L, dtype=np.float32)


def _extract_stem_segments(pairs: List[Tuple[int, int, float]],
                           L: int) -> List[Tuple[int, int]]:
    """Extract consecutive pairing segments (stems) from ViennaRNA pairs.

    Returns [(stem_len, n_pairs), ...], each segment's length and pair count.
    Consecutive pairing: adjacent i-chain residues pair to adjacent j-chain residues
    (i+1 pairs with j-1).
    """
    if not pairs:
        return []
    # build i->j map
    pair_map = {}
    for i, j, _w in pairs:
        if 0 <= i < L and 0 <= j < L:
            pair_map[i] = j
    sorted_i = sorted(pair_map.keys())
    segments = []
    cur_len = 1
    for k in range(1, len(sorted_i)):
        prev_i, cur_i = sorted_i[k - 1], sorted_i[k]
        prev_j, cur_j = pair_map[prev_i], pair_map[cur_i]
        # consecutive: i+1, j-1 (antiparallel pairing)
        if cur_i == prev_i + 1 and cur_j == prev_j - 1:
            cur_len += 1
        else:
            segments.append((cur_len, cur_len))
            cur_len = 1
    segments.append((cur_len, cur_len))
    return segments


def compute_immune_fingerprints(
    coords_aa: np.ndarray,
    structure,
    pairs: List[Tuple[int, int, float]],
    sequence: str,
) -> Dict[str, np.ndarray]:
    """Compute the 10 immune indicators via geometric heuristics.

    Args:
        coords_aa: (N, 3) all-atom coordinates (with or without H; only the C1' index is used)
        structure: AllAtomStructure (used to look up the C1' index per residue)
        pairs: ViennaRNA pairs [(i, j, w), ...] 0-based
        sequence: ACGU string

    Returns:
        dict of per-residue fingerprints (length L) + scalar fingerprints (length 1).
        Field names align with export.py default_per_res / default_scalar.
    """
    L = len(sequence)

    # C1' coordinates per residue (A)
    c1_xyzs = np.array([
        structure.atoms[structure.residue_atom_index[i]["C1'"]].xyz
        for i in range(L)
    ], dtype=np.float64)

    # --- stem/loop markers (for the m6A drach_in_loop feature) ---
    paired = set()
    for (i, j, w) in pairs:
        if 0 <= i < L:
            paired.add(i)
        if 0 <= j < L:
            paired.add(j)

    # --- PKR SASA exposure (true FreeSASA SASA; falls back to the C1' nearest-neighbor proxy) ---
    sasa_free = _freesasa_per_residue(structure, L)
    if sasa_free is not None:
        # normalize to [0,1]: fully exposed ~350-400 A^2, fully buried ~50-100 A^2
        pkr_sasa = np.clip((sasa_free - 50.0) / 300.0, 0.0, 1.0).astype(np.float32)
    else:
        pkr_sasa = _c1p_nn_sasa(structure, L)

    # --- pkr_stem_logit: fraction of long stems (>10 bp); PKR activation needs long dsRNA stems ---
    stem_segments = _extract_stem_segments(pairs, L)
    long_stem_bases = sum(s[0] for s in stem_segments if s[0] > 10)
    paired_count = sum(s[1] for s in stem_segments)
    pkr_stem_logit = np.full(L, min(1.0, long_stem_bases / max(1, L)),
                             dtype=np.float32) if paired_count > 0 else np.zeros(L, dtype=np.float32)

    # --- m6A: DRACH motif detection ---
    drach_is = np.zeros(L, dtype=np.float32)
    drach_in_loop = np.zeros(L, dtype=np.float32)
    m6a_prob = np.zeros(L, dtype=np.float32)
    for i in range(L):
        if sequence[i] != "A" or i < 2 or i > L - 3:
            continue
        mer = sequence[i - 2:i + 3]
        # DRACH: D=A/G/U, R=A/G, A, C, H=A/C/U
        if (mer[0] in "AGU" and mer[1] in "AG" and mer[2] == "A"
                and mer[3] == "C" and mer[4] in "ACU"):
            drach_is[i] = 1.0
            if i not in paired:
                drach_in_loop[i] = 1.0
            # exposure (FreeSASA) + loop region -> higher write probability (heuristic weighting)
            m6a_prob[i] = (0.5
                           + 0.3 * drach_in_loop[i]
                           + 0.2 * pkr_sasa[i])

    # --- TLR7: GU-rich density (fraction of G+U in a 5-nt window) ---
    tlr7 = np.zeros(L, dtype=np.float32)
    for i in range(L):
        lo, hi = max(0, i - 2), min(L, i + 3)
        window = sequence[lo:hi]
        tlr7[i] = sum(1 for c in window if c in "GU") / len(window)

    # --- RIG-I: short dsRNA (<30 bp) recognition, derived from stem-segment lengths ---
    # RIG-I recognizes short duplex RNA (typically 10-30 bp); short internal stems in circRNA activate RIG-I
    # per-residue: residues on short stems (>5 bp and <30 bp) get an activation value
    rigi_per = np.zeros(L, dtype=np.float32)
    if stem_segments:
        # rebuild i->j and find the residues covered by each segment
        pair_map = {}
        for i, j, _w in pairs:
            if 0 <= i < L and 0 <= j < L:
                pair_map[i] = j
        sorted_i = sorted(pair_map.keys())
        seg_starts = []
        if sorted_i:
            seg_start = sorted_i[0]
            for k in range(1, len(sorted_i)):
                prev_i, cur_i = sorted_i[k - 1], sorted_i[k]
                prev_j, cur_j = pair_map[prev_i], pair_map[cur_i]
                if not (cur_i == prev_i + 1 and cur_j == prev_j - 1):
                    seg_starts.append((seg_start, sorted_i[k - 1]))
                    seg_start = cur_i
            seg_starts.append((seg_start, sorted_i[-1]))
        for seg_idx, (s, e) in enumerate(seg_starts):
            seg_len = e - s + 1
            if 5 < seg_len < 30:
                j_end = pair_map[s]
                j_start = pair_map[e]
                # activation grows with segment length (saturating)
                activation = min(1.0, seg_len / 20.0)
                for idx in range(s, e + 1):
                    rigi_per[idx] = activation
                for idx in range(min(j_start, j_end), max(j_start, j_end) + 1):
                    rigi_per[idx] = activation
    rigi_score = float(rigi_per.mean())

    # --- NLRP3: long dsRNA (>19 bp) persistence length, derived from stem segments ---
    # NLRP3 recognizes longer dsRNA (>19 bp); persistence_length is the longest stem-segment length
    if stem_segments:
        longest_stem = max(s[0] for s in stem_segments)
        nlrp3_persist = float(min(1.0, longest_stem / 30.0))
    else:
        nlrp3_persist = 0.0

    # --- miRNA sponge: true miRBase duplex evaluation (falls back to a GU-content proxy on failure) ---
    try:
        from .mirna_sponge import compute_sponge, DEFAULT_MATURE_FA
        sponge_result = compute_sponge(sequence, DEFAULT_MATURE_FA)
        sponge_score = float(sponge_result["sponge_score"])
    except Exception:
        gu_frac = sum(1 for c in sequence if c in "GU") / max(1, L)
        sponge_score = float(gu_frac * min(L / 200.0, 1.0))

    return {
        # per-residue (length L)
        "pkr_sasa": pkr_sasa,
        "pkr_stem_logit": pkr_stem_logit,
        "drach_is_drach": drach_is,
        "drach_in_loop": drach_in_loop,
        "m6a_write_prob": m6a_prob,
        "tlr7_gu_density": tlr7,
        "rigi_per_pos": rigi_per,
        # scalar (length 1)
        "nlrp3_persistence_length": np.array([nlrp3_persist],
                                              dtype=np.float32),
        "sponge_score": np.array([sponge_score], dtype=np.float32),
        "rigi_score": np.array([rigi_score], dtype=np.float32),
    }


def compute_structure_signals(
    coords_aa: np.ndarray,
    structure,
    pairs: List[Tuple[int, int, float]],
    bpp: np.ndarray,
    sequence: str,
    e1_aa: float,
    bsj_dist: float,
    cg_coords: np.ndarray,
) -> Dict[str, float]:
    """Pure-computation structure signals (aligned with native TorusFold TorusFoldSignals + ImmuneSensingResultV3 passthrough fields).

    Covers group A (sequence/pairing-derived) + group B (geometry-derived) + G physical
    fingerprints, none of which depend on DL. Field names strictly follow the native
    contract so downstream ImmuneSensingResultV3 no longer goes through the heuristic fallback.

    Args:
        coords_aa: (N, 3) all-atom coordinates (post-refinement)
        structure: AllAtomStructure (look up C1'/P indices)
        pairs: ViennaRNA pairs [(i, j, w), ...] 0-based
        bpp: (L, L) ViennaRNA base-pair probability matrix
        sequence: ACGU string
        e1_aa: amber-refined energy (kJ/mol)
        bsj_dist: BSJ distance (A), ||coords[0]-coords[-1]||
        cg_coords: (L, 3) CG P coordinates (post-refinement), used for geometric indicators

    Returns:
        dict[str, float], all scalars, field names aligned with the native contract.
    """
    L = len(sequence)
    bond_length = 5.9
    pair_dist_target = 10.6
    clash_dist = 3.0

    # --- Group A: sequence/pairing-derived (ViennaRNA bpp) ---
    upper = np.triu(bpp, k=1)
    pair_mask = upper > 0.5
    dsrna_fraction = float(pair_mask.sum() * 2 / max(1, L))
    mean_pair_prob = float(upper.sum() * 2 / max(1, L * (L - 1)))

    # long_range_pair_fraction: share of pairs with circ_dist > L/4
    pair_count = int(pair_mask.sum())
    if pair_count > 0:
        long_range = 0
        for i, j in zip(*np.where(pair_mask)):
            circ = min(abs(i - j), L - abs(i - j))
            if circ > L / 4:
                long_range += 1
        long_range_frac = float(long_range / pair_count)
    else:
        long_range_frac = 0.0

    # pairing_stability: 1 - normalized entropy of pair probabilities (low entropy = confident pairing = stable)
    probs = upper[upper > 1e-6]
    if probs.size > 0:
        p = probs / probs.sum()
        entropy = float(-(p * np.log(p)).sum())
        max_entropy = float(np.log(probs.size))
        pairing_stability = float(1.0 - entropy / max_entropy) if max_entropy > 0 else 0.5
    else:
        pairing_stability = 0.5

    # --- Group B: geometry-derived (from cg_coords + coords_aa) ---
    closure_err = abs(bsj_dist - bond_length)
    closure_score = float(max(0.0, 1.0 - closure_err / 2.0))

    # bond_rmsd: deviation of adjacent P-P distances
    bb = np.linalg.norm(np.diff(cg_coords, axis=0), axis=1)
    bond_rmsd = float(np.sqrt(((bb - bond_length) ** 2).mean()))

    # pair_satisfaction: fraction of pairs meeting the P-P target distance (10.6 +/- 1.5)
    if pairs:
        pair_hits = 0
        for i, j, _w in pairs:
            if 0 <= i < L and 0 <= j < L:
                d = float(np.linalg.norm(cg_coords[i] - cg_coords[j]))
                if abs(d - pair_dist_target) < 1.5:
                    pair_hits += 1
        pair_satisfaction = float(pair_hits / len(pairs))
    else:
        pair_satisfaction = 0.0

    # clash_count: non-adjacent P-P < 3.0
    clash_count = 0
    for a in range(L):
        for b in range(a + 2, L):
            if (a, b) == (0, L - 1):
                continue
            if float(np.linalg.norm(cg_coords[a] - cg_coords[b])) < clash_dist:
                clash_count += 1

    # SASA: C1' nearest-neighbor distance proxy (exposed = far nearest neighbor)
    c1_xyzs = np.array([
        structure.atoms[structure.residue_atom_index[i]["C1'"]].xyz
        for i in range(L)
    ], dtype=np.float64)
    if L > 1:
        d = np.linalg.norm(c1_xyzs[:, None] - c1_xyzs[None], axis=2)
        np.fill_diagonal(d, 1e9)
        nn = d.min(axis=1)
        sasa_per = np.clip(nn / 15.0, 0.0, 1.0)
    else:
        sasa_per = np.ones(L)
    sasa_mean = float(sasa_per.mean())
    # SASA over the BSJ region (3 residues at each end)
    bsj_idx = list(range(min(3, L))) + list(range(max(0, L - 3), L))
    sasa_bsj = float(sasa_per[bsj_idx].mean()) if bsj_idx else sasa_mean
    surface_exposed_fraction = float((sasa_per > 0.5).mean())

    # bsj_stability / bsj_confidence: derived from closure + energy
    bsj_stability = float(0.5 + 0.3 * closure_score + 0.2 * (1.0 if e1_aa < 0 else 0.0))
    bsj_stability = min(1.0, bsj_stability)
    bsj_confidence = float(closure_score * 0.7 + min(1.0, max(0.0, -e1_aa / 10000.0)) * 0.3)

    # bsj_3d_closure_tightness: compactness of the BSJ region (normalized radius of gyration of the 3 end residues)
    if len(bsj_idx) > 1:
        bsj_pts = c1_xyzs[bsj_idx]
        rg_bsj = float(np.sqrt(((bsj_pts - bsj_pts.mean(axis=0)) ** 2).sum(axis=1).mean()))
        bsj_3d_closure_tightness = float(np.clip(1.0 - rg_bsj / 20.0, 0.0, 1.0))
    else:
        bsj_3d_closure_tightness = 0.5

    # --- Group G physical fingerprints (see torusfold-immune-fingerprints note: pure computation) ---
    # mechanical_stiffness = f(closure, pair_satisfaction, energy)
    energy_norm = float(min(1.0, max(0.0, -e1_aa / 50000.0))) if e1_aa < 0 else 0.0
    mechanical_stiffness = float(0.4 * closure_score + 0.3 * pair_satisfaction + 0.3 * energy_norm)
    mechanical_stiffness = min(1.0, mechanical_stiffness)

    # solvent_response = f(SASA, closure, IRES) - IRES is proxied by mean_pair_prob
    solvent_response = float(0.4 * sasa_mean + 0.3 * bsj_3d_closure_tightness + 0.3 * (1.0 - mean_pair_prob))
    solvent_response = min(1.0, solvent_response)

    return {
        # Group A
        "dsRNA_fraction": dsrna_fraction,
        "mean_pair_prob": mean_pair_prob,
        "long_range_pair_fraction": long_range_frac,
        "pairing_stability": pairing_stability,
        # Group B
        "closure_distance": float(bsj_dist),
        "closure_score": closure_score,
        "bond_rmsd": bond_rmsd,
        "pair_satisfaction": pair_satisfaction,
        "clash_count": clash_count,
        "sasa_mean": sasa_mean,
        "sasa_bsj": sasa_bsj,
        "surface_exposed_fraction": surface_exposed_fraction,
        "bsj_stability": bsj_stability,
        "bsj_confidence": bsj_confidence,
        "bsj_3d_closure_tightness": bsj_3d_closure_tightness,
        "energy_score": float(e1_aa),
        # Group G physical fingerprints
        "mechanical_stiffness": mechanical_stiffness,
        "solvent_response": solvent_response,
    }


if __name__ == "__main__":
    # self-check: rebuild from regular-polygon P coordinates, then compute indicators
    from .allatom_reconstruct import reconstruct_all_atom
    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"  # 32 nt
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles),
                   np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    coords_aa = np.array([a.xyz for a in s.atoms], dtype=np.float64)
    pairs = [(i, L - 1 - i, 1.0) for i in range(L // 2)]
    fp = compute_immune_fingerprints(coords_aa, s, pairs, seq)
    print(f"L={L} indicators={len(fp)}")
    for k, v in fp.items():
        print(f"  {k}: shape={v.shape} min={v.min():.3f} max={v.max():.3f}")
