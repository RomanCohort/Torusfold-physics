"""
pair_graph.py - circRNA pairing-graph construction + complementarity scan + topological distance.

Front-end module for RL far/long-range-pair optimization. Given ViennaRNA pairs
+ a sequence, it supplements the long-range pairs the complementarity scan
misses (e.g. HA inverted repeats), builds a graph (backbone adjacency + pairing
edges), computes topological distances with BFS, flags far/long-range pairs
(dist > 50), and extracts stem blocks.

circRNA circular topology: the backbone adjacency includes the (L-1, 0) BSJ
closure edge. The absolute |i-j| is misleading across the BSJ on a ring (e.g.
for L=2013, (5,2010) has |i-j|=2005 but a ring distance of 8), so graph distance
(BFS over backbone + pairing edges) is used to classify far/long-range pairs
correctly.

Parameters (frozen 2026-07-21; see docs/scheme2_rl_design.md):
  W = 6            sliding-window length (minimum length of a stable RNA stem)
  WC_RATE = 0.80   Watson-Crick pairing-rate threshold (allows one G.U wobble)
  DG_THRESHOLD = -5.0   simple NN free-energy threshold (kcal/mol; removes false positives)
  MIN_STEM = 4     minimum number of consecutive pairs (stem-block extraction)
  FAR_DIST = 50    topological-distance threshold (dist > 50 = far/long-range)
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------- Watson-Crick pairing rules ----------
# Standard WC: A-U, G-C, plus G-U (wobble)
_WC_PAIRS: Set[Tuple[str, str]] = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
}
_GU_WOBBLE: Set[Tuple[str, str]] = {("G", "U"), ("U", "G")}

# ---------- Algorithm parameters ----------
W = 6                 # sliding-window length
WC_RATE = 0.80        # WC pairing-rate threshold
DG_THRESHOLD = -3.0   # NN free-energy threshold (kcal/mol; relaxed for the coarse parameter table)
MIN_STEM = 4          # minimum consecutive pairs in a stem block
MIN_WC_IN_WIN = 4     # scheme E: min consecutive WC pairs inside a scan window (>=4 of 6)
FAR_DIST = 50         # far/long-range topological-distance threshold
MAX_KMER_FREQ = 20    # scheme H: skip rc matching when a single k-mer appears more than this (filters repeats)

# ---------- Simple RNA nearest-neighbor free-energy parameters ----------
# RNA NN model (SantaLucia 1998 approximations, kcal/mol at 37 deg C)
# Used to filter complementarity-scan false positives: only DG < -5 counts as a stable stem
_NN_DG: Dict[Tuple[str, str, str, str], float] = {
    # stacking free energy of 5'-XY-3' / 3'-X'Y'-5'
    ("A", "U", "A", "U"): -1.0, ("U", "A", "U", "A"): -1.0,
    ("A", "U", "C", "G"): -2.0, ("C", "G", "A", "U"): -2.0,
    ("G", "C", "A", "U"): -2.0, ("A", "U", "G", "C"): -2.0,
    ("G", "C", "G", "C"): -3.0, ("C", "G", "C", "G"): -3.0,
    ("G", "C", "U", "A"): -2.0, ("U", "A", "G", "C"): -2.0,
    ("C", "G", "U", "A"): -1.0, ("U", "A", "C", "G"): -1.0,
    ("A", "U", "U", "A"): -0.5, ("U", "A", "U", "A"): -0.5,
    ("G", "U", "A", "U"): -1.0, ("A", "U", "G", "U"): -1.0,
    ("G", "U", "C", "G"): -1.5, ("C", "G", "G", "U"): -1.5,
    ("G", "U", "U", "A"): -0.5, ("U", "A", "G", "U"): -0.5,
    ("G", "U", "G", "C"): -1.5, ("C", "G", "G", "U"): -1.5,
}
# Default stack energy (fallback when not found in the table)
_NN_DEFAULT = -1.0


def _is_wc_pair(b1: str, b2: str) -> bool:
    """Canonical Watson-Crick pairing (A-U, G-C)."""
    return (b1, b2) in _WC_PAIRS


def _is_complementary(b1: str, b2: str) -> bool:
    """Watson-Crick or G.U wobble pairing."""
    return (b1, b2) in _WC_PAIRS or (b1, b2) in _GU_WOBBLE


def _wc_count(seq_a: str, seq_b: str) -> int:
    """Number of WC pairs between seq_a and its antiparallel complement seq_b.

    win_b is the reverse-complement candidate of win_a: seq_a[k] pairs with seq_b[W-1-k].
    """
    n = min(len(seq_a), len(seq_b))
    cnt = 0
    for k in range(n):
        if _is_wc_pair(seq_a[k], seq_b[n - 1 - k]):
            cnt += 1
    return cnt


def _nn_free_energy(seq_a: str, seq_b: str) -> float:
    """Simple nearest-neighbor stacking free energy (kcal/mol).

    seq_a[k] pairs with seq_b[n-1-k] (antiparallel). A stack is two consecutive pairs.
    """
    n = min(len(seq_a), len(seq_b))
    if n < 2:
        return 0.0
    dg = 0.0
    for k in range(n - 1):
        x, y = seq_a[k], seq_a[k + 1]
        # Pairing partners: seq_b[n-1-k], seq_b[n-1-(k+1)] = seq_b[n-2-k]
        xp, yp = seq_b[n - 1 - k], seq_b[n - 2 - k]
        dg += _NN_DG.get((x, y, xp, yp), _NN_DEFAULT)
    return dg


# ---------- Functional-region parsing (case mask) ----------
def parse_case_annotation(
    sequence: str,
    *,
    default_coding: bool = False,
) -> np.ndarray:
    """Parse a coding mask from the letter case of the sequence.

    Engineered vector sequences are often a mix of upper/lowercase: uppercase
    segments are functional elements (ORF/IRES/key motifs) and lowercase
    segments are UTR/linker/regulatory/restriction sites. During the post-RL
    amber refinement, residues in coding regions are pinned (physical restraints
    pull them back to the original CG coordinates), while non-coding regions
    accept the RL optimization plus physical convergence.

    When the sequence has no case distinction (all upper or all lower), the
    whole sequence is treated according to default_coding. CircBase real samples
    are usually all lowercase, so default_coding=False makes everything
    non-coding (RL may optimize the whole sequence); engineered vectors carry
    case, so parse directly by letter case.

    Args:
        sequence: the sequence (mixed case or plain letters)
        default_coding: default used when the sequence has no case distinction
            False (default) = everything is non-coding by default

    Returns:
        np.ndarray[bool], shape (L,). True = residue in a coding region.
    """
    mask = np.zeros(len(sequence), dtype=bool)
    # Detect whether the sequence has a case distinction
    has_upper = any(c.isupper() for c in sequence)
    has_lower = any(c.islower() for c in sequence)
    no_case_distinction = not (has_upper and has_lower)

    if no_case_distinction:
        # No case distinction: apply default to the whole sequence
        mask[:] = default_coding
        return mask

    # Case distinction present: parse by letter case
    for i, c in enumerate(sequence):
        if c.isalpha() and c.isupper():
            mask[i] = True
        elif c.isalpha() and c.islower():
            mask[i] = False
    return mask


# ---------- Complementarity scan ----------
def complementarity_scan(
    sequence: str,
    window: int = W,
    wc_rate: float = WC_RATE,
    dg_threshold: float = DG_THRESHOLD,
    min_gap: int = 10,
) -> List[Tuple[int, int, float]]:
    """Sliding-window complementarity scan (scheme H: k-mer indexing, O(L^2) -> O(L*4^W)).

    The old O(L^2) double loop runs 4.5M pure-Python iterations at L=3000. The
    new version pre-indexes the reverse-complement k-mer of every window, uses
    the k-mer to look up candidates in reverse, and only validates dg on those
    candidates - complexity O(L*4^W).

    Args:
        sequence: ACGU string (circRNA, circular)
        window: sliding-window length (default 6)
        wc_rate: WC pairing-rate threshold (allows G.U wobble)
        dg_threshold: NN free-energy threshold; only DG below this counts as
            stable (kcal/mol)
        min_gap: skip ring distances below this (avoids self-pairing / adjacent)

    Returns:
        [(i, j, dg), ...] window-start position pairs + free energy.
        i, j are window starts (0-based), meaning seq[i:i+W] and seq[j:j+W] are
        antiparallel complements.
    """
    L = len(sequence)
    if L < window * 2 + min_gap:
        return []

    # Circular-sequence buffer (used when a window crosses the end)
    seq_ext = sequence + sequence[:window - 1]

    # Pre-index: k-mer -> [list of start positions]
    kmer_idx: Dict[str, List[int]] = {}
    for i in range(L):
        kmer = seq_ext[i:i + window]
        if len(kmer) == window:
            kmer_idx.setdefault(kmer, []).append(i)

    # Reverse mapping: for each k-mer, find its reverse complement
    # Only match low-frequency k-mers (scheme H: avoids repetitive sequences
    # such as poly-G exploding into mutual matches)
    rc_map: Dict[str, str] = {}
    for kmer, positions in kmer_idx.items():
        if len(positions) > MAX_KMER_FREQ:
            continue
        rc = _reverse_complement(kmer)
        if rc in kmer_idx and kmer != rc and len(kmer_idx[rc]) <= MAX_KMER_FREQ:
            rc_map[kmer] = rc

    pairs: List[Tuple[int, int, float]] = []
    seen: Set[Tuple[int, int]] = set()

    for kmer_a, positions in kmer_idx.items():
        kmer_b_rc = rc_map.get(kmer_a)
        if kmer_b_rc is None:
            continue
        positions_b = kmer_idx[kmer_b_rc]
        for i in positions:
            for j in positions_b:
                ring_dist = min(abs(i - j), L - abs(i - j))
                if ring_dist < min_gap:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                # Antiparallel: win_b = seq[j:j+W] pairs against the reverse of win_a
                win_a = seq_ext[i:i + window]
                win_b = seq_ext[j:j + window]
                # Full-match validation (k-mer RC already guarantees all-WC, but the
                # antiparallel orientation still needs confirming)
                wc = _wc_count(win_a, win_b)
                if wc / window < wc_rate:
                    continue
                # Energy filter
                dg = _nn_free_energy(win_a, win_b)
                if dg >= dg_threshold:
                    continue
                pairs.append((i, j, dg))

    return pairs


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of seq (ACGU)."""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'U': 'A', 'A': 'U', 'N': 'N'}
    return "".join(comp.get(b, 'N') for b in reversed(seq))


# ---------- Pairing-graph construction ----------
def build_pair_graph(
    sequence: str,
    vienna_pairs: List[Tuple[int, int, float]],
    scan_pairs: Optional[List[Tuple[int, int, float]]] = None,
) -> Dict[int, List[int]]:
    """Build the pairing-graph adjacency list.

    Nodes = residues 0..L-1.
    Edges = backbone adjacency (i, (i+1) mod L) + ViennaRNA pairs + complementarity-scan additions.

    Returns:
        adj: {node: [neighbor, ...]} undirected graph adjacency list.
    """
    L = len(sequence)
    adj: Dict[int, List[int]] = {i: [] for i in range(L)}

    # Backbone adjacency edges (including the (L-1, 0) BSJ closure)
    for i in range(L):
        nxt = (i + 1) % L
        adj[i].append(nxt)
        adj[nxt].append(i)

    # ViennaRNA pairing edges
    for (i, j, _w) in vienna_pairs:
        if 0 <= i < L and 0 <= j < L and i != j:
            adj[i].append(j)
            adj[j].append(i)

    # Complementarity-scan additions (scan returns window starts; expand them into per-residue pairs)
    # Scheme E: quality gate - expand only windows with >= 4/6 WC pairs in a row
    if scan_pairs:
        for (i0, j0, _dg) in scan_pairs:
            # Quality gate: require >= MIN_WC_IN_WIN consecutive WC pairs for a genuine far stem
            seq = sequence
            win_a = seq[i0:i0 + W] if len(seq[i0:i0 + W]) == W else (seq + seq[:W - 1])[i0:i0 + W]
            win_b = seq[j0:j0 + W] if len(seq[j0:j0 + W]) == W else (seq + seq[:W - 1])[j0:j0 + W]
            wc_hits = sum(
                _is_complementary(win_a[k], win_b[W - 1 - k])
                for k in range(W)
            )
            if wc_hits < MIN_WC_IN_WIN:
                continue
            for k in range(W):
                ik = (i0 + k) % L
                jk = (j0 + W - 1 - k) % L
                if ik != jk:
                    adj[ik].append(jk)
                    adj[jk].append(ik)

    # Deduplicate (the same pair may have been added multiple times)
    for v in adj:
        adj[v] = list(set(adj[v]))

    return adj


# ---------- BFS topological distance ----------
def topological_distance(
    adj: Dict[int, List[int]], i: int, j: int,
    *,
    exclude_edge: Optional[Tuple[int, int]] = None,
) -> int:
    """Compute the graph distance dist(i, j) by BFS. Unit edge weights.

    exclude_edge: if given (a, b), the BFS skips the a-b edge (removing a pair's own edge).
    Used for the "graph distance without the pair's own edge": measures whether
    this pair can be connected quickly through other pairs.

    circRNA ring: on L=2013, (5, 2010) is reached in 8 backbone steps, dist=8
    (near); an HA inverted repeat is topologically far, so dist is large.
    """
    if i == j:
        return 0
    L = len(adj)
    ea, eb = (None, None)
    if exclude_edge is not None:
        ea, eb = exclude_edge
    visited = {i}
    queue = deque([(i, 0)])
    while queue:
        node, d = queue.popleft()
        for nb in adj.get(node, []):
            # Skip the edge to exclude (in both directions)
            if exclude_edge is not None:
                if (node == ea and nb == eb) or (node == eb and nb == ea):
                    continue
            if nb == j:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, d + 1))
        if len(visited) >= L:
            break
    return -1  # unreachable (unexpected)


def ring_distance(i: int, j: int, L: int) -> int:
    """Ring distance = min(|i-j|, L-|i-j|). Pure-backbone shortest path, no pairing edges.

    Correct across the BSJ: for L=2013, (5, 2010) has ring distance min(2005, 8) = 8.
    """
    return min(abs(i - j), L - abs(i - j))


def far_end_pairs(
    adj: Dict[int, List[int]],
    vienna_pairs: List[Tuple[int, int, float]],
    scan_pairs: Optional[List[Tuple[int, int, float]]] = None,
    far_dist: int = FAR_DIST,
) -> List[Tuple[int, int]]:
    """Flag far/long-range pairs = far in ring distance AND topologically isolated.

    Both conditions must hold for a pair to be far (conditions 1 and 2 are not
    contradictory; they are complementary):
      - ring distance = min(|i-j|, L-|i-j|) > far_dist  (far along the backbone)
      - graph distance without the pair's own edge > far_dist  (not quickly
        connected through other pairs)

    Merges all pairs from ViennaRNA + the scan and evaluates each pair.

    Degraded fallback (added 2026-07-22): if the strong test above yields
    nothing, the pairing graph has been woven into a small world by scan false
    positives (measured: on circBase 4000nt+ the topological distance maxes at
    4-5, so everything is judged near). The topological-isolation test is then
    unreliable, so degrade to "far in ring distance AND pair comes from a true
    ViennaRNA pair":
      ring_dist > max(far_dist, L//4)  AND  (i,j) in vienna_pairs
    Only true ViennaRNA pairs are trusted, avoiding scan false-positive
    contamination. Still returns [] when there is no ViennaRNA input.
    """
    L = len(adj)
    all_pairs: Set[Tuple[int, int]] = set()
    for (i, j, _w) in vienna_pairs:
        all_pairs.add((min(i, j), max(i, j)))
    if scan_pairs:
        for (i0, j0, _dg) in scan_pairs:
            for k in range(W):
                ik = (i0 + k) % L
                jk = (j0 + W - 1 - k) % L
                if ik != jk:
                    all_pairs.add((min(ik, jk), max(ik, jk)))

    # Strong test: far in ring distance AND topologically isolated
    far = []
    for (i, j) in all_pairs:
        # Condition 1: far in ring distance
        if ring_distance(i, j, L) <= far_dist:
            continue
        # Condition 2: graph distance without the pair's own edge is large (topologically isolated)
        d = topological_distance(adj, i, j, exclude_edge=(i, j))
        if d > far_dist:
            far.append((i, j))

    if far:
        return far

    # Degrade: the topological test is unreliable (densely connected graph); use
    # ring distance + true ViennaRNA pairs instead
    vienna_set: Set[Tuple[int, int]] = {
        (min(i, j), max(i, j)) for (i, j, _w) in vienna_pairs
    }
    if not vienna_set:
        return []
    ring_thresh = max(far_dist, L // 4)
    far_fallback = []
    for (i, j) in vienna_set:
        if ring_distance(i, j, L) > ring_thresh:
            far_fallback.append((i, j))
    return far_fallback


# ---------- Stem-block extraction ----------
def extract_stem_blocks(
    vienna_pairs: List[Tuple[int, int, float]],
    scan_pairs: Optional[List[Tuple[int, int, float]]] = None,
    min_stem: int = MIN_STEM,
) -> List[List[Tuple[int, int]]]:
    """Extract stem blocks (runs of >= min_stem consecutive pairs).

    The scan returns window pairs; after expanding them into per-residue pairs,
    cluster them by positional continuity. A stem block is a run of consecutive
    i residues pairing with a run of consecutive j residues (antiparallel).

    Returns:
        [[(i, j), ...], ...] per-residue pair lists, one per stem block.
    """
    # Merge all pairs (expand scan windows)
    pair_set: Set[Tuple[int, int]] = set()
    for (i, j, _w) in vienna_pairs:
        pair_set.add((min(i, j), max(i, j)))
    if scan_pairs:
        # Expand scan-window pairs into per-residue pairs (simplified here: use only the window representative pair)
        for (i0, j0, _dg) in scan_pairs:
            pair_set.add((min(i0, j0), max(i0, j0)))

    # Sort by i and find runs of consecutive i + consecutive j
    sorted_pairs = sorted(pair_set)
    blocks: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []

    for p in sorted_pairs:
        if not current:
            current = [p]
            continue
        prev = current[-1]
        # Consecutive: i increases by 1 and j decreases by 1 (antiparallel), or j
        # increases by 1 (parallel, rare)
        if (p[0] == prev[0] + 1 and p[1] == prev[1] - 1) or \
           (p[0] == prev[0] + 1 and p[1] == prev[1] + 1):
            current.append(p)
        else:
            if len(current) >= min_stem:
                blocks.append(current)
            current = [p]
    if len(current) >= min_stem:
        blocks.append(current)
    return blocks


# ---------- Pseudoknot detection ----------

def detect_pseudoknots_from_bpp(
    sequence: str,
    pp_matrix: np.ndarray,
    existing_pairs: List[Tuple[int, int]],
    *,
    pk_threshold: float = 0.1,
    min_confidence: float = 0.3,
    is_circular: bool = True,
) -> List[Tuple[int, int, float]]:
    """Detect pseudoknot candidates (crossing pairs) from the BPP matrix.

    A pseudoknot is defined as two pairs (i,j) and (k,l) that cross linearly in
    the sequence (i<k<j<l), or that cross on the circular topology (circular
    crossing detection).

    Especially important for circRNA: the circular topology naturally creates
    pseudoknots across the BSJ.

    Args:
        sequence: RNA sequence
        pp_matrix: (L, L) BPP probability matrix
        existing_pairs: already-known pairs (to avoid duplicates)
        pk_threshold: minimum BPP probability threshold
        min_confidence: pseudoknot confidence threshold
        is_circular: whether the topology is circular

    Returns:
        [(i, j, confidence), ...] pseudoknot candidate pairs
    """
    L = len(sequence)
    wc = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}

    # Collect all high-probability pairs
    all_pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            p = pp_matrix[i, j] if pp_matrix.shape == (L, L) else 0
            if p >= pk_threshold:
                all_pairs.append((i, j, float(p)))

    # Set of existing pairs (for exclusion; accepts 2- and 3-tuples)
    existing_set = set()
    for p in existing_pairs:
        i, j = p[0], p[1]
        existing_set.add((min(i, j), max(i, j)))

    # Detect crossings (pseudoknots)
    pk_candidates = []
    n = len(all_pairs)
    for a in range(n):
        i, j, pi = all_pairs[a]
        for b in range(a + 1, n):
            k, l, pk = all_pairs[b]
            if k == i or k == j or l == i or l == j:
                continue  # shared residue, not a pseudoknot

            # Linear crossing: i<k<j<l or k<i<l<j
            linear_cross = (i < k < j < l) or (k < i < l < j)

            # Circular crossing: the two pairs cross on the circular topology
            circ_cross = False
            if is_circular and not linear_cross:
                # On the ring, two pairs cross when:
                # ordering (i,j) and (k,l) around the circle, they interleave
                pos = sorted([i, j, k, l])
                # Check for interleaving: i,k,j,l or i,l,j,k, etc.
                order = [0] * 4
                for idx, p in enumerate([i, j, k, l]):
                    order[pos.index(p)] = idx
                # Crossing: 2 or 3 sits between 0 and 1
                circ_cross = (order[0] < order[2] < order[1]) or \
                             (order[0] < order[3] < order[1]) or \
                             (order[2] < order[0] < order[3]) or \
                             (order[2] < order[1] < order[3])

            if linear_cross or circ_cross:
                # Check base complementarity
                b1, b2 = sequence[i], sequence[j]
                is_wc_pair = (b1, b2) in wc

                # Pseudoknot confidence: BPP probability x complementarity weight
                if is_wc_pair:
                    confidence = min(pi, pk) * 1.0
                else:
                    confidence = min(pi, pk) * 0.7  # non-WC lowers the confidence

                if confidence >= min_confidence:
                    # Avoid duplicating existing pairs
                    pair_key = (min(i, j), max(i, j))
                    if pair_key not in existing_set:
                        pk_candidates.append((i, j, confidence))
                        existing_set.add(pair_key)

    # Sort by confidence
    pk_candidates.sort(key=lambda x: -x[2])
    return pk_candidates


# ---------- End-to-end entry point ----------
def build_full_pair_graph(
    sequence: str,
    vienna_pairs: List[Tuple[int, int, float]],
    *,
    do_scan: bool = True,
    window: int = W,
    wc_rate: float = WC_RATE,
    dg_threshold: float = DG_THRESHOLD,
) -> Tuple[Dict[int, List[int]], List[Tuple[int, int, float]], List[Tuple[int, int]]]:
    """End to end: sequence + ViennaRNA pairs -> pairing graph + scan additions + far/long-range pair list.

    Returns:
        adj: pairing-graph adjacency list
        scan_pairs: complementarity-scan pairs [(i, j, dg), ...]
        far_pairs: far/long-range pairs [(i, j), ...] (topological distance > FAR_DIST)
    """
    scan = complementarity_scan(sequence, window, wc_rate, dg_threshold) if do_scan else []
    adj = build_pair_graph(sequence, vienna_pairs, scan)
    far = far_end_pairs(adj, vienna_pairs, scan)
    return adj, scan, far


if __name__ == "__main__":
    # Self-test: a synthetic sequence containing a known inverted repeat
    # Construction: an inverted repeat of poly-A plus a random segment
    import random
    random.seed(42)

    # stem1: 5'-AUGCAUGC-3' / 3'-UACGUACG-5' (fully complementary, inverted repeat)
    stem = "AUGCAUGC"
    complement = stem[::-1].translate(str.maketrans("AUGC", "UACG"))
    # Sequence = stem + linker + complement (an inverted repeat, expected to be caught by the scan)
    linker = "AAAA" * 5  # 20nt poly-A linker
    seq = stem + linker + complement + linker + stem + linker + complement

    print(f"sequence length: {len(seq)}")
    print(f"stem: {stem}")
    print(f"complement (reverse): {complement}")

    # Fake ViennaRNA pairs (empty, simulating that the inverted repeat is missed)
    vienna_pairs: List[Tuple[int, int, float]] = []
    scan = complementarity_scan(seq)
    print(f"\ncomplementarity-scan hits: {len(scan)} pairs")
    for (i, j, dg) in scan[:10]:
        print(f"  ({i}, {j}) ΔG={dg:.2f}")

    adj = build_pair_graph(seq, vienna_pairs, scan)
    print(f"\ngraph nodes: {len(adj)}")
    print(f"average degree: {sum(len(v) for v in adj.values())/len(adj):.2f}")

    far = far_end_pairs(adj, vienna_pairs, scan)
    print(f"\nfar/long-range pairs (dist>{FAR_DIST}): {len(far)}")
    for (i, j) in far[:5]:
        print(f"  ({i}, {j})")


def vienna_pair_probs(
    sequence: str,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], List[Tuple[int, int, float]], float]:
    """ViennaRNA partition-function pair probabilities + MFE.

    Args:
        sequence: RNA sequence
        threshold: pair-probability threshold (for filtering pairs_pf)

    Returns:
        (bpp_matrix, pairs_pf, pairs_mfe, pf_energy)
        - bpp_matrix: (L, L) pair-probability matrix
        - pairs_pf: [(i, j, p)] PF pair list
        - pairs_mfe: [(i, j)] MFE pair list
        - pf_energy: PF free energy (kcal/mol)
    """
    import RNA

    L = len(sequence)

    # Partition Function
    fc = RNA.fold_compound(sequence)
    fc.pf()
    bpp = np.zeros((L, L), dtype=np.float64)

    # Extract the pair-probability matrix
    # RNA.bpp() returns an (L+1) x (L+1) matrix (1-indexed)
    bpp_raw = np.array(fc.bpp())
    if bpp_raw.shape[0] > L:
        bpp = bpp_raw[1:, 1:]  # drop the 0-indexed row and column
    else:
        bpp = bpp_raw

    # PF pair list
    pairs_pf = []
    for i in range(L):
        for j in range(i + 1, L):
            p = float(bpp[i, j])
            if p > threshold:
                pairs_pf.append((i, j, p))

    # MFE
    ss_mfe, mfe_energy = fc.mfe()
    pairs_mfe = []
    stack = []
    for i, ch in enumerate(ss_mfe):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            j = stack.pop()
            pairs_mfe.append((j, i))

    # PF energy
    pf_energy = mfe_energy  # fc.mfe() returns (ss, energy)

    return bpp, pairs_pf, pairs_mfe, pf_energy
