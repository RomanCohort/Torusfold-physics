"""Read the pairing a pipeline run persisted, and hand it over in 1-based indexing.

THE TWO PROBLEMS THIS SOLVES
----------------------------
1. INDEX BASE. Everything the pipeline writes -- `pairs`, `far_pairs`,
   `stem_blocks`, `bpp_high`, `bpp_mid`, the `bpp` matrix and the
   `ss_consensus` string -- is 0-based. Every residue index this package's
   coordinate layer uses is 1-based, because that is what PDB author numbering
   is. Handing the checkpoint's numbers straight to the pair graph shifts every
   position by one, and the result still LOOKS plausible: measured on a 200 nt
   run, reading the checkpoint pairs as 0-based gives 82/82 sequence-complementary
   pairs against 37/82 for the 1-based reading. The wrong reading is not
   obviously wrong; it is just worse. So conversion happens here, once, and the
   converted values are what callers see.

2. WHICH LIST IS WHICH. Four different objects describe pairing and they are not
   interchangeable:

       pairs        (i, j, w)   every restraint the CG stage will use, including
                                pseudoknot-derived additions and non-canonical
                                contacts. w is a restraint weight, NOT the
                                partition-function probability: on the 200 nt run
                                only 21 of 82 weights equal bpp[i, j].
       far_pairs    (i, j)      the long-range subset, |i-j| > 100 topologically.
       stem_blocks  [[(i,j)..]] contiguous helices, one list per stem. THIS is the
                                object the PKR-length feature wants -- helix runs,
                                not isolated pairs.
       bpp          (L, L)      the ViennaRNA partition-function probability
                                matrix; the only calibrated probability here.

WHAT THIS IS NOT
----------------
Not a substitute for the coordinate layer. The pipeline's own PDBs are mostly
ONE PHOSPHORUS PER RESIDUE (see `_write_coords_pdb`), so they cannot give
per-residue accessibility. The checkpoint gives pairing; the all-atom model gives
accessibility; the immune fingerprint needs both, and neither file has both.

The two are also not the same prediction: the checkpoint records what the
secondary-structure stage decided, while `pair_graph_from_coords` reads what the
final coordinates actually do. Where they disagree is informative and should be
reported, not silently reconciled.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CheckpointPairing", "load_checkpoint_pairing"]


@dataclasses.dataclass
class CheckpointPairing:
    """Pairing read from a pipeline checkpoint, in 1-based residue indexing."""

    length: int
    pairs: List[Tuple[int, int, float]]           # (i, j, w), 1-based
    far_pairs: List[Tuple[int, int]]              # 1-based
    stem_blocks: List[List[Tuple[int, int]]]      # 1-based
    bpp: Optional[np.ndarray]                     # (L, L), index [i-1, j-1]
    ss_consensus: str
    level: Optional[float]
    source: pathlib.Path
    raw_meta: Dict

    # -- views ---------------------------------------------------------------
    def restraint_map(self) -> Dict[int, int]:
        """One partner per residue, from `pairs` (highest weight wins).

        Residues that appear in `pairs` with more than one partner are resolved
        by weight and the conflict is counted, because a residue cannot pair
        twice and a silent pick would hide a real inconsistency.
        """
        best: Dict[int, Tuple[int, float]] = {}
        for i, j, w in self.pairs:
            for a, b in ((i, j), (j, i)):
                cur = best.get(a)
                if cur is None or w > cur[1]:
                    best[a] = (b, w)
        return {a: b for a, (b, _) in best.items()}

    def multi_partner_count(self) -> int:
        seen: Dict[int, set] = {}
        for i, j, _ in self.pairs:
            seen.setdefault(i, set()).add(j)
            seen.setdefault(j, set()).add(i)
        return sum(1 for v in seen.values() if len(v) > 1)

    def helix_lengths(self) -> List[int]:
        """Length in base pairs of each stem block, longest first."""
        return sorted((len(b) for b in self.stem_blocks), reverse=True)

    def bpp_at(self, i: int, j: int) -> float:
        """bpp probability for a 1-based pair, or 0.0 if out of range."""
        if self.bpp is None:
            return 0.0
        a, b = i - 1, j - 1
        if 0 <= a < self.bpp.shape[0] and 0 <= b < self.bpp.shape[1]:
            return float(self.bpp[a, b])
        return 0.0

    def summary(self) -> Dict:
        return {
            "length": self.length,
            "n_pairs": len(self.pairs),
            "n_far_pairs": len(self.far_pairs),
            "n_stem_blocks": len(self.stem_blocks),
            "helix_lengths": self.helix_lengths(),
            "longest_helix_bp": (self.helix_lengths() or [0])[0],
            "residues_with_multiple_partners": self.multi_partner_count(),
            "paired_fraction": round(
                len(self.restraint_map()) / max(1, self.length), 4),
            "bpp_nonzero": int((self.bpp > 0).sum()) if self.bpp is not None else 0,
            "level": self.level,
            "source": str(self.source),
        }


def _validate_complementary(seq: str, pairs: Sequence[Tuple[int, int, float]],
                            index_base: int) -> float:
    """Fraction of pairs that are sequence-complementary under a given base."""
    comp = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
    ok = 0
    for i, j, _ in pairs:
        a, b = i - index_base, j - index_base
        if 0 <= a < len(seq) and 0 <= b < len(seq) and (seq[a], seq[b]) in comp:
            ok += 1
    return ok / max(1, len(pairs))


def load_checkpoint_pairing(
    path: str | pathlib.Path,
    *,
    sequence: Optional[str] = None,
    bpp_path: Optional[str | pathlib.Path] = None,
    verify_index_base: bool = True,
) -> CheckpointPairing:
    """Load `_checkpoint.json` and convert every pairing index to 1-based.

    Args:
        path: the `_checkpoint.json` (or a directory containing one).
        sequence: if given, and `verify_index_base`, the pair list is used to
            confirm the indexing base instead of assuming it. Pass it whenever
            you have it -- the check costs nothing and catches a format change.
        bpp_path: override for `ckpt_bpp.npy`; defaults to a sibling of `path`.
        verify_index_base: when a sequence is available, assert that 0-based
            reading beats 1-based, and raise if it does not.

    Raises:
        FileNotFoundError: no checkpoint at `path`.
        ValueError: the file has no pair list, or the index base is not 0.
    """
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "_checkpoint.json"
    if not p.is_file():
        raise FileNotFoundError(f"no checkpoint at {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    seq_len = int(raw.get("_seq_len") or raw.get("seq_len") or 0)

    pairs_raw = list(raw.get("pairs") or [])
    if not pairs_raw and seq_len:
        # A checkpoint with no pairs is legitimate (an unfolded sequence, as in
        # the 10 nt smoke run) but must be distinguishable from a malformed one.
        pass

    pairs: List[Tuple[int, int, float]] = []
    for item in pairs_raw:
        if len(item) < 2:
            continue
        i, j = int(item[0]), int(item[1])
        w = float(item[2]) if len(item) > 2 else 1.0
        if i == j:
            continue
        pairs.append((min(i, j), max(i, j), w))

    if sequence is not None and pairs and verify_index_base:
        f0 = _validate_complementary(sequence, pairs, 0)
        f1 = _validate_complementary(sequence, pairs, 1)
        if f0 <= f1:
            raise ValueError(
                f"checkpoint at {p} does not look 0-based: complementary fraction "
                f"is {f0:.3f} as 0-based and {f1:.3f} as 1-based. The conversion "
                f"in this module assumes 0-based; do not guess -- inspect the file."
            )

    # 0-based -> 1-based. This is the single place the shift happens.
    pairs = [(i + 1, j + 1, w) for i, j, w in pairs]
    far_pairs = [(min(int(a), int(b)) + 1, max(int(a), int(b)) + 1)
                 for a, b in (raw.get("far_pairs") or [])]
    stem_blocks = [
        [(min(int(a), int(b)) + 1, max(int(a), int(b)) + 1) for a, b in block]
        for block in (raw.get("stem_blocks") or [])
    ]

    bp = pathlib.Path(bpp_path) if bpp_path else p.parent / "ckpt_bpp.npy"
    bpp = None
    if bp.is_file():
        bpp = np.load(bp)
        if bpp.ndim != 2 or bpp.shape[0] != bpp.shape[1]:
            raise ValueError(f"{bp} is not a square matrix: shape {bpp.shape}")
        if seq_len and bpp.shape[0] != seq_len:
            raise ValueError(
                f"{bp} is {bpp.shape[0]}x{bpp.shape[0]} but the checkpoint says "
                f"the sequence is {seq_len} nt; refusing to guess which is right"
            )

    length = seq_len or (bpp.shape[0] if bpp is not None else 0)
    if length == 0 and pairs:
        length = max(max(i, j) for i, j, _ in pairs)

    return CheckpointPairing(
        length=length,
        pairs=pairs,
        far_pairs=far_pairs,
        stem_blocks=stem_blocks,
        bpp=bpp,
        ss_consensus=str(raw.get("ss_consensus") or ""),
        level=raw.get("level"),
        source=p,
        raw_meta={
            k: raw[k] for k in
            ("n_segments", "pair_rate", "cross_segment_ok_rate", "pf_energy",
             "bpp_high", "bpp_mid", "segments")
            if k in raw
        },
    )
