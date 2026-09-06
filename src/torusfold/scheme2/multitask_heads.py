"""
multitask_heads.py — circRNA multi-task prediction heads (inspired by structRFM).

Four prediction heads share one encoder:
  1. Per-position SS head: an independent per-base classifier (A/U/G/C/N -> paired/unpaired)
  2. Pair prediction head: whether i is base-paired with j
  3. BSJ closure head: whether circularization is productive
  4. Clash prediction head: per-position clash probability

Inspiration: structRFM (Zhai et al. 2024, Nature Communications)
  - per-nucleotide independent classifiers
  - structural condition input (embedding_struct)
  - multi-task loss (MLM + SS + NSP)
  - weight mask for overlap regions
"""
from __future__ import annotations

import os

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Nucleotide encoding ──
BASE_TO_IDX = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
BASE_TYPES = ["A", "U", "G", "C", "N"]


class StructRFMEncoder(nn.Module):
    """Wrapper around a pretrained structRFM encoder.

    Returns 768-dim per-nucleotide features.
    max_length=514; long sequences are chunked automatically.
    """

    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self.feature_dim = 768
        self.max_length = 512  # structRFM length cap

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        path = self.model_path or os.environ.get("TF_STRUCTRFM_MODEL", "")
        if not path:
            raise ValueError(
                "A pretrained structRFM model path is required "
                "(pass model_path=... or set TF_STRUCTRFM_MODEL).")
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(path, trust_remote_code=True)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, sequence: str) -> torch.Tensor:
        """Encode a sequence -> (L, 768) per-nucleotide features.

        Long sequences are chunked, max_length-2 tokens per chunk.
        """
        self._load()
        L = len(sequence)
        all_features = []

        for start in range(0, L, self.max_length - 2):
            chunk = sequence[start:start + self.max_length - 2]
            inputs = self._tokenizer(chunk, return_tensors="pt", truncation=True)
            outputs = self._model(**inputs)
            feat = outputs.last_hidden_state[0]  # (chunk_len, 768)
            all_features.append(feat)

        return torch.cat(all_features, dim=0)[:L]  # (L, 768)


class StructConditionEncoder(nn.Module):
    """Encode known structural information into a conditioning vector.

    Mirrors structRFM's embedding_struct:
      struct_input = self.embedding_struct(struct.unsqueeze(-1))
      final_input = torch.cat([mapping_final_input, struct_input], dim=-1)

    Input: per-residue structure signals (3-dim):
      [is_paired, distance_to_partner/100, local_clash_density]
    Output: (L, hidden_dim) conditioning tensor
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, known_structure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            known_structure: (L, 3) per-residue structure signals
        Returns:
            (L, hidden_dim) conditioning tensor
        """
        return self.encoder(known_structure)


class CircRNAPredictionHeads(nn.Module):
    """circRNA multi-task prediction heads.

    Four tasks:
      1. SS: per-position paired/unpaired (per-base-type classifiers)
      2. Pair: i-j base-pair probability
      3. BSJ: back-splice closure probability
      4. Clash: per-position clash probability

    Each head is controlled by an enable_* flag; all enabled by default.
    Backward compatible: does not affect the existing pipeline (when
    use_multi_task_heads=False).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_base_types: int = 5,
        feature_dim: int = 20,
        struct_cond_dim: int = 32,
        dropout: float = 0.1,
        enable_ss_head: bool = True,
        enable_pair_head: bool = True,
        enable_bsj_head: bool = True,
        enable_clash_head: bool = True,
        use_structrfm: bool = False,
        rfm_model_path: str = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_ss_head = enable_ss_head
        self.enable_pair_head = enable_pair_head
        self.enable_bsj_head = enable_bsj_head
        self.enable_clash_head = enable_clash_head
        self.use_structrfm = use_structrfm

        # ── structRFM encoder (optional, frozen) ──
        self.rfm_encoder = None
        if use_structrfm:
            self.rfm_encoder = StructRFMEncoder(model_path=rfm_model_path)
            feature_dim = 768  # structRFM output dim

        # ── Shared encoder: feature_dim + struct_cond_dim -> hidden_dim ──
        input_dim = feature_dim + struct_cond_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── Structure-condition encoder ──
        self.struct_encoder = StructConditionEncoder(
            input_dim=3, hidden_dim=struct_cond_dim
        )

        # ── SS head: per-base-type classifiers (structRFM core pattern) ──
        if enable_ss_head:
            self.ss_classifiers = nn.ModuleDict({
                base: nn.Linear(hidden_dim, 2)  # paired / unpaired
                for base in BASE_TYPES
            })

        # ── Pair head: binary classification of whether (i, j) pair ──
        if enable_pair_head:
            self.pair_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        # ── BSJ head: back-splice closure probability ──
        if enable_bsj_head:
            self.bsj_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # ── Clash head: per-position clash probability ──
        if enable_clash_head:
            self.clash_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def encode_sequence(
        self,
        sequence: str,
        bpp_matrix: Optional[torch.Tensor] = None,
        struct_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a sequence + structure condition -> shared representation.

        Two modes:
          use_structrfm=True:  768-dim structRFM features
          use_structrfm=False: built-in features (one-hot + bpp + distance stats)

        Args:
            sequence: RNA sequence
            bpp_matrix: (L, L) optional ViennaRNA base-pair probability matrix
            struct_condition: (L, struct_cond_dim) structure condition; zeros when None

        Returns:
            (L, hidden_dim) shared encoding
        """
        L = len(sequence)
        device = struct_condition.device if struct_condition is not None else torch.device("cpu")

        if self.use_structrfm and self.rfm_encoder is not None:
            # structRFM mode: 768-dim pretrained features
            feat = self.rfm_encoder(sequence)  # (L, 768)
        else:
            # built-in feature mode: one-hot(5) + pair_context(8) + ss_signal(4) + local(3) = 20 dims
            feat = self._build_own_features(sequence, bpp_matrix, device)

        if struct_condition is None:
            struct_condition = torch.zeros(L, self.shared_encoder[0].in_features - feat.shape[-1],
                                           device=device)

        combined = torch.cat([feat, struct_condition], dim=-1)
        return self.shared_encoder(combined)

    def _build_own_features(
        self,
        sequence: str,
        bpp_matrix: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Build the built-in features (20 dims per nucleotide).

        Feature layout:
          [0:5]   one-hot encoding (A/U/G/C/N)
          [5:13]  pair context: max_bpp, n_partners, mean_dist, ...
          [13:17] ss_signal: is_stem, is_loop, is_hairpin, is_junction
          [17:20] local: gc_content_local, paired_density, entropy
        """
        L = len(sequence)
        feat = torch.zeros(L, 20, device=device)

        # one-hot
        base_idx = {"A": 0, "U": 1, "G": 2, "C": 3}
        for i, ch in enumerate(sequence.upper()):
            if ch in base_idx:
                feat[i, base_idx[ch]] = 1.0
            else:
                feat[i, 4] = 1.0

        # pair context (requires bpp_matrix)
        if bpp_matrix is not None and bpp_matrix.shape == (L, L):
            for i in range(L):
                row = bpp_matrix[i]
                feat[i, 5] = float(row.max())              # max_bpp
                feat[i, 6] = float((row > 0.1).sum())     # n_partners
                partners = torch.where(row > 0.1)[0]
                if len(partners) > 0:
                    dists = (partners.float() - i).abs()
                    feat[i, 7] = float(dists.mean()) / L   # mean_dist (normalized)
                    feat[i, 8] = float(dists.min()) / L    # min_dist
                # bpp entropy
                p = row[row > 0.01]
                if len(p) > 0:
                    p = p / p.sum()
                    feat[i, 9] = float(-(p * torch.log(p + 1e-8)).sum())
                # stem-region signal: paired on both flanks (5 positions each side)
                for offset in range(1, 6):
                    if i - offset >= 0 and i + offset < L:
                        if row[i - offset] > 0.1 or bpp_matrix[i - offset, i] > 0.1:
                            feat[i, 9 + min(offset, 4)] = 1.0

        # ss_signal (inferred from bpp)
        if bpp_matrix is not None:
            for i in range(L):
                paired = bpp_matrix[i].max() > 0.3
                feat[i, 13] = 1.0 if paired else 0.0          # is_stem
                feat[i, 14] = 1.0 if not paired else 0.0      # is_loop
                # hairpin: paired on both flanks but not itself
                left_paired = any(bpp_matrix[i, j] > 0.1 for j in range(max(0, i-5), i))
                right_paired = any(bpp_matrix[i, j] > 0.1 for j in range(i+1, min(L, i+6)))
                feat[i, 15] = 1.0 if (not paired and left_paired and right_paired) else 0.0
                # junction: abrupt change in pairing density
                local_density = bpp_matrix[max(0,i-3):i+4, :].mean()
                feat[i, 16] = float(local_density)

        # local stats
        seq_upper = sequence.upper()
        for i in range(L):
            window = seq_upper[max(0, i-5):i+6]
            gc = sum(1 for c in window if c in "GC") / max(len(window), 1)
            feat[i, 17] = gc                                        # local GC
            feat[i, 18] = float((bpp_matrix[i] > 0.1).sum()) / L   # paired density
            feat[i, 19] = float(i) / L                              # position (normalized)

        return feat

    def predict_ss(
        self,
        encoded: torch.Tensor,
        sequence: str,
    ) -> torch.Tensor:
        """SS prediction: classify each position with its base's classifier.

        Args:
            encoded: (L, hidden_dim) shared encoding
            sequence: RNA sequence

        Returns:
            (L, 2) SS logits (paired/unpaired)
        """
        L = len(sequence)
        logits = torch.zeros(L, 2, device=encoded.device)
        for i, base in enumerate(sequence):
            base_key = base.upper()
            if base_key not in self.ss_classifiers:
                base_key = "N"
            logits[i] = self.ss_classifiers[base_key](encoded[i])
        return logits

    def predict_pairs(
        self,
        encoded: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Pair prediction: whether (i, j) forms a base pair.

        Args:
            encoded: (L, hidden_dim) shared encoding
            pair_indices: (N, 2) candidate pair positions

        Returns:
            (N,) base-pair probabilities
        """
        if len(pair_indices) == 0:
            return torch.tensor([], device=encoded.device)

        i_emb = encoded[pair_indices[:, 0]]  # (N, hidden_dim)
        j_emb = encoded[pair_indices[:, 1]]  # (N, hidden_dim)
        pair_input = torch.cat([i_emb, j_emb], dim=-1)  # (N, 2*hidden_dim)
        return self.pair_head(pair_input).squeeze(-1)

    def predict_bsj(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        """BSJ closure prediction.

        Args:
            encoded: (L, hidden_dim) shared encoding

        Returns:
            () scalar, BSJ closure probability
        """
        # Mean representation of the first and last positions
        pooled = (encoded[0] + encoded[-1]) / 2  # (hidden_dim,)
        return self.bsj_head(pooled.unsqueeze(0)).squeeze()

    def predict_clash(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position clash prediction.

        Args:
            encoded: (L, hidden_dim) shared encoding

        Returns:
            (L,) per-position clash probabilities
        """
        return self.clash_head(encoded).squeeze(-1)

    def forward(
        self,
        sequence: str,
        struct_condition: Optional[torch.Tensor] = None,
        pair_indices: Optional[torch.Tensor] = None,
        bpp_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequence: RNA sequence
            struct_condition: (L, 3) structure signals [is_paired, dist/100, clash_density]
            pair_indices: (N, 2) candidate pairs
            bpp_matrix: (L, L) ViennaRNA bpp matrix (required for built-in feature mode)

        Returns:
            dict with keys:
              ss_logits: (L, 2)
              pair_probs: (N,) or empty
              bsj_logit: scalar
              clash_scores: (L,)
        """
        # Encode the structure condition
        struct_cond = None
        if struct_condition is not None:
            struct_cond = self.struct_encoder(struct_condition)

        # Shared encoding (built-in 20-dim features or structRFM 768-dim + struct_cond)
        encoded = self.encode_sequence(sequence, bpp_matrix, struct_cond)

        result = {"encoded": encoded}

        if self.enable_ss_head:
            result["ss_logits"] = self.predict_ss(encoded, sequence)

        if self.enable_pair_head and pair_indices is not None:
            result["pair_probs"] = torch.sigmoid(
                self.predict_pairs(encoded, pair_indices)
            )

        if self.enable_bsj_head:
            result["bsj_logit"] = self.predict_bsj(encoded)

        if self.enable_clash_head:
            result["clash_scores"] = torch.sigmoid(
                self.predict_clash(encoded)
            )

        return result


# ── Convenience functions ──

def build_struct_condition_from_coords(
    coords: np.ndarray,
    pairs: List[Tuple[int, int]],
    L: int,
) -> np.ndarray:
    """Build a structure condition from coordinates (3 dims per residue).

    Args:
        coords: (L, 3) P coordinates
        pairs: [(i, j), ...] known base pairs
        L: sequence length

    Returns:
        (L, 3) [is_paired, dist_to_partner/100, local_clash_density]
    """
    condition = np.zeros((L, 3), dtype=np.float32)

    # is_paired
    paired_set = set()
    for (i, j) in pairs:
        if i < L and j < L:
            paired_set.add(i)
            paired_set.add(j)
    for i in paired_set:
        condition[i, 0] = 1.0

    # dist_to_partner / 100
    pair_map = {}
    for (i, j) in pairs:
        if i < L and j < L:
            pair_map[i] = j
            pair_map[j] = i
    for i, j in pair_map.items():
        if i < L and j < L:
            d = np.linalg.norm(coords[i] - coords[j])
            condition[i, 1] = d / 100.0
            condition[j, 1] = d / 100.0

    # local_clash_density (number of P-P neighbors < 3A / 10)
    for i in range(L):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        n_clash = np.sum((dists < 3.0) & (np.arange(L) != i))
        condition[i, 2] = min(n_clash / 10.0, 1.0)

    return condition


def build_overlap_weight_mask(
    segments: List[dict],
    full_length: int,
    decay: str = "linear",
    min_weight: float = 0.3,
) -> np.ndarray:
    """Build a weighting mask for chunk overlap regions.

    structRFM pattern: weight_mask = torch.where(input_ids == pad, 0.0, 1.0)
    Refined here: linear/cosine decay over the overlap region.

    Args:
        segments: [{start, end}, ...] chunk boundaries
        full_length: total sequence length
        decay: "linear" or "cosine"
        min_weight: minimum weight inside overlap regions

    Returns:
        (full_length,) float weight mask
    """
    weights = np.ones(full_length, dtype=np.float32)

    for k in range(len(segments) - 1):
        seg_i = segments[k]
        seg_j = segments[k + 1]
        overlap_start = seg_j.get("start", 0)
        overlap_end = seg_i.get("end", full_length)

        if overlap_start >= overlap_end:
            continue

        overlap_len = overlap_end - overlap_start
        for pos in range(overlap_start, overlap_end):
            t = (pos - overlap_start) / max(overlap_len - 1, 1)
            if decay == "cosine":
                w = min_weight + (1.0 - min_weight) * 0.5 * (1.0 + np.cos(np.pi * t))
            else:  # linear
                w = min_weight + (1.0 - min_weight) * t
            weights[pos] = min(weights[pos], w)

    return weights
