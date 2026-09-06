"""
multitask_loss.py — circRNA multi-task loss (structRFM-inspired)

Loss = w_ss * L_ss + w_pair * L_pair + w_bsj * L_bsj + w_clash * L_clash

structRFM pattern:
  - L_ss: only computed at positions where the SS is unknown (struct == -1)
  - weight_mask: down-weights overlapping chunk regions
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircRNAMultiTaskLoss(nn.Module):
    """circRNA multi-task loss function.

    Four sub-losses:
      1. SS loss: CrossEntropy, computed only at positions with known SS
      2. Pair loss: BCE, base-pairing prediction
      3. BSJ loss: BCE, back-spliced junction prediction
      4. Clash loss: BCE, clash prediction

    structRFM pattern: weight_mask down-weights overlapping chunk regions
    """

    def __init__(
        self,
        w_ss: float = 1.0,
        w_pair: float = 1.0,
        w_bsj: float = 0.5,
        w_clash: float = 0.3,
        ss_mask_value: float = -1.0,
    ):
        super().__init__()
        self.w_ss = w_ss
        self.w_pair = w_pair
        self.w_bsj = w_bsj
        self.w_clash = w_clash
        self.ss_mask_value = ss_mask_value

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        weight_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the multi-task loss.

        Args:
            predictions: dict of model outputs
              ss_logits: (L, 2) SS logits
              pair_probs: (N,) base-pairing probabilities
              bsj_logit: scalar BSJ probability
              clash_scores: (L,) clash probabilities
            labels: dict of labels
              ss: (L,) SS labels (-1=unknown, 0=unpaired, 1=paired)
              pair_labels: (N,) base-pairing labels (0/1)
              bsj_label: scalar (0/1)
              clash_labels: (L,) clash labels (0/1)
            weight_mask: (L,) optional, down-weights overlapping chunk regions

        Returns:
            dict with 'total_loss' plus each sub-loss
        """
        losses = {}
        # Determine the device from any tensor in predictions
        _dev = torch.device("cpu")
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                _dev = v.device
                break
        total = torch.tensor(0.0, device=_dev)

        # ── SS Loss: computed only at known positions (structRFM pattern) ──
        if "ss_logits" in predictions and "ss" in labels:
            ss_logits = predictions["ss_logits"]  # (L, 2)
            ss_labels = labels["ss"]  # (L,)

            # Only compute where the label is not -1 (structRFM: skip struct == -1)
            known_mask = ss_labels != self.ss_mask_value
            if known_mask.any():
                loss_ss = F.cross_entropy(
                    ss_logits[known_mask], ss_labels[known_mask].long()
                )
                if weight_mask is not None:
                    # Weight the known positions by weight_mask
                    w = weight_mask[known_mask]
                    loss_ss = (loss_ss * w).sum() / w.sum().clamp(min=1.0)
                losses["loss_ss"] = loss_ss
                total = total + self.w_ss * loss_ss
            else:
                losses["loss_ss"] = torch.tensor(0.0, device=ss_logits.device)

        # ── Pair Loss: BCE ──
        if "pair_probs" in predictions and "pair_labels" in labels:
            pair_probs = predictions["pair_probs"]  # (N,)
            pair_labels = labels["pair_labels"].float()  # (N,)

            if len(pair_probs) > 0:
                loss_pair = F.binary_cross_entropy(pair_probs, pair_labels)
                losses["loss_pair"] = loss_pair
                total = total + self.w_pair * loss_pair
            else:
                losses["loss_pair"] = torch.tensor(0.0, device=total.device)

        # ── BSJ Loss: BCE ──
        if "bsj_logit" in predictions and "bsj_label" in labels:
            bsj_prob = torch.sigmoid(predictions["bsj_logit"])
            bsj_label = labels["bsj_label"].float()
            loss_bsj = F.binary_cross_entropy(bsj_prob.unsqueeze(0), bsj_label.unsqueeze(0))
            losses["loss_bsj"] = loss_bsj
            total = total + self.w_bsj * loss_bsj

        # ── Clash Loss: BCE ──
        if "clash_scores" in predictions and "clash_labels" in labels:
            clash_scores = predictions["clash_scores"]  # (L,)
            clash_labels = labels["clash_labels"].float()  # (L,)

            loss_clash = F.binary_cross_entropy(clash_scores, clash_labels)
            if weight_mask is not None:
                w = weight_mask
                loss_clash = (loss_clash * w).sum() / w.sum().clamp(min=1.0)
            losses["loss_clash"] = loss_clash
            total = total + self.w_clash * loss_clash

        losses["total_loss"] = total
        return losses


def compute_ss_labels_from_dotbracket(
    dotbracket: str,
    unknown_value: float = -1.0,
) -> torch.Tensor:
    """Generate SS labels from a dot-bracket string.

    Args:
        dotbracket: e.g. "(((...)))"
        unknown_value: value used for unknown positions

    Returns:
        (L,) tensor: 1=paired, 0=unpaired, unknown_value=unknown
    """
    labels = []
    for ch in dotbracket:
        if ch in "()" or ch in "[]{}":
            labels.append(1.0)
        elif ch == ".":
            labels.append(0.0)
        else:
            labels.append(unknown_value)
    return torch.tensor(labels, dtype=torch.float32)


def compute_clash_labels_from_coords(
    coords: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """Compute clash labels from coordinates.

    Args:
        coords: (L, 3) P coordinates
        threshold: clash threshold (A)

    Returns:
        (L,) float: 1.0 = has clash, 0.0 = no clash
    """
    L = len(coords)
    labels = np.zeros(L, dtype=np.float32)
    for i in range(L):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        n_clash = np.sum((dists < threshold) & (np.arange(L) != i))
        if n_clash > 0:
            labels[i] = 1.0
    return labels


def compute_pair_labels(
    n_positions: int,
    pairs: list,
    all_possible: bool = False,
    max_neg_ratio: float = 3.0,
) -> tuple:
    """Generate positive/negative sample labels for base-pairing prediction.

    Args:
        n_positions: sequence length
        pairs: [(i, j), ...] known base pairs
        all_possible: whether to generate every possible pair (O(L^2); not recommended when large)
        max_neg_ratio: maximum ratio of negative to positive samples

    Returns:
        (pair_indices (N, 2), pair_labels (N,))
    """
    pos_pairs = [(i, j) for (i, j) in pairs if i < n_positions and j < n_positions]

    if all_possible:
        # Generate every (i, j) pair
        indices = []
        labels = []
        pair_set = set((min(i, j), max(i, j)) for i, j in pos_pairs)
        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                indices.append([i, j])
                labels.append(1 if (i, j) in pair_set else 0)
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )

    # Negative sampling: randomly choose position pairs that do not pair
    n_pos = len(pos_pairs)
    n_neg = min(int(n_pos * max_neg_ratio), n_positions * (n_positions - 1) // 2 - n_pos)

    import random
    neg_pairs = set()
    pair_set = set((min(i, j), max(i, j)) for i, j in pos_pairs)
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 10:
        i = random.randint(0, n_positions - 1)
        j = random.randint(0, n_positions - 1)
        if i != j:
            key = (min(i, j), max(i, j))
            if key not in pair_set and key not in neg_pairs:
                neg_pairs.add(key)
        attempts += 1

    all_indices = []
    all_labels = []
    for (i, j) in pos_pairs:
        all_indices.append([i, j])
        all_labels.append(1.0)
    for (i, j) in neg_pairs:
        all_indices.append([i, j])
        all_labels.append(0.0)

    return (
        torch.tensor(all_indices, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.float32),
    )
