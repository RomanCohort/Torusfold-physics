"""
multitask_loss.py — circRNA 多任务损失 (structRFM 启发)

Loss = w_ss * L_ss + w_pair * L_pair + w_bsj * L_bsj + w_clash * L_clash

structRFM pattern:
  - L_ss: 只在 SS 未知位置算 (struct == -1)
  - weight_mask: chunk 重叠区域降权
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircRNAMultiTaskLoss(nn.Module):
    """circRNA 多任务损失函数.

    四个子损失:
      1. SS loss: CrossEntropy, 只在已知 SS 位置计算
      2. Pair loss: BCE, 配对预测
      3. BSJ loss: BCE, 环化闭合预测
      4. Clash loss: BCE, 碰撞预测

    structRFM pattern: weight_mask 用于 chunk 重叠区域降权
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
        """计算多任务损失.

        Args:
            predictions: 模型输出 dict
              ss_logits: (L, 2) SS logits
              pair_probs: (N,) 配对概率
              bsj_logit: scalar BSJ 概率
              clash_scores: (L,) 碰撞概率
            labels: 标签 dict
              ss: (L,) SS 标签 (-1=未知, 0=unpaired, 1=paired)
              pair_labels: (N,) 配对标签 (0/1)
              bsj_label: scalar (0/1)
              clash_labels: (L,) 碰撞标签 (0/1)
            weight_mask: (L,) 可选, chunk 重叠区域降权

        Returns:
            dict with 'total_loss' + 每个子损失
        """
        losses = {}
        # 获取 device: 从 predictions 中任意 tensor
        _dev = torch.device("cpu")
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                _dev = v.device
                break
        total = torch.tensor(0.0, device=_dev)

        # ── SS Loss: 只在已知位置计算 (structRFM pattern) ──
        if "ss_logits" in predictions and "ss" in labels:
            ss_logits = predictions["ss_logits"]  # (L, 2)
            ss_labels = labels["ss"]  # (L,)

            # 只在非 -1 位置计算 (structRFM: struct == -1 的位置跳过)
            known_mask = ss_labels != self.ss_mask_value
            if known_mask.any():
                loss_ss = F.cross_entropy(
                    ss_logits[known_mask], ss_labels[known_mask].long()
                )
                if weight_mask is not None:
                    # 对已知位置用 weight_mask 加权
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
    """从 dot-bracket 字符串生成 SS 标签.

    Args:
        dotbracket: e.g. "(((...)))"
        unknown_value: 未知位置的值

    Returns:
        (L,) tensor: 1=paired, 0=unpaired, unknown_value=未知
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
    """从坐标计算碰撞标签.

    Args:
        coords: (L, 3) P 坐标
        threshold: 碰撞阈值 (A)

    Returns:
        (L,) float: 1.0 = 有碰撞, 0.0 = 无碰撞
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
    """生成配对预测的正负样本标签.

    Args:
        n_positions: 序列长度
        pairs: [(i, j), ...] 已知配对
        all_possible: 是否生成所有可能对 (O(L^2), 太大时不推荐)
        max_neg_ratio: 负样本/正样本最大比例

    Returns:
        (pair_indices (N, 2), pair_labels (N,))
    """
    pos_pairs = [(i, j) for (i, j) in pairs if i < n_positions and j < n_positions]

    if all_possible:
        # 生成所有 (i, j) 对
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

    # 负采样: 随机选不配对的位置对
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
