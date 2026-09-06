"""
rcm.py — Reverse Complementary Match (RCM) 计算模块.

基于 CircCNNs (Wang & Liang, 2024) 的复数累计和快速算法,
用于检测 circRNA BSJ 侧翼内含子间的反向互补匹配.

核心思想:
  核苷酸→复数映射 (A=1, T=-1, C=1j, G=-1j)
  累计和向量快速求所有 kmer 的分数
  外积广播检测两条序列间的 RCM kmer 对

应用场景:
  - crossing RCM: BSJ 上游 vs 下游内含子 → 促进 back-splicing
  - within RCM: 单侧内含子内 → 促进 linear splicing (竞争)
  - RCM score 作为配对置信度权重注入管线

参考: Wang & Liang, Scientific Reports 14:18982 (2024)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


# ── 核苷酸→复数映射 ──
_BASE_MAP = {
    'A': 1.0,   'a': 1.0,
    'T': -1.0,  't': -1.0,  'U': -1.0, 'u': -1.0,
    'C': 1j,    'c': 1j,
    'G': -1j,   'g': -1j,
    'N': 0.0,   'n': 0.0,
}

# Watson-Crick 互补
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'U': 'A',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'u': 'a',
               'N': 'N', 'n': 'n'}


def _seq_to_complex(seq: str) -> np.ndarray:
    """序列→复数向量. O(L)."""
    return np.array([_BASE_MAP.get(c, 0.0) for c in seq], dtype=np.complex64)


def _kmer_scores(seq: str, k: int) -> np.ndarray:
    """计算序列中所有长度 k 的 kmer 的累计分数. O(L).

    利用累计和: score[i] = sum(seq[i:i+k]) 通过 cumsum 差分快速求得.
    返回 (L-k+1,) 复数向量, 每个元素是一个 kmer 的累计分数.
    """
    # 增广序列: 前面加 'N' (值为 0), 使得 cumsum[0] = 0
    aug = np.concatenate([[0.0 + 0j], _seq_to_complex(seq)])
    cum = aug.cumsum()
    # score[i] = cum[i+k] - cum[i], 即长度 k 的窗口和
    return cum[k:] - cum[:-k]


def _validate_rcm(seq1: str, seq2: str, i: int, j: int, k: int) -> int:
    """逐碱基验证位置 i 和 j 的 kmer 是否反向互补. 返回错配数."""
    mismatches = 0
    for t in range(k):
        if seq1[i + t] != _COMPLEMENT.get(seq2[j + k - 1 - t], 'N'):
            mismatches += 1
            if mismatches > 0:  # 一旦有错配就提前返回 (原版逻辑)
                return mismatches
    return mismatches


def rcm_crossing(
    seq_upstream: str,
    seq_downstream: str,
    k: int = 7,
    max_mismatch: int = 0,
) -> Tuple[int, np.ndarray]:
    """检测两条序列间的 RCM kmer 对 (crossing RCM).

    原版 CircCNNs 两步算法:
      1. 累计和外积快速预筛选 (|real|+|imag| ≤ threshold)
      2. 逐碱基验证 (精确检查反向互补)

    Args:
        seq_upstream: BSJ 上游内含子序列
        seq_downstream: BSJ 下游内含子序列
        k: kmer 长度
        max_mismatch: 允许的错配数 (0=完全互补)

    Returns:
        (n_rcm_pairs, distribution_5x5)
    """
    L1, L2 = len(seq_upstream), len(seq_downstream)
    if L1 < k or L2 < k:
        return 0, np.zeros((5, 5), dtype=np.float64)

    # Step 1: 累计和外积快速预筛选
    s1 = _kmer_scores(seq_upstream, k)
    s2 = _kmer_scores(seq_downstream, k)
    combo = s1.reshape(-1, 1) + s2.reshape(1, -1)
    score_mat = np.abs(combo.real) + np.abs(combo.imag)
    candidates = np.where(score_mat <= max_mismatch)

    # Step 2: 逐碱基验证
    valid_rows, valid_cols = [], []
    for r, c in zip(candidates[0], candidates[1]):
        if _validate_rcm(seq_upstream, seq_downstream, int(r), int(c), k) <= max_mismatch:
            valid_rows.append(r)
            valid_cols.append(c)

    n_pairs = len(valid_rows)

    # 5×5 分布矩阵
    dist = np.zeros((5, 5), dtype=np.float64)
    if n_pairs > 0:
        r_bins = np.clip((np.array(valid_rows) / max(1, len(s1) - 1) * 5).astype(int), 0, 4)
        c_bins = np.clip((np.array(valid_cols) / max(1, len(s2) - 1) * 5).astype(int), 0, 4)
        for rb, cb in zip(r_bins, c_bins):
            dist[rb, cb] += 1.0

    return n_pairs, dist


def rcm_within(
    seq: str,
    k: int = 7,
    max_mismatch: int = 0,
) -> Tuple[int, np.ndarray]:
    """检测单条序列内的 RCM kmer 对 (within RCM).

    原版 CircCNNs 两步算法 + 上三角去重.

    Args:
        seq: 内含子序列
        k: kmer 长度
        max_mismatch: 允许的错配数

    Returns:
        (n_rcm_pairs, distribution_5x5)
    """
    L = len(seq)
    if L < 2 * k:
        return 0, np.zeros((5, 5), dtype=np.float64)

    s = _kmer_scores(seq, k)

    # Step 1: 累计和外积预筛选 (只取上三角 i < j)
    combo = s.reshape(-1, 1) + s.reshape(1, -1)
    score_mat = np.abs(combo.real) + np.abs(combo.imag)
    candidates = np.where(np.triu(score_mat <= max_mismatch, k=1))

    # Step 2: 逐碱基验证
    valid_rows, valid_cols = [], []
    for r, c in zip(candidates[0], candidates[1]):
        if _validate_rcm(seq, seq, int(r), int(c), k) <= max_mismatch:
            valid_rows.append(r)
            valid_cols.append(c)

    n_pairs = len(valid_rows)

    dist = np.zeros((5, 5), dtype=np.float64)
    if n_pairs > 0:
        r_bins = np.clip((np.array(valid_rows) / max(1, len(s) - 1) * 5).astype(int), 0, 4)
        c_bins = np.clip((np.array(valid_cols) / max(1, len(s) - 1) * 5).astype(int), 0, 4)
        for rb, cb in zip(r_bins, c_bins):
            dist[rb, cb] += 1.0

    return n_pairs, dist


def compute_rcm_score(
    seq_upstream: str,
    seq_downstream: str,
    kmer_lengths: Optional[List[int]] = None,
    max_mismatch: int = 0,
) -> dict:
    """计算综合 RCM 得分: crossing + within(upstream) + within(downstream).

    对应论文中 RCM_triCNN 的输入特征.

    Args:
        seq_upstream: BSJ 上游内含子序列
        seq_downstream: BSJ 下游内含子序列
        kmer_lengths: k 值列表, 默认 [5, 7, 9, 11, 13]
        max_mismatch: 允许的错配数

    Returns:
        dict with keys:
            'crossing_total': int, crossing RCM kmer 对总数 (所有 k 之和)
            'within_up_total': int, upstream within RCM 总数
            'within_down_total': int, downstream within RCM 总数
            'crossing_dists': list of (k, 5x5 matrix), 各 k 的 crossing 分布
            'within_up_dists': list of (k, 5x5 matrix)
            'within_down_dists': list of (k, 5x5 matrix)
            'confidence': float, 综合置信度 [0, 1]
    """
    if kmer_lengths is None:
        kmer_lengths = [5, 7, 9, 11, 13]

    crossing_total = 0
    within_up_total = 0
    within_down_total = 0
    crossing_dists = []
    within_up_dists = []
    within_down_dists = []

    for k in kmer_lengths:
        n_cross, d_cross = rcm_crossing(seq_upstream, seq_downstream, k, max_mismatch)
        n_up, d_up = rcm_within(seq_upstream, k, max_mismatch)
        n_down, d_down = rcm_within(seq_downstream, k, max_mismatch)

        crossing_total += n_cross
        within_up_total += n_up
        within_down_total += n_down
        crossing_dists.append((k, d_cross))
        within_up_dists.append((k, d_up))
        within_down_dists.append((k, d_down))

    # 综合置信度: crossing 越多越好, within 越多越差 (竞争)
    # confidence = crossing / (crossing + within_up + within_down + 1)
    total = crossing_total + within_up_total + within_down_total
    confidence = crossing_total / max(1, total)

    return {
        'crossing_total': crossing_total,
        'within_up_total': within_up_total,
        'within_down_total': within_down_total,
        'crossing_dists': crossing_dists,
        'within_up_dists': within_up_dists,
        'within_down_dists': within_down_dists,
        'confidence': confidence,
    }


def rcm_pair_weight(
    seq_upstream: str,
    seq_downstream: str,
    base_weight: float = 1.0,
    kmer_lengths: Optional[List[int]] = None,
) -> float:
    """计算单对配对的 RCM 加权权重.

    用于注入管线: 替代或补充 ViennaRNA BPP 的置信度.

    Args:
        seq_upstream: BSJ 上游内含子
        seq_downstream: BSJ 下游内含子
        base_weight: 基础权重
        kmer_lengths: k 值列表

    Returns:
        加权权重 = base_weight × (1 + crossing_confidence)
    """
    result = compute_rcm_score(seq_upstream, seq_downstream, kmer_lengths)
    return base_weight * (1.0 + result['confidence'])


# ── 自测 ──
if __name__ == "__main__":
    import time

    # 完全互补的两条序列
    seq1 = "AUCGAUCGAUCGAUCG"
    seq2 = "CGAUCGAUCGAUCGAU"  # seq1 的反向互补

    print("=== RCM 自测 ===")
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2} (seq1 的反向互补)")

    t0 = time.time()
    result = compute_rcm_score(seq1, seq2)
    t1 = time.time()

    print(f"\n结果:")
    print(f"  crossing RCM 对数: {result['crossing_total']}")
    print(f"  within(up) 对数:   {result['within_up_total']}")
    print(f"  within(down) 对数: {result['within_down_total']}")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  耗时: {(t1-t0)*1000:.1f}ms")

    # 随机序列 (应有少量 RCM)
    rng = np.random.default_rng(42)
    bases = "AUCG"
    rand_seq1 = "".join(rng.choice(list(bases), 1000))
    rand_seq2 = "".join(rng.choice(list(bases), 1000))

    t0 = time.time()
    result_rand = compute_rcm_score(rand_seq1, rand_seq2)
    t1 = time.time()

    print(f"\n随机序列 (L=1000):")
    print(f"  crossing RCM 对数: {result_rand['crossing_total']}")
    print(f"  within(up) 对数:   {result_rand['within_up_total']}")
    print(f"  within(down) 对数: {result_rand['within_down_total']}")
    print(f"  置信度: {result_rand['confidence']:.3f}")
    print(f"  耗时: {(t1-t0)*1000:.1f}ms")
