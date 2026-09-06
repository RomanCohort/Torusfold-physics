"""ncm_ensemble.py — 从 ensemble 距离矩阵反推非经典配对 (NCM)。

原理: RNAbpFlow/trRNA2 的共识距离矩阵是几何证据——
非 WC 碱基对若被两个独立模型一致预测为 ~10Å (WC 几何),
则该位置大概率形成非经典配对 (Hoogsteen/shear 等 3D 接触).

与 ncm_detector.py 的序列启发式互补:
  - 序列启发式: 检测嵌在规则二级结构里的 tandem/miniloop
  - 距离反推: 检测孤立的、无序列上下文的 3D 接触

用法 (在 Level 0 或 segmented_vfold3d chunk 合并后调用):
    from torusfold.scheme2.ncm_ensemble import infer_ncm_from_distances
    ncm_pairs = infer_ncm_from_distances(
        dist_consensus, sequence, wc_pairs, chunk_offset=seg_start)
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

# WC 目标距离范围: 预测距离在此窗口内的非 WC 对视为 NCM 候选
NCM_DIST_LO = 8.0    # Å
NCM_DIST_HI = 12.5   # Å
# 最小序列间隔 (排除相邻残基和短环)
MIN_SEQ_GAP = 4
# 距离置信度映射: d=10.0Å → conf 最高; 偏离越远越低
NCM_REF_DIST = 10.2
NCM_CONF_MAX = 0.65   # 距离证据封顶 (低于 tandem 的 0.7)
NCM_CONF_MIN = 0.40


def _dist_confidence(d: float) -> float:
    """预测距离 → 置信度. 高斯核以 NCM_REF_DIST 为中心."""
    sigma = 1.5
    c = np.exp(-((d - NCM_REF_DIST) ** 2) / (2 * sigma * sigma))
    return float(NCM_CONF_MIN + (NCM_CONF_MAX - NCM_CONF_MIN) * c)


def infer_ncm_from_distances(
    dist_matrix: np.ndarray,
    sequence: str,
    wc_pairs: Set[Tuple[int, int]],
    *,
    known_pairs: Optional[Set[Tuple[int, int]]] = None,
    min_gap: int = MIN_SEQ_GAP,
    max_pairs: int = 200,
) -> List[Tuple[int, int, str, float]]:
    """从共识距离矩阵反推非经典配对.

    Args:
        dist_matrix: (L,L) 共识距离矩阵 (Å), 来自 trRNA2/RNAbpFlow
        sequence: ACGU 字符串 (chunk 局部序列)
        wc_pairs: 已知 WC 配对 {(i,j)}, 用于排除
        known_pairs: 其他已知配对 (硬/软约束), 也排除
        min_gap: 最小序列间隔
        max_pairs: 最多返回数 (按置信度排序截断)

    Returns:
        [(gi, gj, "ENSEMBLE_DIST", confidence)] — 全局索引
        (若传入 chunk_offset 由调用方平移; 本函数返回局部索引 + offset 参数版见下)
    """
    if dist_matrix is None or len(dist_matrix.shape) != 2:
        return []
    L = min(dist_matrix.shape[0], len(sequence))
    if L < min_gap * 2:
        return []

    seq_arr = np.frombuffer(sequence[:L].encode(), dtype=np.uint8)
    b1 = seq_arr[:, None]
    b2 = seq_arr[None, :]

    # 非 WC 且非 GU wobble
    is_canonical = np.zeros((L, L), dtype=bool)
    for a, b in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                 ("G", "U"), ("U", "G")]:
        is_canonical |= (b1 == ord(a)) & (b2 == ord(b))

    jj, ii = np.meshgrid(np.arange(L), np.arange(L))
    cand = (~is_canonical) & (jj > ii + min_gap)

    # 距离窗口
    dist_ok = (dist_matrix >= NCM_DIST_LO) & (dist_matrix <= NCM_DIST_HI)
    cand &= dist_ok

    # 排除已知配对
    exclude = set(wc_pairs or [])
    if known_pairs:
        exclude |= set(known_pairs)

    idx_i, idx_j = np.nonzero(cand)
    results = []
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        if (min(i, j), max(i, j)) in exclude:
            continue
        # 双侧对称取均值 (距离矩阵可能不对称)
        d = 0.5 * (float(dist_matrix[i, j]) + float(dist_matrix[j, i]))
        conf = _dist_confidence(d)
        results.append((i, j, "ENSEMBLE_DIST", round(conf, 3)))

    # 按置信度降序, 截断
    results.sort(key=lambda x: -x[3])
    return results[:max_pairs]


def infer_ncm_from_chunk(
    ens_dist_consensus: np.ndarray,
    seg_seq: str,
    global_wc_pairs: Set[Tuple[int, int]],
    chunk_start: int,
    overlap: int = 30,
    **kwargs,
) -> List[Tuple[int, int, str, float]]:
    """Chunk 版: 局部索引 → 全局索引.

    Args:
        ens_dist_consensus: chunk 内的共识距离矩阵
        seg_seq: chunk 序列
        global_wc_pairs: 全局 WC 配对集
        chunk_start: 该 chunk 在全序列中的起始位置
        overlap: chunk 重叠区宽度, 重叠区的检出降低置信度 (边界效应)

    Returns:
        [(全局i, 全局j, type, conf)]
    """
    L = len(seg_seq)
    # 全局 WC → chunk 局部
    local_wc = set()
    for gi, gj in global_wc_pairs:
        li, lj = gi - chunk_start, gj - chunk_start
        if 0 <= li < L and 0 <= lj < L:
            local_wc.add((li, lj))

    local_ncms = infer_ncm_from_distances(ens_dist_consensus, seg_seq, local_wc, **kwargs)

    out = []
    for li, lj, etype, conf in local_ncms:
        gi, gj = li + chunk_start, lj + chunk_start
        # 重叠区降权 (两端 overlap 宽度内)
        in_overlap = (li < overlap) or (lj >= L - overlap)
        if in_overlap:
            conf = round(conf * 0.7, 3)
        out.append((gi, gj, etype, conf))
    return out


def merge_chunk_ncms(
    chunk_ncm_lists: List[List[Tuple[int, int, str, float]]],
    iou_dedup: int = 0,
) -> List[Tuple[int, int, str, float]]:
    """合并多个 chunk 的 NCM 检出: 同一对取最高置信度.

    Args:
        chunk_ncm_lists: 各 chunk 的 [(gi,gj,type,conf)] 列表
        iou_dedup: 保留参数 (未来支持近邻去重)

    Returns:
        合并后的列表, 按置信度降序
    """
    best: dict = {}
    for lst in chunk_ncm_lists:
        for gi, gj, etype, conf in lst:
            key = (min(gi, gj), max(gi, gj))
            if key not in best or conf > best[key][3]:
                best[key] = (key[0], key[1], etype, conf)
    merged = list(best.values())
    merged.sort(key=lambda x: -x[3])
    return merged


# ── 自检 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 构造: 20nt, (3,14) A-G Hoogsteen 在距离 ~10.2Å
    rng = np.random.default_rng(42)
    L = 20
    seq = "".join(rng.choice(list("ACGU"), L))
    # 强制 seq[3]='A', seq[14]='G'
    seq = seq[:3] + "A" + seq[4:14] + "G" + seq[15:]

    dist = rng.uniform(20, 80, (L, L))  # 默认远
    np.fill_diagonal(dist, 0)
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= 1:
                dist[i, j] = abs(i - j) * 5.9
    # NCM 对放近距离
    dist[3, 14] = dist[14, 3] = 10.2

    wc = {(0, 19), (1, 18)}
    res = infer_ncm_from_distances(dist, seq, wc)
    hit = [r for r in res if r[0] == 3 and r[1] == 14]
    assert hit, f"expected NCM at (3,14), got {res[:5]}"
    print(f"[PASS] self-test: {hit}")
