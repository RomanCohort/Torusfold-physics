"""
multisource_ss.py — 多源二级结构共识预测

灵感来源: structRFM 的 MUSES (Multiple Sequence & Structure Prediction Fusion)
- 5 个预测器加权平均 → MEA 解码 → 共识 SS

简化版: 用 ViennaRNA + 结构一致性加权，不需要额外依赖

用法:
    from .multisource_ss import multisource_consensus_ss

    # 输入: 序列 + 可选 bpp 矩阵
    ss_consensus, bpp_fused = multisource_consensus_ss(sequence, bpp_vienna)
"""

import numpy as np
from typing import Optional, Tuple, List


def _vienna_fold_consensus(sequence: str) -> Tuple[str, np.ndarray]:
    """单源: ViennaRNA fold → (ss_string, bpp_matrix)"""
    try:
        import RNA
        fc = RNA.fold_compound(sequence)
        ss, mfe = fc.mfe()
        # ViennaRNA bpp: bp[i] = probability that i pairs with any j>i
        # 转为 LxL 对称矩阵
        L = len(sequence)
        bpp_0 = np.zeros((L, L), dtype=np.float32)
        bp = fc.bpp()
        for i in range(1, L + 1):
            if bp[i] > 0 and int(bp[i]) > i:
                j = int(bp[i])
                bpp_0[i-1, j-1] = 1.0
                bpp_0[j-1, i-1] = 1.0
        return ss, bpp_0
    except (ImportError, Exception):
        return None, None


def _ss_to_contact_matrix(ss: str) -> np.ndarray:
    """SS 字符串 → 接触矩阵 (0/1)"""
    L = len(ss)
    contact = np.zeros((L, L), dtype=np.float32)
    stack = []
    bracket_map = {'(': ')', '[': ']', '{': '}', '<': '>'}
    close_to_open = {v: k for k, v in bracket_map.items()}

    for i, ch in enumerate(ss):
        if ch in bracket_map:
            stack.append((ch, i))
        elif ch in close_to_open:
            if stack and stack[-1][0] == close_to_open[ch]:
                _, j = stack.pop()
                contact[j, i] = 1.0
                contact[i, j] = 1.0
    return contact


def _consensus_from_bpp_list(bpp_list: List[np.ndarray],
                               weights: Optional[List[float]] = None) -> np.ndarray:
    """多源 bpp 加权平均 → 共识 bpp"""
    if weights is None:
        weights = [1.0] * len(bpp_list)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    result = np.zeros_like(bpp_list[0], dtype=np.float32)
    for bpp, w in zip(bpp_list, weights):
        result += w * bpp
    return result


def _confidence_weighted_consensus(ss_list: List[str],
                                    bpp_list: List[np.ndarray]) -> Tuple[str, np.ndarray]:
    """置信度加权共识:
    - 每个预测器的 bpp 矩阵用配对概率作为置信度
    - 高置信度的预测器权重更大
    """
    L = len(ss_list[0])
    # 计算每个预测器的置信度: 对角线元素平均值
    confidences = []
    for bpp in bpp_list:
        if bpp is not None:
            diag = np.diag(bpp[:L, :L])
            conf = float(np.mean(diag[diag > 0.05])) if np.any(diag > 0.05) else 0.0
            confidences.append(max(conf, 0.01))
        else:
            confidences.append(0.01)

    # 归一化权重
    confidences = np.array(confidences, dtype=np.float32)
    weights = confidences / confidences.sum()

    # 加权 bpp
    fused_bpp = _consensus_from_bpp_list(bpp_list, weights.tolist())

    # 共识 SS: 多数投票 (只对 Vienna 格式符号)
    consensus = list(ss_list[0])
    for pos in range(L):
        votes = {}
        for ss in ss_list:
            ch = ss[pos] if pos < len(ss) else '.'
            votes[ch] = votes.get(ch, 0) + 1
        # 优先投票: 配对 > 未配对
        paired = {k: v for k, v in votes.items() if k != '.'}
        if paired:
            best_paired = max(paired, key=paired.get)
            if paired[best_paired] > votes.get('.', 0):
                consensus[pos] = best_paired
            else:
                consensus[pos] = '.'
        else:
            consensus[pos] = '.'

    return ''.join(consensus), fused_bpp


def multisource_consensus_ss(
    sequence: str,
    bpp_vienna: Optional[np.ndarray] = None,
    use_nussinov_fallback: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    多源二级结构共识预测

    简化版 MUSES: ViennaRNA + Nussinov(可选) + 置信度加权

    Args:
        sequence: RNA 序列 (ACGU)
        bpp_vienna: 可选的预计算 ViennaRNA bpp 矩阵
        use_nussinov_fallback: 是否用 Nussinov 作为第二个预测器

    Returns:
        ss_consensus: 共识二级结构字符串
        bpp_fused: 融合 bpp 矩阵 (L, L)
    """
    L = len(sequence)
    ss_list = []
    bpp_list = []

    # Source 1: ViennaRNA fold
    if bpp_vienna is not None:
        # 从 bpp 反推 SS
        ss_v = '.' * L
        stack = []
        for i in range(L):
            for j in range(i + 3, L):  # min loop size = 3
                if bpp_vienna[i, j] > 0.5:
                    ss_v = ss_v[:i] + '(' + ss_v[i+1:j] + ')' + ss_v[j+1:]
                    break
        ss_list.append(ss_v)
        bpp_list.append(bpp_vienna)
    else:
        ss_v, bpp_v = _vienna_fold_consensus(sequence)
        if ss_v is not None:
            ss_list.append(ss_v)
            bpp_list.append(bpp_v)

    # Source 2: Nussinov (简单最大匹配)
    if use_nussinov_fallback and L <= 500:
        ss_n, bpp_n = _nussinov_fold(sequence)
        ss_list.append(ss_n)
        bpp_list.append(bpp_n)

    if len(ss_list) == 0:
        return '.' * L, np.zeros((L, L), dtype=np.float32)

    if len(ss_list) == 1:
        return ss_list[0], bpp_list[0]

    return _confidence_weighted_consensus(ss_list, bpp_list)


def _nussinov_fold(sequence: str) -> Tuple[str, np.ndarray]:
    """Nussinov 最大匹配算法 (简单基线)"""
    L = len(sequence)
    can_pair = np.zeros((L, L), dtype=bool)

    # RNA 碱基配对规则
    pairs = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}
    for i in range(L):
        for j in range(i + 4, L):  # min loop = 4
            if (sequence[i], sequence[j]) in pairs:
                can_pair[i, j] = True

    # DP
    dp = np.zeros((L, L), dtype=int)
    for length in range(5, L + 1):
        for i in range(L - length + 1):
            j = i + length - 1
            dp[i, j] = dp[i, j-1]
            for k in range(i, j - 3):
                if can_pair[k, j]:
                    score = 1 + dp[i, k-1] if k > i else 1
                    if k > i:
                        score += dp[i, k-1]
                    if k + 1 < j:
                        score += dp[k+1, j-1]
                    dp[i, j] = max(dp[i, j], score)

    # 回溯
    ss = ['.' ] * L
    traceback_stack = [(0, L - 1)]
    while traceback_stack:
        i, j = traceback_stack.pop()
        if i >= j:
            continue
        if dp[i, j] == dp[i, j-1]:
            traceback_stack.append((i, j-1))
            continue
        found = False
        for k in range(i, j - 3):
            if can_pair[k, j]:
                score = 1
                if k > i:
                    score += dp[i, k-1]
                if k + 1 < j:
                    score += dp[k+1, j-1]
                if score == dp[i, j]:
                    ss[k] = '('
                    ss[j] = ')'
                    if k > i:
                        traceback_stack.append((i, k-1))
                    if k + 1 < j:
                        traceback_stack.append((k+1, j-1))
                    found = True
                    break
        if not found:
            traceback_stack.append((i, j-1))

    # 构建 bpp 矩阵
    bpp = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        if ss[i] == '(':
            for j in range(i+1, L):
                if ss[j] == ')':
                    bpp[i, j] = 1.0
                    bpp[j, i] = 1.0
                    break

    return ''.join(ss), bpp
