"""
pair_graph.py - circRNA 配对图构建 + 互补性扫描 + 拓扑距离。

RL 远端配对优化的前置模块。给 ViennaRNA 配对 + 序列, 补充互补性扫描
漏掉的长程配对 (如 HA 反向重复), 建图 (骨架相邻 + 配对边), BFS 算
拓扑距离, 标记远端配对 (dist > 50), 提取茎块。

circRNA 环形拓扑: 骨架相邻边含 (L-1, 0) BSJ 闭合边。
|i-j| 绝对值在环上有跨 BSJ 误判 (如 L=2013 的 (5,2010) |i-j|=2005
但环距 8), 用图距离 (骨架边+配对边 BFS) 正确判定远端。

参数 (2026-07-21 定稿, 见 docs/scheme2_rl_design.md):
  W = 6            滑窗长度 (RNA 稳定茎最小长度)
  WC_RATE = 0.80   Watson-Crick 配对率阈值 (允许 1 个 G·U wobble)
  DG_THRESHOLD = -5.0   简单 NN 自由能阈值 (kcal/mol, 去假阳性)
  MIN_STEM = 4     连续配对最小长度 (茎块提取)
  FAR_DIST = 50    拓扑距离阈值 (dist > 50 = 远端)
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------- Watson-Crick 配对规则 ----------
# 标准 WC: A-U, G-C, G-U(wobble)
_WC_PAIRS: Set[Tuple[str, str]] = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
}
_GU_WOBBLE: Set[Tuple[str, str]] = {("G", "U"), ("U", "G")}

# ---------- 算法参数 ----------
W = 6                 # 滑窗长度
WC_RATE = 0.80        # WC 配对率阈值
DG_THRESHOLD = -3.0   # NN 自由能阈值 (kcal/mol, 粗参数表故放宽)
MIN_STEM = 4          # 茎块最小连续配对数
MIN_WC_IN_WIN = 4     # 方案 E: 扫描窗口内连续 WC 对数门槛 (6中≥4)
FAR_DIST = 50         # 远端拓扑距离阈值
MAX_KMER_FREQ = 20    # 方案 H: 单个 k-mer 出现 > 此值跳过 rc 匹配 (滤重复序列)

# ---------- 简单 RNA nearest-neighbor 自由能参数 ----------
# RNA NN model (SantaLucia 1998 近似值, kcal/mol at 37°C)
# 用于过滤互补扫描的假阳性: ΔG < -5 才算稳定茎
_NN_DG: Dict[Tuple[str, str, str, str], float] = {
    # 5'-XY-3' / 3'-X'Y'-5' 的 stacking free energy
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
# 默认 stack 能量 (查不到的 fallback)
_NN_DEFAULT = -1.0


def _is_wc_pair(b1: str, b2: str) -> bool:
    """标准 Watson-Crick 配对 (A-U, G-C)。"""
    return (b1, b2) in _WC_PAIRS


def _is_complementary(b1: str, b2: str) -> bool:
    """Watson-Crick 或 G·U wobble 配对。"""
    return (b1, b2) in _WC_PAIRS or (b1, b2) in _GU_WOBBLE


def _wc_count(seq_a: str, seq_b: str) -> int:
    """seq_a 与 seq_b 反向平行互补的 WC 配对数。

    win_b 是 win_a 的反向互补候选: seq_a[k] 配 seq_b[W-1-k]。
    """
    n = min(len(seq_a), len(seq_b))
    cnt = 0
    for k in range(n):
        if _is_wc_pair(seq_a[k], seq_b[n - 1 - k]):
            cnt += 1
    return cnt


def _nn_free_energy(seq_a: str, seq_b: str) -> float:
    """简单 nearest-neighbor stacking 自由能 (kcal/mol)。

    seq_a[k] 配 seq_b[n-1-k] (反向平行)。stack = 相邻两个配对。
    """
    n = min(len(seq_a), len(seq_b))
    if n < 2:
        return 0.0
    dg = 0.0
    for k in range(n - 1):
        x, y = seq_a[k], seq_a[k + 1]
        # 配对伙伴: seq_b[n-1-k], seq_b[n-1-(k+1)] = seq_b[n-2-k]
        xp, yp = seq_b[n - 1 - k], seq_b[n - 2 - k]
        dg += _NN_DG.get((x, y, xp, yp), _NN_DEFAULT)
    return dg


# ---------- 功能分区解析 (大小写 mask) ----------
def parse_case_annotation(
    sequence: str,
    *,
    default_coding: bool = False,
) -> np.ndarray:
    """从序列大小写解析出 coding mask。

    人工载体序列常有大写/小写混排: 大写段 = 功能元件(ORF/IRES/关键 motif),
    小写段 = UTR/linker/调控/限制酶位点。RL 后 amber 精修时, coding 区
    残基位置钉死(物理约束拉回 CG 原坐标), 非 coding 区接受 RL 优化 + 物理收敛。

    当序列无大小写区分 (全大写或全小写) 时, 整段按 default_coding 处理。
    CircBase 真样本通常全小写, default_coding=False 即全非 coding
    (RL 全序列可优化); 人工载体有大小写, 直接按字母大小写解析。

    Args:
        sequence: 序列 (大小写混排或纯字母)
        default_coding: 序列无大小写区分时的默认
            False (默认) = 默认全非 coding

    Returns:
        np.ndarray[bool], shape (L,). True = coding 区残基。
    """
    mask = np.zeros(len(sequence), dtype=bool)
    # 检测序列有无大小写区分
    has_upper = any(c.isupper() for c in sequence)
    has_lower = any(c.islower() for c in sequence)
    no_case_distinction = not (has_upper and has_lower)

    if no_case_distinction:
        # 无大小写区分: 整段按 default
        mask[:] = default_coding
        return mask

    # 有大小写区分: 按字母大小写解析
    for i, c in enumerate(sequence):
        if c.isalpha() and c.isupper():
            mask[i] = True
        elif c.isalpha() and c.islower():
            mask[i] = False
    return mask


# ---------- 互补性扫描 ----------
def complementarity_scan(
    sequence: str,
    window: int = W,
    wc_rate: float = WC_RATE,
    dg_threshold: float = DG_THRESHOLD,
    min_gap: int = 10,
) -> List[Tuple[int, int, float]]:
    """滑窗互补扫描 (方案 H: k-mer 索引, 从 O(L²) 降到 O(L×4^W))。

    旧版 O(L²) 双循环在 L=3000 时 4.5M 次迭代, 纯 Python。
    新版: 预索引所有窗口的 reverse_complement k-mer → 用 k-mer
    反向查找候选 → 只在候选上验 dg, 复杂度 O(L×4^W)。

    Args:
        sequence: ACGU 字符串 (circRNA, 环形)
        window: 滑窗长度 (默认 6)
        wc_rate: WC 配对率阈值 (允许 G·U wobble)
        dg_threshold: NN 自由能阈值, ΔG < 此值才算稳定 (kcal/mol)
        min_gap: 环距 < 此值跳过 (避免自配/相邻)

    Returns:
        [(i, j, dg), ...] 配对起始位置对 + 自由能。
        i, j 是窗口起始 (0-based), 代表 seq[i:i+W] 与 seq[j:j+W] 反向平行互补。
    """
    L = len(sequence)
    if L < window * 2 + min_gap:
        return []

    # 环形序列缓存 (窗口跨末尾时用)
    seq_ext = sequence + sequence[:window - 1]

    # 预索引: k-mer → [起始位置列表]
    kmer_idx: Dict[str, List[int]] = {}
    for i in range(L):
        kmer = seq_ext[i:i + window]
        if len(kmer) == window:
            kmer_idx.setdefault(kmer, []).append(i)

    # 反向映射: 对每个 k-mer, 找它的 reverse_complement
    # 只匹配低频 k-mer (方案 H: 避免 poly-G 等重复序列互相爆炸匹配)
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
                # 反向平行: win_b = seq[j:j+W] 配 win_a 的反向
                win_a = seq_ext[i:i + window]
                win_b = seq_ext[j:j + window]
                # 全匹配验证 (k-mer RC 已保证全 WC, 但需确认反向平行)
                wc = _wc_count(win_a, win_b)
                if wc / window < wc_rate:
                    continue
                # 能量过滤
                dg = _nn_free_energy(win_a, win_b)
                if dg >= dg_threshold:
                    continue
                pairs.append((i, j, dg))

    return pairs


def _reverse_complement(seq: str) -> str:
    """返回 seq 的反向互补 (ACGU)。"""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'U': 'A', 'A': 'U', 'N': 'N'}
    return "".join(comp.get(b, 'N') for b in reversed(seq))


# ---------- 配对图构建 ----------
def build_pair_graph(
    sequence: str,
    vienna_pairs: List[Tuple[int, int, float]],
    scan_pairs: Optional[List[Tuple[int, int, float]]] = None,
) -> Dict[int, List[int]]:
    """建配对图邻接表。

    节点 = 残基 0..L-1。
    边 = 骨架相邻 (i, (i+1) mod L) + ViennaRNA 配对 + 互补扫描补充。

    Returns:
        adj: {node: [neighbor, ...]} 无向图邻接表。
    """
    L = len(sequence)
    adj: Dict[int, List[int]] = {i: [] for i in range(L)}

    # 骨架相邻边 (含 BSJ 闭合 (L-1, 0))
    for i in range(L):
        nxt = (i + 1) % L
        adj[i].append(nxt)
        adj[nxt].append(i)

    # ViennaRNA 配对边
    for (i, j, _w) in vienna_pairs:
        if 0 <= i < L and 0 <= j < L and i != j:
            adj[i].append(j)
            adj[j].append(i)

    # 互补扫描补充 (扫描返回窗口起始, 展开成逐残基配对)
    # 方案 E: 质量门控 — 只在连续窗口内 ≥4/6 是 WC 对才展开
    if scan_pairs:
        for (i0, j0, _dg) in scan_pairs:
            # 高质量门: 要求 ≥ MIN_WC_IN_WIN 个连续 WC 对才算真远端茎
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

    # 去重 (同一对多次添加)
    for v in adj:
        adj[v] = list(set(adj[v]))

    return adj


# ---------- BFS 拓扑距离 ----------
def topological_distance(
    adj: Dict[int, List[int]], i: int, j: int,
    *,
    exclude_edge: Optional[Tuple[int, int]] = None,
) -> int:
    """BFS 算图距离 dist(i, j)。边权=1。

    exclude_edge: 若指定 (a, b), BFS 时跳过 a-b 这条边 (去自身配对边)。
    用于"去自身配对边的图距离": 衡量这对配对能否被其他配对快速连通。

    circRNA 环形: (5, 2010) 在 L=2013 上走骨架 8 步到达, dist=8 (近端);
    HA 反向重复拓扑远, dist 大。
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
            # 跳过要排除的边 (双向)
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
    return -1  # 不可达 (异常)


def ring_distance(i: int, j: int, L: int) -> int:
    """环距 = min(|i-j|, L-|i-j|)。纯骨架最短路径, 不含配对边。

    跨 BSJ 正确: L=2013 的 (5, 2010) 环距 = min(2005, 8) = 8。
    """
    return min(abs(i - j), L - abs(i - j))


def far_end_pairs(
    adj: Dict[int, List[int]],
    vienna_pairs: List[Tuple[int, int, float]],
    scan_pairs: Optional[List[Tuple[int, int, float]]] = None,
    far_dist: int = FAR_DIST,
) -> List[Tuple[int, int]]:
    """标记远端配对 = 环距远 且 拓扑孤立。

    两个条件都满足才算远端 (1 与 2 不矛盾, 互补):
      - 环距 = min(|i-j|, L-|i-j|) > far_dist  (纯骨架远)
      - 去自身配对边的图距离 > far_dist  (不被其他配对快速连通)

    合并 ViennaRNA + 扫描的所有配对, 逐对判定。

    降级兜底 (2026-07-22 加): 若上述强判定 0 产出, 说明配对图被扫描
    假阳性织成小世界 (实测 circBase 4000nt+ 上拓扑距 max=4-5, 全判近端)。
    此时拓扑孤立判失效, 降级到「环距远 且 配对来自 ViennaRNA 真配对」:
      ring_dist > max(far_dist, L//4)  且  (i,j) ∈ vienna_pairs
    只信 ViennaRNA 真配对, 规避扫描假阳性污染。无 ViennaRNA 时仍返回 []。
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

    # 强判定: 环距远 且 拓扑孤立
    far = []
    for (i, j) in all_pairs:
        # 条件 1: 环距远
        if ring_distance(i, j, L) <= far_dist:
            continue
        # 条件 2: 去自身配对边的图距离远 (拓扑孤立)
        d = topological_distance(adj, i, j, exclude_edge=(i, j))
        if d > far_dist:
            far.append((i, j))

    if far:
        return far

    # 降级: 拓扑判失效 (图织密), 改用环距 + ViennaRNA 真配对
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


# ---------- 茎块提取 ----------
def extract_stem_blocks(
    vienna_pairs: List[Tuple[int, int, float]],
    scan_pairs: Optional[List[Tuple[int, int, float]]] = None,
    min_stem: int = MIN_STEM,
) -> List[List[Tuple[int, int]]]:
    """提取茎块 (连续配对 ≥ min_stem 的段)。

    扫描返回窗口对, 展开成逐残基配对后, 按位置连续性聚类。
    一个茎块 = 一串连续 i 配一串连续 j (反向平行)。

    Returns:
        [[(i, j), ...], ...] 每个茎块的逐残基配对列表。
    """
    # 合并所有配对 (展开扫描窗口)
    pair_set: Set[Tuple[int, int]] = set()
    for (i, j, _w) in vienna_pairs:
        pair_set.add((min(i, j), max(i, j)))
    if scan_pairs:
        # 扫描窗口对展开成逐残基配对 (此处简化: 只用窗口代表对)
        for (i0, j0, _dg) in scan_pairs:
            pair_set.add((min(i0, j0), max(i0, j0)))

    # 按 i 排序, 找连续 i + 连续 j 的段
    sorted_pairs = sorted(pair_set)
    blocks: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []

    for p in sorted_pairs:
        if not current:
            current = [p]
            continue
        prev = current[-1]
        # 连续: i 递增 1, j 递减 1 (反向平行) 或 j 递增 1 (平行, 少见)
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


# ---------- 端到端入口 ----------
def build_full_pair_graph(
    sequence: str,
    vienna_pairs: List[Tuple[int, int, float]],
    *,
    do_scan: bool = True,
    window: int = W,
    wc_rate: float = WC_RATE,
    dg_threshold: float = DG_THRESHOLD,
) -> Tuple[Dict[int, List[int]], List[Tuple[int, int, float]], List[Tuple[int, int]]]:
    """端到端: 序列 + ViennaRNA 配对 -> 配对图 + 扫描补充 + 远端配对列表。

    Returns:
        adj: 配对图邻接表
        scan_pairs: 互补扫描补充的配对 [(i, j, dg), ...]
        far_pairs: 远端配对 [(i, j), ...] (拓扑距离 > FAR_DIST)
    """
    scan = complementarity_scan(sequence, window, wc_rate, dg_threshold) if do_scan else []
    adj = build_pair_graph(sequence, vienna_pairs, scan)
    far = far_end_pairs(adj, vienna_pairs, scan)
    return adj, scan, far


if __name__ == "__main__":
    # 自测: 含已知反向重复的合成序列
    # 构造: 一段 poly-A 反向重复 + 一段随机
    import random
    random.seed(42)

    # stem1: 5'-AUGCAUGC-3' / 3'-UACGUACG-5' (完全互补, 反向重复)
    stem = "AUGCAUGC"
    complement = stem[::-1].translate(str.maketrans("AUGC", "UACG"))
    # 序列 = stem + linker + complement (反向重复, 应被扫描捕获)
    linker = "AAAA" * 5  # 20nt poly-A linker
    seq = stem + linker + complement + linker + stem + linker + complement

    print(f"序列长度: {len(seq)}")
    print(f"stem: {stem}")
    print(f"complement (反向): {complement}")

    # 假 ViennaRNA 配对 (空, 模拟漏掉反向重复)
    vienna_pairs: List[Tuple[int, int, float]] = []
    scan = complementarity_scan(seq)
    print(f"\n互补扫描命中: {len(scan)} 对")
    for (i, j, dg) in scan[:10]:
        print(f"  ({i}, {j}) ΔG={dg:.2f}")

    adj = build_pair_graph(seq, vienna_pairs, scan)
    print(f"\n图节点: {len(adj)}")
    print(f"平均度数: {sum(len(v) for v in adj.values())/len(adj):.2f}")

    far = far_end_pairs(adj, vienna_pairs, scan)
    print(f"\n远端配对 (dist>{FAR_DIST}): {len(far)} 对")
    for (i, j) in far[:5]:
        print(f"  ({i}, {j})")
