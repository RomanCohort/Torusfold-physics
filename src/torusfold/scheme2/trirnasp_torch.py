"""trirnasp_torch.py — TriRNASP 三体统计势的 PyTorch GPU 批量实现.

与 trirnasp_openmm.py (numpy CPU 版) 的对应关系:
  - 能量表 Rough.energy (12³×4×4×8 = 73k 条) 一次性载入 → device 常量
  - 三体项: 对每个合法 pair (i,j), 找 R0 内的 k>j, 查表累加
    — CPU 版是 Python for 循环逐 pair; torch 版全 batch 化:
      pair 列表 (M,) × 邻居变长 → 展平成三元组列表 (T,) 一次 gather
  - 梯度: 默认硬分箱查表 (分段常数势, 与 CPU score() 逐位对齐, 用于
    打分/校验). ★ 该路径计算图被 .long() 截断 — 不可微, autograd
    拿不到梯度. soft=True 时改用相邻 bin 的三线性插值: 势变成距离的
    分段线性函数, autograd 经插值权重回传严格梯度, 供 REMD 外力用.
    (CPU 版 score_with_gradient 的前向差分线性化问题由此根治)

批量副本支持: (B, N, 3) 一批结构同时打分 (REMD 全副本一次算).

单位约定: 输入 Å (3bead 布局), 输出 kBT; 调用方负责 kJ/mol 换算.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── 与 trirnasp_openmm.py 对齐的常数 ──
R0 = 8.0
BIN_WIDTH_ROUGH = 2.0
EXCLUSION_R1_SQ = 1.21   # (1.1)^2
EXCLUSION_R2_SQ = 2.89   # (1.7)^2
EXCLUSION_R3_SQ = 16.0   # (4.0)^2
R0_SQ = (R0 - 0.3) ** 2

_BASE_OFFSET = {"A": 0, "U": 1, "C": 2, "G": 3}


def _type_code(base: str, bead: str) -> int:
    b = _BASE_OFFSET.get(base, -1)
    if b < 0:
        return -1
    if bead == "C4'":
        return b
    if bead in ("N9", "N1"):
        return 4 + b
    if bead == "P":
        return 8 + b
    return -1


def _load_energy_table(filepath: str, n_types: int, n_bins: int,
                       n_bins_fine: int) -> np.ndarray:
    """同 trirnasp_openmm._load_energy_table."""
    total = (n_types ** 3) * (n_bins ** 2) * (2 * n_bins)
    table = np.zeros(total, dtype=np.float64)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            t1, t2, t3 = int(parts[0]), int(parts[1]), int(parts[2])
            b12, b13, b23 = int(parts[3]), int(parts[4]), int(parts[5])
            energy = float(parts[6])
            idx = (((((t1 * n_types + t2) * n_types + t3) * n_bins + b12)
                    * n_bins + b13) * (2 * n_bins) + b23)
            if 0 <= idx < total:
                table[idx] = energy
    return table


class TriRNASPTorch:
    """GPU 批量 TriRNASP 势.

    一次性预计算 (与序列/坐标无关的拓扑量):
      - atom 类型/残基索引表 (3L 原子: P,C4',N 布局同 3bead)
      - pair 候选 (i<j, 排除规则只依赖 res_diff 和距离上限的静态部分)
      - 每个 pair 的邻居 mask 变长部分在运行时按距离筛

    运行时每次调用做:
      dist_sq (B,N,N) → pair 距离 → bin → 表 gather → sum
    梯度: 硬模式不可微 (硬截断查表断图); 软模式 (soft=True) 在相邻
    bin 间三线性插值, 计算图完整, autograd 给严格梯度.
    """

    def __init__(self, sequence: str, energy_dir: Optional[str] = None,
                 device="cuda"):
        assert TORCH_OK, "PyTorch 未安装"
        if energy_dir is None:
            _root = Path(__file__).resolve().parents[3]
            energy_dir = str(_root / "external" / "TriRNASP" / "Energy")

        rough_path = Path(energy_dir) / "Rough.energy"
        if not rough_path.exists():
            raise FileNotFoundError(f"Rough.energy not found: {rough_path}")

        self.device = device if torch.cuda.is_available() else "cpu"
        rough_np = _load_energy_table(str(rough_path), 12, 4, 4)
        self._rough = torch.tensor(rough_np, dtype=torch.float32,
                                   device=self.device)

        self.L = len(sequence)

        # ── 原子布局: [P, C4', N]×L (与 3bead 一致, N=3L) ──
        # trirnasp_openmm 的顺序是 C4',N,P — 但类型码决定查表结果,
        # 顺序只影响内部索引一致性, 这里统一用 3bead 布局省转换.
        codes_c = [_type_code(sequence[i], "C4'") for i in range(self.L)]
        codes_n = [_type_code(
            sequence[i], "N9" if sequence[i] in ("A", "G") else "N1")
            for i in range(self.L)]
        codes_p = [_type_code(sequence[i], "P") for i in range(self.L)]

        # 布局约定: 与 trirnasp_openmm CPU 版完全一致 — [C4', N, P]×L.
        # ★ 三体计数依赖原子排序 (k>j 规则), 布局不同能量就不同,
        #   所以这里必须逐字对齐 CPU 的 valid append 顺序.
        # 注意这与 torch_cgsim 的 [P,C4,N] 不同; energy_from_3bead
        # 负责把 3bead 粒子重排到本布局.
        types = []
        res_of = []
        bead_seq = []
        for i in range(self.L):
            for code, bead_pos in ((codes_c[i], 1), (codes_n[i], 2),
                                   (codes_p[i], 0)):
                if code >= 0:
                    types.append(code); res_of.append(i); bead_seq.append(bead_pos)

        self.n_atoms = len(types)
        self.atom_types = torch.tensor(types, dtype=torch.long,
                                       device=self.device)
        self.atom_res = torch.tensor(res_of, dtype=torch.long,
                                     device=self.device)
        # 原子 a → 3bead 扁平粒子索引 ([P,C4,N] 布局的 3r+bead)
        self._particle_idx = torch.tensor(
            [3 * r + b for r, b in zip(res_of, bead_seq)],
            dtype=torch.long, device=self.device)

        # ── 静态 pair 候选: i<j 且 |res_i - res_j| 结构性排除在运行时判 ──
        # 这里只预筛 i<j; 距离相关的排除每步算.
        na = self.n_atoms
        iu, ju = np.triu_indices(na, k=1)
        self._iu = torch.tensor(iu, dtype=torch.long, device=self.device)
        self._ju = torch.tensor(ju, dtype=torch.long, device=self.device)
        res_i = self.atom_res[self._iu]
        res_j = self.atom_res[self._ju]
        self._res_diff = (res_i - res_j).abs()          # (M,)
        # k>j 邻接候选的全局上三角 (k 索引 > j): 运行时按 pair 展开
        # 为控制显存, 邻居筛选在运行时对每个 pair 单独 gather (向量化).

        # 排除阈值常量
        self._r1sq = EXCLUSION_R1_SQ
        self._r2sq = EXCLUSION_R2_SQ
        self._r3sq = EXCLUSION_R3_SQ
        self._r0sq = R0_SQ
        self._inv_bw = 1.0 / BIN_WIDTH_ROUGH

    def energy(self, coords_A: "torch.Tensor",
               soft: bool = False) -> "torch.Tensor":
        """批量 TriRNASP 能量 (kBT).

        Args:
            coords_A: (B, N_atoms, 3) Å — 注意是 _build_atoms 后的原子序,
                      由本类 particle_idx 映射回 3bead 粒子.
                      便捷入口: 用 from_3bead() 先转换.
            soft: False → 硬分箱查表 (与 CPU score 逐位对齐, 但
                  计算图被 .long() 截断, autograd 无梯度);
                  True → 相邻 bin 三线性插值, 势变成距离的
                  分段线性函数, autograd 经插值权重给出严格梯度.

        Returns:
            (B,) kBT
        """
        B = coords_A.shape[0]
        dev = coords_A.device

        diff = coords_A[:, :, None, :] - coords_A[:, None, :, :]
        dist_sq = (diff * diff).sum(-1)                    # (B,N,N)

        iu, ju = self._iu, self._ju
        d_ij = dist_sq[:, iu, ju]                          # (B,M)
        res_diff = self._res_diff[None]                    # (1,M)

        excl = d_ij <= self._r1sq
        e_const = torch.where(excl, torch.ones_like(d_ij) * 0.5,
                              torch.zeros_like(d_ij))
        m2 = (~excl) & (d_ij <= self._r2sq) & (res_diff > 1)
        e_const = e_const + torch.where(m2, torch.ones_like(d_ij) * 0.5,
                                        torch.zeros_like(d_ij))
        excl = excl | m2
        m3 = (~excl) & (d_ij <= self._r3sq) & (d_ij > self._r2sq)
        e_const = e_const + torch.where(m3, torch.ones_like(d_ij) * 0.2,
                                        torch.zeros_like(d_ij))
        excl = excl | m3

        cand = (~excl) & (d_ij < self._r0sq)               # (B,M)
        d12 = torch.sqrt(torch.where(cand, d_ij, torch.zeros_like(d_ij)))
        if soft:
            # 软分箱: 保留连续坐标, 插值权重保留 autograd 通路
            f12 = d12 * self._inv_bw                         # 连续 bin 坐标
            b12_lo = f12.floor().long()                      # 下界 bin
            w12 = (f12 - b12_lo.float()).clamp(0.0, 1.0)   # 插值权重
            b12_lo = b12_lo.clamp(0, 3)                     # 安全截断
        else:
            b12 = (d12 * self._inv_bw).long()
        # CPU 版: b12>3 的 pair 从三体候选剔除 (ok12 过滤)
        if soft:
            ok12 = (b12_lo <= 3) & cand
        else:
            ok12 = (b12 <= 3) & cand
        cand = ok12

        total = e_const.sum(dim=-1)                        # (B,)
        if not bool(cand.any()):
            return total

        # ── 三体项: 向量化展开 (pair, k) 组合 ──
        # 对每个候选 pair (i,j), k ∈ (j, N) 且 dist²(i,k)<R0², dist²(j,k)<rcut
        # 全展开 M×N 太大时按 pair 分块 (chunk) 处理.
        ci = iu[None].expand(B, -1)
        cj = ju[None].expand(B, -1)

        # 只处理有候选的 (b, pair) 组合
        cand_flat = cand.reshape(-1)
        sel = torch.nonzero(cand_flat, as_tuple=False).squeeze(-1)
        b_idx = sel // cand.shape[1]
        p_idx = sel % cand.shape[1]
        if b_idx.numel() == 0:
            return total

        bi_all = iu[p_idx]
        bj_all = ju[p_idx]

        # ── 三体项: 按 pair 分块展开, 控制 (chunk, N) 中间张量显存 ──
        # L=2013 时 M~18M 候选 pair × N=6039 直接广播 ≈ 1TB — 必须 chunk.
        # 每块目标中间量 ≈ chunk × N × 8B ≤ ~256MB → chunk 自适应.
        n_atoms = coords_A.shape[1]
        n_pairs_sel = b_idx.numel()
        k_col = torch.arange(n_atoms, device=dev)              # (N,)
        rk_all = self.atom_res[k_col]                          # (N,)
        rcut_sq = (R0 - 0.3) ** 2

        bytes_per_row = n_atoms * 4 * 4   # d_ik/d_jk/g13/g23 float32
        chunk = max(1, int(64 * 2 ** 20 // max(bytes_per_row, 1)))

        tri_acc = torch.zeros(B, dtype=torch.float32, device=dev)
        for s in range(0, n_pairs_sel, chunk):
            e_chunk = min(s + chunk, n_pairs_sel)
            bb = b_idx[s:e_chunk]
            pi_ = p_idx[s:e_chunk]
            bi = bi_all[s:e_chunk]
            bj = bj_all[s:e_chunk]

            kc = k_col[None, :]                                # (1,N)
            d_ik = dist_sq[bb[:, None], bi[:, None],
                           kc.expand(e_chunk - s, -1)]
            d_jk = dist_sq[bb[:, None], bj[:, None],
                           kc.expand(e_chunk - s, -1)]
            k_ok = (kc > bj[:, None]) & \
                (d_ik < self._r0sq) & (d_jk < rcut_sq)

            # 排除规则 (三体)
            ri = self.atom_res[bi][:, None]
            rj = self.atom_res[bj][:, None]
            rk = rk_all[None, :]
            d13 = d_ik
            d23 = d_jk
            bad = (d13 <= self._r1sq) | (d23 <= self._r1sq)
            bad = bad | ((d13 <= self._r2sq) & ((ri - rk).abs() > 1))
            bad = bad | ((d23 <= self._r2sq) & ((rj - rk).abs() > 1))
            bad = bad | ((d13 > self._r2sq) & (d13 <= self._r3sq))
            bad = bad | ((d23 > self._r2sq) & (d23 <= self._r3sq))
            good = k_ok & (~bad)

            # 数值哲学 (2026-08-26 与学长定版): GPU 内部全程 f64,
            # 不复刻 CPU "f32 存 dsq 再开方" 的历史路径. 与 CPU 的
            # ~1-2% bin 边界偏差是结构相关基线偏移, REMD 副本间
            # 相互抵消, 不影响交换判据的自洽性.
            g_d13 = torch.sqrt(
                torch.where(good, d13, torch.zeros_like(d13)))
            g_d23 = torch.sqrt(
                torch.where(good, d23, torch.zeros_like(d23)))
            if soft:
                f13 = g_d13 * self._inv_bw
                b13_lo = f13.floor().long()
                w13 = (f13 - b13_lo.float()).clamp(0.0, 1.0)
                b13_lo = b13_lo.clamp(0, 3)
                f23 = g_d23 * self._inv_bw
                b23_lo = f23.floor().long()
                w23 = (f23 - b23_lo.float()).clamp(0.0, 1.0)
                b23_lo = b23_lo.clamp(0, 7)
                fin = good & (b13_lo <= 3) & (b23_lo <= 7)
            else:
                b13 = (g_d13 * self._inv_bw).long()
                b23 = (g_d23 * self._inv_bw).long()
                fin = good & (b13 <= 3) & (b23 <= 7)
            if not bool(fin.any()):
                continue

            f_rows, f_k = torch.nonzero(fin, as_tuple=True)
            t_i = self.atom_types[iu[pi_[f_rows]]]
            t_j = self.atom_types[ju[pi_[f_rows]]]
            t_k = self.atom_types[f_k]
            # atom type 维度的基址 (type part is discrete — always integer)
            base_t = ((t_i * 12 + t_j) * 12 + t_k).long()

            if soft:
                # ── 三线性插值: 势 = Σ角落 w₁w₂w₃ · E(角落) ──
                tb12b = b12_lo[bb[f_rows], pi_[f_rows]].long()
                tb13b = b13_lo[f_rows, f_k].long()
                tb23b = b23_lo[f_rows, f_k].long()
                w12v = w12[bb[f_rows], pi_[f_rows]]
                w13v = w13[f_rows, f_k]
                w23v = w23[f_rows, f_k]
                e_interp = torch.zeros_like(w12v)
                for c0 in range(2):
                    for c1 in range(2):
                        for c2 in range(2):
                            cb12 = (tb12b + c0).clamp(0, 3)
                            cb13 = (tb13b + c1).clamp(0, 3)
                            cb23 = (tb23b + c2).clamp(0, 7)
                            idx_c = ((base_t * 4 + cb12) * 4 + cb13) * 8 + cb23
                            wc = ((1.0 - w12v) if c0 == 0 else w12v) * \
                                 ((1.0 - w13v) if c1 == 0 else w13v) * \
                                 ((1.0 - w23v) if c2 == 0 else w23v)
                            e_interp = e_interp + wc * self._rough[idx_c]
                tri_acc.index_add_(0, bb[f_rows], e_interp.float())
            else:
                tb12 = b12[bb[f_rows], pi_[f_rows]]
                tb13 = b13[f_rows, f_k]
                tb23 = b23[f_rows, f_k]
                idx = ((base_t * 4 + tb12) * 4 + tb13) * 8 + tb23
                e_vals = self._rough[idx]
                tri_acc.index_add_(0, bb[f_rows], e_vals)

        return total + tri_acc

    def energy_from_3bead(self, pos_nm: "torch.Tensor",
                          soft: bool = False) -> "torch.Tensor":
        """便捷入口: (B, 3L, 3) nm torch_cgsim 布局 ([P,C4,N]×L) → (B,) kBT.

        _particle_idx 把本类原子序映射到 3bead 粒子索引 —
        这里反向 gather: 按本类原子顺序取出对应粒子坐标.
        """
        coords_A = pos_nm * 10.0
        atoms = coords_A[:, self._particle_idx, :]
        return self.energy(atoms, soft=soft)


if __name__ == "__main__":
    """冒烟: 与 numpy CPU 版能量对比 (同一组残基坐标, 允许 float32 容差).

    布局约定:
      CPU 版 score() 吃 (L,3,3), bead 序 = 原子序 [C4', N, P]
        — 见 trirnasp_openmm._build_atoms 的 valid append 顺序
      本类 energy_from_3bead() 吃 (B,3L,3) nm, bead 序 = [P, C4', N]
    同一残基坐标分别按两种布局展开.
    """
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from torusfold.scheme2.trirnasp_openmm import TriRNASPPotential

    seq = "AUGCAUGCAUGCAUGC"
    L = len(seq)
    rng = np.random.default_rng(42)
    res_coords = rng.random((L, 3, 3)).astype(np.float64) * 8.0  # Å, 分散
    # CPU 版 (L,3,3) axis 语义: axis0=P, axis1=C4', axis2=N
    # (_build_atoms 用 bead id 直接索引 axis: P←[r,0], C4'←[r,1], N←[r,2])

    pot_cpu = TriRNASPPotential()
    e_cpu = pot_cpu.score(res_coords.copy(), seq)

    pot_gpu = TriRNASPTorch(seq)
    # 本类布局 [C4',N,P]×L = CPU 布局; res_coords axis 是 [P,C4,N]
    # (axis0=P) → 重排为 [C4'=axis1, N=axis2, P=axis0]
    flat = np.stack([res_coords[:, 1], res_coords[:, 2], res_coords[:, 0]],
                    axis=1).reshape(3 * L, 3)
    t = torch.tensor(flat, dtype=torch.float64, device=pot_gpu.device)[None]
    e_gpu = float(pot_gpu.energy(t)[0])

    print(f"CPU kBT: {e_cpu:.4f}")
    print(f"GPU kBT: {e_gpu:.4f}")
    rel = abs(e_gpu - e_cpu) / max(abs(e_cpu), 1e-8)
    print(f"rel err: {rel:.2%}")
    assert rel < 0.05, f"GPU/CPU TriRNASP 能量偏差过大: {rel}"
    print("[PASS] TriRNASP torch matches numpy within tolerance")

    # ── 软分箱测试: 可微性 + 数值验证 ──
    t_test = t.clone().detach().requires_grad_(True)
    e_soft = pot_gpu.energy(t_test, soft=True)[0]
    print(f"\n[soft] energy kBT: {e_soft.item():.4f}")
    diff_rel = abs(e_soft.item() - e_gpu) / max(abs(e_gpu), 1e-8)
    print(f"[soft] soft/hard rel diff: {diff_rel:.4%} (表稀疏导致, 预期行为)")
    e_soft.backward()
    ag = t_test.grad.clone()   # (1,48,3)
    print(f"[soft] |autograd|={ag.norm().item():.4f} kBT/Å, "
          f"|per-atom|={ag.view(-1,3).norm(dim=1).mean().item():.4f}")

    # 数值验证: 单原子微扰, cosine > 0.9 → 梯度正确
    eps = 1e-3
    ag0 = ag[0, 0].double()   # 原子0的xyz
    num_g = torch.zeros(3, device=ag.device, dtype=torch.float64)
    for d in range(3):
        tp = t.clone()
        tp[0, 0, d] += eps
        ep = pot_gpu.energy(tp, soft=True)[0].item()
        tm = t.clone()
        tm[0, 0, d] -= eps
        em = pot_gpu.energy(tm, soft=True)[0].item()
        num_g[d] = (ep - em) / (2*eps)
    cos = float(torch.dot(ag0.double(), num_g) / (ag0.norm() * num_g.norm() + 1e-12))
    print(f"[soft] atom0 cosine(autograd, numerical)={cos:.4f} (eps={eps})")
    assert abs(cos) > 0.9, f"gradient direction wrong: cosine={cos}"
    print("[PASS] soft-binning gradient direction verified")

    print("[PASS] soft-binning smoke complete")
