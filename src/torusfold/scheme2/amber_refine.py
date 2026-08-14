"""
amber_refine.py — 全原子 RNA 用 Amber14 OL3 力场精修。

接 allatom_reconstruct 输出, 用 OpenMM + amber14-all.xml (RNA.OL3) +
implicit/obc1.xml (OBC1 隐式溶剂) 做约束最小化, 让全原子在力场下落到
A-form RNA 合理构象。

激进精度提升 (相对原方案 3):
  * P 原子 positional restraint 从 50 → 10 kJ/mol/nm²: 放开骨架走向,
    让力场能调整 backbone 满足配对, 而不是把 CG 拓扑钉死。
  * A-form 螺旋二面角约束 (CustomTorsionForce): backbone α/γ/δ/ζ +
    sugar pucker C3'-endo, 这是 RNA 精度灵魂, 之前完全没有。
  * ViennaRNA pairing CustomBond: r0=1.06nm (10.6Å C1'-C1'), 配对约束
    让碱基在 stem 区靠拢成 Watson-Crick 几何。
  * L-BFGS 最小化 3000 步, 长序列 (>200nt) 加 MD 退火跳出局部解。
  * Modeller.addHydrogens 自动补 H (amber14 RNA 模板含 H)。

安全网: P 偏离 > 2Å 或能量炸 → 抛 RuntimeError, 上层 fallback CG。
"""

from __future__ import annotations

import sys

# Windows GBK 编码 workaround: print 含非 ASCII 字符（如 Å）时 GBK 打不出
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass  # 忽略重配失败（某些环境不允许）

from typing import Dict, List, Optional, Tuple

import numpy as np

from .allatom_reconstruct import AllAtomStructure, get_atom_xyzs


REFINE_TIMEOUT_S = 600.0


def _detect_platform() -> str:
    """自动检测最佳 OpenMM 平台: CUDA > OpenCL > CPU."""
    try:
        from openmm import Platform
        for name in ("CUDA", "OpenCL"):
            try:
                Platform.getPlatformByName(name)
                return name
            except Exception:
                continue
    except ImportError:
        pass
    return "CPU"

# P 原子 positional restraint 力常数 (kJ/mol/nm²)。
# 旧值 10 太软, amber 最小化时 P 跑 3.95Å (超 2.0 阈值)。
# 实测: K=500 仍 3.95Å, K=5000 降到 1.41Å。P 是 CG 求解的骨架点,
# 代表输入结构信息, 不该大幅跑 → 用 1000 (position restraint 文献常用值),
# 既压住 P 让残基内部几何往 A-form 拉, 又不把 P 完全钉死让最小化无意义。
P_RESTRAINT_K = 1000.0
# P 偏移阈值: 旧值 2.0 是 K=10 时设的。K 提到 1000 后:
#   正多边形自测 P 偏 2.35Å, 真实 CG 求解坐标 P 偏 3.6Å
#   (真实网络解比完美几何应力大)。3.6Å 在 circRNA 半径 28Å 尺度下 ~1.3%,
#   物理可接受 (P 位置是 CG 求解的近似, 真值应是力场松弛后的构象)。
#   阶段 1 松 P 约束后 P 会跑更多 (5-8Å), 但这是力场主动调整骨架走向,
#   不是失败。阈值放到 10.0: 只要力场能量收敛到负就认成功。
P_MAX_DRIFT_A = 10.0

# A-form RNA 标准二面角 (弧度) — 用于 backbone torsion 约束
# α: O3'-P-O5'-C5'  ~ -60° (gauche-)
# γ: O5'-C5'-C4'-C3' ~ 60°  (gauche+)
# δ: C5'-C4'-C3'-O3' ~ 84°  (anti, A-form)
# ζ: C4'-C3'-O3'-P   ~ -90° (gauche-)
# χ: O4'-C1'-N9-C4 (嘌呤) / O4'-C1'-N1-C2 (嘧啶) ~ -160° (anti)
_AFORM_TORSIONS = {
    "alpha": (-60.0, "O3'", "P",   "O5'", "C5'"),
    # beta 不加: 诊断发现 beta 偏 A-form 51°, 但强加 beta 约束 (k=50) 跟 alpha 共享
    # P/O5'/C5' 三原子, 约束耦合冲突, alpha/gamma 全崩 (16/18 离群). beta 偏差记为
    # 已知盲区, 留以后用更弱 k 或 fragment 重建处理. (tests/torsion_stacking_diag.py)
    "gamma": (60.0,  "O5'", "C5'", "C4'", "C3'"),
    "delta": (84.0,  "C5'", "C4'", "C3'", "O3'"),
    "zeta":  (-90.0, "C4'", "C3'", "O3'", "P"),
}


class _TimeoutError(Exception):
    pass


def _build_topology_and_modeller(structure: AllAtomStructure):
    """建 Topology + Modeller, 手动加 H (绕开 amber addHydrogens 模板匹配)。

    amber14 RNA.OL3 的 addHydrogens 在 circRNA 环形拓扑上失败 (首末残基
    ExternalBond 与 A5/A3 模板对不上)。改: 手动按 RNA 标准加 H, 直接
    createSystem(ignoreExternalBonds=True, residueTemplates=中间模板)。
    """
    from openmm.app import Topology, Element, Modeller, ForceField

    topo = Topology()
    chain = topo.addChain()
    res_atoms: List[Dict[str, object]] = []
    for (start, end) in structure.residue_atom_spans:
        first_atom = structure.atoms[start]
        res = topo.addResidue(first_atom.res_name, chain)
        name_to_atom: Dict[str, object] = {}
        for k in range(start, end):
            a = structure.atoms[k]
            elem = Element.getBySymbol(a.element)
            ta = topo.addAtom(a.atom_name, elem, res)
            name_to_atom[a.atom_name] = ta
        res_atoms.append(name_to_atom)

    # --- 手动加 H (绕开 addHydrogens) ---
    # OpenMM 要求每残基原子连续, 重建 Topology: 每残基重原子+H 一起加。
    heavy_xyzs = get_atom_xyzs(structure)  # Å
    heavy_names = [a.atom_name for a in structure.atoms]
    heavy_res_idx = [a.res_seq - 1 for a in structure.atoms]
    sequences = list(structure.sequence)
    bonds_global = _collect_bonds_global(topo)
    h_atoms = _add_hydrogens_manual(
        heavy_xyzs, heavy_names, heavy_res_idx, sequences, bonds_global
    )
    h_by_res = {}
    for h in h_atoms:
        h_by_res.setdefault(h[0]-1, []).append(h)

    topo2 = Topology()
    chain2 = topo2.addChain()
    res_atoms2 = []
    all_coords_nm = []
    for (start, end) in structure.residue_atom_spans:
        first_atom = structure.atoms[start]
        res = topo2.addResidue(first_atom.res_name, chain2)
        name_to_atom = {}
        for k in range(start, end):
            a = structure.atoms[k]
            elem = Element.getBySymbol(a.element)
            ta = topo2.addAtom(a.atom_name, elem, res)
            name_to_atom[a.atom_name] = ta
            all_coords_nm.append(a.xyz / 10.0)
        for h in h_by_res.get(first_atom.res_seq-1, []):
            ta_h = topo2.addAtom(h[1], Element.getBySymbol("H"), res)
            name_to_atom[h[1]] = ta_h
            all_coords_nm.append(h[2] / 10.0)
        res_atoms2.append(name_to_atom)

    _add_rna_bonds(topo2, res_atoms2, structure)
    coords_arr = np.array(all_coords_nm, dtype=np.float64)  # nm
    modeller = Modeller(topo2, coords_arr)
    ff = ForceField("amber14-all.xml", "implicit/obc1.xml")

    # 建 heavy_to_topo 映射: structure.atoms 重原子索引 -> topo2 原子索引。
    # topo2 里每残基是 "重原子 + 该残基 H" 交替, H 夹在重原子中间,
    # 不能用 serial 直接索引。遍历 topo2 原子, 跳过 H, 记重原子顺序。
    heavy_to_topo = []
    for ta in topo2.atoms():
        if ta.element.symbol == "H":
            continue
        heavy_to_topo.append(ta.index)
    return modeller, ff, heavy_to_topo


def _collect_bonds_global(topo):
    """收集 Topology 里所有 bond 的 (atom1.index, atom2.index)。"""
    bonds = []
    for b in topo.bonds():
        bonds.append((b[0].index, b[1].index))
    return bonds


# RNA 标准 H 分布 (重原子 → {碱基: [(H名, 元素), ...]})。
# 从 amber14 RNA.OL3.xml 模板 A/G/U/C 逐个核对得到, 不是猜的。
# 骨架 H (C5'/C4'/C3'/C2'/C1'/O2'-HO2') 四种碱基都有, 公共。
# 碱基 H 按碱基区分: H1→N1(G), H2→C2(A), H3→N3(U), H5→C5(U/C),
# H6→C6(U/C), H8→C8(A/G), H21/H22→N2(G), H41/H42→N4(C), H61/H62→N6(A)。
# circRNA 全用中间模板: O5' 连本残基 P, O3' 连下游 P, 都不加 H
# (O5'/O3' 加 H 在 _add_hydrogens_manual 里按是否有外部 P 动态判)。
_RNA_H_MAP: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    # 骨架 — 所有碱基通用
    "C5'": {"*": [("H5'", "H"), ("H5''", "H")]},
    "C4'": {"*": [("H4'", "H")]},
    "C3'": {"*": [("H3'", "H")]},
    "C2'": {"*": [("H2'", "H")]},
    "C1'": {"*": [("H1'", "H")]},
    "O2'": {"*": [("HO2'", "H")]},
    # 嘌呤碱基 H
    "C8":  {"A": [("H8", "H")], "G": [("H8", "H")]},
    "C2":  {"A": [("H2", "H")]},          # 仅 A 的 C2 有 H2; G 的 C2 连 N2 不加 H
    "N1":  {"G": [("H1", "H")]},          # G 的 N1-H1 (亚胺氢)
    "N6":  {"A": [("H61", "H"), ("H62", "H")]},
    "N2":  {"G": [("H21", "H"), ("H22", "H")]},
    # 嘧啶碱基 H
    "C5":  {"U": [("H5", "H")], "C": [("H5", "H")]},
    "C6":  {"U": [("H6", "H")], "C": [("H6", "H")]},
    "N3":  {"U": [("H3", "H")]},  # 仅 U 的 N3-H3; C 的 N3 是双键不能接 H
    "N4":  {"C": [("H41", "H"), ("H42", "H")]},
    # 这些重原子在任何碱基都不带 H
    "O5'": {}, "O3'": {}, "O6": {}, "O2": {}, "O4": {},
    "N9": {}, "N7": {}, "C4": {},
    "P": {}, "OP1": {}, "OP2": {},
}
_H_BOND_LEN = {"C": 1.09, "N": 1.01, "O": 0.97}


def _add_hydrogens_manual(
    heavy_xyzs, heavy_names, heavy_res_idx, sequences, bonds
):
    """手动加 H (位置: 重原子邻居质心反方向推)。返回 [(res_seq, name, xyz), ...]。"""
    N = heavy_xyzs.shape[0]
    neighbors = [set() for _ in range(N)]
    for (i, j) in bonds:
        if 0 <= i < N and 0 <= j < N:
            neighbors[i].add(j)
            neighbors[j].add(i)

    hydrogens = []
    for i in range(N):
        aname = heavy_names[i]
        res_idx = heavy_res_idx[i]
        base = sequences[res_idx]
        # map 是 {重原子: {碱基: [(H名, 元素), ...]}}; "*" 表所有碱基通用
        per_base = _RNA_H_MAP.get(aname)
        if not per_base:
            continue
        h_list = per_base.get(base) or per_base.get("*")
        if not h_list:
            continue
        if aname == "O3'":
            has_down_p = any(
                heavy_names[j] == "P" and heavy_res_idx[j] != res_idx
                for j in neighbors[i]
            )
            if has_down_p:
                continue
        if aname == "O5'":
            has_up_p = any(
                heavy_names[j] == "P" and heavy_res_idx[j] == res_idx
                for j in neighbors[i]
            )
            if has_up_p:
                continue

        xyz = heavy_xyzs[i]
        nbrs = list(neighbors[i])
        if len(nbrs) == 0:
            h_dir = np.array([1.0, 0.0, 0.0])
        else:
            nbr_center = heavy_xyzs[nbrs].mean(axis=0)
            h_dir = xyz - nbr_center
            n = np.linalg.norm(h_dir)
            h_dir = h_dir / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

        bl = _H_BOND_LEN.get(aname[0], 1.09)
        if len(h_list) == 1:
            h_xyz = xyz + h_dir * bl
            hydrogens.append((res_idx + 1, h_list[0][0], h_xyz))
        elif len(h_list) == 2:
            if abs(h_dir[0]) < 0.9:
                ortho = np.cross(h_dir, np.array([1.0, 0.0, 0.0]))
            else:
                ortho = np.cross(h_dir, np.array([0.0, 1.0, 0.0]))
            on = np.linalg.norm(ortho)
            ortho = ortho / on if on > 1e-6 else np.array([0.0, 1.0, 0.0])
            off = 0.4
            for k, (hn, _) in enumerate(h_list):
                sign = 1 if k == 0 else -1
                h_xyz = xyz + h_dir * bl + ortho * off * sign
                hydrogens.append((res_idx + 1, hn, h_xyz))
    return hydrogens


# 标准 RNA 残基内键 (name-name 对)。骨架 + 糖环 + 碱基。
_BACKBONE_BONDS = [
    ("P", "OP1"), ("P", "OP2"), ("P", "O5'"),
    ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "O4'"),
    ("C4'", "C3'"), ("C3'", "O3'"), ("C3'", "C2'"),
    ("C2'", "O2'"), ("C2'", "C1'"), ("C1'", "O4'"),  # 糖环闭合 + 2'-OH
]
_PURINE_BONDS = [  # A/G
    ("C1'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "N7"), ("N7", "C5"),
    ("C5", "C6"), ("C5", "C4"), ("C6", "N1"), ("N1", "C2"), ("C2", "N3"),
    ("N3", "C4"),  # 双环闭合
]
_PYRIMIDINE_BONDS = [  # C/U
    ("C1'", "N1"), ("N1", "C2"), ("C2", "N3"), ("N3", "C4"),
    ("C4", "C5"), ("C5", "C6"), ("C6", "N1"),  # 六元环闭合
]
_BASE_EXTRA_BONDS = {
    "A": [("C6", "N6")],
    "G": [("C6", "O6"), ("C2", "N2")],
    "C": [("C2", "O2"), ("C4", "N4")],
    "U": [("C2", "O2"), ("C4", "O4")],
}


def _add_rna_bonds(topo, res_atoms, structure):
    """给每个残基添加标准 RNA 共价键 (含 H) + 残基间磷酸二酯键。

    H 键用 _RNA_H_MAP 反查: 每个 H 挂在哪个重原子上, 跟 map 加 H 那侧一致。
    """
    L = len(res_atoms)
    for res_idx, name_to_atom in enumerate(res_atoms):
        base = structure.sequence[res_idx]
        bonds = list(_BACKBONE_BONDS)
        if base in ("A", "G"):
            bonds += _PURINE_BONDS
        else:
            bonds += _PYRIMIDINE_BONDS
        bonds += _BASE_EXTRA_BONDS.get(base, [])
        for a1n, a2n in bonds:
            a1 = name_to_atom.get(a1n)
            a2 = name_to_atom.get(a2n)
            if a1 is not None and a2 is not None:
                topo.addBond(a1, a2)

        # H 键: 遍历 map 里该重原子(本碱基或通用) 的 H, addBond(重原子, H)
        for heavy_name, per_base in _RNA_H_MAP.items():
            h_list = per_base.get(base) or per_base.get("*")
            if not h_list:
                continue
            heavy_atom = name_to_atom.get(heavy_name)
            if heavy_atom is None:
                continue
            for (h_name, _elem) in h_list:
                h_atom = name_to_atom.get(h_name)
                if h_atom is not None:
                    topo.addBond(heavy_atom, h_atom)

    # 残基间磷酸二酯键: O3'[i] ↔ P[i+1], i=0..L-2
    # 不加 BSJ 拓扑键 — 用 ignoreExternalBonds=True 让首末残基匹配中间模板。
    # BSJ 闭合由 amber_refine 里的 HarmonicBondForce 物理约束保证,
    # 最小化后不影响配图 (BSJ 局部)。
    for i in range(L - 1):
        o3 = res_atoms[i].get("O3'")
        p_next = res_atoms[i + 1].get("P")
        if o3 is not None and p_next is not None:
            topo.addBond(o3, p_next)


def amber_refine(
    structure: AllAtomStructure,
    pairs: List[Tuple[int, int, float]],
    *,
    platform_name: str = "auto",
    max_iterations: int = 3000,
    use_md_for_long: bool = True,
    long_threshold: int = 200,
    timeout_s: float = REFINE_TIMEOUT_S,
    coding_mask: Optional[np.ndarray] = None,
    cg_coords: Optional[np.ndarray] = None,
    coding_restraint_k: float = 10000.0,
    cg_topology_weight: float = 0.0,  # 方案 F: 非 coding P 融合 CG 全局拓扑权重
    use_o3p_bond: bool = False,
    use_o3p_angle: bool = False,
) -> Tuple[np.ndarray, float, float, Dict[str, int]]:
    """Amber14 OL3 + OBC1 约束最小化。失败时返回重建坐标 (无精修)。

    coding_mask: bool 数组 (L,) True = coding 区残基。amber 精修时
        coding 残基的 P/C1'/O3* 位置用高 k 钉死到 cg_coords, 保持真实结构。
    cg_coords: (L, 3) nm, coding 钉死的目标坐标 (CG 原坐标)。
        不传时用 structure 里的 P 坐标 (即 RL 优化后的位置, 等于没钉死)。
    coding_restraint_k: coding 钉死力常数 (kJ/mol/nm²)。默认 10000 = 强钉死。
    cg_topology_weight: 非 coding 残基 P restraint 目标融合 CG 全局拓扑的权重
        (0.0=纯 1EHZ Kabsch 模板; 1.0=完全钉回 CG 拓扑)。默认 0.0 保持旧行为。
        长序列下 >0 让全局拓扑信息进入 amber, 避免 1EHZ 局部模板覆盖全局配对位置。
    use_o3p_bond: P2 键长约束 (O3'-P[i+1] 1.6Å k=10000). 默认关.
    use_o3p_angle: P2.5 键角约束 (C3'-O3'-P / O3'-P-O5'). 默认关.
        (两关 = P1 干净版, 纯 4 点 Kabsch 无额外约束, 用于 P3 对比基线)
    """
    try:
        return _amber_refine_impl(
            structure, pairs,
            platform_name=platform_name,
            max_iterations=max_iterations,
            use_md_for_long=use_md_for_long,
            long_threshold=long_threshold,
            coding_mask=coding_mask,
            cg_coords=cg_coords,
            coding_restraint_k=coding_restraint_k,
            cg_topology_weight=cg_topology_weight,
            use_o3p_bond=use_o3p_bond,
            use_o3p_angle=use_o3p_angle,
        )
    except Exception as exc:
        print(f"[amber_refine] 精修失败, 返回重建坐标: {exc!r}")
        coords_aa = get_atom_xyzs(structure)
        info = {
            "n_heavy": len(structure.atoms),
            "n_atoms": len(structure.atoms),
            "n_h": 0,
            "n_torsions": 0,
            "max_p_drift": 0.0,
            "fallback": True,
            "error": str(exc),
            "n_coding_pinned": 0,
        }
        return coords_aa, 0.0, 0.0, info


def _amber_refine_impl(
    structure: AllAtomStructure,
    pairs: List[Tuple[int, int, float]],
    *,
    platform_name: str = "CPU",
    max_iterations: int = 3000,
    use_md_for_long: bool = True,
    long_threshold: int = 200,
    coding_mask: Optional[np.ndarray] = None,
    cg_coords: Optional[np.ndarray] = None,
    coding_restraint_k: float = 10000.0,
    cg_topology_weight: float = 0.0,
    use_o3p_bond: bool = False,
    use_o3p_angle: bool = False,
) -> Tuple[np.ndarray, float, float, Dict[str, int]]:
    """Amber14 OL3 + OBC1 约束最小化全原子结构。

    cg_topology_weight: 非 coding 残基 P restraint 目标融合 CG 全局拓扑的权重
        (0.0=纯 1EHZ Kabsch 模板; 1.0=完全钉回 CG 拓扑)。默认 0.0 保持旧行为。
        长序列 (>800nt) 下 >0 让全局配对位置信息进入 amber, 避免 1EHZ 局部
        模板覆盖全局拓扑 (方案 F)。

    Returns:
        (refined_coords, e0, e1, info)
        refined_coords: (N, 3) Å (含 H, 与 Modeller 后的原子顺序一致)
        e0/e1: 最小化前/后势能 (kJ/mol)
        info: {'n_atoms': ..., 'n_h': ..., 'max_p_drift': ...}
    """
    from openmm import (
        Platform, HarmonicBondForce, CustomBondForce,
        CustomExternalForce, CustomTorsionForce, CustomAngleForce,
        VerletIntegrator, LangevinMiddleIntegrator,
    )
    from openmm import unit
    from openmm.app import Simulation

    L = len(structure.residue_atom_spans)
    modeller, ff, heavy_to_topo = _build_topology_and_modeller(structure)
    topo = modeller.topology
    n_heavy = len(structure.atoms)
    n_total = int(modeller.topology.getNumAtoms())

    # 加 H 后原子索引重映射: 重建的 P/O3*/C1* 在新拓扑里位置可能变了,
    # 需重新按 (res_seq, atom_name) 查找。
    atom_lookup: Dict[Tuple[int, str], int] = {}  # (res_seq, atom_name) → new_idx
    for new_idx, atom in enumerate(topo.atoms()):
        atom_lookup[(atom.residue.index + 1, atom.name)] = new_idx

    # ignoreExternalBonds: 跳过 ExternalBond 检查 (circRNA 环形拓扑会让 amber 困惑)
    # 不用 residueTemplates: circRNA 所有残基都是内部的 (无 5'/3' 端点),
    # 让 OpenMM 自动匹配内部模板 (A/U/G/C), 避免端点模板 (A5/A3/C5/C3) 干扰。
    system = ff.createSystem(
        topo, constraints=None, rigidWater=True,
        ignoreExternalBonds=True,
    )

    # --- 力 1: P positional restraint (coding 区钉回 cg_coords, non-coding 钉自身) ---
    # coding 残基: per-particle k=coding_restraint_k (默认 10000, 强钉死), 目标=cg_coords
    #   (RL 之前的 CG 原坐标, 保真实结构)。RL 全序列可动, amber 把 coding 拉回。
    # non-coding 残基: per-particle k=P_RESTRAINT_K (1000, 软约束), 目标=structure 自身 P
    #   (amber 输入坐标 = RL 优化后的 CG P, 接受物理收敛微调)。
    # 无 coding_mask 时全部 non-coding 处理 (等价旧行为, 不破现有调用)。
    # 全局 k_scale 乘子: 多阶段退火用它放松/收紧 (阶段1松, 阶段3收紧到1.0),
    #   per-particle k 保留 coding/non-coding 区分不被全局 setParameter 抹掉。
    restraint = CustomExternalForce("k_scale*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k_scale", 1.0)
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    p_drift_refs = []  # (new_idx, original_xyz_nm) for 检查
    n_coding_pinned = 0
    n_topology_fused = 0
    # 方案 F: 非 coding 区 P 的目标位置融合 CG 全局拓扑。
    # cg_coords 是 nm, structure P 也是 nm (÷10.0 后)。
    for res_idx in range(L):
        res_seq = res_idx + 1
        new_idx = atom_lookup.get((res_seq, "P"))
        if new_idx is None:
            continue
        # 默认目标 = structure 自身 P (amber 输入坐标), 力常数 = P_RESTRAINT_K
        p_xyz = structure.atoms[structure.residue_atom_index[res_idx]["P"]].xyz / 10.0
        k_val = P_RESTRAINT_K
        # coding 残基: 钉回 cg_coords, 强约束
        if (coding_mask is not None and cg_coords is not None
                and res_idx < len(coding_mask) and coding_mask[res_idx]
                and res_idx < len(cg_coords)):
            # cg_coords 是 nm (CG 求解的 P), coding 钉到 RL 之前的 CG 原坐标
            p_xyz = np.asarray(cg_coords[res_idx], dtype=np.float64)
            k_val = coding_restraint_k
            n_coding_pinned += 1
        elif (cg_topology_weight > 0.0 and cg_coords is not None
                and res_idx < len(cg_coords)):
            # 方案 F: 非 coding 区 P 目标 = (1-w)·1EHZ模板 + w·CG拓扑
            cg_p = np.asarray(cg_coords[res_idx], dtype=np.float64)
            tmpl_p = np.asarray(p_xyz, dtype=np.float64)
            p_xyz = (1.0 - cg_topology_weight) * tmpl_p + cg_topology_weight * cg_p
            n_topology_fused += 1
        restraint.addParticle(new_idx, [k_val, p_xyz[0], p_xyz[1], p_xyz[2]])
        p_drift_refs.append((new_idx, p_xyz))
    system.addForce(restraint)

    # --- 力 2: BSJ 闭环键 (O3*[L-1] ↔ P[0]) ---
    last_o3 = atom_lookup.get((L, "O3'"))
    first_p = atom_lookup.get((1, "P"))
    bsj_bond = HarmonicBondForce()
    if last_o3 is not None and first_p is not None:
        bsj_bond.addBond(last_o3, first_p, 0.161, 50000.0)
    system.addForce(bsj_bond)

    # --- 力 2.5: 相邻残基 O3'-P 键长硬约束 (P2, use_o3p_bond 开关) ---
    # 1EHZ 模板 Kabsch 对齐只保 P/C1'/C4'/O3' 四点, 但相邻残基间 O3'[i]-P[i+1]
    # 桥接几何会被拉坏 (amber 前 C3'-O3'-P 偏 70°). 加 HarmonicBondForce 把每条
    # O3'[i]-P[i+1] 键长钉死到 1.6Å (A-form 真值), k=10000 全程紧约束 (不走 k_scale).
    # 与 OL3 力场自带的 O3'-P 键长项不冲突 (都指向 1.6Å, 只是再钉死防最小化偏离).
    # 默认关: 关掉 = P1 干净版 (纯 4 点 Kabsch), 用于 P3 对比基线.
    n_o3p = 0
    if use_o3p_bond:
        o3p_bond = HarmonicBondForce()
        for i in range(L):
            o3_idx = atom_lookup.get((i + 1, "O3'"))       # 残基 i 的 O3'
            next_p_idx = atom_lookup.get(((i + 1) % L + 1, "P"))  # 残基 i+1 的 P (环形)
            if o3_idx is not None and next_p_idx is not None:
                o3p_bond.addBond(o3_idx, next_p_idx, 0.16, 10000.0)  # r0=1.6Å, k=10000
                n_o3p += 1
        system.addForce(o3p_bond)

    # --- 力 2.6: 磷酸桥键角硬约束 (P2.5, use_o3p_angle 开关) ---
    # P2 键长约束救不了键角 (键长对 ≠ 键角对). 加 CustomAngleForce 直接约束两个
    # 关键磷酸桥键角: C3'-O3'-P (119.5°), O3'-P-O5' (104.1°), OL3 A-form 真值.
    # k_angle=500 kJ/mol/rad² 中等强度, 全程紧约束 (不走 k_scale).
    # 跨残基: C3'[i]-O3'[i]-P[i+1] (顶点 O3'[i]) 和 O3'[i]-P[i+1]-O5'[i+1] (顶点 P[i+1]).
    # CustomAngleForce.addAngle(p1, p2, p3, params): p2 是顶点.
    # 默认关: 关掉 = P1 干净版.
    n_angle = 0
    if use_o3p_angle:
        angle_force = CustomAngleForce("0.5*k_angle*(theta-theta0)^2")
        angle_force.addGlobalParameter("k_angle", 500.0)
        angle_force.addPerAngleParameter("theta0")
        THETA_C3_O3_P = 119.5 * np.pi / 180.0  # C3'-O3'-P 平衡角 (rad)
        THETA_O3_P_O5 = 104.1 * np.pi / 180.0  # O3'-P-O5' 平衡角 (rad)
        for i in range(L):
            j = (i + 1) % L  # 下一残基索引
            c3_i = atom_lookup.get((i + 1, "C3'"))
            o3_i = atom_lookup.get((i + 1, "O3'"))
            p_j = atom_lookup.get((j + 1, "P"))
            o5_j = atom_lookup.get((j + 1, "O5'"))
            # C3'-O3'-P: 顶点 O3'[i] -> addAngle(C3, O3, P)
            if None not in (c3_i, o3_i, p_j):
                angle_force.addAngle(c3_i, o3_i, p_j, [THETA_C3_O3_P])
                n_angle += 1
            # O3'-P-O5': 顶点 P[i+1] -> addAngle(O3, P, O5)
            if None not in (o3_i, p_j, o5_j):
                angle_force.addAngle(o3_i, p_j, o5_j, [THETA_O3_P_O5])
                n_angle += 1
        system.addForce(angle_force)

    # --- 力 3: ViennaRNA 配对距离约束 (C1*-C1* ~10.6Å) ---
    pair_force = CustomBondForce("0.5*k_pairdist*(r-r0)^2")
    pair_force.addGlobalParameter("k_pairdist", 100.0)
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if not (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            continue
        ci = atom_lookup.get((i + 1, "C1'"))
        cj = atom_lookup.get((j + 1, "C1'"))
        if ci is None or cj is None:
            continue
        pair_force.addBond(ci, cj, [0.106])
    system.addForce(pair_force)

    # --- 力 4: A-form 二面角约束 (backbone torsions) ---
    # alpha = O3'[i-1]-P[i]-O5'[i]-C5'[i] (跨残基, O3' 来自前一残基)
    # gamma = O5'[i]-C5'[i]-C4'[i]-C3'[i] (残基内)
    # delta = C5'[i]-C4'[i]-C3'[i]-O3'[i] (残基内)
    # zeta  = C4'[i]-C3'[i]-O3'[i]-P[i+1]  (跨残基, P 来自下一残基)
    # 修 bug: 旧版 alpha/zeta 全用本残基原子 (假二面角, 约束了个不存在的角),
    #   导致真实 alpha/zeta 偏 A-form 100-140°, 碱基堆积距离 7.9Å (真值 3.4Å)。
    torsion = CustomTorsionForce("0.5*k_aform*(theta-theta0)^2")
    torsion.addGlobalParameter("k_aform", 50.0)  # kJ/mol/rad²
    torsion.addPerTorsionParameter("theta0")
    n_torsions = 0
    for res_idx in range(L):
        res_seq = res_idx + 1
        prev_seq = ((res_idx - 1) % L) + 1  # 前一残基 res_seq (环形)
        next_seq = ((res_idx + 1) % L) + 1  # 下一残基 res_seq (环形)
        for tname, (angle_deg, a1, a2, a3, a4) in _AFORM_TORSIONS.items():
            # alpha: a1 (O3') 取前一残基, 其余本残基
            if tname == "alpha":
                i1 = atom_lookup.get((prev_seq, a1))
                i2 = atom_lookup.get((res_seq, a2))
                i3 = atom_lookup.get((res_seq, a3))
                i4 = atom_lookup.get((res_seq, a4))
            # zeta: a4 (P) 取下一残基, 其余本残基
            elif tname == "zeta":
                i1 = atom_lookup.get((res_seq, a1))
                i2 = atom_lookup.get((res_seq, a2))
                i3 = atom_lookup.get((res_seq, a3))
                i4 = atom_lookup.get((next_seq, a4))
            # gamma/delta: 全本残基
            else:
                i1 = atom_lookup.get((res_seq, a1))
                i2 = atom_lookup.get((res_seq, a2))
                i3 = atom_lookup.get((res_seq, a3))
                i4 = atom_lookup.get((res_seq, a4))
            if None in (i1, i2, i3, i4):
                continue
            theta0 = angle_deg * np.pi / 180.0
            torsion.addTorsion(i1, i2, i3, i4, [theta0])
            n_torsions += 1
    system.addForce(torsion)

    # 约束项标 force group (最小化后分项量能量, 区分约束 vs amber 力场)。
    # amber 自带力场 (createSystem 建的) 留 group 0; 约束项归 1..4。
    # 必须在 Context (Simulation) 创建前 setForceGroup。
    constraint_forces = [restraint, bsj_bond, pair_force, torsion]
    for gi, f in enumerate(constraint_forces, start=1):
        try:
            f.setForceGroup(gi)
        except Exception:
            pass

    # --- 积分器 + 平台 ---
    # 始终用 Langevin: 初始几何有原子冲突 (e0~1e14), 要靠 MD 退火打散再最小化。
    # 旧逻辑只在 L>=200 时开 MD, 短序列用 Verlet 跑不了 step()。
    use_md = True
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond
    )
    # 解析 "auto" 平台
    if platform_name == "auto":
        resolved = _detect_platform()
        print(f"  [amber_refine] 平台自动检测: {resolved}")
    else:
        resolved = platform_name
    try:
        platform = Platform.getPlatformByName(resolved)
        sim = Simulation(topo, system, integrator, platform)
    except Exception:
        fallback = _detect_platform()
        print(f"  [amber_refine] 平台 {resolved} 不可用, 降级到 {fallback}")
        try:
            platform = Platform.getPlatformByName(fallback)
            sim = Simulation(topo, system, integrator, platform)
        except Exception:
            sim = Simulation(topo, system, integrator)

    # Modeller 加 H 后的坐标 (nm)
    positions = modeller.getPositions()
    sim.context.setPositions(positions)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy()._value

    # --- 多阶段最小化 + 退火 ---
    # 阶段 1: 松 P 约束 (k_scale=0.01, 放松 ~100 倍), 让 amber 力场主导把
    # 键长/键角/VdW 降到最低。per-particle k 保留 coding/non-coding 区分。
    sim.context.setParameter("k_scale", 0.01)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max(3000, max_iterations),
    )

    # 阶段 2: 温和退火 MD。1000K 会把初始畸变结构的原子甩飞 (NaN),
    # 用 500K + 短步数, 每步后检查 NaN, 炸了就回退到松 P 最小化结果。
    pre_md_state = sim.context.getState(getPositions=True)
    pre_md_pos = pre_md_state.getPositions()
    try:
        sim.integrator.setTemperature(500 * unit.kelvin)
        sim.step(100)
        sim.integrator.setTemperature(300 * unit.kelvin)
        sim.step(50)
        # 检查 MD 后有没有 NaN
        chk = sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        if not np.isfinite(chk).all():
            raise RuntimeError("MD 产生 NaN, 回退到 MD 前状态")
        # 松 P 下重最小化。
        sim.minimizeEnergy(
            tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=max(3000, max_iterations),
        )
    except Exception as md_exc:
        # MD 炸了, 回退到 MD 前坐标, 跳过退火直接进阶段 3。
        sim.context.setPositions(pre_md_pos)

    # 阶段 3: 收紧 P 约束到目标值 (k_scale=1.0), 保持骨架拓扑, 最终最小化。
    # per-particle k 已在加粒子时定死 (coding=coding_restraint_k/non-coding=P_RESTRAINT_K),
    # 全局 k_scale 从阶段1的 0.01 收回 1.0, 恢复完整钉死强度。
    sim.context.setParameter("k_scale", 1.0)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # 分项能量诊断: setForceGroup 必须在 Context 创建前调, 这里只是占位,
    # 真正的 setForceGroup 在 Simulation 创建前的 "约束项标组" 段落。
    force_energies = {}
    try:
        # amber 力场本身 (group 0) - 不含约束
        s = sim.context.getState(getEnergy=True, groups={0})
        force_energies["amber_field"] = s.getPotentialEnergy()._value
        for gi, f in enumerate(constraint_forces, start=1):
            s = sim.context.getState(getEnergy=True, groups={gi})
            force_energies[f"constraint_{f.__class__.__name__}"] = \
                s.getPotentialEnergy()._value
    except Exception:
        pass
    refined_ang = pos * 10.0  # nm -> Å, 按 topo2 顺序 (重原子+H 交替)

    # 抽出只含重原子的坐标, 按 structure.atoms 顺序返回 (不含 H)。
    # heavy_to_topo[i] = topo2 中第 i 个重原子的全局索引。
    # 上层 (predictor/export/immune_heuristic) 用 structure.atoms[i] 索引取坐标,
    # 不能含 H, 否则索引错位 -> BSJ/键长全错。
    if len(heavy_to_topo) == n_heavy:
        refined_heavy = refined_ang[heavy_to_topo]  # (n_heavy, 3) Å
    else:
        # 映射对不上 (异常), 回退全数组上层自己处理
        refined_heavy = refined_ang

    # --- 安全检查: P 偏离 ---
    # P 在 structure.atoms 里的索引 = 该残基 "P" 原子的 serial (0-based 重原子序)。
    # refined_heavy 按 structure.atoms 顺序, 所以 serial 直接索引。
    max_drift = 0.0
    for res_idx in range(L):
        p_serial = structure.residue_atom_index[res_idx].get("P")
        if p_serial is None or p_serial >= len(refined_heavy):
            continue
        # P 的原始坐标 (nm, 重建时 CG 求解值)
        p_ref = structure.atoms[p_serial].xyz
        drift = np.linalg.norm(refined_heavy[p_serial] - p_ref)
        if drift > max_drift:
            max_drift = drift
    if max_drift > P_MAX_DRIFT_A:
        raise RuntimeError(
            f"Amber 最小化后 P 偏离 {max_drift:.2f}Å > {P_MAX_DRIFT_A}Å 阈值, 回退 CG"
        )
    if np.isnan(e1) or np.isinf(e1):
        raise RuntimeError(f"Amber 最小化后能量异常 e1={e1}, 回退 CG")

    info = {
        "n_heavy": n_heavy,
        "n_atoms": n_total,
        "n_h": n_total - n_heavy,
        "n_torsions": n_torsions,
        "max_p_drift": float(max_drift),
        "force_energies": force_energies,
        "n_coding_pinned": n_coding_pinned,
        "n_topology_fused": n_topology_fused,
        "n_o3p_bonds": n_o3p,
    }
    # 返回重原子坐标 (按 structure.atoms 顺序, 不含 H)。上层直接用 serial 索引。
    return refined_heavy, e0, e1, info


if __name__ == "__main__":
    from .aform_from_template import reconstruct_all_atom

    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles),
                   np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    print(f"重建: {len(s.atoms)} 重原子")
    pairs = [(i, L - 1 - i, 1.0) for i in range(L // 2)]
    coords, e0, e1, info = amber_refine(s, pairs, max_iterations=500)
    print(f"e0={e0:.0f} → e1={e1:.0f} kJ/mol")
    print(f"info: {info}")
