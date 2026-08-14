"""
openmm_gpu_refiner.py — OpenMM GPU 加速 CG MD 精修

替代 IsRNAcirc.exe 的 CPU-only CG MD 精修。
用 OpenMM 3-bead CG 力场 + GPU 平台加速 + 可选 REMD 增强采样。

接口兼容 isrnacirc_wrapper.isrnacirc_cg_refine():
  openmm_gpu_refine(input_pdb, output_dir, sequence, secondary_structure, ...)
  -> (output_pdb_path, final_energy)

回退链: CUDA -> OpenCL -> CPU
增强采样: 可选 T-REMD (Replica Exchange)

作者: TorusFold Team
日期: 2026-08-05
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import Simulation, Topology, Element
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    mm = None
    app = None
    unit = None


# ── 平台检测 ──

def detect_best_platform(preferred: str = "auto") -> str:
    """检测最佳可用 OpenMM 平台.

    preferred="auto" 时按 CUDA > OpenCL > CPU 顺序探测.
    preferred="CUDA"/"OpenCL"/"CPU" 时直接尝试该平台, 失败回退.

    Args:
        preferred: 首选平台 ("auto", "CUDA", "OpenCL", "CPU")

    Returns:
        可用平台名称
    """
    if not OPENMM_AVAILABLE:
        return "CPU"

    if preferred == "auto":
        # 跳过 OpenCL (Windows 上 LLVM JIT 可能报 "Can't get available size")
        candidates = ["CUDA", "CPU"]
    elif preferred == "OpenCL":
        # 显式请求 OpenCL 时才尝试
        candidates = ["OpenCL", "CPU"]
    else:
        candidates = [preferred, "CPU"]

    for name in candidates:
        try:
            Platform.getPlatformByName(name)
            # 对 OpenCL 做快速测试 (创建空系统), 失败则跳过
            if name == "OpenCL":
                try:
                    test_sys = mm.System()
                    test_sys.addParticle(1.0)
                    test_int = mm.LangevinMiddleIntegrator(
                        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
                    test_sim = app.Simulation(
                        app.Topology(), test_sys, test_int,
                        Platform.getPlatformByName("OpenCL"))
                except Exception:
                    continue
            return name
        except Exception:
            continue
    return "CPU"


# ── PDB 坐标读写 ──

def _read_p_coords(pdb_path: str) -> np.ndarray:
    """从 PDB 读取 P 原子坐标, 返回 (L,3) Å.

    优先按列解析 (标准 PDB 格式), 列错位时回退 whitespace split.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    # 列错位 (坐标溢出等), 回退 split
                    parts = line.split()
                    # ATOM serial name resname chain resid x y z ...
                    x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float64)


def _write_allatom_pdb(
    p_coords_3bead_nm: np.ndarray,
    L: int,
    output_path: str,
):
    """从 3-bead nm 坐标写骨架 PDB (用于后续 CG_to_allatom).

    提取 P bead (索引 0,3,6,...) 输出标准骨架 PDB.
    PDB 列格式 (CG_to_allatom.exe 的 substr 解析):
      col 12-15: atom name (" P  ")
      col 17-19: resname ("RA ")
      col 21:    chain ID ("A")
      col 22-25: resid ("   1")
      col 30-37: x (8.3f)
      col 38-45: y (8.3f)
      col 46-53: z (8.3f)
    """
    coords_ang = p_coords_3bead_nm * 10.0  # nm -> Å
    p_coords = coords_ang[0::3].copy()  # (L,3) Å

    # 平移到正象限 (避免负坐标溢出 8.3f 列宽, 不改变相对几何)
    if len(p_coords) > 0:
        min_xyz = p_coords.min(axis=0)
        shift = np.where(min_xyz < 0, -min_xyz + 5.0, 0.0)
        p_coords = p_coords + shift

    lines = ["HEADER    OpenMM GPU refined CG structure"]
    for i in range(L):
        x, y, z = p_coords[i]
        # 标准 PDB ATOM 格式: 必须精确对齐列
        serial = f"{i + 1:5d}"      # col 6-10
        name = " P  "                # col 12-15 (4 chars)
        resname = "RA "              # col 17-19
        chain = "A"                  # col 21
        resid = f"{i + 1:4d}"       # col 22-25
        x_str = f"{x:8.3f}"         # col 30-37
        y_str = f"{y:8.3f}"         # col 38-45
        z_str = f"{z:8.3f}"         # col 46-53
        # 拼装: "ATOM  " + serial + " " + name + resname + " " + chain + resid + "    " + x + y + z + "  1.00  0.00           P "
        line = f"ATOM  {serial} {name} {resname} {chain}{resid}    {x_str}{y_str}{z_str}  1.00  0.00           P "
        lines.append(line)
    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _write_refined_pdb(
    allatom_pdb_path: str,
    output_path: str,
):
    """复制全原子 PDB 到输出路径."""
    import shutil
    shutil.copy2(allatom_pdb_path, output_path)


# ── 力场参数 (与 cg_forcefield.py 对齐) ──

# 力常数 (kJ/mol/Å², 内部用 nm 需 *100)
K_BB = 310.0       # 骨架 P-P
K_INTRA = 310.0    # P-C4', C4'-N
K_PAIR = 800.0     # WC 配对 N-N
K_STACK = 300.0    # 碱基堆叠
K_ANGLE = 400.0    # 骨架键角 (加强, 减少局部应变)
K_DIHEDRAL = 500.0  # 骨架二面角 (加强10x, 强制A-form几何)
K_CLASH = 200.0    # clash
K_BSJ = 500.0      # BSJ 闭合
K_BSJ_GUIDE = 800.0  # BSJ 引导力

# 几何参数 (Å)
BOND_P_NEXT = 5.90
BOND_P_C4 = 3.90
BOND_C4_N = 3.35
ANGLE_PPP = 2.618   # rad, 150°
DIH_PPPP = 33.0 * np.pi / 180.0  # rad
STACK_R0 = 5.05
PAIR_N_N = 10.0
CLASH_DIST = 3.0
CUTOFF = 12.0


def _build_3bead_system_gpu(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    pair_scale: float = 1.0,
    bsj_k_scale: float = 1.0,
    pair_guide_k: float = 0.0,
    bpp_matrix: Optional[np.ndarray] = None,
    bpp_weight: float = 0.5,
    pair_predictions: Optional[np.ndarray] = None,
    ss_predictions: Optional[np.ndarray] = None,
    bsj_prediction: Optional[float] = None,
):
    """构建 3-bead CG OpenMM system (GPU 优化版).

    与 cg_forcefield.build_3bead_system() 力场一致,
    但简化接口, 去掉统计势 (GPU 路径追求速度).

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] ViennaRNA 配对
        pair_scale: 配对力缩放 (退火用)
        bsj_k_scale: BSJ 力缩放 (退火用)
        pair_guide_k: 配对窗引导力 (kJ/mol). >0 时对远端配对施加
            渐近吸引力, 把相距 100-3000Å 的配对原子逐步拉近到
            力场作用范围 (~20Å), 之后普通配对力接管. 解决环状
            RNA 初始构象配对原子距离过大 (力场够不到) 的问题.
        bpp_matrix: (L,L) ViennaRNA 配对概率矩阵 (可选).
            >0 时用 bpp_ij 加权配对力: k_pair = K_PAIR * (bpp_w * bpp_ij + (1-bpp_w) * w) * pair_scale
        bpp_weight: bpp 加权系数. 1.0=纯bpp, 0.0=纯硬编码w.

    Returns:
        (system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide)
    """
    L = len(p_coords)
    N_total = 3 * L

    # 构建 3-bead 坐标: 每个 nt → P, C4', N
    coords_3bead = np.zeros((N_total, 3), dtype=np.float64)
    rng = np.random.default_rng(42)
    for i in range(L):
        p = p_coords[i]
        coords_3bead[3 * i] = p  # P
        # C4' 和 N 用扰动估计 (后续 minimize 修正)
        coords_3bead[3 * i + 1] = p + rng.normal(0, 0.3, 3)  # C4'
        coords_3bead[3 * i + 2] = p + rng.normal(0, 0.3, 3)  # N

    coords_nm = coords_3bead / 10.0  # Å → nm

    system = mm.System()
    for _ in range(N_total):
        system.addParticle(330.0 / 3.0)  # ~110 Da per bead

    def P(i): return 3 * i
    def C4(i): return 3 * i + 1
    def N(i): return 3 * i + 2

    # 1. 骨架键 P[i]-P[i+1]
    bond_bb = mm.HarmonicBondForce()
    bb_k = K_BB * 100.0  # Å² → nm²
    for i in range(L - 1):
        bond_bb.addBond(P(i), P(i + 1), BOND_P_NEXT / 10.0, bb_k)
    system.addForce(bond_bb)

    # 1b. BSJ 闭合 (首末 P)
    # structRFM: bsj_prediction 调制 BSJ 力常数
    bsj_confidence = float(bsj_prediction) if bsj_prediction is not None else 1.0
    effective_bsj_k = bsj_k_scale * K_BSJ * (0.3 + 0.7 * bsj_confidence)
    bsj_force = mm.CustomBondForce("0.5*k_bsj*(r-r0)^2")
    bsj_force.addPerBondParameter("k_bsj")
    bsj_force.addPerBondParameter("r0")
    bsj_force.addBond(P(L - 1), P(0),
                      [effective_bsj_k, BOND_P_NEXT / 10.0])
    system.addForce(bsj_force)

    # 1c. BSJ 引导力
    bsj_guide = mm.CustomBondForce("0.5*k_guide*(r-r0)^2")
    bsj_guide.addPerBondParameter("k_guide")
    bsj_guide.addPerBondParameter("r0")
    bsj_guide.addBond(P(L - 1), P(0),
                      [bsj_k_scale * K_BSJ_GUIDE, BOND_P_NEXT / 10.0])
    system.addForce(bsj_guide)

    # 1d. BSJ 区接触图: 连接处 ±bsj_contact_nt 的核苷酸应空间聚簇
    # 环状 RNA 的 BSJ 区域 (5'/3' 连接处) 通常有保守结构:
    #   - 茎区跨越 junction, 或
    #   - junction 两侧有碱基堆叠
    # 力: harmonic attractor, r0 = 10Å (略大于 WC 距离, 允许灵活性)
    # 力常数随距 junction 的距离衰减: k = K_BSJ_CONTACT * (1 - d/max_d)^2
    bsj_contact_nt = min(8, L // 4)  # 每侧 8nt 或序列 1/4
    K_BSJ_CONTACT = 200.0  # kJ/mol/Å²
    r0_bsj_contact = 1.0   # nm = 10Å
    bsj_contact_force = mm.CustomBondForce(
        "0.5*k_c*(r-r0)^2 * (1 - dist_ratio)^2")
    bsj_contact_force.addPerBondParameter("k_c")
    bsj_contact_force.addPerBondParameter("r0")
    bsj_contact_force.addGlobalParameter("dist_ratio", 0.0)  # 占位, 实际用 per-bond

    # 改用简单 harmonic (OpenMM CustomBondForce 不支持全局变量 per-bond)
    bsj_contact_force = mm.CustomBondForce("0.5*k_c*(r-r0)^2")
    bsj_contact_force.addPerBondParameter("k_c")
    bsj_contact_force.addPerBondParameter("r0")

    for i in range(-bsj_contact_nt, bsj_contact_nt):
        for j in range(i + 1, bsj_contact_nt + 1):
            # 循环索引
            ii = i % L
            jj = j % L
            if ii == jj:
                continue
            # 距 junction 的距离 (min of direct and wrap-around)
            d_i = min(ii, L - ii)  # 到 position 0 的距离
            d_j = min(jj, L - jj)
            # 距离衰减: 越靠近 junction 越强
            max_d = bsj_contact_nt
            decay_i = max(0.0, 1.0 - d_i / max_d)
            decay_j = max(0.0, 1.0 - d_j / max_d)
            k_contact = K_BSJ_CONTACT * decay_i * decay_j * bsj_k_scale
            if k_contact > 1.0:  # 最小阈值
                bsj_contact_force.addBond(P(ii), P(jj), [k_contact, r0_bsj_contact])

    if bsj_contact_force.getNumBonds() > 0:
        system.addForce(bsj_contact_force)

    # 1e. bpp 软约束势能 (S10 思想 #4: 先验信息当软引导)
    #   U_bpp = Σ bpp(i,j) · k_bpp · (d(i,j) - d_native)²
    #   高 bpp 的残基对被拉向 native 距离, 低 bpp 的自由探索.
    #   d_native = 10.5Å (WC 配对 C1'-C1' 距离)
    if bpp_matrix is not None and bpp_matrix.shape[0] == L:
        K_BPP_SOFT = 100.0  # kJ/mol/Å² (软约束, 比硬配对力弱)
        d_native_bpp = 1.05  # nm = 10.5Å
        bpp_soft_force = mm.CustomBondForce("0.5*k_bpp*(r-r0)^2")
        bpp_soft_force.addPerBondParameter("k_bpp")
        bpp_soft_force.addPerBondParameter("r0")
        n_bpp_soft = 0
        for i in range(L):
            for j in range(i + 5, L):  # 跳过近端 (已有骨架力)
                bpp_val = float(bpp_matrix[i, j])
                if bpp_val < 0.05:  # 低概率跳过
                    continue
                # 力常数 = 基础值 × bpp 概率 × bpp_weight
                k_bpp = K_BPP_SOFT * bpp_val * bpp_weight
                if k_bpp > 0.5:
                    bpp_soft_force.addBond(P(i), P(j), [k_bpp, d_native_bpp])
                    n_bpp_soft += 1
        if n_bpp_soft > 0:
            system.addForce(bpp_soft_force)

    # 2. 残基内键 P-C4', C4'-N
    bond_intra = mm.HarmonicBondForce()
    ik = K_INTRA * 100.0
    for i in range(L):
        bond_intra.addBond(P(i), C4(i), BOND_P_C4 / 10.0, ik)
        bond_intra.addBond(C4(i), N(i), BOND_C4_N / 10.0, ik)
    system.addForce(bond_intra)

    # 3. 骨架键角 P-P-P
    angle_force = mm.HarmonicAngleForce()
    for i in range(L - 2):
        angle_force.addAngle(P(i), P(i + 1), P(i + 2), ANGLE_PPP, K_ANGLE)
    # 环化角
    if L >= 3:
        angle_force.addAngle(P(L - 2), P(L - 1), P(0), ANGLE_PPP, K_ANGLE)
        angle_force.addAngle(P(L - 1), P(0), P(1), ANGLE_PPP, K_ANGLE)
    system.addForce(angle_force)

    # 3.5 骨架二面角
    dih_force = mm.CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
    dih_force.addGlobalParameter("k_dih", K_DIHEDRAL)
    dih_force.addGlobalParameter("theta0", DIH_PPPP)
    for i in range(L - 3):
        dih_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
    if L >= 4:
        dih_force.addTorsion(P(L - 3), P(L - 2), P(L - 1), P(0))
        dih_force.addTorsion(P(L - 2), P(L - 1), P(0), P(1))
        dih_force.addTorsion(P(L - 1), P(0), P(1), P(2))
    system.addForce(dih_force)

    # 4. WC 配对 N-N (bpp 加权: k = K_PAIR * (bpp_w * bpp_ij + (1-bpp_w) * w) * scale)
    pair_force = mm.CustomBondForce("0.5*k_pair*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            # bpp 加权: 如果有 bpp_matrix, 混合 bpp 概率和硬编码权重
            if bpp_matrix is not None and bpp_weight > 0:
                bpp_val = float(bpp_matrix[i, j]) if i < bpp_matrix.shape[0] and j < bpp_matrix.shape[1] else 0.0
                effective_w = bpp_weight * bpp_val + (1.0 - bpp_weight) * w
            else:
                effective_w = w
            # structRFM: pair_predictions 调制
            if pair_predictions is not None:
                pair_idx = None
                for pi, (ii, jj) in enumerate(pairs):
                    if (ii == i and jj == j) or (ii == j and jj == i):
                        pair_idx = pi
                        break
                if pair_idx is not None and pair_idx < len(pair_predictions):
                    struct_w = 0.5 + 0.5 * float(pair_predictions[pair_idx])
                    effective_w *= struct_w
            # structRFM: ss_predictions 调制 stacking (paired→强, unpaired→弱)
            if ss_predictions is not None and i < len(ss_predictions) and j < len(ss_predictions):
                ss_avg = (float(ss_predictions[i]) + float(ss_predictions[j])) / 2.0
                effective_w *= (0.5 + 0.5 * ss_avg)
            pair_force.addBond(
                N(i), N(j),
                [K_PAIR * effective_w * pair_scale, PAIR_N_N / 10.0])
    system.addForce(pair_force)

    # 4b. 配对窗引导力 (远端配对软吸引)
    # V = -k_g * (1/(1+exp(a*(r-r_cap)))) * step(r-r0_lo)
    #   - r >> r_cap: V -> 0 (够不到不强拉, 防止撕裂结构)
    #   - r ~ r_cap: 逻辑斯蒂过渡, 峰值力 ~ k_g*a/4
    #   - r < r0_lo (已配对): 关闭
    # a=0.05 (特征长度 20nm), r_cap=40nm 时覆盖 20-60nm (200-600Å)
    # 的配对, 峰值力温和, 不会像线性窗那样恒定拉力撕裂结构.
    if pair_guide_k > 0:
        guide_force = mm.CustomBondForce(
            "-k_g*(1/(1+exp(a*(r-r_cap))))*step(r-r0_lo)")
        guide_force.addPerBondParameter("k_g")
        guide_force.addGlobalParameter("a", 0.05)     # /nm, 特征长度 ~20nm
        guide_force.addGlobalParameter("r_cap", 40.0)  # nm = 400Å
        guide_force.addGlobalParameter("r0_lo", 1.5)   # nm = 15Å, 已配对关闭
        for (i, j, w) in pairs:
            if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                    and not (i == 0 and j == L - 1)):
                guide_force.addBond(
                    N(i), N(j), [pair_guide_k * w])
        system.addForce(guide_force)

    # 5. 碱基堆叠
    stack_force = mm.CustomBondForce("0.5*k_stack*(r-r0)^2")
    stack_force.addPerBondParameter("k_stack")
    stack_force.addPerBondParameter("r0")
    sk = K_STACK * 100.0
    for i in range(L - 1):
        stack_force.addBond(N(i), N(i + 1), [sk, STACK_R0 / 10.0])
    # 环化堆叠
    stack_force.addBond(N(L - 1), N(0), [sk, STACK_R0 / 10.0])
    system.addForce(stack_force)

    # 6. 非键 clash + 静电 (有界软球, 避免退火塌缩爆炸)
    #   硬截断 step(dmin-r)*k*(dmin-r)^2 在 r~dmin 处突变, 结构塌缩时
    #   斥力爆炸 (轮3-5 E~3.9亿). 改用有界软球:
    #   V = k_clash*(dmin-r)^2/(1+(dmin-r)^2/alpha^2) * step(dmin-r)
    #   r->0 时 V -> k_clash*alpha^2 (有限), 力不会无限增大.
    #   Coulomb 用软化形式 1/sqrt(r^2+soft^2), 避免 1/r 发散.
    clash_force = mm.CustomNonbondedForce(
        "k_clash*(dmin-r)^2/(1+(dmin-r)^2/alpha^2)*step(dmin-r)"
        " + Coul*q1*q2/sqrt(r^2+soft^2)"
    )
    clash_force.addPerParticleParameter("q")
    clash_force.addGlobalParameter("dmin", CLASH_DIST / 10.0)
    clash_force.addGlobalParameter("k_clash", K_CLASH * 100.0)
    clash_force.addGlobalParameter("alpha", 0.5)  # nm, 软球软化尺度
    clash_force.addGlobalParameter("Coul", 138.935456)
    clash_force.addGlobalParameter("soft", 0.5)   # nm, Coulomb 软化
    clash_force.setNonbondedMethod(
        mm.CustomNonbondedForce.CutoffNonPeriodic)
    clash_force.setCutoffDistance(CUTOFF / 10.0)

    for i in range(L):
        clash_force.addParticle([-0.5])  # P
        clash_force.addParticle([0.0])    # C4'
        clash_force.addParticle([0.0])    # N

    # 排除键对
    excluded = set()
    for i in range(L):
        for (a, b) in [(P(i), C4(i)), (C4(i), N(i))]:
            k = (min(a, b), max(a, b))
            if k not in excluded:
                excluded.add(k)
                clash_force.addExclusion(*k)
    for i in range(L - 1):
        k = (P(i), P(i + 1))
        if k not in excluded:
            excluded.add(k)
            clash_force.addExclusion(*k)
    k = (P(L - 1), P(0))
    if k not in excluded:
        excluded.add(k)
        clash_force.addExclusion(*k)
    for i in range(L - 1):
        k = (N(i), N(i + 1))
        if k not in excluded:
            excluded.add(k)
            clash_force.addExclusion(*k)
    k = (N(L - 1), N(0))
    if k not in excluded:
        excluded.add(k)
        clash_force.addExclusion(*k)
    system.addForce(clash_force)

    return (system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide)


def _build_minimal_system_gpu(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    pair_scale: float = 1.0,
):
    """构建极简 P-only 折叠力场 (两阶段方案阶段1).

    只含:
      1. P 骨架键 P[i]-P[i+1] (r0=5.9Å, k=31000 kJ/mol/nm²)
      2. P-P 配对键 (r0=5.9Å, k=40000×w×pair_scale)
    无 clash/堆叠/键角/C4'N — 这些项在完整力场下阻碍折叠
    (实测完整力场配对卡在 45Å, 极简力场折叠到 21Å).

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] ViennaRNA 配对
        pair_scale: 配对力缩放

    Returns:
        (system, coords_nm, pair_force) — 只有 P bead (L 个粒子)
    """
    L = len(p_coords)
    coords_nm = p_coords / 10.0  # Å → nm

    system = mm.System()
    for _ in range(L):
        system.addParticle(110.0)

    # 1. P 骨架键
    bond_bb = mm.HarmonicBondForce()
    bb_k = 31000.0  # kJ/mol/nm²
    for i in range(L - 1):
        bond_bb.addBond(i, i + 1, BOND_P_NEXT / 10.0, bb_k)
    # 1b. BSJ 闭合键: 强制首尾 P-P ~5.9Å, 防止退火时环打开
    bond_bb.addBond(0, L - 1, BOND_P_NEXT / 10.0, 500.0)
    system.addForce(bond_bb)

    # 1c. 骨架角度约束: 防止折叠时 backbone 角度塌缩
    angle_bb = mm.HarmonicAngleForce()
    for i in range(L - 2):
        angle_bb.addAngle(i, i + 1, i + 2,
                          2.618,  # 150° in rad (A-form RNA backbone)
                          500.0)  # kJ/mol/rad²
    system.addForce(angle_bb)

    # 1d. 碰撞排斥: 防止原子重叠 (核心: 没有这个结构会坍缩成球)
    clash = mm.CustomNonbondedForce(
        "step(d_min - r) * 0.5 * k_clash * (d_min - r)^2")
    clash.addGlobalParameter("k_clash", 5000.0)  # kJ/mol/nm²
    clash.addGlobalParameter("d_min", 0.3)  # 3.0A = 0.3nm 最小距离
    for _ in range(L):
        clash.addParticle()
    # 只对近邻检查 (15nt 窗口), 避免 O(n²)
    neighbors = []
    for i in range(L):
        nb = list(range(max(0, i - 15), min(L, i + 16)))
        nb = [j for j in nb if j > i]
        if nb:
            neighbors.append((i, nb))
    # 用 InteractionGroup 分组
    all_a, all_b = [], []
    for i, nbs in neighbors:
        all_a.extend([i] * len(nbs))
        all_b.extend(nbs)
    if all_a:
        clash.addInteractionGroup(all_a, all_b)
    system.addForce(clash)

    # 2. P-P 配对键 (折叠驱动, 加碰撞排斥后可适当减小力常数)
    pair_force = mm.CustomBondForce("0.5*k_pair*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            # 远端配对 (>100nt) 力常数 ×2
            far_boost = 2.0 if (min(abs(j-i), L-abs(j-i)) > 100) else 1.0
            pair_force.addBond(
                i, j, [30000.0 * w * pair_scale * far_boost, BOND_P_NEXT / 10.0])
    system.addForce(pair_force)

    return system, coords_nm, pair_force


def _create_minimal_topology(L: int) -> Topology:
    """创建 P-only 拓扑 (每 nt 一个 P atom)."""
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("RA", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
    return topo


def _create_3bead_topology(L: int) -> Topology:
    """创建 3-bead CG 拓扑 (P/C4'/N per nt)."""
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("N", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
        topo.addAtom(f"C{i}", Element.getBySymbol("C"), res)
        topo.addAtom(f"N{i}", Element.getBySymbol("N"), res)
    return topo


# ── 三阶段退火 ──

def _run_annealing(
    sim: Simulation,
    pair_force,
    bsj_force,
    bsj_guide,
    L: int,
    n_anneal: int = 200,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """三阶段退火: 弱配对+弱BSJ → 强配对+中BSJ → 强配对+强BSJ.

    Returns:
        (final_energy, final_coords_nm)
    """
    def set_pair_k(scale):
        for i in range(pair_force.getNumBonds()):
            p1, p2, params = pair_force.getBondParameters(i)
            # 更新 k, 保持 r0
            pair_force.setBondParameters(
                i, p1, p2,
                [scale * K_PAIR, params[1]])
        pair_force.updateParametersInContext(sim.context)

    def set_bsj_k(scale):
        bsj_force.setBondParameters(
            0, 3 * (L - 1), 0,
            [scale * K_BSJ, BOND_P_NEXT / 10.0])
        bsj_guide.setBondParameters(
            0, 3 * (L - 1), 0,
            [scale * K_BSJ_GUIDE, BOND_P_NEXT / 10.0])
        bsj_force.updateParametersInContext(sim.context)
        bsj_guide.updateParametersInContext(sim.context)

    # 记录初始能量
    pre_state = sim.context.getState(getPositions=True, getEnergy=True)
    e_pre = pre_state.getPotentialEnergy()._value

    # 阶段1: 中温 + 弱配对 + 弱BSJ, 螺旋形成
    set_pair_k(0.1)
    set_bsj_k(0.3)
    sim.integrator.setTemperature(350 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=2000)

    # 阶段2: 中温 + 强配对 + 中BSJ, WC 配对拉拢
    set_pair_k(1.0)
    set_bsj_k(1.0)
    sim.integrator.setTemperature(320 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=2000)

    # 阶段3: 低温 + 强配对 + 强BSJ, 闭合
    set_pair_k(1.0)
    set_bsj_k(5.0)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=3000)

    # 阶段4 (新增): 极低温 + 超强BSJ, 精修闭合
    set_pair_k(1.0)
    set_bsj_k(10.0)
    sim.integrator.setTemperature(280 * unit.kelvin)
    sim.step(n_anneal // 2)
    sim.minimizeEnergy(
        tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=5000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # 安全网: MD 暴走回退
    if e1 > e_pre * 0.5 and e_pre < 0:
        pos = pre_state.getPositions(asNumpy=True)._value
        e1 = e_pre

    return e1, pos


def _run_anneal_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Å P 坐标
    pairs: List[Tuple[int, int, float]],
    n_anneal: int,
    n_threads: int,
):
    """多进程退火 worker: 独立构建 system + 三阶段退火.

    用不同随机种子 (worker_idx) 增加轨迹多样性.
    Returns:
        (final_energy, final_coords_nm)
    """
    import numpy as _np
    # 不同种子 -> 不同 C4'/N 初始扰动
    _np.random.seed(42 + worker_idx)

    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(
            p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.1 + 0.05 * worker_idx,
            pair_guide_k=300.0)  # 配对窗引导力, 把远端配对拉近
    topo = _create_3bead_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # 用全局 _run_annealing 做三阶段退火
    e_final, pos_final = _run_annealing(
        sim, pair_force, bsj_force, bsj_guide, len(p_coords),
        n_anneal=n_anneal, verbose=False)
    return e_final, pos_final


def _run_minimal_anneal_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Å P 坐标
    pairs: List[Tuple[int, int, float]],
    n_anneal: int,
    n_threads: int,
):
    """极简力场退火 worker (两阶段方案阶段1: 折叠).

    只含 P 骨架键 + P-P 配对, 无 clash/堆叠. 高温退火折叠.
    Returns:
        (final_energy, final_coords_ang)  # P-only, Å
    """
    system, coords_nm, pair_force = _build_minimal_system_gpu(
        p_coords, pairs, pair_scale=1.0)
    topo = _create_minimal_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        450 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # 先最小化消除初始 clash
    sim.minimizeEnergy(maxIterations=3000)

    # 逐步降温退火 (折叠驱动): 高温跑配对拉近, 逐步降温
    # 加强版: 8阶段, 更细粒度温度控制
    stages = [
        (400, n_anneal // 8),   # 中高温: 保留局部结构, 远端配对探索
        (380, n_anneal // 8),   # 中温: 螺旋形成
        (360, n_anneal // 8),   # 中温: 配对拉近
        (340, n_anneal // 8),   # 中低温: WC配对收敛
        (320, n_anneal // 8),   # 低温: 碰撞消除
        (310, n_anneal // 8),   # 低温: 结构精修
        (305, n_anneal // 8),   # 接近室温: BSJ闭合
        (300, n_anneal // 8),   # 室温: 最终稳定
    ]
    for T, n in stages:
        integrator.setTemperature(T * unit.kelvin)
        sim.step(max(1, n))
    # 终局最小化 (更严格)
    sim.minimizeEnergy(
        tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=8000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos_nm = state.getPositions(asNumpy=True)._value  # nm
    e = state.getPotentialEnergy()._value
    pos_ang = pos_nm * 10.0  # → Å
    return e, pos_ang


def _run_parallel_minimal_annealing(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    n_anneal: int = 200,
    n_trajectories: int = 4,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """多进程并行极简折叠: N 条轨迹, 取最低能量.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)]

    Returns:
        (best_energy, best_coords_ang)  # P-only, Å
    """
    import multiprocessing as mp

    total_threads = os.cpu_count() or 8
    per_traj_threads = max(1, total_threads // n_trajectories)
    if verbose:
        print(f"  极简折叠: {n_trajectories} 轨迹 x {per_traj_threads} 线程")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_trajectories) as pool:
        results = pool.starmap(
            _run_minimal_anneal_worker,
            [(i, p_coords, pairs, n_anneal, per_traj_threads)
             for i in range(n_trajectories)],
        )

    best_energy = float("inf")
    best_pos = None
    for e, pos in results:
        if e < best_energy:
            best_energy = e
            best_pos = pos
    return best_energy, best_pos


def _run_parallel_annealing(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    n_anneal: int = 200,
    n_trajectories: int = 4,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """多进程并行退火: N 条轨迹各 32/N 线程, 取最低能量.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)]
        n_anneal: 每阶段步数
        n_trajectories: 并行轨迹数
        platform_name: 平台
        verbose: 打印

    Returns:
        (best_energy, best_coords_nm)
    """
    import multiprocessing as mp

    total_threads = os.cpu_count() or 8
    per_traj_threads = max(1, total_threads // n_trajectories)
    if verbose:
        print(f"  并行退火: {n_trajectories} 条轨迹 x {per_traj_threads} 线程 "
              f"(总 {total_threads} 核)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_trajectories) as pool:
        results = pool.starmap(
            _run_anneal_worker,
            [(i, p_coords, pairs, n_anneal, per_traj_threads)
             for i in range(n_trajectories)],
        )

    best_energy = float("inf")
    best_pos = None
    for e, pos in results:
        if e < best_energy:
            best_energy = e
            best_pos = pos

    return best_energy, best_pos


# ── T-REMD (多温度副本交换) ──

def _run_remd_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Å P 坐标
    pairs: List[Tuple[int, int, float]],
    temperature: float,
    n_steps: int,
    exchange_interval: int,
    n_threads: int,
    conn,
    minimal: bool = False,
):
    """REMD 单副本 worker 进程: 本地重建 system + 模拟 + Pipe 交换.

    worker 接收 P 坐标和配对, 自行构建 system (避免 pickle
    OpenMM 对象), 每个 exchange_interval 步报告能量并接收交换坐标.
    minimal=True 时用极简力场 (P骨架+P配对, 保持折叠一致性).
    """
    if minimal:
        system, coords_nm, _pf = _build_minimal_system_gpu(
            p_coords, pairs, pair_scale=1.0)
        topo = _create_minimal_topology(len(p_coords))
    else:
        # 每个 worker 独立构建 system (不同 bsj_k_scale 增加多样性)
        system, coords_nm, _pf, _sf, _bjf, _bjg = _build_3bead_system_gpu(
            p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.5 + 0.1 * worker_idx,
            pair_guide_k=300.0)  # 配对窗引导力
        topo = _create_3bead_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)
    sim.minimizeEnergy(maxIterations=500)

    # 记录初始能量
    state = sim.context.getState(getEnergy=True)
    e0 = state.getPotentialEnergy()._value
    best_energy = e0
    best_pos = coords_nm.copy()

    conn.send(("init", worker_idx, e0))

    # 主循环
    for step_i in range(n_steps):
        sim.step(1)
        if (step_i + 1) % 500 == 0:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()._value
            if energy < best_energy:
                best_energy = energy
                best_pos = state.getPositions(asNumpy=True)._value

        # 交换点: 发能量+坐标, 等交换决策
        if (step_i + 1) % exchange_interval == 0:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()._value
            pos = state.getPositions(asNumpy=True)._value
            conn.send(("report", worker_idx, energy, pos))
            # 等待主进程交换结果
            cmd = conn.recv()
            if cmd[0] == "swap":
                new_pos = cmd[1]
                sim.context.setPositions(new_pos * unit.nanometer)
            # "keep" 则不动

    # 最终报告
    conn.send(("done", worker_idx, best_energy, best_pos))
    conn.close()


def _clamp_replicas_by_memory(n_replicas: int, mem_per_proc_gb: float = 4.0) -> int:
    """根据剩余内存限制并行进程数 (保守策略).

    Args:
        n_replicas: 期望进程数
        mem_per_proc_gb: 每进程估算内存 (GB), 2013nt 全原子 ~4GB

    Returns:
        限制后的进程数 (至少 1)
    """
    try:
        import psutil
        avail = psutil.virtual_memory().available / (1024 ** 3)
        max_by_mem = max(1, int(avail // mem_per_proc_gb))
        return max(1, min(n_replicas, max_by_mem))
    except ImportError:
        return n_replicas


def _run_remd(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str,
    n_replicas: int = 4,
    n_steps: int = 500,
    exchange_interval: int = 100,
    verbose: bool = False,
    minimal: bool = False,
) -> Tuple[float, np.ndarray]:
    """执行 T-REMD 增强采样 (多进程并行).

    每个副本一个进程, 自行构建 system (只传 numpy/list).
    线程数 = cpu_count // n_replicas, 总核心全打满.
    minimal=True 时用极简力场 (P骨架+P配对), 返回 P-only nm 坐标.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] 配对
        platform_name: 平台
        n_replicas: 副本数
        n_steps: 总步数
        exchange_interval: 交换间隔
        verbose: 打印
        minimal: 用极简力场 (默认 False)

    Returns:
        (best_energy, best_coords_nm)  # minimal=True: P-only nm; False: 3-bead nm
    """
    from scipy.constants import k as kB
    import multiprocessing as mp

    # 温度阶梯: 300K -> ~460K (几何间隔)
    temperatures = [300.0 * (1.10 ** i) for i in range(n_replicas)]

    # 每副本线程数 (总核心均分)
    total_threads = os.cpu_count() or 8
    # 内存感知: 根据剩余内存限制进程数 (保守, 每进程 ~6GB)
    n_replicas = _clamp_replicas_by_memory(n_replicas, mem_per_proc_gb=6.0)
    per_replica_threads = max(1, total_threads // n_replicas)
    if verbose:
        print(f"    REMD: {n_replicas} 副本并行, "
              f"每副本 {per_replica_threads} 线程 "
              f"(总 {total_threads} 核)")

    ctx = mp.get_context("spawn")
    processes = []
    conns = []
    for ri in range(n_replicas):
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        p = ctx.Process(
            target=_run_remd_worker,
            args=(ri, p_coords, pairs, temperatures[ri], n_steps,
                  exchange_interval, per_replica_threads, child_conn, minimal),
        )
        p.start()
        child_conn.close()
        processes.append(p)
        conns.append(parent_conn)

    # 主进程: 协调交换
    best_energy = float("inf")
    # 初始坐标 (nm): 极简模式 P-only, 完整模式 3-bead (补 C4'/N)
    if minimal:
        best_pos = p_coords / 10.0  # (L,3) P-only nm
    else:
        L0 = len(p_coords)
        _rng0 = np.random.default_rng(0)
        best_pos = np.zeros((3 * L0, 3), dtype=np.float64)
        for _i in range(L0):
            best_pos[3 * _i] = p_coords[_i] / 10.0
            best_pos[3 * _i + 1] = p_coords[_i] / 10.0 + _rng0.normal(0, 0.03, 3)
            best_pos[3 * _i + 2] = p_coords[_i] / 10.0 + _rng0.normal(0, 0.03, 3)
    accept_count = 0
    total_exchanges = max(1, (n_steps // exchange_interval) * (n_replicas - 1))

    # 阶段1: 等所有副本 init
    for ri in range(n_replicas):
        msg = conns[ri].recv()
        assert msg[0] == "init"
        if msg[2] < best_energy:
            best_energy = msg[2]

    # 阶段2: 协调交换
    n_exchange_points = n_steps // exchange_interval
    for _ in range(n_exchange_points):
        energies = [None] * n_replicas
        positions = [None] * n_replicas
        for ri in range(n_replicas):
            msg = conns[ri].recv()
            assert msg[0] == "report"
            energies[ri] = msg[2]
            positions[ri] = msg[3]
            if msg[2] < best_energy:
                best_energy = msg[2]
                best_pos = msg[3].copy()

        # 相邻副本 Metropolis 交换
        swap_decisions = [False] * (n_replicas - 1)
        for ri in range(n_replicas - 1):
            ui, uj = energies[ri], energies[ri + 1]
            beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
            beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
            exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
            if np.random.random() < min(1.0, np.exp(exponent)):
                swap_decisions[ri] = True
                accept_count += 1

        # 应用交换: 发新坐标给参与交换的副本
        for ri in range(n_replicas):
            new_pos = None
            if ri > 0 and swap_decisions[ri - 1]:
                new_pos = positions[ri - 1]
            elif ri < n_replicas - 1 and swap_decisions[ri]:
                new_pos = positions[ri + 1]
            if new_pos is not None:
                conns[ri].send(("swap", new_pos))
            else:
                conns[ri].send(("keep",))

    # 阶段3: 收尾
    for ri in range(n_replicas):
        msg = conns[ri].recv()
        assert msg[0] == "done"
        if msg[2] < best_energy:
            best_energy = msg[2]
            best_pos = msg[3]

    for p in processes:
        p.join(timeout=10)
    for conn in conns:
        conn.close()

    if verbose:
        rate = accept_count / total_exchanges
        print(f"    REMD: E={best_energy:.0f}, 交换率 {rate:.1%}")

    return best_energy, best_pos


# ── bpp 引导的远端配对发现 ──

def discover_far_pairs_from_bpp(
    bpp_matrix: np.ndarray,
    sequence: str,
    min_gap: int = 24,
    bpp_threshold: float = 0.1,
    top_k: int = 50,
    existing_pairs: Optional[List[Tuple[int, int, float]]] = None,
) -> List[Tuple[int, int, float]]:
    """从 ViennaRNA bpp 概率矩阵发现远端配对.

    参考 scheme10_full.py BppPriorModule 的共享伴侣 Jaccard 相似度:

      核心洞察:
        1. bpp(i,j) 高 → i 和 j 在同一个折叠单元 (茎区)
        2. 同一折叠单元的核苷酸倾向于在空间聚簇
        3. 如果 i 和 k 都在多个高 bpp 茎区中出现 → 它们可能在同一个结构域
        4. 这种"共现关系"推断远端接触的可能性

      算法:
        对每对 (i,j) (|i-j| >= min_gap):
          P(i) = {k | bpp(i,k) > threshold}  -- i 的配对伙伴集
          P(j) = {k | bpp(j,k) > threshold}  -- j 的配对伙伴集
          J(i,j) = |P(i) ∩ P(j)| / |P(i) ∪ P(j)|  -- Jaccard 相似度
          w(i,j) = bpp(i,j) * J(i,j)  -- 直接bpp概率 × 共享伴侣相似度

        J 高 → i 和 j 共享很多配对伙伴 → 同一结构域 → 空间接近
        即使 bpp(i,j) 本身不高, 共享伴侣多也能推断远端接触.

    Args:
        bpp_matrix: (L,L) 配对概率矩阵
        sequence: RNA 序列
        min_gap: 最小序列间隔 (默认 24, 即 >1 轮螺旋)
        bpp_threshold: 最小 bpp 值 (用于定义"配对伙伴")
        top_k: 最多返回多少对
        existing_pairs: 已有配对 [(i,j,w)], 排除重复

    Returns:
        [(i, j, w)] 新发现的远端配对 (w = bpp * Jaccard)
    """
    L = len(sequence)
    if bpp_matrix is None or bpp_matrix.shape[0] != L:
        return []

    # 构建已有配对集合 (避免重复)
    existing_set = set()
    if existing_pairs:
        for (i, j, w) in existing_pairs:
            existing_set.add((min(i, j), max(i, j)))

    # Step 1: 预计算每个位置的配对伙伴集
    partner_sets = []
    for i in range(L):
        partners = set()
        for k in range(L):
            if k != i and float(bpp_matrix[i, k]) > bpp_threshold:
                partners.add(k)
        partner_sets.append(partners)

    # Step 2: 对每对远端 (i,j) 计算 Jaccard 相似度
    candidates = []
    for i in range(L):
        pi = partner_sets[i]
        if len(pi) == 0:
            continue
        for j in range(i + min_gap, L):
            pj = partner_sets[j]
            if len(pj) == 0:
                continue
            key = (i, j)
            if key in existing_set:
                continue

            bpp_val = float(bpp_matrix[i, j])

            # 共享伴侣 Jaccard: |P(i) ∩ P(j)| / |P(i) ∪ P(j)|
            shared = len(pi & pj)
            union = len(pi) + len(pj) - shared
            if union == 0:
                continue
            jaccard = shared / union

            # 综合权重 (加性, 参考 BppPriorModule):
            #   w = bpp_direct + alpha * jaccard_cooccurrence
            # 即使 bpp(i,j)=0, 共享伴侣多也能推断远端接触
            # alpha 控制 co-occurrence 的贡献强度
            alpha = 0.5
            w = bpp_val + alpha * jaccard

            if w > 0.01:  # 最小阈值
                candidates.append((i, j, w, bpp_val, jaccard))

    # Step 3: 按综合权重 w 降序排列 (x[2] = w, x[3] = bpp_val)
    candidates.sort(key=lambda x: -x[2])  # 按 w 排序

    # Step 4: 去冗余 (同一对附近只保留最强)
    result = []
    used = set()
    for (i, j, w, bpp_val, jaccard) in candidates:
        if len(result) >= top_k:
            break
        # 去冗余: 10nt 窗口内只保留一个
        key_red = (i // 10, j // 10)
        if key_red in used:
            continue
        used.add(key_red)
        result.append((i, j, w))

    return result


# ── 多轮 REMD 温度退火 ──

def _run_multistage_remd(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str,
    n_rounds: int = 3,
    n_replicas: int = 12,
    n_steps_per_round: int = 5000,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """多轮 REMD 温度退火: 先高温探索, 再逐步降温精修.

    参考 scheme10_full.py 的 ensemble_temperatures:
      round 0: 300-500K (高温探索, 打破局部极小)
      round 1: 250-400K (中温收敛)
      round 2: 200-350K (低温精修)

    每轮 REMD 取最低能量构象作为下轮起点.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)]
        platform_name: 平台
        n_rounds: 退火轮数
        n_replicas: 每轮副本数
        n_steps_per_round: 每轮步数
        verbose: 打印

    Returns:
        (best_energy, best_coords_ang) — P-only Å (内部 ×10 转换)
    """
    best_energy = float("inf")
    best_pos = p_coords.copy()
    L_remd = len(p_coords)  # P-only 粒子数 (输入总是 P-only)

    for rnd in range(n_rounds):
        # 每轮温度范围递降, 但不低于 280K (RNA 低温冻结)
        temp_high = max(350.0, 500.0 - rnd * 30.0)
        temp_low = max(280.0, 300.0 - rnd * 10.0)

        if verbose:
            print(f"    REMD 退火 round {rnd + 1}/{n_rounds}: "
                  f"T={temp_low:.0f}-{temp_high:.0f}K, "
                  f"{n_replicas} 副本, {n_steps_per_round} 步")

        # 用自定义温度阶梯替代默认的 300*1.1^i
        from scipy.constants import k as kB
        import multiprocessing as mp

        # 先 clamp 再算温度 (避免温度列表长度不匹配)
        n_replicas_clamped = _clamp_replicas_by_memory(n_replicas, mem_per_proc_gb=6.0)
        temperatures = [temp_low + (temp_high - temp_low) * i / max(1, n_replicas_clamped - 1)
                        for i in range(n_replicas_clamped)]

        total_threads = os.cpu_count() or 8
        per_replica_threads = max(1, total_threads // n_replicas_clamped)
        n_replicas = n_replicas_clamped

        ctx = mp.get_context("spawn")
        processes = []
        conns = []
        for ri in range(n_replicas):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_run_remd_worker,
                args=(ri, best_pos, pairs, temperatures[ri], n_steps_per_round,
                      max(10, n_steps_per_round // 10), per_replica_threads,
                      child_conn, False),  # 全力场 REMD
            )
            p.start()
            child_conn.close()
            processes.append(p)
            conns.append(parent_conn)

        # 协调交换
        accept_count = 0
        round_best_e = float("inf")
        round_best_pos = best_pos / 10.0  # Å → nm (workers report in nm)

        # 等 init: ("init", worker_idx, energy) — 3 元素, 无坐标
        for ri in range(n_replicas):
            msg = conns[ri].recv()
            if msg[0] == "init" and msg[2] < round_best_e:
                round_best_e = msg[2]

        n_ex = n_steps_per_round // max(10, n_steps_per_round // 10)
        for _ in range(n_ex):
            energies = [None] * n_replicas
            positions = [None] * n_replicas
            for ri in range(n_replicas):
                msg = conns[ri].recv()
                if msg[0] == "report":
                    energies[ri] = msg[2]
                    positions[ri] = msg[3]
                    if msg[2] < round_best_e:
                        round_best_e = msg[2]
                        round_best_pos = msg[3].copy()

            swap_decisions = [False] * (n_replicas - 1)
            for ri in range(n_replicas - 1):
                ui, uj = energies[ri], energies[ri + 1]
                beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
                beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
                exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
                if np.random.random() < min(1.0, np.exp(exponent)):
                    swap_decisions[ri] = True
                    accept_count += 1

            for ri in range(n_replicas):
                new_pos = None
                if ri > 0 and swap_decisions[ri - 1]:
                    new_pos = positions[ri - 1]
                elif ri < n_replicas - 1 and swap_decisions[ri]:
                    new_pos = positions[ri + 1]
                conns[ri].send(("swap", new_pos) if new_pos is not None else ("keep",))

        # 收尾: ("done", worker_idx, best_energy, best_pos)
        for ri in range(n_replicas):
            try:
                msg = conns[ri].recv()
                if msg[0] == "done" and msg[2] < round_best_e:
                    round_best_e = msg[2]
                    round_best_pos = msg[3]
            except Exception:
                pass

        for p in processes:
            p.join(timeout=10)
        for conn in conns:
            conn.close()

        # 更新全局最优
        if round_best_e < best_energy:
            best_energy = round_best_e
            best_pos = round_best_pos * 10.0  # nm → Å
            # worker 返回 3-bead 坐标 (3L×3), 但下一轮 worker 期望
            # P-only 输入 (L×3) 来构建3-bead系统. 提取 P bead 防止
            # 3-bead→9-bead 膨胀导致坐标垃圾.
            if best_pos.shape[0] == 3 * L_remd:
                best_pos = best_pos[0::3].copy()
        # 确保 best_pos 始终是 P-only (L×3), 防止 worker 收到3-bead 输入
        if best_pos.shape[0] != L_remd:
            if verbose:
                print(f"    [REMD] best_pos shape 异常 ({best_pos.shape}), 强制提取 P-only")
            best_pos = best_pos[0::3].copy() if best_pos.shape[0] == 3 * L_remd else best_pos[:L_remd]

        if verbose:
            rate = accept_count / max(1, n_ex * (n_replicas - 1))
            print(f"    REMD round {rnd + 1}: E={round_best_e:.0f}, "
                  f"交换率 {rate:.1%}")

    return best_energy, best_pos


# ── 势能引导精修 ──

def _potential_guided_refine(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    sequence: str,
    secondary_structure: str,
    n_minimize: int = 3000,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """势能引导精修: 对 REMD 最低能量构象做额外 OpenMM 最小化 + 短 MD.

    参考 scheme10_full.py 的 DynamicEnsembleGenerator:
      potential_weight=0.1, potential_refine_steps=10
    用全 3-bead 力场 (含堆叠/键角/碰撞) 精修, 而非极简力场.

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)]
        sequence: RNA 序列
        secondary_structure: 二级结构
        n_minimize: 最小化步数
        verbose: 打印

    Returns:
        (refined_energy, refined_coords_nm)
    """
    L = len(p_coords)

    # 清洗坐标
    p_coords = _sanitize_p_coords(p_coords.copy())

    # 构建完整 3-bead 力场 (含堆叠/键角/碰撞)
    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(p_coords, pairs, pair_scale=1.0, bsj_k_scale=1.0)

    topo = _create_3bead_topology(L)
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    n_threads = os.cpu_count() or 8
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # 阶段1: 能量最小化
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=n_minimize)

    # 阶段2: 短 MD 精修 (300K, 5ps)
    sim.step(2500)

    # 阶段3: 终局最小化
    sim.minimizeEnergy(
        tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=n_minimize)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    e = state.getPotentialEnergy()._value
    pos = state.getPositions(asNumpy=True)._value  # nm

    if verbose:
        print(f"    势能精修: E={e:.0f} kJ/mol")

    return e, pos


# ── 坐标清洗和紧凑化 ──

def _sanitize_p_coords(p_coords: np.ndarray) -> np.ndarray:
    """清洗 P 坐标: 替换 NaN/Inf 为相邻有效坐标的均值."""
    L = len(p_coords)
    bad = np.any(~np.isfinite(p_coords), axis=1)
    if not np.any(bad):
        return p_coords

    n_bad = int(np.sum(bad))
    print(f"  [OpenMM GPU] 发现 {n_bad}/{L} 个 NaN/Inf P 坐标, 清洗中...")

    for i in range(L):
        if not bad[i]:
            continue
        left, right = None, None
        for j in range(i - 1, -1, -1):
            if not bad[j]:
                left = j
                break
        for j in range(i + 1, L):
            if not bad[j]:
                right = j
                break
        if left is not None and right is not None:
            alpha = (i - left) / (right - left)
            p_coords[i] = (1 - alpha) * p_coords[left] + alpha * p_coords[right]
        elif left is not None:
            p_coords[i] = p_coords[left].copy()
        elif right is not None:
            p_coords[i] = p_coords[right].copy()
        else:
            p_coords[i] = [0.0, 0.0, float(i) * 5.9]

    return p_coords


def _is_extended_helix(p_coords: np.ndarray, threshold: float = 200.0) -> bool:
    """检测 P 坐标是否是展开结构 (首末端距离远超环状 RNA 合理范围)."""
    if len(p_coords) < 2:
        return False
    end_to_end = float(np.linalg.norm(p_coords[-1] - p_coords[0]))
    return end_to_end > threshold


def _generate_compact_coords(L: int, pairs: List[Tuple[int, int, float]]) -> np.ndarray:
    """为长序列生成紧凑的环状起始坐标.

    圆环半径由 P-P 键长和序列长度决定, 加小扰动避免退化.
    """
    coords = np.zeros((L, 3), dtype=np.float64)
    circumference = L * BOND_P_NEXT
    radius = circumference / (2.0 * np.pi)

    for i in range(L):
        angle = 2.0 * np.pi * i / L
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

    rng = np.random.default_rng(42)
    coords += rng.normal(0, 0.5, coords.shape)
    return coords


def _has_nan_energy(sim: 'Simulation') -> bool:
    """检查模拟当前能量是否为 NaN/Inf."""
    try:
        state = sim.context.getState(getEnergy=True)
        e = state.getPotentialEnergy()._value
        return not np.isfinite(e)
    except Exception:
        return True


# ── 主入口: isrnacirc_cg_refine 兼容接口 ──

def openmm_gpu_refine(
    input_pdb: str,
    output_dir: str,
    sequence: str,
    secondary_structure: str,
    name: str = "refine",
    nstep: int = 100000,
    nstep_close: int = 1000,
    nstru: int = 3,
    timeout: int = 600,
    platform_name: str = "auto",
    use_remd: bool = True,
    remd_n_replicas: int = 12,
    remd_n_steps: int = 40000,
    verbose: bool = True,
    skip_cg_to_allatom: bool = False,
    use_physical_relax: bool = True,
    bpp_matrix: Optional[np.ndarray] = None,
    bpp_weight: float = 0.5,
    use_multistage_remd: bool = True,
    use_potential_refine: bool = True,
    skip_minimal_fold: bool = False,
) -> Tuple[str, float]:
    """OpenMM GPU 加速 CG MD 精修 (isrnacirc_cg_refine 兼容接口).

    替代 IsRNAcirc.exe CPU-only 精修:
    1. 读 PDB → 提取 P 坐标 + bpp 远端配对发现
    2. 3-bead CG 力场 (bpp 加权配对力)
    3. 三阶段退火 (弱→强配对+BSJ)
    4. 多轮 REMD 温度退火 (高温探索→低温精修)
    5. 势能引导精修 (全 3-bead 力场最小化+短MD)
    6. 物理约束弛豫 (键长/键角/碰撞/BSJ/WC配对)
    7. CG → 全原子 (cg_to_allatom, 可选跳过)
    8. 输出精修后 PDB

    Args:
        input_pdb: 输入 PDB 路径
        output_dir: 输出目录
        sequence: RNA 序列
        secondary_structure: 二级结构
        name: 项目名
        nstep: 退火步数 (每阶段), 默认 20000
        nstep_close: (兼容参数, 未使用)
        nstru: (兼容参数, 未使用)
        timeout: 超时秒数
        platform_name: "auto"/"CUDA"/"OpenCL"/"CPU"
        use_remd: 是否启用 REMD
        remd_n_replicas: REMD 副本数, 默认 6
        remd_n_steps: REMD 步数, 默认 3000
        verbose: 打印详细信息
        skip_cg_to_allatom: 跳过内部 CG→全原子转换 (输入已是全原子时用)
        use_physical_relax: 是否启用物理约束弛豫 (默认 True)
        bpp_matrix: (L,L) ViennaRNA 配对概率矩阵 (可选, 用于 bpp 加权配对力)
        bpp_weight: bpp 加权系数 (0=纯硬编码, 1=纯bpp)
        use_multistage_remd: 是否启用多轮 REMD 温度退火 (默认 True)
        use_potential_refine: 是否启用势能引导精修 (默认 True)

    Returns:
        (output_pdb_path, final_energy)
    """
    if not OPENMM_AVAILABLE:
        raise ImportError(
            "OpenMM 未安装, 无法使用 GPU 精修。"
            "请安装 OpenMM: conda install -c conda-forge openmm")

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 检测平台
    platform = detect_best_platform(platform_name)
    if verbose:
        print(f"  [OpenMM GPU] 平台: {platform}")

    # 1. 读 P 坐标
    p_coords = _read_p_coords(input_pdb)
    L = len(p_coords)
    if verbose:
        print(f"  [OpenMM GPU] 序列长度: {L} nt")

    if L < 3:
        raise ValueError(f"序列太短 ({L} nt), 无法做 CG MD")

    # 1b. 清洗 NaN/Inf 坐标
    p_coords = _sanitize_p_coords(p_coords)

    # 从 pairs 参数解析配对 (从 secondary_structure 推断)
    pairs = _dotbracket_to_pairs(secondary_structure)

    # 1c. bpp 远端配对发现: 从 bpp 矩阵补充远端配对
    if bpp_matrix is not None and bpp_matrix.shape[0] == L:
        far_pairs_discovered = discover_far_pairs_from_bpp(
            bpp_matrix, sequence, min_gap=24, bpp_threshold=0.01,
            top_k=50, existing_pairs=pairs)
        if far_pairs_discovered:
            pairs = pairs + far_pairs_discovered
            if verbose:
                print(f"  [bpp] 发现 {len(far_pairs_discovered)} 个远端配对, "
                      f"总计 {len(pairs)} 对")

    # 1d. 检查坐标质量, 仅在无效/键长异常时替换为紧凑环状坐标
    # 注意: 首末距大不代表展开 — 环状 RNA 首末距天然可大.
    # 检查 P-P 键长: 若平均键长异常 (远超 5.9A 合理范围), 才判定为坏结构.
    avg_pp = 0.0
    if L > 1:
        diffs = p_coords[1:] - p_coords[:-1]
        pp_dists = np.linalg.norm(diffs, axis=1)
        avg_pp = float(np.mean(pp_dists[:min(L - 1, 500)]))

    # 如果键长在 nm 尺度 (<1.5A), 说明坐标单位是 nm 而非 Å, 乘10转 Å
    # 5.9A 不会误触发, 0.59nm 正确转换
    if avg_pp < 1.5 and L > 1:
        p_coords = p_coords * 10.0
        avg_pp = avg_pp * 10.0
        if verbose:
            print(f"  [OpenMM GPU] 坐标单位修复: avg_pp {avg_pp/10:.2f} -> {avg_pp:.2f}A")

    use_compact = (not np.isfinite(avg_pp)) or avg_pp > 20.0 or avg_pp < 1.0
    if use_compact:
        if verbose:
            print(f"  [OpenMM GPU] P-P 键长异常 (avg={avg_pp:.2f}A), "
                  f"生成紧凑环状起始坐标...")
        p_coords = _generate_compact_coords(L, pairs)

    # 2. 构建系统 (bpp 加权配对力)
    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.1,
                                bpp_matrix=bpp_matrix, bpp_weight=bpp_weight)

    # 3. 创建拓扑和模拟
    topo = _create_3bead_topology(L)

    try:
        plat = Platform.getPlatformByName(platform)
    except Exception:
        plat = Platform.getPlatformByName("Reference")

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    # CPU 平台设多线程 (默认只用1个核)
    plat_props = {}
    if plat.getName() == "CPU":
        n_threads = os.cpu_count() or 8
        plat_props["CpuThreads"] = str(n_threads)
        if verbose:
            print(f"  [OpenMM GPU] CPU 线程数: {n_threads}")

    sim = Simulation(topo, system, integrator, plat, plat_props)
    try:
        sim.context.setPositions(coords_nm * unit.nanometer)
    except Exception as e_pos:
        # GPU 内存不足 (LLVM ERROR), 回退到 CPU
        if verbose:
            print(f"  [OpenMM GPU] 平台 {platform} 失败: {e_pos}, 回退到 CPU...")
        plat = Platform.getPlatformByName("CPU")
        n_threads = os.cpu_count() or 8
        plat_props = {"CpuThreads": str(n_threads)}
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond,
        )
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)

    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    if verbose:
        print(f"  [OpenMM GPU] 初始能量: {e0:.0f} kJ/mol")

    # 4. 检查初始能量, NaN/Inf 时用更紧凑的坐标重试
    if not np.isfinite(e0):
        if verbose:
            print(f"  [OpenMM GPU] 初始能量异常 ({e0}), "
                  f"用更紧凑的环状坐标重试...")
        compact_r = max(10.0, L * BOND_P_NEXT / (2.0 * np.pi) * 0.3)
        rng = np.random.default_rng(123)
        p_fb = np.zeros((L, 3), dtype=np.float64)
        for i in range(L):
            angle = 2.0 * np.pi * i / L
            p_fb[i] = [compact_r * np.cos(angle),
                        compact_r * np.sin(angle), 0.0]
        p_fb += rng.normal(0, 0.3, p_fb.shape)
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(p_fb, pairs,
                                    pair_scale=1.0, bsj_k_scale=0.05)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
        if verbose:
            print(f"  [OpenMM GPU] 重试初始能量: {e0:.0f} kJ/mol")

    # 5. 最小化
    try:
        sim.minimizeEnergy(
            tolerance=100.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=1000)
    except Exception as e:
        if verbose:
            print(f"  [OpenMM GPU] 最小化异常: {e}")

    # 最小化后检查, 如果还是 NaN 尝试极紧凑起始
    if _has_nan_energy(sim):
        if verbose:
            print(f"  [OpenMM GPU] 最小化后能量异常, "
                  f"极紧凑起始+弱力重试...")
        compact_r2 = max(8.0, L * BOND_P_NEXT / (2.0 * np.pi) * 0.15)
        rng2 = np.random.default_rng(456)
        p_v3 = np.zeros((L, 3), dtype=np.float64)
        for i in range(L):
            angle = 2.0 * np.pi * i / L
            p_v3[i] = [compact_r2 * np.cos(angle),
                         compact_r2 * np.sin(angle), 0.0]
        p_v3 += rng2.normal(0, 0.2, p_v3.shape)
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(p_v3, pairs,
                                    pair_scale=0.1, bsj_k_scale=0.01)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)
        sim.minimizeEnergy(
            tolerance=500.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=200)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
        if verbose:
            print(f"  [OpenMM GPU] V3 初始能量: {e0:.0f} kJ/mol")

    # 6. 两阶段折叠+精修
    # 阶段1: 极简力场折叠 (P骨架 + P-P配对, 无 clash) → 配对收敛
    # 阶段2: 完整力场 REMD 精修 (从折叠后坐标出发)
    # skip_minimal_fold: REMD迭代时跳过极简折叠,直接用上轮精修坐标
    if skip_minimal_fold:
        if verbose:
            print(f"  [OpenMM GPU] 跳过极简折叠 (热启动模式), 直接用输入坐标")
        # p_coords 来自全原子 PDB 读取, 需要只取 P 原子坐标
        # 如果 p_coords 行数 > L, 说明读到了全原子, 需要过滤
        if len(p_coords) > L:
            p_only = p_coords[:L]  # 全原子 PDB 的前 L 行是 P (如果格式正确)
            if verbose:
                print(f"  [OpenMM GPU] 输入坐标 {len(p_coords)} 原子, 取前 {L} 个 P 坐标")
        else:
            p_only = p_coords
        anneal_pos_ang = p_only  # P-only 坐标
        # 计算初始能量 (用正确的 P-only 坐标)
        try:
            _tmp_topo = app.Topology()
            _tmp_chain = _tmp_topo.addChain()
            _tmp_res = _tmp_topo.addResidue("RNA", _tmp_chain)
            for _ in range(L):
                _tmp_topo.addAtom("P", app.Element.getBySymbol("P"), _tmp_res)
            _tmp_sys = mm.System()
            for _ in range(L):
                _tmp_sys.addParticle(110.0)
            # 简单配对力
            _tmp_cf = mm.CustomBondForce("4.0 * (1.0 / r^12 - 1.0 / r^6) * epsilon")
            _tmp_cf.addGlobalParameter("epsilon", 0.5)
            for pair in pairs:
                i, j = pair[0], pair[1]
                if i < L and j < L:
                    _tmp_cf.addBond(i, j)
            _tmp_sys.addForce(_tmp_cf)
            _tmp_int = LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
            _tmp_sim = Simulation(_tmp_topo, _tmp_sys, _tmp_int, plat, plat_props)
            _tmp_positions = [mm.Vec3(p_only[k, 0], p_only[k, 1], p_only[k, 2]) * unit.angstrom
                              for k in range(min(L, len(p_only)))]
            _tmp_sim.context.setPositions(_tmp_positions)
            _tmp_e = _tmp_sim.context.getState(getEnergy=True).getPotentialEnergy()._value
            anneal_e = _tmp_e
            if verbose:
                print(f"  [OpenMM GPU] 热启动初始能量: {anneal_e:.0f} kJ/mol")
            del _tmp_sim, _tmp_int, _tmp_sys
        except Exception as e_energy:
            anneal_e = 0.0
            if verbose:
                print(f"  [OpenMM GPU] 热启动能量计算失败: {e_energy}")
    else:
        if verbose:
            print(f"  [OpenMM GPU] 极简力场折叠 ({nstep} 步, 多进程并行)...")
        n_traj = _clamp_replicas_by_memory(2, mem_per_proc_gb=6.0)
        if verbose:
            print(f"  [OpenMM GPU] 极简折叠: {n_traj} 轨迹, 内存感知限制")
        try:
            anneal_e, anneal_pos_ang = _run_parallel_minimal_annealing(
                p_coords, pairs, n_anneal=nstep, n_trajectories=n_traj,
                platform_name="CPU", verbose=verbose)
        except Exception as e_anneal_par:
            if verbose:
                print(f"  [OpenMM GPU] 极简折叠失败: {e_anneal_par}, 回退顺序退火...")
            anneal_e, anneal_pos_ang = _run_annealing(
                sim, pair_force, bsj_force, bsj_guide, L,
                n_anneal=nstep, verbose=verbose)
            anneal_pos_ang = anneal_pos_ang[0::3] * 10.0  # 3-bead nm → P Å

        if verbose:
            print(f"  [OpenMM GPU] 折叠后能量: {anneal_e:.0f} kJ/mol")

    # 5b. 远端配对预拉: 低温+强远端力, 专门把远端配对拉到位
    if anneal_e < 100 and L > 50:
        try:
            _sys_fr, _cfr, _pf_fr = _build_minimal_system_gpu(
                anneal_pos_ang, pairs, pair_scale=3.0)
            _topo_fr = _create_minimal_topology(L)
            _int_fr = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
            _plat_fr = mm.Platform.getPlatformByName("CPU")
            _nthreads = max(1, (os.cpu_count() or 8) // 2)
            _sim_fr = app.Simulation(_topo_fr, _sys_fr, _int_fr, _plat_fr,
                                     {"CpuThreads": str(_nthreads)})
            _sim_fr.context.setPositions(_cfr * unit.nanometer)
            _sim_fr.minimizeEnergy(maxIterations=5000)
            _sim_fr.integrator.setTemperature(320 * unit.kelvin)
            _sim_fr.step(2000)
            _sim_fr.integrator.setTemperature(300 * unit.kelvin)
            _sim_fr.step(2000)
            _sim_fr.minimizeEnergy(
                tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=3000)
            _st = _sim_fr.context.getState(getPositions=True, getEnergy=True)
            _fr_pos = _st.getPositions(asNumpy=True)._value * 10.0  # nm → Å
            _fr_e = _st.getPotentialEnergy()._value
            # 检查是否改善
            _fr_bonds = np.linalg.norm(_fr_pos[1:] - _fr_pos[:-1], axis=1)[:100]
            if np.mean(_fr_bonds) > 3.0 and _fr_e < anneal_e:
                anneal_pos_ang = _fr_pos
                anneal_e = _fr_e
                if verbose:
                    print(f"  [OpenMM GPU] 远端预拉: E={_fr_e:.0f}, "
                          f"远端配对距离改善")
        except Exception as e:
            if verbose:
                print(f"  [OpenMM GPU] 远端预拉跳过: {e}")

    # 6. T-REMD (可选) — 从折叠后 P 坐标 (Å) 出发
    final_e = anneal_e
    final_pos_pang = anneal_pos_ang  # (L,3) Å

    # 能量已很低 (<10 kJ/mol) 时跳过 REMD (已收敛)
    if use_remd and L >= 10 and final_e > 10:
        if use_multistage_remd:
            # 多轮 REMD: 用全力场 (3-bead, 含堆叠/键角/碰撞)
            # 极简力场太简单, 温度差异不影响能量, 交换率为 0
            if verbose:
                print(f"  [OpenMM GPU] 多轮 REMD ({remd_n_replicas} 副本, 8轮, 全力场)...")
            remd_e, remd_pos = _run_multistage_remd(
                final_pos_pang, pairs, platform,
                n_rounds=8,
                n_replicas=remd_n_replicas,
                n_steps_per_round=max(5000, remd_n_steps // 3),
                verbose=verbose)
        else:
            # 单轮 REMD (全力场)
            if verbose:
                print(f"  [OpenMM GPU] T-REMD ({remd_n_replicas} 副本, {remd_n_steps} 步, 全力场)...")
            remd_e, remd_pos = _run_remd(
                final_pos_pang, pairs, platform,
                n_replicas=remd_n_replicas,
                n_steps=remd_n_steps,
                verbose=verbose,
                minimal=False)  # 全力场: 堆叠/键角/碰撞
        # REMD 是精修阶段, 成功后总是采用其坐标 (配对进一步收敛).
        if remd_pos is not None and np.isfinite(remd_e):
            final_e = remd_e
            # 诊断: 打印 REMD 返回的 shape
            if verbose:
                print(f"  [OpenMM GPU] REMD 返回: shape={remd_pos.shape}, E={remd_e:.0f}")
            # _run_multistage_remd 返回3-bead Å (内部已 ×10), _run_remd 返回
            # 3-bead nm (未转换). 下游 final_pos_pang 期望 P-only Å (L×3).
            # 统一处理: 提取 P bead, 确保单位为 Å.
            if remd_pos.ndim == 2 and remd_pos.shape[0] == 3 * L:
                p_only_nm = remd_pos[0::3].copy()  # 3-bead → P-only
                # 检测单位: 如果是 nm (键长~0.6), 乘10转Å
                _avg_pp = float(np.mean(np.linalg.norm(
                    p_only_nm[1:] - p_only_nm[:-1], axis=1)[:100]))
                if _avg_pp < 1.0:  # nm 尺度
                    final_pos_pang = p_only_nm * 10.0
                else:  # Å 尺度
                    final_pos_pang = p_only_nm
                if verbose:
                    _pp = np.linalg.norm(final_pos_pang[1:] - final_pos_pang[:-1], axis=1)
                    print(f"    REMD 3-bead→P-only: avg_PP={np.mean(_pp):.2f}A")
            elif remd_pos.ndim == 2 and remd_pos.shape[0] >= 2:
                # P-only 但可能单位是 nm
                _avg_pp = float(np.mean(np.linalg.norm(
                    remd_pos[1:] - remd_pos[:-1], axis=1)[:100]))
                if _avg_pp < 1.0:
                    final_pos_pang = remd_pos * 10.0
                else:
                    final_pos_pang = remd_pos

    # 6b. 势能引导精修: 跳过 — 力场不兼容会把极简力场坐标炸掉
    # 势能精修用 3-bead 全力场 (stacking/angle/clash) 精修极简力场坐标,
    # 但单位/参数不兼容, 导致 E 从 200K 跳到 58M kJ/mol.
    # 直接用极简折叠输出, 不再做势能精修.
    if False and use_potential_refine and L >= 10 and final_e > 1000:
        try:
            pg_e, pg_pos = _potential_guided_refine(
                final_pos_pang, pairs, sequence, secondary_structure,
                n_minimize=3000, verbose=verbose)
            if np.isfinite(pg_e) and pg_e < final_e:
                final_e = pg_e
                final_pos_pang = pg_pos * 10.0  # nm → Å
        except Exception as e:
            if verbose:
                print(f"    势能精修跳过: {e}")
    elif verbose and final_e <= 100:
        print(f"    势能精修: 跳过 (E={final_e:.0f} 已收敛)")

    # 6b. 3-bead 全力场弛豫 (与 REMD 同一力场, 保证一致性)
    # 用 _build_3bead_system_gpu 构建堆叠/角度/碰撞/配对/BSJ 全力场
    # 然后做短时间300K MD + 最小化, 提取 P 坐标
    if use_physical_relax and L >= 10:
        try:
            _sys_rx, _c_rx, _pf_rx, _sf_rx, _bjf_rx, _bjg_rx = \
                _build_3bead_system_gpu(final_pos_pang, pairs,
                                        pair_scale=1.0, bsj_k_scale=0.5)
            _topo_rx = _create_3bead_topology(L)
            _plat_rx = mm.Platform.getPlatformByName("CPU")
            _int_rx = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
            _nthreads_rx = max(1, (os.cpu_count() or 8) // 2)
            _sim_rx = app.Simulation(_topo_rx, _sys_rx, _int_rx, _plat_rx,
                                     {"CpuThreads": str(_nthreads_rx)})
            _sim_rx.context.setPositions(_c_rx * unit.nanometer)
            # 300K 短 MD (2000步) + 最小化
            _sim_rx.step(2000)
            _sim_rx.minimizeEnergy(
                tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=3000)
            _st_rx = _sim_rx.context.getState(getPositions=True, getEnergy=True)
            _rx_pos = _st_rx.getPositions(asNumpy=True)._value  # nm
            _rx_e = _st_rx.getPotentialEnergy()._value
            # 提取 P 坐标 (Å)
            _rx_p = _rx_pos[0::3] * 10.0
            _rx_bonds = np.linalg.norm(_rx_p[1:] - _rx_p[:-1], axis=1)
            _rx_avg = float(np.mean(_rx_bonds))
            # 弛豫后能量必须优于弛豫前, 否则跳过
            if _rx_avg > 3.0 and np.all(np.isfinite(_rx_p)) and _rx_e < final_e:
                final_pos_pang = _rx_p
                final_e = _rx_e
                if verbose:
                    _bsj_d = np.linalg.norm(_rx_p[0] - _rx_p[-1])
                    print(f"  [3-bead弛豫] E={_rx_e:.0f}, bond={_rx_avg:.2f}A, "
                          f"BSJ={_bsj_d:.2f}A")
            elif verbose:
                if _rx_e >= final_e:
                    print(f"  [3-bead弛豫] 跳过 (E={_rx_e:.0f} >= 当前 {final_e:.0f})")
                else:
                    print(f"  [3-bead弛豫] 跳过 (avg_bond={_rx_avg:.2f}A, 异常)")
        except Exception as e:
            if verbose:
                print(f"  [物理弛豫] 跳过: {e}")

    # 把折叠后 P 坐标 (Å) 转成 3-bead nm (补 C4'/N), 供后续输出
    rng_final = np.random.default_rng(7)
    final_pos = np.zeros((3 * L, 3), dtype=np.float64)
    for i in range(L):
        final_pos[3 * i] = final_pos_pang[i] / 10.0  # P, Å→nm
        final_pos[3 * i + 1] = final_pos_pang[i] / 10.0 + rng_final.normal(0, 0.03, 3)
        final_pos[3 * i + 2] = final_pos_pang[i] / 10.0 + rng_final.normal(0, 0.03, 3)

    # 7. CG → 全原子
    # CG_to_allatom 的模板匹配期望 P-P ~5.9Å (真实 RNA 尺度)
    # 但 OpenMM 退火后的坐标可能尺度偏大, 需要缩放
    _BOND_P_NEXT = 0.59  # 5.9Å = 0.59nm (默认值)
    try:
        from .cg_forcefield import BOND_P_NEXT as _bpn
        _BOND_P_NEXT = _bpn / 10.0  # BOND_P_NEXT 单位是 Å, 转 nm
    except ImportError:
        pass
    p_coords = final_pos[0::3]  # (L,3) P bead in nm

    # 验证: P 键长应在 0.3-1.2nm 范围 (3-12A)
    # 超出范围说明坐标单位有问题, 绝不缩放
    if L > 1:
        avg_pp_nm = float(np.mean(np.linalg.norm(p_coords[1:] - p_coords[:-1], axis=1)[:100]))
        if verbose:
            print(f"  [OpenMM GPU] CG P键长: {avg_pp_nm:.4f}nm ({avg_pp_nm*10:.2f}A)")
        # 只在合理范围内微调 (0.45-0.75nm), 超出范围不动
        if 0.45 < avg_pp_nm < 0.75 and abs(avg_pp_nm - _BOND_P_NEXT) > 0.03:
            scale = _BOND_P_NEXT / avg_pp_nm
            final_pos = final_pos * scale
            if verbose:
                print(f"  [OpenMM GPU] 坐标微调: {avg_pp_nm:.4f} -> {_BOND_P_NEXT:.4f}nm")
        elif avg_pp_nm < 0.45 or avg_pp_nm > 0.75:
            if verbose:
                print(f"  [OpenMM GPU] 键长异常 ({avg_pp_nm:.4f}nm), 跳过缩放")

    cg_pdb = str(out_path / f"{name}_cg.pdb")
    _write_allatom_pdb(final_pos, L, cg_pdb)

    if skip_cg_to_allatom:
        # 输入已是全原子 (merged_aa), 跳过重复 CG→全原子转换.
        # 只写精修后的 CG PDB, Level 2 会自行读取 P 坐标.
        output_pdb = cg_pdb
    else:
        aa_pdb = str(out_path / f"{name}_aa_raw.pdb")
        try:
            from .isrnacirc_wrapper import cg_to_allatom
            cg_to_allatom(cg_pdb, aa_pdb, sequence)
            # 检查输出文件是否有效 (至少 10 行 ATOM)
            with open(aa_pdb) as _f:
                n_atoms = sum(1 for _ in _f if _.startswith("ATOM"))
            if n_atoms < 10:
                raise RuntimeError(f"CG→全原子输出只有 {n_atoms} 个原子, 不够")
            # 验证 P 键长: cg_to_allatom 可能破坏 P 坐标
            _aa_p = _read_p_coords(aa_pdb)
            if len(_aa_p) > 1:
                _aa_bonds = np.linalg.norm(_aa_p[1:] - _aa_p[:-1], axis=1)
                _aa_avg = float(np.mean(_aa_bonds))
                # P-P 键长正常范围 4.5-7.5A, 超出则 cg_to_allatom 坐标损坏
                if _aa_avg < 4.5 or _aa_avg > 7.5:
                    if verbose:
                        print(f"  [OpenMM GPU] cg_to_allatom P键长异常 ({_aa_avg:.2f}A), 回退到 CG PDB")
                    aa_pdb = cg_pdb
        except Exception as e:
            if verbose:
                print(f"  [OpenMM GPU] CG→全原子失败: {e}, 输出 CG 坐标")
            aa_pdb = cg_pdb

        # 8. 写最终输出 PDB
        output_pdb = str(out_path / f"{name}_openmm.pdb")
        _write_refined_pdb(aa_pdb, output_pdb)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [OpenMM GPU] 完成: E={final_e:.0f} kJ/mol, "
              f"耗时 {elapsed:.1f}s")

    return output_pdb, final_e


def _dotbracket_to_pairs(ss: str) -> List[Tuple[int, int, float]]:
    """从 dot-bracket 提取配对列表 [(i,j,1.0)]."""
    pairs = []
    stack = []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i, 1.0))
    return pairs


# ── 冒烟测试 ──

def main():
    """冒烟测试: 生成随机序列, 用 OpenMM GPU 精修."""
    import random
    random.seed(42)
    L = 50
    sequence = "".join(random.choices("AUCG", k=L))
    ss = "(" * (L // 2) + ")" * (L // 2)

    # 随机初始坐标 (平面圆)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    r = L * 5.9 / (2 * np.pi)  # P-P 间距决定半径
    p_coords = np.column_stack([r * np.cos(angles), r * np.sin(angles),
                                 np.zeros(L)])

    # 写临时 PDB
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    input_pdb = Path(tmp_dir) / "test_input.pdb"
    lines = ["HEADER    test"]
    for i in range(L):
        x, y, z = p_coords[i]
        lines.append(
            f"ATOM  {i + 1:5d}  P   RA A{i + 1:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P ")
    lines.append("END")
    with open(input_pdb, "w") as f:
        f.write("\n".join(lines))

    print(f"序列: {L}nt, SS: {ss[:10]}...")
    output_pdb, energy = openmm_gpu_refine(
        str(input_pdb), tmp_dir, sequence, ss,
        name="test", nstep=100, use_remd=True,
        remd_n_replicas=3, remd_n_steps=200,
        platform_name="auto", verbose=True,
    )
    print(f"\n输出: {output_pdb}")
    print(f"能量: {energy:.0f} kJ/mol")


if __name__ == "__main__":
    main()
