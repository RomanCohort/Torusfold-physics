"""
refine.py — Scheme 2 几何初始坐标 + OpenMM 粗粒度精修。

路径 B：ViennaRNA 配对概率 → GeometricConstraintSolver 几何求解 → OpenMM 能量最小化。
每核苷酸 1 个粒子（P 原子代表）。力场：
  1. HarmonicBondForce: 相邻 P-P 键 (5.9Å) + BSJ 首尾键 (5.9Å) → 闭合
  2. CustomBondForce: ViennaRNA 配对距离约束 (10.6Å)
  3. CustomNonbondedForce: 软排斥 (消位阻) + 弱吸引 (促折叠)
  4. 能量最小化 (L-BFGS)，中等序列(401-1000nt)加 MD 退火跳出局部解

注: BOND_LEN = 5.9 是 P-O3'-P 跨距近似 (CG 每核苷酸 1 粒子)。
全原子 amber_field 从 +7 万降到 -1 万靠的不是改这个值, 而是
aform_from_template 用 1EHZ 晶体模板重建 + amber_refine 三阶段
精修 (松 P→退火→收 P)。曾误注 "改 7.0 对齐力场", 实际代码从未
改过, 7.0 是误注, 已删 (2026-07-22 校正)。
"""

from __future__ import annotations
import numpy as np

from .constraint_solver import GeometricConstraintSolver, SolverConfig

# ---------- 几何参数 (Ang) ----------
BOND_LEN = 5.9      # P-P 骨架距离 (P-O3'-P 跨距近似, CG 1 粒子/核苷酸)
PAIR_DIST = 10.6    # WC 配对 C1'-C1' (此处用 P 近似)
CLASH_DIST = 3.0    # 最小非键距离


def _import_rna():
    """lazy import ViennaRNA，缺失时给清晰报错。"""
    try:
        import RNA  # type: ignore
        return RNA
    except ImportError as e:
        raise ImportError(
            "ViennaRNA (RNA module) 未安装。请先装：conda install -c bioconda viennarna"
        ) from e


def vienna_pair_probs(sequence: str, threshold: float = 0.3, circ: bool = True):
    """ViennaRNA 算配对概率矩阵 → 提取 (i,j) 配对约束。

    circ=True (默认): ViennaRNA 环形模式 (VRNA_OPTION_CIRC), 正确处理环形 RNA
    的头尾配对. 项目是 circRNA 管线, 默认环形. 显式传 circ=False 可切回线性.

    返回 (pairs, bpp): pairs=[(i,j,prob)] 为 0-indexed，bpp 为 0-indexed 的 L×L 概率矩阵。

    注：ViennaRNA 的 fc.bpp() 返回 (L+1)×(L+1) 的 1-indexed 矩阵（[0,:] 是 padding），
    这里切 [1:,1:] 转回 0-indexed 与下游 coords 对齐。
    """
    RNA = _import_rna()
    md = RNA.md()
    md.circ = 1 if circ else 0
    fc = RNA.fold_compound(sequence, md)
    fc.pf()
    fc.mfe()
    bpp = np.array(fc.bpp())[1:, 1:]  # 1-indexed → 0-indexed
    L = bpp.shape[0]
    pairs = []
    for i in range(L):
        for j in range(i + 4, L):
            if bpp[i, j] > threshold:
                pairs.append((i, j, float(bpp[i, j])))
    return pairs, bpp


def scheme2_initial_coords(sequence: str, pairs, n_samples: int = 8):
    """Scheme 2 产出初始粗坐标 (P 原子, Ang)。失败返回 None。"""

    class CS:
        def __init__(self, L, p):
            self.seq_len = L
            self.pair_constraints = [(i, j, PAIR_DIST, w) for (i, j, w) in p]

    solver = GeometricConstraintSolver(SolverConfig(n_samples=n_samples))
    confs = solver.solve(CS(len(sequence), pairs))
    return confs[0] if confs else None


def predict_3d(sequence: str, n_samples: int = 8, platform_name: str = "CPU",
               pair_threshold: float = 0.3):
    """端到端：序列 → 3D 坐标 (N,3) Ang。

    调用链 ViennaRNA 配对 → Scheme2 几何求解 → openmm_refine。
    返回 dict: {coords, pairs, e0, e1, bsj_before, bsj_after}。

    pair_threshold=0.3: CG-only 入口, 弱配对也收 (CG 力场对错配容忍度高)。
    对比 predict_3d_allatom 用 0.5 (喂 amber 要更严), 见 __init__.py。
    """
    pairs, _ = vienna_pair_probs(sequence, pair_threshold)
    init = scheme2_initial_coords(sequence, pairs, n_samples)
    if init is None:
        raise RuntimeError(f"Scheme2 几何求解失败 (L={len(sequence)})")
    bsj0 = float(np.linalg.norm(init[0] - init[-1]))
    refined, e0, e1 = openmm_refine(init, pairs, platform_name)
    bsj1 = float(np.linalg.norm(refined[0] - refined[-1]))
    return dict(coords=refined, pairs=pairs, e0=e0, e1=e1,
                bsj_before=bsj0, bsj_after=bsj1)


def build_topology(L: int):
    """每核苷酸 1 粒子的 OpenMM 拓扑。"""
    from openmm.app import Topology, Element
    topo = Topology()
    chain = topo.addChain()
    res_name = "N"  # 通用残基
    for i in range(L):
        res = topo.addResidue(res_name, chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
    return topo


def openmm_refine(coords_angstrom: np.ndarray, pairs, platform_name: str = "CPU"):
    """用 OpenMM 粗粒度力场优化 (N,3) 坐标 (Ang)。

    pairs: [(i, j, weight), ...]
    返回: (refined_coords (N,3) Ang, e0, e1)
    """
    from openmm import (
        System, Platform, VerletIntegrator, HarmonicBondForce,
        CustomBondForce, CustomNonbondedForce, LangevinMiddleIntegrator,
    )
    from openmm import unit
    from openmm.app import Simulation

    L = len(coords_angstrom)
    coords_nm = coords_angstrom / 10.0  # Ang → nm

    topo = build_topology(L)

    system = System()
    for _ in range(L):
        system.addParticle(330.0)  # 核苷酸粗粒质量 (~330 Da)

    # 1. 骨架键 + BSJ 键 (harmonic, nm)
    bond_k = 50000.0  # kJ/mol/nm^2 (强, 保闭合)
    bond_force = HarmonicBondForce()
    for i in range(L - 1):
        bond_force.addBond(i, i + 1, BOND_LEN / 10.0, bond_k)
    bond_force.addBond(L - 1, 0, BOND_LEN / 10.0, bond_k)  # BSJ 首尾键 - 强制闭合
    system.addForce(bond_force)

    # 2. 配对距离约束 (CustomBond, nm)
    pair_k = 2500.0
    pair_force = CustomBondForce("0.5*k*(r-r0)^2")
    pair_force.addPerBondParameter("k")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if 0 <= i < L and 0 <= j < L and abs(i - j) > 1 and not (i == 0 and j == L - 1):
            pair_force.addBond(i, j, [pair_k * w, PAIR_DIST / 10.0])
    system.addForce(pair_force)

    # 3. 软排斥 + 弱吸引 (促立体化)
    nb = CustomNonbondedForce(
        "step(CLASH-r)*K_rep*(CLASH-r)^2 - K_attr*step(r-CLASH)*exp(-(r-CLASH)/sigma)"
    )
    nb.addGlobalParameter("CLASH", CLASH_DIST / 10.0)
    nb.addGlobalParameter("K_rep", 10000.0)   # 消融敲定: 强排斥 (原2000)
    nb.addGlobalParameter("K_attr", 1.0)
    nb.addGlobalParameter("sigma", 1.0)  # nm
    nb.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
    for i in range(L):
        nb.addParticle([])
    # 排除 1-2 (相邻 + BSJ)
    for i in range(L - 1):
        nb.addExclusion(i, i + 1)
    nb.addExclusion(L - 1, 0)
    system.addForce(nb)

    # --- 积分器 + 平台 ---
    # 中等序列(401-1000nt)用 Langevin (支持后续 MD 退火); 其余用 Verlet (纯最小化)
    use_md = 401 <= L <= 1000
    integrator = (LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond)
                  if use_md else VerletIntegrator(0.001 * unit.picosecond))
    try:
        platform = Platform.getPlatformByName(platform_name)
        sim = Simulation(topo, system, integrator, platform)
    except Exception:
        sim = Simulation(topo, system, integrator)

    sim.context.setPositions(coords_nm * unit.nanometer)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy()._value

    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=2000)

    # 中等序列(401-1000nt)专项: 小剂量 MD 退火跳出局部解
    # 消融结论: 短序列(≤400)和长序列(>1000)无需 MD; 中等序列卡局部解,需 MD 破局
    # 最佳: 100步@500K → 降温到300K(50步) → 重最小化. 大剂量(200+步/800K+)反而过度打散
    # 安全网: MD 偶发暴走(冲突暴增), 保留 MD 前结果, 取冲突少者
    pre_md_state = sim.context.getState(getPositions=True, getEnergy=True) if use_md else None
    if use_md:
        sim.integrator.setTemperature(500 * unit.kelvin)
        sim.step(100)                       # 高温打散局部对峙
        sim.integrator.setTemperature(300 * unit.kelvin)
        sim.step(50)                        # 降温
        sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=2000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # 安全网: MD 若使能量显著升高(暴走), 回退到 MD 前的纯最小化结果
    if use_md and pre_md_state is not None:
        e_pre = pre_md_state.getPotentialEnergy()._value
        if e1 > e_pre * 0.5 and e_pre < 0:
            pos = pre_md_state.getPositions(asNumpy=True)._value
            e1 = e_pre

    refined_angstrom = pos * 10.0  # nm → Ang
    return refined_angstrom, e0, e1
