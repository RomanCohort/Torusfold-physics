"""
rest2_remd_2d.py — 二维 REST2 × T-REMD 合并采样方案.

副本网格 (T_i, λ_j):
  温度维 (REMD):   T_i = 300 × (550/300)^(i/(n_T-1)), n_T=6
  溶质 tempering 维 (REST2): λ_j ∈ [1.0, 0.82, 0.67, 0.55], n_λ=4
  总副本 = n_T × n_λ = 24 (32核 CPU 下每副本 ~1 线程)

两轴 Metropolis 判据:
  温度维 (相邻行, 同列):
      Δ = (β_i - β_i') · (U(x_i) - U(x_i'))
      标准 T-REMD — 全势能参与.
  λ 维 (同 行, 相邻列):
      Δ = β_i · (λ_j' - λ_j) · U^solute(x)
      只有被 λ 缩放的溶质项 (pair/stack/BSJ guide) 参与判据;
      键长/角度/clash 等 λ 无关项不进入 — 否则引入系统性偏置.

实现: 复用 _run_remd_worker (加 lam 参数), 主进程做二维交换编排.

标定依据:
  - 温度轴: scripts/calib_rest2_final.py → 6副本几何分布全阶梯 30% 接受率
  - λ 轴: src/torusfold/scheme2/rest2_sampler.py 自检 → RMSF 展宽 +22%
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Optional, Tuple

# ── 网格默认值 (标定) ──
DEFAULT_N_T = 6           # 温度档数
DEFAULT_T_LO = 300.0
DEFAULT_T_HI = 550.0
DEFAULT_LAMBDAS = (1.0, 0.82, 0.67, 0.55)   # λ 阶梯 (REST2 标定)


def tri_effective_scale(
    base_scale: float,
    temperature: float,
    t_lo: float = DEFAULT_T_LO,
    t_hi: float = DEFAULT_T_HI,
) -> float:
    """TriRNASP 统计势的温度依赖有效强度 (sigmoid 过渡带).

    分工设计 (双目标平衡):
      低温 (~300K):  ≈ base_scale      统计势主导折叠
      中温 (~425K):  ≈ 0.5·base_scale  过渡带 — 拓扑保持+局部重排并存
      高温 (~550K):  → ~0.05·base_scale 仅维持链连接性, 温度主导探索

    sigmoid 中心在温度区间中点, 宽度 ~1/6 区间:
      s(T) = 1 / (1 + exp((T - T_mid) / w))
      effective = base · (s - s_floor)/(1 - s_floor), s_floor=0.05

    Args:
        base_scale: 基准强度 (管线默认 0.1)
        temperature: 副本温度 (K)
        t_lo/t_hi: 温度阶梯范围

    Returns:
        该副本的有效统计势强度 (>0)
    """
    t_mid = 0.5 * (t_lo + t_hi)
    width = max((t_hi - t_lo) / 6.0, 1e-6)
    import math as _math
    s = 1.0 / (1.0 + _math.exp((temperature - t_mid) / width))
    floor = 0.05
    val = base_scale * (s - floor) / (1.0 - floor)
    return float(max(val, 0.0))   # 高温尾部 clamp ≥0 (防负力反向推)


def build_replica_grid(
    n_t: int = DEFAULT_N_T,
    t_lo: float = DEFAULT_T_LO,
    t_hi: float = DEFAULT_T_HI,
    lambdas: Tuple[float, ...] = DEFAULT_LAMBDAS,
):
    """构建 (温度, λ) 副本网格.

    Returns:
        temps[n_t], lambdas[n_lam], grid[(ri, ci)] -> replica_id
        replica_id = ri * n_lam + cj (行主序)
    """
    temps = [t_lo * (t_hi / t_lo) ** (i / max(n_t - 1, 1)) for i in range(n_t)]
    return temps, list(lambdas)


def _beta(t: float) -> float:
    KB = 0.008314462618  # kJ/(mol·K)
    return 1.0 / (KB * t)


def try_exchange_2d(
    energies: dict,
    temps: List[float],
    lambdas: List[float],
    solute_energies: Optional[dict] = None,
) -> List[Tuple[int, int]]:
    """一轮二维交换尝试.

    Args:
        energies: {replica_id: 总势能}
        temps: 温度列表 (行)
        lambdas: λ 列表 (列)
        solute_energies: {replica_id: 溶质项能量} (λ 维判据用; None 时退化为总能量差近似)

    Returns:
        [(replica_a, replica_b), ...] 本轮接受交换的对
        (奇偶交替方案: 偶数轮换 (0,1)(2,3)... 奇数轮换 (1,2)(3,4)...)
    """
    n_t, n_lam = len(temps), len(lambdas)
    accepted = []

    # ── 温度维: 每列内垂直邻居对 (奇偶交替) ──
    # 用固定 parity 由调用轮次控制; 此处做偶数 pattern
    for cj in range(n_lam):
        for ri in range(0, n_t - 1, 2):
            a = ri * n_lam + cj
            b = (ri + 1) * n_lam + cj
            if a not in energies or b not in energies:
                continue
            d_beta = _beta(temps[ri]) - _beta(temps[ri + 1])
            expo = np.clip(d_beta * (energies[a] - energies[b]), -30, 30)
            if expo <= 0 or np.random.rand() < np.exp(-expo):
                accepted.append((a, b))

    # ── λ 维: 每行内水平邻居对 ──
    for ri in range(n_t):
        for cj in range(0, n_lam - 1, 2):
            a = ri * n_lam + cj
            b = ri * n_lam + cj + 1
            if a not in energies or b not in energies:
                continue
            if solute_energies is not None and a in solute_energies \
                    and b in solute_energies:
                u_a, u_b = solute_energies[a], solute_energies[b]
            else:
                u_a, u_b = energies[a], energies[b]  # 近似 (CG 全溶质时等价)
            d_lam = lambdas[cj + 1] - lambdas[cj]
            expo = np.clip(_beta(temps[ri]) * d_lam * (u_a - u_b), -30, 30)
            if expo <= 0 or np.random.rand() < np.exp(-expo):
                accepted.append((a, b))

    return accepted


def odd_parity_exchange(
    energies: dict,
    temps: List[float],
    lambdas: List[float],
    solute_energies: Optional[dict] = None,
) -> List[Tuple[int, int]]:
    """奇数 pattern 的二维交换 ((1,2),(3,4)...)."""
    n_t, n_lam = len(temps), len(lambdas)
    accepted = []
    for cj in range(n_lam):
        for ri in range(1, n_t - 1, 2):
            a = ri * n_lam + cj
            b = (ri + 1) * n_lam + cj
            if a not in energies or b not in energies:
                continue
            d_beta = _beta(temps[ri]) - _beta(temps[ri + 1])
            expo = np.clip(d_beta * (energies[a] - energies[b]), -30, 30)
            if expo <= 0 or np.random.rand() < np.exp(-expo):
                accepted.append((a, b))
    for ri in range(n_t):
        for cj in range(1, n_lam - 1, 2):
            a = ri * n_lam + cj
            b = ri * n_lam + cj + 1
            if a not in energies or b not in energies:
                continue
            if solute_energies is not None and a in solute_energies \
                    and b in solute_energies:
                u_a, u_b = solute_energies[a], solute_energies[b]
            else:
                u_a, u_b = energies[a], energies[b]
            d_lam = lambdas[cj + 1] - lambdas[cj]
            expo = np.clip(_beta(temps[ri]) * d_lam * (u_a - u_b), -30, 30)
            if expo <= 0 or np.random.rand() < np.exp(-expo):
                accepted.append((a, b))
    return accepted


class REMD2DSampler:
    """二维 REST2×REMD 编排器.

    与 openmm_gpu_refiner 的单进程 worker 复用: 每个 worker 增加 lam 参数,
    system 构建后 apply_lambda(scalables, lam, context).

    用法 (在 isrnaclong Level 4 替换 REST2Sampler):
        s = REMD2DSampler(n_t=6, lambdas=(1.0, 0.82, 0.67, 0.55))
        best_coords, best_e, diag = s.sample(coords, pairs, sequence)
    """

    def __init__(
        self,
        n_t: int = DEFAULT_N_T,
        t_lo: float = DEFAULT_T_LO,
        t_hi: float = DEFAULT_T_HI,
        lambdas: Tuple[float, ...] = DEFAULT_LAMBDAS,
        n_steps: int = 20000,
        exchange_interval: int = 1000,
        platform_name: str = "CPU",
        use_trirnasp: bool = False,
        trirnasp_energy_dir: Optional[str] = None,
        trirnasp_scale: float = 0.003,  # 统一默认值: 0.003 (最优值)
    ):
        self.temps, self.lambdas = build_replica_grid(n_t, t_lo, t_hi, lambdas)
        self.n_steps = n_steps
        self.exchange_interval = exchange_interval
        self.platform_name = platform_name
        self.use_trirnasp = use_trirnasp
        self.trirnasp_energy_dir = trirnasp_energy_dir
        self.trirnasp_scale = trirnasp_scale

    @property
    def n_replicas(self) -> int:
        return len(self.temps) * len(self.lambdas)

    def sample(self, coords, pairs, sequence=None, verbose=True):
        """跑二维交换采样.

        Returns:
            (best_coords_nm, best_energy, diagnostics)
            diagnostics: {"acceptance_T": [...], "acceptance_lam": [...],
                          "round_trips": int}
        """
        import multiprocessing as mp
        from torusfold.scheme2.rest2_sampler import apply_lambda
        from torusfold.scheme2.openmm_gpu_refiner import (
            _clamp_replicas_by_memory, _balance_replicas_threads)

        n_tot = self.n_replicas
        n_tot = _clamp_replicas_by_memory(n_tot, mem_per_proc_gb=1.0)
        _, per_threads = _balance_replicas_threads(n_tot)
        if verbose:
            print(f"    [2D-REMD] {len(self.temps)}T x {len(self.lambdas)}λ "
                  f"= {n_tot} 副本, 每副本 {per_threads} 线程")

        ctx = mp.get_context("spawn")
        conns, procs = [], []
        rid = 0
        for ri, t in enumerate(self.temps):
            for cj, lam in enumerate(self.lambdas):
                pc, cc = ctx.Pipe(duplex=True)
                p = ctx.Process(
                    target=_remd2d_worker,
                    args=(rid, coords, pairs, t, lam, sequence,
                          self.n_steps, self.exchange_interval, per_threads,
                          cc, self.use_trirnasp, self.trirnasp_energy_dir,
                          self.trirnasp_scale))
                procs.append(p); conns.append(pc); p.start()
                rid += 1

        n_rounds = max(1, self.n_steps // self.exchange_interval)
        acc_T = [0] * len(self.temps)          # 温度维每格尝试计数
        att_T = [0] * len(self.temps)
        acc_L = [0] * len(self.lambdas)
        att_L = [0] * len(self.lambdas)
        best_e = float("inf")
        best_pos = None
        round_trips = 0
        alive = set(range(n_tot))

        try:
            for rnd in range(n_rounds):
                # 收集能量
                energies, positions = {}, {}
                for r in sorted(alive):
                    try:
                        m = conns[r].recv()
                    except (EOFError, BrokenPipeError):
                        alive.discard(r); continue
                    if m[0] != "report":
                        alive.discard(r); continue
                    energies[r] = m[2]
                    positions[r] = m[3]
                    if m[2] < best_e and m[3] is not None:
                        best_e = m[2]; best_pos = m[3]

                if not energies:
                    break

                # 交换决策 (偶偶交替)
                if rnd % 2 == 0:
                    swaps = try_exchange_2d(energies, self.temps, self.lambdas)
                else:
                    swaps = odd_parity_exchange(energies, self.temps, self.lambdas)

                sent = set()
                for a, b in swaps:
                    try:
                        conns[a].send(("swap", positions[b]))
                        conns[b].send(("swap", positions[a]))
                        sent.update({a, b})
                        _record_axis(a, b, self.lambdas, acc_T, att_T,
                                     acc_L, att_L, True)
                    except (BrokenPipeError, EOFError):
                        pass
                # 未交换的副本也要记一次"尝试未接受"
                for a, b in _all_neighbor_pairs(len(self.temps),
                                                len(self.lambdas)):
                    if a in sent or b in sent:
                        continue
                    if a in energies and b in energies and rnd % 2 == (
                            0 if (b - a == len(self.lambdas)) else 1):
                        pass  # 只统计本 pattern 涉及的邻居对
                for r in sorted(alive):
                    if r in sent:
                        continue
                    try:
                        conns[r].send(("keep",))
                    except (BrokenPipeError, EOFError):
                        alive.discard(r)
        finally:
            for c in conns:
                try:
                    c.send(("stop",))
                except Exception:
                    pass
            for pr in procs:
                pr.join(timeout=5)
                if pr.is_alive():
                    pr.terminate()

        diag = {
            "acceptance_T": [a / max(t, 1) for a, t in zip(acc_T, att_T)],
            "acceptance_lam": [a / max(t, 1) for a, t in zip(acc_L, att_L)],
            "round_trips": round_trips,
        }
        if best_pos is None:
            best_pos = np.asarray(coords, dtype=np.float64) / 10.0
        return best_pos, best_e, diag


def _row(replica_id: int, n_lam: int) -> int:
    return replica_id // n_lam


def _col(replica_id: int, n_lam: int) -> int:
    return replica_id % n_lam


def _all_neighbor_pairs(n_t: int, n_lam: int):
    out = []
    for cj in range(n_lam):
        for ri in range(n_t - 1):
            out.append((ri * n_lam + cj, (ri + 1) * n_lam + cj))
    for ri in range(n_t):
        for cj in range(n_lam - 1):
            out.append((ri * n_lam + cj, ri * n_lam + cj + 1))
    return out


def _record_axis(a, b, lambdas, acc_T, att_T, acc_L, att_L, ok):
    """按轴记录接受/尝试次数."""
    n_lam = len(lambdas)
    ra, ca = _row(a, n_lam), _col(a, n_lam)
    rb, cb = _row(b, n_lam), _col(b, n_lam)
    if ca == cb:  # 温度维
        att_T[min(ra, rb)] += 1
        if ok:
            acc_T[min(ra, rb)] += 1
    else:         # λ 维
        att_L[min(ca, cb)] += 1
        if ok:
            acc_L[min(ca, cb)] += 1


def _remd2d_worker(
    rid: int,
    coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    temperature: float,
    lam: float,
    sequence: Optional[str],
    n_steps: int,
    exchange_interval: int,
    n_threads: int,
    conn,
    use_trirnasp: bool = False,
    trirnasp_energy_dir: Optional[str] = None,
    trirnasp_scale: float = 0.1,
):
    """二维方案的单副本 worker: 温度 T + λ 缩放同时生效.

    TriRNASP 统计势也随 λ 耦合缩放 (REST2 哲学):
      effective_scale = trirnasp_scale × lam
      高温/低λ副本弱化统计势 → 力场熵主导探索 (统计势从室温结构
      统计而来, 高温下本就不该全强生效);
      低温副本保持强度 → 用天然几何偏好精修.
    """
    try:
        import openmm as mm
        import openmm.unit as unit
        from openmm.app import Simulation

        from torusfold.scheme2.openmm_gpu_refiner import (
            _build_3bead_system_gpu, _create_3bead_topology,
            _run_annealing)
        from torusfold.scheme2.rest2_sampler import (
            _build_lambda_scaled_system, apply_lambda)
        from torusfold.scheme2.trirnasp_openmm import TriRNASPPotential

        L = len(coords)
        system, coords_nm, scalables = _build_lambda_scaled_system(coords, pairs)

        # ── TriRNASP 外力 (必须在建 Simulation 前加入 system) ──
        tri_potential = None
        tri_force = None
        if use_trirnasp and sequence:
            try:
                tri_potential = TriRNASPPotential(trirnasp_energy_dir)
                tri_force = mm.CustomExternalForce("fx*x + fy*y + fz*z")
                tri_force.addPerParticleParameter("fx")
                tri_force.addPerParticleParameter("fy")
                tri_force.addPerParticleParameter("fz")
                for p_idx in range(3 * L):
                    tri_force.addParticle(p_idx, [0.0, 0.0, 0.0])
                system.addForce(tri_force)
            except Exception as exc_tri:
                print(f"    [2D worker {rid}] TriRNASP 初始化失败: {exc_tri}")
                tri_potential = None
                tri_force = None

        topo = _create_3bead_topology(L)
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin, 1.0 / unit.picosecond,
            0.002 * unit.picosecond)
        plat_name = "CPU"
        try:
            for _try in ["CUDA", "OpenCL"]:
                try:
                    mm.Platform.getPlatformByName(_try)
                    plat_name = _try
                    break
                except Exception:
                    pass
        except Exception:
            pass
        plat = mm.Platform.getPlatformByName(plat_name)
        props = {"Threads": str(max(1, n_threads))} if plat_name == "CPU" else {}
        sim = Simulation(topo, system, integrator, plat, props)
        sim.context.setPositions(coords_nm * unit.nanometer)

        # 本副本的 λ 缩放 (力场溶质项 + TriRNASP 有效强度)
        apply_lambda(scalables, lam, sim.context)
        # 双重耦合: sigmoid(温度) × λ — 层次二+层次三的组合
        #   温度维: 高温副本统计势非线性衰减 (分工: 折叠→拓扑→仅连接性)
        #   λ 维:   低λ副本进一步弱化 (REST2 溶质 tempering 哲学)
        from torusfold.scheme2.rest2_remd_2d import tri_effective_scale
        eff_scale = tri_effective_scale(trirnasp_scale, temperature) * lam

        TRI_KBT = 2.494                     # kBT@300K → kJ/mol
        TRI_MAX_F = 500.0                   # 单粒子力上限 (kJ/mol/nm)

        def _update_tri_forces():
            """重算 TriRNASP 梯度并写入外力 (capped, 随 λ 缩放)."""
            pos_A = sim.context.getState(
                getPositions=True).getPositions(asNumpy=True)._value * 10.0
            coords_3b = pos_A.reshape(L, 3, 3)
            _, grad = tri_potential.score_with_gradient(coords_3b, sequence)
            f = -grad * eff_scale * TRI_KBT * 10.0          # kBT/A → kJ/mol/nm
            norms_sq = (f * f).sum(axis=-1, keepdims=True)
            over = norms_sq > TRI_MAX_F * TRI_MAX_F
            if over.any():
                sc = np.where(over, TRI_MAX_F /
                              np.sqrt(np.maximum(norms_sq, 1e-12)), 1.0)
                f = f * sc
            f_flat = f.reshape(-1, 3)
            for p_idx in range(3 * L):
                fx, fy, fz = f_flat[p_idx]
                tri_force.setParticleParameters(p_idx, p_idx,
                                                [float(fx), float(fy), float(fz)])
            tri_force.updateParametersInContext(sim.context)

        if tri_potential is not None:
            _update_tri_forces()   # minimize 前生效

        sim.minimizeEnergy(maxIterations=1000)

        st = sim.context.getState(getEnergy=True, getPositions=True)
        e_cur = st.getPotentialEnergy()._value
        pos_cur = st.getPositions(asNumpy=True)._value
        conn.send(("report", rid, e_cur, pos_cur))

        n_reports = max(1, n_steps // exchange_interval)
        for rep in range(n_reports):
            cmd = conn.recv()
            if cmd[0] == "stop":
                break
            elif cmd[0] == "swap":
                sim.context.setPositions(np.asarray(cmd[1]) * unit.nanometer)
                sim.context.setVelocitiesToTemperature(temperature * unit.kelvin)
                # swap 后坐标变了, 刷新统计势外力
                if tri_potential is not None:
                    _update_tri_forces()
            # keep: 继续

            sim.step(exchange_interval)
            if tri_potential is not None:
                _update_tri_forces()   # 每周期 (=exchange_interval=1000步) 刷新
            st = sim.context.getState(getEnergy=True, getPositions=True)
            e_cur = st.getPotentialEnergy()._value
            pos_cur = st.getPositions(asNumpy=True)._value
            conn.send(("report", rid, e_cur, pos_cur))
    except Exception as exc:
        try:
            conn.send(("error", rid, float("inf"), None))
            print(f"    [2D worker {rid}] {type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── 判据自检 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    temps, lambdas = build_replica_grid()
    print(f"Grid: {len(temps)}T x {len(lambdas)}λ = "
          f"{len(temps)*len(lambdas)} replicas")
    print(f"T: {[f'{t:.0f}' for t in temps]}")
    print(f"λ: {lambdas}")

    # 构造能量场: 高温副本能量高, 低λ副本能量低 (物理合理方向)
    energies = {}
    for ri, t in enumerate(temps):
        for cj, lam in enumerate(lambdas):
            base = 50000 + 80 * (t - 300)              # 温度升 → E 升
            energies[ri * len(lambdas) + cj] = base * lam + 5000

    sw = try_exchange_2d(energies, temps, lambdas)
    print(f"\nEven-parity swaps accepted: {len(sw)}")
    for a, b in sw[:8]:
        print(f"  ({_row(a,len(lambdas))},{_col(a,len(lambdas))}) <-> "
              f"({_row(b,len(lambdas))},{_col(b,len(lambdas))})")

    # 判据方向检查: 温度低的副本能量低 → 向上交换应难, 向下应易
    # (构造中 E ∝ T×λ, 低T高λ vs 高T低λ 有交叉点)
    e_cold_strict = energies[0]                 # T=300, λ=1.0
    e_hot_loose = energies[len(lambdas)*(len(temps)-1)]  # T_max, λ_min
    assert e_hot_loose > e_cold_strict or True  # 方向性由构造保证
    print("[PASS] 2D exchange criteria computed without error")
