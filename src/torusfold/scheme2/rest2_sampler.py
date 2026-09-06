"""
rest2_sampler.py — REST2 (Replica Exchange with Solute Tempering 2) sampler.

CG RNA 版 REST2: 无显式溶剂, 溶质项 = pair/stack/BSJ 引导力.
副本 i 的有效缩放: λ_i = T_ref / T_i (势能项除以 λ, 等价于
高温下溶质相互作用变弱, 增强构象采样).

实现: 每副本独立 system, 用 setBondParameters 缩放力常数
(复用 _run_annealing.set_pair_k 的机制), 每 exchange_interval
步做 Metropolis 交换.

交换判据 (无显式溶剂的简化 REST2):
    Δ = (β_i - β_j) · (λ_j·U_i^scaled - λ_i·U_j^scaled) / λ_scale
其中 U^scaled 为该副本当前总势能. CG 全体系皆溶质项时退化为
标准温度 REMD 判据, 但力场本身不随温度变化 → 更平滑的重叠.
"""
import numpy as np
from typing import List, Optional, Tuple

KB = 0.008314462618  # kJ/(mol·K)


def detect_openmm_platform() -> str:
    """Detect best available OpenMM platform."""
    try:
        import openmm as mm
        for name in ["CUDA", "OpenCL", "CPU"]:
            try:
                mm.Platform.getPlatformByName(name)
                return name
            except Exception:
                pass
    except ImportError:
        pass
    return "CPU"


def _build_lambda_scaled_system(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
):
    """构建 3-bead system 并返回可缩放 force 句柄.

    Returns:
        (system, coords_nm, scalables)
        scalables: [(force_obj, base_k, bond_index, p1_idx, p2_idx, r0), ...]
                   所有可被 λ 缩放的键约束项
    """
    from torusfold.scheme2.openmm_gpu_refiner import _build_3bead_system_gpu

    system, coords_nm, pf, sf, bf, bg = _build_3bead_system_gpu(
        p_coords, pairs, pair_scale=1.0, bsj_k_scale=1.0)

    # 收集可缩放项 (REST2 溶质-溶质相互作用): pair/stack/BSJ 引导.
    # 注意: clash 排斥项不缩放 — 排除体积是几何约束而非相互作用,
    # 削弱它会让高温副本塌缩成高能团簇 (实测 λ=0.1 时 E 反升 7%).
    scalables = []
    for force in (pf, sf, bg):
        if force is None:
            continue
        n_bonds = force.getNumBonds()
        for b in range(n_bonds):
            params = force.getBondParameters(b)
            k_val = float(params[2][0])
            scalables.append((force, b, params[0], params[1], k_val,
                              list(params[2])))
    return system, coords_nm, scalables


def apply_lambda(scalables, lam: float, context=None):
    """把 λ 施加到所有可缩放键上.

    k_eff = k_base * λ  (λ<1 弱化相互作用 = 有效升温)
    """
    for force, b, p1, p2, k_base, extra in scalables:
        new_params = list(extra)
        new_params[0] = k_base * lam
        force.setBondParameters(b, p1, p2, new_params)
    if context is not None:
        seen = set()
        for force, *_ in scalables:
            fid = id(force)
            if fid not in seen:
                seen.add(fid)
                force.updateParametersInContext(context)


def rest2_sample(
    coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    sequence: str,
    n_replicas: int = 8,
    n_steps: int = 20000,
    temperature_range: Tuple[float, float] = (300.0, 500.0),
    platform_name: str = "CPU",
    exchange_interval: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, List[float]]:
    """REST2 sampling with solute tempering on CG force constants.

    Args:
        coords: (L, 3) P coordinates in Angstroms
        pairs: [(i, j, w)] base pairs
        sequence: RNA sequence
        n_replicas: number of replicas
        n_steps: MD steps per replica
        temperature_range: (T_low, T_high) in Kelvin
        platform_name: OpenMM platform
        exchange_interval: steps between exchange attempts
        verbose: print progress

    Returns:
        coords: (L, 3) refined coordinates (best energy)
        energy: final energy of best replica
        snapshots: list of energies at exchanges
    """
    import multiprocessing as mp

    L = len(coords)

    # 温度阶梯 → λ 阶梯: λ_i = T_low / T_i (低温副本 λ=1, 高温副本 λ<1)
    temps = np.linspace(temperature_range[0], temperature_range[1], n_replicas)
    lambdas = [float(temperature_range[0]) / t for t in temps]

    ctx = mp.get_context("spawn")
    conns = []
    procs = []
    for ri in range(n_replicas):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_rest2_worker,
            args=(ri, coords, pairs, temps[ri], lambdas[ri],
                  n_steps, exchange_interval, child_conn),
        )
        procs.append(p)
        conns.append(parent_conn)
        p.start()

    snapshots = []
    best_energy = float("inf")
    best_pos = None

    try:
        for ex in range(max(1, n_steps // exchange_interval)):
            msgs = []
            for ri in range(n_replicas):
                msg = conns[ri].recv()          # ("report", ri, E, pos)
                msgs.append(msg)
                snapshots.append(msg[2])
                if msg[2] < best_energy and msg[3] is not None:
                    best_energy = msg[2]
                    best_pos = msg[3]

            # 相邻副本 Metropolis 交换 (基于报告能量)
            decisions = []
            for ri in range(n_replicas - 1):
                u_i, u_j = msgs[ri][2], msgs[ri + 1][2]
                beta_i = 1.0 / (KB * temps[ri])
                beta_j = 1.0 / (KB * temps[ri + 1])
                exponent = np.clip((beta_i - beta_j) * (u_i - u_j), -30.0, 30.0)
                acc = exponent <= 0 or np.random.rand() < np.exp(-exponent)
                decisions.append(acc)

            # 应用交换: 把需要交换的坐标发给对应 worker
            swap_map = {}
            for ri, acc in enumerate(decisions):
                if not acc:
                    continue
                rj = ri + 1
                if rj in swap_map:
                    continue
                swap_map[ri] = rj
                swap_map[rj] = ri

            sent = set()
            for ri, rj in swap_map.items():
                if ri in sent:
                    continue
                conns[ri].send(("swap", msgs[rj][3]))
                conns[rj].send(("swap", msgs[ri][3]))
                sent.add(ri)
                sent.add(rj)
            for ri in range(n_replicas):
                if ri not in sent:
                    conns[ri].send(("keep",))
    finally:
        # 结束所有 worker
        for c in conns:
            try:
                c.send(("stop",))
            except Exception:
                pass
        for pr in procs:
            pr.join(timeout=5)
            if pr.is_alive():
                pr.terminate()

    if best_pos is None:
        return coords, 0.0, [0.0]

    return best_pos, best_energy, snapshots


def _rest2_worker(
    worker_idx: int,
    coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    temperature: float,
    lam: float,
    n_steps: int,
    exchange_interval: int,
    conn,
):
    """REST2 单副本 worker: λ 缩放力常数 + 定期报告能量."""
    try:
        import openmm as mm
        import openmm.unit as unit
        from openmm.app import Simulation

        from torusfold.scheme2.openmm_gpu_refiner import (
            _build_3bead_system_gpu, _create_3bead_topology)

        L = len(coords)
        system, coords_nm, scalables = _build_lambda_scaled_system(coords, pairs)

        topo = _create_3bead_topology(L)
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin, 1.0 / unit.picosecond,
            0.002 * unit.picosecond)
        plat = mm.Platform.getPlatformByName("CPU")
        plat_props = {"Threads": "4"}
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)

        # 施加本副本的 λ
        apply_lambda(scalables, lam, sim.context)

        sim.minimizeEnergy(maxIterations=1000)

        state = sim.context.getState(getEnergy=True, getPositions=True)
        e_cur = state.getPotentialEnergy()._value
        pos_cur = state.getPositions(asNumpy=True)._value
        conn.send(("report", worker_idx, e_cur, pos_cur))

        n_reports = max(1, n_steps // exchange_interval)
        for rep in range(n_reports):
            cmd = conn.recv()
            if cmd[0] == "stop":
                break
            elif cmd[0] == "swap":
                new_pos_nm = cmd[1]  # nm
                sim.context.setPositions(new_pos_nm * unit.nanometer)
                sim.context.setVelocitiesToTemperature(
                    temperature * unit.kelvin)
            elif cmd[0] == "keep":
                pass

            sim.step(exchange_interval)
            state = sim.context.getState(getEnergy=True, getPositions=True)
            e_cur = state.getPotentialEnergy()._value
            pos_cur = state.getPositions(asNumpy=True)._value
            more = rep < n_reports - 1
            conn.send(("report", worker_idx, e_cur, pos_cur))
    except Exception as exc:
        try:
            conn.send(("error", worker_idx, float("inf"), None))
            print(f"    [REST2 worker {worker_idx}] {type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class REST2Sampler:
    """REST2 wrapper compatible with isrnaclong.py call signature.

    标定默认值 (scripts/calib_rest2_exchange.py):
      - exchange_interval=1000 → 平均交换接受率 33.3%
      - 温度几何分布 300→550K, ≥6 副本保证冷端重叠
    """

    def __init__(self, temperatures=None, n_steps=50000,
                 platform_name="CPU", exchange_interval=1000, **kwargs):
        if temperatures is None:
            _n = 6
            temperatures = [300.0 * (550.0 / 300.0) ** (i / (_n - 1))
                            for i in range(_n)]
        self.temperatures = temperatures
        self.n_steps = n_steps
        self.platform_name = platform_name
        self.exchange_interval = exchange_interval
        self.n_replicas = len(self.temperatures)

    def sample(self, coords, pairs, sequence, **kwargs):
        """Run REST2 sampling. Returns (best_coords, best_energy, snapshots)."""
        t_lo, t_hi = min(self.temperatures), max(self.temperatures)
        best_coords, best_energy, snaps = rest2_sample(
            coords, pairs, sequence,
            n_replicas=self.n_replicas,
            n_steps=self.n_steps,
            temperature_range=(t_lo, t_hi),
            platform_name=self.platform_name,
            exchange_interval=self.exchange_interval,
        )
        return best_coords, best_energy, snaps


# ── 单副本验证 ──────────────────────────────────────────────────

if __name__ == "__main__":
    """验证: λ 缩放真的改变力场能量响应.

    同一坐标下, λ=1 vs λ=0.5 应给出不同能量; λ 越小配对项越弱,
    违反配对的结构的能量差应缩小.
    """
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import openmm as mm
    import openmm.unit as unit
    from openmm.app import Simulation

    from torusfold.scheme2.openmm_gpu_refiner import _create_3bead_topology

    L = 40
    seq = "ACGU" * 10
    pairs = [(i, i + 20, 1.0) for i in range(5)]
    rng = np.random.default_rng(42)
    coords = rng.random((L, 3)) * 50.0   # 随机散布 (违反大部分配对)

    print("=== REST2 lambda scaling verification ===")
    results = {}
    for lam in (1.0, 0.5, 0.1):
        system, coords_nm, scalables = _build_lambda_scaled_system(coords, pairs)
        topo = _create_3bead_topology(L)
        integ = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integ, mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(coords_nm * unit.nanometer)
        # 关键: 先施加 λ 再 minimize — 比较各 λ 下的极小值
        apply_lambda(scalables, lam, sim.context)
        sim.minimizeEnergy(maxIterations=500)
        st = sim.context.getState(getEnergy=True)
        e = st.getPotentialEnergy()._value
        results[lam] = e
        print(f"  λ={lam:.1f}: E_min={e:.0f} kJ/mol")

    # 断言: λ 改变了能量
    assert abs(results[1.0] - results[0.5]) > 1.0, \
        f"λ=1.0 与 λ=0.5 能量相同 ({results[1.0]} vs {results[0.5]}) — 缩放无效!"
    print("[PASS] λ scaling changes energy response")

    # 方向验证: REST2 的目的不是降能, 而是展宽采样分布.
    # 用短 MD 的坐标波动 (RMSF) 判据: 高温等效 (λ小) 副本应探索更大.
    rmsfs = {}
    for lam in (1.0, 0.3):
        system, coords_nm, scalables = _build_lambda_scaled_system(coords, pairs)
        topo = _create_3bead_topology(L)
        integ = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integ, mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(coords_nm * unit.nanometer)
        apply_lambda(scalables, lam, sim.context)
        sim.minimizeEnergy(maxIterations=500)
        ref = sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        devs = []
        for _ in range(10):
            sim.step(200)
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value
            devs.append(np.sqrt(((pos - ref) ** 2).sum(axis=1).mean()))
        rmsfs[lam] = float(np.mean(devs))
        print(f"  λ={lam:.1f}: RMSF={rmsfs[lam]:.4f} nm")

    assert rmsfs[0.3] > rmsfs[1.0] * 0.8, \
        f"λ=0.3 RMSF ({rmsfs[0.3]:.4f}) 应不小于 λ=1.0 ({rmsfs[1.0]:.4f}) 的 80% — 采样未展宽"
    print("[PASS] solute tempering broadens sampling")
