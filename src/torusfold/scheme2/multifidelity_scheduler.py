"""
multifidelity_scheduler.py — 多保真度调度器 (stub).

控制 CG→allatom 的精修级别和 REMD 参数.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FidelityLevel:
    """单个保真度级别."""
    name: str = "default"
    nstep: int = 100000
    nstep_close: int = 50000
    nstru: int = 10
    use_remd: bool = True
    remd_replicas: int = 16
    temperature_range: tuple = (300.0, 600.0)


@dataclass
class SimulationState:
    """模拟状态."""
    round_idx: int = 0
    best_energy: float = float("inf")
    best_coords: Optional[object] = None
    fidelity_level: int = 0
    converged: bool = False
    pair_rate: float = 0.0
    cross_segment_ok_rate: float = 0.0
    hbond_rate: float = 0.0
    clash_count: int = 0


class RuleScheduler:
    """基于规则的多保真度调度器.

    根据当前模拟状态决定下一个保真度级别.
    """

    def __init__(self, levels: Optional[List[FidelityLevel]] = None):
        if levels is None:
            levels = [
                FidelityLevel("coarse", nstep=50000, nstep_close=10000, nstru=5),
                FidelityLevel("medium", nstep=100000, nstep_close=50000, nstru=10),
                FidelityLevel("fine", nstep=200000, nstep_close=100000, nstru=20),
            ]
        self.levels = levels
        self.history = []  # 调度历史记录

    def next_level(self, state: SimulationState) -> FidelityLevel:
        """根据状态决定下一个保真度级别."""
        if state.converged:
            return self.levels[-1]
        idx = min(state.fidelity_level, len(self.levels) - 1)
        return self.levels[idx]

    def should_advance(self, state: SimulationState, energy_delta: float) -> bool:
        """是否应该进入下一个保真度级别."""
        if abs(energy_delta) < 100.0:  # 能量变化很小
            return True
        return False
