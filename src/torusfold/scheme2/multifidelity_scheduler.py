"""
multifidelity_scheduler.py — multi-fidelity scheduler (stub).

Controls the CG-to-all-atom refinement level and the REMD parameters.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FidelityLevel:
    """A single fidelity level."""
    name: str = "default"
    nstep: int = 100000
    nstep_close: int = 50000
    nstru: int = 10
    use_remd: bool = True
    remd_replicas: int = 16
    temperature_range: tuple = (300.0, 600.0)


@dataclass
class SimulationState:
    """Simulation state."""
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
    """Rule-based multi-fidelity scheduler.

    Decides the next fidelity level from the current simulation state.
    """

    def __init__(self, levels: Optional[List[FidelityLevel]] = None):
        if levels is None:
            levels = [
                FidelityLevel("coarse", nstep=50000, nstep_close=10000, nstru=5),
                FidelityLevel("medium", nstep=100000, nstep_close=50000, nstru=10),
                FidelityLevel("fine", nstep=200000, nstep_close=100000, nstru=20),
            ]
        self.levels = levels
        self.history = []  # scheduling history

    def next_level(self, state: SimulationState) -> FidelityLevel:
        """Decide the next fidelity level from the state."""
        if state.converged:
            return self.levels[-1]
        idx = min(state.fidelity_level, len(self.levels) - 1)
        return self.levels[idx]

    def should_advance(self, state: SimulationState, energy_delta: float) -> bool:
        """Whether to advance to the next fidelity level."""
        if abs(energy_delta) < 100.0:  # Energy change is very small
            return True
        return False
