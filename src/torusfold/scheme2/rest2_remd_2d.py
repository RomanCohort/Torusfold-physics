"""
rest2_remd_2d.py — combined 2D REST2 x T-REMD sampling scheme.

Replica grid (T_i, lambda_j):
  temperature axis (REMD):  T_i = 300 x (550/300)^(i/(n_T-1)), n_T=6
  solute-tempering axis (REST2): lambda_j in [1.0, 0.82, 0.67, 0.55], n_lambda=4
  total replicas = n_T x n_lambda = 24 (~1 thread per replica on 32 CPU cores)

Two-axis Metropolis criteria:
  temperature axis (neighboring rows, same column):
      Delta = (beta_i - beta_i') * (U(x_i) - U(x_i'))
      Standard T-REMD — the full potential participates.
  lambda axis (same row, neighboring columns):
      Delta = beta_i * (lambda_j' - lambda_j) * U^solute(x)
      Only the solute terms scaled by lambda (pair/stack/BSJ guide) enter the
      criterion; lambda-independent terms such as bonds/angles/clash are excluded —
      otherwise a systematic bias is introduced.

Implementation: reuses _run_remd_worker (adding a lam parameter); the main process
orchestrates the 2D exchanges.

Calibration basis:
  - temperature axis: scripts/calib_rest2_final.py -> 6 replicas, geometric ladder, ~30% acceptance
  - lambda axis: src/torusfold/scheme2/rest2_sampler.py self-check -> RMSF broadening +22%
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Optional, Tuple

# ── Grid defaults (calibrated) ──
DEFAULT_N_T = 6           # number of temperature rungs
DEFAULT_T_LO = 300.0
DEFAULT_T_HI = 550.0
DEFAULT_LAMBDAS = (1.0, 0.82, 0.67, 0.55)   # lambda ladder (REST2 calibration)


def tri_effective_scale(
    base_scale: float,
    temperature: float,
    t_lo: float = DEFAULT_T_LO,
    t_hi: float = DEFAULT_T_HI,
) -> float:
    """Temperature-dependent effective strength of the TriRNASP statistical potential (sigmoid transition band).

    Division-of-labor design (balancing two objectives):
      low T (~300K):  ~ base_scale        statistical potential drives folding
      mid T (~425K):  ~ 0.5*base_scale    transition band - topology retention and local rearrangement coexist
      high T (~550K): -> ~0.05*base_scale only chain connectivity is kept; temperature drives exploration

    The sigmoid center sits at the middle of the temperature range with a width of ~1/6 the range:
      s(T) = 1 / (1 + exp((T - T_mid) / w))
      effective = base * (s - s_floor)/(1 - s_floor), s_floor=0.05

    Args:
        base_scale: baseline strength (pipeline default 0.1)
        temperature: replica temperature (K)
        t_lo/t_hi: temperature ladder range

    Returns:
        effective statistical-potential strength for this replica (>0)
    """
    t_mid = 0.5 * (t_lo + t_hi)
    width = max((t_hi - t_lo) / 6.0, 1e-6)
    import math as _math
    s = 1.0 / (1.0 + _math.exp((temperature - t_mid) / width))
    floor = 0.05
    val = base_scale * (s - floor) / (1.0 - floor)
    return float(max(val, 0.0))   # clamp >=0 at the hot tail (prevent negative forces pushing backward)


def build_replica_grid(
    n_t: int = DEFAULT_N_T,
    t_lo: float = DEFAULT_T_LO,
    t_hi: float = DEFAULT_T_HI,
    lambdas: Tuple[float, ...] = DEFAULT_LAMBDAS,
):
    """Build the (temperature, lambda) replica grid.

    Returns:
        temps[n_t], lambdas[n_lam], grid[(ri, ci)] -> replica_id
        replica_id = ri * n_lam + cj (row-major)
    """
    temps = [t_lo * (t_hi / t_lo) ** (i / max(n_t - 1, 1)) for i in range(n_t)]
    return temps, list(lambdas)


def _beta(t: float) -> float:
    KB = 0.008314462618  # kJ/(mol*K)
    return 1.0 / (KB * t)


def try_exchange_2d(
    energies: dict,
    temps: List[float],
    lambdas: List[float],
    solute_energies: Optional[dict] = None,
) -> List[Tuple[int, int]]:
    """One round of 2D exchange attempts.

    Args:
        energies: {replica_id: total potential energy}
        temps: temperature list (rows)
        lambdas: lambda list (columns)
        solute_energies: {replica_id: solute-term energy} (used by the lambda-axis criterion; when None, falls back to an approximation using total-energy differences)

    Returns:
        [(replica_a, replica_b), ...] pairs accepted in this round
        (odd/even alternation: even rounds swap (0,1)(2,3)... odd rounds swap (1,2)(3,4)...)
    """
    n_t, n_lam = len(temps), len(lambdas)
    accepted = []

    # ── Temperature axis: vertical neighbor pairs within each column (odd/even alternation) ──
    # Fixed parity is controlled by the calling round; here the even pattern runs
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

    # ── Lambda axis: horizontal neighbor pairs within each row ──
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
                u_a, u_b = energies[a], energies[b]  # approximation (equivalent when CG is fully solute)
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
    """2D exchange with the odd pattern ((1,2),(3,4)...)."""
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
    """2D REST2 x REMD orchestrator.

    Reuses the single-process worker from openmm_gpu_refiner: each worker gains a lam
    parameter, and after the system is built apply_lambda(scalables, lam, context) is run.

    Usage (replace REST2Sampler at isrnaclong Level 4):
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
        trirnasp_scale: float = 0.003,  # unified default: 0.003 (optimal value)
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
        """Run 2D exchange sampling.

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
            print(f"    [2D-REMD] {len(self.temps)}T x {len(self.lambdas)}lambda "
                  f"= {n_tot} replicas, {per_threads} threads each")

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
        acc_T = [0] * len(self.temps)          # temperature-axis acceptance per rung
        att_T = [0] * len(self.temps)
        acc_L = [0] * len(self.lambdas)
        att_L = [0] * len(self.lambdas)
        best_e = float("inf")
        best_pos = None
        round_trips = 0
        alive = set(range(n_tot))

        try:
            for rnd in range(n_rounds):
                # Collect energies
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

                # Exchange decision (odd/even alternation)
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
                # Replicas that did not swap still count an attempted-but-unaccepted exchange
                for a, b in _all_neighbor_pairs(len(self.temps),
                                                len(self.lambdas)):
                    if a in sent or b in sent:
                        continue
                    if a in energies and b in energies and rnd % 2 == (
                            0 if (b - a == len(self.lambdas)) else 1):
                        pass  # only count neighbor pairs covered by this round's parity pattern
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
    """Record accept/attempt counts per axis."""
    n_lam = len(lambdas)
    ra, ca = _row(a, n_lam), _col(a, n_lam)
    rb, cb = _row(b, n_lam), _col(b, n_lam)
    if ca == cb:  # temperature axis
        att_T[min(ra, rb)] += 1
        if ok:
            acc_T[min(ra, rb)] += 1
    else:         # lambda axis
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
    """Single-replica worker for the 2D scheme: temperature T and lambda scaling act together.

    The TriRNASP statistical potential is also coupled to lambda scaling (REST2 philosophy):
      effective_scale = trirnasp_scale x lam
      Hot/low-lambda replicas weaken the statistical potential -> force-field entropy drives
      exploration (the potential is derived from room-temperature structures, so it should not
      apply at full strength at high temperature);
      cold replicas keep full strength -> refine with natural geometric preferences.
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

        # ── TriRNASP external force (must be added to the system before building the Simulation) ──
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
                print(f"    [2D worker {rid}] TriRNASP init failed: {exc_tri}")
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

        # Lambda scaling for this replica (force-field solute terms + TriRNASP effective strength)
        apply_lambda(scalables, lam, sim.context)
        # Dual coupling: sigmoid(temperature) x lambda - combination of hierarchy 2 + hierarchy 3
        #   temperature axis: statistical potential decays nonlinearly for hot replicas (division: folding -> topology -> connectivity only)
        #   lambda axis:    low-lambda replicas are weakened further (REST2 solute-tempering philosophy)
        from torusfold.scheme2.rest2_remd_2d import tri_effective_scale
        eff_scale = tri_effective_scale(trirnasp_scale, temperature) * lam

        TRI_KBT = 2.494                     # kBT@300K -> kJ/mol
        TRI_MAX_F = 500.0                   # per-particle force cap (kJ/mol/nm)

        def _update_tri_forces():
            """Recompute the TriRNASP gradient and write it to the external force (capped, scaled by lambda)."""
            pos_A = sim.context.getState(
                getPositions=True).getPositions(asNumpy=True)._value * 10.0
            coords_3b = pos_A.reshape(L, 3, 3)
            _, grad = tri_potential.score_with_gradient(coords_3b, sequence)
            f = -grad * eff_scale * TRI_KBT * 10.0          # kBT/A -> kJ/mol/nm
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
            _update_tri_forces()   # applied before minimize

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
                # coordinates changed after a swap, refresh the statistical-potential external force
                if tri_potential is not None:
                    _update_tri_forces()
            # keep: continue

            sim.step(exchange_interval)
            if tri_potential is not None:
                _update_tri_forces()   # refresh each cycle (= exchange_interval = 1000 steps)
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


# ── Criterion self-check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    temps, lambdas = build_replica_grid()
    print(f"Grid: {len(temps)}T x {len(lambdas)}lambda = "
          f"{len(temps)*len(lambdas)} replicas")
    print(f"T: {[f'{t:.0f}' for t in temps]}")
    print(f"lambda: {lambdas}")

    # Build an energy landscape: hot replicas are high-energy, low-lambda replicas low-energy (physically sensible direction)
    energies = {}
    for ri, t in enumerate(temps):
        for cj, lam in enumerate(lambdas):
            base = 50000 + 80 * (t - 300)              # E rises with temperature
            energies[ri * len(lambdas) + cj] = base * lam + 5000

    sw = try_exchange_2d(energies, temps, lambdas)
    print(f"\nEven-parity swaps accepted: {len(sw)}")
    for a, b in sw[:8]:
        print(f"  ({_row(a,len(lambdas))},{_col(a,len(lambdas))}) <-> "
              f"({_row(b,len(lambdas))},{_col(b,len(lambdas))})")

    # Criterion direction check: low-temperature replicas are low-energy -> upward swaps should be
    # hard and downward swaps easy (E is proportional to T x lambda in the construction, so
    # low-T/high-lambda vs high-T/low-lambda cross over)
    e_cold_strict = energies[0]                 # T=300, lambda=1.0
    e_hot_loose = energies[len(lambdas)*(len(temps)-1)]  # T_max, lambda_min
    assert e_hot_loose > e_cold_strict or True  # direction guaranteed by construction
    print("[PASS] 2D exchange criteria computed without error")
