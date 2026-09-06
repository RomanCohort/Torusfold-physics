"""
torch_gpu_refine.py — GPU-accelerated refinement (replacement for openmm_gpu_refiner).

Replaces OpenMM CPU REMD with BatchedREMD2D (torch GPU),
and OpenMM physical relaxation with relax_structure (torch GPU).

The interface is fully compatible with openmm_gpu_refine, so it can be swapped in directly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def torch_gpu_refine(
    input_pdb: str,
    output_dir: str,
    sequence: str,
    secondary_structure: str,
    name: str = "refine",
    nstep: int = 100000,
    nstep_close: int = 1000,
    nstru: int = 3,
    platform_name: str = "auto",
    use_remd: bool = True,
    remd_n_replicas: int = 12,
    remd_n_steps: int = 100000,
    verbose: bool = True,
    skip_cg_to_allatom: bool = False,
    use_physical_relax: bool = True,
    bpp_matrix: Optional[np.ndarray] = None,
    bpp_weight: float = 0.5,
    use_multistage_remd: bool = True,
    use_potential_refine: bool = True,
    skip_minimal_fold: bool = False,
    use_trirnasp: bool = False,
    use_trirnasp_force: bool = False,
    trirnasp_energy_dir: str = None,
    trirnasp_scale: float = 0.002,  # unified default: 0.002 (Tri/CG ~ 11%)
    trirnasp_update_freq: int = 10,
    use_adaptive_tri_weight: bool = False,
    use_staged_tri: bool = False,
    tri_stage_config: Optional[dict] = None,
    lambdas: Optional[Tuple[float, ...]] = None,  # added: custom lambda values
) -> Tuple[str, float, dict]:
    """torch GPU-accelerated refinement (interface compatible with openmm_gpu_refine).

    Replacement path:
    1. Read PDB -> extract P coordinates + far/long-range pairs
    2. Pre-fold: backbone-bond relaxation (400K->300K, 6 stages x 2000 steps)
    3. BatchedREMD2D (torch GPU) multi-round REMD (8 rounds x 5000 steps)
    4. relax_structure (torch GPU) physical relaxation
    5. CG -> all-atom (reuses isrnacirc_wrapper)
    6. Write the refined PDB

    Returns:
        (output_pdb_path, final_energy, diag_dict)
    """
    from .openmm_gpu_refiner import (
        _read_p_coords, _dotbracket_to_pairs,
        _sanitize_p_coords, _generate_compact_coords,
        BOND_P_NEXT,
    )
    from .torch_cgsim import BatchedREMD2D, cg_energy_forces
    from .physical_relaxation import relax_structure

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Read P coordinates
    p_coords = _read_p_coords(input_pdb)
    L = len(p_coords)
    if verbose:
        print(f"  [Torch GPU] sequence length: {L} nt")

    if L < 3:
        raise ValueError(f"sequence too short ({L} nt)")

    p_coords = _sanitize_p_coords(p_coords)
    pairs = _dotbracket_to_pairs(secondary_structure)

    # far pairs from bpp
    if bpp_matrix is not None and bpp_matrix.shape[0] == L:
        from .openmm_gpu_refiner import discover_far_pairs_from_bpp
        far = discover_far_pairs_from_bpp(
            bpp_matrix, sequence, min_gap=24, bpp_threshold=0.01,
            top_k=50, existing_pairs=pairs)
        if far:
            pairs = pairs + far
            if verbose:
                print(f"  [bpp] found {len(far)} far pairs, {len(pairs)} total")

    # unit check
    if L > 1:
        avg_pp = float(np.mean(np.linalg.norm(
            p_coords[1:] - p_coords[:-1], axis=1)[:min(L - 1, 500)]))
        if avg_pp < 1.5:
            p_coords *= 10.0
            avg_pp *= 10.0
        use_compact = (not np.isfinite(avg_pp)) or avg_pp > 20.0 or avg_pp < 1.0
        if use_compact:
            if verbose:
                print(f"  [Torch GPU] abnormal P-P bond length (avg={avg_pp:.2f}A), generating compact coordinates")
            p_coords = _generate_compact_coords(L, pairs)
    else:
        use_compact = False

    # 2. Pre-fold: backbone-bond relaxation (400K->300K, 6 stages x 2000 steps)
    final_e = float("inf")
    final_p_coords = p_coords.copy()

    if use_potential_refine and not skip_minimal_fold:
        if verbose:
            print(f"  [Torch GPU] pre-fold: 400K->300K, 6 stages x 2000 steps")
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            L = len(final_p_coords)
            # P-only -> 3-bead (P, C4', N): C4' offset +0.34nm along the backbone, N offset -0.15nm
            pos_p = torch.tensor(final_p_coords, dtype=torch.float64, device=dev) / 10.0
            # backbone direction vector
            diffs = torch.zeros_like(pos_p)
            diffs[1:] = pos_p[1:] - pos_p[:-1]
            diffs[0] = diffs[1] if L > 1 else torch.zeros(3, device=dev)
            bb_dir = diffs / (diffs.norm(dim=1, keepdim=True).clamp(min=1e-6))
            pos_c4 = pos_p + bb_dir * 0.034   # C4' +0.34nm along the backbone
            pos_n = pos_p + bb_dir * (-0.015)  # N -0.15nm along the backbone
            # interleaved order: (P0, C4'0, N0, P1, C4'1, N1, ...)
            pos_3bead = torch.stack([pos_p, pos_c4, pos_n], dim=1).reshape(1, 3 * L, 3)

            pairs_t = torch.tensor([(i, j) for i, j, _ in pairs], dtype=torch.long, device=dev) if pairs else torch.zeros(0, 2, dtype=torch.long, device=dev)
            pw = torch.tensor([w for _, _, w in pairs], dtype=torch.float64, device=dev) if pairs else torch.zeros(0, dtype=torch.float64, device=dev)

            # 6-stage temperature annealing
            fold_temps = [400.0, 350.0, 325.0, 310.0, 300.0, 300.0]
            for stage, T in enumerate(fold_temps):
                pos_3bead.requires_grad_(True)
                opt = torch.optim.Adam([pos_3bead], lr=1e-3)
                for step in range(2000):
                    opt.zero_grad()
                    e, _ = cg_energy_forces(pos_3bead, pairs_t, pw)
                    if not torch.isfinite(e).all():
                        raise RuntimeError(f"pre-fold stage {stage + 1} energy not finite")
                    e.sum().backward()
                    if pos_3bead.grad is None or not torch.isfinite(pos_3bead.grad).all():
                        raise RuntimeError(f"pre-fold stage {stage + 1} gradient not finite")
                    opt.step()
                    pos_3bead.data.clamp_(-1.0, 10.0)
                    if not torch.isfinite(pos_3bead).all():
                        raise RuntimeError(f"pre-fold stage {stage + 1} coordinates not finite")
                pos_3bead = pos_3bead.detach()

            # extract P coordinates (take bead 0 of every 3)
            with torch.no_grad():
                final_e_fold, _ = cg_energy_forces(pos_3bead, pairs_t, pw)
                final_e_fold = final_e_fold.item()
            final_p_coords = pos_3bead[:, 0::3, :].squeeze(0).cpu().numpy() * 10.0  # nm->A
            if not np.isfinite(final_e_fold) or not np.all(np.isfinite(final_p_coords)):
                raise RuntimeError("pre-fold result has non-finite energy or coordinates")

            if verbose:
                print(f"  [Torch GPU] pre-fold complete: E={final_e_fold:.0f}")
            final_e = final_e_fold
        except Exception as e:
            if verbose:
                print(f"  [Torch GPU] pre-fold failed: {e}")

    # 3. Multi-round REMD (8 rounds x 5000 steps, Langevin restarted each round)
    if use_remd:
        n_rounds = 8 if use_multistage_remd else 1
        steps_per_round = 5000 if use_multistage_remd else remd_n_steps

        if verbose:
            print(f"  [Torch GPU] BatchedREMD2D: {remd_n_replicas} replicas x {n_rounds} rounds x {steps_per_round} steps")

        # 8T x 8lambda: 300-1000K, lambda=1.0->0.65 (64 replicas)
        # Bug 8 fix: use custom lambdas or defaults
        if lambdas is None:
            lambdas = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
        n_lam = len(lambdas)
        n_t = remd_n_replicas // n_lam
        _lambdas = lambdas

        all_diags = []
        backup_p_coords = final_p_coords.copy()  # NaN recovery backup
        prev_3bead_state = None  # carry the full 3-bead state across rounds
        backup_3bead_state = None  # Bug 7 fix: 3-bead state backup
        _global_step_acc = 0  # cumulative global step count across rounds
        prev_vel = None  # Bug 3 fix: carry velocities across rounds

        # Bug 10 fix: create one BatchedREMD2D instance and call run() multiple times
        remd = BatchedREMD2D(
            n_t=n_t,
            t_lo=300.0, t_hi=1000.0,
            lambdas=_lambdas,
            exchange_interval=500,
            use_trirnasp=use_trirnasp,
            use_trirnasp_force=use_trirnasp_force,
            trirnasp_scale=trirnasp_scale,
            sequence=sequence,
            trirnasp_energy_dir=trirnasp_energy_dir,
            force_refresh_freq=500,
            use_adaptive_tri_weight=use_adaptive_tri_weight,
            use_staged_tri=use_staged_tri,
            tri_stage_config=tri_stage_config,
        )

        for round_idx in range(n_rounds):
            try:
                # Bug 10 fix: pass the cross-round state instead of re-creating the instance
                best_coords, best_e, diag = remd.run(
                    final_p_coords, pairs, n_steps=steps_per_round,
                    verbose=verbose, initial_pos_3bead=prev_3bead_state,
                    initial_global_step=_global_step_acc,
                    initial_velocities=prev_vel)
                all_diags.append(diag)

                # NaN recovery: detect NaN -> roll back to the previous step
                if not np.isfinite(best_e) or not np.all(np.isfinite(best_coords)):
                    if verbose:
                        print(f"  [Torch GPU] round {round_idx+1} NaN, rolling back to previous step")
                    final_p_coords = backup_p_coords.copy()
                    # Bug 7 fix: restore the 3-bead state from backup (do not set None)
                    prev_3bead_state = backup_3bead_state
                    prev_vel = None  # Bug 3 fix: velocities are invalid too
                    continue

                backup_p_coords = final_p_coords.copy()  # save backup
                # unconditional update: each round feeds the previous round's output as input
                final_p_coords = best_coords
                # keep the full 3-bead state (P + C4'/N) to avoid re-initializing next round
                prev_3bead_state = diag.get("best_pos_3bead")
                # Bug 7 fix: save the 3-bead state backup
                backup_3bead_state = prev_3bead_state
                # Bug 3 fix: keep the velocity state
                prev_vel = diag.get("velocities")
                # accumulate the global step count (staged strategy across rounds)
                _global_step_acc += steps_per_round
                if not np.isfinite(final_p_coords).all():
                    if verbose:
                        print(f"  [Torch GPU] round {round_idx+1} output not finite, rolling back to previous step")
                    final_p_coords = backup_p_coords.copy()
                    continue
                if best_e < final_e:
                    final_e = best_e

                if verbose:
                    print(f"  [Torch GPU] round {round_idx+1}/{n_rounds}: E={best_e:.0f}")
            except Exception as e:
                import traceback
                if verbose:
                    print(f"  [Torch GPU] round {round_idx+1} failed: {e}")
                    traceback.print_exc()
                final_p_coords = backup_p_coords.copy()
                prev_3bead_state = None
                continue

        if verbose and all_diags:
            print(f"  [Torch GPU] REMD complete: final E={final_e:.0f}, "
                  f"T-acc={np.mean(all_diags[-1]['acceptance_T']):.0%}")

    # 3. Physical relaxation (torch GPU)
    if use_physical_relax and L >= 10:
        try:
            # relaxation: pass the full pair list (includes pseudoknot candidates and BPP weighting)
            relaxed, relax_m = relax_structure(
                final_p_coords, sequence,
                far_pairs=None,
                n_steps=5000,
                pairs_all=pairs)
            # post-relaxation energy must be reasonable
            if np.all(np.isfinite(relaxed)):
                _avg = float(np.mean(np.linalg.norm(
                    relaxed[1:] - relaxed[:-1], axis=1)[:100]))
                if _avg > 3.0 and _avg < 12.0:
                    final_p_coords = relaxed
                    if verbose:
                        print(f"  [Torch GPU] physical relaxation: "
                              f"clash {relax_m['initial']['clash_count']}"
                              f"->{relax_m['final']['clash_count']}, "
                              f"bond_viol {relax_m['final']['bond_violations']}")
        except Exception as e:
            if verbose:
                print(f"  [Torch GPU] physical relaxation failed: {e}")

    # 4. Write CG PDB + CG -> all-atom
    cg_pdb = str(out_path / f"{name}_cg.pdb")
    _write_pdb_simple(cg_pdb, final_p_coords, sequence)

    if skip_cg_to_allatom:
        aa_pdb = cg_pdb
    else:
        aa_pdb = str(out_path / f"{name}.pdb")
        try:
            from .isrnacirc_wrapper import cg_to_allatom
            cg_to_allatom(cg_pdb, aa_pdb, sequence)
            if verbose:
                print(f"  [Torch GPU] CG -> all-atom: {aa_pdb}")
        except Exception as e:
            if verbose:
                print(f"  [Torch GPU] CG -> all-atom failed: {e}, using CG PDB")
            aa_pdb = cg_pdb

    elapsed = time.time() - t0
    if verbose:
        print(f"  [Torch GPU] done: {elapsed:.1f}s, E={final_e:.0f}")

    diag = {
        "success": True,
        "energy": final_e,
        "elapsed": elapsed,
        "remd_rounds": n_rounds if use_remd else 0,
        "use_remd": use_remd,
        "use_multistage_remd": use_multistage_remd,
    }
    return aa_pdb, final_e, diag


def _write_pdb_simple(pdb_path: str, coords_A: np.ndarray, sequence: str):
    """Write a simple CG PDB (P-only, compatible with CG_to_allatom.exe).

    PDB ATOM format: cols 31-38 x, 39-46 y, 47-54 z (8.3f each, whitespace-separated).
    """
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    L = len(coords_A)
    with open(pdb_path, "w", newline="\n") as f:
        # CG_to_allatom.exe requires the first line to be an ATOM record (REMARK lines are not accepted)
        for i in range(L):
            x, y, z = coords_A[i]
            resname = base_map.get(sequence[i].upper(), "ADE")
            # PDB ATOM standard: cols 13-16 atom name " P  ", cols 17-19 resname
            f.write(f"ATOM  {i+1:5d}  P   {resname} A{i+1:4d}"
                    f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
        f.write("END\n")
