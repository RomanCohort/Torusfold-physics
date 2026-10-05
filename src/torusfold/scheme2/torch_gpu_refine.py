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
from typing import Callable, List, Optional, Tuple

import numpy as np


_CG_TABLE_KW = None

# The environment variable that seeds a run without touching any call site.
SEED_ENV = "TORUSFOLD_SEED"


def _resolve_seed(seed: Optional[int], verbose: bool = True) -> Optional[int]:
    """The seed for a stochastic run — and, when there is none, a line that says so.

    WHAT WAS WRONG. Nothing seeded anything. `BatchedREMD2D` starts velocities at zero
    (`torch_cgsim`, `_safe_zeros`) and the randomness enters through the Langevin noise drawn
    inside the integrator, so every run of the same input took a different path. Measured on
    2OIU, 71 nt, ten draws per arm, same input and same code
    (`scripts/structure_spread.py` over `results/plan_c/ab_2oiu/`):

        within-arm mean pairwise P-trace RMSD   7.561 A (analytic) / 7.898 A (tables)
        worst within-arm pair                   9.709 A
        every one of the 20 structures          8.0-10.5 A from the crystal they started from

    The A/B script's own comment records the cause: "REMD velocities are not seeded".

    WHAT SEEDING DOES NOT DO. It does not make a run accurate, and it does not remove the
    7.6 A spread — a spread across seeds is sampling, and averaging it away is not the same as
    being right. What it removes is the third thing: not being able to say which draw a figure
    came from.

    AND IT IS NOT SUFFICIENT. Seeding fixes the noise stream; it does not make the GPU kernels
    deterministic. Any force kernel that accumulates with atomics can return a different
    float for the same input. So a seeded run has to be CHECKED, not assumed: run the same
    seed twice and compare — `scripts/structure_spread.py` computes exactly that number, and
    it should be 0.000 A. If it is not, the kernels, not the seed, are the remaining problem.

    Precedence: the `seed` argument, then `TORUSFOLD_SEED`, then nothing — and "nothing"
    prints. A silent unseeded run is how this stayed unnoticed.

    A caller that invokes this several times (per round, per candidate) and passes the same
    seed gets the same noise stream every time. That is deterministic and it is also
    correlated; pass `seed + round` when the streams should be independent.
    """
    if seed is None:
        raw = os.environ.get(SEED_ENV)
        if raw:
            try:
                seed = int(raw)
            except ValueError:
                raise SystemExit(f"{SEED_ENV}={raw!r} is not an integer")
    if seed is None and verbose:
        print("  [seed] NOT SEEDED: this run cannot be reproduced. Two runs of the same input "
              "differ by ~7.6 A RMSD on 71 nt (scripts/structure_spread.py). "
              f"Set {SEED_ENV} or pass seed= to fix that.")
    return seed


def _cg_potential_kwargs():
    """The tabulated CG potentials named by TORUSFOLD_CG_TABLES, or {} for the analytic field.

    Unset (the default) keeps the analytic field this pipeline has always run. Set to a table file --
    results/production_tables.npz is the one scripts/build_production_tables.py composes -- and the CG
    energy calls in this module take the fitted tabulated potentials instead, exactly as the
    calibration harness builds them (cg_potentials.build_potential_kwargs). The device is not
    special-cased anywhere: the tables follow the coordinates (force_reference.table_for), so a cuda
    run gets a cuda copy of U.

    Built ONCE per process and cached: the potentials close over the installed table, so rebuilding
    them per call would cost time and invite a mid-run table swap nobody asked for. Set the
    environment variable before the first refinement call.
    """
    global _CG_TABLE_KW
    if _CG_TABLE_KW is None:
        path = os.environ.get("TORUSFOLD_CG_TABLES", "").strip()
        spec = os.environ.get("TORUSFOLD_BASE_STACK", "").strip()
        kw = {}
        if path:
            import sys                      # this module does not import sys otherwise
            scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            import cg_potentials as _P
            # THE BASE-LEVEL STACKING TERM, opt-in (2026-10-05). Unset keeps the field this pipeline has
            # always run. Set to "eps" or "eps:w_d,w_r,w_t" -- the calibrated setting is
            # TORUSFOLD_BASE_STACK=16.6:0,1,0.19, the fourth calibration in findings Part 22 -- and the CG
            # energy calls take the term as well, built through the same cg_potentials entry point the loop
            # and the retention instrument use, so there is one object and not three.
            base_stack = None
            if spec:
                parts = spec.split(":")
                w = [float(x) for x in parts[1].split(",")] if len(parts) > 1 else [1.0, 1.0, 1.0]
                base_stack = {"eps": float(parts[0]),
                              "form": os.environ.get("TORUSFOLD_BASE_STACK_FORM", "sum"),
                              "w_d": w[0], "w_r": w[1], "w_t": w[2]}
            pots, kw = _P.build_potential_kwargs(path, base_stack=base_stack)
            print("  [torch_gpu_refine] CG tables from " + path + ": "
                  + ", ".join(str(c) for c, _, _ in pots)
                  + (" + base-level stacking eps %.3g kJ/mol (%s, w %s)"
                     % (base_stack["eps"], base_stack["form"],
                        ",".join("%g" % v for v in (w[0], w[1], w[2]))) if base_stack else ""))
        _CG_TABLE_KW = kw
    return _CG_TABLE_KW


def _refine_langevin(beads_A, sequence, pairs, nsteps, temp, seed=None):
    """A short ROOM-TEMPERATURE Langevin trajectory, in nm, returning (final P in A, final beads in A).

    WHY IT IS HERE. torch_gpu_refine's CG stage is a FOLDING protocol: a six-stage 400 -> 300 K pre-fold and
    an REMD ladder whose replicas reach 1000 K, with the returned state chosen by CG energy. A base-level
    stacking term worth ~6 kBT per pair at 300 K is worth ~1.5 at the top of that ladder, so the base frames
    are scrambled by construction -- measured on 2OIU: the folding path returns beads with a base-frame
    cosine of 0.650 against the deposit's 0.901 and produces an unstacked product, while this protocol keeps
    0.895 and produces one that is 58.3 percent stacked (findings Parts 24-26).

    It is the loop's OWN sampler (ibi_core.run_round) with the loop's protocol, not a second integrator: the
    sweep that chose these numbers (1000 steps, 300 K, no pre-relaxation; 1.58 A of trace drift with sd 0.12
    over eight seeds) ran through exactly this call.
    """
    import sys as _sys
    import torch as _torch
    _scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "scripts")
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    import ibi_core as _IC
    from .torch_cgsim import make_intra_constraints as _mk_con

    L = len(sequence)
    pos = _torch.tensor((np.asarray(beads_A, dtype=np.float64) / 10.0).reshape(1, 3 * L, 3),
                        dtype=_torch.float64)
    # The refiner's pair list is (i, j, w) triples or (i, j) pairs; the sampler wants indices only.
    ij = _torch.tensor(np.asarray([(int(p[0]), int(p[1])) for p in pairs], dtype=np.int64).reshape(-1, 2),
                       dtype=_torch.long)
    pw = _torch.ones(len(ij), dtype=_torch.float32)
    # The binning tables are the reference grids, not the production U: run_round needs a table per scored
    # coordinate while the DYNAMICS comes from pot_kw, and mixing them is deliberate -- the reference file
    # carries the full layout (the production file is U/lo/binw only).
    _tab_path = os.environ.get("TORUSFOLD_CG_TABLES", "").strip()
    _ref = os.path.join(os.path.dirname(_tab_path), "refit_smooth5_with_base.npz") if _tab_path else ""
    if not os.path.exists(_ref):
        _ref = os.path.join(os.path.dirname(_scripts), "results", "refit_smooth5_with_base.npz")
    tab = _IC.load_tables(_ref)
    if _tab_path and os.path.exists(_tab_path):
        with np.load(_tab_path) as z:
            for c in ("bb_bond", "angle", "dihedral"):
                if f"{c}__U" in z.files:
                    tab[c] = dict(tab[c], U=np.asarray(z[f"{c}__U"], dtype=float))
    res = _IC.run_round(pos=pos, vel=_torch.zeros_like(pos), ij=ij, pw=pw,
                        temps=_torch.full((1,), float(temp), dtype=_torch.float64), tab=tab,
                        nsteps=int(nsteps), burn=0, stride=25, blocks=4, friction=1.0,
                        force_cap=5000.0, pot_kw=_cg_potential_kwargs(),
                        seed=(seed if seed is not None else 20261005), nrep=1, progress=False,
                        constraints=_mk_con(L), relax=0, collect_positions=True,
                        log=lambda *a, **k: None)
    frames = res.positions.numpy()[:, 0].reshape(-1, 3 * L, 3)
    final_nm = frames[-1]
    # The CG energy of the final state, so the caller's diag carries a real number: this path does not go
    # through the REMD block, which is the only other place final_e is set.
    import torch as _t2
    from . import torch_cgsim as _C
    with _t2.no_grad():
        _e = float(_C.cg_energy_forces(
            _t2.tensor(final_nm.reshape(1, 3 * L, 3), dtype=_t2.float64),
            ij, pw, force_cap=5000.0, **_cg_potential_kwargs())[0].mean())
    return final_nm[0::3] * 10.0, final_nm.reshape(L, 3, 3) * 10.0, _e


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
    on_report: Optional[Callable] = None,
    cg_bead_sink: Optional[list] = None,
    cg_frame_allatom: bool = False,
    bead_source_pdb: Optional[str] = None,
    refine_mode: str = "fold",        # "fold" (the annealing protocol) or "refine" (see below)
    refine_steps: int = 1000,
    refine_temp: float = 300.0,
    seed: Optional[int] = None,
) -> Tuple[str, float, dict]:
    """torch GPU-accelerated refinement (interface compatible with openmm_gpu_refine).

    Replacement path:
    1. Read PDB -> extract P coordinates + far/long-range pairs
    2. Pre-fold: backbone-bond relaxation (400K->300K, 6 stages x 2000 steps)
    3. BatchedREMD2D (torch GPU) multi-round REMD (8 rounds x 5000 steps)
    4. relax_structure (torch GPU) physical relaxation
    5. CG -> all-atom (reuses isrnacirc_wrapper)
    6. Write the refined PDB

    TWO OPT-IN ARGUMENTS, both about the beads the sampler actually moved (2026-10-05):

      cg_bead_sink      a list; on return it holds one dict with the LAST sampled 3-bead state in
                        Angstrom ("beads", (L, 3, 3), P / C4' / N per residue) and the P trace it
                        belongs to ("p"). Until now the pipeline propagated only the P trace: the REMD
                        output is sliced to bead 0 of every 3 (line ~236) and relax_structure takes and
                        returns P, so the base frames the model had just sampled were discarded and any
                        later consumer re-derived them with a heuristic. Handing them out costs nothing
                        and is what makes "reconstruct from the sampled frame" measurable on a real run.
      cg_frame_allatom  do the CG -> all-atom step with reconstruct_all_atom_from_beads on those beads
                        instead of cg_to_allatom on the P trace. Off by default, so the shipped product
                        is byte-identical to before; measurements on the 1EHZ template path (findings
                        Part 13) say the heuristic loses 57 percent of the helical stacking and 80-90
                        percent of the WC contacts even when the trace is the crystal's own.
      bead_source_pdb   read the initial (P, C4', N) beads from this file instead of fabricating them.
                        Needed because a prepared input is P-only while the deposit it came from has the
                        base frames, and an initial state that guesses them costs the base level its
                        meaning: measured, fabricated beads have the right internal geometry and a
                        base-frame cosine of 0.607 against the deposit's 0.901.
      refine_mode       "fold" (default, unchanged) runs the annealing protocol this pipeline has always
                        run; "refine" skips the 400 -> 300 K pre-fold and the REMD ladder and runs
                        refine_steps at refine_temp through the loop's own sampler. The difference is not
                        cosmetic: a base-level term worth ~6 kBT per pair at 300 K is worth ~1.5 at the
                        top of a 1000 K ladder, so the folding path returns scrambled base frames (cos
                        0.650) while the refinement keeps them (cos 0.907). Measured end to end on 2OIU,
                        with TORUSFOLD_BASE_STACK=16.6:0,1,0.19 and the beads read from the deposit:
                        trace 1.62 A from the crystal, beads d 5.345 / rise +3.297 / cos 0.907 against
                        5.306 / +3.186 / 0.901, product 1551 atoms and **50.0 percent of helical steps
                        stacked** against 0.0 percent for the folding path and 25.0 for the P-trace
                        reconstruction (findings Part 27).

    Returns:
        (output_pdb_path, final_energy, diag_dict)
    """
    from .openmm_gpu_refiner import (
        _read_p_coords, _dotbracket_to_pairs,
        _sanitize_p_coords, _generate_compact_coords,
        BOND_P_NEXT,
    )
    from .torch_cgsim import BatchedREMD2D, cg_energy_forces

    _cg_table_kw = _cg_potential_kwargs()
    from .physical_relaxation import relax_structure

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Seed BEFORE anything stochastic runs. See _resolve_seed for the measurement that made
    # this necessary (7.6 A of run-to-run scatter on a 71 nt target).
    _seed = _resolve_seed(seed, verbose)
    if _seed is not None:
        import torch as _torch
        _torch.manual_seed(_seed)
        if verbose:
            print(f"  [seed] torch RNG seeded with {_seed} "
                  f"(from {'the seed argument' if seed is not None else SEED_ENV})")
            print("  [seed] this fixes the noise stream, not the kernels: verify by running "
                  "the same seed twice and comparing with scripts/structure_spread.py "
                  "(expect 0.000 A)")

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

    # 1b. REFINEMENT MODE: no anneal, no temperature ladder -- the short room-temperature trajectory the
    # protocol sweep measured, run on the beads the refinement was given (bead_source_pdb, or fabricated
    # from the P trace when nothing else is available). Sets final_p_coords and prev_3bead_state so that the
    # bead sink, the all-atom step and the diag all describe the refined state rather than the input.
    _beads_used = None
    if refine_mode == "refine":
        from .aform_from_template import real_cg_beads, beads_from_pdb
        _beads_used = beads_from_pdb(bead_source_pdb or input_pdb)
        if _beads_used is None or _beads_used.shape[0] != len(sequence):
            _beads_used = real_cg_beads(np.asarray(final_p_coords, dtype=np.float64), sequence)
            print("  [Torch GPU] refine mode: initial beads FABRICATED from the P trace")
        else:
            print(f"  [Torch GPU] refine mode: initial beads from {bead_source_pdb or input_pdb}")
        _t0 = time.time()
        _p_A, _b_A, _e_A = _refine_langevin(_beads_used, sequence, pairs, refine_steps, refine_temp, seed=seed)
        final_p_coords = _p_A
        final_e = _e_A
        import torch as _t
        # The sink and the all-atom step read prev_3bead_state as (any shape) -> (L, 3, 3) NANOMETRES and
        # multiply by 10, so this is the same object type the REMD path leaves behind.
        prev_3bead_state = _t.tensor((_b_A / 10.0).reshape(3 * len(sequence), 3), dtype=_t.float64)
        prev_3bead_p = _p_A
        print(f"  [Torch GPU] refine mode: {refine_steps} steps at {refine_temp:.0f} K, no anneal, "
              f"no ladder ({time.time() - _t0:.0f} s)")

    # refine_mode="refine" has already produced the trajectory; the folding stages below are skipped so
    # that the anneal (400 -> 300 K) and the REMD ladder (to 1000 K) cannot scramble the base frames the
    # refinement was asked to keep (findings Part 26).
    if use_potential_refine and not skip_minimal_fold and refine_mode != "refine":
        if verbose:
            print(f"  [Torch GPU] pre-fold: 400K->300K, 6 stages x 2000 steps")
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            L = len(final_p_coords)
            # P-only -> 3-bead (P, C4', N). The C4' and N beads come from the 1EHZ
            # template reconstruction: the previous backbone-direction offsets put both
            # on the backbone axis, where they carry no base information.
            # START FROM THE DEPOSIT'S OWN BASE FRAMES WHEN THE FILE HAS THEM. The heuristic puts both
            # non-backbone beads on the backbone axis and guesses the roll, and the base-level term then has
            # to drag them into place over the run -- measured on 2OIU: fabricated beads have the right
            # internal geometry and a base-frame cosine of 0.607 against the deposit's 0.901, and the product
            # built from them is unstacked (findings Part 25).
            from .aform_from_template import real_cg_beads, beads_from_pdb
            # bead_source_pdb exists because a prepared P-only input has no C4'/N to read, while the
            # deposit it came from does -- and _read_p_coords is not chain-aware, so pointing the whole
            # refinement at a full-atom file would feed it every phosphorus atom in the structure (measured:
            # 1527 P for a 71-nt chain).
            _own = beads_from_pdb(bead_source_pdb or input_pdb)
            if _own is not None and _own.shape[0] == len(sequence):
                _beads = _own
                print(f"  [Torch GPU] initial beads from {bead_source_pdb or input_pdb} "
                      f"({_beads.shape[0]} residues)")
            else:
                _beads = real_cg_beads(np.asarray(final_p_coords, dtype=np.float64), sequence)
                print("  [Torch GPU] initial beads FABRICATED from the P trace (no C4'/N available)")
            # interleaved order: (P0, C4'0, N0, P1, C4'1, N1, ...)
            pos_3bead = torch.tensor(
                _beads.reshape(1, 3 * L, 3), dtype=torch.float64, device=dev) / 10.0

            pairs_t = torch.tensor([(i, j) for i, j, _ in pairs], dtype=torch.long, device=dev) if pairs else torch.zeros(0, 2, dtype=torch.long, device=dev)
            pw = torch.tensor([w for _, _, w in pairs], dtype=torch.float64, device=dev) if pairs else torch.zeros(0, dtype=torch.float64, device=dev)

            # Intra-residue rigid distances.
            #
            # cg_energy_forces no longer carries a P-C4' or C4'-N term: those two are rigid
            # constraints now (see the K_BB comment in torch_cgsim.py). A Langevin sampler gets
            # them from batch_langevin_step, but this is an Adam minimiser, and SHAKE has no
            # place inside a gradient step. So the distance is enforced AFTER each step instead,
            # which makes this projected gradient descent.
            #
            # It has to be here at all: without it C4'(i) is held only by K_LINK_CP and C4'(L-1)
            # by nothing, so the folding would run on a field in which two of the three beads of
            # every nucleotide are free to drift. Only the P beads are returned below, so the
            # drift would not show in the output -- it would show as forces on P that came from a
            # geometry no sampler would ever visit.
            #
            # Adam's momentum carries across the projection, so this is not a true projected
            # method: the search direction may point off the constraint manifold and be clipped
            # back. That is acceptable here because the projection moves the coordinates by ~the
            # step size when the constraints are already close to satisfied, which they are after
            # the first step.
            from .torch_cgsim import make_intra_constraints
            con = make_intra_constraints(L)
            pos_3bead.data = con.shake(pos_3bead.data)

            # 6-stage temperature annealing
            fold_temps = [400.0, 350.0, 325.0, 310.0, 300.0, 300.0]
            for stage, T in enumerate(fold_temps):
                pos_3bead.requires_grad_(True)
                opt = torch.optim.Adam([pos_3bead], lr=1e-3)
                for step in range(2000):
                    opt.zero_grad()
                    e, f = cg_energy_forces(pos_3bead, pairs_t, pw, **_cg_table_kw)
                    if not torch.isfinite(e).all():
                        raise RuntimeError(f"pre-fold stage {stage + 1} energy not finite")
                    # Take the gradient from the returned force, not from e.sum().backward().
                    #
                    # cg_energy_forces adds the GB/SA and Manning energies with .detach()
                    # (torch_cgsim.py:1442/1468), so they are constants as far as autograd is
                    # concerned: backward() over e produced a gradient containing every term's
                    # force EXCEPT the solvation one, while e itself still contained the
                    # solvation energy. Adam was therefore descending a field that is not
                    # -dE/dx. total_F carries all terms, and grad = -force.
                    #
                    # Note total_F is the capped force (torch_cgsim.py:1500+), so this is the
                    # same field the dynamics below integrate, which is the point.
                    pos_3bead.grad = (-f).clone()
                    if pos_3bead.grad is None or not torch.isfinite(pos_3bead.grad).all():
                        raise RuntimeError(f"pre-fold stage {stage + 1} gradient not finite")
                    opt.step()
                    pos_3bead.data.clamp_(-1.0, 10.0)
                    # Projected gradient descent: back onto the constraint manifold after the
                    # unconstrained step. See the note above con for why this is not exact and
                    # why it is still the right thing here.
                    pos_3bead.data = con.shake(pos_3bead.data)
                    if not torch.isfinite(pos_3bead).all():
                        raise RuntimeError(f"pre-fold stage {stage + 1} coordinates not finite")
                pos_3bead = pos_3bead.detach()

            # extract P coordinates (take bead 0 of every 3)
            with torch.no_grad():
                final_e_fold, _ = cg_energy_forces(pos_3bead, pairs_t, pw, **_cg_table_kw)
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
    n_rounds = 0 if refine_mode == "refine" else (8 if use_multistage_remd else 1)
    if use_remd and refine_mode != "refine":
        n_rounds = 8 if use_multistage_remd else 1
        steps_per_round = 5000 if use_multistage_remd else remd_n_steps

        if verbose:
            print(f"  [Torch GPU] BatchedREMD2D: {remd_n_replicas} replicas x {n_rounds} rounds x {steps_per_round} steps")

        # 8T x 8lambda: 300-1000K, lambda=1.0->0.65 (64 replicas)
        # Bug 8 fix: use custom lambdas or defaults
        if lambdas is None:
            lambdas = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
        # The replica budget is split into a temperature axis and a lambda axis, and
        # the grid is their product — so both axes have to be at least 1.
        #
        # `n_t = remd_n_replicas // n_lam` gave 0 for any budget below the lambda
        # count, and the failure was not at the division: the grid became 0 x 8 = 0
        # replicas, and the empty tensor surfaced three frames down as
        # "amax(): Expected reduction dim to be specified for input.numel() == 0"
        # from inside SHAKE, during the 500-step relaxation that runs BEFORE the
        # sampling loop. So REMD failed at its first step of every one of its 8
        # rounds, `on_report` was never reached, and a caller waiting for per-step
        # progress frames saw nothing at all — the picture only moved at the fallback
        # timer's 12-second beat, which reads as "updating, just not live" and gives
        # no hint that the sampler never ran.
        #
        # Trimming the lambda axis instead keeps the temperature axis meaningful,
        # which is the one the REMD criterion acts on: two replicas differing in
        # temperature exchange usefully, two differing only in lambda barely do.
        n_lam = max(1, min(len(lambdas), remd_n_replicas))
        n_t = max(1, -(-remd_n_replicas // n_lam))          # ceil, so no budget is lost
        _lambdas = tuple(lambdas)[:n_lam]
        if verbose and n_lam < len(lambdas):
            print(f"  [Torch GPU] {remd_n_replicas} replica budget: using {n_t} "
                  f"temperature(s) x {n_lam} lambda(s) = {n_t * n_lam}; the "
                  f"remaining {len(lambdas) - n_lam} lambda(s) need replicas this "
                  f"budget does not have")

        all_diags = []
        backup_p_coords = final_p_coords.copy()  # NaN recovery backup
        prev_3bead_state = None  # carry the full 3-bead state across rounds
        prev_3bead_p = None      # ... and the P trace that state belongs to (best_coords of the round)
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
            # Re-stamped per round below: run() reports (round, n_rounds) within a
            # single call, and this loop makes several calls, so a caller would
            # otherwise see the count restart at 1 every round.
            on_report=on_report,
        )

        for round_idx in range(n_rounds):
            try:
                # Re-stamp the callback for this round so the caller sees progress
                # against the whole call rather than against one round of it.
                if on_report is not None:
                    _base = round_idx * max(1, steps_per_round // 500)
                    remd.on_report = (lambda r, n, e, c, _b=_base, _n=n_rounds:
                                      on_report(_b + r, _n * n, e, c))
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
                    prev_3bead_p = np.asarray(backup_p_coords, dtype=np.float64)
                    prev_vel = None  # Bug 3 fix: velocities are invalid too
                    continue

                backup_p_coords = final_p_coords.copy()  # save backup
                # unconditional update: each round feeds the previous round's output as input
                final_p_coords = best_coords
                # keep the full 3-bead state (P + C4'/N) to avoid re-initializing next round
                prev_3bead_state = diag.get("best_pos_3bead")
                # best_pos_3bead and best_coords are the SAME replica (torch_cgsim takes both from
                # pos[i_min]), so this pair is a consistent (beads, P trace) snapshot -- which is what
                # carry_beads_along_trace needs at the end, after the relaxation stage has moved P.
                prev_3bead_p = np.asarray(best_coords, dtype=np.float64)
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
                prev_3bead_p = None
                continue

        if verbose and all_diags:
            print(f"  [Torch GPU] REMD complete: final E={final_e:.0f}, "
                  f"T-acc={np.mean(all_diags[-1]['acceptance_T']):.0%}")

    # 3. Physical relaxation (torch GPU)
    if use_physical_relax and L >= 10 and refine_mode != "refine":
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

    # 3b. The last SAMPLED bead frame, carried onto whatever the P trace became.
    #
    # REMD keeps the full 3-bead state (diag["best_pos_3bead"]) and the relaxation stage downstream
    # takes and returns P alone, so the units are carried by the local trace frame with
    # carry_beads_along_trace -- the sampled unit is kept, only the frame it is expressed in is
    # re-read. Both the sink and the bead-frame all-atom path need this, and neither changes anything
    # when it is not asked for.
    beads_A = None
    try:
        if prev_3bead_state is not None and prev_3bead_p is not None:
            _b = prev_3bead_state
            if hasattr(_b, "detach"):
                _b = _b.detach().cpu().numpy()
            _b = np.asarray(_b, dtype=np.float64).reshape(-1, 3, 3) * 10.0      # nm -> A
            _p_ref = np.asarray(prev_3bead_p, dtype=np.float64).reshape(-1, 3)
            _p_end = np.asarray(final_p_coords, dtype=np.float64).reshape(-1, 3)
            if _b.shape[0] == len(_p_ref) == len(_p_end):
                from .aform_from_template import carry_beads_along_trace
                beads_A = carry_beads_along_trace(_p_end, _b, _p_ref)
    except Exception as e:
        beads_A = None
        if verbose:
            print(f"  [Torch GPU] bead frame unavailable: {e}")
    if cg_bead_sink is not None:
        cg_bead_sink.append({"beads": beads_A, "p": np.asarray(final_p_coords, dtype=np.float64),
                             "energy": float(final_e)})

    # 4. Write CG PDB + CG -> all-atom
    cg_pdb = str(out_path / f"{name}_cg.pdb")
    _write_pdb_simple(cg_pdb, final_p_coords, sequence)

    # THE PRECEDENCE, stated because it used to be accidental and silent. Until 2026-10-05 the order was
    # "bead frame if asked for, else CG_to_allatom.exe, else the CG P trace" -- and the last of those was
    # reported only under verbose, so on a machine without the third-party binary (this one: _CG_TO_AA_EXE
    # is empty) the pipeline's "all-atom product" was a P trace and nothing said so. The order is now:
    #
    #   1. skip_cg_to_allatom            -> the CG P trace, said out loud
    #   2. cg_frame_allatom, or no exe   -> the IN-TREE reconstruction on the SAMPLED bead frame
    #   3. the exe exists                -> CG_to_allatom.exe
    #   4. nothing worked                -> the CG P trace, with a warning instead of a silent success
    #
    # and whichever ran is printed unconditionally. The reconstruction is the better product when the exe is
    # absent, not a consolation: measured on the same 2OIU state (trace 1.58 A from the deposit), 58.3 percent
    # of helical steps stacked against 25.0 for the P-trace path (findings Part 25).
    aa_pdb = cg_pdb
    if skip_cg_to_allatom:
        print("  [Torch GPU] skip_cg_to_allatom: the product is the CG P trace, no all-atom structure")
    else:
        from .aform_from_template import reconstruct_all_atom_from_beads, write_allatom_pdb
        aa_pdb = str(out_path / f"{name}.pdb")
        _exe_ok = False
        try:
            from .isrnacirc_wrapper import _CG_TO_AA_EXE
            _exe_ok = bool(os.path.exists(_CG_TO_AA_EXE))
        except Exception:
            _exe_ok = False
        _used = None

        def _bead_frame(tag):
            write_allatom_pdb(reconstruct_all_atom_from_beads(beads_A, sequence), aa_pdb)
            return tag

        if beads_A is not None and (cg_frame_allatom or not _exe_ok):
            try:
                _used = _bead_frame("the SAMPLED bead frame" if cg_frame_allatom
                                    else "the SAMPLED bead frame (CG_to_allatom.exe not installed)")
            except Exception as e:
                print(f"  [Torch GPU] bead-frame all-atom failed: {e}")
        if _used is None and _exe_ok:
            try:
                from .isrnacirc_wrapper import cg_to_allatom
                cg_to_allatom(cg_pdb, aa_pdb, sequence)
                _used = "CG_to_allatom.exe"
            except Exception as e:
                print(f"  [Torch GPU] CG_to_allatom.exe failed: {e}")
        if _used is None and beads_A is not None:
            try:
                _used = _bead_frame("the SAMPLED bead frame (fallback)")
            except Exception as e:
                print(f"  [Torch GPU] bead-frame all-atom failed: {e}")
        if _used is None:
            aa_pdb = cg_pdb
            print("  [Torch GPU] NO all-atom step succeeded: the product is the CG P trace")
        else:
            print(f"  [Torch GPU] CG -> all-atom via {_used}: {aa_pdb}")

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
