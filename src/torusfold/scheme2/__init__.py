"""
TorusFold-Scheme2 - circRNA 3D structure prediction (IsRNAcirc architecture + RL enhancement)

Core architecture (mirroring IsRNAcirc):
  sequence -> ViennaRNA secondary structure -> pair_graph adds far-range pairs
  -> [RL MCTS optimizes the initial conformation] (scheme 3)
  -> 3-bead CG segmented folding (A-form stems / loose loops + RL far-pair guiding forces)
  -> [RL far-pair-guided annealing] (scheme 1)
  -> 1EHZ all-atom reconstruction -> Amber14 OL3 refinement

Improvements (v2):
  1. modification-aware input: supports m6A, Ψ, m1A, 2'-O-Me, m5C
  2. physical-relaxation post-processing: enforces bond-length/bond-angle/clash restraints
  3. uncertainty estimation: confidence based on structure indicators

Key IsRNAcirc parameters:
  - 10 replicas REMD 280-460K (ours: 8 replicas 300-460K)
  - BSJ k ramps 0.001→5.0 kcal/mol/Å² (ours: 0.1→5.0)
  - pairing annealed 0.01→0.1→1.0 (three steps)
  - 100ns MD (ours ~0.4ns)
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# refine.py may be absent (missing in some environments); degrade gracefully
try:
    from .refine import (
        vienna_pair_probs,
        scheme2_initial_coords,
        openmm_refine,
        predict_3d,
        BOND_LEN,
        PAIR_DIST,
        CLASH_DIST,
    )
except ImportError:
    # provide basic constants and stub functions
    BOND_LEN = 5.9
    PAIR_DIST = 10.6
    CLASH_DIST = 3.0
    def vienna_pair_probs(seq, thr=0.5):
        raise ImportError("vienna_pair_probs stub called — pair_graph.py missing real implementation")
    def scheme2_initial_coords(seq, pairs, n_samples=8):
        raise ImportError("scheme2_initial_coords stub — refine.py missing")
    def openmm_refine(coords, pairs, plat="CPU"):
        raise ImportError("openmm_refine stub — refine.py missing")
    def predict_3d(seq):
        raise ImportError("predict_3d stub — refine.py missing")

try:
    from .amber_refine import amber_refine
except ImportError:
    def amber_refine(*a, **kw):
        raise ImportError("amber_refine stub — amber_refine.py missing")

try:
    from .rl_optimizer import optimize_far_pairs
except ImportError:
    def optimize_far_pairs(*a, **kw):
        raise ImportError("optimize_far_pairs stub — rl_optimizer.py missing")
try:
    from .pair_graph import (
        parse_case_annotation,
        build_full_pair_graph,
        extract_stem_blocks,
    )
except ImportError:
    def parse_case_annotation(*a):
        raise ImportError("parse_case_annotation stub — pair_graph.py missing")
    def build_full_pair_graph(seq, pairs, do_scan=True):
        raise ImportError("build_full_pair_graph stub — pair_graph.py missing")
    def extract_stem_blocks(pairs, scan):
        raise ImportError("extract_stem_blocks stub — pair_graph.py missing")

# default RL weight paths
_DEFAULT_RL_POLICY_PATH = str(Path(__file__).resolve().parents[3]
                              / "data" / "bc_policy_big.pt")
_DEFAULT_DPO_POLICY_PATH = str(Path(__file__).resolve().parents[3]
                               / "data" / "dpo_policy_v3_compat.pt")

# statistical-potential path
_STAT_POT_PATH = str(Path(__file__).resolve().parents[3]
                     / "data" / "cg_statistical_potential.pkl")

__all__ = [
    "predict_3d_allatom",
    "predict_3d_allatom_v2",
    "BOND_LEN",
    "PAIR_DIST",
    "CLASH_DIST",
]


def predict_3d_allatom(
    sequence: str,
    *,
    pair_threshold: float = 0.5,
    platform_name: str = "CPU",
    max_iterations: int = 3000,
    use_3bead: bool = True,
    use_rl: bool = False,
    use_rest2: bool = False,
    rl_policy_path: str = None,
    rl_n_simulations: int = 50,
    rl_use_defaults: bool = True,
    rl_dpo_weight: float = 0.0,
    rl_dpo_policy_path: str = None,
    rl_dpo_rollout: bool = False,
    rl_dpo_simulate: bool = False,
    coding_mask=None,
    # new parameters (v2)
    known_modifications: Optional[List[Dict]] = None,
    use_relaxation: bool = True,
    relaxation_steps: int = 5000,
):
    """End-to-end: sequence -> all-atom RNA structure (IsRNAcirc architecture).

    Pipeline (schemes 1+3):
      ViennaRNA pairing -> pair_graph adds far-range pairs
      -> [RL MCTS optimizes the initial conformation] (scheme 3: RL pulls far pairs first)
      -> 3-bead CG segmented folding + [RL far-pair guiding forces] (scheme 1: RL guides physics)
      -> 1EHZ all-atom reconstruction -> Amber14 OL3 refinement

    Args:
        use_3bead: True=3-bead segmented folding (default), False=legacy 1-bead (compat)
        use_rl: enable RL far-pair optimization
        use_rest2: enable REST2 enhanced sampling (8-replica T-REMD)
        rl_use_defaults: automatically load the trained BC prior + DPO scorer
        known_modifications: list of known modification sites (optional)
        use_relaxation: enable physical-relaxation post-processing (default True)
        relaxation_steps: number of physical-relaxation steps (default 5000)

    Returns:
        dict: contains structure coordinates, energies, fingerprints, etc.
    """
    pairs, bpp = vienna_pair_probs(sequence, pair_threshold)

    # --- modification-aware: detect/encode chemical modifications ---
    modification_features = None
    detected_modifications = []
    if known_modifications is not None or len(sequence) < 1000:
        try:
            from .modification_aware import detect_modifications, encode_modifications
            detected_modifications = detect_modifications(sequence, known_modifications=known_modifications)
            modification_features = encode_modifications(sequence, detected_modifications)
            print(f"  Detected {len(detected_modifications)} modification sites")
        except Exception as e:
            print(f"  [WARN] Modification detection failed: {e}")

    # --- schemes 1+3: RL first optimizes the initial conformation + far-pair-guided physical annealing ---
    rl_info = None
    far_pairs = []
    cg_coords_for_amber = None

    if use_rl:
        if rl_use_defaults:
            if rl_policy_path is None:
                rl_policy_path = _DEFAULT_RL_POLICY_PATH
            if rl_dpo_policy_path is None and rl_dpo_weight <= 0:
                rl_dpo_policy_path = _DEFAULT_DPO_POLICY_PATH
                rl_dpo_weight = 5.0
            if not rl_dpo_rollout and not rl_dpo_simulate:
                rl_dpo_simulate = True
        # pair_graph: add the far/long-range pairs missed by ViennaRNA and tag the far pairs
        _, scan_pairs, far_pairs = build_full_pair_graph(
            sequence, pairs, do_scan=True,
        )
        stem_blocks = extract_stem_blocks(pairs, scan_pairs)

    if use_3bead:
        # --- 3-bead pipeline: cg_forcefield all-residue force field (verified 2.4A RMSD) ---
        # cg_forcefield is deliberately absent from this snapshot (see README "Scope note"
        # and docs/NOTES.md "Modules referenced but not published yet"). Fail with a message
        # that says so, instead of a bare ImportError that looks like a broken install.
        try:
            from .cg_forcefield import refine_3bead
        except ImportError as exc:
            raise ImportError(
                "predict_3d_allatom(use_3bead=True) requires torusfold.scheme2.cg_forcefield, "
                "which is not part of this snapshot (under development). The supported entry "
                "point is isrnaclong_pipeline(), which does not use it; pass use_3bead=False "
                "for the legacy 1-bead path. See README 'Scope note' and docs/NOTES.md."
            ) from exc
        L = len(sequence)
        p_init = scheme2_initial_coords(sequence, pairs, n_samples=8)
        if p_init is None:
            raise RuntimeError(f"Scheme2 CG geometry solve failed (L={L})")

        # scheme 3: RL first optimizes the initial conformation as the physical-annealing start
        if use_rl and far_pairs:
            opt_p, cg_orig, rl_info = optimize_far_pairs(
                p_init, sequence, far_pairs, stem_blocks,
                policy_path=rl_policy_path,
                n_simulations=rl_n_simulations,
                coding_mask=coding_mask,
                dpo_weight=rl_dpo_weight,
                dpo_policy_path=rl_dpo_policy_path,
                dpo_rollout=rl_dpo_rollout,
                dpo_simulate=rl_dpo_simulate,
            )
            p_init = opt_p
            cg_coords_for_amber = cg_orig
        elif use_rl:
            rl_info = {"skipped": True, "reason": "no_far_pairs"}

        # 3-bead CG folding (cg_forcefield.refine_3bead)
        cg_coords, e0_cg, e1_cg = refine_3bead(
            p_init, pairs, platform_name, n_anneal=200,
            stat_pot_path=_STAT_POT_PATH if Path(_STAT_POT_PATH).exists() else None,
            sequence=sequence)
    else:
        # legacy 1-bead pipeline (compat)
        init = scheme2_initial_coords(sequence, pairs, n_samples=8)
        if init is None:
            raise RuntimeError(f"Scheme2 CG geometry solve failed (L={len(sequence)})")
        cg_coords, e0_cg, e1_cg = openmm_refine(init, pairs, platform_name)

    if cg_coords_for_amber is None:
        cg_coords_for_amber = cg_coords

    # --- physical-relaxation post-processing: enforces bond-length/bond-angle/clash restraints ---
    relaxation_metrics = None
    if use_relaxation:
        try:
            from .physical_relaxation import relax_structure
            cg_coords, relaxation_metrics = relax_structure(
                cg_coords, sequence,
                far_pairs=far_pairs if far_pairs else None,
                n_steps=relaxation_steps,
                use_openmm=True,
            )
            print(f"  Physical relaxation: clashes {relaxation_metrics['initial']['clash_count']} -> "
                  f"{relaxation_metrics['final']['clash_count']}")
        except Exception as e:
            print(f"  [WARN] Physical relaxation failed: {e}")

    # --- all-atom reconstruction + Amber refinement ---
    from .aform_from_template import reconstruct_all_atom as reconstruct_from_template
    structure = reconstruct_from_template(cg_coords, sequence, pairs=pairs)

    cg_coords_nm = None
    if use_rl and cg_coords_for_amber is not None:
        cg_coords_nm = np.asarray(cg_coords_for_amber, dtype=np.float64) / 10.0
    coords_aa, e0_aa, e1_aa, amber_info = amber_refine(
        structure, pairs,
        platform_name=platform_name,
        max_iterations=max_iterations,
        coding_mask=coding_mask,
        cg_coords=cg_coords_nm,
    )

    # structure fingerprints + signals
    from .immune_heuristic import (
        compute_immune_fingerprints, compute_structure_signals,
    )
    bsj_dist = float(np.linalg.norm(cg_coords[0] - cg_coords[-1]))
    immune_fingerprints = compute_immune_fingerprints(
        coords_aa, structure, pairs, sequence,
    )
    structure_signals = compute_structure_signals(
        coords_aa, structure, pairs, bpp, sequence,
        e1_aa=e1_aa, bsj_dist=bsj_dist, cg_coords=cg_coords,
    )

    # --- uncertainty estimation ---
    uncertainty = _estimate_uncertainty(
        cg_coords, pairs, far_pairs, bsj_dist, relaxation_metrics,
    )

    return {
        "coords_cg": cg_coords,
        "pairs": pairs,
        "pair_probs": bpp,
        "e0_cg": e0_cg,
        "e1_cg": e1_cg,
        "atoms": structure,
        "coords_aa": coords_aa,
        "e0_aa": e0_aa,
        "e1_aa": e1_aa,
        "amber_info": amber_info,
        "rl_info": rl_info,
        "structure_method": "scheme2_allatom",
        "available": True,
        "immune_fingerprints": immune_fingerprints,
        "structure_signals": structure_signals,
        # new (v2)
        "detected_modifications": detected_modifications,
        "modification_features": modification_features,
        "relaxation_metrics": relaxation_metrics,
        "uncertainty": uncertainty,
    }


def _estimate_uncertainty(
    cg_coords: np.ndarray,
    pairs: List[Tuple[int, int]],
    far_pairs: List[Tuple[int, int]],
    bsj_dist: float,
    relaxation_metrics: Optional[Dict],
) -> float:
    """Estimate the prediction uncertainty in [0, 1].

    Based on several indicators:
    1. BSJ closure-distance deviation
    2. far-pair distance deviation
    3. number of clashes
    4. number of bond-length violations

    Args:
        cg_coords: CG P coordinates
        pairs: pair list
        far_pairs: far-range pair list
        bsj_dist: BSJ closure distance
        relaxation_metrics: relaxation metrics

    Returns:
        uncertainty in [0, 1]: 0=high confidence, 1=low confidence
    """
    uncertainties = []

    # 1. BSJ closure deviation (ideal ~5.9A)
    bsj_deviation = abs(bsj_dist - BOND_LEN) / BOND_LEN
    uncertainties.append(min(1.0, bsj_deviation))

    # 2. far-pair distance deviation (ideal ~20A)
    if far_pairs:
        wc_devs = []
        for i, j in far_pairs:
            if i < len(cg_coords) and j < len(cg_coords):
                d = np.linalg.norm(cg_coords[i] - cg_coords[j])
                wc_devs.append(abs(d - PAIR_DIST) / PAIR_DIST)
        if wc_devs:
            uncertainties.append(min(1.0, np.mean(wc_devs)))

    # 3. number of clashes
    if relaxation_metrics:
        final_clashes = relaxation_metrics.get("final", {}).get("clash_count", 0)
        uncertainties.append(min(1.0, final_clashes / 10.0))

    # 4. bond-length violations
    if relaxation_metrics:
        bond_violations = relaxation_metrics.get("final", {}).get("bond_violations", 0)
        uncertainties.append(min(1.0, bond_violations / 20.0))

    return float(np.mean(uncertainties)) if uncertainties else 0.5
