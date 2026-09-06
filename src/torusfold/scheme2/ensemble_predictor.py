# -*- coding: utf-8 -*-
"""ensemble_predictor.py - three-predictor ensemble: RhoFold+ + trRNA2 + RNAbpFlow

Plan C: weighted fusion + distance calibration

Strategy:
  1. RhoFold+    -> coordinates (high geometric accuracy)
  2. trRNA2      -> distance matrix (strong for non-canonical pairs)
  3. RNAbpFlow   -> coordinates + distance (generative, ensemble)
  4. Distance consensus: average of trRNA2 + RNAbpFlow
  5. Distance calibration: calibrate the RhoFold coordinates with the consensus distances
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class EnsembleResult:
    """Ensemble result from three predictors."""
    coords: np.ndarray          # (L, 3) calibrated P coordinates
    confidence: float           # ensemble confidence
    dist_consensus: np.ndarray  # (L, L) consensus distance matrix
    per_predictor: dict         # per-predictor independent results
    bond_quality: float         # bond-length quality (closer to 5.9 is better)
    calibration_rmse: float     # RMSE change before/after calibration


def _bond_score(coords: np.ndarray) -> float:
    """Assess backbone bond-length quality (0-1). Ideal value 5.9 A."""
    if coords is None or len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    # Gaussian score: the closer the mean is to 5.9 A and the smaller the std, the higher the score
    mean_dev = abs(np.mean(dists) - 5.9) / 5.9
    std_dev = np.std(dists) / 5.9
    return max(0.0, 1.0 - mean_dev - std_dev)


def _extract_p_coords(allatom_coords: np.ndarray, n_res: int) -> np.ndarray:
    """Extract P atoms (atom 0 per residue) from all-atom coordinates."""
    if allatom_coords is None:
        return None
    if allatom_coords.ndim == 3:
        # Format: (n_atoms_per_res, L, 3) - trRNA2 output
        return allatom_coords[0, :n_res].astype(np.float32)
    elif allatom_coords.ndim == 2 and allatom_coords.shape[1] == 3:
        # Already (L, 3)
        return allatom_coords[:n_res].astype(np.float32)
    return None


# ── Region-type dynamic weights ──
# stem: dense WC pairing, RhoFold+ performs best
# bsj:  sparse pairing, needs non-canonical-pair information
# loop: many non-canonical pairs, trRNA2/RNAbpFlow have the advantage
REGION_WEIGHTS = {
    "stem": {"rhofold": 0.50, "trrna2": 0.20, "rnabpflow": 0.30},
    "bsj":  {"rhofold": 1.00, "trrna2": 0.00, "rnabpflow": 0.10},  # coordinates from RhoFold only, but RNAbpFlow supplies the distance restraint
    "loop": {"rhofold": 0.30, "trrna2": 0.40, "rnabpflow": 0.30},
}

# auto-mode density thresholds
_DENSITY_HIGH = 0.3   # > 0.3 -> stem
_DENSITY_LOW = 0.1    # < 0.1 -> bsj
# otherwise -> loop


def _estimate_pairing_density(
    sequence: str,
    secondary_structure: str = None,
) -> float:
    """Estimate the pairing density of a sequence region (pairs / length).

    Uses the secondary structure when available; otherwise falls back to the
    ViennaRNA bpp.

    Args:
        sequence: RNA sequence
        secondary_structure: dot-bracket, optional

    Returns:
        pairing density [0, 1]
    """
    L = len(sequence)
    if L == 0:
        return 0.0

    n_pairs = 0
    if secondary_structure:
        n_pairs = sum(1 for ch in secondary_structure if ch in "()") // 2
    else:
        # ViennaRNA bpp fallback
        try:
            import RNA
            ss, mfe = RNA.fold(sequence)
            n_pairs = sum(1 for ch in ss if ch in "()") // 2
        except Exception:
            return 0.0

    return n_pairs / max(L, 1)


def _classify_region(
    pairing_density: float,
    bsj_aware: bool = False,
) -> str:
    """Classify a region type from the pairing density and the BSJ flag.

    Args:
        pairing_density: pairing density [0, 1]
        bsj_aware: whether the chunk is near the BSJ

    Returns:
        "stem", "bsj", or "loop"
    """
    if bsj_aware:
        return "bsj"
    if pairing_density > _DENSITY_HIGH:
        return "stem"
    if pairing_density < _DENSITY_LOW:
        return "bsj"
    return "loop"


def _select_predictors(region_type, seq_len, gc_content):
    """Select predictors by region type and sequence features.

    Returns: dict of {predictor_name: enabled}
    """
    if region_type == "stem":
        # High-GC stem: prefer RhoFold+ (high pLDDT, good geometry)
        return {"rhofold": True, "trrna2": False, "rnabpflow": False}

    elif region_type == "bsj":
        # BSJ region: prefer the RNAbpFlow distance matrix + a RhoFold coordinate backbone
        return {"rhofold": True, "trrna2": False, "rnabpflow": True}

    elif region_type == "loop":
        # loop: trRNA2 + RNAbpFlow (many non-canonical pairs)
        if seq_len <= 200:
            # Short sequence: run all (fast)
            return {"rhofold": True, "trrna2": True, "rnabpflow": True}
        else:
            # Long sequence: skip trRNA2 (too slow on CPU), use RhoFold + RNAbpFlow
            return {"rhofold": True, "trrna2": False, "rnabpflow": True}

    else:
        # auto/unknown: run all
        return {"rhofold": True, "trrna2": True, "rnabpflow": True}


def ensemble_predict(
    sequence: str,
    secondary_structure: str = None,
    output_dir: str = None,
    verbose: bool = True,
    use_rhofold: bool = True,
    use_trrna2: bool = True,
    use_rnabpflow: bool = True,
    n_rnabpflow_samples: int = 1,
    region_type: str = None,
) -> EnsembleResult:
    """Run the three-predictor ensemble prediction.

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket), optional
        output_dir: output directory
        verbose: print progress
        use_rhofold: whether to use RhoFold+
        use_trrna2: whether to use trRNA2
        use_rnabpflow: whether to use RNAbpFlow
        n_rnabpflow_samples: number of RNAbpFlow samples (ensemble)
        region_type: region type ("stem"/"bsj"/"loop"/"auto"/None).
            None or omitted -> fall back to the original fixed weights
            (confidence*bond_score). "auto" -> decide from the pairing density.

    Returns:
        EnsembleResult
    """
    L = len(sequence)
    t_start = time.time()
    per_pred = {}

    # Predictor selection (by region type)
    if region_type:
        selected = _select_predictors(region_type, L, 0.0)
        use_rhofold = selected.get("rhofold", use_rhofold)
        use_trrna2 = selected.get("trrna2", use_trrna2)
        use_rnabpflow = selected.get("rnabpflow", use_rnabpflow)
        if verbose:
            enabled = [k for k, v in selected.items() if v]
            print(f"  [Ensemble] region={region_type}, selected: {', '.join(enabled)}")

    # ── 1. RhoFold+ ──
    coords_rh = None
    conf_rh = 0.0
    if use_rhofold:
        try:
            if verbose:
                print("  [Ensemble] RhoFold+ ...")
            t0 = time.time()
            from .rhofold_wrapper import rhofold_predict_chunk
            coords_rh, conf_rh = rhofold_predict_chunk(
                sequence, verbose=False)
            t_rh = time.time() - t0
            per_pred["rhofold"] = {
                "coords": coords_rh, "confidence": conf_rh, "time": t_rh,
            }
            if verbose:
                print(f"    RhoFold+: {coords_rh.shape}, conf={conf_rh:.3f}, {t_rh:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    RhoFold+ failed: {e}")
            # Fallback: try replacing with trRNA2 P coordinates
            if use_trrna2:
                try:
                    if verbose:
                        print("  [Fallback] RhoFold+ failed, trying trRNA2 P coordinates instead ...")
                    from .trrna2_wrapper import trrna2_predict_chunk
                    tr_fb = trrna2_predict_chunk(sequence, name="fallback", num_recycles=3)
                    if tr_fb.coords is not None:
                        coords_rh = _extract_p_coords(tr_fb.coords, L)
                        if coords_rh is not None and len(coords_rh) == L:
                            conf_rh = tr_fb.confidence if tr_fb.confidence else 0.0
                            per_pred["rhofold"] = {
                                "coords": coords_rh, "confidence": conf_rh,
                                "time": 0.0, "fallback": "trrna2",
                            }
                            if verbose:
                                print(f"    [Fallback] using trRNA2 P coordinates in place of RhoFold+, conf={conf_rh:.3f}")
                        else:
                            if verbose:
                                print("    [Fallback] trRNA2 P coordinate length mismatch, giving up")
                    else:
                        if verbose:
                            print("    [Fallback] trRNA2 produced no coordinates, giving up")
                except Exception as e2:
                    if verbose:
                        print(f"    [Fallback] trRNA2 substitute also failed: {e2}")

    # ── 2. trRNA2 ──
    dist_tr = None
    coords_tr_p = None
    if use_trrna2:
        try:
            if verbose:
                print("  [Ensemble] trRNA2 ...")
            t0 = time.time()
            from .trrna2_wrapper import trrna2_predict_chunk
            tr = trrna2_predict_chunk(sequence, name="ensemble", num_recycles=3)
            t_tr = time.time() - t0
            if tr.coords is not None:
                coords_tr_p = _extract_p_coords(tr.coords, L)
            dist_tr = tr.dist
            per_pred["trrna2"] = {
                "coords": coords_tr_p, "dist": dist_tr,
                "confidence": tr.confidence, "time": t_tr,
            }
            if verbose:
                shape_str = str(coords_tr_p.shape) if coords_tr_p is not None else "None"
                dist_str = str(dist_tr.shape) if dist_tr is not None else "None"
                print(f"    trRNA2: P={shape_str}, dist={dist_str}, {t_tr:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    trRNA2 failed: {e}")

    # ── 3. RNAbpFlow ──
    coords_rna = None
    dist_rna = None
    if use_rnabpflow:
        try:
            if verbose:
                print("  [Ensemble] RNAbpFlow ...")
            t0 = time.time()
            coords_rna, dist_rna = _run_rnabpflow(sequence, output_dir, verbose)
            t_rna = time.time() - t0
            per_pred["rnabpflow"] = {
                "coords": coords_rna, "dist": dist_rna, "time": t_rna,
            }
            if verbose:
                shape_str = str(coords_rna.shape) if coords_rna is not None else "None"
                dist_str = str(dist_rna.shape) if dist_rna is not None else "None"
                print(f"    RNAbpFlow: coords={shape_str}, dist={dist_str}, {t_rna:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    RNAbpFlow failed: {e}")

    # ── 4. Distance consensus ──
    dist_consensus = None
    if dist_tr is not None and dist_rna is not None:
        # Average the two (lengths may differ; take the intersection)
        L_min = min(dist_tr.shape[0], dist_rna.shape[0])
        dist_consensus = (dist_tr[:L_min, :L_min] + dist_rna[:L_min, :L_min]) / 2.0
        if verbose:
            rmse = np.sqrt(np.mean((dist_tr[:L_min, :L_min] - dist_rna[:L_min, :L_min]) ** 2))
            print(f"    distance consensus: {dist_consensus.shape}, trRNA2<->RNAbp RMSE={rmse:.2f}A")
    elif dist_tr is not None:
        dist_consensus = dist_tr
    elif dist_rna is not None:
        dist_consensus = dist_rna

    # ── 5. Region-type decision ──
    effective_region = region_type
    if region_type == "auto":
        density = _estimate_pairing_density(sequence, secondary_structure)
        effective_region = _classify_region(density, bsj_aware=False)
        if verbose:
            print(f"    region classification: auto -> {effective_region} (density={density:.3f})")
    elif region_type is not None and region_type not in REGION_WEIGHTS:
        if verbose:
            print(f"    [WARN] unknown region_type='{region_type}', falling back to fixed weights")
        effective_region = None

    # ── 6. Weighted coordinate fusion ──
    if coords_rh is not None:
        final_coords = coords_rh.copy()
        all_coords = [coords_rh]

        if coords_tr_p is not None and len(coords_tr_p) == L:
            all_coords.append(coords_tr_p)

        if coords_rna is not None and len(coords_rna) == L:
            all_coords.append(coords_rna)

        if len(all_coords) > 1:
            if effective_region == "bsj":
                # BSJ: use the RhoFold+ coordinate backbone directly, skip the weighted average
                # RNAbpFlow distance matrix is already stored in dist_consensus for downstream weak restraints
                if verbose:
                    print(f"    coordinate fusion [bsj]: using the RhoFold+ coordinate backbone directly"
                          + (", RNAbpFlow distance matrix passed on" if dist_rna is not None else ""))
            elif effective_region is not None and effective_region in REGION_WEIGHTS:
                # Dynamic-weight mode: use the predefined weights for the region type,
                # then re-normalize over the available predictors (missing ones redistribute)
                rw = REGION_WEIGHTS[effective_region]
                raw_w = []
                avail_keys = []
                if coords_rh is not None:
                    raw_w.append(rw["rhofold"])
                    avail_keys.append("rhofold")
                if coords_tr_p is not None and len(coords_tr_p) == L:
                    raw_w.append(rw["trrna2"])
                    avail_keys.append("trrna2")
                if coords_rna is not None and len(coords_rna) == L:
                    raw_w.append(rw["rnabpflow"])
                    avail_keys.append("rnabpflow")
                weights = np.array(raw_w, dtype=np.float64)
                weights = weights / max(weights.sum(), 1e-8)
                if verbose:
                    print(f"    coordinate fusion [{effective_region}]: {len(all_coords)} predictors, "
                          f"weights={weights.round(3)} (sources={avail_keys})")
            else:
                # Original fixed-weight mode (degraded when region_type=None)
                weights = []
                weights.append(conf_rh * _bond_score(coords_rh))
                if coords_tr_p is not None and len(coords_tr_p) == L:
                    w_tr = per_pred.get("trrna2", {}).get("confidence", 0.5) * _bond_score(coords_tr_p)
                    weights.append(w_tr)
                if coords_rna is not None and len(coords_rna) == L:
                    w_rna = 0.7 * _bond_score(coords_rna)
                    weights.append(w_rna)
                weights = np.array(weights, dtype=np.float64)
                weights = weights / max(weights.sum(), 1e-8)
                if verbose:
                    print(f"    coordinate fusion [fixed]: {len(all_coords)} predictors, "
                          f"weights={weights.round(3)}")

            # Weighted average (BSJ already skipped above; final_coords stays coords_rh)
            if effective_region != "bsj":
                final_coords = np.zeros_like(coords_rh)
                for c, w in zip(all_coords, weights):
                    final_coords += c * w
    else:
        # No RhoFold result; use the other predictors
        final_coords = coords_rna if coords_rna is not None else (
            coords_tr_p if coords_tr_p is not None else np.zeros((L, 3)))

    # ── 7. Distance calibration ──
    calibration_rmse = 0.0
    if effective_region == "bsj":
        # BSJ: skip the internal ensemble calibration; distance restraints are handled by
        # the segmented_vfold3d downstream weak restraints
        if verbose:
            print(f"    [BSJ] skipping the internal ensemble distance calibration, passing dist_consensus downstream")
    elif dist_consensus is not None and len(final_coords) >= 2:
        try:
            from .trrna2_calibrator import calibrate_with_distance_matrix
            result = calibrate_with_distance_matrix(
                final_coords, dist_consensus, sequence, n_iterations=200)
            if result.rmse_before > 0:
                calibration_rmse = result.rmse_before - result.rmse_after
                final_coords = result.coords
                if verbose:
                    print(f"    distance calibration: RMSE {result.rmse_before:.2f} -> {result.rmse_after:.2f}A"
                          f" (improved {calibration_rmse:.2f}A)")
        except Exception as e:
            if verbose:
                print(f"    distance calibration skipped: {e}")

    # ── 8. All-failure fallback: geometrically initialized coordinates ──
    if final_coords is None or len(final_coords) == 0:
        if verbose:
            print("  [Fallback] all predictors failed, using geometrically initialized coordinates")
        L = len(sequence)
        coords_list = []
        for i in range(L):
            angle = 2 * np.pi * i / L
            r = L * 5.9 / (2 * np.pi)  # P-P bond 5.9 A, circular
            coords_list.append([r * np.cos(angle), r * np.sin(angle), 0.0])
        final_coords = np.array(coords_list, dtype=np.float32)
        confidence = 0.0
        bond_quality = 0.0

    # ── 9. Final quality assessment ──
    bond_score = _bond_score(final_coords)
    confidence = min(1.0, (conf_rh + bond_score) / 2.0) if coords_rh is not None else bond_score

    total_time = time.time() - t_start
    if verbose:
        print(f"  [Ensemble] done: {total_time:.1f}s, "
              f"bond={bond_score:.3f}, conf={confidence:.3f}")

    return EnsembleResult(
        coords=final_coords,
        confidence=confidence,
        dist_consensus=dist_consensus,
        per_predictor=per_pred,
        bond_quality=bond_score,
        calibration_rmse=calibration_rmse,
    )


def _run_rnabpflow(
    sequence: str, output_dir: str, verbose: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run RNAbpFlow inference (subprocess)."""
    import subprocess
    import tempfile
    import pickle

    checkpoint = os.path.join(os.environ.get("RNABPFLOW_ROOT", ""), "checkpoint", "RNA3DB.ckpt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"RNAbpFlow checkpoint not found: {checkpoint}")

    # Create the RNAbpFlow input directory structure (FASTA + map.pkl + npy under Inputs/seq/)
    tmp_dir = os.path.join(tempfile.gettempdir(), "rnabpflow_input")
    inp_dir = os.path.join(tmp_dir, "Inputs", "seq")
    os.makedirs(inp_dir, exist_ok=True)

    # list.txt (must live under Inputs/)
    with open(os.path.join(tmp_dir, "Inputs", "list.txt"), "w") as f:
        f.write("seq 1\n")

    # FASTA (must live at Inputs/seq/seq.fasta)
    with open(os.path.join(inp_dir, "seq.fasta"), "w") as f:
        f.write(f">seq\n{sequence}\n")

    # Write map.pkl (needed by the dataset)
    L = len(sequence)
    aa_map = {"A": 0, "U": 1, "G": 2, "C": 3}
    onehot = np.zeros((L, 5), dtype=np.float32)
    for i, ch in enumerate(sequence):
        onehot[i, aa_map.get(ch.upper(), 4)] = 1.0
    mapfeat = {
        "onehot": onehot,
        "native_chis": np.zeros((L, 5), dtype=np.float32),
        "dist_to_ft": np.ones((L, 1), dtype=np.float32),
        "map": np.zeros((L, L), dtype=np.float32),
    }
    with open(os.path.join(inp_dir, "map.pkl"), "wb") as f:
        pickle.dump(mapfeat, f)

    # Write the .npy maps
    for name in ["map1.npy", "map2.npy", "map3.npy"]:
        np.save(os.path.join(inp_dir, name), np.zeros((L, L), dtype=np.float32))

    # Invoke inference_rocm.py
    script = os.path.join(os.environ.get("RNABPFLOW_ROOT", ""), "inference_rocm.py")
    out_dir = os.path.join(output_dir or tempfile.gettempdir(), "rnabpflow_out")

    result = subprocess.run(
        [os.environ.get("RNABPFLOW_PYTHON", "python"),
         script, "--name", "seq", "--checkpoint", checkpoint, "--device", "cuda"],
        capture_output=True, text=True, timeout=None,
        cwd=tmp_dir,
    )

    # Read the output PDB
    pdb_path = os.path.join(tmp_dir, "Predictions", "seq", "Sample_0.pdb")
    if not os.path.exists(pdb_path):
        # Try other paths
        for d in [tmp_dir, out_dir]:
            p = os.path.join(d, "Predictions", "seq", "Sample_0.pdb")
            if os.path.exists(p):
                pdb_path = p
                break

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"RNAbpFlow output not found: {pdb_path}")

    # Parse PDB -> P coordinates
    coords = _parse_pdb_p_coords(pdb_path, L)
    # Compute the distance matrix
    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).astype(np.float32)

    return coords, dist


def _parse_pdb_p_coords(pdb_path: str, expected_L: int) -> np.ndarray:
    """Extract P-atom coordinates from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P  " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    coords = np.array(coords, dtype=np.float32)
    if len(coords) >= expected_L:
        return coords[:expected_L]
    # If the P-atom count is still off, fall back to taking one atom every ~21 atoms
    if len(coords) > expected_L:
        step = len(coords) // expected_L
        return coords[::step][:expected_L].astype(np.float32)
    return coords
