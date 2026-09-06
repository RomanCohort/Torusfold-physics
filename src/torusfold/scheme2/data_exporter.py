# -*- coding: utf-8 -*-
"""data_exporter.py — unified output for pipeline intermediate data

After each Level finishes, call export_xxx() to save:
  1. numpy .npy matrices (for heatmaps)
  2. csv tables (for line charts)
  3. json summaries (for dashboards)

All files are written to the output_dir/_plots/ subdirectory.
"""
import json
import csv
import numpy as np
from pathlib import Path


def _ensure_dir(output_dir):
    p = Path(output_dir) / "_plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Level 0: 2D base pairing ──

def export_level0_bpp(bpp_matrix, seq_length, output_dir):
    """Level 0: pairing probability matrix + MFE structure"""
    d = _ensure_dir(output_dir)
    # Matrix
    if bpp_matrix is not None:
        np.save(d / "01_bpp_matrix.npy", np.asarray(bpp_matrix))
    # Save sequence info
    info = {"seq_length": seq_length}
    if bpp_matrix is not None:
        info["bpp_sum"] = float(np.sum(bpp_matrix))
    (d / "00_level0_info.json").write_text(json.dumps(info, indent=2))
    print(f"  [Plot] Level 0 BPP saved: {d}")


def export_level0_ncm(ncm_pairs, seq_length, output_dir):
    """Level 0: NCM non-canonical pairs"""
    d = _ensure_dir(output_dir)
    if ncm_pairs:
        with open(d / "02_ncm_pairs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pos1", "pos2", "type", "probability"])
            for item in ncm_pairs:
                if len(item) == 4:
                    i, j, t, p = item
                    w.writerow([i, j, t, f"{p:.3f}"])
                else:
                    w.writerow(list(item))
    else:
        (d / "02_ncm_pairs.csv").write_text("pos1,pos2,type,probability\n")
    print(f"  [Plot] NCM pairs saved: {d}")


# ── Level 1: chunked prediction ──

def export_level1_chunks(segments, chunk_confidences, chunk_uncertainties, output_dir):
    """Level 1: chunk information + confidence"""
    d = _ensure_dir(output_dir)
    with open(d / "03_chunk_coords.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "start", "end", "length", "confidence",
                     "uncertainty", "region_type"])
        for idx, seg in enumerate(segments):
            conf = chunk_confidences[idx] if idx < len(chunk_confidences) else 0
            unc = chunk_uncertainties[idx] if idx < len(chunk_uncertainties) else 1
            w.writerow([idx, seg["start"], seg["end"], seg["end"] - seg["start"],
                        f"{conf:.3f}", f"{unc:.3f}", seg.get("region_type", "auto")])
    print(f"  [Plot] Chunk info saved: {d}")


def export_level1_weights(region_weights_list, output_dir):
    """Level 1: per-chunk predictor weights"""
    d = _ensure_dir(output_dir)
    with open(d / "04_ensemble_weights.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "rhofold_w", "trrna2_w", "rnabpflow_w"])
        for idx, w_dict in enumerate(region_weights_list):
            w.writerow([idx,
                        f"{w_dict.get('rhofold', 0):.3f}",
                        f"{w_dict.get('trrna2', 0):.3f}",
                        f"{w_dict.get('rnabpflow', 0):.3f}"])
    print(f"  [Plot] Ensemble weights saved: {d}")


# ── Level 1.5: CG relaxation ──

def export_level15_trajectory(energy_trajectory, output_dir):
    """Level 1.5: energy + temperature trajectory (annealing process)"""
    d = _ensure_dir(output_dir)
    if energy_trajectory:
        with open(d / "05_relaxation_trajectory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "temperature", "potential_energy"])
            for row in energy_trajectory:
                w.writerow(row)
    print(f"  [Plot] Relaxation trajectory saved: {d}")


# ── Level 2: RL-REMD ──

def export_level2_remd(remd_history, output_dir):
    """Level 2: REMD convergence data"""
    d = _ensure_dir(output_dir)
    if remd_history:
        with open(d / "06_remd_convergence.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "best_energy", "pair_rate", "clash_count",
                         "rmsd_change", "inject_frac"])
            for row in remd_history:
                w.writerow(row)
    print(f"  [Plot] REMD convergence saved: {d}")


# ── Level 1-2 validation results ──

def export_validation(v1, v15, v2, output_dir):
    """Export the validation results for each Level"""
    d = _ensure_dir(output_dir)
    results = {
        "level1": v1 if v1 else {},
        "level1_5": v15 if v15 else {},
        "level2": v2 if v2 else {},
    }
    (d / "07_validation.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"  [Plot] Validation results saved: {d}")


# ── Final summary ──

def export_final_summary(coords, sequence, output_dir):
    """Final structure statistics"""
    d = _ensure_dir(output_dir)
    L = len(sequence)
    if len(coords) == 0:
        (d / "08_final_summary.json").write_text(
            json.dumps({"length": L, "error": "no coords"}, indent=2))
        return
    diffs = np.diff(coords, axis=0)
    bonds = np.linalg.norm(diffs, axis=1)
    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)

    summary = {
        "length": L,
        "bond_mean": float(np.mean(bonds)),
        "bond_std": float(np.std(bonds)),
        "max_dist": float(np.max(dist)),
        "rog": float(np.sqrt(np.mean(
            np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)))),
    }
    (d / "08_final_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(d / "09_final_coords.npy", coords)
    np.save(d / "10_final_dist.npy", dist)
    print(f"  [Plot] Final summary saved: {d}")
