# TorusFold-Hybrid — Resources & Reproduction

> Judge-facing reference (English). All figures were measured on the team's
> reference machine, September 2026. Figures marked [TBD] are measurements
> still running or not yet collected — they are never estimated.

## 1. Reference hardware

| Item | Value |
|---|---|
| Platform | AMD Ryzen AI MAX 395 (APU, unified memory), Windows |
| GPU mode | GPU-accelerated path of the pipeline (REMD/MetaD/torch CG); peak unified-memory use ≈ **60 GB** |
| CPU mode | CPU-only path (OpenMM CPU platform, multi-replica parallel); peak RAM ≈ **30 GB** |
| Wall time, low config (CPU) | ≈ **7 h** — 2,013 nt end-to-end, 8 replicas × 20,000 steps (measured) |
| Wall time, full config (GPU, default) | ≈ **14 days** (estimated — not measured) on this APU; highly GPU-dependent, ≈ **8× faster on an NVIDIA A100** (≈ 2 days) |

*The 7 h figure is the measured "low configuration" (8 replicas × 20,000 steps),
whose output is shown in the pre-built viewer (Level 4.9, PPR repaired). The
GPU default is a different (full) configuration, so the two wall times are not
directly comparable. The ≈ 14 days GPU estimate has not been measured yet.
Why is the CPU path the reference in practice? On this AMD platform, ROCm
compatibility with the Radeon 8060S integrated GPU is poor (measured
performance far below an RTX 3080), and ROCm bugs required extensive patching
that further slows the GPU path — so all measured reference runs use the CPU
path.*

## 2. Runtime profiles of the headline demo (2,013 nt circRNA)

| Mode | Peak memory | Wall time (low config) | Notes |
|---|---|---|---|
| GPU path (full config, default) | ≈ 60 GB (unified memory) | ≈ 14 days (estimated, not measured); ≈ 2 days on an NVIDIA A100 (est., 8×) | Default is the full configuration; wall time highly GPU-dependent |
| GPU path (low config) | ≈ 60 GB (unified memory) | [TBD — not run] | Low-config timing was only measured on the CPU path |
| CPU path | ≈ 30 GB | ≈ 7 h (8 replicas × 20,000 steps, measured) | Fully reproducible without a discrete GPU |

Scaling rule of thumb: wall time scales with number of parallel replicas and
per-replica steps; see the flags at the top of `run_2013nt.py`.

## 3. Three-level access for judges (no one needs to run the full pipeline)

| Level | What you get | Effort | Where |
|---|---|---|---|
| L0 — View | Interactive 3D of the predicted 2,013 nt structure (42,831 atoms, Level 4.9 PPR repaired) | Open a file in a browser | `docs/circrna_3d_viewer.html` (structure embedded; needs internet for the two CDN scripts) |
| L1 — Download | Full-atom PDB + per-level outputs + quality JSON | 1 click | GitLab release artifact at Wiki Freeze + Zenodo (DOI at freeze) |
| L2 — Force-field check | 2OIU (≈100 nt; only experimentally resolved circRNA): X-ray structure → Level-2 relaxation → RMSD vs crystal | **17 min (CPU), measured** — final RMSD **1.83 Å** | Evidence that the force field does not distort known structures; see wiki Validation page |
| L3 — Full run | End-to-end 2,013 nt all-atom structure | 7 h + 30–60 GB on the machine above | `python run_2013nt.py` (see README + DEPLOY_EXTERNAL.md) |

## 4. External tools (why setup is non-trivial, and what it costs)

TorusFold-Hybrid is an orchestrator: physics core in this repo; the sequence
predictors are external subprocesses configured via environment variables
(see README and DEPLOY_EXTERNAL.md for versions/URLs).

| Tool | Required for | Disk footprint (approx.) |
|---|---|---|
| ViennaRNA ≥ 2.6 | SS folding (Level 0) | small |
| OpenMM ≥ 8.0 | all MD stages | small |
| PyTorch ≥ 2.0 | RL / GPU CG path | ~2–3 GB |
| RhoFold+ / trRosettaRNA2 / RNAbpFlow | Level 1 ensemble | RNAbpFlow checkpoint alone ≈ 1.3 GB |
| isRNAcirc (Windows binaries) | CG→all-atom | small |
| PyRosetta / structRFM / Infernal+RFAM | optional stages | several GB |

First-time environment setup is the dominant cost (hours), not the run itself.
All machine-specific paths are environment variables — no hard-coded paths.

## 5. Known-limitation cross-check (single source of truth)

- Viewer stat panel (Level 4.9 PPR repaired structure): BSJ closure 5.898 Å;
  bond RMSD 0.0082 Å; pair satisfaction **100.0%** — 100% of the residue pairs
  *detected in the 3D structure* within 15 Å that are Watson–Crick complements
  (the high-confidence pairing signal) lie within 12 Å (H-bond distance);
  computed by `pdb_analyzer.compute_pair_satisfaction` on the delivered
  structure. 3dRNAscore 27.18; global RMSD 1.23 Å. **All figures are internal
  self-consistency metrics of the PPR-repaired model — none are comparisons
  against predicted base pairs or experimental ground truth.**
- Experimental ground truth: only one circRNA structure exists (PDB 2OIU).
  2OIU is used as a force-field validity test: starting from the X-ray
  structure, the Level-2 relaxation takes 17 min (CPU) and ends at
  RMSD 1.83 Å vs the crystal — the force field does not distort known
  structures. This is not a from-sequence prediction benchmark, and no
  de-novo accuracy claim is made from it.
- Historical engineering log with measured dead-ends: docs/NOTES.md (dated).

## 6. One-liner for the wiki / software page

"Run anywhere with 30 GB RAM (CPU) or ~60 GB unified memory (GPU); the
2,013 nt demo takes ≈ 7 h at low configuration — or skip the run entirely:
inspect the embedded 3D structure in your browser, download the PDB, and
see the 2OIU force-field check (17 min, RMSD 1.83 Å): known structures are
not distorted by the relaxation stages."
