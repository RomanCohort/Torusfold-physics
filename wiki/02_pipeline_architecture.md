---
title: "Pipeline Architecture"
lead: "Progressive refinement from sequence to all-atom structure"
---

## Pipeline Overview

The isRNAcircLong pipeline predicts circular RNA 3D structures through progressive refinement. Each level increases structural fidelity while building on the output of the previous level. Automatic checkpoint/resume at every level ensures resilience for long-running jobs (2000+ nt sequences may require hours of computation).

**Level numbering** (overview uses 6 main levels; this document details sub-stages):

| Main Level | Sub-stages | Description |
| :--- | :--- | :--- |
| Level 0 | 0 | ViennaRNA + complementarity scan |
| Level 1 | 1, 1.5 | Segmented 3D prediction + CG relaxation |
| Level 2 | 2 | RL-guided CG refinement (3-bead REMD) |
| Level 3 | 2.3, 2.5 | 5-bead CG refinement + all-atom reconstruction |
| Level 4 | 3, 3.5, 4 | RL fine-tuning + Metadynamics + REST2 |
| Level 5 | 5 | AMBER RNA.OL3 all-atom refinement |

![Figure 2: Six-Level Pipeline](../images/fig2-six-level-pipeline.png)
*Figure 2: Six-level pipeline architecture with data flow and key algorithms at each stage.*

---

## Level 0: ViennaRNA Coarse Screening

**Input**: RNA sequence + dot-bracket secondary structure
**Output**: Near-range pairs, far-end pairs, stem blocks

Level 0 extracts pairing information from the circular sequence using ViennaRNA's `RNAfold --circ` mode, then augments it with our custom scanning algorithms:

1. **ViennaRNA circ fold**: Computes base-pair probabilities (bpp) for the circular sequence
2. **Complementarity scan** (`pair_graph.py`): k-mer indexed reverse-complement matching discovers pairs missed by ViennaRNA, reducing complexity from O(L²) to O(L×4^W)
3. **Jaccard shared-partner inference**: For each residue pair (i,j), computes J(i,j) = |P(i) ∩ P(j)| / |P(i) ∪ P(j)| where P(i) = {k | bpp(i,k) > threshold}. High Jaccard → shared partners → spatial proximity
4. **BFS topological distance** (`pair_graph.py`): Builds a paired graph with backbone edges (including BSJ closure) + pairing edges + scan edges, then computes shortest-path distance via BFS. Pairs with high topological distance (>50) are flagged as "far-end pairs" for later stages

**Key outputs**:
- `pairs`: All detected pairs (near-range + scan-supplemented)
- `far_pairs`: Topologically distant pairs requiring long-range folding
- `stem_blocks`: Helical stem regions extracted from pairing patterns

---

## Level 1: Segmented 3D Prediction + Confidence-Weighted Assembly

**Input**: Sequence, secondary structure, pairs from Level 0
**Output**: Initial 3D coordinates (P-only, ~30–40 Å RMSD from final)

Level 1 solves the RhoFold+ collapse problem on long sequences by splitting the circular sequence into manageable chunks:

1. **Sequence segmentation** (`segmented_vfold3d.py`): Non-uniform segmentation at secondary structure boundaries (stem/loop junctions), each chunk ≤200 nt with 20-nt overlap regions
2. **Per-chunk 3D prediction**: Each chunk is predicted independently using RhoFold+ (or isRNAcirc Type=0 as fallback)
3. **Adaptive MSA fusion** (`_resolve_chunk_msa`): Three-tier MSA resolution per chunk:
   - Priority 1: Chunk自带MSA（来自用户提供的 `msa_blocks`）
   - Priority 2: Rfam cmsearch 真 MSA（通过 `rfam_cm` 搜索）
   - Priority 3: 已知家族 MSA 复用（通过 `rfam_dir` 查找）
   - Priority 4: 伪 MSA 兜底（16 序列协同变异）
4. **Confidence-weighted assembly**: Chunks are assembled via confidence-weighted averaging in overlap regions. Higher-confidence chunks receive greater weight, eliminating stitching artifacts without requiring explicit rotational alignment.

**Pseudo MSA generation** (`_build_pseudo_msa_for_chunk`): When no real MSA is available, the original sequence is replicated 16 times. For each replica, paired positions undergo coordinated co-variation at 60% probability, replacing base pairs with alternative complementary pairs (A-U, G-C, G-U) while preserving complementarity. Unpaired positions remain unchanged. This encodes ViennaRNA structural information into the MSA, preventing RhoFold+ single-sequence collapse.

---

## Level 1.5: Coarse-Grained Relaxation

**Input**: RhoFold+-predicted coordinates from Level 1
**Output**: Smoothed CG coordinates

A brief CG relaxation step smooths artifacts from the segmented prediction:

- **3-bead system** (`openmm_gpu_refiner.py`): P/C4'/N representation with softened pairing (pair_scale=0.5) and BSJ (bsj_k_scale=0.3) forces
- **Minimization + short MD**: 1000 iterations energy minimization + 2000 steps (4 ps) MD at 300K
- **Purpose**: Eliminates steric clashes and bond length violations from the initial prediction without over-folding

---

## Level 2: Segmented CG→All-Atom + RL-Scheduled REMD

**Input**: Smoothed CG coordinates from Level 1.5
**Output**: Refined all-atom structure (iterative)

Level 2 is the core refinement stage, combining coarse-grained-to-all-atom conversion with reinforcement learning–guided iterative relaxation.

### RL-MCTS Agent (`RelaxationRL`)

The RL agent uses Monte Carlo Tree Search with a GNN policy network to determine optimal relaxation parameters:

- **State**: Pair distances + energy + clash count + convergence metrics
- **Action**: `pair_weights` (N_far_pairs,). Distance ratio between MCTS-optimal and current conformations.
- **Action**: `md_nstep` (scalar). Number of MD steps (large deviation leads to more steps).
- **Reward**: energy_delta + pair_rate_delta − clash_penalty
- **Policy**: GNN-based `PolicyNetwork` trained via PPO with replay buffer

### Iterative REMD Refinement

The refinement runs in multiple rounds (default: 10, configurable via `n_relax_rounds`):

1. **RL decides parameters**: Agent outputs `pair_weights` and `md_nstep` for this round
2. **Progressive far-end pair injection**: Round 0 injects 30% of far-end pairs, incrementally increasing to 100% by the final round (prevents structural collapse when initial quality is poor)
3. **OpenMM GPU refinement** (`openmm_gpu_refine`): Runs with IsRNAcirc force field, incorporating:
   - Backbone bonds (k=31000 kJ/mol/nm²)
   - Pairing bonds (weighted by RL pair_weights)
   - Stacking, angle, clash, BSJ, and bpp soft constraints
   - 8-replica REMD with temperature exchange (default; configurable via `n_rest2_replicas`)
4. **Convergence monitoring**: 4 metrics tracked per round:
   - `pair_rate`: Fraction of pairs within target distance
   - `cross_segment_ok`: Fraction of cross-segment pairs satisfied
   - `rmsd_change`: RMSD change between consecutive rounds
   - `clash_count`: Number of steric clashes

### Multifidelity Scheduler (`RuleScheduler`)

The scheduler defines three simulation fidelity levels. Currently, Level 2 uses RL-decided or fixed parameters directly; the scheduler is instantiated for history tracking but its `decide()` method is not called in the pipeline loop.

| Fidelity Level | MD Steps | Use Case |
| :--- | :--- | :--- |
| CG_FAST | 500 (~1 ps) | Quick exploration, early rounds |
| CG_MEDIUM | 5000 (~10 ps) | Medium accuracy, convergence phase |
| CG_REST2 | 50,000 (~100 ps) | Temperature REMD enhanced sampling |

---

## Level 2.3: 5-Bead CG Refinement

**Input**: Best CG coordinates from Level 2
**Output**: Refined CG coordinates with improved stacking/H-bond geometry

The 5-bead representation (P/S/B1/B2/B3) provides more accurate description of RNA geometry than the 3-bead model:

- **P**: Phosphate backbone
- **S**: Sugar ring (C3' carbon)
- **B1**: Base (Watson-Crick face). Major groove.
- **B2**: Base (Hoogsteen face). Minor groove.
- **B3**: Glycosidic nitrogen

**Force terms** (8 active blocks in `fivebead_folding.py`):
1. Backbone bond P–P + BSJ closure (k=31000)
2. Intra-residue bonds: P–S, S–B3, S–B1, S–B2
3. Backbone angle P–P–P (A-form 150°)
4. Backbone dihedral P–P–P–P (A-form 33° twist)
5. **S-S stacking** (LJ potential). Primary stacking interaction.
6. **B1-B1 stacking** (LJ potential). Auxiliary stacking (major groove).
7. **WC pairing B1-B1** (direction-dependent 12-10 H-bond potential)
8. Clash repulsion (soft-core, d_min=2.0 Å)

The 5-bead refinement runs 5000 annealing steps at 300K on CPU (5× particle count vs 3-bead).

---

## Level 2.5: CG→All-Atom Conversion

**Input**: Best CG coordinates
**Output**: Full all-atom PDB

Final conversion from coarse-grained to all-atom representation using `cg_to_allatom()` from the isRNAcirc wrapper. This produces the complete all-atom structure with sugar-phosphate backbone and base atoms.

---

## Level 3: RL Fine-Tuning (Continuous Action Space)

**Input**: Best coordinates from Level 2 + far-end pairs
**Output**: Optimized pair weights via PPO

Level 3 applies reinforcement learning with continuous action space to fine-tune far-end pair interactions:

- **Algorithm**: Proximal Policy Optimization (PPO) with `rl_n_simulations` epochs (default: 50)
- **Action space**: Continuous pair weight adjustments for far-end pairs
- **Optimization target**: Maximize pair satisfaction while minimizing energy and clashes
- **Policy network**: GNN-based, loaded from pre-trained checkpoint (`rl_policy_b0.pth` or `rl_policy_bootstrap.pth`)

---

## Level 3.5: Metadynamics Enhanced Sampling

**Input**: Best coordinates from Level 3
**Output**: Metadynamics-refined coordinates

Metadynamics actively crosses free energy barriers by depositing Gaussian hills along collective variables (CVs):

**Parameters** (`MetaDynamicsSampler`):
| Parameter | Value | Description |
| :--- | :--- | :--- |
| hill_height | 1.0 kJ/mol | Height of deposited Gaussian hills |
| hill_sigma | 0.1 nm | Width of hills in CV space |
| hill_freq | 100 | Deposit one hill every 100 MD steps |
| max_hills | 5000 | Maximum number of hills |
| well_tempered | True | Well-tempered metadynamics (Barducci et al. 2008) |
| bias_factor | 5.0 | Well-tempered bias factor γ |
| total_steps | 50,000 (configurable) | Total MD steps |

**Collective Variables**:
1. **BSJ closure distance**: Distance between first and last P atoms
2. **Pair contact fraction**: Fraction of predicted pairs within target distance
3. **Radius of gyration**: Overall compactness of the structure

---

## Level 4: Temperature Replica Exchange Enhanced Sampling

**Input**: Best coordinates from Level 3.5
**Output**: Enhanced-sampled coordinates

Multi-replica temperature exchange provides enhanced conformational sampling by running multiple copies of the system at different temperatures and exchanging configurations:

- **Replicas**: `n_rest2_replicas` (default: 8)
- **Temperature ladder**: Linearly spaced from 1.0 to 4.5 (scaled units)
- **MD steps**: `rest2_nsteps` (default: 100,000)
- **Platform**: Auto-detected (GPU preferred, CPU fallback)
- **Exchange frequency**: Configurable replica exchange attempts

Higher-temperature replicas can cross energy barriers that trap the system at room temperature, while exchanges propagate favorable conformations back to lower temperatures.

---

## Checkpoint/Resume System

The pipeline implements automatic state saving after each level via JSON + numpy checkpoint files:

```
output_dir/
├── _checkpoint.json              # Pipeline state + inline pair data
├── ckpt_coords_vfold.npy         # Level 1 output coordinates
├── ckpt_best_coords.npy          # Best coordinates (Level 2+)
├── vfold3d/                      # Level 1 chunk predictions
│   ├── seg_0/ ... seg_12/        # Per-chunk RhoFold+ output
│   └── assembled.pdb             # Assembled initial structure
├── cg2aa/                        # Level 2 CG→all-atom results
│   └── merged_aa.pdb             # Merged all-atom PDB
├── remd_r0/, remd_r1/...         # Level 2 IsRNAcirc REMD rounds
│   ├── remd_r*_IsRNAcirc.pdb     # Refined PDB per round
│   └── (LAMMPS trajectories, logs)
├── final_allatom.pdb             # Level 2.5 output
└── isrnaclong_final.pdb          # Final output
```

**Atomic writes**: Checkpoint files use temporary-file-then-rename pattern to prevent corruption from mid-write crashes. Numpy arrays are stored as separate `.npy` files referenced from the JSON manifest.

**Resume logic**: On restart, the pipeline reads `_checkpoint.json` and resumes from the last completed level. Each level checks `if ckpt_level >= N:` before running, skipping completed levels automatically.

---

## Data Flow Summary

```
Sequence + Structure
       │
       ▼
   ┌─────────┐
   │ Level 0 │  ViennaRNA + Complementarity Scan + BFS + Jaccard
   └────┬────┘
        │ pairs, far_pairs, stem_blocks
        ▼
   ┌─────────┐
   │ Level 1 │  Segmented RhoFold+/isRNAcirc + Pseudo MSA + Kabsch
   └────┬────┘
        │ coords_vfold (~30-40 Å)
        ▼
   ┌──────────┐
   │ Level 1.5│  3-bead CG Relaxation (smooth artifacts)
   └────┬─────┘
        │ smoothed coords
        ▼
   ┌─────────┐
   │ Level 2 │  RL-MCTS × Iterative REMD (progressive injection)
   └────┬────┘
        │ best_coords, best_energy
        ▼
   ┌──────────┐
   │ Level 2.3│  5-bead CG Refinement (stacking/H-bond geometry)
   └────┬─────┘
        │ refined coords
        ▼
   ┌──────────┐
   │ Level 2.5│  CG → All-Atom Conversion
   └────┬─────┘
        │ all-atom PDB
        ▼
   ┌─────────┐
   │ Level 3 │  RL Fine-Tuning (PPO, continuous action space)
   └────┬────┘
        │ optimized pair weights
        ▼
   ┌──────────┐
   │ Level 3.5│  Metadynamics (well-tempered, 3 CVs)
   └────┬─────┘
        │ barrier-crossed coords
        ▼
   ┌─────────┐
   │ Level 4 │  Temperature REMD (8-replica exchange)
   └────┬────┘
        │ enhanced-sampled coords
        ▼
   Final 3D Structure (PDB)
```

---

## Key Design Decisions

| Decision | Rationale |
| :--- | :--- |
| Minimal force field first | Complete 3-bead stalls at ~45 Å on 2000-nt chains; minimal (backbone + pairing only) folds to ~21 Å |
| Progressive far-end injection | Round 0 at 30% prevents structural collapse; 100% by final round ensures all contacts are enforced |
| RL-MCTS over fixed parameters | RL learns structure-specific optimal parameters; fixed parameters cannot adapt to different convergence patterns |
| Pseudo MSA fallback | Prevents RhoFold+ single-sequence collapse on non-homologous sequences without real MSA |
| 5-bead before all-atom | 5-bead captures sugar pucker and major/minor groove geometry that 3-bead misses, reducing all-atom refinement burden |
| Checkpoint at every level | 2000+ nt sequences take hours; crash recovery is essential |
