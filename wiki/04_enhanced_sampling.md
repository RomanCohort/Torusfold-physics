---
title: "Enhanced Sampling Methods"
lead: "Metadynamics, Temperature REMD, and RL-MCTS for crossing free energy barriers"
---

## Overview

Standard molecular dynamics (MD) simulations at room temperature often become trapped in local energy minima, particularly for large biomolecules like circRNAs where the free energy landscape is rugged with many metastable states. The pipeline employs three complementary enhanced sampling strategies to overcome these barriers:

| Method | Strategy | When Used | Key Advantage |
| :--- | :--- | :--- | :--- |
| **RL-MCTS** | Intelligence-guided parameter search | Level 2, Level 3 | Learns structure-specific optimal parameters |
| **Metadynamics** | Bias potential along collective variables | Level 3.5 | Actively pushes system across barriers |
| **Temperature REMD** | Multi-replica temperature exchange | Level 4 | Explores conformational space via thermal fluctuations |

---

## RL-MCTS: Reinforcement Learning with Monte Carlo Tree Search

**Files**: `isrnaclong.py` → `RelaxationRL`, `rl_optimizer.py` → `PolicyNetwork`, `MCTS`

RL-MCTS replaces fixed heuristic parameters with learned, structure-specific decisions. Instead of running REMD with predetermined settings, the RL agent observes the current structural state and decides the optimal pair weights and MD step count for each refinement round.

*(For detailed architecture, state/action/reward design, and GNN policy network, see [Pipeline Architecture](02_pipeline_architecture.md) §Level 2.)*

### Key Design: Progressive Far-End Pair Injection

A critical design choice: not all far-end pairs are injected at once. Round 0 starts with only 30% of far-end pairs, incrementally increasing to 100%:

```
far_ratio = 0.3 + 0.7 × (round_idx / (n_rounds - 1))
```

**Rationale**: When the initial structure is poor (extended chain), injecting all far-end pairs simultaneously causes the strong pairing forces to collapse the structure into an incorrect fold. Progressive injection allows local structure to form first, then gradually introduces long-range contacts.

---

## Metadynamics

**File**: `metadynamics_sampler.py` → `MetaDynamicsSampler`

Metadynamics deposits Gaussian hills along collective variables (CVs), building a bias potential that fills up energy minima and forces the system to explore new regions of conformational space.

### Algorithm

1. Run MD for `hill_freq` steps (default: 100)
2. Compute current CV values
3. Deposit a Gaussian hill at current CV position
4. Repeat until `n_steps` reached
5. Final energy minimization

### Bias Potential

```
U_bias(CV) = Σ_i h_i × exp(-||CV - CV_i||² / (2σ²))
```

where:
- `h_i` = hill height (may decay in well-tempered mode)
- `CV_i` = CV position of i-th hill
- `σ` = hill width in CV space

### Well-Tempered Mode

In well-tempered metadynamics (Barducci et al. 2008), hill heights decrease with visit frequency:

```
h_i = h₀ / (1 + N_visits(CV_i) / γ)
```

where γ is the bias factor (default: 5.0). This ensures:
- Frequent regions get smaller hills (less pushing)
- Rare regions get larger hills (more exploration)
- Converges to the true free energy surface (up to a scaling factor)

### Collective Variables

| CV | Formula | Physical Meaning |
| :--- | :--- | :--- |
| CV1: BSJ distance | ‖P(0) − P(L-1)‖ / 10 | Closure of the circular backbone |
| CV2: Native contacts | #{(i,j) ∈ pairs : d(i,j) < 1.5 nm} / N_pairs | Fraction of predicted pairs satisfied |
| CV3: Radius of gyration | √(Σᵢ ‖rᵢ − r_cm‖² / L) / 10 | Overall compactness of the structure |

### Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| hill_height | 1.0 kJ/mol | Initial hill height |
| hill_sigma | 0.1 nm | Gaussian width in CV space |
| hill_freq | 100 steps | Hill deposition frequency |
| max_hills | 5000 | Maximum number of hills |
| well_tempered | True | Enable well-tempered mode |
| bias_factor | 5.0 | Well-tempered bias factor γ |
| total_steps | 50,000 | Total MD steps (configurable) |

### Implementation Details

The bias is applied via OpenMM's `CustomExternalForce`, which adds a position-dependent energy term to each P atom. The force (negative gradient of bias) is computed analytically:

```
F_i = -(∂U_bias/∂r_i) = Σ_j h_j × (CV_j - CV_current) / σ² × (∂CV/∂r_i)
```

The chain rule term ∂CV/∂r_i is computed numerically for each CV.

---

## Temperature Replica Exchange MD (T-REMD)

**File**: `rest2_sampler.py` → `REST2Sampler`

Despite the class name `REST2Sampler`, the implementation uses standard Temperature Replica Exchange MD (T-REMD), not REST2 (solute tempering). Multiple copies of the system run at different temperatures, with periodic exchange attempts between neighboring replicas.

### Algorithm

1. Initialize N replicas at different temperatures
2. Each replica runs independent MD for `exchange_interval` steps
3. Attempt exchange between adjacent replicas (i, i+1) using Metropolis criterion:
   ```
   P_accept = min(1, exp[(β_i - β_{i+1}) × (E_i - E_{i+1})])
   ```
4. Repeat until `n_steps` reached
5. Track best conformation across all replicas

### Temperature Ladder

The pipeline passes a linearly-spaced temperature ladder in scaled units:

```python
temperatures = [1.0 + i * 0.5 for i in range(n_replicas)]
# Default 8 replicas: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
```

The `REST2Sampler` class also has a default geometric ladder (300K × 1.10^i) but the pipeline overrides it with the linear schedule.

### Exchange Mechanics

Exchanges are attempted between adjacent temperature replicas every `exchange_interval` steps (default: 100). The Metropolis acceptance criterion ensures detailed balance:

```
Δ = (1/kT_i - 1/kT_{i+1}) × (E_i - E_{i+1})
P_accept = min(1, exp(Δ))
```

High-temperature replicas can cross energy barriers that trap low-temperature replicas. Successful conformations propagate back to low temperatures through exchanges.

### Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| n_replicas | 8 (default) | Number of temperature replicas |
| temperature range | 1.0–4.5 (scaled) | Linear spacing in scaled units |
| exchange_interval | 100 steps | Exchange attempt frequency |
| n_steps | 100,000 (default) | Total MD steps per replica |
| platform | Auto-detected | CUDA > OpenCL > CPU |

### Snapshot Collection

Optional ensemble snapshot collection from low-temperature replicas:

- **temperature_filter**: Only collect from replicas with T < 400K
- **snapshot_interval**: Configurable collection frequency
- **max_snapshots**: Buffer limit (default: 500)

Snapshots provide conformational diversity for downstream analysis.

---

## Method Comparison

| Aspect | RL-MCTS | Metadynamics | Temperature REMD |
| :--- | :--- | :--- | :--- |
| **Strategy** | Learned parameter optimization | Biased exploration | Thermal fluctuation |
| **Adaptive?** | Yes (learns from structure) | Partially (well-tempered decay) | No (fixed temperatures) |
| **CV required?** | No (learns from rewards) | Yes (3 CVs defined) | No |
| **Computational cost** | O(N_simulations × MCTS_depth) | O(n_steps × L²) | O(N_replicas × n_steps × L²) |
| **Best for** | Parameter tuning | Barrier crossing | Conformational exploration |
| **Limitations** | Requires pre-trained policy | CV selection critical | Expensive (N replicas) |

---

## Integration in the Pipeline

The three methods are applied sequentially, each building on the previous:

```
Level 2: RL-MCTS decides pair_weights + md_nstep → OpenMM REMD refinement
    ↓ (structure improved, but may be trapped)
Level 3: RL fine-tunes pair weights via PPO → optimized parameters
    ↓ (parameters optimized, but barriers remain)
Level 3.5: Metadynamics deposits hills along CVs → crosses barriers
    ↓ (barriers crossed, but limited CV space)
Level 4: Temperature REMD explores full conformational space → final sampling
```

**Key insight**: Each method compensates for the others' weaknesses:
- RL-MCTS optimizes parameters but cannot cross barriers
- Metadynamics crosses barriers but depends on CV selection
- Temperature REMD explores broadly but is computationally expensive

---

## Convergence Criteria

The pipeline monitors convergence through four metrics tracked per Level 2 round:

| Metric | Target | Meaning | Status |
| :--- | :--- | :--- | :--- |
| pair_rate | >0.8 | Fraction of predicted pairs within 15 Å | ✅ 97.4% |
| cross_segment_ok | >0.7 | Fraction of cross-segment pairs satisfied | ✅ 83.3% |
| rmsd_change | <1.0 Å | RMSD change between consecutive rounds | ✅ 0.37 Å |
| clash_count | <10 | Number of steric clashes (< 3 Å) | ❌ 14,826 |

These thresholds are **advisory, not enforced** — the pipeline continues beyond 10 rounds regardless. The clash_count target reflects an ideal scenario; in practice, the CG-to-all-atom reconstruction introduces steric overlaps that require further relaxation. The pipeline completes when all rounds finish, using the lowest-energy configuration across rounds as the final output.
