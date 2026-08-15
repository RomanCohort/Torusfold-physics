---
title: "Coarse-Grained Force Fields"
lead: "Three representations for progressive structural refinement"
---

## Overview

The pipeline employs three coarse-grained (CG) representations of increasing fidelity, each optimized for a specific stage of refinement. All force fields are implemented in OpenMM and share a common coordinate convention: P-only initial coordinates (Å) are expanded to multi-bead representations as needed.

| Representation | Beads/nt | Used In | Purpose |
| :--- | :--- | :--- | :--- |
| **Minimal (P-only)** | 1 | Level 2 (initial folding) | Drive initial collapse from extended chain |
| **3-bead (P/C4'/N)** | 3 | Level 1.5, Level 2 | CG relaxation and iterative REMD refinement |
| **5-bead (P/S/B1/B2/B3)** | 5 | Level 2.3 | Precise stacking and H-bond geometry |

---

## Minimal Force Field (P-only)

**File**: `openmm_gpu_refiner.py` → `_build_minimal_system_gpu()`

The minimal force field contains only two interaction terms: backbone bonds and pairing bonds. This deliberately stripped-down model solves the gridlock problem. The full 3-bead force field stalls at ~45 A closure distance on 2000-nt chains because stacking, clash, and angle forces interfere with pairing-driven folding. The minimal field folds to ~21 A.

### Force Terms

| # | Term | Formula | Parameters |
| :--- | :--- | :--- | :--- |
| 1 | Backbone bond P[i]–P[i+1] | ½k(r − r₀)² | k = 31,000 kJ/mol/nm², r₀ = 5.9 Å |
| 2 | Pairing bond P[i]–P[j] | ½k·w·(r − r₀)² | k = 40,000 kJ/mol/nm², r₀ = 5.9 Å, w = ViennaRNA bpp weight |

**No forces**: stacking, clash, bond angle, dihedral, BSJ guide. These are all absent by design.

### Eight-Stage Annealing Protocol

The minimal force field uses a custom eight-stage temperature annealing schedule to drive folding:

| Stage | Temperature (K) | Purpose |
| :--- | :--- | :--- |
| 1 | 480 | Extreme high T: free exploration of pairing landscape |
| 2 | 450 | High T: helix formation begins |
| 3 | 420 | Medium-high T: pairing forces pull distant residues together |
| 4 | 390 | Medium T: WC pair convergence |
| 5 | 360 | Medium-low T: clash elimination |
| 6 | 330 | Low T: structural refinement |
| 7 | 310 | Near room temperature: BSJ closure |
| 8 | 300 | Room temperature: final stabilization |

Each stage runs n_anneal // 8 steps, followed by a strict energy minimization (tolerance 5 kJ/mol/nm, 8000 iterations).

### Physical Rationale

The key insight is that at high temperature (480K), thermal fluctuations allow pairing bonds to overcome local energy barriers and find their global minima. As temperature decreases, the structure progressively settles into the correct fold. The absence of stacking and clash forces means pairing bonds dominate the energy landscape, driving the chain toward the correct topology without interference.

---

## 3-Bead Force Field (P/C4'/N)

**File**: `openmm_gpu_refiner.py` → `_build_3bead_system_gpu()`

The 3-bead model represents each nucleotide with three coarse-grained beads: phosphate (P), sugar ring center (C4'), and base nitrogen (N). This is the workhorse force field for iterative REMD refinement in Level 2.

### Bead Definitions

| Bead | Element | Position | Mass (Da) |
| :--- | :--- | :--- | :--- |
| P | Phosphorus | Phosphate group | 110 |
| C4' | Carbon | Sugar ring center | 110 |
| N | Nitrogen | Base center | 110 |

### Force Terms

| # | Term | Formula | Parameters |
| :--- | :--- | :--- | :--- |
| 1 | Backbone bond P[i]–P[i+1] | Harmonic | k = 31,000 kJ/mol/nm², r₀ = 5.9 Å |
| 2 | Intra-residue P–C4' | Harmonic | k = 31,000, r₀ = 3.9 Å |
| 3 | Intra-residue C4'–N | Harmonic | k = 31,000, r₀ = 3.35 Å |
| 4 | Backbone angle P–P–P | Harmonic | k = 200 kJ/mol/rad², θ₀ = 150° (A-form) |
| 5 | Backbone dihedral P–P–P–P | Harmonic | k = 50, θ₀ = 33° (A-form helical twist) |
| 6 | Stacking N[i]–N[i+1] | LJ 12-6 | ε = 300 kJ/mol, σ = 5.05 Å |
| 7 | WC pairing N[i]–N[j] | Harmonic | k = 80,000×w kJ/mol/nm² (code: 800 kJ/mol/Å² × 100), r₀ = 10.0 Å |
| 8 | BSJ closure P[L-1]–P[0] | Harmonic | k = 500, r₀ = 5.9 Å |
| 9 | BSJ guide P[L-1]–P[0] | Attractive | k = 800, long-range → short-range |
| 10 | Clash repulsion | Soft-core | k = 200, d_min = 3.0 Å, cutoff = 12 Å |

### Three-Stage Annealing

The 3-bead force field uses a three-stage annealing protocol with progressive force strengthening:

| Stage | Temperature | Pair Scale | BSJ Scale | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 350K | 0.1 (weak) | 0.3 (weak) | Helix formation, initial pairing |
| 2 | 320K | 1.0 (strong) | 1.0 (medium) | WC pair convergence |
| 3 | 300K | 1.0 (strong) | 5.0 (strong) | BSJ closure |
| 4 | 280K | 1.0 (strong) | 10.0 (very strong) | Final closure refinement |

### BSJ Guide Force

The BSJ guide force is a long-range attractive potential that pulls the first and last P atoms together from arbitrary distances. Standard harmonic potentials have a limited effective range (~20 Å), but initial structures may have BSJ distances of 100+ Å. The guide force bridges this gap:

```
E_guide = -k_guide × max(0, 1 - r/r_catch)²    for r < r_catch
```

where r_catch is a catchment radius that shrinks during annealing (from 50 Å to 10 Å).

### bpp-Weighted Pairing

The pairing force can incorporate ViennaRNA base-pair probabilities:

```
k_pair = K_PAIR × (bpp_w × bpp_ij + (1 - bpp_w) × w) × pair_scale
```

where `bpp_ij` is the ViennaRNA probability, `w` is the hardcoded weight, and `bpp_w` controls the blend (default 0.5).

---

## 5-Bead Force Field (P/S/B1/B2/B3)

**File**: `fivebead_folding.py` → `build_5bead_system()`

The 5-bead model provides higher fidelity than 3-bead by independently representing the sugar ring and two groove faces of the base. This captures stacking geometry and hydrogen bonding directionality that 3-bead misses.

### Bead Definitions

| Bead | Position | Description |
| :--- | :--- | :--- |
| P | Phosphate backbone | Same as 3-bead |
| S | Sugar ring (C3') | Independent sugar descriptor |
| B1 | Base (Watson-Crick face) | Major groove. Primary H-bond site. |
| B2 | Base (Hoogsteen face) | Minor groove |
| B3 | Glycosidic nitrogen | Glycosidic N connection to base |

### Force Terms (9 Active Blocks)

| # | Block | Term | Formula | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `en[0]` | Backbone bond P[i]–P[i+1] | Harmonic | k = 5,000, r₀ = 5.9 Å |
| 2 | `en[1]` | BSJ closure P[L-1]–P[0] | Harmonic | k = 500, r₀ = 5.9 Å |
| 3 | `en[2]` | Intra-residue P–S, S–B3, S–B1, S–B2 | Harmonic | k = 5,000, r₀ = 2.04/1.09/1.15/1.10 Å |
| 4 | `en[3]` | Backbone angle P–P–P | Harmonic | k = 100, θ₀ = 150° |
| 5 | `en[4]` | Backbone dihedral P–P–P–P | Harmonic | k = 500, θ₀ = 33° |
| 6 | `en[5]` | S-S stacking | WCA-like | ε = 1.0, σ = 3.4 Å, k_rep = 500 |
| 7 | `en[6]` | B1-B1 stacking | WCA-like | ε = 0.8, σ = 3.4 Å, k_rep = 500 |
| 8 | `en[7]` | WC pairing B1–B1 | CustomBond | k = 30×w, r₀ = 5.0 Å |
| 9 | `en[8]` | Clash repulsion | Soft-core | k = 1,000, d_min = 2.0 Å |

### Key Differences from 3-Bead

| Feature | 3-bead | 5-bead |
| :--- | :--- | :--- |
| Stacking | N–N LJ (ε=300) | S-S LJ (ε=8) + B1-B1 LJ (ε=5) |
| H-bond | N–N harmonic (r₀=10 Å) | B1-B1 direction-dependent 12-10 (r₀=5 Å) |
| Sugar | Implicit in C4' | Explicit S bead |
| Groove | Implicit | B1 (major) + B2 (minor) independent |
| Clash d_min | 3.0 Å | 2.0 Å (denser packing) |
| Clash k | 200 | 5,000 (75% reduction from 3-bead to prevent explosion) |

### Dual-Channel Stacking

The 5-bead model separates stacking into two channels:

1. **S-S stacking** (primary): Sugar-sugar LJ interaction between adjacent nucleotides. This captures the dominant stacking energy from sugar ring overlap. ε = 8.0 kJ/mol (reduced 60% from the 3-bead value to prevent 5-bead stacking explosion).

2. **B1-B1 stacking** (auxiliary): Base major-groove LJ interaction. This provides additional stacking directionality that S-S alone cannot capture. ε = 5.0 kJ/mol.

The dual-channel design ensures that stacking energy is distributed across two geometric descriptors rather than concentrated in a single interaction, improving the model's ability to distinguish correct from incorrect stacking geometries.

### Direction-Dependent H-Bonding

The WC pairing uses a 12-10 potential (Lennard-Jones-like but with attractive 1/r¹⁰ term):

```
E_pair = k × w × 30 × [5×(r₀/r)¹² − 6×(r₀/r)¹⁰] × step(r_cut − r)
```

This potential:
- Has a minimum at r₀ = 5.0 Å (representing the B1–B1 distance in correct WC pairing)
- Is truncated at r_cut = 20 Å (beyond which pairs are not interacting)
- Uses the 12-10 form instead of 12-6 to create a steeper attractive well, better modeling the directionality of hydrogen bonds

---

## Physical Relaxation Force Field

**File**: `physical_relaxation.py` → `PhysicalRelaxer`

A lightweight post-processing force field applied after diffusion model predictions to fix physical violations (bond length errors, clashes, improper angles).

### Parameters

| Parameter | Value | Unit | Description |
| :--- | :--- | :--- | :--- |
| bond_k | 100.0 | kcal/mol/Å² | P–P bond length restraint |
| angle_k | 10.0 | kcal/mol/rad² | Backbone angle restraint |
| clash_k | 50.0 | kcal/mol/Å² | Steric clash repulsion |
| bsj_k | 50.0 | kcal/mol/Å² | BSJ closure restraint |
| wc_k | 20.0 | kcal/mol/Å² | WC pairing distance restraint |

### Target Values

| Constraint | Target | Tolerance |
| :--- | :--- | :--- |
| P–P bond length | 5.9 Å | ±0.5 Å |
| Backbone angle | 150° (A-form) | ±10° |
| P–P minimum distance | 3.0 Å | Must exceed |
| BSJ closure (first–last P) | 5.9 Å | ±1.0 Å |
| WC pairing (C1'–C1') | 10.5 Å | ±1.5 Å |

---

## Force Field Selection Guide

| Stage | Force Field | Rationale |
| :--- | :--- | :--- |
| Initial folding (Level 2 start) | Minimal (P-only) | Overcomes 45 Å gridlock; only pairing drives folding |
| Iterative REMD (Level 2) | 3-bead | Balanced accuracy/speed for multi-round refinement |
| Geometry refinement (Level 2.3) | 5-bead | Precise stacking/H-bond geometry before all-atom |
| Post-diffusion fixup | Physical relaxation | Lightweight constraint enforcement |

---

## Unit Conventions

All OpenMM simulations use nanometer (nm) and kilojoule/mol (kJ/mol) internally:

| Quantity | CG coordinates (Å) | OpenMM (nm) |
| :--- | :--- | :--- |
| Position | 5.9 Å | 0.59 nm |
| Bond length | 5.9 Å | 0.59 nm |
| Force constant | 31,000 kJ/mol/nm² | Same |
| Energy | kJ/mol | Same |

Conversion: positions are divided by 10.0 when loading into OpenMM, and multiplied by 10.0 when extracting results.
