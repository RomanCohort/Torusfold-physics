---
title: "Like AlphaFold for Circular RNA"
lead: "An end-to-end pipeline for circular RNA 3D structure prediction from sequence"
author: "Ziyi Yan, Tang Class of Computer Science, Jilin University | FBH Team, IGEM 2026"
---

## Abstract

> Circular RNAs (circRNAs) are covalently closed-loop RNA molecules formed via back splicing, emerging as promising therapeutic agents for triple-negative breast cancer (TNBC). Our FBH team designs circRNA-based drugs, but drug design has been **trapped in two dimensions** — secondary structure predictions tell us which bases pair, but not how the molecule folds in 3D space. Without 3D structure, we cannot understand how IRES elements present themselves to ribosomes, whether miRNA binding sites are accessible on the surface, or how the drug will interact with its cellular targets. Experimental structure determination (cryo-EM, X-ray) takes months and has succeeded for only one circRNA (PDB: 2OIU). We present an end-to-end computational pipeline that predicts circRNA 3D structures directly from sequence. On our 2013-nt TNBC-targeting circRNA, the pipeline produces a physically validated structure in 7 hours on a single CPU, enabling day-level design-predict-characterize cycles with our wet lab collaborators. The predicted structure scores favorably across three independent validation metrics: rsRNASP (-57,171), DFIRE (-319,915), and 3dRNAscore (27.59, exceeding the > 20 threshold for specific folding).

---

## 1. Background

### 1.1 The Problem: Drug Design Trapped in Two Dimensions

Our IGEM FBH team designs circRNA-based therapeutics for triple-negative breast cancer (TNBC). We face a fundamental bottleneck: **every step of circRNA drug design currently operates in 2D**. We can predict which bases pair (secondary structure), but we cannot see how the molecule folds in three-dimensional space. This is like designing a protein drug knowing only its amino acid sequence but not its 3D fold.

Why 3D matters for circRNA drug design:

| Design Question | 2D Can Answer | 3D Can Answer |
| :--- | :--- | :--- |
| Is the IRES element intact? | Partially (base pairing) | Yes (spatial accessibility to ribosome) |
| Are miRNA binding sites exposed? | No | Yes (surface accessibility) |
| How does the drug fold in cells? | No | Yes (tertiary contacts, compactness) |
| Can we rationally mutate for better efficacy? | Limited | Yes (structure-guided mutagenesis) |

Experimental structure determination has succeeded for **only one** circRNA in history (PDB: 2OIU, a 167-nt C. elegans circRNA). For our 2013-nt TNBC-targeting circRNA, cryo-EM or X-ray crystallography would take months with no guarantee of success. We need a computational solution.

### 1.2 Circular RNA: A Unique RNA Molecule

Circular RNA (circRNA) is a class of RNA molecules formed through back splicing, creating a covalently closed loop without a 5' cap or 3' poly(A) tail. Unlike linear RNA, circRNA's 5' and 3' ends are covalently joined at the back-splicing junction (BSJ), imposing a topological constraint with no free ends.

CircRNAs have been implicated in diverse biological functions [1-6]:

| Function | Mechanism |
| :--- | :--- |
| miRNA sponge | Sequester miRNAs, preventing degradation of target mRNAs |
| Protein interaction | Bind RNA-binding proteins (RBPs) and modulate their activity |
| IRES translation | Translate into unique peptides via internal ribosome entry sites |
| Protein localization | Direct proteins to specific cellular positions |

These functions imply that circRNAs should fold into specific three-dimensional structures. However, experimental structure determination remains extremely challenging [7] — only one complete circRNA 3D structure (PDB: 2OIU) has been resolved.

### 1.3 Limitations of Existing Methods

Computational approaches for RNA 3D structure prediction fall into three categories, each with fundamental limitations for circRNAs:

| Category | Representative Tools | Limitation for circRNA |
| :--- | :--- | :--- |
| **Template-based** | ModeRNA, Vfold, 3dRNA | Require known structure templates; too few circRNA structures exist |
| **Ab initio / MD** | IsRNAcirc | Prohibitive cost: ~1,593 CPU hours (~66 days) for 249-nt PIP5K1C [8]; infeasible for long sequences |
| **Deep learning** | trRosettaRNA, RhoFold | Effective for linear RNA, but circRNAs lack experimental training data (crystallization is difficult); end-to-end prediction is essentially impossible |

### 1.3 Core Challenges

CircRNA structure prediction faces three challenges that distinguish it from linear RNA:

**Challenge 1: Circular Topology**

The BSJ creates a covalent ring with no free ends. ViennaRNA offers a circ mode, but it performs poorly on long sequences (>500 nt), struggling to discover long-range contacts spanning the BSJ. Two nucleotides separated by 1000 nt in sequence may be only a few angstroms apart in 3D space via the BSJ.

**Challenge 2: Long-Range Dependencies**

Circular topology brings distant residues into spatial proximity, breaking the fundamental assumption of linear RNA prediction methods (that chains have distinct heads and tails).

**Challenge 3: Data Scarcity**

Complete circRNA structures in the PDB are extremely rare (<10 as of 2026). Template-based methods find no templates, and deep learning lacks training data (proteins have 200,000+ experimental structures; that is why AlphaFold succeeded).

---

## 2. Methodology and Design

We present an end-to-end pipeline implementing five core innovations to address the three challenges above.

![Figure 1: Pipeline Architecture](../images/fig1-pipeline-overview.png)
*Figure 1: Overall pipeline architecture (placeholder; replace with actual figure)*

### Innovation 1: Complementarity Scan + BFS Topological Distance

ViennaRNA's circ mode struggles to discover cross-BSJ contacts on long sequences. We address this through a three-layer mechanism:

**Complementarity scan**: k-mer indexing + reverse-complement matching scans the entire circular sequence, finding pairs missed by ViennaRNA. Complexity reduces from O(L^2) to O(L x 4^W).

**Jaccard shared-partner inference**: Even without direct pairing, spatial contacts can be inferred. For each residue pair (i,j), we compute the pairing partner set P(i) = {k | bpp(i,k) > threshold} and Jaccard similarity J(i,j) = |P(i) intersect P(j)| / |P(i) union P(j)|. High J(i,j) indicates many shared partners, which implies the same structural domain and spatial proximity.

**BFS topological distance**: We construct a paired graph (backbone adjacency edges including the BSJ closure edge + pairing edges + scan supplementary edges) and compute topological distance via BFS. For example, in L=2013, residues (5, 2010) have sequence distance 2005 but graph distance of only 8 (via the BSJ closure edge).

### Innovation 2: MSA + Secondary Structure-Guided Chunking

RhoFold+ generalizes well and requires no license, but fails on sequences >500 nt. We solve this with MSA + secondary structure-guided chunking:

- Non-uniform segmentation at secondary structure boundaries, each chunk <=200 nt, predicted independently
- Real MSA (Rfam cmsearch) prioritized as anchor chunks; gaps filled with pseudo MSA
- Kabsch algorithm aligns overlap regions, eliminating artificial segment boundaries

**Pseudo MSA generation**: When no real MSA is available, we use ViennaRNA pairing information for coordinated base-pair co-variation. We parse dot-bracket to obtain pairs (i,j), replicate the sequence 16 times, and for each replica apply 60% probability co-variation at paired positions (A-U, G-C, G-U complementarity preserved), leaving unpaired positions unchanged. This forces ViennaRNA structural information into RhoFold+, preventing single-sequence collapse.

### Innovation 3: Two-Stage Folding (Minimal to Full)

The full 3-bead force field (clash/stacking/angle/C4'/N) causes mutual interference on 2000-nt chains, with pairing forces unable to drive folding. The structure stalls at ~45 A closure distance.

**Minimal force field**: Only two springs. Backbone bonds (k=31000 kJ/mol/nm^2) and pairing bonds (k=40000 x w, where w is the ViennaRNA bpp weight), with 480K to 300K eight-stage annealing, allowing free folding to ~21 A.

**Full force field**: Stacking + bond angles + clashes + BSJ contacts + bpp soft constraints, refined via REMD.

**5-bead CG refinement**: 5-bead representation (P/S/B1/B2/B3) with independent sugar ring + major/minor groove description, dual-channel stacking (S-S + B1-B1), and direction-dependent H-bonding (B1-B1 12-10 potential).

### Innovation 4: RL-MCTS-Driven Relaxation + Progressive Far-End Pair Injection

Traditional methods use fixed parameters for REMD, lacking insight into which pairs need reinforcement and how many steps to run.

**RL-MCTS**: An RL agent's MCTS search determines per-round parameters. pair_weights represents the distance ratio between MCTS-optimal and current conformations (indicating which pairs need reinforcement). md_nstep represents the number of MD steps (large deviation leads to more steps; small deviation leads to fewer steps as convergence approaches). RL state includes pair distances, energy, clash count, and convergence metrics. RL reward is energy_delta + pair_rate_delta - clash_penalty.

**Progressive far-end pair injection**: Round 0 injects 30% of far-end pairs, incrementally increasing to 100%, preventing structural collapse when initial structure quality is poor.

**Ensemble prediction**: RhoFold+ (weight 0.4) + trRosettaRNA2 (weight 0.4) dual-engine ensemble with uncertainty estimation, improving prediction robustness.

**Confidence-weighted assembly**: After segmented prediction, higher-confidence chunks receive greater weight in overlap regions, eliminating stitching artifacts.

### Innovation 5: Enhanced Sampling (Metadynamics + Temperature REMD)

MD simulations easily become trapped in local minima, particularly at the BSJ closure free energy barrier.

- **Multi-round REMD temperature annealing**: 5 rounds (300-500K to 280-380K), temperature range narrowing per round
- **Temperature REMD**: Multi-replica temperature exchange (Level 4)
- **Metadynamics (well-tempered)**: Gaussian hills (h=1.0 kJ/mol, sigma=0.1 nm) deposited along collective variables (BSJ distance / pair contact fraction / radius of gyration), well-tempered bias factor=5.0, actively crossing free energy barriers

### Engineering Features

- **Checkpoint/Resume**: Automatic state saving after each Level, supporting recovery from any Level. This is important for long sequences (2000+ nt) requiring hours of computation.
- **6-Level Pipeline**: Level 0 (ViennaRNA secondary structure) > Level 1 (segmented 3D prediction) > Level 2 (RL-guided CG refinement) > Level 3 (5-bead + all-atom reconstruction) > Level 4 (Metadynamics + REST2) > Level 5 (AMBER all-atom refinement)

---

## 3. Results

Validated on a 2013-nt real circular RNA:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| BSJ closure distance | 6.07 A | Near-ideal (5.9 A), only 3% deviation |
| P-P bond length RMSD | 0.019 A | Near-perfect backbone bond consistency |
| Pair constraint satisfaction | 97.42% | 341/351 ViennaRNA-predicted pairs within target distance |
| rsRNASP score (RNAdvisor) | -57,171 | All-atom statistical potential (lower = more favorable) |
| DFIRE score (RNAdvisor) | -319,915 | Distance-scaled statistical potential (lower = more favorable) |
| 3dRNAscore | 27.59 | > 20 threshold indicates specific, native-like folding |
| Runtime (CPU-only) | 7 hours | Single CPU, no GPU required |

Three independent scoring functions from RNAdvisor all confirm the predicted structure is physically plausible: rsRNASP (-57,171) and DFIRE (-319,915) indicate favorable energetics, while 3dRNAscore (27.59, > 20 threshold) confirms specific, native-like folding rather than a random coil.

### Practical Impact: Press Enter, Go to Sleep

CircRNA 3D structure prediction has traditionally required GPU clusters or cloud computing. Applying for compute resources involves paperwork, queue times, and approval delays — often slower than the computation itself. Our pipeline changes this: it runs in 7 hours on a single laptop CPU.

> It is 6 PM. You open your gaming laptop, press Enter, and walk away. No cluster application. No cloud computing queue. While the pipeline runs, you can read that paper you have been putting off, learn a new technique, or simply rest. By morning, you have a 3D structure ready for analysis.

```
Traditional workflow:
  Design sequence → Apply for compute → Wait in queue → Run prediction (24-48h)
  └───────────── 3-5 days (application slower than computation) ──────────────┘

Our workflow:
  Design sequence → Press Enter on laptop → Sleep → Analyze results by morning
  └────────────── 1 night ────────────────────────────────────┘
```

| Before | After |
| :--- | :--- |
| GPU cluster or cloud credits required | Runs on any laptop (CPU-only) |
| Apply + queue + 24-48h compute | Press Enter → 7h → Results |
| Research bottleneck: compute access | Research bottleneck: creativity |

---

## 4. DBTL Cycle

Our journey from "we want to evaluate circRNA drug efficacy" to a working 3D structure prediction pipeline was non-linear. Here is the honest story.

### Round 1: From Efficacy to Structure (Learn: data does not exist)

**Design**: We initially wanted to predict drug efficacy of our TNBC-targeting circRNA. The simplest approach seemed to be training an MLP on clinical data to map sequence → efficacy.

**Build**: We searched for clinical data. It does not exist. CircRNA therapeutics are too early-stage — there are no large-scale clinical datasets linking circRNA sequence to therapeutic outcomes.

**Learn**: We cannot predict efficacy directly. Instead, we decomposed "efficacy" into three components: (1) microenvironment simulation, (2) physical 3D structure, and (3) pharmacokinetics. This project focuses on component (2): predicting the 3D structure that determines how the circRNA folds and presents itself to the cellular machinery.

### Round 2: Existing Tools All Fail on Long circRNAs (Learn: nothing off-the-shelf works)

**Design**: We tried AlphaFold3, the state-of-the-art for protein structure prediction.

**Build**: AlphaFold3 produces stereochemical errors on RNA and cannot handle sequences beyond ~500 nt. We tried IsRNAcirc, the only dedicated circRNA tool — it requires ~1,593 CPU hours (~66 days) for a 249-nt sequence, infeasible for our 2013-nt target. We tried 3dRNA, which fails on long sequences without templates. We tried RhoFold+ and trRosettaRNA, both designed for linear RNA — RhoFold+ crashes on long sequences, and trRosettaRNA requires a commercial license and cannot handle circular topology.

**Learn**: No existing tool can predict 3D structure of a 2013-nt circRNA. We must build our own. We decided to pursue two parallel directions: deep learning (DL) and physics-based simulation, converging into a hybrid pipeline.

### Round 3: CG Force Field Explodes (Learn: minimal-first is necessary)

**Design**: Following IsRNAcirc's approach, we started with a coarse-grained (CG) molecular dynamics force field.

**Build**: The full CG force field (backbone bonds + stacking + angles + clashes + base pairing) immediately exploded on the 2013-nt chain. Energy reached 70T kJ/mol. We spent days tuning force constants — reducing stacking epsilon, softening clash potentials, adjusting dihedral weights — with incremental improvements but no fundamental fix.

**Learn**: The full force field has too many competing terms for a long circular chain. Inspired by minimal models in polymer physics, we designed a **minimal force field** with only two terms: backbone bonds and base-pairing springs. This stripped-down field allowed the chain to fold freely from 45 A closure distance to 21 A. Only after this initial collapse did we switch to the full force field for refinement.

### Round 4: The MSA Problem and the Kabsch Moment (Learn: chunking is the key)

**Design**: We looked at CASP15 (the "World Cup of computational structural biology") and saw RhoFold+ and trRosettaRNA competing at the highest level. We decided to use trRosettaRNA to generate a base structure, then circularize it.

**Build**: trRosettaRNA requires an MSA (multiple sequence alignment) as input, but our circRNA has no evolutionary homologs — MSA is essentially empty. Even with a full MSA, trRosettaRNA cannot handle sequences >500 nt.

**Learn**: This was the turning point. We realized that no MSA is long enough for a 2013-nt sequence — but **segments** of the MSA can be. Inspired by structRFM's overlapping sliding window strategy, and drawing on our computer vision background with the Kabsch algorithm for structural alignment, we had the key insight: **predict 3D structure for short chunks independently, then stitch them together using overlap regions**.

We segmented the sequence at secondary structure boundaries (each chunk ≤200 nt), predicted each chunk with RhoFold+, and aligned overlap regions using the Kabsch algorithm to eliminate stitching artifacts. When no real MSA was available, we generated **pseudo MSA** from ViennaRNA pairing information — co-varying paired positions to force structural information into the predictor.

### Round 5: REST2, Dual Force Fields, and RL (Learn: refinement is everything)

**Design**: The stitched structure had visible artifacts at chunk boundaries. We needed refinement.

**Build**: We implemented REST2 (Replica Exchange with Solute Tempering) to relax boundary artifacts. We built a 3-bead CG force field (P/C4'/N) for fast refinement, then a 5-bead force field (P/S/B1/B2/B3) for higher resolution. For long sequences where many predicted pairs are ignored, we added RL-guided pair weight optimization — an agent learns which pairs need reinforcement and how many MD steps to run per round.

**Learn**: Local minima were a persistent problem. Structures would converge to suboptimal conformations and refuse to improve. We added Metadynamics (well-tempered) to actively cross free energy barriers, using BSJ distance, pair contact fraction, and radius of gyration as collective variables.

### Final Architecture

The result is a 6-level progressive refinement pipeline:

```
Level 0: ViennaRNA + complementarity scan + Jaccard inference
Level 1: Segmented 3D prediction (RhoFold+ chunks + Kabsch alignment)
Level 2: RL-guided CG refinement (3-bead REMD + pair weight optimization)
Level 3: 5-bead CG refinement + all-atom reconstruction
Level 4: Metadynamics + REST2 enhanced sampling
Level 5: AMBER RNA.OL3 all-atom refinement
```

### DBTL Summary

| Round | What We Tried | What Failed | What We Learned |
| :--- | :--- | :--- | :--- |
| 1 | MLP on clinical data | No clinical data exists | Decompose efficacy → structure first |
| 2 | AlphaFold3, IsRNAcirc, 3dRNA | All fail on long circRNA | Must build our own pipeline |
| 3 | Full CG force field | Energy explosion (70T) | Minimal-first folding is necessary |
| 4 | trRosettaRNA + MSA | MSA too short, license required | Chunk + Kabsch alignment + pseudo MSA |
| 5 | 3-bead → 5-bead → RL → MetaD | Local minima | Progressive refinement + enhanced sampling |

---

## 5. Conclusion

* **End-to-end pipeline**: From sequence to 3D coordinates, covering 2000+ nt circular RNAs, filling the gap between template-based, ab initio, and deep learning methods
* **Five core innovations**: Complementarity scan + BFS topological distance, Jaccard shared-partner inference, multi-stage physical simulation, RL-guided relaxation, Metadynamics + Temperature REMD enhanced sampling
* **Progressive refinement**: From ViennaRNA secondary structure to enhanced sampling, with automatic checkpoint/resume at each level
* **Validated results**: BSJ closure 6.07 A, 97.42% pair constraint satisfaction; RNAdvisor triple validation: rsRNASP -57,171, DFIRE -319,915, 3dRNAscore 27.59 (> 20, specific folding) on a 2013-nt circRNA
* **Benchmarked against crystal structure**: 1.84 A RMSD on 2OIU (71-nt, PDB: 2OIU), with BSJ closure error of only 0.02 A
* **Ablation validated**: RL-MCTS improves far pair satisfaction by 75% over fixed parameters
* **Practical impact**: 7-hour CPU-only runtime enables day-level design-predict-characterize cycle with wet lab collaborators

---

## Code Availability

The pipeline source code is available at [GitHub: RomanCohort/TorusFold-scheme2-rl](https://github.com/RomanCohort/TorusFold-scheme2-rl). Requirements: Python 3.10+, OpenMM 8.0+, ViennaRNA 2.6+. A single command runs the full pipeline:

```bash
python run_2013nt.py
```

---

## References

1. Chen LL. (2020). The expanding regulatory mechanisms and cellular functions of circular RNAs. *Nat Rev Mol Cell Biol*, 21(8), 475-490.
2. Kristensen LS, et al. (2019). The biogenesis, biology and characterization of circular RNAs. *Nat Rev Genet*, 20(11), 675-691.
3. Liu CX, et al. (2018). Structure and degradation of circular RNAs regulate PKR activation in innate immunity. *Cell*, 173(4), 966-981.
4. Pamudurti NR, et al. (2017). Translation of circRNAs. *Mol Cell*, 66(1), 9-21.
5. Legnini I, et al. (2017). Circ-ZNF609 is a circular RNA that can be translated into proteins in a cap-independent manner. *Cell Rep*, 19(1), 126-133.
6. Yang Y, et al. (2018). Extensive translation of circular RNAs driven by N6-methyladenosine. *Cell Res*, 27(5), 626-641.
7. PDB ID: 2OIU. The only complete circular RNA 3D structure.
8. Xiao M, et al. (2023). IsRNAcirc: prediction of circular RNA 3D structures via coarse-grained molecular dynamics simulations. *J Chem Theory Comput*.
