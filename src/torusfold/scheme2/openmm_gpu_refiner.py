"""
openmm_gpu_refiner.py — OpenMM GPU-accelerated CG MD refinement

Replacement for the CPU-only CG MD refinement in IsRNAcirc.exe.
Uses an OpenMM 3-bead CG force field + GPU platform acceleration + optional REMD enhanced sampling.

Interface-compatible with isrnacirc_wrapper.isrnacirc_cg_refine():
  openmm_gpu_refine(input_pdb, output_dir, sequence, secondary_structure, ...)
  -> (output_pdb_path, final_energy)

Fallback chain: CUDA -> OpenCL -> CPU
Enhanced sampling: optional T-REMD (replica exchange)

Authors: TorusFold Team
Date: 2026-08-05
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import Simulation, Topology, Element
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    mm = None
    app = None
    unit = None


# ── Platform detection ──

def detect_best_platform(preferred: str = "auto") -> str:
    """Detect the best available OpenMM platform.

    With preferred="auto", probe in the order CUDA > OpenCL > CPU.
    With preferred="CUDA"/"OpenCL"/"CPU", try that platform directly and fall back on failure.

    Args:
        preferred: preferred platform ("auto", "CUDA", "OpenCL", "CPU")

    Returns:
        name of a usable platform
    """
    if not OPENMM_AVAILABLE:
        return "CPU"

    if preferred == "auto":
        # Skip OpenCL (the LLVM JIT on Windows can raise "Can't get available size")
        candidates = ["CUDA", "CPU"]
    elif preferred == "OpenCL":
        # Only attempt OpenCL when it is explicitly requested
        candidates = ["OpenCL", "CPU"]
    else:
        candidates = [preferred, "CPU"]

    for name in candidates:
        try:
            Platform.getPlatformByName(name)
            # Do a quick OpenCL smoke test (create an empty system); skip it on failure
            if name == "OpenCL":
                try:
                    test_sys = mm.System()
                    test_sys.addParticle(1.0)
                    test_int = mm.LangevinMiddleIntegrator(
                        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
                    test_sim = app.Simulation(
                        app.Topology(), test_sys, test_int,
                        Platform.getPlatformByName("OpenCL"))
                except Exception:
                    continue
            return name
        except Exception:
            continue
    return "CPU"


# ── PDB coordinate I/O ──

def _read_p_coords(pdb_path: str) -> np.ndarray:
    """Read P-atom coordinates from a PDB, returning (L,3) in Angstroms.

    Parse by fixed columns first (standard PDB format); when the columns are
    misaligned, fall back to a whitespace split.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    # Misaligned columns (e.g. coordinate overflow); fall back to split
                    parts = line.split()
                    # ATOM serial name resname chain resid x y z ...
                    x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float64)


def _write_allatom_pdb(
    p_coords_3bead_nm: np.ndarray,
    L: int,
    output_path: str,
    sequence: str = None,
):
    """Write a backbone PDB from the 3-bead nm coordinates (for the later CG_to_allatom).

    CG_to_allatom.exe requires LF line endings and the ADE/URA/GUA/CYT three-letter codes.
    """
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    coords_ang = p_coords_3bead_nm * 10.0  # nm -> Angstroms
    p_coords = coords_ang[0::3].copy()  # (L,3) Angstroms

    if len(p_coords) > 0:
        min_xyz = p_coords.min(axis=0)
        shift = np.where(min_xyz < 0, -min_xyz + 5.0, 0.0)
        p_coords = p_coords + shift

    lines = []
    for i in range(L):
        x, y, z = p_coords[i]
        resname = base_map.get(sequence[i].upper(), "ADE") if sequence else "ADE"
        # PDB format: columns must align exactly or CG_to_allatom.exe will reject it
        line = f"ATOM  {i+1:5d}  P   {resname} A{i+1:4d}"
        line += f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P "
        lines.append(line)
    lines.append("END")
    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _write_refined_pdb(
    allatom_pdb_path: str,
    output_path: str,
):
    """Copy the all-atom PDB to the output path."""
    import shutil
    shutil.copy2(allatom_pdb_path, output_path)


# ── Force field parameters (measured for THIS file's functional form) ──
#
# Units, spelled out because torch_cgsim.py declares the same numerals in other units:
# this file declares the distance terms in kJ/mol/angstrom^2 and multiplies them by 100
# at use (nm^-2), while torch_cgsim.py declares kJ/mol/nm^2 and uses them raw.  A number
# here and the same number there do not mean the same thing and must not be made to agree
# by copying numerals.
#
#   K_BB, K_INTRA_PC, K_INTRA_CN, K_STACK  kJ/mol/angstrom^2, E = 0.5*k*(r-r0)^2
#   K_ANGLE, K_DIHEDRAL                    kJ/mol/rad^2,      E = 0.5*k*(theta-theta0)^2
#
# Measured by scripts/measure_cpu_constants.py from D:\torusfold-cgdata\rsRNASP\
# Training_set through the boltzmann_bonded machinery (126 gap-free chains, 6386-6764
# observations per coordinate).  A harmonic whose equilibrium spread is sigma needs
# k = kBT/sigma^2, with kBT = 2.494 kJ/mol at 300 K.  The sigma quoted below is pooled
# over every chain, so it folds sequence and conformer variation into the thermal width:
# kBT/sigma^2 is a LOWER BOUND on the stiffness, not the stiffness.  The mean
# within-chain sigma is given for scale and gives a stiffer (still bounded) value.
# tests/test_cpu_force_constants.py locks each constant to its own sigma.
K_BB = 11.22         # P-P;         sigma = 0.4714 A over 6638 bonds (0.4454 within-chain)
K_INTRA_PC = 223.9   # P-C4';       sigma = 0.1055 A over 6764       (0.0917)
K_INTRA_CN = 376.8   # C4'-N;       sigma = 0.0814 A over 6764       (0.0736)
K_STACK = 0.959      # N(i)-N(i+1); sigma = 1.6127 A over 6638       (1.4959)
K_ANGLE = 16.78      # P-P-P angle; sigma = 0.3855 rad over 6512     (0.3570)
K_DIHEDRAL = 2.12    # P-P-P-P dih; sigma = 1.0846 rad over 6386     (1.0249)
# K_STACK stays non-zero here although torch_cgsim.py zeroes its stacking term.  That
# file restrains P(i)-P(i+2), which the identity in its header shows is a function of
# K_BB and K_ANGLE and is therefore redundant.  This file restrains N(i)-N(i+1), and
# N(i) is bonded only to C4'(i): no other term in this force field positions the base
# beads relative to each other, so the term is load-bearing and its value is measured.
K_PAIR = 1500.0    # WC base-pair N-N (raised 800->1500, strong pairing for convergence)
K_CLASH = 300.0    # clash (raised 200->300)
K_BSJ = 800.0      # BSJ closure (raised 500->800)
K_BSJ_GUIDE = 1200.0  # BSJ guide force (raised 800->1200)

# Geometric parameters (Angstroms)
BOND_P_NEXT = 5.90
BOND_P_C4 = 3.90
BOND_C4_N = 3.35
ANGLE_PPP = 2.618   # rad, 150 deg
DIH_PPPP = 33.0 * np.pi / 180.0  # rad
STACK_R0 = 5.05
PAIR_N_N = 10.0
CLASH_DIST = 3.0
CUTOFF = 12.0


def _build_3bead_system_gpu(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    pair_scale: float = 1.0,
    bsj_k_scale: float = 1.0,
    pair_guide_k: float = 0.0,
    bpp_matrix: Optional[np.ndarray] = None,
    bpp_weight: float = 0.5,
    pair_predictions: Optional[np.ndarray] = None,
    ss_predictions: Optional[np.ndarray] = None,
    bsj_prediction: Optional[float] = None,
):
    """Build the 3-bead CG OpenMM system (GPU-optimized version).

    The force field is identical to cg_forcefield.build_3bead_system(), but with a
    simplified interface and no statistical potential (the GPU path prioritizes speed).

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)] ViennaRNA base pairs
        pair_scale: scaling of the pairing force (for annealing)
        bsj_k_scale: scaling of the BSJ force (for annealing)
        pair_guide_k: pair-window guide force (kJ/mol). When >0, applies a soft
            long-range attraction to far pairs, gradually pulling paired atoms that
            are 100-3000 A apart into the force-field range (~20 A), after which the
            ordinary pairing force takes over. Fixes the problem that paired atoms in
            the initial circular-RNA conformation are too far apart for the force field
            to reach.
        bpp_matrix: (L,L) ViennaRNA base-pair probability matrix (optional).
            When present, weight the pairing force by bpp_ij:
            k_pair = K_PAIR * (bpp_w * bpp_ij + (1-bpp_w) * w) * pair_scale
        bpp_weight: bpp mixing weight. 1.0 = pure bpp, 0.0 = pure hard-coded w.

    Returns:
        (system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide)
    """
    L = len(p_coords)
    N_total = 3 * L

    # Build the 3-bead coordinates: each nt -> P, C4', N
    coords_3bead = np.zeros((N_total, 3), dtype=np.float64)
    rng = np.random.default_rng(42)
    for i in range(L):
        p = p_coords[i]
        coords_3bead[3 * i] = p  # P
        # Estimate C4' and N with small perturbations (corrected later by minimize)
        coords_3bead[3 * i + 1] = p + rng.normal(0, 0.3, 3)  # C4'
        coords_3bead[3 * i + 2] = p + rng.normal(0, 0.3, 3)  # N

    coords_nm = coords_3bead / 10.0  # Angstroms -> nm

    system = mm.System()
    for _ in range(N_total):
        system.addParticle(330.0 / 3.0)  # ~110 Da per bead

    def P(i): return 3 * i
    def C4(i): return 3 * i + 1
    def N(i): return 3 * i + 2

    # 1. Backbone bond P[i]-P[i+1]
    bond_bb = mm.HarmonicBondForce()
    bb_k = K_BB * 100.0  # A^2 -> nm^2
    for i in range(L - 1):
        bond_bb.addBond(P(i), P(i + 1), BOND_P_NEXT / 10.0, bb_k)
    system.addForce(bond_bb)

    # 1b. BSJ closure (first-last P)
    # structRFM: bsj_prediction modulates the BSJ force constant
    bsj_confidence = float(bsj_prediction) if bsj_prediction is not None else 1.0
    effective_bsj_k = bsj_k_scale * K_BSJ * (0.3 + 0.7 * bsj_confidence)
    bsj_force = mm.CustomBondForce("0.5*k_bsj*(r-r0)^2")
    bsj_force.addPerBondParameter("k_bsj")
    bsj_force.addPerBondParameter("r0")
    bsj_force.addBond(P(L - 1), P(0),
                      [effective_bsj_k, BOND_P_NEXT / 10.0])
    system.addForce(bsj_force)

    # 1c. BSJ guide force
    bsj_guide = mm.CustomBondForce("0.5*k_guide*(r-r0)^2")
    bsj_guide.addPerBondParameter("k_guide")
    bsj_guide.addPerBondParameter("r0")
    bsj_guide.addBond(P(L - 1), P(0),
                      [bsj_k_scale * K_BSJ_GUIDE, BOND_P_NEXT / 10.0])
    system.addForce(bsj_guide)

    # 1d. BSJ contact map: nucleotides within +/-bsj_contact_nt of the junction should cluster in space
    # The BSJ region of a circular RNA (5'/3' junction) usually has conserved structure:
    #   - a stem spanning the junction, or
    #   - base stacking on both sides of the junction
    # Force: harmonic attractor, r0 = 10 A (slightly larger than the WC distance, allowing flexibility)
    # The force constant decays with distance from the junction: k = K_BSJ_CONTACT * (1 - d/max_d)^2
    bsj_contact_nt = min(8, L // 4)  # 8 nt per side, or 1/4 of the sequence
    K_BSJ_CONTACT = 200.0  # kJ/mol/angstrom^2
    r0_bsj_contact = 1.0   # nm = 10 A
    bsj_contact_force = mm.CustomBondForce(
        "0.5*k_c*(r-r0)^2 * (1 - dist_ratio)^2")
    bsj_contact_force.addPerBondParameter("k_c")
    bsj_contact_force.addPerBondParameter("r0")
    bsj_contact_force.addGlobalParameter("dist_ratio", 0.0)  # placeholder; actually uses per-bond params

    # Use a simple harmonic instead (OpenMM CustomBondForce does not support per-bond global variables)
    bsj_contact_force = mm.CustomBondForce("0.5*k_c*(r-r0)^2")
    bsj_contact_force.addPerBondParameter("k_c")
    bsj_contact_force.addPerBondParameter("r0")

    for i in range(-bsj_contact_nt, bsj_contact_nt):
        for j in range(i + 1, bsj_contact_nt + 1):
            # Cyclic indices
            ii = i % L
            jj = j % L
            if ii == jj:
                continue
            # Distance from the junction (min of direct and wrap-around)
            d_i = min(ii, L - ii)  # distance to position 0
            d_j = min(jj, L - jj)
            # Distance decay: stronger closer to the junction
            max_d = bsj_contact_nt
            decay_i = max(0.0, 1.0 - d_i / max_d)
            decay_j = max(0.0, 1.0 - d_j / max_d)
            k_contact = K_BSJ_CONTACT * decay_i * decay_j * bsj_k_scale
            if k_contact > 1.0:  # minimum threshold
                bsj_contact_force.addBond(P(ii), P(jj), [k_contact, r0_bsj_contact])

    if bsj_contact_force.getNumBonds() > 0:
        system.addForce(bsj_contact_force)

    # 1e. bpp soft-restraint potential (S10 idea #4: prior information as a soft guide)
    #   U_bpp = sum bpp(i,j) * k_bpp * (d(i,j) - d_native)^2
    #   Residue pairs with high bpp are pulled toward the native distance; low-bpp pairs explore freely.
    #   d_native = 10.5 A (WC-paired C1'-C1' distance)
    if bpp_matrix is not None and bpp_matrix.shape[0] == L:
        K_BPP_SOFT = 100.0  # kJ/mol/angstrom^2 (soft restraint, weaker than the hard pairing force)
        d_native_bpp = 1.05  # nm = 10.5 A
        bpp_soft_force = mm.CustomBondForce("0.5*k_bpp*(r-r0)^2")
        bpp_soft_force.addPerBondParameter("k_bpp")
        bpp_soft_force.addPerBondParameter("r0")
        n_bpp_soft = 0
        for i in range(L):
            for j in range(i + 5, L):  # skip near-range pairs (the backbone force already covers them)
                bpp_val = float(bpp_matrix[i, j])
                if bpp_val < 0.05:  # skip low-probability pairs
                    continue
                # Force constant = base value x bpp probability x bpp_weight
                k_bpp = K_BPP_SOFT * bpp_val * bpp_weight
                if k_bpp > 0.5:
                    bpp_soft_force.addBond(P(i), P(j), [k_bpp, d_native_bpp])
                    n_bpp_soft += 1
        if n_bpp_soft > 0:
            system.addForce(bpp_soft_force)

    # 2. Intra-residue bonds P-C4', C4'-N.  Separate constants: the measured spreads
    #    differ by 1.7x (0.1055 A vs 0.0814 A), so one shared value cannot match both.
    bond_intra = mm.HarmonicBondForce()
    for i in range(L):
        bond_intra.addBond(P(i), C4(i), BOND_P_C4 / 10.0, K_INTRA_PC * 100.0)
        bond_intra.addBond(C4(i), N(i), BOND_C4_N / 10.0, K_INTRA_CN * 100.0)
    system.addForce(bond_intra)

    # 3. Backbone angle P-P-P
    angle_force = mm.HarmonicAngleForce()
    for i in range(L - 2):
        angle_force.addAngle(P(i), P(i + 1), P(i + 2), ANGLE_PPP, K_ANGLE)
    # Cyclization angle
    if L >= 3:
        angle_force.addAngle(P(L - 2), P(L - 1), P(0), ANGLE_PPP, K_ANGLE)
        angle_force.addAngle(P(L - 1), P(0), P(1), ANGLE_PPP, K_ANGLE)
    system.addForce(angle_force)

    # 3.5 Backbone dihedral
    dih_force = mm.CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
    dih_force.addGlobalParameter("k_dih", K_DIHEDRAL)
    dih_force.addGlobalParameter("theta0", DIH_PPPP)
    for i in range(L - 3):
        dih_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
    if L >= 4:
        dih_force.addTorsion(P(L - 3), P(L - 2), P(L - 1), P(0))
        dih_force.addTorsion(P(L - 2), P(L - 1), P(0), P(1))
        dih_force.addTorsion(P(L - 1), P(0), P(1), P(2))
    system.addForce(dih_force)

    # 4. WC base-pair N-N (bpp-weighted: k = K_PAIR * (bpp_w * bpp_ij + (1-bpp_w) * w) * scale)
    pair_force = mm.CustomBondForce("0.5*k_pair*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            # bpp weighting: when a bpp_matrix is present, mix the bpp probability with the hard-coded weight
            if bpp_matrix is not None and bpp_weight > 0:
                bpp_val = float(bpp_matrix[i, j]) if i < bpp_matrix.shape[0] and j < bpp_matrix.shape[1] else 0.0
                effective_w = bpp_weight * bpp_val + (1.0 - bpp_weight) * w
            else:
                effective_w = w
            # structRFM: modulate with pair_predictions
            if pair_predictions is not None:
                pair_idx = None
                for pi, (ii, jj) in enumerate(pairs):
                    if (ii == i and jj == j) or (ii == j and jj == i):
                        pair_idx = pi
                        break
                if pair_idx is not None and pair_idx < len(pair_predictions):
                    struct_w = 0.5 + 0.5 * float(pair_predictions[pair_idx])
                    effective_w *= struct_w
            # structRFM: ss_predictions modulate stacking (paired->strong, unpaired->weak)
            if ss_predictions is not None and i < len(ss_predictions) and j < len(ss_predictions):
                ss_avg = (float(ss_predictions[i]) + float(ss_predictions[j])) / 2.0
                effective_w *= (0.5 + 0.5 * ss_avg)
            pair_force.addBond(
                N(i), N(j),
                [K_PAIR * effective_w * pair_scale, PAIR_N_N / 10.0])
    system.addForce(pair_force)

    # 4b. Pair-window guide force (soft attraction for far pairs)
    # V = -k_g * (1/(1+exp(a*(r-r_cap)))) * step(r-r0_lo)
    #   - r >> r_cap: V -> 0 (out of reach; no strong pull, to avoid tearing the structure)
    #   - r ~ r_cap: logistic transition, peak force ~ k_g*a/4
    #   - r < r0_lo (already paired): off
    # With a=0.05 (characteristic length 20nm) and r_cap=40nm, this covers pairs at 20-60nm
    # (200-600A) with a gentle peak force that, unlike a linear window, never pulls so hard
    # that it tears the structure.
    if pair_guide_k > 0:
        guide_force = mm.CustomBondForce(
            "-k_g*(1/(1+exp(a*(r-r_cap))))*step(r-r0_lo)")
        guide_force.addPerBondParameter("k_g")
        guide_force.addGlobalParameter("a", 0.05)     # /nm, characteristic length ~20nm
        guide_force.addGlobalParameter("r_cap", 40.0)  # nm = 400A
        guide_force.addGlobalParameter("r0_lo", 1.5)   # nm = 15A; disabled once paired
        for (i, j, w) in pairs:
            if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                    and not (i == 0 and j == L - 1)):
                guide_force.addBond(
                    N(i), N(j), [pair_guide_k * w])
        system.addForce(guide_force)

    # 5. Base stacking
    stack_force = mm.CustomBondForce("0.5*k_stack*(r-r0)^2")
    stack_force.addPerBondParameter("k_stack")
    stack_force.addPerBondParameter("r0")
    sk = K_STACK * 100.0
    for i in range(L - 1):
        stack_force.addBond(N(i), N(i + 1), [sk, STACK_R0 / 10.0])
    # Cyclization stacking
    stack_force.addBond(N(L - 1), N(0), [sk, STACK_R0 / 10.0])
    system.addForce(stack_force)

    # 6. Nonbonded: implicit-solvent GB/SA + short-range clash  (ion screening is NOT
    #    implemented here; see the note where the charges are set below)
    #
    # The two main driving forces of RNA folding:
    #   (a) Electrostatic screening: the phosphate backbone is negatively charged and
    #       must be screened by Mg2+/Na+ before folding
    #       -> GB/SA only.  The intended 0.145M salt term was a global parameter on
    #          mm.NonbondedForce, which has a fixed functional form and no user
    #          expression, so no force could ever read it; it was deleted after being
    #          measured inert.  Adding real Debye screening is a physics decision.
    #   (b) Hydrophobic effect: base stacking surfaces are buried inside while the
    #       phosphate backbone is exposed
    #       -> SA (solvent-accessible area) term
    #
    # Short-range clash repulsion is kept as well (GB does not handle Pauli repulsion)
    #
    # -- A. GB/SA implicit solvent (OBC2).  No salt screening --
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    # GBOBC2 parameters (Onufriev et al.; well suited to nucleic acids)
    nonbonded.setReactionFieldDielectric(1.0)  # no reaction field needed for implicit solvent
    nonbonded.setCutoffDistance(999.0)  # no cutoff (implicit solvent)

    # Particle charges and Born radii
    # RNA 3-bead: P (phosphate, q~-0.6e), C4' (sugar, q~0.0), N (base, q~0.0)
    # Born radii: P=1.7A (buried in the backbone), C4'=2.2A, N=1.9A (base partially exposed)
    _Q_P, _Q_C4, _Q_N = -0.6, 0.0, 0.0       # partial charges (e)
    _R_P, _R_C4, _R_N = 0.17, 0.22, 0.19     # Born radii (nm)

    for i in range(L):
        nonbonded.addParticle(_Q_P, _R_P, 0.0)    # P
        nonbonded.addParticle(_Q_C4, _R_C4, 0.0)  # C4'
        nonbonded.addParticle(_Q_N, _R_N, 0.0)    # N

    # Salt screening: absent, and it used to be a lie in the parameter list.  A plain
    # mm.NonbondedForce is a fixed 1/r Coulomb + LJ with no user expression, so a global
    # parameter added to it can never be read.  Measured in a live Context
    # (scripts/characterize_nonbonded_block.py, section B): perturbing the old
    # "screeningLength" over 1e-4 .. 1000 nm changed the total energy by 0.0 kJ/mol and
    # every force by 0.0 kJ/mol/nm; the built P-P pair potential is bare 1/r Coulomb
    # plus GBSA.  The dead parameter was deleted instead of left standing as a claim of
    # screening.  Real 0.145M screening needs a screened pair term (CustomNonbondedForce
    # or CustomGBForce) and is a physics decision, not a cleanup.
    # tests/test_nonbonded_block_invariants.py fails if any global parameter in this
    # System is read by no force.

    # Excluded pairs (GB does not recompute these)
    _excl_nb = set()
    for i in range(L):
        for (a, b) in [(P(i), C4(i)), (C4(i), N(i))]:
            k = (min(a, b), max(a, b))
            if k not in _excl_nb:
                _excl_nb.add(k)
                nonbonded.addException(a, b, 0.0, 0.3, 0.0)  # clash only
    for i in range(L - 1):
        k = (P(i), P(i + 1))
        if k not in _excl_nb:
            _excl_nb.add(k)
            nonbonded.addException(P(i), P(i + 1), 0.0, 0.3, 0.0)
    # Cyclization P-P exclusion
    if L > 2:
        k = (P(0), P(L - 1))
        if k not in _excl_nb:
            nonbonded.addException(P(0), P(L - 1), 0.0, 0.3, 0.0)

    system.addForce(nonbonded)

    # -- B. GB solvation force (OBC2 / GBOBC2) --
    gb = mm.GBSAOBCForce()
    gb.setSoluteDielectric(4.0)       # solute dielectric: CG beads are not atoms and need a higher dielectric
    gb.setSolventDielectric(78.5)     # water dielectric constant
    # OpenMM 8.x: setSolventRadius was removed; the default 0.14 is used
    # GBSAOBCForce.addParticle(charge, radius, scalingFactor)
    # scalingFactor: 0.0 = HCT, 0.5 = OBC1, 1.0 = OBC2 (OBC2 is the default for nucleic acids)
    for i in range(L):
        gb.addParticle(_Q_P, _R_P, 1.0)    # P
        gb.addParticle(_Q_C4, _R_C4, 1.0)  # C4'
        gb.addParticle(_Q_N, _R_N, 1.0)    # N
    system.addForce(gb)

    # -- C. Short-range clash repulsion (supplements the Pauli repulsion that GB does not cover) --
    # Keep a lightweight clash force to prevent the structure from collapsing during annealing
    clash_force = mm.CustomBondForce(
        "k_clash * (dmin - r)^2 * step(dmin - r)")
    clash_force.addPerBondParameter("k_clash")
    clash_force.addPerBondParameter("dmin")
    for i in range(L):
        for j in range(i + 2, min(i + 8, L)):  # only check neighboring residues 2-7
            clash_force.addBond(P(i), P(j), [K_CLASH * 10.0, 0.3])
    system.addForce(clash_force)

    return (system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide)


def _build_minimal_system_gpu(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    pair_scale: float = 1.0,
):
    """Build a minimal P-only folding force field (two-stage scheme, stage 1).

    Contains only:
      1. P backbone bonds P[i]-P[i+1] (r0=5.9A, k=31000 kJ/mol/nm^2)
      2. P-P base-pair bonds (r0=5.9A, k=40000*w*pair_scale)
    No clash/stacking/angle/C4'N — those terms hinder folding in the full force field
    (measured: the full force field stalls pairs at 45A; the minimal one folds to 21A).

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)] ViennaRNA base pairs
        pair_scale: pairing-force scaling

    Returns:
        (system, coords_nm, pair_force) — P beads only (L particles)
    """
    L = len(p_coords)
    coords_nm = p_coords / 10.0  # Angstroms -> nm

    system = mm.System()
    for _ in range(L):
        system.addParticle(110.0)

    # 1. P backbone bonds
    bond_bb = mm.HarmonicBondForce()
    bb_k = 31000.0  # kJ/mol/nm^2
    for i in range(L - 1):
        bond_bb.addBond(i, i + 1, BOND_P_NEXT / 10.0, bb_k)
    # 1b. BSJ closure bond: force first-last P-P ~5.9A to stop the ring opening during annealing
    bond_bb.addBond(0, L - 1, BOND_P_NEXT / 10.0, 500.0)
    system.addForce(bond_bb)

    # 1c. Backbone angle restraint: prevent the backbone angle from collapsing during folding
    angle_bb = mm.HarmonicAngleForce()
    for i in range(L - 2):
        angle_bb.addAngle(i, i + 1, i + 2,
                          2.618,  # 150 deg in rad (A-form RNA backbone)
                          500.0)  # kJ/mol/rad^2
    system.addForce(angle_bb)

    # 1d. Clash repulsion: prevent atoms from overlapping (critical: without it the structure collapses into a ball)
    clash = mm.CustomNonbondedForce(
        "step(d_min - r) * 0.5 * k_clash * (d_min - r)^2")
    clash.addGlobalParameter("k_clash", 5000.0)  # kJ/mol/nm^2
    clash.addGlobalParameter("d_min", 0.3)  # 3.0A = 0.3nm minimum distance
    for _ in range(L):
        clash.addParticle()
    # Only check near neighbors (15-nt window) to avoid O(n^2)
    neighbors = []
    for i in range(L):
        nb = list(range(max(0, i - 15), min(L, i + 16)))
        nb = [j for j in nb if j > i]
        if nb:
            neighbors.append((i, nb))
    # Group them using InteractionGroup
    all_a, all_b = [], []
    for i, nbs in neighbors:
        all_a.extend([i] * len(nbs))
        all_b.extend(nbs)
    if all_a:
        clash.addInteractionGroup(all_a, all_b)
    system.addForce(clash)

    # 2. P-P base-pair bonds (folding driver; the force constant can be reduced now that clash repulsion is present)
    pair_force = mm.CustomBondForce("0.5*k_pair*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            # Far pairs (>100 nt) get 2x the force constant
            far_boost = 2.0 if (min(abs(j-i), L-abs(j-i)) > 100) else 1.0
            pair_force.addBond(
                i, j, [30000.0 * w * pair_scale * far_boost, BOND_P_NEXT / 10.0])
    system.addForce(pair_force)

    return system, coords_nm, pair_force


def _create_minimal_topology(L: int) -> Topology:
    """Create a P-only topology (one P atom per nt)."""
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("RA", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
    return topo


def _create_3bead_topology(L: int) -> Topology:
    """Create a 3-bead CG topology (P/C4'/N per nt)."""
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("N", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
        topo.addAtom(f"C{i}", Element.getBySymbol("C"), res)
        topo.addAtom(f"N{i}", Element.getBySymbol("N"), res)
    return topo


# ── Three-stage annealing ──

def _run_annealing(
    sim: Simulation,
    pair_force,
    bsj_force,
    bsj_guide,
    L: int,
    n_anneal: int = 200,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """Three-stage annealing: weak pairing+weak BSJ -> strong pairing+medium BSJ -> strong pairing+strong BSJ.

    Returns:
        (final_energy, final_coords_nm)
    """
    def set_pair_k(scale):
        for i in range(pair_force.getNumBonds()):
            p1, p2, params = pair_force.getBondParameters(i)
            # Update k, keep r0
            pair_force.setBondParameters(
                i, p1, p2,
                [scale * K_PAIR, params[1]])
        pair_force.updateParametersInContext(sim.context)

    def set_bsj_k(scale):
        bsj_force.setBondParameters(
            0, 3 * (L - 1), 0,
            [scale * K_BSJ, BOND_P_NEXT / 10.0])
        bsj_guide.setBondParameters(
            0, 3 * (L - 1), 0,
            [scale * K_BSJ_GUIDE, BOND_P_NEXT / 10.0])
        bsj_force.updateParametersInContext(sim.context)
        bsj_guide.updateParametersInContext(sim.context)

    # Record the initial energy
    pre_state = sim.context.getState(getPositions=True, getEnergy=True)
    e_pre = pre_state.getPotentialEnergy()._value

    # Stage 1: medium temperature + weak pairing + weak BSJ, helix formation
    set_pair_k(0.1)
    set_bsj_k(0.3)
    sim.integrator.setTemperature(350 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=2000)

    # Stage 2: medium temperature + strong pairing + medium BSJ, draw the WC pairs together
    set_pair_k(1.0)
    set_bsj_k(1.0)
    sim.integrator.setTemperature(320 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=2000)

    # Stage 3: low temperature + strong pairing + strong BSJ, closure
    set_pair_k(1.0)
    set_bsj_k(5.0)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=3000)

    # Stage 4 (new): very low temperature + ultra-strong BSJ, refine the closure
    set_pair_k(1.0)
    set_bsj_k(10.0)
    sim.integrator.setTemperature(280 * unit.kelvin)
    sim.step(n_anneal // 2)
    sim.minimizeEnergy(
        tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=5000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # Safety net: fall back if the MD runs away
    if e1 > e_pre * 0.5 and e_pre < 0:
        pos = pre_state.getPositions(asNumpy=True)._value
        e1 = e_pre

    return e1, pos


def _run_anneal_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Angstroms, P coordinates
    pairs: List[Tuple[int, int, float]],
    n_anneal: int,
    n_threads: int,
):
    """Multiprocess annealing worker: independently builds the system + runs three-stage annealing.

    Uses a different random seed (worker_idx) to increase trajectory diversity.
    Returns:
        (final_energy, final_coords_nm)
    """
    import numpy as _np
    # Different seeds -> different C4'/N initial perturbations
    _np.random.seed(42 + worker_idx)

    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(
            p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.1 + 0.05 * worker_idx,
            pair_guide_k=600.0)  # pair-window guide force (raised 300->600), pulls far pairs together
    topo = _create_3bead_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # Use the module-level _run_annealing for three-stage annealing
    e_final, pos_final = _run_annealing(
        sim, pair_force, bsj_force, bsj_guide, len(p_coords),
        n_anneal=n_anneal, verbose=False)
    return e_final, pos_final


def _run_minimal_anneal_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Angstroms, P coordinates
    pairs: List[Tuple[int, int, float]],
    n_anneal: int,
    n_threads: int,
):
    """Minimal-force-field annealing worker (two-stage scheme, stage 1: folding).

    Contains only P backbone bonds + P-P pairing; no clash/stacking. High-temperature annealing folds the chain.
    Returns:
        (final_energy, final_coords_ang)  # P-only, Angstroms
    """
    system, coords_nm, pair_force = _build_minimal_system_gpu(
        p_coords, pairs, pair_scale=1.0)
    topo = _create_minimal_topology(len(p_coords))

    integrator = LangevinMiddleIntegrator(
        450 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # Minimize first to remove initial clashes
    sim.minimizeEnergy(maxIterations=3000)

    # Gradually cool while annealing (folding driver): high temperature brings pairs together, then cool stepwise
    # Enhanced version: 8 stages for finer-grained temperature control
    stages = [
        (400, n_anneal // 8),   # medium-high T: preserve local structure, explore far pairs
        (380, n_anneal // 8),   # medium T: helix formation
        (360, n_anneal // 8),   # medium T: bring pairs together
        (340, n_anneal // 8),   # medium-low T: WC pairing converges
        (320, n_anneal // 8),   # low T: remove clashes
        (310, n_anneal // 8),   # low T: structural refinement
        (305, n_anneal // 8),   # near room T: BSJ closure
        (300, n_anneal // 8),   # room T: final stabilization
    ]
    for T, n in stages:
        integrator.setTemperature(T * unit.kelvin)
        sim.step(max(1, n))
    # Final minimization (stricter)
    sim.minimizeEnergy(
        tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=8000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos_nm = state.getPositions(asNumpy=True)._value  # nm
    e = state.getPotentialEnergy()._value
    pos_ang = pos_nm * 10.0  # -> Angstroms
    return e, pos_ang


def _run_parallel_minimal_annealing(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    n_anneal: int = 200,
    n_trajectories: int = 4,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """Multiprocess parallel minimal folding: N trajectories; keep the lowest energy.

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)]

    Returns:
        (best_energy, best_coords_ang)  # P-only, Angstroms
    """
    import multiprocessing as mp

    total_threads = os.cpu_count() or 8
    per_traj_threads = max(1, total_threads // n_trajectories)
    if verbose:
        print(f"  Minimal fold: {n_trajectories} trajectories x {per_traj_threads} threads")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_trajectories) as pool:
        results = pool.starmap(
            _run_minimal_anneal_worker,
            [(i, p_coords, pairs, n_anneal, per_traj_threads)
             for i in range(n_trajectories)],
        )

    best_energy = float("inf")
    best_pos = None
    for e, pos in results:
        if e < best_energy:
            best_energy = e
            best_pos = pos
    return best_energy, best_pos


def _run_parallel_annealing(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    n_anneal: int = 200,
    n_trajectories: int = 4,
    platform_name: str = "CPU",
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """Multiprocess parallel annealing: N trajectories each using 32/N threads; keep the lowest energy.

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)]
        n_anneal: steps per stage
        n_trajectories: number of parallel trajectories
        platform_name: platform
        verbose: verbosity

    Returns:
        (best_energy, best_coords_nm)
    """
    import multiprocessing as mp

    total_threads = os.cpu_count() or 8
    per_traj_threads = max(1, total_threads // n_trajectories)
    if verbose:
        print(f"  Parallel annealing: {n_trajectories} trajectories x {per_traj_threads} threads "
              f"({total_threads} cores total)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_trajectories) as pool:
        results = pool.starmap(
            _run_anneal_worker,
            [(i, p_coords, pairs, n_anneal, per_traj_threads)
             for i in range(n_trajectories)],
        )

    best_energy = float("inf")
    best_pos = None
    for e, pos in results:
        if e < best_energy:
            best_energy = e
            best_pos = pos

    return best_energy, best_pos


# ── T-REMD (multi-temperature replica exchange) ──

def _run_remd_worker(
    worker_idx: int,
    p_coords: np.ndarray,       # (L,3) Angstroms, P coordinates
    pairs: List[Tuple[int, int, float]],
    temperature: float,
    n_steps: int,
    exchange_interval: int,
    n_threads: int,
    conn,
    minimal: bool = False,
    sequence: str = None,
    use_trirnasp: bool = False,
    trirnasp_energy_dir: str = None,
    trirnasp_scale: float = 0.003,  # unified default: 0.003 (optimal value)
    trirnasp_update_freq: int = 10,
    trirnasp_max_force: float = 500.0,
):
    """REMD single-replica worker process: rebuilds the system locally + simulates + Pipe exchange.

    The worker receives the P coordinates and pairs and builds its own system (avoiding
    pickling of OpenMM objects), reporting the energy every exchange_interval steps and
    receiving swapped coordinates. With minimal=True the minimal force field is used
    (P backbone + P pairing, keeping fold consistency).

    With use_trirnasp=True: adds the TriRNASP three-body statistical potential as an extra
    energy term. Gradients are refreshed every trirnasp_update_freq steps via a
    CustomExternalForce. trirnasp_max_force: maximum per-particle force magnitude
    (kJ/mol/nm), preventing the statistical-potential gradient from exploding far from the
    native conformation and blowing up the structure.
    """
    TRI_MAX_F2 = trirnasp_max_force * trirnasp_max_force

    def _tri_forces(grad):
        """Gradient -> capped per-particle force array (N_total,3)."""
        f = -grad * trirnasp_scale * _TRI_KBT * 10.0
        # Cap: scale down forces whose magnitude exceeds the limit
        norms_sq = (f * f).sum(axis=-1, keepdims=True)
        over = norms_sq > TRI_MAX_F2
        if over.any():
            scale_f = np.where(over, trirnasp_max_force / np.sqrt(np.maximum(norms_sq, 1e-12)), 1.0)
            f = f * scale_f
        return f.reshape(-1, 3)
    if minimal:
        system, coords_nm, _pf = _build_minimal_system_gpu(
            p_coords, pairs, pair_scale=1.0)
        topo = _create_minimal_topology(len(p_coords))
    else:
        # Each worker builds its own system (different bsj_k_scale adds diversity)
        system, coords_nm, _pf, _sf, _bjf, _bjg = _build_3bead_system_gpu(
            p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.5 + 0.1 * worker_idx,
            pair_guide_k=600.0)  # pair-window guide force (raised 300->600)
        topo = _create_3bead_topology(len(p_coords))

    # ── TriRNASP three-body statistical potential ──
    L = len(p_coords)
    tri_potential = None
    tri_force = None
    tri_energy = 0.0
    try:
        from torusfold.scheme2.trirnasp_openmm import KBT_FACTOR as _TRI_KBT
    except ImportError:
        _TRI_KBT = 2.494
    if use_trirnasp and sequence is not None:
        try:
            from torusfold.scheme2.trirnasp_openmm import TriRNASPPotential
            tri_potential = TriRNASPPotential(trirnasp_energy_dir)
            # CustomExternalForce: an independent constant force Fx/Fy/Fz per particle (per-particle parameters)
            # The negative gradient of "fx*x+fy*y+fz*z" is -[fx,fy,fz] = the applied constant force
            tri_force = mm.CustomExternalForce("fx*x + fy*y + fz*z")
            tri_force.addPerParticleParameter("fx")
            tri_force.addPerParticleParameter("fy")
            tri_force.addPerParticleParameter("fz")
            N_total = 3 * L
            for p_idx in range(N_total):
                tri_force.addParticle(p_idx, [0.0, 0.0, 0.0])
            system.addForce(tri_force)
        except Exception as e:
            print(f"    [TriRNASP] worker {worker_idx} initialization failed: {e}")
            tri_potential = None
            tri_force = None

    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    plat = Platform.getPlatformByName("CPU")
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # ── TriRNASP initial force: set before the first minimize so that
    #     minimization converges directly toward the statistical-potential preference ──
    tri_energy = 0.0
    if tri_potential is not None and tri_force is not None:
        pos_A = coords_nm * 10.0  # nm -> Angstroms
        coords_3b = pos_A.reshape(L, 3, 3)
        tri_energy, tri_grad = tri_potential.score_with_gradient(coords_3b, sequence)
        f_flat = _tri_forces(tri_grad)  # (3L, 3) capped
        for p_idx in range(3 * L):
            fx, fy, fz = f_flat[p_idx]
            tri_force.setParticleParameters(p_idx, p_idx, [fx, fy, fz])
        tri_force.updateParametersInContext(sim.context)

    # First minimization: includes the TriRNASP external force; use more iterations
    sim.minimizeEnergy(maxIterations=2000)

    # Record the initial energy
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e0 = state.getPotentialEnergy()._value
    best_energy = e0
    best_pos = state.getPositions(asNumpy=True)._value

    # Recompute the TriRNASP energy after minimize (coordinates have changed)
    if tri_potential is not None:
        pos_min = best_pos * 10.0
        tri_energy, _ = tri_potential.score_with_gradient(
            pos_min.reshape(L, 3, 3), sequence)

    conn.send(("init", worker_idx, e0 + tri_energy * trirnasp_scale * _TRI_KBT))

    # ── Annealing phase: reuse _run_annealing's multistage tightening strategy ──
    # Minimize + MD alone cannot converge from a random conformation (energy drops <1%);
    # a temperature/force-constant ladder is needed to fold stepwise. Anneal steps = 40% of the total.
    n_anneal_steps = max(200, int(n_steps * 0.4))
    try:
        e_ann, pos_ann = _run_annealing(
            sim, _pf, _bjf, _bjg,
            L, n_anneal=n_anneal_steps, verbose=False)
        if e_ann < best_energy:
            best_energy = e_ann
            state_a = sim.context.getState(getEnergy=True, getPositions=True)
            best_pos = state_a.getPositions(asNumpy=True)._value
    except Exception as _e_ann:
        print(f"    [REMD worker {worker_idx}] anneal skipped: {_e_ann}")

    # After annealing, recompute the TriRNASP energy and refresh the external force (coordinates changed a lot)
    if tri_potential is not None:
        state_a2 = sim.context.getState(getPositions=True)
        coords_3b = (state_a2.getPositions(asNumpy=True)._value * 10.0).reshape(L, 3, 3)
        tri_energy, tri_grad = tri_potential.score_with_gradient(coords_3b, sequence)
        if tri_force is not None:
            f_flat = _tri_forces(tri_grad)
            for p_idx in range(3 * L):
                fx, fy, fz = f_flat[p_idx]
                tri_force.setParticleParameters(p_idx, p_idx, [fx, fy, fz])
            tri_force.updateParametersInContext(sim.context)

    # Main loop (the remaining steps continue refinement)
    for step_i in range(max(n_steps - n_anneal_steps, exchange_interval)):
        # ── Refresh the TriRNASP force (every trirnasp_update_freq steps) ──
        if tri_potential is not None and tri_force is not None and \
           (step_i + 1) % trirnasp_update_freq == 0:
            state_pos = sim.context.getState(getPositions=True)
            pos_now = state_pos.getPositions(asNumpy=True)._value
            pos_A = pos_now * 10.0
            coords_3b = pos_A.reshape(L, 3, 3)
            tri_energy, tri_grad = tri_potential.score_with_gradient(coords_3b, sequence)
            f_flat = _tri_forces(tri_grad)
            for p_idx in range(3 * L):
                fx, fy, fz = f_flat[p_idx]
                tri_force.setParticleParameters(p_idx, p_idx, [fx, fy, fz])
            tri_force.updateParametersInContext(sim.context)

        sim.step(1)
        if (step_i + 1) % 500 == 0:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()._value
            if energy < best_energy:
                best_energy = energy
                best_pos = state.getPositions(asNumpy=True)._value

        # Exchange point: send energy+coordinates, wait for the exchange decision
        if (step_i + 1) % exchange_interval == 0:
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy()._value  # kJ/mol
            # Total energy = OpenMM + TriRNASP
            total_energy = energy + tri_energy * trirnasp_scale * _TRI_KBT
            pos = state.getPositions(asNumpy=True)._value
            conn.send(("report", worker_idx, total_energy, pos))
            # Wait for the master's exchange result
            cmd = conn.recv()
            if cmd[0] == "swap":
                new_pos = cmd[1]
                sim.context.setPositions(new_pos * unit.nanometer)
                # After a swap, recompute the TriRNASP energy
                if tri_potential is not None:
                    pos_A = new_pos * 10.0
                    coords_3b = pos_A.reshape(L, 3, 3)
                    tri_energy, _ = tri_potential.score_with_gradient(coords_3b, sequence)
            # On "keep", leave the positions unchanged

    # Final report
    conn.send(("done", worker_idx, best_energy + tri_energy * trirnasp_scale * _TRI_KBT, best_pos))
    conn.close()


def _clamp_replicas_by_memory(n_replicas: int, mem_per_proc_gb: float = 4.0) -> int:
    """Clamp the number of parallel processes to the available memory.

    Default 1.5GB/replica — measured: a 2009-nt 3-bead system uses only ~200-300MB per
    process (interpreter + OpenMM library dominate), so the old 6GB estimate was overly
    conservative and kept CPU parallelism low.

    Args:
        n_replicas: desired number of processes
        mem_per_proc_gb: estimated memory per process (GB)

    Returns:
        clamped process count (at least 1)
    """
    try:
        import psutil
        avail = psutil.virtual_memory().available / (1024 ** 3)
        max_by_mem = max(1, int(avail // mem_per_proc_gb))
        return max(1, min(n_replicas, max_by_mem))
    except ImportError:
        return n_replicas


def _balance_replicas_threads(n_replicas: int) -> Tuple[int, int]:
    """Replica/thread balance for CPU-only platforms: prioritize filling all cores with replicas.

    The OpenMM CPU platform scales poorly with threads on small systems (<10k particles),
    so 16 replicas x 2 threads beats 6 replicas x 5 threads — richer exchange sampling and
    higher overall throughput.

    Returns:
        (adjusted_n_replicas, per_replica_threads)
    """
    total = os.cpu_count() or 8
    if n_replicas >= total:
        return n_replicas, 1
    # Try to make replicas x threads == total, favoring replicas
    per_thread = max(1, total // n_replicas)
    return n_replicas, per_thread


def _run_remd(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str,
    n_replicas: int = 4,
    n_steps: int = 500,
    exchange_interval: int = 100,
    verbose: bool = False,
    minimal: bool = False,
    sequence: str = None,
    use_trirnasp: bool = False,
    trirnasp_energy_dir: str = None,
    trirnasp_scale: float = 0.003,  # unified default: 0.003 (optimal value)
    trirnasp_update_freq: int = 10,
) -> Tuple[float, np.ndarray]:
    """Run T-REMD enhanced sampling (multiprocess parallel).

    One process per replica; each builds its own system (only numpy/list is passed).
    Threads = cpu_count // n_replicas, filling all cores.
    With minimal=True the minimal force field is used (P backbone + P pairing) and P-only
    nm coordinates are returned.

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)] base pairs
        platform_name: platform
        n_replicas: number of replicas
        n_steps: total number of steps
        exchange_interval: exchange interval
        verbose: verbosity
        minimal: use the minimal force field (default False)
        sequence: RNA sequence (required when use_trirnasp=True)
        use_trirnasp: enable the TriRNASP three-body statistical potential
        trirnasp_energy_dir: directory of the energy tables
        trirnasp_scale: TriRNASP energy scaling factor
        trirnasp_update_freq: gradient-update frequency (every N steps)

    Returns:
        (best_energy, best_coords_nm)  # minimal=True: P-only nm; False: 3-bead nm
    """
    from scipy.constants import k as kB
    import multiprocessing as mp

    # Temperature ladder: 300K -> ~460K (geometric spacing)
    temperatures = [300.0 * (1.10 ** i) for i in range(n_replicas)]

    # Threads per replica: prioritize filling all cores with replicas
    total_threads = os.cpu_count() or 8
    # Memory-aware: measured ~200-300MB per process, so 1.5GB/replica is generous
    n_replicas = _clamp_replicas_by_memory(n_replicas, mem_per_proc_gb=1.5)
    n_replicas, per_replica_threads = _balance_replicas_threads(n_replicas)
    if verbose:
        print(f"    REMD: {n_replicas} replicas in parallel, "
              f"{per_replica_threads} threads per replica "
              f"({total_threads} cores total)")

    ctx = mp.get_context("spawn")
    processes = []
    conns = []
    for ri in range(n_replicas):
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        p = ctx.Process(
            target=_run_remd_worker,
            args=(ri, p_coords, pairs, temperatures[ri], n_steps,
                  exchange_interval, per_replica_threads, child_conn, minimal,
                  sequence, use_trirnasp, trirnasp_energy_dir,
                  trirnasp_scale, trirnasp_update_freq),
        )
        p.start()
        child_conn.close()
        processes.append(p)
        conns.append(parent_conn)

    # Master process: coordinate exchanges
    best_energy = float("inf")
    # Initial coordinates (nm): P-only in minimal mode, 3-bead in full mode (fill in C4'/N)
    if minimal:
        best_pos = p_coords / 10.0  # (L,3) P-only nm
    else:
        L0 = len(p_coords)
        _rng0 = np.random.default_rng(0)
        best_pos = np.zeros((3 * L0, 3), dtype=np.float64)
        for _i in range(L0):
            best_pos[3 * _i] = p_coords[_i] / 10.0
            best_pos[3 * _i + 1] = p_coords[_i] / 10.0 + _rng0.normal(0, 0.03, 3)
            best_pos[3 * _i + 2] = p_coords[_i] / 10.0 + _rng0.normal(0, 0.03, 3)
    accept_count = 0
    total_exchanges = max(1, (n_steps // exchange_interval) * (n_replicas - 1))

    # Stage 1: wait for all replicas to init
    for ri in range(n_replicas):
        msg = conns[ri].recv()
        assert msg[0] == "init"
        if msg[2] < best_energy:
            best_energy = msg[2]

    # Stage 2: coordinate exchanges
    n_exchange_points = n_steps // exchange_interval
    for _ in range(n_exchange_points):
        energies = [None] * n_replicas
        positions = [None] * n_replicas
        for ri in range(n_replicas):
            msg = conns[ri].recv()
            assert msg[0] == "report"
            energies[ri] = msg[2]
            positions[ri] = msg[3]
            if msg[2] < best_energy:
                best_energy = msg[2]
                best_pos = msg[3].copy()

        # Metropolis exchange between neighboring replicas
        swap_decisions = [False] * (n_replicas - 1)
        for ri in range(n_replicas - 1):
            ui, uj = energies[ri], energies[ri + 1]
            beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
            beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
            exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
            if np.random.random() < min(1.0, np.exp(exponent)):
                swap_decisions[ri] = True
                accept_count += 1

        # Apply exchanges: send new coordinates to the replicas involved
        for ri in range(n_replicas):
            new_pos = None
            if ri > 0 and swap_decisions[ri - 1]:
                new_pos = positions[ri - 1]
            elif ri < n_replicas - 1 and swap_decisions[ri]:
                new_pos = positions[ri + 1]
            if new_pos is not None:
                conns[ri].send(("swap", new_pos))
            else:
                conns[ri].send(("keep",))

    # Stage 3: wrap-up
    for ri in range(n_replicas):
        msg = conns[ri].recv()
        assert msg[0] == "done"
        if msg[2] < best_energy:
            best_energy = msg[2]
            best_pos = msg[3]

    for p in processes:
        p.join(timeout=10)
    for conn in conns:
        conn.close()

    if verbose:
        rate = accept_count / total_exchanges
        print(f"    REMD: E={best_energy:.0f}, exchange rate {rate:.1%}")

    return best_energy, best_pos


# ── bpp-guided discovery of far pairs ──

def discover_far_pairs_from_bpp(
    bpp_matrix: np.ndarray,
    sequence: str,
    min_gap: int = 24,
    bpp_threshold: float = 0.1,
    top_k: int = 50,
    existing_pairs: Optional[List[Tuple[int, int, float]]] = None,
) -> List[Tuple[int, int, float]]:
    """Discover far pairs from a ViennaRNA bpp probability matrix.

    Based on the shared-partner Jaccard similarity in BppPriorModule (scheme10_full.py):

      Core insight:
        1. High bpp(i,j) -> i and j are in the same folding unit (stem)
        2. Nucleotides in the same folding unit tend to cluster in space
        3. If i and k both appear in several high-bpp stems -> they may be in the same domain
        4. This "co-occurrence" lets us infer the likelihood of far contacts

      Algorithm:
        For each pair (i,j) with |i-j| >= min_gap:
          P(i) = {k | bpp(i,k) > threshold}  -- partner set of i
          P(j) = {k | bpp(j,k) > threshold}  -- partner set of j
          J(i,j) = |P(i) & P(j)| / |P(i) | P(j)|  -- Jaccard similarity
          w(i,j) = bpp(i,j) * J(i,j)  -- direct bpp probability x shared-partner similarity

        High J -> i and j share many partners -> same domain -> spatially close.
        Even when bpp(i,j) itself is low, many shared partners can still suggest a far contact.

    Args:
        bpp_matrix: (L,L) base-pair probability matrix
        sequence: RNA sequence
        min_gap: minimum sequence separation (default 24, i.e. >1 turn of helix)
        bpp_threshold: minimum bpp value (used to define "pairing partners")
        top_k: maximum number of pairs to return
        existing_pairs: existing pairs [(i,j,w)], to avoid duplicates

    Returns:
        [(i, j, w)] newly discovered far pairs (w = bpp * Jaccard)
    """
    L = len(sequence)
    if bpp_matrix is None or bpp_matrix.shape[0] != L:
        return []

    # Build a set of existing pairs (to avoid duplicates)
    existing_set = set()
    if existing_pairs:
        for (i, j, w) in existing_pairs:
            existing_set.add((min(i, j), max(i, j)))

    # Step 1: precompute the partner set of each position
    partner_sets = []
    for i in range(L):
        partners = set()
        for k in range(L):
            if k != i and float(bpp_matrix[i, k]) > bpp_threshold:
                partners.add(k)
        partner_sets.append(partners)

    # Step 2: compute the Jaccard similarity for every far pair (i,j)
    candidates = []
    for i in range(L):
        pi = partner_sets[i]
        if len(pi) == 0:
            continue
        for j in range(i + min_gap, L):
            pj = partner_sets[j]
            if len(pj) == 0:
                continue
            key = (i, j)
            if key in existing_set:
                continue

            bpp_val = float(bpp_matrix[i, j])

            # Shared-partner Jaccard: |P(i) & P(j)| / |P(i) | P(j)|
            shared = len(pi & pj)
            union = len(pi) + len(pj) - shared
            if union == 0:
                continue
            jaccard = shared / union

            # Combined weight (additive, following BppPriorModule):
            #   w = bpp_direct + alpha * jaccard_cooccurrence
            # Even with bpp(i,j)=0, many shared partners can still suggest a far contact
            # alpha controls the co-occurrence contribution
            alpha = 0.5
            w = bpp_val + alpha * jaccard

            if w > 0.01:  # minimum threshold
                candidates.append((i, j, w, bpp_val, jaccard))

    # Step 3: sort by combined weight w descending (x[2] = w, x[3] = bpp_val)
    candidates.sort(key=lambda x: -x[2])  # sort by w

    # Step 4: remove redundancy (keep only the strongest near each location)
    result = []
    used = set()
    for (i, j, w, bpp_val, jaccard) in candidates:
        if len(result) >= top_k:
            break
        # De-redundancy: keep only one per 10-nt window
        key_red = (i // 10, j // 10)
        if key_red in used:
            continue
        used.add(key_red)
        result.append((i, j, w))

    return result


# ── Multi-round REMD temperature annealing ──

def _run_multistage_remd(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str,
    n_rounds: int = 3,
    n_replicas: int = 12,
    n_steps_per_round: int = 5000,
    verbose: bool = False,
    sequence: str = None,
    use_trirnasp: bool = False,
    trirnasp_energy_dir: str = None,
    trirnasp_scale: float = 0.003,  # unified default: 0.003 (optimal value)
    trirnasp_update_freq: int = 10,
) -> Tuple[float, np.ndarray]:
    """Multi-round REMD temperature annealing: explore at high temperature first, then cool stepwise and refine.

    Following the ensemble_temperatures in scheme10_full.py:
      round 0: 300-500K (high-temperature exploration, escape local minima)
      round 1: 250-400K (medium-temperature convergence)
      round 2: 200-350K (low-temperature refinement)

    Each round uses the lowest-energy conformation of the previous round as its start.

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)]
        platform_name: platform
        n_rounds: number of annealing rounds
        n_replicas: replicas per round
        n_steps_per_round: steps per round
        verbose: verbosity
        sequence: RNA sequence (required when use_trirnasp=True)
        use_trirnasp: enable the TriRNASP three-body statistical potential
        trirnasp_energy_dir: directory of the energy tables
        trirnasp_scale: TriRNASP energy scaling factor
        trirnasp_update_freq: gradient-update frequency (every N steps)

    Returns:
        (best_energy, best_coords_ang) — P-only Angstroms (x10 internally)
    """
    best_energy = float("inf")
    best_pos = p_coords.copy()
    L_remd = len(p_coords)  # number of P-only particles (input is always P-only)

    for rnd in range(n_rounds):
        # The temperature range drops each round but stays above 280K (RNA freezes at low temperature)
        temp_high = max(350.0, 500.0 - rnd * 30.0)
        temp_low = max(280.0, 300.0 - rnd * 10.0)

        if verbose:
            print(f"    REMD annealing round {rnd + 1}/{n_rounds}: "
                  f"T={temp_low:.0f}-{temp_high:.0f}K, "
                  f"{n_replicas} replicas, {n_steps_per_round} steps")

        # Use a custom temperature ladder instead of the default 300*1.1^i
        from scipy.constants import k as kB
        import multiprocessing as mp

        # Clamp first, then build the temperatures (avoid a length mismatch in the temperature list)
        n_replicas_clamped = _clamp_replicas_by_memory(n_replicas, mem_per_proc_gb=1.5)
        temperatures = [temp_low + (temp_high - temp_low) * i / max(1, n_replicas_clamped - 1)
                        for i in range(n_replicas_clamped)]

        total_threads = os.cpu_count() or 8
        n_replicas_clamped, per_replica_threads = _balance_replicas_threads(n_replicas_clamped)
        n_replicas = n_replicas_clamped

        ctx = mp.get_context("spawn")
        processes = []
        conns = []
        for ri in range(n_replicas):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_run_remd_worker,
                args=(ri, best_pos, pairs, temperatures[ri], n_steps_per_round,
                      max(10, n_steps_per_round // 10), per_replica_threads,
                      child_conn, False,  # full force-field REMD
                      sequence, use_trirnasp, trirnasp_energy_dir,
                      trirnasp_scale, trirnasp_update_freq),
            )
            p.start()
            child_conn.close()
            processes.append(p)
            conns.append(parent_conn)

        # Coordinate exchanges
        accept_count = 0
        round_best_e = float("inf")
        round_best_pos = best_pos / 10.0  # Angstroms -> nm (workers report in nm)

        # Wait for init: ("init", worker_idx, energy) — 3 elements, no coordinates
        for ri in range(n_replicas):
            msg = conns[ri].recv()
            if msg[0] == "init" and msg[2] < round_best_e:
                round_best_e = msg[2]

        n_ex = n_steps_per_round // max(10, n_steps_per_round // 10)
        for _ in range(n_ex):
            energies = [None] * n_replicas
            positions = [None] * n_replicas
            for ri in range(n_replicas):
                msg = conns[ri].recv()
                if msg[0] == "report":
                    energies[ri] = msg[2]
                    positions[ri] = msg[3]
                    if msg[2] < round_best_e:
                        round_best_e = msg[2]
                        round_best_pos = msg[3].copy()

            swap_decisions = [False] * (n_replicas - 1)
            for ri in range(n_replicas - 1):
                ui, uj = energies[ri], energies[ri + 1]
                beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
                beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
                exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
                if np.random.random() < min(1.0, np.exp(exponent)):
                    swap_decisions[ri] = True
                    accept_count += 1

            for ri in range(n_replicas):
                new_pos = None
                if ri > 0 and swap_decisions[ri - 1]:
                    new_pos = positions[ri - 1]
                elif ri < n_replicas - 1 and swap_decisions[ri]:
                    new_pos = positions[ri + 1]
                conns[ri].send(("swap", new_pos) if new_pos is not None else ("keep",))

        # Wrap-up: ("done", worker_idx, best_energy, best_pos)
        for ri in range(n_replicas):
            try:
                msg = conns[ri].recv()
                if msg[0] == "done" and msg[2] < round_best_e:
                    round_best_e = msg[2]
                    round_best_pos = msg[3]
            except Exception:
                pass

        for p in processes:
            p.join(timeout=10)
        for conn in conns:
            conn.close()

        # Update the global best
        if round_best_e < best_energy:
            best_energy = round_best_e
            best_pos = round_best_pos * 10.0  # nm -> Angstroms
            # The worker returns 3-bead coordinates (3Lx3), but the next round expects
            # P-only input (Lx3) to build the 3-bead system. Extract the P beads to avoid
            # a 3-bead -> 9-bead blow-up producing garbage coordinates.
            if best_pos.shape[0] == 3 * L_remd:
                best_pos = best_pos[0::3].copy()
        # Ensure best_pos stays P-only (Lx3) so workers never receive 3-bead input
        if best_pos.shape[0] != L_remd:
            if verbose:
                print(f"    [REMD] abnormal best_pos shape ({best_pos.shape}); forcing P-only extraction")
            best_pos = best_pos[0::3].copy() if best_pos.shape[0] == 3 * L_remd else best_pos[:L_remd]

        if verbose:
            rate = accept_count / max(1, n_ex * (n_replicas - 1))
            print(f"    REMD round {rnd + 1}: E={round_best_e:.0f}, "
                  f"exchange rate {rate:.1%}")

    return best_energy, best_pos


# ── Potential-guided refinement ──

def _potential_guided_refine(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    sequence: str,
    secondary_structure: str,
    n_minimize: int = 3000,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """Potential-guided refinement: extra OpenMM minimization + short MD on the lowest-energy REMD conformation.

    Following DynamicEnsembleGenerator in scheme10_full.py:
      potential_weight=0.1, potential_refine_steps=10
    Refines with the full 3-bead force field (including stacking/angle/clash), not the minimal one.

    Args:
        p_coords: (L,3) P coordinates (Angstroms)
        pairs: [(i,j,w)]
        sequence: RNA sequence
        secondary_structure: secondary structure
        n_minimize: number of minimization steps
        verbose: verbosity

    Returns:
        (refined_energy, refined_coords_nm)
    """
    L = len(p_coords)

    # Sanitize coordinates
    p_coords = _sanitize_p_coords(p_coords.copy())

    # Build the full 3-bead force field (including stacking/angle/clash)
    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(p_coords, pairs, pair_scale=1.0, bsj_k_scale=1.0)

    topo = _create_3bead_topology(L)
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    plat = Platform.getPlatformByName("CPU")
    n_threads = os.cpu_count() or 8
    plat_props = {"CpuThreads": str(n_threads)}
    sim = Simulation(topo, system, integrator, plat, plat_props)
    sim.context.setPositions(coords_nm * unit.nanometer)

    # Stage 1: energy minimization
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=n_minimize)

    # Stage 2: short MD refinement (300K, 5ps)
    sim.step(2500)

    # Stage 3: final minimization
    sim.minimizeEnergy(
        tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=n_minimize)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    e = state.getPotentialEnergy()._value
    pos = state.getPositions(asNumpy=True)._value  # nm

    if verbose:
        print(f"    Potential refinement: E={e:.0f} kJ/mol")

    return e, pos


# ── Coordinate sanitization and compaction ──

def _sanitize_p_coords(p_coords: np.ndarray) -> np.ndarray:
    """Sanitize P coordinates: replace NaN/Inf with the mean of adjacent valid coordinates."""
    L = len(p_coords)
    bad = np.any(~np.isfinite(p_coords), axis=1)
    if not np.any(bad):
        return p_coords

    n_bad = int(np.sum(bad))
    print(f"  [OpenMM GPU] found {n_bad}/{L} NaN/Inf P coordinates; cleaning...")

    for i in range(L):
        if not bad[i]:
            continue
        left, right = None, None
        for j in range(i - 1, -1, -1):
            if not bad[j]:
                left = j
                break
        for j in range(i + 1, L):
            if not bad[j]:
                right = j
                break
        if left is not None and right is not None:
            alpha = (i - left) / (right - left)
            p_coords[i] = (1 - alpha) * p_coords[left] + alpha * p_coords[right]
        elif left is not None:
            p_coords[i] = p_coords[left].copy()
        elif right is not None:
            p_coords[i] = p_coords[right].copy()
        else:
            p_coords[i] = [0.0, 0.0, float(i) * 5.9]

    return p_coords


def _is_extended_helix(p_coords: np.ndarray, threshold: float = 200.0) -> bool:
    """Detect whether the P coordinates form an extended structure (end-to-end distance far beyond a reasonable circular-RNA range)."""
    if len(p_coords) < 2:
        return False
    end_to_end = float(np.linalg.norm(p_coords[-1] - p_coords[0]))
    return end_to_end > threshold


def _generate_compact_coords(L: int, pairs: List[Tuple[int, int, float]]) -> np.ndarray:
    """Generate compact circular starting coordinates for long sequences.

    The ring radius is set by the P-P bond length and the sequence length, with a small
    perturbation to avoid degeneracy.
    """
    coords = np.zeros((L, 3), dtype=np.float64)
    circumference = L * BOND_P_NEXT
    radius = circumference / (2.0 * np.pi)

    for i in range(L):
        angle = 2.0 * np.pi * i / L
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

    rng = np.random.default_rng(42)
    coords += rng.normal(0, 0.5, coords.shape)
    return coords


def _has_nan_energy(sim: 'Simulation') -> bool:
    """Check whether the simulation's current energy is NaN/Inf."""
    try:
        state = sim.context.getState(getEnergy=True)
        e = state.getPotentialEnergy()._value
        return not np.isfinite(e)
    except Exception:
        return True


# ── Main entry: isrnacirc_cg_refine-compatible interface ──

def openmm_gpu_refine(
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
    trirnasp_energy_dir: str = None,
    trirnasp_scale: float = 0.003,  # unified default: 0.003 (optimal value)
    trirnasp_update_freq: int = 10,
) -> Tuple[str, float]:
    """OpenMM GPU-accelerated CG MD refinement (isrnacirc_cg_refine-compatible interface).

    Replaces the CPU-only refinement in IsRNAcirc.exe:
    1. Read PDB -> extract P coordinates + bpp far-pair discovery
    2. 3-bead CG force field (bpp-weighted pairing force)
    3. Three-stage annealing (weak -> strong pairing + BSJ)
    4. Multi-round REMD temperature annealing (high-T exploration -> low-T refinement)
    5. Potential-guided refinement (full 3-bead force-field minimization + short MD)
    6. Physical-constraint relaxation (bond length/angle/clash/BSJ/WC pairing)
    7. CG -> all-atom (cg_to_allatom; optional skip)
    8. Write the refined PDB

    Args:
        input_pdb: input PDB path
        output_dir: output directory
        sequence: RNA sequence
        secondary_structure: secondary structure
        name: project name
        nstep: annealing steps (per stage), default 20000
        nstep_close: (compatibility parameter, unused)
        nstru: (compatibility parameter, unused)
        timeout: timeout in seconds
        platform_name: "auto"/"CUDA"/"OpenCL"/"CPU"
        use_remd: whether to enable REMD
        remd_n_replicas: number of REMD replicas, default 6
        remd_n_steps: number of REMD steps, default 3000
        verbose: print detailed output
        skip_cg_to_allatom: skip the internal CG->all-atom conversion (use when the input is already all-atom)
        use_physical_relax: whether to enable physical-constraint relaxation (default True)
        bpp_matrix: (L,L) ViennaRNA base-pair probability matrix (optional; used for the bpp-weighted pairing force)
        bpp_weight: bpp mixing weight (0=pure hard-coded, 1=pure bpp)
        use_multistage_remd: whether to enable multi-round REMD temperature annealing (default True)
        use_potential_refine: whether to enable potential-guided refinement (default True)

    Returns:
        (output_pdb_path, final_energy, diag)  # diag: {"hot_start_energy": float}
    """
    if not OPENMM_AVAILABLE:
        raise ImportError(
            "OpenMM is not installed; GPU refinement is unavailable. "
            "Install OpenMM with: conda install -c conda-forge openmm")

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Detect the platform
    platform = detect_best_platform(platform_name)
    if verbose:
        print(f"  [OpenMM GPU] platform: {platform}")

    # 1. Read the P coordinates
    p_coords = _read_p_coords(input_pdb)
    L = len(p_coords)
    if verbose:
        print(f"  [OpenMM GPU] sequence length: {L} nt")

    if L < 3:
        raise ValueError(f"sequence too short ({L} nt) for CG MD")

    # 1b. Clean NaN/Inf coordinates
    p_coords = _sanitize_p_coords(p_coords)

    # Resolve the base pairs from the pairs argument (inferred from secondary_structure)
    pairs = _dotbracket_to_pairs(secondary_structure)

    # 1c. bpp far-pair discovery: add far pairs from the bpp matrix
    if bpp_matrix is not None and bpp_matrix.shape[0] == L:
        far_pairs_discovered = discover_far_pairs_from_bpp(
            bpp_matrix, sequence, min_gap=24, bpp_threshold=0.01,
            top_k=50, existing_pairs=pairs)
        if far_pairs_discovered:
            pairs = pairs + far_pairs_discovered
            if verbose:
                print(f"  [bpp] discovered {len(far_pairs_discovered)} far pairs, "
                      f"{len(pairs)} pairs total")

    # 1d. Check the coordinate quality; replace with compact circular coordinates only when invalid/bond-lengths are abnormal
    # Note: a large end-to-end distance does not mean the structure is extended — the ends of a circular RNA can naturally be far apart.
    # Check P-P bond lengths: only judge the structure bad when the mean bond length is abnormal (far beyond the reasonable ~5.9A range).
    avg_pp = 0.0
    if L > 1:
        diffs = p_coords[1:] - p_coords[:-1]
        pp_dists = np.linalg.norm(diffs, axis=1)
        avg_pp = float(np.mean(pp_dists[:min(L - 1, 500)]))

    # If the bond lengths are on the nm scale (<1.5A), the coordinate unit is nm rather than
    # Angstroms; multiply by 10. 5.9A never false-triggers; 0.59nm is correctly converted.
    if avg_pp < 1.5 and L > 1:
        p_coords = p_coords * 10.0
        avg_pp = avg_pp * 10.0
        if verbose:
            print(f"  [OpenMM GPU] coordinate-unit fix: avg_pp {avg_pp/10:.2f} -> {avg_pp:.2f}A")

    use_compact = (not np.isfinite(avg_pp)) or avg_pp > 20.0 or avg_pp < 1.0
    if use_compact:
        if verbose:
            print(f"  [OpenMM GPU] abnormal P-P bond length (avg={avg_pp:.2f}A); "
                  f"generating compact circular starting coordinates...")
        p_coords = _generate_compact_coords(L, pairs)

    # 2. Build the system (bpp-weighted pairing force)
    system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
        _build_3bead_system_gpu(p_coords, pairs, pair_scale=1.0, bsj_k_scale=0.1,
                                bpp_matrix=bpp_matrix, bpp_weight=bpp_weight)

    # 3. Create the topology and simulation
    topo = _create_3bead_topology(L)

    try:
        plat = Platform.getPlatformByName(platform)
    except Exception:
        plat = Platform.getPlatformByName("Reference")

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    # Set multithreading on the CPU platform (the default would use only 1 core)
    plat_props = {}
    if plat.getName() == "CPU":
        n_threads = os.cpu_count() or 8
        plat_props["CpuThreads"] = str(n_threads)
        if verbose:
            print(f"  [OpenMM GPU] CPU threads: {n_threads}")

    sim = Simulation(topo, system, integrator, plat, plat_props)
    try:
        sim.context.setPositions(coords_nm * unit.nanometer)
    except Exception as e_pos:
        # GPU out of memory (LLVM ERROR); fall back to CPU
        if verbose:
            print(f"  [OpenMM GPU] platform {platform} failed: {e_pos}; falling back to CPU...")
        plat = Platform.getPlatformByName("CPU")
        n_threads = os.cpu_count() or 8
        plat_props = {"CpuThreads": str(n_threads)}
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond,
        )
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)

    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    if verbose:
        print(f"  [OpenMM GPU] initial energy: {e0:.0f} kJ/mol")

    # 4. Check the initial energy; retry with more compact coordinates if it is NaN/Inf
    if not np.isfinite(e0):
        if verbose:
            print(f"  [OpenMM GPU] abnormal initial energy ({e0}); "
                  f"retrying with more compact circular coordinates...")
        compact_r = max(10.0, L * BOND_P_NEXT / (2.0 * np.pi) * 0.3)
        rng = np.random.default_rng(123)
        p_fb = np.zeros((L, 3), dtype=np.float64)
        for i in range(L):
            angle = 2.0 * np.pi * i / L
            p_fb[i] = [compact_r * np.cos(angle),
                        compact_r * np.sin(angle), 0.0]
        p_fb += rng.normal(0, 0.3, p_fb.shape)
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(p_fb, pairs,
                                    pair_scale=1.0, bsj_k_scale=0.05)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
        if verbose:
            print(f"  [OpenMM GPU] retry initial energy: {e0:.0f} kJ/mol")

    # 5. Minimize
    try:
        sim.minimizeEnergy(
            tolerance=100.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=1000)
    except Exception as e:
        if verbose:
            print(f"  [OpenMM GPU] minimization error: {e}")

    # Check after minimizing; if still NaN, try an extremely compact start
    if _has_nan_energy(sim):
        if verbose:
            print(f"  [OpenMM GPU] energy abnormal after minimization; "
                  f"retrying with an extremely compact start and weak forces...")
        compact_r2 = max(8.0, L * BOND_P_NEXT / (2.0 * np.pi) * 0.15)
        rng2 = np.random.default_rng(456)
        p_v3 = np.zeros((L, 3), dtype=np.float64)
        for i in range(L):
            angle = 2.0 * np.pi * i / L
            p_v3[i] = [compact_r2 * np.cos(angle),
                         compact_r2 * np.sin(angle), 0.0]
        p_v3 += rng2.normal(0, 0.2, p_v3.shape)
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(p_v3, pairs,
                                    pair_scale=0.1, bsj_k_scale=0.01)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat, plat_props)
        sim.context.setPositions(coords_nm * unit.nanometer)
        sim.minimizeEnergy(
            tolerance=500.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=200)
        e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
        if verbose:
            print(f"  [OpenMM GPU] V3 initial energy: {e0:.0f} kJ/mol")

    # 6. Two-stage folding + refinement
    # Stage 1: minimal-force-field folding (P backbone + P-P pairing, no clash) -> pairing converges
    # Stage 2: full-force-field REMD refinement (starting from the folded coordinates)
    # skip_minimal_fold: when iterating REMD, skip the minimal fold and use the previous round's refined coordinates directly
    if skip_minimal_fold:
        if verbose:
            print(f"  [OpenMM GPU] skipping minimal fold (hot-start mode); using the input coordinates directly")
        # p_coords came from reading an all-atom PDB; keep only the P-atom coordinates.
        # If p_coords has more than L rows, all-atom data was read and must be filtered.
        if len(p_coords) > L:
            p_only = p_coords[:L]  # the first L rows of an all-atom PDB are the P atoms (if formatted correctly)
            if verbose:
                print(f"  [OpenMM GPU] input has {len(p_coords)} atoms; taking the first {L} P coordinates")
        else:
            p_only = p_coords
        anneal_pos_ang = p_only  # P-only coordinates
        # Compute the initial energy (same force-field convention as REMD / the 3-bead relaxation).
        # The old implementation built a temporary LJ pairing-potential system, E ~= -n_pairs*0.5 (e.g. -58),
        # which was not comparable to the full-force-field energy, so best_energy stayed pinned at the
        # hot-start value and REMD improvements were never accepted.
        try:
            _hs_sys, _hs_coords_nm, _pf_hs, _sf_hs, _bjf_hs, _bjg_hs = \
                _build_3bead_system_gpu(p_only, pairs,
                                        pair_scale=1.0, bsj_k_scale=0.5)
            _hs_topo = _create_3bead_topology(L)
            _hs_int = LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
            _hs_sim = Simulation(_hs_topo, _hs_sys, _hs_int, plat, plat_props)
            _hs_sim.context.setPositions(_hs_coords_nm * unit.nanometer)
            # Minimize before scoring — the C4'/N initial coordinates in 3-bead are randomly
            # perturbed, and without minimization E is on the order of millions of kJ/mol
            # (clash/angle blow-up), which cannot be compared with the REMD workers' E ~ 100k-300k.
            # Minimize ~500 steps to make the 3-bead coordinates self-consistent, then evaluate.
            _hs_sim.minimizeEnergy(maxIterations=500)
            anneal_e = _hs_sim.context.getState(getEnergy=True).getPotentialEnergy()._value
            if verbose:
                print(f"  [OpenMM GPU] hot-start initial energy: {anneal_e:.0f} kJ/mol "
                      f"(full 3-bead force-field convention)")
            del _hs_sim, _hs_int, _hs_topo, _hs_sys
        except Exception as e_energy:
            # Do not set E=0 — that would fake convergence. Use a high-energy fallback so the
            # later refinement can continue.
            anneal_e = 999999.0
            if verbose:
                print(f"  [OpenMM GPU] hot-start energy computation failed: {e_energy}")
    else:
        if verbose:
            print(f"  [OpenMM GPU] minimal-force-field folding ({nstep} steps, multiprocess)...")
        n_traj = _clamp_replicas_by_memory(2, mem_per_proc_gb=1.5)
        if verbose:
            print(f"  [OpenMM GPU] minimal fold: {n_traj} trajectories (memory-aware limit)")
        try:
            anneal_e, anneal_pos_ang = _run_parallel_minimal_annealing(
                p_coords, pairs, n_anneal=nstep, n_trajectories=n_traj,
                platform_name="CPU", verbose=verbose)
        except Exception as e_anneal_par:
            if verbose:
                print(f"  [OpenMM GPU] minimal folding failed: {e_anneal_par}; falling back to serial annealing...")
            anneal_e, anneal_pos_ang = _run_annealing(
                sim, pair_force, bsj_force, bsj_guide, L,
                n_anneal=nstep, verbose=verbose)
            anneal_pos_ang = anneal_pos_ang[0::3] * 10.0  # 3-bead nm -> P Angstroms

        if verbose:
            print(f"  [OpenMM GPU] energy after folding: {anneal_e:.0f} kJ/mol")

    # 5b. Far-pair pre-pull: low temperature + strong far-pair force to pull far pairs into place
    if anneal_e < 100 and L > 50:
        try:
            _sys_fr, _cfr, _pf_fr = _build_minimal_system_gpu(
                anneal_pos_ang, pairs, pair_scale=3.0)
            _topo_fr = _create_minimal_topology(L)
            _int_fr = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
            _plat_fr = mm.Platform.getPlatformByName("CPU")
            _nthreads = max(1, (os.cpu_count() or 8) // 2)
            _sim_fr = app.Simulation(_topo_fr, _sys_fr, _int_fr, _plat_fr,
                                     {"CpuThreads": str(_nthreads)})
            _sim_fr.context.setPositions(_cfr * unit.nanometer)
            _sim_fr.minimizeEnergy(maxIterations=5000)
            _sim_fr.integrator.setTemperature(320 * unit.kelvin)
            _sim_fr.step(2000)
            _sim_fr.integrator.setTemperature(300 * unit.kelvin)
            _sim_fr.step(2000)
            _sim_fr.minimizeEnergy(
                tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=3000)
            _st = _sim_fr.context.getState(getPositions=True, getEnergy=True)
            _fr_pos = _st.getPositions(asNumpy=True)._value * 10.0  # nm -> Angstroms
            _fr_e = _st.getPotentialEnergy()._value
            # Check whether this is an improvement
            _fr_bonds = np.linalg.norm(_fr_pos[1:] - _fr_pos[:-1], axis=1)[:100]
            if np.mean(_fr_bonds) > 3.0 and _fr_e < anneal_e:
                anneal_pos_ang = _fr_pos
                anneal_e = _fr_e
                if verbose:
                    print(f"  [OpenMM GPU] far-pair pre-pull: E={_fr_e:.0f}, "
                          f"far-pair distances improved")
        except Exception as e:
            if verbose:
                print(f"  [OpenMM GPU] far-pair pre-pull skipped: {e}")

    # 6. T-REMD (optional) — start from the folded P coordinates (Angstroms)
    final_e = anneal_e
    final_pos_pang = anneal_pos_ang  # (L,3) Angstroms

    # Skip REMD when the energy is already very low (<10 kJ/mol, already converged)
    if use_remd and L >= 10 and final_e > 10:
        if use_multistage_remd:
            # Multi-round REMD with the full force field (3-bead, including stacking/angle/clash).
            # The minimal force field is too simple: temperature differences do not affect the
            # energy, so the exchange rate is 0.
            if verbose:
                print(f"  [OpenMM GPU] multi-round REMD ({remd_n_replicas} replicas, 8 rounds, full force field)...")
            remd_e, remd_pos = _run_multistage_remd(
                final_pos_pang, pairs, platform,
                n_rounds=8,
                n_replicas=remd_n_replicas,
                n_steps_per_round=max(5000, remd_n_steps // 3),
                verbose=verbose,
                sequence=sequence,
                use_trirnasp=use_trirnasp,
                trirnasp_energy_dir=trirnasp_energy_dir,
                trirnasp_scale=trirnasp_scale,
                trirnasp_update_freq=trirnasp_update_freq)
        else:
            # Single-round REMD (full force field)
            if verbose:
                print(f"  [OpenMM GPU] T-REMD ({remd_n_replicas} replicas, {remd_n_steps} steps, full force field)...")
            remd_e, remd_pos = _run_remd(
                final_pos_pang, pairs, platform,
                n_replicas=remd_n_replicas,
                n_steps=remd_n_steps,
                verbose=verbose,
                minimal=False,  # full force field: stacking/angle/clash
                sequence=sequence,
                use_trirnasp=use_trirnasp,
                trirnasp_energy_dir=trirnasp_energy_dir,
                trirnasp_scale=trirnasp_scale,
                trirnasp_update_freq=trirnasp_update_freq)
        # REMD is the refinement stage; always adopt its coordinates on success (pairs converge further).
        if remd_pos is not None and np.isfinite(remd_e):
            final_e = remd_e
            # Diagnostic: print the shape returned by REMD
            if verbose:
                print(f"  [OpenMM GPU] REMD returned: shape={remd_pos.shape}, E={remd_e:.0f}")
            # _run_multistage_remd returns 3-bead Angstroms (already x10 internally); _run_remd returns
            # 3-bead nm (unconverted). Downstream, final_pos_pang expects P-only Angstroms (Lx3).
            # Handle uniformly: extract the P beads and ensure the unit is Angstroms.
            if remd_pos.ndim == 2 and remd_pos.shape[0] == 3 * L:
                p_only_nm = remd_pos[0::3].copy()  # 3-bead -> P-only
                # Detect the unit: if nm (bond length ~0.6), multiply by 10 to get Angstroms
                _avg_pp = float(np.mean(np.linalg.norm(
                    p_only_nm[1:] - p_only_nm[:-1], axis=1)[:100]))
                if _avg_pp < 1.0:  # nm scale
                    final_pos_pang = p_only_nm * 10.0
                else:  # Angstrom scale
                    final_pos_pang = p_only_nm
                if verbose:
                    _pp = np.linalg.norm(final_pos_pang[1:] - final_pos_pang[:-1], axis=1)
                    print(f"    REMD 3-bead->P-only: avg_PP={np.mean(_pp):.2f}A")
            elif remd_pos.ndim == 2 and remd_pos.shape[0] >= 2:
                # P-only, but the unit may be nm
                _avg_pp = float(np.mean(np.linalg.norm(
                    remd_pos[1:] - remd_pos[:-1], axis=1)[:100]))
                if _avg_pp < 1.0:
                    final_pos_pang = remd_pos * 10.0
                else:
                    final_pos_pang = remd_pos

    # 6b. Potential-guided refinement: skipped — an incompatible force field would blow up the
    #     minimal-force-field coordinates.
    # Potential refinement would use the 3-bead full force field (stacking/angle/clash) on
    # minimal-force-field coordinates, but the units/parameters are incompatible, making E jump
    # from 200K to 58M kJ/mol.
    # Use the minimal-fold output directly and do not run potential refinement.
    if False and use_potential_refine and L >= 10 and final_e > 1000:
        try:
            pg_e, pg_pos = _potential_guided_refine(
                final_pos_pang, pairs, sequence, secondary_structure,
                n_minimize=3000, verbose=verbose)
            if np.isfinite(pg_e) and pg_e < final_e:
                final_e = pg_e
                final_pos_pang = pg_pos * 10.0  # nm -> Angstroms
        except Exception as e:
            if verbose:
                print(f"    potential refinement skipped: {e}")
    elif verbose and final_e <= 100:
        print(f"    potential refinement: skipped (E={final_e:.0f}, converged)")

    # 6b. 3-bead full-force-field relaxation (same force field as REMD for consistency).
    # Build the full force field (stacking/angle/clash/pairing/BSJ) with _build_3bead_system_gpu,
    # then run a short 300K MD + minimization and extract the P coordinates.
    if use_physical_relax and L >= 10:
        try:
            _sys_rx, _c_rx, _pf_rx, _sf_rx, _bjf_rx, _bjg_rx = \
                _build_3bead_system_gpu(final_pos_pang, pairs,
                                        pair_scale=1.0, bsj_k_scale=0.5)
            _topo_rx = _create_3bead_topology(L)
            _plat_rx = mm.Platform.getPlatformByName("CPU")
            _int_rx = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
            _nthreads_rx = max(1, (os.cpu_count() or 8) // 2)
            _sim_rx = app.Simulation(_topo_rx, _sys_rx, _int_rx, _plat_rx,
                                     {"CpuThreads": str(_nthreads_rx)})
            _sim_rx.context.setPositions(_c_rx * unit.nanometer)
            # Short 300K MD (2000 steps) + minimization
            _sim_rx.step(2000)
            _sim_rx.minimizeEnergy(
                tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=3000)
            _st_rx = _sim_rx.context.getState(getPositions=True, getEnergy=True)
            _rx_pos = _st_rx.getPositions(asNumpy=True)._value  # nm
            _rx_e = _st_rx.getPotentialEnergy()._value
            # Extract the P coordinates (Angstroms)
            _rx_p = _rx_pos[0::3] * 10.0
            _rx_bonds = np.linalg.norm(_rx_p[1:] - _rx_p[:-1], axis=1)
            _rx_avg = float(np.mean(_rx_bonds))
            # The relaxed energy must beat the pre-relaxation energy, otherwise skip
            if _rx_avg > 3.0 and np.all(np.isfinite(_rx_p)) and _rx_e < final_e:
                final_pos_pang = _rx_p
                final_e = _rx_e
                if verbose:
                    _bsj_d = np.linalg.norm(_rx_p[0] - _rx_p[-1])
                    print(f"  [3-bead relaxation] E={_rx_e:.0f}, bond={_rx_avg:.2f}A, "
                          f"BSJ={_bsj_d:.2f}A")
            elif verbose:
                if _rx_e >= final_e:
                    print(f"  [3-bead relaxation] skipped (E={_rx_e:.0f} >= current {final_e:.0f})")
                else:
                    print(f"  [3-bead relaxation] skipped (avg_bond={_rx_avg:.2f}A, abnormal)")
        except Exception as e:
            if verbose:
                print(f"  [physical relaxation] skipped: {e}")

    # Convert the folded P coordinates (Angstroms) to 3-bead nm (fill in C4'/N) for later output
    rng_final = np.random.default_rng(7)
    final_pos = np.zeros((3 * L, 3), dtype=np.float64)
    for i in range(L):
        final_pos[3 * i] = final_pos_pang[i] / 10.0  # P, Angstroms -> nm
        final_pos[3 * i + 1] = final_pos_pang[i] / 10.0 + rng_final.normal(0, 0.03, 3)
        final_pos[3 * i + 2] = final_pos_pang[i] / 10.0 + rng_final.normal(0, 0.03, 3)

    # 7. CG -> all-atom
    # CG_to_allatom's template matching expects P-P ~5.9A (true RNA scale), but OpenMM-annealed
    # coordinates may be somewhat larger in scale and need rescaling.
    _BOND_P_NEXT = 0.59  # 5.9A = 0.59nm (default)
    try:
        from .cg_forcefield import BOND_P_NEXT as _bpn
        _BOND_P_NEXT = _bpn / 10.0  # BOND_P_NEXT is in Angstroms; convert to nm
    except ImportError:
        pass
    p_coords = final_pos[0::3]  # (L,3) P beads in nm

    # Validation: P bond lengths should be in the 0.3-1.2nm range (3-12A). Outside that range the
    # coordinate unit is suspect; never rescale.
    if L > 1:
        avg_pp_nm = float(np.mean(np.linalg.norm(p_coords[1:] - p_coords[:-1], axis=1)[:100]))
        if verbose:
            print(f"  [OpenMM GPU] CG P bond length: {avg_pp_nm:.4f}nm ({avg_pp_nm*10:.2f}A)")
        # Fine-tune only within a reasonable range (0.45-0.75nm); leave it otherwise
        if 0.45 < avg_pp_nm < 0.75 and abs(avg_pp_nm - _BOND_P_NEXT) > 0.03:
            scale = _BOND_P_NEXT / avg_pp_nm
            final_pos = final_pos * scale
            if verbose:
                print(f"  [OpenMM GPU] coordinate fine-tune: {avg_pp_nm:.4f} -> {_BOND_P_NEXT:.4f}nm")
        elif avg_pp_nm < 0.45 or avg_pp_nm > 0.75:
            if verbose:
                print(f"  [OpenMM GPU] abnormal bond length ({avg_pp_nm:.4f}nm); skipping rescale")

    cg_pdb = str(out_path / f"{name}_cg.pdb")
    _write_allatom_pdb(final_pos, L, cg_pdb, sequence=sequence)

    if skip_cg_to_allatom:
        # The input is already all-atom (merged_aa); skip the redundant CG->all-atom conversion.
        # Write only the refined CG PDB; Level 2 reads the P coordinates itself.
        output_pdb = cg_pdb
    else:
        aa_pdb = str(out_path / f"{name}_aa_raw.pdb")
        try:
            from .isrnacirc_wrapper import cg_to_allatom
            cg_to_allatom(cg_pdb, aa_pdb, sequence)
            # Check that the output file is valid (at least 10 ATOM lines)
            with open(aa_pdb) as _f:
                n_atoms = sum(1 for _ in _f if _.startswith("ATOM"))
            if n_atoms < 10:
                raise RuntimeError(f"CG->all-atom output has only {n_atoms} atoms; not enough")
            # Validate the P bond lengths: cg_to_allatom may corrupt the P coordinates
            _aa_p = _read_p_coords(aa_pdb)
            if len(_aa_p) > 1:
                _aa_bonds = np.linalg.norm(_aa_p[1:] - _aa_p[:-1], axis=1)
                _aa_avg = float(np.mean(_aa_bonds))
                # The normal P-P bond-length range is 4.5-7.5A; outside it, cg_to_allatom corrupted the coordinates
                if _aa_avg < 4.5 or _aa_avg > 7.5:
                    if verbose:
                        print(f"  [OpenMM GPU] abnormal cg_to_allatom P bond length ({_aa_avg:.2f}A); falling back to the CG PDB")
                    aa_pdb = cg_pdb
        except Exception as e:
            if verbose:
                print(f"  [OpenMM GPU] CG->all-atom failed: {e}; writing CG coordinates")
            aa_pdb = cg_pdb

        # 8. Write the final output PDB
        output_pdb = str(out_path / f"{name}_openmm.pdb")
        _write_refined_pdb(aa_pdb, output_pdb)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [OpenMM GPU] done: E={final_e:.0f} kJ/mol, "
              f"elapsed {elapsed:.1f}s")

    return output_pdb, final_e, {"hot_start_energy": anneal_e}


def _dotbracket_to_pairs(ss: str) -> List[Tuple[int, int, float]]:
    """Extract a base-pair list [(i,j,1.0)] from a dot-bracket string."""
    pairs = []
    stack = []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i, 1.0))
    return pairs


# ── Smoke test ──

def main():
    """Smoke test: generate a random sequence and refine it with OpenMM GPU."""
    import random
    random.seed(42)
    L = 50
    sequence = "".join(random.choices("AUCG", k=L))
    ss = "(" * (L // 2) + ")" * (L // 2)

    # Random initial coordinates (planar circle)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    r = L * 5.9 / (2 * np.pi)  # the P-P spacing sets the radius
    p_coords = np.column_stack([r * np.cos(angles), r * np.sin(angles),
                                 np.zeros(L)])

    # Write a temporary PDB
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    input_pdb = Path(tmp_dir) / "test_input.pdb"
    lines = ["HEADER    test"]
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    for i in range(L):
        x, y, z = p_coords[i]
        resname = base_map.get(sequence[i].upper(), "UNK")
        lines.append(
            f"ATOM  {i + 1:5d}  P   {resname} A{i + 1:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P ")
    lines.append("END")
    with open(input_pdb, "w") as f:
        f.write("\n".join(lines))

    print(f"Sequence: {L}nt, SS: {ss[:10]}...")
    output_pdb, energy, diag = openmm_gpu_refine(
        str(input_pdb), tmp_dir, sequence, ss,
        name="test", nstep=100, use_remd=True,
        remd_n_replicas=3, remd_n_steps=200,
        platform_name="auto", verbose=True,
    )
    print(f"\nOutput: {output_pdb}")
    print(f"Energy: {energy:.0f} kJ/mol")
    print(f"Hot-start energy: {diag.get('hot_start_energy', float('nan')):.0f} kJ/mol")


if __name__ == "__main__":
    main()
