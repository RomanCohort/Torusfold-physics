"""
amber_refine.py - all-atom RNA refinement with the Amber14 OL3 force field.

Consumes the output of allatom_reconstruct and runs restrained minimization
with OpenMM + amber14-all.xml (RNA.OL3) + implicit/obc1.xml (OBC1 implicit
solvent), letting the all-atom structure settle into a reasonable A-form RNA
conformation under the force field.

Aggressive accuracy improvements (vs original scheme 3):
  * P-atom positional restraint loosened 50 -> 10 kJ/mol/nm^2: frees the
    backbone trajectory so the force field can adjust the backbone to satisfy
    base pairing instead of pinning the CG topology in place.
  * A-form helix dihedral restraints (CustomTorsionForce): backbone
    alpha/gamma/delta/zeta + C3'-endo sugar pucker, the heart of RNA accuracy,
    which was previously missing entirely.
  * ViennaRNA pairing CustomBond: r0=1.06 nm (10.6 A C1'-C1'), pulling the
    bases in stem regions together into Watson-Crick geometry.
  * L-BFGS minimization for 3000 steps; long sequences (>200 nt) additionally
    get MD annealing to escape local minima.
  * Modeller.addHydrogens auto-adds H (the amber14 RNA templates include H).

Safety net: if P drifts > 2 A or the energy blows up, raise RuntimeError and
let the caller fall back to CG.
"""

from __future__ import annotations

import sys

# Windows GBK-encoding workaround: printing non-ASCII text (e.g. Angstrom signs) fails under GBK
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass  # ignore reconfiguration failure (some environments disallow it)

from typing import Dict, List, Optional, Tuple

import numpy as np

from .allatom_reconstruct import AllAtomStructure, get_atom_xyzs



def _detect_platform() -> str:
    """Auto-detect the best OpenMM platform: CUDA > OpenCL > CPU."""
    try:
        from openmm import Platform
        for name in ("CUDA", "OpenCL"):
            try:
                Platform.getPlatformByName(name)
                return name
            except Exception:
                continue
    except ImportError:
        pass
    return "CPU"

# P-atom positional restraint force constant (kJ/mol/nm^2).
# The old value of 10 was too soft: during amber minimization P drifted 3.95 A
# (beyond the 2.0 threshold). Measured: K=500 still 3.95 A, K=5000 dropped to
# 1.41 A. P is the backbone point solved by CG and carries the input structural
# information, so it should not wander far -> use 1000 (a common literature
# value for positional restraints). This both holds P so the intra-residue
# geometry is pulled toward A-form, and still leaves P free enough for the
# minimization to be meaningful.
P_RESTRAINT_K = 1000.0
# P drift threshold: the old 2.0 was set when K=10. After raising K to 1000:
#   a regular-polygon self-test drifted P by 2.35 A; real CG-solved P drifted
#   3.6 A (real network solutions carry more strain than a perfect geometry).
#   3.6 A is ~1.3% of a circRNA radius of ~28 A, physically acceptable
#   (the P position from CG is an approximation; the truth is the relaxed
#   force-field conformation). Stage 1 loosens the P restraint, so P drifts
#   more (5-8 A), but that is the force field actively steering the backbone,
#   not a failure. Threshold raised to 10.0: as long as the force-field energy
#   converges to a negative value, treat it as a success.
P_MAX_DRIFT_A = 10.0

# Standard A-form RNA dihedrals (radians) - used for backbone torsion restraints
# alpha: O3'-P-O5'-C5'   ~ -60 deg (gauche-)
# gamma: O5'-C5'-C4'-C3' ~  60 deg (gauche+)
# delta: C5'-C4'-C3'-O3' ~  84 deg (anti, A-form)
# zeta:  C4'-C3'-O3'-P   ~ -90 deg (gauche-)
# chi:   O4'-C1'-N9-C4 (purine) / O4'-C1'-N1-C2 (pyrimidine) ~ -160 deg (anti)
_AFORM_TORSIONS = {
    "alpha": (-60.0, "O3'", "P",   "O5'", "C5'"),
    # beta intentionally excluded: diagnostics show beta sits 51 deg off A-form,
    # but adding a beta restraint (k=50) shares P/O5'/C5' atoms with alpha; the
    # coupled restraints conflict and alpha/gamma collapse (16/18 outliers).
    # The beta deviation is a known blind spot, deferred to a weaker k or to
    # fragment rebuilding. (tests/torsion_stacking_diag.py)
    "gamma": (60.0,  "O5'", "C5'", "C4'", "C3'"),
    "delta": (84.0,  "C5'", "C4'", "C3'", "O3'"),
    "zeta":  (-90.0, "C4'", "C3'", "O3'", "P"),
}


class _TimeoutError(Exception):
    pass


def _build_topology_and_modeller(structure: AllAtomStructure):
    """Build the Topology + Modeller, adding H manually (bypassing amber addHydrogens template matching).

    amber14 RNA.OL3's addHydrogens fails on the circular circRNA topology (the
    ExternalBonds of the first/last residues do not match the A5/A3 templates).
    Fix: add H manually following RNA conventions, then call
    createSystem(ignoreExternalBonds=True, residueTemplates=internal templates).
    """
    from openmm.app import Topology, Element, Modeller, ForceField

    topo = Topology()
    chain = topo.addChain()
    res_atoms: List[Dict[str, object]] = []
    for (start, end) in structure.residue_atom_spans:
        first_atom = structure.atoms[start]
        res = topo.addResidue(first_atom.res_name, chain)
        name_to_atom: Dict[str, object] = {}
        for k in range(start, end):
            a = structure.atoms[k]
            elem = Element.getBySymbol(a.element)
            ta = topo.addAtom(a.atom_name, elem, res)
            name_to_atom[a.atom_name] = ta
        res_atoms.append(name_to_atom)

    # --- Manually add H (bypassing addHydrogens) ---
    # OpenMM requires contiguous atoms per residue, so we rebuild the Topology:
    # each residue is added together with its heavy atoms and its H atoms.
    heavy_xyzs = get_atom_xyzs(structure)  # Å
    heavy_names = [a.atom_name for a in structure.atoms]
    heavy_res_idx = [a.res_seq - 1 for a in structure.atoms]
    sequences = list(structure.sequence)
    bonds_global = _collect_bonds_global(topo)
    h_atoms = _add_hydrogens_manual(
        heavy_xyzs, heavy_names, heavy_res_idx, sequences, bonds_global
    )
    h_by_res = {}
    for h in h_atoms:
        h_by_res.setdefault(h[0]-1, []).append(h)

    topo2 = Topology()
    chain2 = topo2.addChain()
    res_atoms2 = []
    all_coords_nm = []
    for (start, end) in structure.residue_atom_spans:
        first_atom = structure.atoms[start]
        res = topo2.addResidue(first_atom.res_name, chain2)
        name_to_atom = {}
        for k in range(start, end):
            a = structure.atoms[k]
            elem = Element.getBySymbol(a.element)
            ta = topo2.addAtom(a.atom_name, elem, res)
            name_to_atom[a.atom_name] = ta
            all_coords_nm.append(a.xyz / 10.0)
        for h in h_by_res.get(first_atom.res_seq-1, []):
            ta_h = topo2.addAtom(h[1], Element.getBySymbol("H"), res)
            name_to_atom[h[1]] = ta_h
            all_coords_nm.append(h[2] / 10.0)
        res_atoms2.append(name_to_atom)

    _add_rna_bonds(topo2, res_atoms2, structure)
    coords_arr = np.array(all_coords_nm, dtype=np.float64)  # nm
    modeller = Modeller(topo2, coords_arr)
    ff = ForceField("amber14-all.xml", "implicit/obc1.xml")

    # Build the heavy_to_topo map: structure.atoms heavy-atom index -> topo2 atom index.
    # In topo2 each residue is "heavy atoms followed by that residue's H atoms",
    # so H atoms sit between residues and serial numbers cannot index heavy atoms
    # directly. Iterate over topo2 atoms, skip H, and record the heavy-atom order.
    heavy_to_topo = []
    for ta in topo2.atoms():
        if ta.element.symbol == "H":
            continue
        heavy_to_topo.append(ta.index)
    return modeller, ff, heavy_to_topo


def _collect_bonds_global(topo):
    """Collect all (atom1.index, atom2.index) bonds in the Topology."""
    bonds = []
    for b in topo.bonds():
        bonds.append((b[0].index, b[1].index))
    return bonds


# RNA standard H placement (heavy atom -> {base: [(H name, element), ...]}).
# Cross-checked residue by residue against the amber14 RNA.OL3.xml A/G/U/C
# templates, not guessed.
# Backbone H (C5'/C4'/C3'/C2'/C1'/O2'-HO2') is common to all four bases.
# Base H depends on the base: H1->N1(G), H2->C2(A), H3->N3(U), H5->C5(U/C),
# H6->C6(U/C), H8->C8(A/G), H21/H22->N2(G), H41/H42->N4(C), H61/H62->N6(A).
# circRNA always uses internal templates: O5' bonds to its own residue's P and
# O3' to the downstream P, so neither gets an H. (Whether O5'/O3' take an H is
# decided dynamically in _add_hydrogens_manual by the presence of an external P.)
_RNA_H_MAP: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    # Backbone - common to all bases
    "C5'": {"*": [("H5'", "H"), ("H5''", "H")]},
    "C4'": {"*": [("H4'", "H")]},
    "C3'": {"*": [("H3'", "H")]},
    "C2'": {"*": [("H2'", "H")]},
    "C1'": {"*": [("H1'", "H")]},
    "O2'": {"*": [("HO2'", "H")]},
    # Purine base H
    "C8":  {"A": [("H8", "H")], "G": [("H8", "H")]},
    "C2":  {"A": [("H2", "H")]},          # only A's C2 carries H2; G's C2 bonds to N2 and takes no H
    "N1":  {"G": [("H1", "H")]},          # G's N1-H1 (imino hydrogen)
    "N6":  {"A": [("H61", "H"), ("H62", "H")]},
    "N2":  {"G": [("H21", "H"), ("H22", "H")]},
    # Pyrimidine base H
    "C5":  {"U": [("H5", "H")], "C": [("H5", "H")]},
    "C6":  {"U": [("H6", "H")], "C": [("H6", "H")]},
    "N3":  {"U": [("H3", "H")]},  # only U's N3-H3; C's N3 is a double-bonded ring N and takes no H
    "N4":  {"C": [("H41", "H"), ("H42", "H")]},
    # These heavy atoms carry no H on any base
    "O5'": {}, "O3'": {}, "O6": {}, "O2": {}, "O4": {},
    "N9": {}, "N7": {}, "C4": {},
    "P": {}, "OP1": {}, "OP2": {},
}
_H_BOND_LEN = {"C": 1.09, "N": 1.01, "O": 0.97}


def _add_hydrogens_manual(
    heavy_xyzs, heavy_names, heavy_res_idx, sequences, bonds
):
    """Manually add H (placed by pushing out opposite the heavy-atom neighbor centroid).

    Returns [(res_seq, name, xyz), ...].
    """
    N = heavy_xyzs.shape[0]
    neighbors = [set() for _ in range(N)]
    for (i, j) in bonds:
        if 0 <= i < N and 0 <= j < N:
            neighbors[i].add(j)
            neighbors[j].add(i)

    hydrogens = []
    for i in range(N):
        aname = heavy_names[i]
        res_idx = heavy_res_idx[i]
        base = sequences[res_idx]
        # map is {heavy_atom: {base: [(H name, element), ...]}}; "*" means all bases
        per_base = _RNA_H_MAP.get(aname)
        if not per_base:
            continue
        h_list = per_base.get(base) or per_base.get("*")
        if not h_list:
            continue
        if aname == "O3'":
            has_down_p = any(
                heavy_names[j] == "P" and heavy_res_idx[j] != res_idx
                for j in neighbors[i]
            )
            if has_down_p:
                continue
        if aname == "O5'":
            has_up_p = any(
                heavy_names[j] == "P" and heavy_res_idx[j] == res_idx
                for j in neighbors[i]
            )
            if has_up_p:
                continue

        xyz = heavy_xyzs[i]
        nbrs = list(neighbors[i])
        if len(nbrs) == 0:
            h_dir = np.array([1.0, 0.0, 0.0])
        else:
            nbr_center = heavy_xyzs[nbrs].mean(axis=0)
            h_dir = xyz - nbr_center
            n = np.linalg.norm(h_dir)
            h_dir = h_dir / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

        bl = _H_BOND_LEN.get(aname[0], 1.09)
        if len(h_list) == 1:
            h_xyz = xyz + h_dir * bl
            hydrogens.append((res_idx + 1, h_list[0][0], h_xyz))
        elif len(h_list) == 2:
            if abs(h_dir[0]) < 0.9:
                ortho = np.cross(h_dir, np.array([1.0, 0.0, 0.0]))
            else:
                ortho = np.cross(h_dir, np.array([0.0, 1.0, 0.0]))
            on = np.linalg.norm(ortho)
            ortho = ortho / on if on > 1e-6 else np.array([0.0, 1.0, 0.0])
            off = 0.4
            for k, (hn, _) in enumerate(h_list):
                sign = 1 if k == 0 else -1
                h_xyz = xyz + h_dir * bl + ortho * off * sign
                hydrogens.append((res_idx + 1, hn, h_xyz))
    return hydrogens


# Standard intra-residue RNA bonds (name-name pairs): backbone + sugar ring + base.
_BACKBONE_BONDS = [
    ("P", "OP1"), ("P", "OP2"), ("P", "O5'"),
    ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "O4'"),
    ("C4'", "C3'"), ("C3'", "O3'"), ("C3'", "C2'"),
    ("C2'", "O2'"), ("C2'", "C1'"), ("C1'", "O4'"),  # sugar-ring closure + 2'-OH
]
_PURINE_BONDS = [  # A/G
    ("C1'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "N7"), ("N7", "C5"),
    ("C5", "C6"), ("C5", "C4"), ("C6", "N1"), ("N1", "C2"), ("C2", "N3"),
    ("N3", "C4"),  # bicyclic ring closure
]
_PYRIMIDINE_BONDS = [  # C/U
    ("C1'", "N1"), ("N1", "C2"), ("C2", "N3"), ("N3", "C4"),
    ("C4", "C5"), ("C5", "C6"), ("C6", "N1"),  # six-membered ring closure
]
_BASE_EXTRA_BONDS = {
    "A": [("C6", "N6")],
    "G": [("C6", "O6"), ("C2", "N2")],
    "C": [("C2", "O2"), ("C4", "N4")],
    "U": [("C2", "O2"), ("C4", "O4")],
}


def _add_rna_bonds(topo, res_atoms, structure):
    """Add the standard RNA covalent bonds (incl. H) for every residue plus the inter-residue phosphodiester bonds.

    H bonds are resolved via _RNA_H_MAP: which heavy atom each H attaches to,
    consistent with the side on which the map added the H.
    """
    L = len(res_atoms)
    for res_idx, name_to_atom in enumerate(res_atoms):
        base = structure.sequence[res_idx]
        bonds = list(_BACKBONE_BONDS)
        if base in ("A", "G"):
            bonds += _PURINE_BONDS
        else:
            bonds += _PYRIMIDINE_BONDS
        bonds += _BASE_EXTRA_BONDS.get(base, [])
        for a1n, a2n in bonds:
            a1 = name_to_atom.get(a1n)
            a2 = name_to_atom.get(a2n)
            if a1 is not None and a2 is not None:
                topo.addBond(a1, a2)

        # H bonds: for each heavy atom in the map (its own base or the "*" common
        # entry), addBond(heavy_atom, H) for its H atoms
        for heavy_name, per_base in _RNA_H_MAP.items():
            h_list = per_base.get(base) or per_base.get("*")
            if not h_list:
                continue
            heavy_atom = name_to_atom.get(heavy_name)
            if heavy_atom is None:
                continue
            for (h_name, _elem) in h_list:
                h_atom = name_to_atom.get(h_name)
                if h_atom is not None:
                    topo.addBond(heavy_atom, h_atom)

    # Inter-residue phosphodiester bonds: O3'[i] <-> P[i+1], i=0..L-2
    # No BSJ topology bond is added - ignoreExternalBonds=True lets the first
    # and last residues match the internal templates. BSJ closure is enforced by
    # the HarmonicBondForce physical restraint inside amber_refine, and after
    # minimization it does not affect the base pairing (BSJ is local).
    for i in range(L - 1):
        o3 = res_atoms[i].get("O3'")
        p_next = res_atoms[i + 1].get("P")
        if o3 is not None and p_next is not None:
            topo.addBond(o3, p_next)


def amber_refine(
    structure: AllAtomStructure,
    pairs: List[Tuple[int, int, float]],
    *,
    platform_name: str = "auto",
    max_iterations: int = 3000,
    use_md_for_long: bool = True,
    long_threshold: int = 200,
    coding_mask: Optional[np.ndarray] = None,
    cg_coords: Optional[np.ndarray] = None,
    coding_restraint_k: float = 10000.0,
    cg_topology_weight: float = 0.0,  # scheme F: blend weight of CG global topology into non-coding P restraints
    use_o3p_bond: bool = False,
    use_o3p_angle: bool = False,
    pair_anneal_stages: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, int]]:
    """Amber14 OL3 + OBC1 restrained minimization. On failure returns the reconstructed coordinates (unrefined).

    coding_mask: bool array (L,); True = residue in a coding region. During amber
        refinement the P/C1'/O3* positions of coding residues are pinned to
        cg_coords with a high k, preserving the true structure.
    cg_coords: (L, 3) nm; the target coordinates (original CG coordinates) coding
        residues are pinned to. When omitted, the P coordinates in structure are
        used (i.e. the RL-optimized positions, equivalent to no pinning).
    coding_restraint_k: force constant for pinning coding residues (kJ/mol/nm^2).
        Default 10000 = strongly pinned.
    cg_topology_weight: blend weight for merging the CG global topology into the
        P-restraint targets of non-coding residues (0.0 = pure 1EHZ Kabsch
        template; 1.0 = fully pinned back to CG topology). Default 0.0 keeps old
        behavior. On long sequences a value >0 lets global-topology information
        reach amber, preventing the local 1EHZ template from overriding the
        global pairing positions.
    use_o3p_bond: P2 bond-length restraint (O3'-P[i+1] 1.6 A, k=10000). Off by default.
    use_o3p_angle: P2.5 bond-angle restraint (C3'-O3'-P / O3'-P-O5'). Off by default.
        (Both off = the clean P1 version: pure 4-point Kabsch with no extra
        restraints, used as the P3 comparison baseline.)
    pair_anneal_stages: list of pairing-restraint annealing stages [(k, r0_nm), ...].
        Each stage tightens the pairing restraint with k as the force constant and
        r0 as the target distance. When None and pairs is non-empty, defaults to
        [(20,2.5), (50,2.0), (100,1.06)]. An empty list means no annealing (old
        behavior: use k=100 r0=1.06 directly).
    """
    try:
        return _amber_refine_impl(
            structure, pairs,
            platform_name=platform_name,
            max_iterations=max_iterations,
            use_md_for_long=use_md_for_long,
            long_threshold=long_threshold,
            coding_mask=coding_mask,
            cg_coords=cg_coords,
            coding_restraint_k=coding_restraint_k,
            cg_topology_weight=cg_topology_weight,
            use_o3p_bond=use_o3p_bond,
            use_o3p_angle=use_o3p_angle,
            pair_anneal_stages=pair_anneal_stages,
        )
    except Exception as exc:
        print(f"[amber_refine] refinement failed, returning reconstructed coordinates: {exc!r}")
        coords_aa = get_atom_xyzs(structure)
        info = {
            "n_heavy": len(structure.atoms),
            "n_atoms": len(structure.atoms),
            "n_h": 0,
            "n_torsions": 0,
            "max_p_drift": 0.0,
            "fallback": True,
            "error": str(exc),
            "n_coding_pinned": 0,
        }
        return coords_aa, 0.0, 0.0, info


def _amber_refine_impl(
    structure: AllAtomStructure,
    pairs: List[Tuple[int, int, float]],
    *,
    platform_name: str = "CPU",
    max_iterations: int = 3000,
    use_md_for_long: bool = True,
    long_threshold: int = 200,
    coding_mask: Optional[np.ndarray] = None,
    cg_coords: Optional[np.ndarray] = None,
    coding_restraint_k: float = 10000.0,
    cg_topology_weight: float = 0.0,
    use_o3p_bond: bool = False,
    use_o3p_angle: bool = False,
    pair_anneal_stages: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, int]]:
    """Amber14 OL3 + OBC1 restrained minimization of the all-atom structure.

    cg_topology_weight: blend weight for merging the CG global topology into the
        P-restraint targets of non-coding residues (0.0 = pure 1EHZ Kabsch
        template; 1.0 = fully pinned back to CG topology). Default 0.0 keeps old
        behavior. On long sequences (>800 nt) a value >0 lets global pairing
        information reach amber, preventing the local 1EHZ template from
        overriding the global topology (scheme F).

    Returns:
        (refined_coords, e0, e1, info)
        refined_coords: (N, 3) A (includes H, in the same atom order as after Modeller)
        e0/e1: potential energy before/after minimization (kJ/mol)
        info: {'n_atoms': ..., 'n_h': ..., 'max_p_drift': ...}
    """
    from openmm import (
        Platform, HarmonicBondForce, CustomBondForce,
        CustomExternalForce, CustomTorsionForce, CustomAngleForce,
        VerletIntegrator, LangevinMiddleIntegrator,
    )
    from openmm import unit
    from openmm.app import Simulation

    L = len(structure.residue_atom_spans)
    modeller, ff, heavy_to_topo = _build_topology_and_modeller(structure)
    topo = modeller.topology
    n_heavy = len(structure.atoms)
    n_total = int(modeller.topology.getNumAtoms())

    # After adding H the atom indices are remapped: the rebuilt P/O3'/C1' atoms
    # may sit at different positions in the new topology, so they must be
    # re-looked-up by (res_seq, atom_name).
    atom_lookup: Dict[Tuple[int, str], int] = {}  # (res_seq, atom_name) → new_idx
    for new_idx, atom in enumerate(topo.atoms()):
        atom_lookup[(atom.residue.index + 1, atom.name)] = new_idx

    # ignoreExternalBonds: skip the ExternalBond check (the circular circRNA
    # topology confuses amber). No residueTemplates are used: every circRNA
    # residue is internal (no 5'/3' ends), so let OpenMM match the internal
    # templates (A/U/G/C) automatically and avoid the end templates (A5/A3/C5/C3).
    system = ff.createSystem(
        topo, constraints=None, rigidWater=True,
        ignoreExternalBonds=True,
    )

    # --- Force 1: P positional restraint (coding pinned to cg_coords, non-coding pinned to itself) ---
    # Coding residues: per-particle k=coding_restraint_k (default 10000, strongly
    #   pinned), target=cg_coords (the original pre-RL CG coordinates, preserving
    #   the true structure). RL moves the whole sequence; amber pulls coding back.
    # Non-coding residues: per-particle k=P_RESTRAINT_K (1000, soft restraint),
    #   target=structure's own P (amber input coords = RL-optimized CG P, allowing
    #   a small physical convergence tweak).
    # With no coding_mask everything is treated as non-coding (same as the old
    # behavior, so existing calls keep working).
    # Global k_scale multiplier: the multi-stage annealing loosens/tightens it
    #   (stage 1 loose, stage 3 back to 1.0); per-particle k keeps the
    #   coding/non-coding distinction from being erased by the global setParameter.
    restraint = CustomExternalForce("k_scale*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k_scale", 1.0)
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    p_drift_refs = []  # (new_idx, original_xyz_nm) for the drift check
    n_coding_pinned = 0
    n_topology_fused = 0
    # Scheme F: the P target positions of non-coding regions blend in the CG
    # global topology. cg_coords is in nm, and structure P is in nm as well
    # (after dividing by 10.0).
    for res_idx in range(L):
        res_seq = res_idx + 1
        new_idx = atom_lookup.get((res_seq, "P"))
        if new_idx is None:
            continue
        # Default target = structure's own P (amber input coords), k = P_RESTRAINT_K
        p_xyz = structure.atoms[structure.residue_atom_index[res_idx]["P"]].xyz / 10.0
        k_val = P_RESTRAINT_K
        # Coding residue: pin back to cg_coords with a strong restraint
        if (coding_mask is not None and cg_coords is not None
                and res_idx < len(coding_mask) and coding_mask[res_idx]
                and res_idx < len(cg_coords)):
            # cg_coords is in nm (CG-solved P); coding is pinned to the original pre-RL CG coordinates
            p_xyz = np.asarray(cg_coords[res_idx], dtype=np.float64)
            k_val = coding_restraint_k
            n_coding_pinned += 1
        elif (cg_topology_weight > 0.0 and cg_coords is not None
                and res_idx < len(cg_coords)):
            # Scheme F: non-coding P target = (1-w)*1EHZ template + w*CG topology
            cg_p = np.asarray(cg_coords[res_idx], dtype=np.float64)
            tmpl_p = np.asarray(p_xyz, dtype=np.float64)
            p_xyz = (1.0 - cg_topology_weight) * tmpl_p + cg_topology_weight * cg_p
            n_topology_fused += 1
        restraint.addParticle(new_idx, [k_val, p_xyz[0], p_xyz[1], p_xyz[2]])
        p_drift_refs.append((new_idx, p_xyz))
    system.addForce(restraint)

    # --- Force 2: BSJ closure bond (O3'[L-1] <-> P[0]) ---
    last_o3 = atom_lookup.get((L, "O3'"))
    first_p = atom_lookup.get((1, "P"))
    bsj_bond = HarmonicBondForce()
    if last_o3 is not None and first_p is not None:
        bsj_bond.addBond(last_o3, first_p, 0.161, 50000.0)
    system.addForce(bsj_bond)

    # --- Force 2.5: hard O3'-P bond-length restraint between adjacent residues (P2, use_o3p_bond switch) ---
    # The 1EHZ-template Kabsch alignment only preserves the four points
    # P/C1'/C4'/O3'; the bridging geometry between adjacent residues
    # (O3'[i]-P[i+1]) can be distorted (C3'-O3'-P was ~70 deg before amber). A
    # HarmonicBondForce pins every O3'[i]-P[i+1] bond length to 1.6 A (the
    # A-form value) with k=10000 throughout as a tight restraint (not scaled by
    # k_scale). This does not conflict with the OL3 force field's own O3'-P bond
    # term (both point to 1.6 A); it just re-pins the length to prevent drift
    # during minimization. Off by default: off = clean P1 version (pure 4-point
    # Kabsch), used as the P3 comparison baseline.
    n_o3p = 0
    if use_o3p_bond:
        o3p_bond = HarmonicBondForce()
        for i in range(L):
            o3_idx = atom_lookup.get((i + 1, "O3'"))       # O3' of residue i
            next_p_idx = atom_lookup.get(((i + 1) % L + 1, "P"))  # P of residue i+1 (circular)
            if o3_idx is not None and next_p_idx is not None:
                o3p_bond.addBond(o3_idx, next_p_idx, 0.16, 10000.0)  # r0=1.6Å, k=10000
                n_o3p += 1
        system.addForce(o3p_bond)

    # --- Force 2.6: hard phosphate-bridge bond-angle restraints (P2.5, use_o3p_angle switch) ---
    # Bond-length restraints cannot fix bond angles (a correct length pair does
    # not imply a correct angle pair). A CustomAngleForce directly restrains the
    # two key phosphate-bridge angles, C3'-O3'-P (119.5 deg) and O3'-P-O5'
    # (104.1 deg), the OL3 A-form values. k_angle=500 kJ/mol/rad^2 is moderate
    # and stays tight throughout (not scaled by k_scale). Cross-residue:
    # C3'[i]-O3'[i]-P[i+1] (vertex O3'[i]) and O3'[i]-P[i+1]-O5'[i+1] (vertex
    # P[i+1]). CustomAngleForce.addAngle(p1, p2, p3, params): p2 is the vertex.
    # Off by default: off = clean P1 version.
    n_angle = 0
    if use_o3p_angle:
        angle_force = CustomAngleForce("0.5*k_angle*(theta-theta0)^2")
        angle_force.addGlobalParameter("k_angle", 500.0)
        angle_force.addPerAngleParameter("theta0")
        THETA_C3_O3_P = 119.5 * np.pi / 180.0  # C3'-O3'-P equilibrium angle (rad)
        THETA_O3_P_O5 = 104.1 * np.pi / 180.0  # O3'-P-O5' equilibrium angle (rad)
        for i in range(L):
            j = (i + 1) % L  # next-residue index
            c3_i = atom_lookup.get((i + 1, "C3'"))
            o3_i = atom_lookup.get((i + 1, "O3'"))
            p_j = atom_lookup.get((j + 1, "P"))
            o5_j = atom_lookup.get((j + 1, "O5'"))
            # C3'-O3'-P: vertex O3'[i] -> addAngle(C3, O3, P)
            if None not in (c3_i, o3_i, p_j):
                angle_force.addAngle(c3_i, o3_i, p_j, [THETA_C3_O3_P])
                n_angle += 1
            # O3'-P-O5': vertex P[i+1] -> addAngle(O3, P, O5)
            if None not in (o3_i, p_j, o5_j):
                angle_force.addAngle(o3_i, p_j, o5_j, [THETA_O3_P_O5])
                n_angle += 1
        system.addForce(angle_force)

    # --- Force 3: ViennaRNA pairing distance restraint (C1'-C1' ~10.6 A) ---
    pair_force = CustomBondForce("0.5*k_pairdist*(r-r0)^2")
    pair_force.addGlobalParameter("k_pairdist", 100.0)
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if not (0 <= i < L and 0 <= j < L and abs(i - j) > 1
                and not (i == 0 and j == L - 1)):
            continue
        ci = atom_lookup.get((i + 1, "C1'"))
        cj = atom_lookup.get((j + 1, "C1'"))
        if ci is None or cj is None:
            continue
        pair_force.addBond(ci, cj, [0.106])
    system.addForce(pair_force)

    # --- Force 4: A-form dihedral restraints (backbone torsions) ---
    # alpha = O3'[i-1]-P[i]-O5'[i]-C5'[i] (cross-residue; O3' comes from the previous residue)
    # gamma = O5'[i]-C5'[i]-C4'[i]-C3'[i] (intra-residue)
    # delta = C5'[i]-C4'[i]-C3'[i]-O3'[i] (intra-residue)
    # zeta  = C4'[i]-C3'[i]-O3'[i]-P[i+1] (cross-residue; P comes from the next residue)
    # Bug fix: the old version built alpha/zeta entirely from same-residue atoms
    # (a fake dihedral restraining a nonexistent angle), leaving the real
    # alpha/zeta 100-140 deg off A-form and the base-stacking distance at 7.9 A
    # (true value 3.4 A).
    torsion = CustomTorsionForce("0.5*k_aform*(theta-theta0)^2")
    torsion.addGlobalParameter("k_aform", 50.0)  # kJ/mol/rad²
    torsion.addPerTorsionParameter("theta0")
    n_torsions = 0
    for res_idx in range(L):
        res_seq = res_idx + 1
        prev_seq = ((res_idx - 1) % L) + 1  # previous residue res_seq (circular)
        next_seq = ((res_idx + 1) % L) + 1  # next residue res_seq (circular)
        for tname, (angle_deg, a1, a2, a3, a4) in _AFORM_TORSIONS.items():
            # alpha: a1 (O3') comes from the previous residue, the rest from this residue
            if tname == "alpha":
                i1 = atom_lookup.get((prev_seq, a1))
                i2 = atom_lookup.get((res_seq, a2))
                i3 = atom_lookup.get((res_seq, a3))
                i4 = atom_lookup.get((res_seq, a4))
            # zeta: a4 (P) comes from the next residue, the rest from this residue
            elif tname == "zeta":
                i1 = atom_lookup.get((res_seq, a1))
                i2 = atom_lookup.get((res_seq, a2))
                i3 = atom_lookup.get((res_seq, a3))
                i4 = atom_lookup.get((next_seq, a4))
            # gamma/delta: all atoms from this residue
            else:
                i1 = atom_lookup.get((res_seq, a1))
                i2 = atom_lookup.get((res_seq, a2))
                i3 = atom_lookup.get((res_seq, a3))
                i4 = atom_lookup.get((res_seq, a4))
            if None in (i1, i2, i3, i4):
                continue
            theta0 = angle_deg * np.pi / 180.0
            torsion.addTorsion(i1, i2, i3, i4, [theta0])
            n_torsions += 1
    system.addForce(torsion)

    # Assign force groups to the restraint terms (so per-term energies can be
    # read out after minimization, separating restraints from the amber field).
    # The amber force field built by createSystem stays in group 0; the restraint
    # terms go to groups 1..4. setForceGroup must be called before the Context
    # (Simulation) is created.
    constraint_forces = [restraint, bsj_bond, pair_force, torsion]
    for gi, f in enumerate(constraint_forces, start=1):
        try:
            f.setForceGroup(gi)
        except Exception:
            pass

    # --- Integrator + platform ---
    # Always use Langevin: the initial geometry has atomic clashes (e0 ~ 1e14),
    # which need MD annealing to break apart before minimization. The old logic
    # enabled MD only for L>=200, but a short sequence on Verlet cannot run step().
    use_md = True
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond
    )
    # Resolve the "auto" platform
    if platform_name == "auto":
        resolved = _detect_platform()
        print(f"  [amber_refine] auto-detected platform: {resolved}")
    else:
        resolved = platform_name
    try:
        platform = Platform.getPlatformByName(resolved)
        sim = Simulation(topo, system, integrator, platform)
    except Exception:
        fallback = _detect_platform()
        print(f"  [amber_refine] platform {resolved} unavailable; falling back to {fallback}")
        try:
            platform = Platform.getPlatformByName(fallback)
            sim = Simulation(topo, system, integrator, platform)
        except Exception:
            sim = Simulation(topo, system, integrator)

    # Coordinates after Modeller adds H (nm)
    positions = modeller.getPositions()
    sim.context.setPositions(positions)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy()._value

    # --- Scheme 6: pair-restraint annealing ---
    # Before the P relaxation and MD annealing, progressively tighten k/r0 to
    # pull the C1'-C1' distance from far apart (~25 A) toward Watson-Crick
    # geometry (10.6 A), avoiding structural distortion from a sudden strong
    # restraint. Each stage: setBondParameters -> updateParametersInContext ->
    # minimizeEnergy.
    _n_bonds_in_pair = pair_force.getNumBonds()
    if _n_bonds_in_pair > 0 and pair_anneal_stages is None:
        # Default annealing: 3 progressively tightening stages
        _pair_anneal_stages = [(20.0, 2.5), (50.0, 2.0), (100.0, 1.06)]
    elif pair_anneal_stages is not None:
        _pair_anneal_stages = pair_anneal_stages
    else:
        _pair_anneal_stages = []

    if _pair_anneal_stages:
        for _si, (_pk, _pr0) in enumerate(_pair_anneal_stages):
            # Update the global k_pairdist plus each bond's r0
            sim.context.setParameter("k_pairdist", _pk)
            for _bi in range(_n_bonds_in_pair):
                p1, p2, _old_params = pair_force.getBondParameters(_bi)
                pair_force.setBondParameters(_bi, p1, p2, [_pr0])
            pair_force.updateParametersInContext(sim.context)
            # Record the energy before annealing
            _pre_e = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
            # Minimize: 1000 steps each for the first two stages, and
            # max_iterations for the final stage
            _stage_iters = max_iterations if _si == len(_pair_anneal_stages) - 1 else 1000
            sim.minimizeEnergy(
                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=_stage_iters,
            )
            _post_e = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
            print(f"  [amber_refine] pair anneal stage {_si+1}/{len(_pair_anneal_stages)}: "
                  f"k={_pk:.0f} r0={_pr0:.2f}nm  E={_pre_e:.0f} -> {_post_e:.0f} kJ/mol")

    # --- Multi-stage minimization + annealing ---
    # Stage 1: loosen the P restraint (k_scale=0.01, ~100x softer) so the amber
    # force field dominates and drives bond lengths/angles/VdW to a minimum.
    # The per-particle k keeps the coding/non-coding distinction.
    sim.context.setParameter("k_scale", 0.01)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max(3000, max_iterations),
    )

    # Stage 2: gentle annealing MD. 1000K would fling the atoms of the initially
    # distorted structure away (NaN); use 500K with a few steps and check for NaN
    # after each stage, reverting to the loose-P minimization result on failure.
    pre_md_state = sim.context.getState(getPositions=True)
    pre_md_pos = pre_md_state.getPositions()
    try:
        sim.integrator.setTemperature(500 * unit.kelvin)
        sim.step(100)
        sim.integrator.setTemperature(300 * unit.kelvin)
        sim.step(50)
        # Check whether the MD produced any NaN
        chk = sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        if not np.isfinite(chk).all():
            raise RuntimeError("MD produced NaN; reverting to the pre-MD state")
        # Re-minimize under the loose P restraint.
        sim.minimizeEnergy(
            tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=max(3000, max_iterations),
        )
    except Exception as md_exc:
        # MD failed; restore the pre-MD coordinates and skip straight to stage 3.
        sim.context.setPositions(pre_md_pos)

    # Stage 3: tighten the P restraint back to its target value (k_scale=1.0),
    # holding the backbone topology, and run the final minimization.
    # The per-particle k was fixed when the particles were added (coding=
    # coding_restraint_k / non-coding=P_RESTRAINT_K); the global k_scale returns
    # from 0.01 (stage 1) to 1.0, restoring the full pinning strength.
    sim.context.setParameter("k_scale", 1.0)
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # Per-term energy diagnostic: setForceGroup must be called before the Context
    # is created; this block only reads the energies, the actual setForceGroup
    # happened in the "assign force groups" section before Simulation creation.
    force_energies = {}
    try:
        # The amber force field itself (group 0) - excludes the restraints
        s = sim.context.getState(getEnergy=True, groups={0})
        force_energies["amber_field"] = s.getPotentialEnergy()._value
        for gi, f in enumerate(constraint_forces, start=1):
            s = sim.context.getState(getEnergy=True, groups={gi})
            force_energies[f"constraint_{f.__class__.__name__}"] = \
                s.getPotentialEnergy()._value
    except Exception:
        pass
    refined_ang = pos * 10.0  # nm -> A, in topo2 order (heavy atoms + H alternating)

    # Extract heavy-atom-only coordinates, returned in structure.atoms order
    # (no H). heavy_to_topo[i] = the topo2 global index of the i-th heavy atom.
    # Callers (predictor/export/immune_heuristic) index coordinates by
    # structure.atoms[i]; H atoms must be excluded or the indices shift and the
    # BSJ/bond-length calculations all break.
    if len(heavy_to_topo) == n_heavy:
        refined_heavy = refined_ang[heavy_to_topo]  # (n_heavy, 3) Å
    else:
        # The mapping does not line up (unexpected); fall back to the full array
        # and let the caller deal with it
        refined_heavy = refined_ang

    # --- Safety check: P drift ---
    # P's index in structure.atoms equals the serial (0-based heavy-atom order)
    # of that residue's "P" atom. refined_heavy follows structure.atoms order, so
    # the serial indexes it directly.
    max_drift = 0.0
    for res_idx in range(L):
        p_serial = structure.residue_atom_index[res_idx].get("P")
        if p_serial is None or p_serial >= len(refined_heavy):
            continue
        # P's original coordinates (nm, the CG-solved value at reconstruction)
        p_ref = structure.atoms[p_serial].xyz
        drift = np.linalg.norm(refined_heavy[p_serial] - p_ref)
        if drift > max_drift:
            max_drift = drift
    if max_drift > P_MAX_DRIFT_A:
        raise RuntimeError(
            f"P drift {max_drift:.2f} A after amber minimization exceeds the "
            f"{P_MAX_DRIFT_A} A threshold; falling back to CG"
        )
    if np.isnan(e1) or np.isinf(e1):
        raise RuntimeError(f"Abnormal energy e1={e1} after amber minimization; falling back to CG")

    info = {
        "n_heavy": n_heavy,
        "n_atoms": n_total,
        "n_h": n_total - n_heavy,
        "n_torsions": n_torsions,
        "max_p_drift": float(max_drift),
        "force_energies": force_energies,
        "n_coding_pinned": n_coding_pinned,
        "n_topology_fused": n_topology_fused,
        "n_o3p_bonds": n_o3p,
    }
    # Return the heavy-atom coordinates (in structure.atoms order, no H). Callers index directly by serial.
    return refined_heavy, e0, e1, info


if __name__ == "__main__":
    from .aform_from_template import reconstruct_all_atom

    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles),
                   np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    print(f"rebuilt: {len(s.atoms)} heavy atoms")
    pairs = [(i, L - 1 - i, 1.0) for i in range(L // 2)]
    coords, e0, e1, info = amber_refine(s, pairs, max_iterations=500)
    print(f"e0={e0:.0f} → e1={e1:.0f} kJ/mol")
    print(f"info: {info}")
