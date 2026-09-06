"""
boundary_constraints.py — Level 1 boundary constraint injection

Extract pairing information that crosses segment boundaries from the global
pairing matrix (bpp) and feed it to the RhoFold+ prediction as hard
constraints, fixing the boundary-topology breaks of segmented prediction.

Public API:
  build_boundary_pairs()  — build the boundary constraint pairs for each segment
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# --- constants ---
BOUNDARY_MARGIN = 50   # boundary window: pairs within 50 nt either side of a segment edge count as boundary constraints
WC_DIST_TARGET = 20.0  # Watson-Crick C1'-C1' target distance (Å)
EDGE_TYPES = ("wc", "non_wc")  # wc=Watson-Crick, non_wc=non-WC pairing


def build_boundary_pairs(
    global_bpp: np.ndarray,
    segments: List[Dict],
    seq_len: int,
    bpp_threshold: float = 0.3,
    boundary_margin: int = BOUNDARY_MARGIN,
) -> List[List[Tuple[int, int, str]]]:
    """Extract each segment's boundary constraint pairs from the global bpp matrix.

    Boundary definition: pairs that lie within boundary_margin nt of a segment edge
    and cross that segment boundary. I.e. one residue of the pair is in the current
    segment's boundary window while the other is in an adjacent segment.

    Args:
        global_bpp: (L, L) global pairing probability matrix (symmetric, upper triangle)
        segments: segment list returned by split_sequence()
        seq_len: full sequence length
        bpp_threshold: pairing probability threshold; pairs below it are not included
        boundary_margin: boundary window size (nt)

    Returns:
        A list of length == len(segments); each element is the list of boundary
        constraint pairs for that segment.
        Each constraint pair: (global_i, global_j, edge_type)
        edge_type: "wc" (Watson-Crick) or "non_wc" (non-WC)
    """
    L = seq_len
    n_seg = len(segments)
    all_boundary_pairs: List[List[Tuple[int, int, str]]] = [[] for _ in range(n_seg)]

    if global_bpp.shape != (L, L):
        raise ValueError(
            f"global_bpp shape {global_bpp.shape} != ({L}, {L})"
        )

    for idx, seg in enumerate(segments):
        seg_start = seg["start"]
        seg_end = seg["end"]  # exclusive

        # boundary window: boundary_margin nt on each side of the segment
        window_start = max(0, seg_start - boundary_margin)
        window_end = min(L, seg_end + boundary_margin)

        # scan the pairs inside the window
        seen = set()
        for i in range(window_start, window_end):
            for j in range(i + 1, window_end):
                if j >= L:
                    break
                prob = global_bpp[i, j]
                if prob < bpp_threshold:
                    continue

                # check whether the pair crosses a segment boundary:
                # one residue inside the segment, the other outside it
                i_in_seg = seg_start <= i < seg_end
                j_in_seg = seg_start <= j < seg_end

                crosses = (i_in_seg and not j_in_seg) or (j_in_seg and not i_in_seg)
                if not crosses:
                    continue

                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)

                # determine the edge type
                edge_type = _classify_pair_type(i, j, L)
                all_boundary_pairs[idx].append((i, j, edge_type))

    return all_boundary_pairs


def _classify_pair_type(i: int, j: int, seq_len: int) -> str:
    """Naively classify the pair type (used for constraint classification).

    A distance heuristic is assumed here: in an ideal conformation a WC pair has
    C1'-C1' ~10.5Å, while non_wc is farther apart. The bpp matrix does not
    distinguish them directly, so everything is labeled "wc" (RhoFold+'s distance
    restraints apply to both classes; only the target distance differs).

    TODO: later, distinguish using ViennaRNA pair types
    """
    return "wc"


def apply_boundary_constraints_to_coords(
    coords: np.ndarray,
    boundary_pairs: List[Tuple[int, int, str]],
    n_steps: int = 2000,
) -> np.ndarray:
    """Relax the initial coordinates under boundary restraints.

    Applies distance restraints to the boundary pairs so that pairs crossing a
    segment boundary satisfy a reasonable geometric relationship in 3D space.

    Args:
        coords: (L, 3) initial P/C1' coordinates
        boundary_pairs: constraint pair list returned by build_boundary_pairs()
        n_steps: number of energy minimization steps

    Returns:
        (L, 3) coordinates after restraint relaxation
    """
    if not boundary_pairs:
        return coords

    try:
        import openmm
        from openmm import app, unit

        L = len(coords)
        system = openmm.System()

        # topology
        topology = app.Topology()
        chain = topology.addChain()
        res = topology.addResidue("RNA", chain)
        for i in range(L):
            topology.addAtom(f"P{i}", app.Element.getBySymbol("P"), res)

        # particle masses
        for i in range(L):
            system.addParticle(110.0)

        # bond length restraints (harmonic, keep the backbone)
        bond_force = openmm.HarmonicBondForce()
        for i in range(L - 1):
            bond_force.addBond(
                i, i + 1,
                5.9 * unit.angstrom,
                100.0 * unit.kilocalorie_per_mole / unit.angstrom**2,
            )
        system.addForce(bond_force)

        # boundary pair distance restraints
        pair_force = openmm.HarmonicBondForce()
        for gi, gj, edge_type in boundary_pairs:
            if gi >= L or gj >= L:
                continue
            target = WC_DIST_TARGET  # uniform target distance
            pair_force.addBond(
                gi, gj,
                target * unit.angstrom,
                50.0 * unit.kilocalorie_per_mole / unit.angstrom**2,
            )
        system.addForce(pair_force)

        # set the coordinates
        positions = [
            openmm.Vec3(coords[i, 0], coords[i, 1], coords[i, 2]) * unit.angstrom
            for i in range(L)
        ]

        # energy minimization
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond,
        )
        context = openmm.Context(system, integrator)
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=n_steps)

        state = context.getState(getPositions=True)
        positions = state.getPositions()
        result = np.array([
            [positions[i].x, positions[i].y, positions[i].z]
            for i in range(L)
        ])
        return result

    except ImportError:
        return coords
    except Exception:
        return coords


def format_boundary_pdb_remarks(
    boundary_pairs: List[Tuple[int, int, str]],
) -> List[str]:
    """Format the boundary constraints as PDB REMARK lines.

    Convenient for recording in the PDB which pairs are hard constraints, so they
    remain traceable while debugging.

    Returns:
        A list of PDB REMARK lines (each including its newline)
    """
    lines = [
        "REMARK   1 Boundary constraints from global bpp\n",
        "REMARK   1 Format: RESIDUE_I RESIDUE_J EDGE_TYPE PROBABILITY\n",
    ]
    for gi, gj, edge_type in boundary_pairs:
        lines.append(
            f"REMARK   1 BOUNDARY {gi+1:5d} {gj+1:5d} {edge_type:8s}\n"
        )
    return lines
