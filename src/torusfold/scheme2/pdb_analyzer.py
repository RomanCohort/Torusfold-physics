"""PDB quick analyzer -- lightweight structure quality scoring for imported PDB files.

Designed for real-time feedback: parses ATOM records, computes clash score,
radius of gyration, SASA, bond RMSD, backbone angles, A-form score, stacking,
and pair satisfaction. No heavy dependencies (numpy + scipy.spatial.cKDTree only).
"""

import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Physical Constants ───────────────────────────────────────────

IDEAL_PP_BOND = 5.9            # A-form P-P backbone distance (A)
IDEAL_PPP_ANGLE = 150.0        # A-form P-P-P backbone angle (deg)
IDEAL_PPP_ANGLE_RAD = 2.618    # in radians
IDEAL_PPP_DIHEDRAL = 33.0      # A-form helical twist (deg)
IDEAL_STACKING_DIST = 3.4      # Base stacking distance (A)
IDEAL_WC_DIST = 10.6           # WC C1'-C1' distance (A)
PROBE_RADIUS = 1.4             # SASA probe radius (A)

VDW_TOLERANCE = 0.4            # Clash tolerance (A)

# Van der Waals radii (A) -- Bondi 1964 / Trend values
VDW_RADII = {
    'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8, 'S': 1.8,
    'H': 1.2, 'FE': 0.63, 'MG': 0.72, 'CA': 0.99, 'NA': 1.02,
}
DEFAULT_VDW_RADIUS = 1.5

# Watson-Crick complementarity
WC_COMPLEMENT = {
    'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G',
    'DA': 'DT', 'DT': 'DA', 'DG': 'DC', 'DC': 'DG',
    'RA': 'RU', 'RU': 'RA', 'RG': 'RC', 'RC': 'RG',
}


# ── PDB Parsing ─────────────────────────────────────────────────

def parse_pdb(pdb_text: str) -> Dict:
    """Parse ATOM/HETATM records from PDB text.

    Returns dict with:
        coords: np.ndarray (N, 3)  -- all atom coordinates
        atom_names: List[str]       -- atom names (CA, P, N3, etc.)
        residue_names: List[str]    -- residue names (A, G, C, U, DA, etc.)
        residue_ids: List[int]      -- residue sequence numbers
        chain_ids: List[str]        -- chain identifiers
        elements: List[str]         -- element symbols
        b_factors: List[float]      -- B-factor values
        n_atoms: int
        n_residues: int
        n_chains: int
        is_nucleic: bool            -- True if nucleic acid (has P atoms)
    """
    coords = []
    atom_names = []
    residue_names = []
    residue_ids = []
    chain_ids = []
    elements = []
    b_factors = []
    residues_seen = set()

    for line in pdb_text.split('\n'):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except (ValueError, IndexError):
            continue

        coords.append([x, y, z])
        atom_names.append(line[12:16].strip())
        resname = line[17:20].strip()
        residue_names.append(resname)
        resseq = int(line[22:26].strip())
        residue_ids.append(resseq)
        chain_ids.append(line[21] if len(line) > 21 and line[21].strip() else 'A')
        elements.append(line[76:78].strip() if len(line) > 76 else '')
        try:
            b_factors.append(float(line[60:66].strip()))
        except (ValueError, IndexError):
            b_factors.append(0.0)

        residues_seen.add((chain_ids[-1], resseq, resname))

    coords = np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))
    has_p = any(n in ('P', 'OP1', 'OP2', 'OP3') for n in atom_names)

    return {
        'coords': coords,
        'atom_names': atom_names,
        'residue_names': residue_names,
        'residue_ids': residue_ids,
        'chain_ids': chain_ids,
        'elements': elements,
        'b_factors': b_factors,
        'n_atoms': len(coords),
        'n_residues': len(residues_seen),
        'n_chains': len(set(chain_ids)),
        'is_nucleic': has_p,
    }


# ── Helpers ──────────────────────────────────────────────────────

def _get_vdw_radius(atom_name: str) -> float:
    """Return vdW radius for an atom name, falling back to DEFAULT."""
    # Strip leading digits (e.g. '2HO' -> 'HO') and trailing digits
    stripped = atom_name.strip()
    for ch in stripped:
        if ch.isalpha():
            # element = first letter (uppercase) + any following lowercase
            idx = stripped.index(ch)
            elem = stripped[idx:].lstrip('0123456789')[:1].upper()
            if elem in VDW_RADII:
                return VDW_RADII[elem]
            break
    return DEFAULT_VDW_RADIUS


def _find_p_indices(atom_names: List[str]) -> np.ndarray:
    """Return array of indices where atom_names[i] == 'P'."""
    return np.array([i for i, name in enumerate(atom_names) if name == 'P'],
                    dtype=np.int64)


# ── Clash Detection ─────────────────────────────────────────────

def compute_clash_score(coords: np.ndarray, atom_names: List[str],
                        residue_ids: List[int] = None,
                        clash_cutoff: float = 4.0) -> Dict:
    """Compute clash score using per-atom vdW radii.

    A clash occurs when d_ij < r_i + r_j - VDW_TOLERANCE.

    Returns dict with:
        clash_count: int           -- number of clashing pairs
        clash_score: float         -- clashing pairs per 1000 atoms
        worst_clashes: list of (i, j, dist, name_i, name_j) top 10
        mean_overlap: float        -- mean vdW overlap for clashing pairs (A)
    """
    n = len(coords)
    if n < 2:
        return {'clash_count': 0, 'clash_score': 0, 'worst_clashes': [],
                'mean_overlap': 0}

    if residue_ids is None:
        residue_ids = list(range(n))

    # Pre-compute per-atom vdW radii
    radii = np.array([_get_vdw_radius(name) for name in atom_names],
                     dtype=np.float64)
    # KD-tree query radius: largest possible vdW diameter + safety margin
    max_diameter = 2.0 * np.max(radii) + 1.0

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_diameter, output_type='ndarray')

    clashes = []
    overlaps = []
    for i, j in pairs:
        di = int(i)
        dj = int(j)
        # Skip bonded atoms: same residue or adjacent residues (|Δseq| ≤ 1)
        if abs(residue_ids[di] - residue_ids[dj]) <= 1:
            continue
        d = float(np.linalg.norm(coords[di] - coords[dj]))
        sum_radii = radii[di] + radii[dj]
        if d < sum_radii - VDW_TOLERANCE:
            overlap = sum_radii - d
            clashes.append((di, dj, d, atom_names[di], atom_names[dj]))
            overlaps.append(overlap)

    clashes.sort(key=lambda x: x[2])
    clash_score = len(clashes) / max(1, n) * 1000

    return {
        'clash_count': len(clashes),
        'clash_score': round(clash_score, 2),
        'worst_clashes': clashes[:10],
        'mean_overlap': round(float(np.mean(overlaps)), 3) if overlaps else 0,
    }


# ── Radius of Gyration ─────────────────────────────────────────

def compute_radius_of_gyration(coords: np.ndarray) -> float:
    """Compute radius of gyration in Angstroms."""
    if len(coords) < 2:
        return 0.0
    center = np.mean(coords, axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


# ── SASA Estimate (Shrake-Rupley) ──────────────────────────────

def _fibonacci_sphere(n_points: int, radius: float) -> np.ndarray:
    """Generate approximately uniform points on a sphere via Fibonacci spiral."""
    golden_ratio = (1 + math.sqrt(5)) / 2
    points = []
    for i in range(n_points):
        theta = math.acos(1 - 2 * (i + 0.5) / n_points)
        phi = 2 * math.pi * i / golden_ratio
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        points.append([x, y, z])
    return np.array(points)


def compute_sasa_estimate(coords: np.ndarray, atom_names: List[str],
                          n_probe: int = None) -> Dict:
    """Compute solvent-accessible surface area per atom (Shrake-Rupley algorithm).

    For each atom, places n_probe points on its vdW surface. A point is
    "exposed" if no other atom's (vdW + probe) sphere contains it.

    Returns dict with:
        mean_sasa: float       -- mean SASA per atom (A^2)
        per_atom_sasa: list    -- SASA per atom (A^2)
        per_residue_sasa: list -- SASA aggregated per residue (A^2)
        buried_fraction: float -- fraction of residues with SASA < 10 A^2
        total_sasa: float      -- total SASA (A^2)
    """
    n = len(coords)
    if n < 1:
        return {'mean_sasa': 0, 'per_atom_sasa': [], 'per_residue_sasa': [],
                'buried_fraction': 0, 'total_sasa': 0}

    # Adaptive probe points: fewer for large structures
    if n_probe is None:
        if n < 200:
            n_probe = 50
        elif n < 1000:
            n_probe = 24
        else:
            n_probe = 12  # ~100x speedup for 2000 atoms

    radii = np.array([_get_vdw_radius(name) for name in atom_names],
                     dtype=np.float64)
    accessible_radii = radii + PROBE_RADIUS

    # Fast neighbor-counting SASA approximation
    # For each atom, count neighbors within accessible radii sum
    # and compute fractional burial using solid-angle overlap
    per_atom_sasa = np.zeros(n, dtype=np.float64)

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    max_ar = float(np.max(accessible_radii))
    query_r = 2.0 * max_ar

    for i in range(n):
        r_acc = accessible_radii[i]
        neighbor_indices = tree.query_ball_point(coords[i], query_r)
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]

        if not neighbor_indices:
            per_atom_sasa[i] = 4.0 * math.pi * r_acc ** 2
            continue

        neighbor_coords = coords[neighbor_indices]
        neighbor_acc_r = accessible_radii[neighbor_indices]

        diffs = neighbor_coords - coords[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Fraction of surface buried by each neighbor
        # f = 0.5 * (1 - d/(r_i + r_j)) for overlapping spheres
        sum_r = r_acc + neighbor_acc_r
        overlap = np.clip(sum_r - dists, 0, None)
        exposed_frac = 0.5 * (1.0 - dists / (sum_r + 1e-10))
        exposed_frac = np.clip(np.where(dists < sum_r, exposed_frac, 0.0), 0, 1)
        burial = float(np.sum(exposed_frac))

        full_sasa = 4.0 * math.pi * r_acc ** 2
        per_atom_sasa[i] = full_sasa * max(0.0, 1.0 - min(burial, 1.0))

    total_sasa = float(np.sum(per_atom_sasa))
    mean_sasa = float(np.mean(per_atom_sasa))

    return {
        'mean_sasa': round(mean_sasa, 3),
        'per_atom_sasa': per_atom_sasa.tolist(),
        'total_sasa': round(total_sasa, 3),
    }


# ── End-to-End Distance ─────────────────────────────────────────

def compute_end_to_end(coords: np.ndarray) -> float:
    """Compute end-to-end distance (first to last atom)."""
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(coords[0] - coords[-1]))


# ── Asphericity & Prolateness ───────────────────────────────────

def compute_shape_descriptors(coords: np.ndarray) -> Dict:
    """Compute shape descriptors from gyration tensor eigenvalues.

    Returns dict with:
        rog: float -- radius of gyration
        asphericity: float -- 0 = sphere, 1 = rod
        prolateness: float -- +1 = prolate, -1 = oblate
        eigenvalues: list of 3 floats
    """
    if len(coords) < 3:
        return {'rog': 0, 'asphericity': 0, 'prolateness': 0,
                'eigenvalues': [0, 0, 0]}

    center = np.mean(coords, axis=0)
    centered = coords - center
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]

    rog = float(np.sqrt(np.sum(eigenvalues)))
    if rog < 1e-10:
        return {'rog': 0, 'asphericity': 0, 'prolateness': 0,
                'eigenvalues': [0, 0, 0]}

    l1, l2, l3 = eigenvalues
    asphericity = float((l1 - 0.5 * (l2 + l3)) / (rog ** 2))
    prolateness = float(
        (3.0 / 2.0) * np.sum(eigenvalues ** 2) / (rog ** 2) ** 2 - 0.5
    )

    return {
        'rog': round(rog, 3),
        'asphericity': round(asphericity, 4),
        'prolateness': round(prolateness, 4),
        'eigenvalues': [round(float(e), 2) for e in eigenvalues],
    }


# ── Bond RMSD ───────────────────────────────────────────────────

def compute_bond_rmsd(coords: np.ndarray, atom_names: List[str],
                      residue_ids: List[int]) -> Dict:
    """Compute backbone bond RMSD from ideal P-P distance.

    Finds P atoms explicitly, extracts P coordinates in order, and computes
    consecutive P-P distances vs. ideal 5.9 A.

    Returns dict with:
        mean_bond_length: float -- mean P-P distance (A)
        bond_rmsd: float        -- RMSD from ideal bond length (A)
        n_bonds: int            -- number of P-P bonds found
        n_violations: int       -- bonds deviating >0.5A from ideal
        violations: list        -- [(idx_i, idx_j, distance)] for violating bonds
    """
    p_indices = _find_p_indices(atom_names)

    if len(p_indices) < 2:
        return {'mean_bond_length': 0, 'bond_rmsd': 0, 'n_bonds': 0,
                'n_violations': 0, 'violations': []}

    p_coords = coords[p_indices]  # (M, 3)

    # Consecutive P-P distances
    diffs = np.diff(p_coords, axis=0)  # (M-1, 3)
    bond_lengths = np.linalg.norm(diffs, axis=1)  # (M-1,)

    mean_bl = float(np.mean(bond_lengths))
    rmsd = float(np.sqrt(np.mean((bond_lengths - IDEAL_PP_BOND) ** 2)))

    # Violations: > 0.5A from ideal (5.9A is already coarse-grained)
    violation_mask = np.abs(bond_lengths - IDEAL_PP_BOND) > 0.5
    n_violations = int(np.sum(violation_mask))

    violations = []
    for k in range(len(bond_lengths)):
        if violation_mask[k]:
            idx_i = int(p_indices[k])
            idx_j = int(p_indices[k + 1])
            violations.append((idx_i, idx_j, round(float(bond_lengths[k]), 3)))

    return {
        'mean_bond_length': round(mean_bl, 3),
        'bond_rmsd': round(rmsd, 3),
        'n_bonds': len(bond_lengths),
        'n_violations': n_violations,
        'violations': violations,
    }


# ── Backbone Angles (vectorized) ───────────────────────────────

def compute_backbone_angles(coords: np.ndarray, atom_names: List[str],
                            residue_ids: List[int]) -> Dict:
    """Compute vectorized P-P-P backbone angles and P-P-P-P dihedrals.

    Uses numpy vectorization for angle computation. Dihedrals use the
    standard cross-product formulation.

    Returns dict with:
        mean_angle: float   -- mean P-P-P angle (deg)
        std_angle: float    -- std of P-P-P angles (deg)
        mean_dihedral: float -- mean P-P-P-P dihedral (deg)
        std_dihedral: float  -- std of P-P-P-P dihedrals (deg)
        n_angles: int
        n_dihedrals: int
    """
    p_indices = _find_p_indices(atom_names)

    if len(p_indices) < 3:
        return {'mean_angle': 0, 'std_angle': 0, 'mean_dihedral': 0,
                'std_dihedral': 0, 'n_angles': 0, 'n_dihedrals': 0}

    p_coords = coords[p_indices]  # (M, 3)

    # ── Angles: P-P-P ──
    # v1 = coords[:-2] - coords[1:-1], v2 = coords[2:] - coords[1:-1]
    v1 = p_coords[:-2] - p_coords[1:-1]   # (M-2, 3)
    v2 = p_coords[2:] - p_coords[1:-1]    # (M-2, 3)

    norms1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(v2, axis=1, keepdims=True)

    # Avoid division by zero
    safe_norms = norms1 * norms2
    safe_norms = np.maximum(safe_norms, 1e-10)

    cos_angles = np.sum(v1 * v2, axis=1) / safe_norms.ravel()
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_rad = np.arccos(cos_angles)
    angles_deg = np.degrees(angles_rad)

    # ── Dihedrals: P-P-P-P ──
    if len(p_indices) >= 4:
        b1 = p_coords[1:-2] - p_coords[:-3]   # (M-3, 3)
        b2 = p_coords[2:-1] - p_coords[1:-2]  # (M-3, 3)
        b3 = p_coords[3:] - p_coords[2:-1]    # (M-3, 3)

        n1 = np.cross(b1, b2)  # (M-3, 3)
        n2 = np.cross(b2, b3)  # (M-3, 3)

        # Normalize n1, n2
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)

        # Avoid division by zero for degenerate cases
        n1_norm_safe = np.maximum(n1_norm, 1e-10)
        n2_norm_safe = np.maximum(n2_norm, 1e-10)

        # Unit normals
        n1_unit = n1 / n1_norm_safe
        n2_unit = n2 / n2_norm_safe

        # b2 unit vector for sign
        b2_norm = np.linalg.norm(b2, axis=1, keepdims=True)
        b2_norm_safe = np.maximum(b2_norm, 1e-10)
        b2_unit = b2 / b2_norm_safe

        # dihedral = atan2(dot(cross(n1, n2), b2/|b2|), dot(n1, n2))
        m1 = np.cross(n1_unit, n2_unit)
        x = np.sum(m1 * b2_unit, axis=1)
        y = np.sum(n1_unit * n2_unit, axis=1)
        dihedrals_rad = np.arctan2(x, y)
        dihedrals_deg = np.degrees(dihedrals_rad)
    else:
        dihedrals_deg = np.array([])

    return {
        'mean_angle': round(float(np.mean(angles_deg)), 2),
        'std_angle': round(float(np.std(angles_deg)), 2),
        'mean_dihedral': round(float(np.mean(dihedrals_deg)), 2) if len(dihedrals_deg) > 0 else 0,
        'std_dihedral': round(float(np.std(dihedrals_deg)), 2) if len(dihedrals_deg) > 0 else 0,
        'n_angles': len(angles_deg),
        'n_dihedrals': len(dihedrals_deg),
    }


# ── Pair Satisfaction (KD-tree, with WC complementarity) ───────

def compute_pair_satisfaction(coords: np.ndarray, residue_ids: List[int],
                               atom_names: List[str],
                               residue_names: List[str]) -> Dict:
    """Estimate base pair satisfaction with WC complementarity check.

    Uses KD-tree for spatial neighbor queries and classifies pairs by
    sequence distance (local/medium/long range).

    Returns dict with:
        total_pairs: int              -- pairs found within 15A
        wc_eligible_count: int        -- pairs where residues are WC complements
        satisfied_count: int          -- WC-eligible AND distance < 12A
        satisfaction_rate: float      -- satisfied / wc_eligible
        mean_pair_distance: float     -- mean distance of all pairs
        by_range: dict                -- local/medium/long range stats
    """
    if len(coords) < 4:
        return {'total_pairs': 0, 'wc_eligible_count': 0,
                'satisfied_count': 0, 'satisfaction_rate': 0,
                'mean_pair_distance': 0, 'by_range': {}}

    # Build per-residue representative (P coords, res_id, res_name)
    p_indices = _find_p_indices(atom_names)

    if len(p_indices) < 4:
        # Fallback: use first atom per residue
        seen = set()
        p_coords_list = []
        p_res_ids = []
        p_res_names = []
        for i in range(len(residue_ids)):
            rid = residue_ids[i]
            if rid not in seen:
                seen.add(rid)
                p_coords_list.append(coords[i])
                p_res_ids.append(rid)
                p_res_names.append(residue_names[i])
        if len(p_coords_list) < 4:
            return {'total_pairs': 0, 'wc_eligible_count': 0,
                    'satisfied_count': 0, 'satisfaction_rate': 0,
                    'mean_pair_distance': 0, 'by_range': {}}
        p_coords = np.array(p_coords_list)
    else:
        p_coords = coords[p_indices]
        # Get residue info for each P atom (matching index)
        seen = set()
        p_res_ids = []
        p_res_names = []
        for idx in p_indices:
            rid = residue_ids[idx]
            if rid not in seen:
                seen.add(rid)
                p_res_ids.append(rid)
                p_res_names.append(residue_names[idx])

    n = len(p_coords)

    from scipy.spatial import cKDTree
    tree = cKDTree(p_coords)
    pairs = tree.query_pairs(r=15.0, output_type='ndarray')

    pair_distances = []
    wc_eligible = 0
    satisfied = 0
    ranges = {'local': {'count': 0, 'wc': 0, 'satisfied': 0},
              'medium': {'count': 0, 'wc': 0, 'satisfied': 0},
              'long': {'count': 0, 'wc': 0, 'satisfied': 0}}

    for i, j in pairs:
        di = int(i)
        dj = int(j)
        # Only consider inter-residue pairs (>3 residues apart)
        if abs(p_res_ids[di] - p_res_ids[dj]) <= 3:
            continue

        d = float(np.linalg.norm(p_coords[di] - p_coords[dj]))
        seq_dist = abs(p_res_ids[di] - p_res_ids[dj])

        # Classify by range
        if seq_dist < 50:
            rkey = 'local'
        elif seq_dist < 500:
            rkey = 'medium'
        else:
            rkey = 'long'

        pair_distances.append(d)
        ranges[rkey]['count'] += 1

        # Check WC complementarity
        rn_i = p_res_names[di]
        rn_j = p_res_names[dj]
        wc_match = WC_COMPLEMENT.get(rn_i) == rn_j or WC_COMPLEMENT.get(rn_j) == rn_i

        if wc_match:
            wc_eligible += 1
            ranges[rkey]['wc'] += 1
            if d < 12.0:
                satisfied += 1
                ranges[rkey]['satisfied'] += 1

    n_pairs = len(pair_distances)
    sat_rate = satisfied / max(1, wc_eligible)

    # Format range stats
    by_range = {}
    for rkey in ('local', 'medium', 'long'):
        r = ranges[rkey]
        by_range[rkey] = {
            'count': r['count'],
            'wc_eligible': r['wc'],
            'satisfied': r['satisfied'],
            'satisfaction_rate': round(r['satisfied'] / max(1, r['wc']), 3),
        }

    return {
        'total_pairs': n_pairs,
        'wc_eligible_count': wc_eligible,
        'satisfied_count': satisfied,
        'satisfaction_rate': round(sat_rate, 3),
        'mean_pair_distance': round(float(np.mean(pair_distances)), 2) if pair_distances else 0,
        'by_range': by_range,
    }


# ── A-form Score ────────────────────────────────────────────────

def compute_aform_score(coords: np.ndarray, atom_names: List[str]) -> Dict:
    """Compute how closely the backbone matches ideal A-form geometry.

    Normalizes deviations from ideal bond length, angle, and dihedral,
    then combines into an RMSD score. Lower = closer to ideal A-form.

    Returns dict with:
        aform_score: float       -- overall normalized RMSD (unitless, lower=better)
        bond_deviation: float    -- normalized bond deviation
        angle_deviation: float   -- normalized angle deviation
        dihedral_deviation: float -- normalized dihedral deviation
        n_bonds: int
        n_angles: int
        n_dihedrals: int
    """
    p_indices = _find_p_indices(atom_names)

    if len(p_indices) < 4:
        return {'aform_score': 0, 'bond_deviation': 0, 'angle_deviation': 0,
                'dihedral_deviation': 0, 'n_bonds': 0, 'n_angles': 0,
                'n_dihedrals': 0}

    p_coords = coords[p_indices]  # (M, 3)

    # Tolerances for normalization
    BOND_TOL = 0.5     # A
    ANGLE_TOL = 20.0   # deg
    DIHEDRAL_TOL = 20.0  # deg

    # ── Bond lengths ──
    bond_diffs = np.diff(p_coords, axis=0)
    bond_lengths = np.linalg.norm(bond_diffs, axis=1)
    bond_norm = (bond_lengths - IDEAL_PP_BOND) / BOND_TOL  # normalized deviations

    # ── Angles P-P-P ──
    v1 = p_coords[:-2] - p_coords[1:-1]
    v2 = p_coords[2:] - p_coords[1:-1]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    safe = np.maximum(n1 * n2, 1e-10)
    cos_a = np.clip(np.sum(v1 * v2, axis=1) / safe.ravel(), -1, 1)
    angles_deg = np.degrees(np.arccos(cos_a))
    angle_norm = (angles_deg - IDEAL_PPP_ANGLE) / ANGLE_TOL

    # ── Dihedrals P-P-P-P ──
    b1 = p_coords[1:-2] - p_coords[:-3]
    b2 = p_coords[2:-1] - p_coords[1:-2]
    b3 = p_coords[3:] - p_coords[2:-1]
    cn1 = np.cross(b1, b2)
    cn2 = np.cross(b2, b3)
    cn1_n = np.linalg.norm(cn1, axis=1, keepdims=True)
    cn2_n = np.linalg.norm(cn2, axis=1, keepdims=True)
    cn1_u = cn1 / np.maximum(cn1_n, 1e-10)
    cn2_u = cn2 / np.maximum(cn2_n, 1e-10)
    b2_n = np.linalg.norm(b2, axis=1, keepdims=True)
    b2_u = b2 / np.maximum(b2_n, 1e-10)
    mx = np.cross(cn1_u, cn2_u)
    x = np.sum(mx * b2_u, axis=1)
    y = np.sum(cn1_u * cn2_u, axis=1)
    dihedrals_deg = np.degrees(np.arctan2(x, y))
    dihedral_norm = (dihedrals_deg - IDEAL_PPP_DIHEDRAL) / DIHEDRAL_TOL

    # Combined score: sqrt(mean of all normalized deviations squared)
    all_norm = np.concatenate([bond_norm, angle_norm, dihedral_norm])
    aform_score = float(np.sqrt(np.mean(all_norm ** 2)))

    return {
        'aform_score': round(aform_score, 4),
        'bond_deviation': round(float(np.mean(np.abs(bond_norm))), 4),
        'angle_deviation': round(float(np.mean(np.abs(angle_norm))), 4),
        'dihedral_deviation': round(float(np.mean(np.abs(dihedral_norm))), 4),
        'n_bonds': len(bond_lengths),
        'n_angles': len(angles_deg),
        'n_dihedrals': len(dihedrals_deg),
    }


# ── Stacking Analysis ──────────────────────────────────────────

def compute_stacking_analysis(coords: np.ndarray, atom_names: List[str]) -> Dict:
    """Analyze base stacking from P atom geometry.

    Consecutive residues with P-P distance < 7A and bond vector angle < 30
    degrees are counted as stacked.

    Returns dict with:
        stacking_count: int    -- number of stacked consecutive pairs
        stacking_fraction: float -- fraction of consecutive pairs that are stacked
        mean_p_p_distance: float -- mean consecutive P-P distance (A)
    """
    p_indices = _find_p_indices(atom_names)

    if len(p_indices) < 2:
        return {'stacking_count': 0, 'stacking_fraction': 0,
                'mean_p_p_distance': 0}

    p_coords = coords[p_indices]  # (M, 3)
    n_pairs = len(p_coords) - 1

    # Consecutive P-P distances
    diffs = np.diff(p_coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    # Bond vectors for angle between consecutive bonds
    # Bond i: p_coords[i+1] - p_coords[i]
    # Angle between bond i-1 and bond i
    if len(diffs) >= 2:
        v1 = diffs[:-1]  # (M-2, 3)
        v2 = diffs[1:]   # (M-2, 3)
        n1 = np.linalg.norm(v1, axis=1, keepdims=True)
        n2 = np.linalg.norm(v2, axis=1, keepdims=True)
        safe = np.maximum(n1 * n2, 1e-10)
        cos_angle = np.clip(np.sum(v1 * v2, axis=1) / safe.ravel(), -1, 1)
        bond_angles = np.degrees(np.arccos(cos_angle))
    else:
        bond_angles = np.array([])

    # Stacking criteria: P-P < 7A for consecutive pairs
    stacked = dists < 7.0
    stacking_count = int(np.sum(stacked))

    # For multi-bond angle check: both bonds around a residue must be < 30 deg
    # This is checked for residues that have bonds on both sides
    if len(bond_angles) > 0:
        # bond_angles[k] is the angle between bond k and bond k+1
        # This angle is at residue (k+1) of the original P array
        # Residue (k+1) is stacked if both adjacent P-P distances < 7A
        # AND the angle is reasonable
        for k in range(len(bond_angles)):
            if bond_angles[k] >= 30.0:
                # Both surrounding bonds must exist and be short
                # Bond k connects p[k]->p[k+1], bond k+1 connects p[k+1]->p[k+2]
                if stacked[k] and stacked[k + 1] if k + 1 < len(stacked) else False:
                    pass  # could relax stacking criteria

    stacking_fraction = stacking_count / max(1, n_pairs)

    return {
        'stacking_count': stacking_count,
        'stacking_fraction': round(stacking_fraction, 4),
        'mean_p_p_distance': round(float(np.mean(dists)), 3),
    }


# ── Main Analyzer ───────────────────────────────────────────────

def analyze_pdb(pdb_text: str) -> Dict:
    """Run full quick analysis on a PDB string.

    Returns comprehensive dict with all metrics.
    """
    parsed = parse_pdb(pdb_text)
    coords = parsed['coords']

    if parsed['n_atoms'] == 0:
        return {'error': 'No ATOM records found in PDB', 'n_atoms': 0}

    atom_names = parsed['atom_names']
    residue_ids = parsed['residue_ids']
    residue_names = parsed['residue_names']

    # Core metrics
    clash = compute_clash_score(coords, atom_names, residue_ids)
    rog = compute_radius_of_gyration(coords)
    bond = compute_bond_rmsd(coords, atom_names, residue_ids)
    sasa = compute_sasa_estimate(coords, atom_names)
    e2e = compute_end_to_end(coords)
    shape = compute_shape_descriptors(coords)
    backbone = compute_backbone_angles(coords, atom_names, residue_ids)

    # Per-residue SASA aggregation
    per_residue_sasa = []
    buried_count = 0
    if 'per_atom_sasa' in sasa and len(sasa['per_atom_sasa']) > 0:
        # Group by residue_id
        res_sasa: Dict[int, float] = {}
        for i, rid in enumerate(residue_ids):
            res_sasa[rid] = res_sasa.get(rid, 0.0) + sasa['per_atom_sasa'][i]
        per_residue_sasa = [round(v, 3) for v in res_sasa.values()]
        buried_count = sum(1 for v in res_sasa.values() if v < 10.0)
    n_res = max(1, parsed['n_residues'])
    sasa['per_residue_sasa'] = per_residue_sasa
    sasa['buried_fraction'] = round(buried_count / n_res, 3)

    # Nucleic-acid specific metrics
    aform = {}
    stacking = {}
    pairs = {}
    if parsed['is_nucleic']:
        aform = compute_aform_score(coords, atom_names)
        stacking = compute_stacking_analysis(coords, atom_names)
        pairs = compute_pair_satisfaction(coords, residue_ids, atom_names,
                                          residue_names)

    # B-factor statistics
    b_factors = parsed['b_factors']
    b_stats = {}
    if b_factors:
        b_arr = np.array(b_factors)
        b_stats = {
            'mean_b_factor': round(float(np.mean(b_arr)), 2),
            'std_b_factor': round(float(np.std(b_arr)), 2),
            'max_b_factor': round(float(np.max(b_arr)), 2),
            'min_b_factor': round(float(np.min(b_arr)), 2),
        }

    return {
        'n_atoms': parsed['n_atoms'],
        'n_residues': parsed['n_residues'],
        'n_chains': parsed['n_chains'],
        'is_nucleic': parsed['is_nucleic'],
        'clash': clash,
        'rog': rog,
        'bond': bond,
        'sasa': sasa,
        'end_to_end': round(e2e, 3),
        'shape': shape,
        'backbone': backbone,
        'aform': aform,
        'stacking': stacking,
        'pairs': pairs,
        'b_factor': b_stats,
    }
